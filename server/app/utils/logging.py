"""Event log + SSE subscribers, scoped per slot.

Every emitted event is a dict with a `kind` and arbitrary extra fields.
Each slot owns a `SlotLog` that fans events out to: (a) an in-memory buffer
for snapshotting late subscribers, (b) the rich console as a pretty
per-field block (prefixed with the slot id), (c) every active SSE queue
for that slot, (d) a persistent JSONL file that doubles as the cache.

Pipeline code calls the module-level `log()` / `emit_*()` helpers, which
route to the `SlotLog` bound to the current asyncio task via a ContextVar.
Each task binds itself at entry, so no pipeline signature changes.

Two utility modules read the persisted log: `utils/cache.py`
(`cache.llm` hits) and `utils/resumable.py` (`<scope>.submit` /
`<scope>.done` for restart-resilient remote-job submission).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging as _stdlib_logging
import queue
import sys
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any, TextIO

from rich.console import Console
from rich.markup import escape

from app.core.types import BoundingBox, Orientation, ProxyShape

# Fixes flushing issue
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, OSError, ValueError):
    pass

_console_suppressed = False


class _NonBlockingStream:
    """Write-through text stream wrapper that never blocks the caller.

    Pipeline logging and uvicorn's access log both write from the asyncio event
    loop. A blocking write to a stalled stdout — a terminal whose window/panel
    was closed while the process kept running, or a Windows console paused by a
    text selection — freezes the loop and hangs every request until the sink
    drains, which may be never. So `write()` parks the text on a bounded queue
    and returns at once; a daemon thread does the real write. If the sink wedges
    the queue fills and writes are dropped (events.jsonl + SSE still hold every
    event). Non-write attributes delegate to the wrapped stream so terminal
    detection (isatty/encoding/fileno) is unchanged."""

    def __init__(self, stream: TextIO, *, maxsize: int = 65_536) -> None:
        self._stream = stream
        self._queue: queue.Queue[str] = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        threading.Thread(
            target=self._drain, name="log-stream-writer", daemon=True
        ).start()

    def _drain(self) -> None:
        while True:
            chunk = self._queue.get()
            if _console_suppressed:
                continue
            try:
                self._stream.write(chunk)
                self._stream.flush()
            except Exception:
                pass

    def write(self, text: str) -> int:
        if not _console_suppressed:
            try:
                self._queue.put_nowait(text)
            except queue.Full:
                self.dropped += 1
        return len(text)

    def flush(self) -> None:
        # The drainer flushes after every chunk; a caller-side flush must never
        # block the event loop, so it is a no-op.
        pass

    def __getattr__(self, name: str) -> Any:
        stream = self.__dict__.get("_stream")
        if stream is None:
            raise AttributeError(name)
        return getattr(stream, name)


_console = Console(file=_NonBlockingStream(sys.stdout))


def console_note(message: str) -> None:
    """Emit a one-off diagnostic line without blocking the caller — for paths
    off the SlotLog stream (key rotation, prefab-match fallbacks) that would
    otherwise `print()` straight to a possibly-stalled stdout on the loop."""
    if _console_suppressed:
        return
    _console.print(message, markup=False, highlight=False)


def install_nonblocking_stdlib_logging() -> None:
    """Defer uvicorn's per-request log writes off the event loop.

    Uvicorn logs each request from the loop thread; on a stalled stdout that
    blocks the loop exactly like the pipeline console did, so a reconnecting
    client's polling would re-freeze the server. Wrap each uvicorn
    StreamHandler's stream so only the real write is deferred — the handler and
    its formatter are untouched. Idempotent; call once at startup after uvicorn
    has configured logging."""
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        for handler in _stdlib_logging.getLogger(name).handlers:
            stream = getattr(handler, "stream", None)
            if isinstance(handler, _stdlib_logging.StreamHandler) and not isinstance(
                stream, _NonBlockingStream
            ):
                with contextlib.suppress(Exception):
                    handler.setStream(_NonBlockingStream(stream))

_CONSOLE_OMIT_FIELDS = frozenset({"system", "user", "output", "reasoning", "content"})
_CONSOLE_COMPACT_KINDS = frozenset({"cache.llm"})
_CONSOLE_STR_MAX = 240


def derive_status(
    events: list[dict[str, Any]], *, awaiting: bool = False, live: bool = False,
    capped: bool = False,
) -> str:
    """The single, live source of truth for a slot's status.

    Status is NOT purely a function of the event log — distinguishing
    "running" from "paused at a step gate" from "crashed/idle" needs runtime
    that the log can't carry, so callers pass it in:

      * `awaiting` — a live gate is parked before its next call (the stepped
        cell's auto-pause). Surfaced as `paused`; the awaited step lives in the
        gate's `pending` / the trailing `branch.step.pending` event.
      * `live` — a pipeline task is currently executing this slot.
      * `capped` — settled spend has hit the cell's spend cap (a derived,
        log-only fact the API layer computes). A hard stop until the cap is
        manually overridden; see `_cap_reached` in the API layer.

    Resolution order (terminal markers win; then the runtime refinement; then
    the started-but-not-live fallback):

      * `done`    — any `run.done` (STICKY: a post-run mesh retry appends events
        after it, but a completed run is terminal — reset is the only way back).
      * `capped`  — over the spend cap. Beats `error` deliberately: the override
        is the ONLY way forward (a plain resume/retry is refused while capped),
        so the UI must surface the override path, not a dead-end "retry".
      * `error`   — the latest event is `run.error` (a resume strips it first).
      * `paused`  — a gate is parked (`awaiting`).
      * `running` — a task is `live` (executing, between gates).
      * `paused`  — started (non-empty log) but no live task: a clean hard
        pause (`run.paused`), or a process that died / a cell rehydrated at
        boot. All resolve to a resumable `paused` without any boot fix-up.
      * `idle`    — empty log: a fresh, never-started cell.
    """
    return derive_status_fields(
        has_done=any(e.get("kind") == "run.done" for e in events),
        last_is_error=bool(events) and events[-1].get("kind") == "run.error",
        has_events=bool(events),
        awaiting=awaiting,
        live=live,
        capped=capped,
    )


def derive_status_fields(
    *,
    has_done: bool,
    last_is_error: bool,
    has_events: bool,
    awaiting: bool = False,
    live: bool = False,
    capped: bool = False,
) -> str:
    """`derive_status` reduced to the three log-derived facts it actually needs —
    whether the cell ever finished, whether its last event was an error, and
    whether it has any events — plus the runtime overlay. This lets a cached
    board summary resolve status without the events in hand; see `derive_status`
    for the resolution order these follow."""
    if has_done:
        return "done"
    if capped:
        return "capped"
    if last_is_error:
        return "error"
    if awaiting:
        return "paused"
    if live:
        return "running"
    return "paused" if has_events else "idle"


class _LazyState(dict):
    """A SlotLog's `state`, with `events` parsed from disk only on first access.

    Activating a run builds a SlotLog per cell; eagerly parsing every cell's
    `events.jsonl` (the biggest are gigabytes) is what made run-switching slow.
    So the events list loads lazily: `state["events"]` reads the log the first
    time it's touched, and the board reads its cached summary (`boardcache`)
    INSTEAD of the events — so only a cell actually operated on pays to parse.
    `prompt`/`model` are plain eager keys (set from a cheap one-line meta read).
    Every other mapping op is unchanged, so the many `slot_log.state["events"]`
    call sites keep working transparently."""

    def __init__(self, load_events: Callable[[], list[dict[str, Any]]]) -> None:
        super().__init__(prompt=None, model=None)
        self._load_events = load_events
        self.events_loaded = False

    def ensure_events(self) -> None:
        # Mark loaded only AFTER a successful parse. Setting the flag first meant a
        # failed load (e.g. a MemoryError materializing a multi-GB log) left the
        # flag True with no `events` key — poisoning the state so every later
        # access raised a misleading `KeyError: 'events'` forever instead of
        # retrying. Now a failed load leaves the flag False (the exception
        # propagates, surfacing the REAL error, and the next access retries), and
        # the `not-contains` guard heals a state already poisoned that way.
        if self.events_loaded and dict.__contains__(self, "events"):
            return
        events = self._load_events()
        dict.__setitem__(self, "events", events)
        self.events_loaded = True

    def __getitem__(self, key: str) -> Any:
        if key == "events":
            self.ensure_events()
        return dict.__getitem__(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "events":
            self.events_loaded = True  # an explicit assignment replaces the load
        dict.__setitem__(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "events":
            self.ensure_events()
        return dict.get(self, key, default)

    def __contains__(self, key: object) -> bool:
        return key == "events" or dict.__contains__(self, key)


# cache.llm fields kept OUT of the in-memory event buffer. On a large scene these
# (the rendered prompt context, the full variable set, the model's reasoning) are
# ~98% of the log's bytes, and nothing in the running pipeline needs them resident:
# the LLM cache matches on `key`, provenance + object-wipe read `output`, and
# cross-model replay matches a stored `replay_key` (see cache.py). They stay inline
# on disk (events.jsonl is untouched) and are read back by byte offset on demand
# (`SlotLog.read_event_full`) when a client expands a call or the prompt lab
# re-runs a step. `output` is deliberately KEPT — it's tiny and hot (cache hits +
# the `emitted` id fold in `_slim_event`).
_HEAVY_OFFLOAD = ("system", "user", "reasoning", "variables")


def _slim_for_memory(event: dict[str, Any]) -> dict[str, Any]:
    """The in-memory form of an event: for `cache.llm`, the heavy prompt / variables
    / reasoning bytes are dropped (they live on disk, read on demand); every other
    kind passes through unchanged. A light `has_variables` bool is kept so the
    prompt lab can still tell which calls are re-renderable (they logged their
    variables) without holding the variables themselves in RAM."""
    if event.get("kind") != "cache.llm":
        return event
    slim = {k: v for k, v in event.items() if k not in _HEAVY_OFFLOAD}
    if isinstance(event.get("variables"), dict):
        slim["has_variables"] = True
    return slim


class SlotLog:
    """Owns state + disk + subscribers for one slot."""

    def __init__(self, slot_id: str, events_path: Path) -> None:
        self.slot_id = slot_id
        self.events_path = events_path
        # Status is NOT stored here — it's derived live via `derive_status`
        # (which needs the gate/task runtime the SlotLog can't see). `state`
        # holds only the durable, log-derived data. `events` loads lazily from
        # disk on first access (see `_LazyState`), so building a SlotLog is cheap.
        self.state: dict[str, Any] = _LazyState(self._load_events_from_disk)
        self.subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        # Append handle kept open for the life of the cell. Opening events.jsonl
        # on every log() event (hundreds per run × many parallel cells) was
        # blowing through macOS's default 256-fd soft limit.
        self._events_file: TextIO | None = None
        # Byte offset of each event's line in events.jsonl, parallel to
        # `state["events"]`, so the heavy bytes dropped from the in-memory (slim)
        # event can be read back from disk by seeking (`read_event_full`). Built by
        # the lazy load and kept in sync by log() / truncate / replace.
        self._offsets: list[int] = []
        self._byte_len: int = 0  # current on-disk size (the next append offset)
        # Last time this cell was touched by an API request — the idle-unload sweep
        # leaves recently-viewed cells loaded (see routes `_idle_unload_loop`).
        self._touched: float = time.monotonic()

    def close(self) -> None:
        """Release the append handle (shutdown / cell reset)."""
        if self._events_file is not None:
            try:
                self._events_file.close()
            except OSError:
                pass
            self._events_file = None

    def _ensure_events_append(self) -> TextIO:
        if self._events_file is None or self._events_file.closed:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            # newline="" disables the platform newline translation (\n -> \r\n on
            # Windows) so a written line's byte length matches our offset tracking —
            # otherwise `read_event_full`'s seek would land mid-line and fall back to
            # a full-file scan on every byte fetch.
            self._events_file = self.events_path.open("a", encoding="utf-8", newline="")
        return self._events_file

    def _close_events_file(self) -> None:
        self.close()

    def hydrate_from_disk(self) -> None:
        """Reset to the on-disk log WITHOUT parsing it. Events load lazily on
        first access to `state["events"]`, so activating a run needn't read every
        cell's (possibly multi-GB) log — the board reads its cached summary
        instead. Prompt + model are the exception: they come from the first
        run.start, read here via a cheap one-line scan so resume/prefill work
        while the events stay unloaded."""
        self._close_events_file()
        self.state = _LazyState(self._load_events_from_disk)
        self._offsets = []
        self._byte_len = 0
        prompt, model = self._read_meta()
        self.state["prompt"] = prompt
        self.state["model"] = model

    def _load_events_from_disk(self) -> list[dict[str, Any]]:
        """Parse the log into SLIM in-memory events (heavy bytes offloaded — see
        `_slim_for_memory`) and record each line's byte offset so those bytes can be
        read back on demand (`read_event_full`). Streams line-by-line, so even a
        multi-GB log only ever holds one raw line plus the slim list in memory. Run
        at most once per cell (on first access to `state["events"]`)."""
        events: list[dict[str, Any]] = []
        offsets: list[int] = []
        self._offsets = offsets
        self._byte_len = 0
        if not self.events_path.exists():
            return events
        with self.events_path.open("rb") as f:
            while True:
                off = f.tell()
                raw = f.readline()
                if not raw:
                    break
                line = raw.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    _console.print(
                        f"[dim]\\[{self.slot_id}][/dim] [red]skipping malformed line in {self.events_path}[/red]"
                    )
                    continue
                offsets.append(off)
                events.append(_slim_for_memory(event))
        self._byte_len = self.events_path.stat().st_size
        return events

    def read_event_full(self, index: int) -> dict[str, Any] | None:
        """The FULL event at `index` (heavy bytes included), read from disk by byte
        offset — the on-demand backing for the fields the in-memory buffer drops.
        Falls back to a linear scan if the offset index is absent or stale (e.g. a
        log rewritten by another path). Returns None when there's no such event."""
        if not self._offsets and isinstance(self.state, _LazyState):
            self.state["events"]  # trigger the lazy load, which builds `_offsets`
        off = self._offsets[index] if 0 <= index < len(self._offsets) else None
        if off is not None:
            try:
                with self.events_path.open("rb") as f:
                    f.seek(off)
                    raw = f.readline()
                event = json.loads(raw)
                if event.get("index") == index:
                    return event
            except (OSError, json.JSONDecodeError):
                pass
        try:
            with self.events_path.open("rb") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if event.get("index") == index:
                        return event
        except OSError:
            pass
        return None

    def _read_meta(self) -> tuple[str | None, str | None]:
        """`(prompt, model)` from the first run.start — one line, so it's O(1)
        regardless of log size. None for a fresh/legacy log without a leading
        run.start (the full load, when it happens, carries the real values)."""
        try:
            with self.events_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        return None, None
                    if e.get("kind") == "run.start":
                        p, m = e.get("prompt"), e.get("model")
                        return (p if isinstance(p, str) else None,
                                m if isinstance(m, str) else None)
                    return None, None
        except OSError:
            return None, None
        return None, None

    @property
    def loaded(self) -> bool:
        """Whether the event list is in memory. The cost-backfill sweep uses this
        to skip cells that would otherwise be force-loaded just to be scanned."""
        return isinstance(self.state, _LazyState) and self.state.events_loaded

    @property
    def is_empty(self) -> bool:
        """Whether the cell has no events — answered WITHOUT loading them (stats
        the file) so the board/launch path needn't parse a huge log to learn a
        cell is non-empty."""
        if isinstance(self.state, _LazyState) and self.state.events_loaded:
            return not self.state["events"]
        try:
            return self.events_path.stat().st_size == 0
        except OSError:
            return True

    def truncate_events_to(self, n: int) -> int:
        """Keep only the first `n` events on disk and in memory. Returns the new
        length. The file is truncated at the byte offset of line `n` — the kept
        lines (with their full heavy bytes) are left byte-for-byte intact, rather
        than rewritten from the slim in-memory buffer (which would drop them)."""
        self._close_events_file()
        n = max(0, min(n, len(self.state["events"])))
        self.state["events"] = self.state["events"][:n]
        cut = self._offsets[n] if n < len(self._offsets) else self._byte_len
        self._offsets = self._offsets[:n]
        self._byte_len = cut
        if cut <= 0:
            self.events_path.write_text("")
        else:
            with self.events_path.open("r+b") as f:
                f.truncate(cut)
        return n

    def replace_events(self, events: list[dict[str, Any]]) -> None:
        """Overwrite the log with `events` (already in final order), rewriting
        every `index` to its new position so `index == line == list position`
        holds again, and resyncing the in-memory buffer + append handle.

        Truncation only ever drops the tail, so it leaves indices intact; a
        SURGICAL edit (deleting/rewriting lines mid-log, e.g. wiping one object)
        breaks the position==index invariant and must reindex, or the next
        `log()` (a later resume) would mint an index that collides with a
        surviving line — which the client's `idx <= maxIndex` dedup then drops.
        """
        self._close_events_file()
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        offsets: list[int] = []
        slim: list[dict[str, Any]] = []
        byte_len = 0
        # Callers pass FULL events (heavy bytes included); the disk log stays full
        # while only the slim projection is kept resident, and offsets are rebuilt.
        # newline="" keeps a line's byte length == our offset math (see
        # _ensure_events_append) so the rebuilt offsets stay seek-accurate.
        with self.events_path.open("w", encoding="utf-8", newline="") as f:
            for i, event in enumerate(events):
                event["index"] = i
                line = json.dumps(event) + "\n"
                offsets.append(byte_len)
                f.write(line)
                byte_len += len(line.encode("utf-8"))
                slim.append(_slim_for_memory(event))
        self._offsets = offsets
        self._byte_len = byte_len
        self.state["events"] = slim

    def start_run(self, prompt: str, model: str) -> None:
        self._close_events_file()
        self.state["prompt"] = prompt
        self.state["model"] = model
        self.state["events"] = []
        self._offsets = []
        self._byte_len = 0
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text("")
        self.log("run.start", prompt=prompt, model=model)

    def finish_run(self) -> None:
        self.log("run.done")

    def log(self, kind: str, **data: Any) -> None:
        # `ts` (epoch seconds) is stamped on every event so the client can show
        # WHEN a call happened and how long the gap to the next event was — e.g.
        # a transport-retry storm's per-attempt spacing. Additive + optional:
        # older logs without it still replay, and cache keys are computed
        # separately (never from the event dict), so it can't shift a cache hit.
        event: dict[str, Any] = {
            "index": len(self.state["events"]),
            "ts": time.time(),
            "kind": kind,
            **data,
        }
        f = self._ensure_events_append()
        line = json.dumps(event) + "\n"
        # Record this line's byte offset BEFORE writing so the heavy bytes we drop
        # from the in-memory copy can be seeked back later; keep only the slim event
        # resident. The disk line stays FULL (the cache/observability source).
        self._offsets.append(self._byte_len)
        f.write(line)
        f.flush()
        self._byte_len += len(line.encode("utf-8"))
        slim = _slim_for_memory(event)
        self.state["events"].append(slim)
        _print(self.slot_id, event)
        for q in self.subscribers:
            q.put_nowait(slim)

    def append_unloaded(self, kind: str, index: int, **data: Any) -> dict[str, Any]:
        """Append ONE event to disk (and broadcast) at `index`, WITHOUT loading a
        lazily-hydrated log into memory — for an external writer (e.g. a spend-cap
        override) acting on a cell whose log is too large to materialize. `index`
        is the on-disk event count the caller already computed from a bounded
        stream (`cellsummary.iter_events`), so it continues the log's position
        sequence. `log()` stays the path for a loaded cell (it keeps
        `state['events']` in sync); this deliberately leaves the events unloaded,
        so the next load picks the appended event up from disk."""
        event: dict[str, Any] = {"index": index, "ts": time.time(), "kind": kind, **data}
        f = self._ensure_events_append()
        f.write(json.dumps(event) + "\n")
        f.flush()
        _print(self.slot_id, event)
        for q in self.subscribers:
            q.put_nowait(event)
        return event

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        if q in self.subscribers:
            self.subscribers.remove(q)


_current: ContextVar[SlotLog] = ContextVar("current_slot_log")


def bind(slot_log: SlotLog) -> None:
    """Bind the current asyncio task to a slot log. Call at the top of
    every `_run(slot)` task — subsequent `await`s inherit the binding."""
    _current.set(slot_log)


def current_events() -> list[dict[str, Any]]:
    """Snapshot of the bound slot's event list. Used by cache lookups
    (cache.find_llm_cache_hit, resumable.find_done, etc.)."""
    return _current.get().state["events"]


def find_event(kind: str, **fields: Any) -> dict[str, Any] | None:
    """Most recent event whose kind and selected fields match exactly."""
    for event in reversed(current_events()):
        if event.get("kind") != kind:
            continue
        if all(event.get(k) == v for k, v in fields.items()):
            return event
    return None


def log_once(kind: str, *, match_fields: tuple[str, ...], **data: Any) -> None:
    """Append `kind` only if no prior event has the same semantic key."""
    match = {k: data.get(k) for k in match_fields}
    if find_event(kind, **match) is not None:
        return
    log(kind, **data)


def current_slot_id() -> str | None:
    """Slot id of the currently-bound task, or None if no binding (e.g.
    called from a script or before bind())."""
    log = _current.get(None)
    return log.slot_id if log is not None else None


def slot_dir() -> Path:
    """Directory for the currently-bound slot (parent of events.jsonl)."""
    return _current.get().events_path.parent


def log(kind: str, **data: Any) -> None:
    _current.get().log(kind, **data)


def emit_bbox(
    node_id: str,
    bbox: BoundingBox,
    *,
    parent_id: str | None,
    prompt: str,
    kind: str,
    proxy_shape: ProxyShape | None = None,
    orientation: Orientation = 0,
    parent_region: str | None = None,
) -> None:
    if find_event("bbox", id=node_id) is not None:
        return
    log(
        "bbox",
        id=node_id,
        origin=list(bbox.origin),
        dimensions=list(bbox.dimensions),
        parent_id=parent_id,
        prompt=prompt,
        node_kind=kind,
        proxy_shape=proxy_shape.value if proxy_shape is not None else None,
        orientation=orientation,
        parent_region=parent_region,
    )


def emit_model(node_id: str, artifact_kind: str, url: str) -> None:
    if find_event("model", id=node_id, artifact_kind=artifact_kind) is not None:
        return
    log("model", id=node_id, artifact_kind=artifact_kind, url=url)


def emit_step(node_id: str, phase: str, **extra: Any) -> None:
    """Current-location marker: emitted at the start of each pipeline phase
    for a given node. The client uses these to light up the active node in
    the tree view."""
    if find_event("step", node=node_id, phase=phase) is not None:
        return
    log("step", node=node_id, phase=phase, **extra)


# --- console formatting -----------------------------------------------------

_KIND_COLOR = {
    "run.start": "cyan",
    "run.done": "green",
    "run.error": "red",
    "bbox": "yellow",
    "model": "magenta",
}


def suppress_console() -> None:
    """Stop terminal output during process teardown (jsonl/SSE unaffected)."""
    global _console_suppressed
    _console_suppressed = True


def _console_fields(event: dict[str, Any]) -> list[tuple[str, Any]]:
    kind = str(event.get("kind", "?"))
    fields = [(k, v) for k, v in event.items() if k not in ("kind", "ts")]
    if kind in _CONSOLE_COMPACT_KINDS:
        fields = [(k, v) for k, v in fields if k not in _CONSOLE_OMIT_FIELDS]
    return fields


def _print(slot_id: str, event: dict[str, Any]) -> None:
    if _console_suppressed:
        return
    kind = str(event.get("kind", "?"))
    color = _KIND_COLOR.get(kind, "blue")
    fields = _console_fields(event)
    _console.print(
        f"[dim]\\[{slot_id}][/dim] [bold {color}]{kind}[/bold {color}]",
    )
    if not fields:
        return
    width = max(len(k) for k, _ in fields)
    for k, v in fields:
        _console.print(f"  [dim]{k.ljust(width)}[/dim]  {escape(_fmt(v))}")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in value) + "]"
    if isinstance(value, dict):
        parts = [f"{k}={_fmt(v)}" for k, v in value.items()]
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, str):
        if len(value) > _CONSOLE_STR_MAX:
            return f"{value[:_CONSOLE_STR_MAX]}… ({len(value)} chars)"
        return value
    return repr(value)
