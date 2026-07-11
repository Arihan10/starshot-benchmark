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
import gzip
import json
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any, TextIO

from rich.console import Console
from rich.markup import escape

from app.core.types import BoundingBox, Orientation, ProxyShape
from app.utils import cache as _cache

# `cache.llm` events embed the FULL rendered prompt (`system`/`user`, whose
# scene context alone can be ~1MB) plus the model `output`/`reasoning` and the
# raw `variables` (another SCENE_CONTEXT copy). A completed cell's log reaches
# hundreds of MB, so keeping every event's heavy fields resident is what turned
# a few dozen hydrated cells into tens of GB of RAM. We keep only a SLIM
# projection in memory and read the heavy fields back from disk on demand
# (`SlotLog.full_event`) — the event log on disk is unchanged and remains the
# source of truth for the LLM cache, teacher-forcing export, and prompt lab.
_HEAVY_FIELDS = frozenset({"system", "user", "output", "reasoning", "content", "variables"})


def _slim_event(event: dict[str, Any]) -> dict[str, Any]:
    """In-memory projection of a logged event with the big prompt/output/
    variables payloads dropped. Only `cache.llm` events carry them; everything
    else (bbox / model / step / cost / submit …) is already small and kept
    verbatim. The few in-memory read paths that would otherwise need a heavy
    field get a precomputed marker so they never touch disk:

      * `has_scene`     — does the prompt contain scene entities (tf-steps
        timeline, attention gating).
      * `has_variables` — is the call re-renderable (prompt lab eligibility,
        branch fork-point scan).
      * `_rk`           — model-independent content hash for `find_llm_replay`
        (the committed-prefix replay match, which used to compare the raw
        `system`/`user`).
    """
    if event.get("kind") != "cache.llm":
        return event
    slim = {k: v for k, v in event.items() if k not in _HEAVY_FIELDS}
    slim["has_variables"] = isinstance(event.get("variables"), dict)
    system = event.get("system")
    user = event.get("user")
    schema = event.get("schema")
    if isinstance(system, str) and isinstance(user, str) and isinstance(schema, str):
        slim["_rk"] = _cache.replay_key(system, user, schema)
    # `has_scene_context` lives in teacher_forcing (which imports llm → logging);
    # a deferred import avoids the load-time cycle and is cheap at call time.
    try:
        from app.services.teacher_forcing import has_scene_context
        slim["has_scene"] = has_scene_context(event)
    except Exception:  # noqa: BLE001 — a marker miss just disables an optional UI hint
        slim["has_scene"] = False
    return slim

# Fixes flushing issue
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, OSError, ValueError):
    pass

_console = Console()
_console_suppressed = False

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
    if any(e.get("kind") == "run.done" for e in events):
        return "done"
    if capped:
        return "capped"
    if events and events[-1].get("kind") == "run.error":
        return "error"
    if awaiting:
        return "paused"
    if live:
        return "running"
    return "paused" if events else "idle"


class SlotLog:
    """Owns state + disk + subscribers for one slot."""

    def __init__(self, slot_id: str, events_path: Path) -> None:
        self.slot_id = slot_id
        self.events_path = events_path
        # A TERMINAL log (a finished ablation variant) may be stored gzipped at
        # `events.jsonl.gz` (~10× smaller — the log is mostly `cache.llm` text) with
        # no plain twin. READS transparently fall back to it; any WRITE first
        # materializes the plain file (`_ensure_plain`), so append/rewrite logic is
        # unchanged. `_gzipped` tracks which form the last hydrate read.
        self._gz_path = events_path.with_name(events_path.name + ".gz")
        self._gzipped = False
        # Status is NOT stored here — it's derived live via `derive_status`
        # (which needs the gate/task runtime the SlotLog can't see). `state`
        # holds only the durable, log-derived data.
        self.state: dict[str, Any] = {
            "prompt": None,
            "model": None,
            "events": [],
        }
        self.subscribers: list[asyncio.Queue[dict[str, Any]]] = []
        # Append handle kept open for the life of the cell. Opening events.jsonl
        # on every log() event (hundreds per run × many parallel cells) was
        # blowing through macOS's default 256-fd soft limit.
        self._events_file: TextIO | None = None
        # Byte offset of each event's line in events.jsonl, aligned 1:1 with
        # state["events"] positions, plus the file's total byte length (the next
        # append offset). Lets `full_event` seek straight to one event's line and
        # read back its heavy fields without loading the whole (huge) log.
        self._offsets: list[int] = []
        self._end_offset: int = 0

    def close(self) -> None:
        """Release the append handle (shutdown / cell reset)."""
        if self._events_file is not None:
            try:
                self._events_file.close()
            except OSError:
                pass
            self._events_file = None

    def _ensure_plain(self) -> None:
        """Materialize a plain `events.jsonl` from the gzipped terminal sidecar if
        that's all that exists — every WRITE path (append / truncate / replace /
        start) acts on the uncompressed log. A no-op unless a gz-only log is being
        (re)written (a rare variant resume/edit). The decompressed byte layout is
        identical, so byte offsets recorded during a gz hydrate stay valid."""
        if self.events_path.exists() or not self._gz_path.exists():
            self._gzipped = False
            return
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(self._gz_path, "rb") as src, self.events_path.open("wb") as dst:
            for chunk in iter(lambda: src.read(1 << 20), b""):
                dst.write(chunk)
        self._gz_path.unlink()
        self._gzipped = False

    def _ensure_events_append(self) -> TextIO:
        if self._events_file is None or self._events_file.closed:
            self._ensure_plain()
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            self._events_file = self.events_path.open("a", encoding="utf-8")
        return self._events_file

    def _close_events_file(self) -> None:
        self.close()

    def hydrate_from_disk(self) -> None:
        """Load state from an existing events.jsonl. Prompt + model come
        from the first run.start event, so resume works without a side
        file. Only the SLIM projection is retained in memory (heavy fields
        read back on demand via `full_event`); byte offsets are recorded so
        that read is a single seek."""
        self.state["events"] = []
        self.state["prompt"] = None
        self.state["model"] = None
        self._offsets = []
        self._end_offset = 0
        # Prefer the plain log; fall back to the gzipped terminal sidecar. For gz,
        # `len(raw)` is the DECOMPRESSED byte width — identical to the plain layout,
        # so offsets stay valid if the log is later materialized (`_ensure_plain`);
        # `full_event` re-streams instead of seeking while still gzipped.
        read_path, self._gzipped = self.events_path, False
        if not self.events_path.exists():
            if self._gz_path.exists():
                read_path, self._gzipped = self._gz_path, True
            else:
                return
        offset = 0
        # Binary so len(raw) is the exact on-disk byte width (offsets must index
        # bytes, not decoded chars) and a stray non-UTF8 byte can't abort load.
        _open = (lambda: gzip.open(read_path, "rb")) if self._gzipped else (lambda: read_path.open("rb"))
        with _open() as f:
            for raw in f:
                line = raw.decode("utf-8", "replace").strip()
                if line:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        _console.print(
                            f"[dim]\\[{self.slot_id}][/dim] [red]skipping malformed line in {self.events_path}[/red]"
                        )
                    else:
                        self.state["events"].append(_slim_event(event))
                        self._offsets.append(offset)
                        if event.get("kind") == "run.start" and self.state["prompt"] is None:
                            self.state["prompt"] = event.get("prompt")
                            self.state["model"] = event.get("model")
                offset += len(raw)
        self._end_offset = offset

    def full_event(self, index: int) -> dict[str, Any] | None:
        """The COMPLETE logged event (heavy fields included) for the event whose
        `index` is `index`, read straight off disk via its recorded byte offset.
        Used by the LLM cache hit/replay, the teacher-forcing export, and the
        prompt-lab detail endpoints — the paths that genuinely need the raw
        `system`/`user`/`output`/`variables` the slim in-memory buffer drops."""
        pos: int | None = None
        events = self.state["events"]
        if 0 <= index < len(events) and events[index].get("index") == index:
            pos = index  # fast path: the index==position invariant holds
        else:
            for i, e in enumerate(events):
                if e.get("index") == index:
                    pos = i
                    break
        if pos is None or pos >= len(self._offsets):
            return None
        try:
            if self._gzipped:
                # gz can't seek by byte offset — re-stream to the pos-th non-blank
                # line (mirrors hydrate's line accounting). Streaming, so memory is
                # one line at a time, never the whole (decompressed) log.
                count = 0
                with gzip.open(self._gz_path, "rb") as f:
                    for raw in f:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line:
                            continue
                        if count == pos:
                            return json.loads(line)
                        count += 1
                return None
            with self.events_path.open("rb") as f:
                f.seek(self._offsets[pos])
                raw = f.readline()
            return json.loads(raw.decode("utf-8", "replace"))
        except (OSError, ValueError):
            return None

    def full_events(self) -> list[dict[str, Any]]:
        """Every event with its heavy fields, read from disk in order. Only for
        the rare surgical rewrite (object wipe) that must re-emit the full log;
        normal reads use the slim buffer + `full_event`."""
        out: list[dict[str, Any]] = []
        read_path, is_gz = self.events_path, False
        if not self.events_path.exists():
            if self._gz_path.exists():
                read_path, is_gz = self._gz_path, True
            else:
                return out
        _open = (lambda: gzip.open(read_path, "rb")) if is_gz else (lambda: read_path.open("rb"))
        with _open() as f:
            for raw in f:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def truncate_events_to(self, n: int) -> int:
        """Keep only the first `n` events on disk and in memory. Returns
        the new length. Truncates the file at the byte offset of line `n`, so
        the retained prefix's bytes (heavy fields included) are untouched — the
        in-memory buffer being slim never leaks into what's persisted."""
        self._close_events_file()
        self._ensure_plain()  # rewrite acts on the plain log
        n = max(0, min(n, len(self.state["events"])))
        if n == 0:
            self.events_path.write_text("")
            keep_bytes = 0
        else:
            keep_bytes = self._offsets[n] if n < len(self._offsets) else self._end_offset
            with self.events_path.open("rb+") as f:
                f.truncate(keep_bytes)
        self.state["events"] = self.state["events"][:n]
        self._offsets = self._offsets[:n]
        self._end_offset = keep_bytes
        return n

    def replace_events(self, events: list[dict[str, Any]]) -> None:
        """Overwrite the log with `events` (FULL events, already in final order),
        rewriting every `index` to its new position so `index == line == list
        position` holds again, and resyncing the slim buffer + byte offsets.

        Callers MUST pass full events (see `full_events`), never the slim buffer —
        the rewrite is byte-for-byte what lands on disk, so a slim event here
        would permanently drop that step's prompt/output from the cache.

        Truncation only ever drops the tail, so it leaves indices intact; a
        SURGICAL edit (deleting/rewriting lines mid-log, e.g. wiping one object)
        breaks the position==index invariant and must reindex, or the next
        `log()` (a later resume) would mint an index that collides with a
        surviving line — which the client's `idx <= maxIndex` dedup then drops.
        """
        self._close_events_file()
        for i, event in enumerate(events):
            event["index"] = i
        # A full rewrite supersedes any gzipped twin — drop it so reads don't find
        # a stale compressed copy alongside the freshly-written plain log.
        self._gz_path.unlink(missing_ok=True)
        self._gzipped = False
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._offsets = []
        offset = 0
        with self.events_path.open("wb") as f:
            for event in events:
                raw = (json.dumps(event) + "\n").encode("utf-8")
                self._offsets.append(offset)
                offset += len(raw)
                f.write(raw)
        self._end_offset = offset
        self.state["events"] = [_slim_event(e) for e in events]

    def start_run(self, prompt: str, model: str) -> None:
        self._close_events_file()
        self.state["prompt"] = prompt
        self.state["model"] = model
        self.state["events"] = []
        self._offsets = []
        self._end_offset = 0
        # A fresh run supersedes any gzipped twin (e.g. re-running a cell that was
        # archived) — drop it so the new plain log is the only source.
        self._gz_path.unlink(missing_ok=True)
        self._gzipped = False
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text("")
        self.log("run.start", prompt=prompt, model=model)

    def finish_run(self) -> None:
        self.log("run.done")

    def log(self, kind: str, **data: Any) -> None:
        event: dict[str, Any] = {
            "index": len(self.state["events"]),
            "kind": kind,
            **data,
        }
        raw = (json.dumps(event) + "\n").encode("utf-8")
        f = self._ensure_events_append()
        f.write(raw.decode("utf-8"))
        f.flush()
        self._offsets.append(self._end_offset)
        self._end_offset += len(raw)
        # Buffer keeps the SLIM projection (heavy fields read back via
        # full_event); live subscribers still get the FULL event so the
        # observability stream is unchanged.
        self.state["events"].append(_slim_event(event))
        _print(self.slot_id, event)
        for q in self.subscribers:
            q.put_nowait(event)

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


def current_slot() -> SlotLog:
    """The bound slot log itself. Cache lookups need it (not just the event
    list) so they can read a matched event's heavy fields back from disk via
    `full_event` — the slim in-memory buffer no longer carries `output`."""
    return _current.get()


def current_events() -> list[dict[str, Any]]:
    """Snapshot of the bound slot's (SLIM) event list. Used by resumable
    submit/done lookups, which only read small bookkeeping fields."""
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
    fields = [(k, v) for k, v in event.items() if k != "kind"]
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
        _flush_stdout()
        return
    width = max(len(k) for k, _ in fields)
    for k, v in fields:
        _console.print(f"  [dim]{k.ljust(width)}[/dim]  {escape(_fmt(v))}")
    _flush_stdout()


def _flush_stdout() -> None:
    try:
        sys.stdout.flush()
    except OSError:
        pass


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
