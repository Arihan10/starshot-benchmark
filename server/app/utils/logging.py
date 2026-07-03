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
import json
import sys
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
            self._events_file = self.events_path.open("a", encoding="utf-8")
        return self._events_file

    def _close_events_file(self) -> None:
        self.close()

    def hydrate_from_disk(self) -> None:
        """Load state from an existing events.jsonl. Prompt + model come
        from the first run.start event, so resume works without a side
        file."""
        self.state["events"] = []
        self.state["prompt"] = None
        self.state["model"] = None
        if not self.events_path.exists():
            return
        with self.events_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    _console.print(
                        f"[dim]\\[{self.slot_id}][/dim] [red]skipping malformed line in {self.events_path}[/red]"
                    )
                    continue
                self.state["events"].append(event)
                if event.get("kind") == "run.start" and self.state["prompt"] is None:
                    self.state["prompt"] = event.get("prompt")
                    self.state["model"] = event.get("model")

    def truncate_events_to(self, n: int) -> int:
        """Keep only the first `n` events on disk and in memory. Returns
        the new length."""
        self._close_events_file()
        n = max(0, min(n, len(self.state["events"])))
        self.state["events"] = self.state["events"][:n]
        if n == 0:
            self.events_path.write_text("")
        else:
            with self.events_path.open("w", encoding="utf-8") as f:
                for event in self.state["events"]:
                    f.write(json.dumps(event) + "\n")
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
        for i, event in enumerate(events):
            event["index"] = i
        self.state["events"] = events
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("w", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")

    def start_run(self, prompt: str, model: str) -> None:
        self._close_events_file()
        self.state["prompt"] = prompt
        self.state["model"] = model
        self.state["events"] = []
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
        self.state["events"].append(event)
        f = self._ensure_events_append()
        f.write(json.dumps(event) + "\n")
        f.flush()
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
