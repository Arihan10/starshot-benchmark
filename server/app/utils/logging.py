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
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape

from app.core.types import BoundingBox, ProxyShape

_console = Console()


class SlotLog:
    """Owns state + disk + subscribers for one slot."""

    def __init__(self, slot_id: str, events_path: Path) -> None:
        self.slot_id = slot_id
        self.events_path = events_path
        self.state: dict[str, Any] = {
            "status": "idle",
            "prompt": None,
            "model": None,
            "events": [],
        }
        self.subscribers: list[asyncio.Queue[dict[str, Any]]] = []

    def hydrate_from_disk(self) -> None:
        """Load state from an existing events.jsonl. Prompt + model come
        from the first run.start event, so resume works without a side
        file."""
        self.state["events"] = []
        self.state["prompt"] = None
        self.state["model"] = None
        if not self.events_path.exists():
            self.state["status"] = "idle"
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
        last_kind = self.state["events"][-1]["kind"] if self.state["events"] else None
        if last_kind == "run.done":
            self.state["status"] = "done"
        elif last_kind == "run.error":
            self.state["status"] = "error"
        elif last_kind == "run.paused":
            self.state["status"] = "paused"
        elif self.state["events"]:
            self.state["status"] = "running"
        else:
            self.state["status"] = "idle"

    def truncate_events_to(self, n: int) -> int:
        """Keep only the first `n` events on disk and in memory. Returns
        the new length."""
        n = max(0, min(n, len(self.state["events"])))
        self.state["events"] = self.state["events"][:n]
        if n == 0:
            self.events_path.write_text("")
        else:
            with self.events_path.open("w") as f:
                for event in self.state["events"]:
                    f.write(json.dumps(event) + "\n")
        # Status may have changed (e.g. error cleared, or now mid-run).
        last_kind = self.state["events"][-1]["kind"] if self.state["events"] else None
        if last_kind == "run.done":
            self.state["status"] = "done"
        elif last_kind == "run.error":
            self.state["status"] = "error"
        elif last_kind == "run.paused":
            self.state["status"] = "paused"
        elif self.state["events"]:
            self.state["status"] = "running"
        else:
            self.state["status"] = "idle"
        return n

    def start_run(self, prompt: str, model: str) -> None:
        self.state["status"] = "running"
        self.state["prompt"] = prompt
        self.state["model"] = model
        self.state["events"] = []
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text("")
        self.log("run.start", prompt=prompt, model=model)

    def finish_run(self) -> None:
        self.state["status"] = "done"
        self.log("run.done")

    def log(self, kind: str, **data: Any) -> None:
        event: dict[str, Any] = {
            "index": len(self.state["events"]),
            "kind": kind,
            **data,
        }
        self.state["events"].append(event)
        if kind == "run.error":
            self.state["status"] = "error"
        with self.events_path.open("a") as f:
            f.write(json.dumps(event) + "\n")
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
) -> None:
    log(
        "bbox",
        id=node_id,
        origin=list(bbox.origin),
        dimensions=list(bbox.dimensions),
        parent_id=parent_id,
        prompt=prompt,
        node_kind=kind,
        proxy_shape=proxy_shape.value if proxy_shape is not None else None,
    )


def emit_model(node_id: str, artifact_kind: str, url: str) -> None:
    log("model", id=node_id, artifact_kind=artifact_kind, url=url)


def emit_step(node_id: str, phase: str, **extra: Any) -> None:
    """Current-location marker: emitted at the start of each pipeline phase
    for a given node. The client uses these to light up the active node in
    the tree view."""
    log("step", node=node_id, phase=phase, **extra)


# --- console formatting -----------------------------------------------------

_KIND_COLOR = {
    "run.start": "cyan",
    "run.done": "green",
    "run.error": "red",
    "bbox": "yellow",
    "model": "magenta",
}


def _print(slot_id: str, event: dict[str, Any]) -> None:
    kind = str(event.get("kind", "?"))
    color = _KIND_COLOR.get(kind, "blue")
    fields = [(k, v) for k, v in event.items() if k != "kind"]
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
        return value
    return repr(value)
