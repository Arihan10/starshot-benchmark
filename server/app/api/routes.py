"""HTTP API — slot-scoped endpoints.

All seven benchmark slots (see app.core.slots) are initialized on lifespan
startup: fresh slots are seeded with a `run.start` and auto-launched,
completed slots stay idle, and interrupted or errored slots are left
paused for manual resume via POST /slots/{id}/resume. Every asyncio task
is bound to its SlotLog via a ContextVar, so concurrent pipeline work
routes events to the right slot without threading a handle through every
call site.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.core.prompts import wrap_image_prompt
from app.core.slots import DEFAULT_MODEL, SLOTS, SLOTS_BY_ID, Slot
from app.core.types import BoundingBox, Node, Orientation, ProxyShape
from app.pipeline import divider, generation
from app.services import llm, threed
from app.utils import logging as rlog
from app.utils.logging import SlotLog

# Where this process writes per-slot run artifacts. Set STARSHOT_RUNS_DIR to
# point at a different directory so multiple simultaneous processes don't
# trample each other.
RUNS_DIR = Path(os.environ.get("STARSHOT_RUNS_DIR", "./runs"))
LEGACY_CURRENT_DIR = RUNS_DIR / "current"

_slot_logs: dict[str, SlotLog] = {}
_tasks: dict[str, asyncio.Task[None]] = {}


class RewindRequest(BaseModel):
    to_event_index: int


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        llm.set_model(DEFAULT_MODEL)
        for slot in SLOTS:
            slot_dir = RUNS_DIR / slot.id
            slot_dir.mkdir(parents=True, exist_ok=True)
            slot_log = SlotLog(slot.id, slot_dir / "events.jsonl")
            slot_log.hydrate_from_disk()
            _slot_logs[slot.id] = slot_log
            _maybe_launch(slot, slot_log)
        try:
            yield
        finally:
            for task in _tasks.values():
                task.cancel()
            for task in list(_tasks.values()):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            await threed.disconnect_http()

    app = FastAPI(
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/artifacts", StaticFiles(directory=RUNS_DIR), name="artifacts")

    @app.get("/slots")
    async def list_slots() -> list[dict[str, object]]:  # pyright: ignore[reportUnusedFunction]
        return [_slot_summary(s) for s in SLOTS]

    @app.get("/trellis/queue")
    async def trellis_queue() -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Live snapshot of the process-global Trellis in-flight queue.
        The client polls this instead of inferring queue state from the
        replayed event log (historical submits without matching .done
        events leak as stale "processing" rows otherwise)."""
        return {
            "cap": threed.GENERATE_CONCURRENCY,
            "entries": threed.queue_snapshot(),
        }

    @app.get("/slots/{slot_id}/events")
    async def slot_events(slot_id: str) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        slot_log = _require_slot_log(slot_id)
        # Subscribe and snapshot synchronously — no await between them, so no
        # log() call can land in both the snapshot and the live queue.
        q = slot_log.subscribe()
        snapshot = list(slot_log.state["events"])
        return StreamingResponse(
            _sse(slot_log, q, snapshot),
            media_type="text/event-stream",
        )

    @app.post("/slots/{slot_id}/rewind")
    async def slot_rewind(slot_id: str, req: RewindRequest) -> dict[str, int | str]:  # pyright: ignore[reportUnusedFunction]
        slot = _require_slot(slot_id)
        slot_log = _slot_logs[slot_id]
        if slot_log.state.get("prompt") is None or slot_log.state.get("model") is None:
            raise HTTPException(
                status_code=400,
                detail="slot has no run to rewind",
            )
        await _cancel_task(slot_id)
        new_len = slot_log.truncate_events_to(req.to_event_index)
        _tasks[slot.id] = asyncio.create_task(_run(slot.id))
        return {"slot_id": slot.id, "events": new_len}

    @app.post("/slots/{slot_id}/resume")
    async def slot_resume(slot_id: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        slot = _require_slot(slot_id)
        slot_log = _slot_logs[slot_id]
        status = slot_log.state.get("status")
        if status not in ("idle", "paused", "error"):
            raise HTTPException(
                status_code=400,
                detail=f"slot is {status}, not startable",
            )
        await _cancel_task(slot_id)
        events = slot_log.state["events"]
        if status == "idle" and not events:
            # Fresh slot — emit run.start now so the run has somewhere to
            # anchor its events. From here on it's identical to a resume.
            slot_log.start_run(slot.prompt, DEFAULT_MODEL)
        else:
            # Drop the terminal sentinel so cache lookups don't trip over it
            # and so the resumed run lands clean events after the prior tail.
            if events and events[-1].get("kind") in ("run.error", "run.paused"):
                slot_log.truncate_events_to(len(events) - 1)
            slot_log.state["model"] = DEFAULT_MODEL
            slot_log.state["status"] = "running"
        _tasks[slot.id] = asyncio.create_task(_run(slot.id))
        return {"slot_id": slot.id}

    @app.post("/slots/{slot_id}/pause")
    async def slot_pause(slot_id: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        slot = _require_slot(slot_id)
        slot_log = _slot_logs[slot_id]
        status = slot_log.state.get("status")
        if status != "running":
            raise HTTPException(
                status_code=400,
                detail=f"slot is {status}, not pausable",
            )
        await _cancel_task(slot_id)
        # _cancel_task awaits the cancellation, so the pipeline task has
        # already torn down (including generation.cancel_pending via _run's
        # CancelledError branch) by the time we emit the sentinel.
        slot_log.state["status"] = "paused"
        slot_log.log("run.paused")
        return {"slot_id": slot.id}

    @app.post("/slots/{slot_id}/retry-mesh/{node_id}")
    async def slot_retry_mesh(slot_id: str, node_id: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        slot = _require_slot(slot_id)
        slot_log = _slot_logs[slot_id]
        node = _reconstruct_node(slot_log, node_id)
        if node is None:
            raise HTTPException(
                status_code=404,
                detail=f"no bbox event found for node: {node_id}",
            )
        async def _do_retry() -> None:
            rlog.bind(slot_log)
            await generation.retry_node(
                node=node, runs_dir=RUNS_DIR, run_id=slot.id,
            )
        asyncio.create_task(_do_retry())
        return {"slot_id": slot.id, "node_id": node_id}

    @app.post("/slots/{slot_id}/reset")
    async def slot_reset(slot_id: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        slot = _require_slot(slot_id)
        await _cancel_task(slot_id)
        # Cancel any standalone retries (registered on generation._pending
        # but with no owning _run task to drive cleanup) that the running
        # task wouldn't have touched.
        generation.cancel_pending(slot.id)
        slot_dir = RUNS_DIR / slot.id
        shutil.rmtree(slot_dir, ignore_errors=True)
        slot_dir.mkdir(parents=True, exist_ok=True)
        slot_log = SlotLog(slot.id, slot_dir / "events.jsonl")
        _slot_logs[slot.id] = slot_log
        slot_log.start_run(slot.prompt, DEFAULT_MODEL)
        _tasks[slot.id] = asyncio.create_task(_run(slot.id))
        return {"slot_id": slot.id}

    return app


def _slot_summary(slot: Slot) -> dict[str, object]:
    slot_log = _slot_logs.get(slot.id)
    state = (
        slot_log.state
        if slot_log is not None
        else {
            "status": "idle",
            "prompt": slot.prompt,
            "events": [],
        }
    )
    events = state.get("events", [])
    return {
        "id": slot.id,
        "prompt": state.get("prompt") or slot.prompt,
        "status": state.get("status", "idle"),
        "events_count": len(events),
        "last_kind": events[-1]["kind"] if events else None,
    }


def _maybe_launch(slot: Slot, slot_log: SlotLog) -> None:
    """Prepare each slot for manual start. Nothing auto-runs at boot — the
    user clicks start/resume/retry per slot from the viewer. Fresh slots
    sit idle with their seed prompt prefilled; previously-running slots
    come back as paused (resumable); errored slots come back as error
    (retry-able); completed slots stay done."""
    events = slot_log.state["events"]
    if not events:
        # Pre-seed the prompt + model so slot_resume can start_run() without
        # the client having to send them. status="idle" tells the viewer to
        # render a "start" button.
        slot_log.state["prompt"] = slot.prompt
        slot_log.state["model"] = DEFAULT_MODEL
        slot_log.state["status"] = "idle"
        return
    slot_log.state["model"] = DEFAULT_MODEL
    last_kind = events[-1].get("kind")
    if last_kind == "run.done":
        return
    # Interrupted or errored — mark resumable but don't auto-start.
    if last_kind != "run.error":
        slot_log.state["status"] = "paused"


_WRAPPED_PROMPT_PREFIX = "Generate a direct,"


def _reconstruct_node(slot_log: SlotLog, node_id: str) -> Node | None:
    """Rebuild a minimal Node from the slot's event log so a standalone
    retry can call into `_generate_one` without re-running the divider.

    The bbox event carries everything the rescale step needs (origin,
    dimensions, proxy_shape, orientation). `image_prompt` is what banana
    actually sees, so we want to preserve whatever the prior run sent it
    rather than re-deriving from a "subject phrase" that may not exist
    anymore.

    Two log shapes coexist in practice:
      * NEW (current pipeline): `bbox.prompt` and `image.prompt` are bare
        subject phrases. The wrapping happens at the banana boundary.
      * OLD (pre-refactor logs still on disk): `bbox.prompt` and
        `image.prompt` are already the full wrapped studio-shot directive,
        because the pipeline wrapped earlier.

    Detect by sniffing the prefix. For a wrapped string we pass it through
    as `image_prompt` verbatim — re-wrapping would nest the template inside
    itself and produce a malformed banana prompt that Gemini refuses with
    MALFORMED_FUNCTION_CALL. For a bare subject we wrap normally.
    """
    events = slot_log.state["events"]
    bbox_event: dict[str, object] | None = None
    image_event: dict[str, object] | None = None
    for event in events:
        if event.get("id") != node_id:
            continue
        kind = event.get("kind")
        if kind == "bbox":
            bbox_event = event
        elif kind == "image":
            image_event = event
    if bbox_event is None:
        return None
    origin = bbox_event.get("origin")
    dimensions = bbox_event.get("dimensions")
    if not isinstance(origin, list) or not isinstance(dimensions, list):
        return None
    bbox = BoundingBox(
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
        dimensions=(float(dimensions[0]), float(dimensions[1]), float(dimensions[2])),
    )
    proxy_raw = bbox_event.get("proxy_shape")
    proxy_shape = ProxyShape(proxy_raw) if isinstance(proxy_raw, str) else None
    orientation_raw = bbox_event.get("orientation", 0)
    orientation: Orientation = int(orientation_raw) if isinstance(orientation_raw, (int, float, str)) else 0  # type: ignore[assignment]
    raw_prompt = (
        image_event.get("prompt") if image_event is not None else bbox_event.get("prompt")
    )
    raw_str = str(raw_prompt) if raw_prompt is not None else ""
    if raw_str.lstrip().startswith(_WRAPPED_PROMPT_PREFIX):
        image_prompt = raw_str
        subject_str = raw_str
    else:
        subject_str = raw_str
        image_prompt = wrap_image_prompt(subject_str, proxy_shape, bbox.size)
    parent_id = bbox_event.get("parent_id")
    return Node(
        id=node_id,
        prompt=subject_str,
        bbox=bbox,
        proxy_shape=proxy_shape,
        orientation=orientation,
        image_prompt=image_prompt,
        parent_id=str(parent_id) if isinstance(parent_id, str) else None,
    )


def _require_slot(slot_id: str) -> Slot:
    slot = SLOTS_BY_ID.get(slot_id)
    if slot is None:
        raise HTTPException(status_code=404, detail=f"unknown slot: {slot_id}")
    return slot


def _require_slot_log(slot_id: str) -> SlotLog:
    _require_slot(slot_id)
    return _slot_logs[slot_id]


async def _cancel_task(slot_id: str) -> None:
    task = _tasks.pop(slot_id, None)
    if task is None or task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def _sse(
    slot_log: SlotLog,
    q: asyncio.Queue[dict[str, object]],
    snapshot: list[dict[str, object]],
) -> AsyncIterator[str]:
    # All events ride the default SSE "message" type; the client dispatches
    # by `event.kind` internally. Keeps the client listener table flat.
    #
    # Snapshot terminal events do NOT close the stream — only live ones
    # do. That way a client re-subscribing to a finished slot (to drive a
    # standalone mesh retry) gets the historical timeline and then waits
    # on the live queue for the retry's new events, instead of being
    # disconnected the instant the past `run.done` replays.
    try:
        for event in snapshot:
            yield f"data: {json.dumps(event)}\n\n"
        while True:
            event = await q.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["kind"] in {"run.done", "run.error", "run.paused"}:
                return
    finally:
        slot_log.unsubscribe(q)


async def _run(slot_id: str) -> None:
    slot_log = _slot_logs[slot_id]
    rlog.bind(slot_log)
    prompt = slot_log.state["prompt"]
    model = slot_log.state["model"]
    try:
        await divider.run(
            run_id=slot_id,
            prompt=prompt,
            model=model,
            runs_dir=RUNS_DIR,
        )
    except asyncio.CancelledError:
        generation.cancel_pending(slot_id)
        raise
    except Exception as e:
        generation.cancel_pending(slot_id)
        # OpenRouter SDK errors only stringify to the top-level "Provider
        # returned error" message; the actually useful detail (upstream
        # provider's complaint, the request body that tripped it) lives on
        # `data.error.metadata` and on `e.body` (the response body the SDK
        # already read; `raw_response.text` would re-trigger a read on a
        # closed streaming response). Pull both into the logged message so
        # the run.error event tells us what went wrong.
        details = []
        data = getattr(e, "data", None)
        err = getattr(data, "error", None) if data is not None else None
        if err is not None:
            metadata = getattr(err, "metadata", None)
            if metadata:
                details.append(f"metadata={metadata}")
        body = getattr(e, "body", None)
        if body:
            details.append(f"body={body[:2000]}")
        suffix = (" | " + " | ".join(details)) if details else ""
        slot_log.log("run.error", message=f"{type(e).__name__}: {e}{suffix}")
        return
    # Pipeline tree is fully resolved; meshes may still be in flight.
    # Hold the run open until they all land so `run.done` truly means done.
    await generation.await_pending(slot_id)
    slot_log.finish_run()
