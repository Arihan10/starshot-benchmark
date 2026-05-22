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

from app.core.slots import DEFAULT_MODEL, SLOTS, SLOTS_BY_ID, Slot
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
            await threed.disconnect_runware()

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
            if events and events[-1].get("kind") == "run.error":
                slot_log.truncate_events_to(len(events) - 1)
            slot_log.state["model"] = DEFAULT_MODEL
            slot_log.state["status"] = "running"
        _tasks[slot.id] = asyncio.create_task(_run(slot.id))
        return {"slot_id": slot.id}

    @app.post("/slots/{slot_id}/reset")
    async def slot_reset(slot_id: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        slot = _require_slot(slot_id)
        await _cancel_task(slot_id)
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
    try:
        for event in snapshot:
            yield f"data: {json.dumps(event)}\n\n"
            if event["kind"] in {"run.done", "run.error"}:
                return
        while True:
            event = await q.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["kind"] in {"run.done", "run.error"}:
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
