"""HTTP API — endpoints scoped to a (slot, model) run.

Every benchmark slot can be driven by any of the aliased LLMs in
`app.core.slots.MODELS` in parallel — each (slot, model) cell is its own
resumable run with its own events.jsonl, mesh artifacts, and SSE stream.
On lifespan startup every cell is hydrated from disk: fresh ones sit
idle, interrupted ones come back as paused, completed ones stay done.
Nothing auto-launches; the viewer drives start/resume/reset per cell.

Every asyncio task is bound to its SlotLog via a ContextVar, so
concurrent pipeline work (e.g. running `hotel-room` against gpt and
opus at the same time) routes events to the right cell without
threading a handle through every call site. The Trellis queue is
process-global; its rows tag `slot_id` with the composite
`slot/model_alias` so the dashboard can filter to the visible cell.
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
from app.core.slots import (
    DEFAULT_MODEL_ALIAS,
    MODEL_ALIASES,
    MODELS,
    SLOTS,
    SLOTS_BY_ID,
    Slot,
)
from app.core.types import BoundingBox, Node, Orientation, ProxyShape
from app.pipeline import divider, generation
from app.services import llm, threed
from app.utils import logging as rlog
from app.utils.logging import SlotLog

# Where this process writes per-slot run artifacts. Set STARSHOT_RUNS_DIR to
# point at a different directory so multiple simultaneous processes don't
# trample each other.
RUNS_DIR = Path(os.environ.get("STARSHOT_RUNS_DIR", "./runs"))

# Keyed by (slot_id, model_alias). Each cell is an independent run.
RunKey = tuple[str, str]
_slot_logs: dict[RunKey, SlotLog] = {}
_tasks: dict[RunKey, asyncio.Task[None]] = {}


class RewindRequest(BaseModel):
    to_event_index: int


def _run_id(slot_id: str, model_alias: str) -> str:
    """Composite id used as `run_id` in pipeline code (divider, generation,
    threed queue, SlotLog.slot_id). The slash makes it work as a filesystem
    subpath under RUNS_DIR and as an artifact URL segment under /artifacts."""
    return f"{slot_id}/{model_alias}"


def _slot_dir(slot_id: str, model_alias: str) -> Path:
    return RUNS_DIR / slot_id / model_alias


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        for slot in SLOTS:
            for alias in MODEL_ALIASES:
                slot_dir = _slot_dir(slot.id, alias)
                slot_dir.mkdir(parents=True, exist_ok=True)
                slot_log = SlotLog(
                    _run_id(slot.id, alias), slot_dir / "events.jsonl"
                )
                slot_log.hydrate_from_disk()
                _slot_logs[(slot.id, alias)] = slot_log
                _maybe_launch(slot, alias, slot_log)
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
    async def list_slots() -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        return {
            "models": MODEL_ALIASES,
            "default_model": DEFAULT_MODEL_ALIAS,
            "slots": [_slot_summary(s) for s in SLOTS],
        }

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

    @app.get("/slots/{slot_id}/{model_alias}/events")
    async def slot_events(slot_id: str, model_alias: str) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        slot_log = _require_slot_log(slot_id, model_alias)
        # Subscribe and snapshot synchronously — no await between them, so no
        # log() call can land in both the snapshot and the live queue.
        q = slot_log.subscribe()
        snapshot = list(slot_log.state["events"])
        return StreamingResponse(
            _sse(slot_log, q, snapshot),
            media_type="text/event-stream",
        )

    @app.post("/slots/{slot_id}/{model_alias}/rewind")
    async def slot_rewind(  # pyright: ignore[reportUnusedFunction]
        slot_id: str, model_alias: str, req: RewindRequest,
    ) -> dict[str, int | str]:
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        slot_log = _require_slot_log(slot_id, model_alias)
        if slot_log.state.get("prompt") is None or slot_log.state.get("model") is None:
            raise HTTPException(
                status_code=400,
                detail="slot has no run to rewind",
            )
        await _cancel_task(slot_id, model_alias)
        new_len = slot_log.truncate_events_to(req.to_event_index)
        _tasks[(slot.id, model_alias)] = asyncio.create_task(_run(slot.id, model_alias))
        return {"slot_id": slot.id, "model": model_alias, "events": new_len}

    @app.post("/slots/{slot_id}/{model_alias}/resume")
    async def slot_resume(slot_id: str, model_alias: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        slot_log = _require_slot_log(slot_id, model_alias)
        status = slot_log.state.get("status")
        if status not in ("idle", "paused", "error"):
            raise HTTPException(
                status_code=400,
                detail=f"slot is {status}, not startable",
            )
        await _cancel_task(slot_id, model_alias)
        events = slot_log.state["events"]
        model_id = MODELS[model_alias]
        if status == "idle" and not events:
            # Fresh cell — emit run.start now so the run has somewhere to
            # anchor its events. From here on it's identical to a resume.
            slot_log.start_run(slot.prompt, model_id)
        else:
            # Drop the terminal sentinel so cache lookups don't trip over it
            # and so the resumed run lands clean events after the prior tail.
            if events and events[-1].get("kind") in ("run.error", "run.paused"):
                slot_log.truncate_events_to(len(events) - 1)
            slot_log.state["model"] = model_id
            slot_log.state["status"] = "running"
        _tasks[(slot.id, model_alias)] = asyncio.create_task(_run(slot.id, model_alias))
        return {"slot_id": slot.id, "model": model_alias}

    @app.post("/slots/{slot_id}/{model_alias}/pause")
    async def slot_pause(slot_id: str, model_alias: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        slot_log = _require_slot_log(slot_id, model_alias)
        status = slot_log.state.get("status")
        if status != "running":
            raise HTTPException(
                status_code=400,
                detail=f"slot is {status}, not pausable",
            )
        await _cancel_task(slot_id, model_alias)
        # _cancel_task awaits the cancellation, so the pipeline task has
        # already torn down (including generation.cancel_pending via _run's
        # CancelledError branch) by the time we emit the sentinel.
        slot_log.state["status"] = "paused"
        slot_log.log("run.paused")
        return {"slot_id": slot.id, "model": model_alias}

    @app.post("/slots/{slot_id}/{model_alias}/retry-mesh/{node_id}")
    async def slot_retry_mesh(  # pyright: ignore[reportUnusedFunction]
        slot_id: str, model_alias: str, node_id: str,
    ) -> dict[str, str]:
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        slot_log = _require_slot_log(slot_id, model_alias)
        node = _reconstruct_node(slot_log, node_id)
        if node is None:
            raise HTTPException(
                status_code=404,
                detail=f"no bbox event found for node: {node_id}",
            )
        async def _do_retry() -> None:
            rlog.bind(slot_log)
            llm.set_model(MODELS[model_alias])
            await generation.retry_node(
                node=node, runs_dir=RUNS_DIR,
                run_id=_run_id(slot.id, model_alias),
            )
        asyncio.create_task(_do_retry())
        return {"slot_id": slot.id, "model": model_alias, "node_id": node_id}

    @app.post("/slots/{slot_id}/{model_alias}/reset")
    async def slot_reset(slot_id: str, model_alias: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        await _cancel_task(slot_id, model_alias)
        # Cancel any standalone retries (registered on generation._pending
        # but with no owning _run task to drive cleanup) that the running
        # task wouldn't have touched.
        generation.cancel_pending(_run_id(slot.id, model_alias))
        slot_dir = _slot_dir(slot.id, model_alias)
        shutil.rmtree(slot_dir, ignore_errors=True)
        slot_dir.mkdir(parents=True, exist_ok=True)
        slot_log = SlotLog(_run_id(slot.id, model_alias), slot_dir / "events.jsonl")
        _slot_logs[(slot.id, model_alias)] = slot_log
        slot_log.start_run(slot.prompt, MODELS[model_alias])
        _tasks[(slot.id, model_alias)] = asyncio.create_task(_run(slot.id, model_alias))
        return {"slot_id": slot.id, "model": model_alias}

    return app


def _slot_summary(slot: Slot) -> dict[str, object]:
    runs: dict[str, dict[str, object]] = {}
    for alias in MODEL_ALIASES:
        slot_log = _slot_logs.get((slot.id, alias))
        state = (
            slot_log.state
            if slot_log is not None
            else {"status": "idle", "events": []}
        )
        events = state.get("events", [])
        runs[alias] = {
            "status": state.get("status", "idle"),
            "events_count": len(events),
            "last_kind": events[-1]["kind"] if events else None,
        }
    return {
        "id": slot.id,
        "prompt": slot.prompt,
        "runs": runs,
    }


def _maybe_launch(slot: Slot, model_alias: str, slot_log: SlotLog) -> None:
    """Prepare each (slot, model) cell for manual start. Nothing auto-runs
    at boot — the user clicks start/resume/retry per cell from the viewer.
    Fresh cells sit idle with their seed prompt prefilled; previously-
    running cells come back as paused (resumable); errored cells come
    back as error (retry-able); completed cells stay done."""
    model_id = MODELS[model_alias]
    events = slot_log.state["events"]
    if not events:
        # Pre-seed the prompt + model so slot_resume can start_run() without
        # the client having to send them. status="idle" tells the viewer to
        # render a "start" button.
        slot_log.state["prompt"] = slot.prompt
        slot_log.state["model"] = model_id
        slot_log.state["status"] = "idle"
        return
    slot_log.state["model"] = model_id
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


def _require_model(model_alias: str) -> None:
    if model_alias not in MODELS:
        raise HTTPException(status_code=404, detail=f"unknown model: {model_alias}")


def _require_slot_log(slot_id: str, model_alias: str) -> SlotLog:
    _require_slot(slot_id)
    _require_model(model_alias)
    log = _slot_logs.get((slot_id, model_alias))
    if log is None:
        raise HTTPException(
            status_code=404,
            detail=f"no run for slot={slot_id} model={model_alias}",
        )
    return log


async def _cancel_task(slot_id: str, model_alias: str) -> None:
    task = _tasks.pop((slot_id, model_alias), None)
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


async def _run(slot_id: str, model_alias: str) -> None:
    slot_log = _slot_logs[(slot_id, model_alias)]
    rlog.bind(slot_log)
    prompt = slot_log.state["prompt"]
    model = slot_log.state["model"]
    run_id = _run_id(slot_id, model_alias)
    try:
        await divider.run(
            run_id=run_id,
            prompt=prompt,
            model=model,
            runs_dir=RUNS_DIR,
        )
    except asyncio.CancelledError:
        generation.cancel_pending(run_id)
        raise
    except Exception as e:
        generation.cancel_pending(run_id)
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
    await generation.await_pending(run_id)
    slot_log.finish_run()
