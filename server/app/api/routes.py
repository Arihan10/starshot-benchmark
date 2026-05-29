"""HTTP API — endpoints scoped to a (run, slot, model) cell.

`RUNS_DIR` is now a parent directory holding many named *runs*; each run
is a versioned set of (slot, model) cells with its own prompt snapshot,
events.jsonl, mesh artifacts, and SSE streams. The viewer picks the
active run at runtime via `GET /runs` / `POST /runs/{name}/activate` —
no more relaunching the server with a different `STARSHOT_RUNS_DIR`.

For each (run, slot, model) cell: fresh ones sit idle, interrupted ones
come back as paused, completed ones stay done. Nothing auto-launches;
the viewer drives start/resume/reset per cell. Runs are hydrated lazily
on activation; the initial active run is the newest subdir of RUNS_DIR
(or `default` if RUNS_DIR is empty).

Every asyncio task is bound to its SlotLog via a ContextVar so
concurrent pipeline work routes events to the right cell without
threading a handle through every call site. The Trellis queue is
process-global; rows tag `slot_id` with the composite
`run/slot/model_alias` so the dashboard filters to the visible cell.
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

# Parent directory holding many named runs. Each immediate subdirectory
# is one run; cells live at RUNS_DIR/<run>/<slot>/<model>.
RUNS_DIR = Path(os.environ.get("STARSHOT_RUNS_DIR", "./runs"))

# Source file we snapshot into each newly-created run so prompt-versioned
# AB tests are reproducible after the source has moved on.
PROMPTS_SOURCE = Path(__file__).resolve().parent.parent / "core" / "prompts.py"
PROMPT_SNAPSHOT_NAME = "prompts_snapshot.py"

# Keyed by (run_name, slot_id, model_alias). Each cell is an independent
# pipeline. Lazy-populated: only runs the user has activated are loaded.
RunKey = tuple[str, str, str]
_slot_logs: dict[RunKey, SlotLog] = {}
_tasks: dict[RunKey, asyncio.Task[None]] = {}
_hydrated_runs: set[str] = set()
_current_run: str = ""


class RewindRequest(BaseModel):
    to_event_index: int


class CreateRunRequest(BaseModel):
    name: str


def _run_id(run: str, slot_id: str, model_alias: str) -> str:
    """Composite id used as `run_id` in pipeline code (divider, generation,
    threed queue, SlotLog.slot_id). The slashes make it work as a filesystem
    subpath under RUNS_DIR and as an artifact URL segment under /artifacts."""
    return f"{run}/{slot_id}/{model_alias}"


def _run_dir(run: str) -> Path:
    return RUNS_DIR / run


def _slot_dir(run: str, slot_id: str, model_alias: str) -> Path:
    return RUNS_DIR / run / slot_id / model_alias


def _pick_initial_run() -> str:
    """Most-recently-modified subdirectory of RUNS_DIR, or `default` if
    none exist yet. Picked once at startup; subsequent activations are
    driven by the client."""
    if RUNS_DIR.exists():
        subdirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
        if subdirs:
            latest = max(subdirs, key=lambda p: p.stat().st_mtime)
            return latest.name
    return "default"


def _ensure_prompt_snapshot(run: str) -> None:
    """Copy the live prompts.py into RUNS_DIR/<run>/ if not already there.
    Called only when a brand-new run is created via POST /runs; legacy
    runs that pre-date this feature keep whatever (or nothing) they had,
    so we never overwrite the historical record."""
    target = _run_dir(run) / PROMPT_SNAPSHOT_NAME
    if target.exists():
        return
    if PROMPTS_SOURCE.exists():
        target.write_text(PROMPTS_SOURCE.read_text())


def _hydrate_run(run: str) -> None:
    """Build SlotLogs for every (slot, model) cell under RUNS_DIR/<run>/.
    Idempotent — calling twice is a no-op. Hydration is per-run so the
    /runs/ parent can hold many old run sets without paying their startup
    cost until the user clicks one."""
    if run in _hydrated_runs:
        return
    run_dir = _run_dir(run)
    run_dir.mkdir(parents=True, exist_ok=True)
    for slot in SLOTS:
        for alias in MODEL_ALIASES:
            slot_dir = _slot_dir(run, slot.id, alias)
            slot_dir.mkdir(parents=True, exist_ok=True)
            slot_log = SlotLog(
                _run_id(run, slot.id, alias), slot_dir / "events.jsonl"
            )
            slot_log.hydrate_from_disk()
            _slot_logs[(run, slot.id, alias)] = slot_log
            _maybe_launch(slot, alias, slot_log)
    _hydrated_runs.add(run)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        global _current_run
        _current_run = _pick_initial_run()
        _hydrate_run(_current_run)
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

    @app.get("/runs")
    async def list_runs() -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        items: list[dict[str, object]] = []
        if RUNS_DIR.exists():
            for p in sorted(
                (p for p in RUNS_DIR.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                items.append({
                    "name": p.name,
                    "modified_at": p.stat().st_mtime,
                    "has_prompt_snapshot": (p / PROMPT_SNAPSHOT_NAME).exists(),
                })
        return {"runs": items, "current": _current_run}

    @app.post("/runs")
    async def create_run(req: CreateRunRequest) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        name = req.name.strip()
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(status_code=400, detail="invalid run name")
        run_dir = _run_dir(name)
        if run_dir.exists():
            raise HTTPException(status_code=409, detail=f"run already exists: {name}")
        run_dir.mkdir(parents=True)
        _ensure_prompt_snapshot(name)
        _hydrate_run(name)
        global _current_run
        _current_run = name
        return {"current": name}

    @app.post("/runs/{name}/activate")
    async def activate_run(name: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        run_dir = _run_dir(name)
        if not run_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"unknown run: {name}")
        _hydrate_run(name)
        global _current_run
        _current_run = name
        return {"current": name}

    @app.get("/slots")
    async def list_slots() -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        return {
            "run": _current_run,
            "models": MODEL_ALIASES,
            "default_model": DEFAULT_MODEL_ALIAS,
            "slots": [_slot_summary(s, _current_run) for s in SLOTS],
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
        run = _current_run
        slot_log = _require_slot_log(run, slot_id, model_alias)
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
        run = _current_run
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        if slot_log.state.get("prompt") is None or slot_log.state.get("model") is None:
            raise HTTPException(
                status_code=400,
                detail="slot has no run to rewind",
            )
        await _cancel_task(run, slot_id, model_alias)
        new_len = slot_log.truncate_events_to(req.to_event_index)
        _tasks[(run, slot.id, model_alias)] = asyncio.create_task(
            _run(run, slot.id, model_alias)
        )
        return {"run": run, "slot_id": slot.id, "model": model_alias, "events": new_len}

    @app.post("/slots/{slot_id}/{model_alias}/resume")
    async def slot_resume(slot_id: str, model_alias: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        run = _current_run
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        status = slot_log.state.get("status")
        if status not in ("idle", "paused", "error"):
            raise HTTPException(
                status_code=400,
                detail=f"slot is {status}, not startable",
            )
        await _cancel_task(run, slot_id, model_alias)
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
        _tasks[(run, slot.id, model_alias)] = asyncio.create_task(
            _run(run, slot.id, model_alias)
        )
        return {"run": run, "slot_id": slot.id, "model": model_alias}

    @app.post("/slots/{slot_id}/{model_alias}/pause")
    async def slot_pause(slot_id: str, model_alias: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        run = _current_run
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        status = slot_log.state.get("status")
        if status != "running":
            raise HTTPException(
                status_code=400,
                detail=f"slot is {status}, not pausable",
            )
        await _cancel_task(run, slot_id, model_alias)
        # _cancel_task awaits the cancellation, so the pipeline task has
        # already torn down (including generation.cancel_pending via _run's
        # CancelledError branch) by the time we emit the sentinel.
        slot_log.state["status"] = "paused"
        slot_log.log("run.paused")
        return {"run": run, "slot_id": slot.id, "model": model_alias}

    @app.post("/slots/{slot_id}/{model_alias}/retry-mesh/{node_id}")
    async def slot_retry_mesh(  # pyright: ignore[reportUnusedFunction]
        slot_id: str, model_alias: str, node_id: str,
    ) -> dict[str, str]:
        run = _current_run
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        slot_log = _require_slot_log(run, slot_id, model_alias)
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
                run_id=_run_id(run, slot.id, model_alias),
            )
        asyncio.create_task(_do_retry())
        return {"run": run, "slot_id": slot.id, "model": model_alias, "node_id": node_id}

    @app.post("/slots/{slot_id}/{model_alias}/reset")
    async def slot_reset(slot_id: str, model_alias: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        run = _current_run
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        await _cancel_task(run, slot_id, model_alias)
        # Cancel any standalone retries (registered on generation._pending
        # but with no owning _run task to drive cleanup) that the running
        # task wouldn't have touched.
        generation.cancel_pending(_run_id(run, slot.id, model_alias))
        slot_dir = _slot_dir(run, slot.id, model_alias)
        shutil.rmtree(slot_dir, ignore_errors=True)
        slot_dir.mkdir(parents=True, exist_ok=True)
        slot_log = SlotLog(
            _run_id(run, slot.id, model_alias), slot_dir / "events.jsonl"
        )
        _slot_logs[(run, slot.id, model_alias)] = slot_log
        slot_log.start_run(slot.prompt, MODELS[model_alias])
        _tasks[(run, slot.id, model_alias)] = asyncio.create_task(
            _run(run, slot.id, model_alias)
        )
        return {"run": run, "slot_id": slot.id, "model": model_alias}

    return app


def _slot_summary(slot: Slot, run: str) -> dict[str, object]:
    runs: dict[str, dict[str, object]] = {}
    for alias in MODEL_ALIASES:
        slot_log = _slot_logs.get((run, slot.id, alias))
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


def _require_slot_log(run: str, slot_id: str, model_alias: str) -> SlotLog:
    _require_slot(slot_id)
    _require_model(model_alias)
    log = _slot_logs.get((run, slot_id, model_alias))
    if log is None:
        raise HTTPException(
            status_code=404,
            detail=f"no run for run={run} slot={slot_id} model={model_alias}",
        )
    return log


async def _cancel_task(run: str, slot_id: str, model_alias: str) -> None:
    task = _tasks.pop((run, slot_id, model_alias), None)
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


async def _run(run: str, slot_id: str, model_alias: str) -> None:
    slot_log = _slot_logs[(run, slot_id, model_alias)]
    rlog.bind(slot_log)
    prompt = slot_log.state["prompt"]
    model = slot_log.state["model"]
    run_id = _run_id(run, slot_id, model_alias)
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
