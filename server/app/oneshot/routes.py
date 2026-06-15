"""HTTP surface for the one-shot track — everything under `/oneshot`.

Like the pipeline benchmark, the track has named RUNS: `ONESHOT_DIR/<run>/`
holds one (slot × model) cell matrix per run, so renditions are kept and
compared across prompt-file edits instead of overwritten in place. A run
also PINS a pipeline VERSION at creation (`pipeline.VERSIONS`), recorded in
`<run>/version` — comparing versions = one run per version. Within a
version the editable prompts under `prompts/<version>/` are still read
fresh on every launch (there is no per-run snapshot on this experimental
track), so the workflow is: pick version (and/or edit its prompts) →
create a run → start cells → compare.

Cells live at `ONESHOT_DIR/<run>/<slot>/<model>/`, fully separate from the
pipeline benchmark's runs. The event log, SSE framing, status semantics
(idle/running/paused/error/done, sticky run.done), and lifecycle endpoints
mirror the main API so the client machinery transfers; the implementation is
deliberately self-contained.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.oneshot import pipeline
from app.oneshot.pipeline import DEFAULT_VERSION, VERSIONS
from app.oneshot.slots import (
    DEFAULT_MODEL_ALIAS,
    MODEL_ALIASES,
    MODELS,
    SLOTS,
    SLOTS_BY_ID,
    Slot,
)
from app.utils import logging as rlog
from app.utils.logging import SlotLog

router = APIRouter(prefix="/oneshot")

# Parent directory holding the one-shot track's named runs.
ONESHOT_DIR = Path(os.environ.get("STARSHOT_ONESHOT_DIR", "./oneshot-runs"))

_ARTIFACT_MEDIA_TYPES = {".glb": "model/gltf-binary", ".gltf": "model/gltf+json"}

RunKey = tuple[str, str, str]  # (run, slot_id, model_alias)
_logs: dict[RunKey, SlotLog] = {}
_tasks: dict[RunKey, asyncio.Task[None]] = {}
_hydrated_runs: set[str] = set()
_current_run: str = ""
_booted = False


class CreateRunRequest(BaseModel):
    name: str
    version: str = DEFAULT_VERSION


# File inside a run dir recording the pipeline version pinned at creation.
_VERSION_MARKER = "version"


def _read_version(run: str) -> str:
    """The version a run was created with. Runs that predate versioning have
    no marker and behaved like today's default."""
    try:
        v = (_run_dir(run) / _VERSION_MARKER).read_text(encoding="utf-8").strip()
    except OSError:
        return DEFAULT_VERSION
    return v or DEFAULT_VERSION


def _run_id(run: str, slot_id: str, model_alias: str) -> str:
    """Composite id for run_id-keyed tables (pipeline pending, Trellis queue)
    and SlotLog console prefixes. Distinct from every main-benchmark run_id."""
    return f"oneshot/{run}/{slot_id}/{model_alias}"


def _run_dir(run: str) -> Path:
    return ONESHOT_DIR / run


def _cell_dir(run: str, slot_id: str, model_alias: str) -> Path:
    return ONESHOT_DIR / run / slot_id / model_alias


def _ensure_boot() -> None:
    """First-touch init: open on the most recently touched run (if any);
    other runs hydrate lazily when addressed."""
    global _booted, _current_run
    if _booted:
        return
    ONESHOT_DIR.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(
        (p for p in ONESHOT_DIR.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if run_dirs:
        _current_run = run_dirs[0].name
        _hydrate_run(_current_run)
    _booted = True


def _hydrate_run(run: str) -> None:
    """Build SlotLogs for every (slot, model) cell under a run. Mirrors the
    main boot semantics: fresh cells idle with prompt/model pre-seeded, cells
    killed mid-run come back paused, completed cells stay done."""
    if run in _hydrated_runs:
        return
    for slot in SLOTS:
        for alias in MODEL_ALIASES:
            cell = _cell_dir(run, slot.id, alias)
            cell.mkdir(parents=True, exist_ok=True)
            log = SlotLog(_run_id(run, slot.id, alias), cell / "events.jsonl")
            log.hydrate_from_disk()
            if not log.state["events"]:
                log.state["prompt"] = slot.prompt
                log.state["model"] = MODELS[alias].model
                log.state["status"] = "idle"
            else:
                log.state["model"] = MODELS[alias].model
                if log.state["status"] == "running":
                    log.state["status"] = "paused"
            _logs[(run, slot.id, alias)] = log
    _hydrated_runs.add(run)


def _resolve_run(run: str | None) -> str:
    _ensure_boot()
    return run or _current_run


def _require_run(run: str | None) -> str:
    run = _resolve_run(run)
    if not run:
        raise HTTPException(status_code=404, detail="no oneshot run yet — create one first")
    if not _run_dir(run).is_dir():
        raise HTTPException(status_code=404, detail=f"unknown oneshot run: {run}")
    _hydrate_run(run)
    return run


def _require_slot(slot_id: str) -> Slot:
    slot = SLOTS_BY_ID.get(slot_id)
    if slot is None:
        raise HTTPException(status_code=404, detail=f"unknown oneshot slot: {slot_id}")
    return slot


def _require_log(run: str, slot_id: str, model_alias: str) -> SlotLog:
    _require_slot(slot_id)
    if model_alias not in MODELS:
        raise HTTPException(status_code=404, detail=f"unknown oneshot model: {model_alias}")
    log = _logs.get((run, slot_id, model_alias))
    if log is None:
        # A run created before a registry edit may lack this cell; hydrate it.
        cell = _cell_dir(run, slot_id, model_alias)
        cell.mkdir(parents=True, exist_ok=True)
        log = SlotLog(_run_id(run, slot_id, model_alias), cell / "events.jsonl")
        log.hydrate_from_disk()
        if not log.state["events"]:
            log.state["prompt"] = SLOTS_BY_ID[slot_id].prompt
            log.state["model"] = MODELS[model_alias].model
            log.state["status"] = "idle"
        _logs[(run, slot_id, model_alias)] = log
    return log


async def _cancel_task(run: str, slot_id: str, model_alias: str) -> None:
    task = _tasks.pop((run, slot_id, model_alias), None)
    if task is None or task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def _run_cell(run: str, slot_id: str, model_alias: str, version: str) -> None:
    log = _logs[(run, slot_id, model_alias)]
    rlog.bind(log)
    run_id = _run_id(run, slot_id, model_alias)
    try:
        await pipeline.run(
            prompt=log.state["prompt"],
            model_cfg=MODELS[model_alias],
            version=version,
            run_id=run_id,
            cell_dir=_cell_dir(run, slot_id, model_alias),
            out_root=ONESHOT_DIR,
            canvas_ft=SLOTS_BY_ID[slot_id].canvas_ft,
        )
    except asyncio.CancelledError:
        pipeline.cancel_pending(run_id)
        raise
    except Exception as e:
        pipeline.cancel_pending(run_id)
        # Surface every detail the provider error carries — these logs are the
        # debugging surface for experimental backends, so nothing is truncated.
        details = []
        data = getattr(e, "data", None)
        err = getattr(data, "error", None) if data is not None else None
        if err is not None:
            metadata = getattr(err, "metadata", None)
            if metadata:
                details.append(f"metadata={metadata}")
        body = getattr(e, "body", None)
        if body:
            details.append(f"body={body}")
        suffix = (" | " + " | ".join(details)) if details else ""
        log.log("run.error", message=f"{type(e).__name__}: {e}{suffix}")
        return
    await pipeline.await_pending(run_id)
    log.finish_run()


def _require_model_key(model_alias: str) -> None:
    """Fail launches early with a clear message when the provider's key is
    missing, instead of a run.error halfway into the task."""
    cfg = MODELS[model_alias]
    if cfg.api_key_env is not None and not os.environ.get(cfg.api_key_env):
        raise HTTPException(
            status_code=409,
            detail=f"{cfg.api_key_env} is not set — required for {model_alias} ({cfg.model})",
        )


def _start_cell(run: str, slot_id: str, model_alias: str) -> None:
    version = _read_version(run)
    if version not in VERSIONS:
        # A hand-edited marker; fail the launch loudly instead of mid-task.
        raise HTTPException(
            status_code=409,
            detail=f"run {run!r} pins unknown oneshot version {version!r}",
        )
    slot = _require_slot(slot_id)
    log = _logs[(run, slot_id, model_alias)]
    events = log.state["events"]
    if log.state.get("status") == "idle" and not events:
        log.start_run(slot.prompt, MODELS[model_alias].model)
    else:
        if events and events[-1].get("kind") in ("run.error", "run.paused"):
            log.truncate_events_to(len(events) - 1)
        log.state["model"] = MODELS[model_alias].model
        log.state["status"] = "running"
    _tasks[(run, slot_id, model_alias)] = asyncio.create_task(
        _run_cell(run, slot_id, model_alias, version),
    )


@router.get("/runs")
async def list_runs() -> dict[str, object]:
    _ensure_boot()
    items: list[dict[str, object]] = []
    if ONESHOT_DIR.exists():
        for p in sorted(
            (p for p in ONESHOT_DIR.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            items.append({
                "name": p.name,
                "modified_at": p.stat().st_mtime,
                "version": _read_version(p.name),
            })
    return {"runs": items, "current": _current_run}


@router.post("/runs")
async def create_run(req: CreateRunRequest) -> dict[str, str]:
    _ensure_boot()
    name = req.name.strip()
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid run name")
    if req.version not in VERSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown oneshot version: {req.version} (expected one of {', '.join(VERSIONS)})",
        )
    run_dir = _run_dir(name)
    if run_dir.exists():
        raise HTTPException(status_code=409, detail=f"oneshot run already exists: {name}")
    run_dir.mkdir(parents=True)
    (run_dir / _VERSION_MARKER).write_text(req.version + "\n", encoding="utf-8")
    _hydrate_run(name)
    global _current_run
    _current_run = name
    return {"current": name, "version": req.version}


@router.get("/slots")
async def list_slots(run: str | None = None) -> dict[str, object]:
    run = _resolve_run(run)
    if run and _run_dir(run).is_dir():
        _hydrate_run(run)
    slots: list[dict[str, object]] = []
    for slot in SLOTS:
        runs: dict[str, dict[str, object]] = {}
        for alias in MODEL_ALIASES:
            log = _logs.get((run, slot.id, alias)) if run else None
            state = log.state if log is not None else {"status": "idle", "events": []}
            events = state.get("events", [])
            runs[alias] = {
                "status": state.get("status", "idle"),
                "events_count": len(events),
                "last_kind": events[-1]["kind"] if events else None,
            }
        slots.append({"id": slot.id, "prompt": slot.prompt, "runs": runs})
    return {
        "run": run,
        "version": _read_version(run) if run and _run_dir(run).is_dir() else None,
        "versions": VERSIONS,
        "models": MODEL_ALIASES,
        "default_model": DEFAULT_MODEL_ALIAS,
        "slots": slots,
    }


@router.post("/slots/{slot_id}/{model_alias}/resume")
async def slot_resume(slot_id: str, model_alias: str, run: str | None = None) -> dict[str, str]:
    run = _require_run(run)
    log = _require_log(run, slot_id, model_alias)
    if any(e.get("kind") == "run.done" for e in log.state["events"]):
        raise HTTPException(status_code=409, detail="run is complete; reset to start a new run")
    status = log.state.get("status")
    if status not in ("idle", "paused", "error"):
        raise HTTPException(status_code=400, detail=f"cell is {status}, not startable")
    _require_model_key(model_alias)
    await _cancel_task(run, slot_id, model_alias)
    _start_cell(run, slot_id, model_alias)
    return {"run": run, "slot_id": slot_id, "model": model_alias}


@router.post("/slots/{slot_id}/{model_alias}/pause")
async def slot_pause(slot_id: str, model_alias: str, run: str | None = None) -> dict[str, str]:
    run = _require_run(run)
    log = _require_log(run, slot_id, model_alias)
    if log.state.get("status") != "running":
        raise HTTPException(status_code=400, detail=f"cell is {log.state.get('status')}, not pausable")
    await _cancel_task(run, slot_id, model_alias)
    log.state["status"] = "paused"
    log.log("run.paused")
    return {"run": run, "slot_id": slot_id, "model": model_alias}


@router.post("/slots/{slot_id}/{model_alias}/reset")
async def slot_reset(
    slot_id: str,
    model_alias: str,
    run: str | None = None,
    start: bool = True,
) -> dict[str, str]:
    run = _require_run(run)
    slot = _require_slot(slot_id)
    _require_log(run, slot_id, model_alias)
    if start:
        _require_model_key(model_alias)
    await _cancel_task(run, slot_id, model_alias)
    pipeline.cancel_pending(_run_id(run, slot_id, model_alias))
    old = _logs.get((run, slot_id, model_alias))
    if old is not None:
        old.close()
    cell = _cell_dir(run, slot_id, model_alias)
    shutil.rmtree(cell, ignore_errors=True)
    cell.mkdir(parents=True, exist_ok=True)
    fresh = SlotLog(_run_id(run, slot_id, model_alias), cell / "events.jsonl")
    fresh.state["prompt"] = slot.prompt
    fresh.state["model"] = MODELS[model_alias].model
    _logs[(run, slot_id, model_alias)] = fresh
    if start:
        _start_cell(run, slot_id, model_alias)
    return {"run": run, "slot_id": slot_id, "model": model_alias}


@router.get("/slots/{slot_id}/{model_alias}/events")
async def slot_events(
    slot_id: str,
    model_alias: str,
    since: int = -1,
    run: str | None = None,
) -> StreamingResponse:
    run = _require_run(run)
    log = _require_log(run, slot_id, model_alias)
    q = log.subscribe()
    snapshot = list(log.state["events"])
    if since >= 0:
        snapshot = [
            e for e in snapshot
            if isinstance(e.get("index"), int) and e["index"] > since
        ]
    return StreamingResponse(_sse(log, q, snapshot), media_type="text/event-stream")


@router.get("/slots/{slot_id}/events-all")
async def slot_events_all(
    slot_id: str,
    models: str,
    run: str | None = None,
) -> StreamingResponse:
    """One multiplexed SSE for a whole comparison row: every event of every
    requested model's cell on this slot, wrapped as `{"model": alias,
    "event": {...}}`. The grid view needs N cells live at once, and N separate
    EventSources would exhaust the browser's ~6-connection-per-origin budget —
    this carries any number of models over ONE connection. Terminal events do
    not close the stream (other cells keep running; the client owns closing)."""
    run = _require_run(run)
    aliases = [a for a in (s.strip() for s in models.split(",")) if a]
    if not aliases:
        raise HTTPException(status_code=400, detail="models query param is empty")
    logs: dict[str, SlotLog] = {a: _require_log(run, slot_id, a) for a in aliases}
    # Subscribe + snapshot synchronously — no await in between, so no event
    # can land in both a snapshot and its live queue.
    subs = {a: log.subscribe() for a, log in logs.items()}
    snapshots = {a: list(log.state["events"]) for a, log in logs.items()}
    return StreamingResponse(
        _sse_multi(logs, subs, snapshots),
        media_type="text/event-stream",
    )


async def _sse_multi(
    logs: dict[str, SlotLog],
    subs: dict[str, asyncio.Queue[dict[str, object]]],
    snapshots: dict[str, list[dict[str, object]]],
) -> AsyncIterator[str]:
    merged: asyncio.Queue[tuple[str, dict[str, object]]] = asyncio.Queue()

    async def _pump(alias: str, q: asyncio.Queue[dict[str, object]]) -> None:
        while True:
            event = await q.get()
            await merged.put((alias, event))

    pumps = [asyncio.create_task(_pump(a, q)) for a, q in subs.items()]
    try:
        for alias, snapshot in snapshots.items():
            for event in snapshot:
                yield f"data: {json.dumps({'model': alias, 'event': event})}\n\n"
        while True:
            alias, event = await merged.get()
            yield f"data: {json.dumps({'model': alias, 'event': event})}\n\n"
    finally:
        for t in pumps:
            t.cancel()
        for alias, q in subs.items():
            logs[alias].unsubscribe(q)


@router.get("/artifacts/{artifact_path:path}")
async def artifact(artifact_path: str) -> FileResponse:
    base = ONESHOT_DIR.resolve()
    target = (base / artifact_path).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(target, media_type=_ARTIFACT_MEDIA_TYPES.get(target.suffix.lower()))


async def _sse(
    log: SlotLog,
    q: asyncio.Queue[dict[str, object]],
    snapshot: list[dict[str, object]],
) -> AsyncIterator[str]:
    # Snapshot terminal events don't close the stream — only live ones do —
    # so re-subscribing to a finished cell still tails late mesh events.
    try:
        for event in snapshot:
            yield f"data: {json.dumps(event)}\n\n"
        while True:
            event = await q.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["kind"] in {"run.done", "run.error", "run.paused"}:
                return
    finally:
        log.unsubscribe(q)


async def shutdown() -> None:
    """Tear down one-shot tasks + logs; called from the main app lifespan."""
    for task in _tasks.values():
        task.cancel()
    for task in list(_tasks.values()):
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    _tasks.clear()
    for (run, slot_id, alias), log in _logs.items():
        pipeline.cancel_pending(_run_id(run, slot_id, alias))
        log.close()
