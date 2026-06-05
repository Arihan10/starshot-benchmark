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
import struct
from datetime import datetime
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.core import prompt_runtime
from app.core.slots import (
    DEFAULT_MODEL_ALIAS,
    MODEL_ALIASES,
    MODELS,
    SLOTS,
    SLOTS_BY_ID,
    Slot,
)
from app.core.types import BoundingBox, Node, Orientation, ProxyShape
from app.pipeline import versions
from app.services import llm, threed
from app.utils import logging as rlog
from app.utils.logging import SlotLog

# Parent directory holding many named runs. Each immediate subdirectory
# is one run; cells live at RUNS_DIR/<run>/<slot>/<model>.
RUNS_DIR = Path(os.environ.get("STARSHOT_RUNS_DIR", "./runs"))

# Which per-cell mesh directory the scene bundle streams from. Defaults to the
# originals ("objects"); set STARSHOT_OBJECTS_SUBDIR=objects-optimized to serve
# the re-baked optimized set (scripts/rebake_runs.py) instead. Falls back to
# "objects" for any cell that hasn't been migrated.
OBJECTS_SUBDIR = os.environ.get("STARSHOT_OBJECTS_SUBDIR", "objects")

# Python's mimetypes doesn't know glTF; without these the artifact route would
# hand the loader its GLBs as text/plain.
_ARTIFACT_MEDIA_TYPES = {".glb": "model/gltf-binary", ".gltf": "model/gltf+json"}

# Source file we snapshot into each newly-created run so prompt-versioned
# AB tests are reproducible after the source has moved on.
PROMPTS_SOURCE = Path(__file__).resolve().parent.parent / "core" / "prompts.py"
PROMPT_SNAPSHOT_NAME = "prompts_snapshot.py"

# Keyed by (run_name, slot_id, model_alias). Each cell is an independent
# pipeline. Lazy-populated: only runs the user has activated are loaded.
RunKey = tuple[str, str, str]
_slot_logs: dict[RunKey, SlotLog] = {}
_tasks: dict[RunKey, asyncio.Task[None]] = {}
_retry_tasks: set[asyncio.Task[None]] = set()
_hydrated_runs: set[str] = set()
_current_run: str = ""


class RewindRequest(BaseModel):
    to_event_index: int


class CreateRunRequest(BaseModel):
    name: str


class StepTestRequest(BaseModel):
    """One-off "what if I edited this prompt" replay of a single pipeline
    step. `system`/`user` are the (possibly hand-edited) messages;
    `schema_name` is the output-schema class name recorded on the original
    `cache.llm` event (resolved against the run's bound prompt module);
    `model` is the OpenRouter id to run it on. Deliberately carries no slot
    id — the call never reads or writes any cell's event log."""

    system: str
    user: str
    schema_name: str
    model: str


def _run_id(run: str, slot_id: str, model_alias: str) -> str:
    """Composite id used as `run_id` in pipeline code (divider, generation,
    threed queue, SlotLog.slot_id). The slashes make it work as a filesystem
    subpath under RUNS_DIR and as an artifact URL segment under /artifacts."""
    return f"{run}/{slot_id}/{model_alias}"


def _run_dir(run: str) -> Path:
    return RUNS_DIR / run


def _slot_dir(run: str, slot_id: str, model_alias: str) -> Path:
    return RUNS_DIR / run / slot_id / model_alias


def _resolve_run(run: str | None) -> str:
    """Every cell endpoint names its target run/version explicitly so the
    concurrently-running versions never route through a shared global.
    The client always sends `?run=`; `_current_run` is only the fallback for a
    client that hasn't picked one yet (boot, or a legacy caller)."""
    return run or _current_run


def _run_has_data(run: str) -> bool:
    """True when any (slot, model) cell under this run has a non-empty
    events.jsonl — i.e. there is a rendition worth archiving. Lets the
    version snapshot skip versions the user never launched."""
    run_dir = _run_dir(run)
    if not run_dir.is_dir():
        return False
    for events_path in run_dir.glob("*/*/events.jsonl"):
        try:
            if events_path.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


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


def _prompt_module_for_run(run: str):
    snapshot = _run_dir(run) / PROMPT_SNAPSHOT_NAME
    if snapshot.exists():
        return prompt_runtime.load_snapshot(snapshot)
    return None


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
            slot_log = SlotLog(_run_id(run, slot.id, alias), slot_dir / "events.jsonl")
            slot_log.hydrate_from_disk()
            _slot_logs[(run, slot.id, alias)] = slot_log
            _maybe_launch(slot, alias, slot_log)
    _hydrated_runs.add(run)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        global _current_run
        # Seed the reserved version runs so their cells exist and can
        # stream status from boot; the viewer opens on v3 (today's behavior).
        for ver in versions.VERSIONS:
            _hydrate_run(ver.run_name)
        _current_run = versions.DEFAULT_VERSION.run_name
        try:
            yield
        finally:
            rlog.suppress_console()
            for slot_log in _slot_logs.values():
                slot_log.close()
            for run_name, slot_id, model_alias in list(_slot_logs.keys()):
                versions.for_run(run_name).generation.cancel_pending(
                    _run_id(run_name, slot_id, model_alias),
                )
            for task in _tasks.values():
                task.cancel()
            for task in _retry_tasks:
                task.cancel()
            for task in list(_tasks.values()):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for task in list(_retry_tasks):
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

    @app.get("/artifacts/{artifact_path:path}")
    async def artifact(artifact_path: str) -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        # Per-cell artifacts (events.jsonl, reference PNGs, GLBs). The URLs are
        # baked into the event log as `<cell>/objects/<id>.<ext>`. Cells migrated
        # onto the optimized library (rebake_runs.py --prune-originals) hold those
        # files under `objects-optimized/` instead, with `objects/` deleted — so a
        # miss there transparently falls back to the optimized twin.
        base = RUNS_DIR.resolve()
        target = (base / artifact_path).resolve()
        if not target.is_relative_to(base):
            raise HTTPException(status_code=404)
        rooted = f"/{artifact_path}"
        if not target.is_file() and "/objects/" in rooted:
            alt_rel = rooted.replace("/objects/", "/objects-optimized/", 1).lstrip("/")
            alt = (base / alt_rel).resolve()
            if alt.is_relative_to(base) and alt.is_file():
                target = alt
        if not target.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(target, media_type=_ARTIFACT_MEDIA_TYPES.get(target.suffix.lower()))

    @app.get("/runs")
    async def list_runs() -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        items: list[dict[str, object]] = []
        if RUNS_DIR.exists():
            for p in sorted(
                (p for p in RUNS_DIR.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                items.append(
                    {
                        "name": p.name,
                        "modified_at": p.stat().st_mtime,
                        "has_prompt_snapshot": (p / PROMPT_SNAPSHOT_NAME).exists(),
                    }
                )
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

    @app.post("/runs/snapshot")
    async def snapshot_run() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        src = _run_dir(_current_run)
        if not src.is_dir():
            raise HTTPException(status_code=404, detail=f"current run not found: {_current_run}")
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        name = f"save_{ts}"
        dst = _run_dir(name)
        if dst.exists():
            raise HTTPException(status_code=409, detail=f"snapshot already exists: {name}")
        shutil.copytree(src, dst)
        return {"snapshot": name}

    @app.post("/runs/{name}/activate")
    async def activate_run(name: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        run_dir = _run_dir(name)
        if not run_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"unknown run: {name}")
        _hydrate_run(name)
        global _current_run
        _current_run = name
        return {"current": name}

    @app.post("/llm/test")
    async def llm_test(  # pyright: ignore[reportUnusedFunction]
        req: StepTestRequest,
        run: str | None = None,
    ) -> dict[str, object]:
        """Re-run ONE pipeline step's LLM call with (optionally edited)
        system/user prompts and hand back the parsed output — the engine
        behind the prompt-tuning sandbox. Resolves the output schema against
        the run's prompt module (so v1/v2 snapshots and live v3/v4 all work),
        then calls `llm.call_llm_once`, which neither reads the LLM cache nor
        writes a `cache.llm` event. The result is rendered transiently in the
        client and discarded; nothing about any run is mutated."""
        run = _resolve_run(run)
        ver = versions.for_run(run)
        module = ver.prompt_module or _prompt_module_for_run(run)
        prompt_runtime.bind(module)
        schema_cls = getattr(prompt_runtime.current(), req.schema_name, None)
        if not (isinstance(schema_cls, type) and issubclass(schema_cls, BaseModel)):
            raise HTTPException(
                status_code=404,
                detail=f"unknown output schema: {req.schema_name}",
            )
        if not req.model:
            raise HTTPException(status_code=400, detail="model is required")
        llm.set_model(req.model)
        try:
            validated, reasoning, usage = await llm.call_llm_once(
                system=req.system,
                user=req.user,
                output_schema=schema_cls,
                model=req.model,
                log_retries=False,
            )
        except Exception as e:
            # Surface provider/parse failures as a clean 502 the sandbox can
            # show inline, rather than a 500 with a stack trace.
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
        return {
            "output": validated.model_dump(mode="json"),
            "reasoning": reasoning,
            "schema": req.schema_name,
            "tokens_in": getattr(usage, "prompt_tokens", None),
            "tokens_out": getattr(usage, "completion_tokens", None),
        }

    @app.get("/slots")
    async def list_slots(run: str | None = None) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        return {
            "run": run,
            "models": MODEL_ALIASES,
            "default_model": DEFAULT_MODEL_ALIAS,
            "slots": [_slot_summary(s, run) for s in SLOTS],
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
    async def slot_events(slot_id: str, model_alias: str, since: int = -1, run: str | None = None) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        # Subscribe and snapshot synchronously — no await between them, so no
        # log() call can land in both the snapshot and the live queue.
        q = slot_log.subscribe()
        snapshot = list(slot_log.state["events"])
        # `since` is the CQRS live-tail cut: the client already painted the
        # scene from /scene (folded up to `since`) and loads the full history
        # directly, so it only needs events *after* that index here. The gap
        # between the /scene read and this subscribe is covered because we
        # filter the fresh snapshot, not a stale copy. Default -1 = full
        # snapshot (legacy replay path, still used by reset/rewind).
        if since >= 0:
            snapshot = [
                e for e in snapshot
                if isinstance(e.get("index"), int) and e["index"] > since
            ]
        return StreamingResponse(
            _sse(slot_log, q, snapshot),
            media_type="text/event-stream",
        )

    @app.get("/slots/{slot_id}/{model_alias}/scene")
    async def slot_scene(slot_id: str, model_alias: str, run: str | None = None) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        # CQRS read model: fold the event log into the minimal renderable scene
        # state so the client paints directly instead of replaying ~2k events.
        # Returns `last_index` (the fold's cut) so the client can pick up the
        # live tail at `?since=last_index` with no gap or overlap.
        run = _resolve_run(run)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        return _scene_projection(list(slot_log.state["events"]))

    @app.get("/slots/{slot_id}/{model_alias}/meshes")
    async def slot_meshes(slot_id: str, model_alias: str, run: str | None = None) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        # One-request scene bundle: stream every finished GLB for this cell in
        # a single length-prefixed response so the client loads a whole scene
        # over one connection instead of one HTTP request per mesh (browsers
        # cap those at ~6 per origin, and they'd contend with the SSE stream
        # and polls on this same origin). The client keys each blob to its
        # `model` event by file stem; see `_mesh_bundle`.
        run = _resolve_run(run)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        cell_dir = slot_log.events_path.parent
        objects_dir = cell_dir / OBJECTS_SUBDIR
        if not objects_dir.is_dir():
            # Migrated cells keep only objects-optimized (rebake_runs.py
            # --prune-originals); pre-migration cells keep only objects.
            objects_dir = next(
                (cell_dir / d for d in ("objects-optimized", "objects") if (cell_dir / d).is_dir()),
                objects_dir,
            )
        return StreamingResponse(
            _mesh_bundle(objects_dir),
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/slots/{slot_id}/{model_alias}/rewind")
    async def slot_rewind(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        req: RewindRequest,
        run: str | None = None,
    ) -> dict[str, int | str]:
        run = _resolve_run(run)
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
        _tasks[(run, slot.id, model_alias)] = asyncio.create_task(_run(run, slot.id, model_alias))
        return {"run": run, "slot_id": slot.id, "model": model_alias, "events": new_len}

    @app.post("/slots/{slot_id}/{model_alias}/resume")
    async def slot_resume(slot_id: str, model_alias: str, run: str | None = None) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        # A completed run is terminal: resuming it would re-enter the pipeline
        # and generate a second run into the same cell. `run.done` is sticky
        # in the status derivation so the guard below normally catches this;
        # the explicit check encodes the rule directly (complete ⇒ reset-only)
        # and is robust to any stale in-memory status.
        if any(e.get("kind") == "run.done" for e in slot_log.state["events"]):
            raise HTTPException(
                status_code=409,
                detail="run is complete; reset to start a new run",
            )
        status = slot_log.state.get("status")
        if status not in ("idle", "paused", "error"):
            raise HTTPException(
                status_code=400,
                detail=f"slot is {status}, not startable",
            )
        await _start_cell(run, slot_id, model_alias)
        return {"run": run, "slot_id": slot.id, "model": model_alias}

    @app.post("/slots/{slot_id}/{model_alias}/pause")
    async def slot_pause(slot_id: str, model_alias: str, run: str | None = None) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
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
        slot_id: str,
        model_alias: str,
        node_id: str,
        run: str | None = None,
    ) -> dict[str, str]:
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        ver = versions.for_run(run)
        prompt_runtime.bind(ver.prompt_module or _prompt_module_for_run(run))
        node = _reconstruct_node(slot_log, node_id)
        if node is None:
            raise HTTPException(
                status_code=404,
                detail=f"no bbox event found for node: {node_id}",
            )

        async def _do_retry() -> None:
            rlog.bind(slot_log)
            llm.set_model(MODELS[model_alias])
            prompt_runtime.bind(ver.prompt_module or _prompt_module_for_run(run))
            await ver.generation.retry_node(
                node=node,
                runs_dir=RUNS_DIR,
                run_id=_run_id(run, slot.id, model_alias),
            )

        task = asyncio.create_task(_do_retry())
        _retry_tasks.add(task)
        task.add_done_callback(_retry_tasks.discard)
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
            "node_id": node_id,
        }

    @app.post("/slots/{slot_id}/{model_alias}/reset")
    async def slot_reset(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        run: str | None = None,
        start: bool = True,
    ) -> dict[str, str]:
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        ver = versions.for_run(run)
        await _cancel_task(run, slot_id, model_alias)
        # Cancel any standalone retries (registered on this version's
        # generation._pending but with no owning _run task to drive cleanup)
        # that the running task wouldn't have touched.
        ver.generation.cancel_pending(_run_id(run, slot.id, model_alias))
        slot_dir = _slot_dir(run, slot.id, model_alias)
        shutil.rmtree(slot_dir, ignore_errors=True)
        slot_dir.mkdir(parents=True, exist_ok=True)
        old_log = _slot_logs.get((run, slot.id, model_alias))
        if old_log is not None:
            old_log.close()
        slot_log = SlotLog(_run_id(run, slot.id, model_alias), slot_dir / "events.jsonl")
        _slot_logs[(run, slot.id, model_alias)] = slot_log
        if start:
            slot_log.start_run(slot.prompt, MODELS[model_alias])
            _tasks[(run, slot.id, model_alias)] = asyncio.create_task(_run(run, slot.id, model_alias))
        else:
            # Wipe back to a fresh idle cell — prompt + model pre-seeded (so a
            # later /resume can start_run without the client resending them) but
            # no pipeline task launched. Lets "reset all" clear an A/B matrix
            # without spawning dozens of runs; the user then starts exactly the
            # (version, slot, model) cells they pick.
            slot_log.state["prompt"] = slot.prompt
            slot_log.state["model"] = MODELS[model_alias]
            slot_log.state["status"] = "idle"
        return {"run": run, "slot_id": slot.id, "model": model_alias}

    @app.get("/versions")
    async def list_versions(  # pyright: ignore[reportUnusedFunction]
        slot: str | None = None,
        model: str | None = None,
    ) -> dict[str, object]:
        """The pipeline versions for the version bar. When `slot` and
        `model` are given, each entry carries that cell's status (for the
        per-version status dots); the runs are seeded at boot so their cells
        always exist."""
        items: list[dict[str, object]] = []
        for ver in versions.VERSIONS:
            slot_log = None
            if slot is not None and model is not None:
                slot_log = _slot_logs.get((ver.run_name, slot, model))
            status = slot_log.state.get("status") if slot_log is not None else None
            items.append(
                {
                    "id": ver.id,
                    "run_name": ver.run_name,
                    "label": ver.label,
                    "status": status,
                }
            )
        return {"versions": items, "current": _current_run}

    @app.post("/versions/{slot_id}/{model_alias}/launch")
    async def launch_versions(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
    ) -> dict[str, object]:
        """Start every pipeline version on one (slot, model) cell so they
        run concurrently and fully isolated. Independent of `_current_run`;
        each version is its own reserved run and keeps running regardless of
        which one the viewer is currently showing. A version whose cell is
        already complete is left untouched."""
        _require_slot(slot_id)
        _require_model(model_alias)
        results: list[dict[str, object]] = []
        for ver in versions.VERSIONS:
            run = ver.run_name
            try:
                _hydrate_run(run)
                slot_log = _require_slot_log(run, slot_id, model_alias)
                if any(e.get("kind") == "run.done" for e in slot_log.state["events"]):
                    results.append(
                        {"id": ver.id, "run_name": run, "status": "done", "started": False}
                    )
                    continue
                await _start_cell(run, slot_id, model_alias)
                results.append(
                    {"id": ver.id, "run_name": run, "status": "running", "started": True}
                )
            except Exception as e:
                results.append(
                    {
                        "id": ver.id,
                        "run_name": run,
                        "status": "error",
                        "started": False,
                        "error": str(e),
                    }
                )
        return {"slot_id": slot_id, "model": model_alias, "versions": results}

    @app.post("/versions/snapshot")
    async def snapshot_versions() -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Archive every reserved version run (V1/V2/V3/V4) that has data into a
        timestamped, loadable copy, so the live version cells can be reset and
        re-run fresh without losing the current rendition. The
        originals are untouched and stay active; each archive shows up in the
        run picker like any other run and is self-contained (meshes stream
        from its own dir). Versions the user never launched are skipped. A
        shared timestamp keeps the batch grouped in the picker."""
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        snapshots: list[dict[str, str]] = []
        skipped: list[str] = []
        for ver in versions.VERSIONS:
            if not _run_has_data(ver.run_name):
                skipped.append(ver.run_name)
                continue
            name = f"{ver.run_name}@{ts}"
            dst = _run_dir(name)
            if dst.exists():
                skipped.append(ver.run_name)
                continue
            # copytree of a cell's objects/ can be hundreds of MB — run it off
            # the event loop so SSE streams and status polls don't stall.
            await asyncio.to_thread(shutil.copytree, _run_dir(ver.run_name), dst)
            snapshots.append({"run_name": ver.run_name, "snapshot": name})
        if not snapshots:
            raise HTTPException(status_code=404, detail="no version data to archive yet")
        return {"snapshots": snapshots, "skipped": skipped}

    return app


def _slot_summary(slot: Slot, run: str) -> dict[str, object]:
    runs: dict[str, dict[str, object]] = {}
    for alias in MODEL_ALIASES:
        slot_log = _slot_logs.get((run, slot.id, alias))
        state = slot_log.state if slot_log is not None else {"status": "idle", "events": []}
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
    # hydrate_from_disk already derived the status from the full event log
    # (done/error/paused are sticky terminal states; a log with no terminal
    # marker reads as "running"). The only boot-time adjustment is that a
    # process killed mid-run leaves a "running" log with no sentinel —
    # surface that as paused so the user can resume it. A completed run stays
    # "done" (resume blocked, reset only) even when a post-run mesh retry
    # appended events after run.done; an errored run stays "error" (retry).
    if slot_log.state["status"] == "running":
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
    raw_prompt = image_event.get("prompt") if image_event is not None else bbox_event.get("prompt")
    raw_str = str(raw_prompt) if raw_prompt is not None else ""
    if raw_str.lstrip().startswith(_WRAPPED_PROMPT_PREFIX):
        image_prompt = raw_str
        subject_str = raw_str
    else:
        subject_str = raw_str
        p = prompt_runtime.current()
        image_prompt = p.wrap_image_prompt(subject_str, proxy_shape, bbox.size)
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


async def _start_cell(run: str, slot_id: str, model_alias: str) -> None:
    """Cancel any in-flight task for this cell, then (re)start its pipeline.
    A fresh cell emits run.start; a paused/errored cell drops its terminal
    sentinel and resumes. Shared by slot_resume and the version launcher —
    callers own any precondition guards (e.g. blocking a completed run)."""
    slot = _require_slot(slot_id)
    slot_log = _require_slot_log(run, slot_id, model_alias)
    status = slot_log.state.get("status")
    await _cancel_task(run, slot_id, model_alias)
    events = slot_log.state["events"]
    model_id = MODELS[model_alias]
    if status == "idle" and not events:
        slot_log.start_run(slot.prompt, model_id)
    else:
        if events and events[-1].get("kind") in ("run.error", "run.paused"):
            slot_log.truncate_events_to(len(events) - 1)
        slot_log.state["model"] = model_id
        slot_log.state["status"] = "running"
    _tasks[(run, slot_id, model_alias)] = asyncio.create_task(_run(run, slot_id, model_alias))


_MESH_BUNDLE_MAGIC = b"SMB1"


async def _mesh_bundle(objects_dir: Path) -> AsyncIterator[bytes]:
    """Stream every finished GLB under `objects_dir` as one length-prefixed
    binary bundle, so a client can fetch an entire scene in a single request
    instead of one HTTP round-trip per mesh. Framing (little-endian):

        b"SMB1"
        repeat: <uint32 id_len><id utf-8><uint32 glb_len><glb bytes>

    The id is the file stem, which matches the `model` event's `id`. Files are
    read off the event loop via a thread so a large scene doesn't stall it.
    """
    yield _MESH_BUNDLE_MAGIC
    if not objects_dir.is_dir():
        return
    for path in sorted(objects_dir.glob("*.glb")):
        # Skip pre-processing intermediates (`<id>.raw.glb`); only the finished
        # `<id>.glb` is what the client renders.
        if path.name.endswith(".raw.glb"):
            continue
        node_id = path.name[: -len(".glb")]
        data = await asyncio.to_thread(path.read_bytes)
        id_bytes = node_id.encode("utf-8")
        yield struct.pack("<I", len(id_bytes)) + id_bytes
        yield struct.pack("<I", len(data)) + data


def _scene_projection(events: list[dict[str, object]]) -> dict[str, object]:
    """Fold the event log into the minimal renderable scene state — the read
    model the client paints directly instead of replaying every event.

    Mirrors the *scene-building* cases of the client's `dispatch()` (bbox,
    divider.*, step, mesh.*, image, model). Deliberately ignores `cache.llm`
    and the other log/observability-only kinds — those stay in the full event
    log, which the client backfills into the side panels separately.

    Returns `{ "nodes": [...], "last_index": N }` where N is the highest event
    index folded, so the client can subscribe to the live tail at index > N.
    """
    nodes: dict[str, dict[str, object]] = {}
    last_index = -1

    def node(node_id: str) -> dict[str, object]:
        n = nodes.get(node_id)
        if n is None:
            n = {"id": node_id}
            nodes[node_id] = n
        return n

    for e in events:
        idx = e.get("index")
        if isinstance(idx, int):
            last_index = max(last_index, idx)
        kind = e.get("kind")
        nid = e.get("id")
        if kind == "bbox" and isinstance(nid, str):
            n = node(nid)
            n["parent_id"] = e.get("parent_id")
            n["prompt"] = e.get("prompt")
            n["node_kind"] = e.get("node_kind", "zone")
            n["origin"] = e.get("origin")
            n["dimensions"] = e.get("dimensions")
            n["proxy_shape"] = e.get("proxy_shape")
        elif kind in ("divider.decompose", "divider.zone_decompose"):
            children = e.get("children")
            if isinstance(children, list):
                for c in children:
                    if not isinstance(c, dict):
                        continue
                    cid = c.get("id")
                    if not isinstance(cid, str):
                        continue
                    n = node(cid)
                    n.setdefault("parent_id", c.get("parent") or e.get("node"))
                    n.setdefault("prompt", c.get("prompt"))
                    n.setdefault("node_kind", "zone")
        elif kind == "divider.zone_plan":
            anchor = e.get("node")
            if isinstance(anchor, str) and isinstance(e.get("plan"), str):
                node(anchor)["plan"] = e["plan"]
        elif kind == "step":
            anchor = e.get("node")
            if isinstance(anchor, str):
                node(anchor)["phase"] = e.get("phase")
        elif kind == "mesh.submit" and isinstance(nid, str):
            node(nid)["phase"] = "generating_mesh"
        elif kind == "image" and isinstance(nid, str):
            n = node(nid)
            n["image_url"] = e.get("url")
            if isinstance(e.get("prompt"), str):
                n["image_prompt"] = e.get("prompt")
        elif kind == "model" and isinstance(nid, str):
            n = node(nid)
            n["mesh_url"] = e.get("url")
            n["phase"] = "done"
            n.pop("error", None)
        elif kind == "mesh.error" and isinstance(nid, str):
            n = node(nid)
            n["phase"] = "error"
            n["error"] = e.get("message", "unknown error")
        elif kind == "mesh.retry" and isinstance(nid, str):
            n = node(nid)
            n["phase"] = "generating_mesh"
            n.pop("error", None)

    return {"nodes": list(nodes.values()), "last_index": last_index}


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
    ver = versions.for_run(run)
    prompt_runtime.bind(ver.prompt_module or _prompt_module_for_run(run))
    prompt = slot_log.state["prompt"]
    model = slot_log.state["model"]
    run_id = _run_id(run, slot_id, model_alias)
    try:
        await ver.run(
            run_id=run_id,
            prompt=prompt,
            model=model,
            runs_dir=RUNS_DIR,
        )
    except asyncio.CancelledError:
        ver.generation.cancel_pending(run_id)
        raise
    except Exception as e:
        ver.generation.cancel_pending(run_id)
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
    await ver.generation.await_pending(run_id)
    slot_log.finish_run()
