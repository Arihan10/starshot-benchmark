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
import logging
import os
import shutil
import struct
import tempfile
from datetime import datetime
from urllib.parse import quote
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from app.core import prompt_runtime
from app.core.prompts import ImageView
from app.core.slots import (
    DEFAULT_MODEL_ALIAS,
    MODEL_ALIASES,
    MODELS,
    SLOTS,
    SLOTS_BY_ID,
    Slot,
)
from app.core.types import BoundingBox, Node, Orientation, ProxyShape
from app.pipeline import generation, versions
from app.services import anchors as anchors_svc
from app.services import llm, prefabs, proxy as proxy_svc, publish as publish_svc, threed
from app.services import symmetry as sym_svc
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

# Best-effort auto-publish: once a tour is finalized, push the cell's preview +
# tour to R2 and upsert the D1 catalog in the background so the prod client sees
# the new scene. Detached from the request, and failures (missing R2/D1 creds,
# network) are only logged — they never block tour capture. A live task set
# keeps references so the GC can't drop a publish mid-flight.
_autopublish_log = logging.getLogger("starshot.autopublish")
_autopublish_tasks: set[asyncio.Task[None]] = set()


async def _auto_publish_cell(run: str, slot: str, model: str) -> None:
    try:
        rec = await publish_svc.publish_cell(RUNS_DIR, run, slot, model)
        _autopublish_log.info("published %s/%s/%s v%s", run, slot, model, rec["version"])
    except Exception as e:
        _autopublish_log.warning(
            "publish failed for %s/%s/%s: %s: %s", run, slot, model, type(e).__name__, e
        )


def _schedule_auto_publish(run: str, slot: str, model: str) -> None:
    task = asyncio.create_task(_auto_publish_cell(run, slot, model))
    _autopublish_tasks.add(task)
    task.add_done_callback(_autopublish_tasks.discard)

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
# Generated-asset builds are scoped to ONE version of a cell, so their task
# tables key on (run, slot, model, version) — distinct versions build / regen
# concurrently and fully isolated (own dirs + log).
GenKey = tuple[str, str, str, str]
_slot_logs: dict[RunKey, SlotLog] = {}
_tasks: dict[RunKey, asyncio.Task[None]] = {}
_retry_tasks: set[asyncio.Task[None]] = set()
# In-flight "generate from-scratch assets" task per (cell, version). Drives the
# client's generate gate (a version can't re-trigger until its current build
# finishes) and lets shutdown / reset cancel a build cleanly.
_generate_tasks: dict[GenKey, asyncio.Task[None]] = {}
# Per-(cell, version) regeneration worker + its FIFO of
# (node_id, propagate, backend, op, reuse_image) requests, where op is
# "regenerate" (fresh mesh — and, unless reuse_image, a fresh Nano-Banana image
# too) or "unsymmetrize" (reprocess the existing raw mesh with the symmetry
# mirror off — no AI; reuse_image is ignored). `_regen_tasks` holds the single worker
# task per version; `_regen_queues` is the work it drains concurrently (parallel
# across prefab groups, serialized within one), sharing that version's
# generated-events log. Requests enqueue rather than 409 between each other, and
# run concurrently with a whole-scene generate of the same version: they serialize
# per-node with it via generation.node_lock, so neither writes the same asset twice.
_regen_tasks: dict[GenKey, asyncio.Task[None]] = {}
_regen_queues: dict[GenKey, asyncio.Queue[tuple[str, bool, str, str, bool]]] = {}
_hydrated_runs: set[str] = set()
_current_run: str = ""

# Prompt-tuning BRANCHES. A branch is an ephemeral "what-if" fork of one cell:
# the original events up to a tuned step, with that step's output swapped for a
# hand-tested one, then the pipeline resumed so every downstream step re-runs
# against the changed state. Each branch is fully isolated from its source —
# its own SlotLog + events.jsonl + objects dir under `<cell>/_branch/`, and its
# own composite run_id (`run/slot/model/_branch`) so the LLM cache, the
# `committed.*` resume reader, `generation._pending`, and the Trellis queue all
# key off the branch, never the source.
#
# At most ONE branch per run (the sandbox is a single modal; there is no
# recursive "branch a branch"): a branch always forks the original cell, and
# creating one discards any prior branch in that run. So these map RUN ->
# branch; the branch's own composite run_id (stored as the SlotLog's slot_id)
# remembers which cell it forked. Wiped on break-out (DELETE) and on any
# source-cell mutation (reset/rewind).
BRANCH_SUBDIR = "_branch"
_branch_logs: dict[str, SlotLog] = {}
_branch_tasks: dict[str, asyncio.Task[None]] = {}
# Per-run step controller driving the branch's interactive step-through: it
# pauses the pipeline before each downstream LLM call (via `llm`'s step gate)
# and the `/branch/step` endpoint resolves each pause with the edited prompt.
_branch_controllers: dict[str, "BranchStepController"] = {}
# A one-shot prompt to auto-run the FIRST paused step with (no pause), set by
# `/branch/rerun` so re-running a committed step replays it with the edited
# prompt instead of stopping on it again.
_branch_reseed: dict[str, dict[str, object]] = {}


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


class BranchRequest(BaseModel):
    """Fork the cell so it re-simulates from `deviation_index` onward —
    which MUST point at a `cache.llm` event. The branch keeps the events
    BEFORE that step and resumes, so the step-through pauses on that step
    first (its re-rendered prompt is editable) and then every step after it.
    Always forks the original cell; creating a branch replaces any prior
    branch in the run."""

    deviation_index: int


class BranchStepRequest(BaseModel):
    """Advance the branch past the step it's paused on. `system`/`user` are the
    (possibly hand-edited) prompts to actually run; `auto` lets the rest of the
    branch run without further pauses."""

    system: str | None = None
    user: str | None = None
    auto: bool = False


class BranchRerunRequest(BaseModel):
    """Re-run an already-committed branch step (identified by its `cache.llm`
    event index) with the given prompt, INVALIDATING every step after it. The
    step replays with the edit (no pause), then the branch pauses on the next
    step."""

    llm_index: int
    system: str | None = None
    user: str | None = None


def _run_id(run: str, slot_id: str, model_alias: str) -> str:
    """Composite id used as `run_id` in pipeline code (divider, generation,
    threed queue, SlotLog.slot_id). The slashes make it work as a filesystem
    subpath under RUNS_DIR and as an artifact URL segment under /artifacts."""
    return f"{run}/{slot_id}/{model_alias}"


def _run_dir(run: str) -> Path:
    return RUNS_DIR / run


def _slot_dir(run: str, slot_id: str, model_alias: str) -> Path:
    return RUNS_DIR / run / slot_id / model_alias


def _gen_slot_id(run: str, slot_id: str, model_alias: str, version: str) -> str:
    """SlotLog id for one generated version — used as the bound log's slot_id
    and the Trellis queue key, so distinct versions never collide on either."""
    return f"{_run_id(run, slot_id, model_alias)}::generated::{version}"


def _safe_version(version: str | None) -> str | None:
    """Validate a client-supplied generated version id (a positive integer) so
    it stays a safe path segment, normalizing leading zeros. None passes through
    (the caller then falls back to latest / a freshly allocated id)."""
    if version is None:
        return None
    v = version.strip()
    if not v.isdigit() or int(v) < 1:
        raise HTTPException(status_code=400, detail=f"invalid version: {version}")
    return str(int(v))


def _read_gen_version(
    run: str, slot_id: str, model_alias: str, version: str | None,
) -> str | None:
    """Resolve a READ's target generated version: the requested one (validated),
    else the latest existing, else None when the cell has no versions yet."""
    v = _safe_version(version)
    if v is not None:
        return v
    existing = generation.list_generated_versions(RUNS_DIR, _run_id(run, slot_id, model_alias))
    return existing[-1] if existing else None


def _write_gen_version(
    run: str, slot_id: str, model_alias: str, version: str | None, new: bool,
) -> str:
    """Resolve a WRITE's target generated version: a freshly allocated id when
    `new`, else the requested one (created on demand), else the latest existing,
    else "1" for a cell's first build."""
    rid = _run_id(run, slot_id, model_alias)
    if new:
        return generation.next_generated_version(RUNS_DIR, rid)
    v = _safe_version(version)
    if v is not None:
        return v
    existing = generation.list_generated_versions(RUNS_DIR, rid)
    return existing[-1] if existing else "1"


# Parsed symmetry state per generated-events log, cached on the file's
# (mtime_ns, size) so the frequently-polled gate status re-reads it only when a
# build / regen / un-symmetrize has appended to that version's log.
_gen_symmetry_cache: dict[
    Path, tuple[tuple[int, int], dict[str, dict[str, str | None]]]
] = {}


def _generated_symmetry(events_path: Path) -> dict[str, dict[str, str | None]]:
    """Map node id -> {'plane': current symmetry plane, 'was': prior mirror plane}.

    'plane' is 'xy'/'xz' when the served mesh is mirrored across that plane, else
    'none'. 'was' separates the two un-mirrored cases the client must tell apart:
    a node un-symmetrized after being mirrored ('plane'='none', 'was'= its old
    plane) vs one that was never symmetrized ('plane'='none', 'was'=None).

    A node's symmetry follows its PREFAB CANONICAL: a reuse's mesh is always
    re-derived from the canonical's raw at the canonical's current plane, and a
    propagated un-symmetrize records `symmetry.applied`='none' only on the
    canonical (the per-reuse re-derivation logs nothing), so the canonical's
    applied history is the source of truth for the whole group. Ids absent here
    resolve to a canonical that was never mirrored, so callers default to never.

    Handles both event formats: the current `symmetry.applied` event AND the legacy
    `symmetry.decision` that bundled `applied`+`axis` onto the decision itself
    (older builds, before applied was split out). Both update the same per-id state
    in log order, so a node recorded only the legacy way is still detected and a
    later un-symmetrize still wins."""
    try:
        st = events_path.stat()
    except OSError:
        return {}
    sig = (st.st_mtime_ns, st.st_size)
    cached = _gen_symmetry_cache.get(events_path)
    if cached is not None and cached[0] == sig:
        return cached[1]
    # Per-id applied history (latest plane + last mirrored plane, with the log
    # position of that latest update) and the flat prefab star (id -> the canonical
    # it reuses, latest match winning).
    applied: dict[str, dict[str, str | None]] = {}
    applied_idx: dict[str, int] = {}
    reuse_of: dict[str, str] = {}

    def _set_plane(nid: str, cut_plane: str, idx: int) -> None:
        entry = applied.setdefault(nid, {"plane": "none", "was": None})
        entry["plane"] = cut_plane
        if cut_plane in ("xy", "xz"):
            entry["was"] = cut_plane
        applied_idx[nid] = idx

    with events_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("kind")
            node_id = event.get("id")
            if not isinstance(node_id, str):
                continue
            if kind == "symmetry.applied":
                cut_plane = event.get("cut_plane")
                if cut_plane in ("none", "xy", "xz"):
                    _set_plane(node_id, cut_plane, idx)
            elif kind == "symmetry.decision" and "applied" in event:
                # Legacy combined event: older builds recorded the plane AND whether
                # it was mirrored on the decision itself (current builds split that
                # into a separate `symmetry.applied`). Mirrored iff `applied` is
                # truthy and the plane is xy/xz; anything else (applied false, or
                # plane none) means the served mesh is not mirrored.
                cut_plane = event.get("cut_plane")
                _set_plane(
                    node_id,
                    cut_plane if (event.get("applied") and cut_plane in ("xy", "xz")) else "none",
                    idx,
                )
            elif kind == "prefab.match":
                reuse_of[node_id] = str(event.get("reuse_id") or "")
    # Resolve each node to whichever of {its own, its canonical's} applied history
    # is MORE RECENT in log order. The canonical matters because a propagated
    # un-symmetrize records applied=none only on the canonical (the per-reuse
    # re-derivation logs nothing), so its later 'none' must override a reuse's
    # now-stale mirror. But a reuse (re)built more recently keeps its own state —
    # including legacy logs, where each reuse carried its OWN symmetry event, and
    # an independently regenerated reuse. Ids with neither default to never.
    state: dict[str, dict[str, str | None]] = {}
    for node_id in set(applied) | set(reuse_of):
        canonical = reuse_of.get(node_id) or node_id
        own_idx = applied_idx.get(node_id, -1)
        canon_idx = applied_idx.get(canonical, -1) if canonical != node_id else -1
        info = applied.get(node_id if own_idx >= canon_idx else canonical)
        state[node_id] = (
            {"plane": info["plane"], "was": info["was"]}
            if info is not None
            else {"plane": "none", "was": None}
        )
    _gen_symmetry_cache[events_path] = (sig, state)
    return state


def _ensure_run_hydrated(run: str) -> None:
    """Lazily hydrate a run that exists on disk but isn't in memory yet.

    Only the three reserved version runs hydrate at boot; saved/named runs
    hydrate on explicit activation. But the client also reaches a run directly
    via `?run=` — a persisted tab, an archived run in the picker, or any cell
    after a server restart — without re-activating it. Without this, every cell
    endpoint for such a run 404s (`_require_slot_log`) and `/slots` reports its
    cells as empty/idle, so the client's gate poll silently bails and a finished
    regen/build never swaps in. Hydrating on first touch makes cell access
    self-healing; it's idempotent and cheap once done (guarded by
    `_hydrated_runs`), and a genuinely unknown run (no dir) is left alone so the
    downstream 404 still fires."""
    if run and run not in _hydrated_runs and _run_dir(run).is_dir():
        _hydrate_run(run)


def _branch_dir(run: str, slot_id: str, model_alias: str) -> Path:
    return _slot_dir(run, slot_id, model_alias) / BRANCH_SUBDIR


def _branch_run_id(run: str, slot_id: str, model_alias: str) -> str:
    """Composite run_id for a branch — the source cell's run_id plus the
    `_branch` segment, so meshes land in `<cell>/_branch/objects/` and every
    run_id-keyed table (`generation._pending`, the Trellis queue) is isolated
    from the source."""
    return f"{_run_id(run, slot_id, model_alias)}/{BRANCH_SUBDIR}"


def _resolve_run(run: str | None) -> str:
    """Every cell endpoint names its target run/version explicitly so the
    concurrently-running versions never route through a shared global.
    The client always sends `?run=`; `_current_run` is only the fallback for a
    client that hasn't picked one yet (boot, or a legacy caller). Resolving also
    lazily hydrates the target run if it exists on disk but isn't yet in memory,
    so a cell is never spuriously 404/idle just because its run wasn't the one
    explicitly activated."""
    resolved = run or _current_run
    _ensure_run_hydrated(resolved)
    return resolved


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
            # One-time fold of any pre-versioning generated build into
            # generated/1/ so existing artifacts render under the new layout.
            generation.migrate_legacy_generated(RUNS_DIR, _run_id(run, slot.id, alias))
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
            for task in _generate_tasks.values():
                task.cancel()
            for task in _regen_tasks.values():
                task.cancel()
            for task in _branch_tasks.values():
                task.cancel()
            for task in list(_tasks.values()):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for task in list(_retry_tasks):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for task in list(_generate_tasks.values()):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for task in list(_regen_tasks.values()):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for task in list(_generate_tasks.values()):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for task in list(_regen_tasks.values()):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for task in list(_branch_tasks.values()):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for branch_log in _branch_logs.values():
                branch_log.close()
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

    @app.post("/proxy")
    async def build_scene_proxy(request: Request) -> Response:  # pyright: ignore[reportUnusedFunction]
        # Decimate a merged, world-space scene GLB (the viewer bakes one from its
        # placed meshes) into a geometry-only low-poly proxy for the /pano
        # walkthrough's projection mode. Stateless: bytes in, smaller bytes out.
        body = await request.body()
        if not body:
            raise HTTPException(400, "empty request body (expected a binary GLB)")
        with tempfile.TemporaryDirectory(prefix="proxy-") as td:
            src = Path(td) / "scene.glb"
            dst = Path(td) / "proxy.glb"
            src.write_bytes(body)
            try:
                await proxy_svc.build_proxy(src, dst)
            except Exception as e:  # surface decimation failures to the client as 502
                raise HTTPException(502, f"{type(e).__name__}: {e}") from e
            data = dst.read_bytes()
        return Response(content=data, media_type="model/gltf-binary")

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
            _validated, reasoning, usage, raw = await llm.call_llm_once(
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
            "output": raw,
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
        """Live snapshot of the in-flight mesh queue, split into independent
        concurrency pools (Trellis/Modal vs Hunyuan 3.1/Tencent). The client
        polls this instead of inferring queue state from the replayed event log
        (historical submits without matching .done events leak as stale
        "processing" rows otherwise). Each entry is tagged with its `backend` +
        `pool`; `pools` carries each section's label and its own cap."""
        return {
            "pools": threed.queue_pools(),
            "entries": threed.queue_snapshot(),
        }

    @app.post("/generations/stop")
    async def stop_all_generations() -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Preemptively terminate every in-flight generation — process-wide,
        across all runs, versions, and (slot, model) cells — without wiping any
        state, so each can be resumed/retried afterward:

          * Pipeline builds (`_tasks`) are cancelled, then their cell is flipped
            to `paused` + `run.paused` (identical to a per-cell pause), so the
            dashboard shows it as resumable and `/resume` re-enters the pipeline,
            short-circuiting everything already on disk.
          * From-scratch builds (`_generate_tasks`) are cancelled; re-pressing
            generate resumes them (finished optimized twins are skipped).
          * Standalone mesh retries (`_retry_tasks`) are cancelled.

        Underlying Trellis/Banana mesh tasks live on each version's
        `generation._pending`; a cancelled retry's wrapper has already returned
        and no longer owns its mesh, so we drain `_pending` for every hydrated
        cell to be sure none leak. On-disk artifacts + event logs are untouched
        — only in-memory tasks die, which is what makes this resumable rather
        than destructive."""
        pipeline_keys = [k for k, t in _tasks.items() if not t.done()]
        generate_keys = [k for k, t in _generate_tasks.items() if not t.done()]
        regen_keys = [k for k, t in _regen_tasks.items() if not t.done()]
        retry_tasks = [t for t in _retry_tasks if not t.done()]

        # Fire every cancellation before awaiting any, so in-flight network
        # awaits unwind concurrently instead of one cancellation at a time.
        pipeline_tasks = [_tasks.pop(k) for k in pipeline_keys]
        generate_tasks = [_generate_tasks.pop(k) for k in generate_keys]
        regen_tasks = [_regen_tasks.pop(k) for k in regen_keys]
        for task in (*pipeline_tasks, *generate_tasks, *regen_tasks, *retry_tasks):
            task.cancel()

        # Drain the per-run mesh-task tables on every hydrated cell, dispatched
        # to each run's generation module (v1 → generation_old, v2/v3 → generation).
        for run_name, slot_id, model_alias in list(_slot_logs.keys()):
            versions.for_run(run_name).generation.cancel_pending(
                _run_id(run_name, slot_id, model_alias),
            )

        for task in (*pipeline_tasks, *generate_tasks, *regen_tasks, *retry_tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        # Flip each stopped pipeline cell to paused (resumable) + sentinel —
        # only while still "running", so a status the task wrote just before
        # cancellation landed (run.done / run.error) is never clobbered.
        paused: list[str] = []
        for key in pipeline_keys:
            slot_log = _slot_logs.get(key)
            if slot_log is None or slot_log.state.get("status") != "running":
                continue
            slot_log.state["status"] = "paused"
            slot_log.log("run.paused")
            paused.append(_run_id(*key))

        return {
            "stopped_pipelines": [_run_id(*k) for k in pipeline_keys],
            "stopped_generates": [_gen_slot_id(*k) for k in generate_keys],
            "stopped_regens": [_gen_slot_id(*k) for k in regen_keys],
            "stopped_retries": len(retry_tasks),
            "paused": paused,
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
    async def slot_meshes(slot_id: str, model_alias: str, run: str | None = None, mode: str | None = None, version: str | None = None, optimized: bool = True) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        # One-request scene bundle: stream every finished GLB for this cell in
        # a single length-prefixed response so the client loads a whole scene
        # over one connection instead of one HTTP request per mesh (browsers
        # cap those at ~6 per origin, and they'd contend with the SSE stream
        # and polls on this same origin). The client keys each blob to its
        # `model` event by file stem; see `_mesh_bundle`.
        #
        # `mode=generated` streams one VERSION of the from-scratch build instead
        # of the library `objects/`; `version` picks which (latest when omitted).
        # The asset toggle + version picker are the only things the client flips.
        # The library and every generated version coexist under the same cell dir.
        # Generated meshes are served from their OPTIMIZED twin (decimated +
        # KTX2 + Meshopt) by default; `optimized=0` streams the raw, bbox-fitted
        # Trellis mesh (objects-generated/<id>.glb) instead — heavy, but the
        # client's optimized toggle uses it for side-by-side comparison.
        run = _resolve_run(run)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        cell_dir = slot_log.events_path.parent
        if mode == "generated":
            rid = _run_id(run, slot_id, model_alias)
            resolved = _read_gen_version(run, slot_id, model_alias, version)
            # No version yet → point at the (glb-less) version root so the
            # bundle streams empty rather than 404ing. generated_dirs() returns
            # (raw, optimized); index by the toggle. _mesh_bundle skips the
            # `<id>.raw.glb` intermediates in the raw dir, so either folder
            # streams exactly the finished `<id>.glb` set.
            objects_dir = (
                generation.generated_dirs(RUNS_DIR, rid, resolved)[1 if optimized else 0]
                if resolved is not None
                else generation.generated_version_root(RUNS_DIR, rid)
            )
        else:
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

    # --- capture-anchor planning + tour persistence --------------------------
    #
    # The "other side" of the pipeline: a lightweight model reads this cell's
    # scene hierarchy and proposes where to stand for 360 captures; the client
    # then renders a pano at each, builds the proxy, and uploads the whole tour
    # back here so it persists under /artifacts/<cell>/tour/ for the walkthrough
    # viewer (and the website) to load by URL — no manual file handling.

    def _tour_dir(slot_log: SlotLog) -> Path:
        return slot_log.events_path.parent / "tour"

    @app.post("/slots/{slot_id}/{model_alias}/anchors")
    async def slot_anchors(slot_id: str, model_alias: str, run: str | None = None) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        # Plan 360 capture anchors for this cell's scene with the fixed planner
        # model. Read-only analysis: reconstructs the Node tree from the event
        # log and renders it in the pipeline's own scene-context format.
        run = _resolve_run(run)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        nodes = _nodes_from_events(list(slot_log.state["events"]))
        if not nodes:
            raise HTTPException(400, "scene has no placed nodes to plan anchors from")
        try:
            plan, reasoning = await anchors_svc.generate_anchors(nodes)
        except Exception as e:  # surface provider failures as 502
            raise HTTPException(502, f"{type(e).__name__}: {e}") from e
        return {
            "anchors": [a.model_dump() for a in plan.anchors],
            "reasoning": reasoning,
            "model": anchors_svc.ANCHOR_PLANNER_MODEL,
        }

    @app.post("/slots/{slot_id}/{model_alias}/tour/reset")
    async def tour_reset(slot_id: str, model_alias: str, run: str | None = None) -> dict[str, bool]:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        tour_dir = _tour_dir(slot_log)
        shutil.rmtree(tour_dir, ignore_errors=True)
        tour_dir.mkdir(parents=True, exist_ok=True)
        return {"ok": True}

    @app.put("/slots/{slot_id}/{model_alias}/tour/pano/{pano_id}")
    async def tour_pano(slot_id: str, model_alias: str, pano_id: str, request: Request, run: str | None = None) -> dict[str, bool]:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        body = await request.body()
        if not body:
            raise HTTPException(400, "empty pano body")
        tour_dir = _tour_dir(slot_log)
        tour_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(pano_id).name  # defend against path traversal
        (tour_dir / f"{stem}.jpg").write_bytes(body)
        return {"ok": True}

    @app.post("/slots/{slot_id}/{model_alias}/tour/proxy")
    async def tour_proxy(slot_id: str, model_alias: str, request: Request, run: str | None = None) -> dict[str, bool]:  # pyright: ignore[reportUnusedFunction]
        # Body is the client's merged, world-space scene GLB; decimate it into the
        # stored proxy.glb via the same pass the download flow uses.
        run = _resolve_run(run)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        body = await request.body()
        if not body:
            raise HTTPException(400, "empty scene body")
        tour_dir = _tour_dir(slot_log)
        tour_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tourproxy-") as td:
            src = Path(td) / "scene.glb"
            src.write_bytes(body)
            try:
                await proxy_svc.build_proxy(src, tour_dir / "proxy.glb")
            except Exception as e:
                raise HTTPException(502, f"{type(e).__name__}: {e}") from e
        return {"ok": True}

    @app.post("/slots/{slot_id}/{model_alias}/tour/manifest")
    async def tour_manifest(slot_id: str, model_alias: str, request: Request, run: str | None = None) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        body = await request.body()
        try:
            json.loads(body)
        except Exception as e:
            raise HTTPException(400, f"manifest is not valid JSON: {e}")
        tour_dir = _tour_dir(slot_log)
        tour_dir.mkdir(parents=True, exist_ok=True)
        (tour_dir / "tour.json").write_bytes(body)
        # A finalized tour = a publishable scene; push it to R2 + D1 in the
        # background (best-effort) so the prod client's catalog picks it up.
        _schedule_auto_publish(run, slot_id, model_alias)
        tour_url = (
            f"/artifacts/{quote(run)}/{quote(slot_id)}/{quote(model_alias)}/tour/tour.json"
        )
        return {"ok": True, "tour_url": tour_url}

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
        # A rewind rewrites the source log; the run's branch is now stale.
        await _discard_branch(run)
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

    @app.post("/slots/{slot_id}/{model_alias}/generate")
    async def slot_generate(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        run: str | None = None,
        version: str | None = None,
        new: bool = False,
    ) -> dict[str, object]:
        """The generate gate: build from-scratch (Nano-Banana + Trellis) assets
        for this cell into ONE generated `version` (`generated/<version>/`),
        reusing the library build's layout — every object's existing
        bbox/orientation — so the client's "generated" view is an apples-to-apples
        swap of matched assets for freshly generated ones.

        `new=true` allocates a brand-new version (a fresh from-scratch take on the
        same scene); otherwise the targeted/latest version is resumed — meshes
        already on disk for THAT version are skipped, and bookkeeping lands in its
        own events.generated.jsonl. Any number of versions coexist; each builds
        independently. Only one build per (cell, version) at a time; re-pressing
        the same version while it's in flight returns 409 (the gate). The library
        build (objects/ + events.jsonl) is never touched."""
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        lib_log = _require_slot_log(run, slot_id, model_alias)
        gen_version = _write_gen_version(run, slot_id, model_alias, version, new)
        # Reserve the version dir synchronously so a rapid second `new=true`
        # allocates the next id instead of colliding before the build task
        # (which mkdirs) has run.
        generation.generated_dirs(RUNS_DIR, _run_id(run, slot.id, model_alias), gen_version)[0].mkdir(
            parents=True, exist_ok=True
        )
        key: GenKey = (run, slot_id, model_alias, gen_version)
        in_flight = _generate_tasks.get(key)
        if in_flight is not None and not in_flight.done():
            raise HTTPException(status_code=409, detail="generation already running")
        # Per-asset regenerations of this version may be in flight — that's allowed.
        # The scene build and regens serialize per-node via generation.node_lock, so
        # they never write the same asset at once.

        # Reconstruct every concrete (object/frame) node from the library log so
        # generation reuses the exact layout instead of re-running the divider.
        ver = versions.for_run(run)
        prompt_runtime.bind(ver.prompt_module or _prompt_module_for_run(run))
        nodes = _scene_nodes_from_library(lib_log)
        if not nodes:
            raise HTTPException(
                status_code=400,
                detail="no scene to generate from; build the library scene first",
            )

        # Dedicated per-version log for the build's resumable bookkeeping
        # (nano_banana / threed require a bound SlotLog). Kept apart from the
        # library log so the toggle stays a pure folder switch and the library
        # stream is untouched; one log per version isolates each version's resume.
        run_id = _run_id(run, slot.id, model_alias)
        gen_log = SlotLog(
            _gen_slot_id(run, slot.id, model_alias, gen_version),
            generation.generated_events_path(RUNS_DIR, run_id, gen_version),
        )
        gen_log.hydrate_from_disk()

        async def _do_generate() -> None:
            rlog.bind(gen_log)
            prompt_runtime.bind(ver.prompt_module or _prompt_module_for_run(run))
            try:
                await generation.generate_assets(
                    nodes=nodes, runs_dir=RUNS_DIR, run_id=run_id, version=gen_version,
                )
            finally:
                gen_log.close()

        _generate_tasks[key] = asyncio.create_task(_do_generate())
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
            "version": gen_version,
            "nodes": len(nodes),
        }

    @app.post("/slots/{slot_id}/{model_alias}/generate-from-images/{source_version}")
    async def slot_generate_from_images(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        source_version: str,
        run: str | None = None,
    ) -> dict[str, object]:
        """Fork a brand-new generated version that REUSES `source_version`'s
        Nano-Banana images and rebuilds every mesh FRESH on the scene backend
        (Trellis by default) — issuing no new Nano-Banana calls. Use it to re-roll
        a bad mesh pass while keeping a good image pass (e.g. v2's meshes are bad,
        so rebuild from v1's images into a new v3). The source version is untouched;
        the new version is allocated, seeded (see
        `generation.seed_generated_version_from`), built in the background, and
        returned for the client to select. Resilient like the normal gate: the
        per-version log records the reused images so a resume skips them too."""
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        lib_log = _require_slot_log(run, slot_id, model_alias)
        run_id = _run_id(run, slot.id, model_alias)
        src_version = _safe_version(source_version)
        if src_version is None or src_version not in generation.list_generated_versions(RUNS_DIR, run_id):
            raise HTTPException(
                status_code=404, detail=f"no such generated version: {source_version}",
            )

        ver = versions.for_run(run)
        prompt_runtime.bind(ver.prompt_module or _prompt_module_for_run(run))
        nodes = _scene_nodes_from_library(lib_log)
        if not nodes:
            raise HTTPException(
                status_code=400,
                detail="no scene to generate from; build the library scene first",
            )

        # Allocate + seed the new version. The whole reconstruct→allocate→seed path
        # is synchronous (no await), so two rapid clicks can't collide on the same
        # next id before the dir exists.
        target_version = generation.next_generated_version(RUNS_DIR, run_id)
        seeded = generation.seed_generated_version_from(
            RUNS_DIR, run_id, src_version, target_version,
        )
        if seeded == 0:
            shutil.rmtree(
                generation.generated_version_root(RUNS_DIR, run_id) / target_version,
                ignore_errors=True,
            )
            raise HTTPException(
                status_code=400,
                detail=f"version {src_version} has no Nano-Banana images to reuse",
            )

        key: GenKey = (run, slot_id, model_alias, target_version)
        gen_log = SlotLog(
            _gen_slot_id(run, slot.id, model_alias, target_version),
            generation.generated_events_path(RUNS_DIR, run_id, target_version),
        )
        gen_log.hydrate_from_disk()

        async def _do_generate() -> None:
            rlog.bind(gen_log)
            prompt_runtime.bind(ver.prompt_module or _prompt_module_for_run(run))
            try:
                await generation.generate_assets(
                    nodes=nodes, runs_dir=RUNS_DIR, run_id=run_id, version=target_version,
                )
            finally:
                gen_log.close()

        _generate_tasks[key] = asyncio.create_task(_do_generate())
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
            "version": target_version,
            "source_version": src_version,
            "nodes": len(nodes),
            "seeded_images": seeded,
        }

    @app.get("/slots/{slot_id}/{model_alias}/generate")
    async def slot_generate_status(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        run: str | None = None,
        version: str | None = None,
        optimized: bool = True,
    ) -> dict[str, object]:
        """Gate state for the client: every generated version of this cell, plus —
        for the targeted version (`version`, else latest) — whether a build is in
        flight and the ids of its finished GLBs. Polled while the "generated" view
        is active so the client can populate the version picker, enable/disable the
        gate, and attach freshly-built meshes one by one as they land."""
        run = _resolve_run(run)
        _require_slot_log(run, slot_id, model_alias)
        rid = _run_id(run, slot_id, model_alias)
        all_versions = generation.list_generated_versions(RUNS_DIR, rid)
        resolved = _read_gen_version(run, slot_id, model_alias, version)
        running = False
        meshes: list[dict[str, object]] = []
        if resolved is not None:
            key: GenKey = (run, slot_id, model_alias, resolved)
            gen_task = _generate_tasks.get(key)
            regen_task = _regen_tasks.get(key)
            regen_queue = _regen_queues.get(key)
            running = (
                (gen_task is not None and not gen_task.done())
                or (regen_task is not None and not regen_task.done())
                or (regen_queue is not None and not regen_queue.empty())
            )
            # Report the folder the client's optimized toggle is viewing — the
            # OPTIMIZED twins by default (what's served; an id only lands there
            # once its optimize pass finishes), or the raw objects-generated/
            # set when `optimized=0`. Each carries the GLB's mtime as a version
            # token so the client can detect a regenerated asset (same id, new
            # bytes) and reload just it with a cache-busted URL, plus `sym`/`symWas`
            # — the asset's current symmetry plane (none/xy/xz) and, if since
            # un-symmetrized, the plane it used to be mirrored across — so the detail
            # panel tells mirrored / un-symmetrized / never-symmetrized apart. The
            # `.raw.glb` intermediates are skipped below.
            sym_map = _generated_symmetry(
                generation.generated_events_path(RUNS_DIR, rid, resolved)
            )
            gen_dir = generation.generated_dirs(RUNS_DIR, rid, resolved)[1 if optimized else 0]
            if gen_dir.is_dir():
                for p in sorted(gen_dir.glob("*.glb")):
                    if p.name.endswith(".raw.glb"):
                        continue
                    try:
                        mtime = p.stat().st_mtime_ns
                    except OSError:
                        continue
                    mesh_id = p.name[: -len(".glb")]
                    info = sym_map.get(mesh_id)
                    meshes.append({
                        "id": mesh_id,
                        "v": mtime,
                        "sym": info["plane"] if info else "none",
                        "symWas": info["was"] if info else None,
                    })
        ids = [m["id"] for m in meshes]
        return {
            "running": running,
            "version": resolved,
            "versions": all_versions,
            "count": len(ids),
            "ids": ids,
            "meshes": meshes,
        }

    @app.post("/slots/{slot_id}/{model_alias}/regenerate/{node_id}")
    async def slot_regenerate(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        node_id: str,
        run: str | None = None,
        version: str | None = None,
        propagate: bool = False,
        backend: str = "trellis",
        reuse_image: bool = False,
    ) -> dict[str, object]:
        """Enqueue a from-scratch regeneration of a GENERATED asset (in the
        targeted `version`, else the latest) onto that version's regen worker. With
        `propagate=true` (the client default) the worker rebuilds the prefab
        CANONICAL behind `node_id` and re-derives every object that reuses it, so a
        shared prefab updates everywhere within the version at once.

        Requests ENQUEUE rather than 409 against each other: a single per-version
        worker (`_regen_worker`) drains the queue concurrently — in parallel across
        prefab groups, serialized within one — sharing that version's generated-events
        log and resolving each item's prefab group at execution time (so a regen
        enqueued later sees a reuse→canonical promotion an earlier one made). This
        also runs CONCURRENTLY with a whole-scene generate of the same version: the
        scene build and regens serialize per-node via `generation.node_lock` (plus
        atomic GLB writes), so they never write the same asset at once — regenerating
        an already-built asset proceeds while the scene build resumes the missing
        ones. Queued items show as `waiting` rows in the shared /trellis/queue panel
        until the worker picks them up."""
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        if backend not in generation.MESH_BACKENDS:
            raise HTTPException(status_code=400, detail=f"unknown backend: {backend}")
        lib_log = _require_slot_log(run, slot_id, model_alias)
        gen_version = _read_gen_version(run, slot_id, model_alias, version) or "1"
        key: GenKey = (run, slot_id, model_alias, gen_version)
        # A whole-scene generate of this version may be in flight — that's allowed
        # now. The regen and the scene build serialize per-node via
        # generation.node_lock (see _regen_worker), so they never write the same
        # asset's files at once.

        # Validate the node exists in the library layout up front (fast 404); the
        # worker reconstructs it + resolves the prefab group again at execution.
        ver = versions.for_run(run)
        prompt_runtime.bind(ver.prompt_module or _prompt_module_for_run(run))
        if _reconstruct_node(lib_log, node_id) is None:
            raise HTTPException(
                status_code=404, detail=f"no bbox event found for node: {node_id}",
            )

        gen_slot_id = _gen_slot_id(run, slot.id, model_alias, gen_version)
        queue = _regen_queues.setdefault(key, asyncio.Queue())
        queue.put_nowait((node_id, propagate, backend, "regenerate", reuse_image, None))
        # Surface the queued regen in the shared mesh queue panel until the
        # worker dequeues it (then generate_mesh manages its own entry). Tag the
        # backend so it lands in the right pool section (Trellis vs Hunyuan 3.1).
        threed.mark_queued(gen_slot_id, node_id, backend=backend)
        worker = _regen_tasks.get(key)
        if worker is None or worker.done():
            _regen_tasks[key] = asyncio.create_task(
                _regen_worker(run, slot.id, model_alias, gen_version)
            )
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
            "version": gen_version,
            "node_id": node_id,
            "propagate": propagate,
            "backend": backend,
            "queued": True,
            "depth": queue.qsize(),
        }

    @app.post("/slots/{slot_id}/{model_alias}/unsymmetrize/{node_id}")
    async def slot_unsymmetrize(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        node_id: str,
        run: str | None = None,
        version: str | None = None,
        propagate: bool = True,
    ) -> dict[str, object]:
        """Reveal a GENERATED asset's full, un-mirrored mesh: reprocess its existing
        raw mesh with symmetry turned OFF — no Nano-Banana, no mesh backend, so it's
        effectively instant. With `propagate=true` (the client default) the prefab
        CANONICAL behind `node_id` is un-mirrored and every object reusing it is
        re-derived to match, so a shared prefab stays consistent. Runs on the SAME
        per-version worker as regeneration (`_regen_worker`), so it enqueues, drains
        concurrently, and serializes per-node with builds + regens via
        `generation.node_lock`. Pins the node's symmetry decision to `none` so later
        resumes / regenerations keep it un-mirrored."""
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        lib_log = _require_slot_log(run, slot_id, model_alias)
        gen_version = _read_gen_version(run, slot_id, model_alias, version) or "1"
        key: GenKey = (run, slot_id, model_alias, gen_version)
        ver = versions.for_run(run)
        prompt_runtime.bind(ver.prompt_module or _prompt_module_for_run(run))
        if _reconstruct_node(lib_log, node_id) is None:
            raise HTTPException(
                status_code=404, detail=f"no bbox event found for node: {node_id}",
            )
        # Reuses the regen worker's queue; the backend slot is unused for an
        # un-symmetrize (it does no API calls) so it carries the default as filler.
        # Not surfaced in the mesh queue panel — it's a local reprocess, not a
        # backend generation.
        queue = _regen_queues.setdefault(key, asyncio.Queue())
        queue.put_nowait((node_id, propagate, generation.DEFAULT_MESH_BACKEND, "unsymmetrize", False, None))
        worker = _regen_tasks.get(key)
        if worker is None or worker.done():
            _regen_tasks[key] = asyncio.create_task(
                _regen_worker(run, slot.id, model_alias, gen_version)
            )
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
            "version": gen_version,
            "node_id": node_id,
            "propagate": propagate,
            "op": "unsymmetrize",
            "queued": True,
            "depth": queue.qsize(),
        }

    @app.post("/slots/{slot_id}/{model_alias}/symmetrize/{node_id}")
    async def slot_symmetrize(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        node_id: str,
        run: str | None = None,
        version: str | None = None,
        plane: str = "xy",
        keep_positive: bool = True,
        propagate: bool = True,
    ) -> dict[str, object]:
        """Mirror a GENERATED asset across `plane` ('xy' = front/back along Z, 'xz' =
        top/bottom along Y), keeping the `keep_positive` half — the symmetrize
        counterpart to /unsymmetrize. The plane + direction are supplied by the
        caller, so NO symmetry LLM decision is made and NO symmetry log is consulted
        to pick them. Reprocesses the existing raw mesh (no Nano-Banana, no mesh
        backend) on the SAME per-version worker as regenerate/unsymmetrize, so it
        enqueues, drains concurrently, and serializes per-node via
        `generation.node_lock`. With `propagate=true` (the client default) the prefab
        CANONICAL behind `node_id` is mirrored and every object reusing it is
        re-derived to match. Pins the node's symmetry decision so later resumes /
        regenerations keep the mirror."""
        if plane not in ("xy", "xz"):
            raise HTTPException(status_code=400, detail=f"plane must be 'xy' or 'xz', got: {plane}")
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        lib_log = _require_slot_log(run, slot_id, model_alias)
        gen_version = _read_gen_version(run, slot_id, model_alias, version) or "1"
        key: GenKey = (run, slot_id, model_alias, gen_version)
        ver = versions.for_run(run)
        prompt_runtime.bind(ver.prompt_module or _prompt_module_for_run(run))
        if _reconstruct_node(lib_log, node_id) is None:
            raise HTTPException(
                status_code=404, detail=f"no bbox event found for node: {node_id}",
            )
        # Same worker + filler backend slot as un-symmetrize; the (plane, keep)
        # ride the trailing `sym` field of the queue item. Not surfaced in the mesh
        # queue panel — a local reprocess, not a backend generation.
        queue = _regen_queues.setdefault(key, asyncio.Queue())
        queue.put_nowait(
            (node_id, propagate, generation.DEFAULT_MESH_BACKEND, "symmetrize", False, (plane, keep_positive))
        )
        worker = _regen_tasks.get(key)
        if worker is None or worker.done():
            _regen_tasks[key] = asyncio.create_task(
                _regen_worker(run, slot.id, model_alias, gen_version)
            )
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
            "version": gen_version,
            "node_id": node_id,
            "propagate": propagate,
            "op": "symmetrize",
            "plane": plane,
            "keep_positive": keep_positive,
            "queued": True,
            "depth": queue.qsize(),
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
        # Reset wipes the whole cell dir (including any `_branch/`), so tear the
        # run's branch down first to avoid it writing into a dir being deleted.
        await _discard_branch(run)
        await _cancel_task(run, slot_id, model_alias)
        # Tear down every in-flight from-scratch build for this cell (any version)
        # too, so their meshes aren't being written into the dir we're about to
        # wipe. Tasks are keyed per (cell, version); match on the cell prefix.
        cell = (run, slot_id, model_alias)
        for gkey in [k for k in _generate_tasks if k[:3] == cell]:
            gen_task = _generate_tasks.pop(gkey, None)
            if gen_task is not None and not gen_task.done():
                gen_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await gen_task
        # Tear down each version's regeneration worker + drop its queue (same dirs).
        # The worker's finally clears any still-queued rows from the queue panel.
        for gkey in [k for k in _regen_tasks if k[:3] == cell]:
            regen_task = _regen_tasks.pop(gkey, None)
            if regen_task is not None and not regen_task.done():
                regen_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await regen_task
        for gkey in [k for k in _regen_queues if k[:3] == cell]:
            _regen_queues.pop(gkey, None)
        # Drop this cell's per-node build locks (all versions) now that no
        # generate/regen task is left to hold them; the dir is about to be wiped.
        generation.clear_node_locks(_run_id(run, slot.id, model_alias))
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

    @app.post("/slots/{slot_id}/{model_alias}/branch")
    async def create_branch(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        req: BranchRequest,
        run: str | None = None,
    ) -> dict[str, object]:
        """Fork the original cell so it re-simulates from `deviation_index`
        onward, pausing on each step for prompt editing. Replaces any prior
        branch in this run. Isolated: writes only under `<cell>/_branch/`."""
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        src_log = _require_slot_log(run, slot.id, model_alias)
        src_events = list(src_log.state["events"])
        n = req.deviation_index
        if not (0 <= n < len(src_events)) or src_events[n].get("kind") != "cache.llm":
            raise HTTPException(
                status_code=400,
                detail="deviation_index must point at a cache.llm event",
            )

        # One branch per run: drop any existing branch (whatever cell it forked)
        # before forking this one.
        await _discard_branch(run)
        bdir = _branch_dir(run, slot.id, model_alias)
        shutil.rmtree(bdir, ignore_errors=True)
        bdir.mkdir(parents=True, exist_ok=True)

        # Hardlink the source's finished meshes so the replayed prefix doesn't
        # regenerate them — only the deviated subtree re-bills Trellis.
        cell_dir = src_log.events_path.parent
        src_objects = cell_dir / OBJECTS_SUBDIR
        if not src_objects.is_dir():
            src_objects = next(
                (cell_dir / d for d in ("objects-optimized", "objects") if (cell_dir / d).is_dir()),
                cell_dir / "objects",
            )
        await asyncio.to_thread(_hardlink_tree, src_objects, bdir / "objects")

        # Prefix = source events BEFORE the chosen step's cache.llm. Dropping
        # that step (and everything after) means `committed.*` re-runs it, the
        # step gate pauses on it for editing, and the step-through proceeds from
        # there. No output is injected — each step's prompt is edited live.
        branch_events = [dict(e) for e in src_events[:n]]
        bevents = bdir / "events.jsonl"
        with bevents.open("w", encoding="utf-8") as f:
            for e in branch_events:
                f.write(json.dumps(e) + "\n")

        blog = SlotLog(_branch_run_id(run, slot.id, model_alias), bevents)
        blog.hydrate_from_disk()
        if blog.state.get("prompt") is None or blog.state.get("model") is None:
            raise HTTPException(
                status_code=400,
                detail="source has no run.start to branch from",
            )
        _branch_logs[run] = blog
        _branch_tasks[run] = asyncio.create_task(_run_branch(run))
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
            "deviation_index": n,
            "events": len(branch_events),
        }

    @app.get("/slots/{slot_id}/{model_alias}/branch/scene")
    async def branch_scene(slot_id: str, model_alias: str, run: str | None = None) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        blog = _require_branch_log(run, slot_id, model_alias)
        return _scene_projection(list(blog.state["events"]))

    @app.get("/slots/{slot_id}/{model_alias}/branch/events")
    async def branch_events(slot_id: str, model_alias: str, since: int = -1, run: str | None = None) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        blog = _require_branch_log(run, slot_id, model_alias)
        q = blog.subscribe()
        snapshot = list(blog.state["events"])
        if since >= 0:
            snapshot = [
                e for e in snapshot
                if isinstance(e.get("index"), int) and e["index"] > since
            ]
        return StreamingResponse(_sse(blog, q, snapshot), media_type="text/event-stream")

    @app.get("/slots/{slot_id}/{model_alias}/branch/meshes")
    async def branch_meshes(slot_id: str, model_alias: str, run: str | None = None) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        blog = _require_branch_log(run, slot_id, model_alias)
        objects_dir = blog.events_path.parent / "objects"
        return StreamingResponse(
            _mesh_bundle(objects_dir),
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/slots/{slot_id}/{model_alias}/branch/step")
    async def branch_step(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        req: BranchStepRequest,
        run: str | None = None,
    ) -> dict[str, object]:
        """Advance the branch past the step it's paused on, running the step
        with the (possibly hand-edited) prompt. `auto` lets the rest run without
        further pauses. 409 if nothing is currently paused."""
        run = _resolve_run(run)
        _require_branch_log(run, slot_id, model_alias)
        controller = _branch_controllers.get(run)
        if controller is None or not controller.proceed(
            system=req.system, user=req.user, auto=req.auto,
        ):
            raise HTTPException(status_code=409, detail="no paused step to advance")
        return {"ok": True, "auto": req.auto}

    @app.post("/slots/{slot_id}/{model_alias}/branch/rerun")
    async def branch_rerun(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        req: BranchRerunRequest,
        run: str | None = None,
    ) -> dict[str, object]:
        """Re-run an already-committed step (by its `cache.llm` event index)
        with an edited prompt, INVALIDATING everything after it: truncate the
        log to before that step, discard the undone steps' mesh artifacts, seed
        the step's prompt, and relaunch so it replays (no pause) and then pauses
        on the next step. Navigation (prev/next) is client-side and never hits
        this — only an explicit re-run is destructive."""
        run = _resolve_run(run)
        blog = _require_branch_log(run, slot_id, model_alias)
        events = blog.state["events"]
        p = req.llm_index
        if not any(
            e.get("index") == p and e.get("kind") == "cache.llm" for e in events
        ):
            raise HTTPException(status_code=400, detail="llm_index must point at a committed step")
        # Drop the artifacts of every node placed at/after the re-run point, so
        # they regenerate (otherwise `path.exists()` reuses a stale placement).
        objs = blog.events_path.parent / "objects"
        for e in events:
            idx = e.get("index")
            oid = e.get("id")
            if isinstance(idx, int) and idx >= p and isinstance(oid, str):
                for suffix in (".glb", ".raw.glb", ".png"):
                    with contextlib.suppress(OSError):
                        (objs / f"{oid}{suffix}").unlink()
        await _cancel_branch_task(run)
        blog.truncate_events_to(p)
        _branch_reseed[run] = {"system": req.system, "user": req.user}
        _branch_tasks[run] = asyncio.create_task(_run_branch(run))
        return {"ok": True, "events": p}

    @app.delete("/slots/{slot_id}/{model_alias}/branch")
    async def discard_branch(slot_id: str, model_alias: str, run: str | None = None) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        _require_slot(slot_id)
        _require_model(model_alias)
        # Only the cell that owns the run's branch may discard it (idempotent —
        # a missing/mismatched branch is a no-op).
        blog = _branch_logs.get(run)
        if blog is not None and blog.slot_id == _branch_run_id(run, slot_id, model_alias):
            await _discard_branch(run)
        return {"run": run, "slot_id": slot_id, "model": model_alias}

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
    cut_plane: Literal["none", "xy", "xz"] = "none"
    for event in events:
        if event.get("id") == node_id and event.get("kind") == "symmetry.decision":
            raw_cp = event.get("cut_plane")
            if raw_cp in ("none", "xy", "xz"):
                cut_plane = raw_cp  # type: ignore[assignment]
            break
    encapsulating = bbox_event.get("node_kind") == "frame"
    if raw_str.lstrip().startswith(_WRAPPED_PROMPT_PREFIX):
        image_prompt = raw_str
        subject_str = raw_str
    else:
        subject_str = raw_str
        p = prompt_runtime.current()
        view = sym_svc.image_view_for(
            cut_plane=cut_plane, encapsulating=bool(encapsulating),
        )
        image_prompt = p.wrap_image_prompt(subject_str, proxy_shape, bbox.size, view=view)
    parent_id = bbox_event.get("parent_id")
    return Node(
        id=node_id,
        prompt=subject_str,
        bbox=bbox,
        proxy_shape=proxy_shape,
        orientation=orientation,
        image_prompt=image_prompt,
        symmetry_cut_plane=cut_plane,
        parent_id=str(parent_id) if isinstance(parent_id, str) else None,
    )


def _scene_nodes_from_library(slot_log: SlotLog) -> list[Node]:
    """Every concrete (object/frame) node reconstructed from a cell's library log,
    in first-seen order — the exact layout a from-scratch generated build reuses
    (same bboxes/orientations), instead of re-running the divider. Shared by the
    generate gate and the reuse-images fork."""
    seen: set[str] = set()
    nodes: list[Node] = []
    for event in slot_log.state["events"]:
        if event.get("kind") != "bbox" or event.get("node_kind") == "zone":
            continue
        node_id = event.get("id")
        if not isinstance(node_id, str) or node_id in seen:
            continue
        seen.add(node_id)
        node = _reconstruct_node(slot_log, node_id)
        if node is not None:
            nodes.append(node)
    return nodes


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


def _hardlink_tree(src: Path, dst: Path) -> None:
    """Mirror the flat `objects/` dir (`<id>.glb`, `<id>.raw.glb`, `<id>.png`)
    from `src` into `dst` via hardlinks — instant and zero extra disk, so the
    branch's `_spawn_meshes` sees committed prefix meshes as already present
    (`path.exists()`) and skips re-billing them. Prefix meshes are read-only in
    the branch (only NEW ids generate), so sharing inodes is safe. Falls back
    to a copy where hardlinks aren't supported."""
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if not p.is_file():
            continue
        target = dst / p.name
        if target.exists():
            continue
        try:
            os.link(p, target)
        except OSError:
            with contextlib.suppress(OSError):
                shutil.copy2(p, target)


def _require_branch_log(run: str, slot_id: str, model_alias: str) -> SlotLog:
    """The run's branch, but only when it was forked from THIS cell (its
    composite run_id, stored as the SlotLog's slot_id, encodes the source
    cell). 404 otherwise — so a cell only ever sees its own branch."""
    _require_slot(slot_id)
    _require_model(model_alias)
    blog = _branch_logs.get(run)
    if blog is None or blog.slot_id != _branch_run_id(run, slot_id, model_alias):
        raise HTTPException(status_code=404, detail="no active branch for this cell")
    return blog


class BranchStepController:
    """Pauses the branch pipeline before each real (cache-miss) LLM call so the
    user can edit the step's re-rendered prompt, then resumes it. Bound as the
    `llm` step gate inside the branch task; `/branch/step` resolves each pause.

    Single-event-loop, so the cross-coroutine `Future.set_result` from the
    endpoint safely wakes the branch task awaiting the gate."""

    def __init__(self, slot_log: SlotLog) -> None:
        self.slot_log = slot_log
        self._gate: asyncio.Future[dict[str, object]] | None = None
        self.auto = False  # once set, the rest of the branch runs without pausing
        # One-shot {system, user} to run the FIRST step with, no pause (a
        # re-run replays its target step with the edited prompt).
        self.seed: dict[str, object] | None = None

    async def gate(
        self, *, node_id: str | None, step: str | None,
        system: str, user: str, schema_name: str, model: str,
    ) -> tuple[str, str]:
        if self.auto:
            return system, user
        if self.seed is not None:
            seed = self.seed
            self.seed = None
            s, u = seed.get("system"), seed.get("user")
            return (s if isinstance(s, str) else system, u if isinstance(u, str) else user)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, object]] = loop.create_future()
        self._gate = fut
        # Surface the pending step (its freshly re-rendered prompt) to the client.
        self.slot_log.log(
            "branch.step.pending",
            node=node_id,
            step=step,
            system=system,
            user=user,
            schema=schema_name,
            model=model,
        )
        try:
            result = await fut
        finally:
            self._gate = None
        if result.get("auto"):
            self.auto = True
        new_system = result.get("system")
        new_user = result.get("user")
        return (
            new_system if isinstance(new_system, str) else system,
            new_user if isinstance(new_user, str) else user,
        )

    def proceed(self, *, system: str | None = None, user: str | None = None, auto: bool = False) -> bool:
        """Resolve the current pause. Returns False if nothing is paused."""
        if self._gate is None or self._gate.done():
            return False
        self._gate.set_result({"system": system, "user": user, "auto": auto})
        return True


async def _cancel_branch_task(run: str) -> None:
    """Cancel the run's branch task + in-flight branch meshes + stale step
    controller, but KEEP its SlotLog and dir — so a back-step can truncate the
    log and relaunch on the same branch."""
    task = _branch_tasks.pop(run, None)
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    _branch_controllers.pop(run, None)
    _branch_reseed.pop(run, None)
    log = _branch_logs.get(run)
    if log is not None:
        versions.for_run(run).generation.cancel_pending(log.slot_id)


async def _cancel_branch(run: str) -> None:
    """Tear down the run's live branch task + in-flight branch meshes (but
    leave the on-disk `_branch/` dir for the caller to keep or delete)."""
    await _cancel_branch_task(run)
    log = _branch_logs.pop(run, None)
    if log is not None:
        log.close()


async def _discard_branch(run: str) -> None:
    """Full break-out: cancel the run's branch and delete its directory."""
    log = _branch_logs.get(run)
    branch_dir = log.events_path.parent if log is not None else None
    await _cancel_branch(run)
    if branch_dir is not None:
        shutil.rmtree(branch_dir, ignore_errors=True)


async def _run_branch(run: str) -> None:
    """Drive the run's branch pipeline. A mirror of `_run` bound to the branch's
    SlotLog + branch run_id: it resumes (the prefix replays via `committed.*`,
    the tuned step cache-hits the swapped output, and the frontier re-runs),
    streaming into the branch log only."""
    blog = _branch_logs[run]
    rlog.bind(blog)
    # Bind the step gate IN this task's context so only the branch pauses; the
    # main pipeline tasks leave the gate None and never block.
    controller = BranchStepController(blog)
    controller.seed = _branch_reseed.pop(run, None)  # set by /branch/rerun
    _branch_controllers[run] = controller
    llm.set_step_gate(controller.gate)
    ver = versions.for_run(run)
    prompt_runtime.bind(ver.prompt_module or _prompt_module_for_run(run))
    prompt = blog.state["prompt"]
    model = blog.state["model"]
    brun_id = blog.slot_id  # composite branch run_id (run/slot/model/_branch)
    try:
        await ver.run(run_id=brun_id, prompt=prompt, model=model, runs_dir=RUNS_DIR)
    except asyncio.CancelledError:
        ver.generation.cancel_pending(brun_id)
        raise
    except Exception as e:
        ver.generation.cancel_pending(brun_id)
        blog.log("run.error", message=f"{type(e).__name__}: {e}")
        return
    await ver.generation.await_pending(brun_id)
    blog.finish_run()


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
            n["orientation"] = e.get("orientation", 0)
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


_ALLOWED_ORIENTATIONS = frozenset({-180, -135, -90, -45, 0, 45, 90, 135, 180})


def _nodes_from_events(events: list[dict[str, object]]) -> list[Node]:
    """Reconstruct the scene's Node tree from a cell's event log — enough fields
    for the canonical scene-context renderers (bbox, prompt, parent, plan,
    zone/object kind, proxy shape, orientation). Built off `_scene_projection`'s
    fold so the two stay in lockstep. Nodes without a resolved bbox are skipped."""
    nodes: list[Node] = []
    for n in _scene_projection(events)["nodes"]:
        nid = n.get("id")
        origin = n.get("origin")
        dimensions = n.get("dimensions")
        if not isinstance(nid, str):
            continue
        if not (isinstance(origin, (list, tuple)) and len(origin) == 3):
            continue
        if not (isinstance(dimensions, (list, tuple)) and len(dimensions) == 3):
            continue
        proxy_raw = n.get("proxy_shape")
        try:
            proxy = ProxyShape(proxy_raw) if proxy_raw else None
        except ValueError:
            proxy = None
        try:
            orientation = int(n.get("orientation", 0) or 0)
        except (TypeError, ValueError):
            orientation = 0
        try:
            node = Node(
                id=nid,
                prompt=str(n.get("prompt") or ""),
                bbox=BoundingBox(origin=tuple(origin), dimensions=tuple(dimensions)),  # type: ignore[arg-type]
                proxy_shape=proxy,
                orientation=orientation if orientation in _ALLOWED_ORIENTATIONS else 0,
                parent_id=n.get("parent_id") if isinstance(n.get("parent_id"), str) else None,
                plan=n.get("plan") if isinstance(n.get("plan"), str) else None,
                is_zone=(n.get("node_kind") == "zone"),
            )
        except Exception:  # noqa: BLE001 — a malformed node shouldn't sink the plan
            continue
        nodes.append(node)

    # `parent_region` isn't persisted in the bbox event, but the V3/V4 scene-
    # context renderers group objects by it (the zone whose generation pass
    # emitted them). Reconstruct it by walking each object's `parent_id` up past
    # any peer-object anchors to the nearest enclosing zone; without this every
    # object groups under None and renderers (e.g. the anchor planner) drop them,
    # leaving only zones in the context.
    by_id = {n.id: n for n in nodes}
    for n in nodes:
        if n.is_zone:
            continue
        cur = n.parent_id
        while cur in by_id and not by_id[cur].is_zone:
            cur = by_id[cur].parent_id
        n.parent_region = cur if (cur in by_id and by_id[cur].is_zone) else None
    return nodes


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


# How long the regen worker blocks on in-flight builds before looping back to
# re-drain its queue. Bounds how long a regen enqueued mid-batch waits before it
# starts (it shouldn't wait for a prior build to finish); the worker only spins
# this while it has builds in flight, and exits outright once idle.
_REGEN_POLL_INTERVAL_S = 0.25


async def _regen_worker(run: str, slot_id: str, model_alias: str, version: str) -> None:
    """Drain one (cell, version)'s regeneration queue CONCURRENTLY. One worker per
    version owns that version's generated-events log for the whole drain;
    `SlotLog.log()` is synchronous, so concurrent items append through it atomically
    (unique indices, no torn lines). Items run in parallel ACROSS prefab groups but
    are serialized WITHIN a group by a per-canonical lock — so two regens that
    resolve to the same canonical (a stray double-enqueue, or a reuse plus its
    canonical) can't race the same files, and the second re-resolves under the lock
    to see the first's promotion. Spawns are unbounded; the heavy work is throttled
    by the process-global Banana / Trellis / mesh-IO / optimize semaphores. Exits
    when the queue is empty and nothing is in flight; the next enqueue restarts it.
    Cancellation (reset / stop / teardown) cancels in-flight builds and clears any
    still-queued rows from the shared Trellis queue panel."""
    key: GenKey = (run, slot_id, model_alias, version)
    queue = _regen_queues.get(key)
    if queue is None:
        return
    lib_log = _slot_logs.get((run, slot_id, model_alias))
    ver = versions.for_run(run)
    run_id = _run_id(run, slot_id, model_alias)
    gen_slot_id = _gen_slot_id(run, slot_id, model_alias, version)
    raw_subdir = f"{generation.GENERATED_DIR}/{version}/{generation.GENERATED_RAW_SUBDIR}"
    gen_log = SlotLog(gen_slot_id, generation.generated_events_path(RUNS_DIR, run_id, version))
    gen_log.hydrate_from_disk()
    rlog.bind(gen_log)
    llm.set_model(MODELS[model_alias])
    prompt_runtime.bind(ver.prompt_module or _prompt_module_for_run(run))
    canon_locks: dict[str, asyncio.Lock] = {}

    async def _process(
        node_id: str,
        propagate: bool,
        backend: str,
        op: str,
        reuse_image: bool,
        sym: tuple[str, bool] | None,
    ) -> None:
        if lib_log is None:
            return
        try:
            # Pick the per-canonical lock by this node's current group, then do the
            # real resolution + build UNDER the lock so same-group items serialize
            # and a later one observes an earlier item's promotion.
            canonical0, _ = prefabs.resolve_group(gen_log.state["events"], node_id)
            async with canon_locks.setdefault(canonical0, asyncio.Lock()):
                target = _reconstruct_node(lib_log, node_id)
                if target is None:
                    gen_log.log("mesh.error", id=node_id, message=f"{op}: no bbox event for node")
                    return
                canonical_id, reuse_ids = prefabs.resolve_group(gen_log.state["events"], node_id)
                if propagate:
                    build_node = _reconstruct_node(lib_log, canonical_id)
                    if build_node is None:
                        gen_log.log(
                            "mesh.error", id=node_id,
                            message=f"{op}: no bbox event for canonical {canonical_id}",
                        )
                        return
                    reuses = [
                        n for cid in reuse_ids if (n := _reconstruct_node(lib_log, cid)) is not None
                    ]
                else:
                    build_node = target
                    reuses = []
                if op == "unsymmetrize":
                    # Strip the symmetry mirror off the existing mesh (no AI). The
                    # canonical is un-mirrored here; propagate re-derives its reuses
                    # below, which read the canonical's now-`none` symmetry.applied.
                    await generation.unsymmetrize_one(
                        node=build_node, runs_dir=RUNS_DIR, run_id=run_id, version=version,
                    )
                elif op == "symmetrize":
                    # Mirror the existing mesh across the caller-supplied plane (no
                    # AI). The canonical is mirrored here; propagate re-derives its
                    # reuses below, which read the canonical's symmetry.applied
                    # (plane + kept half) and mirror identically.
                    cut_plane, keep_positive = sym  # type: ignore[misc]
                    await generation.symmetrize_one(
                        node=build_node, cut_plane=cut_plane, keep_positive=keep_positive,  # type: ignore[arg-type]
                        runs_dir=RUNS_DIR, run_id=run_id, version=version,
                    )
                else:
                    if reuse_image:
                        # From-image rebuild: ensure the node we're building has its
                        # raw-dir reference image. If that copy is missing (e.g. the
                        # raw image was removed but the optimized twin survived, or a
                        # prefab sibling still has it), restore it so we reuse the
                        # image instead of re-generating it via the API.
                        generation.recover_group_image(
                            RUNS_DIR, run_id, version, build_node.id, [canonical_id, *reuse_ids],
                        )
                    await generation.regenerate_one(
                        node=build_node, runs_dir=RUNS_DIR, run_id=run_id,
                        subdir=raw_subdir, optimize=True, backend=backend, version=version,
                        reuse_image=reuse_image,
                    )
                if propagate:
                    await generation.propagate_reuses(
                        canonical_id=canonical_id, reuses=reuses,
                        runs_dir=RUNS_DIR, run_id=run_id, version=version,
                    )
                elif canonical_id != node_id and op not in ("unsymmetrize", "symmetrize"):
                    # A reuse regenerated on its own now owns a fresh mesh + raw —
                    # record it as canonical so a later propagate of its old source
                    # can't clobber it. (Symmetry ops write no new raw, so they never
                    # promote — a reuse with no own raw is skipped instead.)
                    gen_log.log(
                        "prefab.match", id=node_id, reuse_id="", description=build_node.prompt,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            gen_log.log("mesh.error", id=node_id, message=f"{type(e).__name__}: {e}")

    inflight: set[asyncio.Task[None]] = set()
    try:
        while True:
            # Spawn every currently-queued item; the semaphores bound the real work.
            while True:
                try:
                    node_id, propagate, backend, op, reuse_image, sym = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                # Worker owns the entry now; generate_mesh re-registers it on submit.
                threed.unmark_queued(gen_slot_id, node_id)
                inflight.add(asyncio.create_task(_process(node_id, propagate, backend, op, reuse_image, sym)))
            if not inflight:
                break
            # Wait for a build to finish OR a short tick to elapse, then loop back
            # to the drain — so a regen enqueued WHILE these run starts promptly
            # (concurrently) instead of only after a prior build completes.
            inflight = (
                await asyncio.wait(
                    inflight,
                    timeout=_REGEN_POLL_INTERVAL_S,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            )[1]
    finally:
        for t in inflight:
            t.cancel()
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
        # Clear any items still queued (e.g. drained early by cancellation) from
        # the shared queue panel so they don't linger as phantom waiting rows.
        while True:
            try:
                pending_id, *_ = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            threed.unmark_queued(gen_slot_id, pending_id)
        gen_log.close()


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
