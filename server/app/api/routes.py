"""HTTP API — endpoints scoped to a (run, slot, model) cell.

`RUNS_DIR` is a parent directory holding many named *runs*; each run is a
set of (slot, model) cells with its own prompt snapshot (`prompts/`, copied
from a `versions/` prompt version at creation), events.jsonl, mesh
artifacts, and SSE streams. The viewer picks the active run at runtime via
`GET /runs` / `POST /runs/{name}/activate`.

For each (run, slot, model) cell: fresh ones sit idle, interrupted ones
come back as paused, completed ones stay done. Nothing auto-launches;
the viewer drives start/resume/reset per cell. Runs are hydrated lazily
on activation; the initial active run is the newest subdir of RUNS_DIR.

Every asyncio task is bound to its SlotLog via a ContextVar so
concurrent pipeline work routes events to the right cell without
threading a handle through every call site. The Trellis queue is
process-global; rows tag `slot_id` with the composite
`run/slot/model_alias` so the dashboard filters to the visible cell.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import os
import re
import secrets
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.core import prompt_store, scene_context, schemas
from app.utils import cache
from app.oneshot import routes as oneshot
from app.core.slots import (
    DEFAULT_MODEL_ALIAS,
    MODEL_ALIASES,
    MODELS,
    SLOTS,
    SLOTS_BY_ID,
    Slot,
)
from app.core.types import BoundingBox, Node, Orientation, ProxyShape
from app.pipeline import committed, divider, generation, object_wipe
from app.services import llm, prefabs, threed
from app.services import symmetry as sym_svc
from app.utils import logging as rlog
from app.utils.logging import SlotLog

# Parent directory holding many named runs. Each immediate subdirectory
# is one run; cells live at RUNS_DIR/<run>/<slot>/<model>. Anchored to the
# repo root (this file is server/app/api/routes.py) rather than the launch
# CWD, mirroring prompt_store.VERSIONS_DIR.
_REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = Path(os.environ.get("STARSHOT_RUNS_DIR", _REPO_ROOT / "runs"))

# The splat pipeline lives in the top-level `splat/` package (repo root), outside
# the server's `app` package — put the repo root on sys.path so it imports
# cleanly, mirroring how _REPO_ROOT anchors RUNS_DIR above.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from splat import stage1 as splat_stage1  # noqa: E402
from splat import stage2 as splat_stage2  # noqa: E402
from splat import stage3 as splat_stage3  # noqa: E402
from splat import stage4 as splat_stage4  # noqa: E402
from splat import stage5 as splat_stage5  # noqa: E402  (torch/nvdiffrast lazy inside)
from splat import stage6 as splat_stage6  # noqa: E402  (torch/gsplat lazy inside)

# Node de-optimizer (server/tools/optimize-assets/deoptimize.mjs): rewrites a
# cell's KTX2/Meshopt library GLBs to vanilla glTF so the trimesh-based Stage-1
# assembler can read them. The from-scratch generated RAW build is already
# vanilla and skips this. Reuses the optimizer tool's node_modules.
_DEOPT_DIR = _REPO_ROOT / "server" / "tools" / "optimize-assets"
_DEOPT_SCRIPT = _DEOPT_DIR / "deoptimize.mjs"
_NODE_BIN = os.environ.get("STARSHOT_NODE_BIN", "node")

# Offline "lite" (presentation-tier) asset builder — driven per-cell by the
# build-lite endpoint (background subprocess, progress polled by the client).
_BUILD_LITE_SCRIPT = _REPO_ROOT / "server" / "scripts" / "build_lite_assets.py"

# Which per-cell mesh directory the scene bundle streams from. Defaults to the
# originals ("objects"); set STARSHOT_OBJECTS_SUBDIR=objects-optimized to serve
# the re-baked optimized set (scripts/rebake_runs.py) instead. Falls back to
# "objects" for any cell that hasn't been migrated.
OBJECTS_SUBDIR = os.environ.get("STARSHOT_OBJECTS_SUBDIR", "objects")

# Per-cell (run × slot × model) LLM spend cap, in USD — the DEFAULT ceiling a
# cell starts with. When a cell's settled OpenRouter spend crosses its effective
# cap the backfill sweep auto-pauses it ("spend cap reached"); it stays there
# until the cap is raised past the spend (see /cap-override). The effective cap
# is the ceiling carried by the cell's latest `run.cap_override` event, or this
# default if it has none — so it's entirely derived from the durable log
# (`llm.cost` + `run.cap_override`), hence restart- and rewind-proof. Set to 0
# (or negative) to disable the cap entirely.
SPEND_CAP_USD = float(os.environ.get("STARSHOT_SPEND_CAP_USD", "200"))

# Python's mimetypes doesn't know glTF; without these the artifact route would
# hand the loader its GLBs as text/plain.
_ARTIFACT_MEDIA_TYPES = {
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    # Stage-2 Gaussian clouds + Stage-3 free-voxel packs — raw bytes the splat
    # viewer fetches (.ply cloud, .bin voxel field).
    ".ply": "application/octet-stream",
    ".bin": "application/octet-stream",
    # SOG-encoded trained splat (client/tools/ply-to-sog.mjs) — a zip bundle the
    # PlayCanvas gsplat loader fetches as raw bytes.
    ".sog": "application/octet-stream",
    # Stage-6 training log — served inline so `log_url` opens live in a browser.
    ".log": "text/plain; charset=utf-8",
}

# Per-run metadata written at creation (chosen prompt version, created_at).
RUN_META_NAME = "run.json"

# Keyed by (run_name, slot_id, model_alias). Each cell is an independent
# pipeline. Lazy-populated: only runs the user has activated are loaded.
RunKey = tuple[str, str, str]
# The from-scratch generated build of a cell keys its task tables on
# (run, slot, model) — one build per cell, separate from the library build.
GenKey = tuple[str, str, str]
_slot_logs: dict[RunKey, SlotLog] = {}
_tasks: dict[RunKey, asyncio.Task[None]] = {}
_retry_tasks: set[asyncio.Task[None]] = set()
# In-flight "generate from-scratch assets" task per (cell, version). Drives the
# client's generate gate (a version can't re-trigger until its current build
# finishes) and lets shutdown / reset cancel a build cleanly.
_generate_tasks: dict[GenKey, asyncio.Task[None]] = {}
# Per-(cell, version) regeneration worker + its FIFO of `RegenJob`s, where `op`
# is "regenerate" (fresh mesh — and, unless reuse_image, a fresh Nano-Banana
# image too), "unsymmetrize" / "symmetrize" (reprocess the existing raw mesh
# with the symmetry mirror off/on — no AI; reuse_image is ignored), "link"
# (join the object into another prefab group, re-deriving its mesh from that
# group's canonical — no AI), or "unlink" (split the object out of its group into
# a standalone asset with its own raw mesh — no AI). `_regen_tasks` holds the
# single worker task per
# version; `_regen_queues` is the work it drains concurrently (parallel across
# prefab groups, serialized within one), sharing that version's generated-events
# log. Requests enqueue rather than 409 between each other, and run concurrently
# with a whole-scene generate of the same version: they serialize per-node with it
# via generation.node_lock, so neither writes the same asset twice.
_regen_tasks: dict[GenKey, asyncio.Task[None]] = {}
_regen_queues: dict[GenKey, asyncio.Queue["RegenJob"]] = {}
_hydrated_runs: set[str] = set()

# --- Stage 1 splat conversion jobs (background, one per CELL) ----------------
# Keyed by (run, slot, model): selecting a run lists cells; clicking a cell
# starts its own conversion to a scene manifest (splat/stage1.py). The worker
# updates its job dict's counters in place from a worker thread;
# `_splat_stage1_tasks` holds the live task so it stays referenced.
_splat_stage1_jobs: dict[tuple[str, str, str], dict[str, Any]] = {}
_splat_stage1_tasks: dict[tuple[str, str, str], asyncio.Task[None]] = {}

# --- lite-asset build jobs (background, one per CELL) -------------------------
# A cell's generated raw build -> objects-generated-lite/ via build_lite_assets.py.
# The task streams the driver's "[i/N]" progress into the job dict the status
# endpoint serves.
_lite_build_jobs: dict[tuple[str, str, str], dict[str, Any]] = {}
_lite_build_tasks: dict[tuple[str, str, str], asyncio.Task[None]] = {}

# --- Stage 2 free-space voxelization jobs (background, one per CELL) ----------
# The shared spatial foundation (Option A, runs first): a cell's meshes (generated,
# or library de-optimized) are voxelized into a dual-resolution occupancy +
# clearance + reachability grid `splat/freespace.npz` (splat/stage2.py), with a
# `voxels.bin` viz cloud + `freespace.json` sidecar so 'done' survives a restart.
_splat_stage2_jobs: dict[tuple[str, str, str], dict[str, Any]] = {}
_splat_stage2_tasks: dict[tuple[str, str, str], asyncio.Task[None]] = {}

# --- Stage 3 surfel sampling jobs (background, one per CELL) ------------------
# Same shape/keying: a cell's placed meshes + the Stage-2 free-space grid are
# sampled into a Gaussian cloud `splat/cloud.ply` (splat/stage3.py) — normals
# oriented to free space, hidden faces culled. A `cloud.json` sidecar holds the
# summary; the status carries the .ply's `/artifacts` URL for the viewer.
_splat_stage3_jobs: dict[tuple[str, str, str], dict[str, Any]] = {}
_splat_stage3_tasks: dict[tuple[str, str, str], asyncio.Task[None]] = {}

# --- Stage 4 coverage-camera-planning jobs (background, one per CELL) ---------
# Feature-adaptive patches + greedy camera placement (splat/stage4.py). Writes
# `splat/cameras.json` (+ `patches.bin`); a `cameras.json` on disk is the 'done'
# marker. Same shape/keying as the earlier stages.
_splat_stage4_jobs: dict[tuple[str, str, str], dict[str, Any]] = {}
_splat_stage4_tasks: dict[tuple[str, str, str], asyncio.Task[None]] = {}

# --- Stage 5 reference-render jobs (background, one per CELL) -----------------
# Unlit nvdiffrast renders (splat/stage5.py) of the Stage-4 camera plan → per-view
# RGB/depth/alpha + `splat/refs/transforms.json` (the 'done' marker). CUDA-only:
# on a non-GPU host the job finishes with a clear 'needs CUDA' error. Same
# shape/keying as the earlier stages.
_splat_stage5_jobs: dict[tuple[str, str, str], dict[str, Any]] = {}
_splat_stage5_tasks: dict[tuple[str, str, str], asyncio.Task[None]] = {}

# --- Stage 6 fine-tune processes (background, one per CELL) -------------------
# Unlike stages 1-5 (in-process asyncio jobs), Stage 6 is a long (hours) CUDA
# training run launched as a DETACHED child process (see `_spawn_stage6`) that
# streams to splat/stage6.log and is polled via `popen.poll()` — so it runs async
# and never blocks the event loop. Value: {popen, logf, iterations, started_at,
# returncode}. The record is kept after exit so 'done'/'error' + log survive polls.
_splat_stage6_procs: dict[tuple[str, str, str], dict[str, Any]] = {}
_current_run: str = ""


@dataclass(frozen=True)
class RegenJob:
    """One item the per-version regen worker drains. `op` selects the action:

      * "regenerate"  — rebuild the mesh fresh on `backend` (and the image too
        unless `reuse_image`). `propagate` re-derives the whole prefab group from
        the rebuilt canonical.
      * "unsymmetrize" / "symmetrize" — reprocess the existing raw mesh with the
        mirror off / on (`sym` carries (plane, keep_positive)); no backend call.
      * "reorient" — change the "front view" by rotating the raw mesh 90° about an
        axis (`reorient` carries (axis, degrees)); no backend call.
      * "glassify" — force the window/glass transparency transform (white texels ->
        near-clear) onto the object's served mesh, bypassing the pipeline's keyword
        + symmetry gates. Applies to the WHOLE prefab group (each member's served
        mesh directly, since glass isn't raw-derivable); no backend call.
      * "reset" — rebuild the object's served mesh from its pristine raw, dropping
        any in-place served edit (e.g. a forced glassify) while keeping its current
        symmetry; propagates across the prefab group; no backend call.
      * "link" — join this object into the prefab group of `link_to` (any member
        of the destination group), re-deriving its mesh from that group's
        canonical; no backend call.
      * "unlink" — split this object OUT of its prefab group into a standalone
        asset with its own raw mesh (a reuse clones its canonical's geometry; a
        canonical with reuses hands the group off to one of them), without
        rebuilding it; no backend call.
    """

    node_id: str
    op: str
    propagate: bool = True
    backend: str = generation.DEFAULT_MESH_BACKEND
    reuse_image: bool = False
    # Regenerate the object's noun phrase (re-distill from its authored seed) as
    # part of a "regenerate" op, then re-log the `image` event with it so later
    # regenerations read the new phrase. Forces `reuse_image` off (a new phrase
    # needs a new image).
    regen_noun_phrase: bool = False
    sym: tuple[str, bool] | None = None
    reorient: tuple[str, int] | None = None
    link_to: str | None = None
    link_group: bool = False


# Downstream-simulation BRANCHES. A branch is a "what-if" fork: the prefix of
# some source log (a cell, or another branch) up to a chosen event, then the
# pipeline re-run from there under the run's prompt snapshot PLUS the lab's
# edited step templates — so the forked step (and every later firing of it)
# renders from the edit while everything else replays via `committed.*`.
#
# Every branch is a FIRST-CLASS entity keyed by its own `branch_id`, stored
# flat in ONE per-run temp folder: `<run>/_branches/<branch_id>/` with its own
# events.jsonl + objects/ + branch.json manifest, and its own composite run_id
# (`<run>/_branches/<branch_id>`) so the LLM cache, the `committed.*` resume
# reader, `generation._pending`, and the Trellis queue all key off the branch,
# never the source. This single flat namespace is the whole branching surface:
#
#   * MANY branches per cell — fork a cell at different zones (different event
#     indices) to run several independent downstream sims of one slot at once.
#   * PARALLEL LLM on one sim — "run the next step on N models" forks N CHILD
#     branches (`parent` set) of the active sim at its frontier, each pinned to
#     a model and gated to run exactly that step, all CONCURRENTLY. The user
#     views each child's scene/output, then KEEPS one (it becomes the sim; the
#     parent + losing siblings are dropped). Children reuse the identical
#     branch machinery — no separate "candidate" path.
#
# A top-level sim has `parent=None`; a parallel-LLM preview has `parent=<sim
# id>`. The manifest makes the temp folder self-describing, so branches are
# enumerable and survive a restart (rehydrated paused/resumable on activation).
BRANCHES_SUBDIR = "_branches"
BRANCH_MANIFEST = "branch.json"


@dataclass
class Branch:
    """One live simulation fork. `slot`/`model` name the ORIGIN cell (the
    source of the prompt + the promote target); `parent` is the branch this
    forked from (None ⇒ forked straight from the origin cell). `model_pin` is
    an OpenRouter id the gate re-aims this branch's frontier call at — set on
    parallel-LLM children so each runs the same step on a different model;
    None ⇒ the cell's base model."""

    id: str
    run: str
    slot: str
    model: str
    parent: str | None
    fork_index: int
    overrides: dict[str, dict[str, str]]
    log: SlotLog
    model_pin: str | None = None
    # A source version this branch runs under instead of the run snapshot — the
    # prompt-set A/B lineage (None ⇒ the run's current prompts).
    version: str | None = None
    task: asyncio.Task[None] | None = None
    gate: "CellGate | None" = None
    # Next-launch gate intent for a PAUSED branch (budget / auto / until /
    # model) — consumed by `_run_branch` when a step advance relaunches it.
    gate_intent: dict[str, object] = field(default_factory=dict)
    # Node ids this branch has LOCKED to atomic — the prompt-lab "lock a zone
    # atomic" test override, forced into `is_atomic` when the branch re-runs.
    atomic_locks: list[str] = field(default_factory=list)


# branch_id -> Branch. Process-global: ids are unique across runs, and each
# Branch carries its own run, so the bid-keyed endpoints need no run param.
_branches: dict[str, Branch] = {}

# --- LLM cost backfill --------------------------------------------------------
# Pricing a call against OpenRouter's settled cost is deferred off the pipeline:
# the /generation stats lag the completion, and a live lookup dies with a
# restart. This sweep prices any `cache.llm` still missing its `llm.cost`, purely
# from the durable log — so a run's spend converges to OpenRouter's actual
# billing, restart or not, and the per-cell `pending` count it drives to zero is
# the "all costs resolved, safe to shut down" signal.
_COST_BACKFILL_INTERVAL_S = 20


def _all_cost_logs() -> list[SlotLog]:
    """Snapshot of every in-memory cell + branch log to sweep (the sweep awaits,
    and these registries can change under it)."""
    return [*_slot_logs.values(), *(b.log for b in _branches.values())]


async def _cost_backfill_loop() -> None:
    while True:
        try:
            # Enforce the spend cap PER cost, the instant each one settles (via
            # the per-cost hook), not once at the end of the sweep — so a breaching
            # cell pauses the moment its tipping cost returns rather than waiting
            # on the batch's slowest unrelated lookup.
            await llm.backfill_costs(_all_cost_logs, on_priced=_enforce_cap_for_log)
        except Exception:
            pass  # best-effort observability — never let the sweep die
        await asyncio.sleep(_COST_BACKFILL_INTERVAL_S)


async def _enforce_cap_for_log(slot_log: SlotLog) -> None:
    """Cap enforcement for ONE cell, fired by the backfill the instant a new cost
    lands on it. If the freshly-updated settled spend has crossed the cell's cap
    and its pipeline is still live, cancel it and stamp a `run.cap_reached`
    notice. The cap is inherently a SOFT tripwire — cost is only known once
    OpenRouter settles it, so a cell can drift a little past its cap before the
    tipping cost returns.

    Idempotent under the backfill's concurrent per-cost callbacks: `_cancel_task`
    pops the task before its first await and there's no await here before it, so
    once the first callback has cancelled, a racing second one sees no live task
    and no-ops — exactly one notice is ever logged. A completed cell (`run.done`,
    hence not live) is never paused. Branch logs aren't in `_slot_logs`, so they
    never cap — matching the run-spend accounting, which excludes simulations."""
    if SPEND_CAP_USD <= 0:
        return
    events = slot_log.state["events"]
    if not _cap_reached(events):
        return
    key = next((k for k, v in _slot_logs.items() if v is slot_log), None)
    if key is None or not _live(_tasks.get(key)):
        return
    await _cancel_task(*key)
    slot_log.log(
        "run.cap_reached",
        spend=_cell_spend(events),
        cap=_effective_cap(events),
    )


_cell_gates: dict[RunKey, "CellGate"] = {}
# Source cells currently in stepped mode (one LLM call per "step"). Mirrored
# on disk by a `.stepped` marker in each cell dir so the mode SURVIVES a
# server restart — otherwise a stepped run would come back as a plain paused
# cell with no way to advance it.
_stepped_cells: set[RunKey] = set()
# The intent applied to the NEXT gate a cell's `_run` task creates:
#   {"budget": N} — pass N frontier calls without pausing, then pause (a
#     "step" launches a paused cell with budget 1; rerun-step uses it to
#     auto-execute the edited step before pausing on the one after).
#   {"auto": True} — never pause (a stepped cell told to "run rest").
_gate_intents: dict[RunKey, dict[str, object]] = {}


def _stepped_marker(key: RunKey) -> Path:
    return _slot_dir(*key) / ".stepped"


def _set_stepped(key: RunKey, on: bool) -> None:
    """Toggle a cell's stepped mode in memory AND on disk so it persists
    across restarts."""
    marker = _stepped_marker(key)
    if on:
        _stepped_cells.add(key)
        with contextlib.suppress(OSError):
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("")
    else:
        _stepped_cells.discard(key)
        marker.unlink(missing_ok=True)


class CellGate:
    """Pauses a gated pipeline before each real (cache-miss) LLM call until
    the user advances it. `budget` pre-grants pause-free calls (rerun-step
    uses it to auto-execute the edited step). Single-event-loop, so the
    cross-coroutine `Future.set_result` from the endpoint safely wakes the
    paused task."""

    def __init__(self, slot_log: SlotLog, *, budget: int = 0, model_override: str | None = None) -> None:
        self.slot_log = slot_log
        # `budget` is the count of QUEUED steps: each lets the cell pass one
        # gate without pausing. "Step all" grants one to every stepped cell —
        # including ones still mid-call — so a slower model doesn't miss the
        # step and need an individual catch-up later; it consumes the credit
        # when it reaches its next gate, keeping the whole run in lockstep.
        self.budget = budget
        self._fut: asyncio.Future[dict[str, object]] | None = None
        self.auto = False  # once set, the rest of the run proceeds without pausing
        # Fast-forward target: pass every call up to one with this TEMPLATE.
        # `until_before` picks where to stop relative to that call: False (the
        # default) runs THROUGH it and pauses before the NEXT call ("after X");
        # True pauses in front of it, so X itself doesn't execute ("before X").
        self.until_step: str | None = None
        self.until_before: bool = False
        self.pending: dict[str, object] | None = None
        # The call CURRENTLY in flight — released from a pause, or run on a
        # budget credit / in auto. Lets the UI show "running X" instead of the
        # stale last phase once a gated step is released. None until one runs.
        self.current: dict[str, object] | None = None
        # Per-step LLM override (compare's "run this step on model X"): the
        # OpenRouter model id `wait` hands back so `call_llm` re-aims the gated
        # call at it. Sticky until changed — "run rest" + later steps follow the
        # choice — so the same scene context can be A/B'd across models.
        self.model_override: str | None = model_override

    async def wait(
        self, *, node_id: str | None, step: str | None, template: str | None = None,
        system: str, user: str, schema_name: str, model: str,
    ) -> str | None:
        # Mechanical per-object service steps — NOT pipeline reasoning worth a
        # click-through — never gate: they auto-play without pausing or spending a
        # step budget, and aren't re-aimed by a model override. library_match runs
        # on its own gemini model; image_prompt distills a noun phrase for asset
        # generation. So "step" / "step until" / "step sims" run straight through
        # them (a pinned lineage still runs them on its model, already switched to
        # by an earlier gated call).
        if step in ("library_match", "image_prompt"):
            return None
        call = {"node": node_id, "step": step, "template": template, "schema": schema_name, "model": model}
        if self.auto:
            self.current = call
            return self.model_override
        # Seeking a target step: blow past every call until one whose TEMPLATE
        # matches (root/nested variants differ in `template`, not `step`). Where
        # we stop depends on `until_before`:
        #   * after  (False): run THROUGH the target, then pause at the NEXT call.
        #   * before (True):  pause in front of the target — it does NOT execute.
        if self.until_step is not None:
            if (template or step) != self.until_step:
                self.current = call
                return self.model_override  # not the target yet — pass
            # Reached the target: disarm the seek + drop any queued step credits
            # so the breakpoint can't be skipped past.
            self.until_step = None
            self.budget = 0
            if not self.until_before:
                # "after X": run X now (don't pause before it); the NEXT call
                # hits the pause below.
                self.current = call
                return self.model_override
            # "before X": fall through to the pause branch so we stop in front
            # of this call — X only runs on the next explicit step.
        if self.budget > 0:
            self.budget -= 1
            self.current = call
            return self.model_override
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, object]] = loop.create_future()
        self._fut = fut
        self.pending = call
        # Surface the pending call (including the exact prompt about to be
        # sent) on the stream so the UI can preview it.
        self.slot_log.log(
            "branch.step.pending",
            node=node_id,
            step=step,
            template=template,
            schema=schema_name,
            model=model,
            system=system,
            user=user,
        )
        try:
            result = await fut
        finally:
            self._fut = None
            self.pending = None
            self.current = call  # released — this call is now the one in flight
        if result.get("auto"):
            self.auto = True
        return self.model_override

    def proceed(self, *, auto: bool = False) -> bool:
        """Release the current pause. Returns False if nothing is pending."""
        if self._fut is None or self._fut.done():
            return False
        self._fut.set_result({"auto": auto})
        return True


def _live(task: "asyncio.Task[None] | None") -> bool:
    """True while a pipeline task is executing this slot — the running-vs-
    resumable discriminator the status resolver keys on. A finished/cancelled
    task (still parked in the registry) reads False, so status falls back to
    the log."""
    return task is not None and not task.done()


def _gate_awaiting(gate: "CellGate | None") -> bool:
    """True when a step gate is parked before its next call — surfaced as
    paused/awaiting (the awaited step is the gate's `pending`)."""
    return gate is not None and gate.pending is not None


def _cell_status(key: RunKey, slot_log: SlotLog) -> str:
    """Live status of a source cell — its event log refined by its live
    gate (awaiting), task (running), and settled spend vs. its cap (capped)."""
    return rlog.derive_status(
        slot_log.state["events"],
        awaiting=_gate_awaiting(_cell_gates.get(key)),
        live=_live(_tasks.get(key)),
        capped=_cap_reached(slot_log.state["events"]),
    )


def _branch_status(br: "Branch") -> str:
    """Live status of a simulation branch — same resolver, branch-scoped
    gate/task."""
    return rlog.derive_status(
        br.log.state["events"],
        awaiting=_gate_awaiting(br.gate),
        live=_live(br.task),
    )


class RewindRequest(BaseModel):
    to_event_index: int


class CapOverrideRequest(BaseModel):
    # New spend-cap ceiling in USD; 0 (or less) uncaps the cell.
    cap: float


class CreateRunRequest(BaseModel):
    name: str
    # Name of a source prompt version (a subfolder of `versions/`). Copied
    # into the run as its immutable prompt snapshot at creation.
    prompt_version: str


class PromptTestRequest(BaseModel):
    """One-shot re-run of a HISTORICAL step call with edited templates: the
    target `cache.llm` event's logged `variables` are substituted into
    `system_template`/`user_template` and the call re-issued on the event's
    own model. Nothing is logged or cached — the result is display-only."""

    run: str
    slot: str
    model: str
    event_index: int
    step: str
    system_template: str
    user_template: str


class InquiryMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class InquiryRequest(BaseModel):
    """One turn in a persistent "why did the model do this?" conversation about
    a pipeline step. The client assembles and OWNS the reviewer's full system
    prompt — the analyst framing plus that step's exact system / input / output
    / reasoning — and sends it verbatim as `system`, so what the panel shows
    (prefilled and editable) is byte-for-byte what is sent. `messages` is the
    running thread and MUST end with the new user turn. Stateless: the endpoint
    reads no cell state, so it serves source and simulation-branch calls alike."""

    system: str = ""
    messages: list[InquiryMessage]


class BranchSeed(BaseModel):
    """A prompt-test result to pre-commit into a fresh branch's log, so
    "simulate downstream" continues from EXACTLY the output the user vetted
    (instead of re-rolling the edited step). `system`/`user` are the rendered
    bytes the test actually sent."""

    system: str
    user: str
    output: dict[str, object]
    reasoning: str = ""
    tokens_in: int | None = None
    tokens_out: int | None = None


class BranchRequest(BaseModel):
    """Fork the cell into a NEW simulation branch from `event_index` onward —
    which MUST point at a `cache.llm` event of template `step` carrying logged
    `variables`. The branch keeps the events BEFORE that call and re-runs the
    pipeline with the run snapshot + `overrides` — the prompt lab's FULL edit
    set (`{step: {"system": text, "user": text}}`).

    `model` (an alias) PINS the fork to a chosen LLM and runs exactly one step
    on it, then parks — the compare view's per-LLM lineage fork (its prefix is
    the source's committed events, so the pinned step runs against the original
    run's prior output as context). Omit it for the prompt lab's "simulate
    downstream", which pauses at the forked step on the cell's base model."""

    event_index: int
    step: str
    overrides: dict[str, dict[str, str]]
    seed: BranchSeed | None = None
    model: str | None = None
    # Run this branch under a source version instead of the run snapshot — the
    # prompt-set A/B lineage (the current-version lineage omits it).
    version: str | None = None
    # Node ids to LOCK atomic for this fork — the prompt-lab "lock a zone atomic"
    # toggle. Each forces `is_atomic=True` so the zone is a leaf when it re-runs.
    atomic_locks: list[str] = []


class BranchStepRequest(BaseModel):
    """Advance a gated cell/branch. `auto=True` runs the rest to completion;
    `until` (a template id) fast-forwards to the next call of that step —
    pausing AFTER it (through the call, the default) or BEFORE it when
    `until_before=True` (the call doesn't execute). `model` (a model ALIAS)
    re-aims a branch's next gated call at a chosen LLM — compare's per-step
    model A/B — independent of the model that produced the pre-branch scene;
    None keeps the branch's current model."""

    auto: bool = False
    until: str | None = None
    until_before: bool = False
    model: str | None = None


class BranchRewindRequest(BaseModel):
    """Revert a simulation branch to before `to_event_index` (a re-renderable
    branch `cache.llm` call): drop that call and everything after it, delete the
    meshes generated at/after the cut, and pause there. `overrides` (the lab's
    CURRENT edit set) refreshes the branch's templates so the subsequent step
    re-runs under the current snapshot + edits — the sim's source of truth;
    `None` keeps the branch's existing edit set. The source cell and the
    branch's replayed prefix meshes are untouched."""

    to_event_index: int
    overrides: dict[str, dict[str, str]] | None = None


class ForkVersionRequest(BaseModel):
    """Copy an existing source version to a new name — the "start a fresh
    version to iterate on" entry point used by run creation."""

    name: str
    base: str


class UpdateRunPromptsRequest(BaseModel):
    """Write the prompt lab's edits INTO the run's snapshot (the in-place
    iteration loop). Running cells keep the templates they launched with;
    every relaunch/rerun/resume renders the new bytes. With `update_version`
    the run's source version folder is kept in sync too."""

    overrides: dict[str, dict[str, str]]
    update_version: bool = False


class SimulateStepRequest(BaseModel):
    """Fork a SIMULATION branch in each cell at its EARLIEST call of any
    template in `steps`, running from there under the run's (freshly edited)
    snapshot — the non-destructive iteration loop. Source cells are untouched;
    a branch is promoted to source on demand (/branch/commit). Omitted `cells`
    means every cell that has logged a call of one of those steps."""

    steps: list[str]
    cells: list[dict[str, str]] | None = None


class SaveVersionRequest(BaseModel):
    """Persist `base_run`'s prompt snapshot + the prompt lab's edited step
    templates as a brand-new source version folder under `versions/`."""

    name: str
    base_run: str
    overrides: dict[str, dict[str, str]]


class SaveRunRequest(BaseModel):
    """Materialize the prompt lab's simulation as a NEW run: prompts/ =
    `base_run`'s snapshot + `overrides`, and each listed BRANCH's state (events
    + meshes) copied in as its origin cell's history — paused mid-pipeline
    states stay resumable because the new snapshot matches the prompts the
    branch actually ran with. `branches` is a list of branch ids; when two name
    the same origin cell the later one wins."""

    name: str
    base_run: str
    overrides: dict[str, dict[str, str]]
    branches: list[str]
    # Display label for run.json; pass the freshly-saved version name when
    # "save to new version" ran first.
    version_label: str | None = None


class AbTestCell(BaseModel):
    slot: str
    model: str


class AbTestRequest(BaseModel):
    """Launch a NEW run seeded with `source_run`'s ROOT zone plans. Each
    listed (slot, model) cell's log is copied through its root
    `divider.zone_plan` and no further; the cell is then started on
    `prompt_version`, so the resumable divider replays that one committed
    plan verbatim (committed.zone_plan) and re-derives everything below it
    under the new prompts — a prompt A/B that holds the top-level plan fixed
    and varies the whole scene beneath it.

    `include_overall_bbox` extends each cell's copied prefix through the root's
    overall bounding box too, so B holds BOTH the root zone plan AND the scene
    canvas fixed (committed.bbox("root") hits on resume) and varies only what is
    generated inside that fixed box."""

    name: str
    prompt_version: str
    source_run: str
    cells: list[AbTestCell]
    include_overall_bbox: bool = False


class CopySlotRequest(BaseModel):
    """Copy an entire slot folder (all its model cells + meshes) from
    `source_run` into the destination run (the path `run`), OVERWRITING the
    destination's slot dir. The source run is left untouched."""

    source_run: str
    slot: str


class CopyCellRequest(BaseModel):
    """Copy ONE (slot, model) cell — its whole event log + meshes/images — from
    `(source_run, slot, source_model)` into `(run, slot, dest_model)`,
    OVERWRITING the destination cell. The slot is shared (a cell's content is
    scene-specific), but the run and/or model may differ, so this covers cross-
    model within a run, cross-run, or both. The source cell is untouched, and
    the copy keeps the source model on its run.start/cache.llm events so
    cost/usage stay attributed to the model that did the work — only object/
    image URLs are repointed at the destination path."""

    source_run: str
    slot: str
    source_model: str
    dest_model: str


def _root_plan_cut(
    events: list[dict[str, object]], *, through_overall_bbox: bool = False
) -> int | None:
    """Leading-event count to keep so an A/B copy holds a source cell's ROOT
    reasoning fixed and drops everything after — a true prefix, so the copied
    events keep `index == line` and a resumed `log()` continues cleanly.

    Ends at the root `divider.zone_plan` event (+1) by default, so B replays
    that one plan verbatim and re-derives everything below it — including a
    fresh overall bounding box — under its own prompts. With
    `through_overall_bbox` the prefix extends to the root `bbox` event (the
    overall bounding box, which the divider emits right after the root plan) so
    `committed.bbox("root")` hits on resume and B reuses the exact same canvas.

    None when the required event was never committed: no root plan, or — with
    `through_overall_bbox` — a root plan that never reached its bbox."""
    plan_cut: int | None = None
    for i, e in enumerate(events):
        if e.get("kind") == "divider.zone_plan" and e.get("node") == "root":
            plan_cut = i + 1
            break
    if plan_cut is None or not through_overall_bbox:
        return plan_cut
    # The overall bbox is only ever emitted after the root plan, so scanning
    # from the plan cut keeps the kept slice a contiguous prefix.
    for j in range(plan_cut, len(events)):
        e = events[j]
        if e.get("kind") == "bbox" and e.get("id") == "root":
            return j + 1
    return None


def _llm_error_message(e: Exception) -> str:
    """Full detail for an LLM/provider error. OpenRouter SDK errors only
    stringify to the generic top-level message (e.g. "Provider returned
    error"); the actually useful cause — the upstream provider's complaint and
    the request body that tripped it — lives on `e.data.error.metadata` and
    `e.body` (the response body the SDK already read; `raw_response.text` would
    re-trigger a read on a closed streaming response). Fold both into one string
    so every caller (pipeline run.error, the investigator's /inquire) surfaces
    what actually went wrong instead of the opaque top-level message."""
    details: list[str] = []
    data = getattr(e, "data", None)
    err = getattr(data, "error", None) if data is not None else None
    if err is not None:
        metadata = getattr(err, "metadata", None)
        if metadata:
            details.append(f"metadata={metadata}")
    body = getattr(e, "body", None)
    if body:
        details.append(f"body={str(body)[:2000]}")
    suffix = (" | " + " | ".join(details)) if details else ""
    return f"{type(e).__name__}: {e}{suffix}"


def _run_id(run: str, slot_id: str, model_alias: str) -> str:
    """Composite id used as `run_id` in pipeline code (divider, generation,
    threed queue, SlotLog.slot_id). The slashes make it work as a filesystem
    subpath under RUNS_DIR and as an artifact URL segment under /artifacts."""
    return f"{run}/{slot_id}/{model_alias}"


def _run_dir(run: str) -> Path:
    return RUNS_DIR / run


def _slot_dir(run: str, slot_id: str, model_alias: str) -> Path:
    return RUNS_DIR / run / slot_id / model_alias


def _branches_root(run: str) -> Path:
    """The single temp folder holding every active branch of a run."""
    return _run_dir(run) / BRANCHES_SUBDIR


def _branch_dir(run: str, branch_id: str) -> Path:
    return _branches_root(run) / branch_id


def _branch_run_id(run: str, branch_id: str) -> str:
    """Composite run_id for a branch — `<run>/_branches/<branch_id>`, so meshes
    land in `<run>/_branches/<branch_id>/objects/` and every run_id-keyed table
    (`generation._pending`, the Trellis queue) is isolated from every other
    branch and from the source cells."""
    return f"{run}/{BRANCHES_SUBDIR}/{branch_id}"


def _gen_slot_id(run: str, slot_id: str, model_alias: str) -> str:
    """SlotLog id for a cell's generated build — used as the bound log's slot_id
    and the Trellis queue key, so the generated build never collides with the
    library build on either."""
    return f"{_run_id(run, slot_id, model_alias)}::generated"


# Parsed symmetry state per generated-events log, cached on the file's
# (mtime_ns, size) so the frequently-polled gate status re-reads it only when a
# build / regen / un-symmetrize has appended to the build's log.
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


# Parsed prefab star per generated-events log, cached on the file's (mtime_ns,
# size) like the symmetry map, so the frequently-polled gate status re-folds it
# only when a build / regen / link / unlink has appended to the log.
_gen_prefab_cache: dict[Path, tuple[tuple[int, int], dict[str, str]]] = {}


def _generated_prefab(events_path: Path) -> dict[str, str]:
    """Map node id -> its prefab CANONICAL id, folded from the generated log's
    `prefab.match` events (latest per id wins, mirroring `prefabs.resolve_group`).
    A canonical maps to itself; a reuse maps to the canonical whose mesh it shares.
    Ids with no `prefab.match` are absent (callers treat them as their own
    canonical). Lets the gate status tag each mesh with its group so the client can
    show membership and offer link / unlink without re-reading the log per node."""
    try:
        st = events_path.stat()
    except OSError:
        return {}
    sig = (st.st_mtime_ns, st.st_size)
    cached = _gen_prefab_cache.get(events_path)
    if cached is not None and cached[0] == sig:
        return cached[1]
    reuse_of: dict[str, str] = {}
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") != "prefab.match":
                continue
            node_id = event.get("id")
            if isinstance(node_id, str):
                reuse_of[node_id] = str(event.get("reuse_id") or "")
    canonical_of = {nid: (reuse_of.get(nid) or nid) for nid in reuse_of}
    _gen_prefab_cache[events_path] = (sig, canonical_of)
    return canonical_of


_gen_image_prompt_cache: dict[Path, tuple[tuple[int, int], dict[str, str]]] = {}


def _generated_image_prompts(events_path: Path) -> dict[str, str]:
    """Map node id -> the subject phrase used for its reference-image generation,
    folded from the generated log's `image` events (latest per id wins). Only
    canonicals have their own `image` event; callers resolve reuses to their
    canonical. Cached on the file's (mtime_ns, size)."""
    try:
        st = events_path.stat()
    except OSError:
        return {}
    sig = (st.st_mtime_ns, st.st_size)
    cached = _gen_image_prompt_cache.get(events_path)
    if cached is not None and cached[0] == sig:
        return cached[1]
    prompts: dict[str, str] = {}
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") != "image":
                continue
            node_id = event.get("id")
            prompt = event.get("prompt")
            if isinstance(node_id, str) and isinstance(prompt, str):
                prompts[node_id] = prompt
    _gen_image_prompt_cache[events_path] = (sig, prompts)
    return prompts


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


def _slug(text: str) -> str:
    """A filesystem/URL-safe lowercase slug — branch ids embed the origin
    slot + model so the temp folder reads at a glance."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "x"


def _new_branch_id(slot_id: str, model_alias: str) -> str:
    """A unique, readable branch id: `<slot>-<model>-<token>`. The random
    token keeps several forks of one cell (different zones, or A/B children)
    collision-free; the slug prefix makes the temp folder self-explaining."""
    while True:
        bid = f"{_slug(slot_id)}-{_slug(model_alias)}-{secrets.token_hex(3)}"
        if bid not in _branches:
            return bid


def _require_branch(branch_id: str) -> Branch:
    br = _branches.get(branch_id)
    if br is None:
        raise HTTPException(status_code=404, detail=f"no such branch: {branch_id}")
    return br


def _branch_manifest(br: Branch) -> dict[str, object]:
    return {
        "id": br.id,
        "slot": br.slot,
        "model": br.model,
        "parent": br.parent,
        "fork_index": br.fork_index,
        "overrides": br.overrides,
        "model_pin": br.model_pin,
        "version": br.version,
        "atomic_locks": br.atomic_locks,
    }


def _write_branch_manifest(br: Branch) -> None:
    """Persist the branch's identity to `branch.json` so the temp folder is
    self-describing — enumerable + rehydratable after a restart."""
    path = _branch_dir(br.run, br.id) / BRANCH_MANIFEST
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_branch_manifest(br)) + "\n", encoding="utf-8")


def _branch_summary(br: Branch) -> dict[str, object]:
    """The branch state the client polls: its identity (id / origin cell /
    parent / fork index) plus where it is and what it's awaiting at its gate."""
    bevents = br.log.state.get("events", [])
    gate = br.gate
    return {
        "id": br.id,
        "run": br.run,
        "slot": br.slot,
        "model": br.model,
        "parent": br.parent,
        "fork_index": br.fork_index,
        # The alias this branch's steps are pinned to (compare's per-LLM
        # lineage), or None when it runs on the cell's base model.
        "pin": next((a for a, m in MODELS.items() if m == br.model_pin), None),
        # The source version this lineage runs under (the prompt-set A/B), or
        # None for the run's current prompts.
        "version": br.version,
        "status": _branch_status(br),
        "events_count": len(bevents),
        "last_step": _last_step(bevents),
        # The LLM call waiting for the user's go-ahead, when gated.
        "pending": gate.pending if gate is not None else None,
        # The call currently in flight (released/running), so the UI shows it
        # instead of the stale last phase while it runs.
        "current": gate.current if gate is not None else None,
        "auto": gate.auto if gate is not None else False,
    }


def _validate_overrides(overrides: dict[str, dict[str, str]]) -> None:
    """Reject an override set naming an unknown step or template role."""
    for ostep, roles in overrides.items():
        if ostep not in prompt_store.STEPS:
            raise HTTPException(status_code=400, detail=f"unknown override step: {ostep}")
        for role in roles:
            if role not in ("system", "user"):
                raise HTTPException(status_code=400, detail=f"unknown template role: {role}")


def _resolve_run(run: str | None) -> str:
    """Every cell endpoint names its target run explicitly so concurrently-
    running runs never route through a shared global. The client always sends
    `?run=`; `_current_run` is only the fallback for a client that hasn't picked
    one yet (boot, or a legacy caller). Resolving also lazily hydrates the target
    run if it exists on disk but isn't yet in memory, so a cell is never
    spuriously 404/idle just because its run wasn't the one explicitly activated."""
    resolved = run or _current_run
    _ensure_run_hydrated(resolved)
    return resolved


def _run_meta(run: str) -> dict[str, object]:
    """The run's `run.json` (prompt version name, created_at), or {} for runs
    that predate the file-based prompt versioning."""
    path = _run_dir(run) / RUN_META_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _require_run_prompts(run: str) -> prompt_store.PromptSet:
    """The run's prompt snapshot, required for anything that (re)starts
    pipeline work. Runs created before the file-based prompt versioning have
    no snapshot — they stay loadable (scene/meshes/events) but can't run."""
    run_dir = _run_dir(run)
    if not prompt_store.has_run_prompts(run_dir):
        raise HTTPException(
            status_code=409,
            detail=f"run {run!r} has no prompt snapshot (legacy run) — create a new run to launch pipelines",
        )
    try:
        return prompt_store.load_run_prompts(run_dir)
    except prompt_store.PromptTemplateError as e:
        raise HTTPException(status_code=409, detail=str(e))


def _require_step_event(
    run: str, slot_id: str, model_alias: str, event_index: int, step: str,
) -> tuple[SlotLog, dict[str, object]]:
    """The cell's `cache.llm` event at `event_index`, validated as a
    re-renderable call of template `step` (the prompt lab's eligibility rule:
    same template, already ran, variables logged)."""
    if step not in prompt_store.STEPS:
        raise HTTPException(status_code=404, detail=f"unknown step: {step}")
    slot_log = _require_slot_log(run, slot_id, model_alias)
    events = slot_log.state["events"]
    if not (0 <= event_index < len(events)):
        raise HTTPException(status_code=400, detail="event_index out of range")
    event = events[event_index]
    if event.get("kind") != "cache.llm":
        raise HTTPException(status_code=400, detail="event is not an LLM call")
    if event.get("template") != step:
        raise HTTPException(
            status_code=400,
            detail=f"event is a {event.get('template') or event.get('step')!r} call, not {step!r}",
        )
    if not isinstance(event.get("variables"), dict):
        raise HTTPException(
            status_code=409,
            detail="event predates variable logging — re-run the cell to make it testable",
        )
    return slot_log, event


# --- reviewer chat ----------------------------------------------------------
#
# The stateless reviewer behind the scene investigator: a fixed strong model so
# analysis quality is constant across whatever subject model is being
# benchmarked. The full system prompt (analyst framing + scene grounding + step
# timeline + any attached steps) is assembled CLIENT-SIDE and passed in `system`;
# this endpoint just pins the reviewer model and forwards the conversation. The
# reviewer is Claude Opus 4.8 at xhigh reasoning regardless of the subject model.
INQUIRY_MODEL = "anthropic/claude-opus-4.8"


# --- prompt inspector -------------------------------------------------------
#
# Reconstructs the prompts a run actually used straight from its event logs.
# (Runs launched after the snapshot-first change pin + bind their effective
# prompt module at first launch, so their snapshots ARE authoritative — but
# legacy runs either lack snapshots or were never bound to them, so every
# `cache.llm` event carrying the exact `system` + `user` text per `step`
# remains the universal ground truth.)

# Canonical pipeline-step display order; unlisted steps sort after these.
_PROMPT_STEP_ORDER = [
    "zone_plan",
    "overall_bbox",
    "zone_decompose",
    "child_bbox_batch",
    "encapsulating_decompose",
    "anchor_decompose",
    "object_decomp",
    "object_bbox_batch",
    "next_object",
    "negative_space_decompose",
    "image_prompt",
    "library_match",
]

# Per-cell extraction cache: events.jsonl path -> (mtime, {step: {...}}). A
# single cell log can be 100MB+, so we scan it at most once per modification.
_CELL_PROMPT_CACHE: dict[Path, tuple[float, dict[str, dict]]] = {}

_PROMPT_TOKEN_RE = re.compile(r"\s+|\S+")
# The top-level `"step"` key is logged BEFORE the megabyte system/user/output
# fields, so the first match in a line's head is always the real step. Lets us
# decide whether to skip a line without a full json.loads of its giant payload.
_STEP_PEEK_RE = re.compile(r'"step":\s*"([a-z_]+)"')
# Fully parse only the first few calls of each step per cell. The system prompt
# is a module constant (identical across a step's calls; the sole exception is
# zone_plan's root-vs-non-root pair, both seen within the first 2 calls), and
# the representative user is the FIRST call — so calls beyond this add nothing
# but parse cost. Hundreds of repeated object_bbox_batch/image_prompt/
# library_match payloads are skipped after the cap.
_STEP_PARSE_CAP = 4


def _extract_cell_prompts(events_path: Path) -> dict[str, dict]:
    """Reconstruct one cell's prompts from its `cache.llm` events.

    Returns `{step: {"systems": {system_text: call_count}, "first_user": str,
    "first_node": str | None, "first_index": int}}`. The `first_*` fields are
    the EARLIEST call of that step (lowest event index) — a deterministic
    representative for the (context-bearing) user prompt. Cached by mtime."""
    try:
        st = events_path.stat()
    except OSError:
        return {}
    cached = _CELL_PROMPT_CACHE.get(events_path)
    if cached is not None and cached[0] == st.st_mtime:
        return cached[1]
    out: dict[str, dict] = {}
    parsed_per_step: dict[str, int] = {}
    try:
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                # cache.llm lines are the only ones carrying prompts; the
                # substring pre-filter skips the cheap (bbox/image/model/...)
                # lines outright. Correctness is still gated on the parsed
                # `kind` below, so a false-positive is harmless.
                if '"kind": "cache.llm"' not in line:
                    continue
                # Peek at the step near the line head; once we've parsed the
                # cap for that step, skip the (huge) payload without decoding.
                peek = _STEP_PEEK_RE.search(line, 0, 400)
                if peek is not None and parsed_per_step.get(peek.group(1), 0) >= _STEP_PARSE_CAP:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("kind") != "cache.llm":
                    continue
                step = e.get("step")
                if not isinstance(step, str):
                    continue
                parsed_per_step[step] = parsed_per_step.get(step, 0) + 1
                system = e.get("system") if isinstance(e.get("system"), str) else ""
                user = e.get("user") if isinstance(e.get("user"), str) else ""
                idx = e.get("index")
                idx = idx if isinstance(idx, int) else (1 << 62)
                slot = out.setdefault(
                    step,
                    {"systems": {}, "first_user": "", "first_node": None, "first_index": 1 << 62},
                )
                slot["systems"][system] = slot["systems"].get(system, 0) + 1
                if idx < slot["first_index"]:
                    slot["first_index"] = idx
                    slot["first_user"] = user
                    slot["first_node"] = e.get("node")
    except OSError:
        return {}
    _CELL_PROMPT_CACHE[events_path] = (st.st_mtime, out)
    return out


def _run_cells_with_data(run: str) -> list[tuple[str, Path, float]]:
    """`(cell_key, events_path, created)` for every cell under this run whose
    log is non-empty. `cell_key` is `"<slot>/<model>"` — stable across runs so
    the diff can pick a reference cell present in both runs. `created` is the
    events.jsonl CREATION time (st_birthtime; falls back to mtime on the rare
    platform without it) — i.e. when that cell was first launched."""
    run_dir = _run_dir(run)
    cells: list[tuple[str, Path, float]] = []
    if not run_dir.is_dir():
        return cells
    for ev in sorted(run_dir.glob("*/*/events.jsonl")):
        try:
            st = ev.stat()
        except OSError:
            continue
        if st.st_size <= 0:
            continue
        rel = ev.relative_to(run_dir).parts  # (slot, model, "events.jsonl")
        # The `_branches/<id>/events.jsonl` temp logs also match `*/*/…`; only
        # real slot cells count, so the prompt inspector / run-date scan never
        # mistakes a live branch for a phantom cell.
        if len(rel) >= 2 and rel[0] in SLOTS_BY_ID:
            created = getattr(st, "st_birthtime", st.st_mtime)
            cells.append((f"{rel[0]}/{rel[1]}", ev, created))
    return cells


def _run_first_launch(run: str) -> float | None:
    """Canonical chronological date for a run: its OLDEST SLOT's first-launch.

    Keyed on each cell's CREATION time (st_birthtime), NOT mtime. A cell's
    events.jsonl is created the moment that cell is first launched, and
    creation time never moves afterward — so it is immune to the later writes
    (re-runs, resumes, rebakes, manual log edits, folder copies/renames) that
    make mtime and folder timestamps drift. A slot (scene) is first launched at
    its earliest cell creation; the run's date is the oldest such slot. None
    when the run has no rendition yet."""
    slot_dates: dict[str, float] = {}
    for cell_key, _ev, created in _run_cells_with_data(run):
        slot = cell_key.split("/", 1)[0]
        slot_dates[slot] = min(slot_dates.get(slot, created), created)
    return min(slot_dates.values()) if slot_dates else None


def _extract_run_prompts(run: str, ref_cell_key: str | None = None) -> dict[str, dict]:
    """Union the per-cell extractions for a run into
    `{step: {"systems": {text: count}, "by_cell": {cell_key: {"user", "node"}}}}`.

    Cells are scanned with the reference cell (for diffable user prompts) FIRST,
    then smallest-first; scanning stops once the structural pipeline is covered
    (`zone_decompose` seen) and the reference cell has been read. Since a step's
    system prompt is identical across cells, this avoids reading every cell's
    multi-hundred-MB log just to re-collect the same per-step systems."""
    def _size(p: Path) -> int:
        try:
            return p.stat().st_size
        except OSError:
            return 0

    cells = _run_cells_with_data(run)
    # Reference cell first (so its user prompts are always captured for the
    # diff), then ascending size so the cheapest cell completes coverage.
    cells.sort(key=lambda c: (c[0] != ref_cell_key, _size(c[1])))

    merged: dict[str, dict] = {}
    seen_steps: set[str] = set()
    ref_scanned = ref_cell_key is None
    for cell_key, ev, _m in cells:
        for step, info in _extract_cell_prompts(ev).items():
            m = merged.setdefault(step, {"systems": {}, "by_cell": {}})
            for sys_text, cnt in info["systems"].items():
                m["systems"][sys_text] = m["systems"].get(sys_text, 0) + cnt
            m["by_cell"][cell_key] = {"user": info["first_user"], "node": info["first_node"]}
            seen_steps.add(step)
        if cell_key == ref_cell_key:
            ref_scanned = True
        # A cell that ran `zone_decompose` exercised the full divider+generation
        # pipeline, so every step's system is already captured. Stop there
        # (once the reference cell is in) instead of scanning the rest.
        if ref_scanned and "zone_decompose" in seen_steps:
            break
    return merged


def _dominant_system(systems: dict[str, int]) -> str:
    """The system prompt used by the most calls — a stable, meaningful diff
    target when a step has >1 variant (only `zone_plan`, root vs non-root)."""
    if not systems:
        return ""
    return max(systems.items(), key=lambda kv: (kv[1], kv[0]))[0]


# Cost guards: word-level diff is O(n·m), so a fully-rewritten 100KB user
# prompt (full scene context) would hang the request. We diff at LINE level
# first (few hundred entries) and refine only the small changed hunks to word
# level; anything past these caps degrades to a whole-block replace.
_DIFF_WORD_CAP = 2500
_DIFF_LINE_CAP = 8000


def _word_segments(old: str, new: str) -> list[dict[str, str]]:
    a = _PROMPT_TOKEN_RE.findall(old)
    b = _PROMPT_TOKEN_RE.findall(new)
    if len(a) > _DIFF_WORD_CAP or len(b) > _DIFF_WORD_CAP:
        out: list[dict[str, str]] = []
        if old:
            out.append({"op": "delete", "text": old})
        if new:
            out.append({"op": "insert", "text": new})
        return out
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append({"op": "equal", "text": "".join(a[i1:i2])})
        elif tag == "delete":
            out.append({"op": "delete", "text": "".join(a[i1:i2])})
        elif tag == "insert":
            out.append({"op": "insert", "text": "".join(b[j1:j2])})
        else:  # replace
            out.append({"op": "delete", "text": "".join(a[i1:i2])})
            out.append({"op": "insert", "text": "".join(b[j1:j2])})
    return out


def _prompt_diff_segments(old: str, new: str) -> list[dict[str, str]]:
    """Diff two prompts into ordered `{op, text}` segments (op ∈
    equal/insert/delete). Line-level structure with word-level refinement of
    changed hunks: precise highlighting of the exact words that changed inside a
    long prose block, while staying linear in the common (mostly-equal) case."""
    a = (old or "").splitlines(keepends=True)
    b = (new or "").splitlines(keepends=True)
    if len(a) > _DIFF_LINE_CAP or len(b) > _DIFF_LINE_CAP:
        return _word_segments(old or "", new or "")
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    segs: list[dict[str, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            segs.append({"op": "equal", "text": "".join(a[i1:i2])})
        elif tag == "delete":
            segs.append({"op": "delete", "text": "".join(a[i1:i2])})
        elif tag == "insert":
            segs.append({"op": "insert", "text": "".join(b[j1:j2])})
        else:  # replace — refine the (usually small) changed block to words
            segs.extend(_word_segments("".join(a[i1:i2]), "".join(b[j1:j2])))
    return segs


def _pick_ref_cell(cur_cells: dict, prev_cells: dict) -> tuple[str | None, bool]:
    """Choose the reference cell for a step's representative user prompt:
    the lexicographically-first cell present in BOTH runs (so the user diff
    compares the same scene+model+call), else the first current-run cell
    (display only, no diff)."""
    shared = sorted(set(cur_cells) & set(prev_cells))
    if shared:
        return shared[0], True
    cur_only = sorted(cur_cells)
    return (cur_only[0], False) if cur_only else (None, False)


def _build_run_prompts(run: str, compare: str | None) -> dict[str, object]:
    """Assemble the prompt-inspector payload for `run`, diffed against
    `compare` (the chronologically-previous run) when given."""
    # Pick the shared reference cell up front (from directory listing alone, no
    # scanning) so both runs extract the SAME cell's user prompts — keeping the
    # representative user diff apples-to-apples.
    cur_keys = {k for k, _ev, _m in _run_cells_with_data(run)}
    prev_keys = {k for k, _ev, _m in _run_cells_with_data(compare)} if compare else set()
    shared = sorted(cur_keys & prev_keys)
    ref_cell = shared[0] if shared else None

    cur = _extract_run_prompts(run, ref_cell)
    prev = _extract_run_prompts(compare, ref_cell) if compare else {}
    steps_out: list[dict[str, object]] = []
    order = {s: i for i, s in enumerate(_PROMPT_STEP_ORDER)}
    for step in sorted(cur, key=lambda s: (order.get(s, len(order)), s)):
        cur_info = cur[step]
        prev_info = prev.get(step)
        cur_sys = _dominant_system(cur_info["systems"])
        prev_sys = _dominant_system(prev_info["systems"]) if prev_info else None
        sys_changed = prev_sys is not None and prev_sys != cur_sys
        system_block: dict[str, object] = {
            "text": cur_sys,
            "variant_count": len(cur_info["systems"]),
            "changed": sys_changed,
            "diff": _prompt_diff_segments(prev_sys or "", cur_sys) if sys_changed else None,
        }
        user_block: dict[str, object] | None = None
        ref_cell, shared = _pick_ref_cell(
            cur_info["by_cell"], prev_info["by_cell"] if prev_info else {}
        )
        if ref_cell is not None:
            cu = cur_info["by_cell"][ref_cell]
            node = cu["node"]
            cur_user = cu["user"]
            user_changed = False
            diff = None
            if shared and prev_info is not None:
                pu = prev_info["by_cell"][ref_cell]["user"]
                if pu != cur_user:
                    user_changed = True
                    diff = _prompt_diff_segments(pu, cur_user)
            user_block = {
                "text": cur_user,
                "ref": ref_cell + (f"  ·  node={node}" if node else ""),
                # The first call of a root-anchored step is node="root", whose
                # user prompt has no upstream context — so its diff is template-
                # only. Leaf steps carry scene context, so flag the diff as mixed.
                "root_anchored": node == "root",
                "diffable": shared,
                "changed": user_changed,
                "diff": diff,
            }
        steps_out.append({"step": step, "system": system_block, "user": user_block})
    return {"run": run, "compare": compare, "steps": steps_out}


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
            # Restore stepped mode from its on-disk marker so a stepped run
            # comes back steppable after a restart instead of a dead paused cell.
            if (slot_dir / ".stepped").exists():
                _stepped_cells.add((run, slot.id, alias))
            _maybe_launch(slot, alias, slot_log)
    _hydrate_branches(run)
    _hydrated_runs.add(run)


def _hydrate_branches(run: str) -> None:
    """Rebuild the in-memory branch registry from the run's self-describing
    `_branches/<id>/branch.json` manifests, so the temp folder's TOP-LEVEL sims
    survive a server restart — coming back paused/resumable (no live task/gate)
    exactly like source cells. Transient fan-out children (a `parent` set) are
    stale on restart, so their dirs are swept instead of revived."""
    root = _branches_root(run)
    if not root.is_dir():
        return
    for bdir in sorted(root.iterdir()):
        if not bdir.is_dir() or bdir.name in _branches:
            continue
        manifest = bdir / BRANCH_MANIFEST
        events_path = bdir / "events.jsonl"
        if not manifest.is_file() or not events_path.is_file():
            shutil.rmtree(bdir, ignore_errors=True)
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if not isinstance(data, dict) or data.get("parent"):
            shutil.rmtree(bdir, ignore_errors=True)
            continue
        blog = SlotLog(_branch_run_id(run, bdir.name), events_path)
        blog.hydrate_from_disk()
        # No status fix-up: with no live task/gate, derive_status reports a
        # non-terminal branch log as paused (resumable) automatically.
        _branches[bdir.name] = Branch(
            id=bdir.name, run=run,
            slot=str(data.get("slot", "")),
            model=str(data.get("model", "")),
            parent=None,
            fork_index=int(data.get("fork_index", 0) or 0),
            overrides=data["overrides"] if isinstance(data.get("overrides"), dict) else {},
            log=blog,
            model_pin=data.get("model_pin") if isinstance(data.get("model_pin"), str) else None,
            version=data.get("version") if isinstance(data.get("version"), str) else None,
            atomic_locks=(list(data["atomic_locks"]) if isinstance(data.get("atomic_locks"), list) else []),
        )


def _placed_count(mesh_dir: Path) -> int:
    """Count placed `<id>.glb` in `mesh_dir` (the served meshes), excluding the
    `<id>.raw.glb` pre-placement intermediates. 0 if the dir is absent."""
    if not mesh_dir.is_dir():
        return 0
    return sum(1 for p in mesh_dir.glob("*.glb") if not p.name.endswith(".raw.glb"))


# Generated-build asset variants the scene viewer + splat can select between.
_GEN_VARIANTS = ("raw", "lite", "optimized")
# Splat mesh-source overrides (persisted per cell in splat/source.json).
_SPLAT_SOURCE_CHOICES = ("generated", "generated-lite", "generated-optimized", "library")


def _generated_variant_dir(rid: str, variant: str) -> Path:
    """The generated build dir for a variant: raw (objects-generated), lite
    (objects-generated-lite), or optimized (objects-generated-optimized)."""
    raw_dir, opt_dir = generation.latest_generated_dirs(RUNS_DIR, rid)
    if variant == "raw":
        return raw_dir
    if variant == "lite":
        return raw_dir.parent / generation.GENERATED_LITE_SUBDIR
    return opt_dir


def _resolve_gen_variant(variant: str | None, optimized: bool) -> str:
    """Normalize the client's asset-variant selection: the explicit `variant`
    (raw/lite/optimized) when valid, else the legacy `optimized` bool."""
    if variant in _GEN_VARIANTS:
        return variant  # type: ignore[return-value]
    return "optimized" if optimized else "raw"


def _splat_source_pref_path(run: str, slot: str, model: str) -> Path:
    return _slot_dir(run, slot, model) / "splat" / "source.json"


def _splat_source_pref(run: str, slot: str, model: str) -> str | None:
    """The persisted splat source override for a cell, or None (auto)."""
    try:
        val = json.loads(
            _splat_source_pref_path(run, slot, model).read_text(encoding="utf-8")
        ).get("source")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    return val if val in _SPLAT_SOURCE_CHOICES else None


def _splat_source(
    run: str, slot: str, model: str, source: str | None = None,
) -> tuple[Path, str] | None:
    """The splat mesh source for a cell + its kind, or None if the cell has no placed
    meshes in any build. A `source` (arg, else the cell's persisted override) pins the
    build; when absent or not on disk it AUTO-picks the OPTIMIZED (decimated) twin — the
    intended splat sample/render source, since the raw build's multi-million-triangle
    meshes are far too heavy for nvdiffrast (Stage 5). Auto order:
      1. `objects-generated-optimized/` — the generated optimized twin (KTX2/Meshopt);
      2. the asset-library build (`objects-optimized/` / `objects/`, KTX2/Meshopt);
      3. `objects-generated/` — the raw vanilla build, only as a last resort.
    Everything but the raw build is KTX2/Meshopt and is de-optimized to vanilla first
    (see `_deoptimize_dir`); only kind `"generated"` skips that step. Disk-only; never
    hydrates the run."""
    rid = _run_id(run, slot, model)
    raw_dir, opt_dir = generation.latest_generated_dirs(RUNS_DIR, rid)
    lite_dir = raw_dir.parent / generation.GENERATED_LITE_SUBDIR
    cell_dir = _slot_dir(run, slot, model)

    if source is None:
        source = _splat_source_pref(run, slot, model)
    if source is not None:
        pinned = {
            "generated-lite": (lite_dir, "generated-lite"),
            "generated-optimized": (opt_dir, "generated-optimized"),
            "generated": (raw_dir, "generated"),
        }.get(source)
        if pinned is not None and _placed_count(pinned[0]) > 0:
            return pinned
        if source == "library":
            for name in ("objects-optimized", OBJECTS_SUBDIR, "objects"):
                lib_dir = cell_dir / name
                if _placed_count(lib_dir) > 0:
                    return lib_dir, "library"
        # requested build isn't on disk — fall through to the auto order below.

    if _placed_count(opt_dir) > 0:
        return opt_dir, "generated-optimized"
    for name in ("objects-optimized", OBJECTS_SUBDIR, "objects"):
        lib_dir = cell_dir / name
        if _placed_count(lib_dir) > 0:
            return lib_dir, "library"
    if _placed_count(raw_dir) > 0:
        return raw_dir, "generated"
    return None


async def _run_lite_build(
    run: str, slot: str, model: str, src_dir: Path, out_dir: Path,
) -> None:
    """Background: build the cell's lite tier (src_dir -> out_dir) via
    build_lite_assets.py, streaming its "[i/N]" progress into the job dict."""
    import re
    import sys

    key = (run, slot, model)
    job = _lite_build_jobs[key]
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", str(_BUILD_LITE_SCRIPT),
            "--src-dir", str(src_dir), "--out-dir", str(out_dir),
            cwd=str(_REPO_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        async for chunk in proc.stdout:
            line = chunk.decode(errors="replace").strip()
            if not line:
                continue
            m = re.search(r"\[(\d+)/(\d+)\]", line)
            if m:
                job["done"] = int(m.group(1))
                job["total"] = int(m.group(2))
            job["status"] = line[:200]
        rc = await proc.wait()
        job["ok"] = rc == 0
        job["status"] = "done" if rc == 0 else f"failed (exit {rc})"
    except Exception as exc:  # noqa: BLE001 - surface to the status poller
        job["ok"] = False
        job["error"] = str(exc)
        job["status"] = "error"
    finally:
        job["running"] = False
        job["finished_at"] = datetime.now().isoformat(timespec="seconds")


def _discover_splat_stage1_cells(run: str) -> list[tuple[str, str, int, str]]:
    """(slot, model, placed_count, source_kind) for every cell in `run` that has
    placed meshes in either build — the cells Stage 1 can convert. `source_kind`
    is 'generated' (vanilla raw build) or 'library' (KTX2/Meshopt, de-optimized
    on convert). Disk-only; never hydrates the run."""
    out: list[tuple[str, str, int, str]] = []
    for slot in SLOTS:
        for alias in MODEL_ALIASES:
            source = _splat_source(run, slot.id, alias)
            if source is None:
                continue
            src_dir, kind = source
            out.append((slot.id, alias, _placed_count(src_dir), kind))
    return out


def _splat_cell_status(
    run: str,
    slot: str,
    model: str,
    placed: int | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Public Stage-1 state for one cell: the live job if one exists, else 'done'
    with the summary read back from an existing manifest, else 'idle'. `placed`
    (from a disk scan) seeds the progress denominator before a job starts;
    `source` ('generated' / 'library') is the build it converts from."""
    job = _splat_stage1_jobs.get((run, slot, model))
    if job is not None:
        return dict(job)
    manifest = _slot_dir(run, slot, model) / "splat" / splat_stage1.MANIFEST_NAME
    if manifest.exists():
        summary: Any = None
        with contextlib.suppress(Exception):
            summary = json.loads(manifest.read_text(encoding="utf-8")).get("summary")
        total = (summary or {}).get("counts", {}).get("placed", placed or 0)
        return {
            "run": run, "slot": slot, "model": model, "source": source,
            "total": total, "done": total, "running": False,
            "status": "done", "current_id": None, "error": None, "summary": summary,
        }
    return {
        "run": run, "slot": slot, "model": model, "source": source,
        "total": placed or 0, "done": 0, "running": False,
        "status": "idle", "current_id": None, "error": None, "summary": None,
    }


def _deoptimize_dir(src_dir: Path, out_dir: Path) -> None:
    """Rewrite every placed GLB in `src_dir` (a KTX2/Meshopt library build) to
    vanilla glTF in `out_dir` via the Node de-optimizer, so the trimesh Stage-1
    assembler can read them. Blocking (subprocess) — call off the event loop."""
    result = subprocess.run(
        [
            _NODE_BIN, str(_DEOPT_SCRIPT),
            "--in-dir", str(src_dir), "--out-dir", str(out_dir),
        ],
        cwd=str(_DEOPT_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[:500]
        raise RuntimeError(f"de-optimize failed ({result.returncode}): {detail}")


def _assemble_cell_from_source(
    run: str,
    slot: str,
    model: str,
    src_dir: Path,
    kind: str,
    events_path: Path,
    out_path: Path,
    progress: splat_stage1.ProgressCb,
) -> dict[str, Any]:
    """Run Stage 1 for one cell. A 'generated' source is vanilla and assembled
    directly; a 'library' source is KTX2/Meshopt, so it's de-optimized into a
    throwaway temp dir first (the on-disk originals are untouched). Blocking —
    run via asyncio.to_thread."""
    if kind == "generated":
        return splat_stage1.assemble_cell(
            run=run, slot=slot, model=model, raw_dir=src_dir,
            events_path=events_path, out_path=out_path, runs_dir=RUNS_DIR,
            progress=progress,
        )
    # Temp on the runs volume (not the system temp, which may be a small/full
    # tmpfs) — mirrors scripts/export_scene_usd.py; auto-removed on exit.
    with tempfile.TemporaryDirectory(prefix="splat-deopt-", dir=str(RUNS_DIR)) as tmp:
        vanilla_dir = Path(tmp)
        _deoptimize_dir(src_dir, vanilla_dir)
        return splat_stage1.assemble_cell(
            run=run, slot=slot, model=model, raw_dir=vanilla_dir,
            events_path=events_path, out_path=out_path, runs_dir=RUNS_DIR,
            progress=progress,
        )


async def _run_splat_stage1_cell(run: str, slot: str, model: str) -> None:
    """Convert ONE cell: the blocking work (optional library de-optimize +
    trimesh assembly) runs off the event loop via a thread, and per-object
    progress lands in the cell's job dict that the status endpoints serve."""
    key = (run, slot, model)
    job = _splat_stage1_jobs[key]
    try:
        source = _splat_source(run, slot, model)
        if source is None:
            raise FileNotFoundError(f"no convertible build for {slot}/{model} in {run}")
        src_dir, kind = source
        events_path = _slot_dir(run, slot, model) / "events.jsonl"
        out_path = _slot_dir(run, slot, model) / "splat" / splat_stage1.MANIFEST_NAME

        def _progress(done: int, total: int, current: str) -> None:
            job["done"], job["total"], job["current_id"] = done, total, current

        job["summary"] = await asyncio.to_thread(
            _assemble_cell_from_source,
            run, slot, model, src_dir, kind, events_path, out_path, _progress,
        )
        job["status"] = "done"
        job["current_id"] = None
    except Exception as exc:
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        job["running"] = False
        job["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _splat_stage1_tasks.pop(key, None)


class Stage2Request(BaseModel):
    """Free-space voxelizer knobs (Stage 2 — the shared spatial foundation). All
    optional; omitted → defaults. `pitch` (m) is the coarse navigational edge,
    `refine` sets the fine occupancy scale (pitch/refine), `margin` (m) grows the
    exterior play volume."""

    pitch: float | None = None
    refine: int | None = None
    margin: float | None = None


def _stage2_params(req: Stage2Request | None) -> splat_stage2.FreeSpaceParams:
    """Clamp a request into FreeSpaceParams; omitted fields keep the defaults."""
    d = splat_stage2.FreeSpaceParams()
    if req is None:
        return d
    return splat_stage2.FreeSpaceParams(
        pitch=float(min(max(req.pitch, 0.02), 1.0)) if req.pitch is not None else d.pitch,
        refine=int(min(max(req.refine, 1), 6)) if req.refine is not None else d.refine,
        margin=float(min(max(req.margin, 0.0), 10.0)) if req.margin is not None else d.margin,
    )


def _cloud_path(run: str, slot: str, model: str) -> Path:
    """Where a cell's Stage-3 base Gaussian cloud lives (`splat/cloud.ply`)."""
    return _slot_dir(run, slot, model) / "splat" / splat_stage3.CLOUD_NAME


def _detail_cloud_path(run: str, slot: str, model: str) -> Path:
    """The optional denser LOD streamed in behind the base (`cloud.detail.ply`)."""
    return _cloud_path(run, slot, model).with_suffix(".detail.ply")


def _trained_path(run: str, slot: str, model: str) -> Path:
    """Where a cell's Stage-6 fine-tuned splat lives (`splat/trained.ply`)."""
    return _slot_dir(run, slot, model) / "splat" / splat_stage6.TRAINED_NAME


def _stage6_log_path(run: str, slot: str, model: str) -> Path:
    """Where the Stage-6 training process streams its stdout/stderr
    (`splat/stage6.log`) — also served under `/artifacts` for live viewing."""
    return _slot_dir(run, slot, model) / "splat" / "stage6.log"


def _stage6_ckpt_dir(run: str, slot: str, model: str) -> Path:
    """Where Stage-6 resumable checkpoints live (`splat/ckpt/`, see splat/stage6.py).
    Present (with no trained.ply) means an interrupted run can be resumed."""
    return _slot_dir(run, slot, model) / "splat" / splat_stage6.CKPT_DIRNAME


def _latest_ckpt_step(ckpt_dir: Path) -> int | None:
    """Highest checkpoint step on disk (from the `step_NNNNNN.pt` names), or None."""
    if not ckpt_dir.is_dir():
        return None
    steps = [
        int(p.stem.split("_")[1])
        for p in ckpt_dir.glob("step_*.pt")
        if "_" in p.stem and p.stem.split("_")[-1].isdigit()
    ]
    return max(steps) if steps else None


def _tail_text(path: Path, max_bytes: int = 8192) -> str | None:
    """The last `max_bytes` of a file, decoded leniently, or None if absent — enough
    to show a live training tail without shipping a multi-MB log on every poll."""
    if not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - max_bytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return None


def _freespace_path(run: str, slot: str, model: str) -> Path:
    """Where a cell's Stage-2 free-space grid lives (`splat/freespace.npz`)."""
    return _slot_dir(run, slot, model) / "splat" / splat_stage2.FREESPACE_NAME


def _artifact_url(path: Path) -> str | None:
    """`/artifacts` URL for a file under RUNS_DIR, or None if absent/outside."""
    if not path.is_file():
        return None
    try:
        return f"/artifacts/{path.resolve().relative_to(RUNS_DIR.resolve()).as_posix()}"
    except ValueError:
        return None


def _splat_stage2_status(
    run: str, slot: str, model: str, source: str | None = None
) -> dict[str, Any]:
    """Public Stage-2 (free-space) state: the live job if one exists, else 'done'
    (with the `freespace.json` summary) when the grid is on disk, else 'idle'. `url`
    is the `voxels.bin` viz overlay the viewer draws."""
    grid = _freespace_path(run, slot, model)
    url = _artifact_url(_voxels_path(run, slot, model))
    job = _splat_stage2_jobs.get((run, slot, model))
    if job is not None:
        return {**job, "url": url}
    if grid.is_file():
        summary: Any = None
        with contextlib.suppress(Exception):
            summary = json.loads(grid.with_suffix(".json").read_text(encoding="utf-8"))
        return {
            "run": run, "slot": slot, "model": model, "source": source,
            "running": False, "phase": "done", "status": "done",
            "current_id": None, "error": None, "summary": summary, "url": url,
        }
    return {
        "run": run, "slot": slot, "model": model, "source": source,
        "running": False, "phase": "idle", "status": "idle",
        "current_id": None, "error": None, "summary": None, "url": None,
    }


def _voxels_path(run: str, slot: str, model: str) -> Path:
    """Where a cell's Stage-2 free-voxel viz pack lives (`splat/voxels.bin`)."""
    return _slot_dir(run, slot, model) / "splat" / splat_stage2.VOXELS_NAME


def _voxelize_from_source(
    run: str,
    slot: str,
    model: str,
    src_dir: Path,
    kind: str,
    out_path: Path,
    params: splat_stage2.FreeSpaceParams,
    job: dict[str, Any],
) -> dict[str, Any]:
    """Compute a cell's free-space grid (Stage 2), de-optimizing a library source to
    vanilla ONCE first. Tracks `job['phase']` ('deopt' / 'voxelize'). Blocking."""

    def _progress(done: int, total: int, current: str) -> None:
        job["phase"], job["done"], job["total"], job["current_id"] = (
            "voxelize", done, total, current,
        )

    def _run(vanilla_dir: Path) -> dict[str, Any]:
        job["phase"] = "voxelize"
        return splat_stage2.compute_free_space(
            run=run, slot=slot, model=model, raw_dir=vanilla_dir,
            out_path=out_path, params=params, progress=_progress,
        )

    if kind == "generated":
        return _run(src_dir)
    job["phase"] = "deopt"
    with tempfile.TemporaryDirectory(prefix="splat-deopt-", dir=str(RUNS_DIR)) as tmp:
        vanilla_dir = Path(tmp)
        _deoptimize_dir(src_dir, vanilla_dir)
        return _run(vanilla_dir)


async def _run_splat_stage2_cell(
    run: str, slot: str, model: str, params: splat_stage2.FreeSpaceParams
) -> None:
    """Compute ONE cell's free-space grid off the event loop, tracking phase/progress
    and writing the `freespace.json` sidecar so 'done' survives a restart."""
    key = (run, slot, model)
    job = _splat_stage2_jobs[key]
    try:
        source = _splat_source(run, slot, model)
        if source is None:
            raise FileNotFoundError(f"no convertible build for {slot}/{model} in {run}")
        src_dir, kind = source
        out_path = _freespace_path(run, slot, model)
        summary = await asyncio.to_thread(
            _voxelize_from_source, run, slot, model, src_dir, kind, out_path, params, job,
        )
        job["summary"] = summary
        with contextlib.suppress(Exception):
            out_path.with_suffix(".json").write_text(
                json.dumps(summary, indent=1), encoding="utf-8"
            )
        job["status"] = "done"
        job["phase"] = "done"
        job["current_id"] = None
    except Exception as exc:
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        job["running"] = False
        job["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _splat_stage2_tasks.pop(key, None)


class Stage3Request(BaseModel):
    """Surfel sampler knobs (Stage 3). All optional; omitted → sampler defaults.
    Count is set by `splat_density` (surfels per m², so it scales with surface area);
    `target_splats` (explicit count) or `base_spacing` (m) override it. `detail_splats`
    also builds a denser `cloud.detail.ply` LOD; `cull_hidden` drops surfels with no
    reachable free space on either side (uses the Stage-2 grid)."""

    target_splats: int | None = None      # explicit count override; else density × area
    splat_density: float | None = None     # surfels per m² (area-scaled count; the default)
    base_spacing: float | None = None
    radius_frac: float = splat_stage3.DEFAULT_RADIUS_FRAC
    flatness: float = splat_stage3.DEFAULT_FLATNESS
    adaptive: bool = True
    feature_boost: float = splat_stage3.DEFAULT_FEATURE_BOOST  # crease/boundary refinement (1 = off)
    cull_hidden: bool = True
    detail_splats: int | None = None
    representation: str | None = None  # "2dgs" (default) | "3dgs" (compat/compression)


def _stage3_params(
    req: Stage3Request | None,
) -> tuple[splat_stage3.SampleParams, splat_stage3.SampleParams | None]:
    """Clamp a request into a (base, detail|None) SampleParams pair."""
    req = req or Stage3Request()
    # Count is area-scaled by DENSITY by default; an explicit target_splats overrides.
    # The ceiling is generous (not a room-scale cap) so large scenes aren't clipped.
    target = (
        int(min(max(req.target_splats, 5_000), 50_000_000))
        if req.target_splats is not None
        else None
    )
    density = (
        float(min(max(req.splat_density, 2.0), 2_000.0))
        if req.splat_density is not None
        else splat_stage3.DEFAULT_SPLAT_DENSITY
    )
    radius = float(min(max(req.radius_frac, 0.3), 3.0))
    flat = float(min(max(req.flatness, 0.01), 0.5))
    base_spacing = (
        float(min(max(req.base_spacing, 0.002), 0.5))
        if req.base_spacing is not None
        else None
    )
    rep = req.representation if req.representation in ("2dgs", "3dgs") else "2dgs"
    boost = float(min(max(req.feature_boost, 1.0), 8.0))
    base = splat_stage3.SampleParams(
        target_splats=target, splat_density=density, base_spacing=base_spacing,
        radius_frac=radius, flatness=flat, adaptive=bool(req.adaptive),
        feature_boost=boost, cull_hidden=bool(req.cull_hidden), representation=rep,
    )
    detail = None
    if req.detail_splats:
        detail = splat_stage3.SampleParams(
            target_splats=int(min(max(req.detail_splats, 50_000), 50_000_000)),
            splat_density=None, base_spacing=None, radius_frac=radius, flatness=flat,
            adaptive=bool(req.adaptive), feature_boost=boost,
            cull_hidden=bool(req.cull_hidden), representation=rep,
        )
    return base, detail


def _splat_stage3_status(
    run: str, slot: str, model: str, source: str | None = None
) -> dict[str, Any]:
    """Public Stage-3 (surfels) state: the live job, else 'done' (with the
    `cloud.json` summary) when the base cloud is on disk, else 'idle'. Carries `url`
    (base `.ply`) and `detail_url` (the denser LOD, if built)."""
    url = _artifact_url(_cloud_path(run, slot, model))
    detail_url = _artifact_url(_detail_cloud_path(run, slot, model))
    job = _splat_stage3_jobs.get((run, slot, model))
    if job is not None:
        return {**job, "url": url, "detail_url": detail_url}
    if url is not None:
        summary: Any = None
        with contextlib.suppress(Exception):
            summary = json.loads(
                _cloud_path(run, slot, model).with_suffix(".json").read_text(encoding="utf-8")
            )
        total = (summary or {}).get("objects_total", 0)
        return {
            "run": run, "slot": slot, "model": model, "source": source,
            "total": total, "done": total, "running": False, "phase": "done",
            "status": "done", "current_id": None, "error": None,
            "summary": summary, "url": url, "detail_url": detail_url,
        }
    return {
        "run": run, "slot": slot, "model": model, "source": source,
        "total": 0, "done": 0, "running": False, "phase": "idle",
        "status": "idle", "current_id": None, "error": None,
        "summary": None, "url": None, "detail_url": None,
    }


class Stage6Request(BaseModel):
    """Stage-6 fine-tune knobs (all optional). `iterations` overrides the trainer
    default; omitted → the CLI default (30k). `restart` forces a from-scratch retry
    (drop any checkpoint + trained.ply); omitted → resume from the latest checkpoint."""

    iterations: int | None = None
    restart: bool = False


def _splat_stage6_status(run: str, slot: str, model: str) -> dict[str, Any]:
    """Public Stage-6 (fine-tune) state. Stage 6 runs as a DETACHED background
    training PROCESS (see `_spawn_stage6`), so this reports its live state —
    'running' (with `pid` + a `log_tail` streamed from splat/stage6.log), 'done'
    (trained.ply + its `/artifacts` `url`), 'error' (non-zero exit + log), or 'idle'
    — plus the `sog_url` twin (client/tools/ply-to-sog.mjs) and the `log_url`. Falls
    back to a disk-only view when no process is tracked (e.g. after a server restart,
    or a run done in a bare terminal): 'done' when trained.ply is present."""
    trained = _trained_path(run, slot, model)
    url = _artifact_url(trained)
    log_path = _stage6_log_path(run, slot, model)
    ckpt_step = _latest_ckpt_step(_stage6_ckpt_dir(run, slot, model))
    base = {
        "run": run, "slot": slot, "model": model,
        "sog_url": _artifact_url(trained.with_suffix(".sog")),
        "log_url": _artifact_url(log_path),
        # An interrupted run left a checkpoint but no trained.ply → the next launch
        # resumes it (see `_spawn_stage6` / splat/stage6.py).
        "ckpt_step": ckpt_step,
        "resumable": url is None and ckpt_step is not None,
    }
    rec = _splat_stage6_procs.get((run, slot, model))
    if rec is not None:
        rc = rec["popen"].poll()
        tail = _tail_text(log_path)
        if rc is None:
            return {
                **base, "status": "running", "running": True, "url": url,
                "pid": rec["popen"].pid, "iterations": rec.get("iterations"),
                "started_at": rec.get("started_at"), "error": None, "log_tail": tail,
            }
        # Finished — close the log handle once, but keep the record so the terminal
        # state (done/error + log) survives later polls until the next launch.
        if rec.get("logf") is not None:
            with contextlib.suppress(Exception):
                rec["logf"].close()
            rec["logf"] = None
        rec["returncode"] = rc
        if rc == 0 and url is not None:
            return {**base, "status": "done", "running": False, "url": url,
                    "error": None, "log_tail": tail}
        return {**base, "status": "error", "running": False, "url": url,
                "error": f"training exited with code {rc}", "log_tail": tail}
    return {
        **base, "status": "done" if url else "idle", "running": False,
        "url": url, "error": None, "log_tail": _tail_text(log_path),
    }


def _spawn_stage6(
    run: str, slot: str, model: str, iterations: int | None, restart: bool = False
) -> dict[str, Any]:
    """Launch the Stage-6 fine-tune as a DETACHED background process on this host,
    streaming stdout+stderr to splat/stage6.log. It runs under the server's OWN
    interpreter — the `.venv-splat` CUDA env that already hosts the in-process Stage-5
    renderer (see scripts/run_request.py), with CUDA_HOME inherited — so no separate
    env wiring is needed. Detached (`DETACHED_PROCESS` / `start_new_session`) and
    polled via `popen.poll()`, so it never blocks the event loop.

    Resumes from the latest `splat/ckpt/` checkpoint by default (so a crashed run
    continues where it stopped); `restart` first drops trained.ply + the checkpoints
    for an explicit from-scratch retry."""
    out = _trained_path(run, slot, model)
    log_path = _stage6_log_path(run, slot, model)
    out.parent.mkdir(parents=True, exist_ok=True)
    if restart:
        import shutil

        out.unlink(missing_ok=True)
        shutil.rmtree(_stage6_ckpt_dir(run, slot, model), ignore_errors=True)
    cmd = [
        sys.executable, "-u", "-m", "splat.stage6",
        "--cloud", str(_cloud_path(run, slot, model)),
        "--refs", str(_refs_dir(run, slot, model)),
        "--out", str(out), "--run", run, "--slot", slot, "--model", model,
        "--resume",  # continue from a checkpoint if one survived; restart wiped it
    ]
    if iterations is not None:
        cmd += ["--iterations", str(int(iterations))]
    logf = open(log_path, "wb")  # noqa: SIM115 - inherited by the child; closed on finish
    logf.write(f"$ {' '.join(cmd)}\n\n".encode())
    logf.flush()
    kwargs: dict[str, Any] = {
        "cwd": str(_REPO_ROOT), "stdout": logf, "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        kwargs["start_new_session"] = True
    popen = subprocess.Popen(cmd, **kwargs)
    _splat_stage6_procs[(run, slot, model)] = {
        "popen": popen, "logf": logf, "iterations": iterations,
        "started_at": datetime.now().isoformat(timespec="seconds"), "returncode": None,
    }
    return _splat_stage6_status(run, slot, model)


def _sample_cell_lods(
    run: str,
    slot: str,
    model: str,
    src_dir: Path,
    kind: str,
    freespace_path: Path,
    out_path: Path,
    base_params: splat_stage3.SampleParams,
    detail_params: splat_stage3.SampleParams | None,
    job: dict[str, Any],
) -> dict[str, Any]:
    """Sample a cell into a base surfel cloud (and, if `detail_params`, a denser LOD),
    consuming the Stage-2 free-space grid and de-optimizing a library source ONCE.
    Tracks `job['phase']` ('deopt' / 'base' / 'detail'). Blocking."""
    detail_path = out_path.with_suffix(".detail.ply")

    def _phase_progress(phase: str) -> splat_stage3.ProgressCb:
        def cb(done: int, total: int, current: str) -> None:
            job["phase"], job["done"], job["total"], job["current_id"] = (
                phase, done, total, current,
            )
        return cb

    def _sample(vanilla_dir: Path) -> dict[str, Any]:
        job["phase"] = "base"
        summary = splat_stage3.sample_cell(
            run=run, slot=slot, model=model, raw_dir=vanilla_dir,
            freespace_path=freespace_path, out_path=out_path,
            params=base_params, progress=_phase_progress("base"),
        )
        if detail_params is not None:
            job["phase"], job["done"], job["total"] = "detail", 0, 0
            det = splat_stage3.sample_cell(
                run=run, slot=slot, model=model, raw_dir=vanilla_dir,
                freespace_path=freespace_path, out_path=detail_path,
                params=detail_params, progress=_phase_progress("detail"),
            )
            summary["detail"] = {
                "splats": det["splats"], "bytes": det["bytes"],
                "base_spacing": det["base_spacing"], "params": det["params"],
            }
        else:
            detail_path.unlink(missing_ok=True)  # drop any stale detail LOD
        return summary

    if kind == "generated":
        return _sample(src_dir)
    job["phase"] = "deopt"
    with tempfile.TemporaryDirectory(prefix="splat-deopt-", dir=str(RUNS_DIR)) as tmp:
        vanilla_dir = Path(tmp)
        _deoptimize_dir(src_dir, vanilla_dir)
        return _sample(vanilla_dir)


async def _run_splat_stage3_cell(
    run: str,
    slot: str,
    model: str,
    base_params: splat_stage3.SampleParams,
    detail_params: splat_stage3.SampleParams | None,
) -> None:
    """Sample ONE cell into surfels off the event loop. Requires the Stage-2
    free-space grid; writes the `cloud.json` sidecar so 'done' survives a restart."""
    key = (run, slot, model)
    job = _splat_stage3_jobs[key]
    try:
        source = _splat_source(run, slot, model)
        if source is None:
            raise FileNotFoundError(f"no convertible build for {slot}/{model} in {run}")
        freespace = _freespace_path(run, slot, model)
        if not freespace.is_file():
            raise FileNotFoundError("run Stage 2 (free-space) first")
        src_dir, kind = source
        out_path = _cloud_path(run, slot, model)
        summary = await asyncio.to_thread(
            _sample_cell_lods, run, slot, model, src_dir, kind,
            freespace, out_path, base_params, detail_params, job,
        )
        job["summary"] = summary
        with contextlib.suppress(Exception):
            out_path.with_suffix(".json").write_text(
                json.dumps(summary, indent=1), encoding="utf-8"
            )
        job["status"] = "done"
        job["phase"] = "done"
        job["current_id"] = None
    except Exception as exc:
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        job["running"] = False
        job["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _splat_stage3_tasks.pop(key, None)


class Stage4Request(BaseModel):
    """Coverage-planner knobs (all optional; omitted → defaults). `min_gain`
    truncates the diminishing tail (higher = fewer cameras); `angles_per_patch`
    is K; `patch_fraction` is the uniform surfel→patch thinning (detail now
    arrives in the cloud itself — Stage 3's feature-adaptive density — so there
    are no detector knobs here). The cube-face `render_resolution` +
    `min_px_per_patch` set the shared reference-render intrinsics from which the
    footprint budget (view distance) is derived — matching what Stage 5 will
    render."""

    patch_fraction: float | None = None
    collision_clearance: float | None = None
    angles_per_patch: int | None = None
    angular_sectors: int | None = None
    near_frac: float | None = None
    headon_cos: float | None = None
    min_gain: int | None = None
    candidate_spacing: float | None = None
    max_candidates: int | None = None
    render_resolution: int | None = None
    min_px_per_patch: float | None = None
    gpu: bool | None = None  # run the coverage ray-march on CUDA (default on); False forces CPU


def _stage4_params(req: Stage4Request | None) -> splat_stage4.PlanParams:
    """Clamp a request into PlanParams; omitted fields keep the sampler defaults."""
    d = splat_stage4.PlanParams()
    if req is None:
        return d

    def pick(v: float | None, lo: float, hi: float, default: float) -> float:
        return float(min(max(v, lo), hi)) if v is not None else default

    return splat_stage4.PlanParams(
        patch_fraction=pick(req.patch_fraction, 0.05, 1.0, d.patch_fraction),
        collision_clearance=pick(req.collision_clearance, 0.05, 2.0, d.collision_clearance),
        angles_per_patch=int(pick(req.angles_per_patch, 1, 12, d.angles_per_patch)),
        angular_sectors=int(pick(req.angular_sectors, 3, 16, d.angular_sectors)),
        near_frac=pick(req.near_frac, 0.1, 1.0, d.near_frac),
        headon_cos=pick(req.headon_cos, 0.0, 0.95, d.headon_cos),
        min_gain=int(pick(req.min_gain, 1, 500, d.min_gain)),
        candidate_spacing=pick(req.candidate_spacing, 0.1, 3.0, d.candidate_spacing),
        # Generous safety ceiling (was a room-scale 20k cap); Stage 4 even-downsamples
        # and warns if a scene exceeds it, so large scenes aren't silently clipped.
        max_candidates=int(pick(req.max_candidates, 1000, 5_000_000, d.max_candidates)),
        render_resolution=int(pick(req.render_resolution, 128, 2048, d.render_resolution)),
        min_px_per_patch=pick(req.min_px_per_patch, 2.0, 64.0, d.min_px_per_patch),
        gpu=req.gpu if req.gpu is not None else d.gpu,
    )


def _cameras_path(run: str, slot: str, model: str) -> Path:
    """Where a cell's Stage-4 camera plan lives (`splat/cameras.json`)."""
    return _slot_dir(run, slot, model) / "splat" / splat_stage4.CAMERAS_NAME


def _splat_stage4_status(run: str, slot: str, model: str) -> dict[str, Any]:
    """Public Stage-4 state: live job, else 'done' (with the plan summary) when
    `cameras.json` is on disk, else 'idle'. Carries `url` (cameras.json),
    `patches_url` (patches.bin), and `patch_views_url` (patch_views.json — the
    per-patch → covering camera/face index the debug viewer selects against)."""
    cams = _cameras_path(run, slot, model)
    url = _artifact_url(cams)
    patches_url = _artifact_url(cams.with_name(splat_stage4.PATCHES_NAME))
    patch_views_url = _artifact_url(cams.with_name(splat_stage4.PATCH_VIEWS_NAME))
    job = _splat_stage4_jobs.get((run, slot, model))
    if job is not None:
        return {**job, "url": url, "patches_url": patches_url, "patch_views_url": patch_views_url}
    if url is not None:
        summary: Any = None
        with contextlib.suppress(Exception):
            payload = json.loads(cams.read_text(encoding="utf-8"))
            summary = {k: v for k, v in payload.items() if k != "cameras"}
            if isinstance(payload.get("cameras"), list):
                summary["cameras"] = len(payload["cameras"])
        return {
            "run": run, "slot": slot, "model": model,
            "running": False, "phase": "done", "status": "done",
            "current_id": None, "error": None, "summary": summary,
            "url": url, "patches_url": patches_url, "patch_views_url": patch_views_url,
        }
    return {
        "run": run, "slot": slot, "model": model,
        "running": False, "phase": "idle", "status": "idle",
        "current_id": None, "error": None, "summary": None,
        "url": None, "patches_url": None, "patch_views_url": None,
    }


def _plan_cameras_cell(
    run: str, slot: str, model: str, out_path: Path,
    plan_params: splat_stage4.PlanParams, job: dict[str, Any],
) -> dict[str, Any]:
    """Plan cameras from the Stage-2 free-space grid + Stage-3 surfel cloud — no mesh
    loading, no de-optimization (that's why Stage 4 can't run off a raw mesh)."""

    def _progress(done: int, total: int, current: str) -> None:
        job["phase"], job["done"], job["total"], job["current_id"] = (
            "plan", done, total, current,
        )

    job["phase"] = "plan"
    return splat_stage4.plan_cameras(
        run=run, slot=slot, model=model,
        freespace_path=_freespace_path(run, slot, model),
        surfels_path=_cloud_path(run, slot, model),
        out_path=out_path, params=plan_params, progress=_progress,
    )


async def _run_splat_stage4_cell(
    run: str, slot: str, model: str, plan_params: splat_stage4.PlanParams
) -> None:
    """Plan ONE cell's coverage cameras off the event loop. Requires the Stage-2
    free-space grid + Stage-3 surfel cloud; `cameras.json` is the 'done' marker."""
    key = (run, slot, model)
    job = _splat_stage4_jobs[key]
    try:
        if not _freespace_path(run, slot, model).is_file():
            raise FileNotFoundError("run Stage 2 (free-space) first")
        if not _cloud_path(run, slot, model).is_file():
            raise FileNotFoundError("run Stage 3 (surfels) first")
        out_path = _cameras_path(run, slot, model)
        summary = await asyncio.to_thread(
            _plan_cameras_cell, run, slot, model, out_path, plan_params, job,
        )
        job["summary"] = summary
        job["status"] = "done"
        job["phase"] = "done"
        job["current_id"] = None
    except Exception as exc:
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        job["running"] = False
        job["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _splat_stage4_tasks.pop(key, None)


def _refs_dir(run: str, slot: str, model: str) -> Path:
    """Where a cell's Stage-5 reference renders live (`splat/refs/`)."""
    return _slot_dir(run, slot, model) / "splat" / splat_stage5.REFS_DIRNAME


def _splat_stage_artifacts(run: str, slot: str, model: str) -> dict[int, list[Path]]:
    """The on-disk output(s) of each splat stage for one cell, keyed by stage number
    (2..5; Stage 1's `scene.json` is the base). Used to REVERT downstream on re-run."""
    d = _slot_dir(run, slot, model) / "splat"
    fs = _freespace_path(run, slot, model)
    cloud = _cloud_path(run, slot, model)
    cams = _cameras_path(run, slot, model)
    return {
        1: [d / splat_stage1.MANIFEST_NAME],
        2: [fs, fs.with_suffix(".json"), _voxels_path(run, slot, model)],
        3: [cloud, cloud.with_suffix(".json"), _detail_cloud_path(run, slot, model)],
        4: [
            cams,
            cams.with_name(splat_stage4.PATCHES_NAME),
            cams.with_name(splat_stage4.PATCH_VIEWS_NAME),
        ],
        5: [_refs_dir(run, slot, model)],
    }


def _revert_after(run: str, slot: str, model: str, stage: int) -> None:
    """Re-running stage `stage` invalidates every LATER stage: cancel their live jobs
    and delete their on-disk outputs. This makes a re-run act as a revert — the
    pipeline can never show stale downstream results built on a superseded input."""
    import shutil

    key = (run, slot, model)
    tables: dict[int, tuple[dict[Any, Any], dict[Any, Any]]] = {
        2: (_splat_stage2_jobs, _splat_stage2_tasks),
        3: (_splat_stage3_jobs, _splat_stage3_tasks),
        4: (_splat_stage4_jobs, _splat_stage4_tasks),
        5: (_splat_stage5_jobs, _splat_stage5_tasks),
    }
    artifacts = _splat_stage_artifacts(run, slot, model)
    for s in range(stage + 1, 6):
        jobs, tasks = tables[s]
        task = tasks.pop(key, None)
        if task is not None:
            task.cancel()
        jobs.pop(key, None)
        for p in artifacts.get(s, []):
            with contextlib.suppress(Exception):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
    # Any re-run at/upstream of Stage 5 also invalidates the Stage-6 resume state (its
    # checkpoints were trained against the now-superseded cloud/references), so drop
    # them — a later training launch starts fresh instead of resuming onto stale data.
    with contextlib.suppress(Exception):
        shutil.rmtree(_stage6_ckpt_dir(run, slot, model), ignore_errors=True)


def _splat_stage5_status(run: str, slot: str, model: str) -> dict[str, Any]:
    """Public Stage-5 state: live job, else 'done' (with a small summary read from
    `refs/transforms.json`) when it's on disk, else 'idle'. Carries `url`
    (transforms.json) for downstream consumers."""
    transforms = _refs_dir(run, slot, model) / splat_stage5.TRANSFORMS_NAME
    url = _artifact_url(transforms)
    job = _splat_stage5_jobs.get((run, slot, model))
    if job is not None:
        return {**job, "url": url}
    if url is not None:
        summary: Any = None
        with contextlib.suppress(Exception):
            doc = json.loads(transforms.read_text(encoding="utf-8"))
            summary = {"resolution": doc.get("w"), "views": len(doc.get("frames", []))}
        return {
            "run": run, "slot": slot, "model": model,
            "running": False, "phase": "done", "status": "done",
            "current_id": None, "error": None, "summary": summary, "url": url,
        }
    return {
        "run": run, "slot": slot, "model": model,
        "running": False, "phase": "idle", "status": "idle",
        "current_id": None, "error": None, "summary": None, "url": None,
    }


def _render_refs_from_source(
    run: str, slot: str, model: str, src_dir: Path, kind: str,
    cameras_path: Path, out_dir: Path, job: dict[str, Any],
) -> dict[str, Any]:
    """Render one cell's references, de-optimizing a library source to vanilla ONCE
    first. Tracks `job['phase']` ('deopt' / 'render'). Blocking — nvdiffrast (imported
    lazily inside render_references) needs CUDA."""

    def _progress(done: int, total: int, current: str) -> None:
        job["phase"], job["done"], job["total"], job["current_id"] = (
            "render", done, total, current,
        )

    def _run(vanilla_dir: Path) -> dict[str, Any]:
        job["phase"] = "render"
        return splat_stage5.render_references(
            run=run, slot=slot, model=model, raw_dir=vanilla_dir,
            cameras_path=cameras_path, out_dir=out_dir, progress=_progress,
        )

    if kind == "generated":
        return _run(src_dir)
    job["phase"] = "deopt"
    with tempfile.TemporaryDirectory(prefix="splat-deopt-", dir=str(RUNS_DIR)) as tmp:
        vanilla_dir = Path(tmp)
        _deoptimize_dir(src_dir, vanilla_dir)
        return _run(vanilla_dir)


async def _run_splat_stage5_cell(run: str, slot: str, model: str) -> None:
    """Render ONE cell's references off the event loop. Requires the Stage-4
    `cameras.json`; `refs/transforms.json` (written last) is the 'done' marker."""
    key = (run, slot, model)
    job = _splat_stage5_jobs[key]
    try:
        source = _splat_source(run, slot, model)
        if source is None:
            raise FileNotFoundError(f"no convertible build for {slot}/{model} in {run}")
        cameras_path = _cameras_path(run, slot, model)
        if not cameras_path.is_file():
            raise FileNotFoundError("run Stage 4 first (no camera plan)")
        src_dir, kind = source
        out_dir = _refs_dir(run, slot, model)
        summary = await asyncio.to_thread(
            _render_refs_from_source,
            run, slot, model, src_dir, kind, cameras_path, out_dir, job,
        )
        job["summary"] = summary
        job["status"] = "done"
        job["phase"] = "done"
        job["current_id"] = None
    except Exception as exc:
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        job["running"] = False
        job["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _splat_stage5_tasks.pop(key, None)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        global _current_run
        # Open on the most recently touched run (if any); other runs hydrate
        # lazily on activation. Nothing is seeded — runs only exist when the
        # user creates them.
        run_dirs = sorted(
            (p for p in RUNS_DIR.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if run_dirs:
            _current_run = run_dirs[0].name
            _hydrate_run(_current_run)
        # Resolve LLM costs off the pipeline, forever — including a backlog left
        # unpriced by a prior process that exited mid-lookup (the resumed run's
        # cells hydrate above, so this sweep recovers them).
        cost_task = asyncio.create_task(_cost_backfill_loop())
        try:
            yield
        finally:
            rlog.suppress_console()
            cost_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await cost_task
            for slot_log in _slot_logs.values():
                slot_log.close()
            for run_name, slot_id, model_alias in list(_slot_logs.keys()):
                generation.cancel_pending(_run_id(run_name, slot_id, model_alias))
            for task in _tasks.values():
                task.cancel()
            for task in _retry_tasks:
                task.cancel()
            for br in _branches.values():
                if br.task is not None:
                    br.task.cancel()
            for task in _generate_tasks.values():
                task.cancel()
            for task in _regen_tasks.values():
                task.cancel()
            for task in list(_tasks.values()):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for task in list(_retry_tasks):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for br in list(_branches.values()):
                if br.task is not None:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await br.task
            for br in _branches.values():
                br.log.close()
            for task in list(_generate_tasks.values()):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for task in list(_regen_tasks.values()):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            await oneshot.shutdown()
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
    # The experimental one-shot track (single-call scene design, own slots/
    # models/prompt) — fully isolated under /oneshot; see app/oneshot/.
    app.include_router(oneshot.router)
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
                meta = _run_meta(p.name)
                pv = meta.get("prompt_version")
                items.append(
                    {
                        "name": p.name,
                        "modified_at": p.stat().st_mtime,
                        # null for legacy runs (loadable, not resumable).
                        "prompt_version": pv if isinstance(pv, str) else None,
                    }
                )
        return {"runs": items, "current": _current_run}

    @app.post("/runs")
    async def create_run(req: CreateRunRequest) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        name = req.name.strip()
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(status_code=400, detail="invalid run name")
        version = req.prompt_version.strip()
        if not prompt_store.version_exists(version):
            raise HTTPException(status_code=404, detail=f"unknown prompt version: {version}")
        try:
            # Fail before any directory exists if the source set is incomplete.
            prompt_store.validate_version(version)
        except prompt_store.PromptTemplateError as e:
            raise HTTPException(status_code=409, detail=str(e))
        run_dir = _run_dir(name)
        if run_dir.exists():
            raise HTTPException(status_code=409, detail=f"run already exists: {name}")
        run_dir.mkdir(parents=True)
        prompt_store.snapshot_into_run(version, run_dir)
        (run_dir / RUN_META_NAME).write_text(
            json.dumps(
                {
                    "prompt_version": version,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            + "\n",
            encoding="utf-8",
        )
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
        # The copy carries the run's prompts/ snapshot + run.json, so the
        # archive replays and resumes with the source's exact prompts.
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

    @app.post("/runs/{name}/hydrate")
    async def hydrate_run(name: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        """Load a run's SlotLogs into memory WITHOUT activating it, so its
        cells' /scene + /meshes become readable alongside the active run — the
        run-compare view reads a SECOND run next to run A. Idempotent and
        launches nothing (see _maybe_launch), so it never disturbs the active
        run's board, cost, or pipelines."""
        if not _run_dir(name).is_dir():
            raise HTTPException(status_code=404, detail=f"unknown run: {name}")
        _hydrate_run(name)
        return {"run": name}

    @app.get("/prompt-runs")
    async def prompt_runs() -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Runs that have a rendition, ordered chronologically by when each was
        first launched (the OLDEST events.jsonl mtime among its cells). The
        prompt inspector diffs each run against its immediate predecessor here."""
        def _collect() -> list[dict[str, object]]:
            items: list[dict[str, object]] = []
            if RUNS_DIR.exists():
                for p in RUNS_DIR.iterdir():
                    if not p.is_dir():
                        continue
                    cells = _run_cells_with_data(p.name)
                    if not cells:
                        continue  # never launched — nothing to inspect
                    slots = {k.split("/", 1)[0] for k, _ev, _m in cells}
                    items.append(
                        {
                            "name": p.name,
                            "launched_at": _run_first_launch(p.name),
                            "n_slots": len(slots),
                            "n_cells": len(cells),
                        }
                    )
            items.sort(key=lambda d: (d["launched_at"], d["name"]))
            return items

        runs = await asyncio.to_thread(_collect)
        return {"runs": runs, "current": _current_run}

    @app.get("/runs/{run}/prompts")
    async def run_prompts(  # pyright: ignore[reportUnusedFunction]
        run: str, compare: str | None = None
    ) -> dict[str, object]:
        """Per-step prompts a run actually used, reconstructed from its event
        logs, with a word-level diff against `compare` when supplied."""
        if not _run_dir(run).is_dir():
            raise HTTPException(status_code=404, detail=f"unknown run: {run}")
        return await asyncio.to_thread(_build_run_prompts, run, compare)

    @app.post("/prompt-test")
    async def prompt_test(  # pyright: ignore[reportUnusedFunction]
        req: PromptTestRequest,
    ) -> dict[str, object]:
        """Re-run ONE historical step call with edited templates — the engine
        behind the prompt lab's "test on selected events". The target event's
        logged `variables` are substituted into the edited templates and the
        call re-issued on the event's own model via `llm.call_llm_once`, which
        neither reads the LLM cache nor writes a `cache.llm` event. The result
        is rendered transiently in the client and discarded; nothing about any
        run is mutated."""
        slot_log, event = _require_step_event(
            req.run, req.slot, req.model, req.event_index, req.step,
        )
        variables = event["variables"]
        model_id = str(event.get("model") or slot_log.state.get("model") or "")
        schema_cls = getattr(schemas, str(event.get("schema")), None)
        if not (isinstance(schema_cls, type) and issubclass(schema_cls, BaseModel)):
            raise HTTPException(
                status_code=409,
                detail=f"event's output schema is unknown: {event.get('schema')!r}",
            )
        try:
            system = prompt_store.resolve(
                req.system_template, variables, where=f"{req.step}.system (edited)")
            user = prompt_store.resolve(
                req.user_template, variables, where=f"{req.step}.user (edited)")
        except prompt_store.PromptTemplateError as e:
            raise HTTPException(status_code=400, detail=str(e))
        user = llm.apply_model_quirks(user, model_id)
        llm.set_model(model_id)
        try:
            # Display-only preview: it never logs a cache.llm event, so the
            # billed generation ids go unused (no cost is attributed to a run).
            _validated, reasoning, usage, raw, _gen_ids = await llm.call_llm_once(
                system=system,
                user=user,
                output_schema=schema_cls,
                model=model_id,
                log_retries=False,
            )
        except Exception as e:
            # Surface provider/parse failures as a clean 502 the lab can show
            # inline, rather than a 500 with a stack trace.
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
        return {
            # The exact bytes sent — what a branch seed must carry.
            "system": system,
            "user": user,
            "output": raw,
            "reasoning": reasoning,
            "schema": event.get("schema"),
            "model": model_id,
            "node": event.get("node"),
            "original_output": event.get("output"),
            "tokens_in": getattr(usage, "prompt_tokens", None),
            "tokens_out": getattr(usage, "completion_tokens", None),
        }

    @app.post("/inquire")
    async def inquire(req: InquiryRequest) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        """Ask the reviewer (Claude Opus 4.8, xhigh reasoning) the latest
        question in a conversation, and keep asking — the client carries the
        thread forward in `messages`, so it is persistent. `system` is the exact,
        client-assembled prompt (the investigator's analyst framing + scene
        grounding + any attached steps). Stateless: nothing about any run is read
        or mutated."""
        if not req.messages or req.messages[-1].role != "user":
            raise HTTPException(
                status_code=400,
                detail="messages must be non-empty and end with a user turn",
            )
        convo = [{"role": m.role, "content": m.content} for m in req.messages]
        try:
            answer, reasoning = await llm.chat(model=INQUIRY_MODEL, system=req.system, messages=convo)
        except Exception as e:
            # Provider/transport failure → clean 502 the panel shows inline, with
            # the provider's ACTUAL complaint (metadata/body), not just the SDK's
            # opaque "Provider returned error".
            raise HTTPException(status_code=502, detail=_llm_error_message(e))
        return {"answer": answer, "reasoning": reasoning, "model": INQUIRY_MODEL}

    # --- per-slot investigator context --------------------------------------
    #
    # The holistic "why is the WHOLE scene like this?" chat (vs. the per-step
    # `/inquire` above) needs the cell's faithful base grounding: the full final
    # scene context + a timeline of every executed step's output/reasoning/vars.
    # The client fetches this once per open (and on refresh), pairs it with the
    # run's prompt templates + a static pipeline explainer, and forwards the
    # composed system prompt through the SAME `/inquire` reviewer. Read-only.
    @app.get("/slots/{slot_id}/{model_alias}/investigator")
    async def slot_investigator(slot_id: str, model_alias: str, run: str | None = None) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        return _investigator_bundle(_require_slot_log(run, slot_id, model_alias))

    @app.get("/branches/{branch_id}/investigator")
    async def branch_investigator(branch_id: str) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        return _investigator_bundle(_require_branch(branch_id).log)

    @app.get("/runs/{run}/prompt-templates")
    async def run_prompt_templates(run: str) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """The run's prompt snapshot as editable templates, with each step's
        available variables — the prompt lab's editing source."""
        ps = _require_run_prompts(run)
        return {
            "run": run,
            "steps": [
                {
                    "step": step,
                    "system": ps.template(step, "system"),
                    "user": ps.template(step, "user"),
                    # Every variable is injectable into every template; the
                    # `native` subset is what this step backs with real values
                    # (the rest render empty/placeholder here).
                    "variables": prompt_store.ALL_VARIABLES,
                    "native": prompt_store.STEP_VARIABLES[step],
                }
                # The steps THIS snapshot actually carries (see PromptSet.steps),
                # not the global list — so a snapshot predating an optional step
                # (e.g. object_decomp) neither surfaces it in the lab nor trips
                # the ps.template lookup below, while a newer one exposes it.
                for step in ps.steps()
            ],
        }

    @app.get("/variable-samples")
    async def variable_samples() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        """Every prompt `{VARIABLE}` rendered against a fixed sample scene via
        the real `scene_context` injection functions — the prompt lab's hover
        preview, so each variable's actual injected shape (and any missing
        context / rendering bug) is visible without running a scene. Run-
        independent: the sample scene is static."""
        return scene_context.sample_variables()

    @app.get("/runs/{run}/step-events")
    async def run_step_events(run: str, step: str) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Every logged call of template `step` across the run's cells that is
        re-renderable (carries logged `variables`) — the prompt lab's
        candidate list for testing an edit."""
        if step not in prompt_store.STEPS:
            raise HTTPException(status_code=404, detail=f"unknown step: {step}")
        if not _run_dir(run).is_dir():
            raise HTTPException(status_code=404, detail=f"unknown run: {run}")
        _hydrate_run(run)
        items: list[dict[str, object]] = []
        for (r, slot_id, alias), slot_log in _slot_logs.items():
            if r != run:
                continue
            for e in slot_log.state["events"]:
                if e.get("kind") != "cache.llm" or e.get("template") != step:
                    continue
                if not isinstance(e.get("variables"), dict):
                    continue
                output = json.dumps(e.get("output"), ensure_ascii=False)
                items.append(
                    {
                        "slot": slot_id,
                        "model": alias,
                        "index": e.get("index"),
                        "node": e.get("node"),
                        "model_id": e.get("model"),
                        "tokens_out": e.get("tokens_out"),
                        "output_preview": output[:240] + ("…" if len(output) > 240 else ""),
                        "branch_live": any(
                            b.parent is None and b.run == run and b.slot == slot_id and b.model == alias
                            for b in _branches.values()
                        ),
                    }
                )
        items.sort(key=lambda d: (str(d["slot"]), str(d["model"]), int(d["index"] or 0)))
        return {"run": run, "step": step, "events": items}

    @app.get("/runs/{run}/step-event")
    async def run_step_event(  # pyright: ignore[reportUnusedFunction]
        run: str, slot: str, model: str, index: int, step: str,
    ) -> dict[str, object]:
        """One logged call's FULL bytes (exact system/user sent, output +
        reasoning received) — fetched on demand so the prompt lab's review
        canvas can show every event's input/output without bloating the
        candidate-list poll."""
        _, event = _require_step_event(run, slot, model, index, step)
        return {
            "slot": slot,
            "model": model,
            "index": index,
            "step": step,
            "node": event.get("node"),
            "model_id": event.get("model"),
            "system": event.get("system"),
            "user": event.get("user"),
            "output": event.get("output"),
            "reasoning": event.get("reasoning"),
            "tokens_in": event.get("tokens_in"),
            "tokens_out": event.get("tokens_out"),
        }

    @app.get("/branches/{branch_id}/step-event")
    async def branch_step_event(  # pyright: ignore[reportUnusedFunction]
        branch_id: str, step: str, node: str | None = None,
    ) -> dict[str, object]:
        """The matching call's FULL bytes from a simulation branch — the same
        template (and node, when given), so the lab can diff the live run's
        input/output against the simulated-edit branch. 404 when the branch has
        no such call in it yet."""
        if step not in prompt_store.STEPS:
            raise HTTPException(status_code=404, detail=f"unknown step: {step}")
        br = _require_branch(branch_id)
        events = br.log.state["events"]
        cands = [e for e in events if e.get("kind") == "cache.llm" and e.get("template") == step]
        # Match the SPECIFIC node when one is given — never fall back to another
        # node's call (e.g. a copied-prefix `root` decompose), which would diff a
        # stale prompt/output for the wrong region.
        if node is not None:
            cands = [e for e in cands if e.get("node") == node]
        # Surface the branch's ACTUAL output, not the pre-committed prompt-test
        # SEED. The seed is only a cache prime: when it hits, the branch runs
        # past this step using it (so it IS the output → show it); when it
        # misses, the branch pauses right before re-running this step (so the
        # seed is stale → report "not re-run yet" until the real call lands).
        # Real (non-seeded) calls always win, and the LATEST one is the final
        # validated attempt the pipeline actually used.
        real = [e for e in cands if not e.get("seeded")]
        if real:
            match = real[-1]
        elif cands and not _branch_awaiting(br, step, node):
            match = cands[-1]  # seed cache-hit and consumed — it is the output
        else:
            match = None
        if match is None:
            where = f"{step} @ {node}" if node else step
            raise HTTPException(status_code=404, detail=f"the branch has not re-run {where} yet")
        return {
            "branch": branch_id,
            "slot": br.slot,
            "model": br.model,
            "step": step,
            "index": match.get("index"),
            "node": match.get("node"),
            "model_id": match.get("model"),
            "system": match.get("system"),
            "user": match.get("user"),
            "output": match.get("output"),
            "reasoning": match.get("reasoning"),
            "tokens_in": match.get("tokens_in"),
            "tokens_out": match.get("tokens_out"),
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

    @app.get("/runs/{run}/splat/cells")
    async def splat_cells_list(run: str) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Every convertible cell in `run` (one with a generated or library build)
        plus its current Stage-1 state: 'idle' (never converted), a live job's
        progress, or 'done' (a manifest is already on disk). The screen renders
        this on run select and polls it while any cell is converting. Disk-only —
        never hydrates or activates the run."""
        if not (RUNS_DIR / run).is_dir():
            raise HTTPException(status_code=404, detail=f"run not found: {run}")
        cells = []
        for slot, model, placed, source in _discover_splat_stage1_cells(run):
            cell = _splat_cell_status(run, slot, model, placed, source)
            # Per-stage state so the screen can offer each step on a cell row, in
            # dependency order: Stage 2 free-space → Stage 3 surfels → Stage 4
            # cameras → Stage 5 reference renders.
            cell["stage2"] = _splat_stage2_status(run, slot, model, source)
            cell["stage3"] = _splat_stage3_status(run, slot, model, source)
            cell["stage4"] = _splat_stage4_status(run, slot, model)
            cell["stage5"] = _splat_stage5_status(run, slot, model)
            cell["stage6"] = _splat_stage6_status(run, slot, model)
            cells.append(cell)
        return {"run": run, "cells": cells}

    @app.get("/runs/{run}/splat/source/{slot}/{model}")
    async def splat_source_get(run: str, slot: str, model: str) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """The cell's splat source override ('auto' when none) + the build it
        currently resolves to."""
        pref = _splat_source_pref(run, slot, model)
        resolved = _splat_source(run, slot, model)
        return {"source": pref or "auto", "resolved": resolved[1] if resolved else None}

    @app.post("/runs/{run}/splat/source/{slot}/{model}")
    async def splat_source_set(run: str, slot: str, model: str, source: str = "auto") -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Pin which asset build the splat pipeline samples/renders for this cell:
        'generated' (raw), 'generated-lite', 'generated-optimized', 'library', or
        'auto' (clear). Applied by every stage through _splat_source — re-run the
        stages to regenerate against the new source."""
        path = _splat_source_pref_path(run, slot, model)
        src = (source or "auto").strip()
        if src in ("", "auto"):
            path.unlink(missing_ok=True)
        elif src in _SPLAT_SOURCE_CHOICES:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"source": src}), encoding="utf-8")
        else:
            raise HTTPException(status_code=400, detail=f"invalid source: {src!r}")
        resolved = _splat_source(run, slot, model)
        return {"source": src or "auto", "resolved": resolved[1] if resolved else None}

    @app.post("/runs/{run}/splat/stage1/{slot}/{model}")
    async def splat_stage1_start(run: str, slot: str, model: str) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Start Stage 1 for ONE cell — its meshes (the vanilla `objects-generated/`
        build, or the KTX2/Meshopt library build de-optimized on the fly) +
        `events.jsonl` become a validated `splat/scene.json` (see splat/stage1.py).
        Idempotent while running (returns the live job); a finished cell re-runs on
        a fresh POST (cheap, overwrites). Poll the GET (or the cells list) for
        progress."""
        source = _splat_source(run, slot, model)
        if source is None:
            raise HTTPException(
                status_code=404,
                detail=f"no convertible build for {slot}/{model} in {run}",
            )
        _, kind = source
        key = (run, slot, model)
        existing = _splat_stage1_jobs.get(key)
        if existing is not None and existing.get("running"):
            return dict(existing)
        _revert_after(run, slot, model, 1)  # re-running invalidates all later stages
        job: dict[str, Any] = {
            "run": run,
            "slot": slot,
            "model": model,
            "source": kind,
            "total": 0,
            "done": 0,
            "running": True,
            "status": "pending",
            "current_id": None,
            "error": None,
            "summary": None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
        }
        _splat_stage1_jobs[key] = job
        _splat_stage1_tasks[key] = asyncio.create_task(
            _run_splat_stage1_cell(run, slot, model)
        )
        return dict(job)

    @app.get("/runs/{run}/splat/stage1/{slot}/{model}")
    async def splat_stage1_status(run: str, slot: str, model: str) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Live Stage-1 state of one cell ('idle' / 'pending' / a running job /
        'done' / 'error')."""
        return _splat_cell_status(run, slot, model)

    @app.post("/runs/{run}/splat/stage2/{slot}/{model}")
    async def splat_stage2_start(  # pyright: ignore[reportUnusedFunction]
        run: str, slot: str, model: str, body: Stage2Request | None = None
    ) -> dict[str, object]:
        """Compute ONE cell's free-space grid — dual-resolution occupancy + clearance
        + reachability → `splat/freespace.npz` (+ `voxels.bin` viz), see
        splat/stage2.py. The shared spatial foundation Stage 3 (surfels) and Stage 4
        (cameras) consume. Optional body: `pitch`/`refine`/`margin`. Idempotent while
        running; re-runs (overwrites) on a fresh POST."""
        source = _splat_source(run, slot, model)
        if source is None:
            raise HTTPException(
                status_code=404,
                detail=f"no convertible build for {slot}/{model} in {run}",
            )
        _, kind = source
        params = _stage2_params(body)
        key = (run, slot, model)
        existing = _splat_stage2_jobs.get(key)
        if existing is not None and existing.get("running"):
            return dict(existing)
        _revert_after(run, slot, model, 2)  # re-running invalidates all later stages
        job: dict[str, Any] = {
            "run": run,
            "slot": slot,
            "model": model,
            "source": kind,
            "total": 0,
            "done": 0,
            "running": True,
            "status": "pending",
            "phase": "pending",
            "current_id": None,
            "error": None,
            "summary": None,
            "url": None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
        }
        _splat_stage2_jobs[key] = job
        _splat_stage2_tasks[key] = asyncio.create_task(
            _run_splat_stage2_cell(run, slot, model, params)
        )
        return dict(job)

    @app.get("/runs/{run}/splat/stage2/{slot}/{model}")
    async def splat_stage2_status(run: str, slot: str, model: str) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Live Stage-2 (free-space) state of one cell ('idle' / 'pending' / a running
        job / 'done' with the `voxels.bin` viz `url` / 'error')."""
        return _splat_stage2_status(run, slot, model)

    @app.post("/runs/{run}/splat/stage3/{slot}/{model}")
    async def splat_stage3_start(  # pyright: ignore[reportUnusedFunction]
        run: str, slot: str, model: str, body: Stage3Request | None = None
    ) -> dict[str, object]:
        """(Re-)sample ONE cell's placed meshes into a pre-fine-tuning Gaussian cloud
        `splat/cloud.ply` (see splat/stage3.py), consuming the Stage-2 free-space grid
        to orient normals + cull hidden faces. Knobs: `target_splats`/`base_spacing`/
        `radius_frac`/`flatness`/`adaptive`/`cull_hidden`/`detail_splats`. Requires
        Stage 2 first. Idempotent while running; re-runs (overwrites) on a fresh POST."""
        source = _splat_source(run, slot, model)
        if source is None:
            raise HTTPException(
                status_code=404,
                detail=f"no convertible build for {slot}/{model} in {run}",
            )
        if not _freespace_path(run, slot, model).is_file():
            raise HTTPException(status_code=409, detail="run Stage 2 (free-space) first")
        _, kind = source
        base_params, detail_params = _stage3_params(body)
        key = (run, slot, model)
        existing = _splat_stage3_jobs.get(key)
        if existing is not None and existing.get("running"):
            return dict(existing)
        _revert_after(run, slot, model, 3)  # re-running invalidates all later stages
        job: dict[str, Any] = {
            "run": run,
            "slot": slot,
            "model": model,
            "source": kind,
            "total": 0,
            "done": 0,
            "running": True,
            "status": "pending",
            "phase": "pending",
            "detail": detail_params is not None,
            "current_id": None,
            "error": None,
            "summary": None,
            "url": None,
            "detail_url": None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
        }
        _splat_stage3_jobs[key] = job
        _splat_stage3_tasks[key] = asyncio.create_task(
            _run_splat_stage3_cell(run, slot, model, base_params, detail_params)
        )
        return dict(job)

    @app.get("/runs/{run}/splat/stage3/{slot}/{model}")
    async def splat_stage3_status(run: str, slot: str, model: str) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Live Stage-3 (surfels) state of one cell ('idle' / 'pending' / a running
        job / 'done' with the cloud `url` / 'error')."""
        return _splat_stage3_status(run, slot, model)

    @app.post("/runs/{run}/splat/stage4/{slot}/{model}")
    async def splat_stage4_start(  # pyright: ignore[reportUnusedFunction]
        run: str, slot: str, model: str, body: Stage4Request | None = None
    ) -> dict[str, object]:
        """Plan coverage cameras for ONE cell — feature-adaptive patches + greedy
        set-cover over the free space (see splat/stage4.py) → `splat/cameras.json`
        (+ `patches.bin`), the input for Stage 5 reference renders and the
        occlusion-cull list. Knobs in the body (K, min_gain, patch spacing, …);
        idempotent while running; re-runs (overwrites) on a fresh POST."""
        source = _splat_source(run, slot, model)
        if source is None:
            raise HTTPException(
                status_code=404,
                detail=f"no convertible build for {slot}/{model} in {run}",
            )
        if not _freespace_path(run, slot, model).is_file():
            raise HTTPException(status_code=409, detail="run Stage 2 (free-space) first")
        if not _cloud_path(run, slot, model).is_file():
            raise HTTPException(status_code=409, detail="run Stage 3 (surfels) first")
        plan_params = _stage4_params(body)
        _, kind = source
        key = (run, slot, model)
        existing = _splat_stage4_jobs.get(key)
        if existing is not None and existing.get("running"):
            return dict(existing)
        _revert_after(run, slot, model, 4)  # re-running invalidates Stage 5
        job: dict[str, Any] = {
            "run": run,
            "slot": slot,
            "model": model,
            "source": kind,
            "total": 0,
            "done": 0,
            "running": True,
            "status": "pending",
            "phase": "pending",
            "current_id": None,
            "error": None,
            "summary": None,
            "url": None,
            "patches_url": None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
        }
        _splat_stage4_jobs[key] = job
        _splat_stage4_tasks[key] = asyncio.create_task(
            _run_splat_stage4_cell(run, slot, model, plan_params)
        )
        return dict(job)

    @app.get("/runs/{run}/splat/stage4/{slot}/{model}")
    async def splat_stage4_status(run: str, slot: str, model: str) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Live Stage-4 state of one cell ('idle' / 'pending' / a running job /
        'done' with the camera-plan `url` / 'error')."""
        return _splat_stage4_status(run, slot, model)

    @app.post("/runs/{run}/splat/stage5/{slot}/{model}")
    async def splat_stage5_start(run: str, slot: str, model: str, restart: bool = False) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Render ONE cell's UNLIT reference images from its Stage-4 camera plan —
        per-view RGB (albedo) + planar-Z depth + alpha into `splat/refs/` (see
        splat/stage5.py), the supervision for the Stage-6 gsplat fine-tune. Requires
        Stage 4 (`cameras.json`) first. CUDA-only: on a non-GPU host the job finishes
        with a clear 'needs CUDA' error. Idempotent while running. A fresh POST RESUMES
        (renders only the views still missing on disk); pass `restart=true` to wipe
        `refs/` and re-render every view from scratch."""
        source = _splat_source(run, slot, model)
        if source is None:
            raise HTTPException(
                status_code=404,
                detail=f"no convertible build for {slot}/{model} in {run}",
            )
        if not _cameras_path(run, slot, model).is_file():
            raise HTTPException(status_code=409, detail="run Stage 4 first (no camera plan)")
        _, kind = source
        key = (run, slot, model)
        existing = _splat_stage5_jobs.get(key)
        if existing is not None and existing.get("running"):
            return dict(existing)
        if restart:
            import shutil

            shutil.rmtree(_refs_dir(run, slot, model), ignore_errors=True)
            # New references supersede whatever the Stage-6 checkpoints trained on.
            shutil.rmtree(_stage6_ckpt_dir(run, slot, model), ignore_errors=True)
        job: dict[str, Any] = {
            "run": run, "slot": slot, "model": model, "source": kind,
            "total": 0, "done": 0, "running": True, "status": "pending",
            "phase": "pending", "current_id": None, "error": None, "summary": None,
            "url": None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
        }
        _splat_stage5_jobs[key] = job
        _splat_stage5_tasks[key] = asyncio.create_task(
            _run_splat_stage5_cell(run, slot, model)
        )
        return dict(job)

    @app.get("/runs/{run}/splat/stage5/{slot}/{model}")
    async def splat_stage5_status(run: str, slot: str, model: str) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Live Stage-5 state of one cell ('idle' / 'pending' / a running job /
        'done' with the `refs/transforms.json` `url` / 'error')."""
        return _splat_stage5_status(run, slot, model)

    @app.post("/runs/{run}/splat/stage6/{slot}/{model}")
    async def splat_stage6_start(  # pyright: ignore[reportUnusedFunction]
        run: str, slot: str, model: str, body: Stage6Request | None = None
    ) -> dict[str, object]:
        """Launch ONE cell's Stage-6 fine-tune (gsplat 2DGS) as a DETACHED background
        training process on the server host, streaming to splat/stage6.log — watch it
        live via this cell's status `log_tail` (or open the `log_url` artifact). Needs
        Stage 3 (`cloud.ply`) + Stage 5 (`refs/transforms.json`). Runs async: it does
        not block the server. Idempotent while training (returns the live job). A fresh
        POST after an interruption RESUMES from the latest `splat/ckpt/` checkpoint;
        pass `restart: true` (or POST after a clean finish) to train from scratch,
        overwriting trained.ply."""
        if not _cloud_path(run, slot, model).is_file():
            raise HTTPException(status_code=409, detail="run Stage 3 (surfels) first")
        if not (_refs_dir(run, slot, model) / splat_stage5.TRANSFORMS_NAME).is_file():
            raise HTTPException(status_code=409, detail="run Stage 5 (references) first")
        key = (run, slot, model)
        rec = _splat_stage6_procs.get(key)
        if rec is not None and rec["popen"].poll() is None:
            return _splat_stage6_status(run, slot, model)  # already training
        if rec is not None and rec.get("logf") is not None:
            with contextlib.suppress(Exception):
                rec["logf"].close()
        iterations = body.iterations if body is not None else None
        restart = bool(body.restart) if body is not None else False
        return _spawn_stage6(run, slot, model, iterations, restart)

    @app.get("/runs/{run}/splat/stage6/{slot}/{model}")
    async def splat_stage6_status(run: str, slot: str, model: str) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Live Stage-6 state ('idle' / 'running' with `pid` + `log_tail` / 'done'
        with the trained `.ply` `url` / 'error'). See `_splat_stage6_status`."""
        return _splat_stage6_status(run, slot, model)

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

        # Drain the per-run mesh-task tables on every hydrated cell.
        for run_name, slot_id, model_alias in list(_slot_logs.keys()):
            generation.cancel_pending(_run_id(run_name, slot_id, model_alias))

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
    async def slot_scene(slot_id: str, model_alias: str, run: str | None = None, until_index: int | None = None) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        # CQRS read model: fold the event log into the minimal renderable scene
        # state so the client paints directly instead of replaying ~2k events.
        # Returns `last_index` (the fold's cut) so the client can pick up the
        # live tail at `?since=last_index` with no gap or overlap.
        #
        # `until_index` projects only the prefix BEFORE that event — the compare
        # view's "previous" pane uses it to show the original run's state at the
        # branch's fork step (the shared baseline), not its full final scene.
        run = _resolve_run(run)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        events = list(slot_log.state["events"])
        if until_index is not None:
            events = events[:until_index]
        return _scene_projection(events)

    @app.get("/slots/{slot_id}/{model_alias}/meshes")
    async def slot_meshes(slot_id: str, model_alias: str, run: str | None = None, until_index: int | None = None, mode: str | None = None, optimized: bool = True, variant: str | None = None) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        # One-request scene bundle: stream every finished GLB for this cell in
        # a single length-prefixed response so the client loads a whole scene
        # over one connection instead of one HTTP request per mesh (browsers
        # cap those at ~6 per origin, and they'd contend with the SSE stream
        # and polls on this same origin). The client keys each blob to its
        # `model` event by file stem; see `_mesh_bundle`.
        #
        # Always filtered to ids with a committed `model` event (so a cell that
        # was promoted from a branch, whose objects/ retains stale hardlinks,
        # never serves ghosts); `until_index` further restricts to meshes
        # committed BEFORE that event — paired with /scene?until_index for the
        # compare view's "previous" pane (the original's meshes at the fork).
        #
        # `mode=generated` streams the cell's from-scratch generated build instead
        # of the library `objects/` — the OPTIMIZED twin (decimated + KTX2 +
        # Meshopt) by default, the raw bbox-fitted Trellis mesh when `optimized=0`
        # (the client's side-by-side comparison). _mesh_bundle skips the
        # `<id>.raw.glb` intermediates, so either dir streams the finished set.
        run = _resolve_run(run)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        cell_dir = slot_log.events_path.parent
        events = list(slot_log.state["events"])
        if until_index is not None:
            events = events[:until_index]
        if mode == "generated":
            rid = _run_id(run, slot_id, model_alias)
            objects_dir = _generated_variant_dir(rid, _resolve_gen_variant(variant, optimized))
        else:
            objects_dir = _objects_dir(cell_dir)
        return StreamingResponse(
            _mesh_bundle(objects_dir, _committed_mesh_ids(events)),
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
        _require_run_prompts(run)
        # Reverting rewrites the source log; this cell's branches are now stale
        # and any in-flight task must stop before we rewrite under it.
        await _discard_branches_of_cell(run, slot_id, model_alias)
        await _cancel_task(run, slot_id, model_alias)
        events = slot_log.state["events"]
        cut = max(0, min(req.to_event_index, len(events)))
        # Drop the meshes of every node generated at/after the cut so a later
        # re-run regenerates cleanly instead of silently reusing a stale glb.
        objs = slot_log.events_path.parent / "objects"
        for e in events[cut:]:
            oid = e.get("id")
            if isinstance(oid, str):
                for suffix in (".glb", ".raw.glb", ".png"):
                    with contextlib.suppress(OSError):
                        (objs / f"{oid}{suffix}").unlink()
        slot_log.truncate_events_to(cut)
        # Auto-resume from the cut instead of landing paused: a revert means
        # "re-run from here", so the reverted call (and everything downstream)
        # regenerates with no manual resume. Stepped cells get one free pass so
        # the reverted call itself re-executes before the gate pauses again
        # (mirrors rerun-step).
        key: RunKey = (run, slot_id, model_alias)
        if key in _stepped_cells:
            _gate_intents[key] = {"budget": 1}
        await _start_cell(run, slot_id, model_alias)
        return {"run": run, "slot_id": slot.id, "model": model_alias, "events": len(slot_log.state["events"])}

    @app.post("/slots/{slot_id}/{model_alias}/resume")
    async def slot_resume(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        run: str | None = None,
        stepped: bool | None = None,
    ) -> dict[str, str]:
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        # `stepped` opts the cell in/out of one-call-at-a-time execution
        # (persisted on disk); omitted = keep the cell's current mode.
        if stepped is not None:
            _set_stepped((run, slot_id, model_alias), stepped)
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
        # Over the spend cap: a plain resume would just keep spending toward a
        # limit already hit. The cap holds until raised past the spend (which
        # resumes the cell) — so refuse here and point at that.
        if _cap_reached(slot_log.state["events"]):
            raise HTTPException(
                status_code=409,
                detail="spend cap reached — raise the cap to continue",
            )
        # Startable iff nothing is already driving it (running OR parked at a
        # gate both hold a live task — those advance via /step, not /resume).
        if _live(_tasks.get((run, slot_id, model_alias))):
            raise HTTPException(status_code=400, detail="slot is already running")
        await _start_cell(run, slot_id, model_alias)
        return {"run": run, "slot_id": slot.id, "model": model_alias}

    @app.post("/slots/{slot_id}/{model_alias}/pause")
    async def slot_pause(slot_id: str, model_alias: str, run: str | None = None) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        # Pausable iff a live task exists to cancel — running OR parked at a
        # step gate (cancelling the gate breaks out of stepping into a hard
        # pause). A cell with no live task has nothing to pause.
        if not _live(_tasks.get((run, slot_id, model_alias))):
            raise HTTPException(status_code=400, detail="slot is not running")
        await _cancel_task(run, slot_id, model_alias)
        # _cancel_task awaits the cancellation, so the pipeline task has
        # already torn down (including generation.cancel_pending via _run's
        # CancelledError branch) by the time we emit the sentinel. The
        # run.paused event is what derive_status reads back as paused.
        slot_log.log("run.paused")
        return {"run": run, "slot_id": slot.id, "model": model_alias}

    @app.post("/slots/{slot_id}/{model_alias}/cap-override")
    async def slot_cap_override(  # pyright: ignore[reportUnusedFunction]
        slot_id: str, model_alias: str, req: CapOverrideRequest, run: str | None = None,
    ) -> dict[str, object]:
        """Set a cell's spend cap to an explicit ceiling (0 = uncapped) by
        appending the `run.cap_override` the cap math reads — allowed anytime the
        cell is live (running or paused), not only once it has tripped its cap. If
        the cap was the only thing holding a parked cell and the new ceiling clears
        its settled spend, RESUME it (a plain /resume stays refused while capped);
        if it's live and the new ceiling is already breached, stop it the way the
        backfill enforcer would. 400 when the cap system is off, 409 on a cell that
        hasn't started (nothing to cap) or is complete."""
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        if SPEND_CAP_USD <= 0:
            raise HTTPException(status_code=400, detail="spend cap is disabled")
        events = slot_log.state["events"]
        if not any(e.get("kind") == "run.start" for e in events):
            raise HTTPException(status_code=409, detail="cell hasn't started")
        if any(e.get("kind") == "run.done" for e in events):
            raise HTTPException(status_code=409, detail="run is complete")
        new_cap = max(0.0, float(req.cap))
        was_capped = _cap_reached(events)
        slot_log.log("run.cap_override", cap=new_cap, spend=_cell_spend(events))
        key: RunKey = (run, slot_id, model_alias)
        resumed = False
        if _cap_reached(events) and _live(_tasks.get(key)):
            # Lowered below what's already been spent while still running — stop
            # it now rather than waiting on the next settled cost to trip it.
            await _cancel_task(*key)
            slot_log.log("run.cap_reached", spend=_cell_spend(events), cap=new_cap)
        elif was_capped and not _cap_reached(events) and not _live(_tasks.get(key)):
            # The cap was the only thing holding it and the new ceiling clears
            # spend. A stepped cell gets a one-call budget so it advances like a
            # /step relaunch rather than running free.
            if key in _stepped_cells:
                _gate_intents[key] = {"budget": 1}
            await _start_cell(run, slot_id, model_alias)
            resumed = True
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
            "cap": new_cap,
            "resumed": resumed,
        }

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
    ) -> dict[str, object]:
        """The generate gate: build from-scratch (Nano-Banana + a mesh backend)
        assets for this cell's single generated build (`objects-generated/`),
        reusing the library build's layout — every object's existing
        bbox/orientation — so the client's "generated" view is an apples-to-apples
        swap of matched assets for freshly generated ones.

        Resumes in place: meshes already on disk are skipped and bookkeeping lands
        in events.generated.jsonl. Only one build per cell at a time; re-pressing
        while it's in flight returns 409 (the gate). The library build (objects/ +
        events.jsonl) is never touched."""
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        lib_log = _require_slot_log(run, slot_id, model_alias)
        run_id = _run_id(run, slot.id, model_alias)
        generation.generated_dirs(RUNS_DIR, run_id)[0].mkdir(parents=True, exist_ok=True)
        key: GenKey = (run, slot_id, model_alias)
        in_flight = _generate_tasks.get(key)
        if in_flight is not None and not in_flight.done():
            raise HTTPException(status_code=409, detail="generation already running")
        # Per-asset regenerations of this version may be in flight — that's allowed.
        # The scene build and regens serialize per-node via generation.node_lock, so
        # they never write the same asset at once.

        # Reconstruct every concrete (object/frame) node from the library log so
        # generation reuses the exact layout instead of re-running the divider.
        prompt_store.bind(prompt_store.load_run_prompts(_run_dir(run)))
        nodes = _scene_nodes_from_library(lib_log)
        if not nodes:
            raise HTTPException(
                status_code=400,
                detail="no scene to generate from; build the library scene first",
            )

        # Dedicated generated-build log for resumable bookkeeping (nano_banana /
        # threed require a bound SlotLog). Kept apart from the library log so the
        # asset toggle stays a pure folder switch and the library stream is untouched.
        gen_log = SlotLog(
            _gen_slot_id(run, slot.id, model_alias),
            generation.generated_events_path(RUNS_DIR, run_id),
        )
        gen_log.hydrate_from_disk()

        async def _do_generate() -> None:
            prompt_store.bind(prompt_store.load_run_prompts(_run_dir(run)))
            rlog.bind(gen_log)
            try:
                # Prefab grouping lives in the generated build's log; the library
                # build never groups (it matches an asset per object as built).
                decisions = await generation.ensure_scene_prefab_groups(nodes=nodes, run_id=run_id)
                await generation.generate_assets(
                    nodes=nodes, decisions=decisions, runs_dir=RUNS_DIR, run_id=run_id,
                )
            finally:
                gen_log.close()

        _generate_tasks[key] = asyncio.create_task(_do_generate())
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
            "nodes": len(nodes),
        }

    @app.get("/slots/{slot_id}/{model_alias}/generate")
    async def slot_generate_status(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        run: str | None = None,
        optimized: bool = True,
        variant: str | None = None,
    ) -> dict[str, object]:
        """Gate state for the client: whether a build/regen is in flight for this
        cell's generated build and the ids of its finished GLBs. Polled while the
        "generated" view is active so the client can enable/disable the gate and
        attach freshly-built meshes one by one as they land."""
        run = _resolve_run(run)
        _require_slot_log(run, slot_id, model_alias)
        rid = _run_id(run, slot_id, model_alias)
        key: GenKey = (run, slot_id, model_alias)
        gen_task = _generate_tasks.get(key)
        regen_task = _regen_tasks.get(key)
        regen_queue = _regen_queues.get(key)
        running = (
            (gen_task is not None and not gen_task.done())
            or (regen_task is not None and not regen_task.done())
            or (regen_queue is not None and not regen_queue.empty())
        )
        # Report the folder the client's optimized toggle is viewing — the OPTIMIZED
        # twins by default (what's served), or the raw objects-generated/ set when
        # `optimized=0`. Each carries the GLB's mtime as a token so the client can
        # detect a regenerated asset (same id, new bytes) and reload just it, plus
        # `sym`/`symWas` — the asset's current symmetry plane (none/xy/xz) and, if
        # since un-symmetrized, the plane it used to be mirrored across — so the
        # detail panel tells mirrored / un-symmetrized / never-symmetrized apart.
        gen_events_path = generation.latest_generated_events_path(RUNS_DIR, rid)
        sym_map = _generated_symmetry(gen_events_path)
        # node id -> its prefab canonical, so each mesh carries its group. The
        # client buckets meshes by `canonical` to show group membership and offer
        # link / unlink on a single object.
        canonical_of = _generated_prefab(gen_events_path)
        image_prompts = _generated_image_prompts(gen_events_path)
        raw_dir, opt_dir = generation.latest_generated_dirs(RUNS_DIR, rid)
        lite_dir = raw_dir.parent / generation.GENERATED_LITE_SUBDIR
        gen_dir = {"raw": raw_dir, "lite": lite_dir, "optimized": opt_dir}[
            _resolve_gen_variant(variant, optimized)
        ]

        def _variant(path: Path) -> tuple[str, int] | None:
            """(url, mtime) for a generated GLB, or None when it isn't on disk."""
            try:
                return f"/artifacts/{path.relative_to(RUNS_DIR).as_posix()}", path.stat().st_mtime_ns
            except OSError:
                return None

        meshes: list[dict[str, object]] = []
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
                canonical_id = canonical_of.get(mesh_id, mesh_id)
                # The pristine Trellis/Hunyuan output (pre rescale-to-bbox, pre
                # symmetry) lives in the raw dir as `<id>.raw.glb`, which the
                # per-object viewer renders (reading its uncompressed PBR maps). A
                # prefab REUSE writes no raw of its own — its mesh is re-derived
                # from the canonical's raw — so resolve the raw against the
                # CANONICAL so a reuse shows the source mesh, not its rescaled twin.
                raw_p = raw_dir / f"{canonical_id}.raw.glb"
                img_prompt = image_prompts.get(mesh_id) or image_prompts.get(canonical_id)
                # Both rescaled variants of THIS object — the optimized (KTX2 /
                # Meshopt) twin and the unoptimized ("raw") mesh — so the client can
                # flip a single object between them independently of the scene-wide
                # toggle. Either can be absent mid-build (the unoptimized lands
                # first, the optimized twin after); the client falls back.
                opt_v = _variant(opt_dir / f"{mesh_id}.glb")
                unopt_v = _variant(raw_dir / f"{mesh_id}.glb")
                lite_v = _variant(lite_dir / f"{mesh_id}.glb")
                # A served twin whose raw/unoptimized backing is missing is a torn
                # "ghost" (a regen that died midway): it still renders but can't be
                # re-derived (reorient/symmetrize/reuse) or shown in the per-object
                # raw view. Surface it so it isn't silent — and so the user knows a
                # regenerate is needed to rebuild it. (Reuses own no raw; they only
                # need their served twins.)
                incomplete = not generation.artifacts_complete(
                    raw_dir, opt_dir, mesh_id, is_reuse=canonical_id != mesh_id,
                )
                meshes.append({
                    "id": mesh_id,
                    "v": mtime,
                    "url": f"/artifacts/{p.relative_to(RUNS_DIR).as_posix()}",
                    "optUrl": opt_v[0] if opt_v else None,
                    "optV": opt_v[1] if opt_v else None,
                    "unoptUrl": unopt_v[0] if unopt_v else None,
                    "unoptV": unopt_v[1] if unopt_v else None,
                    "liteUrl": lite_v[0] if lite_v else None,
                    "liteV": lite_v[1] if lite_v else None,
                    "raw": f"/artifacts/{raw_p.relative_to(RUNS_DIR).as_posix()}" if raw_p.exists() else None,
                    "sym": info["plane"] if info else "none",
                    "symWas": info["was"] if info else None,
                    "canonical": canonical_id,
                    "imagePrompt": img_prompt,
                    "incomplete": incomplete,
                })
        ids = [m["id"] for m in meshes]
        # Node ids whose meshes are currently being built or are queued to build —
        # either sitting in the mesh-jobs global queue (waiting/processing on a
        # backend) or still queued in this cell's regen worker (not yet dequeued).
        # The client disables per-object actions on these ids so the user can't
        # double-enqueue a node that's already in flight.
        gen_slot_id = _gen_slot_id(run, slot_id, model_alias)
        busy: set[str] = threed.inflight_ids(gen_slot_id)
        if regen_queue is not None:
            for job in list(regen_queue._queue):  # type: ignore[attr-defined]
                busy.add(job.node_id)
        return {
            "running": running,
            "count": len(ids),
            "ids": ids,
            "meshes": meshes,
            "busy": sorted(busy),
        }

    @app.post("/slots/{slot_id}/{model_alias}/build-lite")
    async def slot_build_lite(  # pyright: ignore[reportUnusedFunction]
        slot_id: str, model_alias: str, run: str | None = None,
    ) -> dict[str, object]:
        """Build (or resume) this cell's LITE presentation tier —
        objects-generated/ -> objects-generated-lite/ via build_lite_assets.py — as a
        background job. Idempotent while running (returns the live job). Poll the GET."""
        run = _resolve_run(run)
        _require_slot_log(run, slot_id, model_alias)
        rid = _run_id(run, slot_id, model_alias)
        raw_dir, _ = generation.latest_generated_dirs(RUNS_DIR, rid)
        out_dir = raw_dir.parent / generation.GENERATED_LITE_SUBDIR
        total = _placed_count(raw_dir)
        if total == 0:
            raise HTTPException(
                status_code=404,
                detail="no generated build to build lite from — run ⚡ generate first",
            )
        key = (run, slot_id, model_alias)
        existing = _lite_build_jobs.get(key)
        if existing is not None and existing.get("running"):
            return dict(existing)
        job: dict[str, Any] = {
            "run": run,
            "slot": slot_id,
            "model": model_alias,
            "running": True,
            "ok": None,
            "done": 0,
            "total": total,
            "status": "pending",
            "error": None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
        }
        _lite_build_jobs[key] = job
        _lite_build_tasks[key] = asyncio.create_task(
            _run_lite_build(run, slot_id, model_alias, raw_dir, out_dir)
        )
        return dict(job)

    @app.get("/slots/{slot_id}/{model_alias}/build-lite")
    async def slot_build_lite_status(  # pyright: ignore[reportUnusedFunction]
        slot_id: str, model_alias: str, run: str | None = None,
    ) -> dict[str, object]:
        """Live lite-build state for a cell: the running job, else a disk summary."""
        run = _resolve_run(run)
        key = (run, slot_id, model_alias)
        job = _lite_build_jobs.get(key)
        if job is not None:
            return dict(job)
        rid = _run_id(run, slot_id, model_alias)
        raw_dir, _ = generation.latest_generated_dirs(RUNS_DIR, rid)
        lite_dir = raw_dir.parent / generation.GENERATED_LITE_SUBDIR
        return {
            "running": False,
            "ok": None,
            "done": _placed_count(lite_dir),
            "total": _placed_count(raw_dir),
            "status": "idle",
            "error": None,
        }

    @app.post("/slots/{slot_id}/{model_alias}/regenerate/{node_id}")
    async def slot_regenerate(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        node_id: str,
        run: str | None = None,
        propagate: bool = False,
        backend: str = "trellis",
        reuse_image: bool = False,
        regen_noun_phrase: bool = False,
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
        key: GenKey = (run, slot_id, model_alias)
        # A whole-scene generate of this version may be in flight — that's allowed
        # now. The regen and the scene build serialize per-node via
        # generation.node_lock (see _regen_worker), so they never write the same
        # asset's files at once.

        # Validate the node exists in the library layout up front (fast 404); the
        # worker reconstructs it + resolves the prefab group again at execution.
        prompt_store.bind(prompt_store.load_run_prompts(_run_dir(run)))
        if _reconstruct_node(lib_log, node_id) is None:
            raise HTTPException(
                status_code=404, detail=f"no bbox event found for node: {node_id}",
            )

        # Regenerating the noun phrase means a fresh image distilled from it, so a
        # reuse-image request can't also apply — the new phrase needs a new image.
        reuse_image = reuse_image and not regen_noun_phrase
        gen_slot_id = _gen_slot_id(run, slot.id, model_alias)
        queue = _regen_queues.setdefault(key, asyncio.Queue())
        queue.put_nowait(RegenJob(
            node_id=node_id, op="regenerate", propagate=propagate,
            backend=backend, reuse_image=reuse_image,
            regen_noun_phrase=regen_noun_phrase,
        ))
        # Surface the queued regen in the shared mesh queue panel until the
        # worker dequeues it (then generate_mesh manages its own entry). Tag the
        # backend so it lands in the right pool section (Trellis vs Hunyuan 3.1).
        threed.mark_queued(gen_slot_id, node_id, backend=backend)
        worker = _regen_tasks.get(key)
        if worker is None or worker.done():
            _regen_tasks[key] = asyncio.create_task(
                _regen_worker(run, slot.id, model_alias)
            )
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
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
        key: GenKey = (run, slot_id, model_alias)
        prompt_store.bind(prompt_store.load_run_prompts(_run_dir(run)))
        if _reconstruct_node(lib_log, node_id) is None:
            raise HTTPException(
                status_code=404, detail=f"no bbox event found for node: {node_id}",
            )
        # Reuses the regen worker's queue; the backend slot is unused for an
        # un-symmetrize (it does no API calls) so it carries the default as filler.
        # Not surfaced in the mesh queue panel — it's a local reprocess, not a
        # backend generation.
        queue = _regen_queues.setdefault(key, asyncio.Queue())
        queue.put_nowait(RegenJob(node_id=node_id, op="unsymmetrize", propagate=propagate))
        worker = _regen_tasks.get(key)
        if worker is None or worker.done():
            _regen_tasks[key] = asyncio.create_task(
                _regen_worker(run, slot.id, model_alias)
            )
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
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
        key: GenKey = (run, slot_id, model_alias)
        prompt_store.bind(prompt_store.load_run_prompts(_run_dir(run)))
        if _reconstruct_node(lib_log, node_id) is None:
            raise HTTPException(
                status_code=404, detail=f"no bbox event found for node: {node_id}",
            )
        # Same worker as un-symmetrize; the (plane, keep) ride the `sym` field of
        # the job. Not surfaced in the mesh queue panel — a local reprocess, not a
        # backend generation.
        queue = _regen_queues.setdefault(key, asyncio.Queue())
        queue.put_nowait(RegenJob(
            node_id=node_id, op="symmetrize", propagate=propagate,
            sym=(plane, keep_positive),
        ))
        worker = _regen_tasks.get(key)
        if worker is None or worker.done():
            _regen_tasks[key] = asyncio.create_task(
                _regen_worker(run, slot.id, model_alias)
            )
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
            "node_id": node_id,
            "propagate": propagate,
            "op": "symmetrize",
            "plane": plane,
            "keep_positive": keep_positive,
            "queued": True,
            "depth": queue.qsize(),
        }

    @app.post("/slots/{slot_id}/{model_alias}/reorient/{node_id}")
    async def slot_reorient(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        node_id: str,
        run: str | None = None,
        axis: str = "y",
        degrees: int = 90,
        propagate: bool = True,
    ) -> dict[str, object]:
        """Change a GENERATED object's "front view" — which face points along +Z in
        its raw, pre-transform mesh — by rotating that raw mesh 90° (`degrees`, a
        multiple of 90) about `axis` ('x' pitch, 'y' yaw, 'z' roll), then re-deriving
        its served + optimized twin. The rotation is baked into the raw, the pristine
        source every prefab reuse derives from, so with `propagate=true` (the client
        default) the prefab CANONICAL behind `node_id` is re-fronted and every object
        reusing it is re-derived to match — keeping the whole group consistent across
        the optimized and unoptimized builds. No Nano-Banana, no mesh backend.
        Reprocesses on the SAME per-version worker as regenerate/symmetrize, so it
        enqueues, drains concurrently, and serializes per-node via
        `generation.node_lock`."""
        if axis not in ("x", "y", "z"):
            raise HTTPException(status_code=400, detail=f"axis must be 'x', 'y', or 'z', got: {axis}")
        if degrees % 90 != 0:
            raise HTTPException(status_code=400, detail=f"degrees must be a multiple of 90, got: {degrees}")
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        lib_log = _require_slot_log(run, slot_id, model_alias)
        key: GenKey = (run, slot_id, model_alias)
        prompt_store.bind(prompt_store.load_run_prompts(_run_dir(run)))
        if _reconstruct_node(lib_log, node_id) is None:
            raise HTTPException(
                status_code=404, detail=f"no bbox event found for node: {node_id}",
            )
        # Same worker as symmetrize; (axis, degrees) ride the `reorient` field. Not
        # surfaced in the mesh queue panel — a local reprocess, not a backend gen.
        queue = _regen_queues.setdefault(key, asyncio.Queue())
        queue.put_nowait(RegenJob(
            node_id=node_id, op="reorient", propagate=propagate,
            reorient=(axis, degrees),
        ))
        worker = _regen_tasks.get(key)
        if worker is None or worker.done():
            _regen_tasks[key] = asyncio.create_task(
                _regen_worker(run, slot.id, model_alias)
            )
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
            "node_id": node_id,
            "propagate": propagate,
            "op": "reorient",
            "axis": axis,
            "degrees": degrees,
            "queued": True,
            "depth": queue.qsize(),
        }

    @app.post("/slots/{slot_id}/{model_alias}/glassify/{node_id}")
    async def slot_glassify(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        node_id: str,
        run: str | None = None,
    ) -> dict[str, object]:
        """Force the window/glass transparency transform onto a generated object,
        bypassing the pipeline's keyword + symmetry gates. Bakes per-texel
        transparency into the served mesh (white / near-white texels -> near-clear)
        and re-optimizes the served twin — no Nano-Banana, no mesh backend, so it's
        effectively instant. Applies to the WHOLE prefab group behind `node_id`
        (each member's served mesh directly, since glass isn't re-derivable from the
        shared raw). Operating on the served mesh, it is not raw-derivable, so a
        later regenerate / symmetrize / reorient / reset rebuilds from raw and drops
        the transparency. Reprocesses on the SAME per-version worker as
        regenerate/symmetrize, so it enqueues, drains concurrently, and serializes
        per-node via `generation.node_lock`."""
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        lib_log = _require_slot_log(run, slot_id, model_alias)
        key: GenKey = (run, slot_id, model_alias)
        prompt_store.bind(prompt_store.load_run_prompts(_run_dir(run)))
        if _reconstruct_node(lib_log, node_id) is None:
            raise HTTPException(
                status_code=404, detail=f"no bbox event found for node: {node_id}",
            )
        # Same worker as symmetrize/reorient. propagate=True: glass is applied to
        # every member of the object's prefab group. Not surfaced in the mesh queue
        # panel — a local reprocess, not a backend gen.
        queue = _regen_queues.setdefault(key, asyncio.Queue())
        queue.put_nowait(RegenJob(node_id=node_id, op="glassify", propagate=True))
        worker = _regen_tasks.get(key)
        if worker is None or worker.done():
            _regen_tasks[key] = asyncio.create_task(
                _regen_worker(run, slot.id, model_alias)
            )
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
            "node_id": node_id,
            "op": "glassify",
            "queued": True,
            "depth": queue.qsize(),
        }

    @app.post("/slots/{slot_id}/{model_alias}/reset-mesh/{node_id}")
    async def slot_reset_mesh(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        node_id: str,
        run: str | None = None,
    ) -> dict[str, object]:
        """Rebuild a GENERATED object's served mesh from its pristine raw Trellis
        output, dropping any in-place served edit (notably a forced glassify) while
        KEEPING its current symmetry/orientation. No Nano-Banana, no mesh backend,
        so it's effectively instant. Applies to the WHOLE prefab group behind
        `node_id`: the canonical is re-derived from its raw and every reuse is
        re-derived from that same clean raw (`propagate_reuses`), so the group
        reverts together. Reprocesses on the SAME per-version worker as
        regenerate/symmetrize, so it enqueues, drains concurrently, and serializes
        per-node via `generation.node_lock`. (To remove the symmetry mirror too, use
        /unsymmetrize.)"""
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        lib_log = _require_slot_log(run, slot_id, model_alias)
        key: GenKey = (run, slot_id, model_alias)
        prompt_store.bind(prompt_store.load_run_prompts(_run_dir(run)))
        if _reconstruct_node(lib_log, node_id) is None:
            raise HTTPException(
                status_code=404, detail=f"no bbox event found for node: {node_id}",
            )
        # Same worker as symmetrize/reorient. propagate=True: the whole prefab group
        # reverts to its clean raw-derived mesh. Not surfaced in the mesh queue
        # panel — a local reprocess, not a backend gen.
        queue = _regen_queues.setdefault(key, asyncio.Queue())
        queue.put_nowait(RegenJob(node_id=node_id, op="reset", propagate=True))
        worker = _regen_tasks.get(key)
        if worker is None or worker.done():
            _regen_tasks[key] = asyncio.create_task(
                _regen_worker(run, slot.id, model_alias)
            )
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
            "node_id": node_id,
            "op": "reset",
            "queued": True,
            "depth": queue.qsize(),
        }

    @app.post("/slots/{slot_id}/{model_alias}/link/{node_id}")
    async def slot_link(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        node_id: str,
        target: str,
        run: str | None = None,
        group: bool = False,
    ) -> dict[str, object]:
        """Link a GENERATED asset INTO another object's prefab group, so it shares
        that group's mesh — the inverse of `unlink`. `target` is ANY member of the
        destination group (we resolve it to that group's canonical); `node_id`
        becomes a reuse of it and its mesh is re-derived from the canonical's raw
        (rescaled into `node_id`'s own bbox/orientation) — no Nano-Banana, no mesh
        backend, so it's effectively instant. If `node_id` was itself a prefab
        canonical with reuses, those reuses move into the destination group too, so
        the flat prefab star is preserved. With `group=true` the ENTIRE source group
        (canonical + every reuse) moves into the destination, even when `node_id` is
        a reuse — its old canonical and siblings come along. Runs on the SAME
        per-version worker as regenerate/symmetrize, serialized per group via the
        worker's per-canonical lock + `generation.node_lock`. Not surfaced in the
        mesh queue panel — a local re-derivation, not a backend generation."""
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        lib_log = _require_slot_log(run, slot_id, model_alias)
        key: GenKey = (run, slot_id, model_alias)
        prompt_store.bind(prompt_store.load_run_prompts(_run_dir(run)))
        if _reconstruct_node(lib_log, node_id) is None:
            raise HTTPException(
                status_code=404, detail=f"no bbox event found for node: {node_id}",
            )
        if _reconstruct_node(lib_log, target) is None:
            raise HTTPException(
                status_code=404, detail=f"no bbox event found for link target: {target}",
            )
        queue = _regen_queues.setdefault(key, asyncio.Queue())
        queue.put_nowait(RegenJob(node_id=node_id, op="link", link_to=target, link_group=group))
        worker = _regen_tasks.get(key)
        if worker is None or worker.done():
            _regen_tasks[key] = asyncio.create_task(
                _regen_worker(run, slot.id, model_alias)
            )
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
            "node_id": node_id,
            "op": "link",
            "target": target,
            "group": group,
            "queued": True,
            "depth": queue.qsize(),
        }

    @app.post("/slots/{slot_id}/{model_alias}/unlink/{node_id}")
    async def slot_unlink(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        node_id: str,
        run: str | None = None,
    ) -> dict[str, object]:
        """Split a GENERATED asset OUT of its prefab group into a STANDALONE asset
        with its OWN raw mesh — the inverse of `link` — WITHOUT rebuilding it. A
        reuse clones its canonical's raw (inheriting its geometry + symmetry) and
        becomes its own canonical; a canonical with reuses hands the group off to
        one of them (which inherits the shared raw) and stays standalone; a lone
        canonical is already independent. No Nano-Banana, no mesh backend, so it's
        effectively instant — the object then diverges only once the user
        regenerates it. Runs on the SAME per-version worker as regenerate/link,
        serialized per group via the worker's per-canonical lock. Not surfaced in
        the mesh queue panel — a local re-derivation, not a backend generation."""
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        lib_log = _require_slot_log(run, slot_id, model_alias)
        key: GenKey = (run, slot_id, model_alias)
        prompt_store.bind(prompt_store.load_run_prompts(_run_dir(run)))
        if _reconstruct_node(lib_log, node_id) is None:
            raise HTTPException(
                status_code=404, detail=f"no bbox event found for node: {node_id}",
            )
        queue = _regen_queues.setdefault(key, asyncio.Queue())
        queue.put_nowait(RegenJob(node_id=node_id, op="unlink"))
        worker = _regen_tasks.get(key)
        if worker is None or worker.done():
            _regen_tasks[key] = asyncio.create_task(
                _regen_worker(run, slot.id, model_alias)
            )
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
            "node_id": node_id,
            "op": "unlink",
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
        if start:
            _require_run_prompts(run)
        # Reset wipes the whole cell dir, so tear this cell's branches down
        # first to avoid one writing into a dir being deleted.
        await _discard_branches_of_cell(run, slot.id, model_alias)
        await _cancel_task(run, slot_id, model_alias)
        # Tear down this cell's in-flight from-scratch build + regen worker so
        # their meshes aren't written into the dir we're about to wipe.
        await _cancel_cell_generation(run, slot.id, model_alias)
        # A reset wipes the cell, including its stepped marker + intent.
        _set_stepped((run, slot.id, model_alias), False)
        _gate_intents.pop((run, slot.id, model_alias), None)
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
            # (slot, model) cells they pick.
            slot_log.state["prompt"] = slot.prompt
            slot_log.state["model"] = MODELS[model_alias]
        return {"run": run, "slot_id": slot.id, "model": model_alias}

    @app.post("/slots/{slot_id}/{model_alias}/delete-object/{node_id}")
    async def slot_delete_object(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        node_id: str,
        run: str | None = None,
    ) -> dict[str, object]:
        """Permanently wipe ONE object from the cell — every reference to it in
        both event logs (library + generated), its mesh/image artifacts in every
        build dir, all reindexed so the logs stay replay-clean. Orphaned children
        are re-anchored onto the object's owning region; if it was a prefab
        canonical, the role is handed to one of its reuses (which inherits the
        shared raw mesh). Irreversible — there is no undo short of re-running.

        Tears the cell's branches + pipeline/generate/regen tasks down first
        (a reindex invalidates absolute branch fork indices and any in-flight
        writer would race the rewrite), exactly like reset/rewind. Leaves the
        cell otherwise intact and does NOT auto-resume."""
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        bbox_ev = next(
            (e for e in reversed(slot_log.state["events"])
             if e.get("kind") == "bbox" and e.get("id") == node_id),
            None,
        )
        if bbox_ev is None:
            raise HTTPException(
                status_code=404, detail=f"no bbox event found for node: {node_id}",
            )
        # Object-scoped tool: a zone owns a whole subtree (its objects re-anchor
        # to it on wipe), so deleting one would orphan its contents into nothing.
        # Frames (encapsulating shells) are concrete meshes and ARE wipeable.
        if bbox_ev.get("node_kind") == "zone":
            raise HTTPException(
                status_code=400,
                detail=f"{node_id} is a zone/region, not an object — cannot wipe",
            )
        await _discard_branches_of_cell(run, slot.id, model_alias)
        await _cancel_task(run, slot_id, model_alias)
        await _cancel_cell_generation(run, slot.id, model_alias)
        summary = object_wipe.wipe_object(
            node_id=node_id,
            library_log=slot_log,
            runs_dir=RUNS_DIR,
            run_id=_run_id(run, slot.id, model_alias),
        )
        if summary is None:
            raise HTTPException(
                status_code=404, detail=f"no bbox event found for node: {node_id}",
            )
        return {"run": run, "slot_id": slot.id, "model": model_alias, **summary}

    @app.post("/slots/{slot_id}/{model_alias}/branch")
    async def create_branch(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        req: BranchRequest,
        run: str | None = None,
    ) -> dict[str, object]:
        """Fork the cell into a NEW simulation branch at `event_index` (a
        re-renderable cache.llm call of template `step`) under the run snapshot +
        the lab's edited templates. MANY branches per cell coexist — fork
        different zones (or LLMs) to run several independent downstream sims of
        one slot at once. Returns the new branch id; all later control is keyed
        by it under `/branches/{id}/…`. Isolated: writes only under
        `<run>/_branches/<id>/`, never the source cell.

        With `model` (an alias) the fork is PINNED to that LLM and runs exactly
        one step before parking — the compare view's per-LLM lineage. Without it
        the branch pauses at the forked step on the cell's base model."""
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        _require_run_prompts(run)
        _validate_overrides(req.overrides)
        if req.model is not None and req.model not in MODELS:
            raise HTTPException(status_code=400, detail=f"unknown model: {req.model}")
        if req.version is not None and not prompt_store.version_exists(req.version):
            raise HTTPException(status_code=404, detail=f"unknown prompt version: {req.version}")
        # A pinned LLM lineage runs its forked step immediately (gate budget 1)
        # then parks; an unpinned branch pauses AT the forked step for a manual
        # step on the base model.
        pin = MODELS[req.model] if req.model is not None else None
        br = await _fork_branch(
            run, slot.id, model_alias,
            event_index=req.event_index, step=req.step,
            overrides=req.overrides, seed=req.seed,
            model_pin=pin, gate_budget=1 if pin else 0,
            version=req.version,
            atomic_locks=req.atomic_locks,
        )
        return {
            "branch": _branch_summary(br),
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
            "seeded": req.seed is not None,
        }

    @app.get("/runs/{run}/branches")
    async def list_branches(run: str) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Every TOP-LEVEL simulation branch of a run (parallel-LLM fan-out
        children excluded) — the prompt lab's sims list. Rehydrated branches
        come back paused/resumable."""
        run = _resolve_run(run)
        _hydrate_run(run)
        return {
            "run": run,
            "branches": [
                _branch_summary(b) for b in _branches.values()
                if b.run == run and b.parent is None
            ],
        }

    @app.get("/branches/{branch_id}")
    async def branch_status(branch_id: str) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        return _branch_summary(_require_branch(branch_id))

    @app.get("/branches/{branch_id}/scene")
    async def branch_scene(branch_id: str) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        return _scene_projection(list(_require_branch(branch_id).log.state["events"]))

    @app.get("/branches/{branch_id}/events")
    async def branch_events(branch_id: str, since: int = -1) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        blog = _require_branch(branch_id).log
        q = blog.subscribe()
        snapshot = list(blog.state["events"])
        if since >= 0:
            snapshot = [
                e for e in snapshot
                if isinstance(e.get("index"), int) and e["index"] > since
            ]
        return StreamingResponse(_sse(blog, q, snapshot), media_type="text/event-stream")

    @app.get("/branches/{branch_id}/meshes")
    async def branch_meshes(branch_id: str, since_index: int | None = None) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        """The branch's committed meshes. A branch hardlinks the prefix's source
        meshes, so serve only the ids it actually committed (never the dropped
        prefix ones). `since_index` restricts to meshes placed at/after that
        event — compare attaches just the NEW meshes a fan-out child placed over
        the shared prefix already on screen."""
        blog = _require_branch(branch_id).log
        evs = list(blog.state["events"])
        if since_index is not None:
            ids = {
                str(e["id"]) for e in evs
                if e.get("kind") == "model" and isinstance(e.get("index"), int)
                and int(e["index"]) >= since_index and isinstance(e.get("id"), str)  # type: ignore[arg-type]
            }
        else:
            ids = _committed_mesh_ids(evs)
        objects_dir = blog.events_path.parent / "objects"
        return StreamingResponse(
            _mesh_bundle(objects_dir, ids),
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/branches/{branch_id}/rewind")
    async def branch_rewind(  # pyright: ignore[reportUnusedFunction]
        branch_id: str,
        req: BranchRewindRequest,
    ) -> dict[str, object]:
        """Revert a simulation branch to before `to_event_index` and pause there
        — the branch mirror of the source `/rewind`. Re-running forward (a plain
        `/branches/{id}/step`) then regenerates the reverted call under the
        CURRENT run snapshot + refreshed edits. Scoped to the branch log: the
        source cell and the branch's hardlinked prefix meshes are untouched."""
        br = _require_branch(branch_id)
        _require_run_prompts(br.run)
        if req.overrides is not None:
            _validate_overrides(req.overrides)
        # Stop the in-flight branch task but KEEP its log / overrides / dir, then
        # rewrite the log under it (mirrors source rewind's discard-then-truncate).
        await _cancel_branch_task(br)
        blog = br.log
        events = blog.state["events"]
        cut = max(0, min(req.to_event_index, len(events)))
        # Drop the meshes of every node generated at/after the cut so a re-run
        # regenerates cleanly. Prefix meshes (before the cut, hardlinked from the
        # source) stay; unlinking a post-cut hardlink only drops the branch's
        # name, never the source's inode.
        objs = blog.events_path.parent / "objects"
        for e in events[cut:]:
            oid = e.get("id")
            if isinstance(oid, str):
                for suffix in (".glb", ".raw.glb", ".png"):
                    with contextlib.suppress(OSError):
                        (objs / f"{oid}{suffix}").unlink()
        blog.truncate_events_to(cut)
        # The task was cancelled above and the log no longer ends in a terminal
        # marker, so derive_status reports this truncated branch as paused.
        # Refresh the edit set so the next step re-runs under the lab's CURRENT
        # edits (the sim's source of truth); None keeps the existing set.
        if req.overrides is not None:
            br.overrides = {s: dict(r) for s, r in req.overrides.items()}
            _write_branch_manifest(br)
        return {"branch": branch_id, "events": len(blog.state["events"])}

    @app.post("/slots/{slot_id}/{model_alias}/step")
    async def cell_step(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        req: BranchStepRequest,
        run: str | None = None,
    ) -> dict[str, object]:
        """Advance a stepped source cell by one LLM call — releasing a live
        gate OR relaunching a paused cell with a 1-call budget, so it works
        even after a restart. `auto` runs the cell to completion. 409 only
        when there's genuinely nothing to advance (done / mid-call / no cell)."""
        run = _resolve_run(run)
        _require_slot(slot_id)
        _require_model(model_alias)
        if req.until is not None and req.until not in prompt_store.STEPS:
            raise HTTPException(status_code=400, detail=f"unknown step: {req.until}")
        result = await _advance_cell(run, slot_id, model_alias, auto=req.auto, until=req.until, until_before=req.until_before)
        if result == "capped":
            raise HTTPException(
                status_code=409,
                detail="spend cap reached — raise the cap to continue",
            )
        if result in ("missing", "done", "not_runnable", "in_flight"):
            raise HTTPException(status_code=409, detail=f"cannot step: {result}")
        return {"run": run, "slot_id": slot_id, "model": model_alias, "auto": req.auto, "result": result}

    @app.post("/runs/{run}/step-all")
    async def run_step_all(run: str, auto: bool = False, until: str | None = None, until_before: bool = False) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Advance EVERY stepped cell in the run — regardless of whether it's
        paused at a live gate, mid-call, or sitting paused with no task (the
        "move the whole experiment forward" action). Each gets ONE queued step
        (so none is skipped for being mid-call), keeping the run in lockstep.
        `auto=true` runs them all to completion; `until=<step>` runs them all to
        the next call of that step — pausing AFTER it, or BEFORE it (it doesn't
        execute) when `until_before=true`."""
        if until is not None and until not in prompt_store.STEPS:
            raise HTTPException(status_code=400, detail=f"unknown step: {until}")
        results: dict[str, list[str]] = {}
        for r, slot_id, alias in [k for k in _stepped_cells if k[0] == run]:
            res = await _advance_cell(r, slot_id, alias, auto=auto, until=until, until_before=until_before)
            results.setdefault(res, []).append(f"{slot_id}/{alias}")
        advanced = results.get("stepped", []) + results.get("launched", []) + results.get("queued", []) + results.get("seeking", [])
        return {"run": run, "advanced": advanced, "auto": auto, "until": until, "by_result": results}

    @app.post("/branches/{branch_id}/step")
    async def branch_step(  # pyright: ignore[reportUnusedFunction]
        branch_id: str,
        req: BranchStepRequest,
    ) -> dict[str, object]:
        """Advance a simulation branch by one LLM call. Like the source
        "step", this QUEUES a credit when the branch is mid-call (so a batch
        "step sims" never errors on a branch that isn't sitting at a gate),
        runs it to completion with `auto`, or fast-forwards THROUGH the next
        call of `until` (it executes) and pauses before the following one.
        `model` (an alias) re-aims the next gated call at a chosen LLM. 409 only
        when there's genuinely nothing to advance."""
        _require_branch(branch_id)
        if req.until is not None and req.until not in prompt_store.STEPS:
            raise HTTPException(status_code=400, detail=f"unknown step: {req.until}")
        if req.model is not None and req.model not in MODELS:
            raise HTTPException(status_code=400, detail=f"unknown model: {req.model}")
        model_id = MODELS[req.model] if req.model is not None else None
        result = await _advance_branch(branch_id, auto=req.auto, until=req.until, until_before=req.until_before, model=model_id)
        if result in ("missing", "done", "not_runnable"):
            raise HTTPException(status_code=409, detail=f"cannot step: {result}")
        return {"branch": branch_id, "auto": req.auto, "until": req.until, "result": result, "ran_model": req.model}

    @app.post("/branches/{branch_id}/pause")
    async def branch_pause(branch_id: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        br = _require_branch(branch_id)
        # Pausable iff a live task exists to cancel (running or parked at a gate).
        if not _live(br.task):
            raise HTTPException(status_code=400, detail="branch is not running")
        await _cancel_branch_task(br)
        br.log.log("run.paused")
        return {"branch": branch_id}

    @app.post("/branches/{branch_id}/resume")
    async def branch_resume(branch_id: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        br = _require_branch(branch_id)
        if any(e.get("kind") == "run.done" for e in br.log.state["events"]):
            raise HTTPException(status_code=409, detail="branch is complete")
        if _live(br.task):
            raise HTTPException(status_code=400, detail="branch is already running")
        events = br.log.state["events"]
        if events and events[-1].get("kind") in ("run.error", "run.paused"):
            br.log.truncate_events_to(len(events) - 1)
        br.task = asyncio.create_task(_run_branch(branch_id))
        return {"branch": branch_id}

    @app.delete("/branches/{branch_id}")
    async def discard_branch(branch_id: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        # Idempotent — a missing branch is a no-op.
        await _discard_branch(branch_id)
        return {"branch": branch_id}

    @app.post("/branches/{branch_id}/commit")
    async def commit_branch(branch_id: str) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Promote a branch to BE its origin source cell: the branch's events +
        meshes replace the cell's own, and the branch (with its children) is
        discarded. The deliberate, on-demand "replace source with the
        simulation" action — the prior source state is overwritten. Sibling
        branches of the same cell are left alone (their prefixes are independent
        copies)."""
        br = _require_branch(branch_id)
        run, slot_id, model_alias = br.run, br.slot, br.model
        src_log = _slot_logs.get((run, slot_id, model_alias))
        if src_log is None:
            raise HTTPException(status_code=404, detail="no source cell to replace")
        # Stop the source pipeline + this branch (keeping its log + dir for the copy).
        await _cancel_task(run, slot_id, model_alias)
        await _cancel_branch_task(br)
        bdir = br.log.events_path.parent
        cell_dir = src_log.events_path.parent
        src_objects = _objects_dir(cell_dir)
        # The branch's `model`/`image` events point at its own
        # `_branches/<id>/objects/` dir, which is moved + deleted below — rewrite
        # those URLs to the source cell's `objects/` so the promoted scene's
        # mesh/image URLs stay valid (the artifact route maps objects ↔
        # objects-optimized, so the bare `/objects/` segment is correct either way).
        seg = f"/{BRANCHES_SUBDIR}/{branch_id}/objects/"
        dest_seg = f"/{slot_id}/{model_alias}/objects/"
        branch_events = list(br.log.state["events"])
        src_log.close()

        def _write_promoted() -> None:
            with src_log.events_path.open("w", encoding="utf-8") as f:
                for e in branch_events:
                    url = e.get("url")
                    if isinstance(url, str) and seg in url:
                        e = {**e, "url": url.replace(seg, dest_seg)}
                    f.write(json.dumps(e) + "\n")
        await asyncio.to_thread(_write_promoted)
        # Move the branch's objects over the source's. The branch's prefix files
        # are hardlinks to the source inodes, so dropping the source dir first is
        # safe; slot_meshes filters to committed ids, so any stale
        # hardlinked-but-dropped files in the moved dir aren't served.
        branch_objects = bdir / "objects"

        def _swap_objects() -> None:
            shutil.rmtree(src_objects, ignore_errors=True)
            if branch_objects.is_dir():
                shutil.move(str(branch_objects), str(src_objects))
            else:
                src_objects.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(_swap_objects)
        src_log.hydrate_from_disk()  # reload source state from the promoted log
        await _discard_branch(branch_id)  # remove the now-consumed branch dir + state
        return {
            "branch": branch_id, "run": run, "slot_id": slot_id, "model": model_alias,
            "events": len(src_log.state["events"]),
        }

    @app.post("/versions/fork")
    async def fork_version(req: ForkVersionRequest) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        name = req.name.strip()
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(status_code=400, detail="invalid version name")
        if prompt_store.version_exists(name):
            raise HTTPException(status_code=409, detail=f"version already exists: {name}")
        if not prompt_store.version_exists(req.base):
            raise HTTPException(status_code=404, detail=f"unknown base version: {req.base}")
        try:
            prompt_store.fork_version(name, base=req.base)
        except prompt_store.PromptTemplateError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return {"name": name, "base": req.base}

    @app.put("/runs/{run}/prompt-templates")
    async def update_run_prompts(  # pyright: ignore[reportUnusedFunction]
        run: str, req: UpdateRunPromptsRequest,
    ) -> dict[str, object]:
        """Apply the prompt lab's edits to the run's snapshot in place — the
        fast iteration loop (edit → re-run step → compare) for a run that IS
        the working copy of a version being authored."""
        _require_run_prompts(run)
        # Allow a no-edit call when it's only syncing the version (push the run's
        # accumulated prompts back onto the source version without a fresh edit).
        if not req.overrides and not req.update_version:
            raise HTTPException(status_code=400, detail="no template edits to apply")
        run_snapshot = _run_dir(run) / prompt_store.RUN_PROMPTS_SUBDIR
        if req.overrides:
            try:
                prompt_store.write_overrides(run_snapshot, req.overrides)
            except prompt_store.PromptTemplateError as e:
                raise HTTPException(status_code=400, detail=str(e))
        version_synced: str | None = None
        if req.update_version:
            meta = _run_meta(run)
            version = meta.get("prompt_version")
            if isinstance(version, str) and prompt_store.version_exists(version):
                # HARD replacement: copy the run's FULL snapshot (every step) onto
                # the source version, so edits applied to the run earlier without
                # syncing land too — not just this call's overrides.
                prompt_store.sync_templates(prompt_store.VERSIONS_DIR / version, run_snapshot)
                version_synced = version
        return {
            "run": run,
            "applied": sorted(req.overrides),
            "version_synced": version_synced,
        }

    @app.post("/runs/{run}/prompt-templates/restore")
    async def restore_run_prompts(run: str, step: str | None = None, version: str | None = None) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Overwrite the run's prompt snapshot from a source version. Defaults
        to the run's BASE version (the inverse of the `update_version` sync,
        which pushes the run's snapshot ONTO the version); pass `version` to
        restore from ANY existing prompt version instead. With `step`, revert
        ONLY that step's templates (the per-prompt revert); without it,
        hard-replace every step. Either way this discards the prompt lab's
        in-place edits to the reverted step(s). The run's base
        (`run.json.prompt_version`) is left unchanged — this pulls content, it
        does not re-base the run. Running cells keep the templates they launched
        with; every later relaunch/rerun/resume renders the restored bytes."""
        _require_run_prompts(run)
        base = _run_meta(run).get("prompt_version")
        source = version if version is not None else base
        if not isinstance(source, str) or not source or not prompt_store.version_exists(source):
            if version is not None:
                raise HTTPException(status_code=404, detail=f"unknown prompt version: {version!r}")
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run {run!r} has no existing base version to restore from"
                    + (f" — {base!r} no longer exists" if isinstance(base, str) and base else "")
                ),
            )
        if step is not None and step not in prompt_store.STEPS:
            raise HTTPException(status_code=404, detail=f"unknown step: {step}")
        run_snapshot = _run_dir(run) / prompt_store.RUN_PROMPTS_SUBDIR
        version_dir = prompt_store.VERSIONS_DIR / source
        try:
            # dest = the run snapshot, src = the chosen source version — one
            # step, or all of them.
            if step is None:
                prompt_store.sync_templates(run_snapshot, version_dir)
            else:
                prompt_store.restore_step(run_snapshot, version_dir, step)
        except prompt_store.PromptTemplateError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"run": run, "restored_from": source, "step": step}

    @app.post("/runs/{run}/simulate-step")
    async def simulate_step(  # pyright: ignore[reportUnusedFunction]
        run: str, req: SimulateStepRequest,
    ) -> dict[str, object]:
        """Fork a SIMULATION branch in every targeted cell at its EARLIEST
        call of any template in `steps`, running the pipeline from there under
        the run's (freshly edited) snapshot. Non-destructive: the source cells
        are untouched, so the original output stays for comparison — promote a
        branch with /branch/commit to replace its source when you're happy.
        Omitted `cells` means every cell that logged a re-renderable call of
        one of those steps."""
        if not req.steps:
            raise HTTPException(status_code=400, detail="no steps given")
        for s in req.steps:
            if s not in prompt_store.STEPS:
                raise HTTPException(status_code=404, detail=f"unknown step: {s}")
        if not _run_dir(run).is_dir():
            raise HTTPException(status_code=404, detail=f"unknown run: {run}")
        _require_run_prompts(run)
        _hydrate_run(run)
        wanted: set[tuple[str, str]] | None = None
        if req.cells is not None:
            wanted = {(str(c.get("slot", "")), str(c.get("model", ""))) for c in req.cells}
        steps = set(req.steps)
        simulated: list[str] = []
        skipped: list[str] = []
        for (r, slot_id, alias), slot_log in list(_slot_logs.items()):
            if r != run:
                continue
            if wanted is not None and (slot_id, alias) not in wanted:
                continue
            events = slot_log.state["events"]
            # Earliest re-renderable call of any target step — its ARRAY index
            # is the fork point (matches _fork_branch / _require_step_event).
            cut: int | None = None
            cut_step: str | None = None
            for i, e in enumerate(events):
                if (e.get("kind") == "cache.llm" and e.get("template") in steps
                        and isinstance(e.get("variables"), dict)):
                    cut = i
                    cut_step = str(e.get("template"))
                    break
            if cut is None or cut_step is None:
                skipped.append(f"{slot_id}/{alias}")
                continue
            # The run snapshot already carries the lab's edits (the apply step
            # writes them first), so the branch needs no per-call overrides.
            try:
                await _fork_branch(run, slot_id, alias, event_index=cut, step=cut_step, overrides={})
            except HTTPException:
                skipped.append(f"{slot_id}/{alias}")
                continue
            simulated.append(f"{slot_id}/{alias}")
        return {"run": run, "steps": sorted(steps), "simulated": simulated, "skipped": skipped}

    @app.post("/versions/save")
    async def save_version(req: SaveVersionRequest) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        """Persist the prompt lab's edit as a new source version: the base
        run's snapshot plus the edited step templates."""
        name = req.name.strip()
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(status_code=400, detail="invalid version name")
        if prompt_store.version_exists(name):
            raise HTTPException(status_code=409, detail=f"version already exists: {name}")
        _require_run_prompts(req.base_run)
        try:
            prompt_store.save_version(
                name,
                base_dir=_run_dir(req.base_run) / prompt_store.RUN_PROMPTS_SUBDIR,
                overrides=req.overrides,
            )
        except prompt_store.PromptTemplateError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"name": name}

    @app.post("/runs/from-branches")
    async def save_run_from_branches(req: SaveRunRequest) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Materialize the prompt lab's simulation as a NEW run: snapshot =
        base run's prompts + overrides, and each listed cell's branch state
        copied in as that cell's history (paused/finished alike — the new
        snapshot matches the prompts the branches actually ran with, so they
        resume cleanly)."""
        name = req.name.strip()
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(status_code=400, detail="invalid run name")
        run_dir = _run_dir(name)
        if run_dir.exists():
            raise HTTPException(status_code=409, detail=f"run already exists: {name}")
        _require_run_prompts(req.base_run)

        copied: list[str] = []
        skipped: list[str] = []
        run_dir.mkdir(parents=True)
        try:
            prompt_store.save_snapshot_with_overrides(
                base_dir=_run_dir(req.base_run) / prompt_store.RUN_PROMPTS_SUBDIR,
                dest=run_dir / prompt_store.RUN_PROMPTS_SUBDIR,
                overrides=req.overrides,
            )
        except prompt_store.PromptTemplateError as e:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(e))
        base_meta = _run_meta(req.base_run)
        label = req.version_label or f"{base_meta.get('prompt_version') or req.base_run}+edit"
        (run_dir / RUN_META_NAME).write_text(
            json.dumps(
                {
                    "prompt_version": label,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "branched_from": req.base_run,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        seen_cells: set[tuple[str, str]] = set()
        for branch_id in req.branches:
            br = _branches.get(branch_id)
            if br is None or br.run != req.base_run:
                skipped.append(branch_id)
                continue
            cell = (br.slot, br.model)
            if cell in seen_cells:  # two kept branches on one cell — later wins
                skipped.append(branch_id)
                continue
            bdir = _branch_dir(br.run, branch_id)
            if not (bdir / "events.jsonl").is_file():
                skipped.append(branch_id)
                continue
            seen_cells.add(cell)
            # Stop the live task first so the copied log is quiescent. The branch
            # itself stays in the base run's temp folder, untouched.
            await _cancel_branch_task(br)
            dest = run_dir / br.slot / br.model
            dest.mkdir(parents=True, exist_ok=True)
            # Rewrite the branch-dir object URLs to the NEW run's cell location, so
            # the copied log's mesh/image URLs resolve there. (GLBs load via the
            # id-keyed bundle regardless; this keeps image-hover URLs valid too.)
            seg = f"/{req.base_run}/{BRANCHES_SUBDIR}/{branch_id}/objects/"
            dest_seg = f"/{name}/{br.slot}/{br.model}/objects/"
            branch_events = list(br.log.state["events"])
            dest_events = dest / "events.jsonl"

            def _write_cell(evs=branch_events, path=dest_events, _seg=seg, _dest=dest_seg) -> None:
                with path.open("w", encoding="utf-8") as f:
                    for e in evs:
                        url = e.get("url")
                        if isinstance(url, str) and _seg in url:
                            e = {**e, "url": url.replace(_seg, _dest)}
                        f.write(json.dumps(e) + "\n")
            await asyncio.to_thread(_write_cell)
            await asyncio.to_thread(_hardlink_tree, bdir / "objects", dest / "objects")
            copied.append(f"{br.slot}/{br.model}")
        _hydrate_run(name)
        global _current_run
        _current_run = name
        return {"current": name, "copied": copied, "skipped": skipped}

    @app.post("/runs/ab-test")
    async def ab_test_run(req: AbTestRequest) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Launch a NEW run (B) seeded with a source run's ROOT zone plans.
        Each selected (slot, model) cell's log is copied through its root
        `divider.zone_plan` and nothing else, then the cell is started on the
        chosen prompt version. The resumable divider replays that one committed
        plan verbatim and re-derives the whole scene below it under B's
        prompts — an A/B that holds the top-level plan fixed and varies
        everything downstream. With `include_overall_bbox` the copied prefix
        extends through the root's overall bounding box as well, so B also holds
        the scene canvas fixed and varies only what fills it. Non-destructive to
        the source run (read-only)."""
        name = req.name.strip()
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(status_code=400, detail="invalid run name")
        version = req.prompt_version.strip()
        if not prompt_store.version_exists(version):
            raise HTTPException(status_code=404, detail=f"unknown prompt version: {version}")
        try:
            prompt_store.validate_version(version)
        except prompt_store.PromptTemplateError as e:
            raise HTTPException(status_code=409, detail=str(e))
        if not _run_dir(req.source_run).is_dir():
            raise HTTPException(status_code=404, detail=f"unknown source run: {req.source_run}")
        run_dir = _run_dir(name)
        if run_dir.exists():
            raise HTTPException(status_code=409, detail=f"run already exists: {name}")
        # Read source cells from memory (idempotent hydrate) so a still-running
        # source run's live log can't hand us a torn final line.
        _hydrate_run(req.source_run)

        run_dir.mkdir(parents=True)
        try:
            prompt_store.snapshot_into_run(version, run_dir)
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        (run_dir / RUN_META_NAME).write_text(
            json.dumps(
                {
                    "prompt_version": version,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "ab_from": req.source_run,
                    "ab_include_overall_bbox": req.include_overall_bbox,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        seeded: list[tuple[str, str]] = []
        skipped: list[str] = []
        seen: set[tuple[str, str]] = set()
        for cell in req.cells:
            key = (cell.slot, cell.model)
            if key in seen:
                continue
            seen.add(key)
            if cell.slot not in SLOTS_BY_ID or cell.model not in MODELS:
                skipped.append(f"{cell.slot}/{cell.model}")
                continue
            src = _slot_logs.get((req.source_run, cell.slot, cell.model))
            events = src.state["events"] if src is not None else []
            cut = _root_plan_cut(events, through_overall_bbox=req.include_overall_bbox)
            if cut is None:
                skipped.append(f"{cell.slot}/{cell.model}")
                continue
            dest = _slot_dir(name, cell.slot, cell.model)
            dest.mkdir(parents=True, exist_ok=True)
            with (dest / "events.jsonl").open("w", encoding="utf-8") as f:
                for e in events[:cut]:
                    f.write(json.dumps(e) + "\n")
            seeded.append(key)
        if not seeded:
            # Nothing to launch — don't leave an empty A/B run behind.
            shutil.rmtree(run_dir, ignore_errors=True)
            need = (
                "root zone plan and overall bounding box"
                if req.include_overall_bbox
                else "root zone plan"
            )
            raise HTTPException(
                status_code=400,
                detail=f"no selected cell had a committed {need} to seed from",
            )
        # Hydrate B (picks up the seeded prefixes) then start each seeded cell:
        # the divider replays the copied root plan and runs the rest fresh.
        _hydrate_run(name)
        for slot_id, model_alias in seeded:
            await _start_cell(name, slot_id, model_alias)
        global _current_run
        _current_run = name
        return {
            "current": name,
            "seeded": [f"{s}/{m}" for s, m in seeded],
            "skipped": skipped,
        }

    @app.post("/runs/{run}/copy-slot")
    async def copy_slot(run: str, req: CopySlotRequest) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Copy an entire slot folder (all its model cells + meshes) from
        `source_run` into this run, OVERWRITING this run's slot dir. Every live
        task / generate-regen worker / sim branch on the destination's cells for
        this slot is torn down first (as reset does), then the source slot is
        copied over and object URLs in the copied logs are rewritten to this run
        so meshes/images resolve here. The source run is untouched."""
        dest_run = run
        source_run = req.source_run
        slot = _require_slot(req.slot)
        if source_run == dest_run:
            raise HTTPException(status_code=400, detail="source and destination runs are the same")
        if not _run_dir(source_run).is_dir():
            raise HTTPException(status_code=404, detail=f"unknown source run: {source_run}")
        if not _run_dir(dest_run).is_dir():
            raise HTTPException(status_code=404, detail=f"unknown run: {dest_run}")
        src_slot_dir = _run_dir(source_run) / slot.id
        if not src_slot_dir.is_dir() or not any(
            (src_slot_dir / alias / "events.jsonl").is_file() for alias in MODEL_ALIASES
        ):
            raise HTTPException(status_code=400, detail=f"slot {slot.id!r} has no data in run {source_run!r}")
        _hydrate_run(source_run)
        _hydrate_run(dest_run)
        # Tear down every destination cell under this slot so nothing writes into
        # the dir we're about to replace (same teardown reset does, applied to the
        # whole slot row), and note which had data — the "replaced" report.
        replaced: list[str] = []
        for alias in MODEL_ALIASES:
            key: RunKey = (dest_run, slot.id, alias)
            dest_log = _slot_logs.get(key)
            if dest_log is not None and dest_log.state["events"]:
                replaced.append(f"{slot.id}/{alias}")
            await _discard_branches_of_cell(dest_run, slot.id, alias)
            await _cancel_task(dest_run, slot.id, alias)
            await _cancel_cell_generation(dest_run, slot.id, alias)
            _set_stepped(key, False)
            _gate_intents.pop(key, None)
            if dest_log is not None:
                dest_log.close()
                _slot_logs.pop(key, None)
        dest_slot_dir = _run_dir(dest_run) / slot.id

        def _copy() -> None:
            shutil.rmtree(dest_slot_dir, ignore_errors=True)
            shutil.copytree(src_slot_dir, dest_slot_dir)
            # Repoint object/image URLs from the source cell path to this run's,
            # per model cell (GLBs load by id regardless; this keeps image-hover
            # URLs valid and self-contained even if the source run is deleted).
            for alias in MODEL_ALIASES:
                events_path = dest_slot_dir / alias / "events.jsonl"
                if not events_path.is_file():
                    continue
                src_seg = f"/{source_run}/{slot.id}/{alias}/objects/"
                dst_seg = f"/{dest_run}/{slot.id}/{alias}/objects/"
                text = events_path.read_text(encoding="utf-8")
                if src_seg in text:
                    events_path.write_text(text.replace(src_seg, dst_seg), encoding="utf-8")

        await asyncio.to_thread(_copy)
        # Rebuild the destination slot's SlotLogs from the copied logs (mirrors
        # _hydrate_run for this slot), restoring any stepped markers the copy
        # brought over. Nothing is auto-launched — the cells come in whatever
        # state their copied log implies.
        copied: list[str] = []
        for alias in MODEL_ALIASES:
            cell_dir = dest_slot_dir / alias
            cell_dir.mkdir(parents=True, exist_ok=True)
            key = (dest_run, slot.id, alias)
            new_log = SlotLog(_run_id(dest_run, slot.id, alias), cell_dir / "events.jsonl")
            new_log.hydrate_from_disk()
            _slot_logs[key] = new_log
            _maybe_launch(slot, alias, new_log)
            if (cell_dir / ".stepped").exists():
                _stepped_cells.add(key)
            if new_log.state["events"]:
                copied.append(f"{slot.id}/{alias}")
        return {
            "run": dest_run,
            "source_run": source_run,
            "slot": slot.id,
            "copied": copied,
            "replaced": replaced,
        }

    @app.post("/runs/{run}/copy-cell")
    async def copy_cell(run: str, req: CopyCellRequest) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Copy one (slot, model) cell into `(run, slot, dest_model)`, overwriting
        it. Source and destination may differ in run and/or model; the slot is
        shared. The destination cell's live task / generation / branches are torn
        down first (as reset does), then the source cell dir is copied over and its
        object/image URLs are repointed to the destination path so meshes/images
        resolve here. The source is untouched; model fields are left as the
        source's, so cost/usage stay attributed to the model that produced them
        (a later resume runs the destination model — it's alias-derived)."""
        dest_run = run
        slot = _require_slot(req.slot)
        _require_model(req.source_model)
        _require_model(req.dest_model)
        source_run = req.source_run
        if source_run == dest_run and req.source_model == req.dest_model:
            raise HTTPException(status_code=400, detail="source and destination are the same cell")
        if not _run_dir(source_run).is_dir():
            raise HTTPException(status_code=404, detail=f"unknown source run: {source_run}")
        if not _run_dir(dest_run).is_dir():
            raise HTTPException(status_code=404, detail=f"unknown run: {dest_run}")
        src_cell_dir = _slot_dir(source_run, slot.id, req.source_model)
        if not (src_cell_dir / "events.jsonl").is_file():
            raise HTTPException(
                status_code=400,
                detail=f"cell {slot.id}/{req.source_model} has no data in run {source_run!r}",
            )
        _hydrate_run(source_run)
        _hydrate_run(dest_run)
        # Tear down the destination cell so nothing writes into the dir we're about
        # to replace (the same teardown reset does), noting whether it held data.
        dest_key: RunKey = (dest_run, slot.id, req.dest_model)
        dest_log = _slot_logs.get(dest_key)
        replaced = bool(dest_log is not None and dest_log.state["events"])
        await _discard_branches_of_cell(dest_run, slot.id, req.dest_model)
        await _cancel_task(dest_run, slot.id, req.dest_model)
        await _cancel_cell_generation(dest_run, slot.id, req.dest_model)
        _set_stepped(dest_key, False)
        _gate_intents.pop(dest_key, None)
        if dest_log is not None:
            dest_log.close()
            _slot_logs.pop(dest_key, None)
        dest_cell_dir = _slot_dir(dest_run, slot.id, req.dest_model)

        def _copy() -> None:
            shutil.rmtree(dest_cell_dir, ignore_errors=True)
            shutil.copytree(src_cell_dir, dest_cell_dir)
            # Repoint object/image URLs from the source cell path to the dest's —
            # both the run and the model segment change (GLBs load by id from the
            # bundle regardless; this keeps image-hover URLs valid + self-contained).
            events_path = dest_cell_dir / "events.jsonl"
            src_seg = f"/{source_run}/{slot.id}/{req.source_model}/objects/"
            dst_seg = f"/{dest_run}/{slot.id}/{req.dest_model}/objects/"
            text = events_path.read_text(encoding="utf-8")
            if src_seg in text:
                events_path.write_text(text.replace(src_seg, dst_seg), encoding="utf-8")

        await asyncio.to_thread(_copy)
        # Rebuild the destination cell's SlotLog from the copied log (mirrors
        # _hydrate_run for one cell), restoring any stepped marker the copy brought
        # over. Nothing auto-launches — it comes in whatever state its log implies.
        new_log = SlotLog(_run_id(dest_run, slot.id, req.dest_model), dest_cell_dir / "events.jsonl")
        new_log.hydrate_from_disk()
        _slot_logs[dest_key] = new_log
        _maybe_launch(slot, req.dest_model, new_log)
        if (dest_cell_dir / ".stepped").exists():
            _stepped_cells.add(dest_key)
        return {
            "run": dest_run,
            "source_run": source_run,
            "slot": slot.id,
            "source_model": req.source_model,
            "dest_model": req.dest_model,
            "events": len(new_log.state["events"]),
            "replaced": replaced,
        }

    @app.get("/versions")
    async def list_versions() -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """The source prompt versions (subfolders of `versions/`) a new run
        can be created from."""
        return {"versions": [{"name": n} for n in prompt_store.list_versions()]}

    return app


def _last_step(events: list[dict[str, object]]) -> dict[str, object] | None:
    """The most recent pipeline-location marker — what the board cards show
    as "where is this cell right now"."""
    for e in reversed(events):
        if e.get("kind") == "step":
            return {"node": e.get("node"), "phase": e.get("phase")}
    return None


def _usage_summary(events: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """Per-model token, request, USD-cost, and unresolved-lookup totals from a
    cell's `cache.llm` events, so the UI cost tracker can show the run's
    authoritative spend and how much of it is still settling. Returns
    `{ model_id: {"in": tokens_in, "out": tokens_out, "req": n, "cost": usd,
    "pending": n} }`.

    `cost` is OpenRouter's own settled `total_cost`, which lands a beat after the
    call (its /generation stats lag the completion) as a separate `llm.cost`
    event keyed by `generation_id` — written by the backfill sweep. We index
    those, then attribute each to its `cache.llm` call's model — so the cost
    lands on the right model, and a cost whose call was since dropped (a rewind
    that raced the lookup) is ignored. `pending` counts calls carrying a
    `generation_id` that the sweep hasn't priced yet; it drains to 0 once every
    call is resolved (the run's spend has caught up). A call with no
    `generation_id` (a legacy log) is neither priced nor pending — it just
    counts as a request, matching the tracker's old request-only degradation."""
    cost_by_gen: dict[str, float] = {}
    for e in events:
        if e.get("kind") != "llm.cost":
            continue
        gid, c = e.get("generation_id"), e.get("cost")
        if isinstance(gid, str) and isinstance(c, (int, float)):
            cost_by_gen[gid] = float(c)
    usage: dict[str, dict[str, float]] = {}
    for e in events:
        if e.get("kind") != "cache.llm":
            continue
        model = str(e.get("model") or "?")
        u = usage.setdefault(model, {"in": 0, "out": 0, "req": 0, "cost": 0.0, "pending": 0})
        ti, to = e.get("tokens_in"), e.get("tokens_out")
        u["in"] += int(ti) if isinstance(ti, (int, float)) else 0
        u["out"] += int(to) if isinstance(to, (int, float)) else 0
        u["req"] += 1
        gid = e.get("generation_id")
        if isinstance(gid, str):
            if gid in cost_by_gen:
                u["cost"] += cost_by_gen[gid]
            else:
                u["pending"] += 1
    return usage


# --- per-cell spend cap -------------------------------------------------------
# A cell's authoritative spend is the sum of its settled `llm.cost` events (the
# backfill prices each `cache.llm` against OpenRouter's own total). The ceiling
# is the value carried by the cell's latest `run.cap_override` event, or
# SPEND_CAP_USD when it has none — so setting a new cap is just appending an
# override the reader picks up, and a rewind past it falls back to the prior one.
# Everything is derived from the durable log, so it survives restarts. A ceiling
# of 0 (or less) means uncapped.


def _cell_spend(events: list[dict[str, object]]) -> float:
    total = 0.0
    for e in events:
        if e.get("kind") == "llm.cost":
            c = e.get("cost")
            if isinstance(c, (int, float)):
                total += float(c)
    return total


def _cap_override_value(events: list[dict[str, object]]) -> float | None:
    """The ceiling set by the cell's most recent `run.cap_override`, or None if
    it was never overridden — the last one wins."""
    value = None
    for e in events:
        if e.get("kind") == "run.cap_override":
            c = e.get("cap")
            if isinstance(c, (int, float)):
                value = float(c)
    return value


def _effective_cap(events: list[dict[str, object]]) -> float:
    """The cell's current ceiling: its latest explicit override, else the
    SPEND_CAP_USD default. A value ≤ 0 means uncapped."""
    override = _cap_override_value(events)
    return override if override is not None else SPEND_CAP_USD


def _cap_reached(events: list[dict[str, object]]) -> bool:
    """True when the cell has a positive ceiling and settled spend has hit it —
    THE source of truth for the `capped` status. The `run.cap_reached` event is
    only a durable notice, never the condition itself, so raising the cap past
    the spend un-caps the cell without any log surgery."""
    cap = _effective_cap(events)
    return cap > 0 and _cell_spend(events) >= cap


def _cap_summary(events: list[dict[str, object]]) -> dict[str, object] | None:
    """The cost tracker's per-cell cap panel: settled `spend` and the current
    `limit` ceiling (0 = uncapped). None only when the cap system is off
    (SPEND_CAP_USD ≤ 0), so there is no cap to show or set."""
    if SPEND_CAP_USD <= 0:
        return None
    return {"spend": _cell_spend(events), "limit": _effective_cap(events)}


def _slot_summary(slot: Slot, run: str) -> dict[str, object]:
    runs: dict[str, dict[str, object]] = {}
    for alias in MODEL_ALIASES:
        key: RunKey = (run, slot.id, alias)
        slot_log = _slot_logs.get(key)
        state = slot_log.state if slot_log is not None else {"events": []}
        events = state.get("events", [])
        # Every TOP-LEVEL simulation forked from this cell (fan-out children are
        # excluded — they live transiently inside the compare view). A cell can
        # now carry several at once (different zones / parallel sims).
        branches = [
            _branch_summary(b) for b in _branches.values()
            if b.parent is None and b.run == run and b.slot == slot.id and b.model == alias
        ]
        cgate = _cell_gates.get(key)
        runs[alias] = {
            "status": _cell_status(key, slot_log) if slot_log is not None else "idle",
            "events_count": len(events),
            "last_kind": events[-1]["kind"] if events else None,
            "last_step": _last_step(events),
            "stepped": (run, slot.id, alias) in _stepped_cells,
            "pending": cgate.pending if cgate is not None else None,
            "current": cgate.current if cgate is not None else None,
            "auto": cgate.auto if cgate is not None else False,
            "branches": branches,
            # Per-model token/request totals for the cost tracker (the run's
            # actual spend = source cells; branch simulations aren't counted).
            "usage": _usage_summary(events),
            # Per-cell spend cap panel: settled spend, current ceiling, whether
            # it's tripped, and the override count. None when the cap is off.
            "cap": _cap_summary(events),
        }
    return {
        "id": slot.id,
        "prompt": slot.prompt,
        "runs": runs,
    }


def _maybe_launch(slot: Slot, model_alias: str, slot_log: SlotLog) -> None:
    """Pre-seed each (slot, model) cell for manual start; nothing auto-runs at
    boot — the user clicks start/resume/retry per cell. A fresh cell gets its
    seed prompt + model prefilled (so a later /resume can start_run without the
    client resending them). Status is no longer stamped here: it's derived live
    by `derive_status`, so a cell killed mid-run reads as paused (no live task)
    and a completed/errored cell reads done/error straight from its log — no
    boot-time fix-up."""
    if not slot_log.state["events"]:
        slot_log.state["prompt"] = slot.prompt
    slot_log.state["model"] = MODELS[model_alias]


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
        view = sym_svc.image_view_for(
            cut_plane=cut_plane, encapsulating=bool(encapsulating),
        )
        image_prompt = scene_context.wrap_image_prompt(subject_str, proxy_shape, bbox.size, view=view)
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


def _node_seed_prompt(slot_log: SlotLog, node_id: str) -> str | None:
    """The object's AUTHORED seed prompt — the `bbox` event's `prompt`, which is
    always the decompose seed (`spec.prompt`), never the distilled noun phrase
    (the `image` event holds that). Used by the regenerate-noun-phrase path to
    re-distill a fresh phrase from the original description. Latest bbox wins."""
    seed: str | None = None
    for event in slot_log.state["events"]:
        if event.get("kind") == "bbox" and event.get("id") == node_id:
            p = event.get("prompt")
            if isinstance(p, str):
                seed = p
    return seed


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


def _scene_nodes_full(slot_log: SlotLog, *, image_log: SlotLog | None = None) -> list[Node]:
    """Reconstruct the COMPLETE scene tree (zones + objects/frames) as `Node`s from
    a cell's library log — enough for the scene-context renderers
    (`scene_context.zone_vars` / `render_embedded_block`): each node carries its
    bbox, seed `prompt`, distilled `noun_phrase`, `plan` (zones), `placement` /
    `parent_kind` / `referenced_ids` / orientation text, `parent_region` (objects),
    and `is_zone`.

    Unlike `_scene_nodes_from_library` (objects only, minimal fields for re-mesh),
    this rebuilds the FULL tree so a regenerate-noun-phrase pass can show the same
    scene context the original `image_prompt` step saw. It reads events directly
    (not the bound-log `committed.*`), so it works against any cell's log.
    Best-effort: malformed / legacy events are skipped rather than raising.

    `noun_phrase` is the distilled subject phrase from each node's `image` event,
    taken to be distinct from the seed. The phrase is sourced from `slot_log`, but
    when `image_log` is given (the GENERATED build's log) its `image` events take
    precedence — that's where a generated/regenerated object's freshly-distilled
    phrase lands, while the library log often only stores the verbose seed. Without
    this overlay a regenerated object's noun phrase would never surface in the
    scene context (the library `image` event still equals its seed)."""
    events = slot_log.state["events"]

    # bbox per id (latest wins), tracking first-seen order so the rendered tree is
    # deterministic and roughly matches creation order.
    bbox_ev: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for e in events:
        if e.get("kind") != "bbox":
            continue
        nid = e.get("id")
        if not isinstance(nid, str):
            continue
        if nid not in bbox_ev:
            order.append(nid)
        bbox_ev[nid] = e

    zone_plans: dict[str, str] = {}
    subregion_specs: dict[str, Any] = {}
    obj_specs: dict[str, Any] = {}
    obj_region: dict[str, str] = {}
    image_subj: dict[str, str] = {}
    for e in events:
        kind = e.get("kind")
        if kind == "divider.zone_plan":
            nid, plan = e.get("node"), e.get("plan")
            if isinstance(nid, str) and isinstance(plan, str):
                zone_plans[nid] = plan
        elif kind == "divider.zone_decompose":
            for c in e.get("children") or []:
                if isinstance(c, dict) and isinstance(c.get("id"), str):
                    try:
                        subregion_specs[c["id"]] = schemas.SubregionSpec.model_validate(c)
                    except Exception:
                        pass
        elif kind == "generation.decompose":
            zone = e.get("zone")
            for o in e.get("objects") or []:
                if isinstance(o, dict) and isinstance(o.get("id"), str):
                    try:
                        obj_specs[o["id"]] = schemas.ObjectSpec.model_validate(o)
                        if isinstance(zone, str):
                            obj_region[o["id"]] = zone
                    except Exception:
                        pass
        elif kind == "generation.next":
            zone, o = e.get("zone"), e.get("object")
            if isinstance(o, dict) and isinstance(o.get("id"), str):
                try:
                    obj_specs[o["id"]] = schemas.ObjectSpec.model_validate(o)
                    if isinstance(zone, str):
                        obj_region[o["id"]] = zone
                except Exception:
                    pass
        elif kind == "image":
            nid, p = e.get("id"), e.get("prompt")
            if isinstance(nid, str) and isinstance(p, str):
                image_subj[nid] = p
    # The generated build's `image` events carry the distilled noun phrases for
    # objects built/regenerated there; let them win over the library log's (which
    # frequently just re-store the seed), so a generated noun phrase shows up.
    if image_log is not None:
        for e in image_log.state["events"]:
            if e.get("kind") == "image":
                nid, p = e.get("id"), e.get("prompt")
                if isinstance(nid, str) and isinstance(p, str):
                    image_subj[nid] = p

    nodes: list[Node] = []
    for nid in order:
        be = bbox_ev[nid]
        origin, dims = be.get("origin"), be.get("dimensions")
        if not isinstance(origin, list) or not isinstance(dims, list):
            continue
        try:
            box = BoundingBox(
                origin=(float(origin[0]), float(origin[1]), float(origin[2])),
                dimensions=(float(dims[0]), float(dims[1]), float(dims[2])),
            )
        except Exception:
            continue
        is_zone = be.get("node_kind") == "zone"
        proxy_raw = be.get("proxy_shape")
        proxy_shape = ProxyShape(proxy_raw) if isinstance(proxy_raw, str) else None
        o_raw = be.get("orientation", 0)
        node_orientation: Orientation = int(o_raw) if isinstance(o_raw, (int, float, str)) else 0  # type: ignore[assignment]
        seed = be.get("prompt")
        seed = seed if isinstance(seed, str) else ""
        parent_id = be.get("parent_id")
        parent_id = parent_id if isinstance(parent_id, str) else None
        # The `image` event's prompt is the distilled noun phrase on current runs
        # (== the seed on legacy ones); only surface it when it actually differs,
        # so a legacy object doesn't show a redundant noun_phrase line.
        img = image_subj.get(nid)
        noun_phrase = img if (img and img != seed) else None
        spec = subregion_specs.get(nid) if is_zone else obj_specs.get(nid)
        nodes.append(Node(
            id=nid,
            prompt=seed,
            noun_phrase=noun_phrase,
            bbox=box,
            proxy_shape=proxy_shape,
            orientation=node_orientation,
            orientation_description=("" if is_zone else str(getattr(spec, "orientation", "") or "")),
            placement=getattr(spec, "placement", None) if spec is not None else None,
            referenced_ids=list(getattr(spec, "referenced_ids", []) or []) if spec is not None else [],
            parent_id=parent_id,
            # parent_kind is rendered only for objects (from the spec); zones don't
            # surface it, so leaving it None there is harmless.
            parent_kind=(None if is_zone else getattr(spec, "parent_kind", None)),
            parent_region=(None if is_zone else obj_region.get(nid)),
            plan=(zone_plans.get(nid) if is_zone else None),
            is_zone=is_zone,
        ))
    return nodes


# Mechanical per-object service steps (asset matching + noun-phrase distillation)
# — NOT spatial reasoning, and by far the most numerous LLM calls. Omitted from
# the investigator context entirely (the step gate flags them the same way).
_INVESTIGATOR_SKIP_STEPS = {"image_prompt", "library_match"}


def _investigator_bundle(slot_log: SlotLog) -> dict[str, object]:
    """The per-slot investigator's faithful base grounding for one cell (source
    or branch): the FULL final scene rendered by the same `scene_context`
    builders the pipeline injects with (byte-faithful to `{SCENE_CONTEXT}`), plus
    a chronological OUTLINE of every divider step that ran — just its template and
    the node it ran on (with the call index). Outputs, reasoning, and rendered
    bytes are intentionally NOT included per step: they are the bulk of the
    context, so the client shows them only for the steps the user puts in FOCUS
    (resolved from its own event log). Mechanical service steps (image_prompt /
    library_match) are omitted. The client composes this with the (separately
    fetched) prompt templates + the static pipeline explainer."""
    nodes = _scene_nodes_full(slot_log)
    root = next((n for n in nodes if n.parent_id is None), None)
    steps: list[dict[str, object]] = []
    for e in slot_log.state["events"]:
        if e.get("kind") != "cache.llm":
            continue
        template = e.get("template") or e.get("step")
        if template in _INVESTIGATOR_SKIP_STEPS:
            continue
        steps.append({
            "index": e.get("index"),
            "step": e.get("step"),
            "template": template,
            "node": e.get("node"),
        })
    return {
        "prompt": root.prompt if root is not None else "",
        "scene_context": scene_context.render_embedded_block(nodes),
        "root_header": scene_context._root_scene_header(root) if root is not None else "",
        "root_objects": scene_context.render_root_objects(nodes),
        "steps": steps,
    }


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
    _cell_gates.pop((run, slot_id, model_alias), None)
    task = _tasks.pop((run, slot_id, model_alias), None)
    if task is None or task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def _cancel_cell_generation(run: str, slot_id: str, model_alias: str) -> None:
    """Tear down the cell's from-scratch generated build: its in-flight whole-scene
    generate, its regen worker, and its queued regen jobs, then drop the per-node
    build locks and cancel any detached mesh retries. Leaves the generated log /
    dirs on disk — only the live tasks are stopped, so nothing keeps writing into
    files a reset or object-wipe is about to remove. Shared by `reset` and
    `delete-object`."""
    cell = (run, slot_id, model_alias)
    for gkey in [k for k in _generate_tasks if k[:3] == cell]:
        gen_task = _generate_tasks.pop(gkey, None)
        if gen_task is not None and not gen_task.done():
            gen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await gen_task
    for gkey in [k for k in _regen_tasks if k[:3] == cell]:
        regen_task = _regen_tasks.pop(gkey, None)
        if regen_task is not None and not regen_task.done():
            regen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await regen_task
    for gkey in [k for k in _regen_queues if k[:3] == cell]:
        _regen_queues.pop(gkey, None)
    run_id = _run_id(run, slot_id, model_alias)
    generation.clear_node_locks(run_id)
    generation.cancel_pending(run_id)


def _objects_dir(cell_dir: Path) -> Path:
    """The cell's mesh dir — `objects/`, or the migrated `objects-optimized/`."""
    primary = cell_dir / OBJECTS_SUBDIR
    if primary.is_dir():
        return primary
    return next(
        (cell_dir / d for d in ("objects-optimized", "objects") if (cell_dir / d).is_dir()),
        primary,
    )


async def _advance_cell(run: str, slot_id: str, model_alias: str, *, auto: bool, until: str | None = None, until_before: bool = False) -> str:
    """Advance a stepped source cell. Three modes:
      * `auto`   — run it to completion, ungated.
      * `until`  — fast-forward to the next call of template `until`, pausing
                   AFTER it (through the call) or, when `until_before`, BEFORE
                   it (the call doesn't execute).
      * default  — advance by exactly one LLM call.

    Works from EITHER state, which is what keeps a multi-model run in sync:
      * live gate (task running, paused OR mid-call): release a current pause,
        else QUEUE a credit (budget++) the cell spends at its next gate — so a
        model still mid-call doesn't miss a "step (all)".
      * paused/idle/error with no task → relaunch the pipeline carrying the
        intent so it replays the committed prefix then stops appropriately.
    Returns a status string for the caller to report/aggregate."""
    key: RunKey = (run, slot_id, model_alias)
    gate = _cell_gates.get(key)
    if gate is not None:
        if auto:
            gate.auto = True
            gate.proceed(auto=True)  # release a current pause if any
            _set_stepped(key, False)  # "run rest" exits step mode
            return "stepped"
        if until is not None:
            if gate.pending and gate.pending.get("template") == until:
                # Already paused right before a call of the target step.
                if until_before:
                    return "stepped"  # that IS the "before X" breakpoint — no-op
                # "after X": run exactly this call, then pause at the next — a
                # single step from here. (Re-seeking would skip past it to the
                # NEXT occurrence of X.)
                gate.proceed()
                return "stepped"
            gate.until_step = until
            gate.until_before = until_before
            gate.proceed()  # release a current pause to fast-forward (no-op if mid-call)
            return "seeking"
        if gate.proceed():
            return "stepped"  # was paused → released one call
        gate.budget += 1  # mid-call → queue the step for its next gate
        return "queued"
    # No live gate. (A non-stepped task could still be running; guard.)
    task = _tasks.get(key)
    if task is not None and not task.done():
        return "in_flight"
    slot_log = _slot_logs.get(key)
    if slot_log is None:
        return "missing"
    if any(e.get("kind") == "run.done" for e in slot_log.state["events"]):
        return "done"
    # Over the spend cap — can't relaunch to advance it (a capped cell has no
    # live gate to release either; the cap enforcement cancelled its task). The
    # override is the only way forward. "step all" folds this into by_result.
    if _cap_reached(slot_log.state["events"]):
        return "capped"
    # No live task and not done → idle / paused / error / crashed: all runnable.
    if auto:
        _set_stepped(key, False)  # finish normally, ungated
    else:
        _set_stepped(key, True)
        _gate_intents[key] = (
            {"until": until, "until_before": until_before} if until is not None else {"budget": 1}
        )
    await _start_cell(run, slot_id, model_alias)
    return "launched"


async def _advance_branch(branch_id: str, *, auto: bool, until: str | None = None, until_before: bool = False, model: str | None = None) -> str:
    """Advance a simulation branch — the branch mirror of `_advance_cell`, so
    "step sims" behaves like "step all": it queues rather than erroring when a
    branch isn't sitting at a live gate.
      * live gate: release a current pause, else QUEUE a credit (budget++) the
        branch spends at its next gate — so a branch mid-call isn't skipped.
      * `auto` runs it to completion; `until` fast-forwards to the next call of
        that template, pausing AFTER it or (when `until_before`) BEFORE it.
      * `model` (an OpenRouter id) re-aims the next gated call at a chosen LLM —
        sticky on the gate until changed, so the same scene context can be
        tested against different models.
      * paused/error with no task → relaunch carrying the intent (overrides live
        on the Branch + its manifest, so this works after a restart too).
    Returns a status string for the caller to report/aggregate."""
    br = _branches.get(branch_id)
    if br is None:
        return "missing"
    blog = br.log
    gate = br.gate
    task = br.task
    if gate is not None and task is not None and not task.done():
        if model is not None:
            gate.model_override = model  # applies from the next gated call onward
        if auto:
            gate.auto = True
            gate.proceed(auto=True)
            return "stepped"
        if until is not None:
            if gate.pending and gate.pending.get("template") == until:
                # Already paused right before a call of the target step.
                if until_before:
                    return "stepped"  # that IS the "before X" breakpoint — no-op
                # "after X": run exactly this call, then pause at the next
                # (re-seeking would skip to the NEXT occurrence of X).
                gate.proceed()
                return "stepped"
            gate.until_step = until
            gate.until_before = until_before
            gate.proceed()  # release a current pause to fast-forward (no-op if mid-call)
            return "seeking"
        if gate.proceed():
            return "stepped"
        gate.budget += 1  # mid-call → queue the step for its next gate
        return "queued"
    # No live gate (a branch always gates while live, so the task has ended).
    if any(e.get("kind") == "run.done" for e in blog.state["events"]):
        return "done"
    # Not done and no live task → paused / error / crashed: all relaunchable.
    events = blog.state["events"]
    if events and events[-1].get("kind") in ("run.error", "run.paused"):
        blog.truncate_events_to(len(events) - 1)
    if auto:
        intent: dict[str, object] = {"auto": True}
    elif until is not None:
        intent = {"until": until, "until_before": until_before}
    else:
        intent = {"budget": 1}
    if model is not None:
        intent["model"] = model  # seed the relaunched gate's override
    br.gate_intent = intent
    br.task = asyncio.create_task(_run_branch(branch_id))
    return "launched"


def _hardlink_tree(src: Path, dst: Path, ids: set[str] | None = None) -> None:
    """Mirror the flat `objects/` dir (`<id>.glb`, `<id>.raw.glb`, `<id>.png`)
    from `src` into `dst` via hardlinks — instant and zero extra disk, so the
    branch's `_spawn_meshes` sees committed prefix meshes as already present
    (`path.exists()`) and skips re-billing them. When `ids` is given, ONLY files
    for those node ids are linked (the branch passes its PREFIX's committed mesh
    ids) — post-deviation objects are deliberately absent so the pipeline
    regenerates them fresh and the branch shares nothing with the source past
    the fork. Falls back to a copy where hardlinks aren't supported."""
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if not p.is_file():
            continue
        # File names are `<id>.glb` / `<id>.raw.glb` / `<id>.png`; the id is the
        # part before the first dot (node ids carry no dots).
        if ids is not None and p.name.split(".", 1)[0] not in ids:
            continue
        target = dst / p.name
        if target.exists():
            continue
        try:
            os.link(p, target)
        except OSError:
            with contextlib.suppress(OSError):
                shutil.copy2(p, target)


def _branch_awaiting(br: Branch, step: str, node: str | None) -> bool:
    """True when the branch is paused at its gate right before a (step[, node])
    call — i.e. a prompt-test seed for that call MISSED the cache and the real
    re-run hasn't happened yet, so the seed is stale, not the branch's output."""
    pending = br.gate.pending if br.gate is not None else None
    if not pending or pending.get("template") != step:
        return False
    return node is None or pending.get("node") == node


def cache_hash_for_seed(event: dict[str, object], seed: BranchSeed) -> str:
    """Cache key for a seeded prompt-test result: identical inputs to what the
    branch's re-reached call will hash, so the pipeline cache-hits the vetted
    output instead of re-rolling the edited step."""
    return cache.hash_llm_call(
        model=str(event.get("model") or ""),
        system=seed.system,
        user=seed.user,
        schema_name=str(event.get("schema") or ""),
    )


async def _fork_branch(
    run: str,
    slot_id: str,
    model_alias: str,
    *,
    event_index: int,
    step: str | None,
    overrides: dict[str, dict[str, str]],
    seed: BranchSeed | None = None,
    parent: str | None = None,
    model_pin: str | None = None,
    gate_budget: int = 0,
    version: str | None = None,
    atomic_locks: list[str] | None = None,
) -> Branch:
    """Fork a new simulation branch (allocating a fresh id) and start it gated.
    Many forks of one cell coexist — nothing prior is discarded.

    `parent=None` forks the ORIGIN cell at `event_index` (a re-renderable
    cache.llm call of template `step`): keep the prefix, hardlink the cell's
    committed prefix meshes, and re-run from there under the run snapshot +
    `overrides`. With no gate budget the branch replays the prefix and pauses at
    the first cache-miss — exactly the forked step — awaiting a manual step (a
    `seed` pre-commits a vetted prompt-test result for that step, so it
    cache-hits past it and pauses at the next one).

    `parent=<bid>` forks that BRANCH at its current frontier — the parallel-LLM
    child: copy the parent's whole log + committed meshes, `model_pin` the
    frontier call to a chosen model, and `gate_budget=1` so it runs exactly that
    one step then parks for the user to view + keep."""
    if parent is not None:
        psrc = _require_branch(parent)
        src_log = psrc.log
        src_objects = src_log.events_path.parent / "objects"
        event: dict[str, object] | None = None  # children carry no seed
    else:
        src_log, event = _require_step_event(run, slot_id, model_alias, event_index, step or "")
        cell_dir = src_log.events_path.parent
        src_objects = cell_dir / OBJECTS_SUBDIR
        if not src_objects.is_dir():
            src_objects = next(
                (cell_dir / d for d in ("objects-optimized", "objects") if (cell_dir / d).is_dir()),
                cell_dir / "objects",
            )
    src_events = list(src_log.state["events"])
    event_index = max(0, min(event_index, len(src_events)))

    bid = _new_branch_id(slot_id, model_alias)
    bdir = _branch_dir(run, bid)
    shutil.rmtree(bdir, ignore_errors=True)
    bdir.mkdir(parents=True, exist_ok=True)

    # Hardlink ONLY the prefix's committed meshes (objects committed BEFORE the
    # fork) so the replayed prefix renders without regenerating. Nothing past
    # the fork is linked: those objects have no file here, so the pipeline makes
    # each one FRESH (library re-match + re-place, or a fresh Trellis run),
    # keeping the branch independent of the source past the branch-off point.
    prefix_ids = _committed_mesh_ids(src_events[:event_index])
    await asyncio.to_thread(_hardlink_tree, src_objects, bdir / "objects", prefix_ids)

    # Prefix = source events BEFORE the fork. Dropping that call (and everything
    # after) means `committed.*` replays the prefix and the pipeline re-reaches
    # it with the (snapshot + overrides) templates.
    branch_events = [dict(e) for e in src_events[:event_index]]
    # A CHILD fork copies the PARENT branch's log; repoint the parent's object
    # URLs at the child's own dir (the parent's committed meshes are hardlinked
    # into it just above), so the child — and any sim KEPT from it — stays
    # self-referential after the parent is discarded. Inductively, a child only
    # ever carries its-own-dir + source-cell URLs (source points at the live
    # cell and needs no rewrite), so a later commit/save needs just one rewrite.
    if parent is not None:
        seg = f"/{BRANCHES_SUBDIR}/{parent}/objects/"
        dest_seg = f"/{BRANCHES_SUBDIR}/{bid}/objects/"
        for e in branch_events:
            url = e.get("url")
            if isinstance(url, str) and seg in url:
                e["url"] = url.replace(seg, dest_seg)
    bevents = bdir / "events.jsonl"
    with bevents.open("w", encoding="utf-8") as f:
        for e in branch_events:
            f.write(json.dumps(e) + "\n")

    blog = SlotLog(_branch_run_id(run, bid), bevents)
    blog.hydrate_from_disk()
    if blog.state.get("prompt") is None or blog.state.get("model") is None:
        shutil.rmtree(bdir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="source has no run.start to branch from")
    if seed is not None and event is not None:
        blog.log(
            "cache.llm",
            key=cache_hash_for_seed(event, seed),
            node=event.get("node"),
            step=event.get("step"),
            template=step,
            model=event.get("model"),
            schema=event.get("schema"),
            system=seed.system,
            user=seed.user,
            variables=event.get("variables"),
            output=seed.output,
            reasoning=seed.reasoning,
            tokens_in=seed.tokens_in,
            tokens_out=seed.tokens_out,
            seeded=True,
        )
    br = Branch(
        id=bid, run=run, slot=slot_id, model=model_alias,
        parent=parent, fork_index=event_index,
        overrides={s: dict(r) for s, r in overrides.items()},
        log=blog, model_pin=model_pin, version=version,
        gate_intent={"budget": gate_budget} if gate_budget else {},
        atomic_locks=list(atomic_locks or []),
    )
    _branches[bid] = br
    _write_branch_manifest(br)
    br.task = asyncio.create_task(_run_branch(bid))
    return br


def _children_of(branch_id: str) -> list[str]:
    """Live child (parallel-LLM preview) branch ids forked from `branch_id`."""
    return [b.id for b in _branches.values() if b.parent == branch_id]


async def _cancel_branch_task(br: Branch) -> None:
    """Cancel the branch's task + in-flight meshes + step gate, but KEEP its
    SlotLog, overrides, manifest, and dir — pause/resume relaunch on the same
    branch (a relaunch binds a fresh gate, back in stepping mode)."""
    task = br.task
    br.task = None
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    br.gate = None
    br.gate_intent = {}
    generation.cancel_pending(br.log.slot_id)


async def _discard_branch(branch_id: str) -> None:
    """Full break-out: discard this branch AND every child forked from it, then
    delete its directory. Idempotent."""
    br = _branches.get(branch_id)
    if br is None:
        return
    for child in _children_of(branch_id):
        await _discard_branch(child)
    await _cancel_branch_task(br)
    _branches.pop(branch_id, None)
    br.log.close()
    shutil.rmtree(_branch_dir(br.run, branch_id), ignore_errors=True)


async def _discard_branches_of_cell(run: str, slot_id: str, model_alias: str) -> None:
    """Discard every branch (and child) whose ORIGIN is this cell — used when
    the source cell is reset/rewound/promoted-onto, which invalidates its forks."""
    for bid in [
        b.id for b in _branches.values()
        if b.run == run and b.slot == slot_id and b.model == model_alias
    ]:
        await _discard_branch(bid)


async def _run_branch(branch_id: str) -> None:
    """Drive one branch — top-level sim OR parallel-LLM child alike. A mirror of
    `_run` bound to the branch's SlotLog + branch run_id + the run snapshot WITH
    the lab's overrides: the prefix replays via `committed.*`, a seeded step
    cache-hits, and the frontier re-runs under the edited templates. The step
    gate (bound in THIS task's context, so only this branch pauses) advances it
    one call at a time; the branch's `model_pin` (a child's chosen model) or a
    per-launch `model` intent re-aims the frontier call."""
    br = _branches[branch_id]
    blog = br.log
    rlog.bind(blog)
    committed.bind_forced_atomic(br.atomic_locks)
    intent = br.gate_intent or {}
    br.gate_intent = {}
    pin = intent.get("model") or br.model_pin
    gate = CellGate(blog, budget=int(intent.get("budget", 0) or 0), model_override=str(pin) if pin else None)
    gate.auto = bool(intent.get("auto", False))
    _b_until = intent.get("until")
    gate.until_step = str(_b_until) if _b_until else None
    gate.until_before = bool(intent.get("until_before", False))
    br.gate = gate
    llm.set_step_gate(gate.wait)
    prompt = blog.state["prompt"]
    model = blog.state["model"]
    brun_id = blog.slot_id  # composite branch run_id (<run>/_branches/<bid>)
    try:
        # A branch runs under its `version` (the prompt-set A/B lineage) when set,
        # else the run's own snapshot — with the lab's overrides layered on either.
        base = (prompt_store.load_version(br.version) if br.version
                else prompt_store.load_run_prompts(_run_dir(br.run)))
        prompt_store.bind(base.with_overrides(br.overrides))
        await divider.run(run_id=brun_id, prompt=prompt, model=model, runs_dir=RUNS_DIR)
    except asyncio.CancelledError:
        generation.cancel_pending(brun_id)
        raise
    except Exception as e:
        generation.cancel_pending(brun_id)
        blog.log("run.error", message=_llm_error_message(e))
        return
    finally:
        # The gate lives only for the task's duration — drop it so a finished
        # branch reports no stale `pending`/`current`. (A live pause is still
        # awaiting inside divider.run, so this runs only once the task ends.)
        br.gate = None
    await generation.await_pending(brun_id)
    blog.finish_run()


async def _start_cell(run: str, slot_id: str, model_alias: str) -> None:
    """Cancel any in-flight task for this cell, then (re)start its pipeline.
    A fresh cell emits run.start; a paused/errored cell drops its terminal
    sentinel and resumes. Callers own any other precondition guards (e.g.
    blocking a completed run)."""
    slot = _require_slot(slot_id)
    slot_log = _require_slot_log(run, slot_id, model_alias)
    # The run's prompt snapshot is the only prompt source — without one
    # (legacy run) nothing can launch.
    _require_run_prompts(run)
    await _cancel_task(run, slot_id, model_alias)
    events = slot_log.state["events"]
    model_id = MODELS[model_alias]
    if not events:
        slot_log.start_run(slot.prompt, model_id)
    else:
        if events[-1].get("kind") in ("run.error", "run.paused"):
            slot_log.truncate_events_to(len(events) - 1)
        slot_log.state["model"] = model_id
    _tasks[(run, slot_id, model_alias)] = asyncio.create_task(_run(run, slot_id, model_alias))


_MESH_BUNDLE_MAGIC = b"SMB1"


def _committed_mesh_ids(events: list[dict[str, object]]) -> set[str]:
    """Node ids with a committed `model` event — the meshes that are actually
    part of the scene. A branch hardlinks the WHOLE source `objects/` dir up
    front (so the replayed prefix doesn't re-bill Trellis), so its bundle must
    exclude files for nodes the re-run dropped (no `model` event in the branch
    log) — otherwise those obsoleted source meshes render as ghosts the branch's
    bboxes/scene-projection don't include."""
    return {
        e["id"]  # type: ignore[misc]
        for e in events
        if e.get("kind") == "model" and isinstance(e.get("id"), str)
    }


async def _mesh_bundle(objects_dir: Path, ids: set[str] | None = None) -> AsyncIterator[bytes]:
    """Stream every finished GLB under `objects_dir` as one length-prefixed
    binary bundle, so a client can fetch an entire scene in a single request
    instead of one HTTP round-trip per mesh. Framing (little-endian):

        b"SMB1"
        repeat: <uint32 id_len><id utf-8><uint32 glb_len><glb bytes>

    The id is the file stem, which matches the `model` event's `id`. Files are
    read off the event loop via a thread so a large scene doesn't stall it. When
    `ids` is given, only those stems are streamed — the branch passes the set of
    its committed `model` ids so stale hardlinked prefix meshes aren't served.
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
        if ids is not None and node_id not in ids:
            continue  # not part of this scene (e.g. a branch's obsoleted object)
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
            if event["kind"] in {"run.done", "run.error", "run.paused", "run.cap_reached"}:
                return
    finally:
        slot_log.unsubscribe(q)


# How long the regen worker blocks on in-flight builds before looping back to
# re-drain its queue. Bounds how long a regen enqueued mid-batch waits before it
# starts (it shouldn't wait for a prior build to finish); the worker only spins
# this while it has builds in flight, and exits outright once idle.
_REGEN_POLL_INTERVAL_S = 0.25


async def _regen_worker(run: str, slot_id: str, model_alias: str) -> None:
    """Drain the cell's generated-build regeneration queue CONCURRENTLY. The worker
    owns the generated-events log for the whole drain; `SlotLog.log()` is
    synchronous, so concurrent items append through it atomically (unique indices,
    no torn lines). Items run in parallel ACROSS prefab groups but are serialized
    WITHIN a group by a per-canonical lock — so two regens that resolve to the same
    canonical (a stray double-enqueue, or a reuse plus its canonical) can't race the
    same files, and the second re-resolves under the lock to see the first's
    promotion. Spawns are unbounded; the heavy work is throttled by the process-
    global Banana / Trellis / mesh-IO / optimize semaphores. Exits when the queue is
    empty and nothing is in flight; the next enqueue restarts it. Cancellation
    (reset / stop / teardown) cancels in-flight builds and clears any still-queued
    rows from the shared Trellis queue panel."""
    key: GenKey = (run, slot_id, model_alias)
    queue = _regen_queues.get(key)
    if queue is None:
        return
    lib_log = _slot_logs.get((run, slot_id, model_alias))
    run_id = _run_id(run, slot_id, model_alias)
    gen_slot_id = _gen_slot_id(run, slot_id, model_alias)
    raw_subdir = generation.GENERATED_RAW_SUBDIR
    gen_log = SlotLog(gen_slot_id, generation.generated_events_path(RUNS_DIR, run_id))
    gen_log.hydrate_from_disk()
    rlog.bind(gen_log)
    llm.set_model(MODELS[model_alias])
    prompt_store.bind(prompt_store.load_run_prompts(_run_dir(run)))
    canon_locks: dict[str, asyncio.Lock] = {}

    def _carry_symmetry(src_id: str, dst_id: str) -> None:
        """Copy src's latest applied symmetry (plane + kept half) onto dst, so a
        later re-derive of dst from its raw mirrors identically — a promoted /
        handed-off node logs no symmetry of its own, so without this it would
        resolve (and re-derive) as un-mirrored."""
        for e in reversed(gen_log.state["events"]):
            if e.get("kind") == "symmetry.applied" and e.get("id") == src_id:
                cp = e.get("cut_plane")
                if cp in ("none", "xy", "xz"):
                    extra = (
                        {"keep_positive": e["keep_positive"]}
                        if isinstance(e.get("keep_positive"), bool) else {}
                    )
                    gen_log.log("symmetry.applied", id=dst_id, cut_plane=cp, **extra)
                break

    async def _do_unlink(node_id: str, node: Node) -> None:
        """Pull `node_id` out of its prefab group into a STANDALONE asset with its
        OWN raw mesh, WITHOUT rebuilding it — so it stops sharing and the user can
        then regenerate it alone. A reuse clones its canonical's raw (inheriting its
        geometry + symmetry) and is promoted to its own canonical; its served mesh,
        already derived from that same raw, stays valid. A canonical WITH reuses
        hands the group off to one of those reuses — that reuse inherits `node_id`'s
        raw (via `clone_canonical_raw`) + symmetry and the rest are repointed to it,
        while `node_id` stays standalone with its own raw. A lone canonical is
        already independent."""
        assert lib_log is not None
        canonical_id, reuse_ids = prefabs.resolve_group(gen_log.state["events"], node_id)
        if canonical_id != node_id:
            await generation.clone_canonical_raw(
                runs_dir=RUNS_DIR, run_id=run_id, source_id=canonical_id, dest_id=node_id,
            )
            _carry_symmetry(canonical_id, node_id)
            gen_log.log("prefab.match", id=node_id, reuse_id="", description=node.prompt)
            return
        if not reuse_ids:
            return
        new_canon = reuse_ids[0]
        await generation.clone_canonical_raw(
            runs_dir=RUNS_DIR, run_id=run_id, source_id=node_id, dest_id=new_canon,
        )
        _carry_symmetry(node_id, new_canon)
        new_node = _reconstruct_node(lib_log, new_canon)
        gen_log.log(
            "prefab.match", id=new_canon, reuse_id="",
            description=new_node.prompt if new_node else "",
        )
        for rid in reuse_ids[1:]:
            rnode = _reconstruct_node(lib_log, rid)
            gen_log.log(
                "prefab.match", id=rid, reuse_id=new_canon,
                description=rnode.prompt if rnode else "",
            )

    async def _do_link(node_id: str, target_id: str, *, link_group: bool = False) -> None:
        """Link `node_id` INTO the prefab group of `target_id` (any group member;
        resolved to that group's canonical), re-deriving its mesh from the
        canonical's raw. If `node_id` was itself a canonical with reuses, those
        reuses come along so the flat prefab star is preserved. With
        `link_group=True` the entire SOURCE group (canonical + every reuse) moves
        into the destination — even when `node_id` is a reuse, its old canonical
        and siblings come along too."""
        assert lib_log is not None
        dest_canonical, _ = prefabs.resolve_group(gen_log.state["events"], target_id)
        async with canon_locks.setdefault(dest_canonical, asyncio.Lock()):
            if _reconstruct_node(lib_log, dest_canonical) is None:
                gen_log.log(
                    "mesh.error", id=node_id,
                    message=f"link: no bbox event for target canonical {dest_canonical}",
                )
                return
            own_canonical, own_reuses = prefabs.resolve_group(gen_log.state["events"], node_id)
            if own_canonical == dest_canonical:
                gen_log.log("prefab.link_noop", id=node_id, reuse_id=dest_canonical)
                return
            if link_group:
                mover_ids = [own_canonical, *own_reuses]
            elif own_canonical == node_id:
                mover_ids = [node_id, *own_reuses]
            else:
                mover_ids = [node_id]
            movers = [
                n for mid in mover_ids if (n := _reconstruct_node(lib_log, mid)) is not None
            ]
            for m in movers:
                gen_log.log(
                    "prefab.match", id=m.id, reuse_id=dest_canonical, description=m.prompt,
                )
            await generation.propagate_reuses(
                canonical_id=dest_canonical, reuses=movers,
                runs_dir=RUNS_DIR, run_id=run_id,
            )

    async def _process(job: RegenJob) -> None:
        node_id, op = job.node_id, job.op
        if lib_log is None:
            return
        try:
            if op == "link":
                await _do_link(node_id, job.link_to or "", link_group=job.link_group)
                return
            # Pick the per-canonical lock by this node's current group, then do the
            # real resolution + build UNDER the lock so same-group items serialize
            # and a later one observes an earlier item's promotion.
            canonical0, _ = prefabs.resolve_group(gen_log.state["events"], node_id)
            async with canon_locks.setdefault(canonical0, asyncio.Lock()):
                target = _reconstruct_node(lib_log, node_id)
                if target is None:
                    gen_log.log("mesh.error", id=node_id, message=f"{op}: no bbox event for node")
                    return
                if op == "unlink":
                    # Split this object into a standalone asset with its own raw —
                    # no rebuild; the user regenerates it separately.
                    await _do_unlink(node_id, target)
                    return
                propagate = job.propagate
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
                        node=build_node, runs_dir=RUNS_DIR, run_id=run_id,
                    )
                elif op == "symmetrize":
                    # Mirror the existing mesh across the caller-supplied plane (no
                    # AI). The canonical is mirrored here; propagate re-derives its
                    # reuses below, which read the canonical's symmetry.applied
                    # (plane + kept half) and mirror identically.
                    cut_plane, keep_positive = job.sym  # type: ignore[misc]
                    await generation.symmetrize_one(
                        node=build_node, cut_plane=cut_plane, keep_positive=keep_positive,  # type: ignore[arg-type]
                        runs_dir=RUNS_DIR, run_id=run_id,
                    )
                elif op == "reorient":
                    # Re-front the canonical's raw mesh (rotate which face is +Z);
                    # propagate re-derives its reuses below from the re-fronted raw,
                    # so the whole prefab group shares the new front view.
                    rax, rdeg = job.reorient  # type: ignore[misc]
                    await generation.reorient_one(
                        node=build_node, axis=rax, degrees=rdeg,  # type: ignore[arg-type]
                        runs_dir=RUNS_DIR, run_id=run_id,
                    )
                elif op == "reset":
                    # Rebuild the canonical's served mesh from its pristine raw
                    # (drop forced glass, keep symmetry); propagate re-derives its
                    # reuses below from that same clean raw, so the whole group
                    # reverts together.
                    await generation.reset_from_raw_one(
                        node=build_node, runs_dir=RUNS_DIR, run_id=run_id,
                    )
                elif op == "glassify":
                    # Force the glass-transparency transform onto the WHOLE prefab
                    # group (no AI). Glass is a per-mesh texture edit, not derivable
                    # from the shared raw, so apply it to the canonical AND every
                    # reuse directly here and SKIP the raw-replay propagate below.
                    await generation.glassify_group(
                        nodes=[build_node, *reuses], runs_dir=RUNS_DIR, run_id=run_id,
                    )
                else:
                    if job.reuse_image:
                        # From-image rebuild: ensure the node we're building has its
                        # raw-dir reference image. If that copy is missing (e.g. the
                        # raw image was removed but the optimized twin survived, or a
                        # prefab sibling still has it), restore it so we reuse the
                        # image instead of re-generating it via the API.
                        generation.recover_group_image(
                            RUNS_DIR, run_id, build_node.id, [canonical_id, *reuse_ids],
                        )
                    # Regenerate-noun-phrase: re-distill from the build node's
                    # AUTHORED seed (the library log's bbox prompt), not its current
                    # distilled phrase — so the new phrase derives from the original
                    # description even when the object has no noun phrase yet. Also
                    # reconstruct the full scene tree + the node's owning zone, so the
                    # re-distillation runs with the SAME scene context (root header,
                    # zone, sibling objects) the original image_prompt step saw rather
                    # than an empty one.
                    seed_prompt: str | None = None
                    scene_zone: Node | None = None
                    scene_nodes: list[Node] | None = None
                    if job.regen_noun_phrase and lib_log is not None:
                        seed_prompt = _node_seed_prompt(lib_log, build_node.id)
                        scene_nodes = _scene_nodes_full(lib_log, image_log=gen_log)
                        target = next((n for n in scene_nodes if n.id == build_node.id), None)
                        zone_id = target.parent_region if target is not None else None
                        scene_zone = (
                            next((n for n in scene_nodes if n.id == zone_id), None)
                            if zone_id else None
                        )
                    await generation.regenerate_one(
                        node=build_node, runs_dir=RUNS_DIR, run_id=run_id,
                        subdir=raw_subdir, optimize=True, backend=job.backend, generated=True,
                        reuse_image=job.reuse_image,
                        regen_noun_phrase=job.regen_noun_phrase, seed_prompt=seed_prompt,
                        scene_zone=scene_zone, scene_nodes=scene_nodes,
                    )
                if propagate and op != "glassify":
                    # glassify already transformed every member itself (above) —
                    # its texture edit isn't re-derivable from the raw that
                    # propagate_reuses replays, so it must not run here.
                    await generation.propagate_reuses(
                        canonical_id=canonical_id, reuses=reuses,
                        runs_dir=RUNS_DIR, run_id=run_id,
                    )
                elif canonical_id != node_id and op not in ("unsymmetrize", "symmetrize", "reorient", "glassify", "reset"):
                    # A reuse regenerated on its own now owns a fresh mesh + raw —
                    # record it as canonical so a later propagate of its old source
                    # can't clobber it. (Symmetry / reorient / glassify / reset ops
                    # write no new raw for a reuse — so they never promote.)
                    gen_log.log(
                        "prefab.match", id=node_id, reuse_id="", description=build_node.prompt,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            gen_log.log("mesh.error", id=node_id, message=f"{type(e).__name__}: {e}")
        finally:
            # Drop the pre-enqueue queue-panel entry (if any). For a regenerate
            # that reached generate_mesh, the mesh-jobs lifecycle already removed
            # it; this catches early failures (bad node, missing image) and
            # non-regenerate ops that were never marked. The pop is a no-op when
            # the entry is already gone.
            threed.unmark_queued(gen_slot_id, node_id)

    inflight: set[asyncio.Task[None]] = set()
    try:
        while True:
            # Spawn every currently-queued item; the semaphores bound the real work.
            while True:
                try:
                    job = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                inflight.add(asyncio.create_task(_process(job)))
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
                pending = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            threed.unmark_queued(gen_slot_id, pending.node_id)
        gen_log.close()


async def _run(run: str, slot_id: str, model_alias: str) -> None:
    key: RunKey = (run, slot_id, model_alias)
    slot_log = _slot_logs[key]
    rlog.bind(slot_log)
    # Stepped mode: gate this cell so it pauses before every frontier LLM
    # call (the per-step prompt-iteration workflow). The next-gate intent
    # (set by "step" / "run rest" / rerun-step) tunes the first launch.
    if key in _stepped_cells:
        intent = _gate_intents.pop(key, {})
        gate = CellGate(slot_log, budget=int(intent.get("budget", 0) or 0))
        gate.auto = bool(intent.get("auto", False))
        _until = intent.get("until")
        gate.until_step = str(_until) if _until else None
        gate.until_before = bool(intent.get("until_before", False))
        _cell_gates[key] = gate
        llm.set_step_gate(gate.wait)
    prompt = slot_log.state["prompt"]
    model = slot_log.state["model"]
    run_id = _run_id(run, slot_id, model_alias)
    try:
        # Bind the run's prompt snapshot inside the try so a broken/missing
        # snapshot surfaces as a clean run.error instead of a dead task.
        prompt_store.bind(prompt_store.load_run_prompts(_run_dir(run)))
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
        # Log the full provider detail (metadata/body), not the SDK's opaque
        # top-level "Provider returned error", so the run.error event is useful.
        slot_log.log("run.error", message=_llm_error_message(e))
        return
    finally:
        # A gate only lives for the duration of its task — drop it so a paused
        # or finished cell reports no `pending`. (A live pause is still
        # awaiting inside divider.run, so this runs only once the task ends.)
        _cell_gates.pop(key, None)
    # Pipeline tree is fully resolved; meshes may still be in flight.
    # Hold the run open until they all land so `run.done` truly means done.
    await generation.await_pending(run_id)
    slot_log.finish_run()
