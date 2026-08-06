"""Phase 2 — populate a zone with objects via Trellis 2.

Three scenarios:
  * "anchor"         — atomic leaf zone; generate defining objects, then
                       iterate (ask "is another object needed?") until
                       the LLM says done.
  * "encapsulating"  — the zone's physical shell / floor / ground.
                       Runs on every placed child zone before recursion;
                       emits walls+floor+ceiling for architectural zones
                       and a single ground mesh for atomic terrain zones.
                       One shot, no loop.
  * "negative-space" — one pass over the scene root after the whole
                       divider tree is built. Enumerates the ambient /
                       drifting content that fills the interstitial
                       space between zones. No completion loop.

Per scenario: decompose objects (LLM, a single call) -> split any object the
mesh step can't build as one coherent mesh into its constituent pieces (the
optional `object_decomp` pass, `_split_objects`; a pass-through on versions
without it) -> resolve every object's bbox in a single batch LLM call (trusted,
no retry) -> spawn background Trellis 2 jobs that fan out via SSE events as each
mesh lands.
There is NO validate-and-retry step: a decomposition whose ids collide with
already-placed nodes is accepted as-is (the retry could never resolve a
genuine boundary/anchor overlap and just re-billed the call), and the
run-wide dedup in `_resolve_and_generate` silently drops the colliding specs.

The anchor-loop's "are more objects needed?" step proposes a LIST of objects
per round (same single-call, dedup-not-retry handling). Bounding-box
resolution is a single batch call.

Prompt text comes from the run's prompt snapshot (`prompt_store.current()`);
this module only decides which step fires when and with which scene state.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

import trimesh

from app.core import prompt_store, scene_context, schemas
from app.core.slots import MODELS
from app.core.types import BoundingBox, Node, ProxyShape
from app.pipeline import committed, context_cull
from app.services import hunyuan, hunyuan_tencent, llm, mesh_jobs, nano_banana, prefabs, symmetry, threed
from app.utils import glass, glb_place, logging
from app.utils.geometry import export_glb, rescale_mesh_to_bbox, rotate_mesh
from app.utils.topology import uniquify_ids

_USE_ASSET_LIBRARY = os.environ.get("USE_ASSET_LIBRARY", "true").lower() == "true"

# When true, the `image_prompt` noun-phrase step (see `_distill_subject`) runs on
# gemini-flash-lite instead of the run's configured model — a cheap distill, like
# library-match / symmetry, kept off the benchmark model surface.
_DOWNGRADE_NOUN_PHRASE = os.environ.get("DOWNGRADE_NOUN_PHRASE", "true").lower() == "true"

# Mesh backends a build can route to. Keys are the values the API accepts; each
# exposes an identical `generate_mesh(...)`, so picking one is a dict lookup. The
# divider, the generate gate, and library retries all use the default; only
# per-asset regeneration overrides it (the regenerate buttons). `trellis` and
# `hunyuan` (Omni) ride the shared Modal spawn-and-poll core (app.services.mesh_jobs,
# ~100 concurrent); `hunyuan-tencent` calls Tencent Cloud's Hunyuan 3D 3.1 Rapid API
# directly and self-serializes to one task at a time (its own account-level limit).
DEFAULT_MESH_BACKEND = "trellis"
MESH_BACKENDS: dict[str, Callable[..., Awaitable[Path]]] = {
    "trellis": threed.generate_mesh,
    "hunyuan": hunyuan.generate_mesh,
    "hunyuan-tencent": hunyuan_tencent.generate_mesh,
}


def _scene_backend() -> str:
    """The mesh backend the from-scratch *scene* generate gate (`generate_assets`)
    routes every object to. Defaults to Trellis; set
    `GENERATE_SCENE_BACKEND=hunyuan-tencent` to build whole scenes on Tencent's
    Hunyuan 3D 3.1 Rapid — which paces itself to one job at a time via that
    backend's own gate, so the gate's fan-out still serializes onto Tencent. Read
    at submit time so flipping the env var takes effect on the next generate
    without a restart; an unknown value falls back to the default with an audit
    log."""
    backend = os.environ.get("GENERATE_SCENE_BACKEND", DEFAULT_MESH_BACKEND).strip()
    if backend not in MESH_BACKENDS:
        logging.log(
            "generate.backend_invalid",
            backend=backend,
            fallback=DEFAULT_MESH_BACKEND,
            valid=sorted(MESH_BACKENDS),
        )
        return DEFAULT_MESH_BACKEND
    return backend

# Guards the trimesh load -> rescale -> export block. API calls and GLB
# downloads stay fully parallel across slots; only the RAM-heavy mesh
# decode is serialized so concurrent slots don't stack Pillow-decoded
# texture buffers and trip the OOM killer.
_MESH_IO = asyncio.Semaphore(1)

# The generate gate fans out EVERY asset at once; the queue is intentionally
# uncapped. Live concurrency is bounded only by the downstream per-stage gates
# each pipeline passes through: Nano-Banana image gen
# (nano_banana.GENERATE_CONCURRENCY), the Trellis in-flight cap
# (threed.GENERATE_CONCURRENCY — the "100 live requests"), the serialized mesh
# decode (`_MESH_IO`), and the optimizer pool (`_OPTIMIZE_FANOUT`). A process-
# global fanout cap here previously double-capped the gate at the Trellis limit,
# so a scene could never queue past 100 — overflow blocked before it registered
# in threed's queue snapshot. Trellis submits are still spaced by `_pace_submit`.

# The asset-library optimizer (tools/optimize-assets/optimize.mjs) — the same
# weld/decimate/prune + KTX2 + Meshopt pass every library asset goes through.
# Raw Trellis output is ~500k tris / ~10-20MB each; this brings every generated
# asset down to the library's ~15-20k-tri, GPU-compressed footprint so the
# generated view streams and renders like the library one. CPU/RAM-heavy (sharp
# decode + Basis encode), so cap concurrent node subprocesses well below the
# number of assets that may be in flight.
_OPTIMIZE_DIR = Path(__file__).resolve().parents[2] / "tools" / "optimize-assets"
_OPTIMIZE_SCRIPT = _OPTIMIZE_DIR / "optimize.mjs"
_NODE_BIN = os.environ.get("STARSHOT_NODE_BIN", "node")
_OPTIMIZE_FANOUT = asyncio.Semaphore(4)

# Cap on concurrent library-asset match calls (flash-lite) when realizing a whole
# scene's assets in one batch. call_llm has no client-side rate limit, so this
# bounds the fan-out under OpenRouter's ceiling while still matching in parallel.
LIBRARY_MATCH_CONCURRENCY = 12
_library_match_slot = asyncio.Semaphore(LIBRARY_MATCH_CONCURRENCY)

# Versioned from-scratch generated builds (Nano-Banana + a mesh backend). A cell
# can hold ANY number of independent generated versions of the SAME scene —
# identical layout (ids / bboxes reconstructed from the library build), but freshly
# generated (or re-derived) assets per object — laid out under the cell as:
#   generated/<version>/objects-generated/            raw meshes (intermediate)
#   generated/<version>/objects-generated-optimized/  served twin (KTX2/Meshopt)
#   generated/<version>/events.generated.jsonl        per-version resumable log
# Each version's own log + dirs make it fully isolated: resume / regen / prefab
# lookups read the bound (per-version) log, so a fresh version builds from scratch
# instead of reusing another version's meshes. The asset-library build (objects/ +
# events.jsonl) is NOT versioned and is unaffected.
GENERATED_DIR = "generated"
GENERATED_RAW_SUBDIR = "objects-generated"
GENERATED_OPT_SUBDIR = "objects-generated-optimized"
# Presentation "lite" twin (server/scripts/build_lite_assets.py): near-lossless
# geometry + high-res UASTC, sitting beside the raw/optimized builds.
GENERATED_LITE_SUBDIR = "objects-generated-lite"
# Splat-pipeline sample/render tier (build_lite_assets.py --preset splat):
# optimized-grade decimation + 1024px KTX2/ETC1S base color, other maps stripped.
# The single asset source every splat stage (2/3/5) reads.
GENERATED_SPLAT_SUBDIR = "objects-generated-splat"
GENERATED_EVENTS_NAME = "events.generated.jsonl"


def generated_version_root(runs_dir: Path, run_id: str) -> Path:
    """The parent dir holding every generated version of one cell."""
    return runs_dir / run_id / GENERATED_DIR


def generated_dirs(runs_dir: Path, run_id: str, version: str) -> tuple[Path, Path]:
    """(raw_dir, opt_dir) for one generated version of a cell."""
    base = generated_version_root(runs_dir, run_id) / str(version)
    return base / GENERATED_RAW_SUBDIR, base / GENERATED_OPT_SUBDIR


def generated_events_path(runs_dir: Path, run_id: str, version: str) -> Path:
    """The per-version resumable event log (events.generated.jsonl)."""
    return generated_version_root(runs_dir, run_id) / str(version) / GENERATED_EVENTS_NAME


def list_generated_versions(runs_dir: Path, run_id: str) -> list[str]:
    """Existing generated version ids for a cell, ascending numeric order."""
    root = generated_version_root(runs_dir, run_id)
    if not root.is_dir():
        return []
    versions = [p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(versions, key=int)


def next_generated_version(runs_dir: Path, run_id: str) -> str:
    """The id for a brand-new version: one past the highest existing, else 1."""
    versions = list_generated_versions(runs_dir, run_id)
    return str(int(versions[-1]) + 1) if versions else "1"


def generated_lite_dir(runs_dir: Path, run_id: str, version: str) -> Path:
    """The `objects-generated-lite/` presentation tier of one generated version,
    sitting beside that version's raw + optimized dirs."""
    return generated_version_root(runs_dir, run_id) / str(version) / GENERATED_LITE_SUBDIR


def latest_generated_version(runs_dir: Path, run_id: str) -> str | None:
    """The highest existing generated version id, or None when the cell has no
    generated build yet."""
    versions = list_generated_versions(runs_dir, run_id)
    return versions[-1] if versions else None


def migrate_legacy_generated(runs_dir: Path, run_id: str) -> None:
    """Fold a pre-versioning generated build (objects-generated*/ +
    events.generated.jsonl sitting directly in the cell dir) into generated/1/,
    so a build made before versioning keeps rendering under the versioned layout
    and stays resumable as version 1 — the backwards-compatibility path for the
    non-versioned system. Idempotent: a no-op once a generated/ dir exists or
    there is nothing legacy to move, and each item is only moved when its
    destination is absent, so a racing second call can't nest it wrongly."""
    root = generated_version_root(runs_dir, run_id)
    if root.exists():
        return
    cell = runs_dir / run_id
    legacy = [
        cell / GENERATED_RAW_SUBDIR,
        cell / GENERATED_OPT_SUBDIR,
        cell / GENERATED_EVENTS_NAME,
    ]
    if not any(p.exists() for p in legacy):
        return
    dst = root / "1"
    dst.mkdir(parents=True, exist_ok=True)
    for p in legacy:
        target = dst / p.name
        if p.exists() and not target.exists():
            shutil.move(str(p), str(target))


def artifacts_complete(
    raw_dir: Path, opt_dir: Path, node_id: str, *, is_reuse: bool,
) -> bool:
    """Whether a generated object's WHOLE on-disk artifact set is present, not
    just the served twin: the optimized GLB, its unoptimized source, and — for a
    canonical — the pristine raw Trellis mesh it (and its reuses) re-derive from.
    A reuse owns no raw (it rescales its canonical's), so it needs only the two
    served twins.

    The resume gate keys on THIS rather than the optimized twin alone, because a
    regen deletes the raw + unoptimized up front: a build that fails midway then
    leaves only the stale optimized GLB, which still renders and so masks the loss
    as a silent "ghost". Treating that torn set as un-built makes a resume rebuild
    it — cheaply re-deriving from the raw when it survives, re-generating when it
    doesn't — instead of skipping it, and lets the status surface it."""
    served = (opt_dir / f"{node_id}.glb").exists()
    unoptimized = (raw_dir / f"{node_id}.glb").exists()
    # A reuse re-derives from its canonical's raw, so it owns none of its own.
    raw = is_reuse or (raw_dir / f"{node_id}.raw.glb").exists()
    return served and unoptimized and raw


async def _optimize_asset(src: Path, dst: Path) -> bool:
    """Run one freshly generated GLB through the library optimizer into `dst`.
    Atomic via a temp file (in src's dir, which isn't served) so a crash can't
    leave a half-written optimized asset that resume would treat as finished.
    On any failure `dst` is left absent, so a re-run retries it — cheaply, since
    the raw mesh and Trellis cache are reused."""
    # Absolute paths: the subprocess runs with cwd=_OPTIMIZE_DIR, and RUNS_DIR
    # may be relative to the server's cwd, so relative paths would resolve wrong.
    src = src.resolve()
    dst = dst.resolve()
    async with _OPTIMIZE_FANOUT:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = src.with_name(f"{src.stem}.opt-tmp.glb")
        try:
            proc = await asyncio.create_subprocess_exec(
                _NODE_BIN, str(_OPTIMIZE_SCRIPT),
                "--file", str(src), "--out-file", str(tmp),
                cwd=str(_OPTIMIZE_DIR),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            logging.log("generate.optimize_error", id=src.stem, message=f"{type(e).__name__}: {e}")
            return False
        try:
            _, stderr = await proc.communicate()
        except asyncio.CancelledError:
            proc.kill()
            tmp.unlink(missing_ok=True)
            raise
        if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            os.replace(tmp, dst)
            png = src.with_suffix(".png")
            if png.exists():
                await asyncio.to_thread(shutil.copyfile, png, dst.with_suffix(".png"))
            return True
        tmp.unlink(missing_ok=True)
        logging.log(
            "generate.optimize_error",
            id=src.stem,
            message=(stderr.decode(errors="replace")[:500] if stderr else f"node exit {proc.returncode}"),
        )
        return False


def _artifact_url(runs_dir: Path, path: Path) -> str:
    return f"/artifacts/{path.relative_to(runs_dir).as_posix()}"


# scenario -> (template/event step name, {SCENE_CONTEXT} target marker text)
_DECOMP_STEPS: dict[str, tuple[str, str]] = {
    "anchor": (
        "anchor_decompose",
        "This is the subregion you are to generate a list of objects for.",
    ),
    "encapsulating": (
        "encapsulating_decompose",
        "This is the region you are to decide whether it has shared geometry with surrounding areas for, and if so, what objects in that set have yet to be generated",
    ),
    "negative-space": (
        "negative_space_decompose",
        "This is the region whose interstitial negative space you are filling.",
    ),
}


async def _split_objects(
    *,
    zone: Node,
    specs: list[Any],
    all_nodes: list[Node],
) -> list[Any]:
    """object_decomp — the object-splitting pass between a region's object
    decomposition and the bbox solver.

    Takes the object specs an upstream step (anchor / encapsulating /
    negative-space / next_object) proposed for `zone` and returns the final,
    generation-ready set: any object the mesh step can't produce as one coherent
    mesh — a container whose interior must hold other objects, a collection
    standing in for many separate items, a surface carrying openings or that has
    to conform to an uneven run of neighbours — is split into its constituent
    pieces (each its own spec, with references re-pointed onto the pieces), while
    everything already single-mesh-buildable passes through unchanged. The
    decompose steps used to carry this splitting burden inline; concentrating it
    here lets them reason about WHAT belongs in the region without also
    pre-fracturing their own output.

    A pass-through — the proposed specs are returned untouched — when the active
    prompt version carries no `object_decomp` template (older versions still
    split inline) or there is nothing to split. Runs only on a FRESH
    decomposition; a resume replays the already-committed post-split specs
    (see `_decompose_objects` / the anchor loop), so it never re-splits."""
    ps = prompt_store.current()
    if not specs or not ps.has("object_decomp"):
        return specs
    by_id = {n.id: n for n in all_nodes}
    variables = scene_context.zone_vars(
        zone_id=zone.id,
        zone_prompt=zone.prompt,
        zone_plan=zone.plan,
        nodes=all_nodes,
        target_text="This is the region whose proposed objects you are to split into buildable pieces.",
    )
    variables["PROPOSED_OBJECTS"] = scene_context.render_to_place_block(
        specs, by_id, parent_zone=zone.id,
    )
    out = await llm.call_llm(
        system=ps.system("object_decomp", variables),
        user=ps.user("object_decomp", variables),
        output_schema=schemas.ObjectDecompOutput,
        node_id=zone.id,
        step="object_decomp",
        template="object_decomp",
        variables=variables,
    )
    return list(out.objects)


async def _decompose_objects(
    *,
    zone: Node,
    scenario: Literal["anchor", "encapsulating", "negative-space"],
    all_nodes: list[Node],
) -> list[Any]:
    """Decompose a zone into object specs in ONE LLM call.

    There is no validate-and-retry: a decomposition whose ids collide with
    already-placed nodes (a boundary/anchor overlap the model can't resolve) is
    no longer re-rolled — the retry only re-billed the call to re-emit the same
    set. The run-wide dedup in `_resolve_and_generate` silently drops any
    colliding spec instead."""
    # Resume: if this (zone, scenario) pass already committed its object set
    # to the log, replay those specs verbatim (ids fixed) instead of asking
    # the LLM to re-decompose — which is exactly where new ids leak in.
    committed_specs = committed.object_specs(zone.id, scenario)
    if committed_specs is not None:
        return committed_specs
    step, target_text = _DECOMP_STEPS[scenario]
    # The encapsulating + negative-space passes may decide a region needs no
    # objects at all, so they emit the `objects_required` gate; the anchor pass
    # always produces its region's defining objects and uses the bare object-list
    # schema (no gate).
    decomp_schema = (
        schemas.GatedObjectDecompOutput
        if scenario in ("encapsulating", "negative-space")
        else schemas.ObjectDecompOutput
    )
    ps = prompt_store.current()
    variables = scene_context.zone_vars(
        zone_id=zone.id,
        zone_prompt=zone.prompt,
        zone_plan=zone.plan,
        nodes=context_cull.for_context(all_nodes, zone.id),
        target_text=target_text,
    )
    out = await llm.call_llm(
        system=ps.system(step, variables),
        user=ps.user(step, variables),
        output_schema=decomp_schema,
        node_id=zone.id,
        step=step,
        template=step,
        variables=variables,
    )
    if isinstance(out, schemas.GatedObjectDecompOutput) and not out.objects_required:
        logging.log_once(
            "generation.decompose.no_objects",
            match_fields=("zone", "scenario"),
            zone=zone.id,
            scenario=scenario,
            emitted=[s.model_dump() for s in out.objects],
        )
        return []
    specs = list(out.objects)
    # object_decomp: split any proposed object the mesh step can't build as a
    # single coherent mesh into its constituent pieces. A pass-through on
    # versions without the step. Runs before uniquify so the emitted split
    # pieces are de-collided against the scene too.
    specs = await _split_objects(zone=zone, specs=specs, all_nodes=all_nodes)
    # Rename any id colliding with an existing node with an incrementing index
    existing_ids = {n.id for n in all_nodes}
    for old, new in uniquify_ids(specs, existing_ids=existing_ids):
        logging.log(
            "generation.id_collision",
            zone=zone.id,
            scenario=scenario,
            old=old,
            new=new,
        )
    return specs


async def _next_object_batch(
    *,
    zone: Node,
    all_nodes: list[Node],
) -> tuple[bool, list[Any]]:
    """Anchor-completion decision in ONE LLM call: the model proposes a LIST of
    objects per round (or sets objects_required=false to finish). Returns
    `(done, objects)`; `objects` is empty when done.

    No validate-and-retry (see `_decompose_objects`): colliding ids are dropped
    silently downstream by the run-wide dedup in `_resolve_and_generate`, and the
    completion loop's own progress guard (`attempted`) stops a model that keeps
    re-proposing already-placed objects."""
    ps = prompt_store.current()
    variables = scene_context.zone_vars(
        zone_id=zone.id,
        zone_prompt=zone.prompt,
        zone_plan=zone.plan,
        nodes=context_cull.for_context(all_nodes, zone.id),
        target_text="This is the subregion you are deciding whether to add more objects to.",
    )
    decision = await llm.call_llm(
        system=ps.system("next_object", variables),
        user=ps.user("next_object", variables),
        output_schema=schemas.GatedObjectDecompOutput,
        node_id=zone.id,
        step="next_object",
        template="next_object",
        variables=variables,
    )
    objects = list(decision.objects)
    if not decision.objects_required or not objects:
        return True, []
    # object_decomp: split this round's proposed detail objects into buildable
    # pieces (a pass-through on versions without the step) before de-colliding.
    objects = await _split_objects(zone=zone, specs=objects, all_nodes=all_nodes)
    # Rename any id colliding with an existing node with an incrementing index
    existing_ids = {n.id for n in all_nodes}
    for old, new in uniquify_ids(objects, existing_ids=existing_ids):
        logging.log(
            "generation.id_collision",
            zone=zone.id,
            scenario="anchor",
            old=old,
            new=new,
        )
    return False, objects


async def _resolve_object_bboxes_batch(
    *,
    specs: list[Any],
    zone: Node,
    all_nodes: list[Node],
) -> tuple[dict[str, BoundingBox], dict[str, int]]:
    """Place every object in `specs` in ONE batch LLM call, which also SOLVES
    each object's discrete yaw from its semantic `orientation` text. Returns `({id: world-frame bbox},
    {id: orientation})`. Objects already committed (resume) keep their world
    position + solved yaw and the LLM is skipped entirely when all are."""
    committed_bboxes = {s.id: committed.bbox(s.id) for s in specs}
    committed_orient = {s.id: committed.orientation(s.id) for s in specs}
    if all(b is not None for b in committed_bboxes.values()):
        return (
            {sid: b for sid, b in committed_bboxes.items() if b is not None},
            {sid: (committed_orient[sid] or 0) for sid in committed_bboxes},
        )
    bbox_by_id = {n.id: n.bbox for n in all_nodes}
    by_id = {n.id: n for n in all_nodes}
    ps = prompt_store.current()
    variables = scene_context.zone_vars(
        zone_id=zone.id,
        zone_prompt=zone.prompt,
        zone_plan=zone.plan,
        nodes=context_cull.for_context(all_nodes, zone.id),
        target_text="This is the subregion whose objects you are to place.",
    )
    variables["TO_PLACE"] = scene_context.render_to_place_block(
        specs, by_id, parent_zone=zone.id,
    )
    out = await llm.call_llm(
        system=ps.system("object_bbox_batch", variables),
        user=ps.user("object_bbox_batch", variables),
        output_schema=schemas.ObjectBboxBatchOutput,
        node_id=zone.id,
        step="object_bbox_batch",
        template="object_bbox_batch",
        variables=variables,
        validate=lambda o: llm.require_matching_ids(
            produced=[a.id for a in o.assignments],
            expected=[s.id for s in specs],
            step="object_bbox_batch",
        ),
    )
    # LLM emits each object's bbox in that object's parent's local frame.
    # Convert to world coordinates per-object. Handle intra-batch parents
    # (spec B parents to spec A in same batch) via topological resolution.
    spec_parent = {s.id: s.parent for s in specs}
    assignments_by_id = {a.id: a.bbox for a in out.assignments}
    orientations = {a.id: a.orientation for a in out.assignments}
    bboxes: dict[str, BoundingBox] = {}
    remaining = set(assignments_by_id.keys())
    while remaining:
        progress = False
        for obj_id in list(remaining):
            parent_id = spec_parent.get(obj_id, zone.id)
            if parent_id in bboxes:
                parent_bbox = bboxes[parent_id]
            elif parent_id in bbox_by_id:
                parent_bbox = bbox_by_id[parent_id]
            elif parent_id in remaining:
                continue
            else:
                parent_bbox = zone.bbox
            bboxes[obj_id] = assignments_by_id[obj_id].to_world_frame(parent_bbox)
            remaining.discard(obj_id)
            progress = True
        if not progress:
            for obj_id in list(remaining):
                bboxes[obj_id] = assignments_by_id[obj_id].to_world_frame(zone.bbox)
            remaining.clear()
    for sid, b in committed_bboxes.items():
        if b is not None:
            bboxes[sid] = b
            if committed_orient[sid] is not None:
                orientations[sid] = committed_orient[sid]
    return bboxes, orientations


async def _resolve_and_generate(
    *,
    specs: list[Any],
    zone: Node,
    all_nodes: list[Node],
    scenario: Literal["anchor", "encapsulating", "negative-space"],
    runs_dir: Path,
    run_id: str,
) -> list[Node]:
    # Run-wide dedup. The LLM occasionally emits the same id twice in one
    # decomposition (e.g. duplicate floor slabs in encapsulating mode), and
    # later recursion levels can re-surface an id already in flight. Drop
    # any spec whose id collides with an already-admitted id, an already-
    # placed node, or an earlier spec in this same call. Without this guard
    # we re-bill Banana + Trellis for every duplicate, since the artifact
    # cache key is the node id but the cache lookup races against
    # concurrent submissions.
    admitted = _admitted_ids.setdefault(run_id, set())
    placed_ids = {n.id for n in all_nodes}
    deduped: list[Any] = []
    seen_in_call: set[str] = set()
    for s in specs:
        # Silently drop any spec whose id collides with an earlier spec in this
        # call, an already-admitted (in-flight) id, or an already-placed node.
        # Not logged: the decompose retry that used to surface these overlaps was
        # removed, so a benign boundary/anchor overlap is just quietly deduped.
        if s.id in seen_in_call or s.id in admitted or s.id in placed_ids:
            continue
        seen_in_call.add(s.id)
        deduped.append(s)
    if not deduped:
        return []
    admitted.update(seen_in_call)
    specs = deduped

    # Resolve every object's world-frame bbox in one batch LLM call (honoring
    # committed bboxes for resume). The anchor completion loop calls this with
    # the objects it proposed that round.
    bboxes, orientations = await _resolve_object_bboxes_batch(
        specs=specs,
        zone=zone,
        all_nodes=all_nodes,
    )

    # Emit EVERY resolved bbox upfront — before the per-object image-prompt /
    # library-match calls below — so the whole sibling set appears at once right
    # after the batch step (matching how the divider emits child bboxes). Without
    # this the boxes dribble out one at a time, paused behind each per-object
    # call in the step-through. emit_bbox dedups, so the reuse paths below are
    # unaffected.
    _bbox_kind = "frame" if scenario == "encapsulating" else "object"
    for spec in specs:
        logging.emit_bbox(
            spec.id,
            bboxes[spec.id],
            parent_id=spec.parent,
            prompt=spec.prompt,
            kind=_bbox_kind,
            proxy_shape=spec.proxy_shape,
            orientation=orientations[spec.id],
        )

    # Distill each object's verbose seed into its concise visual subject phrase
    # (the `image_prompt` LLM step), for BOTH the library and from-scratch flows.
    # The distilled phrase is the object's canonical "what it is" description and
    # drives every downstream decision that wants a direct physical description
    # rather than the placement-laden seed: library matching, the symmetry cut
    # plane, prefab grouping, and the Nano-Banana image. The prior-subjects
    # context fed to each call is the bare distilled phrases of already-placed
    # nodes (Node.noun_phrase), never the wrapped Nano-Banana directives — leaking
    # the wrapper boilerplate would just teach the model to echo it back.
    committed_subjects = [n.noun_phrase or n.prompt for n in all_nodes if n.mesh_url is not None]
    subjects: dict[str, str] = {}
    prior_subjects = list[str](committed_subjects)
    for spec in specs:
        subject = await _distill_subject(
            spec_id=spec.id,
            prompt=spec.prompt,
            bbox=bboxes[spec.id],
            proxy_shape=spec.proxy_shape,
            prior_prompts=prior_subjects,
            zone=zone,
            nodes=all_nodes,
        )
        subjects[spec.id] = subject
        prior_subjects.append(subject)

    if _USE_ASSET_LIBRARY:
        return await _match_library_assets(
            specs=specs,
            bboxes=bboxes,
            orientations=orientations,
            subjects=subjects,
            scenario=scenario,
            runs_dir=runs_dir,
            run_id=run_id,
            zone_id=zone.id,
        )

    # From-scratch: resolve the symmetry cut plane FROM the distilled subject
    # (not the verbose seed), then wrap that subject into the Nano-Banana
    # studio-shot directive for the resolved view.
    resolved: list[Node] = []
    for spec in specs:
        bbox = bboxes[spec.id]
        subject_prompt = subjects[spec.id]
        encapsulating = scenario == "encapsulating"
        cut_plane = await symmetry.resolve_cut_plane(
            prompt=subject_prompt,
            node_id=spec.id,
            encapsulating=encapsulating,
        )
        view = symmetry.image_view_for(
            cut_plane=cut_plane, encapsulating=encapsulating,
        )
        image_prompt = scene_context.wrap_image_prompt(
            subject_prompt, spec.proxy_shape, bbox.size, view=view,
        )
        resolved.append(
            Node(
                id=spec.id,
                prompt=spec.prompt,
                noun_phrase=subject_prompt,
                image_prompt=image_prompt,
                bbox=bbox,
                proxy_shape=spec.proxy_shape,
                orientation=orientations[spec.id],
                orientation_description=spec.orientation,
                placement=spec.placement,
                referenced_ids=list(spec.referenced_ids),
                parent_id=spec.parent,
                parent_kind=spec.parent_kind,
                symmetry_cut_plane=cut_plane,
                parent_region=zone.id,
            )
        )

    return await _spawn_meshes(
        resolved=resolved,
        runs_dir=runs_dir,
        run_id=run_id,
        scenario=scenario,
    )


async def _match_library_assets(
    *,
    specs: list[Any],
    bboxes: dict[str, BoundingBox],
    orientations: dict[str, int],
    subjects: dict[str, str],
    scenario: Literal["anchor", "encapsulating", "negative-space"],
    runs_dir: Path,
    run_id: str,
    zone_id: str,
) -> list[Node]:
    from app.services import library

    objs_dir = runs_dir / run_id / "objects"
    objs_dir.mkdir(parents=True, exist_ok=True)

    async def _one(spec: Any) -> Node:
        bbox = bboxes[spec.id]
        orientation = orientations[spec.id]
        path = objs_dir / f"{spec.id}.glb"
        url = _artifact_url(runs_dir, path)

        # Reuse an on-disk mesh ONLY when THIS log already committed its `model`
        # event — a resume, or a sim branch's replayed prefix — where the file was
        # baked for the committed bbox + orientation, so it's correct to keep (and
        # re-asserting the event is a dedup no-op). A file that exists *only*
        # because the branch hardlinked the source objects/ dir, with NO committed
        # model here, is the SOURCE's mesh: baked to fill the SOURCE's bbox at the
        # SOURCE's yaw. The re-run can resolve a different bbox/orientation — the
        # fill scale AND the yaw are baked per orientation (see glb_place.place_glb),
        # so a changed yaw mis-fits the box — and an edited prompt can match a
        # different asset, so fall through and re-match + re-place it freshly.
        if path.exists() and logging.find_event("model", id=spec.id, artifact_kind="object") is not None:
            logging.emit_model(spec.id, artifact_kind="object", url=url)
            return Node(
                id=spec.id,
                prompt=spec.prompt,
                noun_phrase=subjects[spec.id],
                bbox=bbox,
                proxy_shape=spec.proxy_shape,
                orientation=orientation,
                orientation_description=spec.orientation,
                placement=spec.placement,
                referenced_ids=list(spec.referenced_ids),
                parent_id=spec.parent,
                parent_kind=spec.parent_kind,
                parent_region=zone_id,
                mesh_url=url,
            )

        # bbox already emitted upfront in _resolve_and_generate. Match on the
        # distilled visual subject (not the placement-laden seed) so the library
        # pick keys off a clean physical description. The same phrase is recorded
        # as the `image` event's prompt, which is what the from-scratch generate
        # gate reconstructs each node's prompt from (see routes._reconstruct_node),
        # so the distilled subject carries through to that gate's cut-plane /
        # prefab / Nano-Banana decisions.
        subject = subjects[spec.id]
        match = await library.match(subject)
        asset = library.asset_path(match.library_id)

        logging.log(
            "library.match",
            id=spec.id,
            prompt=subject,
            library_id=match.library_id,
        )

        ref_image = asset.with_suffix(".png")
        if ref_image.exists():
            dest_image = objs_dir / f"{spec.id}.png"
            # Break a hardlink to the source's image before overwriting, so a
            # branch re-match can't mutate the source cell's shared inode.
            dest_image.unlink(missing_ok=True)
            await asyncio.to_thread(shutil.copy2, ref_image, dest_image)
            logging.log(
                "image",
                id=spec.id,
                url=_artifact_url(runs_dir, dest_image),
                prompt=subject,
            )

        if asset.exists():
            # The optimized asset is Meshopt/KTX2-compressed, so we bake the
            # placement as a node-graph transform (preserving the compressed
            # bytes) instead of decoding + re-exporting through trimesh.
            bounds = library.asset_rotated_bounds(match.library_id, orientation)
            # Break any hardlink to the source mesh before (re)baking — writing in
            # place would corrupt the source cell's shared inode.
            path.unlink(missing_ok=True)
            if bounds is not None:
                await asyncio.to_thread(
                    glb_place.place_glb,
                    src=asset,
                    dst=path,
                    bbox=bbox,
                    orientation=orientation,
                    rotated_min=bounds[0],
                    rotated_max=bounds[1],
                )
            else:
                # Asset missing from the manifest: copy through unscaled so the
                # placement still renders rather than vanishing.
                await asyncio.to_thread(shutil.copyfile, asset, path)
                logging.log(
                    "library.bounds_missing",
                    id=spec.id,
                    library_id=match.library_id,
                )
            logging.emit_model(
                spec.id,
                artifact_kind="object",
                url=url,
            )
        else:
            logging.log(
                "library.asset_missing",
                id=spec.id,
                library_id=match.library_id,
            )

        return Node(
            id=spec.id,
            prompt=spec.prompt,
            noun_phrase=subjects[spec.id],
            bbox=bbox,
            proxy_shape=spec.proxy_shape,
            orientation=orientation,
            orientation_description=spec.orientation,
            placement=spec.placement,
            referenced_ids=list(spec.referenced_ids),
            parent_id=spec.parent,
            parent_kind=spec.parent_kind,
            parent_region=zone_id,
            mesh_url=url,
        )

    # Fan the matches out in parallel: each spec's match -> ref-image ->
    # placement pipeline is independent (its LLM prompt depends only on the
    # spec's own text), so only event arrival order changes. gather keeps
    # spec order for the returned nodes; the first exception aborts the
    # call, same as the sequential loop.
    return list(await asyncio.gather(*(_one(s) for s in specs)))

async def _distill_subject(
    *,
    spec_id: str,
    prompt: str,
    bbox: BoundingBox,
    proxy_shape: ProxyShape | None,
    prior_prompts: list[str],
    zone: Node | None = None,
    nodes: list[Node] | None = None,
    force: bool = False,
) -> str:
    """Run the `image_prompt` LLM step: distill the verbose seed `prompt` into a
    concise visual noun phrase — what the object physically IS, stripped of the
    placement / scene-context language the decompose step authors. This distilled
    phrase is the object's canonical subject: stored as Node.prompt and used for
    library matching, the symmetry cut plane, prefab grouping, and (wrapped)
    Nano-Banana — every step that wants a direct physical description.

    Resumes from the committed `image` event when present, so a replayed object
    keeps the exact phrase the rest of the scene already references downstream.

    When `DOWNGRADE_NOUN_PHRASE` is set, this step runs on gemini-flash-lite
    instead of the run's configured model — both the automatic and `force` paths —
    distilling the phrase cheaply, off the benchmark model surface.

    `force=True` (the regenerate-noun-phrase path) skips BOTH the committed-`image`
    short-circuit and the content-addressed LLM cache (via `call_llm_once`), so the
    phrase is re-distilled fresh every call — a repeated regenerate actually
    re-rolls instead of replaying the cached phrase, and an object that has no
    noun phrase yet still gets one."""
    if not force:
        subject = committed.image_subject(spec_id)
        if subject is not None:
            return subject
    ps = prompt_store.current()
    variables = scene_context.image_prompt_vars(
        prompt=prompt,
        bbox=bbox,
        proxy_shape=proxy_shape,
        prior_prompts=prior_prompts,
        zone=zone,
        nodes=nodes,
    )
    # DOWNGRADE_NOUN_PHRASE: force this step onto gemini-flash-lite regardless of
    # the run's configured model (the same override library-match / symmetry use).
    # Set the model ContextVar once so both call paths inherit it; the finally-reset
    # restores the run model even if call_llm re-aims it internally.
    token = (
        llm._current_model.set(MODELS["gemini-flash-lite"])
        if _DOWNGRADE_NOUN_PHRASE
        else None
    )
    try:
        if force:
            # Uncached, unlogged-as-cache.llm one-shot: no cache read (so it re-rolls)
            # and no cache.llm write (so the NEXT forced re-roll also misses). The new
            # phrase becomes the source of truth via a fresh `image` event the caller
            # logs, not via this call.
            validated, *_ = await llm.call_llm_once(
                system=ps.system("image_prompt", variables),
                user=ps.user("image_prompt", variables),
                output_schema=schemas.ImagePromptOutput,
                model=llm.current_model(),
                step="image_prompt",
            )
            return validated.prompt
        out = await llm.call_llm(
            system=ps.system("image_prompt", variables),
            user=ps.user("image_prompt", variables),
            output_schema=schemas.ImagePromptOutput,
            node_id=spec_id,
            step="image_prompt",
            template="image_prompt",
            variables=variables,
        )
        return out.prompt
    finally:
        if token is not None:
            llm._current_model.reset(token)


async def _build_image_prompt(
    *,
    spec_id: str,
    prompt: str,
    bbox: BoundingBox,
    proxy_shape: ProxyShape | None,
    prior_prompts: list[str],
    view: str = "front",
    include_dimensions: bool = True,
    zone: Node | None = None,
    nodes: list[Node] | None = None,
) -> tuple[str, str]:
    """Returns (subject_phrase, wrapped_image_prompt): the distilled subject
    (see `_distill_subject`) plus the full Nano-Banana studio-shot directive for
    `view`. Used where both are wanted in one shot — the mesh-retry refresh and
    the standalone library-asset builder. The main generate path distills and
    wraps separately, since the symmetry view must be resolved (from the distilled
    subject) before the wrap.

    Set include_dimensions=False for library generation where objects have no
    meaningful bbox — omits the dimension constraint so the model renders natural
    proportions."""
    subject = await _distill_subject(
        spec_id=spec_id,
        prompt=prompt,
        bbox=bbox,
        proxy_shape=proxy_shape,
        prior_prompts=prior_prompts,
        zone=zone,
        nodes=nodes,
    )
    dims = bbox.size if include_dimensions else None
    return subject, scene_context.wrap_image_prompt(subject, proxy_shape, dims, view=view)


async def _refresh_node_image_prompt(
    node: Node,
    *,
    regen_noun_phrase: bool = False,
    seed_prompt: str | None = None,
    zone: Node | None = None,
    nodes: list[Node] | None = None,
) -> Node:
    """Re-resolve symmetry + wrap after a mesh retry deleted the reference image.
    Replays `symmetry.decision` from the log when present.

    With `regen_noun_phrase=True`, re-distill a FRESH noun phrase from the
    object's authored seed (`seed_prompt`, falling back to the node's current
    prompt) via a forced `image_prompt` call — so a regenerate can re-roll the
    noun phrase, even for an object that has none yet. The fresh phrase becomes
    the node's prompt + wrapped Nano-Banana directive; the caller re-logs the
    `image` event with it (see `_generate_one(relog_image=...)`) so later
    regenerations, which read the `image` log, pick up the new phrase."""
    cut_plane = await symmetry.resolve_cut_plane(
        prompt=node.prompt, node_id=node.id,
    )
    view = symmetry.image_view_for(cut_plane=cut_plane)
    if regen_noun_phrase:
        # Re-distill against the SAME scene context the original image_prompt step
        # saw (`{ROOT_HEADER}`, `{ZONE_*}`, `{SCENE_CONTEXT}`, prior phrases),
        # reconstructed by the caller — not an empty render. The node being
        # described is dropped from that context: the live first pass distills
        # before appending the node to `all_nodes`, and excluding it stops a re-roll
        # from simply echoing the object's own existing phrase back. `zone`/`nodes`
        # are None when the caller couldn't rebuild the tree, in which case the
        # step still runs, just context-free.
        scene = [n for n in nodes if n.id != node.id] if nodes else nodes
        priors = [n.noun_phrase for n in (scene or []) if n.noun_phrase is not None]
        subject = await _distill_subject(
            spec_id=node.id,
            prompt=seed_prompt or node.prompt,
            bbox=node.bbox,
            proxy_shape=node.proxy_shape,
            prior_prompts=priors,
            zone=zone,
            nodes=scene,
            force=True,
        )
        image_prompt = scene_context.wrap_image_prompt(
            subject, node.proxy_shape, node.bbox.size, view=view,
        )
    else:
        subject = committed.image_subject(node.id) or node.prompt
        _, image_prompt = await _build_image_prompt(
            spec_id=node.id,
            prompt=subject,
            bbox=node.bbox,
            proxy_shape=node.proxy_shape,
            prior_prompts=[],
            view=view,
        )
    return node.model_copy(
        update={
            "prompt": subject,
            "image_prompt": image_prompt,
            "symmetry_cut_plane": cut_plane,
        },
    )


_pending: dict[str, list[asyncio.Task[None]]] = {}

# Run-scoped set of every id that has been admitted into _resolve_and_generate
# for this run. Populated *synchronously* before the bbox-resolution LLM call,
# so concurrent specs (within or across decomposition steps) can't both pass
# the dedup check and race into Banana+Trellis. The per-stage `.done`
# completion cache catches *most* repeats, but two specs that submit before
# either has logged a `trellis.done` will both miss the cache and
# double-bill — this guard closes that window.
_admitted_ids: dict[str, set[str]] = {}

# Per-(cell, version) map of per-node asyncio.Locks. The from-scratch whole-scene
# generate (`generate_assets`) and standalone per-asset regeneration
# (`regenerate_one` / `propagate_reuses`) may run CONCURRENTLY on the same version;
# this serializes them whenever they would write the SAME node's files (raw /
# rescaled / optimized GLB + reference image), so a node is never built by two
# writers at once. Distinct nodes — and distinct versions of the same node — never
# contend, so the common case (regenerating already-built assets while the scene
# build resumes the missing ones, or building two versions at once) stays fully
# parallel. A reuse's read of its canonical's raw is lock-free and kept safe
# instead by atomic (temp + replace) writes.
_node_locks: dict[tuple[str, str], dict[str, asyncio.Lock]] = {}


def node_lock(run_id: str, version: str, node_id: str) -> asyncio.Lock:
    """The build lock for one (cell, version, node), created on first use. Shared
    by the whole-scene generate and per-asset regeneration so the two serialize
    only when they target the same node in the same version."""
    per_version = _node_locks.setdefault((run_id, version), {})
    lock = per_version.get(node_id)
    if lock is None:
        lock = asyncio.Lock()
        per_version[node_id] = lock
    return lock


def clear_node_locks(run_id: str) -> None:
    """Drop every version's node locks for a cell (called on reset / teardown)."""
    for key in [k for k in _node_locks if k[0] == run_id]:
        _node_locks.pop(key, None)


async def _generate_one(
    node: Node,
    *,
    raw: Path,
    path: Path,
    image_stem: Path,
    runs_dir: Path,
    backend: str = DEFAULT_MESH_BACKEND,
    reuse_image: bool = False,
    force_image: bool = False,
    force_mesh: bool = False,
    relog_image: bool = False,
) -> bool:
    """Build the node's image + raw mesh + rescaled served GLB, emitting its
    `model` event on success. Returns whether it succeeded. Non-destructive: the
    image (Nano-Banana), the raw (the mesh backend), and the rescaled served GLB
    are each written only on success (atomically for the meshes), so a failure at
    any step leaves the node's prior artifacts intact — never a headless node.

    `force_mesh=True` forces a fresh mesh from the backend even when a completed
    one is cached, so a regenerate rebuilds WITHOUT first deleting the old raw."""
    try:
        # Nano Banana sees the wrapped studio-shot directive; node.prompt
        # stays the bare subject phrase for everything else.
        banana_prompt = node.image_prompt or node.noun_phrase or node.prompt
        image_path = image_stem.parent / f"{image_stem.name}.png"
        had_image = image_path.exists() and logging.find_event("image", id=node.id) is not None
        if reuse_image:
            # "Regenerate from image": reuse the reference image already on disk
            # and NEVER call Nano-Banana. The on-disk image is always PNG
            # (nano_banana normalizes; seeded/recovered copies are PNG). If it's
            # genuinely gone (and couldn't be recovered from a prefab sibling
            # upstream), fail clearly rather than silently re-generating it.
            if not image_path.exists():
                logging.log(
                    "mesh.error", id=node.id,
                    message="regenerate from image: no reference image to reuse",
                )
                return False
            image = nano_banana.NanoBananaResult(
                image_bytes=image_path.read_bytes(), mime_type="image/png",
            )
        else:
            # Fresh image. `force_image` bypasses the resumable completion cache so
            # a regenerate actually re-rolls (instead of reusing the cached image)
            # while still writing only on success — a failed call leaves any
            # existing image intact instead of wiping it.
            image = await nano_banana.generate_resumable(
                banana_prompt,
                job_id=node.id,
                save_to=image_path,
                force=force_image,
            )
        if not had_image or relog_image:
            # `relog_image` (the regenerate-noun-phrase path) appends a NEW `image`
            # event so the freshly-distilled phrase becomes the latest one — which
            # is what `committed.image_subject` / the displayed image prompt / a
            # later regenerate all read.
            logging.log(
                "image",
                id=node.id,
                url=_artifact_url(runs_dir, image_path),
                prompt=node.noun_phrase or node.prompt,
            )
        # generate_mesh writes `raw` on a fresh run but returns a *cached* path
        # on a resumable hit (which may not be `raw` when the bound log was
        # hydrated from another build). Load whatever it actually produced. It
        # writes `raw` atomically and (with force_mesh) never reads a pre-deleted
        # file to force a rebuild — so `raw` is only ever replaced on success.
        produced = await MESH_BACKENDS[backend](
            image.image_bytes,
            output_path=raw,
            job_id=node.id,
            image_mime=image.mime_type,
            bbox=node.bbox,
            force=force_mesh,
        )
        async with _MESH_IO:
            scene = await asyncio.to_thread(trimesh.load, produced)
            scene = await symmetry.apply_symmetrize(
                scene,
                cut_plane=node.symmetry_cut_plane,
                node_id=node.id,
            )
            rescaled = await asyncio.to_thread(
                rescale_mesh_to_bbox,
                scene,
                node.bbox,
                orientation=node.orientation,
            )
            # Window/glass objects: bake per-texel transparency into the base
            # color (white texels -> near-clear) before export. Gated on the object
            # reading as glass AND having been decided to be symmetrized (a flat
            # glazed panel). Modular + removable — drop this block and the `glass`
            # import to disable. Logged here, in the async context, since the slot
            # log isn't thread-safe.
            glass_stats = await asyncio.to_thread(
                glass.apply_window_glass_transparency,
                rescaled,
                noun_phrase=node.noun_phrase,
                prompt=node.prompt,
                symmetrized=node.symmetry_cut_plane != "none",
            )
            if glass_stats is not None:
                logging.log("mesh.glass", id=node.id, **glass_stats)
            # Atomic write: export to a temp then replace, so a concurrent reader
            # (the client streaming raw meshes, or a reuse) never sees a torn GLB.
            tmp_path = path.with_name(f"{path.name}.part")
            try:
                await asyncio.to_thread(export_glb, rescaled, tmp_path)
                os.replace(tmp_path, path)
            finally:
                tmp_path.unlink(missing_ok=True)
            del scene, rescaled
        logging.emit_model(
            node.id,
            artifact_kind="object",
            url=_artifact_url(runs_dir, path),
        )
        return True
    except Exception as e:
        logging.log("mesh.error", id=node.id, message=f"{type(e).__name__}: {e}")
        return False


async def _rescale_reuse_from_raw(
    node: Node,
    *,
    src_raw: Path,
    raw_dir: Path,
    opt_dir: Path,
    source_id: str,
) -> None:
    """Rescale a canonical's raw Trellis mesh into `node`'s bbox + orientation
    (the prefab-reuse path — no Nano-Banana, no Trellis), carry over the
    canonical's reference image, then optimize into the served twin. Shared by
    the generate gate's in-scene reuse and standalone prefab propagation. A
    missing source raw is logged and skipped rather than raising."""
    if not src_raw.exists():
        logging.log("prefab.reuse_missing", id=node.id, source=source_id)
        return
    try:
        rescaled = raw_dir / f"{node.id}.glb"
        # Replay the CANONICAL's mirror — plane AND kept half — so the reuse lands
        # identically posed, including a non-default direction set via the
        # symmetrize control.
        cut_plane, keep_positive = symmetry.applied_state(source_id)
        async with _MESH_IO:
            scene = await asyncio.to_thread(trimesh.load, src_raw)
            scene = await symmetry.apply_symmetrize(
                scene, cut_plane=cut_plane, node_id=node.id, keep_positive=keep_positive,
            )
            placed = await asyncio.to_thread(
                rescale_mesh_to_bbox, scene, node.bbox, orientation=node.orientation,
            )
            # Atomic write (see _generate_one) so a concurrent reader never tears.
            tmp_rescaled = rescaled.with_name(f"{rescaled.name}.part")
            try:
                await asyncio.to_thread(export_glb, placed, tmp_rescaled)
                os.replace(tmp_rescaled, rescaled)
            finally:
                tmp_rescaled.unlink(missing_ok=True)
            del scene, placed
        src_png = raw_dir / f"{source_id}.png"
        if src_png.exists():
            await asyncio.to_thread(shutil.copyfile, src_png, raw_dir / f"{node.id}.png")
        if await _optimize_asset(rescaled, opt_dir / f"{node.id}.glb"):
            # Every other build path leaves a success marker in the log (a
            # `<scope>.done`, a `model` event); a reuse landed silently, so one
            # that failed once still read as failed after a later rebuild fixed
            # it. Gated on the optimize, which logs its own `generate.optimize_error`
            # first when it fails — so that failure stays the node's latest word.
            logging.log("prefab.reuse_derived", id=node.id, source=source_id)
    except Exception as e:  # noqa: BLE001
        logging.log("mesh.error", id=node.id, message=f"{type(e).__name__}: {e}")


async def _spawn_meshes(
    *,
    resolved: list[Node],
    runs_dir: Path,
    run_id: str,
    scenario: Literal["anchor", "encapsulating", "negative-space"],
) -> list[Node]:
    objs_dir = runs_dir / run_id / "objects"
    objs_dir.mkdir(parents=True, exist_ok=True)

    out: list[Node] = []
    pending = _pending.setdefault(run_id, [])
    for node in resolved:
        raw = objs_dir / f"{node.id}.raw.glb"
        path = objs_dir / f"{node.id}.glb"
        image_stem = objs_dir / node.id
        url = _artifact_url(runs_dir, path)
        if path.exists():
            logging.emit_model(node.id, artifact_kind="object", url=url)
            out.append(node.model_copy(update={"mesh_url": url}))
            continue
        logging.log("mesh.submit", id=node.id, prompt=node.prompt)
        pending.append(
            asyncio.create_task(
                _generate_one(
                    node,
                    raw=raw,
                    path=path,
                    image_stem=image_stem,
                    runs_dir=runs_dir,
                ),
            )
        )
        out.append(node.model_copy(update={"mesh_url": url}))
    return out


# One lock per (cell, version) so a version's generate gate and a concurrent regen
# of that SAME version compute its prefab grouping once instead of racing to
# re-sweep + double-log it. Distinct versions group independently (each owns its
# log), so they never contend.
_scene_prefab_locks: dict[tuple[str, str], asyncio.Lock] = {}


async def ensure_scene_prefab_groups(
    *, nodes: list[Node], run_id: str, version: str,
) -> dict[str, str]:
    """Compute the prefab grouping over `nodes` ONCE for one generated `version`
    and persist it as `prefab.match` events in the BOUND (that version's) log,
    shared by the version's whole-scene generate and standalone regen of its
    assets. Each version groups INDEPENDENTLY — the grouping is a per-version
    artifact, so a link/unlink in one version never affects another. Prefab
    matching is for the from-scratch generated build only — the asset library
    picks an asset per object as it is generated and never groups.

    Seed-and-sweep: the first undecided node is a canonical "seed"; a flash-lite
    call names every remaining node that is the SAME object (each reuses the
    seed), then the next still-undecided node seeds the following group. Idempotent
    — nodes already carrying a logged `prefab.match` are honored and never
    re-swept, so a later regen or a resume is free.
    Returns node_id -> reuse_id ("" = canonical)."""
    lock = _scene_prefab_locks.setdefault((run_id, version), asyncio.Lock())
    async with lock:
        decisions: dict[str, str] = {}
        for node in nodes:
            decided = logging.find_event("prefab.match", id=node.id)
            if decided is not None:
                decisions[node.id] = str(decided.get("reuse_id") or "")
        by_id = {n.id: n for n in nodes}
        for node in nodes:
            if node.id in decisions:
                continue
            decisions[node.id] = ""
            logging.log("prefab.match", id=node.id, reuse_id="", description=node.prompt)
            candidates = [
                (n.id, n.prompt, n.bbox) for n in nodes if n.id not in decisions
            ]
            for dup_id in await prefabs.match_duplicates(
                seed_id=node.id,
                seed_description=node.prompt,
                seed_bbox=node.bbox,
                candidates=candidates,
            ):
                decisions[dup_id] = node.id
                logging.log(
                    "prefab.match", id=dup_id, reuse_id=node.id,
                    description=by_id[dup_id].prompt,
                )
        return decisions


async def generate_assets(
    *,
    nodes: list[Node],
    decisions: dict[str, str],
    runs_dir: Path,
    run_id: str,
    version: str,
) -> None:
    """From-scratch (Nano-Banana + a mesh backend) build of `nodes` for the
    client's "generate" gate, independent of `_USE_ASSET_LIBRARY` and the library
    build's `objects/`. The mesh backend is `GENERATE_SCENE_BACKEND` (default
    Trellis; `hunyuan-tencent` routes the whole scene through Tencent's Hunyuan 3D
    3.1, one job at a time). Writes into ONE generated `version` of the cell — any
    number coexist, each fully isolated by its own dirs + log. Two dirs under
    `generated/<version>/`:

      * `objects-generated/`            raw Trellis meshes (intermediate)
      * `objects-generated-optimized/`  the served set — each raw mesh run
        through the SAME optimizer the asset library uses (decimate + prune +
        KTX2 textures + Meshopt), so a generated asset is never streamed
        un-optimized (raw Trellis output is ~500k tris / tens of MB each).

    Each node carries the bbox / orientation / image_prompt reconstructed from
    the library build's log, so every generated mesh is rescaled into the exact
    same bounding box the library asset occupied. Nodes that already have an
    OPTIMIZED twin are skipped, so re-running this is how the gate "resumes": it
    regenerates only the failed/interrupted assets and leaves finished ones
    untouched.

    PREFAB REUSE: `decisions` (node_id -> reuse_id, "" = canonical) is this
    version's prefab grouping computed once by `ensure_scene_prefab_groups`. This
    gate only APPLIES it: a reuse skips Nano-Banana + Trellis entirely (its mesh is
    the canonical's raw Trellis output rescaled into its slot). The grouping is
    seeded into this version's log — honoring any per-version promotion already on
    disk (a reuse rebuilt standalone logs its own `prefab.match`) and never flipping
    a pre-built asset — so regen keeps resolving groups here.

    Every asset fans out at once into an uncapped queue; the Trellis in-flight cap
    (threed.GENERATE_CONCURRENCY — 100 live submits) plus threed's `_pace_submit`
    (~1s between Trellis `POST /generate`) govern how many actually reach Modal, so
    the batch ramps on instead of bursting into a 429 storm.

    Requires a bound SlotLog: `_generate_one`, `prefabs.match_duplicates`, and the
    nano_banana / threed services record their bookkeeping there. The caller
    binds this version's dedicated log (events.generated.jsonl) so none of this
    lands in the library build's event stream."""
    raw_dir, opt_dir = generated_dirs(runs_dir, run_id, version)
    raw_dir.mkdir(parents=True, exist_ok=True)
    opt_dir.mkdir(parents=True, exist_ok=True)
    backend = _scene_backend()
    logging.log("generate.backend", backend=backend, version=version)

    # Per-scene prefab state: the canonical asset ids (reuse targets) and a "raw on
    # disk" event per canonical so a reuse can wait for a source still in flight.
    # Local to this call.
    canonical_ids: set[str] = set()
    raw_ready: dict[str, asyncio.Event] = {}

    def _has_raw(node_id: str) -> bool:
        return (raw_dir / f"{node_id}.raw.glb").exists()

    def _built(node_id: str) -> bool:
        # Already has a served twin or a raw mesh from a prior run: a resume keeps
        # it canonical (never reclassifies a built asset as a reuse) and a seed
        # never claims it as a duplicate.
        return (opt_dir / f"{node_id}.glb").exists() or _has_raw(node_id)

    async def _fresh(node: Node) -> None:
        # Decide symmetry before the image is made (reconstructed nodes carry no
        # decision, so otherwise the gate leaves every generated mesh un-symmetrized).
        # Symmetric panels switch to a 3/4 view; the plane drives apply_symmetrize,
        # and reuse twins inherit it from this canonical's logged symmetry.applied.
        cut_plane = await symmetry.resolve_cut_plane(prompt=node.prompt, node_id=node.id)
        if cut_plane != node.symmetry_cut_plane:
            view = symmetry.image_view_for(cut_plane=cut_plane)
            image_prompt = node.image_prompt
            if image_prompt and node.prompt != image_prompt:
                image_prompt = scene_context.wrap_image_prompt(
                    node.prompt, node.proxy_shape, node.bbox.size, view=view,
                )
            node = node.model_copy(
                update={"symmetry_cut_plane": cut_plane, "image_prompt": image_prompt},
            )
        rescaled = raw_dir / f"{node.id}.glb"
        # Serialize against a concurrent per-asset regeneration of this same node
        # (both write its files); distinct nodes never contend.
        async with node_lock(run_id, version, node.id):
            # A regeneration may have (re)built this node while we waited for the
            # lock — don't redo it, but still unblock any reuse of this canonical.
            # Gate on the FULL artifact set, not the optimized twin alone, so a
            # torn "ghost" (stale optimized, raw/unoptimized deleted by a failed
            # regen) is rebuilt here instead of skipped.
            if artifacts_complete(raw_dir, opt_dir, node.id, is_reuse=False):
                raw_ready[node.id].set()
                return
            try:
                await _generate_one(
                    node,
                    raw=raw_dir / f"{node.id}.raw.glb",
                    path=rescaled,
                    image_stem=raw_dir / node.id,
                    runs_dir=runs_dir,
                    backend=backend,
                )
            finally:
                # Raw is on disk now (or generation failed) — unblock any reuse
                # waiting on this asset; it re-checks the file and bails if none.
                raw_ready[node.id].set()
            if rescaled.exists():
                await _optimize_asset(rescaled, opt_dir / f"{node.id}.glb")

    async def _reuse(node: Node, source_id: str) -> None:
        # Wait for the source's mesh to land, then rescale ITS raw Trellis output
        # into this node's slot — exactly as a fresh build would, so the reuse
        # lands identically posed. No Nano-Banana, no Trellis. Surfaces in the
        # queue panel nested under its canonical (`canonical=source_id`): waiting
        # while the canonical's mesh is still generating, processing during the
        # rescale, then dropped.
        slot = logging.current_slot_id()
        mesh_jobs.mark_queued(slot, node.id, canonical=source_id)
        try:
            await raw_ready[source_id].wait()
            async with node_lock(run_id, version, node.id):
                # Full-set gate (see `_fresh`): a reuse with a stale optimized twin
                # but a missing unoptimized served mesh is re-derived, not skipped.
                if artifacts_complete(raw_dir, opt_dir, node.id, is_reuse=True):
                    return
                mesh_jobs.mark_processing(slot, node.id, canonical=source_id)
                await _rescale_reuse_from_raw(
                    node,
                    src_raw=raw_dir / f"{source_id}.raw.glb",
                    raw_dir=raw_dir,
                    opt_dir=opt_dir,
                    source_id=source_id,
                )
        finally:
            mesh_jobs.unmark_queued(slot, node.id)

    # Apply the scene grouping. Seed this version's log with each decision so regen
    # / resolve_group / the reuse-images fork read the grouping from here, while
    # honoring a decision already on disk (a per-version promotion) and never
    # flipping a pre-built asset into a reuse.
    tasks: list[asyncio.Task[None]] = []
    for node in nodes:
        decided = logging.find_event("prefab.match", id=node.id)
        if decided is not None:
            reuse_id = str(decided.get("reuse_id") or "")
        else:
            reuse_id = "" if _built(node.id) else decisions.get(node.id, "")
            logging.log("prefab.match", id=node.id, reuse_id=reuse_id, description=node.prompt)

        is_reuse = bool(reuse_id) and reuse_id in canonical_ids
        # "Done" requires the WHOLE artifact set on disk (role-aware), NOT just the
        # served optimized twin: a regen deletes the raw + unoptimized up front, so
        # a build that died midway leaves only the stale optimized — which still
        # renders, hiding the loss. Gating on the full set rebuilds that torn state
        # here instead of skipping it (a re-run keeps retrying until consistent).
        done = artifacts_complete(raw_dir, opt_dir, node.id, is_reuse=is_reuse)
        if not is_reuse:
            # Canonical (a seed or a pre-built original); reuses rescale its raw mesh.
            canonical_ids.add(node.id)
            ev = raw_ready.setdefault(node.id, asyncio.Event())
            # Signal readiness only when the raw is genuinely on disk — a reuse
            # re-derives from it, so a canonical whose raw was deleted (a ghost)
            # must not report ready off its stale optimized twin; `_fresh` sets the
            # event once it has rebuilt the raw.
            if _has_raw(node.id):
                ev.set()

        if done:
            continue
        tasks.append(
            asyncio.create_task(_reuse(node, reuse_id) if is_reuse else _fresh(node))
        )

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def regenerate_one(
    *,
    node: Node,
    runs_dir: Path,
    run_id: str,
    subdir: str = "objects",
    optimize: bool = False,
    backend: str = DEFAULT_MESH_BACKEND,
    version: str | None = None,
    reuse_image: bool = False,
    regen_noun_phrase: bool = False,
    seed_prompt: str | None = None,
    scene_zone: Node | None = None,
    scene_nodes: list[Node] | None = None,
) -> None:
    """Rebuild a single mesh FRESH on `backend` (one of `MESH_BACKENDS`): unlink
    every prior on-disk artifact for this node under `subdir` so the cache-aware
    checks inside `nano_banana.generate_resumable` and the chosen backend's
    `generate_mesh` miss and issue new API calls, then re-run Nano-Banana + that
    backend + rescale. With
    `optimize=True` the freshly built mesh is run through the library optimizer
    into the sibling `objects-generated-optimized/` served twin (the from-scratch
    generated pipeline — `subdir` is then the version's `.../objects-generated`);
    the library path leaves it off.

    `reuse_image=True` keeps the node's existing reference image and rebuilds ONLY
    the mesh ("regenerate from image"): the image isn't unlinked and `_generate_one`
    reuses it instead of calling Nano-Banana. Default False regenerates the image
    too ("regenerate from scratch").

    When `version` is given (the generated path), the rebuild runs under that
    (cell, version, node)'s build lock, so a concurrent whole-scene generate of the
    same version can never write this node's files at the same time. The library
    retry path passes version=None (its build has no concurrent gate).

    Awaitable core shared by the library single-mesh retry (`retry_node`, a
    detached + `_pending`-tracked wrapper) and standalone generated-asset
    regeneration (awaited directly under its own cell task)."""
    if version is None:
        await _rebuild_one(
            node=node, runs_dir=runs_dir, run_id=run_id,
            subdir=subdir, optimize=optimize, backend=backend, reuse_image=reuse_image,
            regen_noun_phrase=regen_noun_phrase, seed_prompt=seed_prompt,
            scene_zone=scene_zone, scene_nodes=scene_nodes,
        )
        return
    async with node_lock(run_id, version, node.id):
        await _rebuild_one(
            node=node, runs_dir=runs_dir, run_id=run_id,
            subdir=subdir, optimize=optimize, backend=backend, reuse_image=reuse_image,
            regen_noun_phrase=regen_noun_phrase, seed_prompt=seed_prompt,
            scene_zone=scene_zone, scene_nodes=scene_nodes,
        )


async def _rebuild_one(
    *,
    node: Node,
    runs_dir: Path,
    run_id: str,
    subdir: str,
    optimize: bool,
    backend: str,
    reuse_image: bool = False,
    regen_noun_phrase: bool = False,
    seed_prompt: str | None = None,
    scene_zone: Node | None = None,
    scene_nodes: list[Node] | None = None,
) -> None:
    objs_dir = runs_dir / run_id / subdir
    objs_dir.mkdir(parents=True, exist_ok=True)
    raw = objs_dir / f"{node.id}.raw.glb"
    path = objs_dir / f"{node.id}.glb"
    image_stem = objs_dir / node.id
    # NON-DESTRUCTIVE regen. Nothing is deleted up front: the backend is forced to
    # rebuild via `force_mesh`/`force_image` (not by deleting the file to trigger a
    # cache miss, which is what used to leave a headless node when a rebuild then
    # failed), and `_generate_one` writes the image, raw, and served GLB only on
    # success (atomically for the meshes). So `<id>.raw.glb` — and the served twin
    # — are only ever REPLACED once the new one is in hand; a failed regen leaves
    # the prior, working asset fully intact instead of a node with no raw to
    # restore or reference.
    #
    # Refresh BEFORE logging: _refresh_node_image_prompt re-rolls the noun phrase
    # (regen_noun_phrase) — or re-reads the committed subject — and rewrites
    # node.prompt, so logging mesh.retry after it records the phrase actually being
    # built. Logging before would capture the stale pre-distillation seed (the
    # reconstructed node still carries the authored prompt), which is why a retry
    # showed the old prompt even though the regenerated `image` event held the new.
    node = await _refresh_node_image_prompt(
        node, regen_noun_phrase=regen_noun_phrase, seed_prompt=seed_prompt,
        zone=scene_zone, nodes=scene_nodes,
    )
    logging.log("mesh.retry", id=node.id, prompt=node.prompt, backend=backend)
    rebuilt = await _generate_one(
        node, raw=raw, path=path, image_stem=image_stem, runs_dir=runs_dir,
        backend=backend, reuse_image=reuse_image, force_image=not reuse_image,
        force_mesh=True, relog_image=regen_noun_phrase,
    )
    # Only re-optimize when the rebuild actually produced a fresh served mesh; on
    # failure `path` still holds the PRIOR served GLB, so re-optimizing it would
    # just churn identical bytes (and the old optimized twin already matches it).
    if optimize and rebuilt and path.exists():
        # Served twin sits beside the raw dir, so the same version subdir
        # (generated/<v>/objects-generated → .../objects-generated-optimized).
        await _optimize_asset(
            path, objs_dir.parent / GENERATED_OPT_SUBDIR / f"{node.id}.glb",
        )


async def retry_node(
    *,
    node: Node,
    runs_dir: Path,
    run_id: str,
) -> asyncio.Task[None]:
    """Standalone re-generation of a single failed mesh in the LIBRARY pipeline
    (`objects/`). Fire-and-forget: the fresh build (`regenerate_one`) runs as a
    detached task registered on `_pending` so `cancel_pending` (slot reset /
    teardown) can tear it down alongside in-flight pipeline meshes."""
    task = asyncio.create_task(
        regenerate_one(node=node, runs_dir=runs_dir, run_id=run_id),
    )
    _pending.setdefault(run_id, []).append(task)
    return task


async def propagate_reuses(
    *,
    canonical_id: str,
    reuses: list[Node],
    runs_dir: Path,
    run_id: str,
    version: str,
) -> None:
    """Re-derive every `reuses` node from `canonical_id`'s raw Trellis mesh —
    rescaling it into each reuse's own bbox/orientation and optimizing into the
    served twin. Pairs with a prior FRESH build of the canonical
    (`regenerate_one` with optimize=True) to push a regenerated prefab out to
    every object that shares it within this generated `version`. Awaited under
    the caller's cell task, so a cancellation there tears the fan-out down via
    the gather."""
    raw_dir, opt_dir = generated_dirs(runs_dir, run_id, version)
    src_raw = raw_dir / f"{canonical_id}.raw.glb"
    slot = logging.current_slot_id()

    async def _one(node: Node) -> None:
        # Lock per reuse so a concurrent whole-scene generate (or another regen)
        # of the same node can't write its files at the same time. Surfaces in the
        # queue panel nested under its canonical while it re-derives.
        mesh_jobs.mark_processing(slot, node.id, canonical=canonical_id)
        try:
            async with node_lock(run_id, version, node.id):
                await _rescale_reuse_from_raw(
                    node, src_raw=src_raw, raw_dir=raw_dir, opt_dir=opt_dir, source_id=canonical_id,
                )
        finally:
            mesh_jobs.unmark_queued(slot, node.id)

    coros = [_one(node) for node in reuses]
    if coros:
        await asyncio.gather(*coros, return_exceptions=True)


async def clone_canonical_raw(
    *,
    runs_dir: Path,
    run_id: str,
    version: str,
    source_id: str,
    dest_id: str,
) -> bool:
    """Copy `source_id`'s raw Trellis mesh (and its reference image) onto
    `dest_id` so `dest_id` becomes a self-sufficient prefab canonical holding the
    SAME shared geometry. Used when a group's canonical is unlinked: the group is
    handed off to one of its reuses, which inherits the canonical's raw here so the
    remaining members keep their shared look (and stay re-derivable) once the old
    canonical regenerates fresh. Runs under `dest_id`'s node lock so it can't race
    a concurrent build/regen of that node. Returns whether the raw was copied (a
    missing source raw is logged and skipped)."""
    raw_dir, _opt_dir = generated_dirs(runs_dir, run_id, version)
    src_raw = raw_dir / f"{source_id}.raw.glb"
    async with node_lock(run_id, version, dest_id):
        if not src_raw.exists():
            logging.log("prefab.reuse_missing", id=dest_id, source=source_id)
            return False
        await asyncio.to_thread(shutil.copyfile, src_raw, raw_dir / f"{dest_id}.raw.glb")
        src_png = raw_dir / f"{source_id}.png"
        if src_png.exists():
            await asyncio.to_thread(shutil.copyfile, src_png, raw_dir / f"{dest_id}.png")
        return True


def recover_group_image(
    runs_dir: Path, run_id: str, version: str, target_id: str, group_ids: list[str],
) -> bool:
    """Best-effort restore of a node's reference image (the raw-dir `<id>.png` a
    from-image rebuild reads) when that copy is missing but the same image still
    exists elsewhere. The same picture is duplicated in a few places:
      * the OPTIMIZED twin (`objects-generated-optimized/<id>.png`) — copied there
        by the optimize pass; survives even when the raw-dir copy was removed;
      * every prefab group member (`group_ids`) holds a copy in either dir (a reuse
        copies its canonical's image).
    So if `<id>.png` is gone we restore it from the first of those that exists,
    letting a from-image rebuild proceed without re-generating. Returns True if the
    raw-dir image is present afterward (already there, or recovered), else False."""
    raw_dir, opt_dir = generated_dirs(runs_dir, run_id, version)
    target_png = raw_dir / f"{target_id}.png"
    if target_png.exists():
        return True
    candidates = [opt_dir / f"{target_id}.png"]
    for gid in group_ids:
        candidates.append(raw_dir / f"{gid}.png")
        candidates.append(opt_dir / f"{gid}.png")
    for src in candidates:
        if src.exists():
            shutil.copyfile(src, target_png)
            logging.log("image.recovered", id=target_id, source=f"{src.parent.name}/{src.stem}")
            return True
    return False


async def unsymmetrize_one(
    *,
    node: Node,
    runs_dir: Path,
    run_id: str,
    version: str,
) -> None:
    """Rebuild a generated object's served mesh from its OWN raw mesh with NO
    symmetry mirror — i.e. reveal the full, un-mirrored model the mirror was
    hiding. No Nano-Banana and no mesh backend: just reload the on-disk raw,
    rescale it into the node's bbox, and re-optimize the served twin, so it's
    effectively instant and free. Pins the node's symmetry decision/applied to
    `none` so a later resume, regeneration, or prefab-reuse keeps it un-mirrored.
    A missing raw (e.g. a prefab reuse with no own raw) is logged and skipped —
    the caller un-symmetrizes the canonical and propagates instead. Runs under the
    node lock so it can't race a concurrent scene build or regen of the same node."""
    raw_dir, opt_dir = generated_dirs(runs_dir, run_id, version)
    src_raw = raw_dir / f"{node.id}.raw.glb"
    async with node_lock(run_id, version, node.id):
        if not src_raw.exists():
            logging.log("symmetry.skip", id=node.id, reason="unsymmetrize: no raw mesh on disk")
            return
        # Pin the decision so the resolver, the reuse-from-raw path, and any future
        # regeneration all stop re-mirroring this node.
        logging.log("symmetry.decision", id=node.id, cut_plane="none", encapsulating=False)
        logging.log("symmetry.applied", id=node.id, cut_plane="none")
        rescaled = raw_dir / f"{node.id}.glb"
        try:
            async with _MESH_IO:
                scene = await asyncio.to_thread(trimesh.load, src_raw)
                placed = await asyncio.to_thread(
                    rescale_mesh_to_bbox, scene, node.bbox, orientation=node.orientation,
                )
                tmp = rescaled.with_name(f"{rescaled.name}.part")
                try:
                    await asyncio.to_thread(export_glb, placed, tmp)
                    os.replace(tmp, rescaled)
                finally:
                    tmp.unlink(missing_ok=True)
                del scene, placed
            await _optimize_asset(rescaled, opt_dir / f"{node.id}.glb")
            logging.log("symmetry.unsymmetrized", id=node.id)
        except Exception as e:  # noqa: BLE001
            logging.log("mesh.error", id=node.id, message=f"unsymmetrize: {type(e).__name__}: {e}")


async def symmetrize_one(
    *,
    node: Node,
    cut_plane: Literal["xy", "xz", "yz"],
    keep_positive: bool,
    runs_dir: Path,
    run_id: str,
    version: str,
) -> None:
    """Mirror a generated object's served mesh across `cut_plane`, keeping the
    `keep_positive` half — the symmetrize counterpart to `unsymmetrize_one`. The
    plane + direction come straight from the caller (the client control), so this
    makes NO symmetry LLM decision and reads NO prior symmetry log to pick them. No
    Nano-Banana and no mesh backend: reload the on-disk raw (always the pristine,
    un-mirrored Trellis output, so re-mirroring is idempotent regardless of the
    current served state), apply the mirror, rescale into the node's bbox, and
    re-optimize the served twin. Pins the symmetry decision to `cut_plane` and lets
    `apply_symmetrize` record `symmetry.applied` (carrying `keep_positive`) so a
    resume / regeneration / prefab-reuse replays the same mirror and direction. A
    missing raw (e.g. a prefab reuse with no own raw) is logged and skipped — the
    caller symmetrizes the canonical and propagates. Runs under the node lock."""
    raw_dir, opt_dir = generated_dirs(runs_dir, run_id, version)
    src_raw = raw_dir / f"{node.id}.raw.glb"
    async with node_lock(run_id, version, node.id):
        if not src_raw.exists():
            logging.log("symmetry.skip", id=node.id, reason="symmetrize: no raw mesh on disk")
            return
        logging.log("symmetry.decision", id=node.id, cut_plane=cut_plane, encapsulating=False)
        rescaled = raw_dir / f"{node.id}.glb"
        try:
            async with _MESH_IO:
                scene = await asyncio.to_thread(trimesh.load, src_raw)
                scene = await symmetry.apply_symmetrize(
                    scene, cut_plane=cut_plane, node_id=node.id, keep_positive=keep_positive,
                )
                placed = await asyncio.to_thread(
                    rescale_mesh_to_bbox, scene, node.bbox, orientation=node.orientation,
                )
                tmp = rescaled.with_name(f"{rescaled.name}.part")
                try:
                    await asyncio.to_thread(export_glb, placed, tmp)
                    os.replace(tmp, rescaled)
                finally:
                    tmp.unlink(missing_ok=True)
                del scene, placed
            await _optimize_asset(rescaled, opt_dir / f"{node.id}.glb")
            logging.log("symmetry.symmetrized", id=node.id, cut_plane=cut_plane, keep_positive=keep_positive)
        except Exception as e:  # noqa: BLE001
            logging.log("mesh.error", id=node.id, message=f"symmetrize: {type(e).__name__}: {e}")


async def reorient_one(
    *,
    node: Node,
    axis: Literal["x", "y", "z"],
    degrees: int,
    runs_dir: Path,
    run_id: str,
    version: str,
) -> None:
    """Change a generated object's "front view" — which face points along +Z in
    the raw, pre-transform mesh — by rotating its RAW mesh 90° (a `degrees`
    multiple of 90) about `axis`, then re-deriving its served + optimized twin.

    The raw is the pristine source every prefab reuse + every re-derivation reads,
    so the rotation is BAKED into it: reload the on-disk raw, rotate, write it back
    (atomically), then re-apply the object's current symmetry and rescale into its
    bbox + yaw. The caller (`_regen_worker`) then `propagate_reuses`, which re-
    derives every reuse from this same re-fronted raw — so the new front view lands
    identically across the whole prefab group, in both the optimized and unoptimized
    builds. No Nano-Banana, no mesh backend, so it's effectively instant. Runs under
    the node lock so it can't race a scene build / regen of the same node. A missing
    raw is logged and skipped (the caller passes the prefab CANONICAL, which always
    owns the group's raw; a reuse has none of its own)."""
    if degrees % 90 != 0:
        logging.log("mesh.error", id=node.id, message=f"reorient: degrees must be a multiple of 90, got {degrees}")
        return
    raw_dir, opt_dir = generated_dirs(runs_dir, run_id, version)
    src_raw = raw_dir / f"{node.id}.raw.glb"
    rescaled = raw_dir / f"{node.id}.glb"
    async with node_lock(run_id, version, node.id):
        if not src_raw.exists():
            logging.log("mesh.reorient_skip", id=node.id, reason="no raw mesh on disk")
            return
        # Re-apply the object's CURRENT symmetry when re-deriving, so re-fronting
        # composes with an existing mirror instead of silently dropping it.
        cut_plane, keep_positive = symmetry.applied_state(node.id)
        try:
            async with _MESH_IO:
                raw_scene = await asyncio.to_thread(trimesh.load, src_raw)
                rotated = await asyncio.to_thread(rotate_mesh, raw_scene, axis=axis, degrees=degrees)
                # Persist the re-fronted raw FIRST (atomic) — it is the new source
                # of truth for this object AND every reuse derived from it.
                tmp_raw = src_raw.with_name(f"{src_raw.name}.part")
                try:
                    await asyncio.to_thread(export_glb, rotated, tmp_raw)
                    os.replace(tmp_raw, src_raw)
                finally:
                    tmp_raw.unlink(missing_ok=True)
                # Re-derive the served mesh from the re-fronted raw.
                placed = await symmetry.apply_symmetrize(
                    rotated, cut_plane=cut_plane, node_id=node.id, keep_positive=keep_positive,
                )
                placed = await asyncio.to_thread(
                    rescale_mesh_to_bbox, placed, node.bbox, orientation=node.orientation,
                )
                tmp_rescaled = rescaled.with_name(f"{rescaled.name}.part")
                try:
                    await asyncio.to_thread(export_glb, placed, tmp_rescaled)
                    os.replace(tmp_rescaled, rescaled)
                finally:
                    tmp_rescaled.unlink(missing_ok=True)
                del raw_scene, rotated, placed
            await _optimize_asset(rescaled, opt_dir / f"{node.id}.glb")
            logging.log("mesh.reorient", id=node.id, axis=axis, degrees=degrees)
        except Exception as e:  # noqa: BLE001
            logging.log("mesh.error", id=node.id, message=f"reorient: {type(e).__name__}: {e}")


async def glassify_one(
    *,
    node: Node,
    runs_dir: Path,
    run_id: str,
    version: str,
) -> None:
    """Force the window/glass transparency transform onto ONE generated object's
    served mesh, bypassing the pipeline's keyword + symmetry gates. The per-node
    primitive behind the frontend "make glass transparent" control; `glassify_group`
    fans it across a prefab group.

    Unlike unsymmetrize/symmetrize/reorient (which re-derive from the pristine
    raw), this edits the object's already-rescaled served mesh
    (`objects-generated/<id>.glb`) IN PLACE: reload it, bake per-texel
    transparency into its base-color texture(s) (white -> near-clear via
    `glass.apply_alpha_from_white`), re-export, and re-optimize the served twin.
    Operating on the served mesh means it applies uniformly to a prefab CANONICAL
    and a REUSE alike (a reuse owns a served `<id>.glb` but no raw), needs no
    bbox/symmetry replay, and is idempotent (re-running re-finds the same white
    texels). It is NOT re-derivable from the raw, so it can't ride the generic
    `propagate_reuses` raw-replay (which would drop it) — `glassify_group` instead
    applies it to each member directly. A later regenerate / symmetrize / reorient
    / reset rebuilds from raw and drops the transparency. Runs under the node lock
    so it can't race a scene build / regen of the same node. A missing served mesh,
    or one with no white texels / no base-color texture, is logged and skipped."""
    raw_dir, opt_dir = generated_dirs(runs_dir, run_id, version)
    rescaled = raw_dir / f"{node.id}.glb"
    async with node_lock(run_id, version, node.id):
        if not rescaled.exists():
            logging.log("mesh.glass_skip", id=node.id, reason="no served mesh on disk")
            return
        try:
            async with _MESH_IO:
                scene = await asyncio.to_thread(trimesh.load, rescaled)
                stats = await asyncio.to_thread(glass.apply_alpha_from_white, scene)
                if stats is None:
                    logging.log(
                        "mesh.glass_skip", id=node.id,
                        reason="no white texels or no base-color texture",
                    )
                    del scene
                    return
                tmp = rescaled.with_name(f"{rescaled.name}.part")
                try:
                    await asyncio.to_thread(export_glb, scene, tmp)
                    os.replace(tmp, rescaled)
                finally:
                    tmp.unlink(missing_ok=True)
                del scene
            await _optimize_asset(rescaled, opt_dir / f"{node.id}.glb")
            logging.log("mesh.glass", id=node.id, forced=True, **stats)
        except Exception as e:  # noqa: BLE001
            logging.log("mesh.error", id=node.id, message=f"glassify: {type(e).__name__}: {e}")


async def glassify_group(
    *,
    nodes: list[Node],
    runs_dir: Path,
    run_id: str,
    version: str,
) -> None:
    """Force the glass-transparency transform onto every node in a prefab group
    (the canonical + its reuses), each on its OWN served mesh via `glassify_one`.

    Glass is a per-mesh texture edit, not re-derivable from the shared raw, so the
    group can't be propagated through `propagate_reuses`' raw replay (it would drop
    the transparency). Instead each member is transformed directly — they share the
    canonical's geometry + texture, so the same white texels are cut on each. Runs
    members concurrently; each `glassify_one` takes its own node lock."""
    coros = [
        glassify_one(node=n, runs_dir=runs_dir, run_id=run_id, version=version)
        for n in nodes
    ]
    if coros:
        await asyncio.gather(*coros, return_exceptions=True)


async def reset_from_raw_one(
    *,
    node: Node,
    runs_dir: Path,
    run_id: str,
    version: str,
) -> None:
    """Rebuild a generated object's served mesh from its pristine raw Trellis
    output, discarding any in-place served-mesh edit (notably a forced glass
    transparency): reload the raw, re-apply the object's CURRENT symmetry decision,
    rescale into its bbox + orientation, and re-optimize the served twin. No
    Nano-Banana, no mesh backend — effectively instant.

    Unlike `unsymmetrize_one` (which forces symmetry OFF and pins it), this
    PRESERVES the existing symmetry/orientation — it reverts only texture/served-
    level changes, restoring the mesh exactly as the pipeline produced it before
    the edit. A missing raw (e.g. a prefab reuse, which owns no raw) is logged and
    skipped — the caller resets the canonical and re-derives reuses via
    `propagate_reuses`. Runs under the node lock."""
    raw_dir, opt_dir = generated_dirs(runs_dir, run_id, version)
    src_raw = raw_dir / f"{node.id}.raw.glb"
    rescaled = raw_dir / f"{node.id}.glb"
    async with node_lock(run_id, version, node.id):
        if not src_raw.exists():
            logging.log("mesh.reset_skip", id=node.id, reason="no raw mesh on disk")
            return
        # Re-apply the object's CURRENT symmetry when re-deriving, so reset keeps
        # the mirror instead of silently dropping it.
        cut_plane, keep_positive = symmetry.applied_state(node.id)
        try:
            async with _MESH_IO:
                scene = await asyncio.to_thread(trimesh.load, src_raw)
                scene = await symmetry.apply_symmetrize(
                    scene, cut_plane=cut_plane, node_id=node.id, keep_positive=keep_positive,
                )
                placed = await asyncio.to_thread(
                    rescale_mesh_to_bbox, scene, node.bbox, orientation=node.orientation,
                )
                tmp = rescaled.with_name(f"{rescaled.name}.part")
                try:
                    await asyncio.to_thread(export_glb, placed, tmp)
                    os.replace(tmp, rescaled)
                finally:
                    tmp.unlink(missing_ok=True)
                del scene, placed
            await _optimize_asset(rescaled, opt_dir / f"{node.id}.glb")
            logging.log("mesh.reset", id=node.id)
        except Exception as e:  # noqa: BLE001
            logging.log("mesh.error", id=node.id, message=f"reset: {type(e).__name__}: {e}")


async def await_pending(run_id: str) -> None:
    """Block until every background mesh task for this run has finished.
    Errors inside individual tasks were logged + swallowed by `_generate_one`,
    so this gather only waits — it never raises."""
    tasks = _pending.pop(run_id, [])
    _admitted_ids.pop(run_id, None)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def cancel_pending(run_id: str) -> None:
    """Cancel any in-flight mesh tasks for this run. Called when the run
    itself is being torn down (cancellation or fatal error)."""
    _admitted_ids.pop(run_id, None)
    for t in _pending.pop(run_id, []):
        t.cancel()


def _next_object_cap() -> int | None:
    """Optional ceiling on how many `next_object` ROUNDS an anchor zone runs —
    each round being one `next_object` LLM call that proposes objects plus the
    single `object_bbox_batch` that places them (the anchor_decompose pass that
    precedes the loop is never counted).

    Read live from `STARSHOT_NEXT_OBJECT_CAP` so it can change between runs
    without a restart. Unset, non-numeric, or negative → UNCAPPED (loop until the
    model itself says the zone is complete); `0` → run no rounds at all; `N` →
    stop after N placing rounds. The cap is checked BEFORE issuing each call, so a
    capped loop never spends an extra `next_object` call just to read the model's
    stop signal — at the cap it moves straight to the next pipeline step."""
    raw = os.environ.get("STARSHOT_NEXT_OBJECT_CAP", "").strip()
    if not raw:
        return None
    try:
        cap = int(raw)
    except ValueError:
        return None
    return cap if cap >= 0 else None


async def run(
    *,
    zone: Node,
    runs_dir: Path,
    run_id: str,
    scenario: Literal["anchor", "encapsulating", "negative-space"],
    all_nodes: list[Node],
) -> None:
    specs = await _decompose_objects(
        zone=zone,
        scenario=scenario,
        all_nodes=all_nodes,
    )
    logging.log_once(
        "generation.decompose",
        match_fields=("zone", "scenario"),
        zone=zone.id,
        scenario=scenario,
        objects=[s.model_dump(mode="json") for s in specs],
    )

    if specs:
        placed = await _resolve_and_generate(
            specs=specs,
            zone=zone,
            all_nodes=all_nodes,
            scenario=scenario,
            runs_dir=runs_dir,
            run_id=run_id,
        )
        all_nodes.extend(placed)

    if scenario != "anchor":
        return

    # Resume: replay the anchor completion loop's already-committed object
    # decisions, re-solving each ROUND's objects together in one
    # `object_bbox_batch` exactly as they were placed live. Grouping by round
    # reproduces that call's prompt so the placement replays from the LLM cache
    # instead of re-billing each object in its own single-object call, and keeps
    # intra-round parent/relationship resolution intact. Then stop if the loop
    # had run to completion; otherwise fall through to fresh `next_object`
    # rounds.
    #
    # Optional next_object round cap (see `_next_object_cap`). Rounds already
    # committed — replayed just below on resume — count toward it, so a resumed
    # capped run can't overshoot the ceiling a prior process already reached.
    cap = _next_object_cap()
    rounds = 0
    for group in committed.next_object_rounds(zone.id):
        replayed = await _resolve_and_generate(
            specs=group,
            zone=zone,
            all_nodes=all_nodes,
            scenario="anchor",
            runs_dir=runs_dir,
            run_id=run_id,
        )
        all_nodes.extend(replayed)
        rounds += 1
    if committed.next_done(zone.id):
        return

    # The completion loop normally terminates on the model's `done`, but the
    # model can also get stuck re-proposing objects that can never be admitted
    # — e.g. one whose bbox the batch step omitted, so it was admitted into
    # `_admitted_ids` but never placed into `all_nodes`, and so never shows up
    # as already-present in the next_object context. _resolve_and_generate
    # silently dedups that repeat to nothing, so without a progress guard the
    # loop spins forever re-billing next_object. Track the ids attempted this
    # loop; a round that proposes only already-attempted ids means no progress
    # is possible — stop.
    #
    # The model proposes a LIST of objects per round; each accepted object is
    # committed as its own `generation.next` event. Those per-round blocks are
    # written contiguously, so a resume regroups them (see
    # `committed.next_object_rounds`) and re-solves each round in one
    # `object_bbox_batch` rather than one call per object.
    attempted: set[str] = set()
    while True:
        # Cap reached: stop WITHOUT issuing another `next_object` call — the
        # requirement is that a cap of N places objects across N rounds and then
        # moves on, never spending an extra call just to fetch the model's stop
        # signal. Recorded as a third terminal exit alongside `.done`/`.stuck` so
        # resume/rewind/fork treat the zone as complete (see `committed.next_done`)
        # rather than re-entering the loop.
        if cap is not None and rounds >= cap:
            logging.log_once(
                "generation.next.capped",
                match_fields=("zone",),
                zone=zone.id,
                cap=cap,
                rounds=rounds,
            )
            return
        done, objects = await _next_object_batch(
            zone=zone,
            all_nodes=all_nodes,
        )
        if done or not objects:
            logging.log_once(
                "generation.next.done",
                match_fields=("zone",),
                zone=zone.id,
            )
            return
        fresh = [o for o in objects if o.id not in attempted]
        if not fresh:
            logging.log_once(
                "generation.next.stuck",
                match_fields=("zone",),
                zone=zone.id,
                id=objects[0].id,
            )
            return
        for o in fresh:
            attempted.add(o.id)
            logging.log_once(
                "generation.next",
                match_fields=("zone", "id"),
                zone=zone.id,
                id=o.id,
                object=o.model_dump(mode="json"),
            )
        new_nodes = await _resolve_and_generate(
            specs=fresh,
            zone=zone,
            all_nodes=all_nodes,
            scenario="anchor",
            runs_dir=runs_dir,
            run_id=run_id,
        )
        all_nodes.extend(new_nodes)
        rounds += 1
