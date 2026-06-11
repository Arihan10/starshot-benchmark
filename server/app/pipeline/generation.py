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

Per scenario: decompose objects (LLM) -> validate relationships and retry
on failure (this is the only validating retry left in the pipeline) ->
resolve every object's bbox in a single batch LLM call (trusted, no
retry) -> spawn background Trellis 2 jobs that fan out via SSE events as
each mesh lands.

The anchor-loop's "are more objects needed?" step uses the same
relationship validator on the emitted specs. V3/V4 let that step propose
a LIST of objects per round (`batch_next_object`); V2 proposes one at a
time. Bounding-box resolution is batch in every version.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

import trimesh

from app.core import prompt_runtime
from app.core.types import BoundingBox, Node, ProxyShape
from app.pipeline import committed
from app.services import hunyuan, hunyuan_tencent, llm, nano_banana, prefabs, symmetry, threed
from app.utils import glb_place, logging
from app.utils.geometry import export_glb, rescale_mesh_to_bbox
from app.utils.topology import validate_parents, validate_referenced_ids

_USE_ASSET_LIBRARY = os.environ.get("USE_ASSET_LIBRARY", "false").lower() == "true"

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

# Versioned from-scratch generated builds. A cell can hold ANY number of
# independent generated versions of the SAME scene — identical bboxes / event
# log / structure, but freshly generated (or re-matched) assets per object —
# laid out under the cell as:
#   generated/<version>/objects-generated/            raw Trellis meshes
#   generated/<version>/objects-generated-optimized/  served twin (KTX2/Meshopt)
#   generated/<version>/events.generated.jsonl        per-version resumable log
# Each version's own log + dirs make it fully isolated: resume/cache lookups
# read the bound (per-version) log, so a fresh version regenerates from scratch
# instead of reusing another version's meshes. The asset-library build
# (objects/ + events.jsonl) is NOT versioned.
GENERATED_DIR = "generated"
GENERATED_RAW_SUBDIR = "objects-generated"
GENERATED_OPT_SUBDIR = "objects-generated-optimized"
GENERATED_EVENTS_NAME = "events.generated.jsonl"


def generated_version_root(runs_dir: Path, run_id: str) -> Path:
    """The parent dir holding every generated version of one cell."""
    return runs_dir / run_id / GENERATED_DIR


def generated_dirs(runs_dir: Path, run_id: str, version: str) -> tuple[Path, Path]:
    """(raw_dir, opt_dir) for one generated version of a cell."""
    base = generated_version_root(runs_dir, run_id) / version
    return base / GENERATED_RAW_SUBDIR, base / GENERATED_OPT_SUBDIR


def generated_events_path(runs_dir: Path, run_id: str, version: str) -> Path:
    """The per-version resumable event log (events.generated.jsonl)."""
    return generated_version_root(runs_dir, run_id) / version / GENERATED_EVENTS_NAME


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


def migrate_legacy_generated(runs_dir: Path, run_id: str) -> None:
    """Fold a pre-versioning generated build (objects-generated*/ +
    events.generated.jsonl sitting directly in the cell dir) into
    generated/1/, so existing data keeps rendering under the versioned
    layout. Idempotent — a no-op once a generated/ dir exists or there is
    nothing legacy to move."""
    cell = runs_dir / run_id
    root = generated_version_root(runs_dir, run_id)
    if root.exists():
        return
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
        if p.exists():
            shutil.move(str(p), str(dst / p.name))


def seed_generated_version_from(
    runs_dir: Path, run_id: str, source_version: str, target_version: str,
) -> int:
    """Seed `target_version` from `source_version`'s Nano-Banana images so a build
    of the target REUSES those images (issuing NO new Nano-Banana calls) while
    regenerating every mesh fresh on the mesh backend.

    Copies each per-object reference image (`<id>.png`) into the target's raw dir
    and writes the target's event log pre-populated with:
      * a `google.banana.done` record per copied image, its saved-path pointed at
        the target copy — this is exactly what makes `nano_banana.generate_resumable`
        short-circuit (it skips the API only when a `google.banana.done` exists in
        the bound log AND the saved file is present), so no image is re-generated;
      * the source's `prefab.match` events — so the SAME objects stay canonicals
        (which own an image) vs reuses (which need none); without them, prefab
        re-matching could promote a former reuse to a canonical that would then
        require a fresh Nano-Banana image, defeating the point;
      * the source's `symmetry.decision` events — so each mesh is mirrored exactly
        as its (reused) image was captured for.

    No mesh artifacts or mesh-completion (`*.done`) records are copied, so the mesh
    backend regenerates every object from the reused images. Returns the count of
    images seeded (0 when the source had none — the caller should treat that as a
    no-op error)."""
    src_raw, _ = generated_dirs(runs_dir, run_id, source_version)
    dst_raw, dst_opt = generated_dirs(runs_dir, run_id, target_version)
    dst_raw.mkdir(parents=True, exist_ok=True)
    dst_opt.mkdir(parents=True, exist_ok=True)

    seed: list[dict[str, Any]] = []
    # Carry the prefab grouping + symmetry decisions over verbatim, in source
    # order so the latest event per id still wins on lookup.
    src_log = generated_events_path(runs_dir, run_id, source_version)
    if src_log.exists():
        with src_log.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("kind") in ("prefab.match", "symmetry.decision"):
                    seed.append({k: v for k, v in event.items() if k != "index"})

    # Copy every reference image and synthesize its completion record pointed at
    # the target copy. Synthesizing (rather than copying the source's done event)
    # is robust to a missing done record; nano_banana always normalizes to PNG, so
    # the mime type is fixed.
    seeded = 0
    for png in sorted(src_raw.glob("*.png")):
        shutil.copy2(png, dst_raw / png.name)
        seed.append({
            "kind": "google.banana.done",
            "job_id": png.stem,
            "saved": str(dst_raw / png.name),
            "mime_type": "image/png",
        })
        seeded += 1

    dst_log = generated_events_path(runs_dir, run_id, target_version)
    with dst_log.open("w", encoding="utf-8") as f:
        for i, event in enumerate(seed):
            f.write(json.dumps({**event, "index": i}) + "\n")
    return seeded


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


RELATIONSHIP_RETRY_ATTEMPTS = 3


async def _decompose_objects_validated(
    *,
    zone: Node,
    scenario: Literal["anchor", "encapsulating", "negative-space"],
    all_nodes: list[Node],
) -> list[Any]:
    # Resume: if this (zone, scenario) pass already committed its object set
    # to the log, replay those specs verbatim (ids fixed) instead of asking
    # the LLM to re-decompose — which is exactly where new ids leak in.
    committed_specs = committed.object_specs(zone.id, scenario)
    if committed_specs is not None:
        return committed_specs
    prior_attempts: list[tuple[list[Any], str]] = []
    existing_ids = {n.id for n in all_nodes}
    specs: list[Any] = []
    for attempt in range(RELATIONSHIP_RETRY_ATTEMPTS):
        p = prompt_runtime.current()
        if scenario == "anchor":
            system = p.SYSTEM_ANCHOR_DECOMP
            user = p.render_anchor_decomp(
                zone_id=zone.id,
                zone_prompt=zone.prompt,
                zone_plan=zone.plan,
                nodes=all_nodes,
                prior_attempts=prior_attempts,
            )
        elif scenario == "encapsulating":
            system = p.SYSTEM_ENCAPSULATING_DECOMP
            user = p.render_encapsulating_decomp(
                zone_id=zone.id,
                zone_prompt=zone.prompt,
                zone_plan=zone.plan,
                nodes=all_nodes,
                prior_attempts=prior_attempts,
            )
        else:
            system = p.SYSTEM_NEGATIVE_SPACE_DECOMP
            user = p.render_negative_space_decomp(
                zone_id=zone.id,
                zone_prompt=zone.prompt,
                zone_plan=zone.plan,
                nodes=all_nodes,
                prior_attempts=prior_attempts,
            )
        out = await llm.call_llm(
            system=system,
            user=user,
            output_schema=p.ObjectDecompOutput,
            node_id=zone.id,
            step=f"{scenario.replace('-', '_')}_decompose",
        )
        if scenario == "encapsulating" and not out.bounding_required:
            logging.log_once(
                "generation.decompose.no_bounding",
                match_fields=("zone",),
                zone=zone.id,
                emitted=[s.model_dump() for s in out.objects],
            )
            return []
        specs = list(out.objects)
        try:
            validate_referenced_ids(specs, parent_id=zone.id, existing_ids=existing_ids)
            return specs
        except ValueError as e:
            reason = str(e)
            logging.log(
                "generation.decompose.retry",
                zone=zone.id,
                attempt=attempt,
                reason=reason,
                emitted=[s.model_dump() for s in specs],
            )
            # Feed this failure back into the next prompt so the LLM has
            # something to react to instead of re-emitting the same invalid
            # set. After exhausting attempts we fall through to the
            # accept_invalid branch below.
            prior_attempts.append((specs, reason))
    # Retries exhausted. Unresolvable parents are a hard fail (orphaned
    # objects would ship a mesh yet be invisible to every later step);
    # secondary referenced_ids stay advisory and are accepted with a log.
    validate_parents(specs, parent_id=zone.id, existing_ids=existing_ids)
    logging.log_once(
        "generation.decompose.accept_invalid",
        match_fields=("zone",),
        zone=zone.id,
        reason=prior_attempts[-1][1] if prior_attempts else "",
    )
    return specs


async def _next_object_validated(
    *,
    zone: Node,
    all_nodes: list[Node],
) -> Any:
    prior_attempts: list[tuple[Any, str]] = []
    existing_ids = {n.id for n in all_nodes}
    decision: Any | None = None
    for attempt in range(RELATIONSHIP_RETRY_ATTEMPTS):
        p = prompt_runtime.current()
        decision = await llm.call_llm(
            system=p.SYSTEM_NEXT_OBJECT,
            user=p.render_next_object(
                zone_id=zone.id,
                zone_prompt=zone.prompt,
                nodes=all_nodes,
                prior_attempts=prior_attempts,
            ),
            output_schema=p.NextObjectOutput,
            node_id=zone.id,
            step="next_object",
        )
        if decision.done or decision.object is None:
            return decision
        try:
            validate_referenced_ids(
                [decision.object],
                parent_id=zone.id,
                existing_ids=existing_ids,
            )
            return decision
        except ValueError as e:
            reason = str(e)
            logging.log(
                "generation.next.retry",
                zone=zone.id,
                attempt=attempt,
                reason=reason,
                emitted=decision.object.model_dump(),
            )
            prior_attempts.append((decision.object, reason))
    assert decision is not None
    # Retries exhausted. An unresolvable parent on the emitted object is a
    # hard fail; a dangling secondary referenced_id is accepted with a log.
    if decision.object is not None:
        validate_parents([decision.object], parent_id=zone.id, existing_ids=existing_ids)
    logging.log_once(
        "generation.next.accept_invalid",
        match_fields=("zone",),
        zone=zone.id,
        reason=prior_attempts[-1][1] if prior_attempts else "",
    )
    return decision


async def _next_object_batch_validated(
    *,
    zone: Node,
    all_nodes: list[Node],
) -> tuple[bool, list[Any]]:
    """V3/V4 anchor-completion decision: the model proposes a LIST of objects
    per round (or signals done). Runs the same relationship-validator retry as
    the single-object path, applied to the whole proposed batch. Returns
    `(done, objects)`; `objects` is empty when done."""
    prior_attempts: list[tuple[list[Any], str]] = []
    existing_ids = {n.id for n in all_nodes}
    objects: list[Any] = []
    for attempt in range(RELATIONSHIP_RETRY_ATTEMPTS):
        p = prompt_runtime.current()
        decision = await llm.call_llm(
            system=p.SYSTEM_NEXT_OBJECT,
            user=p.render_next_object(
                zone_id=zone.id,
                zone_prompt=zone.prompt,
                zone_plan=zone.plan,
                nodes=all_nodes,
                prior_attempts=prior_attempts,
            ),
            output_schema=p.NextObjectOutput,
            node_id=zone.id,
            step="next_object",
        )
        objects = list(decision.objects)
        if decision.done or not objects:
            return True, []
        try:
            validate_referenced_ids(objects, parent_id=zone.id, existing_ids=existing_ids)
            return False, objects
        except ValueError as e:
            reason = str(e)
            logging.log(
                "generation.next.retry",
                zone=zone.id,
                attempt=attempt,
                reason=reason,
                emitted=[o.model_dump() for o in objects],
            )
            prior_attempts.append((objects, reason))
    # Retries exhausted. Unresolvable parents are a hard fail; dangling
    # secondary referenced_ids are accepted with a log.
    validate_parents(objects, parent_id=zone.id, existing_ids=existing_ids)
    logging.log_once(
        "generation.next.accept_invalid",
        match_fields=("zone",),
        zone=zone.id,
        reason=prior_attempts[-1][1] if prior_attempts else "",
    )
    return False, objects


async def _resolve_object_bboxes_batch(
    *,
    specs: list[Any],
    zone: Node,
    all_nodes: list[Node],
) -> dict[str, BoundingBox]:
    """Place every object in `specs` in ONE batch LLM call (the V2 strategy).
    Returns `{id: world-frame bbox}`. Objects already committed (resume) keep
    their world position and the LLM is skipped entirely when all are."""
    committed_bboxes = {s.id: committed.bbox(s.id) for s in specs}
    if all(b is not None for b in committed_bboxes.values()):
        return {sid: b for sid, b in committed_bboxes.items() if b is not None}
    bbox_by_id = {n.id: n.bbox for n in all_nodes}
    p = prompt_runtime.current()
    out = await llm.call_llm(
        system=p.SYSTEM_OBJECT_BBOX_BATCH,
        user=p.render_object_bbox_batch(
            zone_id=zone.id,
            zone_prompt=zone.prompt,
            zone_plan=zone.plan,
            zone_bbox=zone.bbox,
            objects=specs,
            nodes=all_nodes,
        ),
        output_schema=p.BboxBatchOutput,
        node_id=zone.id,
        step="object_bbox_batch",
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
    return bboxes


async def _next_object_batch_validated(
    *,
    zone: Node,
    all_nodes: list[Node],
) -> tuple[bool, list[Any]]:
    """V3/V4 anchor-completion decision: the model proposes a LIST of objects
    per round (or signals done). Runs the same relationship-validator retry as
    the single-object path, applied to the whole proposed batch. Returns
    `(done, objects)`; `objects` is empty when done."""
    prior_attempts: list[tuple[list[Any], str]] = []
    existing_ids = {n.id for n in all_nodes}
    objects: list[Any] = []
    for attempt in range(RELATIONSHIP_RETRY_ATTEMPTS):
        p = prompt_runtime.current()
        decision = await llm.call_llm(
            system=p.SYSTEM_NEXT_OBJECT,
            user=p.render_next_object(
                zone_id=zone.id,
                zone_prompt=zone.prompt,
                zone_plan=zone.plan,
                nodes=all_nodes,
                prior_attempts=prior_attempts,
            ),
            output_schema=p.NextObjectOutput,
            node_id=zone.id,
            step="next_object",
        )
        objects = list(decision.objects)
        if decision.done or not objects:
            return True, []
        try:
            validate_referenced_ids(objects, parent_id=zone.id, existing_ids=existing_ids)
            return False, objects
        except ValueError as e:
            reason = str(e)
            logging.log(
                "generation.next.retry",
                zone=zone.id,
                attempt=attempt,
                reason=reason,
                emitted=[o.model_dump() for o in objects],
            )
            prior_attempts.append((objects, reason))
    # Retries exhausted. Unresolvable parents are a hard fail; dangling
    # secondary referenced_ids are accepted with a log.
    validate_parents(objects, parent_id=zone.id, existing_ids=existing_ids)
    logging.log_once(
        "generation.next.accept_invalid",
        match_fields=("zone",),
        zone=zone.id,
        reason=prior_attempts[-1][1] if prior_attempts else "",
    )
    return False, objects


async def _resolve_object_bboxes_batch(
    *,
    specs: list[Any],
    zone: Node,
    all_nodes: list[Node],
) -> dict[str, BoundingBox]:
    """Place every object in `specs` in ONE batch LLM call (the V2 strategy).
    Returns `{id: world-frame bbox}`. Objects already committed (resume) keep
    their world position and the LLM is skipped entirely when all are."""
    committed_bboxes = {s.id: committed.bbox(s.id) for s in specs}
    if all(b is not None for b in committed_bboxes.values()):
        return {sid: b for sid, b in committed_bboxes.items() if b is not None}
    bbox_by_id = {n.id: n.bbox for n in all_nodes}
    p = prompt_runtime.current()
    out = await llm.call_llm(
        system=p.SYSTEM_OBJECT_BBOX_BATCH,
        user=p.render_object_bbox_batch(
            zone_id=zone.id,
            zone_prompt=zone.prompt,
            zone_plan=zone.plan,
            zone_bbox=zone.bbox,
            objects=specs,
            nodes=all_nodes,
        ),
        output_schema=p.BboxBatchOutput,
        node_id=zone.id,
        step="object_bbox_batch",
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
    return bboxes


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
        if s.id in seen_in_call or s.id in admitted or s.id in placed_ids:
            logging.log(
                "generation.dedup_drop",
                zone=zone.id,
                scenario=scenario,
                id=s.id,
                reason=(
                    "duplicate_in_call"
                    if s.id in seen_in_call
                    else "already_placed"
                    if s.id in placed_ids
                    else "already_in_flight"
                ),
            )
            continue
        seen_in_call.add(s.id)
        deduped.append(s)
    if not deduped:
        return []
    admitted.update(seen_in_call)
    specs = deduped

    # Resolve every object's world-frame bbox in one batch LLM call (honoring
    # committed bboxes for resume). The anchor completion loop calls this with
    # the objects it proposed that round — one for V2, a list for V3/V4.
    bboxes = await _resolve_object_bboxes_batch(
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
            orientation=spec.orientation,
        )

    if _USE_ASSET_LIBRARY:
        # Library mode defers realization: place the nodes now (bboxes already
        # emitted above) but match + bake them in ONE whole-scene pass after the
        # divider finishes (`realize_library_scene`), so a shared prefab grouping
        # can dedup identical objects down to a single library match.
        return [
            Node(
                id=spec.id,
                prompt=spec.prompt,
                bbox=bboxes[spec.id],
                proxy_shape=spec.proxy_shape,
                orientation=spec.orientation,
                placement=spec.placement,
                referenced_ids=list(spec.referenced_ids),
                parent_id=spec.parent,
                parent_kind=spec.parent_kind,
                parent_region=zone.id,
            )
            for spec in specs
        ]

    # Every object (anchor, completion, encapsulating alike) goes through
    # the image-prompt rewrite: Nano Banana needs an isolated studio
    # reference shot — any environmental context bleeds into the mesh.
    #
    # The "prior subjects" context fed to the LLM is the bare subject
    # phrases of already-placed nodes (Node.prompt), NOT their full
    # Nano-Banana directives (Node.image_prompt). Leaking the wrapper
    # boilerplate would just teach the LLM to echo "Generate a direct,
    # perfect orthographic..." back into every new phrase.
    committed_subjects = [n.prompt for n in all_nodes if n.mesh_url is not None]
    resolved: list[Node] = []
    for spec in specs:
        bbox = bboxes[spec.id]
        parent_id = spec.parent
        # bbox already emitted upfront above.
        prior_subjects = committed_subjects + [r.prompt for r in resolved]
        encapsulating = scenario == "encapsulating"
        cut_plane = await symmetry.resolve_cut_plane(
            prompt=spec.prompt,
            node_id=spec.id,
            encapsulating=encapsulating,
        )
        prior_subjects = committed_subjects + [r.prompt for r in resolved]
        encapsulating = scenario == "encapsulating"
        cut_plane = await symmetry.resolve_cut_plane(
            prompt=spec.prompt,
            node_id=spec.id,
            encapsulating=encapsulating,
        )
        view = symmetry.image_view_for(
            cut_plane=cut_plane, encapsulating=encapsulating,
        )
        subject_prompt, image_prompt = await _build_image_prompt(
            spec_id=spec.id,
            prompt=spec.prompt,
            bbox=bbox,
            proxy_shape=spec.proxy_shape,
            prior_prompts=prior_subjects,
            view=view,
        )
        resolved.append(
            Node(
                id=spec.id,
                prompt=subject_prompt,
                image_prompt=image_prompt,
                bbox=bbox,
                proxy_shape=spec.proxy_shape,
                orientation=spec.orientation,
                placement=spec.placement,
                referenced_ids=list(spec.referenced_ids),
                parent_id=parent_id,
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


async def _bake_library_asset(
    *, node: Node, library_id: str, objs_dir: Path, runs_dir: Path,
) -> None:
    """Bake one library asset into `node`'s slot: copy the reference image, place
    the (Meshopt/KTX2) GLB into the node's bbox + orientation as a node-graph
    transform, and emit the model event. Skips work when the placed GLB already
    exists (resume). Every node in a prefab group bakes the SAME `library_id` —
    the canonical and each reuse, each into its own slot."""
    from app.services import library

    path = objs_dir / f"{node.id}.glb"
    url = _artifact_url(runs_dir, path)
    if path.exists():
        logging.emit_model(node.id, artifact_kind="object", url=url)
        return

    asset = library.asset_path(library_id)
    ref_image = asset.with_suffix(".png")
    if ref_image.exists():
        dest_image = objs_dir / f"{node.id}.png"
        await asyncio.to_thread(shutil.copy2, ref_image, dest_image)
        logging.log(
            "image", id=node.id, url=_artifact_url(runs_dir, dest_image), prompt=node.prompt,
        )

    if not asset.exists():
        logging.log("library.asset_missing", id=node.id, library_id=library_id)
        return

    # The optimized asset is Meshopt/KTX2-compressed, so bake the placement as a
    # node-graph transform (preserving the compressed bytes) instead of decoding +
    # re-exporting through trimesh.
    bounds = library.asset_rotated_bounds(library_id, node.orientation)
    if bounds is not None:
        await asyncio.to_thread(
            glb_place.place_glb,
            src=asset, dst=path, bbox=node.bbox, orientation=node.orientation,
            rotated_min=bounds[0], rotated_max=bounds[1],
        )
    else:
        # Asset missing from the manifest: copy through unscaled so the placement
        # still renders rather than vanishing.
        await asyncio.to_thread(shutil.copyfile, asset, path)
        logging.log("library.bounds_missing", id=node.id, library_id=library_id)
    logging.emit_model(node.id, artifact_kind="object", url=url)


async def realize_library_scene(
    *, nodes: list[Node], runs_dir: Path, run_id: str,
) -> None:
    """Library-asset realization for a whole built scene (deferred batch). No-op
    unless USE_ASSET_LIBRARY. Runs the SCENE prefab grouping once, matches every
    prefab CANONICAL to a library asset CONCURRENTLY (one `library.match` per
    distinct object, not per object, bounded by `_library_match_slot`), then bakes
    that asset into the canonical's slot and every reuse's slot — reuses inherit
    the canonical's library_id, rescaled into their own bbox. Must run with the
    SCENE log bound; resumable (skips already-baked objects and replays logged
    matches)."""
    if not _USE_ASSET_LIBRARY:
        return
    from app.services import library

    concrete = [n for n in nodes if not n.is_zone]
    if not concrete:
        return
    decisions = await ensure_scene_prefab_groups(nodes=concrete, run_id=run_id)
    by_id = {n.id: n for n in concrete}
    objs_dir = runs_dir / run_id / "objects"
    objs_dir.mkdir(parents=True, exist_ok=True)

    # The distinct canonicals (reuses inherit their canonical's match), de-duped in
    # node order so a canonical leads its reuses.
    canonical_ids: list[str] = []
    seen: set[str] = set()
    for node in concrete:
        cid = decisions.get(node.id, "") or node.id
        if cid not in seen:
            seen.add(cid)
            canonical_ids.append(cid)

    # Match each canonical to a library asset. The matches are independent (each
    # canonical vs the static catalog), so fire them CONCURRENTLY — bounded by
    # `_library_match_slot`. Resume reuses any already-logged match.
    async def _resolve_canonical(cid: str) -> tuple[str, str]:
        decided = logging.find_event("library.match", id=cid)
        if decided is not None:
            return cid, str(decided.get("library_id") or "")
        canonical_node = by_id.get(cid)
        if canonical_node is None:
            return cid, ""
        async with _library_match_slot:
            lib_id = (await library.match(canonical_node.prompt)).library_id
        logging.log("library.match", id=cid, prompt=canonical_node.prompt, library_id=lib_id)
        return cid, lib_id

    library_id_by_canonical = dict(
        await asyncio.gather(*(_resolve_canonical(cid) for cid in canonical_ids))
    )

    # Bake every node into its slot; reuses inherit their canonical's library_id.
    for node in concrete:
        canonical_id = decisions.get(node.id, "") or node.id
        lib_id = library_id_by_canonical.get(canonical_id, "")
        if node.id != canonical_id and logging.find_event("library.match", id=node.id) is None:
            # Record the reuse's inherited match so the scene log is complete.
            logging.log(
                "library.match", id=node.id, prompt=node.prompt,
                library_id=lib_id, reuse_of=canonical_id,
            )
        await _bake_library_asset(
            node=node, library_id=lib_id, objs_dir=objs_dir, runs_dir=runs_dir,
        )


async def _build_image_prompt(
    *,
    spec_id: str,
    prompt: str,
    bbox: BoundingBox,
    proxy_shape: ProxyShape | None,
    prior_prompts: list[str],
    view: str = "front",
    include_dimensions: bool = True,
) -> tuple[str, str]:
    """Returns (subject_phrase, wrapped_image_prompt). The subject phrase is
    the LLM's bare noun phrase — what gets stored on Node.prompt and shown
    in context. The wrapped prompt is the full Nano-Banana studio-shot
    directive — used only at the image-generation boundary.

    Set include_dimensions=False for library generation where objects have
    no meaningful bbox — omits the dimension constraint from the image
    prompt so the model renders natural proportions."""
    p = prompt_runtime.current()
    dims = bbox.size if include_dimensions else None
    # Resume: reuse the committed subject phrase so a replayed object keeps
    # the exact prompt the rest of the scene already references downstream.
    subject = committed.image_subject(spec_id)
    if subject is not None:
        return subject, p.wrap_image_prompt(subject, proxy_shape, dims, view=view)
    out = await llm.call_llm(
        system=p.SYSTEM_IMAGE_PROMPT,
        user=p.render_image_prompt(
            prompt=prompt,
            bbox=bbox,
            proxy_shape=proxy_shape,
            prior_prompts=prior_prompts,
        ),
        output_schema=p.ImagePromptOutput,
        node_id=spec_id,
        step="image_prompt",
    )
    return out.prompt, p.wrap_image_prompt(out.prompt, proxy_shape, dims, view=view)


async def _refresh_node_image_prompt(node: Node) -> Node:
    """Re-resolve symmetry + wrap after a mesh retry deleted the reference image.
    Replays `symmetry.decision` from the log when present."""
    cut_plane = await symmetry.resolve_cut_plane(
        prompt=node.prompt, node_id=node.id,
    )
    view = symmetry.image_view_for(cut_plane=cut_plane)
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

# Per-(run_id, version) map of per-node asyncio.Locks. The from-scratch
# whole-scene generate (`generate_assets`) and standalone per-asset regeneration
# (`regenerate_one` / `propagate_reuses`) may now run CONCURRENTLY on the same
# version; this serializes them whenever they would write the SAME node's files
# (raw / rescaled / optimized GLB + reference image), so a node is never built by
# two writers at once. Distinct nodes never contend, so the common case —
# regenerating already-built assets while the scene build resumes the missing
# ones — stays fully parallel. A reuse's read of its canonical's raw is lock-free
# and kept safe instead by atomic (temp + replace) writes.
_node_locks: dict[tuple[str, str], dict[str, asyncio.Lock]] = {}


def node_lock(run_id: str, version: str, node_id: str) -> asyncio.Lock:
    """The build lock for one (cell, version, node), created on first use. Shared
    by the whole-scene generate and per-asset regeneration so the two serialize
    only when they target the same node."""
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
) -> None:
    try:
        # Nano Banana sees the wrapped studio-shot directive; node.prompt
        # stays the bare subject phrase for everything else.
        banana_prompt = node.image_prompt or node.prompt
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
                return
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
        if not had_image:
            logging.log(
                "image",
                id=node.id,
                url=_artifact_url(runs_dir, image_path),
                prompt=node.prompt,
            )
        # generate_mesh writes `raw` on a fresh run but returns a *cached* path
        # on a resumable hit (which may not be `raw` when the bound log was
        # hydrated from another build). Load whatever it actually produced.
        produced = await MESH_BACKENDS[backend](
            image.image_bytes,
            output_path=raw,
            job_id=node.id,
            image_mime=image.mime_type,
            bbox=node.bbox,
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
    except Exception as e:
        logging.log("mesh.error", id=node.id, message=f"{type(e).__name__}: {e}")


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
        cut_plane: Literal["none", "xy", "xz"] = "none"
        keep_positive: bool | None = None
        applied = logging.find_event("symmetry.applied", id=source_id)
        if applied is not None:
            raw_cp = applied.get("cut_plane")
            if raw_cp in ("none", "xy", "xz"):
                cut_plane = raw_cp  # type: ignore[assignment]
            # Carry the canonical's kept-half so a non-default direction (set via
            # the symmetrize control) mirrors the reuse identically. Absent on
            # older logs -> None -> the plane's default half.
            raw_keep = applied.get("keep_positive")
            if isinstance(raw_keep, bool):
                keep_positive = raw_keep
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
        await _optimize_asset(rescaled, opt_dir / f"{node.id}.glb")
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


# One lock per scene (run_id) so concurrent version builds compute the shared
# prefab grouping once instead of racing to re-sweep + double-log it.
_scene_prefab_locks: dict[str, asyncio.Lock] = {}


async def ensure_scene_prefab_groups(*, nodes: list[Node], run_id: str) -> dict[str, str]:
    """Compute the scene-wide prefab grouping over `nodes` ONCE and persist it as
    `prefab.match` events in the BOUND log — which must be the SCENE log
    (`events.jsonl`), since the grouping is a per-scene artifact shared by every
    generated version, the library build, and regen.

    Seed-and-sweep: the first undecided node is a canonical "seed"; a flash-lite
    call names every remaining node that is the SAME object (each reuses the
    seed), then the next still-undecided node seeds the following group. Idempotent
    — nodes already carrying a logged `prefab.match` are honored and never
    re-swept, so a later version build, the library realize, or a resume is free.
    Returns node_id -> reuse_id ("" = canonical)."""
    lock = _scene_prefab_locks.setdefault(run_id, asyncio.Lock())
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


# TEMP (prefab-match prompt iteration): fresh, isolated grouping driving the
# `/prefab-test` route + the client's "Prefab Test" button. Remove all three together.
async def dry_run_prefab_groups(*, nodes: list[Node]) -> dict[str, str]:
    """Fresh prefab grouping that reads NO prior prefab.match and persists none —
    the same seed-sweep as `ensure_scene_prefab_groups`, but for iterating on the
    match prompt. The caller binds a throwaway log so each run re-calls the LLM
    with the current prompt. Returns node_id -> reuse_id ("" = canonical)."""
    decisions: dict[str, str] = {}
    for node in nodes:
        if node.id in decisions:
            continue
        decisions[node.id] = ""
        candidates = [(n.id, n.prompt, n.bbox) for n in nodes if n.id not in decisions]
        for dup_id in await prefabs.match_duplicates(
            seed_id=node.id,
            seed_description=node.prompt,
            seed_bbox=node.bbox,
            candidates=candidates,
        ):
            decisions[dup_id] = node.id
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
    3.1, one job at a time). Writes into ONE generated `version` of the cell — any number
    coexist, each fully isolated by its own dirs + log. Two dirs under
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

    PREFAB REUSE: `decisions` (node_id -> reuse_id, "" = canonical) is the
    SCENE-level prefab grouping computed once by `ensure_scene_prefab_groups` and
    shared across every generated version + the library build. This gate only
    APPLIES it: a reuse skips Nano-Banana + Trellis entirely (its mesh is the
    canonical's raw Trellis output rescaled into its slot). The grouping is seeded
    into this version's log — honoring any per-version promotion already on disk (a
    reuse rebuilt standalone logs its own `prefab.match`) and never flipping a
    pre-built asset — so regen + the reuse-images fork keep resolving groups here.

    Every asset fans out at once into an uncapped queue; the Trellis in-flight cap
    (threed.GENERATE_CONCURRENCY — 100 live submits) plus threed's `_pace_submit`
    (~1s between Trellis `POST /generate`) govern how many actually reach Modal, so
    the batch ramps on instead of bursting into a 429 storm.

    Requires a bound SlotLog: `_generate_one`, `prefabs.match_duplicates`, and the
    nano_banana / threed services record their bookkeeping there. The caller
    binds a dedicated log (events.generated.jsonl) so none of this lands in the
    library build's event stream."""
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
                pr = prompt_runtime.current()
                image_prompt = pr.wrap_image_prompt(
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
            if (opt_dir / f"{node.id}.glb").exists():
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
        # lands identically posed. No Nano-Banana, no Trellis.
        await raw_ready[source_id].wait()
        async with node_lock(run_id, version, node.id):
            if (opt_dir / f"{node.id}.glb").exists():
                return
            await _rescale_reuse_from_raw(
                node,
                src_raw=raw_dir / f"{source_id}.raw.glb",
                raw_dir=raw_dir,
                opt_dir=opt_dir,
                source_id=source_id,
            )

    # Apply the scene grouping. Seed this version's log with each decision so regen
    # / resolve_group / the reuse-images fork read the grouping from here, while
    # honoring a decision already on disk (a per-version promotion) and never
    # flipping a pre-built asset into a reuse.
    tasks: list[asyncio.Task[None]] = []
    for node in nodes:
        done = (opt_dir / f"{node.id}.glb").exists()
        decided = logging.find_event("prefab.match", id=node.id)
        if decided is not None:
            reuse_id = str(decided.get("reuse_id") or "")
        else:
            reuse_id = "" if _built(node.id) else decisions.get(node.id, "")
            logging.log("prefab.match", id=node.id, reuse_id=reuse_id, description=node.prompt)

        is_reuse = bool(reuse_id) and reuse_id in canonical_ids
        if not is_reuse:
            # Canonical (a seed or a pre-built original); reuses rescale its raw mesh.
            canonical_ids.add(node.id)
            ev = raw_ready.setdefault(node.id, asyncio.Event())
            if done or _has_raw(node.id):
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
    (cell, version, node)'s build lock, so a concurrent whole-scene generate of
    the same version can never write this node's files at the same time. The
    library retry path passes version=None (its build has no concurrent gate).

    Awaitable core shared by the library single-mesh retry (`retry_node`, a
    detached + `_pending`-tracked wrapper) and standalone generated-asset
    regeneration (awaited directly under its own cell task)."""
    if version is None:
        await _rebuild_one(
            node=node, runs_dir=runs_dir, run_id=run_id,
            subdir=subdir, optimize=optimize, backend=backend, reuse_image=reuse_image,
        )
        return
    async with node_lock(run_id, version, node.id):
        await _rebuild_one(
            node=node, runs_dir=runs_dir, run_id=run_id,
            subdir=subdir, optimize=optimize, backend=backend, reuse_image=reuse_image,
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
) -> None:
    objs_dir = runs_dir / run_id / subdir
    objs_dir.mkdir(parents=True, exist_ok=True)
    raw = objs_dir / f"{node.id}.raw.glb"
    path = objs_dir / f"{node.id}.glb"
    image_stem = objs_dir / node.id
    # Drop only the prior MESH so the backend rebuilds it fresh. The reference
    # image is NEVER deleted here: a from-scratch rebuild re-rolls it via
    # `force_image` (which writes only on success, so a failed image call — e.g.
    # the API key being down — leaves the existing image intact instead of wiping
    # it and breaking a later from-image regen), and a from-image rebuild reuses it.
    for artifact in (raw, path):
        artifact.unlink(missing_ok=True)
    logging.log("mesh.retry", id=node.id, prompt=node.prompt, backend=backend)
    node = await _refresh_node_image_prompt(node)
    await _generate_one(
        node, raw=raw, path=path, image_stem=image_stem, runs_dir=runs_dir,
        backend=backend, reuse_image=reuse_image, force_image=not reuse_image,
    )
    if optimize and path.exists():
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

    async def _one(node: Node) -> None:
        # Lock per reuse so a concurrent whole-scene generate (or another regen)
        # of the same node can't write its files at the same time.
        async with node_lock(run_id, version, node.id):
            await _rescale_reuse_from_raw(
                node, src_raw=src_raw, raw_dir=raw_dir, opt_dir=opt_dir, source_id=canonical_id,
            )

    coros = [_one(node) for node in reuses]
    if coros:
        await asyncio.gather(*coros, return_exceptions=True)


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
    cut_plane: Literal["xy", "xz"],
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


async def run(
    *,
    zone: Node,
    runs_dir: Path,
    run_id: str,
    scenario: Literal["anchor", "encapsulating", "negative-space"],
    all_nodes: list[Node],
    batch_next_object: bool = False,
) -> None:
    specs = await _decompose_objects_validated(
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
    # decisions in order (re-resolving from the log, re-spawning only the
    # meshes still missing), then stop if the loop had run to completion.
    # Otherwise fall through and continue from the frontier with fresh
    # `next_object` decisions.
    for spec in committed.next_object_specs(zone.id):
        replayed = await _resolve_and_generate(
            specs=[spec],
            zone=zone,
            all_nodes=all_nodes,
            scenario="anchor",
            runs_dir=runs_dir,
            run_id=run_id,
        )
        all_nodes.extend(replayed)
    if committed.next_done(zone.id):
        return

    # The completion loop normally terminates on the model's `done`, but the
    # model can also get stuck re-proposing objects that can never be admitted
    # — e.g. one whose bbox the batch step omitted, so it was admitted into
    # `_admitted_ids` but never placed into `all_nodes`, and so never shows up
    # as already-present in the next_object context. _resolve_and_generate
    # dedups that repeat to nothing (`generation.dedup_drop`), so without a
    # progress guard the loop spins forever re-billing next_object. Track the
    # ids attempted this loop; a round that proposes only already-attempted ids
    # means no progress is possible — stop.
    #
    # V3/V4 propose a LIST of objects per round (`batch_next_object`); V2
    # proposes one. Both feed the same frontier loop — V2 just yields a
    # length-1 batch. Each accepted object is committed as its own
    # `generation.next` event so resume replays them one at a time regardless
    # of how they were proposed.
    attempted: set[str] = set()
    while True:
        if batch_next_object:
            done, objects = await _next_object_batch_validated(
                zone=zone,
                all_nodes=all_nodes,
            )
        else:
            decision = await _next_object_validated(
                zone=zone,
                all_nodes=all_nodes,
            )
            done = decision.done or decision.object is None
            objects = [] if done else [decision.object]
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
