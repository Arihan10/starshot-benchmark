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
batch-resolve every object's bbox in a single LLM call (trusted, no
retry) -> spawn background Trellis 2 jobs that fan out via SSE events
as each mesh lands.

The anchor-loop's "is another object needed?" step uses the same
relationship validator on the single emitted spec.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any, Literal

import trimesh

from app.core import prompt_runtime
from app.core.types import BoundingBox, Node, ProxyShape
from app.pipeline import committed
from app.services import llm, nano_banana, prefabs, threed
from app.utils import glb_place, logging
from app.utils.geometry import rescale_mesh_to_bbox
from app.utils.topology import validate_parents, validate_referenced_ids

_USE_ASSET_LIBRARY = os.environ.get("USE_ASSET_LIBRARY", "false").lower() == "true"

# Guards the trimesh load -> rescale -> export block. API calls and GLB
# downloads stay fully parallel across slots; only the RAM-heavy mesh
# decode is serialized so concurrent slots don't stack Pillow-decoded
# texture buffers and trip the OOM killer.
_MESH_IO = asyncio.Semaphore(1)

# How many assets the generate gate works on at once (process-global, shared
# across simultaneous generates). The burst of submits is smoothed by threed's
# `_pace_submit` (≥1s between Trellis `POST /generate`), so this just bounds how
# many pipelines are live concurrently; matches threed.GENERATE_CONCURRENCY so a
# whole scene can be in flight without exceeding the Trellis in-flight cap.
_GENERATE_FANOUT = asyncio.Semaphore(100)

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
        if scenario == "encapsulating":
            specs = [s.model_copy(update={"parent": zone.id}) for s in specs]
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


async def _resolve_and_generate(
    *,
    specs: list[Any],
    zone: Node,
    all_nodes: list[Node],
    scenario: Literal["anchor", "encapsulating", "negative-space"],
    runs_dir: Path,
    run_id: str,
) -> list[Node]:
    if scenario == "encapsulating":
        specs = [s.model_copy(update={"parent": zone.id}) for s in specs]

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

    # Resume: objects already placed (a committed `bbox` event) keep their
    # exact world position. Skip the LLM when every object is committed;
    # otherwise resolve the batch and overwrite the committed ones so only
    # never-placed objects take a fresh assignment.
    committed_bboxes = {s.id: committed.bbox(s.id) for s in specs}
    if all(b is not None for b in committed_bboxes.values()):
        bboxes: dict[str, BoundingBox] = {
            sid: b for sid, b in committed_bboxes.items() if b is not None
        }
    else:
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
        )
        # LLM emits each object's bbox in that object's parent's local
        # frame. Convert to world coordinates per-object. Handle
        # intra-batch parents (spec B parents to spec A in same batch) via
        # topological resolution order.
        spec_parent = {s.id: s.parent for s in specs}
        assignments_by_id = {a.id: a.bbox for a in out.assignments}
        bboxes = {}
        # Resolve in passes: each pass converts objects whose parent bbox
        # is already known. Terminates when all are resolved or no progress
        # (cycle — fall back to zone frame).
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

    if _USE_ASSET_LIBRARY:
        return await _match_library_assets(
            specs=specs,
            bboxes=bboxes,
            scenario=scenario,
            runs_dir=runs_dir,
            run_id=run_id,
        )

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
        logging.emit_bbox(
            spec.id,
            bbox,
            parent_id=parent_id,
            prompt=spec.prompt,
            kind="frame" if scenario == "encapsulating" else "object",
            proxy_shape=spec.proxy_shape,
            orientation=spec.orientation,
        )
        prior_subjects = committed_subjects + [r.prompt for r in resolved]
        view = "three-quarter" if scenario == "encapsulating" else "front"
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
    scenario: Literal["anchor", "encapsulating", "negative-space"],
    runs_dir: Path,
    run_id: str,
) -> list[Node]:
    from app.services import library

    objs_dir = runs_dir / run_id / "objects"
    objs_dir.mkdir(parents=True, exist_ok=True)

    resolved: list[Node] = []
    for spec in specs:
        bbox = bboxes[spec.id]
        path = objs_dir / f"{spec.id}.glb"
        url = _artifact_url(runs_dir, path)

        if path.exists():
            resolved.append(
                Node(
                    id=spec.id,
                    prompt=spec.prompt,
                    bbox=bbox,
                    proxy_shape=spec.proxy_shape,
                    orientation=spec.orientation,
                    placement=spec.placement,
                    referenced_ids=list(spec.referenced_ids),
                    parent_id=spec.parent,
                    parent_kind=spec.parent_kind,
                    mesh_url=url,
                )
            )
            continue

        logging.emit_bbox(
            spec.id,
            bbox,
            parent_id=spec.parent,
            prompt=spec.prompt,
            kind="frame" if scenario == "encapsulating" else "object",
            proxy_shape=spec.proxy_shape,
            orientation=spec.orientation,
        )

        match = await library.match(spec.prompt)
        asset = library.asset_path(match.library_id)

        logging.log(
            "library.match",
            id=spec.id,
            prompt=spec.prompt,
            library_id=match.library_id,
        )

        ref_image = asset.with_suffix(".png")
        if ref_image.exists():
            dest_image = objs_dir / f"{spec.id}.png"
            await asyncio.to_thread(shutil.copy2, ref_image, dest_image)
            logging.log(
                "image",
                id=spec.id,
                url=_artifact_url(runs_dir, dest_image),
                prompt=spec.prompt,
            )

        if asset.exists():
            # The optimized asset is Meshopt/KTX2-compressed, so we bake the
            # placement as a node-graph transform (preserving the compressed
            # bytes) instead of decoding + re-exporting through trimesh.
            bounds = library.asset_rotated_bounds(match.library_id, spec.orientation)
            if bounds is not None:
                await asyncio.to_thread(
                    glb_place.place_glb,
                    src=asset,
                    dst=path,
                    bbox=bbox,
                    orientation=spec.orientation,
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

        resolved.append(
            Node(
                id=spec.id,
                prompt=spec.prompt,
                bbox=bbox,
                proxy_shape=spec.proxy_shape,
                orientation=spec.orientation,
                placement=spec.placement,
                referenced_ids=list(spec.referenced_ids),
                parent_id=spec.parent,
                parent_kind=spec.parent_kind,
                mesh_url=url,
            )
        )

    return resolved


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


_pending: dict[str, list[asyncio.Task[None]]] = {}

# Run-scoped set of every id that has been admitted into _resolve_and_generate
# for this run. Populated *synchronously* before the bbox-resolution LLM call,
# so concurrent specs (within or across decomposition steps) can't both pass
# the dedup check and race into Banana+Trellis. The per-stage `.done`
# completion cache catches *most* repeats, but two specs that submit before
# either has logged a `trellis.done` will both miss the cache and
# double-bill — this guard closes that window.
_admitted_ids: dict[str, set[str]] = {}


async def _generate_one(
    node: Node,
    *,
    raw: Path,
    path: Path,
    image_stem: Path,
    runs_dir: Path,
) -> None:
    try:
        # Nano Banana sees the wrapped studio-shot directive; node.prompt
        # stays the bare subject phrase for everything else.
        banana_prompt = node.image_prompt or node.prompt
        image_path = image_stem.parent / f"{image_stem.name}.png"
        had_image = image_path.exists() and logging.find_event("image", id=node.id) is not None
        image = await nano_banana.generate_resumable(
            banana_prompt,
            job_id=node.id,
            save_to=image_path,
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
        produced = await threed.generate_mesh(
            image.image_bytes,
            output_path=raw,
            job_id=node.id,
            image_mime=image.mime_type,
        )
        async with _MESH_IO:
            scene = await asyncio.to_thread(trimesh.load, produced)
            rescaled = await asyncio.to_thread(
                rescale_mesh_to_bbox,
                scene,
                node.bbox,
                orientation=node.orientation,
            )
            await asyncio.to_thread(rescaled.export, path, file_type="glb")
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
        async with _GENERATE_FANOUT:
            rescaled = raw_dir / f"{node.id}.glb"
            async with _MESH_IO:
                scene = await asyncio.to_thread(trimesh.load, src_raw)
                placed = await asyncio.to_thread(
                    rescale_mesh_to_bbox, scene, node.bbox, orientation=node.orientation,
                )
                await asyncio.to_thread(placed.export, rescaled, file_type="glb")
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


async def generate_assets(
    *,
    nodes: list[Node],
    runs_dir: Path,
    run_id: str,
) -> None:
    """From-scratch (Nano-Banana + Trellis) build of `nodes` for the client's
    "generate" gate, independent of `_USE_ASSET_LIBRARY` and the library build's
    `objects/`. Two dirs under the cell:

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

    PREFAB REUSE: before generating an object fresh, a lightweight LLM
    (`prefabs.match`) checks whether it is essentially the SAME object as one
    already built (or in flight) earlier in this scene. On a hit the object skips
    Nano-Banana + Trellis entirely — its mesh is the matched asset's raw Trellis
    output rescaled into this object's bbox (identical geometry, fitted to the
    slot) — for visual consistency and to avoid paying for duplicates. Decisions
    are logged as `prefab.match` and replayed on resume so they stay stable.

    Up to `_GENERATE_FANOUT` assets run at once; threed's `_pace_submit` spaces
    the actual Trellis submits ~1s apart so the batch ramps onto Modal instead of
    bursting into a 429 storm.

    Requires a bound SlotLog: `_generate_one`, `prefabs.match`, and the
    nano_banana / threed services record their bookkeeping there. The caller
    binds a dedicated log (events.generated.jsonl) so none of this lands in the
    library build's event stream."""
    raw_dir = runs_dir / run_id / "objects-generated"
    opt_dir = runs_dir / run_id / "objects-generated-optimized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    opt_dir.mkdir(parents=True, exist_ok=True)

    # Per-scene prefab state: an ordered catalog of canonical assets (the reuse
    # candidates) and a "raw on disk" event per canonical so a reuse can wait for
    # a source still in flight. Local to this call — the whole scene is processed
    # in one pass, in node order, so a node only matches against earlier ones.
    catalog: list[tuple[str, str]] = []
    canonical_ids: set[str] = set()
    raw_ready: dict[str, asyncio.Event] = {}

    def _has_raw(node_id: str) -> bool:
        return (raw_dir / f"{node_id}.raw.glb").exists()

    async def _fresh(node: Node) -> None:
        async with _GENERATE_FANOUT:
            rescaled = raw_dir / f"{node.id}.glb"
            try:
                await _generate_one(
                    node,
                    raw=raw_dir / f"{node.id}.raw.glb",
                    path=rescaled,
                    image_stem=raw_dir / node.id,
                    runs_dir=runs_dir,
                )
            finally:
                # Raw is on disk now (or generation failed) — unblock any reuse
                # waiting on this asset; it re-checks the file and bails if none.
                raw_ready[node.id].set()
            if rescaled.exists():
                await _optimize_asset(rescaled, opt_dir / f"{node.id}.glb")

    async def _reuse(node: Node, source_id: str) -> None:
        # Wait (outside the fanout, so we don't pin a slot) for the source's
        # mesh to land, then rescale ITS raw Trellis output into this node's
        # slot — exactly as a fresh build would, so the reuse lands identically
        # posed. No Nano-Banana, no Trellis.
        await raw_ready[source_id].wait()
        await _rescale_reuse_from_raw(
            node,
            src_raw=raw_dir / f"{source_id}.raw.glb",
            raw_dir=raw_dir,
            opt_dir=opt_dir,
            source_id=source_id,
        )

    tasks: list[asyncio.Task[None]] = []
    for node in nodes:
        done = (opt_dir / f"{node.id}.glb").exists()
        decided = logging.find_event("prefab.match", id=node.id)
        if decided is not None:
            reuse_id = str(decided.get("reuse_id") or "")
        elif done or _has_raw(node.id):
            # An original built before prefabs (or a prior resume) — keep it an
            # original so resume never flips an already-built asset into a reuse.
            reuse_id = ""
        else:
            reuse_id = await prefabs.match(
                new_id=node.id, new_description=node.prompt, catalog=catalog,
            )
            logging.log("prefab.match", id=node.id, reuse_id=reuse_id, description=node.prompt)

        is_reuse = bool(reuse_id) and reuse_id in canonical_ids
        if not is_reuse:
            # Canonical: matchable by later objects; reuses rescale its raw mesh.
            catalog.append((node.id, node.prompt))
            canonical_ids.add(node.id)
            ev = raw_ready.setdefault(node.id, asyncio.Event())
            if done or _has_raw(node.id):
                ev.set()

        if done:
            continue
        if is_reuse:
            tasks.append(asyncio.create_task(_reuse(node, reuse_id)))
        else:
            tasks.append(asyncio.create_task(_fresh(node)))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def regenerate_one(
    *,
    node: Node,
    runs_dir: Path,
    run_id: str,
    subdir: str = "objects",
    optimize: bool = False,
) -> None:
    """Rebuild a single mesh FRESH: unlink every prior on-disk artifact for this
    node under `subdir` so the cache-aware checks inside
    `nano_banana.generate_resumable` and `threed.generate_mesh` miss and issue
    new API calls, then re-run Nano-Banana + Trellis + rescale. With
    `optimize=True` the freshly built mesh is run through the library optimizer
    into the `objects-generated-optimized/` served twin (the from-scratch
    generated pipeline); the library path leaves it off.

    Awaitable core shared by the library single-mesh retry (`retry_node`, a
    detached + `_pending`-tracked wrapper) and standalone generated-asset
    regeneration (awaited directly under its own cell task)."""
    objs_dir = runs_dir / run_id / subdir
    objs_dir.mkdir(parents=True, exist_ok=True)
    raw = objs_dir / f"{node.id}.raw.glb"
    path = objs_dir / f"{node.id}.glb"
    image_stem = objs_dir / node.id
    image_path = image_stem.parent / f"{image_stem.name}.png"
    for artifact in (image_path, raw, path):
        artifact.unlink(missing_ok=True)
    logging.log("mesh.retry", id=node.id, prompt=node.prompt)
    await _generate_one(node, raw=raw, path=path, image_stem=image_stem, runs_dir=runs_dir)
    if optimize and path.exists():
        await _optimize_asset(
            path, runs_dir / run_id / "objects-generated-optimized" / f"{node.id}.glb",
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
) -> None:
    """Re-derive every `reuses` node from `canonical_id`'s raw Trellis mesh —
    rescaling it into each reuse's own bbox/orientation and optimizing into the
    served twin. Pairs with a prior FRESH build of the canonical
    (`regenerate_one` with optimize=True) to push a regenerated prefab out to
    every object that shares it. Awaited under the caller's cell task, so a
    cancellation there tears the fan-out down via the gather."""
    raw_dir = runs_dir / run_id / "objects-generated"
    opt_dir = runs_dir / run_id / "objects-generated-optimized"
    src_raw = raw_dir / f"{canonical_id}.raw.glb"
    coros = [
        _rescale_reuse_from_raw(
            node, src_raw=src_raw, raw_dir=raw_dir, opt_dir=opt_dir, source_id=canonical_id,
        )
        for node in reuses
    ]
    if coros:
        await asyncio.gather(*coros, return_exceptions=True)


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

    while True:
        decision = await _next_object_validated(
            zone=zone,
            all_nodes=all_nodes,
        )
        if decision.done or decision.object is None:
            logging.log_once(
                "generation.next.done",
                match_fields=("zone",),
                zone=zone.id,
            )
            return
        logging.log_once(
            "generation.next",
            match_fields=("zone", "id"),
            zone=zone.id,
            id=decision.object.id,
            object=decision.object.model_dump(mode="json"),
        )
        new_nodes = await _resolve_and_generate(
            specs=[decision.object],
            zone=zone,
            all_nodes=all_nodes,
            scenario="anchor",
            runs_dir=runs_dir,
            run_id=run_id,
        )
        all_nodes.extend(new_nodes)
