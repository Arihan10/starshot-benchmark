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
relationship validator on the emitted specs and proposes a LIST of
objects per round. Bounding-box resolution is a single batch call.

Prompt text comes from the run's prompt snapshot (`prompt_store.current()`);
this module only decides which step fires when and with which scene state.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any, Literal

import trimesh

from app.core import prompt_store, scene_context, schemas
from app.core.types import BoundingBox, Node, ProxyShape
from app.pipeline import committed
from app.services import llm, nano_banana, threed
from app.utils import glb_place, logging
from app.utils.geometry import rescale_mesh_to_bbox
from app.utils.topology import validate_parents, validate_referenced_ids

_USE_ASSET_LIBRARY = os.environ.get("USE_ASSET_LIBRARY", "false").lower() == "true"

# Guards the trimesh load -> rescale -> export block. API calls and GLB
# downloads stay fully parallel across slots; only the RAM-heavy mesh
# decode is serialized so concurrent slots don't stack Pillow-decoded
# texture buffers and trip the OOM killer.
_MESH_IO = asyncio.Semaphore(1)


def _artifact_url(runs_dir: Path, path: Path) -> str:
    return f"/artifacts/{path.relative_to(runs_dir).as_posix()}"


RELATIONSHIP_RETRY_ATTEMPTS = 3

# scenario -> (template/event step name, {SCENE_CONTEXT} target marker text)
_DECOMP_STEPS: dict[str, tuple[str, str]] = {
    "anchor": (
        "anchor_decompose",
        "This is the subregion you are to generate a list of anchor objects for.",
    ),
    "encapsulating": (
        "encapsulating_decompose",
        "This is the region you are to decide whether a boundary is needed for, and if so, what objects form that boundary",
    ),
    "negative-space": (
        "negative_space_decompose",
        "This is the region whose interstitial negative space you are filling.",
    ),
}


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
    step, target_text = _DECOMP_STEPS[scenario]
    # Only the encapsulating pass emits the `bounding_required` perimeter gate;
    # anchor + negative-space use the bare object-list schema, so they can't (and
    # don't) emit that field.
    decomp_schema = (
        schemas.EncapsulatingDecompOutput
        if scenario == "encapsulating"
        else schemas.ObjectDecompOutput
    )
    for attempt in range(RELATIONSHIP_RETRY_ATTEMPTS):
        ps = prompt_store.current()
        variables = scene_context.zone_vars(
            zone_id=zone.id,
            zone_prompt=zone.prompt,
            zone_plan=zone.plan,
            nodes=all_nodes,
            target_text=target_text,
        )
        variables["RETRY_BLOCK"] = scene_context.render_retry_block(prior_attempts)
        out = await llm.call_llm(
            system=ps.system(step, variables),
            user=ps.user(step, variables),
            output_schema=decomp_schema,
            node_id=zone.id,
            step=step,
            template=step,
            variables=variables,
        )
        if isinstance(out, schemas.EncapsulatingDecompOutput) and not out.bounding_required:
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


async def _next_object_batch_validated(
    *,
    zone: Node,
    all_nodes: list[Node],
) -> tuple[bool, list[Any]]:
    """Anchor-completion decision: the model proposes a LIST of objects per
    round (or signals done), with the relationship-validator retry applied to
    the whole proposed batch. Returns `(done, objects)`; `objects` is empty
    when done."""
    prior_attempts: list[tuple[list[Any], str]] = []
    existing_ids = {n.id for n in all_nodes}
    objects: list[Any] = []
    for attempt in range(RELATIONSHIP_RETRY_ATTEMPTS):
        ps = prompt_store.current()
        variables = scene_context.zone_vars(
            zone_id=zone.id,
            zone_prompt=zone.prompt,
            zone_plan=zone.plan,
            nodes=all_nodes,
            target_text="This is the subregion you are deciding whether to add more objects to.",
        )
        variables["RETRY_BLOCK"] = scene_context.render_next_object_retry_block(prior_attempts)
        decision = await llm.call_llm(
            system=ps.system("next_object", variables),
            user=ps.user("next_object", variables),
            output_schema=schemas.NextObjectOutput,
            node_id=zone.id,
            step="next_object",
            template="next_object",
            variables=variables,
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
        nodes=all_nodes,
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

    if _USE_ASSET_LIBRARY:
        return await _match_library_assets(
            specs=specs,
            bboxes=bboxes,
            orientations=orientations,
            scenario=scenario,
            runs_dir=runs_dir,
            run_id=run_id,
            zone_id=zone.id,
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
        # bbox already emitted upfront above.
        prior_subjects = committed_subjects + [r.prompt for r in resolved]
        view = "three-quarter" if scenario == "encapsulating" else "front"
        subject_prompt, image_prompt = await _build_image_prompt(
            spec_id=spec.id,
            prompt=spec.prompt,
            bbox=bbox,
            proxy_shape=spec.proxy_shape,
            prior_prompts=prior_subjects,
            view=view,
            zone=zone,
            nodes=all_nodes,
        )
        resolved.append(
            Node(
                id=spec.id,
                prompt=subject_prompt,
                image_prompt=image_prompt,
                bbox=bbox,
                proxy_shape=spec.proxy_shape,
                orientation=orientations[spec.id],
                placement=spec.placement,
                referenced_ids=list(spec.referenced_ids),
                parent_id=parent_id,
                parent_kind=spec.parent_kind,
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
                bbox=bbox,
                proxy_shape=spec.proxy_shape,
                orientation=orientation,
                placement=spec.placement,
                referenced_ids=list(spec.referenced_ids),
                parent_id=spec.parent,
                parent_kind=spec.parent_kind,
                parent_region=zone_id,
                mesh_url=url,
            )

        # bbox already emitted upfront in _resolve_and_generate.
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
            # Break a hardlink to the source's image before overwriting, so a
            # branch re-match can't mutate the source cell's shared inode.
            dest_image.unlink(missing_ok=True)
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
            bbox=bbox,
            proxy_shape=spec.proxy_shape,
            orientation=orientation,
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
    """Returns (subject_phrase, wrapped_image_prompt). The subject phrase is
    the LLM's bare noun phrase — what gets stored on Node.prompt and shown
    in context. The wrapped prompt is the full Nano-Banana studio-shot
    directive — used only at the image-generation boundary.

    Set include_dimensions=False for library generation where objects have
    no meaningful bbox — omits the dimension constraint from the image
    prompt so the model renders natural proportions."""
    dims = bbox.size if include_dimensions else None
    # Resume: reuse the committed subject phrase so a replayed object keeps
    # the exact prompt the rest of the scene already references downstream.
    subject = committed.image_subject(spec_id)
    if subject is not None:
        return subject, scene_context.wrap_image_prompt(subject, proxy_shape, dims, view=view)
    ps = prompt_store.current()
    variables = scene_context.image_prompt_vars(
        prompt=prompt,
        bbox=bbox,
        proxy_shape=proxy_shape,
        prior_prompts=prior_prompts,
        zone=zone,
        nodes=nodes,
    )
    out = await llm.call_llm(
        system=ps.system("image_prompt", variables),
        user=ps.user("image_prompt", variables),
        output_schema=schemas.ImagePromptOutput,
        node_id=spec_id,
        step="image_prompt",
        template="image_prompt",
        variables=variables,
    )
    return out.prompt, scene_context.wrap_image_prompt(out.prompt, proxy_shape, dims, view=view)


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
        await threed.generate_mesh(
            image.image_bytes,
            output_path=raw,
            job_id=node.id,
            image_mime=image.mime_type,
        )
        async with _MESH_IO:
            scene = await asyncio.to_thread(trimesh.load, raw)
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


async def retry_node(
    *,
    node: Node,
    runs_dir: Path,
    run_id: str,
) -> asyncio.Task[None]:
    """Standalone re-generation of a single failed mesh — always FRESH:
    every prior on-disk artifact for this node is unlinked so the
    cache-aware checks inside `nano_banana.generate_resumable` and
    `threed.generate_mesh` miss and issue new API calls. The user is
    asking for "give this object another shot," not "reuse what we had,"
    so a stale banana image (e.g. one that produced a broken Trellis run)
    doesn't get recycled. Registered on `_pending` so `cancel_pending`
    (slot reset / teardown) can tear it down alongside in-flight
    pipeline meshes."""
    objs_dir = runs_dir / run_id / "objects"
    objs_dir.mkdir(parents=True, exist_ok=True)
    raw = objs_dir / f"{node.id}.raw.glb"
    path = objs_dir / f"{node.id}.glb"
    image_stem = objs_dir / node.id
    image_path = image_stem.parent / f"{image_stem.name}.png"
    for artifact in (image_path, raw, path):
        artifact.unlink(missing_ok=True)
    logging.log("mesh.retry", id=node.id, prompt=node.prompt)
    task = asyncio.create_task(
        _generate_one(
            node,
            raw=raw,
            path=path,
            image_stem=image_stem,
            runs_dir=runs_dir,
        ),
    )
    _pending.setdefault(run_id, []).append(task)
    return task


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
    # The model proposes a LIST of objects per round. Each accepted object is
    # committed as its own `generation.next` event so resume replays them one
    # at a time regardless of how they were proposed.
    attempted: set[str] = set()
    while True:
        done, objects = await _next_object_batch_validated(
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
