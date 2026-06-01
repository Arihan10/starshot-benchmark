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


async def _decompose_objects_validated(
    *,
    zone: Node,
    scenario: Literal["anchor", "encapsulating", "negative-space"],
    all_nodes: list[Node],
) -> list[Any]:
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
    bboxes: dict[str, BoundingBox] = {}
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
    dims = bbox.size if include_dimensions else None
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
        objects=[s.id for s in specs],
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
