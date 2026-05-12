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
from pathlib import Path
from typing import Literal

import trimesh

# Guards the trimesh load -> rescale -> export block. API calls and GLB
# downloads stay fully parallel across slots; only the RAM-heavy mesh
# decode is serialized so concurrent slots don't stack Pillow-decoded
# texture buffers and trip the OOM killer.
_MESH_IO = asyncio.Semaphore(1)

from app.core.prompts import (
    BboxBatchOutput,
    ImagePromptOutput,
    NextObjectOutput,
    ObjectDecompOutput,
    ObjectSpec,
    SYSTEM_IMAGE_PROMPT,
    SYSTEM_NEXT_OBJECT,
    SYSTEM_OBJECT_BBOX_BATCH,
    SYSTEM_OBJECT_DECOMP,
    render_image_prompt,
    render_next_object,
    render_object_bbox_batch,
    render_object_decomp,
    wrap_image_prompt,
)
from app.core.types import BoundingBox, Node, Orientation, ProxyShape
from app.services import llm, threed
from app.utils import logging
from app.utils.geometry import rescale_mesh_to_bbox
from app.utils.topology import validate_object_relationships


def _artifact_url(runs_dir: Path, path: Path) -> str:
    return f"/artifacts/{path.relative_to(runs_dir).as_posix()}"


# Projection of the live node registry into the tuple shape every render
# function expects for "what's already in the scene". Centralised so the
# shape can evolve without combing every call site.
def _scene_view(
    nodes: list[Node],
) -> list[tuple[str, str, BoundingBox, str | None, ProxyShape | None, Orientation]]:
    return [
        (n.id, n.prompt, n.bbox, n.parent_id, n.proxy_shape, n.orientation)
        for n in nodes
    ]


RELATIONSHIP_RETRY_ATTEMPTS = 3


async def _decompose_objects_validated(
    *,
    zone: Node,
    scenario: Literal["anchor", "encapsulating", "negative-space"],
    all_nodes: list[Node],
) -> list[ObjectSpec]:
    prior_attempts: list[tuple[list[ObjectSpec], str]] = []
    existing_ids = {n.id for n in all_nodes}
    scene = _scene_view(all_nodes)
    specs: list[ObjectSpec] = []
    for attempt in range(RELATIONSHIP_RETRY_ATTEMPTS):
        out = await llm.call_llm(
            system=SYSTEM_OBJECT_DECOMP,
            user=render_object_decomp(
                zone_id=zone.id,
                zone_prompt=zone.prompt,
                zone_bbox=zone.bbox,
                scenario=scenario,
                scene=scene,
                prior_attempts=prior_attempts,
            ),
            output_schema=ObjectDecompOutput,
        )
        specs = list(out.objects)
        try:
            validate_object_relationships(specs, zone_id=zone.id, existing_ids=existing_ids)
            return specs
        except ValueError as e:
            reason = str(e)
            logging.log(
                "generation.decompose.retry",
                zone=zone.id, attempt=attempt, reason=reason,
                emitted=[s.model_dump() for s in specs],
            )
            raise  # TEMP: stop pipeline on first validation failure
    logging.log(
        "generation.decompose.accept_invalid",
        zone=zone.id, reason=prior_attempts[-1][1] if prior_attempts else "",
    )
    return specs


async def _next_object_validated(
    *, zone: Node, all_nodes: list[Node],
) -> NextObjectOutput:
    prior_attempts: list[tuple[ObjectSpec, str]] = []
    existing_ids = {n.id for n in all_nodes}
    scene = _scene_view(all_nodes)
    decision: NextObjectOutput | None = None
    for attempt in range(RELATIONSHIP_RETRY_ATTEMPTS):
        decision = await llm.call_llm(
            system=SYSTEM_NEXT_OBJECT,
            user=render_next_object(
                zone_id=zone.id,
                zone_prompt=zone.prompt,
                zone_bbox=zone.bbox,
                scene=scene,
                prior_attempts=prior_attempts,
            ),
            output_schema=NextObjectOutput,
        )
        if decision.done or decision.object is None:
            return decision
        try:
            validate_object_relationships(
                [decision.object], zone_id=zone.id, existing_ids=existing_ids,
            )
            return decision
        except ValueError as e:
            reason = str(e)
            logging.log(
                "generation.next.retry",
                zone=zone.id, attempt=attempt, reason=reason,
                emitted=decision.object.model_dump(),
            )
            raise  # TEMP: stop pipeline on first validation failure
    assert decision is not None
    logging.log(
        "generation.next.accept_invalid",
        zone=zone.id, reason=prior_attempts[-1][1] if prior_attempts else "",
    )
    return decision


async def _resolve_and_generate(
    *,
    specs: list[ObjectSpec],
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
    deduped: list[ObjectSpec] = []
    seen_in_call: set[str] = set()
    for s in specs:
        if s.id in seen_in_call or s.id in admitted or s.id in placed_ids:
            logging.log(
                "generation.dedup_drop",
                zone=zone.id, scenario=scenario, id=s.id,
                reason=(
                    "duplicate_in_call" if s.id in seen_in_call
                    else "already_placed" if s.id in placed_ids
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

    out = await llm.call_llm(
        system=SYSTEM_OBJECT_BBOX_BATCH,
        user=render_object_bbox_batch(
            zone_id=zone.id,
            zone_prompt=zone.prompt,
            zone_bbox=zone.bbox,
            objects=specs,
            peers=_scene_view(all_nodes),
        ),
        output_schema=BboxBatchOutput,
    )
    bboxes = {a.id: a.bbox for a in out.assignments}

    # Every object (anchor, completion, encapsulating alike) goes through
    # the image-prompt rewrite: Nano Banana needs an isolated studio
    # reference shot — any environmental context bleeds into the mesh.
    committed_image_prompts = [
        n.prompt for n in all_nodes if n.mesh_url is not None
    ]
    resolved: list[Node] = []
    for spec in specs:
        bbox = bboxes[spec.id]
        logging.emit_bbox(
            spec.id, bbox,
            parent_id=spec.parent, prompt=spec.prompt,
            kind="frame" if scenario == "encapsulating" else "object",
            proxy_shape=spec.proxy_shape,
        )
        prior_prompts = committed_image_prompts + [r.prompt for r in resolved]
        image_prompt = await _build_image_prompt(
            prompt=spec.prompt, bbox=bbox, proxy_shape=spec.proxy_shape,
            prior_prompts=prior_prompts,
        )
        resolved.append(Node(
            id=spec.id,
            prompt=image_prompt,
            bbox=bbox,
            proxy_shape=spec.proxy_shape,
            orientation=spec.orientation,
            relationships=list(spec.relationships),
            parent_id=spec.parent,
        ))

    return await _spawn_meshes(
        resolved=resolved, runs_dir=runs_dir, run_id=run_id, scenario=scenario,
    )


async def _build_image_prompt(
    *,
    prompt: str,
    bbox: BoundingBox,
    proxy_shape: ProxyShape | None,
    prior_prompts: list[str],
) -> str:
    out = await llm.call_llm(
        system=SYSTEM_IMAGE_PROMPT,
        user=render_image_prompt(
            prompt=prompt, bbox=bbox, proxy_shape=proxy_shape,
            prior_prompts=prior_prompts,
        ),
        output_schema=ImagePromptOutput,
    )
    return wrap_image_prompt(out.prompt, proxy_shape, bbox.size)


_pending: dict[str, list[asyncio.Task[None]]] = {}

# Run-scoped set of every id that has been admitted into _resolve_and_generate
# for this run. Populated *synchronously* before the bbox-resolution LLM call,
# so concurrent specs (within or across decomposition steps) can't both pass
# the dedup check and race into Banana+Trellis. The artifact cache catches
# *most* repeats, but two specs that submit before either has logged a
# `cache.artifact` will both miss the cache and double-bill — this guard
# closes that window.
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
        paths = await threed.generate_mesh(
            node.prompt, output_path=raw, image_stem=image_stem,
        )
        logging.log(
            "image",
            id=node.id,
            url=_artifact_url(runs_dir, paths["image"]),
            prompt=node.prompt,
        )
        async with _MESH_IO:
            scene = await asyncio.to_thread(trimesh.load, raw)
            rescaled = await asyncio.to_thread(
                rescale_mesh_to_bbox, scene, node.bbox,
                orientation=node.orientation,
            )
            await asyncio.to_thread(rescaled.export, path, file_type="glb")
            del scene, rescaled
        logging.emit_model(
            node.id, artifact_kind="object", url=_artifact_url(runs_dir, path),
        )
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
        zone=zone, scenario=scenario, all_nodes=all_nodes,
    )
    logging.log(
        "generation.decompose",
        zone=zone.id,
        scenario=scenario,
        objects=[s.id for s in specs],
    )

    if specs:
        placed = await _resolve_and_generate(
            specs=specs, zone=zone, all_nodes=all_nodes, scenario=scenario,
            runs_dir=runs_dir, run_id=run_id,
        )
        all_nodes.extend(placed)

    if scenario != "anchor":
        return

    while True:
        decision = await _next_object_validated(zone=zone, all_nodes=all_nodes)
        if decision.done or decision.object is None:
            logging.log("generation.next.done", zone=zone.id)
            return
        logging.log("generation.next", zone=zone.id, id=decision.object.id)
        new_nodes = await _resolve_and_generate(
            specs=[decision.object], zone=zone, all_nodes=all_nodes,
            scenario="anchor",
            runs_dir=runs_dir, run_id=run_id,
        )
        all_nodes.extend(new_nodes)
