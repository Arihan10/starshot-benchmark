"""Phase 1 — recursive top-down decomposition into a tree of zone Nodes.

Flow per node:
  1. ZONE PLAN (LLM) — high-level character/intent for this zone AND the
     is_atomic decision. Runs for EVERY zone, root included; the root's
     plan IS the scene plan.
  2. (root only) Overall bounding box (LLM) — sizes the canvas to match
     the silhouette implied by the root's plan.
  3. If ATOMIC, hand to Phase 2 generation for anchor-object population.
  4. ZONE DECOMPOSE (LLM) — only for non-atomic zones. Emits each child
     fully structured (id, prompt, proxy_shape, relationships) in one
     call. The sibling-relationship DAG is checked for cycles; any cycle
     is logged and accepted (advisory).
  5. Batch-resolve a bbox for EVERY child in one LLM call.
  6. Hand each placed child to Phase 2 generation for its encapsulating
     geometry (walls, moat, fence, etc.) as objects.
  7. Recurse on each child. Children arrive with `plan=None`; step 1
     authors their plan fresh.

Root also gets an encapsulating pass — its world-scale boundary. When
root is non-atomic, it runs immediately after root's children are
placed and before their per-child encapsulating passes, so the world
boundary is in the scene context every child frame sees. When root is
atomic, it runs before root's anchor pass for the same reason. Root
additionally gets a final negative-space pass at the end of the run.
"""

from __future__ import annotations

from pathlib import Path

from app.core.prompts import (
    BboxBatchOutput,
    ChildNodeSpec,
    OverallBboxOutput,
    SYSTEM_OVERALL_BBOX,
    SYSTEM_ROOT_ZONE_PLAN,
    SYSTEM_ZONE_BBOX_BATCH,
    SYSTEM_ZONE_DECOMPOSE,
    SYSTEM_ZONE_PLAN,
    ZoneDecomposeOutput,
    ZonePlanOutput,
    render_overall_bbox,
    render_zone_bbox_batch,
    render_zone_decompose,
    render_zone_plan,
)
from app.core.types import BoundingBox, Node
from app.pipeline import generation
from app.services import llm
from app.utils import logging
from app.utils.topology import validate_referenced_ids


async def _pick_overall_bbox(prompt: str, scene_plan: str) -> BoundingBox:
    out = await llm.call_llm(
        system=SYSTEM_OVERALL_BBOX,
        user=render_overall_bbox(prompt, scene_plan),
        output_schema=OverallBboxOutput,
        node_id="root",
        step="overall_bbox",
    )
    return out.bbox


def _prior_zones(
    all_nodes: list[Node],
) -> list[tuple[str, str, str | None, str, BoundingBox, str | None]]:
    """Every non-root zone already declared. Used as lateral scene
    context for the decomposition step so siblings/cousins can inform
    a zone's structure and relationships.

    We include zones whose plan hasn't been authored yet — depth-first
    traversal plans-then-decomposes one branch at a time, so the rest
    of the parent's siblings (declared in the same batch, recursed
    into later) sit here as `plan=None`. Their bboxes and placement
    text are the load-bearing pieces: a cousin zone the decomposer is
    about to plan around needs to know where its already-placed
    neighbors sit, even if those neighbors' detailed plans don't exist
    yet. Tuple: (id, prompt, plan_or_None, parent_id, bbox, placement).
    """
    out: list[tuple[str, str, str | None, str, BoundingBox, str | None]] = []
    for n in all_nodes:
        if n.mesh_url is not None:
            continue
        if n.parent_id is None:
            continue
        out.append((n.id, n.prompt, n.plan, n.parent_id, n.bbox, n.placement))
    return out


def _ancestors(
    node: Node, all_nodes: list[Node],
) -> list[tuple[str, str, str, BoundingBox, str | None]]:
    """Walk parent_id pointers up to the root, then return root-first →
    parent-of-`node`, excluding `node` itself. Each tuple is (id,
    prompt, plan, bbox, placement). `placement` is None for the root
    (which has no parent and thus no placement); non-None for every
    other ancestor.

    Most ancestors are planned by the time we recurse into a child (the
    depth-first walk plans before decomposing). One exception: when a
    child is sibling-parented to another subzone declared in the same
    decompose batch, the encapsulating pass for that child runs before
    its sibling-parent has been planned. In that case we fall back to
    the sibling's seed prompt with a label so the downstream LLM knows
    the level of detail is reduced for that ancestor."""
    by_id = {n.id: n for n in all_nodes}
    chain: list[Node] = []
    parent_id = node.parent_id
    while parent_id is not None:
        parent = by_id[parent_id]
        chain.append(parent)
        parent_id = parent.parent_id
    chain.reverse()
    out: list[tuple[str, str, str, BoundingBox, str | None]] = []
    for a in chain:
        if a.plan is not None:
            plan_text = a.plan
        else:
            plan_text = (
                f"(seed prompt — full plan not yet authored): {a.prompt}"
            )
        out.append((a.id, a.prompt, plan_text, a.bbox, a.placement))
    return out


def _generated_objects(
    all_nodes: list[Node],
) -> list[tuple[str, str, str | None, BoundingBox, str | None, str | None]]:
    """Every concrete (mesh-bearing) node placed so far, in declaration
    order, with its bbox, placement prose, and parent_kind. Used to
    ground planning context in what the scene actually looks like, not
    just what's been promised — and so these nodes are valid
    `referenced_ids` targets in downstream placement text. Tuple: (id,
    prompt, parent_id, bbox, placement, parent_kind). `parent_kind` lets
    the zone-decompose narrative split shell frames (ATTACHED) from
    interior anchor objects (ON / IN)."""
    return [
        (
            n.id, n.prompt, n.parent_id, n.bbox, n.placement,
            n.parent_kind.value if n.parent_kind is not None else None,
        )
        for n in all_nodes
        if n.mesh_url is not None
    ]


async def _plan_zone(
    *,
    zone_id: str,
    zone_prompt: str,
    ancestors: list[tuple[str, str, str, BoundingBox, str | None]],
    objects: list[tuple[str, str, str | None, BoundingBox, str | None]],
) -> ZonePlanOutput:
    """Author the high-level plan for a zone and decide whether it is atomic.
    Works for any zone (root or nested) — the root just passes empty ancestors."""
    system = SYSTEM_ROOT_ZONE_PLAN if not ancestors else SYSTEM_ZONE_PLAN
    return await llm.call_llm(
        system=system,
        user=render_zone_plan(
            zone_id=zone_id,
            zone_prompt=zone_prompt,
            ancestors=ancestors,
            objects=objects,
        ),
        output_schema=ZonePlanOutput,
        node_id=zone_id,
        step="zone_plan",
    )


async def _decompose_zone(
    *, node: Node, all_nodes: list[Node],
) -> ZoneDecomposeOutput:
    """Emit child zones for a non-atomic zone. Each child is fully
    structured (id, prompt, proxy_shape, relationships) in one call."""
    assert node.plan is not None, "zone must be planned before decomposition"
    root = all_nodes[0]
    assert root.plan is not None, "root.plan must be set before decomposition"
    return await llm.call_llm(
        system=SYSTEM_ZONE_DECOMPOSE,
        user=render_zone_decompose(
            zone_id=node.id,
            zone_prompt=node.prompt,
            zone_bbox=node.bbox,
            zone_plan=node.plan,
            ancestors=_ancestors(node, all_nodes),
            objects=_generated_objects(all_nodes),
            scene_prompt=root.prompt,
            scene_plan=root.plan,
            prior_zones=_prior_zones(all_nodes),
        ),
        output_schema=ZoneDecomposeOutput,
        node_id=node.id,
        step="zone_decompose",
    )


async def _resolve_child_bboxes_batch(
    *, parent: Node, children: list[ChildNodeSpec],
) -> dict[str, BoundingBox]:
    out = await llm.call_llm(
        system=SYSTEM_ZONE_BBOX_BATCH,
        user=render_zone_bbox_batch(
            parent_id=parent.id,
            parent_bbox=parent.bbox,
            children=children,
        ),
        output_schema=BboxBatchOutput,
        node_id=parent.id,
        step="child_bbox_batch",
    )
    return {a.id: a.bbox for a in out.assignments}


async def _build(
    *, node: Node, runs_dir: Path, run_id: str, all_nodes: list[Node],
    is_atomic: bool | None = None,
) -> None:
    if node.plan is None:
        logging.emit_step(node.id, "planning")
        plan_out = await _plan_zone(
            zone_id=node.id,
            zone_prompt=node.prompt,
            ancestors=_ancestors(node, all_nodes),
            objects=_generated_objects(all_nodes),
        )
        planned = node.model_copy(update={"plan": plan_out.plan})
        idx = all_nodes.index(node)
        all_nodes[idx] = planned
        node = planned
        is_atomic = plan_out.is_atomic
        logging.log(
            "divider.zone_plan",
            node=node.id, plan=plan_out.plan, is_atomic=is_atomic,
        )

    assert is_atomic is not None, "is_atomic must be set by plan or caller"

    if is_atomic:
        if node.parent_id is None:
            logging.emit_step(node.id, "generating_frame")
            await generation.run(
                zone=node, runs_dir=runs_dir, run_id=run_id,
                scenario="encapsulating", all_nodes=all_nodes,
                ancestors=_ancestors(node, all_nodes),
            )
        logging.emit_step(node.id, "generating_anchor")
        await generation.run(
            zone=node, runs_dir=runs_dir, run_id=run_id,
            scenario="anchor", all_nodes=all_nodes,
            ancestors=_ancestors(node, all_nodes),
        )
        logging.emit_step(node.id, "done")
        return

    logging.emit_step(node.id, "decomposing")
    decomp = await _decompose_zone(node=node, all_nodes=all_nodes)
    logging.log(
        "divider.zone_decompose",
        node=node.id,
        children=[c.model_dump() for c in decomp.children],
    )

    try:
        validate_referenced_ids(
            decomp.children,
            parent_id=node.id,
            existing_ids={n.id for n in all_nodes},
        )
    except ValueError as e:
        logging.log(
            "divider.validate.referenced_ids.accept_invalid",
            node=node.id, reason=str(e),
        )

    logging.emit_step(node.id, "resolving_bboxes", parent=node.id)
    bboxes = await _resolve_child_bboxes_batch(
        parent=node, children=decomp.children,
    )

    placed: list[Node] = []
    for spec in decomp.children:
        child_bbox = bboxes[spec.id]
        logging.emit_bbox(
            spec.id, child_bbox,
            parent_id=node.id, prompt=spec.prompt, kind="zone",
            proxy_shape=spec.proxy_shape,
        )
        child = Node(
            id=spec.id,
            prompt=spec.prompt,
            bbox=child_bbox,
            proxy_shape=spec.proxy_shape,
            placement=spec.placement,
            referenced_ids=list(spec.referenced_ids),
            parent_id=spec.parent,
            parent_kind=spec.parent_kind,
            plan=None,
        )
        placed.append(child)
        all_nodes.append(child)

    if node.parent_id is None:
        logging.emit_step(node.id, "generating_frame")
        await generation.run(
            zone=node, runs_dir=runs_dir, run_id=run_id,
            scenario="encapsulating", all_nodes=all_nodes,
            ancestors=_ancestors(node, all_nodes),
        )

    for child in placed:
        logging.emit_step(child.id, "generating_frame")
        await generation.run(
            zone=child, runs_dir=runs_dir, run_id=run_id,
            scenario="encapsulating", all_nodes=all_nodes,
            ancestors=_ancestors(child, all_nodes),
        )

    for child in placed:
        await _build(
            node=child, runs_dir=runs_dir, run_id=run_id, all_nodes=all_nodes,
        )
    logging.emit_step(node.id, "done")


async def run(
    *, run_id: str, prompt: str, model: str, runs_dir: Path,
) -> Node:
    llm.set_model(model)
    logging.emit_step("root", "planning")
    plan_out = await _plan_zone(
        zone_id="root", zone_prompt=prompt, ancestors=[], objects=[],
    )
    logging.log(
        "divider.zone_plan",
        node="root", plan=plan_out.plan, is_atomic=plan_out.is_atomic,
    )
    bbox = await _pick_overall_bbox(prompt, plan_out.plan)
    logging.emit_bbox("root", bbox, parent_id=None, prompt=prompt, kind="zone")
    root = Node(
        id="root", prompt=prompt, bbox=bbox, parent_id=None, plan=plan_out.plan,
    )
    all_nodes: list[Node] = [root]
    await _build(
        node=root, runs_dir=runs_dir, run_id=run_id,
        all_nodes=all_nodes, is_atomic=plan_out.is_atomic,
    )
    logging.emit_step(root.id, "generating_negative_space")
    await generation.run(
        zone=root, runs_dir=runs_dir, run_id=run_id,
        scenario="negative-space", all_nodes=all_nodes,
        ancestors=_ancestors(root, all_nodes),
    )
    logging.emit_step(root.id, "done")
    return root
