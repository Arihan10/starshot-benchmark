"""Phase 1 — recursive top-down decomposition into a tree of zone Nodes.

Per-node flow (inside `_build`, which receives the node already planned):
  1. ZONE PLAN (LLM) — high-level character/intent AND the is_atomic
     decision. Authored at the parent level (in the parent's child loop)
     so the plan paragraph is in scope by the time this zone's own shell
     is generated. Root is planned in `run()` before `_build(root)`; the
     root's overall bounding box (LLM) is also sized there.
  2. If ATOMIC: run the encapsulating pass (shell / floor / ground),
     then hand to Phase 2 generation for anchor-object population.
  3. If NON-ATOMIC: ZONE DECOMPOSE (LLM) — emit each child as a seed
     (id, prompt, proxy_shape, relationships) in one call. The sibling-
     relationship DAG is checked for cycles; any cycle is logged and
     accepted (advisory).
  4. Batch-resolve a bbox for EVERY child in one LLM call.
  5. Encapsulating pass — runs AFTER this zone's own decomposition + child
     bbox solver, so the shell author sees where the zone's children are
     actually placed (their bboxes/footprints) rather than only the bare
     zone plan. Generates walls+floor+ceiling for architectural zones and
     a single ground mesh for atomic terrain zones.
  6. For each placed child, in declaration order: author the child's plan
     (LLM), then recurse. Each child's own encapsulating pass runs inside
     its recursion (step 5), once the child's decomposition is known.

Root additionally gets a final negative-space pass at the end of the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core import prompt_runtime
from app.core.types import BoundingBox, Node
from app.pipeline import generation
from app.services import llm
from app.utils import logging
from app.utils.topology import validate_referenced_ids


async def _pick_overall_bbox(prompt: str, scene_plan: str) -> BoundingBox:
    p = prompt_runtime.current()
    out = await llm.call_llm(
        system=p.SYSTEM_OVERALL_BBOX,
        user=p.render_overall_bbox(prompt, scene_plan),
        output_schema=p.OverallBboxOutput,
        node_id="root",
        step="overall_bbox",
    )
    return out.bbox


async def _plan_zone(
    *,
    zone_id: str,
    zone_prompt: str,
    nodes: list[Node],
) -> Any:
    """Author the high-level plan for a zone and decide whether it is atomic.
    Works for any zone (root or nested) — the root just passes empty `nodes`."""
    p = prompt_runtime.current()
    system = p.SYSTEM_ROOT_ZONE_PLAN if not nodes else p.SYSTEM_ZONE_PLAN
    return await llm.call_llm(
        system=system,
        user=p.render_zone_plan(
            zone_id=zone_id,
            zone_prompt=zone_prompt,
            nodes=nodes,
        ),
        output_schema=p.ZonePlanOutput,
        node_id=zone_id,
        step="zone_plan",
    )


async def _decompose_zone(
    *,
    node: Node,
    all_nodes: list[Node],
) -> Any:
    """Emit child zones for a non-atomic zone. Each child is fully
    structured (id, prompt, proxy_shape, relationships) in one call."""
    assert node.plan is not None, "zone must be planned before decomposition"
    p = prompt_runtime.current()
    return await llm.call_llm(
        system=p.SYSTEM_ZONE_DECOMPOSE,
        user=p.render_zone_decompose(
            zone_id=node.id,
            zone_plan=node.plan,
            nodes=all_nodes,
        ),
        output_schema=p.ZoneDecomposeOutput,
        node_id=node.id,
        step="zone_decompose",
    )


async def _resolve_child_bboxes_batch(
    *,
    parent: Node,
    children: list[Any],
    all_nodes: list[Node],
) -> dict[str, BoundingBox]:
    bbox_by_id = {n.id: n.bbox for n in all_nodes}
    p = prompt_runtime.current()
    out = await llm.call_llm(
        system=p.SYSTEM_ZONE_BBOX_BATCH,
        user=p.render_zone_bbox_batch(
            parent_id=parent.id,
            children=children,
            nodes=all_nodes,
        ),
        output_schema=p.BboxBatchOutput,
        node_id=parent.id,
        step="child_bbox_batch",
    )
    # LLM emits each child's bbox in that child's parent's local frame.
    # Convert to world coordinates per-child with topological ordering
    # for intra-batch parents (child B parents to child A in same batch).
    spec_parent = {c.id: c.parent for c in children}
    assignments_by_id = {a.id: a.bbox for a in out.assignments}
    bboxes: dict[str, BoundingBox] = {}
    remaining = set(assignments_by_id.keys())
    while remaining:
        progress = False
        for child_id in list(remaining):
            parent_id = spec_parent.get(child_id, parent.id)
            if parent_id in bboxes:
                parent_bbox_resolved = bboxes[parent_id]
            elif parent_id in bbox_by_id:
                parent_bbox_resolved = bbox_by_id[parent_id]
            elif parent_id in remaining:
                continue
            else:
                parent_bbox_resolved = parent.bbox
            bboxes[child_id] = assignments_by_id[child_id].to_world_frame(parent_bbox_resolved)
            remaining.discard(child_id)
            progress = True
        if not progress:
            for child_id in list(remaining):
                bboxes[child_id] = assignments_by_id[child_id].to_world_frame(parent.bbox)
            remaining.clear()
    return bboxes


async def _build(
    *,
    node: Node,
    runs_dir: Path,
    run_id: str,
    all_nodes: list[Node],
    is_atomic: bool,
) -> None:
    assert node.plan is not None, "node.plan must be set by caller"

    if is_atomic:
        # No decomposition for atomic leaves, so the encapsulating pass runs
        # right before anchor population (the same relative order it had when
        # it lived in the parent's child loop).
        logging.emit_step(node.id, "generating_frame")
        await generation.run(
            zone=node,
            runs_dir=runs_dir,
            run_id=run_id,
            scenario="encapsulating",
            all_nodes=all_nodes,
        )
        logging.emit_step(node.id, "generating_anchor")
        await generation.run(
            zone=node,
            runs_dir=runs_dir,
            run_id=run_id,
            scenario="anchor",
            all_nodes=all_nodes,
        )
        logging.emit_step(node.id, "done")
        return

    logging.emit_step(node.id, "decomposing")
    decomp = await _decompose_zone(node=node, all_nodes=all_nodes)
    logging.log_once(
        "divider.zone_decompose",
        match_fields=("node",),
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
            node=node.id,
            reason=str(e),
        )

    logging.emit_step(node.id, "resolving_bboxes", parent=node.id)
    bboxes = await _resolve_child_bboxes_batch(
        parent=node,
        children=decomp.children,
        all_nodes=all_nodes,
    )

    placed: list[Node] = []
    for spec in decomp.children:
        child_bbox = bboxes[spec.id]
        logging.emit_bbox(
            spec.id,
            child_bbox,
            parent_id=node.id,
            prompt=spec.prompt,
            kind="zone",
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

    # Encapsulating pass runs AFTER this zone's decomposition + child bbox
    # solver: the children are now in `all_nodes` with resolved bboxes, so the
    # shell author sees where they sit and can frame the perimeter around them.
    logging.emit_step(node.id, "generating_frame")
    await generation.run(
        zone=node,
        runs_dir=runs_dir,
        run_id=run_id,
        scenario="encapsulating",
        all_nodes=all_nodes,
    )

    for child in placed:
        logging.emit_step(child.id, "planning")
        plan_out = await _plan_zone(
            zone_id=child.id,
            zone_prompt=child.prompt,
            nodes=all_nodes,
        )
        planned = child.model_copy(update={"plan": plan_out.plan})
        idx = all_nodes.index(child)
        all_nodes[idx] = planned
        logging.log_once(
            "divider.zone_plan",
            match_fields=("node",),
            node=planned.id,
            plan=plan_out.plan,
            is_atomic=plan_out.is_atomic,
        )

        # The child's own encapsulating pass runs inside this recursion, once
        # the child's decomposition + child bbox solver have run.
        await _build(
            node=planned,
            runs_dir=runs_dir,
            run_id=run_id,
            all_nodes=all_nodes,
            is_atomic=plan_out.is_atomic,
        )
    logging.emit_step(node.id, "done")


async def run(
    *,
    run_id: str,
    prompt: str,
    model: str,
    runs_dir: Path,
) -> Node:
    llm.set_model(model)
    logging.emit_step("root", "planning")
    plan_out = await _plan_zone(
        zone_id="root",
        zone_prompt=prompt,
        nodes=[],
    )
    logging.log_once(
        "divider.zone_plan",
        match_fields=("node",),
        node="root",
        plan=plan_out.plan,
        is_atomic=plan_out.is_atomic,
    )
    bbox = await _pick_overall_bbox(prompt, plan_out.plan)
    logging.emit_bbox("root", bbox, parent_id=None, prompt=prompt, kind="zone")
    root = Node(
        id="root",
        prompt=prompt,
        bbox=bbox,
        parent_id=None,
        plan=plan_out.plan,
    )
    all_nodes: list[Node] = [root]
    # The root's encapsulating pass now runs inside `_build(root)`, after the
    # root's own decomposition + child bbox solver (or, if the root is atomic,
    # just before its anchor population).
    await _build(
        node=root,
        runs_dir=runs_dir,
        run_id=run_id,
        all_nodes=all_nodes,
        is_atomic=plan_out.is_atomic,
    )
    logging.emit_step(root.id, "generating_negative_space")
    await generation.run(
        zone=root,
        runs_dir=runs_dir,
        run_id=run_id,
        scenario="negative-space",
        all_nodes=all_nodes,
    )
    logging.emit_step(root.id, "done")
    return root
