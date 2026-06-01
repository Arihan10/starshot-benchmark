"""Phase 1 — recursive top-down decomposition into a tree of zone Nodes.

Per-node flow:
  1. ZONE PLAN (LLM) — high-level character/intent AND the is_atomic
     decision. Authored at the parent level (before this zone's own
     `_build` pass) so the plan paragraph is in scope when its own
     shell is generated. Root is planned in `run()` before `_build(root)`.
  2. (root only) Overall bounding box (LLM) — sizes the canvas to the
     silhouette implied by the root's plan.
  3. ZONE DECOMPOSE (LLM) — non-atomic zones only. Emits each child as
     a seed (id, prompt, proxy_shape, relationships) in one call. Each
     child's `parent` must resolve (HARD fail via `validate_parents` —
     an orphaned subzone would render nowhere yet still ship a mesh);
     secondary `referenced_ids` are advisory (a dangling peer hint is
     logged and accepted).
  4. Batch-resolve a bbox for EVERY child in one LLM call, then place them.
  5. Encapsulating pass — generates this zone's physical shell / ground.
     Its position relative to decomposition is set by `frame_order`:
       * "after" (default, V3): frame each zone AFTER its own
         decomposition, so the shell author sees the placed child zones.
       * "before" (V2): frame each zone BEFORE it is decomposed — the root
         in `run()` before `_build(root)`, every other zone in its parent's
         per-child loop right after its plan is authored and before
         recursing — so the shell is authored from the zone's plan + bbox
         alone, without yet seeing its internal subdivision.
     Either way each zone is encapsulated exactly once. Atomic zones have
     no decomposition and are framed (per the same flag) right before the
     Phase 2 anchor pass.
  6. If ATOMIC, hand to Phase 2 generation for anchor-object population.
  7. For each placed child, in declaration order: author the child's
     plan (LLM), then recurse.

Root additionally gets a final negative-space pass at the end of the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from app.core import prompt_runtime
from app.core.types import BoundingBox, Node
from app.pipeline import committed, generation
from app.services import llm
from app.utils import logging
from app.utils.topology import validate_parents, validate_referenced_ids


async def _pick_overall_bbox(prompt: str, scene_plan: str) -> BoundingBox:
    hit = committed.bbox("root")
    if hit is not None:
        return hit
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
    hit = committed.zone_plan(zone_id)
    if hit is not None:
        return hit
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
    hit = committed.zone_decompose(node.id)
    if hit is not None:
        return hit
    p = prompt_runtime.current()
    return await llm.call_llm(
        system=p.SYSTEM_ZONE_DECOMPOSE,
        user=p.render_zone_decompose(
            zone_id=node.id,
            zone_prompt=node.prompt,
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
    # Resume: any child already placed (a committed `bbox` event) keeps its
    # exact world position. If every child is committed we skip the LLM
    # entirely; otherwise we resolve the batch and overwrite the committed
    # ones so only never-placed children take a fresh assignment.
    committed_bboxes = {c.id: committed.bbox(c.id) for c in children}
    if all(b is not None for b in committed_bboxes.values()):
        return {cid: b for cid, b in committed_bboxes.items() if b is not None}
    bbox_by_id = {n.id: n.bbox for n in all_nodes}
    p = prompt_runtime.current()
    out = await llm.call_llm(
        system=p.SYSTEM_ZONE_BBOX_BATCH,
        user=p.render_zone_bbox_batch(
            parent_id=parent.id,
            parent_prompt=parent.prompt,
            parent_plan=parent.plan,
            parent_bbox=parent.bbox,
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
    for child_id, b in committed_bboxes.items():
        if b is not None:
            bboxes[child_id] = b
    return bboxes


async def _encapsulate(
    zone: Node,
    *,
    runs_dir: Path,
    run_id: str,
    all_nodes: list[Node],
) -> None:
    """Generate a zone's physical shell / ground (the encapsulating pass).

    Every zone is encapsulated exactly once per run; `frame_order` only
    decides WHEN this fires relative to the zone's own decomposition."""
    logging.emit_step(zone.id, "generating_frame")
    await generation.run(
        zone=zone,
        runs_dir=runs_dir,
        run_id=run_id,
        scenario="encapsulating",
        all_nodes=all_nodes,
    )


async def _build(
    *,
    node: Node,
    runs_dir: Path,
    run_id: str,
    all_nodes: list[Node],
    is_atomic: bool,
    frame_order: Literal["before", "after"],
) -> None:
    assert node.plan is not None, "node.plan must be set by caller"

    placed: list[Node] = []
    if not is_atomic:
        logging.emit_step(node.id, "decomposing")
        decomp = await _decompose_zone(node=node, all_nodes=all_nodes)
        logging.log_once(
            "divider.zone_decompose",
            match_fields=("node",),
            node=node.id,
            children=[c.model_dump() for c in decomp.children],
        )

        existing_ids = {n.id for n in all_nodes}
        # Unresolvable parents are a hard fail — an orphaned subzone would
        # be generated into the scene yet invisible to every later step.
        # `_run` catches this and turns it into a clean run.error.
        validate_parents(
            decomp.children,
            parent_id=node.id,
            existing_ids=existing_ids,
        )
        # Secondary referenced_ids stay advisory: a dangling peer hint is
        # logged (a benchmark signal) but doesn't orphan anything.
        try:
            validate_referenced_ids(
                decomp.children,
                parent_id=node.id,
                existing_ids=existing_ids,
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
                is_zone=True,
            )
            placed.append(child)
            all_nodes.append(child)

    # "after" (V3, default): frame this zone now — AFTER its own
    # decomposition — so the shell author sees the placed child zones. In
    # "before" mode (V2) the zone was already framed by its caller (the root
    # in `run()`; every other zone in its parent's per-child loop) before it
    # knew its own subdivision, so it is not framed again here.
    if frame_order == "after":
        await _encapsulate(node, runs_dir=runs_dir, run_id=run_id, all_nodes=all_nodes)

    if is_atomic:
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

        # "before" (V2): frame the child now — right after its plan is
        # authored and before recursing — so its shell is built from the
        # zone's plan + bbox without yet seeing its internal subdivision.
        if frame_order == "before":
            await _encapsulate(planned, runs_dir=runs_dir, run_id=run_id, all_nodes=all_nodes)

        await _build(
            node=planned,
            runs_dir=runs_dir,
            run_id=run_id,
            all_nodes=all_nodes,
            is_atomic=plan_out.is_atomic,
            frame_order=frame_order,
        )
    logging.emit_step(node.id, "done")


async def run(
    *,
    run_id: str,
    prompt: str,
    model: str,
    runs_dir: Path,
    frame_order: Literal["before", "after"] = "after",
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
        is_zone=True,
    )
    all_nodes: list[Node] = [root]
    # "before" (V2): frame the root here, BEFORE `_build(root)` decomposes it,
    # so the world boundary is authored from the root plan alone. In "after"
    # mode (V3) `_build(root)` frames the root after root's own decomposition.
    if frame_order == "before":
        await _encapsulate(root, runs_dir=runs_dir, run_id=run_id, all_nodes=all_nodes)
    await _build(
        node=root,
        runs_dir=runs_dir,
        run_id=run_id,
        all_nodes=all_nodes,
        is_atomic=plan_out.is_atomic,
        frame_order=frame_order,
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
