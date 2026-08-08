"""Phase 1 — recursive top-down decomposition into a tree of zone Nodes.

Per-node flow:
  1. ZONE PLAN (LLM) — high-level character/intent AND the is_atomic
     decision. Authored at the parent level (before this zone's own
     `_build` pass) so the plan paragraph is in scope when its own
     shell is generated. Root is planned in `run()` before `_build(root)`.
  2. (root only) Overall bounding box (LLM) — sizes the canvas to the
     silhouette implied by the root's plan.
  3. ZONE DECOMPOSE (LLM) — non-atomic zones only. Emits each child as
     a seed (id, prompt, proxy_shape, relationships) in one call. A
     subregion is always contained in this zone, so its parent is fixed
     here (not authored) and there's no orphan failure mode; only the
     advisory checks remain (unique ids, resolvable `referenced_ids` peer
     hints — a dangling hint is logged and accepted). Sibling adjacency is
     expressed through `relationships`, never through parenthood.
  4. Batch-resolve a bbox for EVERY child in one LLM call, then place them.
  5. Encapsulating pass — generates this zone's physical shell / ground,
     run AFTER the zone's own decomposition (for EVERY zone, root and
     non-root alike, inside its own `_build` pass) so the shell author sees
     the subregions just placed inside it. Each zone is encapsulated exactly
     once; atomic zones have no decomposition, so theirs runs right before
     the Phase 2 anchor pass.
  6. If ATOMIC, hand to Phase 2 generation for anchor-object population.
  7. For each placed child, in declaration order: author the child's
     plan (LLM), then recurse.

A negative-space pass then fills the interstitial gaps a zone's named children
don't own: the root always gets one (in `run()`), and every non-root NON-atomic
zone gets one inside `_build` (post-order, after its subtree is built).

Prompt text comes from the run's prompt snapshot (`prompt_store.current()`);
this module only decides which step fires when and with which scene state.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from app.core import prompt_store, scene_context, schemas
from app.core.types import BoundingBox, Node, ParentRelationshipKind
from app.pipeline import committed, context_cull, generation
from app.services import llm
from app.utils import logging
from app.utils.topology import uniquify_ids, validate_subregions

# The Phase-2 entry the walk hands each zone to — `generation.run` by default.
# `divider_parallel` injects `generation.run_parallel` instead, which frames the
# zone but defers its interior; nothing else about the walk changes, so the two
# traversals stay identical by construction rather than by being kept in sync.
# Resolved at call time (never as a default argument) so a variant entry that
# rebinds the attribute is still honoured.
GenRun = Callable[..., Awaitable[None]]


async def _pick_overall_bbox(prompt: str, scene_plan: str) -> BoundingBox:
    hit = committed.bbox("root")
    if hit is not None:
        return hit
    ps = prompt_store.current()
    variables = scene_context.overall_bbox_vars(prompt=prompt, scene_plan=scene_plan)
    out = await llm.call_llm(
        system=ps.system("overall_bbox", variables),
        user=ps.user("overall_bbox", variables),
        output_schema=schemas.OverallBboxOutput,
        node_id="root",
        step="overall_bbox",
        template="overall_bbox",
        variables=variables,
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
        return committed.apply_atomic_lock(zone_id, hit)
    ps = prompt_store.current()
    if not nodes:
        step = "zone_plan_root"
        variables = scene_context.root_seed_vars(prompt=zone_prompt)
        # The root step ("OVERALL SCENE PLAN") emits `plan`; nested regions
        # ("REGION DESCRIPTION") emit `description`. Same shape, different field.
        output_schema = schemas.RootZonePlanOutput
    else:
        step = "zone_plan"
        variables = scene_context.zone_vars(
            zone_id=zone_id,
            zone_prompt=zone_prompt,
            zone_plan=None,
            nodes=context_cull.for_context(nodes, zone_id),
            target_text="This is the region you are to plan and flesh out from.",
        )
        output_schema = schemas.ZonePlanOutput
    out = await llm.call_llm(
        system=ps.system(step, variables),
        user=ps.user(step, variables),
        output_schema=output_schema,
        node_id=zone_id,
        step="zone_plan",
        template=step,
        variables=variables,
    )
    return committed.apply_atomic_lock(zone_id, out)


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
    ps = prompt_store.current()
    step = "zone_decompose_root" if node.parent_id is None else "zone_decompose"
    variables = scene_context.zone_vars(
        zone_id=node.id,
        zone_prompt=node.prompt,
        zone_plan=node.plan,
        nodes=context_cull.for_context(all_nodes, node.id),
        target_text="This is the region you are to break down and decompose.",
    )
    return await llm.call_llm(
        system=ps.system(step, variables),
        user=ps.user(step, variables),
        output_schema=schemas.ZoneDecomposeOutput,
        node_id=node.id,
        step="zone_decompose",
        template=step,
        variables=variables,
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
    by_id = {n.id: n for n in all_nodes}
    ps = prompt_store.current()
    variables = scene_context.zone_vars(
        zone_id=parent.id,
        zone_prompt=parent.prompt,
        zone_plan=parent.plan,
        nodes=context_cull.for_context(all_nodes, parent.id),
        target_text="This is the region whose subregions you are to place.",
    )
    variables["TO_PLACE"] = scene_context.render_to_place_block(
        children, by_id, parent_zone=parent.id,
    )
    out = await llm.call_llm(
        system=ps.system("child_bbox_batch", variables),
        user=ps.user("child_bbox_batch", variables),
        output_schema=schemas.BboxBatchOutput,
        node_id=parent.id,
        step="child_bbox_batch",
        template="child_bbox_batch",
        variables=variables,
        validate=lambda o: llm.require_matching_ids(
            produced=[a.id for a in o.assignments],
            expected=[c.id for c in children],
            step="child_bbox_batch",
        ),
    )
    # Subregions are always contained in this zone, so every child's bbox is
    # authored in the zone's local frame — each converts straight to world
    # against the zone bbox (no per-child parent frame, no topological pass).
    assignments_by_id = {a.id: a.bbox for a in out.assignments}
    bboxes: dict[str, BoundingBox] = {
        child_id: assignment.to_world_frame(parent.bbox)
        for child_id, assignment in assignments_by_id.items()
    }
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
    gen_run: GenRun | None = None,
) -> None:
    """Generate a zone's physical shell / ground (the encapsulating pass).
    Every zone is encapsulated exactly once per run."""
    logging.emit_step(zone.id, "generating_frame")
    await (gen_run or generation.run)(
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
    gen_run: GenRun | None = None,
) -> None:
    assert node.plan is not None, "node.plan must be set by caller"
    gen = gen_run or generation.run

    placed: list[Node] = []
    if not is_atomic:
        logging.emit_step(node.id, "decomposing")
        decomp = await _decompose_zone(node=node, all_nodes=all_nodes)
        existing_ids = {n.id for n in all_nodes}
        # Rename any id colliding with an existing node with an incrementing index
        for old, new in uniquify_ids(decomp.subregions, existing_ids=existing_ids):
            logging.log("divider.id_collision", node=node.id, old=old, new=new)
        logging.log_once(
            "divider.zone_decompose",
            match_fields=("node",),
            node=node.id,
            children=[c.model_dump() for c in decomp.subregions],
        )

        # Subregions always parent to THIS zone (deterministic containment), so
        # there's no orphan / parent-chain failure mode to hard-gate. What's
        # left is advisory — unique ids and resolvable peer-relationship hints;
        # a dangling hint is logged (a benchmark signal) but doesn't orphan.
        try:
            validate_subregions(
                decomp.subregions,
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
        # Resolve every child subregion's bbox in one batch LLM call.
        bboxes = await _resolve_child_bboxes_batch(
            parent=node,
            children=decomp.subregions,
            all_nodes=all_nodes,
        )

        for spec in decomp.subregions:
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
                # A subregion is always contained in the zone it was
                # decomposed from — its parent is fixed here, not authored.
                parent_id=node.id,
                parent_kind=ParentRelationshipKind.IN,
                plan=None,
                is_zone=True,
            )
            placed.append(child)
            all_nodes.append(child)

    # Frame EVERY zone here — AFTER its own decomposition — so the shell /
    # ground author sees the subregions just placed inside it. Atomic zones
    # have no decomposition; this still frames them right before the anchor
    # pass below. Each zone is encapsulated exactly once, in its own pass.
    await _encapsulate(
        node, runs_dir=runs_dir, run_id=run_id, all_nodes=all_nodes, gen_run=gen,
    )

    if is_atomic:
        logging.emit_step(node.id, "generating_anchor")
        await gen(
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

        # The child frames itself AFTER its own decomposition, inside the
        # `_build` pass below — not here.
        await _build(
            node=planned,
            runs_dir=runs_dir,
            run_id=run_id,
            all_nodes=all_nodes,
            is_atomic=plan_out.is_atomic,
            gen_run=gen,
        )

    # Every NON-root non-atomic zone gets a negative-space pass once its whole
    # subtree is realized, filling the interstitial gaps between its child
    # zones/objects. Runs here (post-order, after the children loop) so the
    # pass sees this zone's shell and full interior. The root keeps its pass
    # in `run()`, so it's excluded here to avoid running twice.
    if node.parent_id is not None:
        logging.emit_step(node.id, "generating_negative_space")
        await gen(
            zone=node,
            runs_dir=runs_dir,
            run_id=run_id,
            scenario="negative-space",
            all_nodes=all_nodes,
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
        is_zone=True,
    )
    all_nodes: list[Node] = [root]
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
