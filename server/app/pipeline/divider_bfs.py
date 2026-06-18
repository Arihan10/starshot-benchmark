"""Phase 1 — BREADTH-first variant of the top-down decomposition.

This is an experimental alternative to `divider.run`. It builds the exact
same tree of zone Nodes using the exact same per-step building blocks
(`_plan_zone`, `_pick_overall_bbox`, `_decompose_zone`,
`_resolve_child_bboxes_batch`, `_encapsulate`, and `generation.run`) — only
the *traversal order* differs. Where `divider._build` recurses depth-first
(a child's whole subtree is realized before its next sibling is even
planned), this walks the tree level by level so every region is developed to
the same depth before anyone goes deeper. The intent is an evenly-blocked-out
scene (and a cleaner spatial-reasoning benchmark signal), not a faster one.

Per level, the four pipeline steps run as four separate breadth-first passes
(so each pass sees every sibling developed to the same resolution):

  1. PLAN     — plan every zone on the level (sets each zone's plan AND its
                is_atomic decision).
  2. DECOMPOSE— decompose every NON-atomic zone into child subregions + place
                their bboxes; atomic zones are skipped. The children produced
                here are the next level's frontier.
  3. FRAME    — encapsulate (shell / ground) every zone on the level. Runs
                after the whole decompose pass so a shell author sees not just
                its own subregions but every sibling's too.
  4. ANCHOR   — for every ATOMIC zone on the level, generate its anchor
                objects + run the completion loop. Runs after the whole frame
                pass so each leaf's object pass sees all of its siblings'
                shells.

The root is special exactly as in the DFS divider: it's planned from the bare
prompt, sized by the overall-bbox step, then decomposed + framed before the
level walk begins over its children.

Negative space cannot be a per-level pass — a zone's pass needs its COMPLETE
interior, which in a level walk only exists once the deepest level is done. So
it runs as a final post-order sweep (deepest zones first, root last), which
reproduces the order the DFS divider gets for free while keeping the rest of
the walk breadth-first. Atomic zones get no negative-space pass; the root
always gets one — both matching `divider.run`.

Resume is unaffected: every leaf helper still consults `committed.*` keyed on
structural identity, and the idempotent `emit_*`/`log_once` helpers dedup any
re-emitted event, so re-walking in BFS order replays committed work verbatim.

Selected via a dedicated launch entry; the production pipeline keeps using
`divider.run`.
"""

from __future__ import annotations

from pathlib import Path

from app.core.types import Node, ParentRelationshipKind
from app.pipeline import generation
from app.pipeline.divider import (
    _decompose_zone,
    _encapsulate,
    _pick_overall_bbox,
    _plan_zone,
    _resolve_child_bboxes_batch,
)
from app.services import llm
from app.utils import logging
from app.utils.topology import validate_subregions


async def _decompose_and_place(
    *,
    node: Node,
    all_nodes: list[Node],
) -> list[Node]:
    """Decompose one planned, non-atomic zone into child subregions, resolve
    every child's bbox in one batch, and append the children to `all_nodes`.
    Returns the newly created child Nodes (each with `plan=None`).

    This is the non-atomic branch of `divider._build` lifted out verbatim so it
    can run as its own breadth-first pass — keep it in step with that branch."""
    assert node.plan is not None, "zone must be planned before decomposition"
    logging.emit_step(node.id, "decomposing")
    decomp = await _decompose_zone(node=node, all_nodes=all_nodes)
    logging.log_once(
        "divider.zone_decompose",
        match_fields=("node",),
        node=node.id,
        children=[c.model_dump() for c in decomp.subregions],
    )

    existing_ids = {n.id for n in all_nodes}
    # Subregions always parent to THIS zone, so there's no orphan failure mode;
    # the check is advisory (unique ids, resolvable peer hints) and a dangling
    # hint is logged and accepted rather than gated — same as the DFS divider.
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
    bboxes = await _resolve_child_bboxes_batch(
        parent=node,
        children=decomp.subregions,
        all_nodes=all_nodes,
    )

    placed: list[Node] = []
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
            parent_id=node.id,
            parent_kind=ParentRelationshipKind.IN,
            plan=None,
            is_zone=True,
        )
        placed.append(child)
        all_nodes.append(child)
    return placed


async def _walk_levels(
    *,
    frontier: list[Node],
    all_nodes: list[Node],
    runs_dir: Path,
    run_id: str,
) -> list[Node]:
    """Walk the tree breadth-first starting from `frontier` (the root's
    children). Returns every NON-atomic zone it framed, in frame order, so the
    caller can run their negative-space passes post-order afterwards."""
    framed_nonatomic: list[Node] = []
    while frontier:
        # --- PASS 1: PLAN every zone on this level. Each plan is replaced into
        # `all_nodes` immediately so a later sibling's plan call sees the
        # earlier siblings' plans in scene context (sequential-abreast). ---
        planned: list[Node] = []
        is_atomic: dict[str, bool] = {}
        for stub in frontier:
            logging.emit_step(stub.id, "planning")
            plan_out = await _plan_zone(
                zone_id=stub.id,
                zone_prompt=stub.prompt,
                nodes=all_nodes,
            )
            node = stub.model_copy(update={"plan": plan_out.plan})
            all_nodes[all_nodes.index(stub)] = node
            logging.log_once(
                "divider.zone_plan",
                match_fields=("node",),
                node=node.id,
                plan=plan_out.plan,
                is_atomic=plan_out.is_atomic,
            )
            planned.append(node)
            is_atomic[node.id] = plan_out.is_atomic

        # --- PASS 2: DECOMPOSE every non-atomic zone; collect the next level. ---
        next_frontier: list[Node] = []
        for node in planned:
            if is_atomic[node.id]:
                continue
            next_frontier.extend(
                await _decompose_and_place(node=node, all_nodes=all_nodes)
            )

        # --- PASS 3: FRAME every zone on this level. ---
        for node in planned:
            await _encapsulate(
                node, runs_dir=runs_dir, run_id=run_id, all_nodes=all_nodes
            )
            if not is_atomic[node.id]:
                framed_nonatomic.append(node)

        # --- PASS 4: ANCHOR every atomic (leaf) zone on this level. ---
        for node in planned:
            if not is_atomic[node.id]:
                continue
            logging.emit_step(node.id, "generating_anchor")
            await generation.run(
                zone=node,
                runs_dir=runs_dir,
                run_id=run_id,
                scenario="anchor",
                all_nodes=all_nodes,
            )
            logging.emit_step(node.id, "done")

        frontier = next_frontier
    return framed_nonatomic


async def run(
    *,
    run_id: str,
    prompt: str,
    model: str,
    runs_dir: Path,
) -> Node:
    llm.set_model(model)

    # --- Level 0: the root. Plan from the bare prompt, size with the
    # overall-bbox step, then decompose + frame — same as `divider.run`; only
    # the recursion over its children (below) is breadth-first. ---
    logging.emit_step("root", "planning")
    plan_out = await _plan_zone(zone_id="root", zone_prompt=prompt, nodes=[])
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

    # Zones that get a negative-space pass, in frame order (root first); swept
    # in REVERSE at the end so a parent's pass runs after its children's — the
    # post-order the DFS divider gets for free. Root is always included
    # (matching `divider.run`, which runs root's pass unconditionally); non-root
    # atomic leaves never are.
    negspace_zones: list[Node] = [root]

    if plan_out.is_atomic:
        # Degenerate scene: the root is itself the single atomic zone.
        await _encapsulate(root, runs_dir=runs_dir, run_id=run_id, all_nodes=all_nodes)
        logging.emit_step("root", "generating_anchor")
        await generation.run(
            zone=root,
            runs_dir=runs_dir,
            run_id=run_id,
            scenario="anchor",
            all_nodes=all_nodes,
        )
    else:
        frontier = await _decompose_and_place(node=root, all_nodes=all_nodes)
        await _encapsulate(root, runs_dir=runs_dir, run_id=run_id, all_nodes=all_nodes)
        negspace_zones.extend(
            await _walk_levels(
                frontier=frontier,
                all_nodes=all_nodes,
                runs_dir=runs_dir,
                run_id=run_id,
            )
        )

    # --- Final pass: negative-space sweep, post-order (deepest first, root
    # last). The whole tree exists by now, so each zone sees its complete
    # interior. ---
    for zone in reversed(negspace_zones):
        logging.emit_step(zone.id, "generating_negative_space")
        await generation.run(
            zone=zone,
            runs_dir=runs_dir,
            run_id=run_id,
            scenario="negative-space",
            all_nodes=all_nodes,
        )
        logging.emit_step(zone.id, "done")
    return root
