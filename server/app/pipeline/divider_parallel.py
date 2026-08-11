"""Phase 1 — the OVERLAPPED variant of the top-down decomposition.

Builds the exact same tree as `divider.run`, through the exact same helpers, in
the exact same depth-first order: the walk IS `divider._build`, just handed
`generation.run_parallel` instead of `generation.run`. Nothing about the
structure changes — only WHEN each zone's interior is built.

The walk stays serial and keeps every blocking structural step, because a zone's
plan and bbox genuinely depend on the zones already placed. What comes off the
critical path is the interior work:

  * A zone's ANCHOR pass no longer blocks the walk. It is queued, and starts as
    soon as `generation.neighbours_framed` says everything on that zone's
    boundary has resolved — to an atomic zone whose framing is done, or to
    negative space. So it waits for its own surroundings rather than for the
    whole tree, and it runs while the walk carries on elsewhere.
  * NEGATIVE SPACE waits for all of it. A zone's fill covers the gaps between
    its children, so it has to see their interiors; `run_deferred_interiors`
    runs those deepest-level-first once the anchors are in.

The gate is evaluated by the walk, synchronously, right after each shell commits
— never inside the queued task. That is what keeps the run reproducible: the LLM
cache is keyed on the rendered prompt, so a scene snapshot taken at whatever
moment the event loop woke a waiter would fork the cache key between runs and
cost the cell its replay. Read from the walk, what a zone sees depends only on
how far the walk had got when its surroundings settled.

For the same reason the queued passes never write into the live scene. Their
objects merge back in `drain_anchors`, so the walk's own prompts stay free of
interior objects — smaller, and identical run to run.

What this trades away is cross-zone visibility. A zone's objects see the shells
around them but not what a peer put inside itself, so an object anchored to
another zone's INTERIOR object loses its anchor; on a measured modern-house run
55 of 66 cross-zone anchors pointed at a frame and were unaffected. Releasing
early narrows it further: a zone placed later in the walk, in a branch not yet
reached, is not on the boundary of anything already released.

Resume behaves as it does for the BFS variant: every helper still consults
`committed.*` on structural identity and the `emit_*` / `log_once` helpers dedup,
so re-walking replays committed work verbatim. A cell part-built by the SERIAL
pipeline should not be resumed here — see `generation.run_parallel`.

Worth pairing with the BREADTH-FIRST walk. `_build` resolves every child's bbox
up front but plans them one at a time, so under depth-first a sibling sits placed
and unplanned — which blocks — for as long as the walk spends in the branch
before it. Breadth-first plans each level together, so that state is brief and
the gates open far earlier.

Launched by `scripts/run_parallel.py` via `app.main_parallel`.
"""

from __future__ import annotations

from pathlib import Path

from app.core.types import Node
from app.pipeline import generation
from app.pipeline.divider import _build, _pick_overall_bbox, _plan_zone
from app.services import llm
from app.utils import logging


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

    # The structural walk: every zone planned, decomposed and framed, in serial
    # order. Anchor passes start off to the side as the walk frames enough around
    # them, so most interiors are already building by the time this returns.
    await _build(
        node=root,
        runs_dir=runs_dir,
        run_id=run_id,
        all_nodes=all_nodes,
        is_atomic=plan_out.is_atomic,
        gen_run=generation.run_parallel,
    )

    # Join the anchors and fold what they placed back into the scene. Anything
    # the walk never managed to release is released here.
    await generation.drain_anchors(
        runs_dir=runs_dir,
        run_id=run_id,
        all_nodes=all_nodes,
    )

    # Negative space last, once every named object it has to fill around exists.
    logging.emit_step(root.id, "generating_negative_space")
    await generation.run_parallel(
        zone=root,
        runs_dir=runs_dir,
        run_id=run_id,
        scenario="negative-space",
        all_nodes=all_nodes,
    )
    await generation.run_deferred_interiors(
        runs_dir=runs_dir,
        run_id=run_id,
        all_nodes=all_nodes,
    )
    logging.emit_step(root.id, "done")
    return root
