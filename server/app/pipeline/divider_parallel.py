"""Phase 1 — the SPLIT-PHASE variant of the top-down decomposition.

Builds the exact same tree as `divider.run`, through the exact same helpers, in
the exact same depth-first order: the walk IS `divider._build`, just handed
`generation.run_parallel` instead of `generation.run`. Nothing about the
structure changes — only WHEN each zone's interior is built.

  PHASE A (this walk, serial)
      Every zone is planned, decomposed and FRAMED. `run_parallel` executes the
      encapsulating pass inline and records the anchor / negative-space passes
      rather than running them, so the walk ends on a fully planned, fully
      framed scene whose regions are all still empty.

  PHASE B (`generation.run_deferred_interiors`)
      Every recorded interior is built with the zones fanned out concurrently:
      the atomic zones in one wave, then negative space deepest-level-first
      (a parent's fill has to see its children's, since its region contains
      theirs).

Phase A stays serial because a zone's plan and bbox genuinely depend on the
zones already placed. Phase B parallelises because its work does not: by then
every envelope is fixed by the parent's `child_bbox_batch`, so no two zone
interiors can contend for the same space.

What this trades away is cross-zone visibility. A zone's objects no longer see
what a peer zone put inside itself, only the framed shell — so an object
anchored to another zone's INTERIOR object loses its anchor. Objects anchored to
a frame are unaffected, because Phase A places every frame before Phase B
starts, and on a measured modern-house run that was 55 of the 66 cross-zone
anchors.

Resume behaves as it does for the BFS variant: every helper still consults
`committed.*` on structural identity and the `emit_*` / `log_once` helpers dedup,
so re-walking replays committed work verbatim. A cell part-built by the SERIAL
pipeline should not be resumed here — see `generation.run_parallel`.

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

    # PHASE A — the whole tree planned, decomposed and framed; interiors recorded.
    await _build(
        node=root,
        runs_dir=runs_dir,
        run_id=run_id,
        all_nodes=all_nodes,
        is_atomic=plan_out.is_atomic,
        gen_run=generation.run_parallel,
    )
    logging.emit_step(root.id, "generating_negative_space")
    await generation.run_parallel(
        zone=root,
        runs_dir=runs_dir,
        run_id=run_id,
        scenario="negative-space",
        all_nodes=all_nodes,
    )

    # PHASE B — build every deferred interior, zones concurrently. The root's
    # `done` is held until after this so the tree is never reported complete
    # while its regions are still empty.
    await generation.run_deferred_interiors(
        runs_dir=runs_dir,
        run_id=run_id,
        all_nodes=all_nodes,
    )
    logging.emit_step(root.id, "done")
    return root
