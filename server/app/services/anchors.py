"""LLM capture-anchor planner — the "other side" of the pipeline.

Where the pipeline turns a prompt into a scene, this turns a finished scene's
hierarchy back into a capture plan: a lightweight model reads the zones/objects
and their world-space bounding boxes (the SAME canonical scene-context the
divider/generation steps are handed) and proposes a large set of camera anchor
points for a Matterport-style 360 walkthrough. Pure spatial reasoning — decide
where a person would stand to photograph the space with good coverage and
overlap — so it doubles as a spatial-reasoning benchmark for the planner model.

The model is FIXED to gemini-3.1-flash-lite. Anchors are trusted as-is (no
collision / floor post-processing); the capturing client renders a 360 at each.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.core import prompts
from app.core.slots import MODELS
from app.core.types import Node, Vec3Tuple
from app.services import llm

ANCHOR_PLANNER_MODEL = MODELS["opus-new"]  # google/gemini-3.5-flash


SYSTEM_ANCHOR_PLANNER = """\
You are the capture planner for a text-to-3D scene pipeline. A scene has already been built as a tree of regions (zones) and objects, each with an axis-aligned world-space bounding box. Your job is to decide WHERE to place a set of camera "anchor points" from which a 360° panorama will be photographed, so that the set forms a coherent walkthrough of the whole scene.

Coordinate convention (identical to the scene's): right-handed, Y-up, meters.
  +X = right, +Y = up, +Z = toward the viewer (front), -Z = away (back).
A bounding box is given as its world-space dimensions (W by H by D) and a global origin corner; the box spans from that corner along +X/+Y/+Z by the dimensions.

For each anchor also choose a `look_at` world point — an interesting focal target the initial view should face (a centerpiece object, the depth of a room, a vista). It does not constrain the capture (each is a full 360°); it only sets the starting view.

Produce as THOROUGH of a set as possible, inside and out, optimize for FULL scene coverage on all sides and angles in terms of line of sight. Try to anchor the camera to certain object top faces. Be certain not to place the camera within objects and try to place cameras at a realistic viewing height.

Output ONLY the JSON object matching the schema: each anchor is just its `position` and `look_at`, both [x, y, z] world-space meters. Do NOT name, label, classify, or annotate the anchors — emit raw coordinates only."""


class Anchor(BaseModel):
    position: Vec3Tuple
    look_at: Vec3Tuple


class AnchorPlan(BaseModel):
    anchors: list[Anchor]


def _scene_aabb(nodes: list[Node]) -> tuple[Vec3Tuple, Vec3Tuple]:
    """World-space min/max corner spanning every node's bbox."""
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for n in nodes:
        lo = n.bbox.min_corner
        hi = n.bbox.max_corner
        for i in range(3):
            mins[i] = min(mins[i], lo[i])
            maxs[i] = max(maxs[i], hi[i])
    if mins[0] == float("inf"):
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    return (mins[0], mins[1], mins[2]), (maxs[0], maxs[1], maxs[2])


def build_user_prompt(nodes: list[Node]) -> str:
    context = prompts.render_full_scene_context(nodes)
    (lo_x, lo_y, lo_z), (hi_x, hi_y, hi_z) = _scene_aabb(nodes)
    return (
        "Plan the 360° capture anchor points for the scene below.\n\n"
        f"Scene world bounds: min corner ({lo_x:.2f}, {lo_y:.2f}, {lo_z:.2f}) m, "
        f"max corner ({hi_x:.2f}, {hi_y:.2f}, {hi_z:.2f}) m. "
        "=== SCENE HIERARCHY ===\n"
        f"{context}\n"
        "=== END SCENE HIERARCHY ===\n\n"
        "Now produce the anchor set."
    )


async def generate_anchors(nodes: list[Node]) -> tuple[AnchorPlan, str]:
    """Plan capture anchors for `nodes` with the fixed planner model. Standalone:
    no SlotLog binding, no cache, retries unlogged. Returns (plan, reasoning)."""
    # Set the model context too, so llm._normalize_schema applies any
    # provider-specific schema tweaks for the planner provider.
    llm.set_model(ANCHOR_PLANNER_MODEL)
    plan, reasoning, _usage, _raw = await llm.call_llm_once(
        system=SYSTEM_ANCHOR_PLANNER,
        user=build_user_prompt(nodes),
        output_schema=AnchorPlan,
        model=ANCHOR_PLANNER_MODEL,
        step="anchor_plan",
        log_retries=False,
    )
    return plan, reasoning
