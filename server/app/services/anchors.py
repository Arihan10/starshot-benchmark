"""LLM capture-anchor planner — the "other side" of the pipeline.

Where the pipeline turns a prompt into a scene, this turns a finished scene's
hierarchy back into a capture plan: a lightweight model reads the zones/objects
and their world-space bounding boxes — a trimmed view carrying only each node's
identity, proxy shape, orientation, and world-space bbox (parent links, sibling
relationships, parent-local coordinates, and placement prose are all stripped,
since the planner reasons purely from world geometry) — and proposes a large set
of camera anchor points for a Matterport-style 360 walkthrough. Pure spatial
reasoning — decide where a person would stand to photograph the space with good
coverage and overlap — so it doubles as a spatial-reasoning benchmark for the
planner model.

The model is FIXED to gemini-3.1-flash-lite. Anchors are trusted as-is (no
collision / floor post-processing); the capturing client renders a 360 at each.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.core import util
from app.core.slots import MODELS
from app.core.types import Node, ProxyShape, Vec3Tuple
from app.services import llm

ANCHOR_PLANNER_MODEL = MODELS["opus-new"]  # google/gemini-3.5-flash


SYSTEM_ANCHOR_PLANNER = """\
You are the capture planner for a text-to-3D scene pipeline. A scene has already been built as a tree of regions (zones) and objects, each with an axis-aligned world-space bounding box. Your job is to decide WHERE to place a set of camera "anchor points" from which a 360° panorama will be photographed, so that the set forms a coherent walkthrough of the whole scene.

Coordinate convention (identical to the scene's): right-handed, Y-up, meters.
  +X = right, +Y = up, +Z = toward the viewer (front), -Z = away (back).
A bounding box is given as its world-space dimensions (W by H by D) and a global origin corner; the box spans from that corner along +X/+Y/+Z by the dimensions.

Produce as THOROUGH of a set as possible, inside and out, optimize for FULL scene coverage on all sides and angles in terms of line of sight. Try to anchor the camera to certain object top faces. Be certain not to place the anchor point within any bounding boxes of objects. Try to place the anchor points at a realistic viewing height above a surface, do not place the anchor point randomly in the air.

Output ONLY the JSON object matching the schema: each anchor is just its `position`, [x, y, z] world-space meters. Do NOT name, label, classify, or annotate the anchors — emit raw coordinates only."""


class Anchor(BaseModel):
    position: Vec3Tuple


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


# Trimmed scene context for the planner only. Mirrors the pipeline's own tree
# walk but emits just identity + world-space geometry per node: parent links,
# relationships, parent-local coordinates, and placement prose are dropped so
# the planner reasons purely from world boxes. Kept local so the main pipeline's
# context renderers stay untouched.


def _proxy_shape(p: ProxyShape | None) -> str:
    return p.value if p is not None else "BOX"


def _object_entry(obj: Node) -> str:
    return util.braces(
        "\n".join(
            [
                f"Name: {obj.id}",
                f'Description: "{obj.prompt}"',
                f"proxy_shape: {_proxy_shape(obj.proxy_shape)}",
                f"orientation: {obj.orientation}deg",
                f"Dimensions: {util.format_dimensions(obj.bbox)}",
                f"Global origin corner: {util.format_global_origin(obj.bbox)}",
            ]
        )
    )


def _region_entry(
    region: Node,
    idx: dict[str | None, list[Node]],
    oidx: dict[str | None, list[Node]],
) -> str:
    objects, subregions = util.split_region_members_owned(region.id, idx, oidx)
    lines = [
        f"Subregion name: {region.id}",
        f'Description: "{region.prompt}"',
    ]
    if region.plan is not None:
        lines.append(f'Plan for this region: "{region.plan}"')
    lines.append(f"proxy_shape: {_proxy_shape(region.proxy_shape)}")
    lines.append(f"Dimensions: {util.format_dimensions(region.bbox)}")
    lines.append(f"Global origin corner: {util.format_global_origin(region.bbox)}")
    if objects:
        lines += [
            "",
            f'Objects placed directly within "{region.id}":',
            "",
            util.brace_group([_object_entry(o) for o in objects]),
        ]
    if subregions:
        lines += [
            "",
            f'Subregions within "{region.id}":',
            "",
            util.brace_group([_region_entry(s, idx, oidx) for s in subregions]),
        ]
    return util.braces("\n".join(lines))


def _render_scene_context(nodes: list[Node]) -> str:
    """Trimmed scene hierarchy for the planner: root header + root-level shared
    geometry + the subregion/object tree, each node carrying only identity,
    proxy shape, orientation, and world-space bbox."""
    root = util.find_root(nodes)
    if root is None:
        return ""
    idx = util.index_children(nodes)
    oidx = util.index_objects_by_region(nodes)
    dx, dy, dz = root.bbox.dimensions
    ox, oy, oz = root.bbox.origin
    parts = [
        f'Prompt: "{root.prompt}"\n'
        f'Plan: "{root.plan}"\n'
        f"Overall scene (root) bounding box: {dx:.2f}m by {dy:.2f}m by {dz:.2f}m, "
        f"with its origin corner at ({ox:.2f}, {oy:.2f}, {oz:.2f}) m (root)"
    ]
    root_objects = oidx.get(root.id, [])
    if root_objects:
        parts.append(
            "Objects in the root region (shared shell / ground geometry):\n\n"
            + util.brace_group([_object_entry(o) for o in root_objects])
        )
    _, subregions = util.split_region_members(root.id, idx)
    if subregions:
        parts.append(util.brace_group([_region_entry(s, idx, oidx) for s in subregions]))
    return "\n\n".join(parts)


def build_user_prompt(nodes: list[Node]) -> str:
    context = _render_scene_context(nodes)
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
