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

The planner model is FIXED. Anchors are trusted as-is (no collision / floor
post-processing); the capturing client renders a 360 at each.

A second, separate pass NAMES the planned anchors: a fixed gemini-3.1-flash-lite
call reads the SAME scene context plus the planned coordinates and labels each
point of interest from the zone it sits in and the objects around it. Naming is
split from planning on purpose — the planner emits raw coordinates only, so it's
never biased toward producing fewer points just to label them.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.core import util
from app.core.slots import MODELS
from app.core.types import Node, ProxyShape, Vec3Tuple
from app.services import llm

ANCHOR_PLANNER_MODEL = MODELS["gemini-flash"]  # google/gemini-3.5-flash
ANCHOR_NAMER_MODEL = MODELS["gemini-flash-lite"]  # google/gemini-3.1-flash-lite


SYSTEM_ANCHOR_PLANNER = """\
You are the capture planner for a text-to-3D scene pipeline. A scene has already been built as a tree of regions (zones) and objects, each with an axis-aligned world-space bounding box. Your job is to decide WHERE to place a set of camera "anchor points" from which a 360° panorama will be photographed, so that the set forms a coherent walkthrough of the whole scene.

Coordinate convention (identical to the scene's): right-handed, Y-up, meters.
  +X = right, +Y = up, +Z = toward the viewer (front), -Z = away (back).
A bounding box is given as its world-space dimensions (W by H by D) and a global origin corner; the box spans from that corner along +X/+Y/+Z by the dimensions.

Produce as THOROUGH of a set as possible, inside and out, optimize for FULL scene coverage on all sides and angles. Imagine you are a human walking physically through the scene at discrete points, where do these discrete points need to be in order to experience the scene fully without the discrete jumps being jarring. Transitionatory points are just as important as key highlight points: walking from key location A -> key location B involves one or more important intermediate points even if the image is less interesting at those intermediate points. 

Try to anchor the camera to object top faces. DO NOT place the anchor point inside object bounding boxes. Try to place the anchor points at a realistic viewing height above a surface.

Output ONLY the JSON object matching the schema: each anchor is just its `position`, [x, y, z] world-space meters. Do NOT name, label, classify, or annotate the anchors — emit raw coordinates only."""


SYSTEM_ANCHOR_NAMER = """\
You are the point of interest namer for a text-to-3D scene pipeline. A scene tree has already been fully built from a text prompt (made up of zones and objects) and a planner has picked a list of coordinates for certain points of interest within this scene. Your job is to injest the tree context, understand where the points of interest have been placed; Based on the zone it is in, the objects surrounding it and the overall scene context, provide a name for each point of interest

Output ONLY the JSON object matching the schema: each point of interest is its numeric `id` and string `name`."""


class Anchor(BaseModel):
    position: Vec3Tuple


class AnchorPlan(BaseModel):
    anchors: list[Anchor]


class PointName(BaseModel):
    id: int
    name: str


class PointNames(BaseModel):
    points: list[PointName]


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


def build_namer_prompt(nodes: list[Node], anchors: list[Anchor]) -> str:
    """The planner's scene context (verbatim) plus the planned coordinates,
    each tagged with the numeric id the namer echoes back. The namer reasons
    from the same world geometry to decide what each point overlooks."""
    context = _render_scene_context(nodes)
    points = "\n".join(
        f"id {i}: position "
        f"({a.position[0]:.2f}, {a.position[1]:.2f}, {a.position[2]:.2f}) m"
        for i, a in enumerate(anchors)
    )
    return (
        "Name the points of interest the planner placed in the scene below. "
        "Each is a 360° camera capture coordinate; decide what it overlooks "
        "from its position within the hierarchy.\n\n"
        "=== SCENE HIERARCHY ===\n"
        f"{context}\n"
        "=== END SCENE HIERARCHY ===\n\n"
        "=== POINTS OF INTEREST ===\n"
        f"{points}\n"
        "=== END POINTS OF INTEREST ===\n\n"
        "Now name every point of interest by its id."
    )


async def name_anchors(nodes: list[Node], anchors: list[Anchor]) -> dict[int, str]:
    """Name each planned anchor with the fixed namer model — a separate call so
    naming never biases the planner toward fewer points. Returns {anchor index:
    name}; ids the model omits are simply absent. Standalone: no SlotLog
    binding, no cache, retries unlogged."""
    if not anchors:
        return {}
    llm.set_model(ANCHOR_NAMER_MODEL)
    result, _reasoning, _usage, _raw = await llm.call_llm_once(
        system=SYSTEM_ANCHOR_NAMER,
        user=build_namer_prompt(nodes, anchors),
        output_schema=PointNames,
        model=ANCHOR_NAMER_MODEL,
        step="anchor_name",
        log_retries=False,
    )
    return {p.id: p.name for p in result.points}


async def generate_anchors(nodes: list[Node]) -> tuple[AnchorPlan, str, dict[int, str]]:
    """Plan capture anchors with the fixed planner model, then name them with
    the fixed namer model. Standalone: no SlotLog binding, no cache, retries
    unlogged. Returns (plan, reasoning, names) where `names` maps each anchor's
    list index to its point-of-interest name."""
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
    names = await name_anchors(nodes, plan.anchors)
    return plan, reasoning, names
