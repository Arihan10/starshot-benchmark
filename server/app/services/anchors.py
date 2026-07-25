"""LLM capture-anchor planner — the "other side" of the pipeline.

Where the pipeline turns a prompt into a scene, this turns a finished scene's
hierarchy back into a capture plan for a Matterport-style 360 walkthrough. It
runs one planning call PER ZONE: each call sees only that zone's own objects —
but the identity (id + bbox) of every zone in the scene — and emits, for that
zone:

  * anchors    — camera points for a 360° walkthrough of the zone (raw [x, y, z],
                 trusted as-is; no collision / floor post-processing).
  * connectors — the objects in the zone that lead into another zone (doors,
                 gateways, staircases, ladders, …), each naming the object `id`
                 and the `target_zone` it transitions into. All-encompassing: a
                 staircase contributes its treads, handrails, and supports, each
                 pointing at the same target_zone.

Descoping each call to a single zone (while still passing every zone's identity)
keeps the spatial reasoning local and lets the planner mark cross-zone
transitions. The planner model is FIXED; zones are planned sequentially, each call receiving
the anchors already placed in earlier zones so it avoids overlapping them.

A second, separate pass NAMES the aggregated anchors: a fixed gemini-3.1-flash-lite
call reads the full scene context plus the planned coordinates and labels each
point of interest. Naming is split from planning on purpose — the planner emits
raw coordinates only, so it's never biased toward producing fewer points just to
label them.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.core import util
from app.core.slots import MODELS
from app.core.types import BoundingBox, Node, ProxyShape, Vec3Tuple
from app.services import llm

ANCHOR_PLANNER_MODEL = MODELS["gemini-flash"]  # google/gemini-3.5-flash
ANCHOR_NAMER_MODEL = MODELS["gemini-flash-lite"]  # google/gemini-3.1-flash-lite


SYSTEM_ANCHOR_PLANNER = """\
You are the capture planner for a text-to-3D scene pipeline. A scene has already been built as a tree of regions (zones) and objects, each with an axis-aligned world-space bounding box. You plan the capture for ONE zone at a time: you are given that zone's own objects, plus every other zone that lives within the scene. For the given zone you produce two things — the camera "anchor points" for a 360° walkthrough of the zone, and the "connectors" that lead out of it into adjacent zones.

ANCHORS - Produce as THOROUGH of a set as possible for THIS zone, optimizing for full spatial coverage of the zone. Imagine you are a human walking physically through the zone at discrete points: where do those points need to be to experience the zone fully without the jumps being jarring but remember each point of movement incurs a delay that may ruin the user experience if there are too many, as a general rule of thumb, don't space points closer than 2m together. Transitionatory points are just as important as key highlight points: moving from key location A -> key location B involves one or more important intermediate points even if the image is less interesting there. Try to anchor the camera to object top faces. DO NOT place an anchor inside an object's bounding box. Place anchors at a realistic viewing height above a surface. You may also be given a list of anchors ALREADY PLACED in other zones; treat those positions as taken: do NOT place a new anchor at or near any of them, and instead fill the gaps they leave.

CONNECTORS — Some objects in this zone are how a visitor LEAVES it for an adjacent zone: doors, gates, staircases, ladders, elevators, etc. For each such object emit a connector with the object's `id` and the `target_zone` it leads into. `target_zone` MUST be the id of one of the zones in the provided zone list. Connectors are ALL-ENCOMPASSING: when a transition is a composite assembly, emit a connector for EVERY object that makes it up — e.g. for a staircase emit one for the steps/treads, one for each handrail, and one for the stringers/supports, all pointing at the SAME target_zone. all connectors and connector composite objects will be specially highlighted in the envronment, reason about what makes sense to include in the composite objects to make the highlights elegant, for example if a door has a header piece, ignore it as it doesn't need to shown as "part of the door". Only reference object ids from THIS zone's object list. If the zone has no way out, return an empty connectors list.

Coordinate convention (identical to the scene's): right-handed, Y-up, meters.
  +X = right, +Y = up, +Z = toward the viewer (front), -Z = away (back).
A bounding box is given as its world-space dimensions (W by H by D) and a global origin corner; the box spans from that corner along +X/+Y/+Z by the dimensions.

Output ONLY the JSON object matching the schema: `anchors` (each with a `position`) and `connectors` (each with an `id` and a `target_zone`)."""


SYSTEM_ANCHOR_NAMER = """\
You are the point of interest namer for a text-to-3D scene pipeline. A scene tree has already been fully built from a text prompt (made up of zones and objects) and a planner has picked a list of coordinates for certain points of interest within this scene. Your job is to injest the tree context, understand where the points of interest have been placed; Based on the zone it is in, the objects surrounding it and the overall scene context, provide a name for each point of interest

Output ONLY the JSON object matching the schema: each point of interest is its numeric `id` and string `name`."""


class Anchor(BaseModel):
    position: Vec3Tuple


class PlacedAnchor(Anchor):
    """An aggregated anchor tagged with the zone whose pass produced it. `zone`
    is assigned programmatically here (the planner only emits the position); it
    rides through the /anchors response into tour.json so the client knows which
    zone each capture point belongs to."""

    zone: str


class Connector(BaseModel):
    id: str
    target_zone: str


class PlacedConnector(Connector):
    """A connector tagged with the zone it came FROM (its owning zone's pass),
    assigned programmatically here — the planner only emits the object `id` and
    the `target_zone` it leads into. Rides through the /anchors response into
    tour.json beside `target_zone`."""

    starting_zone: str


class ZoneAnchorPlan(BaseModel):
    anchors: list[Anchor]
    connectors: list[Connector]


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


def _zone_brief(region: Node) -> str:
    """Identity-only entry for the connector target list: id + description + bbox,
    no objects (a zone's objects are shown only when it is the zone being
    planned)."""
    lines = [f"Zone id: {region.id}", f'Description: "{region.prompt}"']
    if region.plan is not None:
        lines.append(f'Plan: "{region.plan}"')
    lines.append(f"Dimensions: {util.format_dimensions(region.bbox)}")
    lines.append(f"Global origin corner: {util.format_global_origin(region.bbox)}")
    return util.braces("\n".join(lines))


def build_zone_prompt(
    nodes: list[Node],
    zone: Node,
    zone_objects: list[Node],
    regions: list[Node],
    placed_anchors: list[PlacedAnchor],
) -> str:
    """Descoped planner context for ONE zone: the scene header + the identity of
    every zone (so connectors can name a `target_zone`), object detail for only
    the zone being planned, and the anchors already placed in earlier zones (so
    this zone's anchors don't overlap them)."""
    root = util.find_root(nodes)
    (lo_x, lo_y, lo_z), (hi_x, hi_y, hi_z) = _scene_aabb(nodes)
    dx, dy, dz = root.bbox.dimensions
    ox, oy, oz = root.bbox.origin

    zone_lines = [f"Zone id: {zone.id}", f'Description: "{zone.prompt}"']
    if zone.plan is not None:
        zone_lines.append(f'Plan for this zone: "{zone.plan}"')
    zone_lines.append(f"Dimensions: {util.format_dimensions(zone.bbox)}")
    zone_lines.append(f"Global origin corner: {util.format_global_origin(zone.bbox)}")
    objects_block = (
        util.brace_group([_object_entry(o) for o in zone_objects])
        if zone_objects
        else "{}"
    )
    placed_block = ""
    if placed_anchors:
        positions = "\n".join(
            f"({a.position[0]:.2f}, {a.position[1]:.2f}, {a.position[2]:.2f}) m"
            for a in placed_anchors
        )
        placed_block = (
            "=== ANCHORS ALREADY PLACED IN OTHER ZONES (do not place new anchors "
            "at or near these) ===\n"
            f"{positions}\n"
            "=== END ALREADY PLACED ===\n\n"
        )

    return (
        "Plan the 360° capture anchor points and connectors for ONE zone of the "
        "scene below.\n\n"
        f'Scene prompt: "{root.prompt}"\n'
        f'Scene plan: "{root.plan}"\n'
        f"Overall scene (root) bounding box: {dx:.2f}m by {dy:.2f}m by {dz:.2f}m, "
        f"origin corner at ({ox:.2f}, {oy:.2f}, {oz:.2f}) m.\n"
        f"Scene world bounds: min corner ({lo_x:.2f}, {lo_y:.2f}, {lo_z:.2f}) m, "
        f"max corner ({hi_x:.2f}, {hi_y:.2f}, {hi_z:.2f}) m.\n\n"
        "=== ALL ZONES (use these ids as connector target_zone values) ===\n"
        f"{util.brace_group([_zone_brief(r) for r in regions])}\n"
        "=== END ALL ZONES ===\n\n"
        "=== ZONE TO PLAN ===\n" + "\n".join(zone_lines) + "\n\n"
        "Objects in this zone (place anchors among these; every connector id must "
        "be one of these objects):\n\n"
        f"{objects_block}\n"
        "=== END ZONE TO PLAN ===\n\n"
        f"{placed_block}"
        "Now produce the anchors and connectors for THIS zone."
    )


def build_namer_prompt(nodes: list[Node], anchors: list[PlacedAnchor]) -> str:
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


async def name_anchors(nodes: list[Node], anchors: list[PlacedAnchor]) -> dict[int, str]:
    """Name each planned anchor with the fixed namer model — a separate call so
    naming never biases the planner toward fewer points. Returns {anchor index:
    name}; ids the model omits are simply absent. Standalone: no SlotLog
    binding, no cache, retries unlogged."""
    if not anchors:
        return {}
    llm.set_model(ANCHOR_NAMER_MODEL)
    result, _reasoning, _usage, _raw, _gen_ids = await llm.call_llm_once(
        system=SYSTEM_ANCHOR_NAMER,
        user=build_namer_prompt(nodes, anchors),
        output_schema=PointNames,
        model=ANCHOR_NAMER_MODEL,
        step="anchor_name",
        log_retries=False,
    )
    return {p.id: p.name for p in result.points}


async def _plan_zone(
    nodes: list[Node],
    zone: Node,
    zone_objects: list[Node],
    regions: list[Node],
    placed_anchors: list[PlacedAnchor],
) -> tuple[ZoneAnchorPlan, str]:
    """Plan anchors + connectors for a single zone. `placed_anchors` are the
    anchors already chosen for earlier zones, passed in so the planner doesn't
    overlap them. Sets the planner model so schema normalization sees the planner
    provider."""
    llm.set_model(ANCHOR_PLANNER_MODEL)
    plan, reasoning, _usage, _raw, _gen_ids = await llm.call_llm_once(
        system=SYSTEM_ANCHOR_PLANNER,
        user=build_zone_prompt(nodes, zone, zone_objects, regions, placed_anchors),
        output_schema=ZoneAnchorPlan,
        model=ANCHOR_PLANNER_MODEL,
        step="anchor_plan",
        log_retries=False,
    )
    return plan, reasoning


def _region_depth(node: Node, by_id: dict[str, Node]) -> int:
    """Tree depth of a region (root = 0), counted by walking `parent_id` to root."""
    depth = 0
    seen = {node.id}
    cur = node
    while cur.parent_id is not None:
        parent = by_id.get(cur.parent_id)
        if parent is None or parent.id in seen:
            break
        seen.add(parent.id)
        cur = parent
        depth += 1
    return depth


def _bbox_contains(bbox: BoundingBox, p: Vec3Tuple, eps: float = 1e-3) -> bool:
    """Whether world-space point `p` lies inside `bbox` (tiny eps absorbs FP slack)."""
    lo, hi = bbox.min_corner, bbox.max_corner
    return all(lo[i] - eps <= p[i] <= hi[i] + eps for i in range(3))


def _inhabiting_zone(
    p: Vec3Tuple, regions: list[Node], depth_of: dict[str, int]
) -> Node | None:
    """The region a point mathematically inhabits: the DEEPEST region whose bbox
    contains `p` (ties broken by smallest bbox). None when no region contains it."""
    best: Node | None = None
    best_key: tuple[int, float] | None = None
    for r in regions:
        if not _bbox_contains(r.bbox, p):
            continue
        sx, sy, sz = r.bbox.size
        key = (depth_of[r.id], -(sx * sy * sz))
        if best_key is None or key > best_key:
            best, best_key = r, key
    return best


async def generate_anchors(
    nodes: list[Node],
) -> tuple[list[PlacedAnchor], list[PlacedConnector], str, dict[int, str]]:
    """Plan capture anchors + connectors one zone at a time (a fixed-planner call
    per zone, run sequentially), then name the aggregated anchors with the fixed
    namer model. Each zone call sees only its own objects, every zone's identity,
    and the anchors already placed in earlier zones — so it can mark the objects
    that connect into another zone and avoid overlapping prior anchors. Zones are
    planned deepest-first. Each returned anchor is stamped with the `zone` it
    mathematically inhabits (the deepest region whose bbox contains it), and each
    connector with the `starting_zone` that emitted it (both assigned here, not by
    the planner). Standalone: no SlotLog binding, no cache, retries unlogged.
    Returns (anchors, connectors, reasoning, names); `names` maps each anchor's
    aggregate list index to its point-of-interest name."""
    root = util.find_root(nodes)
    if root is None:
        return [], [], "", {}
    regions = [n for n in nodes if util.is_region(n)]
    # The root is always a zone (its objects are the scene's shared shell / ground
    # geometry); guarantee it's planned even if it wasn't flagged as a region.
    if all(r.id != root.id for r in regions):
        regions.insert(0, root)
    by_id = {n.id: n for n in nodes}
    depth_of = {r.id: _region_depth(r, by_id) for r in regions}
    objects_by_region = util.index_objects_by_region(nodes)
    # One pass per zone that owns objects (root included, for shared shell / ground
    # geometry). Fall back to a single root pass over every object when no zone
    # claims ownership (pre-V3 scenes carry no `parent_region`).
    targets = [
        (z, objects_by_region[z.id]) for z in regions if objects_by_region.get(z.id)
    ]
    if not targets:
        concrete = [n for n in nodes if not util.is_region(n)]
        targets = [(root, concrete)] if concrete else []
    if not targets:
        return [], [], "", {}
    # Plan deepest zones first: the innermost spaces claim their anchors, then the
    # shallower / shared zones fill the gaps around what's already placed.
    targets.sort(key=lambda t: depth_of[t[0].id], reverse=True)

    # Sequential, not concurrent: each zone's call receives the anchors already
    # placed in the zones before it, so the planner can avoid overlapping them.
    anchors: list[PlacedAnchor] = []
    connectors: list[PlacedConnector] = []
    reasoning_parts: list[str] = []
    for zone, objs in targets:
        plan, reasoning = await _plan_zone(nodes, zone, objs, regions, anchors)
        # Anchors: assign each to the zone it mathematically INHABITS (the deepest
        # region whose bbox contains it), not merely the zone that emitted it;
        # fall back to the emitting zone if the point lands outside every bbox.
        # Connectors: the object belongs to the zone that owns it, so the emitting
        # zone IS its starting_zone.
        for a in plan.anchors:
            host = _inhabiting_zone(a.position, regions, depth_of) or zone
            anchors.append(PlacedAnchor(position=a.position, zone=host.id))
        connectors.extend(
            PlacedConnector(id=c.id, target_zone=c.target_zone, starting_zone=zone.id)
            for c in plan.connectors
        )
        if reasoning:
            reasoning_parts.append(f"[{zone.id}]\n{reasoning}")
    names = await name_anchors(nodes, anchors)
    return anchors, connectors, "\n\n".join(reasoning_parts), names