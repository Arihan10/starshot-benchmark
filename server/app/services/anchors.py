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

A third pass DESCRIBES the FLOORS: it gives each storey a name and a world-space
bounding volume.

The anchors cluster by height into storeys (the same Y-gap clustering the capture
uses for its bird's-eye slices), but that clustering is one-dimensional — it knows
a floor's height and nothing about where the floor stops. The walkthrough needs
both:

  * the NAME previews a floor before you travel to it, and must not name one room:
    the click auto-homes to whichever anchor is nearest the cursor, so a floor
    labelled "Master Bedroom" that drops you in the ensuite reads as a broken
    promise. The namer characterizes the storey AS A WHOLE.
  * the VOLUME answers "is this point on a floor, and which one" for arbitrary
    geometry under the cursor. Without it the viewer had to infer a floor from the
    nearest capture point, so pointing at a cliff face — terrain belonging to no
    storey — resolved to whichever anchor sat closest and offered to send you
    there. Terrain must be allowed to belong to NO floor, which is a judgement
    about what the scene is, not a distance computation.

See SYSTEM_FLOOR_DESCRIBER for both instructions.
"""

from __future__ import annotations

from collections import Counter

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


SYSTEM_FLOOR_DESCRIBER = """\
You are the floor describer for a text-to-3D scene pipeline. A scene tree has already been fully built from a text prompt (made up of zones and objects), a planner has placed 360° camera capture points throughout it, and those points have been grouped by height into FLOORS (storeys / levels / decks / terraces). You are given the scene tree plus, for each floor, the zones and points of interest that sit on it. For each floor you produce two things — its NAME and its BOUNDING BOX.

NAME — NAME THE WHOLE FLOOR, NEVER ONE PLACE ON IT. The name is shown to a visitor as the label on a preview BEFORE they travel to that floor — and when they go, they arrive at whichever capture point is nearest where they clicked, which can be ANY point on that floor. So a name that promises one specific room betrays them: label a floor of bedrooms "Upper Bedroom Level", never "Master Bedroom", because the visitor who lands in the ensuite or the hallway was told something untrue. The test to apply to every name: would this name still be honest if the visitor arrived at the LEAST representative point on this floor? If not, make it broader. When a floor has one clear use, name it for that use; when it mixes uses, name it for its character or its two dominant uses ("Living & Dining Level", "Shoreline & Dock Level", "Arrival & Parking Level"). Aim for 2-4 words. Use the vocabulary the scene itself implies — a house has floors, a cliffside has terraces or levels, a ship has decks, a tower has storeys — rather than forcing the word "Floor". Do not include the floor number; the interface already shows it. Every floor must get a distinct name.

BOUNDING BOX — the world-space volume A VISITOR ON THIS FLOOR OCCUPIES: the ground they can stand on, the rooms and outdoor decks that belong to this storey, and the headroom above it. The interface uses this box to answer "the user is pointing at this piece of geometry — which floor is that?", so what you include is what it will offer to travel to.

  * EXCLUDE anything that merely PASSES THROUGH this floor's height without being part of it: a cliff face, a rock wall, the sea or a lake surface, a tall tree, the outer skin of the building, the void beside a balcony. This is the single most important thing you do here. If a point on the cliff below the top storey falls inside the top storey's box, the interface will offer to send a visitor "to the top floor" because they pointed at rock — that failure is exactly what this box exists to prevent.
  * A point belonging to NO floor is a correct and expected outcome. Terrain, scenery and structure between storeys should fall outside every box. Do NOT try to cover the whole scene; the boxes together will and should leave gaps.
  * Floors STACK: their vertical ranges must not overlap each other. Leave the slab/void between two storeys outside both boxes.
  * Horizontally, cover only where a visitor can actually be on this floor. A storey that occupies one wing of a large site gets a box around that wing, not around the site.
  * Every capture point listed for a floor must fall inside that floor's box — they are cameras standing on it.

Coordinate convention (identical to the scene's): right-handed, Y-up, meters.
  +X = right, +Y = up, +Z = toward the viewer (front), -Z = away (back).
Give the box as `origin` — its minimum corner (x, y, z) — and `dimensions` (width, height, depth); the box spans from that corner along +X/+Y/+Z by the dimensions.

Output ONLY the JSON object matching the schema: each floor is its numeric `level`, string `name`, `origin` and `dimensions`."""


# Anchors within this vertical gap (metres) belong to the same floor. MUST match
# MINIMAP_LEVEL_EPS in client/public/js/tourcapture.js: the capture re-derives the
# storeys independently (to cut its bird's-eye slices), and the two groupings have
# to agree or a floor's name would land on the wrong slice. The names are matched
# back by nearest Y rather than by index, so a drift degrades to a mislabel rather
# than a crash — but keep them equal.
FLOOR_LEVEL_EPS = 1.5


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


class FloorSpec(BaseModel):
    level: int
    name: str
    origin: Vec3Tuple  # minimum corner of the floor's world-space box
    dimensions: Vec3Tuple  # width, height, depth from that corner along +X/+Y/+Z


class FloorSpecs(BaseModel):
    floors: list[FloorSpec]


class Floor(BaseModel):
    """One storey of the capture: the anchors that share a height band, its
    representative camera height, and the describer's name + world-space volume.
    `level` counts from the bottom (0 = lowest), matching the capture's minimap
    slice indices. `y` is what the client matches back by, so it must stay the SAME
    median the capture computes. `origin`/`dimensions` are None when the describer
    gave nothing usable — the viewer then falls back to its old nearest-anchor
    reading rather than trusting a broken box."""

    level: int
    y: float
    anchors: list[int]  # indices into the aggregated anchor list
    name: str | None = None
    origin: Vec3Tuple | None = None
    dimensions: Vec3Tuple | None = None


def group_floors(anchors: list[PlacedAnchor]) -> list[Floor]:
    """Cluster the planned anchors into floors by vertical gap, low to high — the
    Python twin of tourcapture.js `groupAnchorLevels`, kept identical (sort by Y,
    cut when the gap exceeds FLOOR_LEVEL_EPS, representative Y = the group's lower
    median) so the floors named here are the floors the capture slices."""
    order = sorted(range(len(anchors)), key=lambda i: anchors[i].position[1])
    groups: list[list[int]] = []
    last_y = 0.0
    for i in order:
        y = anchors[i].position[1]
        if not groups or y - last_y > FLOOR_LEVEL_EPS:
            groups.append([])
        groups[-1].append(i)
        last_y = y
    floors: list[Floor] = []
    for level, members in enumerate(groups):
        ys = sorted(anchors[i].position[1] for i in members)
        floors.append(Floor(level=level, y=ys[(len(ys) - 1) // 2], anchors=members))
    return floors


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


def _floor_entry(
    floor: Floor,
    anchors: list[PlacedAnchor],
    names: dict[int, str],
    total: int,
) -> str:
    """One floor's evidence: how big it is, which zones own it (by capture count,
    so the namer can weigh a dominant use against a mixed one), and every point of
    interest already named on it."""
    counts = Counter(anchors[i].zone for i in floor.anchors)
    zones = ", ".join(f"{z} ({c} capture points)" for z, c in counts.most_common())
    pois = [names[i] for i in floor.anchors if names.get(i)]
    lines = [
        f"level: {floor.level}   (floor {floor.level + 1} of {total}, counting from the bottom)",
        f"Camera height on this floor: y = {floor.y:.2f} m",
        f"Capture points on this floor: {len(floor.anchors)}",
        f"Zones on this floor: {zones or 'none'}",
    ]
    if pois:
        lines.append("Points of interest on this floor: " + ", ".join(pois))
    return util.braces("\n".join(lines))


def build_floor_describer_prompt(
    nodes: list[Node],
    anchors: list[PlacedAnchor],
    floors: list[Floor],
    names: dict[int, str],
) -> str:
    """The same scene context the point namer reads — every zone and object with
    its world-space box, which is what the describer measures a floor's extent
    against — plus one block per floor describing everything that sits on it."""
    context = _render_scene_context(nodes)
    (lo_x, lo_y, lo_z), (hi_x, hi_y, hi_z) = _scene_aabb(nodes)
    return (
        "Name and bound the floors of the scene below. Each floor is a height band "
        "holding some of the scene's 360° capture points; a visitor previews a "
        "floor by its name and may then arrive at ANY capture point on it, and the "
        "interface decides which floor a piece of geometry belongs to by testing "
        "the point against your boxes.\n\n"
        f"Scene world bounds: min corner ({lo_x:.2f}, {lo_y:.2f}, {lo_z:.2f}) m, "
        f"max corner ({hi_x:.2f}, {hi_y:.2f}, {hi_z:.2f}) m.\n\n"
        "=== SCENE HIERARCHY ===\n"
        f"{context}\n"
        "=== END SCENE HIERARCHY ===\n\n"
        "=== FLOORS ===\n"
        f"{util.brace_group([_floor_entry(f, anchors, names, len(floors)) for f in floors])}\n"
        "=== END FLOORS ===\n\n"
        "Now give every floor its name and bounding box, by level."
    )


# Anchors sit at eye height, so a box grown to contain them would cut the floor off
# at the visitor's eyes; this pads that safety expansion out to a plausible room.
_ANCHOR_PAD = 0.5


def _floor_bands(
    floors: list[Floor], scene_lo_y: float, scene_hi_y: float
) -> list[tuple[float, float]]:
    """The vertical slab each storey owns: halfway down to the storey below,
    halfway up to the one above, and out to the scene's own bounds at the bottom
    and top.

    The VERTICAL split is derived, not asked for — the height clustering already
    knows where the storeys separate — which caps how wrong a box can be. Without
    it a single bad response could span the whole scene height, so every point in
    the world would read as that floor: strictly worse than the nearest-anchor
    behaviour this replaces. What's left for the model is the HORIZONTAL extent,
    the part that genuinely needs judgement, since no amount of clustering can tell
    you the cliff beside the top storey isn't part of it.

    This BOUNDS the boxes but does not make them disjoint: `_apply_floor_box` then
    grows each box around its own capture points, which wins over the band, and a
    chained cluster can carry an anchor past the midpoint (a scene here has a
    y=5.7 anchor on a floor whose band ends at 5.0). So adjacent storeys may
    overlap by a little, and the viewer's `floorAt` resolves that deterministically
    by taking the smallest containing volume."""
    bands: list[tuple[float, float]] = []
    for i, f in enumerate(floors):
        lo = scene_lo_y if i == 0 else (floors[i - 1].y + f.y) / 2
        hi = scene_hi_y if i == len(floors) - 1 else (f.y + floors[i + 1].y) / 2
        bands.append((lo, hi))
    return bands


def _apply_floor_box(
    floor: Floor,
    spec: FloorSpec,
    anchors: list[PlacedAnchor],
    scene_lo: Vec3Tuple,
    scene_hi: Vec3Tuple,
    band: tuple[float, float],
) -> None:
    """Validate one described box onto its floor. The model is trusted for
    JUDGEMENT (which ground belongs to a storey and which is passing scenery) but
    not for arithmetic or for the vertical split: inverted corners are un-inverted,
    the box is clipped to the scene's own bounds so a floor can never claim empty
    space, its height is clipped to the storey's derived band (see `_floor_bands`),
    and it is finally grown to contain that floor's own capture points — a box that
    excluded its own camera would classify the visitor standing on it as being on
    no floor. A box left degenerate is dropped, which downgrades that floor to the
    nearest-anchor reading."""
    lo = [float(spec.origin[i]) for i in range(3)]
    hi = [lo[i] + float(spec.dimensions[i]) for i in range(3)]
    for i in range(3):
        if hi[i] < lo[i]:
            lo[i], hi[i] = hi[i], lo[i]
        lo[i] = max(lo[i], scene_lo[i])
        hi[i] = min(hi[i], scene_hi[i])
    lo[1] = max(lo[1], band[0])
    hi[1] = min(hi[1], band[1])
    for idx in floor.anchors:
        p = anchors[idx].position
        for i in range(3):
            lo[i] = min(lo[i], p[i] - _ANCHOR_PAD)
            hi[i] = max(hi[i], p[i] + _ANCHOR_PAD)
    if any(hi[i] <= lo[i] for i in range(3)):
        return
    floor.origin = (lo[0], lo[1], lo[2])
    floor.dimensions = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])


async def describe_floors(
    nodes: list[Node],
    anchors: list[PlacedAnchor],
    floors: list[Floor],
    names: dict[int, str],
) -> None:
    """Name and bound each floor IN PLACE with the fixed namer model. Runs after
    the point namer so it can see what each storey actually holds. A floor the
    model omits keeps `name = None` and no box — the client renders a plain ordinal
    and falls back to its nearest-anchor floor reading, both of which beat a wrong
    answer. Standalone: no SlotLog binding, no cache, retries unlogged."""
    if not floors:
        return
    llm.set_model(ANCHOR_NAMER_MODEL)
    result, _reasoning, _usage, _raw, _gen_ids = await llm.call_llm_once(
        system=SYSTEM_FLOOR_DESCRIBER,
        user=build_floor_describer_prompt(nodes, anchors, floors, names),
        output_schema=FloorSpecs,
        model=ANCHOR_NAMER_MODEL,
        step="floor_describe",
        log_retries=False,
    )
    scene_lo, scene_hi = _scene_aabb(nodes)
    bands = _floor_bands(floors, scene_lo[1], scene_hi[1])
    by_level = {f.level: (f, bands[i]) for i, f in enumerate(floors)}
    for spec in result.floors:
        entry = by_level.get(spec.level)
        if entry is None:
            continue
        floor, band = entry
        if spec.name.strip():
            floor.name = spec.name.strip()
        _apply_floor_box(floor, spec, anchors, scene_lo, scene_hi, band)


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
) -> tuple[list[PlacedAnchor], list[PlacedConnector], str, dict[int, str], list[Floor]]:
    """Plan capture anchors + connectors one zone at a time (a fixed-planner call
    per zone, run sequentially), then name the aggregated anchors with the fixed
    namer model. Each zone call sees only its own objects, every zone's identity,
    and the anchors already placed in earlier zones — so it can mark the objects
    that connect into another zone and avoid overlapping prior anchors. Zones are
    planned deepest-first. Each returned anchor is stamped with the `zone` it
    mathematically inhabits (the deepest region whose bbox contains it), and each
    connector with the `starting_zone` that emitted it (both assigned here, not by
    the planner). Standalone: no SlotLog binding, no cache, retries unlogged.
    Returns (anchors, connectors, reasoning, names, floors); `names` maps each
    anchor's aggregate list index to its point-of-interest name, and `floors` are
    the height-clustered storeys, each carrying the describer's name + volume."""
    root = util.find_root(nodes)
    if root is None:
        return [], [], "", {}, []
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
        return [], [], "", {}, []
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
    # Floors last: clustering needs every anchor placed, and the describer reads
    # the point names to judge what each storey is FOR.
    floors = group_floors(anchors)
    await describe_floors(nodes, anchors, floors, names)
    return anchors, connectors, "\n\n".join(reasoning_parts), names, floors