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

A third pass PLANS the FLOORS: it decides where the scene divides into storeys at
all, then names each one, bounds it, and picks the height its map is cut at.

The split used to be arithmetic — anchors were clustered by vertical gap and the
model was handed the result to decorate. That was fragile in a way the one scene
it was tested on already showed: the modern house's three storeys hang on gaps of
1.90 m and 2.00 m against a 1.5 m threshold, one gap of exactly 1.50 m chains, and
the resulting "ground floor" swallows 9.1 m and half the scene's anchors because
the staircase leaves capture points at every height in between. A gap is evidence,
not an answer: a stair, a ramp or a sloping shore crosses between storeys leaving
no gap at all, and telling one from a floor means reading the scene, which is
exactly what a threshold cannot do.

So the model now chooses the boundaries, and code does the bookkeeping — it is
never asked which anchor goes where (a long list nobody can check), only where the
storeys divide (a handful of numbers that are easy to check). Each floor carries
four things:

  * the NAME previews a floor before you travel to it, and must not name one room:
    the click auto-homes to whichever anchor is nearest the cursor, so a floor
    labelled "Master Bedroom" that drops you in the ensuite reads as a broken
    promise. The namer characterizes the storey AS A WHOLE.
  * the BOX answers "is this point on a floor, and which one" for arbitrary
    geometry under the cursor. Without it the viewer had to infer a floor from the
    nearest capture point, so pointing at a cliff face — terrain belonging to no
    storey — resolved to whichever anchor sat closest and offered to send you
    there. Terrain must be allowed to belong to NO floor, which is a judgement
    about what the scene is, not a distance computation. Its VERTICAL extent is
    also the split itself, so the two can no longer disagree.
  * the CUT is the height the bird's-eye map is sliced at. It is NOT the top of the
    box: the box reaches the ceiling by design (it is the headroom a visitor has),
    and a map sliced there keeps the ceiling and renders a blank lid. The cut has
    to sit below whatever roofs the storey and above what stands on it.
  * the ANCHORS are assigned here, by testing each capture against the boxes — so
    the grouping is computed once and shipped, rather than re-derived by the
    capture and the viewer from a threshold each of them holds a copy of.

See SYSTEM_FLOOR_PLANNER for all four instructions. `group_floors` survives as the
fallback for a response that fails validation.
"""

from __future__ import annotations

import math
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


SYSTEM_FLOOR_PLANNER = """\
You are the floor planner for a text-to-3D scene pipeline. A scene tree has already been fully built from a text prompt (made up of zones and objects) and a planner has placed 360° camera capture points throughout it. You decide how that scene divides into FLOORS (storeys / levels / decks / terraces), and for each floor you produce three things — its NAME, its BOUNDING BOX, and the height its MAP IS CUT at.

You are given the scene tree, and a ladder of every height a capture point stands at, low to high, with the gap to the next height and the zones standing there.

THE SPLIT — decide where one storey ends and the next begins. The ladder is evidence, not the answer.

  * A GAP IS NOT PROOF OF A SPLIT. A sunken lounge, a raised platform or a mezzanine leaves one inside a single storey.
  * THE ABSENCE OF A GAP IS NOT PROOF THERE IS NO SPLIT. An open stairwell, a ramp, a terraced path or a sloping shoreline leaves capture points at every height between two storeys. Those points are the journey between floors, not a floor of their own — and if you let them chain, two real storeys merge into one that is metres tall and answers to no name. Read the zone names on the ladder against the scene tree: a run of heights all belonging to a walkway, a stair or a path segment is a route THROUGH the split, not evidence against it.

EVERY CAPTURE POINT LANDS ON A FLOOR. You give the boundaries; the interface assigns each capture to whichever floor's box contains it, and anything falling in the space between two boxes goes to the nearer one. So you may leave the slab between storeys outside both boxes — that is correct — but do not leave a storey's worth of captures stranded outside every box.

A SCENE NEED NOT HAVE STOREYS, AND MOST DO NOT. A single room, an arena, a campsite, a cavern, an open landscape, a side-on level whose ground climbs continuously from one end to the other — these are one continuous space, and the honest answer is ONE floor covering all of it. Inventing storeys where the scene has none hands the visitor a floor selector that shuffles them between arbitrary heights of the same place. Returning a single floor is a correct, common and expected answer.

NAME — NAME THE WHOLE FLOOR, NEVER ONE PLACE ON IT. The name is shown to a visitor as the label on a preview BEFORE they travel to that floor — and when they go, they arrive at whichever capture point is nearest where they clicked, which can be ANY point on that floor. So a name that promises one specific room betrays them: label a floor of bedrooms "Upper Bedroom Level", never "Master Bedroom", because the visitor who lands in the ensuite or the hallway was told something untrue. The test to apply to every name: would this name still be honest if the visitor arrived at the LEAST representative point on this floor? If not, make it broader. When a floor has one clear use, name it for that use; when it mixes uses, name it for its character or its two dominant uses ("Living & Dining Level", "Shoreline & Dock Level", "Arrival & Parking Level"). Aim for 2-4 words. Use the vocabulary the scene itself implies — a house has floors, a cliffside has terraces or levels, a ship has decks, a tower has storeys — rather than forcing the word "Floor". Do not include the floor number; the interface already shows it. Every floor must get a distinct name.

BOUNDING BOX — the world-space volume A VISITOR ON THIS FLOOR OCCUPIES: the ground they can stand on, the rooms and outdoor decks that belong to this storey, and the headroom above it. The interface uses this box to answer "the user is pointing at this piece of geometry — which floor is that?", so what you include is what it will offer to travel to.

  * EXCLUDE anything that merely PASSES THROUGH this floor's height without being part of it: a cliff face, a rock wall, the sea or a lake surface, a tall tree, the outer skin of the building, the void beside a balcony. This is the single most important thing you do here. If a point on the cliff below the top storey falls inside the top storey's box, the interface will offer to send a visitor "to the top floor" because they pointed at rock — that failure is exactly what this box exists to prevent.
  * A point belonging to NO floor is a correct and expected outcome. Terrain, scenery and structure between storeys should fall outside every box. Do NOT try to cover the whole scene; the boxes together will and should leave gaps.
  * Floors STACK: their vertical ranges must not overlap each other. Leave the slab/void between two storeys outside both boxes. THE VERTICAL EXTENT OF THIS BOX IS THE SPLIT YOU CHOSE — its bottom is where this storey's ground is, its top is the ceiling or the open sky above it — so the boxes and the split cannot disagree.
  * Horizontally, cover only where a visitor can actually be on this floor. A storey that occupies one wing of a large site gets a box around that wing, not around the site.

MAP CUT — the height this floor's bird's-eye map is sliced at. That map is rendered looking straight DOWN with everything ABOVE this height thrown away, so this one number decides whether the visitor sees the floor or sees nothing.

  * IT IS NOT THE TOP OF THE BOX. The box reaches the ceiling on purpose — that is the headroom a visitor has. Slicing there keeps the ceiling, and the map renders as a blank lid with the entire storey hidden underneath it. That is the one failure this number exists to prevent, and it is always better to err LOW than high.
  * Put it BELOW whatever roofs this storey — the ceiling, the floor slab of the storey above, a canopy, an overhang, a bridge — and ABOVE the things a visitor moves among, so the map shows the ground, the furniture and the walls standing around them.
  * Where NOTHING roofs the storey — an open deck, a courtyard, a shoreline, the top storey of anything — there is no lid to get under. Put it above the tallest thing worth seeing from above, and still below anything that would hide the rest of the floor (a tree canopy, a pergola, a gantry).
  * As a rule of thumb it lands a little above head height on the floor in question — roughly where an architect cuts a plan drawing, high enough to show the furniture and the walls, low enough to be under the ceiling.

Coordinate convention (identical to the scene's): right-handed, Y-up, meters.
  +X = right, +Y = up, +Z = toward the viewer (front), -Z = away (back).
Give the box as `origin` — its minimum corner (x, y, z) — and `dimensions` (width, height, depth); the box spans from that corner along +X/+Y/+Z by the dimensions. Give `plan_cut` as an absolute world height (a y value), not a height above the floor.

List the floors from the LOWEST upward.

Output ONLY the JSON object matching the schema: each floor is a string `name`, an `origin`, `dimensions`, and `plan_cut`."""


SYSTEM_MAP_LABELLER = """\
You are the map labeller for a text-to-3D scene pipeline. A scene tree has already been fully built from a text prompt, made up of zones (regions) that may contain further zones. A visitor walking the finished scene sees a small bird's-eye map of the storey they are standing on, and you choose which zones get their name printed onto that map.

CHOOSE THE LEVEL A VISITOR WOULD NAVIGATE BY. The tree is decomposed as finely as the generator found useful, which is far finer than a map should be — a whole level, a wing of it, a room in that wing, and a corner of that room may all be zones. Pick the ones a person would actually use to say where they are, and skip both the sweeping containers above them and the incidental subdivisions below.

NEVER MARK A ZONE THAT SITS INSIDE A ZONE YOU HAVE ALREADY MARKED. The map is a few centimetres across; two names printed over the same patch of floor overlap into an unreadable smudge and contradict each other about what that floor is. Wherever one zone contains another, choose exactly one of them — whichever a visitor would name.

PREFER FEWER LABELS. A map with four names you can read beats one with twelve you cannot. Zones that are small, thin, or purely transitional — a landing, a walkway segment, a flight of steps — are usually better left unnamed: the map already shows them, it does not need to caption them. Say nothing about a zone rather than crowd the map with it.

The `label` is what gets printed, so write it for a reader, not for the pipeline: 1-3 words, title case, no underscores, no level prefixes. "Living Lounge", "Boat Dock", "Carport" — not "middle_interior_living_lounge".

Output ONLY the JSON object matching the schema: each entry is an existing zone `id` and the `label` to print for it."""


# Anchors within this vertical gap (metres) belong to the same floor. The capture
# re-derives the storeys independently (to cut its bird's-eye slices), so the two
# groupings have to agree or a floor's name would land on the wrong slice — which
# is why this value is SHIPPED to it, in the capture manifest's `minimap.level_eps`
# (routes.py). tourcapture.js carries a fallback of its own for a manifest that
# lacks the field, but a live capture always clusters with the number below. Edit
# it here and both sides follow.
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
    """One storey as the model proposes it. There is deliberately no `level`: the
    floors are ordered by where their boxes sit, so a response that numbers them
    wrongly (or not at all) still lands in the right order."""

    name: str
    origin: Vec3Tuple  # minimum corner of the floor's world-space box
    dimensions: Vec3Tuple  # width, height, depth from that corner along +X/+Y/+Z
    plan_cut: float  # world height the bird's-eye map is sliced at


class FloorSpecs(BaseModel):
    floors: list[FloorSpec]


class Floor(BaseModel):
    """One storey of the capture: the anchors assigned to it, its representative
    camera height, the planner's name + world-space volume, and the height its
    bird's-eye map is cut at.

    `level` counts from the bottom (0 = lowest) and is assigned here, matching the
    capture's minimap slice indices. `y` is what the client matches back by.
    `origin`/`dimensions` are None when the model gave nothing usable and the
    clustering fallback ran — the viewer then falls back to its old nearest-anchor
    reading rather than trusting a broken box.

    `cut` is NOT the top of the box. The box's top is the ceiling (it is the
    headroom a visitor has); the cut is where the map is sliced, which has to be
    below the ceiling or the map shows the ceiling and nothing else. It is always
    resolved to a usable value — see `_resolve_plan_cut` — so it is never None on a
    floor that leaves this module."""

    level: int
    y: float
    anchors: list[int]  # indices into the aggregated anchor list
    name: str | None = None
    origin: Vec3Tuple | None = None
    dimensions: Vec3Tuple | None = None
    cut: float | None = None


def group_floors(anchors: list[PlacedAnchor]) -> list[Floor]:
    """Cluster the planned anchors into floors by vertical gap, low to high: sort
    by Y, cut when the gap exceeds FLOOR_LEVEL_EPS, representative Y = the group's
    lower median.

    This is now the FALLBACK, not the primary path — `plan_floors` asks the model
    where the scene divides and only lands here when the response can't be used.
    The floors it returns carry no name and no box, which is the same degraded
    state a failed describe pass used to leave behind: the client renders a plain
    ordinal and falls back to its nearest-anchor floor reading."""
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


class MapZone(BaseModel):
    id: str
    label: str


class MapZones(BaseModel):
    zones: list[MapZone]


class MapLabel(BaseModel):
    """One zone name printed on the bird's-eye map, with the world-space centre of
    the zone it names. The client turns that centre into a position on whichever
    storey's slice it falls on."""

    id: str
    label: str
    center: Vec3Tuple


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


def _height_ladder(anchors: list[PlacedAnchor], names: dict[int, str]) -> str:
    """Every capture height low to high, with what stands there and the gap to the
    next — the evidence the split is read off.

    Grouped BY HEIGHT rather than listed per capture: the planner places many
    captures at one level, so a 152-anchor house collapses to a couple of dozen
    rows, and the gaps that might be storey boundaries line up down the page
    instead of being buried in a list. The zones are printed per rung because they
    are what lets a staircase be told from a floor — a run of heights all belonging
    to a walkway is a route through a split, not evidence against one."""
    by_height: dict[float, list[int]] = {}
    for i, a in enumerate(anchors):
        by_height.setdefault(round(a.position[1], 2), []).append(i)
    heights = sorted(by_height)
    rows: list[str] = []
    for n, y in enumerate(heights):
        members = by_height[y]
        counts = Counter(anchors[i].zone for i in members)
        zones = ", ".join(
            f"{z} ({c})" if c > 1 else f"{z}" for z, c in counts.most_common()
        )
        rows.append(
            f"y = {y:8.2f} m   {len(members):3d} capture(s)   zones: {zones or 'none'}"
        )
        pois = [names[i] for i in members if names.get(i)]
        if pois:
            shown = ", ".join(pois[:6])
            extra = f", +{len(pois) - 6} more" if len(pois) > 6 else ""
            rows.append(f"{'':18}points: {shown}{extra}")
        if n + 1 < len(heights):
            rows.append(f"{'':14}---- {heights[n + 1] - y:.2f} m gap ----")
    return "\n".join(rows)


def build_floor_planner_prompt(
    nodes: list[Node],
    anchors: list[PlacedAnchor],
    names: dict[int, str],
) -> str:
    """The same scene context the point namer reads — every zone and object with
    its world-space box, which is what a floor's extent gets measured against —
    plus the height ladder the split is chosen from."""
    context = _render_scene_context(nodes)
    (lo_x, lo_y, lo_z), (hi_x, hi_y, hi_z) = _scene_aabb(nodes)
    return (
        "Decide how the scene below divides into floors, then name, bound and cut "
        "each one. A visitor previews a floor by its name and may then arrive at "
        "ANY capture point on it; the interface decides which floor a piece of "
        "geometry belongs to by testing the point against your boxes, and draws "
        "each floor's map by slicing the scene at your cut height and looking "
        "straight down.\n\n"
        f"Scene world bounds: min corner ({lo_x:.2f}, {lo_y:.2f}, {lo_z:.2f}) m, "
        f"max corner ({hi_x:.2f}, {hi_y:.2f}, {hi_z:.2f}) m.\n\n"
        "=== SCENE HIERARCHY ===\n"
        f"{context}\n"
        "=== END SCENE HIERARCHY ===\n\n"
        "=== CAPTURE HEIGHTS (low to high) ===\n"
        f"{_height_ladder(anchors, names)}\n"
        "=== END CAPTURE HEIGHTS ===\n\n"
        "Now give the scene's floors, lowest first."
    )


# Anchors sit at eye height, so a box grown to contain them would cut the floor off
# at the visitor's eyes; this pads that safety expansion out to a plausible room.
_ANCHOR_PAD = 0.5

# How far under a storey's ceiling the map cut is forced to sit. The top of a
# floor's box IS its ceiling — the box is the headroom a visitor has — so a cut
# level with it keeps the ceiling and the map renders as a blank lid. This is the
# backstop for the single most likely way to get that number wrong: handing back
# the top of the box.
_PLAN_CUT_CLEARANCE = 0.25

# A runaway guard, not a policy. A real scene can genuinely have many storeys, so
# this sits far above any plausible answer; a response past it is not a reading of
# the scene at all, and the whole thing is dropped for the clustering fallback.
_MAX_FLOORS = 32


def _clip_box(
    spec: FloorSpec, scene_lo: Vec3Tuple, scene_hi: Vec3Tuple
) -> tuple[list[float], list[float]]:
    """One proposed storey as a validated [min corner, max corner] pair: inverted
    corners un-inverted, the whole box clipped to the scene's own bounds so a floor
    can never claim empty space."""
    lo = [float(spec.origin[i]) for i in range(3)]
    hi = [lo[i] + float(spec.dimensions[i]) for i in range(3)]
    for i in range(3):
        if hi[i] < lo[i]:
            lo[i], hi[i] = hi[i], lo[i]
        lo[i] = max(lo[i], scene_lo[i])
        hi[i] = min(hi[i], scene_hi[i])
    return lo, hi


def _stack_vertically(boxes: list[tuple[list[float], list[float]]]) -> None:
    """Force the storeys into a stack, IN PLACE.

    They are asked not to overlap and mostly won't, but an overlap has to be
    impossible rather than unlikely: the assignment below reads each capture into
    the first box that contains it, so two boxes sharing a height band would make
    that answer depend on the order the model happened to list them in. An overlap
    is split down the middle, which is the fair reading of two boxes both claiming
    the same slab.

    Deliberately does NOT close gaps. The void between two storeys belongs to
    neither, which is the entire point of asking for boxes rather than cuts."""
    for lower, upper in zip(boxes, boxes[1:]):
        if lower[1][1] > upper[0][1]:
            mid = (lower[1][1] + upper[0][1]) / 2
            lower[1][1] = mid
            upper[0][1] = mid


def _assign_anchors(
    boxes: list[tuple[list[float], list[float]]], anchors: list[PlacedAnchor]
) -> list[list[int]]:
    """Put every capture on exactly one floor: the storey whose height band holds
    it, or — for one standing in the void between two storeys — the nearer of them.

    This is the bookkeeping the model is deliberately never asked to do. Assigning
    150 captures by hand is a long answer nobody can check, and every capture it
    forgot, duplicated or invented would be a bug; done here, no capture can go
    missing and none can land on two floors."""
    out: list[list[int]] = [[] for _ in boxes]
    for idx, a in enumerate(anchors):
        y = a.position[1]
        best = -1
        best_gap = math.inf
        for i, (lo, hi) in enumerate(boxes):
            if lo[1] <= y <= hi[1]:
                best = i
                break
            gap = lo[1] - y if y < lo[1] else y - hi[1]
            if gap < best_gap:
                best_gap = gap
                best = i
        if best >= 0:
            out[best].append(idx)
    return out


def _assemble_floors(
    specs: list[FloorSpec],
    anchors: list[PlacedAnchor],
    scene_lo: Vec3Tuple,
    scene_hi: Vec3Tuple,
) -> list[Floor]:
    """Turn the model's proposed storeys into the scene's floors.

    The model is trusted for JUDGEMENT — where the scene divides, which ground
    belongs to a storey and which is passing scenery — and for nothing else. Boxes
    are un-inverted and clipped to the scene, ordered bottom-up by where they
    actually sit (not by any number the response put on them), forced into a
    non-overlapping stack, and only then do the captures get assigned.

    A band that ends up holding no captures is DROPPED. That costs nothing and it
    is what quietly prunes an over-split response — a scene cut into eleven storeys
    where only four are stood on comes back as four — without needing a rule about
    how many floors a scene is allowed to have.

    Returns [] when nothing survives, which sends the caller to the clustering
    fallback."""
    boxes = [_clip_box(s, scene_lo, scene_hi) for s in specs]
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0][1])
    boxes = [boxes[i] for i in order]
    ordered_specs = [specs[i] for i in order]
    _stack_vertically(boxes)
    members = _assign_anchors(boxes, anchors)

    # Grow each band to cover the captures that landed in it — one that fell in the
    # void between two storeys now belongs to this one, and the box should say so —
    # but never past a neighbour, which would put back the overlap just removed.
    for i, idxs in enumerate(members):
        if not idxs:
            continue
        ys = [anchors[j].position[1] for j in idxs]
        room_lo = boxes[i - 1][1][1] if i > 0 else scene_lo[1]
        room_hi = boxes[i + 1][0][1] if i + 1 < len(boxes) else scene_hi[1]
        boxes[i][0][1] = max(room_lo, min(boxes[i][0][1], min(ys) - _ANCHOR_PAD))
        boxes[i][1][1] = min(room_hi, max(boxes[i][1][1], max(ys) + _ANCHOR_PAD))

    floors: list[Floor] = []
    for (lo, hi), spec, idxs in zip(boxes, ordered_specs, members):
        if not idxs:
            continue  # a band nobody stands on is not a storey
        # Contain this floor's own captures horizontally too: a box that excluded
        # its own camera would classify the visitor standing on it as being on no
        # floor at all. Vertically that was already done above, under the clamps.
        for j in idxs:
            p = anchors[j].position
            for ax in (0, 2):
                lo[ax] = min(lo[ax], p[ax] - _ANCHOR_PAD)
                hi[ax] = max(hi[ax], p[ax] + _ANCHOR_PAD)
                lo[ax] = max(lo[ax], scene_lo[ax])
                hi[ax] = min(hi[ax], scene_hi[ax])
        if any(hi[ax] <= lo[ax] for ax in range(3)):
            continue
        ys = sorted(anchors[j].position[1] for j in idxs)
        floors.append(
            Floor(
                level=len(floors),
                y=ys[(len(ys) - 1) // 2],
                anchors=idxs,
                name=spec.name.strip() or None,
                origin=(lo[0], lo[1], lo[2]),
                dimensions=(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]),
                cut=float(spec.plan_cut),
            )
        )
    return floors


def _resolve_plan_cut(floor: Floor, anchors: list[PlacedAnchor]) -> None:
    """Settle one floor's map cut, IN PLACE.

    The model's number is a judgement about what roofs this storey, so it is kept
    when it is usable — but it is bounded on both sides, because both ways of
    getting it wrong blank the map:

      * TOO HIGH and the slice keeps the ceiling, so the map renders as a lid with
        the whole storey hidden under it. The cut is forced to clear the top of the
        floor's own box by `_PLAN_CUT_CLEARANCE`.
      * TOO LOW and the slice passes beneath the floor's own cameras, throwing away
        the furniture and walls the map exists to show. It is never dropped below
        the lowest capture standing on this storey.

    With no usable number — the clustering fallback, or a response that omitted it
    — this falls back to the median camera height on the floor, which is exactly
    where the capture used to cut before the cut was asked for at all."""
    ys = sorted(anchors[i].position[1] for i in floor.anchors)
    if not ys:
        floor.cut = floor.y
        return
    lowest = ys[0]
    cut = floor.cut
    if cut is None or not math.isfinite(cut):
        cut = ys[(len(ys) - 1) // 2]  # the median camera — the pre-cut behaviour
    ceiling = math.inf
    if floor.origin is not None and floor.dimensions is not None:
        ceiling = floor.origin[1] + floor.dimensions[1] - _PLAN_CUT_CLEARANCE
    # The storey's own cameras win a fight with a degenerate box: a band shorter
    # than the clearance would otherwise drive the cut below the floor entirely.
    floor.cut = min(max(cut, lowest), max(lowest, ceiling))


async def plan_floors(
    nodes: list[Node],
    anchors: list[PlacedAnchor],
    names: dict[int, str],
) -> list[Floor]:
    """Decide the scene's floors with the fixed namer model: where it divides, what
    each storey is called, how far each reaches, and where each one's map is cut.
    Runs after the point namer so it can read what each storey actually holds.

    Falls back to the height clustering (`group_floors`) whenever the response
    can't be used — the call failed, it named no floors, it named absurdly many, or
    nothing survived validation. That fallback yields unnamed, unbounded floors,
    which is precisely what a failed describe pass yielded before, so this can
    never be worse than the behaviour it replaces. Standalone: no SlotLog binding,
    no cache, retries unlogged."""
    if not anchors:
        return []
    floors: list[Floor] = []
    llm.set_model(ANCHOR_NAMER_MODEL)
    try:
        result, _reasoning, _usage, _raw, _gen_ids = await llm.call_llm_once(
            system=SYSTEM_FLOOR_PLANNER,
            user=build_floor_planner_prompt(nodes, anchors, names),
            output_schema=FloorSpecs,
            model=ANCHOR_NAMER_MODEL,
            step="floor_plan",
            log_retries=False,
        )
    except Exception:  # noqa: BLE001 - an anchor plan is worth far more than its floor names
        result = None
    if result is not None and 0 < len(result.floors) <= _MAX_FLOORS:
        scene_lo, scene_hi = _scene_aabb(nodes)
        floors = _assemble_floors(result.floors, anchors, scene_lo, scene_hi)
    if not floors:
        floors = group_floors(anchors)
    for f in floors:
        _resolve_plan_cut(f, anchors)
    return floors


def build_map_labeller_prompt(nodes: list[Node], regions: list[Node]) -> str:
    """The full scene hierarchy (which already nests zone inside zone, with each
    one's footprint) plus a flat roster of the ids that may be chosen."""
    context = _render_scene_context(nodes)
    roster = "\n".join(
        f"{r.id} — {util.format_dimensions(r.bbox)}" for r in regions
    )
    return (
        "Choose which zones of the scene below get their name printed on its "
        "bird's-eye map, and write the name for each.\n\n"
        "=== SCENE HIERARCHY (zones nest; a subregion sits inside its parent) ===\n"
        f"{context}\n"
        "=== END SCENE HIERARCHY ===\n\n"
        "=== ZONES YOU MAY LABEL ===\n"
        f"{roster}\n"
        "=== END ZONES ===\n\n"
        "Now choose the zones to label, and give each its printed name."
    )


def _ancestors(node: Node, by_id: dict[str, Node]) -> set[str]:
    """Every id on the path from `node` up to the root (exclusive of itself)."""
    out: set[str] = set()
    cur = node
    while cur.parent_id is not None and cur.parent_id not in out:
        out.add(cur.parent_id)
        parent = by_id.get(cur.parent_id)
        if parent is None:
            break
        cur = parent
    return out


async def label_map_zones(nodes: list[Node]) -> list[MapLabel]:
    """Pick the zones worth naming on the bird's-eye map, with the fixed namer
    model. Returns them with the world-space centre of each zone's bbox.

    The containment rule is ENFORCED here, not merely asked for: a scene decomposes
    into zones far more finely than a small map can caption, so a model that marks
    both a wing and a room inside it would stack two names on one patch of floor.
    Where a chosen zone sits inside another chosen zone, the OUTER one is kept —
    that is what "mark a zone and its subzones stay unmarked" means. Unknown ids and
    the root are dropped. Standalone: no SlotLog binding, no cache, retries
    unlogged."""
    regions = [n for n in nodes if util.is_region(n)]
    root = util.find_root(nodes)
    if root is not None:
        regions = [r for r in regions if r.id != root.id]
    if not regions:
        return []
    llm.set_model(ANCHOR_NAMER_MODEL)
    result, _reasoning, _usage, _raw, _gen_ids = await llm.call_llm_once(
        system=SYSTEM_MAP_LABELLER,
        user=build_map_labeller_prompt(nodes, regions),
        output_schema=MapZones,
        model=ANCHOR_NAMER_MODEL,
        step="map_label",
        log_retries=False,
    )
    by_id = {n.id: n for n in nodes}
    allowed = {r.id for r in regions}
    chosen: list[MapZone] = []
    seen: set[str] = set()
    for z in result.zones:
        if z.id in allowed and z.id not in seen and z.label.strip():
            seen.add(z.id)
            chosen.append(z)
    # Outermost wins: drop anything with a chosen ancestor.
    kept = [
        z for z in chosen if not (_ancestors(by_id[z.id], by_id) & seen)
    ]
    out: list[MapLabel] = []
    for z in kept:
        lo, hi = by_id[z.id].bbox.min_corner, by_id[z.id].bbox.max_corner
        out.append(
            MapLabel(
                id=z.id,
                label=z.label.strip(),
                center=(
                    (lo[0] + hi[0]) / 2,
                    (lo[1] + hi[1]) / 2,
                    (lo[2] + hi[2]) / 2,
                ),
            )
        )
    return out


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
    # Floors last: the split needs every anchor placed, and the floor planner reads
    # the point names to judge what each storey is FOR.
    floors = await plan_floors(nodes, anchors, names)
    return anchors, connectors, "\n\n".join(reasoning_parts), names, floors