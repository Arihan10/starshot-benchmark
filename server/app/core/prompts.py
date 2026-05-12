"""Prompts and structured-output schemas for LLM calls."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.core.types import BoundingBox, Orientation, ProxyShape, Relationship


# Shared proxy-shape documentation injected into every prompt that lets
# the LLM emit or reason about proxies. Keep the vocabulary and the math
# identical across decomposition and bbox-resolution steps so there is
# no drift — the bbox-resolution step needs the full formulas to place
# ON-anchored children correctly, so they live here alongside the
# vocabulary.
PROXY_SHAPE_DOC = """\
A `proxy_shape` describes the silhouette of the node's mesh INSIDE its \
axis-aligned bbox. The proxy is always inscribed in the AABB; you do \
NOT emit radii or cap sizes — they derive from the bbox dimensions. \
Emit `proxy_shape` ONLY when the mesh is noticeably non-boxy; omit it \
(i.e. null / absent) when the bbox itself is a good collision proxy.

NOTATION used below. For an AABB with min corner (x_min, y_min, z_min) \
and max corner (x_max, y_max, z_max): center (cx, cy, cz), \
half-extents (hx, hy, hz) = ((x_max-x_min)/2, (y_max-y_min)/2, \
(z_max-z_min)/2), full extents (sx, sy, sz) = (2·hx, 2·hy, 2·hz). \
Every proxy below defines a FOOTPRINT (the XZ region the shape covers) \
and a TOP-SURFACE FUNCTION Y_top(x, z) that returns the proxy's upper \
surface height at a given XZ inside that footprint. There is no \
automatic correction — when you anchor another node ON this one, YOU \
must compute Y_top at the anchor's XZ and place its bbox so its \
bottom face sits there.

Valid values:

  * null / omitted — BOX. The AABB is the proxy. Default.
      WHEN TO USE: walls, floors, ceilings, furniture, crates, \
      buildings, signs, rectangular terrain slabs — anything \
      rectilinear.
      FOOTPRINT: the full AABB rectangle, x ∈ [x_min, x_max], \
      z ∈ [z_min, z_max].
      Y_top(x, z) = y_max   (flat top face everywhere in the \
      footprint).

  * SPHERE — ellipsoid inscribed in the AABB, centered at (cx, cy, \
    cz) with semi-axes (hx, hy, hz).
      WHEN TO USE: boulders, planets, balls, orbs, fruits, pumpkins, \
      beach balls.
      FOOTPRINT: the disk ((x-cx)/hx)² + ((z-cz)/hz)² ≤ 1 in the XZ \
      plane through the centre.
      Y_top(x, z) = cy + hy · √(1 − ((x-cx)/hx)² − ((z-cz)/hz)²).
      Apex: (cx, y_max, cz).

  * CAPSULE — Y-axis capsule inscribed in the AABB. Let r = \
    min(hx, hz). Axis is the vertical line through (cx, cz).
      WHEN TO USE: tree trunks, humans, pillars, lamp posts, \
      bottles — anything columnar.
      Top cap centre: (cx, y_max − r, cz). Bottom cap centre: \
      (cx, y_min + r, cz). Cylindrical section of height sy − 2r \
      between the cap centres (degenerates to a sphere when \
      sy ≤ 2r, in which case r = sy/2 instead of min(hx, hz)).
      FOOTPRINT: the disk (x-cx)² + (z-cz)² ≤ r² centered on the \
      axis — note this is generally smaller than the AABB footprint.
      Y_top(x, z) = (y_max − r) + √(r² − (x-cx)² − (z-cz)²).
      Apex: (cx, y_max, cz).

  * HEMISPHERE — upper half of an ellipsoid with its equatorial \
    disk resting on the AABB's bottom face (y = y_min) and its apex \
    at (cx, y_max, cz). Semi-axes are (hx, sy, hz) — the VERTICAL \
    half-extent is the FULL AABB height sy, NOT hy, because the \
    equator sits at y_min, not at cy.
      WHEN TO USE: DOMED TERRAIN — low islands rising from the \
      waterline, grassy mounds, half-buried boulders, snow hills, \
      cathedral domes.
      FOOTPRINT: the disk ((x-cx)/hx)² + ((z-cz)/hz)² ≤ 1 at \
      y = y_min.
      Y_top(x, z) = y_min + sy · √(1 − ((x-cx)/hx)² − ((z-cz)/hz)²).
      Drops from the apex y_max at the centre to y_min at the \
      footprint boundary.

ON-RELATIONSHIP CONSEQUENCE. When you anchor a node ON a target with \
a non-BOX proxy, the AABB's top face is NOT the resting surface. \
Compute the target's Y_top at the anchored node's XZ centre using \
the target's AABB and the formula above, then set the anchored \
node's bbox so its bottom face Y equals Y_top. Example: a 0.8m tree \
placed ON a HEMISPHERE island whose AABB is (x_min=-5, y_min=0, \
z_min=-5) → (x_max=5, y_max=1.2, z_max=5), at XZ = (3, 0), rests at \
Y_top = 0 + 1.2·√(1 − 0.36) = 0.96, so its bbox spans y ∈ [0.96, \
1.76] — NOT [1.20, 2.00]. Getting this wrong leaves the tree visibly \
floating or sunk. For BOX targets the rule collapses to the familiar \
"bottom face at y_max".\
"""


def _render_proxy_shape(p: ProxyShape | None) -> str:
    return p.value if p is not None else "BOX"


# Shared anti-ephemera guidance injected into every prompt that authors
# scene content (zone plans, zone decomposition, object decomposition,
# next-object polish). The downstream text-to-3D model produces solid
# meshes; gaseous, volumetric, or luminous phenomena render as garbage
# blobs that drag the whole scene down. Centralised here so the
# vocabulary of forbidden phenomena stays consistent across steps.
NO_EPHEMERA_DOC = """\
NO EPHEMERA. The downstream renderer produces SOLID, OPAQUE, BOUNDED \
3D meshes — it cannot represent gases, plasmas, particulate clouds, \
volumetric light, or any phenomenon that lacks a hard surface. Naming \
such phenomena as features, anchors, plan elements, or ambient fill \
produces deformed mesh blobs that visibly tank the scene. Do NOT \
introduce, plan, enumerate, or describe any of the following as \
things the scene must depict:

  * GASES & VAPOURS — fog, mist, haze, smoke, steam, vapour, smog, \
    exhaust plumes, dust clouds, pollen clouds, sandstorms, snow \
    flurries in the air, falling rain or snow as discrete particles.
  * CLOUDS & SKY VOLUMES — clouds, nebulae, gas giants' atmospheres, \
    aurora curtains, rainbows, sunbeams / god rays, light shafts.
  * PLASMAS & ENERGY — lightning bolts, electrical arcs, plasma \
    discharges, fire flames as freestanding objects, sparks, embers \
    in flight, magical glows, force fields, beams of light, laser \
    beams, comet tails, meteor trails, contrails.
  * LIQUIDS IN MOTION — splashes, sprays, waterfalls as freestanding \
    objects, fountains' water arcs, pouring streams, ripples.

You MAY still IMPLY these phenomena through tangible, solid \
consequences that DO have hard surfaces: wet flagstones instead of \
rain, scorched and split bark instead of lightning, soot stains and \
charred timbers instead of smoke, a frost crust instead of fog, \
puddles and damp moss instead of drizzle, a fire pit with glowing \
embers (a solid bowl of coals) instead of freestanding flames, a \
chimney instead of a smoke plume. Atmosphere is conveyed by what the \
weather has DONE to solid surfaces, not by depicting the weather \
itself. A flat-water surface (a pond, a puddle, a lake skin) IS \
allowed because it is a bounded plane; freestanding water in motion \
is not.\
"""


# ---------- Step 1: zone plan (high-level authoring; runs for every zone) ---


class ZonePlanOutput(BaseModel):
    plan: str
    is_atomic: bool


SYSTEM_ROOT_ZONE_PLAN = (
    """\
You are authoring the SCENE PLAN for a 3D scene being built for \
StarshotBench — a head-to-head LLM benchmark for spatial reasoning \
where your generated 3D environment is rendered and judged against \
another LLM's environment from the same user prompt. As a spatial \
reasoning competition, judges weigh creativity & impressiveness, \
compositional coherence, recognizability and plausibility. Remember, \
they see only the final scene after the entire pipeline has run \
through hundreds of downstream generation steps; NOT the prompts and \
NOT the plan.

Your job is to TRANSFORM the user's prompt into a vivid, opinionated, \
concrete vision for what this scene IS. The prompt is a seed \
(e.g. "a beautiful modern mansion", "a low muddy island", "the \
imperial box"). Your output is the AUTHORED VISION — the north-star \
that every descendant zone inherits — that makes the seed specific, \
distinctive, and memorable. A vague, template-y plan produces a scene \
that reads as stock assets. A vivid, committed plan propagates \
specificity all the way to the rendered mesh — and the judges see \
the difference. Don't play it safe.

You are planning the ROOT zone. The root is a PURELY ABSTRACT, \
INTANGIBLE META-CONTAINER for the entire world — it has no walls, \
no floor, no ceiling, no surface, no skin, and never gets a frame, \
mesh, or geometry of its own. It is the canvas inside which the \
actual scene (rooms, buildings, terrain, enclosures) is placed as \
CHILD zones. If the user prompt names a single tangible enclosure \
that needs walls/floor/ceiling (a hotel room, a throne room, a \
garage, a cockpit), the ROOT is NOT that enclosure — the enclosure \
is a child zone INSIDE the root, and the root's plan should describe \
the enclosure as the singular subzone within an otherwise empty world \
canvas.

<zone_definition>
A zone is a REGION OF THE SCENE — a subscene, an area, a place large \
enough to contain multiple distinct objects arranged inside it (a \
master bedroom, the left audience stand of an arena, an entire mansion, a city \
block, etc.). It is SPATIAL — it has room inside it, and its \
character comes from the ensemble of things that live there, not from \
any single object. A single landmark, monument, trophy, centerpiece, \
or hero prop — no matter how important — is an OBJECT inside a zone, \
NOT a zone of its own.
</zone_definition>

<atomic_vs_subdivides>
You also decide whether this zone is ATOMIC — a leaf whose content \
is individual objects, generated later — or SUBDIVIDES into child \
zones with their own plans. This is the single most consequential \
structural decision in the pipeline.

Over-subdivide and the scene splinters into geography without focus — \
an island split into "north end / central mound / south end" becomes \
three forgettable regions where there should be ONE memorable island. \
Under-subdivide and loci of genuine interest disappear into \
undifferentiated masses — a palace interior treated as one atomic zone becomes \
a blob where there should be a downstairs zone with a throne room zone, a great hall, \
a war room, etc. and an upstairs zone with multiple bedrooms, bathrooms, showers, etc.

Default to ATOMIC. Subdivision is opt-in. A zone subdivides ONLY \
when it genuinely contains TWO OR MORE distinct loci of interest, \
each deserving its own dedicated plan and generation pass. Let your \
plan guide the decision: if it names multiple distinct loci (a \
mansion's house + garden + stables; a hotel room's bedroom + \
bathroom), set is_atomic=false. If it describes a single coherent \
place, set is_atomic=true.

The root is a PURELY ABSTRACT, INTANGIBLE META-CONTAINER with no \
walls, floor, ceiling, or geometry of its own — only child zones \
get those. If the user prompt names a single tangible enclosure that \
needs a frame (a hotel room, a throne room, a garage, a cockpit — \
anything with walls/floor/ceiling), you MUST set is_atomic=false. \
Marking the root atomic leaves the scene with no enclosure at all.

If you cannot give each proposed sub-region a unique, named reason \
to exist beyond "this is the left part" or "this is the denser \
part", set is_atomic=true.

The sharpest test: if the "thing" you would carve out is essentially \
ONE SINGLE ARTIFACT — a statue, a monument, a throne, a fountain, \
a hearth, a single tree — it is an OBJECT inside a zone, NOT a zone \
of its own.

A zone is NEVER: an individual object; a distribution/scatter of \
similar items; a surface or connective medium; a band/ring/core \
defined only by density; or NEGATIVE SPACE (the ambient medium \
between zones — a separate pass fills this).

Good: mansion grounds → house, formal garden, stables, rear orchard. \
Good: hotel room → bathroom, bedroom. \
Bad: swamp → open water, lilypad distribution, log debris. \
Bad: island → north end, central mound, south end. \
Bad: bedroom → bed area, dresser area, reading nook.
</atomic_vs_subdivides>

<what_to_write>
ONE cohesive paragraph (no headers, no lists), roughly 5-10 \
sentences, that:
  * Sets the CHARACTER — the mood, era, palette, materials, lighting \
    feel, silhouette of this region. Commit to a point of view: is \
    the mansion sun-bleached coastal modernism or brooding hillside \
    concrete? Is the island lush moss-cushioned or bleached \
    salt-crusted? The user did not pick; you do.
  * Names the FEATURES OF INTEREST that give the region its \
    identity — focal points, landmarks, anchors a viewer would point \
    at (a stone hearth, a central fountain, a writing desk, a lone \
    lightning-split stump). Be concrete about what makes each \
    distinctive.
  * Suggests the OVERALL SILHOUETTE / SHAPE of the scene — tall and \
    narrow (a skyscraper), long and flat (a river), wide and shallow \
    (a coastal vista), roughly cubic (a room). This feeds directly \
    into the next step, which sizes the canvas.
</what_to_write>

<what_NOT_to_write>
  * Do NOT enumerate CHILD ZONES. Do not list specific sub-regions \
    as a planned tree ("the children are X, Y, Z") or structure the \
    subtree. You decide is_atomic (whether this zone subdivides at \
    all); a downstream ZONE DECOMPOSE step authors the actual \
    children. You may describe the region as containing distinct \
    loci of character to support your is_atomic decision.
  * Do NOT enumerate individual OBJECTS. No "a fountain", "a chair", \
    "a chandelier", "a single tree", "a red flag" as a list of \
    things to place. Object selection happens later in each atomic \
    region's generation pass — those steps need AGENCY to pick \
    objects in service of your character. You may describe the \
    character those objects will collectively express ("the imperial \
    box reads as ornate and ceremonial"), but stop short of listing \
    the props.
  * Do NOT pick coordinates, dimensions, counts, materials by brand, \
    or specific instances. Stay at the level of mood, palette, \
    identity, and silhouette.
  * Do NOT describe camera, framing, or rendering. The pipeline \
    handles those.
</what_NOT_to_write>

<no_ephemera>
"""
    + NO_EPHEMERA_DOC
    + """
</no_ephemera>

<inputs>
  * The zone being planned: its id ('root') and the user's prompt.
  * The GENERATED OBJECTS — every concrete (mesh-bearing) object \
    placed anywhere in the scene so far, with its parent. These are \
    what the scene actually LOOKS like at this point in the run; \
    lean on them to know what is already real and to avoid \
    contradicting them.
  There is no ancestor chain — this IS the root.
</inputs>

Respond with ONE JSON object matching the schema. `plan` holds the \
paragraph; `is_atomic` is true if this zone is a leaf (no child \
zones), false if it should subdivide. No prose, no markdown, no \
code fences.\
"""
)


SYSTEM_ZONE_PLAN = (
    """\
You are authoring the HIGH-LEVEL PROMPT for a region of a 3D scene being built for StarshotBench — a head-to-head LLM benchmark for spatial reasoning where your generated 3D environment is rendered and judged against another LLM's environment from the same user prompt. As a spatial reasoning competition, judges weigh creativity & impressiveness, compositional coherence, recognizability and plausibility. Remember, they see only the final scene after the entire pipeline has run through hundreds of downstream generation steps; NOT the prompts and NOT the plan. 

Your job is to TRANSFORM the region's prompt into a vivid, \
opinionated, concrete vision for what this region IS. The prompt is a \
seed (e.g. "a beautiful modern mansion", "a low muddy island", "the \
imperial box"). Your output is the AUTHORED VISION that makes the \
seed specific, distinctive, and memorable. A vague, template-y plan \
produces a region that reads as stock assets. A vivid, committed plan \
propagates specificity all the way to the rendered mesh — and the \
judges see the difference. Don't play it safe.

This is a NESTED region — your plan must stay consistent with the \
character your direct ancestors committed to (shown in the inputs) \
and must not contradict any concrete object already generated in the \
scene.

<zone_definition>
A zone is a REGION OF THE SCENE — a subscene, an area, a place large \
enough to contain multiple distinct objects arranged inside it (a \
master bedroom, the left audience stand of an arena, an entire mansion, a city \
block, etc.). It is SPATIAL — it has room inside it, and its \
character comes from the ensemble of things that live there, not from \
any single object. A single landmark, monument, trophy, centerpiece, \
or hero prop — no matter how important — is an OBJECT inside a zone, \
NOT a zone of its own.
</zone_definition>

<atomic_vs_subdivides>
You also decide whether this zone is ATOMIC — a leaf whose content \
is individual objects, generated later — or SUBDIVIDES into child \
zones with their own plans. This is the single most consequential \
structural decision in the pipeline.

Over-subdivide and the scene splinters into geography without focus — \
an island split into "north end / central mound / south end" becomes \
three forgettable regions where there should be ONE memorable island. \
Under-subdivide and loci of genuine interest disappear into \
undifferentiated masses — a palace interior treated as one atomic zone becomes \
a blob where there should be a downstairs zone with a throne room zone, a great hall, \
a war room, etc. and an upstairs zone with multiple bedrooms, bathrooms, showers, etc.

Default to ATOMIC. Subdivision is opt-in. A zone subdivides ONLY \
when it genuinely contains TWO OR MORE distinct loci of interest, \
each deserving its own dedicated plan and generation pass. Let your \
plan guide the decision: if it names multiple distinct loci (a \
mansion's house + garden + stables; a hotel room's bedroom + \
bathroom), set is_atomic=false. If it describes a single coherent \
place, set is_atomic=true.

If you cannot give each proposed sub-region a unique, named reason \
to exist beyond "this is the left part" or "this is the denser \
part", set is_atomic=true.

The sharpest test: if the "thing" you would carve out is essentially \
ONE SINGLE ARTIFACT — a statue, a monument, a throne, a fountain, \
a hearth, a single tree — it is an OBJECT inside a zone, NOT a zone \
of its own.

A zone is NEVER: an individual object; a distribution/scatter of \
similar items; a surface or connective medium; a band/ring/core \
defined only by density; or NEGATIVE SPACE (the ambient medium \
between zones — a separate pass fills this).

Good: mansion grounds → house, formal garden, stables, rear orchard. \
Good: hotel room → bathroom, bedroom. \
Bad: swamp → open water, lilypad distribution, log debris. \
Bad: island → north end, central mound, south end. \
Bad: bedroom → bed area, dresser area, reading nook.
</atomic_vs_subdivides>

<what_to_write>
ONE cohesive paragraph (no headers, no lists), roughly 5-10 \
sentences, that:
  * Sets the CHARACTER — the mood, era, palette, materials, lighting \
    feel, silhouette of this region. Commit to a point of view: is \
    the mansion sun-bleached coastal modernism or brooding hillside \
    concrete? Is the island lush moss-cushioned or bleached \
    salt-crusted? The user did not pick; you do.
  * Names the FEATURES OF INTEREST that give the region its \
    identity — focal points, landmarks, anchors a viewer would point \
    at (a stone hearth, a central fountain, a writing desk, a lone \
    lightning-split stump). Be concrete about what makes each \
    distinctive.
</what_to_write>

<what_NOT_to_write>
  * Do NOT decide structural decomposition. Do not say "this \
    subdivides", "this is atomic", "the children are X and Y", or \
    enumerate sub-regions as a planned tree. A separate downstream \
    step (the ZONE DECOMPOSE step) makes that call using your plan \
    as input — pre-empting it cripples that step. You may describe a \
    region as containing distinct loci of character ("the grounds \
    pivot between a formal front garden and a wild rear orchard"); \
    the decomposer will read that and decide whether to subdivide.
  * Do NOT enumerate individual OBJECTS. No "a fountain", "a chair", \
    "a chandelier", "a single tree", "a red flag" as a list of \
    things to place. Object selection happens later in each atomic \
    region's generation pass — those steps need AGENCY to pick \
    objects in service of your character. You may describe the \
    character those objects will collectively express ("the imperial \
    box reads as ornate and ceremonial"), but stop short of listing \
    the props.
  * Do NOT pick coordinates, dimensions, counts, materials by brand, \
    or specific instances. Stay at the level of mood, palette, \
    identity, and silhouette.
  * Do NOT describe camera, framing, or rendering. The pipeline \
    handles those.
</what_NOT_to_write>

<no_ephemera>
"""
    + NO_EPHEMERA_DOC
    + """
</no_ephemera>

<inputs>
  * The region being planned: its id and prompt.
  * The ANCESTOR CHAIN — every region above this one in the tree, \
    root first, each with its plan. The root's plan IS the SCENE \
    PLAN.
  * The GENERATED OBJECTS — every concrete (mesh-bearing) object \
    placed anywhere in the scene so far, with its parent. These are \
    what the scene actually LOOKS like at this point in the run; \
    lean on them to know what is already real and to avoid \
    contradicting them. Lateral peer regions that are still abstract \
    are NOT shown — only ancestors and concrete objects flow into \
    your context.
</inputs>

Respond with ONE JSON object matching the schema. `plan` holds the \
paragraph; `is_atomic` is true if this zone is a leaf (no child \
zones), false if it should subdivide. No prose, no markdown, no \
code fences.\
"""
)


def render_zone_plan(
    *,
    zone_id: str,
    zone_prompt: str,
    ancestors: list[tuple[str, str, str]],
    objects: list[tuple[str, str, str | None]],
) -> str:
    """ancestors: (id, prompt, plan) tuples from root → parent of this zone,
    excluding the zone itself. Empty for the root.
    objects: (id, prompt, parent_id) tuples for every concrete (mesh-bearing)
    node placed anywhere in the run so far."""
    zone_block = f"ZONE_ID (being planned): {zone_id!r}\nZone prompt: {zone_prompt!r}"
    if ancestors:
        ancestor_block = "\n".join(
            f"  - id={aid!r}\n    prompt: {aprompt}\n    plan: {aplan}" for aid, aprompt, aplan in ancestors
        )
    else:
        ancestor_block = "  (none — this zone is the root)"
    if objects:
        obj_block = "\n".join(
            f"  - id={oid!r} parent={oparent!r}: {oprompt}" for oid, oprompt, oparent in objects
        )
    else:
        obj_block = "  (none — no concrete objects placed yet)"
    if not ancestors:
        return (
            f"{zone_block}\n\n"
            f"Generated objects placed so far:\n{obj_block}\n\n"
            "This is the ROOT zone — the whole scene. Your plan IS the "
            "SCENE PLAN that every descendant inherits, and it directly "
            "shapes the canvas-sizing step that runs immediately after "
            "this one. Author the high-level vision."
        )
    return (
        f"Ancestor chain (root first → your direct parent, with each "
        f"ancestor's plan):\n{ancestor_block}\n\n"
        f"Generated objects placed so far:\n{obj_block}\n\n"
        f"{zone_block}\n\n"
        "Author this region's high-level plan, consistent with the "
        "ancestor chain above."
    )


# ---------- Step 2: overall bbox --------------------------------------------


class OverallBboxOutput(BaseModel):
    bbox: BoundingBox


SYSTEM_OVERALL_BBOX = """\
You are picking the OVERALL bounding box for a 3D scene — the SCENE'S \
CANVAS that every zone, object, and ambient element will be placed \
inside. This box is a PURELY ABSTRACT, INTANGIBLE META-CONTAINER for \
the world: it has no walls, no floor, no ceiling, no skin, and never \
becomes a tangible frame or mesh. It only sets the outer extents of \
the world the scene lives in. If the scene is a single tangible \
enclosure (a hotel room, a throne room, a cockpit), the canvas \
should be SLIGHTLY LARGER than that enclosure, so the actual room \
fits comfortably as a child zone inside it with a small margin of \
empty world around it — the canvas is NOT the room. This pipeline is \
part of StarshotBench, a head-to-head LLM benchmark. The SCENE PLAN \
has already been authored upstream and is shown to you in the \
inputs; your job is to size the canvas so it matches the silhouette \
that plan implies. Get this wrong and every \
downstream step is fighting the canvas — a skyscraper crammed into a \
cube, a river squeezed into a square, a room ballooned into a \
warehouse. Match the scene's actual silhouette: a skyscraper is tall \
and narrow, a river is long and flat, a room is modest in every \
dimension.

The bounding box is axis-aligned, in meters, interpreted under the \
CANONICAL FRONT VIEW: +X = right, +Y = up, +Z = toward the viewer \
(front), -Z = back. It is defined by an `origin` vertex and a signed \
`dimensions` vector `(dx, dy, dz)` extending from that vertex; the \
sign of each component chooses the direction of expansion along that \
axis.

Emit all coordinates to centimeter precision — two decimal places, \
exact multiples of 0.01 m. Place the origin sensibly (often the world \
origin; floor at y=0 for architectural scenes) and choose signs so the \
box extends into the region you intend.

Respond with ONE JSON object matching the schema. No prose, no markdown, no code fences.\
"""


def render_overall_bbox(user_prompt: str, scene_plan: str) -> str:
    return (
        f"User prompt for the scene: {user_prompt!r}\n\n"
        f"SCENE PLAN (authored upstream — size the canvas to match its "
        f"implied silhouette):\n{scene_plan}\n\n"
        "Produce the overall bounding box for the whole scene."
    )


# ---------- Step 3: zone decompose (atomic vs subzones; runs after plan) ----


class ChildNodeSpec(BaseModel):
    id: str
    prompt: str
    proxy_shape: ProxyShape | None = None
    relationships: list[Relationship]

    @field_validator("proxy_shape", mode="before")
    @classmethod
    def _box_means_none(cls, v: object) -> object:
        # The prompt describes BOX as "null/omitted" — no enum value — but
        # some models emit the literal string "BOX" anyway. Treat it as None.
        if isinstance(v, str) and v.upper() == "BOX":
            return None
        return v


class ZoneDecomposeOutput(BaseModel):
    children: list[ChildNodeSpec]


SYSTEM_ZONE_DECOMPOSE = (
    """\
You are deciding the STRUCTURE of a zone in a 3D scene being built for \
StarshotBench — a head-to-head LLM benchmark for spatial reasoning where your \
generated 3D environment is rendered and judged against another LLM's environment from \
the same user prompt. As a spatial reasoning competition, judges weigh creativity \
& impressiveness, compositional coherence, recognizability, detail richness, and \
plausibility. Remember, they see only the final scene; not the prompts, not the plan. 

The zone's HIGH-LEVEL PLAN was already authored by an \
upstream step and is shown to you in the inputs as the ZONE \
PLAN. Specifically, it was decided that this zone should decompose into further zones. \

Your job: Based on the HIGH-LEVEL PLAN for this zone, create its STRUCTURE by describing \
all of the sub-zones that exist within it. You are taking an idea and fleshing out the \
structure and details from it.

Emit each child created in the zone fully structured — `id`, `prompt`, \
optional `proxy_shape`, and the `relationships` that anchor it inside \
the parent. Subdivision and per-child structure are codependent (you \
cannot anchor children you have not committed to, and you cannot \
commit to children without imagining their layout), so they belong \
in the same step.

You do NOT author each child's full, fleshed out plan. Each child's detailed plan \
is authored later, by the child's own ZONE PLAN step when the pipeline \
recurses into it. You also do NOT pick concrete bbox coordinates or dimensions — a \
downstream batch step resolves each child's bbox from its \
relationships and prompt.

<zone_definition>
A zone is a REGION OF THE SCENE — a subscene, an area, a place large \
enough to contain multiple distinct objects arranged inside it (e.g a \
master bedroom, the left audience stand of an arena, the formal front \
garden of a mansion). It is SPATIAL — it has room inside it, and its \
character comes from the ensemble of things that live there, not from \
any single object. A single landmark, monument, trophy, centerpiece, \
or hero prop — no matter how important — is an OBJECT inside a zone, \
NOT a zone of its own. Zones often contain other zones within them, and these zones can have different structures - e.g the bottom floor of a massive palace might contain a \
war zone and a living zone, where the war zone then further decomposes into a set of war rooms, armory rooms and a fighting arena, and the living zone decomposes into the \
great hall, throne room, the front entrance, the king and queen's chambers, a garden, \
etc. Or, the entire bottom floor zone might decompose directly into a set of zones \
for the grand entrance, inner courtyard, great hall, trophy gallery, king and \
queen's chambers, throne room and bathrooms (as individual zones).

Zones are designed realistically, based on the given input, intended creative direction \
and amount of world space available to define them within.
</zone_definition>

<inputs>
  * The zone being decomposed: its id (PARENT_ID), prompt, and \
    axis-aligned bounding box (in meters, under the canonical front \
    view: +X right, +Y up, +Z front).
  * The ZONE PLAN — the high-level character/intent plan authored \
    upstream for this zone. This is your primary signal. Let its \
    named features and implied loci drive your decision.
  * The SCENE PROMPT and SCENE PLAN — the root's prompt and plan, \
    the north-star for the whole scene.
  * The ANCESTOR CHAIN — every zone above this one in the tree, \
    root first, each with its plan.
  * The PRIOR ZONES — every non-root zone already declared in the \
    run, with its parent and plan. Lateral context: siblings, \
    cousins, and earlier subtrees may inform how THIS zone is \
    structured and anchored.
  * The GENERATED OBJECTS — every concrete (mesh-bearing) object \
    placed anywhere in the scene so far, with its parent.
</inputs>

<output>
Emit a single ZoneDecomposeOutput:

  * `children` — list of ChildNodeSpec. At least two children (or \
    exactly one when the root names a single tangible enclosure). \
    Each ChildNodeSpec has:
      - `id` — unique within the whole scene (do not collide with \
        any existing id, including the ancestor chain or generated \
        objects).
      - `prompt` — a 1-2 sentence concrete description of the child \
        zone — rich enough to seed the child's own high-level plan \
        when its ZONE PLAN step runs, but NOT a full plan. Name \
        the region and its defining feature; do not pre-author its \
        mood, palette, or list of objects.
      - `proxy_shape` — OPTIONAL. The child zone's collision-proxy \
        shape, chosen from the silhouette implied by the prompt (a \
        domed island → HEMISPHERE; a column or trunk-shaped region \
        → CAPSULE; most architectural zones → omit). See the PROXY \
        SHAPE section below.
      - `relationships` — REQUIRED, at least one per child. Anchors \
        the child inside the parent.

A Relationship has:
  * `target` — either PARENT_ID or the `id` of an earlier sibling \
    already listed in this call's `children`.
  * `kind` — one of: ON, BESIDE, BELOW, ABOVE, ATTACHED.
  * `reference_point` — which CORNER of the TARGET's bbox this \
    relationship anchors against, under the canonical front view \
    (+X right, +Y up, +Z front). One of: TOP_LEFT_FRONT, \
    TOP_LEFT_BACK, TOP_RIGHT_FRONT, TOP_RIGHT_BACK, \
    BOTTOM_LEFT_FRONT, BOTTOM_LEFT_BACK, BOTTOM_RIGHT_FRONT, \
    BOTTOM_RIGHT_BACK.

Do NOT pick concrete coordinates or dimensions — a downstream batch \
step resolves each child's bbox from its relationships and prompt.
</output>

<proxy_shape>
"""
    + PROXY_SHAPE_DOC
    + """
</proxy_shape>

<no_ephemera>
"""
    + NO_EPHEMERA_DOC
    + """
</no_ephemera>

Respond with ONE JSON object matching the schema. No prose, no markdown, no code fences.\
"""
)


def render_zone_decompose(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_bbox: BoundingBox,
    zone_plan: str,
    ancestors: list[tuple[str, str, str]],
    objects: list[tuple[str, str, str | None]],
    scene_prompt: str,
    scene_plan: str,
    prior_zones: list[tuple[str, str, str, str]],
) -> str:
    """ancestors: (id, prompt, plan) tuples from root → parent of this zone,
    excluding the zone itself. Empty for the root.
    objects: (id, prompt, parent_id) tuples for every concrete (mesh-bearing)
    node placed anywhere in the run so far.
    prior_zones: (id, prompt, plan, parent_id) for every non-root zone
    already declared in the run, in declaration order."""
    zone_block = (
        f"PARENT_ID (the zone being decomposed): {zone_id!r}\n"
        f"Zone prompt: {zone_prompt!r}\n"
        f"Zone bbox: {zone_bbox.model_dump_json()}\n"
        f"ZONE PLAN (authored upstream — your primary signal):\n{zone_plan}"
    )
    if ancestors:
        ancestor_block = "\n".join(
            f"  - id={aid!r}\n    prompt: {aprompt}\n    plan: {aplan}" for aid, aprompt, aplan in ancestors
        )
    else:
        ancestor_block = "  (none — this zone is the root)"
    if prior_zones:
        prior_block = "\n".join(
            f"  - id={zid!r} parent={zparent!r}\n    prompt: {zprompt}\n    plan: {zplan}"
            for zid, zprompt, zplan, zparent in prior_zones
        )
    else:
        prior_block = "  (none)"
    if objects:
        obj_block = "\n".join(
            f"  - id={oid!r} parent={oparent!r}: {oprompt}" for oid, oprompt, oparent in objects
        )
    else:
        obj_block = "  (none — no concrete objects placed yet)"
    return (
        f"Overall scene prompt: {scene_prompt!r}\n\n"
        f"SCENE PLAN (the north-star for the whole scene):\n{scene_plan}\n\n"
        f"Ancestor chain (root first → your direct parent, with each "
        f"ancestor's plan):\n{ancestor_block}\n\n"
        f"Prior zones declared so far (lateral scene context):\n"
        f"{prior_block}\n\n"
        f"Generated objects placed so far:\n{obj_block}\n\n"
        f"{zone_block}\n\n"
        "This zone has been determined to subdivide. Emit one "
        "ChildNodeSpec per child — id, prompt, optional proxy_shape, "
        "and the relationships that anchor it inside the parent."
    )


# ---------- Step 4: zone bbox batch resolution (all siblings at once) -------


class BboxAssignment(BaseModel):
    id: str
    bbox: BoundingBox


class BboxBatchOutput(BaseModel):
    assignments: list[BboxAssignment] = Field(default_factory=list)


SYSTEM_ZONE_BBOX_BATCH = (
    """\
You are a constraint solver. Place ALL sibling child ZONES inside a \
parent zone in one shot, deriving each child's axis-aligned bounding \
box from the given inputs (parent bbox, child specs, relationships). \
This step has no creative latitude — your job is to produce \
coordinates that satisfy every stated constraint simultaneously.

Inputs:
  * Parent bbox (the enclosing zone).
  * A list of child specs — id, prompt, and relationships that target \
    either the parent or another child in this same list.

Relationships carry `kind` (ON, BESIDE, BELOW, ABOVE, ATTACHED) and a \
`reference_point` — a corner of the TARGET's bbox under the canonical \
front view (+X right, +Y up, +Z front).

Each child carries a `proxy_shape` describing its mesh silhouette \
inside its AABB — BOX, SPHERE, CAPSULE, or HEMISPHERE. The PROXY \
SHAPE section below gives the exact surface formula Y_top(x, z) for \
each. When a child with a non-BOX proxy is the TARGET of another \
child's ON relationship, the ON-child rests on the target's proxy \
TOP SURFACE at its XZ centre — NOT on the target's AABB top face. \
There is no automatic correction: YOU must compute the target's \
Y_top(x, z) from the target's AABB and proxy formula, and place the \
ON-child's bbox so its bottom face Y equals that value. Pick \
dimensions so a HEMISPHERE target has vertical headroom above its \
apex for the things that sit on it.

Produce one assignment per child (id + bbox) such that:
  * Every bbox lies fully inside the parent bbox.
  * No two child bboxes overlap volumetrically. Touching at a shared \
    face is fine; eating into another bbox's volume is not.
  * Every relationship is respected: for each one, the child is \
    anchored near the named corner of its target, in the direction \
    implied by `kind` (ABOVE → higher y; BESIDE → adjacent on x or z; \
    ON → the child's bottom face at the target's Y_top(x, z) — the \
    AABB top face y_max for BOX targets, the proxy formula for \
    SPHERE/CAPSULE/HEMISPHERE targets; ATTACHED → touching the \
    target).
  * Dimensions are appropriate to each child's prompt.

Because you are deciding the entire layout at once, RESERVE SPACE for \
every child up front rather than committing each bbox in isolation. A \
later child's requirements must influence earlier siblings' sizing.

Coordinates in meters, centimeter precision (multiples of 0.01). Use \
a signed `dimensions` vector from an `origin` vertex; sign chooses \
expansion direction. Emit exactly one assignment per requested child \
id, no extras, no omissions.

<proxy_shape>
"""
    + PROXY_SHAPE_DOC
    + """
</proxy_shape>

Respond with ONE JSON object matching the schema. No prose, no markdown, no code fences.\
"""
)


def render_zone_bbox_batch(
    *,
    parent_id: str,
    parent_bbox: BoundingBox,
    children: list["ChildNodeSpec"],
) -> str:
    child_lines = "\n\n".join(
        f"  - id={c.id!r}\n"
        f"    prompt: {c.prompt}\n"
        f"    proxy_shape: {_render_proxy_shape(c.proxy_shape)}\n"
        f"    relationships:\n"
        + (
            "\n".join(
                f"      * target={r.target!r} kind={r.kind.value} reference_point={r.reference_point.value}"
                for r in c.relationships
            )
            or "      (none)"
        )
        for c in children
    )
    return (
        f"Parent id: {parent_id!r}\n"
        f"Parent bbox: {parent_bbox.model_dump_json()}\n\n"
        f"Children to place ({len(children)}):\n{child_lines}\n\n"
        "Produce a bbox for every child in a single coherent layout."
    )


# ---------- Step 5: object decomposition (Phase 2) --------------------------


class ObjectSpec(ChildNodeSpec):
    """A single object in a zone. Inherits id/prompt/relationships."""

    parent: str
    orientation: Orientation = 0


class ObjectDecompOutput(BaseModel):
    objects: list[ObjectSpec] = Field(default_factory=list)


SYSTEM_OBJECT_DECOMP = (
    """\
You are enumerating the OBJECTS that populate a 3D scene zone inside \
StarshotBench — a head-to-head competitive benchmark for 3D spartial reasoning \
where your scene will be rendered and judged against another LLM's rendering of the \
same user prompt.

The final spatial and aesthetic output of the scene you produce here \
is WHAT THE JUDGES ACTUALLY SEE. \
Zones and plans are scaffolding; objects are the scene. Thoughtful \
anchor choices make a zone unmistakably, immediately recognizable as \
its subject — a meeting room's long conference table, chairs and screen on \
the end wall; a castle throne room's raised dais, carved chair, and \
flanking banners; an island's lone lightning-split stump, knotted \
roots at the waterline, and the red objective flag planted at its \
crest. Generic choices — "a chair", "a tree", "a stone" — make the \
zone read as a stock kit of parts. The delta between a masterful \
scene and a mediocre one is largely the quality of your object \
decisions HERE.

Push past the obvious first pick. An adequate LLM emits "a wooden \
table, four chairs, a TV"; a winning LLM emits "a scarred oak \
boardroom table with leather conference chairs around it, a \
wall-mounted 75-inch display, a whiteboard, a water pitcher on a \
tray at one end". Specificity propagates all the way to the rendered \
mesh — the image model, the 3D model, and the final render are \
directly downstream of the words you write.

Remember to give each zone a story, and make it as impressive as possible.

Three modes are available — ANCHOR, ENCAPSULATING, NEGATIVE-SPACE. \
Read the MODE header carefully; each has its own purpose and its own \
rules.

<modes>
You operate in one of three MODES:

* ANCHOR mode — the zone is an atomic leaf (e.g. "meeting room", "toilet \
  area", "hero island"). Enumerate the DEFINING anchor objects that make \
  the zone unmistakably what it is. A meeting room: a large table, chairs \
  around it, a TV on the end wall. A toilet area: a toilet, a toilet \
  paper holder. An island: the flag, the roots at the waterline, a \
  gnarled tree. Do NOT include decorative filler; a later iterative step \
  adds more objects one at a time.

  GROUND-AWARENESS RULE. This zone may already have a GROUND / SHELL \
  peer placed by the encapsulating pass (an island dome, a crater bowl, \
  a curved floor, the walls+floor of a room) — look at the CURRENT SCENE \
  for a peer whose parent is this zone and whose prompt describes \
  terrain or enclosure geometry. If such a peer exists, every anchor \
  object in this zone whose physical support IS that terrain/floor \
  MUST set its `parent` to the ground/shell peer's id (NOT the zone id) \
  and include an ON relationship targeting it. The peer's \
  `proxy_shape` (shown alongside its bbox in the CURRENT SCENE) is the \
  authoritative descriptor of its surface — a HEMISPHERE peer is a \
  dome whose real surface dips from the AABB centre to the edges. \
  You are NOT placing bboxes at this step, but choose the right \
  parent and relationship now: the downstream bbox-resolution step \
  will compute the dome's surface height at each anchor's XZ from \
  the peer's proxy formula and rest the anchor on that surface. Do \
  NOT re-emit the ground itself in anchor mode; the encapsulating \
  pass already placed it.

* ENCAPSULATING mode — the zone needs its physical SHELL / FLOOR / \
  BOUNDARY placed before anything else populates it. For architectural \
  zones about to be decomposed further: the walls, ceiling, floor, \
  enclosing fence, moat, cliff face — whatever physically bounds this \
  zone. For atomic-terrain zones: the GROUND mesh itself (an island \
  dome, a crater bowl, a hill, a curved floor, a mound). Emit one \
  object per shell element. Each object's prompt is sent verbatim to a \
  text-to-3D model, so describe it as a concrete artifact — and for \
  ground/terrain shells, describe the actual surface SHAPE in concrete \
  terms so later anchor-mode placements can reason about the surface \
  height at any XZ ("a muddy domed island raised ~1.2m at the centre, \
  tapering to the waterline at irregular edges"; "a rocky crater bowl \
  with steep inner walls descending ~3m below the rim"; "a tall stone \
  wall with ivy"; "a wooden plank floor").

* NEGATIVE-SPACE mode — you are filling the AMBIENT, CONNECTIVE, \
  INTERSTITIAL space of the scene (or a zone) with drifting, background, \
  or distribution-style content that doesn't belong to any specific \
  zone: lilypads drifting across swamp water between islands, grass \
  tufts scattered over a meadow, floating logs and driftwood across \
  open water, scattered stones across a plain, loose paper blowing \
  across a plaza. Every item must be a SOLID, BOUNDED object (see NO \
  EPHEMERA below) — no clouds, no mist, no fog banks, no smoke plumes, \
  no light shafts; if the scene calls for atmosphere, convey it through \
  the solid, weathered surfaces it leaves behind. This mode runs over the scene root (or \
  another zone that explicitly owns its negative space) once its zones \
  and anchors are placed, so the CURRENT SCENE lists every zone and \
  object already committed. Enumerate the ambient/drifting objects that \
  the scene prompt implies should populate the space BETWEEN and AROUND \
  those placed nodes — individual instanced objects, not abstractions. \
  Set each object's `parent` to the zone id UNLESS it physically rests \
  on an existing peer (a lilypad on an implicit water surface still \
  parents to the zone; a barnacle crusting a sunken log parents to the \
  log). Do NOT re-emit anything that already exists as a zone or a \
  zone's anchor — negative-space content is strictly the ambient layer \
  that named zones do not own.
</modes>

<inputs>
You are given the CURRENT SCENE — every node already placed anywhere in \
the run so far, with id, prompt, bbox, and parent. Reason about it thoroughly \
before emitting.
</inputs>

<global_rules>
  * DO NOT DUPLICATE GEOMETRY. If an ancestor zone or an adjacent sibling \
    zone has already placed a wall / floor / ceiling that covers one of \
    this zone's faces, do NOT emit another one for that face. A \
    neighbouring wall that sits exactly on the shared plane is already \
    doing the job; emitting a second wall there produces a duplicate \
    mesh. This matters most in ENCAPSULATING mode, where thin slabs at \
    zone boundaries are easy to accidentally re-emit.
</global_rules>

<per_object_fields>
For each object, emit:
  * `id` — unique within this call.
  * `prompt` — a detailed description of the object; will be used verbatim \
    as a text-to-3D generation prompt.
  * `parent` — the SEMANTIC parent. Either the enclosing zone id (provided \
    below as ZONE_ID), or the id of ANOTHER object in this list that this \
    one belongs to. A lamp resting on a desk: the lamp's parent is the \
    desk. A book on a shelf: the book's parent is the shelf. Parent does \
    NOT imply spatial containment — a lamp's bbox is NOT inside the \
    desk's bbox; it sits on top.
  * `proxy_shape` — OPTIONAL. The object's collision-proxy shape if its \
    silhouette is noticeably non-rectilinear (see PROXY SHAPE section \
    below). CRITICAL for TERRAIN SHELLS in encapsulating mode: a domed \
    island MUST set proxy_shape=HEMISPHERE, otherwise every anchor \
    object placed ON it will float above its AABB top instead of \
    resting on the actual dome. Omit for architectural shells (walls, \
    floors, ceilings, fences) and any object whose bbox is already a \
    good silhouette.
  * `orientation` — world-frame yaw about +Y in degrees. MUST be one of \
    the allowed values: -180, -135, -90, -45, 0, 45, 90, 135, 180. \
    The image-to-3D model receives an ORTHOGRAPHIC FRONT VIEW of the \
    object, so its mesh comes back with the visible front face along \
    world +Z. `orientation` rotates the mesh into the intended world \
    pose. Right-handed about +Y: `0` = front faces +Z (toward viewer); \
    `90` = front faces -X; `180` = front faces -Z (away); `-90` = \
    front faces +X. Examples: a sofa whose seat opens toward the room \
    centre needs orientation set so its front faces the room interior, \
    not the wall. A door in a wall on the +X face of a room needs \
    `-90` so the door faces +X. The bbox stays an AABB — orientation \
    only rotates the mesh inside it, so a long object's bbox dimensions \
    must match its long axis AFTER rotation. Use 0 for symmetric \
    objects with no preferred facing.
  * `relationships` — how this object is anchored in the scene spatially. EVERY object \
    is REQUIRED to include at least one relationship whose `target` is \
    EXACTLY EQUAL to that same object's `parent` field. This is the \
    primary anchor, it is NOT optional, and any object that lacks it is \
    malformed and will be rejected by the validator. Encapsulating \
    elements are no exception: a wall, floor, ceiling, moat, or fence \
    whose `parent` is the zone id must list the zone id as the target of \
    at least one of its relationships. Additional relationships may \
    target sibling objects (i.e. other objects listed in this call).
    

A Relationship has:
  * `target` — the parent (zone or another object in this list) or a \
    sibling object in this list.
  * `kind` — one of: ON, BESIDE, BELOW, ABOVE, ATTACHED.
  * `reference_point` — a corner of the TARGET's bbox under the canonical \
    front view (+X right, +Y up, +Z front). One of: TOP_LEFT_FRONT, \
    TOP_LEFT_BACK, TOP_RIGHT_FRONT, TOP_RIGHT_BACK, BOTTOM_LEFT_FRONT, \
    BOTTOM_LEFT_BACK, BOTTOM_RIGHT_FRONT, BOTTOM_RIGHT_BACK.

Besides the choice of objects themselves and their aesthetic, the SPATIAL \
RELATIONSHIPS and COHERENCE of the scene is the most important part of the benchmark. \
Always reason thoroughly about all the spatial relationships each object has \
with the other objects in the scene, and generate EACH ONE as a distinct \
relationship. This will be used singularly to determine the POSITION of each \
object within the scene.

REMINDER: EVERY object must have a RELATIONSHIP that refers DIRECTLY TO ITS PARENT \
and the spatial anchoring between them as an explicit relationship object, IN ADDITION to the parent field itself. 

The parent graph across listed objects must form a DAG (no cycles). Do \
NOT pick concrete coordinates here — a downstream step resolves each \
object's bbox.
</per_object_fields>

<proxy_shape>
"""
    + PROXY_SHAPE_DOC
    + """
</proxy_shape>

<no_ephemera>
"""
    + NO_EPHEMERA_DOC
    + """
</no_ephemera>

ALWAYS think deeply before responding.

Respond with ONLY ONE JSON object matching the schema. No prose, no markdown, no code fences.\
"""
)


def render_object_decomp(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_bbox: BoundingBox,
    scenario: Literal["anchor", "encapsulating", "negative-space"],
    scene: list[tuple[str, str, BoundingBox, str | None, ProxyShape | None, Orientation]],
    prior_attempts: list[tuple[list[ObjectSpec], str]] | None = None,
) -> str:
    mode = {
        "anchor": "ANCHOR",
        "encapsulating": "ENCAPSULATING",
        "negative-space": "NEGATIVE-SPACE",
    }[scenario]
    scene_lines = (
        "\n".join(
            f"  - {nid}: prompt={prompt!r} bbox={bbox.model_dump_json()} "
            f"proxy_shape={_render_proxy_shape(proxy)} "
            f"orientation={orient}deg parent={pid!r}"
            for nid, prompt, bbox, pid, proxy, orient in scene
        )
        if scene
        else "  (none)"
    )
    if prior_attempts:
        attempt_lines = "\n\n".join(
            f"  attempt {i}:\n"
            f"    emitted: [{', '.join(s.model_dump_json() for s in specs)}]\n"
            f"    rejected: {reason}"
            for i, (specs, reason) in enumerate(prior_attempts)
        )
        retry_block = (
            "\n\nPRIOR ATTEMPTS — every decomposition below was ALREADY "
            "rejected. Do NOT re-emit the same set of object specs, and do "
            "not repeat the same structural mistake. Treat every listed "
            "reason as a hard constraint you must satisfy this time:\n"
            f"{attempt_lines}\n\n"
            "Produce a NEW decomposition that fixes every listed reason. In "
            "particular, ensure every object's `relationships` list contains "
            "at least one item whose `target` is EXACTLY EQUAL to that same "
            "object's `parent` field."
        )
    else:
        retry_block = ""
    return (
        f"MODE: {mode}\n"
        f"ZONE_ID: {zone_id!r}\n"
        f"Zone prompt: {zone_prompt!r}\n"
        f"Zone bbox: {zone_bbox.model_dump_json()}\n\n"
        f"Current scene (every node placed so far across the run):\n{scene_lines}\n\n"
        "List the objects for this zone in the mode above. Each object has "
        "an id, prompt, parent (zone id or another object in this list), "
        "and at least one relationship whose target is its parent. Respect "
        "the CURRENT SCENE: do not duplicate geometry another zone has "
        "already emitted on a shared face, and keep bboxes inside this "
        "zone so they do not volumetrically overlap any peer."
        f"{retry_block}"
    )


# ---------- Step 6: object bbox resolution ----------------------------------


SYSTEM_OBJECT_BBOX_BATCH = (
    """\
You are a constraint solver. Place ALL objects for a scene ZONE in one \
shot, deriving each object's axis-aligned bounding box from the given \
inputs (zone bbox, object specs, relationships, peer prompts/bboxes). \
This step has limited creative latitude: your job is to produce \
coordinates that satisfy the stated constraints and that respect the \
actual geometry implied by peer prompts.

Key semantics — an object's SEMANTIC parent does NOT constrain its \
bbox. A lamp's parent is the desk it sits on, but the lamp's bbox is \
NOT inside the desk's bbox — the lamp sits above the desk, anchored by \
an ON relationship. Let the RELATIONSHIPS drive placement, not the \
parent pointer.

AABB vs. actual geometry — an AABB describes each peer's EXTENT, NOT \
the shape of its surface. Each peer also carries a `proxy_shape` \
(BOX, SPHERE, CAPSULE, or HEMISPHERE) that IS the authoritative \
silhouette inside that AABB; the PROXY SHAPE section below gives the \
exact surface formula Y_top(x, z) for each. A dome-shaped island has \
proxy_shape=HEMISPHERE: its real surface dips from the AABB apex \
down to the AABB's bottom face at the footprint edge, NOT a flat top \
face.

There is no automatic correction. When an object anchors ON a peer \
or sibling with a non-BOX proxy, YOU must compute the target's \
Y_top(x, z) from its AABB and proxy formula, and place the \
anchored object's bbox so its bottom face Y equals that value at the \
anchored object's XZ centre. For BOX-proxy targets (walls, floors, \
ceilings, generic slabs) this collapses to the familiar "bottom face \
at y_max".

Inputs:
  * Zone id, prompt, and bbox — the overall region being populated.
  * OBJECTS to place: each with id, prompt, proxy_shape, semantic \
    parent (the zone id, another object in this batch, or a prior \
    peer id), and a list of relationships.
  * PEERS already placed elsewhere in the scene — each with id, \
    PROMPT, bbox, proxy_shape, and parent_id. The proxy_shape is the \
    authoritative surface descriptor for ON placement; the prompt \
    supplies richer visual context. AABB overlap with peers is \
    expected in practice — objects resting on curved terrain share \
    airspace with their ground mesh, and semantically-contained \
    objects live inside their parent's bbox — so don't contort \
    placements to avoid overlap that the underlying geometry will \
    resolve.

Relationships carry `kind` (ON / BESIDE / BELOW / ABOVE / ATTACHED) \
and a `reference_point` — a corner of the TARGET's bbox under the \
canonical front view (+X right, +Y up, +Z front).

Produce one assignment per object (id + bbox) such that:
  * Every bbox lies fully inside the zone bbox.
  * Every relationship is respected.
  * Dimensions are appropriate to each object's prompt (size a chair \
    like a chair, a wall like a wall, a roof like a roof).
  * Avoid placing two clearly unrelated objects in the same XZ \
    footprint when nothing about the scene justifies it (two trees \
    stacked on the same spot). Some AABB overlap is fine and often \
    unavoidable — curved ground meshes, semantic parents, stacking — \
    so treat non-overlap as a soft preference driven by physical \
    plausibility, not a hard rule.

Because you are deciding the full layout at once, RESERVE SPACE for \
every object up front — if the zone needs walls AND a roof, the walls \
must stop short of the ceiling so the roof has somewhere to sit.

Coordinates in meters, centimeter precision (multiples of 0.01). \
Signed `dimensions` from an `origin` vertex. Emit exactly one \
assignment per requested object id — no extras, no omissions.

<proxy_shape>
"""
    + PROXY_SHAPE_DOC
    + """
</proxy_shape>

Respond with ONE JSON object matching the schema. No prose, no markdown, no code fences.\
"""
)


def render_object_bbox_batch(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_bbox: BoundingBox,
    objects: list[ObjectSpec],
    peers: list[tuple[str, str, BoundingBox, str | None, ProxyShape | None, Orientation]],
) -> str:
    peer_lines = (
        "\n".join(
            f"  - {pid}: prompt={pprompt!r} bbox={pbbox.model_dump_json()} "
            f"proxy_shape={_render_proxy_shape(pproxy)} "
            f"orientation={porient}deg parent={pparent!r}"
            for pid, pprompt, pbbox, pparent, pproxy, porient in peers
        )
        if peers
        else "  (none)"
    )
    object_lines = "\n\n".join(
        f"  - id={o.id!r}\n"
        f"    prompt: {o.prompt}\n"
        f"    parent: {o.parent!r}\n"
        f"    proxy_shape: {_render_proxy_shape(o.proxy_shape)}\n"
        f"    orientation: {o.orientation}deg\n"
        f"    relationships:\n"
        + (
            "\n".join(
                f"      * target={r.target!r} kind={r.kind.value} reference_point={r.reference_point.value}"
                for r in o.relationships
            )
            or "      (none)"
        )
        for o in objects
    )
    return (
        f"Zone id: {zone_id!r}\n"
        f"Zone prompt: {zone_prompt!r}\n"
        f"Zone bbox: {zone_bbox.model_dump_json()}\n\n"
        f"Objects to place ({len(objects)}):\n{object_lines}\n\n"
        f"Peers already placed in the run:\n{peer_lines}\n\n"
        "Produce a bbox for every object in a single coherent layout."
    )


# ---------- Step 7: iterative next-object decision --------------------------


class NextObjectOutput(BaseModel):
    done: bool
    object: ObjectSpec | None = None


SYSTEM_NEXT_OBJECT = (
    """\
You are iteratively refining a 3D scene zone inside StarshotBench — a \
head-to-head competitive benchmark where your scene is rendered and \
judged against another LLM's rendering of the same user prompt. The \
zone's defining anchor objects are already placed. You are being \
asked a single question: is ONE MORE object needed to make this zone \
read as complete, or is it already right?

The threshold between "rich" and "busy" is the judgment call that \
costs runs. Stopping too early leaves the zone sparse and forgettable \
— the judge sees empty dead space and moves on. Adding too much \
turns the zone into incoherent clutter — the judge sees noise and \
can't find the focal point. A masterful build knows when to stop.

Err on the side of `done = true`. Prefer "this zone has what it \
needs" over adding clutter. Only add another object if there is a \
clearly missing element a viewer of the final render would notice \
was absent.

If `done = true`, leave `object` null and stop.

If `done = false`, emit EXACTLY ONE object. Make it COUNT — not \
decorative filler, something that noticeably improves the zone's \
legibility or character. Same rules as the bulk decomposition step:
  * Unique `id` (not colliding with any existing node in the scene).
  * `prompt` — a detailed description; used verbatim for text-to-3D.
  * `parent` — either this zone's id, or the id of ANY already-placed \
    node in the scene (typically an object already placed in THIS \
    zone, like a cup on a previously-placed desk).
  * `orientation` — world-frame yaw about +Y in degrees. MUST be one of: \
    -180, -135, -90, -45, 0, 45, 90, 135, 180. The mesh comes back \
    with its front along world +Z; orientation rotates it into the \
    pose you intend. `0` = front faces +Z (toward viewer), `90` = \
    front faces -X, `180` = front faces -Z (away), `-90` = front \
    faces +X. Pick a non-zero value when the object has a clear \
    "front" that should face a specific direction; use 0 for symmetric \
    objects.
  * `relationships` — REQUIRED to include at least one relationship \
    whose `target` is EXACTLY EQUAL to this emitted object's `parent` \
    field. This is the primary anchor, it is NOT optional, and any \
    object that lacks it is malformed and will be rejected by the \
    validator. Additional relationships may target already-placed \
    objects.

Parent is semantic ("belongs to"), not a spatial containment \
constraint.

GROUND-AWARENESS RULE. If this zone already has a GROUND / SHELL peer \
placed (a mesh describing terrain or enclosure shape — an island \
dome, a crater bowl, a hill, a curved floor, the room's walls and \
floor), any new object whose physical support is that terrain/floor \
MUST set its `parent` to that peer's id (NOT the zone id) and include \
an ON relationship targeting it. The peer's `proxy_shape` in the \
current scene is authoritative for its surface geometry — a \
HEMISPHERE peer is a dome, and the downstream bbox-resolution step \
will compute the dome's surface height at the new object's XZ from \
the peer's proxy formula and rest the object on that surface. You \
are not placing bboxes at this step; just pick the right parent and \
relationships. Only use the zone id as `parent` for objects \
semantically anchored to the zone rather than to a specific surface.

You MAY emit `proxy_shape` on the new object if its silhouette is \
non-rectilinear (SPHERE for a boulder, CAPSULE for a tree trunk, \
HEMISPHERE for a mound) — omit it otherwise. See the emitter's \
decomposition schema for the full vocabulary; the value set is the \
same.

<no_ephemera>
"""
    + NO_EPHEMERA_DOC
    + """
</no_ephemera>

Respond with ONE JSON object matching the schema. No prose, no markdown, no code fences.\
"""
)


class ImagePromptOutput(BaseModel):
    prompt: str


SYSTEM_IMAGE_PROMPT = """\
You produce ONE short noun phrase naming the object to render. The \
phrase is dropped verbatim into a fixed image-prompt template that \
already handles framing, isolation, backdrop, and hitbox shape — your \
output is the SUBJECT of the sentence and nothing more.

The user message will show you the EXACT wrapper your phrase is \
slotted into, with a `<<<SUBJECT>>>` marker where your output goes. \
Read it before writing. Anything the wrapper already says — \
"orthographic front view", the white background, \
"capture the entire model", the explicit object dimensions — must \
NOT appear in your phrase, or it will read twice. You should, however, \
use descriptive language, e.g. "tall" or "wide".

Inputs you receive:
  * The original object prompt.
  * The bounding box dimensions in meters (width +X, height +Y, \
    depth +Z) — use them to pick proportion-sensitive adjectives \
    ("long", "tall", "squat") only when natural. The wrapper also \
    emits the exact dimensions to the renderer, so you do NOT need \
    to encode them in your phrase.
  * The proxy_shape — already encoded into the wrapper's hitbox \
    language; don't repeat it in your phrase.
  * The full image-prompt template with the `<<<SUBJECT>>>` slot \
    visible.
  * The chronological list of prior subject phrases already submitted \
    in this scene — borrow material vocabulary, palette, and stylistic \
    register so the asset coheres with what came before. Do not copy \
    verbatim.

Rules for the phrase:
  * 5-15 words, lower-case, no trailing period.
  * Names the object directly with its defining attributes (species, \
    material, colour, weathering, character). E.g. "a weathered \
    cypress log with bleached bark and patches of moss", "a sleek \
    matte-black office chair with chromed swivel base", "a low \
    domed mossy island with thin muddy lip".
  * NO scene context: drop "half-submerged in", "surrounded by", \
    "resting in", "on the floor of", "nestled among". Keep only \
    intrinsic features of the object itself.
  * NO camera, framing, backdrop, lighting, or rendering instructions \
    — the wrapper handles all of that.

Respond with ONE JSON object matching the schema. The `prompt` field \
holds the noun phrase only.\
"""


_SUBJECT_SLOT = "<<<SUBJECT>>>"


# (3D hitbox term, 2D silhouette term) — slotted into the image-prompt
# wrapper so the prompt's geometric guidance matches the proxy the rest
# of the pipeline uses.
_HITBOX_TERMS: dict[ProxyShape | None, tuple[str, str]] = {
    None: ("rectangular prism", "rectangle"),
    ProxyShape.SPHERE: ("ellipsoid", "ellipse"),
    ProxyShape.CAPSULE: ("vertical capsule", "pill"),
    ProxyShape.HEMISPHERE: ("dome", "dome"),
}


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def wrap_image_prompt(
    description: str,
    proxy_shape: ProxyShape | None,
    dimensions: tuple[float, float, float] | None = None,
) -> str:
    """Slot the LLM's noun phrase into the fixed image-generation
    template. Hitbox + silhouette terms are picked from the proxy.
    If `dimensions` is provided (width, height, depth in metres) it is
    appended as a closing sentence so the renderer sees the object's
    real-world extents."""
    hitbox, silhouette = _HITBOX_TERMS[proxy_shape]
    base = (
        f"Generate a direct, perfect orthographic front view of {description} "
        f"that roughly can be captured within {_article(hitbox)} {hitbox} "
        "hitbox without bending or deforming the object's natural "
        f"proportions. The object should not fully be in {_article(silhouette)} "
        f"{silhouette} shape unless its dimensions and nature dictate it is naturally that shape. Prioritize "
        "realism over confinement to the hitbox shape."
    )
    if dimensions is None:
        return base
    w, h, d = dimensions
    return (
        f"{base} The object's dimensions are exactly "
        f"{w:.2f}m by {h:.2f}m by {d:.2f}m (width by height by depth)."
        "Capture the entire model in the image. Render against a "
        "clean, empty white background with no other objects or graphics."
    )


def render_image_prompt(
    *,
    prompt: str,
    bbox: BoundingBox,
    proxy_shape: ProxyShape | None,
    prior_prompts: list[str],
) -> str:
    w, h, d = bbox.size
    template_preview = wrap_image_prompt(_SUBJECT_SLOT, proxy_shape, (w, h, d))
    if prior_prompts:
        prior_lines = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(prior_prompts))
        prior_block = f"Prior subject phrases in this scene ({len(prior_prompts)} total):\n{prior_lines}"
    else:
        prior_block = (
            "Prior subject phrases in this scene: (none — this is the "
            "first object; you are setting the aesthetic baseline)."
        )
    return (
        f"Original object prompt: {prompt!r}\n"
        f"Bounding box dimensions: width={w:.2f}m, height={h:.2f}m, depth={d:.2f}m\n"
        f"Proxy shape: {_render_proxy_shape(proxy_shape)}\n\n"
        f"Image-prompt template your phrase will be slotted into "
        f"(`{_SUBJECT_SLOT}` is your output):\n  {template_preview}\n\n"
        f"{prior_block}\n\n"
        "Produce ONE short noun phrase naming the subject."
    )


def render_next_object(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_bbox: BoundingBox,
    scene: list[tuple[str, str, BoundingBox, str | None, ProxyShape | None, Orientation]],
    prior_attempts: list[tuple[ObjectSpec, str]] | None = None,
) -> str:
    scene_lines = (
        "\n".join(
            f"  - {nid}: prompt={prompt!r} bbox={bbox.model_dump_json()} "
            f"proxy_shape={_render_proxy_shape(proxy)} "
            f"orientation={orient}deg parent={pid!r}"
            for nid, prompt, bbox, pid, proxy, orient in scene
        )
        if scene
        else "  (none)"
    )
    if prior_attempts:
        attempt_lines = "\n".join(
            f"  attempt {i}: emitted {spec.model_dump_json()}\n             rejected: {reason}"
            for i, (spec, reason) in enumerate(prior_attempts)
        )
        retry_block = (
            "\n\nPRIOR ATTEMPTS — every object spec below was ALREADY "
            "rejected. Do NOT re-emit the same spec, and do not repeat the "
            "same structural mistake. Treat every listed reason as a hard "
            "constraint you must satisfy this time:\n"
            f"{attempt_lines}\n\n"
            "Either emit a NEW ObjectSpec that fixes every listed reason, or "
            "set done=true. If you emit an object, its `relationships` list "
            "MUST contain at least one item whose `target` is EXACTLY EQUAL "
            "to that object's `parent` field."
        )
    else:
        retry_block = ""
    return (
        f"ZONE_ID: {zone_id!r}\n"
        f"Zone prompt: {zone_prompt!r}\n"
        f"Zone bbox: {zone_bbox.model_dump_json()}\n\n"
        f"Current scene (every node placed so far across the run):\n{scene_lines}\n\n"
        "Decide whether another object is needed in this zone. "
        "If yes, emit exactly one ObjectSpec; otherwise set done=true."
        f"{retry_block}"
    )
