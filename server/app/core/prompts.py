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


SYSTEM_ROOT_ZONE_PLAN = """\
<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create \
detailed 3D environments from text prompts. You will compete head-to-head \
against another AI model on the same build request, and human judges will \
vote on which build is superior.

**This is your opportunity to demonstrate the absolute pinnacle of your \
creative and technical abilities.**
</intro>

<judging_criteria>
The judges will compare builds based on:
- Recognizability (can they tell what you built without being told?)
- Creativity (does your build genuinely standout from the others? does it \
propose a narratively driven build with detailed consideration)
- Scene fidelity (is every part clear and well-thought out? Is it plausibly \
built?)
- Overall impression (does it look impressive and masterfully crafted?)

REMEMBER: This is NOT the judging criteria for YOUR PROMPT, it is for the \
FINAL SCENE. The judges only see the final scene after the entire pipeline \
has run through hundreds of downstream generation steps. Your output is NOT \
shown or judged intrinsically; only the final 3D geometry, shaped through all \
downstream AI expansion and generation steps, is judged. Always keep this in \
consideration - make sure that when your output is filtered through, expanded \
by and propagated down many more AI deconstruction calls, it lends well to \
creating a concrete 3D scene from end-to-end (while avoiding being too \
specific or vague, and allowing downstream steps enough agency over what to \
build).
</judging_criteria>

<output>
Respond with a single JSON object containing:
- `plan` (string): Your scene planning paragraph
- `is_atomic` (boolean): Whether this scene is a single cohesive region or \
should decompose into distinct zones

No additional prose, markdown, or code fences.
</output>\
"""


SYSTEM_ZONE_PLAN = """\
<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create \
detailed 3D environments from text prompts. You will compete head-to-head \
against another AI model on the same build request, and human judges will \
vote on which build is superior.

You are planning one region of the scene. The quality of every region \
directly shapes the final scene the judges evaluate.

**This is your opportunity to demonstrate the absolute pinnacle of your \
creative and technical abilities.**
</intro>

<judging_criteria>
The judges will compare builds based on:
- Recognizability (can they tell what you built without being told?)
- Creativity (does your build genuinely standout from the others? does it \
propose a narratively driven build with detailed consideration)
- Scene fidelity (is every part clear and well-thought out? Is it plausibly \
built?)
- Overall impression (does it look impressive and masterfully crafted?)

REMEMBER: This is NOT the judging criteria for YOUR PROMPT, it is for the \
FINAL SCENE. The judges only see the final scene after the entire pipeline \
has run through hundreds of downstream generation steps. Your output is NOT \
shown or judged intrinsically; only the final 3D geometry, shaped through all \
downstream AI expansion and generation steps, is judged. Always keep this in \
consideration - make sure that when your output is filtered through, expanded \
by and propagated down many more AI deconstruction calls, it lends well to \
creating a concrete 3D scene from end-to-end (while avoiding being too \
specific or vague, and allowing downstream steps enough agency over what to \
build).
</judging_criteria>

<output>
Respond with a single JSON object containing:
- `plan` (string): Your region planning paragraph
- `is_atomic` (boolean): Whether this region is a single cohesive area or \
should decompose into distinct zones

No additional prose, markdown, or code fences.
</output>\
"""


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
    # Root zone uses the new competitive prompt format
    if not ancestors:
        return f"""\
write one paragraph that describes the plan for a 3D scene provided the \
following prompt, and decide whether this scene should decompose into \
multiple distinct zones.

"{zone_prompt}"

<VERY IMPORTANT INSTRUCTIONS>
think deeply about the 3D scene, environment or level you want to build from \
this, and how you can creatively make it stand out enough to WIN. 

write directly and consider every part carefully. you are only the first, \
overall planning step - your plan will go through hundreds of further \
downstream steps where it is expanded on and transformed as the AI pipeline \
to construct it propagates further planning by depth. define the scene \
itself, its top-level shape and character enough that the downstream steps \
have agency over their individual sections while also forming ideas of what \
to build. 

only the final output of the 3D geometry for the scene itself will be judged \
once the pipeline is finished; your prompt itself will NEVER be shown to the \
judges, it will only serve as a base to build upon. 

DO NOT be overly specific - remember, your prompt will NOT be converted \
directly into 3D geometry, it will undergo hundreds of expansion and detail \
steps before reaching any generation steps, so structure your output such \
that it is a base that the downstream tree of pipeline steps can build upon \
it. given a prompt for a building, a bad output provides exact instruction on \
what it looks like; a good prompt defines the narrative premise for the \
scene, the scope of the environment, the building's character and type, the \
surrounding environment, points that may implicitly be expanded, and a \
top-level shape for the scene itself without explicitly shaping the entities \
that form it.

plan differently based on the prompt given and infer the purpose - e.g for a \
house, you might plan the aforementioned details for the overall scene scope, \
general architectural, narrative and character; for a super mario platformer \
level, you might focus on the narrative section, features, progression, \
zones, mechanics, etc.; for a top-down swamp frogger level without a specific \
game mentioned, you might focus on first building out the game's premise \
internally given the more abstract request, and then establish the world, \
general layout, character, mechanics, scope, item types, objectives, etc. 

remember, tune specificity based on whether your intent can be inferred by \
downstream steps. e.g in the super mario level, do not be overly specific - \
do not scope out all individual platforms, items, etc. but rather the general \
idea of each part, since the downstream steps have a shared understanding of \
what a super mario level looks like and the general premise of the game; \
however, for a more abstract request like the top-down swamp frogger level, \
the through-line of what you are trying to build cannot be inferred or \
reconstructed by downstream steps in the pipeline as the specific context for \
the premise of the game was constructed within your internal reasoning, not \
exposed to those steps, and thus will be lost as the pipeline propagates, \
placing the onus for world creation and high-level planning on downstream \
steps (e.g the immediate next step of planning the specific nature of zones \
inferred from your prompt, which was not designed for deciding scene \
structure/mechanics itself as it lacks the lack full frame and is more there \
to decompose the scene and figure out spatial relationships between the \
decomposed zones, and follows a different, more mechanical heuristic for \
generation, which means it would not only not spend a lot of time thinking \
about that, but diverge significantly and perhaps genericly from world you \
were trying to create in different directions, before handing off to \
individual planning steps for each zone that have more agency over their \
nature and so on), which means you would need to provide that through-line \
more explicitly in that scenario where foundational planning is required for \
the world state due to a lack of shared context (as opposed to a house or \
established game level).

always think from the perspective of a narrative through-line to help guide \
and form realistic scene intention - what is this game for, who lives in this \
house, what is the player trying to achieve in this level, what kind of city \
is this, etc. 

do not use flowery language. do not describe any abstract quantities like \
mood, lighting, fog, etc unless they can be converted into concrete 3D \
geometry. do not reference meta-quantities like the pipeline itself, the \
scene's 3D nature itself, etc. NEVER MENTION THOSE THINGS. focus on defining \
the environment intrinsically. remember, this is a full 3D scene, NOT an \
image - do not define any specific perspective.

define the scene itself, its top-level shape, character, and rough spatial \
relationships between major parts enough that the downstream steps have \
agency over their individual sections while also forming ideas of what to \
build, especially spatially.
</VERY IMPORTANT INSTRUCTIONS>

<zone_decomposition>
you must also decide `is_atomic` — whether this scene is a single cohesive \
region or should decompose into multiple distinct zones.

the root scene you are planning is a PURELY ABSTRACT META-CONTAINER — it has \
no walls, floor, ceiling, or geometry of its own. only child zones receive \
physical enclosures and geometry.

CRITICAL: if the prompt names a SINGLE TANGIBLE ENCLOSURE that needs \
walls/floor/ceiling (a hotel room, a throne room, a garage, a cockpit, a \
bathroom), you MUST set is_atomic=false. that enclosure becomes a child zone \
inside this abstract root. marking the root atomic in such cases leaves the \
scene with no physical enclosure at all.

default to is_atomic=true. set is_atomic=false ONLY when the scene genuinely \
contains TWO OR MORE distinct regions, each deserving its own dedicated \
planning and generation pass:
- good decomposition: mansion grounds → house, formal garden, stables \
(distinct functional regions)
- good decomposition: hotel room → bedroom, bathroom (distinct rooms)
- bad decomposition: island → north end, central mound, south end (arbitrary \
geography with no distinct identity)
- bad decomposition: bedroom → bed area, dresser area, reading nook \
(over-fragmented; one cohesive space)

a zone is a place large enough to contain multiple objects arranged inside \
it. a single landmark, monument, centerpiece, or hero prop — no matter how \
important — is an OBJECT inside a zone, not a zone of its own.

DO NOT design your prompt around the concept of explicit zonal fragmentation; keep this concept of "zones" in mind ONLY for the is_atomic assessment AFTER the base plan is generated.
</zone_decomposition>

<thinking>
before ANY output, remember to think HARD and DEEPLY and ALWAYS provide a \
detailed CoT. NEVER skip the thinking step. think through different creative \
approaches you might take to what this scene/environment looks like. think \
deeply through the spatial layout to ensure that everything makes sense - \
this is a 3D spatial environment benchmark competition after all. 

in the interest of winning, always start by thinking of the overall narrative \
and premise such that you provide the option for the pipeline to eventually \
build something truly impressive enough to stand out creatively from all the \
other LLMs.
</thinking>
"""

    # Nested zones use adapted competitive prompt format
    ancestor_block = "\n".join(
        f"  - id={aid!r}\n    prompt: {aprompt}\n    plan: {aplan}"
        for aid, aprompt, aplan in ancestors
    )
    if objects:
        obj_block = "\n".join(
            f"  - id={oid!r} parent={oparent!r}: {oprompt}"
            for oid, oprompt, oparent in objects
        )
    else:
        obj_block = "  (none yet)"
    return f"""\
<scene_context>
Ancestor chain (root first → your direct parent, each with their plan):
{ancestor_block}

Generated objects placed so far:
{obj_block}
</scene_context>

write one paragraph that describes the plan for the following region of the \
scene, and decide whether it should decompose into multiple distinct zones.

"{zone_prompt}"

<VERY IMPORTANT INSTRUCTIONS>
think deeply about what this region is and how you can make it creatively \
compelling. every region of the scene contributes to the final build that \
judges evaluate, and the quality of your plan here directly shapes how \
impressive this part of the scene will be.

write directly and consider every part carefully. you are an early planning \
step for this region - your plan will go through further downstream steps \
where it is expanded on and transformed as the pipeline propagates further \
planning by depth. define this region's character, spatial shape, and what \
makes it distinctive enough that downstream steps have agency over the \
specifics while building coherently.

only the final output of the 3D geometry will be judged once the pipeline is \
finished; your prompt itself will NEVER be shown to the judges, it will only \
serve as a base to build upon for this region.

DO NOT be overly specific - your prompt will undergo further expansion and \
detail steps before reaching any generation steps, so structure your output \
as a base that downstream steps can build upon. DO NOT enumerate specific \
objects (a table, a chair, a tree, a lamp) - object selection happens in a \
later generation step that needs its own agency over what to place.

calibrate your plan's specificity to the scope and nature of this region. a \
well-understood region type (a bedroom, a kitchen, a garden) needs less \
foundational planning because downstream steps share an understanding of what \
that space looks like and what belongs in it. a region with novel character \
or a creative premise that cannot be inferred from its prompt and ancestor \
context alone needs more explicit through-line — downstream steps that \
further decompose and populate this region will not reconstruct creative \
intent that isn't present in your plan.

think from the perspective of a narrative through-line — who uses this space, \
what is its purpose, what has happened here. this grounds the region in \
intention rather than leaving it as generic filler.

do not use flowery language. do not describe any abstract quantities like \
mood, lighting, fog, etc unless they can be converted into concrete 3D \
geometry. do not reference meta-quantities like the pipeline itself. focus on \
defining the region intrinsically. this is a 3D environment, not an image - \
do not define any specific perspective.

define this region's shape, character, and rough spatial relationships \
between its major parts enough that downstream steps have agency over their \
individual sections while forming ideas of what to build, especially \
spatially.
</VERY IMPORTANT INSTRUCTIONS>

<zone_decomposition>
you must also decide `is_atomic` — whether this region is a single cohesive \
area or should decompose into multiple distinct zones.

default to is_atomic=true. set is_atomic=false ONLY when the region genuinely \
contains TWO OR MORE distinct areas, each deserving its own dedicated \
planning and generation pass:
- good decomposition: mansion grounds → house, formal garden, stables \
(distinct functional regions)
- good decomposition: hotel room → bedroom, bathroom (distinct rooms)
- bad decomposition: island → north end, central mound, south end (arbitrary \
geography with no distinct identity)
- bad decomposition: bedroom → bed area, dresser area, reading nook \
(over-fragmented; one cohesive space)

a zone is a place large enough to contain multiple objects arranged inside \
it. a single landmark, monument, centerpiece, or hero prop — no matter how \
important — is an OBJECT inside a zone, not a zone of its own.
</zone_decomposition>

<thinking>
before ANY output, think HARD and DEEPLY and provide a detailed CoT. think \
through the creative direction for this region within the context of the \
larger scene. think through spatial layout and how everything fits together \
physically. think about what would make this region genuinely impressive and \
memorable as part of a winning build.
</thinking>\
"""


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


SYSTEM_ZONE_DECOMPOSE = """\
<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create \
detailed 3D environments from text prompts. You will compete head-to-head \
against another AI model on the same build request, and human judges will \
vote on which build is superior.

You are deciding the STRUCTURAL DECOMPOSITION of one region of the scene — \
how it splits into its top-level sub-zones. The shape of this decomposition \
directly constrains every downstream planning, layout, and generation step \
that recurses into your children.

**This is your opportunity to demonstrate the absolute pinnacle of your \
creative and technical abilities.**
</intro>

<judging_criteria>
The judges will compare builds based on:
- Recognizability (can they tell what you built without being told?)
- Creativity (does your build genuinely standout from the others? does it \
propose a narratively driven build with detailed consideration)
- Scene fidelity (is every part clear and well-thought out? Is it plausibly \
built?)
- Overall impression (does it look impressive and masterfully crafted?)

REMEMBER: This is NOT the judging criteria for YOUR PROMPT, it is for the \
FINAL SCENE. The judges only see the final scene after the entire pipeline \
has run through hundreds of downstream generation steps. Your output is NOT \
shown or judged intrinsically; only the final 3D geometry, shaped through all \
downstream AI expansion and generation steps, is judged. Always keep this in \
consideration - make sure that when your output is filtered through, expanded \
by and propagated down many more AI deconstruction calls, it lends well to \
creating a concrete 3D scene from end-to-end (while avoiding being too \
specific or vague, and allowing downstream steps enough agency over what to \
build).
</judging_criteria>

<output>
Respond with a single JSON object containing:
- `children` (list): the sub-zones this region decomposes into. Each child has:
  - `id` (string): unique within the entire scene
  - `prompt` (string): a short seed describing what this child zone is
  - `proxy_shape` (string | null): BOX / SPHERE / CAPSULE / HEMISPHERE if the \
zone's silhouette is non-rectangular, otherwise null/omitted
  - `relationships` (list): at least one entry anchoring this child to the \
parent or to an earlier sibling already listed

No additional prose, markdown, or code fences.
</output>

<input>
  * The zone being decomposed: its id (PARENT_ID), prompt, and axis-aligned \
bounding box (in meters, under the canonical front view: +X right, +Y up, +Z \
front).
  * The ZONE PLAN — the high-level character/intent plan authored upstream \
for this zone. This is your primary signal. Let its named features and \
implied loci drive your decision.
  * The SCENE PROMPT and SCENE PLAN — the root's prompt and plan, the \
north-star for the whole scene.
  * The ANCESTOR CHAIN — every zone above this one in the tree, root first, \
each with its plan.
  * The PRIOR ZONES — every non-root zone already declared in the run, with \
its parent and plan. Lateral context: siblings, cousins, and earlier subtrees \
may inform how THIS zone is structured and anchored.
  * The GENERATED OBJECTS — every concrete (mesh-bearing) object placed \
anywhere in the scene so far, with its parent.
</input>

<additional_context>
A Relationship has:
  * `target` — either PARENT_ID or the `id` of an earlier sibling already \
listed in this call's `children`.
  * `kind` — one of: ON, BESIDE, BELOW, ABOVE, ATTACHED.
  * `reference_point` — which CORNER of the TARGET's bbox this relationship \
anchors against, under the canonical front view (+X right, +Y up, +Z front). \
One of: TOP_LEFT_FRONT, TOP_LEFT_BACK, TOP_RIGHT_FRONT, TOP_RIGHT_BACK, \
BOTTOM_LEFT_FRONT, BOTTOM_LEFT_BACK, BOTTOM_RIGHT_FRONT, BOTTOM_RIGHT_BACK).

DO NOT pick concrete coordinates or dimensions — a downstream batch step \
resolves each child's bbox from its relationships and prompt.

A zone is a REGION OF THE SCENE — a subscene, an area, a place large enough \
to contain multiple distinct objects arranged inside it (e.g a master \
bedroom, the left audience stand of an arena, the formal front garden of a \
mansion, the downtown section of a city). It has room inside it, and its \
character comes from the ensemble of things that live there, not from any \
single object. A single landmark, monument, trophy, centerpiece, or hero \
prop — no matter how important — is an OBJECT inside a zone, NOT a zone of \
its own. Zones often contain other zones within them, and these zones can \
have different structures - e.g the bottom floor of a massive palace might \
contain a war zone and a living zone, where the war zone then further \
decomposes into a set of war rooms, armory rooms and a fighting arena, and \
the living zone decomposes into the great hall, throne room, the front \
entrance, the king and queen's chambers, a garden, etc. Or, the entire \
bottom floor zone might decompose directly into a set of zones for the \
grand entrance, inner courtyard, great hall, trophy gallery, king and \
queen's chambers, throne room and bathrooms (as individual zones), etc.

Zones are designed realistically, based on the given input, intended \
creative direction and amount of world space available to define them within.
</additional_context>\
"""


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
    if ancestors:
        ancestor_block = "\n".join(
            f"  - id={aid!r}\n    prompt: {aprompt}\n    plan: {aplan}"
            for aid, aprompt, aplan in ancestors
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
            f"  - id={oid!r} parent={oparent!r}: {oprompt}"
            for oid, oprompt, oparent in objects
        )
    else:
        obj_block = "  (none — no concrete objects placed yet)"
    return f"""\
Generate a list of subzones that should be present in the following scene, \
based on its description:

{zone_plan}

<IMPORTANT_INSTRUCTIONS>

<ZONE_SPLITTING_GUIDANCE>
Child zones can keep decomposing into more zones recursively in subsequent \
passes, or end there as atomic leaves if that is appropriate. so always \
decompose at the TOP MOST LEVEL of the current zone — e.g. for a house \
scene with backyard, driveway, and house, do not skip straight to \
backyard-pool zone, backyard-grass zone, house-basement, house-first-floor, \
etc.; decompose into "the house", "the backyard", "the driveway" as \
top-level children, and let the next recursion split the house into floors \
and the backyard into pool and grass. the same principle holds everywhere: \
emit only the zones that exist at THIS level of the hierarchy, and trust \
the recursive planning + decompose passes underneath each of them to handle \
the next layer down.
</ZONE_SPLITTING_GUIDANCE>
Think very intricately and spatially about how this zone splits. Your goal \
is to reason a subzone decomposition layout that fits the narrative \
presented by the scene plan given above as well as the additional plans of \
ancestor scenes in the scene context section given below, while paying \
attention to the semantic relationships between the subzones.

The seed prompt you output for each subzone should be a 1-2 sentences long \
description that explains the subzone's shape and character. Be concrete \
about its description while leaving room for this prompt to be a seed for a \
more detailed plan. The prompt should be succinct without mentioning going \
overly into detail on the subzone's contents, but should mention the the \
narrative meaning behind its existence and the narrative meaning behind its \
relative placement to other subzones.

Keep the prompt tight: the goal is not to plan out the subzone's contents, \
but to establish its character as a piece of the larger scene as a whole.
</IMPORTANT_INSTRUCTIONS>

<SCENE_CONTEXT>

Overall scene prompt for the entire world generation: "{scene_prompt}"

Ancestor chain (use this to guide your generation to maintain plans from \
higher levels) (root first → your direct parent, with each ancestor's plan):
{ancestor_block}

Prior zones declared so far (lateral scene context):
{prior_block}

Generated objects placed so far:
{obj_block}

PARENT_ID (the zone being decomposed): {zone_id!r}
Zone prompt: "{zone_prompt}"
Zone bbox (axis-aligned, meters): {zone_bbox.model_dump_json()}

</SCENE_CONTEXT>"""


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
phrase is dropped verbatim into a fixed image-prompt template — used \
to generate THREE orthographic reference images of the object (front, \
side, and top) that all feed into a multi-image-to-3D reconstructor. \
Your phrase is reused across all three views, so it must describe the \
object intrinsically — never reference a specific viewing angle, \
silhouette, or projection. Your output is the SUBJECT of the sentence \
and nothing more.

The user message will show you the EXACT wrappers your phrase is \
slotted into (one per view), with a `<<<SUBJECT>>>` marker where your \
output goes. Read them before writing. Anything the wrappers already \
say — the orthographic view words, the white background, "capture the \
entire model", the explicit object dimensions — must NOT appear in \
your phrase, or it will read twice. You should, however, use \
descriptive language, e.g. "tall" or "wide".

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


ImageView = Literal["front", "side", "top"]


def wrap_image_prompt(
    description: str,
    proxy_shape: ProxyShape | None,
    dimensions: tuple[float, float, float] | None = None,
    *,
    view: ImageView = "front",
) -> str:
    """Slot the LLM's noun phrase into the fixed image-generation
    template. Hitbox + silhouette terms are picked from the proxy. The
    `view` selects which orthographic projection — front, side (looking
    along +X), or top (looking down -Y) — so the same noun phrase can
    be rendered three times for Meshy-6's multi-image-to-3D input.
    If `dimensions` is provided (width, height, depth in metres) it is
    appended as a closing sentence so the renderer sees the object's
    real-world extents."""
    hitbox, silhouette = _HITBOX_TERMS[proxy_shape]
    view_phrase = {
        "front": "orthographic front view",
        "side": "orthographic side profile view (looking along the +X axis at the object's right-hand side)",
        "top": "orthographic top-down view (looking straight down at the object from directly above)",
    }[view]
    reference_clause = (
        ""
        if view == "front"
        else (
            " A reference image showing the orthographic FRONT view of THE SAME "
            "object is provided — preserve its silhouette, proportions, "
            "materials, colour, and surface detail exactly. Only the camera "
            "angle changes; do not reinterpret the object's identity."
        )
    )
    base = (
        f"Generate a direct, perfect {view_phrase} of {description} "
        f"that roughly can be captured within {_article(hitbox)} {hitbox} "
        "hitbox without bending or deforming the object's natural "
        f"proportions. The object should not fully be in {_article(silhouette)} "
        f"{silhouette} shape unless its dimensions and nature dictate it is naturally that shape. Prioritize "
        "realism over confinement to the hitbox shape."
        f"{reference_clause}"
    )
    if dimensions is None:
        return base
    w, h, d = dimensions
    return (
        f"{base} The object's dimensions are exactly "
        f"{w:.2f}m by {h:.2f}m by {d:.2f}m (width by height by depth)."
        "Capture the entire model in the image. Render against a "
        "clean, empty white background with no other objects, dimension markings, or graphics."
    )


def render_image_prompt(
    *,
    prompt: str,
    bbox: BoundingBox,
    proxy_shape: ProxyShape | None,
    prior_prompts: list[str],
) -> str:
    w, h, d = bbox.size
    front_preview = wrap_image_prompt(_SUBJECT_SLOT, proxy_shape, (w, h, d), view="front")
    side_preview = wrap_image_prompt(_SUBJECT_SLOT, proxy_shape, (w, h, d), view="side")
    top_preview = wrap_image_prompt(_SUBJECT_SLOT, proxy_shape, (w, h, d), view="top")
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
        f"Image-prompt templates your phrase will be slotted into "
        f"(`{_SUBJECT_SLOT}` is your output — same phrase, three views):\n"
        f"  FRONT: {front_preview}\n"
        f"  SIDE:  {side_preview}\n"
        f"  TOP:   {top_preview}\n\n"
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
