"""Prompts and structured-output schemas for LLM calls."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.core.types import (
    BoundingBox,
    Orientation,
    ParentRelationshipKind,
    ProxyShape,
    Relationship,
    RelationshipKind,
)


# Shared proxy-shape documentation injected into every prompt that lets
# the LLM emit or reason about proxies. Keep the vocabulary and the math
# identical across decomposition and bbox-resolution steps so there is
# no drift — the bbox-resolution step needs the full formulas to place
# ON-anchored children correctly, so they live here alongside the
# vocabulary.
PROXY_SHAPE_DOC = """A `proxy_shape` describes the silhouette of the node's mesh INSIDE its axis-aligned bbox. The proxy is always inscribed in the AABB; you do NOT emit radii or cap sizes — they derive from the bbox dimensions. Emit `proxy_shape` ONLY when the mesh is noticeably non-boxy; omit it (i.e. null / absent) when the bbox itself is a good collision proxy.

NOTATION used below. For an AABB with min corner (x_min, y_min, z_min) and max corner (x_max, y_max, z_max): center (cx, cy, cz), half-extents (hx, hy, hz) = ((x_max-x_min)/2, (y_max-y_min)/2, (z_max-z_min)/2), full extents (sx, sy, sz) = (2·hx, 2·hy, 2·hz). Every proxy below defines a FOOTPRINT (the XZ region the shape covers) and a TOP-SURFACE FUNCTION Y_top(x, z) that returns the proxy's upper surface height at a given XZ inside that footprint. There is no automatic correction — when you anchor another node ON this one, YOU must compute Y_top at the anchor's XZ and place its bbox so its bottom face sits there.

Valid values:

  * null / omitted — BOX. The AABB is the proxy. Default.
      WHEN TO USE: walls, floors, ceilings, furniture, crates, buildings, signs, rectangular terrain slabs — anything rectilinear.
      FOOTPRINT: the full AABB rectangle, x ∈ [x_min, x_max], z ∈ [z_min, z_max].
      Y_top(x, z) = y_max   (flat top face everywhere in the footprint).

  * SPHERE — ellipsoid inscribed in the AABB, centered at (cx, cy, cz) with semi-axes (hx, hy, hz).
      WHEN TO USE: boulders, planets, balls, orbs, fruits, pumpkins, beach balls.
      FOOTPRINT: the disk ((x-cx)/hx)² + ((z-cz)/hz)² ≤ 1 in the XZ plane through the centre.
      Y_top(x, z) = cy + hy · √(1 − ((x-cx)/hx)² − ((z-cz)/hz)²).
      Apex: (cx, y_max, cz).

  * CAPSULE — Y-axis capsule inscribed in the AABB. Let r = min(hx, hz). Axis is the vertical line through (cx, cz).
      WHEN TO USE: tree trunks, humans, pillars, lamp posts, bottles — anything columnar.
      Top cap centre: (cx, y_max − r, cz). Bottom cap centre: (cx, y_min + r, cz). Cylindrical section of height sy − 2r between the cap centres (degenerates to a sphere when sy ≤ 2r, in which case r = sy/2 instead of min(hx, hz)).
      FOOTPRINT: the disk (x-cx)² + (z-cz)² ≤ r² centered on the axis — note this is generally smaller than the AABB footprint.
      Y_top(x, z) = (y_max − r) + √(r² − (x-cx)² − (z-cz)²).
      Apex: (cx, y_max, cz).

  * HEMISPHERE — upper half of an ellipsoid with its equatorial disk resting on the AABB's bottom face (y = y_min) and its apex at (cx, y_max, cz). Semi-axes are (hx, sy, hz) — the VERTICAL half-extent is the FULL AABB height sy, NOT hy, because the equator sits at y_min, not at cy.
      WHEN TO USE: DOMED TERRAIN — low islands rising from the waterline, grassy mounds, half-buried boulders, snow hills, cathedral domes.
      FOOTPRINT: the disk ((x-cx)/hx)² + ((z-cz)/hz)² ≤ 1 at y = y_min.
      Y_top(x, z) = y_min + sy · √(1 − ((x-cx)/hx)² − ((z-cz)/hz)²).
      Drops from the apex y_max at the centre to y_min at the footprint boundary.

ON-RELATIONSHIP CONSEQUENCE. When you anchor a node ON a target with a non-BOX proxy, the AABB's top face is NOT the resting surface. Compute the target's Y_top at the anchored node's XZ centre using the target's AABB and the formula above, then set the anchored node's bbox so its bottom face Y equals Y_top. Example: a 0.8m tree placed ON a HEMISPHERE island whose AABB is (x_min=-5, y_min=0, z_min=-5) → (x_max=5, y_max=1.2, z_max=5), at XZ = (3, 0), rests at Y_top = 0 + 1.2·√(1 − 0.36) = 0.96, so its bbox spans y ∈ [0.96, 1.76] — NOT [1.20, 2.00]. Getting this wrong leaves the tree visibly floating or sunk. For BOX targets the rule collapses to the familiar "bottom face at y_max"."""


def _render_proxy_shape(p: ProxyShape | None) -> str:
    return p.value if p is not None else "BOX"


# Shared anti-ephemera guidance injected into every prompt that authors
# scene content (zone plans, zone decomposition, object decomposition,
# next-object polish). The downstream text-to-3D model produces solid
# meshes; gaseous, volumetric, or luminous phenomena render as garbage
# blobs that drag the whole scene down. Centralised here so the
# vocabulary of forbidden phenomena stays consistent across steps.
NO_EPHEMERA_DOC = """NO EPHEMERA. The downstream renderer produces SOLID, OPAQUE, BOUNDED 3D meshes — it cannot represent gases, plasmas, particulate clouds, volumetric light, or any phenomenon that lacks a hard surface. Naming such phenomena as features, anchors, plan elements, or ambient fill produces deformed mesh blobs that visibly tank the scene. Do NOT introduce, plan, enumerate, or describe any of the following as things the scene must depict:

  * GASES & VAPOURS — fog, mist, haze, smoke, steam, vapour, smog, exhaust plumes, dust clouds, pollen clouds, sandstorms, snow flurries in the air, falling rain or snow as discrete particles.
  * CLOUDS & SKY VOLUMES — clouds, nebulae, gas giants' atmospheres, aurora curtains, rainbows, sunbeams / god rays, light shafts.
  * PLASMAS & ENERGY — lightning bolts, electrical arcs, plasma discharges, fire flames as freestanding objects, sparks, embers in flight, magical glows, force fields, beams of light, laser beams, comet tails, meteor trails, contrails.
  * LIQUIDS IN MOTION — splashes, sprays, waterfalls as freestanding objects, fountains' water arcs, pouring streams, ripples.

You MAY still IMPLY these phenomena through tangible, solid consequences that DO have hard surfaces: wet flagstones instead of rain, scorched and split bark instead of lightning, soot stains and charred timbers instead of smoke, a frost crust instead of fog, puddles and damp moss instead of drizzle, a fire pit with glowing embers (a solid bowl of coals) instead of freestanding flames, a chimney instead of a smoke plume. Atmosphere is conveyed by what the weather has DONE to solid surfaces, not by depicting the weather itself. A flat-water surface (a pond, a puddle, a lake skin) IS allowed because it is a bounded plane; freestanding water in motion is not."""


# ---------- Step 1: zone plan (high-level authoring; runs for every zone) ---


class ZonePlanOutput(BaseModel):
    plan: str
    is_atomic: bool


SYSTEM_ROOT_ZONE_PLAN = """<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.

**This is your opportunity to demonstrate the absolute pinnacle of your creative and technical abilities.**
</intro>

<judging_criteria>
The judges will compare builds based on:
- Recognizability (can they tell what you built without being told?)
- Creativity (does your build genuinely standout from the others? does it propose a narratively driven build with detailed consideration)
- Scene fidelity (is every part clear and well-thought out? Is it plausibly built?)
- Overall impression (does it look impressive and masterfully crafted?)

REMEMBER: This is NOT the judging criteria for YOUR PROMPT, it is for the FINAL SCENE. The judges only see the final scene after the entire pipeline has run through hundreds of downstream generation steps. Your output is NOT shown or judged intrinsically; only the final 3D geometry, shaped through all downstream AI expansion and generation steps, is judged. Always keep this in consideration - make sure that when your output is filtered through, expanded by and propagated down many more AI deconstruction calls, it lends well to creating a concrete 3D scene from end-to-end (while avoiding being too specific or vague, and allowing downstream steps enough agency over what to build).
</judging_criteria>

<output>
Respond with a single JSON object containing:
- `plan` (string): Your scene planning paragraph
- `is_atomic` (boolean): Whether this scene is a single cohesive region or should decompose into distinct zones

No additional prose, markdown, or code fences.
</output>"""


SYSTEM_ZONE_PLAN = """<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.

You are planning one region of the scene. The quality of every region directly shapes the final scene the judges evaluate.

**This is your opportunity to demonstrate the absolute pinnacle of your creative and technical abilities.**
</intro>

<judging_criteria>
The judges will compare builds based on:
- Recognizability (can they tell what you built without being told?)
- Creativity (does your build genuinely standout from the others? does it propose a narratively driven build with detailed consideration)
- Scene fidelity (is every part clear and well-thought out? Is it plausibly built?)
- Overall impression (does it look impressive and masterfully crafted?)

REMEMBER: This is NOT the judging criteria for YOUR PROMPT, it is for the FINAL SCENE. The judges only see the final scene after the entire pipeline has run through hundreds of downstream generation steps. Your output is NOT shown or judged intrinsically; only the final 3D geometry, shaped through all downstream AI expansion and generation steps, is judged. Always keep this in consideration - make sure that when your output is filtered through, expanded by and propagated down many more AI deconstruction calls, it lends well to creating a concrete 3D scene from end-to-end (while avoiding being too specific or vague, and allowing downstream steps enough agency over what to build).
</judging_criteria>

<output>
Respond with a single JSON object containing:
- `plan` (string): Your region planning paragraph
- `is_atomic` (boolean): Whether this region is a single cohesive area or should decompose into distinct zones

No additional prose, markdown, or code fences.
</output>"""


def render_zone_plan(
    *,
    zone_id: str,
    zone_prompt: str,
    ancestors: list[tuple[str, str, str, BoundingBox, str | None]],
    objects: list[tuple[str, str, str | None, BoundingBox, str | None, str | None]],
) -> str:
    """ancestors: (id, prompt, plan, bbox, placement) tuples from root → parent of this
    zone, excluding the zone itself. Empty for the root.
    objects: (id, prompt, parent_id, bbox, placement, parent_kind) tuples
    for every concrete (mesh-bearing) node placed anywhere in the run so
    far. `parent_kind` is unused here; consumed only by
    `_scene_context_zone_decompose_narrative` to split frames from
    interior anchor objects."""
    # Root zone uses the new competitive prompt format
    if not ancestors:
        return f"""write one paragraph that describes the plan for a 3D scene provided the following prompt, and decide whether this scene should decompose into multiple distinct zones.

"{zone_prompt}"

<VERY IMPORTANT INSTRUCTIONS>
think deeply about the 3D scene, environment or level you want to build from this, and how you can creatively make it stand out enough to WIN. 

write directly and consider every part carefully. you are only the first, overall planning step - your plan will go through hundreds of further downstream steps where it is expanded on and transformed as the AI pipeline to construct it propagates further planning by depth. define the scene itself, its top-level shape and character enough that the downstream steps have agency over their individual sections while also forming ideas of what to build. 

only the final output of the 3D geometry for the scene itself will be judged once the pipeline is finished; your prompt itself will NEVER be shown to the judges, it will only serve as a base to build upon. 

DO NOT be overly specific - remember, your prompt will NOT be converted directly into 3D geometry, it will undergo hundreds of expansion and detail steps before reaching any generation steps, so structure your output such that it is a base that the downstream tree of pipeline steps can build upon it. given a prompt for a building, a bad output provides exact instruction on what it looks like; a good prompt defines the narrative premise for the scene, the scope of the environment, the building's character and type, the surrounding environment, points that may implicitly be expanded, and a top-level shape for the scene itself without explicitly shaping the entities that form it.

plan differently based on the prompt given and infer the purpose - e.g for a house, you might plan the aforementioned details for the overall scene scope, general architectural, narrative and character; for a super mario platformer level, you might focus on the narrative section, features, progression, zones, mechanics, etc.; for a top-down swamp frogger level without a specific game mentioned, you might focus on first building out the game's premise internally given the more abstract request, and then establish the world, general layout, character, mechanics, scope, item types, objectives, etc. 

remember, tune specificity based on whether your intent can be inferred by downstream steps. e.g in the super mario level, do not be overly specific - do not scope out all individual platforms, items, etc. but rather the general idea of each part, since the downstream steps have a shared understanding of what a super mario level looks like and the general premise of the game; however, for a more abstract request like the top-down swamp frogger level, the through-line of what you are trying to build cannot be inferred or reconstructed by downstream steps in the pipeline as the specific context for the premise of the game was constructed within your internal reasoning, not exposed to those steps, and thus will be lost as the pipeline propagates, placing the onus for world creation and high-level planning on downstream steps (e.g the immediate next step of planning the specific nature of zones inferred from your prompt, which was not designed for deciding scene structure/mechanics itself as it lacks the lack full frame and is more there to decompose the scene and figure out spatial relationships between the decomposed zones, and follows a different, more mechanical heuristic for generation, which means it would not only not spend a lot of time thinking about that, but diverge significantly and perhaps genericly from world you were trying to create in different directions, before handing off to individual planning steps for each zone that have more agency over their nature and so on), which means you would need to provide that through-line more explicitly in that scenario where foundational planning is required for the world state due to a lack of shared context (as opposed to a house or established game level).

always think from the perspective of a narrative through-line to help guide and form realistic scene intention - what is this game for, who lives in this house, what is the player trying to achieve in this level, what kind of city is this, etc. 

do not use flowery language. do not describe any abstract quantities like mood, lighting, fog, etc unless they can be converted into concrete 3D geometry. do not reference meta-quantities like the pipeline itself, the scene's 3D nature itself, etc. NEVER MENTION THOSE THINGS. focus on defining the environment intrinsically. remember, this is a full 3D scene, NOT an image - do not define any specific perspective.

define the scene itself, its top-level shape, character, and rough spatial relationships between major parts enough that the downstream steps have agency over their individual sections while also forming ideas of what to build, especially spatially.
</VERY IMPORTANT INSTRUCTIONS>

<zone_decomposition>
you must also decide `is_atomic` — whether this scene is a single cohesive region or should decompose into multiple distinct zones.

the root scene you are planning is a PURELY ABSTRACT META-CONTAINER — it has no walls, floor, ceiling, or geometry of its own. only child zones receive physical enclosures and geometry.

CRITICAL: if the prompt names a SINGLE TANGIBLE ENCLOSURE that needs walls/floor/ceiling (a hotel room, a throne room, a garage, a cockpit, a bathroom), you MUST set is_atomic=false. that enclosure becomes a child zone inside this abstract root. marking the root atomic in such cases leaves the scene with no physical enclosure at all.

default to is_atomic=true. set is_atomic=false ONLY when the scene genuinely contains TWO OR MORE distinct regions, each deserving its own dedicated planning and generation pass:
- good decomposition: mansion grounds → house, formal garden, stables (distinct functional regions)
- good decomposition: hotel room → bedroom, bathroom (distinct rooms)
- bad decomposition: island → north end, central mound, south end (arbitrary geography with no distinct identity)
- bad decomposition: bedroom → bed area, dresser area, reading nook (over-fragmented; one cohesive space)

a zone is a place large enough to contain multiple objects arranged inside it. a single landmark, monument, centerpiece, or hero prop — no matter how important — is an OBJECT inside a zone, not a zone of its own.

DO NOT design your prompt around the concept of explicit zonal fragmentation; keep this concept of "zones" in mind ONLY for the is_atomic assessment AFTER the base plan is generated.
</zone_decomposition>

<thinking>
before ANY output, remember to think HARD and DEEPLY and ALWAYS provide a detailed CoT. NEVER skip the thinking step. think through different creative approaches you might take to what this scene/environment looks like. think deeply through the spatial layout to ensure that everything makes sense - this is a 3D spatial environment benchmark competition after all. 

in the interest of winning, always start by thinking of the overall narrative and premise such that you provide the option for the pipeline to eventually build something truly impressive enough to stand out creatively from all the other LLMs.
</thinking>
"""

    # Nested zones use adapted competitive prompt format
    ancestor_block = _render_ancestor_block(ancestors)
    obj_entries = [
        _render_node_entry(
            nid=oid,
            parent=oparent,
            prompt=oprompt,
            bbox=obbox,
            placement=oplacement,
        )
        for oid, oprompt, oparent, obbox, oplacement, _okind in objects
    ]
    obj_block = _render_section("OBJECTS", obj_entries, "none yet")
    return f"""write one paragraph that describes the plan for the following region, and decide whether it should decompose into multiple distinct subzones.

"{zone_prompt}"

This region is ONE PART of the larger scene. The following is the ancestor chain for the current region:

{ancestor_block}

Your goal is to elaborate and add to the narrative painted by the ancestor plans through the plan for this region, but also leave sufficient room in your plan for further downstream steps to expand on more using their own agency. what constitutes "sufficient" depends on the specificity of the current region: larger, higher-level regions should have less specificity, while smaller, more constrained regions nearing the atomic level should have more specificity.

<VERY IMPORTANT INSTRUCTIONS>
think deeply about what this region is and how you can make it creatively compelling. every region of the scene contributes to the final build that judges evaluate, and the quality of your plan here directly shapes how impressive this part of the scene will be.

write directly and consider every part carefully. you are the planning step for this region - your plan will go through further downstream steps where it is expanded on and transformed as the pipeline propagates further planning by depth. define this region's character, spatial shape, and what makes it distinctive enough that downstream steps have agency over the specifics while building coherently.

only the final output of the 3D geometry will be judged once the pipeline is finished; your prompt itself will NEVER be shown to the judges, it will only serve as a base to build upon for this region. thus, making the prompt dramatic and sound impressive will only have a contradictory effect, since it will confuse downstream steps when generation actually happens as they don't understand flowery language.

DO NOT be overly specific - your prompt will undergo further subzone divisions, expansion, and detail steps before reaching any generation steps, so structure your output as a base that downstream steps can build upon. DO NOT enumerate specific objects (a table, a chair, a tree, a lamp) - object selection happens in a later generation step that needs its own agency over what to place.

calibrate your plan's specificity to the scope and nature of this region. a well-understood region type (a bedroom, a kitchen, a garden) needs less foundational planning because downstream steps share an understanding of what that space looks like and what belongs in it. a region with novel character or a creative premise that cannot be inferred from its prompt and ancestor context alone needs more explicit through-line — downstream steps that further decompose and populate this region will not reconstruct creative intent that isn't present in your plan. furthermore, a tightly-constrained region that cannot be broken down into further subregions would require more specificity in terms of object enumeration as you are the final planning step before the actual object list gets generated by a downstream step.

your prompt should focus on just the current region: it can reference other defined regions and objects as context, but do not overtly describe them apart from using them as an anchor for relative positioning.

think from the perspective of a narrative through-line. ground the region in concrete use: what physically happens in this space, not in atmosphere or symbolism. you can and should use the plans of ancestor regions mentioned above to help you in coming up with this as they provide additional, already-established context of the scene.

In the plan you write, do not use flowery language. do not describe any abstract quantities like mood, lighting, fog, etc unless they can be converted into concrete 3D geometry. do not reference meta-quantities like the pipeline itself or your role. focus on defining the region intrinsically. this is a 3D environment, not an image - do not define any specific perspective. do not mention meta pipeline-related terms, such as "already-placed" or "existing". the prompt should be direct, accurately describe the scene, and every word should be useful for downstream generation and further processing, grounded concretely in what the scene is and with no relation to the pipeline itself.

define this region's shape, character, and rough spatial relationships between its major parts enough that downstream steps have agency over their individual sections while forming ideas of what to build, especially spatially.
</VERY IMPORTANT INSTRUCTIONS>

<zone_decomposition>
you must also decide is_atomic — whether this region is a single cohesive area or should decompose into multiple distinct zones.

default to is_atomic=true. set is_atomic=false ONLY when the region genuinely contains TWO OR MORE distinct areas, each deserving its own dedicated planning and generation pass:

    good decomposition: mansion grounds → house, formal garden, stables (distinct functional regions)
    good decomposition: hotel room → bedroom, bathroom (distinct rooms)
    bad decomposition: island → north end, central mound, south end (arbitrary geography with no distinct identity)
    bad decomposition: bedroom → bed area, dresser area, reading nook (over-fragmented; one cohesive space)

a zone is a place large enough to contain multiple objects arranged inside it. a single landmark, monument, centerpiece, or hero prop — no matter how important — is an OBJECT inside a zone, not a zone of its own.
</zone_decomposition>

<thinking>
before ANY output, think HARD and DEEPLY and provide a detailed CoT. think through the creative direction for this region within the context of the larger scene. think through spatial layout and how everything fits together physically. think about the constructed narrative of the ancestor plans and how this particular region could add onto it as a detail. think about what would make this region genuinely impressive and memorable as part of a winning build.

The following objects are already fixed in the world. Refer to them by what they are (not ids) when you need a positional anchor, but do not redescribe them.

<scene_context>
{obj_block}
</scene_context>
</thinking>"""


# ---------- Step 2: overall bbox --------------------------------------------


class OverallBboxOutput(BaseModel):
    bbox: BoundingBox


SYSTEM_OVERALL_BBOX = """You are picking the OVERALL bounding box for a 3D scene — the SCENE'S CANVAS that every zone, object, and ambient element will be placed inside. This box is a PURELY ABSTRACT, INTANGIBLE META-CONTAINER for the world: it has no walls, no floor, no ceiling, no skin, and never becomes a tangible frame or mesh. It only sets the outer extents of the world the scene lives in. If the scene is a single tangible enclosure (a hotel room, a throne room, a cockpit), the canvas should be SLIGHTLY LARGER than that enclosure, so the actual room fits comfortably as a child zone inside it with a small margin of empty world around it — the canvas is NOT the room. This pipeline is part of StarshotBench, a head-to-head LLM benchmark. The SCENE PLAN has already been authored upstream and is shown to you in the inputs; your job is to size the canvas so it matches the silhouette that plan implies. Get this wrong and every downstream step is fighting the canvas — a skyscraper crammed into a cube, a river squeezed into a square, a room ballooned into a warehouse. Match the scene's actual silhouette: a skyscraper is tall and narrow, a river is long and flat, a room is modest in every dimension.

The bounding box is axis-aligned, in meters, interpreted under the CANONICAL FRONT VIEW: +X = right, +Y = up, +Z = toward the viewer (front), -Z = back. It is defined by an `origin` vertex and a signed `dimensions` vector `(dx, dy, dz)` extending from that vertex; the sign of each component chooses the direction of expansion along that axis.

Emit all coordinates to centimeter precision — two decimal places, exact multiples of 0.01 m. Place the origin sensibly (often the world origin; floor at y=0 for architectural scenes) and choose signs so the box extends into the region you intend.

Respond with ONE JSON object matching the schema. No prose, no markdown, no code fences."""


def render_overall_bbox(user_prompt: str, scene_plan: str) -> str:
    return f"""User prompt for the scene: {user_prompt!r}

SCENE PLAN (authored upstream — size the canvas to match its implied silhouette):
{scene_plan}

Produce the overall bounding box for the whole scene."""


# ---------- Step 3: zone decompose (atomic vs subzones; runs after plan) ----


class ChildNodeSpec(BaseModel):
    """A single child node (subzone or object) emitted by a decomposition
    LLM call.

    Fields:
      * `parent` — the structural anchor id: the supporter (object or
        frame) this node physically rests on, or the containing zone
        when there's no physical supporter (floating objects, frames
        themselves, subzones inside a parent zone).
      * `parent_kind` — how this node relates to its `parent`. MUST be
        one of ON / ATTACHED / IN:
          - ON: rests on parent's outward surface (most often top face).
          - ATTACHED: flush against any face of the parent (wall mount,
            ceiling fixture, magnet, hanger).
          - IN: contained inside the parent's volume or footprint
            (subzone inside a zone, fish inside a tank, cloud inside a
            sky zone, embedded particle).
        BESIDE / ABOVE / BELOW are reserved for sibling/peer hints in
        `referenced_ids` and are NOT valid here.
      * `placement` — prose describing where this node sits, with
        precise positioning (centered, edge-aligned, two-thirds along,
        etc.). The bbox-resolution step uses this verbatim to choose
        coordinates.
      * `referenced_ids` — optional list of *secondary* relationships
        to other already-placed peers. Each entry is a Relationship
        with `target` (the peer's id) and `kind` (ON, BESIDE, ABOVE,
        BELOW, ATTACHED, IN) — categorical, no anchor point. Do NOT
        repeat the parent here. Empty list is fine when the placement
        only references the parent.
    """

    id: str
    prompt: str
    parent: str
    parent_kind: ParentRelationshipKind
    placement: str
    referenced_ids: list[Relationship] = Field(default_factory=list)
    proxy_shape: ProxyShape | None = None
    orientation: Orientation = 0

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


SYSTEM_ZONE_DECOMPOSE = """<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.

You are deciding the STRUCTURAL DECOMPOSITION of one region of the scene — how it splits into its top-level sub-zones. The shape of this decomposition directly constrains every downstream planning, layout, and generation step that recurses into your children.

**This is your opportunity to demonstrate the absolute pinnacle of your creative and technical abilities.**
</intro>

<judging_criteria>
The judges will compare builds based on:
- Recognizability (can they tell what you built without being told?)
- Creativity (does your build genuinely standout from the others? does it propose a narratively driven build with detailed consideration)
- Scene fidelity (is every part clear and well-thought out? Is it plausibly built?)
- Overall impression (does it look impressive and masterfully crafted?)

REMEMBER: This is NOT the judging criteria for YOUR PROMPT, it is for the FINAL SCENE. The judges only see the final scene after the entire pipeline has run through hundreds of downstream generation steps. Your output is NOT shown or judged intrinsically; only the final 3D geometry, shaped through all downstream AI expansion and generation steps, is judged. Always keep this in consideration - make sure that when your output is filtered through, expanded by and propagated down many more AI deconstruction calls, it lends well to creating a concrete 3D scene from end-to-end (while avoiding being too specific or vague, and allowing downstream steps enough agency over what to build).
</judging_criteria>

<output>
Respond with a single JSON object containing:
- `children` (list): the sub-zones this region decomposes into. Each child has:
  - `id` (string): unique within the entire scene
  - `prompt` (string): a short seed describing what this child zone is
  - `parent` (string): the id of this child's STRUCTURAL PARENT — the containing zone (PARENT_ID for a top-level subzone) or an earlier sibling in this call whose interior/footprint this child sits within. Required.
  - `parent_kind` (string): how this child anchors to its `parent`. MUST be exactly one of `ON`, `ATTACHED`, or `IN`. Top-level subzones contained inside their parent zone use `IN`. A subzone that physically rests on the top surface of a sibling-parent (e.g. a rooftop garden on a tower zone) uses `ON`. A wall- or boundary-attached subzone flush against a parent face uses `ATTACHED`. `BESIDE` / `ABOVE` / `BELOW` are NOT valid here — those are peer hints, reserved for `referenced_ids`.
  - `placement` (string): a prose description of WHERE this child sits within / against / relative to its parent and any referenced peers (see <additional_context> below). The bbox-resolution step uses this verbatim for precise positioning.
  - `referenced_ids` (list of {target, kind}): OPTIONAL secondary relationships to other already-placed nodes referenced in your placement text. Each entry has a `target` (the peer's id) and a `kind` — one of ON, BESIDE, ABOVE, BELOW, ATTACHED, IN — categorizing the relationship. Do NOT repeat the parent here. Leave empty when the placement only references the parent.
  - `proxy_shape` (string | null): BOX / SPHERE / CAPSULE / HEMISPHERE if the zone's silhouette is non-rectangular, otherwise null/omitted

No additional prose, markdown, or code fences.
</output>

<input>
The user message contains the following labelled sections:

  * PARENT_ID, Zone prompt, Zone bbox — the zone being decomposed: its id, seed prompt, and axis-aligned bounding box (in meters, under the canonical front view: +X right, +Y up, +Z front).
  * <SCENE_CONTEXT> — a NARRATIVE description of the scene as it stands. It first drills top-down through the ancestor spine (root → … → the target zone) in flowing prose: each zone gets a plan paragraph, optionally extended with a sentence of the form "The <id> is framed by <frame_id>, <frame_id>, and it contains <anchor_id> (<placement>), <anchor_id> (<placement>)." that names every concrete (mesh-bearing or pending-mesh) node attached inside it — shell frames are listed by id only, interior anchor objects also carry their placement prose in parens; this is followed by a dims sentence "The <id> is W by H by D and origins at (x, y, z). Inside, it contains <next> with the following plan:" that transitions to the next zone on the path. The target zone closes the spine with "You are to decide on the structural decomposition of this <target>." Sibling subtrees off the path are then enumerated under "Other areas in <parent>:" headers (the header for the target's direct parent is suffixed with "<- the <target> you are to decompose exists here"). Sub-area nesting inside those lists uses `> ` as a depth delimiter — every line is prefixed with one `> ` per level of nesting below its enclosing "Other areas in X" or "Areas in X" header — INSTEAD of visual indentation. Any zone shown as `(plan not yet authored …)` has been declared and placed but not individually planned yet; its bbox + placement are still load-bearing, and its named concrete children are valid `referenced_ids` targets.
</input>

<additional_context>
Each child you emit carries four coupled fields — `parent`, `parent_kind`, `placement`, and `referenced_ids` — that together specify where it sits.

`parent` is the id of the STRUCTURAL ANCHOR: the containing zone (for a top-level subzone inside PARENT_ID) or an earlier sibling in this call whose interior/footprint this child sits within. Required. The chain of parent edges across the whole run must eventually reach a real zone — the validator enforces this.

`parent_kind` MUST be one of `ON`, `ATTACHED`, or `IN` — the only relationships that describe physical anchoring or containment. Use:
  * `IN`  — child is contained inside the parent's volume / footprint. This is the default for top-level subzones inside the parent zone, for sub-volumes carved out of a sibling-parent's interior, and for any contained occupant.
  * `ON`  — child rests on the parent's outward surface (most often the top face). Use this when a sub-zone or feature sits on top of its sibling-parent (e.g. a rooftop garden on a tower zone, an island on a lake zone).
  * `ATTACHED` — child is flush against any face of the parent. Use this for boundary-attached sub-zones (e.g. a balcony fused to one face of a building zone, a porch attached to the front of a house).
`BESIDE`, `ABOVE`, and `BELOW` are NOT valid `parent_kind` values — those describe non-contact peer arrangements and only appear in `referenced_ids`.

`placement` is prose describing WHERE in the parent this child sits — position, alignment, edge/center, anything a layout solver needs to pick coordinates. Be specific about WHERE in the parent, not just THAT it is in the parent. Examples of good placements:
  * "centered on the back wall of the parent, hugging the floor"
  * "in the front-left quadrant of the parent, leaving the back 60% for the kitchen zone"
  * "spanning the full width of the parent at the lowest level, with the pool taking the outermost 6m strip toward the cliffside (+Z) and the pavilion stepped back from the pool"

`referenced_ids` is an OPTIONAL list of secondary spatial relationships when your placement text refers to nodes other than the parent. Each entry is `{target: <id>, kind: <ON|BESIDE|ABOVE|BELOW|ATTACHED|IN>}`. Pick the kind that best summarizes the abstract category of the relationship — precise positioning is in the placement prose, not in the kind. Use `IN` when the target physically contains this child (e.g. a sub-zone enclosed inside a peer dome, an alcove carved into a peer wall). Examples:
  * Subzone occupying its parent zone's interior, beside a sibling courtyard:
    parent='ground_floor', parent_kind=IN,
    referenced_ids=[{target='courtyard', kind=BESIDE}]
  * A roof slab resting on the supporting walls below it:
    parent='zone', parent_kind=IN,
    referenced_ids=[{target='north_wall', kind=ON}, {target='south_wall', kind=ON}]
  * A balcony fused to the front face of a house zone:
    parent='house_zone', parent_kind=ATTACHED,
    referenced_ids=[]

DO NOT include the parent in `referenced_ids`. DO NOT add entries that your placement text doesn't actually mention. An empty list is fine when the placement only relates to the parent.

ANY id that appears in the prompt context is a valid `parent` or `target`: the <ANCESTOR_CHAIN>, the <ZONES> block (siblings, cousins, earlier subtrees — including ones whose plan isn't authored yet), the <OBJECTS> block (frames and prior objects), and the earlier-listed siblings in THIS call's `children`. Each lateral entry shows its bbox and its own placement text — use them as concrete anchors.

The canonical front view applies to all spatial language: +X right, +Y up, +Z front, -Z back. Right-handed, Y-up, meters. When you say "front" mean +Z; when you say "left" mean -X; when you say "top" mean +Y.

DO NOT pick concrete coordinates or dimensions — a downstream batch step resolves each child's bbox from its parent, placement, referenced_ids, and the parent's bbox.

A zone is a REGION OF THE SCENE — a subscene, an area, a place large enough to contain multiple distinct objects arranged inside it (e.g a master bedroom, the left audience stand of an arena, the formal front garden of a mansion, the downtown section of a city). It has room inside it, and its character comes from the ensemble of things that live there, not from any single object. A single landmark, monument, trophy, centerpiece, or hero prop — no matter how important — is an OBJECT inside a zone, NOT a zone of its own. Zones often contain other zones within them, and these zones can have different structures - e.g the bottom floor of a massive palace might contain a war zone and a living zone, where the war zone then further decomposes into a set of war rooms, armory rooms and a fighting arena, and the living zone decomposes into the great hall, throne room, the front entrance, the king and queen's chambers, a garden, etc. Or, the entire bottom floor zone might decompose directly into a set of zones for the grand entrance, inner courtyard, great hall, trophy gallery, king and queen's chambers, throne room and bathrooms (as individual zones), etc.

Zones are designed realistically, based on the given input, intended creative direction and amount of world space available to define them within.
</additional_context>"""


def _scene_context_zone_decompose_narrative(
    *,
    target_zone_id: str,
    target_zone_prompt: str,
    target_zone_bbox: BoundingBox,
    target_zone_plan: str,
    scene_prompt: str,
    scene_plan: str,
    ancestors: list[tuple[str, str, str, BoundingBox, str | None]],
    prior_zones: list[tuple[str, str, str | None, str, BoundingBox, str | None]],
    objects: list[tuple[str, str, str | None, BoundingBox, str | None, str | None]],
) -> str:
    """TEMPORARY helper — render scene context for ZONE_DECOMPOSE as a
    flowing narrative.

    The output drills top-down from the root through the ancestor spine
    into the target zone in prose: each zone gets a plan paragraph
    (plan + "is framed by …" sentence) followed by a dims-and-transition
    sentence ("The X is W by H by D and origins at (...). Inside, it
    contains Y with the following plan:") that hands off to the next
    zone in the path. Sibling subtrees off the path are enumerated
    afterwards under "Other areas in <parent>:". Nesting inside those
    sub-area lists uses `> ` as a depth delimiter (one `>` per level)
    in place of visual indentation.
    """
    bag: dict[str, dict] = {}
    for i, (aid, aprompt, aplan, abbox, aplacement) in enumerate(ancestors):
        ap = None if i == 0 else ancestors[i - 1][0]
        bag[aid] = dict(
            prompt=aprompt, bbox=abbox, plan=aplan,
            parent_id=ap, placement=aplacement, kind="zone",
        )
    target_parent = ancestors[-1][0] if ancestors else None
    bag[target_zone_id] = dict(
        prompt=target_zone_prompt, bbox=target_zone_bbox, plan=target_zone_plan,
        parent_id=target_parent, placement=None, kind="zone",
    )
    for zid, zprompt, zplan, zparent, zbbox, zplacement in prior_zones:
        if zid in bag:
            continue
        bag[zid] = dict(
            prompt=zprompt, bbox=zbbox, plan=zplan,
            parent_id=zparent, placement=zplacement, kind="zone",
        )
    for oid, oprompt, oparent, obbox, oplacement, oparent_kind in objects:
        if oid in bag:
            continue
        bag[oid] = dict(
            prompt=oprompt, bbox=obbox, plan=None,
            parent_id=oparent, placement=oplacement, kind="object",
            parent_kind=oparent_kind,
        )

    children: dict[str, list[str]] = {}
    for nid, n in bag.items():
        pid = n["parent_id"]
        if pid is None or pid not in bag:
            continue
        children.setdefault(pid, []).append(nid)

    spine = [a[0] for a in ancestors] + [target_zone_id]

    def fmt_dims_sentence(b: BoundingBox) -> str:
        w, h, d = b.size
        ox, oy, oz = b.origin
        return (
            f"{w:.2f}m by {h:.2f}m by {d:.2f}m and origins at "
            f"({ox:.2f}, {oy:.2f}, {oz:.2f})"
        )

    def fmt_dims_inline(b: BoundingBox) -> str:
        w, h, d = b.size
        ox, oy, oz = b.origin
        return (
            f"{w:.2f}m by {h:.2f}m by {d:.2f}m, origin "
            f"({ox:.2f}, {oy:.2f}, {oz:.2f})"
        )

    def fmt_plan(plan: str | None) -> str:
        if plan is None:
            return "(plan not yet authored — this zone has been declared and placed but not individually planned)"
        text = plan.rstrip()
        if not text.endswith("."):
            text += "."
        return text

    def fmt_framed_by(zone_id: str) -> str:
        """Return the trailing sentence describing the zone's concrete
        children:
          * frames + anchors → "The X is framed by F1, F2, and it contains O1 (P1), O2 (P2)."
          * frames only      → "The X is framed by F1, F2."
          * anchors only     → "The X contains O1 (P1), O2 (P2)."
          * neither          → "" (empty string)
        Frames are concrete children with parent_kind=ATTACHED (shell
        elements); anchors are concrete children with parent_kind=ON
        or IN. Concrete children with unknown parent_kind fall through
        to the anchor list so they're still surfaced."""
        concrete = [k for k in children.get(zone_id, []) if bag[k]["kind"] == "object"]
        if not concrete:
            return ""
        frames = [k for k in concrete if bag[k].get("parent_kind") == "ATTACHED"]
        anchors = [k for k in concrete if bag[k].get("parent_kind") in ("ON", "IN")]
        leftovers = [k for k in concrete if k not in frames and k not in anchors]
        anchors = anchors + leftovers

        def fmt_anchor(cid: str) -> str:
            placement = bag[cid]["placement"] or "no explicit placement recorded"
            return f"{cid} ({placement})"

        if frames and anchors:
            return (
                f" The {zone_id} is framed by " + ", ".join(frames)
                + ", and it contains " + ", ".join(fmt_anchor(a) for a in anchors)
                + "."
            )
        if frames:
            return f" The {zone_id} is framed by " + ", ".join(frames) + "."
        # anchors only
        return f" The {zone_id} contains " + ", ".join(fmt_anchor(a) for a in anchors) + "."

    out: list[str] = []
    out.append(
        f"Here is a list of zones that have already been declared and/or "
        f"planned beforehand. You should use their plans and locations to aid "
        f"you in deciding on the structural decomposition of {target_zone_id}. "
        f"Starting with the root zone encapsulating the entire scene, here is its plan:"
    )
    out.append("")

    for i, nid in enumerate(spine):
        n = bag[nid]
        is_target = nid == target_zone_id
        out.append(fmt_plan(n["plan"]) + fmt_framed_by(nid))
        out.append("")
        if is_target:
            out.append(
                f"The {nid} is {fmt_dims_sentence(n['bbox'])}. "
                f"You are to decide on the structural decomposition of this {nid}."
            )
        else:
            nxt = spine[i + 1]
            out.append(
                f"The {nid} is {fmt_dims_sentence(n['bbox'])}. "
                f"Inside, it contains {nxt} with the following plan:"
            )
        out.append("")

    out.append(
        f"To that end, here's some more context on the other areas inside the "
        f"scene. You should reference this information to help you in deciding "
        f"on the structural decomposition of {target_zone_id}."
    )
    out.append("")
    out.append(
        "Nested sub-area lists below use `> ` as a depth delimiter: every line "
        "is prefixed with one `> ` per level of nesting below its enclosing "
        "\"Other areas in X\" or \"Areas in X\" header (top-level entries have "
        "no prefix; their sub-areas get one `> `; sub-sub-areas get `> > `; etc.)."
    )
    out.append("")

    def render_subtree(zone_id: str, num: int, depth: int) -> list[str]:
        n = bag[zone_id]
        prefix = "> " * depth
        sub_prefix = "> " * (depth + 1)
        lines: list[str] = []
        lines.append(f"{prefix}{num}. {zone_id} ({fmt_dims_inline(n['bbox'])})")
        lines.append(prefix.rstrip() if prefix else "")
        plan_para = fmt_plan(n["plan"]) + fmt_framed_by(zone_id)
        for ln in plan_para.split("\n"):
            lines.append(f"{prefix}{ln}")
        zone_kids = [k for k in children.get(zone_id, []) if bag[k]["kind"] == "zone"]
        if zone_kids:
            lines.append(sub_prefix.rstrip() if sub_prefix else "")
            lines.append(f"{sub_prefix}Areas in {zone_id}:")
            lines.append(sub_prefix.rstrip() if sub_prefix else "")
            for j, sub in enumerate(zone_kids, 1):
                lines.extend(render_subtree(sub, j, depth + 1))
                if j < len(zone_kids):
                    lines.append(sub_prefix.rstrip() if sub_prefix else "")
        return lines

    any_siblings = False
    for i in range(len(spine) - 1):
        parent_id = spine[i]
        path_child = spine[i + 1]
        sibling_zone_ids = [
            k for k in children.get(parent_id, [])
            if bag[k]["kind"] == "zone" and k != path_child
        ]
        if not sibling_zone_ids:
            continue
        any_siblings = True
        is_target_parent = parent_id == target_parent
        marker = (
            f" <- the {target_zone_id} you are to decompose exists here"
            if is_target_parent else ""
        )
        out.append(f"Other areas in {parent_id}:{marker}")
        out.append("")
        for j, sib in enumerate(sibling_zone_ids, 1):
            out.extend(render_subtree(sib, j, 0))
            if j < len(sibling_zone_ids):
                out.append("")
        out.append("")
    if not any_siblings:
        out.append("(no other declared areas — the path above is the entire declared scene so far.)")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def render_zone_decompose(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_bbox: BoundingBox,
    zone_plan: str,
    ancestors: list[tuple[str, str, str, BoundingBox, str | None]],
    objects: list[tuple[str, str, str | None, BoundingBox, str | None, str | None]],
    scene_prompt: str,
    scene_plan: str,
    prior_zones: list[tuple[str, str, str | None, str, BoundingBox, str | None]],
) -> str:
    """ancestors: (id, prompt, plan, bbox, placement) tuples from root →
    parent of this zone, excluding the zone itself. Empty for the root.
    `placement` is None for root only.
    objects: (id, prompt, parent_id, bbox, placement, parent_kind) tuples
    for every concrete (mesh-bearing) node placed anywhere in the run so
    far. `parent_kind` is `'ATTACHED'` for shell frames and `'ON' / 'IN'`
    for interior anchor objects.
    prior_zones: (id, prompt, plan_or_None, parent_id, bbox, placement)
    for every non-root zone already declared in the run, in declaration
    order. `plan` is None for zones that have been declared (bbox
    resolved) but not yet recursed into for individual planning."""
    narrative = _scene_context_zone_decompose_narrative(
        target_zone_id=zone_id,
        target_zone_prompt=zone_prompt,
        target_zone_bbox=zone_bbox,
        target_zone_plan=zone_plan,
        scene_prompt=scene_prompt,
        scene_plan=scene_plan,
        ancestors=ancestors,
        prior_zones=prior_zones,
        objects=objects,
    )
    return f"""Generate a list of subzones that should be present in the following scene, based on its description:

{zone_plan}

<IMPORTANT_INSTRUCTIONS>

<ZONE_SPLITTING_GUIDANCE>
Child zones can keep decomposing into more zones recursively in subsequent passes, or end there as atomic leaves if that is appropriate. so always decompose at the TOP MOST LEVEL of the current zone — e.g. for a house scene with backyard, driveway, and house, do not skip straight to backyard-pool zone, backyard-grass zone, house-basement, house-first-floor, etc.; decompose into "the house", "the backyard", "the driveway" as top-level children, and let the next recursion split the house into floors and the backyard into pool and grass. the same principle holds everywhere: emit only the zones that exist at THIS level of the hierarchy, and trust the recursive planning + decompose passes underneath each of them to handle the next layer down.

</ZONE_SPLITTING_GUIDANCE>

<PHYSICAL_GROUNDING>
For EVERY child subzone you propose, before writing its placement, explicitly answer in your reasoning: "what physically holds this child up?" The answer must point at something CONCRETE — the parent's floor plane, the top face of a sibling subzone in this batch, an existing peer in the scene context, or an attachment face on the parent. Writing a justification only in placement prose ("rests on supporting columns descending from the combat floor") does NOT make those columns exist; placement text is consumed by the downstream bbox-resolution step, it does not spawn new nodes. If your answer references support that does not already exist as a node and that you have not committed to emit as another child in THIS call, ADD that support as a sibling subzone in this same `children` list — a peer subzone whose role is to physically ground or connect the dependent child. The bbox of that support subzone must claim the airspace it occupies, so it ends up as a real physical region in the layout, not just an idea.

If you genuinely believe the child needs no physical support, you must justify it from the scene's PHYSICAL REGIME as established by the scene plan and ancestor plans — e.g. a vacuum / outer-space setting, an explicit antigravity field, a cloud-natured or particulate entity, a dreamscape. A vague "it floats" with no grounding in the scene's physics is not acceptable; the rest of the pipeline will treat it as a physically present zone and place objects inside it as if it were solid ground.

This question is per-child and load-bearing: if it surfaces a gap between what the plan promises and what you have actually committed to emit, the fix is to emit the missing support node now — not to paper over it in prose.
</PHYSICAL_GROUNDING>

Think very intricately and spatially about how this zone splits. Your goal is to reason a subzone decomposition layout that fits the narrative presented by the scene plan given above as well as the additional plans of ancestor scenes in the scene context section given below, while paying attention to the semantic relationships between the subzones.

The seed prompt you output for each subzone should be a 1-2 sentences long description that explains the subzone's shape and character. Be concrete about its description while leaving room for this prompt to be a seed for a more detailed plan. The prompt should be succinct without mentioning going overly into detail on the subzone's contents, but should mention the the narrative meaning behind its existence and the narrative meaning behind its relative placement to other subzones.

Keep the prompt tight: the goal is not to plan out the subzone's contents, but to establish its character as a piece of the larger scene as a whole.
</IMPORTANT_INSTRUCTIONS>

<SCENE_CONTEXT>
{narrative}

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


SYSTEM_ZONE_BBOX_BATCH = f"""You are a natural-language constraint solver. Place ALL sibling child ZONES inside a parent zone in one shot, deriving each child's axis-aligned bounding box from the given inputs. This step has no creative latitude — your job is to produce coordinates that honor every stated placement description simultaneously.

Inputs:
  * The parent zone's id and bbox.
  * A list of child specs — each with:
    - `id`, `prompt`, `proxy_shape`
    - `parent`: id of this child's structural anchor (either the parent zone itself or an earlier sibling in this batch). The child's bbox must lie fully inside the parent zone's bbox; positioning relative to a sibling-parent happens within that container.
    - `parent_kind`: one of ON / ATTACHED / IN. Tells you how the child anchors to its parent — `IN` (containment, child's bbox lies inside the parent's volume), `ON` (rest, child's bottom sits on the parent's outward surface — typically top face), or `ATTACHED` (flush against a face of the parent). Use it to disambiguate placement prose when it's ambiguous (e.g. "centered on the parent" + parent_kind=ON means resting on the top, not embedded inside).
    - `placement`: prose describing precise positioning (centered, edge-aligned, two-thirds along, etc.). Use this verbatim to pick coordinates.
    - `referenced_ids`: optional list of `{{target, kind}}` pairs naming secondary relationships the placement text refers to. `kind` is one of ON / BESIDE / ABOVE / BELOW / ATTACHED / IN — a coarse category that disambiguates the prose (e.g. "next to the courtyard" + kind=BESIDE → adjacent on X/Z, not stacked vertically; kind=IN → contained inside the target's volume). The kind is a hint, not a constraint: precise positioning lives in placement.

How to read a placement:
  * Resolve sibling references against `parent` + `referenced_ids` and the bboxes you have already chosen for prior siblings in this batch.
  * Phrases like "centered", "back-left corner", "spanning the full width", "flush against the front wall", "behind the X", "between X and Y" should map to concrete coordinate choices.
  * If a description is ambiguous, lean on `referenced_ids` kinds (ON → contact on top face, BESIDE → adjacent X/Z, IN → contained inside target's volume, etc.) and the parent's plan to disambiguate.

Canonical front view for all coordinates: +X right, +Y up, +Z front, -Z back. Right-handed, Y-up, meters. Treat the placement text's "front" as +Z, "left" as -X, "top" as +Y, etc.

Each child carries a `proxy_shape` describing its mesh silhouette inside its AABB — BOX, SPHERE, CAPSULE, or HEMISPHERE. The PROXY SHAPE section below gives the exact surface formula Y_top(x, z) for each. When a child's placement describes it as resting on a sibling with a non-BOX proxy, that child's bottom face should sit at the target's Y_top(x, z) at the child's XZ centre — NOT on the target's AABB top face. There is no automatic correction: YOU must compute the target's Y_top(x, z) from its AABB and proxy formula and place the resting child's bbox accordingly. Pick dimensions so a HEMISPHERE target has vertical headroom above its apex.

Produce one assignment per child (id + bbox) such that:
  * Every bbox lies fully inside the parent bbox.
  * No two child bboxes overlap volumetrically. Touching at a shared face is fine; eating into another bbox's volume is not.
  * Every placement description is honored as faithfully as possible.
  * Dimensions are appropriate to each child's prompt.

Because you are deciding the entire layout at once, RESERVE SPACE for every child up front rather than committing each bbox in isolation. A later child's requirements must influence earlier siblings' sizing.

Coordinates in meters, centimeter precision (multiples of 0.01). Use a signed `dimensions` vector from an `origin` vertex; sign chooses expansion direction. Emit exactly one assignment per requested child id, no extras, no omissions.

<proxy_shape>
{PROXY_SHAPE_DOC}
</proxy_shape>

Respond with ONE JSON object matching the schema. No prose, no markdown, no code fences."""


def _render_relationships(rels: list[Relationship]) -> str:
    """Render a node's secondary relationships as an inline JSON-ish
    list. Empty list emits the literal `[]` so the reader can tell the
    model intentionally had no secondaries."""
    if not rels:
        return "[]"
    items = [f"{{target={r.target!r}, kind={r.kind.value}}}" for r in rels]
    return "[" + ", ".join(items) + "]"


def render_zone_bbox_batch(
    *,
    parent_id: str,
    parent_bbox: BoundingBox,
    children: list["ChildNodeSpec"],
) -> str:
    child_lines = "\n\n".join(
        f"""  - id={c.id!r}
    parent: {c.parent!r}
    parent_kind: {c.parent_kind.value}
    prompt: {c.prompt}
    proxy_shape: {_render_proxy_shape(c.proxy_shape)}
    placement: {c.placement}
    referenced_ids: {_render_relationships(c.referenced_ids)}"""
        for c in children
    )
    return f"""Parent id: {parent_id!r}
Parent bbox: {parent_bbox.model_dump_json()}

Children to place ({len(children)}):
{child_lines}

Produce a bbox for every child in a single coherent layout."""


# ---------- Step 5: object decomposition (Phase 2) --------------------------


class ObjectSpec(ChildNodeSpec):
    """A single object in a zone. Identical shape to ChildNodeSpec.
    The structural parent (`parent` field) may be the enclosing zone,
    a frame, an earlier-placed peer, or another object listed in the
    same decomp call. Secondary relationships (`referenced_ids`)
    capture additional spatial connections — sibling alignment, the
    wall a painting hangs against, etc."""


class BoundExistingFrame(BaseModel):
    """Audit entry for the encapsulating step's BINDING CONTRACT.

    When a structural element named in the zone's plan is already
    satisfied by an existing peer (option (b) of the contract), the LLM
    records the binding here instead of emitting a new shell node. These
    entries do NOT become Nodes and do NOT flow into bbox-resolution or
    any downstream step — they're a per-call audit trail proving the
    LLM made the binding choice deliberately rather than silently
    skipping the plan-named element.
    """

    plan_element: str  # noun phrase from the zone plan (e.g. "rear partition wall")
    peer_id: str  # id of an existing node in <OBJECTS> that satisfies it


class ObjectDecompOutput(BaseModel):
    # NEWLY EMITTED shell/object specs. These become Nodes downstream.
    objects: list[ObjectSpec] = Field(default_factory=list)
    # OPTION (b) audit entries for the encapsulating step's binding
    # contract. Used only by the encapsulating decomposer; anchor and
    # negative-space decompositions leave this empty. Captured in the
    # event log (visible in the observability view) but dropped before
    # downstream pipeline steps.
    bound_existing: list[BoundExistingFrame] = Field(default_factory=list)


# Universal scaffolding shared across all three decomposition modes:
# per-object schema, proxy_shape vocabulary, no-ephemera rule, and the
# final "JSON only" output reminder. Each mode-specific system prompt
# concatenates its own intro+rules with this tail.
_OBJECT_DECOMP_TAIL = f"""<per_object_fields>
For each object, emit:
  * `id` — unique within this call.
  * `prompt` — a detailed description of the object; will be used verbatim as a text-to-3D generation prompt.
  * `parent` — the id of this object's STRUCTURAL PARENT: the thing it physically rests on, hangs from, leans against, or is contained by. Examples:
      teacup on a saucer → parent='saucer'
      lamp on a desk → parent='desk'
      painting on a wall → parent='back_wall' (a frame)
      floor frame → parent='<zone_id>' (the floor IS the zone's bottom boundary)
      cloud floating in a sky zone → parent='sky' (the zone, since nothing physically supports it)
    The chain of `parent` edges must terminate at the zone — every object eventually grounds out to the zone via this chain.
  * `parent_kind` — how this object anchors to its `parent`. MUST be exactly one of `ON`, `ATTACHED`, or `IN`. Pick the one that best describes the physical anchor type:
      * `ON` — the object rests on the parent's outward surface. Most often this is the top face (cup on table, lamp on desk, statue on plinth, anchor object on a floor frame), but it can be any outward-facing surface a non-BOX-proxy parent presents (a boulder resting on a hemispherical hill rests `ON` the hill).
      * `ATTACHED` — the object is flush against any face of the parent: wall mounts (painting on wall, sconce on wall, mirror on wall), ceiling mounts (chandelier from ceiling, ceiling fan, smoke detector), embedded fittings (door in a wall frame, window in a wall frame), hangers, and magnets. Walls/floor/ceiling frames themselves are `ATTACHED` to their enclosing zone (they form the zone's shell).
      * `IN` — the object is contained inside the parent's volume / footprint with no specific contact face: a fish inside an aquarium, a cloud floating inside a sky zone, drifting particles, a free-floating drone inside an enclosed dome, a decorative object inside an open container that doesn't sit on the container's floor (e.g. confetti mid-air inside a jar). Also use `IN` when the parent is the enclosing zone and the object simply occupies space inside without a clear supporting face.
    `BESIDE`, `ABOVE`, and `BELOW` are NOT valid here — those describe non-contact peer arrangements and belong in `referenced_ids` only.
  * `proxy_shape` — OPTIONAL. The object's collision-proxy shape if its silhouette is noticeably non-rectilinear (see PROXY SHAPE section below). Omit for objects whose bbox is already a good silhouette.
  * `orientation` — world-frame yaw about +Y in degrees. MUST be one of the allowed values: -180, -135, -90, -45, 0, 45, 90, 135, 180. The image-to-3D model receives an ORTHOGRAPHIC FRONT VIEW of the object, so its mesh comes back with the visible front face along world +Z. `orientation` rotates the mesh into the intended world pose. Right-handed about +Y: `0` = front faces +Z (toward viewer); `90` = front faces -X; `180` = front faces -Z (away); `-90` = front faces +X. Examples: a sofa whose seat opens toward the room centre needs orientation set so its front faces the room interior, not the wall. A door in a wall on the +X face of a room needs `-90` so the door faces +X. The bbox stays an AABB — orientation only rotates the mesh inside it, so a long object's bbox dimensions must match its long axis AFTER rotation. Use 0 for symmetric objects with no preferred facing.
  * `placement` — a prose description of WHERE this object sits. Describe position within or on the parent, and any alignment to other objects or features (which should also appear as entries in `referenced_ids`). Examples:
      "centered on the dining table's top surface"
      "flush against the back wall, mid-height, centered horizontally"
      "on the floor in the back-left corner of the room, oriented to face the room's center"
      "to the left of the sofa, resting on the rug, between the sofa and the side table"
  * `referenced_ids` — OPTIONAL list of secondary spatial relationships when your placement text refers to nodes other than the parent. Each entry is `{{target: <id>, kind: <ON|BESIDE|ABOVE|BELOW|ATTACHED|IN>}}`. Pick the kind that best summarizes the abstract category of the relationship — precise positioning is in the placement prose, not in the kind. Use `IN` when the target physically contains this object (e.g. a fish whose parent is the tank-water, also `IN` the surrounding aquarium glass). Examples:
      Lamp on the desk, to the left of the toaster:
        parent='desk', parent_kind=ON,
        referenced_ids=[{{target='toaster', kind=BESIDE}}]
      Chair between the dining table and the wall:
        parent='dining_room_floor', parent_kind=ON,
        referenced_ids=[{{target='dining_table', kind=BESIDE}}, {{target='back_wall', kind=BESIDE}}]
      Painting hung on a wall, aligned with a sconce below it:
        parent='back_wall', parent_kind=ATTACHED,
        referenced_ids=[{{target='wall_sconce', kind=ABOVE}}]
      Drifting cloud inside the sky zone, near a peer mountain peak:
        parent='sky_zone', parent_kind=IN,
        referenced_ids=[{{target='mountain_peak', kind=BESIDE}}]
    Do NOT include the parent in `referenced_ids`. Do NOT add entries your placement text doesn't actually mention. Empty list is fine.
    ANY id from the prompt context is a valid target — the <ANCESTOR_CHAIN>, the <ZONES> and <OBJECTS> blocks, and earlier-listed objects in THIS call.

Besides the choice of objects themselves and their aesthetic, the SPATIAL COHERENCE of the scene is the most important part of the benchmark. Reason carefully about each object's parent + placement and how everything fits together.

DO NOT pick concrete coordinates here — a downstream constraint solver resolves each object's bbox from its parent, placement, referenced_ids, and the zone + peer geometry.
</per_object_fields>

<proxy_shape>
{PROXY_SHAPE_DOC}
</proxy_shape>

<no_ephemera>
{NO_EPHEMERA_DOC}
</no_ephemera>

ALWAYS think deeply before responding.

Respond with ONLY ONE JSON object matching the schema. No prose, no markdown, no code fences."""


SYSTEM_ANCHOR_DECOMP = f"""You are enumerating the DEFINING ANCHOR OBJECTS of an atomic leaf zone inside StarshotBench — a head-to-head competitive benchmark for 3D spatial reasoning where your scene will be rendered and judged against another LLM's rendering of the same user prompt.

The final spatial and aesthetic output of the scene you produce here is WHAT THE JUDGES ACTUALLY SEE. Zones and plans are scaffolding; objects are the scene. Thoughtful anchor choices make a zone unmistakably, immediately recognizable as its subject — a meeting room's long conference table, chairs and screen on the end wall; a castle throne room's raised dais, carved chair, and flanking banners; an island's lone lightning-split stump, knotted roots at the waterline, and the red objective flag planted at its crest. Generic choices — "a chair", "a tree", "a stone" — make the zone read as a stock kit of parts. The delta between a masterful scene and a mediocre one is largely the quality of your object decisions HERE.

Push past the obvious first pick. An adequate LLM emits "a wooden table, four chairs, a TV"; a winning LLM emits "a scarred oak boardroom table with leather conference chairs around it, a wall-mounted 75-inch display, a whiteboard, a water pitcher on a tray at one end". Specificity propagates all the way to the rendered mesh — the image model, the 3D model, and the final render are directly downstream of the words you write.

Give the zone a story; make it as impressive as possible.

<task>
Enumerate the DEFINING anchor objects that make this zone unmistakably what it is. A meeting room: a large table, chairs around it, a TV on the end wall. A toilet area: a toilet, a toilet paper holder. An island: the flag, the roots at the waterline, a gnarled tree. Do NOT include decorative filler — a later iterative step adds more objects one at a time on top of your anchors.
</task>

<ground_awareness_rule>
This zone may already have a GROUND / SHELL peer placed by the encapsulating pass (an island dome, a crater bowl, a curved floor, the walls+floor of a room) — look in <OBJECTS> for a peer whose parent is this zone and whose prompt describes terrain or enclosure geometry. If such a peer exists, every anchor object in this zone whose physical support IS that terrain/floor MUST have that ground/shell peer's id as the FIRST entry of its `referenced_ids` (NOT the zone id), and its `placement` must describe how it rests on that surface. The peer's `proxy_shape` (shown alongside its bbox in <OBJECTS>) is the authoritative descriptor of its surface — a HEMISPHERE peer is a dome whose real surface dips from the AABB centre to the edges. You are NOT placing bboxes at this step, but choose the right primary anchor now: the downstream bbox-resolution step will compute the dome's surface height at each anchor's XZ from the peer's proxy formula and rest the anchor on that surface. Do NOT re-emit the ground itself in anchor mode; the encapsulating pass already placed it.
</ground_awareness_rule>

<inputs>
You are given two structured XML-tagged context blocks:

  * <ANCESTOR_CHAIN> — every zone above this one, root first → this zone's direct parent. Each `<node>` shows id, parent, prompt, bbox, placement (root shows `(root — has no parent)`), and plan.
  * <ZONES> — every abstract region (has a `plan`). Each `<node>` shows id, parent, prompt, bbox, placement, plan. Useful as a reference target when you want to anchor relative to a containing region's extents rather than to a specific object.
  * <OBJECTS> — every concrete frame, anchor, or prior negative-space piece (no plan; some have a mesh). Each `<node>` shows id, parent, prompt, bbox, proxy_shape, orientation, placement. This is what any new object you emit physically rests on, leans against, or anchors to — by `referenced_ids` and in `placement` prose.

Reason about all three blocks thoroughly before emitting.
</inputs>

<global_rules>
  * DO NOT DUPLICATE GEOMETRY. If a parent zone or sibling has already placed a wall / floor / ceiling that covers one of this zone's faces, do NOT emit another one for that face. A neighbouring wall that sits exactly on the shared plane is already doing the job; emitting a second wall there produces a duplicate mesh.
</global_rules>

{_OBJECT_DECOMP_TAIL}"""


SYSTEM_ENCAPSULATING_DECOMP = f"""You are enumerating the PHYSICAL SHELL of a zone inside StarshotBench — a head-to-head competitive benchmark for 3D spatial reasoning where your scene will be rendered and judged against another LLM's rendering of the same user prompt.

The shell you emit here is what every subsequent object inside this zone rests on, leans against, or is bounded by. Get the surface shape right and the downstream anchor pass places its objects coherently; get it wrong (a flat slab where a dome belongs, a missing back wall, duplicated floor at a shared face) and the whole interior reads as broken. Encapsulating decisions propagate to every anchor placed afterward.

<task>
Emit the geometry that PHYSICALLY BOUNDS this zone before anything else populates it.

  * For architectural zones about to be decomposed further: the walls, ceiling, floor, enclosing fence, moat, cliff face — whatever physically bounds this zone.
  * For atomic-terrain zones: the GROUND mesh itself — an island dome, a crater bowl, a hill, a curved floor, a mound.

Emit one object per shell element. Each object's prompt is sent verbatim to a text-to-3D model, so describe it as a concrete artifact. For ground/terrain shells, describe the actual surface SHAPE in concrete terms so later anchor-mode placements can reason about the surface height at any XZ — "a muddy domed island raised ~1.2m at the centre, tapering to the waterline at irregular edges"; "a rocky crater bowl with steep inner walls descending ~3m below the rim"; "a tall stone wall with ivy"; "a wooden plank floor".
</task>

<proxy_shape_rule>
For TERRAIN SHELLS this is CRITICAL: a domed island MUST set `proxy_shape=HEMISPHERE`, otherwise every anchor object placed ON it will float above its AABB top instead of resting on the actual dome. Boulders use SPHERE, columnar shells use CAPSULE, architectural shells (walls, floors, ceilings, fences) leave proxy_shape unset.
</proxy_shape_rule>

<inputs>
You are given three structured XML-tagged context blocks:

  * <ANCESTOR_CHAIN> — every zone above this one (root → this zone's direct parent). Each `<node>` shows id, parent, prompt, bbox, placement, and plan.
  * <ZONES> — every other abstract region in the run (has a `plan`). Each `<node>` shows id, parent, prompt, bbox, placement, plan.
  * <OBJECTS> — every concrete frame, anchor, or prior negative-space piece (no plan). Each `<node>` shows id, parent, prompt, bbox, proxy_shape, orientation, placement.

The most important things to scan for in <OBJECTS> are slabs at this zone's boundary planes that a neighbour or parent has already emitted — see the duplicate-geometry rule below.
</inputs>

<global_rules>
  * DO NOT DUPLICATE GEOMETRY. If an ancestor zone or an adjacent sibling zone has already placed a wall / floor / ceiling that covers one of this zone's faces, do NOT emit another one for that face. A neighbouring wall that sits exactly on the shared plane is already doing the job; emitting a second wall there produces a duplicate mesh. Thin slabs at zone boundaries are the easiest thing to accidentally re-emit, so audit the CURRENT SCENE for boundary-plane peers before adding any shell face.
</global_rules>

{_OBJECT_DECOMP_TAIL}"""


SYSTEM_NEGATIVE_SPACE_DECOMP = f"""You are filling the AMBIENT, CONNECTIVE, INTERSTITIAL space between the named zones of a 3D scene inside StarshotBench — a head-to-head competitive benchmark for 3D spatial reasoning where your scene will be rendered and judged against another LLM's rendering of the same user prompt.

Every named zone of the scene has already been decomposed and populated. What remains is the layer the named zones do not own: lilypads drifting across swamp water between islands, grass tufts scattered over a meadow, floating logs and driftwood across open water, scattered stones across a plain, loose paper blowing across a plaza. This ambient layer is what binds the scene together — without it the named zones read as isolated dioramas; with it the scene reads as a continuous world.

<task>
Generate a list of objects that fill the negative, unfilled space between the zones and objects already placed in the scene. Each item must be a SOLID, BOUNDED, instanced object — individual things populating the gaps, not abstractions.

Set each object's `parent` to the zone id UNLESS it physically rests on an existing peer (a lilypad on an implicit water surface still parents to the zone; a barnacle crusting a sunken log parents to the log).

Do NOT re-emit anything that already exists as a zone or as a zone's anchor — negative-space content is strictly the ambient layer the named zones do not own.
</task>

<inputs>
You are given two XML-tagged context blocks:

  * <ZONES> — every abstract region in the scene tree. Each `<node>` shows id, parent, prompt, bbox, placement, plan. This is the structure you are filling the gaps BETWEEN. Use it to reason about where the negative space actually lives (between which zones, around which features, across what implied surface).
  * <OBJECTS> — every concrete frame, anchor, or earlier negative-space piece (no plan). Each `<node>` shows id, parent, prompt, bbox, proxy_shape, orientation, and placement. Use it to avoid duplicating geometry that already exists, to reason about what surface each ambient item rests on, and to pick concrete `referenced_ids` for the items you emit.
</inputs>

<solid_only>
{NO_EPHEMERA_DOC}
</solid_only>

<global_rules>
  * DO NOT DUPLICATE GEOMETRY. Do not re-emit any zone or anchor object that already appears in <ZONES> or <OBJECTS>. Your job is the ambient layer that BETWEEN-zone space implies, not a second pass over the named zones.
  * STAY IN THE GAPS. Keep each item's implied volume inside this zone's bbox but OUTSIDE the bboxes of any already-placed peer. Items should populate the unclaimed interstitial regions.
</global_rules>

{_OBJECT_DECOMP_TAIL}"""


# --- structured context rendering ------------------------------------------
#
# Every block of "things already in the scene" the LLM sees uses one shape:
#
#   <SECTION_TAG>
#     <node id="..." parent="...">
#       prompt: ...
#       bbox: {...}
#       placement: ...          # absent on root (kind == "root")
#       proxy_shape: ...        # absent unless emitted
#       orientation: 0deg       # absent unless emitted
#       plan: ...               # absent unless emitted
#     </node>
#     <node id="..." parent="...">
#       ...
#     </node>
#   </SECTION_TAG>
#
# Section tags are stable across all prompts: ANCESTOR_CHAIN, ZONES,
# OBJECTS. Field names are stable too.
# The model can parse this as a structured record stream — XML attrs for
# identifiers, key:value lines for content. One central node-renderer
# guarantees field order and quoting are consistent so the LLM never has
# to recover from inconsistent layouts mid-prompt.

def _render_node_entry(
    *,
    nid: str,
    parent: str | None,
    prompt: str | None = None,
    bbox: BoundingBox | None = None,
    placement: str | None = None,
    placement_unset_label: str | None = None,
    proxy_shape: ProxyShape | None = None,
    orientation: int | None = None,
    plan: str | None = None,
    plan_unset_label: str | None = None,
) -> str:
    """Render one node as a `<node>...</node>` block. Optional fields are
    omitted entirely when None unless an `*_unset_label` is supplied, in
    which case the label is rendered in their place (e.g. root has no
    placement so it shows `placement: (root — has no parent)`)."""
    parent_attr = f' parent="{parent}"' if parent is not None else ' parent="(none)"'
    head = f'<node id="{nid}"{parent_attr}>'
    lines: list[str] = []
    if prompt is not None:
        lines.append(f"  prompt: {prompt}")
    if bbox is not None:
        lines.append(f"  bbox: {bbox.model_dump_json()}")
    if proxy_shape is not None:
        lines.append(f"  proxy_shape: {_render_proxy_shape(proxy_shape)}")
    if orientation is not None:
        lines.append(f"  orientation: {orientation}deg")
    if placement is not None:
        lines.append(f"  placement: {placement}")
    elif placement_unset_label is not None:
        lines.append(f"  placement: {placement_unset_label}")
    if plan is not None:
        lines.append(f"  plan: {plan}")
    elif plan_unset_label is not None:
        lines.append(f"  plan: {plan_unset_label}")
    return head + "\n" + "\n".join(lines) + "\n</node>"


def _render_section(tag: str, entries: list[str], empty_note: str) -> str:
    """Wrap a list of pre-rendered `<node>` entries in a section tag.
    `empty_note` is shown when there are no entries."""
    if not entries:
        body = f"  ({empty_note})"
    else:
        body = "\n".join(entries)
    return f"<{tag}>\n{body}\n</{tag}>"


def _render_ancestor_block(
    ancestors: list[tuple[str, str, str, BoundingBox, str | None]],
) -> str:
    entries = [
        _render_node_entry(
            nid=aid,
            parent=None if i == 0 else ancestors[i - 1][0],
            prompt=aprompt,
            bbox=abbox,
            placement=aplacement,
            placement_unset_label="(root — has no parent)",
            plan=aplan,
        )
        for i, (aid, aprompt, aplan, abbox, aplacement) in enumerate(ancestors)
    ]
    return _render_section(
        "ANCESTOR_CHAIN", entries, "none — this is the root"
    )


def _render_zone_plan_block(zone_plan: str | None) -> str:
    if zone_plan is not None:
        return f"<zone_plan>{zone_plan}</zone_plan>"
    return "<zone_plan>(not yet authored — this zone has been declared by its parent but not individually planned; rely on the ancestor plans above for intent)</zone_plan>"


def _render_scene_lines(
    scene: list[tuple[str, str, BoundingBox, str | None, ProxyShape | None, Orientation, str | None, str | None]],
) -> str:
    """Split the run-wide scene snapshot into two sections — <ZONES>
    (abstract regions, identified by `plan is not None`) and <OBJECTS>
    (concrete frames/anchors/negative-space, identified by `plan is
    None`). Both blocks are emitted, joined by a blank line, so the
    caller can drop the result straight into <scene_context>."""
    zone_entries: list[str] = []
    object_entries: list[str] = []
    for nid, prompt, bbox, pid, proxy, orient, placement, plan in scene:
        entry = _render_node_entry(
            nid=nid,
            parent=pid,
            prompt=prompt,
            bbox=bbox,
            placement=placement,
            placement_unset_label="(root — has no parent)",
            proxy_shape=proxy,
            orientation=orient,
            plan=plan,
        )
        if plan is not None:
            zone_entries.append(entry)
        else:
            object_entries.append(entry)
    zones_block = _render_section("ZONES", zone_entries, "none")
    objects_block = _render_section("OBJECTS", object_entries, "none")
    return f"{zones_block}\n\n{objects_block}"


# Shared one-paragraph header at the top of every <scene_context> block.
# Gives the model a quick map of what tags it's about to see and what
# they mean, so the structure isn't a surprise mid-prompt.
_SCENE_CONTEXT_INTRO = "This is an overview of the current scene context — every entity already placed in the run that you can reference by id. <ANCESTOR_CHAIN> is the path from the root down to this zone's parent. <ZONES> are abstract regions (have a `plan`). <OBJECTS> are concrete frames, anchors, and prior negative-space pieces (have no plan; some have a mesh). Each `<node>` entry shows id, parent, prompt, bbox, placement, and any other applicable fields. Use these as concrete anchors when authoring placement prose and as valid `referenced_ids` targets."


def _render_retry_block(
    prior_attempts: list[tuple[list[ObjectSpec], str]] | None,
) -> str:
    if not prior_attempts:
        return ""
    attempt_lines = "\n\n".join(
        f"""  attempt {i}:
    emitted: [{', '.join(s.model_dump_json() for s in specs)}]
    rejected: {reason}"""
        for i, (specs, reason) in enumerate(prior_attempts)
    )
    return f"""

PRIOR ATTEMPTS — every decomposition below was ALREADY rejected. Do NOT re-emit the same set of object specs, and do not repeat the same structural mistake. Treat every listed reason as a hard constraint you must satisfy this time:
{attempt_lines}

Produce a NEW decomposition that fixes every listed reason. In particular, ensure every object's `parent` field is set to a valid existing id (the supporter or containing zone that anchors it), and every `referenced_ids` entry has a `target` that exists in the scene context."""


def render_anchor_decomp(
    *,
    zone_id: str,
    zone_plan: str | None,
    zone_bbox: BoundingBox,
    ancestors: list[tuple[str, str, str, BoundingBox, str | None]],
    scene: list[tuple[str, str, BoundingBox, str | None, ProxyShape | None, Orientation, str | None, str | None]],
    prior_attempts: list[tuple[list[ObjectSpec], str]] | None = None,
) -> str:
    return f"""Generate a list of defining anchor objects that make the zone described below unmistakably what it is, based on the attached context and zone plan. This zone is the lowest possible breakdown level: no other subzones can exist within, so it is defined by the anchor objects you are responsible for generating. Although the plan will be specific, the objects that the plan mentions may not be the end all be all: you should extrapolate meaning from the plan and higher-level ideas communicated in the ancestor chain to populate and style the objects in the list you generate based on a narrative understanding of the scene. Each object should be treated atomically, in the sense that collections of objects should be broken down into individual objects that allow more granular, precise positioning by you instead of relying on the outputted model's shape of downstream generation steps.

<scene_context>
{_SCENE_CONTEXT_INTRO}

ZONE_ID: {zone_id!r}
Zone bbox: {zone_bbox.model_dump_json()}
{_render_zone_plan_block(zone_plan)}

{_render_ancestor_block(ancestors)}

{_render_scene_lines(scene)}
</scene_context>

<output_guidance>
The concept of a parent should be grounded in a concrete, physical relationship, not a conceptual one. A cantilevered object would be parented to the surface or wall it's cantilevered to with relationship type 'ATTACHED', not parented to the floor below with relationship 'ON'. If no physical relationship is found with another object or frame, the relationship should be of type 'IN', and parented to the zone itself.

Each anchor object in your resultant list has an id, prompt, parent (the structural anchor id — the zone, a ground/shell peer from the encapsulating pass, or another object in this list), parent_kind (one of ON / ATTACHED / IN — how the object physically anchors to that parent: `ON` for resting on an outward surface, `ATTACHED` for wall/ceiling/face mounts, `IN` for free containment inside the parent's volume — BESIDE/ABOVE/BELOW are NOT valid here), placement (prose), and referenced_ids (optional list of `{{target, kind}}` for secondary relationships your placement text mentions; kind may use ON/BESIDE/ABOVE/BELOW/ATTACHED/IN). Respect the scene context: anchor onto any ground/shell peer already placed by the encapsulating pass, do not duplicate geometry another zone has already emitted. Anchor objects are expected to live primarily inside this zone, but their bboxes MAY protrude modestly outside the zone bbox when narratively justified — the object remains semantically part of this zone even though its geometry overhangs. Do not use this as license to claim airspace far from the zone or to volumetrically intersect another zone's load-bearing geometry. In your final output, each object's placement text should be direct and parametric. Avoid flowery language that states the narrative purpose of the positioning. Do not state the abstract reason of the positioning, only details that ground the position concretely. The position description should be absolute and succinct, leaving no creative liberty for the downstream constraint solver.
</output_guidance>

{_render_retry_block(prior_attempts)}"""


def render_encapsulating_decomp(
    *,
    zone_id: str,
    zone_plan: str | None,
    zone_bbox: BoundingBox,
    ancestors: list[tuple[str, str, str, BoundingBox, str | None]],
    scene: list[tuple[str, str, BoundingBox, str | None, ProxyShape | None, Orientation, str | None, str | None]],
    prior_attempts: list[tuple[list[ObjectSpec], str]] | None = None,
) -> str:
    return f"""Generate a list of shell, floor, and boundary geometry elements that physically bound the zone described below, based on the attached context and the plan for the zone.

<scene_context>
{_SCENE_CONTEXT_INTRO}

ZONE_ID: {zone_id!r}
Zone bbox: {zone_bbox.model_dump_json()}
{_render_zone_plan_block(zone_plan)}

{_render_ancestor_block(ancestors)}

{_render_scene_lines(scene)}
</scene_context>

Each shell element in your resultant list has an id, prompt, parent (the zone id — shells are structurally anchored to the zone itself), parent_kind (use `ATTACHED` for all shell elements: walls/floor/ceiling/roof are flush against the zone's faces by definition; only use `IN` for free-floating enclosure pieces that have no contact face, and `ON` is rare for shells — BESIDE/ABOVE/BELOW are NOT valid here), placement (prose), and referenced_ids (optional list of `{{target, kind}}` for any secondary relationships; kind may use ON/BESIDE/ABOVE/BELOW/ATTACHED/IN, e.g. a roof slab supported by walls below it would reference each wall with kind=ON). Shell elements MUST lie fully inside the zone bbox — they are the zone's physical boundary, and protruding outside it would create an inside-out shell. (Non-shell objects emitted in later passes are allowed to overhang the zone; shells are not.)

The purpose of these shell elements is to physically bound the zone before its interior is populated, so every later object inside this zone has a coherent surface to rest on, lean against, or be enclosed by.

It is imperative that every structural element named in this zone's plan (walls, partitions, floor, ceiling, roof, columns, shells, enclosing geometry of any kind) MUST be bound to a real node id by the end of this call. For each one, you have exactly two options, and your output has a SEPARATE LIST for each:
  (a) Emit a new shell element for it — add an entry to the `objects` list (id, prompt, parent, parent_kind, placement, referenced_ids, proxy_shape, orientation). Use this when no existing peer satisfies the role.
  (b) Point to an existing peer that already satisfies it — add an entry to the `bound_existing` list with `plan_element` (the plan's noun phrase verbatim, e.g. "rear partition wall") and `peer_id` (the id of a node already present in <OBJECTS> with a concrete bbox that physically covers that role). DO NOT also add an `objects` entry for the same element — option (b) is the entire response for that structural element.

The `bound_existing` list is an audit trail: it proves you consciously bound each plan-named element, and it is dropped before any downstream pipeline step. Only entries in `objects` become new nodes.

Silently omitting a plan-named structural element from BOTH lists is NOT allowed. Do NOT assume a future sibling, cousin, or descendant zone will emit it later: encapsulation runs independently per zone, and downstream anchor-object generation will reference these structures by id — if the node doesn't exist now, the anchor step will hallucinate around a non-existent reference. Duplicate-avoidance is satisfied by option (b), not by silently skipping.

{_render_retry_block(prior_attempts)}"""


def render_negative_space_decomp(
    *,
    zone_id: str,
    zone_plan: str | None,
    zone_bbox: BoundingBox,
    scene: list[tuple[str, str, BoundingBox, str | None, ProxyShape | None, Orientation, str | None, str | None]],
    prior_attempts: list[tuple[list[ObjectSpec], str]] | None = None,
) -> str:
    return f"""Generate a list of objects that would cover the negative, unfilled space between the objects in the zone described below, based on the attached context.

<scene_context>
{_SCENE_CONTEXT_INTRO}

ZONE_ID: {zone_id!r}
Zone bbox: {zone_bbox.model_dump_json()}
{_render_zone_plan_block(zone_plan)}

{_render_scene_lines(scene)}
</scene_context>

Each negative space object in your resultant list has an id, prompt, parent (structural anchor — the zone, an earlier-placed peer, or another object in this list), parent_kind (one of ON / ATTACHED / IN — how the object physically anchors to that parent: most negative-space pieces sit `ON` a ground/floor peer, `ATTACHED` for things mounted flush to a wall/ceiling, `IN` for free-floating pieces inside an enclosing volume; BESIDE/ABOVE/BELOW are NOT valid here), placement (prose), and referenced_ids (optional list of `{{target, kind}}` for secondary relationships your placement text mentions; kind may use ON/BESIDE/ABOVE/BELOW/ATTACHED/IN). Respect the scene context: do not duplicate geometry another zone has already emitted on a shared face. Negative-space pieces live primarily inside this zone, but their bboxes MAY protrude modestly outside the zone bbox when narratively justified (a vine draped over a wall, a banner hanging off an edge, drifting smoke crossing into an adjacent zone, a connective walkway or rope-bridge reaching toward a peer zone, a stabilizing strut or buttress extending below to ground against a peer). Do not use this as license to claim airspace far from the zone or to volumetrically intersect another zone's load-bearing geometry.

The primary purpose of these negative space objects is to make the scene feel coherent and cohesive, filling in the gaps between subzones or objects. As such, for each negative space to fill in, it is imperative to analyze the existing placed objects and zones surrounding it to create a smooth filling that does not look out of place.{_render_retry_block(prior_attempts)}"""


# ---------- Step 6: object bbox resolution ----------------------------------


SYSTEM_OBJECT_BBOX_BATCH = f"""You are a natural-language constraint solver. Place ALL objects for a scene ZONE in one shot, deriving each object's axis-aligned bounding box from the given inputs (zone bbox, object specs with placement prose, peer prompts/bboxes). This step has limited creative latitude: produce coordinates that honor each placement description and that respect the actual geometry implied by peer prompts.

Inputs:
  * Zone id, prompt, and bbox — the overall region being populated.
  * OBJECTS to place: each with:
    - `id`, `prompt`, `proxy_shape`, `orientation`
    - `parent`: id of the object's structural anchor — the zone, an earlier-placed peer, a frame, or an earlier-listed object in this batch. The object physically rests on / hangs from / is contained by this parent. The object's bbox typically extends OUTSIDE the parent's (a lamp's bbox is not inside the desk's); the "fully inside" rule applies to the ZONE, not to object parents.
    - `parent_kind`: one of ON / ATTACHED / IN. `ON` → the object's bottom face sits on the parent's outward (top) surface — for non-BOX proxies, on the proxy's Y_top(x, z). `ATTACHED` → the object is flush against one of the parent's faces (wall mount, ceiling mount, hanger). `IN` → the object is contained inside the parent's volume or footprint (no specific contact face, e.g. a fish inside a tank, a cloud inside a sky zone). Use it together with the placement prose to disambiguate.
    - `placement`: prose describing precise positioning relative to the parent (and any referenced peers). Use this verbatim to pick coordinates.
    - `referenced_ids`: optional list of `{{target, kind}}` pairs naming secondary relationships the placement text mentions. `kind` is one of ON / BESIDE / ABOVE / BELOW / ATTACHED / IN — a coarse category that disambiguates the prose. The kind is a hint; precise positioning lives in placement.
  * PEERS already placed elsewhere in the scene — each with id, prompt, bbox, proxy_shape, orientation, parent_id, and placement. Use these to ground placement decisions ("on the floor" means on the peer whose id matches the floor frame).

How to read a placement:
  * It describes WHERE this object sits relative to its `parent` and any entries in `referenced_ids`.
  * Phrases like "centered on the desk's top", "flush against the back wall mid-height", "hanging from the center of the ceiling", "to the left of the sofa", "between the table and the wall" should map to concrete coordinate choices.
  * Lean on `referenced_ids` kinds (ON → contact on target's exposed face, BESIDE → adjacent in X or Z, ABOVE/BELOW → higher/lower in Y, ATTACHED → flush touching, IN → contained inside the target's volume / footprint) to disambiguate ambiguous prose.

Canonical front view for all coordinates: +X right, +Y up, +Z front, -Z back. Right-handed, Y-up, meters. Treat the placement text's "front" as +Z, "left" as -X, "top" as +Y, etc.

AABB vs. actual geometry — an AABB describes each peer's EXTENT, NOT the shape of its surface. Each peer also carries a `proxy_shape` (BOX, SPHERE, CAPSULE, or HEMISPHERE) that IS the authoritative silhouette inside that AABB; the PROXY SHAPE section below gives the exact surface formula Y_top(x, z) for each. A dome-shaped island has proxy_shape=HEMISPHERE: its real surface dips from the AABB apex down to the AABB's bottom face at the footprint edge, NOT a flat top face.

There is no automatic correction. When a placement describes an object as resting on a non-BOX-proxy peer or sibling, YOU must compute the target's Y_top(x, z) from its AABB and proxy formula, and place the anchored object's bbox so its bottom face Y equals that value at the anchored object's XZ centre. For BOX-proxy targets (walls, floors, ceilings, generic slabs) this collapses to the familiar "bottom face at y_max".

Produce one assignment per object (id + bbox) such that:
  * Each bbox sits primarily inside the zone bbox, but MAY protrude modestly outside it when the object's prompt or placement narratively justifies overhang — e.g. a sign hanging on a building's exterior, projecting eaves, an awning, an antenna, a balcony railing, a sconce mounted on an outward face, a supporting pillar that descends below the zone's floor to ground against a peer below, a connective bridge / ramp / walkway that extends out toward another zone, or cantilevered structure that reaches across a gap. Protrusion should be limited to what the object's geometry actually demands, and must be treated carefully and checked against all surrounding zones and objects to ensure no clipping of objects that doesn't make sense. Frames / shell elements still lie fully inside the zone — only non-shell objects may protrude.
  * Every placement description is honored as faithfully as possible.
  * Dimensions are appropriate to each object's prompt (size a chair like a chair, a wall like a wall, a roof like a roof).
  * Avoid placing two clearly unrelated objects in the same XZ footprint when nothing about the scene justifies it (two trees stacked on the same spot). Some AABB overlap is fine and often unavoidable — curved ground meshes, parent containment, stacking — so treat non-overlap as a soft preference driven by physical plausibility, not a hard rule.

Because you are deciding the full layout at once, RESERVE SPACE for every object up front — if the zone needs walls AND a roof, the walls must stop short of the ceiling so the roof has somewhere to sit.

Coordinates in meters, centimeter precision (multiples of 0.01). Signed `dimensions` from an `origin` vertex. Emit exactly one assignment per requested object id — no extras, no omissions.

<proxy_shape>
{PROXY_SHAPE_DOC}
</proxy_shape>

Respond with ONE JSON object matching the schema. No prose, no markdown, no code fences."""


def render_object_bbox_batch(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_bbox: BoundingBox,
    objects: list[ObjectSpec],
    peers: list[tuple[str, str, BoundingBox, str | None, ProxyShape | None, Orientation, str | None, str | None]],
) -> str:
    peer_lines = (
        "\n".join(
            f"  - {pid}: prompt={pprompt!r} bbox={pbbox.model_dump_json()} proxy_shape={_render_proxy_shape(pproxy)} orientation={porient}deg parent={pparent!r} placement={pplacement!r}"
            for pid, pprompt, pbbox, pparent, pproxy, porient, pplacement, _pplan in peers
        )
        if peers
        else "  (none)"
    )
    object_lines = "\n\n".join(
        f"""  - id={o.id!r}
    parent: {o.parent!r}
    parent_kind: {o.parent_kind.value}
    prompt: {o.prompt}
    proxy_shape: {_render_proxy_shape(o.proxy_shape)}
    orientation: {o.orientation}deg
    placement: {o.placement}
    referenced_ids: {_render_relationships(o.referenced_ids)}"""
        for o in objects
    )
    return f"""Zone id: {zone_id!r}
Zone prompt: {zone_prompt!r}
Zone bbox: {zone_bbox.model_dump_json()}

Objects to place ({len(objects)}):
{object_lines}

Peers already placed in the run:
{peer_lines}

Produce a bbox for every object in a single coherent layout."""


# ---------- Step 7: iterative next-object decision --------------------------


class NextObjectOutput(BaseModel):
    done: bool
    object: ObjectSpec | None = None


SYSTEM_NEXT_OBJECT = f"""You are iteratively refining a 3D scene zone inside StarshotBench — a head-to-head competitive benchmark where your scene is rendered and judged against another LLM's rendering of the same user prompt. The zone's defining anchor objects are already placed. You are being asked a single question: is ONE MORE object needed to make this zone read as complete, or is it already right?

The threshold between "rich" and "busy" is the judgment call that costs runs. Stopping too early leaves the zone sparse and forgettable — the judge sees empty dead space and moves on. Adding too much turns the zone into incoherent clutter — the judge sees noise and can't find the focal point. A masterful build knows when to stop.

Err on the side of `done = true`. Prefer "this zone has what it needs" over adding clutter. Only add another object if there is a clearly missing element a viewer of the final render would notice was absent.

If `done = true`, leave `object` null and stop.

If `done = false`, emit EXACTLY ONE object. Make it COUNT — not decorative filler, something that noticeably improves the zone's legibility or character. Same rules as the bulk decomposition step:
  * Unique `id` (not colliding with any existing node in the scene).
  * `prompt` — a detailed description; used verbatim for text-to-3D.
  * `parent` — id of the structural anchor (the thing this object rests on, hangs from, leans against, or is contained by). Use the zone id only if the object truly has no physical supporter (e.g. a floating cloud).
  * `parent_kind` — how this object anchors to its `parent`. MUST be one of `ON` (rests on parent's outward surface), `ATTACHED` (flush against any face of the parent), or `IN` (contained inside the parent's volume / footprint with no specific contact face). `BESIDE`, `ABOVE`, `BELOW` are NOT valid — they describe non-contact peer hints and only appear in `referenced_ids`.
  * `orientation` — world-frame yaw about +Y in degrees. MUST be one of: -180, -135, -90, -45, 0, 45, 90, 135, 180. The mesh comes back with its front along world +Z; orientation rotates it into the pose you intend. `0` = front faces +Z (toward viewer), `90` = front faces -X, `180` = front faces -Z (away), `-90` = front faces +X. Pick a non-zero value when the object has a clear "front" that should face a specific direction; use 0 for symmetric objects.
  * `placement` — a prose description of WHERE this object sits within / on / against its `parent`, plus any alignment to other already-placed objects or features.
  * `referenced_ids` — OPTIONAL list of `{{target, kind}}` entries when the placement text refers to nodes other than the parent. `kind` is one of ON / BESIDE / ABOVE / BELOW / ATTACHED / IN. Don't include the parent here. Empty list is fine.

GROUND-AWARENESS RULE. If this zone already has a GROUND / SHELL peer placed (a mesh describing terrain or enclosure shape — an island dome, a crater bowl, a hill, a curved floor, the room's walls and floor), any new object whose physical support is that terrain/floor MUST set `parent` to that peer's id (NOT the zone id), and its `placement` must describe how it rests on that surface. The peer's `proxy_shape` in the current scene is authoritative for its surface geometry — a HEMISPHERE peer is a dome, and the downstream bbox-resolution step will compute the dome's surface height at the new object's XZ from the peer's proxy formula and rest the object on that surface. You are not placing bboxes at this step; just pick the right parent. Only use the zone id as `parent` for objects floating in the zone rather than anchored to a specific surface.

You MAY emit `proxy_shape` on the new object if its silhouette is non-rectilinear (SPHERE for a boulder, CAPSULE for a tree trunk, HEMISPHERE for a mound) — omit it otherwise. See the emitter's decomposition schema for the full vocabulary; the value set is the same.

<no_ephemera>
{NO_EPHEMERA_DOC}
</no_ephemera>

Respond with ONE JSON object matching the schema. No prose, no markdown, no code fences."""


class ImagePromptOutput(BaseModel):
    prompt: str


SYSTEM_IMAGE_PROMPT = """You produce ONE short noun phrase naming the object to render. The phrase is dropped verbatim into a fixed image-prompt template — used to generate THREE orthographic reference images of the object (front, side, and top) that all feed into a multi-image-to-3D reconstructor. Your phrase is reused across all three views, so it must describe the object intrinsically — never reference a specific viewing angle, silhouette, or projection. Your output is the SUBJECT of the sentence and nothing more.

The user message will show you the EXACT wrappers your phrase is slotted into (one per view), with a `<<<SUBJECT>>>` marker where your output goes. Read them before writing. Anything the wrappers already say — the orthographic view words, the white background, "capture the entire model", the explicit object dimensions — must NOT appear in your phrase, or it will read twice. You should, however, use descriptive language, e.g. "tall" or "wide".

Inputs you receive:
  * The original object prompt.
  * The bounding box dimensions in meters (width +X, height +Y, depth +Z) — use them to pick proportion-sensitive adjectives ("long", "tall", "squat") only when natural. The wrapper also emits the exact dimensions to the renderer, so you do NOT need to encode them in your phrase.
  * The proxy_shape — already encoded into the wrapper's hitbox language; don't repeat it in your phrase.
  * The full image-prompt template with the `<<<SUBJECT>>>` slot visible.
  * The chronological list of prior subject phrases already submitted in this scene — borrow material vocabulary, palette, and stylistic register so the asset coheres with what came before. Do not copy verbatim.

Rules for the phrase:
  * 5-15 words, lower-case, no trailing period.
  * Names the object directly with its defining attributes (species, material, colour, weathering, character). E.g. "a weathered cypress log with bleached bark and patches of moss", "a sleek matte-black office chair with chromed swivel base", "a low domed mossy island with thin muddy lip".
  * NO scene context: drop "half-submerged in", "surrounded by", "resting in", "on the floor of", "nestled among". Keep only intrinsic features of the object itself.
  * NO camera, framing, backdrop, lighting, or rendering instructions — the wrapper handles all of that.

Respond with ONE JSON object matching the schema. The `prompt` field holds the noun phrase only."""


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
        else " A reference image showing the orthographic FRONT view of THE SAME object is provided — preserve its silhouette, proportions, materials, colour, and surface detail exactly. Only the camera angle changes; do not reinterpret the object's identity."
    )
    base = f"Generate a direct, perfect {view_phrase} of {description} that roughly can be captured within {_article(hitbox)} {hitbox} hitbox without bending or deforming the object's natural proportions. The object should not fully be in {_article(silhouette)} {silhouette} shape unless its dimensions and nature dictate it is naturally that shape. Prioritize realism over confinement to the hitbox shape.{reference_clause}"
    if dimensions is None:
        return base
    w, h, d = dimensions
    return f"{base} The object's dimensions are exactly {w:.2f}m by {h:.2f}m by {d:.2f}m (width by height by depth).Capture the entire model in the image. Render against a clean, empty white background with no other objects, dimension markings, or graphics."


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
        prior_block = "Prior subject phrases in this scene: (none — this is the first object; you are setting the aesthetic baseline)."
    return f"""Original object prompt: {prompt!r}
Bounding box dimensions: width={w:.2f}m, height={h:.2f}m, depth={d:.2f}m
Proxy shape: {_render_proxy_shape(proxy_shape)}

Image-prompt templates your phrase will be slotted into (`{_SUBJECT_SLOT}` is your output — same phrase, three views):
  FRONT: {front_preview}
  SIDE:  {side_preview}
  TOP:   {top_preview}

{prior_block}

Produce ONE short noun phrase naming the subject."""


def render_next_object(
    *,
    zone_id: str,
    zone_plan: str | None,
    zone_bbox: BoundingBox,
    ancestors: list[tuple[str, str, str, BoundingBox, str | None]],
    scene: list[tuple[str, str, BoundingBox, str | None, ProxyShape | None, Orientation, str | None, str | None]],
    prior_attempts: list[tuple[ObjectSpec, str]] | None = None,
) -> str:
    ancestor_block = _render_ancestor_block(ancestors)
    zone_plan_block = _render_zone_plan_block(zone_plan)
    scene_block = _render_scene_lines(scene)
    if prior_attempts:
        attempt_lines = "\n".join(
            f"  attempt {i}: emitted {spec.model_dump_json()}\n             rejected: {reason}"
            for i, (spec, reason) in enumerate(prior_attempts)
        )
        retry_block = f"""

PRIOR ATTEMPTS — every object spec below was ALREADY rejected. Do NOT re-emit the same spec, and do not repeat the same structural mistake. Treat every listed reason as a hard constraint you must satisfy this time:
{attempt_lines}

Either emit a NEW ObjectSpec that fixes every listed reason, or set done=true. If you emit an object, its `referenced_ids` list must be non-empty and its first entry must be the object's structural parent (the supporter or containing zone)."""
    else:
        retry_block = ""
    return f"""<scene_context>
{_SCENE_CONTEXT_INTRO}

ZONE_ID: {zone_id!r}
Zone bbox: {zone_bbox.model_dump_json()}
{zone_plan_block}

{ancestor_block}

{scene_block}
</scene_context>

Decide whether another object is needed in this zone. If yes, emit exactly one ObjectSpec; otherwise set done=true.{retry_block}"""
