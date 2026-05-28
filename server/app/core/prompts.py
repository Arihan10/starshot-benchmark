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
</intro>

<role>
You are authoring the top-level plan for the scene from the user prompt, and deciding whether it is a single cohesive region or should decompose into distinct zones.
</role>

<input>
The user message contains the user prompt for the scene, plus guidance on how to author the plan and how to decide `is_atomic`.
</input>

<output>
Respond with a single JSON object containing:
- `plan` (string): your scene planning paragraph
- `is_atomic` (boolean): whether this scene is a single cohesive region or should decompose into distinct zones

No additional prose, markdown, or code fences.
</output>"""


SYSTEM_ZONE_PLAN = """<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.
</intro>

<role>
You are authoring the plan for ONE region within a larger scene, and deciding whether that region is a single cohesive area or should decompose further into distinct subzones.
</role>

<input>
The user message contains this region's seed prompt, the ancestor chain of regions above it (with their plans), the scene context already in the run, and guidance on how to author the plan and how to decide `is_atomic`.
</input>

<output>
Respond with a single JSON object containing:
- `plan` (string): your region planning paragraph
- `is_atomic` (boolean): whether this region is a single cohesive area or should decompose into distinct subzones

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
        return f"""you are the first step in the SpatialBench pipeline and the step responsible for determining the high-level description and direction of the overall scene. write one paragraph that describes a 3D scene imagined from the following prompt, and decide whether this scene should decompose into multiple distinct zones.

"{zone_prompt}"

<VERY IMPORTANT INSTRUCTIONS>
think deeply about the 3D scene, environment or level you want to build from this provided prompt, and how you can creatively make it stand out enough to WIN against the other submissions. think about the narrative through-line that will help guide and form realistic scene intention - what is this game for, who lives in this house, what is the player trying to achieve in this level, what kind of city is this, and so on. instead of a stoic description that focuses only on the architectural qualities and layout of the scene, the plan you describe should also introduce a high-level story to the scene while allowing downstream planning steps to build on the narrative further. your goal is to write a guiding plan that downstream steps can build upon to complete a cohesive 3D scene for the given prompt that follows the narrative you imagine.

write directly and consider every part carefully. you are only the first, overall planning step - your plan will go through hundreds of further downstream steps where it is expanded on and transformed as the AI pipeline to construct it propagates further planning by depth. define the scene itself, its top-level shape and character enough that the downstream steps have agency over their individual sections while also forming ideas of what to build. 

only the final output of the 3D geometry for the scene itself will be judged once the pipeline is finished; your prompt itself will NEVER be shown to the judges, it will only serve as a base to build upon. 

DO NOT be overly specific - remember, your prompt will NOT be converted directly into 3D geometry, it will undergo hundreds of expansion and detail steps before reaching any generation steps, so structure your output such that it is a base that the downstream tree of pipeline steps can build upon it. given a prompt for a building, a bad output provides exact instruction on what it looks like; a good prompt defines the narrative premise for the scene, the scope of the environment, the building's character and type, the surrounding environment, points that may implicitly be expanded, and a top-level shape for the scene itself without explicitly shaping the entities that form it.

plan differently based on the prompt given and infer the purpose - e.g for a house, you might plan the aforementioned details for the overall scene scope, general architectural, narrative and character; for a super mario platformer level, you might focus on the narrative section, features, progression, zones, mechanics, etc.; for a top-down swamp frogger level without a specific game mentioned, you might focus on first building out the game's premise internally given the more abstract request, and then establish the world, general layout, character, mechanics, scope, item types, objectives, etc. 

remember, tune specificity based on whether your intent can be inferred by downstream steps. e.g in the super mario level, do not be overly specific - do not scope out all individual platforms, items, etc. but rather the general idea of each part, since the downstream steps have a shared understanding of what a super mario level looks like and the general premise of the game; however, for a more abstract request like the top-down swamp frogger level, the through-line of what you are trying to build cannot be inferred or reconstructed by downstream steps in the pipeline as the specific context for the premise of the game was constructed within your internal reasoning, not exposed to those steps, and thus will be lost as the pipeline propagates, placing the onus for world creation and high-level planning on downstream steps (e.g the immediate next step of planning the specific nature of zones inferred from your prompt, which was not designed for deciding scene structure/mechanics itself as it lacks the lack full frame and is more there to decompose the scene and figure out spatial relationships between the decomposed zones, and follows a different, more mechanical heuristic for generation, which means it would not only not spend a lot of time thinking about that, but diverge significantly and perhaps genericly from world you were trying to create in different directions, before handing off to individual planning steps for each zone that have more agency over their nature and so on), which means you would need to provide that through-line more explicitly in that scenario where foundational planning is required for the world state due to a lack of shared context (as opposed to a house or established game level).

the output plan should be literal - do not use flowery language. do not describe any abstract quantities like mood, lighting, fog, etc unless they can be converted into concrete 3D geometry. do not reference meta-quantities like the pipeline itself, the scene's 3D nature itself, etc. NEVER MENTION THOSE THINGS. focus on defining the environment intrinsically. remember, this is a full 3D scene, NOT an image - do not define any specific perspective.

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
    return f"""You are the step in the SpatialBench pipeline responsible for planning out a particular region within the larger overall scene. write one paragraph that describes the plan for the following region, and decide whether it should decompose into multiple distinct subzones.

"{zone_prompt}"

The following is the ancestor chain for the current region:

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


SYSTEM_OVERALL_BBOX = """<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.
</intro>

<role>
You are sizing the overall bounding box for the scene — the abstract world canvas every zone and object will be placed inside.
</role>

<input>
The user message contains the user prompt and the scene plan authored upstream.
</input>

<output>
Respond with a single JSON object matching the schema: an axis-aligned bounding box in meters, under the canonical front view (+X right, +Y up, +Z front, -Z back). Defined by an `origin` vertex and a signed `dimensions` vector `(dx, dy, dz)` from that vertex; the sign of each component chooses the direction of expansion along that axis. Coordinates to centimeter precision (multiples of 0.01).

No prose, no markdown, no code fences.
</output>"""


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
</intro>

<role>
You are deciding the structural decomposition of one region of the scene — how it splits into its top-level subzones. Each subzone you emit becomes its own planning and recursive decomposition branch downstream.
</role>

<input>
The user message contains this zone's id, prompt, bbox, and plan, plus the scene context (ancestor chain and every other zone/object already placed in the run, which you may reference by id).
</input>

<output>
Respond with a single JSON object containing:
- `children` (list): the subzones this region decomposes into. Each child has:
  - `id` (string): unique within the entire scene
  - `prompt` (string): a short seed describing what this child zone is
  - `parent` (string): the id of this child's structural parent — the containing zone (PARENT_ID for a top-level subzone), or an earlier sibling in this call whose interior/footprint this child sits within
  - `parent_kind` (string): how this child anchors to its `parent`. Exactly one of `ON` (rests on parent's outward surface), `ATTACHED` (flush against any face of the parent), or `IN` (contained inside the parent's volume / footprint). `BESIDE` / `ABOVE` / `BELOW` are NOT valid here — they are peer hints, reserved for `referenced_ids`.
  - `placement` (string): prose describing WHERE this child sits within / against / relative to its parent and any referenced peers. The bbox-resolution step uses this verbatim.
  - `referenced_ids` (list of {target, kind}): OPTIONAL secondary relationships to other already-placed nodes referenced in the placement text. Each entry has a `target` (the peer's id) and a `kind` — one of ON, BESIDE, ABOVE, BELOW, ATTACHED, IN. Do NOT repeat the parent here. Empty list is fine when the placement only references the parent.
  - `proxy_shape` (string | null): BOX / SPHERE / CAPSULE / HEMISPHERE if the zone's silhouette is non-rectangular, otherwise null/omitted.

No additional prose, markdown, or code fences.
</output>"""


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
    return f"""You are the step in the SpatialBench pipeline responsible for breaking down a given area into its top-level subregions, enriching the scene's detailedness in terms of both narrative and physical architecture. Generate a list of subzones that should be present in the following scene, based on its description:

{zone_plan}

<IMPORTANT_INSTRUCTIONS>

<ZONE_SPLITTING_GUIDANCE>
A subzone is an area of spatial interest whose bounding box sits within the parent bounding box and can be treated individually as its own region due to a combination of physical and narrative reasons.

Subzones can keep decomposing into more zones recursively in subsequent passes, or end there as atomic leaves if that is appropriate. so always decompose at the TOP MOST LEVEL of the current zone — e.g. for a house scene with backyard, driveway, and house, do not skip straight to backyard-pool zone, backyard-grass zone, house-basement, house-first-floor, etc.; decompose into "the house", "the backyard", "the driveway" as top-level children, and let the next recursion split the house into floors and the backyard into pool and grass. the same principle holds everywhere: emit only the zones that exist at THIS level of the hierarchy, and trust the recursive planning + decompose passes underneath each of them to handle the next layer down.
</ZONE_SPLITTING_GUIDANCE>

Think very intricately and spatially about how this zone splits. Your goal is to reason a subzone decomposition layout that fits the narrative presented by the scene plan given above as well as the additional plans of ancestor scenes in the scene context section given below, while paying attention to the semantic relationships between the subzones. The subzones presented should each flesh out the guiding narrative further in some way, carrying relevant ideas from previously defined plans (in the scene context below) while also introducing some new ones without being contradictory.

The seed prompt you output for each subzone should be a 1-2 sentences long description that explains the subzone's shape, character, and the new narrative ideas presented by this subzone, if any. Be concrete about its description while leaving room for this prompt to be a seed for a more detailed plan. The prompt should be succinct without mentioning going overly into detail on the subzone's contents.

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


SYSTEM_ZONE_BBOX_BATCH = f"""<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.
</intro>

<role>
You are a constraint solver placing ALL sibling child zones inside a parent zone in one shot — deriving each child's axis-aligned bounding box from its placement prose, parent_kind, referenced_ids, and the parent's bbox.
</role>

<input>
The user message contains the parent zone's id and bbox, and a list of child specs to place. Each child has `id`, `prompt`, `proxy_shape`, `parent`, `parent_kind`, `placement`, and `referenced_ids`.
</input>

<output>
Respond with a single JSON object matching the schema: one `assignment` per child (id + bbox). Coordinates in meters under the canonical front view (+X right, +Y up, +Z front, -Z back), centimeter precision (multiples of 0.01). Use a signed `dimensions` vector from an `origin` vertex; sign chooses expansion direction along each axis. Emit exactly one assignment per requested child id — no extras, no omissions.

No prose, no markdown, no code fences.
</output>

<additional_context>
{PROXY_SHAPE_DOC}
</additional_context>"""


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


# Shared output schema + additional_context for the three object
# decomposition modes (anchor, encapsulating, negative-space). Each
# mode-specific system prompt concatenates its own intro/role/input
# with this tail.
_OBJECT_DECOMP_TAIL = f"""<output>
Respond with a single JSON object containing:
- `objects` (list): the new object specs this call adds to the scene. Each object spec:
  - `id` (string): unique within this call
  - `prompt` (string): detailed description; used verbatim as the text-to-3D generation prompt
  - `parent` (string): id of this object's structural parent (what it physically rests on, hangs from, leans against, or is contained by)
  - `parent_kind` (string): exactly one of `ON` (rests on parent's outward surface), `ATTACHED` (flush against any face of the parent — wall/ceiling mounts, embedded fittings, shell-frame-to-zone), or `IN` (contained inside the parent's volume / footprint with no specific contact face). `BESIDE` / `ABOVE` / `BELOW` are NOT valid here.
  - `proxy_shape` (string | null): BOX / SPHERE / CAPSULE / HEMISPHERE if the object's silhouette is non-rectilinear, otherwise null/omitted.
  - `orientation` (int): world-frame yaw about +Y in degrees. Exactly one of -180, -135, -90, -45, 0, 45, 90, 135, 180. `0` = front faces +Z (toward viewer), `90` = front faces -X, `180` = front faces -Z, `-90` = front faces +X. Use 0 for symmetric objects.
  - `placement` (string): prose describing WHERE this object sits — position within / on / against the parent, plus alignment to any referenced peers.
  - `referenced_ids` (list of {{target, kind}}): OPTIONAL secondary relationships when placement text refers to nodes other than the parent. Each entry: `target` (peer's id) and `kind` — one of ON, BESIDE, ABOVE, BELOW, ATTACHED, IN. Do NOT repeat the parent here. Empty list is fine.
- `bound_existing` (list): used only by the encapsulating step (anchor and negative-space leave this empty). Each entry has `plan_element` (the plan's noun phrase verbatim) and `peer_id` (the id of an existing node that already satisfies it).

No additional prose, markdown, or code fences.
</output>

<additional_context>
{PROXY_SHAPE_DOC}

{NO_EPHEMERA_DOC}
</additional_context>"""


SYSTEM_ANCHOR_DECOMP = f"""<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.
</intro>

<role>
You are enumerating the defining anchor objects of an atomic leaf zone — the objects that make this zone unmistakably what it is. A later iterative pass adds more objects one at a time on top of these anchors.
</role>

<input>
The user message contains this zone's id, bbox, and plan, plus the scene context (ancestor chain and every other zone/object already placed in the run, which you may reference by id).
</input>

{_OBJECT_DECOMP_TAIL}"""


SYSTEM_ENCAPSULATING_DECOMP = f"""<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.
</intro>

<role>
You are defining a list of objects that represent the perimeter of the given region, ONLY if needed.
</role>

<input>
The user message contains this zone's id, bbox, and plan, plus the scene context (ancestor chain and every other zone/object already placed in the run, which you may reference by id).
</input>

{_OBJECT_DECOMP_TAIL}"""


SYSTEM_NEGATIVE_SPACE_DECOMP = f"""<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.
</intro>

<role>
You are filling the ambient, connective, interstitial space between the named zones and objects of the scene — the layer of small instanced solids that binds the scene into a continuous world (lilypads across swamp water, grass tufts over a meadow, scattered stones across a plain). Every named zone has already been decomposed and populated; what remains is the layer the named zones do not own.
</role>

<input>
The user message contains this zone's id, bbox, and plan, plus the scene context (every zone and object already placed in the run, which you may reference by id).
</input>

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
    return f"""You are the step in the SpatialBench pipeline responsible for determining whether a perimeter is needed for the given region, and if so, what that perimeter is made up of. Not every zone needs a perimeter, decide whether it is absolutely required. If the latter, generate a list of bounding geometry elements that form a perimeter for the following region.

{_render_zone_plan_block(zone_plan)}

The list of objects should work together to form a cohesive perimeter or partial perimeter of any arbitrary shape. The purpose of this list of objects is to form a sense of boundary for the given region in every dimension that makes sense based on its plan - perimeter does not necessarily mean in the horizontal axis but in all possible directions, including the vertical direction (e.g. bases, covers). In this case, perimeter or boundary does not automatically imply physically bounding the region on all sides (though depending on the region's plan, that may be the case). You should think carefully and reason spatially about what objects should go in this list to form a well-defined, physically and narratively reasonable boundary for the zone.

Object should be individualistic - composite objects should be broken down into individual or partial objects (abstract fragments that are meant to combine into a more complex object) and placed accordingly, allowing for more granular control of the region's boundary. Objects and partial objects can be stacked, strung, pieced to form larger, cohesive sections for the perimeter. there is no limit on the number of objects in your output list - always prefer individual objects placed close to each other over a single composite object with a prompt to generate them together at once. if the region calls for it, we can have as high fidelity of a perimeter as we want. if a dense perimeter makes sense for the given region, then make it dense - as many objects as you see fit, their bounding boxes right next to each other. You are in control - do not rely on downstream generation steps to output composite geometry: organize your list of objects so that you are in direct control of positioning to form that composite geometry yourself using the individual partial objects.

When generating the list of objects, keep in mind traversal between various regions both horizontally and vertically that might require more complex boundary objects made up of partial objects, keep in mind the semantic meaning of objects that allow passage. Using the scene context provided, carefully determine if passage is needed from this region to another, and if so, what kind of partial objects would be needed to piece together the complicated shapes that would allow that traversal. A good example of this is a wall in a building that has a door embedded within: the wall would be made of multiple rectanglular regions that when pieced together form an arch looking shape, with a door object filling that space in. Apply this same idea of leaving empty gaps or embedding other objects wherever it makes sense to do so. Pay especial attention to the context provided in the plans of the other regions in the scene, and use it to imagine realistically navigating the region as part of the larger scene. Use this thinking to guide you in the generation and placement of your list of objects.

be wary of duplicate geometry - for two neighboring regions separated by some sort of divider, it is only necessary to generate the divider once. study the provided scene context to determine if generating something is necessary.

<scene_context>
{_SCENE_CONTEXT_INTRO}

ZONE_ID: {zone_id!r}
Zone bbox: {zone_bbox.model_dump_json()}
{_render_zone_plan_block(zone_plan)}

{_render_ancestor_block(ancestors)}

{_render_scene_lines(scene)}
</scene_context>

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


SYSTEM_OBJECT_BBOX_BATCH = f"""<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.
</intro>

<role>
You are a constraint solver placing ALL objects for a scene zone in one shot — deriving each object's axis-aligned bounding box from its placement prose, parent_kind, referenced_ids, the zone bbox, and peer geometry.
</role>

<input>
The user message contains the zone id/prompt/bbox, a list of objects to place (each with `id`, `prompt`, `proxy_shape`, `orientation`, `parent`, `parent_kind`, `placement`, `referenced_ids`), and a list of peers already placed elsewhere in the scene (each with id, prompt, bbox, proxy_shape, orientation, parent, placement).
</input>

<output>
Respond with a single JSON object matching the schema: one `assignment` per object (id + bbox). Coordinates in meters under the canonical front view (+X right, +Y up, +Z front, -Z back), centimeter precision (multiples of 0.01). Use a signed `dimensions` vector from an `origin` vertex; sign chooses expansion direction along each axis. Emit exactly one assignment per requested object id — no extras, no omissions.

No prose, no markdown, no code fences.
</output>

<additional_context>
{PROXY_SHAPE_DOC}
</additional_context>"""


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


SYSTEM_NEXT_OBJECT = f"""<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.
</intro>

<role>
You are iteratively refining a 3D scene zone whose defining anchor objects are already placed. Decide whether ONE more object would make this zone read as complete, or whether the zone is already right. When you emit an object, emit exactly one.
</role>

<input>
The user message contains this zone's id, bbox, and plan, plus the scene context (every zone and object already placed in the run, which you may reference by id).
</input>

<output>
Respond with a single JSON object containing:
- `done` (boolean): true when no further object is needed
- `object` (object spec | null): when `done` is false, exactly one new object spec; otherwise null.

Each object spec has the same fields as the bulk decomposition step:
  - `id` (string): unique, not colliding with any existing node in the scene
  - `prompt` (string): detailed description; used verbatim for text-to-3D
  - `parent` (string): id of the structural anchor (what this object physically rests on, hangs from, leans against, or is contained by)
  - `parent_kind` (string): exactly one of `ON` (rests on parent's outward surface), `ATTACHED` (flush against any face of the parent), or `IN` (contained inside the parent's volume / footprint). `BESIDE` / `ABOVE` / `BELOW` are NOT valid here.
  - `proxy_shape` (string | null): BOX / SPHERE / CAPSULE / HEMISPHERE if the object's silhouette is non-rectilinear, otherwise null/omitted.
  - `orientation` (int): world-frame yaw about +Y in degrees. Exactly one of -180, -135, -90, -45, 0, 45, 90, 135, 180. `0` = front faces +Z, `90` = front faces -X, `180` = front faces -Z, `-90` = front faces +X. Use 0 for symmetric objects.
  - `placement` (string): prose describing WHERE this object sits within / on / against its parent, plus alignment to any referenced peers.
  - `referenced_ids` (list of {{target, kind}}): OPTIONAL secondary relationships when the placement text refers to nodes other than the parent. Each entry: `target` (peer's id) and `kind` — one of ON, BESIDE, ABOVE, BELOW, ATTACHED, IN. Do NOT repeat the parent here. Empty list is fine.

No additional prose, markdown, or code fences.
</output>

<additional_context>
{NO_EPHEMERA_DOC}
</additional_context>"""


class ImagePromptOutput(BaseModel):
    prompt: str


SYSTEM_IMAGE_PROMPT = """<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.
</intro>

<role>
You produce ONE short noun phrase naming the object to render. The phrase is dropped verbatim into a fixed image-prompt template that generates three orthographic reference views (front, side, top) of the object for a downstream multi-image-to-3D reconstructor. The phrase describes the object intrinsically; the wrapper handles view, framing, background, and dimensions.
</role>

<input>
The user message contains the original object prompt, the bounding box dimensions in meters, the proxy_shape, the exact wrapper templates with a `<<<SUBJECT>>>` slot showing where your phrase goes, and the chronological list of prior subject phrases already submitted in this scene.
</input>

<output>
Respond with a single JSON object containing:
- `prompt` (string): the noun phrase. 5-15 words, lower-case, no trailing period. Names the object directly with its defining attributes (material, colour, weathering, character).

No additional prose, markdown, or code fences.
</output>"""


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


ImageView = Literal["front", "side", "top", "three-quarter"]


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
        "three-quarter": "3/4 perspective view (camera positioned above and to the right at roughly 30-45 degrees, showing the front face, top face, and one side face simultaneously)",
    }[view]
    reference_clause = (
        ""
        if view in ("front", "three-quarter")
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
        f"{reference_clause} "
        "Capture the entire model in the image. Render against a "
        "clean, empty white background with no other objects, dimension markings, or graphics."
    )
    if dimensions is None:
        return base
    w, h, d = dimensions
    return (
        f"{base} The object's dimensions are exactly "
        f"{w:.2f}m by {h:.2f}m by {d:.2f}m (width by height by depth)."
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
