"""Prompts and structured-output schemas for LLM calls."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.core import util
from app.core.types import (
    BoundingBox,
    Node,
    Orientation,
    ParentRelationshipKind,
    ProxyShape,
    Relationship,
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

DEEPSEEK_INJECTION = """
<IMPORTANT_THINKING>
Be very opinionated in your thinking - do not go around in circles. Once a conclusion is reached, stick to it. Aim to end reasoning as soon as a concrete plan has been formed.
</IMPORTANT_THINKING>
"""


def _deepseek_suffix() -> str:
    # DeepSeek's reasoning tends to spiral; the injected blurb pins it
    # to a concrete plan. Local import avoids a cycle at module load.
    from app.services.llm import _current_model

    model = _current_model.get()
    if model and ("deepseek" in model.lower() or "gpt" in model.lower()):
        return DEEPSEEK_INJECTION
    return ""


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


# --- canonical scene-context tree --------------------------------------------
#
# Every prompt that shows the LLM "what does the scene look like right now" routes through one of the two renderers below.
# `render_scene_tree` keeps the object detail in a second pass grouped by region; `render_scene_tree_embedded` inlines each region's objects under it.
# Both share the entry/formatting helpers here and the type-agnostic utilities in `util`.

_NO_NODES_MESSAGE = "(no regions or objects have been placed yet — this is the very start of the run)"
_NO_SUBREGIONS_MESSAGE = "{(none - no other subregions have been planned yet)}"

# Inline marker appended to a targeted subregion's name line in the
# embedded block. The arrow points back at the node id and labels it as
# the target so a prompt can call the LLM's attention to one zone; the
# caller-supplied text follows it.
_TARGET_MARKER = "<-- TARGET:"


def _root_scene_header(root: Node) -> str:
    """Root prompt, plan, and overall bounding box — injected at the top of every prompt that shows scene context."""
    return (
        f"Prompt: {root.prompt}\n"
        f"Plan: {root.plan}\n"
        f"Overall scene bounding box coordinates: {util.format_global_bbox(root.bbox)}"
    )


def _local_coords_line(node: Node, by_id: dict[str, Node]) -> str | None:
    """`Local coordinates relative to its parent (<pid>): ...` line, or None for the root / a node whose parent is absent from the snapshot."""
    if node.parent_id is not None and node.parent_id in by_id:
        coords = util.format_local_bbox(node.bbox, by_id[node.parent_id].bbox)
        return f"Local coordinates relative to its parent ({node.parent_id}): {coords}"
    return None


def _object_entry(obj: Node, by_id: dict[str, Node]) -> str:
    """Full-detail entry for one concrete object (no plan)."""
    lines = [
        f"Name: {obj.id}",
        f"Description: {obj.prompt}",
    ]
    if obj.parent_id is not None:
        kind_str = obj.parent_kind.value if obj.parent_kind is not None else "(unknown)"
        lines.append(f"parent: {obj.parent_id}  kind: {kind_str}")
    if obj.placement is not None:
        lines.append(f"placement: {obj.placement}")
    if obj.referenced_ids:
        refs = ", ".join(f"{r.target}: {r.kind.value}" for r in obj.referenced_ids)
        lines.append(f"referenced_ids: [{refs}]")
    else:
        lines.append("referenced_ids: []")
    lines.append(f"Global bounding box coordinates: {util.format_global_bbox(obj.bbox)}")
    local = _local_coords_line(obj, by_id)
    if local is not None:
        lines.append(local)
    return util.braces("\n".join(lines))


def _region_plan_entry(
    region: Node,
    idx: dict[str | None, list[Node]],
    by_id: dict[str, Node],
    target_id: str | None = None,
    target_text: str = "",
) -> str:
    """Separate-objects format, pass 1: a subregion's fields, the names of its objects, and (recursively) its nested subregions. No object detail here. When `target_id` matches this region (at any depth), an inline marker carrying `target_text` is appended to its name line so a prompt can point the LLM at this one zone."""
    objects, subregions = util.split_region_members(region.id, idx)
    name_line = f"Name: {region.id}"
    if target_id is not None and region.id == target_id:
        name_line += f"   {_TARGET_MARKER} {target_text}".rstrip()
    lines = [
        name_line,
        f"Description: {region.prompt}",
    ]
    if region.plan is not None:
        lines.append(f"Plan for this region: {region.plan}")
    lines.append(f"Global bounding box coordinates: {util.format_global_bbox(region.bbox)}")
    local = _local_coords_line(region, by_id)
    if local is not None:
        lines.append(local)
    lines.append(f"Objects: {', '.join(o.id for o in objects) if objects else '(none)'}")
    if subregions:
        lines += [
            "",
            "This subregion decomposes into the following further subregions.",
            "",
            util.brace_group(
                [_region_plan_entry(s, idx, by_id, target_id, target_text) for s in subregions]
            ),
        ]
    return util.braces("\n".join(lines))


def _region_objects_entry(region: Node, idx: dict[str | None, list[Node]], by_id: dict[str, Node]) -> str:
    """Separate-objects format, pass 2: the full detail of a subregion's objects and (recursively) the same for its nested subregions."""
    objects, subregions = util.split_region_members(region.id, idx)
    lines = [
        f"Subregion name: {region.id}",
        "",
        "Here's the list of objects that have been placed for this subregion.",
        "",
        util.brace_group([_object_entry(o, by_id) for o in objects]),
    ]
    if subregions:
        lines += [
            "",
            "This subregion's further subregions also have their own objects. Here's a list of further subregions.",
            "",
            util.brace_group([_region_objects_entry(s, idx, by_id) for s in subregions]),
        ]
    return util.braces("\n".join(lines))


def _region_embedded_entry(
    region: Node,
    idx: dict[str | None, list[Node]],
    by_id: dict[str, Node],
    target_id: str | None = None,
    target_text: str = "",
) -> str:
    """Embedded format: a subregion's fields, its objects inline, then (recursively) its nested subregions. When `target_id` matches this region (at any depth), an inline marker carrying `target_text` is appended to its name line so a prompt can point the LLM at this one zone."""
    objects, subregions = util.split_region_members(region.id, idx)
    name_line = f"Subregion name: {region.id}"
    if target_id is not None and region.id == target_id:
        name_line += f"   {_TARGET_MARKER} {target_text}".rstrip()
    lines = [
        name_line,
        f"Description: {region.prompt}",
    ]
    if region.plan is not None:
        lines.append(f"Plan for this region: {region.plan}")
    lines.append(f"Global bounding box coordinates: {util.format_global_bbox(region.bbox)}")
    local = _local_coords_line(region, by_id)
    if local is not None:
        lines.append(local)
    lines += [
        "",
        "Here's the list of objects that have been placed for this subregion.",
        "",
        util.brace_group([_object_entry(o, by_id) for o in objects]),
    ]
    if subregions:
        lines += [
            "",
            "Here's the list of subregions that are present within this region.",
            "",
            util.brace_group(
                [_region_embedded_entry(s, idx, by_id, target_id, target_text) for s in subregions]
            ),
        ]
    return util.braces("\n".join(lines))


def _render_to_place_block(
    to_place: list[ChildNodeSpec] | list[ObjectSpec] | None,
    by_id: dict[str, Node],
) -> str:
    """Trailing block listing the children/objects whose bboxes a bbox-batch step must determine. Empty string when there is nothing to place."""
    if not to_place:
        return ""
    kind = "objects" if isinstance(to_place[0], ObjectSpec) else "sub-regions"
    to_place_ids = {c.id for c in to_place}
    entries: list[str] = []
    for c in to_place:
        kind_str = c.parent_kind.value
        if c.parent in by_id:
            pdims = by_id[c.parent].bbox.size
            pdims_str = f"[{pdims[0]:.2f}, {pdims[1]:.2f}, {pdims[2]:.2f}]"
        elif c.parent in to_place_ids:
            pdims_str = "(parent is also being placed in this batch — use your emitted dimensions for it)"
        else:
            pdims_str = "(parent id not recognised in current scene)"
        lines = [
            f"id: {c.id}",
            f"parent: {c.parent}  kind: {kind_str}",
            f"parent_dimensions: {pdims_str}",
            f"proxy_shape: {_render_proxy_shape(c.proxy_shape)}",
        ]
        if c.orientation:
            lines.append(f"orientation: {c.orientation}deg")
        lines.append(f"prompt: {c.prompt}")
        lines.append(f"placement: {c.placement}")
        if c.referenced_ids:
            refs = ", ".join(f"{r.target}: {r.kind.value}" for r in c.referenced_ids)
            lines.append(f"referenced_ids: [{refs}]")
        else:
            lines.append("referenced_ids: []")
        entries.append(util.braces("\n".join(lines)))
    return (
        f"\n\nHere's the list of {kind} to place (bbox is yours to determine for each):\n\n"
        + util.brace_group(entries)
    )


def render_subregions_block(
    nodes: list[Node],
    *,
    node_id: str | None = None,
    text: str = "",
) -> str:
    """Pseudo-JSON block of the scene's top-level subregions in the separate-objects format: each carries its plan, bbox, and object names, recursing into nested subregions. Renders the single-region placeholder when the scene is one undivided region.

    Pass `node_id` to point the LLM at one specific zone: the subregion whose id matches gets an inline target marker carrying `text` appended to its name line, found at any depth of the tree. With `node_id` unset (the default) the block renders exactly as before."""
    root = util.find_root(nodes)
    if root is None:
        return _NO_SUBREGIONS_MESSAGE
    by_id = {n.id: n for n in nodes}
    idx = util.index_children(nodes)
    _, subregions = util.split_region_members(root.id, idx)
    if not subregions:
        return _NO_SUBREGIONS_MESSAGE
    return util.brace_group(
        [_region_plan_entry(s, idx, by_id, node_id, text) for s in subregions]
    )


def render_root_objects_block(nodes: list[Node]) -> str:
    """Pseudo-JSON block of the objects parented directly to the scene root, each in full detail. Renders an empty `{}` block when the root has no direct objects."""
    root = util.find_root(nodes)
    if root is None:
        return util.brace_group([])
    by_id = {n.id: n for n in nodes}
    idx = util.index_children(nodes)
    objects, _ = util.split_region_members(root.id, idx)
    return util.brace_group([_object_entry(o, by_id) for o in objects])


def render_filled_block(nodes: list[Node]) -> str:
    """Pseudo-JSON block giving the full object detail of every top-level subregion (and its nested subregions) — the separate-objects format's second pass. Renders an empty `{}` block when the scene has no subregions."""
    root = util.find_root(nodes)
    if root is None:
        return util.brace_group([])
    by_id = {n.id: n for n in nodes}
    idx = util.index_children(nodes)
    _, subregions = util.split_region_members(root.id, idx)
    return util.brace_group([_region_objects_entry(s, idx, by_id) for s in subregions])


def render_embedded_block(
    nodes: list[Node],
    *,
    node_id: str | None = None,
    text: str = "",
) -> str:
    """Pseudo-JSON block of the scene's top-level subregions in the embedded format: each carries its objects inline, followed by its nested subregions. Renders the single-region placeholder when the scene has no subregions.

    Objects parented directly to the scene root (e.g. the shell/ground meshes from the root's encapsulating pass) are appended in a trailing section whenever the root owns any — the embedded walk starts at the root's subregions and would otherwise drop them, leaving downstream steps (bbox resolution, decomposition) blind to geometry their children anchor against.

    Pass `node_id` to point the LLM at one specific zone: the subregion whose id matches gets an inline target marker carrying `text` appended to its name line, found at any depth of the embedded tree. With `node_id` unset (the default) the block renders exactly as before."""
    root = util.find_root(nodes)
    if root is None:
        return _NO_SUBREGIONS_MESSAGE
    by_id = {n.id: n for n in nodes}
    idx = util.index_children(nodes)
    root_objects, subregions = util.split_region_members(root.id, idx)
    if subregions:
        block = util.brace_group(
            [_region_embedded_entry(s, idx, by_id, node_id, text) for s in subregions]
        )
    else:
        block = _NO_SUBREGIONS_MESSAGE
    if root_objects:
        block += (
            "\n\nHere's the list of objects parented directly to the overall scene root:\n\n"
            + util.brace_group([_object_entry(o, by_id) for o in root_objects])
        )
    return block


def render_scene_tree(
    *,
    nodes: list[Node],
    to_place: list[ChildNodeSpec] | list[ObjectSpec] | None = None,
) -> str:
    """Render the scene-context tree in the SEPARATE-OBJECTS format: each subregion lists only its object names, and every object's full detail is rendered in a second pass grouped by region. `render_scene_tree_embedded` renders the alternative EMBEDDED format. Both formats are specified below (the second is under the "EMBEDDED OBJECTS IN ZONE LIST ver." divider).
    """
    if not nodes:
        return _NO_NODES_MESSAGE
    root = util.find_root(nodes)
    if root is None:
        return _NO_NODES_MESSAGE
    by_id = {n.id: n for n in nodes}
    _, top_subregions = util.split_region_members(root.id, util.index_children(nodes))

    body = f"""This is the overall plan for the entire scene.

{_root_scene_header(root)}

Each scene is always subdivided into a set of subregions. Each subregion can contain further subregions inside or the set of objects that forms it.

Here's the list of subregions that have been planned for this scene so far. Each subregion has a plan for how it should be built and a bounding box that defines its global position in the scene, given as a 3D coordinate marking one corner and a 3D vector that marks the opposite corner. Additionally, each subregion will also have a set of local coordinates that define its position relative to its parent region, where the origin is the actual minimum corner of the parent's bounding box.

{render_subregions_block(nodes)}

Each region is being filled with its respective objects. Each object has a description detailing what it is and a bounding box that defines its global position in the scene. Additionally, each object will also have a set of local coordinates that define its position relative to its parent, which can either be another object or the region it belongs to itself.

Here's a list of objects that are parented to the global scene itself.

{render_root_objects_block(nodes)}"""

    if top_subregions:
        body += f"""

Here's the list of subregions that have already been filled with their respective objects.

{render_filled_block(nodes)}"""

    return body + _render_to_place_block(to_place, by_id)


def render_scene_tree_embedded(
    *,
    nodes: list[Node],
    to_place: list[ChildNodeSpec] | list[ObjectSpec] | None = None,
) -> str:
    """Render the scene-context tree in the EMBEDDED format: every subregion carries the full detail of its own objects inline, immediately followed by its nested subregions. This is the "EMBEDDED OBJECTS IN ZONE LIST ver." variant specified in `render_scene_tree`'s docstring; `render_scene_tree` renders the separate-objects variant. Identical signature, so the two are drop-in interchangeable at every call site."""
    if not nodes:
        return _NO_NODES_MESSAGE
    root = util.find_root(nodes)
    if root is None:
        return _NO_NODES_MESSAGE
    by_id = {n.id: n for n in nodes}

    body = f"""This is the overall plan for the entire scene.

{_root_scene_header(root)}

Each scene is always subdivided into a set of subregions. Each subregion can contain further subregions inside or the set of objects that forms it. The scene is composed as a tree with every object or region parented to another object or region.

Here's the list of subregions that have been planned for this scene so far, followed by any objects parented directly to the scene root. Each subregion has a plan for how it should be built, a bounding box that defines its global position in the scene, given as a 3D coordinate marking one corner and a 3D dimensions vector that marks the opposite corner, as well as a list of objects present in that subregion (which each come with their own description and bounding boxes). Additionally, each subregion and object mentioned will also have a set of local coordinates that define its position relative to its parent (which can be either another region or another object), where the origin is the actual minimum corner of the parent's bounding box.

{render_embedded_block(nodes)}"""
    return body + _render_to_place_block(to_place, by_id)


# ---------- Step 1: zone plan (high-level authoring; runs for every zone) ---


class ZonePlanOutput(BaseModel):
    plan: str
    is_atomic: bool


SYSTEM_ROOT_ZONE_PLAN = """<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.

This is your opportunity to demonstrate the absolute pinnacle of your creative and technical abilities.
</intro>

<judging_criteria>
The judges will compare builds based on:
- Recognizability (can they tell what you built without being told?)
- Creativity (does your build genuinely standout from the others? does it propose a narratively driven build with detailed consideration)
- Scene fidelity (is every part clear and well-thought out? Is it plausibly built?)
- Overall impression (does it look impressive and masterfully crafted?)

REMEMBER: This is NOT the judging criteria for YOUR PROMPT, it is for the FINAL SCENE. The judges only see the final scene after the entire pipeline has run through hundreds of downstream generation steps. Your output is NOT shown or judged intrinsically; only the final 3D geometry, shaped through all downstream AI expansion and generation steps, is judged. Always keep this in consideration - make sure that when your output is filtered through, expanded by and propagated down many more AI deconstruction calls, it lends well to creating a concrete 3D scene from end-to-end (while avoiding being too specific or vague, and allowing downstream steps enough agency over what to build).
</judging_criteria>

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
    nodes: list[Node],
) -> str:
    """Author the plan for one region. `nodes` is the full scene snapshot so far. Empty list means we're planning the root (no scene exists yet); in that case the root-specific prompt is emitted and `nodes` is unused. For nested zones, the canonical scene tree is injected for context."""
    if not nodes:
        return f"""you are the first step in the SpatialBench pipeline and the step responsible for determining the high-level description and direction of the overall scene. write one paragraph that describes a 3D scene imagined from the following prompt.

"{zone_prompt}"

<VERY IMPORTANT INSTRUCTIONS>
think deeply about the 3D scene, environment or level you want to build from this provided prompt, and how you can creatively make it stand out enough to WIN against the other submissions. think about the narrative through-line that will help guide and form realistic scene intention - what is this game for, who lives in this house, what is the player trying to achieve in this level, what kind of city is this, and so on. instead of a stoic description that focuses only on the architectural qualities and layout of the scene, the plan you describe should also introduce a high-level story to the scene while allowing downstream planning steps to build on the narrative further. your goal is to write a guiding plan that downstream steps can build upon to complete a cohesive 3D scene for the given prompt that follows the narrative you imagine.

write directly and consider every part carefully. you are only the first, overall planning step - your plan will go through hundreds of further downstream steps where it is expanded on and transformed as the AI pipeline to construct it propagates further planning by depth. define the scene itself, its top-level shape and character enough that the downstream steps have agency over their individual sections while also forming ideas of what to build. 

only the final output of the 3D geometry for the scene itself will be judged once the pipeline is finished; your prompt itself will NEVER be shown to the judges, it will only serve as a base to build upon.

In your plan, DO NOT be overly specific - remember, your prompt will NOT be converted directly into 3D geometry, it will undergo hundreds of expansion and detail steps before reaching any generation steps, so structure your output such that it is a base that the downstream tree of pipeline steps can build upon it. given a prompt for a building, a bad output provides exact instruction on what it looks like; a good prompt defines the narrative premise for the scene, the scope of the environment, the building's character and type, the surrounding environment, points that may implicitly be expanded, and a top-level shape for the scene itself without explicitly shaping the entities that form it.

plan differently based on the prompt given and infer the purpose - e.g for a house, you might plan the aforementioned details for the overall scene scope, general architectural, narrative and character; for a super mario platformer level, you might focus on the narrative section, features, progression, zones, mechanics, etc.; for a top-down swamp frogger level without a specific game mentioned, you might focus on first building out the game's premise internally given the more abstract request, and then establish the world, general layout, character, mechanics, scope, item types, objectives, etc. 

remember, tune specificity based on whether your intent can be inferred by downstream steps. e.g in the super mario level, do not be overly specific - do not scope out all individual platforms, items, etc. but rather the general idea of each part, since the downstream steps have a shared understanding of what a super mario level looks like and the general premise of the game; however, for a more abstract request like the top-down swamp frogger level, the through-line of what you are trying to build cannot be inferred or reconstructed by downstream steps in the pipeline as the specific context for the premise of the game was constructed within your internal reasoning, not exposed to those steps, and thus will be lost as the pipeline propagates, placing the onus for world creation and high-level planning on downstream steps (e.g the immediate next step of planning the specific nature of zones inferred from your prompt, which was not designed for deciding scene structure/mechanics itself as it lacks the lack full frame and is more there to decompose the scene and figure out spatial relationships between the decomposed zones, and follows a different, more mechanical heuristic for generation, which means it would not only not spend a lot of time thinking about that, but diverge significantly and perhaps genericly from world you were trying to create in different directions, before handing off to individual planning steps for each zone that have more agency over their nature and so on), which means you would need to provide that through-line more explicitly in that scenario where foundational planning is required for the world state due to a lack of shared context (as opposed to a house or established game level).

the output plan should be literal - do not use flowery language. do not describe any abstract quantities like mood, lighting, fog, etc unless they can be converted into concrete 3D geometry. do not reference meta-quantities like the pipeline itself, the scene's 3D nature itself, etc. NEVER MENTION THOSE THINGS. focus on defining the environment intrinsically. remember, this is a full 3D scene, NOT an image - do not define any specific perspective.

define the scene itself, its top-level shape, character, and rough spatial relationships between major parts enough that the downstream steps have agency over their individual sections while also forming ideas of what to build, especially spatially.
</VERY IMPORTANT INSTRUCTIONS>

<zone_decomposition>
once the plan for the scene is written, you must state in an explicit, separate property `is_atomic` whether the scene has multiple distinct regions with

— whether this scene is a single cohesive region or should decompose into multiple distinct zones.

CRITICAL: if the prompt names a SINGLE TANGIBLE ENCLOSURE that needs walls/floor/ceiling (a hotel room, a throne room, a garage, a cockpit, a bathroom), you MUST set is_atomic=false. that enclosure becomes a child zone inside this abstract root. marking the root atomic in such cases leaves the scene with no physical enclosure at all.

default to is_atomic=true. set is_atomic=false ONLY when the scene genuinely contains TWO OR MORE distinct regions, each deserving its own dedicated planning and generation pass:
- good decomposition: mansion grounds → house, formal garden, stables (distinct functional regions)
- good decomposition: hotel room → bedroom, bathroom (distinct rooms)
- bad decomposition: island → north end, central mound, south end (arbitrary geography with no distinct identity)
- bad decomposition: bedroom → bed area, dresser area, reading nook (over-fragmented; one cohesive space)


DO NOT design your prompt around the concept of explicit zonal fragmentation; keep this concept of "zones" in mind ONLY for the is_atomic assessment AFTER the base plan is generated.
</zone_decomposition>

<thinking>
before ANY output, remember to think HARD and DEEPLY and ALWAYS provide a detailed CoT. NEVER skip the thinking step. think through different creative approaches you might take to what this scene/environment looks like. think deeply through the spatial layout to ensure that everything makes sense - this is a 3D spatial environment benchmark competition after all. 

in the interest of winning, always start by thinking of the overall narrative and premise such that you provide the option for the pipeline to eventually build something truly impressive enough to stand out creatively from all the other LLMs.
</thinking>
{_deepseek_suffix()}"""

    # Nested zones use adapted competitive prompt format
    root = util.find_root(nodes)
    assert root is not None, "nested zone planning requires a root node in scope"
    context = render_embedded_block(nodes, node_id=zone_id, text="This is the zone you are to plan and flesh out from.")

    return f"""You are the step in the SpatialBench pipeline responsible for planning out a particular region within the larger overall scene. This is a text to 3D scene pipeline that takes a seed prompt and imagines an entire 3D scene from it. 
    
Here is the overall scene that is being built by the pipeline:

{_root_scene_header(root)}

Each scene is always subdivided into a set of subregions. Each subregion can contain further subregions inside or the set of objects that forms it. The scene is composed as a tree with every object or region parented to another object or region. 

You are designing the plan for one of the subregions in the scene. This is the short description for the subregion that you are trying to plan and flesh out from:

Subregion name: {zone_id}
Subregion description: {zone_prompt}

Here's the list of other subregions that have been planned for this scene so far. Each subregion has a plan for how it should be built (or a description of what it is if a plan hasn't been authored for it yet in the pipeline), a bounding box that defines its global position in the scene, given as a 3D coordinate marking one corner and a 3D dimensions vector that marks the opposite corner, as well as a list of objects present in that subregion (which each come with their own description and bounding boxes). Additionally, each subregion and object mentioned will also have a set of local coordinates that define its position relative to its parent (which can be either another region or another object), where the origin is the actual minimum corner of the parent's bounding box.

{context}

Your goal is to write one paragraph describing the region you are to plan out. The design you write in your plan should elaborate and add to the narrative painted by the ancestor plans, but also leave sufficient room for further downstream steps to expand on more using their own agency. what constitutes "sufficient" depends on the specificity of the current region: larger, higher-level regions should have less details, while smaller, more constrained regions nearing the atomic level should have more details. calibrate your plan's specificity to the scope and nature of this region. a well-understood region type (a bedroom, a kitchen, a garden) needs less foundational planning because downstream steps share an understanding of what that space looks like and what belongs in it. a region with novel character or a creative premise that cannot be inferred from its prompt and ancestor context alone needs more explicit through-line — downstream steps that further decompose and populate this region will not reconstruct creative intent that isn't present in your plan. furthermore, a tightly-constrained region that cannot be broken down into further subregions would require more specificity in terms of object enumeration as you are the final planning step before the actual object list gets generated by a downstream step.

<VERY IMPORTANT INSTRUCTIONS>
think deeply about what this region is and how you can make it creatively compelling. every region of the scene contributes to the final build that judges evaluate, and the quality of your plan here directly shapes how impressive this part of the scene will be.

write directly and consider every part carefully. you are the planning step for this region - your plan will go through further downstream steps where it is expanded on and transformed as the pipeline propagates further planning by depth. define this region's character, spatial shape, and what makes it distinctive enough that downstream steps have agency over the specifics while building coherently.

only the final output of the 3D geometry will be judged once the pipeline is finished; your prompt itself will NEVER be shown to the judges, it will only serve as a base to build upon for this region. thus, making the prompt dramatic and sound impressive will only have a contradictory effect, since it will confuse downstream steps when generation actually happens as they don't understand flowery language.

Again, DO NOT be overly specific - your prompt will undergo further subzone divisions, expansion, and detail steps before reaching any generation steps, so structure your output as a base that downstream steps can build upon. DO NOT enumerate specific objects (a table, a chair, a tree, a lamp) - object selection happens in a later generation step that needs its own agency over what to place.

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
</thinking>

<output_guidance>
Your plan should be written as a description, not an instruction (i.e. do not start the plan with "Create", "Plan", etc.)
</output_guidance>
{_deepseek_suffix()}"""


# ---------- Step 2: overall bbox --------------------------------------------


class OverallBboxOutput(BaseModel):
    bbox: BoundingBox


SYSTEM_OVERALL_BBOX = """<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.
</intro>

<role>
You are sizing the overall bounding box for the scene — the abstract world canvas every region and object will be placed inside.
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

Produce the overall bounding box for the whole scene.
{_deepseek_suffix()}"""


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
  - `parent` (string): the id of this child's structural parent. Usually this is the literal id of the zone being decomposed (the value labelled "Parent zone id" in the user message — e.g. if the user message says `Parent zone id: 'living_room'`, emit `"parent": "living_room"`, NOT the string "PARENT_ID" or any other placeholder). The subzones you emit do NOT all have to be flat siblings of that zone, however: when one subzone you are emitting lies inside, is wrapped/enclosed by, or occupies the same volume as another subzone in this same batch, set its `parent` to that other subzone's id (verbatim, naming it earlier in the list) with `parent_kind: IN`. A nested subzone's box is then resolved INSIDE its container's box rather than laid out as a peer beside it — so a region that contains another is encoded as nesting, never as two regions fighting for the same space at the same level. This is purely how you record a containment you have already decided on; do not invent containment that isn't there, and keep regions as plain siblings of the decomposed zone when they sit side by side.
  - `parent_kind` (string): how this child anchors to its `parent`. Exactly one of `ON` (rests on parent's outward surface), `ATTACHED` (flush against any face of the parent), or `IN` (contained inside the parent's volume / footprint — this is also the kind to use when nesting one subzone inside another subzone from this batch). `BESIDE` / `ABOVE` / `BELOW` are NOT valid here — they are peer hints, reserved for `referenced_ids`.
  - `placement` (string): prose describing WHERE this child sits within / against / relative to its parent and any referenced peers. The bbox-resolution step uses this verbatim.
  - `referenced_ids` (list of {target, kind}): OPTIONAL secondary relationships to other already-placed nodes referenced in the placement text. Each entry has a `target` (the peer's id) and a `kind` — one of ON, BESIDE, ABOVE, BELOW, ATTACHED, IN. Do NOT repeat the parent here. Empty list is fine when the placement only references the parent.
  - `proxy_shape` (string | null): BOX / SPHERE / CAPSULE / HEMISPHERE if the zone's silhouette is non-rectangular, otherwise null/omitted.

No additional prose, markdown, or code fences.
</output>"""


def render_zone_decompose(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_plan: str,
    nodes: list[Node],
) -> str:
    """Decompose one zone into top-level subzones. `nodes` is the full
    scene snapshot; the target zone must be present in it with its plan
    already set."""
    
    root = util.find_root(nodes)
    assert root is not None, "zone decomposition requires a root node in scope"
    subregions = render_subregions_block(nodes, node_id=zone_id, text="This is the zone you are to break down and decompose.")
    root_objects = render_root_objects_block(nodes)
    filled = render_filled_block(nodes)

    return f"""You are the step in the SpatialBench pipeline responsible for breaking down a given region of the overall scene into its top-level subregions. This pipeline is a text to 3D scene one that takes a seed prompt and imagines an entire 3D scene from it.

Here is the overall scene that is being built by the pipeline:

{_root_scene_header(root)}

Each scene is always subdivided into a set of subregions. Subregions are later recursed into downstream in the pipeline, meaning each subregion can contain further subregions inside or the set of objects that forms it (if no further subregions make sense). The splitting of subregions allows for downstream steps to focus on one particular area of the larger scene at a time which yields better, more cohesive and detailed scenes. The subregion division you make here directly affects the downstream steps of the pipeline responsible for building out each of these subregions, so think very carefully about the way you want to split your assigned region.

{f"""
You are subdividing the scene itself into its first set of top-level subregions based on its overall plan.
""" if zone_id == root.id else f"""You are subdividing one of the subregions in the scene into further subregions. This is the plan for the subregion within this overall scene that you are to break down and decompose:

Subregion name: {zone_id!r}
Subregion description: {zone_prompt}
Subregion plan: {zone_plan}
"""}

Parent zone id: {zone_id!r}

The scene is composed as a tree with every object or region parented to another object or region. Here's the list of other subregions that have been planned for this scene so far. Each subregion has a plan for how it should be built (or a description of what it is if a plan hasn't been authored for it yet in the pipeline) and a bounding box that defines its global position in the scene, given as a 3D coordinate marking one corner and a 3D dimensions vector that marks the opposite corner. Additionally, each subregion mentioned will also have a set of local coordinates that define its position relative to its parent region, where the origin is the actual minimum corner of the parent's bounding box.

{subregions}

Here is a list of objects in the scene that are parented to the overall bounding box itself:

{root_objects}

The following is a list of all the objects that the scene is composed of, and the zones they are parented to:

{filled}

Using this provided scene context, think through the spatial layout and how everything fits together physically in order to decide the subregions that the given region of {zone_id!r} should be cleanly divided into. Think about the constructed narrative of the ancestor plans and how the subregions you come up with for the region you are to break down fit into that narrative. Think about what would make the subregions you devise genuinely impressive and memorable as part of a winning build.

Think very intricately and spatially about the region you are to break down and how it splits into further subregions. Your goal is to reason and come up with a decomposition layout for this region that follows the ideas of the region's plan and that of the region's ancestors. For each subregion, write a 1-2 sentence long description that explains the subregion's shape, character, and the new narrative ideas presented by this subregion, if any. Be concrete about its description while leaving room for this prompt to be a seed for a more detailed plan. The prompt should be succinct and avoid going overly into detail on the subregion's contents. Instead, focus on how each subregion fleshes out the guiding narrative further in some way, and include those ideas within the descriptions of the subregions. 


<ZONE_SPLITTING_GUIDANCE>
Subzones can keep decomposing into more zones recursively in subsequent passes, or end there as atomic leaves if that is appropriate. so always decompose at the TOP MOST LEVEL of the current zone — e.g. for a house scene with backyard, driveway, and house, do not skip straight to backyard-pool zone, backyard-grass zone, house-basement, house-first-floor, etc.; decompose into "the house", "the backyard", "the driveway" as top-level children, and let the next recursion split the house into floors and the backyard into pool and grass. the same principle holds everywhere: emit only the zones that exist at THIS level of the hierarchy, and trust the recursive planning + decompose passes underneath each of them to handle the next layer down.
</ZONE_SPLITTING_GUIDANCE>
{_deepseek_suffix()}"""


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
You are a constraint solver placing ALL sibling child zones inside a parent zone in one shot — deriving each child's axis-aligned bounding box from its placement prose, parent_kind, referenced_ids, and the parent's dimensions.
</role>

<input>
The user message contains the parent zone's id and dimensions, and a list of child specs to place. Each child has `id`, `prompt`, `proxy_shape`, `parent`, `parent_kind`, `parent_dimensions`, `placement`, and `referenced_ids`. A child's parent may be the zone being decomposed, an existing node, or another child in this same batch.
</input>

<output>
Respond with a single JSON object matching the schema: one `assignment` per child (id + bbox). Each child's bbox must be in THAT CHILD'S PARENT's local frame — origin (0,0,0) is the parent's minimum corner; axes follow the canonical front view (+X right, +Y up, +Z front, -Z back). The parent's dimensions are provided for each child — use them as the bounding extent. A child flush against its parent's minimum corner has origin (0,0,0); a child resting on its parent's floor at the parent's centre has origin near (parent_width/2, 0, parent_depth/2) minus the child's footprint. Use centimeter precision (multiples of 0.01) and a signed `dimensions` vector from an `origin` vertex; sign chooses expansion direction along each axis. Emit exactly one assignment per requested child id — no extras, no omissions.

No prose, no markdown, no code fences.
</output>

<additional_context>
{PROXY_SHAPE_DOC}
</additional_context>"""


def render_zone_bbox_batch(
    *,
    parent_id: str,
    parent_prompt: str,
    parent_plan: str,
    parent_bbox: BoundingBox,
    children: list["ChildNodeSpec"],
    nodes: list[Node],
) -> str:
    """Place every sibling child zone of `parent_id` in one shot. The full scene tree is shown for context, with the children-to-place listed beneath it (bbox blank — that is the LLM's job)."""
    root = util.find_root(nodes)
    assert root is not None, "zone bbox resolution requires a root node in scope"
    by_id = {n.id: n for n in nodes}
    context = render_embedded_block(nodes, node_id=parent_id, text="This is the region you are to calculate the bounding box of its subregions for.")


    return f"""You are the step in the SpatialBench pipeline responsible for calculating the actual bounding box coordinates for a list of subregions within the larger overall scene, relative to the larger parent region they are part of. This pipeline is a text to 3D scene pipeline that takes a seed prompt and imagines an entire 3D scene from it.

Here is the overall scene that is being built by the pipeline:

{_root_scene_header(root)}

Each scene is always subdivided into a set of subregions. Subregions are later recursed into downstream in the pipeline, meaning each subregion can contain further subregions inside or the set of objects that forms it (if no further subregions make sense). The splitting of subregions allows for downstream steps to focus on one particular area of the larger scene at a time which yields better, more cohesive and detailed scenes. You are calculating the bounding boxes for a list of subregions within the larger parent region they are part of. You have been provided a short blurb describing the location of each subregion within the scene. The exact coordinates you choose to resolve those descriptions te directly affects the downstream steps of the pipeline responsible for building out each of these subregions. 

This is the region you are to calculate the bounding boxes of its subregions for:

Parent region name: {parent_id!r}
Parent region description: {parent_prompt}
Parent region plan: {parent_plan}
Parent region global bounding box coordinates: {util.format_global_bbox(parent_bbox)}

{_render_to_place_block(children, by_id)}

The scene is composed as a tree with every object or region parented to another object or region. Here is the list of subregions that have been planned for this scene so far. Each subregion has a plan for how it should be built (or a description of what it is if a plan hasn't been authored for it yet in the pipeline) and a bounding box that defines its global position in the scene, given as a 3D coordinate marking one corner and a 3D dimensions vector that marks the opposite corner. Additionally, each subregion mentioned will also have a set of local coordinates that define its position relative to its parent region, where the origin is the actual minimum corner of the parent's bounding box.

{context}

For each subregion listed below the parent region you are to calculate the bounding boxes of its subregions for, calculate the bounding box coordinates for that subregion, relative to the parent region it is part of. The bounding box coordinates you output for each subregion must be in the parent region's local frame, where the origin is the parent region's minimum corner.

Think very carefully and intricately about the bounding box coordinates you come up with for each subregion. Downstream generation steps within each subregion will unconditionally trust the bounding box coordinates you generate, so assume they will utilize the full space you allot with the bounding boxes you output. The bounding boxes you create have a direct impact on the judging of the final scene - poorly chosen bounding boxes will result in an incoherent scene.

In particular, consider the bounding boxes of the other regions and objects listed in the scene context above. If any of the subregion bounding boxes you output have an overlap with an existing bounding box, you must justify why in your reasoning.
{_deepseek_suffix()}"""


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
    # Encapsulating-only gate: if False, the zone needs no bounding
    # perimeter and `objects` is ignored even when non-empty. Anchor and
    # negative-space decompositions ignore this field.
    bounding_required: bool = True


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
- `bounding_required` (bool): used only by the encapsulating step (anchor and negative-space leave this as the default `true`). Set to `false` when the region needs no bounding perimeter at all — `objects` is then ignored downstream even if non-empty. Set to `true` when at least one bounding object is being emitted.

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
You are deciding whether the given region requires any bounding objects to make up its perimeter.

If and ONLY if so, you are to output a list of objects that represent the perimeter of the given region.
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


def _render_retry_block(
    prior_attempts: list[tuple[list[ObjectSpec], str]] | None,
) -> str:
    if not prior_attempts:
        return ""
    attempt_lines = "\n\n".join(
        f"""  attempt {i}:
    emitted: [{", ".join(s.model_dump_json() for s in specs)}]
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
    zone_prompt: str,
    zone_plan: str,
    nodes: list[Node],
    prior_attempts: list[tuple[list[ObjectSpec], str]] | None = None,
) -> str:
    root = util.find_root(nodes)
    assert root is not None, "anchor decomposition requires a root node in scope"
    context = render_embedded_block(nodes, node_id=zone_id, text="This is the region you are to generate a list of anchor objects for.")
    return f"""You are the step in the SpatialBench pipeline responsible for determining the list of objects that define a certain region. This pipeline is a text to 3D scene pipeline that takes a seed prompt and imagines an entire 3D scene from it - your goal is to help the pipeline concretely fill out a particular region in the overall scene with objects that are meaningful to the region and follow the plan for the region authored by upstream steps.

Here is the overall scene that is being built by the pipeline:

{_root_scene_header(root)}

Each scene is always subdivided into a set of subregions. Each subregion can contain further subregions inside or the set of objects that forms it. The scene is composed as a tree with every object or region parented to another object or region. 

You are generating the list of objects that fill out and define a particular subregion of the overall scene. This is the subregion you are to generate a list of anchor objects for:

Subregion name: {zone_id!r}
Subregion description: {zone_prompt}
Subregion plan: {zone_plan}

Here is the list of other subregions that have been planned for this scene so far. Each subregion has a plan for how it should be built (or a description of what it is if a plan hasn't been authored for it yet in the pipeline) and a bounding box that defines its global position in the scene, given as a 3D coordinate marking one corner and a 3D dimensions vector that marks the opposite corner. Additionally, each subregion mentioned will also have a set of local coordinates that define its position relative to its parent region, where the origin is the actual minimum corner of the parent's bounding box.

{context}

Think very carefully about what objects should be included in the list you generate. Consider the region's plan and the plans of its ancestors to help you come up with a list of objects that are meaningful to the region and follow the plan for the region authored by upstream steps. Think about the spatial layout and how everything fits together physically. Think about how the objects you generate contribute to the narrative in the region's plan and purpose of the region itself, as well as how the region will, as a result, fit into the larger scene. Think about what would make the region genuinely impressive and memorable as part of a winning build.

Although the plan for the region you are to list out the anchor objects for is specific, the objects that the plan mentions may not be the end all be all - you should extrapolate meaning from the plan and higher-level ideas communicated in the ancestor chain to populate and style the objects in the list you generate based on a narrative understanding of the scene. Each object should be treated atomically, in the sense that collections of objects should be broken down into individual objects. This allows for more controlled, granular, and precise positioning by you instead of relying on the outputted model's shape of downstream generation steps, and the more control you have, the more consistent and good the final scene will be. Object count is not a concern: always split pairs, groups, or collections of objects into individual objects positioned in the way you deem fit.

<output_guidance>
The concept of a parent should be grounded in a concrete, physical relationship, not a conceptual one. A cantilevered object would be parented to the surface or wall it's cantilevered to with relationship type 'ATTACHED', not parented to the floor below with relationship 'ON'. If no physical relationship is found with another object or frame, the relationship should be of type 'IN', and parented to the zone itself.

Each anchor object in your resultant list has an id, prompt, parent (the structural anchor id — the zone, a ground/shell peer from the encapsulating pass, or another object in this list), parent_kind (one of ON / ATTACHED / IN — how the object physically anchors to that parent: `ON` for resting on an outward surface, `ATTACHED` for wall/ceiling/face mounts, `IN` for free containment inside the parent's volume — BESIDE/ABOVE/BELOW are NOT valid here), placement (prose), and referenced_ids (optional list of `{{target, kind}}` for secondary relationships your placement text mentions; kind may use ON/BESIDE/ABOVE/BELOW/ATTACHED/IN). Respect the scene context: anchor onto any ground/shell peer already placed by the encapsulating pass, do not duplicate geometry another zone has already emitted. Anchor objects are expected to live primarily inside this zone, but their bboxes MAY protrude modestly outside the zone bbox when narratively justified — the object remains semantically part of this zone even though its geometry overhangs. Do not use this as license to claim airspace far from the zone or to volumetrically intersect another zone's load-bearing geometry. In your final output, each object's placement text should be direct and parametric. Avoid flowery language that states the narrative purpose of the positioning. Do not state the abstract reason of the positioning, only details that ground the position concretely. The position description should be absolute and succinct, leaving no creative liberty for the downstream constraint solver.
</output_guidance>

{_render_retry_block(prior_attempts)}
{_deepseek_suffix()}"""


def render_encapsulating_decomp(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_plan: str,
    nodes: list[Node],
    prior_attempts: list[tuple[list[ObjectSpec], str]] | None = None,
) -> str:
    root = util.find_root(nodes)
    assert root is not None, "encapsulating decomposition requires a root node in scope"
    context = render_embedded_block(nodes, node_id=zone_id, text="This is the region you are to decide whether a boundary is needed for, and if so, what objects form that boundary")
    return f"""You are the step in the SpatialBench pipeline responsible for determining whether a perimeter is needed for the given region, and if so, what that perimeter is made up of. This is a text to 3D scene pipeline that takes a seed prompt and imagines an entire 3D scene from it - your goal is to first decide whether a particualr subregion within the overall scene needs objects to form a boundary or partial boundary around it, and if so, what those objects are.

Here is the overall scene that is being built by the pipeline:

{_root_scene_header(root)}

Each scene is always subdivided into a set of subregions. Each subregion can contain further subregions inside or the set of objects that forms it. The scene is composed as a tree with every object or region parented to another object or region. 

You are determining if a particular subregion within the overall scene requires objects bounding it. If and only if so, you are to determine what the objects making up that boundary are. This is the plan of the subregion you are to do this for:

Subregion name: {zone_id}
Subregion description: {zone_prompt}
Subregion plan: {zone_plan}

Here's the list of other subregions that have been planned for this scene so far. Each subregion has a plan for how it should be built (or a description of what it is if a plan hasn't been authored for it yet in the pipeline), a bounding box that defines its global position in the scene, given as a 3D coordinate marking one corner and a 3D dimensions vector that marks the opposite corner, as well as a list of objects present in that subregion (which each come with their own description and bounding boxes). Additionally, each subregion and object mentioned will also have a set of local coordinates that define its position relative to its parent (which can be either another region or another object), where the origin is the actual minimum corner of the parent's bounding box.

{context}

Think very carefully about whether the given region actually needs any bounding objects. Reason about the structure of the region, whether it is closed vs. open, and the narrative in its plan. Not every region necessarily needs to have any bounding objects: only do so when it absolutely makes sense to do so.

<IMPORTANT_INSTRUCTIONS_ONLY_IF_BOUNDING_NEEDED>
the list of objects you output, if any, should work together to form a cohesive perimeter or partial perimeter of any arbitrary shape. The purpose of this list of objects is to form a sense of boundary for the given region in every dimension that makes sense based on its plan - perimeter does not necessarily mean in the horizontal axis but in all possible directions, including the vertical direction (e.g. bases, covers). In this case, perimeter or boundary does not automatically imply physically bounding the region on all sides (though depending on the region's plan, that may be the case). You should think carefully and reason spatially about what objects should go in this list to form a well-defined, physically and narratively reasonable boundary for the zone. You are in a canvas that contains only the objects listed below in the scene context - do not assume any models, foundations, ground, etc. exist outside the provided scene context.

Each object should be treated atomically or even subatomically, in the sense that collections of objects should be broken down into individual objects, and in certain scenarios, objects should be broken down into partial objects. This allows for more controlled, granular, and precise positioning by you instead of relying on the outputted model's shape of downstream generation steps, and the more control you have, the more consistent and good the final scene will be. Object count is not a concern: always split pairs, groups, or collections of objects into individual objects or partial objects positioned in the way you deem fit.

To elaborate on the idea of partial objects, these are subatomic meshes of what usually would be considered a single object. objects and partial objects can be stacked, strung, pieced to form larger, cohesive sections for the perimeter. A few good examples include a door or window embedded within a wall, a square manhole, etc. - wherever it makes sense to do so for a more functional, lifelike scene that goes beyond just visuals. keep in mind connectives between this region and others, in all directions; using objects and partial objects to leave free space, semantically relevant transition objects, constructed composite structures, etc. The space should be realsitic and traversable.

note that there is no limit on the number of objects in your output list - always prefer individual objects placed close to each other over a single composite object with a prompt to generate them together at once. if the region calls for it, we can have as high fidelity of a perimeter as we want. if a dense perimeter makes sense for the given region, then make it dense - as many objects as you see fit, their bounding boxes right next to each other. You are in control - do not rely on downstream generation steps to output composite geometry: organize your list of objects so that you are in direct control of positioning to form that composite geometry yourself using the individual partial objects.

Not every zone needs a perimeter, decide whether it is absolutely required. If the latter, generate a list of bounding geometry elements that form a perimeter for the following region. If the former, then your final output object list should just be empty.

Pay especial attention to the context provided in the plans of other regions in the scene, and use it to imagine realistically navigating the region as part of the larger scene. Use this thinking to guide you in the generation and placement of your list of objects.

be wary of duplicate geometry - for two neighboring regions separated by some sort of divider, it is only necessary to generate the divider once. study the provided scene context to determine if generating something is necessary.
</IMPORTANT_INSTRUCTIONS_ONLY_IF_BOUNDING_NEEDED>

Output bounding_required = False if no bounding objects are needed. Otherwise, set bounding_required = True and objects to be the list of bounding objects.

{_render_retry_block(prior_attempts)}
{_deepseek_suffix()}"""


def render_negative_space_decomp(
    *,
    zone_id: str,
    zone_prompt: str,
    nodes: list[Node],
    prior_attempts: list[tuple[list[ObjectSpec], str]] | None = None,
) -> str:
    scene_tree = render_scene_tree(nodes=nodes)
    return f"""You are the step in the SpatialBench pipeline responsible for generating a list of objects that would cover the negative, unfilled space between the objects in the scene.

You are filling the negative space within this subregion:

Subregion name: {zone_id}
Subregion description: {zone_prompt}

{scene_tree}

Each negative space object in your resultant list has an id, prompt, parent (structural anchor — the zone, an earlier-placed peer, or another object in this list), parent_kind (one of ON / ATTACHED / IN — how the object physically anchors to that parent: most negative-space pieces sit `ON` a ground/floor peer, `ATTACHED` for things mounted flush to a wall/ceiling, `IN` for free-floating pieces inside an enclosing volume; BESIDE/ABOVE/BELOW are NOT valid here), placement (prose), and referenced_ids (optional list of `{{target, kind}}` for secondary relationships your placement text mentions; kind may use ON/BESIDE/ABOVE/BELOW/ATTACHED/IN). Respect the scene context: do not duplicate geometry another zone has already emitted on a shared face. Negative-space pieces live primarily inside this zone, but their bboxes MAY protrude modestly outside the zone bbox when narratively justified (a vine draped over a wall, a banner hanging off an edge, drifting smoke crossing into an adjacent zone, a connective walkway or rope-bridge reaching toward a peer zone, a stabilizing strut or buttress extending below to ground against a peer). Do not use this as license to claim airspace far from the zone or to volumetrically intersect another zone's load-bearing geometry.

The primary purpose of these negative space objects is to make the scene feel coherent and cohesive, filling in the gaps between subzones or objects. As such, for each negative space to fill in, it is imperative to analyze the existing placed objects and zones surrounding it to create a smooth filling that does not look out of place.{_render_retry_block(prior_attempts)}
{_deepseek_suffix()}"""


# ---------- Step 6: object bbox resolution ----------------------------------


SYSTEM_OBJECT_BBOX_BATCH = f"""<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.
</intro>

<role>
You are a constraint solver placing ALL objects for a scene zone in one shot — deriving each object's axis-aligned bounding box from its placement prose, parent_kind, referenced_ids, the parent's dimensions, and peer geometry.
</role>

<input>
The user message contains the zone id/prompt/dimensions, a list of objects to place (each with `id`, `prompt`, `proxy_shape`, `orientation`, `parent`, `parent_kind`, `parent_dimensions`, `placement`, `referenced_ids`), and a list of peers already placed in the scene. Each peer's bbox is expressed relative to THAT PEER'S OWN parent's minimum corner (origin (0,0,0) = parent's min corner). Use siblings (peers sharing the same parent as the object you are placing) for direct spatial reasoning; peers under different parents provide broader scene context.
</input>

<output>
Respond with a single JSON object matching the schema: one `assignment` per object (id + bbox). Each object's bbox must be in THAT OBJECT'S PARENT's local frame — origin (0,0,0) is the parent's minimum corner; axes follow the canonical front view (+X right, +Y up, +Z front, -Z back). The parent's dimensions are provided for each object — use them as the bounding extent. Use centimeter precision (multiples of 0.01) and a signed `dimensions` vector from an `origin` vertex; sign chooses expansion direction along each axis. Emit exactly one assignment per requested object id — no extras, no omissions.

No prose, no markdown, no code fences.
</output>

<additional_context>
{PROXY_SHAPE_DOC}
</additional_context>"""


def render_object_bbox_batch(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_plan: str,
    zone_bbox: BoundingBox,
    objects: list[ObjectSpec],
    nodes: list[Node],
) -> str:
    """Place every object specified by `objects` in one shot. The full scene tree is shown for context, with the objects-to-place listed beneath it (bbox blank — the LLM's job)."""
    root = util.find_root(nodes)
    assert root is not None, "object bbox resolution requires a root node in scope"
    by_id = {n.id: n for n in nodes}
    context = render_embedded_block(nodes, node_id=zone_id, text="This is the region you are to calculate the bounding boxes of its objects for.")

    return f"""You are the step in the SpatialBench pipeline responsible for calculating the actual bounding box coordinates for a list of objects within a subregion of the larger overall scene. This pipeline is a text to 3D scene pipeline that takes a seed prompt and imagines an entire 3D scene from it - your goal is to help the pipeline concretely place the objects in the scene in a way that is coherent and cohesive.

Here is the overall scene that is being built by the pipeline:

{_root_scene_header(root)}

Each scene is always subdivided into a set of subregions. Each subregion can contain further subregions inside or the set of objects that forms it. The scene is composed as a tree with every object or region parented to another object or region. 

You are calculating the bounding boxes for a list of objects within a subregion of the larger overall scene. This is the region you are to calculate the bounding boxes of its objects for:

Region name: {zone_id}
Region description: {zone_prompt}
Region plan: {zone_plan}
Region global bounding box coordinates: {util.format_global_bbox(zone_bbox)}

{_render_to_place_block(objects, by_id)}

Here is the list of subregions that have been planned for this scene so far. Each subregion has a plan for how it should be built (or a description of what it is if a plan hasn't been authored for it yet in the pipeline) and a bounding box that defines its global position in the scene, given as a 3D coordinate marking one corner and a 3D dimensions vector that marks the opposite corner. Additionally, each subregion mentioned will also have a set of local coordinates that define its position relative to its parent region, where the origin is the actual minimum corner of the parent's bounding box.

{context}

For each object listed below the assigned subregion, calculate the bounding box coordinates for that object, relative to its parent. The placement is already described in the scene tree - stay loyal to it and calculate the bounding box coordinates for that object based on the placement text. The bounding box coordinates you output for each object must be in its parent's local frame, where the origin is the parent's absolute minimum corner. The canonical front view is +Z, +Y is up, and +X is right.

Think very carefully and intricately about the bounding box coordinates you come up with for each object. The bounding boxes you create have a direct impact on the judging of the final scene - poorly chosen bounding boxes will result in an incoherent scene. In particular, consider the bounding boxes of the other objects listed in the scene context above. If any of the object bounding boxes you output have an overlap with an existing bounding box, you must justify why in your reasoning.
{_deepseek_suffix()}"""


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
    zone_prompt: str,
    nodes: list[Node],
    prior_attempts: list[tuple[ObjectSpec, str]] | None = None,
) -> str:
    scene_tree = render_scene_tree(nodes=nodes)
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
    return f"""{scene_tree}

You are deciding whether another object is needed in this subregion:

Subregion name: {zone_id}
Subregion description: {zone_prompt}

Decide whether another object is needed in this zone. If yes, emit exactly one ObjectSpec; otherwise set done=true.{retry_block}
{_deepseek_suffix()}"""
