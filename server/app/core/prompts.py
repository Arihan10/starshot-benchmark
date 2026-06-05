"""Prompts and structured-output schemas for LLM calls."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    if model and ("deepseek" in model.lower()):
        return DEEPSEEK_INJECTION
    return ""


def _render_proxy_shape(p: ProxyShape | None) -> str:
    return p.value if p is not None else "BOX"


# Shared anti-ephemera guidance injected into every prompt that authors
# scene content (region plans, region decomposition, object decomposition,
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


# Shared occupancy/occlusion guidance injected into the object decomposition
# tail and the object bbox-resolution solver. The pipeline composes opaque
# solid meshes additively — there is no boolean subtraction — so a node whose
# box falls wholly inside another solid never renders. Authors and the solver
# both need this mental model to avoid burying functional geometry inside a
# region-filling mass (e.g. a walkway "carved into" a solid cliff block).
SOLID_OCCUPANCY_DOC = """SOLID OCCUPANCY. The scene is assembled ADDITIVELY from opaque solid meshes — nothing is ever subtracted from anything, and the renderer shows only outer surfaces. Two boxes may overlap in space, but where they do the enclosing solid simply hides whatever is inside it; no cavity is cut. Reason concretely about the volume each solid occupies:

  * A node whose box sits wholly inside another solid's box is INVISIBLE — wasted geometry that never reads in the final scene. Anything meant to be seen must claim space no other solid already fills, or break that solid's surface and stand proud of it.
  * A solid sized to fill its entire region leaves NO room for anything to be seen inside that region. A backing or bounding mass and the things it is meant to host cannot occupy the same volume — one of them has to give.
  * "Carved-in / inset / recessed / sunken / embedded" features — a path cut into a cliff face, a window set into a wall, a niche, a cave mouth — are NOT produced by nesting a child inside a solid that already fills the space; that only entombs it. They emerge either by shaping the surrounding solid AROUND the void so the opening stays clear, or by the feature breaking the surface and protruding from it."""


# Shared orientation guidance for the object bbox-resolution solver. Orientation
# is a yaw about +Y applied to the mesh BEFORE it is stretched into its
# world-axis-aligned box, so the solver must size the box to the TURNED footprint
# and read facing against the one global frame. The +90 -> +X mapping below
# matches the actual transform baked in `utils/geometry.py` and the equivalent
# +Y quaternion in `utils/glb_place.py` (NOT the older, inverted prose).
ORIENTATION_DOC = """ORIENTATION. A node's `orientation` is a single yaw angle — degrees about the vertical +Y axis, snapped to one of -180/-135/-90/-45/0/45/90/135/180 — that turns the node's intrinsic FRONT (the face that points +Z when orientation is 0). It is the node's only rotation: there is no pitch or roll, so a node can never tip or tilt, only spin about the upright axis. Facing is ALWAYS read against the one global frame shared by the entire scene (+X = right, +Y = up, +Z = toward the viewer / front, -Z = away / back) — never against the parent or any local frame:

  * 0 -> front faces +Z (toward the viewer)
  * +90 -> front faces +X (to the right); -90 -> front faces -X (to the left)
  * 180 / -180 -> front faces -Z (away from the viewer)
  * +45 -> front-right (between +Z and +X); -45 -> front-left; +135 -> back-right; -135 -> back-left

Positive degrees swing the front toward +X (the right); negative degrees toward -X (the left).

Orientation NEVER tilts or rotates the bounding box itself — every bbox stays axis-aligned to world X/Y/Z; the yaw spins the mesh INSIDE the box. The mesh is turned FIRST and then stretched per-axis to fill the box you give it, so the box must describe the node's footprint AS TURNED, not its head-on (orientation 0) footprint:

  * A +/-90 turn swaps width and depth: a node that is W wide (along X) by D deep (along Z) at 0 occupies D along X and W along Z once turned +/-90 — so give the box the swapped extents (D on X, W on Z).
  * 0 and 180 keep the same W-by-D footprint (only the facing flips). +/-45 and +/-135 turns make the axis-aligned footprint grow on BOTH X and Z, because the box has to contain the mesh along its diagonal.
  * Height (Y) is never changed by yaw.

The per-axis fill never crops and never auto-rotates, so a box whose X:Z proportions don't match the turned node will visibly SQUASH or STRETCH it. Honor each node's orientation twice over: size its box to the turned footprint, AND place it so its front — and the clearance its front needs — point the way the orientation dictates (a chair yawed +90 faces +X, so it reads as facing whatever sits to its right and wants open space on that +X side)."""


# --- canonical scene-context tree --------------------------------------------
#
# Every prompt that shows the LLM "what does the scene look like right now" routes through the renderers below.
# They render one data shape: the subregion tree (regions only) followed by a flat list of every object. Objects are never nested under regions — each object entry carries its own `parent` (structural anchor) and `parent_region`, so the rendered information mirrors V1's flat <ZONES>/<OBJECTS> dump.
# They share the entry/formatting helpers here and the type-agnostic utilities in `util`.

_NO_NODES_MESSAGE = "(no regions or objects have been placed yet — this is the very start of the run)"
_NO_SUBREGIONS_MESSAGE = "{(none - no other subregions have been planned yet)}"

# Inline marker appended to a targeted subregion's name line in the
# embedded block. The arrow points back at the node id and labels it as
# the target so a prompt can call the LLM's attention to one region; the
# caller-supplied text follows it.
_TARGET_MARKER = "<-- TARGET:"


def _root_scene_header(root: Node) -> str:
    """Root prompt, plan, and overall bounding box — injected at the top of every prompt that shows scene context. The root bbox is delivered in natural language — its `W by H by D` dimensions plus its origin corner, tagged `(scene root)` — instead of the raw `origin/dimensions` coordinate dump, while still surfacing every value the box carries."""
    dx, dy, dz = root.bbox.dimensions
    ox, oy, oz = root.bbox.origin
    return (
        f'Prompt: "{root.prompt}"\n'
        f'Plan: "{root.plan}"\n'
        f"Overall scene (root) bounding box: {dx:.2f}m by {dy:.2f}m by {dz:.2f}m, with its origin corner at ({ox:.2f}, {oy:.2f}, {oz:.2f}) m (scene root)"
    )


def _local_coords_line(node: Node, by_id: dict[str, Node]) -> str | None:
    """`Local coordinates relative to its parent (<pid>): ...` line, or None for the root / a node whose parent is absent from the snapshot."""
    if node.parent_id is not None and node.parent_id in by_id:
        coords = util.format_local_origin(node.bbox, by_id[node.parent_id].bbox)
        return f"Local origin corner (relative to {node.parent_id}, measured from its min corner): {coords}"
    return None


def _object_entry(obj: Node, by_id: dict[str, Node], parent_zone: str) -> str:
    """Full-detail entry for one concrete object (no plan), rendered as a member of the scene's flat object list.

    `parent` is the object's structural-anchor block — `parent_id` (the peer object or region this object physically rests on / attaches to / sits inside), `parent_relationship_kind` (ON / ATTACHED / IN), `parent_dimensions` (that parent's size), and `parent_global_origin_corner` (its world position) — modeled on the `parent_region` block in `_region_plan_entry`. `parent_region` is the id of the subregion this object belongs to and `parent_region_dimensions` is that region's size; both are omitted when `parent_region` would equal `parent_id` (the object anchors directly to its region — the `parent` block already states them), and they are shown only when the object anchors to a peer object (e.g. a lamp ON a nightstand has parent_id=nightstand, parent_region=<the subregion the nightstand is in>). Together they carry the same information V1's flat <OBJECTS> dump splits across the `parent` pointer and the node's position in the tree."""
    lines = [
        f"Name: {obj.id}",
        f'Description: "{obj.prompt}"',
    ]
    if obj.parent_id is not None:
        kind_str = obj.parent_kind.value if obj.parent_kind is not None else "(unknown)"
        parent_lines = [
            f"parent_id: {obj.parent_id}",
            f"parent_relationship_kind: {kind_str}",
        ]
        if obj.parent_id in by_id:
            parent_bbox = by_id[obj.parent_id].bbox
            parent_lines.append(f"parent_dimensions: {util.format_dimensions(parent_bbox)}")
            parent_lines.append(
                f"parent_global_origin_corner: {util.format_global_origin(parent_bbox)}"
            )
        lines.append("parent: " + util.braces("\n".join(parent_lines)))
    if parent_zone != obj.parent_id:
        lines.append(f"parent_region: {parent_zone}")
        if parent_zone in by_id:
            lines.append(
                f"parent_region_dimensions: {util.format_dimensions(by_id[parent_zone].bbox)}"
            )
    if obj.placement is not None:
        lines.append(f'placement: "{obj.placement}"')
    if obj.referenced_ids:
        refs = ", ".join(f"{r.target}: {r.kind.value}" for r in obj.referenced_ids)
        lines.append(f"relationships: [{refs}]")
    else:
        lines.append("relationships: []")
    lines.append(f"proxy_shape: {_render_proxy_shape(obj.proxy_shape)}")
    lines.append(f"orientation: {obj.orientation}deg")
    lines.append(f"Dimensions: {util.format_dimensions(obj.bbox)}")
    lines.append(f"Global origin corner: {util.format_global_origin(obj.bbox)}")
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
    """Subregion tree, regions only: a subregion's fields and (recursively) its nested subregions. Objects are no longer listed here — every object in the scene is rendered once in the flat list produced by `render_objects_flat`, each carrying its own `parent_region`. When `target_id` matches this region (at any depth), an inline marker carrying `target_text` is appended to its name line so a prompt can point the LLM at this one region."""
    _, subregions = util.split_region_members(region.id, idx)
    name_line = f"Name: {region.id}"
    if target_id is not None and region.id == target_id:
        name_line += f"   {_TARGET_MARKER} {target_text}".rstrip()
    lines = [
        name_line,
        f'Description: "{region.prompt}"',
    ]
    if region.plan is not None:
        lines.append(f'Plan for this region: "{region.plan}"')
    if region.placement is not None:
        lines.append(f'placement: "{region.placement}"')
    lines.append(f"proxy_shape: {_render_proxy_shape(region.proxy_shape)}")
    lines.append(f"Dimensions: {util.format_dimensions(region.bbox)}")
    lines.append(f"Global origin corner: {util.format_global_origin(region.bbox)}")
    local = _local_coords_line(region, by_id)
    if local is not None:
        lines.append(local)
    if region.parent_id is not None and region.parent_id in by_id:
        parent = by_id[region.parent_id]
        parent_placement = f'"{parent.placement}"' if parent.placement is not None else "(none)"
        parent_lines = [
            f"parent_name: {parent.id}",
            f"parent_placement: {parent_placement}",
            f"parent_dimensions: {util.format_dimensions(parent.bbox)}",
            f"parent_global_origin_corner: {util.format_global_origin(parent.bbox)}",
        ]
        lines.append("parent_region: " + util.braces("\n".join(parent_lines)))
    if subregions:
        lines += [
            "",
            f'"{region.id}" decomposes into the following further subregions.',
            "",
            util.brace_group(
                [_region_plan_entry(s, idx, by_id, target_id, target_text) for s in subregions]
            ),
        ]
    return util.braces("\n".join(lines))


def _region_embedded_entry(
    region: Node,
    idx: dict[str | None, list[Node]],
    obj_idx: dict[str | None, list[Node]],
    by_id: dict[str, Node],
    target_id: str | None = None,
    target_text: str = "",
) -> str:
    """Subregion tree (embedded form): a subregion's fields, then a FLAT list of the objects placed directly inside it, then (recursively) its nested subregions. Objects are never nested under one another — a peer-anchored object names its anchor in its own `parent` block rather than nesting beneath it. When `target_id` matches this region (at any depth), an inline marker carrying `target_text` is appended to its name line so a prompt can point the LLM at this one region."""
    objects, subregions = util.split_region_members_owned(region.id, idx, obj_idx)
    name_line = f"Subregion name: {region.id}"
    if target_id is not None and region.id == target_id:
        name_line += f"   {_TARGET_MARKER} {target_text}".rstrip()
    lines = [
        name_line,
        f'Description: "{region.prompt}"',
    ]
    if region.plan is not None:
        lines.append(f'Plan for this region: "{region.plan}"')
    if region.placement is not None:
        lines.append(f'placement: "{region.placement}"')
    lines.append(f"proxy_shape: {_render_proxy_shape(region.proxy_shape)}")
    lines.append(f"Dimensions: {util.format_dimensions(region.bbox)}")
    lines.append(f"Global origin corner: {util.format_global_origin(region.bbox)}")
    local = _local_coords_line(region, by_id)
    if local is not None:
        lines.append(local)
    if region.parent_id is not None and region.parent_id in by_id:
        parent = by_id[region.parent_id]
        parent_placement = f'"{parent.placement}"' if parent.placement is not None else "(none)"
        parent_lines = [
            f"parent_name: {parent.id}",
            f"parent_placement: {parent_placement}",
            f"parent_dimensions: {util.format_dimensions(parent.bbox)}",
            f"parent_global_origin_corner: {util.format_global_origin(parent.bbox)}",
        ]
        lines.append("parent_region: " + util.braces("\n".join(parent_lines)))
    if objects:
        lines += [
            "",
            f'Objects placed directly within "{region.id}" (a flat list — an object anchored to a peer object names that peer in its `parent` block rather than nesting beneath it):',
            "",
            util.brace_group([_object_entry(o, by_id, parent_zone=region.id) for o in objects]),
        ]
    if subregions:
        lines += [
            "",
            f'Here\'s the list of subregions that are present within "{region.id}".',
            "",
            util.brace_group(
                [_region_embedded_entry(s, idx, obj_idx, by_id, target_id, target_text) for s in subregions]
            ),
        ]
    return util.braces("\n".join(lines))


def _render_to_place_block(
    to_place: list[ChildNodeSpec] | list[ObjectSpec] | None,
    by_id: dict[str, Node],
    parent_zone: str | None = None,
    show_orientation: bool = True,
) -> str:
    """Pseudo-JSON block of the children/objects whose bboxes a bbox-batch step must determine — the caller writes the introducing sentence. Empty string when there is nothing to place.

    `parent_zone` is the id of the region these specs are being generated as a part of (the region the bbox step is resolving) and is emitted on each spec as `parent_region`. When supplied, every spec whose structural `parent` is NOT that region also carries `parent_region` + `parent_region_dimensions`, mirroring `_object_entry`: a spec anchored to a peer still names the region it belongs to, while a spec anchored directly to the region omits it (its `parent` block already names it). `show_orientation` stays True for objects (which carry a real yaw) but is set False when placing subregions, since zones are never yawed — their `orientation` is always 0 and would only add noise."""
    if not to_place:
        return ""
    to_place_ids = {c.id for c in to_place}
    entries: list[str] = []
    for c in to_place:
        kind_str = c.parent_kind.value
        if c.parent in by_id:
            pdims_str = util.format_dimensions(by_id[c.parent].bbox)
            porigin_str = util.format_global_origin(by_id[c.parent].bbox)
        elif c.parent in to_place_ids:
            pdims_str = "(parent is also being placed in this batch — use your emitted dimensions for it)"
            porigin_str = "(parent is also being placed in this batch — use your emitted position for it)"
        else:
            pdims_str = "(parent id not recognised in current scene)"
            porigin_str = "(parent id not recognised in current scene)"
        lines = [
            f"id: {c.id}",
            f"parent: {c.parent}",
            f"parent_relationship_kind: {kind_str}",
            f"parent_dimensions: {pdims_str}",
            f"parent_global_origin_corner: {porigin_str}",
        ]
        if parent_zone is not None and c.parent != parent_zone:
            lines.append(f"parent_region: {parent_zone}")
            if parent_zone in by_id:
                lines.append(
                    f"parent_region_dimensions: {util.format_dimensions(by_id[parent_zone].bbox)}"
                )
        lines.append(f"proxy_shape: {_render_proxy_shape(c.proxy_shape)}")
        if show_orientation:
            lines.append(f"orientation: {c.orientation}deg")
        lines.append(f'prompt: "{c.prompt}"')
        lines.append(f'placement: "{c.placement}"')
        if c.referenced_ids:
            refs = ", ".join(f"{r.target}: {r.kind.value}" for r in c.referenced_ids)
            lines.append(f"relationships: [{refs}]")
        else:
            lines.append("relationships: []")
        entries.append(util.braces("\n".join(lines)))
    return util.brace_group(entries)


def render_subregions_block(
    nodes: list[Node],
    *,
    node_id: str | None = None,
    text: str = "",
) -> str:
    """Pseudo-JSON block of the scene's subregion tree (regions only): each carries its plan and bbox, recursing into nested subregions. Objects are rendered separately in the flat list (`render_objects_flat`). Renders the single-region placeholder when the scene is one undivided region.

    Pass `node_id` to point the LLM at one specific region: the subregion whose id matches gets an inline target marker carrying `text` appended to its name line, found at any depth of the tree. With `node_id` unset (the default) the block renders exactly as before."""
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


def render_direct_subregions(nodes: list[Node], zone_id: str) -> str:
    """Pseudo-JSON block of `zone_id`'s DIRECT child subregions (regions only), rendered exactly like the scene's top-level subregion list (`render_subregions_block`): each entry carries its plan and bbox and recurses into its own nested subregions. Objects are not listed here — every object in the scene lives in the flat list produced by `render_objects_flat`. Renders the single-region placeholder when `zone_id` has no child subregions of its own."""
    by_id = {n.id: n for n in nodes}
    idx = util.index_children(nodes)
    _, subregions = util.split_region_members(zone_id, idx)
    if not subregions:
        return _NO_SUBREGIONS_MESSAGE
    return util.brace_group(
        [_region_plan_entry(s, idx, by_id) for s in subregions]
    )


def render_objects_flat(nodes: list[Node]) -> str:
    """Pseudo-JSON block of EVERY object in the scene as a single flat list — no nesting under regions. Each object is rendered once via `_object_entry`, tagged with its `parent_region` (the subregion it belongs to) + `parent_region_dimensions` and its `parent` block (structural anchor). Objects are walked region by region (root first, then a depth-first descent through subregions) so the list is deterministic; an object whose structural parent is a peer object still appears under the region that contains it. Renders an empty `{}` block when the scene has no objects yet."""
    root = util.find_root(nodes)
    if root is None:
        return util.brace_group([])
    by_id = {n.id: n for n in nodes}
    idx = util.index_children(nodes)
    oidx = util.index_objects_by_region(nodes)
    entries: list[str] = []

    def walk(region: Node) -> None:
        objects, subregions = util.split_region_members_owned(region.id, idx, oidx)
        entries.extend(_object_entry(o, by_id, parent_zone=region.id) for o in objects)
        for s in subregions:
            walk(s)

    walk(root)
    return util.brace_group(entries)


def render_root_objects(nodes: list[Node]) -> str:
    """Flat list of the objects anchored directly to the scene root — the
    shared geometry (shells, ground planes, ambient root-level fill) the whole
    scene rests on. Split out of `render_embedded_block` because this is GLOBAL
    context: every region sits within it, so it reads best next to the root
    scene header rather than buried inside the per-subregion tree. Callers wire
    it in wherever that placement makes sense. Returns a short placeholder when
    the root carries no such geometry yet."""
    root = util.find_root(nodes)
    if root is None:
        return _NO_NODES_MESSAGE
    by_id = {n.id: n for n in nodes}
    oidx = util.index_objects_by_region(nodes)
    root_objects = oidx.get(root.id, [])
    if not root_objects:
        return "No objects are parented directly to the root yet."
    return (
        "Here's a list of objects that belong to the root region (the scene's shared shell / ground geometry):\n\n"
        + util.brace_group(
            [_object_entry(o, by_id, parent_zone=root.id) for o in root_objects]
        )
    )


def render_embedded_block(
    nodes: list[Node],
    *,
    node_id: str | None = None,
    text: str = "",
) -> str:
    """Scene context in EMBEDDED form: the subregion tree where each subregion lists the objects placed directly inside it inline as a FLAT list — objects are never nested under one another, a peer-anchored object just names its anchor in its own `parent` block — and then recurses into its nested subregions. Objects anchored directly to the scene root (the shell/ground meshes from the root's encapsulating pass) are NOT included here — they are global geometry, rendered separately by `render_root_objects`, which callers place next to the root scene header. Renders the single-region placeholder when the scene has no subregions yet.

    Pass `node_id` to point the LLM at one specific region: the subregion whose id matches gets an inline target marker carrying `text` appended to its name line, found at any depth of the embedded tree. With `node_id` unset (the default) no marker is drawn."""
    root = util.find_root(nodes)
    if root is None:
        return _NO_SUBREGIONS_MESSAGE
    by_id = {n.id: n for n in nodes}
    idx = util.index_children(nodes)
    oidx = util.index_objects_by_region(nodes)
    _, subregions = util.split_region_members(root.id, idx)
    if not subregions:
        return _NO_SUBREGIONS_MESSAGE
    return util.brace_group(
        [_region_embedded_entry(s, idx, oidx, by_id, node_id, text) for s in subregions]
    )


def render_scene_tree(
    *,
    nodes: list[Node],
    to_place: list[ChildNodeSpec] | list[ObjectSpec] | None = None,
) -> str:
    """Render the scene context as the subregion tree (regions only) followed by the flat list of every object in the scene. Each object appears exactly once in the flat list, tagged with its `parent` (structural anchor) and `parent_region` — there is no nesting of objects under regions.
    """
    if not nodes:
        return _NO_NODES_MESSAGE
    root = util.find_root(nodes)
    if root is None:
        return _NO_NODES_MESSAGE
    by_id = {n.id: n for n in nodes}

    body = f"""This is the overall plan for the entire scene.

{_root_scene_header(root)}

Each scene is always subdivided into a set of subregions. Each subregion can contain further subregions inside; the objects that fill the scene are listed separately as a flat list below.

Here's the list of subregions that have been planned for this scene so far. Each subregion has a plan for how it should be built, its dimensions, and a global coordinate marking its origin corner. Additionally, each subregion will also have a set of local coordinates that define its position relative to its parent region, where the origin is the actual minimum corner of the parent's bounding box.

{render_subregions_block(nodes)}

The objects that fill these regions are listed below as a single flat list. Objects are not nested under their regions. Each object has a description detailing what it is, a `parent` block (its structural anchor — `parent_id`, `parent_relationship_kind`, and `parent_dimensions`; the parent can either be another object or the region it belongs to itself), a `parent_region` (the region it belongs to) with `parent_region_dimensions`, its dimensions, a global coordinate marking its origin corner, and a set of local coordinates that define that origin relative to its parent. An object's `parent_region`/`parent_region_dimensions` are omitted when its structural parent is its region, since the `parent` block already states them.

{render_objects_flat(nodes)}"""

    return body + _render_to_place_block(to_place, by_id)


# ---------- Step 1: region plan (high-level authoring; runs for every region) ---


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
- `is_atomic` (boolean): whether this scene is a single cohesive space or should decompose into distinct subregions

No additional prose, markdown, or code fences.
</output>"""

SYSTEM_ZONE_PLAN = """<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.
</intro>

<role>
You are authoring the plan for ONE region within a larger scene, and deciding whether that region is a single cohesive space or should decompose further into distinct subregions.
</role>

<input>
The user message contains this region's seed prompt, the scene context already in the run, and guidance on how to author the plan and how to decide `is_atomic`.
</input>

<output>
Respond with a single JSON object containing:
- `plan` (string): your region planning paragraph
- `is_atomic` (boolean): whether this region is a single cohesive space or should decompose into distinct subregions

No additional prose, markdown, or code fences.
</output>"""


def render_zone_plan(
    *,
    zone_id: str,
    zone_prompt: str,
    nodes: list[Node],
) -> str:
    """Author the plan for one region. `nodes` is the full scene snapshot so far. Empty list means we're planning the root (no scene exists yet); in that case the root-specific prompt is emitted and `nodes` is unused. For nested regions, the canonical scene tree is injected for context."""
    if not nodes:
        return f"""You are the first step in the SpatialBench pipeline and the step responsible for determining the high-level description and direction of the overall scene. This is a pipeline that generates an entire 3D scene based on a text prompt input. During the generation of the 3D scene, the pipeline breaks down the scene into individual regions to allow downstream steps to recurse into them and focus on each one individually. You sit at the very top of that process: the plan you author here is the seed that every downstream step expands on, and nothing exists in the scene yet beyond the prompt below. write one paragraph that describes a 3D scene imagined from the following prompt, and decide whether this scene should decompose into multiple distinct subregions.

"{zone_prompt}"

<VERY IMPORTANT INSTRUCTIONS>
think deeply about the 3D scene, environment or level you want to build from this provided prompt, and how you can creatively make it stand out enough to WIN against the other submissions. think about the narrative through-line that will help guide and form realistic scene intention - what is this game for, who lives in this house, what is the player trying to achieve in this level, what kind of city is this, and so on. instead of a stoic description that focuses only on the architectural qualities and layout of the scene, the plan you describe should also introduce a high-level story to the scene while allowing downstream planning steps to build on the narrative further. your goal is to write a guiding plan that downstream steps can build upon to complete a cohesive 3D scene for the given prompt that follows the narrative you imagine.

write directly and consider every part carefully. you are only the first, overall planning step - your plan will go through hundreds of further downstream steps where it is expanded on and transformed as the AI pipeline to construct it propagates further planning by depth. define the scene itself, its top-level shape and character enough that the downstream steps have agency over their individual sections while also forming ideas of what to build. 

only the final output of the 3D geometry for the scene itself will be judged once the pipeline is finished; your prompt itself will NEVER be shown to the judges, it will only serve as a base to build upon.

In your plan, DO NOT be overly specific - remember, your prompt will NOT be converted directly into 3D geometry, it will undergo hundreds of expansion and detail steps before reaching any generation steps, so structure your output such that it is a base that the downstream tree of pipeline steps can build upon it. given a prompt for a building, a bad output provides exact instruction on what it looks like; a good prompt defines the narrative premise for the scene, the scope of the environment, the building's character and type, the surrounding environment, points that may implicitly be expanded, and a top-level shape for the scene itself without explicitly shaping the entities that form it.

plan differently based on the prompt given and infer the purpose - e.g for a house, you might plan the aforementioned details for the overall scene scope, general architectural, narrative and character; for a super mario platformer level, you might focus on the narrative section, features, progression, regions, mechanics, etc.; for a top-down swamp frogger level without a specific game mentioned, you might focus on first building out the game's premise internally given the more abstract request, and then establish the world, general layout, character, mechanics, scope, item types, objectives, etc. 

remember, tune specificity based on whether your intent can be inferred by downstream steps. e.g in the super mario level, do not be overly specific - do not scope out all individual platforms, items, etc. but rather the general idea of each part, since the downstream steps have a shared understanding of what a super mario level looks like and the general premise of the game; however, for a more abstract request like the top-down swamp frogger level, the through-line of what you are trying to build cannot be inferred or reconstructed by downstream steps in the pipeline as the specific context for the premise of the game was constructed within your internal reasoning, not exposed to those steps, and thus will be lost as the pipeline propagates, placing the onus for world creation and high-level planning on downstream steps (e.g the immediate next step of planning the specific nature of subregions inferred from your prompt, which was not designed for deciding scene structure/mechanics itself as it lacks the lack full frame and is more there to decompose the scene and figure out spatial relationships between the decomposed subregions, and follows a different, more mechanical heuristic for generation, which means it would not only not spend a lot of time thinking about that, but diverge significantly and perhaps genericly from world you were trying to create in different directions, before handing off to individual planning steps for each region that have more agency over their nature and so on), which means you would need to provide that through-line more explicitly in that scenario where foundational planning is required for the world state due to a lack of shared context (as opposed to a house or established game level).

the output plan should be literal - do not use flowery language. do not describe any abstract quantities like mood, lighting, fog, etc unless they can be converted into concrete 3D geometry. do not reference meta-quantities like the pipeline itself, the scene's 3D nature itself, etc. NEVER MENTION THOSE THINGS. focus on defining the environment intrinsically. remember, this is a full 3D scene, NOT an image - do not define any specific perspective.

define the scene itself, its top-level shape, character, and rough spatial relationships between major parts enough that the downstream steps have agency over their individual sections while also forming ideas of what to build, especially spatially.
</VERY IMPORTANT INSTRUCTIONS>

<region_decomposition>
you must also decide `is_atomic` — whether this scene is a single cohesive space or should decompose into multiple distinct subregions.

the root scene you are planning is a PURELY ABSTRACT META-CONTAINER — it has no walls, floor, ceiling, or geometry of its own. only child regions receive physical enclosures and geometry.

CRITICAL: if the prompt names a SINGLE TANGIBLE ENCLOSURE that needs walls/floor/ceiling (a hotel room, a throne room, a garage, a cockpit, a bathroom), you MUST set is_atomic=false. that enclosure becomes a child region inside this abstract root. marking the root atomic in such cases leaves the scene with no physical enclosure at all.

default to is_atomic=true. set is_atomic=false ONLY when the scene genuinely contains TWO OR MORE distinct regions, each deserving its own dedicated planning and generation pass:
- good decomposition: mansion grounds → house, formal garden, stables (distinct functional regions)
- good decomposition: hotel room → bedroom, bathroom (distinct rooms)
- bad decomposition: island → north end, central mound, south end (arbitrary geography with no distinct identity)
- bad decomposition: bedroom → bed area, dresser area, reading nook (over-fragmented; one cohesive space)

a region is a place large enough to contain multiple objects arranged inside it. a single landmark, monument, centerpiece, or hero prop — no matter how important — is an OBJECT inside a region, not a region of its own.

DO NOT design your prompt around the concept of explicit regional fragmentation; keep this concept of "regions" in mind ONLY for the is_atomic assessment AFTER the base plan is generated.
</region_decomposition>

<thinking>
before ANY output, remember to think HARD and DEEPLY and ALWAYS provide a detailed CoT. NEVER skip the thinking step. think through different creative approaches you might take to what this scene/environment looks like. think deeply through the spatial layout to ensure that everything makes sense - this is a 3D spatial environment benchmark competition after all. 

in the interest of winning, always start by thinking of the overall narrative and premise such that you provide the option for the pipeline to eventually build something truly impressive enough to stand out creatively from all the other LLMs.
</thinking>
{_deepseek_suffix()}"""

    # Nested regions use adapted competitive prompt format
    root = util.find_root(nodes)
    assert root is not None, "nested region planning requires a root node in scope"
    zone_bbox = next((n.bbox for n in nodes if n.id == zone_id), root.bbox)
    context = render_embedded_block(nodes, node_id=zone_id, text="This is the region you are to plan and flesh out from.")

    return f"""You are the step in the SpatialBench pipeline responsible for planning out a particular subregion within the larger overall scene. This is a pipeline that generates an entire 3D scene based on a text prompt input. During the generation of the 3D scene, the pipeline breaks down the scene into individual regions to allow downstream steps to recurse into them and focus on each one individually. You author the plan for one such region, which further downstream steps then expand on, decompose, and populate. 

Here is the overall scene that is being built by the pipeline:

{_root_scene_header(root)}

{render_root_objects(nodes)}

This is the subregion that we are planning:

Subregion name: {zone_id}
Subregion description: "{zone_prompt}"
Subregion dimensions: {util.format_dimensions(zone_bbox)}
Subregion global origin corner: {util.format_global_origin(zone_bbox)}

Here's the scene's subregion tree, with the objects placed in each subregion listed inline as a flat list beneath it, recursing into nested subregions. Each subregion has a plan for how it should be built (or a description of what it is if a plan hasn't been authored for it yet in the pipeline), its dimensions, and a global coordinate marking its origin corner, then a flat list of the objects placed directly inside it, then its nested subregions. Each object carries its description, a `parent` block (its structural anchor — `parent_id`, `parent_relationship_kind`, `parent_dimensions`), and — when it anchors to a peer object rather than directly to its subregion — a `parent_region` (the subregion it belongs to) and `parent_region_dimensions` (that region's size); objects are never nested under one another. Additionally, each subregion and object mentioned will also have a set of local coordinates that define its position relative to its parent (which can be either another region or another object), where the origin is the actual minimum corner of the parent's bounding box.

{context}

Your goal is to elaborate and add to the narrative painted by the ancestor plans through the plan for this region, but also leave sufficient room in your plan for further downstream steps to expand on more using their own agency. what constitutes "sufficient" depends on the specificity of the current region: larger, higher-level regions should have less specificity, while smaller, more constrained regions nearing the atomic level should have more specificity. your prompt will undergo further subregion divisions, expansion, and detail steps before reaching any generation steps, so structure your output as a base that downstream steps can build upon and do not enumerate objects unless this region cannot be subdivided much further.

you should also calibrate your plan's specificity to the scope and nature of this region. a well-understood region type (a bedroom, a kitchen, a garden) needs less foundational planning because downstream steps share an understanding of what that space looks like and what belongs in it. a region with novel character or a creative premise that cannot be inferred from its prompt and ancestor context alone needs more explicit through-line — downstream steps that further decompose and populate this region will not reconstruct creative intent that isn't present in your plan. furthermore, a tightly-constrained region that cannot be broken down into further subregions would require more specificity in terms of object enumeration as you are the final planning step before the actual object list gets generated by a downstream step.

<VERY IMPORTANT INSTRUCTIONS>
think deeply about what this region is and how you can make it creatively compelling. every region of the scene contributes to the final build that judges evaluate, and the quality of your plan here directly shapes how impressive this part of the scene will be.

write directly and consider every part carefully. you are the planning step for this region - your plan will go through further downstream steps where it is expanded on and transformed as the pipeline propagates further planning by depth. define this region's character, spatial shape, and what makes it distinctive enough that downstream steps have agency over the specifics while building coherently.

only the final output of the 3D geometry will be judged once the pipeline is finished; your prompt itself will NEVER be shown to the judges, it will only serve as a base to build upon for this region. thus, making the prompt dramatic and sound impressive will only have a contradictory effect, since it will confuse downstream steps when generation actually happens as they don't understand flowery language.

your prompt should focus on just the current region: it can reference other defined regions and objects as context, but do not overtly describe them apart from using them as an anchor for relative positioning.

think from the perspective of a narrative through-line. ground the region in concrete use: what physically happens in this space, not in atmosphere or symbolism. you can and should use the plans of ancestor regions mentioned above to help you in coming up with this as they provide additional, already-established context of the scene.

In the plan you write, do not use flowery language. do not describe any abstract quantities like mood, lighting, fog, etc unless they can be converted into concrete 3D geometry. do not reference meta-quantities like the pipeline itself or your role. focus on defining the region intrinsically. this is a 3D environment, not an image - do not define any specific perspective. do not mention meta pipeline-related terms, such as "already-placed" or "existing". the prompt should be direct, accurately describe the scene, and every word should be useful for downstream generation and further processing, grounded concretely in what the scene is and with no relation to the pipeline itself.

define this region's shape, character, and rough spatial relationships between its major parts enough that downstream steps have agency over their individual sections while forming ideas of what to build, especially spatially.
</VERY IMPORTANT INSTRUCTIONS>

<region_decomposition>
you must also decide is_atomic — whether this region is a single cohesive space or should decompose into multiple distinct subregions.

default to is_atomic=true. set is_atomic=false ONLY when the region genuinely contains TWO OR MORE distinct regions, each deserving its own dedicated planning and generation pass:

    good decomposition: mansion grounds → house, formal garden, stables (distinct functional regions)
    good decomposition: hotel room → bedroom, bathroom (distinct rooms)
    bad decomposition: island → north end, central mound, south end (arbitrary geography with no distinct identity)
    bad decomposition: bedroom → bed area, dresser area, reading nook (over-fragmented; one cohesive space)

a region is a place large enough to contain multiple objects arranged inside it. a single landmark, monument, centerpiece, or hero prop — no matter how important — is an OBJECT inside a region, not a region of its own.
</region_decomposition>

<thinking>
before ANY output, think HARD and DEEPLY and provide a detailed CoT. think through the creative direction for this region within the context of the larger scene. think through spatial layout and how everything fits together physically. think about the constructed narrative of the ancestor plans and how this particular region could add onto it as a detail. think about what would make this region genuinely impressive and memorable as part of a winning build.
</thinking>
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


# ---------- Step 3: region decompose (atomic vs subregions; runs after plan) ----


class ChildNodeSpec(BaseModel):
    """A single child node (subregion or object) emitted by a decomposition
    LLM call.

    Fields:
      * `parent` — the structural anchor id: the supporter (object or
        frame) this node physically rests on, or the containing region
        when there's no physical supporter (floating objects, frames
        themselves, subregions inside a parent region).
      * `parent_relationship_kind` — how this node relates to its `parent`. MUST be
        one of ON / ATTACHED / IN:
          - ON: rests on parent's outward surface (most often top face).
          - ATTACHED: flush against any face of the parent (wall mount,
            ceiling fixture, magnet, hanger).
          - IN: contained inside the parent's volume or footprint
            (subregion inside a region, fish inside a tank, cloud inside a
            sky region, embedded particle).
        BESIDE / ABOVE / BELOW are reserved for sibling/peer hints in
        `relationships` and are NOT valid here.
      * `placement` — prose describing where this node sits, with
        precise positioning (centered, edge-aligned, two-thirds along,
        etc.). The bbox-resolution step uses this verbatim to choose
        coordinates.
      * `relationships` — optional list of spatial relationships to
        other already-placed nodes that assist the downstream spatial
        resolver. Each entry is a Relationship with `target` (the peer's
        id) and `kind` (ON, BESIDE, ABOVE, BELOW, ATTACHED, IN) —
        categorical, no anchor point. Do NOT repeat the parent here.
        Empty list is fine when the placement only references the parent.
    """

    # Two LLM-facing field names are aliased so the schema/wire names match the
    # prompt text, while the Python attributes stay the names the shared
    # divider/generation/topology + Node use across every version:
    # `relationships` -> attr `referenced_ids`, `parent_relationship_kind`
    # -> attr `parent_kind`. `populate_by_name` lets committed-event replay
    # (which dumps by attribute name) round-trip back in.
    model_config = ConfigDict(populate_by_name=True)

    id: str
    prompt: str
    parent: str
    parent_kind: ParentRelationshipKind = Field(alias="parent_relationship_kind")
    placement: str
    referenced_ids: list[Relationship] = Field(
        default_factory=list, alias="relationships"
    )
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
You are deciding the structural decomposition of one region of the scene — how it splits into its top-level subregions. Each subregion you emit becomes its own planning and recursive decomposition branch downstream.
</role>

<input>
The user message contains this region's id, prompt, bbox, and plan, plus the scene context (every other region/object already placed in the run, which you may reference by id).
</input>

<output>
Respond with a single JSON object containing:
- `children` (list): the subregions this region decomposes into. Each child has:
  - `id` (string): unique within the entire scene
  - `prompt` (string): a short seed describing what this child region is
  - `parent` (string): the id of this child's structural parent. For a top-level subregion, this is the literal id of the region being decomposed (the value labelled "Parent region id" in the user message — e.g. if the user message says `Parent region id: 'living_room'`, emit `"parent": "living_room"`, NOT the string "PARENT_ID" or any other placeholder). For a child anchored to an earlier sibling in this call, use that sibling's id verbatim.
  - `parent_relationship_kind` (string): how this child anchors to its `parent`. Exactly one of `ON` (rests on parent's outward surface), `ATTACHED` (flush against any face of the parent), or `IN` (contained inside the parent's volume / footprint). `BESIDE` / `ABOVE` / `BELOW` are NOT valid here — they are peer hints, reserved for `relationships`.
  - `placement` (string): one string describing where this child sits within the scene relative to other regions and objects around it. This placement should NOT contain any precise coordinates, which will all be resolved later through a downstream solver step; it should only be a semantic spatial description of where the subregion is located. Think very deeply about where each region should lie spatially and designing the placement string for it.
  - `relationships` (list of {target, kind}): OPTIONAL spatial relationships to other already-placed nodes to assist in the downstream spatial resolver step. Think very deeply and precisely about what each of these relationships with other objects is spatially. Each entry has a `target` (the peer's id) and a `kind` — one of ON, BESIDE, ABOVE, BELOW, ATTACHED, IN. Do NOT repeat the parent here. Empty list is fine when the placement only references the parent.
  - `proxy_shape` (string | null): BOX / SPHERE / CAPSULE / HEMISPHERE if the region's silhouette is non-rectangular, otherwise null/omitted.

No additional prose, markdown, or code fences.
</output>"""


def render_zone_decompose(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_plan: str,
    nodes: list[Node],
) -> str:
    """Decompose one region into top-level subregions. `nodes` is the full
    scene snapshot; the target region must be present in it with its plan
    already set."""
    
    root = util.find_root(nodes)
    assert root is not None, "region decomposition requires a root node in scope"
    zone_bbox = next((n.bbox for n in nodes if n.id == zone_id), root.bbox)
    context = render_embedded_block(nodes, node_id=zone_id, text="This is the region you are to break down and decompose.")

    return f"""You are the step in the SpatialBench pipeline responsible for breaking down a given region into its top-level subregions. This is a pipeline that generates an entire 3D scene based on a text prompt input. During the generation of the 3D scene, the pipeline breaks down the scene into individual regions to allow downstream steps to recurse into them and focus on each one individiually - you are the step that performs and determines this subdivison. This is the overall scene we are trying to generate:

{_root_scene_header(root)}

{render_root_objects(nodes)}

{f"""
You are subdividing the scene itself (root) into its first set of top-level subregions based on its overall plan.
""" if zone_id == root.id else f"""This is the plan for the subregion within this overall scene that you are to break down and decompose:

Subregion name: {zone_id!r}
Subregion description: "{zone_prompt}"
Subregion plan: "{zone_plan}"
Subregion dimensions: {util.format_dimensions(zone_bbox)}
Subregion global origin corner: {util.format_global_origin(zone_bbox)}
Parent region id: {zone_id!r}
"""}

Here is the list of other subregions that have been planned for this scene so far. Each subregion has a plan for how it should be built (or a description of what it is if a plan hasn't been authored for it yet in the pipeline), its dimensions, and a global coordinate marking its origin corner, with the objects placed inside it listed inline beneath it. Additionally, each subregion mentioned will also have a set of local coordinates that define its position relative to its parent region, where the origin is the actual minimum corner of the parent's bounding box.

{context}

<IMPORTANT_INSTRUCTIONS>

<REGION_SPLITTING_GUIDANCE>
A subregion is a portion of the parent whose bounding box sits within the parent bounding box and can be treated individually as its own region due to a combination of physical and narrative reasons. The goal of subregions is to guide downstream steps so they can focus on just one region and use the rest as context, allowing for more fleshed-out designs and scenes.

Subregions can keep decomposing into more subregions recursively in subsequent passes, or end there as atomic leaves if that is appropriate. so always decompose at the TOP MOST LEVEL of the current region — e.g. for a house scene with backyard, driveway, and house, do not skip straight to backyard-pool region, backyard-grass region, house-basement, house-first-floor, etc.; decompose into "the house", "the backyard", "the driveway" as top-level children, and let the next recursion split the house into floors and the backyard into pool and grass. the same principle holds everywhere: emit only the regions that exist at THIS level of the hierarchy, and trust the recursive planning + decompose passes underneath each of them to handle the next layer down.
</REGION_SPLITTING_GUIDANCE>

Think very intricately and spatially about how this region splits, being careful and wary about overlapping regions. Your goal is to reason a subregion decomposition layout that fits the narrative presented by the scene plan given above as well as the additional plans of ancestor scenes in the scene context section given below, while paying attention to the semantic relationships between the subregions. The subregions presented should each flesh out the guiding narrative further in some way, carrying relevant ideas from previously defined plans (in the scene context below) while also introducing some new ones without being contradictory.

The seed prompt you output for each subregion should be a 1-2 sentences long description that explains the subregion's shape, character, and the new narrative ideas presented by this subregion, if any. Be concrete about its description while leaving room for this prompt to be a seed for a more detailed plan. The prompt should be succinct without mentioning going overly into detail on the subregion's contents.

Keep the prompt tight: the goal is not to plan out the subregion's contents, but to establish its character as a piece of the larger scene as a whole.

The placement text you output for each subregion should only be a semantically spatial description of where the subregion is located - actual coordinates will get resolved in a downstream step. Think very deeply to figure out the parametric spatial placement of every subregion, and reference only regions/objects that are currently within the scene or part of your own proposed subregions (do not reference items that do not exist within the scene). 
</IMPORTANT_INSTRUCTIONS>

{_deepseek_suffix()}"""


# ---------- Step 4: region bbox batch resolution (all siblings at once) -------


class BboxAssignment(BaseModel):
    id: str
    bbox: BoundingBox


class BboxBatchOutput(BaseModel):
    assignments: list[BboxAssignment] = Field(default_factory=list)


SYSTEM_ZONE_BBOX_BATCH = f"""<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.
</intro>

<role>
You are a constraint solver placing the sibling child regions inside a parent region — deriving each child's axis-aligned bounding box from its placement string, parent_relationship_kind, relationships, and the parent's dimensions.
</role>

<input>
The user message contains the parent region's id, plan, and dimensions; the scene context; and a list of child subregions to place. Each child has `id`, `prompt`, `proxy_shape`, `parent`, `parent_relationship_kind`, `parent_dimensions`, `placement`, and `relationships` — plus `parent_region`/`parent_region_dimensions` whenever the child is anchored to a peer rather than directly to the region being decomposed (naming the region it still belongs to). A child's parent may be the region being decomposed, an existing node, or another child in this same batch.
</input>

<output>
Respond with a single JSON object matching the schema: one `assignment` per child (id + bbox). Each child's bbox must be in THAT CHILD'S PARENT's local frame — origin (0,0,0) is the parent's minimum corner; axes follow the canonical front view (+X right, +Y up, +Z front, -Z back). The parent's dimensions are provided for each child — use them as the bounding extent. A child flush against its parent's minimum corner has origin (0,0,0); a child resting on its parent's floor at the parent's centre has origin near (parent_width/2, 0, parent_depth/2) minus the child's footprint. The working unit is ALWAYS meters: every coordinate and dimension you emit is in meters — the same unit as the parent dimensions you are given, which are also in meters (never centimeters) — to centimeter precision (multiples of 0.01 m). Emit a signed `dimensions` vector from an `origin` vertex; sign chooses expansion direction along each axis. Emit exactly one assignment per requested child id — no extras, no omissions.

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
    """Place every sibling child region of `parent_id` in one shot. `nodes` is
    the scene snapshot; the children-to-place are listed beneath the context
    (bbox blank — that is the LLM's job)."""
    root = util.find_root(nodes)
    assert root is not None, "region bbox resolution requires a root node in scope"
    by_id = {n.id: n for n in nodes}
    context = render_embedded_block(nodes, node_id=parent_id, text="This is the region whose subregions you are to place.")

    return f"""You are the step in the SpatialBench pipeline responsible for resolving the bounding boxes of a region's subregions. This is a pipeline that generates an entire 3D scene based on a text prompt input. During the generation of the 3D scene, the pipeline breaks down the scene into individual regions to allow downstream steps to recurse into them and focus on each one individually. An upstream step has already chosen this region's subregions and written a semantic placement for each. Your goal is to determine the concrete coordinates and dimensions of each subregion relative to their larger parent region to deterministically place them in the scene.

Here is the overall scene that is being built by the pipeline:

{_root_scene_header(root)}

{render_root_objects(nodes)}

This is the region whose subregions you are placing:

Parent region name: {parent_id!r}
Parent prompt: "{parent_prompt}"
Parent region plan: "{parent_plan}"
Parent region dimensions: {util.format_dimensions(parent_bbox)}
Parent region global origin corner: {util.format_global_origin(parent_bbox)}

Here is the context of what else has already been placed in the scene — every subregion and object placed so far. Each carries its dimensions, a global origin corner, and local coordinates relative to its parent.

{context}

Here is the list of subregions you must place:

{_render_to_place_block(children, by_id, parent_zone=parent_id, show_orientation=False)}

Your job is to produce a bounding box for those subregions. Each subregions's bbox coordinates must be relative to that object's parent's local frame — origin (0,0,0) is the parent's minimum corner. When determining the bounding boxes, you should remain loyal to each subregion's provided placement text, but also use your best judgment and think spatially about how the bounding boxes you come up with interact with the already present bounding boxes of other objects/regions inside the scene, as well as the bounding boxes of the other subregions in your output list. You are not simply a translator that translates a placement text into coordinates - you should reason spatially to determine what bounding box coordinates make sense.

{_deepseek_suffix()}"""


# ---------- Step 5: object decomposition (Phase 2) --------------------------


class ObjectSpec(ChildNodeSpec):
    """A single object in a region. Identical shape to ChildNodeSpec.
    The structural parent (`parent` field) may be the enclosing region,
    a frame, an earlier-placed peer, or another object listed in the
    same decomp call. Secondary relationships (`relationships`)
    capture additional spatial connections — sibling alignment, the
    wall a painting hangs against, etc."""


class BoundExistingFrame(BaseModel):
    """Audit entry for the encapsulating step's BINDING CONTRACT.

    When a structural element named in the region's plan is already
    satisfied by an existing peer (option (b) of the contract), the LLM
    records the binding here instead of emitting a new shell node. These
    entries do NOT become Nodes and do NOT flow into bbox-resolution or
    any downstream step — they're a per-call audit trail proving the
    LLM made the binding choice deliberately rather than silently
    skipping the plan-named element.
    """

    plan_element: str  # noun phrase from the region plan (e.g. "rear partition wall")
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
    # Encapsulating-only gate: if False, the region needs no bounding
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
  - `parent_relationship_kind` (string): exactly one of `ON` (rests on parent's outward surface), `ATTACHED` (flush against any face of the parent — wall/ceiling mounts, embedded fittings, shell-frame-to-region), or `IN` (contained inside the parent's volume / footprint with no specific contact face). `BESIDE` / `ABOVE` / `BELOW` are NOT valid here.
  - `proxy_shape` (string | null): BOX / SPHERE / CAPSULE / HEMISPHERE if the object's silhouette is non-rectilinear, otherwise null/omitted.
  - `orientation` (int): world-frame yaw about +Y in degrees. Exactly one of -180, -135, -90, -45, 0, 45, 90, 135, 180. `0` = front faces +Z (toward viewer), `90` = front faces +X (to the right), `180` = front faces -Z (away), `-90` = front faces -X (to the left); positive degrees swing the front toward +X (right), negative toward -X (left). Use 0 for symmetric objects.
  - `placement` (string): one string describing where this object sits within the scene relative to its parent and the other regions and objects around it. This placement should NOT contain any precise coordinates, which will all be resolved later through a downstream solver step; it should only be a semantic spatial description of where the object is located. Think very deeply about where each object should lie spatially and designing the placement string for it.
  - `relationships` (list of {{target, kind}}): OPTIONAL spatial relationships to other already-placed nodes to assist in the downstream spatial resolver step. Think very deeply and precisely about what each of these relationships with other objects is spatially. Each entry has a `target` (the peer's id) and a `kind` — one of ON, BESIDE, ABOVE, BELOW, ATTACHED, IN. Do NOT repeat the parent here. Empty list is fine when the placement only references the parent.
- `bound_existing` (list): used only by the encapsulating step (anchor and negative-space leave this empty). Each entry has `plan_element` (the plan's noun phrase verbatim) and `peer_id` (the id of an existing node that already satisfies it).
- `bounding_required` (bool): used only by the encapsulating step (anchor and negative-space leave this as the default `true`). Set to `false` when the region needs no bounding perimeter at all — `objects` is then ignored downstream even if non-empty. Set to `true` when at least one bounding object is being emitted.

No additional prose, markdown, or code fences.
</output>

<additional_context>
{PROXY_SHAPE_DOC}

{NO_EPHEMERA_DOC}

{SOLID_OCCUPANCY_DOC}
</additional_context>"""


SYSTEM_ANCHOR_DECOMP = f"""<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.
</intro>

<role>
You are enumerating the objects of an atomic leaf region — the objects that make up and fill this region.
</role>

<input>
The user message contains this region's id, description, and plan, plus the scene context (every region and object already placed in the run, which you may reference by id).
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
The user message contains this region's id, description, and plan, plus the scene context (every region and object already placed in the run, which you may reference by id).
</input>

{_OBJECT_DECOMP_TAIL}"""


SYSTEM_NEGATIVE_SPACE_DECOMP = f"""<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.
</intro>

<role>
You are filling the ambient, connective, interstitial space between the named regions and objects of the scene — the layer of small instanced solids that binds the scene into a continuous world (lilypads across swamp water, grass tufts over a meadow, scattered stones across a plain). Every named region has already been decomposed and populated; what remains is the layer the named regions do not own.
</role>

<input>
The user message contains this region's id, description, and plan, plus the scene context (every region and object already placed in the run, which you may reference by id).
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

Produce a NEW decomposition that fixes every listed reason. In particular, ensure every object's `parent` field is set to a valid existing id (the supporter or containing region that anchors it), and every `relationships` entry has a `target` that exists in the scene context."""


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
    zone_bbox = next((n.bbox for n in nodes if n.id == zone_id), root.bbox)
    context = render_embedded_block(nodes, node_id=zone_id, text="This is the subregion you are to generate a list of anchor objects for.")
    return f"""You are the step in the SpatialBench pipeline responsible for determining the list of objects that define a certain subregion. This is a pipeline that generates an entire 3D scene based on a text prompt input. During the generation of the 3D scene, the pipeline breaks down the scene into individual regions to allow downstream steps to recurse into them and focus on each one individually. This subregion is an atomic leaf that the pipeline has decided not to subdivide any further, so instead of splitting it again you are the step that fills it with the objects that define it.

Here is the overall scene that is being built by the pipeline:

{_root_scene_header(root)}

{render_root_objects(nodes)}

This is the subregion we are generating the objects for:

Subregion name: {zone_id!r}
Subregion description: "{zone_prompt}"
Subregion plan: "{zone_plan}"
Subregion dimensions: {util.format_dimensions(zone_bbox)}
Subregion global origin corner: {util.format_global_origin(zone_bbox)}

Here is the list of other subregions that have been planned for this scene so far. Each subregion has a plan for how it should be built (or a description of what it is if a plan hasn't been authored for it yet in the pipeline), its dimensions, and a global coordinate marking its origin corner, with the objects placed inside it listed inline beneath it. Additionally, each subregion mentioned will also have a set of local coordinates that define its position relative to its parent region, where the origin is the actual minimum corner of the parent's bounding box.

{context}

This region is the lowest possible breakdown level: no other subregions can exist within, so it is defined by the objects you are responsible for generating. Although the above plan is specific, the objects that the plan mentions may not be the end all be all - you should extrapolate meaning from the plan and higher-level ideas communicated in the ancestor chain to populate and style the objects in the list you generate based on a narrative understanding of the scene. Each object should be treated atomically, in the sense that collections of objects should be broken down into individual objects that allow more granular, precise positioning by you instead of relying on the downstream 3D model generation step to form the complex arrangement you want. Object count is not a concern: always split pairs, groups, or collections of objects into individual objects positioned in the way you deem fit.

<output_guidance>
The concept of a parent should be grounded in a concrete, physical relationship, not a conceptual one. A cantilevered object would be parented to the surface or wall it's cantilevered to with relationship type 'ATTACHED', not parented to the floor below with relationship 'ON'. If no physical relationship is found with another object or frame, the relationship should be of type 'IN', and parented to the region itself.

Each anchor object in your resultant list's parent_relationship_kind (one of ON / ATTACHED / IN) describes how the object physically anchors to that parent: `ON` for resting on an outward surface, `ATTACHED` for wall/ceiling/face mounts, `IN` for free containment inside the parent's volume — BESIDE/ABOVE/BELOW are NOT valid here.

The placement string is a one sentence description of where the object is placed in terms of the spatial layout of the region its in relative to its containing region or other objects present in the region. This is not meant to be an exact, coordinate-level descriptor, but rather a more semantic description of how the object sits spatially relative to its surroundings.

The relationships list for each object is an optional list of `{{target, kind}}` for spatial relationships to other nodes that assist the downstream solver; kind may use ON/BESIDE/ABOVE/BELOW/ATTACHED/IN). This will be used in conjunction with the placement string you provide by the downstream coordinate solver to calculate the precise coordinates of how to place the object in the scene.

Respect the scene context: anchor onto any ground/shell peer already placed by the encapsulating pass, do not duplicate geometry another region has already emitted. Anchor objects are expected to live primarily inside this region, but their bboxes MAY protrude modestly outside the region bbox when narratively justified — the object remains semantically part of this region even though its geometry overhangs. Do not use this as license to claim airspace far from the region or to volumetrically intersect another region's load-bearing geometry. In your final output, each object's placement text should be direct and parametric. Avoid flowery language that states the narrative purpose of the positioning. Do not state the abstract reason of the positioning, only details that ground the position concretely. The position description should be absolute and succinct, leaving no creative liberty for the downstream constraint solver.
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
    zone_bbox = next((n.bbox for n in nodes if n.id == zone_id), root.bbox)
    context = render_embedded_block(nodes, node_id=zone_id, text="This is the region you are to decide whether a boundary is needed for, and if so, what objects form that boundary")
    return f"""You are the step in the SpatialBench pipeline responsible for determining whether a perimeter is needed for the given subregion, and if so, what that perimeter is made up of. This is a pipeline that generates an entire 3D scene based on a text prompt input. During the generation of the 3D scene, the pipeline breaks down the scene into individual subregions to allow downstream steps to recurse into them and focus on each one individually. Each region within the larger scene may or may not require objects to form a boundary or partial boundary around it - it is your role to decide whether it is absolutely required for a given subregion, and if it is, then generate a list of bounding geometry elements that form this boundary.

Here is the overall scene that is being built by the pipeline:

{_root_scene_header(root)}

{render_root_objects(nodes)}

This is the subregion that we are deciciding needs a boundary or not and generating boundary objects for if so:

Subregion name: {zone_id}
Subregion description: "{zone_prompt}"
Subregion plan: "{zone_plan}"
Subregion dimensions: {util.format_dimensions(zone_bbox)}
Subregion global origin corner: {util.format_global_origin(zone_bbox)}

Here's the list of other subregions that have been planned for this scene so far, with the objects placed inside each subregion listed inline beneath it. Each subregion has a plan for how it should be built (or a description of what it is if a plan hasn't been authored for it yet in the pipeline), its dimensions, and a global coordinate marking its origin corner. Each object carries its description, a `parent` block (its structural anchor — `parent_id`, `parent_relationship_kind`, `parent_dimensions`), a `parent_region` (the subregion it belongs to), and `parent_region_dimensions` (that region's size). Additionally, each subregion and object mentioned will also have a set of local coordinates that define its position relative to its parent (which can be either another region or another object), where the origin is the actual minimum corner of the parent's bounding box.

{context}

Think very carefully about whether the given region actually needs any bounding objects. Reason about the structure of the region, whether it is closed vs. open, and the narrative in its plan. Not every region necessarily needs to have any bounding objects: only do so when it absolutely makes sense to do so.

<IMPORTANT_INSTRUCTIONS_ONLY_IF_BOUNDING_NEEDED>
the list of objects you output, if any, should work together to form a cohesive perimeter or partial perimeter of any arbitrary shape. The purpose of this list of objects is to form a sense of boundary for the given region in every dimension that makes sense based on its plan - perimeter does not necessarily mean in the horizontal axis but in all possible directions, including the vertical direction (e.g. bases, covers). In this case, perimeter or boundary does not automatically imply physically bounding the region on all sides (though depending on the region's plan, that may be the case). You should think carefully and reason spatially about what objects should go in this list to form a well-defined, physically and narratively reasonable boundary for the region. You are in a canvas that contains only the objects listed below in the scene context - do not assume any models, foundations, ground, etc. exist outside the provided scene context.

Object should be individualistic - composite objects should be broken down into individual or partial objects (abstract fragments that are meant to combine into a more complex object) and placed accordingly, allowing for more granular control of the region's boundary. Objects and partial objects can be stacked, strung, pieced to form larger, cohesive sections for the perimeter. there is no limit on the number of objects in your output list - always prefer individual objects placed close to each other over a single composite object with a prompt to generate them together at once. if the region calls for it, we can have as high fidelity of a perimeter as we want, the number of objects you can output is truly unbounded. if a dense perimeter makes sense for the given region, then make it dense - as many objects as you see fit, their bounding boxes right next to each other. You are in control - do not rely on downstream generation steps to output composite geometry: organize your list of objects so that you are in direct control of positioning to form that composite geometry yourself using the individual partial objects.

When generating a list of objects, keep in mind connectives between this region and others, in all directions; using objects and partial objects to leave free space, embed semantically relevant transition objects, construct composite structures, etc. The space should be realistic and traversable. Also keep in mind collisions and overlaps with existing objects and planned regions' bounding boxes.

If you need to leave a hole/gap or embed other objects within a greater bounding section for any purpose, piece objects and partial objects together like a puzzle around the gap or embed. For example, a door or window embedded within a wall, a roofed forest underpass, an ice fishing hole, a concave crater in the ground, etc. Do not rely on downstream steps to generate such kinds of complex geometry or abstract shapes - you are responsible for owning this step and using partial objects and piecing them together to achieve the kind of geometry you want.

Pay special attention to the context provided in the plans of other regions in the scene, and use it to imagine realistically navigating the region as part of the larger scene. Pay especial attention to the traversal from {zone_id} to its neighboring regions and how the boundary formed by the objects you generate support that. Use this thinking to guide you in the generation and placement of your list of objects or partial objects.

be wary of duplicate geometry - for two neighboring regions separated by some sort of divider, it is only necessary to generate the divider once. study the provided scene context to determine if generating something is necessary.

This region may be broken down into smaller subregions by a downstream step, ONLY generate objects relevant to THIS region, trust downstream steps to generate bounding objects for subregions if those bounding objects are more relevant there.

Ensure the space makes perfect sense and is cohesive with the goal; ex. nothing is missing. 

</IMPORTANT_INSTRUCTIONS_ONLY_IF_BOUNDING_NEEDED>

<output_guidance>
Each boundary object in your resultant list's parent_relationship_kind (one of ON / ATTACHED / IN) describes how the object physically anchors to that parent: `ON` for resting on an outward surface, `ATTACHED` for wall/ceiling/face mounts, `IN` for free containment inside the parent's volume — BESIDE/ABOVE/BELOW are NOT valid here.

The placement string is a one sentence description of where the object is placed in terms of the spatial layout of the region its in relative to its containing region or other objects present in the region. This is not meant to be an exact, coordinate-level descriptor, but rather a more semantic description of how the object sits spatially relative to its surroundings.

The relationships list for each object is an optional list of `{{target, kind}}` for spatial relationships to other nodes that assist the downstream solver; kind may use ON/BESIDE/ABOVE/BELOW/ATTACHED/IN). This will be used in conjunction with the placement string you provide by the downstream coordinate solver to calculate the precise coordinates of how to place the object in the scene.

Output bounding_required = False if no bounding objects are needed. Otherwise, set bounding_required = True and objects to be the list of bounding objects.
</output_guidance>

{_render_retry_block(prior_attempts)}
{_deepseek_suffix()}"""


def render_negative_space_decomp(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_plan: str,
    nodes: list[Node],
    prior_attempts: list[tuple[list[ObjectSpec], str]] | None = None,
) -> str:
    root = util.find_root(nodes)
    assert root is not None, "negative-space decomposition requires a root node in scope"
    zone_bbox = next((n.bbox for n in nodes if n.id == zone_id), root.bbox)
    context = render_embedded_block(nodes, node_id=zone_id, text="This is the region whose interstitial negative space you are filling.")
    return f"""You are the step in the SpatialBench pipeline responsible for filling in the negative space of a given subregion. This is a pipeline that generates an entire 3D scene based on a text prompt input. During the generation of the 3D scene, the pipeline breaks down the scene into individual subregions to allow downstream steps to recurse into them and focus on each one individually. When a region subdivides into subregions, there may be negative space left between the subregions that need to be filled in. Your role is to come up with a list of objects to fill this negative space for one such region in the larger overall scene.

    Here is the overall scene that is being built by the pipeline:

{_root_scene_header(root)}

{render_root_objects(nodes)}

This is the region you are to come up with a list of objects to fill its negative space for:

    Region name: {zone_id}
    Region prompt: {zone_prompt}
    Region plan: {zone_plan}
    Region dimensions: {util.format_dimensions(zone_bbox)}
    Region global origin corner: {util.format_global_origin(zone_bbox)}

The following is the full scene context in embedded form — every subregion with the objects placed inside it listed inline beneath it. The region you are filling ({zone_id}) is marked in the tree; fill the interstitial gaps between its direct subregions and the objects already placed:

{context}

The primary purpose of these negative space objects is to make the scene feel coherent and cohesive, filling in the gaps between subregions or objects. As such, for each negative space to fill in, it is imperative to analyze the existing placed objects and regions surrounding it to create a smooth filling that does not look out of place. Each object you generate should be individualistic - do not rely on the downstream 3D model generation step to be able to parse complex arrangements of objects or objects of composite shape properly; you are instead responsible for breaking what you want to add into discrete, self-contained partial objects, each emitted as its own entry in the list with its own placement text. This will allow the downstream step responsible for coming up with the concrete bounding box locations for each object to properly lay out the partial objects in the exact way you want.

<output_guidance>
Each object in your resultant list's parent_relationship_kind (one of ON / ATTACHED / IN) describes how the object physically anchors to that parent: `ON` for resting on an outward surface, `ATTACHED` for wall/ceiling/face mounts, `IN` for free containment inside the parent's volume — BESIDE/ABOVE/BELOW are NOT valid here.

The placement string is a one sentence description of where the object is placed in terms of the spatial layout of the region its in relative to its containing region or other objects present in the region. This is not meant to be an exact, coordinate-level descriptor, but rather a more semantic description of how the object sits spatially relative to its surroundings.

The relationships list for each object is an optional list of `{{target, kind}}` for spatial relationships to other nodes that assist the downstream solver; kind may use ON/BESIDE/ABOVE/BELOW/ATTACHED/IN). This will be used in conjunction with the placement string you provide by the downstream coordinate solver to calculate the precise coordinates of how to place the object in the scene.

</output_guidance>

{_render_retry_block(prior_attempts)}
{_deepseek_suffix()}"""


# ---------- Step 6: object bbox resolution ----------------------------------


SYSTEM_OBJECT_BBOX_BATCH = f"""<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.
</intro>

<role>
You are a constraint solver placing a list of objects in the scene — deriving each object's axis-aligned bounding box from its placement string, parent_relationship_kind, relationships, the parent's dimensions, and peer geometry.
</role>

<input>
The user message contains the region id/prompt/plan/dimensions, a list of objects to place (each with `id`, `prompt`, `proxy_shape`, `orientation`, `parent`, `parent_relationship_kind`, `parent_dimensions`, `placement`, `relationships`, plus `parent_region`/`parent_region_dimensions` whenever the object is anchored to a peer rather than directly to the region being resolved — naming the region it still belongs to), and the scene context of peers already placed. Each peer's bbox is expressed relative to THAT PEER'S OWN parent's minimum corner (origin (0,0,0) = parent's min corner). Use siblings (peers sharing the same parent as the object you are placing) for direct spatial reasoning; peers under different parents provide broader scene context.
</input>

<output>
Respond with a single JSON object matching the schema: one `assignment` per object (id + bbox). Each object's bbox must be in THAT OBJECT'S PARENT's local frame — origin (0,0,0) is the parent's minimum corner; axes follow the canonical front view (+X right, +Y up, +Z front, -Z back). The parent's dimensions are provided for each object — use them as the bounding extent. The working unit is ALWAYS meters: every coordinate and dimension you emit is in meters — the same unit as the parent dimensions you are given, which are also in meters (never centimeters) — to centimeter precision (multiples of 0.01 m). Emit a signed `dimensions` vector from an `origin` vertex; sign chooses expansion direction along each axis. Emit exactly one assignment per requested object id — no extras, no omissions.

No prose, no markdown, no code fences.
</output>

<additional_context>
{PROXY_SHAPE_DOC}

{SOLID_OCCUPANCY_DOC}

{ORIENTATION_DOC}
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
    """Place every object specified by `objects` in one shot. `nodes` is the
    scene snapshot; the objects-to-place are listed beneath the context (bbox
    blank — that is the LLM's job)."""
    root = util.find_root(nodes)
    assert root is not None, "object bbox resolution requires a root node in scope"
    by_id = {n.id: n for n in nodes}
    context = render_embedded_block(nodes, node_id=zone_id, text="This is the subregion whose objects you are to place.")

    return f"""You are the step in the SpatialBench pipeline responsible for resolving the bounding boxes of a subregion's objects. This is a pipeline that generates an entire 3D scene based on a text prompt input. During the generation of the 3D scene, the pipeline breaks down the scene into individual regions to allow downstream steps to recurse into them and focus on each one individually. Once a region has been populated with objects and a semantic placement written for each, you turn those placements into concrete axis-aligned bounding boxes, all of them in one shot. It is your job to parse the placement texts provided and turn them into concrete coordinates relative to the object's parent.

Here is the overall scene that is being built by the pipeline:

{_root_scene_header(root)}

{render_root_objects(nodes)}

This is the subregion whose objects you are placing:

Subregion name: {zone_id}
Subregion description: "{zone_prompt}"
Subregion plan: "{zone_plan}"
Subregion dimensions: {util.format_dimensions(zone_bbox)}
Subregion global origin corner: {util.format_global_origin(zone_bbox)}

Here is the context of what is already in the scene — every subregion and object placed so far. Each carries its dimensions, a global origin corner, and local coordinates relative to its parent.

{context}

Here is the list of objects you must place:

{_render_to_place_block(objects, by_id, parent_zone=zone_id)}

Your job is to produce a bounding box for those objects. Each object's bbox must be relative to that object's parent's local frame — origin (0,0,0) is the parent's minimum corner. When determining the bounding boxes, you should remain loyal to each object's placement text, but also use your best judgment and think spatially about how the bounding boxes you come up with interact with the already present bounding boxes of other objects/regions inside the scene, as well as the bounding boxes of the other objects in your output list. You are not simply a translator that translates a placement text into coordinates - you should reason spatially to determine what bounding box coordinates make sense.

When determining the bounding box coordinates and dimensions of each object, you should also think about the object's orientation (shown in the list above in degrees of yaw about the vertical +Y axis). It turns the object's front against the one global frame — 0 faces +Z (toward the viewer), +90 faces +X (to the right), -90 faces -X (to the left), 180 faces -Z (away). The box you assign stays axis-aligned and is filled by the object after it is turned, so size it to the turned shape: a +/-90 yaw swaps the object's width and depth, so give the box the object's depth as its X extent and its width as its Z extent (a box whose proportions don't match the turned object will stretch it out of shape). And let facing drive placement — seat each object so its front, and the open space its front needs, point the way its orientation dictates.
{_deepseek_suffix()}"""


# ---------- Step 7: iterative next-object decision --------------------------


class NextObjectOutput(BaseModel):
    done: bool
    objects: list[ObjectSpec] = Field(default_factory=list)


SYSTEM_NEXT_OBJECT = f"""<intro>
You are competing in SpatialBench, a competitive benchmark where LLMs create detailed 3D environments from text prompts. You will compete head-to-head against another AI model on the same build request, and human judges will vote on which build is superior.
</intro>

<role>
You are iteratively refining a 3D scene region whose defining anchor objects are already placed. Decide whether more objects would make this region read as complete, or whether the region is already right. When you add objects, emit them as a list — propose every object you want to add this round, from a single object to several at once.
</role>

<input>
The user message contains this region's id and description, plus the scene context (every region and object already placed in the run, which you may reference by id).
</input>

<output>
Respond with a single JSON object containing:
- `done` (boolean): true when no further objects are needed
- `objects` (list of object specs): when `done` is false, one or more new object specs to add this round; otherwise an empty list.

Each object spec has the same fields as the bulk decomposition step:
  - `id` (string): unique, not colliding with any existing node in the scene (or with another object in this same list)
  - `prompt` (string): detailed description; used verbatim for text-to-3D
  - `parent` (string): id of the structural anchor (what this object physically rests on, hangs from, leans against, or is contained by) — an existing node, or another object earlier in this same list
  - `parent_relationship_kind` (string): exactly one of `ON` (rests on parent's outward surface), `ATTACHED` (flush against any face of the parent), or `IN` (contained inside the parent's volume / footprint). `BESIDE` / `ABOVE` / `BELOW` are NOT valid here.
  - `proxy_shape` (string | null): BOX / SPHERE / CAPSULE / HEMISPHERE if the object's silhouette is non-rectilinear, otherwise null/omitted.
  - `orientation` (int): world-frame yaw about +Y in degrees. Exactly one of -180, -135, -90, -45, 0, 45, 90, 135, 180. `0` = front faces +Z, `90` = front faces -X, `180` = front faces -Z, `-90` = front faces +X. Use 0 for symmetric objects.
  - `placement` (string): one string describing where this object sits within the scene relative to its parent and the other regions and objects around it. This placement should NOT contain any precise coordinates, which will all be resolved later through a downstream solver step; it should only be a semantic spatial description of where the object is located. Think very deeply about where each object should lie spatially and designing the placement string for it.
  - `relationships` (list of {{target, kind}}): OPTIONAL spatial relationships to other already-placed nodes to assist in the downstream spatial resolver step. Think very deeply and precisely about what each of these relationships with other objects is spatially. Each entry has a `target` (the peer's id) and a `kind` — one of ON, BESIDE, ABOVE, BELOW, ATTACHED, IN. Do NOT repeat the parent here. Empty list is fine when the placement only references the parent.

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
    zone_plan: str,
    nodes: list[Node],
    prior_attempts: list[tuple[list[ObjectSpec], str]] | None = None,
) -> str:
    root = util.find_root(nodes)
    assert root is not None, "next-object decision requires a root node in scope"
    zone_bbox = next((n.bbox for n in nodes if n.id == zone_id), root.bbox)
    context = render_embedded_block(nodes, node_id=zone_id, text="This is the subregion you are deciding whether to add more objects to.")
    if prior_attempts:
        attempt_lines = "\n\n".join(
            f"""  attempt {i}:
    emitted: [{", ".join(s.model_dump_json() for s in specs)}]
    rejected: {reason}"""
            for i, (specs, reason) in enumerate(prior_attempts)
        )
        retry_block = f"""

PRIOR ATTEMPTS — every object batch below was ALREADY rejected. Do NOT re-emit the same specs, and do not repeat the same structural mistake. Treat every listed reason as a hard constraint you must satisfy this time:
{attempt_lines}

Either emit a NEW list of ObjectSpecs that fixes every listed reason, or set done=true. Every object's `parent` must be a valid existing id (the supporter or containing region that anchors it), and every `relationships` entry's `target` must exist in the scene context."""
    else:
        retry_block = ""
    return f"""You are the step in the SpatialBench pipeline responsible for determining if more objects are needed to bring a given subregion closer to feeling complete in terms of the intention of the overall scene and the given subregion. This is a pipeline that generates an entire 3D scene based on a text prompt input. During the generation of the 3D scene, the pipeline breaks down the scene into individual regions to allow downstream steps to recurse into them and focus on each one individually. Once the pipeline reaches a subregion that it determines to not need any more further subdivisions, it begins to generate a list of concrete objects that fill up that region that make the region what it claims to be.

    However, this pass may or may not be complete - oftentimes, this list of objects can be small and not fully flesh out the intention and narrative given by the plans of the overall scene and the given region. It is your role to take over after this list of anchor objects have been created for the region, and determine whether more objects are needed to further flesh out and expand upon the region's intention and narrative as part of the larger scene.
    
    Here is the overall scene that is being built by the pipeline:

{_root_scene_header(root)}

{render_root_objects(nodes)}

Here's the scene's subregion tree, with the objects placed in each subregion listed inline as a flat list beneath it, recursing into nested subregions. An object anchored to a peer object names that peer in its `parent` block rather than nesting beneath it.

{context}

You are deciding whether more objects are needed in this subregion to flesh out its intention and narrative:

Subregion name: {zone_id}
Subregion description: "{zone_prompt}"
Subregion plan: "{zone_plan}"
Subregion dimensions: {util.format_dimensions(zone_bbox)}
Subregion global origin corner: {util.format_global_origin(zone_bbox)}

If you choose yes, output a list of ObjectSpecs describing the objects to add this round — that may be a single object or several at once. The pipeline will then place them and loop back to this step again, and this continues until you say no (by which point the region should feel complete). If you choose no, set done = true and the pipeline moves onto another region.

Each object you generate should be individualistic - do not rely on the downstream 3D model generation step to be able to parse complex arrangements of objects or objects of composite shape properly; you are instead responsible for breaking what you want to add into discrete, self-contained partial objects, each emitted as its own entry in the list with its own placement text. This will allow the downstream step responsible for coming up with the concrete bounding box locations for each object to properly lay out the partial objects in the exact way you want. {retry_block}
{_deepseek_suffix()}"""
