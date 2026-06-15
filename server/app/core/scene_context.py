"""Deterministic prompt building blocks: the canonical scene-context tree
renderers and the per-step template variables resolved from live scene state.

Prompt WORDING lives in the txt templates of a prompt version (see
versions/TEMPLATE.md); this module is the code layer those templates pull
runtime scene state from. Every `{VARIABLE}` a template can reference is
produced here:

  * the embedded subregion tree (`{SCENE_CONTEXT}`) — each subregion lists the
    objects placed inside it inline beneath it, then its nested subregions.
    Objects are never nested under one another — each object entry carries its
    own `parent` (structural anchor) and `parent_region`.
  * the root scene header / root-anchored object list (`{ROOT_HEADER}`,
    `{ROOT_OBJECTS}`).
  * the to-place spec block for the bbox solvers (`{TO_PLACE}`).
  * the validation-retry feedback block (`{RETRY_BLOCK}`).
  * the Nano-Banana image wrapper (also used directly at the image-generation
    boundary via `wrap_image_prompt`).
"""

from __future__ import annotations

from typing import Literal

from app.core import prompt_store, util
from app.core.schemas import ObjectSpec, SubregionSpec
from app.core.types import BoundingBox, Node, ProxyShape

_NO_NODES_MESSAGE = "(no regions or objects have been placed yet — this is the very start of the run)"
_NO_SUBREGIONS_MESSAGE = "{(none - no other subregions have been planned yet)}"

# Inline marker appended to a targeted subregion's name line in the
# embedded block. The arrow points back at the node id and labels it as
# the target so a prompt can call the LLM's attention to one region; the
# caller-supplied text follows it.
_TARGET_MARKER = "<-- TARGET:"

def _render_proxy_shape(p: ProxyShape | None) -> str:
    return p.value if p is not None else "BOX"


def _root_scene_header(root: Node) -> str:
    """Root prompt, plan, and overall bounding box — injected at the top of every prompt that shows scene context. The root bbox is delivered in natural language — its `W by H by D` dimensions plus its origin corner, tagged `(scene root)` — instead of the raw `origin/dimensions` coordinate dump, while still surfacing every value the box carries."""
    dx, dy, dz = root.bbox.dimensions
    ox, oy, oz = root.bbox.origin
    return (
        f'Prompt: "{root.prompt}"\n'
        f'Plan: "{root.plan}"\n'
        f"Overall scene (root) bounding box: {dx:.2f}m by {dy:.2f}m by {dz:.2f}m, with its origin corner at ({ox:.2f}, {oy:.2f}, {oz:.2f}) m"
    )


def _local_coords_line(node: Node, by_id: dict[str, Node]) -> str | None:
    """`Local coordinates relative to its parent (<pid>): ...` line, or None for the root / a node whose parent is absent from the snapshot."""
    if node.parent_id is not None and node.parent_id in by_id:
        coords = util.format_local_origin(node.bbox, by_id[node.parent_id].bbox)
        return f"Local origin corner (relative to {node.parent_id}, measured from its min corner): {coords}"
    return None


def _object_entry(obj: Node, by_id: dict[str, Node], parent_zone: str) -> str:
    """Full-detail entry for one concrete object (no plan), rendered as a member of the scene's flat object list.

    `parent` is the object's structural-anchor block — `parent_id` (the peer object or region this object physically rests on / attaches to / sits inside), `parent_relationship_kind` (ON / ATTACHED / IN), `parent_dimensions` (that parent's size), and `parent_global_origin_corner` (its world position). `parent_region` is the id of the subregion this object belongs to and `parent_region_dimensions` is that region's size; both are omitted when `parent_region` would equal `parent_id` (the object anchors directly to its region — the `parent` block already states them), and they are shown only when the object anchors to a peer object (e.g. a lamp ON a nightstand has parent_id=nightstand, parent_region=<the subregion the nightstand is in>)."""
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
    if region.referenced_ids:
        refs = ", ".join(f"{r.target}: {r.kind.value}" for r in region.referenced_ids)
        lines.append(f"relationships: [{refs}]")
    else:
        lines.append("relationships: []")
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


def render_root_objects(nodes: list[Node]) -> str:
    """Flat list of the objects anchored directly to the scene root — the
    shared geometry (shells, ground planes, ambient root-level fill) the whole
    scene rests on. Split out of `render_embedded_block` because this is GLOBAL
    context: every region sits within it, so it reads best next to the root
    scene header rather than buried inside the per-subregion tree. Returns a
    short placeholder when the root carries no such geometry yet."""
    root = util.find_root(nodes)
    if root is None:
        return _NO_NODES_MESSAGE
    by_id = {n.id: n for n in nodes}
    oidx = util.index_objects_by_region(nodes)
    root_objects = oidx.get(root.id, [])
    if not root_objects:
        return "No objects are parented directly to the root yet."
    return (
        "Here's a list of objects that belong to the root region:\n\n"
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
    """Scene context in EMBEDDED form: the subregion tree where each subregion lists the objects placed directly inside it inline as a FLAT list — objects are never nested under one another, a peer-anchored object just names its anchor in its own `parent` block — and then recurses into its nested subregions. Objects anchored directly to the scene root (the shell/ground meshes from the root's encapsulating pass) are NOT included here — they are global geometry, rendered separately by `render_root_objects`. Renders the single-region placeholder when the scene has no subregions yet.

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


def render_to_place_block(
    to_place: list[SubregionSpec] | list[ObjectSpec] | None,
    by_id: dict[str, Node],
    parent_zone: str | None = None,
    show_orientation: bool = True,
) -> str:
    """Pseudo-JSON block of the children/objects whose bboxes a bbox-batch step must determine — the template writes the introducing sentence. Empty string when there is nothing to place.

    Two forms, chosen per spec:

    * SUBREGIONS (`SubregionSpec`, no structural `parent`) are abstract subdivisions ALWAYS contained in `parent_zone`. Each is emitted with `contained_in_zone` plus that zone's dimensions / global origin, and an explicit note that its bbox is authored in the zone's LOCAL frame (measured from the zone's min corner). The zone hierarchy is already in the surrounding scene context, so no parent block is repeated; sibling adjacency rides in `relationships`.

    * OBJECTS (`ObjectSpec`, structural `parent`) keep the full anchor block — `parent`, `parent_relationship_kind`, and the parent's dimensions / global origin (the frame the object's bbox is authored in). When the structural parent is a peer rather than `parent_zone`, `parent_region` + `parent_region_dimensions` also name the region the object belongs to, mirroring `_object_entry`.

    `show_orientation` stays True for objects (which carry a real yaw) but is False for subregions, since zones are never yawed."""
    if not to_place:
        return ""
    to_place_ids = {c.id for c in to_place}
    entries: list[str] = []
    for c in to_place:
        spec_parent = getattr(c, "parent", None)
        if spec_parent is None:
            # Subregion: contained in the zone being decomposed; its bbox is
            # authored in that zone's local frame (from the zone's min corner).
            lines = [
                f"id: {c.id}",
                f"parent_region: {parent_zone}",
            ]
        else:
            kind_str = c.parent_kind.value
            if spec_parent in by_id:
                pdims_str = util.format_dimensions(by_id[spec_parent].bbox)
                porigin_str = util.format_global_origin(by_id[spec_parent].bbox)
            elif spec_parent in to_place_ids:
                pdims_str = "(parent is also being placed in this batch — use your emitted dimensions for it)"
                porigin_str = "(parent is also being placed in this batch — use your emitted position for it)"
            else:
                pdims_str = "(parent id not recognised in current scene)"
                porigin_str = "(parent id not recognised in current scene)"
            lines = [
                f"id: {c.id}",
                f"parent: {spec_parent}",
                f"parent_relationship_kind: {kind_str}",
                f"parent_dimensions: {pdims_str}",
                f"parent_global_origin_corner: {porigin_str}",
            ]
            if parent_zone is not None and spec_parent != parent_zone:
                lines.append(f"parent_region: {parent_zone}")
                if parent_zone in by_id:
                    lines.append(
                        f"parent_region_dimensions: {util.format_dimensions(by_id[parent_zone].bbox)}"
                    )
        lines.append(f"proxy_shape: {_render_proxy_shape(c.proxy_shape)}")
        if show_orientation:
            lines.append(f"orientation: {getattr(c, 'orientation', 0)}deg")
        lines.append(f'prompt: "{c.prompt}"')
        lines.append(f'placement: "{c.placement}"')
        if c.referenced_ids:
            refs = ", ".join(f"{r.target}: {r.kind.value}" for r in c.referenced_ids)
            lines.append(f"relationships: [{refs}]")
        else:
            lines.append("relationships: []")
        entries.append(util.braces("\n".join(lines)))
    return util.brace_group(entries)


# --- validation-retry feedback ({RETRY_BLOCK}) -------------------------------


def _attempt_lines(prior_attempts: list[tuple[list[ObjectSpec], str]]) -> str:
    return "\n\n".join(
        f"""  attempt {i}:
    emitted: [{", ".join(s.model_dump_json() for s in specs)}]
    rejected: {reason}"""
        for i, (specs, reason) in enumerate(prior_attempts)
    )


def render_retry_block(
    prior_attempts: list[tuple[list[ObjectSpec], str]] | None,
) -> str:
    """`{RETRY_BLOCK}` for the bulk decomposition steps (anchor / encapsulating /
    negative-space). Empty on the first attempt."""
    if not prior_attempts:
        return ""
    return f"""

PRIOR ATTEMPTS — every decomposition below was ALREADY rejected. Do NOT re-emit the same set of object specs, and do not repeat the same structural mistake. Treat every listed reason as a hard constraint you must satisfy this time:
{_attempt_lines(prior_attempts)}

Produce a NEW decomposition that fixes every listed reason. In particular, ensure every object's `parent` field is set to a valid existing id (the supporter or containing region that anchors it), and every `relationships` entry has a `target` that exists in the scene context."""


def render_next_object_retry_block(
    prior_attempts: list[tuple[list[ObjectSpec], str]] | None,
) -> str:
    """`{RETRY_BLOCK}` for the anchor-completion (next_object) loop. Empty on
    the first attempt."""
    if not prior_attempts:
        return ""
    return f"""

PRIOR ATTEMPTS — every object batch below was ALREADY rejected. Do NOT re-emit the same specs, and do not repeat the same structural mistake. Treat every listed reason as a hard constraint you must satisfy this time:
{_attempt_lines(prior_attempts)}

Either emit a NEW list of ObjectSpecs that fixes every listed reason, or set done=true. Every object's `parent` must be a valid existing id (the supporter or containing region that anchors it), and every `relationships` entry's `target` must exist in the scene context."""


# --- Nano-Banana image wrapper ------------------------------------------------

SUBJECT_SLOT = "<<<SUBJECT>>>"


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


# --- per-step template variables ----------------------------------------------
#
# Every render receives the FULL `prompt_store.ALL_VARIABLES` vocabulary so
# any variable can be injected into any template: builders start from an
# all-empty base and fill in whatever the step's state can actually back.
# Variables outside a step's natural set resolve to "" (or the canonical
# empty-scene placeholders) instead of failing the render.


def base_vars() -> dict[str, str]:
    # Legacy tokens stay resolvable (as "") so snapshots created before their
    # removal keep rendering; they are no longer part of the vocabulary.
    out = {name: "" for name in prompt_store.ALL_VARIABLES}
    for name in prompt_store.LEGACY_VARIABLES:
        out[name] = ""
    return out


def zone_vars(
    *,
    zone_id: str,
    zone_prompt: str,
    zone_plan: str | None,
    nodes: list[Node],
    target_text: str,
) -> dict[str, str]:
    """The scene-wide + target-zone variable set shared by every region-scoped
    step. `target_text` is the inline marker label drawn on the target region
    inside `{SCENE_CONTEXT}`. Requires a placed root in `nodes`.

    The `PARENT_ZONE_*` variables describe the target zone's ENCLOSING parent
    region (the zone one level up the tree) — distinct from the `ZONE_*` set,
    which is the zone in focus. `PARENT_ZONE_ID` is the parent's id; `PARENT_ZONE_PLAN`
    and `PARENT_ZONE_ORIGIN` are its plan and bbox world origin corner. All are
    the empty string for the root (which has no parent) and for any `zone_id`
    not yet present in `nodes`."""
    root = util.find_root(nodes)
    assert root is not None, f"step targeting {zone_id!r} requires a root node in scope"
    focus = next((n for n in nodes if n.id == zone_id), None)
    zone_bbox = focus.bbox if focus is not None else root.bbox
    parent_zone_id = focus.parent_id if focus is not None and focus.parent_id is not None else ""
    parent = next((n for n in nodes if n.id == parent_zone_id), None) if parent_zone_id else None
    out = base_vars()
    out.update({
        "ROOT_PROMPT": root.prompt,
        "ROOT_PLAN": str(root.plan or ""),
        "ROOT_DIMENSIONS": util.format_dimensions(root.bbox),
        "ROOT_ORIGIN": util.format_global_origin(root.bbox),
        "ROOT_HEADER": _root_scene_header(root),
        "ROOT_OBJECTS": render_root_objects(nodes),
        "SCENE_CONTEXT": render_embedded_block(nodes, node_id=zone_id, text=target_text),
        "ZONE_ID": zone_id,
        "ZONE_PROMPT": zone_prompt,
        "ZONE_PLAN": str(zone_plan or ""),
        "ZONE_PLACEMENT": str(focus.placement or "") if focus is not None else "",
        "ZONE_DIMENSIONS": util.format_dimensions(zone_bbox),
        "ZONE_ORIGIN": util.format_global_origin(zone_bbox),
        "PARENT_ZONE_ID": parent_zone_id,
        "PARENT_ZONE_PLAN": str(parent.plan or "") if parent is not None else "",
        "PARENT_ZONE_ORIGIN": util.format_global_origin(parent.bbox) if parent is not None else "",
    })
    return out


def root_seed_vars(*, prompt: str) -> dict[str, str]:
    """Variables for `zone_plan_root` — the very start of a run, before any
    node exists. Scene-state variables resolve to the canonical empty-scene
    placeholders; bbox/plan-derived ones are empty."""
    out = base_vars()
    out.update({
        "ROOT_PROMPT": prompt,
        "ZONE_ID": "root",
        "ZONE_PROMPT": prompt,
        "ROOT_OBJECTS": render_root_objects([]),
        "SCENE_CONTEXT": render_embedded_block([]),
    })
    return out


def overall_bbox_vars(*, prompt: str, scene_plan: str) -> dict[str, str]:
    """Variables for `overall_bbox` — the root plan exists, its bbox doesn't
    yet (that's what this step produces)."""
    out = base_vars()
    out.update({
        "ROOT_PROMPT": prompt,
        "ROOT_PLAN": scene_plan,
        "ZONE_ID": "root",
        "ZONE_PROMPT": prompt,
        "ZONE_PLAN": scene_plan,
        "ROOT_OBJECTS": render_root_objects([]),
        "SCENE_CONTEXT": render_embedded_block([]),
    })
    return out


def image_prompt_vars(
    *,
    prompt: str,
    bbox: BoundingBox,
    proxy_shape: ProxyShape | None,
    prior_prompts: list[str],
    zone: Node | None = None,
    nodes: list[Node] | None = None,
) -> dict[str, str]:
    """The variable set for the `image_prompt` step. When the owning zone +
    scene snapshot are supplied, the scene-wide/zone variables are populated
    too, so image-prompt templates can pull scene context if they want it."""
    w, h, d = bbox.size
    if prior_prompts:
        prior_lines = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(prior_prompts))
        prior_block = f"Prior subject phrases in this scene ({len(prior_prompts)} total):\n{prior_lines}"
    else:
        prior_block = "Prior subject phrases in this scene: (none — this is the first object; you are setting the aesthetic baseline)."
    if zone is not None and nodes:
        out = zone_vars(
            zone_id=zone.id,
            zone_prompt=zone.prompt,
            zone_plan=zone.plan,
            nodes=nodes,
            target_text="This is the subregion the object being described belongs to.",
        )
    else:
        out = base_vars()
    out.update({
        "OBJECT_PROMPT": prompt,
        "OBJECT_DIMENSIONS": f"width={w:.2f}m, height={h:.2f}m, depth={d:.2f}m",
        "PROXY_SHAPE": _render_proxy_shape(proxy_shape),
        "IMAGE_TEMPLATE_FRONT": wrap_image_prompt(SUBJECT_SLOT, proxy_shape, (w, h, d), view="front"),
        "IMAGE_TEMPLATE_SIDE": wrap_image_prompt(SUBJECT_SLOT, proxy_shape, (w, h, d), view="side"),
        "IMAGE_TEMPLATE_TOP": wrap_image_prompt(SUBJECT_SLOT, proxy_shape, (w, h, d), view="top"),
        "PRIOR_SUBJECTS": prior_block,
    })
    return out
