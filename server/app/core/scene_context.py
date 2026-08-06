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

from functools import lru_cache
from typing import Literal

from app.core import prompt_store, util
from app.core.schemas import ObjectSpec, SubregionSpec
from app.core.types import (
    BoundingBox,
    Node,
    ParentRelationshipKind,
    ProxyShape,
    Relationship,
    RelationshipKind,
)

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
    ox, oy, oz = root.bbox.origin
    return (
        f'prompt: "{root.prompt}"\n'
        f'description: "{root.plan}"\n'
        f"Overall scene (root) bounding box: {util.format_dimensions_natural(root.bbox)}, with its origin corner at ({ox:.2f}, {oy:.2f}, {oz:.2f}) m"
    )


def _local_coords_line(node: Node, by_id: dict[str, Node]) -> str | None:
    """`Local coordinates relative to its parent (<pid>): ...` line, or None for the root / a node whose parent is absent from the snapshot."""
    if node.parent_id is not None and node.parent_id in by_id:
        coords = util.format_local_origin(node.bbox, by_id[node.parent_id].bbox)
        return f"Local origin corner (relative to {node.parent_id}, measured from its min corner): {coords}"
    return None


def _object_entry(obj: Node, by_id: dict[str, Node], parent_zone: str, *, compact: bool = False) -> str:
    """Full-detail entry for one concrete object (no plan), rendered as a member of the scene's flat object list.

    `parent` is the object's structural-anchor block — `parent_id` (the peer object or region this object physically rests on / attaches to / sits inside), `parent_relationship_kind` (ON / ATTACHED / IN), `parent_dimensions` (that parent's size), and `parent_global_origin_corner` (its world position). `parent_region` is the id of the subregion this object belongs to and `parent_region_dimensions` is that region's size; both are omitted when `parent_region` would equal `parent_id` (the object anchors directly to its region — the `parent` block already states them), and they are shown only when the object anchors to a peer object (e.g. a lamp ON a nightstand has parent_id=nightstand, parent_region=<the subregion the nightstand is in>).

    `noun_phrase` (the concise visual subject distilled during the object's asset-generation pass) is shown right under `prompt` whenever the object has one — i.e. its image was already generated; objects not yet generated (and zones) omit the line.

    With `compact=True` (the `{SCENE_CONTEXT_COMPACT}` variant) only the SEMANTIC fields are emitted — id, prompt, `noun_phrase` (when generated), semantic parent (`parent: <id> (<kind>)`), `parent_region` id, placement, relationships, the orientation phrase, and the object's own bbox dimensions. Every world/parent coordinate (global + local origin corners, parent + region dimensions/origins), the numeric yaw, and the proxy shape are dropped."""
    lines = [
        f"Name: {obj.id}",
        f'prompt: "{obj.prompt}"',
    ]
    if obj.noun_phrase:
        lines.append(f'noun_phrase/description: "{obj.noun_phrase}"')
    if obj.parent_id is not None:
        kind_str = obj.parent_kind.value if obj.parent_kind is not None else "(unknown)"
        if compact:
            lines.append(f"parent: {obj.parent_id} ({kind_str})")
        else:
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
        if not compact and parent_zone in by_id:
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
    if not compact:
        lines.append(f"proxy_shape: {_render_proxy_shape(obj.proxy_shape)}")
    lines.append(f'orientation: "{obj.orientation_description}"')
    if not compact:
        lines.append(f"global yaw: {obj.orientation}deg")
    lines.append(f"Dimensions: {util.format_dimensions(obj.bbox)}")
    if not compact:
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
    *,
    compact: bool = False,
) -> str:
    """Subregion tree (embedded form): a subregion's fields, then a FLAT list of the objects placed directly inside it, then (recursively) its nested subregions. Objects are never nested under one another — a peer-anchored object names its anchor in its own `parent` block rather than nesting beneath it. When `target_id` matches this region (at any depth), an inline marker carrying `target_text` is appended to its name line so a prompt can point the LLM at this one region.

    With `compact=True` (the `{SCENE_CONTEXT_COMPACT}` variant) a subregion shows only its name, prompt, description, and placement — its relationships, proxy, dimensions, world/local origin corners, and parent block are dropped. Its inline objects and nested subregions recurse in the same compact form."""
    objects, subregions = util.split_region_members_owned(region.id, idx, obj_idx)
    name_line = f"Subregion name: {region.id}"
    if target_id is not None and region.id == target_id:
        name_line += f"   {_TARGET_MARKER} {target_text}".rstrip()
    lines = [
        name_line,
        f'prompt: "{region.prompt}"',
    ]
    if region.plan is not None:
        lines.append(f'description: "{region.plan}"')
    if region.placement is not None:
        lines.append(f'placement: "{region.placement}"')
    if not compact:
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
            util.brace_group([_object_entry(o, by_id, parent_zone=region.id, compact=compact) for o in objects]),
        ]
    if subregions:
        lines += [
            "",
            f'Here\'s the list of subregions that are present within "{region.id}".',
            "",
            util.brace_group(
                [_region_embedded_entry(s, idx, obj_idx, by_id, target_id, target_text, compact=compact) for s in subregions]
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


def render_zone_objects(zone_id: str, nodes: list[Node]) -> str:
    """Flat list of the objects placed DIRECTLY inside `zone_id` (those whose
    `parent_region` is this zone) — the zone's CURRENT contents, pulled out as a
    standalone block so a step like `next_object` can call attention to what's
    already in the zone before deciding what to add next. Mirrors
    `render_root_objects` but for an arbitrary zone; the same objects also appear
    inline beneath this zone in `{SCENE_CONTEXT}`. Returns a short placeholder
    when the zone holds no objects yet."""
    objects = util.index_objects_by_region(nodes).get(zone_id, [])
    if not objects:
        return "No objects have been placed directly in this zone yet."
    by_id = {n.id: n for n in nodes}
    return (
        f'Here\'s a list of objects already placed directly within "{zone_id}":\n\n'
        + util.brace_group(
            [_object_entry(o, by_id, parent_zone=zone_id) for o in objects]
        )
    )


# --- image-prompt scene context (reduced) ------------------------------------
#
# The image_prompt step only distills a noun phrase, so it needs just enough
# context to keep an object's LOOK coherent with its neighbours — not the full
# geometric placement tree. It gets a GRADUATED, far-trimmed view: the subject's
# OWN region + sibling objects in semantic detail (`_object_local`), and
# everything beyond (root + other subregions) as bare id / prompt / noun phrase
# (`_object_brief`). World/local coordinates, yaw, proxy, and parent-frame dumps
# are all dropped here.


def _object_brief(obj: Node) -> str:
    """Barest object entry — id, seed prompt, and the distilled noun phrase when
    it exists. The FAR-context form (root + other subregions): only a neighbour's
    identity and look matter for aesthetic coherence."""
    lines = [f"Name: {obj.id}", f'prompt: "{obj.prompt}"']
    if obj.noun_phrase:
        lines.append(f'noun_phrase: "{obj.noun_phrase}"')
    return util.braces("\n".join(lines))


def _object_local(obj: Node) -> str:
    """Sibling-object entry for the subject's OWN region: the semantic fields that
    pin down the local arrangement — noun phrase, structural anchor
    (`parent: <id> (<kind>)`), placement, relationships, and size — without the
    world/local coordinate dump the placement steps consume."""
    lines = [f"Name: {obj.id}", f'prompt: "{obj.prompt}"']
    if obj.noun_phrase:
        lines.append(f'noun_phrase: "{obj.noun_phrase}"')
    if obj.parent_id is not None:
        kind_str = obj.parent_kind.value if obj.parent_kind is not None else "(unknown)"
        lines.append(f"parent: {obj.parent_id} ({kind_str})")
    if obj.placement is not None:
        lines.append(f'placement: "{obj.placement}"')
    refs = ", ".join(f"{r.target}: {r.kind.value}" for r in obj.referenced_ids)
    lines.append(f"relationships: [{refs}]")
    lines.append(f"Dimensions: {util.format_dimensions(obj.bbox)}")
    return util.braces("\n".join(lines))


def render_image_sibling_objects(zone_id: str, nodes: list[Node]) -> str:
    """`{SIBLING_OBJECTS}` — the objects already placed directly within the
    subject's own region (`_object_local` detail): the immediate company the
    distilled object shares space with. Placeholder when it is the first."""
    objects = util.index_objects_by_region(nodes).get(zone_id, [])
    if not objects:
        return "No other objects have been placed directly in this region yet."
    return util.brace_group([_object_local(o) for o in objects])


def render_image_root_objects(nodes: list[Node]) -> str:
    """`{ROOT_OBJECTS_BRIEF}` — the root-region objects (shells / ground / ambient
    fill) in bare `_object_brief` form. Placeholder when the root holds none."""
    root = util.find_root(nodes)
    if root is None:
        return _NO_NODES_MESSAGE
    objects = util.index_objects_by_region(nodes).get(root.id, [])
    if not objects:
        return "No objects belong to the root region yet."
    return util.brace_group([_object_brief(o) for o in objects])


def render_image_other_subregions(focus_zone_id: str, nodes: list[Node]) -> str:
    """`{OTHER_SUBREGIONS_BRIEF}` — every region EXCEPT the root and the subject's
    own region, as a FLAT list (no hierarchy), each its name + prompt with a bare
    `_object_brief` list of the objects inside it. The far aesthetic backdrop for
    the image step; placeholder when there are none."""
    root = util.find_root(nodes)
    oidx = util.index_objects_by_region(nodes)
    entries: list[str] = []
    for n in nodes:
        if not util.is_region(n) or n.id == focus_zone_id or (root is not None and n.id == root.id):
            continue
        lines = [f"Subregion name: {n.id}", f'prompt: "{n.prompt}"']
        objs = oidx.get(n.id, [])
        if objs:
            lines += [
                "",
                f'Objects placed directly within "{n.id}":',
                "",
                util.brace_group([_object_brief(o) for o in objs]),
            ]
        entries.append(util.braces("\n".join(lines)))
    if not entries:
        return _NO_SUBREGIONS_MESSAGE
    return util.brace_group(entries)


def render_embedded_block(
    nodes: list[Node],
    *,
    node_id: str | None = None,
    text: str = "",
    compact: bool = False,
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
        [_region_embedded_entry(s, idx, oidx, by_id, node_id, text, compact=compact) for s in subregions]
    )


def render_to_place_block(
    to_place: list[SubregionSpec] | list[ObjectSpec] | None,
    by_id: dict[str, Node],
    parent_zone: str | None = None,
) -> str:
    """Pseudo-JSON block of the children/objects whose bboxes a bbox-batch step must determine — the template writes the introducing sentence. Empty string when there is nothing to place.

    Two forms, chosen per spec:

    * SUBREGIONS (`SubregionSpec`, no structural `parent`) are abstract subdivisions ALWAYS contained in `parent_zone`. Each is emitted with `contained_in_zone` plus that zone's dimensions / global origin, and an explicit note that its bbox is authored in the zone's LOCAL frame (measured from the zone's min corner). The zone hierarchy is already in the surrounding scene context, so no parent block is repeated; sibling adjacency rides in `relationships`.

    * OBJECTS (`ObjectSpec`, structural `parent`) keep the full anchor block — `parent`, `parent_relationship_kind`, and the parent's dimensions / global origin (the frame the object's bbox is authored in). When the structural parent is a peer rather than `parent_zone`, `parent_region` + `parent_region_dimensions` also name the region the object belongs to, mirroring `_object_entry`.

    Objects also carry their `orientation` text — the semantic heading the decompose step authored (free text, not a yaw) — which the object_bbox_batch solver resolves to a discrete yaw; subregions have none."""
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
        if spec_parent is not None:
            lines.append(f'orientation: "{getattr(c, "orientation", "")}"')
        lines.append(f'prompt: "{c.prompt}"')
        lines.append(f'placement: "{c.placement}"')
        if c.referenced_ids:
            refs = ", ".join(f"{r.target}: {r.kind.value}" for r in c.referenced_ids)
            lines.append(f"relationships: [{refs}]")
        else:
            lines.append("relationships: []")
        entries.append(util.braces("\n".join(lines)))
    return util.brace_group(entries)


def render_adjacent_zones_block(target_id: str, nodes: list[Node]) -> str:
    """The zones ADJACENT to `target_id` — the nearest region in each direction
    from the target's centre, found by ray-casting (see `util.adjacent_zones`) —
    rendered in the SAME embedded form as `{SCENE_CONTEXT}` (each zone with its
    plan, bbox, inline objects, and nested subregions), but TRIMMED to just the
    neighbours. A neighbour that has already been developed shows its full
    subtree; one only just placed shows its seed prompt + bbox (no plan yet).
    Empty string when the target has no neighbours, so a template can gate on it.

    When one neighbour is nested inside another (both border the target), only
    the outermost is a top-level entry — the inner one appears nested within it
    via the recursive entry, so every neighbour is shown exactly once."""
    adjacent = util.adjacent_zones(target_id, nodes)
    if not adjacent:
        return ""
    by_id = {n.id: n for n in nodes}
    idx = util.index_children(nodes)
    oidx = util.index_objects_by_region(nodes)
    adjacent_ids = {z.id for z in adjacent}

    def _nested_under_neighbour(z: Node) -> bool:
        ancestor = z.parent_id
        while ancestor is not None and ancestor in by_id:
            if ancestor in adjacent_ids:
                return True
            ancestor = by_id[ancestor].parent_id
        return False

    tops = [z for z in adjacent if not _nested_under_neighbour(z)]
    return util.brace_group(
        [_region_embedded_entry(z, idx, oidx, by_id) for z in tops]
    )


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

Either emit a NEW list of ObjectSpecs that fixes every listed reason, or set objects_required=false. Every object's `parent` must be a valid existing id (the supporter or containing region that anchors it), and every `relationships` entry's `target` must exist in the scene context."""


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
        f"Generate a direct, perfect {view_phrase} of {description}, one-to-one"
        f"that roughly can be captured within {_article(hitbox)} {hitbox} "
        "hitbox without bending or deforming the object's natural "
        f"proportions. The object should not fully be in {_article(silhouette)} "
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
        "SCENE_CONTEXT_COMPACT": render_embedded_block(nodes, node_id=zone_id, text=target_text, compact=True),
        "ADJACENT_ZONES": render_adjacent_zones_block(zone_id, nodes),
        "ZONE_ID": zone_id,
        "ZONE_PROMPT": zone_prompt,
        "ZONE_PLAN": str(zone_plan or ""),
        "ZONE_PLACEMENT": str(focus.placement or "") if focus is not None else "",
        "ZONE_DIMENSIONS": util.format_dimensions(zone_bbox),
        "FORMATTED_ZONE_DIMENSIONS": util.format_dimensions_natural(zone_bbox),
        "ZONE_ORIGIN": util.format_global_origin(zone_bbox),
        "ZONE_OBJECTS": render_zone_objects(zone_id, nodes),
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
        "SCENE_CONTEXT_COMPACT": render_embedded_block([], compact=True),
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
        "SCENE_CONTEXT_COMPACT": render_embedded_block([], compact=True),
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
    """The variable set for the `image_prompt` step (noun-phrase distillation).

    The step only needs enough context to keep an object's LOOK coherent with its
    neighbours, so it gets a GRADUATED, far-trimmed view rather than the full
    placement tree: the subject's OWN region in semantic detail (`ZONE_*` +
    `{SIBLING_OBJECTS}`), and everything beyond — the root region
    (`ROOT_PROMPT` / `ROOT_PLAN` + `{ROOT_OBJECTS_BRIEF}`) and every other
    subregion (`{OTHER_SUBREGIONS_BRIEF}`) — as bare id / prompt / noun phrase.
    The heavy `{SCENE_CONTEXT}` / `{ROOT_HEADER}` / `{ZONE_OBJECTS}` blocks are
    intentionally NOT built here. `zone`/`nodes` are optional; without them only
    the object's own fields populate and the scene vars stay empty."""
    w, h, d = bbox.size
    if prior_prompts:
        prior_lines = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(prior_prompts))
        prior_block = f"Prior subject phrases in this scene ({len(prior_prompts)} total):\n{prior_lines}"
    else:
        prior_block = "Prior subject phrases in this scene: (none — this is the first object; you are setting the aesthetic baseline)."
    out = base_vars()
    if zone is not None and nodes:
        root = util.find_root(nodes)
        out.update({
            "ROOT_PROMPT": root.prompt if root is not None else "",
            "ROOT_PLAN": str(root.plan or "") if root is not None else "",
            "ROOT_OBJECTS_BRIEF": render_image_root_objects(nodes),
            "ZONE_ID": zone.id,
            "ZONE_PROMPT": zone.prompt,
            "ZONE_PLAN": str(zone.plan or ""),
            "ZONE_PLACEMENT": str(zone.placement or ""),
            "SIBLING_OBJECTS": render_image_sibling_objects(zone.id, nodes),
            "OTHER_SUBREGIONS_BRIEF": render_image_other_subregions(zone.id, nodes),
        })
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


# --- sample variable rendering (prompt-lab hover preview) ---------------------


@lru_cache(maxsize=1)
def sample_variables() -> dict[str, str]:
    """Render EVERY `{VARIABLE}` against a fixed, representative sample scene
    using the SAME builders the live pipeline injects with — so the prompt lab
    can show each variable's real shape on hover, surfacing missing context or a
    rendering bug without having to run a scene. Nothing here is hard-coded text:
    every value comes out of `zone_vars` / `render_to_place_block` /
    `render_retry_block` / `image_prompt_vars`, exactly as the pipeline calls
    them. Cached — the sample scene is static.

    The scene is a two-room cottage that exercises the real structure: a root
    shell with a floor slab (root-anchored object), a living room holding a sofa
    + coffee table + a lamp anchored ON that table (object-on-peer), a reading
    nook nested INSIDE the living room (zone-in-zone), and a bedroom. The reading
    nook is the focus zone, so ZONE_* gets a real region and PARENT_ZONE_* gets a
    real non-root parent."""

    def _box(origin: tuple[float, float, float], dims: tuple[float, float, float]) -> BoundingBox:
        return BoundingBox(origin=origin, dimensions=dims)

    root = Node(
        id="root", prompt="A cozy two-room timber cottage",
        bbox=_box((0.0, 0.0, 0.0), (8.0, 3.0, 6.0)), parent_id=None, is_zone=True,
        plan="A warm, lived-in cottage split into a sunlit living room and a snug bedroom, finished in rustic timber and soft textiles.",
    )
    floor = Node(
        id="floor_slab", prompt="a wide oak-plank floor slab",
        noun_phrase="a wide oak-plank floor slab in warm honey tones",
        bbox=_box((0.0, 0.0, 0.0), (8.0, 0.1, 6.0)),
        parent_id="root", parent_kind=ParentRelationshipKind.IN, parent_region="root",
        orientation_description="lies flat; no inherent facing",
        placement="covering the whole footprint at floor level", mesh_url="sample://floor_slab",
    )
    living = Node(
        id="living_room", prompt="the cottage's living room",
        bbox=_box((0.0, 0.0, 0.0), (5.0, 3.0, 6.0)),
        parent_id="root", parent_kind=ParentRelationshipKind.IN, is_zone=True,
        plan="A sunlit lounge centered on a low conversation set, warm and inviting.",
        placement="the western two-thirds of the cottage",
    )
    bedroom = Node(
        id="bedroom", prompt="the cottage's bedroom",
        bbox=_box((5.0, 0.0, 0.0), (3.0, 3.0, 6.0)),
        parent_id="root", parent_kind=ParentRelationshipKind.IN, is_zone=True,
        plan="A snug sleeping nook with a single bed against the back wall.",
        placement="the eastern third of the cottage",
    )
    sofa = Node(
        id="linen_sofa", prompt="a tufted two-seater linen sofa",
        noun_phrase="a tufted two-seater sofa in oatmeal linen",
        bbox=_box((0.5, 0.1, 3.5), (2.0, 0.9, 0.9)),
        parent_id="living_room", parent_kind=ParentRelationshipKind.IN, parent_region="living_room",
        orientation=90, orientation_description="facing east toward the coffee table",
        placement="against the west wall, facing the coffee table",
        referenced_ids=[Relationship(target="coffee_table", kind=RelationshipKind.BESIDE)],
        mesh_url="sample://linen_sofa",
    )
    table = Node(
        id="coffee_table", prompt="a round walnut coffee table",
        noun_phrase="a round walnut coffee table with tapered legs",
        bbox=_box((1.0, 0.1, 2.0), (1.2, 0.45, 0.7)),
        parent_id="living_room", parent_kind=ParentRelationshipKind.IN, parent_region="living_room",
        orientation_description="round; no inherent facing",
        placement="centered in front of the sofa", mesh_url="sample://coffee_table",
    )
    lamp = Node(
        id="table_lamp", prompt="a brass table lamp with a linen shade",
        noun_phrase="a brass table lamp with a cream linen drum shade",
        bbox=_box((1.4, 0.55, 2.2), (0.3, 0.6, 0.3)),
        parent_id="coffee_table", parent_kind=ParentRelationshipKind.ON, parent_region="living_room",
        orientation_description="symmetric shade; no inherent facing",
        placement="centered on the coffee table", mesh_url="sample://table_lamp",
    )
    nook = Node(
        id="reading_nook", prompt="a quiet reading nook",
        bbox=_box((3.0, 0.0, 4.5), (2.0, 3.0, 1.5)),
        parent_id="living_room", parent_kind=ParentRelationshipKind.IN, is_zone=True,
        plan="A calm corner for reading by the window, anchored by a single armchair.",
        placement="the southeast corner of the living room",
    )
    armchair = Node(
        id="leather_armchair", prompt="a worn leather wingback armchair",
        noun_phrase="a worn cognac leather wingback armchair",
        bbox=_box((3.3, 0.1, 4.7), (0.9, 1.0, 0.9)),
        parent_id="reading_nook", parent_kind=ParentRelationshipKind.IN, parent_region="reading_nook",
        orientation=45, orientation_description="angled toward the bay window in the corner",
        placement="angled toward the window in the corner of the nook",
        mesh_url="sample://leather_armchair",
    )
    bed = Node(
        id="platform_bed", prompt="a low platform single bed with a quilt",
        noun_phrase="a low oak platform bed with a sage-green quilt",
        bbox=_box((5.5, 0.1, 0.5), (1.4, 0.6, 2.0)),
        parent_id="bedroom", parent_kind=ParentRelationshipKind.IN, parent_region="bedroom",
        orientation_description="headboard to the back wall, foot facing into the room",
        placement="against the back (north) wall", mesh_url="sample://platform_bed",
    )
    nodes = [root, floor, living, bedroom, sofa, table, lamp, nook, armchair, bed]
    by_id = {n.id: n for n in nodes}

    # Scene-wide + target-zone set (focus = the nested reading nook).
    out = zone_vars(
        zone_id="reading_nook", zone_prompt=nook.prompt, zone_plan=nook.plan,
        nodes=nodes, target_text="the region this step is acting on",
    )

    # TO_PLACE: a batch about to be placed in the nook — one anchored to the zone
    # itself, one with a sibling hint, and one anchored ON a peer object (the
    # armchair), so both the zone-parent and peer-parent branches render.
    to_place = [
        ObjectSpec(
            id="oak_bookshelf", prompt="a narrow oak bookshelf", parent="reading_nook",
            parent_kind=ParentRelationshipKind.IN,
            placement="flush against the east wall of the nook", orientation="facing into the nook",
        ),
        ObjectSpec(
            id="arc_floor_lamp", prompt="a slim arc floor lamp", parent="reading_nook",
            parent_kind=ParentRelationshipKind.IN, placement="beside the armchair, arcing over it",
            referenced_ids=[Relationship(target="leather_armchair", kind=RelationshipKind.BESIDE)],
        ),
        ObjectSpec(
            id="folded_reading_glasses", prompt="a folded pair of reading glasses",
            parent="leather_armchair", parent_kind=ParentRelationshipKind.ON,
            placement="resting on the armchair's near arm",
        ),
    ]
    out["TO_PLACE"] = render_to_place_block(to_place, by_id, parent_zone="reading_nook")

    # PROPOSED_OBJECTS: the object_decomp step's input — the objects an upstream
    # decompose step proposed for a region, which object_decomp analyses for
    # splitting. Rendered with the same block as TO_PLACE (same spec shape), so
    # the sample reuses that batch.
    out["PROPOSED_OBJECTS"] = out["TO_PLACE"]

    # RETRY_BLOCK: one rejected prior attempt, in the bulk-decompose feedback shape.
    rejected = [
        ObjectSpec(
            id="arc_floor_lamp", prompt="a slim arc floor lamp", parent="reading_nook",
            parent_kind=ParentRelationshipKind.IN, placement="beside the armchair",
            referenced_ids=[Relationship(target="window", kind=RelationshipKind.BESIDE)],
        ),
    ]
    out["RETRY_BLOCK"] = render_retry_block(
        [(rejected, "relationships target 'window' does not exist in the scene context")]
    )

    # OBJECT_* / PROXY_SHAPE / IMAGE_TEMPLATE_* / PRIOR_SUBJECTS: the image-prompt
    # step's set, sampled for the sofa. (image_prompt_vars rebuilds the zone set
    # for the sofa's zone, so only the image-specific keys are taken from it.)
    img = image_prompt_vars(
        prompt=sofa.prompt, bbox=sofa.bbox, proxy_shape=sofa.proxy_shape,
        prior_prompts=[floor.prompt, table.prompt, lamp.prompt], zone=living, nodes=nodes,
    )
    for k in (
        "OBJECT_PROMPT", "OBJECT_DIMENSIONS", "PROXY_SHAPE",
        "IMAGE_TEMPLATE_FRONT", "IMAGE_TEMPLATE_SIDE", "IMAGE_TEMPLATE_TOP", "PRIOR_SUBJECTS",
        "SIBLING_OBJECTS", "ROOT_OBJECTS_BRIEF", "OTHER_SUBREGIONS_BRIEF",
    ):
        out[k] = img[k]

    return {name: out.get(name, "") for name in prompt_store.ALL_VARIABLES}
