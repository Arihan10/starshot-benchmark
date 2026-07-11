"""Formatting helpers for the scene-context tree renderers in `prompts.py`.

These are intentionally type-light and free of any LLM / spec dependencies so `prompts.py` can import them without a cycle. They cover the two chores the scene renderers share:

  * turning a `BoundingBox` into the "corner + opposite-corner vector" prose the prompts describe, in either world or parent-local frame, and
  * walking the flat `Node` list into the region / object groupings the two tree formats need, plus the brace-block indentation used to print them.
"""

from __future__ import annotations

import math

import numpy as np

from app.core.types import BoundingBox, Node

# --- node classification + tree walking --------------------------------------


def is_region(node: Node) -> bool:
    """A node is a region (subzone) if it was created as a zone (`is_zone`) or carries an authored `plan`. A freshly placed zone is a region the moment it exists — before its plan is authored — so it surfaces in scene context as a subregion (seed prompt + bbox) rather than being misclassified as a concrete object. The `plan` fallback keeps any synthetically-built node that only sets a plan classifying correctly. Concrete objects, frames, and perimeter shells set neither."""
    return node.is_zone or node.plan is not None


def index_children(nodes: list[Node]) -> dict[str | None, list[Node]]:
    """Group nodes by `parent_id`, preserving insertion order so the rendered tree is deterministic."""
    children: dict[str | None, list[Node]] = {}
    for n in nodes:
        children.setdefault(n.parent_id, []).append(n)
    return children


def find_root(nodes: list[Node]) -> Node | None:
    """The scene root is the single node with no parent."""
    for n in nodes:
        if n.parent_id is None:
            return n
    return None


def split_region_members(
    region_id: str,
    children_index: dict[str | None, list[Node]],
) -> tuple[list[Node], list[Node]]:
    """Partition everything hanging under `region_id` into (objects, subregions).

    Objects are collected transitively through object -> object chains (a mug resting ON a table both belong to the table's region), but the walk stops at any nested region: that region — and everything beneath it — belongs to the nested region instead and is rendered when it is visited. Both lists come back in breadth-first order.
    """
    objects: list[Node] = []
    subregions: list[Node] = []
    queue: list[Node] = list(children_index.get(region_id, []))
    i = 0
    while i < len(queue):
        node = queue[i]
        i += 1
        if is_region(node):
            subregions.append(node)
        else:
            objects.append(node)
            queue.extend(children_index.get(node.id, []))
    return objects, subregions


def index_objects_by_region(nodes: list[Node]) -> dict[str | None, list[Node]]:
    """Group concrete (non-region) nodes by their `parent_region` field — the
    region whose generation pass emitted each object — preserving insertion
    order. V3/V4 object grouping reads this instead of walking `parent_id`, so a
    frame anchored to a structural supporter in another region still renders
    under the region that owns it. Zones carry no `parent_region` and are
    excluded; they stay grouped by `parent_id`."""
    out: dict[str | None, list[Node]] = {}
    for n in nodes:
        if is_region(n):
            continue
        out.setdefault(n.parent_region, []).append(n)
    return out


def split_region_members_owned(
    region_id: str,
    children_index: dict[str | None, list[Node]],
    objects_by_region: dict[str | None, list[Node]],
) -> tuple[list[Node], list[Node]]:
    """V3/V4 variant of `split_region_members`. Subregions are resolved exactly
    as before (region-nodes hanging under `region_id` by `parent_id`), but
    objects come from the `parent_region` ownership index rather than the
    object->object `parent_id` ancestor walk. V1/V2 keep calling
    `split_region_members` unchanged."""
    _, subregions = split_region_members(region_id, children_index)
    objects = objects_by_region.get(region_id, [])
    return objects, subregions


# --- zone adjacency ----------------------------------------------------------

# Adjacency is computed by ray-casting: from the target zone's centre we shoot a
# near-uniform sphere of rays and keep, per ray, the FIRST region entered.
# Occlusion is intentional (a zone hidden behind a nearer one in some direction
# is not a neighbour), distance is uncapped (the nearest zone in a direction
# counts however far it sits), and the union of every ray's first hit is the
# neighbour set. 256 directions ~ 32 angular cones x ~8 sub-rays each: enough
# resolution that side-by-side neighbours land on different rays and a thin
# neighbour is unlikely to slip between them.
_ADJACENCY_RAYS = 256
_RAY_EPS = 1e-9


def _fibonacci_directions(n: int) -> np.ndarray:
    """`n` near-uniform unit vectors on the sphere via the golden-spiral
    construction — deterministic, so adjacency is reproducible. Returns an
    `(n, 3)` array of directions in the canonical (X, Y, Z) world frame."""
    i = np.arange(n) + 0.5
    y = 1.0 - 2.0 * i / n  # equal-area latitude bands
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = math.pi * (3.0 - math.sqrt(5.0)) * i  # golden angle
    return np.column_stack((r * np.cos(theta), y, r * np.sin(theta)))


def _aabb_interpenetrates(a: BoundingBox, b: BoundingBox) -> bool:
    """True when the two world boxes overlap in volume (interiors meet on all
    three axes). Such a neighbour engulfs part of the target — possibly its
    centre, where every ray starts — so it has no clean entry point for a ray
    to detect, and is added to the neighbour set directly instead."""
    amin, amax = a.min_corner, a.max_corner
    bmin, bmax = b.min_corner, b.max_corner
    return all(min(amax[i], bmax[i]) - max(amin[i], bmin[i]) > _RAY_EPS for i in range(3))


def _deepest_of_tie(ids: list[str], by_id: dict[str, Node]) -> list[str]:
    """For the zones a single ray entered at the same distance — which happens
    when a child sits flush against its parent's crossed face — drop any that is
    an ancestor of another in the set, keeping the deepest (the child). An
    unrelated coincidence (degenerate overlapping placements) keeps every member."""
    id_set = set(ids)

    def is_ancestor_of_another(cid: str) -> bool:
        for other in id_set:
            if other == cid:
                continue
            anc = by_id[other].parent_id
            while anc is not None and anc in by_id:
                if anc == cid:
                    return True
                anc = by_id[anc].parent_id
        return False

    return [cid for cid in ids if not is_ancestor_of_another(cid)]


def adjacent_zones(target_id: str, nodes: list[Node]) -> list[Node]:
    """Every region adjacent to `target_id`, found by casting a sphere of rays
    from the target's centre and unioning the first region each ray enters.

    Occlusion is a feature (a zone behind a nearer one in a direction is not a
    neighbour) and distance is uncapped (the nearest zone in any direction
    counts, however far). A neighbour that interpenetrates the target — and so
    has no clean entry point, since rays start inside it — is added directly.
    When one ray enters a flush parent and child at the same distance, the
    deeper region (the child) is taken and its ancestor ignored.

    Containment is not adjacency, so the target's ancestor chain (it sits inside
    them) and its whole subtree (they sit inside it) are excluded as candidates.
    Only zones (regions) are returned, in `nodes` order."""
    by_id = {n.id: n for n in nodes}
    target = by_id.get(target_id)
    if target is None:
        return []

    excluded = {target_id}
    ancestor = target.parent_id
    while ancestor is not None and ancestor in by_id and ancestor not in excluded:
        excluded.add(ancestor)
        ancestor = by_id[ancestor].parent_id
    children = index_children(nodes)
    stack = [target_id]
    while stack:
        for child in children.get(stack.pop(), []):
            if child.id not in excluded:
                excluded.add(child.id)
                stack.append(child.id)

    candidates = [n for n in nodes if n.id not in excluded and is_region(n)]
    if not candidates:
        return []

    # A neighbour overlapping the ray origin is invisible to the rays, so seed
    # the hit set with any candidate that interpenetrates the target.
    hit_ids: set[str] = {
        c.id for c in candidates if _aabb_interpenetrates(target.bbox, c.bbox)
    }

    origin = np.asarray(target.bbox.center, dtype=float)
    mins = np.array([c.bbox.min_corner for c in candidates], dtype=float)  # (C, 3)
    maxs = np.array([c.bbox.max_corner for c in candidates], dtype=float)  # (C, 3)
    dirs = _fibonacci_directions(_ADJACENCY_RAYS)                          # (N, 3)
    dirs = np.where(np.abs(dirs) < _RAY_EPS, np.copysign(_RAY_EPS, dirs), dirs)

    # Ray-AABB (slab method), vectorised over candidates for every ray at once.
    # `t_enter` / `t_exit` are entry / exit distances along each ray; a ray
    # enters a box in front of the origin when t_enter <= t_exit and t_exit >= 0.
    t1 = (mins[None, :, :] - origin) / dirs[:, None, :]  # (N, C, 3)
    t2 = (maxs[None, :, :] - origin) / dirs[:, None, :]
    t_enter = np.minimum(t1, t2).max(axis=2)             # (N, C)
    t_exit = np.maximum(t1, t2).min(axis=2)              # (N, C)
    entered = (t_enter <= t_exit) & (t_exit >= 0.0) & (t_enter > _RAY_EPS)
    t_enter = np.where(entered, t_enter, np.inf)

    cand_ids = [c.id for c in candidates]
    for row in t_enter:  # one ray's entry distance to every candidate
        nearest = float(row.min())
        if not np.isfinite(nearest):
            continue  # this ray escaped into open space
        tied = np.flatnonzero(row <= nearest + _RAY_EPS)
        if tied.size == 1:
            hit_ids.add(cand_ids[int(tied[0])])
        else:
            hit_ids.update(_deepest_of_tie([cand_ids[int(j)] for j in tied], by_id))

    return [n for n in nodes if n.id in hit_ids]


def object_spatial_ranks(target_id: str, nodes: list[Node]) -> dict[str, dict[str, float]]:
    """Per-OBJECT spatial relevance to `target_id`, for the ablation "does it
    attend to what's close / visible?" graph: the Euclidean distance from the
    target's centre to each object's centre, and a ray-trace VISIBILITY score —
    how many of a Fibonacci sphere of rays cast from the target hit that object
    FIRST (occlusion-aware, reusing the same slab ray-AABB machinery as
    `adjacent_zones`). OBJECTS only (zones excluded); the target itself excluded.
    Returns `{ id: {"distance": float, "ray_hits": int} }` — the caller ranks
    (nearest / most-visible = rank 1)."""
    by_id = {n.id: n for n in nodes}
    target = by_id.get(target_id)
    if target is None:
        return {}
    origin = np.asarray(target.bbox.center, dtype=float)
    objs = [n for n in nodes if not is_region(n) and n.id != target_id]
    if not objs:
        return {}
    out: dict[str, dict[str, float]] = {
        n.id: {
            "distance": float(np.linalg.norm(np.asarray(n.bbox.center, dtype=float) - origin)),
            "ray_hits": 0,
        }
        for n in objs
    }
    mins = np.array([n.bbox.min_corner for n in objs], dtype=float)  # (C, 3)
    maxs = np.array([n.bbox.max_corner for n in objs], dtype=float)
    dirs = _fibonacci_directions(_ADJACENCY_RAYS)                    # (N, 3)
    dirs = np.where(np.abs(dirs) < _RAY_EPS, np.copysign(_RAY_EPS, dirs), dirs)
    t1 = (mins[None, :, :] - origin) / dirs[:, None, :]              # (N, C, 3)
    t2 = (maxs[None, :, :] - origin) / dirs[:, None, :]
    t_enter = np.minimum(t1, t2).max(axis=2)                         # (N, C)
    t_exit = np.maximum(t1, t2).min(axis=2)
    entered = (t_enter <= t_exit) & (t_exit >= 0.0) & (t_enter > _RAY_EPS)
    t_enter = np.where(entered, t_enter, np.inf)
    ids = [n.id for n in objs]
    for row in t_enter:  # each ray: the object(s) it enters first are "visible"
        nearest = float(row.min())
        if not np.isfinite(nearest):
            continue
        for j in np.flatnonzero(row <= nearest + _RAY_EPS):
            out[ids[int(j)]]["ray_hits"] += 1
    return out


# --- bounding-box prose ------------------------------------------------------


def _fmt_vec(v: tuple[float, float, float]) -> str:
    return f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})"


def format_global_bbox(bbox: BoundingBox) -> str:
    """World-space bbox as a corner vertex plus the signed vector to the opposite corner — the exact shape the prompts describe."""
    return f"origin {_fmt_vec(bbox.origin)} m, dimensions {_fmt_vec(bbox.dimensions)} m"


def format_local_bbox(bbox: BoundingBox, parent: BoundingBox) -> str:
    """Same as `format_global_bbox`, but translated into the parent's local frame (origin at the parent's minimum corner)."""
    return format_global_bbox(bbox.to_local_frame(parent))


def format_dimensions(bbox: BoundingBox) -> str:
    """The bbox's signed `dimensions` vector in meters, rendered on its own line so the global and parent-local frames need not each repeat it."""
    return f"{_fmt_vec(bbox.dimensions)} m"


def format_global_origin(bbox: BoundingBox) -> str:
    """World-space origin corner of a bbox in meters; its size is rendered separately via `format_dimensions`. The caller supplies the leading label (e.g. "Global origin corner:")."""
    return f"{_fmt_vec(bbox.origin)} m"


def format_local_origin(bbox: BoundingBox, parent: BoundingBox) -> str:
    """`bbox`'s origin corner measured from `parent`'s minimum corner (i.e. in `parent`'s local frame), in meters. The caller supplies the leading label."""
    return f"{_fmt_vec(bbox.to_local_frame(parent).origin)} m"


# --- brace-block indentation -------------------------------------------------


def indent(text: str, prefix: str = "    ") -> str:
    """Prefix every non-empty line of `text`; empty lines stay empty to avoid trailing whitespace."""
    return "\n".join(prefix + line if line else line for line in text.split("\n"))


def braces(inner: str) -> str:
    """Wrap a block of content in `{ ... }`, indented one level."""
    return "{\n" + indent(inner) + "\n}"


def brace_group(entries: list[str]) -> str:
    """Wrap a list of already-`{}`-wrapped entries in an outer brace block, separating siblings with a trailing comma + blank line. An empty group renders as `{}`."""
    if not entries:
        return "{}"
    return braces(",\n\n".join(entries))
