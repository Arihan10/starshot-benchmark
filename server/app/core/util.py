"""Type-agnostic rendering helpers for the v2 prompt structure.

`prompts_v2` renders the scene as a nested pseudo-JSON tree of regions and
their objects. These helpers derive the tree views from the flat,
run-scoped `nodes` list and format bounding boxes / brace blocks
consistently so the prompt layout never drifts mid-render.

Zone vs. object classification mirrors the rest of the pipeline: a concrete
(mesh-bearing) node carries a `mesh_url`; abstract regions (root, declared,
and planned zones) never do. That single bit splits a region's members into
its objects and its subregions.
"""

from __future__ import annotations

from app.core.types import BoundingBox, Node


def find_root(nodes: list[Node]) -> Node | None:
    """The single parent-less node, or None when the snapshot is empty."""
    return next((n for n in nodes if n.parent_id is None), None)


def index_children(nodes: list[Node]) -> dict[str | None, list[Node]]:
    """Map every parent id to its children, preserving declaration order.

    A node that names itself as parent is dropped from its own child list so
    the recursive region renderers can never loop on a self-edge. The
    parent-id graph is otherwise acyclic by construction (a node's parent
    always pre-exists it), so plain recursion over this index terminates.
    """
    idx: dict[str | None, list[Node]] = {}
    for n in nodes:
        if n.id == n.parent_id:
            continue
        idx.setdefault(n.parent_id, []).append(n)
    return idx


def split_region_members(
    region_id: str, idx: dict[str | None, list[Node]]
) -> tuple[list[Node], list[Node]]:
    """Split a region's direct children into (objects, subregions).

    Objects are concrete, mesh-bearing nodes; subregions are abstract zones
    (no mesh). Both lists keep declaration order.
    """
    children = idx.get(region_id, [])
    objects = [n for n in children if n.mesh_url is not None]
    subregions = [n for n in children if n.mesh_url is None]
    return objects, subregions


def _fmt_vec(v: tuple[float, float, float]) -> str:
    return f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})"


def format_global_bbox(bbox: BoundingBox) -> str:
    """World-frame bbox as an origin vertex plus a signed dimensions vector."""
    return f"origin {_fmt_vec(bbox.origin)}, dimensions {_fmt_vec(bbox.dimensions)}"


def format_local_bbox(bbox: BoundingBox, parent: BoundingBox) -> str:
    """`bbox` expressed in `parent`'s local frame (origin at parent min corner)."""
    local = bbox.to_local_frame(parent)
    return f"origin {_fmt_vec(local.origin)}, dimensions {_fmt_vec(local.dimensions)}"


def _indent(text: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + line if line else line for line in text.split("\n"))


def braces(body: str) -> str:
    """Wrap one entry's body in an indented `{ ... }` pseudo-JSON block."""
    return "{\n" + _indent(body) + "\n}"


def brace_group(entries: list[str]) -> str:
    """Group already-braced entries inside an outer `{ ... }`. Empty -> `{}`."""
    if not entries:
        return "{}"
    return "{\n" + _indent("\n".join(entries)) + "\n}"
