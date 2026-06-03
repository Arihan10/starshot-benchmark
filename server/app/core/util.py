"""Formatting helpers for the scene-context tree renderers in `prompts.py`.

These are intentionally type-light and free of any LLM / spec dependencies so `prompts.py` can import them without a cycle. They cover the two chores the scene renderers share:

  * turning a `BoundingBox` into the "corner + opposite-corner vector" prose the prompts describe, in either world or parent-local frame, and
  * walking the flat `Node` list into the region / object groupings the two tree formats need, plus the brace-block indentation used to print them.
"""

from __future__ import annotations

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
