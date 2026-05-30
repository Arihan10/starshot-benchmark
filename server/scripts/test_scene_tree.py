"""Manual inspection harness for the scene-context tree renderers in `app.core.prompts`.

Prints `render_scene_tree` (separate objects) and `render_scene_tree_embedded` (objects inline) for a few scenarios so the rendered prompt text can be eyeballed in the terminal — there are no assertions here, just output to read:

    uv run python scripts/test_scene_tree.py

Scenarios, in order: an empty scene, a root-only scene, and a decently populated hotel room.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `app` importable when invoked as a loose script from server/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.prompts import render_scene_tree, render_scene_tree_embedded
from app.core.types import BoundingBox, Node, ParentRelationshipKind

PK = ParentRelationshipKind


def _bbox(origin: tuple[float, float, float], dims: tuple[float, float, float]) -> BoundingBox:
    return BoundingBox(origin=origin, dimensions=dims)


def _room() -> Node:
    return Node(
        id="hotel_room",
        prompt="a modern hotel room",
        plan="A compact modern hotel room with a sleeping area, a small work nook, and an ensuite bathroom.",
        bbox=_bbox((-3, 0, -4), (6, 3, 8)),
        parent_id=None,
    )


def empty_scene() -> list[Node]:
    return []


def root_only_scene() -> list[Node]:
    return [_room()]


def hotel_room_scene() -> list[Node]:
    return [
        _room(),
        # objects parented directly to the room
        Node(id="floor", prompt="a carpeted floor slab",
             bbox=_bbox((-3, -0.1, -4), (6, 0.1, 8)), parent_id="hotel_room", parent_kind=PK.ATTACHED),
        Node(id="tv", prompt="a wall-mounted flat-screen TV",
             bbox=_bbox((-0.6, 1.2, -3.95), (1.2, 0.7, 0.05)), parent_id="hotel_room", parent_kind=PK.ATTACHED),
        Node(id="entry_door", prompt="the entry door",
             bbox=_bbox((1.5, 0, 3.95), (1.0, 2.1, 0.05)), parent_id="hotel_room", parent_kind=PK.ATTACHED),
        # sleeping area: a region with an object resting on another object (pillows on bed, lamp on nightstand)
        Node(id="sleeping_area", prompt="the sleeping area",
             plan="A queen bed set against the back wall, flanked by a nightstand.",
             bbox=_bbox((-3, 0, -4), (4, 3, 5)), parent_id="hotel_room", parent_kind=PK.IN),
        Node(id="bed", prompt="a queen bed",
             bbox=_bbox((-2.5, 0, -3.5), (2.0, 0.6, 2.2)), parent_id="sleeping_area", parent_kind=PK.IN),
        Node(id="pillows", prompt="two stacked pillows",
             bbox=_bbox((-2.3, 0.6, -3.4), (1.6, 0.2, 0.5)), parent_id="bed", parent_kind=PK.ON),
        Node(id="nightstand", prompt="a wooden nightstand",
             bbox=_bbox((-0.4, 0, -3.5), (0.4, 0.5, 0.4)), parent_id="sleeping_area", parent_kind=PK.IN),
        Node(id="lamp", prompt="a bedside lamp",
             bbox=_bbox((-0.35, 0.5, -3.45), (0.25, 0.4, 0.25)), parent_id="nightstand", parent_kind=PK.ON),
        # work nook
        Node(id="work_nook", prompt="the work nook",
             plan="A compact wall-mounted desk and chair beside the window.",
             bbox=_bbox((1, 0, -4), (2, 3, 5)), parent_id="hotel_room", parent_kind=PK.IN),
        Node(id="desk", prompt="a wall-mounted writing desk",
             bbox=_bbox((1.2, 0.7, -3.8), (1.4, 0.05, 0.6)), parent_id="work_nook", parent_kind=PK.ATTACHED),
        Node(id="chair", prompt="an ergonomic office chair",
             bbox=_bbox((1.4, 0, -3.0), (0.6, 1.0, 0.6)), parent_id="work_nook", parent_kind=PK.IN),
        # bathroom: a region with its own nested subregion (the shower stall)
        Node(id="bathroom", prompt="the ensuite bathroom",
             plan="An ensuite with a vanity, a toilet, and a glass shower stall.",
             bbox=_bbox((-3, 0, 1), (6, 3, 3)), parent_id="hotel_room", parent_kind=PK.IN),
        Node(id="toilet", prompt="a white toilet",
             bbox=_bbox((-2.6, 0, 1.4), (0.6, 0.8, 0.7)), parent_id="bathroom", parent_kind=PK.IN),
        Node(id="vanity", prompt="a vanity with a sink and mirror",
             bbox=_bbox((-1.5, 0, 1.1), (1.2, 0.9, 0.5)), parent_id="bathroom", parent_kind=PK.ATTACHED),
        Node(id="shower_stall", prompt="the shower stall",
             plan="A glass-enclosed corner shower.",
             bbox=_bbox((1, 0, 1), (2, 2.4, 2)), parent_id="bathroom", parent_kind=PK.IN),
        Node(id="showerhead", prompt="a rainfall showerhead",
             bbox=_bbox((1.8, 2.2, 1.8), (0.3, 0.1, 0.3)), parent_id="shower_stall", parent_kind=PK.ATTACHED),
    ]


def show(title: str, nodes: list[Node]) -> None:
    bar = "=" * 96
    print(f"\n{bar}\n=== SCENARIO: {title}  ({len(nodes)} node(s))\n{bar}")
    print("\n----- render_scene_tree (separate objects) -----\n")
    print(render_scene_tree(nodes=nodes))
    print("\n----- render_scene_tree_embedded (objects inline) -----\n")
    print(render_scene_tree_embedded(nodes=nodes))


def main() -> None:
    show("empty scene", empty_scene())
    show("root only", root_only_scene())
    show("populated hotel room", hotel_room_scene())
    print()


if __name__ == "__main__":
    main()
