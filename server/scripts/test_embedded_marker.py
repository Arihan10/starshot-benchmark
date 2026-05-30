"""Manual inspection harness for the target-marker feature shared by
`render_subregions_block` and `render_embedded_block` in `app.core.prompts`.

Both renderers take an optional `node_id` + `text` pair: the subregion
whose id matches `node_id` gets a little inline target marker carrying
`text` appended to its name line, found at any depth of the tree. This
lets a prompt point the LLM at one specific zone.

This script just prints the rendered blocks for a few targets so the
marker can be eyeballed in the terminal — there are NO assertions here,
only output to read:

    uv run python scripts/test_embedded_marker.py

It reuses the populated hotel-room scene from `test_scene_tree.py`, which
has three top-level subregions (sleeping_area, work_nook, bathroom) and a
nested subregion (shower_stall, inside bathroom).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

# Make `app` importable when invoked as a loose script from server/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.prompts import render_embedded_block, render_subregions_block
from app.core.types import Node
from test_scene_tree import hotel_room_scene


def dump(title: str, block: str) -> None:
    bar = "=" * 96
    print(f"\n{bar}\n=== {title}\n{bar}\n")
    print(block)


def demo_targets(label: str, render: Callable[..., str], nodes: list[Node]) -> None:
    """Print the standard target scenarios for a renderer that takes the
    (nodes, *, node_id, text) marker signature."""
    dump(
        f"{label}: no target — baseline {label}(nodes)",
        render(nodes),
    )
    dump(
        f"{label}: target a TOP-LEVEL subregion node_id='sleeping_area'",
        render(
            nodes,
            node_id="sleeping_area",
            text="this is the zone you are currently planning — fill it in",
        ),
    )
    dump(
        f"{label}: target a NESTED subregion node_id='shower_stall' (inside bathroom)",
        render(nodes, node_id="shower_stall", text="focus here next"),
    )
    dump(
        f"{label}: target an id that does not exist node_id='no_such_zone' — no marker, nothing breaks",
        render(nodes, node_id="no_such_zone", text="(this text should not appear anywhere)"),
    )


def main() -> None:
    nodes = hotel_room_scene()
    demo_targets("render_subregions_block", render_subregions_block, nodes)
    demo_targets("render_embedded_block", render_embedded_block, nodes)
    print()


if __name__ == "__main__":
    main()
