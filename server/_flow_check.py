"""Scratch check for the overlapped walk: does an anchor start mid-walk, as soon
as its surroundings settle, and does the drain pick up the rest?

Drives `generation.run_parallel` in exactly the order `divider._build` calls it,
with `generation.run` stubbed so nothing hits an LLM.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace as NS

from app.core.types import BoundingBox, Node
from app.pipeline import committed, generation
from app.utils import logging as slotlog

RID = "flow-test"
RD = Path(".")


def zone(i, p, o, d):
    return Node(id=i, prompt=i, bbox=BoundingBox(origin=o, dimensions=d),
                parent_id=p, plan="x", is_zone=True)


# root 10x3x10 split into two flush atomic halves.
root = zone("root", None, (0, 0, 0), (10, 3, 10))
A = zone("A", "root", (0, 0, 0), (5, 3, 10))
B = zone("B", "root", (5, 0, 0), (5, 3, 10))

PLANNED: dict[str, bool] = {}      # id -> is_atomic, as the walk plans each zone
SHELL: set[str] = set()            # ids whose encapsulating pass has committed
KIDS = {"root": ["A", "B"]}
BOXES = {"root": 1, "A": 1, "B": 1}
calls: list[tuple[str, str]] = []


async def fake_run(*, zone, runs_dir, run_id, scenario, all_nodes):
    calls.append((scenario, zone.id))
    if scenario == "encapsulating":
        SHELL.add(zone.id)
        return
    await asyncio.sleep(0)         # a real pass yields; make sure we do too
    all_nodes.append(
        Node(id=f"{zone.id}_obj", prompt="o",
             bbox=BoundingBox(origin=(0, 0, 0), dimensions=(1, 1, 1)),
             parent_id=zone.id, is_zone=False)
    )


generation.run = fake_run
committed.zone_plan = lambda i: NS(is_atomic=PLANNED[i], plan="p") if i in PLANNED else None
committed.object_specs = lambda z, s: ([] if z in SHELL else None) if s == "encapsulating" else None
committed.bbox = lambda i: BOXES.get(i)
committed.zone_decompose = lambda i: NS(subregions=[NS(id=k) for k in KIDS[i]]) if i in KIDS else None
slotlog.log_once = lambda *a, **k: None


def pending():
    return list(generation._pending_anchors.get(RID, {}))


def started():
    return [z for s, z in calls if s == "anchor"]


async def rp(z, scenario, nodes):
    await generation.run_parallel(
        zone=z, runs_dir=RD, run_id=RID, scenario=scenario, all_nodes=nodes,
    )


async def main() -> None:
    nodes = [root, A, B]           # root decomposed; both children placed
    PLANNED["root"] = False

    await rp(root, "encapsulating", nodes)
    print(f"root framed          pending={pending()} started={started()}")

    PLANNED["A"] = True            # walk plans A, then builds it
    await rp(A, "encapsulating", nodes)
    await rp(A, "anchor", nodes)
    await asyncio.sleep(0)
    print(f"A framed + queued    pending={pending()} started={started()}"
          f"   <- B unplanned, so A waits")

    PLANNED["B"] = True            # walk plans B, then builds it
    await rp(B, "encapsulating", nodes)
    await asyncio.sleep(0)
    print(f"B framed             pending={pending()} started={started()}"
          f"   <- A's surroundings settled mid-walk")
    await rp(B, "anchor", nodes)
    await asyncio.sleep(0)
    print(f"B queued             pending={pending()} started={started()}")

    await generation.drain_anchors(runs_dir=RD, run_id=RID, all_nodes=nodes)
    print(f"drained              pending={pending()} started={started()}")
    print()
    print("call order :", calls)
    print("scene      :", [n.id for n in nodes])

    ok = (
        started() == ["A", "B"]
        and calls.index(("anchor", "A")) < calls.index(("anchor", "B"))
        and {"A_obj", "B_obj"} <= {n.id for n in nodes}
        and not pending()
        and not generation._anchor_tasks.get(RID)
    )
    print()
    print("ALL PASS" if ok else "FAILURE")


asyncio.run(main())
