"""Verify twin detection fires at plan time, one-sided.

Drives the real `divider._build` with its LLM helpers stubbed, recording what the
`before_plan` hook is handed for each zone. What matters is WHEN it runs (before
the plan, after the zone is placed) and WHAT it can see (everything conceived so
far, and nothing later).
"""

import asyncio
from types import SimpleNamespace as NS

from app.core.types import BoundingBox, Node
from app.pipeline import divider
from app.utils import logging as slotlog

# root -> [A, B];  A -> [A1, A2];  B atomic.
TREE = {"root": ["A", "B"], "A": ["A1", "A2"]}
ATOMIC = {"A": False, "B": True, "A1": True, "A2": True}
BOX = BoundingBox(origin=(0, 0, 0), dimensions=(1, 1, 1))

seen: list[tuple[str, list[str]]] = []
order: list[str] = []


async def fake_plan(*, zone_id, zone_prompt, nodes):
    order.append(f"plan:{zone_id}")
    return NS(plan="p", is_atomic=ATOMIC.get(zone_id, True))


async def fake_decompose(*, node, all_nodes):
    kids = TREE.get(node.id, [])
    return NS(subregions=[
        NS(id=k, prompt=k, proxy_shape=None, placement=None, referenced_ids=[],
           model_dump=lambda: {})
        for k in kids
    ])


async def fake_boxes(*, parent, children, all_nodes):
    return {c.id: BOX for c in children}


async def fake_gen(**kw):
    order.append(f"gen:{kw['scenario']}:{kw['zone'].id}")


async def hook(zone: Node, nodes: list[Node]) -> None:
    order.append(f"twins:{zone.id}")
    seen.append((zone.id, sorted(n.id for n in nodes if n.is_zone)))


divider._plan_zone = fake_plan
divider._decompose_zone = fake_decompose
divider._resolve_child_bboxes_batch = fake_boxes
divider.uniquify_ids = lambda specs, existing_ids: []
divider.validate_subregions = lambda *a, **k: None
slotlog.emit_step = lambda *a, **k: None
slotlog.emit_bbox = lambda *a, **k: None
slotlog.log_once = lambda *a, **k: None
slotlog.log = lambda *a, **k: None


async def main() -> None:
    root = Node(id="root", prompt="root", bbox=BOX, parent_id=None, plan="p", is_zone=True)
    nodes = [root]
    await divider._build(
        node=root, runs_dir=None, run_id="t", all_nodes=nodes,
        is_atomic=False, gen_run=fake_gen, before_plan=hook,
    )

    print("call order:")
    for o in order:
        print("  ", o)
    print()
    print("what each zone could see when its twins were decided:")
    for zid, zones in seen:
        print(f"  {zid:5} -> {zones}")

    ok = True

    def check(label, got, want):
        nonlocal ok
        ok &= got == want
        print(f"{'PASS' if got == want else 'FAIL'}  {label:46} got={got!s:34} want={want}")

    print()
    # Depth-first: A is planned, its whole subtree is built, then B.
    check("every child got a twin pass", [z for z, _ in seen], ["A", "A1", "A2", "B"])
    check("twins decided before the plan",
          order.index("twins:A") < order.index("plan:A"), True)
    check("A could not see its nephews yet",
          [z for z in dict(seen)["A"] if z.startswith("A1")], [])
    check("A1 saw its own sibling", "A2" in dict(seen)["A1"], True)
    check("B, conceived alongside A, saw A", "A" in dict(seen)["B"], True)
    # One-sided: A was decided before A1/A2 existed and is never revisited.
    check("A's view stayed frozen", dict(seen)["A"], ["A", "B", "root"])

    print()
    print("ALL PASS" if ok else "FAILURES PRESENT")


asyncio.run(main())
