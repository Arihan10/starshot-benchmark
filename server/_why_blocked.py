"""Why is a zone's anchor still queued?

Rebuilds the committed state from a cell's events.jsonl, then runs the REAL
`generation.neighbours_framed` against it and reports, per frontier region, which
condition is failing.
"""

import json
import sys
from types import SimpleNamespace as NS

from app.core import util
from app.core.types import BoundingBox, Node
from app.pipeline import committed, generation

CELL = sys.argv[1] if len(sys.argv) > 1 else (
    r"d:\starshot\starshot-benchmark\runs\new-parallel-test\modern-house\new-opus-new"
)
TARGET = sys.argv[2] if len(sys.argv) > 2 else "driveway_parking_apron"

nodes: list[Node] = []
PLAN: dict[str, NS] = {}
DECOMP: dict[str, NS] = {}
SPECS: dict[tuple[str, str], list] = {}
BOX: dict[str, BoundingBox] = {}

buf = b""
with open(f"{CELL}\\events.jsonl", "rb") as f:
    while True:
        chunk = f.read(8 * 1024 * 1024)
        if not chunk:
            break
        buf += chunk
        while True:
            nl = buf.find(b"\n")
            if nl == -1:
                break
            line, buf = buf[:nl], buf[nl + 1 :]
            if len(line) > 200_000:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            k = e.get("kind")
            if k == "bbox":
                o, d = e.get("origin"), e.get("dimensions")
                if not (isinstance(o, list) and isinstance(d, list)):
                    continue
                bb = BoundingBox(origin=tuple(o), dimensions=tuple(d))
                BOX[e["id"]] = bb
                nodes.append(
                    Node(
                        id=e["id"],
                        prompt=e.get("prompt") or e["id"],
                        bbox=bb,
                        parent_id=e.get("parent_id"),
                        is_zone=e.get("node_kind") == "zone",
                        plan="x" if e.get("node_kind") == "zone" else None,
                    )
                )
            elif k == "divider.zone_plan":
                PLAN[e["node"]] = NS(is_atomic=bool(e.get("is_atomic")), plan="p")
            elif k == "divider.zone_decompose":
                kids = [c.get("id") for c in (e.get("children") or []) if isinstance(c, dict)]
                DECOMP[e["node"]] = NS(subregions=[NS(id=i) for i in kids])
            elif k == "generation.decompose":
                SPECS[(e.get("zone"), e.get("scenario"))] = [
                    NS(id=o.get("id")) for o in (e.get("objects") or []) if isinstance(o, dict)
                ]
            elif k == "generation.decompose.no_objects":
                SPECS[(e.get("zone"), e.get("scenario"))] = []

committed.zone_plan = lambda i: PLAN.get(i)
committed.zone_decompose = lambda i: DECOMP.get(i)
committed.object_specs = lambda z, s: SPECS.get((z, s))
committed.bbox = lambda i: BOX.get(i)

by_id = {n.id: n for n in nodes}
print(f"cell   : {CELL}")
print(f"nodes  : {len(nodes)}  ({sum(1 for n in nodes if n.is_zone)} zones)")
print(f"planned: {len(PLAN)}   decomposed: {len(DECOMP)}   framed: {sum(1 for (z, s) in SPECS if s == 'encapsulating')}")
print()

if TARGET not in by_id:
    print(f"!! {TARGET} has no bbox event yet")
    raise SystemExit(1)

print(f"TARGET {TARGET}")
print(f"  plan       : {PLAN.get(TARGET)}")
print(f"  own shell  : settled={generation._shell_settled(TARGET)}")
print(f"  GATE       : neighbours_framed = {generation.neighbours_framed(TARGET, nodes)}")
print()

raw = util.adjacent_zones(TARGET, nodes)
tb = by_id[TARGET].bbox
print(f"ray hits ({len(raw)}) — 'far' ones are across negative space and do not gate:")
for r in raw:
    gap = generation._aabb_gap(tb, r.bbox)
    touching = gap <= generation._CONTACT_TOLERANCE_M
    plan = PLAN.get(r.id)
    ok = generation._region_settled(r)
    if not touching:
        print(f"  [far] {r.id[:62]:62} gap={gap:6.2f} m  (ignored)")
        continue
    if plan is None:
        why = "UNPLANNED — placed but zone_plan has not committed"
    elif plan.is_atomic:
        why = f"atomic, shell_settled={generation._shell_settled(r.id)}"
        if not generation._shell_settled(r.id):
            why += f"  (encapsulating specs={SPECS.get((r.id, 'encapsulating'))!r})"
    else:
        sp = generation._subregions_placed(r.id)
        sh = generation._shell_settled(r.id)
        why = f"subdivided, subregions_placed={sp}, shell_settled={sh}"
        if not sp:
            d = DECOMP.get(r.id)
            if d is None:
                why += "  (no zone_decompose yet)"
            else:
                missing = [s.id for s in d.subregions if s.id not in BOX]
                why += f"  (children without bbox: {missing})"
    print(f"  [{'OK ' if ok else 'BLK'}] {r.id[:62]:62} gap={gap:6.2f} m  {why}")
