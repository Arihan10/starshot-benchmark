"""Verify the focus scope: serial renders byte-identical, parallel renders narrow.

Loads a real cell, renders the scene context unscoped (what the serial pipeline
sends) and scoped (what the overlapped pipeline now sends), and checks both the
size delta and that the right regions kept their contents.
"""

import hashlib
import json
import sys
from collections import defaultdict

from app.core import scene_context, util
from app.core.types import BoundingBox, Node
from app.pipeline import generation

CELL = sys.argv[1] if len(sys.argv) > 1 else (
    r"d:\starshot\starshot-benchmark\runs\new-parallel-test\modern-house\muse-spark-contributor"
)


def load():
    nodes = []
    # Straight line iteration: the buffered reader keeps only the current line
    # resident, where hand-rolled chunk slicing copies the tail on every newline.
    with open(f"{CELL}\\events.jsonl", "rb") as f:
        for line in f:
            if len(line) > 200_000 or b'"bbox"' not in line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("kind") == "bbox":
                o, d = e.get("origin"), e.get("dimensions")
                if not (isinstance(o, list) and isinstance(d, list)):
                    continue
                is_zone = e.get("node_kind") == "zone"
                nodes.append(Node(
                    id=e["id"], prompt=e.get("prompt") or e["id"],
                    bbox=BoundingBox(origin=tuple(o), dimensions=tuple(d)),
                    parent_id=e.get("parent_id"),
                    parent_region=e.get("parent_region") if not is_zone else None,
                    is_zone=is_zone,
                    plan=(e.get("prompt") or "zone") if is_zone else None,
                ))
    return nodes


nodes = load()
zones = [n for n in nodes if n.is_zone]
objects = [n for n in nodes if not n.is_zone]
by_region = defaultdict(list)
for o in objects:
    by_region[o.parent_region].append(o)
print(f"scene: {len(zones)} zones, {len(objects)} objects")

ok = True


def check(label, got, want):
    global ok
    ok &= got == want
    print(f"{'PASS' if got == want else 'FAIL'}  {label:52} got={got!s:6} want={want}")


# --- serial: nothing bound, nothing changes -----------------------------------
# Hashed and released one at a time: the full block is ~1.3MB and the renderer's
# recursive indent/join makes transient copies of it at every level.
def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


scene_context.bind_focus_scope(None)
target = max(zones, key=lambda z: len(by_region.get(z.id, [])))
rendered = scene_context.render_embedded_block(nodes, node_id=target.id, text="t")
full_len, full_hash = len(rendered), digest(rendered)
del rendered
rendered = scene_context.render_embedded_block(nodes, node_id=target.id, text="t", detail=None)
check("unbound render == explicit detail=None", digest(rendered) == full_hash, True)
del rendered
full = " " * full_len  # only its length is needed from here on

# --- parallel: scope bound ----------------------------------------------------
scene_context.bind_focus_scope(generation.focus_scope_for("focus-test"))
scoped = scene_context.render_embedded_block(
    nodes, node_id=target.id, text="t", detail=scene_context.focus_detail(target.id, nodes),
)
detail = scene_context.focus_detail(target.id, nodes)
touching = util.touching_zones(target.id, nodes)

print()
print(f"target   : {target.id}")
print(f"detail   : {len(detail)} of {len(zones)} zones ({100*len(detail)/len(zones):.0f}%)  "
      f"= self + {len(touching)} touching + 0 twins")
print(f"full     : {len(full):>9,} chars")
print(f"scoped   : {len(scoped):>9,} chars   ({100*len(scoped)/len(full):.1f}% retained, "
      f"{100-100*len(scoped)/len(full):.1f}% saved)")
print()

# Every detail zone with objects should still list them; no other zone should.
kept_wrongly = [
    z.id for z in zones
    if z.id not in detail and by_region.get(z.id)
    and f'Objects placed directly within "{z.id}"' in scoped
]
dropped_wrongly = [
    z.id for z in zones
    if z.id in detail and by_region.get(z.id)
    and f'Objects placed directly within "{z.id}"' not in scoped
]
check("no non-detail zone leaked its objects", len(kept_wrongly), 0)
if dropped_wrongly:
    print(f"   dropped: {dropped_wrongly}  (root is expected — render_root_objects "
          f"renders its objects separately, by design)")
# The root's own objects are global geometry rendered by `render_root_objects`,
# never inside the embedded block, so it can't be judged here.
check("every detail zone kept its objects", [z for z in dropped_wrongly if z != "root"], [])
check("target is in its own detail set", target.id in detail, True)
check("scoped is strictly smaller", len(scoped) < len(full), True)

# A neighbour nested inside a collapsed container must still render in full.
nested = [
    z for z in zones
    if z.id in detail and z.parent_id and z.parent_id not in detail and by_region.get(z.id)
]
if nested:
    n = nested[0]
    check(f"neighbour under a collapsed parent kept ({n.id[:26]}…)",
          f'Objects placed directly within "{n.id}"' in scoped, True)
else:
    print("note: no detail zone sits under a collapsed parent in this scene")

scene_context.bind_focus_scope(None)
print()
print("ALL PASS" if ok else "FAILURES PRESENT")
