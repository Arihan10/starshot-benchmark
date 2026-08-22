"""Measure what a neighbour-scoped scene context would save.

Renders each zone's context twice with the REAL renderer — once with every
object in the scene (what we send today) and once with objects kept only for the
zone itself, the zones physically touching it, and N stand-in "twins" — then
weights the difference by the tokens each zone's calls actually burned.

Usage: uv run python _ctx_saving.py [cell_dir] [twins]
"""

import json
import sqlite3
import sys
from collections import defaultdict

from app.core import scene_context, util
from app.core.types import BoundingBox, Node

CELL = sys.argv[1] if len(sys.argv) > 1 else (
    r"d:\starshot\starshot-benchmark\runs\new-parallel-test\modern-house\muse-spark-contributor"
)
TWINS = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# Same contact rule the interior gate uses: overlapping counts as touching, so a
# zone's ancestors (which contain it) come along automatically.
TOL = 0.05
# Structural steps whose prompt carries the full SCENE_CONTEXT.
STRUCT = {
    "zone_plan", "zone_decompose", "child_bbox_batch", "encapsulating_decompose",
    "anchor_decompose", "object_bbox_batch", "next_object", "negative_space_decompose",
}


def load_nodes():
    nodes, buf = [], b""
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
                line, buf = buf[:nl], buf[nl + 1:]
                if len(line) > 200_000 or b'"bbox"' not in line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("kind") != "bbox":
                    continue
                o, d = e.get("origin"), e.get("dimensions")
                if not (isinstance(o, list) and isinstance(d, list)):
                    continue
                is_zone = e.get("node_kind") == "zone"
                nodes.append(Node(
                    id=e["id"],
                    prompt=e.get("prompt") or e["id"],
                    bbox=BoundingBox(origin=tuple(o), dimensions=tuple(d)),
                    parent_id=e.get("parent_id"),
                    parent_region=e.get("parent_region") if not is_zone else None,
                    is_zone=is_zone,
                    plan=(e.get("prompt") or "zone") if is_zone else None,
                ))
    return nodes


def gap(a, b):
    return max(
        max(a.min_corner[i] - b.max_corner[i], b.min_corner[i] - a.max_corner[i], 0.0)
        for i in range(3)
    )


def main() -> None:
    nodes = load_nodes()
    zones = [n for n in nodes if n.is_zone]
    objects = [n for n in nodes if not n.is_zone]
    by_region = defaultdict(list)
    for o in objects:
        by_region[o.parent_region].append(o)
    print(f"cell    : {CELL.split(chr(92))[-2]}/{CELL.split(chr(92))[-1]}")
    print(f"scene   : {len(zones)} zones, {len(objects)} objects")

    touching = {
        z.id: {w.id for w in zones if w.id != z.id and gap(z.bbox, w.bbox) <= TOL}
        for z in zones
    }
    avg_n = sum(len(v) for v in touching.values()) / max(len(zones), 1)
    print(f"contact : {avg_n:.1f} touching zones per zone on average "
          f"(min {min(len(v) for v in touching.values())}, max {max(len(v) for v in touching.values())})")

    # Tokens each zone actually spent, so the saving is weighted by where the
    # spend really is rather than treated as if every zone cost the same.
    con = sqlite3.connect(f"{CELL}\\flights.db")
    spend = defaultdict(int)
    calls = defaultdict(int)
    for node, step, tin in con.execute(
        "SELECT node, step, tokens_in FROM flights WHERE ok=1 AND tokens_in IS NOT NULL"
    ):
        if step in STRUCT and node:
            spend[node] += tin
            calls[node] += 1
    total_tokens = sum(spend.values())
    print(f"spend   : {total_tokens:,} structural input tokens over {sum(calls.values())} calls")
    print()

    # Twin stand-ins: the biggest zones this one does NOT touch — a pessimistic
    # proxy, since a real twin is usually a same-size sibling, not the largest
    # region in the scene.
    by_size = sorted(zones, key=lambda z: -len(by_region.get(z.id, [])))

    # The unscoped block is identical whatever zone is targeted (only the inline
    # target marker moves), so render it once instead of per zone.
    full_len = len(scene_context.render_embedded_block(nodes, node_id=zones[0].id, text="target"))

    rows = []
    weighted_full = weighted_cut = 0
    for z in zones:
        if not spend.get(z.id):
            continue
        keep = {z.id} | touching[z.id]
        if TWINS:
            extra = [w.id for w in by_size if w.id not in keep][:TWINS]
            keep |= set(extra)
        # The real code path: `detail` is what the bound focus scope produces.
        cut_len = len(scene_context.render_embedded_block(
            nodes, node_id=z.id, text="target", detail=keep,
        ))
        frac = cut_len / max(full_len, 1)
        rows.append((z.id, full_len, cut_len, frac, spend[z.id]))
        weighted_full += spend[z.id]
        weighted_cut += spend[z.id] * frac

    rows.sort(key=lambda r: -r[4])
    print(f"{'zone':46} {'kept':>6} {'ctx chars':>11} {'reduced':>10} {'tokens':>12}")
    for zid, fu, cu, frac, tk in rows[:14]:
        print(f"{zid[:46]:46} {frac*100:5.1f}% {fu:>11,} {cu:>10,} {tk:>12,}")

    print()
    saving = 1 - (weighted_cut / max(weighted_full, 1))
    print(f"TWINS PER ZONE ASSUMED : {TWINS}")
    print(f"context retained       : {100 * weighted_cut / max(weighted_full,1):.1f}%")
    print(f"SCENE_CONTEXT SAVING   : {saving*100:.1f}%  (token-weighted across every structural call)")
    print()
    # SCENE_CONTEXT is most of a structural prompt but not all of it; the rest is
    # the template, the zone's own block, TO_PLACE, etc.
    for share in (0.75, 0.85, 0.95):
        print(f"  if SCENE_CONTEXT is {share*100:.0f}% of a prompt -> "
              f"{saving*share*100:.1f}% of ALL structural input tokens "
              f"({int(total_tokens*saving*share):,} of {total_tokens:,})")


main()
