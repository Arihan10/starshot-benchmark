"""Where does a cell's stage-3 splat budget go, object by object? (read-only)

Re-samples each placed GLB exactly as `stage3.sample_cell` would (same scene-scale
k, per-object spacing, thinning, cull, coarsening) and reports per-object splat
counts, byte shares (64 B/splat in the raw float32 PLY), areas, the spacing each
object resolved to, and its coarsening-level mix — sorted by cost, with a rough
name-based category rollup.

Usage:
    python -m scripts.analyze_splat_sizes <cell_dir> [raw_subdir]
e.g.
    python -m scripts.analyze_splat_sizes \
        runs/good_opus_new_hotel2/swamp-land/gemini-pro generated/2/objects-generated-optimized
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from splat import stage3 as s3
from splat.assets import load_geoms
from splat.stage2 import load_free_space

_GROUPS = (
    ("wall", ("wall",)),
    ("ground/terrain", ("ground", "terrain", "mud", "floor", "bank", "island", "path")),
    ("water", ("water", "pond", "swamp_pool", "duckweed", "lily")),
    ("vegetation", ("tree", "cypress", "cattail", "moss", "vine", "bush", "reed",
                    "grass", "fern", "mangrove", "root")),
    ("rock", ("rock", "boulder", "stone", "rubble")),
)


def group_of(name: str) -> str:
    low = name.lower()
    for label, keys in _GROUPS:
        if any(k in low for k in keys):
            return label
    return "other"


def main() -> None:
    cell = Path(sys.argv[1])
    raw = cell / (sys.argv[2] if len(sys.argv) > 2 else "objects-optimized")
    fs = load_free_space(cell / "splat" / "freespace.npz")
    k = s3._scene_scale(fs.cand_clear)
    base = s3.SampleParams().spacing

    rows = []
    for nid in s3.placed_object_ids(raw):
        tot = 0
        area = 0.0
        spacing = None
        levels: dict[int, int] = defaultdict(int)
        for geom in load_geoms(raw / f"{nid}.glb"):
            if len(geom.faces) == 0 or geom.area <= 0:
                continue
            area += float(geom.area)
            part, _ = s3._sample_object(geom, base, k, fs)
            if part is None:
                continue
            spacing = s3._object_spacing(geom, base, k)
            r = part["radius"]
            tot += len(r)
            for j in range(0, s3._COARSEN_OCTAVES + 1):
                rr = s3._RADIUS_FRAC * spacing * (2 ** j)
                c = int((np.abs(r - rr) < rr * 0.05).sum())
                if c:
                    levels[2 ** j] += c
        rows.append((nid, tot, area, spacing, dict(levels)))

    rows.sort(key=lambda r: -r[1])
    total = sum(r[1] for r in rows)
    print(f"scene_scale k={k:.2f}  base={base*100:.2f} cm  "
          f"total={total:,} splats ≈ {total*64/1e6:.0f} MB (raw f32 PLY)\n")

    print(f"{'object':34s} {'splats':>11} {'share':>6} {'cum':>6} "
          f"{'MB':>7} {'area m²':>9} {'sp cm':>6}  levels")
    cum = 0
    for nid, tot_o, area, sp, lv in rows[:15]:
        cum += tot_o
        lvs = " ".join(f"{a}s:{c:,}" for a, c in sorted(lv.items()))
        print(f"{nid[:34]:34s} {tot_o:>11,} {100*tot_o/total:5.1f}% {100*cum/total:5.1f}% "
              f"{tot_o*64/1e6:7.1f} {area:9.0f} "
              f"{(sp or 0)*100:6.1f}  {lvs}")

    agg: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for nid, tot_o, area, _sp, _lv in rows:
        g = agg[group_of(nid)]
        g[0] += tot_o
        g[1] += area
    print("\ncategory rollup:")
    for label, (cnt, area) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f"  {label:16s} {int(cnt):>11,} splats ({100*cnt/total:4.1f}%) "
              f"≈ {cnt*64/1e6:6.1f} MB   area {area:9.0f} m²")


if __name__ == "__main__":
    main()
