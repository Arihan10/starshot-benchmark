"""Full reordered-pipeline chain on a REAL de-optimized cell: Stage 2 -> 3 -> 4 -> 5.
Validates the new artifact contracts end to end and renders a few real views on GPU.

Uses .smoke_work/vanilla (21-object modern-house cell, de-optimized earlier).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from splat import stage2, stage3, stage4, stage5  # noqa: E402

VANILLA = ROOT / ".smoke_work" / "vanilla"
OUT = ROOT / ".smoke_work" / "splat2"
OUT.mkdir(parents=True, exist_ok=True)
FAILS: list[str] = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def timed(label, fn):
    t = time.time(); r = fn(); print(f"  [{label}] {time.time()-t:.1f}s"); return r


if not VANILLA.is_dir():
    print(f"missing {VANILLA} — run the de-optimize step first"); sys.exit(2)

# Stage 2 — free space FIRST
s2 = timed("stage2", lambda: stage2.compute_free_space(
    run="c", slot="c", model="c", raw_dir=VANILLA, out_path=OUT / stage2.FREESPACE_NAME))
print(f"   stage2: dims_fine={s2['dims_fine']} reachable={s2['reachable_voxels']} "
      f"free={s2['free_voxels']} pitch_fine={s2['pitch_fine']}")
check("Stage 2 wrote freespace.npz + voxels.bin",
      (OUT / stage2.FREESPACE_NAME).is_file() and (OUT / stage2.VOXELS_NAME).is_file())
check("reachability kept a navigable component but dropped some hollows",
      0 < s2["reachable_voxels"] <= s2["free_voxels"])

# Stage 3 — surfels consume the grid
s3 = timed("stage3", lambda: stage3.sample_cell(
    run="c", slot="c", model="c", raw_dir=VANILLA,
    freespace_path=OUT / stage2.FREESPACE_NAME, out_path=OUT / stage3.CLOUD_NAME))
print(f"   stage3: kept={s3['splats']} culled_hidden={s3['culled_hidden']} "
      f"({100*s3['culled_hidden']/max(s3['sampled'],1):.1f}% of {s3['sampled']})")
hdr = (OUT / stage3.CLOUD_NAME).read_bytes()[:400].decode("ascii", "replace")
check("Stage 3 emitted a 2DGS cloud (no scale_2)", "scale_2" not in hdr.split("end_header")[0])
check("Stage 3 culled some hidden faces on the real scene", s3["culled_hidden"] > 0)

# Stage 4 — cameras from grid + cloud (no meshes)
s4 = timed("stage4", lambda: stage4.plan_cameras(
    run="c", slot="c", model="c", freespace_path=OUT / stage2.FREESPACE_NAME,
    surfels_path=OUT / stage3.CLOUD_NAME, out_path=OUT / stage4.CAMERAS_NAME))
print(f"   stage4: patches={s4['patches']} cameras={s4['cameras']} "
      f"coverage={s4['coverage']['satisfied_pct']}%")
check("Stage 4 wrote cameras.json + patches.bin + patch_views.json",
      all((OUT / n).is_file() for n in
          (stage4.CAMERAS_NAME, stage4.PATCHES_NAME, stage4.PATCH_VIEWS_NAME)))

# Trim the plan to a few cameras so the GPU render is quick.
plan = json.loads((OUT / stage4.CAMERAS_NAME).read_text(encoding="utf-8"))
plan["cameras"] = plan["cameras"][:8]
smoke_cams = OUT / "cameras_smoke.json"
smoke_cams.write_text(json.dumps(plan), encoding="utf-8")
n_views = sum(len(c.get("faces", [])) for c in plan["cameras"])
print(f"   rendering {len(plan['cameras'])} cameras -> {n_views} views")

# Stage 5 — GPU render of the real meshes from the real plan
refs = OUT / "refs"
s5 = timed("stage5", lambda: stage5.render_references(
    run="c", slot="c", model="c", raw_dir=VANILLA, cameras_path=smoke_cams, out_dir=refs))
print(f"   stage5: views={s5['views']} materials={s5['materials']} res={s5['resolution']}")
check("Stage 5 rendered the expected number of views", s5["views"] == n_views)
check("Stage 5 wrote transforms.json", (refs / stage5.TRANSFORMS_NAME).is_file())

doc = json.loads((refs / stage5.TRANSFORMS_NAME).read_text(encoding="utf-8"))
check("transforms tagged opencv_c2w + log-uint16 PNG depth",
      doc["convention"] == "opencv_c2w" and doc["depth"] == stage5.DEPTH_ENCODING)
# spot-check one rendered view: depth positive where alpha>0, alpha in [0,1]
f0 = doc["frames"][0]
depth = stage5.load_depth_png(refs / f0["depth_path"], doc["near"], doc["far"])
alpha = np.asarray(Image.open(refs / f0["alpha_path"]), np.float32) / 255.0
hit = alpha > 0.5
check("rendered depth is finite + positive on hit pixels",
      bool(hit.any()) and np.isfinite(depth[hit]).all() and float(depth[hit].min()) > 0,
      f"{int(hit.sum())} hit px, depth range [{float(depth[hit].min()):.2f},{float(depth[hit].max()):.2f}] m")
Image.open(refs / f0["file_path"]).save(OUT / "preview_real.png")

print("\n" + ("FULL REORDERED CHAIN OK" if not FAILS else f"FAILURES: {FAILS}"))
print(f"artifacts: {OUT}")
sys.exit(1 if FAILS else 0)
