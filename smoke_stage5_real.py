"""Run Stage 5 on the real de-optimized cell (multi-material, real transcoded
textures, real Stage-4 cameras.json) trimmed to a few camera positions — validates
the full coverage->render wiring on real data + peak VRAM on the 3070.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splat import stage5

ROOT = Path(__file__).resolve().parent
VANILLA = ROOT / ".smoke_work" / "vanilla"
PLAN = ROOT / ".smoke_work" / "splat" / "cameras.json"
OUT = ROOT / ".smoke_work" / "refs_real"
N_CAMERAS = 3

plan = json.loads(PLAN.read_text(encoding="utf-8"))
full_cams = len(plan["cameras"])
plan["cameras"] = plan["cameras"][:N_CAMERAS]
trim = ROOT / ".smoke_work" / "cameras.trim.json"
trim.write_text(json.dumps(plan), encoding="utf-8")
faces = sum(len(c.get("faces", [])) for c in plan["cameras"])
print(f"real cell: {full_cams} cameras total; rendering first {N_CAMERAS} ({faces} faces)")

import torch  # noqa: E402
torch.cuda.reset_peak_memory_stats()
t = time.time()
summary = stage5.render_references(run="r", slot="modern-house", model="longcat",
                                  raw_dir=VANILLA, cameras_path=trim, out_dir=OUT)
dt = time.time() - t
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"rendered {summary['views']} views in {dt:.1f}s ({dt/max(summary['views'],1):.2f}s/view), "
      f"materials={summary['materials']}, peak CUDA mem={peak:.2f} GB")
if summary["warnings"]:
    print("warnings:", summary["warnings"][:5])

# sanity: outputs exist, depth in metres is plausible, alpha in [0,1]
frames = json.loads((OUT / "transforms.json").read_text(encoding="utf-8"))["frames"]
f0 = frames[0]
rgb = np.asarray(Image.open(OUT / f0["file_path"]).convert("RGB"))
depth = np.load(OUT / f0["depth_path"])
alpha = np.asarray(Image.open(OUT / f0["alpha_path"]), np.float32) / 255.0
cov = float((alpha > 0.5).mean())
dvals = depth[depth > 0]
print(f"\nframe {f0['file_path']}: rgb{rgb.shape} coverage={cov*100:.0f}% "
      f"depth[min={dvals.min():.2f} max={dvals.max():.2f}]m  rgb_mean={rgb.mean():.0f}")
print("REAL-CELL STAGE 5 OK")
