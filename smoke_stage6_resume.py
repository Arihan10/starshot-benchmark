"""Tiled-resume check (needs the scene smoke_stage6_tiled.py leaves behind):
interrupt a tiled run after 5 tiles, rerun with resume, and assert the finished
tiles load from cache, the rest train, and the output verifies.

Run on the GPU box, AFTER smoke_stage6_tiled.py:  python smoke_stage6_resume.py
"""

from __future__ import annotations

import json
from pathlib import Path

from splat.stage6 import TrainParams, train_splat

OUT = Path(__file__).parent / "runs" / "_smoke" / "stage6_tiled"
PARAMS = TrainParams(
    iterations=300, refine_start_iter=100, tile_max=6000, lod_levels=0,
    ckpt_every=0, eval_max_views=6, log_every=200,
    batch=1,  # pin the reference (batch-1) schedule: this tests resume mechanics
)


class Stop(Exception):
    pass


def interrupting(done: int, total: int, msg: str) -> None:
    if msg.startswith("[tile 6/"):
        raise Stop("simulated interrupt before tile 6")


try:
    train_splat(
        run="smoke", slot="resume", model="x",
        colmap_dir=OUT / "colmap",
        out_path=OUT / "trained_resume.ply",
        params=PARAMS, resume=False, progress=interrupting,
    )
    raise SystemExit("expected the simulated interrupt to fire")
except Stop:
    pass

manifest = json.loads((OUT / "ckpt" / "tiles" / "tiles.json").read_text(encoding="utf-8"))
n_done = len(manifest["done"])
assert n_done == 5, f"expected 5 finished tiles cached, got {n_done}"
print(f"interrupted with {n_done} tiles cached")

cached_hits: list[str] = []


def counting(done: int, total: int, msg: str) -> None:
    if "cached" in msg:
        cached_hits.append(msg)


summary = train_splat(
    run="smoke", slot="resume", model="x",
    colmap_dir=OUT / "colmap",
    out_path=OUT / "trained_resume.ply",
    params=PARAMS, resume=True, progress=counting,
)
assert len(cached_hits) == 5, f"expected 5 cache hits, got {len(cached_hits)}"
assert summary["tiles"] is not None and len(summary["tiles"]) == 20
assert summary["splats_final"] > 5000
assert not (OUT / "ckpt").exists(), "ckpt root should be cleaned after export"
print("resume summary:", summary["splats_final"], "splats,",
      f"psnr={summary['metrics']['psnr'] if summary['metrics'] else None}")
print("PASS: tiled resume skips finished tiles and completes")
