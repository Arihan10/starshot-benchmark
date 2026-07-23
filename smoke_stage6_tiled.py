"""End-to-end smoke test for Stage-6 TILED training + LOD export.

Needs no pipeline artifacts: builds a synthetic surfel scene (checkered ground,
walls, colored pillars), renders PNG references with gsplat itself (exact poses,
like Stage 5 provides), EXPORTS a COLMAP model from them (points3D + cameras +
images — the Postshot-style input Stage 6 now trains from), then trains TILED with
a tiny budget that forces a multi-tile grid, and asserts the merged output + LOD
ladder. A short single-run control guards the untiled path against regressions.

Run on the GPU box:  python smoke_stage6_tiled.py
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from splat.colmap import export_colmap
from splat.stage6 import (
    _SH_C0,
    TrainParams,
    _encode_trained_ply,
    _load_cloud,
    train_splat,
)

OUT = Path(__file__).parent / "runs" / "_smoke" / "stage6_tiled"
RES = 200
FOCAL = 160.0


def _surfels(points: np.ndarray, normals: np.ndarray, rgb: np.ndarray, radius: float):
    n = len(points)
    f_dc = (rgb - 0.5) / _SH_C0
    opacity = np.full(n, 6.0, dtype=np.float32)  # sigmoid ~ 0.9975
    scales2 = np.full((n, 2), np.log(radius), dtype=np.float32)
    # +Z-aligned frames rotated onto the normal (matches stage3's convention).
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0 + normals[:, 2]
    quats[:, 1] = -normals[:, 1]
    quats[:, 2] = normals[:, 0]
    anti = quats[:, 0] < 1e-6
    quats[anti] = (0.0, 1.0, 0.0, 0.0)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return points.astype(np.float32), quats, f_dc.astype(np.float32), opacity, scales2


def build_scene(cloud_path: Path) -> None:
    """Checkered 16x16 m ground + two walls + 9 pillars, ~46k surfels."""
    rng = np.random.default_rng(7)
    pts, nrm, col = [], [], []

    xs = np.arange(-8.0, 8.0, 0.08)
    gx, gz = np.meshgrid(xs, xs, indexing="ij")
    g = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1)
    checker = ((np.floor(g[:, 0]) + np.floor(g[:, 2])) % 2).astype(bool)
    c = np.where(checker[:, None], [0.85, 0.82, 0.75], [0.25, 0.45, 0.3])
    pts.append(g)
    nrm.append(np.tile([0.0, 1.0, 0.0], (len(g), 1)))
    col.append(c)

    for wx, color in ((-8.0, [0.7, 0.3, 0.25]), (8.0, [0.25, 0.35, 0.7])):
        ys = np.arange(0.0, 2.5, 0.08)
        zs = np.arange(-8.0, 8.0, 0.08)
        wy, wz = np.meshgrid(ys, zs, indexing="ij")
        w = np.stack([np.full(wy.size, wx), wy.ravel(), wz.ravel()], axis=1)
        pts.append(w)
        nrm.append(np.tile([-np.sign(wx), 0.0, 0.0], (len(w), 1)))
        col.append(np.tile(color, (len(w), 1)))

    for px, pz in [(-5, -5), (0, -5), (5, -5), (-5, 0), (0, 0), (5, 0), (-5, 5), (0, 5), (5, 5)]:
        th = rng.uniform(0, 2 * np.pi, 900)
        h = rng.uniform(0.0, 2.0, 900)
        r = 0.4
        p = np.stack([px + r * np.cos(th), h, pz + r * np.sin(th)], axis=1)
        n = np.stack([np.cos(th), np.zeros(900), np.sin(th)], axis=1)
        pts.append(p)
        nrm.append(n)
        col.append(np.tile(rng.uniform(0.2, 0.95, 3), (900, 1)))

    points = np.concatenate(pts).astype(np.float64)
    normals = np.concatenate(nrm).astype(np.float64)
    rgb = np.concatenate(col).astype(np.float64)
    means, quats, f_dc, opacity, scales2 = _surfels(points, normals, rgb, radius=0.07)
    _encode_trained_ply(means, quats, f_dc, opacity, scales2, cloud_path)
    print(f"scene: {len(means)} surfels -> {cloud_path.name}")


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / np.linalg.norm(f)
    up = np.array([0.0, 1.0, 0.0]) if abs(f[1]) < 0.95 else np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = r, d, f, eye
    return c2w


def cameras() -> list[np.ndarray]:
    cams = []
    for i in range(28):  # orbit ring, two heights
        a = 2 * np.pi * i / 28
        h = 3.5 if i % 2 == 0 else 6.5
        eye = np.array([11.0 * np.cos(a), h, 11.0 * np.sin(a)])
        cams.append(_look_at(eye, np.array([0.0, 0.5, 0.0])))
    for cx in (-5.0, 0.0, 5.0):  # interior coverage for every tile
        for cz in (-5.0, 0.0, 5.0):
            eye = np.array([cx + 1.5, 5.0, cz + 1.5])
            cams.append(_look_at(eye, np.array([cx, 0.0, cz])))
    return cams


def render_refs(cloud_path: Path, refs_dir: Path) -> None:
    import torch
    from gsplat import rasterization_2dgs
    from PIL import Image

    dev = torch.device("cuda")
    init = _load_cloud(cloud_path)
    splats = {k: torch.from_numpy(v).to(dev) for k, v in init.items()}
    K = torch.tensor(
        [[FOCAL, 0.0, RES / 2], [0.0, FOCAL, RES / 2], [0.0, 0.0, 1.0]],
        dtype=torch.float32, device=dev,
    )
    frames = []
    refs_dir.mkdir(parents=True, exist_ok=True)
    for i, c2w in enumerate(cameras()):
        vm = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)).to(dev).unsqueeze(0)
        with torch.no_grad():
            renders, alpha, *_rest = rasterization_2dgs(
                means=splats["means"], quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=splats["sh0"], viewmats=vm, Ks=K.unsqueeze(0),
                width=RES, height=RES, sh_degree=0, packed=False,
                render_mode="RGB+ED",
            )
        rgb = renders[0, ..., :3].clamp(0, 1).cpu().numpy()
        dep = renders[0, ..., 3].cpu().numpy().astype(np.float32)
        alp = alpha[0, ..., 0].clamp(0, 1).cpu().numpy()
        Image.fromarray((rgb * 255).astype(np.uint8)).save(refs_dir / f"v{i:03d}.png")
        Image.fromarray((alp * 255).astype(np.uint8), mode="L").save(refs_dir / f"v{i:03d}_a.png")
        np.save(refs_dir / f"v{i:03d}_d.npy", dep)
        frames.append({
            "transform_matrix": c2w.tolist(),
            "file_path": f"v{i:03d}.png",
            "alpha_path": f"v{i:03d}_a.png",
            "depth_path": f"v{i:03d}_d.npy",
        })
    doc = {"w": RES, "h": RES, "fl_x": FOCAL, "fl_y": FOCAL, "cx": RES / 2, "cy": RES / 2,
           "frames": frames}
    (refs_dir / "transforms.json").write_text(json.dumps(doc), encoding="utf-8")
    print(f"refs: {len(frames)} views -> {refs_dir}")


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cloud, refs = OUT / "cloud.ply", OUT / "refs"
    colmap_dir = OUT / "colmap"
    build_scene(cloud)
    render_refs(cloud, refs)
    # Stage 6 trains from a COLMAP model now — export one (points3D + cameras +
    # images) from the rendered refs + the surfel cloud, exactly like
    # splat_to_colmap.py, then train the tiler from it.
    export_colmap(refs, cloud, colmap_dir)
    print(f"colmap model -> {colmap_dir}")

    def log(done: int, total: int, msg: str) -> None:
        print(f"  [{done}/{total}] {msg}", flush=True)

    print("\n=== tiled run (tile_max=6000 forces a grid) ===")
    t0 = time.perf_counter()
    tiled = train_splat(
        run="smoke", slot="tiled", model="x",
        colmap_dir=colmap_dir, out_path=OUT / "trained_tiled.ply",
        params=TrainParams(
            iterations=500, refine_start_iter=100,
            tile_max=6000, lod_levels=2, lod_min_count=1500,
            ckpt_every=0, eval_max_views=12, log_every=100,
            batch=1,  # pin the reference (batch-1) schedule: this tests tiling
        ),
        resume=False, progress=log,
    )
    print(f"tiled done in {time.perf_counter() - t0:.1f}s")

    assert tiled["tiles"] is not None and len(tiled["tiles"]) >= 4, "expected a multi-tile grid"
    assert all(t["views"] > 0 for t in tiled["tiles"]), "a tile trained with no views"
    assert tiled["splats_final"] > 5000, "merged model suspiciously small"
    # Point-cloud init (opacity/scale/orientation reset the gsplat way) needs more
    # steps than the old surfel init to reach a given PSNR, so this short run only
    # guards that the tiled path trains + merges coherently, not final quality.
    assert tiled["metrics"] is not None and tiled["metrics"]["psnr"] > 12.0, (
        f"tiled PSNR too low: {tiled['metrics']}"
    )
    assert tiled["lod"], "LOD ladder missing"
    reloaded = _load_cloud(OUT / "trained_tiled.ply")
    assert int(reloaded["means"].shape[0]) == tiled["splats_final"]
    lo, hi = reloaded["means"].min(0), reloaded["means"].max(0)
    assert (lo > -12).all() and (hi < 12).all(), f"merged AABB out of bounds: {lo} {hi}"
    for lv in tiled["lod"]:
        p = OUT / lv["path"]
        assert p.is_file() and lv["splats"] < tiled["splats_final"]
    print("tiled summary:", json.dumps({k: tiled[k] for k in (
        "splats_init", "splats_final", "splats_depth_seeded", "metrics")}, indent=1))
    print("tiles:", json.dumps(tiled["tiles"], indent=1))
    print("lod:", json.dumps(tiled["lod"], indent=1))

    print("\n=== single-run control (tile_max=0) ===")
    single = train_splat(
        run="smoke", slot="single", model="x",
        colmap_dir=colmap_dir, out_path=OUT / "trained_single.ply",
        params=TrainParams(
            iterations=250, refine_start_iter=100, tile_max=0,
            lod_levels=1, lod_min_count=1500, ckpt_every=0,
            eval_max_views=8, log_every=100,
            batch=1,  # pin the reference (batch-1) schedule: single-run control
        ),
        resume=False, progress=log,
    )
    assert single["tiles"] is None, "single run must not tile"
    assert single["metrics"] is not None and single["metrics"]["psnr"] > 12.0, (
        f"single PSNR too low: {single['metrics']}"
    )
    print("single summary:", json.dumps({k: single[k] for k in (
        "splats_init", "splats_final", "metrics")}, indent=1))

    print("\nPASS: tiled training + LOD export verified end-to-end")


if __name__ == "__main__":
    main()
