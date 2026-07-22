"""Post-hoc occlusion compaction of an EXISTING trained cell, on Modal (A100).

Runs the Stage-6 §COMPACTION measurement against a cell that already finished
training: every Gaussian's TOTAL rendered blend weight is accumulated over ALL
of the cell's Stage-5 reference views (through the rasterizer itself — the exact
quantity pixels see; only the poses are needed), and Gaussians below `eps`
(default 0.5/255 — unable to move any single rendered pixel by half an 8-bit
display step even with their whole mass on one pixel) are deleted. Survivors are
NOT re-optimized, and gradient-blind pure-black Gaussians are force-kept, so the
result is the trained model minus provably-unrenderable Gaussians.

`trained.ply` is never touched: the result lands beside it on the Volume as
`trained.compact.ply`, with a `compact.json` summary. The pass then VERIFIES
itself two ways, so the fidelity claim is measured, not asserted:
  * original-vs-compacted renders on a random view sample — max per-pixel RGB /
    alpha delta and PSNR between the two models;
  * the Stage-6 reference-view eval (`_evaluate`: foreground PSNR / RGB L1 /
    depth L1) for BOTH models on the same seeded view subset.

The local entrypoint spawns the GPU function, then pulls
`trained.compact.ply` + `compact.json` back into the local run dir.

Usage (repo root; modal lives in the server env):
    cd server && uv run modal run ../splat/modal_compact.py \
        --cell ../runs/good_opus_new_hotel2/hotel-room/opus-new [--eps 0.00196]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

import splat.modal_app as modal_app  # image / volume / names (its App stays in its own namespace)

app = modal.App("starshot-splat-compact")


@app.function(
    image=modal_app.image,
    gpu="A100-40GB",
    cpu=4.0,
    memory=8192,
    timeout=3600,
    volumes={modal_app.VOL: modal_app.volume},
)
def compact_cell(spec: dict[str, Any]) -> dict[str, Any]:
    """Measure -> delete -> verify for one cell on the Volume. Returns (and
    writes to `splat/compact.json`) the full summary."""
    import time

    import numpy as np
    import torch
    from gsplat import rasterization_2dgs

    from splat.stage6 import (
        TrainParams,
        _blind_mask,
        _contribution_weights,
        _encode_trained_ply,
        _evaluate,
        _load_cloud,
        _read_transforms,
        _render_batch,
        _render_inputs,
    )

    run, slot, model = spec["run"], spec["slot"], spec["model"]
    eps = float(spec["eps"])
    diff_views = int(spec["diff_views"])

    modal_app.volume.reload()
    cell = Path(modal_app.VOL) / "cells" / run / slot / model / "splat"
    trained = cell / "trained.ply"
    if not trained.is_file():
        raise FileNotFoundError(f"{trained} not on the Volume — train the cell first (stage 6)")

    # Reference poses + supervision index, exactly as train_splat builds them.
    refs_dir = cell / "refs"
    doc = _read_transforms(refs_dir)
    frames = doc.get("frames", [])
    if not frames:
        raise RuntimeError(f"{refs_dir}/transforms.json has no frames")
    device = torch.device("cuda")
    width, height = int(doc["w"]), int(doc["h"])
    K = torch.tensor(
        [[doc["fl_x"], 0.0, doc["cx"]], [0.0, doc["fl_y"], doc["cy"]], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    depth_near = float(doc["near"]) if "near" in doc else None
    depth_far = float(doc["far"]) if "far" in doc else None
    views = []
    for fr in frames:
        c2w = np.asarray(fr["transform_matrix"], dtype=np.float64)
        views.append(
            {
                "viewmat": torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)),
                "frame": refs_dir / fr["frame_path"] if fr.get("frame_path") else None,
                "rgb": refs_dir / fr["file_path"] if fr.get("file_path") else None,
                "alpha": refs_dir / fr["alpha_path"] if fr.get("alpha_path") else None,
                "depth": refs_dir / fr["depth_path"] if fr.get("depth_path") else None,
                "depth_near": depth_near,
                "depth_far": depth_far,
            }
        )

    init = _load_cloud(trained)
    splats = {k: torch.from_numpy(v).to(device) for k, v in init.items()}
    n = int(splats["means"].shape[0])
    params = TrainParams()

    def log(done: int, total: int, msg: str) -> None:
        print(f"[compact {run}/{slot}/{model}] {done}/{total} {msg}", flush=True)

    # --- measure ------------------------------------------------------------
    t0 = time.perf_counter()
    weights = _contribution_weights(
        torch, rasterization_2dgs, splats, views, K, width, height, params, log
    )
    blind = _blind_mask(torch, splats["sh0"])
    measure_s = time.perf_counter() - t0

    # Sensitivity: how many Gaussians fall under a few candidate thresholds
    # (before the blind force-keep), so the knob's shape is visible in one run.
    w_np = weights.detach().cpu().numpy()
    sensitivity = {
        f"{t:g}": int(((w_np <= t) & ~blind.cpu().numpy()).sum())
        for t in (0.0, 0.5 / 2550, 0.5 / 255, 1.0 / 255, 2.0 / 255)
    }

    # --- delete ---------------------------------------------------------------
    keep = (weights > eps) | blind
    removed = int((~keep).sum())
    kept = {k: v[keep] for k, v in splats.items()}
    arrays = {k: v.detach().cpu().numpy() for k, v in kept.items()}
    quats = arrays["quats"] / (np.linalg.norm(arrays["quats"], axis=1, keepdims=True) + 1e-12)
    out_path = cell / "trained.compact.ply"
    _encode_trained_ply(
        arrays["means"], quats.astype(np.float32), arrays["sh0"].reshape(-1, 3),
        arrays["opacities"], arrays["scales"][:, :2], out_path,
    )

    # --- verify A: original vs compacted, same poses --------------------------
    t1 = time.perf_counter()
    rng = np.random.default_rng(params.seed)
    sample = rng.choice(len(views), size=min(diff_views, len(views)), replace=False)
    max_rgb = max_alpha = 0.0
    sq_sum, px_count = 0.0, 0
    with torch.no_grad():
        for i in sample:
            viewmats = views[int(i)]["viewmat"].to(device).unsqueeze(0)
            outs = []
            for model_splats in (splats, kept):
                colors, sh_deg = _render_inputs(torch, model_splats, params.sh_degree)
                renders, alpha, *_ = _render_batch(
                    torch, rasterization_2dgs, model_splats, colors, sh_deg,
                    viewmats, K, width, height, params, dist_on=False,
                )
                outs.append((renders[..., :3].clamp(0, 1), alpha))
            d_rgb = (outs[0][0] - outs[1][0]).abs()
            d_alpha = (outs[0][1] - outs[1][1]).abs()
            max_rgb = max(max_rgb, float(d_rgb.max()))
            max_alpha = max(max_alpha, float(d_alpha.max()))
            sq_sum += float((d_rgb ** 2).sum())
            px_count += d_rgb.numel()
    mse = sq_sum / max(px_count, 1)
    psnr_between = float(-10.0 * np.log10(mse + 1e-20))

    # --- verify B: both models against the reference frames -------------------
    ev_before = _evaluate(torch, rasterization_2dgs, splats, views, K, width, height, params, device)
    ev_after = _evaluate(torch, rasterization_2dgs, kept, views, K, width, height, params, device)
    verify_s = time.perf_counter() - t1

    summary = {
        "cell": f"{run}/{slot}/{model}",
        "eps": eps,
        "views": len(views),
        "splats_in": n,
        "splats_kept": int(keep.sum()),
        "splats_removed": removed,
        "removed_pct": round(100.0 * removed / max(n, 1), 2),
        "blind_force_kept": int(blind.sum()),
        "strictly_zero_weight": int((w_np == 0.0).sum()),
        "eps_sensitivity_counts": sensitivity,
        "bytes_in": trained.stat().st_size,
        "bytes_out": out_path.stat().st_size,
        "diff_vs_original": {
            "views_sampled": int(len(sample)),
            "max_abs_rgb": round(max_rgb, 6),
            "max_abs_rgb_8bit_steps": round(max_rgb * 255.0, 3),
            "max_abs_alpha": round(max_alpha, 6),
            "psnr_between_models": round(psnr_between, 2),
        },
        "eval_vs_refs_before": ev_before,
        "eval_vs_refs_after": ev_after,
        "seconds": {"measure": round(measure_s, 1), "verify": round(verify_s, 1)},
        "out_path": str(out_path),
    }
    (cell / "compact.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    modal_app.volume.commit()
    return summary


@app.local_entrypoint()
def main(cell: str, eps: float = 0.5 / 255.0, diff_views: int = 128) -> None:
    """Compact one cell (`--cell runs/R/S/M`) and pull the artifacts back beside
    the local trained.ply."""
    cell_dir = Path(cell).resolve()
    run, slot, model = cell_dir.parts[-3], cell_dir.parts[-2], cell_dir.parts[-1]
    summary = compact_cell.remote(
        {"run": run, "slot": slot, "model": model, "eps": eps, "diff_views": diff_views}
    )
    print(json.dumps(summary, indent=1))

    vol = modal.Volume.from_name(modal_app.VOLUME_NAME, version=2)
    remote_base = f"cells/{run}/{slot}/{model}/splat"
    local = cell_dir / "splat"
    local.mkdir(parents=True, exist_ok=True)
    for name in ("trained.compact.ply", "compact.json"):
        dst = local / name
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with tmp.open("wb") as f:
            for chunk in vol.read_file(f"{remote_base}/{name}"):
                f.write(chunk)
        tmp.replace(dst)
        print(f"pulled {name} ({dst.stat().st_size / 1e6:.1f} MB)", flush=True)
