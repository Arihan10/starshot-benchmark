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


def _build_views(torch, np, doc: dict[str, Any], refs_dir: Path):  # noqa: ANN001
    """Reference views exactly as `train_splat` builds them: per frame the w2c
    `viewmat`, the camera-to-world (for scene scale), and the supervision frame
    paths + shared depth near/far. Returns (views, K_cpu [3,3] float32 tensor,
    width, height)."""
    frames = doc.get("frames", [])
    if not frames:
        raise RuntimeError(f"{refs_dir}/transforms.json has no frames")
    width, height = int(doc["w"]), int(doc["h"])
    K = torch.tensor(
        [[doc["fl_x"], 0.0, doc["cx"]], [0.0, doc["fl_y"], doc["cy"]], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    depth_near = float(doc["near"]) if "near" in doc else None
    depth_far = float(doc["far"]) if "far" in doc else None
    views = []
    for fr in frames:
        c2w = np.asarray(fr["transform_matrix"], dtype=np.float64)
        views.append(
            {
                "viewmat": torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)),
                "c2w": c2w,
                "frame": refs_dir / fr["frame_path"] if fr.get("frame_path") else None,
                "rgb": refs_dir / fr["file_path"] if fr.get("file_path") else None,
                "alpha": refs_dir / fr["alpha_path"] if fr.get("alpha_path") else None,
                "depth": refs_dir / fr["depth_path"] if fr.get("depth_path") else None,
                "depth_near": depth_near,
                "depth_far": depth_far,
            }
        )
    return views, K, width, height


def _diff_models(torch, np, rasterization_2dgs, model_a, model_b, views, K, width, height, params, n_sample):  # noqa: ANN001
    """Render two splat models on the same seeded view sample and measure how
    differently they draw: max per-pixel |ΔRGB| / |Δalpha| and the PSNR between
    the two render sets. This is the model-vs-model fidelity check (needs no
    reference pixels)."""
    from splat.stage6 import _render_batch, _render_inputs

    device = K.device
    rng = np.random.default_rng(params.seed)
    sample = rng.choice(len(views), size=min(n_sample, len(views)), replace=False)
    max_rgb = max_alpha = 0.0
    sq_sum, px_count = 0.0, 0
    with torch.no_grad():
        for i in sample:
            viewmats = views[int(i)]["viewmat"].to(device).unsqueeze(0)
            outs = []
            for model_splats in (model_a, model_b):
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
    return {
        "views_sampled": int(len(sample)),
        "max_abs_rgb": round(max_rgb, 6),
        "max_abs_rgb_8bit_steps": round(max_rgb * 255.0, 3),
        "max_abs_alpha": round(max_alpha, 6),
        "psnr_between_models": round(float(-10.0 * np.log10(mse + 1e-20)), 2),
    }


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
    device = torch.device("cuda")
    views, K, width, height = _build_views(torch, np, _read_transforms(refs_dir), refs_dir)
    K = K.to(device)

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
        sh_rest=arrays.get("shN"),  # preserve view-dependent colour through compaction
    )

    # --- verify A: original vs compacted, same poses --------------------------
    t1 = time.perf_counter()
    diff = _diff_models(
        torch, np, rasterization_2dgs, splats, kept, views, K, width, height, params, diff_views
    )

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
        "diff_vs_original": diff,
        "eval_vs_refs_before": ev_before,
        "eval_vs_refs_after": ev_after,
        "seconds": {"measure": round(measure_s, 1), "verify": round(verify_s, 1)},
        "out_path": str(out_path),
    }
    (cell / "compact.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    modal_app.volume.commit()
    return summary


@app.function(
    image=modal_app.image,
    gpu="A100-40GB",
    cpu=4.0,
    memory=8192,
    timeout=7200,
    volumes={modal_app.VOL: modal_app.volume},
)
def slim_cell(spec: dict[str, Any]) -> dict[str, Any]:
    """EXPERIMENT: contribution-ranked prune to a target count + short heal +
    full verification — the measured (NOT guaranteed) middle path between the
    provably-lossless compaction (~1% removed) and naive alpha·area cutting
    (which visibly destroys detail).

    1. Measure every Gaussian's total rendered blend weight over ALL reference
       views (exact, through the rasterizer — the same measure compact_cell uses).
    2. Keep the top `keep` by that weight (gradient-blind pure-black Gaussians
       are always kept); delete the rest — these are the splats the renders use
       LEAST, mostly transmittance-starved stacked layers, unlike alpha·area
       which deletes small-but-visible detail.
    3. HEAL: `heal_steps` optimizer steps of the standard Stage-6 supervision
       loss on the survivors (no densification, means LR damped — the model is
       already converged), so neighbors absorb the removed splats' residuals.
    4. Verify: render slim-vs-original on a seeded view sample (max per-pixel
       delta + PSNR between models) and run the reference-view eval for both.

    Writes trained.slim.ply + slim.json beside trained.ply on the Volume; the
    original is never modified. This is an experiment runner: `keep` is a
    per-invocation choice, not a pipeline cap."""
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization_2dgs

    from splat.stage6 import (
        TrainParams,
        _blind_mask,
        _contribution_weights,
        _encode_trained_ply,
        _evaluate,
        _gaussian_window,
        _load_cloud,
        _read_transforms,
        _render_batch,
        _render_inputs,
        _supervision_loss,
        _view_stream,
    )

    run, slot, model = spec["run"], spec["slot"], spec["model"]
    keep_target = int(spec["keep"])
    heal_steps = int(spec["heal_steps"])
    diff_views = int(spec["diff_views"])
    means_lr_frac = float(spec.get("means_lr_frac", 0.1))

    modal_app.volume.reload()
    cell = Path(modal_app.VOL) / "cells" / run / slot / model / "splat"
    trained = cell / "trained.ply"
    if not trained.is_file():
        raise FileNotFoundError(f"{trained} not on the Volume")
    refs_dir = cell / "refs"
    device = torch.device("cuda")
    views, K, width, height = _build_views(torch, np, _read_transforms(refs_dir), refs_dir)
    K = K.to(device)
    params = TrainParams()

    init = _load_cloud(trained)
    original = {k: torch.from_numpy(v).to(device) for k, v in init.items()}
    n = int(original["means"].shape[0])

    def log(done: int, total: int, msg: str) -> None:
        print(f"[slim {run}/{slot}/{model}] {done}/{total} {msg}", flush=True)

    # --- 1. measure + 2. rank-prune -------------------------------------------
    t0 = time.perf_counter()
    weights = _contribution_weights(
        torch, rasterization_2dgs, original, views, K, width, height, params, log
    )
    blind = _blind_mask(torch, original["sh0"])
    n_blind = int(blind.sum())
    keep = blind.clone()
    n_rank = max(keep_target - n_blind, 0)
    if n_rank > 0:
        w_rank = weights.clone()
        w_rank[blind] = -1.0  # blind ones already kept; exclude from ranking
        top = torch.topk(w_rank, k=min(n_rank, n - n_blind)).indices
        keep[top] = True
    removed = int(n - int(keep.sum()))
    measure_s = time.perf_counter() - t0
    log(1, 1, f"pruned {removed:,} of {n:,} by measured contribution "
              f"(kept {int(keep.sum()):,} = top-{keep_target:,} incl. {n_blind:,} blind)")

    # Survivors become trainable parameters for the heal.
    splats = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v[keep].detach().clone()) for k, v in original.items()}
    )

    # --- 3. heal ----------------------------------------------------------------
    # Stage-6 LRs (batch-scaled), means damped by `means_lr_frac`: the model is
    # already converged, the heal only redistributes the removed splats' residual
    # onto neighbors (opacity/scale/color do most of the work).
    t1 = time.perf_counter()
    centers = np.stack([v["c2w"][:3, 3] for v in views])
    scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) * 1.1
    b = max(int(params.batch), 1)
    lr_scale = float(np.sqrt(b))
    lrs = {
        "means": params.means_lr * scene_scale * lr_scale * means_lr_frac,
        "scales": params.scales_lr * lr_scale,
        "quats": params.quats_lr * lr_scale,
        "opacities": params.opacities_lr * lr_scale,
        "sh0": params.sh0_lr * lr_scale,
        # shN is present whenever the trained splat is view-dependent (SH>0); the
        # optimizer loop below iterates splats.keys(), so its LR must exist here.
        "shN": params.shN_lr * lr_scale,
    }
    optimizers = {
        name: torch.optim.Adam([{"params": [splats[name]], "lr": lrs[name]}], eps=1e-15)
        for name in splats.keys()
    }
    window = _gaussian_window(torch, 11, 1.5, device, channels=3)

    import threading

    stop_ev = threading.Event()
    stream = _view_stream(torch, views, device, b, params.prefetch, params.seed, 0, stop=stop_ev)
    for step in range(heal_steps):
        viewmats, gt_rgb, gt_alpha, gt_depth = next(stream)
        colors, sh_deg = _render_inputs(torch, splats, params.sh_degree)
        renders, pred_alpha, normals, nfd, distort, median_depth, _info = _render_batch(
            torch, rasterization_2dgs, splats, colors, sh_deg, viewmats, K,
            width, height, params, dist_on=False,
        )
        pred_rgb = renders[..., :3]
        pred_depth = median_depth if params.depth_mode == "median" else renders[..., 3:4]
        loss = _supervision_loss(
            torch, F, params, window, gt_rgb, gt_alpha, gt_depth,
            pred_rgb, pred_alpha, pred_depth, normals, nfd, distort, None,
            normals_active=True, dist_active=False,
        )
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if step % 50 == 0 or step == heal_steps - 1:
            log(step + 1, heal_steps, f"heal loss={float(loss):.4f} n={int(splats['means'].shape[0])}")
    stop_ev.set()
    heal_s = time.perf_counter() - t1

    # --- export + 4. verify -----------------------------------------------------
    slim = {k: v.detach() for k, v in splats.items()}
    arrays = {k: v.cpu().numpy() for k, v in slim.items()}
    quats = arrays["quats"] / (np.linalg.norm(arrays["quats"], axis=1, keepdims=True) + 1e-12)
    out_path = cell / "trained.slim.ply"
    _encode_trained_ply(
        arrays["means"], quats.astype(np.float32), arrays["sh0"].reshape(-1, 3),
        arrays["opacities"], arrays["scales"][:, :2], out_path,
        sh_rest=arrays.get("shN"),  # preserve view-dependent colour through the slim
    )

    t2 = time.perf_counter()
    diff = _diff_models(
        torch, np, rasterization_2dgs, original, slim, views, K, width, height, params, diff_views
    )
    ev_before = _evaluate(torch, rasterization_2dgs, original, views, K, width, height, params, device)
    ev_after = _evaluate(torch, rasterization_2dgs, slim, views, K, width, height, params, device)
    verify_s = time.perf_counter() - t2

    summary = {
        "cell": f"{run}/{slot}/{model}",
        "keep_target": keep_target,
        "heal_steps": heal_steps,
        "means_lr_frac": means_lr_frac,
        "views": len(views),
        "splats_in": n,
        "splats_kept": int(keep.sum()),
        "splats_removed": removed,
        "removed_pct": round(100.0 * removed / max(n, 1), 2),
        "blind_force_kept": n_blind,
        "bytes_in": trained.stat().st_size,
        "bytes_out": out_path.stat().st_size,
        "diff_vs_original": diff,
        "eval_vs_refs_original": ev_before,
        "eval_vs_refs_slim": ev_after,
        "seconds": {"measure": round(measure_s, 1), "heal": round(heal_s, 1), "verify": round(verify_s, 1)},
        "out_path": str(out_path),
    }
    (cell / "slim.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    modal_app.volume.commit()
    return summary


@app.function(
    image=modal_app.image,
    gpu="A100-40GB",
    cpu=4.0,
    memory=8192,
    timeout=1800,
    volumes={modal_app.VOL: modal_app.volume},
)
def smoke_train(spec: dict[str, Any]) -> dict[str, Any]:
    """Non-destructive densification smoke test: run a short Stage-6 fine-tune on
    a cell's init cloud + refs (both read-only from the Volume) to a SCRATCH out
    path — never `trained.ply` — purely to confirm the gsplat DefaultStrategy
    densification step runs. The absgrad crash fired inside `_update_state`, which
    runs every step from step 0, so even a few hundred steps that reach the refine
    window is a decisive check. Returns the run summary (init/final counts)."""
    from pathlib import Path as _P

    from splat.stage6 import TrainParams, train_splat

    run, slot, model = spec["run"], spec["slot"], spec["model"]
    iterations = int(spec.get("iterations", 1500))

    modal_app.volume.reload()
    cell = _P(modal_app.VOL) / "cells" / run / slot / model
    refs = cell / "splat" / "refs"
    if not (refs / "transforms.json").is_file():
        raise FileNotFoundError(f"{refs}/transforms.json not on the Volume (run stage 5)")

    scratch = _P("/tmp/smoke") / run / slot / model
    scratch.mkdir(parents=True, exist_ok=True)
    scratch_out = scratch / "trained.smoke.ply"

    # The init cloud is the Stage-3 surfel cloud stored in the cell's content-
    # addressed inputs (raw or `.zst` transport), NOT the splat/ dir — mirror how
    # modal_app materializes it for training.
    inputs = cell / "inputs"
    if (inputs / "cloud.ply").is_file():
        cloud = inputs / "cloud.ply"
    elif (inputs / "cloud.ply.zst").is_file():
        cloud = scratch / "cloud.ply"
        modal_app._zstd_decompress(inputs / "cloud.ply.zst", cloud)
    else:
        raise FileNotFoundError(f"cloud.ply[.zst] not under {inputs}")

    def log(done: int, total: int, msg: str) -> None:
        print(f"[smoke {run}/{slot}/{model}] {done}/{total} {msg}", flush=True)

    summary = train_splat(
        run=run, slot=slot, model=model,
        cloud_path=cloud, refs_dir=refs, out_path=scratch_out,
        # Small run, densification ON (the crashing path), compaction + LODs off
        # to keep it quick. Nothing is written to the Volume.
        params=TrainParams(iterations=iterations, compact=False, lod_levels=0, ckpt_every=0),
        resume=False,
        progress=log,
    )
    return {
        "ok": True,
        "cell": f"{run}/{slot}/{model}",
        "splats_init": summary["splats_init"],
        "splats_final": summary["splats_final"],
        "iterations_steps": summary["iterations"],
        "metrics": summary.get("metrics"),
    }


@app.local_entrypoint()
def slim(cell: str, keep: int = 350_000, heal_steps: int = 1000, diff_views: int = 128) -> None:
    """Run the contribution-prune + heal experiment on a cell and pull the
    artifacts back beside the local trained.ply."""
    cell_dir = Path(cell).resolve()
    run, slot, model = cell_dir.parts[-3], cell_dir.parts[-2], cell_dir.parts[-1]
    summary = slim_cell.remote(
        {"run": run, "slot": slot, "model": model, "keep": keep,
         "heal_steps": heal_steps, "diff_views": diff_views}
    )
    print(json.dumps(summary, indent=1))

    vol = modal.Volume.from_name(modal_app.VOLUME_NAME, version=2)
    remote_base = f"cells/{run}/{slot}/{model}/splat"
    local = cell_dir / "splat"
    local.mkdir(parents=True, exist_ok=True)
    for name in ("trained.slim.ply", "slim.json"):
        dst = local / name
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with tmp.open("wb") as f:
            for chunk in vol.read_file(f"{remote_base}/{name}"):
                f.write(chunk)
        tmp.replace(dst)
        print(f"pulled {name} ({dst.stat().st_size / 1e6:.1f} MB)", flush=True)


@app.local_entrypoint()
def smoke(cell: str, iterations: int = 1500) -> None:
    """Run the non-destructive densification smoke test on a cell."""
    cell_dir = Path(cell).resolve()
    run, slot, model = cell_dir.parts[-3], cell_dir.parts[-2], cell_dir.parts[-1]
    summary = smoke_train.remote(
        {"run": run, "slot": slot, "model": model, "iterations": iterations}
    )
    print(json.dumps(summary, indent=1))


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
