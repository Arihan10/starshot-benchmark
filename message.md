I have the full picture. Here's what I found.

## The headline

`batch: 16` in the job spec silently cut the run from ~24,500 optimizer steps to **1,532**, while `status.json` reports `"iterations": 30000`. So when you compared step counts with Postshot, you were comparing 30,000 against 1,532.

```501:507:splat/stage6.py
        b = max(int(self.batch), 1)
        budget = (
            max(1, round(self.epochs * max(n_views, 1)))
            if self.epochs is not None
            else max(1, int(self.iterations))
        )
        steps = max(_MIN_STEPS, round(budget / b))
```

`epochs=15 × 1634 views = 24,510` view-draws, divided by `batch=16` → 1,532 Adam updates. `as_summary()` reports the *unresolved* fields, so the recorded summary says `iterations: 30000, refine_stop_iter: 15000` when the loop actually ran 1,532 steps with densification stopping at 766.

gsplat's own reference 2DGS trainer does the opposite: `max_steps = 30_000` is the optimizer-step count and `batch_size` multiplies images *per step* (batch 16 → 480k image draws, still 30k steps). Postshot's step box is also optimizer steps. The docstring's claim that batch is "a pure SPEED knob at constant work" is false for an iterative optimizer — Adam's progress is bounded by the number of updates, and the `sqrt(16) = 4×` LR compensation recovers only a quarter of what was lost while quadrupling per-step jitter.

## The evidence

I wrote a CPU 2DGS rasterizer and rendered the Stage-3 init, the trained model, and the healed model from two Stage-5 reference poses:

![splat comparison](d:\starshot\benchmark-2\scratch_out\comparison.png)

The Stage-3 surfel cloud is already almost the reference. Training turns it into exactly the luminous haze you described — and stage 7 doesn't remove it. **The fine-tune makes the scene strictly worse than its own initialization.**

## Ranked causes

**1. 1,532 optimizer steps.** Every cadence inherits the 16× cut: densification runs steps 31→766 with `refine_every = 6`, so a cloned Gaussian gets **six** Adam updates before the next densification decision (reference: 100), and there are only 766 steps of settling afterwards (reference: 15,000). Wall clock isn't the constraint — stage 6 took 40 minutes; `batch=1` at 30k steps costs roughly the same and gives 20× the updates.

**2. The loss is mostly metric depth, not photometry.**

```2393:2399:splat/stage6.py
    if gt_depth is not None and params.depth_lambda > 0.0 and depth_active:
        gate = (gt_alpha > params.alpha_gate) & (gt_depth > 0)
        if mask is not None:
            gate = gate & (mask > 0)
        if gate.any():
            dl = ((pred_depth - gt_depth).abs() * gate).sum() / (gate.sum() + 1e-8)
            loss = loss + params.depth_lambda * dl
```

From the run's own metrics: depth term = `0.5 × 0.345 = 0.173`; the entire photometric part ≈ 0.05. Depth is ~70% of the converged loss and far more early on. gsplat's reference depth loss is **off by default**, and when enabled uses `depth_lambda = 1e-2` in **disparity** space on **sparse** points. Ours is 50× the weight, dense, and in metres so it scales with room size.

It's also a pathological gradient. With `depth_mode="median"`, gsplat's 2DGS backward attributes the whole per-pixel depth gradient to the single Gaussian that crossed transmittance 0.5, unweighted:

```cuda
// RasterizeToPixels2DGSBwd.cu
if (batch_end - t == median_idx) {
    // v_median is a special gradient input from forward pass
    // not yet clear what this is for
    v_rgb_local[CDIM - 1] += v_median;
}
```

Every pixel yanks exactly one Gaussian along its ray at full strength. And it never converges anyway — final depth L1 is 0.345 m on views whose median depth is 0.3–3 m.

**3. All Gaussian-size control is disabled.** `reset_every = 0` is passed to gsplat as a sentinel past the last step:

```2672:2675:splat/stage6.py
        refine_start_iter=params.refine_start_iter,
        refine_stop_iter=params.resolved_refine_stop,
        reset_every=params.reset_every if params.reset_every > 0 else params.iterations + 1,
        refine_every=params.refine_every,
```

But in `DefaultStrategy._prune_gs` the size prunes sit behind `if step > self.reset_every:`, so setting `reset_every=0` removes three mechanisms at once: the opacity reset (intended), `prune_scale3d`, and `prune_scale2d`. Nothing bounds Gaussian size during training. In the artifact: max disc radius **2.74 m** in a 9.4×3.6×7.4 m room, anisotropy up to **256:1**, and the 1.17% of Gaussians with radius > 0.3 m carry **33.5% of all rendered opacity·area**. That is the bloom.

The comment justifying this ("Stage 7 settles them geometrically via `surface_max_dist`") doesn't hold: only **0.05%** of trained Gaussians are more than 0.6 m from a Stage-3 surfel. The junk is near-surface haze (median 5.9 cm off-surface, p90 11.6 cm), invisible to the surface prior.

**4. The run used the from-points init, not the surfels.** `status.json` records `init: "points"` and `depth_start_iter: 500` — the latter is only possible on the points branch, since the surfels branch rewrites it to 0. Every branch in the repo has `init: str = "surfels"`, so **the deployed Modal code differs from the repo**. That discarded on-surface means, mesh-true disc orientation, exact texel colour and solid opacity, replacing them with random quaternions and opacity 0.1.

It also re-armed a trap the surfel path deliberately guards against. `_load_cloud` synthesizes a tiny third scale:

```649:651:splat/stage6.py
        two = stack("scale_0", "scale_1")
        third = (two.min(axis=1, keepdims=True) + np.log(0.01)).astype(np.float32)
        scales = np.concatenate([two, third], axis=1)
```

`_init_from_points` does not:

```767:768:splat/stage6.py
    log_scale = np.log(np.maximum(dist_avg * params.init_scale, 1e-9)).astype(np.float32)
    scales = np.repeat(log_scale[:, None], 3, axis=1)
```

gsplat's `split` displaces children by `R · diag(scales) · randn(3)`, so with an equal third axis every split ejects children a full tangent radius **off the surface along the normal**. Meanwhile `grow_scale3d × scene_scale = 0.137 m` puts essentially every Gaussian below the clone/split boundary, so densification mostly **clones** — exact duplicates that, with 6 steps between refinements, never separate. Hence 8.9% of the final model is sub-2 mm, and stage 7 deleted **79%** of it while PSNR went *up* (24.43 → 25.15 dB).

**5. Depth-guided densification supplies 52% of the model.** `splats_depth_seeded: 269,918` of `523,088`. Twenty-three seeding events, each up to 20k Gaussians, every one a ~4 mm disc at opacity 0.5 whose normal faces the seeding camera and whose colour is one reference pixel:

```1265:1270:splat/stage6.py
    quats = _quat_from_normal(torch, normals).float()
    radius = (params.depth_densify_scale_px * depths / max(focal, 1e-6)).clamp_min(1e-6)
    log_r = torch.log(radius)
    scales = torch.stack([log_r, log_r, log_r + float(np.log(0.01))], dim=1).float()
    a = float(np.clip(params.depth_densify_opacity, 1e-3, 1.0 - 1e-3))
    opac = torch.full((m,), float(np.log(a / (1.0 - a))), device=device).float()
```

Camera-facing discs vanish edge-on from other views. That's per-view private geometry: it flatters training PSNR and disintegrates under free-fly. Postshot has no analogue. It also fires from step 61, when the opacity-0.1 points init still reads as a "hole" at nearly every foreground pixel.

**6. `alpha_lambda = 0.5` is a "fill the frame" prior.** I decoded the reference frames — interior views have `alpha ≡ 1.0` everywhere. So this term is a uniform push to raise coverage during exactly the window when Gaussians choose their size, at ten times the effective weight of the photometric L1.

**7. Normal consistency is applied harder and earlier than the reference.** gsplat multiplies the depth-derived normal by the rendered alpha (`normals_from_depth *= alphas.detach()`) so uncovered pixels contribute nothing; we mask on *ground-truth* alpha instead, penalizing pixels the splat doesn't cover yet. And it starts at step 125/1532 (8%) versus the reference 7000/30000 (23%).

## What I'd change, in order

1. Make `iterations` mean optimizer steps and have `batch` multiply images per step. Alternatively adopt gsplat's `adjust_steps(factor)` pattern so shortening a run scales every cadence coherently.
2. Drop `depth_lambda` to ~1e-2 in disparity space, or turn it off entirely for a Postshot-parity baseline. Same for `alpha_lambda`.
3. Decouple `prune_scale3d`/`prune_scale2d` from `reset_every` — pass them to `DefaultStrategy` on their own trigger — and re-enable the opacity reset at a cadence scaled to the run length.
4. Fix the deployment drift so `init="surfels"` actually runs, and give `_init_from_points` the same tiny third scale.
5. Disable `depth_densify` and let gradient-driven densification do the work.

The fastest A/B to confirm all of this: rerun with `train={"init":"surfels","batch":1,"iterations":30000,"epochs":null,"depth_lambda":0.0,"alpha_lambda":0.0,"depth_densify":false,"reset_every":3000}`. That is a genuine Postshot-parity configuration, and it should land in the low-30s dB rather than 24.4.

`scratch_splat_render.py` and `scratch_out/` are throwaway diagnostics — delete them whenever.