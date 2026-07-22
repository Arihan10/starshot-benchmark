"""Stage 6 — Splat fine-tune (the training run) via gsplat 2DGS.

Takes the **Stage-3 surfel cloud** (`cloud.ply`) as the initialization and
optimizes it against the **Stage-5 unlit reference renders** (`refs/` — per-view
RGB + depth + alpha + exact OpenCV poses in `transforms.json`) into an optimized
2DGS splat (`trained.ply`). This is where the raw mesh-sampled surfels become a
clean, crisp splat: densification adds Gaussians where the render disagrees with
the reference (appearance detail on geometrically-flat surfaces), pruning removes
redundant ones, and the depth loss suppresses floaters.

WHAT THE FINE-TUNE FIXES (all lighting-independent, so it runs unlit): render-
operator errors that only appear once flat disks are depth-sorted + alpha-blended
through the real rasterizer — silhouettes, thin geometry, zoom-in detail, alpha
edges — plus floaters. The surfel init is ~90% there; this polishes the rest.

CONTRACT (locked, shared with Stages 4/5 — see overview §12):
  * Init map (`cloud.ply`, a Stage-3 2DGS `.ply`, SH degree 0):
      means = xyz; quats = rot_0..3 (wxyz); opacity = logit (pre-sigmoid);
      scale_0/1 = log tangent radii (isotropic); f_dc_0..2 = SH0 colour coeffs.
      gsplat 2DGS wants 3 scales, so a third log-scale = the tangent one is
      synthesized (rendering uses only the two in-plane axes; the third only
      feeds the strategy's scene-scale-normalized grow/prune thresholds).
  * Poses: `transform_matrix` is OpenCV camera-to-world → `viewmats = inv(c2w)`.
    Intrinsics: pinhole `K = [[fl_x,0,cx],[0,fl_y,cy],[0,0,1]]`.
  * Depth: planar camera-space Z (metres), decoded from the SZF frame's log-uint16
    codes via the shared [near, far] (legacy 16-bit PNG / float32 `.npy` sets still
    read as-is). The loss compares the reference against the splat's EXPECTED depth
    (`depth_mode="expected"`, the ED channel) — the one per-pixel term that sees a
    low-opacity floater stranded in front of an opaque surface (photometric L1 and
    the alpha loss are both blind to it there). `depth_mode="median"` (the
    transmittance-0.5 crossing) is cleaner at silhouettes and never fades BLEND
    glass (α ≈ 0.065 stays below the crossing), but is blind to those floaters —
    reserve it for genuinely glass-heavy scenes.
  * Colour: sRGB albedo compared directly (no sRGB↔linear); SH degree a config
    flag (0 = unlit, the decided default), raisable later for shiny surfaces.

LOSSES (per view):
  * photometric — full-frame L1 + D-SSIM on RGB vs the unlit reference. Both
    sides are premultiplied-over-black (the reference alpha-blends over a black
    clear colour; gsplat composites with no background), so glass/MASK pixels
    compare like-with-like and background pixels directly penalize floater
    energy;
  * alpha (mask) — L1(render α, reference α): the renderer's exact coverage
    masks make empty space stay empty and glass stay see-through;
  * depth — alpha-gated L1 on the expected depth (the one term that suppresses
    low-opacity floaters in front of opaque surfaces; `depth_mode` switches it to
    median for glass scenes);
  * 2DGS regularizers — normal consistency (render normals vs normals-from-depth)
    and optional depth distortion.

RESUMABLE: every `ckpt_every` steps the full training state (params + per-param
Adam + means LR schedule + densification-strategy accumulators + step) is written
to `splat/ckpt/` (atomic, most-recent-`ckpt_keep` kept). An interrupted run resumes
from the latest checkpoint and continues to `iterations`; the checkpoints are
deleted once trained.ply is written. Pass `resume=False` to force a fresh run.

TILED TRAINING (scenes past the single-GPU VRAM wall): a seed larger than the
tile budget (default 2/3 of `cap_max`, leaving densification headroom) trains as
GROUND-PLANE TILES instead of one frozen over-budget run — the smallest (x,z)
grid whose largest expanded tile (core + margin ring) fits the budget. Each tile
trains sequentially in this process with the FULL budget to itself: views are
assigned by exact visibility (their reference-depth pixels unproject into the
tile's expanded box), every loss is masked to pixels the tile OWNS (its surface
+ true background, so boundary Gaussians never chase foreign content), and
depth-seeding is clipped to the box. Merge keeps each Gaussian iff its mean lies
in its tile's CORE cell — cores partition space exactly, so overlaps never
double-ship and there is no seam by construction. Per-tile results are cached
(`splat/ckpt/tiles/`) with a params signature: an interrupted tiled run resumes
at the first unfinished tile (and inside it, from its own checkpoint). This is
strictly MORE capacity than a monolithic run: n tiles × cap_max total, on the
same GPU. Because the references are exact synthetic renders (posed, unlit,
depth-true), the classic tiling artifacts (exposure seams, mis-assigned cameras)
don't apply.

LOD EXPORT (wide shots / progressive delivery): beside trained.ply, an octave
ladder `trained.lod1.ply`, `trained.lod2.ply`, … — each level ~4× fewer
Gaussians, built by opacity·area-weighted MOMENT MATCHING (cluster mean/
covariance → tangent frame via eigendecomposition; opacity preserves the
cluster's opacity·area within the new disk). A pulled-back camera renders the
coarse level as a prefiltered anti-aliased average instead of shimmering
sub-pixel splats, and the same 2DGS `.ply` layout means every existing viewer /
compressor reads the levels unchanged.

CUDA-ONLY: torch + gsplat compile/require CUDA, so this runs on the GPU box, NOT
Apple Silicon. Both are imported LAZILY inside the trainer so the server (which
imports this module for the route + the torch-free PLY/pose IO) stays importable
on a machine without them.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from splat.stage5 import (
    TRANSFORMS_NAME,
    decode_depth_u16,
    load_depth_png,
    load_reference_frame,
)

logging.getLogger("PIL").setLevel(logging.ERROR)

# The optimized splat written under a cell's `splat/` dir (a 2DGS `.ply`, the
# same layout Stage 3 emits so Stage 7/8 + the viewer read it identically).
TRAINED_NAME = "trained.ply"

# SH degree-0 basis constant (matches Stage 3/4): colour = 0.5 + C0·f_dc, so a
# seeded Gaussian's albedo → f_dc = (rgb − 0.5)/C0.
_SH_C0 = 0.28209479177387814

# progress(done, total, message) — called periodically during training.
ProgressCb = Callable[[int, int, str], None]

# Floor on the resolved optimizer-step count (matches the tiled per-tile floor):
# even a tiny epoch/budget still runs enough steps to seat the LR schedule.
_MIN_STEPS = 200


@dataclass(frozen=True)
class TrainParams:
    """Stage-6 knobs. Learning rates follow the gsplat/3DGS defaults; the means LR
    is scaled by the scene extent at runtime and decayed exponentially.

    SCHEDULE (resolved per run by `resolve_schedule`, against the plan's view
    count + `batch`): `iterations` is a VIEW-DRAW budget — the number of
    reference-image presentations, i.e. optimizer_steps × batch — so the actual
    step count is `budget // batch`. Raising `batch` therefore SPEEDS a run at
    constant work (fewer, fuller optimizer steps) instead of multiplying it, and
    the step-denominated cadences below (refine/densify windows, `ckpt_every`,
    the regularizer start iters, …) are written at batch-1 and divided by
    `batch` here so densification stays view-consistent across batch sizes.
    `epochs`, when set, OVERRIDES the budget as whole passes over the view set
    (`epochs × n_views`) — the scene-size-independent way to dial training
    length, since a bigger scene has proportionally more views (Stage 4 places
    cameras by surface area), and a mesh-exact init needs far fewer epochs than
    photogrammetry's ~16.

    `refine_stop_iter` defaults to None → resolved to 50% of the (post-batch)
    step count, so densification scales automatically with training length."""

    # VIEW-DRAW budget (optimizer steps × batch); `epochs` overrides it when set.
    # See the class docstring + `resolve_schedule` — `iterations // batch` is the
    # step count the loop actually runs.
    iterations: int = 30_000
    epochs: float | None = None
    sh_degree: int = 0                 # active SH bands (0 = unlit albedo, decided default)
    sh_degree_interval: int = 1000     # raise the active degree every N steps (if sh_degree > 0)

    # Loss weights.
    ssim_lambda: float = 0.2           # photometric = (1-λ)·L1 + λ·(1-SSIM)
    alpha_lambda: float = 0.5          # L1(render α, reference α)
    depth_lambda: float = 0.5          # alpha-gated depth L1 (metres)
    alpha_gate: float = 0.5            # reference α above this = opaque pixel (depth/normal masks)
    normal_lambda: float = 0.05        # 2DGS normal consistency
    dist_lambda: float = 0.0           # 2DGS depth distortion (off by default; over-flattens bounded scenes)
    normal_start_iter: int = 2000      # let geometry settle before the regularizers bite
    dist_start_iter: int = 1000

    # Learning rates (means_lr is × scene_scale at runtime).
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20.0

    # Densification (gsplat DefaultStrategy, 2DGS gradient key).
    refine: bool = True
    refine_start_iter: int = 500
    refine_stop_iter: int | None = None  # None → int(iterations * 0.5) at runtime
    refine_every: int = 100
    # 0 DISABLES periodic opacity resets (the default here): a reset clamps every
    # opacity to 2·prune_opa, but surfels no training view covers (occluded
    # regions kept at their mesh-true init) receive no gradient and would stay
    # dimmed forever. The pinned gsplat 1.5.3 never fires resets anyway (its
    # trigger `step % reset_every == 0 & step > 0` parses as `… and (0 > 0)`);
    # this makes that behaviour deliberate and upgrade-proof.
    reset_every: int = 0
    # grow_grad2d is the ABS-gradient split threshold (absgrad below): absolute
    # gradients don't cancel across a splat, so they run larger — raised ~3x from
    # the non-abs 0.0002 (gsplat's guidance). Drop it and the cloud over-densifies.
    grow_grad2d: float = 0.0006
    grow_scale3d: float = 0.01
    # Keep pruning BELOW the glass init: glass.py panes seed opacity at
    # GLASS_ALPHA = 0.065 (logit −2.67), and the old 0.05 threshold left only
    # ~0.3 logits of drift before a pane surfel was permanently pruned.
    prune_opa: float = 0.03
    # AbsGS: accumulate the ABSOLUTE per-pixel view-space position gradient for
    # the grow/split test, so a splat straddling high-frequency detail on a flat
    # surface — whose opposite-side gradients cancel under the mean criterion —
    # still densifies. Requires the raised grow_grad2d above.
    absgrad: bool = True
    cap_max: int = 3_000_000           # freeze densification past this many Gaussians (VRAM guard)
    # Final cleanup prune (once, before eval + export). Densification/pruning stop at
    # refine_stop (50% of iters), so opacity that drifts below prune_opa in the back
    # half — the low-opacity floaters stranded at silhouette/depth edges — otherwise
    # ships. This drops them, plus any Gaussian whose max tangent radius exceeds
    # prune_scale3d·scene_scale (runaway blobs; a safe no-op on well-behaved clouds,
    # whose largest surfel sits well under that). prune_scale3d=0 disables the scale
    # guard; final_prune=False disables the whole pass.
    final_prune: bool = True
    prune_scale3d: float = 0.1

    # Adaptive VRAM guard. cap_max is the hard ceiling; additionally freeze
    # densification (and pause depth-seeding) when free VRAM drops below this many
    # GB — after one empty_cache retry to release reusable cached blocks — so an
    # 8 GB card can't densify itself over the WDDM shared-memory cliff that
    # collapsed a real run to 0.05 it/s. 0 disables (rely on cap_max alone).
    # (`expandable_segments` would fight fragmentation on Linux/Modal but is a
    # no-op on Windows, so this free-margin freeze is the portable mechanism.)
    vram_min_free_gb: float = 0.8

    # Anti-aliasing — a Mip-Splatting-style 3D low-pass computed from the exact
    # cameras. Every `aa_every` steps, lower-bound each Gaussian's two in-plane
    # log-scales so it projects to ≥ `aa_min_scale_px` std from its NEAREST camera
    # (radius floor = px·d_nn/focal). A Gaussian the closest camera can't resolve
    # would alias — shimmer across the scale ladder's octaves under free-fly. The
    # floor scales with distance, so close-viewed detail and large surfaces are
    # untouched; only genuinely sub-pixel Gaussians are inflated.
    antialias: bool = True
    aa_min_scale_px: float = 1.0
    aa_every: int = 200

    # Depth-guided densification (unique to this pipeline's EXACT reference depth).
    # Every `depth_densify_every` steps, find reference pixels holding an opaque
    # surface the splat is MISSING — rendered α < `miss_alpha` (a hole) or rendered
    # depth behind the reference by > `depth_tol` (a near surface absent) — unproject
    # them to world at the reference depth, and seed Gaussians there (colour =
    # reference albedo, normal toward the camera). This fills real geometry directly
    # instead of waiting for screen-space gradients to diffuse a clone into the gap,
    # and it grows surface rather than the edge floaters densification-at-silhouettes
    # produces. Capped at `depth_densify_max` per pass, paused under VRAM pressure,
    # and only inside the densification window.
    depth_densify: bool = True
    depth_densify_every: int = 500
    depth_densify_start: int = 500
    depth_densify_max: int = 20_000
    depth_densify_miss_alpha: float = 0.5
    depth_densify_depth_tol: float = 0.1
    depth_densify_opacity: float = 0.5
    depth_densify_scale_px: float = 1.0

    # Runtime. (2DGS renders non-packed: gsplat's packed depth path is broken —
    # its SH branch mis-broadcasts and its precomputed-colour path skips the
    # visible-subset gather, so RGB+ED crashes once a view sees only part of the
    # cloud. Non-packed is also cheap at our per-cell Gaussian counts.)
    near_plane: float = 0.01
    far_plane: float = 1e10
    # Depth statistic the depth loss (and normals-from-depth) compares. "expected"
    # (default) = the alpha-weighted mean (ED): the ONLY per-pixel term that sees a
    # low-opacity floater stranded in front of an opaque surface — L1 is camouflaged
    # (the floater blends toward its backing's colour), the alpha loss is saturated
    # (coverage behind it is already 1), and the median's transmittance-0.5 crossing
    # sits on the surface BEHIND it, so only the expected depth is shifted by a front
    # floater. "median" = the transmittance-0.5 crossing: cleaner at silhouettes and
    # it never fades BLEND glass (α ≈ 0.065 stays below the crossing), but it's blind
    # to those floaters — use it only for genuinely glass-heavy scenes.
    depth_mode: str = "expected"
    seed: int = 0
    eval_max_views: int = 128          # cap final-metric renders (plans can have thousands of views)
    log_every: int = 50                # emit a progress line every N steps
    # Views per optimizer step. The schedule holds VIEW-DRAWS constant
    # (steps = iterations // batch), so this is a pure SPEED knob — it fills the
    # A100's idle SMs by amortizing per-step overhead, at constant total work.
    # Conservative default: rasterization activation memory (esp. the tile-
    # intersection buffer) scales ~linearly with batch and is scene-dependent, so
    # 3 keeps a dense/large cell well clear of the VRAM ceiling and the single-
    # step OOM the free-margin guard can't catch (it only throttles growth).
    batch: int = 3
    prefetch: bool = True              # decode/stack the next batch on a background thread (hide disk I/O)

    # Resumable checkpoints: every `ckpt_every` steps write the full training state
    # (params + per-param Adam + densification strategy + step) to `splat/ckpt/`,
    # keeping the most recent `ckpt_keep`. An interrupted run resumes from the latest
    # and continues to `iterations`; they're deleted once trained.ply is written.
    ckpt_every: int = 2000             # 0 disables checkpointing
    ckpt_keep: int = 2

    # Tiled training (module docstring §TILED TRAINING). Seeds beyond the budget
    # train as ground-plane tiles, each with the full budget of densification
    # headroom, merged by core ownership. None → 2/3 of cap_max (room to grow to
    # the cap); 0 disables tiling (a huge seed then trains frozen, as before).
    tile_max: int | None = None
    tile_margin_frac: float = 0.10       # context ring, fraction of the tile side
    tile_margin_min_m: float = 0.5       # … clamped to physical bounds (metres)
    tile_margin_max_m: float = 4.0
    tile_max_tiles: int = 64             # split-loop safety cap (VRAM guards still hold)
    tile_assign_stride: int = 8          # depth-pixel subsample for view→tile assignment
    tile_assign_min_frac: float = 0.005  # assign a view when ≥ this fraction of its fg …
    tile_assign_min_px: int = 24         # … or ≥ this many strided fg pixels land in the tile
    # LOD ladder exported beside trained.ply (module docstring §LOD EXPORT):
    # levels stop early once a level would hold ≤ `lod_min_count`. 0 disables.
    lod_levels: int = 3
    lod_min_count: int = 150_000

    @property
    def resolved_refine_stop(self) -> int:
        return self.refine_stop_iter if self.refine_stop_iter is not None else int(self.iterations * 0.5)

    @property
    def resolved_tile_budget(self) -> int:
        """Seed count above which the run tiles (0 = tiling disabled)."""
        if self.tile_max is not None:
            return max(int(self.tile_max), 0)
        return max(int(self.cap_max * 2 / 3), 1)

    def resolve_schedule(self, n_views: int) -> "TrainParams":
        """Concrete per-run (or per-TILE) schedule for `n_views` reference images
        at this `batch`, in actual optimizer-STEP units. `epochs`, when set, is
        the training length as whole passes over the view set — `iterations` is
        derived as `epochs × n_views` view-draws — else the `iterations` field is
        used directly as that budget. The step count is then `budget // batch`
        (batch is a pure speed knob at constant work), and the step-denominated
        cadences (refine/densify windows, `ckpt_every`, regularizer starts, …)
        are divided by `batch` so densification stays view-consistent. Returns a
        NEW TrainParams whose `iterations` + cadences are the step counts the
        trainer runs, so every downstream consumer (the loop, `resolved_refine_stop`,
        the LR schedule) is unchanged. Called per TILE with the tile's own view
        count, so each tile trains `epochs` passes over the views that supervise
        it."""
        from dataclasses import replace

        b = max(int(self.batch), 1)
        budget = (
            max(1, round(self.epochs * max(n_views, 1)))
            if self.epochs is not None
            else max(1, int(self.iterations))
        )
        steps = max(_MIN_STEPS, round(budget / b))

        def per_batch(v: int, floor: int = 1) -> int:
            return max(floor, round(v / b))

        return replace(
            self,
            iterations=steps,
            refine_start_iter=per_batch(self.refine_start_iter),
            refine_every=per_batch(self.refine_every),
            refine_stop_iter=(
                None if self.refine_stop_iter is None
                else per_batch(self.refine_stop_iter)
            ),
            depth_densify_every=per_batch(self.depth_densify_every),
            depth_densify_start=per_batch(self.depth_densify_start),
            normal_start_iter=per_batch(self.normal_start_iter),
            dist_start_iter=per_batch(self.dist_start_iter),
            sh_degree_interval=per_batch(self.sh_degree_interval),
            aa_every=per_batch(self.aa_every),
            ckpt_every=(0 if self.ckpt_every == 0 else per_batch(self.ckpt_every)),
        )

    def as_summary(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "epochs": self.epochs,
            "batch": self.batch,
            "sh_degree": self.sh_degree,
            "ssim_lambda": self.ssim_lambda,
            "alpha_lambda": self.alpha_lambda,
            "depth_lambda": self.depth_lambda,
            "depth_mode": self.depth_mode,
            "alpha_gate": self.alpha_gate,
            "normal_lambda": self.normal_lambda,
            "dist_lambda": self.dist_lambda,
            "refine": self.refine,
            "refine_stop_iter": self.resolved_refine_stop,
            "reset_every": self.reset_every,
            "grow_grad2d": self.grow_grad2d,
            "absgrad": self.absgrad,
            "prune_opa": self.prune_opa,
            "final_prune": self.final_prune,
            "prune_scale3d": self.prune_scale3d,
            "vram_min_free_gb": self.vram_min_free_gb,
            "antialias": self.antialias,
            "aa_min_scale_px": self.aa_min_scale_px,
            "depth_densify": self.depth_densify,
            "depth_densify_every": self.depth_densify_every,
            "depth_densify_max": self.depth_densify_max,
            "ckpt_every": self.ckpt_every,
            "tile_budget": self.resolved_tile_budget,
            "tile_margin_frac": self.tile_margin_frac,
            "lod_levels": self.lod_levels,
        }


# --- torch-free layer (PLY / pose / image IO — runs anywhere) ------------------


def _load_cloud(path: Path) -> dict[str, np.ndarray]:
    """Parse a Stage-3 binary-little-endian float `.ply` into gsplat 2DGS init
    arrays: means [N,3], quats [N,4] (wxyz), opacities [N] (logit), sh0 [N,1,3]
    (SH0 coeffs), scales [N,3] (log; a synthesized third axis when the cloud is
    the 2-scale 2DGS format)."""
    raw = Path(path).read_bytes()
    marker = b"end_header\n"
    cut = raw.find(marker)
    if cut < 0:
        raise ValueError(f"{path}: not a PLY (no end_header)")
    header = raw[:cut].decode("ascii", "replace").splitlines()
    if not header or header[0].strip() != "ply":
        raise ValueError(f"{path}: missing 'ply' magic")
    if not any(l.startswith("format") and "binary_little_endian" in l for l in header):
        raise ValueError(f"{path}: only binary_little_endian is supported")

    count: int | None = None
    props: list[str] = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            count, in_vertex = int(parts[2]), True
        elif parts[:1] == ["element"]:
            in_vertex = False
        elif in_vertex and parts[:1] == ["property"]:
            if parts[1] != "float":
                raise ValueError(f"{path}: non-float property '{parts[2]}' unsupported")
            props.append(parts[2])
    if count is None:
        raise ValueError(f"{path}: no vertex element")

    body = np.frombuffer(raw, dtype="<f4", count=count * len(props), offset=cut + len(marker))
    table = body.reshape(count, len(props))
    col = {name: table[:, i] for i, name in enumerate(props)}

    def stack(*names: str) -> np.ndarray:
        return np.stack([col[n] for n in names], axis=1).astype(np.float32)

    means = stack("x", "y", "z")
    quats = stack("rot_0", "rot_1", "rot_2", "rot_3")
    opacities = col["opacity"].astype(np.float32)
    sh0 = stack("f_dc_0", "f_dc_1", "f_dc_2").reshape(-1, 1, 3)
    if "scale_2" in col:
        scales = stack("scale_0", "scale_1", "scale_2")
    else:
        # 2DGS cloud: synthesize a TINY third log-scale. Rendering ignores it and
        # the strategy's grow/prune thresholds take max() over scales (unchanged),
        # but DefaultStrategy's split() displaces children along ALL three scaled
        # axes — a normal-axis scale equal to the tangent radius would eject every
        # child a full radius off the surface. 1% of the smaller tangent radius
        # keeps splits in-plane.
        two = stack("scale_0", "scale_1")
        third = (two.min(axis=1, keepdims=True) + np.log(0.01)).astype(np.float32)
        scales = np.concatenate([two, third], axis=1)
    return {"means": means, "quats": quats, "opacities": opacities, "sh0": sh0, "scales": scales}


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_alpha(path: Path) -> np.ndarray:
    from PIL import Image

    return (np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0)[..., None]


def _read_depth(view: dict[str, Any]) -> np.ndarray | None:
    """LEGACY reference depth for a view as [H,W,1] float32 planar-Z metres, or
    None: the log-uint16 depth PNG of pre-SZF reference sets (decoded via
    `load_depth_png` with the view's [near, far]), or the even older float32
    `.npy`. Current sets carry depth inside the SZF frame (`_view_arrays`)."""
    dp = view["depth"]
    if dp is None:
        return None
    if dp.suffix == ".png":
        near, far = view.get("depth_near"), view.get("depth_far")
        if near is None or far is None:
            raise ValueError(f"{dp}: PNG depth needs 'near'/'far' in transforms.json")
        return load_depth_png(dp, near, far)[..., None]
    return np.load(dp).astype(np.float32)[..., None]


def _view_arrays(view: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """One view's supervision as numpy: (rgb [H,W,3] in [0,1], alpha [H,W,1] in
    [0,1], depth [H,W,1] planar-Z metres | None). The current Stage-5 contract is
    ONE SZF frame per view (all three planes, one open + zstd decode — ~4x faster
    than the PNG triple it replaced); the legacy branch keeps pre-SZF reference
    sets (PNG triple, `.npy` depth) trainable without regeneration."""
    frame = view.get("frame")
    if frame is not None:
        rgba, codes = load_reference_frame(frame)
        rgb = rgba[..., :3].astype(np.float32) / 255.0
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        near, far = view.get("depth_near"), view.get("depth_far")
        if near is None or far is None:
            raise ValueError(f"{frame}: SZF depth needs 'near'/'far' in transforms.json")
        return rgb, alpha, decode_depth_u16(codes, near, far)[..., None]
    rgb = _load_rgb(view["rgb"])
    alpha = (
        _load_alpha(view["alpha"])
        if view["alpha"] is not None
        else np.ones((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)
    )
    return rgb, alpha, _read_depth(view)


def _fmt_hms(seconds: float) -> str:
    s = int(max(seconds, 0.0))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _quats_to_normals(quats: np.ndarray) -> np.ndarray:
    """Surfel normal = third column of the (wxyz) quaternion's rotation matrix —
    the axis Stage 3 aligned to the surface normal."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    n = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def _encode_trained_ply(
    means: np.ndarray,
    quats: np.ndarray,
    f_dc: np.ndarray,
    opacity: np.ndarray,
    scales2: np.ndarray,
    out_path: Path,
) -> None:
    """Write the trained model as a Stage-3-compatible 2DGS `.ply` (two tangent
    log-scales, SH degree 0). Values are stored raw (opacity as logit, scales as
    log, colour as `f_dc`) exactly as the viewers/Stage 7 expect."""
    n = means.shape[0]
    normals = _quats_to_normals(quats)
    cols = [
        means[:, 0], means[:, 1], means[:, 2],
        normals[:, 0], normals[:, 1], normals[:, 2],
        f_dc[:, 0], f_dc[:, 1], f_dc[:, 2],
        opacity,
        scales2[:, 0], scales2[:, 1],
        quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3],
    ]
    data = np.stack(cols, axis=1).astype("<f4")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n" "property float y\n" "property float z\n"
        "property float nx\n" "property float ny\n" "property float nz\n"
        "property float f_dc_0\n" "property float f_dc_1\n" "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n" "property float scale_1\n"
        "property float rot_0\n" "property float rot_1\n"
        "property float rot_2\n" "property float rot_3\n"
        "end_header\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    tmp.replace(out_path)


# --- CUDA layer (torch + gsplat imported lazily) -------------------------------


def _require_cuda_trainer():  # noqa: ANN202 - returns (torch, torch.nn.functional, gsplat)
    """Import torch + gsplat and assert a CUDA device, or raise a clear error.
    Kept out of module import so the server (Apple Silicon) can import this file."""
    try:
        import torch
        import torch.nn.functional as F
        import gsplat
    except Exception as exc:
        raise RuntimeError(
            "Stage 6 needs torch + gsplat (CUDA). These are CUDA-only and not "
            "installed here — run Stage 6 on the GPU box, not on Apple Silicon. "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 6 needs a CUDA GPU; torch reports none available.")
    return torch, F, gsplat


def _gaussian_window(torch, ksize: int, sigma: float, device, channels: int):  # noqa: ANN001
    coords = torch.arange(ksize, dtype=torch.float32, device=device) - (ksize - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    win2d = (g[:, None] @ g[None, :])[None, None]        # [1,1,k,k]
    return win2d.expand(channels, 1, ksize, ksize).contiguous()


def _ssim(F, x, y, window):  # noqa: ANN001
    """Mean SSIM of two [B,C,H,W] images in [0,1] (11×11 Gaussian window)."""
    ch = x.shape[1]
    pad = window.shape[-1] // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=ch)
    mu_y = F.conv2d(y, window, padding=pad, groups=ch)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x = F.conv2d(x * x, window, padding=pad, groups=ch) - mu_x2
    sig_y = F.conv2d(y * y, window, padding=pad, groups=ch) - mu_y2
    sig_xy = F.conv2d(x * y, window, padding=pad, groups=ch) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2)
    )
    return ssim_map.mean()


def _read_transforms(refs_dir: Path) -> dict[str, Any]:
    tp = Path(refs_dir) / TRANSFORMS_NAME
    if not tp.is_file():
        raise FileNotFoundError(f"reference renders not found: {tp} (run Stage 5)")
    return json.loads(tp.read_text(encoding="utf-8"))


def _render_inputs(torch, splats, active_sh: int):  # noqa: ANN001
    """SH coefficients [N,K,3] + active `sh_degree` for `rasterization_2dgs`.
    Always the SH path (degree 0 = just the DC term) rendered non-packed — the
    only combination that produces depth correctly here: gsplat's packed SH branch
    mis-broadcasts, and its precomputed-colour path never gathers colours to the
    visible subset, so both crash on RGB+ED once a view sees only part of the
    cloud (nnz < N). gsplat applies the +0.5 offset, matching Stage-3's f_dc."""
    sh = splats["sh0"] if "shN" not in splats else torch.cat([splats["sh0"], splats["shN"]], dim=1)
    return sh, active_sh


def _load_view(torch, view: dict[str, Any], device):  # noqa: ANN001
    """Stream one view's supervision to the GPU: rgb [1,H,W,3], alpha [1,H,W,1],
    depth [1,H,W,1] | None. Pixels are read from disk on demand so plans with
    thousands of 512² views never have to fit in host RAM."""
    rgb_np, alpha_np, d = _view_arrays(view)
    rgb = torch.from_numpy(np.ascontiguousarray(rgb_np)).to(device).unsqueeze(0)
    alpha = torch.from_numpy(np.ascontiguousarray(alpha_np)).to(device).unsqueeze(0)
    depth = None
    if d is not None:
        depth = torch.from_numpy(np.ascontiguousarray(d)).to(device).unsqueeze(0)
    return rgb, alpha, depth


def _view_stream(torch, views, device, batch: int, prefetch: bool, seed: int, start_step: int = 0, stop=None):  # noqa: ANN001
    """Infinite stream of training batches — `(viewmats [B,4,4], rgb [B,H,W,3],
    alpha [B,H,W,1], depth [B,H,W,1] | None)` on `device`, B views each.

    Views are drawn WITHOUT replacement: each epoch is a fresh permutation of all
    n views walked `batch` at a time (reshuffled on wrap), so every reference
    image contributes and exposure is uniform — the coverage Stage 4 solved for is
    actually delivered to the optimizer, instead of the ~e^(-steps·batch/n) of
    views a with-replacement draw would never touch on a large plan. The schedule
    is a pure function of the global draw counter (`start_step·batch`), each
    epoch's permutation seeded by `(seed, epoch)`, so a checkpoint resume continues
    the exact same view order rather than restarting it.

    With `prefetch`, a daemon thread does the expensive part (SZF zstd / legacy
    PNG decode + stacking) for the NEXT batch while the GPU trains on the current
    one, so the render loop never stalls on disk. The host→device copy stays
    synchronous (the decode is the real cost), which keeps it correct with no
    tensor-lifetime traps. `stop` (a threading.Event) lets a caller that abandons
    the stream mid-epoch (tiled runs train many streams per process) release the
    worker thread and its queued batches instead of leaking them."""
    n = len(views)

    def epoch_perm(e: int) -> np.ndarray:
        """View order for epoch `e`, seeded by (seed, epoch) so a resumed run
        reproduces the identical shuffle."""
        return np.random.default_rng([seed, e]).permutation(n)

    def draw_indices(start_draw: int):  # noqa: ANN202 - infinite view-index stream
        """Concatenated per-epoch permutations from global draw `start_draw` on:
        view = perm(d // n)[d % n]. Every view appears exactly once per n draws; a
        batch straddling an epoch boundary simply spans two permutations."""
        d = start_draw
        cur_e, perm = -1, None
        while True:
            e = d // n
            if e != cur_e:
                cur_e, perm = e, epoch_perm(e)
            yield int(perm[d % n])
            d += 1

    def decode_batch(idx_gen) -> tuple:  # noqa: ANN001 - CPU tensors, stacked to [B,...]
        vms, rgbs, alphas, depths = [], [], [], []
        for _ in range(batch):
            v = views[next(idx_gen)]
            rgb, alpha, d = _view_arrays(v)
            vms.append(v["viewmat"])
            rgbs.append(torch.from_numpy(np.ascontiguousarray(rgb)))
            alphas.append(torch.from_numpy(np.ascontiguousarray(alpha)))
            if d is not None:
                depths.append(torch.from_numpy(np.ascontiguousarray(d)))
        depth = torch.stack(depths) if len(depths) == batch else None
        # Page-lock the host batch so the H2D copy overlaps compute (its cost
        # grows with `batch`; pinning + non_blocking keeps it off the step's
        # critical path). Pinning runs on the prefetch thread, not the GPU one.
        pin = torch.cuda.is_available()

        def _fin(t):  # noqa: ANN001, ANN202
            return t.pin_memory() if (pin and t is not None) else t

        return _fin(torch.stack(vms)), _fin(torch.stack(rgbs)), _fin(torch.stack(alphas)), _fin(depth)

    def to_dev(b) -> tuple:  # noqa: ANN001
        vm, rgb, alpha, depth = b
        return (
            vm.to(device, non_blocking=True),
            rgb.to(device, non_blocking=True),
            alpha.to(device, non_blocking=True),
            depth.to(device, non_blocking=True) if depth is not None else None,
        )

    if not prefetch:
        idx_gen = draw_indices(start_step * batch)
        while True:
            yield to_dev(decode_batch(idx_gen))

    import queue
    import threading

    q: queue.Queue = queue.Queue(maxsize=3)

    def worker() -> None:
        idx_gen = draw_indices(start_step * batch)
        try:
            while stop is None or not stop.is_set():
                item = decode_batch(idx_gen)
                while True:  # bounded put so a set `stop` can't strand a full queue
                    try:
                        q.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        if stop is not None and stop.is_set():
                            return
        except Exception as exc:  # surface to the main thread rather than deadlock
            q.put(exc)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if isinstance(item, Exception):
            raise item
        yield to_dev(item)


CKPT_DIRNAME = "ckpt"


def _ckpt_dir(out_path: Path) -> Path:
    """Where Stage-6 resumable checkpoints live (`splat/ckpt/`, beside trained.ply)."""
    return out_path.parent / CKPT_DIRNAME


def _strat_state_to(state: dict[str, Any], mover: Callable[[Any], Any]) -> dict[str, Any]:
    """Copy a gsplat strategy-state dict, moving its tensors via `mover` (`.cpu()` for
    save, `.to(device)` for load) and passing scalars / None through untouched."""
    return {k: mover(v) if hasattr(v, "to") else v for k, v in state.items()}


def _save_checkpoint(  # noqa: ANN001
    torch, ckpt_dir, step, splats, optimizers, means_sched, strat_state, meta, keep
) -> None:
    """Atomically write a full training checkpoint (params + per-param Adam state +
    means scheduler + densification-strategy state + step), then prune to the most
    recent `keep`. Everything is stored on the CPU so it reloads on any device."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "params": {k: v.detach().cpu() for k, v in splats.items()},
        "optimizers": {n: o.state_dict() for n, o in optimizers.items()},
        "means_sched": means_sched.state_dict(),
        "strat_state": _strat_state_to(strat_state, lambda v: v.detach().cpu()),
        "meta": meta,
    }
    path = ckpt_dir / f"step_{int(step):06d}.pt"
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    if keep > 0:
        for old in sorted(ckpt_dir.glob("step_*.pt"))[:-keep]:
            old.unlink(missing_ok=True)


def _load_checkpoint(torch, ckpt_dir: Path, device):  # noqa: ANN001
    """The newest valid checkpoint in `ckpt_dir` (loaded to `device`), or None. Tries
    newest-first, skipping any that fail to load (tolerating a torn tail file)."""
    if not ckpt_dir.is_dir():
        return None
    for path in sorted(ckpt_dir.glob("step_*.pt"), reverse=True):
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except Exception:
            continue
    return None


def _quat_from_normal(torch, normals):  # noqa: ANN001 - [M,3] unit → [M,4] wxyz (+Z→normal)
    """Shortest-arc quaternions rotating +Z onto each unit normal (the axis Stage 3
    and the exporter treat as the surfel normal, so `_quats_to_normals` inverts this).
    Antipodal (+Z ≈ −normal) rotates 180° about X."""
    nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
    q = torch.stack([1.0 + nz, -ny, nx, torch.zeros_like(nz)], dim=1)
    antip = q[:, 0] < 1e-6
    if bool(antip.any()):
        flip = torch.tensor([0.0, 1.0, 0.0, 0.0], device=normals.device, dtype=q.dtype)
        q = torch.where(antip[:, None], flip, q)
    return q / (q.norm(dim=1, keepdim=True) + 1e-12)


def _append_gaussians(torch, splats, optimizers, strat_state, new):  # noqa: ANN001
    """Grow the model by the Gaussians in `new` (per-key tensors; any key a splat has
    but `new` omits — e.g. shN — is zero-filled), extending each Adam optimizer's
    moments and the strategy's running state (grad2d/count) with zeros so training
    stays consistent. Call AFTER the strategy's per-step work, so the step's `info`
    (sized to the pre-append count) is never used against the grown tensors."""
    m = int(new["means"].shape[0])
    for name in list(splats.keys()):
        p_old = splats[name]
        add = (
            new[name].to(device=p_old.device, dtype=p_old.dtype)
            if name in new
            else torch.zeros((m, *p_old.shape[1:]), device=p_old.device, dtype=p_old.dtype)
        )
        p_new = torch.nn.Parameter(
            torch.cat([p_old.detach(), add], dim=0), requires_grad=p_old.requires_grad
        )
        opt = optimizers.get(name)
        if opt is not None:
            st = opt.state.pop(p_old, None)
            if st is not None:
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in st and torch.is_tensor(st[key]):
                        z = torch.zeros((m, *st[key].shape[1:]), device=st[key].device, dtype=st[key].dtype)
                        st[key] = torch.cat([st[key], z], dim=0)
                opt.state[p_new] = st
            for g in opt.param_groups:
                if len(g["params"]) == 1 and g["params"][0] is p_old:
                    g["params"] = [p_new]
        splats[name] = p_new
    for key in ("grad2d", "count"):
        v = strat_state.get(key)
        if torch.is_tensor(v):
            strat_state[key] = torch.cat([v, torch.zeros(m, device=v.device, dtype=v.dtype)], dim=0)


def _aa_scale_floor(torch, splats, cam_tree, focal, aa_min_px):  # noqa: ANN001
    """Lower-bound each Gaussian's two in-plane log-scales so its projected std is
    ≥ `aa_min_px` from the NEAREST camera (radius floor = aa_min_px·d_nn/focal). Only
    inflates Gaussians a close camera would render sub-pixel; larger/close-viewed
    ones are untouched (the floor shrinks with camera distance). Returns the median
    floor radius in metres, for logging."""
    means_np = splats["means"].detach().cpu().numpy()
    d_nn = np.asarray(cam_tree.query(means_np, k=1, workers=-1)[0], dtype=np.float64)
    floor_r = (aa_min_px * np.maximum(d_nn, 1e-6) / max(float(focal), 1e-6)).astype(np.float32)
    log_floor = torch.from_numpy(np.log(np.maximum(floor_r, 1e-9))).to(splats["scales"].device)
    with torch.no_grad():
        s = splats["scales"]
        s[:, 0] = torch.maximum(s[:, 0], log_floor)
        s[:, 1] = torch.maximum(s[:, 1], log_floor)
    return float(np.median(floor_r))


def _depth_seed(torch, gt_rgb, gt_alpha, gt_depth, pred_alpha, pred_depth, viewmats, K, params, max_new, box=None):  # noqa: ANN001
    """Seed Gaussians at reference-depth surfaces the splat is missing. Per batched
    view: pixels with a reference opaque surface (gt_alpha > alpha_gate, gt_depth > 0)
    that the render doesn't cover (pred_alpha < miss_alpha) or resolves well BEHIND
    (pred_depth > gt_depth·(1+tol)) are unprojected to world at the reference depth
    (OpenCV: x=(px−cx)/fx·d, y=(py−cy)/fy·d, z=d; world = c2w·p_cam) and returned as
    new-Gaussian tensors (colour = reference albedo, normal toward the camera, scale
    from the pixel footprint). None when nothing is deficient / no budget. `box`
    ((lo, hi) device tensors) clips seeds to a tile's expanded region — a tiled run
    must not grow Gaussians for surface a neighbouring tile owns."""
    if gt_depth is None or max_new <= 0:
        return None
    device = gt_rgb.device
    B = gt_rgb.shape[0]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    focal = float(K[0, 0])
    cap_b = max(1, max_new // B)
    pos_l, col_l, nrm_l, dep_l = [], [], [], []
    for b in range(B):
        surface = (gt_alpha[b, ..., 0] > params.alpha_gate) & (gt_depth[b, ..., 0] > 0)
        hole = pred_alpha[b, ..., 0] < params.depth_densify_miss_alpha
        behind = (~hole) & (pred_depth[b, ..., 0] > gt_depth[b, ..., 0] * (1.0 + params.depth_densify_depth_tol))
        idx = (surface & (hole | behind)).nonzero(as_tuple=False)
        if idx.shape[0] == 0:
            continue
        if idx.shape[0] > cap_b:
            idx = idx[torch.randperm(idx.shape[0], device=device)[:cap_b]]
        row, col = idx[:, 0], idx[:, 1]
        d = gt_depth[b, row, col, 0]
        x = (col.to(d.dtype) + 0.5 - cx) / fx * d
        y = (row.to(d.dtype) + 0.5 - cy) / fy * d
        p_cam = torch.stack([x, y, d], dim=1)                       # [K,3] OpenCV camera frame
        c2w = torch.linalg.inv(viewmats[b])
        world = p_cam @ c2w[:3, :3].transpose(0, 1) + c2w[:3, 3]    # [K,3]
        if box is not None:
            inb = ((world >= box[0]) & (world <= box[1])).all(dim=1)
            if not bool(inb.any()):
                continue
            row, col, d, world = row[inb], col[inb], d[inb], world[inb]
        normal = c2w[:3, 3][None, :] - world                        # face the camera that saw it
        normal = normal / (normal.norm(dim=1, keepdim=True) + 1e-12)
        pos_l.append(world)
        col_l.append(gt_rgb[b, row, col, :])
        nrm_l.append(normal)
        dep_l.append(d)
    if not pos_l:
        return None
    means = torch.cat(pos_l, 0).float()
    color = torch.cat(col_l, 0).clamp(0.0, 1.0)
    normals = torch.cat(nrm_l, 0)
    depths = torch.cat(dep_l, 0)
    m = means.shape[0]
    sh0 = ((color - 0.5) / _SH_C0)[:, None, :].float()
    quats = _quat_from_normal(torch, normals).float()
    radius = (params.depth_densify_scale_px * depths / max(focal, 1e-6)).clamp_min(1e-6)
    log_r = torch.log(radius)
    scales = torch.stack([log_r, log_r, log_r + float(np.log(0.01))], dim=1).float()
    a = float(np.clip(params.depth_densify_opacity, 1e-3, 1.0 - 1e-3))
    opac = torch.full((m,), float(np.log(a / (1.0 - a))), device=device).float()
    return {"means": means, "scales": scales, "quats": quats, "opacities": opac, "sh0": sh0}


# --- tiled training (scenes past the single-GPU VRAM wall) + LOD export --------


@dataclass(frozen=True)
class _TileGrid:
    """Ground-plane (x,z) tiling. CORE cells partition space exactly (clamped
    floor-division ownership, so boundary/fly-away points always belong to
    exactly one cell); each tile TRAINS its core plus a `margin` context ring,
    and the merge keeps core-owned Gaussians only — overlaps never double-ship."""

    lo: tuple[float, float]      # grid origin (min x, min z)
    step: tuple[float, float]    # cell size per axis (m)
    k: tuple[int, int]           # cells per axis
    margin: float                # trained-context ring (m)
    y: tuple[float, float]       # vertical bounds shared by every expanded box

    @property
    def n_tiles(self) -> int:
        return self.k[0] * self.k[1]

    def owner(self, means: np.ndarray) -> np.ndarray:
        """Flat core-cell index per point (clamped at the rim)."""
        ix = np.clip(((means[:, 0] - self.lo[0]) / self.step[0]).astype(np.int64), 0, self.k[0] - 1)
        iz = np.clip(((means[:, 2] - self.lo[1]) / self.step[1]).astype(np.int64), 0, self.k[1] - 1)
        return ix * self.k[1] + iz

    def expanded_box(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """World AABB a tile trains (core + margin ring, full height)."""
        ix, iz = divmod(t, self.k[1])
        lo = np.array(
            [self.lo[0] + ix * self.step[0] - self.margin, self.y[0],
             self.lo[1] + iz * self.step[1] - self.margin],
            dtype=np.float64,
        )
        hi = np.array(
            [self.lo[0] + (ix + 1) * self.step[0] + self.margin, self.y[1],
             self.lo[1] + (iz + 1) * self.step[1] + self.margin],
            dtype=np.float64,
        )
        return lo, hi

    def signature(self, n_init: int, params: TrainParams) -> str:
        """Cache key for per-tile results: any change to the cloud, the grid, or
        the training length invalidates cached tiles."""
        return (
            f"{n_init}:{self.k[0]}x{self.k[1]}:{self.margin:.3f}"
            f":{params.iterations}:{params.sh_degree}:{params.resolved_tile_budget}"
        )


def _plan_tiles(
    means: np.ndarray,
    budget: int,
    margin_frac: float,
    margin_min: float,
    margin_max: float,
    max_tiles: int,
) -> _TileGrid:
    """Smallest ground-plane grid whose largest EXPANDED tile (core + margin ring)
    holds ≤ `budget` seeds: start 1×1 and repeatedly split the axis with the longer
    cell side. Counting the seeds themselves means density skew (a packed corner)
    splits further while empty water/air adds no tiles. The margin rides the cell
    size (clamped to physical bounds and < half a cell, so a point reaches at most
    the adjacent cell's ring); `max_tiles` caps the loop — the VRAM guards still
    bound whatever the worst tile does."""
    x = means[:, 0].astype(np.float64)
    z = means[:, 2].astype(np.float64)
    lo = (float(x.min()), float(z.min()))
    ext = (max(float(x.max()) - lo[0], 1e-6), max(float(z.max()) - lo[1], 1e-6))
    y = (float(means[:, 1].min()) - margin_max, float(means[:, 1].max()) + margin_max)

    kx = kz = 1
    while True:
        step = (ext[0] / kx, ext[1] / kz)
        margin = float(np.clip(margin_frac * min(step), margin_min, min(margin_max, 0.45 * min(step))))
        ix = np.clip((x - lo[0]) / step[0], 0, kx - 1e-9).astype(np.int64)
        iz = np.clip((z - lo[1]) / step[1], 0, kz - 1e-9).astype(np.int64)
        counts = np.zeros(kx * kz, dtype=np.int64)
        # A point loads its own cell plus any neighbour whose margin band it sits
        # in (margin < step/2 → only the 8 adjacent cells can qualify).
        for ox in (-1, 0, 1):
            for oz in (-1, 0, 1):
                jx, jz = ix + ox, iz + oz
                ok = (jx >= 0) & (jx < kx) & (jz >= 0) & (jz < kz)
                if ox != 0:
                    edge = lo[0] + (jx + (1 if ox < 0 else 0)) * step[0]
                    ok &= np.abs(x - edge) <= margin
                if oz != 0:
                    edge = lo[1] + (jz + (1 if oz < 0 else 0)) * step[1]
                    ok &= np.abs(z - edge) <= margin
                if ok.any():
                    counts += np.bincount(jx[ok] * kz + jz[ok], minlength=kx * kz)
        if int(counts.max()) <= budget or kx * kz >= max_tiles:
            if int(counts.max()) > budget:
                logging.getLogger(__name__).warning(
                    "stage6: tile split capped at %d tiles with %d seeds in the worst "
                    "tile (budget %d) — the VRAM guards will bound that tile's growth",
                    kx * kz, int(counts.max()), budget,
                )
            return _TileGrid(lo, step, (kx, kz), margin, y)
        if step[0] >= step[1]:
            kx += 1
        else:
            kz += 1


def _frustum_sees(c2w: np.ndarray, K_np: np.ndarray, width: int, height: int, box) -> bool:  # noqa: ANN001
    """Fallback view→tile test for reference sets WITHOUT depth (legacy): does the
    camera plausibly see the box? True when the camera sits inside it or any box
    corner projects into the (padded) image in front of the camera. Coarser than
    the depth-exact assignment — it can only over-assign, never starve a tile."""
    lo, hi = box
    c = c2w[:3, 3]
    if bool(np.all(c >= lo) and np.all(c <= hi)):
        return True
    corners = np.array([[cx, cy, cz] for cx in (lo[0], hi[0]) for cy in (lo[1], hi[1]) for cz in (lo[2], hi[2])])
    p_cam = (corners - c) @ c2w[:3, :3]  # R.T rows applied — camera-frame points
    zed = p_cam[:, 2]
    front = zed > 1e-6
    if not front.any():
        return False
    u = K_np[0, 0] * p_cam[front, 0] / zed[front] + K_np[0, 2]
    v = K_np[1, 1] * p_cam[front, 1] / zed[front] + K_np[1, 2]
    pad_w, pad_h = 0.2 * width, 0.2 * height
    return bool(((u >= -pad_w) & (u <= width + pad_w) & (v >= -pad_h) & (v <= height + pad_h)).any())


def _assign_views(  # noqa: ANN001
    views,
    grid: _TileGrid,
    K_np: np.ndarray,
    width: int,
    height: int,
    stride: int,
    min_frac: float,
    min_px: int,
    progress: ProgressCb | None,
) -> list[list[int]]:
    """Which reference views supervise each tile: a view is assigned wherever its
    (strided) foreground depth pixels unproject into the tile's EXPANDED box — the
    same exact-ownership signal the loss mask uses, so every assigned view has real
    work and no tile trains against cameras that can't see it. Views without depth
    fall back to the frustum test; pure-background views supervise nothing."""
    fx, fy, cx, cy = K_np[0, 0], K_np[1, 1], K_np[0, 2], K_np[1, 2]
    boxes = [grid.expanded_box(t) for t in range(grid.n_tiles)]
    lo_all = np.stack([b[0] for b in boxes])  # [T,3]
    hi_all = np.stack([b[1] for b in boxes])
    assigned: list[list[int]] = [[] for _ in range(grid.n_tiles)]
    n = len(views)
    for i, v in enumerate(views):
        if progress is not None and i % 250 == 0:
            progress(i, n, f"assigning views to tiles ({i}/{n})")
        try:
            _rgb, alpha, depth = _view_arrays(v)
        except Exception:
            alpha, depth = None, None
        if depth is None or alpha is None:
            for t in range(grid.n_tiles):
                if _frustum_sees(v["c2w"], K_np, width, height, boxes[t]):
                    assigned[t].append(i)
            continue
        a = alpha[::stride, ::stride, 0]
        d = depth[::stride, ::stride, 0]
        fg = (a > 0.01) & (d > 0)
        total = int(fg.sum())
        if total == 0:
            continue
        rr, cc = np.nonzero(fg)
        dz = d[fg].astype(np.float64)
        px = (cc * stride + 0.5 - cx) / fx * dz
        py = (rr * stride + 0.5 - cy) / fy * dz
        c2w = v["c2w"]
        world = np.stack([px, py, dz], axis=1) @ c2w[:3, :3].T + c2w[:3, 3]
        inb = (world[None, :, :] >= lo_all[:, None, :]).all(-1)
        inb &= (world[None, :, :] <= hi_all[:, None, :]).all(-1)  # [T,P]
        need = max(min_px, int(np.ceil(min_frac * total)))
        for t in np.nonzero(inb.sum(axis=1) >= need)[0]:
            assigned[int(t)].append(i)
    return assigned


def _unproject_depth(torch, gt_depth, viewmats, K, px, py):  # noqa: ANN001
    """Reference depth → world points [B,H,W,3] (OpenCV pinhole; `px`/`py` are the
    precomputed pixel-centre grids). Zero-depth (background) pixels land at the
    camera centre — callers gate on depth > 0."""
    d = gt_depth[..., 0]
    x = (px[None] - K[0, 2]) / K[0, 0] * d
    y = (py[None] - K[1, 2]) / K[1, 1] * d
    p_cam = torch.stack([x, y, d], dim=-1)
    c2w = torch.linalg.inv(viewmats)
    return torch.einsum("bhwc,brc->bhwr", p_cam, c2w[:, :3, :3]) + c2w[:, None, None, :3, 3]


def _tile_mask(torch, lo, hi, gt_alpha, gt_depth, world):  # noqa: ANN001
    """Per-pixel supervision mask [B,H,W,1] for one tile: TRUE-BACKGROUND pixels
    (empty along the whole ray — every tile must keep its airspace clear) plus
    pixels whose reference surface lies INSIDE the tile's expanded box (content
    this tile owns). Foreground owned by other tiles is excluded: supervising it
    would drag boundary Gaussians toward content this tile cannot represent (the
    classic tiling haze)."""
    bg = gt_alpha[..., 0] <= 0.005
    inbox = (world >= lo).all(-1) & (world <= hi).all(-1)
    owned = (gt_depth[..., 0] > 0.0) & inbox
    return (bg | owned).unsqueeze(-1).float()


def _train_tiled(  # noqa: ANN001
    torch,
    F,
    views,
    K,
    width: int,
    height: int,
    init: dict[str, np.ndarray],
    grid: _TileGrid,
    scene_scale: float,
    params: TrainParams,
    resume: bool,
    ckpt_root: Path,
    progress: ProgressCb | None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Train the scene one tile at a time — each within the VRAM budget, each with
    the FULL densification headroom a monolithic run would have had to share — and
    merge by core ownership. Per-tile results are cached (`ckpt/tiles/tile_NNN.npz`)
    under a params signature, so an interrupted run resumes at the first unfinished
    tile; a signature mismatch discards the stale cache. Tiles whose seeds no view
    supervises pass their init through untouched (exactly what an unsupervised
    region does in a single run — no gradient ever reaches it).

    Per-tile ITERATIONS scale with the tile's supervision share (2× its fraction
    of the plan's views, floored at 20%, capped at 100% of `params.iterations`):
    a tile that sees a twentieth of the views converges in far fewer steps than
    the whole scene needs, so total wall clock stays a few × a single run rather
    than n_tiles ×, while every tile keeps full densification headroom. All
    step-derived schedules (refine window, LR decay, depth-seed window) ride the
    scaled count automatically."""
    import gc
    import shutil

    K_np = K.detach().cpu().numpy().astype(np.float64)
    owner = grid.owner(init["means"])
    core_counts = np.bincount(owner, minlength=grid.n_tiles)
    live = [t for t in range(grid.n_tiles) if core_counts[t] > 0]
    n_init = int(init["means"].shape[0])
    bands = (params.sh_degree + 1) ** 2

    assigned = _assign_views(
        views, grid, K_np, width, height, params.tile_assign_stride,
        params.tile_assign_min_frac, params.tile_assign_min_px, progress,
    )

    tiles_dir = ckpt_root / "tiles"
    manifest_path = tiles_dir / "tiles.json"
    sig = grid.signature(n_init, params)
    manifest: dict[str, Any] = {"signature": sig, "done": {}}
    if not resume:
        shutil.rmtree(ckpt_root, ignore_errors=True)
    elif manifest_path.is_file():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old.get("signature") == sig:
                manifest = old
            else:
                shutil.rmtree(ckpt_root, ignore_errors=True)
        except Exception:
            shutil.rmtree(ckpt_root, ignore_errors=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    def save_manifest() -> None:
        tmp = manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        tmp.replace(manifest_path)

    # Per-tile SCHEDULE: resolve each tile against ITS OWN assigned-view count
    # (TrainParams.resolve_schedule), so a tile's step cap / densify window /
    # convergence bounds scale to the work it actually has — a sparse tile trains
    # (and, under convergence, plateaus) in far fewer steps than a dense one,
    # while every tile keeps full densification headroom. Empty tiles cost 0.
    tile_params = {
        t: params.resolve_schedule(len(assigned[t])) for t in live if assigned[t]
    }
    iters = [tile_params[t].iterations if assigned[t] else 0 for t in live]
    offsets = np.concatenate([[0], np.cumsum(iters)])
    grand_total = max(int(offsets[-1]), 1)

    parts: list[dict[str, np.ndarray]] = []
    infos: list[dict[str, Any]] = []
    for pos, t in enumerate(live):
        tag = f"tile {pos + 1}/{len(live)}"
        npz_path = tiles_dir / f"tile_{t:03d}.npz"
        cached = manifest["done"].get(str(t))
        if cached is not None and npz_path.is_file():
            with np.load(npz_path) as zf:
                parts.append({k: zf[k] for k in zf.files})
            infos.append(cached)
            if progress is not None:
                progress(int(offsets[pos + 1]), grand_total,
                         f"[{tag}] cached ({cached['final']} splats)")
            continue

        lo, hi = grid.expanded_box(t)
        lo32, hi32 = lo.astype(np.float32), hi.astype(np.float32)
        sel = ((init["means"] >= lo32).all(axis=1)) & ((init["means"] <= hi32).all(axis=1))
        sub = {k: v[sel] for k, v in init.items()}
        vidx = assigned[t]

        if not vidx:
            core = grid.owner(sub["means"]) == t
            arrays = {k: v[core] for k, v in sub.items()}
            if bands > 1:
                arrays["shN"] = np.zeros((int(core.sum()), bands - 1, 3), dtype=np.float32)
            info: dict[str, Any] = {
                "tile": t, "init": int(sel.sum()), "final": int(core.sum()),
                "seeded": 0, "pruned": 0, "views": 0, "iters": 0,
                "skipped": "no views see this tile",
            }
            logging.getLogger(__name__).warning(
                "stage6: %s has %d seeds but no assigned views — passing its init through",
                tag, int(sel.sum()),
            )
        else:
            torch.manual_seed(params.seed + t)
            t_params = tile_params[t]
            base = int(offsets[pos])

            def tile_progress(done_s: int, total_s: int, msg: str, _base=base, _tag=tag) -> None:
                if progress is not None:
                    progress(_base + done_s, grand_total, f"[{_tag}] {msg}")

            arrays, tinfo = _train_one(
                torch, F, [views[i] for i in vidx], K, width, height, sub,
                scene_scale, np.stack([views[i]["c2w"][:3, 3] for i in vidx]),
                t_params, resume, ckpt_root / f"tile_{t:03d}",
                tile_progress if progress is not None else None,
                tile_box=(lo, hi),
            )
            core = grid.owner(arrays["means"]) == t
            arrays = {k: v[core] for k, v in arrays.items()}
            info = {"tile": t, **tinfo, "final": int(arrays["means"].shape[0]), "iters": iters[pos]}
            shutil.rmtree(ckpt_root / f"tile_{t:03d}", ignore_errors=True)
            torch.cuda.empty_cache()
            gc.collect()

        tmp = tiles_dir / f".tile_{t:03d}.tmp.npz"
        np.savez(tmp, **arrays)
        tmp.replace(npz_path)
        parts.append(arrays)
        infos.append(info)
        manifest["done"][str(t)] = info
        save_manifest()

    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    return merged, infos


def _lod_aggregate(arrays: dict[str, np.ndarray], voxel: float) -> dict[str, np.ndarray]:
    """One LOD octave: Gaussians sharing a `voxel` collapse to ONE whose first two
    moments match the cluster's opacity·area-weighted mixture — mean and covariance
    (each disk's R·diag(s²)·Rᵀ plus the spread of the means) — with the tangent
    frame recovered by eigendecomposition. Colour is the same weighted mean, and
    opacity preserves the cluster's opacity·area within the new disk (a fully
    covered patch stays opaque; sparse glass stays translucent). Sub-pixel detail
    thereby collapses to its prefiltered average — the same reason mip maps beat
    point-sampling minified textures."""
    means = arrays["means"].astype(np.float64)
    q = arrays["quats"].astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    s = np.exp(arrays["scales"].astype(np.float64)[:, :2])
    alpha = 1.0 / (1.0 + np.exp(-arrays["opacities"].astype(np.float64)))
    col = arrays["sh0"].reshape(-1, 3).astype(np.float64)

    ids = np.floor((means - means.min(axis=0, keepdims=True)) / voxel).astype(np.int64)
    _, inv = np.unique(ids, axis=0, return_inverse=True)
    m = int(inv.max()) + 1

    w = np.maximum(alpha * s[:, 0] * s[:, 1], 1e-12)  # opacity·area
    wsum = np.bincount(inv, weights=w, minlength=m)

    def wmean(v: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.bincount(inv, weights=w * v[:, j], minlength=m) for j in range(v.shape[1])],
            axis=1,
        ) / wsum[:, None]

    mu = wmean(means)
    color = wmean(col)

    # Per-Gaussian covariance R·diag(s1²,s2²,0)·Rᵀ from the (wxyz) quats.
    ww, xx, yy, zz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (yy * yy + zz * zz)
    R[:, 0, 1] = 2 * (xx * yy - ww * zz)
    R[:, 0, 2] = 2 * (xx * zz + ww * yy)
    R[:, 1, 0] = 2 * (xx * yy + ww * zz)
    R[:, 1, 1] = 1 - 2 * (xx * xx + zz * zz)
    R[:, 1, 2] = 2 * (yy * zz - ww * xx)
    R[:, 2, 0] = 2 * (xx * zz - ww * yy)
    R[:, 2, 1] = 2 * (yy * zz + ww * xx)
    R[:, 2, 2] = 1 - 2 * (xx * xx + yy * yy)
    lam = np.zeros((len(q), 3))
    lam[:, 0] = s[:, 0] ** 2
    lam[:, 1] = s[:, 1] ** 2
    cov = np.einsum("nij,nj,nkj->nik", R, lam, R)

    # Cluster covariance = E[C + μμᵀ] − μ̄μ̄ᵀ (moment matching), accumulated per
    # unique symmetric entry.
    second = cov + means[:, :, None] * means[:, None, :]
    ccov = np.zeros((m, 3, 3))
    for a_i in range(3):
        for b_i in range(a_i, 3):
            v = np.bincount(inv, weights=w * second[:, a_i, b_i], minlength=m) / wsum
            ccov[:, a_i, b_i] = v
            ccov[:, b_i, a_i] = v
    ccov -= mu[:, :, None] * mu[:, None, :]
    ccov += np.eye(3)[None] * 1e-12

    evals, evecs = np.linalg.eigh(ccov)  # ascending
    r1 = np.sqrt(np.maximum(evals[:, 2], 1e-12))
    r2 = np.sqrt(np.maximum(evals[:, 1], 1e-12))
    frame = np.stack([evecs[:, :, 2], evecs[:, :, 1], evecs[:, :, 0]], axis=2)
    neg = np.linalg.det(frame) < 0
    frame[neg, :, 1] *= -1.0
    from scipy.spatial.transform import Rotation

    qn = Rotation.from_matrix(frame).as_quat()  # xyzw
    quats = np.concatenate([qn[:, 3:4], qn[:, :3]], axis=1)

    alpha_new = np.clip(wsum / np.maximum(r1 * r2, 1e-12), 0.02, 0.995)
    logit = np.log(alpha_new / (1.0 - alpha_new))
    log_r = np.log(np.stack([r1, r2], axis=1))
    third = log_r.min(axis=1, keepdims=True) + np.log(0.01)
    return {
        "means": mu.astype(np.float32),
        "quats": quats.astype(np.float32),
        "opacities": logit.astype(np.float32),
        "sh0": color.astype(np.float32).reshape(-1, 1, 3),
        "scales": np.concatenate([log_r, third], axis=1).astype(np.float32),
    }


def _export_lod(  # noqa: ANN001
    arrays: dict[str, np.ndarray],
    out_path: Path,
    params: TrainParams,
    progress: ProgressCb | None,
) -> list[dict[str, Any]] | None:
    """Write the LOD ladder beside the trained splat (`trained.lod1.ply`, …): each
    level ~4× fewer via `_lod_aggregate` at a doubling voxel, stopping once a level
    reaches `lod_min_count`. Same 2DGS layout as trained.ply, so every existing
    viewer/compressor reads the levels unchanged (SH0 — the ladder is unlit like
    the base). Stale lod files from earlier runs are removed first."""
    for old in out_path.parent.glob(f"{out_path.stem}.lod*.ply"):
        old.unlink(missing_ok=True)
    if params.lod_levels <= 0:
        return None
    out: list[dict[str, Any]] = []
    cur = arrays
    # 2× the median disk radius ≈ one 2×2-neighbour cluster per voxel on a
    # surface — the ~4×-per-octave reduction the ladder documents.
    base_voxel = 2.0 * float(np.median(np.exp(arrays["scales"][:, :2]).max(axis=1)))
    for level in range(1, params.lod_levels + 1):
        if int(cur["means"].shape[0]) <= params.lod_min_count:
            break
        if progress is not None:
            progress(1, 1, f"LOD {level}: aggregating {int(cur['means'].shape[0])} splats")
        cur = _lod_aggregate(cur, base_voxel * (2.0 ** (level - 1)))
        path = out_path.with_name(f"{out_path.stem}.lod{level}.ply")
        quats = cur["quats"] / (np.linalg.norm(cur["quats"], axis=1, keepdims=True) + 1e-12)
        _encode_trained_ply(
            cur["means"], quats.astype(np.float32), cur["sh0"].reshape(-1, 3),
            cur["opacities"], cur["scales"][:, :2], path,
        )
        out.append({
            "level": level, "splats": int(cur["means"].shape[0]),
            "bytes": path.stat().st_size, "path": path.name,
        })
    return out or None


def train_splat(
    *,
    run: str,
    slot: str,
    model: str,
    cloud_path: Path,
    refs_dir: Path,
    out_path: Path,
    params: TrainParams = TrainParams(),
    resume: bool = True,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Fine-tune the Stage-3 surfel cloud at `cloud_path` against the Stage-5
    references in `refs_dir`, writing the optimized 2DGS splat to `out_path`
    (plus the `trained.lodK.ply` ladder). Requires a CUDA GPU + gsplat (raises a
    clear error otherwise). Returns a compact summary (init/final splat counts,
    per-tile breakdown when tiled, final metrics, bytes).

    Seeds larger than `params.resolved_tile_budget` train TILED (module docstring
    §TILED TRAINING): ground-plane cells trained one at a time — each fits VRAM
    with densification headroom a monolithic run wouldn't have — then merged by
    core ownership. Smaller seeds keep the single-run path unchanged.

    With `resume` (default), continues from the latest `splat/ckpt/` checkpoint
    when present and compatible (same SH config); tiled runs additionally resume
    at the first unfinished tile (`splat/ckpt/tiles/`). `resume=False` starts
    fresh."""
    torch, F, _ = _require_cuda_trainer()

    cloud_path, refs_dir, out_path = Path(cloud_path), Path(refs_dir), Path(out_path)
    if not cloud_path.is_file():
        raise FileNotFoundError(f"surfel cloud not found: {cloud_path} (run Stage 3)")

    doc = _read_transforms(refs_dir)
    frames = doc.get("frames", [])
    if not frames:
        raise RuntimeError(f"{refs_dir/TRANSFORMS_NAME} has no frames (run Stage 5)")

    device = torch.device("cuda")
    torch.manual_seed(params.seed)

    width, height = int(doc["w"]), int(doc["h"])
    K = torch.tensor(
        [[doc["fl_x"], 0.0, doc["cx"]], [0.0, doc["fl_y"], doc["cy"]], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )

    # Index the supervision set (paths + poses only; pixels stream from disk per
    # step). Current sets carry ONE SZF frame per view (`frame_path`); legacy
    # PNG-triple sets (`file_path`/`alpha_path`/`depth_path`) stay trainable.
    # `near`/`far` (shared across the plan) decode the log-uint16 depth codes
    # back to metric metres. `c2w` is kept for tile view-assignment.
    depth_near = float(doc["near"]) if "near" in doc else None
    depth_far = float(doc["far"]) if "far" in doc else None
    views: list[dict[str, Any]] = []
    cam_centers: list[np.ndarray] = []
    for i, fr in enumerate(frames):
        c2w = np.asarray(fr["transform_matrix"], dtype=np.float64)
        cam_centers.append(c2w[:3, 3])
        frame_rel = fr.get("frame_path")
        rgb_rel = fr.get("file_path")
        if frame_rel is None and rgb_rel is None:
            raise ValueError(
                f"{refs_dir / TRANSFORMS_NAME}: frame {i} has neither 'frame_path' "
                "(SZF) nor 'file_path' (legacy PNG) — re-run Stage 5"
            )
        views.append(
            {
                "viewmat": torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)),
                "c2w": c2w,
                "frame": refs_dir / frame_rel if frame_rel else None,
                "rgb": refs_dir / rgb_rel if rgb_rel else None,
                "alpha": refs_dir / fr["alpha_path"] if fr.get("alpha_path") else None,
                "depth": refs_dir / fr["depth_path"] if fr.get("depth_path") else None,
                "depth_near": depth_near,
                "depth_far": depth_far,
            }
        )
    n_views = len(views)

    # NOTE: the training SCHEDULE (epochs/convergence bounds → concrete step
    # counts, cadences scaled by batch) is resolved per RUN just before
    # _train_one — against the full view count for a single run, and per TILE
    # (each against its own view count) inside _train_tiled — so `params` here
    # stays the raw client-supplied policy.

    # Surfel init (also the scene-scale fallback source when there's ≤1 camera).
    init = _load_cloud(cloud_path)
    n_init = int(init["means"].shape[0])

    # Scene scale = camera-cloud radius (× 1.1), the 3DGS spatial LR / density
    # unit — GLOBAL even when tiled, so every tile's thresholds match the
    # single-run semantics.
    centers = np.stack(cam_centers, axis=0)
    scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) if n_views > 1 else 0.0
    if scene_scale <= 1e-6:
        scene_scale = float(np.linalg.norm(init["means"].max(0) - init["means"].min(0))) * 0.5
    scene_scale = max(scene_scale * 1.1, 1e-3)

    # Single run when the seed fits the budget; otherwise plan the tile grid.
    ckpt_root = _ckpt_dir(out_path)
    budget = params.resolved_tile_budget
    grid = None
    if budget > 0 and n_init > budget:
        grid = _plan_tiles(
            init["means"], budget, params.tile_margin_frac,
            params.tile_margin_min_m, params.tile_margin_max_m, params.tile_max_tiles,
        )
        if grid.n_tiles <= 1:
            grid = None

    t_start = time.perf_counter()
    if grid is None:
        arrays, one = _train_one(
            torch, F, views, K, width, height, init, scene_scale, centers,
            params.resolve_schedule(n_views), resume, ckpt_root, progress, tile_box=None,
        )
        tiles_summary = None
        n_seeded, n_pruned = one["seeded"], one["pruned"]
    else:
        if progress is not None:
            progress(
                0, grid.n_tiles * params.iterations,
                f"tiling: {n_init} seeds > budget {budget} -> {grid.k[0]}x{grid.k[1]} "
                f"grid, margin {grid.margin:.2f}m",
            )
        arrays, tiles_summary = _train_tiled(
            torch, F, views, K, width, height, init, grid, scene_scale,
            params, resume, ckpt_root, progress,
        )
        n_seeded = int(sum(t["seeded"] for t in tiles_summary))
        n_pruned = int(sum(t["pruned"] for t in tiles_summary))

    n_final = int(arrays["means"].shape[0])

    # Final metrics over a capped subset of ALL views against the merged model
    # (OOM-guarded: a merged cloud too big to render skips metrics, not the run),
    # then export the splat + its LOD ladder.
    if progress is not None:
        progress(
            1, 1,
            f"training done in {_fmt_hms(time.perf_counter() - t_start)} - "
            f"evaluating {min(n_views, params.eval_max_views)} views + writing {out_path.name}",
        )
    metrics = None
    try:
        from gsplat import rasterization_2dgs

        splats_t = {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}
        metrics = _evaluate(
            torch, rasterization_2dgs, splats_t, views, K, width, height, params, device
        )
        del splats_t
    except (torch.cuda.OutOfMemoryError, MemoryError) as exc:
        # Eval is best-effort: a CUDA OOM (merged model too big to render) OR a HOST
        # OOM (reference-frame zstd decode on a RAM-starved box) must not lose the
        # trained splat — skip metrics and let the export below run regardless.
        torch.cuda.empty_cache()
        if progress is not None:
            progress(1, 1, f"eval skipped ({type(exc).__name__}: {n_final} splats) - exporting splat anyway")

    quats = arrays["quats"] / (np.linalg.norm(arrays["quats"], axis=1, keepdims=True) + 1e-12)
    _encode_trained_ply(
        arrays["means"], quats.astype(np.float32), arrays["sh0"].reshape(-1, 3),
        arrays["opacities"], arrays["scales"][:, :2], out_path,
    )
    lod_summary = _export_lod(arrays, out_path, params, progress)

    # Training finished + the final splat is on disk → the checkpoints (and any
    # tile caches under them) are obsolete; drop them so a later resume doesn't
    # re-enter a completed run.
    import shutil

    shutil.rmtree(ckpt_root, ignore_errors=True)

    return {
        "run": run,
        "slot": slot,
        "model": model,
        "splats_init": n_init,
        "splats_final": n_final,
        "splats_pruned_final": n_pruned,
        "splats_depth_seeded": n_seeded,
        "iterations": params.iterations,
        "views": n_views,
        "resolution": width,
        "scene_scale": round(scene_scale, 4),
        "tiles": tiles_summary,
        "lod": lod_summary,
        "metrics": metrics,
        "params": params.as_summary(),
        "bytes": out_path.stat().st_size,
        "out_path": str(out_path),
    }


def _train_one(  # noqa: ANN001
    torch,
    F,
    views,
    K,
    width: int,
    height: int,
    init: dict[str, np.ndarray],
    scene_scale: float,
    centers: np.ndarray,
    params: TrainParams,
    resume: bool,
    ckpt_dir: Path,
    progress: ProgressCb | None,
    tile_box=None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """ONE gsplat 2DGS optimization — the whole scene, or one tile — returning
    `(trained arrays as numpy, counters)`. `tile_box` ((lo, hi) world corners of
    the tile's expanded region) masks every loss to pixels this run OWNS (its own
    surface + true background — `_tile_mask`) and clips depth-seeding to the box;
    None trains unmasked, the single-run behavior. Checkpoints under `ckpt_dir`
    (the caller owns cleanup)."""
    from gsplat import DefaultStrategy, rasterization_2dgs

    device = torch.device("cuda")
    n_views = len(views)
    n_init = int(init["means"].shape[0])
    bands = (params.sh_degree + 1) ** 2

    # Resume from the latest checkpoint when present + compatible (same SH config),
    # else build from the surfel init. `meta` guards against reloading a checkpoint
    # whose model shape no longer matches this cloud's SH bands.
    meta = {"sh_degree": params.sh_degree, "n_init": n_init}
    ckpt = _load_checkpoint(torch, ckpt_dir, device) if resume else None
    if ckpt is not None and ckpt.get("meta", {}).get("sh_degree") != params.sh_degree:
        ckpt = None

    if ckpt is not None:
        splats = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v.to(device)) for k, v in ckpt["params"].items()}
        )
    else:
        splats = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.from_numpy(init["means"])),
                "scales": torch.nn.Parameter(torch.from_numpy(init["scales"])),
                "quats": torch.nn.Parameter(torch.from_numpy(init["quats"])),
                "opacities": torch.nn.Parameter(torch.from_numpy(init["opacities"])),
                "sh0": torch.nn.Parameter(torch.from_numpy(init["sh0"])),
            }
        ).to(device)
        if bands > 1:
            splats["shN"] = torch.nn.Parameter(
                torch.zeros((n_init, bands - 1, 3), dtype=torch.float32, device=device)
            )

    # A batched step averages B views' gradients, so scale every LR by sqrt(B) to
    # keep per-view convergence ~constant as `batch` rises — variance-matching
    # (sqrt, not linear), the safe rule for the quat/opacity/densification heads.
    # batch=1 → factor 1 (bit-identical to the pre-batch schedule).
    lr_scale = float(np.sqrt(max(params.batch, 1)))
    lrs = {
        "means": params.means_lr * scene_scale * lr_scale,
        "scales": params.scales_lr * lr_scale,
        "quats": params.quats_lr * lr_scale,
        "opacities": params.opacities_lr * lr_scale,
        "sh0": params.sh0_lr * lr_scale,
        "shN": params.shN_lr * lr_scale,
    }
    optimizers = {
        name: torch.optim.Adam(
            [{"params": [splats[name]], "lr": lrs[name], "name": name}], eps=1e-15
        )
        for name in splats.keys()
    }
    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / max(params.iterations, 1))
    )

    # reset_every=0 disables opacity resets by pushing the trigger past the last
    # step (see TrainParams). Accepted side effect: the strategy's scale-based
    # prune (gated on `step > reset_every`) also never fires — our init has no
    # giant blobs and the alpha/depth losses punish oversized splats directly.
    strategy = DefaultStrategy(
        prune_opa=params.prune_opa,
        grow_grad2d=params.grow_grad2d,
        grow_scale3d=params.grow_scale3d,
        refine_start_iter=params.refine_start_iter,
        refine_stop_iter=params.resolved_refine_stop,
        reset_every=params.reset_every if params.reset_every > 0 else params.iterations + 1,
        refine_every=params.refine_every,
        absgrad=params.absgrad,
        key_for_gradient="gradient_2dgs",
        verbose=False,
    )
    strategy.check_sanity(splats, optimizers)
    strat_state = strategy.initialize_state(scene_scale=scene_scale)

    # Restore optimizer / scheduler / densification state so a resumed run continues
    # exactly where it stopped (Adam moments, LR-decay position, grow/prune counters).
    # The scheduler is created above from the base (undecayed) LR, then its state is
    # loaded, so base_lrs stay correct while the current LR follows the checkpoint.
    start_step = 0
    if ckpt is not None:
        for name, opt in optimizers.items():
            sd = ckpt["optimizers"].get(name)
            if sd is not None:
                opt.load_state_dict(sd)
        means_sched.load_state_dict(ckpt["means_sched"])
        strat_state = _strat_state_to(ckpt["strat_state"], lambda v: v.to(device))
        start_step = int(ckpt["step"]) + 1

    window = _gaussian_window(torch, 11, 1.5, device, channels=3)
    dist_on = params.dist_lambda > 0.0

    # Nearest-camera KD-tree for the anti-aliasing scale floor (cameras are fixed,
    # so build once); `focal` (= fl_x) converts a camera distance to a pixel span.
    cam_tree = None
    focal = float(K[0, 0].item())
    if params.antialias and n_views > 0:
        from scipy.spatial import cKDTree

        cam_tree = cKDTree(centers)
    n_seeded = 0

    n_last = int(splats["means"].shape[0])
    if progress is not None:
        where = f"resume@{start_step} n={n_last}" if start_step else f"init={n_init}"
        progress(start_step, params.iterations, f"{where} views={n_views} scale={scene_scale:.2f}")

    t_start = t_last = time.perf_counter()
    done_last = start_step
    log_every = max(params.log_every, 1)
    import threading

    stop_ev = threading.Event()
    stream = _view_stream(
        torch, views, device, max(params.batch, 1), params.prefetch, params.seed,
        start_step, stop=stop_ev,
    )

    # Tile-mask constants: the box corners on-device + pixel-centre grids for the
    # per-batch depth unprojection. None → unmasked (single-run behavior).
    box_lo = box_hi = px_grid = py_grid = None
    if tile_box is not None:
        box_lo = torch.tensor(tile_box[0], dtype=torch.float32, device=device)
        box_hi = torch.tensor(tile_box[1], dtype=torch.float32, device=device)
        ys = torch.arange(height, dtype=torch.float32, device=device) + 0.5
        xs = torch.arange(width, dtype=torch.float32, device=device) + 0.5
        py_grid, px_grid = torch.meshgrid(ys, xs, indexing="ij")

    for step in range(start_step, params.iterations):
        viewmats, gt_rgb, gt_alpha, gt_depth = next(stream)

        # Tiled runs supervise only pixels this tile OWNS: its surface + true
        # background (exact ownership from the reference depth).
        mask = None
        if box_lo is not None and gt_depth is not None:
            world = _unproject_depth(torch, gt_depth, viewmats, K, px_grid, py_grid)
            mask = _tile_mask(torch, box_lo, box_hi, gt_alpha, gt_depth, world)

        active_sh = min(step // max(params.sh_degree_interval, 1), params.sh_degree)
        colors, sh_deg = _render_inputs(torch, splats, active_sh)

        renders, pred_alpha, normals, normals_from_depth, distort, median_depth, info = rasterization_2dgs(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=viewmats,
            Ks=K.unsqueeze(0).expand(viewmats.shape[0], -1, -1),
            width=width,
            height=height,
            sh_degree=sh_deg,
            packed=False,
            near_plane=params.near_plane,
            far_plane=params.far_plane,
            render_mode="RGB+ED",
            distloss=dist_on,
            depth_mode=params.depth_mode,
            absgrad=params.absgrad,
        )
        pred_rgb = renders[..., :3]
        pred_depth = median_depth if params.depth_mode == "median" else renders[..., 3:4]

        if params.refine:
            strategy.step_pre_backward(splats, optimizers, strat_state, step, info)

        # Photometric: full-frame L1 + D-SSIM. Both sides are premultiplied-over-
        # black (the reference alpha-blends over a black clear colour; gsplat
        # composites with no background), so glass/MASK pixels compare like-with-
        # like and background pixels directly penalize floater energy. Under a
        # tile mask, L1 averages over owned pixels only and SSIM sees ground
        # truth composited outside the mask (its windows carry no gradient from
        # content a neighbouring tile owns).
        if mask is None:
            l1 = (pred_rgb - gt_rgb).abs().mean()
            ssim_rgb = pred_rgb
        else:
            msum = mask.sum().clamp_min(1.0)
            l1 = ((pred_rgb - gt_rgb).abs() * mask).sum() / (msum * 3.0)
            ssim_rgb = torch.where(mask > 0, pred_rgb, gt_rgb.detach())
        ssim = _ssim(F, ssim_rgb.clamp(0, 1).permute(0, 3, 1, 2), gt_rgb.permute(0, 3, 1, 2), window)
        loss = (1.0 - params.ssim_lambda) * l1 + params.ssim_lambda * (1.0 - ssim)

        # Alpha (coverage/opacity) — the renderer's exact coverage masks.
        if params.alpha_lambda > 0.0:
            aerr = (pred_alpha - gt_alpha).abs()
            aloss = aerr.mean() if mask is None else (aerr * mask).sum() / mask.sum().clamp_min(1.0)
            loss = loss + params.alpha_lambda * aloss

        # Depth — alpha-gated L1 (metres) on the `depth_mode` statistic.
        if gt_depth is not None and params.depth_lambda > 0.0:
            gate = (gt_alpha > params.alpha_gate) & (gt_depth > 0)
            if mask is not None:
                gate = gate & (mask > 0)
            if gate.any():
                dl = ((pred_depth - gt_depth).abs() * gate).sum() / (gate.sum() + 1e-8)
                loss = loss + params.depth_lambda * dl

        # 2DGS normal consistency (render normals vs normals-from-depth), fg only.
        if params.normal_lambda > 0.0 and step >= params.normal_start_iter:
            nerr = 1.0 - (normals * normals_from_depth).sum(dim=-1)
            fg_mask = gt_alpha.squeeze(-1) > params.alpha_gate
            if mask is not None:
                fg_mask = fg_mask & (mask.squeeze(-1) > 0)
            if fg_mask.any():
                loss = loss + params.normal_lambda * nerr[fg_mask].mean()

        # 2DGS depth distortion.
        if dist_on and step >= params.dist_start_iter:
            loss = loss + params.dist_lambda * distort.mean()

        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()

        if params.refine:
            strategy.step_post_backward(splats, optimizers, strat_state, step, info, packed=False)
            # Adaptive VRAM: cap_max is the hard ceiling; also freeze densification
            # when free VRAM nears the WDDM shared-memory cliff (after one empty_cache
            # retry to release reusable cached blocks), so growth can't collapse
            # throughput into PCIe paging.
            if len(splats["means"]) > params.cap_max:
                strategy.refine_stop_iter = step
            elif params.vram_min_free_gb > 0.0 and step < strategy.refine_stop_iter:
                need = params.vram_min_free_gb * 1e9
                free_b = torch.cuda.mem_get_info()[0]
                if free_b < need:
                    torch.cuda.empty_cache()
                    free_b = torch.cuda.mem_get_info()[0]
                    if free_b < need:
                        strategy.refine_stop_iter = step
                        if progress is not None:
                            progress(
                                step + 1, params.iterations,
                                f"VRAM guard: {free_b / 1e9:.2f}GB free < {params.vram_min_free_gb}GB - "
                                f"froze densification at n={len(splats['means'])}",
                            )

        # Depth-guided densification: seed Gaussians at reference surfaces the splat
        # is missing (holes / too-far). After the strategy so this step's `info` is
        # never used against the grown tensors; inside the densification window and
        # only while VRAM allows growth.
        if (
            params.depth_densify
            and params.depth_densify_start <= step < strategy.refine_stop_iter
            and (step + 1) % max(params.depth_densify_every, 1) == 0
            and len(splats["means"]) < params.cap_max
        ):
            tight = (
                params.vram_min_free_gb > 0.0
                and torch.cuda.mem_get_info()[0] < 1.5 * params.vram_min_free_gb * 1e9
            )
            new = (
                None if tight
                else _depth_seed(
                    torch, gt_rgb, gt_alpha, gt_depth, pred_alpha.detach(),
                    pred_depth.detach(), viewmats, K, params, int(params.depth_densify_max),
                    box=(box_lo, box_hi) if box_lo is not None else None,
                )
            )
            if new is not None:
                _append_gaussians(torch, splats, optimizers, strat_state, new)
                n_seeded += int(new["means"].shape[0])

        # Anti-aliasing floor: keep every Gaussian resolvable by its nearest camera
        # so it can't shimmer across the scale-ladder octaves under free-fly.
        if params.antialias and cam_tree is not None and (step + 1) % max(params.aa_every, 1) == 0:
            _aa_scale_floor(torch, splats, cam_tree, focal, params.aa_min_scale_px)

        if progress is not None and (step % log_every == 0 or step == params.iterations - 1):
            now = time.perf_counter()
            done = step + 1
            n_now = len(splats["means"])
            rate = (done - done_last) / max(now - t_last, 1e-6)
            eta = (params.iterations - done) / max(rate, 1e-6)
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            rsv_gb = torch.cuda.memory_reserved() / 1e9
            progress(
                done,
                params.iterations,
                f"loss={float(loss):.4f} n={n_now} ({n_now - n_last:+d}) | "
                f"{rate:.2f} it/s ({rate * params.batch:.0f} views/s) | "
                f"elapsed {_fmt_hms(now - t_start)} ETA {_fmt_hms(eta)} | "
                f"vram {rsv_gb:.1f}GB used / {free_gb:.1f}GB free",
            )
            t_last, done_last, n_last = now, done, n_now

        if (
            params.ckpt_every > 0
            and (step + 1) % params.ckpt_every == 0
            and step + 1 < params.iterations
        ):
            _save_checkpoint(
                torch, ckpt_dir, step, splats, optimizers, means_sched,
                strat_state, meta, params.ckpt_keep,
            )

    # Release the prefetch worker (tiled runs create one stream per tile).
    stop_ev.set()

    # One-shot cleanup prune before returning: densification stopped at
    # refine_stop, so low-opacity floaters that drifted below prune_opa in the
    # back half are still present. Prune here so the returned model is the
    # shipped one.
    n_pruned = (
        _final_prune(torch, splats, params.prune_opa, params.prune_scale3d, scene_scale)
        if params.final_prune
        else 0
    )
    if progress is not None and n_pruned:
        progress(
            params.iterations, params.iterations,
            f"final prune: -{n_pruned} floaters below opacity {params.prune_opa} "
            f"(-> {int(splats['means'].shape[0])} splats)",
        )

    with torch.no_grad():
        arrays = {k: v.detach().cpu().numpy() for k, v in splats.items()}
    info = {
        "init": n_init,
        "final": int(arrays["means"].shape[0]),
        "seeded": n_seeded,
        "pruned": n_pruned,
        "views": n_views,
        "train_s": round(time.perf_counter() - t_start, 1),
    }
    return arrays, info


def _final_prune(torch, splats, prune_opa: float, prune_scale3d: float, scene_scale: float) -> int:  # noqa: ANN001
    """One-shot cleanup before export: drop Gaussians below `prune_opa` opacity (the
    low-opacity floaters left after densification stopped) and, when `prune_scale3d
    > 0`, any whose max tangent radius exceeds `prune_scale3d·scene_scale` (runaway
    blobs). Mutates `splats` in place — the optimizer is done, so the pruned
    parameters need no grad or Adam state — and returns the count removed."""
    with torch.no_grad():
        keep = torch.sigmoid(splats["opacities"]).flatten() >= prune_opa
        if prune_scale3d > 0.0:
            max_radius = torch.exp(splats["scales"]).max(dim=-1).values
            keep = keep & (max_radius <= prune_scale3d * scene_scale)
        removed = int((~keep).sum())
        if removed:
            for name in list(splats.keys()):
                splats[name] = torch.nn.Parameter(splats[name][keep], requires_grad=False)
    return removed


def _evaluate(torch, rasterization_2dgs, splats, views, K, width, height, params, device):  # noqa: ANN001
    """Mean PSNR (foreground), RGB L1, and alpha-gated depth L1 over a random
    subset of views (capped by `eval_max_views` — plans can have thousands)."""
    colors, sh_deg = _render_inputs(torch, splats, params.sh_degree)
    n_eval = min(len(views), params.eval_max_views)
    sel = (
        np.random.default_rng(params.seed).choice(len(views), size=n_eval, replace=False)
        if n_eval < len(views)
        else np.arange(len(views))
    )
    psnr_sum = l1_sum = depth_sum = 0.0
    rgb_views = depth_views = 0
    with torch.no_grad():
        for i in sel:
            v = views[int(i)]
            gt_rgb, gt_alpha, gt_depth = _load_view(torch, v, device)
            renders, _alpha, _normals, _nfd, _distort, median_depth, _info = rasterization_2dgs(
                means=splats["means"],
                quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=colors,
                viewmats=v["viewmat"].to(device).unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=width,
                height=height,
                sh_degree=sh_deg,
                packed=False,
                near_plane=params.near_plane,
                far_plane=params.far_plane,
                render_mode="RGB+ED",
                depth_mode=params.depth_mode,
            )
            pred_rgb = renders[..., :3].clamp(0, 1)
            pred_depth = median_depth if params.depth_mode == "median" else renders[..., 3:4]
            fg = gt_alpha > params.alpha_gate
            fg3 = fg.expand_as(gt_rgb)
            if fg3.any():
                mse = ((pred_rgb - gt_rgb)[fg3] ** 2).mean()
                psnr_sum += float(-10.0 * torch.log10(mse + 1e-10))
                l1_sum += float((pred_rgb - gt_rgb)[fg3].abs().mean())
                rgb_views += 1
            if gt_depth is not None:
                gate = fg & (gt_depth > 0)
                if gate.any():
                    depth_sum += float((pred_depth - gt_depth).abs()[gate].mean())
                    depth_views += 1
    return {
        "psnr": round(psnr_sum / max(rgb_views, 1), 3),
        "rgb_l1": round(l1_sum / max(rgb_views, 1), 5),
        "depth_l1": round(depth_sum / max(depth_views, 1), 5) if depth_views else None,
        "eval_views": int(n_eval),
    }


def _main() -> None:
    """Standalone CLI for the GPU box: fine-tune one cell's surfel cloud against
    its Stage-5 references."""
    import argparse

    ap = argparse.ArgumentParser(description="Stage 6 — gsplat 2DGS splat fine-tune")
    ap.add_argument("--cloud", required=True, type=Path, help="Stage-3 cloud.ply (init)")
    ap.add_argument("--refs", required=True, type=Path, help="Stage-5 refs/ dir")
    ap.add_argument("--out", required=True, type=Path, help="output trained.ply")
    ap.add_argument(
        "--iterations", type=int, default=TrainParams.iterations,
        help="VIEW-DRAW budget (optimizer steps × batch); steps run = iterations // batch",
    )
    ap.add_argument(
        "--epochs", type=float, default=TrainParams.epochs,
        help="passes over the view set; overrides --iterations "
             "(budget = epochs × n_views, steps = budget // batch)",
    )
    ap.add_argument("--sh-degree", type=int, default=TrainParams.sh_degree)
    ap.add_argument(
        "--refine-stop-iter", type=int, default=None,
        help="stop densification at this step (default: 50%% of iterations)",
    )
    ap.add_argument(
        "--batch", type=int, default=TrainParams.batch,
        help="views per optimizer step (fills the GPU); steps = budget // batch, "
             "so it's a speed knob at CONSTANT work (no need to adjust iterations)",
    )
    ap.add_argument(
        "--ckpt-every", type=int, default=TrainParams.ckpt_every,
        help="write a resumable checkpoint every N steps (0 disables)",
    )
    ap.add_argument(
        "--vram-min-free-gb", type=float, default=TrainParams.vram_min_free_gb,
        help="freeze densification when free VRAM drops below this (0 disables)",
    )
    ap.add_argument(
        "--antialias", action=argparse.BooleanOptionalAction, default=TrainParams.antialias,
        help="Mip-style per-Gaussian scale floor from the nearest camera (anti-shimmer)",
    )
    ap.add_argument(
        "--depth-densify", action=argparse.BooleanOptionalAction, default=TrainParams.depth_densify,
        help="seed Gaussians at reference-depth surfaces the splat is missing",
    )
    ap.add_argument(
        "--tile-max", type=int, default=None,
        help="max seed Gaussians for a single run; larger clouds train as ground-"
             "plane tiles and merge (default: 2/3 of cap_max; 0 disables tiling)",
    )
    ap.add_argument(
        "--lod-levels", type=int, default=TrainParams.lod_levels,
        help="LOD ladder levels exported beside trained.ply (0 disables)",
    )
    ap.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="resume from the latest checkpoint beside --out (splat/ckpt) if present",
    )
    ap.add_argument("--run", default="?")
    ap.add_argument("--slot", default="?")
    ap.add_argument("--model", default="?")
    args = ap.parse_args()

    def _log(done: int, total: int, msg: str) -> None:
        pct = 100.0 * done / max(total, 1)
        print(f"[stage6 {time.strftime('%H:%M:%S')}] {done}/{total} ({pct:4.1f}%) {msg}", flush=True)

    summary = train_splat(
        run=args.run,
        slot=args.slot,
        model=args.model,
        cloud_path=args.cloud,
        refs_dir=args.refs,
        out_path=args.out,
        params=TrainParams(
            iterations=args.iterations,
            epochs=args.epochs,
            sh_degree=args.sh_degree,
            refine_stop_iter=args.refine_stop_iter,
            batch=args.batch,
            ckpt_every=args.ckpt_every,
            vram_min_free_gb=args.vram_min_free_gb,
            antialias=args.antialias,
            depth_densify=args.depth_densify,
            tile_max=args.tile_max,
            lod_levels=args.lod_levels,
        ),
        resume=args.resume,
        progress=_log,
    )
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    _main()
