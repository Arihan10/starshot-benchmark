"""Stage 6 — Splat fine-tune (the training run) via gsplat.

REPRESENTATION (`TrainParams.representation`) selects which gsplat rasterizer —
and therefore which primitive — the whole stage trains, exports and heals:

  * **"2dgs"** (default) — `rasterization_2dgs`: flat, surface-aligned surfels
    (two tangent scales, the third axis a splitter decoy). The render operator
    matches Stage 3's init exactly, its MEDIAN depth lands on the nearest opaque
    surface (the glass-safe depth target), and it carries the normal-consistency
    + distortion regularizers. Best geometry; the historical path, unchanged.
  * **"3dgs"** — `rasterization`: full 3D Gaussians (three real scales). Gains
    what gsplat only implements on the 3DGS path — AbsGS densification
    (`absgrad`), the Mip-Splatting 2D opacity compensation
    (`rasterize_mode="antialiased"`), and the packed projection — and matches
    what every DELIVERY renderer actually is (PlayCanvas / mkkellogg gsplat both
    rasterize 3D Gaussians; the SOG/ksplat encoders currently have to fake a
    third scale). Costs the three 2DGS-only rasterizer outputs: median depth
    (→ expected depth, see `depth_mode`), rendered normals (→ the `flat_lambda` /
    `aniso_lambda` parameter-space surface priors) and distortion (unused here).

Everything else in the stage — the view stream, the tiling, depth-guided
densification, the compaction measure, the LOD ladder, the checkpoints — is
shared, and the on-disk `.ply` differs only in its scale-column count (two vs
three), which every downstream consumer already handles.

Takes a **COLMAP model** as its ONLY input — the same (point cloud + camera poses
+ reference images) triple Postshot ingests and gsplat's `simple_trainer_2dgs`
trains on, written by `splat_to_colmap.py` / `splat.colmap.export_colmap`
(`cameras.txt` / `images.txt` / `points3D.txt` + RGBA images). The splat is
INITIALIZED from the point cloud the gsplat way (`_init_from_points`: means = point
xyz, colour = point RGB, isotropic KNN scales, random quats, opacity =
logit(`init_opa`)) and optimized against the reference images into a clean 2DGS
splat (`trained.ply`): densification adds Gaussians where the render disagrees with
the reference, pruning removes redundant ones.

The trainer reads the images as RGB — the export's PNGs also carry a coverage
alpha (kept for Postshot masking) but it isn't read here — and COLMAP has no depth,
so the alpha/coverage loss, the dense depth loss, and depth-guided densification are
DISABLED here (nothing to compare against) — the active loss set is exactly the reference trainer's: photometric L1 +
D-SSIM, 2DGS normal consistency, and optional depth distortion. (The depth/alpha
machinery below is retained because Stage 7 `heal_splat` still consumes the
alpha+depth Stage-5 references.)

WHAT THE FINE-TUNE FIXES (all lighting-independent of the loss's own): render-
operator errors that only appear once the primitives are depth-sorted +
alpha-blended through the real rasterizer — silhouettes, thin geometry, zoom-in
detail, alpha edges — plus floaters. The surfel init is ~90% there; this polishes
the rest.

CONTRACT (locked, shared with Stages 4/5 — see overview §12):
  * Init map (`cloud.ply`, a Stage-3 2DGS `.ply`, SH degree 0):
      means = xyz; quats = rot_0..3 (wxyz); opacity = logit (pre-sigmoid);
      scale_0/1 = log tangent radii (isotropic); f_dc_0..2 = SH0 colour coeffs.
      Both rasterizers take 3 scales, so a third log-scale is synthesized off the
      smaller tangent radius — and since Stage 3 aligns the quaternion's local +Z
      to the surface normal, that axis IS the thickness. Under 2DGS it is a
      training-only decoy at `_THIN_AXIS_FRAC` (nothing renders it; it exists so
      the strategy's split doesn't eject children off the surface plane, and it
      never ships). Under 3DGS it is real geometry at
      `TrainParams.init_thickness_frac`, and it ships.
  * Poses: `transform_matrix` is OpenCV camera-to-world → `viewmats = inv(c2w)`.
    Intrinsics: pinhole `K = [[fl_x,0,cx],[0,fl_y,cy],[0,0,1]]`.
  * Depth: planar camera-space Z (metres), decoded from the SZF frame's log-uint16
    codes via the shared [near, far] (legacy 16-bit PNG / float32 `.npy` sets still
    read as-is). The loss compares the reference against the splat's MEDIAN depth
    (`depth_mode="median"` default, the transmittance-0.5 crossing) — it lands on
    the nearest OPAQUE surface, exactly what the capture stores as depth GT (BLEND
    glass doesn't write depth), so a transmissive pane at α ≈ 0.065 stays below the
    crossing and is invisible to the depth loss instead of being razed as a
    floater. `depth_mode="expected"` (the ED channel) additionally penalizes low-
    opacity floaters in front of opaque surfaces, but treats real glass as one of
    them — reserve it for runs with no glass and no geometric floater cull.
  * Colour: spherical harmonics to degree 3 (PostShot's default) — a per-Gaussian
    DC term (`sh0`) plus 15 higher-order RGB coefficients (`shN`) — compared
    directly against the LIT references (no sRGB->linear). The Stage-5 capture
    renders view-DEPENDENT PBR (per-view specular + reflections), so higher-order
    SH reconstructs those moving highlights as f(view direction) instead of
    averaging them into one flat colour; the active degree warms up 0→3 over
    training (`sh_degree` / `sh_degree_interval`). Set `sh_degree=0` for the old
    flat / view-independent model.

LOSSES (per view). The photometric pair is the objective; everything else is a
weak prior on top of it, at the reference trainer's weights — the capture's extra
channels are worth having, but only as a nudge on geometry the images already
determine:
  * photometric — full-frame L1 + D-SSIM on RGB vs the reference. Both
    sides are premultiplied-over-black (the reference alpha-blends over a black
    clear colour; gsplat composites with no background), so glass/MASK pixels
    compare like-with-like and background pixels directly penalize floater
    energy;
  * alpha (mask) — L1(render α, reference α): the renderer's exact coverage
    masks make empty space stay empty and glass stay see-through. Only informative
    on views that see the void — an enclosed interior view is α≡1 everywhere;
  * depth — L1 in DISPARITY space (× scene_scale) on the median depth (the
    transmittance-0.5 crossing, which lands on the nearest opaque surface and so
    leaves transmissive glass untouched; `depth_mode="expected"` instead hunts
    front floaters at the cost of razing glass — see TrainParams.depth_mode),
    gated to pixels where BOTH the reference and the render hold a surface. On the
    3DGS path only expected depth exists, so the gate additionally EXCLUDES the
    pixels the init cloud says hold a transmissive layer (`glass_guard`);
  * geometry prior, per representation — 2DGS: normal consistency (render normals
    vs normals-from-depth, the latter scaled by rendered alpha as in the reference)
    plus optional depth distortion. 3DGS: flatness and bounded anisotropy on the
    scale parameters (`flat_lambda` / `aniso_lambda`), since that rasterizer
    returns neither a normal nor a distortion map to tie a loss to.

RESUMABLE: every `ckpt_every` steps the full training state (params + per-param
Adam + means LR schedule + densification-strategy accumulators + step) is written
to `splat/ckpt/` (atomic, most-recent-`ckpt_keep` kept). An interrupted run resumes
from the latest checkpoint and continues to `iterations`; the checkpoints are
deleted once trained.ply is written. Pass `resume=False` to force a fresh run.

TILED TRAINING (scenes past the single-GPU VRAM wall): a seed larger than the
tile budget (`tile_max`, default `_TILE_BUDGET_DEFAULT`) trains as
GROUND-PLANE TILES instead of one frozen over-budget run — the smallest (x,z)
grid whose largest expanded tile (core + margin ring) fits the budget. Each tile
trains sequentially in this process with the FULL budget to itself: views are
assigned by exact visibility (their reference-depth pixels unproject into the
tile's expanded box), every loss is masked to pixels the tile OWNS (its surface
+ true background, so boundary Gaussians never chase foreign content), and
depth-seeding is clipped to the box. Merge keeps each Gaussian iff its mean lies
in its tile's CORE cell — cores partition space exactly, so overlaps never
double-ship. Per-tile results are cached (`splat/ckpt/tiles/`) with a params
signature: an interrupted tiled run resumes at the first unfinished tile (and
inside it, from its own checkpoint). Because the references are exact synthetic
renders (posed, depth-true), the classic tiling artifacts of photogrammetry
(exposure seams, mis-assigned cameras) don't apply.

TILING IS A FALLBACK, NOT A FREE WIN — prefer a single run whenever the seed
fits. Exact core ownership rules out DUPLICATION, not appearance discontinuity:
two Gaussians either side of a core boundary are optimized in different runs,
under different masked context, and are never evaluated together, so their SH can
disagree. Two further limits follow from the per-tile loss mask:
  * A pixel whose reference surface lies outside a tile's expanded box is masked
    OUT for that tile, so geometry occluded by another tile's content receives no
    supervision at all and keeps its init.
  * TRANSMISSIVE content spanning a boundary is baked wrong. Glass writes no
    depth, so a window pixel's reference depth is the opaque surface BEHIND the
    pane. If that surface sits outside the pane's tile, the pane is unsupervised
    there; and the tile owning the surface fits it WITHOUT the pane in front, so
    it converges to the already-attenuated reference colour. Compositing the two
    at merge attenuates twice. Only bites when pane and backing surface land in
    different expanded boxes (the margin ring covers the common case), so it is a
    boundary defect, not a pervasive one — but it is a reason to raise the budget
    rather than tile eagerly.

LOD EXPORT (wide shots / progressive delivery): beside trained.ply, an octave
ladder `trained.lod1.ply`, `trained.lod2.ply`, … — each level ~4× fewer
Gaussians, built by opacity·area-weighted MOMENT MATCHING (cluster mean/
covariance → tangent frame via eigendecomposition; opacity preserves the
cluster's opacity·area within the new disk). A pulled-back camera renders the
coarse level as a prefiltered anti-aliased average instead of shimmering
sub-pixel splats, and the same `.ply` layout (and representation) as the base
model means every existing viewer / compressor reads the levels unchanged.

COMPACTION (delivered-size cleanup): after the optimization, a measured pass
deletes Gaussians the delivered splat doesn't need, then briefly re-tunes the
survivors — the dominant lever on delivered size (the SOG/PLY footprint is
~linear in count, and densification stacks Gaussians many-deep while depth-
seeding scatters some off-surface). Three signals compose (`_select_keep`):
  1. LOSSLESS contribution — each Gaussian's TOTAL rendered blend weight is
     MEASURED through the rasterizer (a pixel is Σᵢ wᵢ·cᵢ, wᵢ = opacity × kernel
     × transmittance, so one backward of Σ(all pixels of all views) w.r.t. the
     SH0 colours accumulates each Gaussian's Σ wᵢ). Below `compact_eps`
     (0.5/255) a Gaussian can't shift ANY pixel by half a display step even with
     its whole mass on one pixel — deleting it is imperceptible by construction.
  2. SURFACE PRIOR (`surface_max_dist` m) — drop any Gaussian farther than that
     from every Stage-3 init surfel. Those surfels sit EXACTLY on the source
     meshes, so "airborne" is decidable, not inferred (a filter photogrammetry
     can't have): this removes the opaque floater clouds an under-supervised
     open scene grows above/around objects (and the depth-seeds that landed at
     wrong depths). Set 0 to disable; raise it for scenes with metre-scale-
     sampled huge objects (skydome/terrain), whose legitimate surfels are sparse.
  3. BUDGET — cut the low-contributors, gated by MEASURED quality. Default mode
     is ADAPTIVE (`compact_max_db_drop`, 1.0 dB): bisect the keep fraction, and
     for each probe prune → short heal → evaluate PSNR against the reference
     views; accept the deepest cut whose drop from the post-cull baseline stays
     within the budget. Every scene thus finds its own safe cut — a bloated one
     sheds 60-70%, an information-dense one automatically keeps more — instead
     of trusting a fixed percentage. `compact_keep_frac` (used when the dB gate
     is None) applies a fixed scene-relative fraction instead; neither is ever
     an absolute count cap. Ranking is OPACITY-NORMALIZED (weight/α), so
     translucent-by-design content (glass panes at α≈0.065) is scored by what
     it covers, not penalized for being see-through.
Pure-black Gaussians (gradient-blind to the measure) are force-kept unless the
surface prior culls them. Then `compact_heal_steps` of the standard supervision
loss (no densification, means LR damped) let neighbours absorb the deleted
Gaussians' residual so nothing tears. Runs per TILE against that tile's own
views + init surfels. The dB gate is a GLOBAL-AVERAGE guard (PSNR over the eval
subset) — it bounds overall fidelity, not any single detail. `compact=False`
disables the whole pass.

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
from typing import Any, NamedTuple

import numpy as np

from splat.colmap import SIDECAR_NAME as _SIDECAR_NAME
from splat.stage5 import (
    TRANSFORMS_NAME,
    decode_depth_u16,
    load_depth_png,
    load_reference_frame,
)

logging.getLogger("PIL").setLevel(logging.ERROR)

# The optimized splat written under a cell's `splat/` dir (the same `.ply` layout
# Stage 3 emits, so Stage 7/8 + the viewer read it identically; the scale-column
# count records which representation trained it — see `_ply_representation`).
TRAINED_NAME = "trained.ply"

# SH degree-0 basis constant (matches Stage 3/4): colour = 0.5 + C0·f_dc, so a
# seeded Gaussian's albedo → f_dc = (rgb − 0.5)/C0.
_SH_C0 = 0.28209479177387814

# Lower bound on the rendered depth entering the DISPARITY loss, as a fraction of
# the reference depth for that pixel — the singularity guard for 1/d. See the depth
# term in `_supervision_loss`; 0.1 caps the residual at 9× the true disparity.
_DEPTH_DISP_FLOOR_FRAC = 0.1

# The two primitives this stage can train (see the module docstring).
REPRESENTATIONS = ("2dgs", "3dgs")

# 2DGS third-scale fraction. On the 2DGS path the normal axis is NOT rendered, so
# this is purely a splitter guard: `DefaultStrategy.split` displaces children along
# all three scaled axes, and a normal-axis scale equal to the tangent radius would
# eject every child a full radius off its own plane. 1% of the smaller tangent
# radius keeps splits in-plane. (On the 3DGS path the third axis is real geometry,
# sized by `TrainParams.init_thickness_frac` instead.)
_THIN_AXIS_FRAC = 0.01

# Tolerance (fraction of reference depth) below which the init cloud's two-layer
# EXPECTED depth is taken to agree with the stored opaque plane — i.e. the pixel
# holds no transmissive layer. Used only when the depth statistic is `expected`
# (always, on the 3DGS path), to keep the opaque-plane depth loss off glass; see
# the depth term in `_supervision_loss`.
_ED_GLASS_TOL_FRAC = 0.02

# progress(done, total, message) — called periodically during training.
ProgressCb = Callable[[int, int, str], None]

# Floor on the resolved optimizer-step count (matches the tiled per-tile floor):
# even a tiny epoch/budget still runs enough steps to seat the LR schedule.
_MIN_STEPS = 200

# Sampling period for the in-loop speed profile (`_profile_report`). The GPU
# sections need a `cuda.synchronize()` on each side to be attributed correctly,
# which would distort the run if done every step — so they are measured on 1 step
# in this many. The loader-wait figure, which is the one that decides the verdict,
# is exact and measured on every step.
_PROFILE_EVERY = 250

# Default single-run seed ceiling: clouds larger than this train as ground-plane
# TILES (module docstring §TILED TRAINING). Deliberately HIGH, because tiling is a
# fallback for seeds that cannot train monolithically AT ALL — not a routine path.
# It costs n_tiles × the run, bakes transmissive content wrongly where a pane and
# its backing surface straddle a boundary, and can leave appearance
# discontinuities across core edges, none of which a single run has. It has also
# never fired on any recorded run (the largest cloud trained so far is 281k), so
# the whole path is unvalidated in production — one more reason to reach it late.
#
# Sized for the 48 GB L40S `modal_app.py` provisions (it was set against a 40 GB
# A100, so the move only added headroom): 4M seeds is ~3.8 GB of
# persistent state (59 floats/Gaussian for params, ×4 for the two Adam moments and
# the gradient), leaving the bulk of the card for rasterization activations, which
# are the real consumer — the tile-intersection buffers scale with projected AREA,
# not count. Growth PAST this point is bounded reactively by `vram_min_free_gb`
# rather than by any count, now that `cap_max` is off by default; this number only
# decides whether the SEED gets one run or several. Confirm against the training
# heartbeat's `vram used / free` line before raising it further.
#
# (It used to be derived as 2/3 of `cap_max`, which coupled a scene's tiling
# structure to an unrelated knob: lowering the count ceiling silently re-tiled the
# scene, changing the grid, the view assignment and the cache signature.)
_TILE_BUDGET_DEFAULT = 4_000_000


@dataclass(frozen=True)
class TrainParams:
    """Stage-6 knobs. Learning rates follow the gsplat/3DGS defaults; the means LR
    is scaled by the scene extent at runtime and decayed exponentially.

    SCHEDULE (resolved per run/tile by `resolve_schedule`): `iterations` is the
    OPTIMIZER-STEP count — what gsplat's `max_steps` and PostShot's step box both
    mean — and `batch` is how many reference images each of those steps averages,
    so raising `batch` MULTIPLIES the work per step and leaves the number of Adam
    updates alone. (An iterative optimizer's progress is bounded by its update
    count, and every densification cadence below is denominated in updates, so
    dividing steps by `batch` — as this class used to — silently shortened a run
    by that factor: a `batch=16` job spec turned a nominal 30k into 1.5k.) Every
    cadence here is written against the reference 30k-step length.

    `epochs`, when set, instead sizes the run as whole passes over the view set
    (`steps = epochs × n_views / batch`) — the scene-size-independent way to dial
    training length, since a bigger scene has proportionally more views (Stage 4
    places cameras by surface area). Shortening a run that way rescales EVERY
    cadence by the same factor (gsplat's `adjust_steps`), so the densify window,
    the SH warm-up and the regularizer starts keep a full-length run's
    proportions instead of eating the whole budget.

    `refine_stop_iter` defaults to None → 50% of the step count, the reference's
    15k/30k."""

    # PRIMITIVE — "2dgs" (surfels, the default) or "3dgs" (full 3D Gaussians).
    # See the module docstring §REPRESENTATION. This one field selects the gsplat
    # rasterizer, which regularizers are available, the third-scale semantics, the
    # densification gradient key and the exported `.ply` scale-column count, so it
    # is part of every cache signature (`as_summary`, `_TileGrid.signature`) and of
    # the checkpoint metadata — a run can never resume another representation's
    # state.
    representation: str = "2dgs"

    # --- 3DGS-ONLY quality knobs (ignored, and inert, under "2dgs") ------------
    # AbsGS densification (arXiv:2404.10484): densify on the ABSOLUTE per-view 2D
    # gradient instead of its signed mean, so a Gaussian straddling two edges that
    # pull opposite ways still registers as under-fitted. It is the single biggest
    # quality lever gsplat offers that the 2DGS path CANNOT use — the 2DGS backward
    # attaches `.absgrad` to `means2d` while `DefaultStrategy` densifies off the
    # separate `gradient_2dgs` tensor, which never receives one (see `grow_grad2d`).
    # None = auto: ON for 3dgs, forced OFF for 2dgs. Absolute gradients are
    # strictly larger than signed means, so the split threshold must rise with it —
    # `grow_grad2d_abs` replaces `grow_grad2d` whenever absgrad is active (0.0008
    # is gsplat's own recommendation for the swap).
    absgrad: bool | None = None
    grow_grad2d_abs: float = 0.0008
    # Screen-space filter (gsplat `rasterize_mode`). "antialiased" applies the
    # Mip-Splatting 2D compensation: the rasterizer dilates every projected 2D
    # covariance by `eps2d` to keep sub-pixel Gaussians from vanishing between
    # samples, and this scales opacity by sqrt(det(Σ)/det(Σ+eps2d)) so a Gaussian
    # the camera cannot resolve gets DIMMER instead of being smeared over a full
    # pixel at full strength. Together with the `antialias` scale floor below (the
    # Mip-Splatting 3D filter) this is the complete Mip-Splatting pair, and it is
    # the right answer to the free-fly shimmer the scale ladder otherwise fights.
    # "classic" = the original 3DGS dilation, no compensation.
    rasterize_mode: str = "antialiased"
    # gsplat's packed projection: keep only the (camera, Gaussian) pairs that
    # actually intersect, instead of a dense [C,N] buffer. Big VRAM saving on wide
    # plans — which is what decides whether a scene needs TILING at all — at a
    # small indexing cost. 3DGS-only: gsplat's packed 2DGS RGB+ED path is broken
    # (its SH branch mis-broadcasts and the precomputed-colour path skips the
    # visible-subset gather), so it is forced off under "2dgs".
    packed: bool = False
    # Init thickness of the third (normal) axis, as a fraction of the tangent
    # radius, when a 3DGS run seeds from a 2-scale Stage-3 cloud. Matches Stage 3's
    # own `_FLATNESS`, i.e. a ~3 mm shell on a 3 cm surfel: thin enough to read as
    # a surface from step 0, thick enough that the axis carries real gradient
    # (the 2DGS decoy `_THIN_AXIS_FRAC` of 1% is 10x below that and would start the
    # normal axis effectively frozen).
    init_thickness_frac: float = 0.1
    # SURFACE PRIOR, the 3DGS stand-in for 2DGS normal consistency. 3DGS has no
    # rendered normal to tie to the depth, so flatness is imposed in PARAMETER
    # space instead. Every surface in these scenes comes from a mesh, so "stay a
    # disk" is exactly the right prior — it is what keeps a 3D Gaussian from
    # fattening off-surface once the photometric loss is satisfied.
    #
    # A CONSTRAINT, not a standing pressure: the penalty is relu(smallest/largest −
    # `flat_max`), so anything already flatter than that ratio costs nothing (the
    # init, at `init_thickness_frac`, starts free). A plain ratio penalty would
    # instead push every Gaussian toward zero thickness for the whole run — and
    # since Adam sizes its step by gradient CONSISTENCY rather than magnitude, even
    # a tiny constant pressure on an axis the render barely depends on walks that
    # axis down ~lr per step, i.e. to collapse. The denominator is detached either
    # way, so the term can only THIN a Gaussian, never inflate its footprint to
    # satisfy the ratio. Shares `normal_start_iter`'s warm-up: like normal
    # consistency, it must not lock in half-built geometry.
    flat_lambda: float = 0.05
    flat_max: float = 0.25
    # NEEDLE guard: penalize (largest / middle scale) above `aniso_max`. Measured
    # on a real run, unbounded 3D Gaussians reached 256:1 anisotropy, which is the
    # classic 3DGS spike artifact — invisible from the training views that made it
    # and a bright sliver from anywhere else. Unlike `flat_lambda` this term is
    # two-sided by design (shrink the long axis, grow the middle one).
    aniso_lambda: float = 0.01
    aniso_max: float = 10.0

    # OPTIMIZER STEPS (gsplat `max_steps` / PostShot's step box), and the length
    # every cadence below is written against. `epochs` overrides it when set.
    iterations: int = 30_000
    epochs: float | None = None

    # Loss weights.
    ssim_lambda: float = 0.2           # photometric = (1-λ)·L1 + λ·(1-SSIM)
    # L1(render α, reference α). The capture's alpha is renderer COVERAGE, not an
    # object matte, so an enclosed interior view is α≡1 everywhere and this term
    # degenerates into a uniform "fill the frame" push. It carries real
    # information only on the shell views that see the black void, where it
    # penalizes floater coverage the photometric L1 can miss (a BLACK floater over
    # black background). Kept at regularizer strength for exactly that: at the old
    # 0.5 it outweighed the photometric term through the whole densification
    # window, so Gaussians were sized to cover rather than to match.
    alpha_lambda: float = 0.05
    # Depth L1 in DISPARITY space (× scene_scale, so the term is dimensionless and
    # a far wall's metric error no longer outweighs a near surface's) — the
    # reference trainer's formulation and weight. Deliberately small: under
    # `depth_mode="median"` gsplat's 2DGS backward hands the whole per-pixel depth
    # gradient to the ONE Gaussian that crossed transmittance 0.5, unweighted by
    # its blend weight (`RasterizeToPixels2DGSBwd.cu`: `v_rgb_local[CDIM-1] +=
    # v_median`), so this is a nudge on geometry the photometric terms already
    # place — not a term to converge on. At the old metric 0.5 it was ~70% of the
    # converged loss and every pixel yanked one Gaussian along its ray at full
    # strength.
    depth_lambda: float = 0.01
    # "Record both" — the SECOND depth target, the glass-maker. The stored depth
    # plane is the nearest OPAQUE surface, so `depth_lambda` (median) pins the wall
    # but says nothing about a transmissive pane in front of it. This term
    # supervises the splat's EXPECTED depth (the α-weighted mean) toward the true
    # TWO-LAYER expected depth, derived at train time from the mesh-exact init
    # cloud (whose glass surfels sit at α≈0.065 on the panes). Deleting a pane
    # shifts the splat's expected depth back onto the wall — off this target — so
    # it is a POSITIVE signal that *requires* the glass, while the median term
    # keeps the wall pinned (the two together are well-posed: median forbids
    # sliding the wall forward to fake the expected depth, so the only way to
    # satisfy both is a wall at its true depth + a pane in front). Derived from the
    # init cloud, so NO capture change / re-render is needed. 0 disables (default
    # off until validated); only meaningful with the surfels init.
    depth_expected_lambda: float = 0.0
    # Keep the opaque-plane depth term OFF transmissive pixels when the depth
    # statistic is EXPECTED (i.e. always, on the 3DGS path — see `depth_mode`).
    # Expected depth cannot distinguish a correctly-reconstructed pane from a
    # floater, so supervising it against the stored opaque plane is a standing
    # instruction to delete the pane; this renders the frozen init cloud's own
    # two-layer expected depth and drops the pixels where the two disagree. It is
    # what stops the 3DGS path from repeating the measured 2DGS `depth_mode=
    # "expected"` result (glass razed). Costs ONE extra no-grad degree-0 forward
    # per step — and nothing at all when `depth_expected_lambda` > 0, which already
    # renders that cloud, or when `depth_lambda` is 0. Turn it off for a scene with
    # no transmissive materials.
    glass_guard: bool = True
    alpha_gate: float = 0.5            # reference α above this = opaque pixel (depth/normal masks)
    # 2DGS-ONLY (the 3DGS rasterizer returns neither a rendered normal nor a
    # distortion map): normal consistency and depth distortion. Under "3dgs" both
    # are inert and `flat_lambda` / `aniso_lambda` stand in for the first.
    normal_lambda: float = 0.05        # 2DGS normal consistency
    dist_lambda: float = 0.0           # 2DGS depth distortion (off by default; over-flattens bounded scenes)
    # Let geometry settle before the regularizers bite — the reference's 7k/30k
    # and 3k/30k. Normal consistency in particular ties the rendered normals to
    # the render's OWN depth, so it locks in whatever surface exists when it
    # switches on; starting it at 2k (7%) froze half-built geometry.
    normal_start_iter: int = 7000
    dist_start_iter: int = 3000
    # Depth-loss warm-up (optimizer steps, like every cadence here): the
    # point-cloud init starts translucent (init_opa), so the early rendered depth
    # (expected blends through surfaces; median sits behind them until transmittance
    # crosses 0.5) is unreliable; let opacity saturate before the metric term bites.
    # (The opaque surfel init never needs this — its depth is true from step 0.)
    depth_start_iter: int = 500

    # Learning rates (means_lr is × scene_scale at runtime). sh0_lr is the SH DC
    # (base-colour) LR (INRIA `feature_lr`); shN_lr is the higher-order SH LR at
    # 1/20 of it (INRIA `feature_lr / 20`) — the standard 3DGS split PostShot's
    # lineage follows, so the view-dependent bands move slowly and don't fight the
    # DC colour early on.
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shN_lr: float = 1.25e-4

    # Spherical harmonics (view-dependent colour). Mirrors PostShot's "Max Sph.
    # Hrm. Degree" (default 3): each Gaussian carries (sh_degree+1)²−1 higher-order
    # RGB coefficients (15 at degree 3) on top of the DC term, so the specular
    # highlights + reflections the LIT Stage-5 capture bakes per view are
    # reconstructed as f(view direction) instead of averaged into one flat colour.
    # The ACTIVE degree WARMS UP from 0, +1 every `sh_degree_interval` optimizer
    # steps (the INRIA/3DGS schedule), so geometry settles before the
    # view-dependent bands switch on. sh_degree=0 = flat model (the old
    # view-independent behaviour), degree 3 = PostShot's default.
    sh_degree: int = 3
    sh_degree_interval: int = 1000

    # INIT SOURCE. "surfels" (default): the Stage-3 cloud (`init_ply`) seeds
    # every Gaussian at the 2DGS solution — on-surface means, mesh-true quats,
    # tangent-disc scales, exact texel colours, solid opacity. "points": the
    # gsplat `create_splats_with_optimizers` recipe from the COLMAP points3D
    # (positions + colours only): opacity = logit(init_opa); each scale =
    # log(mean distance to 3 nearest neighbours × init_scale) (isotropic);
    # random quats. The Postshot-parity A/B baseline.
    init: str = "surfels"
    # Surfel-init opacity CEILING (the one saturation trap of a near-solution
    # init): Stage 3 clamps alpha at 1-1e-3 → logit ≈ 6.9, where the sigmoid's
    # gradient is ~1e-3 and opacity is effectively FROZEN — a mirror/glass
    # surface could never turn transmissive and floaters could never be
    # trained away. Capping at 0.9 (logit ≈ 2.2) keeps opacity live while
    # still rendering solid from step 0.
    init_opa_max: float = 0.9
    # "points"-init knobs (gsplat defaults).
    init_opa: float = 0.1
    init_scale: float = 1.0

    # Densification (gsplat DefaultStrategy, 2DGS gradient key).
    refine: bool = True
    refine_start_iter: int = 500
    refine_stop_iter: int | None = None  # None → int(iterations * 0.5) at runtime
    refine_every: int = 100
    # Periodic opacity resets (the reference 3DGS/2DGS floater purge: clamp every
    # opacity to 2·prune_opa + zero its Adam moments, so only image-justified
    # Gaussians re-earn visibility) — DISABLED by default, deliberately:
    #   * Passes between resets = reset_every·batch / n_views. Our plans carry
    #     THOUSANDS of views (swamp-land: 4212 → 0.7 passes between resets at the
    #     reference 3000/batch 1, against the ~10 the reference's few-hundred-view
    #     scenes get), so the clamp outruns recovery and ships a half-transparent
    #     scene — measured on the 2026-07-23 runs (hotel 24.5 dB, swamp healed to
    #     9.7k splats, vs the old loop's 33 dB).
    #   * This pipeline doesn't need the amnesty cycle to kill floaters: Stage 7
    #     settles them GEOMETRICALLY — the Stage-3 cloud sits exactly on the
    #     true mesh surfaces, so heal's `surface_max_dist` cull deletes airborne
    #     Gaussians decidably, and its measured-contribution cull + opacity
    #     prune handle near-surface haze.
    # ^ THAT MEASUREMENT IS VOID. It was taken under the schedule bug: `reset_every`
    # was denominated in view-draws and divided by `batch`, so on those runs a reset
    # fired every ~375 optimizer steps of a ~2,000-step run. Nothing could recover
    # from that, and resets were blamed for what the schedule was doing. At the
    # reference 3000 STEPS inside a 30,000-step run there are ten resets with 3,000
    # steps of recovery each; a Gaussian visible in 10% of views gets ~300 updates
    # to re-earn its opacity, which is ample. (Recovery is governed by updates per
    # Gaussian, not by passes over the view set — the old note measured the wrong
    # quantity.)
    #
    # It is back ON because `DefaultStrategy` has NO opacity regularizer: the reset
    # IS the regularizer. Without it opacity only ever ratchets up, and it did —
    # measured on the 30k run, trained opacity was p50 0.997 / p90 1.000, so
    # transmittance after ONE Gaussian was 0.003. That makes the frontmost surfel
    # own each pixel outright while its ~5 overlapping neighbours keep near-init
    # (2.1x too bright) colour, which is both the blotchy shading and why the
    # trained DC luma was still 0.512 against a 0.285 reference after 30,000 steps.
    #
    # Fires inside the refine window via the in-loop reset in `_train_one`, because
    # the pinned gsplat 1.5.3 wheel's own trigger is dead code
    # (`step % reset_every == 0 & step > 0` parses as
    # `(step % reset_every == 0) and (0 > 0)`; 1.5.3 is still the newest release
    # everywhere and the fix is unreleased on main). 0 disables it; size pruning is
    # independent either way (see `prune_scale_start_iter`).
    reset_every: int = 3000
    # Opacity ceiling a reset clamps to — NOT `2·prune_opa`, which is what gsplat's
    # own trainer uses and what this loop used to pass. That coupling is a trap
    # here: `prune_opa` is 0.005 (set from the glass-stacking math), so the reset
    # would clamp every Gaussian to 0.01 and black the scene out for thousands of
    # steps. 0.1 is the value the reference actually lands on (its prune_opa is
    # 0.05). `reset_opa` CLAMPS rather than assigns, so anything already below this
    # — glass at ~0.012 — passes through untouched.
    reset_opa_value: float = 0.1
    # Step after which the STRATEGY's size prune goes live. gsplat gates
    # `prune_scale3d·scene_scale` (and `prune_scale2d`) behind
    # `step > strategy.reset_every` in `DefaultStrategy._prune_gs`, so the old
    # reset_every=0 sentinel (which passed `iterations + 1` to keep the reset from
    # firing) also removed the ONLY bound on Gaussian size for the entire run —
    # the measured result was disc radii up to 2.74 m in a 9.4 m room and 1.2% of
    # Gaussians carrying a third of all rendered opacity·area, i.e. the bloom.
    # `_train_one` passes THIS in that slot instead, so the bound holds whether or
    # not opacity resets are on. The reference's own gate lands at 3000/30k: a
    # warm-up long enough that a legitimately large surfel isn't culled before it
    # has been supervised.
    prune_scale_start_iter: int = 3000
    # Mean (non-absolute) 2D-gradient split threshold — gsplat's default value.
    # AbsGS (absgrad) is deliberately NOT used: in the pinned gsplat 1.5.3 the
    # 2DGS backward attaches `.absgrad` to `means2d`, but DefaultStrategy densifies
    # the 2DGS path off a SEPARATE `gradient_2dgs` tensor (`rendering.py` densify)
    # that never receives an `.absgrad` — so `absgrad=True` + key_for_gradient=
    # "gradient_2dgs" raises `'Tensor' object has no attribute 'absgrad'` in
    # `_update_state`. AbsGS is therefore a `means2d`-only feature, i.e. the 3DGS
    # path — where it IS enabled by default and this threshold is superseded by
    # `grow_grad2d_abs`. Under "2dgs" we stay on the mean-gradient criterion
    # (which is what trained every working run).
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    # SCREEN-SPACE split. `grow_scale3d` is scene-relative, so it asks "is this
    # Gaussian large compared to the scene?" — at scene_scale 12.8 the clone/split
    # boundary is 12.8 cm, which 95.9% of a room's surfels sit under. They
    # therefore only ever CLONE (exact duplicates at the same size), and the
    # measured 30k run came out COARSER than its init: median tangent radius 2.81
    # -> 5.28 cm, median projected 64 -> 83 px. Nothing in the loop can make a
    # Gaussian smaller, so detail finer than ~80 px is unrepresentable no matter
    # how long it trains.
    #
    # This criterion asks the question that matters instead — "is it large on
    # SCREEN?" — splitting (scale/1.6, two children) anything whose projected
    # radius exceeds `grow_scale2d` of the image's long side. gsplat gates it
    # behind `refine_scale2d_stop_iter`, 0 = OFF, which is why it never ran.
    #
    # `prune_scale2d` is deliberately neutralized at 1.0 (a radius larger than the
    # whole frame, so it never fires). The same gate arms BOTH, and the prune arm
    # at gsplat's 0.15 default would delete rather than subdivide: a 5 cm Gaussian
    # seen by a 0.12 m detail camera projects to ~0.30 normalized, so most of the
    # model would be pruned the moment this switched on. Split, don't delete.
    #
    # Enabling this multiplies the count (~2^k for k rounds down to threshold), so
    # the window ENDS well before `refine_stop_iter` (8k of the reference 30k, vs a
    # 15k refine stop) leaving 7k steps of settling, and the VRAM guard — which now
    # pauses growth reversibly instead of latching refinement off — is the backstop.
    #
    # MEASURED, same cell, same references, 30k steps: off -> 559,823 splats at a
    # 83 px median projected radius, 27.41 dB. On -> 1,536,708 at 22.8 px, 28.73 dB,
    # at the SAME 126 ms/step because smaller Gaussians touch fewer tiles (render
    # 47.3 -> 38.1 ms, densification bookkeeping 35.7 -> 11.2 ms). It also let
    # opacity and colour converge: opacity p50 0.997 -> 0.189, DC luma 0.512 ->
    # 0.347 against a 0.285 reference.
    refine_scale2d_stop_iter: int = 8000
    grow_scale2d: float = 0.05
    prune_scale2d: float = 1.0
    # Keep pruning below the opacity a CORRECT glass pane converges to — which is
    # NOT the authored GLASS_ALPHA. Stage 3 writes 0.065 onto every surfel of a
    # pane, but surfels overlap (radius = 0.9 × spacing), so a ray blends through
    # ~5-6 of them: measured stack depth 5.3 (hotel) / 6.1 (test-SH), rendering the
    # pane at 1−(1−0.065)^5.3 = 0.31 instead of 0.065. Per-surfel opacity for a
    # correct pane is therefore 1−(1−0.065)^(1/N) ≈ 0.012, and the old 0.03 floor
    # sat ABOVE that — a pane converging toward the right answer was pruned on the
    # way there, both by DefaultStrategy inside the refine window and again by
    # `_final_prune` in Stage 7. 0.005 (also the gsplat/INRIA default) leaves the
    # same ~0.8 logits of headroom below that target that 0.03 left below the init.
    # Low-opacity junk is culled better downstream regardless: Stage 7's
    # `compact_eps` measures each Gaussian's ACTUAL rendered blend weight, so it is
    # stack-aware — it drops a lone invisible splat while keeping a pane whose six
    # layers sum to something visible, a distinction a flat opacity threshold
    # cannot make.
    # (If `reset_every` > 0: gsplat's reset clamps to 2·prune_opa, so at 0.005 a
    # reset lands on 0.01 — far deeper than the reference's 0.1. Raise prune_opa or
    # lengthen the reset cadence before enabling resets.)
    prune_opa: float = 0.005
    # OPTIONAL Gaussian-count ceiling — None (the default) means UNCAPPED.
    # Densification is how the model earns detail, so a count ceiling is a direct
    # cap on quality, and neither thing it used to justify still holds: delivered
    # size is decided entirely by Stage 7's compaction (Stage 6 emits the raw
    # model), and VRAM is guarded properly by `vram_min_free_gb`, which reacts to
    # MEASURED free memory rather than guessing a count that cannot know the
    # scene's resolution, batch size or SH degree. At the old 2.5M it never fired
    # on the 40 GB A100 of the time (and the L40S since has more), so its only
    # real effect was deriving the tile budget —
    # which is now `_TILE_BUDGET_DEFAULT` / `tile_max` instead, so changing this
    # can no longer restructure a scene's tiles as a side effect.
    # Set an int only to bound per-step runtime on a small GPU. Exceeding it pauses
    # GROWTH (and growth RESUMES if pruning brings the count back under) while
    # pruning keeps running the whole time — see the growth guards in `_train_one`.
    cap_max: int | None = None
    # Final cleanup prune (once, before eval + export). Densification/pruning stop at
    # refine_stop (50% of iters), so opacity that drifts below prune_opa in the back
    # half — the low-opacity floaters stranded at silhouette/depth edges — otherwise
    # ships. This drops them, plus any Gaussian whose max tangent radius exceeds
    # prune_scale3d·scene_scale (runaway blobs; a safe no-op on well-behaved clouds,
    # whose largest surfel sits well under that). prune_scale3d=0 disables the scale
    # guard; final_prune=False disables the whole pass.
    final_prune: bool = True
    prune_scale3d: float = 0.1

    # COMPACTION (post-training cleanup; module docstring §COMPACTION). A measured
    # deletion (three composable signals) + a short heal of the survivors.
    compact: bool = True
    # (1) LOSSLESS: total rendered blend weight below which a Gaussian can't move
    # any pixel by half a display step — deleting it is imperceptible.
    compact_eps: float = 0.5 / 255.0
    # (2) SURFACE PRIOR (m): drop Gaussians farther than this from every Stage-3
    # surfel (opaque floaters / mis-seeded depth points open scenes grow). The
    # base surfel spacing clamps to ≤0.25 m, so 0.6 m cleanly separates airborne
    # junk from real surface; 0 disables. Raise for metre-scale-sampled huge
    # objects (skydome/terrain) whose legitimate surfels are sparse.
    surface_max_dist: float = 0.6
    # (3) BUDGET, quality-gated (the default): bisect the keep fraction and
    # accept the deepest cut whose measured PSNR drop (probe: prune -> short heal
    # -> eval on the seeded eval subset) stays within this many dB of the
    # post-cull baseline. Each scene finds its own safe cut; None disables the
    # search and falls back to `compact_keep_frac`. Cost ≈ compact_probes ×
    # (probe-heal + eval) per run/tile.
    compact_max_db_drop: float | None = 1.0
    compact_probes: int = 3               # bisection probes (granularity ~0.9/2^probes)
    compact_probe_heal_steps: int = 150   # short heal before each probe's eval
    # Fixed-fraction fallback (used only when compact_max_db_drop is None):
    # keep the top this-fraction by normalized contribution — SCENE-RELATIVE,
    # never an absolute cap. None = no budget cut at all.
    compact_keep_frac: float | None = None
    # Heal after deletion: fine-tune steps (no densification, means LR damped) so
    # survivors fill the deleted Gaussians' residual. 0 disables.
    compact_heal_steps: int = 500
    compact_heal_means_lr_frac: float = 0.1

    # Adaptive VRAM guard — the PRIMARY protection now that `cap_max` is off by
    # default, and the only one that reacts to what the GPU is actually doing. Freeze
    # densification (and pause depth-seeding) when free VRAM drops below this many
    # GB — after one empty_cache retry to release reusable cached blocks — so an
    # 8 GB card can't densify itself over the WDDM shared-memory cliff that
    # collapsed a real run to 0.05 it/s. 0 disables it (leaving NO growth guard
    # unless `cap_max` is set).
    # (`expandable_segments` would fight fragmentation on Linux/Modal but is a
    # no-op on Windows, so this free-margin freeze is the portable mechanism.)
    # A near-full floor, so it never triggers where VRAM is plentiful (the 48 GB
    # L40S) — no quality cost there — but on a small card it catches the cliff before a
    # densification growth step overshoots it.
    vram_min_free_gb: float = 1.5

    # Anti-aliasing — a Mip-Splatting-style 3D low-pass computed from the exact
    # cameras. Every `aa_every` steps, lower-bound each Gaussian's two in-plane
    # log-scales so it projects to ≥ `aa_min_scale_px` std from its NEAREST camera
    # (radius floor = px·d_nn/focal). A Gaussian the closest camera can't resolve
    # would alias — shimmer across the scale ladder's octaves under free-fly. The
    # floor scales with distance, so close-viewed detail and large surfaces are
    # untouched; only genuinely sub-pixel Gaussians are inflated.
    #
    # Under "3dgs" the floor applies to each Gaussian's two LARGEST axes rather
    # than to columns 0/1, so it guarantees screen footprint without inflating the
    # thin (normal) axis and undoing `flat_lambda`. It is the 3D half of
    # Mip-Splatting; `rasterize_mode="antialiased"` is the 2D half.
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
    #
    # OFF by default. A seed's disc faces the SEEDING CAMERA, not the surface, so
    # it is per-view private geometry: near-invisible edge-on from anywhere else,
    # it flatters the seeding view's PSNR and disintegrates under free-fly, and
    # PostShot has no analogue. It also misfires badly from a translucent init —
    # at `init_opa` = 0.1 nearly every foreground pixel reads as a "hole", so the
    # first passes dump `depth_densify_max` Gaussians at arbitrary surface points
    # (a measured 270k of 523k on one from-points run). With the surfels init the
    # surface is already covered and gradient-driven densification has the budget
    # to itself. Re-enable only for inits that genuinely start with holes.
    depth_densify: bool = False
    depth_densify_every: int = 500
    depth_densify_start: int = 500
    depth_densify_max: int = 20_000
    depth_densify_miss_alpha: float = 0.5
    depth_densify_depth_tol: float = 0.1
    depth_densify_opacity: float = 0.5
    depth_densify_scale_px: float = 1.0

    # Runtime. (The 2DGS path always renders non-packed — gsplat's packed 2DGS depth
    # path is broken, and non-packed is cheap at our per-cell counts anyway. The
    # 3DGS path can pack; see `packed`.)
    near_plane: float = 0.01
    far_plane: float = 1e10
    # Depth statistic the depth loss (and normals-from-depth) compares. "median"
    # (default) = the transmittance-0.5 crossing: it sits on the nearest OPAQUE
    # surface, which is exactly what the capture stores as depth GT (BLEND glass
    # doesn't write depth), so a correctly-reproduced glass pane at α ≈ 0.065 stays
    # below the crossing and contributes ZERO depth error — the pane is no longer
    # seen as a floater to delete. It's also cleaner at silhouettes. "expected" =
    # the alpha-weighted mean (ED): the one per-pixel term that sees a low-opacity
    # floater stranded in front of an opaque surface (L1 is camouflaged, the alpha
    # loss is saturated) — BUT it reads a real transmissive pane as that same
    # floater and drags its opacity to zero (glass-over-wall gets a residual error
    # even when perfectly reconstructed), which is why it razed the glass panes.
    # This pipeline doesn't need expected's floater-hunting: Stage 7 culls airborne
    # Gaussians GEOMETRICALLY against the exact Stage-3 surfels (surface_max_dist)
    # plus a measured-contribution + opacity prune. Choose "expected" only for a
    # run whose geometric post-cull is disabled and that has no transmissive
    # surfaces to protect.
    #
    # 2DGS-ONLY CHOICE. The 3DGS rasterizer renders no median depth (gsplat's
    # `rasterization` offers accumulated "D" and expected "ED" and nothing else),
    # so `resolved_depth_mode` forces "expected" there. Because expected depth
    # cannot tell a real pane from a floater in front of a wall, the depth term
    # then EXCLUDES pixels where the init cloud's own two-layer expected depth
    # disagrees with the stored opaque plane (`_ED_GLASS_TOL_FRAC`) — glass is left
    # to the photometric terms and to `depth_expected_lambda`, which is the one
    # depth target already formulated in expected-depth space.
    depth_mode: str = "median"
    seed: int = 0
    eval_max_views: int = 128          # cap final-metric renders (plans can have thousands of views)
    log_every: int = 50                # emit a progress line every N steps
    # Views averaged per optimizer step (gsplat `batch_size`). This MULTIPLIES the
    # work — `iterations` Adam updates happen either way — so it buys lower
    # gradient noise and better GPU utilization, never a shorter run. Every LR is
    # scaled by sqrt(batch) at runtime, the reference's variance-matching rule.
    # 1 is exact reference / PostShot parity; raise it only with VRAM to spare,
    # since rasterization activation memory (esp. the tile-intersection buffer)
    # grows ~linearly with batch and can OOM in a single step the free-margin
    # guard can't catch (it only throttles growth).
    batch: int = 1
    prefetch: bool = True              # decode/stack the next batch on a background thread (hide disk I/O)

    # Resumable checkpoints: every `ckpt_every` steps write the full training state
    # (params + per-param Adam + densification strategy + step) to `splat/ckpt/`,
    # keeping the most recent `ckpt_keep`. An interrupted run resumes from the latest
    # and continues to `iterations`; they're deleted once trained.ply is written.
    ckpt_every: int = 2000             # 0 disables checkpointing
    ckpt_keep: int = 2

    # Tiled training (module docstring §TILED TRAINING). Seeds beyond the budget
    # train as ground-plane tiles, each with the full budget of densification
    # headroom, merged by core ownership. None → `_TILE_BUDGET_DEFAULT`; 0 disables
    # tiling (a huge seed then trains as one over-budget run, as before).
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

    def __post_init__(self) -> None:
        """Reject unknown enum values at construction. These arrive from JSON job
        specs (`TrainParams(**spec["train"])`), and a typo like "3DGS" would
        otherwise fall through every `== "3dgs"` test and silently train the other
        representation for hours."""
        if self.representation not in REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {REPRESENTATIONS}, got {self.representation!r}"
            )
        if self.rasterize_mode not in ("classic", "antialiased"):
            raise ValueError(
                f"rasterize_mode must be 'classic' or 'antialiased', got {self.rasterize_mode!r}"
            )
        if self.depth_mode not in ("median", "expected"):
            raise ValueError(
                f"depth_mode must be 'median' or 'expected', got {self.depth_mode!r}"
            )

    @property
    def is_3dgs(self) -> bool:
        return self.representation == "3dgs"

    @property
    def n_scales(self) -> int:
        """Scale columns in the exported `.ply`: 3 real axes for 3DGS, the two
        tangent radii for 2DGS (whose third axis is a training-only decoy)."""
        return 3 if self.is_3dgs else 2

    @property
    def resolved_absgrad(self) -> bool:
        """AbsGS densification. Forced OFF on 2DGS (gsplat never writes `.absgrad`
        on that path's densification tensor — it would raise mid-run); defaults ON
        for 3DGS, where it is the biggest available quality win."""
        if not self.is_3dgs:
            return False
        return True if self.absgrad is None else bool(self.absgrad)

    @property
    def resolved_grow_grad2d(self) -> float:
        """Split threshold matched to the gradient statistic in use — absolute
        gradients are larger than signed means, so they need a higher bar."""
        return self.grow_grad2d_abs if self.resolved_absgrad else self.grow_grad2d

    @property
    def resolved_depth_mode(self) -> str:
        """The depth statistic the loss compares. 3DGS has no median depth, so the
        expected-depth channel is the only option there (see `depth_mode`)."""
        return "expected" if self.is_3dgs else self.depth_mode

    @property
    def resolved_glass_guard(self) -> bool:
        """Whether the transmissive-pixel exclusion actually engages: only the
        EXPECTED depth statistic needs it, and only if there is a depth term to
        exclude pixels from (see `glass_guard`)."""
        return (
            bool(self.glass_guard)
            and self.resolved_depth_mode == "expected"
            and self.depth_lambda > 0.0
        )

    @property
    def resolved_packed(self) -> bool:
        """gsplat's packed projection — 3DGS only (its packed 2DGS RGB+ED path is
        broken upstream)."""
        return bool(self.packed) and self.is_3dgs

    @property
    def resolved_thickness_frac(self) -> float:
        """Third-scale fraction for a seed built from two tangent radii: real
        thickness on the 3DGS path, splitter decoy on the 2DGS one."""
        return self.init_thickness_frac if self.is_3dgs else _THIN_AXIS_FRAC

    @property
    def resolved_refine_stop(self) -> int:
        return self.refine_stop_iter if self.refine_stop_iter is not None else int(self.iterations * 0.5)

    @property
    def resolved_tile_budget(self) -> int:
        """Seed count above which the run tiles (0 = tiling disabled)."""
        if self.tile_max is not None:
            return max(int(self.tile_max), 0)
        return _TILE_BUDGET_DEFAULT

    def resolve_schedule(self, n_views: int) -> "TrainParams":
        """Concrete per-run (or per-TILE) schedule for `n_views` reference images,
        in optimizer-STEP units.

        With `epochs` unset this is the IDENTITY: `iterations` is already the step
        count and every cadence is already written against it. With `epochs` set
        the run is sized as whole passes over the view set
        (`epochs × n_views / batch` steps) and every cadence is rescaled by
        `steps / iterations` — gsplat's `adjust_steps` — so a short run is a
        proportionally compressed full run rather than a truncated one. Returns a
        NEW TrainParams, so every downstream consumer (the loop,
        `resolved_refine_stop`, the LR schedule) reads step counts unchanged.
        Called per TILE with the tile's own view count, so each tile trains
        `epochs` passes over the views that supervise it."""
        from dataclasses import replace

        if self.epochs is None:
            return self

        b = max(int(self.batch), 1)
        base = max(1, int(self.iterations))
        steps = max(_MIN_STEPS, round(self.epochs * max(n_views, 1) / b))
        factor = steps / base

        def scaled(v: int, floor: int = 1) -> int:
            return max(floor, round(v * factor))

        return replace(
            self,
            iterations=steps,
            refine_start_iter=scaled(self.refine_start_iter),
            refine_every=scaled(self.refine_every),
            refine_stop_iter=(
                None if self.refine_stop_iter is None else scaled(self.refine_stop_iter)
            ),
            prune_scale_start_iter=scaled(self.prune_scale_start_iter, 0),
            refine_scale2d_stop_iter=(
                0 if self.refine_scale2d_stop_iter <= 0
                else scaled(self.refine_scale2d_stop_iter)
            ),
            depth_densify_every=scaled(self.depth_densify_every),
            depth_densify_start=scaled(self.depth_densify_start),
            depth_start_iter=scaled(self.depth_start_iter, 0),
            normal_start_iter=scaled(self.normal_start_iter, 0),
            dist_start_iter=scaled(self.dist_start_iter, 0),
            aa_every=scaled(self.aa_every),
            sh_degree_interval=scaled(self.sh_degree_interval),
            reset_every=(0 if self.reset_every == 0 else scaled(self.reset_every)),
            ckpt_every=(0 if self.ckpt_every == 0 else scaled(self.ckpt_every)),
        )

    def as_summary(self) -> dict[str, Any]:
        """The knobs that define a run — recorded in `status.json` AND hashed into
        the Modal stage-6/7 idempotency signature, so anything omitted here can
        silently skip a re-run that should have happened. Representation-specific
        knobs are reported only for the representation they act on (RESOLVED, i.e.
        what the loop will actually execute), so flipping an inert one can't
        invalidate an unrelated cell's cache."""
        s: dict[str, Any] = {
            "representation": self.representation,
            "iterations": self.iterations,
            "epochs": self.epochs,
            "batch": self.batch,
            "init": self.init,
            "init_opa_max": self.init_opa_max,
            "init_opa": self.init_opa,
            "init_scale": self.init_scale,
            "ssim_lambda": self.ssim_lambda,
            "alpha_lambda": self.alpha_lambda,
            "depth_lambda": self.depth_lambda,
            "depth_expected_lambda": self.depth_expected_lambda,
            "depth_start_iter": self.depth_start_iter,
            "depth_mode": self.resolved_depth_mode,
            "glass_guard": self.resolved_glass_guard,
            "alpha_gate": self.alpha_gate,
            "normal_start_iter": self.normal_start_iter,
            "refine": self.refine,
            "refine_stop_iter": self.resolved_refine_stop,
            "reset_every": self.reset_every,
            "reset_opa_value": self.reset_opa_value,
            "prune_scale_start_iter": self.prune_scale_start_iter,
            "refine_scale2d_stop_iter": self.refine_scale2d_stop_iter,
            "grow_scale2d": self.grow_scale2d,
            "prune_scale2d": self.prune_scale2d,
            "grow_grad2d": self.resolved_grow_grad2d,
            "absgrad": self.resolved_absgrad,
            "prune_opa": self.prune_opa,
            "final_prune": self.final_prune,
            "prune_scale3d": self.prune_scale3d,
            "compact": self.compact,
            "compact_eps": self.compact_eps,
            "surface_max_dist": self.surface_max_dist,
            "compact_max_db_drop": self.compact_max_db_drop,
            "compact_keep_frac": self.compact_keep_frac,
            "compact_heal_steps": self.compact_heal_steps,
            "cap_max": self.cap_max,
            "vram_min_free_gb": self.vram_min_free_gb,
            "antialias": self.antialias,
            "aa_min_scale_px": self.aa_min_scale_px,
            "sh_degree": self.sh_degree,
            "sh_degree_interval": self.sh_degree_interval,
            "shN_lr": self.shN_lr,
            "depth_densify": self.depth_densify,
            "depth_densify_every": self.depth_densify_every,
            "depth_densify_max": self.depth_densify_max,
            "ckpt_every": self.ckpt_every,
            "tile_budget": self.resolved_tile_budget,
            "tile_margin_frac": self.tile_margin_frac,
            "lod_levels": self.lod_levels,
        }
        if self.is_3dgs:
            s.update({
                "rasterize_mode": self.rasterize_mode,
                "packed": self.resolved_packed,
                "init_thickness_frac": self.resolved_thickness_frac,
                "flat_lambda": self.flat_lambda,
                "flat_max": self.flat_max,
                "aniso_lambda": self.aniso_lambda,
                "aniso_max": self.aniso_max,
            })
        else:
            s.update({
                "normal_lambda": self.normal_lambda,
                "dist_lambda": self.dist_lambda,
                "dist_start_iter": self.dist_start_iter,
            })
        return s


# --- torch-free layer (PLY / pose / image IO — runs anywhere) ------------------


def _load_cloud(path: Path, *, thickness_frac: float = _THIN_AXIS_FRAC) -> dict[str, np.ndarray]:
    """Parse a binary-little-endian float `.ply` into gsplat init arrays: means
    [N,3], quats [N,4] (wxyz), opacities [N] (logit), sh0 [N,1,3] (SH DC colour
    coeffs), shN [N,15,3] (the degree-1..3 view-dependent coeffs decoded from
    `f_rest_*`, always degree-3-sized; zeros for a degree-0 cloud like the Stage-3
    init, so the SH-degree warmup grows them from flat), scales [N,3] (log).

    Both rasterizers take three scales, so a 2-scale (2DGS) `.ply` gets a third
    synthesized at `thickness_frac` of its smaller tangent radius. The caller owns
    that fraction because its MEANING depends on the representation being trained:
    the 2DGS default (`_THIN_AXIS_FRAC`) is an unrendered splitter decoy, while a
    3DGS run passes `TrainParams.resolved_thickness_frac` and gets real geometry."""
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
    # Higher-order SH → shN [N,15,3] (degree-3 storage). A trained/degree-3 .ply
    # carries `f_rest_0..44` in INRIA channel-major order (all K coeffs of R, then
    # G, then B): reshape [N,3,K] then transpose to gsplat's coeff-major [N,K,3].
    # A degree-0 cloud (the Stage-3 init) has none → shN stays zero.
    rest = sorted(
        (n for n in col if n.startswith("f_rest_")),
        key=lambda s: int(s.rsplit("_", 1)[-1]),
    )
    shN = np.zeros((count, 15, 3), dtype=np.float32)
    if rest:
        flat = np.stack([col[n] for n in rest], axis=1).astype(np.float32)  # [N, 3K]
        k = min(flat.shape[1] // 3, 15)
        shN[:, :k, :] = flat[:, : 3 * k].reshape(count, 3, k).transpose(0, 2, 1)
    if "scale_2" in col:
        scales = stack("scale_0", "scale_1", "scale_2")
    else:
        # 2-scale cloud: synthesize the third log-scale off the smaller tangent
        # radius. Stage 3 aligns the quaternion's local +Z to the surface normal, so
        # this axis IS the thickness — which is why the same construction serves as
        # a 2DGS splitter decoy at 1% and as a real 3DGS shell at 10%.
        two = stack("scale_0", "scale_1")
        third = (two.min(axis=1, keepdims=True) + np.log(thickness_frac)).astype(np.float32)
        scales = np.concatenate([two, third], axis=1)
    return {
        "means": means,
        "quats": quats,
        "opacities": opacities,
        "sh0": sh0,
        "shN": shN,
        "scales": scales,
    }


def _ply_representation(path: Path) -> str:
    """Which primitive a splat `.ply` on disk holds, read from its scale columns
    alone: "3dgs" when it carries a `scale_2` property, else "2dgs".

    Header-only (the body is never touched), so it is cheap enough to guard every
    stage that consumes a model someone ELSE trained. That guard matters: Stage 7
    and the compaction runners must render with the same rasterizer the file was
    optimized under — heal a 2-scale surfel model through the 3DGS rasterizer and
    every disk becomes an invisible sliver, so the measured contribution is noise
    and the heal optimizes against it."""
    with Path(path).open("rb") as f:
        head = f.read(1 << 16)
    cut = head.find(b"end_header")
    if cut < 0:
        raise ValueError(f"{path}: not a PLY (no end_header in the first 64 KiB)")
    return "3dgs" if b"scale_2" in head[:cut] else "2dgs"


# --- COLMAP text model (the Postshot-style input: points3D + cameras + images) --
# The model written by `splat.colmap.export_colmap` / `splat_to_colmap.py`: one
# shared PINHOLE camera, per-image world-to-camera poses, an xyz+rgb point cloud,
# and the reference images as RGBA (RGB + coverage alpha; depth dropped — the
# trainer reads only RGB, the alpha is kept for Postshot masking). This is
# now the ONLY Stage-6 input; the trainer inits from the point cloud exactly like
# gsplat's simple_trainer_2dgs.

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    """COLMAP `(qw,qx,qy,qz)` world-to-camera quaternion → 3×3 rotation (the exact
    inverse of `splat.colmap.rotmat2qvec`)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _read_colmap_cameras(path: Path) -> tuple[dict[int, np.ndarray], int, int]:
    """Parse `cameras.txt` → ({camera_id: K [3,3]}, width, height).

    A model can declare MANY cameras, and ours does: Stage 4 derives FOV per
    camera from the distance to what each one frames, so `export_colmap` dedupes
    the plan's intrinsics into one record per distinct focal length and every
    image line names its own. Reading only the first record — which is what this
    did while one shared `K` was the contract — would silently render most frames
    at the wrong focal length. All records must share a resolution, since the
    trainer rasterizes one width/height per batch."""
    cams: dict[int, np.ndarray] = {}
    size: tuple[int, int] | None = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cid, model, w, h = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        p = [float(v) for v in parts[4:]]
        if model in ("PINHOLE", "OPENCV"):
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            fx, fy, cx, cy = p[0], p[0], p[1], p[2]
        else:
            raise ValueError(f"{path}: unsupported COLMAP camera model {model!r} (need PINHOLE)")
        if size is None:
            size = (w, h)
        elif size != (w, h):
            raise ValueError(
                f"{path}: cameras disagree on resolution ({size} vs {(w, h)}); the "
                "trainer rasterizes one frame size per batch"
            )
        cams[cid] = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
    if not cams or size is None:
        raise ValueError(f"{path}: no camera record")
    return cams, size[0], size[1]


def _read_colmap_images(path: Path) -> list[tuple[str, np.ndarray, int]]:
    """Parse `images.txt` → [(image_name, c2w [4,4], camera_id), …] in file order.
    Each image's pose line is `ID QW QX QY QZ TX TY TZ CAMERA_ID NAME`
    (world-to-camera); the optional POINTS2D line (empty in our export) is
    identified by NOT ending in an image name and skipped, so both our export and
    a standard model parse. The CAMERA_ID is carried because a model may declare
    several intrinsics and each image names the one it was shot with."""
    out: list[tuple[str, np.ndarray, int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 10 or not parts[9].lower().endswith(_IMAGE_EXTS):
            continue  # a POINTS2D line, not a pose line
        q = np.array(parts[1:5], dtype=np.float64)
        t = np.array(parts[5:8], dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = _qvec_to_rotmat(q)
        w2c[:3, 3] = t
        out.append((parts[9], np.linalg.inv(w2c), int(parts[8])))
    return out


def _read_colmap_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse `points3D.txt` → (xyz [N,3] float64, rgb [N,3] uint8). Only the first
    seven fields per line (ID X Y Z R G B) are read; the variable-length TRACK tail
    is left unsplit (`maxsplit=7`) so multi-million-point clouds parse quickly."""
    xyz: list[tuple[float, float, float]] = []
    rgb: list[tuple[int, int, int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line or line[0] == "#":
            continue
        p = line.split(maxsplit=7)
        if len(p) < 7:
            continue
        xyz.append((float(p[1]), float(p[2]), float(p[3])))
        rgb.append((int(p[4]), int(p[5]), int(p[6])))
    if not xyz:
        raise ValueError(f"{path}: no 3D points")
    return np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.uint8)


def _init_from_points(xyz: np.ndarray, rgb: np.ndarray, params: TrainParams) -> dict[str, np.ndarray]:
    """gsplat `create_splats_with_optimizers`-style init from a bare point cloud
    (xyz + rgb) — the same recipe simple_trainer uses for an SfM cloud, so a
    Postshot-style input trains identically. means = xyz; each scale =
    log(mean-distance-to-3-nearest-neighbours × init_scale) — isotropic over all
    three axes for 3DGS, over the two tangent axes for 2DGS (whose third stays the
    unrendered decoy); quats random (rasterizer normalizes); opacity =
    logit(init_opa); colour = the points3D RGB mapped to the SH0 DC term
    ((rgb−0.5)/C0, matching Stage 3)."""
    from scipy.spatial import cKDTree

    means = np.ascontiguousarray(xyz, dtype=np.float32)
    n = int(means.shape[0])
    k = min(4, n)
    if k >= 2:
        dists, _ = cKDTree(means).query(means, k=k, workers=-1)
        dist_avg = np.sqrt((dists[:, 1:] ** 2).mean(axis=1))
    else:
        dist_avg = np.full(n, 0.01, dtype=np.float64)
    log_scale = np.log(np.maximum(dist_avg * params.init_scale, 1e-9)).astype(np.float32)
    if params.is_3dgs:
        # ISOTROPIC on all three axes — gsplat's own from-points recipe, so this
        # really is the reference baseline it exists to be. A 3D Gaussian has a
        # thickness to learn, and starting it at the KNN radius lets the images
        # decide it; there is no plane to keep splits inside.
        scales = np.repeat(log_scale[:, None], 3, axis=1)
    else:
        # Two in-plane axes + a TINY third, exactly as `_load_cloud` synthesizes for
        # a 2-scale cloud: the 2DGS rasterizer ignores the third and the strategy's
        # grow/prune thresholds take max() over the three (unchanged, since these
        # are isotropic), but DefaultStrategy's split() displaces children along ALL
        # three scaled axes — an equal third would eject every child a full disc
        # radius off its own plane.
        third = log_scale + np.float32(np.log(_THIN_AXIS_FRAC))
        scales = np.stack([log_scale, log_scale, third], axis=1)

    rng = np.random.default_rng(params.seed)
    quats = rng.standard_normal((n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12

    a = float(np.clip(params.init_opa, 1e-3, 1.0 - 1e-3))
    opacities = np.full(n, float(np.log(a / (1.0 - a))), dtype=np.float32)

    sh0 = (((rgb.astype(np.float32) / 255.0) - 0.5) / _SH_C0)[:, None, :].astype(np.float32)
    # Degree-3-sized higher-order SH (zeros): a from-points init is flat, and the
    # SH-degree warmup grows these from 0 exactly as it does for the degree-0
    # surfel cloud — keeps the init dict shape identical for `_train_one`.
    shN = np.zeros((n, 15, 3), dtype=np.float32)
    return {"means": means, "scales": scales, "quats": quats, "opacities": opacities, "sh0": sh0, "shN": shN}


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
    scales: np.ndarray,
    out_path: Path,
    sh_rest: np.ndarray | None = None,
) -> None:
    """Write the trained model as a Stage-3-compatible splat `.ply`, with as many
    `scale_*` columns as `scales` has: TWO for a 2DGS surfel model (the tangent
    radii; its third training axis is an unrendered decoy and must not ship), THREE
    for a 3DGS one. That count is the ONLY on-disk difference between the two
    representations, and every consumer already keys off it — `_load_cloud` and
    `splat/quantize.py` read whatever is present, and the client's SOG/ksplat
    encoders skip their synthetic-third-scale flatten when `scale_2` is there.

    `nx,ny,nz` stay the local +Z axis of the quaternion, which is the surface
    normal Stage 3 aligned and (for 3DGS) the thin axis at init. They are
    decorative — no viewer reads them and `quantize.py` drops and recomputes them
    from the quaternion on decode — so the convention is kept identical across
    representations to keep that round-trip exact.

    Values are stored raw (opacity as logit, scales as log, colour as
    `f_dc`) exactly as the viewers/Stage 7 expect. The DC colour (`f_dc`) is always
    written; when `sh_rest` (gsplat coeff-major degree-1..3 coeffs [N,K,3]) is
    given and non-empty its K·3 `f_rest_*` view-dependent coefficients are written
    too, in INRIA channel-major order (all K coeffs of R, then G, then B) between
    `f_dc` and `opacity`, so the SOG encoder / PlayCanvas viewer reconstruct
    view-dependent colour. `sh_rest=None` writes a flat degree-0 model (e.g. the
    LOD ladder), readable identically by every consumer."""
    n = means.shape[0]
    normals = _quats_to_normals(quats)
    cols = [
        means[:, 0], means[:, 1], means[:, 2],
        normals[:, 0], normals[:, 1], normals[:, 2],
        f_dc[:, 0], f_dc[:, 1], f_dc[:, 2],
    ]
    rest_props = ""
    if sh_rest is not None and np.asarray(sh_rest).shape[1] > 0:
        rest = np.ascontiguousarray(
            np.asarray(sh_rest, dtype=np.float32).transpose(0, 2, 1).reshape(n, -1)
        )
        cols.extend(rest[:, j] for j in range(rest.shape[1]))
        rest_props = "".join(f"property float f_rest_{j}\n" for j in range(rest.shape[1]))
    n_scales = int(np.asarray(scales).shape[1])
    cols.append(opacity)
    cols.extend(scales[:, j] for j in range(n_scales))
    cols += [quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]]
    data = np.stack(cols, axis=1).astype("<f4")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n" "property float y\n" "property float z\n"
        "property float nx\n" "property float ny\n" "property float nz\n"
        "property float f_dc_0\n" "property float f_dc_1\n" "property float f_dc_2\n"
        + rest_props
        + "property float opacity\n"
        + "".join(f"property float scale_{j}\n" for j in range(n_scales))
        + "property float rot_0\n" "property float rot_1\n"
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


def _rasterizer(params: TrainParams):  # noqa: ANN202 - the gsplat entrypoint for this representation
    """The gsplat rasterizer this run trains through — `rasterization` for 3DGS,
    `rasterization_2dgs` for surfels. Imported lazily (gsplat is CUDA-only, and the
    server imports this module for the route + the torch-free IO) and passed down
    explicitly rather than looked up per call, so every helper renders through the
    same operator the loop does. Callers go through `_render_batch`, which
    normalizes the two different return shapes."""
    if params.is_3dgs:
        from gsplat import rasterization

        return rasterization
    from gsplat import rasterization_2dgs

    return rasterization_2dgs


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


def _render_inputs(torch, splats, sh_degree):  # noqa: ANN001
    """SH coefficients [N,16,3] + the ACTIVE `sh_degree`, for either rasterizer.
    Concatenates the DC term (`sh0` [N,1,3]) with the higher-order bands (`shN`
    [N,15,3]) so gsplat evaluates colour as f(view direction) up to the active
    degree — the SH-degree warmup passes a degree below the stored 3 early on, and
    gsplat uses only the first (degree+1)² bands (higher bands keep zero gradient
    until unlocked). On the 2DGS path this must be rendered NON-PACKED: gsplat's
    packed SH branch mis-broadcasts and its precomputed-colour path never gathers
    colours to the visible subset, so both crash on RGB+ED once a view sees only
    part of the cloud (nnz < N). The 3DGS path has no such bug, which is why
    `TrainParams.packed` is available there. gsplat applies the +0.5 offset,
    matching Stage-3's f_dc."""
    return torch.cat([splats["sh0"], splats["shN"]], dim=1), int(sh_degree)


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
        vms, ks, rgbs, alphas, depths = [], [], [], [], []
        for _ in range(batch):
            v = views[next(idx_gen)]
            rgb, alpha, d = _view_arrays(v)
            vms.append(v["viewmat"])
            # Intrinsics travel WITH the view, not beside the batch: Stage 4
            # derives each camera's FOV from the distance to what it frames, so
            # the focal length varies frame to frame and one shared K would be
            # wrong for most of them.
            ks.append(v["K"])
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

        return (
            _fin(torch.stack(vms)), _fin(torch.stack(ks)),
            _fin(torch.stack(rgbs)), _fin(torch.stack(alphas)), _fin(depth),
        )

    def to_dev(b) -> tuple:  # noqa: ANN001
        vm, ks, rgb, alpha, depth = b
        return (
            vm.to(device, non_blocking=True),
            ks.to(device, non_blocking=True),
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
    but `new` omits is zero-filled), extending each Adam optimizer's
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
    # `radii` only exists when the screen-space criterion is armed
    # (`refine_scale2d_stop_iter` > 0); it must grow with the others or the
    # strategy indexes a stale-length tensor on the next refine.
    for key in ("grad2d", "count", "radii"):
        v = strat_state.get(key)
        if torch.is_tensor(v):
            strat_state[key] = torch.cat([v, torch.zeros(m, device=v.device, dtype=v.dtype)], dim=0)


def _aa_scale_floor(torch, splats, cam_tree, focal, aa_min_px, tangent_only=True):  # noqa: ANN001
    """Lower-bound each Gaussian's FOOTPRINT log-scales so its projected std is
    ≥ `aa_min_px` from the NEAREST camera (radius floor = aa_min_px·d_nn/focal). Only
    inflates Gaussians a close camera would render sub-pixel; larger/close-viewed
    ones are untouched (the floor shrinks with camera distance). Returns the median
    floor radius in metres, for logging.

    `tangent_only` (2DGS) floors columns 0 and 1, the two axes that rasterizer
    renders. For 3DGS it floors each Gaussian's two LARGEST axes instead, whichever
    columns those are: those are what set its screen footprint, and leaving the
    smallest alone is what keeps this from silently inflating thin surfaces into
    blobs and undoing `flat_lambda`."""
    means_np = splats["means"].detach().cpu().numpy()
    d_nn = np.asarray(cam_tree.query(means_np, k=1, workers=-1)[0], dtype=np.float64)
    floor_r = (aa_min_px * np.maximum(d_nn, 1e-6) / max(float(focal), 1e-6)).astype(np.float32)
    log_floor = torch.from_numpy(np.log(np.maximum(floor_r, 1e-9))).to(splats["scales"].device)
    with torch.no_grad():
        s = splats["scales"]
        if tangent_only:
            s[:, 0] = torch.maximum(s[:, 0], log_floor)
            s[:, 1] = torch.maximum(s[:, 1], log_floor)
        else:
            idx = torch.topk(s, 2, dim=1).indices
            s.scatter_(1, idx, torch.maximum(s.gather(1, idx), log_floor[:, None]))
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
    third = log_r + float(np.log(params.resolved_thickness_frac))
    scales = torch.stack([log_r, log_r, third], dim=1).float()
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
        """Cache key for per-tile results: any change to the cloud, the grid, the
        training length, the SH degree or the REPRESENTATION invalidates cached
        tiles (the SH degree so a degree-0 cache never merges into a degree-3 run
        without the shN column; the representation so 2DGS tiles — whose third
        scale is an unrendered decoy — can never be merged into a 3DGS model,
        where that column is real thickness)."""
        return (
            f"{n_init}:{self.k[0]}x{self.k[1]}:{self.margin:.3f}"
            f":{params.iterations}:{params.resolved_tile_budget}:sh{params.sh_degree}"
            f":{params.representation}"
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


def _unproject_depth(torch, gt_depth, viewmats, Ks, px, py):  # noqa: ANN001
    """Reference depth → world points [B,H,W,3] (OpenCV pinhole; `px`/`py` are the
    precomputed pixel-centre grids). Zero-depth (background) pixels land at the
    camera centre — callers gate on depth > 0. `Ks` is PER VIEW [B,3,3]: focal
    length varies frame to frame (Stage 4 derives FOV per camera), so unprojecting
    a batch through one shared K would place most of it at the wrong scale."""
    d = gt_depth[..., 0]
    fx = Ks[:, 0, 0][:, None, None]
    fy = Ks[:, 1, 1][:, None, None]
    cx = Ks[:, 0, 2][:, None, None]
    cy = Ks[:, 1, 2][:, None, None]
    x = (px[None] - cx) / fx * d
    y = (py[None] - cy) / fy * d
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

    Each tile resolves its OWN schedule against the views assigned to it
    (`TrainParams.resolve_schedule`), so an `epochs`-sized run gives a tile that
    sees a twentieth of the plan a twentieth of the steps — total wall clock stays
    a few × a single run rather than n_tiles × — while every tile keeps full
    densification headroom. An `iterations`-sized run gives every tile the full
    step count, since that is what the field means. All step-derived schedules
    (refine window, LR decay, SH warm-up) ride the resolved count automatically."""
    import gc
    import shutil

    K_np = K.detach().cpu().numpy().astype(np.float64)
    owner = grid.owner(init["means"])
    core_counts = np.bincount(owner, minlength=grid.n_tiles)
    live = [t for t in range(grid.n_tiles) if core_counts[t] > 0]
    n_init = int(init["means"].shape[0])

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
            info: dict[str, Any] = {
                "tile": t, "init": int(sel.sum()), "final": int(core.sum()),
                "seeded": 0, "pruned": 0, "compacted": 0, "views": 0, "iters": 0,
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


def _lod_aggregate(
    arrays: dict[str, np.ndarray], voxel: float, n_scales: int = 2
) -> dict[str, np.ndarray]:
    """One LOD octave: Gaussians sharing a `voxel` collapse to ONE whose first two
    moments match the cluster's opacity·area-weighted mixture — mean and covariance
    (each primitive's R·diag(s²)·Rᵀ plus the spread of the means) — with the frame
    recovered by eigendecomposition. Colour is the same weighted mean, and opacity
    preserves the cluster's opacity·cross-section within the new primitive (a fully
    covered patch stays opaque; sparse glass stays translucent). Sub-pixel detail
    thereby collapses to its prefiltered average — the same reason mip maps beat
    point-sampling minified textures.

    `n_scales` is the representation's axis count. At 2 the input covariance is
    rank-2 (a disk: `diag(s0², s1², 0)`) and the output carries two radii plus the
    decoy third — the surfel behaviour, unchanged. At 3 all three axes participate
    in both directions, so an aggregate is as thick as the patch it replaces is
    curved, which is exactly the volumetric average a 3D Gaussian should hold.
    Either way the WEIGHT is opacity × the CROSS-SECTION (the two largest axes),
    since that is what a pixel sees and what the new opacity has to conserve."""
    means = arrays["means"].astype(np.float64)
    q = arrays["quats"].astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    s = np.exp(arrays["scales"].astype(np.float64)[:, :max(n_scales, 2)])
    alpha = 1.0 / (1.0 + np.exp(-arrays["opacities"].astype(np.float64)))
    col = arrays["sh0"].reshape(-1, 3).astype(np.float64)

    ids = np.floor((means - means.min(axis=0, keepdims=True)) / voxel).astype(np.int64)
    _, inv = np.unique(ids, axis=0, return_inverse=True)
    m = int(inv.max()) + 1

    # opacity·cross-section: the two largest axes (for a disk, its area).
    two_largest = np.sort(s, axis=1)[:, ::-1][:, :2] if s.shape[1] > 2 else s
    w = np.maximum(alpha * two_largest[:, 0] * two_largest[:, 1], 1e-12)
    wsum = np.bincount(inv, weights=w, minlength=m)

    def wmean(v: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.bincount(inv, weights=w * v[:, j], minlength=m) for j in range(v.shape[1])],
            axis=1,
        ) / wsum[:, None]

    mu = wmean(means)
    color = wmean(col)

    # Per-Gaussian covariance R·diag(s²)·Rᵀ from the (wxyz) quats — the unused
    # third eigenvalue stays 0 for a two-scale disk, giving the rank-2 surfel form.
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
    lam[:, : s.shape[1]] = s ** 2
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
    if n_scales >= 3:
        # The cluster covariance already carries the patch's out-of-plane spread, so
        # the smallest eigenvalue IS the aggregate's thickness — no decoy needed.
        r3 = np.sqrt(np.maximum(evals[:, 0], 1e-12))
        third = np.log(r3)[:, None]
    else:
        third = log_r.min(axis=1, keepdims=True) + np.log(_THIN_AXIS_FRAC)
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
    reaches `lod_min_count`. Same layout and representation as trained.ply, so
    every existing viewer/compressor reads the levels unchanged (SH0 — the ladder
    is unlit like the base). Stale lod files from earlier runs are removed first."""
    for old in out_path.parent.glob(f"{out_path.stem}.lod*.ply"):
        old.unlink(missing_ok=True)
    if params.lod_levels <= 0:
        return None
    out: list[dict[str, Any]] = []
    cur = arrays
    # 2× the median disk radius ≈ one 2×2-neighbour cluster per voxel on a
    # surface — the ~4×-per-octave reduction the ladder documents.
    base_voxel = 2.0 * float(np.median(np.exp(arrays["scales"]).max(axis=1)))
    for level in range(1, params.lod_levels + 1):
        if int(cur["means"].shape[0]) <= params.lod_min_count:
            break
        if progress is not None:
            progress(1, 1, f"LOD {level}: aggregating {int(cur['means'].shape[0])} splats")
        cur = _lod_aggregate(cur, base_voxel * (2.0 ** (level - 1)), params.n_scales)
        path = out_path.with_name(f"{out_path.stem}.lod{level}.ply")
        quats = cur["quats"] / (np.linalg.norm(cur["quats"], axis=1, keepdims=True) + 1e-12)
        _encode_trained_ply(
            cur["means"], quats.astype(np.float32), cur["sh0"].reshape(-1, 3),
            cur["opacities"], cur["scales"][:, :params.n_scales], path,
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
    colmap_dir: Path,
    out_path: Path,
    init_ply: Path | None = None,
    params: TrainParams = TrainParams(),
    resume: bool = True,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Fine-tune a splat, in `params.representation` (module docstring
    §REPRESENTATION), from a COLMAP model (poses + images — `colmap_dir`
    holds `cameras.txt` / `images.txt` / `points3D.txt` + the RGB images, exactly
    as `splat_to_colmap.py` / `splat.colmap.export_colmap` writes) plus the
    init selected by `params.init` (module docstring):

      * "surfels" (default) — `init_ply` (the Stage-3 `cloud.ply`) seeds every
        Gaussian on the mesh surface (on-surface means, mesh-true quats,
        tangent discs, exact texel colours), with opacity capped at
        `init_opa_max` for gradient headroom. The metric-depth warm-up
        (`depth_start_iter`) is zeroed: an opaque surface init's expected
        depth is true from step 0. A 3DGS run gets the same seed with a real
        third axis at `init_thickness_frac` of the disc radius, so it starts as
        a thin shell rather than a disc.
      * "points" — `_init_from_points` from the COLMAP points3D (the gsplat
        `create_splats_with_optimizers` recipe) — the Postshot-parity A/B.

    Supervision follows the data (photometric L1 + D-SSIM plus the
    representation's geometry prior). When the model carries the SZF
    SUPERVISION SIDECAR (`export_colmap` writes it for SZF refs), every view
    additionally supervises with the capture's exact alpha (coverage) +
    metric-depth planes at `alpha_lambda`/`depth_lambda`, and depth-guided
    densification seeds Gaussians at surfaces the render is missing — set
    those to 0/0/False for a Postshot-parity RGB-only run. Without the sidecar
    the terms are forced off (nothing to compare against). Writes the RAW
    optimized splat to `out_path` (`trained.ply`) with two or three scale columns
    per `params.representation`; requires a CUDA GPU + gsplat (raises a clear error
    otherwise).

    STAGE 6 IS TRAIN-ONLY: compaction (delete + heal + final-prune) and the LOD
    ladder live in Stage 7 (`heal_splat`), which reads this `trained.ply`.

    Clouds larger than `params.resolved_tile_budget` train TILED (ground-plane
    cells trained one at a time, merged by core ownership); smaller clouds train as
    a single run. With `resume` (default), continues from the latest `splat/ckpt/`
    checkpoint; `resume=False` starts fresh."""
    torch, F, _ = _require_cuda_trainer()

    colmap_dir, out_path = Path(colmap_dir), Path(out_path)
    for req in ("cameras.txt", "images.txt", "points3D.txt"):
        if not (colmap_dir / req).is_file():
            raise FileNotFoundError(
                f"COLMAP model incomplete: missing {req} in {colmap_dir} "
                "(build it with splat_to_colmap.py / splat.colmap.export_colmap)"
            )

    device = torch.device("cuda")
    torch.manual_seed(params.seed)

    # The Postshot-style input: reference views + shared intrinsics + the init point
    # cloud, all from the COLMAP model (pixels stream from disk per step). The SZF
    # sidecar, when present, attaches each view's exact alpha + depth planes.
    views, K, width, height, centers, pts_xyz, pts_rgb = _load_colmap(torch, colmap_dir, device)
    n_views = len(views)

    # ALWAYS announce the representation and the knobs it resolved, for the same
    # reason the supervision mode is announced below: these decide which rasterizer,
    # which regularizers and which densification criterion ran, and reading them off
    # the heartbeat is how a surprising result gets attributed to the right path.
    if progress is not None:
        if params.is_3dgs:
            detail = (
                f"absgrad={params.resolved_absgrad} (grow_grad2d={params.resolved_grow_grad2d:g}) "
                f"rasterize={params.rasterize_mode} packed={params.resolved_packed} "
                f"depth={params.resolved_depth_mode} glass_guard={params.resolved_glass_guard} "
                f"flat={params.flat_lambda:g} aniso={params.aniso_lambda:g}"
            )
        else:
            detail = (
                f"depth={params.resolved_depth_mode} normal={params.normal_lambda:g} "
                f"dist={params.dist_lambda:g}"
            )
        progress(0, 1, f"representation: {params.representation} — {detail}")

    # Supervision keys off the DATA: with the SZF sidecar covering EVERY view, the
    # alpha (coverage) + metric-depth losses and depth-guided densification run at
    # their configured weights — pass alpha_lambda=0 / depth_lambda=0 /
    # depth_densify=False for a Postshot-parity (RGB-only) run. Without full
    # coverage (legacy PNG refs, a hand-built model, torn frames) those terms are
    # forced OFF: an all-ones alpha target or a missing depth plane would
    # mis-supervise. The schedule stays raw here; it is resolved per run/tile just
    # before training.
    from dataclasses import replace
    n_frames = sum(1 for v in views if v["frame"] is not None)
    supervised = n_frames == n_views
    if not supervised:
        params = replace(params, alpha_lambda=0.0, depth_lambda=0.0,
                         depth_expected_lambda=0.0, depth_densify=False)
    # ALWAYS announce the supervision mode: a silent RGB-only fallback (stale
    # export, moved folder breaking the relative pointer, torn frames) would
    # waste a whole run before anyone noticed the depth/alpha terms never fired.
    if progress is not None:
        progress(
            0, 1,
            (
                f"supervision: RGB + alpha + depth via szf sidecar ({n_views} views, "
                f"alpha_lambda={params.alpha_lambda} depth_lambda={params.depth_lambda})"
                if supervised
                else "supervision: RGB-only ("
                + ("no szf sidecar" if n_frames == 0 else f"sidecar resolves only {n_frames}/{n_views} views")
                + ")"
            ),
        )

    # Init the splat (module docstring): the Stage-3 surfel cloud when
    # params.init == "surfels" (geometry starts ON the mesh surfaces), else the
    # gsplat from-points recipe. scene_scale is GLOBAL even when tiled, so every
    # tile's thresholds match the single-run semantics.
    if params.init == "surfels":
        if init_ply is None or not Path(init_ply).is_file():
            raise FileNotFoundError(
                "params.init='surfels' needs the Stage-3 surfel cloud — pass "
                f"init_ply= (got {init_ply}); or set init='points' for the "
                "from-points3D baseline"
            )
        # A 2-scale Stage-3 cloud gets its third axis here, at the fraction this
        # representation means by it (decoy vs real thickness); a cloud Stage 3
        # already wrote in `3dgs` mode carries its own and is used verbatim.
        init = _load_cloud(Path(init_ply), thickness_frac=params.resolved_thickness_frac)
        # Opacity CEILING (see TrainParams.init_opa_max): Stage 3 stores
        # near-saturated logits whose sigmoid gradient is ~1e-3 — frozen.
        # Cap the logit so opacity stays trainable (glass panes init well
        # below the cap and are untouched).
        p = min(max(params.init_opa_max, 1e-4), 1.0 - 1e-4)
        init["opacities"] = np.minimum(
            init["opacities"], np.float32(np.log(p / (1.0 - p)))
        )
        # An opaque surface init's rendered depth (expected or median) is true from
        # step 0 — the translucent-init warm-up would only delay the metric depth
        # term for no benefit.
        params = replace(params, depth_start_iter=0)
        if progress is not None:
            thick = (
                f", thickness {params.resolved_thickness_frac:.0%} of radius"
                if params.is_3dgs and _ply_representation(Path(init_ply)) == "2dgs"
                else ""
            )
            progress(
                0, 1,
                f"init: surfels ({init['means'].shape[0]:,} from {Path(init_ply).name}, "
                f"opacity ≤ {params.init_opa_max}{thick})",
            )
    else:
        init = _init_from_points(pts_xyz, pts_rgb, params)
        if progress is not None:
            progress(0, 1, f"init: points3D synthesize ({init['means'].shape[0]:,})")
    n_init = int(init["means"].shape[0])
    scene_scale = _scene_scale(centers, init["means"])

    # The schedule the single-run loop actually executes — reported below so the
    # recorded summary can't disagree with the run (it used to echo the unresolved
    # fields, so a 1,532-step run recorded `iterations: 30000`). Tiled runs resolve
    # per tile against their own view counts; `tiles_summary` carries those.
    resolved = params.resolve_schedule(n_views)

    # SCHEDULE SANITY. A run whose cadences all sit past its own step count trains,
    # scores a PSNR and ships a `.ply` — it just does none of the work, which is
    # indistinguishable from a broken representation switch when you look only at
    # the output. Announce it before the GPU spends the hour (`_inert_cadences`).
    inert = _inert_cadences(resolved)
    if inert:
        head = (
            f"schedule warning: {resolved.iterations} steps is too few for the "
            "default cadences (they are written against 30,000) — "
        )
        tail = (
            f" Raise `iterations` past the thresholds above (30,000 reaches all of "
            f"them; densification alone needs > {2 * resolved.refine_start_iter}), or "
            "size the run with `epochs`, which rescales every cadence to match."
        )
        logging.getLogger(__name__).warning("stage6: %s%s%s", head, "; ".join(inert), tail)
        if progress is not None:
            for line in (head.rstrip("— "), *(f"  - {i}" for i in inert), tail.strip()):
                progress(0, 1, line)

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
            resolved, resume, ckpt_root, progress, tile_box=None,
        )
        tiles_summary = None
        n_seeded, n_pruned = one["seeded"], one["pruned"]
        n_compacted = one.get("compacted", 0)
        compact_search = one.get("compact_search")
        speed_profile = one.get("profile")
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
        n_compacted = int(sum(t.get("compacted", 0) for t in tiles_summary))
        compact_search = None  # per-tile searches live in tiles_summary entries
        speed_profile = None   # per-tile profiles live in tiles_summary entries

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
        splats_t = {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}
        metrics = _evaluate(
            torch, _rasterizer(params), splats_t, views, K, width, height, params, device
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
        arrays["opacities"], arrays["scales"][:, :params.n_scales], out_path,
        sh_rest=arrays["shN"],
    )
    # No LOD ladder here: trained.ply is the RAW model. The delivered LODs are
    # built on the cleaned model in Stage 7 (`heal_splat` → healed.lodK.ply).
    # Drop any stale trained.lod*.ply an older (LOD-exporting) run left beside
    # this path — a fresh trained.ply must never ship next to giant stale
    # ladders (a 2M-vertex leftover sat beside swamp-land's 84k rewrite).
    for old in out_path.parent.glob(f"{out_path.stem}.lod*.ply"):
        old.unlink(missing_ok=True)
    lod_summary = None

    # Training finished + the final splat is on disk → the checkpoints (and any
    # tile caches under them) are obsolete; drop them so a later resume doesn't
    # re-enter a completed run.
    import shutil

    shutil.rmtree(ckpt_root, ignore_errors=True)

    return {
        "run": run,
        "slot": slot,
        "model": model,
        "representation": params.representation,
        "splats_init": n_init,
        "splats_final": n_final,
        "splats_pruned_final": n_pruned,
        "splats_compacted": n_compacted,
        "compact_search": compact_search,
        "splats_depth_seeded": n_seeded,
        "iterations": resolved.iterations,
        "schedule": {
            "steps": resolved.iterations,
            "batch": resolved.batch,
            "view_draws": resolved.iterations * max(resolved.batch, 1),
            "refine_start_iter": resolved.refine_start_iter,
            "refine_stop_iter": resolved.resolved_refine_stop,
            "refine_every": resolved.refine_every,
            "reset_every": resolved.reset_every,
            "prune_scale_start_iter": resolved.prune_scale_start_iter,
            "sh_degree_interval": resolved.sh_degree_interval,
            "normal_start_iter": resolved.normal_start_iter,
            "depth_start_iter": resolved.depth_start_iter,
            "tiled": tiles_summary is not None,
            # Machinery this run's step count put out of reach (`_inert_cadences`),
            # recorded so a disappointing model can be attributed from the summary
            # alone instead of re-deriving the cadences by hand.
            "inert": inert or None,
        },
        "speed_profile": speed_profile,
        "views": n_views,
        "resolution": width,
        "scene_scale": round(scene_scale, 4),
        "tiles": tiles_summary,
        "lod": lod_summary,
        "metrics": metrics,
        "params": params.as_summary(),
        "bytes": out_path.stat().st_size,
        "colmap_dir": str(colmap_dir),
        "out_path": str(out_path),
    }


def _load_scene(torch, refs_dir: Path, device):  # noqa: ANN001
    """Reference views + shared intrinsics from a Stage-5 refs dir, exactly as the
    trainer + the heal stage consume them: per frame the w2c `viewmat`, the c2w
    (for scene scale / AA), and the SZF (or legacy PNG) frame paths + shared depth
    near/far. Returns (views, K, width, height, centers). Shared by `train_splat`
    (Stage 6) and `heal_splat` (Stage 7) so both read the plan identically."""
    doc = _read_transforms(refs_dir)
    frames = doc.get("frames", [])
    if not frames:
        raise RuntimeError(f"{refs_dir/TRANSFORMS_NAME} has no frames (run Stage 5)")
    width, height = int(doc["w"]), int(doc["h"])
    K = torch.tensor(
        [[doc["fl_x"], 0.0, doc["cx"]], [0.0, doc["fl_y"], doc["cy"]], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )

    def _frame_K(fr: dict[str, Any]):  # noqa: ANN202 - CPU [3,3], stacked per batch
        """That frame's own intrinsics, falling back to the document's. Stage 5
        writes per-frame `fl_x`/`fl_y`/`cx`/`cy` because Stage 4 derives FOV per
        camera; a refs dir written before that carries none and every frame
        resolves to the shared values, so old sets train exactly as before."""
        fx = float(fr.get("fl_x", doc["fl_x"]))
        fy = float(fr.get("fl_y", doc["fl_y"]))
        cx = float(fr.get("cx", doc["cx"]))
        cy = float(fr.get("cy", doc["cy"]))
        return torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32
        )

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
                "K": _frame_K(fr),
                "c2w": c2w,
                "frame": refs_dir / frame_rel if frame_rel else None,
                "rgb": refs_dir / rgb_rel if rgb_rel else None,
                "alpha": refs_dir / fr["alpha_path"] if fr.get("alpha_path") else None,
                "depth": refs_dir / fr["depth_path"] if fr.get("depth_path") else None,
                "depth_near": depth_near,
                "depth_far": depth_far,
            }
        )
    return views, K, width, height, np.stack(cam_centers, axis=0)


def _load_colmap(torch, colmap_dir: Path, device):  # noqa: ANN001, ANN202
    """The COLMAP model (the Postshot-style input) as the trainer consumes it:
    (views, K, width, height, centers, points_xyz, points_rgb). Each view is the
    same dict `_view_stream`/`_evaluate` expect — the w2c `viewmat`, the c2w, and
    the RGB image path — but with `frame`/`alpha`/`depth` None: the image is read as
    RGB (the PNG's coverage alpha, kept for Postshot masking, is dropped) and COLMAP
    has no depth. `points_xyz`/`points_rgb` seed the splat via `_init_from_points`.
    Image names resolve against the model dir (flat, as `export_colmap` writes) or
    an `images/` subdir (a standard COLMAP layout)."""
    colmap_dir = Path(colmap_dir)
    cams_np, width, height = _read_colmap_cameras(colmap_dir / "cameras.txt")
    images = _read_colmap_images(colmap_dir / "images.txt")
    pts_xyz, pts_rgb = _read_colmap_points(colmap_dir / "points3D.txt")
    cam_K = {
        cid: torch.tensor(m, dtype=torch.float32) for cid, m in cams_np.items()
    }
    # The representative K, for the scene-scale and anti-aliasing arithmetic that
    # wants ONE focal length. The median rather than the first record, so a plan
    # with a range of angles is characterized by its middle instead of by whichever
    # camera happened to be written first.
    med = float(np.median([m[0, 0] for m in cams_np.values()]))
    K = torch.tensor(
        np.array([[med, 0.0, width / 2.0], [0.0, med, height / 2.0], [0.0, 0.0, 1.0]]),
        dtype=torch.float32,
        device=device,
    )

    frames_dir = suffix = near = far = None
    sidecar_path = colmap_dir / _SIDECAR_NAME
    if sidecar_path.is_file():
        sc = json.loads(sidecar_path.read_text(encoding="utf-8"))
        frames_dir = colmap_dir / sc["frames_dir"]
        suffix = sc.get("suffix", ".szf")
        near, far = float(sc["near"]), float(sc["far"])

    views: list[dict[str, Any]] = []
    centers: list[np.ndarray] = []
    for name, c2w, cid in images:
        img = colmap_dir / name
        if not img.is_file():
            img = colmap_dir / "images" / name
        if not img.is_file():
            raise FileNotFoundError(f"{colmap_dir}: image '{name}' from images.txt not found")
        if cid not in cam_K:
            raise ValueError(
                f"{colmap_dir}: image '{name}' names CAMERA_ID {cid}, which cameras.txt "
                "does not declare"
            )
        frame = None
        if frames_dir is not None:
            f = frames_dir / (Path(name).stem + suffix)
            if f.is_file():
                frame = f
        centers.append(c2w[:3, 3])
        views.append(
            {
                "viewmat": torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)),
                "K": cam_K[cid],
                "c2w": c2w,
                "frame": frame,
                "rgb": img,
                "alpha": None,
                "depth": None,
                "depth_near": near,
                "depth_far": far,
            }
        )
    if not views:
        raise RuntimeError(f"{colmap_dir}/images.txt has no image poses")
    return views, K, width, height, np.stack(centers, axis=0), pts_xyz, pts_rgb


def _inert_cadences(params: TrainParams) -> list[str]:
    """The parts of the loop a RESOLVED schedule can never reach, as human-readable
    strings (empty = everything fires at least once).

    Every cadence in `TrainParams` is written against the reference 30,000-step
    length, so shortening a run by lowering `iterations` alone silently switches
    machinery OFF rather than compressing it — and the result still trains, still
    reports a PSNR, and still ships a `.ply`, so nothing announces the loss.

    The densification window is the trap that bites first, because
    `resolved_refine_stop` defaults to HALF the step count while
    `refine_start_iter` is absolute: at `iterations` ≤ 2·refine_start_iter the two
    cross and gsplat's `step_post_backward` returns before it can ever grow OR
    prune. A 1,000-step run then emits exactly its init cloud, re-fitted — which
    for a 3DGS run reads as "the surfels came back", since Stage 3's layout and
    count are all that is left.

    The fix is to give the run enough steps: the reference 30,000 reaches every
    cadence, and `iterations` has to clear each threshold named below for that part
    of the loop to fire at all. (`epochs` also works — `resolve_schedule` rescales
    every cadence with the step count — but it is a library/API field, not a
    dashboard knob, and it REPLACES `iterations` as the run length while reusing it
    as the reference denominator, so setting both is its own trap.)"""
    out: list[str] = []
    n = params.iterations
    if params.refine and params.resolved_refine_stop <= params.refine_start_iter:
        out.append(
            f"DENSIFICATION never runs (window is empty: refine_start_iter="
            f"{params.refine_start_iter} >= refine_stop={params.resolved_refine_stop}), "
            "so the model cannot grow or prune a single Gaussian — it will be your "
            "init cloud, re-fitted"
        )
    if params.sh_degree > 0 and n <= params.sh_degree_interval:
        out.append(
            f"VIEW-DEPENDENT COLOUR never activates (sh_degree_interval="
            f"{params.sh_degree_interval} >= {n} steps), so the model stays flat "
            "degree-0 no matter what `sh_degree` says"
        )
    prior = params.flat_lambda if params.is_3dgs else params.normal_lambda
    if prior > 0.0 and params.normal_start_iter >= n:
        out.append(
            f"the GEOMETRY PRIOR never engages (normal_start_iter="
            f"{params.normal_start_iter} >= {n} steps)"
        )
    if params.prune_scale_start_iter >= n:
        out.append(
            f"the SIZE PRUNE never arms (prune_scale_start_iter="
            f"{params.prune_scale_start_iter} >= {n} steps), so nothing bounds "
            "Gaussian size"
        )
    if params.reset_every > 0 and params.reset_every >= n:
        out.append(
            f"the OPACITY RESET never fires (reset_every={params.reset_every} >= {n} steps)"
        )
    return out


def _scene_scale(centers: np.ndarray, fallback_means: np.ndarray) -> float:
    """Camera-cloud radius (× 1.1), the 3DGS spatial-LR / density unit — the same
    value Stage 6 trained with, so the heal's LRs and the prune-scale threshold
    match. Falls back to half the model's own extent when there's ≤1 camera."""
    n = int(centers.shape[0])
    s = float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) if n > 1 else 0.0
    if s <= 1e-6:
        s = float(np.linalg.norm(fallback_means.max(0) - fallback_means.min(0))) * 0.5
    return max(s * 1.1, 1e-3)


def _compact_and_heal(  # noqa: ANN001
    torch, F, raster, splats, views, K, width: int, height: int,
    params: "TrainParams", scene_scale: float, device, init_means: np.ndarray,
    cam_tree, focal: float, progress: ProgressCb | None,
):
    """Stage-7 compaction on a TRAINED model (module docstring §COMPACTION), run
    GLOBALLY over all `views` (no tile box): measure each Gaussian's total rendered
    contribution, apply the unconditional culls (lossless-invisible + off-surface
    floaters), the quality-gated budget cut, heal the survivors, then the one-shot
    cleanup prune. Mutates `splats`; returns `(arrays cpu-numpy, info)`.

    Extracted verbatim from the old end-of-`_train_one` tail so the delivered model
    is identical to what training used to fold in — only now it is its own stage,
    operating on the merged `trained.ply` instead of per-tile."""
    box_lo = box_hi = px_grid = py_grid = None   # global heal: no tile mask
    n_before = int(splats["means"].shape[0])
    n_compacted = 0
    compact_search = None
    if params.compact:
        weights = _contribution_weights(
            torch, raster, splats, views, K, width, height, params, progress,
        )
        with torch.no_grad():
            keep = _select_keep(
                torch, weights, splats["sh0"], splats["opacities"], splats["means"],
                init_means, params.compact_eps, params.surface_max_dist, None,
            )
            for name in list(splats.keys()):
                splats[name] = torch.nn.Parameter(splats[name][keep].detach(), requires_grad=True)
            weights = weights[keep]
        n0 = int(splats["means"].shape[0])
        if progress is not None:
            progress(
                1, 1,
                f"compact: -{n_before - n0} unrenderable (<{params.compact_eps:.4f}) or "
                f">{params.surface_max_dist}m off-surface -> {n0}",
            )

        chosen = None
        if params.compact_max_db_drop is not None and n0 > 0:
            ev0 = _evaluate(
                torch, raster, {k: v.detach() for k, v in splats.items()},
                views, K, width, height, params, device,
            )
            psnr0 = float(ev0["psnr"])
            lo, hi = 0.10, 1.0
            best = None
            for pi in range(max(params.compact_probes, 1)):
                mid = (lo + hi) / 2.0
                with torch.no_grad():
                    kmask = _select_keep(
                        torch, weights, splats["sh0"], splats["opacities"], splats["means"],
                        None, params.compact_eps, 0.0, max(1, round(mid * n0)),
                        alpha_floor=params.prune_opa,
                    )
                    probe = {
                        k: torch.nn.Parameter(v.detach()[kmask].clone()) for k, v in splats.items()
                    }
                if params.compact_probe_heal_steps > 0:
                    _heal(
                        torch, F, raster, probe, views, K, width, height, params,
                        scene_scale, params.compact_probe_heal_steps,
                        box_lo, box_hi, px_grid, py_grid, None,
                    )
                ev = _evaluate(
                    torch, raster, {k: v.detach() for k, v in probe.items()},
                    views, K, width, height, params, device,
                )
                drop = psnr0 - float(ev["psnr"])
                ok = drop <= params.compact_max_db_drop
                if progress is not None:
                    progress(
                        1, 1,
                        f"compact/search {pi + 1}/{params.compact_probes}: keep {mid:.2f} "
                        f"-> {ev['psnr']:.2f} dB (drop {drop:+.2f}) {'ok' if ok else 'too lossy'}",
                    )
                if ok:
                    best, hi = (mid, kmask), mid
                else:
                    lo = mid
                del probe
                torch.cuda.empty_cache()
            if best is not None and best[0] < 1.0:
                chosen = best[1]
                compact_search = {
                    "psnr_clean": round(psnr0, 3),
                    "keep_frac": round(best[0], 3),
                    "max_db_drop": params.compact_max_db_drop,
                    "probes": params.compact_probes,
                }
        elif params.compact_keep_frac is not None and n0 > 0:
            with torch.no_grad():
                chosen = _select_keep(
                    torch, weights, splats["sh0"], splats["opacities"], splats["means"],
                    None, params.compact_eps, 0.0, max(1, round(params.compact_keep_frac * n0)),
                    alpha_floor=params.prune_opa,
                )

        if chosen is not None:
            with torch.no_grad():
                for name in list(splats.keys()):
                    splats[name] = torch.nn.Parameter(splats[name][chosen].detach(), requires_grad=True)
            if progress is not None:
                progress(
                    1, 1,
                    f"compact/budget: -{n0 - int(splats['means'].shape[0])} low-contribution "
                    f"-> {int(splats['means'].shape[0])}",
                )

        n_compacted = n_before - int(splats["means"].shape[0])
        if params.compact_heal_steps > 0 and n_compacted > 0:
            _heal(
                torch, F, raster, splats, views, K, width, height, params,
                scene_scale, params.compact_heal_steps, box_lo, box_hi, px_grid, py_grid, progress,
            )
            if params.antialias and cam_tree is not None:
                _aa_scale_floor(
                    torch, splats, cam_tree, focal, params.aa_min_scale_px,
                    tangent_only=not params.is_3dgs,
                )

    n_pruned = (
        _final_prune(torch, splats, params.prune_opa, params.prune_scale3d, scene_scale)
        if params.final_prune
        else 0
    )
    if progress is not None and n_pruned:
        progress(
            1, 1,
            f"final prune: -{n_pruned} floaters below opacity {params.prune_opa} "
            f"(-> {int(splats['means'].shape[0])} splats)",
        )

    with torch.no_grad():
        arrays = {k: v.detach().cpu().numpy() for k, v in splats.items()}
    info = {
        "final": int(arrays["means"].shape[0]),
        "compacted": n_compacted,
        "pruned": n_pruned,
        "compact_search": compact_search,
    }
    return arrays, info


def heal_splat(
    *,
    run: str,
    slot: str,
    model: str,
    trained_path: Path,
    cloud_path: Path,
    refs_dir: Path,
    out_path: Path,
    params: TrainParams = TrainParams(),
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """STAGE 7 — delete + heal + final-prune on the Stage-6 `trained.ply`, writing
    the delivered `healed.ply` (+ its LOD ladder) to `out_path`. This is the
    compaction that used to be folded into training's `trained.ply`; split out so
    `trained.ply` is the raw optimization output and `healed.ply` is the cleaned
    deliverable, viewable side by side. Global (whole merged model, all views) —
    the per-tile compaction is gone with the fold-out. Requires CUDA + gsplat.

    `params.representation` must match what `trained.ply` actually holds: this
    stage RE-RENDERS the model to measure each Gaussian's contribution and then
    re-optimizes the survivors, so the wrong rasterizer would both mis-measure and
    mis-heal. The file is self-describing (`_ply_representation`), so the mismatch
    is caught up front instead of shipping a quietly wrong `healed.ply`."""
    torch, F, _ = _require_cuda_trainer()
    from scipy.spatial import cKDTree

    trained_path, cloud_path = Path(trained_path), Path(cloud_path)
    refs_dir, out_path = Path(refs_dir), Path(out_path)
    if not trained_path.is_file():
        raise FileNotFoundError(f"trained splat not found: {trained_path} (run Stage 6)")
    if not cloud_path.is_file():
        raise FileNotFoundError(f"surfel cloud not found: {cloud_path} (run Stage 3)")
    on_disk = _ply_representation(trained_path)
    if on_disk != params.representation:
        raise ValueError(
            f"{trained_path.name} is a {on_disk} model but params.representation is "
            f"{params.representation!r} — heal renders and re-optimizes the model, so "
            f"the two must agree. Re-run Stage 6 with representation={params.representation!r}, "
            f"or heal with representation={on_disk!r}."
        )

    raster = _rasterizer(params)
    device = torch.device("cuda")
    torch.manual_seed(params.seed)
    views, K, width, height, centers = _load_scene(torch, refs_dir, device)
    n_views = len(views)

    trained_model = _load_cloud(trained_path)   # the trained model to heal
    init = _load_cloud(cloud_path)              # Stage-3 surfels = the surface prior
    n_before = int(trained_model["means"].shape[0])
    scene_scale = _scene_scale(centers, trained_model["means"])

    splats = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(torch.from_numpy(trained_model[k]).to(device)) for k in trained_model}
    )
    cam_tree = cKDTree(centers) if (params.antialias and n_views > 0) else None
    focal = float(K[0, 0].item())

    t0 = time.perf_counter()
    if progress is not None:
        progress(0, 1, f"heal: {n_before} splats, {n_views} views, scale={scene_scale:.2f}")
    arrays, hinfo = _compact_and_heal(
        torch, F, raster, splats, views, K, width, height, params,
        scene_scale, device, init["means"], cam_tree, focal, progress,
    )
    n_final = int(arrays["means"].shape[0])

    metrics = None
    try:
        splats_t = {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}
        metrics = _evaluate(
            torch, raster, splats_t, views, K, width, height, params, device
        )
        del splats_t
    except (torch.cuda.OutOfMemoryError, MemoryError):
        torch.cuda.empty_cache()

    if progress is not None:
        progress(1, 1, f"heal done in {_fmt_hms(time.perf_counter() - t0)} - writing {out_path.name}")
    quats = arrays["quats"] / (np.linalg.norm(arrays["quats"], axis=1, keepdims=True) + 1e-12)
    _encode_trained_ply(
        arrays["means"], quats.astype(np.float32), arrays["sh0"].reshape(-1, 3),
        arrays["opacities"], arrays["scales"][:, :params.n_scales], out_path,
        sh_rest=arrays["shN"],
    )
    lod_summary = _export_lod(arrays, out_path, params, progress)

    return {
        "run": run,
        "slot": slot,
        "model": model,
        "representation": params.representation,
        "splats_in": n_before,
        "splats_final": n_final,
        "splats_compacted": hinfo["compacted"],
        "splats_pruned_final": hinfo["pruned"],
        "compact_search": hinfo["compact_search"],
        "views": n_views,
        "resolution": width,
        "scene_scale": round(scene_scale, 4),
        "lod": lod_summary,
        "metrics": metrics,
        "params": params.as_summary(),
        "bytes": out_path.stat().st_size,
        "out_path": str(out_path),
    }


class _Render(NamedTuple):
    """One rasterizer forward, normalized across representations so nothing
    downstream has to know which one ran. The three 2DGS-only buffers are None on
    the 3DGS path (`rasterization` returns no rendered normal, no normals-from-
    depth and no distortion map), and every consumer of them is gated accordingly."""

    rgb: Any                 # [B,H,W,3], premultiplied over black
    alpha: Any               # [B,H,W,1] accumulated coverage
    depth: Any               # [B,H,W,1] the statistic the depth loss compares
    expected: Any            # [B,H,W,1] the ED channel (both paths)
    normals: Any             # [B,H,W,3] rendered normals — 2DGS only
    normals_from_depth: Any  # [B,H,W,3] normals of the rendered depth — 2DGS only
    distort: Any             # [B,H,W,1] depth distortion — 2DGS only
    info: dict               # the rasterizer's meta dict (densification reads it)


def _render_batch(torch, raster, splats, colors, sh_deg, viewmats, Ks, width, height, params, dist_on):  # noqa: ANN001
    """One gsplat forward for a batch of views — the single render call every part
    of the stage goes through (training loop, heal, contribution measure, eval), so
    all of them necessarily agree on the operator.

    `raster` is `_rasterizer(params)`. The two entrypoints differ in their return
    arity (3 vs 7) and in three kwargs each doesn't accept, which is the whole
    reason this wrapper exists: it hands back a `_Render` either way.

    `Ks` is PER VIEW [B,3,3], matching `viewmats` — gsplat takes per-camera
    intrinsics natively, and Stage 4 varies FOV per camera."""
    common = dict(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_deg,
        near_plane=params.near_plane,
        far_plane=params.far_plane,
        render_mode="RGB+ED",
    )
    if params.is_3dgs:
        renders, alpha, info = raster(
            **common,
            packed=params.resolved_packed,
            # AbsGS: the backward hangs `.absgrad` on `info["means2d"]`, which is
            # exactly the tensor DefaultStrategy densifies off here — so unlike the
            # 2DGS path this actually works (see TrainParams.absgrad).
            absgrad=params.resolved_absgrad,
            rasterize_mode=params.rasterize_mode,
        )
        # No median depth exists on this path, so the ED channel IS the depth the
        # loss compares (TrainParams.resolved_depth_mode pins it to "expected").
        expected = renders[..., 3:4]
        return _Render(renders[..., :3], alpha, expected, expected, None, None, None, info)

    renders, alpha, normals, nfd, distort, median, info = raster(
        **common,
        # Non-packed: gsplat's packed 2DGS depth path is broken — its SH branch
        # mis-broadcasts and its precomputed-colour path skips the visible-subset
        # gather, so RGB+ED crashes once a view sees only part of the cloud.
        packed=False,
        distloss=dist_on,
        depth_mode=params.resolved_depth_mode,
        # absgrad is unsupported on gsplat 1.5.3's 2DGS densification path
        # (`gradient_2dgs` never gets `.absgrad`; see grow_grad2d in TrainParams).
        absgrad=False,
    )
    expected = renders[..., 3:4]
    depth = median if params.resolved_depth_mode == "median" else expected
    return _Render(renders[..., :3], alpha, depth, expected, normals, nfd, distort, info)


def _supervision_loss(  # noqa: ANN001
    torch, F, params, window, gt_rgb, gt_alpha, gt_depth, render, scales, mask,
    normals_active, dist_active, depth_active=True, scene_scale=1.0,
    ed_target=None, ed_gate=None,
):
    """Combined per-view supervision loss — photometric L1 + D-SSIM, alpha
    (coverage), disparity-space depth L1, and the GEOMETRY prior for whichever
    representation is training — shared by the training loop and the compaction
    pass. Both RGB sides are premultiplied-over-black. `mask` (a tile's
    owned-pixel mask, or None for a single run) restricts every term to owned
    pixels; `normals_active` / `dist_active` / `depth_active` gate the terms that
    only switch on partway through a run (`depth_active` defaults True for the
    heal/compact callers, whose models are already opaque).

    The photometric, alpha and depth terms are representation-independent — they
    compare rendered pixels against the capture, and `_Render` has already
    normalized those. Only the geometry prior differs, and `normals_active` gates
    both variants because both exist for the same reason (don't freeze half-built
    geometry): 2DGS ties rendered normals to the render's own depth, while 3DGS —
    which has no rendered normal — imposes flatness and bounded anisotropy on the
    scale parameters directly (`params.flat_lambda` / `aniso_lambda`)."""
    pred_rgb, pred_alpha, pred_depth = render.rgb, render.alpha, render.depth
    if mask is None:
        l1 = (pred_rgb - gt_rgb).abs().mean()
        ssim_rgb = pred_rgb
    else:
        msum = mask.sum().clamp_min(1.0)
        l1 = ((pred_rgb - gt_rgb).abs() * mask).sum() / (msum * 3.0)
        ssim_rgb = torch.where(mask > 0, pred_rgb, gt_rgb.detach())
    # Clamp the prediction into display range before SSIM. This costs the gradient
    # on over-bright pixels (the L1 term still supplies it), but `_ssim` forms
    # variances as E[x²] − E[x]², which catastrophically cancels once x is large,
    # and the denominator carries only c2 = 9e-4 of protection before it crosses
    # zero and produces NaN. Not worth removing for the marginal gradient.
    ssim = _ssim(F, ssim_rgb.clamp(0, 1).permute(0, 3, 1, 2), gt_rgb.permute(0, 3, 1, 2), window)
    loss = (1.0 - params.ssim_lambda) * l1 + params.ssim_lambda * (1.0 - ssim)

    if params.alpha_lambda > 0.0:
        aerr = (pred_alpha - gt_alpha).abs()
        aloss = aerr.mean() if mask is None else (aerr * mask).sum() / mask.sum().clamp_min(1.0)
        loss = loss + params.alpha_lambda * aloss

    if gt_depth is not None and params.depth_lambda > 0.0 and depth_active:
        # Gate on the RENDER as well as the reference: where nothing crossed
        # transmittance 0.5 the median depth is 0, and comparing that against a
        # metres-away reference is a full-strength pull on whichever Gaussian the
        # kernel names the median — pure noise during the window when coverage is
        # still being built. Surface is the photometric/alpha terms' job; depth
        # only positions surface that already renders.
        gate = (gt_alpha > params.alpha_gate) & (gt_depth > 0) & (pred_depth.detach() > 0)
        if mask is not None:
            gate = gate & (mask > 0)
        if params.resolved_glass_guard and ed_target is not None:
            # GLASS EXCLUSION. The stored plane is the nearest OPAQUE surface, and
            # expected depth (the only statistic 3DGS renders) is the α-weighted
            # mean along the ray — so at a pixel with a transmissive pane in front
            # of a wall, pulling expected depth onto the wall is a demand to DELETE
            # the pane. The init cloud renders the true two-layer expected depth
            # (its glass surfels sit on the panes), so wherever that disagrees with
            # the stored plane there is a transmissive layer, and this term has
            # nothing correct to say: drop those pixels and leave them to the
            # photometric loss and `depth_expected_lambda`. Median depth needs none
            # of this — it already lands on the opaque surface — so the 2DGS default
            # keeps the full gate.
            gate = gate & ((ed_target - gt_depth).abs() <= _ED_GLASS_TOL_FRAC * gt_depth)
        if gate.any():
            # Disparity space × scene_scale (the reference trainer's form): a
            # dimensionless residual whose gradient falls off as 1/d², so the far
            # wall stops outvoting the near table.
            #
            # FLOOR pred at a fraction of the reference depth, or 1/d is a
            # SINGULARITY. gsplat's median depth is the frontmost thing on the ray
            # (`RasterizeToPixels2DGSFwd.cu` overwrites it while T > 0.5), and one
            # Gaussian that drifts near a camera projects to a huge screen radius —
            # at 4 cm from a 1024² camera a 3 cm surfel covers the WHOLE frame, so
            # every pixel's median depth collapses to 4 cm at once. Metric L1 saw a
            # bounded ~3 m error there; raw disparity sees 25 and diverges (measured:
            # loss 2.77 at step 1151, then non-finite means). Flooring at 0.1·gt caps
            # the residual at 9·disp_gt — scale-free, and still a strong pull toward
            # the truth — while leaving genuinely-nearer-than-truth surfaces the
            # photometric term's job, which is where floater removal belongs.
            pred = torch.maximum(pred_depth, _DEPTH_DISP_FLOOR_FRAC * gt_depth)
            disp = torch.where(
                pred > 0, 1.0 / pred.clamp_min(1e-6), torch.zeros_like(pred)
            )
            disp_gt = 1.0 / gt_depth.clamp_min(1e-6)
            dl = ((disp - disp_gt).abs() * gate).sum() / (gate.sum() + 1e-8)
            loss = loss + params.depth_lambda * scene_scale * dl

    # "Record both" glass-maker: the splat's EXPECTED depth vs the frozen cloud's
    # two-layer expected depth, over opaque-covered pixels where the cloud target
    # is valid. Deleting a pane pulls expected depth back onto the wall (off
    # target), so this requires the transmissive layer; the median term above keeps
    # the wall pinned so the pair can't be faked by sliding the wall forward.
    if (
        ed_target is not None and gt_depth is not None
        and params.depth_expected_lambda > 0.0 and depth_active
    ):
        eg = (gt_alpha > params.alpha_gate) & (gt_depth > 0)
        if ed_gate is not None:
            eg = eg & ed_gate
        if mask is not None:
            eg = eg & (mask > 0)
        if eg.any():
            el = ((render.expected - ed_target).abs() * eg).sum() / (eg.sum() + 1e-8)
            loss = loss + params.depth_expected_lambda * el

    if params.normal_lambda > 0.0 and normals_active and render.normals is not None:
        # Scale the depth-derived normal by the RENDERED alpha, as the reference
        # trainer does: at a pixel the splat hasn't covered yet the depth patch is
        # empty and its normal is arbitrary, so without this the term pulls the
        # rendered normals toward garbage over exactly the pixels still being
        # filled in. Zeroed there, the error is a constant 1 with no gradient.
        nfd = render.normals_from_depth * pred_alpha.detach()
        nerr = 1.0 - (render.normals * nfd).sum(dim=-1)
        fg_mask = gt_alpha.squeeze(-1) > params.alpha_gate
        if mask is not None:
            fg_mask = fg_mask & (mask.squeeze(-1) > 0)
        if fg_mask.any():
            loss = loss + params.normal_lambda * nerr[fg_mask].mean()

    if dist_active and params.dist_lambda > 0.0 and render.distort is not None:
        loss = loss + params.dist_lambda * render.distort.mean()

    # 3DGS geometry priors, in PARAMETER space — the stand-in for the normal
    # consistency this rasterizer can't provide (TrainParams.flat_lambda). Both are
    # ratios of a Gaussian's own axes, so they are scale-free and cost one [N,3]
    # sort; neither looks at the reference, so a masked tile pays nothing extra.
    if (
        params.is_3dgs and normals_active and scales is not None
        and (params.flat_lambda > 0.0 or params.aniso_lambda > 0.0)
    ):
        s = torch.sort(torch.exp(scales), dim=-1, descending=True).values   # [N,3]
        if params.flat_lambda > 0.0:
            # FLATNESS: smallest/largest above `flat_max`, denominator DETACHED so
            # the only way to satisfy it is to thin the normal axis — never to widen
            # the footprint, which an undetached quotient would reward equally.
            # Bounded by the relu, so a Gaussian already flatter than the target is
            # left alone instead of being walked toward a degenerate covariance.
            flat = s[:, 2] / s[:, 0].detach().clamp_min(1e-12)
            loss = loss + params.flat_lambda * (flat - params.flat_max).clamp_min(0.0).mean()
        if params.aniso_lambda > 0.0:
            # NEEDLES: largest/middle above `aniso_max`. A disk is 1.0 and free, a
            # spike is unbounded. Two-sided on purpose — shortening the long axis
            # and fattening the middle one both fix it.
            ratio = s[:, 0] / s[:, 1].clamp_min(1e-12)
            loss = loss + params.aniso_lambda * (ratio - params.aniso_max).clamp_min(0.0).mean()
    return loss


def _contribution_weights(  # noqa: ANN001
    torch, raster, splats, views, K, width: int, height: int, params, progress=None,
):
    """EXACT per-Gaussian rendered contribution, measured through the rasterizer:
    a pixel is Σᵢ wᵢ·cᵢ with wᵢ the blend weight (opacity × kernel ×
    transmittance), so the gradient of Σ(all pixels) w.r.t. each Gaussian's SH0
    colour is C0·Σ wᵢ over every pixel it touches. Accumulated over ALL `views`
    (batched, one backward each — gradients only flow to a cloned sh0 leaf, the
    model itself is untouched) and returned as a [N] tensor of total blend
    weight. Reference pixels are NOT needed — only the poses — so the pass costs
    roughly one epoch of forwards plus a cheap colour-only backward.

    gsplat clamps SH colours at 0, which zeroes the gradient for clamped channels.
    The measure differentiates the DC leaf only — the DC basis contributes C0·Σw to
    every view regardless of the higher SH bands — so it stays a valid blend-weight
    proxy at degree 3; a Gaussian whose DC renders pure black in a view is just
    unmeasured there, so callers force-keep `_blind_mask` (all-DC-channels < 0)
    Gaussians."""
    device = splats["means"].device
    sh0_leaf = splats["sh0"].detach().clone().requires_grad_(True)
    shadow = {k: (sh0_leaf if k == "sh0" else v.detach()) for k, v in splats.items()}
    b = max(int(params.batch), 1)
    n_batches = (len(views) + b - 1) // b
    for bi, lo in enumerate(range(0, len(views), b)):
        sel = range(lo, min(lo + b, len(views)))
        viewmats = torch.stack([views[i]["viewmat"] for i in sel]).to(device)
        Ks = torch.stack([views[i]["K"] for i in sel]).to(device)
        colors, sh_deg = _render_inputs(torch, shadow, params.sh_degree)
        render = _render_batch(
            torch, raster, shadow, colors, sh_deg, viewmats, Ks,
            width, height, params, dist_on=False,
        )
        render.rgb.sum().backward()
        if progress is not None and (bi % 50 == 0 or bi == n_batches - 1):
            progress(bi + 1, n_batches, f"compact: measuring contribution ({bi + 1}/{n_batches} batches)")
    if sh0_leaf.grad is None:
        return torch.zeros(len(sh0_leaf), device=device)
    return sh0_leaf.grad.abs().amax(dim=(1, 2)) / _SH_C0


def _blind_mask(torch, sh0):  # noqa: ANN001
    """Gaussians whose SH0 colour clamps to pure black in every channel (gsplat
    clamps colours at 0, killing their gradient): `_contribution_weights` cannot
    see these even when they are visible, so compaction keeps them
    unconditionally."""
    rgb = 0.5 + _SH_C0 * sh0.detach()[:, 0, :]
    return (rgb < 0).all(dim=1)


def _select_keep(  # noqa: ANN001
    torch, weights, sh0, opacities, means, init_means, eps, surface_max_dist, keep_count,
    alpha_floor=0.005,
):
    """Boolean keep-mask for compaction (module docstring §COMPACTION), composing
    up to three signals from the measured contribution `weights`:
      * LOSSLESS — keep every Gaussian with contribution > `eps` (below it it
        cannot move any pixel by half a display step), plus pure-black
        `_blind_mask` Gaussians the measure can't see;
      * SURFACE PRIOR — with `surface_max_dist` > 0 and `init_means` given, drop
        any Gaussian farther than that (metres) from every Stage-3 surfel; this
        OVERRIDES lossless/blind, so bright and black floaters both go;
      * BUDGET — with `keep_count` not None, additionally keep only the top
        `keep_count` survivors by OPACITY-NORMALIZED contribution (weight/α:
        what the Gaussian would paint if opaque). Raw blend weight scores
        translucent-BY-DESIGN content as unimportant — ranked raw, the hotel's
        α≈0.065 window panes were 99.5% deleted at keep-30% — while w/α ranks a
        pane like the surface it covers. Blind Gaussians are kept regardless.
    Shared by the training-loop compaction and the post-hoc experiment runners so
    the selection is identical everywhere."""
    blind = _blind_mask(torch, sh0)
    keep = (weights > eps) | blind
    if surface_max_dist and surface_max_dist > 0 and init_means is not None:
        from scipy.spatial import cKDTree

        d, _ = cKDTree(np.asarray(init_means)).query(means.detach().cpu().numpy(), k=1, workers=-1)
        keep = keep & torch.from_numpy(d <= float(surface_max_dist)).to(keep.device)
    if keep_count is not None and int(keep_count) < int(keep.sum()):
        # Normalization floor = `prune_opa`, the lowest opacity a survivor can have.
        # It bounds how far a near-dead Gaussian can be promoted (≤ 1/prune_opa)
        # WITHOUT clamping anything that legitimately lives down there: a correct
        # glass pane converges to ≈0.012 per surfel (see TrainParams.prune_opa), so
        # the old fixed 0.02 floor under-ranked panes by ~1.7× — penalizing exactly
        # the translucent-by-design content this normalization exists to protect.
        alpha = torch.sigmoid(opacities.detach().flatten()).clamp_min(alpha_floor)
        rank = weights / alpha
        forced = blind & keep                         # blind survivors can't be ranked
        rankable = keep & ~forced
        n_extra = max(int(keep_count) - int(forced.sum()), 0)
        new_keep = forced.clone()
        if n_extra > 0 and int(rankable.sum()) > 0:
            r = rank.clone()
            r[~rankable] = float("-inf")
            top = torch.topk(r, min(n_extra, int(rankable.sum()))).indices
            new_keep[top] = True
        keep = new_keep
    return keep


def _heal(  # noqa: ANN001
    torch, F, raster, splats, views, K, width, height, params,
    scene_scale, steps, box_lo, box_hi, px_grid, py_grid, progress,
):
    """Short fine-tune of the compaction SURVIVORS (no densification) so
    neighbours absorb the deleted Gaussians' residual and the model doesn't tear.
    Fresh Adam (training already converged; moments aren't worth carrying), means
    LR damped by `compact_heal_means_lr_frac`. `box_lo`/… (a tile's box) mask the
    loss to owned pixels, matching how the tile trained; None trains unmasked."""
    import threading

    device = splats["means"].device
    b = max(int(params.batch), 1)
    lr_scale = float(np.sqrt(b))
    lrs = {
        "means": params.means_lr * scene_scale * lr_scale * params.compact_heal_means_lr_frac,
        "scales": params.scales_lr * lr_scale,
        "quats": params.quats_lr * lr_scale,
        "opacities": params.opacities_lr * lr_scale,
        "sh0": params.sh0_lr * lr_scale,
        "shN": params.shN_lr * lr_scale,
    }
    optimizers = {
        name: torch.optim.Adam([{"params": [splats[name]], "lr": lrs[name]}], eps=1e-15)
        for name in splats.keys()
    }
    window = _gaussian_window(torch, 11, 1.5, device, channels=3)
    stop_ev = threading.Event()
    stream = _view_stream(torch, views, device, b, params.prefetch, params.seed, 0, stop=stop_ev)
    try:
        for step in range(int(steps)):
            viewmats, Ks, gt_rgb, gt_alpha, gt_depth = next(stream)
            mask = None
            if box_lo is not None and gt_depth is not None:
                world = _unproject_depth(torch, gt_depth, viewmats, Ks, px_grid, py_grid)
                mask = _tile_mask(torch, box_lo, box_hi, gt_alpha, gt_depth, world)
            colors, sh_deg = _render_inputs(torch, splats, params.sh_degree)
            render = _render_batch(
                torch, raster, splats, colors, sh_deg, viewmats, Ks,
                width, height, params, dist_on=False,
            )
            loss = _supervision_loss(
                torch, F, params, window, gt_rgb, gt_alpha, gt_depth,
                render, splats["scales"], mask,
                normals_active=True, dist_active=False,
                scene_scale=scene_scale,
            )
            loss.backward()
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            if progress is not None and (step % max(params.log_every, 1) == 0 or step == int(steps) - 1):
                progress(
                    params.iterations, params.iterations,
                    f"compact/heal {step + 1}/{int(steps)} loss={float(loss):.4f} "
                    f"n={int(splats['means'].shape[0])}",
                )
    finally:
        stop_ev.set()


def _profile_report(prof, n_sampled, n_steps, wall_s, radii_px, params, n_final):  # noqa: ANN001
    """Turn the in-loop timings into a RANKED, actionable speed report.

    Returns `(summary, lines)` — `summary` lands in the run summary (status.json),
    `lines` are logged so the bottleneck is visible without digging. Every
    recommendation is gated on its own measurement, so this reports what THIS run
    was actually limited by instead of listing generic advice. Two clocks are
    involved: `data` is wall time the training thread spent BLOCKED waiting for the
    prefetcher (measured every step, exact, no sync needed), while the GPU sections
    are sync-bracketed on 1 step in `_PROFILE_EVERY` — accurate but only sampled,
    since a per-step `cuda.synchronize()` would itself distort the run."""
    step_ms = wall_s / max(n_steps, 1) * 1000.0
    data_ms = prof["data"] / max(n_steps, 1) * 1000.0
    s = max(n_sampled, 1)
    gpu = {k: prof[k] / s * 1000.0 for k in ("render", "loss", "backward", "optim", "strategy")}
    sampled_ms = max(sum(gpu.values()), 1e-9)
    aa_ms = prof["aa"] / max(n_steps, 1) * 1000.0
    ckpt_ms = prof["ckpt"] / max(n_steps, 1) * 1000.0
    data_frac = data_ms / max(step_ms, 1e-9)
    rad = {}
    if radii_px:
        a = np.asarray(radii_px, dtype=np.float64)
        rad = {"median_px": float(np.median(a)), "p90_px": float(np.percentile(a, 90))}

    summary = {
        "step_ms": round(step_ms, 2),
        "views_per_s": round(n_steps * max(params.batch, 1) / max(wall_s, 1e-9), 2),
        "data_wait_ms": round(data_ms, 2),
        "data_wait_frac": round(data_frac, 3),
        "gpu_ms_sampled": {k: round(v, 2) for k, v in gpu.items()},
        "aa_ms": round(aa_ms, 3),
        "ckpt_ms": round(ckpt_ms, 3),
        "projected_radius": {k: round(v, 1) for k, v in rad.items()},
        "samples": n_sampled,
    }

    L = [
        "=" * 72,
        f"SPEED PROFILE — {n_steps:,} steps in {_fmt_hms(wall_s)} "
        f"({step_ms:.1f} ms/step, {summary['views_per_s']:.1f} views/s, n={n_final:,})",
        f"  data wait (blocked on loader) {data_ms:7.1f} ms/step  {data_frac * 100:5.1f}% of wall",
        f"  --- GPU sections, sampled every {_PROFILE_EVERY} steps ({n_sampled} samples) ---",
    ]
    for k, v in sorted(gpu.items(), key=lambda kv: -kv[1]):
        L.append(f"  {k:<29}{v:7.1f} ms/step  {v / sampled_ms * 100:5.1f}% of GPU time")
    L.append(f"  anti-alias floor (amortized)  {aa_ms:7.1f} ms/step")
    L.append(f"  checkpointing (amortized)     {ckpt_ms:7.1f} ms/step")
    if rad:
        L.append(
            f"  projected Gaussian radius: median {rad['median_px']:.0f} px, "
            f"p90 {rad['p90_px']:.0f} px (tile size 16 px)"
        )

    recs: list[str] = []
    # 1. Loader. `data` is exact, so this verdict is not a guess.
    if data_frac > 0.25:
        recs.append(
            f"LOADER-BOUND ({data_frac * 100:.0f}% of wall clock blocked on the prefetcher). "
            f"`_view_stream` runs ONE decode thread; a pool of 4 should recover most of "
            f"{data_ms:.0f} ms/step, i.e. ~{data_ms * 0.75 * n_steps / 1000 / 60:.0f} min of this run."
        )
    else:
        recs.append(
            f"Loader is NOT the bottleneck ({data_frac * 100:.0f}% of wall). Parallelizing the "
            f"prefetch decode would buy at most {data_ms * n_steps / 1000 / 60:.1f} min — skip it."
        )
    # 2. Waste in the input path — only worth naming if the input path costs
    #    anything. Listing these under a "loader is not the bottleneck" verdict
    #    would contradict the line above and send someone optimizing 0.6% of a run.
    if data_frac > 0.05:
        if params.depth_lambda <= 0.0 and not params.depth_densify:
            recs.append(
                "FREE: the depth plane is decoded + uploaded every view but NOTHING consumes it "
                "(depth_lambda=0, depth_densify off). Skipping it saves ~2.7 ms/view of CPU and "
                "4.2 MB/view of PCIe."
            )
        recs.append(
            "FREE: reference planes are cast to float32 on the CPU before upload (21.0 MB/view). "
            "Uploading uint8 RGBA + uint16 depth codes and casting on the GPU is 3.3x less PCIe "
            "and bit-identical for colour."
        )
    # 3. Rasterization, and the quality lever that shares its fix.
    raster_frac = (gpu["render"] + gpu["backward"]) / sampled_ms
    if raster_frac > 0.5:
        # Qualify the headline: "most of GPU time" only means "most of the run"
        # when the GPU is what the run is waiting on.
        lead = (
            f"RASTERIZATION-BOUND ({raster_frac * 100:.0f}% of GPU time in render+backward)"
            if data_frac <= 0.25
            else f"Within GPU time (only {(1 - data_frac) * 100:.0f}% of wall here), "
                 f"render+backward is {raster_frac * 100:.0f}% — this becomes the ceiling "
                 "once the loader is fixed"
        )
        msg = (
            f"{lead}. {params.representation.upper()} cost scales with covered pixels x "
            "blend depth, not primitive count."
        )
        if rad and rad["median_px"] > 16.0:
            msg += (
                f" The median Gaussian covers {rad['median_px']:.0f} px vs a 16 px tile, so each "
                "one touches many tiles. `refine_scale2d_stop_iter` (gsplat's screen-space split "
                "criterion) is currently 0 = OFF, so oversized Gaussians can only CLONE at the "
                "same size. Enabling it makes them smaller: faster AND sharper."
            )
        recs.append(msg)
    if gpu["optim"] / sampled_ms > 0.15:
        recs.append(
            f"Optimizer is {gpu['optim'] / sampled_ms * 100:.0f}% of GPU time — Adam over "
            f"{n_final:,} x 59 floats. Degree-3 SH is 45 of those 59; sh_degree=2 would cut the "
            "parameter state ~40% if the scene's view-dependence allows it."
        )
    if gpu["strategy"] / sampled_ms > 0.10:
        recs.append(
            f"Densification bookkeeping is {gpu['strategy'] / sampled_ms * 100:.0f}% of GPU time; "
            "raising `refine_every` above 100 trades a little adaptivity for it."
        )
    if aa_ms > 2.0:
        recs.append(
            f"The anti-alias floor costs {aa_ms:.1f} ms/step amortized — it round-trips every mean "
            "to the CPU for a KD-tree query. Raise `aa_every` or move it to the GPU."
        )

    L.append("  --- what to change ---")
    L += [f"  {i + 1}. {r}" for i, r in enumerate(recs)]
    L.append("=" * 72)
    summary["recommendations"] = recs
    return summary, L


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
    """ONE gsplat optimization, in `params.representation` — the whole scene, or one
    tile — returning
    `(trained arrays as numpy, counters)`. `tile_box` ((lo, hi) world corners of
    the tile's expanded region) masks every loss to pixels this run OWNS (its own
    surface + true background — `_tile_mask`) and clips depth-seeding to the box;
    None trains unmasked, the single-run behavior. Checkpoints under `ckpt_dir`
    (the caller owns cleanup)."""
    from gsplat import DefaultStrategy
    from gsplat.strategy.ops import reset_opa

    raster = _rasterizer(params)
    device = torch.device("cuda")
    n_views = len(views)
    n_init = int(init["means"].shape[0])

    # Resume from the latest checkpoint when present + compatible, else build from
    # the surfel init. Two incompatibilities are rejected rather than crashed on:
    #   * a legacy FLAT checkpoint (no `shN` param) against this degree-3 model;
    #   * a checkpoint from the OTHER representation — which no shape check could
    #     catch, since the parameter tensors are identically shaped and only the
    #     MEANING of the third scale column differs (1% decoy vs real thickness).
    #     Checkpoints written before this field existed are 2DGS by definition.
    meta = {"n_init": n_init, "representation": params.representation}
    ckpt = _load_checkpoint(torch, ckpt_dir, device) if resume else None
    if ckpt is not None and (
        "shN" not in ckpt.get("params", {})
        or (ckpt.get("meta") or {}).get("representation", "2dgs") != params.representation
    ):
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
                "shN": torch.nn.Parameter(torch.from_numpy(init["shN"])),
            }
        ).to(device)

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

    # `DefaultStrategy.reset_every` is NOT the opacity-reset cadence here: in the
    # pinned gsplat 1.5.3 its only live use is `_prune_gs`'s `step > reset_every`
    # gate on the scale prune (the wheel's own reset trigger is dead code — see
    # TrainParams.reset_every), so we pass the size-prune warm-up in that slot and
    # run the reset ourselves below. Gaussian size is therefore bounded for the
    # whole refine window whether or not resets are enabled. It must stay ≥ 1: the
    # dead trigger still evaluates `step % reset_every`.
    strategy = DefaultStrategy(
        prune_opa=params.prune_opa,
        grow_grad2d=params.resolved_grow_grad2d,
        grow_scale3d=params.grow_scale3d,
        prune_scale3d=params.prune_scale3d,
        # Screen-space split (see TrainParams). `refine_scale2d_stop_iter` arms
        # both the split and the prune arm, so `prune_scale2d` is passed
        # neutralized — subdividing oversized Gaussians is the point, deleting
        # them is not.
        refine_scale2d_stop_iter=params.refine_scale2d_stop_iter,
        grow_scale2d=params.grow_scale2d,
        prune_scale2d=params.prune_scale2d,
        refine_start_iter=params.refine_start_iter,
        refine_stop_iter=params.resolved_refine_stop,
        reset_every=max(params.prune_scale_start_iter, 1),
        refine_every=params.refine_every,
        # WHICH 2D GRADIENT densification reads, and whether it may use the
        # ABSOLUTE one. The 3DGS rasterizer's densification tensor is `means2d`,
        # which its backward also decorates with `.absgrad` — so AbsGS is live
        # there. The 2DGS path densifies off a separate `gradient_2dgs` tensor that
        # never receives one, so absgrad resolves to False and would raise if forced
        # (see TrainParams.absgrad / grow_grad2d).
        absgrad=params.resolved_absgrad,
        key_for_gradient="means2d" if params.is_3dgs else "gradient_2dgs",
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
    # The distortion map is a 2DGS-only rasterizer output, and asking for it costs
    # an extra buffer — so the request itself is gated on the representation, not
    # just the weight.
    dist_on = params.dist_lambda > 0.0 and not params.is_3dgs

    # Nearest-camera KD-tree for the anti-aliasing scale floor (cameras are fixed,
    # so build once); `focal` (= fl_x) converts a camera distance to a pixel span.
    cam_tree = None
    focal = float(K[0, 0].item())
    if params.antialias and n_views > 0:
        from scipy.spatial import cKDTree

        cam_tree = cKDTree(centers)
    n_seeded = 0
    # Latch for the reversible growth guards below, so the pause/resume transition
    # is logged once instead of every step it holds.
    growth_frozen = False

    # Speed profile (see `_profile_report`). `data` is exact per-step wall time
    # blocked on the prefetcher; the GPU sections are sync-bracketed on 1 step in
    # `_PROFILE_EVERY`. `_lap` is only ever called on those sampled steps.
    prof = dict.fromkeys(("data", "render", "loss", "backward", "optim", "strategy", "aa", "ckpt"), 0.0)
    prof_n = 0
    radii_px: list[float] = []
    _mark = 0.0

    def _lap(key: str) -> None:
        nonlocal _mark
        torch.cuda.synchronize()
        now = time.perf_counter()
        prof[key] += now - _mark
        _mark = now

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

    # "Record both" reference: a FROZEN copy of the init cloud, whose EXPECTED
    # depth is the true two-layer α-weighted depth (it sits on the glass panes at
    # α≈0.065). Rendered per batch (no grad) and used two ways, so it is built when
    # EITHER wants it: as the positive target of the expected-depth term
    # (`depth_expected_lambda`), and as the transmissive-pixel detector that keeps
    # the opaque-plane depth term off glass when the depth statistic is expected
    # (`resolved_glass_guard` — i.e. always on the 3DGS path). One render serves
    # both.
    ref_splats = ref_colors = None
    if params.depth_expected_lambda > 0.0 or params.resolved_glass_guard:
        ref_splats = {
            k: torch.from_numpy(np.ascontiguousarray(init[k])).to(device)
            for k in ("means", "scales", "quats", "opacities", "sh0", "shN")
        }
        ref_colors, _ = _render_inputs(torch, ref_splats, 0)

    for step in range(start_step, params.iterations):
        _t_data = time.perf_counter()
        viewmats, Ks, gt_rgb, gt_alpha, gt_depth = next(stream)
        prof["data"] += time.perf_counter() - _t_data
        sample = (step - start_step) % _PROFILE_EVERY == 0
        if sample:
            torch.cuda.synchronize()
            _mark = time.perf_counter()

        # Tiled runs supervise only pixels this tile OWNS: its surface + true
        # background (exact ownership from the reference depth).
        mask = None
        if box_lo is not None and gt_depth is not None:
            world = _unproject_depth(torch, gt_depth, viewmats, Ks, px_grid, py_grid)
            mask = _tile_mask(torch, box_lo, box_hi, gt_alpha, gt_depth, world)

        active_sh = min(params.sh_degree, step // max(params.sh_degree_interval, 1))
        colors, sh_deg = _render_inputs(torch, splats, active_sh)
        render = _render_batch(
            torch, raster, splats, colors, sh_deg, viewmats, Ks, width, height, params, dist_on
        )
        info = render.info
        if sample:
            _lap("render")
            # Screen-space size drives rasterizer cost (covered pixels x blend
            # depth) and is also what decides clone-vs-split, so it feeds both halves
            # of the report. `info["radii"]` is [...,2] in pixels; 0 = not visible.
            with torch.no_grad():
                r = info["radii"].detach().amax(dim=-1).flatten()
                r = r[r > 0]
                if r.numel():
                    radii_px.append(float(r.median()))
        # The frozen-cloud target for the glass-maker term / the glass guard. The
        # cloud's own rendered alpha gates out its sampling holes, where its expected
        # depth would be ill-defined.
        ed_target = ed_gate = None
        if ref_splats is not None and gt_depth is not None:
            with torch.no_grad():
                ref = _render_batch(
                    torch, raster, ref_splats, ref_colors, 0,
                    viewmats, Ks, width, height, params, False,
                )
            ed_target = ref.expected
            ed_gate = ref.alpha > 0.5

        if params.refine:
            strategy.step_pre_backward(splats, optimizers, strat_state, step, info)

        # Photometric L1 + D-SSIM, alpha, alpha-gated depth (the opaque plane +
        # expected vs the two-layer cloud target), and the representation's geometry
        # prior — the shared supervision loss. Under a tile mask every term is
        # restricted to pixels this tile OWNS (its surface + true background), so
        # boundary Gaussians never chase content a neighbour owns.
        loss = _supervision_loss(
            torch, F, params, window, gt_rgb, gt_alpha, gt_depth,
            render, splats["scales"], mask,
            normals_active=step >= params.normal_start_iter,
            dist_active=dist_on and step >= params.dist_start_iter,
            depth_active=step >= params.depth_start_iter,
            scene_scale=scene_scale,
            ed_target=ed_target, ed_gate=ed_gate,
        )
        if sample:
            _lap("loss")

        loss.backward()
        if sample:
            _lap("backward")
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()
        if sample:
            _lap("optim")

        if params.refine:
            strategy.step_post_backward(
                splats, optimizers, strat_state, step, info, packed=params.resolved_packed
            )
            # Opacity reset at the reference cadence, under upstream's FIXED
            # trigger (`step % reset_every == 0 and step > 0`, refine window
            # only) — executed here because the pinned gsplat 1.5.3 wheel's own
            # trigger never fires (precedence bug; see TrainParams.reset_every).
            # gsplat's reset_opa clamps every opacity to 2·prune_opa and zeroes
            # its Adam moments; only image-justified Gaussians re-earn opacity,
            # the rest drop below prune_opa and are pruned. Reading the LIVE
            # strategy.refine_stop_iter keeps resets coupled to the refine
            # window even after the VRAM guard freezes it.
            if (
                params.reset_every > 0
                and 0 < step < strategy.refine_stop_iter
                and step % params.reset_every == 0
            ):
                reset_opa(
                    params=splats, optimizers=optimizers, state=strat_state,
                    value=params.reset_opa_value,
                )
                if progress is not None:
                    progress(
                        step + 1, params.iterations,
                        f"opacity reset @{step} (ceiling {params.reset_opa_value:.2f}) "
                        f"n={len(splats['means'])}",
                    )
            # GROWTH guards — the optional count ceiling and the VRAM free-margin
            # (the latter after one empty_cache retry to release reusable cached
            # blocks, so growth can't collapse throughput into PCIe paging).
            # Neither may end refinement outright: gsplat's `step_post_backward`
            # returns early once `step >= refine_stop_iter`, so moving that (what
            # this used to do) stops PRUNING as well and freezes the model around
            # whatever it was holding — including the size prune, leaving runaway
            # Gaussians permanently. Raising the growth threshold out of reach
            # instead leaves `_prune_gs` running every `refine_every`, which is
            # precisely what you want under pressure since pruning gives memory
            # back. Both guards are REVERSIBLE: growth resumes once the count or
            # the free margin recovers.
            if step < strategy.refine_stop_iter:
                over_cap = params.cap_max is not None and len(splats["means"]) > params.cap_max
                tight_vram = False
                if not over_cap and params.vram_min_free_gb > 0.0:
                    need = params.vram_min_free_gb * 1e9
                    if torch.cuda.mem_get_info()[0] < need:
                        torch.cuda.empty_cache()
                        tight_vram = torch.cuda.mem_get_info()[0] < need
                freeze_growth = over_cap or tight_vram
                strategy.grow_grad2d = (
                    float("inf") if freeze_growth else params.resolved_grow_grad2d
                )
                if freeze_growth != growth_frozen and progress is not None:
                    why = (
                        f"count ceiling {params.cap_max:,} reached" if over_cap
                        else f"free VRAM < {params.vram_min_free_gb}GB" if tight_vram
                        else "pressure cleared"
                    )
                    progress(
                        step + 1, params.iterations,
                        f"{'pausing' if freeze_growth else 'resuming'} densification "
                        f"@{step} ({why}) n={len(splats['means'])} - pruning stays on",
                    )
                growth_frozen = freeze_growth
        if sample:
            _lap("strategy")
            prof_n += 1

        # Depth-guided densification: seed Gaussians at reference surfaces the splat
        # is missing (holes / too-far). After the strategy so this step's `info` is
        # never used against the grown tensors; inside the densification window and
        # only while VRAM allows growth.
        if (
            params.depth_densify
            and params.depth_densify_start <= step < strategy.refine_stop_iter
            and (step + 1) % max(params.depth_densify_every, 1) == 0
            and (params.cap_max is None or len(splats["means"]) < params.cap_max)
        ):
            tight = (
                params.vram_min_free_gb > 0.0
                and torch.cuda.mem_get_info()[0] < 1.5 * params.vram_min_free_gb * 1e9
            )
            new = (
                None if tight
                else _depth_seed(
                    torch, gt_rgb, gt_alpha, gt_depth, render.alpha.detach(),
                    render.depth.detach(), viewmats, K, params, int(params.depth_densify_max),
                    box=(box_lo, box_hi) if box_lo is not None else None,
                )
            )
            if new is not None:
                _append_gaussians(torch, splats, optimizers, strat_state, new)
                n_seeded += int(new["means"].shape[0])

        # Anti-aliasing floor: keep every Gaussian resolvable by its nearest camera
        # so it can't shimmer across the scale-ladder octaves under free-fly.
        if params.antialias and cam_tree is not None and (step + 1) % max(params.aa_every, 1) == 0:
            _t_aa = time.perf_counter()
            _aa_scale_floor(
                torch, splats, cam_tree, focal, params.aa_min_scale_px,
                tangent_only=not params.is_3dgs,
            )
            prof["aa"] += time.perf_counter() - _t_aa

        # Divergence guard — free, because the log below already syncs on the loss.
        # Without it a NaN propagates silently into the params and surfaces much
        # later somewhere unrelated (a non-finite mean crashes `_aa_scale_floor`'s
        # KD-tree up to `aa_every` steps on), which hides both the step it began and
        # the term responsible.
        if step % log_every == 0 and not np.isfinite(float(loss)):
            raise RuntimeError(
                f"stage6: loss went non-finite at step {step} "
                f"(n={len(splats['means'])}, scene_scale={scene_scale:.2f}) — training "
                "diverged. Bisect the loss terms: depth_lambda first (1/d is the only "
                "term with a singularity), then alpha_lambda, then the LRs."
            )

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
            _t_ck = time.perf_counter()
            _save_checkpoint(
                torch, ckpt_dir, step, splats, optimizers, means_sched,
                strat_state, meta, params.ckpt_keep,
            )
            prof["ckpt"] += time.perf_counter() - _t_ck

    # Release the prefetch worker (tiled runs create one stream per tile).
    stop_ev.set()

    # Stage 6 emits the RAW optimization output. Delete + heal + final-prune is
    # now its OWN stage (`heal_splat`, Stage 7) operating on the merged
    # `trained.ply`, so `trained.ply` is the un-cleaned model and `healed.ply` the
    # deliverable — the two viewable side by side. (The per-tile compaction that
    # used to run here is gone with the fold-out; Stage 7 heals globally.)
    with torch.no_grad():
        arrays = {k: v.detach().cpu().numpy() for k, v in splats.items()}

    # Where the wall clock actually went, and what to do about it. Logged as well
    # as returned, so it is readable in the live heartbeat and durable in the
    # recorded summary.
    train_s = time.perf_counter() - t_start
    n_ran = max(params.iterations - start_step, 1)
    profile, lines = _profile_report(
        prof, prof_n, n_ran, train_s, radii_px, params, int(arrays["means"].shape[0])
    )
    if progress is not None:
        for line in lines:
            progress(params.iterations, params.iterations, line)

    info = {
        "init": n_init,
        "final": int(arrays["means"].shape[0]),
        "seeded": n_seeded,
        "pruned": 0,             # moved to Stage 7 (heal_splat)
        "compacted": 0,          # moved to Stage 7 (heal_splat)
        "compact_search": None,
        "views": n_views,
        "train_s": round(train_s, 1),
        "profile": profile,
    }
    return arrays, info


def _final_prune(torch, splats, prune_opa: float, prune_scale3d: float, scene_scale: float) -> int:  # noqa: ANN001
    """One-shot cleanup before export: drop Gaussians below `prune_opa` opacity (the
    low-opacity floaters left after densification stopped) and, when `prune_scale3d
    > 0`, any whose largest axis exceeds `prune_scale3d·scene_scale` (runaway
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


def _evaluate(torch, raster, splats, views, K, width, height, params, device):  # noqa: ANN001
    """Mean PSNR (foreground), RGB L1, and alpha-gated depth L1 over a random
    subset of views (capped by `eval_max_views` — plans can have thousands). Renders
    through `_render_batch`, so the metric is measured with the exact operator the
    model was trained under — which is also what makes the compaction pass's dB
    gate meaningful, since it bisects on this number."""
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
            render = _render_batch(
                torch, raster, splats, colors, sh_deg,
                v["viewmat"].to(device).unsqueeze(0),
                v["K"].to(device).unsqueeze(0), width, height, params,
                dist_on=False,
            )
            pred_rgb = render.rgb.clamp(0, 1)
            pred_depth = render.depth
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
    """Standalone CLI for the GPU box: fine-tune a splat from a COLMAP model —
    the Postshot-style (point cloud + poses + images) folder that
    `splat_to_colmap.py` / `splat.colmap.export_colmap` writes."""
    import argparse

    ap = argparse.ArgumentParser(description="Stage 6 — gsplat fine-tune from a COLMAP model")
    ap.add_argument(
        "--representation", choices=REPRESENTATIONS, default=TrainParams.representation,
        help="primitive to train: '2dgs' = surface-aligned surfels (best geometry, "
             "median-depth + normal-consistency supervision); '3dgs' = full 3D "
             "Gaussians (AbsGS densification, antialiased rasterization, and the "
             "primitive every delivery viewer actually renders)",
    )
    ap.add_argument(
        "--colmap", required=True, type=Path,
        help="COLMAP model dir (cameras.txt / images.txt / points3D.txt + RGB "
             "images) — the folder splat_to_colmap.py writes",
    )
    ap.add_argument("--out", required=True, type=Path, help="output trained.ply")
    ap.add_argument(
        "--init", choices=("surfels", "points"), default=TrainParams.init,
        help="init source: 'surfels' = the Stage-3 cloud.ply via --init-ply "
             "(geometry starts AT the 2DGS solution — far fewer epochs); "
             "'points' = gsplat's from-points3D recipe (Postshot-parity A/B)",
    )
    ap.add_argument(
        "--init-ply", type=Path, default=None,
        help="the Stage-3 surfel cloud (splat/cloud.ply) — required for --init surfels",
    )
    ap.add_argument(
        "--iterations", type=int, default=TrainParams.iterations,
        help="OPTIMIZER STEPS (gsplat max_steps / PostShot's step box), and the "
             "length every cadence is written against",
    )
    ap.add_argument(
        "--epochs", type=float, default=TrainParams.epochs,
        help="size the run as passes over the view set instead "
             "(steps = epochs × n_views / batch); rescales every cadence to match",
    )
    ap.add_argument(
        "--refine-stop-iter", type=int, default=None,
        help="stop densification at this step (default: 50%% of iterations)",
    )
    ap.add_argument(
        "--batch", type=int, default=TrainParams.batch,
        help="views averaged per optimizer step (gsplat batch_size); MULTIPLIES "
             "work — the step count is --iterations either way",
    )
    ap.add_argument(
        "--init-opa", type=float, default=TrainParams.init_opa,
        help="initial Gaussian opacity for the point-cloud init (gsplat default 0.1)",
    )
    ap.add_argument(
        "--init-scale", type=float, default=TrainParams.init_scale,
        help="scale multiplier on the KNN-derived initial Gaussian size",
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
        "--sh-degree", type=int, default=TrainParams.sh_degree,
        help="max spherical-harmonics degree for view-dependent colour "
             "(PostShot 'Max SH Degree'; default 3, 0 = flat/view-independent)",
    )
    ap.add_argument(
        "--tile-max", type=int, default=None,
        help="max init Gaussians for a single run; larger clouds train as ground-"
             f"plane tiles and merge (default {_TILE_BUDGET_DEFAULT:,}; 0 disables tiling)",
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
        colmap_dir=args.colmap,
        out_path=args.out,
        init_ply=args.init_ply,
        params=TrainParams(
            representation=args.representation,
            init=args.init,
            iterations=args.iterations,
            epochs=args.epochs,
            refine_stop_iter=args.refine_stop_iter,
            batch=args.batch,
            init_opa=args.init_opa,
            init_scale=args.init_scale,
            ckpt_every=args.ckpt_every,
            vram_min_free_gb=args.vram_min_free_gb,
            antialias=args.antialias,
            sh_degree=args.sh_degree,
            tile_max=args.tile_max,
        ),
        resume=args.resume,
        progress=_log,
    )
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    _main()
