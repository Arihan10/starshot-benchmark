"""Stage 4 — Coverage camera planner (scale-ladder demands + greedy set-cover).

Picks the fewest reference IMAGES — (position, cube-face) pairs in free space —
so every visible surface patch is seen from every suppliable DIRECTION CELL
(equal-solid-angle bins below) AND once per OCTAVE of viewing scale (the mip
ladder below). The output feeds Stage 5 (reference renders) and yields the
occlusion-cull list as a byproduct.

CONNECTED (Option A): Stage 4 consumes the outputs of the earlier stages and loads
NO meshes and computes NO occupancy of its own —
  * **Stage 2 free-space grid** (`freespace.npz`): candidate camera positions
    (reachable free cells) + the single-grid occupancy the line-of-sight
    ray-march uses. No re-voxelization.
  * **Stage 3 surfel cloud** (`cloud.ply`): the patch source. Patches are a
    UNIFORM thinning of the surfels, so their density INHERITS the cloud's own
    feature-adaptive sampling (denser over BOTH geometry and texture detail)
    rather than re-detecting it. Surfel normals were already oriented to free
    space in Stage 3 — so the facing test is reliable regardless of the winding.

THE SCALE LADDER (distance is a resolution variable, not a knob). A patch of
feature size s viewed from effective distance d spans `s·focal_px/d` pixels, so
distance IS resolution: each factor-of-2 of distance is one mip level of the
signal. The plan therefore demands, per patch, one view per OCTAVE (factor-of-2
band) of effective distance over

    d_eff ∈ [ s·focal_px / finest_px , s·focal_px ]

— from `finest_px` pixels-per-feature (the ONE scale-quality dial; its ceiling is
the render resolution, where a single patch fills the frame and views saturate the
representation) down to the canonical 1-px observability limit. Both endpoints
derive from the patch's own feature scale and the shared render intrinsics — there
is no minimum or maximum view distance anywhere, so the same parameters plan a
closet or a city. Trained this way, the splat has references at every scale a
free-fly viewer can render it from.

EFFECTIVE DISTANCE (incidence folded into scale): a view at physical distance
`dist` and incidence θ off the patch normal (two-sided — winding-agnostic)
resolves the patch like a head-on view at `d_eff = dist/|cos θ|`. Grazing views
aren't rejected by an angular cutoff; they just supply coarser octaves, until
they pass the 1-px limit and stop counting entirely. (In-plane grazing rays along
a flat wall die in the occlusion ray-march anyway — they run through the wall's
own surface voxels.)

THE PATCH MIP PYRAMID (keeps the ladder tractable): at octave o the image
resolves detail at 2^o × the base scale, so demands only need to exist at 2^o ×
coarser patch spacing — patch p is demanded at octaves 0..oct_top[p] with
P(oct_top ≥ o) = 4^-o, a NESTED thinning (total demands ≈ 4/3 × patches). Each
owed octave is supplied demand-major (below): a patch emits a few head-on aim
points only for the octaves it actually owes, so coarse octaves cost a handful
of aim points over exponentially fewer patches.

DIRECTION CELLS (equal-solid-angle bins; replaces azimuth-only sectors): view
directions are binned on the folded hemisphere around each patch normal into
`angular_bins` equal-area cells — an explicit polar CAP (the direct-facing
field) + one ring of azimuth cells (`_bin_of`) — so elevation diversity is
enforced and near-normal views land deterministically. The demand is every
SUPPLIABLE cell. Direction supply is scale-free (any resolving view counts), and
comes from each patch's APERTURE aim points — a shell at ~1.5·d_min in
hemisphere directions around the normal (`_build_coverage`). A shell (not a
single fine ray) is what supplies ring cells reliably: a lone fine-band camera
sits at dist = d_eff·cosθ, which for grazing elevations collapses onto the
surface itself. An aperture pair whose octave the pyramid doesn't demand carries
`owed = False`: it supplies a direction cell without inflating scale demands.

Pipeline:
  1. Read the surfel cloud → points + oriented normals.
  2. PATCHES: uniformly thin the surfels (their density is already feature-
     adaptive from Stage 3 — geometry + texture), reading the local surfel
     spacing as each patch's ladder feature scale; denser where the cloud is.
  3. CANDIDATE positions = reachable free cells (Stage 2), subsampled denser
     where clearance is small; the greedy's own scale demands keep chosen
     cameras off surfaces (no demand exists below a patch's d_min).
  4. COVERAGE (demand-major): each patch asks directly for its few suppliers —
     a bounded set of aim points (an aperture SHELL around the patch normal for
     direction cells + the finest octave, plus head-on points per owed coarse
     octave) each snapped to the nearest candidate and verified by the fine-grid
     ray-march — instead of enumerating every candidate in range. Work and memory
     scale with DEMANDS (patches × cells), not free volume × surface. Each
     surviving pair carries its direction cell + supplied octave.
  5. GREEDY multicover over IMAGES — (candidate, face) units, the true cost unit
     of Stages 5/6 — until every visible patch hits every suppliable direction
     cell + every suppliable owed octave. Faces that add nothing are never
     selected (emergent face culling).

Output: `cameras.json` — a shared cube-face `intrinsics` block (90° FOV, render
resolution, the scale-ladder anchors), the six `cube_faces`, and the chosen
cameras each carrying exactly its SELECTED faces + coverage — plus `patches.bin`
(packed float32 [x,y,z, nx,ny,nz, feature_scale, bins_seen] per patch) and a
summary. CUBEMAP-NATIVE: each position renders only its selected 90° pinhole
faces in Stage 5, so no single look direction is emitted.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil, log2, radians, tan
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from splat.stage2 import FreeSpace, load_free_space

logging.getLogger("trimesh").setLevel(logging.ERROR)

CAMERAS_NAME = "cameras.json"
PATCHES_NAME = "patches.bin"
PATCH_VIEWS_NAME = "patch_views.json"

# SH degree-0 basis constant (matches Stage 3): colour = 0.5 + C0 * f_dc.
_SH_C0 = 0.28209479177387814

# Rescue-pass budget (see `_rescue_pairs`): nearest candidates tried per unseen
# patch, and the flat pair count per ray-march slice.
_RESCUE_K = 32
_RESCUE_SLICE = 200_000

# progress(done, total, current_id) — called during the coverage build.
ProgressCb = Callable[[int, int, str], None]

# The six cube-map faces a camera POSITION renders in Stage 5: outward view
# directions (with a non-degenerate up vector for each, incl. the ±Y poles) in the
# repo's Y-up, right-handed, metres frame. Shared with Stage 5 so the render poses
# match the plan exactly; the tuple order fixes the face index `_face_of` returns.
CUBE_FACE_NAMES = ("+x", "-x", "+y", "-y", "+z", "-z")
CUBE_FACES: dict[str, dict[str, list[float]]] = {
    "+x": {"forward": [1.0, 0.0, 0.0], "up": [0.0, 1.0, 0.0]},
    "-x": {"forward": [-1.0, 0.0, 0.0], "up": [0.0, 1.0, 0.0]},
    "+y": {"forward": [0.0, 1.0, 0.0], "up": [0.0, 0.0, -1.0]},
    "-y": {"forward": [0.0, -1.0, 0.0], "up": [0.0, 0.0, 1.0]},
    "+z": {"forward": [0.0, 0.0, 1.0], "up": [0.0, 1.0, 0.0]},
    "-z": {"forward": [0.0, 0.0, -1.0], "up": [0.0, 1.0, 0.0]},
}


@dataclass(frozen=True)
class PlanParams:
    """Stage-4 knobs (overview §9). Occupancy pitch/margin live in the Stage-2
    grid, not here. Scale is NOT a knob: all view distances derive per patch from
    the scale ladder (module docstring) — `finest_px_per_patch` is the only scale
    dial, and it's a resolution, not a distance, so the defaults are scene-size
    independent (the old view_dist_min/max, min_px_per_patch and near_frac knobs
    are subsumed by the ladder). Camera–surface standoff is not a knob either
    (the old collision_clearance is gone): the ladder demands nothing below a
    patch's d_min, so the greedy keeps cameras off open surfaces by itself, and
    near-wall picks emerge only where occlusion leaves no other supplier."""

    # Patch FEATURE-SCALE band (the scale ladder's floor/ceiling). Detail is NOT
    # re-detected here: the local surfel spacing — Stage 3's own feature-adaptive
    # density, dense over BOTH geometry and texture — is mapped into [s_min, s_max]
    # to drive each patch's ladder, and patches are a UNIFORM thinning of the
    # surfels (rate (s_min/s_max)²) so their density RIDES the cloud's (denser
    # surfels → more patches) instead of a separate curvature/texture model. s_min
    # also sets the effective camera standoff: d_min = s_min·focal/finest_px is the
    # closest any demand sits, so the greedy never places a camera nearer than that.
    patch_min_spacing: float = 0.06   # s_min (m): finest patch feature scale
    patch_max_spacing: float = 0.30   # s_max (m): coarsest patch feature scale
    # The angular-quality dial: the number of EQUAL-SOLID-ANGLE direction cells
    # tiling the (folded, two-sided) hemisphere around each patch normal — cell 0
    # is an explicit polar CAP (|cos θ| ≥ 1 − 1/B: the direct-facing field, 1/B of
    # the hemisphere by area), cells 1..B-1 split the remaining ring azimuthally.
    # The demand is every SUPPLIABLE cell, so elevation diversity is enforced and
    # near-normal views land in a deterministic bin instead of an arbitrary
    # azimuth sector. Replaces the old angles_per_patch (K-of-N quota) +
    # angular_sectors (azimuth-only bins) pair.
    angular_bins: int = 3
    # The scale-quality dial: the sharpest demanded resolution, in pixels spanned
    # by one patch feature. The ladder runs from here down to 1 px in octaves.
    # Ceiling = render_resolution (a patch fills the frame — views past that
    # saturate the representation); raising it buys close-up crispness at
    # O(4^octaves) more close cameras, so it is a real cost dial, not free.
    finest_px_per_patch: float = 64
    min_gain: int = 2                 # stop once the best image adds < this many demands
    # The image BUDGET — the product cost dial, in the unit the cost is actually
    # paid in (rendered/stored/trained reference images). None = run to
    # completion of every suppliable demand; a number stops the greedy there.
    # Because greedy coverage is submodular, the first N picks are near-optimal
    # for ANY N, and seeded tie-break noise keeps a budget cut spatially uniform
    # (equal-gain plateaus would otherwise be taken in candidate scan order —
    # visible as left-to-right sweeps — leaving one side of the scene denser
    # when cut). The summary's coverage `curve` reports what any budget buys.
    max_views: int | None = None
    # Candidate positions are drawn from the reachable FREE cells (see
    # `_candidates`) at ~`candidate_spacing` apart, so the candidate COUNT
    # scales with the free volume, not a fixed budget. Stage 4 reads no per-cell
    # clearance — the FREE mask already encodes the standoff floor.
    # `max_candidates` is only a SAFETY ceiling (even-downsample + warn if a
    # scene ever exceeds it), not the room-scale cap it used to be.
    candidate_spacing: float = 0.5    # target spacing (m) between candidate positions
    max_candidates: int = 200_000     # safety ceiling on candidate positions
    # Cube-face reference-render intrinsics (SHARED with Stage 5). FOV fixed at
    # 90° (six faces tile 360°); the ladder's distance bands are DERIVED from the
    # render resolution + finest_px_per_patch so coverage scales match what
    # Stage 5 renders.
    face_fov_deg: float = 90.0
    render_resolution: int = 1024
    seed: int = 0

    @property
    def focal_px(self) -> float:
        """Pinhole focal length in pixels of one cube face: (R/2) / tan(fov/2).
        At the fixed 90° FOV this is simply R/2."""
        return (self.render_resolution / 2.0) / tan(radians(self.face_fov_deg) / 2.0)

    @property
    def finest_px(self) -> float:
        """`finest_px_per_patch` clamped to [2, render_resolution] — the ladder's
        fine end. The ceiling is saturation: at `render_resolution` px/feature one
        patch fills the frame, past which a view holds no new signal."""
        return float(min(max(self.finest_px_per_patch, 2.0), float(self.render_resolution)))

    @property
    def n_octaves(self) -> int:
        """Bands in the scale ladder: one per factor-of-2 of effective distance
        between the finest demanded view (`finest_px` px/feature) and the 1-px
        observability limit."""
        return max(1, int(ceil(log2(self.finest_px))))

    @property
    def bins(self) -> int:
        """`angular_bins` clamped to [2, 63]: cell 0 is the cap, so at least one
        ring cell must exist; 63 bounds the int64 coverage bitmask."""
        return max(2, min(63, int(self.angular_bins)))

    def as_summary(self) -> dict[str, Any]:
        return {
            "patch_min_spacing": self.patch_min_spacing,
            "patch_max_spacing": self.patch_max_spacing,
            "angular_bins": self.angular_bins,
            "finest_px_per_patch": self.finest_px,
            "octaves": self.n_octaves,
            "max_views": self.max_views,
            "min_gain": self.min_gain,
            "candidate_spacing": self.candidate_spacing,
            "max_candidates": self.max_candidates,
            "face_fov_deg": self.face_fov_deg,
            "render_resolution": self.render_resolution,
        }


def _read_cloud(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a Stage-3 surfel `.ply` → (positions, unit normals, rgb in [0,1]). Handles
    both the 2DGS (16 float/vertex, two scales) and 3DGS (17, three scales) layouts:
    the first nine properties (xyz, normal, f_dc_0..2) are identical in both and are
    all Stage 4 needs — only the trailing scale count differs. Normals are already
    oriented to free space by Stage 3."""
    raw = Path(path).read_bytes()
    marker = b"end_header\n"
    i = raw.find(marker)
    if i < 0:
        raise ValueError(f"{path} is not a binary .ply cloud")
    header = raw[:i].decode("ascii", errors="replace")
    n = None
    stride = 0
    for line in header.splitlines():
        s = line.strip()
        if s.startswith("element vertex"):
            n = int(s.split()[-1])
        elif s.startswith("property "):
            stride += 1
    if n is None or stride < 9:
        raise ValueError(f"{path}: unexpected .ply header (n={n}, stride={stride})")
    body = np.frombuffer(raw[i + len(marker):], dtype="<f4")
    if body.size < n * stride:
        raise ValueError(f"{path}: truncated cloud ({body.size} < {n * stride})")
    arr = body[: n * stride].reshape(n, stride)
    pos = arr[:, 0:3].astype(np.float64)
    nrm = arr[:, 3:6].astype(np.float64)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    nrm = nrm / ln
    col = np.clip(0.5 + _SH_C0 * arr[:, 6:9], 0.0, 1.0).astype(np.float32)
    return pos, nrm, col


def _local_spacing(points: np.ndarray, k: int = 8) -> np.ndarray:
    """Per-surfel LOCAL SAMPLE SPACING: the mean distance to its `k` nearest
    surfels — i.e. the Stage-3 cloud's OWN density read back out. The sampler
    already made that density feature-adaptive over BOTH geometry (crease /
    boundary bands) and texture (albedo-complexity bands), so `dense == detailed`.
    Stage 4 reads this instead of re-detecting curvature / albedo variance from
    the cloud (the retired curvature_k / tex_k path), honouring Stage 3's contract
    that the cloud's local density IS the detail field."""
    n = len(points)
    if n < 2:
        return np.full(n, 1.0, dtype=np.float64)
    kk = min(k + 1, n)
    d, _ = cKDTree(points).query(points, k=kk, workers=-1)
    return d[:, 1:].mean(axis=1)


def _candidates(fs: FreeSpace, p: PlanParams) -> np.ndarray:
    """Candidate camera positions: reachable FREE cells sampled ~`candidate_spacing`
    apart, so the count scales with the free volume rather than a fixed budget.
    Standoff is fully baked into the FREE mask (Stage 2's one-voxel floor) and
    otherwise EMERGENT — the ladder demands nothing below a patch's d_min, so
    point-blank candidates only win where occlusion leaves no farther supplier,
    and the ray-march fails closed on segments too short to verify. Stage 4
    reads no per-cell clearance: `free_candidates` thins the raw FREE mask by
    spacing alone. If a scene exceeds `max_candidates`, evenly downsample (and
    warn) — never silently fall back to a room-scale cap. Returns (M,3) world
    points."""
    centers = fs.free_candidates(p.candidate_spacing)
    if len(centers) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if p.max_candidates and len(centers) > p.max_candidates:
        step = int(np.ceil(len(centers) / p.max_candidates))
        logging.warning(
            "stage4: %d free candidates at spacing %.2fm exceeds the "
            "max_candidates ceiling %d; even-downsampling by %dx (raise "
            "max_candidates or candidate_spacing to keep full density)",
            len(centers), p.candidate_spacing, p.max_candidates, step,
        )
        centers = centers[::step]
    return centers


def _tangent_frames(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-normal orthonormal tangent basis (t1, t2) so a viewing direction can be
    binned into an azimuth around the patch normal (the ring cells of `_bin_of`)."""
    ref = np.where(
        np.abs(normals[:, 2:3]) < 0.9,
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    t1 = np.cross(normals, ref)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-9
    t2 = np.cross(normals, t1)
    return t1.astype(np.float32), t2.astype(np.float32)


def _bin_of(z: np.ndarray, az: np.ndarray, b: int) -> np.ndarray:
    """EQUAL-SOLID-ANGLE direction cell of each view on the folded hemisphere:
    `z` = |cos θ| to the patch normal (two-sided — a direction and its mirror
    share a cell, matching the winding-agnostic facing test; mirroring flips only
    the normal component, so the azimuth `az` is fold-invariant), `b` = cell
    count. Cell 0 is the polar CAP `z ≥ 1 − 1/b` — the explicit direct-facing
    field, exactly 1/b of the hemisphere by solid angle (cap area 2π(1−cos θc)) —
    and cells 1..b-1 split the remaining ring into b−1 equal azimuth cells (each
    also 2π/b). Elevation is honoured (grazing and head-on views cannot share a
    cell) and near-normal views land deterministically in the cap instead of an
    arbitrary azimuth sector."""
    ring = 1 + np.clip(
        ((az + np.pi) / (2 * np.pi) * (b - 1)).astype(np.int64), 0, b - 2
    )
    return np.where(z >= 1.0 - 1.0 / b, np.int64(0), ring)


def _grid_diag(fs: FreeSpace) -> float:
    """World-space diagonal of the free-space grid — the longest physical segment
    a visibility ray can span, used to bound per-octave ray-march step counts."""
    return float(np.linalg.norm(np.asarray(fs.shape, dtype=np.float64) * fs.pitch))


def _band_of(d_eff: np.ndarray, d_min: np.ndarray, n_oct: int) -> np.ndarray:
    """Scale-ladder octave of each effective distance: floor(log2(d_eff/d_min)),
    clamped into [0, n_oct-1]. Views finer than d_min (closer than the finest
    demanded scale) supply band 0 — they oversample it; the sub-pixel far cutoff
    (d_eff > d_max) is the caller's separate reject."""
    return np.clip(
        np.floor(np.log2(np.maximum(d_eff, 1e-12) / d_min)), 0, n_oct - 1
    ).astype(np.int64)


def _hemisphere_dirs(n: int) -> np.ndarray:
    """`n` roughly even unit directions on the +z hemisphere (local frame), with
    the FIRST one exactly head-on ((0,0,1)). Rotated into a patch's tangent frame
    these are its supplier AIM POINTS — a handful of directions around the patch
    normal that between them fall into every direction cell, so the snap-to-
    nearest-candidate search finds one supplier per suppliable cell instead of
    testing every candidate in range. Fibonacci spiral in the polar angle (z from
    1 down toward the equator) for near-uniform spacing."""
    i = np.arange(int(max(n, 1)), dtype=np.float64)
    z = np.clip(1.0 - i / max(n, 1), 1e-3, 1.0)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = i * (np.pi * (3.0 - np.sqrt(5.0)))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def _rescue_pairs(
    candidates: np.ndarray,
    patch_pos: np.ndarray,
    patch_nrm: np.ndarray,
    d_min: np.ndarray,
    d_max: np.ndarray,
    idx_r: np.ndarray,
    bins_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    visible_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    fs: FreeSpace,
    p: PlanParams,
    diag: float,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Best-effort finest-band pairs for the patches no octave pass could see.

    Patch-major and BUDGETED: each unseen patch tests only its `_RESCUE_K`
    nearest candidates within the ladder's reach. Most unseen patches are
    GENUINELY buried (seams, under-furniture voids — the occlusion-culled class)
    and no candidate anywhere can see them; an unbounded any-candidate search
    pays its worst case on exactly those patches. If none of a patch's nearest
    candidates has a clear sightline, farther ones essentially never do — the
    patch stays unseen and is reported occlusion-culled, as before.

    `visible_fn(cams, patches, n_steps) -> bool` supplies the pairwise ray-march
    (numpy or CUDA — the caller picks); `bins_fn(vd, patch_ids)` the direction
    cell. Pairs are marched in distance-sorted slices so short rays take few
    steps and only the far tail pays long ones. Each rescued patch keeps every
    pair of its FINEST visible band (the greedy wants position options +
    direction cells). Returns (cand, patch, bin, band) COO arrays, or None if
    nothing was rescued."""
    if idx_r.size == 0 or len(candidates) == 0:
        return None
    reach = min(max(float(d_max[idx_r].max()), 1e-6), diag)
    kq = int(min(_RESCUE_K, len(candidates)))
    dist_k, ci_k = cKDTree(candidates).query(
        patch_pos[idx_r], k=kq, distance_upper_bound=reach, workers=-1
    )
    dist_k = np.asarray(dist_k, dtype=np.float64).reshape(len(idx_r), kq)
    ci_k = np.asarray(ci_k, dtype=np.int64).reshape(len(idx_r), kq)
    ok = np.isfinite(dist_k)  # scipy pads missing neighbours with inf
    gi = np.repeat(idx_r, ok.sum(axis=1))
    ci = ci_k[ok]
    dist = np.maximum(dist_k[ok], 1e-9)
    d = candidates[ci] - patch_pos[gi]
    cos = np.abs(np.einsum("mc,mc->m", d, patch_nrm[gi])) / dist
    d_eff = dist / np.maximum(cos, 1e-12)
    band = _band_of(d_eff, d_min[gi], p.n_octaves)
    keep = d_eff <= d_max[gi]
    gi, ci, dist, band = gi[keep], ci[keep], dist[keep], band[keep]
    if not gi.size:
        return None

    order = np.argsort(dist, kind="stable")
    gi, ci, dist, band = gi[order], ci[order], dist[order], band[order]
    vis = np.zeros(gi.size, dtype=bool)
    for s0 in range(0, gi.size, _RESCUE_SLICE):
        s1 = min(s0 + _RESCUE_SLICE, gi.size)
        n_steps = int(np.ceil(min(float(dist[s1 - 1]), diag) / fs.pitch)) + 2
        vis[s0:s1] = visible_fn(candidates[ci[s0:s1]], patch_pos[gi[s0:s1]], n_steps)
        if progress is not None:
            progress(s1, gi.size)
    gi, ci, band = gi[vis], ci[vis], band[vis]
    if not gi.size:
        return None
    best = np.full(len(patch_pos), np.int64(1 << 30))
    np.minimum.at(best, gi, band)
    fin = band == best[gi]
    gi, ci, band = gi[fin], ci[fin], band[fin]
    vd = candidates[ci] - patch_pos[gi]
    vd /= np.linalg.norm(vd, axis=1, keepdims=True) + 1e-12
    return ci, gi, bins_fn(vd, gi), band


def _try_cuda():  # noqa: ANN202 - returns the torch module or None
    """Return the `torch` module iff it imports AND a CUDA device is present, else
    None. Imported lazily so Stage 4 stays importable without torch (Stage 5
    re-uses `CUBE_FACES` from here and is torch-free); `plan_cameras` REQUIRES a
    CUDA device — the coverage ray-march runs only on the GPU."""
    try:
        import torch
    except Exception:
        return None
    return torch if torch.cuda.is_available() else None


def _build_coverage(
    candidates: np.ndarray,
    patch_pos: np.ndarray,
    patch_nrm: np.ndarray,
    d_min: np.ndarray,
    d_max: np.ndarray,
    oct_top: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    fs: FreeSpace,
    p: PlanParams,
    torch: Any,
    progress: ProgressCb | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """DEMAND-MAJOR covering (candidate, patch) pairs, each tagged with the
    DIRECTION CELL the camera views the patch from (`_bin_of`: cap + ring cells),
    the scale-ladder OCTAVE the view supplies (`_band_of`), and whether that
    octave is OWED (a real scale demand) or exists for angular supply only.
    Returns COO arrays (cand_idx, patch_idx, bin, octave, owed).

    Rather than enumerate every (candidate, patch) pair in range and ray-march
    all of them — the supplier relation is ~1000× larger than the handful of
    views the greedy keeps, which is what made this stage's memory blow up — this
    asks each patch directly for its few suppliers. Per patch we generate a
    BOUNDED set of aim points and snap each to the nearest real candidate cell:

      * APERTURE aim points — a shell at ~1.5·d_min in `_hemisphere_dirs`
        directions around the patch normal — supply its DIRECTION CELLS and
        finest octave. Real standoff cameras spread over the free hemisphere, so
        ring cells are supplied from an aperture wider than one band (the flaw a
        single fine shell has: its grazing cameras collapse onto the surface).
      * PYRAMID aim points — head-on (+ two tilts) at ~1.5·2^o·d_min for each
        OWED octave o ≥ 1 — supply the coarse scale demands. Owed octaves are
        rare (P(oct_top ≥ o) = 4^-o), so this is a few aim points per patch.

    Each snapped (candidate, patch) pair is verified EXACTLY with the same
    formulas the demand model uses: two-sided incidence `d_eff = dist/|cos θ|`
    (winding-agnostic; incidence folds into scale, no grazing cutoff), octave
    (`_band_of`) and direction cell (`_bin_of`) — so a pair lands in exactly the
    cell/octave a full enumeration would give it; only the redundant extra
    suppliers per cell are never generated. Patches left with no visible pair are
    handed to `_rescue_pairs` (their nearest candidates, best effort) as before.
    Work and memory scale with DEMANDS (patches × cells), not free volume ×
    surface.

    Runs on CUDA: only the occlusion RAY-MARCH touches the GPU (occupancy is the
    single grid's sparse `occ_lin` searched with `torch.searchsorted`; see
    `_occluded`); aim-point construction, snapping (a CPU KD-tree) and the exact
    per-pair binning are numpy. Pairs march in distance-sorted slices so short
    rays take few steps; `progress(done, total, "coverage")` streams the march."""
    dev = torch.device("cuda")
    n_cand = len(candidates)
    n_oct = p.n_octaves
    n_patch = len(patch_pos)
    b = p.bins
    pf = float(fs.pitch)
    d1, d2 = int(fs.dims[1]), int(fs.dims[2])
    diag = _grid_diag(fs)

    occ_lin = torch.as_tensor(fs.occ_lin, dtype=torch.int64, device=dev)
    origin = torch.as_tensor(fs.origin, dtype=torch.float32, device=dev)
    dims = torch.as_tensor(fs.dims, dtype=torch.int64, device=dev)

    def _occluded(cam, pp_, dist_, t_lin):  # noqa: ANN001 - (P,3),(P,3),(P,),(K,) → (P,) bool
        """True where a solid voxel lies strictly between camera and patch (the
        negation of a clear sightline; endpoints skipped via `tvalid`) — or where
        the segment is too short to verify at all (FAIL CLOSED, below).

        ONE grid: every in-bounds, in-range sample is looked up in the sparse
        cover set `occ_lin` with `torch.searchsorted` (mirrors `FreeSpace._member`
        on the single `pitch` lattice); a hit means an occupied voxel sits on the
        open segment. `occ_lin` is ALL cover (glass included), matching the
        physical-presence `occupied` query."""
        pts = cam[:, None, :] + t_lin[None, :, None] * (pp_ - cam)[:, None, :]   # (P,K,3)
        fidx = torch.floor((pts - origin) / pf).to(torch.int64)                  # (P,K,3)
        inb = ((fidx >= 0) & (fidx < dims)).all(dim=2)                           # (P,K)
        tvalid = (t_lin[None, :] > (pf / dist_)[:, None]) & (
            t_lin[None, :] < 1.0 - (1.5 * pf / dist_)[:, None]
        )                                                                        # (P,K)
        need = inb & tvalid                                                      # (P,K)
        # FAIL CLOSED: a segment so short that no sample survives the endpoint
        # skip is unverifiable — count it occluded rather than silently "clear"
        # (wall-adjacent candidate cells make such pairs real).
        res = ~tvalid.any(dim=1)
        nz = need.nonzero(as_tuple=True)
        if nz[0].numel():
            fx = fidx[nz[0], nz[1]]                                             # (M,3)
            lin_m = (fx[:, 0] * d1 + fx[:, 1]) * d2 + fx[:, 2]                  # (M,)
            posi = torch.searchsorted(occ_lin, lin_m.clamp(min=0)).clamp(max=occ_lin.numel() - 1)
            res[nz[0][occ_lin[posi] == lin_m]] = True
        return res

    def _visible(ci_arr: np.ndarray, gi_arr: np.ndarray) -> np.ndarray:
        """Occlusion ray-march for (candidate ci_arr[i] → patch gi_arr[i]); True
        where the sightline is clear. Distance-sorted slices keep the (P,K,3)
        buffer bounded — short rays march few steps, only the far tail pays long
        ones (n_steps sized to each slice's longest segment)."""
        out = np.zeros(len(ci_arr), dtype=bool)
        if not len(ci_arr):
            return out
        dist = np.linalg.norm(candidates[ci_arr] - patch_pos[gi_arr], axis=1)
        order = np.argsort(dist, kind="stable")
        ci_s, gi_s, dist_s = ci_arr[order], gi_arr[order], dist[order]
        vis_s = np.zeros(len(ci_s), dtype=bool)
        for s0 in range(0, len(ci_s), _RESCUE_SLICE):
            s1 = min(s0 + _RESCUE_SLICE, len(ci_s))
            n_steps = int(np.ceil(min(float(dist_s[s1 - 1]), diag) / pf)) + 2
            cam = torch.as_tensor(
                np.ascontiguousarray(candidates[ci_s[s0:s1]], dtype=np.float32), device=dev
            )
            pat = torch.as_tensor(
                np.ascontiguousarray(patch_pos[gi_s[s0:s1]], dtype=np.float32), device=dev
            )
            dst = torch.linalg.norm(pat - cam, dim=1).clamp_min(1e-6)
            t_lin = torch.linspace(0.0, 1.0, n_steps, dtype=torch.float32, device=dev)
            vis_s[s0:s1] = (~_occluded(cam, pat, dst, t_lin)).cpu().numpy()
            if progress is not None:
                progress(s1, len(ci_s), "coverage")
        out[order] = vis_s
        return out

    # --- BOUNDED aim points per patch, snapped to the nearest candidate -------
    # Row 0 of the dictionary is head-on; rotated into each patch's tangent frame
    # (local +z → normal, +x → t1, +y → t2) the set spreads over the free
    # hemisphere so it lands in every direction cell.
    n_dirs = int(max(8, 3 * b))
    dl = _hemisphere_dirs(n_dirs)

    def _aim(dirs_local: np.ndarray, idx: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World aim points (`dirs_local` rotated into each patch's tangent frame,
        scaled by per-patch `radius`) + their parent patch index, patch-major."""
        uw = (
            dirs_local[:, 2][None, :, None] * patch_nrm[idx][:, None, :]
            + dirs_local[:, 0][None, :, None] * t1[idx][:, None, :]
            + dirs_local[:, 1][None, :, None] * t2[idx][:, None, :]
        )  # (S, D, 3)
        tgt = patch_pos[idx][:, None, :] + radius[:, None, None] * uw
        return tgt.reshape(-1, 3), np.repeat(idx, dirs_local.shape[0])

    all_idx = np.arange(n_patch, dtype=np.int64)
    tgt_parts, par_parts = [], []
    if n_patch:
        ap_tgt, ap_par = _aim(dl, all_idx, 1.5 * d_min)  # aperture: bins + octave 0
        tgt_parts.append(ap_tgt)
        par_parts.append(ap_par)
        coarse = dl[: min(3, n_dirs)]  # head-on + up to two tilts
        for o in range(1, n_oct):      # pyramid: one shell per owed octave ≥ 1
            sel = np.nonzero(oct_top >= o)[0]
            if sel.size:
                t_o, p_o = _aim(coarse, sel, (1.5 * (2.0 ** o)) * d_min[sel])
                tgt_parts.append(t_o)
                par_parts.append(p_o)

    if tgt_parts:
        tgt = np.concatenate(tgt_parts, axis=0).astype(np.float64)
        par = np.concatenate(par_parts, axis=0)
        _, ci = cKDTree(candidates).query(tgt, k=1, workers=-1)
        ci = np.asarray(ci, dtype=np.int64)
        keep = ci < n_cand  # scipy pads a miss (empty tree) with n_cand
        ci, par = ci[keep], par[keep]
    else:
        ci = np.zeros(0, dtype=np.int64)
        par = np.zeros(0, dtype=np.int64)

    # Exact per-pair geometry; drop unresolvable (d_eff > d_max) pairs.
    if len(ci):
        vd = candidates[ci] - patch_pos[par]
        dist = np.linalg.norm(vd, axis=1)
        ok = dist > 1e-6
        ci, par, vd, dist = ci[ok], par[ok], vd[ok], dist[ok]
        cos = np.abs(np.einsum("mc,mc->m", vd, patch_nrm[par])) / dist
        d_eff = dist / np.maximum(cos, 1e-12)
        ok = d_eff <= d_max[par]
        ci, par, dist, d_eff = ci[ok], par[ok], dist[ok], d_eff[ok]
        # One pair per (patch, candidate): sort by distance so `unique` keeps the
        # nearest instance of each (multiple aim points can snap to one cell).
        order = np.argsort(dist, kind="stable")
        ci, par, d_eff = ci[order], par[order], d_eff[order]
        _, uidx = np.unique(par * np.int64(n_cand) + ci, return_index=True)
        ci, par, d_eff = ci[uidx], par[uidx], d_eff[uidx]

    cc_o: list[np.ndarray] = []
    pp_o: list[np.ndarray] = []
    bin_o: list[np.ndarray] = []
    octv_o: list[np.ndarray] = []
    owed_o: list[np.ndarray] = []
    seen = np.zeros(n_patch, dtype=bool)

    if len(ci):
        vis = _visible(ci, par)
        ci, par, d_eff = ci[vis], par[vis], d_eff[vis]
    if len(ci):
        vd = candidates[ci] - patch_pos[par]
        vd /= np.linalg.norm(vd, axis=1, keepdims=True) + 1e-12
        z = np.abs(np.einsum("mc,mc->m", vd, patch_nrm[par]))
        az = np.arctan2(
            np.einsum("mc,mc->m", vd, t2[par]), np.einsum("mc,mc->m", vd, t1[par])
        )
        band = _band_of(d_eff, d_min[par], n_oct)
        cc_o.append(ci)
        pp_o.append(par)
        bin_o.append(_bin_of(z, az, b))
        octv_o.append(band)
        owed_o.append(oct_top[par] >= band)
        seen[par] = True

    # RESCUE (`_rescue_pairs`): best effort for patches no aim point could see,
    # against their nearest candidates; the ray-march runs on CUDA.
    def _bins_np(vd: np.ndarray, hit: np.ndarray) -> np.ndarray:
        z = np.abs(np.einsum("mc,mc->m", vd, patch_nrm[hit]))
        az = np.arctan2(
            np.einsum("mc,mc->m", vd, t2[hit]), np.einsum("mc,mc->m", vd, t1[hit])
        )
        return _bin_of(z, az, b)

    def _visible_gpu(cams: np.ndarray, pts: np.ndarray, n_steps: int) -> np.ndarray:
        cams_t = torch.as_tensor(np.ascontiguousarray(cams, dtype=np.float32), device=dev)
        pts_t = torch.as_tensor(np.ascontiguousarray(pts, dtype=np.float32), device=dev)
        dist_t = torch.linalg.norm(pts_t - cams_t, dim=1).clamp_min(1e-6)
        t_lin = torch.linspace(0.0, 1.0, n_steps, dtype=torch.float32, device=dev)
        return (~_occluded(cams_t, pts_t, dist_t, t_lin)).cpu().numpy()

    rescue = _rescue_pairs(
        candidates, patch_pos, patch_nrm, d_min, d_max, np.nonzero(~seen)[0],
        _bins_np, _visible_gpu, fs, p, diag,
        None if progress is None else (lambda done, tot: progress(done, max(tot, 1), "coverage")),
    )
    if rescue is not None:
        cc_o.append(rescue[0])
        pp_o.append(rescue[1])
        bin_o.append(rescue[2])
        octv_o.append(rescue[3])
        owed_o.append(np.ones(len(rescue[0]), dtype=bool))

    if not cc_o:
        e = np.zeros(0, dtype=np.int64)
        return e, e, e, e.copy(), np.zeros(0, dtype=bool)
    return (
        np.concatenate(cc_o),
        np.concatenate(pp_o),
        np.concatenate(bin_o),
        np.concatenate(octv_o),
        np.concatenate(owed_o),
    )


def _greedy_cover(
    unit: np.ndarray,
    pp: np.ndarray,
    binv: np.ndarray,
    octave: np.ndarray,
    owed: np.ndarray,
    n_units: int,
    n_patch: int,
    min_gain: int,
    max_views: int | None,
    tie_noise: np.ndarray,
    progress: ProgressCb | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray, list[int]]:
    """LAZY (CELF) greedy multicover over IMAGES — the selectable unit is one
    (candidate, cube-face) pair, `unit = candidate·6 + face`, the true cost unit
    of Stages 5/6, so a face that contributes no new demand is never picked
    (emergent face culling). A patch wants every SUPPLIABLE direction cell (bin)
    AND every OWED scale octave; each pick takes the image covering the most
    still-missing demands (a pair contributes a bin AND, if owed, an octave —
    up to 2); it stops below `min_gain` or at the `max_views` image budget.

    BIT-IDENTICAL to the naive "recompute every unit's gain each round, take the
    argmax" greedy — it just skips recomputing units that provably can't be the
    argmax. Coverage is monotone submodular, so a unit's marginal gain only ever
    DECREASES; a stored gain is therefore an upper bound on the current gain.
    Each round pops the unit with the highest stored bound, recomputes its true
    current gain, and commits it iff that still beats the next-highest stored
    bound (then no other unit can exceed it); otherwise it reinserts the fresh
    (lower) bound and pops again. `tie_noise` (per-unit, [0,0.5), seeded) makes
    every gain a distinct real `int_gain + noise`, so the argmax is unique;
    heap entries `(-gain, unit)` break the impossible exact-tie toward the lower
    unit id, matching `np.argmax`'s lowest-index rule. The result — pick
    sequence, per-pick integer gains, and both bitmasks — is identical to the
    naive greedy round for round (verified against it in the equivalence
    harness), at a fraction of the work.

    Returns (chosen unit ids, per-patch bin bitmask, per-patch covered-octave
    bitmask, per-pick integer gains). Streams `progress("select", picks, 0)`."""
    import heapq

    binmask = np.zeros(n_patch, dtype=np.int64)
    octmask = np.zeros(n_patch, dtype=np.int64)
    chosen: list[int] = []
    gains: list[int] = []
    if len(unit) == 0:
        return chosen, binmask, octmask, gains

    # CSR: group pairs by unit so one unit's current gain is a contiguous slice.
    # (argsort-stable by unit is ascending, matching the cumsum(bincount)
    # boundaries below.)
    order = np.argsort(unit, kind="stable")
    p_pat = pp[order]
    p_binbit = (np.int64(1) << binv[order].astype(np.int64))
    p_octbit = (np.int64(1) << octave[order].astype(np.int64))
    p_owed = owed[order].astype(bool)
    counts = np.bincount(unit, minlength=n_units).astype(np.int64)
    seg = np.zeros(n_units + 1, dtype=np.int64)
    np.cumsum(counts, out=seg[1:])

    # Initial gain (all demands uncovered): every pair's bin demand counts, plus
    # its octave iff owed — the identical value the naive path's first bincount
    # produces. Real gain = int gain + tie noise (same array, same semantics).
    owed_counts = np.bincount(
        unit, weights=owed.astype(np.float64), minlength=n_units
    ).astype(np.int64)
    int_gain0 = counts + owed_counts

    min_g = max(int(min_gain), 1)

    def _cur_gain(u: int) -> int:
        """u's TRUE current integer gain against the live masks — the same count
        the naive per-round bincount would give for u (bin demand per pair, plus
        an owed uncovered octave)."""
        s, e = int(seg[u]), int(seg[u + 1])
        if e == s:
            return 0
        pat = p_pat[s:e]
        bin_new = (binmask[pat] & p_binbit[s:e]) == 0
        oct_new = p_owed[s:e] & ((octmask[pat] & p_octbit[s:e]) == 0)
        return int(bin_new.sum()) + int(oct_new.sum())

    def _commit(u: int) -> None:
        """Mark u's demands covered (patches within a unit are unique, but `.at`
        is order/collision-independent — exactly the naive update)."""
        s, e = int(seg[u]), int(seg[u + 1])
        pat = p_pat[s:e]
        np.bitwise_or.at(binmask, pat, p_binbit[s:e])
        ow = p_owed[s:e]
        if ow.any():
            np.bitwise_or.at(octmask, pat[ow], p_octbit[s:e][ow])

    # Heap of (-real_gain, unit) over units that have any pair; -real so heapq's
    # min-heap yields the max gain, and the unit id tiebreaks toward the lowest
    # (== np.argmax) on the (noise-precluded) exact tie.
    present = np.nonzero(counts > 0)[0]
    real0 = int_gain0[present].astype(np.float64) + tie_noise[present]
    heap = [(-float(r), int(u)) for r, u in zip(real0, present)]
    heapq.heapify(heap)

    while heap and (max_views is None or len(chosen) < max_views):
        neg, u = heapq.heappop(heap)
        cur_int = _cur_gain(u)
        cur_real = cur_int + float(tie_noise[u])
        nxt = -heap[0][0] if heap else float("-inf")
        if cur_real >= nxt:
            # No other unit's current gain can exceed cur_real (all ≤ their
            # stored bound ≤ nxt), so u is the exact argmax this round.
            if cur_int < min_g:
                break
            _commit(u)
            chosen.append(u)
            gains.append(cur_int)
            if progress is not None and len(chosen) % 200 == 0:
                progress(len(chosen), 0, "select")
        else:
            heapq.heappush(heap, (-cur_real, u))
    return chosen, binmask, octmask, gains


def _face_of(dirs: np.ndarray) -> np.ndarray:
    """Bin each (N,3) camera→patch direction into a cube-face index (0..5, matching
    CUBE_FACE_NAMES) by its dominant axis and sign — the outward face that view
    lands on."""
    axis = np.argmax(np.abs(dirs), axis=1)  # 0=x, 1=y, 2=z
    negative = dirs[np.arange(len(dirs)), axis] < 0
    return axis * 2 + negative.astype(np.int64)


def plan_cameras(
    *,
    run: str,
    slot: str,
    model: str,
    freespace_path: Path,
    surfels_path: Path,
    out_path: Path,
    params: PlanParams = PlanParams(),
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Plan coverage cameras for one cell from the Stage-2 free-space grid
    (`freespace_path`) + Stage-3 surfel cloud (`surfels_path`); write `cameras.json`
    (to `out_path`) + `patches.bin` (beside it) and return a summary."""
    if not Path(freespace_path).is_file():
        raise FileNotFoundError(f"free-space grid not found: {freespace_path} (run Stage 2)")
    if not Path(surfels_path).is_file():
        raise FileNotFoundError(f"surfel cloud not found: {surfels_path} (run Stage 3)")
    rng = np.random.default_rng(params.seed)

    if progress is not None:
        progress(0, 0, "load")
    fs = load_free_space(Path(freespace_path))
    points, normals, _ = _read_cloud(Path(surfels_path))
    lo = points.min(axis=0)
    hi = points.max(axis=0)

    # Patches = the Stage-3 surfels UNIFORMLY thinned, so patch density inherits
    # the cloud's own feature-adaptive density (geometry AND texture) — detail is
    # ridden, not re-detected. patch_feat (each patch's ladder feature scale) is
    # the local surfel spacing mapped into [s_min, s_max]: the coarse (flat) end
    # anchors at s_max, finer surfels ride proportionally below it. The uniform
    # keep rate (s_min/s_max)² matches the old flat-region thinning exactly, so
    # flats are unchanged and detail simply rides on top.
    if progress is not None:
        progress(0, 0, "patches")
    surf_spacing = _local_spacing(points)
    scale = params.patch_max_spacing / max(float(np.percentile(surf_spacing, 90)), 1e-6)
    keep = np.nonzero(
        rng.random(len(points))
        < (params.patch_min_spacing / params.patch_max_spacing) ** 2
    )[0]
    patch_pos = points[keep].astype(np.float32)
    patch_nrm = normals[keep].astype(np.float32)
    patch_feat = np.clip(
        surf_spacing[keep] * scale, params.patch_min_spacing, params.patch_max_spacing
    ).astype(np.float32)
    t1, t2 = _tangent_frames(patch_nrm)
    n_patch = len(patch_pos)

    # THE SCALE LADDER (module docstring): per-patch effective-distance bands from
    # `finest_px` pixels-per-feature down to the 1-px observability limit. Both
    # anchors derive from the patch's own feature scale + the shared focal length
    # — no minimum/maximum view distance exists anywhere in the plan.
    n_oct = params.n_octaves
    d_min = (patch_feat * (params.focal_px / params.finest_px)).astype(np.float32)
    d_max = (patch_feat * params.focal_px).astype(np.float32)
    # THE PYRAMID: which octaves each patch DEMANDS. At octave o the image
    # resolves 2^o × coarser detail, so demands only need 2^o × coarser spacing:
    # patch p is demanded at octaves 0..oct_top[p] with P(oct_top ≥ o) = 4^-o — a
    # NESTED, self-similar thinning (each coarser octave over 4× fewer patches).
    # Total demands ≈ 4/3 × patches.
    u = rng.random(n_patch)
    oct_top = np.minimum(
        np.floor(-np.log(np.maximum(u, 1e-12)) / np.log(4.0)).astype(np.int64),
        n_oct - 1,
    )

    # Candidate camera positions from the reachable free cells (no re-voxelization).
    candidates = _candidates(fs, params)
    if len(candidates) == 0:
        raise RuntimeError("no candidate camera positions (reachable free space empty)")

    # Coverage (per-pair direction cell + octave + owed) + greedy multicover over
    # IMAGES. Coverage is the long phase (a demand-major occlusion ray-march on
    # CUDA — each patch's own bounded aim points), so it streams progress as the
    # snapped pairs are marched.
    b = params.bins
    torch = _try_cuda()
    if torch is None:
        raise RuntimeError(
            "stage4: the coverage ray-march runs on CUDA, but no CUDA device / "
            "torch is available. Run Stage 4 in the GPU env (the same one Stage 6 "
            "uses)."
        )
    cc, pp, binv, octave, owed = _build_coverage(
        candidates, patch_pos, patch_nrm, d_min, d_max, oct_top, t1, t2, fs,
        params, torch, progress,
    )

    # IMAGE units (the greedy's selectable): unit = candidate·6 + cube face of
    # the pair's direction. The image is the true cost unit — Stage 5 renders,
    # stores and trains per (position, face) — so faces compete individually and
    # a face adding nothing new is never selected at all.
    n_faces = len(CUBE_FACE_NAMES)
    face = (
        _face_of(patch_pos[pp] - candidates[cc]) if len(cc) else np.zeros(0, dtype=np.int64)
    )
    unit = cc * n_faces + face
    n_units = len(candidates) * n_faces
    # Seeded per-unit tie-break noise (see `_greedy_cover`): integer gains tie in
    # huge plateaus over uniform geometry, and argmax would take them in
    # candidate scan order — the "left-to-right density sweeps"; under a budget
    # cut that leaves one side of the scene denser. Noise < 0.5 never outvotes a
    # real demand.
    tie_noise = (rng.random(n_units) * 0.5).astype(np.float64)
    # Single greedy path: lazy (CELF) multicover on the CPU — exact-equal to the
    # naive per-round argmax greedy (see `_greedy_cover`), but skipping the units
    # that can't be the argmax, so it doesn't rescan all pairs every pick. Runs
    # off the numpy COO the coverage builder returns (GPU or CPU), and frees the
    # GPU during selection.
    chosen_units, binmask, octmask, gains = _greedy_cover(
        unit, pp, binv, octave, owed, n_units, n_patch,
        params.min_gain, params.max_views, tie_noise, progress,
    )
    if progress is not None:
        progress(0, 0, "write")

    # Per-patch stats: direction cells seen, octave coverage, and satisfaction.
    # `oct_supply` = the OWED bands any candidate could actually see per patch
    # (the pyramid's a-priori bands where they had supply, plus each rescued
    # patch's finest visible band); `bin_supply` = the direction cells any
    # candidate could see (any pair — direction supply is scale-free). A demand
    # with no supply at all (the far bands of an indoor wall; the ring cells of
    # a patch at the bottom of a slot) can't be held against the plan; it shows
    # up in the demanded-vs-suppliable gaps below.
    lut_b = np.array([bin(i).count("1") for i in range(1 << b)], dtype=np.int16)
    bins_seen = lut_b[binmask].astype(np.int32)
    oct_need = (np.int64(1) << (oct_top + 1)) - 1
    oct_supply = np.zeros(n_patch, dtype=np.int64)
    bin_supply = np.zeros(n_patch, dtype=np.int64)
    if len(pp):
        np.bitwise_or.at(oct_supply, pp[owed], np.int64(1) << octave[owed])
        np.bitwise_or.at(bin_supply, pp, np.int64(1) << binv)
    seen_any = np.zeros(n_patch, dtype=bool)
    if len(pp):
        seen_any[np.unique(pp)] = True
    # Masks only accumulate from chosen pairs, so mask == supply means every
    # suppliable demand (scale AND direction) was met.
    satisfied_mask = (
        seen_any
        & ((binmask & bin_supply) == bin_supply)
        & ((octmask & oct_supply) == oct_supply)
    )
    satisfied = int(satisfied_mask.sum())
    seen_once = int(seen_any.sum())
    occluded = int((~seen_any).sum())

    # Ladder + direction totals and per-level fill, for the summary (and the
    # debug viewer).
    lut_o = np.array([bin(i).count("1") for i in range(1 << n_oct)], dtype=np.int16)
    octave_levels = [
        {
            "demanded": int((oct_top >= o).sum()),
            "suppliable": int(((oct_supply >> o) & 1).sum()),
            "covered": int(((octmask >> o) & 1).sum()),
        }
        for o in range(n_oct)
    ]
    octave_stats = {
        "levels": n_oct,
        "finest_px": params.finest_px,
        "demanded": int(lut_o[oct_need].sum()),
        "suppliable": int(lut_o[oct_supply].sum()),
        "covered": int(lut_o[octmask].sum()),
        "per_level": octave_levels,
    }
    bin_stats = {
        "cells": b,
        "cap_cos": round(1.0 - 1.0 / b, 4),
        "suppliable": int(lut_b[bin_supply].sum()),
        "covered": int(lut_b[binmask].sum()),
        "mean_seen": round(float(bins_seen[seen_any].mean()), 2) if seen_once else 0.0,
    }

    # The COVERAGE CURVE: cumulative demands covered per image picked, reported
    # as the image count reaching each fraction of the suppliable total. This is
    # the diminishing-returns ledger a `max_views` budget is chosen against —
    # set-cover curves are strongly concave, so most demands are covered by a
    # small prefix and the completion tail (gain-1 picks) dominates raw counts.
    suppl_total = octave_stats["suppliable"] + bin_stats["suppliable"]
    cum = np.cumsum(np.asarray(gains, dtype=np.int64))

    def _images_for(frac: float) -> int | None:
        if not len(cum):
            return None
        i = int(np.searchsorted(cum, frac * suppl_total))
        return i + 1 if i < len(cum) else None

    curve = {
        "demands_suppliable": suppl_total,
        "demands_covered": int(cum[-1]) if len(cum) else 0,
        "images": len(gains),
        "images_p50": _images_for(0.50),
        "images_p90": _images_for(0.90),
        "images_p95": _images_for(0.95),
        "images_p99": _images_for(0.99),
    }

    # Emit ONLY the chosen images: chosen units grouped by candidate → one
    # camera entry per position carrying exactly its selected faces (an image
    # the greedy never picked is never rendered — emergent face culling). Camera
    # order = greedy pick order of each position's first face; face order within
    # a camera follows CUBE_FACE_NAMES for stable ids. Also build the per-patch
    # VIEW INDEX (patch → [(camera_index, face_index), …]) from the chosen
    # images' pairs so the debug viewer can map a selected surface patch to its
    # Stage-5 reference images.
    cameras: list[dict[str, Any]] = []
    patch_views: list[list[list[int]]] = [[] for _ in range(n_patch)]
    if chosen_units:
        cam_order: list[int] = []
        cam_faces: dict[int, list[int]] = {}
        for u in chosen_units:
            ci, fi = divmod(int(u), n_faces)
            if ci not in cam_faces:
                cam_faces[ci] = []
                cam_order.append(ci)
            cam_faces[ci].append(fi)
        order = np.argsort(unit, kind="stable")
        unit_s, pp_s = unit[order], pp[order]
        for out_idx, ci in enumerate(cam_order):
            cam = candidates[ci]
            faces: list[dict[str, Any]] = []
            covers = 0
            for face_pos, fi in enumerate(sorted(cam_faces[ci])):
                u = ci * n_faces + fi
                a0 = int(np.searchsorted(unit_s, u, "left"))
                a1 = int(np.searchsorted(unit_s, u, "right"))
                seen = pp_s[a0:a1]
                faces.append({"dir": CUBE_FACE_NAMES[fi], "covers": int(len(seen))})
                covers += int(len(seen))
                for pt in seen.tolist():
                    patch_views[pt].append([out_idx, face_pos])
            cameras.append(
                {
                    "pos": [round(float(v), 4) for v in cam],
                    "faces": faces,
                    "covers": covers,
                }
            )

    # patch_views.json: for each patch (index matches patches.bin row order), the
    # list of [camera_index, face_index] that cover it — face_index into `faces`.
    # camera_index matches Stage 5's `cam{index:05d}_{face}` render ids.
    patch_views_path = out_path.with_name(PATCH_VIEWS_NAME)
    tmp_pv = patch_views_path.with_suffix(patch_views_path.suffix + ".tmp")
    tmp_pv.write_text(
        json.dumps({"faces": list(CUBE_FACE_NAMES), "views": patch_views}),
        encoding="utf-8",
    )
    tmp_pv.replace(patch_views_path)

    # patches.bin: [x,y,z, nx,ny,nz, feature_scale, bins_seen] × N (column 8 was
    # `sectors_seen`; same layout, now the count of distinct direction CELLS).
    pdata = np.concatenate(
        [
            patch_pos.astype("<f4"),
            patch_nrm.astype("<f4"),
            patch_feat.reshape(-1, 1).astype("<f4"),
            bins_seen.reshape(-1, 1).astype("<f4"),
        ],
        axis=1,
    )
    patches_path = out_path.with_name(PATCHES_NAME)
    patches_path.parent.mkdir(parents=True, exist_ok=True)
    patches_path.write_bytes(pdata.tobytes())

    summary = {
        "run": run,
        "slot": slot,
        "model": model,
        "patches": n_patch,
        "candidates": int(len(candidates)),
        "cameras": len(cameras),
        "views": len(chosen_units),
        "angular_bins": b,
        "coverage": {
            "satisfied": satisfied,
            "satisfied_pct": round(100.0 * satisfied / max(n_patch, 1), 1),
            "seen_at_least_once": seen_once,
            "occlusion_culled": occluded,
            "mean_angles_seen": bin_stats["mean_seen"],
            "bins": bin_stats,
            "octaves": octave_stats,
            "curve": curve,
        },
        "scene_aabb": {"min": lo.tolist(), "max": hi.tolist()},
        "params": params.as_summary(),
    }

    # Shared cube-face intrinsics Stage 5 renders with (and Stage 6 trains against),
    # plus the scale-ladder anchors the plan was built with (informational).
    # near sits at or below the closest possible camera–surface distance
    # (candidates are free coarse-cell centres, ≥ pitch/2 from any surface
    # voxel), so no chosen view clips geometry; far spans the full free-space
    # grid diagonal (the play volume).
    grid_diag = _grid_diag(fs)
    intrinsics = {
        "face_fov_deg": params.face_fov_deg,
        "resolution": params.render_resolution,
        "finest_px_per_patch": params.finest_px,
        "octaves": n_oct,
        "focal_px": round(params.focal_px, 3),
        "near": round(min(0.05, fs.pitch * 0.5), 4),
        "far": round(grid_diag, 3),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **summary,
        "intrinsics": intrinsics,
        "cube_faces": CUBE_FACES,
        "cameras": cameras,
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(out_path)
    summary["bytes"] = out_path.stat().st_size
    summary["patches_bytes"] = patches_path.stat().st_size
    summary["patch_views_bytes"] = patch_views_path.stat().st_size
    return summary
