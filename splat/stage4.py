"""Stage 4 — Coverage camera planner (scale-ladder demands + greedy set-cover).

Picks the fewest reference IMAGES — (position, cube-face) pairs in free space —
so every visible surface patch is seen from every suppliable DIRECTION CELL
(equal-solid-angle bins below) AND once per OCTAVE of viewing scale (the mip
ladder below). The output feeds Stage 5 (reference renders) and yields the
occlusion-cull list as a byproduct.

CONNECTED (Option A): Stage 4 consumes the outputs of the earlier stages and loads
NO meshes and computes NO occupancy of its own —
  * **Stage 2 free-space grid** (`freespace.npz`): candidate camera positions
    (reachable free cells) + the fine occupancy the line-of-sight ray-march
    uses. No re-voxelization.
  * **Stage 3 surfel cloud** (`cloud.ply`): the patch source. Patches are a
    feature-adaptive thinning of the surfels, whose normals were already oriented to
    free space in Stage 3 — so the facing test is reliable regardless of the
    mesh's original winding.

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
P(oct_top ≥ o) = 4^-o, a NESTED thinning (total demands ≈ 4/3 × patches). Pairs
for octave o are generated only against the octave-o demand set, with a
per-octave search radius: coarse octaves search far but over exponentially fewer
patches, so pairs-per-candidate stay ~constant per octave and each (candidate,
patch) pair lands in exactly one octave.

DIRECTION CELLS (equal-solid-angle bins; replaces azimuth-only sectors): view
directions are binned on the folded hemisphere around each patch normal into
`angular_bins` equal-area cells — an explicit polar CAP (the direct-facing
field) + one ring of azimuth cells (`_bin_of`) — so elevation diversity is
enforced and near-normal views land deterministically. The demand is every
SUPPLIABLE cell. Direction supply is scale-free (any resolving view counts), and
comes from a dedicated band-0/1 APERTURE pass over the full patch set: one band
alone cannot reliably supply ring cells (their cameras sit at dist = d_eff·cosθ,
which for the finest band's grazing elevations collapses onto the surface
itself), which is exactly the flaw that used to confine angular supply to the
finest shell and multiply close cameras. Aperture pairs whose band the pyramid
doesn't demand carry `owed = False`: they supply direction cells without
inflating scale demands.

Pipeline:
  1. Read the surfel cloud → points + oriented normals + albedo.
  2. Feature-adaptive PATCHES: spacing shrinks with local detail (curvature via
     normal variance, texture via albedo variance); denser where detail is.
  3. CANDIDATE positions = reachable free cells (Stage 2), subsampled denser
     where clearance is small; the greedy's own scale demands keep chosen
     cameras off surfaces (no demand exists below a patch's d_min).
  4. COVERAGE: aperture pass (bands 0-1, all patches) + pyramid passes (band ≥ 2
     demand sets); a pair exists where d_eff lands in the pass's bands and the
     fine-grid ray-march is clear; each pair is tagged with its cube FACE.
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

    patch_min_spacing: float = 0.06   # s_min (m): finest patch spacing = footprint detail
    patch_max_spacing: float = 0.30   # s_max (m): flat-region patch spacing
    curvature_k: float = 14.0         # curvature → spacing sensitivity
    tex_k: float = 8.0                # texture-gradient → spacing sensitivity
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
    # Candidate positions are drawn from the NEAR-SURFACE band (reachable cells
    # with clearance up to the band-0 reach of the coarsest patch — see
    # `_candidates`) at ~`candidate_spacing` apart, so the candidate COUNT
    # scales with the near-surface area, not a fixed budget.
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


def _feature_spacing(
    points: np.ndarray, normals: np.ndarray, albedo: np.ndarray, p: PlanParams
) -> np.ndarray:
    """Per-point target spacing s(x): small where detail is high. Combines curvature
    (local normal variance) and texture gradient (local albedo variance); the densest
    wins. (Triangle-size detail is folded into the surfel density already.)"""
    n = len(points)
    tree = cKDTree(points)
    k = min(9, n)
    _, idx = tree.query(points, k=k)
    neigh_idx = idx[:, 1:] if k > 1 else idx
    cos = np.clip(np.einsum("nkc,nc->nk", normals[neigh_idx], normals), -1.0, 1.0)
    curv = 1.0 - cos.mean(axis=1)
    s_curv = p.patch_max_spacing / (1.0 + p.curvature_k * curv)
    tex_var = albedo[neigh_idx].std(axis=1).mean(axis=1)
    s_tex = p.patch_max_spacing / (1.0 + p.tex_k * tex_var)
    s = np.minimum(s_curv, s_tex)
    return np.clip(s, p.patch_min_spacing, p.patch_max_spacing).astype(np.float32)


def _adaptive_patches(
    points: np.ndarray,
    normals: np.ndarray,
    spacing: np.ndarray,
    s_min: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Thin the dense surfel set to feature-adaptive density: keep point i with
    probability (s_min / s_i)^2, so flat regions get sparse patches and detailed
    regions keep (near-)all of theirs. Returns the kept indices."""
    keep_p = np.clip((s_min / spacing) ** 2, 0.0, 1.0)
    keep = rng.random(len(points)) < keep_p
    return np.nonzero(keep)[0]


def _candidate_band(p: PlanParams) -> float:
    """Outer clearance edge (m) of the candidate field, DERIVED from the ladder:
    a candidate is only ever needed within band 0 of some patch (every coarser
    band is reachable from farther away, i.e. from other patches' bands), and the
    farthest band-0 view of the coarsest patch is at
    2 · patch_max_spacing · focal / finest_px — beyond that clearance a cell can
    supply no finest-scale demand. Also the rescue pass's search reach: the
    farthest a candidate the field even CONTAINS can plausibly be from a surface
    it must cover."""
    return 2.0 * p.patch_max_spacing * p.focal_px / p.finest_px


def _candidates(fs: FreeSpace, p: PlanParams) -> np.ndarray:
    """Candidate camera positions: reachable NEAR-SURFACE free cells sampled
    ~`candidate_spacing` apart, so the count scales with the near-surface area
    rather than a fixed budget. The band runs from the wall-adjacent cells (no
    clearance floor — standoff is EMERGENT: the ladder demands nothing below a
    patch's d_min, so point-blank candidates only win where occlusion leaves no
    farther supplier, and the ray-march fails closed on segments too short to
    verify) out to `_candidate_band`, the ladder-derived outer edge. (Interim
    until candidates span the whole reachable volume; grazing views already let
    this band supply far octaves.) If a scene still exceeds `max_candidates`,
    evenly downsample (and warn) — never silently fall back to a room-scale cap.
    Returns (M,3) world points."""
    centers, _ = fs.free_candidates(_candidate_band(p), p.candidate_spacing)
    if len(centers) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if p.max_candidates and len(centers) > p.max_candidates:
        step = int(np.ceil(len(centers) / p.max_candidates))
        logging.warning(
            "stage4: %d near-surface candidates at spacing %.2fm exceeds the "
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
    return float(np.linalg.norm(np.asarray(fs.fine_shape, dtype=np.float64) * fs.pitch_fine))


def _band_of(d_eff: np.ndarray, d_min: np.ndarray, n_oct: int) -> np.ndarray:
    """Scale-ladder octave of each effective distance: floor(log2(d_eff/d_min)),
    clamped into [0, n_oct-1]. Views finer than d_min (closer than the finest
    demanded scale) supply band 0 — they oversample it; the sub-pixel far cutoff
    (d_eff > d_max) is the caller's separate reject."""
    return np.clip(
        np.floor(np.log2(np.maximum(d_eff, 1e-12) / d_min)), 0, n_oct - 1
    ).astype(np.int64)


def _scale_buckets(d_min: np.ndarray) -> list[np.ndarray]:
    """Group patch indices by the octave of their `d_min` (their feature scale),
    so a mixed-scale pass can use a per-bucket search radius instead of the
    global max — the aperture pass over the FULL patch set would otherwise query
    the coarsest patch's radius against the finest patches' density."""
    lo = float(d_min.min())
    bucket = np.clip(np.floor(np.log2(np.maximum(d_min, 1e-12) / lo)), 0, 62).astype(np.int64)
    return [np.nonzero(bucket == b)[0] for b in range(int(bucket.max()) + 1) if (bucket == b).any()]


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
        n_steps = int(np.ceil(min(float(dist[s1 - 1]), diag) / fs.pitch_fine)) + 2
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
    """Covering (candidate, patch) pairs, each tagged with the DIRECTION CELL the
    camera views the patch from (`_bin_of`: cap + ring cells), the scale-ladder
    OCTAVE the view supplies, and whether that octave is OWED (a real scale
    demand) or exists for angular supply only. Returns COO arrays (cand_idx,
    patch_idx, bin, octave, owed).

    Pass plan: an APERTURE pass (bands 0-1 over the full patch set, scale-
    bucketed radii) supplies every patch's direction cells — a single band's
    ring-cell views sit at dist = d_eff·cos, which for the finest band collapses
    onto the surface itself (below the grid's own resolution), so direction
    diversity needs an aperture wider than one band; band-0 pairs are owed by
    every patch, band-1 pairs only where
    the pyramid demands band 1 (`oct_top ≥ 1`). Then PYRAMID passes for bands
    ≥ 2, one per octave o against the demand set `oct_top >= o` (nested 4^-o
    thinning), all owed. `d_eff = dist / |cos θ|` is two-sided (winding-agnostic;
    incidence folds into scale, no grazing cutoff) and a pure function of the
    pair, so every pair lands in exactly one band/pass; per-pass radii keep
    pairs-per-candidate bounded. A patch with NO pair is rescued
    (`_rescue_pairs`) at its FINEST visible band rather than dropped.

    Runs on CUDA: the CPU KD-tree finds each candidate's in-range patches per
    pass (cheap, not the bottleneck), but the per-pair band filter, direction-
    cell binning and the occlusion RAY-MARCH (the hot loop: m × n_steps
    sparse-grid membership tests) run on the GPU in candidate batches. The fine
    occupancy is the sparse `occ_lin` searched with `torch.searchsorted`
    (mirrors `FreeSpace.fine_occupied`), accelerated by a coarse empty-space skip
    (see `_occluded`). Coarse octaves march longer rays, so the pair slice size
    scales down with the pass's step count to hold the ray-march buffer
    ~constant. Streams `progress(pass·n_cand + cand, n_passes·n_cand,
    "coverage")` — the stage's long pole."""
    dev = torch.device("cuda")
    n_cand = len(candidates)
    n_oct = p.n_octaves
    n_patch = len(patch_pos)
    b = p.bins
    pf = float(fs.pitch_fine)
    d1, d2 = int(fs.fine_dims[1]), int(fs.fine_dims[2])
    diag = _grid_diag(fs)

    occ_lin = torch.as_tensor(fs.occ_lin, dtype=torch.int64, device=dev)
    origin = torch.as_tensor(fs.origin, dtype=torch.float32, device=dev)
    dims = torch.as_tensor(fs.fine_dims, dtype=torch.int64, device=dev)
    # Coarse occupancy (dense, 1 byte/cell) for the empty-space skip in
    # `_occluded`: a fine voxel can only be occupied if its COARSE block is solid
    # (`clearance == 0`); `clearance > 0` blocks hold no occupied fine voxel.
    # Derived from the SAME Stage-2 build as `occ_lin`, so the skip is exact.
    coarse_solid = torch.as_tensor(
        np.ascontiguousarray(fs.clearance <= 0.0).reshape(-1), device=dev
    )
    cd0, cd1, cd2 = int(fs.clearance.shape[0]), int(fs.clearance.shape[1]), int(fs.clearance.shape[2])
    refine = int(fs.refine)
    ppos = torch.as_tensor(patch_pos, dtype=torch.float32, device=dev)
    pnrm = torch.as_tensor(patch_nrm, dtype=torch.float32, device=dev)
    dmin_t = torch.as_tensor(d_min, dtype=torch.float32, device=dev)
    dmax_t = torch.as_tensor(d_max, dtype=torch.float32, device=dev)
    octtop_t = torch.as_tensor(oct_top, dtype=torch.int64, device=dev)
    t1g = torch.as_tensor(t1, dtype=torch.float32, device=dev)
    t2g = torch.as_tensor(t2, dtype=torch.float32, device=dev)

    cand_batch = 4096
    cc_o: list[np.ndarray] = []
    pp_o: list[np.ndarray] = []
    bin_o: list[np.ndarray] = []
    octv_o: list[np.ndarray] = []
    owed_o: list[np.ndarray] = []
    seen = np.zeros(n_patch, dtype=bool)

    def _occluded(cam, pp_, dist_, t_lin):  # noqa: ANN001 - (P,3),(P,3),(P,),(K,) → (P,) bool
        """True where a solid FINE voxel lies strictly between camera and patch
        (the negation of a clear sightline; endpoints skipped via `tvalid`) — or
        where the segment is too short to verify at all (FAIL CLOSED, below).

        COARSE EMPTY-SPACE SKIP: a sample can only occlude if it is in-bounds,
        inside the valid t-range, AND its coarse block is solid. `clearance > 0`
        blocks contain no occupied fine voxel, so the sparse `searchsorted` (the
        log-N hot op) runs ONLY on the samples in solid coarse cells — a small
        fraction on open scenes. Bit-exact vs the all-samples search: a fine hit
        implies its coarse block is solid, so no skipped sample could ever have
        matched (`occ_lin` and `coarse_solid` come from the one Stage-2 build,
        and the coarse cell is `fine_index // refine` of the SAME index the
        search uses — an integer identity, robust to any binning rounding)."""
        pts = cam[:, None, :] + t_lin[None, :, None] * (pp_ - cam)[:, None, :]   # (P,K,3)
        fidx = torch.floor((pts - origin) / pf).to(torch.int64)                  # (P,K,3)
        inb = ((fidx >= 0) & (fidx < dims)).all(dim=2)                           # (P,K)
        tvalid = (t_lin[None, :] > (pf / dist_)[:, None]) & (
            t_lin[None, :] < 1.0 - (1.5 * pf / dist_)[:, None]
        )                                                                        # (P,K)
        # Coarse block of each sample (clamped only to keep the gather in range;
        # out-of-grid samples are dropped by `inb`).
        cflat = (
            (fidx[..., 0] // refine).clamp(0, cd0 - 1) * cd1
            + (fidx[..., 1] // refine).clamp(0, cd1 - 1)
        ) * cd2 + (fidx[..., 2] // refine).clamp(0, cd2 - 1)                     # (P,K)
        need = inb & tvalid & coarse_solid[cflat]                               # (P,K)
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

    def _pass(idx_sub: np.ndarray, r_pass: float, band_lo: int, band_hi: int, prog_base: int, total: int) -> (
        list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ):
        """One pass of every candidate against `idx_sub` patches within `r_pass`:
        per-pair band filter (keep bands in [band_lo, band_hi]) + occlusion
        ray-march. Returns kept (cand, patch, bin, band, owed) chunks; marks
        `seen`."""
        out: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        tree = cKDTree(patch_pos[idx_sub])
        n_steps = int(np.ceil(min(r_pass, diag) / fs.pitch_fine)) + 2
        t_lin = torch.linspace(0.0, 1.0, n_steps, dtype=torch.float32, device=dev)
        # Slice size × step count ≈ constant so the (P, K, 3) ray-march buffer
        # stays within VRAM whether the pass marches 0.5 m or 50 m rays.
        pair_cap = int(np.clip(15_000_000 // n_steps, 4_096, 150_000))
        # Pre-size candidate chunks by neighbour COUNTS (return_length is a cheap
        # C pass) so the Python list-of-lists a positional query materialises
        # never holds more than ~pair_chunk entries: a room-scale ball can span
        # most of the patch cloud, and thousands of such lists at once is an
        # out-of-memory, not a working set.
        pair_chunk = 4_000_000
        lengths_all = tree.query_ball_point(
            np.ascontiguousarray(candidates, dtype=np.float32), r_pass,
            workers=-1, return_length=True,
        ).astype(np.int64)
        b0 = 0
        while b0 < n_cand:
            b1 = b0 + 1
            tot = int(lengths_all[b0])
            while (
                b1 < n_cand
                and b1 - b0 < cand_batch
                and tot + int(lengths_all[b1]) <= pair_chunk
            ):
                tot += int(lengths_all[b1])
                b1 += 1
            cams_np = np.ascontiguousarray(candidates[b0:b1], dtype=np.float32)
            neigh = tree.query_ball_point(cams_np, r_pass, workers=-1)
            lengths = np.fromiter((len(x) for x in neigh), dtype=np.int64, count=len(neigh))
            n_pairs = int(lengths.sum())
            if n_pairs:
                local = np.repeat(np.arange(len(neigh), dtype=np.int64), lengths)
                pidx = idx_sub[
                    np.concatenate([np.asarray(x, dtype=np.int64) for x in neigh if len(x)])
                ]
                cams_b = torch.as_tensor(cams_np, device=dev)
                # Walk the (candidate, in-range-patch) pairs in ≤pair_cap slices so
                # the per-slice filter AND ray-march stay within VRAM no matter how
                # dense the scene is around a candidate.
                for s0 in range(0, n_pairs, pair_cap):
                    s1 = s0 + pair_cap
                    loc = torch.as_tensor(local[s0:s1], device=dev)
                    pj = torch.as_tensor(pidx[s0:s1], device=dev)
                    cam = cams_b[loc]
                    dvec = ppos[pj] - cam
                    dist = torch.linalg.norm(dvec, dim=1).clamp_min(1e-9)
                    # Two-sided |cos| (winding-agnostic); incidence folds into the
                    # effective distance, which picks the octave — no grazing cutoff.
                    cosang = (pnrm[pj] * dvec).sum(1).abs() / dist
                    d_eff = dist / cosang.clamp_min(1e-12)
                    band = torch.clamp(
                        torch.floor(torch.log2(d_eff / dmin_t[pj])), 0.0, float(n_oct - 1)
                    ).to(torch.int64)
                    sel = (d_eff <= dmax_t[pj]) & (band >= band_lo) & (band <= band_hi)
                    loc, pj, dist, band = loc[sel], pj[sel], dist[sel], band[sel]
                    if pj.shape[0] == 0:
                        continue
                    cam = cams_b[loc]
                    vis = ~_occluded(cam, ppos[pj], dist, t_lin)
                    loc, pj, dist, band = loc[vis], pj[vis], dist[vis], band[vis]
                    if pj.shape[0] == 0:
                        continue
                    cam = cams_b[loc]
                    vd = (cam - ppos[pj]) / dist[:, None]
                    # Direction cell (`_bin_of` in torch): cap if |cos| clears
                    # 1 − 1/b, else an azimuth cell of the remaining ring.
                    z = (vd * pnrm[pj]).sum(1).abs()
                    az = torch.atan2((vd * t2g[pj]).sum(1), (vd * t1g[pj]).sum(1))
                    ring = 1 + torch.clamp(
                        ((az + np.pi) / (2 * np.pi) * (b - 1)).to(torch.int64), 0, b - 2
                    )
                    bint = torch.where(z >= 1.0 - 1.0 / b, torch.zeros_like(ring), ring)
                    owed_t = octtop_t[pj] >= band
                    pj_np = pj.cpu().numpy()
                    seen[pj_np] = True
                    out.append(
                        (
                            (loc + b0).cpu().numpy(),
                            pj_np,
                            bint.cpu().numpy(),
                            band.cpu().numpy(),
                            owed_t.cpu().numpy(),
                        )
                    )
            if progress is not None:
                progress(prog_base + b1, total, "coverage")
            b0 = b1
        return out

    # Pass plan (see docstring): aperture buckets (bands 0..1, full set) +
    # pyramid bands ≥ 2, then the rescue.
    ap_hi = min(1, n_oct - 1)
    plan: list[tuple[np.ndarray, float, int, int]] = [
        (idx, float(d_min[idx].max()) * float(2 ** (ap_hi + 1)), 0, ap_hi)
        for idx in _scale_buckets(d_min)
    ]
    for o in range(2, n_oct):
        idx_o = np.nonzero(oct_top >= o)[0]
        if idx_o.size:
            plan.append((idx_o, float(d_min[idx_o].max()) * float(2 ** (o + 1)), o, o))
    total = (len(plan) + 1) * n_cand
    for pi, (idx_sub, r_pass, lo, hi) in enumerate(plan):
        for chunk in _pass(idx_sub, r_pass, lo, hi, pi * n_cand, total):
            cc_o.append(chunk[0])
            pp_o.append(chunk[1])
            bin_o.append(chunk[2])
            octv_o.append(chunk[3])
            owed_o.append(chunk[4])

    # RESCUE pass (`_rescue_pairs`): best effort for patches with no pair at
    # all, against their nearest candidates; the ray-march runs on CUDA.
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
        None if progress is None
        else (lambda done, tot: progress(len(plan) * n_cand + int(done / max(tot, 1) * n_cand), total, "coverage")),
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
    points, normals, albedo = _read_cloud(Path(surfels_path))
    lo = points.min(axis=0)
    hi = points.max(axis=0)

    # Feature-adaptive patches (curvature + texture gradient) from the surfels.
    if progress is not None:
        progress(0, 0, "patches")
    spacing = _feature_spacing(points, normals, albedo, params)
    keep = _adaptive_patches(points, normals, spacing, params.patch_min_spacing, rng)
    patch_pos = points[keep].astype(np.float32)
    patch_nrm = normals[keep].astype(np.float32)
    patch_feat = spacing[keep]
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
    # NESTED thinning matching the base cloud's (s_min/s)² keep rule. Total
    # demands ≈ 4/3 × patches.
    u = rng.random(n_patch)
    oct_top = np.minimum(
        np.floor(-np.log(np.maximum(u, 1e-12)) / np.log(4.0)).astype(np.int64),
        n_oct - 1,
    )

    # Candidate camera positions from the reachable near-surface band (no re-voxelization).
    candidates = _candidates(fs, params)
    if len(candidates) == 0:
        raise RuntimeError("no candidate camera positions (reachable free space empty)")

    # Coverage (per-pair direction cell + octave + owed) + greedy multicover over
    # IMAGES. Coverage is the long phase (a per-candidate occlusion ray-march on
    # CUDA), so it streams fine-grained progress (passes × candidates).
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
