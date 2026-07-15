"""Stage 4 — Coverage camera planner (feature-adaptive patches + greedy set-cover).

Picks the fewest camera POSITIONS in free space so every visible surface patch is
seen from ≥ K distinct azimuthal sectors, close enough to meet a feature-adaptive
footprint budget. The output feeds Stage 5 (reference renders) and yields the
occlusion-cull list as a byproduct.

CONNECTED (Option A): Stage 4 consumes the outputs of the earlier stages and loads
NO meshes and computes NO occupancy of its own —
  * **Stage 2 free-space grid** (`freespace.npz`): candidate camera positions
    (reachable free cells with enough clearance) + the fine occupancy the
    line-of-sight ray-march uses. No re-voxelization.
  * **Stage 3 surfel cloud** (`cloud.ply`): the patch source. Patches are a
    feature-adaptive thinning of the surfels, whose normals were already oriented to
    free space in Stage 3 — so the front-facing test is reliable regardless of the
    mesh's original winding.

Pipeline:
  1. Read the surfel cloud → points + oriented normals + albedo.
  2. Feature-adaptive PATCHES: spacing shrinks with local detail (curvature via
     normal variance, texture via albedo variance); denser where detail is.
  3. CANDIDATE positions = reachable free cells (Stage 2) with clearance ≥
     collision_clearance, subsampled denser where clearance is small.
  4. COVERAGE: a candidate covers a patch if front-facing, within the patch's
     feature-scaled view distance, and unoccluded (fine-grid ray-march). Cubemaps
     mean orientation isn't a variable, so a covered patch lands on some face.
  5. GREEDY multicover until every visible patch hits K sectors + a near view.

Output: `cameras.json` — a shared cube-face `intrinsics` block (90° FOV, render
resolution, DERIVED footprint budget), the six `cube_faces`, and the chosen camera
POSITIONS each tagged with the cube faces worth rendering + coverage — plus
`patches.bin` (packed float32 [x,y,z, nx,ny,nz, feature_scale, sectors_seen] per
patch) and a summary. CUBEMAP-NATIVE: each position renders as up to six 90° pinhole
faces in Stage 5, so no single look direction is emitted.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from math import radians, tan
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
    """Stage-4 knobs (overview §9). Defaults target a room-scale cell. Occupancy
    pitch/margin now live in the Stage-2 grid, not here."""

    patch_min_spacing: float = 0.06   # s_min (m): finest patch spacing = footprint detail
    patch_max_spacing: float = 0.30   # s_max (m): flat-region patch spacing
    curvature_k: float = 14.0         # curvature → spacing sensitivity
    tex_k: float = 8.0                # texture-gradient → spacing sensitivity
    collision_clearance: float = 0.25  # cameras stay ≥ this from any surface (m)
    angles_per_patch: int = 3         # K: distinct viewing ANGLES (sectors) per patch
    angular_sectors: int = 8          # azimuthal quantization around the patch normal
    near_frac: float = 0.5            # a "near" (detail) view is within near_frac*view_dist
    min_gain: int = 1                 # stop once the best camera adds < this many
    max_candidates: int = 5000        # cap on candidate camera positions
    # Cube-face reference-render intrinsics (SHARED with Stage 5). FOV fixed at 90°
    # (six faces tile 360°); footprint_k (hence view_dist) is DERIVED from the render
    # resolution + min_px_per_patch so coverage distances match what Stage 5 renders.
    face_fov_deg: float = 90.0
    render_resolution: int = 512
    min_px_per_patch: float = 10.0
    view_dist_min: float = 0.5        # (m)
    view_dist_max: float = 4.0        # (m)
    seed: int = 0

    @property
    def focal_px(self) -> float:
        """Pinhole focal length in pixels of one cube face: (R/2) / tan(fov/2).
        At the fixed 90° FOV this is simply R/2."""
        return (self.render_resolution / 2.0) / tan(radians(self.face_fov_deg) / 2.0)

    @property
    def footprint_k(self) -> float:
        """view_dist = footprint_k * feature_scale. Derived so a patch of size s seen
        at view_dist spans exactly `min_px_per_patch` pixels."""
        return self.focal_px / self.min_px_per_patch

    def as_summary(self) -> dict[str, Any]:
        return {
            "patch_min_spacing": self.patch_min_spacing,
            "patch_max_spacing": self.patch_max_spacing,
            "collision_clearance": self.collision_clearance,
            "angles_per_patch": self.angles_per_patch,
            "angular_sectors": self.angular_sectors,
            "near_frac": self.near_frac,
            "max_candidates": self.max_candidates,
            "face_fov_deg": self.face_fov_deg,
            "render_resolution": self.render_resolution,
            "min_px_per_patch": self.min_px_per_patch,
            "footprint_k": round(self.footprint_k, 3),
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


def _candidates(fs: FreeSpace, p: PlanParams, rng: np.random.Generator) -> np.ndarray:
    """Candidate camera positions: reachable free cells (Stage 2) with clearance ≥
    collision_clearance, subsampled to ≤ max_candidates with weight ∝ 1/clearance²
    (denser near surfaces). Returns (M,3) world points."""
    centers, clv = fs.free_candidates(p.collision_clearance)
    if len(centers) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if len(centers) > p.max_candidates:
        w = 1.0 / np.maximum(clv, fs.pitch) ** 2
        w /= w.sum()
        pick = rng.choice(len(centers), size=p.max_candidates, replace=False, p=w)
        centers = centers[pick]
    return centers


def _visible(
    cam: np.ndarray, patch_pos: np.ndarray, fs: FreeSpace, n_steps: int
) -> np.ndarray:
    """Line-of-sight from one camera to many patches: True where no solid FINE voxel
    lies strictly between them. Samples n_steps points along each segment and skips
    the endpoints (camera cell, patch's own surface cell)."""
    m = len(patch_pos)
    if m == 0:
        return np.zeros(0, dtype=bool)
    d = patch_pos - cam
    dist = np.linalg.norm(d, axis=1)
    dist = np.where(dist < 1e-6, 1e-6, dist)
    t = np.linspace(0.0, 1.0, n_steps)  # (K,)
    pts = cam[None, None, :] + t[None, :, None] * d[:, None, :]  # (m,K,3)
    hit = fs.fine_occupied(pts.reshape(-1, 3)).reshape(m, n_steps)  # (m,K)
    pf = fs.pitch_fine
    tvalid = (t[None, :] > (pf / dist)[:, None]) & (
        t[None, :] < 1.0 - (1.5 * pf / dist)[:, None]
    )
    return ~(hit & tvalid).any(axis=1)


def _tangent_frames(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-normal orthonormal tangent basis (t1, t2) so a viewing direction can be
    binned into an azimuth around the patch normal."""
    ref = np.where(
        np.abs(normals[:, 2:3]) < 0.9,
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    t1 = np.cross(normals, ref)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-9
    t2 = np.cross(normals, t1)
    return t1.astype(np.float32), t2.astype(np.float32)


def _build_coverage(
    candidates: np.ndarray,
    patch_pos: np.ndarray,
    patch_nrm: np.ndarray,
    view_dist: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    fs: FreeSpace,
    p: PlanParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Covering (candidate, patch) pairs, each tagged with the azimuthal SECTOR the
    camera views the patch from and whether it's a NEAR (detail-distance) view. A
    pair exists if the camera is within the patch's view distance from EITHER side
    (two-sided; winding-agnostic) and unoccluded. Returns COO arrays (cand_idx,
    patch_idx, sector, is_near)."""
    tree = cKDTree(patch_pos)
    n_steps = int(np.ceil(p.view_dist_max / fs.pitch_fine)) + 2
    a = p.angular_sectors
    cc: list[np.ndarray] = []
    pp: list[np.ndarray] = []
    sec: list[np.ndarray] = []
    near: list[np.ndarray] = []
    for ci, cam in enumerate(candidates):
        idx = np.asarray(tree.query_ball_point(cam, p.view_dist_max), dtype=int)
        if len(idx) == 0:
            continue
        d = patch_pos[idx] - cam
        dist = np.linalg.norm(d, axis=1)
        dist = np.where(dist < 1e-6, 1e-6, dist)
        # TWO-SIDED coverage: a candidate may see a patch from EITHER side (|cos|),
        # not just the side its normal points. TRELLIS winding is unreliable and its
        # meshes are thin shells with reachable free space on BOTH sides — so Stage
        # 3's orient-to-free-space keeps the original (often inward) winding, and a
        # SIGNED front-facing test then wrongly culls ~half the surface. A small
        # grazing cutoff drops near-edge-on views; the occlusion ray-march (below)
        # still stops a camera seeing a patch through solid geometry.
        cos = np.abs(np.einsum("mc,mc->m", patch_nrm[idx], d)) / dist
        sel = (cos > 0.1) & (dist <= view_dist[idx])
        cand, cdist = idx[sel], dist[sel]
        if len(cand) == 0:
            continue
        vis = _visible(cam, patch_pos[cand], fs, n_steps)
        hit, hdist = cand[vis], cdist[vis]
        if len(hit) == 0:
            continue
        vd = (cam - patch_pos[hit]) / hdist[:, None]
        az = np.arctan2(
            np.einsum("mc,mc->m", vd, t2[hit]), np.einsum("mc,mc->m", vd, t1[hit])
        )
        sector = np.clip(((az + np.pi) / (2 * np.pi) * a).astype(np.int64), 0, a - 1)
        cc.append(np.full(len(hit), ci, dtype=np.int64))
        pp.append(hit.astype(np.int64))
        sec.append(sector)
        near.append(hdist <= p.near_frac * view_dist[hit])
    if not cc:
        e = np.zeros(0, dtype=np.int64)
        return e, e, e, np.zeros(0, dtype=bool)
    return (
        np.concatenate(cc),
        np.concatenate(pp),
        np.concatenate(sec),
        np.concatenate(near),
    )


def _greedy_angular(
    cc: np.ndarray,
    pp: np.ndarray,
    sector: np.ndarray,
    is_near: np.ndarray,
    n_cand: int,
    n_patch: int,
    k: int,
    min_gain: int,
    a: int,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Greedy coverage where a patch is satisfied by ≥ k DISTINCT azimuthal sectors
    AND ≥ 1 near view. Each step takes the camera advancing the most patches; stops
    below `min_gain`. Returns (chosen candidate indices, sector bitmask, has-near)."""
    angmask = np.zeros(n_patch, dtype=np.int64)
    hasnear = np.zeros(n_patch, dtype=bool)
    lut = np.array([bin(i).count("1") for i in range(1 << a)], dtype=np.int16)
    pair_bit = np.int64(1) << sector
    taken = np.zeros(n_cand, dtype=bool)
    chosen: list[int] = []
    while True:
        pop = lut[angmask[pp]]
        new_sector = (angmask[pp] & pair_bit) == 0
        ang_contrib = new_sector & (pop < k)
        near_contrib = is_near & (~hasnear[pp])
        contrib = ang_contrib | near_contrib
        if not contrib.any():
            break
        gain = np.bincount(cc, weights=contrib.astype(np.float64), minlength=n_cand)
        gain[taken] = -1.0
        c = int(np.argmax(gain))
        if gain[c] < max(min_gain, 1):
            break
        taken[c] = True
        chosen.append(c)
        m = cc == c
        np.bitwise_or.at(angmask, pp[m], pair_bit[m])
        near_hits = pp[m][is_near[m]]
        if len(near_hits):
            hasnear[near_hits] = True
    return chosen, angmask, hasnear


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
        progress(0, 3, "load")
    fs = load_free_space(Path(freespace_path))
    points, normals, albedo = _read_cloud(Path(surfels_path))
    lo = points.min(axis=0)
    hi = points.max(axis=0)

    # Feature-adaptive patches (curvature + texture gradient) from the surfels.
    if progress is not None:
        progress(1, 3, "patches")
    spacing = _feature_spacing(points, normals, albedo, params)
    keep = _adaptive_patches(points, normals, spacing, params.patch_min_spacing, rng)
    patch_pos = points[keep].astype(np.float32)
    patch_nrm = normals[keep].astype(np.float32)
    patch_feat = spacing[keep]
    t1, t2 = _tangent_frames(patch_nrm)
    view_dist = np.clip(
        params.footprint_k * patch_feat, params.view_dist_min, params.view_dist_max
    ).astype(np.float32)

    # Candidate camera positions from the reachable free space (no re-voxelization).
    candidates = _candidates(fs, params, rng)
    if len(candidates) == 0:
        raise RuntimeError("no candidate camera positions (reachable free space empty)")

    # Coverage (per-pair sector + near) + angular greedy.
    if progress is not None:
        progress(2, 3, "coverage")
    n_patch = len(patch_pos)
    k = params.angles_per_patch
    a = params.angular_sectors
    cc, pp, sector, is_near = _build_coverage(
        candidates, patch_pos, patch_nrm, view_dist, t1, t2, fs, params
    )
    chosen, angmask, hasnear = _greedy_angular(
        cc, pp, sector, is_near, len(candidates), n_patch, k, params.min_gain, a
    )

    # Per-patch stats: distinct angles seen, and satisfaction (≥K angles + near view).
    lut = np.array([bin(i).count("1") for i in range(1 << a)], dtype=np.int16)
    sectors_seen = lut[angmask].astype(np.int32)
    satisfied_mask = (sectors_seen >= k) & hasnear
    seen_any = np.zeros(n_patch, dtype=bool)
    if len(pp):
        seen_any[np.unique(pp)] = True
    satisfied = int(satisfied_mask.sum())
    seen_once = int(seen_any.sum())
    occluded = int((~seen_any).sum())

    # Per-camera cube faces + coverage count (group the COO pairs by candidate).
    # CUBEMAP-NATIVE: emit only the faces that actually saw covered patches. Also
    # build the per-patch VIEW INDEX (patch → [(camera_index, face_index), …]) so the
    # debug viewer can map a selected surface patch to its Stage-5 reference images.
    cameras: list[dict[str, Any]] = []
    patch_views: list[list[list[int]]] = [[] for _ in range(n_patch)]
    if len(cc):
        order = np.argsort(cc, kind="stable")
        cc_s, pp_s = cc[order], pp[order]
        for out_idx, ci in enumerate(chosen):
            a0 = int(np.searchsorted(cc_s, ci, "left"))
            a1 = int(np.searchsorted(cc_s, ci, "right"))
            seen = pp_s[a0:a1]
            cam = candidates[ci]
            faces: list[dict[str, Any]] = []
            if len(seen):
                face_idx = _face_of(patch_pos[seen] - cam)  # per-seen-patch face
                counts = np.bincount(face_idx, minlength=len(CUBE_FACE_NAMES))
                faces = [
                    {"dir": CUBE_FACE_NAMES[i], "covers": int(counts[i])}
                    for i in range(len(CUBE_FACE_NAMES))
                    if counts[i] > 0
                ]
                for p, fi in zip(seen.tolist(), face_idx.tolist()):
                    patch_views[p].append([out_idx, fi])
            cameras.append(
                {
                    "pos": [round(float(v), 4) for v in cam],
                    "faces": faces,
                    "covers": int(len(seen)),
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

    # patches.bin: [x,y,z, nx,ny,nz, feature_scale, sectors_seen] × N.
    pdata = np.concatenate(
        [
            patch_pos.astype("<f4"),
            patch_nrm.astype("<f4"),
            patch_feat.reshape(-1, 1).astype("<f4"),
            sectors_seen.reshape(-1, 1).astype("<f4"),
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
        "angles_per_patch": k,
        "angular_sectors": a,
        "coverage": {
            "satisfied": satisfied,
            "satisfied_pct": round(100.0 * satisfied / max(n_patch, 1), 1),
            "seen_at_least_once": seen_once,
            "occlusion_culled": occluded,
            "mean_angles_seen": (
                round(float(sectors_seen[seen_any].mean()), 2) if seen_once else 0.0
            ),
        },
        "scene_aabb": {"min": lo.tolist(), "max": hi.tolist()},
        "params": params.as_summary(),
    }

    # Shared cube-face intrinsics Stage 5 renders with (and Stage 6 trains against).
    # near < collision_clearance so a surface at the clearance limit isn't clipped;
    # far spans the full free-space grid diagonal (the play volume).
    grid_diag = float(np.linalg.norm(np.array(fs.occ_fine.shape) * fs.pitch_fine))
    intrinsics = {
        "face_fov_deg": params.face_fov_deg,
        "resolution": params.render_resolution,
        "min_px_per_patch": params.min_px_per_patch,
        "footprint_k": round(params.footprint_k, 4),
        "focal_px": round(params.focal_px, 3),
        "near": round(min(0.05, params.collision_clearance * 0.5), 4),
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
