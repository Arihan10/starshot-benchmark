"""Stage 4 — Coverage camera planner (feature-adaptive patches + greedy set-cover).

Picks the fewest camera POSITIONS in free space so every visible surface patch is
seen from ≥ K distinct cameras, close enough to meet a feature-adaptive footprint
budget. Per the overview (§5 Stage 4): the output feeds Stage 5 (reference renders)
and yields the occlusion-cull list as a byproduct.

Self-contained (like stage1/2/3): takes explicit paths; the server resolves the
cell and de-optimizes a library build to vanilla first. One surface-sampling pass
drives everything — occupancy grid, clearance field, adaptive patches, candidates.

Pipeline:
  1. Sample the placed meshes (blue-noise) → points + normals + triangle areas.
  2. Feature-adaptive PATCHES: target spacing s(x) shrinks with local detail
     (curvature via normal variance, and triangle size); denser where detail is,
     so the plan captures the scene's real fidelity. (Texture-gradient detail —
     the "busy wallpaper" case — is the next signal to add.)
  3. Occupancy + clearance over the scene AABB PLUS an exterior margin, so cameras
     can also see the outer silhouette (both faces of a dividing wall, etc. — only
     genuinely buried surface is culled).
  4. CANDIDATE positions = safe free voxels (clearance ≥ collision_clearance),
     denser where clearance is small.
  5. COVERAGE: a candidate covers a patch if it's front-facing, within the patch's
     feature-scaled view distance, and has line-of-sight (voxel ray-march on the
     occupancy grid — no mesh raycasting). Cubemaps mean orientation isn't a
     variable, so a covered patch lands on some face.
  6. GREEDY multicover (sparse mat-vec): repeatedly take the candidate covering the
     most still-under-covered patches until all visible patches hit K, or no gain
     remains (the rest → occlusion-cull list).

Output: `cameras.json` — a shared cube-face `intrinsics` block (90° FOV, render
resolution, and the footprint budget DERIVED from them), the six `cube_faces`, and
the chosen camera POSITIONS, each tagged with the cube faces worth rendering +
coverage — plus `patches.bin` (packed float32 [x,y,z, nx,ny,nz, feature_scale,
covered_count] per patch) and a summary. The plan is CUBEMAP-NATIVE: each position
renders as up to six 90° pinhole faces in Stage 5, so no single look direction is
emitted.
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
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

from splat import stage2  # per-texel albedo for the texture-detail signal

logging.getLogger("trimesh").setLevel(logging.ERROR)

CAMERAS_NAME = "cameras.json"
PATCHES_NAME = "patches.bin"

# progress(done, total, current_id) — called during the surface-sampling pass.
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
    """Stage-4 knobs (overview §9). Defaults target a room-scale cell."""

    patch_min_spacing: float = 0.06   # s_min (m): finest patch spacing = footprint detail
    patch_max_spacing: float = 0.30   # s_max (m): flat-region patch spacing
    curvature_k: float = 14.0         # curvature → spacing sensitivity
    tri_k: float = 3.0                # triangle-size → spacing sensitivity
    tex_k: float = 8.0                # texture-gradient → spacing sensitivity
    collision_clearance: float = 0.25  # cameras stay ≥ this from any surface (m)
    margin: float = 1.5               # exterior play-volume margin around the AABB (m)
    pitch: float = 0.12               # occupancy / visibility voxel size (m)
    angles_per_patch: int = 3         # K: distinct viewing ANGLES (sectors) per patch
    angular_sectors: int = 8          # azimuthal quantization around the patch normal
    near_frac: float = 0.5            # a "near" (detail) view is within near_frac*view_dist
    min_gain: int = 1                 # stop once the best camera adds < this many
                                      # new patch-satisfactions (truncates the
                                      # diminishing tail; 1 = cover all reachable)
    max_candidates: int = 5000        # cap on candidate camera positions (more =
                                      # more coverage of hard-to-reach patches)
    # Cube-face reference-render intrinsics (SHARED with Stage 5). A cubemap tiles
    # every direction with six square 90° faces, so face_fov_deg is fixed at 90;
    # render_resolution is each face's pixel size and min_px_per_patch the sharpness
    # target. footprint_k (hence view_dist) is DERIVED from these (see the property)
    # so the plan's coverage distances match exactly what Stage 5 renders, instead
    # of a magic constant that silently bakes in an assumed resolution/FOV.
    face_fov_deg: float = 90.0        # cubemap face FOV; keep at 90 (six faces tile 360°)
    render_resolution: int = 512      # each cube face is R×R px (overview §12: 512–1024)
    min_px_per_patch: float = 10.0    # a patch must span ≥ this many px in its best view
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
        """view_dist = footprint_k * feature_scale. Derived so a patch of size s
        seen at view_dist spans exactly `min_px_per_patch` pixels: inverting
        px ≈ focal_px * s / d at px = min_px_per_patch gives
        d = (focal_px / min_px_per_patch) * s."""
        return self.focal_px / self.min_px_per_patch

    def as_summary(self) -> dict[str, Any]:
        return {
            "patch_min_spacing": self.patch_min_spacing,
            "patch_max_spacing": self.patch_max_spacing,
            "collision_clearance": self.collision_clearance,
            "margin": self.margin,
            "pitch": self.pitch,
            "angles_per_patch": self.angles_per_patch,
            "angular_sectors": self.angular_sectors,
            "near_frac": self.near_frac,
            "max_candidates": self.max_candidates,
            "face_fov_deg": self.face_fov_deg,
            "render_resolution": self.render_resolution,
            "min_px_per_patch": self.min_px_per_patch,
            "footprint_k": round(self.footprint_k, 3),
        }


def _iter_geoms(mesh: trimesh.Trimesh | trimesh.Scene) -> list[trimesh.Trimesh]:
    if isinstance(mesh, trimesh.Scene):
        return [g for g in mesh.geometry.values() if hasattr(g, "faces")]
    return [mesh]


def placed_object_ids(raw_dir: Path) -> list[str]:
    return sorted(
        p.name[: -len(".glb")]
        for p in raw_dir.glob("*.glb")
        if not p.name.endswith(".raw.glb")
    )


def _sample_surface(
    raw_dir: Path, ids: list[str], s_min: float, progress: ProgressCb | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Blue-noise sample every placed mesh at ~s_min. Returns (points, normals,
    tri_area, albedo, aabb_min, aabb_max) — the shared pass that seeds everything.
    Albedo (per-texel base colour) feeds the texture-detail signal."""
    pos_parts: list[np.ndarray] = []
    nrm_parts: list[np.ndarray] = []
    area_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    lo = np.array([np.inf, np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf, -np.inf])
    total = len(ids)
    for done, node_id in enumerate(ids, start=1):
        try:
            m = trimesh.load(raw_dir / f"{node_id}.glb", process=False)
            for g in _iter_geoms(m):
                if len(g.faces) == 0 or g.area <= 0:
                    continue
                b = np.asarray(g.bounds, dtype=float)
                lo, hi = np.minimum(lo, b[0]), np.maximum(hi, b[1])
                n = int(g.area / (s_min * s_min) * 1.5) + 8
                try:
                    pts, fidx = trimesh.sample.sample_surface_even(g, n, radius=s_min)
                except Exception:
                    pts, fidx = trimesh.sample.sample_surface(g, n)
                if len(pts) == 0:
                    continue
                fidx = np.asarray(fidx)
                pts = np.asarray(pts, dtype=np.float32)
                nrm = np.asarray(g.face_normals[fidx], dtype=np.float64)
                ln = np.linalg.norm(nrm, axis=1, keepdims=True)
                ln[ln == 0] = 1.0
                try:
                    col = stage2.surfel_colors(g, pts, fidx)[:, :3].astype(np.float32)
                except Exception:
                    col = np.full((len(pts), 3), 0.6, dtype=np.float32)
                pos_parts.append(pts)
                nrm_parts.append((nrm / ln).astype(np.float32))
                area_parts.append(np.asarray(g.area_faces[fidx], dtype=np.float32))
                col_parts.append(col)
            del m
        except Exception:
            pass
        if progress is not None:
            progress(done, total, node_id)
    if not pos_parts:
        raise RuntimeError("no surface sampled (every mesh failed or was empty)")
    return (
        np.concatenate(pos_parts),
        np.concatenate(nrm_parts),
        np.concatenate(area_parts),
        np.concatenate(col_parts),
        lo,
        hi,
    )


def _feature_spacing(
    points: np.ndarray,
    normals: np.ndarray,
    tri_area: np.ndarray,
    albedo: np.ndarray,
    p: PlanParams,
) -> np.ndarray:
    """Per-point target spacing s(x): small where detail is high. Combines
    curvature (local normal variance), triangle size, and texture gradient (local
    albedo variance — the "busy wallpaper on a flat wall" case); the densest wins."""
    n = len(points)
    tree = cKDTree(points)
    k = min(9, n)
    _, idx = tree.query(points, k=k)
    neigh_idx = idx[:, 1:] if k > 1 else idx  # exclude self
    # Curvature ≈ 1 − mean cosine between a point's normal and its neighbours'.
    cos = np.clip(np.einsum("nkc,nc->nk", normals[neigh_idx], normals), -1.0, 1.0)
    curv = 1.0 - cos.mean(axis=1)  # 0 flat → ~2 sharp
    s_curv = p.patch_max_spacing / (1.0 + p.curvature_k * curv)
    # Small source triangle ⇒ fine detail in the original mesh.
    s_tri = p.tri_k * np.sqrt(np.maximum(tri_area, 1e-9))
    # Texture gradient ≈ local albedo variation among neighbours (per-channel std).
    tex_var = albedo[neigh_idx].std(axis=1).mean(axis=1)
    s_tex = p.patch_max_spacing / (1.0 + p.tex_k * tex_var)
    s = np.minimum(np.minimum(s_curv, s_tri), s_tex)
    return np.clip(s, p.patch_min_spacing, p.patch_max_spacing).astype(np.float32)


def _adaptive_patches(
    points: np.ndarray,
    normals: np.ndarray,
    spacing: np.ndarray,
    s_min: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Thin the dense blue-noise sample to feature-adaptive density: keep point i
    with probability (s_min / s_i)^2, so flat regions get sparse patches and
    detailed regions keep (near-)all of theirs. Returns the kept indices."""
    keep_p = np.clip((s_min / spacing) ** 2, 0.0, 1.0)
    keep = rng.random(len(points)) < keep_p
    return np.nonzero(keep)[0]


def _occupancy_clearance(
    points: np.ndarray, lo: np.ndarray, hi: np.ndarray, pitch: float, margin: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solid/empty grid (from binned surface points) + clearance field (EDT, m)
    over the AABB grown by `margin` (so exterior cameras exist). Returns
    (occupancy, clearance, origin)."""
    origin = lo - margin
    dims = np.ceil((hi - lo + 2 * margin) / pitch).astype(int) + 1
    dims = np.maximum(dims, 1)
    nx, ny, nz = (int(dims[0]), int(dims[1]), int(dims[2]))
    idx = np.floor((points - origin) / pitch).astype(np.int64)
    inb = np.all((idx >= 0) & (idx < dims), axis=1)
    idx = idx[inb]
    occ = np.zeros((nx, ny, nz), dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    clearance = ndimage.distance_transform_edt(~occ).astype(np.float32) * pitch
    return occ, clearance, origin


def _candidates(
    occ: np.ndarray,
    clearance: np.ndarray,
    origin: np.ndarray,
    pitch: float,
    p: PlanParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Candidate camera positions: free voxels with clearance ≥ collision_clearance,
    subsampled to ≤ max_candidates with weight ∝ 1/clearance² (denser near
    surfaces, where more/closer vantages are needed). Returns (M,3) world points."""
    free = (~occ) & (clearance >= p.collision_clearance)
    cells = np.argwhere(free)
    if len(cells) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    centers = (origin + (cells + 0.5) * pitch).astype(np.float32)
    if len(centers) > p.max_candidates:
        cl = clearance[free]
        w = 1.0 / np.maximum(cl, pitch) ** 2
        w /= w.sum()
        pick = rng.choice(len(centers), size=p.max_candidates, replace=False, p=w)
        centers = centers[pick]
    return centers


def _visible(
    cam: np.ndarray,
    patch_pos: np.ndarray,
    occ: np.ndarray,
    origin: np.ndarray,
    pitch: float,
    n_steps: int,
) -> np.ndarray:
    """Line-of-sight from one camera to many patches: True where no solid voxel
    lies strictly between them. Samples n_steps points along each segment (spacing
    ≤ pitch for in-range patches) and skips the endpoints (camera cell, patch's own
    surface cell)."""
    m = len(patch_pos)
    if m == 0:
        return np.zeros(0, dtype=bool)
    d = patch_pos - cam  # (m,3)
    dist = np.linalg.norm(d, axis=1)
    dist = np.where(dist < 1e-6, 1e-6, dist)
    t = np.linspace(0.0, 1.0, n_steps)  # (K,)
    pts = cam[None, None, :] + t[None, :, None] * d[:, None, :]  # (m,K,3)
    idx = np.floor((pts - origin) / pitch).astype(np.int64)  # (m,K,3)
    dims = np.array(occ.shape)
    idx = np.clip(idx, 0, dims - 1)
    hit = occ[idx[:, :, 0], idx[:, :, 1], idx[:, :, 2]]  # (m,K)
    # Only count samples strictly between the endpoints: skip ~1 voxel from the
    # camera and ~1.5 voxels before the patch (its own surface voxel).
    tvalid = (t[None, :] > (pitch / dist)[:, None]) & (
        t[None, :] < 1.0 - (1.5 * pitch / dist)[:, None]
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
    occ: np.ndarray,
    origin: np.ndarray,
    p: PlanParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Covering (candidate, patch) pairs, each tagged with the azimuthal SECTOR the
    camera views the patch from and whether it's a NEAR (detail-distance) view. A
    pair exists if the camera is front-facing, within the patch's view distance,
    and unoccluded. Returns COO arrays (cand_idx, patch_idx, sector, is_near)."""
    tree = cKDTree(patch_pos)
    n_steps = int(np.ceil(p.view_dist_max / p.pitch)) + 2
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
        facing = np.einsum("mc,mc->m", patch_nrm[idx], -d) > 0
        sel = facing & (dist <= view_dist[idx])
        cand, cdist = idx[sel], dist[sel]
        if len(cand) == 0:
            continue
        vis = _visible(cam, patch_pos[cand], occ, origin, p.pitch, n_steps)
        hit, hdist = cand[vis], cdist[vis]
        if len(hit) == 0:
            continue
        # Azimuth of the patch→camera direction in each patch's tangent frame.
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
    AND ≥ 1 near view. Each step takes the camera advancing the most patches (adding
    a new sector, or the missing near view); stops below `min_gain`. Returns (chosen
    candidate indices, per-patch sector bitmask, per-patch has-near)."""
    angmask = np.zeros(n_patch, dtype=np.int64)
    hasnear = np.zeros(n_patch, dtype=bool)
    lut = np.array([bin(i).count("1") for i in range(1 << a)], dtype=np.int16)
    pair_bit = np.int64(1) << sector  # (nnz,) the sector bit each pair would set
    taken = np.zeros(n_cand, dtype=bool)
    chosen: list[int] = []
    while True:
        pop = lut[angmask[pp]]                          # distinct sectors so far
        new_sector = (angmask[pp] & pair_bit) == 0
        ang_contrib = new_sector & (pop < k)            # adds a still-needed sector
        near_contrib = is_near & (~hasnear[pp])         # supplies the missing near view
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
    raw_dir: Path,
    out_path: Path,
    params: PlanParams = PlanParams(),
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Plan coverage cameras for one cell; write `cameras.json` (to `out_path`) +
    `patches.bin` (beside it) and return a summary."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"placed-mesh dir not found: {raw_dir}")
    ids = placed_object_ids(raw_dir)
    if not ids:
        raise FileNotFoundError(f"no placed meshes in {raw_dir}")
    rng = np.random.default_rng(params.seed)

    points, normals, tri_area, albedo, lo, hi = _sample_surface(
        raw_dir, ids, params.patch_min_spacing, progress
    )

    # Feature-adaptive patches (curvature + triangle size + texture gradient).
    spacing = _feature_spacing(points, normals, tri_area, albedo, params)
    keep = _adaptive_patches(points, normals, spacing, params.patch_min_spacing, rng)
    patch_pos = points[keep]
    patch_nrm = normals[keep]
    patch_feat = spacing[keep]
    t1, t2 = _tangent_frames(patch_nrm)
    view_dist = np.clip(
        params.footprint_k * patch_feat, params.view_dist_min, params.view_dist_max
    ).astype(np.float32)

    # Occupancy + clearance (+ exterior margin) and candidate camera positions.
    occ, clearance, origin = _occupancy_clearance(
        points, lo, hi, params.pitch, params.margin
    )
    candidates = _candidates(occ, clearance, origin, params.pitch, params, rng)
    if len(candidates) == 0:
        raise RuntimeError("no candidate camera positions (free space empty)")

    # Coverage (per-pair sector + near) + angular greedy.
    n_patch = len(patch_pos)
    k = params.angles_per_patch
    a = params.angular_sectors
    cc, pp, sector, is_near = _build_coverage(
        candidates, patch_pos, patch_nrm, view_dist, t1, t2, occ, origin, params
    )
    chosen, angmask, hasnear = _greedy_angular(
        cc, pp, sector, is_near, len(candidates), n_patch, k, params.min_gain, a
    )

    # Per-patch stats: distinct angles seen, and satisfaction (≥K angles + a near view).
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
    # CUBEMAP-NATIVE: each chosen POSITION renders as up to six 90° pinhole faces in
    # Stage 5; we emit only the faces that actually saw covered patches (the rest
    # stare into void/backfaces), so Stage 5 can skip rendering empty faces.
    cameras: list[dict[str, Any]] = []
    if len(cc):
        order = np.argsort(cc, kind="stable")
        cc_s, pp_s = cc[order], pp[order]
        for ci in chosen:
            a0 = int(np.searchsorted(cc_s, ci, "left"))
            a1 = int(np.searchsorted(cc_s, ci, "right"))
            seen = pp_s[a0:a1]
            cam = candidates[ci]
            faces: list[dict[str, Any]] = []
            if len(seen):
                counts = np.bincount(
                    _face_of(patch_pos[seen] - cam), minlength=len(CUBE_FACE_NAMES)
                )
                faces = [
                    {"dir": CUBE_FACE_NAMES[i], "covers": int(counts[i])}
                    for i in range(len(CUBE_FACE_NAMES))
                    if counts[i] > 0
                ]
            cameras.append(
                {
                    "pos": [round(float(v), 4) for v in cam],
                    "faces": faces,
                    "covers": int(len(seen)),
                }
            )

    # patches.bin: [x,y,z, nx,ny,nz, feature_scale, sectors_seen] × N (the viewer
    # colours each patch by how many distinct angles saw it; 0 = occlusion-culled).
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

    # Shared cube-face intrinsics Stage 5 renders with (and Stage 6 trains against):
    # FOV is fixed at 90°, so `resolution` alone sets sharpness and `footprint_k` is
    # the derived value the plan above actually used. near < collision_clearance so a
    # surface at the clearance limit isn't clipped; far spans the padded play volume.
    padded_ext = (hi - lo) + 2.0 * params.margin
    intrinsics = {
        "face_fov_deg": params.face_fov_deg,
        "resolution": params.render_resolution,
        "min_px_per_patch": params.min_px_per_patch,
        "footprint_k": round(params.footprint_k, 4),
        "focal_px": round(params.focal_px, 3),
        "near": round(min(0.05, params.collision_clearance * 0.5), 4),
        "far": round(float(np.linalg.norm(padded_ext)), 3),
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
    return summary
