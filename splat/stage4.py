"""Stage 4 — Coverage camera planner (density-carried detail + greedy set-cover).

Picks the fewest camera POSITIONS in free space so every visible surface patch is
seen from ≥ K distinct azimuthal sectors, close enough to meet a feature-adaptive
footprint budget. The output feeds Stage 5 (reference renders) and yields the
occlusion-cull list as a byproduct.

CONNECTED (Option A): Stage 4 consumes the outputs of the earlier stages and loads
NO meshes and computes NO occupancy of its own —
  * **Stage 2 free-space grid** (`freespace.npz`): candidate camera positions
    (reachable free cells with enough clearance) + the OPAQUE fine occupancy the
    line-of-sight ray-march uses (glass-classed cells occupy space but pass
    light, so geometry behind windows is coverable). No re-voxelization.
  * **Stage 3 surfel cloud** (`cloud.ply`): the patch source. Stage 3 samples at
    FEATURE-ADAPTIVE density (finer near creases/boundaries/thin members), so
    the cloud's local density IS the detail field — Stage 4 reads it (each
    point's local sample spacing) rather than re-detecting detail from cloud
    statistics, per the measure-once-consume-downstream principle. Normals were
    already oriented to free space in Stage 3, so the front-facing test is
    reliable regardless of the mesh's original winding.

Pipeline:
  1. Read the surfel cloud → points + oriented normals.
  2. PATCHES: a UNIFORM thinning (`patch_fraction`) of the surfels — uniform so
     the patches inherit Stage 3's adaptive density profile (an adaptive re-thin
     would apply detail weighting twice). Each patch's feature scale = its local
     sample spacing (mean 3-NN distance), which drives its view distance.
  3. CANDIDATE positions = reachable free cells (Stage 2) with clearance ≥
     collision_clearance: a uniform ~candidate_spacing lattice PLUS a refined
     full-resolution tier inside fine patches' near-view shells — the standpoint
     SUPPLY follows the cloud's density field, so close-up demands are actually
     satisfiable (engages only where demand outruns the lattice).
  4. COVERAGE: a candidate covers a patch if front-facing, within the patch's
     feature-scaled view distance, and unoccluded (fine-grid ray-march). Cubemaps
     mean orientation isn't a variable, so a covered patch lands on some face.
  5. GREEDY multicover until every visible patch hits K sectors + a near view +
     a head-on view (anti-grazing: sectors alone can all be edge-on).

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
    pitch/margin live in the Stage-2 grid; DETAIL lives in the Stage-3 cloud's
    density (patches inherit it via uniform thinning, so there are no detector
    knobs here any more)."""

    patch_fraction: float = 0.5       # uniform fraction of surfels kept as patches
    collision_clearance: float = 0.25  # cameras stay ≥ this from any surface (m)
    angles_per_patch: int = 3         # K: distinct viewing ANGLES (sectors) per patch
    angular_sectors: int = 8          # azimuthal quantization around the patch normal
    near_frac: float = 0.5            # a "near" (detail) view is within near_frac*view_dist
    # ≥1 view per patch must be HEAD-ON: |incidence cosine| at least this (0.5 =
    # within 60° of the normal). Azimuth sectors alone can all be satisfied by
    # grazing views, which supervise silhouettes but never the surface's face.
    headon_cos: float = 0.5
    min_gain: int = 1                 # stop once the best camera adds < this many
    # Candidate positions are drawn from the NEAR-SURFACE band (reachable cells with
    # clearance in [collision_clearance, view_dist_max] — farther cells see nothing
    # within view distance) at ~`candidate_spacing` apart, so the candidate COUNT
    # scales with the near-surface area, not a fixed budget. `max_candidates` is only
    # a SAFETY ceiling (even-downsample + warn if a scene ever exceeds it), not the
    # room-scale cap it used to be.
    candidate_spacing: float = 0.5    # target spacing (m) between candidate positions
    max_candidates: int = 200_000     # safety ceiling on candidate positions
    # Cube-face reference-render intrinsics (SHARED with Stage 5). FOV fixed at 90°
    # (six faces tile 360°); footprint_k (hence view_dist) is DERIVED from the render
    # resolution + min_px_per_patch so coverage distances match what Stage 5 renders.
    face_fov_deg: float = 90.0
    render_resolution: int = 1024
    min_px_per_patch: float = 20.0
    view_dist_min: float = 0.5        # (m)
    view_dist_max: float = 4.0        # (m)
    seed: int = 0                     # reserved (patch selection is deterministic now)
    gpu: bool = True                  # run the coverage ray-march on CUDA when available (else CPU)

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
            "patch_fraction": self.patch_fraction,
            "collision_clearance": self.collision_clearance,
            "angles_per_patch": self.angles_per_patch,
            "angular_sectors": self.angular_sectors,
            "near_frac": self.near_frac,
            "headon_cos": self.headon_cos,
            "candidate_spacing": self.candidate_spacing,
            "max_candidates": self.max_candidates,
            "gpu": self.gpu,
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


def _local_spacing(points: np.ndarray) -> np.ndarray:
    """Per-point local sample spacing (mean distance to the 3 nearest
    neighbours) — the feature-scale field. Stage 3 samples finer where the mesh
    has detail, so the cloud's own density carries the detail signal; reading it
    here replaces the old cloud-statistics detector (curvature/texture variance),
    which could never see features finer than the sampling it measured."""
    n = len(points)
    if n < 2:
        return np.full(n, 0.1, dtype=np.float32)
    k = min(4, n)
    dist, _ = cKDTree(points).query(points, k=k)
    return dist[:, 1:].mean(axis=1).astype(np.float32)


def _candidates(
    fs: FreeSpace,
    p: PlanParams,
    fine_pos: np.ndarray | None = None,
    fine_shell: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Candidate camera positions in two tiers, returned as `(centers (M,3),
    refined_count)`:

      * UNIFORM tier — reachable near-surface cells (clearance in
        [collision_clearance, view_dist_max]) thinned to ~`candidate_spacing`,
        so the count scales with near-surface area (as before).
      * REFINED tier — the SUPPLY side of feature-adaptive density: every
        navigation cell (full grid resolution, no thinning) within each fine
        patch's near-view shell (`fine_shell` metres around `fine_pos`). Fine
        patches demand views closer than the uniform lattice reliably provides
        (its nearest standpoint is ~one spacing away); without this tier they
        read permanently unsatisfied no matter what the greedy picks. Engages
        only where demand outruns the lattice — uniform clouds get no refined
        tier and behave exactly as before.

    Both tiers are cell centres, so the merge dedupes on integer cell coords.
    `max_candidates` stays a safety ceiling on the merged set (even-downsample
    + warn)."""
    centers, _ = fs.free_candidates(
        p.collision_clearance, p.view_dist_max, p.candidate_spacing
    )
    tiers = [centers] if len(centers) else []
    n_refined = 0
    if fine_pos is not None and len(fine_pos):
        # Full-resolution near-surface band (bounded: the coarse grid is capped
        # by Stage 2's max_coarse_cells, and this keeps only the band subset).
        all_cells, _ = fs.free_candidates(p.collision_clearance, p.view_dist_max)
        if len(all_cells):
            d, i = cKDTree(fine_pos).query(all_cells, k=1)
            shell = np.asarray(fine_shell, dtype=np.float64)[i]
            refined = all_cells[d <= shell]
            n_refined = int(len(refined))
            if n_refined:
                tiers.append(refined)
    if not tiers:
        return np.zeros((0, 3), dtype=np.float32), 0
    cand = np.concatenate(tiers, axis=0)
    # Dedupe on the coarse cell lattice (both tiers emit exact cell centres).
    cell = np.round((cand - fs.origin.astype(np.float32)) / fs.pitch - 0.5).astype(np.int64)
    _, uniq = np.unique(cell, axis=0, return_index=True)
    cand = cand[np.sort(uniq)]
    if p.max_candidates and len(cand) > p.max_candidates:
        step = int(np.ceil(len(cand) / p.max_candidates))
        logging.warning(
            "stage4: %d candidates (incl. %d refined) exceeds the max_candidates "
            "ceiling %d; even-downsampling by %dx (raise max_candidates or "
            "candidate_spacing to keep full density)",
            len(cand), n_refined, p.max_candidates, step,
        )
        cand = cand[::step]
    return cand, n_refined


def _visible(
    cam: np.ndarray, patch_pos: np.ndarray, fs: FreeSpace, n_steps: int
) -> np.ndarray:
    """Line-of-sight from one camera to many patches: True where no OCCLUDING fine
    voxel lies strictly between them. Glass-classed cells (window panes, cutout
    gaps — Stage 2's `occ_lin_glass`) pass light, so surfaces behind glazing are
    coverable. Samples n_steps points along each segment and skips the endpoints
    (camera cell, patch's own surface cell)."""
    m = len(patch_pos)
    if m == 0:
        return np.zeros(0, dtype=bool)
    d = patch_pos - cam
    dist = np.linalg.norm(d, axis=1)
    dist = np.where(dist < 1e-6, 1e-6, dist)
    t = np.linspace(0.0, 1.0, n_steps)  # (K,)
    pts = cam[None, None, :] + t[None, :, None] * d[:, None, :]  # (m,K,3)
    hit = fs.fine_occluding(pts.reshape(-1, 3)).reshape(m, n_steps)  # (m,K)
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
    near_req: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    fs: FreeSpace,
    p: PlanParams,
    progress: ProgressCb | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Covering (candidate, patch) pairs, each tagged with the azimuthal SECTOR
    the camera views the patch from, whether it's a NEAR (detail-distance,
    `near_req`) view, and whether it's HEAD-ON (|incidence cosine| ≥ headon_cos —
    the anti-grazing signal: K azimuth sectors alone can all be edge-on, which
    supervises silhouettes but never the face of the surface). A pair exists if
    the camera is within the patch's view distance from EITHER side (two-sided;
    winding-agnostic) and unoccluded. Returns COO arrays (cand_idx, patch_idx,
    sector, is_near, is_head). Streams `progress(candidates_done, total,
    "coverage")` as it goes — this per-candidate ray-march is the stage's long
    pole."""
    tree = cKDTree(patch_pos)
    n_steps = int(np.ceil(p.view_dist_max / fs.pitch_fine)) + 2
    a = p.angular_sectors
    n_cand = len(candidates)
    report_every = max(1, n_cand // 200)   # ~200 progress ticks across the candidate loop
    cc: list[np.ndarray] = []
    pp: list[np.ndarray] = []
    sec: list[np.ndarray] = []
    near: list[np.ndarray] = []
    head: list[np.ndarray] = []
    for ci, cam in enumerate(candidates):
        if progress is not None and ci % report_every == 0:
            progress(ci, n_cand, "coverage")
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
        cand, cdist, ccos = idx[sel], dist[sel], cos[sel]
        if len(cand) == 0:
            continue
        vis = _visible(cam, patch_pos[cand], fs, n_steps)
        hit, hdist, hcos = cand[vis], cdist[vis], ccos[vis]
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
        near.append(hdist <= near_req[hit])
        head.append(hcos >= p.headon_cos)
    if not cc:
        e = np.zeros(0, dtype=np.int64)
        b = np.zeros(0, dtype=bool)
        return e, e, e, b, b.copy()
    return (
        np.concatenate(cc),
        np.concatenate(pp),
        np.concatenate(sec),
        np.concatenate(near),
        np.concatenate(head),
    )


def _try_cuda():  # noqa: ANN202 - returns the torch module or None
    """Return the `torch` module iff it imports AND a CUDA device is present, else
    None. Imported lazily so Stage 4 stays importable + CPU-runnable without torch;
    the GPU coverage path is opt-in via `PlanParams.gpu`."""
    try:
        import torch
    except Exception:
        return None
    return torch if torch.cuda.is_available() else None


def _build_coverage_gpu(
    candidates: np.ndarray,
    patch_pos: np.ndarray,
    patch_nrm: np.ndarray,
    view_dist: np.ndarray,
    near_req: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    fs: FreeSpace,
    p: PlanParams,
    torch: Any,
    progress: ProgressCb | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GPU port of `_build_coverage` (identical COO outputs incl. the near /
    head-on channels). The CPU KD-tree still finds each candidate's in-range
    patches — cheap, and NOT the bottleneck — but the per-pair front-facing /
    distance filter and the occlusion RAY-MARCH (the hot loop: m × n_steps
    sparse-grid membership tests) run on CUDA in candidate batches. Occlusion
    tests the OPAQUE occupancy only (`occ_lin_opaque` — glass passes light,
    mirroring `FreeSpace.fine_occluding`), searched with `torch.searchsorted`.
    Not bit-identical to the CPU path (float ordering / voxel-boundary
    rounding), by design."""
    dev = torch.device("cuda")
    n_cand = len(candidates)
    n_steps = int(np.ceil(p.view_dist_max / fs.pitch_fine)) + 2
    a = p.angular_sectors
    pf = float(fs.pitch_fine)
    d1, d2 = int(fs.fine_dims[1]), int(fs.fine_dims[2])

    occ_lin = torch.as_tensor(fs.occ_lin_opaque, dtype=torch.int64, device=dev)
    origin = torch.as_tensor(fs.origin, dtype=torch.float32, device=dev)
    dims = torch.as_tensor(fs.fine_dims, dtype=torch.int64, device=dev)
    ppos = torch.as_tensor(patch_pos, dtype=torch.float32, device=dev)
    pnrm = torch.as_tensor(patch_nrm, dtype=torch.float32, device=dev)
    vdist = torch.as_tensor(view_dist, dtype=torch.float32, device=dev)
    nreq = torch.as_tensor(near_req, dtype=torch.float32, device=dev)
    t1g = torch.as_tensor(t1, dtype=torch.float32, device=dev)
    t2g = torch.as_tensor(t2, dtype=torch.float32, device=dev)
    t_lin = torch.linspace(0.0, 1.0, n_steps, dtype=torch.float32, device=dev)

    tree = cKDTree(patch_pos)
    cand_batch = 4096
    pair_cap = 150_000                 # (candidate, patch) pairs processed at once (VRAM bound)
    cc_o: list[np.ndarray] = []
    pp_o: list[np.ndarray] = []
    sec_o: list[np.ndarray] = []
    near_o: list[np.ndarray] = []
    head_o: list[np.ndarray] = []

    def _occluded(cam, pp_, dist_):  # noqa: ANN001 - (P,3),(P,3),(P,) → (P,) bool
        """True where an OPAQUE fine voxel lies strictly between camera and patch
        (mirrors `_visible`'s negation, endpoints skipped via `tvalid`)."""
        pts = cam[:, None, :] + t_lin[None, :, None] * (pp_ - cam)[:, None, :]   # (P,K,3)
        fidx = torch.floor((pts - origin) / pf).to(torch.int64)                  # (P,K,3)
        inb = ((fidx >= 0) & (fidx < dims)).all(dim=2)                           # (P,K)
        lin = (fidx[..., 0] * d1 + fidx[..., 1]) * d2 + fidx[..., 2]             # (P,K)
        posi = torch.searchsorted(occ_lin, lin.clamp(min=0)).clamp(max=occ_lin.numel() - 1)
        occ = (occ_lin[posi] == lin) & inb                                      # (P,K)
        tvalid = (t_lin[None, :] > (pf / dist_)[:, None]) & (
            t_lin[None, :] < 1.0 - (1.5 * pf / dist_)[:, None]
        )
        return (occ & tvalid).any(dim=1)

    for b0 in range(0, n_cand, cand_batch):
        cams_np = np.ascontiguousarray(candidates[b0 : b0 + cand_batch], dtype=np.float32)
        neigh = tree.query_ball_point(cams_np, p.view_dist_max, workers=-1)
        lengths = np.fromiter((len(x) for x in neigh), dtype=np.int64, count=len(neigh))
        n_pairs = int(lengths.sum())
        if n_pairs:
            local = np.repeat(np.arange(len(neigh), dtype=np.int64), lengths)
            pidx = np.concatenate([np.asarray(x, dtype=np.int64) for x in neigh if len(x)])
            cams_b = torch.as_tensor(cams_np, device=dev)
            # Walk the (candidate, in-range-patch) pairs in ≤pair_cap slices so the
            # per-slice filter AND ray-march stay within VRAM no matter how dense the
            # scene is around a candidate — a volumetric interior can put thousands of
            # patches within view distance of one camera, so the whole batch's pair
            # list must never be materialised at once.
            for s0 in range(0, n_pairs, pair_cap):
                s1 = s0 + pair_cap
                loc = torch.as_tensor(local[s0:s1], device=dev)
                pj = torch.as_tensor(pidx[s0:s1], device=dev)
                cam = cams_b[loc]
                dvec = ppos[pj] - cam
                dist = torch.linalg.norm(dvec, dim=1).clamp_min(1e-6)
                cosang = (pnrm[pj] * dvec).sum(1).abs() / dist
                sel = (cosang > 0.1) & (dist <= vdist[pj])
                loc, pj, cam, dist, ccos = loc[sel], pj[sel], cam[sel], dist[sel], cosang[sel]
                if pj.shape[0] == 0:
                    continue
                vis = ~_occluded(cam, ppos[pj], dist)
                loc, pj, cam, dist, ccos = loc[vis], pj[vis], cam[vis], dist[vis], ccos[vis]
                if pj.shape[0] == 0:
                    continue
                vd = (cam - ppos[pj]) / dist[:, None]
                az = torch.atan2((vd * t2g[pj]).sum(1), (vd * t1g[pj]).sum(1))
                sector = torch.clamp(((az + np.pi) / (2 * np.pi) * a).to(torch.int64), 0, a - 1)
                near = dist <= nreq[pj]
                head = ccos >= p.headon_cos
                cc_o.append((loc + b0).cpu().numpy())
                pp_o.append(pj.cpu().numpy())
                sec_o.append(sector.cpu().numpy())
                near_o.append(near.cpu().numpy())
                head_o.append(head.cpu().numpy())
        if progress is not None:
            progress(min(b0 + cand_batch, n_cand), n_cand, "coverage")

    if not cc_o:
        e = np.zeros(0, dtype=np.int64)
        b = np.zeros(0, dtype=bool)
        return e, e, e, b, b.copy()
    return (
        np.concatenate(cc_o),
        np.concatenate(pp_o),
        np.concatenate(sec_o),
        np.concatenate(near_o),
        np.concatenate(head_o),
    )


def _greedy_angular(
    cc: np.ndarray,
    pp: np.ndarray,
    sector: np.ndarray,
    is_near: np.ndarray,
    is_head: np.ndarray,
    n_cand: int,
    n_patch: int,
    k: int,
    min_gain: int,
    a: int,
    progress: ProgressCb | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Greedy coverage where a patch is satisfied by ≥ k DISTINCT azimuthal
    sectors AND ≥ 1 near view AND ≥ 1 head-on view. Each step takes the camera
    advancing the most patches; stops below `min_gain`. Returns (chosen
    candidate indices, sector bitmask, has-near, has-headon). Streams
    `progress(cameras_chosen, 0, "select")` as cameras accumulate."""
    angmask = np.zeros(n_patch, dtype=np.int64)
    hasnear = np.zeros(n_patch, dtype=bool)
    hashead = np.zeros(n_patch, dtype=bool)
    lut = np.array([bin(i).count("1") for i in range(1 << a)], dtype=np.int16)
    pair_bit = np.int64(1) << sector
    taken = np.zeros(n_cand, dtype=bool)
    chosen: list[int] = []
    while True:
        pop = lut[angmask[pp]]
        new_sector = (angmask[pp] & pair_bit) == 0
        ang_contrib = new_sector & (pop < k)
        near_contrib = is_near & (~hasnear[pp])
        head_contrib = is_head & (~hashead[pp])
        contrib = ang_contrib | near_contrib | head_contrib
        if not contrib.any():
            break
        gain = np.bincount(cc, weights=contrib.astype(np.float64), minlength=n_cand)
        gain[taken] = -1.0
        c = int(np.argmax(gain))
        if gain[c] < max(min_gain, 1):
            break
        taken[c] = True
        chosen.append(c)
        if progress is not None and len(chosen) % 25 == 0:
            progress(len(chosen), 0, "select")
        m = cc == c
        np.bitwise_or.at(angmask, pp[m], pair_bit[m])
        near_hits = pp[m][is_near[m]]
        if len(near_hits):
            hasnear[near_hits] = True
        head_hits = pp[m][is_head[m]]
        if len(head_hits):
            hashead[head_hits] = True
    return chosen, angmask, hasnear, hashead


def _greedy_angular_gpu(
    cc: np.ndarray,
    pp: np.ndarray,
    sector: np.ndarray,
    is_near: np.ndarray,
    is_head: np.ndarray,
    n_cand: int,
    n_patch: int,
    k: int,
    min_gain: int,
    a: int,
    torch: Any,
    progress: ProgressCb | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """GPU port of `_greedy_angular` — same greedy, same coverage guarantee (each
    patch reaching k distinct sectors + a near view + a head-on view), same result
    up to argmax tie-breaks. The sequential ROUND loop stays in Python (each pick
    depends on the last), but every round's per-pair contribution, per-candidate
    gain (bincount) and the winner's patch updates run on CUDA — turning the
    O(rounds × pairs) tail from a single-thread numpy grind into batched GPU
    passes. Each candidate's patches are unique, so the winner's sector-mask
    update is a collision-free scatter. Returns (chosen candidate indices, sector
    bitmask, has-near, has-headon) — the arrays as numpy for the same downstream
    stats as the CPU path."""
    dev = torch.device("cuda")
    cc_t = torch.as_tensor(cc, dtype=torch.int64, device=dev)
    pp_t = torch.as_tensor(pp, dtype=torch.int64, device=dev)
    near_t = torch.as_tensor(is_near, dtype=torch.bool, device=dev)
    head_t = torch.as_tensor(is_head, dtype=torch.bool, device=dev)
    pair_bit = torch.ones((), dtype=torch.int64, device=dev) << torch.as_tensor(
        sector, dtype=torch.int64, device=dev
    )
    lut = torch.tensor(
        [bin(i).count("1") for i in range(1 << a)], dtype=torch.int16, device=dev
    )
    angmask = torch.zeros(n_patch, dtype=torch.int64, device=dev)
    hasnear = torch.zeros(n_patch, dtype=torch.bool, device=dev)
    hashead = torch.zeros(n_patch, dtype=torch.bool, device=dev)
    taken = torch.zeros(n_cand, dtype=torch.bool, device=dev)
    min_g = float(max(min_gain, 1))
    chosen: list[int] = []
    while True:
        am = angmask[pp_t]
        contrib = (
            (((am & pair_bit) == 0) & (lut[am] < k))
            | (near_t & ~hasnear[pp_t])
            | (head_t & ~hashead[pp_t])
        )
        if not bool(contrib.any()):
            break
        gain = torch.bincount(cc_t, weights=contrib.to(torch.float32), minlength=n_cand)
        gain[taken] = -1.0
        c = int(gain.argmax())
        if float(gain[c]) < min_g:
            break
        taken[c] = True
        chosen.append(c)
        if progress is not None and len(chosen) % 200 == 0:
            progress(len(chosen), 0, "select")
        m = cc_t == c
        patches_c = pp_t[m]
        angmask[patches_c] = angmask[patches_c] | pair_bit[m]   # unique patches → collision-free
        near_c = patches_c[near_t[m]]
        if near_c.numel():
            hasnear[near_c] = True
        head_c = patches_c[head_t[m]]
        if head_c.numel():
            hashead[head_c] = True
    return chosen, angmask.cpu().numpy(), hasnear.cpu().numpy(), hashead.cpu().numpy()


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

    if progress is not None:
        progress(0, 0, "load")
    fs = load_free_space(Path(freespace_path))
    points, normals, _albedo = _read_cloud(Path(surfels_path))
    lo = points.min(axis=0)
    hi = points.max(axis=0)

    # Patches: UNIFORM thinning of the cloud, deterministic (every stride-th
    # surfel). Stage 3's density is already feature-adaptive, so the patches
    # inherit its profile — an adaptive re-thin here would weight detail twice.
    # Each patch's feature scale is its LOCAL SAMPLE SPACING (measured on the
    # full cloud, before thinning), which drives the footprint view distance:
    # fine-band patches demand close cameras, flat-band patches allow far ones.
    if progress is not None:
        progress(0, 0, "patches")
    spacing = _local_spacing(points)
    stride = max(1, int(round(1.0 / min(max(params.patch_fraction, 1e-3), 1.0))))
    keep = np.arange(0, len(points), stride)
    patch_pos = points[keep].astype(np.float32)
    patch_nrm = normals[keep].astype(np.float32)
    patch_feat = spacing[keep]
    t1, t2 = _tangent_frames(patch_nrm)
    view_dist = np.clip(
        params.footprint_k * patch_feat, params.view_dist_min, params.view_dist_max
    ).astype(np.float32)
    # A "near" view can never be REQUIRED closer than cameras can physically
    # get: collision clearance plus one navigation cell of discretization.
    # Without this floor, fine-band patches (feature scale a few cm →
    # near_frac·view_dist below the clearance) would demand views no candidate
    # can provide and read as permanently unsatisfied.
    near_floor = params.collision_clearance + fs.pitch
    near_req = np.maximum(params.near_frac * view_dist, near_floor).astype(np.float32)

    # Candidate camera positions from the reachable near-surface band (no
    # re-voxelization). FINE patches — those whose near-view requirement is
    # tighter than the uniform lattice reliably supplies (nearest standpoint ~one
    # candidate_spacing away) — get a REFINED full-resolution tier inside their
    # near shells, so the standpoint supply follows Stage 3's density field.
    fine = near_req < 1.5 * params.candidate_spacing
    candidates, n_refined = _candidates(
        fs, params,
        fine_pos=patch_pos[fine],
        fine_shell=near_req[fine] + fs.pitch,
    )
    if len(candidates) == 0:
        raise RuntimeError("no candidate camera positions (reachable free space empty)")

    # Coverage (per-pair sector + near + head-on) + angular greedy. Coverage is
    # the long phase (a per-candidate occlusion ray-march), so it streams
    # fine-grained progress (candidates processed / total) instead of a single
    # coarse tick; both phases run on CUDA when available (identical algorithm,
    # see the _gpu variants) and fall back to the CPU path otherwise.
    n_patch = len(patch_pos)
    k = params.angles_per_patch
    a = params.angular_sectors
    torch = _try_cuda() if params.gpu else None
    if params.gpu and torch is None:
        logging.warning(
            "stage4: gpu requested but CUDA/torch unavailable — using the CPU coverage path"
        )
    if torch is not None:
        cc, pp, sector, is_near, is_head = _build_coverage_gpu(
            candidates, patch_pos, patch_nrm, view_dist, near_req, t1, t2, fs,
            params, torch, progress,
        )
        chosen, angmask, hasnear, hashead = _greedy_angular_gpu(
            cc, pp, sector, is_near, is_head, len(candidates), n_patch, k,
            params.min_gain, a, torch, progress,
        )
    else:
        cc, pp, sector, is_near, is_head = _build_coverage(
            candidates, patch_pos, patch_nrm, view_dist, near_req, t1, t2, fs,
            params, progress,
        )
        chosen, angmask, hasnear, hashead = _greedy_angular(
            cc, pp, sector, is_near, is_head, len(candidates), n_patch, k,
            params.min_gain, a, progress,
        )
    if progress is not None:
        progress(0, 0, "write")

    # Per-patch stats: distinct angles seen, and satisfaction (≥K angles + near
    # view + head-on view).
    lut = np.array([bin(i).count("1") for i in range(1 << a)], dtype=np.int16)
    sectors_seen = lut[angmask].astype(np.int32)
    satisfied_mask = (sectors_seen >= k) & hasnear & hashead
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
        "candidates_refined": int(n_refined),
        "cameras": len(cameras),
        "angles_per_patch": k,
        "angular_sectors": a,
        "coverage": {
            "satisfied": satisfied,
            "satisfied_pct": round(100.0 * satisfied / max(n_patch, 1), 1),
            "seen_at_least_once": seen_once,
            "occlusion_culled": occluded,
            "has_near": int(hasnear.sum()),
            "has_headon": int(hashead.sum()),
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
    grid_diag = float(np.linalg.norm(np.array(fs.fine_shape) * fs.pitch_fine))
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
