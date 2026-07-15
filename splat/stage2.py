"""Stage 2 — Free-space voxelizer + clearance field (adaptive).

The shared spatial FOUNDATION of the splat pipeline: discretizes a cell's composed
scene into occupancy + a clearance field over the scene AABB (grown by an exterior
margin so cameras can see outer silhouettes), and writes a REUSABLE grid that both
Stage 3 (surfel sampler — orient normals to free space, cull hidden faces) and
Stage 4 (camera planner — candidates + occlusion) consume. Nothing downstream
recomputes occupancy; that's the point of running this first (overview §, Option A).

DUAL RESOLUTION (the "adaptive" part). Two aligned grids sharing one origin:
  * a FINE occupancy grid (`pitch_fine = pitch / refine`) — accurate near surfaces,
    so thin occluders/gaps aren't missed by the visibility ray-march or the surfel
    hidden-face test;
  * a COARSE clearance field (`pitch`, a navigational scale) — the distance from
    each empty cell to the nearest surface, driving camera candidate placement.
The coarse occupancy is a block-max of the fine grid, so the two never disagree.
A cell-count guard coarsens `pitch_fine` for very large scenes to bound memory.

Occupancy is a SURFACE voxelization via dense area-weighted point sampling (robust
for the non-watertight composed scene). Interior cells of thick solids read as
empty here; that's fine — Stage 3/4 use the fine occupancy for line-of-sight, which
naturally excludes buried points.

Pure library (like the other stages): takes explicit paths; the server resolves the
cell and de-optimizes a library build to vanilla first (trimesh can't read
KTX2/Meshopt).

Outputs (under a cell's `splat/` dir):
  * `freespace.npz` — the reusable grid (origin, pitch, pitch_fine, refine, fine
    occupancy, coarse clearance). Consumed by Stages 3 and 4; load via
    `load_free_space`.
  * `voxels.bin` — packed float32 `[x,y,z,clearance]` per free coarse voxel (world
    centre + metres), strided under a cap, for the client overlay.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy import ndimage

# trimesh's sampler logs "only got N/M samples!"; silence for whole-cell runs.
logging.getLogger("trimesh").setLevel(logging.ERROR)

# Artifacts written under a cell's `splat/` dir.
FREESPACE_NAME = "freespace.npz"
VOXELS_NAME = "voxels.bin"

DEFAULT_PITCH = 0.12       # coarse voxel edge (m) — navigational, for clearance
DEFAULT_REFINE = 3         # fine = pitch / refine (0.04 m by default) — for occlusion
_PITCH_CLAMP = (0.02, 1.0)
# Surface points sampled per FINE-voxel-area; a few so no thin surface is skipped.
_OVERSAMPLE = 4.0
# Cap the exported free-voxel viz cloud (stride if a scene has more).
_MAX_VIZ_VOXELS = 500_000
# Guard on the dense fine grid — coarsen pitch_fine if a scene would exceed this.
_MAX_FINE_CELLS = 120_000_000

# progress(done, total, current_id) — called after each object is voxelized.
ProgressCb = Callable[[int, int, str], None]


@dataclass(frozen=True)
class FreeSpaceParams:
    """Stage-2 knobs. `pitch` is the coarse (clearance/navigation) scale; the fine
    occupancy scale is `pitch / refine`. `margin` grows the grid beyond the scene
    AABB so exterior camera vantages exist (Stage 4). `reachable_min_volume` (m³) is
    the smallest free pocket kept as navigable — larger drops tiny object-interior
    hollows, smaller keeps closets/nooks (see reachability below)."""

    pitch: float = DEFAULT_PITCH
    refine: int = DEFAULT_REFINE
    margin: float = 1.5
    reachable_min_volume: float = 0.25

    def as_summary(self) -> dict[str, Any]:
        return {
            "pitch": self.pitch,
            "refine": self.refine,
            "margin": self.margin,
            "reachable_min_volume": self.reachable_min_volume,
        }


@dataclass(frozen=True)
class FreeSpace:
    """A loaded free-space grid. Aligned grids share `origin`: the fine occupancy
    (`occ_fine`, edge `pitch_fine`), the coarse clearance (`clearance`, edge
    `pitch`, metres), and the coarse `reachable` mask (free cells in a large enough
    connected component — exterior + rooms, minus tiny object-interior hollows).
    `refine = pitch / pitch_fine`."""

    origin: np.ndarray        # (3,) world corner of fine cell [0,0,0]
    pitch: float              # coarse edge (m)
    pitch_fine: float         # fine edge (m)
    refine: int
    occ_fine: np.ndarray      # bool [fx,fy,fz]
    clearance: np.ndarray     # float32 [cx,cy,cz] (m)
    reachable: np.ndarray     # bool [cx,cy,cz] — navigable free space

    @property
    def coarse_dims(self) -> tuple[int, int, int]:
        return self.clearance.shape  # type: ignore[return-value]

    def fine_occupied(self, points: np.ndarray) -> np.ndarray:
        """Boolean per world point: does its FINE voxel contain surface? Points
        outside the grid read as False (empty)."""
        idx = np.floor((points - self.origin) / self.pitch_fine).astype(np.int64)
        dims = np.asarray(self.occ_fine.shape)
        inb = np.all((idx >= 0) & (idx < dims), axis=1)
        out = np.zeros(len(points), dtype=bool)
        ii = idx[inb]
        out[inb] = self.occ_fine[ii[:, 0], ii[:, 1], ii[:, 2]]
        return out

    def reachable_free(self, points: np.ndarray) -> np.ndarray:
        """Boolean per world point: is it in NAVIGABLE free space (a large enough
        free component)? This is what distinguishes exterior/room air from a solid's
        hollow interior — the signal Stage 3 uses to orient normals + cull hidden
        faces, and Stage 4 to place cameras. Outside-grid points read as False."""
        idx = np.floor((points - self.origin) / self.pitch).astype(np.int64)
        dims = np.asarray(self.reachable.shape)
        inb = np.all((idx >= 0) & (idx < dims), axis=1)
        out = np.zeros(len(points), dtype=bool)
        ii = idx[inb]
        out[inb] = self.reachable[ii[:, 0], ii[:, 1], ii[:, 2]]
        return out

    def free_candidates(self, min_clearance: float) -> tuple[np.ndarray, np.ndarray]:
        """NAVIGABLE free-cell world centres with clearance ≥ `min_clearance`, and
        their clearances — the camera-candidate pool for Stage 4. Restricted to
        reachable space, so cameras never spawn inside solids/tiny hollows."""
        free = self.reachable & (self.clearance >= min_clearance)
        cells = np.argwhere(free)
        centers = (self.origin + (cells + 0.5) * self.pitch).astype(np.float32)
        return centers, self.clearance[free]


def load_free_space(path: Path) -> FreeSpace:
    """Load a `freespace.npz` written by `compute_free_space`."""
    with np.load(path) as z:
        return FreeSpace(
            origin=z["origin"].astype(np.float64),
            pitch=float(z["pitch"]),
            pitch_fine=float(z["pitch_fine"]),
            refine=int(z["refine"]),
            occ_fine=z["occ_fine"].astype(bool),
            clearance=z["clearance"].astype(np.float32),
            reachable=z["reachable"].astype(bool),
        )


def _iter_geoms(mesh: trimesh.Trimesh | trimesh.Scene) -> list[trimesh.Trimesh]:
    if isinstance(mesh, trimesh.Scene):
        return [g for g in mesh.geometry.values() if hasattr(g, "faces")]
    return [mesh]


def placed_object_ids(raw_dir: Path) -> list[str]:
    """Ids with a placed, world-space mesh in `raw_dir` (the served `<id>.glb`,
    excluding `<id>.raw.glb` pre-placement intermediates)."""
    return sorted(
        p.name[: -len(".glb")]
        for p in raw_dir.glob("*.glb")
        if not p.name.endswith(".raw.glb")
    )


def _block_max(occ: np.ndarray, r: int) -> np.ndarray:
    """Down-sample a fine boolean grid to coarse by OR-ing each r×r×r block (a cell
    is solid if any fine sub-cell is). `occ` dims must be multiples of r."""
    fx, fy, fz = occ.shape
    return occ.reshape(fx // r, r, fy // r, r, fz // r, r).any(axis=(1, 3, 5))


def compute_free_space(
    *,
    run: str,
    slot: str,
    model: str,
    raw_dir: Path,
    out_path: Path,
    params: FreeSpaceParams = FreeSpaceParams(),
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Voxelize the cell's placed meshes into a dual-resolution occupancy + clearance
    grid, writing `freespace.npz` (to `out_path`) + `voxels.bin` (beside it). Returns
    a summary (grid dims, counts, clearance stats)."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"placed-mesh dir not found: {raw_dir}")
    ids = placed_object_ids(raw_dir)
    if not ids:
        raise FileNotFoundError(f"no placed meshes in {raw_dir}")
    pitch = float(np.clip(params.pitch, *_PITCH_CLAMP))
    refine = max(1, int(params.refine))
    pitch_fine = pitch / refine
    margin = float(max(0.0, params.margin))

    total = len(ids)
    if progress is not None:
        progress(0, total, "")

    # One pass: accumulate world surface points + the union AABB.
    pts_parts: list[np.ndarray] = []
    lo = np.array([np.inf, np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf, -np.inf])
    for done, node_id in enumerate(ids, start=1):
        try:
            m = trimesh.load(raw_dir / f"{node_id}.glb", process=False)
            for g in _iter_geoms(m):
                if len(g.faces) == 0 or g.area <= 0:
                    continue
                b = np.asarray(g.bounds, dtype=float)
                lo, hi = np.minimum(lo, b[0]), np.maximum(hi, b[1])
                count = int(g.area / (pitch_fine * pitch_fine) * _OVERSAMPLE) + 8
                p, _ = trimesh.sample.sample_surface(g, count)
                pts_parts.append(np.asarray(p, dtype=np.float32))
            del m
        except Exception:  # skip a bad mesh, keep going
            pass
        if progress is not None:
            progress(done, total, node_id)

    if not pts_parts or not np.isfinite(lo).all():
        raise RuntimeError("no surface sampled (every mesh failed or was empty)")
    pts = np.concatenate(pts_parts, axis=0)

    # Fine grid over the AABB + margin. Origin one fine-voxel below the padded box so
    # boundary cells are empty; fine dims padded to a multiple of `refine` so the
    # coarse grid is an exact block-reduction.
    origin = lo - margin - pitch_fine
    extent = (hi - lo) + 2.0 * margin + 2.0 * pitch_fine
    fdims = np.ceil(extent / pitch_fine).astype(int) + 1
    fdims = ((np.maximum(fdims, refine) + refine - 1) // refine) * refine  # multiple of refine

    # Guard memory: coarsen pitch_fine (drop refine) if the fine grid is too large.
    while int(np.prod(fdims)) > _MAX_FINE_CELLS and refine > 1:
        refine -= 1
        pitch_fine = pitch / refine
        origin = lo - margin - pitch_fine
        extent = (hi - lo) + 2.0 * margin + 2.0 * pitch_fine
        fdims = np.ceil(extent / pitch_fine).astype(int) + 1
        fdims = ((np.maximum(fdims, refine) + refine - 1) // refine) * refine
    fnx, fny, fnz = int(fdims[0]), int(fdims[1]), int(fdims[2])

    # Fine occupancy: bin surface points → solid fine cells.
    vidx = np.floor((pts - origin) / pitch_fine).astype(np.int64)
    inb = np.all((vidx >= 0) & (vidx < fdims), axis=1)
    vidx = vidx[inb]
    occ_fine = np.zeros((fnx, fny, fnz), dtype=bool)
    occ_fine[vidx[:, 0], vidx[:, 1], vidx[:, 2]] = True

    # Coarse occupancy (block-max) → clearance (EDT over empties), in metres.
    occ_coarse = _block_max(occ_fine, refine)
    clearance = ndimage.distance_transform_edt(~occ_coarse).astype(np.float32) * pitch

    # Reachability: label connected free components (6-connectivity) and keep only
    # those large enough to be navigable (exterior + rooms), dropping the tiny
    # hollows a surface voxelization leaves inside solid objects. This is the signal
    # that separates real free space from a solid's interior.
    free_coarse = ~occ_coarse
    labels, n_comp = ndimage.label(free_coarse, ndimage.generate_binary_structure(3, 1))
    reachable = np.zeros_like(free_coarse)
    if n_comp > 0:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0  # label 0 is the solid background
        min_cells = max(8, int(params.reachable_min_volume / (pitch**3)))
        keep = np.nonzero(sizes >= min_cells)[0]
        reachable = np.isin(labels, keep)

    # Persist the reusable grid.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = out_path.with_suffix(out_path.suffix + ".tmp.npz")
    np.savez_compressed(
        tmp_npz,
        origin=origin.astype(np.float64),
        pitch=np.float64(pitch),
        pitch_fine=np.float64(pitch_fine),
        refine=np.int64(refine),
        occ_fine=occ_fine,
        clearance=clearance,
        reachable=reachable,
    )
    tmp_npz.replace(out_path)

    # Viz cloud: NAVIGABLE (reachable) coarse cells + clearance, strided under a cap.
    reach_idx = np.argwhere(reachable)
    reach_count = int(reach_idx.shape[0])
    stride = max(1, int(np.ceil(reach_count / _MAX_VIZ_VOXELS)))
    sel = reach_idx[::stride]
    centers = origin + (sel.astype(np.float32) + 0.5) * pitch
    clv = clearance[sel[:, 0], sel[:, 1], sel[:, 2]] if len(sel) else np.zeros(0, dtype=np.float32)
    viz = (
        np.concatenate([centers, clv[:, None]], axis=1).astype("<f4")
        if len(sel)
        else np.zeros((0, 4), dtype="<f4")
    )
    viz_path = out_path.with_name(VOXELS_NAME)
    tmp_viz = viz_path.with_suffix(viz_path.suffix + ".tmp")
    tmp_viz.write_bytes(viz.tobytes())
    tmp_viz.replace(viz_path)

    cl_free = clearance[free_coarse]
    return {
        "run": run,
        "slot": slot,
        "model": model,
        "pitch": pitch,
        "pitch_fine": round(pitch_fine, 5),
        "refine": refine,
        "dims_fine": [fnx, fny, fnz],
        "dims_coarse": list(occ_coarse.shape),
        "origin": [round(v, 5) for v in origin.tolist()],
        "scene_aabb": {"min": lo.tolist(), "max": hi.tolist()},
        "solid_voxels_fine": int(occ_fine.sum()),
        "free_voxels": int(free_coarse.sum()),
        "reachable_voxels": reach_count,
        "exported_voxels": int(sel.shape[0]),
        "stride": stride,
        "clearance_max": float(cl_free.max()) if cl_free.size else 0.0,
        "clearance_mean": round(float(cl_free.mean()), 4) if cl_free.size else 0.0,
        "params": params.as_summary(),
        "bytes": viz_path.stat().st_size,
        "grid_bytes": out_path.stat().st_size,
    }
