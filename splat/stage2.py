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
    hidden-face test. Stored SPARSELY (sorted linear indices of the occupied cells
    only) so its memory tracks the amount of SURFACE, not the scene's bounding-box
    VOLUME — the fine resolution stays fixed regardless of scene size (no
    auto-coarsening), and huge open scenes no longer blow up a dense array.
  * a COARSE clearance field (`pitch`, a navigational scale) — the distance from
    each empty cell to the nearest surface, driving camera candidate placement.
The coarse occupancy is derived from the same occupied fine cells, so the two never
disagree. The coarse field is genuinely volumetric (it measures empty space) and so
stays dense; it is the next thing to chunk for extreme scene sizes.

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

# Absolute-lattice voxel encoding: occupied fine voxels are binned in a scene-
# independent integer lattice (floor(point / pitch_fine)) so binning needs no origin
# (computed only after the AABB is known). Coords are packed into one int64 with a
# large offset+radix, supporting a ±_VOX_OFF-cell span (~±42 km at 0.04 m).
_VOX_OFF = 1 << 20
_VOX_RADIX = 1 << 21


def _abs_encode(vox: np.ndarray) -> np.ndarray:
    """Pack (N,3) absolute integer voxel coords into sorted-able int64 keys."""
    v = vox.astype(np.int64)
    return ((v[:, 0] + _VOX_OFF) * _VOX_RADIX + (v[:, 1] + _VOX_OFF)) * _VOX_RADIX + (
        v[:, 2] + _VOX_OFF
    )


def _abs_decode(lin: np.ndarray) -> np.ndarray:
    """Inverse of `_abs_encode`: int64 keys → (N,3) absolute voxel coords."""
    lin = lin.astype(np.int64)
    iz = lin % _VOX_RADIX - _VOX_OFF
    iy = (lin // _VOX_RADIX) % _VOX_RADIX - _VOX_OFF
    ix = lin // (_VOX_RADIX * _VOX_RADIX) - _VOX_OFF
    return np.stack([ix, iy, iz], axis=1)

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
    """A loaded free-space grid. Aligned grids share `origin`: the SPARSE fine
    occupancy (`occ_lin` — sorted linear indices into a `fine_dims` grid of edge
    `pitch_fine`), the coarse clearance (`clearance`, edge `pitch`, metres), and the
    coarse `reachable` mask (free cells in a large enough connected component —
    exterior + rooms, minus tiny object-interior hollows). `refine = pitch /
    pitch_fine`."""

    origin: np.ndarray        # (3,) world corner of fine cell [0,0,0]
    pitch: float              # coarse edge (m)
    pitch_fine: float         # fine edge (m)
    refine: int
    fine_dims: np.ndarray     # (3,) int64 fine grid dims [fnx,fny,fnz]
    occ_lin: np.ndarray       # (K,) int64 SORTED linear indices of occupied fine cells
    clearance: np.ndarray     # float32 [cx,cy,cz] (m)
    reachable: np.ndarray     # bool [cx,cy,cz] — navigable free space

    @property
    def coarse_dims(self) -> tuple[int, int, int]:
        return self.clearance.shape  # type: ignore[return-value]

    @property
    def fine_shape(self) -> tuple[int, int, int]:
        return (int(self.fine_dims[0]), int(self.fine_dims[1]), int(self.fine_dims[2]))

    def fine_occupied(self, points: np.ndarray) -> np.ndarray:
        """Boolean per world point: does its FINE voxel contain surface? Membership
        is a binary search into the sorted sparse `occ_lin`. Points outside the grid
        read as False (empty)."""
        idx = np.floor((points - self.origin) / self.pitch_fine).astype(np.int64)
        dims = self.fine_dims
        inb = np.all((idx >= 0) & (idx < dims), axis=1)
        out = np.zeros(len(points), dtype=bool)
        if not self.occ_lin.size or not inb.any():
            return out
        ii = idx[inb]
        lin = (ii[:, 0] * dims[1] + ii[:, 1]) * dims[2] + ii[:, 2]
        pos = np.searchsorted(self.occ_lin, lin)
        pos = np.clip(pos, 0, self.occ_lin.size - 1)
        out[inb] = self.occ_lin[pos] == lin
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

    def free_candidates(
        self,
        min_clearance: float,
        max_clearance: float | None = None,
        spacing: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """NAVIGABLE free-cell world centres (+ their clearances) — the camera-
        candidate pool for Stage 4. Restricted to reachable space, so cameras never
        spawn inside solids/tiny hollows.

        `max_clearance` keeps only the NEAR-SURFACE band (cells whose nearest surface
        is within reach of a camera); cells farther than any view distance can see
        nothing and are dropped. `spacing` (m) thins the band to ~one candidate per
        `spacing`-sized block, picking the MOST-OPEN cell in each block — so the
        candidate count scales with the near-surface area, not a fixed budget."""
        free = self.reachable & (self.clearance >= min_clearance)
        if max_clearance is not None:
            free &= self.clearance <= max_clearance
        stride = 1 if spacing is None else max(1, int(round(spacing / self.pitch)))
        if stride == 1:
            cells = np.argwhere(free)
            centers = (self.origin + (cells + 0.5) * self.pitch).astype(np.float32)
            return centers, self.clearance[free]
        # One representative per stride³ block: the eligible cell CLOSEST to a surface
        # (smallest clearance, but still ≥ collision_clearance) — closest safe camera
        # spots best serve the near/detail-view coverage requirement. Pad up so no
        # edge block is lost; +inf marks ineligible cells.
        score = np.where(free, self.clearance, np.float32(np.inf))
        nx, ny, nz = score.shape
        pad = [(-nx) % stride, (-ny) % stride, (-nz) % stride]
        score = np.pad(score, [(0, pad[0]), (0, pad[1]), (0, pad[2])], constant_values=np.inf)
        pnx, pny, pnz = score.shape
        blocks = (
            score.reshape(pnx // stride, stride, pny // stride, stride, pnz // stride, stride)
            .transpose(0, 2, 4, 1, 3, 5)
            .reshape(pnx // stride, pny // stride, pnz // stride, stride**3)
        )
        best = blocks.argmin(axis=3)
        keep = np.isfinite(blocks.min(axis=3))
        bxyz = np.argwhere(keep)
        off = best[keep]
        ox, oy, oz = off // (stride * stride), (off // stride) % stride, off % stride
        cells = np.stack(
            [bxyz[:, 0] * stride + ox, bxyz[:, 1] * stride + oy, bxyz[:, 2] * stride + oz],
            axis=1,
        )
        centers = (self.origin + (cells + 0.5) * self.pitch).astype(np.float32)
        return centers, self.clearance[cells[:, 0], cells[:, 1], cells[:, 2]]


def load_free_space(path: Path) -> FreeSpace:
    """Load a `freespace.npz` written by `compute_free_space`. Accepts the sparse
    layout (`fine_dims` + `occ_lin`) and, for back-compat, a legacy dense `occ_fine`
    (flattened to sparse on load)."""
    with np.load(path) as z:
        files = set(z.files)
        if "occ_lin" in files:
            fine_dims = z["fine_dims"].astype(np.int64)
            occ_lin = z["occ_lin"].astype(np.int64)
        else:  # legacy dense occupancy → sparse (C-order linear matches fine_occupied)
            occ = z["occ_fine"].astype(bool)
            fine_dims = np.array(occ.shape, dtype=np.int64)
            occ_lin = np.flatnonzero(occ.reshape(-1)).astype(np.int64)
        return FreeSpace(
            origin=z["origin"].astype(np.float64),
            pitch=float(z["pitch"]),
            pitch_fine=float(z["pitch_fine"]),
            refine=int(z["refine"]),
            fine_dims=fine_dims,
            occ_lin=occ_lin,
            clearance=z["clearance"].astype(np.float32),
            reachable=z["reachable"].astype(bool),
        )


def _iter_geoms(mesh: trimesh.Trimesh | trimesh.Scene) -> list[trimesh.Trimesh]:
    # dump() bakes each geometry's scene-graph node transform into world space; plain
    # scene.geometry returns LOCAL vertices, which collapses node-placed objects
    # (generated assets carry placement on the node) to the origin.
    if isinstance(mesh, trimesh.Scene):
        return [g for g in mesh.dump(concatenate=False) if hasattr(g, "faces")]
    return [mesh]


def placed_object_ids(raw_dir: Path) -> list[str]:
    """Ids with a placed, world-space mesh in `raw_dir` (the served `<id>.glb`,
    excluding `<id>.raw.glb` pre-placement intermediates)."""
    return sorted(
        p.name[: -len(".glb")]
        for p in raw_dir.glob("*.glb")
        if not p.name.endswith(".raw.glb")
    )


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

    # One pass: sample each surface, binning points into an ABSOLUTE integer voxel
    # lattice (floor(point / pitch_fine)) and keeping only the UNIQUE occupied cells,
    # so we never hold the raw point cloud — memory tracks surface, not volume. The
    # grid origin (which needs the AABB) is resolved afterwards; absolute-lattice
    # keys are origin-independent.
    key_parts: list[np.ndarray] = []
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
                vox = np.floor(np.asarray(p, dtype=np.float64) / pitch_fine).astype(np.int64)
                key_parts.append(np.unique(_abs_encode(vox)))
            del m
        except Exception:  # skip a bad mesh, keep going
            pass
        if progress is not None:
            progress(done, total, node_id)

    if not key_parts or not np.isfinite(lo).all():
        raise RuntimeError("no surface sampled (every mesh failed or was empty)")
    occ_abs = _abs_decode(np.unique(np.concatenate(key_parts)))  # (K,3) absolute coords

    # Fine grid over the AABB + margin, origin snapped to the absolute lattice (a
    # multiple of pitch_fine, and of `refine` cells so the coarse grid is an exact
    # block-reduction). No cell-count guard: occupancy is sparse, so the fine
    # resolution is held fixed regardless of scene size.
    imin = np.floor((lo - margin) / pitch_fine).astype(np.int64) - 1
    imin -= imin % refine  # align down to a coarse-cell boundary
    imax = np.ceil((hi + margin) / pitch_fine).astype(np.int64) + 1
    fdims = (imax - imin) + 1
    fdims = ((np.maximum(fdims, refine) + refine - 1) // refine) * refine  # multiple of refine
    fnx, fny, fnz = int(fdims[0]), int(fdims[1]), int(fdims[2])
    origin = imin.astype(np.float64) * pitch_fine

    # Occupied fine cells → local linear indices (drop any outside the padded grid).
    local = occ_abs - imin
    inb = np.all((local >= 0) & (local < fdims), axis=1)
    local = local[inb]
    occ_lin = np.unique((local[:, 0] * fny + local[:, 1]) * fnz + local[:, 2])

    # Coarse occupancy from the same occupied cells (a coarse cell is solid if any
    # fine sub-cell is) → clearance (EDT over empties), in metres.
    occ_coarse = np.zeros((fnx // refine, fny // refine, fnz // refine), dtype=bool)
    cc = local // refine
    occ_coarse[cc[:, 0], cc[:, 1], cc[:, 2]] = True
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
        fine_dims=np.array([fnx, fny, fnz], dtype=np.int64),
        occ_lin=occ_lin.astype(np.int64),
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
        "solid_voxels_fine": int(occ_lin.size),
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
