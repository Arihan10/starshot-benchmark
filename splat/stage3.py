"""Stage 3 — Free-space voxelizer + clearance field.

Discretizes a cell's composed scene into a uniform voxel grid over the scene
AABB, marks SOLID cells (a surface passes through) vs EMPTY, and computes a
CLEARANCE FIELD — the distance from each empty cell to the nearest surface. Per
the overview (§5 Stage 3), this is the foundation for Stage 4 (camera placement,
denser where clearance is small) and occlusion culling.

Pure library (like stage1/stage2): takes explicit paths; the server resolves the
cell and de-optimizes a library build to vanilla first (trimesh can't read
KTX2/Meshopt).

Occupancy is a SURFACE voxelization via dense area-weighted point sampling —
robust for the non-watertight composed scene (a large flat triangle can't slip
between voxels at the sampling density used). Interior cells of thick solids read
as empty here; reachability (flood fill, enclosed rooms) is deliberately a Stage-4
concern. Clearance is a Euclidean distance transform (scipy) over the empty mask,
in metres.

Output: `voxels.bin` — packed little-endian float32 `[x, y, z, clearance]` per
FREE voxel (world centre + metres) for the client to visualize — plus a summary
(grid dims, counts, clearance stats). The exported cloud is strided to a cap so
the viz payload stays small even for large scenes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy import ndimage

# trimesh's sampler logs "only got N/M samples!"; silence for whole-cell runs.
logging.getLogger("trimesh").setLevel(logging.ERROR)

# The free-voxel cloud filename written under a cell's `splat/` dir.
VOXELS_NAME = "voxels.bin"

DEFAULT_PITCH = 0.12  # voxel edge (m) — a navigational grid, not cm-scale detail
_PITCH_CLAMP = (0.02, 1.0)
# Surface points sampled per voxel-area; > a few so no surface voxel is skipped.
_OVERSAMPLE = 6.0
# Cap the exported free-voxel cloud (stride if a scene has more) so the client
# payload + point count stay manageable.
_MAX_VIZ_VOXELS = 500_000

# progress(done, total, current_id) — called after each object is voxelized.
ProgressCb = Callable[[int, int, str], None]


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


def compute_free_space(
    *,
    run: str,
    slot: str,
    model: str,
    raw_dir: Path,
    out_path: Path,
    pitch: float = DEFAULT_PITCH,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Voxelize the cell's placed meshes into a solid/empty grid + a clearance
    field, writing the free voxels to `out_path` (`voxels.bin`). Returns a summary
    (grid dims, origin, counts, clearance stats) — also the viz metadata."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"placed-mesh dir not found: {raw_dir}")
    ids = placed_object_ids(raw_dir)
    if not ids:
        raise FileNotFoundError(f"no placed meshes in {raw_dir}")
    pitch = float(np.clip(pitch, *_PITCH_CLAMP))

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
                count = int(g.area / (pitch * pitch) * _OVERSAMPLE) + 8
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

    # Uniform grid over the AABB, padded one voxel each side so boundary cells are
    # empty (well-defined clearance) and the AABB isn't clipped.
    origin = lo - pitch
    dims = np.ceil((hi - lo) / pitch).astype(int) + 3
    dims = np.maximum(dims, 1)
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])

    # Occupancy: bin surface points → solid cells.
    vidx = np.floor((pts - origin) / pitch).astype(np.int64)
    inb = np.all((vidx >= 0) & (vidx < dims), axis=1)
    vidx = vidx[inb]
    occ = np.zeros((nx, ny, nz), dtype=bool)
    occ[vidx[:, 0], vidx[:, 1], vidx[:, 2]] = True
    solid_count = int(occ.sum())

    # Clearance = Euclidean distance transform over the empty mask (distance from
    # each empty cell to the nearest solid cell), scaled to metres.
    empty = ~occ
    clearance = ndimage.distance_transform_edt(empty).astype(np.float32) * pitch
    free_idx = np.argwhere(empty)
    free_count = int(free_idx.shape[0])

    # Export free-voxel world centres + clearance (strided under the cap).
    stride = max(1, int(np.ceil(free_count / _MAX_VIZ_VOXELS)))
    sel = free_idx[::stride]
    centers = origin + (sel.astype(np.float32) + 0.5) * pitch
    clv = clearance[sel[:, 0], sel[:, 1], sel[:, 2]]
    data = np.concatenate([centers, clv[:, None]], axis=1).astype("<f4")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_bytes(data.tobytes())
    tmp.replace(out_path)

    cl_free = clearance[empty]
    return {
        "run": run,
        "slot": slot,
        "model": model,
        "pitch": pitch,
        "dims": [nx, ny, nz],
        "origin": [round(v, 5) for v in origin.tolist()],
        "scene_aabb": {"min": lo.tolist(), "max": hi.tolist()},
        "solid_voxels": solid_count,
        "free_voxels": free_count,
        "exported_voxels": int(sel.shape[0]),
        "stride": stride,
        "clearance_max": float(cl_free.max()) if cl_free.size else 0.0,
        "clearance_mean": round(float(cl_free.mean()), 4) if cl_free.size else 0.0,
        "bytes": out_path.stat().st_size,
    }
