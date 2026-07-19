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
stays dense — the memory wall for large/sparse scenes (open arenas, cities, mostly-
empty space). To bound it, the coarse block factor is chosen ADAPTIVELY per scene so
the dense arrays never exceed `max_coarse_cells` (a coarser navigation grid on huge
scenes); the FINE occupancy is sparse and keeps its fixed resolution, so occlusion /
thin-gap accuracy is unchanged. Scenes that already fit are byte-identical to before.

Occupancy is an EXACT surface voxelization: every fine cell whose cube a triangle
touches is marked — big triangles are midpoint-split until each piece spans at most
a 3×3×3 cell block, then a separating-axis triangle/cube test decides each cell.
Deterministic and gap-free, unlike the random point sampling it replaces, which
left ~2% of surface cells unmarked; each missed cell was a pinhole the Stage-4
visibility ray-march could see through (false line-of-sight through a wall) or the
reachability flood fill could leak through (a sealed interior reading as navigable).
No winding/closure assumptions, so the non-watertight composed scene is fine.
Interior cells of thick solids still read as empty; that's also fine — Stage 3/4
use the fine occupancy for line-of-sight, which naturally excludes buried points.

GLASS (transparent surfaces occupy space but don't block sight). Surfaces whose
material is BLEND/MASK are classified per ~cell-sized piece by sampled base-color
alpha (`glass.py` drives window panes to alpha ≈ 0.065 inside an otherwise-opaque
texture): pieces below the occlusion cutoff land in a separate GLASS cell class.
Glass cells still count as surface for clearance / navigation / reachability — a
camera can't sit inside a pane, and the flood fill doesn't walk through a closed
window — but they are EXCLUDED from the occlusion set the Stage-4 visibility
ray-march tests (`fine_occluding`), so surfaces behind glazing are coverable.
OPAQUE materials (the glTF default, and the vast majority) ignore texture alpha
entirely, matching Stage 3/5 semantics.

Pure library (like the other stages): takes explicit paths; the server resolves the
cell and de-optimizes a library build to vanilla first (trimesh can't read
KTX2/Meshopt).

Outputs (under a cell's `splat/` dir):
  * `freespace.npz` — the reusable grid (origin, pitch, pitch_fine, refine, fine
    occupancy incl. the glass subset, coarse clearance). Consumed by Stages 3 and
    4; load via `load_free_space`.
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

# trimesh logs per-file load noise; silence it for whole-cell runs.
logging.getLogger("trimesh").setLevel(logging.ERROR)

# Artifacts written under a cell's `splat/` dir.
FREESPACE_NAME = "freespace.npz"
VOXELS_NAME = "voxels.bin"

DEFAULT_PITCH = 0.12       # coarse voxel edge (m) — navigational, for clearance
DEFAULT_REFINE = 3         # fine = pitch / refine (0.04 m by default) — for occlusion
_PITCH_CLAMP = (0.02, 1.0)
# Ceiling on the DENSE coarse grid (clearance + reachability arrays). These scale as
# O(volume) = (extent / coarse_pitch)³, so a large or mostly-empty scene (open arena,
# city, outer space) would otherwise exhaust RAM. When a scene exceeds this, the
# coarse block factor is grown (coarser navigation grid) until it fits; the FINE
# occupancy is sparse and stays at its fixed resolution, so occlusion/thin-gap
# accuracy is unaffected. ~48M cells ≈ ~1 GB peak across the transient EDT/label
# temporaries. Scenes under this are byte-identical to the pre-cap behavior.
DEFAULT_MAX_COARSE_CELLS = 48_000_000
# Exact voxelization: triangles are midpoint-split until every edge spans at most
# this many fine cells, so each piece's candidate block for the separating-axis
# test is at most 3×3×3 cells (a triangle's extent is bounded by its longest edge).
_SUBDIV_EDGE_CELLS = 2.0
# Material alpha modes whose sampled base-color alpha is meaningful (matches Stage
# 3/5): only BLEND/MASK surfaces can be classed as glass; OPAQUE ignores alpha.
_TRANSPARENT_ALPHA_MODES = ("BLEND", "MASK")
# A surface piece occludes line-of-sight when its base-color alpha is at least
# this. glass.py panes carry alpha ≈ 0.065 (transmissive — light passes); window
# frames / solid texels carry ≈ 1. MASK materials use their own glTF alphaCutoff.
_OCCLUDING_ALPHA = 0.5
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
    hollows, smaller keeps closets/nooks (see reachability below).

    `max_coarse_cells` bounds the DENSE coarse grid: if the requested `pitch` would
    make it larger, the coarse block factor is grown (a coarser navigation grid,
    coarser `pitch`) until it fits, so large/sparse scenes don't exhaust RAM. The
    fine occupancy is sparse and keeps its `pitch / refine` resolution regardless.
    Set to 0 to disable the cap (the exact pre-cap behavior)."""

    pitch: float = DEFAULT_PITCH
    refine: int = DEFAULT_REFINE
    margin: float = 1.5
    reachable_min_volume: float = 0.25
    max_coarse_cells: int = DEFAULT_MAX_COARSE_CELLS

    def as_summary(self) -> dict[str, Any]:
        return {
            "pitch": self.pitch,
            "refine": self.refine,
            "margin": self.margin,
            "reachable_min_volume": self.reachable_min_volume,
            "max_coarse_cells": self.max_coarse_cells,
        }


@dataclass(frozen=True)
class FreeSpace:
    """A loaded free-space grid. Aligned grids share `origin`: the SPARSE fine
    occupancy (`occ_lin` — sorted linear indices into a `fine_dims` grid of edge
    `pitch_fine`; `occ_lin_opaque` is the subset with opaque surface), the coarse
    clearance (`clearance`, edge `pitch`, metres), and the coarse `reachable` mask
    (free cells in a large enough connected component — exterior + rooms, minus
    tiny object-interior hollows). `refine = pitch / pitch_fine`.

    Two fine queries with different jobs: `fine_occupied` answers "is there ANY
    surface here" (physical presence — glass included), `fine_occluding` answers
    "does this cell block sight" (opaque only — glass passes light)."""

    origin: np.ndarray        # (3,) world corner of fine cell [0,0,0]
    pitch: float              # coarse edge (m)
    pitch_fine: float         # fine edge (m)
    refine: int
    fine_dims: np.ndarray     # (3,) int64 fine grid dims [fnx,fny,fnz]
    occ_lin: np.ndarray       # (K,) int64 SORTED linear indices of ALL occupied fine
                              # cells (opaque + glass) — clearance/reachability basis
    occ_lin_opaque: np.ndarray  # (K2,) int64 SORTED subset that blocks line-of-sight
    clearance: np.ndarray     # float32 [cx,cy,cz] (m)
    reachable: np.ndarray     # bool [cx,cy,cz] — navigable free space

    @property
    def coarse_dims(self) -> tuple[int, int, int]:
        return self.clearance.shape  # type: ignore[return-value]

    @property
    def fine_shape(self) -> tuple[int, int, int]:
        return (int(self.fine_dims[0]), int(self.fine_dims[1]), int(self.fine_dims[2]))

    @property
    def imin(self) -> np.ndarray:
        """Absolute fine-lattice index of the grid corner (`origin = imin *
        pitch_fine`). Queries bin with `floor(point / pitch_fine) - imin` — the SAME
        absolute-lattice expression the occupancy was BUILT with — so a query point
        and a build-time sample in the same physical cell get the identical index.
        Subtracting the float `origin` first (`floor((point - origin) / pitch_fine)`)
        rounds differently and, for some origins, mis-bins nearly every point."""
        return np.round(self.origin / self.pitch_fine).astype(np.int64)

    def _fine_member(self, points: np.ndarray, lin_sorted: np.ndarray) -> np.ndarray:
        """Boolean per world point: is its FINE voxel in the sorted sparse index set
        `lin_sorted`? Binary search; points outside the grid read as False."""
        idx = np.floor(points / self.pitch_fine).astype(np.int64) - self.imin
        dims = self.fine_dims
        inb = np.all((idx >= 0) & (idx < dims), axis=1)
        out = np.zeros(len(points), dtype=bool)
        if not lin_sorted.size or not inb.any():
            return out
        ii = idx[inb]
        lin = (ii[:, 0] * dims[1] + ii[:, 1]) * dims[2] + ii[:, 2]
        pos = np.searchsorted(lin_sorted, lin)
        pos = np.clip(pos, 0, lin_sorted.size - 1)
        out[inb] = lin_sorted[pos] == lin
        return out

    def fine_occupied(self, points: np.ndarray) -> np.ndarray:
        """Boolean per world point: does its FINE voxel contain ANY surface
        (opaque or glass)? The physical-presence query — clearance, navigation,
        and "is there something here" all mean this one."""
        return self._fine_member(points, self.occ_lin)

    def fine_occluding(self, points: np.ndarray) -> np.ndarray:
        """Boolean per world point: does its FINE voxel BLOCK line-of-sight?
        Opaque surface only — glass-classed cells pass light, so the Stage-4
        visibility ray-march can see (and plan coverage) through window panes."""
        return self._fine_member(points, self.occ_lin_opaque)

    def reachable_free(self, points: np.ndarray) -> np.ndarray:
        """Boolean per world point: is it in NAVIGABLE free space (a large enough
        free component)? This is what distinguishes exterior/room air from a solid's
        hollow interior — the signal Stage 3 uses to orient normals + cull hidden
        faces, and Stage 4 to place cameras. Outside-grid points read as False."""
        # Bin via the fine absolute lattice then block-reduce (matches the build's
        # `local // refine`); using `floor((point - origin) / pitch)` would round
        # inconsistently with how occupancy was built (see `imin`).
        idx = (np.floor(points / self.pitch_fine).astype(np.int64) - self.imin) // self.refine
        dims = np.asarray(self.reachable.shape)
        inb = np.all((idx >= 0) & (idx < dims), axis=1)
        out = np.zeros(len(points), dtype=bool)
        ii = idx[inb]
        out[inb] = self.reachable[ii[:, 0], ii[:, 1], ii[:, 2]]
        return out

    def free_candidates(
        self,
        max_clearance: float | None = None,
        spacing: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """NAVIGABLE free-cell world centres (+ their clearances) — the camera-
        candidate pool for Stage 4. Restricted to reachable space, so cameras never
        spawn inside solids/tiny hollows. No clearance floor: wall-adjacent cells
        are eligible, and camera–surface standoff is EMERGENT in Stage 4 (its
        scale ladder demands nothing closer than a patch's d_min).

        `max_clearance` keeps only the NEAR-SURFACE band (cells whose nearest surface
        is within reach of a camera); cells farther than any view distance can see
        nothing and are dropped. `spacing` (m) thins the band to ~one candidate per
        `spacing`-sized block, picking each block's cell CLOSEST to a surface — so
        the candidate count scales with the near-surface area, not a fixed budget."""
        free = self.reachable
        if max_clearance is not None:
            free = free & (self.clearance <= max_clearance)
        stride = 1 if spacing is None else max(1, int(round(spacing / self.pitch)))
        if stride == 1:
            cells = np.argwhere(free)
            centers = (self.origin + (cells + 0.5) * self.pitch).astype(np.float32)
            return centers, self.clearance[free]
        # One representative per stride³ block: the eligible cell CLOSEST to a
        # surface (smallest clearance) — the closest camera spots best serve the
        # near/detail-view coverage requirement. Pad up so no edge block is lost;
        # +inf marks ineligible cells.
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
    (flattened to sparse on load). Grids written before the glass class (no
    `occ_lin_glass`) degrade gracefully: every cell occludes, the old behavior."""
    with np.load(path) as z:
        files = set(z.files)
        if "occ_lin" in files:
            fine_dims = z["fine_dims"].astype(np.int64)
            occ_lin = z["occ_lin"].astype(np.int64)
        else:  # legacy dense occupancy → sparse (C-order linear matches fine_occupied)
            occ = z["occ_fine"].astype(bool)
            fine_dims = np.array(occ.shape, dtype=np.int64)
            occ_lin = np.flatnonzero(occ.reshape(-1)).astype(np.int64)
        glass = (
            z["occ_lin_glass"].astype(np.int64)
            if "occ_lin_glass" in files
            else np.zeros(0, dtype=np.int64)
        )
        return FreeSpace(
            origin=z["origin"].astype(np.float64),
            pitch=float(z["pitch"]),
            pitch_fine=float(z["pitch_fine"]),
            refine=int(z["refine"]),
            fine_dims=fine_dims,
            occ_lin=occ_lin,
            # Glass is stored as the disjoint transmissive-only subset, so the
            # occlusion set is simply "all minus glass" (stays sorted).
            occ_lin_opaque=np.setdiff1d(occ_lin, glass, assume_unique=True),
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


def _subdivide_edges(
    tris: np.ndarray, limit: float, attrs: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Midpoint-split triangles (T,3,3) until every edge is ≤ `limit` metres.
    Pure refinement — the pieces tile exactly the same surface — so voxelizing
    the pieces equals voxelizing the originals, but each piece's lattice AABB is
    small, which bounds the candidate-cell block the overlap test must check.
    Optional per-vertex `attrs` (T,3,K) — e.g. UVs — are split alongside
    (midpoint = linear interpolation, exact for triangle-affine attributes).
    Terminates: each split halves the longest edge and no edge ever grows."""
    k = 0 if attrs is None else attrs.shape[2]
    work = tris if k == 0 else np.concatenate([tris, attrs], axis=2)  # (T,3,3+K)
    out: list[np.ndarray] = []
    while len(work):
        edges = np.stack(
            [
                work[:, 1, :3] - work[:, 0, :3],
                work[:, 2, :3] - work[:, 1, :3],
                work[:, 0, :3] - work[:, 2, :3],
            ],
            axis=1,
        )
        elen = np.linalg.norm(edges, axis=2)  # (T, 3) — geometric lengths only
        big = elen.max(axis=1) > limit
        if not big.any():
            out.append(work)
            break
        out.append(work[~big])
        w = work[big]
        li = elen[big].argmax(axis=1)  # longest edge (a→b); c is opposite
        r = np.arange(len(w))
        a, b, c = w[r, li], w[r, (li + 1) % 3], w[r, (li + 2) % 3]
        mid = 0.5 * (a + b)  # interpolates position AND attrs
        work = np.concatenate(
            [np.stack([a, mid, c], axis=1), np.stack([mid, b, c], axis=1)], axis=0
        )
    done = np.concatenate(out, axis=0) if out else work
    if k == 0:
        return done, None
    return done[:, :, :3], done[:, :, 3:]


def _tri_cell_overlap(tris: np.ndarray, cells: np.ndarray, pitch: float) -> np.ndarray:
    """EXACT triangle ↔ cell intersection (separating-axis test), vectorized over
    pairs: `tris` (P,3,3) world vertices vs `cells` (P,3) absolute lattice coords
    of each cell's min corner (a closed cube of edge `pitch`). Returns (P,) bool.

    Two convex shapes overlap unless some axis separates them. The cube's three
    face axes are separated already by construction (candidates come from the
    triangle's own lattice AABB), which leaves: the triangle's plane (tested as
    |signed distance of the cube centre| ≤ the cube's half-diagonal projected
    onto the normal) and the nine edge×axis cross terms (tested as three 2D
    edge-function tests, one per axis projection, each against the cube corner
    most inside that edge — the Schwarz–Seidel / Akenine-Möller formulation).
    A degenerate projection (normal component 0) passes trivially and simply
    constrains nothing; the other projections + the plane test still decide."""
    v = tris - cells.astype(np.float64)[:, None, :] * pitch  # cell-local vertices
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])       # (P,3) plane normal
    # Plane vs cube (cube spans [0, pitch]³ in cell-local coords).
    s = np.einsum("pc,pc->p", n, 0.5 * pitch - v[:, 0])      # n · (centre − v0)
    ok = np.abs(s) <= 0.5 * pitch * np.abs(n).sum(axis=1)
    # 2D edge functions on the xy / yz / zx projections, signed by the normal
    # component so the half-planes face inward for either triangle winding.
    for w, u, t in ((2, 0, 1), (0, 1, 2), (1, 2, 0)):
        if not ok.any():
            break
        p2 = v[:, :, (u, t)]                                 # (P,3verts,2)
        e = np.roll(p2, -1, axis=1) - p2                     # (P,3edges,2)
        ne = np.stack([-e[..., 1], e[..., 0]], axis=-1)
        ne *= np.sign(n[:, w])[:, None, None]
        # Pass iff the cube corner most inside the edge is inside its half-plane.
        d = np.maximum(ne, 0.0).sum(axis=2) * pitch - np.einsum("pev,pev->pe", ne, p2)
        ok &= (d >= 0.0).all(axis=1)
    return ok


def _valid_tri_mask(tris: np.ndarray) -> np.ndarray:
    """Finite AND non-degenerate (nonzero area) triangles. Zero-area junk is
    dropped, matching what the old area-weighted sampler effectively did."""
    ok = np.isfinite(tris).all(axis=(1, 2))
    if ok.any():
        nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        ok &= np.nan_to_num(np.einsum("pc,pc->p", nrm, nrm)) > 0.0
    return ok


def _voxelize_cells(tris: np.ndarray, pitch: float) -> np.ndarray:
    """Absolute lattice coords (K,3) of every fine cell touched by the given
    (already-small, post-`_subdivide_edges`) triangles. Each triangle's lattice
    AABB bounds its candidate block; offsets are enumerated to the batch's max
    span, so any triangle size is handled (post-subdivision the span is ≤ 2)."""
    if not len(tris):
        return np.zeros((0, 3), dtype=np.int64)
    lo = np.floor(tris.min(axis=1) / pitch).astype(np.int64)         # (T,3)
    span = np.floor(tris.max(axis=1) / pitch).astype(np.int64) - lo  # ≤ 2/axis
    out: list[np.ndarray] = []
    single = (span == 0).all(axis=1)  # entirely inside one cell — no test needed
    if single.any():
        out.append(lo[single])
    rest = ~single
    if rest.any():
        t_r, lo_r, span_r = tris[rest], lo[rest], span[rest]
        dims = span_r.max(axis=0) + 1
        # Test each offset of the candidate block across all triangles at once;
        # iterating offsets (not pairs) keeps peak memory at O(T) per pass.
        for off in np.ndindex(int(dims[0]), int(dims[1]), int(dims[2])):
            offv = np.asarray(off, dtype=np.int64)
            m = (span_r >= offv).all(axis=1)
            if not m.any():
                continue
            cand = lo_r[m] + offv
            hit = _tri_cell_overlap(t_r[m], cand, pitch)
            if hit.any():
                out.append(cand[hit])
    if not out:
        return np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(out, axis=0)


def _voxelize_surface(tris: np.ndarray, pitch: float) -> np.ndarray:
    """Absolute lattice coords (K,3) of every fine cell touched by any triangle.
    DETERMINISTIC and complete — every cell a triangle overlaps is marked, none
    are missed — unlike the random point sampling this replaces (which left
    ~e^-oversample of surface cells unmarked: pinholes for rays / flood fill)."""
    tris = np.asarray(tris, dtype=np.float64)
    tris = tris[_valid_tri_mask(tris)]
    if not len(tris):
        return np.zeros((0, 3), dtype=np.int64)
    tris, _ = _subdivide_edges(tris, _SUBDIV_EDGE_CELLS * pitch)
    return _voxelize_cells(tris, pitch)


def _geom_alpha_sampler(geom: trimesh.Trimesh):  # noqa: ANN202 - (sampler|None, cutoff)
    """`(sampler, cutoff)` for one geometry, where `sampler` maps piece-centroid
    UVs (N,2) to base-color alpha in [0,1] — or `None` when the material is
    OPAQUE (the glTF default): opaque materials ignore texture alpha entirely,
    matching Stage 3's opacity init and Stage 5's rendering, so only BLEND/MASK
    geometry is ever classified as glass.

    Alpha comes from the base-color texture sampled via the same
    `trimesh.visual.color.uv_to_color` path Stage 3 uses (consistent UV
    conventions), times the `baseColorFactor` alpha; with no readable
    texture/UVs it falls back to the constant factor alpha. MASK materials use
    their own glTF `alphaCutoff`; BLEND uses the module occlusion cutoff.
    Unreadable textures and NaN UVs read as opaque (occluding) — the
    conservative direction for visibility."""
    visual = getattr(geom, "visual", None)
    material = getattr(visual, "material", None)
    mode = str(getattr(material, "alphaMode", None) or "OPAQUE").upper()
    if mode not in _TRANSPARENT_ALPHA_MODES:
        return None, _OCCLUDING_ALPHA
    cutoff = _OCCLUDING_ALPHA
    if mode == "MASK":
        cutoff = float(getattr(material, "alphaCutoff", None) or 0.5)

    fa = 1.0
    factor = getattr(material, "baseColorFactor", None)
    if factor is not None:
        arr = np.asarray(factor, dtype=np.float64).reshape(-1)
        if arr.size >= 4:
            fa = float(arr[3] / 255.0) if arr.max() > 1.0 else float(arr[3])

    image = getattr(material, "baseColorTexture", None)
    uv = getattr(visual, "uv", None)
    if image is None or uv is None:

        def constant(uvs: np.ndarray) -> np.ndarray:
            return np.full(len(uvs), fa, dtype=np.float64)

        return constant, cutoff

    def sample(uvs: np.ndarray) -> np.ndarray:
        try:
            rgba = np.asarray(
                trimesh.visual.color.uv_to_color(uvs, image), dtype=np.float64
            )
            return np.nan_to_num(rgba[:, 3] / 255.0 * fa, nan=1.0)
        except Exception:
            return np.full(len(uvs), 1.0, dtype=np.float64)

    return sample, cutoff


def _voxelize_geom(
    geom: trimesh.Trimesh, pitch: float
) -> tuple[np.ndarray, np.ndarray]:
    """`(opaque_cells, glass_cells)` — each (K,3) absolute lattice coords — for
    one geometry. OPAQUE materials take the pure-geometry path (no texture
    reads). BLEND/MASK triangles are subdivided to ~cell-sized pieces WITH their
    UVs, each piece classified by base-color alpha at its centroid, and the two
    groups voxelized separately — so one window mesh yields occluding frame
    cells AND transmissive pane cells at cell granularity. A cell containing
    both classes is resolved to opaque by the caller (opaque wins)."""
    tris = np.asarray(geom.triangles, dtype=np.float64)
    empty = np.zeros((0, 3), dtype=np.int64)
    sampler, cutoff = _geom_alpha_sampler(geom)
    if sampler is None:
        return _voxelize_surface(tris, pitch), empty

    uv = getattr(getattr(geom, "visual", None), "uv", None)
    if uv is not None and len(uv) == len(geom.vertices):
        uv3 = np.asarray(uv, dtype=np.float64)[np.asarray(geom.faces)]  # (T,3,2)
    else:
        uv3 = np.zeros((len(tris), 3, 2), dtype=np.float64)
    keep = _valid_tri_mask(tris)
    tris, uv3 = tris[keep], uv3[keep]
    if not len(tris):
        return empty, empty
    tris, uv3 = _subdivide_edges(tris, _SUBDIV_EDGE_CELLS * pitch, uv3)
    alpha = sampler(uv3.mean(axis=1))
    occluding = alpha >= cutoff
    return (
        _voxelize_cells(tris[occluding], pitch),
        _voxelize_cells(tris[~occluding], pitch),
    )


def _fine_grid_dims(
    lo: np.ndarray, hi: np.ndarray, margin: float, pitch_fine: float, refine: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fine-grid lower corner (absolute lattice index `imin`) + dims for a given
    coarse block factor `refine`. Dims are padded to a multiple of `refine` so the
    coarse grid (`dims // refine`) is an exact block-reduction. Factored out so the
    adaptive cap below sizes `refine` with the SAME arithmetic the build uses."""
    imin = np.floor((lo - margin) / pitch_fine).astype(np.int64) - 1
    imin -= imin % refine  # align down to a coarse-cell boundary
    imax = np.ceil((hi + margin) / pitch_fine).astype(np.int64) + 1
    fdims = (imax - imin) + 1
    fdims = ((np.maximum(fdims, refine) + refine - 1) // refine) * refine
    return imin, fdims


def _fit_refine(
    lo: np.ndarray,
    hi: np.ndarray,
    margin: float,
    pitch_fine: float,
    base_refine: int,
    max_coarse_cells: int,
) -> int:
    """Smallest coarse block factor ≥ `base_refine` whose DENSE coarse grid holds
    ≤ `max_coarse_cells` cells. This is what bounds the clearance/reachability
    memory regardless of scene size — only the COARSE grid coarsens; the fine
    occupancy stays sparse at `pitch_fine`. Returns `base_refine` unchanged whenever
    the scene already fits, so small/medium cells are byte-identical to the pre-cap
    behavior; only oversized scenes coarsen."""
    refine = max(1, int(base_refine))
    if not max_coarse_cells or max_coarse_cells <= 0:
        return refine
    _, fdims = _fine_grid_dims(lo, hi, margin, pitch_fine, refine)
    coarse = int(np.prod(fdims // refine))
    if coarse <= max_coarse_cells:
        return refine
    # Coarse cells ~ 1/refine³, so seed from the cube-root scaling, then verify and
    # bump by one until it fits (robust to the margin/padding rounding).
    refine = max(refine, int(np.ceil(refine * (coarse / max_coarse_cells) ** (1.0 / 3.0))))
    while refine < 100_000:
        _, fdims = _fine_grid_dims(lo, hi, margin, pitch_fine, refine)
        if int(np.prod(fdims // refine)) <= max_coarse_cells:
            break
        refine += 1
    return refine


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
    base_pitch = float(np.clip(params.pitch, *_PITCH_CLAMP))
    base_refine = max(1, int(params.refine))
    # FINE occupancy resolution is FIXED (scene-independent): the fine grid is sparse
    # (occ_lin), so accurate occlusion / thin-gap detection is kept at `pitch_fine`
    # for every scene. The COARSE block factor (hence `pitch`) is picked AFTER the
    # AABB is known so the dense coarse grid can be bounded (see `_fit_refine`).
    pitch_fine = base_pitch / base_refine
    margin = float(max(0.0, params.margin))

    total = len(ids)
    if progress is not None:
        progress(0, total, "")

    # One pass: voxelize each surface EXACTLY (every fine cell a triangle touches,
    # via _voxelize_geom) into an ABSOLUTE integer voxel lattice
    # (floor(point / pitch_fine)), keeping only the UNIQUE occupied cells — memory
    # tracks surface, not volume. Deterministic and gap-free, so the visibility
    # ray-march can't leak through pinholes in walls and the reachability flood
    # fill can't escape into sealed interiors. BLEND/MASK surfaces are classified
    # per piece into OPAQUE vs GLASS cells (glass occupies space but won't block
    # sight). The grid origin (which needs the AABB) is resolved afterwards;
    # absolute-lattice keys are origin-independent.
    opaque_parts: list[np.ndarray] = []
    glass_parts: list[np.ndarray] = []
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
                opaque_cells, glass_cells = _voxelize_geom(g, pitch_fine)
                if len(opaque_cells):
                    opaque_parts.append(np.unique(_abs_encode(opaque_cells)))
                if len(glass_cells):
                    glass_parts.append(np.unique(_abs_encode(glass_cells)))
            del m
        except Exception:  # skip a bad mesh, keep going
            pass
        if progress is not None:
            progress(done, total, node_id)

    if (not opaque_parts and not glass_parts) or not np.isfinite(lo).all():
        raise RuntimeError("no surface voxelized (every mesh failed or was empty)")
    empty_keys = np.zeros(0, dtype=np.int64)
    opaque_keys = (
        np.unique(np.concatenate(opaque_parts)) if opaque_parts else empty_keys
    )
    glass_keys = (
        np.unique(np.concatenate(glass_parts)) if glass_parts else empty_keys
    )
    # A cell with BOTH classes (e.g. window frame + pane) occludes: opaque wins,
    # keeping the two stored sets disjoint (glass = transmissive-only cells).
    glass_keys = np.setdiff1d(glass_keys, opaque_keys, assume_unique=True)
    all_keys = np.sort(np.concatenate([opaque_keys, glass_keys]))
    occ_abs = _abs_decode(all_keys)  # (K,3) absolute coords — ALL surface

    # Adaptive COARSE resolution — the memory bound. Pick the coarse block factor so
    # the dense clearance/reachability arrays hold ≤ max_coarse_cells; scenes that
    # already fit keep base_refine/base_pitch (byte-identical output), only large or
    # mostly-empty scenes coarsen. The fine occupancy below stays sparse + fixed-res.
    refine = _fit_refine(lo, hi, margin, pitch_fine, base_refine, params.max_coarse_cells)
    pitch = refine * pitch_fine
    coarse_capped = refine != base_refine

    # Fine grid over the AABB + margin, origin snapped to the absolute lattice (a
    # multiple of pitch_fine, and of `refine` cells so the coarse grid is an exact
    # block-reduction).
    imin, fdims = _fine_grid_dims(lo, hi, margin, pitch_fine, refine)
    fnx, fny, fnz = int(fdims[0]), int(fdims[1]), int(fdims[2])
    origin = imin.astype(np.float64) * pitch_fine

    # Occupied fine cells → local linear indices (drop any outside the padded grid).
    local = occ_abs - imin
    inb = np.all((local >= 0) & (local < fdims), axis=1)
    local = local[inb]
    occ_lin = np.unique((local[:, 0] * fny + local[:, 1]) * fnz + local[:, 2])

    # The glass subset through the identical transform (stays disjoint from the
    # opaque cells and a subset of occ_lin; load derives opaque = all − glass).
    glass_local = _abs_decode(glass_keys) - imin
    glass_local = glass_local[np.all((glass_local >= 0) & (glass_local < fdims), axis=1)]
    glass_lin = np.unique(
        (glass_local[:, 0] * fny + glass_local[:, 1]) * fnz + glass_local[:, 2]
    )

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
        # Floor at 8 cells so a coarsened (capped) grid still drops sub-voxel specks;
        # when coarse_capped raises `pitch`, the effective min pocket is 8·pitch³.
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
        occ_lin_glass=glass_lin.astype(np.int64),
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
        "refine_base": base_refine,
        "coarse_capped": coarse_capped,
        "dims_fine": [fnx, fny, fnz],
        "dims_coarse": list(occ_coarse.shape),
        "origin": [round(v, 5) for v in origin.tolist()],
        "scene_aabb": {"min": lo.tolist(), "max": hi.tolist()},
        "solid_voxels_fine": int(occ_lin.size),
        "glass_voxels_fine": int(glass_lin.size),
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
