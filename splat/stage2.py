"""Stage 2 — Free-space voxelizer (single uniform grid + two-phase flood fill).

The shared spatial FOUNDATION of the splat pipeline: discretizes a cell's composed
scene into ONE voxel grid of uniform pitch (the scene AABB grown by an exterior
margin so cameras can see outer silhouettes), classifies every cell as COVER
(surface), EMPTY (air a viewer could see — ambient + rescued cavities) or
GARBAGE (sealed interiors — object hollows, seams), then derives FREE — the
camera-placeable subset of EMPTY at sufficient clearance from any cover — and
writes a REUSABLE grid that both Stage 3 (surfel sampler — orient normals to
empty space, cull hidden faces) and Stage 4 (camera planner — candidates +
occlusion) consume. Nothing downstream recomputes occupancy; that's the point
of running this first.

OCCUPANCY is an EXACT surface voxelization: every cell whose cube a triangle
touches is marked — big triangles are midpoint-split until each piece spans at
most a 3×3×3 cell block, then a separating-axis triangle/cube test decides each
cell. Deterministic and gap-free, so the visibility ray-march can't leak through
pinholes in walls and the flood fill can't escape into sealed interiors. No
winding/closure assumptions, so the non-watertight composed scene is fine. The
cover set is stored SPARSELY (sorted linear indices), so its memory tracks the
amount of SURFACE; the dense fields (the empty mask, clearance) are built
STREAMED in X-SLABS — x is the outermost axis, so a slab is contiguous in
memory, on disk, and in the sorted cover keys — with full-resolution
intermediates in disk-backed .npy scratch files the OS pages in and out freely.
Resident memory is slab-sized plus O(surface + components), nearly independent
of scene volume; disk and runtime still scale with it.

GLASS (transparent surfaces occupy space but don't block sight). Classified
DURING voxelization — it is per-triangle-piece work fused into the same
subdivision pass that marks cover, so it touches only transparent-candidate
surface pieces, never the grid: surfaces whose material is BLEND/MASK are
classified per ~cell-sized piece by sampled base-color alpha (`glass.py` drives
window panes to alpha ≈ 0.065 inside an otherwise-opaque texture); pieces below
the occlusion cutoff land in a separate GLASS cell class. Glass cells still
count as cover for clearance / the flood fill — a camera can't sit inside a
pane, and the fill doesn't walk through a closed window — but they are EXCLUDED
from the occlusion set the Stage-4 visibility ray-march tests (`occluding`), so
surfaces behind glazing are coverable. OPAQUE materials (the glTF default, and
the vast majority) ignore texture alpha entirely.

EMPTY vs GARBAGE — the two-phase fill. Cover is voxelized for the WHOLE scene
first, then empty space is discovered by one 6-connected flood fill (face
neighbours only — a diagonal fill would slip through the corner-touching
staircase a thin oblique wall voxelizes into) whose result depends only on the
seed set; labels are assigned exactly once at the end (empty = reached, garbage
= the unreached remainder), so no precedence/override bookkeeping exists:

  * PHASE 1 (ambient). Seeds are (a) the grid's boundary shell — the exterior
    margin, always ambient — and (b) every object's AUGMENTED SHELL: the 1-cell
    ring around its mesh-derived AABB, keeping only cells that are unoccupied
    AND outside every other object's AABB. Since meshes (and therefore their
    hollow interiors) lie inside their AABBs, a surviving seed is provably
    ambient air — it can never sit inside another object's sealed hollow. The
    per-voxel exclusion (not per-direction) is what lets flush, mutually
    abutting architecture still seed: a wall's room-facing shell loses only the
    rows inside the floor/ceiling/side-wall boxes, so sealed rooms are seeded
    directly by their own bounding surfaces. The exclusion constrains SEEDS
    only — the fill itself flows wherever occupancy permits (over sofa seats,
    under arches), or bbox-covered air would wrongly read sealed.
  * PHASE 2 (nested rescue, to fixpoint). An object NONE of whose shell cells
    were reached is sealed away from all discovered free space — fully nested
    (a bottle in a closed cabinet), which per the product decision should be
    coverable: its shell seeds the cavity (exclusion waived). The trigger is
    per-object ALL-OR-NOTHING: one reached shell cell means the object is
    exposed (a cushion whose top sits in room air) and its buried faces are
    interpenetration seams to cull, not a cavity to open. Rounds admit ONE
    cavity at a time, preferring the component adjacent to the most pending
    objects (component size, then lowest label, break ties), and re-evaluate —
    an outside-in order, so in a scene whose interior phase 1 couldn't seed at
    all (a sealed hull whose bbox swallows every shell) the room air opens
    first and a clipping cushion drops out of the pending set BEFORE its
    sofa's hollow could be considered. Deterministic; rounds ≤ nesting depth.

    Consequences, decided deliberately: rescue is a FURNISHED-cavity detector —
    a sealed room containing no standalone object stays garbage (nothing worth
    seeing there); an object straddling an open and a sealed region rescues
    neither; the nesting distinction is only as sharp as the pitch.

CLEARANCE → FREE (the stage-2 clearance pass, run right after the fill).
Clearance = metres from each cell to the nearest cover cell (EDT over the one
grid). FREE = the EMPTY cells with clearance ≥ `clearance_m` — where a camera
can physically sit. The mask is DERIVED, not stored: the npz carries the empty
mask, the clearance field and the `clearance_m` scalar, and `load_free_space`
computes `free = empty & (clearance >= clearance_m)` — so mask and threshold
can never disagree, and re-thresholding (`apply_clearance`) rewrites ONE scalar
without re-voxelizing. Consumers split cleanly: Stage 3's orient/cull reads
EMPTY (`empty_at` — a surfel beside a 10 cm gap has visible air there even
though no camera fits); Stage 4's candidates read FREE (`free_candidates`,
which further band-limits by clearance + thins by spacing).

Pure library (like the other stages): takes explicit paths and reads the
meshes AS-IS through `splat.assets.load_geoms` — vanilla glTF via trimesh,
KTX2/Meshopt sets via the in-process decoder (no de-optimization step). On
compressed sets texel data is unavailable, so BLEND/MASK glass classification
falls back to the constant `baseColorFactor` alpha.

Outputs (under a cell's `splat/` dir):
  * `freespace.npz` — the reusable grid (origin, pitch, dims, sparse cover incl.
    the glass subset, clearance, the empty mask, the baked `clearance_m`).
    Consumed by Stages 3 and 4; load via `load_free_space`. Older layouts
    (dual-resolution, or pre-clearance single-grid) are rejected with a clear
    re-run error rather than carried as compat shims.
  * `freespace.npz.empty.npy` — the empty mask as an uncompressed sidecar,
    promoted from the fill's scratch so Stage 3's pool workers memory-map one
    shared copy without ever decompressing the npz mask themselves.
  * `voxels.bin` — the SVX3 viz pack for the client overlay: VOLUMETRIC
    boundary shells, not points. Cover, garbage, and the free volume (at a
    ladder of clearance thresholds, the baked one included) are each
    surface-extracted into run-merged exposed-face quads (`_boundary_quads`),
    so the client renders merged translucent shapes with interior faces
    culled, and the clearance slider swaps pre-meshed shells (see the wire
    layout at `_VIZ_MAGIC`).
"""

from __future__ import annotations

import logging
import os
import struct
import time
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
# Uncompressed empty-mask twin (`freespace.npz.empty.npy`), written beside the
# npz so Stage 3's pool workers memory-map ONE shared copy instead of each
# decompressing a private mask. Stage 3 treats a sidecar older than the npz as
# stale — hence the post-write touch in `compute_free_space`.
_EMPTY_SIDECAR_SUFFIX = ".empty.npy"
# Viz pack (SVX3): VOLUMETRIC boundary shells per voxel class. Header = magic +
# u32 LE counts (cover quads, garbage quads, free shells); then cover quads,
# garbage quads, then per shell a header (f32 clearance threshold, u32 quads,
# u32 cell count) + its quads. One quad = one run-merged exposed voxel face:
# u16 cell x,y,z · u8 face (axis*2, +side at the cell's max corner on that
# axis; axis*2+1, −side at the min corner) · u8 pad · u16 run length along the
# in-plane run axis (z for x/y faces, y for z faces) — 10 bytes.
_VIZ_MAGIC = b"SVX3"
_VIZ_HEADER = struct.Struct("<4sIII")
_SHELL_HEADER = struct.Struct("<fII")
_QUAD_DTYPE = np.dtype(
    [("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("f", "u1"), ("p", "u1"), ("r", "<u2")]
)
# The free volume is pre-meshed at these thresholds (voxel multiples, capped at
# the scene's clearance ceiling, the baked threshold always included) so the
# client slider swaps shells instantly instead of re-meshing.
_SHELL_STEPS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)

DEFAULT_PITCH = 0.03      # the uniform voxel edge (m) — one grid for everything
# The baked FREE threshold (m): a camera needs at least this much clearance.
# The default equals ONE VOXEL at the default pitch — the most permissive
# meaningful setting: every empty cell is ≥ 1 voxel from cover by definition,
# so FREE ≡ EMPTY out of the box, and per-scene tightening happens through the
# client slider's apply (`apply_clearance`, in place). For reference, 0.35 was
# the measured knee of the passage-survival curve across the benchmark runs
# (~2/3 of sub-2m facing gaps keep centerline candidates) — a sensible value
# when tightening for production plans.
DEFAULT_CLEARANCE = 0.03
# Tolerance for every `clearance >= threshold` test. A cell exactly ON a
# lattice distance stores just below its true value (float32 pitch products,
# then float16 in the npz: 1 voxel × 0.03 → 0.0299988), so an exact-equality
# threshold — e.g. the one-voxel default above — would otherwise exclude the
# very tier it names. 2.5e-4 absorbs float16 half-ulp rounding for thresholds
# up to ~0.5 m while staying far below the smallest gap between distinct EDT
# tiers, so no cell from a genuinely lower tier can sneak in.
_CLEARANCE_EPS = 2.5e-4
_PITCH_CLAMP = (0.02, 1.0)
# Flat-chunk length for whole-grid reductions (the FREE count, clearance stats,
# float16 conversion): bounds each per-chunk temporary to tens of MB where a
# single whole-grid call would materialize multi-GB copies.
_CHUNK_CELLS = 1 << 24
# Per-slab working-set target for the streamed grid passes (labels + masks +
# temps, ~16 B/cell during the fill). Purely a resource knob: every streamed
# pass produces identical output for ANY slab partition.
_SLAB_TARGET_BYTES = 256 * 1024**2
# Scratch files older than this are swept at build start. Live scratches are
# never this old (phases write continuously); only hard-killed builds — whose
# cleanup never ran — leave older ones behind, and those are multi-GB litter.
_SCRATCH_TTL_S = 6 * 3600.0
# Parabola x-pass sentinel: "this plane holds no cover" (see `_parabola_pass`).
_NO_SITE = np.int64(1) << 62
# Exact voxelization: triangles are midpoint-split until every edge spans at most
# this many cells, so each piece's candidate block for the separating-axis
# test is at most 3×3×3 cells (a triangle's extent is bounded by its longest edge).
_SUBDIV_EDGE_CELLS = 2.0
# Material alpha modes whose sampled base-color alpha is meaningful (matches Stage
# 3/5): only BLEND/MASK surfaces can be classed as glass; OPAQUE ignores alpha.
_TRANSPARENT_ALPHA_MODES = ("BLEND", "MASK")
# A surface piece occludes line-of-sight when its base-color alpha is at least
# this. glass.py panes carry alpha ≈ 0.065 (transmissive — light passes); window
# frames / solid texels carry ≈ 1. MASK materials use their own glTF alphaCutoff.
_OCCLUDING_ALPHA = 0.5

# Absolute-lattice voxel encoding: occupied voxels are binned in a scene-
# independent integer lattice (floor(point / pitch)) so binning needs no origin
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

# progress(done, total, step) — `step` names the current phase ("voxelize" |
# "reduce" | "fill" | "clearance" | "write" | "viz"). done/total count items in
# the STREAMING phases (objects for "voxelize"; grid slabs/bundles for "fill",
# "clearance" and "viz"); single-pass marker phases report (0, 0) — the step
# name is the signal, and a constant step string lets the caller meter a rate.
ProgressCb = Callable[[int, int, str], None]


@dataclass(frozen=True)
class FreeSpaceParams:
    """Stage-2 knobs. `pitch` is the uniform voxel edge of the single grid;
    `margin` grows the grid beyond the scene AABB so exterior camera vantages
    exist (Stage 4); `clearance` (m) is the baked FREE threshold — empty cells
    at least this far from any cover cell are camera-placeable. `workers`
    parallelizes the per-object mesh pass (0 = auto: sized to fit available
    RAM, capped at min(cores, 8); 1 = serial); the output is byte-identical
    for any value."""

    pitch: float = DEFAULT_PITCH
    margin: float = 1.5
    clearance: float = DEFAULT_CLEARANCE
    workers: int = 0

    def as_summary(self) -> dict[str, Any]:
        return {
            "pitch": self.pitch,
            "margin": self.margin,
            "clearance": self.clearance,
            "workers": self.workers,
        }


@dataclass(frozen=True)
class FreeSpace:
    """A loaded free-space grid — ONE uniform lattice of edge `pitch`: the
    SPARSE cover (`occ_lin` — sorted linear indices into a `dims` grid;
    `occ_lin_opaque` is the subset with opaque surface), `clearance` (metres to
    the nearest cover cell), the `empty` mask (ambient air + rescued cavities —
    everything a viewer could see; the unreached remainder is garbage: sealed
    object hollows and seams), and the baked `clearance_m` threshold. FREE —
    the camera-placeable subset — is DERIVED: `free = empty & (clearance >=
    clearance_m)`, so the mask can never disagree with the threshold.

    Two cover queries with different jobs: `occupied` answers "is there ANY
    surface here" (physical presence — glass included), `occluding` answers
    "does this cell block sight" (opaque only — glass passes light)."""

    origin: np.ndarray        # (3,) world corner of cell [0,0,0]
    pitch: float              # voxel edge (m)
    dims: np.ndarray          # (3,) int64 grid dims [nx,ny,nz]
    occ_lin: np.ndarray       # (K,) int64 SORTED linear indices of ALL cover
                              # cells (opaque + glass) — clearance/fill basis
    occ_lin_opaque: np.ndarray  # (K2,) int64 SORTED subset that blocks line-of-sight
    clearance: np.ndarray     # float32 [nx,ny,nz] (m)
    empty: np.ndarray         # bool [nx,ny,nz] — EMPTY (viewable air) cells
    clearance_m: float        # baked FREE threshold (m)

    @property
    def free(self) -> np.ndarray:
        """bool [nx,ny,nz] — FREE (camera-placeable) cells, derived from the
        baked threshold (epsilon absorbs float16/lattice rounding so a
        threshold equal to a lattice distance includes its own tier)."""
        return self.empty & (self.clearance >= self.clearance_m - _CLEARANCE_EPS)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.dims[0]), int(self.dims[1]), int(self.dims[2]))

    @property
    def imin(self) -> np.ndarray:
        """Absolute lattice index of the grid corner (`origin = imin * pitch`).
        Queries bin with `floor(point / pitch) - imin` — the SAME absolute-
        lattice expression the occupancy was BUILT with — so a query point and a
        build-time sample in the same physical cell get the identical index.
        Subtracting the float `origin` first (`floor((point - origin) / pitch)`)
        rounds differently and, for some origins, mis-bins nearly every point."""
        return np.round(self.origin / self.pitch).astype(np.int64)

    def _bin(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(grid indices, in-bounds mask) per world point."""
        idx = np.floor(points / self.pitch).astype(np.int64) - self.imin
        inb = np.all((idx >= 0) & (idx < self.dims), axis=1)
        return idx, inb

    def _member(self, points: np.ndarray, lin_sorted: np.ndarray) -> np.ndarray:
        """Boolean per world point: is its voxel in the sorted sparse index set
        `lin_sorted`? Binary search; points outside the grid read as False."""
        idx, inb = self._bin(points)
        out = np.zeros(len(points), dtype=bool)
        if not lin_sorted.size or not inb.any():
            return out
        ii = idx[inb]
        lin = (ii[:, 0] * self.dims[1] + ii[:, 1]) * self.dims[2] + ii[:, 2]
        pos = np.searchsorted(lin_sorted, lin)
        pos = np.clip(pos, 0, lin_sorted.size - 1)
        out[inb] = lin_sorted[pos] == lin
        return out

    def occupied(self, points: np.ndarray) -> np.ndarray:
        """Boolean per world point: does its voxel contain ANY surface
        (opaque or glass)? The physical-presence query — clearance, navigation,
        and "is there something here" all mean this one."""
        return self._member(points, self.occ_lin)

    def occluding(self, points: np.ndarray) -> np.ndarray:
        """Boolean per world point: does its voxel BLOCK line-of-sight?
        Opaque surface only — glass-classed cells pass light, so the Stage-4
        visibility ray-march can see (and plan coverage) through window panes."""
        return self._member(points, self.occ_lin_opaque)

    def empty_at(self, points: np.ndarray) -> np.ndarray:
        """Boolean per world point: is it in EMPTY space (ambient air or a
        rescued cavity)? This is what distinguishes viewable air from a solid's
        sealed hollow — the signal Stage 3 uses to orient normals + cull hidden
        faces. Deliberately NOT clearance-filtered: a surfel beside a thin gap
        has visible air there even though no camera fits (that's `free`).
        Outside-grid points read as False."""
        idx, inb = self._bin(points)
        out = np.zeros(len(points), dtype=bool)
        ii = idx[inb]
        out[inb] = self.empty[ii[:, 0], ii[:, 1], ii[:, 2]]
        return out

    def free_candidates(
        self,
        min_clearance: float,
        max_clearance: float | None = None,
        spacing: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """FREE-cell world centres (+ their clearances) — the camera-candidate
        pool for Stage 4, based on the BAKED free mask (empty ∧ clearance ≥
        `clearance_m`), so cameras never spawn inside solids, sealed hollows,
        or hugging a surface. `min_clearance` can only tighten further — the
        baked threshold is the floor (re-bake via `apply_clearance` to relax).

        `max_clearance` keeps only the NEAR-SURFACE band (cells whose nearest surface
        is within reach of a camera); cells farther than any view distance can see
        nothing and are dropped. `spacing` (m) thins the band to ~one candidate per
        `spacing`-sized block, picking the MOST-OPEN cell in each block — so the
        candidate count scales with the near-surface area, not a fixed budget."""
        eligible = self.free & (self.clearance >= min_clearance - _CLEARANCE_EPS)
        if max_clearance is not None:
            eligible &= self.clearance <= max_clearance
        stride = 1 if spacing is None else max(1, int(round(spacing / self.pitch)))
        if stride == 1:
            cells = np.argwhere(eligible)
            centers = (self.origin + (cells + 0.5) * self.pitch).astype(np.float32)
            return centers, self.clearance[eligible]
        # One representative per stride³ block: the eligible cell CLOSEST to a surface
        # (smallest clearance, but still ≥ collision_clearance) — closest safe camera
        # spots best serve the near/detail-view coverage requirement. Pad up so no
        # edge block is lost; +inf marks ineligible cells.
        score = np.where(eligible, self.clearance, np.float32(np.inf))
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
    """Load a `freespace.npz` written by `compute_free_space`. Older layouts
    (dual-resolution, or single-grid without the baked clearance) are rejected
    with a re-run error — no compat shims."""
    with np.load(path) as z:
        files = set(z.files)
        if "empty" not in files or "clearance_m" not in files or "dims" not in files:
            raise ValueError(
                f"{path} is a pre-clearance free-space grid — re-run Stage 2"
            )
        occ_lin = z["occ_lin"].astype(np.int64)
        glass = z["occ_lin_glass"].astype(np.int64)
        return FreeSpace(
            origin=z["origin"].astype(np.float64),
            pitch=float(z["pitch"]),
            dims=z["dims"].astype(np.int64),
            occ_lin=occ_lin,
            # Glass is stored as the disjoint transmissive-only subset, so the
            # occlusion set is simply "all minus glass" (stays sorted).
            occ_lin_opaque=np.setdiff1d(occ_lin, glass, assume_unique=True),
            clearance=z["clearance"].astype(np.float32),
            empty=z["empty"].astype(bool),
            clearance_m=float(z["clearance_m"]),
        )


def apply_clearance(path: Path, clearance: float) -> dict[str, Any]:
    """Re-bake the FREE threshold of an existing `freespace.npz` IN PLACE — the
    'apply' behind the client's clearance slider. No re-voxelization: the fill's
    empty mask and the clearance field are threshold-independent, so only the
    `clearance_m` scalar changes (the viz pack stays valid too — its empty
    quads carry per-cell clearance). Returns a summary patch
    `{clearance, free_voxels}` for the sidecar."""
    path = Path(path)
    with np.load(path) as z:
        if "empty" not in z.files or "clearance_m" not in z.files:
            raise ValueError(f"{path} is a pre-clearance grid — re-run Stage 2")
        data = {k: z[k] for k in z.files}
    c = float(clearance)
    data["clearance_m"] = np.float64(c)
    free_count = int(
        (
            data["empty"].astype(bool)
            & (data["clearance"].astype(np.float32) >= c - _CLEARANCE_EPS)
        ).sum()
    )
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **data)
    tmp.replace(path)
    return {"clearance": c, "free_voxels": free_count}


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
    """Absolute lattice coords (K,3) of every cell touched by the given
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
    """Absolute lattice coords (K,3) of every cell touched by any triangle.
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


def _grid_dims(
    lo: np.ndarray, hi: np.ndarray, margin: float, pitch: float
) -> tuple[np.ndarray, np.ndarray]:
    """Grid lower corner (absolute lattice index) + dims covering the scene AABB
    + `margin`, padded one extra cell so the boundary shell — phase 1's
    always-ambient seed layer — lies strictly outside the padded AABB."""
    imin = np.floor((lo - margin) / pitch).astype(np.int64) - 1
    imax = np.ceil((hi + margin) / pitch).astype(np.int64) + 1
    return imin, imax - imin + 1


def _shell_slabs(
    vlo: np.ndarray, vhi: np.ndarray, dims: np.ndarray
) -> list[tuple[slice, slice, slice]]:
    """The six one-cell-thick faces of the AUGMENTED box `[vlo-1, vhi+1]` — an
    object's shell ring — as index slabs, skipping faces that fall outside the
    grid. Slabs overlap along the box edges; every consumer treats them as a
    set, so the duplication is harmless."""
    a_lo = vlo - 1
    a_hi = vhi + 1
    c_lo = np.maximum(a_lo, 0)
    c_hi = np.minimum(a_hi, dims - 1)
    slabs: list[tuple[slice, slice, slice]] = []
    for axis in range(3):
        for face in (int(a_lo[axis]), int(a_hi[axis])):
            if face < 0 or face >= int(dims[axis]):
                continue  # clipped by the grid — that side has no ring layer
            sl = [
                slice(int(c_lo[0]), int(c_hi[0]) + 1),
                slice(int(c_lo[1]), int(c_hi[1]) + 1),
                slice(int(c_lo[2]), int(c_hi[2]) + 1),
            ]
            sl[axis] = slice(face, face + 1)
            slabs.append((sl[0], sl[1], sl[2]))
    return slabs


# --- streamed grid passes ------------------------------------------------------
# The dense fields are never held whole: the grid is processed in X-SLABS. x is
# the array's outermost axis, so a slab [x0, x1) is one contiguous range of
# linear indices — contiguous in memory, on disk, and (because `occ_lin` is
# sorted) one contiguous SEGMENT of the sparse cover keys, so any slab's dense
# occupancy is rebuilt on demand and no full occupancy array exists at all.


def _slab_planes(ny: int, nz: int) -> int:
    """X-planes per slab so a slab's working set (~16 B/cell across labels,
    masks and temps) stays near `_SLAB_TARGET_BYTES`. A resource knob only —
    every streamed pass yields identical output for any slab partition."""
    return max(1, int(_SLAB_TARGET_BYTES // max(1, ny * nz * 16)))


def _occ_slab(occ_lin: np.ndarray, x0: int, x1: int, ny: int, nz: int) -> np.ndarray:
    """Dense occupancy of x-planes [x0, x1), rebuilt from the sorted sparse
    cover keys (binary search bounds one contiguous key segment per slab)."""
    plane = ny * nz
    lo, hi = np.searchsorted(occ_lin, [x0 * plane, x1 * plane])
    slab = np.zeros((x1 - x0) * plane, dtype=bool)
    slab[occ_lin[lo:hi] - x0 * plane] = True
    return slab.reshape(x1 - x0, ny, nz)


def _covered_slab(
    boxes: list[tuple[str, np.ndarray, np.ndarray]], x0: int, x1: int, ny: int, nz: int
) -> np.ndarray:
    """Cells of x-planes [x0, x1) covered by ANY object AABB. Ring cells sit on
    the shell of the AUGMENTED box [vlo-1, vhi+1], strictly outside their own
    object's box, so exactly where the seed test reads it "covered by any box"
    ⇔ "covered by ANOTHER object's box" (the phase-1 seed exclusion)."""
    slab = np.zeros((x1 - x0, ny, nz), dtype=bool)
    for _nid, vlo, vhi in boxes:
        a, b = max(int(vlo[0]), x0), min(int(vhi[0]) + 1, x1)
        if a < b:
            slab[a - x0 : b - x0, vlo[1] : vhi[1] + 1, vlo[2] : vhi[2] + 1] = True
    return slab


def _fill_streamed(
    occ_lin: np.ndarray,
    dims: np.ndarray,
    boxes: list[tuple[str, np.ndarray, np.ndarray]],
    empty_path: Path,
    tick: Callable[[], None] | None = None,
) -> tuple[int, dict[str, Any]]:
    """EMPTY mask via the two-phase fill (module docstring), streamed: classic
    two-pass connected-component labeling over x-slabs. Sweep 1 labels each
    slab independently (6-connectivity, so slabs couple only through the one
    shared face plane), records seam label pairs, per-component sizes, and the
    margin/ring seed bookkeeping; a union-find merge then yields the global
    components, on which phase 1 + the nested-object rescue run UNCHANGED
    (they only ever consumed component-level tables). Sweep 2 relabels each
    slab (deterministic) and writes the final mask straight into the
    uncompressed `.npy` at `empty_path` — the multi-GB label array of the
    whole-grid version never exists.

    Provisional ids are `slab base + scipy local label`; scipy numbers labels
    in first-encounter raster order (and slabs partition raster order), so the
    MIN provisional id of a merged component ranks components exactly as a
    whole-grid scan would — final ids replicate the unstreamed numbering, and
    with it the rescue's lowest-label tie-break, for any slab partition.
    Returns (count of cells opened by the rescue, fill stats)."""
    nx, ny, nz = (int(v) for v in dims)
    step = _slab_planes(ny, nz)
    starts = list(range(0, nx, step))
    struct = ndimage.generate_binary_structure(3, 1)

    bases: list[int] = [0] * len(starts)  # provisional-id base per slab
    n_prov = 0
    size_parts: list[np.ndarray] = []     # per-slab component cell counts
    pairs: list[np.ndarray] = []          # (P,2) provisional ids touching a seam
    seed_parts: list[np.ndarray] = []     # provisional ids seeded (margin/rings)
    ring_prov: list[list[np.ndarray]] = [[] for _ in boxes]
    prev_plane: np.ndarray | None = None  # previous slab's last plane, prov ids

    for si, x0 in enumerate(starts):
        x1 = min(x0 + step, nx)
        air = _occ_slab(occ_lin, x0, x1, ny, nz)
        np.logical_not(air, out=air)
        labels, n_local = ndimage.label(air, struct)
        del air
        base = bases[si] = n_prov
        n_prov += int(n_local)
        size_parts.append(np.bincount(labels.ravel(), minlength=n_local + 1)[1:])

        def prov_of(lab: np.ndarray) -> np.ndarray:
            u = np.unique(lab)
            return u[u > 0].astype(np.int64) + base

        # Seam with the previous slab: 6-connectivity couples only same-(y,z)
        # cells of the two touching planes.
        first = labels[0].astype(np.int64)
        first[first > 0] += base
        if prev_plane is not None:
            both = (prev_plane > 0) & (first > 0)
            if both.any():
                pairs.append(
                    np.unique(np.stack([prev_plane[both], first[both]], axis=1), axis=0)
                )
        prev_plane = labels[-1].astype(np.int64)
        prev_plane[prev_plane > 0] += base

        # Margin seeds: the grid's boundary shell lies beyond the scene AABB +
        # margin (see `_grid_dims`), so its air cells are ambient by construction.
        faces = [labels[:, 0, :], labels[:, -1, :], labels[:, :, 0], labels[:, :, -1]]
        if x0 == 0:
            faces.append(labels[0])
        if x1 == nx:
            faces.append(labels[-1])
        seed_parts.extend(prov_of(f) for f in faces)

        # Phase-1 rings: this slab's slice of every object's shell ring — ALL
        # air ring components (rescue trigger) + the uncovered ones (seeds).
        covered = _covered_slab(boxes, x0, x1, ny, nz)
        for bi, (_nid, vlo, vhi) in enumerate(boxes):
            for sl in _shell_slabs(vlo, vhi, dims):
                a, b = max(sl[0].start, x0), min(sl[0].stop, x1)
                if a >= b:
                    continue
                lsl = (slice(a - x0, b - x0), sl[1], sl[2])
                lab = labels[lsl]
                airm = lab > 0
                if not airm.any():
                    continue
                ring_prov[bi].append(np.unique(lab[airm]).astype(np.int64) + base)
                seed = airm & ~covered[lsl]
                if seed.any():
                    seed_parts.append(np.unique(lab[seed]).astype(np.int64) + base)
        del labels, covered
        if tick is not None:
            tick()

    # Merge the per-slab labelings: union-find keeping the SMALLEST provisional
    # id as each root (= global first-encounter order, see docstring).
    parent = np.arange(n_prov + 1, dtype=np.int64)

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = int(parent[root])
        while parent[i] != root:  # path compression
            parent[i], i = root, int(parent[i])
        return root

    for pr in pairs:
        for a, b in pr:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                if ra > rb:
                    ra, rb = rb, ra
                parent[rb] = ra
    ids = np.arange(1, n_prov + 1, dtype=np.int64)
    roots = parent[ids]
    while True:  # vectorized pointer jumping to full resolution
        nxt = parent[roots]
        if (nxt == roots).all():
            break
        roots = nxt

    uniq_roots, root_rank = np.unique(roots, return_inverse=True)
    n_comp = int(uniq_roots.size)
    # Final ids 1..K in ascending root order = first-encounter raster order.
    prov_to_final = np.zeros(n_prov + 1, dtype=np.int64)
    prov_to_final[1:] = root_rank + 1
    sizes = np.zeros(n_comp + 1, dtype=np.int64)
    np.add.at(
        sizes,
        prov_to_final[1:],
        np.concatenate(size_parts) if size_parts else np.zeros(0, np.int64),
    )

    reached = np.zeros(n_comp + 1, dtype=bool)
    for part in seed_parts:
        reached[prov_to_final[part]] = True
    seeded_components = int(reached.sum())
    ring_labels = [
        np.unique(prov_to_final[np.concatenate(rp)]) if rp else np.zeros(0, np.int64)
        for rp in ring_prov
    ]

    # Phase 2 — nested rescue, to fixpoint. Trigger: an object with ZERO reached
    # ring cells is sealed away from all discovered free space (fully nested).
    # Admit ONE cavity per round — the component adjacent to the most pending
    # objects (then the largest, then the lowest label) — and re-evaluate: the
    # outside-in order that opens a hull's room air before a clipping cushion's
    # sofa hollow ever gets considered (the cushion drops out of pending first).
    pending = [i for i in range(len(boxes)) if ring_labels[i].size]
    dead = len(boxes) - len(pending)  # ring fully occupied — zero-gap embedded
    sealed_ids: set[str] = set()
    rescued_comps: list[int] = []
    rescue_rounds = 0
    while True:
        pending = [i for i in pending if not reached[ring_labels[i]].any()]
        if not pending:
            break
        sealed_ids.update(boxes[i][0] for i in pending)
        adj = np.zeros(n_comp + 1, dtype=np.int64)
        for i in pending:
            rl = ring_labels[i]
            adj[rl[~reached[rl]]] += 1
        order = np.lexsort(
            (-np.arange(n_comp + 1, dtype=np.int64), sizes, adj)
        )
        reached[int(order[-1])] = True
        rescued_comps.append(int(order[-1]))
        rescue_rounds += 1

    # Sweep 2: relabel each slab (scipy is deterministic, so local labels and
    # bases replay exactly) and stream the final mask to disk.
    mm = np.lib.format.open_memmap(
        empty_path, mode="w+", dtype=bool, shape=(nx, ny, nz)
    )
    for si, x0 in enumerate(starts):
        x1 = min(x0 + step, nx)
        air = _occ_slab(occ_lin, x0, x1, ny, nz)
        np.logical_not(air, out=air)
        labels, n_local = ndimage.label(air, struct)
        del air
        base = bases[si]
        lut = np.zeros(n_local + 1, dtype=bool)
        lut[1:] = reached[prov_to_final[base + 1 : base + n_local + 1]]
        mm[x0:x1] = lut[labels]
        del labels
        if tick is not None:
            tick()
    mm.flush()
    del mm

    rescued_cells = (
        int(sizes[np.asarray(rescued_comps, dtype=np.int64)].sum())
        if rescued_comps
        else 0
    )
    stats = {
        "components": int(n_comp),
        "components_open": int(reached.sum()),
        "components_seeded": seeded_components,
        "rescue_rounds": rescue_rounds,
        "objects": len(boxes),
        "objects_sealed": sorted(sealed_ids),
        "objects_embedded": dead,
    }
    return rescued_cells, stats


def _voxelize_object_task(
    node_id: str, glb_path: str, pitch: float
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Voxelize ONE placed GLB → `(node_id, opaque_keys, glass_keys, lo, hi)`.

    The process-pool work unit of the mesh pass: everything it returns is
    origin-independent (sorted unique absolute-lattice int64 keys + the mesh
    world AABB), so per-object results merge with `np.unique(concatenate(...))`
    in ANY completion order — the parallel build is byte-identical to the
    serial one. A bad mesh returns empty keys (the serial skip behavior)."""
    from splat.assets import load_geoms

    empty = np.zeros(0, dtype=np.int64)
    opaque_parts: list[np.ndarray] = []
    glass_parts: list[np.ndarray] = []
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    try:
        geoms = load_geoms(Path(glb_path))
        for g in geoms:
            if len(g.faces) == 0 or g.area <= 0:
                continue
            b = np.asarray(g.bounds, dtype=float)
            lo = b[0].copy() if lo is None else np.minimum(lo, b[0])
            hi = b[1].copy() if hi is None else np.maximum(hi, b[1])
            opaque_cells, glass_cells = _voxelize_geom(g, pitch)
            if len(opaque_cells):
                opaque_parts.append(np.unique(_abs_encode(opaque_cells)))
            if len(glass_cells):
                glass_parts.append(np.unique(_abs_encode(glass_cells)))
        del geoms
    except Exception:  # a bad mesh voxelizes to nothing — keep going
        return node_id, empty, empty, None, None
    opaque = np.unique(np.concatenate(opaque_parts)) if opaque_parts else empty
    glass = np.unique(np.concatenate(glass_parts)) if glass_parts else empty
    return node_id, opaque, glass, lo, hi


def _iter_voxelized(
    ids: list[str], raw_dir: Path, pitch: float, workers: int
):  # noqa: ANN201 - yields _voxelize_object_task results
    """Yield per-object voxelization results, serially (`workers <= 1`) or from
    a process pool. The pool uses the SPAWN context: stage jobs run off worker
    threads (asyncio.to_thread), where forking risks deadlock; spawn costs one
    interpreter+import per worker (~seconds), amortized over a minutes-long
    mesh pass. Object-level parallelism is the right grain — the loop is
    embarrassingly parallel and each task is itself vectorized numpy."""
    if workers <= 1 or len(ids) == 1:
        for node_id in ids:
            yield _voxelize_object_task(node_id, str(raw_dir / f"{node_id}.glb"), pitch)
        return
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    try:
        ctx = multiprocessing.get_context("spawn")
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
    except Exception:  # sandboxed/limited environments — degrade, don't die
        logging.getLogger(__name__).warning(
            "stage2: process pool unavailable — voxelizing serially"
        )
        for node_id in ids:
            yield _voxelize_object_task(node_id, str(raw_dir / f"{node_id}.glb"), pitch)
        return
    with pool:
        futures = [
            pool.submit(
                _voxelize_object_task, node_id, str(raw_dir / f"{node_id}.glb"), pitch
            )
            for node_id in ids
        ]
        for fut in as_completed(futures):
            yield fut.result()


def _boundary_quads(mask: np.ndarray) -> np.ndarray:
    """Surface-extract a boolean voxel volume into run-merged face quads —
    (Q,5) int32 `[x, y, z, face, run]`. A face survives only where the
    neighbour on that side leaves the mask (interior faces culled), so the
    quads tile exactly the boundary of each connected region — the merged,
    volumetric look. Contiguous faces are merged into RUNS along one in-plane
    axis (z for x/y faces, y for z faces): fully vectorized 1-D greedy meshing
    (flat walls/floors collapse into long strips; staircase surfaces stay per
    cell). `face` = axis*2 (+side, plane at the cell's max corner on that
    axis) or axis*2+1 (−side, at the min corner)."""
    parts: list[np.ndarray] = []
    for axis in range(3):
        for neg in (0, 1):
            exposed = mask.copy()
            dst = [slice(None)] * 3
            src = [slice(None)] * 3
            if neg == 0:  # +side face: exposed unless the +axis neighbour is set
                dst[axis], src[axis] = slice(None, -1), slice(1, None)
            else:  # −side face: exposed unless the −axis neighbour is set
                dst[axis], src[axis] = slice(1, None), slice(None, -1)
            exposed[tuple(dst)] &= ~mask[tuple(src)]
            if not exposed.any():
                continue
            run_axis = 1 if axis == 2 else 2
            moved = np.moveaxis(exposed, run_axis, 2)  # run axis last
            d0, d1, run_len = moved.shape
            rows = moved.reshape(-1, run_len)
            # Run extraction: pad each row, diff — +1 marks a run start, −1 one
            # past its end. argwhere scans row-major, so starts/ends pair 1:1.
            padded = np.zeros((rows.shape[0], run_len + 2), dtype=np.int8)
            padded[:, 1:-1] = rows
            d = np.diff(padded, axis=1)
            starts = np.argwhere(d == 1)
            ends = np.argwhere(d == -1)
            if not len(starts):
                continue
            row_i, col0 = starts[:, 0], starts[:, 1]
            runs = ends[:, 1] - col0
            i0, i1 = np.divmod(row_i, d1)
            # Invert the moveaxis: moved coords (i0, i1, col0) → grid (x,y,z).
            if run_axis == 2:
                x, y, z = i0, i1, col0
            else:  # axis == 2 faces run along y: moved order is (x, z, y)
                x, y, z = i0, col0, i1
            quad = np.empty((len(runs), 5), dtype=np.int32)
            quad[:, 0], quad[:, 1], quad[:, 2] = x, y, z
            quad[:, 3] = axis * 2 + neg
            quad[:, 4] = runs
            parts.append(quad)
    if not parts:
        return np.zeros((0, 5), dtype=np.int32)
    return np.concatenate(parts, axis=0)


def _pack_quads(quads: np.ndarray) -> bytes:
    """(Q,5) int32 quads → the SVX3 10-byte wire records (`_QUAD_DTYPE`)."""
    rec = np.zeros(len(quads), dtype=_QUAD_DTYPE)
    rec["x"], rec["y"], rec["z"] = quads[:, 0], quads[:, 1], quads[:, 2]
    rec["f"] = quads[:, 3]
    rec["r"] = quads[:, 4]
    return rec.tobytes()


def _viz_streamed(
    occ_lin: np.ndarray,
    empty_mm: np.ndarray,
    d2_mm: np.ndarray,
    dims: np.ndarray,
    pitch: float,
    thresholds: list[float],
    tick: Callable[[], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, np.ndarray, int]]]:
    """All SVX3 boundary extractions (every clearance shell + cover + garbage)
    in ONE x-slab sweep instead of a full-grid pass per class. Exactness and
    order both survive slab chunking: face exposure looks one cell along the
    face axis (a one-plane halo covers the x-faces), runs merge along z (x/y
    faces) or y (z faces) — never along x — so no run crosses a slab seam, and
    `_boundary_quads` emits face ids in ascending order with row-major (x-outer)
    rows, so regrouping per-face across ascending slabs reproduces the
    whole-grid quad order byte for byte. Returns (cover, garbage, shells) in
    the shapes the pack/write step already consumes."""
    nx, ny, nz = (int(v) for v in dims)
    step = _slab_planes(ny, nz)
    n_cat = len(thresholds) + 2  # shells…, cover, garbage
    face_parts: list[list[list[np.ndarray]]] = [
        [[] for _ in range(6)] for _ in range(n_cat)
    ]
    cells = [0] * len(thresholds)

    for x0 in range(0, nx, step):
        x1 = min(x0 + step, nx)
        sx = x1 - x0
        h0, h1 = max(x0 - 1, 0), min(x1 + 1, nx)
        lo_pad = x0 - h0  # 1 when a left halo plane exists
        occ_h = _occ_slab(occ_lin, h0, h1, ny, nz)
        emp_h = np.asarray(empty_mm[h0:h1])
        cl_h = _clearance_chunk(np.asarray(d2_mm[h0:h1]), pitch)

        def padded(mask_h: np.ndarray) -> np.ndarray:
            """Slab core in [1, sx] with halo planes (or grid-edge False)."""
            p = np.zeros((sx + 2, ny, nz), dtype=bool)
            p[1 : 1 + sx] = mask_h[lo_pad : lo_pad + sx]
            if lo_pad:
                p[0] = mask_h[0]
            if h1 > x1:
                p[sx + 1] = mask_h[-1]
            return p

        def emit(cat: int, mask_h: np.ndarray) -> None:
            q = _boundary_quads(padded(mask_h))
            if len(q):
                q = q[(q[:, 0] >= 1) & (q[:, 0] <= sx)]  # core cells only
            if not len(q):
                return
            q[:, 0] += x0 - 1  # padded → grid x
            b = np.searchsorted(q[:, 3], np.arange(7))  # face ids ascend
            for f in range(6):
                if b[f + 1] > b[f]:
                    face_parts[cat][f].append(q[b[f] : b[f + 1]])

        for ti, t in enumerate(thresholds):
            m_h = emp_h & (cl_h >= t - _CLEARANCE_EPS)
            cells[ti] += int(m_h[lo_pad : lo_pad + sx].sum())
            emit(ti, m_h)
        emit(len(thresholds), occ_h)
        emit(len(thresholds) + 1, ~occ_h & ~emp_h)
        if tick is not None:
            tick()

    def gather(cat: int) -> np.ndarray:
        parts = [a for f in range(6) for a in face_parts[cat][f]]
        return (
            np.concatenate(parts, axis=0) if parts else np.zeros((0, 5), np.int32)
        )

    shells = [(t, gather(ti), cells[ti]) for ti, t in enumerate(thresholds)]
    return gather(len(thresholds)), gather(len(thresholds) + 1), shells


def _shell_thresholds(pitch: float, clearance_m: float, cl_max: float) -> list[float]:
    """The free-shell ladder: voxel multiples (`_SHELL_STEPS`) up to the scene's
    clearance ceiling, with the BAKED threshold always included (so the default
    shell equals the FREE mask stage 4 plans against). Sorted, deduped."""
    top = max(cl_max, pitch) + 1e-9
    ts = {round(pitch * s, 4) for s in _SHELL_STEPS if pitch * s <= top}
    ts.add(round(float(clearance_m), 4))
    return sorted(t for t in ts if t > 0)


# ---------------------------------------------------------------------------
# Memory-aware worker sizing. Each mesh-pass worker is a fresh SPAWN interpreter
# (numpy + scipy + trimesh re-imported, ~0.3-0.5 GB RSS each) that then holds a
# decoded mesh plus its float64 voxel scratch — so peak RAM is `workers ×
# per-object`, and a fixed `min(cpu, 8)` OOMs small boxes and memory-capped
# containers. The auto path (`workers == 0`) sizes the pool to a fraction of the
# ACTUALLY usable memory, and degrades to the static cap when memory can't be
# probed. Worker count never changes the output (absolute-lattice keys merge
# order-independently), so this is purely a resource guard.
_MEM_BUDGET_FRACTION = 0.6          # share of usable RAM the mesh pass may claim
_WORKER_BASE_BYTES = 600 * 1024**2  # spawn interpreter + numpy/scipy/trimesh + slack
_WORKER_FILE_MULT = 16              # decoded mesh + voxel scratch, per GLB byte
_MAX_AUTO_WORKERS = 8               # ceiling even on huge-RAM, many-core hosts


def _cgroup_available_bytes() -> int | None:
    """Bytes left under a Linux cgroup memory cap (v2 then v1), or None when
    unconstrained / not on Linux. psutil and sysconf report the HOST, so a
    container's real ceiling is only visible here — callers `min()` it in."""
    for limit_f, usage_f in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
         "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            raw = Path(limit_f).read_text().strip()
            if raw == "max":
                continue
            limit = int(raw)
            if limit >= 1 << 62:  # v1 "unlimited" sentinel — treat as no cap
                continue
            usage = int(Path(usage_f).read_text().strip())
            return max(0, limit - usage)
        except (OSError, ValueError):
            continue
    return None


def _windows_available_bytes() -> int | None:
    """Available physical bytes via `GlobalMemoryStatusEx.ullAvailPhys` — the
    stdlib-only path where psutil is absent (as on the splat venv)."""
    if os.name != "nt":
        return None
    import ctypes

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        stat = _MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullAvailPhys)
    except Exception:
        return None
    return None


def _posix_available_bytes() -> int | None:
    """Available physical bytes via `SC_AVPHYS_PAGES` (Linux / most Unix); None
    where the name is unsupported (macOS, Windows)."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    except (ValueError, AttributeError, OSError):
        return None


def _available_memory_bytes() -> int | None:
    """Best-effort USABLE memory in bytes: the MIN of every signal we can read
    (psutil, POSIX sysconf, the Windows API, a Linux cgroup cap), so a container
    limit wins over host RAM. None when nothing is detectable — the caller then
    keeps its static worker cap rather than guessing."""
    vals: list[int] = []
    try:
        import psutil  # type: ignore  # optional; never a hard dependency

        vals.append(int(psutil.virtual_memory().available))
    except Exception:
        pass
    for probe in (
        _posix_available_bytes(),
        _windows_available_bytes(),
        _cgroup_available_bytes(),
    ):
        if probe is not None and probe > 0:
            vals.append(probe)
    return min(vals) if vals else None


def _auto_worker_count(ids: list[str], raw_dir: Path, hard_cap: int) -> int:
    """Mesh-pass worker count sized so `workers × per-object peak` fits the
    memory budget. Per-object peak is estimated from the LARGEST placed GLB
    (decode + voxel scratch scale with it) over the fixed spawn-import RSS floor.
    Clamped to [1, min(hard_cap, #objects)]; falls back to `hard_cap` when
    memory or the files can't be read (never worse than the old static cap)."""
    ceiling = max(1, min(hard_cap, len(ids)))
    if ceiling <= 1:
        return ceiling
    avail = _available_memory_bytes()
    if avail is None:
        return ceiling
    try:
        largest = max((raw_dir / f"{i}.glb").stat().st_size for i in ids)
    except OSError:
        return ceiling
    per_worker = _WORKER_BASE_BYTES + _WORKER_FILE_MULT * largest
    fit = int(avail * _MEM_BUDGET_FRACTION) // per_worker
    n = int(max(1, min(ceiling, fit)))
    if n < ceiling:
        logging.getLogger(__name__).info(
            "stage2: mesh pass using %d/%d workers (~%.1f GB usable, ~%.0f MB/worker est.)",
            n, ceiling, avail / 1024**3, per_worker / 1024**2,
        )
    return n


# --- streamed exact clearance (squared EDT in integer voxel units) -------------
# The exact Euclidean distance transform is SEPARABLE: with e²(x', y, z) the 2-D
# squared EDT of plane x' (over (y, z)),
#
#     d²(x, y, z) = min over x' of [ (x - x')² + e²(x', y, z) ]
#
# so the field is built in two streamed phases sharing ONE int32 scratch file:
# phase 1 computes each plane's 2-D squared EDT (the `edt` package; its float32
# results are exact integers while the plane diagonal² < 2^24) during an x-slab
# sweep; phase 2 runs the remaining 1-D minimization along x (`_parabola_pass`,
# exact int64 arithmetic) over column bundles read back from the scratch. All
# values are exact integer squared distances — deterministic for any slab or
# bundle partition — and metres are derived per chunk as
# float32(pitch · sqrt(d²)). Cover cells carry d² = 0, the grid edge is not a
# cover source (matches `black_border=False`), and `pitch` is isotropic.


def _dist2_planes(
    occ_lin: np.ndarray,
    dims: np.ndarray,
    scratch: Path,
    workers: int = 1,
    tick: Callable[[], None] | None = None,
) -> None:
    """Phase 1: per-plane 2-D squared EDT → int32 scratch memmap. Planes with
    no cover at all store −1 (no in-plane site; resolved by phase 2). `workers`
    threads the C++ transform — integer-exact results either way."""
    # Imported at call time (not module load) so read-only users of stage2 —
    # grid loading, FreeSpace, Stage 4 — don't need `edt`; only building a grid does.
    import edt

    nx, ny, nz = (int(v) for v in dims)
    if ny * ny + nz * nz >= 1 << 24 or nx * nx + ny * ny + nz * nz >= 1 << 31:
        raise RuntimeError(
            f"grid {nx}x{ny}x{nz} exceeds the exact-EDT integer range — "
            "use a coarser pitch"
        )
    mm = np.lib.format.open_memmap(
        scratch, mode="w+", dtype=np.int32, shape=(nx, ny, nz)
    )
    step = _slab_planes(ny, nz)
    for x0 in range(0, nx, step):
        x1 = min(x0 + step, nx)
        occ = _occ_slab(occ_lin, x0, x1, ny, nz)
        out = np.empty((x1 - x0, ny, nz), dtype=np.int32)
        for i in range(x1 - x0):
            if occ[i].any():
                out[i] = edt.edtsq(
                    ~occ[i], black_border=False, parallel=max(1, workers)
                ).astype(np.int32)
            else:
                out[i] = -1
        mm[x0:x1] = out
        if tick is not None:
            tick()
    mm.flush()
    del mm


def _parabola_pass(g: np.ndarray) -> np.ndarray:
    """Exact 1-D lower-envelope pass of the squared EDT (Felzenszwalb &
    Huttenlocher), vectorized ACROSS columns: `g` is (n, C) int64 per-column
    parabola costs (`_NO_SITE` marks positions with no site) and the result is
    f[x] = min over q of (g[q] + (x − q)²). Each column keeps its own site
    stack in shared arrays; the pop loop runs while ANY column still pops
    (amortized O(1) per push, as in the scalar algorithm). Envelope boundaries
    are compared as int64 rationals (cross-multiplied), so results are exact
    integers — no float ties, identical for any column batching."""
    n, c_cnt = g.shape
    cols = np.arange(c_cnt)
    v = np.zeros((n, c_cnt), dtype=np.int64)        # stacked site positions
    gv = np.full((n, c_cnt), _NO_SITE, np.int64)    # stacked site costs
    num = np.zeros((n + 1, c_cnt), dtype=np.int64)  # left boundary of site k,
    den = np.ones((n + 1, c_cnt), dtype=np.int64)   # as a rational num/den
    top = np.full(c_cnt, -1, dtype=np.int64)        # stack depth − 1
    neg_inf = np.int64(-1) << 40                    # below any real boundary

    for q in range(n):
        gq = g[q]
        has = gq < _NO_SITE
        if not has.any():
            continue
        # Pop stacked sites the new parabola dominates: intersection abscissa
        # s(q, top) ≤ the top site's own left boundary.
        while True:
            act = has & (top >= 0)
            if not act.any():
                break
            ci = cols[act]
            ti = top[act]
            vt = v[ti, ci]
            s_num = (gq[act] + q * q) - (gv[ti, ci] + vt * vt)
            s_den = 2 * (q - vt)  # > 0: sites are pushed in ascending order
            pop = s_num * den[ti, ci] <= num[ti, ci] * s_den
            if not pop.any():
                break
            top[ci[pop]] -= 1
        # Push site q on every column that has it.
        ci = cols[has]
        ti = top[has] + 1
        prev = np.maximum(ti - 1, 0)
        vt = v[prev, ci]
        s_num = np.where(
            ti > 0, (gq[has] + q * q) - (gv[prev, ci] + vt * vt), neg_inf
        )
        s_den = np.where(ti > 0, 2 * (q - vt), np.int64(1))
        v[ti, ci] = q
        gv[ti, ci] = gq[has]
        num[ti, ci] = s_num
        den[ti, ci] = s_den
        top[has] = ti

    f = np.full((n, c_cnt), _NO_SITE, np.int64)
    k = np.zeros(c_cnt, dtype=np.int64)
    for x in range(n):
        while True:  # advance past boundaries left of x: z[k+1] < x
            adv = (k < top) & (num[k + 1, cols] < x * den[k + 1, cols])
            if not adv.any():
                break
            k[adv] += 1
        vk = v[k, cols]
        f[x] = np.where(top >= 0, gv[k, cols] + (x - vk) * (x - vk), _NO_SITE)
    return f


def _dist2_xpass(
    dims: np.ndarray, scratch: Path, tick: Callable[[], None] | None = None
) -> None:
    """Phase 2: finish the squared EDT along x, in place in the scratch —
    bundles of full-x column blocks stream through `_parabola_pass`."""
    nx, ny, nz = (int(v) for v in dims)
    mm = np.lib.format.open_memmap(scratch, mode="r+")
    # ~6 int64 arrays of (nx, rows·nz) live during a bundle.
    rows = max(1, int(_SLAB_TARGET_BYTES // max(1, nx * nz * 8 * 6)))
    for y0 in range(0, ny, rows):
        y1 = min(y0 + rows, ny)
        g = mm[:, y0:y1, :].reshape(nx, -1).astype(np.int64)
        g[g < 0] = _NO_SITE  # planes without cover contribute no site
        f = _parabola_pass(g)
        # Every column has ≥ 1 site (the grid has cover somewhere), so the
        # minimum is always finite and fits int32 (guarded in _dist2_planes).
        mm[:, y0:y1, :] = f.reshape(nx, y1 - y0, nz).astype(np.int32)
        if tick is not None:
            tick()
    mm.flush()
    del mm


def _clearance_chunk(d2: np.ndarray, pitch: float) -> np.ndarray:
    """Integer squared distances → metres, float32: the ONE conversion every
    consumer of the field shares (stats, shells, the stored float16)."""
    return (np.sqrt(d2.astype(np.float64)) * pitch).astype(np.float32)


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
    """Voxelize the cell's placed meshes into the single uniform occupancy grid
    (in parallel across objects — see `_iter_voxelized`), classify empty vs
    garbage with the two-phase fill, run the clearance pass (EDT + the baked
    FREE threshold), and write `freespace.npz` (to `out_path`) + the SVX3
    `voxels.bin` viz pack + the `.empty.npy` sidecar (beside it). Returns a
    summary (grid dims, counts, fill + clearance stats). Output is
    byte-identical for any worker count and any slab partition."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"placed-mesh dir not found: {raw_dir}")
    ids = placed_object_ids(raw_dir)
    if not ids:
        raise FileNotFoundError(f"no placed meshes in {raw_dir}")
    pitch = float(np.clip(params.pitch, *_PITCH_CLAMP))
    margin = float(max(0.0, params.margin))
    clearance_m = float(max(0.0, params.clearance))
    workers = (
        params.workers if params.workers > 0
        else _auto_worker_count(ids, raw_dir, min(os.cpu_count() or 1, _MAX_AUTO_WORKERS))
    )

    total = len(ids)
    if progress is not None:
        progress(0, total, "voxelize")

    # The MESH PASS — the stage's long pole, parallelized per object: voxelize
    # each surface EXACTLY (every cell a triangle touches, via _voxelize_geom)
    # into an ABSOLUTE integer lattice, keeping only the UNIQUE occupied cells,
    # and record each object's mesh-derived world AABB — the boxes the fill's
    # seeding rules run on (actual placed geometry, immune to divider bbox
    # drift / unmeshed leaves). BLEND/MASK surfaces are classified per piece
    # into OPAQUE vs GLASS cells (glass occupies space but won't block sight).
    # The grid origin (which needs the AABB) is resolved afterwards; absolute-
    # lattice keys are origin-independent, which is exactly what makes the
    # merge order-independent and the parallel output byte-identical.
    opaque_parts: list[np.ndarray] = []
    glass_parts: list[np.ndarray] = []
    obj_lo: dict[str, np.ndarray] = {}
    obj_hi: dict[str, np.ndarray] = {}
    lo = np.array([np.inf, np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf, -np.inf])
    for done, (node_id, op_keys, gl_keys, olo, ohi) in enumerate(
        _iter_voxelized(ids, raw_dir, pitch, workers), start=1
    ):
        if len(op_keys):
            opaque_parts.append(op_keys)
        if len(gl_keys):
            glass_parts.append(gl_keys)
        if olo is not None and ohi is not None:
            lo, hi = np.minimum(lo, olo), np.maximum(hi, ohi)
            obj_lo[node_id], obj_hi[node_id] = olo, ohi
        if progress is not None:
            # Constant "voxelize" step (not the node id) so the caller can meter an
            # objects/second rate; done/total already says which object. Parallel
            # completion order makes a per-id label noise anyway.
            progress(done, total, "voxelize")

    if (not opaque_parts and not glass_parts) or not np.isfinite(lo).all():
        raise RuntimeError("no surface voxelized (every mesh failed or was empty)")
    if progress is not None:
        progress(0, 0, "reduce")  # merge keys → grid + occupancy (single pass)
    empty_keys = np.zeros(0, dtype=np.int64)
    opaque_keys = (
        np.unique(np.concatenate(opaque_parts)) if opaque_parts else empty_keys
    )
    glass_keys = (
        np.unique(np.concatenate(glass_parts)) if glass_parts else empty_keys
    )
    del opaque_parts, glass_parts  # the merged keys supersede the partials

    imin, dims = _grid_dims(lo, hi, margin, pitch)
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    origin = imin.astype(np.float64) * pitch

    def to_lin(keys: np.ndarray) -> np.ndarray:
        """Absolute lattice keys → sorted unique local linear indices (drops
        out-of-grid cells)."""
        v = _abs_decode(keys) - imin
        v = v[np.all((v >= 0) & (v < dims), axis=1)]
        return np.unique((v[:, 0] * ny + v[:, 1]) * nz + v[:, 2])

    opaque_lin = to_lin(opaque_keys)
    # A cell with BOTH classes (e.g. window frame + pane) occludes: opaque wins,
    # keeping the two stored sets disjoint (glass = transmissive-only cells).
    glass_lin = np.setdiff1d(to_lin(glass_keys), opaque_lin, assume_unique=True)
    occ_lin = np.union1d(opaque_lin, glass_lin)
    del opaque_keys, glass_keys, opaque_lin  # occ_lin/glass_lin carry it forward

    # Per-object grid-index AABBs (all cells the mesh AABB touches — a superset
    # of the object's own cover cells, so its ring is always strictly outside).
    boxes: list[tuple[str, np.ndarray, np.ndarray]] = []
    for node_id in ids:
        if node_id not in obj_lo:
            continue
        vlo = np.floor(obj_lo[node_id] / pitch).astype(np.int64) - imin
        vhi = np.floor(obj_hi[node_id] / pitch).astype(np.int64) - imin
        boxes.append(
            (node_id, np.clip(vlo, 0, dims - 1), np.clip(vhi, 0, dims - 1))
        )

    # Streamed grid phases (module docstring): the dense fields live in
    # disk-backed scratch files beside the outputs, processed in x-slabs —
    # nothing grid-sized is ever resident. The empty scratch is PROMOTED to the
    # `.empty.npy` sidecar Stage 3 memory-maps, so its first sampler run no
    # longer decompresses and rewrites the mask itself.
    #
    # Scratch names carry a per-invocation token: builds for the same cell can
    # OVERLAP (a revert's task.cancel() can't stop a build already running in a
    # worker thread, and a fresh POST then starts another), and with fixed
    # names the first build's cleanup deletes the scratches out from under the
    # survivor — a mid-build FileNotFoundError. Unique names make overlap
    # harmless: each build only ever touches (and finally deletes) its own
    # scratches, while the age-gated sweep below reclaims the litter of builds
    # killed too hard for their cleanup to run.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = out_path.with_name(out_path.name + _EMPTY_SIDECAR_SUFFIX)
    token = f"{os.getpid():x}-{int(time.time() * 1000):x}"
    empty_tmp = out_path.with_name(f"{out_path.name}.empty.{token}.tmp.npy")
    dist2_tmp = out_path.with_name(f"{out_path.name}.dist2.{token}.tmp.npy")
    f16_tmp = out_path.with_name(f"{out_path.name}.cl16.{token}.tmp.npy")
    tmp_npz = out_path.with_name(f"{out_path.name}.{token}.tmp.npz")
    now = time.time()
    for stale in out_path.parent.glob(out_path.name + ".*.tmp.np*"):
        try:
            if now - stale.stat().st_mtime > _SCRATCH_TTL_S:
                stale.unlink()
        except OSError:
            pass
    step = _slab_planes(ny, nz)
    n_slabs = (nx + step - 1) // step

    def _stepper(total: int, name: str) -> Callable[[], None]:
        done = 0
        if progress is not None:
            progress(0, total, name)

        def tick() -> None:
            nonlocal done
            done += 1
            if progress is not None:
                progress(done, total, name)

        return tick

    try:
        # THE FILL — streamed two-pass labeling (a tick per slab per sweep).
        rescued_cells, fill_stats = _fill_streamed(
            occ_lin, dims, boxes, empty_tmp, _stepper(2 * n_slabs, "fill")
        )

        # THE CLEARANCE PASS — exact squared EDT into the int32 scratch:
        # per-plane 2-D transforms (a tick per slab), then the 1-D x-pass over
        # column bundles (a tick per bundle). FREE stays a pure derivation
        # (empty ∧ clearance ≥ clearance_m), re-bakeable via `apply_clearance`.
        bundle_rows = max(1, int(_SLAB_TARGET_BYTES // max(1, nx * nz * 8 * 6)))
        tick = _stepper(n_slabs + (ny + bundle_rows - 1) // bundle_rows, "clearance")
        _dist2_planes(occ_lin, dims, dist2_tmp, workers, tick)
        _dist2_xpass(dims, dist2_tmp, tick)

        # Windows note: every memmap handle (and every VIEW of one) below is
        # scoped inside a helper or deleted before the file is renamed/unlinked
        # — a surviving view pins the file and the promote/cleanup would fail.

        def _empty_stats() -> tuple[int, int, float, float]:
            """(empty_count, free_count, cl_max, cl_mean) — FREE count +
            clearance stats over EMPTY cells, flat-chunked off the memmaps:
            three scalars, never a grid-sized copy."""
            emp = np.load(empty_tmp, mmap_mode="r")
            d2 = np.load(dist2_tmp, mmap_mode="r")
            n_empty = int(emp.sum())
            if not n_empty:
                # `pitch` keeps the shell ladder non-empty; the summary reports 0.
                return 0, 0, pitch, 0.0
            thr = clearance_m - _CLEARANCE_EPS
            n_free = 0
            mx = 0.0
            acc = 0.0
            flat_d2 = d2.reshape(-1)
            flat_emp = emp.reshape(-1)
            for i in range(0, flat_d2.size, _CHUNK_CELLS):
                ce = _clearance_chunk(
                    flat_d2[i : i + _CHUNK_CELLS][flat_emp[i : i + _CHUNK_CELLS]],
                    pitch,
                )
                if ce.size:
                    n_free += int((ce >= thr).sum())
                    mx = max(mx, float(ce.max()))
                    acc += float(ce.sum(dtype=np.float64))
            return n_empty, n_free, mx, acc / n_empty

        empty_count, free_count, cl_max, cl_mean = _empty_stats()

        # Persist the reusable grid (float16 clearance halves the file at
        # ~0.05% relative error — comparisons carry _CLEARANCE_EPS, so
        # exact-lattice thresholds survive the quantization). Both grid-sized
        # entries are MEMMAPS: np.savez_compressed streams each into the zip in
        # buffered chunks, so the write holds no grid-sized array either.
        if progress is not None:
            progress(0, 0, "write")

        def _bake_f16() -> None:
            d2 = np.load(dist2_tmp, mmap_mode="r")
            cl16 = np.lib.format.open_memmap(
                f16_tmp, mode="w+", dtype=np.float16, shape=(nx, ny, nz)
            )
            flat16 = cl16.reshape(-1)
            flat_d2 = d2.reshape(-1)
            for i in range(0, flat_d2.size, _CHUNK_CELLS):
                flat16[i : i + _CHUNK_CELLS] = _clearance_chunk(
                    flat_d2[i : i + _CHUNK_CELLS], pitch
                ).astype(np.float16)
            cl16.flush()

        _bake_f16()
        cl16 = np.load(f16_tmp, mmap_mode="r")
        empty_mm = np.load(empty_tmp, mmap_mode="r")
        np.savez_compressed(
            tmp_npz,
            origin=origin.astype(np.float64),
            pitch=np.float64(pitch),
            dims=np.array([nx, ny, nz], dtype=np.int64),
            occ_lin=occ_lin.astype(np.int64),
            occ_lin_glass=glass_lin.astype(np.int64),
            clearance=cl16,
            empty=empty_mm,
            clearance_m=np.float64(clearance_m),
        )
        del cl16, empty_mm
        tmp_npz.replace(out_path)
        # Promote the empty scratch to the Stage-3 sidecar AFTER the npz lands,
        # and touch it: rename keeps the content mtime, and Stage 3 rebuilds a
        # sidecar that reads older than its npz.
        empty_tmp.replace(sidecar)
        os.utime(sidecar)

        # SVX3 viz pack (module docstring): every class extracted in ONE slab
        # sweep — the tail's slowest pass touches the grid once, not per class.
        thresholds = _shell_thresholds(pitch, clearance_m, cl_max)
        empty_mm = np.load(sidecar, mmap_mode="r")
        d2_mm = np.load(dist2_tmp, mmap_mode="r")
        cover_q, garbage_q, shells = _viz_streamed(
            occ_lin, empty_mm, d2_mm, dims, pitch, thresholds,
            _stepper(n_slabs, "viz"),
        )
        del empty_mm, d2_mm
    finally:
        # Scratches are build-local (unique names). On failure a leaked memmap
        # handle can pin one on Windows — best-effort: the TTL sweep of the
        # next build reclaims anything left behind.
        for p in (dist2_tmp, f16_tmp, empty_tmp, tmp_npz):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    viz_path = out_path.with_name(VOXELS_NAME)
    tmp_viz = viz_path.with_suffix(viz_path.suffix + ".tmp")
    with tmp_viz.open("wb") as f:
        f.write(_VIZ_HEADER.pack(_VIZ_MAGIC, len(cover_q), len(garbage_q), len(shells)))
        f.write(_pack_quads(cover_q))
        f.write(_pack_quads(garbage_q))
        for t, q, cells in shells:
            f.write(_SHELL_HEADER.pack(t, len(q), cells))
            f.write(_pack_quads(q))
    tmp_viz.replace(viz_path)

    return {
        "run": run,
        "slot": slot,
        "model": model,
        "pitch": pitch,
        "dims": [nx, ny, nz],
        "origin": [round(v, 5) for v in origin.tolist()],
        "scene_aabb": {"min": lo.tolist(), "max": hi.tolist()},
        "solid_voxels": int(occ_lin.size),
        "glass_voxels": int(glass_lin.size),
        "empty_voxels": empty_count,
        "free_voxels": free_count,
        "garbage_voxels": int(nx * ny * nz - occ_lin.size) - empty_count,
        "rescued_voxels": rescued_cells,
        "fill": fill_stats,
        "viz": {
            "cover_quads": int(len(cover_q)),
            "garbage_quads": int(len(garbage_q)),
            "shells": [
                {"clearance": round(t, 4), "quads": int(len(q)), "cells": c}
                for t, q, cells in shells
                for c in (cells,)
            ],
        },
        "clearance_max": cl_max if empty_count else 0.0,
        "clearance_mean": round(cl_mean, 4) if empty_count else 0.0,
        "params": params.as_summary(),
        "bytes": viz_path.stat().st_size,
        "grid_bytes": out_path.stat().st_size,
    }
