"""Stage 2 — Free-space voxelizer (surface-band two-tier flood fill).

The shared spatial FOUNDATION of the splat pipeline: discretizes a cell's
composed scene on ONE uniform 3 cm lattice (the fidelity scale, deliberately
scene-size independent — big scenes contain small objects), classifies the air
NEAR SURFACES as COVER (surface), EMPTY (air a viewer could see — ambient +
rescued cavities) or GARBAGE (sealed interiors — object hollows, seams), and
emits the camera CANDIDATE list Stage 4 plans from. Work and storage scale
with SURFACE AREA, never scene volume: air is only ever examined within the
camera-coverage band of a surface, at two resolutions —

  * the FINE SKIN — every 8³-voxel brick within one brick of a cover brick —
    is processed at full 3 cm resolution. All correctness lives here: thin
    walls, seams, seal detection, Stage 3's orient/cull probes (which only
    ever look 1-2 voxels off a surface).
  * the COARSE ZONE — cover-free bricks from the skin out to `coverage`
    metres — is processed per brick. A brick out here provably contains no
    surface (any brick with cover is, by definition, inside the skin), so it
    is a single all-air node: connectivity needs no fine resolution, and thin
    barriers can never be crossed at brick granularity because they only
    exist inside the skin.
  * beyond `coverage`: unexplored, unlabeled, unstored. Cameras are never
    placed there (a camera farther than the band from every surface supplies
    no finest-scale demand), and per the training design far appearance comes
    from close-up references + the Stage-6 LOD ladder, not far cameras.

Nothing downstream recomputes occupancy; that's the point of running this
first. Stage 4 receives its candidate positions directly (each annotated with
its distance to the nearest surface), so no dense free-space volume exists
anywhere in the pipeline.

OCCUPANCY is an EXACT surface voxelization: every cell whose cube a triangle
touches is marked — big triangles are midpoint-split until each piece spans at
most a 3×3×3 cell block, then a separating-axis triangle/cube test decides each
cell. Deterministic and gap-free, so the visibility ray-march can't leak through
pinholes in walls and the flood fill can't escape into sealed interiors. No
winding/closure assumptions, so the non-watertight composed scene is fine.
Pieces STREAM through the test in bounded `_BATCH_PIECES` batches
(`_piece_batches`), each deduped to sorted keys on its own, so a worker's
transient memory tracks the batch — never an object's whole subdivided soup —
and the key set is byte-identical for any batch size. The cover set is stored
SPARSELY (sorted linear indices), so its memory tracks the amount of SURFACE.
Fine occupancy is only ever materialized per 8³ BRICK inside the skin (rebuilt
on demand from the sorted keys, which are x-major and therefore contiguous per
brick x-layer); no dense fine-resolution array over the grid ever exists, in
RAM or on disk.

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

EMPTY vs GARBAGE — the two-phase, two-tier fill. Connectivity is 6-connected
at fine resolution inside the skin (face neighbours only — a diagonal fill
would slip through the corner-touching staircase a thin oblique wall voxelizes
into) and per-brick in the zone; the two tiers stitch at skin-brick faces.
Restricted to the explored band, the resulting components are IDENTICAL to a
whole-grid fine fill's: a zone brick is pure air (internally 6-connected), two
adjacent air bricks always connect (their shared 8×8 face is all air), and a
skin face cell connects to an adjacent air brick exactly as its fine neighbour
would. Labels are assigned once at the end (empty = reached, garbage = the
unreached remainder within the band):

  * PHASE 1 (ambient). Seeds are (a) the grid's boundary shell — the exterior
    margin, always ambient — and (b) every object's AUGMENTED SHELL: the 1-cell
    ring around its mesh-derived AABB, keeping only cells that are unoccupied
    AND outside every other object's AABB. Since meshes (and therefore their
    hollow interiors) lie inside their AABBs, a surviving seed is provably
    ambient air — it can never sit inside another object's sealed hollow. The
    per-voxel exclusion (not per-direction) is what lets flush, mutually
    abutting architecture still seed: a wall's room-facing shell loses only the
    rows inside the floor/ceiling/side-wall boxes, so sealed rooms are seeded
    directly by their own bounding surfaces. Per-object self-seeding is ALSO
    what makes the band cut safe for disconnected content (a swamp's islands,
    a platformer's floating platforms): every object's neighbourhood seeds
    itself, so no path through unexplored open air is ever required. The
    exclusion constrains SEEDS only — the fill itself flows wherever occupancy
    permits (over sofa seats, under arches), or bbox-covered air would wrongly
    read sealed.
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

CANDIDATES (the free-voxel product, emitted directly). Camera positions are
picked from reached air at ~half-metre spacing — in the skin, the first free
fine cell per 2×2×2-brick block; in the zone, the first reached brick's centre
per block — and each is annotated with its distance to the nearest surface
(exact within the skin via a local search; brick-granular in the zone, where
sub-voxel precision cannot matter). The baked `clearance_m` (and the client
slider's re-bakes via `apply_clearance`, now an instant metadata rewrite) is a
FILTER over the annotated list. Stage 4 consumes the list as-is: it computes
no clearance and derives no free volume of its own. Camera standoff beyond
the baked floor stays EMERGENT (the scale ladder demands nothing below a
patch's d_min).

Pure library (like the other stages): takes explicit paths and reads the
meshes AS-IS through `splat.assets.load_geoms` — vanilla glTF via trimesh,
KTX2/Meshopt sets via the in-process decoder (no de-optimization step). On
compressed sets texel data is unavailable, so BLEND/MASK glass classification
falls back to the constant `baseColorFactor` alpha.

Outputs (under a cell's `splat/` dir), all O(surface + candidates):
  * `freespace.npz` — metadata + sparse structures: origin/pitch/dims, the
    sorted fine cover (incl. the glass subset), the sorted skin-brick ids, the
    reached zone-brick ids, the candidate positions + clearance annotations,
    and the baked `clearance_m`. Older layouts (dense fields, mask sidecars)
    are rejected with a clear re-run error rather than carried as compat shims.
  * `freespace.npz.skin.npy` — the per-skin-brick EMPTY bitmasks (512 bits =
    one uint64[8] per brick), the ONLY per-cell payload, memory-mapped by the
    loaders so Stage 3's pool workers share one physical copy.
  * `voxels.bin` — the SVX3 viz pack for the client overlay: VOLUMETRIC
    boundary shells as run-merged exposed-face quads (`_boundary_quads`).
    Cover and garbage are fine-resolution (they live in the skin); the free
    volume is ONE shell at brick resolution (the slider's preview ladder died
    with the dense clearance field — its 'apply' still re-filters candidates).
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
# Uncompressed sidecar holding the per-skin-brick EMPTY bitmasks (uint64
# (S, 8) — 512 bits per brick, raster order within the brick). The one
# per-cell payload; kept out of the npz so loaders memory-map it and Stage 3's
# pool workers share ONE physical copy through the page cache.
_SKIN_SIDECAR_SUFFIX = ".skin.npy"
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
DEFAULT_PITCH = 0.03      # the uniform voxel edge (m) — the FIDELITY scale,
                          # deliberately scene-size independent
# The baked FREE threshold (m): a camera needs at least this much clearance.
# The default equals ONE VOXEL at the default pitch — the most permissive
# meaningful setting: every empty cell is ≥ 1 voxel from cover by definition,
# so every reached cell is a candidate source out of the box, and per-scene
# tightening happens through the client slider's apply (`apply_clearance`,
# now an instant candidate re-filter). For reference, 0.35 was the measured
# knee of the passage-survival curve across the benchmark runs (~2/3 of
# sub-2m facing gaps keep centerline candidates) — a sensible value when
# tightening for production plans.
DEFAULT_CLEARANCE = 0.03
# The camera-coverage band (m): air is explored, labeled and candidate-seeded
# only within this distance of a surface. DERIVED from the scale ladder at the
# Stage-4 defaults — a camera supplies the finest demanded scale of even the
# coarsest patch only within 2·patch_max_spacing·focal_px/finest_px =
# 2·0.30·512/64 ≈ 4.8 m — and rounded up. Beyond it a camera supplies only
# coarse octaves, which near-surface cameras already supply across horizontal
# distance, and far appearance is the Stage-6 LOD ladder's job.
DEFAULT_COVERAGE = 5.0
# Tolerance for `clearance >= threshold` candidate filters: absorbs float32
# annotation rounding so a threshold equal to an exact lattice distance (the
# one-voxel default) includes its own tier.
_CLEARANCE_EPS = 2.5e-4
_PITCH_CLAMP = (0.02, 1.0)
# The two-tier granularities, in fine voxels per axis. A BRICK (8³ = 24 cm at
# default pitch) is the skin/zone unit: cover-containing bricks + their 1-ring
# form the fine skin, so fine treatment always extends ≥ 8 voxels beyond any
# surface (Stage 3 probes need 3). Candidates thin to one per CAND block
# (2×2×2 bricks ≈ 0.5 m — the camera-planning granularity, matching the old
# stage-4 candidate spacing).
_BRICK = 8
_CAND_BRICKS = 2
# Scratch files older than this are swept at build start. Live scratches are
# never this old (phases write continuously); only hard-killed builds — whose
# cleanup never ran — leave older ones behind.
_SCRATCH_TTL_S = 6 * 3600.0
# Exact voxelization: triangles are midpoint-split until every edge spans at most
# this many cells, so each piece's candidate block for the separating-axis
# test is at most 3×3×3 cells (a triangle's extent is bounded by its longest edge).
_SUBDIV_EDGE_CELLS = 2.0
# Mesh-pass batch size, in subdivided PIECES: each batch is subdivided, SAT-
# tested and key-encoded on its own, so a worker's transient working set
# (pieces + overlap temporaries + hit cells, ~0.5-1 KB/piece at peak) stays
# ~100 MB regardless of object size — a big slab/window used to materialize
# its whole multi-million-piece soup at once (~1.4 GB). Purely an evaluation-
# order knob: pieces and their deduped keys are identical for ANY batch size.
_BATCH_PIECES = 1 << 17
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
    """Stage-2 knobs. `pitch` is the uniform fine voxel edge (the fidelity
    scale — scene-size independent); `margin` grows the grid beyond the scene
    AABB so exterior camera vantages exist (Stage 4); `clearance` (m) is the
    baked candidate filter — camera spots at least this far from any surface;
    `coverage` (m) is the camera band — air is explored and candidate-seeded
    only within this distance of a surface (module docstring). `workers`
    parallelizes the per-object mesh pass (0 = auto: sized to fit available
    RAM, capped at min(cores, 16); 1 = serial); the output is byte-identical
    for any value."""

    pitch: float = DEFAULT_PITCH
    margin: float = 1.5
    clearance: float = DEFAULT_CLEARANCE
    coverage: float = DEFAULT_COVERAGE
    workers: int = 0

    def as_summary(self) -> dict[str, Any]:
        return {
            "pitch": self.pitch,
            "margin": self.margin,
            "clearance": self.clearance,
            "coverage": self.coverage,
            "workers": self.workers,
        }


@dataclass(frozen=True)
class FreeSpace:
    """A loaded free-space grid — ONE uniform fine lattice of edge `pitch`,
    represented SPARSELY throughout (module docstring):

      * `occ_lin` — sorted fine linear indices of ALL cover cells (opaque +
        glass); `occ_lin_opaque` the line-of-sight-blocking subset.
      * `skin_lin` + `skin_empty` — the fine skin: sorted brick linear ids
        (bricks of `_BRICK`³ fine cells on the derived `bdims` grid) and each
        brick's 512-bit EMPTY bitmask (uint8 (S, 64), little bit order, raster
        order within the brick). Air near surfaces at full resolution — what
        Stage 3's probes read.
      * `zone_lin` — sorted brick ids of REACHED coarse-zone bricks (pure air
        between the skin and the coverage band edge). Empty at brick
        granularity.
      * `cand_pos` + `cand_clear` — the camera-candidate list (world positions
        + distance-to-nearest-surface annotations). What Stage 4 plans from;
        the baked `clearance_m` filters it at load (`free_candidates`).

    Two cover queries with different jobs: `occupied` answers "is there ANY
    surface here" (physical presence — glass included), `occluding` answers
    "does this cell block sight" (opaque only — glass passes light)."""

    origin: np.ndarray        # (3,) world corner of cell [0,0,0]
    pitch: float              # fine voxel edge (m)
    dims: np.ndarray          # (3,) int64 fine grid dims [nx,ny,nz]
    occ_lin: np.ndarray       # (K,) int64 SORTED fine linear cover indices
    occ_lin_opaque: np.ndarray  # (K2,) int64 SORTED subset that blocks line-of-sight
    skin_lin: np.ndarray      # (S,) int64 SORTED skin-brick linear ids
    skin_empty: np.ndarray    # (S,64) uint8 per-brick EMPTY bitmasks
    zone_lin: np.ndarray      # (Z,) int64 SORTED reached zone-brick ids
    cand_pos: np.ndarray      # (M,3) float32 candidate world positions
    cand_clear: np.ndarray    # (M,) float32 distance to nearest surface (m)
    clearance_m: float        # baked candidate clearance filter (m)

    @property
    def bdims(self) -> np.ndarray:
        """(3,) int64 brick-grid dims (`ceil(dims / _BRICK)`)."""
        return (self.dims + _BRICK - 1) // _BRICK

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

    def _brick_lin(self, cells: np.ndarray) -> np.ndarray:
        """(N,) int64 brick linear ids of (N,3) fine cell indices."""
        b = cells >> 3  # _BRICK == 8
        bd = self.bdims
        return (b[:, 0] * bd[1] + b[:, 1]) * bd[2] + b[:, 2]

    def empty_at(self, points: np.ndarray) -> np.ndarray:
        """Boolean per world point: is it in EMPTY space (ambient air or a
        rescued cavity)? Answered at fine resolution inside the skin (the
        per-brick bitmasks) and at brick resolution in the coarse zone (a
        reached zone brick is all air). This is what distinguishes viewable
        air from a solid's sealed hollow — the signal Stage 3 uses to orient
        normals + cull hidden faces; its probes sit 1-2 voxels off surfaces,
        always inside the skin. Points outside the grid or beyond the
        coverage band read as False."""
        idx, inb = self._bin(points)
        out = np.zeros(len(points), dtype=bool)
        if not inb.any():
            return out
        ii = idx[inb]
        blin = self._brick_lin(ii)
        res = np.zeros(len(ii), dtype=bool)
        if self.skin_lin.size:
            pos = np.searchsorted(self.skin_lin, blin)
            pos_c = np.clip(pos, 0, self.skin_lin.size - 1)
            in_skin = self.skin_lin[pos_c] == blin
            if in_skin.any():
                loc = ii[in_skin] & 7
                bit = (loc[:, 0] * 8 + loc[:, 1]) * 8 + loc[:, 2]
                words = np.asarray(
                    self.skin_empty[pos_c[in_skin], bit >> 3]
                )
                res[in_skin] = (words >> (bit & 7).astype(np.uint8)) & 1 == 1
        else:
            in_skin = np.zeros(len(ii), dtype=bool)
        rest = ~in_skin
        if rest.any() and self.zone_lin.size:
            bl = blin[rest]
            pos = np.clip(np.searchsorted(self.zone_lin, bl), 0, self.zone_lin.size - 1)
            res[rest] = self.zone_lin[pos] == bl
        out[inb] = res
        return out

    def free_candidates(self, spacing: float | None = None) -> np.ndarray:
        """The camera-candidate positions for Stage 4: the stored candidate
        list filtered by the baked clearance (each candidate carries its
        distance to the nearest surface). Candidates were emitted at
        ~`_CAND_BRICKS`-brick spacing (≈ 0.5 m) at build time, so `spacing` is
        accepted for API compatibility but does no further thinning below
        that. Returns (M,3) float32 world points."""
        del spacing  # emission spacing is baked at build time
        keep = self.cand_clear >= (self.clearance_m - _CLEARANCE_EPS)
        return np.ascontiguousarray(self.cand_pos[keep])


def _savez_fast(path: Path, **arrays: np.ndarray) -> None:
    """`np.savez_compressed`-compatible writer at deflate level 1. numpy
    hardcodes level 6, which crawls (~25 MB/s) through the multi-hundred-MB
    sorted cover arrays of foliage scenes for a ~15% size win over level 1;
    level 1 writes ~4× faster and `np.load` reads either unchanged."""
    import zipfile

    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1,
        allowZip64=True,
    ) as zf:
        for name, arr in arrays.items():
            with zf.open(f"{name}.npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(arr))


def _load_skin_sidecar(npz_path: Path, n_bricks: int) -> np.ndarray:
    """Memory-map the per-skin-brick EMPTY bitmask sidecar. mmap keeps it off
    the heap (Stage 3's pool workers share one physical copy); a mmap failure
    with the file present falls back to an in-RAM load; missing/mismatched
    means an old or torn grid — re-run Stage 2."""
    p = npz_path.with_name(npz_path.name + _SKIN_SIDECAR_SUFFIX)
    if not p.is_file():
        raise ValueError(f"{p} missing — re-run Stage 2 (no skin sidecar)")
    try:
        m = np.load(p, mmap_mode="r")
    except Exception:
        m = np.load(p)
    if m.dtype != np.uint8 or m.shape != (n_bricks, 64):
        raise ValueError(
            f"{p} is not the ({n_bricks}, 64) uint8 bitmask this grid expects — "
            "re-run Stage 2"
        )
    return m


_NPZ_REQUIRED = (
    "origin", "pitch", "dims", "occ_lin", "occ_lin_glass",
    "skin_lin", "zone_lin", "cand_pos", "cand_clear", "clearance_m",
)


def load_free_space(path: Path) -> FreeSpace:
    """Load a `freespace.npz` (metadata + sparse structures) plus its skin
    bitmask sidecar (memory-mapped). Older layouts (dense fields, mask
    sidecars, pre-candidate grids) are rejected with a re-run error — no
    compat shims."""
    path = Path(path)
    with np.load(path) as z:
        files = set(z.files)
        if any(k not in files for k in _NPZ_REQUIRED):
            raise ValueError(
                f"{path} is a pre-candidate free-space grid — re-run Stage 2"
            )
        origin = z["origin"].astype(np.float64)
        pitch = float(z["pitch"])
        dims = z["dims"].astype(np.int64)
        occ_lin = z["occ_lin"].astype(np.int64)
        glass = z["occ_lin_glass"].astype(np.int64)
        skin_lin = z["skin_lin"].astype(np.int64)
        zone_lin = z["zone_lin"].astype(np.int64)
        cand_pos = z["cand_pos"].astype(np.float32)
        cand_clear = z["cand_clear"].astype(np.float32)
        clearance_m = float(z["clearance_m"])
    return FreeSpace(
        origin=origin,
        pitch=pitch,
        dims=dims,
        occ_lin=occ_lin,
        # Glass is stored as the disjoint transmissive-only subset, so the
        # occlusion set is simply "all minus glass" (stays sorted).
        occ_lin_opaque=np.setdiff1d(occ_lin, glass, assume_unique=True),
        skin_lin=skin_lin,
        skin_empty=_load_skin_sidecar(path, int(skin_lin.size)),
        zone_lin=zone_lin,
        cand_pos=cand_pos,
        cand_clear=cand_clear,
        clearance_m=clearance_m,
    )


def apply_clearance(path: Path, clearance: float) -> dict[str, Any]:
    """Re-bake the candidate clearance filter of an existing grid — the
    'apply' behind the client's clearance slider. INSTANT: candidates carry
    their distance-to-surface annotations, so only the `clearance_m` scalar
    is rewritten and the filtered count reported. Returns a summary patch
    `{clearance, free_voxels}` (free_voxels = candidates passing) for the
    sidecar."""
    path = Path(path)
    with np.load(path) as z:
        if any(k not in z.files for k in _NPZ_REQUIRED):
            raise ValueError(f"{path} is a pre-candidate grid — re-run Stage 2")
        data = {k: z[k] for k in z.files}
    c = float(clearance)
    data["clearance_m"] = np.float64(c)
    count = int(
        (data["cand_clear"].astype(np.float32) >= c - _CLEARANCE_EPS).sum()
    )
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    _savez_fast(tmp, **data)
    tmp.replace(path)
    return {"clearance": c, "free_voxels": count}


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


def _piece_batches(
    tris: np.ndarray, limit: float, attrs: np.ndarray | None = None
):  # noqa: ANN201 - yields (pieces (P,3,3), piece_attrs (P,3,K) | None)
    """Stream `_subdivide_edges(tris, limit, attrs)` in BOUNDED batches of
    ~`_BATCH_PIECES` pieces, so the mesh pass never materializes an object's
    whole subdivided soup.

    Correctness: the midpoint recursion is per-triangle independent (a split
    reads only the triangle's own vertices) and its termination test only the
    current piece's edges — so subdividing any GROUPING of the inputs, and
    even PRE-SPLITTING an oversized triangle at a coarser limit and resuming,
    replays the identical recursion tree and emits exactly the pieces of the
    one-shot call, just grouped. Consumers dedupe through sorted unique keys,
    so the grouping is invisible in the output: byte-identical for any batch
    size. Only peak memory changes."""
    if not len(tris):
        return
    limit = max(float(limit), 1e-9)

    def est(t: np.ndarray) -> np.ndarray:
        """~2× overestimate of each triangle's leaf-piece count ((longest
        edge / limit)² tracks the recursion's leaf count within a small
        constant). Sizing only — error moves per-batch memory, never output."""
        e = np.stack(
            [t[:, 1] - t[:, 0], t[:, 2] - t[:, 1], t[:, 0] - t[:, 2]], axis=1
        )
        longest = np.linalg.norm(e, axis=2).max(axis=1)
        n = 2.0 * np.ceil(np.maximum(longest, 0.0) / limit) ** 2
        return np.maximum(n, 1.0).astype(np.int64)

    n_est = est(tris)
    if int(n_est.sum()) <= _BATCH_PIECES:  # common small-object fast path
        yield _subdivide_edges(tris, limit, attrs)
        return

    # A single triangle whose estimate alone exceeds the budget (a huge
    # ground/wall quad) is pre-split at a coarser limit — the same recursion,
    # paused (children's edges ≤ limit·√(budget/2) ⇒ each child re-estimates
    # within the budget) — then its children batch like ordinary inputs.
    big = n_est > _BATCH_PIECES
    if big.any():
        coarse = limit * float(np.sqrt(_BATCH_PIECES / 2.0))
        bt, ba = _subdivide_edges(
            tris[big], coarse, attrs[big] if attrs is not None else None
        )
        tris = np.concatenate([tris[~big], bt], axis=0)
        if attrs is not None:
            attrs = np.concatenate([attrs[~big], ba], axis=0)
        n_est = est(tris)

    # Contiguous groups whose estimates sum to ≲ the budget (a group may run
    # one entry past the boundary, so ≤ ~2× worst case — still batch-sized).
    start = np.cumsum(n_est) - n_est
    group = start // _BATCH_PIECES
    for gid in range(int(group[-1]) + 1):
        m = group == gid
        if m.any():
            yield _subdivide_edges(
                tris[m], limit, attrs[m] if attrs is not None else None
            )


def _tri_cell_overlap(tris: np.ndarray, cells: np.ndarray, pitch: float) -> np.ndarray:
    """Triangle ↔ cell intersection (separating-axis test), vectorized over
    pairs: `tris` (P,3,3) world vertices vs `cells` (P,3) absolute lattice coords
    of each cell's min corner (a closed cube of edge `pitch`). Returns (P,) bool.
    Runs in the INPUT dtype — float32 from the mesh pass: the sources store at
    most float32 (vanilla glTF) or normalized-int16 (quantized sets, ~0.1-1 mm
    steps), so float32's ≤ tens-of-µm decision fuzz at world scale is orders of
    magnitude below the geometry's own quantization, and half the memory
    traffic of the float64 this replaced.

    Two convex shapes overlap unless some axis separates them. The cube's three
    face axes are separated already by construction (candidates come from the
    triangle's own lattice AABB), which leaves: the triangle's plane (tested as
    |signed distance of the cube centre| ≤ the cube's half-diagonal projected
    onto the normal) and the nine edge×axis cross terms (tested as three 2D
    edge-function tests, one per axis projection, each against the cube corner
    most inside that edge — the Schwarz–Seidel / Akenine-Möller formulation).
    A degenerate projection (normal component 0) passes trivially and simply
    constrains nothing; the other projections + the plane test still decide."""
    v = tris - cells.astype(tris.dtype)[:, None, :] * tris.dtype.type(pitch)
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


def _merge_keys(parts: list[np.ndarray]) -> np.ndarray:
    """Sorted unique union of ALREADY-SORTED unique key arrays (empty-safe).
    Concatenate + STABLE sort — numpy's timsort detects the pre-sorted runs
    and effectively k-way MERGES them (~O(N log k)) where `np.unique`'s
    run-blind introsort pays a full O(N log N) — then a linear dedupe. At
    swamp-scale cover counts (10⁸ keys) this is the difference between
    seconds and minutes in the reduce phase."""
    if not parts:
        return np.zeros(0, dtype=np.int64)
    if len(parts) == 1:
        return parts[0]
    merged = np.concatenate(parts)
    merged.sort(kind="stable")
    keep = np.empty(merged.size, dtype=bool)
    keep[0] = True
    np.not_equal(merged[1:], merged[:-1], out=keep[1:])
    return merged[keep]


def _voxelize_surface(tris: np.ndarray, pitch: float) -> np.ndarray:
    """Sorted unique absolute-lattice KEYS of every cell touched by any
    triangle. DETERMINISTIC and complete — every cell a triangle overlaps is
    marked, none are missed — unlike the random point sampling this replaces
    (which left ~e^-oversample of surface cells unmarked: pinholes for rays /
    the flood fill). Pieces stream through the overlap test in `_BATCH_PIECES`
    groups (each deduped on its own), so peak memory tracks the batch, not the
    object; the merged key set is identical for any batch size. float32
    throughout — matching the sources' own precision ceiling (float32 or
    normalized-int16 positions; see `_tri_cell_overlap`)."""
    tris = np.asarray(tris, dtype=np.float32)
    tris = tris[_valid_tri_mask(tris)]
    parts: list[np.ndarray] = []
    for pieces, _ in _piece_batches(tris, _SUBDIV_EDGE_CELLS * pitch):
        cells = _voxelize_cells(pieces, pitch)
        if len(cells):
            parts.append(np.unique(_abs_encode(cells)))
    return _merge_keys(parts)


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
    """`(opaque_keys, glass_keys)` — each sorted unique absolute-lattice int64
    keys — for one geometry. OPAQUE materials take the pure-geometry path (no
    texture reads). BLEND/MASK triangles are subdivided to ~cell-sized pieces
    WITH their UVs, each piece classified by base-color alpha at its centroid,
    and the two groups voxelized separately — so one window mesh yields
    occluding frame cells AND transmissive pane cells at cell granularity. A
    cell containing both classes is resolved to opaque by the caller (opaque
    wins). Pieces stream in `_BATCH_PIECES` groups (see `_piece_batches`), so
    a big mesh never holds its whole subdivided soup in RAM. float32
    throughout — the sources' own precision ceiling (`_tri_cell_overlap`);
    UVs are stored float32 / normalized-uint16 to begin with."""
    tris = np.asarray(geom.triangles, dtype=np.float32)
    nothing = np.zeros(0, dtype=np.int64)
    sampler, cutoff = _geom_alpha_sampler(geom)
    if sampler is None:
        return _voxelize_surface(tris, pitch), nothing

    uv = getattr(getattr(geom, "visual", None), "uv", None)
    if uv is not None and len(uv) == len(geom.vertices):
        uv3 = np.asarray(uv, dtype=np.float32)[np.asarray(geom.faces)]  # (T,3,2)
    else:
        uv3 = np.zeros((len(tris), 3, 2), dtype=np.float32)
    keep = _valid_tri_mask(tris)
    tris, uv3 = tris[keep], uv3[keep]
    op_parts: list[np.ndarray] = []
    gl_parts: list[np.ndarray] = []
    for pieces, uvp in _piece_batches(tris, _SUBDIV_EDGE_CELLS * pitch, uv3):
        alpha = sampler(uvp.mean(axis=1))
        occluding = alpha >= cutoff
        for mask, parts in ((occluding, op_parts), (~occluding, gl_parts)):
            cells = _voxelize_cells(pieces[mask], pitch)
            if len(cells):
                parts.append(np.unique(_abs_encode(cells)))
    return _merge_keys(op_parts), _merge_keys(gl_parts)


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


# --- the two-tier band fill ------------------------------------------------
# Fine resolution exists only inside SKIN bricks (cover bricks + their 1-ring);
# everything else within the coverage band is a single all-air BRICK node.
# The brick grid itself (volume / 8³) is always small enough to hold dense, so
# brick-level classification, distance and connectivity are single scipy calls.

_STRUCT6 = ndimage.generate_binary_structure(3, 1)
# Labels a BATCH of bricks in ONE C call: 6-connectivity inside each brick,
# no connectivity across the batch axis.
_STRUCT6_BATCH = np.zeros((3, 3, 3, 3), dtype=bool)
_STRUCT6_BATCH[1] = _STRUCT6


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


def _layer_bricks(
    occ_lin: np.ndarray,
    bx: int,
    dims: np.ndarray,
    skin_plane: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One brick x-layer's fine occupancy: `(slot, coords, occ4)` where `slot`
    is the (nby, nbz) int32 plane mapping brick (by,bz) → batch index (−1
    outside the skin), `coords` the (S,2) raster-ordered (by, bz) of the
    layer's skin bricks, and `occ4` the (S,8,8,8) bool occupancy — cover cells
    scattered from the x-contiguous slice of the sorted keys, with the
    out-of-grid padding of partial edge bricks marked OCCUPIED so it can never
    read as air. Deterministic: identical for both fill sweeps."""
    nx, ny, nz = (int(v) for v in dims)
    plane = ny * nz
    coords = np.argwhere(skin_plane)  # (S,2) raster (by, bz)
    slot = np.full(skin_plane.shape, -1, dtype=np.int32)
    slot[coords[:, 0], coords[:, 1]] = np.arange(len(coords), dtype=np.int32)
    occ4 = np.zeros((len(coords), _BRICK, _BRICK, _BRICK), dtype=bool)
    x0, x1 = bx * _BRICK, min(bx * _BRICK + _BRICK, nx)
    lo, hi = np.searchsorted(occ_lin, [x0 * plane, x1 * plane])
    seg = occ_lin[lo:hi]
    if seg.size:
        x = seg // plane
        rem = seg % plane
        y, z = rem // nz, rem % nz
        s = slot[y >> 3, z >> 3]  # cover bricks are always skin → s ≥ 0
        occ4[s, (x - x0), y & 7, z & 7] = True
    # Solid-pad the grid's ragged edges (partial bricks) so padding never
    # counts as air in labeling, sizes, or the packed bitmasks.
    if x1 - x0 < _BRICK:
        occ4[:, x1 - x0 :, :, :] = True
    ye, ze = ny - (ny >> 3 << 3), nz - (nz >> 3 << 3)
    if ye:
        occ4[slot[ny >> 3, :][slot[ny >> 3, :] >= 0], :, ye:, :] = True
    if ze:
        occ4[slot[:, nz >> 3][slot[:, nz >> 3] >= 0], :, :, ze:] = True
    return slot, coords, occ4


def _face_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(P,2) unique positive label pairs from two matching face planes."""
    both = (a > 0) & (b > 0)
    if not both.any():
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.stack([a[both], b[both]], axis=1), axis=0)


def _fill_two_tier(
    occ_lin: np.ndarray,
    dims: np.ndarray,
    boxes: list[tuple[str, np.ndarray, np.ndarray]],
    coverage_bricks: float,
    tick: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """The band-limited two-phase fill (module docstring), two-tier:

    Sweep 1 labels each SKIN brick's air at fine resolution (one batched
    scipy call per brick x-layer; provisional ids = layer base + local label,
    raster-deterministic), records brick-face stitching pairs (fine↔fine
    across skin-brick faces, fine↔zone where a face borders a coarse-zone
    brick), per-component sizes, and the seed/ring bookkeeping — per-object
    AABB shell rings with the cross-object exclusion, exactly the classic
    phase-1 rules, evaluated per tier. Zone bricks (cover-free, within the
    coverage band) are labeled in one dense brick-grid scipy call; their
    components join the graph through the stitching pairs. Restricted to the
    explored band the components equal a whole-grid fine fill's (module
    docstring). Global components come from one sparse connected-components
    call, renumbered by minimum provisional id (first-encounter raster order);
    phase 2 (nested rescue) then runs UNCHANGED on the component tables.
    Sweep 2 replays the layer labeling and packs the per-brick EMPTY / AIR
    bitmasks.

    Returns everything downstream steps consume: the sorted skin-brick ids +
    their bitmasks, reached / garbage zone-brick ids, the brick-level distance
    field (zone candidate annotations), counts and fill stats."""
    nx, ny, nz = (int(v) for v in dims)
    nbx = (nx + _BRICK - 1) // _BRICK
    nby = (ny + _BRICK - 1) // _BRICK
    nbz = (nz + _BRICK - 1) // _BRICK

    # --- brick classification (dense brick grids — volume/512, always small) --
    plane = ny * nz
    brick_occ = np.zeros((nbx, nby, nbz), dtype=bool)
    for bx in range(nbx):
        x0, x1 = bx * _BRICK, min(bx * _BRICK + _BRICK, nx)
        lo, hi = np.searchsorted(occ_lin, [x0 * plane, x1 * plane])
        seg = occ_lin[lo:hi]
        if seg.size:
            rem = seg % plane
            brick_occ[bx, (rem // nz) >> 3, (rem % nz) >> 3] = True
    brick_skin = ndimage.binary_dilation(brick_occ, structure=np.ones((3, 3, 3), bool))
    # Distance (brick units) to the nearest cover brick: the band mask + the
    # coarse candidate clearance annotation.
    brick_dist = ndimage.distance_transform_edt(~brick_occ)
    zone = ~brick_skin & (brick_dist <= coverage_bricks)
    zone_lbl, n_zone = ndimage.label(zone, _STRUCT6)

    # Exact real-cell count per brick (edge bricks are partial).
    def _spans(n: int, nb: int) -> np.ndarray:
        s = np.full(nb, _BRICK, dtype=np.int64)
        if n & 7:
            s[-1] = n & 7
        return s

    sx, sy, sz = _spans(nx, nbx), _spans(ny, nby), _spans(nz, nbz)
    brick_cells = sx[:, None, None] * sy[None, :, None] * sz[None, None, :]

    # --- sweep 1: fine labels per skin brick + stitching + seeds/rings --------
    bases: list[int] = [0] * nbx      # provisional-id base per brick layer
    n_fine = 0
    size_parts: list[np.ndarray] = []
    ff_pairs: list[np.ndarray] = []   # fine↔fine provisional pairs
    fz_pairs: list[np.ndarray] = []   # (fine prov, zone comp) pairs
    seed_fine: list[np.ndarray] = []  # seeded fine provisional ids
    seed_zone: list[np.ndarray] = []  # seeded zone comp ids
    ring_fine: list[list[np.ndarray]] = [[] for _ in boxes]
    ring_zone: list[list[np.ndarray]] = [[] for _ in boxes]
    prev_face: np.ndarray | None = None  # (nby,nbz,8,8) prov ids of +x faces
    # Per-object augmented-box x-range, to skip layers cheaply.
    box_x = [(int(v[1][0]) - 1, int(v[2][0]) + 1) for v in boxes]
    # Stitching pairs dominate the fill's memory on dense-foliage scenes
    # (hundreds of millions of label adjacencies): store them at the smallest
    # dtype the node-id ceiling permits (int32 covers 512 cells per skin brick
    # + every zone comp on any realistic grid) and dedupe once per layer.
    cap_nodes = 512 * int(brick_skin.sum()) + n_zone + 2
    pair_dtype = np.int32 if cap_nodes < 2**31 - 1 else np.int64

    def _fold_pairs(dst: list[np.ndarray], parts: list[np.ndarray]) -> None:
        """Concat + dedupe one layer's pair blocks, store at `pair_dtype`."""
        if parts:
            dst.append(
                np.unique(np.concatenate(parts), axis=0).astype(pair_dtype)
            )

    for bx in range(nbx):
        x0, x1 = bx * _BRICK, min(bx * _BRICK + _BRICK, nx)
        skin_plane = brick_skin[bx]
        have_skin = bool(skin_plane.any())
        labels4 = None
        slot = np.full((nby, nbz), -1, dtype=np.int32)
        base = bases[bx] = n_fine
        if have_skin:
            slot, coords, occ4 = _layer_bricks(occ_lin, bx, dims, skin_plane)
            air4 = ~occ4
            labels4, n_local = ndimage.label(air4, _STRUCT6_BATCH)
            n_fine += int(n_local)
            size_parts.append(
                np.bincount(labels4.ravel(), minlength=n_local + 1)[1:]
            )
            lxe = x1 - x0 - 1  # last REAL local x plane of this layer
            layer_ff: list[np.ndarray] = []
            layer_fz: list[np.ndarray] = []

            # fine↔fine, within-layer (+y / +z brick faces).
            for axis, (dy, dz) in ((1, (1, 0)), (2, (0, 1))):
                sa = slot[: nby - dy, : nbz - dz]
                sb = slot[dy:, dz:]
                m = (sa >= 0) & (sb >= 0)
                if not m.any():
                    continue
                ia, ib = sa[m], sb[m]
                if axis == 1:
                    fa, fb = labels4[ia, :, 7, :], labels4[ib, :, 0, :]
                else:
                    fa, fb = labels4[ia, :, :, 7], labels4[ib, :, :, 0]
                p = _face_pairs(fa.astype(np.int64), fb.astype(np.int64))
                if len(p):
                    layer_ff.append(p + base)

            # fine↔fine, cross-layer (−x faces vs the previous layer's +x).
            cur_first = np.zeros((nby, nbz, _BRICK, _BRICK), dtype=np.int64)
            has = slot >= 0
            cur_first[has] = labels4[slot[has], 0].astype(np.int64)
            cur_first[cur_first > 0] += base
            if prev_face is not None:
                p = _face_pairs(prev_face.reshape(-1), cur_first.reshape(-1))
                if len(p):
                    layer_ff.append(p)
            nxt_face = np.zeros((nby, nbz, _BRICK, _BRICK), dtype=np.int64)
            if lxe == _BRICK - 1:  # a partial layer's +x side is grid edge
                nxt_face[has] = labels4[slot[has], _BRICK - 1].astype(np.int64)
                nxt_face[nxt_face > 0] += base
            prev_face = nxt_face

            # fine↔zone: any skin-brick face bordering a zone brick connects
            # its air cells to that brick's coarse component. Fully vectorized:
            # the selected bricks' face labels pair with their zone comp
            # broadcast per cell, deduped in one pass — no per-brick loop.
            for dxyz, face in (
                ((-1, 0, 0), lambda s: labels4[s, 0]),
                ((1, 0, 0), lambda s: labels4[s, lxe]),
                ((0, -1, 0), lambda s: labels4[s, :, 0]),
                ((0, 1, 0), lambda s: labels4[s, :, 7]),
                ((0, 0, -1), lambda s: labels4[s, :, :, 0]),
                ((0, 0, 1), lambda s: labels4[s, :, :, 7]),
            ):
                dx, dy, dz = dxyz
                nbx_i = bx + dx
                if nbx_i < 0 or nbx_i >= nbx:
                    continue
                by = coords[:, 0] + dy
                bz = coords[:, 1] + dz
                ok = (by >= 0) & (by < nby) & (bz >= 0) & (bz < nbz)
                if not ok.any():
                    continue
                zc = np.zeros(len(coords), dtype=np.int64)
                zc[ok] = zone_lbl[nbx_i, by[ok], bz[ok]]
                sel = np.nonzero(zc > 0)[0]
                if not sel.size:
                    continue
                fl = face(sel).reshape(len(sel), -1).astype(np.int64)
                zz = np.broadcast_to(zc[sel][:, None], fl.shape)
                m = fl > 0
                if m.any():
                    layer_fz.append(
                        np.unique(
                            np.stack([fl[m] + base, zz[m]], axis=1), axis=0
                        )
                    )

            # Boundary seeds (fine): air on the six grid faces is ambient.
            def _seed(lab: np.ndarray) -> None:
                u = np.unique(lab)
                u = u[u > 0]
                if u.size:
                    seed_fine.append(u.astype(np.int64) + base)

            if bx == 0:
                _seed(labels4[:, 0])
            if x1 == nx:
                _seed(labels4[:, lxe])
            for edge_b, edge_l, axis in (
                (0, 0, 1), ((ny - 1) >> 3, (ny - 1) & 7, 1),
                (0, 0, 2), ((nz - 1) >> 3, (nz - 1) & 7, 2),
            ):
                s = slot[edge_b, :] if axis == 1 else slot[:, edge_b]
                s = s[s >= 0]
                if s.size:
                    _seed(labels4[s, :, edge_l, :] if axis == 1 else labels4[s, :, :, edge_l])
            _fold_pairs(ff_pairs, layer_ff)
            _fold_pairs(fz_pairs, layer_fz)
        else:
            prev_face = np.zeros((nby, nbz, _BRICK, _BRICK), dtype=np.int64)

        # Phase-1 rings: this layer's slice of every object's shell ring — ALL
        # air ring nodes (rescue trigger) + the uncovered ones (seeds), each
        # resolved per tier (fine label inside skin bricks, coarse component
        # inside zone bricks, nothing beyond the band).
        touching = [
            bi for bi, (lo_x, hi_x) in enumerate(box_x)
            if lo_x <= x1 - 1 and hi_x >= x0
        ]
        if touching:
            covered = _covered_slab(boxes, x0, x1, ny, nz)
            for bi in touching:
                _nid, vlo, vhi = boxes[bi]
                for sl in _shell_slabs(vlo, vhi, dims):
                    a, b = max(sl[0].start, x0), min(sl[0].stop, x1)
                    if a >= b:
                        continue
                    X = np.arange(a, b, dtype=np.int64)[:, None, None]
                    Y = np.arange(sl[1].start, sl[1].stop, dtype=np.int64)[None, :, None]
                    Z = np.arange(sl[2].start, sl[2].stop, dtype=np.int64)[None, None, :]
                    s = slot[Y >> 3, Z >> 3]
                    fine_lab = np.zeros(np.broadcast_shapes(X.shape, Y.shape, Z.shape), np.int64)
                    if labels4 is not None:
                        m = np.broadcast_to(s >= 0, fine_lab.shape)
                        lab = labels4[
                            np.broadcast_to(np.maximum(s, 0), fine_lab.shape),
                            np.broadcast_to(X & 7, fine_lab.shape),
                            np.broadcast_to(Y & 7, fine_lab.shape),
                            np.broadcast_to(Z & 7, fine_lab.shape),
                        ]
                        fine_lab = np.where(m, lab, 0).astype(np.int64)
                    zc = np.broadcast_to(zone_lbl[bx, Y >> 3, Z >> 3], fine_lab.shape)
                    cov = covered[X - x0, Y, Z]
                    fa = fine_lab > 0
                    za = (zc > 0) & ~np.broadcast_to(s >= 0, fine_lab.shape)
                    if fa.any():
                        ring_fine[bi].append(np.unique(fine_lab[fa]) + base)
                        sm = fa & ~cov
                        if sm.any():
                            seed_fine.append(np.unique(fine_lab[sm]) + base)
                    if za.any():
                        ring_zone[bi].append(np.unique(zc[za]))
                        sm = za & ~cov
                        if sm.any():
                            seed_zone.append(np.unique(zc[sm]))
        if tick is not None:
            tick()

    # Zone boundary seeds: zone bricks on the grid's outer brick shell hold
    # boundary cells — ambient by construction (the margin).
    for face in (
        zone_lbl[0], zone_lbl[-1], zone_lbl[:, 0], zone_lbl[:, -1],
        zone_lbl[:, :, 0], zone_lbl[:, :, -1],
    ):
        u = np.unique(face)
        u = u[u > 0]
        if u.size:
            seed_zone.append(u.astype(np.int64))

    # --- global components: one sparse connected-components call ------------
    # Node space: fine provisional ids 1..n_fine, zone comps n_fine+1..+n_zone.
    n_nodes = n_fine + n_zone + 1
    edges = ff_pairs + [
        np.stack([p[:, 0], p[:, 1] + n_fine], axis=1) for p in fz_pairs
    ]
    if edges:
        e = np.concatenate(edges, axis=0)
    else:
        e = np.zeros((0, 2), dtype=np.int64)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    g = coo_matrix(
        (np.ones(len(e), dtype=np.int8), (e[:, 0], e[:, 1])),
        shape=(n_nodes, n_nodes),
    )
    _n_raw, comp_raw = connected_components(g, directed=False)
    # Renumber components 1..K by their minimum member node id — global
    # first-encounter raster order, replicating the classic fill's numbering
    # semantics (and with it the rescue's lowest-label tie-break).
    minid = np.full(_n_raw, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(minid, comp_raw, np.arange(n_nodes, dtype=np.int64))
    live = np.zeros(_n_raw, dtype=bool)
    live[comp_raw[1:]] = True  # node 0 is a dummy
    order = np.argsort(np.where(live, minid, np.iinfo(np.int64).max), kind="stable")
    final_of_raw = np.zeros(_n_raw, dtype=np.int64)
    n_comp = int(live.sum())
    final_of_raw[order[:n_comp]] = np.arange(1, n_comp + 1)
    node_final = final_of_raw[comp_raw]  # node id → final comp (1..K)
    node_final[0] = 0

    sizes = np.zeros(n_comp + 1, dtype=np.int64)
    if n_fine:
        np.add.at(
            sizes, node_final[1 : n_fine + 1], np.concatenate(size_parts)
        )
    if n_zone:
        zone_sizes = np.bincount(
            zone_lbl[zone].ravel(), weights=brick_cells[zone].ravel(),
            minlength=n_zone + 1,
        )[1:].astype(np.int64)
        np.add.at(sizes, node_final[n_fine + 1 :], zone_sizes)

    reached = np.zeros(n_comp + 1, dtype=bool)
    for part in seed_fine:
        reached[node_final[part]] = True
    for part in seed_zone:
        reached[node_final[part + n_fine]] = True
    seeded_components = int(reached.sum())
    ring_labels = []
    for bi in range(len(boxes)):
        nodes = [node_final[p] for p in ring_fine[bi]]
        nodes += [node_final[p + n_fine] for p in ring_zone[bi]]
        ring_labels.append(
            np.unique(np.concatenate(nodes)) if nodes else np.zeros(0, np.int64)
        )

    # Phase 2 — nested rescue, to fixpoint (UNCHANGED semantics: an object with
    # ZERO reached ring cells is sealed; admit ONE cavity per round — most
    # pending-adjacent, then largest, then lowest label — and re-evaluate).
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
        order = np.lexsort((-np.arange(n_comp + 1, dtype=np.int64), sizes, adj))
        reached[int(order[-1])] = True
        rescued_comps.append(int(order[-1]))
        rescue_rounds += 1

    # --- sweep 2: replay labels, pack the per-brick bitmasks ----------------
    skin_lin_parts: list[np.ndarray] = []
    air_parts: list[np.ndarray] = []
    empty_parts: list[np.ndarray] = []
    first_empty_parts: list[np.ndarray] = []
    empty_fine = 0
    air_fine = 0
    for bx in range(nbx):
        skin_plane = brick_skin[bx]
        if not skin_plane.any():
            if tick is not None:
                tick()
            continue
        slot, coords, occ4 = _layer_bricks(occ_lin, bx, dims, skin_plane)
        air4 = ~occ4
        labels4, n_local = ndimage.label(air4, _STRUCT6_BATCH)
        base = bases[bx]
        lut = np.zeros(n_local + 1, dtype=bool)
        lut[1:] = reached[node_final[base + 1 : base + n_local + 1]]
        empty4 = lut[labels4]
        s_l = len(coords)
        flat_e = empty4.reshape(s_l, 512)
        flat_a = air4.reshape(s_l, 512)
        empty_parts.append(np.packbits(flat_e, axis=1, bitorder="little"))
        air_parts.append(np.packbits(flat_a, axis=1, bitorder="little"))
        first = np.argmax(flat_e, axis=1).astype(np.int16)
        first[~flat_e.any(axis=1)] = -1
        first_empty_parts.append(first)
        empty_fine += int(flat_e.sum())
        air_fine += int(flat_a.sum())
        skin_lin_parts.append(
            (np.int64(bx) * nby + coords[:, 0]) * nbz + coords[:, 1]
        )
        if tick is not None:
            tick()

    nothing64 = np.zeros(0, dtype=np.int64)
    skin_lin = (
        np.concatenate(skin_lin_parts) if skin_lin_parts else nothing64
    )
    skin_air = (
        np.concatenate(air_parts) if air_parts else np.zeros((0, 64), np.uint8)
    )
    skin_empty = (
        np.concatenate(empty_parts) if empty_parts else np.zeros((0, 64), np.uint8)
    )
    skin_first_empty = (
        np.concatenate(first_empty_parts) if first_empty_parts
        else np.zeros(0, np.int16)
    )

    # Zone brick partitions by reached-ness (their comps are labeled air).
    zone_ids_grid = np.flatnonzero(zone)  # raster brick linear ids
    zone_comp = node_final[zone_lbl[zone].ravel() + n_fine]
    zone_reached_m = reached[zone_comp]
    zone_reached = zone_ids_grid[zone_reached_m]
    zone_garbage = zone_ids_grid[~zone_reached_m]
    zone_cells_all = brick_cells[zone].ravel()
    empty_zone = int(zone_cells_all[zone_reached_m].sum())
    garbage_zone = int(zone_cells_all[~zone_reached_m].sum())

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
    return {
        "skin_lin": skin_lin,
        "skin_air": skin_air,
        "skin_empty": skin_empty,
        "skin_first_empty": skin_first_empty,
        "zone_reached": zone_reached,
        "zone_garbage": zone_garbage,
        "brick_dist": brick_dist,
        "brick_skin": brick_skin,
        "empty_voxels": empty_fine + empty_zone,
        "garbage_voxels": (air_fine - empty_fine) + garbage_zone,
        "rescued_cells": rescued_cells,
        "stats": stats,
        "n_layers": nbx,
    }


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
            op_keys, gl_keys = _voxelize_geom(g, pitch)  # already sorted unique
            if len(op_keys):
                opaque_parts.append(op_keys)
            if len(gl_keys):
                glass_parts.append(gl_keys)
        del geoms
    except Exception:  # a bad mesh voxelizes to nothing — keep going
        return node_id, empty, empty, None, None
    return node_id, _merge_keys(opaque_parts), _merge_keys(glass_parts), lo, hi


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


def _brick_quads(mask: np.ndarray) -> np.ndarray:
    """Boundary quads of a BRICK-level bool grid, scaled to fine cell coords:
    each brick-face quad becomes `_BRICK` fine rows of run `_BRICK × run`.
    Coarse by design — used for the zone-level classes of the viz pack, where
    per-cell resolution can't exist (and wouldn't render legibly anyway)."""
    q = _boundary_quads(mask)
    if not len(q):
        return q
    B = _BRICK
    axis = q[:, 3] >> 1
    neg = q[:, 3] & 1
    # Row axis: the in-plane axis that is NOT the run axis (run: z for x/y
    # faces, y for z faces) and not the face axis.
    row_axis = np.where(axis == 0, 1, np.where(axis == 1, 0, 0))
    out = np.repeat(q, B, axis=0).astype(np.int32)
    out[:, :3] *= B
    out[:, 4] *= B
    rows = np.tile(np.arange(B, dtype=np.int32), len(q))
    out[np.arange(len(out)), row_axis.repeat(B)] += rows
    # The face plane sits at the cell's max corner for +faces: those quads'
    # face-axis coordinate must point at the LAST fine cell of the brick.
    plus = np.repeat(neg == 0, B)
    out[np.arange(len(out))[plus], axis.repeat(B)[plus]] += B - 1
    return out


# Bricks per chunk in the viz extraction: an unpacked chunk plus its exposure
# copies stays ~100-200 MB regardless of scene size (the packed inputs stay
# packed — 64 B/brick — and neighbour face planes unpack on demand).
_VIZ_CHUNK = 1 << 16
# Cell budget for FINE-resolution viz shells. The client builds the overlay's
# geometry quad-by-quad on its main thread (~0.4 quads per class cell), so a
# foliage scene's 54M cover cells → 21M quads → a 250 MB pack and >1 GB of
# JS arrays: the overlay never renders. Past this budget a class falls back
# to BRICK-resolution shells (24 cm — still perfectly legible as a debug
# overlay, ~50× fewer quads), which is what the free volume always uses.
_VIZ_FINE_CELL_CAP = 4_000_000


def _brick_class_quads(
    brick_ids: np.ndarray, bits: np.ndarray, dims: np.ndarray
) -> np.ndarray:
    """EXACT fine-resolution boundary quads of a per-brick cell class:
    `brick_ids` (B,) sorted brick linear ids, `bits` (B,64) uint8 PACKED cell
    masks (little bit order — the storage format, so callers compose classes
    with bytewise logic and never unpack whole scenes). Bricks unpack in
    `_VIZ_CHUNK` batches; exposure is evaluated with true cross-brick
    neighbours — in-brick shifts plus the adjacent brick's matching face
    plane, unpacked on demand through the sorted ids. A missing neighbour
    reads as exposed (grid edge / not-in-class air), exactly like a whole-grid
    extraction treats beyond-array; the out-of-grid padding of partial edge
    bricks is masked off after unpacking. Quads are emitted per direction in
    ascending brick order — identical output for any chunk size."""
    if not brick_ids.size:
        return np.zeros((0, 5), dtype=np.int32)
    nx, ny, nz = (int(v) for v in dims)
    nbx = (nx + _BRICK - 1) // _BRICK
    nby = (ny + _BRICK - 1) // _BRICK
    nbz = (nz + _BRICK - 1) // _BRICK
    S = len(brick_ids)
    bx = brick_ids // (nby * nbz)
    by = (brick_ids // nbz) % nby
    bz = brick_ids % nbz
    strides = {0: nby * nbz, 1: nbz, 2: 1}
    coords = {0: bx, 1: by, 2: bz}
    limits = {0: nbx, 1: nby, 2: nbz}

    def unpack(rows: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """(K,8,8,8) bool cells of `bits[idx]` with grid padding masked off."""
        c = (
            np.unpackbits(rows, axis=1, bitorder="little")
            .astype(bool)
            .reshape(-1, _BRICK, _BRICK, _BRICK)
        )
        if nx & 7:
            c[bx[idx] == nbx - 1, (nx & 7):, :, :] = False
        if ny & 7:
            c[by[idx] == nby - 1, :, (ny & 7):, :] = False
        if nz & 7:
            c[bz[idx] == nbz - 1, :, :, (nz & 7):] = False
        return c

    parts: list[np.ndarray] = []
    for axis in range(3):
        for neg in (0, 1):
            step = -1 if neg else 1
            # Neighbour lookup over the FULL id array (cheap int ops).
            nb = brick_ids + step * strides[axis]
            valid = (coords[axis] + step >= 0) & (coords[axis] + step < limits[axis])
            pos = np.clip(np.searchsorted(brick_ids, nb), 0, S - 1)
            found = valid & (brick_ids[pos] == nb)
            face_i = 0 if neg == 0 else _BRICK - 1
            for c0 in range(0, S, _VIZ_CHUNK):
                c1 = min(c0 + _VIZ_CHUNK, S)
                idx = np.arange(c0, c1)
                cells = unpack(bits[c0:c1], idx)
                exposed = cells.copy()
                sl_dst = [slice(None)] * 4
                sl_src = [slice(None)] * 4
                if neg == 0:
                    sl_dst[axis + 1], sl_src[axis + 1] = slice(None, -1), slice(1, None)
                else:
                    sl_dst[axis + 1], sl_src[axis + 1] = slice(1, None), slice(None, -1)
                exposed[tuple(sl_dst)] &= ~cells[tuple(sl_src)]
                # Brick-boundary plane: the neighbour's matching face, unpacked
                # only for the neighbours this chunk actually has.
                f_c = found[c0:c1]
                nb_plane = np.zeros((c1 - c0, _BRICK, _BRICK), dtype=bool)
                if f_c.any():
                    nb_idx = pos[c0:c1][f_c]
                    nb_cells = unpack(bits[nb_idx], nb_idx)
                    if axis == 0:
                        nb_plane[f_c] = nb_cells[:, face_i, :, :]
                    elif axis == 1:
                        nb_plane[f_c] = nb_cells[:, :, face_i, :]
                    else:
                        nb_plane[f_c] = nb_cells[:, :, :, face_i]
                sl_edge = [slice(None)] * 4
                sl_edge[axis + 1] = _BRICK - 1 if neg == 0 else 0
                exposed[tuple(sl_edge)] &= ~nb_plane
                # Run extraction: z-runs for x/y faces, y-runs for z faces.
                if axis == 2:
                    rows = np.moveaxis(exposed, 3, 2).reshape(-1, _BRICK)
                else:
                    rows = exposed.reshape(-1, _BRICK)
                padded = np.zeros((rows.shape[0], _BRICK + 2), dtype=np.int8)
                padded[:, 1:-1] = rows
                d = np.diff(padded, axis=1)
                starts = np.argwhere(d == 1)
                ends = np.argwhere(d == -1)
                if not len(starts):
                    continue
                row_i, col0 = starts[:, 0], starts[:, 1]
                runs = (ends[:, 1] - col0).astype(np.int32)
                s_i, r = np.divmod(row_i, _BRICK * _BRICK)
                s_i += c0
                i0, i1 = np.divmod(r, _BRICK)
                if axis == 2:  # rows were (lx, lz, ly): run along y
                    lx, ly, lz = i0, col0, i1
                else:          # rows were (lx, ly, lz): run along z
                    lx, ly, lz = i0, i1, col0
                quad = np.empty((len(runs), 5), dtype=np.int32)
                quad[:, 0] = bx[s_i] * _BRICK + lx
                quad[:, 1] = by[s_i] * _BRICK + ly
                quad[:, 2] = bz[s_i] * _BRICK + lz
                quad[:, 3] = axis * 2 + neg
                quad[:, 4] = runs
                parts.append(quad)
    if not parts:
        return np.zeros((0, 5), dtype=np.int32)
    return np.concatenate(parts, axis=0)


def _viz_two_tier(
    fill: dict[str, Any], dims: np.ndarray, clearance_m: float, n_cover: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, np.ndarray, int]], dict[str, str]]:
    """The SVX3 viz classes from the two-tier fill. Cover and garbage are
    EXACT fine-resolution boundary extractions while their cell counts fit
    `_VIZ_FINE_CELL_CAP`, composed as PACKED bitmasks (bytewise logic on the
    storage format — nothing scene-sized is ever unpacked): cover = ¬air over
    the skin bricks; garbage = (air ∧ ¬empty) over the skin bricks unioned
    with garbage zone bricks as 0xFF rows (the union extraction culls
    interior faces across the fine/coarse boundary automatically). PAST the
    cap — foliage scenes with tens of millions of cover cells — a class falls
    back to brick-resolution shells: the client assembles overlay geometry
    quad-by-quad on its main thread, and 20M+ fine quads simply never render
    there. The free volume is always ONE brick-resolution shell (a per-cell
    free ladder died with the dense clearance field; the slider's 'apply'
    still re-filters candidates server-side). Returns (cover, garbage,
    shells, resolutions) — `resolutions` names the per-class choice for the
    summary."""
    nbx = (int(dims[0]) + _BRICK - 1) // _BRICK
    nby = (int(dims[1]) + _BRICK - 1) // _BRICK
    nbz = (int(dims[2]) + _BRICK - 1) // _BRICK
    skin_lin = fill["skin_lin"]
    air_b = fill["skin_air"]
    emp_b = fill["skin_empty"]
    res: dict[str, str] = {}

    if n_cover <= _VIZ_FINE_CELL_CAP:
        cover_q = _brick_class_quads(skin_lin, np.invert(air_b), dims)
        res["cover"] = "fine"
    else:
        # Occupied bricks are exactly the zeros of the brick distance field.
        occ_b = fill["brick_dist"] == 0
        cover_q = _brick_quads(occ_b)
        res["cover"] = "brick"

    n_garbage = int(fill["garbage_voxels"])
    zg = fill["zone_garbage"]
    if n_garbage <= _VIZ_FINE_CELL_CAP:
        garb_ids = skin_lin
        garb_bits = air_b & np.invert(emp_b)
        if zg.size:
            garb_ids = np.concatenate([garb_ids, zg])
            garb_bits = np.concatenate(
                [garb_bits, np.full((zg.size, 64), 0xFF, dtype=np.uint8)]
            )
            order = np.argsort(garb_ids, kind="stable")
            garb_ids, garb_bits = garb_ids[order], garb_bits[order]
        garbage_q = _brick_class_quads(garb_ids, garb_bits, dims)
        res["garbage"] = "fine"
    else:
        gar_b = np.zeros((nbx, nby, nbz), dtype=bool)
        has_garb = (air_b & np.invert(emp_b)).any(axis=1)
        if has_garb.any():
            gar_b.reshape(-1)[skin_lin[has_garb]] = True
        if zg.size:
            gar_b.reshape(-1)[zg] = True
        garbage_q = _brick_quads(gar_b)
        res["garbage"] = "brick"

    free_b = np.zeros((nbx, nby, nbz), dtype=bool)
    has_empty = emp_b.any(axis=1)
    if has_empty.any():
        free_b.reshape(-1)[skin_lin[has_empty]] = True
    if fill["zone_reached"].size:
        free_b.reshape(-1)[fill["zone_reached"]] = True
    shells = [(float(clearance_m), _brick_quads(free_b), int(fill["empty_voxels"]))]
    res["free"] = "brick"
    return cover_q, garbage_q, shells, res


def _emit_candidates(
    fill: dict[str, Any],
    occ_lin: np.ndarray,
    dims: np.ndarray,
    origin: np.ndarray,
    pitch: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The camera-candidate list: one position per `_CAND_BRICKS`³-brick block
    of reached air, each annotated with its distance to the nearest surface.

    Skin blocks pick the lowest-raster skin brick with empty air and that
    brick's first empty cell (deterministic), annotated EXACTLY via a local
    search over the cover cells of the brick's 3×3×3 neighbourhood (a skin
    brick always has cover within it — that's what makes it skin). Blocks
    with no skin-empty fall to their lowest reached zone brick, positioned at
    the brick's central cell and annotated from the brick-level distance
    field (±1 brick — sub-voxel precision cannot matter half a metre from
    everything). Returns (cand_pos (M,3) float32 world, cand_clear (M,)
    float32 metres)."""
    nx, ny, nz = (int(v) for v in dims)
    nby = (ny + _BRICK - 1) // _BRICK
    nbz = (nz + _BRICK - 1) // _BRICK
    skin_lin = fill["skin_lin"]
    first = fill["skin_first_empty"]

    def brick_xyz(lin: np.ndarray) -> np.ndarray:
        return np.stack(
            [lin // (nby * nbz), (lin // nbz) % nby, lin % nbz], axis=1
        )

    # --- skin candidates: per cand-block, first empty cell of lowest brick --
    havef = first >= 0
    s_lin = skin_lin[havef]
    s_first = first[havef].astype(np.int64)
    sxyz = brick_xyz(s_lin)
    blk = (
        (sxyz[:, 0] // _CAND_BRICKS) * ((nby + 1) // _CAND_BRICKS + 1)
        + sxyz[:, 1] // _CAND_BRICKS
    ) * ((nbz + 1) // _CAND_BRICKS + 1) + sxyz[:, 2] // _CAND_BRICKS
    # skin_lin ascends ⇒ within a block the first row is the lowest brick.
    _u, idx = np.unique(blk, return_index=True)
    pick_lin, pick_first, pick_xyz = s_lin[idx], s_first[idx], sxyz[idx]
    lx, r = np.divmod(pick_first, _BRICK * _BRICK)
    ly, lz = np.divmod(r, _BRICK)
    cells = pick_xyz * _BRICK + np.stack([lx, ly, lz], axis=1)

    # Exact local clearance: min distance to any cover cell in the 27 bricks
    # around the candidate's brick (cover coords grouped per occupied brick).
    # Everything cover-sized is held at the SMALLEST sufficient dtype — int16
    # coords (grid axes are < 32K cells by construction), int32 brick ids —
    # and dropped as soon as the sorted copy exists: ~10 B per cover cell
    # instead of the ~50 B the int64 version peaked at on foliage scenes.
    plane = ny * nz
    cov_x = occ_lin // plane
    cov_rem = occ_lin % plane
    cov = np.empty((len(occ_lin), 3), dtype=np.int16)
    cov[:, 0] = cov_x
    cov[:, 1] = cov_rem // nz
    cov[:, 2] = cov_rem % nz
    del cov_x, cov_rem
    cov_brick = (
        (cov[:, 0].astype(np.int32) >> 3) * nby + (cov[:, 1] >> 3)
    ) * nbz + (cov[:, 2] >> 3)
    order = np.argsort(cov_brick, kind="stable")
    cov_sorted = cov[order]
    cb_sorted = cov_brick[order]
    del cov, cov_brick, order
    cb_ids, cb_starts = np.unique(cb_sorted, return_index=True)
    cb_ends = np.append(cb_starts[1:], len(cb_sorted))
    del cb_sorted

    clear = np.full(len(cells), np.inf, dtype=np.float64)
    offs = np.array(
        [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
        dtype=np.int64,
    )
    nbx = (nx + _BRICK - 1) // _BRICK
    # Vectorized over candidates, 27 neighbour offsets at a time: gather each
    # candidate's cover cells through the CSR ranges (repeat/cumsum expansion)
    # and reduce min distance with minimum.at — squared distances in int32
    # (local extents span ≤ 3 bricks, d² ≤ 3·24²). Chunked so the expanded
    # pair arrays stay tens of MB.
    for c0 in range(0, len(cells), 65536):
        c1 = min(c0 + 65536, len(cells))
        cc = cells[c0:c1].astype(np.int32)
        bb = pick_xyz[c0:c1]
        best = np.full(c1 - c0, np.iinfo(np.int32).max, dtype=np.int32)
        for off in offs:
            qb = bb + off
            ok = np.all((qb >= 0) & (qb < [nbx, nby, nbz]), axis=1)
            if not ok.any():
                continue
            ql = (qb[:, 0] * nby + qb[:, 1]) * nbz + qb[:, 2]
            p = np.clip(np.searchsorted(cb_ids, ql), 0, len(cb_ids) - 1)
            found = ok & (cb_ids[p] == ql)
            if not found.any():
                continue
            fi = np.nonzero(found)[0]
            starts, ends = cb_starts[p[fi]], cb_ends[p[fi]]
            counts = (ends - starts).astype(np.int64)
            # ranges: for each found candidate, indices starts[k]..ends[k)
            total = int(counts.sum())
            if not total:
                continue
            rep = np.repeat(np.arange(len(fi)), counts)
            base = np.repeat(starts, counts)
            step = np.arange(total) - np.repeat(
                np.concatenate([[0], np.cumsum(counts)[:-1]]), counts
            )
            diff = cov_sorted[base + step].astype(np.int32) - cc[fi][rep]
            d2 = (diff * diff).sum(axis=1)
            np.minimum.at(best, fi[rep], d2)
        clear[c0:c1] = np.sqrt(best.astype(np.float64)) * pitch
    skin_pos = (origin + (cells + 0.5) * pitch).astype(np.float32)
    skin_clear = clear.astype(np.float32)
    skin_blk = blk[idx]

    # --- zone candidates: per cand-block, lowest reached zone brick ----------
    z_lin = fill["zone_reached"]
    if z_lin.size:
        zxyz = brick_xyz(z_lin)
        zblk = (
            (zxyz[:, 0] // _CAND_BRICKS) * ((nby + 1) // _CAND_BRICKS + 1)
            + zxyz[:, 1] // _CAND_BRICKS
        ) * ((nbz + 1) // _CAND_BRICKS + 1) + zxyz[:, 2] // _CAND_BRICKS
        _zu, zidx = np.unique(zblk, return_index=True)
        # Skin wins a contested block (nearer the content).
        keep = ~np.isin(_zu, skin_blk)
        zidx = zidx[keep]
        zc = np.minimum(zxyz[zidx] * _BRICK + _BRICK // 2, np.array([nx, ny, nz]) - 1)
        zpos = (origin + (zc + 0.5) * pitch).astype(np.float32)
        bd = fill["brick_dist"].reshape(-1)[z_lin[zidx]]
        zclear = (bd * _BRICK * pitch).astype(np.float32)
        pos = np.concatenate([skin_pos, zpos], axis=0)
        cl = np.concatenate([skin_clear, zclear], axis=0)
    else:
        pos, cl = skin_pos, skin_clear
    return pos, cl


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
# Ceiling even on huge-RAM, many-core hosts. 16 (was 8): since the mesh pass
# streams pieces in bounded batches a worker peaks ~300 MB, so 16 workers fit
# a 16 GB box with headroom — the old cap predates the batching and was
# sized for 1.5 GB workers.
_MAX_AUTO_WORKERS = 16


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
    """Voxelize the cell's placed meshes into the uniform fine occupancy
    lattice (in parallel across objects — see `_iter_voxelized`), classify the
    near-surface air empty vs garbage with the two-tier band fill, emit the
    annotated camera-candidate list, and write `freespace.npz` (metadata +
    sparse structures, to `out_path`) + the SVX3 `voxels.bin` viz pack + the
    `.skin.npy` bitmask sidecar (beside it). Returns a summary (grid dims,
    counts, fill stats). Output is byte-identical for any worker count and
    any mesh-pass batch size."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"placed-mesh dir not found: {raw_dir}")
    ids = placed_object_ids(raw_dir)
    if not ids:
        raise FileNotFoundError(f"no placed meshes in {raw_dir}")
    pitch = float(np.clip(params.pitch, *_PITCH_CLAMP))
    margin = float(max(0.0, params.margin))
    clearance_m = float(max(0.0, params.clearance))
    coverage_m = float(max(params.coverage, pitch * _BRICK * 2))
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
        progress(0, 0, "reduce")  # merge keys → grid + occupancy
    # Run-aware merges: per-object key arrays are already sorted unique, so
    # `_merge_keys`' timsort effectively k-way merges them instead of paying a
    # run-blind full sort over every cover incidence.
    opaque_keys = _merge_keys(opaque_parts)
    glass_keys = _merge_keys(glass_parts)
    del opaque_parts, glass_parts  # the merged keys supersede the partials

    imin, dims = _grid_dims(lo, hi, margin, pitch)
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    origin = imin.astype(np.float64) * pitch
    shift = _VOX_RADIX.bit_length() - 1  # the key packing is pure bit-fields

    def to_lin(keys: np.ndarray) -> np.ndarray:
        """SORTED unique absolute keys → sorted unique local linear indices
        (drops out-of-grid cells) with NO sort: both encodings are x-major /
        y-mid / z-minor lexicographic, so the mapping is strictly monotone and
        the input's order survives the arithmetic. Shift/mask decode (the
        radix is a power of two — integer division is ~10× slower), chunked
        so temporaries stay tens of MB at swamp-scale key counts."""
        out = np.empty(keys.size, dtype=np.int64)
        n = 0
        mask = _VOX_RADIX - 1
        for i in range(0, keys.size, 1 << 24):
            k = keys[i : i + (1 << 24)]
            xl = (k >> (2 * shift)) - _VOX_OFF - imin[0]
            yl = ((k >> shift) & mask) - _VOX_OFF - imin[1]
            zl = (k & mask) - _VOX_OFF - imin[2]
            m = (
                (xl >= 0) & (xl < nx) & (yl >= 0) & (yl < ny)
                & (zl >= 0) & (zl < nz)
            )
            lin = (xl[m] * ny + yl[m]) * nz + zl[m]
            out[n : n + lin.size] = lin
            n += lin.size
        return out[:n]

    opaque_lin = to_lin(opaque_keys)
    # A cell with BOTH classes (e.g. window frame + pane) occludes: opaque wins,
    # keeping the two stored sets disjoint (glass = transmissive-only cells).
    glass_lin = np.setdiff1d(to_lin(glass_keys), opaque_lin, assume_unique=True)
    # Disjoint sorted union: timsort merges the two runs in ~O(N).
    occ_lin = np.concatenate([opaque_lin, glass_lin])
    occ_lin.sort(kind="stable")
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

    # Scratch names carry a per-invocation token: builds for the same cell can
    # OVERLAP (a revert's task.cancel() can't stop a build already running in a
    # worker thread, and a fresh POST then starts another); unique names make
    # overlap harmless, and the age-gated sweep reclaims the litter of builds
    # killed too hard for their cleanup to run.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skin_sidecar = out_path.with_name(out_path.name + _SKIN_SIDECAR_SUFFIX)
    token = f"{os.getpid():x}-{int(time.time() * 1000):x}"
    skin_tmp = out_path.with_name(f"{out_path.name}.skin.{token}.tmp.npy")
    tmp_npz = out_path.with_name(f"{out_path.name}.{token}.tmp.npz")
    now = time.time()
    for stale in out_path.parent.glob(out_path.name + ".*.tmp.np*"):
        try:
            if now - stale.stat().st_mtime > _SCRATCH_TTL_S:
                stale.unlink()
        except OSError:
            pass

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

    # THE FILL — the two-tier band fill (module docstring): fine labeling per
    # skin-brick layer (2 sweeps → a tick per layer per sweep), coarse zone in
    # dense brick-grid calls, seeds/rescue on component tables.
    coverage_bricks = max(1.0, coverage_m / (pitch * _BRICK))
    nbx = (nx + _BRICK - 1) // _BRICK
    fill = _fill_two_tier(
        occ_lin, dims, boxes, coverage_bricks, _stepper(2 * nbx, "fill")
    )

    # CANDIDATES — the camera positions Stage 4 plans from, annotated with
    # their distance to the nearest surface (module docstring).
    if progress is not None:
        progress(0, 0, "candidates")
    cand_pos, cand_clear = _emit_candidates(fill, occ_lin, dims, origin, pitch)
    free_count = int((cand_clear >= clearance_m - _CLEARANCE_EPS).sum())

    # SVX3 viz pack: fine cover/garbage quads (brick-level past the client's
    # renderable budget) + the brick-level free shell.
    if progress is not None:
        progress(0, 0, "viz")
    cover_q, garbage_q, shells, viz_res = _viz_two_tier(
        fill, dims, clearance_m, int(occ_lin.size)
    )

    if progress is not None:
        progress(0, 0, "write")
    try:
        np.save(skin_tmp, fill["skin_empty"])
        _savez_fast(
            tmp_npz,
            origin=origin.astype(np.float64),
            pitch=np.float64(pitch),
            dims=np.array([nx, ny, nz], dtype=np.int64),
            occ_lin=occ_lin.astype(np.int64),
            occ_lin_glass=glass_lin.astype(np.int64),
            skin_lin=fill["skin_lin"].astype(np.int64),
            zone_lin=fill["zone_reached"].astype(np.int64),
            cand_pos=cand_pos.astype(np.float32),
            cand_clear=cand_clear.astype(np.float32),
            clearance_m=np.float64(clearance_m),
        )
        tmp_npz.replace(out_path)
        # Promote the sidecar AFTER the npz lands (loaders treat it as the
        # sole home of the per-cell empty bits).
        skin_tmp.replace(skin_sidecar)
        os.utime(skin_sidecar)
    finally:
        for p in (skin_tmp, tmp_npz):
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
            # `cells` is DISPLAY-ONLY (the client's slider label) and the wire
            # field is u32: a large open scene's in-band air can exceed 2³²
            # cells (a 200 m platformer level did), so saturate rather than
            # overflow — quad counts stay exact (they drive parsing offsets).
            f.write(_SHELL_HEADER.pack(t, len(q), min(int(cells), 0xFFFFFFFF)))
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
        "empty_voxels": int(fill["empty_voxels"]),
        "free_voxels": free_count,       # candidates passing the baked filter
        "candidates": int(len(cand_pos)),
        "garbage_voxels": int(fill["garbage_voxels"]),
        "rescued_voxels": int(fill["rescued_cells"]),
        "skin_bricks": int(fill["skin_lin"].size),
        "zone_bricks": int(fill["zone_reached"].size + fill["zone_garbage"].size),
        "fill": fill["stats"],
        "viz": {
            "cover_quads": int(len(cover_q)),
            "garbage_quads": int(len(garbage_q)),
            "resolution": viz_res,
            "shells": [
                {"clearance": round(t, 4), "quads": int(len(q)), "cells": c}
                for t, q, cells in shells
                for c in (cells,)
            ],
        },
        "params": params.as_summary(),
        "bytes": viz_path.stat().st_size,
        "grid_bytes": out_path.stat().st_size,
    }
