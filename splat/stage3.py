"""Stage 3 — Surfel sampler (mesh → Gaussian splat).

Turns a cell's placed meshes into a **pre-fine-tuning Gaussian splat**:
a cloud of flat, surface-aligned 2D Gaussians (surfels), written as a **2DGS `.ply`**
(two tangent scales, no thickness) that 2DGS-capable web viewers (mkkellogg, PlayCanvas
v2.16+) read and the Stage-6 gsplat **2DGS** fine-tune inits from. A `3dgs` mode appends
a thin third scale for 3DGS-only viewers / SOG-SPZ compression.

Consumes the **Stage-2 free-space grid** (`stage2.load_free_space`):
  * **Normal orientation (fixes unreliable TRELLIS winding):** each surfel's normal is
    flipped to point toward EMPTY space (`fs.empty_at`), rather than trusting the mesh's
    face winding — so disks face the viewer regardless of authoring.
  * **Hidden-face culling:** a surfel with no empty space on EITHER side is buried
    (a solid's interior, or the seam between two abutting objects) and is dropped.

ONE KNOB — `detail`. Base spacing = `_BASE_SPACING / sqrt(detail)`, so detail=1
reproduces the default ~3 cm base, detail=2 doubles density, 0.5 halves it. Disk
radius = `_RADIUS_FRAC` × the surfel's cell size; the rest are fixed constants.

SCENE-SCALE ADAPTIVITY (angular, not metric). A screen pixel is an ANGLE, so the
metric detail worth storing scales with typical viewing distance — which scales
with a scene's open sightlines, not its bounding box. Per cell we derive a scale
`k` from the Stage-2 candidates' distance-to-surface (`_scene_scale`) and sample
each object at `base · clamp(obj_diag / _VIEW_REF_M, 1, k)` (`_object_spacing`):
small props stay at `base` at any scene size, large-area surface (terrain, long
walls) reaches the ceiling `base·k`. Indoor scenes (hotel-anchored) get k=1 and
are UNCHANGED; a 60,000 m² swamp gets k≈2.2, and with coarsening its cloud drops
from ~1.25 GB (uniform-base coarsened) to ~240 MB — angular fidelity held roughly
constant while metric fidelity tracks how far the surface is actually seen.

SAMPLING — SEEDED DARTS + LATTICE-HASH THINNING. Candidates are area-weighted random
surface points (`trimesh.sample.sample_surface`, SEEDED per object/slab — no
worker-count dependence), then thinned by an integer lattice: bin each candidate into
a cell of edge `_THIN_CELL_FACTOR·spacing` and keep the ONE candidate nearest the cell
centre. This replaces the old oversample-and-eliminate Poisson sampler — which,
MEASURED on a hotel cell, generated ~36× the kept points, built a KD-tree over them
and materialized a ~1.7-BILLION-entry too-close-pair table (~54 GB transient, ~22 GB
peak RSS, ~63 % of runtime) — with a few integer passes: no KD-tree, no pair table,
2.4× oversample instead of ~6×. Every surfel is a real dart sitting exactly on the
(lite) mesh surface with its face index → exact barycentric UV → exact base-colour
texel, so the colour/orient/cull paths are the long-validated ones.

FEATURE-ADAPTIVE DENSITY = GEOMETRY-DRIVEN COARSENING (both directions matter; this
is the one that pays). After orient+cull+colour, surviving base surfels are merged
BOTTOM-UP in octaves (cell edge 2·s, 4·s, … up to `2^_COARSEN_OCTAVES·s`): the nodes
in each parent cell are split into two FACING SIDES (by the sign of each oriented
normal against the group's principal axis — orientation is already air-corrected, so
the two faces of a wall/sheet split cleanly and NEVER merge), and a side collapses
into ONE bigger surfel only when a single flat disk truly represents it:
  * FILLED  — accumulated base-cell count ≥ `_MERGE_FILL·4^k` (a sliver or border
    never becomes a big disk overhanging air);
  * SMOOTH  — oriented normals agree (`1 − |mean n̂| ≤ _MERGE_FLAT`): gently curved
    mud/rock coarsens, tight curvature stays fine;
  * COPLANAR — member positions lie within `_MERGE_OFFSET_FRAC·s` (ABSOLUTE, base-
    spacing scale) of the side's plane: silhouettes stay tight and stacked parallel
    sheets can never fuse, whatever their normals say.
Merged colour is the base-count-weighted mean of the members' EXACT texel colours —
an average of true on-surface samples (never a mip/footprint filter, so UV-atlas
gutters can't bleed in). NO colour gate: a textured-but-flat surface coarsens; its
high-frequency albedo is Stage 6's job (it retrains colour against the reference
renders and densifies only where those renders demand it). This is what makes splat
size track scene INFORMATION instead of scene AREA: a swamp's mudflats collapse to
~24 cm disks while its cattails stay at 3 cm — without it, a 59,600 m² outdoor cell
at uniform 3 cm density is ~46 M surfels ≈ 3 GB of float32 PLY versus 156 MB of lite
GLBs, i.e. paying area-rent for content the source encodes in kilobytes.

CHUNKING: an object whose candidate budget exceeds `_CHUNK_MAX_CANDIDATES` is sampled
in `_CHUNK_EXTENT`-metre spatial slabs (partition its triangles, sample each), so a
single huge mesh (terrain, a racetrack ribbon) has BOUNDED transient memory — the
lattice keys are absolute, so slab seams cost only a one-cell layer at base density
(merging is per-slab too; a parent cell cut by a slab boundary just stays finer).

Optional `_FEATURE_REFINE` (OFF): the legacy crease/boundary band refinement
(`_spacing_bands`) that samples near feature edges up to `_FEATURE_BOOST`× finer. On
dense organic TRELLIS meshes the >25° crease test fires on ~85 % of surface, so it
multiplies counts 4-6× — the old sampler only survived it because its eliminator
undersampled those bands back to ~base density. Off until a detector separates real
edges from organic bumpiness.

PIPELINE PER OBJECT (parallel across objects — `_iter_sampled`, spawn pool,
id-ordered merge): load → darts → THIN → ORIENT + CULL → colour survivors → COARSEN →
emit. Deterministic end to end (seeded darts, integer lattices), so a cell resamples
to the identical cloud for any worker count.

Each surfel: position (world), rotation (quaternion aligning +Z to the oriented
normal), scale (r, r, ~0), color (exact texel for base surfels; member-mean for
merged), opacity (texel alpha, honouring `alphaMode`). Stored SH is degree 0
(`f_dc`) — unlit / view-independent.

Pure library: `sample_cell` takes explicit paths. Meshes are consumed AS-IS through
`splat.assets.load_geoms` — vanilla glTF via trimesh, KTX2/Meshopt sets via the
in-process decoder — so albedo sampling behaves identically on every encoding.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from splat.assets import load_geoms
from splat.stage2 import FreeSpace, _subdivide_edges, _valid_tri_mask, load_free_space

# trimesh logs per-file load noise; silence it for whole-cell runs.
logging.getLogger("trimesh").setLevel(logging.ERROR)

# The Gaussian cloud filename written under a cell's `splat/` dir (a 2DGS `.ply`).
CLOUD_NAME = "cloud.ply"

# SH degree-0 basis constant: colour c in [0,1] maps to f_dc = (c - 0.5) / C0.
_SH_C0 = 0.28209479177387814

# The calibrated default resolution: base sample spacing (m) at detail = 1.
_BASE_SPACING = 0.0295
DEFAULT_DETAIL = 1.0
_DETAIL_CLAMP = (0.05, 16.0)
_SPACING_CLAMP = (0.004, 0.25)

# Disk σ = _RADIUS_FRAC × the surfel's cell size (base spacing for unmerged surfels,
# 2^k× that for merged ones) — disks tile their cell block; Stage 6 fine-tunes scales.
_RADIUS_FRAC = 0.9
# 3dgs-mode only: fake thickness (third scale = radius × _FLATNESS).
_FLATNESS = 0.1

# --- lattice-hash thinning (replaces the Poisson oversample-and-eliminate) ----
# Thinning cell edge, × spacing: ONE winner survives per cell (the candidate nearest
# the cell centre — a soft, even spacing; no hard Poisson floor, no all-pairs pass).
_THIN_CELL_FACTOR = 1.0
# Candidate budget, × area/spacing²: ~2.4 per cell → ~91 % of cells occupied
# (Poisson e^-2.4); the old sampler generated ~6×/spacing².
_THIN_CANDIDATES = 2.4

# --- geometry-driven coarsening (see module docstring) -------------------------
# Octaves of bottom-up merging above base spacing: 3 → biggest surfel 8·s (~24 cm at
# detail 1). Raising it grows the flat-region win quadratically but tests each extra
# level against the same absolute planarity guard.
_COARSEN_OCTAVES = 3
# FILLED: a (parent cell, side) merges only when its accumulated base-cell count is
# at least this fraction of the in-plane expectation 4^k (a flat sheet through a
# 2^k-cell cube crosses ~(2^k)² base cells; borders/slivers fall short and stay fine).
_MERGE_FILL = 0.5
# SMOOTH: oriented-normal agreement, as 1 − |mean unit normal| ≤ this. cos-form of a
# ~12° spread — gentle terrain curvature merges, tight curvature doesn't.
_MERGE_FLAT = float(1.0 - np.cos(np.deg2rad(12.0)))
# COPLANAR: members must lie within this × BASE spacing of the side's plane —
# absolute, not per-level, so merged silhouettes stay sub-spacing tight and two
# stacked sheets (thin wall faces, leaf layers) can never fuse.
_MERGE_OFFSET_FRAC = 0.35

# --- scene-scale-adaptive spacing (angular fidelity, not metric) ---------------
# A screen pixel is an ANGLE, so the metric detail worth storing scales with
# typical viewing distance. We estimate that per cell from the Stage-2 camera
# candidates' distance-to-surface (`cand_clear`) — the scene's open-sightline
# statistic — NOT the bounding-box diagonal (which misreads a big building full
# of small rooms as far-viewed). `k = clamp(P_pctl(cand_clear)/ref, 1, kmax)`,
# anchored so a hotel-room's ~1.7 m P90 gives k=1 (indoor scenes UNCHANGED).
# cand_clear is capped by the Stage-2 coverage band (~5 m), so k is a
# conservative under-estimate for very open scenes — the safe direction.
_SIGHTLINE_PCTL = 90
_SIGHTLINE_REF_M = 1.73         # hotel-room P90(cand_clear): the k=1 anchor
_SCENE_K_CLAMP = (1.0, 4.0)
# Per-object floor: an object is typically framed from ~its own diagonal away, so
# its spacing scales with `obj_diag / _VIEW_REF_M`, clamped to [base, base·k].
# So small props (cattail, chair) stay at base however large the scene, mid-size
# hero objects scale with their own size, and only large-area surface (terrain,
# long walls) reaches the scene ceiling. 2 m = the hotel-anchor framing distance.
_VIEW_REF_M = 2.0

# --- legacy feature refinement (OFF by default; see module docstring) ----------
_FEATURE_REFINE = False
_FEATURE_BOOST = 4.0
_CREASE_ANGLE = np.deg2rad(25.0)
_EDGE_PT_CAP = 400_000  # stride a pathological crease soup instead of exploding
_CLASSIFY_COARSE = 4.0  # band labeling's first-pass split limit (× spacing)

# --- chunking: bound a single huge object's transient memory -------------------
_CHUNK_MAX_CANDIDATES = 8_000_000
_CHUNK_EXTENT = 8.0

# When the skin bitmask sidecar can't be memory-mapped, workers hold a private
# copy; cap the pool so resident copies stay under this budget.
_WORKER_GRID_BUDGET = 3_000_000_000  # bytes, across all workers

# Opacity is stored pre-sigmoid; clamp alpha off the 0/1 asymptotes.
_ALPHA_CLAMP = (1e-3, 1.0 - 1e-3)

# Material alpha modes whose sampled base-color alpha is meaningful (glass, cutout).
_TRANSPARENT_ALPHA_MODES = ("BLEND", "MASK")


@dataclass(frozen=True)
class SampleParams:
    """THE sampling knob (plus a format flag). `detail` multiplies surfel density
    around the calibrated default look. `representation` picks the output encoding:
    `2dgs` (native flat surfels, the default and what Stage 6 trains) or `3dgs`
    (adds a thin third scale)."""

    detail: float = DEFAULT_DETAIL
    representation: str = "2dgs"

    @property
    def spacing(self) -> float:
        """Base sample spacing (m) this detail level resolves to."""
        d = float(np.clip(self.detail, *_DETAIL_CLAMP))
        return float(np.clip(_BASE_SPACING / np.sqrt(d), *_SPACING_CLAMP))

    def as_summary(self) -> dict[str, Any]:
        return {
            "detail": self.detail,
            "spacing": self.spacing,
            "representation": self.representation,
        }


# progress(done, total, current_id) — called after each object is sampled.
ProgressCb = Callable[[int, int, str], None]

_MID_GREY = np.array([0.6, 0.6, 0.6], dtype=np.float32)


def placed_object_ids(raw_dir: Path) -> list[str]:
    """Ids with a placed, world-space mesh in `raw_dir` — the served `<id>.glb`,
    excluding `<id>.raw.glb` pre-placement intermediates. Sorted for determinism."""
    return sorted(
        p.name[: -len(".glb")]
        for p in raw_dir.glob("*.glb")
        if not p.name.endswith(".raw.glb")
    )


def _alpha_mode(geom: trimesh.Trimesh) -> str:
    material = getattr(getattr(geom, "visual", None), "material", None)
    return str(getattr(material, "alphaMode", None) or "OPAQUE").upper()


def _vertex_colors(geom: trimesh.Trimesh) -> np.ndarray:
    """Per-vertex RGBA in [0,1]: from a readable base-color texture at the vertex
    UVs when present, else the material's baseColorFactor, else neutral grey."""
    n = len(geom.vertices)
    visual = getattr(geom, "visual", None)
    try:
        if visual is not None and getattr(visual, "uv", None) is not None:
            material = getattr(visual, "material", None)
            if getattr(material, "baseColorTexture", None) is not None:
                cols = np.asarray(visual.to_color().vertex_colors, dtype=np.float32)
                if cols.shape == (n, 4):
                    return cols / 255.0
    except Exception:
        pass
    material = getattr(visual, "material", None)
    factor = getattr(material, "baseColorFactor", None)
    if factor is not None:
        arr = np.asarray(factor, dtype=np.float32).reshape(-1)
        if arr.size >= 3 and arr.max() > 1.0:
            arr = arr / 255.0
        rgb = arr[:3]
        alpha = float(arr[3]) if arr.size >= 4 else 1.0
        if not np.allclose(rgb, 1.0):
            out = np.empty((n, 4), dtype=np.float32)
            out[:, :3] = rgb
            out[:, 3] = alpha
            return out
    out = np.empty((n, 4), dtype=np.float32)
    out[:, :3] = _MID_GREY
    out[:, 3] = 1.0
    return out


def _align_quaternions(normals: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternions (w,x,y,z) rotating +Z onto each unit `normal`.
    Antipodal case (n ≈ −Z) rotates 180° about X."""
    n = normals.shape[0]
    dot = normals[:, 2]
    quat = np.empty((n, 4), dtype=np.float64)
    quat[:, 0] = 1.0 + dot
    quat[:, 1] = -normals[:, 1]
    quat[:, 2] = normals[:, 0]
    quat[:, 3] = 0.0
    antipodal = quat[:, 0] < 1e-6
    quat[antipodal] = (0.0, 1.0, 0.0, 0.0)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return quat


def _feature_edge_points(geom: trimesh.Trimesh, step: float) -> np.ndarray | None:
    """Points sampled every ~`step` m along the mesh's FEATURE edges: open boundary
    edges + creases sharper than `_CREASE_ANGLE`. None when the mesh has none."""
    try:
        parts = []
        ang = np.asarray(geom.face_adjacency_angles)
        if ang.size:
            parts.append(np.asarray(geom.face_adjacency_edges)[ang > _CREASE_ANGLE])
        boundary = trimesh.grouping.group_rows(geom.edges_sorted, require_count=1)
        if len(boundary):
            parts.append(np.asarray(geom.edges_sorted)[boundary])
        if not parts:
            return None
        edges = np.concatenate(parts, axis=0)
    except Exception:
        return None
    if not len(edges):
        return None
    v0 = np.asarray(geom.vertices)[edges[:, 0]]
    v1 = np.asarray(geom.vertices)[edges[:, 1]]
    seg = v1 - v0
    length = np.linalg.norm(seg, axis=1)
    cnt = np.maximum(np.ceil(length / max(step, 1e-6)).astype(np.int64) + 1, 2)
    total = int(cnt.sum())
    if total > _EDGE_PT_CAP:
        stride = int(np.ceil(total / _EDGE_PT_CAP))
        cnt = np.maximum(cnt // stride, 2)
        total = int(cnt.sum())
    rep = np.repeat(np.arange(len(edges)), cnt)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    t = (np.arange(total) - start[rep]) / (cnt[rep] - 1)
    return (v0[rep] + t[:, None] * seg[rep]).astype(np.float64)


def _soup_mesh(tris: np.ndarray, uv3: np.ndarray | None, material) -> trimesh.Trimesh:  # noqa: ANN001
    """Standalone mesh from a triangle soup (vertices duplicated per triangle),
    carrying per-corner UVs + the ORIGINAL material so colour lookups go through
    the same texture path as on the whole object."""
    verts = tris.reshape(-1, 3)
    faces = np.arange(len(verts), dtype=np.int64).reshape(-1, 3)
    visual = None
    if uv3 is not None and material is not None:
        visual = trimesh.visual.TextureVisuals(uv=uv3.reshape(-1, 2), material=material)
    return trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=False)


def _spacing_bands(
    geom: trimesh.Trimesh, spacing: float, boost: float = _FEATURE_BOOST
) -> list[tuple[trimesh.Trimesh, float]]:
    """LEGACY feature-adaptive band split (only used when `_FEATURE_REFINE` is on):
    split one object into `(mesh, spacing)` bands by distance to its feature edges,
    refining crease/boundary neighbourhoods up to `boost`× finer. Two-level labeling
    for speed; falls back to a single base-spacing band when the mesh has no feature
    edges or its triangles are degenerate."""
    edge_pts = _feature_edge_points(geom, step=spacing * 0.5)
    if edge_pts is None:
        return [(geom, spacing)]

    tris = np.asarray(geom.triangles, dtype=np.float64)
    uv = getattr(getattr(geom, "visual", None), "uv", None)
    material = getattr(getattr(geom, "visual", None), "material", None)
    uv3 = None
    if uv is not None and len(uv) == len(geom.vertices):
        uv3 = np.asarray(uv, dtype=np.float64)[np.asarray(geom.faces)]
    keep = _valid_tri_mask(tris)
    tris = tris[keep]
    uv3 = uv3[keep] if uv3 is not None else None
    if not len(tris):
        return [(geom, spacing)]

    tree = cKDTree(edge_pts)
    tris_c, uv3_c = _subdivide_edges(tris, spacing * _CLASSIFY_COARSE, uv3)
    cent = tris_c.mean(axis=1)
    d_c = tree.query(cent)[0]
    circum = np.linalg.norm(tris_c - cent[:, None, :], axis=2).max(axis=1)
    flat = (d_c - circum) >= spacing

    base_tris: list[np.ndarray] = [tris_c[flat]]
    base_uv: list[np.ndarray] | None = [uv3_c[flat]] if uv3_c is not None else None
    finer: list[tuple[float, np.ndarray, np.ndarray | None]] = []

    near = ~flat
    if near.any():
        tris_f, uv3_f = _subdivide_edges(
            tris_c[near], spacing, uv3_c[near] if uv3_c is not None else None
        )
        d = tree.query(tris_f.mean(axis=1))[0]
        factor = spacing / np.clip(d, spacing / boost, spacing)
        level = np.ceil(np.log2(np.maximum(factor, 1.0)) - 1e-9)
        f = np.minimum(2.0 ** level, boost)
        for lf in np.unique(f):
            m = f == lf
            if lf == 1.0:
                base_tris.append(tris_f[m])
                if base_uv is not None and uv3_f is not None:
                    base_uv.append(uv3_f[m])
            else:
                finer.append(
                    (float(lf), tris_f[m], uv3_f[m] if uv3_f is not None else None)
                )

    bands: list[tuple[trimesh.Trimesh, float]] = []
    bt = np.concatenate(base_tris, axis=0)
    if len(bt):
        bu = np.concatenate(base_uv, axis=0) if base_uv is not None else None
        bands.append((_soup_mesh(bt, bu, material), spacing))
    for lf, band_tris, band_uv in finer:
        bands.append((_soup_mesh(band_tris, band_uv, material), spacing / lf))
    return bands or [(geom, spacing)]


def _thin_band(points: np.ndarray, spacing: float) -> np.ndarray:
    """Lattice-hash thin candidate darts → kept indices. Bins each candidate into a
    cell of edge `_THIN_CELL_FACTOR·spacing` and keeps the ONE candidate nearest the
    cell centre. O(N) integer passes — no KD-tree, no pair table. A single winner
    per cell matches the old sampler's effective handling of coincident opposite
    faces and avoids the winding-driven doubling a pre-orientation facing split
    causes (facing handling happens later, in the coarsener, POST-orientation)."""
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    cs = spacing * _THIN_CELL_FACTOR
    cell = np.floor(points / cs).astype(np.int64)
    loc = cell - cell.min(axis=0)
    nx = int(loc[:, 0].max()) + 1
    ny = int(loc[:, 1].max()) + 1
    nz = int(loc[:, 2].max()) + 1
    if nx * ny * nz < (1 << 61):
        gid = (loc[:, 0] * ny + loc[:, 1]) * nz + loc[:, 2]
    else:  # astronomically large single object — fall back to row-unique
        _, gid = np.unique(loc, axis=0, return_inverse=True)
        gid = gid.reshape(-1)
    center = (cell.astype(np.float64) + 0.5) * cs
    d2 = ((points - center) ** 2).sum(axis=1)
    order = np.lexsort((d2, gid))
    gid_s = gid[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = gid_s[1:] != gid_s[:-1]
    return order[first]


def surfel_colors(
    geom: trimesh.Trimesh, points: np.ndarray, face_idx: np.ndarray
) -> np.ndarray:
    """Per-surfel RGBA in [0,1]. With a readable base-color texture, samples it at
    each surfel's barycentric-interpolated UV — the EXACT texel at that spot, full
    resolution. Otherwise the per-face mean of the per-vertex colours."""
    faces = np.asarray(geom.faces[face_idx])
    visual = getattr(geom, "visual", None)
    material = getattr(visual, "material", None)
    image = getattr(material, "baseColorTexture", None)
    uv = getattr(visual, "uv", None)
    if image is not None and uv is not None:
        try:
            tris = np.asarray(geom.triangles[face_idx])
            bary = trimesh.triangles.points_to_barycentric(tris, points)
            face_uv = np.asarray(uv)[faces]
            uvs = np.einsum("sj,sjk->sk", bary, face_uv)
            cols = np.asarray(
                trimesh.visual.color.uv_to_color(uvs, image), dtype=np.float32
            )
            cols = cols / 255.0
            factor = getattr(material, "baseColorFactor", None)
            if factor is not None:
                fa = np.asarray(factor, dtype=np.float32).reshape(-1)
                if fa.size >= 3:
                    if fa.max() > 1.0:
                        fa = fa / 255.0
                    cols[:, :3] *= fa[:3]
                    if fa.size >= 4:
                        cols[:, 3] *= fa[3]
            return cols
        except Exception:
            pass
    return _vertex_colors(geom)[faces].mean(axis=1)


def _orient_and_cull(
    positions: np.ndarray, normals: np.ndarray, fs: FreeSpace
) -> tuple[np.ndarray, np.ndarray]:
    """Use the Stage-2 empty query to (1) flip each normal toward the open side,
    and (2) mark surfels to keep. A side is "open" if any probe offset along
    ±normal lands in empty (viewable) air. Returns (keep_mask, oriented_normals).

    Deliberately EMPTY, not clearance-filtered: a surfel beside a thin gap is
    visible through it even though no camera fits. Buried surfels (neither side
    open) are dropped; free-standing thin surfaces (both sides open) keep their
    original normal."""
    offsets = np.array([1.0, 1.5, 2.0], dtype=np.float64) * fs.pitch
    plus_open = np.zeros(len(positions), dtype=bool)
    minus_open = np.zeros(len(positions), dtype=bool)
    for d in offsets:
        plus_open |= fs.empty_at(positions + normals * d)
        minus_open |= fs.empty_at(positions - normals * d)
    oriented = normals.copy()
    flip = minus_open & ~plus_open
    oriented[flip] = -oriented[flip]
    return plus_open | minus_open, oriented


def _group_ids(cells: np.ndarray) -> np.ndarray:
    """(N,) int64 group id per (N,3) integer cell row (offset to non-negative and
    packed to one int64 when the local lattice fits, else row-unique)."""
    loc = cells - cells.min(axis=0)
    nx = int(loc[:, 0].max()) + 1
    ny = int(loc[:, 1].max()) + 1
    nz = int(loc[:, 2].max()) + 1
    if nx * ny * nz < (1 << 61):
        return (loc[:, 0] * ny + loc[:, 1]) * nz + loc[:, 2]
    _, gid = np.unique(loc, axis=0, return_inverse=True)
    return gid.reshape(-1)


def _coarsen(
    pos: np.ndarray,
    nrm: np.ndarray,
    col: np.ndarray,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Geometry-driven bottom-up merging of surviving base surfels (module
    docstring): per octave level k, group nodes by parent cell of edge `2^k·s`,
    split each group into two facing SIDES against its principal normal axis
    (normals are already air-oriented), and collapse a side into ONE surfel when it
    is FILLED, SMOOTH and COPLANAR. Returns `(positions, normals, colors, radii)`
    with radii = `_RADIUS_FRAC` × each node's cell size. Deterministic; O(N) integer
    group passes per level."""
    n = len(pos)
    radii = np.full(n, _RADIUS_FRAC * spacing, dtype=np.float32)
    if n < 4 or _COARSEN_OCTAVES <= 0:
        return pos, nrm, col, radii

    pos = pos.astype(np.float64)
    nrm = nrm.astype(np.float64)
    col = col.astype(np.float64)
    count = np.ones(n, dtype=np.float64)  # accumulated base-cell count per node
    offset_tol = _MERGE_OFFSET_FRAC * spacing

    for k in range(1, _COARSEN_OCTAVES + 1):
        L = spacing * (2.0 ** k)
        cell = np.floor(pos / L).astype(np.int64)
        gid = _group_ids(cell)
        _, inv = np.unique(gid, return_inverse=True)
        inv = inv.reshape(-1)
        g = int(inv.max()) + 1

        # Principal normal axis per group: top eigenvector of Σ n·nᵀ (sign-free,
        # so mixed winding/orientation across the group cannot hide a two-sided
        # cell). Six bincounts build the symmetric tensors — no (N,3,3) temp,
        # no unbuffered np.add.at.
        tens = np.empty((g, 3, 3))
        for i in range(3):
            for j in range(i, 3):
                tij = np.bincount(inv, nrm[:, i] * nrm[:, j], g)
                tens[:, i, j] = tij
                tens[:, j, i] = tij
        _, v = np.linalg.eigh(tens)
        axis = v[:, :, 2]  # eigenvector of the largest eigenvalue
        side = (np.einsum("nj,nj->n", nrm, axis[inv]) >= 0.0).astype(np.int64)
        sid = inv * 2 + side
        ns = 2 * g

        # Per-(group, side) reductions.
        scnt = np.bincount(sid, minlength=ns)                     # member nodes
        sbase = np.bincount(sid, count, ns)                       # base cells
        wsum = np.maximum(sbase, 1e-12)
        mpos = np.stack(
            [np.bincount(sid, count * pos[:, i], ns) for i in range(3)], axis=1
        ) / wsum[:, None]
        mnrm = np.stack(
            [np.bincount(sid, count * nrm[:, i], ns) for i in range(3)], axis=1
        )
        mlen = np.linalg.norm(mnrm, axis=1)
        mdir = mnrm / np.maximum(mlen, 1e-12)[:, None]
        mcol = np.stack(
            [np.bincount(sid, count * col[:, i], ns) for i in range(4)], axis=1
        ) / wsum[:, None]

        # COPLANAR: spread of member projections onto the side's mean normal,
        # reduced per side via one sort + reduceat (buffered, unlike ufunc.at).
        off = np.einsum("nj,nj->n", pos - mpos[sid], mdir[sid])
        order = np.argsort(sid, kind="stable")
        sid_s = sid[order]
        starts = np.ones(len(order), dtype=bool)
        starts[1:] = sid_s[1:] != sid_s[:-1]
        idx = np.flatnonzero(starts)
        present = sid_s[idx]
        omin = np.full(ns, np.inf)
        omax = np.full(ns, -np.inf)
        omin[present] = np.minimum.reduceat(off[order], idx)
        omax[present] = np.maximum.reduceat(off[order], idx)

        # SMOOTH: oriented-normal agreement (mean of unit normals stays ~unit).
        nspread = 1.0 - mlen / np.maximum(sbase, 1e-12)

        merge = (
            (scnt >= 2)
            & (sbase >= _MERGE_FILL * (4.0 ** k))
            & (nspread <= _MERGE_FLAT)
            & ((omax - omin) <= offset_tol)
        )
        if not merge.any():
            break

        drop = merge[sid]  # members absorbed into a merged node
        keep_pos = pos[~drop]
        keep_nrm = nrm[~drop]
        keep_col = col[~drop]
        keep_cnt = count[~drop]
        keep_rad = radii[~drop]

        m = np.nonzero(merge)[0]
        new_pos = mpos[m]
        new_nrm = mdir[m]
        new_col = mcol[m]
        new_cnt = sbase[m]
        new_rad = np.full(len(m), _RADIUS_FRAC * L, dtype=np.float32)

        pos = np.concatenate([keep_pos, new_pos], axis=0)
        nrm = np.concatenate([keep_nrm, new_nrm], axis=0)
        col = np.concatenate([keep_col, new_col], axis=0)
        count = np.concatenate([keep_cnt, new_cnt], axis=0)
        radii = np.concatenate([keep_rad, new_rad], axis=0)

    return (
        pos.astype(np.float32),
        nrm.astype(np.float32),
        col.astype(np.float32),
        radii.astype(np.float32),
    )


def _band_seed(mesh: trimesh.Trimesh, spacing: float, salt: int) -> int:
    """Deterministic per-(sub)mesh seed from stable geometry facts + a slab salt,
    so a mesh resamples identically for any worker count / run."""
    a = int(round(float(mesh.area) * 1000.0))
    return (len(mesh.faces) * 2654435761 + a + int(spacing * 1e6) + salt * 40503) & 0x7FFFFFFF


def _band_darts(
    mesh: trimesh.Trimesh, spacing: float, opaque: bool, fs: FreeSpace, seed: int
) -> tuple[dict[str, np.ndarray] | None, int]:
    """One (sub)mesh → surfels: seeded darts, lattice thin, orient+cull against
    `fs.empty_at`, colour survivors (exact texels), then geometry-driven COARSEN.
    Returns `(per-surfel arrays | None, base-surfel count before coarsening)`."""
    if len(mesh.faces) == 0 or mesh.area <= 0:
        return None, 0
    budget = int(mesh.area / (spacing * spacing) * _THIN_CANDIDATES) + 8
    try:
        points, face_idx = trimesh.sample.sample_surface(mesh, budget, seed=seed)
    except Exception:
        return None, 0
    points = np.asarray(points, dtype=np.float64)
    face_idx = np.asarray(face_idx)
    if len(points) == 0:
        return None, 0

    keep_idx = _thin_band(points, spacing)
    if len(keep_idx) == 0:
        return None, 0
    points = points[keep_idx]
    face_idx = face_idx[keep_idx]
    thinned = len(points)

    normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float64)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    normals = normals / lens

    pos32 = points.astype(np.float32)
    nrm32 = normals.astype(np.float32)
    keep, oriented = _orient_and_cull(pos32, nrm32, fs)
    if not keep.any():
        return None, thinned

    colors = surfel_colors(mesh, points[keep], face_idx[keep]).astype(np.float32)
    if opaque:
        colors[:, 3] = 1.0

    cpos, cnrm, ccol, crad = _coarsen(
        pos32[keep], oriented[keep].astype(np.float32), colors, spacing
    )
    return {
        "position": cpos,
        "normal": cnrm,
        "color": ccol,
        "radius": crad,
    }, thinned


def _sample_band(
    mesh: trimesh.Trimesh, spacing: float, opaque: bool, fs: FreeSpace
) -> tuple[dict[str, np.ndarray] | None, int]:
    """Sample one mesh at one spacing. Small meshes go through `_band_darts`
    directly; a mesh whose dart budget exceeds `_CHUNK_MAX_CANDIDATES` is
    partitioned into `_CHUNK_EXTENT`-metre spatial slabs (by triangle centroid) and
    sampled slab by slab, so transient memory is bounded regardless of size."""
    if len(mesh.faces) == 0 or mesh.area <= 0:
        return None, 0
    budget = int(mesh.area / (spacing * spacing) * _THIN_CANDIDATES) + 8
    if budget <= _CHUNK_MAX_CANDIDATES:
        return _band_darts(mesh, spacing, opaque, fs, _band_seed(mesh, spacing, 0))

    tris = np.asarray(mesh.triangles, dtype=np.float64)
    uv = getattr(getattr(mesh, "visual", None), "uv", None)
    material = getattr(getattr(mesh, "visual", None), "material", None)
    # Index UVs BY FACE — correct for a raw geom's per-vertex UVs and a soup's
    # per-corner UVs alike (a soup's faces are arange, so this equals a reshape).
    uv3 = (
        np.asarray(uv, dtype=np.float64)[np.asarray(mesh.faces)]
        if uv is not None and len(uv) == len(mesh.vertices)
        else None
    )
    cent = tris.mean(axis=1)
    slab = np.floor(cent / _CHUNK_EXTENT).astype(np.int64)
    slab -= slab.min(axis=0)
    sy = int(slab[:, 1].max()) + 1
    sz = int(slab[:, 2].max()) + 1
    skey = (slab[:, 0] * sy + slab[:, 1]) * sz + slab[:, 2]
    parts: list[dict[str, np.ndarray]] = []
    thinned = 0
    for si, sk in enumerate(np.unique(skey)):
        m = skey == sk
        sub = _soup_mesh(tris[m], uv3[m] if uv3 is not None else None, material)
        part, n = _band_darts(sub, spacing, opaque, fs, _band_seed(sub, spacing, si + 1))
        thinned += n
        if part is not None:
            parts.append(part)
    if not parts:
        return None, thinned
    if len(parts) == 1:
        return parts[0], thinned
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}, thinned


def _scene_scale(cand_clear: np.ndarray) -> float:
    """Scene-scale factor k from the Stage-2 candidate clearances (open-sightline
    statistic): `clamp(P_pctl(cand_clear) / _SIGHTLINE_REF_M, *_SCENE_K_CLAMP)`.
    Falls back to 1.0 (no scaling) when candidates are too sparse to estimate."""
    if cand_clear is None or len(cand_clear) < 64:
        return 1.0
    p = float(np.percentile(np.asarray(cand_clear, dtype=np.float64), _SIGHTLINE_PCTL))
    return float(np.clip(p / _SIGHTLINE_REF_M, *_SCENE_K_CLAMP))


def _object_spacing(geom: trimesh.Trimesh, base: float, k: float) -> float:
    """Per-object sample spacing: `base · clamp(obj_diag / _VIEW_REF_M, 1, k)`.
    Small props stay at `base`; large-area surface reaches the scene ceiling
    `base·k`; mid-size objects scale with their own diagonal (framed from ~their
    own size away)."""
    b = np.asarray(geom.bounds, dtype=np.float64)
    diag = float(np.linalg.norm(b[1] - b[0]))
    return base * float(np.clip(diag / _VIEW_REF_M, 1.0, k))


def _sample_object(
    geom: trimesh.Trimesh, base: float, k: float, fs: FreeSpace
) -> tuple[dict[str, np.ndarray] | None, int]:
    """Sample one placed mesh into surfels at its scene-scale-adaptive spacing
    (`_object_spacing`), then geometry-coarsen. With `_FEATURE_REFINE` on, split
    into legacy feature bands first. Returns `(concatenated per-surfel arrays for
    the VISIBLE surfels | None, total base-surfel count)`. Opacity is forced to 1
    unless the material is genuinely BLEND/MASK."""
    if len(geom.faces) == 0 or geom.area <= 0:
        return None, 0
    spacing = _object_spacing(geom, base, k)
    opaque = _alpha_mode(geom) not in _TRANSPARENT_ALPHA_MODES
    bands = _spacing_bands(geom, spacing) if _FEATURE_REFINE else [(geom, spacing)]
    parts: list[dict[str, np.ndarray]] = []
    sampled = 0
    for band_mesh, band_spacing in bands:
        band, n = _sample_band(band_mesh, band_spacing, opaque, fs)
        sampled += n
        if band is not None:
            parts.append(band)
    if not parts:
        return None, sampled
    if len(parts) == 1:
        return parts[0], sampled
    return {
        k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]
    }, sampled


def _encode_ply(
    positions: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    radii: np.ndarray,
    out_path: Path,
    representation: str = "2dgs",
) -> None:
    """Write surfels as a binary Gaussian `.ply` (SH degree 0). `2dgs` = two tangent
    scales (no thickness); `3dgs` appends a thin third scale. Shared fields: xyz,
    normal, f_dc_0..2, opacity, scales, rot_0..3."""
    n = positions.shape[0]
    quats = _align_quaternions(normals).astype(np.float32)

    rgb = np.clip(colors[:, :3], 0.0, 1.0)
    f_dc = ((rgb - 0.5) / _SH_C0).astype(np.float32)

    alpha = np.clip(colors[:, 3], _ALPHA_CLAMP[0], _ALPHA_CLAMP[1])
    opacity = np.log(alpha / (1.0 - alpha)).astype(np.float32)

    if representation == "3dgs":
        scale = np.stack([radii, radii, radii * _FLATNESS], axis=1)
        log_scale = np.log(np.maximum(scale, 1e-9)).astype(np.float32)
        scale_cols = [log_scale[:, 0], log_scale[:, 1], log_scale[:, 2]]
        scale_props = (
            "property float scale_0\n" "property float scale_1\n" "property float scale_2\n"
        )
    else:  # "2dgs": two in-plane tangent scales, no thickness axis
        log_scale = np.log(np.maximum(np.stack([radii, radii], axis=1), 1e-9)).astype(np.float32)
        scale_cols = [log_scale[:, 0], log_scale[:, 1]]
        scale_props = "property float scale_0\n" "property float scale_1\n"

    cols = [
        positions[:, 0], positions[:, 1], positions[:, 2],
        normals[:, 0], normals[:, 1], normals[:, 2],
        f_dc[:, 0], f_dc[:, 1], f_dc[:, 2],
        opacity,
        *scale_cols,
        quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3],
    ]
    data = np.stack(cols, axis=1).astype("<f4")

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n" "property float y\n" "property float z\n"
        "property float nx\n" "property float ny\n" "property float nz\n"
        "property float f_dc_0\n" "property float f_dc_1\n" "property float f_dc_2\n"
        "property float opacity\n"
        + scale_props
        + "property float rot_0\n" "property float rot_1\n" "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    tmp.replace(out_path)


# --- per-object work units (serial and process-pool) ---------------------------
# The grid is loaded once per process (spawn workers cache it) via stage2's loader;
# the skin bitmask sidecar is memory-mapped, so every worker probing the same grid
# shares ONE physical copy through the OS page cache.

_TASK_FS: tuple[str, FreeSpace] | None = None


def _task_free_space(path: str) -> FreeSpace:
    """Process-wide cached free-space grid (spawn workers load it once)."""
    global _TASK_FS
    cached = _TASK_FS
    if cached is None or cached[0] != path:
        cached = (path, load_free_space(Path(path)))
        _TASK_FS = cached
    return cached[1]


def _sample_object_task(
    node_id: str, glb_path: str, base: float, k: float, freespace_path: str
) -> tuple[str, dict[str, np.ndarray] | None, float, int, str | None]:
    """Sample ONE placed GLB → `(node_id, visible-surfel arrays | None, surface
    area, base-surfel count, error | None)`. The process-pool work unit; results
    merge by id order. A failing mesh reports its error instead of raising."""
    try:
        fs = _task_free_space(freespace_path)
        parts: list[dict[str, np.ndarray]] = []
        area = 0.0
        sampled = 0
        for geom in load_geoms(Path(glb_path)):
            part, n = _sample_object(geom, base, k, fs)
            sampled += n
            if part is None:
                continue
            area += float(geom.area)
            parts.append(part)
        if not parts:
            return node_id, None, area, sampled, None
        if len(parts) == 1:
            return node_id, parts[0], area, sampled, None
        merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
        return node_id, merged, area, sampled, None
    except Exception as exc:
        return node_id, None, 0.0, 0, f"{type(exc).__name__}: {exc}"


def _make_pool(workers: int):  # noqa: ANN202 - ProcessPoolExecutor | None
    """A SPAWN-context process pool, or None when subprocesses are forbidden."""
    if workers <= 1:
        return None
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    try:
        ctx = multiprocessing.get_context("spawn")
        return ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
    except Exception:
        logging.getLogger(__name__).warning(
            "stage3: process pool unavailable — sampling serially"
        )
        return None


def _iter_sampled(
    ids: list[str], raw_dir: Path, base: float, k: float, freespace_path: str, pool
):  # noqa: ANN201, ANN001 - yields _sample_object_task results
    """Yield per-object sampling results, serially (`pool=None`) or from the given
    process pool. Object-level parallelism is the grain; lattice thinning bounds
    each object's memory (no pair table), and huge single meshes chunk internally
    (`_sample_band`), so no object OOMs a worker."""
    if pool is None:
        for node_id in ids:
            yield _sample_object_task(
                node_id, str(raw_dir / f"{node_id}.glb"), base, k, freespace_path
            )
        return
    from concurrent.futures import as_completed

    with pool:
        futures = [
            pool.submit(
                _sample_object_task,
                node_id,
                str(raw_dir / f"{node_id}.glb"),
                base,
                k,
                freespace_path,
            )
            for node_id in ids
        ]
        for fut in as_completed(futures):
            yield fut.result()


def sample_cell(
    *,
    run: str,
    slot: str,
    model: str,
    raw_dir: Path,
    freespace_path: Path,
    out_path: Path,
    params: SampleParams = SampleParams(),
    progress: ProgressCb | None = None,
    workers: int = 0,
) -> dict[str, Any]:
    """Sample every placed mesh in `raw_dir` into a Gaussian splat written to
    `out_path`, consuming the Stage-2 free-space grid at `freespace_path` to orient
    normals + cull hidden faces. `workers` parallelizes the per-object pass (0 =
    auto: min(cores, 8); 1 = serial). Sampling is SEEDED, so the output is
    deterministic for any worker count; results merge in id order. The summary's
    `sampled` counts BASE surfels (pre-coarsening); `splats` counts the emitted
    (post-coarsening) cloud."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"placed-mesh dir not found: {raw_dir}")
    if not Path(freespace_path).is_file():
        raise FileNotFoundError(f"free-space grid not found: {freespace_path} (run Stage 2)")
    ids = placed_object_ids(raw_dir)
    if not ids:
        raise FileNotFoundError(f"no placed meshes in {raw_dir}")

    fs_path = str(freespace_path)
    probe = _task_free_space(fs_path)
    base = params.spacing
    scene_k = _scene_scale(probe.cand_clear)

    if workers <= 0:
        workers = min(os.cpu_count() or 1, 8)
    if not isinstance(probe.skin_empty, np.memmap):
        mem_cap = max(1, int(_WORKER_GRID_BUDGET // max(probe.skin_empty.nbytes, 1)))
        workers = min(workers, mem_cap)
    workers = max(1, min(workers, len(ids)))
    pool = _make_pool(workers)
    if pool is None:
        workers = 1  # report the truth when the environment degraded us

    total = len(ids)
    if progress is not None:
        progress(0, total, "")

    t0 = time.perf_counter()
    results: dict[str, tuple[dict[str, np.ndarray] | None, float, int, str | None]] = {}
    done = 0
    for node_id, part, area, n_sampled, err in _iter_sampled(
        ids, raw_dir, base, scene_k, fs_path, pool
    ):
        done += 1
        results[node_id] = (part, area, n_sampled, err)
        if progress is not None:
            progress(done, total, node_id)
    sample_s = time.perf_counter() - t0

    pos_parts: list[np.ndarray] = []
    nrm_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    rad_parts: list[np.ndarray] = []
    warnings: list[str] = []
    objects_sampled = 0
    total_area = 0.0
    sampled = 0
    for node_id in ids:
        part, area, n_sampled, err = results[node_id]
        if err is not None:
            warnings.append(f"{node_id}: failed to sample ({err})")
            continue
        objects_sampled += 1
        total_area += area
        sampled += n_sampled
        if part is None:
            continue
        pos_parts.append(part["position"])
        nrm_parts.append(part["normal"])
        col_parts.append(part["color"])
        rad_parts.append(part["radius"])

    if not pos_parts:
        if sampled > 0:
            raise RuntimeError("all surfels culled as hidden (check the free-space grid)")
        raise RuntimeError("no surfels sampled (every mesh failed or was empty)")

    positions = np.concatenate(pos_parts, axis=0)
    normals = np.concatenate(nrm_parts, axis=0)
    colors = np.concatenate(col_parts, axis=0)
    radii = np.concatenate(rad_parts, axis=0)

    t1 = time.perf_counter()
    _encode_ply(positions, normals, colors, radii, out_path, params.representation)
    encode_s = time.perf_counter() - t1

    aabb_min = positions.min(axis=0).tolist()
    aabb_max = positions.max(axis=0).tolist()
    return {
        "run": run,
        "slot": slot,
        "model": model,
        "splats": int(positions.shape[0]),
        "sampled": int(sampled),
        "culled_hidden": max(0, int(sampled - positions.shape[0])),
        "objects_sampled": objects_sampled,
        "objects_total": total,
        "spacing": base,
        "scene_scale": round(scene_k, 3),
        "scene_spacing": round(base * scene_k, 4),
        "total_area": round(total_area, 2),
        "workers": workers,
        "timings": {"sample_s": round(sample_s, 2), "encode_s": round(encode_s, 2)},
        "params": params.as_summary(),
        "scene_aabb": {"min": aabb_min, "max": aabb_max},
        "warnings": warnings,
        "bytes": out_path.stat().st_size,
    }
