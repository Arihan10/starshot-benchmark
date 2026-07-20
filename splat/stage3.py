"""Stage 3 — Surfel sampler (mesh → Gaussian splat).

Turns a cell's placed meshes into a **pre-fine-tuning Gaussian splat**:
a cloud of flat, surface-aligned 2D Gaussians (surfels), written as a **2DGS `.ply`**
(two tangent scales, no thickness) that 2DGS-capable web viewers (mkkellogg, PlayCanvas
v2.16+) read and the Stage-6 gsplat **2DGS** fine-tune inits from. A `3dgs` mode appends
a thin third scale for 3DGS-only viewers / SOG-SPZ compression.

Consumes the **Stage-2 free-space grid** (per Option A the free-space foundation runs
first). Two things use it:
  * **Normal orientation (fixes unreliable TRELLIS winding):** each surfel's normal is
    flipped to point toward FREE space (the Stage-2 free mask), rather than trusting
    the mesh's face winding — so disks face the viewer regardless of authoring.
  * **Hidden-face culling:** a surfel with no free space on EITHER side is
    buried (a solid's interior, or the seam between two abutting objects) and is
    dropped — it would never be seen, so it's wasted budget and a floater seed.

ONE KNOB — `detail`. It is a density multiplier around the calibrated default
look: sample spacing = `_BASE_SPACING / sqrt(detail)`, so detail=1 reproduces
    the default resolution (~3 cm between surfels at base, refined to ~0.75 cm
near feature edges and relaxed to ~6 cm on flat, uniformly-coloured regions),
detail=2 doubles the surfel density, detail=0.5 halves
it. Everything that used to be a knob is a derived law or a fixed constant:
  * spacing      — the knob itself (count follows: N ∝ area × detail);
  * disk radius  — `_RADIUS_FRAC` × the local sample spacing (disks tile);
  * flatness     — `_FLATNESS`, used ONLY by the `3dgs` compat export (2D
    surfels have no thickness; 3DGS-only viewers/compressors need a nonzero
    third scale, so it writes radius × 0.1);
  * feature refinement — `_FEATURE_BOOST`, always on (see below);
  * flat coarsening    — `_COARSEN_MAX`, always on (the refinement's mirror:
    flat + uniform regions sample coarser; see below);
  * hidden-face culling — always on (pure win, not a fidelity choice).

Sampling: per-object GRID-THINNED blue noise in FEATURE-ADAPTIVE BANDS.
Candidates are seeded area-weighted surface points; a lattice of
~`_THIN_CELL_FACTOR`×spacing cells keeps ONE winner per (cell, normal-facing
bucket), then a same-facing-only minimum-distance pass removes near-duplicates
(`_thin_band`). Equivalent evenness/density to the old oversample-and-eliminate
Poisson sampler (calibrated to the same yield) at a fraction of the memory —
no KD-trees, no all-pairs arrays — and two deliberate improvements: THIN
GEOMETRY keeps both faces (opposite-facing points never eliminate each other,
where the old 3D-distance elimination silently starved double-sided sheets),
and sampling is SEEDED per (object, geometry, band), so a cell resamples to
the byte-identical cloud for any worker count.

Detail is measured ONCE, from the mesh itself (its feature edges: creases
sharper than `_CREASE_ANGLE` + open boundaries — full-fidelity signals, not
statistics of an already-sampled proxy): flat interiors sample at the knob
spacing exactly, while crease/boundary neighbourhoods — thin members, frames,
slats, corners, which by construction lie within a base-spacing of a feature
edge — sample up to `_FEATURE_BOOST`× finer (boost²× the density). Purely
additive: flats are unchanged and detail rides on top, so the knob keeps its
meaning. Band LABELING is two-level (coarse split → conservative flat test →
fine split only near edges), which changes the cost, not the labels
(`_spacing_bands`). Downstream, the cloud's local density IS the detail field:
Stage 4 reads it (local sample spacing) instead of re-detecting detail from
the cloud, and Stage 6's densification starts from capacity that already
exists where the close-up cameras will look.

PIPELINE PER OBJECT (parallel across objects — `_iter_sampled`, spawn pool,
id-ordered merge): load → band labeling → per band: sample, radii, ORIENT +
CULL against the Stage-2 EMPTY mask, then color ONLY the visible surfels
(buried seams never pay texture lookups).

Each surfel: position (world; placement baked into the vertices), rotation
(quaternion aligning +Z to the oriented normal), scale (r, r, ~0), color (the
EXACT base-color texel at the surfel's UV — see `surfel_colors` for why the
footprint-averaged variant was reverted), opacity (that texel's alpha,
honouring `alphaMode`). Stored SH is degree 0 (`f_dc`) — unlit /
view-independent.

Pure library: `sample_cell` takes explicit paths; the server resolves a cell to
them and passes the Stage-2 `freespace.npz`. Meshes are consumed AS-IS through
`splat.assets.load_geoms` — vanilla glTF via trimesh, KTX2/Meshopt sets via the
in-process decoder (geometry natively; KTX2 base color BasisU-transcoded), so
albedo sampling behaves identically on every encoding.
"""

from __future__ import annotations

import logging
import os
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from splat.assets import load_geoms
from splat.stage2 import FreeSpace, _abs_encode, _subdivide_edges, _valid_tri_mask

# trimesh logs per-file load noise; silence it for whole-cell runs.
logging.getLogger("trimesh").setLevel(logging.ERROR)

# The Gaussian cloud filename written under a cell's `splat/` dir (a 3DGS `.ply`).
CLOUD_NAME = "cloud.ply"

# SH degree-0 basis constant: colour c in [0,1] maps to f_dc = (c - 0.5) / C0
# (the 3DGS convention every viewer inverts as colour = 0.5 + C0 * f_dc).
_SH_C0 = 0.28209479177387814

# The calibrated default resolution: the flat-interior sample spacing (m) at
# detail = 1, MEASURED so detail=1 reproduces the accepted default cloud's
# budget (hotel-room benchmark: ~2.1M sampled / ~1.6M kept / ~100 MB). The old
# defaults hit that budget with per-object adaptive coarsening (walls ~4 cm,
# props ~1-2 cm); the one-knob sampler is uniform, and 2.95 cm uniform lands
# the same total with the budget spread evenly (walls slightly finer than
# before, tiny props slightly coarser, feature bands still at spacing/4).
# Calibrated against the FIXED asset loader (correct meshopt index decode).
_BASE_SPACING = 0.0295
# spacing = _BASE_SPACING / sqrt(detail): detail multiplies DENSITY (surfels
# per m²), the intuitive direction (2 = twice the surfels, 0.5 = half).
DEFAULT_DETAIL = 1.0
_DETAIL_CLAMP = (0.05, 16.0)
_SPACING_CLAMP = (0.004, 0.25)

# Derived-value constants (were knobs; see module docstring for their laws).
# Disk σ = _RADIUS_FRAC × local sample spacing — disks tile without gaps at
# blue-noise spacing; Stage 6 fine-tunes scales from there.
_RADIUS_FRAC = 0.9
# 3dgs-mode only: the fake thickness (third scale = radius × _FLATNESS) that
# 3DGS-only viewers / SOG-SPZ compressors need (a true 0 breaks log-scale
# encoding). The default 2dgs output has no thickness axis at all.
_FLATNESS = 0.1
# Feature-adaptive density: sampling within one base-spacing of a FEATURE EDGE
# (a crease sharper than _CREASE_ANGLE, or an open boundary) is refined in
# octave bands down to base/_FEATURE_BOOST. 25° cleanly separates man-made
# creases (box corners, frames, rail edges ~60-90°) from the small per-edge
# angles of dense smooth meshes (~10-15°), so smooth blobs stay single-band.
_FEATURE_BOOST = 4.0
_CREASE_ANGLE = np.deg2rad(25.0)
_EDGE_PT_CAP = 400_000  # stride a pathological crease soup instead of exploding
# Band labeling's first-pass split limit (× spacing). Pieces PROVABLY at base-
# band distance stop here (16× fewer pieces than a full split on big flats);
# only near-edge pieces refine to `spacing`. 4× balances the two costs: higher
# skips more interior but widens the must-refine strip (the conservative test
# subtracts the piece circumradius, which grows with the limit).
_CLASSIFY_COARSE = 4.0

# --- grid-thinning sampler knobs (all internal; calibrated together so the
# kept density matches the old Poisson eliminator's ~0.75/spacing² yield) -----
# Thinning lattice cell edge, × spacing: one winner survives per (cell, facing
# bucket). Bigger cells = sparser; 1.25 + the dedup below lands on the old
# yield (verified against the old sampler on synthetic planes and the hotel
# benchmark cell).
_THIN_CELL_FACTOR = 1.25
# Candidate budget, × area/spacing² (≈ 2.9 candidates per surface cell → ~94%
# of cells occupied). The old sampler generated 6×/spacing² internally.
_THIN_CANDIDATES = 2.4
# Near-duplicate floor, × spacing: SAME-FACING winners closer than this lose
# one point (the old eliminator's hard floor was 1.0 × spacing; radii clamp to
# the band median, so the softer floor shows up as a few smaller disks, not
# holes). OPPOSITE-facing points are never duplicates — that's what keeps both
# faces of thin sheets alive.
_THIN_MIN_DIST_FRAC = 0.7
# Two winners count as "same-facing" when their unit normals' dot exceeds
# this: parallel faces (sheet sides, dot −1) and perpendicular crease faces
# (dot 0) both survive; only genuinely co-oriented near-pairs dedup.
_THIN_DUP_NORMAL_DOT = 0.5
# The 13 lexicographically-positive neighbour offsets: enumerating pairs from
# one side of each cell boundary visits every unordered cross-cell pair once.
_HALF_OFFSETS = np.array(
    [o for o in np.ndindex(3, 3, 3)], dtype=np.int64
).__sub__(1)
_HALF_OFFSETS = _HALF_OFFSETS[
    np.lexsort((_HALF_OFFSETS[:, 2], _HALF_OFFSETS[:, 1], _HALF_OFFSETS[:, 0]))
][14:]  # strictly greater than (0,0,0) in lexicographic order
# Feature-adaptive COARSENING — the mirror of the refinement above. Flat pieces
# far from every feature edge sample up to _COARSEN_MAX× COARSER where the surface
# is planar at the COARSENED FOOTPRINT scale (normal spread within _FLAT_ANGLE,
# from an area-weighted normal structure tensor per footprint-sized voxel — it
# sums NNᵀ, so it's sign-invariant, unreliable TRELLIS winding can't read a flat
# wall as folded, and it's tessellation-independent; flats AND gently curved
# panels qualify while tight curvature stays at base) AND the base-colour texels
# are near-uniform (channel range < _FLAT_COLOR_VAR, so a textured rug/painting
# keeps its density). Stage 6 retrains colour, so this is ~lossless.
_COARSEN_MAX = 2.0
_FLAT_ANGLE = np.deg2rad(10.0)
_FLAT_COLOR_VAR = 0.06
_FLAT_MIN_PIECES = 8  # skip coarsening on meshes with fewer flat pieces than this
# The EMPTY mask is served to samplers as a MEMORY-MAPPED uncompressed sidecar
# (`freespace.npz.empty.npy`, written beside the npz on first use): pool
# workers then share ONE physical copy through the page cache instead of each
# decompressing a private one — the difference between parallel and OOM on
# outdoor grids (a swamp/city cell reaches billions of cells ≈ GBs of mask).
_EMPTY_SIDECAR_SUFFIX = ".empty.npy"
# When the sidecar can't be written/mapped (read-only runs dir), workers fall
# back to private in-memory masks; cap the pool so resident copies stay under
# this budget instead of OOMing the box on huge grids.
_WORKER_GRID_BUDGET = 3_000_000_000  # bytes, across all workers

# Opacity is stored pre-sigmoid; clamp alpha off the 0/1 asymptotes.
_ALPHA_CLAMP = (1e-3, 1.0 - 1e-3)


# Material alpha modes whose sampled base-color alpha is meaningful (glass, cutout).
# Everything else is forced opaque so a stray texture alpha channel can't punch holes.
_TRANSPARENT_ALPHA_MODES = ("BLEND", "MASK")


@dataclass(frozen=True)
class SampleParams:
    """THE sampling knob (plus a format flag). `detail` multiplies surfel
    density around the calibrated default look (1 = today's resolution, 2 =
    twice the surfels per m², 0.5 = half); everything else — spacing, disk
    radius, feature refinement, culling, 3dgs thickness — is derived (module
    docstring). `representation` picks the output encoding only: `2dgs`
    (native flat surfels, the default and what Stage 6 trains) or `3dgs`
    (adds a thin third scale for 3DGS-only viewers / SOG-SPZ compression)."""

    detail: float = DEFAULT_DETAIL
    representation: str = "2dgs"

    @property
    def spacing(self) -> float:
        """Flat-interior sample spacing (m) this detail level resolves to."""
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
    """Per-vertex RGBA in [0,1] for `geom`: sampled from a readable base-color
    texture at the vertex UVs when present, else the material's baseColorFactor,
    else neutral grey."""
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
    """Shortest-arc quaternions (w,x,y,z) rotating +Z (the surfel's flat axis) onto
    each unit `normal`. Antipodal case (n ≈ −Z) rotates 180° about X."""
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
    """Points sampled every ~`step` metres along the mesh's FEATURE edges: open
    boundary edges (used by exactly one face) + creases where adjacent faces meet
    at more than `_CREASE_ANGLE`. These mark where the surface actually has
    detail — corners, frames, the sides and ends of thin members — and drive the
    feature-adaptive sampling bands. None when the mesh has no feature edges
    (smooth closed blobs sample uniformly, exactly as before)."""
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
    carrying per-vertex UVs + the ORIGINAL material so `surfel_colors` samples
    the same texture through the same path on a band as on the whole object."""
    verts = tris.reshape(-1, 3)
    faces = np.arange(len(verts), dtype=np.int64).reshape(-1, 3)
    visual = None
    if uv3 is not None and material is not None:
        visual = trimesh.visual.TextureVisuals(uv=uv3.reshape(-1, 2), material=material)
    return trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=False)


def _piece_color_uniform(
    tris: np.ndarray, uv3: np.ndarray | None, material  # noqa: ANN001
) -> np.ndarray:
    """Per-piece bool: do the base-colour texels stay within `_FLAT_COLOR_VAR`
    across the piece? Sampled at each piece's corners, edge midpoints and centroid
    through the same `uv_to_color` path as `surfel_colors`. Pieces with no readable
    texture read as uniform (nothing to resolve)."""
    n = len(tris)
    image = getattr(material, "baseColorTexture", None)
    if image is None or uv3 is None:
        return np.ones(n, dtype=bool)
    bary = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1],
         [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5],
         [1 / 3, 1 / 3, 1 / 3]],
        dtype=np.float64,
    )
    uvs = np.einsum("pb,nbc->npc", bary, uv3)  # (n, P, 2) — affine over each triangle
    try:
        cols = np.asarray(
            trimesh.visual.color.uv_to_color(uvs.reshape(-1, 2), image), dtype=np.float32
        )[:, :3] / 255.0
    except Exception:
        return np.ones(n, dtype=bool)
    cols = cols.reshape(n, len(bary), 3)
    rng = (cols.max(axis=1) - cols.min(axis=1)).max(axis=1)  # widest channel spread
    return rng <= _FLAT_COLOR_VAR


def _coarsen_factors(
    tris: np.ndarray, uv3: np.ndarray | None, material, spacing: float  # noqa: ANN001
) -> np.ndarray:
    """Per-piece coarsening factor (1 = keep base spacing, up to `_COARSEN_MAX`,
    octave-quantized) for the flat, far-from-edge pieces — how far BELOW base
    density a piece can sample without losing signal. Planarity is read at the
    COARSENED FOOTPRINT scale from an area-weighted normal structure tensor per
    footprint-sized voxel: its top eigenvalue → 1 on a plane, and summing NNᵀ
    makes it sign-invariant (unreliable winding can't read a flat wall as folded)
    and tessellation-independent. A piece coarsens where that spread is within
    `_FLAT_ANGLE` (flats and gently curved panels qualify; tight curvature stays
    at base), gated to 1 wherever the base colour is not near-uniform
    (`_piece_color_uniform`). All-ones when there are too few pieces."""
    n = len(tris)
    if n <= _FLAT_MIN_PIECES:
        return np.ones(n, dtype=np.float64)
    cent = tris.mean(axis=1)
    cx = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    area = 0.5 * np.linalg.norm(cx, axis=1)
    nrm = cx / (2.0 * area[:, None] + 1e-12)
    key = np.floor(cent / (_COARSEN_MAX * spacing)).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    v = int(inv.max()) + 1
    tens = np.zeros((v, 3, 3))
    np.add.at(tens, inv, (nrm[:, :, None] * nrm[:, None, :]) * area[:, None, None])
    wsum = np.zeros(v)
    np.add.at(wsum, inv, area)
    tens /= wsum[:, None, None] + 1e-12
    lam1 = np.linalg.eigvalsh(tens)[:, -1]  # top eigenvalue ∈ (~1/3, 1]; 1 = planar
    theta = np.arcsin(np.sqrt(np.clip(1.0 - lam1, 0.0, 1.0)))[inv]
    factor = np.clip(_FLAT_ANGLE / np.maximum(theta, 1e-6), 1.0, _COARSEN_MAX)
    factor[~_piece_color_uniform(tris, uv3, material)] = 1.0
    level = np.floor(np.log2(factor) + 1e-9)
    return np.minimum(2.0 ** level, _COARSEN_MAX)


def _spacing_bands(
    geom: trimesh.Trimesh, spacing: float, boost: float = _FEATURE_BOOST
) -> list[tuple[trimesh.Trimesh, float]]:
    """Split one object into `(mesh, spacing)` sampling bands by distance to its
    feature edges — the feature-adaptive density core. Each piece's target
    spacing is its centroid's distance to the nearest feature edge, clamped to
    [spacing/boost, spacing] and quantized to octaves: a piece at distance d
    needs disks no larger than ~d to keep them off the crease, and pieces ON
    thin members (everywhere within a half-width of an edge) land in the
    finest band — which is precisely what makes rails/slats/frames sample at
    feature scale instead of wall scale. Flat interiors keep the base spacing
    unless they are also COARSENABLE (below); refinement is ADDITIVE off that
    base, so the detail knob still anchors the look while the total grows with
    the feature-area fraction and shrinks with the flat, uniform fraction.

    Labeling is TWO-LEVEL for speed, with labels IDENTICAL to a full split:
      1. split to `_CLASSIFY_COARSE`× spacing and keep any piece whose whole
         extent provably sits at base-band distance (centroid distance minus
         circumradius ≥ spacing — conservative, so no descendant could have
         classified finer);
      2. only the remaining near-edge pieces split down to `spacing` and
         classify by centroid, exactly as the one-level version did.
    Midpoint splitting is per-piece deterministic, so the refined pieces are
    the very pieces the full split would have produced; skipped interiors stay
    coarse, which sampling cannot see (area-weighted candidates, affine UVs,
    identical plane normals — tessellation granularity doesn't affect the
    sampled distribution). Big flat scenes skip almost everything; foliage-like
    meshes (edges everywhere) refine everything and merely match the old cost.

    The MIRROR of the refinement also runs: flat pieces far from every feature
    edge are COARSENED up to `_COARSEN_MAX`× where they are nearly planar AND
    uniformly coloured (`_coarsen_factors`), so walls/floors/ceilings and smooth
    low-curvature panels sample below base density. A mesh with no feature edges
    (smooth blobs, terrain) skips the edge test and coarsens throughout; a
    degenerate mesh falls back to a single base-spacing band."""
    edge_pts = _feature_edge_points(geom, step=spacing * 0.5)

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

    # Pass 1 — coarse split, then the conservative base-band test (skipped when
    # the mesh has no feature edges: every piece is then a coarsening candidate).
    tris_c, uv3_c = _subdivide_edges(tris, spacing * _CLASSIFY_COARSE, uv3)
    cent = tris_c.mean(axis=1)
    if edge_pts is not None:
        tree = cKDTree(edge_pts)
        d_c = tree.query(cent)[0]
        circum = np.linalg.norm(tris_c - cent[:, None, :], axis=2).max(axis=1)
        flat = (d_c - circum) >= spacing
    else:
        tree = None
        flat = np.ones(len(tris_c), dtype=bool)

    # The flat, far-from-edge pieces are the COARSENING candidates: split them by
    # planarity + colour uniformity into base (factor 1) and coarser octave bands.
    flat_tris = tris_c[flat]
    flat_uv = uv3_c[flat] if uv3_c is not None else None
    cf = _coarsen_factors(flat_tris, flat_uv, material, spacing)
    base_tris: list[np.ndarray] = []
    base_uv: list[np.ndarray] | None = [] if uv3_c is not None else None
    coarser: list[tuple[float, np.ndarray, np.ndarray | None]] = []
    for c in np.unique(cf):
        m = cf == c
        if c <= 1.0:
            base_tris.append(flat_tris[m])
            if base_uv is not None and flat_uv is not None:
                base_uv.append(flat_uv[m])
        else:
            coarser.append(
                (float(c), flat_tris[m], flat_uv[m] if flat_uv is not None else None)
            )
    finer: list[tuple[float, np.ndarray, np.ndarray | None]] = []

    # Pass 2 — full-resolution classification for the near-edge remainder.
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
    if base_tris:
        bt = np.concatenate(base_tris, axis=0)
        if len(bt):
            bu = np.concatenate(base_uv, axis=0) if base_uv is not None else None
            bands.append((_soup_mesh(bt, bu, material), spacing))
    for cfac, band_tris, band_uv in coarser:
        bands.append((_soup_mesh(band_tris, band_uv, material), spacing * cfac))
    for lf, band_tris, band_uv in finer:
        bands.append((_soup_mesh(band_tris, band_uv, material), spacing / lf))
    return bands or [(geom, spacing)]


def _surfel_radii(points: np.ndarray, spacing: float) -> np.ndarray:
    """Per-surfel disk radius: `_RADIUS_FRAC` × the LOCAL sample spacing
    (nearest-neighbour distance), so disks tile without gaps. Outliers clamped
    to the band median."""
    n = len(points)
    if n < 2:
        return np.full(n, spacing * _RADIUS_FRAC, dtype=np.float32)
    nn = cKDTree(points).query(points, k=2)[0][:, 1]
    good = nn[nn > 0]
    med = float(np.median(good)) if good.size else spacing
    nn = np.where(nn > 0, nn, med)
    nn = np.clip(nn, med * 0.5, med * 2.5)
    return (nn * _RADIUS_FRAC).astype(np.float32)


def surfel_colors(
    geom: trimesh.Trimesh, points: np.ndarray, face_idx: np.ndarray
) -> np.ndarray:
    """Per-surfel RGBA in [0,1]. When the mesh has a readable base-color texture,
    samples it at each surfel's barycentric-interpolated UV — the EXACT texel at
    that spot on the surface, full resolution. (A footprint-averaged variant
    using mip pyramids was tried and reverted: on TRELLIS UV atlases the filter
    crossed island boundaries into the black gutters, painting dark blotches on
    small detailed objects; the point sample is the long-validated behavior and
    Stage 6 retrains colors against the reference renders regardless.)
    Otherwise falls back to the per-face mean of the per-vertex colours."""
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


def _sample_band(
    mesh: trimesh.Trimesh,
    spacing: float,
    opaque: bool,
    fs: FreeSpace,
) -> tuple[dict[str, np.ndarray] | None, int]:
    """Blue-noise sample one band mesh at one spacing → `(per-surfel arrays for
    the VISIBLE surfels | None, sampled count)`.

    Order matters for cost, not results: radii come from ALL sampled points
    (buried neighbours count toward nearest-neighbour spacing, as before), the
    orient/cull probe runs next, and colors — the texture lookups — are
    computed for the survivors only. The probe reads the float32-rounded
    positions/normals, exactly the values the encoder will write, so the keep
    mask is bit-identical to the old cull-after-everything flow."""
    if len(mesh.faces) == 0 or mesh.area <= 0:
        return None, 0
    budget = int(mesh.area / (spacing * spacing) * 2.0) + 8
    try:
        points, face_idx = trimesh.sample.sample_surface_even(mesh, budget, radius=spacing)
    except Exception:
        points, face_idx = trimesh.sample.sample_surface(mesh, budget)
    points = np.asarray(points, dtype=np.float64)
    face_idx = np.asarray(face_idx)
    sampled = len(points)
    if sampled == 0:
        return None, 0

    normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float64)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    normals = normals / lens

    radius = _surfel_radii(points, spacing)
    pos32 = points.astype(np.float32)
    nrm32 = normals.astype(np.float32)
    keep, oriented = _orient_and_cull(pos32, nrm32, fs)
    if not keep.any():
        return None, sampled

    colors = surfel_colors(mesh, points[keep], face_idx[keep])  # (S,4) in [0,1]
    if opaque:
        colors[:, 3] = 1.0  # opaque material → ignore any texture alpha channel
    return {
        "position": pos32[keep],
        "normal": oriented[keep],
        "color": colors.astype(np.float32),
        "radius": radius[keep],
    }, sampled


def _sample_object(
    geom: trimesh.Trimesh, spacing: float, fs: FreeSpace
) -> tuple[dict[str, np.ndarray] | None, int]:
    """Sample one placed mesh into surfels across its feature-adaptive spacing
    bands (`_spacing_bands`): flat regions at the knob spacing, crease/boundary
    neighbourhoods up to `_FEATURE_BOOST`× finer. Returns `(concatenated
    per-surfel arrays for the VISIBLE surfels | None, total sampled count)`.
    Opacity is forced to 1 unless the material is genuinely BLEND/MASK (honour
    `alphaMode`, so a stray opaque-texture alpha channel can't punch holes)."""
    if len(geom.faces) == 0 or geom.area <= 0:
        return None, 0
    opaque = _alpha_mode(geom) not in _TRANSPARENT_ALPHA_MODES
    parts: list[dict[str, np.ndarray]] = []
    sampled = 0
    for band_mesh, band_spacing in _spacing_bands(geom, spacing):
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


def _orient_and_cull(
    positions: np.ndarray, normals: np.ndarray, fs: FreeSpace
) -> tuple[np.ndarray, np.ndarray]:
    """Use the Stage-2 EMPTY mask to (1) flip each normal toward the open side,
    and (2) mark surfels to keep. A side is "open" if any probe offset along
    ±normal lands in empty (viewable) air. Returns (keep_mask, oriented_normals).
    Runs per band, BEFORE color lookups, so buried surfels never pay for
    texture sampling.

    Deliberately EMPTY, not the clearance-filtered FREE: a surfel beside a thin
    gap is visible through it even though no camera fits there — filtering to
    FREE would wrongly cull every surface bordering a sub-clearance passage.

    Buried surfels (neither side open — solid interiors, seams between abutting
    objects) are dropped: they would never be seen, so they are wasted budget
    and floater seeds. Free-standing thin surfaces (both sides open) keep their
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


def _encode_ply(
    positions: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    radii: np.ndarray,
    out_path: Path,
    representation: str = "2dgs",
) -> None:
    """Write surfels as a binary Gaussian `.ply` (SH degree 0). Default `2dgs` emits a
    true flat disk: **two tangent scales** (scale_0/scale_1, no thickness) — the
    orientation quaternion's first two columns are the tangent vectors, its third the
    normal. `3dgs` appends a thin third scale (`scale_2 = radius × _FLATNESS`) for
    3DGS-only viewers / SOG-SPZ compression. Shared fields: xyz, normal, f_dc_0..2,
    opacity, scales, rot_0..3."""
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
# Stage 3's probe reads ONLY the EMPTY mask (`empty_at`), so workers load a slim
# grid: the mask + lattice geometry, skipping the clearance field and sparse
# cover sets that dominate a large grid's decode time and memory. One copy per
# process, cached across that worker's objects.

# (path, grid) as ONE reference: reads/writes are atomic under the GIL, so
# concurrent stage-3 jobs on different cells in one server process can thrash
# the slot but never observe a mismatched pair.
_TASK_FS: tuple[str, FreeSpace] | None = None


def _mapped_empty_mask(npz_path: Path, z, dims: np.ndarray):  # noqa: ANN001, ANN202
    """The EMPTY mask as a read-only memory map. Writes the uncompressed
    sidecar on first use (or when the npz is newer — a stage-2 re-run), with a
    per-process temp + atomic replace so concurrent jobs can't tear it. Any
    failure (read-only volume, corrupt sidecar) falls back to a private
    in-memory copy; None signals the caller to decompress from the npz."""
    sidecar = npz_path.with_name(npz_path.name + _EMPTY_SIDECAR_SUFFIX)
    try:
        if (
            not sidecar.is_file()
            or sidecar.stat().st_mtime < npz_path.stat().st_mtime
        ):
            tmp = sidecar.with_name(f"{sidecar.name}.{os.getpid()}.tmp.npy")
            np.save(tmp, np.asarray(z["empty"], dtype=bool))
            tmp.replace(sidecar)
        mask = np.load(sidecar, mmap_mode="r")
        if mask.dtype == np.bool_ and tuple(mask.shape) == tuple(int(v) for v in dims):
            return mask
    except Exception:
        pass
    return None


def _load_empty_probe(path: Path) -> FreeSpace:
    """The slice of the Stage-2 grid stage 3 actually consumes: the EMPTY mask
    plus origin/pitch/dims. Validates the layout exactly like
    `stage2.load_free_space` (clear re-run error on stale grids); the untouched
    heavy members (clearance, cover sets) are left empty. The mask itself is
    memory-mapped whenever possible, so every process probing the same grid
    shares one physical copy through the OS page cache."""
    nothing = np.zeros(0, dtype=np.int64)
    with np.load(path) as z:
        files = set(z.files)
        if "empty" not in files or "clearance_m" not in files or "dims" not in files:
            raise ValueError(f"{path} is a pre-clearance free-space grid — re-run Stage 2")
        dims = z["dims"].astype(np.int64)
        empty = _mapped_empty_mask(path, z, dims)
        if empty is None:
            empty = z["empty"].astype(bool)
        return FreeSpace(
            origin=z["origin"].astype(np.float64),
            pitch=float(z["pitch"]),
            dims=dims,
            occ_lin=nothing,
            occ_lin_opaque=nothing,
            clearance=np.zeros((0, 0, 0), dtype=np.float32),
            empty=empty,
            clearance_m=float(z["clearance_m"]),
        )


def _task_free_space(path: str) -> FreeSpace:
    """Process-wide cached probe grid (spawn workers load it once, then reuse
    it for every object they sample)."""
    global _TASK_FS
    cached = _TASK_FS
    if cached is None or cached[0] != path:
        cached = (path, _load_empty_probe(Path(path)))
        _TASK_FS = cached
    return cached[1]


def _sample_object_task(
    node_id: str, glb_path: str, spacing: float, freespace_path: str
) -> tuple[str, dict[str, np.ndarray] | None, float, int, str | None]:
    """Sample ONE placed GLB → `(node_id, visible-surfel arrays | None, surface
    area, sampled count, error | None)`. The process-pool work unit: results
    are per-object and merged by the caller in id order, so the parallel run
    has the same structure as the serial one. A failing mesh reports its error
    instead of raising (the serial skip-and-warn behavior)."""
    try:
        fs = _task_free_space(freespace_path)
        parts: list[dict[str, np.ndarray]] = []
        area = 0.0
        sampled = 0
        for geom in load_geoms(Path(glb_path)):
            part, n = _sample_object(geom, spacing, fs)
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
    """A SPAWN-context process pool, or None when the environment forbids
    subprocesses (degrade to serial, don't die). Spawn, not fork: stage jobs
    run off worker threads (asyncio.to_thread), where forking risks deadlock;
    spawn costs one interpreter+import per worker, amortized over the pass."""
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
    ids: list[str], raw_dir: Path, spacing: float, freespace_path: str, pool
):  # noqa: ANN201, ANN001 - yields _sample_object_task results
    """Yield per-object sampling results, serially (`pool=None`) or from the
    given process pool. Object-level parallelism is the right grain today — a
    scene dominated by ONE huge mesh (terrain, a racetrack ribbon) serializes
    on it and simply matches the old cost; splitting big objects into chunks
    needs the grid-thinning sampler first (chunk-local Poisson elimination
    would double density along the seams)."""
    if pool is None:
        for node_id in ids:
            yield _sample_object_task(
                node_id, str(raw_dir / f"{node_id}.glb"), spacing, freespace_path
            )
        return
    from concurrent.futures import as_completed

    with pool:
        futures = [
            pool.submit(
                _sample_object_task,
                node_id,
                str(raw_dir / f"{node_id}.glb"),
                spacing,
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
    `out_path`, consuming the Stage-2 free-space grid at `freespace_path` to
    orient normals + cull hidden faces (per object, before coloring). ONE pass
    over the meshes: the `detail` knob resolves to a spacing directly (no area
    pre-pass), and total surface area is accumulated during sampling for the
    summary (it feeds the client's count/size estimate, not the math).

    `workers` parallelizes the per-object pass (0 = auto: min(cores, 8), then
    capped so every worker's copy of the EMPTY mask fits the memory budget;
    1 = serial). Results merge in id order, so the output's structure doesn't
    depend on the worker count (sampling itself is unseeded, as always)."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"placed-mesh dir not found: {raw_dir}")
    if not Path(freespace_path).is_file():
        raise FileNotFoundError(f"free-space grid not found: {freespace_path} (run Stage 2)")
    ids = placed_object_ids(raw_dir)
    if not ids:
        raise FileNotFoundError(f"no placed meshes in {raw_dir}")

    # Validate the grid layout up front (one clear error instead of a warning
    # per object) and prime the parent-side cache for the serial path.
    fs_path = str(freespace_path)
    probe = _task_free_space(fs_path)
    spacing = params.spacing

    if workers <= 0:
        workers = min(os.cpu_count() or 1, 8)
    if not isinstance(probe.empty, np.memmap):
        # The mask sidecar couldn't be mapped, so every worker holds a PRIVATE
        # decompressed copy — cap the pool so huge outdoor grids (billions of
        # cells) shrink parallelism instead of OOMing the box.
        mem_cap = max(1, int(_WORKER_GRID_BUDGET // max(probe.empty.nbytes, 1)))
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
        ids, raw_dir, spacing, fs_path, pool
    ):
        done += 1
        results[node_id] = (part, area, n_sampled, err)
        if progress is not None:
            progress(done, total, node_id)
    sample_s = time.perf_counter() - t0

    # Merge in id order — deterministic structure whatever the completion order.
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
    culled = int(sampled - positions.shape[0])

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
        "culled_hidden": culled,
        "objects_sampled": objects_sampled,
        "objects_total": total,
        "spacing": spacing,
        "total_area": round(total_area, 2),
        "workers": workers,
        "timings": {"sample_s": round(sample_s, 2), "encode_s": round(encode_s, 2)},
        "params": params.as_summary(),
        "scene_aabb": {"min": aabb_min, "max": aabb_max},
        "warnings": warnings,
        "bytes": out_path.stat().st_size,
    }
