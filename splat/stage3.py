"""Stage 3 — Surfel sampler (mesh → Gaussian splat).

Turns a cell's placed, *vanilla* meshes into a **pre-fine-tuning Gaussian splat**:
a cloud of flat, surface-aligned 2D Gaussians (surfels), written as a **2DGS `.ply`**
(two tangent scales, no thickness) that 2DGS-capable web viewers (mkkellogg, PlayCanvas
v2.16+) read and the Stage-6 gsplat **2DGS** fine-tune inits from. A `3dgs` mode appends
a thin third scale for 3DGS-only viewers / SOG-SPZ compression.

Consumes the **Stage-2 free-space grid** (per Option A the free-space foundation runs
first). Two things use it:
  * **Normal orientation (fixes unreliable TRELLIS winding):** each surfel's normal is
    flipped to point toward NAVIGABLE (reachable) free space, rather than trusting the
    mesh's face winding — so disks face the viewer regardless of authoring.
  * **Hidden-face culling:** a surfel with no reachable free space on EITHER side is
    buried (a solid's interior, or the seam between two abutting objects) and is
    dropped — it would never be seen, so it's wasted budget and a floater seed.

Sampling (overview §): per-object blue-noise (Poisson-disk) even spacing in
FEATURE-ADAPTIVE BANDS. Detail is measured ONCE, from the mesh itself (its
feature edges: creases sharper than `_CREASE_ANGLE` + open boundaries — full-
fidelity signals, not statistics of an already-sampled proxy): flat interiors
sample at the density-knob spacing exactly as before, while crease/boundary
neighbourhoods — thin members, frames, slats, corners, which by construction lie
within a base-spacing of a feature edge — sample up to `feature_boost`× finer
(boost²× the density). Purely additive: flats are unchanged and detail rides on
top, so the density knob keeps its meaning. Downstream, the cloud's local
density IS the detail field: Stage 4 reads it (local sample spacing) instead of
re-detecting detail from the cloud, and Stage 6's densification starts from
capacity that already exists where the close-up cameras will look.

Each surfel: position (world; placement baked into the vertices), rotation
(quaternion aligning +Z to the oriented normal), scale (r, r, ~0), color (albedo
FOOTPRINT-AVERAGED over the disk — the texture's mip level matching the disk's
texel diameter, so one disk summarizes the area it covers instead of a single
pinprick texel), opacity (base-color alpha honouring `alphaMode`, averaged the
same way). Stored SH is degree 0 (`f_dc`) — unlit / view-independent.

Pure library: `sample_cell` takes explicit paths; the server resolves a cell to them
(de-optimizing a library build to vanilla first) and passes the Stage-2
`freespace.npz`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from splat.stage2 import FreeSpace, _subdivide_edges, _valid_tri_mask, load_free_space

# Poisson-disk elimination routinely returns fewer than the oversample budget
# (that's the point — it thins clumps), which trimesh logs as "only got N/M
# samples!" per object. Silence it so a whole-cell run isn't a wall of warnings.
logging.getLogger("trimesh").setLevel(logging.ERROR)

# The Gaussian cloud filename written under a cell's `splat/` dir (a 3DGS `.ply`).
CLOUD_NAME = "cloud.ply"

# SH degree-0 basis constant: colour c in [0,1] maps to f_dc = (c - 0.5) / C0
# (the 3DGS convention every viewer inverts as colour = 0.5 + C0 * f_dc).
_SH_C0 = 0.28209479177387814

DEFAULT_TARGET_SPLATS = 150_000
# Surfel budget as a DENSITY (surfels per m² of surface) rather than a fixed count,
# so the cloud resolution stays constant as scenes grow (count scales with area).
# ~80/m² reproduces the previous 150k default on a room-scale (~1900 m²) cell.
DEFAULT_SPLAT_DENSITY = 80.0
DEFAULT_RADIUS_FRAC = 0.9
DEFAULT_FLATNESS = 0.1
# Feature-adaptive density: sampling within one base-spacing of a FEATURE EDGE
# (a crease sharper than _CREASE_ANGLE, or an open boundary) is refined in
# octave bands down to base/feature_boost. 25° cleanly separates man-made
# creases (box corners, frames, rail edges ~60-90°) from the small per-edge
# angles of dense smooth meshes (~10-15°), so smooth blobs stay single-band.
DEFAULT_FEATURE_BOOST = 4.0
_CREASE_ANGLE = np.deg2rad(25.0)
_EDGE_PT_CAP = 400_000  # stride a pathological crease soup instead of exploding

# Empirical density model tying surfel count to spacing (see overview §7).
_SPACING_EXP = 2.5
_AREAL_K = 0.006
_SPACING_CLAMP = (0.004, 0.25)

# Opacity is stored pre-sigmoid; clamp alpha off the 0/1 asymptotes.
_ALPHA_CLAMP = (1e-3, 1.0 - 1e-3)


# Material alpha modes whose sampled base-color alpha is meaningful (glass, cutout).
# Everything else is forced opaque so a stray texture alpha channel can't punch holes.
_TRANSPARENT_ALPHA_MODES = ("BLEND", "MASK")


@dataclass(frozen=True)
class SampleParams:
    """Global sampling knobs for one pass — set live from the splat client. `adaptive`
    scales spacing per object; `feature_boost` is the feature-adaptive density
    dial — crease/boundary neighbourhoods sample up to this factor FINER than the
    base spacing (1.0 disables, restoring uniform density); `cull_hidden` drops
    surfels with no reachable free space on either side (needs the Stage-2 grid)."""

    target_splats: int | None = None      # explicit count override; else density × area
    splat_density: float | None = DEFAULT_SPLAT_DENSITY  # surfels per m² (area-scaled count)
    base_spacing: float | None = None
    radius_frac: float = DEFAULT_RADIUS_FRAC
    flatness: float = DEFAULT_FLATNESS   # 3dgs-mode only: thin third scale = radius*flatness
    adaptive: bool = True
    feature_boost: float = DEFAULT_FEATURE_BOOST  # max refinement near feature edges (1 = off)
    cull_hidden: bool = True
    representation: str = "2dgs"          # "2dgs" (native surfels) | "3dgs" (compat/compression)

    def as_summary(self) -> dict[str, Any]:
        return {
            "target_splats": self.target_splats,
            "splat_density": self.splat_density,
            "base_spacing": self.base_spacing,
            "radius_frac": self.radius_frac,
            "flatness": self.flatness,
            "adaptive": self.adaptive,
            "feature_boost": self.feature_boost,
            "cull_hidden": self.cull_hidden,
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


def _iter_geoms(mesh: trimesh.Trimesh | trimesh.Scene) -> list[trimesh.Trimesh]:
    # dump() bakes each geometry's scene-graph node transform into world space; plain
    # scene.geometry returns LOCAL vertices, which collapses node-placed objects
    # (generated assets carry placement on the node) to the origin.
    if isinstance(mesh, trimesh.Scene):
        return [g for g in mesh.dump(concatenate=False) if hasattr(g, "faces")]
    return [mesh]


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


def _object_spacing(geom: trimesh.Trimesh, base_spacing: float, adaptive: bool) -> float:
    """Per-object sample spacing. With `adaptive`, `base_spacing` is scaled by the
    object's size (small/detailed finer, large flat coarser)."""
    if not adaptive:
        return base_spacing
    ext = np.asarray(geom.bounds[1], dtype=float) - np.asarray(geom.bounds[0], dtype=float)
    diag = float(np.linalg.norm(ext)) or base_spacing
    scale = float(np.clip((diag / 1.5) ** 0.5, 0.5, 1.8))
    return base_spacing * scale


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


def _spacing_bands(
    geom: trimesh.Trimesh, spacing: float, boost: float
) -> list[tuple[trimesh.Trimesh, float]]:
    """Split one object into `(mesh, spacing)` sampling bands by distance to its
    feature edges — the feature-adaptive density core. Faces are midpoint-split
    to ~`spacing` resolution (so a large wall face can straddle bands), then each
    piece's target spacing is its centroid's distance to the nearest feature
    edge, clamped to [spacing/boost, spacing] and quantized to octaves: a piece
    at distance d needs disks no larger than ~d to keep them off the crease, and
    pieces ON thin members (everywhere within a half-width of an edge) land in
    the finest band — which is precisely what makes rails/slats/frames sample at
    feature scale instead of wall scale. Flat interiors (d ≥ spacing) keep the
    base spacing bit-for-bit; detail is ADDITIVE, so the density knob's meaning
    is unchanged on flats and the total grows only with feature-area fraction.

    Falls back to a single base-spacing band when boost ≤ 1, the mesh has no
    feature edges (smooth blobs), or its triangles are degenerate."""
    if boost <= 1.0:
        return [(geom, spacing)]
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
    tris, uv3 = _subdivide_edges(tris, spacing, uv3)

    d = cKDTree(edge_pts).query(tris.mean(axis=1))[0]
    factor = spacing / np.clip(d, spacing / boost, spacing)
    level = np.ceil(np.log2(np.maximum(factor, 1.0)) - 1e-9)
    f = np.minimum(2.0 ** level, boost)

    bands: list[tuple[trimesh.Trimesh, float]] = []
    for lf in np.unique(f):
        m = f == lf
        band_uv = uv3[m] if uv3 is not None else None
        bands.append((_soup_mesh(tris[m], band_uv, material), spacing / float(lf)))
    return bands


def _surfel_radii(points: np.ndarray, spacing: float, radius_frac: float) -> np.ndarray:
    """Per-surfel disk radius from the LOCAL sample spacing (nearest-neighbour
    distance), so disks tile without gaps. Outliers clamped to the object median."""
    n = len(points)
    if n < 2:
        return np.full(n, spacing * radius_frac, dtype=np.float32)
    nn = cKDTree(points).query(points, k=2)[0][:, 1]
    good = nn[nn > 0]
    med = float(np.median(good)) if good.size else spacing
    nn = np.where(nn > 0, nn, med)
    nn = np.clip(nn, med * 0.5, med * 2.5)
    return (nn * radius_frac).astype(np.float32)


def _build_mips(image) -> list[np.ndarray]:  # noqa: ANN001 - PIL.Image
    """RGBA float32 [0,1] mip pyramid: level 0 is the full-resolution texture,
    each next level a 2× box (area-average) downsample, ending at 1×1. One
    level-k texel stores the mean of a ~2^k-texel-wide block of the original,
    so sampling level log2(d) approximates a d-texel-wide average in a single
    lookup."""
    from PIL import Image

    img = image.convert("RGBA")
    levels = [np.asarray(img, dtype=np.float32) / 255.0]
    w, h = img.size
    while max(w, h) > 1:
        w, h = max(1, w // 2), max(1, h // 2)
        img = img.resize((w, h), Image.BOX)
        levels.append(np.asarray(img, dtype=np.float32) / 255.0)
    return levels


def _mip_sample(
    mips: list[np.ndarray], uvs: np.ndarray, levels: np.ndarray
) -> np.ndarray:
    """RGBA per (uv, mip level), replicating `trimesh.visual.color.uv_to_color`'s
    pixel convention at every level (nearest texel, v=0 at the image bottom,
    wrap): a level-0 lookup matches the old point-sample path texel-for-texel,
    and coarser levels return footprint averages in the same frame."""
    uvs = np.nan_to_num(np.asarray(uvs, dtype=np.float64), nan=0.0)
    out = np.empty((len(uvs), 4), dtype=np.float32)
    for k in np.unique(levels):
        tex = mips[int(k)]
        h, w = tex.shape[:2]
        sel = levels == k
        x = np.round((uvs[sel, 0] * (w - 1)) % w).astype(np.int64) % w
        y = np.round(((1.0 - uvs[sel, 1]) * (h - 1)) % h).astype(np.int64) % h
        out[sel] = tex[y, x]
    return out


def _footprint_levels(
    geom: trimesh.Trimesh,
    face_idx: np.ndarray,
    radii: np.ndarray,
    tex_wh: tuple[int, int],
    n_levels: int,
) -> np.ndarray:
    """Mip level per surfel. A disk of radius r covers ≈ 2·r·ρ texels across,
    where ρ (texels/metre) is its triangle's linear texel density —
    sqrt(UV area × texture pixel count / world area). Level = round(log2 of the
    texel diameter), clamped to the pyramid. Degenerate UV or world triangles
    give ρ = 0 → level 0 (the old point sample), a graceful fallback."""
    uf, inv = np.unique(np.asarray(face_idx), return_inverse=True)
    fv = np.asarray(geom.faces)[uf]                              # (U,3) vert ids
    tri = np.asarray(geom.vertices)[fv]                          # (U,3,3) world
    aw = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    uvt = np.asarray(geom.visual.uv, dtype=np.float64)[fv]       # (U,3,2)
    d1 = uvt[:, 1] - uvt[:, 0]
    d2 = uvt[:, 2] - uvt[:, 0]
    auv = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    auv_tex = auv * float(tex_wh[0]) * float(tex_wh[1])          # texel² / face
    rho = np.sqrt(auv_tex / np.maximum(aw, 1e-18))               # texels / metre
    diam = 2.0 * np.asarray(radii, dtype=np.float64) * rho[inv]
    lvl = np.log2(np.maximum(np.nan_to_num(diam), 1.0))
    return np.clip(np.round(lvl), 0, n_levels - 1).astype(np.int64)


def surfel_colors(
    geom: trimesh.Trimesh,
    points: np.ndarray,
    face_idx: np.ndarray,
    radii: np.ndarray | None = None,
) -> np.ndarray:
    """Per-surfel RGBA in [0,1]. With a readable base-color texture, each surfel
    gets the FOOTPRINT AVERAGE of the texture over its disk: the mip level whose
    filter width matches the disk's texel diameter is sampled at the surfel's
    barycentric-interpolated UV. One disk summarizes the area it covers instead
    of one pinprick texel — busy textures init as their local mean rather than
    per-surfel noise — and alpha averages identically (soft MASK-cutout / glass
    boundaries). With `radii=None`, or wherever the UV mapping is degenerate,
    this reduces to the old level-0 point sample (texel-identical to
    `uv_to_color`). No texture → per-face mean of the per-vertex colours."""
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
            mips = _build_mips(image)
            if radii is not None and len(mips) > 1:
                levels = _footprint_levels(
                    geom, face_idx, radii, (image.width, image.height), len(mips)
                )
            else:
                levels = np.zeros(len(points), dtype=np.int64)
            cols = _mip_sample(mips, uvs, levels)
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
    mesh: trimesh.Trimesh, spacing: float, opaque: bool, params: SampleParams
) -> dict[str, np.ndarray] | None:
    """Blue-noise sample one band mesh at one spacing → per-surfel arrays
    (position, normal, color rgba, radius), or None if nothing sampled. Radii and
    colour footprints are band-local, so a fine band's small disks average small
    texture regions and a coarse band's large disks average large ones."""
    if len(mesh.faces) == 0 or mesh.area <= 0:
        return None
    budget = int(mesh.area / (spacing * spacing) * 2.0) + 8
    try:
        points, face_idx = trimesh.sample.sample_surface_even(mesh, budget, radius=spacing)
    except Exception:
        points, face_idx = trimesh.sample.sample_surface(mesh, budget)
    points = np.asarray(points, dtype=np.float64)
    face_idx = np.asarray(face_idx)
    if len(points) == 0:
        return None

    normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float64)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    normals = normals / lens

    # Radii first: colors are footprint-averaged over each disk, so the color
    # lookup needs to know how much surface each surfel covers.
    radius = _surfel_radii(points, spacing, params.radius_frac)
    colors = surfel_colors(mesh, points, face_idx, radii=radius)  # (S,4) in [0,1]
    if opaque:
        colors[:, 3] = 1.0  # opaque material → ignore any texture alpha channel
    return {
        "position": points.astype(np.float32),
        "normal": normals.astype(np.float32),
        "color": colors.astype(np.float32),
        "radius": radius,
    }


def _sample_object(
    geom: trimesh.Trimesh, base_spacing: float, params: SampleParams
) -> dict[str, np.ndarray] | None:
    """Sample one placed mesh into surfels across its feature-adaptive spacing
    bands (`_spacing_bands`): flat regions at the base spacing, crease/boundary
    neighbourhoods up to `feature_boost`× finer. Returns concatenated per-surfel
    arrays (position, normal, color rgba, radius) or None if the mesh is empty.
    Opacity is forced to 1 unless the material is genuinely BLEND/MASK (honour
    `alphaMode`, so a stray opaque-texture alpha channel can't punch holes)."""
    if len(geom.faces) == 0 or geom.area <= 0:
        return None
    spacing = _object_spacing(geom, base_spacing, params.adaptive)
    opaque = _alpha_mode(geom) not in _TRANSPARENT_ALPHA_MODES
    parts: list[dict[str, np.ndarray]] = []
    for band_mesh, band_spacing in _spacing_bands(geom, spacing, params.feature_boost):
        band = _sample_band(band_mesh, band_spacing, opaque, params)
        if band is not None:
            parts.append(band)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


def _orient_and_cull(
    positions: np.ndarray, normals: np.ndarray, fs: FreeSpace, cull: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Use the reachable free-space mask to (1) flip each normal toward the navigable
    side, and (2) mark surfels to keep. A side is "open" if any probe offset along
    ±normal lands in reachable free space. Returns (keep_mask, oriented_normals).

    Buried surfels (neither side reachable — solid interiors, seams between abutting
    objects) are dropped when `cull`. Free-standing thin surfaces (both sides open)
    keep their original normal."""
    offsets = np.array([1.0, 1.5, 2.0], dtype=np.float64) * fs.pitch
    plus_open = np.zeros(len(positions), dtype=bool)
    minus_open = np.zeros(len(positions), dtype=bool)
    for d in offsets:
        plus_open |= fs.reachable_free(positions + normals * d)
        minus_open |= fs.reachable_free(positions - normals * d)
    oriented = normals.copy()
    flip = minus_open & ~plus_open
    oriented[flip] = -oriented[flip]
    keep = (plus_open | minus_open) if cull else np.ones(len(positions), dtype=bool)
    return keep, oriented


def _encode_ply(
    positions: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    radii: np.ndarray,
    out_path: Path,
    flatness: float,
    representation: str = "2dgs",
) -> None:
    """Write surfels as a binary Gaussian `.ply` (SH degree 0). Default `2dgs` emits a
    true flat disk: **two tangent scales** (scale_0/scale_1, no thickness) — the
    orientation quaternion's first two columns are the tangent vectors, its third the
    normal. `3dgs` appends a thin third scale (`scale_2 = radius*flatness`) for
    3DGS-only viewers / SOG-SPZ compression. Shared fields: xyz, normal, f_dc_0..2,
    opacity, scales, rot_0..3."""
    n = positions.shape[0]
    quats = _align_quaternions(normals).astype(np.float32)

    rgb = np.clip(colors[:, :3], 0.0, 1.0)
    f_dc = ((rgb - 0.5) / _SH_C0).astype(np.float32)

    alpha = np.clip(colors[:, 3], _ALPHA_CLAMP[0], _ALPHA_CLAMP[1])
    opacity = np.log(alpha / (1.0 - alpha)).astype(np.float32)

    if representation == "3dgs":
        scale = np.stack([radii, radii, radii * flatness], axis=1)
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


def _scene_area_and_diag(raw_dir: Path, ids: list[str]) -> tuple[float, float]:
    """Total surface area (m²) and bounding-box diagonal (m) over all placed
    meshes — one trimesh load each. Area turns a target surfel count into a spacing."""
    lo = np.array([np.inf, np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf, -np.inf])
    area = 0.0
    for node_id in ids:
        try:
            m = trimesh.load(raw_dir / f"{node_id}.glb", process=False)
        except Exception:
            continue
        b = np.asarray(m.bounds, dtype=float)
        lo, hi = np.minimum(lo, b[0]), np.maximum(hi, b[1])
        area += float(sum(g.area for g in _iter_geoms(m)))
        del m
    diag = float(np.linalg.norm(hi - lo)) if np.isfinite(lo).all() else 4.0
    return area, diag


def _effective_target(params: SampleParams, total_area: float) -> int | None:
    """The surfel count the packing model aims for: an explicit `target_splats` if
    given, else `splat_density × total_area` (so the count — and thus fidelity —
    scales with surface area instead of a fixed room-scale budget)."""
    if params.target_splats:
        return int(params.target_splats)
    if params.splat_density and total_area > 0:
        return int(round(params.splat_density * total_area))
    return None


def _resolve_base_spacing(params: SampleParams, total_area: float, diag: float) -> float:
    """Base surfel spacing (metres): explicit override, else the area-scaled target
    (density × area, or an explicit count) via the packing model, else ~0.3% of the
    scene diagonal. Always clamped."""
    target = _effective_target(params, total_area)
    if params.base_spacing is not None:
        s = params.base_spacing
    elif target and total_area > 0:
        s = float((_AREAL_K * total_area / target) ** (1.0 / _SPACING_EXP))
    else:
        s = diag * 0.003
    return float(np.clip(s, *_SPACING_CLAMP))


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
) -> dict[str, Any]:
    """Sample every placed mesh in `raw_dir` into a Gaussian splat written to
    `out_path` (a 3DGS `.ply`), consuming the Stage-2 free-space grid at
    `freespace_path` to orient normals + cull hidden faces. Returns a compact
    summary (resolved spacing, splat count kept/culled, total surface area)."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"placed-mesh dir not found: {raw_dir}")
    if not Path(freespace_path).is_file():
        raise FileNotFoundError(f"free-space grid not found: {freespace_path} (run Stage 2)")
    ids = placed_object_ids(raw_dir)
    if not ids:
        raise FileNotFoundError(f"no placed meshes in {raw_dir}")

    fs = load_free_space(Path(freespace_path))
    total_area, diag = _scene_area_and_diag(raw_dir, ids)
    base_spacing = _resolve_base_spacing(params, total_area, diag)

    total = len(ids)
    if progress is not None:
        progress(0, total, "")

    pos_parts: list[np.ndarray] = []
    nrm_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    rad_parts: list[np.ndarray] = []
    warnings: list[str] = []
    objects_sampled = 0

    for done, node_id in enumerate(ids, start=1):
        try:
            loaded = trimesh.load(raw_dir / f"{node_id}.glb", process=False)
            for geom in _iter_geoms(loaded):
                surf = _sample_object(geom, base_spacing, params)
                if surf is None:
                    continue
                pos_parts.append(surf["position"])
                nrm_parts.append(surf["normal"])
                col_parts.append(surf["color"])
                rad_parts.append(surf["radius"])
            objects_sampled += 1
            del loaded
        except Exception as exc:
            warnings.append(f"{node_id}: failed to sample ({type(exc).__name__}: {exc})")
        if progress is not None:
            progress(done, total, node_id)

    if not pos_parts:
        raise RuntimeError("no surfels sampled (every mesh failed or was empty)")

    positions = np.concatenate(pos_parts, axis=0)
    normals = np.concatenate(nrm_parts, axis=0)
    colors = np.concatenate(col_parts, axis=0)
    radii = np.concatenate(rad_parts, axis=0)

    # Free-space normal orientation + hidden-face culling (needs the Stage-2 grid).
    sampled = positions.shape[0]
    keep, normals = _orient_and_cull(positions, normals, fs, params.cull_hidden)
    positions, normals = positions[keep], normals[keep]
    colors, radii = colors[keep], radii[keep]
    culled = int(sampled - positions.shape[0])

    if positions.shape[0] == 0:
        raise RuntimeError("all surfels culled as hidden (check the free-space grid)")

    _encode_ply(positions, normals, colors, radii, out_path, params.flatness, params.representation)

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
        "base_spacing": base_spacing,
        "target_effective": _effective_target(params, total_area),
        "total_area": round(total_area, 2),
        "params": params.as_summary(),
        "scene_aabb": {"min": aabb_min, "max": aabb_max},
        "warnings": warnings,
        "bytes": out_path.stat().st_size,
    }
