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

Sampling (overview §): per-object blue-noise (Poisson-disk) even spacing, adaptive
per object by feature size. Each surfel: position (world; placement baked into the
vertices), rotation (quaternion aligning +Z to the oriented normal), scale (r, r,
~0), color (per-texel albedo), opacity (base-color alpha honouring `alphaMode`).
Stored SH is degree 0 (`f_dc`) — unlit / view-independent.

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

from splat.stage2 import FreeSpace, load_free_space

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
    scales spacing per object; `cull_hidden` drops surfels with no reachable free
    space on either side (needs the Stage-2 grid)."""

    target_splats: int | None = None      # explicit count override; else density × area
    splat_density: float | None = DEFAULT_SPLAT_DENSITY  # surfels per m² (area-scaled count)
    base_spacing: float | None = None
    radius_frac: float = DEFAULT_RADIUS_FRAC
    flatness: float = DEFAULT_FLATNESS   # 3dgs-mode only: thin third scale = radius*flatness
    adaptive: bool = True
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


def surfel_colors(
    geom: trimesh.Trimesh, points: np.ndarray, face_idx: np.ndarray
) -> np.ndarray:
    """Per-surfel RGBA in [0,1]. When the mesh has a readable base-color texture,
    samples it at each surfel's barycentric-interpolated UV (per-texel albedo);
    otherwise falls back to the per-face mean of the per-vertex colours."""
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


def _sample_object(
    geom: trimesh.Trimesh, base_spacing: float, params: SampleParams
) -> dict[str, np.ndarray] | None:
    """Blue-noise sample one placed mesh into surfels. Returns per-surfel arrays
    (position, normal, color rgba, radius) or None if the mesh is empty. Opacity is
    forced to 1 unless the material is genuinely BLEND/MASK (honour `alphaMode`, so a
    stray opaque-texture alpha channel can't punch holes)."""
    if len(geom.faces) == 0 or geom.area <= 0:
        return None
    spacing = _object_spacing(geom, base_spacing, params.adaptive)
    budget = int(geom.area / (spacing * spacing) * 2.0) + 8
    try:
        points, face_idx = trimesh.sample.sample_surface_even(geom, budget, radius=spacing)
    except Exception:
        points, face_idx = trimesh.sample.sample_surface(geom, budget)
    points = np.asarray(points, dtype=np.float64)
    face_idx = np.asarray(face_idx)
    if len(points) == 0:
        return None

    normals = np.asarray(geom.face_normals[face_idx], dtype=np.float64)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    normals = normals / lens

    colors = surfel_colors(geom, points, face_idx)  # (S,4) in [0,1]
    if _alpha_mode(geom) not in _TRANSPARENT_ALPHA_MODES:
        colors[:, 3] = 1.0  # opaque material → ignore any texture alpha channel
    radius = _surfel_radii(points, spacing, params.radius_frac)
    return {
        "position": points.astype(np.float32),
        "normal": normals.astype(np.float32),
        "color": colors.astype(np.float32),
        "radius": radius,
    }


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
