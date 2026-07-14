"""Stage 2 — Surfel sampler (mesh → Gaussian splat).

Turns a cell's placed, *vanilla* meshes (the Stage-1 source — the generated raw
build, or the de-optimized library build) into a **pre-fine-tuning Gaussian
splat**: a cloud of flat, surface-aligned 2D Gaussians (surfels), written as a
standard 3DGS `.ply` the web viewers (Spark / mkkellogg / PlayCanvas) read.

Like Stage 1 this is a *pure library* — `sample_cell` takes explicit paths; the
server resolves a `(run, slot, model)` cell to them (and de-optimizes a library
build to vanilla first, since trimesh can't read KTX2/Meshopt).

Sampling (per the overview doc, §5 Stage 2 / §7):
  * Blue noise — per-object Poisson-disk (even) sampling, so surfels are evenly
    spaced with no clumps/gaps (via `trimesh.sample.sample_surface_even`, which
    does Poisson-disk elimination with a min-spacing radius).
  * Adaptive density — the spacing is scaled per object by its feature size, so
    small/detailed objects are sampled finer and large flat surfaces coarser.
Each surfel:
  * position — the surface sample (WORLD space; placement is baked into the
    vertices by generation's `rescale_mesh_to_bbox`, so no transform is needed).
  * rotation — a quaternion aligning the flat disk to the surface normal.
  * scale    — (r, r, ~0): a disk of radius ≈ the local sample spacing.
  * color    — albedo, per-texel from the base-color texture at each surfel's
    interpolated UV when present (library KTX2 maps are transcoded to PNG during
    de-opt, so those cells are coloured too), else `baseColorFactor`, else grey.
  * opacity  — the base-colour alpha honouring `alphaMode` (1 for opaque).
Stored SH is degree 0 only (`f_dc`) — unlit / view-independent, pre-fine-tuning.

Density and the other knobs (radius/flatness/adaptive) live in `SampleParams`, so
the client can retune and re-splat live without a server restart.
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

# Poisson-disk elimination routinely returns fewer than the oversample budget
# (that's the point — it thins clumps), which trimesh logs as "only got N/M
# samples!" per object. Silence it so a whole-cell run isn't a wall of warnings.
logging.getLogger("trimesh").setLevel(logging.ERROR)

# The Gaussian cloud filename written under a cell's `splat/` dir (a 3DGS `.ply`).
CLOUD_NAME = "cloud.ply"

# SH degree-0 basis constant: colour c in [0,1] maps to f_dc = (c - 0.5) / C0
# (the 3DGS convention every viewer inverts as colour = 0.5 + C0 * f_dc).
_SH_C0 = 0.28209479177387814

# Defaults for the tunable knobs (SampleParams) — also the client control
# defaults. target_splats ~150k ≈ 10 MB at ~68 B/surfel (the base-LOD budget).
DEFAULT_TARGET_SPLATS = 150_000
# Disk radius as a multiple of the LOCAL sample spacing (nearest-neighbour
# distance). ≳1 so each surfel's Gaussian overlaps its neighbours and the surface
# tiles WITHOUT gaps; sizing off the *achieved* spacing (not the nominal target)
# is what closes them, since blue-noise elimination spreads points further apart.
DEFAULT_RADIUS_FRAC = 0.9
# The flat (normal) axis of a surfel ÷ its in-plane radius — thin but not paper-
# thin, so surfels don't wink out edge-on and leave slivers.
DEFAULT_FLATNESS = 0.1

# Empirical density model tying surfel count to spacing:
#   count ≈ _AREAL_K * total_area / base_spacing^_SPACING_EXP,
# so to hit a target count we take
#   base_spacing = (_AREAL_K * total_area / target)^(1/_SPACING_EXP).
# The exponent is ~2.5, not 2, because small objects keep "blooming" as spacing
# refines (their tiny sample budgets fill in). Calibrated on room cells with
# adaptive spacing on — it folds in the blue-noise yield, per-object coarsening,
# and sample budget, so it's a good DEFAULT; the client shows the ACTUAL count +
# size so the target can be nudged. Off for very different scenes, which is fine.
_SPACING_EXP = 2.5
_AREAL_K = 0.006
# base_spacing (metres) is clamped here however it's derived, so a runaway target
# or degenerate scene can't produce absurd surfels.
_SPACING_CLAMP = (0.004, 0.25)

# Opacity is stored pre-sigmoid (the renderer applies sigmoid); clamp alpha off
# the 0/1 asymptotes so the logit is finite.
_ALPHA_CLAMP = (1e-3, 1.0 - 1e-3)


@dataclass(frozen=True)
class SampleParams:
    """The global sampling knobs for one pass — set live from the splat client
    (no server restart). `base_spacing` (metres), when given, overrides the
    `target_splats` density budget; `adaptive` scales spacing per object by size
    (small/detailed finer, big flat coarser)."""

    target_splats: int = DEFAULT_TARGET_SPLATS
    base_spacing: float | None = None
    radius_frac: float = DEFAULT_RADIUS_FRAC
    flatness: float = DEFAULT_FLATNESS
    adaptive: bool = True

    def as_summary(self) -> dict[str, Any]:
        return {
            "target_splats": self.target_splats,
            "base_spacing": self.base_spacing,
            "radius_frac": self.radius_frac,
            "flatness": self.flatness,
            "adaptive": self.adaptive,
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
    if isinstance(mesh, trimesh.Scene):
        return [g for g in mesh.geometry.values() if hasattr(g, "faces")]
    return [mesh]


def _vertex_colors(geom: trimesh.Trimesh) -> np.ndarray:
    """Per-vertex RGBA in [0,1] for `geom`: sampled from a readable base-color
    texture at the vertex UVs when present, else the material's baseColorFactor,
    else neutral grey. Grey (not white) is used as the fallback so geometry-only
    (textureless) meshes stay legible instead of blowing out to white."""
    n = len(geom.vertices)
    visual = getattr(geom, "visual", None)
    # 1) A readable texture (PNG/JPEG) + UVs → per-vertex texel colour. trimesh's
    #    `to_color()` samples the base-color texture at each vertex UV. KTX2
    #    textures aren't decodable, so those meshes fall through to the factor.
    try:
        if visual is not None and getattr(visual, "uv", None) is not None:
            material = getattr(visual, "material", None)
            if getattr(material, "baseColorTexture", None) is not None:
                cols = np.asarray(visual.to_color().vertex_colors, dtype=np.float32)
                if cols.shape == (n, 4):
                    return cols / 255.0
    except Exception:
        pass
    # 2) A meaningful baseColorFactor (anything but the default opaque white).
    material = getattr(visual, "material", None)
    factor = getattr(material, "baseColorFactor", None)
    if factor is not None:
        arr = np.asarray(factor, dtype=np.float32).reshape(-1)
        if arr.size >= 3 and arr.max() > 1.0:
            arr = arr / 255.0
        rgb = arr[:3]
        alpha = float(arr[3]) if arr.size >= 4 else 1.0
        if not np.allclose(rgb, 1.0):  # skip the texture-driven default white
            out = np.empty((n, 4), dtype=np.float32)
            out[:, :3] = rgb
            out[:, 3] = alpha
            return out
    # 3) Fallback: opaque neutral grey.
    out = np.empty((n, 4), dtype=np.float32)
    out[:, :3] = _MID_GREY
    out[:, 3] = 1.0
    return out


def _align_quaternions(normals: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternions (w,x,y,z) rotating +Z (the surfel's flat axis)
    onto each unit `normal`. Vectorised; the antipodal case (n ≈ −Z) rotates
    180° about X."""
    n = normals.shape[0]
    dot = normals[:, 2]  # +Z · n
    # cross(+Z, n) = (−n_y, n_x, 0)
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
    object's size so small/detailed objects sample finer and large flat ones
    coarser (0.5×–1.8×, size = bbox diagonal vs 1 m); otherwise it's uniform."""
    if not adaptive:
        return base_spacing
    ext = np.asarray(geom.bounds[1], dtype=float) - np.asarray(geom.bounds[0], dtype=float)
    diag = float(np.linalg.norm(ext)) or base_spacing
    scale = float(np.clip((diag / 1.5) ** 0.5, 0.5, 1.8))
    return base_spacing * scale


def _surfel_radii(points: np.ndarray, spacing: float, radius_frac: float) -> np.ndarray:
    """Per-surfel disk radius from the LOCAL sample spacing (nearest-neighbour
    distance), so disks track the *achieved* blue-noise density and tile without
    gaps — sparser spots get proportionally larger disks. Outliers (near-isolated
    points) are clamped to the object's median so one stray surfel can't bloat."""
    n = len(points)
    if n < 2:
        return np.full(n, spacing * radius_frac, dtype=np.float32)
    nn = cKDTree(points).query(points, k=2)[0][:, 1]  # distance to nearest neighbour
    good = nn[nn > 0]
    med = float(np.median(good)) if good.size else spacing
    nn = np.where(nn > 0, nn, med)
    nn = np.clip(nn, med * 0.5, med * 2.5)
    return (nn * radius_frac).astype(np.float32)


def surfel_colors(
    geom: trimesh.Trimesh, points: np.ndarray, face_idx: np.ndarray
) -> np.ndarray:
    """Per-surfel RGBA in [0,1]. When the mesh has a readable base-color texture,
    samples it at each surfel's barycentric-interpolated UV (PER-TEXEL albedo, so
    texture detail survives); otherwise falls back to the per-face mean of the
    per-vertex colours (baseColorFactor / grey)."""
    faces = np.asarray(geom.faces[face_idx])  # (S,3)
    visual = getattr(geom, "visual", None)
    material = getattr(visual, "material", None)
    image = getattr(material, "baseColorTexture", None)
    uv = getattr(visual, "uv", None)
    if image is not None and uv is not None:
        try:
            tris = np.asarray(geom.triangles[face_idx])  # (S,3,3)
            bary = trimesh.triangles.points_to_barycentric(tris, points)  # (S,3)
            face_uv = np.asarray(uv)[faces]  # (S,3,2)
            uvs = np.einsum("sj,sjk->sk", bary, face_uv)  # (S,2)
            cols = np.asarray(
                trimesh.visual.color.uv_to_color(uvs, image), dtype=np.float32
            )
            cols = cols / 255.0  # (S,4)
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
            pass  # fall through to the per-vertex approximation
    return _vertex_colors(geom)[faces].mean(axis=1)


def _sample_object(
    geom: trimesh.Trimesh, base_spacing: float, params: SampleParams
) -> dict[str, np.ndarray] | None:
    """Blue-noise sample one placed mesh into surfels. Returns per-surfel arrays
    (position, normal, color rgba, radius) or None if the mesh is empty."""
    if len(geom.faces) == 0 or geom.area <= 0:
        return None
    spacing = _object_spacing(geom, base_spacing, params.adaptive)
    # Upper bound on the even-sample count (Poisson-disk elimination returns ≤).
    budget = int(geom.area / (spacing * spacing) * 2.0) + 8
    try:
        points, face_idx = trimesh.sample.sample_surface_even(
            geom, budget, radius=spacing
        )
    except Exception:
        # Even sampling can fail on degenerate meshes; fall back to plain
        # area-weighted sampling (still gives coverage, just not blue noise).
        points, face_idx = trimesh.sample.sample_surface(geom, budget)
    points = np.asarray(points, dtype=np.float64)
    face_idx = np.asarray(face_idx)
    if len(points) == 0:
        return None

    normals = np.asarray(geom.face_normals[face_idx], dtype=np.float64)
    # Guard against zero-length normals (degenerate faces).
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    normals = normals / lens

    colors = surfel_colors(geom, points, face_idx)  # (S,4) in [0,1]
    radius = _surfel_radii(points, spacing, params.radius_frac)  # sized to spacing
    return {
        "position": points.astype(np.float32),
        "normal": normals.astype(np.float32),
        "color": colors.astype(np.float32),
        "radius": radius,
    }


def _encode_ply(
    positions: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    radii: np.ndarray,
    out_path: Path,
    flatness: float,
) -> None:
    """Write surfels as a standard 3DGS binary `.ply` (SH degree 0). Fields per
    vertex: xyz, normal, f_dc_0..2, opacity, scale_0..2, rot_0..3 — 17 float32."""
    n = positions.shape[0]
    quats = _align_quaternions(normals).astype(np.float32)  # (n,4) w,x,y,z

    rgb = np.clip(colors[:, :3], 0.0, 1.0)
    f_dc = ((rgb - 0.5) / _SH_C0).astype(np.float32)

    alpha = np.clip(colors[:, 3], _ALPHA_CLAMP[0], _ALPHA_CLAMP[1])
    opacity = np.log(alpha / (1.0 - alpha)).astype(np.float32)  # inverse sigmoid

    # Flat disk: two in-plane axes at the surfel radius, the normal axis thin.
    scale = np.empty((n, 3), dtype=np.float32)
    scale[:, 0] = radii
    scale[:, 1] = radii
    scale[:, 2] = radii * flatness
    log_scale = np.log(np.maximum(scale, 1e-9)).astype(np.float32)

    # Column order matches the header below.
    cols = [
        positions[:, 0], positions[:, 1], positions[:, 2],
        normals[:, 0], normals[:, 1], normals[:, 2],
        f_dc[:, 0], f_dc[:, 1], f_dc[:, 2],
        opacity,
        log_scale[:, 0], log_scale[:, 1], log_scale[:, 2],
        quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3],
    ]
    data = np.stack(cols, axis=1).astype("<f4")

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
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
    meshes — one trimesh load each. Area turns a target surfel count into a
    spacing; the diagonal is the auto-density fallback."""
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


def _resolve_base_spacing(params: SampleParams, total_area: float, diag: float) -> float:
    """Base surfel spacing (metres): an explicit `base_spacing` override wins,
    else a `target_splats` count is turned into a spacing via the packing model,
    else it falls back to ~0.3% of the scene diagonal. Always clamped."""
    if params.base_spacing is not None:
        s = params.base_spacing
    elif params.target_splats and total_area > 0:
        s = float((_AREAL_K * total_area / params.target_splats) ** (1.0 / _SPACING_EXP))
    else:
        s = diag * 0.003
    return float(np.clip(s, *_SPACING_CLAMP))


def sample_cell(
    *,
    run: str,
    slot: str,
    model: str,
    raw_dir: Path,
    out_path: Path,
    params: SampleParams = SampleParams(),
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Sample every placed mesh in `raw_dir` into a Gaussian splat written to
    `out_path` (a 3DGS `.ply`). `params` carries the tunable knobs (density via
    `target_splats`, `radius_frac`, `flatness`, `adaptive`). Returns a compact
    summary including the resolved spacing, splat count, and total surface area."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"placed-mesh dir not found: {raw_dir}")
    ids = placed_object_ids(raw_dir)
    if not ids:
        raise FileNotFoundError(f"no placed meshes in {raw_dir}")

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
        except Exception as exc:  # skip a bad mesh, keep going
            warnings.append(f"{node_id}: failed to sample ({type(exc).__name__}: {exc})")
        if progress is not None:
            progress(done, total, node_id)

    if not pos_parts:
        raise RuntimeError("no surfels sampled (every mesh failed or was empty)")

    positions = np.concatenate(pos_parts, axis=0)
    normals = np.concatenate(nrm_parts, axis=0)
    colors = np.concatenate(col_parts, axis=0)
    radii = np.concatenate(rad_parts, axis=0)

    _encode_ply(positions, normals, colors, radii, out_path, params.flatness)

    aabb_min = positions.min(axis=0).tolist()
    aabb_max = positions.max(axis=0).tolist()
    return {
        "run": run,
        "slot": slot,
        "model": model,
        "splats": int(positions.shape[0]),
        "objects_sampled": objects_sampled,
        "objects_total": total,
        "base_spacing": base_spacing,
        "total_area": round(total_area, 2),
        "params": params.as_summary(),
        "scene_aabb": {"min": aabb_min, "max": aabb_max},
        "warnings": warnings,
        "bytes": out_path.stat().st_size,
    }
