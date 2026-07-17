"""Stage 5 — Reference renders (UNLIT) via nvdiffrast, COMPOSITED through glass.

Renders the composed mesh scene from the Stage-4 camera plan into the per-view
supervision the Stage-6 gsplat fine-tune trains against. Surfaces are DEPTH-
PEELED (nvdiffrast `DepthPeeler`, up to `peel_layers` nearest layers per pixel)
and alpha-composited front-to-back, exactly how the splat renderer blends — so
a window pixel shows the room behind the pane, not the bare pane:

  * **RGB** — unlit albedo, "over"-composited: Σ cᵢ·αᵢ·Πⱼ<ᵢ(1−αⱼ), plus the
    background under the remaining transmittance.
  * **depth** — alpha-weighted EXPECTED planar camera-space Z in metres —
    the same statistic gsplat's `render_mode="RGB+ED"` renders, so the depth
    loss compares like with like. For an opaque nearest surface this reduces
    to its plain Z (the old value); 0 where nothing is hit. Persisted as a
    16-bit PNG (log-quantized over the shared [near, far] to uint16, code 0 =
    background) — ~5-10x smaller than raw float32 and near-lossless for the
    alpha-gated depth loss (see `encode_depth_u16` / `load_depth_png`).
  * **alpha** — ACCUMULATED coverage 1−Π(1−αᵢ): glass-over-wall reads ~1 (the
    wall is still there), glass-over-void reads the pane's own alpha.

WHY compositing matters (the glass fix, paired with Stage 2's `occ_lin_glass` +
Stage 4's see-through-glass planning): single-layer references claimed
α = 0.065 and pure-pane RGB for every window pixel, while a correct splat
renders pane-over-room with α ≈ 1 — so the Stage-6 alpha loss actively pushed
everything behind windows transparent. Composited references make the alpha /
photometric / depth losses consistent with what a correct splat produces.
Per glTF, OPAQUE materials ignore their texture's alpha channel entirely
(forced to 1 here, matching Stages 2/3); MASK binarizes at its cutoff; only
BLEND composites fractionally. All-opaque scenes terminate after one extra
(empty) peel pass, so they pay ~nothing.

Plus a `transforms.json` recording, per rendered face, the exact camera-to-world
pose + shared pinhole intrinsics, all in **gsplat's native OpenCV convention**
(see below), so Stage 6 is a thin loop: `viewmats = inv(c2w)`,
`rasterization(..., Ks, render_mode="RGB+ED")`.

CUBEMAP-NATIVE input: Stage 4 emits camera POSITIONS, each tagged with the cube
faces worth rendering (`cameras.json` → `intrinsics` + `cube_faces` + `cameras`).
Every (position, face) becomes one 90° pinhole reference image.

Conventions (LOCKED to match gsplat + nvdiffrast):
  * World: Y-up, right-handed, metres (repo-native; unchanged).
  * Camera: OpenCV — +X right, +Y **down**, +Z **forward** (into the scene). We
    store camera-to-world; Stage 6 inverts to gsplat `viewmats` (world-to-camera).
  * Intrinsics: pinhole `K = [[f,0,cx],[0,f,cy],[0,0,1]]`, f = R/(2·tan(fov/2))
    (= R/2 at the fixed 90° cube FOV), cx = cy = R/2.
  * Depth: planar camera-space Z (not ray distance) — gsplat `ED`.
  * Colour: base-color albedo as stored (sRGB 8-bit); no sRGB↔linear conversion,
    matching Stage 2's surfel init and the gsplat photometric loss.

Renderer: **nvdiffrast** (headless CUDA rasterizer) — CUDA-only, so this stage
runs on the GPU box / Modal, NOT on Apple Silicon. torch + nvdiffrast are imported
LAZILY inside the renderer so this module stays importable (for the server route
and the torch-free pose/IO helpers) on a machine without them.

Two convention flips live behind nvdiffrast and are isolated + asserted by the
smoke tests (see `render_references` docstring): the output is flipped vertically
(nvdiffrast frame memory is bottom-up) and texture V is flipped (glTF v=0 is top,
nvdiffrast t=0 is bottom).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from math import radians, tan
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

# The cube-face basis is defined ONCE in Stage 4; reuse it so the render poses
# match the plan exactly (torch-free import — pulls only numpy/scipy via stage4).
from splat.stage4 import CUBE_FACES

logging.getLogger("trimesh").setLevel(logging.ERROR)

# Written under a cell's `splat/refs/` dir.
REFS_DIRNAME = "refs"
TRANSFORMS_NAME = "transforms.json"

# Depth storage: a 16-bit PNG (not raw float32). Expected planar-Z metres are
# LOG-quantized over the shared [near, far] into uint16 codes 1..65535, with code
# 0 reserved for background (no surface). Log spacing keeps relative precision
# constant (~0.01% — sub-mm near the camera), which is near-lossless for the
# alpha-gated depth L1, while PNG's predictive filters pack the smooth integer
# field ~5-10x smaller than float32. Decoded to metric metres by `load_depth_png`
# (Stage 6); the [near, far] used here is the same pair recorded in
# transforms.json, so decode is unambiguous.
DEPTH_ENCODING = "planar_z_expected_log_uint16_png"
_DEPTH_CODE_MAX = 65535   # uint16 max; code 0 == background

# progress(done, total, current) — called after each rendered view.
ProgressCb = Callable[[int, int, str], None]

_MID_GREY = np.array([0.6, 0.6, 0.6, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class RenderParams:
    """Stage-5 knobs. Intrinsics (resolution, FOV, near/far) are READ from the
    Stage-4 `cameras.json` `intrinsics` block so the renders match the plan; only
    the render-side options live here.

    `peel_layers` caps the depth-peeling passes per view. Peeling stops early
    once every covered pixel is opacity-saturated — an all-opaque view exits
    after the second (empty) pass — so the cap only bites on stacks of many
    BLEND surfaces (each pane multiplies transmittance by ~0.935; truncation
    error after 8 layers is negligible)."""

    background: tuple[float, float, float] = (0.0, 0.0, 0.0)  # empty-pixel RGB
    mask_alpha_cutoff: float = 0.5   # alphaMode=MASK: texel alpha below this → empty
    peel_layers: int = 8             # max composited depth layers per pixel
    save_alpha: bool = True
    save_depth: bool = True

    def as_summary(self) -> dict[str, Any]:
        return {
            "background": list(self.background),
            "mask_alpha_cutoff": self.mask_alpha_cutoff,
            "peel_layers": self.peel_layers,
        }


# --- torch-free layer (pose / intrinsics / IO — runs anywhere, unit-tested) ----


def load_camera_plan(cameras_path: Path) -> dict[str, Any]:
    """Read a Stage-4 `cameras.json`. Requires the cubemap-native schema
    (`intrinsics` + `cube_faces` + `cameras[].faces`)."""
    data = json.loads(cameras_path.read_text(encoding="utf-8"))
    for key in ("intrinsics", "cube_faces", "cameras"):
        if key not in data:
            raise ValueError(
                f"{cameras_path} is not a cubemap-native camera plan (missing "
                f"'{key}'); re-run Stage 4."
            )
    return data


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError(f"cannot normalize near-zero vector {v}")
    return v / n


def opencv_c2w(pos: np.ndarray, forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Camera-to-world 4×4 in OpenCV convention (columns = camera axes in world):
    +Z = `forward` (view dir), +Y = **down** = −`up`, +X = right = Y×Z. `forward`
    and `up` are the cube face's outward direction + image-up."""
    z = _normalize(np.asarray(forward, dtype=np.float64))
    y = -_normalize(np.asarray(up, dtype=np.float64))          # OpenCV +Y is image-down
    x = _normalize(np.cross(y, z))                             # right = Y × Z (right-handed)
    y = np.cross(z, x)                                         # re-orthogonalize
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = x, y, z
    c2w[:3, 3] = np.asarray(pos, dtype=np.float64)
    return c2w


def intrinsics_matrix(resolution: int, fov_deg: float) -> np.ndarray:
    """Pinhole `K` for a square `resolution`² image at `fov_deg`: f = R/(2 tan(fov/2)),
    principal point at the centre (R/2, R/2)."""
    f = (resolution / 2.0) / tan(radians(fov_deg) / 2.0)
    c = resolution / 2.0
    return np.array([[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]], dtype=np.float64)


def gl_projection(K: np.ndarray, width: int, height: int, near: float, far: float) -> np.ndarray:
    """OpenGL clip-space projection (4×4) that maps OpenCV camera coords
    (X right, Y down, Z forward) to nvdiffrast's clip space (NDC x right, y **up**,
    z away from viewer, all in [-1,1]). Encodes the pinhole `K` and the OpenCV→GL
    Y-flip; depth mapping sends Z=near→-1, Z=far→+1."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    a = (far + near) / (far - near)
    b = -2.0 * far * near / (far - near)
    return np.array(
        [
            [2.0 * fx / width, 0.0, 2.0 * cx / width - 1.0, 0.0],
            [0.0, -2.0 * fy / height, 1.0 - 2.0 * cy / height, 0.0],
            [0.0, 0.0, a, b],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def enumerate_views(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the plan into one entry per (camera, face) to render: the render
    id, camera index, face name, world position, and the OpenCV camera-to-world."""
    faces = plan["cube_faces"]
    views: list[dict[str, Any]] = []
    for ci, cam in enumerate(plan["cameras"]):
        pos = np.asarray(cam["pos"], dtype=np.float64)
        for face in cam.get("faces", []):
            name = face["dir"] if isinstance(face, dict) else face
            basis = faces[name]
            c2w = opencv_c2w(pos, np.asarray(basis["forward"]), np.asarray(basis["up"]))
            views.append(
                {
                    "id": f"cam{ci:05d}_{name}",
                    "camera_index": ci,
                    "face": name,
                    "pos": pos,
                    "c2w": c2w,
                    "covers": int(face["covers"]) if isinstance(face, dict) else None,
                }
            )
    return views


def _view_rendered(out_dir: Path, vid: str, save_depth: bool, save_alpha: bool) -> bool:
    """Stage-5 resume signal: True when a view's expected reference images are all
    already on disk. Writes are atomic (`_save_png_atomic` / `save_depth_png`), so a
    present file is a complete one — a resumed render skips these and renders only
    the views still missing."""
    if not (out_dir / "rgb" / f"{vid}.png").is_file():
        return False
    if save_depth and not (out_dir / "depth" / f"{vid}.png").is_file():
        return False
    if save_alpha and not (out_dir / "alpha" / f"{vid}.png").is_file():
        return False
    return True


def encode_depth_u16(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """Planar-Z depth (metres; 0 = background) → uint16 codes for a 16-bit PNG:
    code 0 = background, 1..65535 = LOG-spaced positions over [near, far]. A
    foreground sample is clamped into [near, far] first, so a surface closer than
    `near` still encodes to code 1 (never 0) and stays in the alpha-gated depth
    loss instead of being read back as background."""
    near = max(float(near), 1e-6)
    far = max(float(far), near * (1.0 + 1e-6))
    z = np.clip(depth, near, far)
    t = np.log(z / near) / np.log(far / near)                        # 0..1 over [near, far]
    codes = np.rint(t * (_DEPTH_CODE_MAX - 1)).astype(np.int32) + 1   # 1.._DEPTH_CODE_MAX
    return np.where(depth > 0.0, codes, 0).astype(np.uint16)


def decode_depth_u16(codes: np.ndarray, near: float, far: float) -> np.ndarray:
    """Inverse of `encode_depth_u16`: uint16 codes → planar-Z metres, 0 where the
    code is 0 (background). `near`/`far` must match the encode."""
    near = max(float(near), 1e-6)
    far = max(float(far), near * (1.0 + 1e-6))
    codes = np.asarray(codes)
    t = (codes.astype(np.float32) - 1.0) / (_DEPTH_CODE_MAX - 1)
    z = near * np.exp(t * np.log(far / near))
    return np.where(codes > 0, z, 0.0).astype(np.float32)


def _save_png_atomic(img: Any, path: Path) -> None:
    """Save a PIL image via a temp file + atomic replace, so a crash mid-write never
    leaves a torn PNG. The presence of a view's images is Stage 5's resume signal
    (`_view_rendered`); a half-written one would fool it (and later break the Stage-6
    decode). Format is forced since the `.tmp` suffix hides the extension PIL sniffs."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    img.save(tmp, format="PNG")
    tmp.replace(path)


def save_depth_png(path: Path, depth: np.ndarray, near: float, far: float) -> None:
    """Write planar-Z depth [H,W] as a 16-bit grayscale PNG (log-uint16; see
    `encode_depth_u16`). Atomic (temp-file + replace), like the RGB/alpha writes."""
    from PIL import Image

    codes = encode_depth_u16(depth, near, far)
    h, w = codes.shape
    data = np.ascontiguousarray(codes, dtype="<u2").tobytes()   # little-endian, matches "I;16"
    _save_png_atomic(Image.frombytes("I;16", (w, h), data), path)


def load_depth_png(path: Path, near: float, far: float) -> np.ndarray:
    """Read a 16-bit depth PNG written by `save_depth_png` → planar-Z metres [H,W].
    `near`/`far` must match the encode (both are recorded in transforms.json)."""
    from PIL import Image

    codes = np.asarray(Image.open(path))
    return decode_depth_u16(codes, near, far)


def write_transforms(
    out_dir: Path,
    K: np.ndarray,
    resolution: int,
    near: float,
    far: float,
    frames: list[dict[str, Any]],
    depth_encoding: str = DEPTH_ENCODING,
) -> Path:
    """Write `transforms.json`: shared pinhole intrinsics + per-frame OpenCV
    camera-to-world and artifact paths. `convention` is tagged so Stage 6 knows to
    take `viewmats = inv(transform_matrix)` (gsplat OpenCV), NOT the Nerfstudio
    OpenGL c2w. `depth_encoding` records how the depth maps are stored (the
    log-uint16 PNG scheme, decoded with `near`/`far`); the VALUES are the
    alpha-weighted expected planar Z over the composited layers — the statistic
    gsplat's render_mode="RGB+ED" produces (plain surface Z wherever the nearest
    hit is opaque)."""
    doc = {
        "camera_model": "pinhole",
        "convention": "opencv_c2w",     # transform_matrix is OpenCV camera-to-world
        "depth": depth_encoding,        # expected planar Z, log-uint16 PNG storage
        "color_space": "srgb",
        "w": resolution,
        "h": resolution,
        "fl_x": float(K[0, 0]),
        "fl_y": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "near": float(near),
        "far": float(far),
        "frames": frames,
    }
    out_path = out_dir / TRANSFORMS_NAME
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    tmp.replace(out_path)
    return out_path


# --- nvdiffrast layer (CUDA-only; torch + nvdiffrast imported lazily) ----------


def _require_cuda_renderer():  # noqa: ANN202 - returns (torch, nvdiffrast.torch)
    """Import torch + nvdiffrast and assert a CUDA device, or raise a clear error.
    Kept out of module import so the server (Apple Silicon) can import this file."""
    try:
        import torch
        import nvdiffrast.torch as dr
    except Exception as exc:  # ImportError, or a torch/CUDA build mismatch
        raise RuntimeError(
            "Stage 5 needs torch + nvdiffrast (headless CUDA rasterizer). These are "
            "CUDA-only and not installed here — run Stage 5 on the GPU box / Modal, "
            f"not on Apple Silicon. ({type(exc).__name__}: {exc})"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 5 needs a CUDA GPU; torch reports none available.")
    return torch, dr


def _iter_geoms(mesh: trimesh.Trimesh | trimesh.Scene) -> list[trimesh.Trimesh]:
    # dump() bakes each geometry's scene-graph node transform into world space; plain
    # scene.geometry returns LOCAL vertices, which collapses node-placed objects
    # (generated assets carry placement on the node) to the origin.
    if isinstance(mesh, trimesh.Scene):
        return [g for g in mesh.dump(concatenate=False) if hasattr(g, "faces")]
    return [mesh]


def _material_texture(geom: trimesh.Trimesh) -> tuple[np.ndarray, str]:
    """(RGBA texture [th,tw,4] float32 in [0,1], alpha_mode) for one geometry.
    A readable base-color image is used directly (× baseColorFactor); otherwise a
    1×1 texture of the factor colour (or neutral grey) stands in, so every material
    samples through the same path. glTF's default `alphaMode` is OPAQUE."""
    visual = getattr(geom, "visual", None)
    material = getattr(visual, "material", None)
    alpha_mode = str(getattr(material, "alphaMode", None) or "OPAQUE").upper()

    factor = getattr(material, "baseColorFactor", None)
    fa = np.ones(4, dtype=np.float32)
    if factor is not None:
        arr = np.asarray(factor, dtype=np.float32).reshape(-1)
        if arr.size >= 3:
            if arr.max() > 1.0:
                arr = arr / 255.0
            fa[:3] = arr[:3]
            if arr.size >= 4:
                fa[3] = arr[3]

    image = getattr(material, "baseColorTexture", None)
    uv = getattr(visual, "uv", None)
    if image is not None and uv is not None:
        try:
            tex = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0  # [th,tw,4]
            tex = tex * fa[None, None, :]  # fold in baseColorFactor
            return np.ascontiguousarray(tex), alpha_mode
        except Exception:
            pass  # unreadable image → fall through to the flat factor texture
    # Flat 1×1 texture: the factor colour, or neutral grey when there's no material.
    flat = fa if factor is not None else _MID_GREY
    return flat.reshape(1, 1, 4).copy(), alpha_mode


def placed_object_ids(raw_dir: Path) -> list[str]:
    """Ids with a placed world-space mesh in `raw_dir` (`<id>.glb`, excluding the
    `<id>.raw.glb` pre-placement intermediates)."""
    return sorted(
        p.name[: -len(".glb")]
        for p in raw_dir.glob("*.glb")
        if not p.name.endswith(".raw.glb")
    )


# nvdiffrast's CudaRaster has TWO independent per-rasterize() limits:
#   * the triangle id is returned in a float32 channel (24-bit mantissa), so at most
#     2**24 triangles can be indexed exactly; AND
#   * the coarse binner stores triangle-tile "subtriangles" in a fixed segment buffer
#     that overflows ("subtriangle count overflow") far BELOW 2**24 when triangles are
#     large or densely overlapping — it scales with triangles × tiles-covered ×
#     resolution, so a ~16M-triangle 1024² view of a dense scene trips it even though
#     the id limit is fine. (Small, scattered triangles rasterize fine well past 12M.)
# So a view is DEPTH-PEELED in conservative ≤_RASTER_CHUNK_TRIS triangle batches
# (well under both limits) whose peeled layers are merged into one per-pixel
# nearest-first layer stack; a batch that STILL overflows the bin buffer is halved
# and retried. That overflow is a CHECKED failure that leaves the CUDA context
# intact (unlike the CUDA-700 fault from pathological overdraw), so the retry is
# safe. Merging is exact: the global k-th nearest surface at a pixel is always
# among its own batch's k nearest, so sorting the union of per-batch layer stacks
# reproduces the global peel order.
_NVDIFFRAST_MAX_TRIS = 1 << 24     # hard triangle-id ceiling (float32 mantissa)
_RASTER_CHUNK_TRIS = 2_000_000     # initial triangles per rasterize() call (subtriangle-safe)
_MIN_CHUNK_TRIS = 65_536           # split floor; a subtriangle overflow below this re-raises


def _frustum_planes(mvp: np.ndarray) -> np.ndarray:
    """The 6 frustum planes [6,4] (a,b,c,d) of a world->clip matrix `mvp` (row-major,
    clip = mvp @ [x,y,z,1]); a world point is inside when plane·[x,y,z,1] >= 0. The
    Gribb-Hartmann combinations of the clip rows (left,right,bottom,top,near,far)."""
    r0, r1, r2, r3 = mvp[0], mvp[1], mvp[2], mvp[3]
    return np.stack([r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 + r2, r3 - r2])


def _objects_in_frustum(
    aabb_min: np.ndarray, aabb_max: np.ndarray, planes: np.ndarray
) -> np.ndarray:
    """Conservative AABB-vs-frustum test -> [G] bool keep mask: an object is kept
    unless its AABB is fully outside some plane (the corner furthest toward the plane
    normal is still outside). Never culls a partially-visible box."""
    n = planes[:, :3]                                                        # [6,3]
    pv = np.where(n[None] > 0.0, aabb_max[:, None, :], aabb_min[:, None, :])  # [G,6,3]
    dot = np.einsum("gpc,pc->gp", pv, n) + planes[:, 3]                       # [G,6]
    return (dot >= 0.0).all(axis=1)


class _Scene:
    """Composed-scene GPU buffers, built once per cell: one global vertex/triangle
    soup + per-triangle material index + per-material RGBA textures & alpha modes.
    Also records per-object (per-geom) world AABBs + triangle ranges (on the CPU) so
    Stage-5 can frustum-cull each view."""

    def __init__(self, torch, raw_dir: Path, ids: list[str]):  # noqa: ANN001
        verts: list[np.ndarray] = []
        faces: list[np.ndarray] = []
        uvs: list[np.ndarray] = []
        tri_mat: list[np.ndarray] = []
        aabb_min: list[np.ndarray] = []
        aabb_max: list[np.ndarray] = []
        tri_count: list[int] = []
        self.textures: list[Any] = []          # list of [1,th,tw,4] cuda tensors
        self.alpha_modes: list[str] = []
        self.warnings: list[str] = []
        voff = 0
        for node_id in ids:
            try:
                mesh = trimesh.load(raw_dir / f"{node_id}.glb", process=False)
            except Exception as exc:
                self.warnings.append(f"{node_id}: failed to load ({type(exc).__name__})")
                continue
            for geom in _iter_geoms(mesh):
                nf = len(geom.faces)
                if nf == 0:
                    continue
                v = np.asarray(geom.vertices, dtype=np.float32)
                uv = getattr(getattr(geom, "visual", None), "uv", None)
                uv = (
                    np.asarray(uv, dtype=np.float32)
                    if uv is not None and len(uv) == len(v)
                    else np.zeros((len(v), 2), dtype=np.float32)
                )
                tex, alpha_mode = _material_texture(geom)
                mat_id = len(self.textures)
                verts.append(v)
                uvs.append(uv)
                faces.append(np.asarray(geom.faces, dtype=np.int32) + voff)
                tri_mat.append(np.full(nf, mat_id, dtype=np.int64))
                aabb_min.append(v.min(axis=0))
                aabb_max.append(v.max(axis=0))
                tri_count.append(nf)
                self.textures.append(
                    torch.as_tensor(tex, dtype=torch.float32, device="cuda")[None]
                )
                self.alpha_modes.append(alpha_mode)
                voff += len(v)
            del mesh

        if not verts:
            raise RuntimeError("no renderable geometry (every mesh empty or failed)")
        self.verts = torch.as_tensor(np.concatenate(verts), dtype=torch.float32, device="cuda")
        self.uv = torch.as_tensor(np.concatenate(uvs), dtype=torch.float32, device="cuda")
        self.tris = torch.as_tensor(np.concatenate(faces), dtype=torch.int32, device="cuda")
        self.tri_mat = torch.as_tensor(
            np.concatenate(tri_mat), dtype=torch.int64, device="cuda"
        )
        # Homogeneous world positions, cached for the per-view world→cam transform.
        ones = torch.ones((self.verts.shape[0], 1), dtype=torch.float32, device="cuda")
        self.verts_h = torch.cat([self.verts, ones], dim=1)  # [V,4]
        self.n_materials = len(self.textures)

        # Per-object world AABBs + triangle ranges (kept on the CPU) for frustum
        # culling: the cull test runs in numpy with no GPU sync, and the ranges index
        # self.tris / self.tri_mat directly (vertices stay shared and uncapped, so
        # culling only ever subsets triangles).
        counts = np.asarray(tri_count, dtype=np.int64)
        self.obj_tri_count = counts
        self.obj_tri_start = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        self.obj_aabb_min = np.asarray(aabb_min, dtype=np.float32)  # [G,3] world
        self.obj_aabb_max = np.asarray(aabb_max, dtype=np.float32)  # [G,3] world
        self.n_objects = int(counts.shape[0])


def _visible_tris(torch, scene, keep):  # noqa: ANN001
    """(tris, tri_mat) restricted to the frustum-visible objects: the full buffers
    when every object is kept (zero-copy), else the concatenation of the kept
    objects' contiguous triangle ranges (copies only the visible triangles)."""
    if keep.all():
        return scene.tris, scene.tri_mat
    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        return scene.tris[:0], scene.tri_mat[:0]
    starts, counts = scene.obj_tri_start[idx], scene.obj_tri_count[idx]
    tri_parts = [scene.tris[int(s) : int(s + c)] for s, c in zip(starts, counts)]
    mat_parts = [scene.tri_mat[int(s) : int(s + c)] for s, c in zip(starts, counts)]
    return torch.cat(tri_parts), torch.cat(mat_parts)


# A pixel counts as opacity-saturated (deeper layers invisible) once its
# remaining transmittance drops below this; also the floor for the expected-
# depth normalization so empty pixels stay exactly 0.
_SATURATION_EPS = 1e-4


def _render_view(torch, dr, glctx, scene, c2w, proj_np, resolution, params):  # noqa: ANN001
    """Render one pinhole view → (rgb [H,W,3], depth [H,W], alpha [H,W]) as numpy,
    already flipped to top-down (row 0 = top) to match gsplat / normal images.

    DEPTH-PEELED COMPOSITING, IN TRIANGLE BATCHES. Two mechanisms compose here:

      * PEELING (glass correctness): surfaces are extracted nearest-first
        (nvdiffrast `DepthPeeler`) and blended front-to-back with the standard
        "over" operator — per layer i, weight w = (remaining transmittance)·αᵢ;
        rgb += w·cᵢ; alpha += w; the depth accumulator += w·zᵢ is normalized to
        EXPECTED depth (gsplat "ED") at the end.
      * BATCHING (feasibility): the frustum-visible triangles are peeled in
        ≤_RASTER_CHUNK_TRIS batches (halve-and-retry on a checked bin-buffer
        overflow), and each batch's peeled layers are merged into ONE global
        per-pixel stack of the `peel_layers` nearest layers, sorted by depth.
        The merge is exact: a pixel's global k-th nearest surface is always
        among its own batch's k nearest, so sorting the union of per-batch
        stacks reproduces the global peel order — a single-batch view runs the
        original peel bit-for-bit.

    Per batch, peeling stops early when the peeler exhausts (a layer covers
    nothing) or when every covered pixel is already behind an OPAQUE stacked
    layer (all-opaque views do one real pass + one raster-only check pass).

    Objects whose world AABB is fully outside the view frustum are culled before
    rasterizing — exact for an unlit rasterizer (off-screen geometry can't affect
    a flat-albedo image), and the cheap way to shrink the triangle count.
    `c2w`/`proj_np` are numpy so the cull test runs on the CPU without a GPU
    sync."""
    w2c_np = np.linalg.inv(c2w)                     # OpenCV world→camera [4,4]
    keep = _objects_in_frustum(
        scene.obj_aabb_min, scene.obj_aabb_max, _frustum_planes(proj_np @ w2c_np)
    )
    if not keep.any():                              # nothing in view → background frame
        empty = np.zeros((resolution, resolution), dtype=np.float32)
        rgb_bg = np.broadcast_to(
            np.asarray(params.background, dtype=np.float32), (resolution, resolution, 3)
        ).copy()
        return rgb_bg, empty, empty.copy()
    tris, tri_mat = _visible_tris(torch, scene, keep)

    # Per-vertex transforms are shared by every triangle batch → compute them once
    # over the full vertex buffer; batches only ever subset the triangle indices.
    w2c = torch.as_tensor(w2c_np, dtype=torch.float32, device="cuda")
    proj = torch.as_tensor(proj_np, dtype=torch.float32, device="cuda")
    v_cam = scene.verts_h @ w2c.T                   # [V,4]; column 2 = planar Z
    cam_z = v_cam[:, 2:3].contiguous()              # planar depth attribute (metres)
    pos_clip = (v_cam @ proj.T).contiguous()        # [V,4] clip space

    # Global nearest-first layer stack across batches: [L,H,W(,3)]. Empty slots
    # carry z=+inf / a=0 so they sort to the back and composite to nothing.
    n_layers = max(1, params.peel_layers)
    inf = float("inf")
    z_stk = torch.full((n_layers, resolution, resolution), inf, device="cuda")
    a_stk = torch.zeros((n_layers, resolution, resolution), device="cuda")
    rgb_stk = torch.zeros((n_layers, resolution, resolution, 3), device="cuda")
    # Nearest OPAQUE stacked depth per pixel — layers behind it get weight 0
    # exactly, so batches may skip them (the per-batch early-exit).
    opaque_front = torch.full((resolution, resolution), inf, device="cuda")

    def _insert_layer(z_l, a_l, rgb_l):  # noqa: ANN001 - [H,W], [H,W], [H,W,3]
        nonlocal z_stk, a_stk, rgb_stk, opaque_front
        z_all = torch.cat([z_stk, z_l[None]], dim=0)
        a_all = torch.cat([a_stk, a_l[None]], dim=0)
        rgb_all = torch.cat([rgb_stk, rgb_l[None]], dim=0)
        order = z_all.argsort(dim=0)[:n_layers]                     # [L,H,W]
        z_stk = torch.gather(z_all, 0, order)
        a_stk = torch.gather(a_all, 0, order)
        rgb_stk = torch.gather(rgb_all, 0, order[..., None].expand(-1, -1, -1, 3))
        opaque_front = torch.where(
            a_stk >= 0.999, z_stk, torch.full_like(z_stk, inf)
        ).amin(dim=0)

    # Triangle ranges (start, count) as a LIFO stack: start at the conservative
    # chunk size; a range that still overflows the bin buffer is halved and
    # retried (depth-merge is order-independent, so any split is equivalent).
    n_tris = int(tris.shape[0])
    work: list[tuple[int, int]] = []
    start = 0
    while start < n_tris:
        count = min(_RASTER_CHUNK_TRIS, n_tris - start)
        work.append((start, count))
        start += count
    while work:
        start, count = work.pop()
        tris_b = tris[start : start + count]
        tri_mat_b = tri_mat[start : start + count]
        with dr.DepthPeeler(glctx, pos_clip[None], tris_b, (resolution, resolution)) as peeler:
            for _layer in range(n_layers):
                try:
                    rast, _ = peeler.rasterize_next_layer()
                except RuntimeError as exc:
                    # "subtriangle count overflow" is a CHECKED bin-buffer overflow
                    # (context intact) → halve this range and retry the halves.
                    # Every peel pass rasterizes the same triangle set, so an
                    # overflow can only occur on the batch's FIRST pass — i.e.
                    # before anything from this batch entered the stack; a later
                    # -pass overflow is unexpected and propagates.
                    if (
                        _layer == 0
                        and count > _MIN_CHUNK_TRIS
                        and "subtriangle" in str(exc).lower()
                    ):
                        half = count // 2
                        work.append((start, half))
                        work.append((start + half, count - half))
                        break
                    raise
                tri_id = rast[..., 3].long()[0]         # [H,W]; 0 = empty, else idx+1
                covered = tri_id > 0
                if not covered.any():                   # this batch is exhausted
                    break
                # Per-pixel planar depth (perspective-correct camera-space Z).
                z_l = dr.interpolate(cam_z[None], rast, tris_b)[0][0, ..., 0]  # [H,W]
                # Skip pixels already behind an opaque stacked layer — they can
                # never contribute (transmittance there is exactly 0).
                useful = covered & (z_l < opaque_front)
                if not useful.any():
                    break
                # Per-pixel UV; flip V because glTF v=0 is the texture TOP but
                # nvdiffrast t=0 is the BOTTOM (see module docstring / smoke test).
                uv, _ = dr.interpolate(scene.uv[None], rast, tris_b)  # [1,H,W,2]
                uv = torch.cat([uv[..., 0:1], 1.0 - uv[..., 1:2]], dim=-1).contiguous()

                # Per-pixel material id (gather via triangle id into THIS batch's
                # tri_mat), then shade each material present among useful pixels.
                mat_pix = torch.where(
                    covered, tri_mat_b[(tri_id - 1).clamp(min=0)],
                    tri_id.new_full(tri_id.shape, -1),
                )
                layer_rgb = torch.zeros((resolution, resolution, 3), device="cuda")
                layer_a = torch.zeros((resolution, resolution), device="cuda")
                for m in torch.unique(mat_pix[useful]).tolist():
                    sel = (mat_pix == m) & useful                    # [H,W]
                    col = dr.texture(
                        scene.textures[m], uv, filter_mode="linear", boundary_mode="wrap"
                    )[0]                                             # [H,W,4]
                    a = col[..., 3]
                    mode = scene.alpha_modes[m]
                    if mode == "MASK":
                        a = (a >= params.mask_alpha_cutoff).float()
                    elif mode != "BLEND":
                        a = torch.ones_like(a)  # glTF OPAQUE ignores texture alpha
                    layer_rgb = torch.where(sel[..., None], col[..., :3], layer_rgb)
                    layer_a = torch.where(sel, a, layer_a)

                _insert_layer(
                    torch.where(useful, z_l, torch.full_like(z_l, inf)),
                    torch.where(useful, layer_a, torch.zeros_like(layer_a)),
                    layer_rgb,
                )

    # Composite the merged stack front-to-back (the standard "over" operator) —
    # identical math to single-pass peeling; expected depth = Σw·z / Σw, exact
    # surface Z wherever the first hit is opaque. Background under the remaining
    # transmittance, matching how the splat premultiplies over the (black)
    # training background.
    trans = torch.ones((resolution, resolution), device="cuda")
    alpha = torch.zeros((resolution, resolution), device="cuda")
    rgb = torch.zeros((resolution, resolution, 3), device="cuda")
    depth_acc = torch.zeros((resolution, resolution), device="cuda")
    for l in range(n_layers):
        a_l = a_stk[l]
        w = trans * a_l
        rgb = rgb + w[..., None] * rgb_stk[l]
        depth_acc = depth_acc + w * torch.where(torch.isfinite(z_stk[l]), z_stk[l], 0.0)
        alpha = alpha + w
        trans = trans * (1.0 - a_l)
    depth = torch.where(
        alpha > _SATURATION_EPS,
        depth_acc / alpha.clamp(min=_SATURATION_EPS),
        torch.zeros_like(depth_acc),
    )
    bg = torch.tensor(params.background, dtype=torch.float32, device="cuda")
    rgb = rgb + (1.0 - alpha)[..., None] * bg

    # nvdiffrast frame memory is bottom-up → flip vertically for top-down images.
    def _top_down(t):  # noqa: ANN001
        return torch.flip(t, dims=[0]).detach().cpu().numpy()

    return _top_down(rgb), _top_down(depth), _top_down(alpha)


def render_references(
    *,
    run: str,
    slot: str,
    model: str,
    raw_dir: Path,
    cameras_path: Path,
    out_dir: Path,
    params: RenderParams = RenderParams(),
    resume: bool = True,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Render every (camera, face) in `cameras_path` from the placed vanilla meshes
    in `raw_dir` into `out_dir` (`rgb/`, `depth/`, `alpha/`, `transforms.json`).

    Resume-safe: with `resume` (default), views whose images are already on disk are
    skipped and only the missing ones re-render — so an interrupted render continues
    where it stopped. `transforms.json` is rebuilt from the full, deterministic view
    list either way, so a resumed run reproduces the identical file. Pass
    `resume=False` (the server wipes `refs/` for this) to force a full re-render.

    Requires a CUDA GPU + nvdiffrast only when something is left to render (raises a
    clear error then); a finalize-only resume — every image already present, just
    `transforms.json` missing after a crash — completes on any host. Returns a
    compact summary.

    Smoke tests to run once on the GPU box (they validate the convention flips +
    the peeling path, none of which can be checked without a device):
      1. Pose round-trip — feed a frame's c2w + K into gsplat on the Stage-2 cloud;
         the splat render must align pixel-for-pixel with the reference RGB.
      2. Depth — the rendered planar Z of a known box equals the measured
         perpendicular distance (not the ray distance), surviving the log-uint16
         PNG round-trip within ~0.01%.
      3. Glass compositing — a BLEND pane (alpha a) at z1 in front of an opaque
         wall at z2 must produce rgb = a·pane + (1−a)·wall, alpha = 1, and
         depth = a·z1 + (1−a)·z2 (expected depth); with no wall behind it,
         alpha = a and rgb = a·pane + (1−a)·background. An all-opaque view must
         match the pre-peeling output exactly, including when triangle batching
         engages (force a tiny _RASTER_CHUNK_TRIS to exercise the stack merge).
    """
    from PIL import Image

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"placed-mesh dir not found: {raw_dir}")
    ids = placed_object_ids(raw_dir)
    if not ids:
        raise FileNotFoundError(f"no placed meshes in {raw_dir}")

    plan = load_camera_plan(cameras_path)
    intr = plan["intrinsics"]
    resolution = int(intr["resolution"])
    if resolution % 8 != 0:  # nvdiffrast CUDA rasterizer wants multiples of 8
        raise ValueError(f"render resolution {resolution} must be a multiple of 8")
    fov = float(intr["face_fov_deg"])
    near, far = float(intr["near"]), float(intr["far"])
    K = intrinsics_matrix(resolution, fov)
    proj_np = gl_projection(K, resolution, resolution, near, far)

    views = enumerate_views(plan)
    total = len(views)
    if total == 0:
        raise RuntimeError("camera plan has no faces to render")

    for sub in ("rgb", "depth", "alpha"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # Resume: render only the views whose images aren't already on disk. The plan is
    # deterministic, so `frames` (below) is assembled for every view regardless of
    # which this run rendered — the finished `transforms.json` is identical.
    pending = {
        v["id"]
        for v in views
        if not (resume and _view_rendered(out_dir, v["id"], params.save_depth, params.save_alpha))
    }
    skipped = total - len(pending)

    # The GPU rasterizer (and the CUDA requirement) is only needed when there's
    # something left to render, so a finalize-only resume completes on any host.
    torch = dr = glctx = scene = None
    warnings: list[str] = []
    if pending:
        torch, dr = _require_cuda_renderer()
        glctx = dr.RasterizeCudaContext()
        scene = _Scene(torch, raw_dir, ids)
        warnings = scene.warnings

    if progress is not None:
        progress(0, total, "resume" if skipped else "")

    frames: list[dict[str, Any]] = []
    rendered = 0
    for done, view in enumerate(views, start=1):
        vid = view["id"]
        if vid in pending:
            rgb, depth, alpha = _render_view(
                torch, dr, glctx, scene, view["c2w"], proj_np, resolution, params
            )
            _save_png_atomic(
                Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)),
                out_dir / "rgb" / f"{vid}.png",
            )
            if params.save_depth:
                save_depth_png(out_dir / "depth" / f"{vid}.png", depth, near, far)
            if params.save_alpha:
                _save_png_atomic(
                    Image.fromarray((np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)),
                    out_dir / "alpha" / f"{vid}.png",
                )
            rendered += 1
        frame: dict[str, Any] = {
            "file_path": f"rgb/{vid}.png",
            "camera_index": view["camera_index"],
            "face": view["face"],
            "transform_matrix": view["c2w"].tolist(),
        }
        if params.save_depth:
            frame["depth_path"] = f"depth/{vid}.png"
        if params.save_alpha:
            frame["alpha_path"] = f"alpha/{vid}.png"
        frames.append(frame)
        if progress is not None:
            progress(done, total, vid)

    write_transforms(
        out_dir, K, resolution, near, far, frames,
        depth_encoding=DEPTH_ENCODING if params.save_depth else "none",
    )

    return {
        "run": run,
        "slot": slot,
        "model": model,
        "views": total,
        "views_rendered": rendered,
        "views_skipped": skipped,
        "cameras": len(plan["cameras"]),
        "resolution": resolution,
        "materials": scene.n_materials if scene is not None else None,
        "params": params.as_summary(),
        "warnings": warnings,
        "out_dir": str(out_dir),
    }


def _main() -> None:
    """Standalone CLI for the GPU box: render one cell's references from an already
    de-optimized vanilla mesh dir + its Stage-4 `cameras.json`."""
    import argparse

    ap = argparse.ArgumentParser(description="Stage 5 — unlit nvdiffrast reference renders")
    ap.add_argument("--raw-dir", required=True, type=Path, help="vanilla placed GLBs")
    ap.add_argument("--cameras", required=True, type=Path, help="Stage-4 cameras.json")
    ap.add_argument("--out-dir", required=True, type=Path, help="output refs/ dir")
    ap.add_argument("--run", default="?")
    ap.add_argument("--slot", default="?")
    ap.add_argument("--model", default="?")
    args = ap.parse_args()

    def _log(done: int, total: int, current: str) -> None:
        if done == 0 or done == total or done % 50 == 0:
            print(f"[stage5] {done}/{total} {current}")

    summary = render_references(
        run=args.run, slot=args.slot, model=args.model,
        raw_dir=args.raw_dir, cameras_path=args.cameras, out_dir=args.out_dir,
        progress=_log,
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    _main()
