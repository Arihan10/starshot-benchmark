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
    to its plain Z (the old value); 0 where nothing is hit.
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


def write_transforms(
    out_dir: Path,
    K: np.ndarray,
    resolution: int,
    near: float,
    far: float,
    frames: list[dict[str, Any]],
) -> Path:
    """Write `transforms.json`: shared pinhole intrinsics + per-frame OpenCV
    camera-to-world and artifact paths. `convention` is tagged so Stage 6 knows to
    take `viewmats = inv(transform_matrix)` (gsplat OpenCV), NOT the Nerfstudio
    OpenGL c2w."""
    doc = {
        "camera_model": "pinhole",
        "convention": "opencv_c2w",     # transform_matrix is OpenCV camera-to-world
        # Alpha-weighted expected planar Z over the composited layers (metres) —
        # the statistic gsplat's render_mode="RGB+ED" produces; plain surface Z
        # wherever the nearest hit is opaque.
        "depth": "planar_z_expected_metric",
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


# nvdiffrast's CudaRaster packs the triangle index into 24 bits, so ONE rasterize()
# call handles at most 2**24 triangles; beyond that it faults (CUDA 700) and poisons
# the context. Frustum culling keeps most views under this; a view still over it after
# culling raises instead of crashing.
_NVDIFFRAST_MAX_TRIS = 1 << 24  # 16,777,216


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

    DEPTH-PEELED COMPOSITING: layers are rasterized nearest-first (nvdiffrast
    `DepthPeeler`; each pass reports the next surface behind the previous one)
    and blended front-to-back with the standard "over" operator — per layer i,
    weight w = (remaining transmittance)·αᵢ; rgb += w·cᵢ; alpha += w; the
    depth accumulator += w·zᵢ is normalized to EXPECTED depth at the end
    (gsplat "ED"). Peeling stops as soon as no covered pixel has transmittance
    left, so all-opaque views do exactly one real pass + one empty one.

    Objects whose world AABB is fully outside the view frustum are culled before
    rasterizing — exact for an unlit rasterizer (off-screen geometry can't affect a
    flat-albedo image), and the mechanism that keeps a view's triangle count under
    nvdiffrast's per-call limit. `c2w`/`proj_np` are numpy so the cull test runs on
    the CPU without a GPU sync."""
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
    if int(tris.shape[0]) > _NVDIFFRAST_MAX_TRIS:
        raise RuntimeError(
            f"{int(tris.shape[0]):,} triangles visible in one view after frustum "
            f"culling exceeds nvdiffrast's {_NVDIFFRAST_MAX_TRIS:,} per-call limit — "
            f"use the optimized (lighter) source or triangle-chunked rendering."
        )

    w2c = torch.as_tensor(w2c_np, dtype=torch.float32, device="cuda")
    proj = torch.as_tensor(proj_np, dtype=torch.float32, device="cuda")
    v_cam = scene.verts_h @ w2c.T                   # [V,4]; column 2 = planar Z
    cam_z = v_cam[:, 2:3].contiguous()              # planar depth attribute (metres)
    pos_clip = (v_cam @ proj.T).contiguous()        # [V,4] clip space

    rgb = torch.zeros((1, resolution, resolution, 3), dtype=torch.float32, device="cuda")
    alpha = torch.zeros((1, resolution, resolution, 1), dtype=torch.float32, device="cuda")
    depth_acc = torch.zeros_like(alpha)             # Σ w·z, normalized at the end
    with dr.DepthPeeler(glctx, pos_clip[None], tris, (resolution, resolution)) as peeler:
        for _layer in range(max(1, params.peel_layers)):
            rast, _ = peeler.rasterize_next_layer()
            tri_id = rast[..., 3].long()            # [1,H,W]; 0 = empty, else idx+1
            covered = tri_id > 0
            trans = 1.0 - alpha                     # transmittance BEFORE this layer
            active = covered & (trans[..., 0] > _SATURATION_EPS)  # [1,H,W]
            if not active.any():                    # no geometry left, or all saturated
                break

            # Per-pixel planar depth (perspective-correct camera-space Z) + UV;
            # flip V because glTF v=0 is the texture TOP but nvdiffrast t=0 is
            # the BOTTOM (see module docstring / smoke test).
            layer_z, _ = dr.interpolate(cam_z[None], rast, tris)     # [1,H,W,1]
            uv, _ = dr.interpolate(scene.uv[None], rast, tris)       # [1,H,W,2]
            uv = torch.cat([uv[..., 0:1], 1.0 - uv[..., 1:2]], dim=-1).contiguous()

            # Per-pixel material id (gather via triangle id into the VISIBLE
            # subset's tri_mat), then shade each material present in this layer.
            mat_pix = torch.where(
                covered, tri_mat[(tri_id - 1).clamp(min=0)], tri_id.new_full(tri_id.shape, -1)
            )
            layer_rgb = torch.zeros_like(rgb)
            layer_a = torch.zeros_like(alpha)
            for m in torch.unique(mat_pix[active]).tolist():
                sel = ((mat_pix == m) & active)[..., None]           # [1,H,W,1]
                col = dr.texture(
                    scene.textures[m], uv, filter_mode="linear", boundary_mode="wrap"
                )
                a = col[..., 3:4]
                mode = scene.alpha_modes[m]
                if mode == "MASK":
                    a = (a >= params.mask_alpha_cutoff).float()
                elif mode != "BLEND":
                    a = torch.ones_like(a)  # glTF OPAQUE ignores texture alpha
                layer_rgb = torch.where(sel, col[..., :3], layer_rgb)
                layer_a = torch.where(sel, a, layer_a)

            # Front-to-back "over": this layer contributes what still gets through.
            w = trans * layer_a                     # layer_a is 0 off-active pixels
            rgb = rgb + w * layer_rgb
            depth_acc = depth_acc + w * layer_z
            alpha = alpha + w

    # Expected depth (Σ w·z / Σ w); exact surface Z wherever the first hit is
    # opaque. Background under the remaining transmittance, matching how the
    # splat premultiplies over the (black) training background.
    depth = torch.where(
        alpha > _SATURATION_EPS,
        depth_acc / alpha.clamp(min=_SATURATION_EPS),
        torch.zeros_like(depth_acc),
    )
    bg = torch.tensor(params.background, dtype=torch.float32, device="cuda")
    rgb = rgb + (1.0 - alpha) * bg

    # nvdiffrast frame memory is bottom-up → flip vertically for top-down images.
    def _top_down(t):  # noqa: ANN001
        return torch.flip(t[0], dims=[0]).detach().cpu().numpy()

    return _top_down(rgb), _top_down(depth)[..., 0], _top_down(alpha)[..., 0]


def render_references(
    *,
    run: str,
    slot: str,
    model: str,
    raw_dir: Path,
    cameras_path: Path,
    out_dir: Path,
    params: RenderParams = RenderParams(),
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Render every (camera, face) in `cameras_path` from the placed vanilla meshes
    in `raw_dir` into `out_dir` (`rgb/`, `depth/`, `alpha/`, `transforms.json`).

    Requires a CUDA GPU + nvdiffrast (raises a clear error otherwise). Returns a
    compact summary.

    Smoke tests to run once on the GPU box (they validate the convention flips +
    the peeling path, none of which can be checked without a device):
      1. Pose round-trip — feed a frame's c2w + K into gsplat on the Stage-2 cloud;
         the splat render must align pixel-for-pixel with the reference RGB.
      2. Depth — the rendered planar Z of a known box equals the measured
         perpendicular distance (not the ray distance).
      3. Glass compositing — a BLEND pane (alpha a) at z1 in front of an opaque
         wall at z2 must produce rgb = a·pane + (1−a)·wall, alpha = 1, and
         depth = a·z1 + (1−a)·z2 (expected depth); with no wall behind it,
         alpha = a and rgb = a·pane + (1−a)·background. An all-opaque view must
         match the pre-peeling output exactly.
    """
    torch, dr = _require_cuda_renderer()
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

    glctx = dr.RasterizeCudaContext()
    scene = _Scene(torch, raw_dir, ids)

    if progress is not None:
        progress(0, total, "")

    frames: list[dict[str, Any]] = []
    for done, view in enumerate(views, start=1):
        rgb, depth, alpha = _render_view(torch, dr, glctx, scene, view["c2w"], proj_np, resolution, params)

        vid = view["id"]
        rgb_path = out_dir / "rgb" / f"{vid}.png"
        Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)).save(rgb_path)
        frame: dict[str, Any] = {
            "file_path": f"rgb/{vid}.png",
            "camera_index": view["camera_index"],
            "face": view["face"],
            "transform_matrix": view["c2w"].tolist(),
        }
        if params.save_depth:
            np.save(out_dir / "depth" / f"{vid}.npy", depth.astype(np.float32))
            frame["depth_path"] = f"depth/{vid}.npy"
        if params.save_alpha:
            Image.fromarray((np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)).save(
                out_dir / "alpha" / f"{vid}.png"
            )
            frame["alpha_path"] = f"alpha/{vid}.png"
        frames.append(frame)
        if progress is not None:
            progress(done, total, vid)

    write_transforms(out_dir, K, resolution, near, far, frames)

    return {
        "run": run,
        "slot": slot,
        "model": model,
        "views": total,
        "cameras": len(plan["cameras"]),
        "resolution": resolution,
        "materials": scene.n_materials,
        "params": params.as_summary(),
        "warnings": scene.warnings,
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
