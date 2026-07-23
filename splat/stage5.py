"""Stage 5 — Reference renders (UNLIT) via the debug viewer's WebGL stack.

Stage 5 turns the Stage-4 camera plan into the per-view supervision the Stage-6
gsplat fine-tune trains against. It is split across three pieces; THIS module is
the torch-free CONTRACT they all share (poses, intrinsics, the SZF frame codec,
resume signal, transforms writer) — it renders nothing itself:

  * `client/public/js/splatcapture.js` — the RENDERER: a headless capture page
    running the exact same WebGL stack as the debug viewer (three.js GLTFLoader
    + KTX2Loader + MeshoptDecoder). It loads the cell's SPLAT asset tier
    (KTX2/ETC1S textures stay GPU-compressed, Meshopt geometry stays quantized),
    renders every (camera, face) of the plan unlit, and streams raw frames back.
  * `server/app/services/refcapture.py` — the ORCHESTRATOR: launches a headless
    Chromium/Edge at the capture page, ingests its frame batches, and encodes
    frames on a process pool (this module's `write_reference_frame`).
  * `server/app/api/routes.py` — the stage-5 endpoints (start/status, the
    capture manifest/frames/finish protocol, and the on-demand PNG preview).

STORAGE: one **SZF frame container** per view, `refs/frames/{view}.szf` — a
16-byte header + two zstd streams (RGBA8, then uint16 depth codes), each
sub-left-filtered first (PNG's "Sub" predictor: horizontal per-channel deltas,
which zstd then crushes). Bit-exact lossless, but ~50x cheaper to encode than
PNG's zlib (~3 ms vs ~90-175 ms per 512² view) and ~4x faster for Stage 6 to
read back — so frame encoding never paces the renderer, and the trainer's data
loader speeds up too, at ~1.2x PNG's disk footprint (measured on real frames).
PNG exists only on demand: the server transcodes single frames for the debug
UI's patch inspector (`frame_preview_png`); machines never read PNG.

Per view the SZF planes are:

  * **RGB** — unlit albedo (base-color texel × baseColorFactor; no lighting, no
    tone mapping, no sRGB↔linear conversion — texels pass through and BLEND in
    the stored sRGB space, exactly how gsplat composites its training target).
    Transparent (alphaMode=BLEND) surfaces alpha-blend over what's behind them;
    empty pixels show the background (black).
  * **alpha** — the RGBA A channel: ACCUMULATED coverage 1−Π(1−αᵢ) in [0,1]
    (glass-over-wall ~1, glass-over-void the pane's own alpha).
  * **depth** — planar camera-space Z in metres (gsplat `D`/`ED` "projection
    depth") of the nearest DEPTH-WRITING surface: opaque + alphaMode=MASK write
    depth, BLEND glass does not — so a window pixel carries the depth of the
    room behind the pane, matching the alpha-gated depth loss. Stored as the
    uint16 codes of a LOG mapping over the shared [near, far] (code 0 =
    background; `encode_depth_u16` / `decode_depth_u16`), ~0.01% relative
    error — near-lossless for the depth L1.

Plus `transforms.json` recording, per rendered face, the exact camera-to-world
pose + shared pinhole intrinsics, all in **gsplat's native OpenCV convention**,
so Stage 6 is a thin loop: `viewmats = inv(c2w)`, `rasterization(..., Ks,
render_mode="RGB+ED")`.

Conventions (LOCKED to match gsplat; validated by the capture page's ?selftest):
  * World: Y-up, right-handed, metres (repo-native; unchanged).
  * Camera: OpenCV — +X right, +Y **down**, +Z **forward** (into the scene). We
    store camera-to-world; Stage 6 inverts to gsplat `viewmats`. The capture
    page aims its GL camera with three.js `lookAt(pos + face.forward, face.up)`
    from the SAME plan basis — the two agree by construction, since the GL
    camera frame is exactly `R_cv · diag(1, −1, −1)`.
  * Intrinsics: pinhole `K = [[f,0,cx],[0,f,cy],[0,0,1]]`, f = R/(2·tan(fov/2))
    (= R/2 at the fixed 90° cube FOV), cx = cy = R/2 — a symmetric frustum, so a
    square three.js PerspectiveCamera(fov, 1, near, far) reproduces it exactly.
  * Depth: planar camera-space Z (not ray distance) — gsplat `ED`.
  * Pixel origin: WebGL reads frames bottom-up; `write_reference_frame` flips
    rows so row 0 = top, matching gsplat / normal images.

CUBEMAP-NATIVE input: Stage 4 emits camera POSITIONS, each tagged with the cube
faces worth rendering (`cameras.json` → `intrinsics` + `cube_faces` + `cameras`).
Every (position, face) becomes one 90° pinhole reference image.

No CUDA, torch, or trimesh anywhere in this stage: rendering needs only a
hardware-WebGL browser, and this module needs numpy + zstd (the Python 3.14+
stdlib module, else the `zstandard` package) + PIL for previews/legacy PNG.
"""

from __future__ import annotations

import json
import struct
from math import radians, tan
from pathlib import Path
from typing import Any

import numpy as np

# The cube-face basis is defined ONCE in Stage 4; reuse it so poses derived here
# match the plan exactly (torch-free import — pulls only numpy/scipy via stage4).
from splat.stage4 import CUBE_FACES  # noqa: F401  (re-exported for consumers)

# Written under a cell's `splat/refs/` dir.
REFS_DIRNAME = "refs"
TRANSFORMS_NAME = "transforms.json"
FRAMES_DIRNAME = "frames"
FRAME_SUFFIX = ".szf"

# --- the SZF frame container ---------------------------------------------------
# frames/{view}.szf, little-endian:
#   magic  b"SZF1"  | u16 resolution (square) | u8 filter | u8 reserved
#   u32 rgba_clen   | u32 depth_clen
#   zstd(filtered RGBA8  [H,W,4], top-down)   rgba_clen bytes
#   zstd(filtered u16-LE depth codes [H,W])   depth_clen bytes
# `filter` 1 = sub-left (PNG "Sub": per-row, per-channel horizontal delta in the
# plane's own unsigned dtype — wraps mod 2^8 / 2^16); 0 = raw. Filtered zstd-3
# measured 28% smaller than raw zstd-3 on real frames for ~1 ms extra decode.
FRAME_MAGIC = b"SZF1"
FRAME_FORMAT = "szf1"          # transforms.json container tag
_FRAME_HEADER = struct.Struct("<4sHBBII")
_FILTER_RAW = 0
_FILTER_SUBLEFT = 1
_FRAME_ZSTD_LEVEL = 3          # ~3 ms/512² view; level 1 is barely faster, 4% bigger

# Depth VALUE semantics (independent of the container): planar-Z metres are
# LOG-quantized over the shared [near, far] into uint16 codes 1..65535, code 0 =
# background (no depth-writing surface). Log spacing keeps relative precision
# constant (~0.01% — sub-mm near the camera), near-lossless for the alpha-gated
# depth L1. The capture page's depth-pack shader emits these exact codes on the
# GPU; decode back to metres with `decode_depth_u16` using the [near, far]
# recorded in transforms.json.
DEPTH_ENCODING = "planar_z_log_uint16"
_DEPTH_CODE_MAX = 65535   # uint16 max; code 0 == background

# Reference-render background (empty-pixel RGB) — shared by the capture page's
# clear colour and anything compositing against the refs.
BACKGROUND_RGB = (0.0, 0.0, 0.0)

# --- lighting (Phase 1: baked lighting) ----------------------------------------
# The FIXED light rig every reference view is rendered with, mirrored from the
# debug mesh viewer (client/public/js/scene3d.js) so the trained splat looks like
# the mesh preview. It is deliberately constant and scene-independent: identical
# for every view AND every session (the sun is placed per scene FROM these angles
# at capture time), so a resumed render can never mix two lightings. `azimuth_deg`
# / `elevation_deg` orient the sun (0° = +Z front, 90° = +X right; elevation above
# the horizon). Colour is shaded in LINEAR light and encoded to sRGB through a
# fixed ACES-filmic tone map at `exposure` — no auto-exposure — in splatcapture.js.
LIGHTING: dict[str, Any] = {
    "env": 0.35,            # image-based ambient (RoomEnvironment) intensity
    "key": 3.5,             # sun (directional light) intensity
    "fill": 0.2,            # hemisphere fill intensity
    "azimuth_deg": 34.0,
    "elevation_deg": 48.0,
    "shadows": True,
    "tone_mapping": "aces_filmic",
    "exposure": 1.0,
}

# Bumped whenever the COLOUR PIPELINE itself changes (tone map / transfer / how
# materials are shaded), so a reference set rendered by an older pipeline is
# detected as stale even when `LIGHTING` is byte-identical. v2: materials render
# with their authored metallic-roughness (reflective) instead of forced matte.
# v3: transparent surfaces composite through weighted-blended OIT (order-
# independent), so objects behind glass render correctly instead of going black.
COLOR_PIPELINE = "linear-aces-srgb-v3"

# Sidecar under a cell's refs/ recording the capture settings the on-disk frames
# were rendered with, so a resume can detect a change (see `reconcile_capture_meta`).
CAPTURE_META_NAME = "capture.json"


# --- zstd (stdlib on Python 3.14+, else the `zstandard` package) ---------------

_ZSTD: tuple[Any, Any] | None = None


def _zstd() -> tuple[Any, Any]:
    """(compress(data, level) -> bytes, decompress(data) -> bytes)."""
    global _ZSTD
    if _ZSTD is None:
        try:
            from compression import zstd as _z  # Python 3.14+ stdlib

            _ZSTD = (lambda data, level: _z.compress(data, level=level), _z.decompress)
        except ImportError:
            try:
                import zstandard as _z
            except ImportError as exc:
                raise RuntimeError(
                    "SZF reference frames need zstd — Python 3.14+ (stdlib "
                    "compression.zstd) or `pip install zstandard`"
                ) from exc
            _ZSTD = (
                lambda data, level: _z.ZstdCompressor(level=level).compress(data),
                lambda data: _z.ZstdDecompressor().decompress(data),
            )
    return _ZSTD


# --- camera plan / poses / intrinsics -----------------------------------------


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


# --- resume signal -------------------------------------------------------------


def frame_path(out_dir: Path, vid: str) -> Path:
    """Where a view's SZF frame lives under a refs dir."""
    return out_dir / FRAMES_DIRNAME / f"{vid}{FRAME_SUFFIX}"


def view_rendered(out_dir: Path, vid: str) -> bool:
    """Stage-5 resume signal: True when the view's frame is on disk. Writes are
    atomic (temp + replace), so a present file is a complete one — a resumed
    render re-renders only the views still missing."""
    return frame_path(out_dir, vid).is_file()


def pending_views(out_dir: Path, views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The subset of `views` whose frames are not yet on disk — the work list a
    (possibly resumed) capture session renders."""
    return [v for v in views if not view_rendered(out_dir, v["id"])]


def capture_meta() -> dict[str, Any]:
    """The capture settings that fully determine every stored pixel (lighting,
    background, colour pipeline) — recorded beside the frames so a resume can tell
    whether the frames on disk were rendered under the SAME conditions."""
    return {
        "lighting": LIGHTING,
        "background": list(BACKGROUND_RGB),
        "color_pipeline": COLOR_PIPELINE,
    }


def reconcile_capture_meta(out_dir: Path) -> bool:
    """Keep a refs dir's frames consistent with the CURRENT capture settings.

    Reads the recorded `capture.json`; if it exists and differs from the current
    settings — the lighting or the colour pipeline changed since those frames were
    rendered — deletes the stale frames so the (resumed) render re-does every view
    under one consistent lighting, then rewrites the record. Returns True when a
    reset happened. A fresh dir (no record yet) just writes the record. This is
    what makes "record the lighting so a resume stays consistent" actually hold."""
    out_dir = Path(out_dir)
    meta = capture_meta()
    path = out_dir / CAPTURE_META_NAME
    prev: Any = None
    if path.is_file():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            prev = None
    reset = prev is not None and prev != meta
    if reset:
        import shutil

        shutil.rmtree(out_dir / FRAMES_DIRNAME, ignore_errors=True)
    if prev != meta:
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        tmp.replace(path)
    return reset


# --- depth codec ---------------------------------------------------------------


def encode_depth_u16(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """Planar-Z depth (metres; 0 = background) → uint16 codes: code 0 = background,
    1..65535 = LOG-spaced positions over [near, far]. A foreground sample is
    clamped into [near, far] first, so a surface closer than `near` still encodes
    to code 1 (never 0) and stays in the alpha-gated depth loss instead of being
    read back as background. The capture page's depth-pack shader implements this
    same mapping on the GPU (to ±1 code from float32 rounding); this is the
    reference implementation the smoke tests pin."""
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


def load_depth_png(path: Path, near: float, far: float) -> np.ndarray:
    """LEGACY reader: a 16-bit depth PNG from a pre-SZF reference set → planar-Z
    metres [H,W]. `near`/`far` must match the encode (both in transforms.json).
    New sets carry depth inside the SZF frame (`load_reference_frame`)."""
    from PIL import Image

    codes = np.asarray(Image.open(path))
    return decode_depth_u16(codes, near, far)


# --- SZF frame codec -------------------------------------------------------------


def _filter_subleft(plane: np.ndarray) -> np.ndarray:
    """PNG "Sub" predictor along each row (axis 1): keep column 0, store every
    other sample as the delta to its left same-channel neighbour, wrapping in the
    plane's own unsigned dtype. Turns smooth image/depth rows into near-zero
    residue that zstd compresses far better than raw pixels."""
    d = plane.copy()
    d[:, 1:] = plane[:, 1:] - plane[:, :-1]
    return d


def _unfilter_subleft(plane: np.ndarray) -> np.ndarray:
    """Inverse of `_filter_subleft`: a wrapping prefix-sum along each row."""
    return np.add.accumulate(plane, axis=1, dtype=plane.dtype)


def write_reference_frame(
    out_dir: Path, vid: str, resolution: int, rgba: bytes, depth_codes: bytes
) -> None:
    """Encode + persist ONE captured view from the raw buffers the capture page
    streams: `rgba` = RGBA8 (WebGL readPixels — bottom-up rows, RGB = composited
    albedo, A = accumulated coverage), `depth_codes` = little-endian uint16
    log-depth codes from the depth-pack pass (bottom-up). Writes the view's SZF
    frame atomically, flipped to top-down. Pure CPU (~3 ms at 512², ~12 ms at
    1024²) — this is the function the server's encode process pool runs."""
    r = int(resolution)
    px = np.frombuffer(rgba, dtype=np.uint8)
    if px.size != r * r * 4:
        raise ValueError(f"{vid}: rgba buffer is {px.size} bytes, want {r * r * 4}")
    codes = np.frombuffer(depth_codes, dtype="<u2")
    if codes.size != r * r:
        raise ValueError(f"{vid}: depth buffer has {codes.size} px, want {r * r}")

    px = np.flipud(px.reshape(r, r, 4))            # GL rows are bottom-up → top-down
    codes = np.flipud(codes.reshape(r, r))

    compress, _ = _zstd()
    rgba_c = compress(_filter_subleft(px).tobytes(), _FRAME_ZSTD_LEVEL)
    depth_c = compress(
        np.asarray(_filter_subleft(codes), dtype="<u2").tobytes(), _FRAME_ZSTD_LEVEL
    )

    path = frame_path(out_dir, vid)
    # The writer owns its destination: encode workers are separate processes (and
    # import this module fresh from disk), so never assume the supervising runner
    # — possibly an older in-memory build — created frames/ for us.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(_FRAME_HEADER.pack(FRAME_MAGIC, r, _FILTER_SUBLEFT, 0, len(rgba_c), len(depth_c)))
        f.write(rgba_c)
        f.write(depth_c)
    tmp.replace(path)


def _read_frame(path: Path, *, want_rgba: bool, want_depth: bool):
    """Parse an SZF frame, decompressing only the requested planes.
    Returns (rgba [H,W,4] uint8 | None, depth codes [H,W] uint16 | None)."""
    data = Path(path).read_bytes()
    if len(data) < _FRAME_HEADER.size or data[:4] != FRAME_MAGIC:
        raise ValueError(f"{path}: not an SZF reference frame")
    _, r, filt, _, rgba_clen, depth_clen = _FRAME_HEADER.unpack_from(data)
    if len(data) != _FRAME_HEADER.size + rgba_clen + depth_clen:
        raise ValueError(f"{path}: truncated SZF frame")
    if filt not in (_FILTER_RAW, _FILTER_SUBLEFT):
        raise ValueError(f"{path}: unknown SZF filter {filt}")
    _, decompress = _zstd()
    o = _FRAME_HEADER.size
    rgba = codes = None
    if want_rgba:
        px = np.frombuffer(decompress(data[o : o + rgba_clen]), dtype=np.uint8)
        rgba = px.reshape(r, r, 4)
        if filt == _FILTER_SUBLEFT:
            rgba = _unfilter_subleft(rgba)
    if want_depth:
        cd = np.frombuffer(decompress(data[o + rgba_clen :]), dtype="<u2")
        codes = cd.reshape(r, r)
        if filt == _FILTER_SUBLEFT:
            codes = _unfilter_subleft(codes)
    return rgba, codes


def load_reference_frame(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one SZF frame → (rgba [H,W,4] uint8 top-down, depth codes [H,W]
    uint16). RGB/alpha are the raw stored bytes; decode depth to metres with
    `decode_depth_u16` + the plan's [near, far]. This is Stage 6's reader —
    one file open per training sample, ~4x faster than the old PNG triple."""
    rgba, codes = _read_frame(path, want_rgba=True, want_depth=True)
    return rgba, codes


def frame_preview_png(path: Path) -> bytes:
    """One frame's RGB as PNG bytes — the on-demand transcode behind the debug
    UI's patch inspector (the only consumer that ever needs PNG). Decompresses
    just the RGBA stream; encodes fast (level 1) since previews are ephemeral."""
    import io

    from PIL import Image

    rgba, _ = _read_frame(path, want_rgba=True, want_depth=False)
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgba[..., :3])).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


# --- transforms.json -------------------------------------------------------------


def reference_frames(views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The `transforms.json` frame entries for a plan's views — deterministic from
    the plan alone, so a resumed/multi-session render reproduces the identical
    file once every view is on disk. `frame_path` points at the SZF container
    holding all three planes (rgb + alpha + depth)."""
    return [
        {
            "frame_path": f"{FRAMES_DIRNAME}/{v['id']}{FRAME_SUFFIX}",
            "camera_index": v["camera_index"],
            "face": v["face"],
            "transform_matrix": v["c2w"].tolist(),
        }
        for v in views
    ]


def write_transforms(
    out_dir: Path,
    K: np.ndarray,
    resolution: int,
    near: float,
    far: float,
    frames: list[dict[str, Any]],
    depth_encoding: str = DEPTH_ENCODING,
    lighting: dict[str, Any] | None = None,
) -> Path:
    """Write `transforms.json`: shared pinhole intrinsics + per-frame OpenCV
    camera-to-world and SZF frame paths. `convention` is tagged so Stage 6 knows
    to take `viewmats = inv(transform_matrix)` (gsplat OpenCV), NOT the Nerfstudio
    OpenGL c2w; `frame_format` names the container; `depth` names the value
    encoding (log-uint16 codes, decoded with `near`/`far`). `lighting` (when the
    frames were rendered lit — Phase 1) records the fixed rig + tone map the
    colour was baked with, so the provenance of the supervision is explicit and
    later relighting phases can read it back; None tags the set as unlit albedo."""
    doc = {
        "camera_model": "pinhole",
        "convention": "opencv_c2w",     # transform_matrix is OpenCV camera-to-world
        "frame_format": FRAME_FORMAT,   # SZF container (zstd RGBA + depth codes)
        "depth": depth_encoding,        # gsplat render_mode D/ED value semantics
        "color_space": "srgb",
        "lighting": lighting,           # fixed bake rig (Phase 1) or None (unlit albedo)
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
