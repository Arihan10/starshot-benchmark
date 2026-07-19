"""In-process GLB loading for the splat pipeline — no de-optimization step.

The splat stages consume the cell's serving asset sets AS-IS: either vanilla
glTF (the raw generated build) or the KTX2/Meshopt-encoded sets (`objects/`,
`objects-optimized/`, `objects-generated-lite/`, `objects-generated-optimized/`
— all carry `EXT_meshopt_compression` + `KHR_mesh_quantization` +
`KHR_texture_basisu` in extensionsRequired, which trimesh cannot read). This
module replaces the old build-a-splat-tier + Node-de-optimize round trip with a
direct reader:

  * vanilla files load through trimesh unchanged (full materials + textures);
  * compressed files are decoded natively: the GLB JSON/BIN chunks are parsed
    here, `EXT_meshopt_compression` buffer views are decompressed with the
    `meshoptimizer` package (ATTRIBUTES / TRIANGLES / INDICES modes + the
    OCTAHEDRAL / EXPONENTIAL / QUATERNION filters), `KHR_mesh_quantization`
    normalized accessors are dequantized per the glTF spec, and node TRS
    hierarchies are baked into world space — the same world-baked geometry the
    old de-optimized twin produced.

TEXTURES are decoded too: KTX2/BasisU base-color images (ETC1S and UASTC —
the optimized/library and lite presets respectively) are transcoded to RGBA8
in-process by the VENDORED Basis Universal transcoder
(`splat/vendor/basisu_transcoder_module_st.wasm`, the same codebase the WebGL
viewer's KTX2Loader runs) executed through `wasmtime` — pure-Python wheels on
Windows/macOS/Linux, so one code path everywhere. The decode is a raw texel
passthrough (no sRGB↔linear conversion), matching the vanilla-PNG path and the
Stage-5 unlit renderer, and it feeds the resulting PIL image straight into
`baseColorTexture`, so Stage 3's footprint-averaged albedo and Stage 2's
per-texel glass alpha work identically on compressed and vanilla sets. When
the transcoder is unavailable (missing wasmtime / wasm file) or a decode
fails, materials degrade to the previous stubs: `alphaMode` / `alphaCutoff` /
`baseColorFactor` (+ the KTX2 pixel size on `ktx2_texture_size`), with
`baseColorTexture = None`.

Every mesh consumer of stages 0-3 goes through `load_geoms(path)`:
world-space `trimesh.Trimesh` parts, one per glTF primitive.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import trimesh

# glTF componentType → numpy dtype.
_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
# Normalized-integer dequantization divisors (glTF 2.0 §3.6.2.2).
_NORM_DIVISORS = {
    np.int8: 127.0,
    np.uint8: 255.0,
    np.int16: 32767.0,
    np.uint16: 65535.0,
}

_MESHOPT_EXT = "EXT_meshopt_compression"


def _parse_glb(raw: bytes) -> tuple[dict, bytes]:
    """GLB container → (gltf json, BIN chunk bytes)."""
    if raw[:4] != b"glTF":
        raise ValueError("not a GLB container")
    doc = None
    binary = b""
    offset = 12
    while offset + 8 <= len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        chunk = raw[offset + 8 : offset + 8 + length]
        if kind == 0x4E4F534A:  # 'JSON'
            doc = json.loads(chunk)
        elif kind == 0x004E4942:  # 'BIN\0'
            binary = chunk
        offset += 8 + length + (-length % 4)
    if doc is None:
        raise ValueError("GLB has no JSON chunk")
    return doc, binary


def _decode_view(doc: dict, view: dict, binary: bytes) -> bytes:
    """Raw bytes of one bufferView, decompressing `EXT_meshopt_compression`
    when present. Fallback buffers (spec: a placeholder with no data) are only
    ever reached through the compressed path."""
    ext = (view.get("extensions") or {}).get(_MESHOPT_EXT)
    if ext is None:
        start = view.get("byteOffset", 0)
        return binary[start : start + view["byteLength"]]

    import meshoptimizer as mo

    comp = binary[ext.get("byteOffset", 0) : ext.get("byteOffset", 0) + ext["byteLength"]]
    count, stride = int(ext["count"]), int(ext["byteStride"])
    mode = ext.get("mode", "ATTRIBUTES")
    # The python bindings return a uint32-typed array whose BYTES are the raw
    # decoded buffer — NOT widened index values. So reinterpret the bytes at the
    # accessor's real index width and take the first `count`; value-casting
    # (astype) instead would truncate each uint16 pair read as one uint32 (e.g.
    # bytes for [0,1] -> 65536 -> 0), scrambling ~70% of faces into degenerate
    # triangles and tearing holes in every meshopt/u16 mesh. The vertex decoder
    # returns a float32-shaped view of the raw bytes; normalize both back to the
    # exact uncompressed buffer layout the accessors describe.
    if mode in ("TRIANGLES", "INDICES"):
        decode = mo.decode_index_buffer if mode == "TRIANGLES" else mo.decode_index_sequence
        raw = np.ascontiguousarray(np.asarray(decode(count, stride, comp)))
        tgt = np.uint16 if stride == 2 else np.uint32
        return raw.view(np.uint8).view(tgt).reshape(-1)[:count].astype(tgt).tobytes()

    out = np.ascontiguousarray(np.asarray(mo.decode_vertex_buffer(count, stride, comp)))
    data = out.view(np.uint8).reshape(count, stride)
    filt = ext.get("filter")
    if filt and filt != "NONE":
        buf = np.ascontiguousarray(data)
        if filt == "OCTAHEDRAL":
            buf = mo.decode_filter_oct(buf, count, stride)
        elif filt == "EXPONENTIAL":
            buf = mo.decode_filter_exp(buf, count, stride)
        elif filt == "QUATERNION":
            buf = mo.decode_filter_quat(buf, count, stride)
        data = (
            np.ascontiguousarray(np.asarray(buf)).view(np.uint8).reshape(count, stride)
        )
    return data.tobytes()


def _read_accessor(doc: dict, idx: int, binary: bytes) -> np.ndarray:
    """Accessor → (count, width) float64/int array, dequantizing normalized
    integers. Interleaved (strided) layouts are handled; sparse accessors are
    not (never emitted by our builders)."""
    acc = doc["accessors"][idx]
    if "sparse" in acc:
        raise ValueError("sparse accessors unsupported")
    dtype = np.dtype(_COMPONENT_DTYPES[acc["componentType"]])
    width = _TYPE_WIDTHS[acc["type"]]
    count = int(acc["count"])
    view = doc["bufferViews"][acc["bufferView"]]
    raw = _decode_view(doc, view, binary)
    start = acc.get("byteOffset", 0)
    stride = view.get("byteStride") or dtype.itemsize * width
    if stride == dtype.itemsize * width:
        arr = np.frombuffer(
            raw, dtype=dtype, count=count * width, offset=start
        ).reshape(count, width)
    else:  # interleaved — gather row by row via strided view
        rows = np.frombuffer(raw, dtype=np.uint8)
        arr = np.lib.stride_tricks.as_strided(
            rows[start:], shape=(count, dtype.itemsize * width), strides=(stride, 1)
        )
        arr = np.frombuffer(np.ascontiguousarray(arr).tobytes(), dtype=dtype).reshape(
            count, width
        )
    if acc.get("normalized"):
        div = _NORM_DIVISORS[dtype.type]
        out = arr.astype(np.float64) / div
        if dtype.kind == "i":
            out = np.maximum(out, -1.0)  # signed normalized floor (spec)
        return out
    return arr.astype(np.float64) if dtype != np.uint32 else arr


def _node_transforms(doc: dict) -> list[tuple[int, np.ndarray]]:
    """(node index, world 4×4) for every node reachable from the scene roots."""
    nodes = doc.get("nodes", [])
    scene = doc.get("scenes", [{}])[doc.get("scene", 0)]
    out: list[tuple[int, np.ndarray]] = []

    def local(n: dict) -> np.ndarray:
        if "matrix" in n:
            return np.asarray(n["matrix"], dtype=np.float64).reshape(4, 4).T
        m = np.eye(4)
        if "rotation" in n:
            x, y, z, w = n["rotation"]
            m[:3, :3] = trimesh.transformations.quaternion_matrix([w, x, y, z])[:3, :3]
        if "scale" in n:
            m[:3, :3] = m[:3, :3] * np.asarray(n["scale"], dtype=np.float64)
        if "translation" in n:
            m[:3, 3] = n["translation"]
        return m

    def walk(idx: int, parent: np.ndarray) -> None:
        n = nodes[idx]
        world = parent @ local(n)
        out.append((idx, world))
        for c in n.get("children", []):
            walk(c, world)

    for root in scene.get("nodes", []):
        walk(root, np.eye(4))
    return out


def _ktx2_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a KTX2 header (pixelWidth/Height at offsets 20/24)."""
    if len(data) < 28 or data[:12] != b"\xabKTX 20\xbb\r\n\x1a\n":
        return None
    w, h = struct.unpack_from("<II", data, 20)
    return (int(w), int(h)) if w and h else None


# --- KTX2 → RGBA8 (BasisU transcode via the vendored WASM module) --------------
# basist::transcoder_texture_format::cTFRGBA32 — uncompressed RGBA8, a raw texel
# passthrough (no color-space conversion), matching the vanilla-PNG path.
_TF_RGBA32 = 13
_WASM_PATH = Path(__file__).parent / "vendor" / "basisu_transcoder_module_st.wasm"


class _BasisTranscoder:
    """Thin wrapper over the official Basis Universal single-threaded WASM
    transcoder (`bt_*` C API), run in-process through wasmtime. One instance
    per process (module-level singleton via `_transcoder()`); WASM is
    single-threaded by construction, so per-process is exactly the safe grain
    for stage 2's process-pool workers."""

    def __init__(self) -> None:
        import wasmtime

        self._engine = wasmtime.Engine()
        self._store = wasmtime.Store(self._engine)
        wasi = wasmtime.WasiConfig()
        wasi.argv = ["basisu-transcoder"]
        wasi.inherit_stdout()
        wasi.inherit_stderr()
        self._store.set_wasi(wasi)
        module = wasmtime.Module.from_file(self._engine, str(_WASM_PATH))
        linker = wasmtime.Linker(self._engine)
        linker.define_wasi()
        instance = linker.instantiate(self._store, module)
        self._x = instance.exports(self._store)
        self._mem = self._x["memory"]
        if "bt_init" in self._x:
            self._x["bt_init"](self._store)

    def _call(self, name: str, *args: int):  # noqa: ANN202
        return self._x[name](self._store, *args)

    def decode_rgba8(self, ktx2: bytes) -> np.ndarray:
        """Transcode mip level 0 of a KTX2 payload (ETC1S or UASTC, zstd
        supercompression handled inside) to an (H, W, 4) uint8 RGBA array."""
        in_ptr = self._call("bt_alloc", len(ktx2))
        handle = 0
        out_ptr = 0
        try:
            self._mem.write(self._store, ktx2, in_ptr)
            handle = self._call("bt_ktx2_open", in_ptr, len(ktx2))
            if not handle:
                raise ValueError("bt_ktx2_open failed (not a BasisU KTX2?)")
            w = self._call("bt_ktx2_get_level_orig_width", handle, 0, 0, 0)
            h = self._call("bt_ktx2_get_level_orig_height", handle, 0, 0, 0)
            if not (0 < w <= 16384 and 0 < h <= 16384):
                raise ValueError(f"bad KTX2 dimensions {w}x{h}")
            if not self._call("bt_ktx2_start_transcoding", handle):
                raise ValueError("bt_ktx2_start_transcoding failed")
            out_size = w * h * 4
            out_ptr = self._call("bt_alloc", out_size)
            ok = self._call(
                "bt_ktx2_transcode_image_level",
                handle, 0, 0, 0,                 # level / layer / face
                out_ptr, w * h, _TF_RGBA32,
                0, 0, 0,                          # decode flags / row pitch / rows
                -1, -1, 0,                        # channel0 / channel1 / state
            )
            if not ok:
                raise ValueError("bt_ktx2_transcode_image_level failed")
            raw = self._mem.read(self._store, out_ptr, out_ptr + out_size)
            return np.frombuffer(bytes(raw), dtype=np.uint8).reshape(h, w, 4)
        finally:
            if out_ptr:
                self._call("bt_free", out_ptr)
            if handle:
                self._call("bt_ktx2_close", handle)
            self._call("bt_free", in_ptr)


_TRANSCODER: _BasisTranscoder | None = None
_TRANSCODER_FAILED = False


def _transcoder() -> _BasisTranscoder | None:
    """The process-wide transcoder, created on first use; None (and never
    retried) when wasmtime or the vendored wasm module is unavailable."""
    global _TRANSCODER, _TRANSCODER_FAILED
    if _TRANSCODER is None and not _TRANSCODER_FAILED:
        try:
            _TRANSCODER = _BasisTranscoder()
        except Exception:
            _TRANSCODER_FAILED = True
    return _TRANSCODER


def _decode_ktx2_image(data: bytes):  # noqa: ANN202 - PIL.Image | None
    """KTX2 payload → PIL RGBA image via the BasisU transcode, or None when the
    transcoder is unavailable or the payload doesn't decode."""
    t = _transcoder()
    if t is None:
        return None
    try:
        from PIL import Image

        return Image.fromarray(t.decode_rgba8(data), mode="RGBA")
    except Exception:
        return None


def _material_stub(
    doc: dict,
    mat_idx: int | None,
    binary: bytes,
    image_cache: dict[int, object] | None = None,
):  # noqa: ANN202
    """A trimesh PBRMaterial for one glTF material: the JSON-level facts
    (alphaMode/cutoff/baseColorFactor) plus the base-color TEXTURE, transcoded
    from its KTX2/BasisU payload to a PIL RGBA image (`_decode_ktx2_image`) so
    downstream texture consumers behave exactly as on vanilla sets. The KTX2
    pixel size is also attached as `ktx2_texture_size` (Stage 1's cheap
    texture report). `image_cache` dedupes decodes per glTF image index within
    one GLB parse (primitives routinely share one texture). Without a working
    transcoder the material degrades to the old stub (`baseColorTexture`
    stays None)."""
    material = trimesh.visual.material.PBRMaterial()
    size = None
    if mat_idx is not None:
        m = doc.get("materials", [])[mat_idx]
        pbr = m.get("pbrMetallicRoughness", {})
        if m.get("alphaMode"):
            material.alphaMode = m["alphaMode"]
        if m.get("alphaCutoff") is not None:
            material.alphaCutoff = float(m["alphaCutoff"])
        if pbr.get("baseColorFactor") is not None:
            material.baseColorFactor = pbr["baseColorFactor"]
        tex = pbr.get("baseColorTexture")
        if tex is not None:
            t = doc.get("textures", [])[tex["index"]]
            src = t.get("source")
            if src is None:
                src = (t.get("extensions", {}).get("KHR_texture_basisu") or {}).get(
                    "source"
                )
            if src is not None:
                img = doc.get("images", [])[src]
                if img.get("bufferView") is not None:
                    view = doc["bufferViews"][img["bufferView"]]
                    start = view.get("byteOffset", 0)
                    payload = binary[start : start + view["byteLength"]]
                    size = _ktx2_size(payload)
                    if image_cache is not None and src in image_cache:
                        material.baseColorTexture = image_cache[src]
                    else:
                        decoded = _decode_ktx2_image(payload)
                        if decoded is not None:
                            material.baseColorTexture = decoded
                        if image_cache is not None:
                            image_cache[src] = decoded
    material.ktx2_texture_size = size
    return material


def _load_compressed(path: Path) -> list[trimesh.Trimesh]:
    """Native decode of a meshopt/quantized/KTX2 GLB → world-baked Trimesh parts
    (positions + faces + UVs + materials with BasisU-transcoded base-color
    textures; normals/tangents skipped — no stage consumes authored normals)."""
    doc, binary = _parse_glb(Path(path).read_bytes())
    meshes = doc.get("meshes", [])
    parts: list[trimesh.Trimesh] = []
    image_cache: dict[int, object] = {}  # glTF image index → decoded PIL | None
    for node_idx, world in _node_transforms(doc):
        mesh_idx = doc["nodes"][node_idx].get("mesh")
        if mesh_idx is None:
            continue
        for prim in meshes[mesh_idx].get("primitives", []):
            if prim.get("mode", 4) != 4:  # triangles only
                continue
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs:
                continue
            pos = _read_accessor(doc, attrs["POSITION"], binary)[:, :3]
            pos = pos @ world[:3, :3].T + world[:3, 3]
            if "indices" in prim:
                idx = _read_accessor(doc, prim["indices"], binary).reshape(-1)
                faces = idx.astype(np.int64).reshape(-1, 3)
            else:
                faces = np.arange(len(pos), dtype=np.int64).reshape(-1, 3)
            visual = None
            material = _material_stub(doc, prim.get("material"), binary, image_cache)
            if "TEXCOORD_0" in attrs:
                uv = _read_accessor(doc, attrs["TEXCOORD_0"], binary)[:, :2].copy()
                # glTF puts v=0 at the image TOP; trimesh's convention (which
                # `uv_to_color` compensates for with its own flip) puts v=0 at
                # the BOTTOM. Mirror V exactly like trimesh's GLB loader does,
                # or every texture lookup downstream (stage-3 albedo, stage-2
                # glass alpha) reads the vertically mirrored texel.
                uv[:, 1] = 1.0 - uv[:, 1]
                visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
            else:
                visual = trimesh.visual.TextureVisuals(material=material)
            parts.append(
                trimesh.Trimesh(
                    vertices=pos, faces=faces, visual=visual, process=False
                )
            )
    return parts


def _needs_native(path: Path) -> bool:
    """True when the GLB declares extensions trimesh cannot read (meshopt)."""
    try:
        with Path(path).open("rb") as f:
            head = f.read(20)
            if head[:4] != b"glTF":
                return False
            length = struct.unpack_from("<I", head, 12)[0]
            doc = json.loads(f.read(length))
        return _MESHOPT_EXT in (doc.get("extensionsRequired") or []) or _MESHOPT_EXT in (
            doc.get("extensionsUsed") or []
        )
    except Exception:
        return False


def load_geoms(path: Path) -> list[trimesh.Trimesh]:
    """World-baked triangle geometries of one placed GLB, whatever its encoding:
    vanilla files through trimesh, meshopt/KTX2 files through the native decoder
    — full base-color textures either way (KTX2 texels BasisU-transcoded in
    process). The single mesh entry point for stages 0-3."""
    path = Path(path)
    if _needs_native(path):
        return _load_compressed(path)
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        # dump() bakes node transforms into world space; plain scene.geometry
        # would return LOCAL vertices and collapse node-placed objects.
        return [g for g in loaded.dump(concatenate=False) if hasattr(g, "faces")]
    return [loaded]
