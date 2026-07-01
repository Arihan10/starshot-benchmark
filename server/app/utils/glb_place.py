"""Placement bake for optimized library assets.

Writes a per-placement world transform into a COPY of a pre-optimized
(Meshopt + KTX2) library GLB *without* decoding geometry or textures, so the
compression survives all the way to the client. The transform reproduces the
`rescale_mesh_to_bbox` contract — yaw, then a scale that fits the
world-axis-aligned bbox — expressed as a 2-node nest the client applies
natively when it loads the GLB:

    outer node: translation + scale (world axes)
      inner node: yaw rotation
        └─ the GLB's original scene roots

The scale is per-axis (fills the bbox exactly) for an axis-aligned yaw. For an
oblique yaw (±45/±135) the X and Z axes share one scale (the tighter fill) so
the asset isn't sheared, while Y still fills on its own — the asset keeps full
height, and since an oblique yaw makes the footprint square, a square bbox still
fills exactly (see `rescale_mesh_to_bbox` for the full rationale).

Scale and rotation are split across two nodes so each node stays a clean TRS:
a single matrix of `scale · rotate` would carry shear for a non-uniform scale,
which GLTFLoader mangles when it decomposes the node into TRS (outer =
translate + scale, inner = pure rotation).

The edit is done as raw GLB chunk surgery: only the JSON chunk is rewritten
(two nodes appended, the scene re-rooted), and the binary chunk — holding the
Meshopt-compressed geometry and KTX2 images — is copied through byte-for-byte.
A full glTF library (pygltflib) can't be used here: re-serializing the buffer
shifts the byte offsets the EXT_meshopt_compression bufferViews point at and
corrupts the compressed geometry.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

from app.core.types import BoundingBox

_GLB_MAGIC = 0x46546C67  # "glTF"
_CHUNK_JSON = 0x4E4F534A  # "JSON"
_CHUNK_BIN = 0x004E4942  # "BIN\0"


def _quat_y(deg: float) -> list[float]:
    """glTF quaternion [x, y, z, w] for a right-handed yaw about +Y. Matches the
    rotation the bounds were measured under (augment-bounds.mjs) and the one
    three.js reconstructs from the node, so fill + facing stay consistent."""
    half = math.radians(deg) / 2.0
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def quat_x(deg: float) -> list[float]:
    """glTF quaternion [x, y, z, w] for a rotation about +X. `quat_x(-90)` maps
    the asset's +Z to +Y — the fixed reorientation v5 bakes so upright assets
    stand along the side-view's vertical (world z) instead of lying flat."""
    half = math.radians(deg) / 2.0
    return [math.sin(half), 0.0, 0.0, math.cos(half)]


def _rotate_vec(q: list[float], v: tuple[float, float, float]) -> list[float]:
    """Rotate a 3-vector by glTF quaternion q = [x, y, z, w]."""
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def rotate_aabb(
    rmin: list[float], rmax: list[float], quat: list[float]
) -> tuple[list[float], list[float]]:
    """The axis-aligned bounds of the box [rmin, rmax] after rotation by `quat`.
    Exact for the 90° rotations v5 uses (they permute axes), so the fill scale
    in place_glb stays correct when an extra `model_rotation` is baked in."""
    corners = [
        (rmin[0] if i & 1 else rmax[0],
         rmin[1] if i & 2 else rmax[1],
         rmin[2] if i & 4 else rmax[2])
        for i in range(8)
    ]
    rotated = [_rotate_vec(quat, c) for c in corners]
    return (
        [min(p[a] for p in rotated) for a in range(3)],
        [max(p[a] for p in rotated) for a in range(3)],
    )


def _parse_glb(data: bytes) -> tuple[dict, bytes | None]:
    """Split a GLB into (json_dict, bin_chunk_data). bin is None if absent."""
    magic, _version, length = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        raise ValueError("not a binary glTF (.glb)")
    json_obj: dict | None = None
    bin_data: bytes | None = None
    offset = 12
    while offset < length:
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8 : offset + 8 + chunk_len]
        if chunk_type == _CHUNK_JSON:
            json_obj = json.loads(chunk)
        elif chunk_type == _CHUNK_BIN:
            bin_data = chunk
        offset += 8 + chunk_len
    if json_obj is None:
        raise ValueError("GLB missing JSON chunk")
    return json_obj, bin_data


def _write_glb(json_obj: dict, bin_data: bytes | None, dst: Path) -> None:
    json_bytes = json.dumps(json_obj, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)  # pad with spaces
    chunks: list[tuple[int, bytes]] = [(_CHUNK_JSON, json_bytes)]
    if bin_data is not None:
        bin_bytes = bin_data + b"\x00" * ((4 - len(bin_data) % 4) % 4)
        chunks.append((_CHUNK_BIN, bin_bytes))
    total = 12 + sum(8 + len(d) for _, d in chunks)
    out = bytearray(struct.pack("<III", _GLB_MAGIC, 2, total))
    for chunk_type, d in chunks:
        out += struct.pack("<II", len(d), chunk_type)
        out += d
    dst.write_bytes(bytes(out))


def place_glb(
    *,
    src: Path,
    dst: Path,
    bbox: BoundingBox,
    orientation: int,
    rotated_min: list[float],
    rotated_max: list[float],
    model_rotation: list[float] | None = None,
) -> None:
    """Bake `bbox` + `orientation` placement into `src` → `dst`.

    `rotated_min`/`rotated_max` are the asset's world-space AABB *after* every
    baked rotation (the yaw from optimize_manifest.json and, when given,
    `model_rotation`), which is what the fill scale needs — pass bounds already
    run through `rotate_aabb(..., model_rotation)`.

    `model_rotation` is an optional fixed glTF quaternion applied INNERMOST
    (in asset-local space, before yaw): v5 bakes `quat_x(-90)` so library
    assets stand upright along the side-view's vertical axis.
    """
    target_extents = bbox.size
    target_center = bbox.center
    extents = [rotated_max[i] - rotated_min[i] for i in range(3)]
    centers = [(rotated_max[i] + rotated_min[i]) / 2.0 for i in range(3)]
    # Per-axis fill. Flat assets (walls/floors) have a zero extent on one axis;
    # leave it at scale 1 instead of dividing by zero.
    scale: list[float] = [
        target_extents[i] / extents[i] if abs(extents[i]) > 1e-9 else 1.0
        for i in range(3)
    ]
    if orientation % 90 != 0:
        # Oblique yaw (about +Y) rotates X and Z off the world axes, so scaling
        # them by DIFFERENT amounts shears the asset; they share the tighter fill
        # instead (inscribing the footprint — a ±45/±135 yaw makes it square, so a
        # square bbox still fills exactly). Y is the yaw axis, so it keeps its own
        # fill and the asset keeps full height; an in-plane-uniform X/Z scale
        # commutes with the yaw, so the scale + rotation nodes stay shear-free.
        # (model_rotation, when set, is an axis-aligned 90°, so the yaw alone
        # decides obliqueness.)
        h = min(scale[0], scale[2])
        scale = [h, scale[1], h]
    translate: list[float] = [
        target_center[i] - scale[i] * centers[i] for i in range(3)
    ]

    gltf, bin_data = _parse_glb(src.read_bytes())
    nodes: list[dict] = gltf.setdefault("nodes", [])
    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
    old_roots = list(scene.get("nodes", []))

    # Optional fixed reorientation, applied innermost (asset-local, before yaw).
    yaw_children = old_roots
    if model_rotation is not None:
        model_node: dict = {"rotation": model_rotation}
        if old_roots:
            model_node["children"] = old_roots
        nodes.append(model_node)
        yaw_children = [len(nodes) - 1]

    inner: dict = {"rotation": _quat_y(orientation)}
    if yaw_children:
        inner["children"] = yaw_children
    nodes.append(inner)
    inner_idx = len(nodes) - 1

    nodes.append({"translation": translate, "scale": scale, "children": [inner_idx]})
    outer_idx = len(nodes) - 1
    scene["nodes"] = [outer_idx]

    _write_glb(gltf, bin_data, dst)
