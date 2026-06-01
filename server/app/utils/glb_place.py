"""Placement bake for optimized library assets.

Writes a per-placement world transform into a COPY of a pre-optimized
(Meshopt + KTX2) library GLB *without* decoding geometry or textures, so the
compression survives all the way to the client. The transform reproduces the
`rescale_mesh_to_bbox` contract — yaw, then per-axis stretch to fill the
world-axis-aligned bbox — expressed as a 2-node nest the client applies
natively when it loads the GLB:

    outer node: translation + non-uniform scale (world axes)
      inner node: yaw rotation
        └─ the GLB's original scene roots

Scale and rotation are split across two nodes on purpose: a single matrix of
`scale · rotate` carries shear for a non-uniform scale, which GLTFLoader would
mangle when it decomposes the node into TRS. Keeping each node shear-free
(outer = translate + scale, inner = pure rotation) decomposes cleanly.

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
) -> None:
    """Bake `bbox` + `orientation` placement into `src` → `dst`.

    `rotated_min`/`rotated_max` are the asset's world-space AABB *after* the
    yaw rotation (precomputed in optimize_manifest.json), which is what the
    per-axis fill scale needs.
    """
    target_extents = bbox.size
    target_center = bbox.center
    scale: list[float] = []
    translate: list[float] = []
    for i in range(3):
        extent = rotated_max[i] - rotated_min[i]
        center = (rotated_max[i] + rotated_min[i]) / 2.0
        # Flat assets (walls/floors) have a zero extent on one axis; leave it
        # at scale 1 instead of dividing by zero.
        s = target_extents[i] / extent if abs(extent) > 1e-9 else 1.0
        scale.append(s)
        translate.append(target_center[i] - s * center)

    gltf, bin_data = _parse_glb(src.read_bytes())
    nodes: list[dict] = gltf.setdefault("nodes", [])
    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
    old_roots = list(scene.get("nodes", []))

    inner: dict = {"rotation": _quat_y(orientation)}
    if old_roots:
        inner["children"] = old_roots
    nodes.append(inner)
    inner_idx = len(nodes) - 1

    nodes.append({"translation": translate, "scale": scale, "children": [inner_idx]})
    outer_idx = len(nodes) - 1
    scene["nodes"] = [outer_idx]

    _write_glb(gltf, bin_data, dst)
