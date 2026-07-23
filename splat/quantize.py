"""Near-lossless quantization of a trained 2DGS splat (the Stage-7 compress core).

Isolated, self-contained module: reads a Stage-3 / Stage-6 2DGS `.ply` (16 float32
per Gaussian — see splat/stage6.py `_encode_trained_ply`) and re-encodes every
attribute at just-noticeable precision into a compact, self-describing container
(`.sqz`). This is the single biggest space saver in the compression stack: the
float32 fields carry far more precision than the eye or the renderer needs, so
dropping to 8/16-bit per attribute is visually near-lossless yet ~3.5x smaller —
before any entropy coding (a lossless zstd/brotli pass composes on top later).

Each attribute is quantized in its PERCEPTUAL space, over its OBSERVED range:
  * position xyz  -> 16-bit / axis over the per-axis AABB (~0.1 mm across a room);
  * colour f_dc   -> 8-bit / channel (f_dc is linear in [0,1] display colour);
  * opacity       -> 8-bit in SIGMOID space (alpha), not the stored logit;
  * scale_0/1     -> 16-bit over the stored LOG range (multiplicative precision);
  * rotation quat -> "smallest-three": drop the largest component (recovered as
    sqrt(1 - Σ others²), sign canonicalized), store the other three at `quat_bits`
    + a 1-byte index. Normals (nx,ny,nz) are NOT stored — they are recomputed from
    the quaternion on decode (a free lossless drop).

Anything unrecognized (e.g. higher-order SH `f_rest_*`) is passed through as raw
float32 so nothing is silently lost.

The module round-trips: `dequantize` / `sqz_to_ply` rebuild a standard float32
2DGS `.ply` (viewable in the existing mkkellogg viewer) so the near-lossless
result can be compared directly, and `quantize_ply` reports per-attribute
reconstruction error so the bit depths are auditable and tunable.

Pure library + CLI, like the pipeline stages; numpy only (no torch/CUDA).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"SQZ1"
CONTAINER_VERSION = 1
SQZ_NAME = "trained.sqz"

# SH degree-0 basis constant (matches Stage 3/4): colour = 0.5 + C0 * f_dc.
_SH_C0 = 0.28209479177387814

_POS = ("x", "y", "z")
_NORMAL = ("nx", "ny", "nz")            # never stored — recomputed from the quat
_COLOR = ("f_dc_0", "f_dc_1", "f_dc_2")
_OPACITY = "opacity"
_QUAT = ("rot_0", "rot_1", "rot_2", "rot_3")

# Max magnitude of any component of a unit quaternion's "smallest three".
_QUAT_LIMIT = float(1.0 / np.sqrt(2.0))


@dataclass(frozen=True)
class QuantConfig:
    """Per-attribute bit depths (byte-aligned: 8 -> uint8, 9..16 -> uint16). The
    defaults are near-lossless for room-scale albedo splats. `opacity_clamp` keeps
    alpha off the 0/1 asymptotes so the logit round-trip stays finite."""

    pos_bits: int = 16
    color_bits: int = 8
    opacity_bits: int = 8
    scale_bits: int = 16
    quat_bits: int = 8
    opacity_clamp: float = 1e-4

    def as_summary(self) -> dict[str, Any]:
        return asdict(self)


# --- PLY IO (self-contained) --------------------------------------------------


def _read_ply(path: Path) -> tuple[int, list[str], dict[str, np.ndarray]]:
    """Parse a binary-little-endian all-float `.ply` into (count, property order,
    {name: (N,) float32}). Matches the Stage-3/6 splat layout."""
    raw = Path(path).read_bytes()
    marker = b"end_header\n"
    cut = raw.find(marker)
    if cut < 0:
        raise ValueError(f"{path}: not a PLY (no end_header)")
    header = raw[:cut].decode("ascii", "replace").splitlines()
    if not header or header[0].strip() != "ply":
        raise ValueError(f"{path}: missing 'ply' magic")
    if not any(l.startswith("format") and "binary_little_endian" in l for l in header):
        raise ValueError(f"{path}: only binary_little_endian is supported")

    count: int | None = None
    props: list[str] = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            count, in_vertex = int(parts[2]), True
        elif parts[:1] == ["element"]:
            in_vertex = False
        elif in_vertex and parts[:1] == ["property"]:
            if parts[1] != "float":
                raise ValueError(f"{path}: non-float property '{parts[-1]}' unsupported")
            props.append(parts[-1])
    if count is None:
        raise ValueError(f"{path}: no vertex element")

    body = np.frombuffer(raw, dtype="<f4", count=count * len(props), offset=cut + len(marker))
    if body.size < count * len(props):
        raise ValueError(f"{path}: truncated body ({body.size} < {count * len(props)})")
    table = body.reshape(count, len(props))
    cols = {name: table[:, i].astype(np.float32) for i, name in enumerate(props)}
    return count, props, cols


def _write_ply(path: Path, order: list[str], cols: dict[str, np.ndarray]) -> None:
    """Write named float columns back out as a binary-little-endian `.ply`."""
    n = len(cols[order[0]])
    data = np.stack([cols[name] for name in order], axis=1).astype("<f4")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {name}\n" for name in order)
        + "end_header\n"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    tmp.replace(path)


# --- quantization primitives --------------------------------------------------


def _uint_dtype(bits: int) -> np.dtype:
    if bits <= 8:
        return np.dtype("<u1")
    if bits <= 16:
        return np.dtype("<u2")
    return np.dtype("<u4")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(a: np.ndarray) -> np.ndarray:
    return np.log(a / (1.0 - a))


def _quantize(values: np.ndarray, bits: int) -> tuple[np.ndarray, float, float]:
    """Uniform quantize a 1-D array to `bits` over its observed [min,max]. Returns
    (codes, lo, hi). A degenerate (flat) column encodes to all-zero codes."""
    lo, hi = float(values.min()), float(values.max())
    maxcode = (1 << bits) - 1
    dt = _uint_dtype(bits)
    if hi <= lo:
        return np.zeros(len(values), dtype=dt), lo, hi
    norm = np.clip((values.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(norm * maxcode).astype(dt), lo, hi


def _dequantize(codes: np.ndarray, bits: int, lo: float, hi: float) -> np.ndarray:
    maxcode = (1 << bits) - 1
    if hi <= lo or maxcode == 0:
        return np.full(len(codes), lo, dtype=np.float32)
    return (lo + codes.astype(np.float64) / maxcode * (hi - lo)).astype(np.float32)


def _quant_group(arr2d: np.ndarray, bits: int) -> tuple[np.ndarray, list[list[float]]]:
    """Column-wise uniform quantize an (N,C) array. Returns (codes (N,C), ranges)."""
    dt = _uint_dtype(bits)
    out = np.empty(arr2d.shape, dtype=dt)
    ranges: list[list[float]] = []
    for j in range(arr2d.shape[1]):
        codes, lo, hi = _quantize(arr2d[:, j], bits)
        out[:, j] = codes
        ranges.append([lo, hi])
    return out, ranges


def _dequant_group(codes2d: np.ndarray, bits: int, ranges: list[list[float]]) -> np.ndarray:
    out = np.empty(codes2d.shape, dtype=np.float32)
    for j, (lo, hi) in enumerate(ranges):
        out[:, j] = _dequantize(codes2d[:, j], bits, lo, hi)
    return out


def _encode_quats(quats: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Smallest-three encode unit quats. The largest-magnitude component is made
    positive (q == -q as a rotation) and dropped; the other three are quantized.
    Returns (component codes (N,3), largest-index (N,) uint8)."""
    q = quats.astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    rows = np.arange(len(q))
    idx = np.argmax(np.abs(q), axis=1)
    sign = np.sign(q[rows, idx])
    sign[sign == 0] = 1.0
    q *= sign[:, None]
    keep = np.ones(q.shape, dtype=bool)
    keep[rows, idx] = False
    others = np.clip(q[keep].reshape(-1, 3), -_QUAT_LIMIT, _QUAT_LIMIT)
    codes, _lo, _hi = _quantize(others.ravel(), bits)  # fixed range below
    # Re-quantize over the FIXED symmetric range so decode needs no per-file range.
    maxcode = (1 << bits) - 1
    norm = np.clip((others + _QUAT_LIMIT) / (2.0 * _QUAT_LIMIT), 0.0, 1.0)
    codes = np.rint(norm * maxcode).astype(_uint_dtype(bits))
    return codes, idx.astype(np.uint8)


def _decode_quats(codes: np.ndarray, idx: np.ndarray, bits: int) -> np.ndarray:
    maxcode = (1 << bits) - 1
    others = (codes.astype(np.float64) / maxcode) * (2.0 * _QUAT_LIMIT) - _QUAT_LIMIT
    n = len(idx)
    rows = np.arange(n)
    q = np.zeros((n, 4), dtype=np.float64)
    place = np.ones((n, 4), dtype=bool)
    place[rows, idx] = False
    q[place] = others.reshape(-1)
    q[rows, idx] = np.sqrt(np.clip(1.0 - (others ** 2).sum(axis=1), 0.0, 1.0))
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    return q.astype(np.float32)


def _quats_to_normals(quats: np.ndarray) -> np.ndarray:
    """Surfel normal = third column of the (wxyz) quaternion's rotation matrix
    (matches Stage 3/6), so decode reproduces the dropped nx,ny,nz exactly."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    n = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)
    return (n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


# --- container encode / decode ------------------------------------------------


def quantize_ply(
    in_path: Path, out_path: Path, config: QuantConfig = QuantConfig()
) -> dict[str, Any]:
    """Quantize a 2DGS `.ply` into a `.sqz` container at `out_path`, returning a
    summary (sizes, ratio, and measured per-attribute reconstruction error)."""
    in_path, out_path = Path(in_path), Path(out_path)
    count, ply_order, cols = _read_ply(in_path)
    if count == 0:
        raise ValueError(f"{in_path}: empty cloud")
    for req in (*_POS, *_QUAT):
        if req not in cols:
            raise ValueError(f"{in_path}: missing required property '{req}' (not a 2DGS splat?)")

    scale_fields = sorted(n for n in ply_order if n.startswith("scale_"))
    known = (
        set(_POS) | set(_NORMAL) | set(_COLOR) | {_OPACITY}
        | set(scale_fields) | set(_QUAT)
    )
    # Any higher-order SH (`f_rest_*`) from a legacy view-dependent .ply is not
    # recognized (the pipeline is flat, degree-0), so it lands in `extras` and
    # passes through as raw float32 — kept losslessly, just uncompressed.
    extras = [n for n in ply_order if n not in known]

    blocks: list[dict[str, Any]] = []
    blob = bytearray()

    def add_uniform(names: list[str], bits: int) -> None:
        arr = np.stack([cols[n] for n in names], axis=1)
        codes, ranges = _quant_group(arr, bits)
        raw = np.ascontiguousarray(codes).tobytes()
        blocks.append(
            {"kind": "uniform", "names": names, "dtype": codes.dtype.str, "bits": bits,
             "ranges": ranges, "shape": list(codes.shape), "nbytes": len(raw)}
        )
        blob.extend(raw)

    add_uniform(list(_POS), config.pos_bits)
    if all(c in cols for c in _COLOR):
        add_uniform(list(_COLOR), config.color_bits)
    if _OPACITY in cols:
        a = np.clip(_sigmoid(cols[_OPACITY].astype(np.float64)), config.opacity_clamp, 1 - config.opacity_clamp)
        codes, lo, hi = _quantize(a, config.opacity_bits)
        codes = codes.reshape(-1, 1)
        raw = np.ascontiguousarray(codes).tobytes()
        blocks.append(
            {"kind": "opacity", "names": [_OPACITY], "dtype": codes.dtype.str,
             "bits": config.opacity_bits, "ranges": [[lo, hi]], "shape": [count, 1], "nbytes": len(raw)}
        )
        blob.extend(raw)
    if scale_fields:
        add_uniform(scale_fields, config.scale_bits)

    quats = np.stack([cols[r] for r in _QUAT], axis=1)
    comp, idx = _encode_quats(quats, config.quat_bits)
    raw_c = np.ascontiguousarray(comp).tobytes()
    raw_i = np.ascontiguousarray(idx).tobytes()
    blocks.append(
        {"kind": "quat", "names": list(_QUAT), "dtype": comp.dtype.str, "bits": config.quat_bits,
         "shape": list(comp.shape), "nbytes": len(raw_c),
         "idx_dtype": idx.dtype.str, "idx_nbytes": len(raw_i)}
    )
    blob.extend(raw_c)
    blob.extend(raw_i)

    if extras:
        arr = np.stack([cols[n] for n in extras], axis=1).astype("<f4")
        raw = np.ascontiguousarray(arr).tobytes()
        blocks.append(
            {"kind": "raw", "names": extras, "dtype": "<f4", "shape": list(arr.shape), "nbytes": len(raw)}
        )
        blob.extend(raw)

    header = {
        "version": CONTAINER_VERSION,
        "count": count,
        "config": config.as_summary(),
        "ply_order": ply_order,
        "has_normals": all(n in ply_order for n in _NORMAL),
        "blocks": blocks,
    }
    header_bytes = json.dumps(header).encode("utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(MAGIC)
        f.write(len(header_bytes).to_bytes(4, "little"))
        f.write(header_bytes)
        f.write(blob)
    tmp.replace(out_path)

    error = _measure_error(cols, dequantize(out_path)[1])
    in_bytes = in_path.stat().st_size
    out_bytes = out_path.stat().st_size
    return {
        "count": count,
        "in_bytes": in_bytes,
        "out_bytes": out_bytes,
        "ratio": round(in_bytes / max(out_bytes, 1), 2),
        "bytes_per_splat_in": round(in_bytes / count, 2),
        "bytes_per_splat_out": round(out_bytes / count, 2),
        "config": config.as_summary(),
        "error": error,
        "out_path": str(out_path),
    }


def dequantize(in_path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """Decode a `.sqz` back to (ply property order, {name: (N,) float32}) in the
    original stored spaces (opacity as logit, scale as log). Normals are recomputed
    from the quaternion when the source had them."""
    raw = Path(in_path).read_bytes()
    if raw[:4] != MAGIC:
        raise ValueError(f"{in_path}: not an {MAGIC.decode()} container")
    hlen = int.from_bytes(raw[4:8], "little")
    header = json.loads(raw[8 : 8 + hlen].decode("utf-8"))
    payload = raw[8 + hlen :]
    clamp = float(header["config"]["opacity_clamp"])

    cols: dict[str, np.ndarray] = {}
    off = 0
    for b in header["blocks"]:
        nb = b["nbytes"]
        arr = np.frombuffer(payload, dtype=np.dtype(b["dtype"]), count=int(np.prod(b["shape"])), offset=off)
        arr = arr.reshape(b["shape"])
        off += nb
        kind = b["kind"]
        if kind == "uniform":
            deq = _dequant_group(arr, b["bits"], b["ranges"])
            for j, name in enumerate(b["names"]):
                cols[name] = deq[:, j]
        elif kind == "opacity":
            a = _dequant_group(arr, b["bits"], b["ranges"])[:, 0].astype(np.float64)
            a = np.clip(a, clamp, 1 - clamp)
            cols[_OPACITY] = _logit(a).astype(np.float32)
        elif kind == "quat":
            idx = np.frombuffer(
                payload, dtype=np.dtype(b["idx_dtype"]), count=b["shape"][0], offset=off
            )
            off += b["idx_nbytes"]
            q = _decode_quats(arr, idx, b["bits"])
            for j, name in enumerate(b["names"]):
                cols[name] = q[:, j]
        elif kind == "raw":
            for j, name in enumerate(b["names"]):
                cols[name] = np.ascontiguousarray(arr[:, j]).astype(np.float32)
        else:
            raise ValueError(f"{in_path}: unknown block kind '{kind}'")

    if header.get("has_normals"):
        normals = _quats_to_normals(np.stack([cols[r] for r in _QUAT], axis=1))
        cols["nx"], cols["ny"], cols["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    return header["ply_order"], cols


def sqz_to_ply(in_path: Path, out_path: Path) -> None:
    """Decode a `.sqz` container back to a standard float32 2DGS `.ply`."""
    order, cols = dequantize(Path(in_path))
    _write_ply(Path(out_path), order, cols)


def _measure_error(orig: dict[str, np.ndarray], deq: dict[str, np.ndarray]) -> dict[str, Any]:
    """Reconstruction error in perceptual units, over the real decoded file."""
    out: dict[str, Any] = {}

    pos_o = np.stack([orig[n] for n in _POS], axis=1).astype(np.float64)
    pos_d = np.stack([deq[n] for n in _POS], axis=1).astype(np.float64)
    d = np.linalg.norm(pos_o - pos_d, axis=1) * 1000.0  # mm
    out["pos_mean_mm"] = round(float(d.mean()), 4)
    out["pos_max_mm"] = round(float(d.max()), 4)

    if all(c in orig for c in _COLOR):
        co = 0.5 + _SH_C0 * np.stack([orig[n] for n in _COLOR], axis=1).astype(np.float64)
        cd = 0.5 + _SH_C0 * np.stack([deq[n] for n in _COLOR], axis=1).astype(np.float64)
        e = np.abs(co - cd)
        out["color_mean"] = round(float(e.mean()), 5)
        out["color_max"] = round(float(e.max()), 5)
        out["color_max_8bit_levels"] = round(float(e.max()) * 255.0, 2)

    if _OPACITY in orig:
        ao = _sigmoid(orig[_OPACITY].astype(np.float64))
        ad = _sigmoid(deq[_OPACITY].astype(np.float64))
        e = np.abs(ao - ad)
        out["opacity_mean"] = round(float(e.mean()), 5)
        out["opacity_max"] = round(float(e.max()), 5)

    scale_fields = sorted(n for n in orig if n.startswith("scale_"))
    if scale_fields:
        so = np.exp(np.stack([orig[n] for n in scale_fields], axis=1).astype(np.float64))
        sd = np.exp(np.stack([deq[n] for n in scale_fields], axis=1).astype(np.float64))
        rel = np.abs(sd - so) / (so + 1e-12)
        out["scale_rel_mean"] = round(float(rel.mean()), 5)
        out["scale_rel_max"] = round(float(rel.max()), 5)

    qo = np.stack([orig[r] for r in _QUAT], axis=1).astype(np.float64)
    qd = np.stack([deq[r] for r in _QUAT], axis=1).astype(np.float64)
    qo /= np.linalg.norm(qo, axis=1, keepdims=True) + 1e-12
    dot = np.clip(np.abs((qo * qd).sum(axis=1)), 0.0, 1.0)
    ang = np.degrees(2.0 * np.arccos(dot))
    out["quat_mean_deg"] = round(float(ang.mean()), 4)
    out["quat_max_deg"] = round(float(ang.max()), 4)
    return out


def _main() -> None:
    ap = argparse.ArgumentParser(description="Near-lossless quantization of a 2DGS splat PLY")
    sub = ap.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode", help="quantize a .ply into a .sqz container")
    enc.add_argument("--in", dest="inp", required=True, type=Path, help="trained 2DGS .ply")
    enc.add_argument("--out", required=True, type=Path, help="output .sqz")
    enc.add_argument("--pos-bits", type=int, default=QuantConfig.pos_bits)
    enc.add_argument("--color-bits", type=int, default=QuantConfig.color_bits)
    enc.add_argument("--opacity-bits", type=int, default=QuantConfig.opacity_bits)
    enc.add_argument("--scale-bits", type=int, default=QuantConfig.scale_bits)
    enc.add_argument("--quat-bits", type=int, default=QuantConfig.quat_bits)
    enc.add_argument("--roundtrip-ply", type=Path, default=None, help="also write the dequantized .ply for viewing")

    dec = sub.add_parser("decode", help="rebuild a float32 .ply from a .sqz")
    dec.add_argument("--in", dest="inp", required=True, type=Path, help="input .sqz")
    dec.add_argument("--out", required=True, type=Path, help="output .ply")

    args = ap.parse_args()
    if args.cmd == "encode":
        config = QuantConfig(
            pos_bits=args.pos_bits, color_bits=args.color_bits, opacity_bits=args.opacity_bits,
            scale_bits=args.scale_bits, quat_bits=args.quat_bits,
        )
        summary = quantize_ply(args.inp, args.out, config)
        if args.roundtrip_ply is not None:
            sqz_to_ply(args.out, args.roundtrip_ply)
            summary["roundtrip_ply"] = str(args.roundtrip_ply)
        print(json.dumps(summary, indent=1))
    else:
        sqz_to_ply(args.inp, args.out)
        print(json.dumps({"out_path": str(args.out), "bytes": Path(args.out).stat().st_size}, indent=1))


if __name__ == "__main__":
    _main()
