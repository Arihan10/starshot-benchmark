"""Stage-5 refs + Stage-3 surfels -> a COLMAP text model (for Postshot / COLMAP).

Postshot and other radiance-field trainers don't read `transforms.json`; they
ingest a COLMAP sparse model — `cameras.txt` / `images.txt` / `points3D.txt`
alongside the RGB images. This module builds that from what stages 3-5 already
produced for a cell:

  * intrinsics + per-frame poses  <- refs/transforms.json      (Stage 5)
  * the RGB images                <- refs/frames/*.szf, decoded (Stage 5)
  * the sparse init point cloud   <- cloud.ply surfels          (Stage 3)

`transforms.json` poses are OpenCV camera-to-world (the same axes COLMAP uses),
so the extrinsics are simply `w2c = inv(c2w)` with NO OpenGL flip. Point colours
are the SH degree-0 decode of the PLY's `f_dc` (the inverse of Stage 3's encode).
The whole folder is drag-and-drop ingestible: Postshot flips Camera Poses ->
Import; COLMAP-based tools read it as `sparse/0` laid flat.
"""
from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from . import stage5

CAMERAS_TXT = "cameras.txt"
IMAGES_TXT = "images.txt"
POINTS_TXT = "points3D.txt"

# SH degree-0 basis constant: Stage 3 stores f_dc = (rgb - 0.5) / C0, so rgb =
# 0.5 + C0 * f_dc (matches splat/stage3.py's _SH_C0).
_SH_C0 = 0.28209479177387814

_PLY_DTYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """COLMAP's rotation-matrix -> (qw,qx,qy,qz). `eigh` reads the lower triangle,
    which is why K is filled below the diagonal only."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    return -q if q[0] < 0 else q


def _read_ply_vertices(path: Path) -> dict[str, np.ndarray]:
    """Binary-little-endian PLY vertex table -> {property_name: column array}."""
    data = Path(path).read_bytes()
    tag = data.find(b"end_header")
    if data[:3] != b"ply" or tag < 0:
        raise ValueError(f"{path}: not a PLY")
    end = data.find(b"\n", tag) + 1
    header = data[:end].decode("ascii", "replace").splitlines()
    if not any(h.startswith("format binary_little_endian") for h in header):
        raise ValueError(f"{path}: only binary_little_endian PLY is supported")
    count, props, in_vertex = 0, [], False
    for line in header:
        tok = line.split()
        if tok[:1] == ["element"]:
            in_vertex = tok[1] == "vertex"
            if in_vertex:
                count = int(tok[2])
        elif tok[:1] == ["property"] and in_vertex:
            ptype, name = tok[1], tok[2]
            if ptype not in _PLY_DTYPES:
                raise ValueError(f"{path}: unsupported PLY property type {ptype!r}")
            props.append((name, "<" + _PLY_DTYPES[ptype]))
    table = np.frombuffer(data, dtype=np.dtype(props), count=count, offset=end)
    return {name: table[name] for name, _ in props}


def _workers(jobs: int | None) -> int:
    return jobs if jobs and jobs > 0 else min(8, (os.cpu_count() or 4))


def decode_frames_to_png(frames_dir: Path, out_dir: Path, *, jobs: int | None = None) -> int:
    """Decode every `*.szf` reference frame in `frames_dir` to an RGB `.png` in
    `out_dir` (alpha + depth dropped), reusing the Stage-5 codec. Returns the count."""
    frames_dir, out_dir = Path(frames_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    szfs = sorted(frames_dir.glob(f"*{stage5.FRAME_SUFFIX}"))

    def _decode(szf: Path) -> None:
        (out_dir / f"{szf.stem}.png").write_bytes(stage5.frame_preview_png(szf))

    with ThreadPoolExecutor(max_workers=_workers(jobs)) as pool:
        list(pool.map(_decode, szfs))
    return len(szfs)


def export_colmap(
    refs_dir: Path, cloud_ply: Path, out_dir: Path, *, jobs: int | None = None
) -> dict[str, Any]:
    """Write a COLMAP text model (+ decoded RGB PNGs) into `out_dir` from a cell's
    Stage-5 refs (`refs_dir/transforms.json` + `frames/*.szf`) and Stage-3
    `cloud_ply`. `out_dir` is wiped first so it always reflects the current refs.
    Returns a summary: {cameras, images, points, dir}."""
    refs_dir, cloud_ply, out_dir = Path(refs_dir), Path(cloud_ply), Path(out_dir)
    doc = json.loads((refs_dir / stage5.TRANSFORMS_NAME).read_text(encoding="utf-8"))
    frames = doc["frames"]
    frames_dir = refs_dir / stage5.FRAMES_DIRNAME

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    w, h = int(doc["w"]), int(doc["h"])
    fx, fy, cx, cy = doc["fl_x"], doc["fl_y"], doc["cx"], doc["cy"]
    (out_dir / CAMERAS_TXT).write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
        f"1 PINHOLE {w} {h} {fx:.10g} {fy:.10g} {cx:.10g} {cy:.10g}\n",
        encoding="utf-8",
    )

    records = []
    for i, fr in enumerate(frames, 1):
        src = fr.get("frame_path") or fr.get("file_path")
        if src is None:
            raise ValueError(f"frame {i} has neither 'frame_path' (SZF) nor 'file_path' (PNG)")
        stem = Path(src).stem
        w2c = np.linalg.inv(np.asarray(fr["transform_matrix"], dtype=np.float64))
        q = rotmat2qvec(w2c[:3, :3])
        t = w2c[:3, 3]
        records.append((i, stem, q, t, fr))

    def _decode(rec: tuple) -> None:
        stem, fr = rec[1], rec[4]
        dst = out_dir / f"{stem}.png"
        if fr.get("frame_path"):
            dst.write_bytes(
                stage5.frame_preview_png(frames_dir / f"{stem}{stage5.FRAME_SUFFIX}")
            )
        else:  # legacy PNG-triple refs: the RGB is already a PNG beside transforms.json
            shutil.copyfile(refs_dir / fr["file_path"], dst)

    with ThreadPoolExecutor(max_workers=_workers(jobs)) as pool:
        list(pool.map(_decode, records))

    img_lines = [
        "# Image list with two lines of data per image:\n",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
        f"# Number of images: {len(records)}, mean observations per image: 0\n",
    ]
    for i, stem, q, t, _fr in records:
        img_lines.append(
            f"{i} {q[0]:.10g} {q[1]:.10g} {q[2]:.10g} {q[3]:.10g} "
            f"{t[0]:.10g} {t[1]:.10g} {t[2]:.10g} 1 {stem}.png\n\n"
        )
    (out_dir / IMAGES_TXT).write_text("".join(img_lines), encoding="utf-8")

    ply = _read_ply_vertices(cloud_ply)
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], axis=1).astype(np.float64)
    fdc = np.stack([ply["f_dc_0"], ply["f_dc_1"], ply["f_dc_2"]], axis=1)
    rgb = np.rint(np.clip(0.5 + _SH_C0 * fdc, 0.0, 1.0) * 255).astype(int)
    pt_lines = [
        "# 3D point list with one line of data per point:\n",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n",
        f"# Number of points: {len(xyz)}, mean track length: 0\n",
    ]
    pt_lines += [
        f"{j} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]} 1.0\n"
        for j, (p, c) in enumerate(zip(xyz, rgb), 1)
    ]
    (out_dir / POINTS_TXT).write_text("".join(pt_lines), encoding="utf-8")

    return {"cameras": 1, "images": len(records), "points": int(len(xyz)), "dir": str(out_dir)}
