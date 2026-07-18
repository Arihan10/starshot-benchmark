"""Torch-free smoke tests for splat/stage5.py (no GPU / browser needed).

Validates the reference-render CONTRACT module — everything the capture server
side runs — on any host:
  * intrinsics_matrix          — focal + principal point at 90 deg
  * opencv_c2w                 — orthonormal, right-handed, OpenCV axis semantics
  * enumerate_views            — (camera, face) fan-out + c2w consistency
  * load_camera_plan           — schema guard
  * depth codec                — encode/decode round-trip, background code 0,
                                 near-clamp, and the GPU pack-shader formula match
  * write_reference_frame      — the encode-pool worker: raw RGBA + depth codes →
                                 rgb/alpha/depth PNGs, bottom-up flip, atomicity
  * view_rendered/pending      — the resume signal over partial artifacts
  * write_transforms           — transforms.json round-trip + convention tags

The GPU-side conventions (pose aiming, texture V, planar-vs-ray depth, the
pack shader's exact codes) are validated in-browser by the capture page's
?selftest=1 mode (client/public/js/splatcapture.js).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splat import stage5
from splat.stage4 import CUBE_FACES, CUBE_FACE_NAMES

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# --- intrinsics --------------------------------------------------------------
R = 512
K = stage5.intrinsics_matrix(R, 90.0)
check("intrinsics f = R/2 at 90deg", np.isclose(K[0, 0], R / 2) and np.isclose(K[1, 1], R / 2),
      f"f={K[0,0]}")
check("intrinsics principal point centered", np.isclose(K[0, 2], R / 2) and np.isclose(K[1, 2], R / 2),
      f"cx={K[0,2]}, cy={K[1,2]}")

# --- opencv_c2w axis semantics ----------------------------------------------
for name, basis in CUBE_FACES.items():
    c2w = stage5.opencv_c2w(np.zeros(3), np.asarray(basis["forward"]), np.asarray(basis["up"]))
    Rm = c2w[:3, :3]
    orth = np.allclose(Rm @ Rm.T, np.eye(3), atol=1e-9)
    det = float(np.linalg.det(Rm))
    x, y, z = Rm[:, 0], Rm[:, 1], Rm[:, 2]
    # +Z_cam must equal the face forward; +Y_cam must equal -up (OpenCV image-down).
    fwd_ok = np.allclose(z, np.asarray(basis["forward"], float), atol=1e-9)
    down_ok = np.allclose(y, -np.asarray(basis["up"], float), atol=1e-9)
    right_ok = np.allclose(x, np.cross(y, z), atol=1e-9)
    check(f"opencv_c2w[{name}] orthonormal + right-handed",
          orth and np.isclose(det, 1.0), f"det={det:.6f}")
    check(f"opencv_c2w[{name}] axes (fwd=+Z, down=+Y, right=X)",
          fwd_ok and down_ok and right_ok)

# The capture page aims a GL camera (lookAt) from the same forward/up; its frame
# is R_cv · diag(1,−1,−1). Verify that mapping is a pure axis flip (so the two
# conventions agree by construction — the page never builds an OpenCV matrix).
flip = np.diag([1.0, -1.0, -1.0])
for name, basis in CUBE_FACES.items():
    Rm = stage5.opencv_c2w(np.zeros(3), np.asarray(basis["forward"]), np.asarray(basis["up"]))[:3, :3]
    gl = Rm @ flip
    # GL camera: column 2 = backward (−forward), column 1 = up.
    ok = np.allclose(gl[:, 2], -np.asarray(basis["forward"], float), atol=1e-9) and np.allclose(
        gl[:, 1], np.asarray(basis["up"], float), atol=1e-9
    )
    check(f"GL twin of opencv_c2w[{name}] has -fwd back / up up", ok)

# --- depth codec --------------------------------------------------------------
near, far = 0.05, 10.0
z = np.array([[0.0, near, 0.02, 1.0], [2.0, 5.0, far, far * 2]], dtype=np.float32)
codes = stage5.encode_depth_u16(z, near, far)
check("depth code 0 iff background", bool(codes[0, 0] == 0) and bool((codes[z > 0] > 0).all()))
check("depth closer than near clamps to code 1 (not background)", int(codes[0, 2]) == 1,
      f"code={codes[0,2]}")
check("depth at far encodes to max code", int(codes[1, 2]) == 65535, f"code={codes[1,2]}")
dec = stage5.decode_depth_u16(codes, near, far)
in_range = (z >= near) & (z <= far)
rel = np.abs(dec[in_range] - z[in_range]) / z[in_range]
check("depth round-trip relative error < 0.02%", float(rel.max()) < 2e-4, f"max={rel.max():.2e}")
check("background decodes to 0", float(dec[0, 0]) == 0.0)

# The capture page's pack shader computes: floor(t*65534 + 0.5) + 1 with
# t = log(z/near)/log(far/near) — assert it lands within ±1 of the reference.
zs = np.linspace(near * 1.01, far * 0.99, 500)
t = np.log(zs / near) / np.log(far / near)
shader_codes = np.floor(t * 65534.0 + 0.5) + 1.0
ref_codes = stage5.encode_depth_u16(zs, near, far).astype(np.float64)
check("GPU pack-shader formula matches encode_depth_u16 (±1 code)",
      float(np.abs(shader_codes - ref_codes).max()) <= 1.0)

# --- enumerate_views ---------------------------------------------------------
plan = {
    "intrinsics": {"resolution": R, "face_fov_deg": 90.0, "near": near, "far": far},
    "cube_faces": CUBE_FACES,
    "cameras": [
        {"pos": [0.0, 1.0, 0.0], "faces": [{"dir": "+x", "covers": 5}, {"dir": "-z", "covers": 3}]},
        {"pos": [1.0, 1.0, 2.0], "faces": [{"dir": "+y", "covers": 2}]},
    ],
}
views = stage5.enumerate_views(plan)
check("enumerate_views count = sum of faces", len(views) == 3, f"got {len(views)}")
v0 = views[0]
c2w_expected = stage5.opencv_c2w(np.array([0.0, 1.0, 0.0]),
                                 np.asarray(CUBE_FACES["+x"]["forward"]),
                                 np.asarray(CUBE_FACES["+x"]["up"]))
check("enumerate_views c2w matches opencv_c2w", np.allclose(v0["c2w"], c2w_expected))
check("enumerate_views ids unique + formatted",
      {v["id"] for v in views} == {"cam00000_+x", "cam00000_-z", "cam00001_+y"})

# --- load_camera_plan schema guard + frame writer + resume + transforms -------
with tempfile.TemporaryDirectory() as td:
    good = Path(td) / "cameras.json"
    good.write_text(json.dumps(plan), encoding="utf-8")
    try:
        stage5.load_camera_plan(good)
        ok_good = True
    except Exception as e:
        ok_good = False
        print("   load good raised:", e)
    check("load_camera_plan accepts valid schema", ok_good)

    bad = Path(td) / "bad.json"
    bad.write_text(json.dumps({"cameras": []}), encoding="utf-8")  # missing intrinsics/cube_faces
    try:
        stage5.load_camera_plan(bad)
        raised = False
    except ValueError:
        raised = True
    check("load_camera_plan rejects legacy/incomplete schema", raised)

    # --- write_reference_frame: the encode-pool worker (SZF container) --------
    # NOTE: refs/frames/ is deliberately NOT pre-created — the writer must own
    # its destination dir (encode workers can't assume the runner made it).
    out_dir = Path(td) / "refs"
    r = 8
    # Distinct corner pixels so the bottom-up→top-down flip is observable: put
    # RED at the GL buffer's LAST row start (= top-left after the flip).
    rgba = np.zeros((r, r, 4), dtype=np.uint8)
    rgba[-1, 0] = [255, 0, 0, 255]     # GL bottom-up row r-1 == image row 0
    rgba[0, -1] = [0, 255, 0, 128]     # GL row 0 == image bottom
    depth_m = np.zeros((r, r), dtype=np.float32)
    depth_m[-1, 0] = 2.0               # same flip check for depth
    depth_codes = stage5.encode_depth_u16(depth_m, near, far)
    vid = views[0]["id"]
    stage5.write_reference_frame(out_dir, vid, r, rgba.tobytes(),
                                 depth_codes.astype("<u2").tobytes())
    fpath = stage5.frame_path(out_dir, vid)
    check("frame writer creates frames/ itself", fpath.is_file(), str(fpath))
    px, codes_rt = stage5.load_reference_frame(fpath)
    check("frame round-trips rgba (rows flipped: red lands top-left)",
          tuple(px[0, 0, :3]) == (255, 0, 0) and tuple(px[-1, -1, :3]) == (0, 255, 0))
    check("frame round-trips the alpha channel",
          int(px[0, 0, 3]) == 255 and int(px[-1, -1, 3]) == 128)
    dec = stage5.decode_depth_u16(codes_rt, near, far)
    check("frame round-trips depth codes (flipped, decodable)",
          abs(float(dec[0, 0]) - 2.0) < 1e-3 and float(dec[-1, -1]) == 0.0,
          f"z={dec[0,0]:.4f}")
    png_bytes = stage5.frame_preview_png(fpath)
    from io import BytesIO

    from PIL import Image
    prev = np.asarray(Image.open(BytesIO(png_bytes)).convert("RGB"))
    check("on-demand PNG preview matches the frame", tuple(prev[0, 0]) == (255, 0, 0))

    # --- resume signal ---------------------------------------------------------
    check("view_rendered true once the SZF frame exists", stage5.view_rendered(out_dir, vid))
    pend = stage5.pending_views(out_dir, views)
    check("pending_views excludes the rendered view",
          {v["id"] for v in pend} == {"cam00000_-z", "cam00001_+y"})

    # --- write_transforms round-trip -------------------------------------------
    out = stage5.write_transforms(out_dir, K, R, near, far, stage5.reference_frames(views))
    doc = json.loads(out.read_text(encoding="utf-8"))
    ok = (doc["convention"] == "opencv_c2w" and doc["depth"] == stage5.DEPTH_ENCODING
          and doc["frame_format"] == stage5.FRAME_FORMAT
          and doc["w"] == R and np.isclose(doc["fl_x"], K[0, 0]) and len(doc["frames"]) == 3
          and doc["frames"][0]["frame_path"] == f"frames/{vid}{stage5.FRAME_SUFFIX}")
    check("write_transforms fields + convention tags + frame paths", ok)

print("\n" + ("ALL TORCH-FREE SMOKE TESTS PASSED" if not FAILS else f"FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
