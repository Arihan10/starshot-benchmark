"""Torch-free smoke tests for splat/stage5.py (no GPU / nvdiffrast needed).

Validates the pose / intrinsics / projection / IO layer that runs on any host:
  * intrinsics_matrix          — focal + principal point at 90 deg
  * opencv_c2w                 — orthonormal, right-handed, OpenCV axis semantics
  * gl_projection              — near->-1, far->+1 depth mapping
  * PROJECTION ROUND-TRIP      — an OpenCV pinhole pixel lands at the same (row,col)
                                 that nvdiffrast's gl_projection + top-down flip
                                 produce. This is the math half of smoke test #1
                                 (pose round-trip) without needing gsplat.
  * enumerate_views            — (camera, face) fan-out + c2w consistency
  * load_camera_plan           — schema guard
  * write_transforms           — transforms.json round-trip
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

# --- gl_projection depth mapping --------------------------------------------
near, far = 0.05, 10.0
proj = stage5.gl_projection(K, R, R, near, far)
p_near = proj @ np.array([0, 0, near, 1.0]); p_near /= p_near[3]
p_far = proj @ np.array([0, 0, far, 1.0]); p_far /= p_far[3]
check("gl_projection Z=near -> NDC z=-1", np.isclose(p_near[2], -1.0, atol=1e-6), f"{p_near[2]:.6f}")
check("gl_projection Z=far  -> NDC z=+1", np.isclose(p_far[2], 1.0, atol=1e-6), f"{p_far[2]:.6f}")

# --- PROJECTION ROUND-TRIP (the crux) ---------------------------------------
# For several world points and camera poses, the OpenCV pinhole pixel (u,v) must
# equal the pixel nvdiffrast produces: col from NDC.x, row from NDC.y AFTER the
# _render_view top-down flip (row = (1 - ndc_y)/2 * H).
rng = np.random.default_rng(0)
max_err = 0.0
tested = 0
for name in CUBE_FACE_NAMES:
    basis = CUBE_FACES[name]
    cam_pos = rng.uniform(-2, 2, size=3)
    c2w = stage5.opencv_c2w(cam_pos, np.asarray(basis["forward"]), np.asarray(basis["up"]))
    w2c = np.linalg.inv(c2w)
    for _ in range(200):
        p_world = rng.uniform(-3, 3, size=3)
        p_cam = w2c @ np.array([*p_world, 1.0])
        x, y, z = p_cam[:3]
        if z <= near or z >= far:
            continue  # outside frustum depth range
        u = K[0, 0] * x / z + K[0, 2]       # OpenCV pixel (v measured from top)
        v = K[1, 1] * y / z + K[1, 2]
        if not (0 <= u <= R and 0 <= v <= R):
            continue  # off-image; only round-trip visible pixels
        clip = proj @ p_cam
        ndc = clip[:3] / clip[3]
        col = (ndc[0] + 1.0) / 2.0 * R
        row = (1.0 - ndc[1]) / 2.0 * R       # after the top-down vertical flip
        err = max(abs(col - u), abs(row - v))
        max_err = max(max_err, err)
        tested += 1
check("projection round-trip OpenCV pixel == nvdiffrast pixel",
      tested > 100 and max_err < 1e-3, f"n={tested}, max_err={max_err:.2e}px")

# --- point behind camera is culled (clip.w <= 0) ----------------------------
c2w = stage5.opencv_c2w(np.zeros(3), np.array([0, 0, 1.0]), np.array([0, 1.0, 0]))
w2c = np.linalg.inv(c2w)
behind = w2c @ np.array([0, 0, -5.0, 1.0])   # 5 m behind the +Z-looking camera
clipw = (proj @ behind)[3]
check("point behind camera has clip.w <= 0", clipw <= 0, f"clip.w={clipw:.3f}")

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

# --- load_camera_plan schema guard ------------------------------------------
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

    # --- write_transforms round-trip ----------------------------------------
    frames = [{"file_path": "rgb/cam00000_+x.png", "camera_index": 0, "face": "+x",
               "transform_matrix": v0["c2w"].tolist()}]
    out = stage5.write_transforms(Path(td), K, R, near, far, frames)
    doc = json.loads(out.read_text(encoding="utf-8"))
    ok = (doc["convention"] == "opencv_c2w" and doc["depth"] == "planar_z_metric"
          and doc["w"] == R and np.isclose(doc["fl_x"], K[0, 0]) and len(doc["frames"]) == 1)
    check("write_transforms fields + convention tags", ok)

print("\n" + ("ALL TORCH-FREE SMOKE TESTS PASSED" if not FAILS else f"FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
