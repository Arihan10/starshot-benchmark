"""Stage-5 depth-peeled compositing smoke test — REQUIRES CUDA (GPU box, not
Apple Silicon). Validates the glass-compositing change end to end through
`render_references` with closed-form expected values.

Scene (world units metres, camera at the origin looking +z):
    * a 2×2 m BLEND pane at z = 2, uniform red texels with alpha 64/255 ≈ 0.251
    * a 1×1 m opaque green wall whose front face is at z = 2.9 (behind the
      pane's centre; the pane's edges see nothing behind them)

Checks, per pixel class (a = 64/255, colors in [0,1], background black):
    pane-over-wall (image centre):
        rgb   = a·red + (1−a)·green
        alpha = 1.0                       (accumulated coverage)
        depth = a·2.0 + (1−a)·2.9         (expected planar Z, gsplat "ED")
    pane-over-void (off-centre, wall missed):
        rgb   = a·red   (over black)
        alpha = a
        depth = 2.0
    empty corner: rgb = 0, alpha = 0, depth = 0
    OPAQUE control (same pane, alphaMode OPAQUE): texture alpha must be
        IGNORED (glTF) → centre rgb = red, alpha = 1, depth = 2.0 — also
        proves an all-opaque view reduces to the pre-peeling output.

Run: python smoke_stage5_glass.py   (in the GPU box venv with torch+nvdiffrast)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splat import stage4, stage5  # noqa: E402

trimesh.util.log.setLevel("ERROR")

FAILS: list[str] = []
RES = 512
A = 64.0 / 255.0          # pane alpha
Z_PANE, Z_WALL = 2.0, 2.9  # planar depths of the pane and the wall's front face


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def approx(x, y, tol) -> bool:
    return bool(np.all(np.abs(np.asarray(x, float) - np.asarray(y, float)) <= tol))


def build_scene(d: Path, pane_mode: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    tex = np.zeros((64, 64, 4), dtype=np.uint8)
    tex[..., 0] = 255                      # red
    tex[..., 3] = 64                       # alpha ≈ 0.251
    verts = np.array([[-1, -1, Z_PANE], [1, -1, Z_PANE], [1, 1, Z_PANE], [-1, 1, Z_PANE]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(tex, "RGBA"), alphaMode=pane_mode
    )
    pane = trimesh.Trimesh(verts, faces,
                           visual=trimesh.visual.TextureVisuals(uv=uv, material=mat),
                           process=False)
    pane.export(d / "o000.glb")

    wall = trimesh.creation.box(extents=[1.0, 1.0, 0.2])
    wall.apply_translation([0, 0, Z_WALL + 0.1])   # front face lands at Z_WALL
    wmat = trimesh.visual.material.PBRMaterial(baseColorFactor=[0.0, 1.0, 0.0, 1.0])
    wall.visual = trimesh.visual.TextureVisuals(material=wmat)
    wall.export(d / "o001.glb")


def write_plan(path: Path) -> None:
    plan = {
        "intrinsics": {"face_fov_deg": 90.0, "resolution": RES, "near": 0.05,
                       "far": 50.0, "min_px_per_patch": 20.0,
                       "footprint_k": 12.8, "focal_px": RES / 2.0},
        "cube_faces": stage4.CUBE_FACES,
        "cameras": [{"pos": [0.0, 0.0, 0.0], "faces": [{"dir": "+z", "covers": 1}],
                     "covers": 1}],
    }
    path.write_text(json.dumps(plan), encoding="utf-8")


def render(raw: Path, out: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plan_path = out / "cameras.json"
    out.mkdir(parents=True, exist_ok=True)
    write_plan(plan_path)
    stage5.render_references(run="t", slot="t", model="t", raw_dir=raw,
                             cameras_path=plan_path, out_dir=out)
    rgb = np.asarray(Image.open(out / "rgb" / "cam00000_+z.png"), np.float32) / 255.0
    alpha = np.asarray(Image.open(out / "alpha" / "cam00000_+z.png"), np.float32) / 255.0
    depth = np.load(out / "depth" / "cam00000_+z.npy")
    return rgb, alpha, depth


work = Path(tempfile.mkdtemp(prefix="stage5_glass_"))
print(f"workdir: {work}")

# Pixel coordinates. Pinhole: col = f·(x/z) + cx, row = f·(y/z) + cy (OpenCV
# y-down; the saved image is already top-down). f = cx = cy = RES/2.
CENTRE = (RES // 2, RES // 2)                    # pane(z=2) over wall(z=2.9)
side_col = int(RES / 2 * 0.4 + RES / 2)          # x/z = 0.4 → pane only:
PANE_ONLY = (RES // 2, side_col)                 #   pane hit at x=0.8; wall ray
CORNER = (8, 8)                                  #   exits at x=1.16 > 0.5. Empty sky.

print("\n=== BLEND pane over wall (composited references) ===")
build_scene(work / "glass" / "objects", "BLEND")
rgb, alpha, depth = render(work / "glass" / "objects", work / "glass" / "refs")
r, c = CENTRE
check("centre rgb = a·red + (1−a)·green",
      approx(rgb[r, c], [A, 1 - A, 0.0], 0.02), f"{rgb[r, c]}")
check("centre alpha = 1 (accumulated coverage)", approx(alpha[r, c], 1.0, 0.01),
      f"{alpha[r, c]:.4f}")
want_d = A * Z_PANE + (1 - A) * Z_WALL
check("centre depth = expected planar Z", approx(depth[r, c], want_d, 0.02),
      f"{depth[r, c]:.4f} vs {want_d:.4f}")
r, c = PANE_ONLY
check("pane-only rgb = a·red over black", approx(rgb[r, c], [A, 0.0, 0.0], 0.02),
      f"{rgb[r, c]}")
check("pane-only alpha = a", approx(alpha[r, c], A, 0.01), f"{alpha[r, c]:.4f}")
check("pane-only depth = pane Z", approx(depth[r, c], Z_PANE, 0.02),
      f"{depth[r, c]:.4f}")
r, c = CORNER
check("empty pixel: rgb 0 / alpha 0 / depth 0",
      approx(rgb[r, c], 0.0, 0.005) and approx(alpha[r, c], 0.0, 0.005)
      and approx(depth[r, c], 0.0, 1e-6),
      f"rgb={rgb[r, c]} a={alpha[r, c]:.4f} d={depth[r, c]:.4f}")

print("\n=== OPAQUE control (texture alpha ignored; pre-peeling behavior) ===")
build_scene(work / "opaque" / "objects", "OPAQUE")
rgb, alpha, depth = render(work / "opaque" / "objects", work / "opaque" / "refs")
r, c = CENTRE
check("centre rgb = pure red (alpha channel ignored)",
      approx(rgb[r, c], [1.0, 0.0, 0.0], 0.02), f"{rgb[r, c]}")
check("centre alpha = 1", approx(alpha[r, c], 1.0, 0.01), f"{alpha[r, c]:.4f}")
check("centre depth = pane Z (nearest opaque)", approx(depth[r, c], Z_PANE, 0.02),
      f"{depth[r, c]:.4f}")

print("\n" + ("ALL STAGE-5 GLASS TESTS PASSED" if not FAILS else f"FAILURES ({len(FAILS)}): {FAILS}"))
print(f"artifacts: {work}")
sys.exit(1 if FAILS else 0)
