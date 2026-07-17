"""GPU smoke test for splat/stage5.py — exercises the REAL render_references path
on a synthetic scene, so every failure is a wiring/convention bug, not data.

Scene: one fronto-parallel textured quad at world z = Z0, camera at the origin
looking +Z (cube face "+z"). The texture encodes its own coordinates as colour
(R = u_tex, G = v_tex), so a rendered pixel's colour reveals exactly which texel
was sampled — a direct, unambiguous check of BOTH convention flips at once.

Checks:
  1. nvdiffrast RasterizeCudaContext builds + runs on this GPU.
  2. Depth == planar camera-space Z (constant Z0 across the plane), NOT ray
     distance (which grows off-axis) — smoke test #2.
  3. Image orientation + texture-V flip: the four image quadrants show the four
     expected texel colours (top-left dark, top-right red, bottom-left green,
     bottom-right yellow). A wrong flip mirrors/rotates this.
  4. Alpha: 1 on the quad, 0 on background; background RGB = configured colour.
  5. transforms.json written with the right convention tags + frame count.
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
from splat import stage5
from splat.stage4 import CUBE_FACES

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# --- 0. torch + CUDA ---------------------------------------------------------
try:
    import torch
except Exception as e:
    print(f"[FAIL] import torch — {type(e).__name__}: {e}")
    sys.exit(2)
print(f"torch {torch.__version__}, cuda available={torch.cuda.is_available()}, "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
check("torch.cuda.is_available()", torch.cuda.is_available())
try:
    import nvdiffrast.torch as _dr  # noqa: F401
    print("nvdiffrast imported OK")
except Exception as e:
    print(f"[FAIL] import nvdiffrast — {type(e).__name__}: {e}")
    print("=> GPU render path cannot be tested until nvdiffrast is installed + compilable.")
    sys.exit(3)

R = 512
Z0 = 2.0
HALF = 1.3           # quad half-extent (world); < view half-width (=Z0 at 90deg) => border
near, far = 0.05, 10.0

work = Path(tempfile.mkdtemp(prefix="stage5_smoke_"))
raw_dir = work / "objects"
raw_dir.mkdir(parents=True)
out_dir = work / "refs"

# --- build the UV-encoding texture (row 0 = top = glTF v=0) ------------------
TW = 256
xs = np.linspace(0, 1, TW, dtype=np.float32)
u_grid, v_grid = np.meshgrid(xs, xs)                 # u_grid varies across cols, v down rows
tex = np.zeros((TW, TW, 3), dtype=np.float32)
tex[..., 0] = u_grid                                  # R = u_tex
tex[..., 1] = v_grid                                  # G = v_tex
tex[..., 2] = 0.5
tex_img = Image.fromarray((tex * 255).astype(np.uint8), mode="RGB")

# quad vertices (A,B,C,D) with glTF UVs; A(UV 0,0)=texture top-left.
verts = np.array([[ HALF,  HALF, Z0],   # A
                  [-HALF,  HALF, Z0],   # B
                  [-HALF, -HALF, Z0],   # C
                  [ HALF, -HALF, Z0]], dtype=np.float64)
faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
material = trimesh.visual.material.PBRMaterial(baseColorTexture=tex_img, alphaMode="OPAQUE")
mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                       visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
                       process=False)
mesh.export(raw_dir / "quad.glb")

# --- minimal Stage-4 cameras.json -------------------------------------------
plan = {
    "intrinsics": {"resolution": R, "face_fov_deg": 90.0, "near": near, "far": far},
    "cube_faces": CUBE_FACES,
    "cameras": [{"pos": [0.0, 0.0, 0.0], "faces": [{"dir": "+z", "covers": 2}]}],
}
cameras_path = work / "cameras.json"
cameras_path.write_text(json.dumps(plan), encoding="utf-8")

# --- 1. run the real entry point --------------------------------------------
try:
    summary = stage5.render_references(
        run="smoke", slot="smoke", model="smoke",
        raw_dir=raw_dir, cameras_path=cameras_path, out_dir=out_dir,
    )
    check("render_references ran without error", True)
    print("  summary:", json.dumps(summary))
except Exception as e:
    import traceback
    traceback.print_exc()
    check("render_references ran without error", False, f"{type(e).__name__}: {e}")
    print(f"\nFAILURES: {FAILS}")
    sys.exit(4)

vid = "cam00000_+z"
rgb = np.asarray(Image.open(out_dir / "rgb" / f"{vid}.png").convert("RGB"), dtype=np.float32) / 255.0
depth = stage5.load_depth_png(out_dir / "depth" / f"{vid}.png", near, far)
alpha = np.asarray(Image.open(out_dir / "alpha" / f"{vid}.png"), dtype=np.float32) / 255.0
print(f"  rgb {rgb.shape}, depth {depth.shape}, alpha {alpha.shape}")


def px(frac_row, frac_col):
    return int(frac_row * R), int(frac_col * R)


# --- 2. depth == planar Z (not ray distance) --------------------------------
cr, cc = px(0.5, 0.5)
check("depth at center == Z0", abs(depth[cr, cc] - Z0) < 0.02, f"{depth[cr, cc]:.4f} (want {Z0})")
# off-axis covered pixel: planar stays Z0, ray distance would be larger.
orow, ocol = px(0.75, 0.75)
d_off = depth[orow, ocol]
# reconstruct the ray distance this pixel WOULD have if depth were ray-length.
K = stage5.intrinsics_matrix(R, 90.0)
xn = (ocol - K[0, 2]) / K[0, 0]
yn = (orow - K[1, 2]) / K[1, 1]
ray_len = Z0 * np.sqrt(1 + xn * xn + yn * yn)
check("off-axis depth == planar Z0 (NOT ray distance)",
      abs(d_off - Z0) < 0.03 and abs(d_off - ray_len) > 0.1,
      f"depth={d_off:.4f}, planar={Z0}, ray={ray_len:.4f}")

# --- 3. orientation + texture-V flip: four quadrant colours -----------------
def region_mean(fr, fc):
    r, c = px(fr, fc)
    patch = rgb[r - 8:r + 8, c - 8:c + 8]
    return patch.reshape(-1, 3).mean(axis=0)

tl, tr = region_mean(0.30, 0.30), region_mean(0.30, 0.70)
bl, br = region_mean(0.70, 0.30), region_mean(0.70, 0.70)
print(f"  quadrant means  TL={tl.round(2)} TR={tr.round(2)} BL={bl.round(2)} BR={br.round(2)}")
check("top-left is low-u low-v (dark)",   tl[0] < 0.4 and tl[1] < 0.4)
check("top-right is high-u low-v (red)",  tr[0] > 0.6 and tr[1] < 0.4)
check("bottom-left is low-u high-v (green)", bl[0] < 0.4 and bl[1] > 0.6)
check("bottom-right is high-u high-v (yellow)", br[0] > 0.6 and br[1] > 0.6)

# --- 4. alpha coverage + background -----------------------------------------
check("alpha == 1 on quad center", alpha[cr, cc] > 0.99, f"{alpha[cr, cc]:.3f}")
brow, bcol = px(0.02, 0.02)                       # image corner -> world +/-2 > HALF => background
check("alpha == 0 on background corner", alpha[brow, bcol] < 0.01, f"{alpha[brow, bcol]:.3f}")
check("background RGB == configured (0,0,0)", float(rgb[brow, bcol].max()) < 0.02)
check("background depth == 0", abs(float(depth[brow, bcol])) < 1e-6)

# --- 5. transforms.json ------------------------------------------------------
doc = json.loads((out_dir / "transforms.json").read_text(encoding="utf-8"))
check("transforms convention/depth tags + 1 frame",
      doc.get("convention") == "opencv_c2w" and doc.get("depth") == stage5.DEPTH_ENCODING
      and len(doc.get("frames", [])) == 1)

# --- 6. chunked rasterization is output-identical (pure compute change) ------
# Force a 1-triangle chunk so the 2-triangle quad rasterizes in TWO batches merged by
# nearest depth; the result must match the single-batch render pixel-for-pixel.
chunk_dir = work / "refs_chunked"
_orig_cap = stage5._RASTER_CHUNK_TRIS
try:
    stage5._RASTER_CHUNK_TRIS = 1
    stage5.render_references(run="smoke", slot="smoke", model="smoke",
                             raw_dir=raw_dir, cameras_path=cameras_path, out_dir=chunk_dir)
finally:
    stage5._RASTER_CHUNK_TRIS = _orig_cap
rgb_c = np.asarray(Image.open(chunk_dir / "rgb" / f"{vid}.png").convert("RGB"), np.float32) / 255.0
depth_c = stage5.load_depth_png(chunk_dir / "depth" / f"{vid}.png", near, far)
alpha_c = np.asarray(Image.open(chunk_dir / "alpha" / f"{vid}.png"), np.float32) / 255.0
check("chunked rgb == single-pass rgb", np.array_equal(rgb_c, rgb))
check("chunked depth == single-pass depth", np.allclose(depth_c, depth, atol=1e-4))
check("chunked alpha == single-pass alpha", np.array_equal(alpha_c, alpha))

# save a copy of the render for eyeballing
Image.open(out_dir / "rgb" / f"{vid}.png").save(work / "preview_rgb.png")
print(f"\nartifacts in: {work}")
print("ALL GPU SMOKE TESTS PASSED" if not FAILS else f"FAILURES: {FAILS}")
sys.exit(1 if FAILS else 0)
