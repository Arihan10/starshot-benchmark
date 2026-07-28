"""Disambiguate the vertical inversion found in smoke_stage5_gpu.py:
is it the image top-down flip (_top_down) or the texture-V flip (1 - uv.v)?

Asymmetric quad Y in [-1.3, +0.6] (so it does NOT reach the top of the view) with
a texture that is RED on its top half (v<0.5) and BLUE on the bottom half (v>0.5).

  * alpha at row 130 vs row 400  -> tests the IMAGE flip (where geometry lands).
      correct: row130 = background(0), row400 = quad(1)
  * colour at image center       -> tests the TEXTURE-V flip (center is geometry-
      stable, so its colour reflects only texture orientation).
      correct: center = RED (world Y=0 -> UV v=0.316 -> top half = red)
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

R, Z0, near, far = 512, 2.0, 0.05, 10.0
work = Path(tempfile.mkdtemp(prefix="stage5_flip_"))
raw = work / "objects"; raw.mkdir(parents=True)

# texture: top half (v<0.5, rows 0:H/2) RED, bottom half BLUE. row 0 = top.
TW = 256
tex = np.zeros((TW, TW, 3), dtype=np.uint8)
tex[: TW // 2, :, 0] = 255          # top rows -> red
tex[TW // 2 :, :, 2] = 255          # bottom rows -> blue
tex_img = Image.fromarray(tex, "RGB")

# quad covering world Y in [-1.3, +0.6] (upper edge below the top of view), full UVs.
verts = np.array([[ 1.3,  0.6, Z0],   # A UV(0,0) top-left
                  [-1.3,  0.6, Z0],   # B UV(1,0)
                  [-1.3, -1.3, Z0],   # C UV(1,1)
                  [ 1.3, -1.3, Z0]], dtype=np.float64)  # D UV(0,1)
faces = np.array([[0, 1, 2], [0, 2, 3]])
uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
mat = trimesh.visual.material.PBRMaterial(baseColorTexture=tex_img, alphaMode="OPAQUE")
trimesh.Trimesh(vertices=verts, faces=faces,
                visual=trimesh.visual.TextureVisuals(uv=uv, material=mat),
                process=False).export(raw / "quad.glb")

plan = {"intrinsics": {"resolution": R, "face_fov_deg": 90.0, "near": near, "far": far},
        "cube_faces": CUBE_FACES,
        "cameras": [{"pos": [0.0, 0.0, 0.0], "faces": [{"dir": "+z", "covers": 2}]}]}
(work / "cameras.json").write_text(json.dumps(plan), encoding="utf-8")

stage5.render_references(run="f", slot="f", model="f", raw_dir=raw,
                        cameras_path=work / "cameras.json", out_dir=work / "refs")
vid = "cam00000_+z"
rgb = np.asarray(Image.open(work / "refs" / "rgb" / f"{vid}.png").convert("RGB"), np.float32) / 255.0
alpha = np.asarray(Image.open(work / "refs" / "alpha" / f"{vid}.png"), np.float32) / 255.0

a130 = float(alpha[130, 256])   # correct: 0 (background above quad)
a400 = float(alpha[400, 256])   # correct: 1 (inside quad)
center = rgb[256, 256]          # correct: RED
print(f"alpha[row130]={a130:.2f} (correct~0)   alpha[row400]={a400:.2f} (correct~1)")
print(f"center RGB={center.round(2)} (correct = red [1,0,0])")

image_flip_ok = a130 < 0.5 and a400 > 0.5
texture_v_ok = center[0] > 0.5 and center[2] < 0.5   # red not blue
print(f"\nIMAGE _top_down flip correct? {image_flip_ok}")
print(f"TEXTURE-V flip correct?       {texture_v_ok}")
if image_flip_ok and not texture_v_ok:
    print("=> DIAGNOSIS: geometry/image is upright; the TEXTURE-V flip (1-uv.v) is WRONG.")
elif not image_flip_ok and texture_v_ok:
    print("=> DIAGNOSIS: texture ok; the IMAGE _top_down flip is WRONG.")
elif not image_flip_ok and not texture_v_ok:
    print("=> DIAGNOSIS: BOTH flips wrong (net effect may cancel in some cases).")
else:
    print("=> both correct (unexpected given gpu test failure).")
print(f"artifacts: {work}")
