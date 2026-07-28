"""Convention-free check: does Stage 5's per-fragment texture colour agree with
Stage 2's surfel colour (trimesh uv_to_color) on the SAME on-disk GLB?

Both stages read the identical file, so trimesh authoring/export conventions
cancel out. The overview requires the references (Stage 5) and the surfel init
(Stage 2) to sample albedo identically. If they disagree vertically, Stage 5's
texture-V handling is the outlier.
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
from splat import stage2, stage5
from splat.stage4 import CUBE_FACES

R, Z0, near, far = 512, 2.0, 0.05, 10.0
work = Path(tempfile.mkdtemp(prefix="stage5_cons_"))
raw = work / "objects"; raw.mkdir(parents=True)

# Red top-half / blue bottom-half texture (row 0 = top).
TW = 256
tex = np.zeros((TW, TW, 3), np.uint8); tex[: TW // 2, :, 0] = 255; tex[TW // 2 :, :, 2] = 255
verts = np.array([[1.3, 1.3, Z0], [-1.3, 1.3, Z0], [-1.3, -1.3, Z0], [1.3, -1.3, Z0]], float)
faces = np.array([[0, 1, 2], [0, 2, 3]])
uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(tex, "RGB"), alphaMode="OPAQUE")
trimesh.Trimesh(vertices=verts, faces=faces,
                visual=trimesh.visual.TextureVisuals(uv=uv, material=mat),
                process=False).export(raw / "quad.glb")

# --- Stage 5 render ---
plan = {"intrinsics": {"resolution": R, "face_fov_deg": 90.0, "near": near, "far": far},
        "cube_faces": CUBE_FACES, "cameras": [{"pos": [0.0, 0.0, 0.0], "faces": [{"dir": "+z", "covers": 2}]}]}
(work / "cameras.json").write_text(json.dumps(plan), encoding="utf-8")
stage5.render_references(run="c", slot="c", model="c", raw_dir=raw,
                        cameras_path=work / "cameras.json", out_dir=work / "refs")
rgb = np.asarray(Image.open(work / "refs" / "rgb" / "cam00000_+z.png").convert("RGB"), np.float32) / 255.0

# --- Stage 2 sampling on the same GLB ---
scene = trimesh.load(raw / "quad.glb", process=False)
geom = next(iter(scene.geometry.values())) if hasattr(scene, "geometry") else scene

def face_of(P, geom):
    """Index of the triangle containing P (projected to XY; the quad is planar)."""
    tris = np.asarray(geom.triangles)  # (F,3,3)
    for i, t in enumerate(tris):
        a, b, c = t[:, :2]
        v0, v1, v2 = b - a, c - a, P[:2] - a
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        d20, d21 = v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01
        u = (d11 * d20 - d01 * d21) / den
        v = (d00 * d21 - d01 * d20) / den
        if u >= -1e-4 and v >= -1e-4 and u + v <= 1 + 1e-4:
            return i
    return 0

# Two probe points: upper (+Y) and lower (-Y) on the quad.
probes = {"upper (+Y)": np.array([0.0, 0.9, Z0]), "lower (-Y)": np.array([0.0, -0.9, Z0])}
K = stage5.intrinsics_matrix(R, 90.0)
print(f"{'probe':12} {'stage2 (trimesh)':22} {'stage5 pixel':22} agree?")
all_agree = True
for name, P in probes.items():
    f = face_of(P, geom)
    s2 = stage2.surfel_colors(geom, P[None], np.array([f]))[0, :3]
    # stage5: project P (OpenCV) -> pixel; world X=-x_cam, Y=-y_cam
    u = K[0, 0] * (-P[0]) / Z0 + K[0, 2]
    v = K[1, 1] * (-P[1]) / Z0 + K[1, 2]
    s5 = rgb[int(round(v)), int(round(u))]
    s2c = "red" if s2[0] > s2[2] else "blue"
    s5c = "red" if s5[0] > s5[2] else "blue"
    agree = s2c == s5c
    all_agree &= agree
    print(f"{name:12} {str(s2.round(2)):22} {str(s5.round(2)):22} {agree}  ({s2c} vs {s5c})")

print("\n" + ("Stage 2 and Stage 5 AGREE on texture orientation."
              if all_agree else
              "MISMATCH: Stage 5 texture-V is inverted relative to Stage 2 (the surfel init)."))
