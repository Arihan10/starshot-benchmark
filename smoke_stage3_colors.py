"""Stage-3 footprint-averaged color smoke test (no GPU).

Surfel colors are now the AVERAGE of the base-color texture over each disk's
footprint (mip pyramid sampled at the level matching the disk's texel
diameter), instead of a single pinprick texel at the disk centre. On a busy
texture, point samples make neighbouring disks wildly different colors
(confetti noise); footprint averages make each disk the local mean.

Checks:
  * level-0 (tiny radius) path is texel-identical to trimesh's `uv_to_color`
    — the old behavior is exactly preserved as the fine limit;
  * the mip level matches the closed-form footprint math (2·r·texel-density);
  * a checkerboard whose period divides the filter width averages to exactly
    0.5 (both color and ALPHA channels), with per-surfel variance collapsing
    versus the bimodal point-sample distribution;
  * `baseColorFactor` still multiplies after averaging;
  * untextured fallback (factor / grey) unchanged; degenerate UVs fall back
    to level 0 without error;
  * end-to-end `sample_cell` on a checker-textured scene writes a cloud whose
    quad surfels all sit near the texture mean — no saturated outliers.

Run: ./splat/.venv/bin/python smoke_stage3_colors.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splat import stage2, stage3, stage4  # noqa: E402
from splat.stage3 import _build_mips, _footprint_levels, surfel_colors  # noqa: E402

trimesh.util.log.setLevel("ERROR")

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def checker_image(size: int, square: int, alpha_checker: bool = False) -> Image.Image:
    """Black/white checkerboard, `square` px per square. With `alpha_checker`,
    RGB is constant and the CHECKER lives in the alpha channel instead."""
    yy, xx = np.mgrid[0:size, 0:size]
    board = (((xx // square) + (yy // square)) % 2 == 0)
    img = np.zeros((size, size, 4), dtype=np.uint8)
    if alpha_checker:
        img[..., :3] = 200
        img[..., 3] = np.where(board, 255, 0)
    else:
        img[..., :3] = np.where(board[..., None], 255, 0)
        img[..., 3] = 255
    return Image.fromarray(img, "RGBA")


def textured_quad(image: Image.Image | None, factor=None, side: float = 2.0,
                  standing_at_z: float | None = None) -> trimesh.Trimesh:
    """2-triangle quad of edge `side` with UV [0,1]²; flat in XY, or standing
    vertically at z=`standing_at_z`."""
    s = side
    if standing_at_z is None:
        verts = np.array([[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0]], float)
    else:
        z = standing_at_z
        verts = np.array([[0.5, 0, z], [0.5 + s, 0, z], [0.5 + s, s, z], [0.5, s, z]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=image, baseColorFactor=factor, alphaMode="OPAQUE"
    )
    vis = trimesh.visual.TextureVisuals(uv=uv if image is not None else None, material=mat)
    return trimesh.Trimesh(verts, faces, visual=vis, process=False)


def surface_points(geom: trimesh.Trimesh, n_per_face: int = 60, seed: int = 3):
    """Deterministic interior points on each face + their face indices."""
    rng = np.random.default_rng(seed)
    pts, fid = [], []
    tris = np.asarray(geom.triangles)
    for f in range(len(tris)):
        a = rng.uniform(0.05, 0.9, n_per_face)
        b = rng.uniform(0.05, 0.9, n_per_face)
        swap = a + b > 0.95
        a[swap], b[swap] = (0.95 - a)[swap], (0.95 - b)[swap]
        w = np.stack([1 - a - b, a, b], axis=1)
        pts.append(w @ tris[f])
        fid.append(np.full(n_per_face, f))
    return np.concatenate(pts), np.concatenate(fid)


# 64² texture with 8-px squares on a 2 m quad → 32 texels/m linear density.
DENS = 32.0

section("level-0 limit == old point sample (texel-identical to uv_to_color)")
quad = textured_quad(checker_image(64, 8))
pts, fid = surface_points(quad)
tiny = np.full(len(pts), 1e-5, dtype=np.float32)
got = surfel_colors(quad, pts, fid, radii=tiny)
tris = np.asarray(quad.triangles)[fid]
bary = trimesh.triangles.points_to_barycentric(tris, pts)
uvs = np.einsum("sj,sjk->sk", bary, np.asarray(quad.visual.uv)[np.asarray(quad.faces)[fid]])
want = np.asarray(trimesh.visual.color.uv_to_color(uvs, quad.visual.material.baseColorTexture),
                  np.float32) / 255.0
check("tiny-radius colors == uv_to_color exactly",
      np.allclose(got, want, atol=1e-6), f"max |Δ| = {np.abs(got - want).max():.2e}")
check("point samples are bimodal (the old confetti noise)",
      float(got[:, 0].std()) > 0.4, f"std = {got[:, 0].std():.3f}")

section("footprint math: mip level = round(log2(2·r·texel_density))")
for r, want_lvl in ((0.125, 3), (0.5, 5), (1e-5, 0)):
    lvl = _footprint_levels(quad, fid, np.full(len(pts), r), (64, 64),
                            n_levels=len(_build_mips(quad.visual.material.baseColorTexture)))
    check(f"radius {r} m → level {want_lvl}",
          np.all(lvl == want_lvl), f"levels seen: {sorted(set(lvl.tolist()))}")

section("big footprint averages the checker to 0.5 (color AND alpha)")
big = np.full(len(pts), 0.5, dtype=np.float32)   # diameter 32 texels = 4×4 squares
got = surfel_colors(quad, pts, fid, radii=big)
check("colors ≈ 0.5 with tiny spread",
      abs(float(got[:, :3].mean()) - 0.5) < 0.02 and float(got[:, 0].std()) < 0.02,
      f"mean = {got[:, :3].mean():.4f} std = {got[:, 0].std():.4f}")
quad_a = textured_quad(checker_image(64, 8, alpha_checker=True))
got_a = surfel_colors(quad_a, pts, fid, radii=big)
check("alpha averages identically (≈ 0.5)",
      abs(float(got_a[:, 3].mean()) - 0.5) < 0.02 and float(got_a[:, 3].std()) < 0.02,
      f"mean = {got_a[:, 3].mean():.4f}")

section("baseColorFactor still multiplies after averaging; fallbacks intact")
quad_f = textured_quad(checker_image(64, 8), factor=[0.5, 0.5, 0.5, 1.0])
got_f = surfel_colors(quad_f, pts, fid, radii=big)
check("factor × averaged color (≈ 0.25)",
      abs(float(got_f[:, :3].mean()) - 0.25) < 0.02, f"mean = {got_f[:, :3].mean():.4f}")
quad_nt = textured_quad(None, factor=[0.2, 0.4, 0.6, 1.0])
got_nt = surfel_colors(quad_nt, pts, fid, radii=big)
check("untextured factor fallback unchanged",
      np.allclose(got_nt[:, :3].mean(axis=0), [0.2, 0.4, 0.6], atol=0.02),
      f"{got_nt[:, :3].mean(axis=0).round(3)}")
quad_bad = textured_quad(checker_image(64, 8))
quad_bad.visual.uv = np.zeros_like(np.asarray(quad_bad.visual.uv))  # degenerate UVs
got_bad = surfel_colors(quad_bad, pts, fid, radii=big)
check("degenerate UVs: level-0 fallback, no crash", got_bad.shape == (len(pts), 4))

section("end-to-end: sample_cell writes footprint-averaged surfels")
work = Path(tempfile.mkdtemp(prefix="stage3_colors_"))
raw = work / "objects"
out = work / "splat"
raw.mkdir(parents=True)
ground = trimesh.creation.box(extents=[4.2, 0.1, 4.2])
ground.apply_translation([2, -0.05, 2])
ground.export(raw / "o000.glb")
# 256² checker with 2-px squares on the 2 m quad: every plausible surfel
# footprint (levels ≥ 2 at default density) spans whole checker periods, so a
# correct average is ≈ 0.5 for EVERY quad surfel regardless of position.
textured_quad(checker_image(256, 2), standing_at_z=2.0).export(raw / "o001.glb")
stage2.compute_free_space(run="t", slot="t", model="t", raw_dir=raw,
                          out_path=out / stage2.FREESPACE_NAME)
# feature_boost=1: this test isolates COLOR averaging at a known footprint.
# With adaptive density on, the quad's boundary band samples ~4× finer and those
# small disks legitimately resolve the 2-px checker (level-0 lookups) — correct
# behavior, but a different property than the one asserted here.
s3 = stage3.sample_cell(run="t", slot="t", model="t", raw_dir=raw,
                        freespace_path=out / stage2.FREESPACE_NAME,
                        out_path=out / stage3.CLOUD_NAME,
                        params=stage3.SampleParams(feature_boost=1.0))
pos, _nrm, col = stage4._read_cloud(out / stage3.CLOUD_NAME)
on_quad = (np.abs(pos[:, 2] - 2.0) < 0.05) & (pos[:, 0] > 0.6) & (pos[:, 0] < 2.4) \
          & (pos[:, 1] > 0.1) & (pos[:, 1] < 1.9)
qc = col[on_quad][:, 0]
print(f"  surfels={s3['splats']:,} on_quad={int(on_quad.sum())} "
      f"| quad color mean={qc.mean():.3f} std={qc.std():.3f} "
      f"min={qc.min():.3f} max={qc.max():.3f}")
check("quad surfels exist", int(on_quad.sum()) > 100, f"{int(on_quad.sum())}")
check("every quad surfel near the texture mean (no confetti outliers)",
      abs(float(qc.mean()) - 0.5) < 0.05 and float(qc.std()) < 0.06
      and float(qc.min()) > 0.3 and float(qc.max()) < 0.7,
      f"mean={qc.mean():.3f} std={qc.std():.3f} range=[{qc.min():.3f},{qc.max():.3f}]")

print("\n" + ("ALL STAGE-3 COLOR TESTS PASSED" if not FAILS else f"FAILURES ({len(FAILS)}): {FAILS}"))
print(f"artifacts: {work}")
sys.exit(1 if FAILS else 0)
