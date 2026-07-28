"""Feature-adaptive density smoke test (no GPU): detail measured from the MESH
(feature edges: creases + boundaries) drives finer sampling bands in Stage 3,
and Stage 4 consumes the resulting density instead of re-detecting detail.

What must hold:
  * thin members (a rail) sample several× finer than flats — the budget finally
    goes where the detail is;
  * flat interiors are UNCHANGED (adaptivity is additive; the density knob keeps
    its meaning on flats);
  * crease neighbourhoods on a plain box are denser than face centres;
  * smooth feature-less meshes (sphere) take the identical single-band path;
  * texture UVs survive the banding (band meshes carry the original material);
  * `feature_boost=1` restores uniform sampling exactly (the off switch);
  * Stage 4's patches inherit the profile (uniform thinning) and rail patches
    carry a smaller feature scale → closer required views — with the detector
    (curvature_k / tex_k / patch_min_spacing) deleted.

Run: ./splat/.venv/bin/python smoke_stage3_adaptive.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splat import stage2, stage3, stage4  # noqa: E402
from splat.stage3 import _feature_edge_points  # noqa: E402

trimesh.util.log.setLevel("ERROR")

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def _box(center, extents) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=np.asarray(extents, dtype=float))
    m.apply_translation(np.asarray(center, dtype=float))
    return m


def _export(meshes, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(meshes):
        m.export(d / f"o{i:03d}.glb")


def run_cell(raw: Path, out: Path, boost: float, base_spacing: float = 0.08):
    stage2.compute_free_space(run="t", slot="t", model="t", raw_dir=raw,
                              out_path=out / stage2.FREESPACE_NAME)
    s3 = stage3.sample_cell(
        run="t", slot="t", model="t", raw_dir=raw,
        freespace_path=out / stage2.FREESPACE_NAME,
        out_path=out / stage3.CLOUD_NAME,
        params=stage3.SampleParams(base_spacing=base_spacing, feature_boost=boost,
                                   cull_hidden=False),
    )
    pos, nrm, col = stage4._read_cloud(out / stage3.CLOUD_NAME)
    return s3, pos, col


def med_nn(points: np.ndarray) -> float:
    if len(points) < 2:
        return float("nan")
    return float(np.median(cKDTree(points).query(points, k=2)[0][:, 1]))


work = Path(tempfile.mkdtemp(prefix="stage3_adaptive_"))
print(f"workdir: {work}")

# =====================================================================================
section("rail + floor: budget flows to the thin member, flats unchanged")
# =====================================================================================
rawA = work / "rail" / "objects"
_export([_box([0, 0, 0], [6.0, 0.1, 6.0]),        # floor
         _box([0, 0.8, 0], [2.0, 0.03, 0.03])],   # rail, floating
        rawA)
sA1, posA1, _ = run_cell(rawA, work / "rail" / "a1", boost=1.0)
sA4, posA4, _ = run_cell(rawA, work / "rail" / "a4", boost=4.0)


def rail_mask(p):
    return (np.abs(p[:, 0]) < 1.05) & (np.abs(p[:, 1] - 0.8) < 0.08) & (np.abs(p[:, 2]) < 0.08)


def floor_center_mask(p):
    return (np.abs(p[:, 0]) < 1.5) & (np.abs(p[:, 2]) < 1.5) & (np.abs(p[:, 1] - 0.05) < 0.02)


r1, r4 = posA1[rail_mask(posA1)], posA4[rail_mask(posA4)]
f1, f4 = posA1[floor_center_mask(posA1)], posA4[floor_center_mask(posA4)]
print(f"  boost=1: total={len(posA1):,} rail={len(r1)} (nn {med_nn(r1)*100:.1f}cm) "
      f"floor-centre nn {med_nn(f1)*100:.1f}cm")
print(f"  boost=4: total={len(posA4):,} rail={len(r4)} (nn {med_nn(r4)*100:.1f}cm) "
      f"floor-centre nn {med_nn(f4)*100:.1f}cm")
check("rail samples ≥ 3× finer count", len(r4) >= 3 * max(len(r1), 1),
      f"{len(r4)} vs {len(r1)}")
check("rail spacing ≤ 0.5× uniform", med_nn(r4) <= 0.5 * med_nn(r1),
      f"{med_nn(r4)*100:.1f} vs {med_nn(r1)*100:.1f} cm")
check("flat interior spacing unchanged (±25%)",
      abs(med_nn(f4) - med_nn(f1)) <= 0.25 * med_nn(f1),
      f"{med_nn(f4)*100:.1f} vs {med_nn(f1)*100:.1f} cm")
check("total grows additively, bounded",
      len(posA4) > len(posA1) and len(posA4) < 8 * len(posA1),
      f"{len(posA4):,} vs {len(posA1):,}")

# =====================================================================================
section("plain box: crease bands denser than face centres")
# =====================================================================================
rawB = work / "box" / "objects"
_export([_box([0, 1.0, 0], [2.0, 2.0, 2.0])], rawB)
_, posB1, _ = run_cell(rawB, work / "box" / "a1", boost=1.0)
_, posB4, _ = run_cell(rawB, work / "box" / "a4", boost=4.0)


def near_edge_frac(p):
    """Fraction of surfels within 0.12 m of ≥2 of the box's face planes (i.e.
    near an edge line of the 2×2×2 box centred at (0,1,0))."""
    q = p - np.array([0.0, 1.0, 0.0])
    near = (np.abs(np.abs(q) - 1.0) < 0.12).sum(axis=1)
    return float((near >= 2).mean())


print(f"  near-edge fraction: boost=1 {near_edge_frac(posB1):.3f} | "
      f"boost=4 {near_edge_frac(posB4):.3f}")
check("crease neighbourhoods dominate the added samples",
      near_edge_frac(posB4) >= 2.0 * near_edge_frac(posB1),
      f"{near_edge_frac(posB4):.3f} vs {near_edge_frac(posB1):.3f}")

# =====================================================================================
section("smooth sphere: no feature edges → identical single-band path")
# =====================================================================================
sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
sphere.apply_translation([0, 1.0, 0])
check("sphere has no feature edges (detector returns None)",
      _feature_edge_points(sphere, step=0.04) is None)
rawC = work / "sphere" / "objects"
_export([sphere], rawC)
_, posC1, _ = run_cell(rawC, work / "sphere" / "a1", boost=1.0, base_spacing=0.04)
_, posC4, _ = run_cell(rawC, work / "sphere" / "a4", boost=4.0, base_spacing=0.04)
check("sphere counts match across boosts (RNG-level only)",
      abs(len(posC4) - len(posC1)) <= 0.10 * len(posC1),
      f"{len(posC4)} vs {len(posC1)}")

# =====================================================================================
section("textured quad: UVs survive banding (colors sane, not grey fallback)")
# =====================================================================================
tex = np.zeros((64, 64, 4), dtype=np.uint8)
yy, xx = np.mgrid[0:64, 0:64]
board = (((xx // 4) + (yy // 4)) % 2 == 0)
tex[..., :3] = np.where(board[..., None], 255, 0)
tex[..., 3] = 255
verts = np.array([[0.5, 0.0, 2.0], [2.5, 0.0, 2.0], [2.5, 2.0, 2.0], [0.5, 2.0, 2.0]], float)
quad = trimesh.Trimesh(
    verts, np.array([[0, 1, 2], [0, 2, 3]]),
    visual=trimesh.visual.TextureVisuals(
        uv=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(tex, "RGBA"), alphaMode="OPAQUE"),
    ),
    process=False,
)
rawD = work / "quad" / "objects"
rawD.mkdir(parents=True)
_box([1.5, -0.05, 1.5], [4.0, 0.1, 4.0]).export(rawD / "o000.glb")
quad.export(rawD / "o001.glb")
_, posD, colD = run_cell(rawD, work / "quad" / "a4", boost=4.0)
on_quad = (np.abs(posD[:, 2] - 2.0) < 0.05) & (posD[:, 0] > 0.6) & (posD[:, 0] < 2.4) \
          & (posD[:, 1] > 0.1) & (posD[:, 1] < 1.9)
qc = colD[on_quad]
print(f"  quad surfels={int(on_quad.sum())} color mean={qc.mean():.3f} std={qc[:, 0].std():.3f}")
check("quad colors track the checker (mean ≈ 0.5, not the 0.6 grey fallback)",
      int(on_quad.sum()) > 100 and abs(float(qc.mean()) - 0.5) < 0.06,
      f"mean={qc.mean():.3f}")

# =====================================================================================
section("stage 4 consumes the density (uniform patches, spacing-driven view dist)")
# =====================================================================================
s4 = stage4.plan_cameras(run="t", slot="t", model="t",
                         freespace_path=work / "rail" / "a4" / stage2.FREESPACE_NAME,
                         surfels_path=work / "rail" / "a4" / stage3.CLOUD_NAME,
                         out_path=work / "rail" / "a4" / stage4.CAMERAS_NAME,
                         params=stage4.PlanParams(max_candidates=50_000))
patches = np.frombuffer((work / "rail" / "a4" / stage4.PATCHES_NAME).read_bytes(),
                        dtype="<f4").reshape(-1, 8)
ppos, pfeat = patches[:, :3], patches[:, 6]
pr, pf = rail_mask(ppos), floor_center_mask(ppos)
n_cloud = len(posA4)
print(f"  patches={len(patches):,} (cloud {n_cloud:,}) | cams={s4['cameras']} "
      f"coverage={s4['coverage']['satisfied_pct']}% "
      f"(near {s4['coverage']['has_near']}, headon {s4['coverage']['has_headon']}) | "
      f"refined cands={s4['candidates_refined']:,}/{s4['candidates']:,} | "
      f"rail feat {pfeat[pr].mean()*100:.1f}cm vs floor feat {pfeat[pf].mean()*100:.1f}cm")
check("patches ≈ patch_fraction × surfels (uniform thinning)",
      abs(len(patches) - 0.5 * n_cloud) <= 0.15 * n_cloud,
      f"{len(patches):,} vs {n_cloud//2:,}")
check("rail patches exist and carry a FINER feature scale (≤ 0.5× floor's)",
      pr.any() and pf.any() and pfeat[pr].mean() <= 0.5 * pfeat[pf].mean(),
      f"{pfeat[pr].mean()*100:.1f} vs {pfeat[pf].mean()*100:.1f} cm")
check("plan completes with sane coverage", s4["coverage"]["satisfied_pct"] > 50.0,
      f"{s4['coverage']['satisfied_pct']}%")

# =====================================================================================
section("stage 4 supply follows demand (refined candidates) + head-on requirement")
# =====================================================================================
# Refined tier engages on the adaptive cloud and buys close-up standpoints.
check("refined candidate tier engaged near fine patches",
      s4["candidates_refined"] > 0, f"{s4['candidates_refined']:,}")
cam_pos = np.array([c["pos"] for c in
                    __import__("json").loads(
                        (work / "rail" / "a4" / stage4.CAMERAS_NAME).read_text())["cameras"]])
rail_p = ppos[pr]
if len(cam_pos) and len(rail_p):
    dmin = float(np.sqrt(((rail_p[:, None, :] - cam_pos[None, :, :]) ** 2).sum(-1)).min())
else:
    dmin = float("inf")
check("a chosen camera stands within the rail's near-shell (≤ 0.55 m)",
      dmin <= 0.55, f"closest camera {dmin:.2f} m")
check("satisfied patches all have near + head-on views",
      s4["coverage"]["satisfied"] <= min(s4["coverage"]["has_near"],
                                         s4["coverage"]["has_headon"]))

# Uniform cloud (boost=1): near demands are lattice-satisfiable → no refined tier.
s4u = stage4.plan_cameras(run="t", slot="t", model="t",
                          freespace_path=work / "rail" / "a1" / stage2.FREESPACE_NAME,
                          surfels_path=work / "rail" / "a1" / stage3.CLOUD_NAME,
                          out_path=work / "rail" / "a1" / stage4.CAMERAS_NAME,
                          params=stage4.PlanParams(max_candidates=50_000))
check("uniform cloud: refined tier stays OFF (back-compat path)",
      s4u["candidates_refined"] == 0, f"{s4u['candidates_refined']}")

# Negative control: a near-impossible head-on threshold must collapse satisfaction
# (proves the requirement actually binds rather than being vacuously true).
s4h = stage4.plan_cameras(run="t", slot="t", model="t",
                          freespace_path=work / "rail" / "a4" / stage2.FREESPACE_NAME,
                          surfels_path=work / "rail" / "a4" / stage3.CLOUD_NAME,
                          out_path=work / "rail" / "a4" / "cameras_h.json",
                          params=stage4.PlanParams(max_candidates=50_000,
                                                   headon_cos=0.999))
print(f"  headon_cos=0.999 control: satisfied {s4h['coverage']['satisfied_pct']}% "
      f"vs {s4['coverage']['satisfied_pct']}%")
check("head-on requirement binds (impossible threshold collapses satisfaction)",
      s4h["coverage"]["satisfied_pct"] < 0.5 * max(s4["coverage"]["satisfied_pct"], 1e-6),
      f"{s4h['coverage']['satisfied_pct']}% vs {s4['coverage']['satisfied_pct']}%")

print("\n" + ("ALL ADAPTIVE-DENSITY TESTS PASSED" if not FAILS
              else f"FAILURES ({len(FAILS)}): {FAILS}"))
print(f"artifacts: {work}")
sys.exit(1 if FAILS else 0)
