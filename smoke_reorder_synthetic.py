"""Synthetic smoke test for the REORDERED splat pipeline (Option A), no GPU.

Controlled geometry so every assertion has a known answer:
  * a sealed BOX (0.9 m) authored with INWARD normals (inverted winding), placed in
    open space, plus a thin internal DIVIDER inside its cavity.

Validates the new inter-stage contracts:
  Stage 2 (free-space FIRST): dual-res grid + reachability separates the exterior
      (navigable) from the box's tiny interior hollow (dropped).
  Stage 3 (consumes the grid): normals flipped OUTWARD toward reachable free space
      despite inward authoring; the buried divider is hidden-face-culled; the cloud
      is a 2DGS .ply (scale_0/1, no scale_2); 3dgs mode adds scale_2.
  Stage 4 (consumes grid + cloud, NO meshes): errors without either input; otherwise
      emits a Stage-5-consumable cameras.json + patches.bin + patch_views.json.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splat import stage2, stage3, stage4, stage5

FAILS: list[str] = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def ply_scale_props(path: Path) -> list[str]:
    raw = path.read_bytes()
    header = raw[: raw.find(b"end_header")].decode("ascii", "replace")
    return [ln.split()[-1] for ln in header.splitlines() if ln.startswith("property float scale_")]


work = Path(tempfile.mkdtemp(prefix="reorder_syn_"))
raw = work / "objects"; raw.mkdir(parents=True)
out = work / "splat"; out.mkdir(parents=True)

BOX_C = np.array([1.2, 1.2, 1.2])
box = trimesh.creation.box(extents=[0.9, 0.9, 0.9])
box.apply_translation(BOX_C)
box.invert()  # author INWARD normals — Stage 3 must override this via the grid
box.export(raw / "box.glb")
divider = trimesh.creation.box(extents=[0.7, 0.7, 0.02])
divider.apply_translation(BOX_C)
divider.export(raw / "divider.glb")

# ---- Stage 2: free space FIRST -------------------------------------------------
params2 = stage2.FreeSpaceParams(pitch=0.12, refine=3, margin=1.5, reachable_min_volume=2.0)
s2 = stage2.compute_free_space(run="s", slot="s", model="s", raw_dir=raw,
                               out_path=out / stage2.FREESPACE_NAME, params=params2)
fs = stage2.load_free_space(out / stage2.FREESPACE_NAME)
print(f"stage2: dims_fine={s2['dims_fine']} free={s2['free_voxels']} reachable={s2['reachable_voxels']}")

ext = np.array([[2.8, 1.2, 1.2]])       # clearly outside the box, inside grid+margin
check("exterior is reachable free space", bool(fs.reachable_free(ext)[0]))
check("box interior hollow is NOT reachable", not bool(fs.reachable_free(BOX_C[None])[0]))
# Probe surface samples from the RELOADED float32 GLB (the exact geometry Stage 2
# voxelized) so binning precision matches the grid; a single point can also miss a
# ~2% pinhole cell, so assert on a fraction.
reloaded = trimesh.load(raw / "box.glb", process=False)
rgeom = next(iter(reloaded.geometry.values())) if hasattr(reloaded, "geometry") else reloaded
wall_pts, _ = trimesh.sample.sample_surface(rgeom, 2000)
occ_frac = float(fs.fine_occupied(np.asarray(wall_pts)).mean())
check("box walls are fine-occupied (>=95% of surface samples)", occ_frac >= 0.95,
      f"{occ_frac*100:.1f}% occupied")
cands, clv = fs.free_candidates(0.25)
inside_box = np.all(np.abs(cands - BOX_C) < 0.45, axis=1)
check("free candidates exist + none inside the box",
      len(cands) > 0 and not inside_box.any(), f"{len(cands)} cands, {int(inside_box.sum())} inside")
check("reachability dropped the interior hollow (free > reachable)",
      s2["reachable_voxels"] < s2["free_voxels"])

# ---- Stage 3: orient to free space + cull hidden + 2DGS format ------------------
s3 = stage3.sample_cell(run="s", slot="s", model="s", raw_dir=raw,
                        freespace_path=out / stage2.FREESPACE_NAME,
                        out_path=out / stage3.CLOUD_NAME,
                        params=stage3.SampleParams(target_splats=40_000, representation="2dgs"))
print(f"stage3: sampled={s3['sampled']} kept={s3['splats']} culled_hidden={s3['culled_hidden']}")
scales = ply_scale_props(out / stage3.CLOUD_NAME)
check("2DGS cloud has exactly scale_0, scale_1 (no thickness)", scales == ["scale_0", "scale_1"], str(scales))

pos, nrm, _ = stage4._read_cloud(out / stage3.CLOUD_NAME)
# after culling only box walls remain; their normals must point OUTWARD from center
outward = np.einsum("ij,ij->i", nrm, pos - BOX_C)
frac_out = float((outward > 0).mean())
check("normals flipped OUTWARD toward free space (despite inward authoring)",
      frac_out > 0.9, f"{frac_out*100:.1f}% outward")
check("buried divider was hidden-face culled", s3["culled_hidden"] > 0, f"culled={s3['culled_hidden']}")
# divider sits on the z=BOX_C plane inside the box; no kept surfel should remain there
near_div = np.abs(pos[:, 2] - BOX_C[2]) < 0.03
inside_xy = np.all(np.abs(pos[:, :2] - BOX_C[:2]) < 0.3, axis=1)
check("no surviving surfels on the culled divider plane", int((near_div & inside_xy).sum()) == 0,
      f"{int((near_div & inside_xy).sum())} left")

# 3dgs mode appends the thin third scale
stage3.sample_cell(run="s", slot="s", model="s", raw_dir=raw,
                   freespace_path=out / stage2.FREESPACE_NAME, out_path=out / "cloud_3dgs.ply",
                   params=stage3.SampleParams(target_splats=40_000, representation="3dgs"))
check("3dgs mode appends scale_2", ply_scale_props(out / "cloud_3dgs.ply") == ["scale_0", "scale_1", "scale_2"])

# ---- Stage 4: no-mesh contract -------------------------------------------------
raised_fs = raised_cloud = False
try:
    stage4.plan_cameras(run="s", slot="s", model="s", freespace_path=out / "nope.npz",
                        surfels_path=out / stage3.CLOUD_NAME, out_path=out / stage4.CAMERAS_NAME)
except FileNotFoundError:
    raised_fs = True
try:
    stage4.plan_cameras(run="s", slot="s", model="s", freespace_path=out / stage2.FREESPACE_NAME,
                        surfels_path=out / "nope.ply", out_path=out / stage4.CAMERAS_NAME)
except FileNotFoundError:
    raised_cloud = True
check("Stage 4 errors without the Stage-2 grid", raised_fs)
check("Stage 4 errors without the Stage-3 cloud", raised_cloud)

s4 = stage4.plan_cameras(run="s", slot="s", model="s",
                         freespace_path=out / stage2.FREESPACE_NAME,
                         surfels_path=out / stage3.CLOUD_NAME, out_path=out / stage4.CAMERAS_NAME,
                         params=stage4.PlanParams(max_candidates=800))
print(f"stage4: patches={s4['patches']} candidates={s4['candidates']} cameras={s4['cameras']}")
check("Stage 4 wrote patch_views.json", (out / stage4.PATCH_VIEWS_NAME).is_file())
plan = stage5.load_camera_plan(out / stage4.CAMERAS_NAME)
views = stage5.enumerate_views(plan)
check("cameras.json is Stage-5-consumable", len(plan["cameras"]) > 0 and len(views) > 0,
      f"{len(plan['cameras'])} cams, {len(views)} views")

print("\n" + ("ALL SYNTHETIC REORDER TESTS PASSED" if not FAILS else f"FAILURES: {FAILS}"))
print(f"artifacts: {work}")
sys.exit(1 if FAILS else 0)
