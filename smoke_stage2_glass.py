"""Glass-occupancy smoke test (no GPU): transparent surfaces occupy space but
don't block sight.

Stage 2 now classifies BLEND/MASK-material surfaces by sampled base-color alpha:
cells whose surface is transmissive (alpha below the occlusion cutoff — glass.py
panes carry ~0.065; cutout gaps carry 0) land in a GLASS class that still counts
for clearance / navigation / reachability (a camera can't sit inside a pane, the
flood fill doesn't walk through a closed window) but is EXCLUDED from the
occlusion set Stage 4's line-of-sight ray-march tests — so surfaces behind
glazing are finally coverable.

Scene types exercised:
  window-room     opaque room; one wall holds a textured BLEND pane whose ONE
                  mesh carries opaque frame texels + alpha-0.065 pane texels
                  (exactly the glass.py output shape)
  opaque-control  identical geometry, pane material OPAQUE — texture alpha must
                  be IGNORED (glTF semantics; matches Stage 3/5), rays blocked
  glass-case      factor-alpha BLEND display case (no texture) over a trophy
  mask-stripes    MASK quad: left half cut out (alpha 0), right half solid
  glass-room      fully sealed all-glass room (enclosure + sight-through)

Checks: occupied-vs-occluding split per cell, per-texel frame/pane
classification inside one mesh, Stage-4 `_visible` rays through panes / walls /
frames / cutouts on real Stage-2 grids, reachability with glass barriers, the
Stage 2→3→4 chain on glass scenes (pane keeps its own surfels; interior surfels
are sightable from exterior cameras through the window and NOT in the opaque
control), and legacy `.npz` files (no glass field) degrading to the old
block-everything behavior.

Run: ./splat/.venv/bin/python smoke_stage2_glass.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splat import stage2, stage3, stage4  # noqa: E402

trimesh.util.log.setLevel("ERROR")

FAILS: list[str] = []
PF = stage2.DEFAULT_PITCH / stage2.DEFAULT_REFINE  # 0.04 m fine pitch


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# --- builders -------------------------------------------------------------------

def _box(center, extents) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=np.asarray(extents, dtype=float))
    m.apply_translation(np.asarray(center, dtype=float))
    return m


def _pbr_quad(corner, u_vec, v_vec, *, image=None, factor=None,
              alpha_mode="BLEND", cutoff=None) -> trimesh.Trimesh:
    """Two-triangle quad with a PBR material (textured or factor-only)."""
    c = np.asarray(corner, float)
    u, v = np.asarray(u_vec, float), np.asarray(v_vec, float)
    verts = np.array([c, c + u, c + u + v, c + v])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=image, baseColorFactor=factor,
        alphaMode=alpha_mode, alphaCutoff=cutoff,
    )
    vis = trimesh.visual.TextureVisuals(uv=uv if image is not None else None, material=mat)
    return trimesh.Trimesh(verts, faces, visual=vis, process=False)


def _glass_box(center, extents) -> trimesh.Trimesh:
    """Factor-alpha glass box (no texture) — the `glassify` route shape."""
    m = _box(center, extents)
    mat = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 0.065], alphaMode="BLEND"
    )
    m.visual = trimesh.visual.TextureVisuals(material=mat)
    return m


def window_texture() -> Image.Image:
    """64² RGBA: 8-px opaque border (frame, alpha 255) around a pane at the
    glass.py alpha (17/255 ≈ 0.067). Border is symmetric so the test is immune
    to any UV V-flip convention."""
    a = np.full((64, 64, 4), 200, dtype=np.uint8)
    a[..., 3] = 255
    a[8:56, 8:56, 3] = 17
    return Image.fromarray(a, "RGBA")


def stripes_texture() -> Image.Image:
    """64² RGBA: left half (u < 0.5) alpha 0 (cut out), right half alpha 255.
    U-dependent only, so immune to V-flip conventions."""
    a = np.full((64, 64, 4), 200, dtype=np.uint8)
    a[:, :32, 3] = 0
    a[:, 32:, 3] = 255
    return Image.fromarray(a, "RGBA")


def _export(meshes: list[trimesh.Trimesh], d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(meshes):
        m.export(d / f"o{i:03d}.glb")


# Room: interior 4 × 2.5 × 4, walls 0.1 thick; window opening x∈[1.4,2.6],
# y∈[0.9,1.9] in the front (z ≈ 4.05) wall; furniture box inside.
def build_window_room(d: Path, pane_alpha_mode: str) -> None:
    meshes = [
        _box([2, -0.05, 2], [4.2, 0.1, 4.2]),    # floor
        _box([2, 2.55, 2], [4.2, 0.1, 4.2]),     # ceiling
        _box([2, 1.25, -0.05], [4.2, 2.5, 0.1]), # back wall
        _box([-0.05, 1.25, 2], [0.1, 2.5, 4.2]), # left wall
        _box([4.05, 1.25, 2], [0.1, 2.5, 4.2]),  # right wall
        _box([2, 0.45, 4.05], [4.2, 0.9, 0.1]),  # front below window
        _box([2, 2.2, 4.05], [4.2, 0.6, 0.1]),   # front above window
        _box([0.65, 1.4, 4.05], [1.5, 1.0, 0.1]),  # front left of window
        _box([3.35, 1.4, 4.05], [1.5, 1.0, 0.1]),  # front right of window
        _pbr_quad([1.4, 0.9, 4.05], [1.2, 0, 0], [0, 1.0, 0],
                  image=window_texture(), alpha_mode=pane_alpha_mode),
        _box([2, 0.5, 1.5], [1.0, 1.0, 1.0]),    # furniture
    ]
    _export(meshes, d)


def build_glass_case(d: Path) -> None:
    meshes = [
        _box([3, -0.05, 3], [6.0, 0.1, 6.0]),    # ground
        _box([3, 0.4, 3], [1.2, 0.8, 1.2]),      # table (top at y=0.8)
        _box([3, 0.95, 3], [0.2, 0.3, 0.2]),     # trophy (y 0.8..1.1)
        _glass_box([3, 1.0, 3], [0.5, 0.6, 0.5]),  # sealed glass case
    ]
    _export(meshes, d)


def build_mask_stripes(d: Path) -> None:
    meshes = [
        _box([2, -0.05, 2], [4.2, 0.1, 4.2]),    # ground
        _box([2, 0.8, 0.8], [3.4, 1.6, 0.1]),    # target wall behind the quad
        _pbr_quad([1.0, 0.0, 2.02], [2.0, 0, 0], [0, 1.6, 0],
                  image=stripes_texture(), alpha_mode="MASK", cutoff=0.5),
    ]
    _export(meshes, d)


def build_glass_room(d: Path) -> None:
    meshes = [
        _box([3, -0.05, 3], [6.2, 0.1, 6.2]),        # opaque floor
        _glass_box([-0.05, 1.0, 3], [0.1, 2.0, 6.2]),  # four glass walls
        _glass_box([6.05, 1.0, 3], [0.1, 2.0, 6.2]),
        _glass_box([3, 1.0, -0.05], [6.2, 2.0, 0.1]),
        _glass_box([3, 1.0, 6.05], [6.2, 2.0, 0.1]),
        _glass_box([3, 2.05, 3], [6.2, 0.1, 6.2]),     # glass ceiling (sealed)
        _box([3, 0.5, 3], [1.0, 1.0, 1.0]),            # opaque furniture
    ]
    _export(meshes, d)


# --- helpers ---------------------------------------------------------------------

def run_stage2(raw: Path, out: Path) -> tuple[dict, stage2.FreeSpace]:
    s = stage2.compute_free_space(run="t", slot="t", model="t", raw_dir=raw,
                                  out_path=out / stage2.FREESPACE_NAME)
    return s, stage2.load_free_space(out / stage2.FREESPACE_NAME)


def ray_visible(fs: stage2.FreeSpace, cam, targets) -> np.ndarray:
    """Stage-4's real line-of-sight test (opaque-only occlusion)."""
    n_steps = int(np.ceil(4.0 / fs.pitch_fine)) + 2
    return stage4._visible(np.asarray(cam, float), np.atleast_2d(np.asarray(targets, float)),
                           fs, n_steps)


def pt(x, y, z) -> np.ndarray:
    return np.array([[x, y, z]], dtype=float)


work = Path(tempfile.mkdtemp(prefix="stage2_glass_"))
print(f"workdir: {work} | fine pitch {PF} m | occlusion cutoff {stage2._OCCLUDING_ALPHA}")

# =====================================================================================
section("material round-trip (GLB preserves alphaMode / texture alpha)")
# =====================================================================================
probe_dir = work / "probe"
build_window_room(probe_dir / "objects", "BLEND")
pane = trimesh.load(probe_dir / "objects" / "o009.glb", process=False)
pg = next(iter(pane.geometry.values()))
mode = str(getattr(pg.visual.material, "alphaMode", None))
tex = getattr(pg.visual.material, "baseColorTexture", None)
alpha_vals = (np.unique(np.asarray(tex.convert("RGBA"))[..., 3]).tolist()
              if tex is not None else [])
check("pane alphaMode survives GLB round-trip", mode == "BLEND", f"mode={mode}")
check("pane texture alpha survives (17 + 255 texels)",
      tex is not None and 17 in alpha_vals and 255 in alpha_vals, f"alphas={alpha_vals}")

# =====================================================================================
section("window-room (textured BLEND pane: frame occludes, pane doesn't)")
# =====================================================================================
rawA = work / "window-room" / "objects"
outA = work / "window-room" / "splat"
build_window_room(rawA, "BLEND")
t0 = time.perf_counter()
sA, fsA = run_stage2(rawA, outA)
print(f"  stage2 {time.perf_counter() - t0:.1f}s | solid={sA['solid_voxels_fine']:,} "
      f"glass={sA['glass_voxels_fine']:,}")
check("glass cells detected", sA["glass_voxels_fine"] > 0)
pane_c = pt(2.0, 1.4, 4.05)     # pane centre (alpha 0.065 texels)
frame_c = pt(2.0, 0.925, 4.05)  # frame band (alpha 255 texels), same mesh
check("pane centre: physically occupied", bool(fsA.fine_occupied(pane_c)[0]))
check("pane centre: does NOT occlude", not bool(fsA.fine_occluding(pane_c)[0]))
check("frame texels of the SAME mesh: occlude", bool(fsA.fine_occluding(frame_c)[0]))
check("interior navigable", bool(fsA.reachable_free(pt(2, 1.4, 2.9))[0]))
check("exterior navigable", bool(fsA.reachable_free(pt(2, 1.4, 4.8))[0]))
vis_pane = ray_visible(fsA, [2.0, 1.4, 5.0], [2.0, 1.4, 2.0])[0]   # through pane
vis_wall = ray_visible(fsA, [0.5, 1.4, 5.0], [0.5, 1.4, 2.0])[0]   # through wall
vis_frame = ray_visible(fsA, [1.45, 0.95, 5.0], [1.45, 0.95, 2.0])[0]  # through frame
check("sight-line THROUGH the pane", bool(vis_pane))
check("sight-line through the wall still blocked", not bool(vis_wall))
check("sight-line through the frame still blocked", not bool(vis_frame))

# =====================================================================================
section("opaque-control (same geometry, pane OPAQUE: texture alpha ignored)")
# =====================================================================================
rawB = work / "opaque-control" / "objects"
outB = work / "opaque-control" / "splat"
build_window_room(rawB, "OPAQUE")
sB, fsB = run_stage2(rawB, outB)
check("no glass cells", sB["glass_voxels_fine"] == 0, f"{sB['glass_voxels_fine']}")
check("pane centre occludes (OPAQUE ignores texture alpha)",
      bool(fsB.fine_occluding(pane_c)[0]))
check("sight-line through the pane blocked",
      not bool(ray_visible(fsB, [2.0, 1.4, 5.0], [2.0, 1.4, 2.0])[0]))

# =====================================================================================
section("glass-case (factor-alpha BLEND, no texture) + mask-stripes (MASK cutout)")
# =====================================================================================
rawC = work / "glass-case" / "objects"
outC = work / "glass-case" / "splat"
build_glass_case(rawC)
sC, fsC = run_stage2(rawC, outC)
case_wall = pt(3.25, 1.0, 3.0)
check("case glass cells detected", sC["glass_voxels_fine"] > 0)
check("case wall: occupied but not occluding",
      bool(fsC.fine_occupied(case_wall)[0]) and not bool(fsC.fine_occluding(case_wall)[0]))
check("trophy visible THROUGH the case",
      bool(ray_visible(fsC, [4.2, 1.0, 3.0], [3.1, 0.95, 3.0])[0]))
check("in-case air still NOT navigable (tiny pocket; camera keeps out)",
      not bool(fsC.reachable_free(pt(2.82, 1.02, 3.06))[0]))

rawD = work / "mask-stripes" / "objects"
outD = work / "mask-stripes" / "splat"
build_mask_stripes(rawD)
sD, fsD = run_stage2(rawD, outD)
check("cutout cells classed as glass", sD["glass_voxels_fine"] > 0)
check("sight-line through the CUT-OUT stripe",
      bool(ray_visible(fsD, [1.5, 0.8, 3.4], [1.5, 0.8, 0.9])[0]))
check("sight-line through the SOLID stripe blocked",
      not bool(ray_visible(fsD, [2.5, 0.8, 3.4], [2.5, 0.8, 0.9])[0]))

# =====================================================================================
section("glass-room (sealed all-glass building)")
# =====================================================================================
rawE = work / "glass-room" / "objects"
outE = work / "glass-room" / "splat"
build_glass_room(rawE)
sE, fsE = run_stage2(rawE, outE)
check("glass cells detected", sE["glass_voxels_fine"] > 0)
check("sealed interior still navigable (big component survives min-volume)",
      bool(fsE.reachable_free(pt(1.0, 1.0, 1.0))[0]))
check("furniture visible from OUTSIDE through the glass wall",
      bool(ray_visible(fsE, [-1.0, 1.0, 3.0], [2.5, 1.0, 3.0])[0]))

# =====================================================================================
section("chain: Stage 2→3→4 on the window-room (and the opaque control)")
# =====================================================================================
for name, raw, out, fs in (("glass", rawA, outA, fsA), ("opaque", rawB, outB, fsB)):
    s3 = stage3.sample_cell(run="t", slot="t", model="t", raw_dir=raw,
                            freespace_path=out / stage2.FREESPACE_NAME,
                            out_path=out / stage3.CLOUD_NAME,
                            params=stage3.SampleParams(target_splats=60_000))
    s4 = stage4.plan_cameras(run="t", slot="t", model="t",
                             freespace_path=out / stage2.FREESPACE_NAME,
                             surfels_path=out / stage3.CLOUD_NAME,
                             out_path=out / stage4.CAMERAS_NAME,
                             params=stage4.PlanParams(max_candidates=4000))
    pos, _nrm, _col = stage4._read_cloud(out / stage3.CLOUD_NAME)
    # Pane keeps its own surfels (visible glazing must stay in the splat).
    on_pane = (np.abs(pos[:, 2] - 4.05) < 0.06) & (pos[:, 0] > 1.4) & (pos[:, 0] < 2.6) \
              & (pos[:, 1] > 0.9) & (pos[:, 1] < 1.9)
    # Deep-interior surfels, tested for sight from an exterior point at the window.
    interior = (pos[:, 0] > 0.3) & (pos[:, 0] < 3.7) & (pos[:, 1] > 0.3) \
               & (pos[:, 1] < 2.2) & (pos[:, 2] > 0.3) & (pos[:, 2] < 3.5)
    n_through = int(ray_visible(fs, [2.0, 1.4, 4.6], pos[interior]).sum()) if interior.any() else 0
    print(f"  [{name}] surfels={s3['splats']:,} culled={s3['culled_hidden']:,} "
          f"pane_surfels={int(on_pane.sum())} | cams={s4['cameras']} "
          f"coverage={s4['coverage']['satisfied_pct']}% | interior seen from outside: {n_through}")
    if name == "glass":
        check("pane keeps its own surfels", int(on_pane.sum()) > 0)
        check("interior surfels sightable from an EXTERIOR camera through the pane",
              n_through > 0, f"{n_through}")
    else:
        check("opaque control: NO interior surfel sightable from outside",
              n_through == 0, f"{n_through}")

# =====================================================================================
section("legacy freespace.npz (no glass field) degrades to block-everything")
# =====================================================================================
src = outA / stage2.FREESPACE_NAME
with np.load(src) as z:
    legacy = {k: z[k] for k in z.files if k != "occ_lin_glass"}
legacy_path = work / "legacy.npz"
np.savez_compressed(legacy_path, **legacy)
fs_old = stage2.load_free_space(legacy_path)
check("legacy: occluding set == full occupancy",
      fs_old.occ_lin_opaque.size == fs_old.occ_lin.size)
check("legacy: pane blocks sight again (old behavior)",
      not bool(ray_visible(fs_old, [2.0, 1.4, 5.0], [2.0, 1.4, 2.0])[0]))

print("\n" + ("ALL GLASS TESTS PASSED" if not FAILS else f"FAILURES ({len(FAILS)}): {FAILS}"))
print(f"artifacts: {work}")
sys.exit(1 if FAILS else 0)
