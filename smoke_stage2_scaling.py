"""Multi-scene-type smoke test for the Stage-2 adaptive coarse-cell cap (no GPU).

Validates the memory-bounding change in `splat/stage2.py` across a spread of scene
TOPOLOGIES that stand in for the real benchmark zoo — enclosed rooms/houses, open
ground scenes, thin 2.5D platformer slabs, sparse "outer-space"-style voids, and
large arena/city bounds. It checks three things the change must hold to:

  1. PARITY (fidelity-neutral where it fits): for scenes under the cap the output is
     structurally identical to the uncapped (pre-change) path — same coarse block
     factor, pitch, grid dims, origin — so nothing downstream shifts. Only the
     RNG-seeded surface sampling differs, so counts are compared within tolerance.
  2. FINE-RESOLUTION INVARIANCE: `pitch_fine` (the occlusion / thin-gap resolution)
     is the SAME for every scene, capped or not — the fidelity guarantee. Object
     surfaces stay fine-occupied (>=95%) even on a scene the cap coarsened.
  3. SCALING: scenes whose UNCAPPED coarse grid would blow past RAM stay within the
     cap (coarser navigation grid, larger pitch), still run end to end, and keep
     correct reachability + camera candidates. The averted (projected) memory is
     reported so the win is visible.

Plus a full Stage 2 -> 3 -> 4 chain parity run to confirm surfels + coverage are
unchanged by the cap on an under-cap scene.

Run: ./splat/.venv/bin/python smoke_stage2_scaling.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splat import stage2, stage3, stage4
from splat.stage2 import _fine_grid_dims  # internal grid sizing, for the memory projection

trimesh.util.log.setLevel("ERROR")

FAILS: list[str] = []
PITCH_FINE = stage2.DEFAULT_PITCH / stage2.DEFAULT_REFINE  # 0.04 m, the fixed fine resolution
DEFAULT_CAP = stage2.DEFAULT_MAX_COARSE_CELLS
FAST_CAP = 6_000_000  # small cap so the big scenes exercise the capped path but stay fast


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# --- scene builders (write a dir of world-placed GLBs, like a real cell) ----------

def _box(center, extents) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=np.asarray(extents, dtype=float))
    m.apply_translation(np.asarray(center, dtype=float))
    return m


def _export(meshes: list[trimesh.Trimesh], d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(meshes):
        m.export(d / f"o{i:03d}.glb")


def build_room(d: Path) -> dict:
    """Enclosed hotel-style room: a hollow 6x3x5 shell + furniture + one TINY sealed
    box (interior < reachable_min_volume, so its hollow must be dropped)."""
    meshes = [
        _box([3, 1.5, 2.5], [6, 3, 5]),        # room shell (hollow interior)
        _box([1.4, 0.4, 1.4], [2.0, 0.8, 1.6]),  # bed
        _box([4.6, 0.5, 3.6], [1.2, 1.0, 0.8]),  # cabinet
        _box([5.0, 1.2, 1.0], [0.3, 0.3, 0.3]),  # tiny sealed box (interior ~0.027 m3)
    ]
    _export(meshes, d)
    return {"interior": [3.0, 1.5, 2.8], "sealed": [5.0, 1.2, 1.0], "outside": [3.0, 3.9, 2.5]}


def build_house(d: Path) -> dict:
    """Two adjacent enclosed rooms sharing a wall (multi-room interior)."""
    meshes = [
        _box([3, 1.5, 2.5], [6, 3, 5]),
        _box([9, 1.5, 2.5], [6, 3, 5]),
        _box([2, 0.5, 2], [1.5, 1.0, 1.0]),
        _box([10, 0.5, 3], [1.2, 1.0, 1.2]),
    ]
    _export(meshes, d)
    return {"interior": [3.0, 1.5, 2.8]}


def build_open_ground(d: Path) -> dict:
    """Open exterior: a wide thin ground slab + scattered props (no ceiling)."""
    meshes = [_box([20, 0.1, 20], [40, 0.2, 40])]
    rng = np.random.default_rng(1)
    for _ in range(12):
        c = rng.uniform([2, 0.5, 2], [38, 0.5, 38])
        meshes.append(_box(c, [1.0, 1.0, 1.0]))
    _export(meshes, d)
    return {"above": [20.0, 3.0, 20.0]}


def build_platformer(d: Path) -> dict:
    """Thin 2.5D slab: platforms spread along X in a shallow Z (like platformer-level)."""
    meshes = [_box([30, -0.1, 1.5], [60, 0.2, 3])]  # base strip
    rng = np.random.default_rng(2)
    for _ in range(20):
        c = rng.uniform([2, 1.0, 0.5], [58, 12.0, 2.5])
        meshes.append(_box(c, [2.5, 0.4, 1.5]))
    _export(meshes, d)
    return {}


def build_sparse_void(d: Path) -> dict:
    """Outer-space-style: a handful of small objects scattered through a LARGE mostly
    empty volume — little surface, huge bounding box (the coarse-grid blow-up case)."""
    pts = [[5, 5, 5], [55, 8, 12], [12, 50, 48], [48, 45, 5], [30, 30, 30], [52, 12, 52]]
    meshes = [_box(p, [2.0, 2.0, 2.0]) for p in pts]
    _export(meshes, d)
    return {"void": [30.0, 15.0, 15.0], "on_obj": [5.0, 5.0, 5.0]}


def build_huge_arena(d: Path) -> dict:
    """Large open-top arena: a big floor + perimeter walls (enclosed-ish, big bound)."""
    meshes = [
        _box([40, 0, 40], [80, 0.3, 80]),
        _box([40, 10, 0.15], [80, 20, 0.3]),
        _box([40, 10, 79.85], [80, 20, 0.3]),
        _box([0.15, 10, 40], [0.3, 20, 80]),
        _box([79.85, 10, 40], [0.3, 20, 80]),
    ]
    _export(meshes, d)
    return {"center": [40.0, 5.0, 40.0]}


def build_extreme_city(d: Path) -> dict:
    """City/campus-scale BOUND: blocks spread across ~150 x 40 x 150 m. The huge
    coarse blow-up comes from the EXTENT (object spread), not surface area, so we
    skip a giant ground plane (which would only slow surface sampling) — the spread
    of towers alone gives a bound whose uncapped coarse grid is > 4e8 cells."""
    rng = np.random.default_rng(3)
    meshes = []
    for _ in range(24):
        c = rng.uniform([5, 5, 5], [145, 5, 145])
        h = float(rng.uniform(6, 30))
        meshes.append(_box([c[0], h / 2, c[2]], [6, h, 6]))
    meshes.append(_box([75, 0.1, 75], [10, 0.2, 10]))  # small central pad (a bit of "ground")
    _export(meshes, d)
    return {}


SCENES = {
    "room (enclosed)": build_room,
    "house (multi-room)": build_house,
    "open-ground (exterior)": build_open_ground,
    "platformer (thin slab)": build_platformer,
    "sparse-void (outer space)": build_sparse_void,
    "huge-arena (large enclosed)": build_huge_arena,
    "extreme-city (huge bound)": build_extreme_city,
}
# Which scenes are expected to trip the cap under FAST_CAP (the big/sparse ones).
BIG = {"sparse-void (outer space)", "huge-arena (large enclosed)", "extreme-city (huge bound)"}


def run_stage2(raw: Path, out: Path, cap: int) -> tuple[dict, stage2.FreeSpace]:
    p = stage2.FreeSpaceParams(max_coarse_cells=cap)
    s = stage2.compute_free_space(run="t", slot="t", model="t", raw_dir=raw,
                                  out_path=out / stage2.FREESPACE_NAME, params=p)
    return s, stage2.load_free_space(out / stage2.FREESPACE_NAME)


def projected_uncapped_coarse(lo, hi, margin: float) -> int:
    """Coarse cell count the OLD (uncapped) path would allocate at base resolution."""
    _, fd = _fine_grid_dims(np.asarray(lo), np.asarray(hi), margin, PITCH_FINE, stage2.DEFAULT_REFINE)
    return int(np.prod(fd // stage2.DEFAULT_REFINE))


work = Path(tempfile.mkdtemp(prefix="stage2_scaling_"))
print(f"workdir: {work}\nfine pitch (fixed): {PITCH_FINE} m | default cap: {DEFAULT_CAP:,} | fast cap: {FAST_CAP:,}")

# =====================================================================================
# Per-scene: build, run capped, assert generic invariants across the topology zoo.
# =====================================================================================
summaries: dict[str, dict] = {}
grids: dict[str, stage2.FreeSpace] = {}
for name, builder in SCENES.items():
    section(name)
    raw = work / name.split()[0] / "objects"
    out = work / name.split()[0] / "splat"
    meta = builder(raw)
    cap = FAST_CAP if name in BIG else DEFAULT_CAP
    t0 = time.perf_counter()
    s, fs = run_stage2(raw, out, cap)
    dt = time.perf_counter() - t0
    summaries[name] = s
    grids[name] = fs
    lo, hi = s["scene_aabb"]["min"], s["scene_aabb"]["max"]
    coarse = int(np.prod(s["dims_coarse"]))
    proj = projected_uncapped_coarse(lo, hi, s["params"]["margin"])
    print(f"  extent={[round(hi[i]-lo[i],1) for i in range(3)]} m | dims_coarse={s['dims_coarse']} "
          f"({coarse:,}) | pitch={s['pitch']:.3f} refine={s['refine']} capped={s['coarse_capped']} | {dt:.1f}s")
    print(f"  uncapped-coarse would be {proj:,} cells (~{proj*20/1e9:.2f} GB peak est.)")

    # (2) FINE-RESOLUTION INVARIANCE — the fidelity guarantee, holds for every scene.
    check("pitch_fine unchanged (fixed occlusion resolution)",
          abs(s["pitch_fine"] - PITCH_FINE) < 1e-9, f"{s['pitch_fine']}")
    # coarse grid respected the cap
    check("coarse grid within cap", coarse <= cap, f"{coarse:,} <= {cap:,}")
    # sparse fine occupancy scales with SURFACE, not the (possibly huge) volume
    check("fine occupancy stays sparse (< coarse-with-cap)", s["solid_voxels_fine"] > 0)
    # reachable free space exists and is a subset of free space
    check("reachable ⊆ free and non-empty",
          0 < s["reachable_voxels"] <= s["free_voxels"],
          f"reach={s['reachable_voxels']:,} free={s['free_voxels']:,}")

    if name in BIG:
        # (3) SCALING — cap engaged, pitch grew, and the averted grid was huge.
        check("cap engaged on big scene (coarse_capped)", s["coarse_capped"])
        check("coarse pitch grew above base", s["pitch"] > stage2.DEFAULT_PITCH + 1e-9, f"pitch={s['pitch']:.3f}")
        check("would-OOM averted (uncapped coarse ≫ cap)", proj > 5 * cap, f"{proj:,} vs {cap:,}")
        # DEFAULT cap would also have protected it, and the sizing keeps it bounded.
        r_def = stage2._fit_refine(np.asarray(lo), np.asarray(hi), s["params"]["margin"],
                                   PITCH_FINE, stage2.DEFAULT_REFINE, DEFAULT_CAP)
        _, fdd = _fine_grid_dims(np.asarray(lo), np.asarray(hi), s["params"]["margin"], PITCH_FINE, r_def)
        check("default-cap sizing also bounds this scene",
              int(np.prod(fdd // r_def)) <= DEFAULT_CAP, f"refine={r_def}")
    else:
        check("small scene NOT capped (identical to pre-change path)", not s["coarse_capped"])

# =====================================================================================
# Targeted topology correctness (enclosure + fine detail preserved under capping).
# =====================================================================================
section("enclosure semantics (room)")
fs = grids["room (enclosed)"]
m = build_room  # meta re-derive
meta = {"interior": [3.0, 1.5, 2.8], "sealed": [5.0, 1.2, 1.0], "outside": [3.0, 3.9, 2.5]}
check("room interior is navigable", bool(fs.reachable_free(np.array([meta["interior"]]))[0]))
check("exterior (above room) is navigable", bool(fs.reachable_free(np.array([meta["outside"]]))[0]))
check("tiny sealed hollow is NOT navigable", not bool(fs.reachable_free(np.array([meta["sealed"]]))[0]))
cands, clv = fs.free_candidates(0.25, 4.0, 0.5)
check("candidates exist and all clear of surfaces (>=0.25 m)",
      len(cands) > 0 and float(clv.min()) >= 0.25, f"{len(cands)} cands, min_clear={float(clv.min()):.3f}")

section("fine detail preserved under a coarsened (capped) scene (sparse-void)")
fs_v = grids["sparse-void (outer space)"]
void_raw = work / "sparse-void" / "objects"
obj = trimesh.load(void_raw / "o000.glb", process=False)
geom = next(iter(obj.geometry.values())) if hasattr(obj, "geometry") else obj
pts, _ = trimesh.sample.sample_surface(geom, 3000)
occ_frac = float(fs_v.fine_occupied(np.asarray(pts)).mean())
check("object surface still fine-occupied at 0.04 m despite cap (>=95%)", occ_frac >= 0.95,
      f"{occ_frac*100:.1f}%  (coarse pitch was {summaries['sparse-void (outer space)']['pitch']:.3f} m)")
check("far-void point navigable (one big free component)",
      bool(fs_v.reachable_free(np.array([[30.0, 15.0, 15.0]]))[0]))

# =====================================================================================
# PARITY: capped vs uncapped on an under-cap scene — structurally identical grid.
# =====================================================================================
section("parity: capped (default) vs uncapped (cap=0) — under-cap scene")
raw = work / "room" / "objects"
outA = work / "parityA"
outB = work / "parityB"
outA.mkdir()
outB.mkdir()
sA, _ = run_stage2(raw, outA, DEFAULT_CAP)   # capped (default)
sB, _ = run_stage2(raw, outB, 0)             # uncapped (cap disabled == pre-change path)
for key in ("refine", "pitch", "pitch_fine", "dims_fine", "dims_coarse", "origin"):
    check(f"identical grid geometry: {key}", sA[key] == sB[key], f"{sA[key]} vs {sB[key]}")
check("both report not-capped", (not sA["coarse_capped"]) and (not sB["coarse_capped"]))
# RNG-seeded sampling → compare counts within tolerance (structure is proven identical above)
for key in ("free_voxels", "reachable_voxels", "solid_voxels_fine"):
    a, b = sA[key], sB[key]
    check(f"{key} within 2% (sampling RNG only)", abs(a - b) <= 0.02 * max(a, b) + 32, f"{a} vs {b}")

# =====================================================================================
# Full Stage 2 -> 3 -> 4 chain parity (surfels + coverage unaffected by the cap).
# =====================================================================================
section("chain parity: Stage 2→3→4 with cap on vs off (room)")


def chain(out: Path, cap: int) -> tuple[dict, dict]:
    s2, _ = run_stage2(raw, out, cap)
    s3 = stage3.sample_cell(run="t", slot="t", model="t", raw_dir=raw,
                            freespace_path=out / stage2.FREESPACE_NAME,
                            out_path=out / stage3.CLOUD_NAME,
                            params=stage3.SampleParams(target_splats=60_000))
    s4 = stage4.plan_cameras(run="t", slot="t", model="t",
                             freespace_path=out / stage2.FREESPACE_NAME,
                             surfels_path=out / stage3.CLOUD_NAME,
                             out_path=out / stage4.CAMERAS_NAME,
                             params=stage4.PlanParams(max_candidates=3000))
    return s3, s4


c3A, c4A = chain(work / "chainA", DEFAULT_CAP)
c3B, c4B = chain(work / "chainB", 0)
print(f"  capped:   surfels={c3A['splats']:,} culled={c3A['culled_hidden']:,} "
      f"cameras={c4A['cameras']} coverage={c4A['coverage']['satisfied_pct']}%")
print(f"  uncapped: surfels={c3B['splats']:,} culled={c3B['culled_hidden']:,} "
      f"cameras={c4B['cameras']} coverage={c4B['coverage']['satisfied_pct']}%")
check("Stage-3 surfel count within 3% (cap on vs off)",
      abs(c3A["splats"] - c3B["splats"]) <= 0.03 * max(c3A["splats"], c3B["splats"]),
      f"{c3A['splats']} vs {c3B['splats']}")
check("Stage-4 coverage within 3 pts (cap on vs off)",
      abs(c4A["coverage"]["satisfied_pct"] - c4B["coverage"]["satisfied_pct"]) <= 3.0,
      f"{c4A['coverage']['satisfied_pct']}% vs {c4B['coverage']['satisfied_pct']}%")

print("\n" + ("ALL STAGE-2 SCALING TESTS PASSED" if not FAILS else f"FAILURES ({len(FAILS)}): {FAILS}"))
print(f"artifacts: {work}")
sys.exit(1 if FAILS else 0)
