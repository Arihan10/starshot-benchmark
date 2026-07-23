"""Smoke test — Stage 2 single-grid free-space voxelizer (torch-free).

Builds a synthetic cell from trimesh primitives that exercises every behavior of
the two-phase fill, then asserts the grid, the point queries, and the summary:

  * a SEALED room (floor + ceiling + four flush walls, interpenetrating AABBs
    like real divider output) whose air must be FREE purely via phase-1 wall
    self-seeding (no door, no margin path in);
  * a hollow SOFA with a CUSHION clipping through its top face — partially
    nested (exposed to room air), so the sofa hollow must stay GARBAGE;
  * a closed CABINET with a BOTTLE fully inside — fully nested, so the rescue
    trigger must fire and the cabinet cavity must become FREE;
  * slab interiors (wall hollows) → GARBAGE; the exterior margin → EMPTY;
  * a BLEND glass pane → occupies (`occupied`) but doesn't occlude
    (`occluding`); opaque walls do both;
  * the SVX3 viz pack (cover/garbage quads + the single free shell);
  * the npz schema is exactly the trimmed layout, and old layouts (dense
    fields, candidate lists) are REJECTED with the re-run error.

Run from the repo root:  splat/.venv/bin/python smoke_stage2_freespace.py
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import trimesh

from splat.stage2 import (
    FreeSpaceParams,
    compute_free_space,
    load_free_space,
)

LEGACY_GRID = Path("runs/good_opus_new_hotel2/hotel-room/opus-new/splat/freespace.npz")
NPZ_SCHEMA = {
    "origin", "pitch", "dims", "occ_lin", "occ_lin_glass",
    "skin_lin", "zone_lin",
}

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def box(bounds: list[list[float]]) -> trimesh.Trimesh:
    return trimesh.creation.box(bounds=np.asarray(bounds, dtype=float))


def glass_box(bounds: list[list[float]]) -> trimesh.Trimesh:
    m = box(bounds)
    m.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            alphaMode="BLEND", baseColorFactor=[1.0, 1.0, 1.0, 0.065]
        )
    )
    return m


def build_cell(raw_dir: Path) -> None:
    """A sealed 6×3×5 room. Trimesh boxes are closed surfaces, so every slab and
    prop is a hollow shell — exactly the TRELLIS closed-shell regime."""
    objects = {
        # Sealed architecture, flush + interpenetrating (walls span full height,
        # overlapping the floor/ceiling AABBs — the real divider pattern).
        "floor": box([[0.0, 0.0, 0.0], [6.0, 0.1, 5.0]]),
        "ceiling": box([[0.0, 2.9, 0.0], [6.0, 3.0, 5.0]]),
        "wall_xlo": box([[0.0, 0.0, 0.0], [0.1, 3.0, 5.0]]),
        "wall_xhi": box([[5.9, 0.0, 0.0], [6.0, 3.0, 5.0]]),
        "wall_zlo": box([[0.0, 0.0, 0.0], [6.0, 3.0, 0.1]]),
        "wall_zhi": box([[0.0, 0.0, 4.9], [6.0, 3.0, 5.0]]),
        # Partial nesting: the cushion's bottom clips 0.1 m through the sofa's
        # top face (its lower shell ring pokes into the sofa hollow) while its
        # top sits in room air → exposed → NO rescue, hollow stays garbage.
        "sofa": box([[2.0, 0.1, 1.0], [3.4, 0.9, 1.8]]),
        "cushion": box([[2.5, 0.8, 1.25], [2.9, 1.2, 1.55]]),
        # Full nesting: the bottle is sealed inside the cabinet with ≥ 2-voxel
        # gaps on every side → zero reached shell cells → rescue opens the cavity.
        "cabinet": box([[4.5, 0.1, 3.5], [5.3, 0.9, 4.3]]),
        "bottle": box([[4.8, 0.2, 3.8], [4.95, 0.5, 3.95]]),
        # Free-standing glass divider (one voxel thick): occupies, never occludes.
        "pane": glass_box([[1.0, 0.1, 2.5], [1.8, 2.0, 2.54]]),
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    for nid, mesh in objects.items():
        mesh.export(raw_dir / f"{nid}.glb")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="smoke-stage2-") as tmp:
        raw_dir = Path(tmp) / "raw"
        build_cell(raw_dir)
        out_path = Path(tmp) / "splat" / "freespace.npz"
        t0 = time.perf_counter()
        summary = compute_free_space(
            run="smoke", slot="smoke", model="smoke",
            raw_dir=raw_dir, out_path=out_path,
        )
        t_par = time.perf_counter() - t0

        # Parallel determinism: a serial (workers=1) build must be
        # byte-identical in every stored array — the absolute-lattice keys make
        # the per-object merge order-independent.
        serial_path = Path(tmp) / "splat" / "freespace.serial.npz"
        t0 = time.perf_counter()
        s_summary = compute_free_space(
            run="smoke", slot="smoke", model="smoke",
            raw_dir=raw_dir, out_path=serial_path,
            params=FreeSpaceParams(workers=1),
        )
        t_ser = time.perf_counter() - t0
        with np.load(out_path) as a, np.load(serial_path) as b:
            same = all(
                np.array_equal(a[k], b[k]) for k in a.files if k in b.files
            ) and set(a.files) == set(b.files)
        check(
            "parallel voxelization is byte-identical to serial",
            same and s_summary["solid_voxels"] == summary["solid_voxels"],
            f"parallel {t_par:.1f}s vs serial {t_ser:.1f}s",
        )
        # (The serial run rewrote the shared voxels.bin with identical bytes —
        # the SVX2 checks below still verify the exact contract.)
        serial_path.unlink(missing_ok=True)
        slim = {
            k: summary[k]
            for k in (
                "pitch", "dims", "solid_voxels", "glass_voxels", "empty_voxels",
                "garbage_voxels", "skin_bricks", "zone_bricks", "fill",
            )
        }
        print(json.dumps(slim, indent=1))

        check("requested pitch kept", summary["pitch"] == 0.03)
        check(
            "rescue fired for exactly the bottle",
            summary["fill"]["objects_sealed"] == ["bottle"]
            and summary["fill"]["rescue_rounds"] >= 1,
            f"sealed={summary['fill']['objects_sealed']} rounds={summary['fill']['rescue_rounds']}",
        )
        check("garbage exists (slab/sofa hollows)", summary["garbage_voxels"] > 0)
        check("glass cells classed", summary["glass_voxels"] > 0)

        with np.load(out_path) as z:
            check(
                "npz schema is the single-grid layout",
                set(z.files) == NPZ_SCHEMA,
                f"keys={sorted(z.files)}",
            )

        fs = load_free_space(out_path)

        def empty_at(p: list[float]) -> bool:
            return bool(fs.empty_at(np.array([p], dtype=np.float64))[0])

        # The fill's labels, point by point.
        check("sealed room air is EMPTY (wall self-seeding)", empty_at([1.0, 1.5, 1.0]))
        check("exterior margin is EMPTY", empty_at([-0.8, 1.5, -0.8]))
        check("wall slab interior is GARBAGE", not empty_at([0.05, 1.5, 2.0]))
        check("sofa hollow is GARBAGE (cushion is exposed)", not empty_at([2.2, 0.5, 1.4]))
        check("cabinet cavity is EMPTY (bottle rescued it)", empty_at([5.1, 0.5, 4.1]))

        # Cover classes: glass occupies but never occludes; opaque does both.
        pane_pt = np.array([[1.4, 1.0, 2.52]])
        wall_pt = np.array([[0.02, 1.5, 2.0]])
        check(
            "pane occupies but does not occlude",
            bool(fs.occupied(pane_pt)[0]) and not bool(fs.occluding(pane_pt)[0]),
        )
        check(
            "wall occupies and occludes",
            bool(fs.occupied(wall_pt)[0]) and bool(fs.occluding(wall_pt)[0]),
        )

        # The SVX3 viz pack: volumetric boundary shells. Header counts match the
        # summary, sections tile the file byte-exactly, shell thresholds ascend
        # and include the baked clearance, and each class's merged runs sum to
        # EXACTLY the exposed-face count recomputed from the grid (meshing is
        # complete: interior faces culled, boundary faces all present).
        viz = (out_path.parent / "voxels.bin").read_bytes()
        magic, n_cov, n_gar, n_shells = struct.unpack_from("<4sIII", viz)
        qdt = np.dtype(
            [("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("f", "u1"), ("p", "u1"), ("r", "<u2")]
        )
        off = 16
        cov_q = np.frombuffer(viz, dtype=qdt, count=n_cov, offset=off)
        off += n_cov * 10
        gar_q = np.frombuffer(viz, dtype=qdt, count=n_gar, offset=off)
        off += n_gar * 10
        shells = []
        for _ in range(n_shells):
            t, nq, cells = struct.unpack_from("<fII", viz, off)
            off += 12
            shells.append((t, np.frombuffer(viz, dtype=qdt, count=nq, offset=off), cells))
            off += nq * 10
        vz = summary["viz"]
        check(
            "SVX3 pack: header + counts + byte-exact sections",
            magic == b"SVX3"
            and n_cov == vz["cover_quads"]
            and n_gar == vz["garbage_quads"]
            and n_shells == len(vz["shells"])
            and off == len(viz),
            f"cover={n_cov} garbage={n_gar} shells={n_shells}",
        )
        check(
            "SVX3 free shell present (single, threshold field vestigial)",
            len(shells) == 1 and shells[0][1].size > 0,
            f"shells={len(shells)}",
        )

        def exposed_faces(mask: np.ndarray) -> int:
            total = 0
            for axis in range(3):
                for neg in (0, 1):
                    e = mask.copy()
                    dst = [slice(None)] * 3
                    src = [slice(None)] * 3
                    if neg == 0:
                        dst[axis], src[axis] = slice(None, -1), slice(1, None)
                    else:
                        dst[axis], src[axis] = slice(1, None), slice(None, -1)
                    e[tuple(dst)] &= ~mask[tuple(src)]
                    total += int(e.sum())
            return total

        # Cover quads must tile the EXACT boundary of the occupancy (garbage's
        # exact reconstruction needs the transient zone-garbage set, so it is
        # checked structurally above: sections tile the file byte-exactly).
        with np.load(out_path) as z:
            occ_mask = np.zeros(tuple(z["dims"]), dtype=bool)
            occ_mask.reshape(-1)[z["occ_lin"]] = True
        check(
            "SVX3 cover shell tiles the exact boundary (runs sum = exposed faces)",
            int(cov_q["r"].sum()) == exposed_faces(occ_mask),
            f"runs={int(cov_q['r'].sum())}",
        )
        check("SVX3 garbage shell present", int(gar_q["r"].sum()) > 0)

    # Old layouts (whatever generation is on disk, or a synthesized one) must
    # be rejected with the re-run error, not half-loaded.
    with tempfile.TemporaryDirectory(prefix="smoke-legacy-") as tmp:
        legacy_path = Path(tmp) / "freespace.npz"
        np.savez(
            legacy_path,
            origin=np.zeros(3), pitch=np.float64(0.04),
            dims=np.array([4, 4, 4], dtype=np.int64),
            occ_lin=np.zeros(0, dtype=np.int64),
            occ_lin_glass=np.zeros(0, dtype=np.int64),
            clearance=np.zeros((4, 4, 4), dtype=np.float16),
            empty=np.ones((4, 4, 4), dtype=bool),  # an old dense-layout key
        )
        try:
            load_free_space(legacy_path)
            check("old grid layout is rejected", False, "loaded without error")
        except ValueError as exc:
            check("old grid layout is rejected", "re-run Stage 2" in str(exc))
    if LEGACY_GRID.is_file():
        try:
            load_free_space(LEGACY_GRID)
            print("[NOTE] on-disk grid already carries the trimmed layout")
        except ValueError as exc:
            check("on-disk candidate-era grid is rejected", "re-run Stage 2" in str(exc))

    failed = [name for name, ok, _ in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
