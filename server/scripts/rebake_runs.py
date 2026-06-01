"""Re-bake existing library-mode runs onto the optimized asset library.

NON-DESTRUCTIVE: writes optimized, placement-baked objects to
``<cell>/objects-optimized/`` alongside the untouched original
``<cell>/objects/``. The server serves whichever you point it at via
``STARSHOT_OBJECTS_SUBDIR`` (default ``objects``), so the two sets coexist and
you can switch back and forth.

A "cell" is one (run/slot/model) directory — anything containing an
events.jsonl. For each cell whose log has library matches, the placement of
each object is recovered straight from the event log and re-applied to the
optimized asset (which the live pipeline can no longer do via trimesh, since the
optimized GLBs are Meshopt/KTX2-compressed):

  * id -> library_id          from ``library.match`` events
  * id -> origin/dims/yaw      from ``bbox`` events
  * for every object with a baked ``objects/<id>.glb``, write
    ``objects-optimized/<id>.glb`` via app.utils.glb_place using the asset's
    precomputed per-orientation bounds, and copy the optimized reference PNG.

Usage (from server/):
  uv run python scripts/rebake_runs.py                          # every cell
  uv run python scripts/rebake_runs.py --cell default/hotel-room/gpt
  uv run python scripts/rebake_runs.py --cell <c> --limit 5      # first 5 objects
  uv run python scripts/rebake_runs.py --force                   # redo existing
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SERVER_DIR))

from app.core.types import BoundingBox  # noqa: E402
from app.services import library  # noqa: E402
from app.utils import glb_place  # noqa: E402

_REPO_ROOT = _SERVER_DIR.parent
DEFAULT_RUNS_DIR = Path(os.environ.get("STARSHOT_RUNS_DIR", str(_REPO_ROOT / "runs")))
OPTIMIZED_SUBDIR = "objects-optimized"


def _dir_size_mb(path: Path, suffix: str = ".glb") -> float:
    if not path.is_dir():
        return 0.0
    total = sum(p.stat().st_size for p in path.glob(f"*{suffix}"))
    return round(total / 1048576, 1)


def _parse_cell(events_path: Path) -> tuple[dict[str, str], dict[str, tuple]]:
    """Returns (library_id_by_node, (origin, dims, orientation)_by_node)."""
    lib_by_id: dict[str, str] = {}
    bbox_by_id: dict[str, tuple] = {}
    with events_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("kind")
            if kind == "library.match":
                node_id, lib = event.get("id"), event.get("library_id")
                if node_id and lib:
                    lib_by_id[node_id] = lib
            elif kind == "bbox":
                node_id = event.get("id")
                origin, dims = event.get("origin"), event.get("dimensions")
                if node_id and origin and dims:
                    orientation = int(event.get("orientation", 0) or 0)
                    bbox_by_id[node_id] = (origin, dims, orientation)
    return lib_by_id, bbox_by_id


def rebake_cell(cell_dir: Path, *, force: bool, limit: int) -> dict[str, int] | None:
    events_path = cell_dir / "events.jsonl"
    if not events_path.exists():
        return None
    lib_by_id, bbox_by_id = _parse_cell(events_path)
    if not lib_by_id:
        return None  # not a library-mode cell — nothing to re-bake

    objects_dir = cell_dir / "objects"
    out_dir = cell_dir / OPTIMIZED_SUBDIR
    out_dir.mkdir(exist_ok=True)

    done = skipped = missing = 0
    for node_id, library_id in lib_by_id.items():
        if done >= limit:
            break
        src_object = objects_dir / f"{node_id}.glb"
        bb = bbox_by_id.get(node_id)
        if not src_object.exists() or bb is None:
            # The original object never landed (errored mid-run) or has no bbox.
            skipped += 1
            continue
        dst = out_dir / f"{node_id}.glb"
        if dst.exists() and not force:
            skipped += 1
            continue
        origin, dims, orientation = bb
        asset = library.asset_path(library_id)
        bounds = library.asset_rotated_bounds(library_id, orientation)
        if not asset.exists() or bounds is None:
            missing += 1
            continue
        glb_place.place_glb(
            src=asset,
            dst=dst,
            bbox=BoundingBox(origin=tuple(origin), dimensions=tuple(dims)),
            orientation=orientation,
            rotated_min=bounds[0],
            rotated_max=bounds[1],
        )
        png = asset.with_suffix(".png")
        if png.exists():
            shutil.copyfile(png, out_dir / f"{node_id}.png")
        done += 1

    return {"done": done, "skipped": skipped, "missing": missing}


def _discover_cells(runs_dir: Path, cell: str | None) -> list[Path]:
    if cell:
        return [runs_dir / cell]
    return sorted({p.parent for p in runs_dir.rglob("events.jsonl")})


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-bake runs onto the optimized library.")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--cell", type=str, default=None, help="run/slot/model under runs-dir; omit for all")
    parser.add_argument("--limit", type=int, default=10**9, help="max objects per cell (testing)")
    parser.add_argument("--force", action="store_true", help="re-bake even if the optimized object exists")
    args = parser.parse_args()

    cells = _discover_cells(args.runs_dir, args.cell)
    grand = {"done": 0, "skipped": 0, "missing": 0}
    rebaked_cells = 0
    for cell_dir in cells:
        result = rebake_cell(cell_dir, force=args.force, limit=args.limit)
        if result is None:
            continue
        rebaked_cells += 1
        rel = cell_dir.relative_to(args.runs_dir).as_posix()
        orig_mb = _dir_size_mb(cell_dir / "objects")
        opt_mb = _dir_size_mb(cell_dir / OPTIMIZED_SUBDIR)
        for k in grand:
            grand[k] += result[k]
        print(
            f"[rebake] {rel}: {result['done']} baked, {result['skipped']} skipped, "
            f"{result['missing']} missing  |  objects {orig_mb}MB -> optimized {opt_mb}MB",
            flush=True,
        )

    print(
        f"\n[rebake] {rebaked_cells} library cells  "
        f"done={grand['done']} skipped={grand['skipped']} missing={grand['missing']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
