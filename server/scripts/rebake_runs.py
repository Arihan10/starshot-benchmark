"""Re-bake existing library-mode runs onto the optimized asset library.

By default NON-DESTRUCTIVE: writes optimized, placement-baked objects to
``<cell>/objects-optimized/`` alongside the untouched original
``<cell>/objects/``. The server serves whichever you point it at via
``STARSHOT_OBJECTS_SUBDIR`` (default ``objects``), so the two sets coexist and
you can switch back and forth.

Pass ``--prune-originals`` to reclaim disk as you go: once every convertible
object in a cell has a non-empty optimized twin, that cell's ``objects/`` is
deleted. This bounds peak disk to one cell's optimized output at a time, which
is what makes a full conversion fit when the raw ``objects/`` already fill the
volume. Cells with no library matches (pure Trellis generations) are never
touched.

A "cell" is one (run/slot/model) directory — anything containing an
events.jsonl. For each cell whose log has library matches, the placement of
each object is recovered straight from the event log and re-applied to the
optimized asset (which the live pipeline can no longer do via trimesh, since the
optimized GLBs are Meshopt/KTX2-compressed):

  * id -> library_id          from ``library.match`` events
  * id -> origin/dims/yaw      from ``bbox`` events
  * id rendered?              from ``model`` events
  * for every rendered match, write ``objects-optimized/<id>.glb`` via
    app.utils.glb_place using the asset's precomputed per-orientation bounds,
    and copy the optimized reference PNG. The bake reads only the event log +
    the library, so the original ``objects/`` need not be present — a bare
    ``events.jsonl`` (shared without its meshes, or already pruned) re-bakes
    exactly the set the run rendered.

Usage (from server/):
  uv run python scripts/rebake_runs.py                          # every cell
  uv run python scripts/rebake_runs.py --cell default/hotel-room/gpt
  uv run python scripts/rebake_runs.py --cell <c> --limit 5      # first 5 objects
  uv run python scripts/rebake_runs.py --force                   # redo existing
  uv run python scripts/rebake_runs.py --prune-originals         # convert + delete objects/
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


def _parse_cell(
    events_path: Path,
) -> tuple[dict[str, str], dict[str, tuple], set[str]]:
    """Returns (library_id_by_node, (origin, dims, orientation)_by_node,
    rendered_node_ids).

    `rendered_node_ids` are the nodes the run actually placed (emitted a
    ``model`` event) — the log-only equivalent of the old "has a baked
    objects/<id>.glb" check. Keying off the log instead of disk lets a cell
    whose heavy objects/ dir is absent (a bare events.jsonl shared without its
    meshes, or one already pruned) re-bake exactly the set the run rendered."""
    lib_by_id: dict[str, str] = {}
    bbox_by_id: dict[str, tuple] = {}
    rendered_ids: set[str] = set()
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
            elif kind == "model":
                node_id = event.get("id")
                if node_id:
                    rendered_ids.add(node_id)
    return lib_by_id, bbox_by_id, rendered_ids


def rebake_cell(cell_dir: Path, *, force: bool, limit: int, prune: bool) -> dict | None:
    events_path = cell_dir / "events.jsonl"
    if not events_path.exists():
        return None
    lib_by_id, bbox_by_id, rendered_ids = _parse_cell(events_path)
    if not lib_by_id:
        return None  # not a library-mode cell — nothing to re-bake

    objects_dir = cell_dir / "objects"
    out_dir = cell_dir / OPTIMIZED_SUBDIR

    # Convertible objects: a library match the run actually rendered (emitted a
    # ``model`` event) and whose placement we can recover from the log. Each
    # twin is baked from the event log + the library asset alone, so the
    # original objects/ dir need not exist — a bare events.jsonl is enough.
    targets = [
        node_id
        for node_id in lib_by_id
        if bbox_by_id.get(node_id) is not None and node_id in rendered_ids
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    orig_mb = _dir_size_mb(objects_dir)

    def _have(node_id: str) -> bool:
        twin = out_dir / f"{node_id}.glb"
        return twin.exists() and twin.stat().st_size > 0

    baked = unconvertible = already = 0
    if targets and not force and all(_have(nid) for nid in targets):
        already = len(targets)
    else:
        # Re-bake the whole convertible set rather than only the gaps, so a
        # half-written file from an interrupted pass can't survive into the
        # optimized set and then get treated as complete.
        count = 0
        for node_id in targets:
            if count >= limit:
                break
            origin, dims, orientation = bbox_by_id[node_id]
            asset = library.asset_path(lib_by_id[node_id])
            bounds = library.asset_rotated_bounds(lib_by_id[node_id], orientation)
            if not asset.exists() or bounds is None:
                unconvertible += 1
                continue
            glb_place.place_glb(
                src=asset,
                dst=out_dir / f"{node_id}.glb",
                bbox=BoundingBox(origin=tuple(origin), dimensions=tuple(dims)),
                orientation=orientation,
                rotated_min=bounds[0],
                rotated_max=bounds[1],
            )
            png = asset.with_suffix(".png")
            if png.exists():
                shutil.copyfile(png, out_dir / f"{node_id}.png")
            baked += 1
            count += 1

    # Prune only when every convertible object now has a non-empty optimized
    # twin — never when an asset was missing or a --limit left gaps.
    complete = bool(targets) and unconvertible == 0 and all(_have(nid) for nid in targets)
    pruned = False
    if prune and complete and objects_dir.exists():
        shutil.rmtree(objects_dir)
        pruned = True

    return {
        "baked": baked, "already": already, "unconvertible": unconvertible,
        "pruned": pruned, "orig_mb": orig_mb, "opt_mb": _dir_size_mb(out_dir),
        "status": "ok",
    }


def _discover_cells(runs_dir: Path, cell: str | None) -> list[Path]:
    if cell:
        return [runs_dir / cell]
    # Skip each run's `_branches/` temp folder — those are ephemeral simulation
    # forks (and their fan-out children), not source cells to optimize.
    return sorted({
        p.parent for p in runs_dir.rglob("events.jsonl")
        if "_branches" not in p.parts
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-bake runs onto the optimized library.")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--cell", type=str, default=None, help="run/slot/model under runs-dir; omit for all")
    parser.add_argument("--limit", type=int, default=10**9, help="max objects per cell (testing)")
    parser.add_argument("--force", action="store_true", help="re-bake even if the optimized object exists")
    parser.add_argument(
        "--prune-originals",
        action="store_true",
        help="delete a cell's objects/ once every convertible object has a non-empty optimized twin",
    )
    args = parser.parse_args()

    cells = _discover_cells(args.runs_dir, args.cell)
    grand = {"baked": 0, "already": 0, "unconvertible": 0, "pruned_cells": 0}
    rebaked_cells = 0
    kept: list[str] = []
    for cell_dir in cells:
        result = rebake_cell(
            cell_dir, force=args.force, limit=args.limit, prune=args.prune_originals
        )
        if result is None:
            continue
        rebaked_cells += 1
        rel = cell_dir.relative_to(args.runs_dir).as_posix()
        grand["baked"] += result["baked"]
        grand["already"] += result["already"]
        grand["unconvertible"] += result["unconvertible"]
        if result["pruned"]:
            grand["pruned_cells"] += 1
        elif args.prune_originals and result["status"] == "ok":
            kept.append(rel)
        print(
            f"[rebake] {rel}: baked={result['baked']} already={result['already']} "
            f"unconvertible={result['unconvertible']} pruned={result['pruned']}  |  "
            f"objects {result['orig_mb']}MB -> optimized {result['opt_mb']}MB",
            flush=True,
        )

    print(
        f"\n[rebake] {rebaked_cells} library cells  baked={grand['baked']} "
        f"already={grand['already']} unconvertible={grand['unconvertible']} "
        f"pruned_cells={grand['pruned_cells']}",
        flush=True,
    )
    if args.prune_originals and kept:
        print(
            f"[rebake] kept objects/ for {len(kept)} cell(s) (incomplete or unconvertible):",
            flush=True,
        )
        for rel in kept:
            print(f"  - {rel}", flush=True)


if __name__ == "__main__":
    main()
