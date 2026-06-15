"""Re-bake the upright reorientation into already-finished v5 runs.

v5-sidescroller now bakes a fixed +Z->+Y turn (pipeline.V5_UPRIGHT_ROTATION)
into every placement so library assets stand upright on the side-view plane
instead of lying flat. Runs baked before that change have flat assets; this
walks each v5 run, reconstructs every placement from its events.jsonl
(`bbox` for the box, `library.match` for the asset), and re-bakes
`objects/<id>.glb` from the source library asset with the rotation applied —
exactly what `_place_one` does live.

Idempotent: re-baking always starts from the pristine source asset, so
running twice produces the same result. Objects whose asset lacked augmented
bounds (logged `library.bounds_missing`, copied through unscaled) are left
alone, mirroring the runtime fallback.

Usage (from server/):
    uv run python scripts/reorient_v5_assets.py            # all v5 runs
    uv run python scripts/reorient_v5_assets.py <run>      # one run
    uv run python scripts/reorient_v5_assets.py --dry-run
    uv run python scripts/reorient_v5_assets.py --oneshot-dir ../oneshot-runs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.types import BoundingBox
from app.oneshot.pipeline import V5_UPRIGHT_ROTATION
from app.services import library
from app.utils import glb_place

V5_VERSION = "v5-sidescroller"


def _read_version(run_dir: Path) -> str:
    try:
        return (run_dir / "version").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _placements(events_path: Path) -> dict[str, dict]:
    """id -> {bbox, orientation, library_id} reconstructed from the cell log.
    Only ids with both a bbox event and a library match are returned."""
    boxes: dict[str, dict] = {}
    matches: dict[str, str] = {}
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        kind = e.get("kind")
        if kind == "bbox" and e.get("id"):
            boxes[e["id"]] = e
        elif kind == "library.match" and e.get("id"):
            matches[e["id"]] = e["library_id"]
    out: dict[str, dict] = {}
    for node_id, library_id in matches.items():
        b = boxes.get(node_id)
        if b is None:
            continue
        out[node_id] = {
            "bbox": BoundingBox(origin=tuple(b["origin"]), dimensions=tuple(b["dimensions"])),
            "orientation": int(b.get("orientation") or 0),
            "library_id": library_id,
        }
    return out


def _rebake_cell(cell_dir: Path, *, dry_run: bool) -> tuple[int, int]:
    events_path = cell_dir / "events.jsonl"
    objs_dir = cell_dir / "objects"
    if not events_path.exists() or not objs_dir.is_dir():
        return (0, 0)
    done = skipped = 0
    for node_id, p in _placements(events_path).items():
        glb = objs_dir / f"{node_id}.glb"
        if not glb.exists():
            continue
        asset = library.asset_path(p["library_id"])
        bounds = library.asset_rotated_bounds(p["library_id"], p["orientation"])
        if not asset.exists() or bounds is None:
            skipped += 1  # bounds_missing / asset gone: matches the live fallback
            continue
        if dry_run:
            done += 1
            continue
        rmin, rmax = glb_place.rotate_aabb(bounds[0], bounds[1], V5_UPRIGHT_ROTATION)
        glb_place.place_glb(
            src=asset,
            dst=glb,
            bbox=p["bbox"],
            orientation=p["orientation"],
            rotated_min=rmin,
            rotated_max=rmax,
            model_rotation=V5_UPRIGHT_ROTATION,
        )
        done += 1
    return (done, skipped)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", help="single run name (default: every v5 run)")
    ap.add_argument(
        "--oneshot-dir",
        default=os.environ.get("STARSHOT_ONESHOT_DIR", "./oneshot-runs"),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.oneshot_dir)
    if not root.is_dir():
        print(f"oneshot dir not found: {root}")
        sys.exit(1)

    run_dirs = [root / args.run] if args.run else sorted(p for p in root.iterdir() if p.is_dir())
    total_done = total_skip = total_runs = 0
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            print(f"no such run: {run_dir.name}")
            continue
        if _read_version(run_dir) != V5_VERSION:
            continue
        total_runs += 1
        for events_path in sorted(run_dir.glob("*/*/events.jsonl")):
            cell = events_path.parent
            done, skipped = _rebake_cell(cell, dry_run=args.dry_run)
            if done or skipped:
                rel = cell.relative_to(root)
                print(f"  {rel}: {done} re-baked, {skipped} skipped (no bounds)")
            total_done += done
            total_skip += skipped

    verb = "would re-bake" if args.dry_run else "re-baked"
    print(f"\n{total_runs} v5 run(s): {verb} {total_done} assets, {total_skip} skipped")


if __name__ == "__main__":
    main()
