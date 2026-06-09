"""Populate ``runs/`` from a shared ``runs_bundle/`` + the local asset library.

A teammate ships a ``runs_bundle/`` holding only the per-cell event logs
(``<run>/<slot>/<model>/events.jsonl``) — none of the heavy mesh artifacts.
This script reconstitutes those cells locally: it copies each log into the
runs dir and re-bakes the placement-baked ``objects-optimized/`` straight from
the log + the local optimized asset library (the same log->scene contract
``scripts/rebake_runs.py`` implements), so every cell renders in the viewer
exactly as the sender saw it.

Prereqs:
  * The optimized asset library at ``server/app/assets_library/assets-optimized/``
    (the ``.glb`` files + ``optimize_manifest.json``). The ``library_id``s the
    logs reference must exist there — i.e. the same library version the sender
    built against.
  * A ``runs_bundle/`` at the repo root (or pass ``--bundle``).

Usage (from server/):
  uv run python scripts/populate_from_bundle.py
  uv run python scripts/populate_from_bundle.py --bundle /path/to/runs_bundle
  uv run python scripts/populate_from_bundle.py --runs-dir /path/to/runs
  uv run python scripts/populate_from_bundle.py --force   # overwrite logs + re-bake
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SERVER_DIR = _SCRIPTS_DIR.parent
_REPO_ROOT = _SERVER_DIR.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Populate runs/ from a runs_bundle/ + the local asset library."
    )
    p.add_argument(
        "--bundle",
        type=Path,
        default=_REPO_ROOT / "runs_bundle",
        help="bundle of per-cell event logs (default: <repo>/runs_bundle)",
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(os.environ.get("STARSHOT_RUNS_DIR", str(_REPO_ROOT / "runs"))),
        help="destination runs dir (default: $STARSHOT_RUNS_DIR or <repo>/runs)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrite an already-present cell log and re-bake even existing objects",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    bundle = args.bundle.resolve()
    runs_dir = args.runs_dir.resolve()

    if not bundle.is_dir():
        print(f"[populate] bundle not found: {bundle}", file=sys.stderr)
        return 1
    logs = sorted(bundle.rglob("events.jsonl"))
    if not logs:
        print(f"[populate] no events.jsonl under {bundle}", file=sys.stderr)
        return 1

    print(f"[populate] bundle   : {bundle}")
    print(f"[populate] runs dir : {runs_dir}")
    print(f"[populate] cells    : {len(logs)}")

    # --- 1. place the logs (no heavy deps needed) -------------------------
    copied = skipped = 0
    cell_dirs: list[Path] = []
    for src in logs:
        rel = src.relative_to(bundle)  # <run>/<slot>/<model>/events.jsonl
        dst = runs_dir / rel
        cell_dirs.append(dst.parent)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not args.force:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    print(f"[populate] logs copied={copied} skipped_existing={skipped}")

    # --- 2. re-bake the placed cells onto the local library ---------------
    # Imported here (not at top) so the copy above still works outside the
    # server env; the rebake needs app.* on the path, which rebake_runs wires
    # up on import.
    try:
        sys.path.insert(0, str(_SCRIPTS_DIR))
        import rebake_runs
        from app.services import library
    except ImportError as e:
        print(
            "[populate] logs are in place, but the rebake pipeline could not be "
            "imported. Run me under the server env:\n"
            "  cd server && uv run python scripts/populate_from_bundle.py\n"
            f"  (import error: {e})",
            file=sys.stderr,
        )
        return 1

    manifest = library.ASSETS_DIR / "optimize_manifest.json"
    if not library.ASSETS_DIR.is_dir() or not manifest.is_file():
        print(
            f"[populate] WARNING: optimized asset library not found at "
            f"{library.ASSETS_DIR} (need the .glb files + optimize_manifest.json). "
            "Logs are placed, but no meshes can be baked until the library is present.",
            file=sys.stderr,
        )
    else:
        print(f"[populate] library  : {library.ASSETS_DIR}")

    baked = already = unconvertible = library_cells = 0
    # Re-bake only the cells this bundle carries — never the friend's other runs.
    for cell_dir in dict.fromkeys(cell_dirs):
        res = rebake_runs.rebake_cell(cell_dir, force=args.force, limit=10**9, prune=False)
        if res is None:
            continue  # not a library-mode cell — nothing to reconstruct
        library_cells += 1
        baked += res["baked"]
        already += res["already"]
        unconvertible += res["unconvertible"]

    print(
        f"\n[populate] done: {library_cells} library cells | "
        f"baked={baked} already={already} unconvertible={unconvertible}"
    )
    if unconvertible:
        print(
            "[populate] NOTE: some objects had no matching library asset/bounds — "
            "your runs_bundle was likely built against a different library version."
        )

    run_names = sorted({rel.parts[0] for rel in (p.relative_to(bundle) for p in logs)})
    print("\n[populate] runs now available in the viewer:")
    for name in run_names:
        print(f"  - {name}")
    print(
        "\nNext:\n"
        f"  cd server && STARSHOT_RUNS_DIR={runs_dir} uv run uvicorn app.main:app --port 8765\n"
        "  cd client && npm install && npm start   # open http://127.0.0.1:8766/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
