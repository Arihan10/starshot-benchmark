"""One-time migration: fold every cell's pre-versioning from-scratch generated
build into the versioned layout `generated/1/`.

Before versioning, a cell's generated build sat DIRECTLY in the cell dir:
    <cell>/objects-generated/
    <cell>/objects-generated-optimized/
    <cell>/events.generated.jsonl
The versioned system keeps every build under `generated/<version>/`:
    <cell>/generated/1/objects-generated/
    <cell>/generated/1/objects-generated-optimized/
    <cell>/generated/1/events.generated.jsonl
so an existing build becomes version 1 and new builds are 2, 3, …. The running
server already does this lazily on first touch (see
`app.pipeline.generation.migrate_legacy_generated`, whose move logic this
mirrors); this script does it EAGERLY across every scene so they're all migrated
up front.

A "cell" is exactly `<run>/<slot>/<model>` — the same run/slot/model depth the
server itself enumerates (`run_dir.glob("*/*/events.jsonl")` in the API). Walking
by that fixed depth (not a recursive marker search) is deliberate: it never
mistakes a nested export/backup snapshot (e.g. `<cell>/_changed_export/` or
`<cell>/_dedup_backup/`, which hold the same generated-build layout) or a branch
temp log (`<run>/_branches/<id>/`) for a scene — those are left untouched.

Idempotent + non-destructive: a cell already on the versioned layout (has a
`generated/` dir) is left untouched, and each item is moved only when its
destination is absent, so re-running is safe. The library build (objects/ +
events.jsonl) is never touched. A cell that never had a generated build has
nothing to migrate — it's already on the new structure.

Run with the server STOPPED so a live build can't write into a dir mid-move.

Usage (from server/):
    uv run python scripts/migrate_generated_versions.py                  # default runs dir
    uv run python scripts/migrate_generated_versions.py --runs-dir ../runs
    uv run python scripts/migrate_generated_versions.py --run "KIMI K3 TEST"  # one run only
    uv run python scripts/migrate_generated_versions.py --dry-run        # preview only
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Layout constants — kept in sync with app.pipeline.generation. Duplicated (not
# imported) so this stays a pure, dependency-free folder move: importing
# generation pulls in trimesh + the mesh services, overkill for relocating dirs.
GENERATED_DIR = "generated"
RAW_SUBDIR = "objects-generated"
OPT_SUBDIR = "objects-generated-optimized"
EVENTS_NAME = "events.generated.jsonl"
_LEGACY_NAMES = (RAW_SUBDIR, OPT_SUBDIR, EVENTS_NAME)
_BRANCHES_DIR = "_branches"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RUNS_DIR = Path(os.environ.get("STARSHOT_RUNS_DIR", _REPO_ROOT / "runs"))


def _cells(runs_dir: Path, only_run: str | None = None) -> list[Path]:
    """Every pipeline cell under `runs_dir`, i.e. `<run>/<slot>/<model>`. Walks the
    fixed run/slot/model depth (the server's own cell shape) rather than searching
    recursively for markers, so nested export/backup snapshots and `_branches/`
    temp logs — which carry the same generated-build layout — are never treated as
    cells. `only_run` limits the sweep to a single run."""
    run_dirs = (
        [runs_dir / only_run]
        if only_run is not None
        else sorted(p for p in runs_dir.iterdir() if p.is_dir())
    )
    cells: list[Path] = []
    for run_dir in run_dirs:
        if not run_dir.is_dir() or run_dir.name == _BRANCHES_DIR:
            continue
        for slot_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            if slot_dir.name == _BRANCHES_DIR:  # <run>/_branches/<id> — not a cell
                continue
            cells.extend(sorted(p for p in slot_dir.iterdir() if p.is_dir()))
    return cells


def _legacy_items(cell: Path) -> list[Path]:
    """The pre-versioning generated-build items sitting DIRECTLY in `cell`."""
    return [cell / name for name in _LEGACY_NAMES if (cell / name).exists()]


def migrate_cell(cell: Path, *, dry_run: bool) -> tuple[str, list[str]]:
    """Fold `cell`'s legacy generated build into `generated/1/`. Returns
    (status, item_names):
      * "migrated" — the legacy build was (or, in dry run, would be) moved;
      * "conflict" — a `generated/` dir already exists beside the legacy items,
        so it's left for manual review (mirrors the runtime guard, which skips);
      * "empty"    — nothing to move (already versioned, or never generated).
    Each item is moved only when its destination slot is free, so a partial or
    repeated run never clobbers or nests wrongly."""
    items = _legacy_items(cell)
    if not items:
        return "empty", []
    root = cell / GENERATED_DIR
    if root.exists():
        return "conflict", [p.name for p in items]
    if dry_run:
        return "migrated", [p.name for p in items]
    dst = root / "1"
    dst.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for p in items:
        target = dst / p.name
        if p.exists() and not target.exists():
            shutil.move(str(p), str(target))
            moved.append(p.name)
    return "migrated", moved


def main(runs_dir: Path, *, only_run: str | None, dry_run: bool) -> None:
    if not runs_dir.is_dir():
        print(f"no such directory: {runs_dir}", file=sys.stderr)
        sys.exit(2)
    cells = _cells(runs_dir, only_run)
    migrated = conflicts = 0
    for cell in cells:
        status, items = migrate_cell(cell, dry_run=dry_run)
        label = cell.relative_to(runs_dir).as_posix()
        if status == "migrated":
            migrated += 1
            verb = "would migrate" if dry_run else "migrated"
            print(f"  {verb} {label} -> {GENERATED_DIR}/1/  ({', '.join(items)})")
        elif status == "conflict":
            conflicts += 1
            print(
                f"  ! skip {label}: legacy build sits beside an existing "
                f"{GENERATED_DIR}/ dir - resolve manually ({', '.join(items)})",
                file=sys.stderr,
            )
    tail = "  (dry run - nothing moved)" if dry_run else ""
    print(
        f"done: {migrated} migrated, {conflicts} conflict(s) skipped, "
        f"{len(cells)} cell(s) scanned{tail}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fold pre-versioning generated builds into generated/1/.",
    )
    parser.add_argument(
        "runs_dir", nargs="?", type=Path, default=None,
        help="the runs directory to scan (default: the app runs dir / $STARSHOT_RUNS_DIR)",
    )
    parser.add_argument(
        "--runs-dir", dest="runs_dir_opt", type=Path, default=None,
        help="alias for the positional runs dir",
    )
    parser.add_argument(
        "--run", default=None,
        help="limit the sweep to a single run (its top-level dir name under runs)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report what would move without touching disk",
    )
    args = parser.parse_args()
    runs_dir = args.runs_dir or args.runs_dir_opt or _DEFAULT_RUNS_DIR
    main(runs_dir.resolve(), only_run=args.run, dry_run=args.dry_run)
