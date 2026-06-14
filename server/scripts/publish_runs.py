"""Publish generated cells to Cloudflare R2 + the D1 catalog.

Thin CLI over app.services.publish.publish_cell — handy for pushing the existing
runs into the new per-(run/slot/model/version) key layout in one go. Needs the
R2 + D1 credentials in the environment (see .env.example).

Usage (from server/):
  uv run python scripts/publish_runs.py --cell good_opus_new_suburb/modern-house/opus-new
  uv run python scripts/publish_runs.py --cell <run/slot/model> --version 2
  uv run python scripts/publish_runs.py --all
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SERVER_DIR))
load_dotenv()

from app.services import publish as publish_svc  # noqa: E402

_REPO_ROOT = _SERVER_DIR.parent
RUNS_DIR = Path(os.environ.get("STARSHOT_RUNS_DIR", str(_REPO_ROOT / "runs"))).resolve()


def _iter_cells(runs_dir: Path) -> list[tuple[str, str, str]]:
    """Every (run, slot, model) under the runs tree that has generated meshes."""
    cells: list[tuple[str, str, str]] = []
    if not runs_dir.is_dir():
        return cells
    for run in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        for slot in sorted(p for p in run.iterdir() if p.is_dir()):
            for model in sorted(p for p in slot.iterdir() if p.is_dir()):
                if (model / "generated").is_dir():
                    cells.append((run.name, slot.name, model.name))
    return cells


async def _run(args: argparse.Namespace) -> int:
    if args.all:
        cells = _iter_cells(RUNS_DIR)
    elif args.cell:
        parts = args.cell.strip("/").split("/")
        if len(parts) != 3:
            sys.exit("--cell must be run/slot/model")
        cells = [(parts[0], parts[1], parts[2])]
    else:
        sys.exit("pass --cell run/slot/model or --all")

    if not cells:
        print("[publish] no cells to publish")
        return 0

    failures = 0
    for run, slot, model in cells:
        try:
            rec = await publish_svc.publish_cell(
                RUNS_DIR, run, slot, model, args.version, dry_run=args.dry_run
            )
            tag = "[plan]" if args.dry_run else "[publish]"
            print(
                f"{tag} {run}/{slot}/{model}: "
                f"{rec['pano_count']} panos, proxy={'y' if rec['proxy_key'] else 'n'} "
                f"-> {rec['preview_key']}"
            )
        except Exception as e:  # keep going across a batch
            failures += 1
            print(f"[publish] FAIL {run}/{slot}/{model}: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"[publish] done: {len(cells) - failures}/{len(cells)} published")
    return 1 if failures else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish generated cells to R2 + D1.")
    ap.add_argument("--cell", help="run/slot/model to publish")
    ap.add_argument("--all", action="store_true", help="publish every cell with generated meshes")
    ap.add_argument(
        "--version",
        default=None,
        help="generated build to bake the dollhouse from (default: the tour's build, else latest)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print planned keys; no upload / D1 write")
    raise SystemExit(asyncio.run(_run(ap.parse_args())))


if __name__ == "__main__":
    main()
