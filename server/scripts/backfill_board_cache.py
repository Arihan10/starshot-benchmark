"""Pre-warm the per-run board-summary cache (`<run>/board.db`).

The board (status, #events, pricing) is a fold over each cell's `events.jsonl`.
The server now reads that fold from `board.db` on run activation instead of
parsing the logs — but a run that predates the cache has no entries yet, so its
FIRST activation would still pay the full multi-GB parse (once) to populate them.
This does that parse offline so the first switch is already instant.

Idempotent: each cell is re-stamped with its log's current `(size, mtime)`, so
re-running just refreshes anything that changed. Reads each log with the heavy
prompt/reasoning fields projected away, so even a gigabyte log costs little RAM.

Usage (from server/):
    uv run python scripts/backfill_board_cache.py "KIMI K3 TEST"
    uv run python scripts/backfill_board_cache.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.slots import MODEL_ALIASES, SLOTS_BY_ID
from app.utils import boardcache, cellsummary, flightlog

# The only fields `cellsummary.summarize` reads. Projecting to them as each line
# is parsed keeps a giant log's system/user/output/reasoning out of memory.
_KEEP = (
    "kind", "cost", "generation_id", "key", "model",
    "tokens_in", "tokens_out", "cap", "node", "phase", "prompt",
)


def _load_slim(path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append({k: e.get(k) for k in _KEEP})
    return events


def warm_run(run_dir: Path) -> tuple[int, int]:
    """Populate `run_dir/board.db` for every source cell. Returns
    `(cells_warmed, cells_skipped)` — branch/unknown dirs are skipped (the board
    reads board.db only for `slot/model` cells; branches auto-load)."""
    warmed = skipped = 0
    for events_path in sorted(run_dir.rglob("events.jsonl")):
        rel = events_path.parent.relative_to(run_dir).as_posix()
        parts = rel.split("/")
        if len(parts) != 2 or parts[0] not in SLOTS_BY_ID or parts[1] not in MODEL_ALIASES:
            skipped += 1
            continue
        sig = boardcache.file_sig(events_path)
        if sig is None:
            skipped += 1
            continue
        summary = cellsummary.summarize(_load_slim(events_path))
        boardcache.put(run_dir, rel, sig, summary)
        warmed += 1
        print(f"  {rel}: {summary['events_count']} events, ${summary['spend']:.4f}")
    return warmed, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-warm the board-summary cache for a run.")
    parser.add_argument("run", nargs="?", help="run name under the runs dir")
    parser.add_argument("--all", action="store_true", help="warm every run under the runs dir")
    args = parser.parse_args()

    runs_dir = flightlog._runs_dir()
    if args.all:
        run_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir())
    elif args.run:
        run_dirs = [runs_dir / args.run]
    else:
        parser.error("provide a run name or --all")
        return 2

    total_warmed = 0
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            print(f"no such run dir: {run_dir}", file=sys.stderr)
            continue
        print(f"== {run_dir.name} ==")
        warmed, skipped = warm_run(run_dir)
        total_warmed += warmed
        print(f"  -> warmed {warmed} cell(s), skipped {skipped}\n")
    print(f"Done. Warmed {total_warmed} cell(s) across {len(run_dirs)} run(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
