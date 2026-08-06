"""Bring every run's SQLite stores in line with its `events.jsonl` (the source of
truth), so historical pricing is accurate and the board + Flights dashboard + the
spend cap all work on old runs.

Per source cell (`<run>/<slot>/<model>/events.jsonl`), in ONE streaming pass:

  1. **events.jsonl** — append a token-priced `llm.cost` for each compat-model
     call (`kimi-k3`/`qwen3.8-max-preview`/`longcat`/...) that lacks one. This is
     what `_cell_spend` (the spend cap) and the cost tracker actually read.
     Idempotent by the call's content-hash `key`.
  2. **flights.db** — the per-request ledger the Flights dashboard reads. The
     SQLite ledger was added mid-history, so most cells have no `flights.db` (or
     miss their earliest calls). Reconstruct the winning-attempt row for every
     call that has none (metadata + compat cost; the heavy prompt columns stay in
     events.jsonl), and price any existing unpriced compat rows. Deduped by
     `generation_id` / `t_request`, so re-running never duplicates.
  3. **board.db** — recompute the cached board summary (status / #events /
     pricing) `api/routes.py` reads on run activation.

Branch/sim dirs and unknown cells are skipped (only real `slot/model` cells).
Additive + idempotent; safe to re-run. `--dry-run` reports without writing.

Usage (from server/):
    uv run python scripts/backfill_scenes.py "KIMI K3 TEST"
    uv run python scripts/backfill_scenes.py --all
    uv run python scripts/backfill_scenes.py --all --dry-run
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core import slots
from app.core.slots import MODEL_ALIASES, SLOTS_BY_ID
from app.utils import boardcache, cellsummary, flightlog


def _load_slim(path: Path) -> tuple[list[dict], set[str], int]:
    """Stream the log into slim event dicts (via the shared bounded reader).
    Returns `(events, keyed_cost_keys, count)` — `keyed_cost_keys` are calls that
    already carry a token-priced `llm.cost` (so we don't double-append); `count`
    is the next free event index for anything appended (matching `SlotLog.log`)."""
    events: list[dict] = []
    keyed: set[str] = set()
    for e in cellsummary.iter_events(path):
        events.append(e)
        if e.get("kind") == "llm.cost" and isinstance(e.get("key"), str):
            keyed.add(e["key"])
    return events, keyed, len(events)


def _append_costs(path: Path, rows: list[tuple], start_index: int) -> None:
    """Append `llm.cost` events (matching the live shape) after ensuring a
    trailing newline so the first line can't fuse onto a partial final line."""
    if path.stat().st_size:
        with path.open("rb") as f:
            f.seek(-1, 2)
            needs_nl = f.read(1) != b"\n"
    else:
        needs_nl = False
    with path.open("a", encoding="utf-8") as f:
        if needs_nl:
            f.write("\n")
        for i, (key, ts, cost, model, ti, to) in enumerate(rows):
            f.write(json.dumps({
                "index": start_index + i, "ts": ts, "kind": "llm.cost",
                "key": key, "cost": cost, "model": model,
                "tokens_in": ti, "tokens_out": to,
            }) + "\n")


def backfill_cell(events_path: Path, runs_dir: Path, run_dir: Path, dry_run: bool) -> dict:
    scene = events_path.parent.relative_to(run_dir).as_posix()          # slot/model
    slot_full = events_path.parent.relative_to(runs_dir).as_posix()      # run/slot/model
    events, keyed, count = _load_slim(events_path)

    # 1. events.jsonl — append missing compat llm.cost.
    to_append: list[tuple] = []
    for e in events:
        if e.get("kind") != "cache.llm":
            continue
        cost = slots.token_cost(e.get("model"), e.get("tokens_in"), e.get("tokens_out"))
        key = e.get("key")
        if cost is not None and isinstance(key, str) and key not in keyed:
            to_append.append((key, e.get("ts") or time.time(), cost,
                              e.get("model"), e.get("tokens_in"), e.get("tokens_out")))
    if to_append and not dry_run:
        _append_costs(events_path, to_append, start_index=count)
    # Reflect the (to-be-)appended costs in memory so the flights/board folds +
    # the reported spend are accurate this pass (in dry-run this stays in-memory).
    for key, ts, cost, model, ti, to in to_append:
        events.append({"kind": "llm.cost", "key": key, "cost": cost, "ts": ts,
                       "model": model, "tokens_in": ti, "tokens_out": to})

    # 2. flights.db — reconstruct/price the request ledger (shared with the live
    # dashboard's legacy-scene fallback in flightlog).
    flights_added, flights_priced = flightlog.reconstruct_flights(
        slot_full, events, dry_run=dry_run
    )

    # 3. board.db — recompute the cached summary (log now includes the new costs).
    if not dry_run:
        sig = boardcache.file_sig(events_path)
        if sig is not None:
            boardcache.put(run_dir, scene, sig, cellsummary.summarize(events))

    result = {
        "scene": scene, "costs": len(to_append),
        "flights_added": flights_added, "flights_priced": flights_priced,
        "spend": cellsummary.cell_spend(events),
    }
    del events, keyed
    gc.collect()  # release a giant cell's parse before the next one
    return result


def backfill_run(run_dir: Path, runs_dir: Path, dry_run: bool) -> None:
    print(f"== {run_dir.name} ==")
    total = {"cells": 0, "costs": 0, "flights_added": 0, "flights_priced": 0, "spend": 0.0}
    for events_path in sorted(run_dir.rglob("events.jsonl")):
        scene = events_path.parent.relative_to(run_dir).as_posix()
        parts = scene.split("/")
        if len(parts) != 2 or parts[0] not in SLOTS_BY_ID or parts[1] not in MODEL_ALIASES:
            continue  # branch/sim or unknown dir
        r = backfill_cell(events_path, runs_dir, run_dir, dry_run)
        total["cells"] += 1
        total["costs"] += r["costs"]
        total["flights_added"] += r["flights_added"]
        total["flights_priced"] += r["flights_priced"]
        total["spend"] += r["spend"]
        print(f"  {r['scene']}: +{r['costs']} llm.cost, +{r['flights_added']} flight rows, "
              f"{r['flights_priced']} priced, ${r['spend']:.4f}")
    print(f"  -> {total['cells']} cell(s): +{total['costs']} costs, "
          f"+{total['flights_added']} flight rows, {total['flights_priced']} priced, "
          f"${total['spend']:.2f}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill SQLite stores for a run from its event logs.")
    parser.add_argument("run", nargs="?", help="run name under the runs dir")
    parser.add_argument("--all", action="store_true", help="process every run")
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args()

    runs_dir = flightlog._runs_dir()
    if args.all:
        run_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir())
    elif args.run:
        run_dirs = [runs_dir / args.run]
    else:
        parser.error("provide a run name or --all")
        return 2

    tag = " [DRY RUN]" if args.dry_run else ""
    print(f"Backfilling {len(run_dirs)} run(s){tag}\n")
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            print(f"no such run dir: {run_dir}", file=sys.stderr)
            continue
        backfill_run(run_dir, runs_dir, args.dry_run)
    print(f"Done{tag}.")
    if not args.dry_run:
        print("Restart the server so it re-hydrates the updated logs + caches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
