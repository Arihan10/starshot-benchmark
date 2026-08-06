"""Backfill token-priced costs for the OpenAI-compatible backends across a run.

OpenRouter returns each call's settled cost; the compat backends (Moonshot,
Alibaba, ...) don't — so their spend was never tracked. Now that `slots` carries
a per-1M-token price for them, this reprices the historical logs of a run off
its recorded token counts, into BOTH stores the live pipeline now writes:

  * **flights.db** — adds the `cost` column (if the DB predates it) and fills it
    for every statically-priced flight row (`slots.token_cost`), leaving
    OpenRouter rows NULL (their cost lives in the event log, as before).
  * **events.jsonl** — appends one `llm.cost` per statically-priced `cache.llm`,
    keyed by the call's content-hash `key` (the external analogue of an
    OpenRouter `generation_id`). `_usage_summary` prefers this key-join over any
    OpenRouter cost for the same call, so the cost tracker settles at the true
    token price.

The Kimi K3 hiccup this exists for: the run billed kimi-k3 through TWO transports
— the direct Moonshot API (no generation_id) AND OpenRouter BYOK (a real
generation_id but a $0 settled cost). Pricing is matched on the MODEL, so both
legs are repriced at the Moonshot rate regardless of how they were routed, and
the key-join overrides the misleading $0 BYOK `llm.cost` events already in the log.

Idempotent: flights UPDATEs recompute the same value, and an event whose `key`
already carries a keyed `llm.cost` is skipped — so re-running never duplicates.
Streams the event logs line by line (they can be gigabytes) and appends only, so
no multi-GB rewrite. Run with the server STOPPED so no live writer races the append.

Usage (from server/):
    uv run python scripts/backfill_external_costs.py "KIMI K3 TEST"
    uv run python scripts/backfill_external_costs.py "KIMI K3 TEST" --dry-run
    uv run python scripts/backfill_external_costs.py --runs-dir ../runs "KIMI K3 TEST"
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core import slots
from app.utils import flightlog

_PROGRESS_EVERY = 1_000_000  # heartbeat line cadence when streaming a big log


def _fmt_usd(v: float) -> str:
    return f"${v:,.4f}"


# --- flights.db ---------------------------------------------------------------


def backfill_flights(db: Path, *, dry_run: bool) -> tuple[int, float]:
    """Price every statically-priced row in one scene DB. Returns
    `(rows_priced, total_cost)`. Adds the `cost` column first if the DB predates
    it (unless dry-run, which never writes)."""
    con = sqlite3.connect(db)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(flights)")}
        if "cost" not in cols and not dry_run:
            con.execute("ALTER TABLE flights ADD COLUMN cost REAL")
        models = [
            r[0] for r in con.execute(
                "SELECT DISTINCT model FROM flights WHERE model IS NOT NULL"
            )
        ]
        rows_priced = 0
        total = 0.0
        for model in models:
            pricing = slots.model_pricing(model)
            if pricing is None:
                continue
            price_in, price_out = pricing
            cost_expr = (
                "(COALESCE(tokens_in,0)/1000000.0*?) + (COALESCE(tokens_out,0)/1000000.0*?)"
            )
            where = "model = ? AND (tokens_in IS NOT NULL OR tokens_out IS NOT NULL)"
            agg = con.execute(
                f"SELECT COUNT(*), COALESCE(SUM({cost_expr}),0) FROM flights WHERE {where}",
                (price_in, price_out, model),
            ).fetchone()
            rows_priced += agg[0]
            total += agg[1]
            if not dry_run:
                con.execute(
                    f"UPDATE flights SET cost = {cost_expr} WHERE {where}",
                    (price_in, price_out, model),
                )
        if not dry_run:
            con.commit()
        return rows_priced, total
    finally:
        con.close()


# --- events.jsonl -------------------------------------------------------------


def _scan_events(path: Path) -> tuple[int, int, list[tuple[str, float, float]]]:
    """One streaming pass over an event log. Returns `(total_events, max_index,
    to_price)` where `to_price` is one `(key, ts, cost)` per statically-priced
    `cache.llm` that lacks a keyed `llm.cost` — the exact rows to append. Only
    the small cost/call events are JSON-parsed; every non-blank line is counted
    so the appended indices continue the log's position sequence."""
    total = 0
    max_index = -1
    priced_keys: set[str] = set()  # keys that already carry a keyed llm.cost
    calls: dict[str, tuple[float, float]] = {}  # key -> (ts, cost) for priced compat calls
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            if total % _PROGRESS_EVERY == 0:
                print(f"    …scanned {total:,} events")
            if "cache.llm" not in line and "llm.cost" not in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = e.get("kind")
            idx = e.get("index")
            if isinstance(idx, int) and idx > max_index:
                max_index = idx
            if kind == "llm.cost":
                key = e.get("key")
                if isinstance(key, str):
                    priced_keys.add(key)
            elif kind == "cache.llm":
                key = e.get("key")
                cost = slots.token_cost(
                    e.get("model"), e.get("tokens_in"), e.get("tokens_out")
                )
                if isinstance(key, str) and cost is not None:
                    calls[key] = (e.get("ts") or time.time(), cost)
    to_price = [
        (key, ts, cost) for key, (ts, cost) in calls.items() if key not in priced_keys
    ]
    return total, max_index, to_price


def _append_costs(
    path: Path, rows: list[tuple[str, float, float]], *, start_index: int
) -> None:
    """Append the `llm.cost` events, matching the live path's shape. Ensures the
    file ends with a newline first so the first append can't fuse onto a partial
    final line."""
    if path.stat().st_size:
        with path.open("rb") as f:
            f.seek(-1, 2)
            needs_nl = f.read(1) != b"\n"
    else:
        needs_nl = False
    with path.open("a", encoding="utf-8") as f:
        if needs_nl:
            f.write("\n")
        for i, (key, ts, cost) in enumerate(rows):
            event = {
                "index": start_index + i,
                "ts": ts,
                "kind": "llm.cost",
                "key": key,
                "cost": cost,
            }
            f.write(json.dumps(event) + "\n")


def backfill_events(path: Path, *, dry_run: bool) -> tuple[int, float]:
    """Append a keyed `llm.cost` for each unpriced statically-priced call in one
    event log. Returns `(events_appended, total_cost)`."""
    total, max_index, to_price = _scan_events(path)
    total_cost = sum(cost for _, _, cost in to_price)
    if to_price and not dry_run:
        # Appended events take the next free positions; guard the rare pre-existing
        # upward index drift (a backfill race — see reindex_events.py) so a new
        # index can't collide with a surviving one.
        start = max(total, max_index + 1)
        _append_costs(path, to_price, start_index=start)
    return len(to_price), total_cost


# --- driver -------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill token-priced compat-backend costs for a run's logs + flights DBs."
    )
    parser.add_argument("run", help="run name under the runs dir, e.g. 'KIMI K3 TEST'")
    parser.add_argument("--runs-dir", type=Path, default=None, help="override the runs dir")
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would change; write nothing"
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir.resolve() if args.runs_dir else flightlog._runs_dir()
    run_dir = runs_dir / args.run
    if not run_dir.is_dir():
        print(f"no such run dir: {run_dir}", file=sys.stderr)
        return 2

    tag = " [DRY RUN]" if args.dry_run else ""
    print(f"Backfilling external costs for {run_dir}{tag}\n")

    dbs = sorted(run_dir.rglob("flights.db"))
    print(f"== flights.db ({len(dbs)}) ==")
    flight_rows = 0
    flight_cost = 0.0
    for db in dbs:
        rows, cost = backfill_flights(db, dry_run=args.dry_run)
        flight_rows += rows
        flight_cost += cost
        rel = db.relative_to(runs_dir).as_posix()
        print(f"  {rel}: priced {rows:,} row(s), {_fmt_usd(cost)}")
    print(f"  -> {flight_rows:,} row(s), {_fmt_usd(flight_cost)}\n")

    logs = sorted(run_dir.rglob("events.jsonl"))
    print(f"== events.jsonl ({len(logs)}) ==")
    ev_appended = 0
    ev_cost = 0.0
    for log in logs:
        rel = log.relative_to(runs_dir).as_posix()
        print(f"  scanning {rel} …")
        appended, cost = backfill_events(log, dry_run=args.dry_run)
        ev_appended += appended
        ev_cost += cost
        verb = "would append" if args.dry_run else "appended"
        print(f"    {verb} {appended:,} llm.cost event(s), {_fmt_usd(cost)}")
    print(f"  -> {ev_appended:,} event(s), {_fmt_usd(ev_cost)}\n")

    print(
        f"Done{tag}. flights: {_fmt_usd(flight_cost)} across {flight_rows:,} row(s); "
        f"events: {_fmt_usd(ev_cost)} across {ev_appended:,} new cost event(s)."
    )
    if not args.dry_run:
        print("Restart the server so it re-hydrates the updated event logs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
