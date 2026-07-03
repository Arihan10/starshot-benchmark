"""Repair event logs whose stored `index` field has drifted from the event's
actual position in the log (`events[i]["index"] != i`).

Every consumer that treats `index` as an array position relies on the invariant
`events[i].index == i`: the prompt lab's per-event fetch (`/step-event`), rewind
(`truncate_events_to`), and branch forking. A race in the cost backfill — a
`SlotLog` swapped out mid-sweep (reset / restart / A/B launch) and then written
through its stale object, whose `index` counter is frozen at the old length —
could interleave `llm.cost` events at out-of-position indices and break the
invariant, surfacing as "event is not an LLM call" in the prompt lab.

This rewrites each drifted `events.jsonl` in place, setting every event's
`index` to its line position (order preserved; nothing added or removed). Clean
files are left untouched, so it's safe and idempotent. A `.bak` is dropped
beside each file it rewrites (first run only).

The running server keeps its in-memory copy of a log until it re-hydrates, so
RESTART the server after repairing (and prefer to run this with it stopped, so
a live writer can't race the rewrite).

Usage (from server/):
    uv run python scripts/reindex_events.py <events.jsonl | slot_dir | run_dir>
    uv run python scripts/reindex_events.py --runs-dir ../runs   # scan + fix all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: Path) -> list[dict] | None:
    """Parse an events.jsonl into event dicts, or None if any non-blank line
    fails to parse — we never rewrite a file we can't fully read (better to skip
    it loudly than risk dropping events)."""
    events: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  ! {path}: unparseable line {lineno} ({e}); skipping file", file=sys.stderr)
            return None
    return events


def reindex_file(path: Path) -> int:
    """Rewrite `path` so each event's `index` equals its position. Returns the
    number of events whose `index` was corrected (0 = already clean, no write)."""
    events = _load(path)
    if events is None:
        return 0
    drift = sum(1 for i, e in enumerate(events) if e.get("index") != i)
    if drift == 0:
        return 0
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        bak.write_bytes(path.read_bytes())
    with path.open("w", encoding="utf-8") as f:
        for i, e in enumerate(events):
            e["index"] = i
            f.write(json.dumps(e) + "\n")
    return drift


def _collect(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.is_dir():
        print(f"no such path: {target}", file=sys.stderr)
        sys.exit(2)
    # A single slot dir holds events.jsonl directly; a run/runs dir holds many
    # beneath it (cells + `_branches/<id>/`), so recurse.
    if (target / "events.jsonl").is_file():
        return [target / "events.jsonl"]
    return sorted(target.rglob("events.jsonl"))


def main(target: Path) -> None:
    files = _collect(target)
    if not files:
        print(f"no events.jsonl under {target}", file=sys.stderr)
        sys.exit(2)
    fixed = 0
    for p in files:
        drift = reindex_file(p)
        if drift:
            fixed += 1
            print(f"  reindexed {drift} event(s): {p}")
    print(f"done: {fixed}/{len(files)} file(s) needed reindexing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Repair drifted event-log indices so index == position."
    )
    parser.add_argument(
        "target", nargs="?", type=Path, default=None,
        help="an events.jsonl, a slot dir, or a run dir",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=None,
        help="a runs directory; every events.jsonl beneath it is scanned + fixed",
    )
    args = parser.parse_args()
    target = args.target or args.runs_dir
    if target is None:
        parser.error("provide an events.jsonl / slot dir / run dir, or --runs-dir")
    main(target.resolve())
