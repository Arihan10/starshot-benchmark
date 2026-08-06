"""Migrate the legacy global `runs/flights.jsonl` into per-scene SQLite DBs.

Metadata only — prompts are captured going forward by the live pipeline and are
never backfilled from `events.jsonl` (the ledger is detached from the event
log). Idempotent: skips any scene whose `flights.db` already exists, so a re-run
never duplicates rows or clobbers live data. Run once, early.

Usage (from server/):
    uv run python scripts/migrate_flights_to_sqlite.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils import flightlog

_INSERT = (
    "INSERT INTO flights (t_request, t_response, flight_ms, transport, provider, base_url, "
    "model, kind, status, ok, error, exc_type, api_key, tokens_in, tokens_out, generation_id, "
    "slot, step, node, call, attempt) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)


def _values(scene: str, r: dict) -> tuple:
    return (
        r.get("t_request"), r.get("t_response"), r.get("flight_ms"),
        r.get("transport"), r.get("provider"), r.get("base_url"), r.get("model"), r.get("kind"),
        r.get("status"), 1 if r.get("ok") else 0, r.get("error"), r.get("exc_type"),
        r.get("key"), r.get("tokens_in"), r.get("tokens_out"), r.get("generation_id"),
        scene, r.get("step"), r.get("node"), r.get("call"), r.get("attempt"),
    )


def main() -> int:
    src = flightlog._runs_dir() / "flights.jsonl"
    if not src.exists():
        print(f"no {src} — nothing to migrate")
        return 0

    by_scene: dict[str, list[dict]] = {}
    null_rows = 0
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            scene = row.get("slot")
            if not scene:
                null_rows += 1
                continue
            by_scene.setdefault(scene, []).append(row)

    scenes = rows_migrated = skipped = 0
    for scene, rows in sorted(by_scene.items()):
        db = flightlog._scene_db(scene)
        if db.exists():
            skipped += 1
            continue
        db.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(db)
        try:
            con.executescript(flightlog._SCHEMA)
            con.executemany(_INSERT, [_values(scene, r) for r in rows])
            con.commit()
        finally:
            con.close()
        scenes += 1
        rows_migrated += len(rows)

    print(
        f"migrated {rows_migrated} row(s) into {scenes} new scene DB(s); "
        f"skipped {skipped} scene(s) with an existing flights.db, {null_rows} null-scene row(s)."
    )
    print("Prompts are not backfilled (detached from events.jsonl); new calls capture them.")
    print("You can archive/remove runs/flights.jsonl once satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
