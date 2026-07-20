"""Per-run board-summary cache, so switching runs doesn't re-parse every cell's
`events.jsonl`.

The board (status, event count, per-model pricing) is a fold over each cell's
event log — cheap to hold in memory, but expensive to REBUILD when a run is
activated, because the biggest logs are gigabytes. This caches the fold result
in one small SQLite per run (`<run>/board.db`), keyed by the cell's scene and
stamped with the event log's `(size, mtime)`. On activation the board reads the
cache instead of parsing the log; a cell whose log has since grown/shrunk fails
the freshness check and falls back to a one-time parse (which then refreshes the
cache).

Purely a cache: a miss or any SQLite fault just means "recompute from the log",
so a corrupt/absent `board.db` is never fatal — it self-heals on the next write.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cell_summary (
  scene TEXT PRIMARY KEY,
  size INTEGER,
  mtime REAL,
  summary TEXT
);
"""


def _db(run_dir: Path) -> Path:
    return run_dir / "board.db"


def file_sig(events_path: Path) -> tuple[int, float] | None:
    """`(size, mtime)` of an events log, the cache freshness key, or None if the
    file doesn't exist yet (an untouched cell — nothing to cache)."""
    try:
        st = events_path.stat()
    except OSError:
        return None
    return st.st_size, st.st_mtime


def load_all(run_dir: Path) -> dict[str, tuple[tuple[int, float], dict[str, Any]]]:
    """Every cached cell summary for a run in ONE read: `{scene: ((size, mtime),
    summary)}`. The board loads this once per poll and freshness-checks each cell
    against its current log stat, so activating a run costs one SQLite open — not
    one per cell. An absent/faulty DB → `{}` (every cell falls back to compute)."""
    db = _db(run_dir)
    if not db.exists():
        return {}
    out: dict[str, tuple[tuple[int, float], dict[str, Any]]] = {}
    try:
        con = sqlite3.connect(db, timeout=5.0)
        try:
            rows = con.execute("SELECT scene, size, mtime, summary FROM cell_summary").fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return {}
    for scene, size, mtime, summary in rows:
        try:
            out[scene] = ((size, mtime), json.loads(summary))
        except (TypeError, ValueError):
            continue
    return out


def put(run_dir: Path, scene: str, sig: tuple[int, float], summary: dict[str, Any]) -> None:
    """Upsert `scene`'s summary stamped with the log `sig`. Best-effort: a write
    fault is swallowed (the board still served the freshly-computed value; the
    next write retries)."""
    size, mtime = sig
    payload = json.dumps(summary, ensure_ascii=False)
    with contextlib.suppress(sqlite3.Error, OSError):
        run_dir.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(_db(run_dir), timeout=5.0)
        try:
            con.executescript(_SCHEMA)
            con.execute(
                "INSERT INTO cell_summary (scene, size, mtime, summary) VALUES (?,?,?,?) "
                "ON CONFLICT(scene) DO UPDATE SET size=excluded.size, mtime=excluded.mtime, "
                "summary=excluded.summary",
                (scene, size, mtime, payload),
            )
            con.commit()
        finally:
            con.close()
