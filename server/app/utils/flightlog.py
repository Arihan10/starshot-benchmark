"""First-party LLM-request ledger, stored in per-scene SQLite databases.

OpenRouter shows its own traffic on their dashboard; the third-party
OpenAI-compatible backends (Moonshot, LongCat, SiliconFlow, the dLLM gateway)
show nothing. This ledger records BOTH, so the whole picture — plus the exact
system/user prompt and model output — lives locally.

Architecture:

  * **One SQLite DB per scene** at `<runs>/<run>/<slot>/<model>/flights.db`
    (branches: `<run>/_branches/<bid>/flights.db`). The scene key is exactly
    the composite id `rlog.current_slot_id()` returns, which is also the DB's
    path under the runs dir — so the `slot` column doubles as the scene id.
    The DB is co-located with the cell, so cell copy/reset/delete (all
    directory-level) carry or drop it for free.
  * **Fully detached** from `events.jsonl`: prompts are written straight into
    the scene DB (`attach_prompt`), never read back out of the event log.
  * **Unified reads via ATTACH**: a run's scene DBs are attached (in batches
    under SQLite's 10-attach limit) and `UNION ALL`-queried as one entity;
    results are merge-sorted across batches. One code path serves runs of any
    size — a small run is just a single batch.

Capture is deliberately thin at the call sites: `begin_call` binds a per-task
context (ContextVar), `record` writes one row per HTTP attempt (open-write-
close, so no handle lingers to lock the file on Windows teardown), and a single
`attach_prompt` on success fills the winning attempt's prompt columns. Reads
(`page` / `detail` / `facets`) are separate and used only by the API layer.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
import sqlite3
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.utils import logging as rlog

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ERROR_MAX = 500
_ATTACH_BATCH = 8  # headroom under SQLite's hard limit of 10 attached DBs

_SCHEMA = """
CREATE TABLE IF NOT EXISTS flights (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  t_request REAL, t_response REAL, flight_ms INTEGER,
  transport TEXT, provider TEXT, base_url TEXT, model TEXT, kind TEXT,
  status INTEGER, ok INTEGER, error TEXT, exc_type TEXT,
  api_key TEXT, tokens_in INTEGER, tokens_out INTEGER, generation_id TEXT,
  slot TEXT, step TEXT, node TEXT, call INTEGER, attempt INTEGER,
  system TEXT, user TEXT, output TEXT, reasoning TEXT, schema TEXT
);
CREATE INDEX IF NOT EXISTS idx_flights_page ON flights (t_response DESC, id DESC);
"""

# Metadata projection (never the heavy prompt columns). `api_key` is surfaced as
# the masked `key` the client already renders; `slot` IS the scene id.
_LIST_COLS = (
    "id, t_request, t_response, flight_ms, transport, provider, base_url, model, kind, "
    "status, ok, error, exc_type, api_key AS key, tokens_in, tokens_out, generation_id, "
    "slot, step, node, call, attempt, (system IS NOT NULL) AS has_prompt"
)

# Facet key -> the SQL expression its distinct values group on.
_STATUS_EXPR = "CASE WHEN status IS NOT NULL THEN CAST(status AS TEXT) ELSE COALESCE(exc_type, 'error') END"
_FACET_EXPR: dict[str, str] = {
    "transport": "transport",
    "status": _STATUS_EXPR,
    "kind": "kind",
    "model": "model",
    "provider": "provider",
    "slot": "slot",
    "step": "step",
    "key": "api_key",
}
# Facet key -> the column a WHERE filter on it targets.
_FILTER_COL: dict[str, str] = {
    "transport": "transport", "kind": "kind", "model": "model", "provider": "provider",
    "slot": "slot", "step": "step", "key": "api_key",
}

_call_counter = itertools.count(1)
_call_ctx: ContextVar[dict[str, Any] | None] = ContextVar("_flight_call_ctx", default=None)
_subscribers: list[asyncio.Queue[dict[str, Any]]] = []
_initialized: set[str] = set()


def _runs_dir() -> Path:
    return Path(os.environ.get("STARSHOT_RUNS_DIR", _REPO_ROOT / "runs"))


def _scene_db(scene: str) -> Path:
    return _runs_dir() / scene / "flights.db"


def _mask(key: str) -> str:
    return f"...{key[-4:]}" if len(key) >= 6 else "..."


def _connect(path: Path, *, create: bool = False) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=5.0)
    con.row_factory = sqlite3.Row
    if create:
        con.executescript(_SCHEMA)
    return con


# --- capture ------------------------------------------------------------------


def begin_call(*, step: str | None = None, node: str | None = None, kind: str = "structured") -> None:
    """Bind a fresh logical-call context to the current task. Every `record`
    issued while it's bound shares the call id and a 1-based attempt number, so a
    retry storm groups back to the one call that caused it."""
    _call_ctx.set({"id": next(_call_counter), "step": step, "node": node, "kind": kind,
                   "seq": 0, "last": None})


def last_flight() -> dict[str, Any] | None:
    """The most recent row recorded under this task's call — the final (winning)
    attempt. Lets `call_llm` stamp flight timing onto its `cache.llm` event."""
    ctx = _call_ctx.get()
    return ctx["last"] if ctx else None


def record(
    *,
    transport: str,
    model: str,
    t_request: float,
    t_response: float,
    base_url: str | None = None,
    api_key: str | None = None,
    status: int | None = None,
    error: str | None = None,
    exc_type: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    generation_id: str | None = None,
    attempt: int | None = None,
    kind: str | None = None,
) -> dict[str, Any]:
    """Persist one HTTP-attempt row to the current scene's DB and fan it out to
    live SSE subscribers. Calls with no bound scene (the prompt-lab sandbox) are
    returned but not persisted — there's no scene to own them."""
    ctx = _call_ctx.get()
    if attempt is None and ctx is not None:
        ctx["seq"] += 1
        attempt = ctx["seq"]
    scene = rlog.current_slot_id()
    provider = "openrouter" if transport == "openrouter" else (urlparse(base_url).hostname if base_url else None)
    masked = _mask(api_key) if api_key else None
    ok = 1 if (status is not None and 200 <= status < 300 and error is None) else 0
    err = error[:_ERROR_MAX] if error else None
    kind = kind or (ctx["kind"] if ctx else "structured")
    step = ctx["step"] if ctx else None
    node = ctx["node"] if ctx else None
    call = ctx["id"] if ctx else None
    t_req, t_res = round(t_request, 3), round(t_response, 3)
    flight_ms = max(0, int((t_response - t_request) * 1000))

    row_id: int | None = None
    if scene:
        row_id = _insert(scene, (
            t_req, t_res, flight_ms, transport, provider, base_url, model, kind,
            status, ok, err, exc_type, masked, tokens_in, tokens_out, generation_id,
            scene, step, node, call, attempt,
        ))

    event: dict[str, Any] = {
        "id": row_id, "slot": scene, "t_request": t_req, "t_response": t_res,
        "flight_ms": flight_ms, "transport": transport, "provider": provider,
        "base_url": base_url, "model": model, "kind": kind, "status": status,
        "ok": bool(ok), "error": err, "exc_type": exc_type, "key": masked,
        "tokens_in": tokens_in, "tokens_out": tokens_out, "generation_id": generation_id,
        "step": step, "node": node, "call": call, "attempt": attempt, "has_prompt": False,
    }
    if ctx is not None:
        ctx["last"] = event
    if scene:
        for q in _subscribers:
            q.put_nowait(event)
    return event


def _insert(scene: str, values: tuple[Any, ...]) -> int | None:
    path = _scene_db(scene)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        con = _connect(path, create=scene not in _initialized)
    except (sqlite3.Error, OSError):
        return None  # a logging fault must never break the LLM call it describes
    try:
        _initialized.add(scene)
        cur = con.execute(
            "INSERT INTO flights (t_request, t_response, flight_ms, transport, provider, "
            "base_url, model, kind, status, ok, error, exc_type, api_key, tokens_in, "
            "tokens_out, generation_id, slot, step, node, call, attempt) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            values,
        )
        con.commit()
        return cur.lastrowid
    except sqlite3.Error:
        return None
    finally:
        con.close()


def attach_prompt(*, system: str, user: str, output: Any, reasoning: str, schema: str) -> None:
    """Fill the winning attempt row's prompt columns — the one place the exact
    system/user/output/reasoning enter the ledger. No-op when the current call
    never persisted a row (sandbox / no scene)."""
    ctx = _call_ctx.get()
    last = ctx["last"] if ctx else None
    if not last or last.get("id") is None or not last.get("slot"):
        return
    out_text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, indent=2)
    with contextlib.suppress(sqlite3.Error, OSError):
        con = _connect(_scene_db(last["slot"]))
        try:
            con.execute(
                "UPDATE flights SET system=?, user=?, output=?, reasoning=?, schema=? WHERE id=?",
                (system, user, out_text, reasoning, schema, last["id"]),
            )
            con.commit()
        finally:
            con.close()
    last["has_prompt"] = True


# --- live tail (SSE) ----------------------------------------------------------


def subscribe() -> asyncio.Queue[dict[str, Any]]:
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    _subscribers.append(q)
    return q


def unsubscribe(q: asyncio.Queue[dict[str, Any]]) -> None:
    if q in _subscribers:
        _subscribers.remove(q)


# --- unified reads (ATTACH + batched merge) -----------------------------------


def _scene_keys(run: str) -> list[str]:
    base = _runs_dir() / run
    if not run or not base.is_dir():
        return []  # never fall back to globbing the entire runs dir
    return sorted(p.parent.relative_to(_runs_dir()).as_posix() for p in base.glob("**/flights.db"))


def _batches(scenes: list[str]) -> list[list[str]]:
    return [scenes[i:i + _ATTACH_BATCH] for i in range(0, len(scenes), _ATTACH_BATCH)]


def _attach(con: sqlite3.Connection, batch: list[str]) -> None:
    for i, scene in enumerate(batch):
        con.execute(f"ATTACH DATABASE ? AS s{i}", (str(_scene_db(scene)),))


def _union(batch: list[str], cols: str) -> str:
    return " UNION ALL ".join(f"SELECT {cols} FROM s{i}.flights" for i in range(len(batch)))


def _where(filters: dict[str, list[str]], cursor: tuple[float, str, int] | None) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for key, col in _FILTER_COL.items():
        vals = filters.get(key)
        if vals:
            clauses.append(f"{col} IN ({','.join('?' * len(vals))})")
            params.extend(vals)
    if filters.get("status"):
        vals = filters["status"]
        clauses.append(f"{_STATUS_EXPR} IN ({','.join('?' * len(vals))})")
        params.extend(vals)
    if cursor is not None:
        ct, cs, ci = cursor
        clauses.append("(t_response < ? OR (t_response = ? AND (slot < ? OR (slot = ? AND id < ?))))")
        params.extend([ct, ct, cs, cs, ci])
    where = (" AND " + " AND ".join(clauses)) if clauses else ""
    return where, params


def _parse_cursor(cursor: str | None) -> tuple[float, str, int] | None:
    if not cursor:
        return None
    t, scene, rid = cursor.rsplit("|", 2)
    return float(t), scene, int(rid)


def page(run: str, *, cursor: str | None, limit: int, filters: dict[str, list[str]]) -> dict[str, Any]:
    """One keyset page (newest first) across the run's scene DBs. Metadata only —
    no prompt bytes. Correct top-K merge: each batch returns its own top `limit`
    below the cursor, so the global top `limit` is always within their union."""
    scenes = _scene_keys(run)
    where, wparams = _where(filters, _parse_cursor(cursor))
    sql = f"SELECT * FROM ({{union}}) WHERE 1=1{where} ORDER BY t_response DESC, id DESC LIMIT ?"
    merged: list[dict[str, Any]] = []
    capped = False
    for batch in _batches(scenes):
        con = sqlite3.connect(":memory:")
        con.row_factory = sqlite3.Row
        try:
            _attach(con, batch)
            rows = con.execute(sql.format(union=_union(batch, _LIST_COLS)), (*wparams, limit)).fetchall()
        finally:
            con.close()
        if len(rows) == limit:
            capped = True
        merged.extend(dict(r) for r in rows)
    merged.sort(key=lambda r: (r["t_response"] or 0.0, r["slot"] or "", r["id"] or 0), reverse=True)
    rows = merged[:limit]
    for r in rows:
        r["ok"] = bool(r["ok"])
        r["has_prompt"] = bool(r["has_prompt"])
    has_more = len(merged) > limit or capped
    next_cursor = None
    if has_more and rows:
        last = rows[-1]
        next_cursor = f"{last['t_response']}|{last['slot']}|{last['id']}"
    return {"rows": rows, "cursor": next_cursor, "has_more": has_more}


def detail(scene: str, row_id: int) -> dict[str, Any] | None:
    """The prompt bytes for one row — the only path that returns system/user/
    output/reasoning, fetched on demand when the user opens the detail panel."""
    path = _scene_db(scene)
    if not path.exists():
        return None
    con = _connect(path)
    try:
        r = con.execute(
            "SELECT system, user, output, reasoning, schema FROM flights WHERE id=?", (row_id,)
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        con.close()
    if r is None:
        return None
    return {"system": r["system"], "user": r["user"], "output": r["output"],
            "reasoning": r["reasoning"], "schema": r["schema"]}


def histogram(run: str, *, filters: dict[str, list[str]], buckets: int = 48) -> dict[str, Any]:
    """Request counts bucketed uniformly over the run's time span (the activity
    chart above the log table). Two passes over the batched ATTACH: min/max of
    `t_response` under the filters, then a GROUP BY on the bucket index."""
    scenes = _scene_keys(run)
    where, params = _where(filters, None)
    t0 = t1 = None
    for batch in _batches(scenes):
        con = sqlite3.connect(":memory:")
        try:
            _attach(con, batch)
            union = _union(batch, "t_response, status, exc_type, transport, kind, model, provider, slot, step, api_key")
            row = con.execute(
                f"SELECT MIN(t_response), MAX(t_response) FROM ({union}) WHERE 1=1{where}", params
            ).fetchone()
        finally:
            con.close()
        if row and row[0] is not None:
            t0 = row[0] if t0 is None else min(t0, row[0])
            t1 = row[1] if t1 is None else max(t1, row[1])
    if t0 is None or t1 is None:
        return {"buckets": [], "t0": None, "t1": None, "bucket_s": 0}
    width = max((t1 - t0) / buckets, 1e-6)
    counts = [0] * buckets
    for batch in _batches(scenes):
        con = sqlite3.connect(":memory:")
        try:
            _attach(con, batch)
            union = _union(batch, "t_response, status, exc_type, transport, kind, model, provider, slot, step, api_key")
            q = (
                f"SELECT CAST((t_response - ?) / ? AS INTEGER) AS b, COUNT(*) "
                f"FROM ({union}) WHERE t_response IS NOT NULL{where} GROUP BY b"
            )
            for b, c in con.execute(q, (t0, width, *params)).fetchall():
                counts[min(max(int(b), 0), buckets - 1)] += c
        finally:
            con.close()
    return {"buckets": counts, "t0": t0, "t1": t1, "bucket_s": width}


def facets(run: str, *, filters: dict[str, list[str]]) -> dict[str, Any]:
    """Per-attribute distinct values + counts for the filter UI, plus the total.
    Each facet's counts reflect all OTHER active filters (standard faceted
    behavior). One attach per batch; every facet + the total run on it."""
    scenes = _scene_keys(run)
    acc: dict[str, dict[Any, int]] = {fk: {} for fk in _FACET_EXPR}
    total = 0
    cols = "status, exc_type, transport, kind, model, provider, slot, step, api_key"
    for batch in _batches(scenes):
        con = sqlite3.connect(":memory:")
        con.row_factory = sqlite3.Row
        try:
            _attach(con, batch)
            union = _union(batch, cols)
            for fk, expr in _FACET_EXPR.items():
                sub = {k: v for k, v in filters.items() if k != fk}
                where, params = _where(sub, None)
                q = f"SELECT {expr} AS v, COUNT(*) AS c FROM ({union}) WHERE 1=1{where} GROUP BY v"
                for row in con.execute(q, params).fetchall():
                    acc[fk][row["v"]] = acc[fk].get(row["v"], 0) + row["c"]
            where, params = _where(filters, None)
            total += con.execute(
                f"SELECT COUNT(*) FROM ({union}) WHERE 1=1{where}", params
            ).fetchone()[0]
        finally:
            con.close()
    out = {
        fk: sorted(
            ({"value": v, "count": c} for v, c in counts.items()),
            key=lambda x: (-x["count"], str(x["value"])),
        )
        for fk, counts in acc.items()
    }
    return {"facets": out, "total": total}
