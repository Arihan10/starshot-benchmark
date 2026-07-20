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
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.core import slots
from app.utils import cellsummary
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
  system TEXT, user TEXT, output TEXT, reasoning TEXT, schema TEXT,
  cost REAL
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
    # Token-priced cost for the compat backends (Moonshot, Alibaba, ...), which
    # bill by token and return no per-request cost. None for OpenRouter (its
    # settled cost lands later as an `llm.cost` event) and for errored attempts
    # with no token counts. `model` here is the provider-side id on the direct
    # transport and the OpenRouter id on a BYOK flight — `token_cost` matches both.
    cost = slots.token_cost(model, tokens_in, tokens_out)

    row_id: int | None = None
    if scene:
        row_id = _insert(scene, (
            t_req, t_res, flight_ms, transport, provider, base_url, model, kind,
            status, ok, err, exc_type, masked, tokens_in, tokens_out, generation_id,
            scene, step, node, call, attempt, cost,
        ))

    event: dict[str, Any] = {
        "id": row_id, "slot": scene, "t_request": t_req, "t_response": t_res,
        "flight_ms": flight_ms, "transport": transport, "provider": provider,
        "base_url": base_url, "model": model, "kind": kind, "status": status,
        "ok": bool(ok), "error": err, "exc_type": exc_type, "key": masked,
        "tokens_in": tokens_in, "tokens_out": tokens_out, "generation_id": generation_id,
        "step": step, "node": node, "call": call, "attempt": attempt, "cost": cost,
        "has_prompt": False,
    }
    if ctx is not None:
        ctx["last"] = event
    if scene:
        for q in _subscribers:
            q.put_nowait(event)
    return event


def _ensure_cost_column(con: sqlite3.Connection) -> None:
    """Add the `cost` column to a pre-existing scene DB created before it was
    part of the schema. `CREATE TABLE IF NOT EXISTS` never alters an existing
    table, so a DB from an older build keeps the old shape until this ALTER runs
    — idempotent (checked against `PRAGMA table_info`), so it fires at most once
    per scene per process."""
    cols = {r[1] for r in con.execute("PRAGMA table_info(flights)")}
    if "cost" not in cols:
        con.execute("ALTER TABLE flights ADD COLUMN cost REAL")
        con.commit()


def _insert(scene: str, values: tuple[Any, ...]) -> int | None:
    path = _scene_db(scene)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        first = scene not in _initialized
        con = _connect(path, create=first)
        if first:
            _ensure_cost_column(con)
    except (sqlite3.Error, OSError):
        return None  # a logging fault must never break the LLM call it describes
    try:
        _initialized.add(scene)
        cur = con.execute(
            "INSERT INTO flights (t_request, t_response, flight_ms, transport, provider, "
            "base_url, model, kind, status, ok, error, exc_type, api_key, tokens_in, "
            "tokens_out, generation_id, slot, step, node, call, attempt, cost) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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


# --- legacy-scene fallback: reconstruct the ledger from events.jsonl ----------
# Scenes that predate the SQLite ledger (and branches) have an events.jsonl but
# no flights.db, so the dashboard would show nothing for them. The read paths
# reconstruct a ledger from the event log on first access: the winning-attempt
# row per `cache.llm` call (metadata + compat token cost; the heavy prompt
# columns stay in events.jsonl). This is the same reconstruction the offline
# `scripts/backfill_scenes.py` uses, so a scene is identical whether it was
# backfilled ahead of time or lazily on read.

_RECON_COLS = (
    "t_request, t_response, flight_ms, transport, provider, base_url, model, kind, "
    "status, ok, error, exc_type, api_key, tokens_in, tokens_out, generation_id, "
    "slot, step, node, call, attempt, cost"
)


def _row_treq(e: dict[str, Any]) -> float | None:
    """The request time a reconstructed row is stored (and deduped) with: the
    call's own `t_request`, or its event `ts` for calls that predate the flight
    ledger (so `t_request` was never captured). None only when both are absent."""
    t = e.get("t_request")
    return t if t is not None else e.get("ts")


def _flight_row(e: dict[str, Any], scene: str) -> tuple:
    """Reconstruct one flight ledger row (metadata + compat cost) from a
    `cache.llm` event — the winning attempt. Transport/provider/model mirror what
    the live ledger records: a compat model with no generation_id was the direct
    endpoint; anything with a generation_id (or a non-compat model) went through
    OpenRouter. Prompt columns are left NULL (they remain in events.jsonl)."""
    model = e.get("model")
    gid = e.get("generation_id")
    cfg = slots.OPENAI_COMPAT_MODELS.get(model)
    if cfg is not None and not gid:
        transport = "direct"
        base_url = cfg.base_url
        provider = urlparse(base_url).hostname if base_url else None
        model_col = cfg.model  # provider-side id, as the live direct row stores
    else:
        transport, base_url, provider, model_col = "openrouter", None, "openrouter", model
    t_res = e.get("t_response") if e.get("t_response") is not None else e.get("ts")
    cost = slots.token_cost(model, e.get("tokens_in"), e.get("tokens_out"))
    return (
        _row_treq(e), t_res, e.get("flight_ms"), transport, provider, base_url, model_col,
        "structured", 200, 1, None, None, None, e.get("tokens_in"), e.get("tokens_out"),
        gid, scene, e.get("step"), e.get("node"), None, e.get("attempts"), cost,
    )


def reconstruct_flights(scene: str, events: list[dict[str, Any]], *, dry_run: bool = False) -> tuple[int, int]:
    """Create/extend `scene`'s flight ledger from its `cache.llm` events. Returns
    `(rows_added, rows_priced)`. Adds only calls not already present (deduped by
    generation_id / rounded request time, matched to how the row is stored) and
    prices any existing unpriced compat rows. Idempotent — safe to re-run."""
    db_path = _scene_db(scene)
    existing_gids: set[str] = set()
    existing_treq: set[float] = set()
    if db_path.exists():
        try:
            con = _connect(db_path)
            try:
                _ensure_cost_column(con)
                for gid, tr in con.execute("SELECT generation_id, t_request FROM flights"):
                    if isinstance(gid, str):
                        existing_gids.add(gid)
                    elif tr is not None:
                        existing_treq.add(round(float(tr), 3))
            finally:
                con.close()
        except sqlite3.Error:
            return 0, 0

    new_rows: list[tuple] = []
    for e in events:
        if e.get("kind") != "cache.llm":
            continue
        gid = e.get("generation_id")
        if isinstance(gid, str):
            if gid in existing_gids:
                continue
            existing_gids.add(gid)
        else:
            tr = _row_treq(e)
            if tr is None:
                continue  # unplaceable (no id, no time) — skip rather than dup
            rtr = round(float(tr), 3)
            if rtr in existing_treq:
                continue
            existing_treq.add(rtr)
        new_rows.append(_flight_row(e, scene))

    if dry_run:
        return len(new_rows), 0
    priced = 0
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        con = _connect(db_path, create=True)
    except (sqlite3.Error, OSError):
        return 0, 0
    try:
        _ensure_cost_column(con)
        if new_rows:
            con.executemany(
                f"INSERT INTO flights ({_RECON_COLS}) VALUES ({','.join('?' * 22)})", new_rows,
            )
        for (m,) in con.execute(
            "SELECT DISTINCT model FROM flights WHERE model IS NOT NULL AND cost IS NULL"
        ).fetchall():
            pricing = slots.model_pricing(m)
            if pricing is None:
                continue
            pin, pout = pricing
            cur = con.execute(
                "UPDATE flights SET cost = (COALESCE(tokens_in,0)/1000000.0*?) + "
                "(COALESCE(tokens_out,0)/1000000.0*?) WHERE model = ? AND cost IS NULL "
                "AND (tokens_in IS NOT NULL OR tokens_out IS NOT NULL)",
                (pin, pout, m),
            )
            priced += cur.rowcount
        con.commit()
    except sqlite3.Error:
        return 0, 0
    finally:
        con.close()
    return len(new_rows), priced


_ensure_lock = threading.Lock()


def _ensure_ledger(scene: str) -> None:
    """Legacy-scene fallback: if `scene` has an event log but no flights.db yet,
    reconstruct the ledger from events.jsonl so the dashboard shows its calls.
    Lock-guarded + idempotent, so concurrent dashboard reads never double-build."""
    if _scene_db(scene).exists():
        return
    events_path = _runs_dir() / scene / "events.jsonl"
    if not events_path.exists():
        return
    with _ensure_lock:
        if _scene_db(scene).exists():  # built while we waited on the lock
            return
        with contextlib.suppress(OSError):
            reconstruct_flights(scene, list(cellsummary.iter_events(events_path)))


def _ensure_ledgers(run: str) -> None:
    """Reconstruct a ledger for every scene under `run` that has an event log but
    no flights.db — the run-scoped fallback the multi-scene reads run first."""
    base = _runs_dir() / run
    if not run or not base.is_dir():
        return
    for events_path in base.glob("**/events.jsonl"):
        _ensure_ledger(events_path.parent.relative_to(_runs_dir()).as_posix())


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
    _ensure_ledgers(run)  # legacy scenes with no flights.db → reconstruct from events
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
    output/reasoning, fetched on demand when the user opens the detail panel.
    (A reconstructed legacy row has none — prompts stay in events.jsonl.)"""
    _ensure_ledger(scene)
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


def locate(scene: str, *, generation_id: str | None = None, t_request: float | None = None) -> dict[str, Any] | None:
    """The metadata row for ONE call in `scene`'s DB, matched by generation_id
    (preferred) or t_request — the join from a scene's `cache.llm` event back to
    its flight row, so the dashboard can jump straight to that call's log entry.
    Queries the single known scene DB directly (no ATTACH)."""
    _ensure_ledger(scene)
    path = _scene_db(scene)
    if not path.exists() or (generation_id is None and t_request is None):
        return None
    con = _connect(path)
    row = None
    try:
        if generation_id:
            row = con.execute(
                f"SELECT {_LIST_COLS} FROM flights WHERE generation_id=? ORDER BY t_response DESC LIMIT 1",
                (generation_id,),
            ).fetchone()
        if row is None and t_request is not None:
            row = con.execute(
                f"SELECT {_LIST_COLS} FROM flights WHERE t_request=? ORDER BY t_response DESC LIMIT 1",
                (t_request,),
            ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        con.close()
    if row is None:
        return None
    d = dict(row)
    d["ok"] = bool(d["ok"])
    d["has_prompt"] = bool(d["has_prompt"])
    return d


def histogram(run: str, *, filters: dict[str, list[str]], buckets: int = 48) -> dict[str, Any]:
    """Request counts bucketed uniformly over the run's time span (the activity
    chart above the log table). Two passes over the batched ATTACH: min/max of
    `t_response` under the filters, then a GROUP BY on the bucket index."""
    _ensure_ledgers(run)
    scenes = _scene_keys(run)
    where, params = _where(filters, None)
    t0 = t1 = None
    cols = "t_response, status, exc_type, transport, kind, model, provider, slot, step, api_key"
    for batch in _batches(scenes):
        con = sqlite3.connect(":memory:")
        try:
            _attach(con, batch)
            row = con.execute(
                f"SELECT MIN(t_response), MAX(t_response) FROM ({_union(batch, cols)}) WHERE 1=1{where}",
                params,
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
            q = (
                f"SELECT CAST((t_response - ?) / ? AS INTEGER) AS b, COUNT(*) "
                f"FROM ({_union(batch, cols)}) WHERE t_response IS NOT NULL{where} GROUP BY b"
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
    _ensure_ledgers(run)
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
