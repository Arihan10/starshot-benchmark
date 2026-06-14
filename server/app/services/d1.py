"""The published-scene catalog in Cloudflare D1 (SQLite), via its REST API.

One row per (run, slot, model, version) recording the R2 keys of that scene's
assets, upserted so a re-publish overwrites in place. The prod client can then
look a scene up in D1 and resolve its assets straight from the stored keys.

Talks to the D1 HTTP query endpoint (no Worker needed):
  POST /accounts/{account}/d1/database/{db}/query  {sql, params}

Credentials come from the environment:
  CLOUDFLARE_ACCOUNT_ID   — account the database lives in
  CLOUDFLARE_API_TOKEN    — token with D1 edit permission
  D1_DATABASE_ID          — database id (defaults to database-prod's id)
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

_API_BASE = "https://api.cloudflare.com/client/v4"
_DEFAULT_DB_ID = "7437d4b3-a91c-4587-90cc-3a2ef269031d"  # database-prod

# One row per (run, slot, model) — assets are NOT versioned. Re-publishing a
# cell overwrites its row (and its R2 objects) in place.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS scenes (
  run          TEXT NOT NULL,
  slot         TEXT NOT NULL,
  model        TEXT NOT NULL,
  preview_key  TEXT NOT NULL,
  tour_key     TEXT,
  proxy_key    TEXT,
  pano_prefix  TEXT,
  pano_count   INTEGER NOT NULL DEFAULT 0,
  published_at TEXT NOT NULL,
  PRIMARY KEY (run, slot, model)
)
""".strip()

_UPSERT = """
INSERT INTO scenes
  (run, slot, model, preview_key, tour_key, proxy_key, pano_prefix, pano_count, published_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(run, slot, model) DO UPDATE SET
  preview_key  = excluded.preview_key,
  tour_key     = excluded.tour_key,
  proxy_key    = excluded.proxy_key,
  pano_prefix  = excluded.pano_prefix,
  pano_count   = excluded.pano_count,
  published_at = excluded.published_at
""".strip()

_schema_ready = False
_schema_lock = asyncio.Lock()


def _config() -> tuple[str, str, str]:
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    db_id = os.environ.get("D1_DATABASE_ID", _DEFAULT_DB_ID)
    if not (account and token):
        raise RuntimeError(
            "D1 access needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in the environment"
        )
    return account, token, db_id


async def query(sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    """Run one SQL statement against D1; return its result rows (empty for writes)."""
    account, token, db_id = _config()
    url = f"{_API_BASE}/accounts/{account}/d1/database/{db_id}/query"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json={"sql": sql, "params": params or []},
        )
    try:
        data: dict[str, Any] = resp.json()
    except ValueError:
        data = {}
    if resp.status_code >= 400 or not data.get("success", False):
        detail = data.get("errors") or resp.text[:300]
        raise RuntimeError(f"D1 query failed ({resp.status_code}): {detail}")
    result = data.get("result") or []
    return result[0].get("results", []) if result else []


async def ensure_schema() -> None:
    """Create the `scenes` table if absent (idempotent; runs once per process)."""
    global _schema_ready
    if _schema_ready:
        return
    async with _schema_lock:
        if _schema_ready:
            return
        await query(_SCHEMA)
        _schema_ready = True


async def upsert_scene(
    *,
    run: str,
    slot: str,
    model: str,
    preview_key: str,
    tour_key: str | None,
    proxy_key: str | None,
    pano_prefix: str | None,
    pano_count: int,
    published_at: str,
) -> None:
    """Insert or overwrite the catalog row for one published (run/slot/model)."""
    await ensure_schema()
    await query(
        _UPSERT,
        [
            run,
            slot,
            model,
            preview_key,
            tour_key,
            proxy_key,
            pano_prefix,
            pano_count,
            published_at,
        ],
    )
