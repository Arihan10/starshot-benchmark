"""Event-log-backed LLM-call cache.

The event log at runs/current/events.jsonl doubles as the cache. Every
successful LLM call emits a `cache.llm` event carrying the call key and
validated output; cache lookup is a backward scan over the in-memory
event buffer, so truncating the event log rewinds the cache in one
step. The provider-job resumability layer (submit + reattach + per-
stage `.done` completion events) lives in `app.utils.resumable`.
"""

from __future__ import annotations

import hashlib
from typing import Any

_SEP = "\x1e"  # ASCII record separator — guards against boundary collisions.


def hash_llm_call(*, model: str, system: str, user: str, schema_name: str) -> str:
    payload = _SEP.join((model, system, user, schema_name)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def find_llm_cache_hit(
    events: list[dict[str, Any]], key: str,
) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("kind") == "cache.llm" and event.get("key") == key:
            output = event.get("output")
            return output if isinstance(output, dict) else None
    return None


def find_llm_replay(
    events: list[dict[str, Any]], *, system: str, user: str, schema_name: str,
) -> dict[str, Any] | None:
    """A committed `cache.llm` whose inputs match IGNORING the model. The normal
    cache key binds the model (so a fresh call on a different model re-runs),
    but a step COMMITTED earlier on one model and then REPLAYED while the task
    runs a different model (e.g. a simulation branch whose per-step A/B picked a
    non-default model, now being replayed as a prefix) must still hit its
    committed output instead of re-running. The cut is exact: a reverted step's
    `cache.llm` is truncated away, so only genuinely-committed prefix steps match
    — the frontier the user is re-running finds nothing here and proceeds to the
    live call. Used ONLY on the gated (branch) path, never the main pipeline."""
    for event in reversed(events):
        if (
            event.get("kind") == "cache.llm"
            and event.get("schema") == schema_name
            and event.get("system") == system
            and event.get("user") == user
        ):
            output = event.get("output")
            return output if isinstance(output, dict) else None
    return None
