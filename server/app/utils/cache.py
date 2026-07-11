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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # runtime import would be circular (logging imports this module)
    from app.utils.logging import SlotLog

_SEP = "\x1e"  # ASCII record separator — guards against boundary collisions.


def hash_llm_call(*, model: str, system: str, user: str, schema_name: str) -> str:
    payload = _SEP.join((model, system, user, schema_name)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def replay_key(system: str, user: str, schema_name: str) -> str:
    """Model-INDEPENDENT content hash of a call (system + user + schema). Stored
    on each slim `cache.llm` event as `_rk` so `find_llm_replay` can match the
    committed-prefix output without the raw `system`/`user` in memory."""
    payload = _SEP.join((system, user, schema_name)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hit_output(slot_log: SlotLog, event: dict[str, Any]) -> dict[str, Any] | None:
    """Read a matched slim event's committed `output` back from disk (the slim
    buffer drops it). Returns None if the event body can't be loaded."""
    idx = event.get("index")
    full = slot_log.full_event(idx) if isinstance(idx, int) else None
    output = full.get("output") if isinstance(full, dict) else None
    return output if isinstance(output, dict) else None


def find_llm_cache_hit(slot_log: SlotLog, key: str) -> dict[str, Any] | None:
    for event in reversed(slot_log.state["events"]):
        if event.get("kind") == "cache.llm" and event.get("key") == key:
            return _hit_output(slot_log, event)
    return None


def find_llm_replay(
    slot_log: SlotLog, *, system: str, user: str, schema_name: str,
) -> dict[str, Any] | None:
    """A committed `cache.llm` whose inputs match IGNORING the model. The normal
    cache key binds the model (so a fresh call on a different model re-runs),
    but a step COMMITTED earlier on one model and then REPLAYED while the task
    runs a different model (e.g. a simulation branch whose per-step A/B picked a
    non-default model, now being replayed as a prefix) must still hit its
    committed output instead of re-running. The cut is exact: a reverted step's
    `cache.llm` is truncated away, so only genuinely-committed prefix steps match
    — the frontier the user is re-running finds nothing here and proceeds to the
    live call. Used ONLY on the gated (branch) path, never the main pipeline.

    Matches on the slim `_rk` (model-independent content hash) so it needs no
    heavy fields in memory; the winning event's output is read from disk."""
    rk = replay_key(system, user, schema_name)
    for event in reversed(slot_log.state["events"]):
        if event.get("kind") == "cache.llm" and event.get("_rk") == rk:
            return _hit_output(slot_log, event)
    return None
