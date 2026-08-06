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


def hash_llm_replay_key(*, system: str, user: str, schema_name: str) -> str:
    """Model-INDEPENDENT hash of a call's inputs — `hash_llm_call` minus the model.
    Stamped on every `cache.llm` event as `replay_key` so `find_llm_replay` can
    match a committed step across a model swap WITHOUT holding the (huge) system /
    user prompt bytes in memory. Events logged before this field existed simply
    won't match (the branch re-runs that step — safe, just not free)."""
    payload = _SEP.join((system, user, schema_name)).encode("utf-8")
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
    live call. Used ONLY on the gated (branch) path, never the main pipeline.

    Matches on the stored model-independent `replay_key` rather than the raw
    system/user bytes, so it works while those bytes are offloaded from the
    in-memory event buffer (see `logging._HEAVY_OFFLOAD`). `output` is kept
    resident, so a match returns without a disk read."""
    replay_key = hash_llm_replay_key(system=system, user=user, schema_name=schema_name)
    for event in reversed(events):
        if event.get("kind") == "cache.llm" and event.get("replay_key") == replay_key:
            output = event.get("output")
            return output if isinstance(output, dict) else None
    return None
