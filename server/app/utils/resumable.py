"""Provider-agnostic resumable-submit machinery.

The pipeline talks to several remote services — Runware for image-gen
and 3D, Google GenAI for direct Nano Banana calls, and eventually others
— each with its own SDK. Two restart-resilience patterns recur:

1. **Completion cache.** After a successful submission we log
   `<scope>.done` with the on-disk artifact path. On re-entry we look
   that up; if the file still exists we short-circuit instead of paying
   for the same generation again. Works for any provider — it's just a
   "did we already do this" check.

2. **In-flight reattach.** Some providers (Runware) accept a
   client-supplied task id and expose a fetch-by-id endpoint, so a
   process restart mid-job can recover the in-flight or
   recently-finished result without re-billing. Providers that don't
   (Google's GenAI text-to-image is a single HTTP roundtrip with no
   resumption token) get only the completion cache.

`run_resumable` glues the two together. Callers supply a `submit`
coroutine for the fresh-submit path and an optional `reattach` for the
in-flight recovery; transient-error retry happens inside the wrapper.
The wrapper logs `<scope>.submit` (before each attempt) and
`<scope>.reattach` (on any prior-submit lookup) so events.jsonl is the
source of truth for both bookkeeping and the completion cache.

Callers wrap their own completion-cache check around `run_resumable`
(see `find_done` / `log_done`); that step is provider-specific because
it knows what the cached artifact looks like on disk.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from app.utils import logging

T = TypeVar("T")


def hash_input(payload: object) -> str:
    """Stable SHA-256 of a JSON-serializable payload. Use as
    `input_hash` so a prompt or config change invalidates prior
    submits."""
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def find_prior_submit(
    *, scope: str, job_id: str, input_hash: str,
) -> dict[str, Any] | None:
    """Most recent `<scope>.submit` matching (job_id, input_hash) in the
    bound slot's event log. Used by `run_resumable` to drive the
    reattach attempt; callers normally don't need to call this
    directly."""
    kind = f"{scope}.submit"
    for event in reversed(logging.current_events()):
        if (
            event.get("kind") == kind
            and event.get("job_id") == job_id
            and event.get("input_hash") == input_hash
        ):
            return event
    return None


def find_done(*, scope: str, job_id: str) -> dict[str, Any] | None:
    """Most recent `<scope>.done` for a job. Carries the provider-
    specific completion metadata (saved path, remote URL, mime type)
    the caller passed to `log_done`. Caller is responsible for
    re-verifying any on-disk paths before trusting the hit."""
    kind = f"{scope}.done"
    for event in reversed(logging.current_events()):
        if event.get("kind") == kind and event.get("job_id") == job_id:
            return event
    return None


def log_done(*, scope: str, job_id: str, **fields: Any) -> None:
    """Mark a job as completed. `**fields` lands verbatim on the event
    so `find_done` can return whatever the caller wants to cache
    (saved path, remote URL, mime type, etc.)."""
    logging.log(f"{scope}.done", job_id=job_id, **fields)


async def run_resumable(
    *,
    scope: str,
    job_id: str,
    input_hash: str,
    submit: Callable[[str], Awaitable[T]],
    reattach: Callable[[str], Awaitable[T | None]] | None = None,
    max_attempts: int = 3,
    retryable: tuple[type[BaseException], ...] = (),
    retry_backoff_base_s: float = 0.0,
) -> T:
    """Run a resumable submission.

    Flow:
      1. If `reattach` is provided AND a prior `<scope>.submit` exists
         for (job_id, input_hash), call `reattach(task_id)` once. On a
         non-None return we log `<scope>.reattach` outcome=success and
         return. On None or any exception we log outcome=expired and
         fall through to step 2.
      2. Up to `max_attempts` fresh submits. Each attempt mints a new
         UUID4 task id, logs `<scope>.submit` (carrying the task_id and
         input_hash) BEFORE awaiting `submit`, then calls
         `submit(task_id)`. Exceptions matching `retryable` retry up to
         the budget, logging `<scope>.retry`; anything else propagates.

    `submit` SHOULD use its `task_id` argument as the provider's
    resumption key when supported; providers without a resumption
    concept can ignore it (the wrapper still logs it for symmetry and
    debugging).

    Pass `reattach=None` for providers that have no in-flight recovery
    (Google GenAI text-to-image, for instance). Pass `retryable=()` to
    surface every submit exception immediately without retry.
    """
    if reattach is not None:
        prior = find_prior_submit(
            scope=scope, job_id=job_id, input_hash=input_hash,
        )
        if prior is not None:
            prior_task_id = str(prior.get("task_id"))
            ctx = {"scope": scope, "job_id": job_id, "task_id": prior_task_id}
            try:
                recovered = await reattach(prior_task_id)
                if recovered is not None:
                    logging.log(f"{scope}.reattach", outcome="success", **ctx)
                    return recovered
                logging.log(
                    f"{scope}.reattach",
                    outcome="expired",
                    reason="no_result",
                    **ctx,
                )
            except Exception as e:
                logging.log(
                    f"{scope}.reattach",
                    outcome="expired",
                    reason=f"{type(e).__name__}: {str(e)[:200]}",
                    **ctx,
                )

    for attempt in range(max_attempts):
        task_id = str(uuid.uuid4())
        logging.log(
            f"{scope}.submit",
            job_id=job_id,
            task_id=task_id,
            input_hash=input_hash,
            attempt=attempt,
        )
        try:
            return await submit(task_id)
        except retryable as e:
            if attempt == max_attempts - 1:
                raise
            delay = (
                retry_backoff_base_s * (2 ** attempt) + random.random()
                if retry_backoff_base_s > 0
                else 0.0
            )
            logging.log(
                f"{scope}.retry",
                job_id=job_id,
                attempt=attempt,
                delay_s=delay,
                reason=f"{type(e).__name__}: {str(e)[:200]}",
            )
            if delay > 0:
                await asyncio.sleep(delay)
    raise AssertionError("unreachable")
