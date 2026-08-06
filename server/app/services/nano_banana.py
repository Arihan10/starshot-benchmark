"""Standalone Nano Banana client — async text-to-image only.

Calls Google's GenAI image-gen model directly so we can iterate on
image-gen parameters in one place. Configuration lives in the
module-level constants below; tweak them in place while iterating.

Exposed API:
    generate(prompt)                                -> NanoBananaResult
    generate_resumable(prompt, *, job_id, save_to)  -> NanoBananaResult
    save(result, path)                              -> Path
    NanoBananaResult                                — return type

`generate` is the raw call: no caching, no event logging, safe to use
without a bound SlotLog (smoke tests, ad-hoc scripts). `generate_resumable`
adds restart-resilience via app.utils.resumable — requires a bound
SlotLog, suitable for in-pipeline use.
"""

from __future__ import annotations

import asyncio
import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image

from app.utils import logging, resumable

# ── Config ──────────────────────────────────────────────────────────────
MODEL = "gemini-3.1-flash-image-preview"

# Aspect ratio for the generated image. Supported values include "1:1",
# "16:9", "9:16", "4:3", "3:4". Leave as None to use the model default.
ASPECT_RATIO: str | None = "1:1"

# Output image size. For gemini-3.1-flash: "512" (no K suffix), "1K",
# "2K", or "4K". 512 is the lowest-latency option and matches the size
# Trellis expects downstream.
IMAGE_SIZE: str | None = "512"

# Thinking budget for the image model. "high" trades latency for
# quality (slower than "minimal" but cleaner outputs).
# `include_thoughts=False` keeps thought tokens out of the response
# stream (they are still billed).
THINKING_LEVEL: str | None = "high"

# Network timeout for the underlying HTTP client, in milliseconds.
TIMEOUT_MS: int | None = 180_000

# Resumable scope. Used as the event-kind prefix in events.jsonl
# (`google.banana.submit` / `.retry` / `.done`). Distinct from
# `runware.banana` so the two providers can coexist in one log.
_RESUMABLE_SCOPE = "google.banana"

# Reactive 429 handling. We don't proactively cap RPM — Google's 429
# response tells us when we're over, and its `RetryInfo.retryDelay`
# tells us how long to wait. A 429 pushes a shared `_backoff_until`
# forward; every caller (in-flight + new) waits past that timestamp
# before issuing, so the failed request and all newcomers serialize
# through the same gate. 429s do NOT count against the resumable
# wrapper's retry budget — they're absorbed inside `_generate_once`.
_FALLBACK_429_DELAY_S = 30.0
_MAX_429_RETRIES = 8

# IMAGE_RECITATION retries — the recitation filter fires probabilistically
# and almost always clears on a re-roll, so absorb it inside _generate_once
# with a generous budget instead of spending the wrapper's general-transient
# budget on it.
_MAX_RECITATION_RETRIES = 15
_RECITATION_BACKOFF_BASE_S = 0.5
_RECITATION_BACKOFF_CAP_S = 5.0

class NanoBananaTransientError(Exception):
    """Model-side weirdness where the call returns no image with a
    finish_reason that historically clears on retry (e.g.
    MALFORMED_FUNCTION_CALL — the model's internal image-emit tool
    routing occasionally trips). Distinct from RuntimeError, which we
    reserve for deterministic refusals (SAFETY / PROHIBITED_CONTENT /
    etc.) where retrying would just waste budget."""


class NanoBananaRecitationError(Exception):
    """IMAGE_RECITATION refusal. Absorbed inside _generate_once with
    its own retry budget — not a NanoBananaTransientError so the
    resumable wrapper doesn't double-retry on top of the internal
    retries."""


# finish_reason values that are worth retrying via the resumable wrapper.
# IMAGE_RECITATION is handled separately (see NanoBananaRecitationError).
# Anything else (SAFETY, PROHIBITED_CONTENT, ...) is a deterministic
# refusal — re-issuing the same prompt will produce the same refusal, so
# we surface immediately.
_TRANSIENT_FINISH_REASONS = frozenset({
    "MALFORMED_FUNCTION_CALL",
    "FINISH_REASON_UNSPECIFIED",
    "OTHER",
})

# Transient non-429 errors retried via the resumable wrapper (network
# blips, 5xx, transport timeouts, model-side transient no-image returns).
# 429s are filtered out above.
_RETRYABLE: tuple[type[BaseException], ...] = (
    genai_errors.APIError,
    NanoBananaTransientError,
)
_MAX_ATTEMPTS = 3
# Exponential backoff base between resumable-wrapper retries:
# sleep `base * 2**attempt` + jitter seconds before re-trying.
_RETRY_BACKOFF_BASE_S = 4.0

# Cap on concurrent Google calls in flight. Pairs with the shared 429
# backoff: the semaphore prevents bursts, the backoff handles quota
# trips that still slip through.
GENERATE_CONCURRENCY = 25
_generate_slot = asyncio.Semaphore(GENERATE_CONCURRENCY)


@dataclass
class NanoBananaResult:
    image_bytes: bytes
    mime_type: str


def _make_client() -> genai.Client:
    http_options = (
        types.HttpOptions(timeout=TIMEOUT_MS) if TIMEOUT_MS is not None else None
    )
    return genai.Client(
        api_key=os.environ["GOOGLE_API_KEY"],
        http_options=http_options,
    )


def _build_config() -> types.GenerateContentConfig | None:
    image_config = (
        types.ImageConfig(aspect_ratio=ASPECT_RATIO, image_size=IMAGE_SIZE)
        if (ASPECT_RATIO is not None or IMAGE_SIZE is not None)
        else None
    )
    thinking_config = (
        types.ThinkingConfig(thinking_level=THINKING_LEVEL, include_thoughts=False)
        if THINKING_LEVEL is not None
        else None
    )
    if image_config is None and thinking_config is None:
        return None
    return types.GenerateContentConfig(
        image_config=image_config,
        thinking_config=thinking_config,
    )


def _unwrap(response: types.GenerateContentResponse) -> NanoBananaResult:
    """Pull the first image out of the response and normalize it to PNG.
    Gemini's Developer API ignores `output_mime_type` and returns JPEG
    by default; we re-encode here so downstream consumers (Trellis) see
    a single consistent format.

    When the response carries no image, we read the candidate's
    finish_reason to decide whether to raise a transient (retryable)
    or deterministic (terminal) exception. The bare response repr is
    100+ lines of SDK objects, so we include just the finish_reason in
    the message and drop the rest."""
    for part in response.parts or []:
        inline = part.inline_data
        if inline and inline.data:
            return _to_png(inline.data, inline.mime_type or "image/jpeg")
    finish_reason = _finish_reason_name(response)
    if finish_reason == "IMAGE_RECITATION":
        raise NanoBananaRecitationError(
            f"recitation refusal (finish_reason={finish_reason})"
        )
    if finish_reason in _TRANSIENT_FINISH_REASONS:
        raise NanoBananaTransientError(
            f"transient empty response (finish_reason={finish_reason})"
        )
    raise RuntimeError(
        f"Nano Banana returned no image (finish_reason={finish_reason})"
    )


def _finish_reason_name(response: types.GenerateContentResponse) -> str | None:
    """Best-effort extraction of the first candidate's finish_reason as
    its enum name (e.g. 'MALFORMED_FUNCTION_CALL'). Returns None when
    the response shape is unexpected so the caller can fall back to
    treating it as deterministic."""
    candidates = response.candidates or []
    if not candidates:
        return None
    fr = getattr(candidates[0], "finish_reason", None)
    if fr is None:
        return None
    return getattr(fr, "name", None) or str(fr)


def _to_png(data: bytes, src_mime: str) -> NanoBananaResult:
    if src_mime == "image/png":
        return NanoBananaResult(image_bytes=data, mime_type="image/png")
    img = Image.open(io.BytesIO(data))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return NanoBananaResult(image_bytes=buf.getvalue(), mime_type="image/png")


# Shared backoff: monotonic timestamp every caller waits past before
# issuing a Google call. Bumped forward by `_report_429` whenever a
# request trips the quota; cleared naturally by the passage of time.
_backoff_until: float = 0.0
_backoff_lock = asyncio.Lock()


async def _wait_clear() -> None:
    """Block until `time.monotonic() >= _backoff_until`. Re-checks
    under the lock after each sleep so a fresh 429 from another caller
    extends the wait correctly."""
    while True:
        async with _backoff_lock:
            wait = _backoff_until - time.monotonic()
            if wait <= 0:
                return
        await asyncio.sleep(wait)


async def _report_429(delay: float) -> None:
    """Push the shared backoff forward by `delay` seconds (or further
    if some other caller already reported a longer wait)."""
    global _backoff_until
    async with _backoff_lock:
        _backoff_until = max(_backoff_until, time.monotonic() + delay)


def _emit(kind: str, **fields: Any) -> None:
    """Log via the bound SlotLog if there is one, fall back to stderr.

    Lets `_generate_once` emit rate-limit telemetry both inside the
    pipeline (where a SlotLog is bound) and from the standalone smoke
    test (where it isn't)."""
    try:
        logging.log(kind, **fields)
    except LookupError:
        logging.console_note(f"[{kind}] {fields}")


def _is_429(err: genai_errors.APIError) -> bool:
    return getattr(err, "code", None) == 429


def _retry_after_seconds(err: genai_errors.APIError) -> float | None:
    """Best-effort parse of Google's `RetryInfo.retryDelay` hint
    (e.g. `"30s"`). Returns None when the hint isn't present so the
    caller can fall back to `_FALLBACK_429_DELAY_S`."""
    details = getattr(err, "details", None)
    if not isinstance(details, dict):
        return None
    error = details.get("error")
    if not isinstance(error, dict):
        return None
    nested = error.get("details")
    if not isinstance(nested, list):
        return None
    for entry in nested:
        if not isinstance(entry, dict):
            continue
        delay = entry.get("retryDelay") or entry.get("retry_delay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                continue
    return None


async def _generate_once(prompt: str) -> NanoBananaResult:
    """Single image generation, with two internal absorption loops:

      * 429 rate-limit backoff (shared across callers via _backoff_until),
        absorbed inside `_call_with_429_absorption`.
      * IMAGE_RECITATION refusals, absorbed here with their own budget —
        the filter is probabilistic on re-rolls of the same prompt, so a
        long backoff-capped retry chain usually pushes through. Spending
        the resumable wrapper's general-transient budget on these would
        leave nothing for actual transient errors.

    Both absorptions are invisible to the resumable wrapper: only
    terminal failures (budget exhausted, deterministic refusals,
    non-retryable APIErrors) surface."""
    for recitation_attempt in range(_MAX_RECITATION_RETRIES + 1):
        try:
            return await _call_with_429_absorption(prompt)
        except NanoBananaRecitationError as e:
            if recitation_attempt >= _MAX_RECITATION_RETRIES:
                raise RuntimeError(
                    f"Nano Banana recitation refusal after "
                    f"{recitation_attempt + 1} attempts: {e}"
                ) from e
            delay = min(
                _RECITATION_BACKOFF_CAP_S,
                _RECITATION_BACKOFF_BASE_S * (2 ** recitation_attempt),
            )
            _emit(
                "google.banana.recitation_retry",
                attempt=recitation_attempt,
                delay_s=delay,
                reason=str(e),
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def _call_with_429_absorption(prompt: str) -> NanoBananaResult:
    """Single Google GenAI call, gated by the shared 429 backoff and a
    concurrency semaphore. A 429 here pushes the shared backoff forward
    and the call retries — other in-flight and new callers waiting at
    `_wait_clear` see the same backoff and serialize behind it. 429s
    are absorbed here so they don't spend the resumable wrapper's
    transient-error budget."""
    async with _generate_slot:
        for attempt in range(_MAX_429_RETRIES + 1):
            await _wait_clear()
            client = _make_client()
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL,
                    contents=[prompt],
                    config=_build_config(),
                )
                return _unwrap(response)
            except genai_errors.APIError as e:
                if not _is_429(e) or attempt >= _MAX_429_RETRIES:
                    raise
                delay = _retry_after_seconds(e) or _FALLBACK_429_DELAY_S
                await _report_429(delay)
                _emit(
                    "google.banana.rate_limited",
                    attempt=attempt,
                    delay_s=delay,
                    reason=f"{type(e).__name__}: {str(e)[:160]}",
                )
        raise AssertionError("unreachable")


async def generate(prompt: str) -> NanoBananaResult:
    """Async text-to-image. No caching, no event logging."""
    return await _generate_once(prompt)


async def generate_resumable(
    prompt: str,
    *,
    job_id: str,
    save_to: Path,
    force: bool = False,
) -> NanoBananaResult:
    """Resumable text-to-image. Requires a bound SlotLog.

    Skips the API call if a prior `google.banana.done` was logged for
    `job_id` and the file at the recorded path still exists. Otherwise
    issues a fresh call (retried up to 3x on transient errors), writes
    the bytes to `save_to`, and logs `google.banana.done` so the next
    run short-circuits.

    `force=True` bypasses that completion-cache short-circuit so the image
    is always re-generated (used by a from-scratch regenerate). It stays
    NON-destructive: the bytes are written to `save_to` only after the API
    call succeeds, so a failed call (e.g. invalid/down key) leaves any
    existing image on disk untouched — never deletes it.

    Google's GenAI API has no in-flight reattach (text-to-image is a
    single HTTP roundtrip with no resumption token), so a process kill
    MID-call will re-bill on the next attempt — only completed-and-
    saved calls are cheap to resume.
    """
    if not force:
        done = resumable.find_done(scope=_RESUMABLE_SCOPE, job_id=job_id)
        if done is not None:
            saved = Path(str(done.get("saved", "")))
            if saved.exists():
                return NanoBananaResult(
                    image_bytes=saved.read_bytes(),
                    mime_type=str(done.get("mime_type", "image/png")),
                )

    input_hash = resumable.hash_input({
        "model": MODEL,
        "prompt": prompt,
        "aspect_ratio": ASPECT_RATIO,
    })

    async def _submit(_task_id: str) -> NanoBananaResult:
        return await _generate_once(prompt)

    result = await resumable.run_resumable(
        scope=_RESUMABLE_SCOPE,
        job_id=job_id,
        input_hash=input_hash,
        submit=_submit,
        reattach=None,
        max_attempts=_MAX_ATTEMPTS,
        retryable=_RETRYABLE,
        retry_backoff_base_s=_RETRY_BACKOFF_BASE_S,
    )

    save_to.parent.mkdir(parents=True, exist_ok=True)
    save_to.write_bytes(result.image_bytes)
    resumable.log_done(
        scope=_RESUMABLE_SCOPE,
        job_id=job_id,
        saved=str(save_to),
        mime_type=result.mime_type,
    )
    return result


def save(result: NanoBananaResult, path: Path) -> Path:
    """Write `result.image_bytes` to `path`, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result.image_bytes)
    return path
