"""Trellis 2 mesh generation via the hosted spawn-and-poll HTTP API.

`POST /generate` uploads an image (multipart) and returns a
server-assigned `job_id` immediately — the GPU work runs detached so
the request itself doesn't have to outlive Modal's ~60s HTTP edge
timeout. `GET /jobs/{job_id}` is the non-blocking status probe;
`GET /jobs/{job_id}/result` streams the GLB binary back once status
flips to `done`.

Image generation lives in `app.services.nano_banana` — callers run that
first, then pass the resulting bytes (or a hosted URL) here.

Restart-resilience (in-flight reattach + completion caching) lives in
`app.utils.resumable`; we record the server's `job_id` under our own
`task_id` slot so a process restart can re-poll the same job instead
of re-billing it. Submissions whose server job is gone (404 / failed)
fall through to a fresh `POST /generate`.

The returned GLB has textures embedded in its binary chunk, but
trimesh cannot decode them unless Pillow is installed at import time.
Pillow is a project dependency (see pyproject.toml) for that reason.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
from pathlib import Path
from typing import Any

import httpx

from app.utils import logging, resumable

TRELLIS_BASE_URL = os.environ.get(
    "TRELLIS_BASE_URL",
    "https://starshot-aitools--trellis2-image-to-3d-router-fastapi-app.modal.run",
)

# Trellis production knobs. Kept module-level so the playground / batch
# scripts can read or override them without round-tripping through env.
# `resolution` is a string per the API contract ("512" | "1024" | "1536").
TRELLIS_RESOLUTION = "512"
TRELLIS_TEXTURE_SIZE = 1024
TRELLIS_DECIMATION_TARGET = 500_000
TRELLIS_SEED = 0

# Spawn-and-poll cadence. The API caps any single job at ~10 min
# wall-clock; warm 512-res jobs finish in ~3s, 1536-res in ~60s, cold
# starts add 30-90s. 10s × GENERATE_CONCURRENCY=8 = 48 polls/min,
# comfortably under the per-IP-per-container `/jobs/{id}` rate limit
# (60/60s) even if Modal's load balancer pins us to a single container.
POLL_INTERVAL_SECONDS = 10.0
POLL_TIMEOUT_SECONDS = 600.0

MAX_ATTEMPTS = 3
# Download has its own (larger) budget. The status endpoint can flip to
# "done" a few hundred ms before the GLB is committed to storage, so
# the result endpoint may 425 ("Too Early") for several seconds after.
# We don't want to give up — the job ran, the bytes are coming.
DOWNLOAD_MAX_ATTEMPTS = 8
# Exponential backoff between retries: sleep `base * 2**attempt` + jitter
# seconds before re-trying, capped to keep the schedule sane. Honors
# `Retry-After` on 429 responses when Modal sets it; otherwise the
# exponential schedule applies.
RETRY_BACKOFF_BASE_S = 4.0
RETRY_BACKOFF_MAX_S = 60.0
# Transient failures we retry on: the httpx network-layer errors
# (RemoteProtocolError, ReadError, ConnectError, timeouts) raised by
# spawn/poll/download, plus TimeoutError from our own poll budget.
RETRYABLE: tuple[type[BaseException], ...] = (
    httpx.HTTPError,
    ConnectionError,
    TimeoutError,
)

# Cap the number of in-flight Trellis jobs at any moment. Modal's
# FastAPI router returns 429 once concurrent inputs exceed its
# per-container × container-count budget; this gates the full
# submit→poll→download lifecycle so we never overshoot.
GENERATE_CONCURRENCY = 8
_generate_slot = asyncio.Semaphore(GENERATE_CONCURRENCY)


def _retry_delay(attempt: int, err: BaseException) -> float:
    """Sleep before the next retry. 429s with a `Retry-After` header use
    that hint verbatim; everything else uses exponential backoff capped
    at `RETRY_BACKOFF_MAX_S` with [0, 1)s jitter."""
    if (
        isinstance(err, httpx.HTTPStatusError)
        and err.response.status_code == 429
    ):
        retry_after = err.response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
    raw = RETRY_BACKOFF_BASE_S * (2 ** attempt)
    return min(raw, RETRY_BACKOFF_MAX_S) + random.random()


# Shared async HTTP client, lazily initialized. Single client so HTTP/2
# connection pooling kicks in across concurrent jobs.
_http: httpx.AsyncClient | None = None
_http_lock = asyncio.Lock()


async def _get_http() -> httpx.AsyncClient:
    global _http
    async with _http_lock:
        if _http is None:
            _http = httpx.AsyncClient(follow_redirects=True)
        return _http


async def disconnect_http() -> None:
    """Close the shared HTTP client. Call once during FastAPI lifespan
    teardown so the server exits cleanly."""
    global _http
    async with _http_lock:
        if _http is not None:
            try:
                await _http.aclose()
            except Exception:  # noqa: BLE001
                pass
            _http = None


async def _fetch_url(url: str) -> bytes:
    http = await _get_http()
    resp = await http.get(url, timeout=180.0)
    resp.raise_for_status()
    return resp.content


async def _post_generate(image_bytes: bytes, image_mime: str) -> str:
    http = await _get_http()
    files = {"image": ("image.png", image_bytes, image_mime)}
    data = {
        "seed": str(TRELLIS_SEED),
        "resolution": TRELLIS_RESOLUTION,
        "texture_size": str(TRELLIS_TEXTURE_SIZE),
        "decimation_target": str(TRELLIS_DECIMATION_TARGET),
    }
    resp = await http.post(
        f"{TRELLIS_BASE_URL}/generate",
        files=files, data=data, timeout=60.0,
    )
    resp.raise_for_status()
    body = resp.json()
    server_job_id = body.get("job_id")
    if not server_job_id:
        raise RuntimeError(f"trellis /generate returned no job_id: {body!r}")
    return str(server_job_id)


async def _poll_status(server_job_id: str) -> dict[str, Any]:
    """Single non-blocking status probe. Caller drives the poll loop."""
    http = await _get_http()
    resp = await http.get(
        f"{TRELLIS_BASE_URL}/jobs/{server_job_id}", timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


async def _poll_until_done(server_job_id: str, *, timeout: float) -> None:
    """Block until status flips to `done`. Raises on `failed` or our
    own poll-timeout budget.

    Transient errors on the status endpoint (429s, network blips) do
    NOT tear out of the poll — they log a `trellis.poll.retry` and
    back off, then the loop checks status again. The Modal job is
    still running; we just lost a probe."""
    deadline = asyncio.get_running_loop().time() + timeout
    transient_count = 0
    while True:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(
                f"trellis job {server_job_id} did not finish in {timeout:.0f}s"
            )
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        try:
            status = await _poll_status(server_job_id)
        except (httpx.HTTPError, ConnectionError, TimeoutError) as e:
            delay = _retry_delay(transient_count, e)
            logging.log(
                "trellis.poll.retry",
                task_id=server_job_id,
                attempt=transient_count,
                delay_s=delay,
                reason=f"{type(e).__name__}: {str(e)[:200]}",
            )
            transient_count += 1
            await asyncio.sleep(delay)
            continue
        transient_count = 0
        s = status.get("status")
        if s == "done":
            return
        if s == "failed":
            err = status.get("error", {})
            raise RuntimeError(
                f"trellis worker failed: "
                f"{err.get('type', '?')}: {err.get('message', '?')}"
            )
        # "pending" → keep polling


async def _download_result(server_job_id: str) -> bytes:
    http = await _get_http()
    for attempt in range(DOWNLOAD_MAX_ATTEMPTS):
        try:
            resp = await http.get(
                f"{TRELLIS_BASE_URL}/jobs/{server_job_id}/result",
                timeout=180.0,
            )
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPError as e:
            if attempt == DOWNLOAD_MAX_ATTEMPTS - 1:
                raise
            delay = _retry_delay(attempt, e)
            logging.log(
                "trellis.download.retry",
                task_id=server_job_id,
                attempt=attempt,
                delay_s=delay,
                reason=f"{type(e).__name__}: {str(e)[:200]}",
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def _try_reattach(server_job_id: str) -> bytes | None:
    """Probe a previously-submitted server job. Returns GLB bytes if
    the job is done or finishes within the poll budget; None if the
    job is gone (404) or surfaced a worker failure (caller treats both
    as "fall through to a fresh submit"). Re-raises only on network
    errors that should bubble up as retryable."""
    http = await _get_http()
    try:
        resp = await http.get(
            f"{TRELLIS_BASE_URL}/jobs/{server_job_id}", timeout=30.0,
        )
    except httpx.HTTPError:
        raise
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    status = resp.json()
    s = status.get("status")
    if s == "failed":
        return None
    if s == "pending":
        try:
            await _poll_until_done(
                server_job_id, timeout=POLL_TIMEOUT_SECONDS,
            )
        except (RuntimeError, TimeoutError):
            return None
    return await _download_result(server_job_id)


def _input_hash(image_bytes: bytes) -> str:
    """Stable hash of every input that would change the GLB. Same
    payload + same settings → same hash → reattach to the prior
    submit; any change invalidates and forces a fresh submit."""
    return resumable.hash_input({
        "base_url": TRELLIS_BASE_URL,
        "resolution": TRELLIS_RESOLUTION,
        "texture_size": TRELLIS_TEXTURE_SIZE,
        "decimation_target": TRELLIS_DECIMATION_TARGET,
        "seed": TRELLIS_SEED,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
    })


async def generate_mesh(
    image: bytes | str,
    *,
    output_path: Path,
    job_id: str,
    image_mime: str = "image/png",
    skip_reattach: bool = False,
) -> Path:
    """Run Trellis 2 on `image` and save the textured GLB to `output_path`.

    `image` is either raw image bytes uploaded via multipart, or a
    remote URL which we fetch to bytes first (the new API doesn't
    accept URLs server-side).

    Restart-resilient: if `trellis.done` was previously logged for
    `job_id` and the file at the recorded path still exists, the call
    short-circuits. Otherwise we look up the most recent
    `trellis.submit` matching `(job_id, input_hash)` and probe its
    server-assigned job; on a hit we download the result, on a miss
    we fall through to a fresh `POST /generate`.

    `skip_reattach=True` bypasses the prior-submit probe and goes
    straight to a fresh submit.
    """
    done = resumable.find_done(scope="trellis", job_id=job_id)
    if done is not None:
        cached = Path(str(done["saved"]))
        if cached.exists():
            return cached

    image_bytes = await _fetch_url(image) if isinstance(image, str) else image
    input_hash = _input_hash(image_bytes)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with _generate_slot:
        if not skip_reattach:
            prior = resumable.find_prior_submit(
                scope="trellis", job_id=job_id, input_hash=input_hash,
            )
            if prior is not None:
                server_job_id = str(prior.get("task_id"))
                ctx = {"scope": "trellis", "job_id": job_id, "task_id": server_job_id}
                try:
                    recovered = await _try_reattach(server_job_id)
                    if recovered is not None:
                        output_path.write_bytes(recovered)
                        logging.log(
                            "trellis.reattach", outcome="success", **ctx,
                        )
                        resumable.log_done(
                            scope="trellis", job_id=job_id,
                            server_job_id=server_job_id,
                            saved=str(output_path),
                        )
                        return output_path
                    logging.log(
                        "trellis.reattach",
                        outcome="expired",
                        reason="job_gone_or_failed",
                        **ctx,
                    )
                except Exception as e:  # noqa: BLE001
                    logging.log(
                        "trellis.reattach",
                        outcome="expired",
                        reason=f"{type(e).__name__}: {str(e)[:200]}",
                        **ctx,
                    )

        # Hold `server_job_id` across outer retries. Once we have one,
        # we NEVER call `_post_generate` again in this lifecycle —
        # any retryable failure during poll or download re-enters the
        # same Modal job instead of orphaning it and burning a fresh
        # generation. (The previous design re-submitted on poll/download
        # errors, leaving the original job to finish on Modal with no
        # client able to download it.)
        server_job_id: str | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                if server_job_id is None:
                    server_job_id = await _post_generate(image_bytes, image_mime)
                    logging.log(
                        "trellis.submit",
                        job_id=job_id,
                        task_id=server_job_id,
                        input_hash=input_hash,
                        attempt=attempt,
                    )
                await _poll_until_done(
                    server_job_id, timeout=POLL_TIMEOUT_SECONDS,
                )
                content = await _download_result(server_job_id)
                output_path.write_bytes(content)
                resumable.log_done(
                    scope="trellis",
                    job_id=job_id,
                    server_job_id=server_job_id,
                    saved=str(output_path),
                )
                return output_path
            except RETRYABLE as e:
                if attempt == MAX_ATTEMPTS - 1:
                    raise
                delay = _retry_delay(attempt, e)
                logging.log(
                    "trellis.retry",
                    job_id=job_id,
                    task_id=server_job_id,
                    attempt=attempt,
                    delay_s=delay,
                    reason=f"{type(e).__name__}: {str(e)[:200]}",
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")
