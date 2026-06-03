"""Trellis 2 mesh generation via the hosted spawn-and-poll HTTP API.

`POST /generate` uploads an image (multipart) and returns a
server-assigned `job_id` immediately — the GPU work runs detached so
the request itself doesn't have to outlive Modal's ~60s HTTP edge
timeout. `GET /jobs/{job_id}` is the non-blocking status probe;
`GET /jobs/{job_id}/result` streams the GLB binary back once status
flips to `done`.

Image generation lives in `app.services.nano_banana` — callers run that
first, then pass the resulting bytes (or a hosted URL) here.

Restart-resilience on completed work lives in `app.utils.resumable`:
if `trellis.done` was logged and the saved GLB still exists, we
short-circuit and reuse it. Anything not done re-submits fresh —
we don't probe stale Modal task_ids because Modal GCs them on its
own schedule and the in-flight semaphore (capped at Modal's
container count) keeps Modal-side queueing at zero, so each fresh
submit goes straight to a GPU.

The returned GLB has textures embedded in its binary chunk, but
trimesh cannot decode them unless Pillow is installed at import time.
Pillow is a project dependency (see pyproject.toml) for that reason.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
import time
from pathlib import Path
from typing import Any

import httpx

from app.utils import logging, resumable

TRELLIS_BASE_URL = os.environ.get(
    "TRELLIS_BASE_URL",
    "https://starshot-aitools--starshot-assets-router-fastapi-app.modal.run",
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
# starts add 30-90s. 12s × GENERATE_CONCURRENCY=100 = 500 polls/min
# aggregate — OK while Modal's LB spreads it across the 100 containers
# (~5/min each, under the per-IP-per-container `/jobs/{id}` 60/60s limit),
# but a pin to a single container would now exceed it.
POLL_INTERVAL_SECONDS = 12.0
POLL_TIMEOUT_SECONDS = 1200.0

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


class JobLostError(Exception):
    """Modal's job registry returned 404 for our task_id. The job is
    permanently gone — most often because the worker container that
    held the in-memory job table restarted (auto-scale, deploy, OOM).
    Polling the same id again is hopeless; the outer retry loop should
    treat this as a resubmit signal: drop the dead task_id and call
    `_post_generate` again on the next attempt."""

# Cap on in-flight Trellis jobs at any moment (process-global FIFO across all
# slots). Submits are additionally spaced by `_pace_submit` so a burst up to
# this many ramps onto the endpoint instead of hitting it all at once — Trellis
# 429s on bursts. Modal autoscales / queues beyond its live GPU count, so a
# "pending" status here can mean queued rather than actively processing.
GENERATE_CONCURRENCY = 100
_inflight_sem = asyncio.Semaphore(GENERATE_CONCURRENCY)

# Minimum spacing between successive `POST /generate` submits, process-globally.
# The in-flight cap governs how MANY jobs run at once; this governs how FAST we
# hand them to Modal. Without it, a batch (the generate gate firing a whole
# scene, or a large zone) bursts every submit simultaneously and trips the
# endpoint's 429 rate limit; spacing them ~1/s keeps us under it while still
# letting up to GENERATE_CONCURRENCY run concurrently. The read-modify-write of
# `_next_submit_at` needs no lock — asyncio runs it without an intervening await.
_SUBMIT_MIN_INTERVAL_S = 1.0
_next_submit_at = 0.0


async def _pace_submit() -> None:
    global _next_submit_at
    loop = asyncio.get_running_loop()
    now = loop.time()
    target = max(now, _next_submit_at)
    _next_submit_at = target + _SUBMIT_MIN_INTERVAL_S
    delay = target - now
    if delay > 0:
        await asyncio.sleep(delay)

# Live snapshot of in-flight Trellis work. Process-global so the queue
# view reflects the same scope as the semaphore. Keyed by (slot_id,
# job_id) since node ids are unique within a slot but can collide
# across slots. Lifecycle:
#   waiting    — `generate_mesh` was called, awaiting `_inflight_sem`
#   processing — semaphore acquired, Modal task_id assigned
#   (removed)  — `generate_mesh` returned (success) or raised
# Read via `queue_snapshot()`; the GET /trellis/queue endpoint hands
# it to the client, which polls every ~1.5s. We expose this instead
# of letting the client infer state from the streamed event log
# because the event log replays historical submits on every SSE
# subscribe, with no way to distinguish "still running" from "process
# was killed before it could log .done" — so historical inference
# leaks stale rows. Live state, by contrast, resets to empty on
# process restart, which is exactly the truth.
_QUEUE: dict[tuple[str, str], dict[str, Any]] = {}


def queue_snapshot() -> list[dict[str, Any]]:
    """Current in-flight + waiting Trellis jobs across all slots."""
    out: list[dict[str, Any]] = []
    for (slot_id, job_id), entry in _QUEUE.items():
        out.append({
            "slot_id": slot_id,
            "job_id": job_id,
            "state": entry["state"],
            "since": entry["since"],
            "task_id": entry.get("task_id"),
        })
    return out


def _queue_set(slot_id: str | None, job_id: str, state: str, **extra: Any) -> None:
    if slot_id is None:
        return
    key = (slot_id, job_id)
    cur = _QUEUE.get(key)
    _QUEUE[key] = {
        "state": state,
        "since": cur["since"] if cur and cur["state"] == state else time.time(),
        **({k: v for k, v in (cur or {}).items() if k not in {"state", "since"}}),
        **extra,
    }


def _queue_drop(slot_id: str | None, job_id: str) -> None:
    if slot_id is None:
        return
    _QUEUE.pop((slot_id, job_id), None)


def mark_queued(slot_id: str | None, job_id: str) -> None:
    """Register an externally-managed job (e.g. a regeneration awaiting its turn
    in a per-cell worker) as `waiting` in the shared queue snapshot, so it shows
    in the same `/trellis/queue` panel as live Trellis work. When the job actually
    submits, `generate_mesh` takes over the (slot_id, job_id) entry; pair this
    with `unmark_queued` at hand-off / cancellation so it can't leak."""
    _queue_set(slot_id, job_id, "waiting")


def unmark_queued(slot_id: str | None, job_id: str) -> None:
    """Drop an entry registered via `mark_queued` — the job is starting (so
    `generate_mesh` will manage its own entry) or was cancelled before it ran."""
    _queue_drop(slot_id, job_id)


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
    """Block until status flips to `done`. Raises on `failed`, our own
    poll-timeout budget, or `JobLostError` when Modal returns 404 for
    the task_id.

    Transient errors on the status endpoint (429s, network blips) do
    NOT tear out of the poll — they log a `trellis.poll.retry` and
    back off, then the loop checks status again. 404, on the other
    hand, is terminal: the worker that held this job is gone and
    polling the same id will never recover."""
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
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise JobLostError(
                    f"trellis job {server_job_id} not found on server "
                    "(Modal worker likely restarted; resubmitting)"
                ) from e
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
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise JobLostError(
                    f"trellis job {server_job_id} not found on server "
                    "during result download (Modal worker likely restarted; "
                    "resubmitting)"
                ) from e
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


def _input_hash(image_bytes: bytes) -> str:
    """Stable hash of every input that would change the GLB. Logged on
    `trellis.submit` for audit so two submits with identical inputs are
    visibly equivalent."""
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
) -> Path:
    """Run Trellis 2 on `image` and save the textured GLB to `output_path`.

    `image` is either raw image bytes uploaded via multipart, or a
    remote URL which we fetch to bytes first (the new API doesn't
    accept URLs server-side).

    Restart-resilient on completed work: if `trellis.done` was
    previously logged for `job_id` and the file at the recorded path
    still exists, the call short-circuits. Otherwise we always issue
    a fresh `POST /generate` — we don't probe prior Modal task_ids
    because Modal GCs them on its own schedule, and the in-flight
    semaphore (`_inflight_sem`) gates submits at Modal's container
    count so Modal never queues on its side.
    """
    done = resumable.find_done(scope="trellis", job_id=job_id)
    if done is not None:
        cached = Path(str(done["saved"]))
        if cached.exists():
            return cached

    image_bytes = await _fetch_url(image) if isinstance(image, str) else image
    input_hash = _input_hash(image_bytes)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    slot_id = logging.current_slot_id()
    _queue_set(slot_id, job_id, "waiting")
    try:
        async with _inflight_sem:
            # By default we hold `server_job_id` across outer retries —
            # any retryable failure during poll or download re-enters the
            # same Modal job instead of orphaning it and burning a fresh
            # generation. The exception is JobLostError (404 from Modal):
            # the worker that held the in-memory job table is gone, and
            # polling the same id is pointless. On JobLost we clear
            # `server_job_id` so the next attempt does a fresh submit.
            server_job_id: str | None = None
            for attempt in range(MAX_ATTEMPTS):
                try:
                    if server_job_id is None:
                        # Space submits ~1/s globally so a batch ramps onto
                        # Modal instead of bursting into a 429 storm.
                        await _pace_submit()
                        server_job_id = await _post_generate(image_bytes, image_mime)
                        _queue_set(slot_id, job_id, "processing", task_id=server_job_id)
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
                except JobLostError as e:
                    logging.log(
                        "trellis.job_lost",
                        job_id=job_id,
                        task_id=server_job_id,
                        attempt=attempt,
                        reason=str(e)[:200],
                    )
                    server_job_id = None
                    if attempt == MAX_ATTEMPTS - 1:
                        raise
                    delay = _retry_delay(attempt, e)
                    await asyncio.sleep(delay)
                except RETRYABLE as e:
                    if attempt == MAX_ATTEMPTS - 1:
                        raise
                    # A 404 on POST /generate (server_job_id still None) is a
                    # flapping-deployment signal, not load: Modal is between
                    # versions, or the router cold-routed before the worker was
                    # ready. The exponential schedule (4s → 8s → ... → 60s)
                    # exists for rate limits and transient network errors; for
                    # 404, just immediately resubmit with the same image bytes.
                    # `server_job_id` is already None, so the next loop hits
                    # the fresh POST path.
                    is_fresh_post_404 = (
                        server_job_id is None
                        and isinstance(e, httpx.HTTPStatusError)
                        and e.response.status_code == 404
                    )
                    delay = 0.0 if is_fresh_post_404 else _retry_delay(attempt, e)
                    logging.log(
                        "trellis.retry",
                        job_id=job_id,
                        task_id=server_job_id,
                        attempt=attempt,
                        delay_s=delay,
                        reason=f"{type(e).__name__}: {str(e)[:200]}",
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
            raise AssertionError("unreachable")
    finally:
        # Whether we succeeded, errored, or were cancelled, the job is no
        # longer in flight. Drop unconditionally so a crashed task can't
        # leak into the queue snapshot.
        _queue_drop(slot_id, job_id)
