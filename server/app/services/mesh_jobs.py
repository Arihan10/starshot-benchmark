"""Shared spawn-and-poll core for the hosted image-to-3D "assets router".

One Modal deployment serves multiple mesh backends (Trellis as the default
model, Hunyuan via `?model=hunyuan-omni`) behind the same HTTP contract:

`POST /generate` uploads an image (multipart) and returns a server-assigned
`job_id` immediately — the GPU work runs detached so the request itself
doesn't have to outlive Modal's ~60s HTTP edge timeout. `GET /jobs/{job_id}`
is the non-blocking status probe; `GET /jobs/{job_id}/result` streams the GLB
binary back once status flips to `done`.

Everything here is backend-agnostic: the shared HTTP client, the poll/download
lifecycle, transient-error retry, the in-flight cap + submit pacing, the live
queue snapshot, and the resumable completion cache. The parts that DO vary per
backend — the base URL, the `?model=` query, the `/generate` form fields, and
the knobs folded into the resumable input hash — are supplied by a
`MeshBackend` spec. `app.services.threed` (Trellis) and `app.services.hunyuan`
(Hunyuan) are thin adapters that build that spec and delegate to
`generate_mesh` here.

Because it's one deployment, the in-flight semaphore, submit pacing, and queue
snapshot are process-global and SHARED across both backends — there's a single
container pool to stay under, so the budget is counted once across Trellis +
Hunyuan rather than per-backend.

Image generation lives in `app.services.nano_banana`; callers run that first,
then pass the resulting bytes (or a hosted URL) here.

Every job that reaches the submit loop logs exactly one terminal event —
`<scope>.done` or `<scope>.abandoned` — so a reader of the log can always tell a
finished job from one that ended with nothing to show for it. The API layer folds
those into the per-asset failure list the client renders.

Restart-resilience on completed work lives in `app.utils.resumable`: if
`<scope>.done` was logged and the saved GLB still exists, we short-circuit and
reuse it. Anything not done re-submits fresh — we don't probe stale Modal
task_ids because Modal GCs them on its own schedule and the in-flight semaphore
(capped at Modal's container count) keeps Modal-side queueing at zero, so each
fresh submit goes straight to a GPU.

The returned GLB has textures embedded in its binary chunk, but trimesh cannot
decode them unless Pillow is installed at import time. Pillow is a project
dependency (see pyproject.toml) for that reason.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from app.core.types import BoundingBox
from app.utils import logging, resumable


@dataclass(frozen=True)
class MeshBackend:
    """Per-backend knobs the shared lifecycle needs. Everything else (poll,
    download, retry, concurrency, queue, resumable cache) is identical across
    backends and lives in this module.

    * `scope` — event-kind prefix + resumable scope (`<scope>.submit` /
      `.retry` / `.done`), distinct per backend so their logs and completion
      caches never collide (e.g. `trellis` vs `hunyuan`).
    * `base_url` — the router deployment.
    * `form_fields(object_name, bbox)` — the non-image multipart fields for this
      backend's `POST /generate` (the image part is added by the core). `bbox`
      is the job's target box, an optional per-request shape hint (Hunyuan turns
      it into its bbox/aspect-ratio control; Trellis ignores it).
    * `hash_payload(object_name, bbox)` — the knobs that change the GLB, folded
      into the resumable input hash alongside `base_url` + the image's sha256.
    * `model_query` — value for the `?model=` selector, or None for the
      router's default model (Trellis).
    """

    scope: str
    base_url: str
    form_fields: Callable[[str, BoundingBox | None], dict[str, str]]
    hash_payload: Callable[[str, BoundingBox | None], dict[str, object]]
    model_query: str | None = None


# Spawn-and-poll cadence. The API caps any single job at ~10 min wall-clock;
# warm 512-res jobs finish in ~3s, 1536-res in ~60s, cold starts add 30-90s.
# 12s × GENERATE_CONCURRENCY=100 = 500 polls/min aggregate — OK while Modal's
# LB spreads it across the 100 containers (~5/min each, under the
# per-IP-per-container `/jobs/{id}` 60/60s limit), but a pin to a single
# container would now exceed it.
POLL_INTERVAL_SECONDS = 12.0
POLL_TIMEOUT_SECONDS = 3600.0

MAX_ATTEMPTS = 3
# Download has its own (larger) budget. The status endpoint can flip to
# "done" a few hundred ms before the GLB is committed to storage, so the
# result endpoint may 425 ("Too Early") for several seconds after. We don't
# want to give up — the job ran, the bytes are coming.
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
    """The router's job registry returned 404 for our task_id. The job is
    permanently gone — most often because the worker container that held the
    in-memory job table restarted (auto-scale, deploy, OOM). Polling the same
    id again is hopeless; the outer retry loop should treat this as a resubmit
    signal: drop the dead task_id and call `_post_generate` again on the next
    attempt."""


# Every `generate_mesh` call that gets as far as submitting ends with EXACTLY one
# terminal event: `<scope>.done` on success, or `<scope>.abandoned` on any other
# exit. The abandoned case is logged from a `finally`, so it also covers the paths
# that previously logged nothing at all — notably task cancellation, which raises
# BaseException and so slips past the `except Exception` in the pipeline's
# `_generate_one`. That invariant is what lets the client tell "still running"
# apart from "ended with no mesh and no explanation".
ABANDONED_CANCELLED = "cancelled before the backend returned a result"


# Cap on in-flight jobs at any moment (process-global FIFO across all slots AND
# backends — one container pool serves them all). Submits are additionally
# spaced by `_pace_submit` so a burst up to this many ramps onto the endpoint
# instead of hitting it all at once — the router 429s on bursts. Modal
# autoscales / queues beyond its live GPU count, so a "pending" status here can
# mean queued rather than actively processing.
GENERATE_CONCURRENCY = 100
_inflight_sem = asyncio.Semaphore(GENERATE_CONCURRENCY)

# Hunyuan 3.1 (Tencent's direct rapid API) is a SEPARATE concurrency pool from
# the Modal container pool above. Tencent hard-caps an account to one in-flight
# rapid job, and — by design — a Hunyuan 3.1 job must never consume one of
# Modal's GENERATE_CONCURRENCY slots, so the two backends can't starve each
# other. The semaphore that enforces this lives in app.services.hunyuan_tencent;
# this constant is its size, kept here so the queue panel and that gate share a
# single source of truth.
HUNYUAN_TENCENT_CONCURRENCY = 1

# Concurrency pools surfaced to the queue panel. A pool is an independent budget;
# each backend scope maps to exactly one, and the panel renders one section per
# pool with that pool's own cap — so Trellis/Modal work and Hunyuan 3.1 work are
# counted and displayed separately rather than looking like one shared budget.
# Hunyuan-Omni rides the Modal router, so it shares Modal's pool with Trellis.
POOL_MODAL = "modal"
POOL_HUNYUAN_TENCENT = "hunyuan-tencent"
# Accepts both the service scopes ("hunyuan_tencent") and the API backend names
# ("hunyuan-tencent"), so a row is grouped correctly no matter which tagged it.
_SCOPE_POOL = {
    "trellis": POOL_MODAL,
    "hunyuan": POOL_MODAL,
    "hunyuan-omni": POOL_MODAL,
    "hunyuan_tencent": POOL_HUNYUAN_TENCENT,
    "hunyuan-tencent": POOL_HUNYUAN_TENCENT,
}


def _pool_for(scope: str | None) -> str:
    return _SCOPE_POOL.get(scope or "", POOL_MODAL)


def queue_pools() -> list[dict[str, Any]]:
    """Ordered pool sections for the queue panel — each its own independent cap.
    Caps are read live so a script overriding GENERATE_CONCURRENCY is reflected."""
    return [
        {"id": POOL_MODAL, "label": "Trellis · Modal", "cap": GENERATE_CONCURRENCY},
        {
            "id": POOL_HUNYUAN_TENCENT,
            "label": "Hunyuan 3.1 · Tencent",
            "cap": HUNYUAN_TENCENT_CONCURRENCY,
        },
    ]


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


# Live snapshot of in-flight work. Process-global so the queue view reflects the
# same scope as the semaphore (one budget across slots + backends). Keyed by
# (slot_id, job_id) since node ids are unique within a slot but can collide
# across slots. Lifecycle:
#   waiting    — `generate_mesh` was called, awaiting `_inflight_sem`
#   processing — semaphore acquired, Modal task_id assigned
#   (removed)  — `generate_mesh` returned (success) or raised
# Read via `queue_snapshot()`; the GET /trellis/queue endpoint hands it to the
# client, which polls every ~1.5s. We expose this instead of letting the client
# infer state from the streamed event log because the event log replays
# historical submits on every SSE subscribe, with no way to distinguish "still
# running" from "process was killed before it could log .done" — so historical
# inference leaks stale rows. Live state, by contrast, resets to empty on
# process restart, which is exactly the truth.
_QUEUE: dict[tuple[str, str], dict[str, Any]] = {}


def queue_snapshot() -> list[dict[str, Any]]:
    """Current in-flight + waiting jobs across all slots and backends. Each row
    carries its `backend` scope and the concurrency `pool` it draws from, so the
    panel can split Trellis/Modal rows from Hunyuan 3.1 rows into their own
    sections."""
    out: list[dict[str, Any]] = []
    for (slot_id, job_id), entry in _QUEUE.items():
        backend = entry.get("backend")
        out.append({
            "slot_id": slot_id,
            "job_id": job_id,
            "state": entry["state"],
            "since": entry["since"],
            # First-seen epoch time (total in-flight); the canonical this row
            # shares a mesh with, so the panel can nest reuses under it (None on
            # a canonical / standalone job).
            "enqueued_at": entry.get("enqueued_at", entry["since"]),
            "canonical": entry.get("canonical"),
            "task_id": entry.get("task_id"),
            "backend": backend,
            "pool": _pool_for(backend),
        })
    return out


def _queue_set(slot_id: str | None, job_id: str, state: str, **extra: Any) -> None:
    if slot_id is None:
        return
    key = (slot_id, job_id)
    cur = _QUEUE.get(key)
    now = time.time()
    merged = {
        k: v for k, v in (cur or {}).items() if k not in {"state", "since", "enqueued_at"}
    }
    # Overlay only explicitly-provided fields, so a state transition that omits
    # `backend`/`task_id`/`canonical` keeps what an earlier call recorded (e.g.
    # mark_queued tags the backend; the later mark_processing needn't repeat it).
    merged.update({k: v for k, v in extra.items() if v is not None})
    _QUEUE[key] = {
        "state": state,
        # `since` marks the CURRENT state's start (resets on transition);
        # `enqueued_at` is the first-seen time and never resets, so the panel can
        # show total time-in-flight across the waiting→processing handoff.
        "since": cur["since"] if cur and cur["state"] == state else now,
        "enqueued_at": (cur or {}).get("enqueued_at", now),
        **merged,
    }


def _queue_drop(slot_id: str | None, job_id: str) -> None:
    if slot_id is None:
        return
    _QUEUE.pop((slot_id, job_id), None)


def inflight_ids(slot_id: str) -> set[str]:
    """Job ids currently waiting or processing for `slot_id` in the global queue."""
    return {jid for (sid, jid) in _QUEUE if sid == slot_id}


def mark_queued(
    slot_id: str | None,
    job_id: str,
    *,
    backend: str | None = None,
    canonical: str | None = None,
) -> None:
    """Register an externally-managed job (e.g. a regeneration awaiting its turn
    in a per-cell worker) as `waiting` in the shared queue snapshot, so it shows
    in the same queue panel as live mesh work. `backend` tags the row so the
    panel buckets it into the right pool section. `canonical` (a prefab
    canonical's job id) nests this row under that canonical's entry — set it on a
    reuse so the shared-mesh group shows as one expandable entry. When the job
    actually submits, `generate_mesh` takes over the (slot_id, job_id) entry; pair
    this with `unmark_queued` at hand-off / cancellation so it can't leak."""
    _queue_set(slot_id, job_id, "waiting", backend=backend, canonical=canonical)


def unmark_queued(slot_id: str | None, job_id: str) -> None:
    """Drop an entry registered via `mark_queued` — the job is starting (so
    `generate_mesh` will manage its own entry) or was cancelled before it ran."""
    _queue_drop(slot_id, job_id)


def mark_processing(
    slot_id: str | None,
    job_id: str,
    *,
    task_id: str | None = None,
    backend: str | None = None,
    canonical: str | None = None,
) -> None:
    """Promote a queue entry to `processing` for an externally-managed job that
    runs its own submit/poll lifecycle outside `generate_mesh` — e.g. the direct
    Tencent Hunyuan backend, which still belongs in the shared queue panel as the
    live in-flight row. `backend` tags the row's pool section; `canonical` nests
    it under a shared-mesh canonical (see `mark_queued`). Pair with `mark_queued`
    (waiting) and `unmark_queued`."""
    _queue_set(slot_id, job_id, "processing", task_id=task_id, backend=backend, canonical=canonical)


def _retry_delay(attempt: int, err: BaseException) -> float:
    """Sleep before the next retry. 429s with a `Retry-After` header use that
    hint verbatim; everything else uses exponential backoff capped at
    `RETRY_BACKOFF_MAX_S` with [0, 1)s jitter."""
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


# Shared async HTTP client, lazily initialized. Single client (across both
# backends) so HTTP/2 connection pooling kicks in across concurrent jobs.
_http: httpx.AsyncClient | None = None
_http_lock = asyncio.Lock()


async def _get_http() -> httpx.AsyncClient:
    global _http
    async with _http_lock:
        if _http is None:
            _http = httpx.AsyncClient(follow_redirects=True)
        return _http


async def disconnect_http() -> None:
    """Close the shared HTTP client. Call once during FastAPI lifespan teardown
    so the server exits cleanly."""
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


async def _post_generate(
    backend: MeshBackend, image_bytes: bytes, image_mime: str, object_name: str,
    bbox: BoundingBox | None,
) -> str:
    http = await _get_http()
    files = {"image": ("image.png", image_bytes, image_mime)}
    data = backend.form_fields(object_name, bbox)
    params = {"model": backend.model_query} if backend.model_query else None
    resp = await http.post(
        f"{backend.base_url}/generate",
        params=params, files=files, data=data, timeout=60.0,
    )
    resp.raise_for_status()
    body = resp.json()
    server_job_id = body.get("job_id")
    if not server_job_id:
        raise RuntimeError(f"{backend.scope} /generate returned no job_id: {body!r}")
    return str(server_job_id)


async def _poll_status(backend: MeshBackend, server_job_id: str) -> dict[str, Any]:
    """Single non-blocking status probe. Caller drives the poll loop."""
    http = await _get_http()
    resp = await http.get(
        f"{backend.base_url}/jobs/{server_job_id}", timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


async def _poll_until_done(
    backend: MeshBackend, server_job_id: str, *, timeout: float,
) -> None:
    """Block until status flips to `done`. Raises on `failed`, our own
    poll-timeout budget, or `JobLostError` when the router returns 404 for the
    task_id.

    Transient errors on the status endpoint (429s, network blips) do NOT tear
    out of the poll — they log a `<scope>.poll.retry` and back off, then the
    loop checks status again. 404, on the other hand, is terminal: the worker
    that held this job is gone and polling the same id will never recover."""
    deadline = asyncio.get_running_loop().time() + timeout
    transient_count = 0
    while True:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(
                f"{backend.scope} job {server_job_id} did not finish in {timeout:.0f}s"
            )
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        try:
            status = await _poll_status(backend, server_job_id)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise JobLostError(
                    f"{backend.scope} job {server_job_id} not found on server "
                    "(Modal worker likely restarted; resubmitting)"
                ) from e
            delay = _retry_delay(transient_count, e)
            logging.log(
                f"{backend.scope}.poll.retry",
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
                f"{backend.scope}.poll.retry",
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
                f"{backend.scope} worker failed: "
                f"{err.get('type', '?')}: {err.get('message', '?')}"
            )
        # "pending" → keep polling


async def _download_result(backend: MeshBackend, server_job_id: str) -> bytes:
    http = await _get_http()
    for attempt in range(DOWNLOAD_MAX_ATTEMPTS):
        try:
            resp = await http.get(
                f"{backend.base_url}/jobs/{server_job_id}/result",
                timeout=180.0,
            )
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise JobLostError(
                    f"{backend.scope} job {server_job_id} not found on server "
                    "during result download (Modal worker likely restarted; "
                    "resubmitting)"
                ) from e
            if attempt == DOWNLOAD_MAX_ATTEMPTS - 1:
                raise
            delay = _retry_delay(attempt, e)
            logging.log(
                f"{backend.scope}.download.retry",
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
                f"{backend.scope}.download.retry",
                task_id=server_job_id,
                attempt=attempt,
                delay_s=delay,
                reason=f"{type(e).__name__}: {str(e)[:200]}",
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


def _input_hash(
    backend: MeshBackend, image_bytes: bytes, object_name: str,
    bbox: BoundingBox | None,
) -> str:
    """Stable hash of every input that would change the GLB. Logged on
    `<scope>.submit` for audit so two submits with identical inputs are visibly
    equivalent. The backend supplies its own knobs; `base_url` and the image
    sha256 are common to every backend."""
    return resumable.hash_input({
        **backend.hash_payload(object_name, bbox),
        "base_url": backend.base_url,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
    })


def log_abandoned(
    scope: str, job_id: str, task_id: str | None, reason: str,
) -> None:
    """Record `<scope>.abandoned` — the job ended without a `<scope>.done`.
    `task_id` is the backend job we walked away from (None when we never got one).
    Shared with `app.services.hunyuan_tencent`, which runs its own submit/poll
    lifecycle but owes the log the same terminal event.

    Emitted from a `finally`, so it must not raise: an unbound SlotLog (a script
    or a test calling `generate_mesh` outside a run) would otherwise replace the
    exception that is already unwinding with a `LookupError`."""
    try:
        logging.log(
            f"{scope}.abandoned", job_id=job_id, task_id=task_id, reason=reason,
        )
    except LookupError:
        logging.console_note(f"[{scope}.abandoned] {job_id}: {reason}")


async def generate_mesh(
    image: bytes | str,
    *,
    backend: MeshBackend,
    output_path: Path,
    job_id: str,
    image_mime: str = "image/png",
    bbox: BoundingBox | None = None,
    force: bool = False,
) -> Path:
    """Run `backend` on `image` and save the textured GLB to `output_path`.

    `image` is either raw image bytes uploaded via multipart, or a remote URL
    which we fetch to bytes first (the API doesn't accept URLs server-side).
    `bbox` is an optional per-request shape hint handed to the backend's
    `form_fields` (Hunyuan uses it for bbox control; Trellis ignores it).

    Restart-resilient on completed work: if `<scope>.done` was previously logged
    for `job_id` and the file at the recorded path still exists, the call
    short-circuits. Otherwise we always issue a fresh `POST /generate` — we don't
    probe prior Modal task_ids because Modal GCs them on its own schedule, and
    the in-flight semaphore (`_inflight_sem`) gates submits at Modal's container
    count so Modal never queues on its side.

    `force=True` skips that completion-cache short-circuit and always regenerates,
    even when a prior `<scope>.done` + on-disk GLB exist. This is how a regenerate
    forces a fresh mesh WITHOUT first deleting the old one: the new GLB is written
    atomically (temp + replace) at the end, so `output_path` is only ever REPLACED
    on success — a regen that fails leaves the prior mesh intact instead of a
    headless node.
    """
    if not force:
        done = resumable.find_done(scope=backend.scope, job_id=job_id)
        if done is not None:
            cached = Path(str(done["saved"]))
            if cached.exists():
                return cached

    image_bytes = await _fetch_url(image) if isinstance(image, str) else image
    input_hash = _input_hash(backend, image_bytes, job_id, bbox)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    slot_id = logging.current_slot_id()
    _queue_set(slot_id, job_id, "waiting", backend=backend.scope)
    # By default we hold `server_job_id` across outer retries — any retryable
    # failure during poll or download re-enters the same Modal job instead of
    # orphaning it and burning a fresh generation. The exception is JobLostError
    # (404 from Modal): the worker that held the in-memory job table is gone, and
    # polling the same id is pointless. On JobLost we clear `server_job_id` so the
    # next attempt does a fresh submit. Hoisted above the `try` so the terminal
    # event logged in the `finally` can name the task that was left behind.
    server_job_id: str | None = None
    logged_done = False
    exit_reason = ABANDONED_CANCELLED
    try:
        async with _inflight_sem:
            for attempt in range(MAX_ATTEMPTS):
                try:
                    if server_job_id is None:
                        # Space submits ~1/s globally so a batch ramps onto
                        # Modal instead of bursting into a 429 storm.
                        await _pace_submit()
                        server_job_id = await _post_generate(
                            backend, image_bytes, image_mime, job_id, bbox,
                        )
                        _queue_set(slot_id, job_id, "processing", task_id=server_job_id, backend=backend.scope)
                        logging.log(
                            f"{backend.scope}.submit",
                            job_id=job_id,
                            task_id=server_job_id,
                            input_hash=input_hash,
                            attempt=attempt,
                        )
                    await _poll_until_done(
                        backend, server_job_id, timeout=POLL_TIMEOUT_SECONDS,
                    )
                    content = await _download_result(backend, server_job_id)
                    # Atomic write: a concurrent reader (a prefab reuse rescaling
                    # this raw mesh) must never see a half-written GLB. Write to a
                    # temp in the same dir, then replace.
                    tmp_path = output_path.with_name(f"{output_path.name}.part")
                    tmp_path.write_bytes(content)
                    tmp_path.replace(output_path)
                    resumable.log_done(
                        scope=backend.scope,
                        job_id=job_id,
                        server_job_id=server_job_id,
                        saved=str(output_path),
                    )
                    logged_done = True
                    return output_path
                except JobLostError as e:
                    logging.log(
                        f"{backend.scope}.job_lost",
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
                    # `server_job_id` is already None, so the next loop hits the
                    # fresh POST path.
                    is_fresh_post_404 = (
                        server_job_id is None
                        and isinstance(e, httpx.HTTPStatusError)
                        and e.response.status_code == 404
                    )
                    delay = 0.0 if is_fresh_post_404 else _retry_delay(attempt, e)
                    logging.log(
                        f"{backend.scope}.retry",
                        job_id=job_id,
                        task_id=server_job_id,
                        attempt=attempt,
                        delay_s=delay,
                        reason=f"{type(e).__name__}: {str(e)[:200]}",
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
            raise AssertionError("unreachable")
    except Exception as e:
        exit_reason = f"{type(e).__name__}: {str(e)[:200]}"
        raise
    finally:
        # Cancellation raises BaseException, so it reaches this `finally` without
        # any `except` of ours seeing it — and `_generate_one` won't catch it
        # either. That is precisely the exit that used to leave no trace, so the
        # default `exit_reason` covers it.
        if not logged_done:
            log_abandoned(backend.scope, job_id, server_job_id, exit_reason)
        # Whether we succeeded, errored, or were cancelled, the job is no longer
        # in flight. Drop unconditionally so a crashed task can't leak into the
        # queue snapshot.
        _queue_drop(slot_id, job_id)
