"""Call the deployed Modal attention worker from the API server.

Routes over the Modal **web endpoint** with a SPAWN + POLL handshake (plain HTTPS
— no modal client needed at runtime). Modal caps every web request at 150s (past
that it 303-redirects to a result URL, and an exhausted/interrupted chain shows
up as `httpx.RemoteProtocolError: Server disconnected without sending a
response`). A teacher-forced forward — especially a cold model load — routinely
exceeds 150s, so we never hold one HTTP request open for it:

  1. POST the item -> the endpoint SPAWNS the GPU job and returns a `call_id`
     immediately (the result is saved in Modal's result store).
  2. GET `/result/{call_id}` on a short interval until it's ready (200), still
     running (202), or failed (500) — each poll returns fast.

Configure the base URL with `ATTENTION_MODAL_URL`; it defaults to the deployed
`web` endpoint. The API server keeps building the export + storing the pulled
result, so the frontend flow (enqueue -> poll our server -> GET auto-load) is
unchanged; only the compute happens on Modal.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

# The deployed `web` ASGI app base URL (see modal_app.web). Override per-env.
DEFAULT_MODAL_URL = "https://starshot-aitools--starshot-attention-web.modal.run"
# Per-request timeouts: submit/poll are both meant to return quickly (spawn is
# instant once the web container is warm; the poll does a non-blocking
# `.get(timeout=0)` on the worker). They're generous for a cold web container,
# and — crucially — a flaky INDIVIDUAL call is retried, never fatal: the GPU job
# keeps running on Modal regardless, so we just keep polling until the result
# lands or the overall budget elapses.
_SUBMIT_TIMEOUT_S = 60.0
_POLL_TIMEOUT_S = 15.0
# Bound the CONNECT phase separately so an unreachable/redeploying Modal fails
# fast (seconds) instead of tying up a threadpool worker for the whole timeout.
_CONNECT_TIMEOUT_S = 8.0
_POLL_INTERVAL_S = 2.0
_SUBMIT_RETRIES = 3
_RETRY_BACKOFF_S = 2.0
# /enqueue carries the full export PER item, and the web container is small +
# highly concurrent — a big single body can OOM it, which surfaces as
# "RemoteProtocolError: Server disconnected without sending a response". So we
# stream items in SMALL chunks and RETRY transient disconnects per chunk.
_ENQUEUE_CHUNK = 4
_ENQUEUE_RETRIES = 3
# Overall budget for a single step (cold model download/load + the full forward).
_TOTAL_TIMEOUT_S = 1800.0


def _timeout(total: float) -> httpx.Timeout:
    return httpx.Timeout(total, connect=_CONNECT_TIMEOUT_S)


def _safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except Exception:  # noqa: BLE001 — non-JSON body (e.g. a gateway error page)
        return None


def modal_url() -> str:
    return os.environ.get("ATTENTION_MODAL_URL", DEFAULT_MODAL_URL).strip()


def modal_available() -> bool:
    """The modal backend is usable iff a web URL is configured."""
    return bool(modal_url())


def warm_modal(*, model_id: str | None = None) -> None:
    """Wake the GPU worker (loads weights on first ping). Best-effort."""
    base = modal_url().rstrip("/")
    payload: dict[str, Any] = {}
    if model_id:
        payload["model_id"] = model_id
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=30.0)) as client:
        client.post(f"{base}/warm", json=payload, timeout=60.0)


def _post_enqueue_chunk(client: httpx.Client, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST one small chunk, retrying transient disconnects / 5xx. A dropped
    connection (`RemoteProtocolError`) is an `httpx.TransportError`, so it's
    retried; a 4xx (bad model/payload) is surfaced immediately. Always raises a
    plain RuntimeError on failure so the caller (`attention_enqueue`) maps it to a
    clean 503 instead of a 500 stack trace."""
    last_err: Exception | None = None
    for attempt in range(_ENQUEUE_RETRIES):
        try:
            resp = client.post(url, json=payload, timeout=_timeout(_SUBMIT_TIMEOUT_S))
        except httpx.TransportError as e:  # incl. RemoteProtocolError (server dropped us)
            last_err = e
            time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
            continue
        if resp.status_code >= 500:  # transient server-side fault — retry
            last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
            continue
        if resp.status_code >= 400:  # client error — don't retry
            raise RuntimeError(f"modal enqueue rejected (HTTP {resp.status_code}): {resp.text[:200]}")
        return _safe_json(resp) or {}
    raise RuntimeError(f"modal enqueue failed after {_ENQUEUE_RETRIES} attempts: {last_err}")


def enqueue_modal(
    *,
    cell_hash: str,
    ident: dict[str, Any],
    prompt_version: str | None,
    items: list[dict[str, Any]],
    model_id: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Stream a cell's steps into the durable Modal queue (keyed by cell_hash).
    Each item carries its full `compute_item` (export) so the worker needs no
    callback. Items are POSTed in SMALL chunks (a big single body can OOM the web
    container → disconnect) and results are merged. Returns
    {accepted, cached, already_active}."""
    if not items:
        return {"accepted": [], "cached": [], "already_active": []}
    base = modal_url().rstrip("/")
    url = f"{base}/enqueue"
    accepted: list[int] = []
    cached: list[int] = []
    already_active: list[int] = []
    with httpx.Client(follow_redirects=True) as client:
        for start in range(0, len(items), _ENQUEUE_CHUNK):
            payload: dict[str, Any] = {
                "cell_hash": cell_hash, "ident": ident,
                "prompt_version": prompt_version,
                "items": items[start:start + _ENQUEUE_CHUNK], "force": force,
            }
            if model_id:
                payload["model_id"] = model_id
            body = _post_enqueue_chunk(client, url, payload)
            accepted += [int(x) for x in (body.get("accepted") or [])]
            cached += [int(x) for x in (body.get("cached") or [])]
            already_active += [int(x) for x in (body.get("already_active") or [])]
    return {"accepted": accepted, "cached": cached, "already_active": already_active}


def reset_modal(*, cell_hash: str, model_id: str | None = None) -> dict[str, Any]:
    """Reset a cell's PENDING compute on Modal — clears the model's durable queue
    partition (stale refs from a prior run/scene of this model) AND this cell's
    queued/running lists (see the web `/reset` route). Committed results are
    preserved. Called before a fresh compute-all so it never inherits phantom jobs."""
    base = modal_url().rstrip("/")
    payload: dict[str, Any] = {"cell_hash": cell_hash}
    if model_id:
        payload["model_id"] = model_id
    with httpx.Client(follow_redirects=True) as client:
        resp = client.post(f"{base}/reset", json=payload, timeout=_timeout(_SUBMIT_TIMEOUT_S))
        resp.raise_for_status()
        return _safe_json(resp) or {"cleared": []}


def pull_remote_queue(*, cell_hash: str) -> dict[str, Any]:
    """FAST status poll — queue metadata only (Dict reads on the web container, no
    Volume touch). Finished steps come back in `done`; fetch their (large) result
    blobs separately with `fetch_results`. This keeps the frequent poll cheap so a
    slow blob transfer can never freeze the status loop."""
    base = modal_url().rstrip("/")
    payload = {"cell_hash": cell_hash}
    with httpx.Client(follow_redirects=True) as client:
        resp = client.post(f"{base}/pull", json=payload, timeout=_timeout(_POLL_TIMEOUT_S))
        resp.raise_for_status()
        return _safe_json(resp) or {"queued": [], "running": [], "errors": {}, "done": []}


def fetch_results(*, cell_hash: str, event_indices: list[int]) -> dict[str, Any]:
    """Fetch finished result blobs for specific steps — the heavy Volume transfer,
    kept OUT of the frequent status poll. Each blob is read TARGETED from its
    deterministic Volume path; there is NO ack/write-back (the reads are idempotent
    and the server dedups via its own on-disk cache). Returns
    `{"results": [{event_index, input_key, prompt_version, stamp, result}, ...]}`."""
    if not event_indices:
        return {"results": []}
    base = modal_url().rstrip("/")
    payload = {"cell_hash": cell_hash, "event_indices": list(event_indices)}
    with httpx.Client(follow_redirects=True) as client:
        # `compact` payloads are small, so the submit-sized budget (connect-bounded)
        # is plenty — the big `.full.json` is never returned here, only via /blob.
        resp = client.post(f"{base}/results", json=payload, timeout=_timeout(_SUBMIT_TIMEOUT_S))
        resp.raise_for_status()
        return _safe_json(resp) or {"results": []}


def fetch_full_to_file(*, cell_hash: str, event_index: int, dest_path: str) -> bool:
    """STREAM one step's big `.full.json` result straight to `dest_path` — the
    on-demand heavy pull for a single step's token/present detail, kept OFF the
    frequent poll. Chunks are written as they arrive (the whole blob is NEVER held
    in memory on either tier) into a temp file, then atomically renamed, so a
    partial/interrupted transfer can't leave a half-written result. httpx's read
    timeout is PER-CHUNK, not total, so an arbitrarily large blob is fine as long
    as bytes keep flowing. Returns True when written, False when Modal has no
    committed result for the step yet (404 → caller retries later)."""
    base = modal_url().rstrip("/")
    payload = {"cell_hash": cell_hash, "event_index": int(event_index)}
    tmp = f"{dest_path}.{os.getpid()}.part"
    try:
        wrote = 0
        with httpx.Client(follow_redirects=True) as client, client.stream(
            "POST", f"{base}/blob", json=payload, timeout=_timeout(_SUBMIT_TIMEOUT_S),
        ) as resp:
            if resp.status_code == 404:
                return False
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
                    wrote += len(chunk)
        if wrote == 0:
            return False  # committed meta but full not visible yet → caller retries
        os.replace(tmp, dest_path)  # atomic: readers see all-or-nothing
        return True
    finally:
        try:
            os.unlink(tmp)  # no-op after a successful replace; cleans a failed transfer
        except OSError:
            pass


def remote_queue_status(*, cell_hash: str) -> dict[str, Any]:
    """Read-only remote queue snapshot by cell_hash (no result pickup)."""
    base = modal_url().rstrip("/")
    with httpx.Client(follow_redirects=True) as client:
        resp = client.get(f"{base}/queue", params={"cell_hash": cell_hash}, timeout=_timeout(_POLL_TIMEOUT_S))
        resp.raise_for_status()
        return _safe_json(resp) or {"queued": [], "running": [], "errors": {}}


def _modal_submit(client: httpx.Client, base: str, payload: dict[str, Any]) -> str:
    """POST a batch payload and return the spawned call id (retries transient faults)."""
    last_err: Exception | None = None
    for attempt in range(_SUBMIT_RETRIES):
        try:
            sub = client.post(base, json=payload, timeout=_SUBMIT_TIMEOUT_S)
            sub.raise_for_status()
            body = _safe_json(sub) or {}
            call_ids = body.get("call_ids") or ([body["call_id"]] if body.get("call_id") else [])
            if not call_ids:
                raise RuntimeError(f"modal submit returned no call id: {str(body)[:200]}")
            return call_ids[0]
        except httpx.TransportError as e:
            last_err = e
            time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
    raise RuntimeError(f"modal submit failed after {_SUBMIT_RETRIES} attempts: {last_err}")


def _modal_poll(client: httpx.Client, base: str, call_id: str, *, budget_s: float) -> Any:
    """Poll until the spawned job resolves or the budget elapses."""
    deadline = time.monotonic() + budget_s
    while True:
        try:
            resp = client.get(f"{base}/result/{call_id}", timeout=_POLL_TIMEOUT_S)
            if resp.status_code == 200:
                data = _safe_json(resp) or {}
                if data.get("status") == "done":
                    return data["result"]
            elif resp.status_code >= 500:
                body = _safe_json(resp)
                if isinstance(body, dict) and body.get("status") == "error":
                    raise RuntimeError(body.get("error") or "modal attention compute failed")
        except httpx.TransportError:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"modal attention did not finish within {budget_s:.0f}s")
        time.sleep(_POLL_INTERVAL_S)


def analyze_batch_via_modal(
    items: list[dict[str, Any]],
    *,
    model_id: str | None = None,
) -> list[dict[str, Any]]:
    """Submit a batch as ONE Modal spawn; poll until done; return per-item results."""
    if not items:
        return []
    payload: dict[str, Any] = {"items": items}
    if model_id:
        payload["model_id"] = model_id
    base = modal_url().rstrip("/")
    budget = _TOTAL_TIMEOUT_S * max(1, len(items))
    with httpx.Client(follow_redirects=True) as client:
        call_id = _modal_submit(client, base, payload)
        result = _modal_poll(client, base, call_id, budget_s=budget)
    if isinstance(result, list):
        return result
    return [{"ok": True, "index": 0, "result": result}]


def analyze_via_modal(
    export: dict[str, Any],
    *,
    remote_logprobs: dict[str, Any] | None = None,
    max_heads: int = 4,
    top_k: int = 12,
    max_query_tokens: int = 0,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Submit one step as a single-item batch (see analyze_batch_via_modal)."""
    item: dict[str, Any] = {
        "export": export,
        "remote_logprobs": remote_logprobs,
        "max_heads": max_heads,
        "top_k": top_k,
        "max_query_tokens": max_query_tokens,
    }
    mid = model_id or export["meta"]["model_id"]
    results = analyze_batch_via_modal([item], model_id=mid)
    if not results:
        raise RuntimeError("modal returned empty batch result")
    first = results[0]
    if not first.get("ok"):
        raise RuntimeError(first.get("error", "modal item failed"))
    return first["result"]
