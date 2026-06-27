"""Hunyuan 3D 3.1 Rapid mesh generation via Tencent Cloud's DIRECT API.

A regenerate-asset backend, alongside Trellis and Hunyuan-Omni (both on the Modal
router). Unlike those, this calls Tencent Cloud API 3.0 itself over signed HTTPS:
`SubmitHunyuanTo3DRapidJob` returns a JobId, then `QueryHunyuanTo3DRapidJob` is
polled until the GLB is ready. Auth is TC3-HMAC-SHA256 request signing (stdlib
hmac/hashlib — no SDK); input is a single front image as `ImageBase64`.

Tencent allows ONE concurrent generation task per account — concurrent submits are
hard-rejected with `RequestLimitExceeded.JobNumExceed` — so a process-global
`asyncio.Semaphore(1)` IS the queue: it serializes the whole submit -> poll ->
download lifecycle, so callers (the regen worker fans out concurrently) line up and
run strictly one at a time. While a caller waits then runs, it registers in the
shared mesh queue snapshot (`waiting` then `processing`) so the dashboard's queue
panel reflects it like any other backend.

Exposes the same `generate_mesh(image, *, output_path, job_id, image_mime, bbox)`
signature as `app.services.threed` / `app.services.hunyuan`, so it slots into
`generation.MESH_BACKENDS` as the `"hunyuan-tencent"` backend. `bbox` is accepted
for signature uniformity but unused — the rapid API has no aspect-ratio control.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.core.types import BoundingBox
from app.services import mesh_jobs
from app.utils import logging

SCOPE = "hunyuan_tencent"

_HOST = "hunyuan.intl.tencentcloudapi.com"
_ENDPOINT = f"https://{_HOST}"
_SERVICE = "hunyuan"
_VERSION = "2023-09-01"
_ALGORITHM = "TC3-HMAC-SHA256"
_CONTENT_TYPE = "application/json; charset=utf-8"
_SUBMIT_ACTION = "SubmitHunyuanTo3DRapidJob"
_QUERY_ACTION = "QueryHunyuanTo3DRapidJob"

# The only region that currently hosts this action on the intl endpoint (others
# return UnsupportedRegion). Override via TENCENTCLOUD_REGION.
_DEFAULT_REGION = "ap-singapore"

# GLB shape knobs (Tencent rapid defaults). Module-level so they can be tuned in
# one place; read at submit time. EnableGeometry yields a texture-free white model
# (and forbids OBJ); EnablePBR adds PBR material textures.
RESULT_FORMAT = "GLB"
ENABLE_PBR = False
ENABLE_GEOMETRY = False

# Submit-then-poll cadence. Rapid jobs finish in ~50-120s; poll every 2s and cap a
# single job at 5 min so a wedged job can't block the 1-at-a-time queue forever.
_POLL_INTERVAL_S = 2.0
_POLL_TIMEOUT_S = 300.0
_HTTP_TIMEOUT_S = 60.0
_DOWNLOAD_TIMEOUT_S = 180.0

# THE queue AND the pool boundary. Tencent processes one task at a time across
# the account, so the whole lifecycle serializes here. Critically this is its OWN
# semaphore (sized from mesh_jobs.HUNYUAN_TENCENT_CONCURRENCY), NOT Modal's
# _inflight_sem — so a Hunyuan 3.1 job never consumes one of Modal's
# GENERATE_CONCURRENCY slots and the two pools can't starve each other. Process-
# global, created at import like the other services' module-level semaphores.
_GATE = asyncio.Semaphore(mesh_jobs.HUNYUAN_TENCENT_CONCURRENCY)


@dataclass(frozen=True)
class _Creds:
    secret_id: str
    secret_key: str


class TencentAPIError(RuntimeError):
    """A Tencent API 3.0 business/auth error (HTTP 200 carrying `Response.Error`)."""

    def __init__(self, code: str, message: str, request_id: str | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.request_id = request_id

    def is_transient(self) -> bool:
        # Rate-limit / fleeting server-side conditions worth re-polling rather than
        # failing an already-submitted job over.
        code = self.code or ""
        return "RequestLimit" in code or code.startswith(
            ("InternalError", "ResourceUnavailable", "FailedOperation.ServiceBusy"),
        )


def _creds() -> _Creds:
    secret_id = os.environ.get("TENCENTCLOUD_SECRET_ID")
    secret_key = os.environ.get("TENCENTCLOUD_SECRET_KEY")
    if not secret_id or not secret_key:
        raise RuntimeError(
            "Tencent credentials missing: set TENCENTCLOUD_SECRET_ID and "
            "TENCENTCLOUD_SECRET_KEY (server/.env)",
        )
    return _Creds(secret_id, secret_key)


def _region() -> str:
    return os.environ.get("TENCENTCLOUD_REGION") or _DEFAULT_REGION


def _sign_headers(creds: _Creds, region: str, action: str, payload: str, ts: int) -> dict[str, str]:
    """Build the TC3-HMAC-SHA256 headers for one API 3.0 POST. Signed headers are
    `content-type;host`, the date is the UTC date of `ts`, and the service is the
    host's first label (`hunyuan`) — per Tencent's canonical reference."""
    date = datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d")
    canonical_headers = f"content-type:{_CONTENT_TYPE}\nhost:{_HOST}\n"
    signed_headers = "content-type;host"
    hashed_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_request = "\n".join(
        ["POST", "/", "", canonical_headers, signed_headers, hashed_payload],
    )
    scope = f"{date}/{_SERVICE}/tc3_request"
    string_to_sign = "\n".join(
        [_ALGORITHM, str(ts), scope, hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()],
    )

    def _hmac(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    secret_date = _hmac(("TC3" + creds.secret_key).encode("utf-8"), date)
    secret_service = _hmac(secret_date, _SERVICE)
    secret_signing = _hmac(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    authorization = (
        f"{_ALGORITHM} Credential={creds.secret_id}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return {
        "Authorization": authorization,
        "Content-Type": _CONTENT_TYPE,
        "Host": _HOST,
        "X-TC-Action": action,
        "X-TC-Timestamp": str(ts),
        "X-TC-Version": _VERSION,
        "X-TC-Region": region,
    }


def _unwrap(resp: httpx.Response) -> dict[str, Any]:
    """Validate an API 3.0 response and return the inner `Response` object. A signed
    call can return HTTP 200 yet carry `Response.Error`; surface that as a
    `TencentAPIError` so the poll loop can tell transient from terminal."""
    resp.raise_for_status()
    body = resp.json()
    response = body.get("Response") if isinstance(body, dict) else None
    if not isinstance(response, dict):
        raise RuntimeError(f"unexpected Tencent response shape: {body!r}")
    err = response.get("Error")
    if err:
        raise TencentAPIError(
            str(err.get("Code", "")), str(err.get("Message", "")), response.get("RequestId"),
        )
    return response


async def _post_signed(
    http: httpx.AsyncClient, creds: _Creds, region: str, action: str, params: dict[str, Any],
) -> dict[str, Any]:
    # Sign the EXACT bytes sent: serialize once and post as bytes with an explicit
    # content-type so httpx can't re-encode it out of sync with the signature.
    payload = json.dumps(params)
    headers = _sign_headers(creds, region, action, payload, int(time.time()))
    resp = await http.post(
        _ENDPOINT, headers=headers, content=payload.encode("utf-8"), timeout=_HTTP_TIMEOUT_S,
    )
    return _unwrap(resp)


async def _poll(http: httpx.AsyncClient, creds: _Creds, region: str, tencent_job_id: str) -> list[dict[str, Any]]:
    """Poll QueryHunyuanTo3DRapidJob until DONE, returning `ResultFile3Ds`. Transient
    query errors are absorbed (re-poll); FAIL, a terminal API error, or the timeout
    raise."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _POLL_TIMEOUT_S
    while True:
        if loop.time() >= deadline:
            raise TimeoutError(f"Tencent job {tencent_job_id} did not finish in {_POLL_TIMEOUT_S:.0f}s")
        await asyncio.sleep(_POLL_INTERVAL_S)
        try:
            response = await _post_signed(http, creds, region, _QUERY_ACTION, {"JobId": tencent_job_id})
        except TencentAPIError as e:
            if not e.is_transient():
                raise
            continue
        except httpx.HTTPError:
            continue
        status = response.get("Status")
        if status == "DONE":
            files = response.get("ResultFile3Ds") or []
            if not files:
                raise RuntimeError("Tencent reported DONE but ResultFile3Ds was empty")
            return files
        if status == "FAIL":
            raise RuntimeError(
                f"Tencent job failed: {response.get('ErrorCode')}: {response.get('ErrorMessage')}",
            )


def _pick_url(files: list[dict[str, Any]]) -> str:
    """The download URL for the requested format (else the first asset)."""
    want = RESULT_FORMAT.upper()
    chosen = next((f for f in files if str(f.get("Type", "")).upper() == want), files[0])
    url = chosen.get("Url")
    if not url:
        raise RuntimeError(f"Tencent ResultFile3Ds entry has no Url: {chosen!r}")
    return str(url)


def _submit_params(image_b64: str) -> dict[str, Any]:
    params: dict[str, Any] = {"ImageBase64": image_b64, "ResultFormat": RESULT_FORMAT}
    if ENABLE_PBR:
        params["EnablePBR"] = True
    if ENABLE_GEOMETRY:
        params["EnableGeometry"] = True
    return params


async def generate_mesh(
    image: bytes | str,
    *,
    output_path: Path,
    job_id: str,
    image_mime: str = "image/png",
    bbox: BoundingBox | None = None,
) -> Path:
    """Run Hunyuan 3D 3.1 Rapid on `image` and save the GLB to `output_path`.

    `image` is raw image bytes or a URL (fetched first). `image_mime`/`bbox` are
    accepted for backend-signature uniformity but unused (Tencent infers the format
    from the bytes and has no aspect-ratio control). Serialized process-globally via
    `_GATE` so only one Tencent task is ever in flight, and registered in the shared
    mesh queue snapshot so it shows in the dashboard's queue panel.
    """
    creds = _creds()
    region = _region()
    slot_id = logging.current_slot_id()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh_jobs.mark_queued(slot_id, job_id, backend=SCOPE)
    try:
        async with _GATE, httpx.AsyncClient(follow_redirects=True) as http:
            if isinstance(image, str):
                fetched = await http.get(image, timeout=_DOWNLOAD_TIMEOUT_S)
                fetched.raise_for_status()
                image_bytes = fetched.content
            else:
                image_bytes = image
            image_b64 = base64.b64encode(image_bytes).decode("ascii")

            ack = await _post_signed(http, creds, region, _SUBMIT_ACTION, _submit_params(image_b64))
            tencent_job_id = ack.get("JobId")
            if not tencent_job_id:
                raise RuntimeError(f"Tencent submit returned no JobId: {ack!r}")
            tencent_job_id = str(tencent_job_id)
            mesh_jobs.mark_processing(slot_id, job_id, task_id=tencent_job_id, backend=SCOPE)
            logging.log(f"{SCOPE}.submit", job_id=job_id, task_id=tencent_job_id)

            files = await _poll(http, creds, region, tencent_job_id)
            resp = await http.get(_pick_url(files), timeout=_DOWNLOAD_TIMEOUT_S)
            resp.raise_for_status()
            output_path.write_bytes(resp.content)
            logging.log(f"{SCOPE}.done", job_id=job_id, task_id=tencent_job_id, saved=str(output_path))
            return output_path
    finally:
        mesh_jobs.unmark_queued(slot_id, job_id)
