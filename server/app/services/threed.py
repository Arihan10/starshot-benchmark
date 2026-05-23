"""Two-stage asset generation: text -> image (Nano Banana / Gemini) -> 3D (Trellis 2).

Text-to-3D alone produces unreliable geometry for thin architectural shells
(walls, ceilings, floors). Going through an image model first gives Trellis
a concrete visual reference with correct proportions, which is much more
robust.

The Banana stage calls Google's GenAI API directly (model `gemini-2.5-flash-image`
for nano-banana, `gemini-3-pro-image-preview` for nano-banana-pro). The
generated image is saved to disk and base64-encoded for the Trellis input.

The Trellis stage runs on Runware (https://runware.ai/) over its WebSocket
SDK. We use a caller-supplied `taskUUID` so the resumption record (logged
as `runware.submit`) can survive process restarts: on the next attempt the
same UUID is reused via `getResponse` and the in-flight or recently-finished
job returns its result without re-billing.

The returned GLB has textures embedded in its binary chunk, but trimesh
cannot decode them unless Pillow is installed at import time. Pillow is a
project dependency (see pyproject.toml) for that reason.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from runware import (
    I3dInference,
    I3dInputs,
    IAsyncTaskResponse,
    ISettings,
    Runware,
    RunwareAPIError,
)

from app.utils import cache, logging

# Google GenAI model IDs. `gemini-2.5-flash-image` is the public alias for
# "nano-banana"; `gemini-3-pro-image-preview` is nano-banana-pro.
NANO_BANANA_PRO = "gemini-3-pro-image-preview"
NANO_BANANA_2 = "gemini-2.5-flash-image"

NANO_BANANA_MODEL = os.environ.get("NANO_BANANA_MODEL", NANO_BANANA_2)
TRELLIS_MODEL = os.environ.get("TRELLIS_MODEL", "microsoft:trellis-2@4b")


MAX_ATTEMPTS = 3
# Trellis-side transients we retry on: Runware API errors, the httpx
# network-layer errors (RemoteProtocolError "stream closed", ReadError,
# ConnectError, timeouts, etc.) raised by the GLB download, and
# TimeoutError — the SDK raises this when its internal
# `asyncio.wait_for(future, ...)` fires because the WS reader stalled
# and no response message ever arrived. TimeoutError + ConnectionError
# from the WS path are specifically the signals that the shared
# WebSocket is sick; on those, the retry loop also recreates the
# client (see _get_client below) so the next attempt runs on a fresh
# WS instead of the same wedged one.
RETRYABLE: tuple[type[BaseException], ...] = (
    RunwareAPIError,
    httpx.HTTPError,
    ConnectionError,
    TimeoutError,
)

# Banana-side transients: any Google API error (covers ServerError and
# ClientError subclasses; the SDK's own retry logic is disabled so we
# see them here) plus generic network failures from the underlying HTTP
# transport.
GENAI_RETRYABLE: tuple[type[BaseException], ...] = (
    genai_errors.APIError,
    httpx.HTTPError,
    ConnectionError,
    TimeoutError,
)


# Module-level singleton Runware client. Lazy-init on first call so the
# server can come up even if Runware is briefly unreachable; reused across
# all calls because the SDK manages a single WebSocket connection that
# multiplexes requests internally.
_client: Runware | None = None
_client_lock = asyncio.Lock()


async def _get_client(known_bad: Runware | None = None) -> Runware:
    """Return the shared Runware client.

    Pass `known_bad` to force a fresh connection after a WS-level
    failure (TimeoutError / ConnectionError). The check `_client is
    known_bad` makes recreation idempotent: if 50 tasks all timed out
    against the same wedged client and all reach this point, the
    first one through the lock discards the old client and builds a
    new one; the rest see that `_client` is no longer their bad
    client (someone already replaced it) and just use the new one.
    Without this guard, concurrent retriers would keep discarding
    each other's freshly-built clients in a thrash."""
    global _client
    async with _client_lock:
        if known_bad is not None and _client is known_bad:
            # Bound disconnect with its own timeout — the underlying
            # WS is sick and a clean close may itself never return.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(_client.disconnect(), timeout=10)
            _client = None
        if _client is None:
            _client = Runware(
                api_key=os.environ["RUNWARE_API_KEY"],
                timeout=180,
                max_retries=0,            # outer loop manages retries
            )
            await _client.connect()
        else:
            await _client.ensureConnection()
        return _client


async def disconnect_runware() -> None:
    """Close the singleton WebSocket. Call once during FastAPI lifespan
    teardown so the server exits cleanly."""
    global _client
    async with _client_lock:
        if _client is not None:
            with contextlib.suppress(Exception):
                await _client.disconnect()
            _client = None


# Module-level Google GenAI client. Cheap to construct but we still keep
# a singleton so we don't churn HTTP pools on every Banana call.
_genai_client: genai.Client | None = None
_genai_client_lock = asyncio.Lock()


async def _get_genai_client() -> genai.Client:
    global _genai_client
    async with _genai_client_lock:
        if _genai_client is None:
            _genai_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        return _genai_client


async def generate_banana_image(
    prompt: str, *, node_id: str, model: str | None = None,
) -> tuple[bytes, str]:
    """Call Google's GenAI image-generation model and return
    (image_bytes, mime_type). Retries transient API/network errors up to
    MAX_ATTEMPTS; on the final attempt the exception propagates."""
    client = await _get_genai_client()
    use_model = model or NANO_BANANA_MODEL
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = await client.aio.models.generate_content(
                model=use_model,
                contents=prompt,
            )
            candidates = response.candidates or []
            for cand in candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) or []
                for part in parts:
                    data = getattr(part, "inline_data", None)
                    if data is not None and data.data:
                        return data.data, data.mime_type or "image/png"
            raise RuntimeError(
                f"banana returned no inline image (model={use_model!r})"
            )
        except GENAI_RETRYABLE as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logging.log(
                "nano_banana.retry",
                node_id=node_id,
                attempt=attempt,
                reason=f"{type(e).__name__}: {str(e)[:200]}",
            )
    raise AssertionError("unreachable")


_MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}


def _ext_from_mime(mime_type: str) -> str:
    return _MIME_EXT.get(mime_type.lower(), ".png")


def _mime_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    for mime, ext in _MIME_EXT.items():
        if ext == suffix or (ext == ".jpg" and suffix == ".jpeg"):
            return mime
    return "image/png"


def image_to_data_uri(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def file_to_data_uri(path: Path) -> str:
    return image_to_data_uri(path.read_bytes(), _mime_from_path(path))


def _build_trellis_request(
    arguments: dict[str, Any], task_uuid: str,
) -> I3dInference:
    """Translate the trellis arguments dict into the Runware request
    dataclass. Caller-supplied `task_uuid` is the resumption key — same
    value goes into the `runware.submit` event log and into the request."""
    return I3dInference(
        taskUUID=task_uuid,
        model=arguments["model"],
        inputs=I3dInputs(image=arguments["image"]),
        settings=ISettings(
            remesh=arguments["remesh"],
            resolution=arguments["resolution"],
            textureSize=arguments["textureSize"],
        ),
        outputFormat=arguments["outputFormat"],
        outputType=arguments["outputType"],
        deliveryMethod="async",
        numberResults=1,
    )


def _unwrap_trellis(item: Any) -> dict[str, Any]:
    """Normalize Trellis's SDK result dataclass into the plain dict
    `generate_mesh` consumes. The GLB URL is nested under
    `outputs.files[0]`."""
    outputs = getattr(item, "outputs", None)
    files = getattr(outputs, "files", None) if outputs else None
    if not files:
        raise RuntimeError(f"Trellis result missing outputs.files: {item!r}")
    first = files[0]
    url = first.get("url") if isinstance(first, dict) else getattr(first, "url", None)
    if not url:
        raise RuntimeError(f"Trellis result missing url: {first!r}")
    return {"glb_url": url}


async def _submit_trellis(
    arguments: dict[str, Any],
    *,
    node_id: str,
) -> dict[str, Any]:
    """Submit a Runware Trellis job with restart-resilient resumption.

    On entry, scan the events log for a prior `runware.submit` matching
    (node_id, stage="trellis", input_hash). If found, attempt
    `client.getResponse(taskUUID=...)` against the persisted UUID —
    Runware keeps task results around long enough that an in-flight or
    recently-completed job returns immediately with no new billing. On
    any RunwareAPIError from the reattach we treat the prior task as
    expired and fall through to a fresh submit.

    Fresh submits log `runware.submit` *before* awaiting the SDK call.
    Because we generate the taskUUID client-side (UUID v4), the
    resumption record is durable as soon as `SlotLog.log` flushes — no
    dependency on the response landing.
    """
    stage = "trellis"
    model = arguments["model"]
    input_hash = cache.hash_runware_input(model, arguments)
    prior = cache.find_runware_submit(
        logging.current_events(), node_id, stage, input_hash,
    )
    client = await _get_client()
    if prior is not None:
        try:
            results = await client.getResponse(
                taskUUID=prior["task_uuid"], numberResults=1,
            )
            if results:
                logging.log(
                    "runware.reattach",
                    node_id=node_id,
                    stage=stage,
                    task_uuid=prior["task_uuid"],
                    outcome="success",
                )
                return _unwrap_trellis(results[0])
            # Empty result list — treat as expired and submit fresh.
            logging.log(
                "runware.reattach",
                node_id=node_id,
                stage=stage,
                task_uuid=prior["task_uuid"],
                outcome="expired",
                reason="empty_result",
            )
        except (RunwareAPIError, TimeoutError, ConnectionError) as e:
            # Any error on reattach -> fall through to fresh submit.
            # The new submit's task_uuid overwrites the lookup, so the
            # next restart reattaches to the new task — no double-bill.
            # TimeoutError / ConnectionError additionally signal that
            # the WS is sick, so we discard the client before the
            # fresh-submit loop below opens a new one.
            logging.log(
                "runware.reattach",
                node_id=node_id,
                stage=stage,
                task_uuid=prior["task_uuid"],
                outcome="expired",
                reason=f"{type(e).__name__}: {str(e)[:200]}",
            )
            if isinstance(e, (TimeoutError, ConnectionError)):
                client = await _get_client(known_bad=client)

    for attempt in range(MAX_ATTEMPTS):
        task_uuid = str(uuid.uuid4())
        try:
            request = _build_trellis_request(arguments, task_uuid)
            logging.log(
                "runware.submit",
                node_id=node_id,
                stage=stage,
                model=model,
                task_uuid=task_uuid,
                input_hash=input_hash,
            )
            ack = await client.inference3d(request3d=request)
            if isinstance(ack, IAsyncTaskResponse):
                results = await client.getResponse(
                    taskUUID=task_uuid, numberResults=1,
                )
            else:
                results = ack
            if not results:
                raise RuntimeError(f"empty result list for task {task_uuid}")
            return _unwrap_trellis(results[0])
        except RETRYABLE as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logging.log(
                f"{stage}.retry",
                attempt=attempt,
                reason=f"{type(e).__name__}: {str(e)[:200]}",
            )
            # TimeoutError / ConnectionError mean the WS that delivered
            # (or failed to deliver) this response is unhealthy. Swap to
            # a fresh client so the next attempt isn't going to the same
            # wedged socket. Other RETRYABLE errors (RunwareAPIError,
            # httpx.HTTPError) are application- or download-layer and
            # don't indicate the WS is the problem — keep the client.
            if isinstance(e, (TimeoutError, ConnectionError)):
                client = await _get_client(known_bad=client)
    raise AssertionError("unreachable")


async def _download_with_retry(url: str, *, stage: str) -> bytes:
    for attempt in range(MAX_ATTEMPTS):
        try:
            async with httpx.AsyncClient(
                timeout=180.0,
                follow_redirects=True,
            ) as http:
                resp = await http.get(url)
                resp.raise_for_status()
                return resp.content
        except httpx.HTTPError as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logging.log(
                f"{stage}.download.retry",
                attempt=attempt,
                reason=f"{type(e).__name__}: {str(e)[:200]}",
            )
    raise AssertionError("unreachable")


async def generate_mesh(
    prompt: str,
    *,
    output_path: Path,
    image_stem: Path,
) -> dict[str, Path]:
    """Run text -> image -> 3D. Saves the reference image alongside the
    GLB so the client asset browser can display it. Returns both paths."""
    node_id = image_stem.name
    hit = cache.find_artifact_cache_hit(logging.current_events(), node_id)
    if hit is not None:
        cached_raw = Path(hit["raw_glb_path"])
        cached_image = Path(hit["image_path"])
        if cached_raw.exists() and cached_image.exists():
            logging.log(
                "cache.artifact.hit",
                node_id=node_id,
                image_path=str(cached_image),
                raw_glb_path=str(cached_raw),
            )
            return {"glb": cached_raw, "image": cached_image}

    # Banana-skip gate: if Banana already finished for this node and the
    # saved image is still on disk, skip the Banana call and reuse the
    # local file. Closes the Banana re-bill window for process deaths
    # between Banana and Trellis.
    image_path: Path | None = None
    banana_hit = cache.find_banana_done(logging.current_events(), node_id)
    if banana_hit is not None:
        candidate = Path(banana_hit["saved"])
        if candidate.exists():
            image_path = candidate
            logging.log("nano_banana.skip", node_id=node_id)

    if image_path is None:
        image_bytes, mime_type = await generate_banana_image(
            prompt, node_id=node_id, model=NANO_BANANA_MODEL,
        )
        ext = _ext_from_mime(mime_type)
        image_path = image_stem.parent / (image_stem.name + ext)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)
        logging.log(
            "nano_banana.done",
            node_id=node_id,
            saved=str(image_path),
        )

    # Trellis accepts the image as a base64 data URI in I3dInputs.image.
    # Re-reading from disk on every call keeps the Banana-skip path and
    # the fresh path symmetric — both encode the same on-disk bytes.
    trellis_args: dict[str, Any] = {
        "model": TRELLIS_MODEL,
        "image": file_to_data_uri(image_path),
        "remesh": False,
        "resolution": 512,
        "textureSize": 1024,
        "outputFormat": "GLB",
        "outputType": "URL",
    }
    mesh = await _submit_trellis(trellis_args, node_id=node_id)
    glb_url = mesh["glb_url"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = await _download_with_retry(glb_url, stage="trellis")
    output_path.write_bytes(content)
    logging.log(
        "cache.artifact",
        node_id=node_id,
        image_path=str(image_path),
        raw_glb_path=str(output_path),
    )
    return {"glb": output_path, "image": image_path}
