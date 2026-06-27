"""Hunyuan 3D 3.1 Pro mesh generation via Runware's cloud `3dInference` API.

The cloud counterpart to `app.services.hunyuan` (which drives the self-hosted
"hunyuan-omni" worker on the Modal assets router). Here we call Tencent's
production model hosted by Runware — `tencent:hunyuan-3d@3.1-pro` — through the
same Runware SDK the playground uses for Trellis (see `app.playground`): submit
a `3dInference` task, then poll `getResponse` for the GLB URL.

Image-to-3D (one frontal image, or up to 8 multi-view images) plus an optional
text prompt; returns a hosted GLB URL per variation. The GLB shape knobs
(`GENERATE_TYPE`, `PBR`, `FACE_COUNT`, ...) are module-level so batch scripts
can override them in place — they're read at submit time. The per-object
`positive_prompt` is a call argument since it varies per node.

The pinned Runware SDK (0.5.9) predates this model's `faceCount` /
`generateType` knobs: its `ISettings` dataclass can't carry them, so we emit the
documented `settings` block through a small carrier that implements the SDK's
`to_request_dict` serialization contract. `positivePrompt` is sent at the top
level (where the SDK and Runware's API both expect it), not inside `inputs`.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from runware import File, I3dInference, I3dInputs, IAsyncTaskResponse, Runware

HUNYUAN_RUNWARE_MODEL = os.environ.get(
    "HUNYUAN_RUNWARE_MODEL", "tencent:hunyuan-3d@3.1-pro",
)

# GLB shape knobs (Runware's documented defaults). Module-level so batch
# scripts can override them without env round-trips; read at submit time.
GENERATE_TYPE: str = "Normal"   # "Normal" (textured) | "Geometry" (white model)
PBR: bool = False               # +$0.15 add-on when True; forbidden with "Geometry"
FACE_COUNT: int | None = None   # None -> Runware default (500000); a custom value is a +$0.15 add-on
NUMBER_RESULTS: int = 1         # 1-4 variations, each a different seed
SEED: int | None = None         # None -> Runware default
INCLUDE_COST: bool = True       # surface per-task cost on the result

_MIN_IMAGES = 1
_MAX_IMAGES = 8


@dataclass(frozen=True)
class Hunyuan3DAsset:
    """One generated variation: a hosted GLB URL plus task metadata."""

    glb_url: str
    seed: int | None
    cost: float | None
    task_uuid: str


@dataclass
class _Hunyuan3DSettings:
    """The `settings` block, serialized via the SDK's `to_request_dict`
    contract (the same hook `inference3d` calls on `ISettings` / `I3dInputs`).
    Separate from `ISettings` because SDK 0.5.9 doesn't model this model's
    `faceCount` / `generateType`."""

    faceCount: int | None = None
    generateType: str | None = None
    pbr: bool | None = None

    def to_request_dict(self) -> dict[str, Any]:
        payload = {
            k: v
            for k, v in (
                ("faceCount", self.faceCount),
                ("generateType", self.generateType),
                ("pbr", self.pbr),
            )
            if v is not None
        }
        return {"settings": payload} if payload else {}


_client: Runware | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> Runware:
    """Lazily connect a process-shared Runware client, reusing the live socket
    on subsequent calls (mirrors `app.playground`)."""
    global _client
    async with _client_lock:
        if _client is None:
            _client = Runware(
                api_key=os.environ["RUNWARE_API_KEY"],
                timeout=180,
                max_retries=0,
            )
            await _client.connect()
        else:
            await _client.ensureConnection()
        return _client


async def disconnect() -> None:
    """Close the shared Runware client. Call once during FastAPI lifespan
    teardown so the server exits cleanly."""
    global _client
    async with _client_lock:
        if _client is not None:
            with contextlib.suppress(Exception):
                await _client.disconnect()
            _client = None


def _normalize_images(images: bytes | str | Sequence[str]) -> list[str | File]:
    """Coerce input to the `inputs.images` array: each entry is a URL, data
    URI, base64, Runware UUID, or local file path (the SDK base64-encodes local
    paths itself). Raw bytes become a single PNG data URI."""
    refs: list[str | File] = []
    if isinstance(images, bytes):
        encoded = base64.b64encode(images).decode("ascii")
        refs.append(f"data:image/png;base64,{encoded}")
        return refs
    src = [images] if isinstance(images, str) else list(images)
    if not _MIN_IMAGES <= len(src) <= _MAX_IMAGES:
        raise ValueError(
            f"hunyuan 3d expects {_MIN_IMAGES}-{_MAX_IMAGES} images, got {len(src)}"
        )
    refs.extend(src)
    return refs


def _to_asset(result: Any, fallback_uuid: str) -> Hunyuan3DAsset:
    """Pull the GLB URL + metadata out of one `I3d` result. `files[0]` may be
    an SDK object or a raw dict depending on the delivery path."""
    outputs = getattr(result, "outputs", None)
    files = getattr(outputs, "files", None) if outputs is not None else None
    if not files:
        raise RuntimeError(f"hunyuan 3d result missing files: {result!r}")
    first = files[0]
    url = first.get("url") if isinstance(first, dict) else getattr(first, "url", None)
    if not url:
        raise RuntimeError(f"hunyuan 3d result missing url: {first!r}")
    return Hunyuan3DAsset(
        glb_url=url,
        seed=getattr(result, "seed", None),
        cost=getattr(result, "cost", None),
        task_uuid=getattr(result, "taskUUID", fallback_uuid),
    )


async def generate_3d(
    images: bytes | str | Sequence[str],
    *,
    positive_prompt: str | None = None,
    task_uuid: str | None = None,
) -> list[Hunyuan3DAsset]:
    """Run Hunyuan 3D 3.1 Pro on `images` (+ optional `positive_prompt`) and
    return one `Hunyuan3DAsset` per variation (`NUMBER_RESULTS`).

    `images` is a single frontal image or up to 8 multi-view images, each a URL,
    data URI, base64, Runware UUID, local file path, or — for a single image —
    raw PNG bytes.
    """
    if GENERATE_TYPE == "Geometry" and PBR:
        raise ValueError("pbr cannot be True when generateType is 'Geometry'")

    image_refs = _normalize_images(images)
    settings = _Hunyuan3DSettings(
        faceCount=FACE_COUNT, generateType=GENERATE_TYPE, pbr=PBR,
    )
    task = task_uuid or str(uuid.uuid4())

    request = I3dInference(
        taskUUID=task,
        model=HUNYUAN_RUNWARE_MODEL,
        positivePrompt=positive_prompt,
        seed=SEED,
        numberResults=NUMBER_RESULTS,
        outputType="URL",
        outputFormat="GLB",
        deliveryMethod="async",
        includeCost=INCLUDE_COST,
        inputs=I3dInputs(images=image_refs),
        settings=settings,  # pyright: ignore[reportArgumentType]
    )

    client = await _get_client()
    ack = await client.inference3d(request3d=request)
    results = (
        await client.getResponse(taskUUID=task, numberResults=NUMBER_RESULTS)
        if isinstance(ack, IAsyncTaskResponse)
        else ack
    )
    if not results:
        raise RuntimeError(f"hunyuan 3d returned no results for task {task}")
    return [_to_asset(result, task) for result in results]
