"""Hunyuan3D-Omni structural mesh generation via its spawn-and-poll HTTP API.

Structural shell geometry (decided by app.services.structural) is routed here
instead of Trellis: Hunyuan3D-Omni is bbox-conditioned, so passing the node's
bounding box yields a mesh that respects its proportions rather than Trellis's
free-form silhouette.

The contract differs from Trellis (app.services.threed) in two ways:
  * `POST /generate` takes a `control_type` + `control` payload — we send the
    bbox as an origin-centered, aspect-normalized size vector — and returns a
    `call_id`.
  * `GET /result/{call_id}` folds status and download into one endpoint: 202
    while running, 200 with the GLB when done, 404 if the job is lost, 500 on
    worker failure.

Everything else — the shared httpx client (and its lifespan teardown), the
retry/backoff schedule, submit pacing, the in-flight queue snapshot behind the
/trellis/queue panel, and the resumable completion cache — is reused from
threed and app.utils.resumable, so structural jobs behave identically to
Trellis ones operationally and only the wire format above is Hunyuan-specific.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

import httpx

from app.core.types import BoundingBox
from app.services import threed
from app.utils import logging, resumable

HUNYUAN_BASE_URL = os.environ.get(
    "HUNYUAN_BASE_URL",
    "https://starshot-aitools--hunyuan3d-omni-web.modal.run",
)

# Generation knobs (Hunyuan3D-Omni API defaults). The bbox control is
# size-based and origin-centered — only the box extents are injected — so we
# always send the aspect-normalized size and let the downstream rescale fit the
# mesh into the node's true bbox.
HUNYUAN_STEPS = 50
HUNYUAN_OCTREE_RESOLUTION = 512
HUNYUAN_GUIDANCE_SCALE = 6.7
HUNYUAN_SEED = 1234
HUNYUAN_TEXTURE = True
HUNYUAN_REMOVE_BACKGROUND = True

# Hunyuan on A100 (shape + PBR paint) is much slower than 512-res Trellis, so
# poll less aggressively with a generous wall-clock budget. The retry budget,
# backoff, queue snapshot, and pacing are reused from threed via run_spawn_poll.
POLL_INTERVAL_SECONDS = 5.0
POLL_TIMEOUT_SECONDS = 1200.0

# Independent in-flight cap: Hunyuan hits a different Modal app (its own GPU
# pool) and its slower jobs would otherwise starve the shared Trellis
# semaphore. Submits are still spaced through threed's global pacing gate.
GENERATE_CONCURRENCY = 50
_inflight_sem = asyncio.Semaphore(GENERATE_CONCURRENCY)

_SCOPE = "hunyuan"


def _bbox_control(bbox: BoundingBox) -> list[float]:
    """The node's bbox as an origin-centered SIZE vector with the largest axis
    normalized to 1.0. Hunyuan's encoder only uses the box extents and the GLB
    is rescaled to the node's true bbox downstream, so the aspect ratio is all
    that needs to survive."""
    sx, sy, sz = bbox.size
    longest = max(sx, sy, sz)
    if longest <= 0:
        return [1.0, 1.0, 1.0]
    return [round(sx / longest, 6), round(sy / longest, 6), round(sz / longest, 6)]


async def _post_generate(image_bytes: bytes, image_mime: str, bbox: BoundingBox) -> str:
    http = await threed._get_http()
    files = {"image": ("image.png", image_bytes, image_mime)}
    data = {
        "control_type": "bbox",
        "control": json.dumps({"bbox": _bbox_control(bbox)}),
        "steps": str(HUNYUAN_STEPS),
        "octree_resolution": str(HUNYUAN_OCTREE_RESOLUTION),
        "guidance_scale": str(HUNYUAN_GUIDANCE_SCALE),
        "seed": str(HUNYUAN_SEED),
        "texture": str(HUNYUAN_TEXTURE).lower(),
        "remove_background": str(HUNYUAN_REMOVE_BACKGROUND).lower(),
    }
    resp = await http.post(
        f"{HUNYUAN_BASE_URL}/generate", files=files, data=data, timeout=60.0,
    )
    resp.raise_for_status()
    body = resp.json()
    call_id = body.get("call_id")
    if not call_id:
        raise RuntimeError(f"hunyuan /generate returned no call_id: {body!r}")
    return str(call_id)


async def _poll_result(call_id: str, *, timeout: float) -> bytes:
    """Poll `GET /result/{call_id}` until the GLB is ready, returning its bytes.

    202 → keep polling; 200 → the GLB; 404 → JobLostError (the worker holding
    the call is gone, resubmit fresh); 500/other → terminal worker failure.
    Transient network errors back off via threed's schedule without consuming
    the outer attempt budget."""
    http = await threed._get_http()
    deadline = asyncio.get_running_loop().time() + timeout
    transient_count = 0
    while True:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(
                f"hunyuan job {call_id} did not finish in {timeout:.0f}s"
            )
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        try:
            resp = await http.get(
                f"{HUNYUAN_BASE_URL}/result/{call_id}", timeout=180.0,
            )
        except (httpx.HTTPError, ConnectionError, TimeoutError) as e:
            delay = threed._retry_delay(transient_count, e)
            logging.log(
                "hunyuan.poll.retry",
                task_id=call_id,
                attempt=transient_count,
                delay_s=delay,
                reason=f"{type(e).__name__}: {str(e)[:200]}",
            )
            transient_count += 1
            await asyncio.sleep(delay)
            continue
        transient_count = 0
        if resp.status_code == 202:
            continue
        if resp.status_code == 200:
            return resp.content
        if resp.status_code == 404:
            raise threed.JobLostError(
                f"hunyuan job {call_id} not found on server "
                "(Modal worker likely restarted; resubmitting)"
            )
        raise RuntimeError(
            f"hunyuan worker failed ({resp.status_code}): {resp.text[:300]}"
        )


def _input_hash(image_bytes: bytes, bbox: BoundingBox) -> str:
    """Stable hash of every input that would change the GLB, logged on
    `hunyuan.submit` so two submits with identical inputs are visibly
    equivalent."""
    return resumable.hash_input({
        "base_url": HUNYUAN_BASE_URL,
        "control_type": "bbox",
        "bbox": _bbox_control(bbox),
        "steps": HUNYUAN_STEPS,
        "octree_resolution": HUNYUAN_OCTREE_RESOLUTION,
        "guidance_scale": HUNYUAN_GUIDANCE_SCALE,
        "seed": HUNYUAN_SEED,
        "texture": HUNYUAN_TEXTURE,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
    })


async def generate_structural_mesh(
    image: bytes | str,
    *,
    output_path: Path,
    job_id: str,
    bbox: BoundingBox,
    image_mime: str = "image/png",
) -> Path:
    """Run Hunyuan3D-Omni (bbox control) on `image` and save the GLB to
    `output_path`. Short-circuits on a prior `hunyuan.done` whose file still
    exists; otherwise drives a fresh submit + poll through the shared
    `threed.run_spawn_poll` runner with Hunyuan's own in-flight cap and scope,
    supplying only the bbox-conditioned submit and the 202/200 poll."""
    done = resumable.find_done(scope=_SCOPE, job_id=job_id)
    if done is not None:
        cached = Path(str(done["saved"]))
        if cached.exists():
            return cached

    image_bytes = await threed._fetch_url(image) if isinstance(image, str) else image
    input_hash = _input_hash(image_bytes, bbox)

    async def _submit() -> str:
        return await _post_generate(image_bytes, image_mime, bbox)

    async def _fetch(task_id: str) -> bytes:
        return await _poll_result(task_id, timeout=POLL_TIMEOUT_SECONDS)

    return await threed.run_spawn_poll(
        scope=_SCOPE,
        job_id=job_id,
        input_hash=input_hash,
        output_path=output_path,
        sem=_inflight_sem,
        submit=_submit,
        fetch=_fetch,
    )
