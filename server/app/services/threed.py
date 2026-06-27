"""Trellis 2 mesh generation — a thin adapter over the shared spawn-and-poll
core in `app.services.mesh_jobs`.

The hosted "assets router" serves Trellis as its default model (no `?model=`
query needed). Trellis-specific config — the base URL and the production knobs
that shape the GLB — and the `/generate` multipart body live here; the whole
job lifecycle (poll/download, transient-error retry, the in-flight cap + submit
pacing, the live queue snapshot, and the resumable completion cache) is shared
with Hunyuan in `mesh_jobs`.

The shared surface (`generate_mesh`, `disconnect_http`, `GENERATE_CONCURRENCY`,
`queue_snapshot`, `mark_queued`, `unmark_queued`, `JobLostError`) is re-exported
here so existing callers keep importing it from `app.services.threed`.

The returned GLB has textures embedded in its binary chunk, but trimesh cannot
decode them unless Pillow is installed at import time. Pillow is a project
dependency (see pyproject.toml) for that reason.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.core.types import BoundingBox
from app.services import mesh_jobs
from app.services.mesh_jobs import (  # re-exported: keep the threed.* call surface stable
    GENERATE_CONCURRENCY,
    JobLostError,
    disconnect_http,
    mark_queued,
    queue_pools,
    queue_snapshot,
    unmark_queued,
)

__all__ = [
    "GENERATE_CONCURRENCY",
    "JobLostError",
    "disconnect_http",
    "generate_mesh",
    "mark_queued",
    "queue_pools",
    "queue_snapshot",
    "unmark_queued",
]

TRELLIS_BASE_URL = os.environ.get(
    "TRELLIS_BASE_URL",
    "https://starshot-aitools--starshot-assets-router-fastapi-app.modal.run",
)

# Trellis production knobs. Kept module-level so the playground / batch scripts
# can read or override them without round-tripping through env. Read at submit
# time (inside `_form_fields` / `_hash_payload`), so an override before a call
# is honored. `resolution` is a string per the API contract
# ("512" | "1024" | "1536").
TRELLIS_RESOLUTION = "1024"
TRELLIS_TEXTURE_SIZE = 2048
TRELLIS_DECIMATION_TARGET = 500_000
TRELLIS_SEED = 0


def _form_fields(object_name: str, bbox: BoundingBox | None) -> dict[str, str]:
    """The non-image multipart fields for Trellis' `POST /generate`. Trellis has
    no bbox/aspect-ratio control, so `bbox` is accepted (uniform backend hook)
    but ignored."""
    return {
        "seed": str(TRELLIS_SEED),
        "resolution": TRELLIS_RESOLUTION,
        "texture_size": str(TRELLIS_TEXTURE_SIZE),
        "decimation_target": str(TRELLIS_DECIMATION_TARGET),
        "object_name": object_name,
    }


def _hash_payload(object_name: str, bbox: BoundingBox | None) -> dict[str, object]:
    """Knobs that change the GLB — folded into the resumable input hash. Kept
    in sync with `_form_fields` (the core adds base_url + image sha256); `bbox`
    is ignored, matching `_form_fields`."""
    return {
        "resolution": TRELLIS_RESOLUTION,
        "texture_size": TRELLIS_TEXTURE_SIZE,
        "decimation_target": TRELLIS_DECIMATION_TARGET,
        "seed": TRELLIS_SEED,
        "object_name": object_name,
    }


_BACKEND = mesh_jobs.MeshBackend(
    scope="trellis",
    base_url=TRELLIS_BASE_URL,
    model_query=None,
    form_fields=_form_fields,
    hash_payload=_hash_payload,
)


async def generate_mesh(
    image: bytes | str,
    *,
    output_path: Path,
    job_id: str,
    image_mime: str = "image/png",
    bbox: BoundingBox | None = None,
) -> Path:
    """Run Trellis 2 on `image` and save the textured GLB to `output_path`.
    `bbox` is accepted for a uniform backend signature but unused (Trellis has no
    aspect-ratio control). See `mesh_jobs.generate_mesh` for the resumable /
    retry semantics."""
    return await mesh_jobs.generate_mesh(
        image,
        output_path=output_path,
        job_id=job_id,
        image_mime=image_mime,
        backend=_BACKEND,
        bbox=bbox,
    )
