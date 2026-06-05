"""Hunyuan-Omni mesh generation — a thin adapter over the shared spawn-and-poll
core in `app.services.mesh_jobs`.

Hunyuan lives on the same hosted "assets router" as Trellis, selected with the
`?model=hunyuan-omni` query on `POST /generate`. Only the Hunyuan-specific bits
live here — the base URL, the production knobs that shape the GLB, and the
`/generate` multipart body. The whole job lifecycle (poll/download, transient-
error retry, the in-flight cap + submit pacing, the live queue snapshot, and the
resumable completion cache) is shared with Trellis in `mesh_jobs`, including the
process-global concurrency budget (one container pool serves both models).

Shared controls (`disconnect_http`, `queue_snapshot`, `mark_queued`,
`unmark_queued`, `GENERATE_CONCURRENCY`) deliberately are NOT re-exported here:
they're cross-backend and belong to `mesh_jobs`. Import them from there.

The Hunyuan worker also keeps these knobs as server-side defaults (per its API
docs, editable in its own orchestrator). We pin them here too — mirroring how
`threed.py` pins the Trellis knobs — so the orchestrator's runs are reproducible
and the exact config is recorded in the resumable input hash. Set any knob to
match the worker's default to defer to it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from app.services import mesh_jobs

HUNYUAN_MODEL = "hunyuan-omni"

# Same router deployment as Trellis by default (Hunyuan is just a `?model=`
# selection on it); kept as its own env knob so the two can be split if the
# deployment ever diverges.
HUNYUAN_BASE_URL = os.environ.get(
    "HUNYUAN_BASE_URL",
    "https://starshot-aitools--starshot-assets-router-fastapi-app.modal.run",
)

# Hunyuan production knobs (the worker's documented defaults). Module-level so
# the playground / batch scripts can read or override them without round-
# tripping through env; read at submit time (inside `_form_fields` /
# `_hash_payload`), so an override before a call is honored.
HUNYUAN_CONTROL_TYPE = "point"          # one of: point, voxel, bbox, pose
HUNYUAN_CONTROL: dict[str, object] = {}  # control payload (authored client-side)
HUNYUAN_STEPS = 50                       # flow-matching steps
HUNYUAN_OCTREE_RESOLUTION = 512          # marching-cubes grid: 256 / 384 / 512
HUNYUAN_GUIDANCE_SCALE = 6.7             # CFG scale
HUNYUAN_SEED = 1234
HUNYUAN_TEXTURE = True                   # run the PBR paint stack after shape
HUNYUAN_REMOVE_BACKGROUND = True         # BiRefNet matte if no usable alpha
HUNYUAN_USE_EMA = False                  # use EMA shape weights
HUNYUAN_TEXT_PROMPT = ""                 # optional caption nudge


def _bool(value: bool) -> str:
    """Multipart form fields are strings; send bools the way the worker parses."""
    return "true" if value else "false"


def _form_fields(object_name: str) -> dict[str, str]:
    """The non-image multipart fields for Hunyuan's `POST /generate`. Hunyuan
    has no `object_name` field — the closest is `text_prompt`, a caption nudge,
    not an id — so `object_name` is intentionally unused in the request body
    (it still scopes the resumable cache via the core's `job_id`)."""
    return {
        "control_type": HUNYUAN_CONTROL_TYPE,
        "control": json.dumps(HUNYUAN_CONTROL),
        "steps": str(HUNYUAN_STEPS),
        "octree_resolution": str(HUNYUAN_OCTREE_RESOLUTION),
        "guidance_scale": str(HUNYUAN_GUIDANCE_SCALE),
        "seed": str(HUNYUAN_SEED),
        "texture": _bool(HUNYUAN_TEXTURE),
        "remove_background": _bool(HUNYUAN_REMOVE_BACKGROUND),
        "use_ema": _bool(HUNYUAN_USE_EMA),
        "text_prompt": HUNYUAN_TEXT_PROMPT,
    }


def _hash_payload(object_name: str) -> dict[str, object]:
    """Knobs that change the GLB — folded into the resumable input hash. Kept in
    sync with `_form_fields` (the core adds base_url + image sha256). `model`
    distinguishes Hunyuan submits from Trellis ones in the audit hash."""
    return {
        "model": HUNYUAN_MODEL,
        "control_type": HUNYUAN_CONTROL_TYPE,
        "control": HUNYUAN_CONTROL,
        "steps": HUNYUAN_STEPS,
        "octree_resolution": HUNYUAN_OCTREE_RESOLUTION,
        "guidance_scale": HUNYUAN_GUIDANCE_SCALE,
        "seed": HUNYUAN_SEED,
        "texture": HUNYUAN_TEXTURE,
        "remove_background": HUNYUAN_REMOVE_BACKGROUND,
        "use_ema": HUNYUAN_USE_EMA,
        "text_prompt": HUNYUAN_TEXT_PROMPT,
        "object_name": object_name,
    }


_BACKEND = mesh_jobs.MeshBackend(
    scope="hunyuan",
    base_url=HUNYUAN_BASE_URL,
    model_query=HUNYUAN_MODEL,
    form_fields=_form_fields,
    hash_payload=_hash_payload,
)


async def generate_mesh(
    image: bytes | str,
    *,
    output_path: Path,
    job_id: str,
    image_mime: str = "image/png",
) -> Path:
    """Run Hunyuan-Omni on `image` and save the textured GLB to `output_path`.
    See `mesh_jobs.generate_mesh` for the resumable / retry semantics."""
    return await mesh_jobs.generate_mesh(
        image,
        output_path=output_path,
        job_id=job_id,
        image_mime=image_mime,
        backend=_BACKEND,
    )
