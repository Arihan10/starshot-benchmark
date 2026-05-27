"""Server entry that skips Trellis 2 + Nano Banana (image+mesh generation).

The full LLM pipeline still runs end-to-end — every zone and object gets
emitted as a bbox event, and the viewer renders them as wireframes. No
`.glb` files are written; no `model` events are emitted, so the client
never attempts a mesh fetch.

Bbox validation runs in production but is advisory (see divider /
generation), so this entry doesn't need to stub anything beyond the mesh
pass. The referenced_ids validator still aborts the run on cycles /
unknown targets, in this mode and every other.

Used by `scripts/run_bboxes_only.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

from app.core.types import Node  # noqa: E402
from app.pipeline import generation  # noqa: E402
from app.utils import logging  # noqa: E402


async def _bbox_only_spawn(
    *,
    resolved: list[Node],
    runs_dir: Path,  # noqa: ARG001
    run_id: str,  # noqa: ARG001
    scenario: Literal["anchor", "encapsulating"],  # noqa: ARG001
) -> list[Node]:
    # mesh_url stays non-None so divider._prior_zones keeps filtering
    # objects out of zone-plan context. We deliberately skip emit_model —
    # the caller already fired emit_bbox, which is all the viewer needs.
    out: list[Node] = []
    for node in resolved:
        logging.log("nomesh.skip", id=node.id, prompt=node.prompt)
        out.append(node.model_copy(update={"mesh_url": f"bbox://{node.id}"}))
    return out


generation._spawn_meshes = _bbox_only_spawn

from app.api.routes import create_app  # noqa: E402

app = create_app()
