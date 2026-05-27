"""Server entry that skips Trellis 2 + Nano Banana ONLY for frame shells
(encapsulating scenario). Anchor objects and the completion-loop pass
generate normally — they get real meshes. Frames render as bbox
wireframes in the viewer (kind="frame", same red as zones).

Bbox validators are advisory in production (see divider / generation),
so this entry only needs to short-circuit the frame mesh pass.
The referenced_ids validator still aborts the run on cycles /
unknown targets.

Used by `scripts/run_noframes.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

from app.core.types import Node  # noqa: E402
from app.pipeline import generation  # noqa: E402
from app.utils import logging  # noqa: E402

_orig_spawn_meshes = generation._spawn_meshes


async def _no_frame_spawn(
    *,
    resolved: list[Node],
    runs_dir: Path,
    run_id: str,
    scenario: Literal["anchor", "encapsulating"],
) -> list[Node]:
    if scenario != "encapsulating":
        return await _orig_spawn_meshes(
            resolved=resolved, runs_dir=runs_dir, run_id=run_id, scenario=scenario,
        )
    # Frame shell: stamp a sentinel mesh_url so divider._prior_zones still
    # filters it out, but skip Trellis + emit_model — the bbox event
    # already gives the viewer everything it needs to render the shell.
    out: list[Node] = []
    for node in resolved:
        logging.log("nomesh.skip", id=node.id, prompt=node.prompt)
        out.append(node.model_copy(update={"mesh_url": f"bbox://{node.id}"}))
    return out


generation._spawn_meshes = _no_frame_spawn

from app.api.routes import create_app  # noqa: E402

app = create_app()
