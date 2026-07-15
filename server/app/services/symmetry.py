"""Mesh symmetry for the generation pipeline (+ studio batch helper).

Why: each object is generated from ONE image, so 3D diffusion hallucinates the
unseen back of ambiguous flat panels (walls, floors, ceilings, doors). For
those we mirror the photographed half onto the other side so both faces match.

The decision is made ONCE per object, *before* the Nano-Banana image is
generated, so two things key off it:
  * `image_view_for` — symmetric panels are imaged from a 3/4 view (which
    captures the real depth/profile of the half we keep) instead of a flat
    head-on front view; everything else stays on the front view.
  * `apply_symmetrize` — after Trellis, the stored plane drives the mirror.

Cut planes (mesh frame, Trellis front = +Z); the name is the plane we mirror
ACROSS, the axis is its normal:
  * `xy` — mirror across the XY plane (reflect along Z, front/back). Walls,
    partitions, doors, vertical panels.
  * `xz` — mirror across the XZ plane (reflect along Y, top/bottom). Thin
    horizontal slabs (floor tile, ceiling panel).
  * `none` — leave the mesh unchanged (chairs, plants, asymmetric props).

The decision is logged as a `symmetry.decision` event so resumes / mesh
retries / regenerations replay the same plane instead of re-asking.

Studio:
  `build_symmetric_glb` runs symmetrize → rescale → optimize without the LLM.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Literal

import trimesh
from pydantic import BaseModel

from app.core.scene_context import ImageView
from app.core.slots import MODELS
from app.core.types import BoundingBox, Orientation
from app.services import llm
from app.utils import logging
from app.utils.geometry import export_glb, rescale_mesh_to_bbox, symmetrize_mesh

CutPlane = Literal["none", "xy", "xz"]

_SERVER_DIR = Path(__file__).resolve().parents[2]
_OPTIMIZE_DIR = _SERVER_DIR / "tools" / "optimize-assets"
_OPTIMIZE_SCRIPT = _OPTIMIZE_DIR / "optimize.mjs"
_NODE_BIN = os.environ.get("STARSHOT_NODE_BIN", "node")

# Mesh-frame axis (the cut plane's normal) + keep_positive for symmetrize_mesh.
# Trellis front = +Z.
_PLANE_PARAMS: dict[CutPlane, tuple[int, bool] | None] = {
    "none": None,
    "xy": (2, True),   # mirror across XY → reflect along Z: keep +Z front, mirror to -Z back
    "xz": (1, True),   # mirror across XZ → reflect along Y: keep +Y, mirror to -Y
}

SYSTEM_SYMMETRY_DECISION = """\
You are part of a text-to-3D scene pipeline. Each object is generated from ONE orthographic image. The 3D model’s “front” is always facing along the +Z axis or along the -Y axis; the opposite side is often hallucinated and generates incorrectly for ambiguous panels (walls, floors tiles, ceiling sections, doors, windows, glass panels, partition walls, etc.).

For such panels we mirror the front half of the mesh onto the other side so both 
faces match exactly. IF the object should be symmetrical across a plane and the backside may be hallucinated by a 3D model generator, pick exactly one cut plane:

  * `xy` — mirror across the XY plane (front/back symmetry along Z). Use for 
walls, vertical partitions, doors, windows, wall-mounted panels, etc..

  * `xz` — mirror across the XZ plane (top/bottom symmetry along Y). Use
for thin horizontal slabs (floor tile, ceiling panel, etc.) 
where the top and bottom surfaces would both be ambiguous from a single view.

  * `none` — do NOT symmetrize. Use for objects with a clear front/back design, anything asymmetrical, anything with enough context to understand what it is, etc. 

Respond with ONE JSON object: `cut_plane` must be exactly `none`, `xy`, or 
`xz`. No prose, no markdown, no code fences.
"""


class SymmetryDecisionOutput(BaseModel):
    cut_plane: CutPlane


async def decide_symmetry(
    *, prompt: str, node_id: str, encapsulating: bool = False,
) -> SymmetryDecisionOutput:
    """Lightweight LLM gate: should this object be symmetrized, and on which
    plane? Always runs on gemini-flash-lite (cheap retrieval, not the benchmark
    model surface). Cached via `cache.llm` on the call key + (node_id, step)."""
    context = ""
    if encapsulating:
        # Encapsulating-pass pieces are the structural shell (walls / floor /
        # ceiling / perimeter) — the prime symmetry candidates. Tell the model.
        context = (
            "\nContext: this is a structural shell piece of an enclosure "
            "(a wall, floor, ceiling, or perimeter panel)."
        )
    user = (
        f"Object description: {prompt!r}{context}\n\n"
        "Choose cut_plane: none, xy, or xz."
    )
    token = llm._current_model.set(MODELS["gemini-flash-lite"])
    try:
        return await llm.call_llm(
            system=SYSTEM_SYMMETRY_DECISION,
            user=user,
            output_schema=SymmetryDecisionOutput,
            node_id=node_id,
            step="symmetry_decision",
        )
    finally:
        llm._current_model.reset(token)


async def resolve_cut_plane(
    *, prompt: str, node_id: str, encapsulating: bool = False,
) -> CutPlane:
    """The symmetry decision for `node_id`, made once and reused everywhere.

    Replays a prior `symmetry.decision` event when present (resume, mesh retry,
    regeneration), so the image-prompt view and the post-Trellis mirror always
    agree and the LLM is asked at most once. A decision failure degrades to
    `none` without aborting generation and is not logged as a decision, so a
    later retry can still attempt it."""
    prior = logging.find_event("symmetry.decision", id=node_id)
    if prior is not None:
        cp = prior.get("cut_plane")
        if cp in ("none", "xy", "xz"):
            return cp  # type: ignore[return-value]

    try:
        decision = await decide_symmetry(
            prompt=prompt, node_id=node_id, encapsulating=encapsulating,
        )
        cut_plane: CutPlane = decision.cut_plane
    except Exception as e:  # noqa: BLE001
        logging.log(
            "symmetry.skip",
            id=node_id,
            reason=f"decision_failed: {type(e).__name__}: {str(e)[:200]}",
        )
        return "none"

    logging.log(
        "symmetry.decision", id=node_id, cut_plane=cut_plane, encapsulating=encapsulating,
    )
    return cut_plane


def image_view_for(*, cut_plane: CutPlane, encapsulating: bool = False) -> ImageView:
    """The Nano-Banana view for an object given its symmetry decision. Panels we
    will mirror are imaged from a 3/4 view so Trellis captures the kept half's
    real depth and profile; everything else keeps the flat head-on front view."""
    return "three-quarter" if cut_plane != "none" else "front"


async def apply_symmetrize(
    mesh: trimesh.Trimesh | trimesh.Scene,
    *,
    cut_plane: CutPlane,
    node_id: str,
    keep_positive: bool | None = None,
) -> trimesh.Scene:
    """Mirror `mesh` per the already-resolved `cut_plane` (no LLM call). Returns
    the mesh unchanged for `none`; a mesh-level failure is logged and degrades
    to the original so generation never aborts over symmetry.

    Always returns a `Scene`: `symmetrize_mesh` yields a single `Trimesh`, which is
    re-wrapped so downstream `rescale_mesh_to_bbox` records placement as a node
    transform (a `Scene`'s transform lands in the scene graph, a `Trimesh`'s bakes
    into vertices) — keeping symmetrized objects on the same local-geometry +
    node-transform paradigm as un-symmetrized ones.

    `keep_positive` overrides which half is kept (None = the plane's default,
    +Z/+Y). It's recorded in the `symmetry.applied` event so a prefab reuse
    re-derived from this node's mesh mirrors the same half (see
    `_rescale_reuse_from_raw`)."""
    params = _PLANE_PARAMS.get(cut_plane)
    if params is None:
        return mesh if isinstance(mesh, trimesh.Scene) else trimesh.Scene(mesh)
    axis, default_keep = params
    keep = default_keep if keep_positive is None else keep_positive
    try:
        out = await asyncio.to_thread(
            symmetrize_mesh, mesh, axis=axis, keep_positive=keep,
        )
    except Exception as e:  # noqa: BLE001
        logging.log(
            "symmetry.skip",
            id=node_id,
            cut_plane=cut_plane,
            reason=f"mesh_failed: {type(e).__name__}: {str(e)[:200]}",
        )
        return mesh if isinstance(mesh, trimesh.Scene) else trimesh.Scene(mesh)
    logging.log("symmetry.applied", id=node_id, cut_plane=cut_plane, axis=axis, keep_positive=keep)
    return trimesh.Scene(out)


async def _optimize(src: Path, dst: Path) -> None:
    """Run one GLB through optimize.mjs (decimate + KTX2 + Meshopt) into `dst`."""
    src, dst = src.resolve(), dst.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        _NODE_BIN,
        str(_OPTIMIZE_SCRIPT),
        "--file", str(src),
        "--out-file", str(dst),
        cwd=str(_OPTIMIZE_DIR),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        detail = stderr.decode(errors="replace")[:500] if stderr else f"node exit {proc.returncode}"
        raise RuntimeError(f"optimize.mjs failed: {detail}")


def _natural_bbox(mesh: trimesh.Trimesh) -> BoundingBox:
    """The mesh's own AABB as a target bbox — a proportion-preserving rescale
    (identity at orientation 0) so the symmetry is shown without distortion."""
    lo, hi = mesh.bounds
    center = tuple(float(x) for x in (lo + hi) / 2.0)
    size = tuple(float(x) for x in (hi - lo))
    return BoundingBox.from_center_size(center, size)  # type: ignore[arg-type]


async def build_symmetric_glb(
    *,
    src: Path,
    raw_out: Path,
    opt_out: Path | None = None,
    axis: int = 2,
    keep_positive: bool = True,
    orientation: Orientation = 0,
    bbox: BoundingBox | None = None,
) -> dict[str, Any]:
    """Symmetrize `src` across `axis`, rescale into `bbox` (+ `orientation` yaw),
    export the raw symmetric GLB to `raw_out`, then optimize into `opt_out`.

    `bbox` defaults to the symmetric mesh's natural extents (no distortion). When
    `opt_out` is None the optimize step is skipped. Returns triangle / byte stats.
    """

    def _process() -> dict[str, Any]:
        scene = trimesh.load(src)
        symmetric = symmetrize_mesh(scene, axis=axis, keep_positive=keep_positive)
        target = bbox if bbox is not None else _natural_bbox(symmetric)
        triangles = int(len(symmetric.faces))
        # Wrap in a Scene so placement lands as a node transform (not baked into
        # vertices) — the same paradigm every other served GLB uses.
        placed = rescale_mesh_to_bbox(trimesh.Scene(symmetric), target, orientation=orientation)
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        export_glb(placed, raw_out)
        return {"triangles": triangles, "raw_bytes": raw_out.stat().st_size}

    stats = await asyncio.to_thread(_process)
    if opt_out is not None:
        await _optimize(raw_out, opt_out)
        stats["optimized_bytes"] = opt_out.stat().st_size
    return stats
