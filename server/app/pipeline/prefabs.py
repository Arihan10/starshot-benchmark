"""Per-run prefab registry — reuse already-built assets for consistency.

In generated mode every object would otherwise get its own Nano-Banana image +
Trellis mesh, so a scene with twenty "cream plaster wall panel"s pays for twenty
near-identical generations that also don't visually match each other. The prefab
system keeps a per-run catalog of the objects built (or still in flight) so far;
before a new object is generated, a lightweight LLM call decides whether it can
reuse one of those assets. A reuse skips image + mesh generation entirely and
instead rescales the matched asset's GLB into the new object's bounding box —
identical geometry, fitted to the new slot.

State is scoped per (run, mode) TRACK — the same `track_key` the generation
module keys its in-flight tables on — and is cleared when the run finishes or is
cancelled. Reuse of an asset that is still generating is supported via a per-node
"ready" Event: the reuse task waits for the source's mesh to land before
rescaling it. Resume is handled in `generation` (it replays the committed
`prefab.reuse` decision and checks the on-disk GLB), so the match LLM only ever
runs for genuinely-new objects.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic import BaseModel

from app.core.slots import MODELS
from app.services import llm


@dataclass
class Prefab:
    id: str          # the node id whose mesh can be reused
    description: str  # the object's decomposition prompt (its "name"/description)


class PrefabMatchOutput(BaseModel):
    # The id of an existing asset to reuse, or "" to generate a fresh one.
    # A plain string (not str | None) keeps the strict json_schema simple.
    reuse_id: str = ""


# track_key -> ordered catalog of assets built so far (the match candidates).
_registry: dict[str, list[Prefab]] = {}
# track_key -> {node_id: Event}. The Event fires once that node's GLB is on disk
# (set by the node's generate/reuse task, even on failure — a waiting reuse then
# checks the file and errors out if the source produced nothing). Lets a reuse
# wait for a source that is still generating.
_ready: dict[str, dict[str, asyncio.Event]] = {}


def register(track_key: str, node_id: str, description: str) -> None:
    """Add an asset to the match catalog and allocate its ready-Event. Called
    as each object is resolved, BEFORE its mesh task is spawned, so subsequent
    objects in the same pass can match it."""
    cat = _registry.setdefault(track_key, [])
    if not any(p.id == node_id for p in cat):
        cat.append(Prefab(id=node_id, description=description))
    _ready.setdefault(track_key, {}).setdefault(node_id, asyncio.Event())


def mark_ready(track_key: str, node_id: str) -> None:
    """Signal that `node_id`'s GLB is on disk (or that its task ended). No-op
    once the track has been cleared (run finished), so a standalone post-run
    retry doesn't resurrect the registry."""
    ready = _ready.get(track_key)
    if ready is None:
        return
    ready.setdefault(node_id, asyncio.Event()).set()


async def await_ready(track_key: str, node_id: str) -> None:
    """Block until `node_id`'s mesh task has finished. Returns immediately if
    the node is unknown (nothing to wait for)."""
    ev = _ready.get(track_key, {}).get(node_id)
    if ev is not None:
        await ev.wait()


def candidates(track_key: str, *, exclude_id: str) -> list[Prefab]:
    return [p for p in _registry.get(track_key, []) if p.id != exclude_id]


def clear(track_key: str) -> None:
    _registry.pop(track_key, None)
    _ready.pop(track_key, None)


SYSTEM_PREFAB_MATCH = """\
You are part of a 3D scene builder that generates assets from scratch. To keep \
the scene visually consistent AND avoid regenerating duplicates, you decide \
whether a NEW object can REUSE an asset that has already been built (or is being \
built) elsewhere in the SAME scene.

You receive the new object's id and description, plus a catalog of existing \
assets (each with an id and description). Reuse an existing asset ONLY when it \
is essentially the SAME object — a repeat or near-identical instance that would \
look correct if the existing mesh were dropped into the new object's slot (it \
will be rescaled to fit). Do not get caught up by flourishes in the mesh's \
description, identify the exact type of object it is and match if a similar mesh exists. 

Respond with ONE JSON object: set `reuse_id` to the id of the asset to reuse, or \
to an empty string "" to generate a fresh asset. No prose, no markdown, no code \
fences.\
"""


async def match(description: str, cands: list[Prefab], *, node_id: str) -> str | None:
    """Lightweight LLM check: can `description` reuse one of `cands`? Returns the
    chosen prefab id, or None to generate fresh. Always runs on gemini-flash-lite
    (a cheap retrieval, not part of the spatial-reasoning benchmark surface),
    regardless of the run's configured model — mirroring library matching."""
    if not cands:
        return None
    catalog = "\n".join(f"  - id={p.id!r}: {p.description}" for p in cands)
    user = (
        f"New object:\n  id={node_id!r}: {description}\n\n"
        f"Existing assets already in this scene:\n{catalog}\n\n"
        "Return the id of the asset to reuse, or \"\" to generate fresh."
    )
    token = llm._current_model.set(MODELS["gemini-flash-lite"])
    try:
        out = await llm.call_llm(
            system=SYSTEM_PREFAB_MATCH,
            user=user,
            output_schema=PrefabMatchOutput,
            node_id=node_id,
            step="prefab_match",
        )
    finally:
        llm._current_model.reset(token)
    chosen = (out.reuse_id or "").strip()
    # Guard against a hallucinated id that isn't in the catalog.
    if chosen and any(p.id == chosen for p in cands):
        return chosen
    return None
