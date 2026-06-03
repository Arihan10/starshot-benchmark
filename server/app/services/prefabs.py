"""Per-scene prefab matching for the from-scratch generate gate.

Generating every object independently means a scene with twenty near-identical
"wall panel"s pays for twenty Nano-Banana + Trellis runs that also don't visually
match each other. Before generating a new object, a lightweight LLM call checks
whether it can REUSE an asset already built (or still in flight) in the SAME
scene — for visual consistency AND to skip duplicate generation. A reuse rescales
the matched asset's mesh into the new object's bbox instead of generating from
scratch (see generation._reuse).

The match always runs on gemini-flash-lite regardless of the run's configured
model — it's a cheap retrieval, not part of the spatial-reasoning benchmark
surface, mirroring library matching. The per-scene catalog + reuse bookkeeping
live in the generation module that drives this.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.core.slots import MODELS
from app.services import llm

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

Respond with ONE JSON object: set reuse_id to the id of the asset to reuse, or \
to an empty string "" to generate a fresh asset. No prose, no markdown, no code \
fences.\
"""


class PrefabMatchOutput(BaseModel):
    # The id of an existing asset to reuse, or "" to generate a fresh one.
    reuse_id: str = ""


async def match(*, new_id: str, new_description: str, catalog: list[tuple[str, str]]) -> str:
    """Lightweight check: can the new object reuse one of `catalog` (each an
    `(id, description)`)? Returns the chosen id, or "" to generate fresh. Always
    runs on gemini-flash-lite. The caller is responsible for validating the
    returned id against the catalog (guards a hallucinated id)."""
    if not catalog:
        return ""
    listing = "\n".join(f"  - id={cid!r}: {desc}" for cid, desc in catalog)
    user = (
        f"New object:\n  id={new_id!r}: {new_description}\n\n"
        f"Existing assets already in this scene:\n{listing}\n\n"
        'Return the id of the asset to reuse, or "" to generate fresh.'
    )
    token = llm._current_model.set(MODELS["gemini-flash-lite"])
    try:
        out = await llm.call_llm(
            system=SYSTEM_PREFAB_MATCH,
            user=user,
            output_schema=PrefabMatchOutput,
            node_id=new_id,
            step="prefab_match",
        )
    finally:
        llm._current_model.reset(token)
    return (out.reuse_id or "").strip()
