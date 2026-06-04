"""Lightweight structural-vs-prop classifier for asset generation.

Trellis (app.services.threed) is free-form image-to-3D and tends to produce
organic, blobby geometry — fine for props, poor for the rectilinear building
shell (walls, floors, ceilings, roofs, slabs). Before a fresh mesh is built, a
cheap LLM call decides whether the object is STRUCTURAL; if so, generation
routes to the bbox-conditioned Hunyuan3D-Omni endpoint instead of Trellis (see
generation._generate_one).

Like prefab/library matching, this always runs on gemini-flash-lite regardless
of the run's configured model — it's a cheap classification, not part of the
spatial-reasoning benchmark surface.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.core.slots import MODELS
from app.services import llm

SYSTEM_STRUCTURAL_CLASSIFY = """\
You are part of a 3D scene builder. Each object is generated as a mesh, and you \
decide which generator it should use based on whether it is STRUCTURAL.

A STRUCTURAL object is a large, rectilinear, load-bearing or enclosing part of \
the built environment's shell — walls, floors, ceilings, roofs, foundations and \
slabs, retaining walls, large platforms and decks, staircases, columns and \
piers, and similar architectural shell geometry. These are defined primarily by \
their bounding box and benefit from box-conditioned generation.

A NON-structural object is a prop, furnishing, fixture, vegetation, vehicle, \
character, or decoration that sits within or on the shell — chairs, lamps, \
plants, signs, rocks, appliances, and the like.

Respond with ONE JSON object: set is_structural to true or false. No prose, no \
markdown, no code fences.\
"""


class StructuralClassifyOutput(BaseModel):
    is_structural: bool = False


async def classify(*, node_id: str, description: str) -> bool:
    """Decide whether `description` names a structural shell element (route to
    the bbox-conditioned endpoint) or a prop (route to Trellis). Always runs on
    gemini-flash-lite; the decision is cached + replayed on resume by call_llm."""
    user = (
        f"Object id: {node_id!r}\n"
        f"Object description: {description}\n\n"
        "Is this a structural shell element? Respond with is_structural true or false."
    )
    token = llm._current_model.set(MODELS["gemini-flash-lite"])
    try:
        out = await llm.call_llm(
            system=SYSTEM_STRUCTURAL_CLASSIFY,
            user=user,
            output_schema=StructuralClassifyOutput,
            node_id=node_id,
            step="structural_classify",
        )
    finally:
        llm._current_model.reset(token)
    return out.is_structural
