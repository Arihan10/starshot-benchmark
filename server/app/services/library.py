"""Asset library matching — LLM-powered selection from a pre-built catalog.

When USE_ASSET_LIBRARY is set, the generation pipeline skips image and 3D
generation entirely, instead using an LLM call to match each object's
prompt to the closest item in the library catalog, then copies + rescales
the pre-built .glb into the run's output directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from app.core.slots import MODELS
from app.services import llm

_LIBRARY_DIR = Path(__file__).resolve().parent.parent / "assets_library"
_CATALOG_PATH = _LIBRARY_DIR / "library.json"
ASSETS_DIR = _LIBRARY_DIR / "assets"


class LibraryItem(BaseModel):
    id: str
    description: str
    category: str = ""


class LibraryMatchOutput(BaseModel):
    library_id: str


_catalog: list[LibraryItem] | None = None


def _load_catalog() -> list[LibraryItem]:
    global _catalog
    if _catalog is None:
        with open(_CATALOG_PATH) as f:
            raw = json.load(f)
        _catalog = [LibraryItem(**item) for item in raw]
    return _catalog


SYSTEM_LIBRARY_MATCH = """\
You are competing in SpatialBench, a competitive benchmark where LLMs create \
detailed 3D environments from text prompts. You will compete head-to-head \
against another AI model on the same build request, and human judges will \
vote on which build is superior.

Your job is to match a 3D object description to the closest item in a \
pre-built asset library. You will receive the object's prompt, id, and the \
full library catalog. You must always return a match — pick the closest \
available option even if nothing is perfect. Even if the object is not \
semantically similar to any of the library assets, use a 3D model that will\
serve the same purpose if we pick and scale it. 

Respond with ONE JSON object containing `library_id` — the id of the \
best-matching library asset. No prose, no markdown, no code fences.\
"""


async def match(prompt: str) -> LibraryMatchOutput:
    catalog = _load_catalog()
    by_cat: dict[str, list[LibraryItem]] = {}
    for item in catalog:
        by_cat.setdefault(item.category or "UNCATEGORIZED", []).append(item)
    sections = []
    for cat in sorted(by_cat):
        lines = "\n".join(f"    - id={it.id!r}: {it.description}" for it in by_cat[cat])
        sections.append(f"  [{cat}]\n{lines}")
    items_block = "\n\n".join(sections)
    user = (
        f"Object to match: {prompt!r}\n\n"
        f"Available library assets (grouped by category):\n{items_block}\n\n"
        "Pick the best-matching asset id."
    )
    # Library matching is always run on gemini-flash regardless of the run's
    # configured model — the match step is a cheap retrieval, not part of the
    # spatial-reasoning benchmark surface.
    token = llm._current_model.set(MODELS["gemini-flash"])
    try:
        return await llm.call_llm(
            system=SYSTEM_LIBRARY_MATCH,
            user=user,
            output_schema=LibraryMatchOutput,
        )
    finally:
        llm._current_model.reset(token)


def asset_path(library_id: str) -> Path:
    return ASSETS_DIR / f"{library_id}.glb"
