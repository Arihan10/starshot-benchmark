"""Asset library matching — LLM-powered selection from a pre-built catalog.

When USE_ASSET_LIBRARY is set, the generation pipeline skips image and 3D
generation entirely, instead using an LLM call to match each object's
prompt to the closest item in the library catalog, then bakes a placement
transform into the pre-built .glb (see app.utils.glb_place) under the run's
output directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel

from app.core.slots import MODELS
from app.services import llm

_LIBRARY_DIR = Path(__file__).resolve().parent.parent / "assets_library"
_CATALOG_PATH = _LIBRARY_DIR / "library.json"
# Default to the optimized library (decimated geometry, Meshopt + KTX2). Set
# LIBRARY_ASSETS_SUBDIR=assets to fall back to the raw Trellis assets.
ASSETS_DIR = _LIBRARY_DIR / os.environ.get("LIBRARY_ASSETS_SUBDIR", "assets-optimized")
_MANIFEST_PATH = ASSETS_DIR / "optimize_manifest.json"


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
    # spatial-reasoning benchmark surface. log_call=False keeps this call out
    # of the event log / observability view entirely: its system prompt, the
    # full catalog it sends as input, the chosen id, and the model's reasoning
    # are all retrieval noise. The lightweight `library.match` event emitted by
    # the caller remains the record of the routing decision.
    token = llm._current_model.set(MODELS["gemini-flash"])
    try:
        return await llm.call_llm(
            system=SYSTEM_LIBRARY_MATCH,
            user=user,
            output_schema=LibraryMatchOutput,
            log_call=False,
        )
    finally:
        llm._current_model.reset(token)


def asset_path(library_id: str) -> Path:
    return ASSETS_DIR / f"{library_id}.glb"


_bounds_by_id: dict[str, dict[str, dict[str, list[float]]]] | None = None


def _load_bounds() -> dict[str, dict[str, dict[str, list[float]]]]:
    """Per-asset, per-orientation world-space AABBs from optimize_manifest.json.
    The placement bake needs the rotated extents to fill a target bbox, but the
    optimized GLBs are Meshopt/KTX2-compressed and can't be measured server-side,
    so the bounds are precomputed (see tools/optimize-assets/augment-bounds.mjs)."""
    global _bounds_by_id
    if _bounds_by_id is None:
        _bounds_by_id = {}
        if _MANIFEST_PATH.exists():
            data = json.loads(_MANIFEST_PATH.read_text())
            for entry in data.get("assets", []):
                bbo = entry.get("bounds_by_orientation")
                if bbo:
                    _bounds_by_id[entry["id"]] = bbo
    return _bounds_by_id


def asset_rotated_bounds(
    library_id: str, orientation: int
) -> tuple[list[float], list[float]] | None:
    """(min, max) world-space AABB of `library_id` after the given yaw, or None
    when the asset is absent from the manifest or lacks augmented bounds."""
    bbo = _load_bounds().get(library_id)
    if not bbo:
        return None
    entry = bbo.get(str(int(orientation)))
    if not entry:
        return None
    return entry["min"], entry["max"]
