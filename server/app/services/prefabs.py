"""Per-scene prefab matching for the from-scratch generate gate.

Generating every object independently means a scene with twenty near-identical
"wall panel"s pays for twenty Nano-Banana + Trellis runs that also don't visually
match each other. Instead, objects are de-duplicated by a seed-and-sweep pass: the
first undecided object is taken as a canonical "seed" and a lightweight LLM call
names every other object that is the SAME object, so each of those REUSES the
seed's mesh (rescaled into its bbox; see generation._reuse) instead of being
generated. The next undecided object seeds the following group, and so on — so
matching costs one call per distinct object TYPE, not one per object.

The match always runs on gemini-flash-lite regardless of the run's configured
model — it's a cheap retrieval, not part of the spatial-reasoning benchmark
surface, mirroring library matching. The seed/reuse bookkeeping lives in the
generation module that drives this.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel

from app.core.slots import MODELS
from app.core.types import BoundingBox
from app.services import llm
from app.utils import logging as rlog

SYSTEM_PREFAB_MATCH = """\
You are part of a 3D scene builder that generates assets from scratch. To keep \
the scene visually consistent, you find every \
object that is a REPEAT of a given "seed" object, so each one can REUSE the \
seed's mesh instead of being generated again.

You receive the seed object's id, description, and bounding-box size \
(width×height×depth, in metres), plus a list of candidate objects (each with the \
same fields). Return the ids of every candidate that is essentially the SAME \
object as the seed — a repeat or near-identical instance that would look correct \
if the seed's mesh were dropped into the candidate's slot (it will be rescaled to \
fit). Think about physical descriptive features that would make two objects visually distinct to the human eye; only match objects that are near identical/will be used in the exact context. Your goal is not to minimize the number of assets, it is to make the scene MORE COHERENT.  



A reused mesh is rescaled per-axis to exactly fill the candidate's bounding box, \
which does not preserve proportions.

Respond with ONE JSON object: set `matches` to the list of matching candidate ids \
(an empty list if none match). No prose, no markdown, no code fences.\
"""


class DuplicateMatchOutput(BaseModel):
    # ids of the candidates that are the same object as the seed (they reuse it).
    matches: list[str] = []


def _fmt_size(bbox: BoundingBox) -> str:
    w, h, d = bbox.size
    return f"{w:.2f}×{h:.2f}×{d:.2f}m"


# Prefab matching is a cheap, best-effort de-dup call, so a hung/failing one must
# never stall the whole generate. Cap each attempt at a hard wall-clock minute
# (`wait_for` cancels the in-flight request outright, unlike the shared 180s
# client timeout + the SDK's own long internal retry) and retry a couple of
# times. Deliberately light: no backoff, a tiny attempt budget, and a "no match"
# fallback so every candidate simply generates on its own if matching gives up.
PREFAB_MATCH_TIMEOUT_S = 60.0
PREFAB_MATCH_ATTEMPTS = 3


async def _match_call(*, user: str, seed_id: str) -> DuplicateMatchOutput | None:
    """One prefab-match LLM call, capped at PREFAB_MATCH_TIMEOUT_S per attempt and
    retried up to PREFAB_MATCH_ATTEMPTS times. Returns the parsed output, or None
    when every attempt timed out / errored. Cancellation (build teardown) is a
    BaseException, so it propagates past the `except Exception` untouched."""
    for attempt in range(1, PREFAB_MATCH_ATTEMPTS + 1):
        try:
            return await asyncio.wait_for(
                llm.call_llm(
                    system=SYSTEM_PREFAB_MATCH,
                    user=user,
                    output_schema=DuplicateMatchOutput,
                    node_id=seed_id,
                    step="prefab_match",
                ),
                timeout=PREFAB_MATCH_TIMEOUT_S,
            )
        except Exception as e:  # noqa: BLE001 — best-effort: any failure falls back to no-match
            giving_up = attempt >= PREFAB_MATCH_ATTEMPTS
            rlog.console_note(
                f"[prefabs] prefab_match {seed_id!r} attempt {attempt}/{PREFAB_MATCH_ATTEMPTS} "
                f"failed ({type(e).__name__}: {str(e)[:140]}) — "
                f"{'giving up (no matches)' if giving_up else 'retrying'}"
            )
    return None


async def match_duplicates(
    *,
    seed_id: str,
    seed_description: str,
    seed_bbox: BoundingBox,
    candidates: list[tuple[str, str, BoundingBox]],
) -> list[str]:
    """Name every candidate that is essentially the SAME object as the seed (and so
    can reuse the seed's mesh). `candidates` is a list of `(id, description, bbox)`;
    returns the matching ids, validated against the candidate set so a hallucinated
    or duplicated id is dropped. Always runs on gemini-flash-lite."""
    if not candidates:
        return []
    listing = "\n".join(
        f"  - id={cid!r} [{_fmt_size(bbox)}]: {desc}" for cid, desc, bbox in candidates
    )
    user = (
        f"Seed object:\n  id={seed_id!r} [{_fmt_size(seed_bbox)}]: {seed_description}\n\n"
        "Candidate objects "
        "(id, [bounding-box size as width×height×depth in metres], description):\n"
        f"{listing}\n\n"
        "Return the ids of every candidate that is the same object as the seed."
    )
    token = llm._current_model.set(MODELS["gemini-flash-lite"])
    try:
        out = await _match_call(user=user, seed_id=seed_id)
    finally:
        llm._current_model.reset(token)
    if out is None:
        # Timed out / errored past the retry budget — match nothing so every
        # candidate just generates on its own rather than stalling the build.
        return []
    valid = {cid for cid, _, _ in candidates}
    seen: set[str] = set()
    matches: list[str] = []
    for mid in out.matches:
        if mid in valid and mid not in seen:
            seen.add(mid)
            matches.append(mid)
    return matches


def resolve_group(events: list[dict[str, Any]], node_id: str) -> tuple[str, list[str]]:
    """Resolve `node_id`'s prefab group from a generated log's `prefab.match`
    events. Returns `(canonical_id, reuse_ids)`:

      * canonical_id — the asset whose raw Trellis mesh the group's geometry is
        derived from: `node_id` itself when it's a canonical, else its source.
      * reuse_ids    — every OTHER node that reuses that canonical.

    Decisions are folded in log order so the LATEST `prefab.match` per id wins —
    a reuse promoted to canonical by a standalone regen reads back as canonical.
    The prefab graph is a flat star (a reuse_id always points at a canonical,
    never another reuse), so no transitive resolution is needed.
    """
    reuse_of: dict[str, str] = {}
    for e in events:
        if e.get("kind") != "prefab.match":
            continue
        nid = e.get("id")
        if isinstance(nid, str):
            reuse_of[nid] = str(e.get("reuse_id") or "")
    canonical = reuse_of.get(node_id) or node_id
    reuses = [oid for oid, rid in reuse_of.items() if rid == canonical and oid != canonical]
    return canonical, reuses
