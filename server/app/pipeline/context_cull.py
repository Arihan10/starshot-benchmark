"""TEMPORARY context-size patch — cull the `next_object` "detail" tier from every
region EXCEPT the focus zone, to shrink `{SCENE_CONTEXT}` for the very largest
scenes (where it can reach ~1M tokens and overflow the model window / degrade
reasoning).

Full context is always best; this is a deliberate, opt-in compromise for scenes
that would otherwise overflow. It keeps the structural skeleton intact — zones,
anchors, shells — and keeps the focus zone's OWN next_object objects (so the
placement / completion steps still see what's already in the zone they're
working on); it only drops the small "fill" objects (keyboards, coasters, …)
that other zones accumulated, which are noise when reasoning about a different
zone.

Two levers, both under the single `STARSHOT_CULL_NEXT_OBJECT` toggle:
  1. `cull_next_object` — drop the next_object detail tier from non-focus zones.
  2. `cull_early_regions` — additionally drop the earliest ~20% of regions (and
     their contents) when (1) alone isn't enough. Fraction is env-tunable via
     `STARSHOT_CULL_EARLY_REGION_FRAC` (default 0).

`for_context` (the single entry point) emits a `context.cull` audit event each
time it runs — so the log/console shows the reduction when the cull is ON, and
shows nothing at all when it's OFF (peace of mind that it isn't touching context).

Kept deliberately self-contained so it can be:
  * DISABLED — leave `STARSHOT_CULL_NEXT_OBJECT` unset (the default is off), or
  * EXTRACTED — delete this module, `committed.next_object_ids`, and the
    `_context_nodes` wrapper (+ its three uses) in `generation.py`.
"""

from __future__ import annotations

import os
from collections.abc import Collection

from app.core.types import Node
from app.utils import logging

_TOGGLE_ENV = "STARSHOT_CULL_NEXT_OBJECT"
_EARLY_REGION_FRAC_ENV = "STARSHOT_CULL_EARLY_REGION_FRAC"


def enabled() -> bool:
    """Whether the context cull is active — a backend env toggle, OFF by default
    so normal-sized runs pass the complete context unchanged."""
    return os.environ.get(_TOGGLE_ENV, "false").strip().lower() == "true"


def _early_region_fraction() -> float:
    """Fraction of regions (earliest-first) to drop in `cull_early_regions`.
    Env-tunable so it can be pushed past the 0 default if a scene still
    overflows, without a code change."""
    try:
        return float(os.environ.get(_EARLY_REGION_FRAC_ENV, "0") or "0")
    except ValueError:
        return 0


def cull_next_object(
    nodes: list[Node], *, focus_zone_id: str, next_object_ids: Collection[str],
) -> list[Node]:
    """`nodes` with every `next_object` object that lives OUTSIDE `focus_zone_id`
    dropped. Zones (structural nodes) and all non-next_object objects are kept,
    and the focus zone keeps its own next_object objects. A no-op when there are
    no next_object ids to cull. Pure: no I/O, no mutation of the input list."""
    if not next_object_ids:
        return nodes
    ids = next_object_ids if isinstance(next_object_ids, (set, frozenset)) else set(next_object_ids)
    return [
        n for n in nodes
        if not (n.id in ids and n.parent_region != focus_zone_id)
    ]


def cull_early_regions(
    nodes: list[Node], *, focus_zone_id: str, fraction: float | None = None,
) -> list[Node]:
    """Drop the EARLIEST `fraction` of regions (zones) — and everything inside
    them (nested regions + their objects) — from the context. A blunter, heavier
    trim than `cull_next_object`, for scenes that still overflow after it.

    Regions are ordered by their position in `nodes` (≈ creation order), so
    'earliest' = first placed. NEVER drops the root, the focus zone, or the
    focus's ancestor chain — those define the focus's frame and its path in the
    tree, so removing them would break the render. A no-op when `fraction <= 0`
    or nothing is left to cull after protecting those."""
    frac = _early_region_fraction() if fraction is None else fraction
    if frac <= 0:
        return nodes
    by_id = {n.id: n for n in nodes}
    # Protected: root(s) + the focus zone + every ancestor up to the root.
    protected = {n.id for n in nodes if n.parent_id is None}
    cur: str | None = focus_zone_id
    while cur is not None and cur in by_id:
        protected.add(cur)
        cur = by_id[cur].parent_id
    region_ids = [n.id for n in nodes if n.is_zone]
    k = int(len(region_ids) * frac)
    culled = {rid for rid in region_ids[:k] if rid not in protected}
    if not culled:
        return nodes
    # A subregion of a culled region is culled too (its parent is gone).
    region_children: dict[str, list[str]] = {}
    for n in nodes:
        if n.is_zone and n.parent_id is not None:
            region_children.setdefault(n.parent_id, []).append(n.id)
    stack = list(culled)
    while stack:
        for child in region_children.get(stack.pop(), []):
            if child not in culled and child not in protected:
                culled.add(child)
                stack.append(child)
    # Drop the culled regions themselves and any object owned by one.
    return [
        n for n in nodes
        if n.id not in culled and not (not n.is_zone and n.parent_region in culled)
    ]


def for_context(nodes: list[Node], focus_zone_id: str) -> list[Node]:
    """The node list to render `{SCENE_CONTEXT}` from for a step focused on
    `focus_zone_id`. THE single entry point every context-building step routes
    through — divider (zone_plan / zone_decompose / child_bbox_batch) AND
    generation (decompose / next_object / object_bbox_batch).

    Toggle OFF → `nodes` unchanged (full context, always preferred). ON → drop
    the next_object detail tier from non-focus zones, then drop the earliest
    fraction of regions. [temporary context-cull patch]"""
    if not enabled():
        return nodes
    # Lazy import so this module stays import-order-independent; `committed`
    # reads the bound slot log's committed `generation.next` events.
    from app.pipeline import committed

    before = len(nodes)
    after_next = cull_next_object(
        nodes, focus_zone_id=focus_zone_id, next_object_ids=committed.next_object_ids(),
    )
    result = cull_early_regions(after_next, focus_zone_id=focus_zone_id)
    # AUDIT — one `context.cull` event per context build, emitted ONLY when the
    # cull actually runs. Its presence (server console + events.jsonl) confirms
    # the cull is ON and shows how much each lever dropped; its total ABSENCE
    # confirms it's OFF. Counts only — no prompt bytes — so it's cheap and never
    # itself bloats the log.
    if logging.current_slot_id() is not None:
        logging.log(
            "context.cull",
            focus=focus_zone_id,
            raw_nodes=before,
            kept_nodes=len(result),
            dropped_next_object=before - len(after_next),
            dropped_regions=len(after_next) - len(result),
            region_fraction=_early_region_fraction(),
        )
    return result
