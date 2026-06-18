"""Log-driven reconstruction of already-committed pipeline results.

This is the core of *robust* resume. The pipeline still re-walks the
divider/generation tree top-down on a resume, but instead of re-issuing
every LLM call and trusting the content-addressed LLM cache (keyed on the
*prompt text*) to hand back identical node ids, each step first asks here
for the result it already wrote to `events.jsonl`.

Lookups are keyed on **structural identity** — a node id, or a
`(zone, scenario)` pair — never on prompt bytes or the model id. So a
prompt edit or a model remap between a run and its resume can no longer
fork a brand-new tree: committed work is replayed verbatim from the log,
and only the genuinely-uncommitted *frontier* falls through to a fresh
LLM call (which is correct — that work never happened).

Every getter returns the shared `schemas` object, so a reconstructed value
is a drop-in for what the corresponding LLM call would have returned. A
`None` return means "not committed" — the caller should do the real work.
An empty list is distinct from `None`: it means "committed, but produced
nothing" (e.g. an encapsulating pass that decided no bounding geometry was
needed).

The committed events this reads:
  * `divider.zone_plan`     — `node`, `plan`, `is_atomic`
  * `bbox`                  — `id`, world-frame `origin`/`dimensions`, ...
  * `divider.zone_decompose`— `node`, `children` (full SubregionSpec dumps)
  * `generation.decompose`  — `zone`, `scenario`, `objects` (full ObjectSpec dumps)
  * `generation.next`       — `zone`, `id`, `object` (full ObjectSpec dump)
  * `generation.next.done`  — `zone`
  * `image`                 — `id`, `prompt` (the committed subject phrase)

`generation.decompose` / `generation.next` only carry full spec dumps on
runs created after that enrichment landed; legacy ids-only events are
reconstructed best-effort from each object's `bbox` event so an already
positioned object still replays, and fall back to `None` (re-run the LLM)
when an object was decomposed but never bbox-resolved.
"""

from __future__ import annotations

from typing import Any

from app.core import schemas
from app.core.types import BoundingBox
from app.utils import logging


def zone_plan(node_id: str) -> Any | None:
    """Committed `ZonePlanOutput` (plan + is_atomic) for a zone, or None."""
    e = logging.find_event("divider.zone_plan", node=node_id)
    if e is None:
        return None
    return schemas.ZonePlanOutput(
        plan=str(e.get("plan", "")),
        is_atomic=bool(e.get("is_atomic", False)),
    )


def bbox(node_id: str) -> BoundingBox | None:
    """Committed world-frame bounding box for a node, or None."""
    e = logging.find_event("bbox", id=node_id)
    if e is None:
        return None
    origin = e.get("origin")
    dims = e.get("dimensions")
    if not isinstance(origin, list) or not isinstance(dims, list):
        return None
    return BoundingBox(
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
        dimensions=(float(dims[0]), float(dims[1]), float(dims[2])),
    )


def orientation(node_id: str) -> int | None:
    """Committed yaw for a node, read from its `bbox` event (object_bbox_batch
    solves it now and emit_bbox logs it). None when absent."""
    e = logging.find_event("bbox", id=node_id)
    if e is None:
        return None
    o = e.get("orientation")
    return int(o) if isinstance(o, (int, float)) else None


def zone_decompose(node_id: str) -> Any | None:
    """Committed `ZoneDecomposeOutput` (child specs) for a non-atomic zone,
    or None if this zone was never decomposed."""
    e = logging.find_event("divider.zone_decompose", node=node_id)
    if e is None:
        return None
    children = e.get("children")
    if not isinstance(children, list):
        return None
    try:
        # Old logs carry per-child `parent`/`parent_relationship_kind`; those
        # extra keys are simply ignored when validated into a SubregionSpec.
        specs = [schemas.SubregionSpec.model_validate(c) for c in children]
    except Exception:
        return None
    return schemas.ZoneDecomposeOutput(subregions=specs)


def object_specs(zone_id: str, scenario: str) -> list[Any] | None:
    """Committed object specs for a `(zone, scenario)` generation pass.

    Returns `[]` for a pass that committed but emitted nothing, so the
    caller can tell "ran, nothing to do" apart from "never ran" (`None`)."""
    e = logging.find_event("generation.decompose", zone=zone_id, scenario=scenario)
    if e is None:
        return None
    objects = e.get("objects")
    if not isinstance(objects, list):
        return None
    if not objects:
        return []
    if all(isinstance(o, dict) for o in objects):
        try:
            return [schemas.ObjectSpec.model_validate(o) for o in objects]
        except Exception:
            pass
    # Legacy ids-only event: rebuild each object from its bbox event.
    specs: list[Any] = []
    for o in objects:
        oid = o if isinstance(o, str) else o.get("id") if isinstance(o, dict) else None
        if not isinstance(oid, str):
            return None
        spec = _spec_from_bbox(oid)
        if spec is None:
            return None
        specs.append(spec)
    return specs


def next_object_specs(zone_id: str) -> list[Any]:
    """Ordered committed next-object specs from a zone's anchor completion
    loop — one per accepted `generation.next`, in emission order."""
    out: list[Any] = []
    for e in logging.current_events():
        if e.get("kind") != "generation.next" or e.get("zone") != zone_id:
            continue
        obj = e.get("object")
        if isinstance(obj, dict):
            try:
                out.append(schemas.ObjectSpec.model_validate(obj))
                continue
            except Exception:
                pass
        oid = e.get("id")
        if isinstance(oid, str):
            spec = _spec_from_bbox(oid)
            if spec is not None:
                out.append(spec)
    return out


def next_done(zone_id: str) -> bool:
    """True if the zone's anchor completion loop committed its terminal
    `generation.next.done` — i.e. the loop already ran to completion."""
    return logging.find_event("generation.next.done", zone=zone_id) is not None


def image_subject(node_id: str) -> str | None:
    """Committed subject phrase (the `image` event's `prompt`) for a node,
    or None if its reference image was never generated."""
    e = logging.find_event("image", id=node_id)
    if e is None:
        return None
    prompt = e.get("prompt")
    return prompt if isinstance(prompt, str) else None


def _spec_from_bbox(node_id: str) -> Any | None:
    """Best-effort `ObjectSpec` rebuilt from a node's `bbox` event, for
    legacy logs that recorded object decisions by id only. `parent_kind`
    and `placement` are filled with inert defaults — a reconstructed spec
    is only ever re-fed to generation for an object whose bbox is already
    committed, so those fields are never consulted again."""
    e = logging.find_event("bbox", id=node_id)
    if e is None:
        return None
    try:
        return schemas.ObjectSpec(
            id=node_id,
            prompt=str(e.get("prompt", "")),
            parent=str(e.get("parent_id") or ""),
            parent_kind="IN",
            placement="",
            proxy_shape=e.get("proxy_shape"),
        )
    except Exception:
        return None
