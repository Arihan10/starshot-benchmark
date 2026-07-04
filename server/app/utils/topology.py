"""Graph utilities for the v2 prose-placement spec.

The decomposition LLM emits `ChildNodeSpec`s (subzones) or `ObjectSpec`s
(objects), each carrying an explicit `parent: str` (its structural
anchor) and an optional `referenced_ids: list[Relationship]` listing
secondary spatial relationships. We validate the well-formedness of
the spec graph here. We do NOT check that the chosen parent is the
"semantically right" one; the prompt is responsible for that.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.core.schemas import ChildNodeSpec, SubregionSpec


def validate_parents(
    specs: list[ChildNodeSpec],
    *,
    parent_id: str,
    existing_ids: set[str],
    depth_cap: int = 32,
) -> None:
    """Raise ValueError if any spec's `parent` is unresolvable.

    The parent edge is load-bearing: every scene-context renderer walks
    strictly top-down from the root via `parent_id`, so a node whose
    parent does not resolve never appears in any downstream prompt — yet
    it still gets a bbox (the resolver falls back to the zone) and a
    mesh, landing in the final scene as a "ghost" later steps can't see
    and may overlap. That silent divergence is unacceptable, so a bad
    parent is a HARD error rather than something the optimistic flow
    swallows. Secondary `referenced_ids` are advisory and validated
    separately (a dangling one doesn't orphan anything).

    Checks, for each spec:
      * `spec.parent != self.id` — no self-parent.
      * `spec.parent` resolves to `parent_id`, another spec in `specs`,
        or an `existing_ids` entry.
      * Walking `spec.parent` as parent edges within `specs`, the chain
        is acyclic and terminates at `parent_id` or an `existing_ids`
        entry within `depth_cap` hops. Chains are not chased through
        `existing_ids` — those nodes were validated by their own call.
    """
    known = {parent_id} | existing_ids | {s.id for s in specs}
    by_id = {s.id: s for s in specs}
    for s in specs:
        if s.parent == s.id:
            raise ValueError(f"spec {s.id!r} lists itself as parent")
        if s.parent not in known:
            raise ValueError(f"spec {s.id!r} has unknown parent {s.parent!r}")

    for s in specs:
        seen: set[str] = {s.id}
        cur: str = s.parent
        hops = 1
        while cur in by_id:
            if cur in seen:
                raise ValueError(f"cyclic parent chain through {s.id!r}")
            if hops > depth_cap:
                raise ValueError(
                    f"parent chain from {s.id!r} exceeds depth cap of {depth_cap}"
                )
            seen.add(cur)
            cur = by_id[cur].parent
            hops += 1
        # cur is now either parent_id or in existing_ids; both terminate.
        if cur != parent_id and cur not in existing_ids:
            raise ValueError(
                f"parent chain from {s.id!r} terminates at unknown id {cur!r}"
            )


def validate_referenced_ids(
    specs: list[ChildNodeSpec],
    *,
    parent_id: str,
    existing_ids: set[str],
    depth_cap: int = 32,
) -> None:
    """Raise ValueError if the spec graph is malformed.

    Checks:
      1. ids unique within `specs` and disjoint from `existing_ids`.
      2. Every `spec.parent` is resolvable and acyclic (delegated to
         `validate_parents`).
      3. Every `referenced_ids[i].target` resolves to `parent_id`,
         another spec, or an `existing_ids` entry, and is not the spec
         itself.

    This is the full advisory check fed to the decomposition retry
    loops. The parent portion (check 2) is the only one promoted to a
    hard gate via `validate_parents`; the rest stay advisory.
    """
    spec_ids = [s.id for s in specs]
    if len(spec_ids) != len(set(spec_ids)):
        raise ValueError(f"duplicate ids among specs: {spec_ids}")
    collisions = set(spec_ids) & existing_ids
    if collisions:
        raise ValueError(
            f"spec ids collide with existing nodes: {sorted(collisions)}"
        )

    validate_parents(
        specs, parent_id=parent_id, existing_ids=existing_ids, depth_cap=depth_cap
    )

    known = {parent_id} | existing_ids | set(spec_ids)
    for s in specs:
        for rel in s.referenced_ids:
            if rel.target == s.id:
                raise ValueError(
                    f"spec {s.id!r} has a relationship targeting itself"
                )
            if rel.target not in known:
                raise ValueError(
                    f"spec {s.id!r} has a relationship with unknown "
                    f"target {rel.target!r}"
                )


def validate_subregions(
    specs: list[SubregionSpec],
    *,
    parent_id: str,
    existing_ids: set[str],
) -> None:
    """Raise ValueError if a batch of subregion specs is malformed.

    Subregions carry no structural `parent` — they are always contained in the
    zone being decomposed (`parent_id`), so there is no orphan / parent-chain
    failure mode to guard (the whole reason `validate_parents` is a hard gate).
    What remains is purely advisory, mirroring the non-parent checks of
    `validate_referenced_ids`:
      1. ids unique within `specs` and disjoint from `existing_ids`.
      2. every `relationships[i].target` resolves to the zone, another spec,
         or an existing node, and is not the spec itself.
    """
    spec_ids = [s.id for s in specs]
    if len(spec_ids) != len(set(spec_ids)):
        raise ValueError(f"duplicate ids among subregions: {spec_ids}")
    collisions = set(spec_ids) & existing_ids
    if collisions:
        raise ValueError(
            f"subregion ids collide with existing nodes: {sorted(collisions)}"
        )

    known = {parent_id} | existing_ids | set(spec_ids)
    for s in specs:
        for rel in s.referenced_ids:
            if rel.target == s.id:
                raise ValueError(
                    f"subregion {s.id!r} has a relationship targeting itself"
                )
            if rel.target not in known:
                raise ValueError(
                    f"subregion {s.id!r} has a relationship with unknown "
                    f"target {rel.target!r}"
                )


_TRAILING_INDEX = re.compile(r"^(?P<stem>.+?)_(?P<n>\d+)$")


def _next_free_id(desired: str, taken: set[str]) -> str:
    """`desired` if free, else the next free `{stem}_{k}`, reading a trailing
    `_<n>` as the index (bare name = 0): `chair`->`chair_1`, `tree_2`->`tree_3`."""
    if desired not in taken:
        return desired
    m = _TRAILING_INDEX.match(desired)
    stem, n = (m.group("stem"), int(m.group("n"))) if m else (desired, 0)
    k = n + 1
    while f"{stem}_{k}" in taken:
        k += 1
    return f"{stem}_{k}"


def uniquify_ids(
    specs: Sequence[ChildNodeSpec | SubregionSpec],
    *,
    existing_ids: set[str],
) -> list[tuple[str, str]]:
    """Rename colliding spec ids in place — unique against `existing_ids` and
    within the batch — so a reused id keeps its node instead of being dropped.
    In-batch `parent`/`referenced_ids` targets pointing at a renamed
    existing-node collision are rewritten to follow it. Returns `(old, new)`s."""
    taken = set(existing_ids)
    renames: list[tuple[str, str]] = []
    rebind: dict[str, str] = {}
    for s in specs:
        old = s.id
        new = _next_free_id(old, taken)
        taken.add(new)
        if new != old:
            s.id = new
            renames.append((old, new))
            # Rebind refs only when an existing node owns `old` (a batch dup keeps it).
            if old in existing_ids:
                rebind[old] = new
    if rebind:
        for s in specs:
            if isinstance(s, ChildNodeSpec) and s.parent in rebind:
                s.parent = rebind[s.parent]
            if any(r.target in rebind for r in s.referenced_ids):
                s.referenced_ids = [
                    r.model_copy(update={"target": rebind[r.target]})
                    if r.target in rebind
                    else r
                    for r in s.referenced_ids
                ]
    return renames
