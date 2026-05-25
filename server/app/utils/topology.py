"""Graph utilities for the v2 prose-placement spec.

The decomposition LLM emits `ChildNodeSpec`s (subzones) or `ObjectSpec`s
(objects), each carrying `referenced_ids: list[str]` whose first entry
is — by prompting convention — that spec's structural parent. We validate
the well-formedness of the spec graph here. We do NOT check that the
chosen parent is the "semantically right" one; the prompt is responsible
for that.
"""

from __future__ import annotations

from app.core.prompts import ChildNodeSpec


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
      2. `referenced_ids` is non-empty for every spec (also enforced by
         the Pydantic `min_length=1`, but re-checked here for clarity).
      3. Every id in `referenced_ids` resolves to `parent_id`, another
         spec in `specs`, or an `existing_ids` entry. No dangling refs.
      4. `referenced_ids[0] != self.id`. No self-parent.
      5. Walking `referenced_ids[0]` as parent edges within `specs`, the
         chain is acyclic and terminates at `parent_id` or any
         `existing_ids` entry within `depth_cap` hops. We do not chase
         chains through `existing_ids` — those nodes were validated by
         their own decompose call.
    """
    spec_ids = [s.id for s in specs]
    if len(spec_ids) != len(set(spec_ids)):
        raise ValueError(f"duplicate ids among specs: {spec_ids}")
    collisions = set(spec_ids) & existing_ids
    if collisions:
        raise ValueError(
            f"spec ids collide with existing nodes: {sorted(collisions)}"
        )

    known = {parent_id} | existing_ids | set(spec_ids)
    by_id = {s.id: s for s in specs}
    for s in specs:
        if len(s.referenced_ids) < 1:
            raise ValueError(f"spec {s.id!r} has empty referenced_ids")
        if s.referenced_ids[0] == s.id:
            raise ValueError(
                f"spec {s.id!r} lists itself as primary parent"
            )
        for rid in s.referenced_ids:
            if rid not in known:
                raise ValueError(
                    f"spec {s.id!r} references unknown id {rid!r}"
                )

    # Walk primary-parent edges (referenced_ids[0]) within `specs`. Each
    # chain must terminate at parent_id or an existing_ids node within
    # depth_cap hops; cycles inside `specs` are rejected.
    for s in specs:
        seen: set[str] = {s.id}
        cur: str = s.referenced_ids[0]
        hops = 1
        while cur in by_id:
            if cur in seen:
                raise ValueError(
                    f"cyclic primary-parent chain through {s.id!r}"
                )
            if hops > depth_cap:
                raise ValueError(
                    f"primary-parent chain from {s.id!r} exceeds depth "
                    f"cap of {depth_cap}"
                )
            seen.add(cur)
            cur = by_id[cur].referenced_ids[0]
            hops += 1
        # cur is now either parent_id or in existing_ids; both terminate.
        if cur != parent_id and cur not in existing_ids:
            raise ValueError(
                f"primary-parent chain from {s.id!r} terminates at "
                f"unknown id {cur!r}"
            )
