"""Attention statistics — pure functions over a single (post-softmax) attention
row, computed on the fly and immediately reduced to scalars / sparse top-k so no
dense attention matrix is ever materialized.

For query step i over scene key columns S (|S| = L), with P_ij the attention
weight from query i to key j:

  * Attention scale     c_i   = Σ_{j∈S} P_ij                (mass on the scene)
  * Attention map       P_i,S / c_i                          (renormalized dist.)
  * Entropy ratio       R_i   = (H_full/log N_i) / (H_scene/log L)

`H_full` is over the whole causal row (length N_i = i+1); `H_scene` over the
renormalized scene distribution (length L). The log-normalizations keep both
metrics stable as context grows. Everything is per (layer, head).

Multi-token entities aggregate their tokens with `max` (recommended — captures
the strongest interaction / primary connection), `sum`, or `mean`; the same
reduction applies hierarchically to each attribute (placement / dimensions / …).
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

AggMethod = Literal["max", "sum", "mean"]


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy (nats) of a non-negative vector, treating it as a
    distribution over its support (zeros ignored)."""
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def scene_scale(row: np.ndarray, scene_cols: np.ndarray) -> float:
    """c_i — the fraction of attention mass query i puts on the scene columns."""
    if scene_cols.size == 0:
        return 0.0
    return float(row[scene_cols].sum())


def scene_distribution(row: np.ndarray, scene_cols: np.ndarray, c_i: float) -> np.ndarray:
    """P_i,S renormalized by c_i — a proper distribution over scene tokens.
    All-zero when the query attends nothing to the scene."""
    if scene_cols.size == 0 or c_i <= 0:
        return np.zeros(scene_cols.size, dtype=np.float64)
    return row[scene_cols] / c_i


def entropy_ratio(row: np.ndarray, scene_probs: np.ndarray, c_i: float) -> float:
    """R_i = (H_full/log N_i) / (H_scene/log L). Returns 0.0 when undefined
    (no scene mass, or degenerate lengths)."""
    n_i = int(row.size)
    ell = int(scene_probs.size)
    if n_i <= 1 or ell <= 1 or c_i <= 0:
        return 0.0
    h_full = _entropy(row) / math.log(n_i)
    h_scene = _entropy(scene_probs) / math.log(ell)
    if h_scene <= 0:
        return 0.0
    return float(h_full / h_scene)


def _reduce(values: list[float], method: AggMethod) -> float:
    if not values:
        return 0.0
    if method == "sum":
        return float(sum(values))
    if method == "mean":
        return float(sum(values) / len(values))
    return float(max(values))  # default / recommended


def aggregate_entities(
    scene_cols: np.ndarray,
    scene_probs: np.ndarray,
    entities: dict[str, dict[str, Any]],
    *,
    method: AggMethod = "max",
) -> list[dict[str, Any]]:
    """Reduce the renormalized scene distribution onto semantic entities and,
    hierarchically, onto each of their attribute token spans.

    Returns [{id, kind, score, components:{component: score}}], one per entity
    that received any attention, unsorted."""
    col_prob = {int(c): float(p) for c, p in zip(scene_cols.tolist(), scene_probs.tolist(), strict=False)}
    out: list[dict[str, Any]] = []
    for eid, rec in entities.items():
        tok_probs = [col_prob.get(t, 0.0) for t in rec["tokens"]]
        score = _reduce(tok_probs, method)
        if score <= 0:
            continue
        comp_scores: dict[str, float] = {}
        for comp, ctoks in rec.get("components", {}).items():
            cs = _reduce([col_prob.get(t, 0.0) for t in ctoks], method)
            if cs > 0:
                comp_scores[comp] = cs
        out.append({"id": eid, "kind": rec["kind"], "score": score, "components": comp_scores})
    return out


def top_entities(entity_scores: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    return sorted(entity_scores, key=lambda e: e["score"], reverse=True)[:k]


def top_attributes(entity_scores: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Flatten (entity, component) -> score across all entities and take the
    strongest k — 'which attribute did this token rely on', scene-wide."""
    flat: list[dict[str, Any]] = []
    for e in entity_scores:
        for comp, s in e.get("components", {}).items():
            flat.append({"entity": e["id"], "component": comp, "score": s})
    return sorted(flat, key=lambda a: a["score"], reverse=True)[:k]
