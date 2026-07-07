"""The attention-analysis worker: reconstruct -> tokenize -> forward pass ->
stats -> semantic remap -> SPARSE result. Runs locally today via a deterministic
mock attention provider; the real GPU path (HF Gemma / FlashAttention on Modal)
implements the same `AttentionProvider` interface (see modal_app.py).

Discipline (matches the spec's compute constraints):
  * per (query, layer, head) the attention ROW is produced, immediately reduced
    to scalars + top-k, and discarded — no dense attention matrix is ever held.
  * only SELECTED scene-attending heads on GLOBAL layers are instrumented; heads
    are never averaged (that washes out the signal and multiplies cost).
  * causal masking is intrinsic: `attention_row(l, h, i)` has length i+1, and
    all scene columns live in the input (< completion_start <= i), so every
    scene key is legitimately attendable.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np

from app.attention import semantic, stats
from app.attention.schema import (
    AGG_VERSION,
    ANALYSIS_VERSION,
    DEFAULT_TOP_K,
    TO_PLACE_VERSION,
    AnalysisResult,
    EntityScore,
    HeadTokenStat,
    SceneEntityTokens,
    TokenRecord,
)

# Generation-progression compression: the per-query region/type masses are averaged
# into at most this many buckets along the generation axis (reasoning and output are
# bucketed separately so a bucket never straddles the phase boundary). Keeps the
# aggregate view tiny (B x (R+T) floats) and flat regardless of step length. Higher
# resolution also lets the "per feature" view infer each output item's region mix by
# mapping its token range onto these buckets (items ≈ buckets → ~one bucket/item).
PROGRESSION_BUCKETS = 128


class AttentionProvider(Protocol):
    """Source of post-softmax, causal attention rows for selected heads. The
    mock computes them; the real worker serves them from an instrumented
    forward pass (cached per selected head, or streamed from the kernel)."""

    def global_attention_layers(self) -> list[int]:
        """Layer indices eligible for instrumentation. For Gemma 4 these are the
        GLOBAL-attention layers only (local sliding-window layers can't see the
        whole scene)."""
        ...

    def num_heads(self) -> int: ...

    def attention_row(self, layer: int, head: int, query_i: int) -> np.ndarray:
        """Row P_{query_i, 0..query_i} (length query_i+1), post-softmax, causal."""
        ...


# --- deterministic mock provider --------------------------------------------


class MockAttentionProvider:
    """Reproducible synthetic attention so the pipeline + frontend can be built
    and validated with no model. Rows are causal and vary by (layer, head, i);
    ~1/3 of heads are 'scene-attending' (extra mass on scene columns) so head
    selection has real signal to find."""

    def __init__(self, n_tokens: int, scene_cols: set[int], *, n_layers: int = 6, n_heads: int = 8, global_layers: list[int] | None = None) -> None:
        self.n_tokens = n_tokens
        self._scene = np.zeros(n_tokens, dtype=bool)
        if scene_cols:
            self._scene[np.fromiter(scene_cols, dtype=int)] = True
        self.n_layers = n_layers
        self.n_heads = n_heads
        # Alternate global layers (a stand-in for Gemma's global/local pattern).
        self._global = global_layers if global_layers is not None else [idx for idx in range(n_layers) if idx % 2 == 1]

    def global_attention_layers(self) -> list[int]:
        return list(self._global)

    def num_heads(self) -> int:
        return self.n_heads

    def _is_scene_head(self, layer: int, head: int) -> bool:
        return head % 3 == 0

    def attention_row(self, layer: int, head: int, query_i: int) -> np.ndarray:
        j = np.arange(query_i + 1, dtype=np.int64)
        # Stable pseudo-random logits from (layer, head, i, j).
        mix = (layer * 2654435761) ^ (head * 40503) ^ (query_i * 2246822519) ^ (j * 3266489917)
        base = ((mix & 0xFFFF).astype(np.float64) / 65536.0)
        logits = base
        logits += 0.6 * (j / max(query_i, 1))          # mild recency bias
        logits[query_i] += 1.5                           # self-token
        if self._is_scene_head(layer, head):
            logits[self._scene[: query_i + 1]] += 2.2    # scene-attending head
        logits -= logits.max()
        row = np.exp(logits)
        row /= row.sum()
        return row


def rank_scene_heads(
    provider: AttentionProvider,
    scene_cols: np.ndarray,
    query_indices: list[int],
    *,
    sample: int = 32,
) -> list[dict[str, Any]]:
    """Mean scene scale for EVERY (global layer, head) across a sample of query
    steps, sorted high→low. This is the across-heads / across-depths grid; the
    caller instruments only its top-k (never averaging over heads, which washes
    out the signal). Rows are cheap: for the real provider `attention_row` just
    slices the already-cached per-layer attention."""
    if not query_indices or scene_cols.size == 0:
        return []
    step = max(1, len(query_indices) // sample)
    sampled = query_indices[::step]
    ranked: list[dict[str, Any]] = []
    for layer in provider.global_attention_layers():
        for head in range(provider.num_heads()):
            scales = [stats.scene_scale(provider.attention_row(layer, head, i), scene_cols) for i in sampled]
            ranked.append({"layer": layer, "head": head, "mean_scale": float(np.mean(scales)) if scales else 0.0})
    ranked.sort(key=lambda h: h["mean_scale"], reverse=True)
    return ranked


def select_scene_heads(
    provider: AttentionProvider,
    scene_cols: np.ndarray,
    query_indices: list[int],
    *,
    k: int,
    sample: int = 32,
) -> list[dict[str, Any]]:
    """The top-k scene-attending heads — `rank_scene_heads(...)[:k]`."""
    return rank_scene_heads(provider, scene_cols, query_indices, sample=sample)[:k]


def _topk_desc(arr: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k along the last axis, sorted descending. arr: [G, Nq, M] ->
    (indices [G, Nq, k], scores [G, Nq, k]). Stable so ties order deterministically
    (matches the per-row `sorted`)."""
    m = arr.shape[-1]
    if m == 0 or k <= 0:
        return (np.zeros(arr.shape[:2] + (0,), np.int64), np.zeros(arr.shape[:2] + (0,), np.float32))
    idx = np.argsort(-arr, axis=-1, kind="stable")[:, :, :k]
    return idx, np.take_along_axis(arr, idx, axis=-1)


def _group_membership(cols: np.ndarray, entities: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Static token<->column membership for a target group: per-entity and
    per-component LOCAL column indices (positions into `cols`) plus id/kind/name
    metadata. The GPU provider reduces the on-device attention distribution onto
    these segments (so the dense [G, Nq, S] never crosses PCIe); the CPU then only
    takes the top-k over the returned [G, Nq, E]/[G, Nq, C] scores.

    Entity order (all entities, including any with no visible column) and the flat
    component order match the legacy dense reduction exactly, so the top-k — and
    its tie ordering — is identical."""
    col_to_s = {int(c): j for j, c in enumerate(cols.tolist())}
    ent_ids = list(entities.keys())
    ent_kinds = [entities[e]["kind"] for e in ent_ids]
    ent_local: list[np.ndarray] = []                     # per entity: local col indices
    entity_comp_cols: list[list[tuple[str, int]]] = []   # per entity: [(comp_name, flat_c_index)]
    comp_entity: list[str] = []      # flat attribute -> owning entity id
    comp_name: list[str] = []        # flat attribute -> component name
    comp_local: list[np.ndarray] = []
    for eid in ent_ids:
        rec = entities[eid]
        ent_local.append(np.fromiter((col_to_s[t] for t in rec["tokens"] if t in col_to_s), dtype=np.int64))
        clist: list[tuple[str, int]] = []
        for cname, ctoks in rec.get("components", {}).items():
            ccols = np.fromiter((col_to_s[t] for t in ctoks if t in col_to_s), dtype=np.int64)
            if not ccols.size:
                continue
            clist.append((cname, len(comp_entity)))
            comp_entity.append(eid)
            comp_name.append(cname)
            comp_local.append(ccols)
        entity_comp_cols.append(clist)
    return {
        "cols": cols, "ent_ids": ent_ids, "ent_kinds": ent_kinds, "ent_local": ent_local,
        "entity_comp_cols": entity_comp_cols, "comp_entity": comp_entity,
        "comp_name": comp_name, "comp_local": comp_local,
    }


def _prep_group_from_scores(mem: dict[str, Any], ent_scores: np.ndarray, comp_scores: np.ndarray,
                            top_k: int) -> dict[str, Any]:
    """Take the top-k over the provider's GPU-reduced entity/component scores,
    producing the same dict `_build_head` consumes. The O(S) segment reduction
    already ran on-device — this only sorts the (small) per-entity/-component
    scores, so it's the tail of the old `_prep_group` with the reduction removed."""
    E, C = len(mem["ent_ids"]), len(mem["comp_entity"])
    ent_ti, ent_ts = _topk_desc(ent_scores, min(top_k, E))
    comp_ti, comp_ts = _topk_desc(comp_scores, min(top_k, C))
    return {
        "ent_ids": mem["ent_ids"], "ent_kinds": mem["ent_kinds"],
        "entity_comp_cols": mem["entity_comp_cols"], "comp_entity": mem["comp_entity"],
        "comp_name": mem["comp_name"], "comp_scores": comp_scores,
        "ent_ti": ent_ti, "ent_ts": ent_ts, "comp_ti": comp_ti, "comp_ts": comp_ts,
    }


def _build_head(grp: dict[str, Any], g: int, qi: int, top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assemble (top_entities, top_attributes) for one (head g, token qi) from a
    prepped group's arrays — as plain JSON-ready dicts (the shape asdict(EntityScore)
    would produce), so the fast path never builds/serializes dataclasses."""
    top_entities: list[dict[str, Any]] = []
    for rank in range(grp["ent_ti"].shape[-1]):
        s = float(grp["ent_ts"][g, qi, rank])
        if s <= 0:
            break
        e = int(grp["ent_ti"][g, qi, rank])
        comps = {cn: round(float(grp["comp_scores"][g, qi, ci]), 6)
                 for (cn, ci) in grp["entity_comp_cols"][e] if grp["comp_scores"][g, qi, ci] > 0}
        top_entities.append({"id": grp["ent_ids"][e], "kind": grp["ent_kinds"][e], "score": round(s, 6), "components": comps})
    top_attributes: list[dict[str, Any]] = []
    for rank in range(grp["comp_ti"].shape[-1]):
        s = float(grp["comp_ts"][g, qi, rank])
        if s <= 0:
            break
        c = int(grp["comp_ti"][g, qi, rank])
        top_attributes.append({"entity": grp["comp_entity"][c], "component": grp["comp_name"][c], "score": round(s, 6)})
    return top_entities, top_attributes


def _fast_token_records(
	trace: Any, offsets: list[tuple[str, int, int]], query_indices: list[int],
	selected: list[dict[str, Any]], scene: dict[str, Any], to_place: dict[str, Any] | None,
	*, top_k: int,
) -> list[dict[str, Any]]:
	"""Build per-token records from the GPU-reduced arrays as plain JSON-ready dicts
	(matching asdict(TokenRecord/HeadTokenStat) exactly). This is the GPU worker's
	post-processing hot loop — emitting dicts directly avoids building ~100k+
	dataclass objects AND the slow recursive asdict() pass in to_dict(), which was
	the dominant CPU cost stalling the GPU between forwards. `scene`/`to_place` are
	{scale [G,Nq], ratio [G,Nq], grp} from `_prep_group_from_scores`."""
	scene_grp = scene["grp"]
	tp_grp = to_place["grp"] if to_place else None
	records: list[dict[str, Any]] = []
	for qi, i in enumerate(query_indices):
		out_entity = semantic.output_entity_at(trace.output_map, i, i, offsets)
		heads: list[dict[str, Any]] = []
		for g, h in enumerate(selected):
			te, ta = _build_head(scene_grp, g, qi, top_k)
			stat: dict[str, Any] = {
				"layer": h["layer"], "head": h["head"],
				"scale": round(float(scene["scale"][g, qi]), 6),
				"entropy_ratio": round(float(scene["ratio"][g, qi]), 6),
				"top_entities": te, "top_attributes": ta, "to_place": None,
			}
			if tp_grp is not None:
				tte, tta = _build_head(tp_grp, g, qi, top_k)
				stat["to_place"] = {
					"scale": round(float(to_place["scale"][g, qi]), 6),
					"entropy_ratio": round(float(to_place["ratio"][g, qi]), 6),
					"top_entities": tte, "top_attributes": tta,
				}
			heads.append(stat)
		records.append({
			"index": i, "char": [offsets[i][1], offsets[i][2]], "text": offsets[i][0],
			"output_entity": out_entity, "logprob": None, "remote_logprob": None, "heads": heads,
		})
	return records


def _bucketize(mat: np.ndarray, out_q: int, max_buckets: int) -> tuple[list[list[float]], int]:
    """Average a [Nq, K] per-query matrix into <= max_buckets rows along the query
    axis. Reasoning ([0, out_q)) and output ([out_q, Nq)) are bucketed SEPARATELY
    (buckets split proportional to their query counts) so a bucket never straddles
    the phase boundary. Returns (grid [B][K], out_bucket = first output bucket)."""
    nq, k = mat.shape
    if nq == 0:
        return [], 0
    out_q = max(0, min(int(out_q), nq))
    if out_q <= 0 or out_q >= nq:                 # single phase
        segments = [(0, nq, min(max_buckets, nq))]
        out_bucket = 0 if out_q <= 0 else min(max_buckets, nq)
    else:
        br = max(1, min(max_buckets - 1, round(max_buckets * out_q / nq)))
        segments = [(0, out_q, min(br, out_q)), (out_q, nq, min(max_buckets - br, nq - out_q))]
        out_bucket = min(br, out_q)
    grid: list[list[float]] = []
    for lo, hi, nb in segments:
        n = hi - lo
        if n <= 0 or nb <= 0:
            continue
        for b in range(nb):
            a = lo + (b * n) // nb
            z = min(hi, max(a + 1, lo + ((b + 1) * n) // nb))
            sub = mat[a:z]
            grid.append([round(float(v), 6) for v in sub.mean(axis=0)] if sub.shape[0] else [0.0] * k)
    return grid, out_bucket


def _assemble_buckets(whole: dict[str, Any] | None, out_q: int,
                      region_tokens: list[int], type_tokens: list[int], n_tokens: int) -> dict[str, Any]:
    """The unified aggregation-expansion view. Head-average each per-(head, query)
    whole-row group in `whole["groups"]` and compress it along generation
    progression into buckets (reasoning/output split respected). Emits the region
    partition (+ per-leaf category/sub meta + token counts), the aggregate word-type
    grid, its organized/free splits, and the scene token count — everything the
    stacked-area graphs + normalize toggle need. A scene-less step (no selected
    heads) carries no view."""
    groups = (whole or {}).get("groups") or {}
    reg = np.asarray(groups.get("region"), dtype=np.float64) if "region" in groups else np.zeros((0, 0, 0))
    if reg.ndim != 3 or reg.shape[0] == 0 or reg.shape[1] == 0:
        return {}
    nq = reg.shape[1]
    b = max(1, min(PROGRESSION_BUCKETS, nq))
    names = whole.get("names") or {}

    def _grid(name: str) -> list[list[float]]:
        arr = groups.get(name)
        if arr is None:
            return []
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim != 3 or a.shape[2] == 0:
            return []
        g, _ = _bucketize(a.mean(axis=0), out_q, b)
        return g

    reg_grid, out_bucket = _bucketize(reg.mean(axis=0), out_q, b)
    return {
        "n_buckets": len(reg_grid), "out_bucket": out_bucket, "n_query": int(nq), "n_tokens": int(n_tokens),
        "region_names": list(names.get("region", [])), "region_meta": whole.get("region_meta", []),
        "region": reg_grid, "region_tokens": list(region_tokens),
        "type_names": list(names.get("type", [])), "type": _grid("type"),
        "type_organized": _grid("type_organized"), "type_free": _grid("type_free"),
        "type_tokens": list(type_tokens),
    }


def _scene_entities_out(scene_idx: semantic.SceneTokenIndex) -> list[SceneEntityTokens]:
    return [
        SceneEntityTokens(
            id=eid, kind=rec["kind"], parent=rec["parent"], region=rec["region"],
            token_span=rec["tokens"], components=rec["components"],
        )
        for eid, rec in scene_idx.entities.items()
    ]


def select_query_indices(
    offsets: list[tuple[str, int, int]], trace: Any, max_query_tokens: int = 0,
) -> list[int]:
    """The completion token positions that get SCORED (ascending). Every completion
    token by default; when `max_query_tokens` caps a long step, keep EVERY output
    token (the output->object view relies on the full output trace) and evenly
    subsample only the reasoning region — long steps stay fast without dropping
    output detail (the per-query reduction is O(query x head x scene)).

    SINGLE SOURCE OF TRUTH for the scored set: the GPU worker calls this to choose
    which query rows' Q to RETAIN during the capture forward
    (`modal_app.prepare(capture_rows=...)`), and `analyze` calls it to choose which
    rows to score. Sharing it guarantees the scored positions (`query_indices`, and
    `sampled` ⊆ it) are a subset of the captured rows, so the provider's
    position<->row remap is exact."""
    q0 = semantic.completion_token_start(offsets, trace.completion_start)
    all_queries = list(range(q0, len(offsets)))
    out_char = trace.frames.get("output", {}).get("start", -1)
    out_char = out_char if isinstance(out_char, int) and out_char >= 0 else None
    reasoning = [i for i in all_queries if out_char is None or offsets[i][1] < out_char]
    output = [i for i in all_queries if out_char is not None and offsets[i][1] >= out_char]
    # EVERY output token is ALWAYS scored — the output->object trajectory + the
    # per-section/word-type views depend on the complete output. The cap bounds ONLY
    # the reasoning region: it keeps whatever budget remains after the (full) output,
    # evenly subsampled. So a long output keeps all its tokens (reasoning collapses to
    # nothing) instead of being decimated — the old "subsample a huge output too"
    # branch violated this contract and silently truncated 10k-token outputs to 512.
    if max_query_tokens and len(reasoning) > max(max_query_tokens - len(output), 0):
        keep_r = max(max_query_tokens - len(output), 0)
        if keep_r <= 0:
            reasoning = []
        else:
            stepf = len(reasoning) / keep_r
            reasoning = [reasoning[int(n * stepf)] for n in range(keep_r)]
        return sorted(set(reasoning + output))
    return all_queries


def analyze(
    trace: Any,  # schema.GenerationTrace
    *,
    tokenizer: semantic.Tokenizer | None = None,
    provider: AttentionProvider | None = None,
    top_k: int = DEFAULT_TOP_K,
    agg: stats.AggMethod = "max",
    max_heads: int = 4,
    max_query_tokens: int = 0,
) -> AnalysisResult:
    """Full analysis for one step. With no tokenizer/provider it runs the mock
    path (local, model-free); pass a real Gemma tokenizer + provider for the GPU
    replay. When `max_query_tokens` is 0 (default), every completion token is
    scored. A positive cap evenly subsamples very long reasoning regions so the
    (O(N) per query, per head) reduction stays responsive."""
    tok = tokenizer or semantic.get_tokenizer(trace.model_id, prefer_real=False)
    offsets = tok.encode_with_offsets(trace.full_text)
    scene_idx = semantic.build_scene_entities(trace.scene_map, offsets)
    scene_cols = np.array(sorted(scene_idx.scene_tokens), dtype=np.int64)
    # The TO-PLACE batch (bbox-batch steps) is instrumented in PARALLEL to the
    # scene — same math on a different input column set. Data-driven: present iff
    # the step carried a to-place batch.
    tp_map = getattr(trace, "to_place_map", None) or []
    tp_idx = semantic.build_scene_entities(tp_map, offsets) if tp_map else None
    tp_cols = np.array(sorted(tp_idx.scene_tokens), dtype=np.int64) if tp_idx else np.array([], dtype=np.int64)
    has_to_place = bool(tp_idx is not None and tp_cols.size)

    # Which completion positions get SCORED. Shared with the GPU worker's Q capture
    # (`modal_app.prepare(capture_rows=...)`) via `select_query_indices`, so the
    # retained Q rows EXACTLY cover the scored queries.
    q0 = semantic.completion_token_start(offsets, trace.completion_start)
    n_all = len(offsets) - q0            # every completion token is a candidate query
    query_indices = select_query_indices(offsets, trace, max_query_tokens)

    # Aggregation expansion inputs — shared by both compute paths, built on the SAME
    # offsets the attention columns use so they line up exactly. Regions come from
    # the tf-export frames (input / reasoning / output); word/token classes from a
    # dependency-free classifier (entity NAME tokens across scene + to-place mark the
    # proper-noun/id class). `out_q` = split index in query order (reasoning|output)
    # for progression bucketing.
    name_tokens = semantic.name_token_set(scene_idx, tp_idx)
    variables_map = getattr(trace, "variables_map", None) or []
    region_names, region_seg, region_meta = semantic.region_segments(
        trace.frames, offsets, trace.full_text, variables_map)
    type_names, type_ids = semantic.classify_tokens(offsets, name_tokens)
    type_seg: list[list[int]] = [[] for _ in type_names]
    for _ti, _c in enumerate(type_ids):
        type_seg[_c].append(_ti)
    # Word-type mass restricted to ORGANIZED vs FREE prompt text (Graph 2 subgraphs):
    # intersect each type class's columns with the organized-tag / free-text regions.
    organized_cols: set[int] = set()
    free_cols: set[int] = set()
    for _seg, _m in zip(region_seg, region_meta):
        if _m.get("category") == "text" and _m.get("sub") == "organized":
            organized_cols.update(_seg)
        elif _m.get("category") == "text" and _m.get("sub") == "free":
            free_cols.update(_seg)
    type_org_seg = [[c for c in seg if c in organized_cols] for seg in type_seg]
    type_free_seg = [[c for c in seg if c in free_cols] for seg in type_seg]
    _out_char = trace.frames.get("output", {}).get("start", -1) if isinstance(trace.frames, dict) else -1
    _out_char = _out_char if isinstance(_out_char, int) and _out_char >= 0 else None
    out_q = sum(1 for i in query_indices if _out_char is None or offsets[i][1] < _out_char)
    # Named whole-row reduction groups — one reusable primitive (GPU + fallback) over
    # all decompositions. Region meta + per-leaf token counts ride along for the
    # frontend's category rollup and the scene-length normalize toggle.
    whole_plan = {
        "groups": {"region": region_seg, "type": type_seg,
                   "type_organized": type_org_seg, "type_free": type_free_seg},
        "names": {"region": region_names, "type": type_names,
                  "type_organized": type_names, "type_free": type_names},
        "region_meta": region_meta,
    }
    region_tokens = [len(s) for s in region_seg]
    type_tokens = [len(s) for s in type_seg]

    prov = provider or MockAttentionProvider(len(offsets), scene_idx.scene_tokens)
    buckets: dict[str, Any] = {}

    # Fast path: the provider computes rank + per-token scene stats as BATCHED GPU
    # matmuls (the real HF worker) and reduces the distribution onto entities/
    # components ON-DEVICE — only the small reduced scores cross to the host, where
    # we take the top-k. Falls back to the per-row loop for the mock/local provider
    # or on any error.
    head_grid: list[dict[str, Any]] = []
    token_records: list[TokenRecord] = []
    selected: list[dict[str, Any]] = []
    fast_ok = False
    if getattr(prov, "supports_batched", False) and scene_cols.size and query_indices:
        try:
            step = max(1, len(query_indices) // 32)
            sampled = np.array(query_indices[::step], dtype=np.int64)
            head_grid = prov.rank_heads(scene_cols, sampled)          # every (global layer, head)
            selected = head_grid[:max_heads] if max_heads else head_grid
            # Token<->entity membership per group; the provider reduces the on-device
            # distribution onto it (no [G, Nq, S] transfer) and returns [G, Nq, E] /
            # [G, Nq, C] scores.
            scene_mem = _group_membership(scene_cols, scene_idx.entities)
            plans: dict[str, dict[str, Any]] = {"scene": {
                "cols": scene_cols, "ent_local": scene_mem["ent_local"],
                "comp_local": scene_mem["comp_local"], "agg": agg,
            }}
            tp_mem: dict[str, Any] | None = None
            if has_to_place:
                tp_mem = _group_membership(tp_cols, tp_idx.entities)
                plans["to_place"] = {
                    "cols": tp_cols, "ent_local": tp_mem["ent_local"],
                    "comp_local": tp_mem["comp_local"], "agg": agg,
                }
            by_group, whole = prov.head_token_stats(
                selected, np.array(query_indices, dtype=np.int64), plans, whole_plan=whole_plan)
            buckets = _assemble_buckets(whole, out_q, region_tokens, type_tokens, len(offsets))
            sc_scale, sc_ratio, sc_ent, sc_comp = by_group["scene"]
            scene_g = {"scale": sc_scale, "ratio": sc_ratio,
                       "grp": _prep_group_from_scores(scene_mem, sc_ent, sc_comp, top_k)}
            tp_g = None
            if has_to_place and tp_mem is not None:
                tp_scale, tp_ratio, tp_ent, tp_comp = by_group["to_place"]
                tp_g = {"scale": tp_scale, "ratio": tp_ratio,
                        "grp": _prep_group_from_scores(tp_mem, tp_ent, tp_comp, top_k)}
            token_records = _fast_token_records(
                trace, offsets, query_indices, selected, scene_g, tp_g, top_k=top_k)
            fast_ok = True
        except Exception:
            # A batched-capable provider that throws is a REAL bug — fail LOUD rather
            # than silently degrading to the slow per-row path (idle GPU), which is
            # exactly what masked a bf16 crash. The mock/local provider isn't batched,
            # so it never enters this branch and still uses the per-row path below.
            logging.exception("attention fast (GPU-batched) path failed")
            raise

    if not fast_ok:
        head_grid = rank_scene_heads(prov, scene_cols, query_indices)  # every (global layer, head)
        selected = head_grid[:max_heads] if max_heads else head_grid   # only these get per-token detail

        def _per_row(row: np.ndarray, cols: np.ndarray, entities: dict[str, Any]) -> tuple[float, float, list[EntityScore], list[dict[str, Any]]]:
            cols_i = cols[cols <= i]  # visible keys (all scene/to-place tokens are input, so < i)
            c_i = stats.scene_scale(row, cols_i)
            probs = stats.scene_distribution(row, cols_i, c_i)
            r_i = stats.entropy_ratio(row, probs, c_i)
            ent = stats.aggregate_entities(cols_i, probs, entities, method=agg)
            te = [EntityScore(id=e["id"], kind=e["kind"], score=round(e["score"], 6),
                              components={k: round(v, 6) for k, v in e["components"].items()})
                  for e in stats.top_entities(ent, top_k)]
            ta = [{**a, "score": round(a["score"], 6)} for a in stats.top_attributes(ent, top_k)]
            return round(c_i, 6), round(r_i, 6), te, ta

        # Aggregation expansion (per-row): reduce each renormalized row onto every
        # named whole-row group (clipped to visible cols <= i, so causality holds),
        # mirroring the GPU path's _row_reduce. Same reusable group structure.
        groups_np = {name: [np.asarray(s, dtype=np.int64) for s in segs]
                     for name, segs in whole_plan["groups"].items()}
        G, Nq = len(selected), len(query_indices)
        whole_acc = {name: np.zeros((G, Nq, len(segs)), np.float64) for name, segs in groups_np.items()}

        for qi, i in enumerate(query_indices):
            tok_s, tok_e = offsets[i][1], offsets[i][2]
            out_entity = semantic.output_entity_at(trace.output_map, i, i, offsets)
            heads: list[HeadTokenStat] = []
            for g, h in enumerate(selected):
                row = prov.attention_row(h["layer"], h["head"], i)
                c_i, r_i, te, ta = _per_row(row, scene_cols, scene_idx.entities)
                stat = HeadTokenStat(layer=h["layer"], head=h["head"], scale=c_i, entropy_ratio=r_i,
                                     top_entities=te, top_attributes=ta)
                if has_to_place:
                    tc, tr, tte, tta = _per_row(row, tp_cols, tp_idx.entities)
                    stat.to_place = {"scale": tc, "entropy_ratio": tr, "top_entities": tte, "top_attributes": tta}
                heads.append(stat)
                for gname, segs in groups_np.items():
                    acc = whole_acc[gname]
                    for k, seg in enumerate(segs):
                        if seg.size:
                            cc = seg[seg <= i]
                            if cc.size:
                                acc[g, qi, k] = float(row[cc].sum())
            token_records.append(TokenRecord(
                index=i, char=[tok_s, tok_e], text=offsets[i][0], output_entity=out_entity, heads=heads,
            ))
        whole = {"groups": whole_acc, "names": whole_plan["names"], "region_meta": region_meta}
        buckets = _assemble_buckets(whole, out_q, region_tokens, type_tokens, len(offsets))

    return AnalysisResult(
        # Trace/run context + export assumptions ride along FIRST, so the worker-
        # computed fields below are AUTHORITATIVE and can't be clobbered. CRITICAL:
        # the tf-export sets meta.mock=True as a *reconstruction* stand-in flag; if it
        # overrode the real attention `mock` (provider is None), every REAL result
        # would read mock=True → store.is_fresh treats it as never-fresh → the step
        # recomputes forever and results never settle. Worker keys win.
        meta={
            **(trace.meta or {}),
            "model_id": trace.model_id,
            "tokenizer": tok.name,
            "analysis_version": ANALYSIS_VERSION,
            "mock": provider is None,
            "n_tokens": len(offsets),
            "n_scene_tokens": int(scene_cols.size),
            "n_scene_entities": len(scene_idx.entities),
            "completion_token_start": q0,
            "n_query_tokens": len(query_indices),
            "n_query_tokens_total": n_all,
            "subsampled": len(query_indices) != n_all,
            "agg": agg,
            "top_k": top_k,
            # How many top (layer, head) pairs got per-token detail (the heatmap /
            # head selector). The head_grid below always covers EVERY (layer, head).
            "max_heads": len(selected),
            # Whether the batched-GPU compute path ran (vs the per-row fallback).
            "fast_path": fast_ok,
            # To-place (bbox-batch) parallel readout: version + presence, so the
            # store can target recompute of ONLY to-place-bearing steps.
            "to_place_version": TO_PLACE_VERSION,
            "to_place_present": has_to_place,
            "n_to_place_entities": len(tp_idx.entities) if tp_idx is not None else 0,
            # Aggregation expansion: region/type/progression view version + axes.
            "agg_version": AGG_VERSION,
            "n_regions": len(region_names),
            "n_type_classes": len(type_names),
            # Axes for the across-heads / across-depths grid.
            "n_heads": prov.num_heads(),
            "global_layers": list(prov.global_attention_layers()),
        },
        selected_heads=selected,
        head_grid=head_grid,
        scene_entities=_scene_entities_out(scene_idx),
        to_place_entities=_scene_entities_out(tp_idx) if tp_idx is not None else [],
        tokens=token_records,
        logprob_check=_logprob_check(trace, offsets),
        buckets=buckets,
    )


def _logprob_check(trace: Any, offsets: list[tuple[str, int, int]]) -> dict[str, Any]:
    """Surface the local-vs-remote logprob comparison (a diagnostic, never a
    gate). The mock path computes no local logprobs, so it just reports what the
    real path would compare against."""
    remote = (trace.remote_logprobs or {}).get("tokens") if trace.remote_logprobs else None
    return {
        "computed_local": False,
        "remote_available": bool(remote),
        "remote_token_count": len(remote) if remote else 0,
        "note": "mock path: local per-token logprobs require the real forward pass; "
                "the real worker compares them against remote and surfaces mismatches without blocking.",
    }
