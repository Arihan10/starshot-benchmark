# Attention aggregation expansion — context, token-type, and progression views

Scope: `server/app/attention/*`. A design + high-level rewrite plan for extending the
teacher-forced attention instrumentation so it resolves attention not only onto scene
entities, but also onto (a) **context regions** (the model's own output, reasoning vs
prompt, XML-tagged spans, and an unorganized remainder), (b) **token / word types**
(punctuation, numbers, nouns, adjectives, function words, structural tokens), tracked on
an aggregate basis, and (c) a **generation-progression** view compressed into token-chunk
averages.

This is a planning document, not code. It reuses the existing pipeline's core property —
the expensive full-row softmax normalizer is computed once per head and shared across every
column group — so the whole expansion lands at negligible compute/VRAM cost and, done
right, **without growing per-step storage**.

---

## 1. Context — what exists today

Per step, `worker.analyze` runs one teacher-forced forward (`HFAttentionProvider.prepare`),
captures post-RoPE Q/K for the global/instrumentable layers (scored rows only), then does a
sparse readout. The readout is organized around **column groups**:

- The worker builds a plan per group and hands it to the provider:

  ```python
  # worker.analyze (today)
  plans = {"scene": {"cols": scene_cols, "ent_local": ..., "comp_local": ..., "agg": agg}}
  if has_to_place:
      plans["to_place"] = {...}
  by_group = prov.head_token_stats(selected, query_indices, plans)
  # by_group[name] = (scale[G,Nq], ratio[G,Nq], ent[G,Nq,E], comp[G,Nq,C])
  ```

- `head_token_stats` (in `modal_app.py`) computes, per selected head `g`:

  ```python
  logZ, h_full = self._full_row_norm(qm, kf, qi)      # ONCE per head — the dominant cost
  for name, (cols_t, ent_t, comp_t, agg) in grp.items():
      c, dist, ratio = self._col_stats(qm, kf, logZ, h_full, qi, cols_t)  # per group
      ent_red  = _seg_reduce(dist, ent_t, agg)         # reduce [Nq,S] -> [Nq,E] on-device
      comp_red = _seg_reduce(dist, comp_t, agg)         # ...            -> [Nq,C]
  ```

Two facts drive everything below:

1. **The full-row normalizer is group-independent and shared.** `_full_row_norm` builds the
   `[Nq, seq]` scores, `logZ`, and the full renormalized `row` once per head; every group
   reuses `logZ`/`h_full`. Adding groups only adds `_col_stats` (a column-sliced matmul) and
   `_seg_reduce` (a segment reduction) — never another full-width pass.
2. **Entities, components, types, and regions are all the same primitive:** a set of
   column-index segments reduced off the on-device distribution. `_seg_reduce` already does
   exactly this for entities and components. Types and regions are just more segment sets.

Supporting structure that already exists and we reuse verbatim:

- `trace.frames` partitions `full_text` into char-span regions: `input`
  (`0 → completion_start`, not scored), `completion` (scored), `reasoning`
  (`<think>…</think>` when present), and `output`. See `teacher_forcing.build_export`.
- `semantic.tokens_in_span(offsets, cs, ce)` maps any char span to its token columns;
  `build_scene_entities` already folds char-span maps into per-entity / per-component token
  sets. The same machinery builds region and type segments.
- `derive.py` projects the canonical result into small client views (`compact`, `token`,
  `present`), gzip-compressed at rest, so the browser never pulls the whole blob.

Real scale (from `runs/`): Gemma-4-31b — 60 layers, 10 global, 32 heads, `head_dim` 128,
seq 13–23k, scene 7–16k, ~3.5k scored tokens. Qwen3.5-122B — 48 layers, 12 instrumentable,
32 Q / 2 KV heads, `head_dim` 256, seq up to **136k**, scene up to 125k, scored capped at
512. Stored sparse result ≈ 2.6 MB base / up to 25 MB in the expanded `present` view; ~1.6 GB
across 834 files today.

---

## 2. Need — what's missing analytically

Attention is currently resolved onto **scene entities only**. The benchmark can ask "which
object did this token look at," but not the questions that most distinguish good vs bad
spatial reasoning:

- **Did the model attend to its own prior output?** The completion frame is never a column
  group today, so we can't see self-referential attention (does the model consult the
  coordinates it already emitted when placing the next object?).
- **Prompt vs reasoning vs output.** We can't decompose attention mass by region — e.g. is a
  placement token grounded in the user's scene description, or in the model's own reasoning
  chain? The `frames` split exists but isn't projected onto attention.
- **XML / tag structure.** Outputs and reasoning are wrapped in delimiters and tags. We can't
  currently say "attention concentrated on the `<dimensions>` block" vs "on free-text prose."
- **Token / word types.** We can't say whether attention lands on nouns (entity names),
  numbers (coordinates/sizes), punctuation/structural tokens (JSON scaffolding), or function
  words — nor how that mix **evolves across generation** (a progression signal that likely
  separates careful reasoning from pattern-matching).

All four are aggregate, model-agnostic decompositions that the current per-entity view can't
express, and all four are cheap to add because they ride on the distribution we already
materialize on-device.

---

## 3. Goals

Analytical:

- Resolve attention onto **region groups** (input / reasoning / output / completion, plus
  per-XML-tag spans and an "unorganized" remainder), including the model's own output.
- Track **token/word-type mass** per (head, query) on an aggregate basis — a fixed, small set
  of type classes.
- Provide a **progression view**: both of the above as a function of generation position,
  compressed into token-chunk averages.

Performance (the hard constraints, unchanged from today):

- No change to the forward/capture. No dense `[heads, seq, seq]` ever. No `[Nq, seq]` or
  per-group dense distribution ever crosses PCIe — reduce on-device, ship only `[Nq, T]`.
- Added GPU compute within a few percent of the (already sub-dominant) readout; peak VRAM
  unchanged.
- Per-step storage **flat or reduced** — the new region/type/progression views are stored as
  chunked aggregates, not per-token × per-head lists.

Compatibility:

- Extend the group-plan interface **additively**; keep the mock/per-row fallback numerically
  equivalent; version the outputs so existing results stay valid and only affected steps
  recompute.

Success criteria (measurable): readout wall-clock delta < ~10%; peak VRAM delta ≈ 0; new
aggregate view < ~150 KB/step; per-token blob size non-increasing.

---

## 4. Approach — high-level rewrite plan

### 4.1 Generalize "everything is a segment set"

Refactor the reduction so entities, components, **types**, and **regions** are all expressed
as named segment sets over a group's columns, reduced by the same on-device primitive.
`_seg_reduce` already is that primitive; the change is in how plans are built and threaded,
not in the kernel.

### 4.2 Extend the group-plan interface (additive)

```python
# proposed plan shape (superset of today's)
plan = {
    "cols":        cols,          # column set for this group (or the FULL row — see 4.4)
    "ent_local":   [...],         # per-entity   column-index segments   (unchanged)
    "comp_local":  [...],         # per-component column-index segments  (unchanged)
    "type_local":  [...],         # NEW: per-type column-index segments  (punc/num/noun/...)
    "type_names":  [...],         # NEW: labels aligned to type_local
    "causal_cols": False,         # NEW: mask cols > query_i before reduce (output columns)
    "agg":         "max",
}
# head_token_stats return gains a fifth array per group:
#   (scale[G,Nq], ratio[G,Nq], ent[G,Nq,E], comp[G,Nq,C], types[G,Nq,T])
```

Types reduce with `sum`/`mean` (mass fractions), independent of the `max` used for entities;
`_seg_reduce` already takes an `agg` argument.

### 4.3 Region groups from `frames` (+ XML) and the model's own output

- Build region column sets directly from `trace.frames` with `semantic.tokens_in_span`:
  `input`, `reasoning`, `output`, and `completion`. No new span logic — the char spans already
  exist.
- **XML/tag sub-grouping:** a small tag-span scanner over a frame's text yields one column set
  per tag region (e.g. `<dimensions>`, `<placement>`), plus an **`unorganized`** bucket for
  spans covered by no tag. This mirrors `build_scene_entities` (span → tokens), just keyed by
  tag instead of entity id.
- **Output-as-columns is a correctness change, not a cost one.** Output/completion columns live
  *after* `completion_start`, so for query `i` some are causally invisible (`j > i`).
  `_col_stats` today assumes "columns are all < completion start (always causally visible), so
  no masking needed." A completion group must mask `cols > q_abs` before the softmax sum:

  ```python
  # _col_stats, when plan["causal_cols"]:
  vis = cols.unsqueeze(0) <= q_abs.unsqueeze(-1)      # [Nq, S] causal visibility
  p = torch.where(vis, torch.exp(scores - logZ.unsqueeze(-1)), 0.0)
  ```

  Cheap (one compare + mask); required, or a query picks up mass from tokens it hasn't emitted
  yet. The per-row fallback already handles this via `cols_i = cols[cols <= i]`.

### 4.4 On-device token-type reduction

- Build a per-column type label vector once per step (CPU, `O(seq)`), via a **pluggable
  classifier** in a new `semantic`-adjacent helper. Default is a dependency-free heuristic
  (punctuation, number, whitespace/structural, word-run → coarse noun/verb/adjective/function
  buckets); a real POS tagger can drop in behind the same interface. Tag **only the columns a
  group needs** (scene + output ≪ full seq on long Qwen traces) to keep tagging cheap.
- Reduce the on-device distribution onto type classes with a `scatter_reduce` over the class-id
  vector (or per-class `index_select` + `sum`/`amax`, mirroring `_seg_reduce`). Result
  `[Nq, T]`, `T ≈ 10–15`.
- **Whole-row type mass (attention on *all* nouns anywhere, not just scene) is free** if we
  reduce the `row` that `_full_row_norm` already materializes, *before* it is discarded — no
  second matmul. Only `[Nq, T]` leaves the device.

### 4.5 Progression + chunked compression

- A new `progression` helper compresses any per-token array `[G, Nq, …]` along the query axis
  into `[G, B, …]` by averaging within `B` buckets (fixed `B`, e.g. 64, or fixed width),
  **respecting the reasoning/output split** so a bucket never straddles the two frames.
- Store region/type results as `[G, B, T]` / `[B, …]`, not `[G, Nq, …]`. At `B = 64` on a
  3,500-token step that is ~**55× smaller** on the query axis.
- Keep full per-token top-k **only** for `scene` and `output` (where the UI scrubs
  token-by-token); everything else is chunked-aggregate.

### 4.6 Storage & serialization

- Add one derived projection in `derive.py` — a `buckets` view: region × type × progression
  grids (per selected head and head-summed), precomputed, small, gzipped like the others.
- Version the outputs so recompute is **targeted**, following the existing `TO_PLACE_VERSION`
  pattern: a new `AGG_VERSION` (region/type/progression) whose bump restamps only steps that
  carry the new view; `ANALYSIS_VERSION` only moves if a per-token blob shape changes.
  `store.is_fresh` / `list_status` gain the same freshness check they already do for to-place.

### 4.7 Cost, tied to the plan

| Addition | Compute | Peak VRAM | Storage |
|---|---|---|---|
| Region groups (input/reasoning/output/completion, XML, unorganized) | shared `logZ` not re-paid; adds `_col_stats` bounded by ≤ one extra full-width pass total → **< 0.1% of the step** | ~0 (sequential, dist freed each iter; peak stays at scene's `[Nq,seq]`) | flat — stored as chunked aggregates, not per-token |
| Token/word types (aggregate) | reduce off the **already-materialized** `row`; scatter over ~15 classes ≈ the `h_full` work already done | ~0 (`[Nq,T]`, T≈15) | tiny |
| Progression + chunked average | cheap axis reduction | ~0 | **reduces** (Nq → B, ~55× at B=64) |

Forward FLOPs (dominant, unchanged) ≈ `2·P_active·seq`. Readout ≈ `2·G_sel·Nq·seq·d` shared +
`2·G_sel·Nq·(ΣS_group)·d`; since `ΣS_group ≤ seq`, all new groups add at most one more
full-width pass — for Gemma that is ~6.5e11 FLOPs, **~0.05% of the forward**. The only CPU
addition is `O(seq)` tagging, one-time, off the GPU critical path (use the heuristic or tag
only grouped columns on 136k-token traces).

Net: **not a drastic increase in compute or memory; storage stays flat or shrinks.**

---

## 5. Phased rollout

Each phase lands behind the mock/per-row fallback, is numerically checked against that
fallback, and is independently shippable.

- **Phase 0 — segment generalization.** Make types/regions first-class segment sets in the plan
  and return shape. Pure refactor; no new outputs, no behavior change.
- **Phase 1 — region groups + output causal mask.** Add input/reasoning/output/completion
  groups from `frames`; add the `causal_cols` mask to `_col_stats`. Store as chunked
  aggregates. This alone answers "prompt vs reasoning vs own-output."
- **Phase 2 — token-type reduction.** Heuristic classifier + on-device type reduce (off the
  shared `row` for whole-sequence mass).
- **Phase 3 — XML/tag sub-grouping.** Tag scanner over frame text; `unorganized` remainder.
- **Phase 4 — progression views + `buckets` derived projection + UI.** Wire the chunked grids
  into `derive.py` and the frontend.

---

## 6. Risks & correctness notes

- **Output-column causal masking (must-fix).** Any completion/output/reasoning column group
  needs `cols ≤ i` masking in the batched `_col_stats`, or mass leaks from not-yet-generated
  tokens. Covered in 4.3.
- **Don't spawn a group-per-type over the full row.** Reduce all types in one scatter off the
  shared `row`; a separate `_col_stats` per class over `seq` columns would be ~T full-width
  passes for no reason.
- **Never transfer `[Nq, seq]` or dense per-group distributions.** Keep the PERF_REVIEW P0
  discipline: reduce on-device, ship only `[Nq, groups/types]`.
- **Tagger cost on long traces.** A real POS tagger on a 136k-token Qwen trace is ~10s CPU;
  prefer the heuristic, or tag only the columns actually grouped.
- **Offset-frame alignment.** Type labels and region spans must be built on the *same*
  `offsets` the attention is indexed by (reuse `provider.encode_with_offsets`), or classes
  misalign with columns.
- **Versioning.** Use a targeted `AGG_VERSION` so the expansion doesn't invalidate the ~834
  existing results; only steps that gain the new view restamp.

---

## 7. Interfaces touched (file-by-file)

- `schema.py` — add `AGG_VERSION` (+ bump notes); extend the plan/return contracts and
  `HeadTokenStat`/result meta for region/type/progression aggregates.
- `worker.py` — build region + type segments from `frames`/classifier; thread the extended
  plans through `analyze`; chunk per-token arrays into progression buckets.
- `modal_app.py` — `_col_stats` gains `causal_cols` masking; `head_token_stats` returns the
  `types` array; add the on-device `row`-based type reduction in `_full_row_norm`'s scope.
- `semantic.py` — region column-set builders from `frames`; a new pluggable token-type
  classifier (heuristic default).
- `derive.py` — new `buckets` projection (region × type × progression), gzipped.
- `store.py` — targeted freshness/recompute for `AGG_VERSION` (mirrors the to-place path).
