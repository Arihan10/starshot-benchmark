"""Wire contracts for the attention-analysis pipeline.

Two payloads cross the (potential) machine boundary between the API server and
the compute worker (Modal):

  1. `GenerationTrace` — everything a worker needs to reconstruct the model's
     native input and replay one teacher-forced forward pass. Assembled from a
     logged `cache.llm` step via `teacher_forcing.build_export` (which already
     carries the reconstructed sequence, the raw input/output frames, and the
     id -> char-span maps for scene entities / attributes).

  2. `AnalysisResult` — the SPARSE result streamed back: per generated token,
     for each selected (layer, head), the scene attention scale, entropy ratio,
     and the top-k attended entities/attributes. No dense attention tensors —
     the frontend reconstructs the visualization from these summaries.

These are plain dataclasses; `to_dict()` yields JSON-safe dicts for transport /
storage, and everything round-trips through `from_export(...)` in worker.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

# Bump when the analysis/reconstruction logic changes in a way that invalidates
# stored results. A cached result whose `meta.analysis_version` differs is
# treated as stale and recomputed by an additive "compute all" (no force needed).
#   1: initial.
#   2: trim-aware native reconstruction (recovers scene maps for prompts whose
#      user text has trailing whitespace, e.g. *_decompose steps).
#   3: top_k raised 6 -> 12 (more attended objects per token/head).
#   4: aggregation expansion — every scored token's attention is additionally
#      decomposed onto CONTEXT REGIONS (input / reasoning / output) and onto
#      WORD/TOKEN TYPES (number, structural, spatial, function, entity-name,
#      content, …), compressed into a generation-progression bucket grid and
#      exported as the unified `AnalysisResult.buckets`. New readout + wire field.
#   5: word-type `structural` split into its tag kinds (bracket / separator / quote
#      / operator) for a finer token-type breakdown — changes the type grid width,
#      so every step recomputes.
#   6: generation-progression bucket resolution raised (48 → 128) — the buckets are
#      part of the stored analysis, so bump the freshness gate (is_reusable only
#      checks ANALYSIS_VERSION) to recompute; powers the "per feature" region view.
ANALYSIS_VERSION = 6

# Version of the region/type/progression aggregate view (`AnalysisResult.buckets`).
# Recorded in `meta.agg_version` and folded into the compute-request identity
# (identity.req_token) so a bump changes the dedup hash and forces recompute.
# ANALYSIS_VERSION is the global freshness gate; AGG_VERSION lets a future
# aggregate-only change be reasoned about / targeted on its own.
#   1: initial — region x word-type mass, bucketed over generation progression,
#      head-averaged (whole-row reduction, causality-respecting).
#   2: prompt region decomposed into its XML-tag sections (`prompt.<tag>` + a
#      `prompt.unorganized` remainder), so the readout resolves WHICH organized tag
#      the model attends to vs. free prompt text.
#   3: full categorized region partition (Variables{scene_content,to_place,other} /
#      Text{organized tags, free} / Completion{reasoning,output}) with per-leaf
#      category/sub meta + token counts, plus word-type mass split by organized vs
#      free text (type_organized / type_free). Powers the stacked-area graphs.
#   4: word-type `structural` decomposed into bracket / separator / quote / operator
#      tag kinds (wider type grid); refreshes the compute-request identity.
#   5: generation-progression bucket resolution raised (48 → 128) so the "per feature"
#      view can infer each output item's region mix from the finer token buckets.
AGG_VERSION = 5

# TARGETED version for the to-place attention readout (bbox-batch steps only).
# Bumping this re-computes ONLY steps that carry a to-place batch (their stored
# `meta.to_place_version` goes stale); every other step stays fresh. See
# store.is_fresh / list_status.
#   1: initial — per-token attention onto the to-place objects (+ placement view).
TO_PLACE_VERSION = 1

# Templates known to carry a to-place batch — used only as the LEGACY heuristic
# for recompute of pre-to-place results (which lack `meta.to_place_present`).
# Going forward the worker records to-place presence data-drivenly.
BBOX_TEMPLATES = ("object_bbox_batch", "child_bbox_batch")

# How many top attended entities/attributes to keep per (token, head).
DEFAULT_TOP_K = 12

# Default cap on scored query tokens for a NEW compute. 0 = score EVERY completion
# token (full reasoning + full output) — the default, so the full output trajectory
# is always captured. A positive value bounds ONLY the reasoning region (every
# OUTPUT token is scored regardless, see worker.select_query_indices), which keeps a
# reasoning-heavy step's file small without ever dropping output detail. Folded into
# the compute-request identity, so changing it re-hashes and forces recompute.
DEFAULT_MAX_QUERY_TOKENS = 0

# --- 1. inbound: what the worker needs --------------------------------------


@dataclass
class HeadSelection:
    """A (layer, head) to instrument. For Gemma 4 only GLOBAL-attention layers
    are eligible (local sliding-window layers can't see the whole scene), so
    the eligible layer set is provider-supplied; see worker.global_attention_layers."""

    layer: int
    head: int


@dataclass
class GenerationTrace:
    """Everything streamed to a worker for one step's teacher-forced replay.

    `full_text` is the reconstructed native sequence; `completion_start` is the
    char offset where the model's generated tokens begin (the teacher-forcing
    boundary — only these are queries). `scene_map` / `output_map` are the
    tf-export maps (char spans + components + parent, in the `full` frame). The
    remote `logprobs` (output-token offsets, if captured) are carried for the
    round-trip logprob comparison, NOT as a correctness gate."""

    model_id: str
    full_text: str
    completion_start: int
    frames: dict[str, Any]
    scene_map: list[dict[str, Any]]
    output_map: list[dict[str, Any]]
    # The batch of objects being PLACED this step (bbox-batch steps), same shape
    # as scene_map (id / char span / parent / region / components). Empty for
    # steps with no to-place batch. Instrumented in parallel to the scene.
    to_place_map: list[dict[str, Any]] = field(default_factory=list)
    # Every rendered prompt variable's span in `full_text` ({name, start, end, len}),
    # from tf-export. Powers the aggregation-expansion "Variables" region category
    # (scene context / to-place / other large context blocks). Empty on older exports.
    variables_map: list[dict[str, Any]] = field(default_factory=list)
    # Optional remote generation trace for validation / alignment.
    remote_logprobs: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- 2. outbound: the sparse result -----------------------------------------


@dataclass
class EntityScore:
    """One attended scene entity (region/object) for one (token, layer, head).
    `score` is the aggregated (max-over-span, renormalized) attention onto the
    entity. `components` holds the per-attribute breakdown (placement,
    dimensions, relationships, …) so the frontend can say WHICH part was used."""

    id: str
    kind: str  # "zone" | "object"
    score: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class HeadTokenStat:
    """The per-(layer, head) stats for one generated token."""

    layer: int
    head: int
    scale: float          # c_i — fraction of attention mass on the scene
    entropy_ratio: float  # R_i — scene-normalized entropy ratio
    top_entities: list[EntityScore] = field(default_factory=list)
    # Flattened top attributes across entities (entity.component -> score), for
    # a quick "which attribute did it rely on" view without walking entities.
    top_attributes: list[dict[str, Any]] = field(default_factory=list)
    # Parallel readout onto the TO-PLACE batch (bbox-batch steps): the same
    # {scale, entropy_ratio, top_entities, top_attributes} computed over the
    # to-place columns instead of the scene. None when the step has no batch.
    to_place: dict[str, Any] | None = None


@dataclass
class TokenRecord:
    """One generated (query) token: its identity in the output frame plus the
    per-selected-head sparse attention summary."""

    index: int                 # token index in the full sequence
    char: list[int]            # [start, end) char span in full_text
    text: str
    output_entity: str | None = None   # the output assignment id this token is part of, if any
    logprob: float | None = None       # locally recomputed (real path); None for mock
    remote_logprob: float | None = None
    heads: list[HeadTokenStat] = field(default_factory=list)


@dataclass
class SceneEntityTokens:
    """Token-space location of a scene entity — the mapping the frontend uses to
    resolve attention back onto semantic components. Spans are [start, end)
    token-index ranges; `components` maps each attribute to its own token span."""

    id: str
    kind: str
    parent: str | None
    region: str | None
    token_span: list[int]
    components: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """The full sparse analysis for one step."""

    meta: dict[str, Any]
    selected_heads: list[dict[str, Any]]           # [{layer, head, mean_scale}]
    scene_entities: list[SceneEntityTokens]
    tokens: list[TokenRecord]
    # Mean scene scale for EVERY (global layer, head) — the across-heads /
    # across-depths overview grid. `selected_heads` is this list's top-k (the
    # only heads with per-token detail in `tokens`).
    head_grid: list[dict[str, Any]] = field(default_factory=list)
    # Token-space location of each TO-PLACE object (parallel to scene_entities),
    # so the frontend can resolve to-place attention back onto its components.
    # Empty for steps with no to-place batch.
    to_place_entities: list[SceneEntityTokens] = field(default_factory=list)
    logprob_check: dict[str, Any] = field(default_factory=dict)
    # Aggregation expansion (one unified, monolithic view — never fragmented into
    # side files): per generation-progression bucket, the head-averaged attention
    # mass split across context REGIONS and WORD/TOKEN TYPES. Shape + builder in
    # worker._build_buckets; passed through verbatim into the compact projection.
    #   {region_names, type_names, n_buckets, out_bucket, n_query,
    #    region: [n_buckets][n_regions], type: [n_buckets][n_types]}
    buckets: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "selected_heads": self.selected_heads,
            "head_grid": self.head_grid,
            "scene_entities": [asdict(e) for e in self.scene_entities],
            "to_place_entities": [asdict(e) for e in self.to_place_entities],
            # The fast (GPU) path builds token records as plain JSON-ready dicts
            # (skipping dataclass + the slow asdict pass — the dominant post-process
            # CPU cost that idled the GPU); the per-row fallback still yields
            # dataclasses. Handle both.
            "tokens": [asdict(t) if is_dataclass(t) else t for t in self.tokens],
            "logprob_check": self.logprob_check,
            "buckets": self.buckets,
        }
