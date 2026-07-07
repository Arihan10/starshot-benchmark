"""Modal GPU backend for teacher-forced attention instrumentation.

Deploy from `server/`:  `modal deploy -m app.attention.modal_app`
Everything is defined at MODULE scope (Modal requires it) and guarded on
`import modal`, so the module still imports where modal isn't installed (the
symbol `app` is then None — the API server never uses it).

It reuses the local pipeline verbatim: `reconstruct.build_real_trace` (the
model's native chat-templated input + remapped maps), then `worker.analyze`
with a real tokenizer + attention provider — producing the identical sparse
`AnalysisResult` the mock path does, only with real attention numbers.

Open models are declared in `models.py` (HF path, GPU tier, per-model quirks);
closed models are gated off. The `web` endpoint routes each model to its GPU-tier
worker class (there is one `@app.cls` per tier because Modal fixes `gpu` per class).

GPU / reliability
  * per-tier workers: `WorkerH200` (one **H200**, 141 GB — Gemma-31b bf16 ~62 GB)
    and `WorkerB200x2` (**B200:2**, ~360 GB, `device_map="auto"` pipeline-shard —
    Qwen3.5-122B-A10B, a hybrid Gated-DeltaNet/Gated-Attention MoE, ~250 GB bf16;
    only its 12 Gated-Attention layers carry instrumentable softmax attention). Add
    a tier by adding an `@app.cls` + an entry in `_WORKER_BY_GPU`. Each tier ships
    its own image: Gemma on the proven cu12x/H200 build, the big MoE on a
    Blackwell (cu128) torch + transformers>=5.2 build (see `big_image`).
  * bf16 (reference precision; fp16 is unstable on large variants).
  * weights cached in a Volume + `hf_transfer` for fast, reliable downloads;
    model loaded ONCE per container via `@modal.enter`.
  * ONE container only (`max_containers=1`, no `@modal.concurrent` on the GPU
    class): Modal never fans out to multiple GPU replicas and never runs two
    forwards at once. Spawned jobs queue to this single worker and drain FIFO
    (VRAM-safe). The front-door `web` function is a single concurrent CPU
    container that SPAWNS jobs and hands back call ids to poll (Modal caps web
    requests at 150s, so we never block one on the GPU forward).

Memory safety (the seq^2 footgun)
  `output_attentions=True` forces the EAGER kernel to materialize a full
  [heads, seq, seq] score matrix (fp32 in softmax) for the forward — on long
  traces that alone is tens-to-hundreds of GiB and OOMs *inside* the model,
  before any hook can run. So we DON'T ask for attentions. The forward runs on
  SDPA (flash/mem-efficient — never materializes seq^2), and a registered custom
  attention op captures only the post-RoPE Q/K of the GLOBAL layers. The scene
  statistics are then batched matmuls over those cached Q/K (only scene columns +
  the per-row softmax normalizer), so nothing seq^2 is ever held — peak extra
  memory is O(global_layers * heads * seq * head_dim) for the cached Q/K (kept on
  the GPU in the model's bf16 for those matmuls; a per-row recompute is the
  fallback) plus the scene-column scores. Teacher-forced logprobs are computed in
  vocab-chunks for the same reason.

Accuracy
  native `apply_chat_template`; the forward runs on EXACTLY the ids tokenized
  for the offset map; per-token logprobs are recomputed and compared to the
  remote trace (diagnostic only — surfaced, never blocking).

NOTE: deploy with Python >= 3.10. The `X | Y` type unions are evaluated at import,
and `from __future__ import annotations` is NOT usable here — it turns the
`modal.parameter` field annotations into strings and breaks Modal's class-parameter
type validation (`'str' object has no attribute '__name__'`).
"""

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import modal  # type: ignore
    _HAS_MODAL = True
except Exception:  # not installed (e.g. imported by the API server)
    modal = None  # type: ignore
    _HAS_MODAL = False

# Open-weight model registry (HF paths, GPU tiers, per-model quirks). Dual import
# so it resolves both as `app.attention.models` (server / `-m app.attention.modal_app`)
# and as top-level `models` (`modal deploy modal_app.py` from this dir).
try:
    from app.attention import models as reg  # type: ignore
except Exception:  # pragma: no cover
    import models as reg  # type: ignore

# Shared cell-identity + HMAC freshness stamp (dual import, same as `reg`).
try:
    from app.attention import identity as ident  # type: ignore
except Exception:  # pragma: no cover
    import identity as ident  # type: ignore

# View projections (dual import, same as `reg`). The worker builds the SMALL
# `compact` view right here (it already holds the full result in memory) and
# stores it as its own blob, so the frequent status poll pulls only the compact —
# the multi-hundred-MB full result never rides the web tier on the hot path.
try:
    from app.attention import derive as derive  # type: ignore
except Exception:  # pragma: no cover
    import derive as derive  # type: ignore

# HF chat templates render the model's native thinking delimiters; OpenRouter
# strips them. reconstruct.py re-inserts the model's delimiters (from its
# OpenModelSpec) around the reasoning; the logprob round-trip confirms the format.

MODAL_APP_NAME = "starshot-attention"
DEFAULT_MODEL_ID = "google/gemma-4-31b-it"

# Registered name for our custom SDPA-based attention op (captures global-layer
# Q/K without materializing the seq^2 score matrix). See `_scene_capture_attention`.
SCENE_ATTN_IMPL = "scene_capture"
# Vocab-chunk (in query positions) for the teacher-forced logprob pass, so we
# never hold a full [seq, vocab] fp32 tensor at once.
LOGPROB_CHUNK = 512
# The teacher-forced logprob check is DIAGNOSTIC (never blocking) and only reads the
# OUTPUT frame, which is the tail of the sequence. Materializing logits for EVERY
# position is the seq*vocab footgun: HF upcasts the LM-head output to fp32, so a long
# trace (e.g. ~93k tokens * ~262k vocab * 4 B ≈ 98 GiB) OOMs the GPU on that ONE
# tensor. So we project the LM head over only the last `LOGITS_KEEP_MAX` positions
# (via `logits_to_keep`) — enough to cover any real completion — which bounds that
# tensor to a few GiB regardless of prompt length. Attention capture is unaffected
# (it reads per-layer Q/K during the forward, not the final logits).
LOGITS_KEEP_MAX = 8192

# device_map="auto" fills GPUs greedily and PACKS GPU 0 with weights (~95% of the
# card), leaving little for the forward's transient activations, the lm-head window,
# or this item's captured Q — the imbalance behind long-trace "GPU 0 OOM while GPU 1
# is half empty" failures. Reserve this many GiB of WEIGHT-FREE headroom PER GPU
# (capping each device's weight budget) so that runtime memory has somewhere to go.
# The big MoE (~250 GB bf16) still fits with margin: on B200:2 that's
# 2 x (180 - 24) = 312 GB of weight budget. Single-GPU tiers (device_map="cuda")
# are unaffected.
_WEIGHT_HEADROOM_GIB = 24.0

# Module-global the custom attention op writes into during a single forward.
# Safe because a container runs its items sequentially (one forward at a time).
_ACTIVE_CAPTURE: dict[str, Any] | None = None


def _scene_capture_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Custom attention op registered as `SCENE_ATTN_IMPL`. Computes the layer
    output with SDPA and, for the GLOBAL layers only, stashes the post-RoPE Q/K so
    `HFAttentionProvider` can recompute individual attention rows on demand. Returns
    `(attn_output, None)` — we never return dense weights. Shapes match the eager op:
    q/k/v are [B, heads, seq, d] (k/v with num_kv_heads); output is [B, seq, heads, d].

    MEMORY: the whole point of the design is to never materialize the [heads, seq,
    seq] score matrix. PyTorch's MATH SDPA backend DOES materialize it (in fp32),
    which OOMs long traces — e.g. 32 heads x 40k^2 x 4B ~= 190 GiB. The MATH backend
    gets selected whenever a dense `attn_mask` is passed together with `enable_gqa`
    (Flash rejects the dense mask; the mem-efficient kernel rejects GQA), which is
    exactly what a Gemma/Qwen forward hands us. So we steer every layer onto a
    tiling kernel (Flash / cuDNN / mem-efficient), which keeps O(seq) memory:

      * FULL-attention layers (all the layers we CAPTURE, plus any all-global model)
        use a plain causal mask under batch-1, unpadded teacher forcing, so we DROP
        the dense mask and pass `is_causal=True`. For head_dim <= 256 Flash handles
        causal + GQA directly (no K/V copy). For a BIGGER head_dim (e.g. 512, which
        Flash <=256 and cuDNN <=128 both reject) only the mem-efficient kernel is
        eligible, and it has no `enable_gqa` — so we expand K/V to Q's head count and
        run causal there. Either way O(seq), never a [heads, seq, seq] score matrix.
      * SLIDING-window layers keep their banded mask but run on the mem-efficient /
        cuDNN kernels, with K/V physically expanded (mem-efficient lacks GQA). The
        banded mask stays [B, 1, seq, seq] (broadcast over heads) — never [heads,
        seq, seq] scores.

    A `sdpa_kernel(...)` context forces the chosen backends (MATH excluded), so a
    long trace can't silently fall back to the seq^2 path. The reference (MATH)
    path is kept only as a last-resort fallback for inputs no tiling kernel accepts
    (short traces — long ones would OOM there, which the caller reports clearly)."""
    import torch  # noqa: F401  (available on the GPU container)
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    n_rep = query.shape[1] // key.shape[1]
    gqa = n_rep > 1
    cap = _ACTIVE_CAPTURE
    idx = getattr(module, "_attn_layer_idx", -1)
    # Full attention: no mask at all, or a layer we know is global (full causal).
    is_full = attention_mask is None or (cap is not None and idx in cap["global"])
    causal = query.shape[2] > 1
    # Flash caps head_dim at 256 and cuDNN at 128; ABOVE that (e.g. a head_dim=512 model)
    # the mem-efficient kernel is the ONLY eligible tiling kernel — and it has no
    # `enable_gqa`, so GQA K/V must be PHYSICALLY expanded to match Q. Detect the big
    # head_dim up front so we route straight to the expand path instead of asking Flash
    # to run something it can't: that mismatch left a head_dim=512 GQA model with NO
    # eligible fused kernel, dropping it onto the seq^2 MATH path — the "375 GiB OOM"
    # that was never actually a size limit.
    big_head = query.shape[-1] > 256
    tiling = [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    try:
        if is_full and not big_head:
            # Fast path: Flash does causal + GQA WITHOUT copying K/V (head_dim ≤ 256).
            with sdpa_kernel(tiling):
                attn_output = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, scale=scaling,
                    is_causal=causal, enable_gqa=gqa,
                )
        else:
            # Sliding/dense mask OR a big head_dim: expand K/V (the mem-efficient kernel
            # has no enable_gqa) and run a tiling kernel. A FULL layer drops the mask and
            # goes causal; a SLIDING layer keeps its [B, 1, seq, seq] banded mask
            # (broadcast over heads — never [heads, seq, seq] scores). O(seq) either way.
            k_e = key.repeat_interleave(n_rep, dim=1) if gqa else key
            v_e = value.repeat_interleave(n_rep, dim=1) if gqa else value
            mask = None if is_full else attention_mask[:, :, :, : key.shape[-2]].contiguous()
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
                attn_output = F.scaled_dot_product_attention(
                    query, k_e, v_e, attn_mask=mask, dropout_p=0.0, scale=scaling,
                    is_causal=is_full and causal,
                )
    except RuntimeError as e:
        # A genuine OOM from a tiling kernel means the trace truly doesn't fit — don't
        # retry MATH (it materializes seq^2 and would OOM harder); let it propagate so
        # `_worker_analyze` reports the clear non-retryable reason.
        if _is_oom(e):
            raise
        # Otherwise no tiling kernel ACCEPTED these inputs (eligibility, not memory) —
        # reference path materializes seq^2, viable only for the short traces that
        # reach here (long ones would have OOM'd on a tiling kernel above).
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=None if is_full else attention_mask[:, :, :, : key.shape[-2]],
            dropout_p=0.0, scale=scaling,
            is_causal=is_full and causal, enable_gqa=gqa,
        )
    attn_output = attn_output.transpose(1, 2).contiguous()

    if cap is not None and idx in cap["global"]:
        # Retain Q ONLY for the query rows the scene stats will actually score
        # (`cap["rows"]` = the completion positions `worker.select_query_indices`
        # picked). The forward computed Q for all `seq` positions, but KEEPING every
        # row is the accumulator that OOMs long traces: 32 heads x 146k x 256 x 2B
        # ~= 2.4 GiB PER captured layer x 12 ~= 29 GiB of retained Q (packed onto a
        # couple of shards). `index_select` copies out just the scored rows (a few
        # hundred–thousand), so the full-length Q is freed with the layer and the
        # gathered vectors are BIT-IDENTICAL to what the old absolute indexing read
        # — a pure memory win, no numeric change. rows=None -> keep all (back-compat).
        # K stays FULL: the causal-row softmax normalizer needs every key.
        rows = cap.get("rows")
        if rows is None:
            cap["q"][idx] = query.detach()
        else:
            ri = cap["_rows_dev"].get(query.device)
            if ri is None:
                ri = torch.as_tensor(rows, device=query.device, dtype=torch.long)
                cap["_rows_dev"][query.device] = ri
            cap["q"][idx] = query.detach().index_select(2, ri)  # [B, H, n_rows, d]
        cap["k"][idx] = key.detach()  # keep kv heads un-repeated; rows map head -> head // n_rep
        cap["scaling"] = scaling
        cap["n_rep"] = n_rep
    return attn_output, None


def resolve_text_config(config: Any) -> Any:
    """The TEXT sub-config. Multimodal Gemma (`Gemma*ForConditionalGeneration`)
    nests the language-model fields under `config.text_config`; the top-level
    config does NOT proxy `num_hidden_layers` / `num_attention_heads` /
    `sliding_window_pattern`, so reading them off it yields 0 / None. HF's
    official accessor is `get_text_config()` (returns self for text-only)."""
    getter = getattr(config, "get_text_config", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            pass
    return getattr(config, "text_config", config)


def decoder_layers(model: Any) -> Any:
    """The decoder's `nn.ModuleList` of transformer blocks, matching the official
    HF layout. `model.get_decoder()` is the canonical accessor: text-only
    `Gemma*ForCausalLM.get_decoder()` returns its `Gemma*TextModel`; the
    multimodal `Gemma*ForConditionalGeneration.get_decoder()` unwraps through
    `Gemma*Model.get_decoder()` to the inner `language_model`. Both expose
    `.layers`. Fall back to the known nestings for other wrappers."""
    getter = getattr(model, "get_decoder", None)
    if callable(getter):
        try:
            layers = getattr(getter(), "layers", None)
            if layers is not None:
                return layers
        except Exception:
            pass
    # Fallbacks: text-only (.model.layers) and multimodal (.model.language_model.layers).
    for path in (("model",), ("model", "language_model"), ("language_model",)):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        layers = getattr(obj, "layers", None)
        if layers is not None:
            return layers
    raise AttributeError(
        f"could not locate decoder layers on {type(model).__name__} "
        "(tried get_decoder() and .model[.language_model].layers)"
    )


def global_attention_layers(config: Any) -> list[int]:
    """Indices of the layers whose SOFTMAX attention we can instrument (pass the
    result of `resolve_text_config`).

    These are the "full/global attention" layers:
      * Gemma interleaves local (sliding-window) + global (Gemma 3/4: 5:1).
      * Qwen3.5/3.6/Next are HYBRID: ~3x Gated-DeltaNet (LINEAR attention -> no
        softmax scene distribution) : 1x Gated-Attention (standard softmax). Only
        the Gated-Attention layers are instrumentable.

    Prefer explicit `layer_types` (keep full/global, DROP linear/delta/sliding);
    else a cadence field (`sliding_window_pattern` for Gemma, `full_attention_
    interval` / `full_attn_interval` for Qwen-style); else all layers."""
    n = int(getattr(config, "num_hidden_layers", 0) or 0)
    layer_types = getattr(config, "layer_types", None)
    if isinstance(layer_types, (list, tuple)) and layer_types:
        picked = [
            i for i, t in enumerate(layer_types)
            if any(k in str(t).lower() for k in ("global", "full"))
            and not any(k in str(t).lower() for k in ("linear", "delta", "sliding"))
        ]
        if picked:
            return picked
    for field_name in ("sliding_window_pattern", "full_attention_interval", "full_attn_interval"):
        pattern = getattr(config, field_name, None)
        if isinstance(pattern, int) and pattern > 0:
            return [i for i in range(n) if (i + 1) % pattern == 0]
    return list(range(n))


def _log(msg: str, **fields: Any) -> None:
    """One flushed, structured stderr line for the Modal container logs. Worker
    tracing is intentionally verbose (the B200:2/122B tier is hard to observe
    otherwise): lifecycle, token counts, per-stage timings, and — via `_vram` — the
    PER-DEVICE memory split that a single aggregate number would hide."""
    if fields:
        msg = msg + " " + " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[attn] {msg}", flush=True)


def _vram(torch_mod: Any) -> str:
    """Compact PER-CUDA-DEVICE memory line: `gpuN=used/total(alloc rsv peak)` GiB.
    device_map='auto' pipeline-shards weights and tends to PACK GPU 0 (it also
    carries the embeddings, the lm-head window, and this item's captured Q), so a
    single summed number hides exactly the imbalance that OOMs one device while the
    other is half-empty. `used/total` come from the driver (all processes on the
    card); `alloc/rsv/peak` are this process's PyTorch allocator (peak since the
    last `reset_peak_memory_stats`). Best-effort — never raises into a caller."""
    try:
        if torch_mod is None or not torch_mod.cuda.is_available():
            return "cpu"
        gib = float(2**30)
        parts = []
        for i in range(torch_mod.cuda.device_count()):
            free, total = torch_mod.cuda.mem_get_info(i)
            alloc = torch_mod.cuda.memory_allocated(i) / gib
            rsv = torch_mod.cuda.memory_reserved(i) / gib
            peak = torch_mod.cuda.max_memory_allocated(i) / gib
            parts.append(
                f"gpu{i}={(total - free) / gib:.1f}/{total / gib:.0f}"
                f"(a{alloc:.1f} r{rsv:.1f} pk{peak:.1f})"
            )
        return " ".join(parts)
    except Exception as e:  # noqa: BLE001 — diagnostics must never raise
        return f"?{type(e).__name__}"


def _reset_vram_peak(torch_mod: Any) -> None:
    """Zero every device's peak-allocated counter so the next `_vram` peak reading
    reflects only the upcoming stage (e.g. one item's forward)."""
    try:
        if torch_mod is not None and torch_mod.cuda.is_available():
            for i in range(torch_mod.cuda.device_count()):
                torch_mod.cuda.reset_peak_memory_stats(i)
    except Exception:  # noqa: BLE001
        pass


def _gpu_cleanup(torch_mod: Any) -> None:
    """Reclaim GPU memory between items so a previous run/job — or a failed/OOM'd one
    that unwound mid-forward — can't shrink the next forward's headroom. Runs Python
    GC first (drops tensors still held by dead frames / tracebacks), synchronizes so
    freed tensors are actually released, then empties the CUDA allocator cache.
    Best-effort: safe to call with no CUDA (mock/local path) and never raises."""
    import gc
    gc.collect()
    try:
        if torch_mod is not None and torch_mod.cuda.is_available():
            torch_mod.cuda.synchronize()  # let async frees settle before empty_cache
            torch_mod.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — cleanup must never raise
        pass


def _is_oom(exc: BaseException) -> bool:
    """True for a (recoverable) CUDA out-of-memory error — either torch's dedicated
    `OutOfMemoryError` or a plain RuntimeError whose message says so. A caught CUDA
    OOM leaves the context usable, so the worker can drain the allocator and keep
    serving the rest of the queue."""
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


class HFAttentionProvider:
    """Wraps a loaded Gemma (shared by the container) and implements BOTH the
    `Tokenizer` and `AttentionProvider` interfaces so `worker.analyze` reads
    attention indexed by the exact tokenization the forward pass ran on."""

    # `worker.analyze` prefers the batched GPU path (rank_heads / head_token_stats)
    # over the per-row `attention_row` loop when this is set.
    supports_batched = True

    def __init__(self, model: Any, tokenizer: Any, config: Any, global_layers: list[int], torch_mod: Any) -> None:
        self.name = f"hf:{getattr(config, '_name_or_path', 'gemma')}"
        self.model = model
        self.tok = tokenizer
        self.config = config
        self._global = list(global_layers)
        self._torch = torch_mod
        self._offsets: list[tuple[str, int, int]] = []
        self._text: str | None = None
        self._ids: list[int] = []
        # Post-RoPE Q/K per GLOBAL layer, kept ON THE GPU (native dtype) so the
        # scene stats are batched matmuls on the same device as the model — no
        # dense [heads, seq, seq] is ever stored (we read only scene columns + the
        # per-row softmax normalizer). Freed per item in `release()`.
        self._q: dict[int, Any] = {}   # layer -> tensor [heads, n_rows, d] (scored rows only)
        self._k: dict[int, Any] = {}   # layer -> tensor [kv_heads, seq, d] (FULL: causal normalizer)
        # Absolute query positions whose Q we captured (sorted), i.e. the row axis of
        # every `self._q[layer]`. None => full capture (row == absolute position).
        # `_rows_for` maps an absolute query position to its row via searchsorted; a
        # per-device tensor cache avoids rebuilding it for every layer/head.
        self._q_rows: np.ndarray | None = None
        self._q_rows_dev: dict[Any, Any] = {}
        self._scaling: float = 1.0
        self._n_rep: int = 1
        self.logprobs: np.ndarray | None = None

    # --- Tokenizer interface --------------------------------------------------
    def encode_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        if text != self._text:
            self._tokenize(text)
        return self._offsets

    def _tokenize(self, text: str) -> None:
        # Keep offsets 1:1 with `input_ids` (no filtering) — the offset-list
        # index MUST equal the token position in the forward pass, or
        # `attention_row(layer, head, i)` reads the wrong row.
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        self._ids = list(enc["input_ids"])
        # Token display text is the SOURCE SLICE `text[s:e]`, NOT `decode([id])`:
        # decoding a single id in isolation drops SentencePiece leading-space
        # markers, so prose fuses / looks scrambled (dense JSON survives). The
        # offset-mapping slice reproduces the sequence verbatim when concatenated.
        self._offsets = [
            (text[int(s):int(e)], int(s), int(e))
            for (s, e) in enc["offset_mapping"]
        ]
        self._text = text

    # --- AttentionProvider interface -----------------------------------------
    def global_attention_layers(self) -> list[int]:
        return list(self._global)

    def num_heads(self) -> int:
        return int(getattr(self.config, "num_attention_heads", 0))

    def _rows_for(self, abs_positions: Any, dev: Any) -> Any:
        """Map absolute query positions -> row indices into the captured Q on `dev`.
        When only the scored rows were captured (`self._q_rows` set), the Q row axis
        is a compact subset, so an absolute position must be translated to its row.
        searchsorted is EXACT here because every scored position is in `self._q_rows`
        (both the capture set and the scoring set come from the same
        `worker.select_query_indices`). Full capture (`_q_rows is None`) -> identity
        (row == absolute position). Cached per device (device_map shards layers)."""
        if self._q_rows is None:
            return abs_positions
        torch = self._torch
        cached = self._q_rows_dev.get(dev)
        if cached is None:
            cached = torch.as_tensor(self._q_rows, device=dev, dtype=torch.long)
            self._q_rows_dev[dev] = cached
        return torch.searchsorted(cached, abs_positions)

    def attention_row(self, layer: int, head: int, query_i: int) -> np.ndarray:
        """Recompute post-softmax causal attention for ONE query from the cached
        Q/K: softmax((q_i . K[:i+1]) * scaling). GQA-aware (query head -> kv head
        head // n_rep). Kept for the per-row fallback; the batched GPU methods
        below are what `worker.analyze` normally uses. `query_i` is an ABSOLUTE
        position; map it to its captured Q row (K is full, so its slice stays
        absolute)."""
        torch = self._torch
        row = query_i if self._q_rows is None else int(np.searchsorted(self._q_rows, query_i))
        qi = self._q[layer][head, row, :].float()                   # [d]
        kk = self._k[layer][head // self._n_rep, : query_i + 1, :].float()  # [i+1, d]
        scores = (kk @ qi) * self._scaling                          # [i+1]
        row_p = torch.softmax(scores, dim=-1)
        return row_p.detach().cpu().numpy()

    # --- batched GPU scene stats (the fast path) ------------------------------
    def _full_row_norm(self, qm: Any, kf: Any, q_abs: Any, want_row: bool = False) -> tuple[Any, Any, Any]:
        """Full causal-row softmax normalizer `logZ` and full-row entropy `H_full`
        for a (broadcastable) batch of heads. These are GROUP-INDEPENDENT — shared
        by every column group (scene, to-place) for the same head/query — so we
        compute them once. qm: [..., Nq, d], kf: [..., seq, d], q_abs: [Nq].

        AGGREGATION EXPANSION — the free hook: with `want_row=True` (single-head use
        in head_token_stats) the ALREADY-materialized renormalized `row`
        [..., Nq, seq] is also returned so the caller can reduce it onto any number
        of whole-row segment GROUPS (regions, word-types, …) via `_row_reduce`. No
        second matmul, and since `row` is 0 beyond each query's causal horizon those
        masses respect causality with no extra cols<=i logic. Returns
        (logZ, h_full, row); `row` is None when `want_row` is False."""
        torch = self._torch
        seq = kf.shape[-2]
        full = torch.matmul(qm, kf.transpose(-1, -2)) * self._scaling  # [..., Nq, seq]
        causal = torch.arange(seq, device=full.device) > q_abs.unsqueeze(-1)  # mask keys j>i
        full_sm = full.masked_fill(causal, torch.finfo(full.dtype).min)
        logZ = torch.logsumexp(full_sm, dim=-1)                        # [..., Nq]
        row = torch.exp(full_sm - logZ.unsqueeze(-1))                  # 0 at masked
        # H_full = logZ - Σ_j P_j·score_j  (masked j contribute 0; avoid -inf*0)
        h_full = logZ - (row * full.masked_fill(causal, 0.0)).sum(-1)  # [..., Nq]
        return logZ, h_full, (row if want_row else None)

    def _row_reduce(self, row: Any, segs: list[Any]) -> Any | None:
        """Sum the full renormalized `row` onto each ABSOLUTE-column segment →
        [..., Nq, len(segs)] fp32 mass fractions. Segments are disjoint (context
        regions; word/token classes), so a plain per-segment sum is correct; `row`
        is 0 beyond each query's causal horizon, so the masses already respect
        causality. Upcast each selected slice to fp32 BEFORE summing: `row` is bf16
        (the Q·Kᵀ matmul output isn't upcast here), so a bf16 sum over thousands of
        keys both loses precision and can't cross to numpy. An empty segment → zeros."""
        if not segs:
            return None
        torch = self._torch
        out = []
        for ix in segs:
            if ix.numel() == 0:
                out.append(torch.zeros(row.shape[:-1], dtype=torch.float32, device=row.device))
            else:
                out.append(row.index_select(-1, ix).float().sum(-1))
        return torch.stack(out, dim=-1)

    def _col_stats(self, qm: Any, kf: Any, logZ: Any, h_full: Any, q_abs: Any, cols: Any) -> tuple[Any, Any, Any]:
        """Attention onto ONE column group (scene or to-place), reusing the full
        row's `logZ`/`h_full`. Columns are all < completion start (always causally
        visible), so no masking needed. Returns (c, dist, entropy_ratio) — exactly
        stats.scene_scale / scene_distribution / entropy_ratio, vectorized."""
        torch = self._torch
        kc = kf.index_select(-2, cols)                                 # [..., S, d]
        # Narrow scene-column score: upcast to fp32 for the softmax only (the big
        # bf16 Q/K matmul already accumulated in fp32 on the tensor cores).
        scores = torch.matmul(qm, kc.transpose(-1, -2)).float() * self._scaling  # [..., Nq, S]
        p = torch.exp(scores - logZ.unsqueeze(-1))                     # [..., Nq, S]
        c = p.sum(-1)                                                  # [..., Nq]
        dist = p / c.clamp_min(1e-20).unsqueeze(-1)
        h_grp = -(torch.where(dist > 0, dist * torch.log(dist.clamp_min(1e-20)), torch.zeros_like(dist))).sum(-1)
        n_i = (q_abs + 1).to(p.dtype)
        log_n = torch.log(n_i.clamp_min(2.0))
        log_s = float(np.log(max(int(cols.shape[0]), 2)))
        ratio = (h_full / log_n) / (h_grp / log_s).clamp_min(1e-20)
        ratio = torch.where((c > 0) & (h_grp > 0) & (n_i > 1), ratio, torch.zeros_like(ratio))
        return c, dist, ratio

    def rank_heads(self, scene_cols: np.ndarray, sampled: np.ndarray) -> list[dict[str, Any]]:
        """Mean scene scale for EVERY (global layer, head) over a sample of query
        steps — the across-heads/depths grid — computed as per-layer batched GPU
        matmuls. Sorted high->low. (Ranking is by SCENE attention; the same
        selected heads then also get a to-place readout in head_token_stats.)"""
        torch = self._torch
        if scene_cols.size == 0 or sampled.size == 0:
            return []
        ranked: list[dict[str, Any]] = []
        # device_map="auto" shards layers across GPUs — index tensors MUST live on
        # THIS layer's device. scene_cols/sampled are identical for every layer, so
        # cache them per device (a handful of GPUs) instead of rebuilding per layer.
        idx_cache: dict[Any, tuple[Any, Any, Any]] = {}

        def _idx_for(dev: Any) -> tuple[Any, Any, Any]:
            cached = idx_cache.get(dev)
            if cached is None:
                sq_abs = torch.as_tensor(sampled, device=dev, dtype=torch.long)
                # sq_abs: absolute positions (causal mask vs full K); sq_row: their
                # rows in the compact captured Q.
                cached = (torch.as_tensor(scene_cols, device=dev, dtype=torch.long),
                          sq_abs, self._rows_for(sq_abs, dev))
                idx_cache[dev] = cached
            return cached

        # Keep each layer's per-head mean scale ON GPU during the loop so the layer
        # matmuls pipeline (no blocking sync between layers), then transfer in ONE
        # pass — the first .cpu() drains all the queued layer work rather than a
        # sync-per-layer stall.
        layer_means: list[tuple[int, int, Any]] = []  # (layer, n_heads, mean [H] on GPU)
        with torch.inference_mode():
            for layer in self._global:
                # Keep Q/K in native bf16; _full_row_norm / _col_stats upcast the
                # (small) scores to fp32 internally.
                q = self._q[layer]                                    # [H, n_rows, d]
                k = self._k[layer]                                    # [Hkv, seq, d]
                sc, sq, sq_row = _idx_for(q.device)
                kf = k.repeat_interleave(self._n_rep, dim=0) if self._n_rep > 1 else k  # [H, seq, d]
                qm = q.index_select(1, sq_row)                        # [H, Ns, d]
                logZ, h_full, _ = self._full_row_norm(qm, kf, sq)    # sq ABSOLUTE: mask vs full K
                c, _, _ = self._col_stats(qm, kf, logZ, h_full, sq, sc)  # c: [H, Ns]
                layer_means.append((int(layer), int(q.shape[0]), c.mean(-1)))  # [H] on GPU
        for layer, n_heads, means_t in layer_means:
            mean_scale = means_t.detach().to("cpu").numpy()           # [H]
            for head in range(n_heads):
                ranked.append({"layer": layer, "head": int(head), "mean_scale": float(mean_scale[head])})
        ranked.sort(key=lambda h: h["mean_scale"], reverse=True)
        return ranked

    def head_token_stats(
        self, selected: list[dict[str, Any]], query_indices: np.ndarray,
        group_plans: dict[str, dict[str, Any]], whole_plan: dict[str, Any] | None = None,
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dict[str, Any] | None]:
        """For each selected (layer, head) g and query token q, and each column
        GROUP (e.g. scene / to_place): the scene scale, entropy ratio, and — reduced
        ON-DEVICE onto the group's semantic entities + attribute spans — the
        per-entity and per-component scores.

        The dense [Nq, S] renormalized distribution is produced and consumed on the
        GPU: only the (far smaller) [G, Nq, E] / [G, Nq, C] reduced scores cross to
        the host (E, C are tens; S is thousands), so PCIe traffic + host memory drop
        by ~S/E and the worker no longer re-reduces over S in numpy — it only takes
        the top-k. Each `group_plans[name]` is `{cols [S], ent_local: [idx into
        cols], comp_local: [idx into cols], agg}`, built by the worker from the
        token<->entity membership. The full-row normalizer is computed ONCE per head
        and shared across groups. Returns {name: (scale [G,Nq], entropy_ratio
        [G,Nq], ent_scores [G,Nq,E], comp_scores [G,Nq,C])}."""
        torch = self._torch
        G, Nq = len(selected), int(query_indices.size)
        out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
            name: (np.zeros((G, Nq), np.float32), np.zeros((G, Nq), np.float32),
                   np.zeros((G, Nq, len(plan["ent_local"])), np.float32),
                   np.zeros((G, Nq, len(plan["comp_local"])), np.float32))
            for name, plan in group_plans.items()
        }
        # Aggregation expansion: per (head, query) mass reduced off the shared full
        # row (whole-sequence, absolute columns) onto any number of NAMED segment
        # groups (regions, word-types, word-types-within-organized/free text, …) —
        # generic, so new decompositions are just another group. `whole_plan` =
        # {"groups": {name: [abs-col arrays]}, "names": {...}, "region_meta": [...]}.
        wp = whole_plan or {}
        wgroups: dict[str, list[Any]] = wp.get("groups") or {}
        whole_acc = {name: np.zeros((G, Nq, len(segs)), np.float32) for name, segs in wgroups.items()}
        if not G or not Nq:
            return out, None

        def _seg_reduce(dist: Any, seg: list[Any], agg: str) -> Any | None:
            """On-GPU equivalent of the worker's numpy `reduce_over`: reduce a
            [Nq, S] distribution onto len(seg) targets, each a LongTensor of column
            indices into `dist`. Segments may OVERLAP (a token can belong to two
            entities), so each is reduced explicitly rather than via one scatter.
            Aggregation matches the worker: max (default) / sum / mean."""
            if not seg:
                return None
            reduced = []
            for ix in seg:
                if ix.numel() == 0:  # entity with no visible column — score 0
                    reduced.append(torch.zeros(dist.shape[0], dtype=dist.dtype, device=dist.device))
                    continue
                sub = dist.index_select(-1, ix)                        # [Nq, |ix|]
                red = sub.sum(-1) if agg == "sum" else sub.mean(-1) if agg == "mean" else sub.amax(-1)
                reduced.append(red)
            return torch.stack(reduced, dim=-1)                        # [Nq, len(seg)]

        # device_map="auto" shards layers across GPUs, so the query/column index
        # tensors must match EACH layer's device. Cache them (+ the per-entity /
        # per-component index lists) per device — a handful of GPUs — rather than
        # rebuilding for every head.
        idx_cache: dict[Any, tuple[Any, Any, dict[str, Any], dict[str, list[Any]]]] = {}

        def _idx_for(dev: Any) -> tuple[Any, Any, dict[str, Any], dict[str, list[Any]]]:
            cached = idx_cache.get(dev)
            if cached is None:
                # qi: absolute query positions (causal mask vs full K); qi_row: their
                # rows in the compact captured Q.
                qi = torch.as_tensor(query_indices, device=dev, dtype=torch.long)
                grp: dict[str, Any] = {}
                for name, plan in group_plans.items():
                    cols_t = torch.as_tensor(np.asarray(plan["cols"]), device=dev, dtype=torch.long)
                    ent_t = [torch.as_tensor(a, device=dev, dtype=torch.long) for a in plan["ent_local"]]
                    comp_t = [torch.as_tensor(a, device=dev, dtype=torch.long) for a in plan["comp_local"]]
                    grp[name] = (cols_t, ent_t, comp_t, plan.get("agg", "max"))
                # Whole-row segment groups (ABSOLUTE columns, so no local remap).
                whole_t = {gname: [torch.as_tensor(np.asarray(a, dtype=np.int64), device=dev, dtype=torch.long)
                                   for a in segs]
                           for gname, segs in wgroups.items()}
                cached = (qi, self._rows_for(qi, dev), grp, whole_t)
                idx_cache[dev] = cached
            return cached

        # Accumulate every (head, group) reduced result ON GPU during the loop so
        # the per-head matmuls pipeline (no blocking sync between heads), then do a
        # SINGLE host transfer pass — the first .cpu() drains all the queued work
        # instead of ~5 blocking .cpu() PER head (the idle-GPU stall that made a
        # cheap forward, e.g. gemma, spend most of its wall time serializing on the
        # host). The retained per-head tensors are tiny ([Nq], [Nq,E], [Nq,C]); the
        # big [Nq,S] dist is consumed inside the iteration and never held.
        whole_g: dict[str, list[Any]] = {gname: [None] * G for gname in wgroups}
        grp_g: dict[str, dict[str, list[Any]]] = {
            name: {"scale": [None] * G, "ratio": [None] * G, "ent": [None] * G, "comp": [None] * G}
            for name in group_plans
        }
        with torch.inference_mode():
            for g, h in enumerate(selected):
                layer, head = int(h["layer"]), int(h["head"])
                q_layer = self._q[layer]
                qi, qi_row, grp, whole_t = _idx_for(q_layer.device)
                qm = q_layer[head].index_select(0, qi_row)             # [Nq, d] (bf16)
                kf = self._k[layer][head // self._n_rep]               # [seq, d] (bf16)
                logZ, h_full, row = self._full_row_norm(qm, kf, qi, want_row=bool(whole_t))
                if row is not None:
                    for gname, segs_t in whole_t.items():              # whole-row group masses
                        red = self._row_reduce(row, segs_t)
                        if red is not None:
                            whole_g[gname][g] = red
                for name, (cols_t, ent_t, comp_t, agg) in grp.items():
                    if cols_t.numel() == 0:
                        continue
                    c, dist, ratio = self._col_stats(qm, kf, logZ, h_full, qi, cols_t)
                    grp_g[name]["scale"][g] = c
                    grp_g[name]["ratio"][g] = ratio
                    grp_g[name]["ent"][g] = _seg_reduce(dist, ent_t, agg)
                    grp_g[name]["comp"][g] = _seg_reduce(dist, comp_t, agg)
        # One host transfer pass (the first .cpu() syncs once; the rest are ready).
        for gname, per_head in whole_g.items():
            for g, t in enumerate(per_head):
                if t is not None:
                    whole_acc[gname][g] = t.detach().to("cpu").numpy()
        for name, cols in grp_g.items():
            sc_a, ra_a, ent_a, comp_a = out[name]
            for g in range(G):
                if cols["scale"][g] is not None:
                    sc_a[g] = cols["scale"][g].detach().to("cpu").numpy()
                if cols["ratio"][g] is not None:
                    ra_a[g] = cols["ratio"][g].detach().to("cpu").numpy()
                if cols["ent"][g] is not None:
                    ent_a[g] = cols["ent"][g].detach().to("cpu").numpy()
                if cols["comp"][g] is not None:
                    comp_a[g] = cols["comp"][g].detach().to("cpu").numpy()
        whole = {"groups": whole_acc, "names": wp.get("names") or {},
                 "region_meta": wp.get("region_meta") or []} if whole_acc else None
        return out, whole

    def _head_dim(self) -> int:
        hd = getattr(self.config, "head_dim", None)
        if hd:
            return int(hd)
        h = int(getattr(self.config, "num_attention_heads", 1)) or 1
        return int(getattr(self.config, "hidden_size", h)) // h

    def release(self) -> None:
        """Drop EVERYTHING this item put on the GPU (cached Q/K + logprobs) and
        reclaim the allocator cache before the next item — called in a `finally` so
        even a failed/OOM'd item can't leave residual allocations that poison the
        next forward's headroom. Also clears the module-global capture in case a
        forward was interrupted mid-flight (it's normally reset in prepare)."""
        global _ACTIVE_CAPTURE
        _ACTIVE_CAPTURE = None
        self._q, self._k = {}, {}
        self._q_rows, self._q_rows_dev = None, {}
        self.logprobs = None
        _gpu_cleanup(self._torch)

    def _logits_keep_kwarg(self) -> str | None:
        """The forward's supported "keep only the last N logit rows" kwarg name (HF
        renamed `num_logits_to_keep` -> `logits_to_keep` around 4.49), or None if the
        model doesn't expose it. Limiting the LM-head projection to a tail window is
        what stops a long trace from materializing the full [seq, vocab] logits."""
        import inspect
        try:
            params = inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):
            return None
        for name in ("logits_to_keep", "num_logits_to_keep"):
            if name in params:
                return name
        return None

    # --- the instrumented forward pass ---------------------------------------
    def prepare(self, full_text: str, capture_rows: list[int] | None = None) -> None:
        """Tokenize `full_text`, run ONE teacher-forced forward on SDPA (no
        `output_attentions`, so no seq^2 score matrix), and cache the GLOBAL
        layers' post-RoPE Q/K (kept on the GPU in the model's bf16) + per-token
        logprobs. The custom `SCENE_ATTN_IMPL` op fills the module-global capture
        buffer during the forward; the scene stats read the cached Q/K later.

        `capture_rows` (the SCORED query positions from
        `worker.select_query_indices`) bounds how much Q we RETAIN: only those rows
        are kept per global layer, not all `seq`. On a long trace that is the
        difference between <1 GiB and ~29 GiB of captured Q — the accumulation that
        OOMs a shard. None keeps every row (mock/back-compat). K is always full.

        The LM head is projected over only the last `LOGITS_KEEP_MAX` positions
        (`logits_to_keep`) so the (diagnostic) logprobs never force a full
        [seq, vocab] fp32 tensor — the allocation that OOMs long traces."""
        global _ACTIVE_CAPTURE
        torch = self._torch
        self.encode_with_offsets(full_text)  # idempotent tokenize (skips if already done)
        seq = len(self._ids)
        # Rows to RETAIN: the scored query positions, clamped to the sequence and
        # sorted (searchsorted in `_rows_for` needs ascending). None -> keep all.
        rows: list[int] | None = None
        if capture_rows is not None:
            rows = sorted({int(r) for r in capture_rows if 0 <= int(r) < seq})
            self._q_rows = np.asarray(rows, dtype=np.int64)
        else:
            self._q_rows = None
        self._q_rows_dev = {}
        input_ids = torch.tensor([self._ids], device=self.model.device)
        # Cap the LM-head projection to the tail; `base` is the absolute position of
        # the first kept logit row (0 when we keep them all).
        kw: dict[str, Any] = {"input_ids": input_ids, "use_cache": False}
        base = 0
        keep_name = self._logits_keep_kwarg()
        if keep_name and seq > LOGITS_KEEP_MAX:
            kw[keep_name] = LOGITS_KEEP_MAX
            base = seq - LOGITS_KEEP_MAX
        # `rows`/`_rows_dev`: the capture op slices Q to these scored rows and caches
        # the per-device index tensor (device_map shards layers across GPUs).
        cap: dict[str, Any] = {
            "global": set(self._global), "q": {}, "k": {}, "scaling": None, "n_rep": 1,
            "rows": rows, "_rows_dev": {},
        }
        _ACTIVE_CAPTURE = cap
        try:
            with torch.inference_mode():
                out = self.model(**kw)
        finally:
            _ACTIVE_CAPTURE = None

        self._n_rep = int(cap["n_rep"]) or 1
        sc = cap["scaling"]
        self._scaling = float(sc) if sc is not None else float(self._head_dim() ** -0.5)
        # Keep captured Q/K ON THE GPU (native dtype) — the scene stats are batched
        # matmuls on-device (rank_heads / head_token_stats). Q is only the scored
        # rows [H, n_rows, d]; K is full [Hkv, seq, d]. Freed per item in release().
        # Drop the batch dim and make contiguous for fast matmuls.
        self._q = {li: t[0].contiguous() for li, t in cap["q"].items()}
        self._k = {li: t[0].contiguous() for li, t in cap["k"].items()}
        self.logprobs = self._token_logprobs(out.logits[0], input_ids[0], base=base)
        del out, cap
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _token_logprobs(self, logits: Any, ids: Any, *, base: int = 0) -> np.ndarray:
        """Teacher-forced per-token logprobs as a FULL-length [seq-1] array (index i =
        logprob of token i+1), computed in position-chunks so we never hold more than
        LOGPROB_CHUNK rows in fp32. `logits` holds ONLY the tail positions
        [base .. seq-1] (a bounded `logits_to_keep` window); entries before `base`
        stay 0 — the diagnostic round-trip only reads the completion tail. Diagnostic
        only; the attention result never depends on these."""
        torch = self._torch
        seq = int(ids.shape[0])
        n = max(seq - 1, 0)
        full = np.zeros(n, dtype=np.float32)
        rows = int(logits.shape[0])                      # kept rows for positions [seq-rows .. seq-1]
        m = min(rows, n - base)                          # positions [base .. base+m-1] predict real tokens
        if m <= 0:
            return full
        buf = torch.empty(m, dtype=torch.float32, device=logits.device)
        tgt = ids[base + 1: base + 1 + m]
        for a in range(0, m, LOGPROB_CHUNK):
            b = min(a + LOGPROB_CHUNK, m)
            lg = logits[a:b].float()                                  # [chunk, vocab]
            lse = torch.logsumexp(lg, dim=-1)                         # [chunk]
            sel = lg[torch.arange(b - a, device=lg.device), tgt[a:b]]
            buf[a:b] = sel - lse
        full[base: base + m] = buf.detach().cpu().numpy()
        return full


def _logprob_roundtrip(provider: HFAttentionProvider, trace: Any) -> dict[str, Any]:
    """Best-effort local-vs-remote per-token logprob comparison over the output
    frame. Diagnostic only — surfaced, never blocking."""
    remote = (trace.remote_logprobs or {}).get("tokens") if trace.remote_logprobs else None
    lp = provider.logprobs
    if not remote or lp is None:
        return {"computed_local": lp is not None, "remote_available": bool(remote)}
    out_off = trace.frames.get("output", {}).get("start", -1)
    if out_off < 0:
        return {"computed_local": True, "remote_available": True, "aligned": 0}
    diffs: list[float] = []
    for rt in remote:
        cs, ce = out_off + rt["start"], out_off + rt["end"]
        for i, (_t, s, e) in enumerate(provider._offsets):
            if s < ce and e > cs and 0 <= i - 1 < len(lp):
                diffs.append(abs(float(lp[i - 1]) - float(rt.get("logprob", 0.0))))
                break
    arr = np.array(diffs) if diffs else np.array([0.0])
    return {
        "computed_local": True, "remote_available": True, "aligned": len(diffs),
        "mean_abs_delta": float(arr.mean()), "max_abs_delta": float(arr.max()),
        "note": "logprobs are diagnostic; small deltas from kernel/precision are expected.",
    }


# --- Modal app (module scope, guarded) ---------------------------------------

app = None

if _HAS_MODAL:
    HF_CACHE = "/root/.cache/huggingface"
    # Ship ONLY the attention subpackage — not the whole repo. `app` is a
    # namespace package (no app/__init__.py), so mounting app/attention alone at
    # /root/app/attention + PYTHONPATH=/root makes `app.attention.*` importable
    # (the worker's imports never reach app.services/api/core). This is this
    # file's own directory, resolved on the client at deploy time.
    _attn_dir = Path(__file__).resolve().parent
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.5.1", "transformers>=4.45", "accelerate>=1.0",
            "numpy>=2.0", "hf_transfer>=0.1.8", "fastapi[standard]",
        )
        .env({
            "HF_HOME": HF_CACHE,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # `/root` makes `app.attention.*` importable; `/root/app/attention`
            # also makes the module importable as top-level `modal_app`, so the
            # container can load the function whether it was deployed as
            # `app.attention.modal_app` (`-m ...`) OR `modal_app` (`modal deploy
            # modal_app.py` from this dir). Both resolve; siblings still import
            # via `app.attention.*`.
            "PYTHONPATH": "/root:/root/app/attention",
            # Reduce allocator fragmentation on the long, variable-length forwards.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        .add_local_dir(
            _attn_dir.as_posix(), "/root/app/attention",
            ignore=["**/__pycache__/**", "**/*.pyc"],
        )
    )
    app = modal.App(MODAL_APP_NAME)
    hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
    hf_secret = modal.Secret.from_name("huggingface-secret")  # exposes HF_TOKEN
    # ---- Durable, Modal-owned queue (survives API-server reloads/restarts) -----
    # A cell is identified by a HASH of (run, slot, full-model-name) computed on the
    # API server; ALL state is keyed by that hash so the server can recover the
    # whole queue after a restart with a single pull. `modal.Queue` is the durable
    # job stream (partitioned per model so each tier's consumer reads its own jobs).
    # There are only TWO Dicts now — one coarse status doc per cell + the lease:
    #
    #   attn_state[cell:{hash}]   -> {queued:[ev], running:[ev], done:[ev],
    #                                 errors:{ev:msg}, ident:{run,slot,model_alias,model_id}}
    #   attn_lease[consumer:{model_id}] -> heartbeat ts  (is a GPU consumer alive?)
    #
    # ONE Dict op per Modal round-trip is a ~0.3s RPC (and spikes under load), so the
    # design is deliberately COARSE: exactly ONE compact status doc per cell holds the
    # whole queue state, and the big payloads/results live on the Volume at
    # DETERMINISTIC paths (`jobs/{h}/{ev}` = compute payload). A finished step is split
    # into THREE result blobs so the transport is sized to the need — the frequent poll
    # never moves the big one:
    #   cell/{h}/{ev}.meta.json    tiny  {stamp, input_key, prompt_version, req}
    #   cell/{h}/{ev}.compact.json small  the projected `compact` view — the poll's payload
    #   cell/{h}/{ev}.full.json    big   raw per-token/per-head result — STREAMED on demand
    # A status poll is a single `Dict.get`; result-fetch reads meta+compact (small) with
    # NO Dict lookups, no per-step Dict churn, no ack write-backs; the multi-hundred-MB
    # full result only streams over `/blob` when a user opens a step's token/present
    # detail. The worker owns state writes; the server is read-only on Modal (except
    # enqueuing) and dedups via its own disk.
    attn_state = modal.Dict.from_name("starshot-attn-state", create_if_missing=True)
    attn_lease = modal.Dict.from_name("starshot-attn-lease", create_if_missing=True)
    job_queue = modal.Queue.from_name("starshot-attn-jobqueue", create_if_missing=True)
    attn_vol = modal.Volume.from_name("starshot-attn-blobs", create_if_missing=True)
    ATTN_BLOBS = "/attn-blobs"  # mount point for attn_vol on web + workers

    # A GPU consumer, once spawned, streams jobs until the queue is idle for this
    # long, then exits (container stays warm for `scaledown_window`, so re-spawn is
    # cheap — no model reload). On exit it CLEARS its lease so the next enqueue/poll
    # re-spawns immediately; the TTL is only a fallback for a hard crash (no clean
    # exit). It must exceed the WORST-CASE COLD LOAD, because the lease heartbeat
    # thread only starts once `consume` runs — i.e. AFTER `@modal.enter` finishes
    # loading the model. Until then just the single optimistic pre-spawn beat holds
    # the lease. The 122B MoE streamed onto B200:2 from the Volume can take several
    # minutes, so a 300s TTL would lapse mid-load and a concurrent `/pull` reaper
    # would spawn a SECOND B200 cold load (serialized by max_containers=1, but a
    # wasteful ~5min reload). 900s comfortably covers it; clean idle exits still
    # clear the lease instantly, so this only delays recovery from a true hard crash.
    _CONSUMER_IDLE_S = 20.0
    _LEASE_TTL_S = 900.0
    _RUNNING_GRACE_S = 45.0
    # A single forward (cold model load + a long trace) can run for minutes with the
    # main thread blocked in CUDA — longer than a between-jobs beat can cover. So a
    # background thread beats the lease on this interval for the consumer's whole life,
    # guaranteeing the lease never lapses mid-forward and the reaper can't spawn a
    # SECOND consumer (which — even serialized by max_containers=1 — would redo work).
    _LEASE_BEAT_INTERVAL_S = 60.0

    def _part(model_id: str) -> str:
        """modal.Queue partition key: one FIFO stream per model (sanitized)."""
        return (re.sub(r"[^A-Za-z0-9_.-]", "-", model_id or "")[:60]) or "default"

    def _jk(h: str, ev: int) -> str:
        return f"{h}:{int(ev)}"

    def _cell_state(h: str) -> dict[str, Any]:
        st = attn_state.get(f"cell:{h}")
        if not st:
            return {"queued": [], "running": [], "done": [], "errors": {}, "ident": {}}
        for k in ("queued", "running", "done"):
            st.setdefault(k, [])
        st.setdefault("errors", {})
        st.setdefault("ident", {})
        return st

    def _cell_save(h: str, st: dict[str, Any]) -> None:
        attn_state[f"cell:{h}"] = st

    def _list_add(st: dict[str, Any], name: str, ev: int) -> None:
        st[name] = sorted(set(st.get(name, [])) | {int(ev)})

    def _list_remove(st: dict[str, Any], name: str, ev: int) -> None:
        st[name] = [x for x in st.get(name, []) if int(x) != int(ev)]

    def _stamp(h: str, ev: int, input_key: str | None, prompt_version: str | None) -> str:
        """HMAC binding a result to its cell + step content + prompt version."""
        return ident.stamp(cell_hash=h, event_index=ev, input_key=input_key,
                           prompt_version=prompt_version)

    # --- Volume blob store (big payloads; DETERMINISTIC per (cell, step)) --------
    # Paths are fully determined by (cell_hash, event) — no content-addressing — so a
    # reader knows exactly where a step's payload/result lives WITHOUT any Dict lookup.
    # A finished step is stored as THREE separate blobs so the transport can be sized
    # to the need — the frequent poll never has to move the big one:
    #   .meta.json    tiny  {stamp, input_key, prompt_version, req} — dedup + verify
    #   .compact.json small  the projected `compact` view — what `/results` returns
    #   .full.json    big   the raw per-token/per-head result — STREAMED on demand only
    def _meta_path(h: str, ev: int) -> str:
        """Tiny freshness/verify sidecar: `{stamp, input_key, prompt_version, req}`."""
        return f"{ATTN_BLOBS}/cell/{h}/{int(ev)}.meta.json"

    def _compact_path(h: str, ev: int) -> str:
        """Small projected `compact` view (heatmap + aggregates); the poll's payload."""
        return f"{ATTN_BLOBS}/cell/{h}/{int(ev)}.compact.json"

    def _full_path(h: str, ev: int) -> str:
        """The big raw result dict — stored UNWRAPPED so it can be streamed verbatim
        (no parse/re-serialize on the web tier) for the on-demand token/present pull."""
        return f"{ATTN_BLOBS}/cell/{h}/{int(ev)}.full.json"

    def _job_blob_path(h: str, ev: int) -> str:
        """Compute payload the worker reads: `{compute_item, input_key, prompt_version}`."""
        return f"{ATTN_BLOBS}/jobs/{h}/{int(ev)}.json"

    def _blob_write(path: str, obj: Any) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj), encoding="utf-8")

    def _blob_read(path: str) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def _blob_unlink(path: str | None) -> None:
        if not path:
            return
        try:
            Path(path).unlink()
        except OSError:
            pass

    def _vol_commit() -> None:
        attn_vol.commit()  # make writes visible to the other container

    def _vol_reload() -> None:
        try:
            attn_vol.reload()  # see the latest committed blobs
        except Exception:  # noqa: BLE001 — a failed reload just keeps the current view
            pass

    # A Volume reload is a network round-trip (100s ms–seconds), so we never sync on
    # every read: `/results` READS FIRST and only reloads on a genuine miss — and even
    # then at most once per interval PER CONTAINER. Because a blob is always committed
    # BEFORE its `done` pointer is published, a read miss just means our local view is
    # stale, and one recent reload refreshes it for every concurrent request on this
    # container. A benign race (two threads reload at once) only costs one extra sync.
    _RELOAD_MIN_INTERVAL_S = 1.0
    _reload_at = {"ts": 0.0}

    def _vol_reload_throttled() -> bool:
        """Reload at most once per interval; return True if a reload actually ran."""
        now = time.time()
        if now - _reload_at["ts"] < _RELOAD_MIN_INTERVAL_S:
            return False  # a very recent reload already refreshed this container's view
        _reload_at["ts"] = now
        _vol_reload()
        return True

    def _vol_rel(path: str) -> str:
        """Volume-root-relative path for read_file (strip the mount prefix)."""
        pref = ATTN_BLOBS.rstrip("/") + "/"
        return path[len(pref):] if path.startswith(pref) else path.lstrip("/")

    def _blob_pull(path: str) -> Any:
        """TARGETED download of ONE committed blob straight from the Volume backend
        (VolumeGetFile2) — cost is O(1) in the file, INDEPENDENT of how large the
        volume grows, and it always sees the latest commit, so there's no
        whole-Volume `reload` and no staleness window. Falls back to the mounted view
        (then a single throttled reload) only if the targeted read is unavailable.
        Returns None when the blob isn't committed yet (the caller retries)."""
        try:
            buf = bytearray()
            for chunk in attn_vol.read_file(_vol_rel(path)):  # blocking sync iterator
                buf += chunk
            return json.loads(bytes(buf).decode("utf-8"))
        except FileNotFoundError:
            return None  # not committed into the backend yet → retry next fetch
        except Exception:  # noqa: BLE001 — targeted read unavailable; fall back to mount
            pass
        try:
            return _blob_read(path)  # mounted view — fast when already fresh
        except FileNotFoundError:
            if _vol_reload_throttled():  # last resort: refresh the whole view once
                try:
                    return _blob_read(path)
                except Exception:  # noqa: BLE001
                    return None
            return None
        except Exception:  # noqa: BLE001
            return None

    def _lease_beat(model_id: str) -> None:
        attn_lease[f"consumer:{model_id}"] = time.time()

    def _lease_alive(model_id: str) -> bool:
        return (time.time() - float(attn_lease.get(f"consumer:{model_id}", 0.0))) < _LEASE_TTL_S

    def _lease_clear(model_id: str) -> None:
        """Relinquish the lease on consumer exit so the next enqueue/poll re-spawns
        one right away (rather than waiting out the crash-fallback TTL)."""
        attn_lease.pop(f"consumer:{model_id}", None)

    def _ensure_consumer(model_id: str) -> None:
        """Spawn a GPU consumer for this model if none is heartbeating."""
        if _lease_alive(model_id):
            return
        spec = reg.resolve_open_model(model_id)
        Worker = _WORKER_BY_GPU.get(spec.gpu) if spec else None
        if Worker is None:
            return
        _lease_beat(model_id)  # optimistic — collapses a burst of enqueues to one spawn
        Worker(model_id=spec.hf_path).consume.spawn(model_id)

    def _mark_running(h: str, ev: int) -> None:
        """Flip a step queued -> running (one status-doc RMW). Re-reads the doc fresh
        so a concurrent enqueue/publish on the SAME cell isn't clobbered by a stale
        in-memory copy (there's no CAS on `modal.Dict`, so every writer re-reads to
        keep the write window as short as possible)."""
        st = _cell_state(h)
        _list_remove(st, "queued", ev)
        _list_add(st, "running", ev)
        st.get("errors", {}).pop(str(ev), None)
        _cell_save(h, st)

    def _mark_error(h: str, ev: int, error: str, input_key: str | None = None) -> None:
        """Record a failed step in the status doc (one Dict RMW) and drop its compute
        payload blob (errored steps are never reaped, so the payload is dead weight).
        No result blob is produced, so there's nothing to commit. The failure is stored
        WITH the content key it failed on (`{input_key, msg}`) so it's a diagnostic of a
        SPECIFIC request, not a durable verdict: the server hides it once the step's
        content changes, and a redeploy clears it wholesale (see `web`)."""
        st = _cell_state(h)
        _list_remove(st, "running", ev)
        _list_remove(st, "queued", ev)
        st.setdefault("errors", {})[str(ev)] = {"input_key": input_key, "msg": error}
        _cell_save(h, st)
        _blob_unlink(_job_blob_path(h, ev))

    def _stage_write(h: str, ev: int, input_key: str | None, prompt_version: str | None,
                     req: str, result: dict[str, Any]) -> dict[str, Any]:
        """Stage a finished step to the local Volume view (NOT yet committed/published)
        as THREE blobs so the transport can be sized to the need:

          * `.full.json`    the raw result dict, UNWRAPPED so it streams verbatim on
                            the rare on-demand token/present pull (never on the poll);
          * `.compact.json` the projected `compact` view — small, what `/results`
                            returns every poll, built HERE (the worker already holds
                            the full result) so the server never parses it;
          * `.meta.json`    the HMAC stamp (server verifies with no Dict lookup) + the
                            `req` token (a later life / reap / unchanged re-enqueue
                            adopts this exact request instead of recomputing).

        The big `.full.json` is written FIRST and the tiny `.meta.json` LAST, so a
        committed `.meta` always implies a complete `.full`/`.compact`. The step stays
        queued until the batched `_vol_commit` + `_publish_batch` makes them durable —
        a crash before that leaves it queued (reap re-streams) rather than 'done'
        pointing at a lost blob."""
        _blob_write(_full_path(h, int(ev)), result)
        _blob_write(_compact_path(h, int(ev)), derive.build_compact(result))
        _blob_write(_meta_path(h, int(ev)), {
            "stamp": _stamp(h, ev, input_key, prompt_version),
            "input_key": input_key,
            "prompt_version": prompt_version,
            "req": req,
        })
        return {"h": h, "ev": int(ev)}

    def _publish_batch(recs: list[dict[str, Any]]) -> None:
        """Mark a committed batch of results 'done' in ONE status-doc RMW per cell
        (grouped), then drop their now-consumed compute-payload blobs. Called only
        AFTER `_vol_commit` has persisted the staged result blobs, so a step marked
        done ALWAYS has a readable result blob. If a concurrent enqueue clobbers this
        'done' write, the reaper re-streams the step and the worker restores it via
        the result-blob-exists self-heal (see `_worker_consume`)."""
        by_h: dict[str, list[int]] = {}
        for rec in recs:
            by_h.setdefault(rec["h"], []).append(int(rec["ev"]))
        for h, evs in by_h.items():
            st = _cell_state(h)
            for ev in evs:
                _list_remove(st, "queued", ev)
                _list_remove(st, "running", ev)
                st.get("errors", {}).pop(str(ev), None)
                _list_add(st, "done", ev)
            _cell_save(h, st)
            for ev in evs:
                _blob_unlink(_job_blob_path(h, ev))  # payload consumed; result blob stays

    def _remark_done(h: str, ev: int) -> None:
        """Restore a 'done' marker a concurrent RMW clobbered — used when the reaper
        re-streams a step whose result blob already exists on the Volume. One RMW."""
        st = _cell_state(h)
        _list_remove(st, "queued", ev)
        _list_remove(st, "running", ev)
        st.get("errors", {}).pop(str(ev), None)  # a result exists now → drop any stale failure
        _list_add(st, "done", ev)
        _cell_save(h, st)

    def _reap_stale(st: dict[str, Any], h: str) -> None:
        """Recover from a dead consumer (crash / redeploy): if the lease has expired
        while steps are still outstanding, re-stream them and re-spawn a consumer.
        Operates on an ALREADY-fetched status doc (no extra Dict.get), and only touches
        the lease when there's outstanding work. `ensure_consumer` beats the lease, so
        this fires once per death — not on every poll. The worker dedups a re-streamed
        step it already staged via its in-memory `seen` set, and restores a step it
        finished in a PRIOR life via the result-blob-exists check, so a re-stream is
        always harmless (at worst one duplicate forward)."""
        outstanding = list(st.get("queued", [])) + list(st.get("running", []))
        model_id = (st.get("ident") or {}).get("model_id")
        if not outstanding or not model_id or _lease_alive(model_id):
            return
        for ev in outstanding:
            job_queue.put({"cell_hash": h, "event_index": int(ev), "model_id": model_id},
                          partition=_part(model_id))
        _ensure_consumer(model_id)

    _COMMIT_EVERY = 4          # commit at most this many computes between Volume syncs
    _COMMIT_INTERVAL_S = 1.5   # …or this long, so results stay fresh during slow bursts
    _JOB_READ_RETRIES = 6      # payload targeted-read attempts before giving up (backend lag)
    _JOB_READ_BACKOFF_S = 0.5  # …spaced this far apart (~3s total) — enqueue commits before streaming

    def _worker_consume(self, model_id: str) -> str:
        """Long-running GPU consumer: stream jobs off the durable queue and compute
        each (the model stays warm). It's a TRACKED Modal input (not a background
        task), so Modal won't recycle it out from under an in-flight forward — the
        bug behind the old cancellation storm.

        Keeping the GPU FED is the priority here: a Volume `reload`/`commit` costs
        100s of ms–seconds, so doing one per job (as before) left the GPU idling in
        an IO valley between every short forward. Instead we BATCH the syncs —
        results are STAGED to the local Volume view and committed together every few
        jobs (or the moment the queue drains), and we only `reload` when a job's
        export isn't visible yet. Exits when the queue is idle so the container can
        scale down (lease cleared in `finally` → the next enqueue re-spawns)."""
        import queue as _stdq
        import threading

        part = _part(model_id)
        processed = 0
        pending: list[dict[str, Any]] = []   # staged results awaiting a batched commit
        seen: dict[str, str] = {}            # jk -> req this consumer resolved (dedup re-streams)
        last_commit = time.time()
        _gpu_cleanup(self._torch)  # reclaim any GPU a previous consume left on this warm container

        # Background lease heartbeat: beat immediately, then every interval for the
        # consumer's WHOLE life. A single forward (cold load + long trace) blocks the
        # main thread for minutes, but CUDA releases the GIL, so this thread keeps the
        # lease alive — the reaper never mistakes a busy consumer for dead and can't
        # spawn a second one. Runs strictly one forward at a time (single container).
        _stop = threading.Event()

        def _beat_loop() -> None:
            _lease_beat(model_id)
            while not _stop.wait(_LEASE_BEAT_INTERVAL_S):
                try:
                    _lease_beat(model_id)
                except Exception:  # noqa: BLE001 — a missed beat is recovered on the next tick
                    pass

        _beater = threading.Thread(target=_beat_loop, name="attn-lease-beat", daemon=True)

        def _flush() -> None:
            nonlocal pending, last_commit
            if pending:
                _vol_commit()                # ONE commit persists the whole staged batch…
                _publish_batch(pending)      # …THEN flip to done (blobs are now durable)
                pending = []
            last_commit = time.time()

        def _read_job(h: str, ev: int) -> dict[str, Any] | None:
            """The compute payload `{compute_item, input_key, prompt_version}`, read
            from its deterministic Volume path. Enqueue commits it BEFORE streaming the
            ref, so it IS durable by the time we dequeue — but the container's mounted
            view lags the backend, so a plain mount read (`_blob_read`) can miss it and
            wrongly report 'missing compute payload'. Read it TARGETED off the backend
            (`read_file` always sees the latest commit, no whole-Volume reload), which
            also can't discard our staged (uncommitted) result writes. Retry a few
            times for backend-propagation lag; flush once first so the payload we're
            waiting on can't be shadowed by pending local state."""
            path = _job_blob_path(h, ev)
            rel = _vol_rel(path)

            def _pull() -> dict[str, Any] | None:
                try:
                    buf = bytearray()
                    for chunk in attn_vol.read_file(rel):  # blocking sync iterator
                        buf += chunk
                    return json.loads(bytes(buf).decode("utf-8"))
                except FileNotFoundError:
                    return None  # not committed into the backend yet → retry
                except Exception:  # noqa: BLE001 — targeted read unavailable; fall back to mount
                    try:
                        return _blob_read(path)
                    except Exception:  # noqa: BLE001
                        return None

            obj = _pull()
            if obj is not None:
                return obj
            _flush()  # publish staged results so a retry can't be starved by pending state
            for _ in range(_JOB_READ_RETRIES):
                time.sleep(_JOB_READ_BACKOFF_S)
                obj = _pull()
                if obj is not None:
                    return obj
            return None

        def _read_meta(h: str, ev: int) -> dict[str, Any] | None:
            """The committed result's tiny meta sidecar (`{stamp, input_key,
            prompt_version, req}`), or None if the step has no committed result yet.
            Lets the consumer ADOPT an already-computed result — a prior container's
            life, a reap re-stream, or an unchanged re-enqueue — instead of recomputing,
            and the `req` inside says whether it's the SAME request."""
            try:
                return _blob_read(_meta_path(h, ev))
            except Exception:  # noqa: BLE001
                return None

        _beater.start()  # keep the lease warm for the whole drain (esp. long forwards)
        _log("consumer start", model=model_id, part=part)
        try:
            while True:
                # Drain everything immediately available back-to-back; only when the
                # queue is momentarily empty do we flush results + block for more.
                ref = job_queue.get(partition=part, block=False)
                if ref is None:
                    _flush()
                    try:
                        ref = job_queue.get(partition=part, block=True, timeout=_CONSUMER_IDLE_S)
                    except _stdq.Empty:
                        break  # idle → exit
                if not ref:
                    continue
                h = str(ref["cell_hash"])
                ev = int(ref["event_index"])
                jk = _jk(h, ev)
                ref_req = str(ref.get("req") or "")
                if ref_req and seen.get(jk) == ref_req:
                    continue  # this exact request already resolved this life — cheap skip
                job = _read_job(h, ev)
                if job is None:
                    # Payload gone: truly missing, OR a published step whose result is
                    # already committed — restore 'done' (self-heal a clobbered marker)
                    # and remember its req so repeat refs skip cheaply above.
                    meta = _read_meta(h, ev)
                    if meta is not None:
                        _remark_done(h, ev)
                        seen[jk] = str(meta.get("req") or ref_req)
                    else:
                        _log("job payload missing", h=h[:8], ev=ev)
                        _mark_error(h, ev, "missing compute payload")
                    continue
                cur_key = job.get("input_key")
                # The committed payload's req is authoritative: a re-enqueue overwrites
                # the job blob in place, so even a stale queued ref computes the LATEST
                # request for the step rather than a superseded one.
                req = str(job.get("req") or ref_req)
                if seen.get(jk) == req:
                    continue  # already resolved this exact request this life — no recompute
                # Durable cross-life dedup: a committed result for this EXACT request
                # (same token) is ADOPTED with no forward. A different token (raised
                # heads / edited content / force → new token) recomputes and overwrites.
                meta = _read_meta(h, ev)
                if meta is not None and str(meta.get("req") or "") == req:
                    _remark_done(h, ev)
                    seen[jk] = req
                    continue
                _mark_running(h, ev)
                _log("job compute", h=h[:8], ev=ev, req=req[:8])
                try:
                    result = _worker_analyze(self, job["compute_item"])
                    pending.append(_stage_write(
                        h, ev, job.get("input_key"), job.get("prompt_version"), req, result))
                    seen[jk] = req  # staged: never recompute this exact request again
                    processed += 1
                except Exception as e:  # noqa: BLE001 — one bad step (incl. OOM) must not kill the stream
                    _log("job failed", h=h[:8], ev=ev, err=f"{type(e).__name__}: {e}")
                    _mark_error(h, ev, f"{type(e).__name__}: {e}", cur_key)
                if len(pending) >= _COMMIT_EVERY or (time.time() - last_commit) >= _COMMIT_INTERVAL_S:
                    _flush()
        finally:
            # Stop the heartbeat, publish anything still staged, then ALWAYS relinquish
            # the lease — clean idle-exit OR crash — so the next enqueue/poll re-spawns
            # a consumer instead of jobs sitting queued until the TTL fallback expires.
            _stop.set()
            _beater.join(timeout=2.0)
            try:
                _flush()
            finally:
                _gpu_cleanup(self._torch)  # leave the GPU clean for the next consume/model
                _lease_clear(model_id)
        _log("consumer exit", model=model_id, processed=processed)
        return f"drained {processed}"

    # A second image tier for the big models: Blackwell-capable torch (B200) +
    # a transformers new enough for the Qwen3.5 hybrid-MoE arch. Gemma keeps the
    # proven `image` above (H200 is Hopper/sm_90 — its default-CUDA wheel is fine).
    #
    # TWO things MUST be right here or the container dies before the first forward:
    #   1. torch from the CUDA 12.8 index. B200 is Blackwell (sm_100); the default
    #      PyPI wheel is cu126 and ships NO sm_100 kernels, so every CUDA op fails
    #      with "no kernel image is available for execution on the device". Blackwell
    #      kernels exist only in the cu128+ builds (torch >= 2.7). Installed in its
    #      OWN layer so the pytorch index (which also serves torch's nvidia-* deps)
    #      doesn't shadow PyPI for the rest.
    #   2. transformers >= 5.2.0. The `qwen3_5_moe` architecture (Qwen3.5-122B-A10B)
    #      landed in 5.2.0; 4.57.x raises "model type not supported" from its config.
    big_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch>=2.7.0", index_url="https://download.pytorch.org/whl/cu128")
        .pip_install(
            "transformers>=5.2.0", "accelerate>=1.2",
            "numpy>=2.0", "hf_transfer>=0.1.8", "fastapi[standard]",
        )
        .env({
            "HF_HOME": HF_CACHE,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": "/root:/root/app/attention",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        .add_local_dir(_attn_dir.as_posix(), "/root/app/attention", ignore=["**/__pycache__/**", "**/*.pyc"])
    )

    # --- shared worker logic (the per-tier @app.cls below are thin wrappers) ----
    # Modal fixes `gpu` per class, so we define one thin class per GPU tier and
    # share ALL logic here. Every model is driven by its OpenModelSpec (HF path,
    # loader class, thinking delimiters, GPU count -> device_map).
    def _load_hf_model(model_id: str) -> tuple:
        """Load an open-weight model + tokenizer for the capture: register our SDPA
        op, pick the loader class + device_map from the model's spec, and tag each
        decoder block with its index so the op stashes only the (global / gated-)
        attention layers. Returns (torch, config, text_config, global_layers,
        tokenizer, model)."""
        import os

        import torch  # type: ignore
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # type: ignore

        try:  # public top-level export in recent transformers…
            from transformers import AttentionInterface  # type: ignore
        except ImportError:  # …else it lives in modeling_utils
            from transformers.modeling_utils import AttentionInterface  # type: ignore

        # Allow TF32 for any residual fp32 matmuls (the scene-stat matmuls run in
        # bf16 and accumulate in fp32, so this only affects stray fp32 paths).
        torch.backends.cuda.matmul.allow_tf32 = True
        spec = reg.resolve_open_model(model_id)
        token = os.environ.get("HF_TOKEN")  # from `huggingface-secret`
        hf_cache.reload()  # see weights a `prefetch` run committed to the Volume
        AttentionInterface.register(SCENE_ATTN_IMPL, _scene_capture_attention)
        config = AutoConfig.from_pretrained(model_id, token=token)
        # Multimodal (Gemma / Qwen-VL) nest the LM fields under `.text_config`;
        # layer + head counts MUST come from the text config, not the composite.
        text_config = resolve_text_config(config)
        glayers = global_attention_layers(text_config)
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        # 1 GPU -> everything on cuda; multi-GPU tier -> pipeline-shard (device_map=auto).
        device_map = "auto" if (spec and reg.n_gpus(spec.gpu) > 1) else "cuda"
        kw = dict(dtype=torch.bfloat16, attn_implementation=SCENE_ATTN_IMPL, device_map=device_map, token=token)
        # Cap the per-GPU WEIGHT budget so a fixed headroom stays free on every
        # device for the forward's activations + lm-head window + captured Q (see
        # _WEIGHT_HEADROOM_GIB) — otherwise auto packs GPU 0 and a long trace OOMs it.
        if device_map == "auto" and torch.cuda.is_available():
            gib = float(2**30)
            budget = {}
            for i in range(torch.cuda.device_count()):
                _, total = torch.cuda.mem_get_info(i)
                budget[i] = f"{max(total / gib - _WEIGHT_HEADROOM_GIB, 8.0):.0f}GiB"
            kw["max_memory"] = budget
            _log("device_map=auto weight budget", reserve_gib=_WEIGHT_HEADROOM_GIB, budget=budget)
        _log("loading model", model=model_id, loader=(spec.loader if spec else "causal_lm"),
             device_map=device_map, glayers=len(glayers))
        # VL checkpoints (Qwen3_5MoeForConditionalGeneration, multimodal Gemma) may
        # need the image-text-to-text auto class; fall back to causal-LM.
        model = None
        if spec and spec.loader == "image_text_to_text":
            try:
                from transformers import AutoModelForImageTextToText  # type: ignore
                model = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
            except Exception as e:  # noqa: BLE001 — fall back to the causal-LM loader
                _log("image_text_to_text loader failed; trying causal-LM", err=f"{type(e).__name__}: {e}")
                model = None
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
        # Belt-and-suspenders: route every sub-config to our op, then tag each
        # decoder block so the op knows which layer it is (global vs local/linear).
        model.config._attn_implementation = SCENE_ATTN_IMPL
        resolve_text_config(model.config)._attn_implementation = SCENE_ATTN_IMPL
        for i, layer in enumerate(decoder_layers(model)):
            sa = getattr(layer, "self_attn", None)
            if sa is not None:  # hybrid models: DeltaNet blocks have no self_attn
                sa._attn_layer_idx = i
        # Surface the actual shard layout + resident memory so an imbalance is
        # visible in the logs BEFORE the first forward.
        try:
            dm = getattr(model, "hf_device_map", None)
            if dm:
                from collections import Counter
                _log("model loaded (sharded)", model=model_id, glayers=len(glayers),
                     shards=dict(Counter(str(v) for v in dm.values())))
            else:
                _log("model loaded", model=model_id, glayers=len(glayers),
                     device=str(getattr(model, "device", "?")))
        except Exception:  # noqa: BLE001 — logging must never break the load
            pass
        _log(f"resident | {_vram(torch)}")
        return torch, config, text_config, glayers, tokenizer, model

    def _worker_load(self) -> None:
        (self._torch, self.config, self.text_config, self.global_layers,
         self.tokenizer, self.model) = _load_hf_model(self.model_id)
        self.spec = reg.resolve_open_model(self.model_id)

    def _worker_analyze(self, item: dict[str, Any]) -> dict[str, Any]:
        """Compute ONE item: a single teacher-forced forward (SDPA + attention rows
        recomputed on demand — VRAM-safe; the item's Q/K is released before return).
        Runs strictly one-at-a-time on the single GPU container, and reclaims GPU
        memory BEFORE (so a previous run/item can't shrink this forward's headroom)
        and AFTER (so this item can't shrink the next). A CUDA OOM is caught, the
        allocator drained, and a CLEAR non-retryable reason surfaced — the consumer
        keeps serving the rest of the queue instead of dying. Item: {export,
        remote_logprobs?, max_heads?, top_k?, max_query_tokens?}."""
        from app.attention import reconstruct, worker

        torch = self._torch
        spec = getattr(self, "spec", None) or reg.resolve_open_model(self.model_id)
        render = lambda m: self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)  # noqa: E731
        trace, check = reconstruct.build_real_trace(
            item["export"], render,
            thinking_open=(spec.thinking_open if spec else ""),
            thinking_close=(spec.thinking_close if spec else ""),
            remote_logprobs=item.get("remote_logprobs"),
        )
        _gpu_cleanup(torch)  # clean slate — reclaim anything a prior item/run left behind
        _reset_vram_peak(torch)  # so the peak we log reflects THIS item's forward, not history
        provider = HFAttentionProvider(
            self.model, self.tokenizer, self.text_config, self.global_layers, torch)
        ev = item.get("event_index", "?")
        mq = int(item.get("max_query_tokens", 0))
        tier = getattr(spec, "gpu", "?")
        try:
            # The SCORED query rows — the ONLY Q rows we retain during the capture
            # forward. Same helper `analyze` scores with, so the captured rows and
            # the scored positions line up exactly (keeps peak VRAM off the 29 GiB
            # full-Q accumulation on long traces).
            offsets = provider.encode_with_offsets(trace.full_text)
            capture_rows = worker.select_query_indices(offsets, trace, mq)
            _log("analyze start", ev=ev, tier=tier, tokens=len(offsets),
                 scored=len(capture_rows), glayers=len(self.global_layers), mq=mq)
            t0 = time.perf_counter()
            provider.prepare(trace.full_text, capture_rows=capture_rows)  # forward + Q/K capture
            _log(f"forward done | {_vram(torch)}", ev=ev,
                 secs=round(time.perf_counter() - t0, 1), n_rep=provider._n_rep)
            t1 = time.perf_counter()
            result = worker.analyze(
                trace, tokenizer=provider, provider=provider,
                max_heads=item.get("max_heads", 4), top_k=item.get("top_k", 12),
                max_query_tokens=mq,
            )
            _log(f"scored | {_vram(torch)}", ev=ev, secs=round(time.perf_counter() - t1, 1))
            out = result.to_dict()
            meta = out.setdefault("meta", {})
            # NEVER return an empty or mock analysis. A real GPU worker that produced
            # no scored tokens or no attended heads (a step with no scene context, or a
            # degenerate reconstruction) would surface as a blank "done" map that hides
            # the real problem — fail LOUD so the step is marked error and the frontend
            # shows why, instead of committing an empty/fake result. (We do NOT overwrite
            # meta.mock to False here: if the mock path ever leaked onto the GPU worker,
            # that must raise, not be relabelled real.)
            if meta.get("mock"):
                raise RuntimeError(f"ev {ev}: analysis ran on the MOCK path — refusing to return a fake result")
            if not out.get("tokens") or not out.get("selected_heads"):
                raise RuntimeError(
                    f"ev {ev}: empty analysis (scored_tokens={len(out.get('tokens') or [])}, "
                    f"heads={len(out.get('selected_heads') or [])}) — nothing attended; "
                    "refusing to commit an empty result")
            meta["reconstruction"] = check
            meta["hf_path"] = self.model_id
            out["logprob_check"] = _logprob_roundtrip(provider, trace)
            return out
        except Exception as e:  # noqa: BLE001 — enrich a CUDA OOM; re-raise anything else as-is
            if _is_oom(e):
                n_tok = len(getattr(provider, "_ids", []) or [])
                vram = _vram(torch)
                _log("OOM", ev=ev, tier=tier, tokens=n_tok, mem=vram)
                raise RuntimeError(
                    f"OutOfMemoryError: trace too large for the {tier} GPU tier "
                    f"(~{n_tok} tokens); recompute won't help. {type(e).__name__}: {e} "
                    f"[per-gpu: {vram}]"
                ) from e
            _log("analyze error", ev=ev, err=f"{type(e).__name__}: {e}")
            raise
        finally:
            provider.release()  # free cached Q/K + drain the allocator before the next item

    def _worker_analyze_many(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process a batch FIFO inside one spawn — model stays warm, no per-item
        cancel storm from dozens of independent FunctionCalls."""
        out: list[dict[str, Any]] = []
        for i, item in enumerate(items):
            try:
                out.append({"ok": True, "index": i, "result": _worker_analyze(self, item)})
            except Exception as e:  # noqa: BLE001 — one bad step must not abort the batch
                out.append({"ok": False, "index": i, "error": f"{type(e).__name__}: {e}"})
        return out

    def _worker_ping(self) -> str:
        return "ok"

    # Shared @app.cls kwargs. SINGLE container per tier (no `@modal.concurrent` on
    # the class) -> Modal runs spawned `analyze` calls FIFO, never two forwards at
    # once (VRAM-safe). include_source=False: the image already ships app/attention.
    # cpu: give the post-processing real cores. Between forwards the GPU idles while
    # the CPU takes the top-k, builds per-token records, and JSON-serializes the
    # result; the default fractional CPU reservation makes that tail dominate — and
    # with a QUEUE of many requests that idle-GPU tail is paid once PER item, so it
    # bounds throughput. 16 cores lets numpy (top-k / reductions' copies + BLAS
    # threads), json, and GC run without starving, shrinking the idle-GPU gap
    # between queued items (cheap vs. an idle H200/B200). Pair with the batched
    # device->host transfer (one sync/step) so the CPU isn't also blocked on the GPU.
    _WORKER_KW: dict[str, Any] = dict(
        volumes={HF_CACHE: hf_cache, ATTN_BLOBS: attn_vol}, secrets=[hf_secret], timeout=3600,
        max_containers=1, include_source=False, cpu=16.0,
    )

    @app.cls(gpu="H200", image=image, scaledown_window=360, memory=131072, **_WORKER_KW)
    class WorkerH200:
        """Single-H200 tier: Gemma-31b and similar (<=~70 GB bf16). Idle-killed at 6 min."""
        model_id: str = modal.parameter(default=DEFAULT_MODEL_ID)

        @modal.enter()
        def load(self) -> None:
            _worker_load(self)

        @modal.method()
        def analyze(self, item: dict[str, Any]) -> dict[str, Any]:
            return _worker_analyze(self, item)

        @modal.method()
        def analyze_many(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return _worker_analyze_many(self, items)

        @modal.method()
        def consume(self, model_id: str) -> str:
            return _worker_consume(self, model_id)

        @modal.method()
        def ping(self) -> str:
            return _worker_ping(self)

    @app.cls(gpu="B200:2", image=big_image, scaledown_window=600, memory=262144, **_WORKER_KW)
    class WorkerB200x2:
        """Two-B200 tier (device_map=auto) for ~100-130B MoE models like
        Qwen3.5-122B-A10B (~244 GB bf16). Longer idle window (pricier cold load)."""
        model_id: str = modal.parameter(default="Qwen/Qwen3.5-122B-A10B")

        @modal.enter()
        def load(self) -> None:
            _worker_load(self)

        @modal.method()
        def analyze(self, item: dict[str, Any]) -> dict[str, Any]:
            return _worker_analyze(self, item)

        @modal.method()
        def analyze_many(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return _worker_analyze_many(self, items)

        @modal.method()
        def consume(self, model_id: str) -> str:
            return _worker_consume(self, model_id)

        @modal.method()
        def ping(self) -> str:
            return _worker_ping(self)

    # GPU tier -> worker class. The web endpoint routes each model to its tier
    # (from the model's OpenModelSpec.gpu). Add a tier here to support a new GPU size.
    _WORKER_BY_GPU: dict[str, Any] = {"H200": WorkerH200, "B200:2": WorkerB200x2}

    @app.function(image=image, volumes={HF_CACHE: hf_cache}, secrets=[hf_secret], timeout=3600, include_source=False)
    def prefetch(model_id: str = DEFAULT_MODEL_ID) -> str:
        """Download the model weights into the Volume up front, so the first
        real inference doesn't stall on a cold download.

        Run once (CPU, no GPU): `modal run -m app.attention.modal_app::prefetch`
        (optionally `--model-id <hf-id>`).

        Volume best practices applied:
          * `hf_transfer` (enabled via the image env) does fast PARALLEL
            large-file downloads — not many small chunked requests.
          * a SINGLE `hf_cache.commit()` at the end persists the whole snapshot
            in one large commit (never commit per-file)."""
        import os

        from huggingface_hub import snapshot_download  # type: ignore

        path = snapshot_download(model_id, token=os.environ.get("HF_TOKEN"))
        hf_cache.commit()  # one large commit, after the full download
        return path

    # A light FastAPI worker (no GPU/torch) that fronts the GPU class over HTTP.
    # Modal caps EVERY web request at 150s (past that it 303-redirects to a result
    # URL; an exhausted/interrupted redirect chain is what surfaces client-side as
    # "RemoteProtocolError: Server disconnected without sending a response"). A
    # teacher-forced forward (esp. cold model load) routinely exceeds 150s, so we
    # DON'T block the request on it: submit SPAWNS the GPU work (its result is
    # saved in Modal's result store, keyed by call id) and returns immediately; the
    # server then PULLS the result with short GET polls. See Modal's "Request
    # timeouts" guide (spawn + poll is the documented long-job pattern).
    #
    # EXACTLY one web container, ALWAYS ON (`min_containers=1` + `max_containers=1`),
    # made concurrent (`@modal.concurrent`) so a single CPU container serves many
    # simultaneous submit/poll calls with no cold start on the front door. It's
    # CPU-only (no GPU/torch), so keeping it warm is cheap — unlike the H200 worker,
    # which still scales to zero after its idle window.
    web_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("fastapi[standard]", "numpy>=2.0", "httpx")
        # See the GPU image: importable both as `app.attention.modal_app` and as
        # top-level `modal_app`, so the deploy command doesn't matter.
        .env({"PYTHONPATH": "/root:/root/app/attention"})
        .add_local_dir(_attn_dir.as_posix(), "/root/app/attention", ignore=["**/__pycache__/**", "**/*.pyc"])
    )

    # `memory` matters here: this ONE container serves up to `max_inputs`
    # concurrent requests, and /enqueue parses a full export per item. With the
    # default (~128 MiB) a burst of enqueues OOM-kills the container mid-request —
    # the client sees "RemoteProtocolError: Server disconnected without sending a
    # response" and nothing ever queues. Give it real headroom (it's CPU-only, so
    # cheap to keep warm) and cap concurrency to a saner number.
    @app.function(image=web_image, include_source=False, min_containers=1,
                  max_containers=1, memory=4096, volumes={ATTN_BLOBS: attn_vol})
    @modal.concurrent(max_inputs=16)
    @modal.asgi_app()
    def web():
        """Durable, cell-hash-keyed queue API — every route returns fast (no route
        rides Modal's 150s web-request cap) and holds NO in-container background
        work, so recycling the web container can't cancel a GPU job:

          POST /enqueue   {cell_hash, ident, prompt_version, items:[…]} -> stream jobs
          POST /pull      {cell_hash} -> {queued, running, done, errors}  (1 Dict.get, cheap)
          POST /results   {cell_hash, event_indices:[…]} -> {results:[…]}  (targeted Volume reads)
          GET  /queue     ?cell_hash=…  -> {queued, running, done_pending, errors, ident}
          POST /warm      {model_id?}       -> wake the GPU container
          POST /          legacy sync batch spawn ; GET /result/{call_id}

        State is deliberately COARSE: ONE status doc per cell in `attn_state` holds
        queued/running/done/errors, and the big compute payloads + result blobs live
        on `attn_vol` at DETERMINISTIC paths keyed by (cell_hash, event). So `/pull`
        is a single `Dict.get`, `/results` is a targeted blob read with NO Dict lookup
        and NO ack write-back, and the worker owns all status writes. Keyed by the
        server-supplied `cell_hash` = H(run, slot, model, prompt_version). The GPU work
        runs in a long-running `consume` input (see `_worker_consume`), NOT a background
        task here, so it survives web-container recycles."""
        import fastapi

        web_app = fastapi.FastAPI(title="starshot-attention", docs_url="/docs")

        # A fresh web container == a fresh deployment (any consumer container from a
        # previous deploy is already gone), so drop stale consumer leases now. Without
        # this, a lease no live consumer still holds could suppress `_ensure_consumer`
        # until its TTL expires — leaving jobs queued-but-never-dispatched right after
        # a redeploy. Live consumers simply re-beat within one loop iteration.
        try:
            attn_lease.clear()
        except Exception:  # noqa: BLE001 — best-effort; TTL fallback still recovers
            pass

        # RESET PENDING COMPUTE ON RESTART. A fresh deployment has NO live consumers
        # (a prior deploy's are gone), so every PENDING job is orphaned: drop the whole
        # durable queue so a prior run's stale refs can't sit ahead of new work in the
        # shared per-model FIFO. Committed result blobs (on the Volume) are untouched.
        try:
            job_queue.clear(all=True)
        except Exception:  # noqa: BLE001 — best-effort; orphaned refs also self-heal via dedup on drain
            pass

        # Per cell, wipe the TRANSIENT state that a dead consumer would otherwise leave
        # haunting the cell: `errors` (results live on the Volume, errors don't — a
        # redeploy may have changed the worker code, so a prior failure must not read
        # back as "failed" while the GPU never reran it) AND `queued`/`running` (phantom
        # pending from the just-dropped queue would make the frontend wait forever on
        # jobs no consumer will ever run). `done` / `ident` and the result blobs are
        # PRESERVED — the client just re-enqueues what it still wants and a fresh
        # consumer picks it up. (Content edits also self-clear via the server's per-step
        # content-key check; this covers the same-content redeploy case.)
        try:
            for key in list(attn_state.keys()):
                if not (isinstance(key, str) and key.startswith("cell:")):
                    continue
                st = attn_state.get(key)
                if not isinstance(st, dict):
                    continue
                if st.get("errors") or st.get("queued") or st.get("running"):
                    st["errors"] = {}
                    st["queued"] = []
                    st["running"] = []
                    attn_state[key] = st
        except Exception:  # noqa: BLE001 — best-effort; stale state at worst re-clears on next compute
            pass

        def _pick_model_id(payload: dict[str, Any], items: list[dict[str, Any]]) -> str:
            return (
                payload.get("model_id")
                or (items[0].get("export", {}).get("meta", {}).get("model_id") if items else None)
                or (items[0].get("compute_item", {}).get("export", {}).get("meta", {}).get("model_id") if items else None)
                or DEFAULT_MODEL_ID
            )

        @web_app.post("/enqueue")
        def enqueue(payload: dict[str, Any]) -> dict[str, Any]:
            """Stream a WINDOW of a cell's steps into the durable queue with a SINGLE
            status-doc write. Body: `{cell_hash, ident:{run,slot,model_alias,model_id},
            prompt_version, model_id, force?, items:[{event_index, input_key, req,
            compute_item, max_heads?, top_k?, max_query_tokens?}]}`. Each item's full
            compute payload is stashed on the Volume at its deterministic path (the
            queue carries only a tiny ref), so the worker needs no callback to us.

            DEDUP is keyed on the server-supplied opaque `req` token (content key +
            params + versions + force nonce): a step already IN FLIGHT with the SAME
            token is a duplicate re-send and is skipped, so the client can re-send its
            rolling window cheaply. A CHANGED token (raised heads / edited content /
            force) falls through and (re)streams — that's how a recompute is flagged.
            `done` is NOT skipped here: the server only re-sends a done step it deems
            stale, and the worker adopts an unchanged result (no GPU) or recomputes a
            changed one — so it can never be computed twice, yet always recomputes when
            the request truly changed."""
            h = str(payload["cell_hash"])
            ident_in = payload.get("ident") or {}
            model_id = str(payload.get("model_id") or ident_in.get("model_id") or DEFAULT_MODEL_ID)
            prompt_version = payload.get("prompt_version")
            ident = {**ident_in, "model_id": model_id, "prompt_version": prompt_version}

            st = _cell_state(h)
            _reap_stale(st, h)  # recover a dead consumer using the doc we already fetched
            active = set(st.get("queued", [])) | set(st.get("running", []))

            def _inflight_req(ev: int) -> str | None:
                """The request token already queued/running for this step (read from
                its job blob), so an identical re-send is recognized as a duplicate.
                None when the step isn't in flight — a done step is left to the worker's
                result-blob dedup, which adopts a matching result with no GPU and
                recomputes a changed one."""
                if ev not in active:
                    return None
                try:
                    return str((_blob_read(_job_blob_path(h, ev)) or {}).get("req") or "")
                except Exception:  # noqa: BLE001 — no readable payload → treat as not in flight
                    return None

            to_stream: list[tuple[int, str]] = []
            already_active: list[int] = []
            for raw in payload.get("items") or []:
                ev = int(raw["event_index"])
                req = str(raw.get("req") or "")
                if req and _inflight_req(ev) == req:
                    already_active.append(ev)  # identical request already in flight — dedup
                    continue
                # The export can exceed a Dict.put, so the payload lives on the Volume;
                # the queue ref is tiny and the worker reads the payload at compute time.
                # `req` rides on BOTH so the worker can cheaply skip a duplicate ref
                # (from the ref) and authoritatively dedup on the committed payload.
                _blob_write(_job_blob_path(h, ev), {
                    "compute_item": raw["compute_item"],
                    "input_key": raw.get("input_key"),
                    "prompt_version": prompt_version,
                    "req": req,
                })
                to_stream.append((ev, req))
            if not to_stream:
                return {"accepted": [], "cached": [], "already_active": sorted(already_active)}

            _vol_commit()          # payloads MUST be durable/visible before refs stream
            st = _cell_state(h)    # re-read: the commit above is slow — minimize RMW clobber
            for ev, _req in to_stream:
                _list_add(st, "queued", ev)
                _list_remove(st, "done", ev)  # (re)queued → not done until the worker republishes
                st.get("errors", {}).pop(str(ev), None)
            st["ident"] = ident
            _cell_save(h, st)
            for ev, req in to_stream:
                job_queue.put({"cell_hash": h, "event_index": ev, "model_id": model_id, "req": req},
                              partition=_part(model_id))
            _ensure_consumer(model_id)
            return {"accepted": sorted(ev for ev, _ in to_stream), "cached": [],
                    "already_active": sorted(already_active)}

        @web_app.post("/reset")
        def reset(payload: dict[str, Any]) -> dict[str, Any]:
            """Reset a cell's PENDING compute so a fresh compute never inherits a
            prior run's phantom jobs. Clears (1) the model's durable queue PARTITION
            — the queue is shared per model, so stale refs from an earlier run/scene
            sit ahead of the new work in the FIFO and would be chewed through first —
            and (2) this cell's queued/running lists + their staged job payloads (a
            straggler ref then no-ops on drain). Committed results (`done`, on the
            Volume) are PRESERVED; only the *pending* compute is dropped. A live
            consumer's in-flight forward finishes and publishes normally, then idle-
            exits; the next enqueue re-spawns one."""
            h = str(payload["cell_hash"])
            model_id = str(payload.get("model_id") or DEFAULT_MODEL_ID)
            try:
                job_queue.clear(partition=_part(model_id))
            except Exception:  # noqa: BLE001 — best-effort; the doc + blob clear below still resets
                pass
            st = _cell_state(h)
            pending = sorted({int(e) for e in list(st.get("queued", [])) + list(st.get("running", []))})
            for ev in pending:
                try:
                    _blob_unlink(_job_blob_path(h, ev))  # drained straggler ref then no-ops
                except Exception:  # noqa: BLE001
                    pass
            st["queued"] = []
            st["running"] = []
            _cell_save(h, st)
            _log("reset pending", h=h[:8], model=model_id, cleared=len(pending))
            return {"cleared": pending}

        @web_app.get("/queue")
        def queue_status(cell_hash: str) -> dict[str, Any]:
            st = _cell_state(cell_hash)
            _reap_stale(st, cell_hash)
            return {
                "queued": st.get("queued", []),
                "running": st.get("running", []),
                "done_pending": st.get("done", []),
                "errors": st.get("errors", {}),
                "ident": st.get("ident", {}),
            }

        @web_app.post("/pull")
        def pull(payload: dict[str, Any]) -> dict[str, Any]:
            """FAST status poll — the whole cell's queue metadata in ONE `Dict.get`.
            NO Volume reload, NO blob reads: finished steps are just listed in `done`,
            and the caller fetches their (large) result blobs separately via `/results`,
            so the frequent poll stays cheap and never blocks on IO."""
            h = str(payload["cell_hash"])
            st = _cell_state(h)
            _reap_stale(st, h)
            # Keep the GPU consumer alive if there's still queued work to stream.
            if st.get("queued"):
                mid = (st.get("ident") or {}).get("model_id")
                if mid:
                    _ensure_consumer(mid)
            return {
                "queued": st.get("queued", []),
                "running": st.get("running", []),
                "errors": st.get("errors", {}),
                "done": st.get("done", []),
            }

        @web_app.post("/results")
        def results_fetch(payload: dict[str, Any]) -> dict[str, Any]:
            """Pull finished steps' SMALL `compact` view (heatmap + aggregates) — the
            frequent poll's result path. It reads only the tiny `.meta.json` (stamp +
            keys to verify) and the `.compact.json` projection, both TARGETED from the
            Volume backend at DETERMINISTIC paths (no Dict lookup, no ack, no whole-
            Volume reload). The multi-hundred-MB `.full.json` is NEVER touched here —
            that only rides `/blob`, on demand, when a user opens a step's token/present
            detail. A step whose blobs aren't committed yet is omitted (caller retries;
            at-least-once). Reads run in PARALLEL so latency is one read, not the sum.
            Body: `{cell_hash, event_indices:[int]}`."""
            import concurrent.futures as _fut

            h = str(payload["cell_hash"])
            evs = [int(x) for x in (payload.get("event_indices") or [])]
            if not evs:
                return {"results": []}

            def _one(ev: int) -> dict[str, Any] | None:
                meta = _blob_pull(_meta_path(h, ev))        # tiny; None until committed
                if not meta:
                    return None
                compact = _blob_pull(_compact_path(h, ev))  # small projected view
                if compact is None:
                    return None
                return {
                    "event_index": ev,
                    "input_key": meta.get("input_key"),
                    "prompt_version": meta.get("prompt_version"),
                    "stamp": meta.get("stamp"),
                    "result": compact,
                }

            with _fut.ThreadPoolExecutor(max_workers=min(len(evs), 8)) as ex:
                results = [r for r in ex.map(_one, evs) if r is not None]  # map preserves order
            return {"results": results}

        @web_app.post("/blob")
        def blob_fetch(payload: dict[str, Any]):
            """STREAM one step's big `.full.json` result verbatim off the Volume backend
            — the on-demand heavy pull for a single step's token/present detail. The file
            is chunked straight from `read_file` into the response, so the web container
            NEVER buffers the whole (hundreds-of-MB) blob in memory or re-serializes it —
            that buffering was what stalled the tier and made large results never return.
            404 when the step has no committed result yet (caller falls back to a retry).
            Body: `{cell_hash, event_index}`."""
            h = str(payload["cell_hash"])
            ev = int(payload["event_index"])
            rel = _vol_rel(_full_path(h, ev))
            if _blob_pull(_meta_path(h, ev)) is None:  # cheap existence probe before we stream
                return fastapi.responses.JSONResponse({"error": "no result blob"}, status_code=404)

            def _stream():
                try:
                    for chunk in attn_vol.read_file(rel):  # blocking sync iterator, streamed
                        yield chunk
                except FileNotFoundError:
                    return  # committed meta but full not visible yet — caller retries

            return fastapi.responses.StreamingResponse(_stream(), media_type="application/json")

        # These routes only SPAWN/heartbeat (quick blocking Modal calls) or read
        # Dicts, so they run as sync `def` — Starlette offloads them to a threadpool
        # and the blocking Modal client calls can't stall the container's event loop
        # (which would starve heartbeats/cancellation → the container gets killed).
        # `/result` stays async because it legitimately awaits `fc.get.aio()`.
        @web_app.post("/")
        def submit(payload: dict[str, Any]) -> dict[str, Any]:
            items = payload["items"] if isinstance(payload, dict) and "items" in payload else [payload]
            incoming = _pick_model_id(payload, items)
            # Resolve to an OPEN-weight model (by HF path / alias / OpenRouter id).
            # Closed / unsupported models are rejected — attention needs HF weights.
            spec = reg.resolve_open_model(incoming)
            if spec is None:
                return fastapi.responses.JSONResponse(
                    {"error": f"'{incoming}' is not a supported open-weight model for attention analysis",
                     "supported": [{"alias": m.alias, "hf_path": m.hf_path, "hf_url": m.hf_url} for m in reg.OPEN_MODELS]},
                    status_code=422,
                )
            Worker = _WORKER_BY_GPU.get(spec.gpu)
            if Worker is None:
                return fastapi.responses.JSONResponse(
                    {"error": f"no worker configured for GPU tier '{spec.gpu}' (model {spec.alias})"},
                    status_code=501,
                )
            # One spawn for the whole batch — the worker drains items FIFO internally
            # instead of Modal canceling a pile of independent FunctionCalls.
            worker = Worker(model_id=spec.hf_path)
            call_id = worker.analyze_many.spawn(items).object_id
            return {"call_id": call_id, "call_ids": [call_id], "count": len(items)}

        @web_app.post("/warm")
        def warm(payload: dict[str, Any] | None = None) -> dict[str, Any]:
            payload = payload or {}
            incoming = payload.get("model_id") or DEFAULT_MODEL_ID
            spec = reg.resolve_open_model(incoming)
            if spec is None:
                return fastapi.responses.JSONResponse(
                    {"error": f"'{incoming}' is not a supported open-weight model"},
                    status_code=422,
                )
            Worker = _WORKER_BY_GPU.get(spec.gpu)
            if Worker is None:
                return fastapi.responses.JSONResponse(
                    {"error": f"no worker configured for GPU tier '{spec.gpu}'"},
                    status_code=501,
                )
            worker = Worker(model_id=spec.hf_path)
            call_id = worker.ping.spawn().object_id
            return {"call_id": call_id, "status": "warming"}

        @web_app.get("/result/{call_id}")
        async def result(call_id: str):
            fc = modal.FunctionCall.from_id(call_id)
            try:
                return {"status": "done", "result": await fc.get.aio(timeout=0)}  # non-blocking pull
            except TimeoutError:
                return fastapi.responses.JSONResponse({"status": "pending"}, status_code=202)
            except Exception as e:  # noqa: BLE001 — the spawned compute raised; surface it to the poller
                return fastapi.responses.JSONResponse(
                    {"status": "error", "error": f"{type(e).__name__}: {e}"}, status_code=500,
                )

        return web_app
