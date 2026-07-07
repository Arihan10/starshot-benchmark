"""Registry of OPEN-WEIGHT models the attention analysis can run on.

The generation side (`core.slots.MODELS`) maps aliases to OpenRouter ids, most of
which are CLOSED APIs (gpt / claude / gemini). Attention analysis is different: it
loads the REAL HF weights on the Modal GPU worker and replays a teacher-forced
forward, so it only works for open-weight models. This module is the single source
of truth for:

  * WHICH models are supported (everything else is gated off),
  * their HF repo path + human link,
  * per-model quirks: native thinking delimiters, the HF loader class, and which
    layers carry the softmax attention we can instrument,
  * the Modal GPU tier each model needs (drives which worker class runs it).

It is PURE python (no torch / modal / transformers), so it imports cleanly on the
API server (for gating), on the Modal worker (for quirks), and at deploy time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpenModelSpec:
    """One open-weight model the attention worker can load + instrument."""

    alias: str                       # short attention-side id
    hf_path: str                     # HF repo id loaded via from_pretrained
    hf_url: str                      # human-facing model card link
    gpu: str = "H200"                # Modal gpu spec ("H200", "B200:2", ...) -> worker tier
    thinking_open: str = ""          # native reasoning delimiters (OpenRouter strips them);
    thinking_close: str = ""         # reconstruct.py re-inserts these around the reasoning
    loader: str = "causal_lm"        # "causal_lm" | "image_text_to_text" (VL / *ForConditionalGeneration)
    openrouter_id: str | None = None # generation id in core.slots.MODELS, for matching
    memory_mb: int = 131072          # container RAM (CPU-side headroom)
    scaledown_window: int = 360      # idle-kill seconds for this model's worker
    notes: str = ""


# The models we can run attention on. Add an entry (HF path + tier + quirks) to
# support a new open model; closed models simply have no entry and are gated off.
OPEN_MODELS: tuple[OpenModelSpec, ...] = (
    OpenModelSpec(
        alias="gemma",
        hf_path="google/gemma-4-31b-it",
        hf_url="https://huggingface.co/google/gemma-4-31b-it",
        gpu="H200",
        openrouter_id="google/gemma-4-31b-it",
        notes="Multimodal Gemma; interleaved local/global attention (5 local : 1 global). "
              "~62 GB bf16 -> one H200. Instruments the global-attention layers.",
    ),
    OpenModelSpec(
        alias="qwen-122b",
        hf_path="Qwen/Qwen3.5-122B-A10B",
        hf_url="https://huggingface.co/Qwen/Qwen3.5-122B-A10B",
        gpu="B200:2",
        thinking_open="<think>",
        thinking_close="</think>",
        loader="image_text_to_text",   # arch Qwen3_5MoeForConditionalGeneration (VL wrapper)
        openrouter_id="qwen/qwen3.5-122b-a10b",
        memory_mb=262144,
        scaledown_window=600,
        notes="Qwen3.5-122B-A10B (Apache-2.0, Feb 2026). Hybrid MoE, 48 decoder layers "
              "laid out 12 x (3 x linear_attention[Gated-DeltaNet] -> 1 x full_attention"
              "[Gated-Attention]); ~122B total / 10B active over 256 experts (8+1). Only "
              "the 12 full_attention layers (idx 3,7,..,47) carry softmax we can "
              "instrument -- selected data-drivenly via config.text_config.layer_types; "
              "the 36 DeltaNet layers are linear attention (no softmax distribution). "
              "Gated-Attention: 32 Q / 2 KV heads (GQA n_rep=16), head_dim 256, partial "
              "RoPE (0.25); the sigmoid output gate is applied AFTER attention so the "
              "captured post-RoPE/QK-norm Q/K are the true softmax inputs. mRoPE reduces "
              "to standard RoPE for our text-only teacher-forced forward. ~250 GB bf16 -> "
              "B200:2 (~360 GB) with device_map=auto pipeline-shard. Needs the cu128/"
              "transformers>=5.2 image (see modal_app.big_image).",
    ),
)


def _norm(s: str) -> str:
    """Normalize a model id for fuzzy matching: drop the vendor prefix and any
    separators/case, so an OpenRouter id ('qwen/qwen3.5-122b-a10b') matches its HF
    path ('Qwen/Qwen3.5-122B-A10B') -> both become 'qwen35122ba10b'."""
    tail = str(s).strip().split("/")[-1].lower()
    for ch in ("-", ".", "_", " "):
        tail = tail.replace(ch, "")
    return tail


_BY_HF = {m.hf_path: m for m in OPEN_MODELS}
_BY_ALIAS = {m.alias: m for m in OPEN_MODELS}
_BY_OR = {m.openrouter_id: m for m in OPEN_MODELS if m.openrouter_id}
_BY_NORM: dict[str, OpenModelSpec] = {}
for _m in OPEN_MODELS:
    for _key in (_m.hf_path, _m.alias, _m.openrouter_id):
        if _key:
            _BY_NORM.setdefault(_norm(_key), _m)


def resolve_open_model(model_id_or_alias: str | None) -> OpenModelSpec | None:
    """The OpenModelSpec for a model id (OpenRouter id, HF path, or attention
    alias), or None for a closed / unsupported model. Matching is exact first
    (HF path / alias / OpenRouter id), then normalized (vendor + separators
    stripped) so OpenRouter-vs-HF id spelling differences still resolve."""
    if not model_id_or_alias:
        return None
    key = str(model_id_or_alias).strip()
    return (
        _BY_HF.get(key)
        or _BY_ALIAS.get(key)
        or _BY_OR.get(key)
        or _BY_NORM.get(_norm(key))
    )


def is_open_model(model_id_or_alias: str | None) -> bool:
    return resolve_open_model(model_id_or_alias) is not None


def n_gpus(gpu: str) -> int:
    """Number of GPUs in a Modal gpu spec ('H200' -> 1, 'B200:2' -> 2)."""
    parts = str(gpu).split(":")
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return 1
