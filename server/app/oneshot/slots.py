"""Slots + models for the one-shot track.

This track benchmarks DIFFUSION LLMs (dLLMs) — the Starshot dLLM gateway on
Modal (LLaDA, DiffusionGemma) and Inception's Mercury API — plus an
autoregressive frontier baseline (Opus over OpenRouter). Every entry is a
full provider config for an OpenAI-compatible chat-completions endpoint;
`app.oneshot.llm` talks straight to `base_url` with the named key. Edit
freely; nothing here touches the pipeline benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Slot:
    id: str
    prompt: str
    # Side of the square canvas in feet. Read by v3-room, where the
    # harness-built room shell (and the geometry stated in the prompt)
    # scales with the brief; every other version uses the fixed
    # pipeline.CANVAS_SIDE_FT canvas.
    canvas_ft: float = 50.0


@dataclass(frozen=True)
class DllmModel:
    """One OpenAI-compatible chat-completions backend."""

    # Provider-side model id (what goes in the request body).
    model: str
    # OpenAI-compatible API root, e.g. https://host/v1 (no trailing slash).
    base_url: str
    # Env var holding the bearer key; None for auth-less endpoints.
    api_key_env: str | None = None
    # Generation cap; None defers to the provider's default.
    max_tokens: int | None = None
    # Extra request-body fields merged verbatim (e.g. reasoning_effort).
    extra: dict[str, object] = field(default_factory=dict)


SLOTS: list[Slot] = [
    
    Slot("bedroom", "luxurious and detailed master bedroom featuring a king-sized bed with nightstands a series of chairs laid out in a T shape, an L-shaped sofa in the corner, floor lamps, an area for professional work, a contemporary music player, and an overall lavish design", canvas_ft=50.0),
    Slot("bunker", "emergency underground bunker with storage units for food, bunk beds, first aid kits, and a workstation with a console", canvas_ft=50.0),
    Slot("church", "prayer hall where mass is held, filled with rows of pews and a giant cross at the front", canvas_ft=50.0),
    Slot("mario (V5)", "super mario brothers"),
    Slot("hollow knight (V5)", "hollow knight")
]

SLOTS_BY_ID: dict[str, Slot] = {s.id: s for s in SLOTS}

MODELS: dict[str, DllmModel] = {
    # 16B-MoE LLaDA 2.0 mini behind the Starshot dLLM gateway (auth-less
    # internal Modal deployment). The gateway exposes NO decoding/reasoning
    # knobs by design (it owns diffusion steps + canvas) and already runs
    # structured output at its max step budget (structured_steps=2048 vs
    # chat_steps=128 per /health); dLLMs emit no separate reasoning channel.
    # Temperature is deliberately not sent — the gateway applies the model's
    # default (0.0; higher temps cause language mixing).
    "llada": DllmModel(
        model="inclusionai/llada2.0-mini",
        base_url="https://starshot-aitools--starshot-dllm-gateway.modal.run/v1",
    ),
    # DiffusionGemma 26B (A4B) — the gateway's other live backend (it replaced
    # the doc's dream model; confirmed warm via /v1/models, 262k context).
    # Same gateway semantics as llada above.
    "gemma": DllmModel(
        model="google/diffusiongemma-26b-a4b-it",
        base_url="https://starshot-aitools--starshot-dllm-gateway.modal.run/v1",
    ),
    # Inception's Mercury 2 reasoning dLLM at MAX reasoning: effort "high"
    # (the API's top setting; default is medium). max_tokens is the API
    # maximum (50k) because REASONING TOKENS COUNT AGAINST IT — a tight cap
    # gets burned by high-effort reasoning and returns finish_reason="length"
    # with EMPTY content (verified live). Temperature is not sent (API
    # default 0.75; valid range is 0.5-1).
    #
    # Do NOT add `reasoning_summary` / `reasoning_summary_wait`: combined
    # with `response_format: json_schema` on the scene schema, mercury
    # returns finish_reason="stop" with EMPTY content while the summary
    # itself completes (verified live, with and without `wait`; the tiny
    # smoke schema worked, the real one reliably fails). Scene > trace, so
    # the reasoning panel stays empty for this model.
    "inception": DllmModel(
        model="mercury-2",
        base_url="https://api.inceptionlabs.ai/v1",
        api_key_env="INCEPTION_API_KEY",
        max_tokens=50000,
        extra={"reasoning_effort": "high"},
    ),
    # Same mercury-2 with reasoning OFF ("instant" is the API's no-reasoning
    # setting; the default is medium, so it must be explicit). Pairs with
    # "inception" to isolate the reasoning-effort axis on one dLLM.
    "inception-instant": DllmModel(
        model="mercury-2",
        base_url="https://api.inceptionlabs.ai/v1",
        api_key_env="INCEPTION_API_KEY",
        max_tokens=50000,
        extra={"reasoning_effort": "instant"},
    ),
    # Autoregressive frontier baseline: Claude Opus 4.8 over OpenRouter's
    # OpenAI-compatible API, at the same max reasoning effort the pipeline
    # benchmark uses (`xhigh`). Reasoning comes back on `message.reasoning`.
    "opus": DllmModel(
        model="anthropic/claude-opus-4.8",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        extra={"reasoning": {"effort": "xhigh"}},
    ),
    "gemma-4-normal": DllmModel(
        model="google/gemma-4-26b-a4b-it",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        extra={"reasoning": {"effort": "xhigh"}},
    ),
    "qwen": DllmModel(
        model="qwen/qwen3.6-27b",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        extra={"reasoning": {"effort": "xhigh"}},
    )
}

MODEL_ALIASES: list[str] = list(MODELS.keys())

DEFAULT_MODEL_ALIAS = "llada"
