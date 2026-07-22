"""Fixed benchmark slots. Each is a resumable pipeline run keyed by id.

A "run" is a (slot, model) cell — every slot can be driven by any of the
aliased LLMs in parallel, and the dashboard switches between cells by
flipping the active model. Aliases map to the OpenRouter model IDs the
llm service feeds straight into chat.send_async.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Slot:
    id: str
    prompt: str


SLOTS: list[Slot] = [
    Slot("swamp-land", "A swamp with islands, designed as a top-down arcade level where a frog can jump from island to island"),
    Slot("hotel-room", "A hotel room"),
    Slot("modern-house", "A modern house"),
    Slot("platformer-level", "A super mario bros type platformer level"),
    Slot("battle-arena", "A battle arena for a two player game"),
    Slot("urban-city", "An imperial japanese house"),
    Slot("university-campus", "An open university-campus"),
    Slot("shooter", "FPS 5v5 3 lane map"),
    Slot("racetrack", "A car racing track"),
    Slot("campsite", "A campsite in the middle of a forest"),
    Slot("desert", "a desert landscape"),
    Slot("campsites", "three family campsites in an open field"),
    Slot("outer space", "a planetary system"),
    Slot("suburban home", "a traditional two-story suburban house"),
    Slot("toy room", "shrunk down in gigantic toy room"),
    Slot("modern-house-cliff", "a modern house built into a cliff"),
    Slot("battle-arena-medieval", "a two-player battle arena in a medieval castle"),
    Slot("A startup office", "a startup office"),
    Slot("Among Us", "an among us map"),
    Slot("interesting modern house", "A luxurious, elegantly designed and architecturally creative modern house built on flat ground"),
    Slot("more interesting modern house", "A compact, artistic, elegantly designed modern house built on flat ground"),
    Slot("specific house", "modern house constructed of several independent volumes stacked/offset together"),
    Slot("factory", "a factory assembly line"),
    Slot("CS Dust2", "Dust II from Counter-Strike"),
    Slot("fixed platformer", "A fixed-camera platformer level in the style of Celeste"),
    Slot("hotel lobby", "create a small, old luxury hotel lobby room that has an elevator and a cozy lounge with paintings and photos on one side and the hotel reception on the opposite side with a bell to ring")
]

SLOTS_BY_ID: dict[str, Slot] = {s.id: s for s in SLOTS}


@dataclass(frozen=True)
class OpenAICompatModel:
    """A third-party OpenAI-compatible /chat/completions backend, slotted in
    beside the OpenRouter-routed models. The pipeline still drives it through the
    same `llm.call_llm` path (cache, resample/transport retries, reasoning +
    token capture, the compare gate) — only the transport differs."""

    # Provider-side model id sent in the request body.
    model: str
    # OpenAI-compatible API root, no trailing slash; the call posts to
    # `{base_url}/chat/completions`.
    base_url: str
    # Env var holding the bearer key; None for auth-less endpoints.
    api_key_env: str | None = None
    # When True, the bearer key comes from a rotating pool that rolls to the
    # next key whenever the provider answers HTTP 429. The pool loads from
    # `{api_key_env}_ARRAY` (JSON array or comma/newline-separated), falling
    # back to the single `api_key_env` var — a one-key pool never rotates, so
    # enabling this without an `_ARRAY` var changes nothing.
    rotate: bool = False
    # Sampling / length knobs, each sent only when set (None defers to the
    # provider's own default).
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    # Extra request-body fields merged verbatim (e.g. LongCat's `thinking`).
    extra: dict[str, object] = field(default_factory=dict)
    # USD per 1,000,000 tokens, input and output. These backends bill by token
    # and (unlike OpenRouter) return no per-request cost, so this IS the price:
    # `token_cost` derives each call's spend from its token counts. Both must be
    # set for a model to be priced; None leaves it cost-tracked only by tokens.
    price_in: float | None = None
    price_out: float | None = None


# Short alias -> OpenRouter model id. Aliases are the stable identifiers
# the client + storage layout key on; the OpenRouter id is what llm.py
# passes into the SDK call. Order is the order the dashboard renders.
MODELS: dict[str, str] = {
    "gemini-flash": "google/gemini-3.5-flash",
    "gemini-flash-lite": "google/gemini-3.1-flash-lite",
    "gemini-pro": "google/gemini-3.1-pro-preview",
    "gpt": "openai/gpt-5.5",
    "opus": "anthropic/claude-opus-4.6",
    "deepseek": "deepseek/deepseek-v4-pro",
    "opus-new": "anthropic/claude-opus-4.8",
    "opus-fable": "anthropic/claude-fable-5",
    "grokky": "x-ai/grok-4.5",
    "glm": "z-ai/glm-5.2",                                                                                                                                                      
    "qwen": "qwen/qwen3.7-max",
    "qwen-max-preview": "qwen/qwen3.8-max-preview",
    "kimi": "moonshotai/kimi-k2-thinking",
    "sonnet": "anthropic/claude-sonnet-4.6",
    "sonnet-new": "anthropic/claude-sonnet-5",
    "minimax": "minimax/minimax-m3",
    "gemma": "google/gemma-4-31b-it",
    "luna-pro": "openai/gpt-5.6-luna-pro",
    "terra-pro": "openai/gpt-5.6-terra-pro",
    "sol-pro": "openai/gpt-5.6-sol-pro",
    "kimi-k3": "moonshotai/kimi-k3",
    "hy3": "tencent/hy3",
    "longcat": "longcat/LongCat-2.0",
    "longcat-sf": "siliconflow/LongCat-2.0",
    "gemini-flash-new": "google/gemini-3.6-flash",
    "laguna": "poolside/laguna-s-2.1",
}

# Model ids from MODELS that are actually served by a third-party
# OpenAI-compatible /chat/completions endpoint. Keyed by the SAME id string
# MODELS maps to, so that id stays the cache key, the log/dashboard identity,
# and the compare pin — llm.py swaps only the transport (direct httpx to
# `base_url` instead of the OpenRouter SDK). Any id absent here routes through
# OpenRouter exactly as before.
OPENAI_COMPAT_MODELS: dict[str, OpenAICompatModel] = {
    "moonshotai/kimi-k3": OpenAICompatModel(
        model="kimi-k3",
        base_url="https://api.moonshot.ai/v1",
        api_key_env="MOONSHOT_API_KEY",
        rotate=True,
        extra={"reasoning_effort": "max", "max_completion_tokens": 1048576},
        # Moonshot list price (July 2026): $3.00 / $15.00 per 1M tokens. Applied
        # to kimi-k3 spend regardless of transport — the run also billed some
        # calls through OpenRouter BYOK, which reports $0, so the token rate is
        # the only true cost for both the direct and BYOK legs.
        price_in=3.00,
        price_out=15.00,
    ),
    "longcat/LongCat-2.0": OpenAICompatModel(
        model="LongCat-2.0",
        base_url="https://api.longcat.chat/openai/v1",
        api_key_env="LONGCAT_API_KEY",
        rotate=True,
        max_tokens=131072,
        extra={"thinking": {"type": "enabled"}},
    ),
    "siliconflow/LongCat-2.0": OpenAICompatModel(
        model="meituan-longcat/LongCat-2.0",
        base_url="https://api.siliconflow.com/v1",
        api_key_env="SILICONFLOW_API_KEY",
        rotate=True,
        extra={"enable_thinking": True},
    ),
    "qwen/qwen3.8-max-preview": OpenAICompatModel(
        model="qwen3.8-max-preview",
        base_url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        api_key_env="ALIBABA_API_KEY",
        rotate=False,
        extra={"reasoning_effort": "xhigh"},
        # Alibaba Model Studio (International) Qwen-Max rate. `qwen3.8-max-preview`
        # is a preview alias — confirm against the current Model Studio price
        # sheet and adjust if it differs.
        price_in=1.20,
        price_out=6.00,
    ),
}


def model_pricing(model_id: str | None) -> tuple[float, float] | None:
    """`(price_in, price_out)` USD per 1M tokens for `model_id`, or None if it
    has no static price. Matches BOTH the OpenRouter id an event/BYOK-flight
    records (`moonshotai/kimi-k3`) and the provider-side id a direct flight
    records (`cfg.model`, e.g. `kimi-k3`), so one price covers a model however
    it was routed."""
    if not model_id:
        return None
    for or_id, cfg in OPENAI_COMPAT_MODELS.items():
        if cfg.price_in is None or cfg.price_out is None:
            continue
        if model_id == or_id or model_id == cfg.model:
            return cfg.price_in, cfg.price_out
    return None


def token_cost(
    model_id: str | None, tokens_in: int | None, tokens_out: int | None
) -> float | None:
    """USD cost of one call from its token counts and `model_id`'s static price,
    or None when the model is unpriced (routes through OpenRouter's settled cost
    instead) or no tokens were recorded (an errored attempt). Missing side
    counts to 0 so a partial usage record still prices what it has."""
    pricing = model_pricing(model_id)
    if pricing is None or (tokens_in is None and tokens_out is None):
        return None
    price_in, price_out = pricing
    return (tokens_in or 0) / 1_000_000 * price_in + (tokens_out or 0) / 1_000_000 * price_out

MODEL_ALIASES: list[str] = list(MODELS.keys())

DEFAULT_MODEL_ALIAS = "gemini-flash-lite"

DEFAULT_MODEL = MODELS[DEFAULT_MODEL_ALIAS]

DEFAULT_REASONING = "xhigh"

REASONING_DOWNGRADE_LIST = [
    "openai/gpt-5.5"
]

# OpenRouter model ids whose only provider can't honor structured outputs
# (`response_format: json_schema`). For these, `llm._send_structured` omits the
# response_format param — which would otherwise 404 under `require_parameters`
# ("No endpoints found that can handle the requested parameters") — and relies on
# the prompt's `<output>` contract to shape the JSON instead. Keep this to models
# that genuinely lack structured-output support (e.g. poolside/laguna-s-2.1),
# since dropping the schema removes the wire-level guarantee that output parses.
NO_STRUCTURED_OUTPUT_LIST = [
    "poolside/laguna-s-2.1",
]