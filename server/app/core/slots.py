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
    Slot("fixed platformer", "A fixed-camera platformer level in the style of Celeste")
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
    # Sampling / length knobs, each sent only when set (None defers to the
    # provider's own default).
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    # Extra request-body fields merged verbatim (e.g. LongCat's `thinking`).
    extra: dict[str, object] = field(default_factory=dict)


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
    "opus-fable": "anthropic/claude-fable-latest",
    "grokky": "x-ai/grok-4.5",
    "glm": "z-ai/glm-5.2",
    "qwen": "qwen/qwen3.7-max",
    "kimi": "moonshotai/kimi-k2-thinking",
    "sonnet": "anthropic/claude-sonnet-4.6",
    "sonnet-new": "anthropic/claude-sonnet-5",
    "minimax": "minimax/minimax-m3",
    "gemma": "google/gemma-4-31b-it",
    "luna-pro": "openai/gpt-5.6-luna-pro",
    "hy3": "tencent/hy3",
    "longcat": "longcat/LongCat-2.0",
    "longcat-sf": "siliconflow/LongCat-2.0",
}

# Model ids from MODELS that are actually served by a third-party
# OpenAI-compatible /chat/completions endpoint. Keyed by the SAME id string
# MODELS maps to, so that id stays the cache key, the log/dashboard identity,
# and the compare pin — llm.py swaps only the transport (direct httpx to
# `base_url` instead of the OpenRouter SDK). Any id absent here routes through
# OpenRouter exactly as before.
OPENAI_COMPAT_MODELS: dict[str, OpenAICompatModel] = {
    "longcat/LongCat-2.0": OpenAICompatModel(
        model="LongCat-2.0",
        base_url="https://api.longcat.chat/openai/v1",
        api_key_env="LONGCAT_API_KEY",
        stream=False,
        max_tokens=131072,
        # Thinking on; the trace comes back on `reasoning_content`.
        extra={"thinking": {"type": "enabled"}},
    ),
    # Same LongCat-2.0 weights, served by SiliconFlow instead of Meituan's own
    # gateway. SiliconFlow's thinking toggle is `enable_thinking` (not LongCat's
    # native `thinking` object); the answer/trace come back the same way.
    "siliconflow/LongCat-2.0": OpenAICompatModel(
        model="meituan-longcat/LongCat-2.0",
        base_url="https://api.siliconflow.com/v1",
        api_key_env="SILICONFLOW_API_KEY",
        extra={"enable_thinking": True},
    ),

}

MODEL_ALIASES: list[str] = list(MODELS.keys())

DEFAULT_MODEL_ALIAS = "gemini-flash-lite"

DEFAULT_MODEL = MODELS[DEFAULT_MODEL_ALIAS]

DEFAULT_REASONING = "xhigh"

REASONING_DOWNGRADE_LIST = [
    "openai/gpt-5.5"
]