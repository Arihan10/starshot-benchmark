"""Fixed benchmark slots. Each is a resumable pipeline run keyed by id.

A "run" is a (slot, model) cell — every slot can be driven by any of the
aliased LLMs in parallel, and the dashboard switches between cells by
flipping the active model. Aliases map to the OpenRouter model IDs the
llm service feeds straight into chat.send_async.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


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
    Slot("modern-house-cliff", "a modern house built into a cliff"),
    Slot("battle-arena-medieval", "a two-player battle arena in a medieval castle")
]

SLOTS_BY_ID: dict[str, Slot] = {s.id: s for s in SLOTS}

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
    "opus-new": "anthropic/claude-opus-4.8"
}

MODEL_ALIASES: list[str] = list(MODELS.keys())

DEFAULT_MODEL_ALIAS = "gemini-flash"

DEFAULT_MODEL = MODELS[DEFAULT_MODEL_ALIAS]
