"""Accurate native-input reconstruction for the real (GPU) attention path.

The mock path tokenizes the tf-export's `text.full` (a stand-in chat template).
For a faithful teacher-forced pass we must instead build the model's OWN input
with its real chat template, then remap the scene/output char-spans onto that
text. This module does exactly that, and it's a PURE function of the export plus
an injected `apply_chat_template` callable — so it runs (and is validated) with
the mock tokenizer locally, and with the real Gemma tokenizer on the worker.

Frames (see teacher_forcing.build_export):
  * scene / to-place / variable entities carry `user_rel` — offsets into the raw
    `text.user` string (what the model was actually fed as the user turn).
  * output assignments carry `output_rel` — offsets into `text.output` (the raw
    emitted string, = the logprobs token frame).

The chat template inserts `user` (and the model emits `output`) VERBATIM, so we
locate each raw string inside the reconstructed `full` text and shift the
relative offsets into that frame. `reconstruction_check` round-trips this (both
raw strings must be found) — the spec's "validate with round-trip comparisons".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.attention.schema import GenerationTrace

# Signature of the injected template renderer: messages -> rendered string
# (tokenize=False, add_generation_prompt=True). On the worker this is a thin
# wrapper over `tokenizer.apply_chat_template`.
ApplyChatTemplate = Callable[[list[dict[str, str]]], str]


def _shift_entry(entry: dict[str, Any], base: int, rel_key: str) -> dict[str, Any]:
    """Copy a map entry with `start`/`end` (and each component's) rebased into
    the reconstructed `full` frame from its raw-frame `rel_key` offsets."""
    rel = entry.get(rel_key)
    out = dict(entry)
    if rel:
        out["start"] = base + rel[0]
        out["end"] = base + rel[1]
    comps = []
    for c in entry.get("components", []) or []:
        crel = c.get("user_rel")
        cc = dict(c)
        if crel:
            cc["start"] = base + crel[0]
            cc["end"] = base + crel[1]
        comps.append(cc)
    if comps:
        out["components"] = comps
    return out


def build_real_trace(
    export: dict[str, Any],
    apply_chat_template: ApplyChatTemplate,
    *,
    thinking_open: str = "",
    thinking_close: str = "",
    remote_logprobs: dict[str, Any] | None = None,
) -> tuple[GenerationTrace, dict[str, Any]]:
    """Reconstruct the model's native sequence and remap the maps onto it.

    `thinking_open`/`thinking_close` wrap the reasoning inside the model turn to
    match the provider's native thinking format (OpenRouter strips it). Defaults
    to empty (reasoning inline before the answer); set them to Gemma's real
    thinking delimiters once known — the logprob round-trip will confirm.

    Returns `(GenerationTrace, check)` where `check` reports whether the raw
    `user`/`output` strings were found verbatim (a formatting round-trip)."""
    text = export["text"]
    system = str(text.get("system") or "")
    user = str(text.get("user") or "")
    reasoning = str(text.get("reasoning") or "")
    output = str(text.get("output") or "")

    messages: list[dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    input_text = apply_chat_template(messages)  # ends with the model-turn opener

    completion = ""
    if reasoning.strip():
        completion += f"{thinking_open}{reasoning}{thinking_close}"
    completion += output
    full = input_text + completion
    completion_start = len(input_text)

    # Locate the raw frames inside `full` to rebase the maps.
    #
    # Chat templates (Gemma's included) apply `| trim` to message content, so the
    # raw `user` — which often has leading/trailing whitespace (e.g. a trailing
    # newline after `</IMPORTANT_INSTRUCTIONS>`) — is NOT present verbatim; the
    # TRIMMED user is. Find the trimmed text and rebase by the stripped lead so
    # `user_rel` offsets (relative to the raw user) still land correctly. Without
    # this, decompose-style prompts (trailing newline) fail `find(user)` and lose
    # their entire scene map -> "no scene-attending heads".
    user_stripped = user.strip()
    lead = len(user) - len(user.lstrip())
    user_pos = full.find(user_stripped) if user_stripped else -1
    user_off = user_pos - lead if user_pos >= 0 else -1
    out_off = full.find(output, completion_start) if output else -1

    scene_map = [_shift_entry(m, user_off, "user_rel") for m in export.get("scene_map", [])] if user_pos >= 0 else []
    # The to-place batch lives in the same `user` frame as the scene, so it
    # rebases identically (used for the parallel to-place attention readout).
    to_place_map = [_shift_entry(m, user_off, "user_rel") for m in export.get("to_place_map", [])] if user_pos >= 0 else []
    output_map = (
        [{**m, "start": out_off + (m.get("output_rel") or [0, 0])[0], "end": out_off + (m.get("output_rel") or [0, 0])[1]}
         for m in export.get("output_map", [])]
        if out_off >= 0 else []
    )

    check = {
        "user_found": user_pos >= 0,
        "output_found": out_off >= 0,
        "input_chars": len(input_text),
        "full_chars": len(full),
        "completion_start": completion_start,
        "note": "user/output located in the reconstructed input (trim-aware) — offsets rebased."
        if (user_pos >= 0 and out_off >= 0)
        else "WARNING: a raw frame was not found (chat-template escaping beyond trim?); maps may be degraded.",
    }

    # Frames must COVER the whole completion so the region decomposition partitions
    # all attention (else the model's reasoning mass is unaccounted → region shares
    # sum well below 1). `reasoning` = the completion BEFORE output (the <think> span
    # + its delimiters); `output` = the emitted assignment span. Anything else (e.g.
    # the trailing end-of-turn) falls into region_segments' catch-all.
    frames: dict[str, Any] = {
        "input": {"start": 0, "end": completion_start, "raw_field": "user"},
        "output": {"start": out_off, "end": out_off + len(output), "raw_field": "output"},
    }
    if out_off > completion_start:
        frames["reasoning"] = {"start": completion_start, "end": out_off}
    trace = GenerationTrace(
        model_id=export["meta"]["model_id"],
        full_text=full,
        completion_start=completion_start,
        frames=frames,
        scene_map=scene_map,
        output_map=output_map,
        to_place_map=to_place_map,
        variables_map=export.get("variables_map", []),  # powers the "Variables" region category
        remote_logprobs=remote_logprobs,
        meta={**export.get("meta", {}), "reconstruction": check, "native_template": True},
    )
    return trace, check
