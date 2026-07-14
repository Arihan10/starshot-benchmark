"""Single-shot structured call to OpenRouter via `response_format: json_schema`.

The model is task-local: each `_run` task calls `set_model()` once with
the OpenRouter id for its alias, and every `call_llm()` on that task
inherits via a ContextVar. Concurrent runs against the same slot with
different models therefore don't race on a module global.

Up to 4 resamples on parse / validation failures.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from collections.abc import Awaitable, Callable, Iterable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar

import httpx
from openrouter import OpenRouter
from openrouter.errors import OpenRouterError
from pydantic import BaseModel, ValidationError

from app.utils import cache, logging

T = TypeVar("T", bound=BaseModel)

# Lift Python 3.11+'s 4300-digit ceiling on int<->str conversion.
# When an LLM hallucinates a runaway numeric literal into a structured-
# output JSON field, `json.loads` raises ValueError("Exceeds the limit")
# before pydantic ever sees the value, which means we burn a retry
# without a useful diagnostic. Letting json parse the giant int through
# moves the rejection into pydantic, where ValidationError surfaces the
# bad field name instead of a cryptic conversion error.
sys.set_int_max_str_digits(0)

_current_model: ContextVar[str | None] = ContextVar("_current_model", default=None)


def set_model(model: str) -> None:
    _current_model.set(model)


def current_model() -> str:
    """The model bound to the current task. Raises if `set_model` hasn't run —
    same contract as `call_llm`. Lets callers that use `call_llm_once` directly
    (which takes `model` explicitly) reuse the task-bound model."""
    model = _current_model.get()
    if model is None:
        raise RuntimeError("llm.set_model() must be called before current_model()")
    return model


# --- Custom OpenAI-compatible endpoints (bypass OpenRouter) -----------------
#
# The pipeline normally routes every call through the OpenRouter SDK. To
# benchmark a model served by our OWN OpenAI-compatible endpoint (a local
# vLLM/SGLang, a Modal gateway, a proxy, ...) WITHOUT it hitting OpenRouter or
# accruing spend, register it here.
#
# The hard filter is the model id itself: whatever `set_model()` binds (the
# VALUE side of `app.core.slots.MODELS`) is looked up in this registry on every
# call. A match short-circuits the OpenRouter transport in `call_llm_once` /
# `chat` and issues a plain httpx POST to `{base_url}/chat/completions` instead.
#
# Cost: a custom call is NEVER given a generation_id (there's no OpenRouter
# billing record to price), so `call_llm` logs `generation_id=None`, the cost
# backfill skips it, and `_usage_summary` counts it as a request at $0 with 0
# pending. Tokens are still surfaced when the endpoint returns a `usage` block.
#
# To activate one:
#   1. Add a CustomEndpoint below, keyed by the model id you'll route on
#      (e.g. "custom/local-model").
#   2. Point an alias at that same id in `app.core.slots.MODELS` so the
#      dashboard can select it (e.g. "custom": "custom/local-model").
# Attention analysis stays gated off for it (no open-weight spec), same as any
# closed model — see app.attention.models.resolve_open_model.


@dataclass(frozen=True)
class CustomEndpoint:
    """One OpenAI-compatible chat-completions backend that replaces OpenRouter
    for a specific model id."""

    # Model id sent in the request body (what the endpoint expects). May differ
    # from the registry key, which is the id the pipeline routes on.
    model: str
    # OpenAI-compatible API root, e.g. "http://localhost:8000/v1" (no trailing
    # slash). "/chat/completions" is appended.
    base_url: str
    # Env var holding the bearer key; None for auth-less endpoints.
    api_key_env: str | None = None
    # Generation cap; None defers to the endpoint's default.
    max_tokens: int | None = None
    # Extra request-body fields merged verbatim (e.g. {"reasoning": {"effort":
    # "high"}} or {"temperature": 0.7}). The endpoint owns these knobs — nothing
    # OpenRouter-specific (require_parameters, provider routing) is sent here.
    extra: dict[str, object] = field(default_factory=dict)


# Keyed by the model id `set_model()` binds (the value side of core.slots.MODELS).
# With no entries every call goes to OpenRouter exactly as before.
CUSTOM_ENDPOINTS: dict[str, CustomEndpoint] = {
    # Tencent Hy3 (Hy3-FP8) served by our own vLLM router on Modal. Auth-less
    # (the reference client's api_key "EMPTY" == no bearer, verified: requests
    # with no Authorization header return 200). hy3 is a reasoning model whose
    # thinking is a CHAT-TEMPLATE kwarg — it reasons when reasoning_effort !=
    # "no_think" and returns the trace on `message.reasoning` (NOT inline
    # <think> tags, verified live). Critically, guided decoding
    # (`response_format: json_schema`, strict) and reasoning COEXIST: content is
    # clean schema-conforming JSON while reasoning rides its own field — so the
    # all-structured pipeline works unchanged. $0 in the cost meter (a custom
    # call gets no generation_id). max_tokens is left unset so verbose reasoning
    # can't truncate the JSON body against a tight cap (262k context).
    "custom/hy3": CustomEndpoint(
        model="hy3",
        base_url="https://starshot-aitools--hy3-vllm-openai-router.modal.run/v1",
        api_key_env=None,
        extra={
            "chat_template_kwargs": {
                "reasoning_effort": "high",
                "enable_thinking": True,
            },
        },
    ),
}


def _custom_endpoint(model: str | None) -> CustomEndpoint | None:
    """The CustomEndpoint a model id routes to, or None to use OpenRouter."""
    return CUSTOM_ENDPOINTS.get(model) if model else None


# Custom endpoints may be local/cold-starting and form the whole response at
# once, so the read window is generous (mirrors the one-shot dLLM client);
# there's no token stream to keep the connection chatty.
_CUSTOM_TIMEOUT = httpx.Timeout(connect=30.0, read=900.0, write=60.0, pool=60.0)


# Model-specific runtime injections — quirks of particular providers, NOT
# prompt content, so they never appear in templates. Applied to the user
# message at the call boundary, BEFORE cache hashing and logging, so the
# cache key and the logged ground-truth bytes both reflect exactly what was
# sent. (Appending at the end matches where the old template token rendered,
# keeping cache keys byte-identical across the cutover.)
DEEPSEEK_INJECTION = """
<IMPORTANT_THINKING>
Be very opinionated in your thinking - do not go around in circles. Once a conclusion is reached, stick to it. Aim to end reasoning as soon as a concrete plan has been formed.
</IMPORTANT_THINKING>
"""


def apply_model_quirks(user: str, model: str) -> str:
    """The user message as it must actually leave for `model` — DeepSeek's
    reasoning tends to spiral, so it gets the opinionated-thinking pin.
    Idempotent: prompts rendered from pre-cutover snapshots/events may carry
    the injection already (it used to be a template variable)."""
    if model and "deepseek" in model.lower():
        if DEEPSEEK_INJECTION.strip() not in user:
            return user + DEEPSEEK_INJECTION
    return user


def _reasoning_effort(model: str) -> str:
    """The reasoning effort to request for `model`. OpenAI/GPT models are pinned
    to "minimal" for now — xhigh makes them slow + costly and we're not
    measuring their reasoning depth here; every other provider stays at "xhigh"."""
    return "medium" if (model or "").startswith("openai/") else "xhigh"


# Optional per-task breakpoint. When set, `call_llm` awaits it right before
# issuing a REAL (cache-miss) call. Downstream-simulation branch tasks bind
# one so the user advances them one step at a time; every other task leaves
# it None (the default), so the normal pipeline never blocks. Replayed
# prefix steps and seeded results cache-hit BEFORE the gate, so only the
# genuinely-new frontier pauses.
StepGate = Callable[..., Awaitable[None]]
_step_gate: ContextVar[StepGate | None] = ContextVar("_step_gate", default=None)


def set_step_gate(gate: StepGate | None) -> None:
    _step_gate.set(gate)


# How many alternative tokens to request per position when logprob capture is
# on. OpenRouter caps `top_logprobs` at 20; we request the max so the saved
# distribution is as complete as the provider will return.
TOP_LOGPROBS = 20

# Task-local logprob capture, bound once by `_run` for cells started with the
# "capture logprobs" toggle (mirrors `set_model`). When on, every real
# (cache-miss) `call_llm` asks OpenRouter for the top-`TOP_LOGPROBS` tokens at
# each output position and persists them next to the cell's events (see
# `_write_logprobs_sidecar`). Off by default so normal runs are byte-identical.
_capture_logprobs: ContextVar[bool] = ContextVar("_capture_logprobs", default=False)
# Set within a task once we learn its model has no provider that serves
# logprobs alongside our other required params (`require_parameters` filters
# them out → 404). Latches capture off for the rest of the run so we stop
# paying the failed first round-trip on every subsequent call.
_logprobs_unsupported: ContextVar[bool] = ContextVar("_logprobs_unsupported", default=False)


def set_capture_logprobs(on: bool) -> None:
    _capture_logprobs.set(on)


def _looks_like_logprobs_unsupported(e: Exception) -> bool:
    """Whether a provider error on a logprobs-enabled call is the routing/param
    rejection we should degrade past (retry without logprobs) rather than a real
    transport flap. With `provider.require_parameters` on, a model whose
    providers don't serve logprobs yields a 404 "no endpoints found that support
    your parameters"; some providers instead 400/422 with a logprobs complaint."""
    status = getattr(getattr(e, "raw_response", None), "status_code", None)
    if status == 404:
        return True
    body = getattr(e, "body", None)
    msg = f"{e} {body or ''}".lower()
    return any(
        kw in msg
        for kw in ("logprob", "no endpoints", "no allowed providers", "does not support")
    )


def _http_status(e: BaseException) -> int | None:
    """Best-effort backend HTTP status off an OpenRouter/httpx error, for the
    retry log's `code` field. None when the flap carried no response (a dropped
    connection or timeout) — those have no status to attribute it to."""
    for attr in ("raw_response", "response"):
        code = getattr(getattr(e, attr, None), "status_code", None)
        if isinstance(code, int):
            return code
    code = getattr(e, "status_code", None)
    return code if isinstance(code, int) else None


def _serialize_logprobs(choice: object) -> dict[str, object] | None:
    """Fold one choice's `logprobs.content` into a compact, self-describing map
    that stays aligned to the original response text.

    OpenRouter returns the output token stream in order, and concatenating the
    tokens reproduces the exact `message.content` bytes the model emitted (the
    same string we parse into structured output). We keep that ordering and
    additionally record each token's `[start, end)` character span into the
    reconstructed `text`, so a consumer can map any position in the original
    response to its chosen token and the alternatives considered there. Returns
    None when the provider sent no usable logprobs (best-effort capture)."""
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None)
    if not isinstance(content, list) or not content:
        return None
    tokens: list[dict[str, object]] = []
    parts: list[str] = []
    offset = 0
    for tok in content:
        token = getattr(tok, "token", None)
        if not isinstance(token, str):
            continue
        start = offset
        offset += len(token)
        parts.append(token)
        top: list[dict[str, object]] = []
        for alt in getattr(tok, "top_logprobs", None) or []:
            alt_token = getattr(alt, "token", None)
            alt_logprob = getattr(alt, "logprob", None)
            if isinstance(alt_token, str) and isinstance(alt_logprob, (int, float)):
                top.append({"token": alt_token, "logprob": float(alt_logprob)})
        logprob = getattr(tok, "logprob", None)
        tokens.append(
            {
                "start": start,
                "end": offset,
                "token": token,
                "logprob": float(logprob) if isinstance(logprob, (int, float)) else None,
                "top": top,
            }
        )
    if not tokens:
        return None
    return {"text": "".join(parts), "tokens": tokens}


def _write_logprobs_sidecar(
    key: str,
    logprobs_map: dict[str, object],
    *,
    model: str,
    step: str | None,
    node: str | None,
    schema_name: str,
    generation_id: str | None,
) -> bool:
    """Persist a call's captured logprobs beside the cell's events, keyed by the
    same content hash as its `cache.llm` event (so a cache hit reuses the same
    file and a rewind that rewinds the cache also orphans the right sidecar).
    The `cache.llm` event stays small — it only gets a `logprobs: true` flag —
    while the (potentially large) per-token distribution lives in
    `logprobs/<key>.json`, served through the normal `/artifacts` route.
    Best-effort: a write failure just means the flag isn't set."""
    try:
        directory = logging.slot_dir() / "logprobs"
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": key,
            "generation_id": generation_id,
            "model": model,
            "step": step,
            "node": node,
            "schema": schema_name,
            "top_n": TOP_LOGPROBS,
            "text": logprobs_map.get("text"),
            "tokens": logprobs_map.get("tokens"),
        }
        (directory / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")
        return True
    except (OSError, LookupError, KeyError, RuntimeError, TypeError):
        return False


class OutputValidationError(Exception):
    """A structured output parsed cleanly but failed a caller-supplied semantic
    check — e.g. a batch step echoing back ids that don't match the ones it was
    asked to place (a model-level output defect, not a pipeline decision).

    `call_llm` resamples on it up to ID_VALIDATION_MAX times, and because it is
    raised BEFORE the `cache.llm` event is written, no failing output is ever
    cached — so every resample (and any later manual retry) re-runs the call
    fresh instead of replaying the bad result."""


def require_matching_ids(
    *, produced: Iterable[str], expected: Iterable[str], step: str
) -> None:
    """Assert a batch step's output ids are exactly the ids it was given — no
    missing, extra, or duplicated ids — raising OutputValidationError otherwise.
    Catches the model mangling an id when echoing the request (e.g. emitting
    `foo_balustraded_1` for the requested `foo_balustrade_1`)."""
    produced_list = list(produced)
    want = set(expected)
    got = set(produced_list)
    if want != got or len(produced_list) != len(want):
        dupes = sorted({i for i in produced_list if produced_list.count(i) > 1})
        raise OutputValidationError(
            f"{step}: output ids != requested ids "
            f"(missing={sorted(want - got)}, "
            f"unexpected={sorted(got - want)}, duplicated={dupes})"
        )


async def call_llm(
    *,
    system: str,
    user: str,
    output_schema: type[T],
    node_id: str | None = None,
    step: str | None = None,
    template: str | None = None,
    variables: dict[str, str] | None = None,
    validate: Callable[[T], None] | None = None,
) -> T:
    """`system`/`user` are the EXACT bytes sent to the provider; they are
    logged verbatim so the event log stays ground truth for what the model
    saw. `template` is the prompt-template name that produced them (root and
    nested variants of a step differ here, while `step` stays the event-log
    step id) and `variables` the resolved values that were substituted —
    logged so the prompt lab can re-render a historical call against an
    edited template."""
    model = _current_model.get()
    if model is None:
        raise RuntimeError("llm.set_model() must be called before call_llm()")
    user_raw = user  # pre-quirk source; re-quirked if the gate re-aims this call
    user = apply_model_quirks(user_raw, model)
    schema_name = output_schema.__name__
    key = cache.hash_llm_call(
        model=model,
        system=system,
        user=user,
        schema_name=schema_name,
    )
    hit = cache.find_llm_cache_hit(logging.current_slot(), key)
    if hit is not None:
        cached = output_schema.model_validate(hit)
        # Post-fix runs never cache an output that fails `validate`, so a hit is
        # normally valid. A cell poisoned by an older run can still hold a bad
        # entry though — surface it as a clean OutputValidationError (→ run.error,
        # reset to re-run) rather than a cryptic downstream KeyError.
        if validate is not None:
            validate(cached)
        return cached

    # Step gate: a bound gate (downstream-simulation branches) pauses here —
    # AFTER the cache check, so committed/cached/seeded steps replay untouched
    # and only genuinely-new frontier steps stop for the user's go-ahead. The
    # gate may also RE-AIM this call at a different model (compare's per-step
    # "run this step on model X"): re-quirk for it, switch the task model (so
    # schema normalization + any later calls follow it), re-hash, and re-check
    # the cache under the new key — a revert cleared any prior output, so this
    # normally misses and the step runs fresh on the chosen model.
    gate = _step_gate.get()
    if gate is not None:
        # Gated (branch) replay: a step committed earlier in THIS log on a
        # different model (a per-step A/B pick, now replaying as a prefix) won't
        # match the model-bound key above, but its output is committed truth —
        # return it without re-running. Only the frontier (whose cache.llm was
        # truncated by the revert) finds nothing and falls through to the gate.
        replay = cache.find_llm_replay(
            logging.current_slot(), system=system, user=user, schema_name=schema_name,
        )
        if replay is not None:
            cached = output_schema.model_validate(replay)
            if validate is not None:
                validate(cached)
            return cached
        chosen = await gate(
            node_id=node_id,
            step=step,
            template=template,
            system=system,
            user=user,
            schema_name=schema_name,
            model=model,
        )
        if chosen and chosen != model:
            model = chosen
            _current_model.set(model)
            user = apply_model_quirks(user_raw, model)
            key = cache.hash_llm_call(
                model=model,
                system=system,
                user=user,
                schema_name=schema_name,
            )
            hit = cache.find_llm_cache_hit(logging.current_slot(), key)
            if hit is not None:
                cached = output_schema.model_validate(hit)
                if validate is not None:
                    validate(cached)
                return cached

    # Ablation: a variant's RE-INFERRED treated step samples with the treatment's
    # temperature + per-replicate seed, so duplicate re-runs (replicates) are
    # independent-but-reproducible draws. None on normal runs / non-treated steps
    # → provider default, byte-identical to before.
    from app.ablation import context as _abl_ctx
    sampling = _abl_ctx.sampling_for(template if template is not None else step)
    # XML-gravity ablation: rewrite the treated step's instruction-block XML tags
    # per the treatment (strip / move the closing <prompt> across word-count
    # quarters) and log the quarter+tag span-map under `variables["__GRAVITY__"]`.
    # Placed AFTER the cache + gate checks so ONLY the variant's re-inferred treated
    # firing is rewritten — replayed prefix firings cache-hit above and are
    # untouched. Re-hash so the re-inferred output caches + logs under the ACTUAL
    # sent user. A no-op (byte-identical) on normal runs / non-treated steps.
    from app.ablation import gravity as _abl_gravity
    _g_user = _abl_gravity.rewrite_and_stash(user, variables, template if template is not None else step)
    if _g_user != user:
        user = _g_user
        key = cache.hash_llm_call(model=model, system=system, user=user, schema_name=schema_name)
    validated, reasoning, usage, raw, generation_ids, logprobs_map = await call_llm_once(
        system=system,
        user=user,
        output_schema=output_schema,
        model=model,
        validate=validate,
        step=step,
        sampling=sampling,
    )
    # When logprob capture is on and the provider returned them, spill the
    # (potentially large) per-token distribution to a content-addressed sidecar
    # and flag the event — kept OUT of the log line so events.jsonl stays lean
    # and cheap to fold. `logprobs=True` tells the client a sidecar exists at
    # `logprobs/<key>.json`.
    logprobs_extra: dict[str, object] = {}
    if logprobs_map is not None and _write_logprobs_sidecar(
        key,
        logprobs_map,
        model=model,
        step=step,
        node=node_id,
        schema_name=schema_name,
        generation_id=generation_ids[-1] if generation_ids else None,
    ):
        logprobs_extra["logprobs"] = True
    # cache.llm carries everything needed for the LLM-call cache (key +
    # output), the observability view (node + step + model + system + user +
    # reasoning), and the prompt lab (template + variables). Older log lines
    # that lack the newer fields still replay correctly — the client treats
    # them as unattributed / not re-renderable.
    #
    # `generation_id` ties this call to OpenRouter's billing record. Its settled
    # USD cost lags the completion (and a live lookup would die with a restart),
    # so we DON'T resolve it here: the backfill sweep (`backfill_costs`) prices
    # it off the log and appends a separate `llm.cost` event, which
    # `_usage_summary` joins back by id.
    logging.log(
        "cache.llm",
        key=key,
        node=node_id,
        step=step,
        template=template,
        model=model,
        schema=schema_name,
        system=system,
        user=user,
        variables=variables,
        output=raw,
        reasoning=reasoning,
        tokens_in=getattr(usage, "prompt_tokens", None),
        tokens_out=getattr(usage, "completion_tokens", None),
        generation_id=generation_ids[-1] if generation_ids else None,
        **logprobs_extra,
    )
    # Ablation runs stop the moment their treated step is committed — no
    # downstream (image prompts, flash-lite library/prefab matching, meshes).
    # The ablation targets the TEMPLATE name (e.g. object_bbox_batch,
    # zone_decompose) — root/nested variants differ in `template` while sharing a
    # `step` id — so match on template (falling back to step). (_abl_ctx already
    # imported above for the sampling override.)
    _abl_ctx.stop_if_treated_step(template if template is not None else step)
    return validated


async def call_llm_once(
    *,
    system: str,
    user: str,
    output_schema: type[T],
    model: str,
    validate: Callable[[T], None] | None = None,
    step: str | None = None,
    log_retries: bool = True,
    sampling: dict[str, object] | None = None,
) -> tuple[T, str, object, object, list[str], dict[str, object] | None]:
    """One structured-output call with the full resample/backoff budget,
    WITHOUT the content-addressed cache lookup or the `cache.llm` log write
    that `call_llm` wraps around it. Returns
    `(validated, reasoning, usage, raw, generation_ids, logprobs)`, where
    `generation_ids` lists every billed attempt (a resample bills each try) and
    `logprobs` is the ordered top-token map (aligned to the response text) when
    capture is on and the provider served it, else None.

    The prompt-tuning sandbox (`POST /llm/test`) calls this directly so a
    throwaway "what if I edited this prompt" test re-runs the exact step's
    call yet never reads or mutates any cell's event log — its result is
    rendered transiently and discarded. `call_llm` is the cached, logged
    pipeline path. `log_retries=False` (the sandbox runs outside any cell's
    SlotLog binding) skips the per-attempt diagnostic events so no `logging`
    ContextVar is required."""

    # Hard filter: a registered custom id bypasses OpenRouter entirely and is
    # served by our own OpenAI-compatible endpoint at $0 cost (no generation_id).
    cfg = _custom_endpoint(model)
    if cfg is not None:
        return await _call_custom_once(
            cfg,
            system=system,
            user=user,
            output_schema=output_schema,
            model=model,
            validate=validate,
            step=step,
            log_retries=log_retries,
            sampling=sampling,
        )

    def _retry_log(kind: str, **data: object) -> None:
        if log_retries:
            logging.log(kind, **data)

    # Two independent budgets:
    #   * `parse_attempt` — JSON-decode / Pydantic-validation failures.
    #     The model misbehaved; resampling fixes it. 4 tries.
    #   * `transport_attempt` — provider returned a non-success body
    #     (Anthropic 503 "Overloaded", upstream timeouts, etc.) which the
    #     SDK surfaces as OpenRouterError / ResponseValidationError. The
    #     model never saw the request, so this isn't "the LLM was wrong" —
    #     it's a flap. Larger budget + longer backoff so a multi-second
    #     provider hiccup doesn't fail the run.
    parse_attempt = 0
    transport_attempt = 0
    id_validation_attempt = 0
    PARSE_MAX = 4
    TRANSPORT_MAX = 8
    # Auto-resample when a batch step echoes back ids that don't match its
    # request (see `validate`). Each attempt is a fresh call; exhausting the
    # budget raises a hard error (the bad output was never cached, so a manual
    # retry re-runs it too).
    ID_VALIDATION_MAX = 5
    TRANSPORT_BACKOFF = [2, 4, 8, 16, 30, 30, 30, 30]
    # Every billed generation for this logical call, in order. A resample bills
    # each attempt, so we collect the id of every response we received (even
    # ones we later reject) — `call_llm` sums their costs so the run total
    # matches what OpenRouter actually charged.
    generation_ids: list[str] = []
    while True:
        content: object = None
        # Recomputed each attempt: a prior attempt may have latched capture off
        # for this model (its providers don't serve logprobs under
        # `require_parameters`), in which case we retry clean.
        want_logprobs = _capture_logprobs.get() and not _logprobs_unsupported.get()
        logprobs_kwargs: dict[str, object] = (
            {"logprobs": True, "top_logprobs": TOP_LOGPROBS} if want_logprobs else {}
        )
        # Ablation sampling override (temperature only) — passed straight through to
        # the provider; empty on every normal call so routing is unchanged. NB: only
        # universally-supported params may go here — `provider.require_parameters`
        # (set below) 404s the call if the chosen provider lacks ANY sent param, so
        # e.g. `seed` (unsupported by most open-model providers) must NOT be added.
        sampling_kwargs: dict[str, object] = {}
        if sampling and sampling.get("temperature") is not None:
            sampling_kwargs["temperature"] = sampling["temperature"]
        try:
            async with OpenRouter(
                api_key=os.environ["OPENROUTER_API_KEY"],
                timeout_ms=180_000,
            ) as client:
                response = await client.chat.send_async(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": output_schema.__name__,
                            "strict": True,
                            "schema_": _normalize_schema(output_schema.model_json_schema()),
                        },
                    },
                    reasoning={"effort": _reasoning_effort(model)},
                    # Force routing to a provider that actually honors the
                    # parameters we send. Omitted, OpenRouter silently strips
                    # any param the chosen provider lacks — so a model whose
                    # provider can't do `response_format: json_schema` (GLM)
                    # falls back to a free-form completion and emits fenced /
                    # prose / null content instead of schema-conformant JSON.
                    # NOTE: this also gates `logprobs` — a model with no
                    # logprob-serving provider 404s, which we degrade past below.
                    provider={
                        "require_parameters": True,
                        "sort": "latency",
                        "ignore": ["decart"]
                    },
                    **logprobs_kwargs,
                    **sampling_kwargs,
                )
            # Recorded before parsing: this generation was billed regardless of
            # whether its output survives validation below.
            if getattr(response, "id", None):
                generation_ids.append(response.id)
            message = response.choices[0].message
            content = message.content
            args = json.loads(content) if isinstance(content, str) else content
            # A reasoning-heavy turn can finish with no answer body (`content`
            # null, or the bare literal `null`) — resample it with a clear reason
            # instead of letting `model_validate(None)` raise a cryptic
            # `model_type` error. finish_reason/refusal pin down truncation vs.
            # a content filter for the retry log.
            if args is None:
                refusal = message.refusal if isinstance(message.refusal, str) else None
                raise ValueError(
                    "model returned no content "
                    f"(finish_reason={response.choices[0].finish_reason!r}, refusal={refusal!r})"
                )
            validated = output_schema.model_validate(args)
            if validate is not None:
                # Semantic check (e.g. batch id echo). Raised BEFORE the
                # caller's cache.llm log, so a failing output is never cached
                # and each resample re-runs the call fresh.
                validate(validated)
            reasoning = getattr(message, "reasoning", None) or ""
            # Token counts feed the per-call trace panels only — the run's spend
            # is the authoritative `total_cost` `call_llm` pulls from the
            # /generation endpoint, not derived from these. `usage` is absent on
            # the rare provider that omits it; completion_tokens already includes
            # reasoning tokens, so no separate reasoning field is needed.
            usage = getattr(response, "usage", None)
            # Ordered top-token map aligned to the response text (None unless
            # capture is on and the provider served logprobs). Serialized here,
            # inside the client context, before the response is discarded.
            logprobs_map = _serialize_logprobs(response.choices[0]) if want_logprobs else None
            # Return the raw parsed response (`args`) alongside the validated
            # model so callers can log EXACTLY what the model emitted — the wire
            # field names and values — rather than a re-serialized, attribute-
            # named `model_dump` — plus every billed generation id so the caller
            # can price the call against OpenRouter's settled cost.
            return validated, reasoning, usage, args, generation_ids, logprobs_map
        except OutputValidationError as e:
            if id_validation_attempt >= ID_VALIDATION_MAX - 1:
                raise
            _retry_log(
                "llm.validation_retry",
                step=step,
                reason=str(e),
                attempt=id_validation_attempt,
            )
            id_validation_attempt += 1
        except json.JSONDecodeError as e:
            final = parse_attempt >= PARSE_MAX - 1
            _retry_log(
                "llm.json_decode_error",
                reason=f"JSONDecodeError: {e}",
                attempt=parse_attempt,
                final=final,
                content=content if isinstance(content, str) else repr(content),
            )
            if final:
                raise
            parse_attempt += 1
        except (OpenRouterError, httpx.HTTPError) as e:
            # Logprobs degrade: if THIS attempt asked for logprobs and the error
            # is the "no provider serves logprobs under our required params"
            # rejection, latch capture off for the run and retry immediately
            # (clean, no logprobs) — don't spend a transport attempt on it.
            if want_logprobs and _looks_like_logprobs_unsupported(e):
                _logprobs_unsupported.set(True)
                _retry_log(
                    "llm.logprobs_unsupported",
                    step=step,
                    model=model,
                    reason=str(e)[:200],
                )
                continue
            # httpx.RemoteProtocolError ("incomplete chunked read") and kin
            # surface when OpenRouter or an upstream provider drops the HTTP
            # connection mid-response. The SDK does not always wrap these as
            # OpenRouterError, so treat all httpx transport failures like the
            # other provider flaps we already retry.
            if transport_attempt >= TRANSPORT_MAX - 1:
                raise
            backoff = TRANSPORT_BACKOFF[
                min(transport_attempt, len(TRANSPORT_BACKOFF) - 1)
            ]
            _retry_log(
                "llm.transport_retry",
                reason=f"{type(e).__name__}: {str(e)[:200]}",
                attempt=transport_attempt,
                backoff_s=backoff,
                code=_http_status(e),  # backend HTTP status (429/503/…); None on a dropped connection
            )
            await asyncio.sleep(backoff)
            transport_attempt += 1
        except (ValidationError, ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            if parse_attempt >= PARSE_MAX - 1:
                raise
            _retry_log("llm.retry", reason=f"{type(e).__name__}: {str(e)[:160]}")
            parse_attempt += 1


class CustomEndpointError(Exception):
    """A custom OpenAI-compatible endpoint returned a non-retryable (4xx) error
    — a config problem (bad model id, auth, unsupported body), not a flap. It
    exhausts no resample/transport budget; the call fails fast."""


class _EmptyResponseError(httpx.HTTPError):
    """A self-hosted endpoint returned a 2xx with an EMPTY body/answer. On our
    Modal-hosted vLLM router that means the edge was cold-starting, the request
    timed out, or it was cancelled — the proxy forwards the empty 2xx through
    unchanged (it only absorbs 3xx/4xx/5xx transients). That's infrastructure,
    NOT the model emitting bad JSON, so we subclass httpx.HTTPError to retry it
    on the transport budget (longer, cold-start backoff) instead of spending the
    parse budget and mis-blaming the model with an `llm.json_decode_error`."""


def _custom_headers(cfg: CustomEndpoint) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if cfg.api_key_env is not None:
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise CustomEndpointError(
                f"{cfg.api_key_env} is not set — required for {cfg.model} at {cfg.base_url}"
            )
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _custom_reasoning(message: dict[str, object]) -> str:
    """The reasoning trace wherever an OpenAI-compatible endpoint puts it
    (`reasoning` or `reasoning_content`); empty when it emits none."""
    for k in ("reasoning", "reasoning_content"):
        v = message.get(k)
        if isinstance(v, str) and v:
            return v
    return ""


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _split_think_tags(content: str) -> tuple[str, str]:
    """Pull inline `<think>...</think>` reasoning out of `content`, returning
    `(reasoning, cleaned_content)` and unwrapping any `<answer>...</answer>`.

    A vLLM `--reasoning-parser` normally separates thinking into the `reasoning`
    field, leaving `content` clean (this is what hy3 does — verified live). This
    is the FALLBACK for a server/config that leaves the scaffold inline, so the
    JSON body still parses and the reasoning is still captured (mirrors the
    reference hy3 client). Caller only invokes it when `<think>` is present, so
    the normal clean-content path is byte-identical."""
    m = _THINK_RE.search(content)
    if m:
        reasoning, answer = m.group(1), content[m.end():]
    else:  # opened but never closed (shouldn't happen on a non-streamed body)
        reasoning, answer = content.split("<think>", 1)[1], ""
    answer = answer.replace("<answer>", "").replace("</answer>", "")
    return reasoning.strip(), answer.strip()


def _custom_usage(usage: object) -> object | None:
    """Wrap the endpoint's `usage` dict as an attribute object so `call_llm`'s
    `getattr(usage, "prompt_tokens", None)` reads it the same as an SDK usage.
    None when the endpoint omitted usage (tokens then show as blank, like a
    provider that skips usage on the OpenRouter path)."""
    if not isinstance(usage, dict):
        return None
    return SimpleNamespace(
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


async def _call_custom_once(
    cfg: CustomEndpoint,
    *,
    system: str,
    user: str,
    output_schema: type[T],
    model: str,
    validate: Callable[[T], None] | None = None,
    step: str | None = None,
    log_retries: bool = True,
    sampling: dict[str, object] | None = None,
) -> tuple[T, str, object, object, list[str], dict[str, object] | None]:
    """`call_llm_once` for a hard-filtered custom id: a plain httpx POST to
    `{cfg.base_url}/chat/completions` (OpenAI-compatible, `response_format:
    json_schema`) instead of the OpenRouter SDK, keeping the SAME
    parse/validate/transport resample budgets and the SAME return contract.

    The 5th tuple slot (`generation_ids`) is ALWAYS empty: a custom call has no
    OpenRouter billing record, so `call_llm` logs `generation_id=None`, the cost
    backfill skips it, and `_usage_summary` prices it at $0 with 0 pending. The
    6th (`logprobs`) is always None — the OpenRouter-specific top-token capture
    doesn't ride this path."""

    def _retry_log(kind: str, **data: object) -> None:
        if log_retries:
            logging.log(kind, **data)

    parse_attempt = 0
    transport_attempt = 0
    id_validation_attempt = 0
    PARSE_MAX = 4
    TRANSPORT_MAX = 8
    ID_VALIDATION_MAX = 5
    TRANSPORT_BACKOFF = [2, 4, 8, 16, 30, 30, 30, 30]

    url = f"{cfg.base_url}/chat/completions"
    headers = _custom_headers(cfg)
    body: dict[str, object] = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": output_schema.__name__,
                "strict": True,
                # Raw wire key is "schema" (the OpenRouter SDK's alias is
                # "schema_"); a plain OpenAI-compatible endpoint wants "schema".
                "schema": _normalize_schema(output_schema.model_json_schema()),
            },
        },
        "stream": False,
    }
    # Ablation temperature override, same as the OpenRouter path; the endpoint
    # owns whether it honors it. Absent on every normal call.
    if sampling and sampling.get("temperature") is not None:
        body["temperature"] = sampling["temperature"]
    if cfg.max_tokens is not None:
        body["max_tokens"] = cfg.max_tokens
    if cfg.extra:
        body.update(cfg.extra)

    while True:
        content: object = None
        finish_reason: object = None
        status: object = None  # backend HTTP status, for the retry log's `code`
        try:
            async with httpx.AsyncClient(timeout=_CUSTOM_TIMEOUT) as client:
                res = await client.post(url, headers=headers, json=body)
            status = res.status_code
            # 5xx / cold-start hiccups are flaps (the model never saw the
            # request) → transport retry; 4xx is a config error → fail fast.
            if res.status_code >= 500:
                raise httpx.HTTPError(f"HTTP {res.status_code}: {res.text[:500]}")
            if res.status_code >= 400:
                raise CustomEndpointError(
                    f"{cfg.model} HTTP {res.status_code}: {res.text[:1000]}"
                )
            # A 2xx whose BODY isn't a real completion — EMPTY/blank, or not even
            # valid JSON — is the vLLM/Modal edge cold-starting, timing out, or the
            # request being cancelled (the proxy forwards the 2xx through unchanged,
            # only absorbing 3xx/4xx/5xx). Infrastructure, not the model: retry on
            # the transport budget instead of mis-blaming it as json_decode_error.
            # (A malformed *content* string below — the model's own answer — still
            # counts as a genuine parse failure.)
            if not (res.text or "").strip():
                raise _EmptyResponseError(f"empty HTTP {res.status_code} body")
            try:
                response = res.json()
            except json.JSONDecodeError as e:
                raise _EmptyResponseError(
                    f"non-JSON HTTP {res.status_code} body: {res.text[:160]!r}"
                ) from e
            choice = (response.get("choices") or [{}])[0]
            finish_reason = choice.get("finish_reason")
            message = choice.get("message") or {}
            content = message.get("content")
            reasoning = _custom_reasoning(message)
            # Fallback for a server that leaves the reasoning scaffold inline
            # instead of on `message.reasoning` — strip it so the JSON parses.
            # No-op for hy3 (clean content), so the normal path is untouched.
            if isinstance(content, str) and "<think>" in content:
                inline_reasoning, content = _split_think_tags(content)
                reasoning = reasoning or inline_reasoning
            # An empty / whitespace-only answer (a 2xx that never carried a real
            # completion) is the same cold-start / cancel / timeout transient as
            # an empty HTTP body — retry on the transport budget, NOT as bad JSON.
            # The exception is finish_reason="length": the model DID answer but
            # its (long) reasoning truncated the JSON — a real budget problem, so
            # that stays on the parse path below.
            if (
                content is None or (isinstance(content, str) and not content.strip())
            ) and finish_reason != "length":
                raise _EmptyResponseError(
                    f"empty answer body (HTTP {res.status_code}, finish_reason={finish_reason!r})"
                )
            args = json.loads(content) if isinstance(content, str) else content
            if args is None:
                refusal = message.get("refusal")
                raise ValueError(
                    "model returned no content "
                    f"(finish_reason={choice.get('finish_reason')!r}, "
                    f"refusal={refusal if isinstance(refusal, str) else None!r})"
                )
            validated = output_schema.model_validate(args)
            if validate is not None:
                validate(validated)
            usage = _custom_usage(response.get("usage"))
            # Empty generation_ids → $0 in the cost meter (see docstring).
            return validated, reasoning, usage, args, [], None
        except OutputValidationError as e:
            if id_validation_attempt >= ID_VALIDATION_MAX - 1:
                raise
            _retry_log(
                "llm.validation_retry",
                step=step,
                reason=str(e),
                attempt=id_validation_attempt,
            )
            id_validation_attempt += 1
        except CustomEndpointError:
            raise  # 4xx: config problem — no retry would help
        except json.JSONDecodeError as e:
            final = parse_attempt >= PARSE_MAX - 1
            # finish_reason is the key attribution signal here: "length" = the
            # (very long) reasoning ate the token budget so the JSON answer was
            # truncated/empty; "stop" with empty content = the model ended after
            # reasoning without emitting the answer. Logged so empties self-explain.
            _retry_log(
                "llm.json_decode_error",
                step=step,
                reason=f"JSONDecodeError: {e}",
                attempt=parse_attempt,
                final=final,
                finish_reason=finish_reason,
                code=status,
                content=content if isinstance(content, str) else repr(content),
            )
            if final:
                raise
            parse_attempt += 1
        except httpx.HTTPError as e:
            if transport_attempt >= TRANSPORT_MAX - 1:
                raise
            backoff = TRANSPORT_BACKOFF[min(transport_attempt, len(TRANSPORT_BACKOFF) - 1)]
            _retry_log(
                "llm.transport_retry",
                reason=f"{type(e).__name__}: {str(e)[:200]}",
                attempt=transport_attempt,
                backoff_s=backoff,
                code=status,  # backend HTTP status (empty-body 2xx, 5xx flap, …); None on a dropped connection
            )
            await asyncio.sleep(backoff)
            transport_attempt += 1
        except (ValidationError, ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            if parse_attempt >= PARSE_MAX - 1:
                raise
            _retry_log("llm.retry", reason=f"{type(e).__name__}: {str(e)[:160]}")
            parse_attempt += 1


# Cost resolution runs OFF the pipeline: every `cache.llm` event carries its
# `generation_id`, and a backfill sweep (driven on a timer by the API layer)
# prices any that still lack an `llm.cost`. This survives restarts and the
# /generation stats lagging the completion — the sweep cadence IS the retry, so
# each lookup is a single best-effort attempt.
_COST_FETCH_CONCURRENCY = 6


async def fetch_generation_costs(generation_ids: Iterable[str]) -> dict[str, float]:
    """Best-effort `GET /generation` lookup of OpenRouter's settled `total_cost`
    for each (deduped) id, over one shared client with bounded concurrency.
    Returns only the ids that resolved; a 404 (stats not ready yet) or transient
    error simply omits that id, for the caller to retry on its next sweep. The
    SDK's own retry is disabled — its default budget is an hour."""
    ids = list(dict.fromkeys(generation_ids))
    if not ids:
        return {}
    out: dict[str, float] = {}
    sem = asyncio.Semaphore(_COST_FETCH_CONCURRENCY)

    async def _one(client: OpenRouter, gid: str) -> None:
        async with sem:
            try:
                res = await client.generations.get_generation_async(
                    id=gid, retries=None, timeout_ms=30_000,
                )
            except (OpenRouterError, httpx.HTTPError):
                return  # not settled yet / transient — next sweep retries
            out[gid] = res.data.total_cost

    async with OpenRouter(
        api_key=os.environ["OPENROUTER_API_KEY"], timeout_ms=30_000,
    ) as client:
        await asyncio.gather(*(_one(client, gid) for gid in ids))
    return out


async def backfill_costs(slot_logs: Iterable[logging.SlotLog]) -> int:
    """Price every logged-but-unpriced LLM call across the given cells: find each
    `cache.llm` carrying a `generation_id` with no matching `llm.cost`, fetch the
    settled cost, and append an `llm.cost` event. Idempotent and restart-proof —
    it reads only the durable log, so a call left unpriced by a slow stat or a
    live lookup killed mid-flight is recovered on a later pass. Returns the count
    priced this pass."""
    # Resolve the pending set up front: the fetch awaits, and the pipeline keeps
    # appending to these same logs meanwhile — anything new is caught next pass.
    pending: list[tuple[logging.SlotLog, str]] = []
    for sl in slot_logs:
        events = sl.state["events"]
        resolved = {
            e.get("generation_id") for e in events if e.get("kind") == "llm.cost"
        }
        for e in events:
            if e.get("kind") != "cache.llm":
                continue
            gid = e.get("generation_id")
            if isinstance(gid, str) and gid not in resolved:
                pending.append((sl, gid))
    if not pending:
        return 0
    costs = await fetch_generation_costs(gid for _, gid in pending)
    priced = 0
    for sl, gid in pending:
        cost = costs.get(gid)
        if cost is not None:
            sl.log("llm.cost", generation_id=gid, cost=cost)
            priced += 1
    return priced


async def chat(
    *,
    model: str,
    system: str,
    messages: list[dict[str, str]],
    reasoning_effort: str = "xhigh",
) -> tuple[str, str]:
    """A free-form, multi-turn completion with NO structured-output schema —
    the engine behind the decision-inquiry endpoint. `system` is prepended as
    the system message; `messages` is the running conversation (alternating
    user/assistant turns). Returns `(text, reasoning)`.

    Unlike `call_llm`/`call_llm_once` this neither forces `response_format`
    nor reads/writes any cache or event log, and `model` is passed explicitly
    rather than via the `_current_model` ContextVar — it runs outside any
    cell's pipeline. It keeps only the transport-retry budget (provider flaps
    like Anthropic 503s); there is no JSON to parse, so no parse/validation
    resampling."""
    cfg = _custom_endpoint(model)
    if cfg is not None:
        return await _chat_custom(
            cfg, system=system, messages=messages, reasoning_effort=reasoning_effort
        )
    transport_attempt = 0
    TRANSPORT_MAX = 8
    TRANSPORT_BACKOFF = [2, 4, 8, 16, 30, 30, 30, 30]
    wire: list[dict[str, str]] = [{"role": "system", "content": system}, *messages]
    while True:
        try:
            async with OpenRouter(
                api_key=os.environ["OPENROUTER_API_KEY"],
                timeout_ms=180_000,
            ) as client:
                response = await client.chat.send_async(
                    model=model,
                    # The SDK types messages as a TypedDict list; our runtime-built
                    # role/content dicts are structurally identical (same friction
                    # the structured path hits on response_format).
                    messages=wire,  # pyright: ignore[reportArgumentType]
                    reasoning={"effort": reasoning_effort},
                )
            message = response.choices[0].message
            content = message.content
            text = content if isinstance(content, str) else str(content or "")
            reasoning = getattr(message, "reasoning", None) or ""
            return text, reasoning
        except (OpenRouterError, httpx.HTTPError):
            # Provider flap (Anthropic 503, dropped connection) — the model
            # never saw the request, so back off and resend, same budget as
            # call_llm_once's transport retries.
            if transport_attempt >= TRANSPORT_MAX - 1:
                raise
            backoff = TRANSPORT_BACKOFF[min(transport_attempt, len(TRANSPORT_BACKOFF) - 1)]
            await asyncio.sleep(backoff)
            transport_attempt += 1


async def _chat_custom(
    cfg: CustomEndpoint,
    *,
    system: str,
    messages: list[dict[str, str]],
    reasoning_effort: str = "xhigh",
) -> tuple[str, str]:
    """`chat` for a hard-filtered custom id: a plain httpx POST to
    `{cfg.base_url}/chat/completions` with no `response_format`, same
    transport-retry budget, same `(text, reasoning)` return. `reasoning_effort`
    is advisory here — the endpoint's own reasoning knobs come from `cfg.extra`
    (nothing OpenRouter-shaped is assumed)."""
    transport_attempt = 0
    TRANSPORT_MAX = 8
    TRANSPORT_BACKOFF = [2, 4, 8, 16, 30, 30, 30, 30]
    url = f"{cfg.base_url}/chat/completions"
    headers = _custom_headers(cfg)
    body: dict[str, object] = {
        "model": cfg.model,
        "messages": [{"role": "system", "content": system}, *messages],
        "stream": False,
    }
    if cfg.max_tokens is not None:
        body["max_tokens"] = cfg.max_tokens
    if cfg.extra:
        body.update(cfg.extra)
    while True:
        try:
            async with httpx.AsyncClient(timeout=_CUSTOM_TIMEOUT) as client:
                res = await client.post(url, headers=headers, json=body)
            if res.status_code >= 500:
                raise httpx.HTTPError(f"HTTP {res.status_code}: {res.text[:500]}")
            if res.status_code >= 400:
                raise CustomEndpointError(
                    f"{cfg.model} HTTP {res.status_code}: {res.text[:1000]}"
                )
            response = res.json()
            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content")
            text = content if isinstance(content, str) else str(content or "")
            reasoning = _custom_reasoning(message)
            if "<think>" in text:
                inline_reasoning, text = _split_think_tags(text)
                reasoning = reasoning or inline_reasoning
            return text, reasoning
        except httpx.HTTPError:
            if transport_attempt >= TRANSPORT_MAX - 1:
                raise
            backoff = TRANSPORT_BACKOFF[min(transport_attempt, len(TRANSPORT_BACKOFF) - 1)]
            await asyncio.sleep(backoff)
            transport_attempt += 1


def _normalize_schema(schema: object) -> object:
    """Recursively normalize the Pydantic-emitted schema for providers that
    reject draft-2020-12 features. Transforms:

      * Drop `minItems`/`maxItems`/`minimum`/`maximum`/`default` —
        Anthropic rejects them on `array` / `integer` / etc.
      * Collapse `prefixItems` (Pydantic emits this for fixed-length
        tuples like `tuple[float, float, float]`) into a single `items`
        schema. Anthropic rejects `prefixItems` outright. We assume
        homogeneous tuples (all our tuples are `Vec3` of floats); the
        first prefix item is reused as `items`.
      * Inject `additionalProperties: false` on every object node —
        OpenAI strict mode requires it; Anthropic accepts it.

    Pydantic still enforces the original constraints on the parsed
    response, so loosening the wire schema is safe."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k in {"minItems", "maxItems", "minimum", "maximum", "default"}:
                continue
            if k == "prefixItems":
                if isinstance(v, list) and v and "items" not in schema:
                    out["items"] = _normalize_schema(v[0])
                continue
            out[k] = _normalize_schema(v)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            # OpenAI strict mode and Google both require every property to
            # appear in `required`. Pydantic still validates the parsed
            # response, so widening the wire schema is safe.
            out["required"] = sorted(out["properties"].keys())
        current = _current_model.get()
        if (
            current and current.startswith("google/")
            and out.get("type") == "integer"
            and "enum" in out
        ):
            out["type"] = "string"
            out["enum"] = [str(v) for v in out["enum"]]
        return out
    if isinstance(schema, list):
        return [_normalize_schema(v) for v in schema]
    return schema
