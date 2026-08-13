"""Single-shot structured LLM call via `response_format: json_schema`.

Most models route through OpenRouter; ids registered in
`slots.OPENAI_COMPAT_MODELS` (e.g. LongCat) are sent straight to their own
OpenAI-compatible `/chat/completions` endpoint instead. Only the transport
differs — cache, resample/transport retries, reasoning + token capture, and
the compare gate are shared by both.

The model is task-local: each `_run` task calls `set_model()` once with
the model id for its alias, and every `call_llm()` on that task inherits via
a ContextVar. Concurrent runs against the same slot with different models
therefore don't race on a module global.

Up to 4 resamples on parse / validation failures.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable, Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, TypeVar

import httpx
from openrouter import OpenRouter
from openrouter.errors import OpenRouterError
from pydantic import BaseModel, ValidationError

from app.core.slots import (
    NO_STRUCTURED_OUTPUT_LIST,
    OPENAI_COMPAT_MODELS,
    OpenAICompatModel,
    call_reasoning,
    model_pricing,
    token_cost,
)
from app.utils import cache, flightlog, keypool, logging

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


# --- graceful ("finish current, start no more") pause ------------------------
# A soft pause, distinct from the hard cancel: `request_soft_pause` marks a cell's
# slot id so `call_llm` stops STARTING new calls (it parks until the task is later
# cancelled), while every call already in flight is left to finish and commit its
# `cache.llm`. `_inflight` counts logical calls in progress per cell so the
# /pause-soft handler can wait for that drain before cancelling the (now idle)
# task and marking the run paused. Keyed by the composite slot id
# `logging.current_slot_id()` returns (== `SlotLog.slot_id` == `_run_id(...)`).
_soft_paused: set[str] = set()
_inflight: dict[str, int] = {}


def request_soft_pause(slot_id: str) -> None:
    """Stop `slot_id` from STARTING new LLM calls; in-flight ones still finish."""
    _soft_paused.add(slot_id)


def clear_soft_pause(slot_id: str) -> None:
    """Lift the soft pause so a relaunch / resume can issue calls again."""
    _soft_paused.discard(slot_id)


def is_soft_paused(slot_id: str | None) -> bool:
    return slot_id is not None and slot_id in _soft_paused


async def await_drain(
    slot_id: str, *, poll_s: float = 0.2, timeout_s: float | None = None
) -> None:
    """Block until no logical LLM call is in flight for `slot_id`. A soft pause
    stops new calls from starting, so the count only falls — this returns once the
    currently-running call(s) have returned AND committed. `timeout_s=None` waits
    as long as the longest in-flight call needs (which is the whole point)."""
    waited = 0.0
    while _inflight.get(slot_id, 0) > 0:
        await asyncio.sleep(poll_s)
        if timeout_s is not None:
            waited += poll_s
            if waited >= timeout_s:
                return


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
    zone_id: str | None = None,
    step: str | None = None,
    template: str | None = None,
    variables: dict[str, str] | None = None,
    validate: Callable[[T], None] | None = None,
) -> T:
    """`system`/`user` are the EXACT bytes sent to the provider; they are
    logged verbatim so the event log stays ground truth for what the model
    saw. `template` is the prompt-template name that produced them (root and
    nested variants of a step differ here, while `step` stays the event-log
    step id), `zone_id` is the owning/target region independent of `node_id`,
    and `variables` are the resolved values that were substituted — logged so
    the prompt lab can re-render a historical call against an edited template."""
    if zone_id is None and variables is not None:
        candidate = variables.get("ZONE_ID")
        zone_id = candidate or None
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
    hit = cache.find_llm_cache_hit(logging.current_events(), key)
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
            logging.current_events(), system=system, user=user, schema_name=schema_name,
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
            hit = cache.find_llm_cache_hit(logging.current_events(), key)
            if hit is not None:
                cached = output_schema.model_validate(hit)
                if validate is not None:
                    validate(cached)
                return cached

    # The thinking effort this call asks for. Resolved AFTER the gate, so a
    # re-aimed model brings its own level, and keyed on `template or step` — the
    # step identity the log and dashboard already group calls on, which is what
    # the per-step flow's `STEP_REASONING` is written against.
    effort = call_reasoning(model, template or step)

    # Graceful pause: while a soft pause is active, DON'T start a new call — park
    # until the task is cancelled (the /pause-soft handler cancels once in-flight
    # calls have drained + committed). Parking (rather than raising) keeps sibling
    # in-flight calls in the same gather from being torn down. `_inflight` spans
    # the call AND its cache.llm commit below, so a call that started always
    # finishes and is recorded before the count can reach zero.
    slot = logging.current_slot_id()
    if is_soft_paused(slot):
        await asyncio.Event().wait()  # never set → parks until CancelledError
    if slot is not None:
        _inflight[slot] = _inflight.get(slot, 0) + 1
    try:
        validated, reasoning, usage, raw, generation_ids = await call_llm_once(
            system=system,
            user=user,
            output_schema=output_schema,
            model=model,
            reasoning_effort=effort,
            validate=validate,
            step=step,
            node_id=node_id,
            zone_id=zone_id,
        )
        # cache.llm carries everything needed for the LLM-call cache (key +
        # output), the observability view (node + step + model + reasoning_effort +
        # system + user + reasoning), and the prompt lab (template + variables).
        # Older log lines that lack the newer fields still replay correctly — the
        # client treats them as unattributed / not re-renderable.
        #
        # `generation_id` ties this call to OpenRouter's billing record. Its settled
        # USD cost lags the completion (and a live lookup would die with a restart),
        # so we DON'T resolve it here: the backfill sweep (`backfill_costs`) prices
        # it off the log and appends a separate `llm.cost` event, which
        # `_usage_summary` joins back by id.
        #
        # `flight` is the ledger row of the FINAL (successful) HTTP attempt — its
        # request/response wall-clock times and flight duration land on the event so
        # the trace panels show call latency next to the tokens. Additive: older
        # logs without these fields replay unchanged.
        flight = flightlog.last_flight()
        tokens_in = getattr(usage, "prompt_tokens", None)
        tokens_out = getattr(usage, "completion_tokens", None)
        logging.log(
            "cache.llm",
            key=key,
            # Model-independent input hash: lets a committed step replay across a
            # model swap (find_llm_replay) without keeping system/user bytes in RAM.
            replay_key=cache.hash_llm_replay_key(
                system=system, user=user, schema_name=schema_name
            ),
            node=node_id,
            zone_id=zone_id,
            step=step,
            template=template,
            model=model,
            # The level actually requested — the per-step flow's audit trail, and
            # the only record of it, since the cache key doesn't bind the effort.
            reasoning_effort=effort,
            schema=schema_name,
            system=system,
            user=user,
            variables=variables,
            output=raw,
            reasoning=reasoning,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            generation_id=generation_ids[-1] if generation_ids else None,
            t_request=flight["t_request"] if flight else None,
            t_response=flight["t_response"] if flight else None,
            flight_ms=flight["flight_ms"] if flight else None,
            attempts=flight["attempt"] if flight else None,
        )
        # Statically-priced compat backends (Moonshot, Alibaba, ...) return no
        # per-request cost, so price this call from its token counts NOW and
        # record it as an `llm.cost` keyed by the content-hash `key` — the
        # external analogue of OpenRouter's `generation_id`. `_usage_summary`
        # prefers this key-join over any OpenRouter cost for the same call, which
        # is what makes a BYOK leg (billed at $0 through OpenRouter) settle at its
        # true token cost. Unpriced models emit nothing here and fall through to
        # the settled-cost sweep as before.
        cost = token_cost(model, tokens_in, tokens_out)
        if cost is not None:
            logging.log(
                "llm.cost", key=key, cost=cost, model=model,
                tokens_in=tokens_in, tokens_out=tokens_out,
            )
        return validated
    finally:
        if slot is not None:
            _inflight[slot] = max(0, _inflight.get(slot, 1) - 1)


# --- transport dispatch -------------------------------------------------------


@dataclass(frozen=True)
class _Completion:
    """One structured response, normalized across transports so
    `call_llm_once`'s parse/validate/resample loop is backend-agnostic.
    `generation_id` is OpenRouter's billed id (summed into the run's settled USD
    cost); it is None for third-party OpenAI-compatible backends, which have no
    `/generation` record — those calls still log token counts, but carry no cost
    and never show as "resolving"."""

    content: object
    reasoning: str
    usage: object
    generation_id: str | None
    finish_reason: object = None
    refusal: str | None = None


# Read budget for the direct OpenAI-compatible transport. The call is
# non-streamed, so this caps the WHOLE generation — a thinking-enabled model
# forms its entire body before responding. Set high (1.5h) because Kimi K3 at
# reasoning_effort=max can genuinely take many minutes to complete.
_OPENAI_COMPAT_TIMEOUT = httpx.Timeout(connect=30.0, read=7200.0, write=60.0, pool=60.0)

# OpenRouter SDK per-request read budget (ms in the SDK). Kept as a constant so
# the transport-retry diagnostics can report the cap a timeout actually hit.
_OPENROUTER_TIMEOUT_S = 180.0

# 429 (rate-limit) retry policy — separate from the general transport-flap
# budget. Every 429 gets EXPONENTIAL backoff, but a rotating key pool defers the
# sleep until a whole sweep of the pool has 429'd (each key tried once): keys
# are rolled with NO delay between them, and only an all-keys-exhausted sweep
# backs off. OpenRouter and single-key compat models have an effective pool size
# of 1, so every 429 backs off — the identical arithmetic, `pool_size == 1`.
RATE_LIMIT_MAX = 8
_RATE_LIMIT_BASE_S = 2.0
_RATE_LIMIT_CAP_S = 60.0


def _http_status(exc: Exception) -> int | None:
    """The HTTP status behind a transport error, or None for a status-less flap
    (timeout, dropped connection). A 429 surfaces as `OpenRouterError.status_code`
    on the SDK path and `HTTPStatusError.response.status_code` on the direct
    httpx (compat) path; a bare `ReadTimeout`/`RemoteProtocolError` carries no
    status and stays a general flap."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    if isinstance(exc, OpenRouterError):
        return getattr(exc, "status_code", None)
    return None


def _rate_limit_backoff(sweep_index: int) -> float:
    """Exponential backoff (seconds) for the `sweep_index`-th (0-based) all-keys-
    exhausted sweep: base·2^i, capped at `_RATE_LIMIT_CAP_S`."""
    return min(_RATE_LIMIT_BASE_S * (2**sweep_index), _RATE_LIMIT_CAP_S)


def _rate_limit_pool_size(model: str) -> int:
    """Keys to try before a 429 backs off: a rotating compat model's live pool
    size, else 1 (OpenRouter / single-key compat — back off on every 429)."""
    cfg = OPENAI_COMPAT_MODELS.get(model)
    if cfg is not None and cfg.rotate and cfg.api_key_env is not None:
        return keypool.get_pool(cfg.api_key_env).size
    return 1


async def _send_openai_compatible(
    cfg: OpenAICompatModel,
    *,
    system: str,
    user: str,
    schema_name: str,
    wire_schema: object,
) -> _Completion:
    """One non-streamed `POST {base_url}/chat/completions` to a third-party
    OpenAI-compatible endpoint, returning the same normalized shape as the
    OpenRouter path. `response_format: json_schema` is sent because the
    pipeline's prompts carry no free-form JSON instructions — the schema param
    IS the structure contract. Sampling knobs + `extra` (e.g. `thinking` /
    `reasoning_effort`) come from the model config; a non-2xx raises
    `httpx.HTTPStatusError`, which `call_llm_once` folds into its transport-retry
    budget alongside the SDK's provider flaps. The generous read timeout
    (`_OPENAI_COMPAT_TIMEOUT`, 45 min) covers a multi-minute thinking generation
    forming its whole body before the response arrives.

    `rotate`-enabled models draw their bearer key from the shared per-provider
    `keypool` instead of the plain env var; a 429 reports the key's generation
    back to the pool (first reporter advances it, concurrent duplicates no-op)
    and still raises — the transport retry re-enters here and picks up the
    pool's new active key."""
    body: dict[str, object] = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": wire_schema},
        },
        "stream": False,
    }
    if cfg.max_tokens is not None:
        body["max_tokens"] = cfg.max_tokens
    if cfg.temperature is not None:
        body["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        body["top_p"] = cfg.top_p
    body.update(cfg.extra)
    headers = {"Content-Type": "application/json"}
    pool: keypool.KeyPool | None = None
    generation = 0
    api_key: str | None = None
    if cfg.api_key_env is not None:
        if cfg.rotate:
            pool = keypool.get_pool(cfg.api_key_env)
            generation, api_key = pool.current()
        else:
            api_key = os.environ.get(cfg.api_key_env, "")
            if not api_key:
                raise RuntimeError(
                    f"{cfg.api_key_env} is not set — required for {cfg.model} at {cfg.base_url}"
                )
        headers["Authorization"] = f"Bearer {api_key}"
    t_request = time.time()
    try:
        async with httpx.AsyncClient(timeout=_OPENAI_COMPAT_TIMEOUT) as client:
            res = await client.post(
                f"{cfg.base_url}/chat/completions", headers=headers, json=body,
            )
    except httpx.HTTPError as e:
        # Status-less flap (timeout, dropped connection) — the flight still
        # happened, so it still gets a ledger row.
        flightlog.record(
            transport="direct", model=cfg.model, base_url=cfg.base_url, api_key=api_key,
            error=f"{type(e).__name__}: {e}", exc_type=type(e).__name__,
            t_request=t_request, t_response=time.time(),
        )
        raise
    t_response = time.time()
    # Parse the success body BEFORE recording so the row carries token usage; a
    # 200 with an unparseable body is recorded, then raised into the parse-retry
    # budget exactly where `res.json()` used to raise.
    data: Any = None
    body_error: json.JSONDecodeError | None = None
    if res.is_success:
        try:
            data = res.json()
        except json.JSONDecodeError as e:
            body_error = e
    usage_raw = (data.get("usage") or {}) if isinstance(data, dict) else {}
    flightlog.record(
        transport="direct", model=cfg.model, base_url=cfg.base_url, api_key=api_key,
        status=res.status_code,
        error=(f"unparseable response body: {body_error}" if body_error
               else None if res.is_success else res.text[:500]),
        tokens_in=usage_raw.get("prompt_tokens"),
        tokens_out=usage_raw.get("completion_tokens"),
        t_request=t_request, t_response=t_response,
    )
    if pool is not None and res.status_code == 429:
        # Roll the pool to the next key, then raise into the transport-retry
        # budget as before — the retried attempt re-enters this function and
        # picks up the pool's new active key. The generation CAS makes a burst
        # of parallel 429s on one key advance the pool exactly once.
        pool.rotate(generation)
    res.raise_for_status()
    if body_error is not None:
        raise body_error
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    # Attribute access so `call_llm` reads it exactly like the SDK's typed usage.
    usage = SimpleNamespace(
        prompt_tokens=usage_raw.get("prompt_tokens"),
        completion_tokens=usage_raw.get("completion_tokens"),
    )
    # Answer channel is `content`; the thinking trace is `reasoning_content`
    # (LongCat / Kimi) or `reasoning`. But LongCat returns the schema-conformant
    # JSON ON `reasoning_content` and leaves `content` empty whenever a
    # `response_format` is set (verified live, thinking on OR off) — so when
    # `content` is blank, take the reasoning text AS the answer (it IS the
    # answer, not a separate trace, hence no reasoning is reported then).
    content = message.get("content")
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if not isinstance(reasoning, str):
        reasoning = ""
    if (content is None or (isinstance(content, str) and not content.strip())) and reasoning:
        content, reasoning = reasoning, ""
    return _Completion(
        content=content,
        reasoning=reasoning,
        usage=usage,
        generation_id=None,
        finish_reason=choice.get("finish_reason"),
    )


async def _send_structured(
    *,
    model: str,
    system: str,
    user: str,
    output_schema: type[BaseModel],
    reasoning_effort: str,
) -> _Completion:
    """Issue ONE structured request for `model`, dispatching by transport: an id
    registered in `OPENAI_COMPAT_MODELS` goes straight to its OpenAI-compatible
    `base_url`; every other id routes through the OpenRouter SDK (unchanged).
    Both return a normalized `_Completion`.

    `reasoning_effort` is the thinking level to request, already resolved by the
    caller (per-step under that flow, else the model's own level). It applies to
    the OpenRouter path only: a compat backend spells thinking its own way in its
    config's `extra`, which `_send_openai_compatible` sends verbatim."""
    schema_name = output_schema.__name__
    wire_schema = _normalize_schema(output_schema.model_json_schema())
    cfg = OPENAI_COMPAT_MODELS.get(model)
    if cfg is not None:
        return await _send_openai_compatible(
            cfg, system=system, user=user, schema_name=schema_name, wire_schema=wire_schema,
        )
    t_request = time.time()
    try:
        async with OpenRouter(
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout_ms=int(_OPENROUTER_TIMEOUT_S * 1000),
        ) as client:
            send_kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "reasoning": {"effort": reasoning_effort},
                # Force routing to a provider that actually honors the parameters we
                # send. Omitted, OpenRouter silently strips any param the chosen
                # provider lacks — so a model whose provider can't do
                # `response_format: json_schema` (GLM) falls back to a free-form
                # completion and emits fenced / prose / null content instead of
                # schema-conformant JSON.
                "provider": {
                    "require_parameters": True,
                    "sort": "latency",
                    "ignore": ["decart"],
                },
            }
            # A model whose ONLY OpenRouter endpoint lacks structured outputs
            # (poolside/laguna-s-2.1, …) 404s under require_parameters when we
            # demand response_format. Drop the schema param for those and lean on
            # the prompt's `<output>` contract to shape the JSON instead.
            if model not in NO_STRUCTURED_OUTPUT_LIST:
                send_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema_": wire_schema,
                    },
                }
            response = await client.chat.send_async(**send_kwargs)  # pyright: ignore[reportArgumentType]
    except (OpenRouterError, httpx.HTTPError) as e:
        # OpenRouter's own dashboard logs its side; this row is the LOCAL record
        # of the same flight, so the ledger reads uniformly across transports.
        flightlog.record(
            transport="openrouter", model=model,
            status=_http_status(e), error=str(e)[:500], exc_type=type(e).__name__,
            t_request=t_request, t_response=time.time(),
        )
        raise
    usage = getattr(response, "usage", None)
    flightlog.record(
        transport="openrouter", model=model, status=200,
        tokens_in=getattr(usage, "prompt_tokens", None),
        tokens_out=getattr(usage, "completion_tokens", None),
        generation_id=getattr(response, "id", None),
        t_request=t_request, t_response=time.time(),
    )
    message = response.choices[0].message
    return _Completion(
        content=message.content,
        reasoning=getattr(message, "reasoning", None) or "",
        usage=usage,
        generation_id=getattr(response, "id", None),
        finish_reason=response.choices[0].finish_reason,
        refusal=message.refusal if isinstance(message.refusal, str) else None,
    )


async def call_llm_once(
    *,
    system: str,
    user: str,
    output_schema: type[T],
    model: str,
    reasoning_effort: str | None = None,
    validate: Callable[[T], None] | None = None,
    step: str | None = None,
    node_id: str | None = None,
    zone_id: str | None = None,
    log_retries: bool = True,
) -> tuple[T, str, object, object, list[str]]:
    """One structured-output call with the full resample/backoff budget,
    WITHOUT the content-addressed cache lookup or the `cache.llm` log write
    that `call_llm` wraps around it. Returns
    `(validated, reasoning, usage, raw, generation_ids)`, where
    `generation_ids` lists every billed attempt (a resample bills each try).

    The prompt-tuning sandbox (`POST /llm/test`) calls this directly so a
    throwaway "what if I edited this prompt" test re-runs the exact step's
    call yet never reads or mutates any cell's event log — its result is
    rendered transiently and discarded. `call_llm` is the cached, logged
    pipeline path. `log_retries=False` (the sandbox runs outside any cell's
    SlotLog binding) skips the per-attempt diagnostic events so no `logging`
    ContextVar is required.

    `reasoning_effort` is the thinking level to request. `call_llm` resolves it
    from the call's template; a direct caller may pass its own, and omitting it
    resolves from `model` + `step` the same way (so a step the per-step flow names
    gets its level here too)."""

    def _retry_log(kind: str, **data: object) -> None:
        if log_retries:
            logging.log(kind, **data)

    # Bind the flight-ledger context: every HTTP attempt below records a row
    # stamped with this logical call's id + step/node, so a retry storm groups
    # back to the one pipeline call that caused it.
    flightlog.begin_call(step=step, node=node_id, zone_id=zone_id)

    effort = reasoning_effort or call_reasoning(model, step)

    # Independent retry budgets, one per failure class:
    #   * `parse_attempt` — JSON-decode / Pydantic-validation failures.
    #     The model misbehaved; resampling fixes it. 4 tries.
    #   * `transport_attempt` — provider returned a non-success body
    #     (Anthropic 503 "Overloaded", upstream timeouts, etc.) which the
    #     SDK surfaces as OpenRouterError / ResponseValidationError. The
    #     model never saw the request, so this isn't "the LLM was wrong" —
    #     it's a flap. Larger budget + longer backoff so a multi-second
    #     provider hiccup doesn't fail the run.
    #   * `rate_limit_attempt` — a 429, peeled off the transport class: it's the
    #     provider throttling us, not flapping. Gets its own EXPONENTIAL backoff,
    #     and for a rotating key pool the sleep is deferred until a full sweep
    #     of keys has 429'd (see `_rate_limit_*`).
    parse_attempt = 0
    transport_attempt = 0
    rate_limit_attempt = 0
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
        attempt_start = time.monotonic()
        try:
            comp = await _send_structured(
                model=model, system=system, user=user, output_schema=output_schema,
                reasoning_effort=effort,
            )
            # Recorded before parsing: an OpenRouter generation was billed
            # regardless of whether its output survives validation below.
            # Third-party OpenAI-compatible backends carry no billed id, so
            # nothing is appended and those calls never show as cost-pending.
            if comp.generation_id:
                generation_ids.append(comp.generation_id)
            content = comp.content
            args = json.loads(content) if isinstance(content, str) else content
            # A reasoning-heavy turn can finish with no answer body (`content`
            # null, or the bare literal `null`) — resample it with a clear reason
            # instead of letting `model_validate(None)` raise a cryptic
            # `model_type` error. finish_reason/refusal pin down truncation vs.
            # a content filter for the retry log.
            if args is None:
                raise ValueError(
                    "model returned no content "
                    f"(finish_reason={comp.finish_reason!r}, refusal={comp.refusal!r})"
                )
            validated = output_schema.model_validate(args)
            if validate is not None:
                # Semantic check (e.g. batch id echo). Raised BEFORE the
                # caller's cache.llm log, so a failing output is never cached
                # and each resample re-runs the call fresh.
                validate(validated)
            # Return the raw parsed response (`args`) alongside the validated
            # model so callers can log EXACTLY what the model emitted — the wire
            # field names and values — rather than a re-serialized, attribute-
            # named `model_dump`. `comp.usage` feeds the per-call trace panels;
            # `generation_ids` lets `call_llm` price the call against OpenRouter's
            # settled cost (empty for third-party backends, which have no
            # /generation record — completion_tokens already includes reasoning
            # tokens, so no separate reasoning field is needed).
            #
            # The one place the exact prompt + output enter the flight ledger:
            # fill the winning attempt's row in its scene DB. No-op when there's
            # no bound scene (sandbox) or logging is off.
            if log_retries:
                flightlog.attach_prompt(
                    system=system, user=user, output=args,
                    reasoning=comp.reasoning, schema=output_schema.__name__,
                )
            return validated, comp.reasoning, comp.usage, args, generation_ids
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
            # A 429 is the provider throttling us, not a flap. Rate-limit path:
            # for a rotating pool `_send_openai_compatible` has already rolled to
            # the next key, so the next attempt tries it with NO delay; we only
            # sleep once a whole sweep of the pool has 429'd (`attempt % pool`).
            # OpenRouter / single-key have pool_size 1, so every 429 backs off —
            # the same arithmetic. Backoff grows exponentially per exhausted
            # sweep.
            if _http_status(e) == 429:
                if rate_limit_attempt >= RATE_LIMIT_MAX - 1:
                    raise
                rate_limit_attempt += 1
                pool_size = _rate_limit_pool_size(model)
                swept = rate_limit_attempt % pool_size == 0
                backoff = (
                    _rate_limit_backoff(rate_limit_attempt // pool_size - 1) if swept else 0.0
                )
                _retry_log(
                    "llm.rate_limit_retry",
                    model=model,
                    step=step,
                    node=node_id,
                    attempt=rate_limit_attempt,
                    pool_size=pool_size,
                    action="backoff" if swept else "rotate",
                    backoff_s=backoff,
                )
                if backoff:
                    await asyncio.sleep(backoff)
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
            # Attribute the flap so the log strip is self-diagnosable: which
            # model/step/node, how long the attempt ran, and the read cap it hit
            # (a `ReadTimeout` whose elapsed ≈ timeout_s is a generation-too-long
            # timeout, not a transient drop). `httpx.ReadTimeout` carries no
            # message, so `exc_type` — not `reason` — is what actually names it.
            timeout_s = (
                _OPENAI_COMPAT_TIMEOUT.read
                if model in OPENAI_COMPAT_MODELS
                else _OPENROUTER_TIMEOUT_S
            )
            _retry_log(
                "llm.transport_retry",
                reason=f"{type(e).__name__}: {str(e)[:2000]}",
                exc_type=type(e).__name__,
                model=model,
                step=step,
                node=node_id,
                elapsed_s=round(time.monotonic() - attempt_start, 1),
                timeout_s=timeout_s,
                attempt=transport_attempt,
                backoff_s=backoff,
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


async def fetch_generation_costs(
    generation_ids: Iterable[str],
    on_cost: Callable[[str, float], Awaitable[None]] | None = None,
) -> dict[str, float]:
    """Best-effort `GET /generation` lookup of OpenRouter's settled `total_cost`
    for each (deduped) id, over one shared client with bounded concurrency.
    Returns only the ids that resolved; a 404 (stats not ready yet) or transient
    error simply omits that id, for the caller to retry on its next sweep. The
    SDK's own retry is disabled — its default budget is an hour.

    `on_cost`, when given, is awaited with `(gid, total_cost)` the MOMENT each
    lookup returns — after its fetch slot is released, so the callback (which may
    itself await) doesn't stall the other lookups. This lets a caller act on each
    settled cost as it lands (e.g. a spend-cap check) instead of waiting on the
    batch's slowest fetch. A callback fault is isolated so it can't abort the
    rest of the batch."""
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
        cost = res.data.total_cost
        out[gid] = cost
        if on_cost is not None:
            try:
                await on_cost(gid, cost)
            except Exception:
                pass  # a per-cost callback fault must not abort the batch

    async with OpenRouter(
        api_key=os.environ["OPENROUTER_API_KEY"], timeout_ms=30_000,
    ) as client:
        await asyncio.gather(*(_one(client, gid) for gid in ids))
    return out


async def backfill_costs(
    get_logs: Callable[[], Iterable[logging.SlotLog]],
    on_priced: Callable[[logging.SlotLog], Awaitable[None]] | None = None,
) -> int:
    """Price every logged-but-unpriced LLM call across the current cells: find
    each `cache.llm` carrying a `generation_id` with no matching `llm.cost`,
    fetch the settled cost, and append an `llm.cost` event. Idempotent and
    restart-proof — it reads only the durable log, so a call left unpriced by a
    slow stat or a live lookup killed mid-flight is recovered on a later pass.
    Returns the count priced this pass.

    Each cost is appended AS IT RETURNS (via `fetch_generation_costs`'s per-cost
    hook), and `on_priced(slot_log)` is awaited right after — so a caller's
    spend-cap check fires the instant the tipping cost lands, not gated behind
    the sweep's slowest unrelated lookup.

    A cell reset / (re)started / A/B-launched mid-sweep swaps in a FRESH SlotLog;
    appending through the stale object would stamp the event with its frozen
    `index` (its old length), colliding with the live writer and breaking the
    log's index↔position invariant that the prompt lab, rewind, and branch
    forking all rely on. So liveness is re-checked against `get_logs()`
    immediately before each append (which has no `await`, so it can't go stale
    between the check and the write)."""
    # Resolve the pending set up front: the fetch awaits, and the pipeline keeps
    # appending to these same logs meanwhile — anything new is caught next pass.
    pending: list[tuple[logging.SlotLog, str]] = []
    seen: set[str] = set()
    for sl in get_logs():
        # Never FORCE-LOAD a cell's log just to price it. A cell's events parse
        # lazily on first access (see `_LazyState`); an idle/never-opened cell's
        # log can be gigabytes, and eagerly loading every cell each sweep is what
        # OOMs the process with no scene even open. A cell that's actually active
        # (generating / opened) is already loaded, so its fresh costs still get
        # priced here; a genuinely idle cell's costs were priced while it ran and
        # get re-priced the moment it's next loaded.
        if not sl.loaded:
            continue
        events = sl.state["events"]
        resolved = {
            e.get("generation_id") for e in events if e.get("kind") == "llm.cost"
        }
        for e in events:
            if e.get("kind") != "cache.llm":
                continue
            # A statically-priced compat backend is token-priced at call time
            # (its own `llm.cost`, key-joined); it must NOT be OpenRouter-priced
            # here. This is the fix for a BYOK leg, which carries a real
            # generation_id yet bills $0 through OpenRouter — leaving it in the
            # sweep would append that $0 and mask the true token cost.
            if model_pricing(e.get("model")) is not None:
                continue
            gid = e.get("generation_id")
            if isinstance(gid, str) and gid not in resolved and gid not in seen:
                seen.add(gid)  # generation ids are unique; guard against dupes
                pending.append((sl, gid))
    if not pending:
        return 0
    by_gid = {gid: sl for sl, gid in pending}
    priced = 0

    async def _apply(gid: str, cost: float) -> None:
        nonlocal priced
        sl = by_gid.get(gid)
        # Skip a SlotLog swapped out during the fetch (reset/restart/A/B): writing
        # through it would break its index invariant. Re-checked here, right
        # before the await-free append, so a swap during a prior cost's
        # `on_priced` await can't slip a stale write through.
        if sl is None or sl not in get_logs():
            return
        sl.log("llm.cost", generation_id=gid, cost=cost)
        priced += 1
        if on_priced is not None:
            await on_priced(sl)

    await fetch_generation_costs((gid for _, gid in pending), on_cost=_apply)
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
    transport_attempt = 0
    TRANSPORT_MAX = 8
    TRANSPORT_BACKOFF = [2, 4, 8, 16, 30, 30, 30, 30]
    wire: list[dict[str, str]] = [{"role": "system", "content": system}, *messages]
    flightlog.begin_call(kind="chat")
    while True:
        t_request = time.time()
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
            usage = getattr(response, "usage", None)
            flightlog.record(
                transport="openrouter", model=model, status=200,
                tokens_in=getattr(usage, "prompt_tokens", None),
                tokens_out=getattr(usage, "completion_tokens", None),
                generation_id=getattr(response, "id", None),
                t_request=t_request, t_response=time.time(),
            )
            message = response.choices[0].message
            content = message.content
            text = content if isinstance(content, str) else str(content or "")
            reasoning = getattr(message, "reasoning", None) or ""
            return text, reasoning
        except (OpenRouterError, httpx.HTTPError) as e:
            # Provider flap (Anthropic 503, dropped connection) — the model
            # never saw the request, so back off and resend, same budget as
            # call_llm_once's transport retries.
            flightlog.record(
                transport="openrouter", model=model,
                status=_http_status(e), error=str(e)[:500], exc_type=type(e).__name__,
                t_request=t_request, t_response=time.time(),
            )
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
