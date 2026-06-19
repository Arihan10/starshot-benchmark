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
import sys
from collections.abc import Awaitable, Callable, Iterable
from contextvars import ContextVar
from pathlib import Path
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

    validated, reasoning, usage, raw = await call_llm_once(
        system=system,
        user=user,
        output_schema=output_schema,
        model=model,
        validate=validate,
        step=step,
    )
    # cache.llm carries everything needed for the LLM-call cache (key +
    # output), the observability view (node + step + model + system + user +
    # reasoning), and the prompt lab (template + variables). Older log lines
    # that lack the newer fields still replay correctly — the client treats
    # them as unattributed / not re-renderable.
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
    )
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
) -> tuple[T, str, object, object]:
    """One structured-output call with the full resample/backoff budget,
    WITHOUT the content-addressed cache lookup or the `cache.llm` log write
    that `call_llm` wraps around it. Returns `(validated, reasoning, usage)`.

    The prompt-tuning sandbox (`POST /llm/test`) calls this directly so a
    throwaway "what if I edited this prompt" test re-runs the exact step's
    call yet never reads or mutates any cell's event log — its result is
    rendered transiently and discarded. `call_llm` is the cached, logged
    pipeline path. `log_retries=False` (the sandbox runs outside any cell's
    SlotLog binding) skips the per-attempt diagnostic events so no `logging`
    ContextVar is required."""

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
    while True:
        content: object = None
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
                    provider={
                        "require_parameters": True,
                        "sort": "latency"
                    },
                )
            message = response.choices[0].message
            content = message.content
            args = json.loads(content) if isinstance(content, str) else content
            validated = output_schema.model_validate(args)
            if validate is not None:
                # Semantic check (e.g. batch id echo). Raised BEFORE the
                # caller's cache.llm log, so a failing output is never cached
                # and each resample re-runs the call fresh.
                validate(validated)
            reasoning = getattr(message, "reasoning", None) or ""
            # Token counts for the client's per-run spend tracker. `usage` is
            # absent on the rare provider that omits it; cost falls back to a
            # char-length estimate client-side. completion_tokens already
            # includes reasoning tokens (OpenRouter bills them at the
            # completion rate), so no separate reasoning field is needed.
            usage = getattr(response, "usage", None)
            # Return the raw parsed response (`args`) alongside the validated
            # model so callers can log EXACTLY what the model emitted — the wire
            # field names and values — rather than a re-serialized, attribute-
            # named `model_dump`.
            return validated, reasoning, usage, args
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
            )
            await asyncio.sleep(backoff)
            transport_attempt += 1
        except (ValidationError, ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            if parse_attempt >= PARSE_MAX - 1:
                raise
            _retry_log("llm.retry", reason=f"{type(e).__name__}: {str(e)[:160]}")
            parse_attempt += 1


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
