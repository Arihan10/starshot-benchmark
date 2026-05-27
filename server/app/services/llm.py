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
from contextvars import ContextVar
from pathlib import Path

from openrouter import OpenRouter
from openrouter.errors import OpenRouterError
from pydantic import BaseModel, ValidationError

from app.utils import cache, logging

_call_seq: dict[str, int] = {}

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


async def call_llm[T: BaseModel](
    *,
    system: str,
    user: str,
    output_schema: type[T],
    node_id: str | None = None,
    step: str | None = None,
) -> T:
    model = _current_model.get()
    if model is None:
        raise RuntimeError("llm.set_model() must be called before call_llm()")
    key = cache.hash_llm_call(
        model=model,
        system=system,
        user=user,
        schema_name=output_schema.__name__,
    )
    hit = cache.find_llm_cache_hit(logging.current_events(), key)
    if hit is not None:
        return output_schema.model_validate(hit)

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
    PARSE_MAX = 4
    TRANSPORT_MAX = 8
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
                    reasoning={"effort": "xhigh"},
                )
            message = response.choices[0].message
            content = message.content
            args = json.loads(content) if isinstance(content, str) else content
            validated = output_schema.model_validate(args)
            reasoning = getattr(message, "reasoning", None) or ""
            # cache.llm carries everything needed for both the LLM-call cache
            # (key + output) and the observability view (node + step + model
            # + system + user + reasoning). Older log lines that lacked the
            # new fields still replay correctly — the client treats them as
            # unattributed.
            logging.log(
                "cache.llm",
                key=key,
                node=node_id,
                step=step,
                model=model,
                system=system,
                user=user,
                output=validated.model_dump(mode="json"),
                reasoning=reasoning,
            )
            return validated
        except json.JSONDecodeError as e:
            final = parse_attempt >= PARSE_MAX - 1
            logging.log(
                "llm.json_decode_error",
                reason=f"JSONDecodeError: {e}",
                attempt=parse_attempt,
                final=final,
                content=content if isinstance(content, str) else repr(content),
            )
            if final:
                raise
            parse_attempt += 1
        except OpenRouterError as e:
            if transport_attempt >= TRANSPORT_MAX - 1:
                raise
            backoff = TRANSPORT_BACKOFF[
                min(transport_attempt, len(TRANSPORT_BACKOFF) - 1)
            ]
            logging.log(
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
            logging.log("llm.retry", reason=f"{type(e).__name__}: {str(e)[:160]}")
            parse_attempt += 1


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
