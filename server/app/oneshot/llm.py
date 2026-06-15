"""Minimal OpenAI-compatible completion client for the dLLM providers.

The one-shot track benchmarks diffusion LLMs served outside OpenRouter (the
Starshot dLLM gateway on Modal, Inception's Mercury API), so it owns this
client instead of the main pipeline's OpenRouter SDK wrapper. Plain httpx
against `{base_url}/chat/completions`, two flavors:

  * `call_structured` — `response_format: json_schema`; both providers
    validate/repair structured output server-side, we still parse and
    pydantic-validate.
  * `call_text` — plain completion for steps whose output contract is text
    (CSV object lists, bracket-array grid plans); the caller supplies the
    parser and the canonical pydantic model is logged as `output` (with the
    verbatim text as `raw`) so replay and the dashboards stay format-blind.

Every design call is a SINGLE attempt — there is no validation auto-retry
or resample-with-feedback. The model's output is judged as-is: a violation
fails the call (and the run) with the validator's message. Only transport
faults (connection flaps, 5xx, cold starts) retry, because they never gave
the model a second chance at the content.

Event-log integration mirrors the main client: a content-addressed cache
lookup before the call, a `cache.llm` event on success (so the observability
panels, structural replay, and resume all work unchanged), and
`llm.transport_retry` diagnostics. A failed call commits nothing — it logs
one `llm.failed` event with the exact prompts sent, the verbatim output and
reasoning received, and the validator's reason; the dashboard's info panel
renders it exactly like a committed call.

Diffusion backends form the whole response at once and cold starts take
minutes, so the read timeout is generous — there is no token stream to keep
the connection chatty.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.oneshot.slots import DllmModel
from app.utils import cache, logging

T = TypeVar("T", bound=BaseModel)

_TIMEOUT = httpx.Timeout(connect=30.0, read=900.0, write=60.0, pool=60.0)

TRANSPORT_MAX = 4
_TRANSPORT_BACKOFF = [5, 15, 30]


class DllmRequestError(Exception):
    """Provider returned a non-retryable (4xx) error."""


def _normalize_schema(schema: object) -> object:
    """Normalize the pydantic-emitted JSON schema for strict structured-output
    modes: collapse `prefixItems` (fixed-length tuples like Vec3) into a single
    `items`, drop numeric/length constraint keywords some validators reject,
    and require every property with no additionals. Pydantic still enforces
    the original constraints on the parsed response."""
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
            out["required"] = sorted(out["properties"].keys())
        return out
    if isinstance(schema, list):
        return [_normalize_schema(v) for v in schema]
    return schema


def _extract_reasoning(response: dict, message: dict) -> str:
    """The reasoning trace, wherever the provider puts it: OpenRouter-style
    `message.reasoning` (or `reasoning_content`), else Inception's top-level
    `reasoning_summary` object. Empty string when the model emitted none
    (the gateway's dLLMs have no reasoning channel)."""
    for key in ("reasoning", "reasoning_content"):
        v = message.get(key)
        if isinstance(v, str) and v:
            return v
    rs = response.get("reasoning_summary")
    if isinstance(rs, str):
        return rs
    if isinstance(rs, dict):
        for key in ("text", "summary", "content"):
            v = rs.get(key)
            if isinstance(v, str) and v:
                return v
        return json.dumps(rs)
    return ""


def _headers(cfg: DllmModel) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if cfg.api_key_env is not None:
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise DllmRequestError(
                f"{cfg.api_key_env} is not set — required for {cfg.model} at {cfg.base_url}"
            )
        headers["Authorization"] = f"Bearer {key}"
    return headers


async def _post_completion(cfg: DllmModel, body: dict) -> dict:
    """POST /chat/completions with transport retries (connection flaps, 5xx,
    cold-start hiccups). 4xx raises immediately — that's a config problem.
    Error bodies are carried in full — these logs are the debugging surface
    for experimental backends, so nothing is truncated."""
    attempt = 0
    while True:
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                res = await client.post(
                    f"{cfg.base_url}/chat/completions",
                    headers=_headers(cfg),
                    json=body,
                )
            if res.status_code < 400:
                return res.json()
            if res.status_code < 500:
                raise DllmRequestError(
                    f"{cfg.model} HTTP {res.status_code}: {res.text}"
                )
            reason = f"HTTP {res.status_code}: {res.text}"
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            reason = f"{type(e).__name__}: {e}"
        if attempt >= TRANSPORT_MAX - 1:
            raise DllmRequestError(f"{cfg.model} transport failed: {reason}")
        backoff = _TRANSPORT_BACKOFF[min(attempt, len(_TRANSPORT_BACKOFF) - 1)]
        logging.log(
            "llm.transport_retry",
            step="oneshot_scene",
            reason=reason,
            attempt=attempt,
            backoff_s=backoff,
        )
        await asyncio.sleep(backoff)
        attempt += 1


def _log_failed(
    cfg: DllmModel,
    *,
    node_id: str | None,
    step: str | None,
    schema_name: str,
    system: str,
    user: str,
    reason: str,
    raw: str | None = None,
    reasoning: str = "",
    finish_reason: object = None,
) -> None:
    """The no-commit counterpart of cache.llm, sharing its field names so the
    dashboard's info panel renders failed and committed calls uniformly: the
    exact prompts sent, the verbatim output/reasoning when a response existed,
    and why the call failed."""
    logging.log(
        "llm.failed",
        node=node_id,
        step=step,
        model=cfg.model,
        schema=schema_name,
        system=system,
        user=user,
        reason=reason,
        raw=raw,
        reasoning=reasoning,
        finish_reason=finish_reason,
    )


async def call_structured(
    cfg: DllmModel,
    *,
    system: str,
    user: str,
    output_schema: type[T],
    node_id: str | None = None,
    step: str | None = None,
) -> T:
    schema_name = output_schema.__name__
    # base_url disambiguates the same model id served by two providers.
    key = cache.hash_llm_call(
        model=f"{cfg.base_url}::{cfg.model}",
        system=system,
        user=user,
        schema_name=schema_name,
    )
    hit = cache.find_llm_cache_hit(logging.current_events(), key)
    if hit is not None:
        return output_schema.model_validate(hit)

    body: dict = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": _normalize_schema(output_schema.model_json_schema()),
            },
        },
        "stream": False,
    }
    if cfg.max_tokens is not None:
        body["max_tokens"] = cfg.max_tokens
    if cfg.extra:
        body.update(cfg.extra)

    try:
        response = await _post_completion(cfg, body)
    except DllmRequestError as e:
        _log_failed(
            cfg, node_id=node_id, step=step, schema_name=schema_name,
            system=system, user=user, reason=str(e),
        )
        raise
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    raw = content if isinstance(content, str) else repr(content)
    reasoning = _extract_reasoning(response, message)
    try:
        args = json.loads(content) if isinstance(content, str) else content
        validated = output_schema.model_validate(args)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        # finish_reason matters most when debugging: "length" means the token
        # budget was consumed (reasoning tokens count against max_tokens on
        # Inception), not that the model emitted bad JSON.
        reason = f"{type(e).__name__}: {e}"
        _log_failed(
            cfg, node_id=node_id, step=step, schema_name=schema_name,
            system=system, user=user, reason=reason, raw=raw,
            reasoning=reasoning, finish_reason=choice.get("finish_reason"),
        )
        raise DllmRequestError(
            f"{cfg.model} failed schema validation: {reason}"
        ) from None
    usage = response.get("usage") or {}
    logging.log(
        "cache.llm",
        key=key,
        node=node_id,
        step=step,
        model=cfg.model,
        schema=schema_name,
        system=system,
        user=user,
        output=args,
        raw=raw,
        reasoning=reasoning,
        tokens_in=usage.get("prompt_tokens"),
        tokens_out=usage.get("completion_tokens"),
    )
    return validated


async def call_text(
    cfg: DllmModel,
    *,
    system: str,
    user: str,
    schema: type[T],
    parse: Callable[[str], T],
    node_id: str | None = None,
    step: str | None = None,
) -> T:
    """Plain (non-structured) completion for text output contracts. `parse`
    turns the raw content into the canonical pydantic model, raising
    ValueError with a precise message on violations — which fails the call
    as-is. The committed `cache.llm` event carries the parsed model as
    `output` (same shape the structured path logs, so resume replay and the
    dashboards are format-blind) plus the verbatim model text as `raw`."""
    schema_name = schema.__name__
    key = cache.hash_llm_call(
        model=f"{cfg.base_url}::{cfg.model}",
        system=system,
        user=user,
        schema_name=schema_name,
    )
    hit = cache.find_llm_cache_hit(logging.current_events(), key)
    if hit is not None:
        return schema.model_validate(hit)

    body: dict = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    if cfg.max_tokens is not None:
        body["max_tokens"] = cfg.max_tokens
    if cfg.extra:
        body.update(cfg.extra)

    try:
        response = await _post_completion(cfg, body)
    except DllmRequestError as e:
        _log_failed(
            cfg, node_id=node_id, step=step, schema_name=schema_name,
            system=system, user=user, reason=str(e),
        )
        raise
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    raw = content if isinstance(content, str) else repr(content)
    reasoning = _extract_reasoning(response, message)
    try:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("response content is empty")
        validated = parse(content)
    except ValueError as e:
        _log_failed(
            cfg, node_id=node_id, step=step, schema_name=schema_name,
            system=system, user=user, reason=str(e), raw=raw,
            reasoning=reasoning, finish_reason=choice.get("finish_reason"),
        )
        raise DllmRequestError(
            f"{cfg.model} failed output validation: {e}"
        ) from None
    usage = response.get("usage") or {}
    logging.log(
        "cache.llm",
        key=key,
        node=node_id,
        step=step,
        model=cfg.model,
        schema=schema_name,
        system=system,
        user=user,
        output=validated.model_dump(mode="json"),
        raw=content,
        reasoning=reasoning,
        tokens_in=usage.get("prompt_tokens"),
        tokens_out=usage.get("completion_tokens"),
    )
    return validated
