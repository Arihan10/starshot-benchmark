"""Probe OpenRouter structured-output support, model by model.

Standalone: stdlib only, no repo imports, raw HTTP to openrouter.ai.

    python scripts/test_structured_output.py [model_id ...]

Four signals per model:
  1. what /models advertises in supported_parameters
  2. a live json_schema call, routed normally
  3. the same call with provider.require_parameters=true, which makes
     OpenRouter refuse providers that cannot enforce the schema -- a 404
     here means nothing actually enforces it, whatever (2) returned
  4. the same call with sort=latency, the routing the pipeline uses.
     Enforcement is a property of the ENDPOINT, not the model, and several
     models have sibling endpoints under one provider name where the
     fastest one accepts response_format without honoring it.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://openrouter.ai/api/v1"

DEFAULT_MODELS = [
    "anthropic/claude-opus-5",
    "anthropic/claude-opus-4.8",
    "openai/gpt-5.6-sol-pro",
]

# Field names are deliberately ones the prompt below never says and no model
# would volunteer, so they are reachable ONLY through the schema. With obvious
# names ("name", "materials") a model that never saw the schema still guesses
# them right, and an endpoint that ignores response_format scores as a pass.
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["designation", "elevation_m", "substances", "components"],
    "properties": {
        "designation": {"type": "string"},
        "elevation_m": {"type": "number"},
        "substances": {
            "type": "array",
            "items": {"type": "string", "enum": ["wood", "stone", "metal", "glass"]},
        },
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["descriptor", "tally"],
                "properties": {
                    "descriptor": {"type": "string"},
                    "tally": {"type": "integer"},
                },
            },
        },
    },
}

PROMPT = (
    "Describe a small wooden watchtower with a stone base: what it is called, "
    "how tall it is in meters, what it is made of, and its parts with a count "
    "for each. Respond with a single JSON object matching the schema."
)


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env = Path(__file__).resolve().parent.parent / "server" / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENROUTER_API_KEY":
                return value.strip().strip("'\"")
    sys.exit("OPENROUTER_API_KEY not set (env or server/.env)")


def request(path: str, key: str, payload: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(payload).encode() if payload else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, {"error": {"message": body[:300]}}


def violations(value, schema, path="$") -> list[str]:
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, dict):
            return [f"{path}: expected object, got {type(value).__name__}"]
        out = [f"{path}.{k}: missing" for k in schema["required"] if k not in value]
        for k, v in value.items():
            if k not in schema["properties"]:
                out.append(f"{path}.{k}: key not in schema")
            else:
                out += violations(v, schema["properties"][k], f"{path}.{k}")
        return out
    if kind == "array":
        if not isinstance(value, list):
            return [f"{path}: expected array, got {type(value).__name__}"]
        return [
            err
            for i, item in enumerate(value)
            for err in violations(item, schema["items"], f"{path}[{i}]")
        ]
    expected = {"string": str, "number": (int, float), "integer": int, "boolean": bool}[kind]
    if not isinstance(value, expected) or (kind != "boolean" and isinstance(value, bool)):
        return [f"{path}: expected {kind}, got {json.dumps(value)[:40]}"]
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path}: {value!r} not in enum {schema['enum']}"]
    return []


def probe(model: str, key: str, name: str, provider: dict | None) -> bool:
    label = name.ljust(20)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 4000,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "watchtower", "strict": True, "schema": SCHEMA},
        },
    }
    if provider:
        payload["provider"] = provider

    t0 = time.time()
    status, body = request("/chat/completions", key, payload)
    secs = time.time() - t0

    if status != 200:
        message = (body.get("error") or {}).get("message", json.dumps(body)[:200])
        print(f"  [{label}] {status} {secs:5.1f}s  {message}")
        return False

    choice = body["choices"][0]
    content = choice["message"].get("content")
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    provider = body.get("provider", "?")
    finish = choice.get("finish_reason") or choice.get("native_finish_reason")

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        preview = (content or "").strip().replace("\n", " ")[:120]
        print(f"  [{label}] 200 {secs:5.1f}s  provider={provider} finish={finish}  "
              f"NOT RAW JSON: {preview!r}")
        return False

    errs = violations(parsed, SCHEMA)
    verdict = "schema PASS" if not errs else f"schema FAIL ({len(errs)})"
    print(f"  [{label}] 200 {secs:5.1f}s  provider={provider} finish={finish}  {verdict}")
    for err in errs[:5]:
        print(f"      - {err}")
    return not errs


def main() -> None:
    key = api_key()
    models = sys.argv[1:] or DEFAULT_MODELS

    status, catalog = request("/models", key)
    advertised = {
        m["id"]: m.get("supported_parameters", []) for m in catalog.get("data", [])
    } if status == 200 else {}

    for model in models:
        print(f"\n=== {model} ===")
        if model in advertised:
            flags = [p for p in ("response_format", "structured_outputs") if p in advertised[model]]
            print(f"  advertised: {', '.join(flags) if flags else 'neither response_format nor structured_outputs'}")
        else:
            print("  advertised: model not found in /models catalog")

        plain_ok = probe(model, key, "plain", None)
        strict_ok = probe(model, key, "require_parameters", {"require_parameters": True})
        latency_ok = probe(
            model, key, "+ sort=latency", {"require_parameters": True, "sort": "latency"}
        )

        if strict_ok and latency_ok:
            print("  VERDICT: supported on both routes")
        elif strict_ok:
            print("  VERDICT: supported, BUT the latency-sorted endpoint ignores the schema")
        elif plain_ok:
            print("  VERDICT: partial -- valid JSON, but no route enforces the schema")
        else:
            print("  VERDICT: not supported")


if __name__ == "__main__":
    main()
