"""Direct probe for the LongCat / SiliconFlow OpenAI-compatible chat endpoints.

Hits `{base_url}/chat/completions` with raw httpx — deliberately NOT through
`app.services.llm` or `app.core.slots`, so every request field is hand-tunable
and you see the untouched response (content vs reasoning_content, usage,
finish_reason). For poking at structured-output + thinking behaviour per
provider.

Keys load from server/.env by default; paste into the *_API_KEY_OVERRIDE
constants below to override. Two ways to drive it:

    # CLI
    python compat_probe.py                      # both providers, defaults
    python compat_probe.py longcat --no-thinking
    python compat_probe.py siliconflow --no-schema --raw

    # Browser UI (serves compat_client.html, proxies the call to dodge CORS)
    python compat_probe.py --serve              # open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Provider responses (and our own chrome) may carry non-ASCII; Windows consoles
# default to cp1252 and would crash on it.
sys.stdout.reconfigure(encoding="utf-8")

SERVER_DIR = Path(__file__).resolve().parent
CLIENT_HTML = SERVER_DIR / "compat_client.html"
load_dotenv(SERVER_DIR / ".env")

# Paste a key here to override server/.env (purely for local testing).
LONGCAT_API_KEY_OVERRIDE = ""
SILICONFLOW_API_KEY_OVERRIDE = ""

# ---- editable request defaults ------------------------------------------------
SYSTEM_PROMPT = "You are a concise assistant. Return only what is asked."
USER_PROMPT = "Give me a quick fact sheet about the capital city of Japan."
MAX_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 1.0
THINKING = True
STRUCTURED_OUTPUT = True
STRICT_SCHEMA = True
TIMEOUT_S = 300.0
DEFAULT_PORT = 8765

SCHEMA_NAME = "city_fact"
JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "country": {"type": "string"},
        "population_millions": {"type": "number"},
        "landmarks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["city", "country", "population_millions", "landmarks"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    model: str
    api_key_env: str
    api_key_override: str
    # Thinking is toggled with a different field per provider.
    thinking_on: dict[str, object]
    thinking_off: dict[str, object]

    def api_key(self) -> str | None:
        return self.api_key_override or os.environ.get(self.api_key_env)


PROVIDERS: dict[str, Provider] = {
    "longcat": Provider(
        name="longcat",
        base_url="https://api.longcat.chat/openai/v1",
        model="LongCat-2.0",
        api_key_env="LONGCAT_API_KEY",
        api_key_override=LONGCAT_API_KEY_OVERRIDE,
        thinking_on={"thinking": {"type": "enabled"}},
        thinking_off={"thinking": {"type": "disabled"}},
    ),
    "siliconflow": Provider(
        name="siliconflow",
        base_url="https://api.siliconflow.com/v1",
        model="meituan-longcat/LongCat-2.0",
        api_key_env="SILICONFLOW_API_KEY",
        api_key_override=SILICONFLOW_API_KEY_OVERRIDE,
        thinking_on={"enable_thinking": True},
        thinking_off={"enable_thinking": False},
    ),
}


def build_body(
    p: Provider,
    *,
    model: str | None,
    system: str,
    user: str,
    max_tokens: object,
    temperature: object,
    top_p: object,
    thinking: bool,
    response_format: object,
) -> dict[str, object]:
    body: dict[str, object] = {
        "model": model or p.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if response_format is not None:
        body["response_format"] = response_format
    body.update(p.thinking_on if thinking else p.thinking_off)
    return body


def send(
    p: Provider, body: dict[str, object], key: str, timeout: float, base_url: str | None = None
) -> tuple[httpx.Response, float]:
    url = f"{(base_url or p.base_url).rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    t0 = time.time()
    resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
    return resp, time.time() - t0


# ---- CLI ----------------------------------------------------------------------

def _mask(key: str | None) -> str:
    if not key:
        return "<missing>"
    return f"{key[:5]}...{key[-3:]}" if len(key) > 10 else "<set>"


def _indent(text: object, pad: str = "    ") -> str:
    return "\n".join(pad + line for line in str(text).splitlines()) or pad + "<empty>"


def _cli_response_format(schema_on: bool) -> object:
    if not schema_on:
        return None
    return {
        "type": "json_schema",
        "json_schema": {"name": SCHEMA_NAME, "strict": STRICT_SCHEMA, "schema": JSON_SCHEMA},
    }


def probe(p: Provider, a: argparse.Namespace) -> None:
    key = p.api_key()
    body = build_body(
        p,
        model=a.model,
        system=a.system,
        user=a.user,
        max_tokens=a.max_tokens,
        temperature=a.temperature,
        top_p=a.top_p,
        thinking=a.thinking,
        response_format=_cli_response_format(a.schema),
    )
    print("=" * 88)
    print(f"{p.name}  ->  {p.base_url}/chat/completions")
    print(f"  model={body['model']!r}  key={_mask(key)}  thinking={a.thinking}  structured={a.schema}")
    print("  request body:")
    print(_indent(json.dumps(body, indent=2, ensure_ascii=False)))
    if not key:
        print(f"  SKIP: no key ({p.api_key_env} unset and no override)")
        return

    try:
        resp, elapsed = send(p, body, key, a.timeout)
    except httpx.HTTPError as e:
        print(f"  TRANSPORT ERROR: {type(e).__name__}: {e}")
        return

    print(f"  HTTP {resp.status_code}   {elapsed:.1f}s")
    if resp.status_code // 100 != 2:
        print(_indent(resp.text))
        return

    data = resp.json()
    if a.raw:
        print("  raw response:")
        print(_indent(json.dumps(data, indent=2, ensure_ascii=False)))

    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = data.get("usage") or {}
    content = msg.get("content")
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""

    print(
        f"  finish_reason={choice.get('finish_reason')!r}  "
        f"tokens_in={usage.get('prompt_tokens')}  tokens_out={usage.get('completion_tokens')}"
    )
    print(f"  -- content ({len(content or '')} chars) --")
    print(_indent(content or "<empty>"))
    print(f"  -- reasoning_content ({len(reasoning)} chars) --")
    print(_indent(reasoning or "<empty>"))

    if a.schema:
        any_ok = False
        for chan, val in (("content", content), ("reasoning_content", reasoning)):
            if not (isinstance(val, str) and val.strip()):
                continue
            try:
                parsed = json.loads(val)
            except json.JSONDecodeError as e:
                print(f"  [{chan}] not valid JSON: {e}")
                continue
            any_ok = True
            print(f"  [{chan}] valid schema JSON:")
            print(_indent(json.dumps(parsed, indent=2, ensure_ascii=False)))
        if not any_ok:
            print("  NO channel returned valid schema JSON")


# ---- browser UI (local proxy so the page dodges provider CORS) ----------------

def _config() -> dict[str, object]:
    return {
        "providers": {
            name: {
                "base_url": p.base_url,
                "model": p.model,
                "api_key_env": p.api_key_env,
                "key_present": bool(p.api_key()),
                "thinking_on": p.thinking_on,
                "thinking_off": p.thinking_off,
            }
            for name, p in PROVIDERS.items()
        },
        "defaults": {
            "provider": next(iter(PROVIDERS)),
            "system": SYSTEM_PROMPT,
            "user": USER_PROMPT,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "thinking": THINKING,
            "structured": STRUCTURED_OUTPUT,
            "schema_name": SCHEMA_NAME,
            "strict": STRICT_SCHEMA,
            "schema": JSON_SCHEMA,
            "timeout": TIMEOUT_S,
        },
    }


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # silence default per-request noise
        pass

    def _write(self, status: int, ctype: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: object, status: int = 200) -> None:
        self._write(status, "application/json; charset=utf-8", json.dumps(obj).encode("utf-8"))

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            if not CLIENT_HTML.is_file():
                self._write(500, "text/plain; charset=utf-8", b"compat_client.html not found")
                return
            self._write(200, "text/html; charset=utf-8", CLIENT_HTML.read_bytes())
        elif self.path == "/api/config":
            self._json(_config())
        else:
            self._write(404, "text/plain; charset=utf-8", b"not found")

    def do_POST(self) -> None:
        if self.path != "/api/probe":
            self._write(404, "text/plain; charset=utf-8", b"not found")
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            self._json({"ok": False, "error": f"bad request JSON: {e}"}, 400)
            return

        p = PROVIDERS.get(payload.get("provider"))
        if p is None:
            self._json({"ok": False, "error": f"unknown provider {payload.get('provider')!r}"}, 400)
            return
        key = (payload.get("api_key") or "").strip() or p.api_key()
        if not key:
            self._json(
                {"ok": False, "error": f"no API key (set {p.api_key_env} in server/.env or enter one)"},
                400,
            )
            return

        body = build_body(
            p,
            model=(payload.get("model") or None),
            system=payload.get("system") or "",
            user=payload.get("user") or "",
            max_tokens=payload.get("max_tokens"),
            temperature=payload.get("temperature"),
            top_p=payload.get("top_p"),
            thinking=bool(payload.get("thinking")),
            response_format=payload.get("response_format"),
        )
        timeout = float(payload.get("timeout") or TIMEOUT_S)
        try:
            resp, elapsed = send(p, body, key, timeout, base_url=payload.get("base_url") or None)
        except httpx.HTTPError as e:
            self._json({"ok": False, "error": f"{type(e).__name__}: {e}", "request_body": body})
            return

        try:
            data = resp.json()
        except ValueError:
            data = None
        print(f"  probe {p.name} -> {resp.status_code} {elapsed:.1f}s")
        self._json(
            {
                "ok": True,
                "status": resp.status_code,
                "elapsed": round(elapsed, 2),
                "request_body": body,
                "response_json": data,
                "response_text": resp.text,
            }
        )


def serve(port: int) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    url = f"http://127.0.0.1:{port}/"
    for name, p in PROVIDERS.items():
        print(f"  {name}: key {'loaded' if p.api_key() else 'MISSING'} ({p.api_key_env})")
    print(f"compat client on {url}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("provider", nargs="?", default="both", choices=[*PROVIDERS, "both"])
    ap.add_argument("--serve", action="store_true", help="launch the browser UI instead of a CLI probe")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--model", default=None, help="override the provider's default model id")
    ap.add_argument("--system", default=SYSTEM_PROMPT)
    ap.add_argument("--user", default=USER_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=TOP_P)
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=THINKING)
    ap.add_argument("--schema", action=argparse.BooleanOptionalAction, default=STRUCTURED_OUTPUT)
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    ap.add_argument("--raw", action="store_true", help="dump the full response JSON")
    a = ap.parse_args()

    if a.serve:
        serve(a.port)
        return

    names = list(PROVIDERS) if a.provider == "both" else [a.provider]
    for name in names:
        probe(PROVIDERS[name], a)


if __name__ == "__main__":
    main()
