# hy3 / self-hosted vLLM: "empty response → JSON parse failure" under load

## TL;DR

A self-hosted (Modal + vLLM) OpenAI-compatible endpoint intermittently (~10% under
concurrency) makes structured-output calls fail to parse. It is **not** the model,
the reasoning, or the JSON schema. Under concurrent load **Modal's edge returns a
non-2xx HTTP response with an empty / non-JSON body** (most often `HTTP 303` with
`content-length: 0`), and the client parses that empty body as if it were the
model's completion → `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`.

Fix has two parts:
1. **Client (+ any reverse proxy):** treat Modal-edge transients (`3xx`, `5xx`, and
   any body starting with `modal-http:`) as **retryable**, instead of parsing them.
2. **Endpoint:** the GPU container is OOM-crashing under concurrent load
   (`HTTP 500 … function was terminated by signal`). Reduce concurrency / memory
   pressure and/or add replicas.

---

## Symptom / signature

- Elevated structured-output failure rate (~10%) only under concurrency; near-zero
  at low concurrency.
- Failure event logged by the client looks like:
  ```
  kind:    llm.json_decode_error
  reason:  JSONDecodeError: Expecting value: line 1 column 1 (char 0)
  content: "None"
  ```
- `"content": "None"` is a red herring: the client never received model content. It
  is `repr(None)` — the content variable was still `None` because the HTTP body
  failed to parse *before* the message was read.

## Why it is NOT the model / reasoning / schema

Controlled probes against the same endpoint at **low concurrency (2–3)**:

- 26 calls across simple and complex real schemas, **including a production-sized
  input (~5,000 prompt tokens) and reasoning traces up to ~23,000 chars / ~7,900
  completion tokens** → **0 failures**. Every response `finish_reason=stop`, valid
  JSON, reasoning cleanly separated (no `<think>` leak).
- The same model hosted by a normal online provider (via the OpenRouter path) at
  `xhigh` reasoning through the *same* pipeline → no failures.

So reason-then-JSON works fine; guided decoding + reasoning coexist. The failures
only appear under **concurrent load against the self-hosted endpoint**.

## Root cause (measured)

Under a concurrent burst (conc 10–12) the endpoint returned, with **empty or
non-JSON bodies**:

| HTTP status | body | meaning | how the client currently handles it |
|---|---|---|---|
| `303` | *(empty, content-length 0)* | Modal edge redirect while scaling/load-balancing | **falls through to `res.json()` → crash** (this is the recurring ~10%) |
| `500` | `modal-http: internal error: function was terminated by signal` | **GPU container crashed (OOM)** under concurrent load | retried as 5xx (transport) |
| `404` | `modal-http: invalid function call` | no live container during crash/restart | fail-fast as 4xx (surfaces as a hard error) |

Measured rates (conc 12): router **7/24** empty-body `303`; direct vLLM serve
**6/24** empty-body `303` — i.e. the empty `303` comes from **Modal's edge, on both
the router and the raw serve URL**, not from the proxy logic or from vLLM output.

Heavier load additionally tipped the GPU container over: `HTTP 500 … terminated by
signal`, then a run of `404 invalid function call` while it restarted. That is a
real **capacity/OOM** problem (very long reason+JSON sequences × high concurrency ×
high GPU memory utilization on a single replica).

### The exact client bug

The client's OpenAI-compatible caller only guards `>= 500` and `>= 400`:

```python
res = await client.post(url, headers=headers, json=body)   # httpx: follow_redirects=False by default
if res.status_code >= 500:
    raise TransportError(...)          # retried
if res.status_code >= 400:
    raise NonRetryableError(...)       # fail fast
response = res.json()                  # a 303 (>=300, <400) reaches HERE on an empty body
                                       # -> JSONDecodeError "Expecting value ... char 0"
```

A `303` is `>= 300` but `< 400`, so it passes both guards and reaches `res.json()`,
which fails on the empty body. Because `content` is only assigned *after* a
successful `res.json()`, it is still `None`, and the diagnostic logs `repr(None)` =
`"None"`. httpx does **not** follow redirects by default, so the `303` is never
chased. (OpenAI-SDK-based clients partly avoid this because the SDK follows
redirects.)

---

## Fix

### 1. Client (and any reverse proxy in front of the endpoint)

Detect Modal-edge transients and **retry the same request with backoff** instead of
parsing them. This uniformly covers the empty `303`, the `500 … terminated by
signal`, and the transient `404 invalid function call`.

```python
def _is_modal_transient(res) -> bool:
    """Modal edge returned a non-answer we should retry, not parse:
      * any 3xx (scaling/redirect; body is empty)         -> retry
      * any 5xx                                            -> retry
      * a Modal-edge error body ("modal-http: ...")        -> retry
        (covers the transient 404 "invalid function call" during a restart)
      * an empty/blank body on a nominal 200               -> retry
    A genuine 4xx with a real JSON error body is NOT transient -> fail fast."""
    text = res.text or ""
    if 300 <= res.status_code < 400:
        return True
    if res.status_code >= 500:
        return True
    if text.lstrip().startswith("modal-http:"):
        return True
    if res.status_code == 200 and not text.strip():
        return True
    return False
```

Wire it into the send loop *before* any `res.json()`:

```python
res = await client.post(url, headers=headers, json=body)
if _is_modal_transient(res):
    # back off and retry the SAME POST (same budget as other transport retries)
    log("llm.transport_retry", status=res.status_code, body=res.text[:200])
    await asyncio.sleep(backoff); continue
if res.status_code >= 400:
    raise NonRetryableError(f"HTTP {res.status_code}: {res.text[:500]}")   # real client error
response = res.json()
```

Notes:
- Do this in **every** OpenAI-compatible call path (structured + free-form chat).
- Keep the existing transport-retry budget/backoff (e.g. up to 8 tries, exp backoff).
- Optional complementary hardening: create the httpx client with
  `follow_redirects=True`. (The retry-on-3xx approach above is preferred because it
  does not depend on the redirect's `Location`/method semantics; a `303` turns a
  POST into a GET, which is not what we want for a chat/completions call.)

### 2. Reverse proxy (the always-on router in front of the GPU)

If you run a proxy that re-streams responses, it must **not forward** the empty
`303` / `5xx` / `modal-http:` bodies to callers. It should retry them the same way
it already retries `503` (Modal's "no warm replica yet"). Extend the proxy's retry
condition from "only 503" to "any 3xx, any 5xx, or a `modal-http:` body", with the
same cold-start wait/backoff loop.

### 3. Endpoint (Modal + vLLM) — the deeper capacity fix

The `HTTP 500 … terminated by signal` means the container is being killed (OOM)
under concurrent load. The retries above make the client resilient, but the
endpoint should also stop crashing:

- **Lower per-replica concurrency** (`max_inputs` / `MAX_CONCURRENCY`): these are
  very long sequences (input + multi-thousand-token reasoning + structured answer);
  32 concurrent is aggressive. Try 8–16.
- **Reduce memory pressure**: lower `--gpu-memory-utilization` (e.g. 0.90 → 0.85)
  and/or shrink `--max-model-len` toward what you actually use.
- **Add capacity**: allow more replicas (raise autoscale max) or keep one warm
  (`min_containers >= 1`) so load spreads and cold-start `303`s are rarer.
- **Bound the sequence length**: reasoning is unbounded at `reasoning_effort=high`
  (4–8k+ tokens). A real reasoning cap (a working `thinking_token_budget`, or drop
  to `low`) shrinks each sequence and reduces KV-cache pressure.

> Side finding: on this stack a per-request `thinking_token_budget` (in
> `chat_template_kwargs`, top-level, or as `max_thinking_tokens`) was **ignored** —
> reasoning stayed ~12–15k chars vs a requested 512-token cap. If you rely on that
> knob to bound reasoning, verify it actually binds; otherwise use `reasoning_effort`
> (`no_think` / `low` / `high` are the only valid values — `medium` returns HTTP 400).

---

## Reproduction

Fire a concurrent burst of structured calls at the endpoint and classify the HTTP
body (not just the parsed `content`):

```python
import asyncio, json, httpx

URL = "https://<workspace>--<app>-router.modal.run/v1/chat/completions"
BODY = {  # any real structured request; reasoning on
    "model": "hy3",
    "messages": [{"role": "system", "content": "Return ONLY JSON per the schema."},
                 {"role": "user", "content": "Decompose a hotel room into 10-14 objects ..."}],
    "response_format": {"type": "json_schema", "json_schema": {"name": "X", "strict": True,
        "schema": {"type": "object", "additionalProperties": False, "required": ["objects"],
                   "properties": {"objects": {"type": "array", "items": {"type": "object",
                       "additionalProperties": False, "required": ["id"],
                       "properties": {"id": {"type": "string"}}}}}}}},
    "chat_template_kwargs": {"reasoning_effort": "high", "enable_thinking": True},
    "stream": False,
}

async def one(client):
    r = await client.post(URL, headers={"Content-Type": "application/json"}, json=BODY)
    body = r.text
    if not body.strip():
        return f"EMPTY_BODY status={r.status_code} loc={r.headers.get('location')!r}"
    try:
        json.loads(body); return "OK" if (json.loads(body).get("choices")) else "OK?"
    except json.JSONDecodeError:
        return f"NONJSON status={r.status_code} body={body[:80]!r}"

async def main():
    sem = asyncio.Semaphore(12)
    async def g(c):
        async with sem: return await one(c)
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30, read=900, write=60, pool=60)) as c:
        for r in await asyncio.gather(*[g(c) for _ in range(24)]):
            print(r)

asyncio.run(main())
```

Expected under load (pre-fix): a fraction come back `EMPTY_BODY status=303` and/or
`NONJSON status=500 body='modal-http: ... terminated by signal'`. At low
concurrency: all `OK`.

## Verification (after the fix)

- Re-run the burst; transient statuses should be **retried and recovered** (0 hard
  failures at the client), and the endpoint should stop emitting `500 … terminated
  by signal` once concurrency/memory are tuned.
- In production logs, the `llm.json_decode_error` with `content:"None"` events should
  disappear (replaced by a small number of `llm.transport_retry` that then succeed).
