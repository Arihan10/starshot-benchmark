"""Simulation test for the rotating key system + 429 exponential backoff.

NO real API calls. Drives the REAL production path — `call_llm_once` ->
`_send_structured` -> `_send_openai_compatible` -> `app.utils.keypool` — against
an in-process fake provider mounted with `httpx.MockTransport`. The fake gives
each key a request quota and answers 429 once it's spent, so we watch the pool
roll keys and the retry loop back off exactly the way a live rate limit would.

`asyncio.sleep` is patched to (a) RECORD every requested backoff so the schedule
can be asserted and (b) shrink the real wait to microseconds so the suite is
fast.

Scenarios:
  1. env parsing        — JSON array / comma / newline pools, dedupe, single-key
                          fallback, malformed JSON raises.
  2. CAS semantics      — duplicate 429 reports for one generation advance once.
  3. sequential 429     — one key spent: rotate to the next with NO backoff.
  4. concurrent burst   — N parallel 429s roll the pool once, no backoff.
  5. rotating exhaustion— every key dead: rotate through a whole sweep with no
                          delay, THEN back off exponentially per exhausted sweep,
                          raise at the budget, and recover after a quota refill.
  6. single-key backoff — a rotate pool of ONE key (== OpenRouter shape): every
                          429 backs off exponentially, capped, never rotates.
  7. openrouter backoff — the SDK path (OpenRouterError status_code 429) backs
                          off exponentially then recovers.
  8. rotate=False       — control: the plain env key is used, pools untouched.

Usage (from server/):
    uv run python scripts/test_key_rotation.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep the flight ledger the instrumented transports write out of the real
# runs/ dir — this simulation's requests are not production traffic.
os.environ["STARSHOT_RUNS_DIR"] = tempfile.mkdtemp(prefix="keyrotation-sim-")

with contextlib.suppress(AttributeError, OSError, ValueError):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 consoles

import httpx
from openrouter.errors import OpenRouterError
from pydantic import BaseModel

from app.core.slots import OPENAI_COMPAT_MODELS, OpenAICompatModel
from app.services import llm
from app.utils import keypool

SIM_MODEL_ID = "sim/rotating"
SIM_ENV = "SIM_ROTATE_API_KEY"
KEYS = ["key-alpha-0000", "key-bravo-1111", "key-charlie-2222"]


class SimOutput(BaseModel):
    answer: str


class FakeProvider:
    """OpenAI-compatible /chat/completions stand-in. Each key gets a quota;
    a spent key answers 429 (like a rate-limit window with no reset)."""

    def __init__(self, quotas: dict[str, int]) -> None:
        self.quotas = dict(quotas)
        self.requests: list[tuple[str, int]] = []  # (key, status) per request

    def handle(self, request: httpx.Request) -> httpx.Response:
        key = request.headers.get("authorization", "").removeprefix("Bearer ")
        if self.quotas.get(key, 0) <= 0:
            self.requests.append((key, 429))
            return httpx.Response(429, json={"error": {"message": "rate limited (simulated)"}})
        self.quotas[key] -= 1
        self.requests.append((key, 200))
        body = {
            "choices": [{
                "message": {"content": json.dumps({"answer": f"served by {key}"})},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        return httpx.Response(200, json=body)

    def timeline(self) -> str:
        return "\n".join(
            f"    #{i + 1:<2} key={k.split('-')[1]:<8} -> {s}"
            for i, (k, s) in enumerate(self.requests)
        )


_real_async_client = httpx.AsyncClient
_real_sleep = asyncio.sleep

_sleeps: list[float] = []  # backoff durations requested since the last patch_sleep()
_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"    {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        _failures.append(f"{name}: {detail}")


def patch_sleep() -> None:
    """Record every requested backoff (the real, unscaled seconds) and collapse
    the actual wait to microseconds. Resets the recording for a new scenario."""
    _sleeps.clear()

    async def fast_sleep(delay: float, *args: object, **kwargs: object):
        _sleeps.append(delay)
        return await _real_sleep(min(delay * 0.001, 0.005))

    asyncio.sleep = fast_sleep  # type: ignore[assignment]


def install_fakes(provider: FakeProvider) -> None:
    """Route every httpx.AsyncClient at the fake provider + patch sleep."""
    patch_sleep()

    def patched_client(*args: object, **kwargs: object):
        kwargs["transport"] = httpx.MockTransport(provider.handle)
        return _real_async_client(*args, **kwargs)  # type: ignore[arg-type]

    httpx.AsyncClient = patched_client  # type: ignore[misc, assignment]


def fresh_pool(keys: list[str] | None = None) -> None:
    """Reset pool state + env between scenarios."""
    keypool._POOLS.clear()
    os.environ.pop(SIM_ENV, None)
    os.environ.pop(f"{SIM_ENV}_ARRAY", None)
    if keys is not None:
        os.environ[f"{SIM_ENV}_ARRAY"] = json.dumps(keys)


async def call_once(model: str = SIM_MODEL_ID) -> SimOutput:
    validated, _reasoning, _usage, _raw, _gids = await llm.call_llm_once(
        system="You are a simulation.",
        user="ping",
        output_schema=SimOutput,
        model=model,
        log_retries=False,  # runs outside any SlotLog binding
    )
    return validated


def scenario_env_parsing() -> None:
    print("\n[1] env parsing")
    fresh_pool()
    os.environ[f"{SIM_ENV}_ARRAY"] = '["a-key", "b-key", "a-key", ""]'
    pool = keypool.get_pool(SIM_ENV)
    check("JSON array parses + dedupes", pool.size == 2, f"size={pool.size}")

    fresh_pool()
    os.environ[f"{SIM_ENV}_ARRAY"] = "k1, k2,\nk3"
    pool = keypool.get_pool(SIM_ENV)
    check("comma/newline list parses", pool.size == 3, f"size={pool.size}")

    fresh_pool()
    os.environ[SIM_ENV] = "solo-key"
    pool = keypool.get_pool(SIM_ENV)
    check("falls back to single env var", pool.size == 1 and pool.current()[1] == "solo-key")

    fresh_pool()
    os.environ[f"{SIM_ENV}_ARRAY"] = '["unterminated'
    try:
        keypool.get_pool(SIM_ENV)
        check("malformed JSON array raises", False, "no exception")
    except RuntimeError as e:
        check("malformed JSON array raises", "not a valid JSON array" in str(e))

    fresh_pool()
    try:
        keypool.get_pool(SIM_ENV)
        check("missing key raises", False, "no exception")
    except RuntimeError:
        check("missing key raises", True)


def scenario_cas() -> None:
    print("\n[2] compare-and-swap rotation")
    fresh_pool(KEYS)
    pool = keypool.get_pool(SIM_ENV)
    gen, key = pool.current()
    check("starts on key[0]", gen == 0 and key == KEYS[0])

    # Three tasks saw a 429 on the same generation; only the first advances.
    results = [pool.rotate(gen), pool.rotate(gen), pool.rotate(gen)]
    check("burst of stale reports advances once", results == [True, False, False], f"{results}")
    check("now on key[1], generation 1", pool.current() == (1, KEYS[1]), f"{pool.current()}")

    fresh_pool()
    os.environ[SIM_ENV] = "solo-key"
    solo = keypool.get_pool(SIM_ENV)
    check("single-key pool never rotates", solo.rotate(0) is False and solo.generation == 0)


async def scenario_sequential() -> None:
    print("\n[3] sequential rotation (rotate with NO backoff)")
    fresh_pool(KEYS)
    provider = FakeProvider({KEYS[0]: 2, KEYS[1]: 100, KEYS[2]: 100})
    install_fakes(provider)

    answers = [await call_once() for _ in range(4)]
    pool = keypool.get_pool(SIM_ENV)
    print(provider.timeline())
    expected = [
        (KEYS[0], 200), (KEYS[0], 200),          # quota of key[0]
        (KEYS[0], 429), (KEYS[1], 200),          # limit hit -> rotate -> retry
        (KEYS[1], 200),                          # sticks to key[1]
    ]
    check("request timeline matches", provider.requests == expected)
    check("pool advanced exactly once", pool.generation == 1)
    check("rotation did NOT back off (fresh key available)", _sleeps == [], f"{_sleeps}")
    check(
        "all calls returned valid output",
        [a.answer for a in answers]
        == [f"served by {k}" for k in (KEYS[0], KEYS[0], KEYS[1], KEYS[1])],
    )


async def scenario_concurrent() -> None:
    print("\n[4] concurrent 429 burst")
    fresh_pool(KEYS)
    provider = FakeProvider({KEYS[0]: 0, KEYS[1]: 100, KEYS[2]: 100})
    install_fakes(provider)

    n = 8
    answers = await asyncio.gather(*(call_once() for _ in range(n)))
    pool = keypool.get_pool(SIM_ENV)
    print(provider.timeline())
    hits_429 = [(k, s) for k, s in provider.requests if s == 429]
    check("every call completed", len(answers) == n and all(a.answer for a in answers))
    check(
        "pool advanced exactly once despite parallel 429s",
        pool.generation == 1,
        f"generation={pool.generation}",
    )
    check("no backoff — a single rotation cleared it", _sleeps == [], f"{_sleeps}")
    check(
        "all 429s were on key[0]",
        len(hits_429) >= 1 and all(k == KEYS[0] for k, _ in hits_429),
        f"{len(hits_429)} x 429",
    )
    check(
        "all successes on key[1]",
        all(k == KEYS[1] for k, s in provider.requests if s == 200),
    )


async def scenario_rotating_exhaustion() -> None:
    print("\n[5] rotating exhaustion: sweep-then-backoff + wrap recovery")
    fresh_pool(KEYS)
    provider = FakeProvider(dict.fromkeys(KEYS, 0))  # every key dead
    install_fakes(provider)

    raised: httpx.HTTPStatusError | None = None
    try:
        await call_once()
    except httpx.HTTPStatusError as e:
        raised = e
    pool = keypool.get_pool(SIM_ENV)
    exhaustion_sleeps = list(_sleeps)
    print(provider.timeline())
    check(
        "raises 429 at the rate-limit budget (8 attempts)",
        raised is not None
        and raised.response.status_code == 429
        and len(provider.requests) == 8,
        f"attempts={len(provider.requests)}",
    )
    # Sweep of 3 keys with no delay, then a backoff, repeated. Budget hits mid
    # third sweep, so two backoffs land: 2s (after sweep 1), 4s (after sweep 2).
    check(
        "backed off ONLY on exhausted sweeps, exponentially",
        exhaustion_sleeps == [2.0, 4.0],
        f"sleeps={exhaustion_sleeps}",
    )
    check(
        "tried every key each sweep (wrapping modulo the pool)",
        pool.generation == 8,
        f"generation={pool.generation}",
    )

    provider.quotas[KEYS[0]] = 100  # the provider's rate-limit window reset
    answer = await call_once()
    tail = provider.requests[8:]
    print(provider.timeline())
    check(
        "wraps around to the refilled key and recovers",
        answer.answer == f"served by {KEYS[0]}" and tail == [(KEYS[2], 429), (KEYS[0], 200)],
        f"tail={tail}",
    )


async def scenario_single_key_backoff() -> None:
    print("\n[6] single-key rotate pool: exponential backoff, capped, no rotation")
    fresh_pool()
    os.environ[SIM_ENV] = "solo-key-only"  # rotate=True but a pool of one
    provider = FakeProvider({"solo-key-only": 0})  # permanently rate limited
    install_fakes(provider)

    raised: httpx.HTTPStatusError | None = None
    try:
        await call_once()
    except httpx.HTTPStatusError as e:
        raised = e
    pool = keypool.get_pool(SIM_ENV)
    print(provider.timeline())
    check(
        "raises 429 at the budget (8 attempts, all on the one key)",
        raised is not None
        and raised.response.status_code == 429
        and len(provider.requests) == 8
        and all(k == "solo-key-only" for k, _ in provider.requests),
        f"attempts={len(provider.requests)}",
    )
    check("pool never rotated (one key)", pool.generation == 0 and pool.size == 1)
    check(
        "exponential backoff on EVERY 429, capped at 60s",
        _sleeps == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0],
        f"sleeps={_sleeps}",
    )


async def scenario_openrouter_backoff() -> None:
    print("\n[7] OpenRouter (SDK path) 429 -> exponential backoff then recover")
    keypool._POOLS.clear()
    patch_sleep()

    state = {"n": 0}
    fail_times = 4

    async def fake_send(*, model: str, system: str, user: str, output_schema: object):
        state["n"] += 1
        if state["n"] <= fail_times:
            req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            resp = httpx.Response(429, json={"error": {"message": "rate limited"}}, request=req)
            raise OpenRouterError("Too Many Requests", resp)
        return llm._Completion(
            content=json.dumps({"answer": "openrouter-ok"}),
            reasoning="",
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            generation_id="gen-sim",
            finish_reason="stop",
        )

    orig = llm._send_structured
    llm._send_structured = fake_send  # type: ignore[assignment]
    try:
        # A stock OpenRouter alias — not in OPENAI_COMPAT_MODELS, so pool_size 1.
        answer = await call_once(model="openai/gpt-5.5")
    finally:
        llm._send_structured = orig  # type: ignore[assignment]

    check("openrouter 429 retried then succeeded", answer.answer == "openrouter-ok")
    check("one HTTP attempt per 429 + final success", state["n"] == fail_times + 1)
    check(
        "exponential backoff on every openrouter 429 (pool_size 1)",
        _sleeps == [2.0, 4.0, 8.0, 16.0],
        f"sleeps={_sleeps}",
    )


async def scenario_rotate_off() -> None:
    print("\n[8] rotate=False control")
    keypool._POOLS.clear()
    os.environ["SIM_STATIC_API_KEY"] = "static-key-9999"
    provider = FakeProvider({"static-key-9999": 100})
    install_fakes(provider)

    static_id = "sim/static"
    OPENAI_COMPAT_MODELS[static_id] = OpenAICompatModel(
        model="sim-static", base_url="https://sim.invalid/v1", api_key_env="SIM_STATIC_API_KEY",
    )
    try:
        answer = await call_once(model=static_id)
    finally:
        OPENAI_COMPAT_MODELS.pop(static_id, None)
    check(
        "plain env key used verbatim",
        provider.requests == [("static-key-9999", 200)]
        and answer.answer == "served by static-key-9999",
    )
    check("no pool was created for it", "SIM_STATIC_API_KEY" not in keypool._POOLS)


async def main() -> int:
    print("rotating-key + 429-backoff simulation — no real API calls (httpx.MockTransport)")
    OPENAI_COMPAT_MODELS[SIM_MODEL_ID] = OpenAICompatModel(
        model="sim-rotating",
        base_url="https://sim.invalid/v1",  # never resolved; transport is mocked
        api_key_env=SIM_ENV,
        rotate=True,
    )

    scenario_env_parsing()
    scenario_cas()
    await scenario_sequential()
    await scenario_concurrent()
    await scenario_rotating_exhaustion()
    await scenario_single_key_backoff()
    await scenario_openrouter_backoff()
    await scenario_rotate_off()

    print(f"\n{'=' * 60}")
    if _failures:
        print(f"FAILED — {len(_failures)} check(s):")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
