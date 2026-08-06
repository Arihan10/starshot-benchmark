"""Simulation test for the SQLite flight-ledger. NO real API calls.

Exercises the new per-scene-DB system end to end:
  1. capture      — a real `call_llm_once` (compat MockTransport) writes a row
                    into its scene DB with the exact system/user/output.
  2. rotation     — a 429 then success: two attempt rows, one call, prompts on
                    the winning row only.
  3. pagination   — 250 rows in one scene paginate by 100 (keyset, newest
                    first), no dupes, strictly descending.
  4. facets       — server-side distinct values + counts + total.
  5. big run      — 12 scenes (> the 8-attach batch) merge into one correct
                    newest-first paginated stream.

Usage (from server/):
    uv run python scripts/test_flightlog.py
"""
# ruff: noqa: E402 — STARSHOT_RUNS_DIR must be set before app modules import.

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_TMP = Path(tempfile.mkdtemp(prefix="flightlog-sqlite-"))
os.environ["STARSHOT_RUNS_DIR"] = str(_TMP)

with contextlib.suppress(AttributeError, OSError, ValueError):
    sys.stdout.reconfigure(encoding="utf-8")

import httpx
from pydantic import BaseModel

from app.core.slots import OPENAI_COMPAT_MODELS, OpenAICompatModel
from app.services import llm
from app.utils import flightlog, keypool
from app.utils import logging as rlog
from app.utils.logging import SlotLog

SIM_MODEL_ID = "sim/flights"
SIM_ENV = "SIM_FLIGHT_API_KEY"
KEYS = ["key-alpha-0000", "key-bravo-1111"]

_real_async_client = httpx.AsyncClient
_real_sleep = asyncio.sleep
_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"    {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        _failures.append(f"{name}: {detail}")


class SimOut(BaseModel):
    answer: str


class FakeProvider:
    def __init__(self, script: list[object]) -> None:
        self.script = list(script)

    def handle(self, request: httpx.Request) -> httpx.Response:
        action = self.script.pop(0) if self.script else 200
        if action == 429:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        key = request.headers.get("authorization", "").removeprefix("Bearer ")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({"answer": f"served by {key}"})},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })


def install_fakes(provider: FakeProvider) -> None:
    def patched_client(*a: object, **k: object):
        k["transport"] = httpx.MockTransport(provider.handle)
        return _real_async_client(*a, **k)  # type: ignore[arg-type]

    async def fast_sleep(delay: float, *a: object, **k: object):
        return await _real_sleep(min(delay * 0.001, 0.005))

    httpx.AsyncClient = patched_client  # type: ignore[misc, assignment]
    asyncio.sleep = fast_sleep  # type: ignore[assignment]


def bind(scene: str) -> None:
    rlog.bind(SlotLog(scene, _TMP / scene / "events.jsonl"))


def put(scene: str, t: float, *, status: int = 200, model: str = "m",
        key: str = "key-zzzz", transport: str = "direct") -> None:
    """Record one attempt row directly (bypassing the transport) for the bulk
    pagination/facet/merge tests."""
    bind(scene)
    flightlog.begin_call()
    flightlog.record(
        transport=transport, model=model, t_request=t, t_response=t,
        base_url="https://api.example/v1", api_key=key, status=status,
        tokens_in=10, tokens_out=5,
    )


async def call_once(scene: str, step: str = "st") -> SimOut:
    bind(scene)
    llm.set_model(SIM_MODEL_ID)
    validated, *_ = await llm.call_llm_once(
        system="SYS-BYTES", user="USER-BYTES", output_schema=SimOut,
        model=SIM_MODEL_ID, step=step, node_id="nd",
    )
    return validated


async def scenario_capture() -> None:
    print("\n[1] capture through call_llm_once")
    install_fakes(FakeProvider([200]))
    out = await call_once("run1/slotA/modelX")
    res = flightlog.page("run1", cursor=None, limit=100, filters={})
    check("one row persisted", len(res["rows"]) == 1, str(len(res["rows"])))
    r = res["rows"][0]
    check("row metadata", r["ok"] and r["model"] == "sim-flights" and r["step"] == "st" and r["node"] == "nd")
    check("slot is the scene path", r["slot"] == "run1/slotA/modelX")
    check("has_prompt flagged", r["has_prompt"] is True)
    check("key masked", r["key"] == "...0000")
    d = flightlog.detail(r["slot"], r["id"])
    check("detail has exact prompts", d is not None and d["system"] == "SYS-BYTES" and d["user"] == "USER-BYTES")
    check("detail output", d is not None and "served by" in (d["output"] or ""))
    check("detail schema", d is not None and d["schema"] == "SimOut")
    check("output flowed through", out.answer.startswith("served by"))


async def scenario_rotation() -> None:
    print("\n[2] rotation 429 -> 200")
    keypool._POOLS.clear()
    install_fakes(FakeProvider([429, 200]))
    await call_once("run1/slotB/modelX")
    res = flightlog.page("run1", cursor=None, limit=100, filters={"slot": ["run1/slotB/modelX"]})
    rows = res["rows"]
    check("two attempt rows", len(rows) == 2, str(len(rows)))
    statuses = sorted(x["status"] for x in rows)
    check("a 429 and a 200", statuses == [200, 429], str(statuses))
    check("same call id", rows[0]["call"] == rows[1]["call"])
    ok_row = next(x for x in rows if x["ok"])
    bad_row = next(x for x in rows if not x["ok"])
    check("prompt only on the winning row", ok_row["has_prompt"] and not bad_row["has_prompt"])
    check("winning key rotated", ok_row["key"] == "...1111", ok_row["key"])


def scenario_pagination() -> None:
    print("\n[3] keyset pagination (250 rows, 100/page)")
    for i in range(250):
        put("pg/s/m", 2000.0 + i * 0.01, status=200 if i % 5 else 429, model="m")
    seen: set[str] = set()
    pages = 0
    cursor = None
    last_t = float("inf")
    while True:
        res = flightlog.page("pg", cursor=cursor, limit=100, filters={})
        pages += 1
        for r in res["rows"]:
            key = f"{r['slot']}\u0000{r['id']}"
            if key in seen:
                check("no duplicate across pages", False, key)
            seen.add(key)
            if r["t_response"] > last_t + 1e-9:
                check("strictly descending by t_response", False, f"{r['t_response']} > {last_t}")
            last_t = r["t_response"]
        cursor = res["cursor"]
        if not res["has_more"]:
            break
        if pages > 10:
            check("terminates", False, "runaway pagination")
            break
    check("all 250 rows, no dupes", len(seen) == 250, str(len(seen)))
    check("exactly 3 pages (100+100+50)", pages == 3, str(pages))


def scenario_facets() -> None:
    print("\n[4] server-side facets")
    res = flightlog.facets("pg", filters={})
    check("total counts the run", res["total"] == 250, str(res["total"]))
    status = {str(x["value"]): x["count"] for x in res["facets"]["status"]}
    # i%5==0 -> 429 (50 of them), rest 200 (200).
    check("status facet split", status.get("200") == 200 and status.get("429") == 50, str(status))
    models = {x["value"]: x["count"] for x in res["facets"]["model"]}
    check("model facet", models.get("m") == 250, str(models))
    # exclude-self: filtering status=429 leaves the status facet showing all
    # statuses (computed without the status filter), but total reflects it.
    res2 = flightlog.facets("pg", filters={"status": ["429"]})
    check("filtered total", res2["total"] == 50, str(res2["total"]))


def scenario_big_run() -> None:
    print("\n[5] 12-scene run merges past the 8-attach batch")
    n_scenes, per = 12, 20
    t = 3000.0
    for s in range(n_scenes):
        for _j in range(per):
            # interleave timestamps across scenes so the global order mixes them
            put(f"big/s{s}/m", t)
            t += 0.01
    total = n_scenes * per  # 240
    seen: set[str] = set()
    cursor = None
    pages = 0
    last_t = float("inf")
    while True:
        res = flightlog.page("big", cursor=cursor, limit=100, filters={})
        pages += 1
        for r in res["rows"]:
            k = f"{r['slot']}\u0000{r['id']}"
            if k in seen:
                check("no dup in merge", False, k)
            seen.add(k)
            if r["t_response"] > last_t + 1e-9:
                check("merge stays descending", False, f"{r['t_response']} > {last_t}")
            last_t = r["t_response"]
        cursor = res["cursor"]
        if not res["has_more"] or pages > 10:
            break
    check("all 240 merged, no dupes", len(seen) == total, str(len(seen)))
    check("3 pages (100+100+40)", pages == 3, str(pages))
    fac = flightlog.facets("big", filters={})
    check("facet total across 12 scenes", fac["total"] == total, str(fac["total"]))
    check("scene facet lists 12 scenes", len(fac["facets"]["slot"]) == n_scenes, str(len(fac["facets"]["slot"])))


async def main() -> int:
    print(f"SQLite flight-ledger simulation — no real API calls (ledger at {_TMP})")
    os.environ[f"{SIM_ENV}_ARRAY"] = json.dumps(KEYS)
    os.environ["OPENROUTER_API_KEY"] = "sk-or-sim"
    keypool._POOLS.clear()
    OPENAI_COMPAT_MODELS[SIM_MODEL_ID] = OpenAICompatModel(
        model="sim-flights", base_url="https://sim.invalid/v1", api_key_env=SIM_ENV, rotate=True,
    )

    await scenario_capture()
    await scenario_rotation()
    scenario_pagination()
    scenario_facets()
    scenario_big_run()

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
