"""Pure event-log folds for a cell's board summary.

Everything the board shows for a cell — its status markers, event count, last
step, per-model token/cost usage, and spend — is a fold over that cell's
`events.jsonl`. These functions are the single definition of those folds, shared
by the live API (`api/routes.py`), the board-summary cache (`boardcache.py`), and
the pre-warm migration (`scripts/backfill_board_cache.py`).

No FastAPI / service / env imports on purpose: this is a leaf module so a
lightweight script can compute a summary without pulling in the whole app. The
spend-cap DEFAULT (an env knob) is applied by the caller — this module only
surfaces the log-derived `cap_override` and `spend`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# --- bounded event-log reader -------------------------------------------------
# Fields the folds + flight-ledger reconstruction read, projected as each line is
# parsed so a log's system/user/output/reasoning never enter memory. Kimi at
# reasoning_effort=max writes multi-MB `cache.llm` lines, so a line over
# `_BIG_LINE` is field-extracted by regex from just its HEAD (fields before the
# reasoning) + TAIL (fields after it) rather than parsed whole — bounding memory
# to a fixed slice regardless of line size.
_KEEP = (
    "kind", "key", "model", "tokens_in", "tokens_out", "generation_id", "cost",
    "cap", "node", "phase", "step", "prompt", "ts",
    "t_request", "t_response", "flight_ms", "attempts",
)
_BIG_LINE = 256_000
_TAIL = 2048
_RX_HEAD = {
    "kind": re.compile(r'"kind":\s*"([^"]+)"'),
    "key": re.compile(r'"key":\s*"([0-9a-f]{64})"'),
    "model": re.compile(r'"model":\s*"([^"]+)"'),
    "node": re.compile(r'"node":\s*"([^"]*)"'),
    "step": re.compile(r'"step":\s*"([^"]+)"'),
    "ts": re.compile(r'"ts":\s*([0-9.]+)'),
}
_RX_TAIL = {
    "tokens_in": re.compile(r'"tokens_in":\s*(\d+)'),
    "tokens_out": re.compile(r'"tokens_out":\s*(\d+)'),
    "generation_id": re.compile(r'"generation_id":\s*"([^"]+)"'),
    "t_request": re.compile(r'"t_request":\s*([0-9.]+)'),
    "t_response": re.compile(r'"t_response":\s*([0-9.]+)'),
    "flight_ms": re.compile(r'"flight_ms":\s*(\d+)'),
    "attempts": re.compile(r'"attempts":\s*(\d+)'),
}
_INT_FIELDS = ("tokens_in", "tokens_out", "flight_ms", "attempts")
_FLOAT_FIELDS = ("ts", "t_request", "t_response")


def _parse_small(line: str) -> dict | None:
    """Full parse of a line that fits in `_BIG_LINE` — so llm.cost / cap-override
    / run.start scalars are exact."""
    try:
        e = json.loads(line)
    except json.JSONDecodeError:
        return None
    return {k: e.get(k) for k in _KEEP}


def _parse_big(head: str, tail: str) -> dict:
    """Regex-extract a `cache.llm` line too big to hold in full, from its head +
    tail. cache.llm carries no cost/cap/prompt, so those stay None."""
    d: dict[str, Any] = dict.fromkeys(_KEEP)
    for field, rx in _RX_HEAD.items():
        m = rx.search(head)
        if m:
            d[field] = m.group(1)
    for field, rx in _RX_TAIL.items():
        m = rx.search(tail)
        if m:
            d[field] = m.group(1)
    for f in _INT_FIELDS:
        d[f] = int(d[f]) if d[f] is not None else None
    for f in _FLOAT_FIELDS:
        d[f] = float(d[f]) if d[f] is not None else None
    return d


def iter_events(path: Path) -> Iterator[dict]:
    """Yield slim event dicts from an events.jsonl in BOUNDED memory even for
    multi-MB lines: reads fixed binary chunks and, per line, keeps only its first
    `_BIG_LINE` bytes + a rolling last `_TAIL` bytes. A line that fits is parsed
    whole; a bigger one (a huge `cache.llm` reasoning trace) is regex-extracted
    from head+tail."""
    head = bytearray()
    tail = bytearray()
    length = 0

    def finish() -> dict | None:
        if length == 0:
            return None
        if length <= len(head):  # fit entirely in head → full parse
            return _parse_small(head.decode("utf-8", "replace"))
        return _parse_big(head.decode("utf-8", "replace"), tail.decode("utf-8", "replace"))

    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            start = 0
            while True:
                nl = chunk.find(b"\n", start)
                seg = chunk[start:nl] if nl != -1 else chunk[start:]
                length += len(seg)
                if len(head) < _BIG_LINE:
                    head += seg[: _BIG_LINE - len(head)]
                tail += seg
                if len(tail) > _TAIL:
                    del tail[: len(tail) - _TAIL]
                if nl == -1:
                    break
                ev = finish()
                if ev is not None:
                    yield ev
                head, tail, length = bytearray(), bytearray(), 0
                start = nl + 1
        ev = finish()
        if ev is not None:
            yield ev


def cost_index(events: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    """Split a cell's settled `llm.cost` events by how they join back to the
    call they price: `(by_gen, by_key)`. OpenRouter costs carry the call's
    `generation_id`; the compat backends' token-priced costs (Moonshot, Alibaba,
    ...) carry its content-hash `key` instead, since they have no OpenRouter
    generation. Last write wins, so a backfilled correction supersedes an earlier
    value for the same call."""
    by_gen: dict[str, float] = {}
    by_key: dict[str, float] = {}
    for e in events:
        if e.get("kind") != "llm.cost":
            continue
        c = e.get("cost")
        if not isinstance(c, (int, float)):
            continue
        gid, key = e.get("generation_id"), e.get("key")
        if isinstance(gid, str):
            by_gen[gid] = float(c)
        elif isinstance(key, str):
            by_key[key] = float(c)
    return by_gen, by_key


def call_cost(
    event: dict[str, Any], by_gen: dict[str, float], by_key: dict[str, float],
) -> tuple[float, bool]:
    """`(cost, pending)` for one `cache.llm` event. A token-priced compat cost
    (matched on the content-hash `key`) WINS over any OpenRouter cost for the
    same call: a BYOK leg records both — a real `generation_id` billed at $0
    through OpenRouter and the true token cost — and the token cost is the real
    one. A call with a `generation_id` but no settled cost yet is `pending` (the
    sweep hasn't priced it); one with neither key- nor gen-cost is a legacy /
    unpriced call — no cost, not pending."""
    key, gid = event.get("key"), event.get("generation_id")
    if isinstance(key, str) and key in by_key:
        return by_key[key], False
    if isinstance(gid, str):
        if gid in by_gen:
            return by_gen[gid], False
        return 0.0, True
    return 0.0, False


def usage_summary(events: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-model token, request, USD-cost, and unresolved-lookup totals from a
    cell's `cache.llm` events. Returns
    `{ model_id: {"in", "out", "req", "cost", "pending"} }`.

    `cost` comes from a call's `llm.cost` event, joined per `call_cost`: the
    compat backends are token-priced at call time (keyed by the call's content
    hash), while OpenRouter's own settled `total_cost` lands a beat later keyed
    by `generation_id`. `pending` counts OpenRouter calls the sweep hasn't priced
    yet; it drains to 0 once the run's spend has caught up. Token-priced calls
    are never pending, and a call with no cost of either kind (a legacy log) just
    counts as a request."""
    by_gen, by_key = cost_index(events)
    usage: dict[str, dict[str, float]] = {}
    for e in events:
        if e.get("kind") != "cache.llm":
            continue
        model = str(e.get("model") or "?")
        u = usage.setdefault(model, {"in": 0, "out": 0, "req": 0, "cost": 0.0, "pending": 0})
        ti, to = e.get("tokens_in"), e.get("tokens_out")
        u["in"] += int(ti) if isinstance(ti, (int, float)) else 0
        u["out"] += int(to) if isinstance(to, (int, float)) else 0
        u["req"] += 1
        cost, pending = call_cost(e, by_gen, by_key)
        u["cost"] += cost
        if pending:
            u["pending"] += 1
    return usage


def cell_spend(events: list[dict[str, Any]]) -> float:
    """Total settled spend, attributed per call so a BYOK leg counts once at its
    true token cost (not that cost PLUS its $0 OpenRouter record)."""
    by_gen, by_key = cost_index(events)
    return sum(
        call_cost(e, by_gen, by_key)[0]
        for e in events
        if e.get("kind") == "cache.llm"
    )


def cap_override_value(events: list[dict[str, Any]]) -> float | None:
    """The ceiling set by the cell's most recent `run.cap_override`, or None if
    it was never overridden — the last one wins."""
    value = None
    for e in events:
        if e.get("kind") == "run.cap_override":
            c = e.get("cap")
            if isinstance(c, (int, float)):
                value = float(c)
    return value


def last_step(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The most recent pipeline-location marker — what the board cards show as
    "where is this cell right now"."""
    for e in reversed(events):
        if e.get("kind") == "step":
            return {"node": e.get("node"), "phase": e.get("phase")}
    return None


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold a cell's full event list into the compact board summary the cache
    persists and the board renders — everything derivable from the log alone.
    The runtime overlay (a live task / parked gate / the spend-cap DEFAULT) is
    applied by the caller; here we only surface the log-derived facts.

    `prompt`/`model` come from the first `run.start` so a hydrated-but-unloaded
    cell can still prefill its resume form; `has_done`/`last_is_error`/
    `events_count` are what `derive_status` needs without the events in hand."""
    count = len(events)
    prompt: str | None = None
    model: str | None = None
    has_done = False
    for e in events:
        kind = e.get("kind")
        if kind == "run.start" and prompt is None:
            p, m = e.get("prompt"), e.get("model")
            prompt = p if isinstance(p, str) else None
            model = m if isinstance(m, str) else None
        elif kind == "run.done":
            has_done = True
    return {
        "events_count": count,
        "last_kind": events[-1].get("kind") if events else None,
        "last_step": last_step(events),
        "has_done": has_done,
        "last_is_error": bool(events) and events[-1].get("kind") == "run.error",
        "prompt": prompt,
        "model": model,
        "spend": cell_spend(events),
        "cap_override": cap_override_value(events),
        "usage": usage_summary(events),
    }
