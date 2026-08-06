"""Per-cell run clock — how long a cell has actually spent EXECUTING.

The wall span from a cell's first event to its `run.done` is NOT its runtime: it
also holds every hard/soft pause, every hour parked at a step gate, and the dead
stretch between a crash and the resume that recovered it. None of that is
recoverable from the event log — a resume TRUNCATES the trailing `run.paused`
(see `_start_cell`), a crash leaves no marker at all, and the gap between two
events can't separate the two cases either way: a single call can legitimately
run silent for the better part of an hour (the compat transport allows a 2h read),
while a pause can last twenty seconds.

So runtime is MEASURED, not inferred. A supervisor loop (`_clock_loop`) appends
one heartbeat per `TICK_S` for every cell that is genuinely running, and `read`
folds them back: adjacent ticks from the same process count as continuous work,
anything else starts a new span. A pause, cap trip, crash, or restart simply
stops the heartbeat, so its gap is never credited — which is what makes the total
accurate to within one tick per interruption and structurally unable to
over-count.

Deliberately a sidecar (`<runs>/<scene>/timing.jsonl`) rather than events in the
cell's own log: heartbeats would inflate every event index, flood the SSE stream
and the observability tree, and a tick landing after a terminal marker would
break the resume path's terminal-sentinel truncation. Co-located with the cell
exactly like `flights.db`, so cell reset/copy/delete (all directory-level) drop
or carry it for free.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NAME = "timing.jsonl"

# Heartbeat cadence, and the widest gap between two ticks still read as
# continuous work. The slack absorbs a loop delayed by a busy event loop; a real
# interruption is always far wider, since stopping the ticks at all takes a
# cancelled task or a dead process.
TICK_S = 5.0
_MAX_GAP_S = TICK_S * 2.4

# One token per process. Monotonic clocks are comparable only within a process,
# so a tick pair is credited only when both sides carry this token — which is
# also what stops a restart's first tick from being credited against the last
# tick of the process that died.
_SESSION = secrets.token_hex(4)

# Folded results keyed by path and stamped with the log's (size, mtime). The
# board re-reads every visible cell on every poll and a finished cell's file
# never changes again; a running cell re-folds once per tick, which is what keeps
# its displayed runtime advancing.
_cache: dict[Path, tuple[tuple[int, float], dict[str, Any]]] = {}

_EMPTY: dict[str, Any] = {"active_s": 0.0, "spans": 0, "first_t": None, "last_t": None}


def _runs_dir() -> Path:
    return Path(os.environ.get("STARSHOT_RUNS_DIR", _REPO_ROOT / "runs"))


def path(scene: str) -> Path:
    """The heartbeat log for `scene` — the composite `<run>/<slot>/<model>` id
    that doubles as the cell's path under the runs dir."""
    return _runs_dir() / scene / _NAME


def tick(scene: str) -> None:
    """Append one heartbeat for `scene`. Open-write-close, so no handle lingers
    on a cell that a reset or copy may replace, and every fault is swallowed — a
    clock that cannot write must never disturb the run it is measuring."""
    with contextlib.suppress(OSError, ValueError):
        p = path(scene)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8", newline="") as f:
            f.write(json.dumps({
                "t": round(time.time(), 3),
                "m": round(time.monotonic(), 3),
                "sid": _SESSION,
            }) + "\n")


def read(scene: str) -> dict[str, Any]:
    """`{active_s, spans, first_t, last_t}` for `scene`:

      * `active_s` — seconds the cell was actually executing, summed across every
        span, so an interrupted cell reports the work done rather than the
        calendar time it was open.
      * `spans` — how many separate stretches that took. 1 is an uninterrupted
        run; more means it was paused/capped/crashed and resumed that many times,
        and each extra span costs at most one tick of accuracy.
      * `first_t` / `last_t` — wall clock of the first and last heartbeat, for
        display only (the credited arithmetic is all monotonic).

    A cell that never ran has no heartbeat log and reads as zeros. A torn final
    line — the process was killed mid-append — is skipped."""
    p = path(scene)
    try:
        st = p.stat()
    except OSError:
        return _EMPTY
    sig = (st.st_size, st.st_mtime)
    hit = _cache.get(p)
    if hit is not None and hit[0] == sig:
        return hit[1]
    active = 0.0
    spans = 0
    first_t: float | None = None
    last_t: float | None = None
    prev_m: float | None = None
    prev_sid: object = None
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    t, m, sid = float(event["t"]), float(event["m"]), event.get("sid")
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                gap = None if prev_m is None else m - prev_m
                if gap is not None and sid == prev_sid and 0.0 <= gap <= _MAX_GAP_S:
                    active += gap
                else:
                    spans += 1
                if first_t is None:
                    first_t = t
                last_t = t
                prev_m, prev_sid = m, sid
    except OSError:
        return _EMPTY
    out = {"active_s": round(active, 1), "spans": spans, "first_t": first_t, "last_t": last_t}
    _cache[p] = (sig, out)
    return out
