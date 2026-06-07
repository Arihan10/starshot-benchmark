"""Concurrency + flight-time benchmark for the Runware Hunyuan 3D **Rapid** endpoint.

Drives `tencent:hunyuan-3d@3.1-rapid` under a sweep of concurrency levels and
reports, per level: completed / failed counts, flight-time stats
(mean / median / p95 / min / max), submit-ack latency, throughput (jobs/min),
and total billed cost. This characterizes both the endpoint's concurrency
ceiling (where errors begin / throughput plateaus) and its average flight time.

Rapid differs from the Pro model: a SINGLE `inputs.image` (not an array),
`settings.geometryOnly` + `settings.pbr` (no `generateType` / `faceCount`), and
a shorter `positivePrompt` (max 200 chars). geometryOnly produces a texture-free
white model, cannot combine with `pbr`, and forbids an explicit `outputFormat`.

Correctness measures:
  * Each distinct image in `test_images/` is encoded once as a base64 data URI
    and sent inline in `inputs.image` (the endpoint's transfer step rejects
    pre-uploaded image UUIDs). Submit latency therefore includes the inline
    image upload; flight time (server-side generation) is the headline metric.
  * Every job in a run uses a DISTINCT image (the model rejects a `seed`
    parameter, so distinct inputs are the only dedup guard), so Runware can't
    return a falsely-instant cache hit. If jobs exceed the image count they
    wrap and the script warns.
  * Each completed job is appended to a results JSONL immediately, so a crash
    mid-sweep never loses already-billed data, and its GLB is downloaded to
    --glb-dir right away (while the URL is fresh); download time is recorded
    separately and is NOT counted in flight time.
  * Submits within a level fire concurrently with NO client-side pacing — the
    whole point is to see whether the endpoint rate-limits.

THIS SPENDS MONEY. Rapid bills $0.225 / call, +$0.15 with PBR (geometryOnly is
texture-free and cannot combine with PBR). DRY RUN by default: prints the plan +
cost estimate and exits without calling the API. Pass --yes to execute;
--max-cost hard-caps the estimated spend (abort if exceeded).

Usage (from server/):
    # free: validate plan + cost + that images load, no API calls
    uv run python scripts/bench_hunyuan_runware.py --levels 1,4,8

    # cheapest config ($0.225/call), cap spend at $5
    uv run python scripts/bench_hunyuan_runware.py --levels 1,4,8 \
        --yes --max-cost 5 --api-key <KEY>

    # textured + PBR ($0.375/call), push concurrency
    uv run python scripts/bench_hunyuan_runware.py --levels 1,8,16,32 \
        --pbr --yes --max-cost 30 --api-key <KEY>

    # geometry-only white models (texture-free, $0.225/call)
    uv run python scripts/bench_hunyuan_runware.py --levels 1,8,16 \
        --geometry-only --yes --max-cost 15 --api-key <KEY>

    # stream N requests through the API-key pool: 1 key = serial (1 at a time),
    # K keys = K-at-a-time (cycles keys to beat the per-key 1-concurrent cap)
    uv run python scripts/bench_hunyuan_runware.py --stream 8 --pbr \
        --api-keys KEY1,KEY2 --yes --max-cost 4

The SDK caps in-flight requests per connection at RUNWARE_MAX_CONCURRENT_REQUESTS
(default 15) and holds that slot for the whole poll, so it's raised to fit the
largest level BEFORE the SDK is imported — otherwise the client, not the
endpoint, would be the bottleneck.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import math
import os
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_DIR = SERVER_DIR.parent / "test_images"

RAPID_MODEL = "tencent:hunyuan-3d@3.1-rapid"

# Pricing from Runware's Hunyuan 3D 3.1 Rapid docs: $0.225 base (text or image
# to 3D), +$0.15 for the EnablePBR add-on. geometryOnly stays at base and can't
# combine with PBR. Actual per-task cost is read back from the API (includeCost).
_RAPID_BASE_COST = 0.225
_PBR_ADDON = 0.150
_PROMPT_MAX = 200


@dataclass
class JobRecord:
    level: int
    trial: int
    index: int
    image: str
    task_uuid: str
    status: str            # "ok" | "error" | "timeout"
    submit_s: float | None  # time to receive the async submission ack
    flight_s: float | None  # submit start -> result downloaded
    cost: float | None
    glb_url: str | None
    glb_path: str | None
    download_s: float | None
    error: str | None
    ts: float


@dataclass
class _RapidSettings:
    """The `settings` block for the rapid model, serialized via the SDK's
    `to_request_dict` hook (the SDK's `ISettings` dataclass doesn't model
    `geometryOnly`). Only non-None fields are emitted."""

    geometryOnly: bool | None = None
    pbr: bool | None = None

    def to_request_dict(self) -> dict[str, Any]:
        payload = {
            k: v
            for k, v in (("geometryOnly", self.geometryOnly), ("pbr", self.pbr))
            if v is not None
        }
        return {"settings": payload} if payload else {}


class _ResultsLog:
    """Append-and-flush JSONL writer, serialized across concurrent jobs."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")
        self._lock = asyncio.Lock()

    async def write(self, kind: str, payload: dict[str, Any]) -> None:
        line = json.dumps({"kind": kind, **payload})
        async with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def _per_job_cost(pbr: bool) -> float:
    # geometryOnly does not change the price; only PBR adds an add-on (and PBR
    # is mutually exclusive with geometryOnly).
    return round(_RAPID_BASE_COST + (_PBR_ADDON if pbr else 0.0), 3)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _extract_result(result: Any) -> tuple[str, float | None]:
    """Pull the GLB URL + billed cost out of one `I3d` result. `files[0]` may be
    an SDK object or a raw dict depending on the delivery path."""
    outputs = getattr(result, "outputs", None)
    files = getattr(outputs, "files", None) if outputs is not None else None
    if not files:
        raise RuntimeError(f"result missing files: {result!r}")
    first = files[0]
    url = first.get("url") if isinstance(first, dict) else getattr(first, "url", None)
    if not url:
        raise RuntimeError(f"result missing url: {first!r}")
    return url, getattr(result, "cost", None)


async def _download_glb(http: httpx.AsyncClient, url: str, dest: Path) -> None:
    resp = await http.get(url, timeout=180.0)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


async def _safe_download(http: httpx.AsyncClient, url: str, dest: Path) -> bool:
    """Background-friendly download: never raises, prints outcome, returns ok."""
    try:
        await _download_glb(http, url, dest)
        print(f"  saved {dest.name}", flush=True)
        return True
    except Exception as e:
        print(f"  download failed {dest.name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return False


async def _run_job(
    rw: Any,
    client: Any,
    http: httpx.AsyncClient,
    *,
    image_name: str,
    image_uri: str,
    args: argparse.Namespace,
    level: int,
    trial: int,
    index: int,
    log: _ResultsLog,
    download: bool = True,
) -> JobRecord:
    settings = _RapidSettings(
        geometryOnly=args.geometry_only or None, pbr=args.pbr or None,
    )
    task = str(uuid.uuid4())
    request = rw.I3dInference(
        taskUUID=task,
        model=RAPID_MODEL,
        positivePrompt=args.prompt,
        numberResults=1,
        outputType="URL",
        # geometryOnly forbids an explicit outputFormat; the server defaults to GLB.
        outputFormat=None if args.geometry_only else "GLB",
        deliveryMethod="async",
        includeCost=True,
        inputs=rw.I3dInputs(image=image_uri),
        settings=settings,
    )

    t0 = time.monotonic()
    submit_s: float | None = None
    status, cost, glb_url, error = "ok", None, None, None
    try:
        ack = await asyncio.wait_for(
            client.inference3d(request3d=request), timeout=args.job_timeout,
        )
        submit_s = time.monotonic() - t0
        results = (
            await asyncio.wait_for(
                client.getResponse(taskUUID=task, numberResults=1),
                timeout=args.job_timeout,
            )
            if isinstance(ack, rw.IAsyncTaskResponse)
            else ack
        )
        if not results:
            raise RuntimeError("empty result list")
        glb_url, cost = _extract_result(results[0])
        flight_s = time.monotonic() - t0
    except TimeoutError:
        flight_s = time.monotonic() - t0
        status, error = "timeout", f"timed out after {flight_s:.0f}s"
    except Exception as e:
        flight_s = time.monotonic() - t0
        status, error = "error", f"{type(e).__name__}: {str(e)[:300]}"

    # Download the GLB right away (URL is fresh) — best-effort; a download
    # failure does NOT fail the already-billed job. Timed separately so flight
    # time stays a pure generation metric.
    glb_path: str | None = None
    download_s: float | None = None
    download_note = ""
    if download and status == "ok" and glb_url and not args.no_download:
        d0 = time.monotonic()
        try:
            dest = args.glb_dir / f"{Path(image_name).stem}.glb"
            await _download_glb(http, glb_url, dest)
            glb_path, download_s = str(dest), time.monotonic() - d0
        except Exception as e:
            download_note = f"  (download failed: {type(e).__name__}: {str(e)[:120]})"

    rec = JobRecord(
        level=level, trial=trial, index=index, image=image_name, task_uuid=task,
        status=status, submit_s=submit_s, flight_s=flight_s, cost=cost,
        glb_url=glb_url, glb_path=glb_path, download_s=download_s, error=error,
        ts=time.time(),
    )
    await log.write("job", asdict(rec))
    saved = f" dl={_fmt(download_s)}s -> {Path(glb_path).name}" if glb_path else download_note
    print(
        f"  [L{level:>3} t{trial} #{index:02d}] {status:7s} "
        f"flight={_fmt(flight_s):>6}s submit={_fmt(submit_s):>5}s "
        f"cost={'-' if cost is None else f'${cost:.3f}'} img={image_name}{saved}"
        + (f"  !! {error}" if error else ""),
        flush=True,
    )
    return rec


def _summarize(level: int, trial: int, records: list[JobRecord], wall: float) -> dict[str, Any]:
    ok = [r for r in records if r.status == "ok"]
    flights = [r.flight_s for r in ok if r.flight_s is not None]
    submits = [r.submit_s for r in ok if r.submit_s is not None]
    errors: dict[str, int] = {}
    for r in records:
        if r.status != "ok":
            label = r.error.split(":")[0] if r.error else r.status
            errors[label] = errors.get(label, 0) + 1
    return {
        "level": level,
        "trial": trial,
        "launched": len(records),
        "completed": len(ok),
        "failed": len(records) - len(ok),
        "wall_s": round(wall, 2),
        "throughput_per_min": round(len(ok) / wall * 60, 2) if wall > 0 and ok else 0.0,
        "flight_mean": round(statistics.mean(flights), 1) if flights else None,
        "flight_median": round(statistics.median(flights), 1) if flights else None,
        "flight_p95": round(_percentile(flights, 0.95), 1) if flights else None,
        "flight_min": round(min(flights), 1) if flights else None,
        "flight_max": round(max(flights), 1) if flights else None,
        "submit_mean": round(statistics.mean(submits), 2) if submits else None,
        "cost_billed": round(sum(r.cost or 0.0 for r in ok), 3),
        "errors": errors,
    }


async def _run_level(
    rw: Any,
    client: Any,
    http: httpx.AsyncClient,
    *,
    level: int,
    trial: int,
    images: list[tuple[str, str]],
    img_offset: int,
    args: argparse.Namespace,
    log: _ResultsLog,
) -> dict[str, Any]:
    print(
        f"\n=== level {level} (trial {trial + 1}/{args.trials}) — "
        f"firing {level} concurrent jobs ===",
        flush=True,
    )
    t0 = time.monotonic()
    jobs = [
        _run_job(
            rw, client, http,
            image_name=images[(img_offset + i) % len(images)][0],
            image_uri=images[(img_offset + i) % len(images)][1],
            args=args, level=level, trial=trial, index=i, log=log,
        )
        for i in range(level)
    ]
    records = await asyncio.gather(*jobs)
    summary = _summarize(level, trial, records, time.monotonic() - t0)
    await log.write("level_summary", summary)
    print(
        f"--- level {level}: {summary['completed']}/{summary['launched']} ok, "
        f"flight mean={_fmt(summary['flight_mean'])}s "
        f"p95={_fmt(summary['flight_p95'])}s, "
        f"throughput={_fmt(summary['throughput_per_min'])}/min, "
        f"billed=${summary['cost_billed']:.3f}"
        + (f", errors={summary['errors']}" if summary["errors"] else ""),
        flush=True,
    )
    return summary


def _stream_summary(
    records: list[JobRecord], total_wall: float, saved: int, keys: int,
) -> dict[str, Any]:
    ok = [r for r in records if r.status == "ok"]
    flights = [r.flight_s for r in ok if r.flight_s is not None]
    submits = [r.submit_s for r in ok if r.submit_s is not None]
    gen_s = round(sum(flights), 1)
    capacity = keys * total_wall  # total GPU-seconds available across the key pool
    errors: dict[str, int] = {}
    for r in records:
        if r.status != "ok":
            label = r.error.split(":")[0] if r.error else r.status
            errors[label] = errors.get(label, 0) + 1
    return {
        "requests": len(records),
        "keys": keys,
        "completed": len(ok),
        "failed": len(records) - len(ok),
        "total_wall_s": round(total_wall, 1),
        "generation_s": gen_s,
        "pool_idle_s": round(capacity - gen_s, 1),
        "utilization": round(gen_s / capacity, 3) if capacity > 0 else None,
        "sustained_per_min": round(len(ok) / total_wall * 60, 2) if total_wall > 0 and ok else 0.0,
        "flight_mean": round(statistics.mean(flights), 1) if flights else None,
        "flight_median": round(statistics.median(flights), 1) if flights else None,
        "flight_p95": round(_percentile(flights, 0.95), 1) if flights else None,
        "flight_min": round(min(flights), 1) if flights else None,
        "flight_max": round(max(flights), 1) if flights else None,
        "submit_mean": round(statistics.mean(submits), 2) if submits else None,
        "cost_billed": round(sum(r.cost or 0.0 for r in ok), 3),
        "glbs_saved": saved,
        "errors": errors,
    }


def _print_stream_summary(s: dict[str, Any], args: argparse.Namespace) -> None:
    keys = s["keys"]
    print("\n" + "=" * 78)
    print(f"STREAM SUMMARY -- {s['requests']} requests via {keys} key(s) (up to {keys} concurrent)")
    print("=" * 78)
    print(
        f"  completed:       {s['completed']}/{s['requests']}"
        + (f"   FAILURES: {s['errors']}" if s["failed"] else "   (no failures)")
    )
    print(
        f"  flight/request:  mean {_fmt(s['flight_mean'])}s  median {_fmt(s['flight_median'])}s  "
        f"p95 {_fmt(s['flight_p95'])}s  min {_fmt(s['flight_min'])}s  max {_fmt(s['flight_max'])}s"
    )
    print(f"  submit-ack mean: {_fmt(s['submit_mean'])}s")
    print(
        f"  total wall:      {s['total_wall_s']}s "
        f"(generation {s['generation_s']}s across {keys} key(s); pool idle {s['pool_idle_s']}s)"
    )
    print(f"  utilization:     {s['utilization']}  (1.0 = all {keys} key(s) always busy)")
    print(f"  sustained rate:  {s['sustained_per_min']} requests/min")
    print(f"  GLBs saved:      {s['glbs_saved']}")
    print(f"  billed:          ${s['cost_billed']:.3f}")
    print(f"  per-request log: {args.out}")


class _ClientPool:
    """Key-cycling pool: one Runware client per API key. Hunyuan caps each key
    at 1 in-flight 3D task, so handing out one client per key gives len(keys)
    true concurrency with no JobNumExceed. `acquire` blocks until a key is free;
    `release` returns it to the free queue (so the next request cycles onto it).
    """

    def __init__(self, rw: Any, keys: list[str], timeout: int) -> None:
        self._clients = [
            rw.Runware(api_key=k, timeout=timeout, max_retries=0) for k in keys
        ]
        self._free: asyncio.Queue[Any] = asyncio.Queue()

    @property
    def size(self) -> int:
        return len(self._clients)

    async def connect(self) -> None:
        for c in self._clients:
            await c.connect()
            self._free.put_nowait(c)

    async def disconnect(self) -> None:
        for c in self._clients:
            with contextlib.suppress(Exception):
                await c.disconnect()

    async def acquire(self) -> Any:
        return await self._free.get()

    def release(self, client: Any) -> None:
        self._free.put_nowait(client)


async def _run_stream(
    rw: Any,
    pool: _ClientPool,
    http: httpx.AsyncClient,
    *,
    images: list[tuple[str, str]],
    args: argparse.Namespace,
    log: _ResultsLog,
) -> None:
    n = args.stream
    k = pool.size
    desc = "one at a time (single key)" if k == 1 else f"{k} at a time (cycling {k} keys, 1 task/key)"
    print(f"\n=== stream: {n} requests, {desc} ===", flush=True)
    if n > len(images):
        print(
            f"WARNING: {n} requests exceed {len(images)} distinct images; "
            "images will repeat (Runware may dedup repeats into cache hits).",
            flush=True,
        )
    records: list[JobRecord] = []
    downloads: list[asyncio.Task[bool]] = []
    t0 = time.monotonic()

    async def _one(idx: int) -> None:
        name, uri = images[idx % len(images)]
        client = await pool.acquire()
        try:
            # Submit + await generation only, then release the key immediately so
            # the next request can reuse it; the GLB is fetched in the background
            # (over the shared HTTP client, not the key slot).
            rec = await _run_job(
                rw, client, http, image_name=name, image_uri=uri, args=args,
                level=k, trial=0, index=idx, log=log, download=False,
            )
        finally:
            pool.release(client)
        records.append(rec)
        if rec.status == "ok" and rec.glb_url and not args.no_download:
            dest = args.glb_dir / f"{Path(name).stem}.glb"
            downloads.append(asyncio.create_task(_safe_download(http, rec.glb_url, dest)))

    await asyncio.gather(*[_one(i) for i in range(n)])
    total_wall = time.monotonic() - t0

    saved = 0
    if downloads:
        print(f"\nawaiting {len(downloads)} background GLB download(s)...", flush=True)
        saved = sum(1 for r in await asyncio.gather(*downloads, return_exceptions=True) if r is True)
    summary = _stream_summary(records, total_wall, saved, k)
    await log.write("stream_summary", summary)
    _print_stream_summary(summary, args)


async def _run_sweep(
    rw: Any, args: argparse.Namespace, image_paths: list[Path], keys: list[str],
) -> None:
    log = _ResultsLog(args.out)
    summaries: list[dict[str, Any]] = []
    await log.write("run_config", {
        "model": RAPID_MODEL,
        "geometry_only": args.geometry_only, "pbr": args.pbr,
        "levels": args.levels, "trials": args.trials,
        "stream": args.stream, "keys": len(keys),
        "per_job_cost": _per_job_cost(args.pbr),
        "max_concurrent_requests": os.environ.get("RUNWARE_MAX_CONCURRENT_REQUESTS"),
        "ts": time.time(),
    })
    if not args.no_download:
        args.glb_dir.mkdir(parents=True, exist_ok=True)
    print(f"encoding {len(image_paths)} images as data URIs (one-time)...", flush=True)
    images = [(p.name, _data_uri(p.read_bytes())) for p in image_paths]

    async with httpx.AsyncClient(follow_redirects=True) as http:
        if args.stream > 0:
            pool = _ClientPool(rw, keys, args.timeout)
            try:
                await pool.connect()
                await _run_stream(rw, pool, http, images=images, args=args, log=log)
            finally:
                await pool.disconnect()
                log.close()
            return

        # Level sweep probes a SINGLE key's concurrency (where JobNumExceed hits).
        total_jobs = sum(args.levels) * args.trials
        if total_jobs > len(images):
            print(
                f"WARNING: {total_jobs} jobs exceed {len(images)} distinct images; "
                "images will repeat and Runware may dedup repeats into cache hits.",
                flush=True,
            )
        client = rw.Runware(api_key=keys[0], timeout=args.timeout, max_retries=0)
        try:
            await client.connect()
            cursor = 0
            for level in args.levels:
                for trial in range(args.trials):
                    summaries.append(
                        await _run_level(
                            rw, client, http, level=level, trial=trial, images=images,
                            img_offset=cursor, args=args, log=log,
                        )
                    )
                    cursor += level
        finally:
            with contextlib.suppress(Exception):
                await client.disconnect()
            log.close()

    _print_final(summaries, args)


def _print_final(summaries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    print("\n" + "=" * 100)
    print("SUMMARY  (flight times in seconds)")
    print("=" * 100)
    header = (
        f"{'level':>5} {'trial':>5} {'ok':>4} {'fail':>4} "
        f"{'mean':>7} {'median':>7} {'p95':>7} {'min':>7} {'max':>7} "
        f"{'submit':>7} {'thr/min':>8} {'cost$':>7}"
    )
    print(header)
    print("-" * len(header))
    total_cost = 0.0
    for s in summaries:
        total_cost += s["cost_billed"]
        print(
            f"{s['level']:>5} {s['trial']:>5} {s['completed']:>4} {s['failed']:>4} "
            f"{_fmt(s['flight_mean']):>7} {_fmt(s['flight_median']):>7} "
            f"{_fmt(s['flight_p95']):>7} {_fmt(s['flight_min']):>7} "
            f"{_fmt(s['flight_max']):>7} {_fmt(s['submit_mean']):>7} "
            f"{_fmt(s['throughput_per_min']):>8} {s['cost_billed']:>7.3f}"
        )
    print("-" * len(header))
    print(f"total billed (from API includeCost): ${total_cost:.3f}")
    print(f"per-job results: {args.out}")

    first_err = next((s for s in summaries if s["failed"] > 0), None)
    if first_err is not None:
        print(
            f"\n>> errors first appeared at level {first_err['level']} "
            f"({first_err['failed']}/{first_err['launched']} failed: {first_err['errors']}) "
            "— likely at/above the endpoint's concurrency or rate ceiling."
        )
    else:
        print(
            f"\n>> no failures up to level {max(s['level'] for s in summaries)} "
            "— the concurrency ceiling was not reached; raise --levels to probe higher."
        )
    best = max(summaries, key=lambda s: s["throughput_per_min"] or 0.0)
    print(
        f">> peak throughput {best['throughput_per_min']}/min at level {best['level']}; "
        "watch where throughput stops rising with concurrency to spot server-side parallelism."
    )


def _load_image_paths(image_dir: Path, max_images: int) -> list[Path]:
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() == ".png")
    for p in paths:  # surface unreadable files now, not mid-paid-run
        p.read_bytes()
    return paths[:max_images]


def main() -> None:
    # Windows consoles default to cp1252, which can't encode some glyphs we
    # print; force UTF-8 so output never crashes the run mid-sweep.
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Concurrency + flight-time benchmark for the Runware Hunyuan 3D Rapid endpoint.",
    )
    parser.add_argument("--levels", default="1,4,8", help="Comma-separated concurrency levels. Default: 1,4,8")
    parser.add_argument("--trials", type=int, default=1, help="Repeats per level. Default: 1")
    parser.add_argument("--stream", type=int, default=0,
                        help="Stream mode: run N requests through the API-key pool "
                             "(concurrency = number of keys, 1 in-flight task per key; ignores --levels).")
    parser.add_argument("--geometry-only", action="store_true",
                        help="Texture-free white model (mutually exclusive with --pbr).")
    parser.add_argument("--pbr", action="store_true", help="Enable PBR (+$0.15; forbidden with --geometry-only).")
    parser.add_argument("--prompt", default=None, help="Optional positivePrompt (2-200 chars) for every job.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--max-images", type=int, default=50, help="Max distinct images to use. Default: 50")
    parser.add_argument("--timeout", type=int, default=600, help="Runware client timeout (s). Default: 600")
    parser.add_argument("--job-timeout", type=float, default=600.0, help="Per-job wait_for cap (s). Default: 600")
    parser.add_argument("--out", type=Path, default=Path("bench_hunyuan_rapid_results.jsonl"))
    parser.add_argument("--glb-dir", type=Path, default=Path("bench_hunyuan_rapid_glbs"),
                        help="Directory for downloaded GLBs. Default: bench_hunyuan_rapid_glbs/")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading the generated GLBs.")
    parser.add_argument("--max-cost", type=float, default=5.0, help="Abort if estimate exceeds this. Default: $5")
    parser.add_argument("--api-key", default=None, help="Single Runware API key (else RUNWARE_API_KEY env / .env).")
    parser.add_argument("--api-keys", default=None,
                        help="Comma-separated API keys for the cycling pool; stream concurrency = key count "
                             "(each key tolerates 1 in-flight task, sidestepping the per-key limit).")
    parser.add_argument("--yes", action="store_true", help="Actually execute (and spend). Default: dry run.")
    args = parser.parse_args()

    args.levels = [int(x) for x in args.levels.split(",") if x.strip()]
    if not args.levels or any(level <= 0 for level in args.levels):
        parser.error("--levels must be positive integers, e.g. 1,4,8")
    if args.geometry_only and args.pbr:
        parser.error("--pbr cannot be used with --geometry-only (texture-free model)")
    if args.prompt is not None and not 2 <= len(args.prompt) <= _PROMPT_MAX:
        parser.error(f"--prompt must be 2-{_PROMPT_MAX} chars (rapid limit)")
    if args.stream < 0:
        parser.error("--stream must be >= 0")

    load_dotenv(SERVER_DIR / ".env")
    # Resolve the API-key pool: --api-keys (cycling pool) > --api-key > env.
    if args.api_keys:
        keys = [k.strip() for k in args.api_keys.split(",") if k.strip()]
    elif args.api_key:
        keys = [args.api_key]
    else:
        env_key = os.environ.get("RUNWARE_API_KEY")
        keys = [env_key] if env_key else []
    # Must be set BEFORE importing runware: it's read into each client's
    # per-connection semaphore. The single-key level sweep fires up to
    # max(levels); pool clients each run one task at a time.
    os.environ["RUNWARE_MAX_CONCURRENT_REQUESTS"] = str(max(args.levels) + 16)

    import runware as rw

    if not args.image_dir.is_dir():
        parser.error(f"image dir not found: {args.image_dir}")
    image_paths = _load_image_paths(args.image_dir, args.max_images)
    if not image_paths:
        parser.error(f"no .png images in {args.image_dir}")

    per_job = _per_job_cost(args.pbr)
    total_jobs = args.stream if args.stream > 0 else sum(args.levels) * args.trials
    est_cost = round(per_job * total_jobs, 2)

    print("Runware Hunyuan 3D Rapid benchmark plan")
    print(f"  model:          {RAPID_MODEL}")
    print(f"  settings:       geometryOnly={args.geometry_only} pbr={args.pbr}")
    if args.stream > 0:
        print(f"  mode:           stream of {args.stream} requests via {len(keys)} key(s) "
              f"(up to {len(keys)} concurrent, 1 task/key)")
    else:
        print(f"  levels:         {args.levels}  x {args.trials} trial(s)  (single key)")
    print(f"  api keys:       {len(keys)}")
    print(f"  images:         {len(image_paths)} distinct from {args.image_dir}")
    print(f"  total jobs:     {total_jobs}")
    print(f"  per-job cost:   ${per_job:.3f}")
    print(f"  EST. MAX SPEND: ${est_cost:.2f}")
    print(f"  results -> {args.out}")
    print(f"  GLBs ->    {'(download disabled)' if args.no_download else args.glb_dir}")

    if not args.yes:
        print("\n[DRY RUN] no API calls made. Re-run with --yes to execute and spend.")
        return
    if est_cost > args.max_cost:
        print(
            f"\nABORT: estimate ${est_cost:.2f} exceeds --max-cost ${args.max_cost:.2f}. "
            "Lower --levels/--trials or raise --max-cost.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not keys:
        print("\nERROR: no API key. Pass --api-key / --api-keys or set RUNWARE_API_KEY.", file=sys.stderr)
        sys.exit(1)

    print(f"\nexecuting (spending up to ~${est_cost:.2f})...")
    asyncio.run(_run_sweep(rw, args, image_paths, keys))


if __name__ == "__main__":
    main()
