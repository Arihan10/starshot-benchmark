"""Concurrency + flight-time benchmark for Tencent Cloud's **direct** Hunyuan 3D
Rapid endpoint (`SubmitHunyuanTo3DRapidJob` / `QueryHunyuanTo3DRapidJob`).

The direct-from-Tencent counterpart to `bench_hunyuan_runware.py`. That script
drives the same model resold through Runware's WebSocket SDK; this one talks to
Tencent Cloud API 3.0 itself over signed HTTPS, so it measures the source
endpoint with no reseller in the path. The transport is fundamentally different:

  * Auth is TC3-HMAC-SHA256 request signing (SecretId/SecretKey), implemented
    inline with stdlib `hmac`/`hashlib` — no SDK dependency. Region/Action/
    Version/Timestamp ride in `X-TC-*` headers; the JSON body carries only the
    business params. The signed `content-type`/`host` must match what's sent, so
    the body is serialized once and posted as exact bytes.
  * It's submit-then-poll, not a single awaited call: `SubmitHunyuanTo3DRapidJob`
    returns a `JobId` (fast ack), then `QueryHunyuanTo3DRapidJob` is polled until
    `Status` is `DONE`/`FAIL`. Flight time = submit start -> DONE.
  * Input is `ImageBase64` (plain base64, NOT a data URI — that's a Runware-ism).
  * Tencent doesn't return a per-call price, so there's no `includeCost` readback;
    spend is gated only by dry-run-by-default + an optional `--price-per-call`.

Two Tencent quotas shape the measurement (both from the API docs):
  * **1 concurrent task by default** — the account processes generations
    sequentially, so firing N jobs at once doesn't parallelize generation; later
    jobs sit in `WAIT` until earlier ones finish. This shows up as stacked flight
    times and a near-flat throughput curve, NOT as errors. To expose it, the
    script records `queue_wait` (submit -> first observed `RUN`) per job.
  * **20 QPS** on the API itself (submit + every poll). This is a client-pacing
    limit, not an interesting endpoint property, and tripping it would pollute the
    benchmark with spurious `RequestLimitExceeded` errors on polling. A shared
    token-bucket paces ALL signed calls just under 20 QPS so the run observes
    generation-concurrency behavior cleanly. Transient poll errors are absorbed
    (re-poll) so an already-submitted job is never failed by a network blip;
    submit failures and `FAIL` statuses ARE recorded, like the Runware bench.

Correctness measures (mirroring the Runware bench):
  * Every job uses a DISTINCT image from `test_images/`, encoded once to base64,
    so results aren't deduped server-side. If jobs exceed the image count they
    wrap and the script warns.
  * Each completed job is appended to a results JSONL immediately (a crash
    mid-sweep never loses data) and its asset is downloaded to --glb-dir right
    away while the signed URL is fresh; download time is recorded separately and
    is NOT counted in flight time.
  * Submits within a level fire concurrently (subject only to the QPS pacer) — no
    artificial spacing beyond the shared rate limit.

THIS SPENDS MONEY / QUOTA. DRY RUN by default: prints the plan and exits without
calling the API. Pass --yes to execute.

Usage (from server/):
    # free: validate the plan + that images load, no API calls
    uv run python scripts/bench_hunyuan_tencent.py --levels 1,4,8

    # one real connect-proof generation (cheapest possible real run)
    uv run python scripts/bench_hunyuan_tencent.py --stream 1 --yes \
        --secret-id <ID> --secret-key <KEY>

    # concurrency sweep to watch the "1 concurrent task" queue wall
    uv run python scripts/bench_hunyuan_tencent.py --levels 1,4,8,16 --yes \
        --secret-id <ID> --secret-key <KEY>

    # textured + PBR; geometry-only white model is mutually exclusive with --pbr
    uv run python scripts/bench_hunyuan_tencent.py --stream 4 --pbr --yes ...
    uv run python scripts/bench_hunyuan_tencent.py --stream 4 --geometry-only --yes ...

Credentials resolve from --secret-id/--secret-key, else TENCENTCLOUD_SECRET_ID /
TENCENTCLOUD_SECRET_KEY (env or server/.env). Region defaults to ap-singapore
(--region / TENCENTCLOUD_REGION) — the only region that currently hosts this
action on the intl host; most other regions return UnsupportedRegion.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_DIR = SERVER_DIR.parent / "test_images"

_HOST = "hunyuan.intl.tencentcloudapi.com"
_ENDPOINT = f"https://{_HOST}"
_SERVICE = "hunyuan"
_VERSION = "2023-09-01"
_ALGORITHM = "TC3-HMAC-SHA256"
_CONTENT_TYPE = "application/json; charset=utf-8"

_SUBMIT_ACTION = "SubmitHunyuanTo3DRapidJob"
_QUERY_ACTION = "QueryHunyuanTo3DRapidJob"

_RESULT_FORMATS = ("OBJ", "GLB", "STL", "USDZ", "FBX", "MP4", "GIF")
_DOWNLOAD_TIMEOUT = 180.0
_MAX_IMAGE_BYTES = 6 * 1024 * 1024  # ImageBase64 cap is 6MB pre-encode


@dataclass(frozen=True)
class _Creds:
    secret_id: str
    secret_key: str


@dataclass
class JobRecord:
    level: int
    trial: int
    index: int
    image: str
    job_id: str | None
    status: str               # "ok" | "error" | "timeout"
    submit_s: float | None    # submit start -> JobId ack
    queue_wait_s: float | None  # submit start -> first observed RUN (best-effort)
    flight_s: float | None    # submit start -> DONE
    polls: int
    result_type: str | None
    file_url: str | None
    file_path: str | None
    download_s: float | None
    request_id: str | None
    error: str | None
    ts: float


class TencentAPIError(RuntimeError):
    """A Tencent API 3.0 business/auth error (HTTP 200 with `Response.Error`)."""

    def __init__(self, code: str, message: str, request_id: str | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.request_id = request_id

    def is_transient(self) -> bool:
        # Rate-limit / fleeting server-side conditions worth re-polling rather
        # than failing an already-submitted job over.
        code = self.code or ""
        return (
            "RequestLimit" in code
            or code.startswith(("InternalError", "ResourceUnavailable", "FailedOperation.ServiceBusy"))
        )


class _RateLimiter:
    """Shared token-bucket pacing every signed call to <= `qps` requests/sec.

    Tencent caps this API at 20 QPS across submit + all polls; exceeding it
    yields RequestLimitExceeded that would masquerade as endpoint failures. Each
    `acquire()` reserves the next slot under a lock, then sleeps until it — so N
    concurrent submits stagger out at 1/qps spacing instead of bursting."""

    def __init__(self, qps: float) -> None:
        self._min_interval = 1.0 / qps
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            scheduled = max(time.monotonic(), self._next)
            self._next = scheduled + self._min_interval
        delay = scheduled - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)


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


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _sign_headers(creds: _Creds, region: str, action: str, payload: str, ts: int) -> dict[str, str]:
    """Build the TC3-HMAC-SHA256 headers for one API 3.0 POST. Follows Tencent's
    canonical reference exactly: signed headers are `content-type;host`, the date
    is derived from the UTC timestamp, and the service is the host's first label
    (`hunyuan`)."""
    date = datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d")

    canonical_headers = f"content-type:{_CONTENT_TYPE}\nhost:{_HOST}\n"
    signed_headers = "content-type;host"
    hashed_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_request = "\n".join(
        ["POST", "/", "", canonical_headers, signed_headers, hashed_payload]
    )

    scope = f"{date}/{_SERVICE}/tc3_request"
    string_to_sign = "\n".join(
        [_ALGORITHM, str(ts), scope, hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()]
    )

    def _hmac(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    secret_date = _hmac(("TC3" + creds.secret_key).encode("utf-8"), date)
    secret_service = _hmac(secret_date, _SERVICE)
    secret_signing = _hmac(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    authorization = (
        f"{_ALGORITHM} Credential={creds.secret_id}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return {
        "Authorization": authorization,
        "Content-Type": _CONTENT_TYPE,
        "Host": _HOST,
        "X-TC-Action": action,
        "X-TC-Timestamp": str(ts),
        "X-TC-Version": _VERSION,
        "X-TC-Region": region,
    }


def _unwrap(resp: httpx.Response) -> dict[str, Any]:
    """Validate an API 3.0 response and return the inner `Response` object.
    A signed call can return HTTP 200 yet carry `Response.Error` — surface that
    as a `TencentAPIError` so callers can distinguish transient from terminal."""
    resp.raise_for_status()
    body = resp.json()
    response = body.get("Response") if isinstance(body, dict) else None
    if not isinstance(response, dict):
        raise RuntimeError(f"unexpected API response shape: {body!r}")
    err = response.get("Error")
    if err:
        raise TencentAPIError(
            str(err.get("Code", "")), str(err.get("Message", "")), response.get("RequestId"),
        )
    return response


async def _post_signed(
    http: httpx.AsyncClient,
    limiter: _RateLimiter,
    creds: _Creds,
    region: str,
    action: str,
    params: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    payload = json.dumps(params)
    headers = _sign_headers(creds, region, action, payload, int(time.time()))
    await limiter.acquire()
    resp = await http.post(
        _ENDPOINT, headers=headers, content=payload.encode("utf-8"), timeout=timeout,
    )
    return _unwrap(resp)


def _submit_params(image_b64: str, args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {"ImageBase64": image_b64, "ResultFormat": args.format}
    if args.pbr:
        params["EnablePBR"] = True
    if args.geometry_only:
        params["EnableGeometry"] = True
    return params


def _pick_file(files: list[dict[str, Any]], want_format: str) -> tuple[str, str | None]:
    """Choose the asset matching the requested format (else the first), returning
    (url, type). Raises if no entry carries a downloadable Url."""
    want = want_format.upper()
    chosen = next((f for f in files if str(f.get("Type", "")).upper() == want), files[0])
    url = chosen.get("Url")
    if not url:
        raise RuntimeError(f"ResultFile3Ds entry has no Url: {chosen!r}")
    return url, chosen.get("Type")


async def _poll(
    http: httpx.AsyncClient,
    limiter: _RateLimiter,
    creds: _Creds,
    region: str,
    job_id: str,
    *,
    t0: float,
    interval: float,
    timeout: float,
    api_timeout: float,
) -> tuple[list[dict[str, Any]], int, float | None]:
    """Poll QueryHunyuanTo3DRapidJob until DONE. Returns (files, polls,
    queue_wait_s). Transient query errors are absorbed (re-poll); a FAIL status
    or terminal API error raises."""
    deadline = time.monotonic() + timeout
    polls = 0
    queue_wait_s: float | None = None
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"poll timed out after {timeout:.0f}s ({polls} polls)")
        await asyncio.sleep(interval)
        try:
            response = await _post_signed(
                http, limiter, creds, region, _QUERY_ACTION, {"JobId": job_id},
                timeout=api_timeout,
            )
        except TencentAPIError as e:
            if not e.is_transient():
                raise
            continue
        except httpx.HTTPError:
            continue
        polls += 1
        status = response.get("Status")
        if status == "RUN" and queue_wait_s is None:
            queue_wait_s = time.monotonic() - t0
        if status == "DONE":
            files = response.get("ResultFile3Ds") or []
            if not files:
                raise RuntimeError("status DONE but ResultFile3Ds is empty")
            return files, polls, queue_wait_s
        if status == "FAIL":
            raise RuntimeError(
                f"job FAIL: {response.get('ErrorCode')}: {response.get('ErrorMessage')}"
            )


async def _download(http: httpx.AsyncClient, url: str, dest: Path) -> None:
    resp = await http.get(url, timeout=_DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


async def _safe_download(http: httpx.AsyncClient, url: str, dest: Path) -> bool:
    try:
        await _download(http, url, dest)
        print(f"  saved {dest.name}", flush=True)
        return True
    except Exception as e:
        print(f"  download failed {dest.name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return False


async def _run_job(
    http: httpx.AsyncClient,
    limiter: _RateLimiter,
    creds: _Creds,
    region: str,
    *,
    image_name: str,
    image_b64: str,
    args: argparse.Namespace,
    level: int,
    trial: int,
    index: int,
    log: _ResultsLog,
    download: bool = True,
) -> JobRecord:
    t0 = time.monotonic()
    status, error = "ok", None
    job_id: str | None = None
    request_id: str | None = None
    submit_s: float | None = None
    queue_wait_s: float | None = None
    flight_s: float | None = None
    polls = 0
    file_url: str | None = None
    result_type: str | None = None
    try:
        ack = await _post_signed(
            http, limiter, creds, region, _SUBMIT_ACTION, _submit_params(image_b64, args),
            timeout=args.timeout,
        )
        job_id = ack.get("JobId")
        request_id = ack.get("RequestId")
        if not job_id:
            raise RuntimeError(f"submit returned no JobId: {ack!r}")
        submit_s = time.monotonic() - t0
        files, polls, queue_wait_s = await _poll(
            http, limiter, creds, region, job_id,
            t0=t0, interval=args.poll_interval, timeout=args.job_timeout, api_timeout=args.timeout,
        )
        file_url, result_type = _pick_file(files, args.format)
        flight_s = time.monotonic() - t0
    except TimeoutError as e:
        flight_s = time.monotonic() - t0
        status, error = "timeout", str(e)[:300]
    except Exception as e:
        flight_s = time.monotonic() - t0
        status, error = "error", f"{type(e).__name__}: {str(e)[:300]}"

    # Download right away (signed URL is fresh) — best-effort and timed
    # separately so flight time stays a pure generation metric.
    file_path: str | None = None
    download_s: float | None = None
    download_note = ""
    if download and status == "ok" and file_url and not args.no_download:
        d0 = time.monotonic()
        try:
            ext = (result_type or args.format).lower()
            dest = args.glb_dir / f"{Path(image_name).stem}.{ext}"
            await _download(http, file_url, dest)
            file_path, download_s = str(dest), time.monotonic() - d0
            if ext == "glb" and dest.read_bytes()[:4] != b"glTF":
                download_note = "  (warn: not glTF magic — packed/zip?)"
        except Exception as e:
            download_note = f"  (download failed: {type(e).__name__}: {str(e)[:120]})"

    rec = JobRecord(
        level=level, trial=trial, index=index, image=image_name, job_id=job_id,
        status=status, submit_s=submit_s, queue_wait_s=queue_wait_s, flight_s=flight_s,
        polls=polls, result_type=result_type, file_url=file_url, file_path=file_path,
        download_s=download_s, request_id=request_id, error=error, ts=time.time(),
    )
    await log.write("job", asdict(rec))
    saved = f" dl={_fmt(download_s)}s -> {Path(file_path).name}" if file_path else download_note
    print(
        f"  [L{level:>3} t{trial} #{index:02d}] {status:7s} "
        f"flight={_fmt(flight_s):>6}s submit={_fmt(submit_s):>5}s "
        f"qwait={_fmt(queue_wait_s):>6}s polls={polls:>2} img={image_name}{saved}"
        + (f"  !! {error}" if error else ""),
        flush=True,
    )
    return rec


def _summarize(
    level: int, trial: int, records: list[JobRecord], wall: float, price_per_call: float,
) -> dict[str, Any]:
    ok = [r for r in records if r.status == "ok"]
    flights = [r.flight_s for r in ok if r.flight_s is not None]
    submits = [r.submit_s for r in ok if r.submit_s is not None]
    qwaits = [r.queue_wait_s for r in ok if r.queue_wait_s is not None]
    polls = [r.polls for r in ok]
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
        "queue_wait_mean": round(statistics.mean(qwaits), 1) if qwaits else None,
        "polls_mean": round(statistics.mean(polls), 1) if polls else None,
        "cost_est": round(price_per_call * len(ok), 3),
        "errors": errors,
    }


async def _run_level(
    http: httpx.AsyncClient,
    limiter: _RateLimiter,
    creds: _Creds,
    region: str,
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
        f"firing {level} concurrent submits ===",
        flush=True,
    )
    t0 = time.monotonic()
    jobs = [
        _run_job(
            http, limiter, creds, region,
            image_name=images[(img_offset + i) % len(images)][0],
            image_b64=images[(img_offset + i) % len(images)][1],
            args=args, level=level, trial=trial, index=i, log=log,
        )
        for i in range(level)
    ]
    records = await asyncio.gather(*jobs)
    summary = _summarize(level, trial, records, time.monotonic() - t0, args.price_per_call)
    await log.write("level_summary", summary)
    print(
        f"--- level {level}: {summary['completed']}/{summary['launched']} ok, "
        f"flight mean={_fmt(summary['flight_mean'])}s p95={_fmt(summary['flight_p95'])}s, "
        f"qwait mean={_fmt(summary['queue_wait_mean'])}s, "
        f"throughput={_fmt(summary['throughput_per_min'])}/min"
        + (f", errors={summary['errors']}" if summary["errors"] else ""),
        flush=True,
    )
    return summary


def _stream_summary(
    records: list[JobRecord], total_wall: float, saved: int, price_per_call: float,
) -> dict[str, Any]:
    ok = [r for r in records if r.status == "ok"]
    flights = [r.flight_s for r in ok if r.flight_s is not None]
    submits = [r.submit_s for r in ok if r.submit_s is not None]
    gen_s = round(sum(flights), 1)
    errors: dict[str, int] = {}
    for r in records:
        if r.status != "ok":
            label = r.error.split(":")[0] if r.error else r.status
            errors[label] = errors.get(label, 0) + 1
    return {
        "requests": len(records),
        "completed": len(ok),
        "failed": len(records) - len(ok),
        "total_wall_s": round(total_wall, 1),
        "generation_s": gen_s,
        "idle_s": round(total_wall - gen_s, 1),
        "busy_fraction": round(gen_s / total_wall, 3) if total_wall > 0 else None,
        "sustained_per_min": round(len(ok) / total_wall * 60, 2) if total_wall > 0 and ok else 0.0,
        "flight_mean": round(statistics.mean(flights), 1) if flights else None,
        "flight_median": round(statistics.median(flights), 1) if flights else None,
        "flight_p95": round(_percentile(flights, 0.95), 1) if flights else None,
        "flight_min": round(min(flights), 1) if flights else None,
        "flight_max": round(max(flights), 1) if flights else None,
        "submit_mean": round(statistics.mean(submits), 2) if submits else None,
        "cost_est": round(price_per_call * len(ok), 3),
        "assets_saved": saved,
        "errors": errors,
    }


def _print_stream_summary(s: dict[str, Any], args: argparse.Namespace) -> None:
    print("\n" + "=" * 78)
    print(f"SERIAL STREAM SUMMARY -- {s['requests']} requests, one at a time")
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
        f"(generation {s['generation_s']}s, inter-request idle {s['idle_s']}s)"
    )
    print(f"  busy fraction:   {s['busy_fraction']}  (1.0 = constant stream, no idle between jobs)")
    print(f"  sustained rate:  {s['sustained_per_min']} requests/min")
    print(f"  assets saved:    {s['assets_saved']}")
    if args.price_per_call > 0:
        print(f"  est cost:        ${s['cost_est']:.3f}  (--price-per-call ${args.price_per_call})")
    print(f"  per-request log: {args.out}")


async def _run_stream(
    http: httpx.AsyncClient,
    limiter: _RateLimiter,
    creds: _Creds,
    region: str,
    *,
    images: list[tuple[str, str]],
    args: argparse.Namespace,
    log: _ResultsLog,
) -> None:
    n = args.stream
    mode = f"{args.gap_s:.1f}s gap between" if args.gap_s > 0 else "back-to-back"
    print(f"\n=== serial stream: {n} requests, one at a time ({mode}) ===", flush=True)
    if n > len(images):
        print(
            f"WARNING: {n} requests exceed {len(images)} distinct images; images will repeat.",
            flush=True,
        )
    records: list[JobRecord] = []
    downloads: list[asyncio.Task[bool]] = []
    t0 = time.monotonic()
    for k in range(n):
        if args.gap_s > 0 and k > 0:
            await asyncio.sleep(args.gap_s)
        name, b64 = images[k % len(images)]
        rec = await _run_job(
            http, limiter, creds, region, image_name=name, image_b64=b64, args=args,
            level=1, trial=0, index=k, log=log, download=False,
        )
        records.append(rec)
        if rec.status == "ok" and rec.file_url and not args.no_download:
            ext = (rec.result_type or args.format).lower()
            dest = args.glb_dir / f"{Path(name).stem}.{ext}"
            downloads.append(asyncio.create_task(_safe_download(http, rec.file_url, dest)))
    total_wall = time.monotonic() - t0

    saved = 0
    if downloads:
        print(f"\nawaiting {len(downloads)} background asset download(s)...", flush=True)
        saved = sum(1 for r in await asyncio.gather(*downloads, return_exceptions=True) if r is True)
    summary = _stream_summary(records, total_wall, saved, args.price_per_call)
    await log.write("stream_summary", summary)
    _print_stream_summary(summary, args)


async def _run_sweep(
    creds: _Creds, region: str, args: argparse.Namespace, image_paths: list[Path],
) -> None:
    log = _ResultsLog(args.out)
    limiter = _RateLimiter(args.qps)
    levels: list[int] = args.levels
    summaries: list[dict[str, Any]] = []
    await log.write("run_config", {
        "endpoint": _ENDPOINT,
        "action": _SUBMIT_ACTION,
        "version": _VERSION,
        "region": region,
        "result_format": args.format,
        "pbr": args.pbr,
        "geometry_only": args.geometry_only,
        "levels": levels,
        "trials": args.trials,
        "stream": args.stream,
        "gap_s": args.gap_s,
        "qps": args.qps,
        "poll_interval_s": args.poll_interval,
        "price_per_call": args.price_per_call,
        "ts": time.time(),
    })
    if not args.no_download:
        args.glb_dir.mkdir(parents=True, exist_ok=True)
    print(f"encoding {len(image_paths)} images as base64 (one-time)…", flush=True)
    images = [(p.name, base64.b64encode(p.read_bytes()).decode("ascii")) for p in image_paths]
    total_jobs = args.stream if args.stream > 0 else sum(levels) * args.trials
    if total_jobs > len(images):
        print(
            f"WARNING: {total_jobs} jobs exceed {len(images)} distinct images; images will repeat.",
            flush=True,
        )
    try:
        async with httpx.AsyncClient(follow_redirects=True) as http:
            if args.stream > 0:
                await _run_stream(http, limiter, creds, region, images=images, args=args, log=log)
                return
            cursor = 0
            for level in levels:
                for trial in range(args.trials):
                    summaries.append(
                        await _run_level(
                            http, limiter, creds, region, level=level, trial=trial,
                            images=images, img_offset=cursor, args=args, log=log,
                        )
                    )
                    cursor += level
    finally:
        log.close()

    _print_final(summaries, args)


def _print_final(summaries: list[dict[str, Any]], args: argparse.Namespace) -> None:
    print("\n" + "=" * 104)
    print("SUMMARY  (flight / queue-wait times in seconds)")
    print("=" * 104)
    header = (
        f"{'level':>5} {'trial':>5} {'ok':>4} {'fail':>4} "
        f"{'mean':>7} {'median':>7} {'p95':>7} {'min':>7} {'max':>7} "
        f"{'submit':>7} {'qwait':>7} {'polls':>6} {'thr/min':>8}"
    )
    print(header)
    print("-" * len(header))
    total_cost = 0.0
    for s in summaries:
        total_cost += s["cost_est"]
        print(
            f"{s['level']:>5} {s['trial']:>5} {s['completed']:>4} {s['failed']:>4} "
            f"{_fmt(s['flight_mean']):>7} {_fmt(s['flight_median']):>7} "
            f"{_fmt(s['flight_p95']):>7} {_fmt(s['flight_min']):>7} "
            f"{_fmt(s['flight_max']):>7} {_fmt(s['submit_mean']):>7} "
            f"{_fmt(s['queue_wait_mean']):>7} {_fmt(s['polls_mean']):>6} "
            f"{_fmt(s['throughput_per_min']):>8}"
        )
    print("-" * len(header))
    if args.price_per_call > 0:
        print(f"est total cost: ${total_cost:.3f}  (--price-per-call ${args.price_per_call})")
    print(f"per-job results: {args.out}")

    first_err = next((s for s in summaries if s["failed"] > 0), None)
    if first_err is not None:
        print(
            f"\n>> errors first appeared at level {first_err['level']} "
            f"({first_err['failed']}/{first_err['launched']} failed: {first_err['errors']})."
        )
    else:
        print(
            f"\n>> no failures up to level {max(s['level'] for s in summaries)}."
        )
    print(
        ">> Tencent defaults to 1 concurrent task: rising 'qwait' and flat 'thr/min' "
        "as 'level' grows means generation is serializing server-side, not parallelizing."
    )


def _load_image_paths(image_dir: Path, max_images: int) -> list[Path]:
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() == ".png")
    for p in paths:  # surface unreadable / oversized files now, not mid-paid-run
        size = len(p.read_bytes())
        if size > _MAX_IMAGE_BYTES:
            print(
                f"WARNING: {p.name} is {size / 1e6:.1f}MB > 6MB ImageBase64 cap; "
                "Tencent may reject it.",
                file=sys.stderr,
            )
    return paths[:max_images]


def main() -> None:
    # Windows consoles default to cp1252; force UTF-8 so output never crashes a run.
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Concurrency + flight-time benchmark for Tencent's direct Hunyuan 3D Rapid API.",
    )
    parser.add_argument("--levels", default="1,4,8", help="Comma-separated concurrency levels. Default: 1,4,8")
    parser.add_argument("--trials", type=int, default=1, help="Repeats per level. Default: 1")
    parser.add_argument("--stream", type=int, default=0,
                        help="Serial-stream mode: run N requests one at a time (ignores --levels).")
    parser.add_argument("--gap-s", type=float, default=0.0,
                        help="Sleep between serial-stream requests. Default: 0 (back-to-back).")
    parser.add_argument("--format", default="GLB", choices=_RESULT_FORMATS,
                        help="ResultFormat. Default: GLB")
    parser.add_argument("--pbr", action="store_true", help="EnablePBR (forbidden with --geometry-only).")
    parser.add_argument("--geometry-only", action="store_true",
                        help="EnableGeometry: texture-free white model (no OBJ; mutually exclusive with --pbr).")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--max-images", type=int, default=50, help="Max distinct images to use. Default: 50")
    parser.add_argument("--qps", type=float, default=18.0,
                        help="Client-side QPS cap on signed calls (Tencent limit is 20). Default: 18")
    parser.add_argument("--poll-interval", type=float, default=2.0,
                        help="Seconds between QueryHunyuanTo3DRapidJob polls. Default: 2.0")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-API-call HTTP timeout (s). Default: 60")
    parser.add_argument("--job-timeout", type=float, default=600.0,
                        help="Per-job poll deadline from submit (s). Default: 600")
    parser.add_argument("--out", type=Path, default=Path("bench_hunyuan_tencent_results.jsonl"))
    parser.add_argument("--glb-dir", type=Path, default=Path("bench_hunyuan_tencent_glbs"),
                        help="Directory for downloaded assets. Default: bench_hunyuan_tencent_glbs/")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading the generated assets.")
    parser.add_argument("--price-per-call", type=float, default=0.0,
                        help="Optional per-call price for a spend estimate + --max-cost gate. Default: 0 (unmetered).")
    parser.add_argument("--max-cost", type=float, default=5.0,
                        help="Abort if estimate exceeds this (only enforced when --price-per-call > 0). Default: $5")
    parser.add_argument("--secret-id", default=None, help="Tencent SecretId (else TENCENTCLOUD_SECRET_ID).")
    parser.add_argument("--secret-key", default=None, help="Tencent SecretKey (else TENCENTCLOUD_SECRET_KEY).")
    parser.add_argument("--region", default=None, help="Tencent region (else TENCENTCLOUD_REGION, default ap-singapore).")
    parser.add_argument("--yes", action="store_true", help="Actually execute (and spend). Default: dry run.")
    args = parser.parse_args()

    args.levels = [int(x) for x in args.levels.split(",") if x.strip()]
    if not args.levels or any(level <= 0 for level in args.levels):
        parser.error("--levels must be positive integers, e.g. 1,4,8")
    if args.geometry_only and args.pbr:
        parser.error("--pbr cannot be used with --geometry-only (texture-free white model)")
    if args.geometry_only and args.format == "OBJ":
        parser.error("--geometry-only does not support OBJ; pick another --format (default GLB)")
    if args.stream < 0:
        parser.error("--stream must be >= 0")
    if args.qps <= 0 or args.qps > 20:
        parser.error("--qps must be in (0, 20] (Tencent caps this API at 20 QPS)")

    load_dotenv(SERVER_DIR / ".env")
    secret_id = args.secret_id or os.environ.get("TENCENTCLOUD_SECRET_ID")
    secret_key = args.secret_key or os.environ.get("TENCENTCLOUD_SECRET_KEY")
    region = args.region or os.environ.get("TENCENTCLOUD_REGION") or "ap-singapore"

    if not args.image_dir.is_dir():
        parser.error(f"image dir not found: {args.image_dir}")
    image_paths = _load_image_paths(args.image_dir, args.max_images)
    if not image_paths:
        parser.error(f"no .png images in {args.image_dir}")

    total_jobs = args.stream if args.stream > 0 else sum(args.levels) * args.trials
    est_cost = round(args.price_per_call * total_jobs, 2)

    print("Tencent Hunyuan 3D Rapid (direct) benchmark plan")
    print(f"  endpoint:       {_ENDPOINT}  ({_SUBMIT_ACTION}, v{_VERSION})")
    print(f"  region:         {region}")
    print(f"  settings:       format={args.format} pbr={args.pbr} geometryOnly={args.geometry_only}")
    if args.stream > 0:
        print(f"  mode:           serial stream of {args.stream} requests (1 at a time)")
    else:
        print(f"  levels:         {args.levels}  x {args.trials} trial(s)")
    print(f"  images:         {len(image_paths)} distinct from {args.image_dir}")
    print(f"  total jobs:     {total_jobs}")
    print(f"  qps cap:        {args.qps}  (Tencent: 20 QPS, 1 concurrent generation task)")
    print(f"  poll interval:  {args.poll_interval}s")
    if args.price_per_call > 0:
        print(f"  per-call price: ${args.price_per_call:.3f}")
        print(f"  EST. MAX SPEND: ${est_cost:.2f}")
    else:
        print("  per-call price: unmetered (Tencent bills directly; pass --price-per-call to estimate)")
    print(f"  results -> {args.out}")
    print(f"  assets ->  {'(download disabled)' if args.no_download else args.glb_dir}")

    if not args.yes:
        print("\n[DRY RUN] no API calls made. Re-run with --yes to execute.")
        return
    if args.price_per_call > 0 and est_cost > args.max_cost:
        print(
            f"\nABORT: estimate ${est_cost:.2f} exceeds --max-cost ${args.max_cost:.2f}.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not secret_id or not secret_key:
        print(
            "\nERROR: missing credentials. Pass --secret-id/--secret-key or set "
            "TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nexecuting{f' (est ~${est_cost:.2f})' if args.price_per_call > 0 else ''}…")
    asyncio.run(_run_sweep(_Creds(secret_id, secret_key), region, args, image_paths))


if __name__ == "__main__":
    main()
