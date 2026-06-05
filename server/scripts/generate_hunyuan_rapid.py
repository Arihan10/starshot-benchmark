"""TEMPORARY — benchmark Tencent Hunyuan (3D AI Studio, *Rapid* edition) against
the in-house Trellis pipeline.

For every object in a run's ``test-images/`` directory (each object is
identified by its ``<name>.glb``), this submits that object's Nano-Banana
reference image to the 3D AI Studio Rapid image-to-3D endpoint (PBR on, no
text prompt), polls until the job finishes, downloads the returned OBJ archive
(a zip), converts it to a single textured ``.glb``, and writes ``<name>.glb``
into the output directory.

Why the reference image and not a text prompt: the requirement is "no prompt",
so this is pure image-to-3D. The faithful benchmark input is the exact
Nano-Banana studio shot that Trellis consumed — it lives next to the generated
meshes (``generated/<v>/objects-generated/<name>.png``).

Rate limiting: 10 API keys, each capped by the provider at 3 requests/min. We
round-robin the keys one-by-one and pace every api.3daistudio.com call (submit
+ status poll) so combined throughput holds at 30 requests/min and no single
key exceeds 3/min. Archive downloads hit storage.3daistudio.com (a different
host) and are not rate limited.

Resumable: an object whose output ``.glb`` already exists is skipped, so a
re-run after an interruption never re-bills a finished object.

Provide the keys with ``--keys-file <path>`` (one per line, or comma/space
separated) or the ``AISTUDIO_API_KEYS`` env var (same separators).

Usage (from server/):
    # validate selection + rate plan without spending credits
    uv run python scripts/generate_hunyuan_rapid.py --dry-run

    # smoke-test on the first object only
    uv run python scripts/generate_hunyuan_rapid.py --keys-file keys.txt --limit 1

    # generate everything
    AISTUDIO_API_KEYS="k1,...,k10" uv run python scripts/generate_hunyuan_rapid.py
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import os
import re
import sys
import tempfile
import time
import zipfile
from collections import deque
from pathlib import Path

import httpx
import trimesh

# --- endpoints ---------------------------------------------------------------

BASE_URL = "https://api.3daistudio.com"
RAPID_PATH = "/v1/3d-models/tencent/generate/rapid/"
STATUS_PATH = "/v1/generation-request/{task_id}/status/"

# --- defaults (this benchmark targets one specific run) ----------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO_ROOT / "runs" / "god run" / "modern-house" / "opus-new"
DEFAULT_OBJECTS_DIR = DEFAULT_RUN / "test-images"
DEFAULT_IMAGES_DIR = DEFAULT_RUN / "generated" / "2" / "objects-generated"
DEFAULT_OUT_DIR = DEFAULT_RUN / "hunyuan-rapid"

# --- rate limit --------------------------------------------------------------

PER_KEY_RPM = 3
TARGET_RPM = 30
RATE_WINDOW_S = 60.0

# --- timeouts / retries ------------------------------------------------------

SUBMIT_TIMEOUT_S = 120.0
POLL_HTTP_TIMEOUT_S = 60.0
DOWNLOAD_TIMEOUT_S = 300.0
JOB_TIMEOUT_S = 900.0          # give up on a single job after 15 min
INITIAL_WAIT_S = 60.0          # Rapid takes ~2-3 min; don't poll before this
POLL_INTERVAL_S = 10.0         # per-job desired cadence (global limiter still applies)
MAX_API_ATTEMPTS = 5           # transient-error attempts per submit/poll call
FALLBACK_429_DELAY_S = 30.0
RETRY_BACKOFF_BASE_S = 3.0
RETRY_BACKOFF_MAX_S = 45.0

# Serialize the RAM-heavy trimesh decode/export so concurrent conversions
# don't stack Pillow-decoded texture buffers and trip the OOM killer.
_CONVERT_FANOUT = 3


class InsufficientCreditsError(Exception):
    """Provider returned 402 — the run can't continue, abort everything."""


class RateLimiter:
    """Round-robin key picker that enforces both a per-key sliding window
    (PER_KEY_RPM per RATE_WINDOW_S) and a global minimum spacing derived from
    the target throughput. ``acquire`` blocks until a key is free, records the
    use, and returns it. With 10 keys this yields exactly 30/min total and
    3/min per key, cycling 1-by-1."""

    def __init__(self, keys: list[str], *, per_key_rpm: int, target_rpm: int) -> None:
        self._keys = list(keys)
        effective_rpm = min(target_rpm, len(self._keys) * per_key_rpm)
        self._min_gap_s = RATE_WINDOW_S / effective_rpm if effective_rpm else 0.0
        self._per_key = per_key_rpm
        self._hist: dict[int, deque[float]] = {i: deque() for i in range(len(self._keys))}
        self._rr = 0
        self._last_global = 0.0
        self._lock = asyncio.Lock()

    @property
    def effective_rpm(self) -> float:
        return RATE_WINDOW_S / self._min_gap_s if self._min_gap_s else 0.0

    async def acquire(self) -> str:
        while True:
            async with self._lock:
                now = time.monotonic()
                gap_wait = self._last_global + self._min_gap_s - now
                if gap_wait <= 0:
                    n = len(self._keys)
                    for offset in range(n):
                        idx = (self._rr + offset) % n
                        hist = self._hist[idx]
                        while hist and now - hist[0] >= RATE_WINDOW_S:
                            hist.popleft()
                        if len(hist) < self._per_key:
                            hist.append(now)
                            self._rr = idx + 1
                            self._last_global = now
                            return self._keys[idx]
                    # Every key is saturated; wait until the oldest use ages out.
                    soonest = min(self._hist[i][0] for i in range(n)) + RATE_WINDOW_S - now
                    wait = max(soonest, 0.05)
                else:
                    wait = gap_wait
            await asyncio.sleep(wait)


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _backoff(attempt: int) -> float:
    return min(RETRY_BACKOFF_BASE_S * (2**attempt), RETRY_BACKOFF_MAX_S)


def _retry_after(resp: httpx.Response) -> float:
    raw = resp.headers.get("Retry-After")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return FALLBACK_429_DELAY_S


async def _submit(
    client: httpx.AsyncClient, limiter: RateLimiter, data_uri: str, *, enable_pbr: bool
) -> str:
    body = {"image": data_uri, "enable_pbr": enable_pbr}
    last_err: str = ""
    for attempt in range(MAX_API_ATTEMPTS):
        key = await limiter.acquire()
        try:
            resp = await client.post(
                f"{BASE_URL}{RAPID_PATH}", json=body, headers=_auth(key),
                timeout=SUBMIT_TIMEOUT_S,
            )
        except httpx.HTTPError as e:
            last_err = f"{type(e).__name__}: {str(e)[:160]}"
            await asyncio.sleep(_backoff(attempt))
            continue
        if resp.status_code == 402:
            raise InsufficientCreditsError(resp.text[:200])
        if resp.status_code == 429:
            await asyncio.sleep(_retry_after(resp))
            continue
        if resp.status_code >= 400:
            # 400 validation_failed and other 4xx are deterministic for this input.
            raise RuntimeError(f"submit {resp.status_code}: {resp.text[:200]}")
        task_id = resp.json().get("task_id")
        if not task_id:
            raise RuntimeError(f"submit returned no task_id: {resp.text[:200]}")
        return str(task_id)
    raise RuntimeError(f"submit failed after {MAX_API_ATTEMPTS} attempts: {last_err}")


def _pick_archive_url(results: list[dict]) -> str:
    for r in results:
        if r.get("asset_type") == "ARCHIVE" and r.get("asset"):
            return str(r["asset"])
    for r in results:
        if r.get("asset"):
            return str(r["asset"])
    raise RuntimeError(f"FINISHED but no downloadable asset in results: {results!r}")


async def _poll(client: httpx.AsyncClient, limiter: RateLimiter, task_id: str) -> str:
    url = f"{BASE_URL}{STATUS_PATH.format(task_id=task_id)}"
    deadline = time.monotonic() + JOB_TIMEOUT_S
    await asyncio.sleep(INITIAL_WAIT_S)
    transient = 0
    while time.monotonic() < deadline:
        key = await limiter.acquire()
        try:
            resp = await client.get(url, headers=_auth(key), timeout=POLL_HTTP_TIMEOUT_S)
        except httpx.HTTPError:
            transient += 1
            if transient >= MAX_API_ATTEMPTS:
                raise
            await asyncio.sleep(_backoff(transient))
            continue
        if resp.status_code == 429:
            await asyncio.sleep(_retry_after(resp))
            continue
        if resp.status_code == 404:
            raise RuntimeError(f"status 404 (task {task_id} expired or unknown)")
        resp.raise_for_status()
        transient = 0
        body = resp.json()
        status = str(body.get("status", "")).upper()
        if status == "FINISHED":
            return _pick_archive_url(body.get("results") or [])
        if body.get("failure_reason") or status in {"FAILED", "ERROR", "CANCELLED"}:
            raise RuntimeError(f"job {status or 'FAILED'}: {body.get('failure_reason')!r}")
        await asyncio.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"job {task_id} did not finish within {JOB_TIMEOUT_S:.0f}s")


async def _download(client: httpx.AsyncClient, url: str) -> bytes:
    last_err = ""
    for attempt in range(MAX_API_ATTEMPTS):
        try:
            resp = await client.get(url, timeout=DOWNLOAD_TIMEOUT_S)
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPError as e:
            last_err = f"{type(e).__name__}: {str(e)[:160]}"
            await asyncio.sleep(_backoff(attempt))
    raise RuntimeError(f"download failed after {MAX_API_ATTEMPTS} attempts: {last_err}")


def _convert_archive_to_glb(zip_bytes: bytes, out_path: Path) -> None:
    """Extract the Rapid OBJ archive and re-export it as one textured GLB.

    The zip holds an OBJ plus its MTL and texture images; extracting the whole
    archive preserves the relative paths the MTL references, so trimesh resolves
    the textures and embeds them into the GLB binary chunk on export."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(tmp_dir)
        objs = sorted(tmp_dir.rglob("*.obj"))
        if not objs:
            names = [p.name for p in tmp_dir.rglob("*") if p.is_file()]
            raise RuntimeError(f"no .obj in archive (contents: {names})")
        scene = trimesh.load(objs[0], process=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scene.export(out_path, file_type="glb")
    data = out_path.read_bytes()
    if not data.startswith(b"glTF"):
        out_path.unlink(missing_ok=True)
        raise RuntimeError("converted file is not a valid GLB (missing glTF magic)")


async def _process(
    name: str,
    image_path: Path,
    out_path: Path,
    *,
    client: httpx.AsyncClient,
    limiter: RateLimiter,
    convert_sem: asyncio.Semaphore,
    abort: asyncio.Event,
    enable_pbr: bool,
) -> tuple[str, str]:
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] {name} (output exists)", flush=True)
        return (name, "skipped")
    if abort.is_set():
        return (name, "aborted")
    t0 = time.monotonic()
    try:
        img_bytes = image_path.read_bytes()
        data_uri = "data:image/png;base64," + base64.b64encode(img_bytes).decode("ascii")
        print(f"[submit] {name}", flush=True)
        task_id = await _submit(client, limiter, data_uri, enable_pbr=enable_pbr)
        print(f"[poll] {name} task={task_id}", flush=True)
        asset_url = await _poll(client, limiter, task_id)
        zip_bytes = await _download(client, asset_url)
        async with convert_sem:
            await asyncio.to_thread(_convert_archive_to_glb, zip_bytes, out_path)
        print(f"[done] {name} -> {out_path.name} ({time.monotonic() - t0:.0f}s)", flush=True)
        return (name, "done")
    except InsufficientCreditsError as e:
        abort.set()
        print(f"[ABORT] {name}: insufficient credits ({e})", file=sys.stderr, flush=True)
        return (name, "insufficient_credits")
    except Exception as e:  # noqa: BLE001 — one object's failure must not sink the batch
        print(f"[error] {name}: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr, flush=True)
        return (name, f"error: {type(e).__name__}")


def _load_keys(keys_file: str | None) -> list[str]:
    if keys_file:
        raw = Path(keys_file).read_text(encoding="utf-8")
    else:
        raw = os.environ.get("AISTUDIO_API_KEYS", "")
    return [k.strip() for k in re.split(r"[\s,]+", raw) if k.strip()]


def _resolve_jobs(
    objects_dir: Path, images_dir: Path, out_dir: Path, limit: int | None
) -> tuple[list[tuple[str, Path, Path]], list[str]]:
    """Returns (jobs, missing) where jobs are (name, image_path, out_path) for
    every test-images object that has a reference PNG, and missing lists the
    object names with no reference image."""
    names = sorted(p.stem for p in objects_dir.glob("*.glb"))
    if limit is not None:
        names = names[:limit]
    jobs: list[tuple[str, Path, Path]] = []
    missing: list[str] = []
    for name in names:
        img = images_dir / f"{name}.png"
        if img.exists():
            jobs.append((name, img, out_dir / f"{name}.glb"))
        else:
            missing.append(name)
    return jobs, missing


async def _run(jobs: list[tuple[str, Path, Path]], keys: list[str], enable_pbr: bool) -> int:
    limiter = RateLimiter(keys, per_key_rpm=PER_KEY_RPM, target_rpm=TARGET_RPM)
    convert_sem = asyncio.Semaphore(_CONVERT_FANOUT)
    abort = asyncio.Event()
    async with httpx.AsyncClient(follow_redirects=True) as client:
        results = await asyncio.gather(
            *(
                _process(
                    name, img, out,
                    client=client, limiter=limiter, convert_sem=convert_sem,
                    abort=abort, enable_pbr=enable_pbr,
                )
                for name, img, out in jobs
            )
        )
    summary: dict[str, int] = {}
    for _, status in results:
        bucket = status.split(":")[0]
        summary[bucket] = summary.get(bucket, 0) + 1
    print("\n=== summary ===")
    for bucket in sorted(summary):
        print(f"  {bucket}: {summary[bucket]}")
    failed = sum(c for b, c in summary.items() if b not in {"done", "skipped"})
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-dir", type=Path, default=DEFAULT_OBJECTS_DIR,
                        help="dir whose *.glb names define the objects to generate")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR,
                        help="dir holding each object's <name>.png reference image")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="where to write <name>.glb (defaults to a sibling of test-images)")
    parser.add_argument("--keys-file", default=None,
                        help="file of API keys (newline/comma/space separated); "
                             "else read AISTUDIO_API_KEYS")
    parser.add_argument("--no-pbr", action="store_true", help="disable PBR (default: enabled)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N objects")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan (objects, images, rate) without calling the API")
    args = parser.parse_args()

    if not args.objects_dir.is_dir():
        print(f"objects dir not found: {args.objects_dir}", file=sys.stderr)
        return 2
    if not args.images_dir.is_dir():
        print(f"images dir not found: {args.images_dir}", file=sys.stderr)
        return 2

    jobs, missing = _resolve_jobs(args.objects_dir, args.images_dir, args.out_dir, args.limit)
    keys = _load_keys(args.keys_file)

    print(f"objects dir: {args.objects_dir}")
    print(f"images dir:  {args.images_dir}")
    print(f"out dir:     {args.out_dir}")
    print(f"objects:     {len(jobs)} with a reference image"
          + (f", {len(missing)} missing image" if missing else ""))
    print(f"keys loaded: {len(keys)}")
    print(f"PBR:         {'disabled' if args.no_pbr else 'enabled'}")
    if missing:
        print(f"  missing images: {', '.join(missing)}", file=sys.stderr)

    if args.dry_run:
        rpm = RateLimiter(keys or ["x"], per_key_rpm=PER_KEY_RPM, target_rpm=TARGET_RPM).effective_rpm
        print(f"rate plan:   {rpm:.0f} req/min total, {PER_KEY_RPM}/min per key")
        already = sum(1 for _, _, out in jobs if out.exists() and out.stat().st_size > 0)
        print(f"resumable:   {already} already done (would be skipped)")
        print("\n(dry run — no API calls made)")
        return 0

    if not jobs:
        print("nothing to do", file=sys.stderr)
        return 2
    if not keys:
        print("no API keys (set AISTUDIO_API_KEYS or pass --keys-file)", file=sys.stderr)
        return 2
    if len(keys) != 10:
        print(f"warning: expected 10 keys, got {len(keys)} "
              f"(throughput will be ~{min(TARGET_RPM, len(keys) * PER_KEY_RPM)}/min)",
              file=sys.stderr)

    return asyncio.run(_run(jobs, keys, enable_pbr=not args.no_pbr))


if __name__ == "__main__":
    sys.exit(main())
