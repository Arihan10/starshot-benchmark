"""Smoke-test the new hosted Trellis 2 HTTP API end-to-end.

Spawns a job, polls /jobs/{id} until done, downloads the GLB, and
writes it to disk. Uses only the standard library + httpx so it can
run with `uv run --script` against a stale checkout.

Usage (from server/):
    uv run python scripts/test_trellis.py                  # uses default image
    uv run python scripts/test_trellis.py path/to/img.png
    uv run python scripts/test_trellis.py img.png --resolution 1024 --out my.glb
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx

BASE_URL = os.environ.get(
    "TRELLIS_BASE_URL",
    "https://starshot-aitools--starshot-assets-router-fastapi-app.modal.run",
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starshot_paths import runs_root  # noqa: E402

DEFAULT_IMAGE = runs_root() / "modern-house" / "objects" / "pavilion_roof.png"

POLL_INTERVAL_SECONDS = 2.0
POLL_TIMEOUT_SECONDS = 600.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image", nargs="?", type=Path, default=DEFAULT_IMAGE,
        help=f"input image. Default: {DEFAULT_IMAGE}",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("test_trellis_out.glb"),
        help="where to save the GLB. Default: ./test_trellis_out.glb",
    )
    parser.add_argument(
        "--resolution", default="512", choices=["512", "1024", "1536"],
    )
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--decimation-target", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.image.exists():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 2

    image_bytes = args.image.read_bytes()
    print(f"base url:   {BASE_URL}")
    print(f"image:      {args.image}  ({len(image_bytes):,} bytes)")
    print(f"resolution: {args.resolution}")
    print(f"texture:    {args.texture_size}")
    print(f"decimation: {args.decimation_target}")
    print(f"seed:       {args.seed}")
    print()

    with httpx.Client(follow_redirects=True, timeout=60.0) as http:
        # 0. cheap health probe so we can fail fast on a bad URL
        try:
            health = http.get(f"{BASE_URL}/health", timeout=10.0)
            health.raise_for_status()
            print(f"[health] {health.json()}")
        except Exception as e:
            print(f"[health] WARN: {type(e).__name__}: {e}", file=sys.stderr)

        # 1. POST /generate
        t0 = time.monotonic()
        print("\n[1/3] POST /generate …", flush=True)
        spawn = http.post(
            f"{BASE_URL}/generate",
            files={"image": (args.image.name, image_bytes, "image/png")},
            data={
                "seed": str(args.seed),
                "resolution": args.resolution,
                "texture_size": str(args.texture_size),
                "decimation_target": str(args.decimation_target),
            },
            timeout=60.0,
        )
        ctype = spawn.headers.get("content-type", "")
        if spawn.status_code != 200:
            preview = spawn.content[:300]
            print(
                f"  FAIL {spawn.status_code} (ct={ctype}, len={len(spawn.content)}): {preview!r}",
                file=sys.stderr,
            )
            return 1
        try:
            body = spawn.json()
        except Exception as e:
            preview = spawn.content[:300]
            print(
                f"  FAIL: 200 OK but body not JSON ({type(e).__name__}: {e}); "
                f"ct={ctype}, len={len(spawn.content)}, first 300 bytes: {preview!r}",
                file=sys.stderr,
            )
            return 1
        job_id = body.get("job_id")
        if not job_id:
            print(f"  FAIL: no job_id in response: {body!r}", file=sys.stderr)
            return 1
        print(f"  job_id: {job_id}  ({(time.monotonic() - t0):.2f}s)")

        # 2. Poll /jobs/{id}
        print(f"\n[2/3] poll /jobs/{job_id} every {POLL_INTERVAL_SECONDS}s …", flush=True)
        deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
        polls = 0
        while True:
            if time.monotonic() >= deadline:
                print(f"  FAIL: timed out after {POLL_TIMEOUT_SECONDS:.0f}s", file=sys.stderr)
                return 1
            time.sleep(POLL_INTERVAL_SECONDS)
            polls += 1
            r = http.get(f"{BASE_URL}/jobs/{job_id}", timeout=30.0)
            r.raise_for_status()
            status = r.json()
            s = status.get("status")
            elapsed = time.monotonic() - t0
            if s == "done":
                size = status.get("size_bytes")
                print(f"  done after {polls} polls ({elapsed:.1f}s, size={size})")
                break
            if s == "failed":
                err = status.get("error", {})
                print(f"  FAIL worker: {err}", file=sys.stderr)
                return 1
            if polls % 5 == 0:
                print(f"  …still pending ({polls} polls, {elapsed:.1f}s)", flush=True)

        # 3. GET /jobs/{id}/result
        print(f"\n[3/3] GET /jobs/{job_id}/result …", flush=True)
        result = http.get(f"{BASE_URL}/jobs/{job_id}/result", timeout=180.0)
        if result.status_code != 200:
            print(f"  FAIL {result.status_code}: {result.text}", file=sys.stderr)
            return 1
        content = result.content
        ctype = result.headers.get("content-type", "")
        print(f"  got {len(content):,} bytes  (content-type: {ctype})")

        # Sanity check: GLB binary begins with magic "glTF"
        if not content.startswith(b"glTF"):
            print(f"  WARN: payload does not start with 'glTF' magic — first 8 bytes = {content[:8]!r}", file=sys.stderr)
        else:
            print("  glTF magic OK")

        args.out.write_bytes(content)
        print(f"\nwrote {args.out.resolve()}  ({len(content):,} bytes, total {(time.monotonic() - t0):.1f}s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
