"""Smoke-test the standalone Nano Banana client.

Prints model + settings, then times the call(s) so latency is visible
at a glance. Pass --n to repeat and see variance.

Usage (from server/):
    uv run python scripts/test_nano_banana.py "a cozy attic library at dusk"
    uv run python scripts/test_nano_banana.py "..." --out out.png
    uv run python scripts/test_nano_banana.py "..." --n 3
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from app.services import nano_banana


def _print_settings() -> None:
    print("settings:")
    print(f"  model:        {nano_banana.MODEL}")
    print(f"  aspect_ratio: {nano_banana.ASPECT_RATIO}")
    print(f"  timeout_ms:   {nano_banana.TIMEOUT_MS}")
    print(f"  scope:        {nano_banana._RESUMABLE_SCOPE}")
    # Pro vs. Flash + presence of a thinking knob is the dominant
    # latency driver, so call it out if it's not configured.
    cfg = nano_banana._build_config()
    print(f"  config:       {cfg!r}")
    print()


async def _run(prompt: str, out: Path, n: int) -> None:
    _print_settings()
    print(f"prompt: {prompt!r}")
    print()

    timings: list[float] = []
    last_result: nano_banana.NanoBananaResult | None = None
    for i in range(1, n + 1):
        label = f"[{i}/{n}]" if n > 1 else ""
        print(f"{label} calling… ", end="", flush=True)
        t0 = time.monotonic()
        try:
            result = await nano_banana.generate(prompt)
            dt = time.monotonic() - t0
            print(f"done in {dt:.2f}s  ({len(result.image_bytes):,} bytes, {result.mime_type})")
            timings.append(dt)
            last_result = result
        except Exception as e:  # noqa: BLE001
            dt = time.monotonic() - t0
            print(f"FAIL after {dt:.2f}s — {type(e).__name__}: {e}")

    if last_result is None:
        print("\nno successful calls")
        sys.exit(1)

    saved = nano_banana.save(last_result, out)
    print(f"\nsaved: {saved} ({len(last_result.image_bytes):,} bytes, {last_result.mime_type})")

    print("\ntimings (s):")
    for i, t in enumerate(timings, 1):
        print(f"  call {i}: {t:.2f}")
    if len(timings) > 1:
        avg = sum(timings) / len(timings)
        print(f"  avg:    {avg:.2f}")
        print(f"  min:    {min(timings):.2f}")
        print(f"  max:    {max(timings):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Nano Banana client.")
    parser.add_argument("prompt", help="Text prompt")
    parser.add_argument(
        "--out", type=Path, default=Path("nano_banana_out.png"),
        help="Where to save the generated image (final iteration only).",
    )
    parser.add_argument(
        "--n", type=int, default=1,
        help="Number of calls to make. Default: 1.",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.prompt, args.out, args.n))


if __name__ == "__main__":
    main()
