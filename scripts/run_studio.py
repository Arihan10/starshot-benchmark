#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Boot the phrase studio: noun phrase -> Nano Banana -> Trellis -> 3D.

A standalone single-page client for exercising the real production asset
pipeline one object at a time. Pick a noun phrase from
`object_noun_phrases.json`, wrap it in the production `wrap_image_prompt`,
generate an image with Nano Banana, then turn that image into a textured
GLB with Trellis and inspect it in a three.js viewer.

Needs `server/.env` populated with `GOOGLE_API_KEY` (Nano Banana) and, if
your Trellis endpoint differs from the default, `TRELLIS_BASE_URL`.

Usage: ./scripts/run_studio.py   (or: python scripts/run_studio.py)
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = REPO_ROOT / "server"

HOST = "127.0.0.1"
PORT = int(os.environ.get("STUDIO_PORT", "8770"))


def _child_env() -> dict[str, str]:
    # `uv run --script` exports VIRTUAL_ENV / UV_PROJECT_ENVIRONMENT for its
    # own ephemeral env; leaking them into the inner `uv run` makes it warn
    # about a mismatched env. Strip them so the child resolves server/'s env.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
    }
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _wait_for_port(host: str, port: int, *, proc: subprocess.Popen[bytes], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def _shutdown(proc: subprocess.Popen[bytes]) -> None:
    """Cross-platform best-effort teardown (terminate, then kill)."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def main() -> int:
    if "GOOGLE_API_KEY" not in os.environ and not (SERVER_DIR / ".env").exists():
        print(
            "[studio] GOOGLE_API_KEY not set and server/.env missing — "
            "Nano Banana calls will fail with an auth error",
            file=sys.stderr,
        )

    url = f"http://{HOST}:{PORT}/"
    print(f"[studio] starting at {url}", flush=True)
    proc = subprocess.Popen(
        [
            "uv", "run", "uvicorn", "app.studio:app",
            "--host", HOST, "--port", str(PORT),
            "--log-level", "info",
        ],
        cwd=SERVER_DIR,
        env=_child_env(),
    )
    if not _wait_for_port(HOST, PORT, proc=proc, timeout=45.0):
        print(f"[studio] server never became reachable at {url}", file=sys.stderr)
        _shutdown(proc)
        return 1
    print(f"[studio] ready at {url}", flush=True)
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass
    try:
        return proc.wait() or 0
    except KeyboardInterrupt:
        return 0
    finally:
        _shutdown(proc)


if __name__ == "__main__":
    sys.exit(main())
