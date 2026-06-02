#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Boot the prompt → Nano Banana → Trellis 2 dev playground.

Single-page UI for prompt experimentation: type a prompt, generate an
image with Nano Banana Pro, then forward it to Trellis 2 (auto or
manual) and inspect the GLB in a three.js viewer.

Usage: ./scripts/run_playground.py
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = REPO_ROOT / "server"

HOST = "127.0.0.1"
PORT = 8767


def _child_env() -> dict[str, str]:
    env = {
        k: v for k, v in os.environ.items()
        if k not in {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
    }
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _shutdown(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    _signal_group(pgid, signal.SIGINT)
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_group(pgid, signal.SIGKILL)
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


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


def main() -> int:
    if "FAL_KEY" not in os.environ and not (SERVER_DIR / ".env").exists():
        print(
            "[playground] FAL_KEY not set and server/.env missing — "
            "fal calls will fail with an auth error",
            file=sys.stderr,
        )

    url = f"http://{HOST}:{PORT}/"
    print(f"[playground] starting at {url}", flush=True)
    proc = subprocess.Popen(
        [
            "uv", "run", "uvicorn", "app.playground:app",
            "--host", HOST, "--port", str(PORT),
            "--log-level", "info",
        ],
        cwd=SERVER_DIR,
        env=_child_env(),
        process_group=0,
    )
    if not _wait_for_port(HOST, PORT, proc=proc, timeout=30.0):
        print(f"[playground] server never became reachable at {url}", file=sys.stderr)
        _shutdown(proc)
        return 1
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
