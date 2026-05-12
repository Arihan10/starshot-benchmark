#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Boot the API server and the Node-hosted three.js viewer.

The browser (not this script) POSTs /generate and subscribes to the SSE
stream. This script just keeps the two processes alive until Ctrl-C.

Usage: ./scripts/run_request.py
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = REPO_ROOT / "server"
CLIENT_DIR = REPO_ROOT / "client"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765
CLIENT_PORT = 8766
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


def _child_env() -> dict[str, str]:
    """Clean env for spawned subprocesses.

    `uv run --script` (how `enx test` invokes this file) sets VIRTUAL_ENV to
    its ephemeral env. Leaking that into `uv run uvicorn` in server/ makes
    the inner uv warn about a mismatched env and can mask failures. Strip
    the handful of uv-managed vars so each child resolves its own project
    environment from scratch.
    """
    env = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}}
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


def _wait_for_port(
    host: str, port: int, *, proc: subprocess.Popen[bytes], timeout: float
) -> bool:
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
    if not (CLIENT_DIR / "node_modules" / "three").exists():
        print(
            "[run_request] client/node_modules/three missing — run `npm install` in client/ first",
            file=sys.stderr,
        )
        return 1

    env = _child_env()

    print(f"[run_request] starting API server on {SERVER_URL}", flush=True)
    server = subprocess.Popen(
        [
            "uv", "run", "uvicorn", "app.main:app",
            "--host", SERVER_HOST, "--port", str(SERVER_PORT),
            "--log-level", "info",
            "--reload",
        ],
        cwd=SERVER_DIR,
        env=env,
        process_group=0,
    )

    if not _wait_for_port(SERVER_HOST, SERVER_PORT, proc=server, timeout=30.0):
        print(
            f"[run_request] server never became reachable at {SERVER_URL} — aborting",
            file=sys.stderr,
        )
        _shutdown(server)
        return 1
    print(f"[run_request] server ready, launching viewer", flush=True)

    client = subprocess.Popen(
        ["node", "server.mjs"],
        cwd=CLIENT_DIR,
        env={**env, "SERVER_URL": SERVER_URL, "PORT": str(CLIENT_PORT)},
        process_group=0,
    )
    try:
        while True:
            if server.poll() is not None:
                return server.returncode or 0
            if client.poll() is not None:
                return client.returncode or 0
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        _shutdown(client)
        _shutdown(server)


if __name__ == "__main__":
    sys.exit(main())
