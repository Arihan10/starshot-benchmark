#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Boot the API server in bbox-only mode plus the three.js viewer.

Same as run_request.py, but the server skips Trellis 2 + Nano Banana — the
pipeline decomposes the scene end-to-end and the client shows every node
as a wireframe bbox. Tree navigation and bbox selection work as normal.

Usage: ./scripts/run_bboxes_only.py
"""

from __future__ import annotations

import argparse
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


def _child_env() -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
    }
    return env


def _pick_free_port() -> int:
    """Ask the OS for a free TCP port. Race-prone (another process could
    grab it before uvicorn binds), but good enough for local multi-run."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((SERVER_HOST, 0))
        return s.getsockname()[1]


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
    parser = argparse.ArgumentParser(description="Boot the bbox-only API server + viewer.")
    parser.add_argument(
        "--runs-dir",
        "--runs",
        dest="runs_dir",
        type=Path,
        default=SERVER_DIR / "runs",
        help="Directory the server writes per-slot run artifacts to. Created if missing.",
    )
    args = parser.parse_args()

    if not (CLIENT_DIR / "node_modules" / "three").exists():
        print(
            "[run_bboxes_only] client/node_modules/three missing — run `npm install` in client/ first",
            file=sys.stderr,
        )
        return 1

    runs_dir = args.runs_dir.resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    server_port = _pick_free_port()
    client_port = _pick_free_port()
    server_url = f"http://{SERVER_HOST}:{server_port}"
    client_url = f"http://{SERVER_HOST}:{client_port}"

    env = _child_env()

    print(
        f"[run_bboxes_only] starting API server on {server_url} (bbox-only mode, runs={runs_dir})",
        flush=True,
    )
    server = subprocess.Popen(
        [
            "uv", "run", "uvicorn", "app.main_nomesh:app",
            "--host", SERVER_HOST, "--port", str(server_port),
            "--log-level", "info",
        ],
        cwd=SERVER_DIR,
        env={**env, "STARSHOT_RUNS_DIR": str(runs_dir)},
        process_group=0,
    )

    if not _wait_for_port(SERVER_HOST, server_port, proc=server, timeout=30.0):
        print(
            f"[run_bboxes_only] server never became reachable at {server_url} — aborting",
            file=sys.stderr,
        )
        _shutdown(server)
        return 1
    print(f"[run_bboxes_only] server ready, launching viewer at {client_url}", flush=True)

    client = subprocess.Popen(
        ["node", "server.mjs"],
        cwd=CLIENT_DIR,
        env={**env, "SERVER_URL": server_url, "PORT": str(client_port)},
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
