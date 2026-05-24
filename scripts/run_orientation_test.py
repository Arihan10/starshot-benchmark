#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Boot the Trellis 2 orientation-fidelity test client.

Usage: ./scripts/run_orientation_test.py
"""

from __future__ import annotations

import contextlib
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
PORT = 8768


def _child_env() -> dict[str, str]:
    return {
        k: v for k, v in os.environ.items()
        if k not in {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
    }


def _popen_kwargs() -> dict[str, object]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"process_group": 0}


def _shutdown(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        with contextlib.suppress(Exception):
            ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM)
            proc.send_signal(ctrl_break)
            proc.wait(timeout=5)
            return
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
            return
        proc.kill()
        return

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    with contextlib.suppress(Exception):
        os.killpg(pgid, signal.SIGINT)
        proc.wait(timeout=5)
        return
    with contextlib.suppress(Exception):
        os.killpg(pgid, signal.SIGKILL)
        proc.wait(timeout=2)


def _wait_for_port(
    host: str,
    port: int,
    *,
    proc: subprocess.Popen[bytes],
    timeout: float,
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
    if "RUNWARE_API_KEY" not in os.environ and not (SERVER_DIR / ".env").exists():
        print(
            "[orientation-test] RUNWARE_API_KEY not set and server/.env missing; "
            "Runware calls will fail with an auth error",
            file=sys.stderr,
        )

    url = f"http://{HOST}:{PORT}/"
    print(f"[orientation-test] starting at {url}", flush=True)
    proc = subprocess.Popen(
        [
            "uv", "run", "uvicorn", "app.orientation_test:app",
            "--host", HOST, "--port", str(PORT),
            "--log-level", "info",
        ],
        cwd=SERVER_DIR,
        env=_child_env(),
        **_popen_kwargs(),
    )
    if not _wait_for_port(HOST, PORT, proc=proc, timeout=30.0):
        print(f"[orientation-test] server never became reachable at {url}", file=sys.stderr)
        _shutdown(proc)
        return 1
    with contextlib.suppress(Exception):
        webbrowser.open(url)
    try:
        return proc.wait() or 0
    except KeyboardInterrupt:
        return 0
    finally:
        _shutdown(proc)


if __name__ == "__main__":
    sys.exit(main())
