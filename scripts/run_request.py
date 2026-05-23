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

import argparse
import json
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
    """Clean env for spawned subprocesses.

    `uv run --script` (how `enx test` invokes this file) sets VIRTUAL_ENV to
    its ephemeral env. Leaking that into `uv run uvicorn` in server/ makes
    the inner uv warn about a mismatched env and can mask failures. Strip
    the handful of uv-managed vars so each child resolves its own project
    environment from scratch.
    """
    env = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}}
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


def _promote_runs_dir(runs_dir: Path) -> int:
    """Flip every bbox-only completed slot in `runs_dir` into a resumable
    state so the full-pipeline server picks it up as paused. Identifies
    candidates as slots whose events.jsonl ends in `run.done` with zero
    `model` events (i.e. main_nomesh.py's _spawn_meshes was patched in
    and never wrote real meshes). The promotion just strips the trailing
    `run.done` line — every LLM decision is already cached in earlier
    `cache.llm` events, so the resumed run replays phase 1 instantly and
    runs phase 2 for real."""
    if not runs_dir.is_dir():
        print(f"[promote] runs dir not found: {runs_dir}", file=sys.stderr)
        return 1
    promoted = 0
    skipped = 0
    for slot_dir in sorted(runs_dir.iterdir()):
        if not slot_dir.is_dir():
            continue
        events_path = slot_dir / "events.jsonl"
        if not events_path.exists():
            continue
        events: list[dict[str, object]] = []
        with events_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    print(
                        f"[promote] {slot_dir.name}: malformed line, aborting this slot",
                        file=sys.stderr,
                    )
                    events = []
                    break
        if not events:
            continue
        last_kind = events[-1].get("kind")
        if last_kind != "run.done":
            print(f"[promote] {slot_dir.name}: last event is {last_kind!r}, skipping")
            skipped += 1
            continue
        if any(e.get("kind") == "model" for e in events):
            print(f"[promote] {slot_dir.name}: already has mesh `model` events, skipping")
            skipped += 1
            continue
        events.pop()
        with events_path.open("w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        print(f"[promote] {slot_dir.name}: dropped run.done — resume to generate meshes")
        promoted += 1
    print(f"[promote] done: promoted={promoted}, skipped={skipped}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Boot the full-pipeline API server + viewer.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=SERVER_DIR / "runs",
        help="Directory the server writes per-slot run artifacts to. Created if missing.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote bbox-only completed slots in --runs-dir to resumable so the "
             "next server boot picks them up as paused. LLM decisions replay from "
             "the events.jsonl cache; mesh generation runs for real on resume. "
             "Exits after promotion — does NOT boot the server.",
    )
    args = parser.parse_args()

    if args.promote:
        return _promote_runs_dir(args.runs_dir.resolve())

    if not (CLIENT_DIR / "node_modules" / "three").exists():
        print(
            "[run_request] client/node_modules/three missing — run `npm install` in client/ first",
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
        f"[run_request] starting API server on {server_url} (runs={runs_dir})",
        flush=True,
    )
    server = subprocess.Popen(
        [
            "uv", "run", "uvicorn", "app.main:app",
            "--host", SERVER_HOST, "--port", str(server_port),
            "--log-level", "info",
        ],
        cwd=SERVER_DIR,
        env={**env, "STARSHOT_RUNS_DIR": str(runs_dir)},
        process_group=0,
    )

    if not _wait_for_port(SERVER_HOST, server_port, proc=server, timeout=30.0):
        print(
            f"[run_request] server never became reachable at {server_url} — aborting",
            file=sys.stderr,
        )
        _shutdown(server)
        return 1
    print(f"[run_request] server ready, launching viewer at {client_url}", flush=True)

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
