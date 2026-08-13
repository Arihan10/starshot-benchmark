#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Boot the API server with the PARALLEL-INTERIOR divider plus the three.js viewer.

Same full pipeline as run_request.py (real Trellis 2 + Nano Banana meshes) and
the same dashboard, but Phase 1 is split in two (`app.main_parallel`): the walk
plans, decomposes and FRAMES the whole tree first, then every zone's interior is
built concurrently instead of one zone at a time. Use it to A/B wall-clock and
scene quality against run_request.py on the same prompt + model.

Three knobs in `server/.env` shape how much that wins, all echoed at boot:
  STARSHOT_NEXT_OBJECT_CAP  caps the anchor completion loop (2 = the tuned value)
  USE_ASSET_LIBRARY         library matching vs. generating every mesh
  STARSHOT_STEP_REASONING   per-step thinking levels (slots.STEP_REASONING)

Usage: uv run scripts/run_parallel.py [--run NAME]
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

# Shared runs-root resolver ($STARSHOT_RUNS_DIR from server/.env, else <repo>/runs),
# so the dir we hand the server is the one it would have picked on its own.
sys.path.insert(0, str(REPO_ROOT))
from starshot_paths import runs_root, server_env_value  # noqa: E402

SERVER_HOST = "127.0.0.1"

# Same interpreter selection as run_request.py, so the parallel arm and the
# serial arm boot in the SAME env and only the traversal differs. See that file
# for why the CUDA env is launched directly rather than through `uv run`.
_SPLAT_PYTHON_DEFAULT = REPO_ROOT / ".venv-splat" / (
    "Scripts/python.exe" if os.name == "nt" else "bin/python"
)
SPLAT_PYTHON = Path(os.environ.get("STARSHOT_SPLAT_PYTHON", str(_SPLAT_PYTHON_DEFAULT)))
CUDA_HOME = os.environ.get("STARSHOT_CUDA_HOME", r"D:\cuda12")


def _child_env() -> dict[str, str]:
    """Clean env for spawned subprocesses — strip the uv-managed vars so each
    child resolves its own project environment from scratch."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
    }
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _server_command(server_port: int) -> tuple[list[str], dict[str, str], str]:
    """(argv, extra_env, label) to launch the API server on the split-phase divider."""
    uvicorn_args = [
        "-m", "uvicorn", "app.main_parallel:app",
        "--host", SERVER_HOST, "--port", str(server_port),
        "--log-level", "info",
    ]
    if SPLAT_PYTHON.is_file():
        extra_env: dict[str, str] = {}
        if Path(CUDA_HOME).is_dir():
            extra_env["CUDA_HOME"] = CUDA_HOME
            extra_env["CUDA_PATH"] = CUDA_HOME
        return [str(SPLAT_PYTHON), *uvicorn_args], extra_env, f"CUDA env {SPLAT_PYTHON}"
    return ["uv", "run", "python", *uvicorn_args], {}, "uv run (server env — no torch)"


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
    if os.name == "nt":
        # Windows has no POSIX process groups / SIGINT-to-tree, and `uv run` spawns
        # uvicorn (or node) as a child — so force-kill the whole tree by PID.
        # Without this, os.getpgid/os.killpg below raise AttributeError on Windows
        # and the server is orphaned, draining in-flight work in the background.
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
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


def _settings_banner() -> str:
    """The `server/.env` settings that decide how much the split walk actually
    buys — worth seeing before committing to a paid run."""
    cap = os.environ.get("STARSHOT_NEXT_OBJECT_CAP") or server_env_value(
        "STARSHOT_NEXT_OBJECT_CAP"
    )
    library = os.environ.get("USE_ASSET_LIBRARY") or server_env_value("USE_ASSET_LIBRARY")
    per_step = os.environ.get("STARSHOT_STEP_REASONING") or server_env_value(
        "STARSHOT_STEP_REASONING"
    )
    return (
        f"next_object cap={cap or 'UNCAPPED'}, "
        f"asset library={'on' if (library or 'true').lower() != 'false' else 'off (from scratch)'}, "
        f"thinking={'per-step' if (per_step or 'false').lower() == 'true' else 'per-model'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Boot the parallel-interior API server + viewer.",
    )
    parser.add_argument(
        "--run",
        dest="run",
        default=None,
        help="Subfolder name under the runs root ($STARSHOT_RUNS_DIR, default "
             "<repo>/runs) to write per-slot artifacts into. Omit to use the "
             "runs root directly.",
    )
    args = parser.parse_args()

    if not (CLIENT_DIR / "node_modules" / "three").exists():
        print(
            "[run_parallel] client/node_modules/three missing — run `npm install` in client/ first",
            file=sys.stderr,
        )
        return 1

    runs_base = runs_root()
    runs_dir = (runs_base / args.run if args.run else runs_base).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    server_port = _pick_free_port()
    client_port = _pick_free_port()
    server_url = f"http://{SERVER_HOST}:{server_port}"
    client_url = f"http://{SERVER_HOST}:{client_port}"

    env = _child_env()

    server_cmd, server_extra_env, server_label = _server_command(server_port)
    print(
        f"[run_parallel] starting API server on {server_url} "
        f"(parallel interiors, runs={runs_dir}) [{server_label}]",
        flush=True,
    )
    print(f"[run_parallel] {_settings_banner()}", flush=True)
    server = subprocess.Popen(
        server_cmd,
        cwd=SERVER_DIR,
        env={
            **env,
            "STARSHOT_RUNS_DIR": str(runs_dir),
            "STARSHOT_API_ORIGIN": server_url,
            "STARSHOT_CLIENT_ORIGIN": client_url,
            **server_extra_env,
        },
        process_group=0,
    )

    if not _wait_for_port(SERVER_HOST, server_port, proc=server, timeout=30.0):
        print(
            f"[run_parallel] server never became reachable at {server_url} — aborting",
            file=sys.stderr,
        )
        _shutdown(server)
        return 1
    print(f"[run_parallel] server ready, launching viewer at {client_url}", flush=True)

    client = subprocess.Popen(
        ["node", "server.mjs"],
        cwd=CLIENT_DIR,
        env={
            **env,
            "SERVER_URL": server_url,
            "PORT": str(client_port),
            "PIPELINE_MODE": "parallel",
        },
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
