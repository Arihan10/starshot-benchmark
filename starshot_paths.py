"""Where the local runs tree lives.

A *run* is one named subdirectory of the runs root; cells sit at
``<runs root>/<run>/<slot>/<model>``. Every entry point — the API server, the
launchers in ``scripts/``, the offline tools — resolves that root through
``runs_root()`` so they all agree on one location: ``STARSHOT_RUNS_DIR`` from
the process environment, else the value declared in ``server/.env``, else
``<repo>/runs``. A relative value is anchored to the repo root, never to the
launch CWD (the server runs with ``cwd=server/``, tools run from anywhere).

``server/.env`` is parsed here instead of through python-dotenv because the
launchers run under ``uv run --script`` with no third-party dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR_ENV = "STARSHOT_RUNS_DIR"
_SERVER_ENV_FILE = REPO_ROOT / "server" / ".env"


def _server_env_value(key: str) -> str | None:
    try:
        text = _SERVER_ENV_FILE.read_text(encoding="utf-8")
    except OSError:
        return None
    for raw in text.splitlines():
        line = raw.strip().removeprefix("export ").lstrip()
        name, sep, value = line.partition("=")
        if not sep or name.strip() != key:
            continue
        value = value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        return value or None
    return None


def runs_root() -> Path:
    """Parent directory holding the named runs."""
    value = os.environ.get(RUNS_DIR_ENV) or _server_env_value(RUNS_DIR_ENV)
    if not value:
        return REPO_ROOT / "runs"
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path
