"""Per-run prompt module binding.

Runs snapshot `core/prompts.py` at creation time. Pipeline tasks bind to
that snapshot so later prompt-source edits do not change an existing run's
cache keys or future resumed prompts.
"""

from __future__ import annotations

import hashlib
import importlib.util
from contextvars import ContextVar
from pathlib import Path
from types import ModuleType

from app.core import prompts as live_prompts

_current: ContextVar[ModuleType | None] = ContextVar(
    "current_prompt_module",
    default=None,
)
_cache: dict[Path, ModuleType] = {}


def bind(module: ModuleType | None) -> None:
    _current.set(module)


def current() -> ModuleType:
    return _current.get() or live_prompts


def load_snapshot(path: Path) -> ModuleType:
    path = path.resolve()
    cached = _cache.get(path)
    if cached is not None:
        return cached
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    name = f"app.core.prompts_snapshot_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load prompt snapshot: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _cache[path] = module
    return module
