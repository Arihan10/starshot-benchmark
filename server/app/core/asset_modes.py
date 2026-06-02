"""Per-cell asset mode — library matching vs. from-scratch generation.

Historically a single `USE_ASSET_LIBRARY` env flag chose, process-wide,
whether the generation pipeline matched each object to a pre-built asset
library or generated it from scratch (Nano-Banana + Trellis). That made
the choice global and required a server restart to flip.

The mode is now a request-scoped value the client picks per cell, exactly
like the LLM model selection (`llm.set_model`) and the prompt module
(`prompt_runtime.bind`). Each `(run, slot, model)` cell can hold BOTH a
library build and a generated build at once, kept apart on disk so they
never clobber each other:

  * library  -> `events.jsonl`           + `objects/`           (the legacy
    layout; any pre-existing cell with an `objects/` folder is a library
    build, so reading it back stays backwards-compatible)
  * generated -> `events.generated.jsonl` + `objects-generated/`

Pipeline code reads the bound mode via `current()`; the divider/generation
modules use `objects_subdir()` to pick where meshes land. The HTTP layer
binds the mode at the top of each run task (like `logging.bind`) and threads
it through the cell endpoints as a query param.
"""

from __future__ import annotations

import os
from contextvars import ContextVar

LIBRARY = "library"
GENERATED = "generated"
ASSET_MODES: tuple[str, ...] = (LIBRARY, GENERATED)

# Default for any request that doesn't name a mode. Honors the legacy
# USE_ASSET_LIBRARY env so an operator who set it false to default to
# from-scratch generation keeps that behavior; the client toggle overrides
# it per cell regardless. Library is the fallback so existing `objects/`
# cells read back as library builds.
DEFAULT_ASSET_MODE = (
    LIBRARY
    if os.environ.get("USE_ASSET_LIBRARY", "true").lower() == "true"
    else GENERATED
)

# Where each mode writes its meshes/reference images under the cell dir.
# library keeps the legacy "objects" name; generated is a sibling folder so
# the two coexist. (Serving may map library -> objects-optimized; see routes.)
_OBJECTS_SUBDIR = {LIBRARY: "objects", GENERATED: "objects-generated"}

# Per-mode event log filename within the cell dir. library keeps the legacy
# events.jsonl so old runs hydrate unchanged.
_EVENTS_FILENAME = {LIBRARY: "events.jsonl", GENERATED: "events.generated.jsonl"}


def normalize(mode: str | None) -> str:
    """Coerce an arbitrary input to a known mode, falling back to the
    default. Keeps a malformed `?mode=` query param from ever selecting a
    bogus folder."""
    return mode if mode in ASSET_MODES else DEFAULT_ASSET_MODE


def objects_subdir(mode: str | None = None) -> str:
    """Mesh output subdir for `mode` (or the currently-bound mode)."""
    return _OBJECTS_SUBDIR[normalize(mode if mode is not None else current())]


def events_filename(mode: str | None) -> str:
    """Event-log filename for `mode` within a cell directory."""
    return _EVENTS_FILENAME[normalize(mode)]


_current: ContextVar[str] = ContextVar("asset_mode", default=DEFAULT_ASSET_MODE)


def bind(mode: str | None) -> None:
    """Bind the asset mode for the current asyncio task. Call at the top of
    each run/retry task — subsequent awaits (and tasks they spawn, which
    copy the context) inherit it."""
    _current.set(normalize(mode))


def current() -> str:
    return _current.get()


def uses_library(mode: str | None = None) -> bool:
    return normalize(mode if mode is not None else current()) == LIBRARY
