"""Task-local ablation seam — mirrors ``llm.set_model`` / ``prompt_store.bind``.

A normal run has no ablation (``current()`` is ``None``). An inherited variant
run binds its ``AblationRuntime`` at pipeline entry; the scene renderer reads
``current()`` and, for calls whose step kind matches ``target_step_kind``,
applies the treatment (shuffle order / distractors / …) to the scene payload it
builds. Being a ``ContextVar``, concurrent cells / variants don't race.
"""

from __future__ import annotations

import contextvars

from .config import AblationRuntime

_current: contextvars.ContextVar[AblationRuntime | None] = contextvars.ContextVar(
    "ablation_runtime", default=None
)


def set_runtime(runtime: AblationRuntime | None) -> None:
    """Bind (or clear, with ``None``) the active variant's treatment for this task."""
    _current.set(runtime)


def current() -> AblationRuntime | None:
    """The active variant treatment, or ``None`` on a normal (non-ablation) run."""
    return _current.get()


def clear() -> None:
    _current.set(None)


class AblationComplete(BaseException):
    """Raised the instant an ablation variant re-infers its treated step, to
    unwind the pipeline and skip ALL downstream work — next_object, object_bbox,
    image prompts, library/prefab matching (which run on gemini-flash-lite), and
    mesh generation. A variant exists only to capture the treated step's
    attention, so none of that is wanted. Subclasses BaseException so the
    pipeline's ``except Exception`` handlers don't swallow it; the run task
    catches it and marks the cell ``run.done``."""


def stop_if_treated_step(step: str | None) -> None:
    """Call right after a NEW (non-replayed) LLM call is committed. On an ablation
    run, once the treated target step has fired, raise ``AblationComplete`` to end
    the run there. During teacher-forced replay of the shared prefix the calls hit
    the cache and never reach here, so the first match is the treated firing at the
    fork cut."""
    rt = _current.get()
    if rt is not None and step and step == rt.target_step_kind:
        raise AblationComplete()
