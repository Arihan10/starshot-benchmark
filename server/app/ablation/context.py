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


def sampling_for(step: str | None) -> dict | None:
    """Sampling overrides for a treated ablation step, else ``None``. When the
    current run is a variant and `step` is its target kind, return the treatment's
    `temperature` (if set) — a positive temperature makes each replicate an
    INDEPENDENT draw (fresh sampling per call), which is what raises certainty.

    NOTE: we deliberately do NOT put `seed` on the wire. OpenRouter is called with
    ``provider.require_parameters=True`` (needed so json_schema is honored), which
    also demands the chosen provider support EVERY param sent — and `seed` is
    unsupported by most open-model providers (qwen/gemma), so sending it 404s the
    whole call ("no endpoints found that support your parameters"). `temperature`
    is universally supported, so it's safe. The treatment's `seed` still drives the
    server-side scene shuffle (that never touches the provider)."""
    rt = _current.get()
    if rt is None or not step or step != rt.target_step_kind:
        return None
    temp = getattr(rt.treatment, "temperature", None)
    return {"temperature": float(temp)} if temp is not None else None


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
