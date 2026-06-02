"""Pipeline version registry for the A/B/C benchmark.

A *version* bundles a prompt module with a divider/generation code path.
The three versions isolate two independent variables:

  * v1-legacy-xml   - the original pipeline verbatim: XML-node prompts
    (`prompts_old`) on the extracted `divider_old`/`generation_old`, which
    frame each zone BEFORE decomposing it.
  * v2-frame-first  - the prompt WORDING pinned to commit 28830e8
    (`prompts_v2`) on the current `divider`/`generation`, framing BEFORE
    decomposition.
  * v3-decomp-first - the latest "jace" prompt wording (live `prompts.py`),
    framing AFTER decomposition (today's default behavior).

v2 and v3 share every context/data improvement made since 28830e8; they
carry different prompt WORDING (pinned-old vs latest) — the variable the
A/B test isolates — and currently also differ in `frame_order`. v1 swaps
the whole reasoning/prompt structure. All three share the mesh backend,
event schema, `types`, and geometry, so the comparison measures
pipeline/prompt structure rather than infra.

Each version is modeled as a reserved *run* (`runs/<run_name>/...`), so the
existing per-run isolation (prompt_runtime binding, per-cell asyncio tasks,
separate events.jsonl + artifacts) keeps the three from leaking into one
another. The `_pending`/`_admitted_ids` task tables live in whichever
generation module a version uses AND are keyed by the composite `run_id`
(which embeds `run_name`), so v2 and v3 never collide and v1 lives in its
own module entirely.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol

from app.core import prompts_old, prompts_v2
from app.core.types import Node
from app.pipeline import divider, divider_old, generation, generation_old

RunFn = Callable[..., Awaitable[Node]]


class GenerationModule(Protocol):
    """The subset of a generation module's surface that routes drives
    directly. Both `generation` and `generation_old` satisfy it, so a
    version can route its mesh lifecycle to the right module (and the right
    `_pending` table) without the caller knowing which one it is."""

    def cancel_pending(self, run_id: str, mode: str | None = None) -> None: ...
    def await_pending(self, run_id: str, mode: str | None = None) -> Awaitable[None]: ...
    def retry_node(
        self, *, node: Node, runs_dir: Path, run_id: str, mode: str | None = None
    ) -> Awaitable[asyncio.Task[None]]: ...


@dataclass(frozen=True)
class PipelineVersion:
    id: str
    run_name: str
    label: str
    # Async dispatch entry point — `(*, run_id, prompt, model, runs_dir) -> Node`.
    run: RunFn
    # The generation module whose `cancel_pending`/`await_pending`/
    # `retry_node` + `_pending` table this version's meshes live in.
    generation: GenerationModule
    # The prompt module to bind for this version, or None to use the live
    # `prompts.py` (resolved per-run by routes). v1 pins `prompts_old`.
    prompt_module: ModuleType | None


async def _run_v1(*, run_id: str, prompt: str, model: str, runs_dir: Path) -> Node:
    return await divider_old.run(
        run_id=run_id, prompt=prompt, model=model, runs_dir=runs_dir,
    )


async def _run_v2(*, run_id: str, prompt: str, model: str, runs_dir: Path) -> Node:
    return await divider.run(
        run_id=run_id, prompt=prompt, model=model, runs_dir=runs_dir,
        frame_order="before",
    )


async def _run_v3(*, run_id: str, prompt: str, model: str, runs_dir: Path) -> Node:
    return await divider.run(
        run_id=run_id, prompt=prompt, model=model, runs_dir=runs_dir,
        frame_order="after",
    )


VERSIONS: list[PipelineVersion] = [
    PipelineVersion(
        id="v1",
        run_name="v1-legacy-xml",
        label="V1: Legacy XML (frame-first)",
        run=_run_v1,
        generation=generation_old,
        prompt_module=prompts_old,
    ),
    PipelineVersion(
        id="v2",
        run_name="v2-frame-first",
        label="V2: Old prompts (frame-first)",
        run=_run_v2,
        generation=generation,
        prompt_module=prompts_v2,
    ),
    PipelineVersion(
        id="v3",
        run_name="v3-decomp-first",
        label="V3: New prompts (decomp-first)",
        run=_run_v3,
        generation=generation,
        prompt_module=None,
    ),
]

# v3 mirrors today's behavior, so any non-version run (e.g. arbitrary
# user-created runs) falls back to it.
DEFAULT_VERSION: PipelineVersion = VERSIONS[2]

_BY_RUN_NAME: dict[str, PipelineVersion] = {v.run_name: v for v in VERSIONS}


def for_run(run_name: str) -> PipelineVersion:
    """The version that owns `run_name`, or DEFAULT_VERSION (v3) for any run
    that isn't one of the three reserved version runs."""
    return _BY_RUN_NAME.get(run_name, DEFAULT_VERSION)
