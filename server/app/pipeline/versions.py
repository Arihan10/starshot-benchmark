"""Pipeline version registry for the A/B benchmark.

A *version* bundles a prompt module with a divider/generation code path.

  * v1-legacy-xml       - the original pipeline verbatim: XML-node prompts
    (`prompts_old`) on the extracted `divider_old`/`generation_old`, which
    frame each zone BEFORE decomposing it.
  * v2-frame-first      - the pinned `prompts_v2` snapshot on the current
    `divider`/`generation`, framing EVERY zone BEFORE it is decomposed
    (the root included).
  * v3-decomp-first     - the live `prompts.py` on the current
    `divider`/`generation`, framing the ROOT AFTER its own decomposition
    but every other zone BEFORE — today's default behavior. Its
    anchor-completion (next_object) step proposes a LIST of objects per
    round.
  * v4-decomp-first-all - identical to v3 (same live `prompts.py`, same
    `divider`/`generation`, same list-based anchor completion), except it
    frames EVERY zone AFTER its own decomposition, not just the root.

v3 and v4 bind the SAME live `prompts.py`, so their prompt wording is
identical; v2 binds the pinned `prompts_v2` snapshot (which `prompts.py` was
copied from but has since picked up small prompt edits). The variable the
benchmark isolates across v2/v3/v4 is `frame_order` — when each zone is framed
relative to its own decomposition:
  * v2 = "before"     (every zone framed before it is decomposed)
  * v3 = "after_root" (root framed after; every other zone before)
  * v4 = "after"      (every zone framed after it is decomposed)

A second axis, `batch_next_object`, separates the two prompts.py versions from
the snapshot baseline: in the anchor-completion (next_object) loop, v3/v4 let
the model propose a LIST of objects per round, while v2 proposes one object at a
time. Bounding-box resolution is a single batch call per sibling set in every
version. v1 swaps the whole reasoning/prompt structure. All share the mesh
backend, event schema, `types`, and geometry, so the comparison measures
pipeline structure rather than infra.

Each version is modeled as a reserved *run* (`runs/<run_name>/...`), so the
existing per-run isolation (prompt_runtime binding, per-cell asyncio tasks,
separate events.jsonl + artifacts) keeps the versions from leaking into one
another. The `_pending`/`_admitted_ids` task tables live in whichever
generation module a version uses AND are keyed by the composite `run_id`
(which embeds `run_name`), so v2/v3/v4 never collide and v1 lives in its own
module entirely.
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

    def cancel_pending(self, run_id: str) -> None: ...
    def await_pending(self, run_id: str) -> Awaitable[None]: ...
    def retry_node(
        self, *, node: Node, runs_dir: Path, run_id: str
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
        frame_order="before", batch_next_object=False,
    )


async def _run_v3(*, run_id: str, prompt: str, model: str, runs_dir: Path) -> Node:
    return await divider.run(
        run_id=run_id, prompt=prompt, model=model, runs_dir=runs_dir,
        frame_order="after_root", batch_next_object=True,
    )


async def _run_v4(*, run_id: str, prompt: str, model: str, runs_dir: Path) -> Node:
    return await divider.run(
        run_id=run_id, prompt=prompt, model=model, runs_dir=runs_dir,
        frame_order="after", batch_next_object=True,
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
        label="V2: shared prompts, root framed before decompose",
        run=_run_v2,
        generation=generation,
        prompt_module=prompts_v2,
    ),
    PipelineVersion(
        id="v3",
        run_name="v3-decomp-first",
        label="V3: improved prompts, root framed after decompose",
        run=_run_v3,
        generation=generation,
        prompt_module=None,
    ),
    PipelineVersion(
        id="v4",
        run_name="v4-decomp-first-all",
        label="V4: improved prompts, every zone framed after decompose",
        run=_run_v4,
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
    that isn't one of the reserved version runs."""
    return _BY_RUN_NAME.get(run_name, DEFAULT_VERSION)
