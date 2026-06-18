"""HTTP API — endpoints scoped to a (run, slot, model) cell.

`RUNS_DIR` is a parent directory holding many named *runs*; each run is a
set of (slot, model) cells with its own prompt snapshot (`prompts/`, copied
from a `versions/` prompt version at creation), events.jsonl, mesh
artifacts, and SSE streams. The viewer picks the active run at runtime via
`GET /runs` / `POST /runs/{name}/activate`.

For each (run, slot, model) cell: fresh ones sit idle, interrupted ones
come back as paused, completed ones stay done. Nothing auto-launches;
the viewer drives start/resume/reset per cell. Runs are hydrated lazily
on activation; the initial active run is the newest subdir of RUNS_DIR.

Every asyncio task is bound to its SlotLog via a ContextVar so
concurrent pipeline work routes events to the right cell without
threading a handle through every call site. The Trellis queue is
process-global; rows tag `slot_id` with the composite
`run/slot/model_alias` so the dashboard filters to the visible cell.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import os
import re
import shutil
import struct
from datetime import datetime
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.core import prompt_store, scene_context, schemas
from app.utils import cache
from app.oneshot import routes as oneshot
from app.core.slots import (
    DEFAULT_MODEL_ALIAS,
    MODEL_ALIASES,
    MODELS,
    SLOTS,
    SLOTS_BY_ID,
    Slot,
)
from app.core.types import BoundingBox, Node, Orientation, ProxyShape
from app.pipeline import divider, generation
from app.services import llm, threed
from app.utils import logging as rlog
from app.utils.logging import SlotLog

# Parent directory holding many named runs. Each immediate subdirectory
# is one run; cells live at RUNS_DIR/<run>/<slot>/<model>. Anchored to the
# repo root (this file is server/app/api/routes.py) rather than the launch
# CWD, mirroring prompt_store.VERSIONS_DIR.
_REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = Path(os.environ.get("STARSHOT_RUNS_DIR", _REPO_ROOT / "runs"))

# Which per-cell mesh directory the scene bundle streams from. Defaults to the
# originals ("objects"); set STARSHOT_OBJECTS_SUBDIR=objects-optimized to serve
# the re-baked optimized set (scripts/rebake_runs.py) instead. Falls back to
# "objects" for any cell that hasn't been migrated.
OBJECTS_SUBDIR = os.environ.get("STARSHOT_OBJECTS_SUBDIR", "objects")

# Python's mimetypes doesn't know glTF; without these the artifact route would
# hand the loader its GLBs as text/plain.
_ARTIFACT_MEDIA_TYPES = {".glb": "model/gltf-binary", ".gltf": "model/gltf+json"}

# Per-run metadata written at creation (chosen prompt version, created_at).
RUN_META_NAME = "run.json"

# Keyed by (run_name, slot_id, model_alias). Each cell is an independent
# pipeline. Lazy-populated: only runs the user has activated are loaded.
RunKey = tuple[str, str, str]
_slot_logs: dict[RunKey, SlotLog] = {}
_tasks: dict[RunKey, asyncio.Task[None]] = {}
_retry_tasks: set[asyncio.Task[None]] = set()
_hydrated_runs: set[str] = set()
_current_run: str = ""

# Downstream-simulation BRANCHES. A branch is an ephemeral "what-if" fork of
# one cell: the original events up to (excluding) an edited step's call, then
# the pipeline re-run with the run's prompt snapshot PLUS the edited step
# templates, so that step — and every later firing of it — renders from the
# edit while everything else replays via `committed.*`. Each branch is fully
# isolated from its source — its own SlotLog + events.jsonl + objects dir
# under `<cell>/_branch/`, and its own composite run_id
# (`run/slot/model/_branch`) so the LLM cache, the `committed.*` resume
# reader, `generation._pending`, and the Trellis queue all key off the
# branch, never the source.
#
# One branch per CELL; branches across cells run in parallel (the prompt
# lab's horizontal simulation), each individually pausable/resumable/
# discardable. Wiped on break-out (DELETE) and on any source-cell mutation
# (reset/rewind). In-memory only: a server restart orphans branch dirs until
# the next branch/reset on that cell replaces them.
BRANCH_SUBDIR = "_branch"
_branch_logs: dict[RunKey, SlotLog] = {}
_branch_tasks: dict[RunKey, asyncio.Task[None]] = {}
# The prompt-template overrides each live branch simulates with — bound (on
# top of the run snapshot) every time its task (re)launches.
_branch_overrides: dict[RunKey, dict[str, dict[str, str]]] = {}
# Step gates: gated pipelines advance ONE LLM call at a time, pausing before
# each frontier call until a step endpoint releases it (or flips the task to
# auto). Branches are ALWAYS gated; source cells are gated when launched in
# stepped mode (the per-step prompt-iteration workflow).
_branch_gates: dict[RunKey, "CellGate"] = {}
# Next-launch gate intent for a PAUSED branch (budget / auto / until) — the
# branch mirror of `_gate_intents`, consumed by `_run_branch` when a step
# advance relaunches a paused branch.
_branch_gate_intents: dict[RunKey, dict[str, object]] = {}
_cell_gates: dict[RunKey, "CellGate"] = {}
# Source cells currently in stepped mode (one LLM call per "step"). Mirrored
# on disk by a `.stepped` marker in each cell dir so the mode SURVIVES a
# server restart — otherwise a stepped run would come back as a plain paused
# cell with no way to advance it.
_stepped_cells: set[RunKey] = set()
# The intent applied to the NEXT gate a cell's `_run` task creates:
#   {"budget": N} — pass N frontier calls without pausing, then pause (a
#     "step" launches a paused cell with budget 1; rerun-step uses it to
#     auto-execute the edited step before pausing on the one after).
#   {"auto": True} — never pause (a stepped cell told to "run rest").
_gate_intents: dict[RunKey, dict[str, object]] = {}


def _stepped_marker(key: RunKey) -> Path:
    return _slot_dir(*key) / ".stepped"


def _set_stepped(key: RunKey, on: bool) -> None:
    """Toggle a cell's stepped mode in memory AND on disk so it persists
    across restarts."""
    marker = _stepped_marker(key)
    if on:
        _stepped_cells.add(key)
        with contextlib.suppress(OSError):
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("")
    else:
        _stepped_cells.discard(key)
        marker.unlink(missing_ok=True)


class CellGate:
    """Pauses a gated pipeline before each real (cache-miss) LLM call until
    the user advances it. `budget` pre-grants pause-free calls (rerun-step
    uses it to auto-execute the edited step). Single-event-loop, so the
    cross-coroutine `Future.set_result` from the endpoint safely wakes the
    paused task."""

    def __init__(self, slot_log: SlotLog, *, budget: int = 0, model_override: str | None = None) -> None:
        self.slot_log = slot_log
        # `budget` is the count of QUEUED steps: each lets the cell pass one
        # gate without pausing. "Step all" grants one to every stepped cell —
        # including ones still mid-call — so a slower model doesn't miss the
        # step and need an individual catch-up later; it consumes the credit
        # when it reaches its next gate, keeping the whole run in lockstep.
        self.budget = budget
        self._fut: asyncio.Future[dict[str, object]] | None = None
        self.auto = False  # once set, the rest of the run proceeds without pausing
        # Fast-forward target: pass every call until one with this TEMPLATE comes
        # up, then pause right before it ("run to breakpoint at step X").
        self.until_step: str | None = None
        self.pending: dict[str, object] | None = None
        # The call CURRENTLY in flight — released from a pause, or run on a
        # budget credit / in auto. Lets the UI show "running X" instead of the
        # stale last phase once a gated step is released. None until one runs.
        self.current: dict[str, object] | None = None
        # Per-step LLM override (compare's "run this step on model X"): the
        # OpenRouter model id `wait` hands back so `call_llm` re-aims the gated
        # call at it. Sticky until changed — "run rest" + later steps follow the
        # choice — so the same scene context can be A/B'd across models.
        self.model_override: str | None = model_override

    async def wait(
        self, *, node_id: str | None, step: str | None, template: str | None = None,
        system: str, user: str, schema_name: str, model: str,
    ) -> str | None:
        # Library matching is a mechanical per-object service step (always on its
        # own gemini model), not a pipeline step worth click-through — never gate
        # it and never let a model override re-aim it.
        if step == "library_match":
            return None
        call = {"node": node_id, "step": step, "template": template, "schema": schema_name, "model": model}
        if self.auto:
            self.current = call
            return self.model_override
        # Seeking a target step: blow past everything else, then stop AT it.
        # `until_step` is a template id (the lab's granular step name), so match
        # on `template` (root/nested variants differ there, not in `step`).
        if self.until_step is not None:
            if (template or step) == self.until_step:
                self.until_step = None  # arrived — fall through and pause here
            else:
                self.current = call
                return self.model_override
        elif self.budget > 0:
            self.budget -= 1
            self.current = call
            return self.model_override
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, object]] = loop.create_future()
        self._fut = fut
        self.pending = call
        # Surface the pending call (including the exact prompt about to be
        # sent) on the stream so the UI can preview it.
        self.slot_log.log(
            "branch.step.pending",
            node=node_id,
            step=step,
            template=template,
            schema=schema_name,
            model=model,
            system=system,
            user=user,
        )
        try:
            result = await fut
        finally:
            self._fut = None
            self.pending = None
            self.current = call  # released — this call is now the one in flight
        if result.get("auto"):
            self.auto = True
        return self.model_override

    def proceed(self, *, auto: bool = False) -> bool:
        """Release the current pause. Returns False if nothing is pending."""
        if self._fut is None or self._fut.done():
            return False
        self._fut.set_result({"auto": auto})
        return True


class RewindRequest(BaseModel):
    to_event_index: int


class CreateRunRequest(BaseModel):
    name: str
    # Name of a source prompt version (a subfolder of `versions/`). Copied
    # into the run as its immutable prompt snapshot at creation.
    prompt_version: str


class PromptTestRequest(BaseModel):
    """One-shot re-run of a HISTORICAL step call with edited templates: the
    target `cache.llm` event's logged `variables` are substituted into
    `system_template`/`user_template` and the call re-issued on the event's
    own model. Nothing is logged or cached — the result is display-only."""

    run: str
    slot: str
    model: str
    event_index: int
    step: str
    system_template: str
    user_template: str


class InquiryMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class InquiryRequest(BaseModel):
    """One turn in a persistent "why did the model do this?" conversation about
    a pipeline step. The client assembles and OWNS the reviewer's full system
    prompt — the analyst framing plus that step's exact system / input / output
    / reasoning — and sends it verbatim as `system`, so what the panel shows
    (prefilled and editable) is byte-for-byte what is sent. `messages` is the
    running thread and MUST end with the new user turn. Stateless: the endpoint
    reads no cell state, so it serves source and simulation-branch calls alike."""

    system: str = ""
    messages: list[InquiryMessage]


class BranchSeed(BaseModel):
    """A prompt-test result to pre-commit into a fresh branch's log, so
    "simulate downstream" continues from EXACTLY the output the user vetted
    (instead of re-rolling the edited step). `system`/`user` are the rendered
    bytes the test actually sent."""

    system: str
    user: str
    output: dict[str, object]
    reasoning: str = ""
    tokens_in: int | None = None
    tokens_out: int | None = None


class BranchRequest(BaseModel):
    """Fork the cell so it re-simulates from `event_index` onward — which
    MUST point at a `cache.llm` event of template `step` carrying logged
    `variables`. The branch keeps the events BEFORE that call and re-runs the
    pipeline with the run snapshot + `overrides` — the prompt lab's FULL edit
    set (`{step: {"system": text, "user": text}}`), so a session editing
    several steps simulates exactly what "save to new run" would persist.
    Replaces any prior branch on this cell."""

    event_index: int
    step: str
    overrides: dict[str, dict[str, str]]
    seed: BranchSeed | None = None


class BranchStepRequest(BaseModel):
    """Advance a gated cell/branch. `auto=True` runs the rest to completion;
    `until` (a template id) fast-forwards a source cell to the next call of
    that step and pauses there. `model` (a model ALIAS) re-aims a branch's next
    gated call at a chosen LLM — compare's per-step model A/B — independent of
    the model that produced the pre-branch scene; None keeps the branch's
    current model."""

    auto: bool = False
    until: str | None = None
    model: str | None = None


class BranchRewindRequest(BaseModel):
    """Revert a simulation branch to before `to_event_index` (a re-renderable
    branch `cache.llm` call): drop that call and everything after it, delete the
    meshes generated at/after the cut, and pause there. `overrides` (the lab's
    CURRENT edit set) refreshes the branch's templates so the subsequent step
    re-runs under the current snapshot + edits — the sim's source of truth;
    `None` keeps the branch's existing edit set. The source cell and the
    branch's replayed prefix meshes are untouched."""

    to_event_index: int
    overrides: dict[str, dict[str, str]] | None = None


class ForkVersionRequest(BaseModel):
    """Copy an existing source version to a new name — the "start a fresh
    version to iterate on" entry point used by run creation."""

    name: str
    base: str


class UpdateRunPromptsRequest(BaseModel):
    """Write the prompt lab's edits INTO the run's snapshot (the in-place
    iteration loop). Running cells keep the templates they launched with;
    every relaunch/rerun/resume renders the new bytes. With `update_version`
    the run's source version folder is kept in sync too."""

    overrides: dict[str, dict[str, str]]
    update_version: bool = False


class SimulateStepRequest(BaseModel):
    """Fork a SIMULATION branch in each cell at its EARLIEST call of any
    template in `steps`, running from there under the run's (freshly edited)
    snapshot — the non-destructive iteration loop. Source cells are untouched;
    a branch is promoted to source on demand (/branch/commit). Omitted `cells`
    means every cell that has logged a call of one of those steps."""

    steps: list[str]
    cells: list[dict[str, str]] | None = None


class SaveVersionRequest(BaseModel):
    """Persist `base_run`'s prompt snapshot + the prompt lab's edited step
    templates as a brand-new source version folder under `versions/`."""

    name: str
    base_run: str
    overrides: dict[str, dict[str, str]]


class SaveRunRequest(BaseModel):
    """Materialize the prompt lab's simulation as a NEW run: prompts/ =
    `base_run`'s snapshot + `overrides`, and each listed cell's BRANCH state
    (events + meshes) copied in as that cell's history — paused mid-pipeline
    states stay resumable because the new snapshot matches the prompts the
    branch actually ran with."""

    name: str
    base_run: str
    overrides: dict[str, dict[str, str]]
    cells: list[dict[str, str]]
    # Display label for run.json; pass the freshly-saved version name when
    # "save to new version" ran first.
    version_label: str | None = None


def _run_id(run: str, slot_id: str, model_alias: str) -> str:
    """Composite id used as `run_id` in pipeline code (divider, generation,
    threed queue, SlotLog.slot_id). The slashes make it work as a filesystem
    subpath under RUNS_DIR and as an artifact URL segment under /artifacts."""
    return f"{run}/{slot_id}/{model_alias}"


def _run_dir(run: str) -> Path:
    return RUNS_DIR / run


def _slot_dir(run: str, slot_id: str, model_alias: str) -> Path:
    return RUNS_DIR / run / slot_id / model_alias


def _branch_dir(run: str, slot_id: str, model_alias: str) -> Path:
    return _slot_dir(run, slot_id, model_alias) / BRANCH_SUBDIR


def _branch_run_id(run: str, slot_id: str, model_alias: str) -> str:
    """Composite run_id for a branch — the source cell's run_id plus the
    `_branch` segment, so meshes land in `<cell>/_branch/objects/` and every
    run_id-keyed table (`generation._pending`, the Trellis queue) is isolated
    from the source."""
    return f"{_run_id(run, slot_id, model_alias)}/{BRANCH_SUBDIR}"


def _resolve_run(run: str | None) -> str:
    """Every cell endpoint names its target run explicitly so concurrently-
    running runs never route through a shared global. The client always sends
    `?run=`; `_current_run` is only the fallback for a client that hasn't
    picked one yet (boot, or a legacy caller)."""
    return run or _current_run


def _run_meta(run: str) -> dict[str, object]:
    """The run's `run.json` (prompt version name, created_at), or {} for runs
    that predate the file-based prompt versioning."""
    path = _run_dir(run) / RUN_META_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _require_run_prompts(run: str) -> prompt_store.PromptSet:
    """The run's prompt snapshot, required for anything that (re)starts
    pipeline work. Runs created before the file-based prompt versioning have
    no snapshot — they stay loadable (scene/meshes/events) but can't run."""
    run_dir = _run_dir(run)
    if not prompt_store.has_run_prompts(run_dir):
        raise HTTPException(
            status_code=409,
            detail=f"run {run!r} has no prompt snapshot (legacy run) — create a new run to launch pipelines",
        )
    try:
        return prompt_store.load_run_prompts(run_dir)
    except prompt_store.PromptTemplateError as e:
        raise HTTPException(status_code=409, detail=str(e))


def _require_step_event(
    run: str, slot_id: str, model_alias: str, event_index: int, step: str,
) -> tuple[SlotLog, dict[str, object]]:
    """The cell's `cache.llm` event at `event_index`, validated as a
    re-renderable call of template `step` (the prompt lab's eligibility rule:
    same template, already ran, variables logged)."""
    if step not in prompt_store.STEPS:
        raise HTTPException(status_code=404, detail=f"unknown step: {step}")
    slot_log = _require_slot_log(run, slot_id, model_alias)
    events = slot_log.state["events"]
    if not (0 <= event_index < len(events)):
        raise HTTPException(status_code=400, detail="event_index out of range")
    event = events[event_index]
    if event.get("kind") != "cache.llm":
        raise HTTPException(status_code=400, detail="event is not an LLM call")
    if event.get("template") != step:
        raise HTTPException(
            status_code=400,
            detail=f"event is a {event.get('template') or event.get('step')!r} call, not {step!r}",
        )
    if not isinstance(event.get("variables"), dict):
        raise HTTPException(
            status_code=409,
            detail="event predates variable logging — re-run the cell to make it testable",
        )
    return slot_log, event


# --- decision inquiry -------------------------------------------------------
#
# "Why did the model do this?" — a persistent conversation answered by a fixed
# strong reviewer so analysis quality is constant across whatever subject model
# is being benchmarked. The reviewer's full prompt (analyst framing + the step's
# exact system / input / output / reasoning) is assembled and shown CLIENT-SIDE
# in an editable, prefilled box, so the user sees byte-for-byte what gets sent;
# this endpoint just pins the reviewer model and forwards. The reviewer is
# Claude Opus 4.8 at xhigh reasoning regardless of which model ran the step.
INQUIRY_MODEL = "anthropic/claude-opus-4.8"


# --- prompt inspector -------------------------------------------------------
#
# Reconstructs the prompts a run actually used straight from its event logs.
# (Runs launched after the snapshot-first change pin + bind their effective
# prompt module at first launch, so their snapshots ARE authoritative — but
# legacy runs either lack snapshots or were never bound to them, so every
# `cache.llm` event carrying the exact `system` + `user` text per `step`
# remains the universal ground truth.)

# Canonical pipeline-step display order; unlisted steps sort after these.
_PROMPT_STEP_ORDER = [
    "zone_plan",
    "overall_bbox",
    "zone_decompose",
    "child_bbox_batch",
    "encapsulating_decompose",
    "anchor_decompose",
    "object_bbox_batch",
    "next_object",
    "negative_space_decompose",
    "image_prompt",
    "library_match",
]

# Per-cell extraction cache: events.jsonl path -> (mtime, {step: {...}}). A
# single cell log can be 100MB+, so we scan it at most once per modification.
_CELL_PROMPT_CACHE: dict[Path, tuple[float, dict[str, dict]]] = {}

_PROMPT_TOKEN_RE = re.compile(r"\s+|\S+")
# The top-level `"step"` key is logged BEFORE the megabyte system/user/output
# fields, so the first match in a line's head is always the real step. Lets us
# decide whether to skip a line without a full json.loads of its giant payload.
_STEP_PEEK_RE = re.compile(r'"step":\s*"([a-z_]+)"')
# Fully parse only the first few calls of each step per cell. The system prompt
# is a module constant (identical across a step's calls; the sole exception is
# zone_plan's root-vs-non-root pair, both seen within the first 2 calls), and
# the representative user is the FIRST call — so calls beyond this add nothing
# but parse cost. Hundreds of repeated object_bbox_batch/image_prompt/
# library_match payloads are skipped after the cap.
_STEP_PARSE_CAP = 4


def _extract_cell_prompts(events_path: Path) -> dict[str, dict]:
    """Reconstruct one cell's prompts from its `cache.llm` events.

    Returns `{step: {"systems": {system_text: call_count}, "first_user": str,
    "first_node": str | None, "first_index": int}}`. The `first_*` fields are
    the EARLIEST call of that step (lowest event index) — a deterministic
    representative for the (context-bearing) user prompt. Cached by mtime."""
    try:
        st = events_path.stat()
    except OSError:
        return {}
    cached = _CELL_PROMPT_CACHE.get(events_path)
    if cached is not None and cached[0] == st.st_mtime:
        return cached[1]
    out: dict[str, dict] = {}
    parsed_per_step: dict[str, int] = {}
    try:
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                # cache.llm lines are the only ones carrying prompts; the
                # substring pre-filter skips the cheap (bbox/image/model/...)
                # lines outright. Correctness is still gated on the parsed
                # `kind` below, so a false-positive is harmless.
                if '"kind": "cache.llm"' not in line:
                    continue
                # Peek at the step near the line head; once we've parsed the
                # cap for that step, skip the (huge) payload without decoding.
                peek = _STEP_PEEK_RE.search(line, 0, 400)
                if peek is not None and parsed_per_step.get(peek.group(1), 0) >= _STEP_PARSE_CAP:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("kind") != "cache.llm":
                    continue
                step = e.get("step")
                if not isinstance(step, str):
                    continue
                parsed_per_step[step] = parsed_per_step.get(step, 0) + 1
                system = e.get("system") if isinstance(e.get("system"), str) else ""
                user = e.get("user") if isinstance(e.get("user"), str) else ""
                idx = e.get("index")
                idx = idx if isinstance(idx, int) else (1 << 62)
                slot = out.setdefault(
                    step,
                    {"systems": {}, "first_user": "", "first_node": None, "first_index": 1 << 62},
                )
                slot["systems"][system] = slot["systems"].get(system, 0) + 1
                if idx < slot["first_index"]:
                    slot["first_index"] = idx
                    slot["first_user"] = user
                    slot["first_node"] = e.get("node")
    except OSError:
        return {}
    _CELL_PROMPT_CACHE[events_path] = (st.st_mtime, out)
    return out


def _run_cells_with_data(run: str) -> list[tuple[str, Path, float]]:
    """`(cell_key, events_path, created)` for every cell under this run whose
    log is non-empty. `cell_key` is `"<slot>/<model>"` — stable across runs so
    the diff can pick a reference cell present in both runs. `created` is the
    events.jsonl CREATION time (st_birthtime; falls back to mtime on the rare
    platform without it) — i.e. when that cell was first launched."""
    run_dir = _run_dir(run)
    cells: list[tuple[str, Path, float]] = []
    if not run_dir.is_dir():
        return cells
    for ev in sorted(run_dir.glob("*/*/events.jsonl")):
        try:
            st = ev.stat()
        except OSError:
            continue
        if st.st_size <= 0:
            continue
        rel = ev.relative_to(run_dir).parts  # (slot, model, "events.jsonl")
        if len(rel) >= 2:
            created = getattr(st, "st_birthtime", st.st_mtime)
            cells.append((f"{rel[0]}/{rel[1]}", ev, created))
    return cells


def _run_first_launch(run: str) -> float | None:
    """Canonical chronological date for a run: its OLDEST SLOT's first-launch.

    Keyed on each cell's CREATION time (st_birthtime), NOT mtime. A cell's
    events.jsonl is created the moment that cell is first launched, and
    creation time never moves afterward — so it is immune to the later writes
    (re-runs, resumes, rebakes, manual log edits, folder copies/renames) that
    make mtime and folder timestamps drift. A slot (scene) is first launched at
    its earliest cell creation; the run's date is the oldest such slot. None
    when the run has no rendition yet."""
    slot_dates: dict[str, float] = {}
    for cell_key, _ev, created in _run_cells_with_data(run):
        slot = cell_key.split("/", 1)[0]
        slot_dates[slot] = min(slot_dates.get(slot, created), created)
    return min(slot_dates.values()) if slot_dates else None


def _extract_run_prompts(run: str, ref_cell_key: str | None = None) -> dict[str, dict]:
    """Union the per-cell extractions for a run into
    `{step: {"systems": {text: count}, "by_cell": {cell_key: {"user", "node"}}}}`.

    Cells are scanned with the reference cell (for diffable user prompts) FIRST,
    then smallest-first; scanning stops once the structural pipeline is covered
    (`zone_decompose` seen) and the reference cell has been read. Since a step's
    system prompt is identical across cells, this avoids reading every cell's
    multi-hundred-MB log just to re-collect the same per-step systems."""
    def _size(p: Path) -> int:
        try:
            return p.stat().st_size
        except OSError:
            return 0

    cells = _run_cells_with_data(run)
    # Reference cell first (so its user prompts are always captured for the
    # diff), then ascending size so the cheapest cell completes coverage.
    cells.sort(key=lambda c: (c[0] != ref_cell_key, _size(c[1])))

    merged: dict[str, dict] = {}
    seen_steps: set[str] = set()
    ref_scanned = ref_cell_key is None
    for cell_key, ev, _m in cells:
        for step, info in _extract_cell_prompts(ev).items():
            m = merged.setdefault(step, {"systems": {}, "by_cell": {}})
            for sys_text, cnt in info["systems"].items():
                m["systems"][sys_text] = m["systems"].get(sys_text, 0) + cnt
            m["by_cell"][cell_key] = {"user": info["first_user"], "node": info["first_node"]}
            seen_steps.add(step)
        if cell_key == ref_cell_key:
            ref_scanned = True
        # A cell that ran `zone_decompose` exercised the full divider+generation
        # pipeline, so every step's system is already captured. Stop there
        # (once the reference cell is in) instead of scanning the rest.
        if ref_scanned and "zone_decompose" in seen_steps:
            break
    return merged


def _dominant_system(systems: dict[str, int]) -> str:
    """The system prompt used by the most calls — a stable, meaningful diff
    target when a step has >1 variant (only `zone_plan`, root vs non-root)."""
    if not systems:
        return ""
    return max(systems.items(), key=lambda kv: (kv[1], kv[0]))[0]


# Cost guards: word-level diff is O(n·m), so a fully-rewritten 100KB user
# prompt (full scene context) would hang the request. We diff at LINE level
# first (few hundred entries) and refine only the small changed hunks to word
# level; anything past these caps degrades to a whole-block replace.
_DIFF_WORD_CAP = 2500
_DIFF_LINE_CAP = 8000


def _word_segments(old: str, new: str) -> list[dict[str, str]]:
    a = _PROMPT_TOKEN_RE.findall(old)
    b = _PROMPT_TOKEN_RE.findall(new)
    if len(a) > _DIFF_WORD_CAP or len(b) > _DIFF_WORD_CAP:
        out: list[dict[str, str]] = []
        if old:
            out.append({"op": "delete", "text": old})
        if new:
            out.append({"op": "insert", "text": new})
        return out
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append({"op": "equal", "text": "".join(a[i1:i2])})
        elif tag == "delete":
            out.append({"op": "delete", "text": "".join(a[i1:i2])})
        elif tag == "insert":
            out.append({"op": "insert", "text": "".join(b[j1:j2])})
        else:  # replace
            out.append({"op": "delete", "text": "".join(a[i1:i2])})
            out.append({"op": "insert", "text": "".join(b[j1:j2])})
    return out


def _prompt_diff_segments(old: str, new: str) -> list[dict[str, str]]:
    """Diff two prompts into ordered `{op, text}` segments (op ∈
    equal/insert/delete). Line-level structure with word-level refinement of
    changed hunks: precise highlighting of the exact words that changed inside a
    long prose block, while staying linear in the common (mostly-equal) case."""
    a = (old or "").splitlines(keepends=True)
    b = (new or "").splitlines(keepends=True)
    if len(a) > _DIFF_LINE_CAP or len(b) > _DIFF_LINE_CAP:
        return _word_segments(old or "", new or "")
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    segs: list[dict[str, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            segs.append({"op": "equal", "text": "".join(a[i1:i2])})
        elif tag == "delete":
            segs.append({"op": "delete", "text": "".join(a[i1:i2])})
        elif tag == "insert":
            segs.append({"op": "insert", "text": "".join(b[j1:j2])})
        else:  # replace — refine the (usually small) changed block to words
            segs.extend(_word_segments("".join(a[i1:i2]), "".join(b[j1:j2])))
    return segs


def _pick_ref_cell(cur_cells: dict, prev_cells: dict) -> tuple[str | None, bool]:
    """Choose the reference cell for a step's representative user prompt:
    the lexicographically-first cell present in BOTH runs (so the user diff
    compares the same scene+model+call), else the first current-run cell
    (display only, no diff)."""
    shared = sorted(set(cur_cells) & set(prev_cells))
    if shared:
        return shared[0], True
    cur_only = sorted(cur_cells)
    return (cur_only[0], False) if cur_only else (None, False)


def _build_run_prompts(run: str, compare: str | None) -> dict[str, object]:
    """Assemble the prompt-inspector payload for `run`, diffed against
    `compare` (the chronologically-previous run) when given."""
    # Pick the shared reference cell up front (from directory listing alone, no
    # scanning) so both runs extract the SAME cell's user prompts — keeping the
    # representative user diff apples-to-apples.
    cur_keys = {k for k, _ev, _m in _run_cells_with_data(run)}
    prev_keys = {k for k, _ev, _m in _run_cells_with_data(compare)} if compare else set()
    shared = sorted(cur_keys & prev_keys)
    ref_cell = shared[0] if shared else None

    cur = _extract_run_prompts(run, ref_cell)
    prev = _extract_run_prompts(compare, ref_cell) if compare else {}
    steps_out: list[dict[str, object]] = []
    order = {s: i for i, s in enumerate(_PROMPT_STEP_ORDER)}
    for step in sorted(cur, key=lambda s: (order.get(s, len(order)), s)):
        cur_info = cur[step]
        prev_info = prev.get(step)
        cur_sys = _dominant_system(cur_info["systems"])
        prev_sys = _dominant_system(prev_info["systems"]) if prev_info else None
        sys_changed = prev_sys is not None and prev_sys != cur_sys
        system_block: dict[str, object] = {
            "text": cur_sys,
            "variant_count": len(cur_info["systems"]),
            "changed": sys_changed,
            "diff": _prompt_diff_segments(prev_sys or "", cur_sys) if sys_changed else None,
        }
        user_block: dict[str, object] | None = None
        ref_cell, shared = _pick_ref_cell(
            cur_info["by_cell"], prev_info["by_cell"] if prev_info else {}
        )
        if ref_cell is not None:
            cu = cur_info["by_cell"][ref_cell]
            node = cu["node"]
            cur_user = cu["user"]
            user_changed = False
            diff = None
            if shared and prev_info is not None:
                pu = prev_info["by_cell"][ref_cell]["user"]
                if pu != cur_user:
                    user_changed = True
                    diff = _prompt_diff_segments(pu, cur_user)
            user_block = {
                "text": cur_user,
                "ref": ref_cell + (f"  ·  node={node}" if node else ""),
                # The first call of a root-anchored step is node="root", whose
                # user prompt has no upstream context — so its diff is template-
                # only. Leaf steps carry scene context, so flag the diff as mixed.
                "root_anchored": node == "root",
                "diffable": shared,
                "changed": user_changed,
                "diff": diff,
            }
        steps_out.append({"step": step, "system": system_block, "user": user_block})
    return {"run": run, "compare": compare, "steps": steps_out}


def _hydrate_run(run: str) -> None:
    """Build SlotLogs for every (slot, model) cell under RUNS_DIR/<run>/.
    Idempotent — calling twice is a no-op. Hydration is per-run so the
    /runs/ parent can hold many old run sets without paying their startup
    cost until the user clicks one."""
    if run in _hydrated_runs:
        return
    run_dir = _run_dir(run)
    run_dir.mkdir(parents=True, exist_ok=True)
    for slot in SLOTS:
        for alias in MODEL_ALIASES:
            slot_dir = _slot_dir(run, slot.id, alias)
            slot_dir.mkdir(parents=True, exist_ok=True)
            slot_log = SlotLog(_run_id(run, slot.id, alias), slot_dir / "events.jsonl")
            slot_log.hydrate_from_disk()
            _slot_logs[(run, slot.id, alias)] = slot_log
            # Restore stepped mode from its on-disk marker so a stepped run
            # comes back steppable after a restart instead of a dead paused cell.
            if (slot_dir / ".stepped").exists():
                _stepped_cells.add((run, slot.id, alias))
            _maybe_launch(slot, alias, slot_log)
    _hydrated_runs.add(run)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        global _current_run
        # Open on the most recently touched run (if any); other runs hydrate
        # lazily on activation. Nothing is seeded — runs only exist when the
        # user creates them.
        run_dirs = sorted(
            (p for p in RUNS_DIR.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if run_dirs:
            _current_run = run_dirs[0].name
            _hydrate_run(_current_run)
        try:
            yield
        finally:
            rlog.suppress_console()
            for slot_log in _slot_logs.values():
                slot_log.close()
            for run_name, slot_id, model_alias in list(_slot_logs.keys()):
                generation.cancel_pending(_run_id(run_name, slot_id, model_alias))
            for task in _tasks.values():
                task.cancel()
            for task in _retry_tasks:
                task.cancel()
            for task in _branch_tasks.values():
                task.cancel()
            for task in list(_tasks.values()):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for task in list(_retry_tasks):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for task in list(_branch_tasks.values()):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            for branch_log in _branch_logs.values():
                branch_log.close()
            await oneshot.shutdown()
            await threed.disconnect_http()

    app = FastAPI(
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # The experimental one-shot track (single-call scene design, own slots/
    # models/prompt) — fully isolated under /oneshot; see app/oneshot/.
    app.include_router(oneshot.router)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    @app.get("/artifacts/{artifact_path:path}")
    async def artifact(artifact_path: str) -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        # Per-cell artifacts (events.jsonl, reference PNGs, GLBs). The URLs are
        # baked into the event log as `<cell>/objects/<id>.<ext>`. Cells migrated
        # onto the optimized library (rebake_runs.py --prune-originals) hold those
        # files under `objects-optimized/` instead, with `objects/` deleted — so a
        # miss there transparently falls back to the optimized twin.
        base = RUNS_DIR.resolve()
        target = (base / artifact_path).resolve()
        if not target.is_relative_to(base):
            raise HTTPException(status_code=404)
        rooted = f"/{artifact_path}"
        if not target.is_file() and "/objects/" in rooted:
            alt_rel = rooted.replace("/objects/", "/objects-optimized/", 1).lstrip("/")
            alt = (base / alt_rel).resolve()
            if alt.is_relative_to(base) and alt.is_file():
                target = alt
        if not target.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(target, media_type=_ARTIFACT_MEDIA_TYPES.get(target.suffix.lower()))

    @app.get("/runs")
    async def list_runs() -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        items: list[dict[str, object]] = []
        if RUNS_DIR.exists():
            for p in sorted(
                (p for p in RUNS_DIR.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ):
                meta = _run_meta(p.name)
                pv = meta.get("prompt_version")
                items.append(
                    {
                        "name": p.name,
                        "modified_at": p.stat().st_mtime,
                        # null for legacy runs (loadable, not resumable).
                        "prompt_version": pv if isinstance(pv, str) else None,
                    }
                )
        return {"runs": items, "current": _current_run}

    @app.post("/runs")
    async def create_run(req: CreateRunRequest) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        name = req.name.strip()
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(status_code=400, detail="invalid run name")
        version = req.prompt_version.strip()
        if not prompt_store.version_exists(version):
            raise HTTPException(status_code=404, detail=f"unknown prompt version: {version}")
        try:
            # Fail before any directory exists if the source set is incomplete.
            prompt_store.validate_version(version)
        except prompt_store.PromptTemplateError as e:
            raise HTTPException(status_code=409, detail=str(e))
        run_dir = _run_dir(name)
        if run_dir.exists():
            raise HTTPException(status_code=409, detail=f"run already exists: {name}")
        run_dir.mkdir(parents=True)
        prompt_store.snapshot_into_run(version, run_dir)
        (run_dir / RUN_META_NAME).write_text(
            json.dumps(
                {
                    "prompt_version": version,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        _hydrate_run(name)
        global _current_run
        _current_run = name
        return {"current": name}

    @app.post("/runs/snapshot")
    async def snapshot_run() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        src = _run_dir(_current_run)
        if not src.is_dir():
            raise HTTPException(status_code=404, detail=f"current run not found: {_current_run}")
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        name = f"save_{ts}"
        dst = _run_dir(name)
        if dst.exists():
            raise HTTPException(status_code=409, detail=f"snapshot already exists: {name}")
        # The copy carries the run's prompts/ snapshot + run.json, so the
        # archive replays and resumes with the source's exact prompts.
        shutil.copytree(src, dst)
        return {"snapshot": name}

    @app.post("/runs/{name}/activate")
    async def activate_run(name: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        run_dir = _run_dir(name)
        if not run_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"unknown run: {name}")
        _hydrate_run(name)
        global _current_run
        _current_run = name
        return {"current": name}

    @app.get("/prompt-runs")
    async def prompt_runs() -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Runs that have a rendition, ordered chronologically by when each was
        first launched (the OLDEST events.jsonl mtime among its cells). The
        prompt inspector diffs each run against its immediate predecessor here."""
        def _collect() -> list[dict[str, object]]:
            items: list[dict[str, object]] = []
            if RUNS_DIR.exists():
                for p in RUNS_DIR.iterdir():
                    if not p.is_dir():
                        continue
                    cells = _run_cells_with_data(p.name)
                    if not cells:
                        continue  # never launched — nothing to inspect
                    slots = {k.split("/", 1)[0] for k, _ev, _m in cells}
                    items.append(
                        {
                            "name": p.name,
                            "launched_at": _run_first_launch(p.name),
                            "n_slots": len(slots),
                            "n_cells": len(cells),
                        }
                    )
            items.sort(key=lambda d: (d["launched_at"], d["name"]))
            return items

        runs = await asyncio.to_thread(_collect)
        return {"runs": runs, "current": _current_run}

    @app.get("/runs/{run}/prompts")
    async def run_prompts(  # pyright: ignore[reportUnusedFunction]
        run: str, compare: str | None = None
    ) -> dict[str, object]:
        """Per-step prompts a run actually used, reconstructed from its event
        logs, with a word-level diff against `compare` when supplied."""
        if not _run_dir(run).is_dir():
            raise HTTPException(status_code=404, detail=f"unknown run: {run}")
        return await asyncio.to_thread(_build_run_prompts, run, compare)

    @app.post("/prompt-test")
    async def prompt_test(  # pyright: ignore[reportUnusedFunction]
        req: PromptTestRequest,
    ) -> dict[str, object]:
        """Re-run ONE historical step call with edited templates — the engine
        behind the prompt lab's "test on selected events". The target event's
        logged `variables` are substituted into the edited templates and the
        call re-issued on the event's own model via `llm.call_llm_once`, which
        neither reads the LLM cache nor writes a `cache.llm` event. The result
        is rendered transiently in the client and discarded; nothing about any
        run is mutated."""
        slot_log, event = _require_step_event(
            req.run, req.slot, req.model, req.event_index, req.step,
        )
        variables = event["variables"]
        model_id = str(event.get("model") or slot_log.state.get("model") or "")
        schema_cls = getattr(schemas, str(event.get("schema")), None)
        if not (isinstance(schema_cls, type) and issubclass(schema_cls, BaseModel)):
            raise HTTPException(
                status_code=409,
                detail=f"event's output schema is unknown: {event.get('schema')!r}",
            )
        try:
            system = prompt_store.resolve(
                req.system_template, variables, where=f"{req.step}.system (edited)")
            user = prompt_store.resolve(
                req.user_template, variables, where=f"{req.step}.user (edited)")
        except prompt_store.PromptTemplateError as e:
            raise HTTPException(status_code=400, detail=str(e))
        user = llm.apply_model_quirks(user, model_id)
        llm.set_model(model_id)
        try:
            _validated, reasoning, usage, raw = await llm.call_llm_once(
                system=system,
                user=user,
                output_schema=schema_cls,
                model=model_id,
                log_retries=False,
            )
        except Exception as e:
            # Surface provider/parse failures as a clean 502 the lab can show
            # inline, rather than a 500 with a stack trace.
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
        return {
            # The exact bytes sent — what a branch seed must carry.
            "system": system,
            "user": user,
            "output": raw,
            "reasoning": reasoning,
            "schema": event.get("schema"),
            "model": model_id,
            "node": event.get("node"),
            "original_output": event.get("output"),
            "tokens_in": getattr(usage, "prompt_tokens", None),
            "tokens_out": getattr(usage, "completion_tokens", None),
        }

    @app.post("/inquire")
    async def inquire(req: InquiryRequest) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        """Ask the reviewer (Claude Opus 4.8, xhigh reasoning) the latest
        question in an inquiry thread, and keep asking — the client carries the
        conversation forward in `messages`, so the thread is persistent. `system`
        is the exact, client-assembled prompt (analyst framing + the step's
        grounding — shown and editable in the panel). Stateless: nothing about
        any run is read or mutated."""
        if not req.messages or req.messages[-1].role != "user":
            raise HTTPException(
                status_code=400,
                detail="messages must be non-empty and end with a user turn",
            )
        convo = [{"role": m.role, "content": m.content} for m in req.messages]
        try:
            answer, reasoning = await llm.chat(model=INQUIRY_MODEL, system=req.system, messages=convo)
        except Exception as e:
            # Provider/transport failure → clean 502 the panel shows inline.
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}")
        return {"answer": answer, "reasoning": reasoning, "model": INQUIRY_MODEL}

    @app.get("/runs/{run}/prompt-templates")
    async def run_prompt_templates(run: str) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """The run's prompt snapshot as editable templates, with each step's
        available variables — the prompt lab's editing source."""
        ps = _require_run_prompts(run)
        return {
            "run": run,
            "steps": [
                {
                    "step": step,
                    "system": ps.template(step, "system"),
                    "user": ps.template(step, "user"),
                    # Every variable is injectable into every template; the
                    # `native` subset is what this step backs with real values
                    # (the rest render empty/placeholder here).
                    "variables": prompt_store.ALL_VARIABLES,
                    "native": prompt_store.STEP_VARIABLES[step],
                }
                for step in prompt_store.STEPS
            ],
        }

    @app.get("/variable-samples")
    async def variable_samples() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        """Every prompt `{VARIABLE}` rendered against a fixed sample scene via
        the real `scene_context` injection functions — the prompt lab's hover
        preview, so each variable's actual injected shape (and any missing
        context / rendering bug) is visible without running a scene. Run-
        independent: the sample scene is static."""
        return scene_context.sample_variables()

    @app.get("/runs/{run}/step-events")
    async def run_step_events(run: str, step: str) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Every logged call of template `step` across the run's cells that is
        re-renderable (carries logged `variables`) — the prompt lab's
        candidate list for testing an edit."""
        if step not in prompt_store.STEPS:
            raise HTTPException(status_code=404, detail=f"unknown step: {step}")
        if not _run_dir(run).is_dir():
            raise HTTPException(status_code=404, detail=f"unknown run: {run}")
        _hydrate_run(run)
        items: list[dict[str, object]] = []
        for (r, slot_id, alias), slot_log in _slot_logs.items():
            if r != run:
                continue
            for e in slot_log.state["events"]:
                if e.get("kind") != "cache.llm" or e.get("template") != step:
                    continue
                if not isinstance(e.get("variables"), dict):
                    continue
                output = json.dumps(e.get("output"), ensure_ascii=False)
                items.append(
                    {
                        "slot": slot_id,
                        "model": alias,
                        "index": e.get("index"),
                        "node": e.get("node"),
                        "model_id": e.get("model"),
                        "tokens_out": e.get("tokens_out"),
                        "output_preview": output[:240] + ("…" if len(output) > 240 else ""),
                        "branch_live": (run, slot_id, alias) in _branch_logs,
                    }
                )
        items.sort(key=lambda d: (str(d["slot"]), str(d["model"]), int(d["index"] or 0)))
        return {"run": run, "step": step, "events": items}

    @app.get("/runs/{run}/step-event")
    async def run_step_event(  # pyright: ignore[reportUnusedFunction]
        run: str, slot: str, model: str, index: int, step: str,
    ) -> dict[str, object]:
        """One logged call's FULL bytes (exact system/user sent, output +
        reasoning received) — fetched on demand so the prompt lab's review
        canvas can show every event's input/output without bloating the
        candidate-list poll."""
        _, event = _require_step_event(run, slot, model, index, step)
        return {
            "slot": slot,
            "model": model,
            "index": index,
            "step": step,
            "node": event.get("node"),
            "model_id": event.get("model"),
            "system": event.get("system"),
            "user": event.get("user"),
            "output": event.get("output"),
            "reasoning": event.get("reasoning"),
            "tokens_in": event.get("tokens_in"),
            "tokens_out": event.get("tokens_out"),
        }

    @app.get("/runs/{run}/branch-step-event")
    async def run_branch_step_event(  # pyright: ignore[reportUnusedFunction]
        run: str, slot: str, model: str, step: str, node: str | None = None,
    ) -> dict[str, object]:
        """The matching call's FULL bytes from the cell's simulation branch —
        the same template (and node, when given), so the lab can diff the live
        run's input/output against the simulated-edit branch. 404 when the cell
        has no live branch or no such call in it yet."""
        if step not in prompt_store.STEPS:
            raise HTTPException(status_code=404, detail=f"unknown step: {step}")
        blog = _branch_logs.get((run, slot, model))
        if blog is None:
            raise HTTPException(status_code=404, detail="no simulation branch for this cell")
        events = blog.state["events"]
        cands = [e for e in events if e.get("kind") == "cache.llm" and e.get("template") == step]
        # Match the SPECIFIC node when one is given — never fall back to another
        # node's call (e.g. a copied-prefix `root` decompose), which would diff a
        # stale prompt/output for the wrong region.
        if node is not None:
            cands = [e for e in cands if e.get("node") == node]
        # Surface the branch's ACTUAL output, not the pre-committed prompt-test
        # SEED. The seed is only a cache prime: when it hits, the branch runs
        # past this step using it (so it IS the output → show it); when it
        # misses, the branch pauses right before re-running this step (so the
        # seed is stale → report "not re-run yet" until the real call lands).
        # Real (non-seeded) calls always win, and the LATEST one is the final
        # validated attempt the pipeline actually used.
        real = [e for e in cands if not e.get("seeded")]
        if real:
            match = real[-1]
        elif cands and not _branch_awaiting(run, slot, model, step, node):
            match = cands[-1]  # seed cache-hit and consumed — it is the output
        else:
            match = None
        if match is None:
            where = f"{step} @ {node}" if node else step
            raise HTTPException(status_code=404, detail=f"the branch has not re-run {where} yet")
        return {
            "slot": slot,
            "model": model,
            "step": step,
            "index": match.get("index"),
            "node": match.get("node"),
            "model_id": match.get("model"),
            "system": match.get("system"),
            "user": match.get("user"),
            "output": match.get("output"),
            "reasoning": match.get("reasoning"),
            "tokens_in": match.get("tokens_in"),
            "tokens_out": match.get("tokens_out"),
        }

    @app.get("/slots")
    async def list_slots(run: str | None = None) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        return {
            "run": run,
            "models": MODEL_ALIASES,
            "default_model": DEFAULT_MODEL_ALIAS,
            "slots": [_slot_summary(s, run) for s in SLOTS],
        }

    @app.get("/trellis/queue")
    async def trellis_queue() -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Live snapshot of the process-global Trellis in-flight queue.
        The client polls this instead of inferring queue state from the
        replayed event log (historical submits without matching .done
        events leak as stale "processing" rows otherwise)."""
        return {
            "cap": threed.GENERATE_CONCURRENCY,
            "entries": threed.queue_snapshot(),
        }

    @app.get("/slots/{slot_id}/{model_alias}/events")
    async def slot_events(slot_id: str, model_alias: str, since: int = -1, run: str | None = None) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        # Subscribe and snapshot synchronously — no await between them, so no
        # log() call can land in both the snapshot and the live queue.
        q = slot_log.subscribe()
        snapshot = list(slot_log.state["events"])
        # `since` is the CQRS live-tail cut: the client already painted the
        # scene from /scene (folded up to `since`) and loads the full history
        # directly, so it only needs events *after* that index here. The gap
        # between the /scene read and this subscribe is covered because we
        # filter the fresh snapshot, not a stale copy. Default -1 = full
        # snapshot (legacy replay path, still used by reset/rewind).
        if since >= 0:
            snapshot = [
                e for e in snapshot
                if isinstance(e.get("index"), int) and e["index"] > since
            ]
        return StreamingResponse(
            _sse(slot_log, q, snapshot),
            media_type="text/event-stream",
        )

    @app.get("/slots/{slot_id}/{model_alias}/scene")
    async def slot_scene(slot_id: str, model_alias: str, run: str | None = None, until_index: int | None = None) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        # CQRS read model: fold the event log into the minimal renderable scene
        # state so the client paints directly instead of replaying ~2k events.
        # Returns `last_index` (the fold's cut) so the client can pick up the
        # live tail at `?since=last_index` with no gap or overlap.
        #
        # `until_index` projects only the prefix BEFORE that event — the compare
        # view's "previous" pane uses it to show the original run's state at the
        # branch's fork step (the shared baseline), not its full final scene.
        run = _resolve_run(run)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        events = list(slot_log.state["events"])
        if until_index is not None:
            events = events[:until_index]
        return _scene_projection(events)

    @app.get("/slots/{slot_id}/{model_alias}/meshes")
    async def slot_meshes(slot_id: str, model_alias: str, run: str | None = None, until_index: int | None = None) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        # One-request scene bundle: stream every finished GLB for this cell in
        # a single length-prefixed response so the client loads a whole scene
        # over one connection instead of one HTTP request per mesh (browsers
        # cap those at ~6 per origin, and they'd contend with the SSE stream
        # and polls on this same origin). The client keys each blob to its
        # `model` event by file stem; see `_mesh_bundle`.
        #
        # Always filtered to ids with a committed `model` event (so a cell that
        # was promoted from a branch, whose objects/ retains stale hardlinks,
        # never serves ghosts); `until_index` further restricts to meshes
        # committed BEFORE that event — paired with /scene?until_index for the
        # compare view's "previous" pane (the original's meshes at the fork).
        run = _resolve_run(run)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        cell_dir = slot_log.events_path.parent
        objects_dir = _objects_dir(cell_dir)
        events = list(slot_log.state["events"])
        if until_index is not None:
            events = events[:until_index]
        return StreamingResponse(
            _mesh_bundle(objects_dir, _committed_mesh_ids(events)),
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/slots/{slot_id}/{model_alias}/rewind")
    async def slot_rewind(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        req: RewindRequest,
        run: str | None = None,
    ) -> dict[str, int | str]:
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        if slot_log.state.get("prompt") is None or slot_log.state.get("model") is None:
            raise HTTPException(
                status_code=400,
                detail="slot has no run to rewind",
            )
        _require_run_prompts(run)
        # Reverting rewrites the source log; this cell's branch is now stale and
        # any in-flight task must stop before we rewrite under it.
        await _discard_branch((run, slot_id, model_alias))
        await _cancel_task(run, slot_id, model_alias)
        events = slot_log.state["events"]
        cut = max(0, min(req.to_event_index, len(events)))
        # Drop the meshes of every node generated at/after the cut so a later
        # re-run regenerates cleanly instead of silently reusing a stale glb.
        objs = slot_log.events_path.parent / "objects"
        for e in events[cut:]:
            oid = e.get("id")
            if isinstance(oid, str):
                for suffix in (".glb", ".raw.glb", ".png"):
                    with contextlib.suppress(OSError):
                        (objs / f"{oid}{suffix}").unlink()
        slot_log.truncate_events_to(cut)
        # Auto-resume from the cut instead of landing paused: a revert means
        # "re-run from here", so the reverted call (and everything downstream)
        # regenerates with no manual resume. Stepped cells get one free pass so
        # the reverted call itself re-executes before the gate pauses again
        # (mirrors rerun-step).
        key: RunKey = (run, slot_id, model_alias)
        if key in _stepped_cells:
            _gate_intents[key] = {"budget": 1}
        await _start_cell(run, slot_id, model_alias)
        return {"run": run, "slot_id": slot.id, "model": model_alias, "events": len(slot_log.state["events"])}

    @app.post("/slots/{slot_id}/{model_alias}/resume")
    async def slot_resume(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        run: str | None = None,
        stepped: bool | None = None,
    ) -> dict[str, str]:
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        # `stepped` opts the cell in/out of one-call-at-a-time execution
        # (persisted on disk); omitted = keep the cell's current mode.
        if stepped is not None:
            _set_stepped((run, slot_id, model_alias), stepped)
        # A completed run is terminal: resuming it would re-enter the pipeline
        # and generate a second run into the same cell. `run.done` is sticky
        # in the status derivation so the guard below normally catches this;
        # the explicit check encodes the rule directly (complete ⇒ reset-only)
        # and is robust to any stale in-memory status.
        if any(e.get("kind") == "run.done" for e in slot_log.state["events"]):
            raise HTTPException(
                status_code=409,
                detail="run is complete; reset to start a new run",
            )
        status = slot_log.state.get("status")
        if status not in ("idle", "paused", "error"):
            raise HTTPException(
                status_code=400,
                detail=f"slot is {status}, not startable",
            )
        await _start_cell(run, slot_id, model_alias)
        return {"run": run, "slot_id": slot.id, "model": model_alias}

    @app.post("/slots/{slot_id}/{model_alias}/pause")
    async def slot_pause(slot_id: str, model_alias: str, run: str | None = None) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        status = slot_log.state.get("status")
        if status != "running":
            raise HTTPException(
                status_code=400,
                detail=f"slot is {status}, not pausable",
            )
        await _cancel_task(run, slot_id, model_alias)
        # _cancel_task awaits the cancellation, so the pipeline task has
        # already torn down (including generation.cancel_pending via _run's
        # CancelledError branch) by the time we emit the sentinel.
        slot_log.state["status"] = "paused"
        slot_log.log("run.paused")
        return {"run": run, "slot_id": slot.id, "model": model_alias}

    @app.post("/slots/{slot_id}/{model_alias}/retry-mesh/{node_id}")
    async def slot_retry_mesh(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        node_id: str,
        run: str | None = None,
    ) -> dict[str, str]:
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        slot_log = _require_slot_log(run, slot_id, model_alias)
        node = _reconstruct_node(slot_log, node_id)
        if node is None:
            raise HTTPException(
                status_code=404,
                detail=f"no bbox event found for node: {node_id}",
            )

        async def _do_retry() -> None:
            rlog.bind(slot_log)
            llm.set_model(MODELS[model_alias])
            await generation.retry_node(
                node=node,
                runs_dir=RUNS_DIR,
                run_id=_run_id(run, slot.id, model_alias),
            )

        task = asyncio.create_task(_do_retry())
        _retry_tasks.add(task)
        task.add_done_callback(_retry_tasks.discard)
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
            "node_id": node_id,
        }

    @app.post("/slots/{slot_id}/{model_alias}/reset")
    async def slot_reset(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        run: str | None = None,
        start: bool = True,
    ) -> dict[str, str]:
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        if start:
            _require_run_prompts(run)
        # Reset wipes the whole cell dir (including any `_branch/`), so tear
        # this cell's branch down first to avoid it writing into a dir being
        # deleted.
        await _discard_branch((run, slot.id, model_alias))
        await _cancel_task(run, slot_id, model_alias)
        # Cancel any standalone retries (registered on generation._pending but
        # with no owning _run task to drive cleanup) that the running task
        # wouldn't have touched.
        generation.cancel_pending(_run_id(run, slot.id, model_alias))
        # A reset wipes the cell, including its stepped marker + intent.
        _set_stepped((run, slot.id, model_alias), False)
        _gate_intents.pop((run, slot.id, model_alias), None)
        slot_dir = _slot_dir(run, slot.id, model_alias)
        shutil.rmtree(slot_dir, ignore_errors=True)
        slot_dir.mkdir(parents=True, exist_ok=True)
        old_log = _slot_logs.get((run, slot.id, model_alias))
        if old_log is not None:
            old_log.close()
        slot_log = SlotLog(_run_id(run, slot.id, model_alias), slot_dir / "events.jsonl")
        _slot_logs[(run, slot.id, model_alias)] = slot_log
        if start:
            slot_log.start_run(slot.prompt, MODELS[model_alias])
            _tasks[(run, slot.id, model_alias)] = asyncio.create_task(_run(run, slot.id, model_alias))
        else:
            # Wipe back to a fresh idle cell — prompt + model pre-seeded (so a
            # later /resume can start_run without the client resending them) but
            # no pipeline task launched. Lets "reset all" clear an A/B matrix
            # without spawning dozens of runs; the user then starts exactly the
            # (slot, model) cells they pick.
            slot_log.state["prompt"] = slot.prompt
            slot_log.state["model"] = MODELS[model_alias]
            slot_log.state["status"] = "idle"
        return {"run": run, "slot_id": slot.id, "model": model_alias}

    @app.post("/slots/{slot_id}/{model_alias}/branch")
    async def create_branch(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        req: BranchRequest,
        run: str | None = None,
    ) -> dict[str, object]:
        """Fork the cell so it re-simulates from `event_index` onward with the
        edited step templates overriding the run snapshot. Replaces any prior
        branch on this cell. Isolated: writes only under `<cell>/_branch/`."""
        run = _resolve_run(run)
        slot = _require_slot(slot_id)
        _require_model(model_alias)
        _require_run_prompts(run)
        for ostep, roles in req.overrides.items():
            if ostep not in prompt_store.STEPS:
                raise HTTPException(status_code=400, detail=f"unknown override step: {ostep}")
            for role in roles:
                if role not in ("system", "user"):
                    raise HTTPException(status_code=400, detail=f"unknown template role: {role}")
        events = await _fork_branch(
            run, slot.id, model_alias,
            event_index=req.event_index, step=req.step,
            overrides=req.overrides, seed=req.seed,
        )
        return {
            "run": run,
            "slot_id": slot.id,
            "model": model_alias,
            "event_index": req.event_index,
            "events": events,
            "seeded": req.seed is not None,
        }

    @app.get("/slots/{slot_id}/{model_alias}/branch/scene")
    async def branch_scene(slot_id: str, model_alias: str, run: str | None = None) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        blog = _require_branch_log(run, slot_id, model_alias)
        return _scene_projection(list(blog.state["events"]))

    @app.get("/slots/{slot_id}/{model_alias}/branch/events")
    async def branch_events(slot_id: str, model_alias: str, since: int = -1, run: str | None = None) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        blog = _require_branch_log(run, slot_id, model_alias)
        q = blog.subscribe()
        snapshot = list(blog.state["events"])
        if since >= 0:
            snapshot = [
                e for e in snapshot
                if isinstance(e.get("index"), int) and e["index"] > since
            ]
        return StreamingResponse(_sse(blog, q, snapshot), media_type="text/event-stream")

    @app.get("/slots/{slot_id}/{model_alias}/branch/meshes")
    async def branch_meshes(slot_id: str, model_alias: str, run: str | None = None) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        blog = _require_branch_log(run, slot_id, model_alias)
        objects_dir = blog.events_path.parent / "objects"
        # Branch hardlinks the whole source objects/ dir, so serve only the
        # meshes the branch actually committed — never the dropped prefix ones.
        return StreamingResponse(
            _mesh_bundle(objects_dir, _committed_mesh_ids(list(blog.state["events"]))),
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/slots/{slot_id}/{model_alias}/branch/rewind")
    async def branch_rewind(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        req: BranchRewindRequest,
        run: str | None = None,
    ) -> dict[str, object]:
        """Revert a simulation branch to before `to_event_index` and pause there
        — the branch mirror of the source `/rewind`. Re-running forward (a plain
        `/branch/step`) then regenerates the reverted call under the CURRENT run
        snapshot + refreshed edits. Scoped to the branch log: the source cell and
        the branch's hardlinked prefix meshes are untouched."""
        run = _resolve_run(run)
        _require_slot(slot_id)
        _require_model(model_alias)
        _require_run_prompts(run)
        key: RunKey = (run, slot_id, model_alias)
        blog = _require_branch_log(run, slot_id, model_alias)
        if req.overrides is not None:
            for ostep, roles in req.overrides.items():
                if ostep not in prompt_store.STEPS:
                    raise HTTPException(status_code=400, detail=f"unknown override step: {ostep}")
                for role in roles:
                    if role not in ("system", "user"):
                        raise HTTPException(status_code=400, detail=f"unknown template role: {role}")
        # Stop the in-flight branch task but KEEP its log / overrides / dir, then
        # rewrite the log under it (mirrors source rewind's discard-then-truncate).
        await _cancel_branch_task(key)
        events = blog.state["events"]
        cut = max(0, min(req.to_event_index, len(events)))
        # Drop the meshes of every node generated at/after the cut so a re-run
        # regenerates cleanly. Prefix meshes (before the cut, hardlinked from the
        # source) stay; unlinking a post-cut hardlink only drops the branch's
        # name, never the source's inode.
        objs = blog.events_path.parent / "objects"
        for e in events[cut:]:
            oid = e.get("id")
            if isinstance(oid, str):
                for suffix in (".glb", ".raw.glb", ".png"):
                    with contextlib.suppress(OSError):
                        (objs / f"{oid}{suffix}").unlink()
        blog.truncate_events_to(cut)
        blog.state["status"] = "paused"
        # Refresh the edit set so the next step re-runs under the lab's CURRENT
        # edits (the sim's source of truth); None keeps the existing set.
        if req.overrides is not None:
            _branch_overrides[key] = {s: dict(r) for s, r in req.overrides.items()}
        return {"run": run, "slot_id": slot_id, "model": model_alias, "events": len(blog.state["events"])}

    @app.post("/slots/{slot_id}/{model_alias}/step")
    async def cell_step(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        req: BranchStepRequest,
        run: str | None = None,
    ) -> dict[str, object]:
        """Advance a stepped source cell by one LLM call — releasing a live
        gate OR relaunching a paused cell with a 1-call budget, so it works
        even after a restart. `auto` runs the cell to completion. 409 only
        when there's genuinely nothing to advance (done / mid-call / no cell)."""
        run = _resolve_run(run)
        _require_slot(slot_id)
        _require_model(model_alias)
        if req.until is not None and req.until not in prompt_store.STEPS:
            raise HTTPException(status_code=400, detail=f"unknown step: {req.until}")
        result = await _advance_cell(run, slot_id, model_alias, auto=req.auto, until=req.until)
        if result in ("missing", "done", "not_runnable", "in_flight"):
            raise HTTPException(status_code=409, detail=f"cannot step: {result}")
        return {"run": run, "slot_id": slot_id, "model": model_alias, "auto": req.auto, "result": result}

    @app.post("/runs/{run}/step-all")
    async def run_step_all(run: str, auto: bool = False, until: str | None = None) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Advance EVERY stepped cell in the run — regardless of whether it's
        paused at a live gate, mid-call, or sitting paused with no task (the
        "move the whole experiment forward" action). Each gets ONE queued step
        (so none is skipped for being mid-call), keeping the run in lockstep.
        `auto=true` runs them all to completion; `until=<step>` fast-forwards
        them all to the next call of that step."""
        if until is not None and until not in prompt_store.STEPS:
            raise HTTPException(status_code=400, detail=f"unknown step: {until}")
        results: dict[str, list[str]] = {}
        for r, slot_id, alias in [k for k in _stepped_cells if k[0] == run]:
            res = await _advance_cell(r, slot_id, alias, auto=auto, until=until)
            results.setdefault(res, []).append(f"{slot_id}/{alias}")
        advanced = results.get("stepped", []) + results.get("launched", []) + results.get("queued", []) + results.get("seeking", [])
        return {"run": run, "advanced": advanced, "auto": auto, "until": until, "by_result": results}

    @app.post("/slots/{slot_id}/{model_alias}/branch/step")
    async def branch_step(  # pyright: ignore[reportUnusedFunction]
        slot_id: str,
        model_alias: str,
        req: BranchStepRequest,
        run: str | None = None,
    ) -> dict[str, object]:
        """Advance a simulation branch by one LLM call. Like the source
        "step", this QUEUES a credit when the branch is mid-call (so a batch
        "step sims" never errors on a branch that isn't sitting at a gate),
        runs it to completion with `auto`, or fast-forwards to the next call of
        `until` and pauses there. 409 only when there's genuinely nothing to
        advance (done / no branch / overrides lost)."""
        run = _resolve_run(run)
        _require_branch_log(run, slot_id, model_alias)
        if req.until is not None and req.until not in prompt_store.STEPS:
            raise HTTPException(status_code=400, detail=f"unknown step: {req.until}")
        # `req.model` is a model ALIAS chosen in the compare UI; map to its
        # OpenRouter id so the next gated call re-aims at it (None = unchanged).
        if req.model is not None and req.model not in MODELS:
            raise HTTPException(status_code=400, detail=f"unknown model: {req.model}")
        model_id = MODELS[req.model] if req.model is not None else None
        result = await _advance_branch(run, slot_id, model_alias, auto=req.auto, until=req.until, model=model_id)
        if result in ("missing", "done", "not_runnable"):
            raise HTTPException(status_code=409, detail=f"cannot step: {result}")
        return {"run": run, "slot_id": slot_id, "model": model_alias, "auto": req.auto, "until": req.until, "result": result, "ran_model": req.model}

    @app.post("/slots/{slot_id}/{model_alias}/branch/pause")
    async def branch_pause(slot_id: str, model_alias: str, run: str | None = None) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        blog = _require_branch_log(run, slot_id, model_alias)
        key: RunKey = (run, slot_id, model_alias)
        if blog.state.get("status") != "running":
            raise HTTPException(
                status_code=400,
                detail=f"branch is {blog.state.get('status')}, not pausable",
            )
        await _cancel_branch_task(key)
        blog.state["status"] = "paused"
        blog.log("run.paused")
        return {"run": run, "slot_id": slot_id, "model": model_alias}

    @app.post("/slots/{slot_id}/{model_alias}/branch/resume")
    async def branch_resume(slot_id: str, model_alias: str, run: str | None = None) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        blog = _require_branch_log(run, slot_id, model_alias)
        key: RunKey = (run, slot_id, model_alias)
        if key not in _branch_overrides:
            raise HTTPException(
                status_code=409,
                detail="branch overrides are gone (server restarted) — re-create the branch",
            )
        status = blog.state.get("status")
        if status not in ("paused", "error"):
            raise HTTPException(status_code=400, detail=f"branch is {status}, not resumable")
        events = blog.state["events"]
        if events and events[-1].get("kind") in ("run.error", "run.paused"):
            blog.truncate_events_to(len(events) - 1)
        blog.state["status"] = "running"
        _branch_tasks[key] = asyncio.create_task(_run_branch(run, slot_id, model_alias))
        return {"run": run, "slot_id": slot_id, "model": model_alias}

    @app.delete("/slots/{slot_id}/{model_alias}/branch")
    async def discard_branch(slot_id: str, model_alias: str, run: str | None = None) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        run = _resolve_run(run)
        _require_slot(slot_id)
        _require_model(model_alias)
        # Idempotent — a missing branch is a no-op.
        await _discard_branch((run, slot_id, model_alias))
        return {"run": run, "slot_id": slot_id, "model": model_alias}

    @app.post("/versions/fork")
    async def fork_version(req: ForkVersionRequest) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        name = req.name.strip()
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(status_code=400, detail="invalid version name")
        if prompt_store.version_exists(name):
            raise HTTPException(status_code=409, detail=f"version already exists: {name}")
        if not prompt_store.version_exists(req.base):
            raise HTTPException(status_code=404, detail=f"unknown base version: {req.base}")
        try:
            prompt_store.fork_version(name, base=req.base)
        except prompt_store.PromptTemplateError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return {"name": name, "base": req.base}

    @app.put("/runs/{run}/prompt-templates")
    async def update_run_prompts(  # pyright: ignore[reportUnusedFunction]
        run: str, req: UpdateRunPromptsRequest,
    ) -> dict[str, object]:
        """Apply the prompt lab's edits to the run's snapshot in place — the
        fast iteration loop (edit → re-run step → compare) for a run that IS
        the working copy of a version being authored."""
        _require_run_prompts(run)
        # Allow a no-edit call when it's only syncing the version (push the run's
        # accumulated prompts back onto the source version without a fresh edit).
        if not req.overrides and not req.update_version:
            raise HTTPException(status_code=400, detail="no template edits to apply")
        run_snapshot = _run_dir(run) / prompt_store.RUN_PROMPTS_SUBDIR
        if req.overrides:
            try:
                prompt_store.write_overrides(run_snapshot, req.overrides)
            except prompt_store.PromptTemplateError as e:
                raise HTTPException(status_code=400, detail=str(e))
        version_synced: str | None = None
        if req.update_version:
            meta = _run_meta(run)
            version = meta.get("prompt_version")
            if isinstance(version, str) and prompt_store.version_exists(version):
                # HARD replacement: copy the run's FULL snapshot (every step) onto
                # the source version, so edits applied to the run earlier without
                # syncing land too — not just this call's overrides.
                prompt_store.sync_templates(prompt_store.VERSIONS_DIR / version, run_snapshot)
                version_synced = version
        return {
            "run": run,
            "applied": sorted(req.overrides),
            "version_synced": version_synced,
        }

    @app.post("/runs/{run}/simulate-step")
    async def simulate_step(  # pyright: ignore[reportUnusedFunction]
        run: str, req: SimulateStepRequest,
    ) -> dict[str, object]:
        """Fork a SIMULATION branch in every targeted cell at its EARLIEST
        call of any template in `steps`, running the pipeline from there under
        the run's (freshly edited) snapshot. Non-destructive: the source cells
        are untouched, so the original output stays for comparison — promote a
        branch with /branch/commit to replace its source when you're happy.
        Omitted `cells` means every cell that logged a re-renderable call of
        one of those steps."""
        if not req.steps:
            raise HTTPException(status_code=400, detail="no steps given")
        for s in req.steps:
            if s not in prompt_store.STEPS:
                raise HTTPException(status_code=404, detail=f"unknown step: {s}")
        if not _run_dir(run).is_dir():
            raise HTTPException(status_code=404, detail=f"unknown run: {run}")
        _require_run_prompts(run)
        _hydrate_run(run)
        wanted: set[tuple[str, str]] | None = None
        if req.cells is not None:
            wanted = {(str(c.get("slot", "")), str(c.get("model", ""))) for c in req.cells}
        steps = set(req.steps)
        simulated: list[str] = []
        skipped: list[str] = []
        for (r, slot_id, alias), slot_log in list(_slot_logs.items()):
            if r != run:
                continue
            if wanted is not None and (slot_id, alias) not in wanted:
                continue
            events = slot_log.state["events"]
            # Earliest re-renderable call of any target step — its ARRAY index
            # is the fork point (matches _fork_branch / _require_step_event).
            cut: int | None = None
            cut_step: str | None = None
            for i, e in enumerate(events):
                if (e.get("kind") == "cache.llm" and e.get("template") in steps
                        and isinstance(e.get("variables"), dict)):
                    cut = i
                    cut_step = str(e.get("template"))
                    break
            if cut is None or cut_step is None:
                skipped.append(f"{slot_id}/{alias}")
                continue
            # The run snapshot already carries the lab's edits (the apply step
            # writes them first), so the branch needs no per-call overrides.
            try:
                await _fork_branch(run, slot_id, alias, event_index=cut, step=cut_step, overrides={})
            except HTTPException:
                skipped.append(f"{slot_id}/{alias}")
                continue
            simulated.append(f"{slot_id}/{alias}")
        return {"run": run, "steps": sorted(steps), "simulated": simulated, "skipped": skipped}

    @app.post("/slots/{slot_id}/{model_alias}/branch/commit")
    async def commit_branch(slot_id: str, model_alias: str, run: str | None = None) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Promote a cell's simulation branch to BE the source cell: the
        branch's events + meshes replace the cell's own, and the branch is
        discarded. The deliberate, on-demand "replace source with the
        simulation" action — the inverse of the non-destructive fork. The
        prior source state is overwritten (gone)."""
        run = _resolve_run(run)
        _require_slot(slot_id)
        _require_model(model_alias)
        key: RunKey = (run, slot_id, model_alias)
        blog = _branch_logs.get(key)
        if blog is None:
            raise HTTPException(status_code=404, detail="no active branch for this cell")
        src_log = _slot_logs.get(key)
        if src_log is None:
            raise HTTPException(status_code=404, detail="no source cell to replace")
        # Stop both pipelines, then swap the branch's log + meshes into source.
        await _cancel_task(run, slot_id, model_alias)
        await _cancel_branch_task(key)  # keeps blog + dir for the copy below
        bdir = blog.events_path.parent
        cell_dir = src_log.events_path.parent
        src_objects = _objects_dir(cell_dir)
        src_log.close()
        await asyncio.to_thread(shutil.copyfile, bdir / "events.jsonl", src_log.events_path)
        # Move the branch's objects over the source's. The branch's prefix
        # files are hardlinks to the source inodes, so dropping the source dir
        # first is safe; slot_meshes filters to committed ids, so any stale
        # hardlinked-but-dropped files in the moved dir aren't served.
        branch_objects = bdir / "objects"

        def _swap_objects() -> None:
            shutil.rmtree(src_objects, ignore_errors=True)
            if branch_objects.is_dir():
                shutil.move(str(branch_objects), str(src_objects))
            else:
                src_objects.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(_swap_objects)
        src_log.hydrate_from_disk()  # reload source state from the promoted log
        await _discard_branch(key)   # remove the now-consumed branch dir + state
        emit_index = len(src_log.state["events"])
        return {"run": run, "slot_id": slot_id, "model": model_alias, "events": emit_index}

    @app.post("/versions/save")
    async def save_version(req: SaveVersionRequest) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        """Persist the prompt lab's edit as a new source version: the base
        run's snapshot plus the edited step templates."""
        name = req.name.strip()
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(status_code=400, detail="invalid version name")
        if prompt_store.version_exists(name):
            raise HTTPException(status_code=409, detail=f"version already exists: {name}")
        _require_run_prompts(req.base_run)
        try:
            prompt_store.save_version(
                name,
                base_dir=_run_dir(req.base_run) / prompt_store.RUN_PROMPTS_SUBDIR,
                overrides=req.overrides,
            )
        except prompt_store.PromptTemplateError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"name": name}

    @app.post("/runs/from-branches")
    async def save_run_from_branches(req: SaveRunRequest) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """Materialize the prompt lab's simulation as a NEW run: snapshot =
        base run's prompts + overrides, and each listed cell's branch state
        copied in as that cell's history (paused/finished alike — the new
        snapshot matches the prompts the branches actually ran with, so they
        resume cleanly)."""
        name = req.name.strip()
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(status_code=400, detail="invalid run name")
        run_dir = _run_dir(name)
        if run_dir.exists():
            raise HTTPException(status_code=409, detail=f"run already exists: {name}")
        _require_run_prompts(req.base_run)

        copied: list[str] = []
        skipped: list[str] = []
        run_dir.mkdir(parents=True)
        try:
            prompt_store.save_snapshot_with_overrides(
                base_dir=_run_dir(req.base_run) / prompt_store.RUN_PROMPTS_SUBDIR,
                dest=run_dir / prompt_store.RUN_PROMPTS_SUBDIR,
                overrides=req.overrides,
            )
        except prompt_store.PromptTemplateError as e:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(e))
        base_meta = _run_meta(req.base_run)
        label = req.version_label or f"{base_meta.get('prompt_version') or req.base_run}+edit"
        (run_dir / RUN_META_NAME).write_text(
            json.dumps(
                {
                    "prompt_version": label,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "branched_from": req.base_run,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        for cell in req.cells:
            slot_id = str(cell.get("slot", ""))
            alias = str(cell.get("model", ""))
            key: RunKey = (req.base_run, slot_id, alias)
            bdir = _branch_dir(req.base_run, slot_id, alias)
            if not (bdir / "events.jsonl").is_file():
                skipped.append(f"{slot_id}/{alias}")
                continue
            # Stop the live task first so the copied log is quiescent. The
            # branch itself stays on the source cell, untouched.
            await _cancel_branch_task(key)
            dest = run_dir / slot_id / alias
            dest.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(
                shutil.copyfile, bdir / "events.jsonl", dest / "events.jsonl",
            )
            await asyncio.to_thread(_hardlink_tree, bdir / "objects", dest / "objects")
            copied.append(f"{slot_id}/{alias}")
        _hydrate_run(name)
        global _current_run
        _current_run = name
        return {"current": name, "copied": copied, "skipped": skipped}

    @app.get("/versions")
    async def list_versions() -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        """The source prompt versions (subfolders of `versions/`) a new run
        can be created from."""
        return {"versions": [{"name": n} for n in prompt_store.list_versions()]}

    return app


def _last_step(events: list[dict[str, object]]) -> dict[str, object] | None:
    """The most recent pipeline-location marker — what the board cards show
    as "where is this cell right now"."""
    for e in reversed(events):
        if e.get("kind") == "step":
            return {"node": e.get("node"), "phase": e.get("phase")}
    return None


def _usage_summary(events: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    """Per-model token + request totals from a cell's `cache.llm` events, so the
    UI cost tracker can price the run (it applies the per-model USD rates client
    side). Returns `{ model_id: {"in": tokens_in, "out": tokens_out, "req": n} }`
    — a model with no logged usage still counts requests (cost shows as 0 for it
    until a rate exists), matching the old tracker's request-only degradation."""
    usage: dict[str, dict[str, int]] = {}
    for e in events:
        if e.get("kind") != "cache.llm":
            continue
        model = str(e.get("model") or "?")
        u = usage.setdefault(model, {"in": 0, "out": 0, "req": 0})
        ti, to = e.get("tokens_in"), e.get("tokens_out")
        u["in"] += int(ti) if isinstance(ti, (int, float)) else 0
        u["out"] += int(to) if isinstance(to, (int, float)) else 0
        u["req"] += 1
    return usage


def _slot_summary(slot: Slot, run: str) -> dict[str, object]:
    runs: dict[str, dict[str, object]] = {}
    for alias in MODEL_ALIASES:
        slot_log = _slot_logs.get((run, slot.id, alias))
        state = slot_log.state if slot_log is not None else {"status": "idle", "events": []}
        events = state.get("events", [])
        blog = _branch_logs.get((run, slot.id, alias))
        branch: dict[str, object] | None = None
        if blog is not None:
            bevents = blog.state.get("events", [])
            bgate = _branch_gates.get((run, slot.id, alias))
            branch = {
                "status": blog.state.get("status", "idle"),
                "events_count": len(bevents),
                "last_step": _last_step(bevents),
                # The LLM call waiting for the user's go-ahead, when gated.
                "pending": bgate.pending if bgate is not None else None,
                # The call currently in flight (released/running), so the UI
                # shows it instead of the stale last phase while it runs.
                "current": bgate.current if bgate is not None else None,
                "auto": bgate.auto if bgate is not None else False,
            }
        cgate = _cell_gates.get((run, slot.id, alias))
        runs[alias] = {
            "status": state.get("status", "idle"),
            "events_count": len(events),
            "last_kind": events[-1]["kind"] if events else None,
            "last_step": _last_step(events),
            "stepped": (run, slot.id, alias) in _stepped_cells,
            "pending": cgate.pending if cgate is not None else None,
            "current": cgate.current if cgate is not None else None,
            "auto": cgate.auto if cgate is not None else False,
            "branch": branch,
            # Per-model token/request totals for the cost tracker (the run's
            # actual spend = source cells; branch simulations aren't counted).
            "usage": _usage_summary(events),
        }
    return {
        "id": slot.id,
        "prompt": slot.prompt,
        "runs": runs,
    }


def _maybe_launch(slot: Slot, model_alias: str, slot_log: SlotLog) -> None:
    """Prepare each (slot, model) cell for manual start. Nothing auto-runs
    at boot — the user clicks start/resume/retry per cell from the viewer.
    Fresh cells sit idle with their seed prompt prefilled; previously-
    running cells come back as paused (resumable); errored cells come
    back as error (retry-able); completed cells stay done."""
    model_id = MODELS[model_alias]
    events = slot_log.state["events"]
    if not events:
        # Pre-seed the prompt + model so slot_resume can start_run() without
        # the client having to send them. status="idle" tells the viewer to
        # render a "start" button.
        slot_log.state["prompt"] = slot.prompt
        slot_log.state["model"] = model_id
        slot_log.state["status"] = "idle"
        return
    slot_log.state["model"] = model_id
    # hydrate_from_disk already derived the status from the full event log
    # (done/error/paused are sticky terminal states; a log with no terminal
    # marker reads as "running"). The only boot-time adjustment is that a
    # process killed mid-run leaves a "running" log with no sentinel —
    # surface that as paused so the user can resume it. A completed run stays
    # "done" (resume blocked, reset only) even when a post-run mesh retry
    # appended events after run.done; an errored run stays "error" (retry).
    if slot_log.state["status"] == "running":
        slot_log.state["status"] = "paused"


_WRAPPED_PROMPT_PREFIX = "Generate a direct,"


def _reconstruct_node(slot_log: SlotLog, node_id: str) -> Node | None:
    """Rebuild a minimal Node from the slot's event log so a standalone
    retry can call into `_generate_one` without re-running the divider.

    The bbox event carries everything the rescale step needs (origin,
    dimensions, proxy_shape, orientation). `image_prompt` is what banana
    actually sees, so we want to preserve whatever the prior run sent it
    rather than re-deriving from a "subject phrase" that may not exist
    anymore.

    Two log shapes coexist in practice:
      * NEW (current pipeline): `bbox.prompt` and `image.prompt` are bare
        subject phrases. The wrapping happens at the banana boundary.
      * OLD (pre-refactor logs still on disk): `bbox.prompt` and
        `image.prompt` are already the full wrapped studio-shot directive,
        because the pipeline wrapped earlier.

    Detect by sniffing the prefix. For a wrapped string we pass it through
    as `image_prompt` verbatim — re-wrapping would nest the template inside
    itself and produce a malformed banana prompt that Gemini refuses with
    MALFORMED_FUNCTION_CALL. For a bare subject we wrap normally.
    """
    events = slot_log.state["events"]
    bbox_event: dict[str, object] | None = None
    image_event: dict[str, object] | None = None
    for event in events:
        if event.get("id") != node_id:
            continue
        kind = event.get("kind")
        if kind == "bbox":
            bbox_event = event
        elif kind == "image":
            image_event = event
    if bbox_event is None:
        return None
    origin = bbox_event.get("origin")
    dimensions = bbox_event.get("dimensions")
    if not isinstance(origin, list) or not isinstance(dimensions, list):
        return None
    bbox = BoundingBox(
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
        dimensions=(float(dimensions[0]), float(dimensions[1]), float(dimensions[2])),
    )
    proxy_raw = bbox_event.get("proxy_shape")
    proxy_shape = ProxyShape(proxy_raw) if isinstance(proxy_raw, str) else None
    orientation_raw = bbox_event.get("orientation", 0)
    orientation: Orientation = int(orientation_raw) if isinstance(orientation_raw, (int, float, str)) else 0  # type: ignore[assignment]
    raw_prompt = image_event.get("prompt") if image_event is not None else bbox_event.get("prompt")
    raw_str = str(raw_prompt) if raw_prompt is not None else ""
    if raw_str.lstrip().startswith(_WRAPPED_PROMPT_PREFIX):
        image_prompt = raw_str
        subject_str = raw_str
    else:
        subject_str = raw_str
        image_prompt = scene_context.wrap_image_prompt(subject_str, proxy_shape, bbox.size)
    parent_id = bbox_event.get("parent_id")
    return Node(
        id=node_id,
        prompt=subject_str,
        bbox=bbox,
        proxy_shape=proxy_shape,
        orientation=orientation,
        image_prompt=image_prompt,
        parent_id=str(parent_id) if isinstance(parent_id, str) else None,
    )


def _require_slot(slot_id: str) -> Slot:
    slot = SLOTS_BY_ID.get(slot_id)
    if slot is None:
        raise HTTPException(status_code=404, detail=f"unknown slot: {slot_id}")
    return slot


def _require_model(model_alias: str) -> None:
    if model_alias not in MODELS:
        raise HTTPException(status_code=404, detail=f"unknown model: {model_alias}")


def _require_slot_log(run: str, slot_id: str, model_alias: str) -> SlotLog:
    _require_slot(slot_id)
    _require_model(model_alias)
    log = _slot_logs.get((run, slot_id, model_alias))
    if log is None:
        raise HTTPException(
            status_code=404,
            detail=f"no run for run={run} slot={slot_id} model={model_alias}",
        )
    return log


async def _cancel_task(run: str, slot_id: str, model_alias: str) -> None:
    _cell_gates.pop((run, slot_id, model_alias), None)
    task = _tasks.pop((run, slot_id, model_alias), None)
    if task is None or task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


def _objects_dir(cell_dir: Path) -> Path:
    """The cell's mesh dir — `objects/`, or the migrated `objects-optimized/`."""
    primary = cell_dir / OBJECTS_SUBDIR
    if primary.is_dir():
        return primary
    return next(
        (cell_dir / d for d in ("objects-optimized", "objects") if (cell_dir / d).is_dir()),
        primary,
    )


async def _advance_cell(run: str, slot_id: str, model_alias: str, *, auto: bool, until: str | None = None) -> str:
    """Advance a stepped source cell. Three modes:
      * `auto`   — run it to completion, ungated.
      * `until`  — fast-forward to the next call of template `until`, pause there.
      * default  — advance by exactly one LLM call.

    Works from EITHER state, which is what keeps a multi-model run in sync:
      * live gate (task running, paused OR mid-call): release a current pause,
        else QUEUE a credit (budget++) the cell spends at its next gate — so a
        model still mid-call doesn't miss a "step (all)".
      * paused/idle/error with no task → relaunch the pipeline carrying the
        intent so it replays the committed prefix then stops appropriately.
    Returns a status string for the caller to report/aggregate."""
    key: RunKey = (run, slot_id, model_alias)
    gate = _cell_gates.get(key)
    if gate is not None:
        if auto:
            gate.auto = True
            gate.proceed(auto=True)  # release a current pause if any
            _set_stepped(key, False)  # "run rest" exits step mode
            return "stepped"
        if until is not None:
            if gate.pending and gate.pending.get("template") == until:
                return "at_target"  # already paused right at it
            gate.until_step = until
            gate.proceed()  # release a current pause to fast-forward (no-op if mid-call)
            return "seeking"
        if gate.proceed():
            return "stepped"  # was paused → released one call
        gate.budget += 1  # mid-call → queue the step for its next gate
        return "queued"
    # No live gate. (A non-stepped task could still be running; guard.)
    task = _tasks.get(key)
    if task is not None and not task.done():
        return "in_flight"
    slot_log = _slot_logs.get(key)
    if slot_log is None:
        return "missing"
    if any(e.get("kind") == "run.done" for e in slot_log.state["events"]):
        return "done"
    if slot_log.state.get("status") not in ("idle", "paused", "error"):
        return "not_runnable"
    if auto:
        _set_stepped(key, False)  # finish normally, ungated
    else:
        _set_stepped(key, True)
        _gate_intents[key] = {"until": until} if until is not None else {"budget": 1}
    await _start_cell(run, slot_id, model_alias)
    return "launched"


async def _advance_branch(run: str, slot_id: str, model_alias: str, *, auto: bool, until: str | None = None, model: str | None = None) -> str:
    """Advance a simulation branch — the branch mirror of `_advance_cell`, so
    "step sims" behaves like "step all": it queues rather than erroring when a
    branch isn't sitting at a live gate.
      * live gate: release a current pause, else QUEUE a credit (budget++) the
        branch spends at its next gate — so a branch mid-call isn't skipped.
      * `auto` runs it to completion; `until` fast-forwards to the next call of
        that template and pauses there.
      * `model` (an OpenRouter id) re-aims the next gated call at a chosen LLM
        (compare's per-step model A/B) — sticky on the gate until changed, so
        the same scene context can be tested against different models.
      * paused/error with no task → relaunch carrying the intent.
    Returns a status string for the caller to report/aggregate."""
    key: RunKey = (run, slot_id, model_alias)
    blog = _branch_logs.get(key)
    if blog is None:
        return "missing"
    gate = _branch_gates.get(key)
    task = _branch_tasks.get(key)
    if gate is not None and task is not None and not task.done():
        if model is not None:
            gate.model_override = model  # applies from the next gated call onward
        if auto:
            gate.auto = True
            gate.proceed(auto=True)
            return "stepped"
        if until is not None:
            if gate.pending and gate.pending.get("template") == until:
                return "at_target"
            gate.until_step = until
            gate.proceed()  # release a current pause to fast-forward (no-op if mid-call)
            return "seeking"
        if gate.proceed():
            return "stepped"
        gate.budget += 1  # mid-call → queue the step for its next gate
        return "queued"
    # No live gate. (A finished task leaves no gate; a paused branch was cancelled.)
    if any(e.get("kind") == "run.done" for e in blog.state["events"]):
        return "done"
    if key not in _branch_overrides:
        return "not_runnable"  # overrides gone (server restart) — re-create it
    if blog.state.get("status") not in ("paused", "error"):
        return "not_runnable"
    events = blog.state["events"]
    if events and events[-1].get("kind") in ("run.error", "run.paused"):
        blog.truncate_events_to(len(events) - 1)
    if auto:
        intent: dict[str, object] = {"auto": True}
    else:
        intent = {"until": until} if until is not None else {"budget": 1}
    if model is not None:
        intent["model"] = model  # seed the relaunched gate's override
    _branch_gate_intents[key] = intent
    blog.state["status"] = "running"
    _branch_tasks[key] = asyncio.create_task(_run_branch(run, slot_id, model_alias))
    return "launched"


def _hardlink_tree(src: Path, dst: Path, ids: set[str] | None = None) -> None:
    """Mirror the flat `objects/` dir (`<id>.glb`, `<id>.raw.glb`, `<id>.png`)
    from `src` into `dst` via hardlinks — instant and zero extra disk, so the
    branch's `_spawn_meshes` sees committed prefix meshes as already present
    (`path.exists()`) and skips re-billing them. When `ids` is given, ONLY files
    for those node ids are linked (the branch passes its PREFIX's committed mesh
    ids) — post-deviation objects are deliberately absent so the pipeline
    regenerates them fresh and the branch shares nothing with the source past
    the fork. Falls back to a copy where hardlinks aren't supported."""
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if not p.is_file():
            continue
        # File names are `<id>.glb` / `<id>.raw.glb` / `<id>.png`; the id is the
        # part before the first dot (node ids carry no dots).
        if ids is not None and p.name.split(".", 1)[0] not in ids:
            continue
        target = dst / p.name
        if target.exists():
            continue
        try:
            os.link(p, target)
        except OSError:
            with contextlib.suppress(OSError):
                shutil.copy2(p, target)


def _require_branch_log(run: str, slot_id: str, model_alias: str) -> SlotLog:
    """The cell's live branch. 404 when this cell has none."""
    _require_slot(slot_id)
    _require_model(model_alias)
    blog = _branch_logs.get((run, slot_id, model_alias))
    if blog is None:
        raise HTTPException(status_code=404, detail="no active branch for this cell")
    return blog


def _branch_awaiting(run: str, slot_id: str, model_alias: str, step: str, node: str | None) -> bool:
    """True when the branch is paused at its gate right before a (step[, node])
    call — i.e. a prompt-test seed for that call MISSED the cache and the real
    re-run hasn't happened yet, so the seed is stale, not the branch's output."""
    gate = _branch_gates.get((run, slot_id, model_alias))
    pending = gate.pending if gate is not None else None
    if not pending or pending.get("template") != step:
        return False
    return node is None or pending.get("node") == node


def cache_hash_for_seed(event: dict[str, object], seed: BranchSeed) -> str:
    """Cache key for a seeded prompt-test result: identical inputs to what the
    branch's re-reached call will hash, so the pipeline cache-hits the vetted
    output instead of re-rolling the edited step."""
    return cache.hash_llm_call(
        model=str(event.get("model") or ""),
        system=seed.system,
        user=seed.user,
        schema_name=str(event.get("schema") or ""),
    )


async def _fork_branch(
    run: str,
    slot_id: str,
    model_alias: str,
    *,
    event_index: int,
    step: str,
    overrides: dict[str, dict[str, str]],
    seed: BranchSeed | None = None,
) -> int:
    """Fork a cell into a simulation branch at `event_index` (a re-renderable
    cache.llm call of template `step`): keep the prefix, hardlink the source's
    finished meshes, and run the pipeline from there under the run snapshot +
    `overrides`. Replaces any prior branch. Returns the branch's starting event
    count. Shared by the lab's "simulate downstream" (create_branch) and
    apply-to-run's "simulate from step" (simulate_step) — both are NON-
    destructive: the source cell is never touched."""
    src_log, event = _require_step_event(run, slot_id, model_alias, event_index, step)
    src_events = list(src_log.state["events"])
    key: RunKey = (run, slot_id, model_alias)

    # One branch per cell: replace any prior fork of this cell.
    await _discard_branch(key)
    bdir = _branch_dir(run, slot_id, model_alias)
    shutil.rmtree(bdir, ignore_errors=True)
    bdir.mkdir(parents=True, exist_ok=True)

    # Hardlink ONLY the prefix's finished meshes (objects committed BEFORE the
    # fork) so the replayed prefix renders without regenerating. Nothing after
    # the deviation is linked: post-fork objects have no file here, so the
    # pipeline generates each one FRESH (library re-match + re-place, or a fresh
    # Trellis run), keeping the branch fully independent of the source past the
    # branch-off point.
    cell_dir = src_log.events_path.parent
    src_objects = cell_dir / OBJECTS_SUBDIR
    if not src_objects.is_dir():
        src_objects = next(
            (cell_dir / d for d in ("objects-optimized", "objects") if (cell_dir / d).is_dir()),
            cell_dir / "objects",
        )
    prefix_ids = _committed_mesh_ids(src_events[:event_index])
    await asyncio.to_thread(_hardlink_tree, src_objects, bdir / "objects", prefix_ids)

    # Prefix = source events BEFORE the edited step's call. Dropping that call
    # (and everything after) means `committed.*` replays the prefix and the
    # pipeline re-reaches the call with the (snapshot + overrides) templates.
    branch_events = [dict(e) for e in src_events[:event_index]]
    bevents = bdir / "events.jsonl"
    with bevents.open("w", encoding="utf-8") as f:
        for e in branch_events:
            f.write(json.dumps(e) + "\n")

    blog = SlotLog(_branch_run_id(run, slot_id, model_alias), bevents)
    blog.hydrate_from_disk()
    if blog.state.get("prompt") is None or blog.state.get("model") is None:
        raise HTTPException(status_code=400, detail="source has no run.start to branch from")
    if seed is not None:
        # Pre-commit the vetted prompt-test result so the re-reached call
        # cache-hits it and the simulation continues from EXACTLY the output
        # the user approved (a drifted render simply misses and re-runs fresh).
        blog.log(
            "cache.llm",
            key=cache_hash_for_seed(event, seed),
            node=event.get("node"),
            step=event.get("step"),
            template=step,
            model=event.get("model"),
            schema=event.get("schema"),
            system=seed.system,
            user=seed.user,
            variables=event.get("variables"),
            output=seed.output,
            reasoning=seed.reasoning,
            tokens_in=seed.tokens_in,
            tokens_out=seed.tokens_out,
            seeded=True,
        )
    _branch_logs[key] = blog
    _branch_overrides[key] = {s: dict(r) for s, r in overrides.items()}
    # Run the forked step IMMEDIATELY (one gate credit) so "simulate downstream"
    # shows the edited step's result on fork, with no manual first step. A seed
    # already pre-commits that result, so leave it paused right after it.
    if seed is None:
        _branch_gate_intents[key] = {"budget": 1}
    _branch_tasks[key] = asyncio.create_task(_run_branch(run, slot_id, model_alias))
    return len(blog.state["events"])


async def _cancel_branch_task(key: RunKey) -> None:
    """Cancel the cell branch's task + in-flight branch meshes + step gate,
    but KEEP its SlotLog, overrides, and dir — pause/resume relaunch on the
    same branch (a relaunch binds a fresh gate, back in stepping mode)."""
    task = _branch_tasks.pop(key, None)
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    _branch_gates.pop(key, None)
    _branch_gate_intents.pop(key, None)
    log = _branch_logs.get(key)
    if log is not None:
        generation.cancel_pending(log.slot_id)


async def _discard_branch(key: RunKey) -> None:
    """Full break-out: cancel the cell's branch and delete its directory."""
    await _cancel_branch_task(key)
    _branch_overrides.pop(key, None)
    log = _branch_logs.pop(key, None)
    if log is not None:
        log.close()
        shutil.rmtree(log.events_path.parent, ignore_errors=True)


async def _run_branch(run: str, slot_id: str, model_alias: str) -> None:
    """Drive one cell branch. A mirror of `_run` bound to the branch's SlotLog
    + branch run_id + the run snapshot WITH the lab's template overrides: the
    prefix replays via `committed.*`, the edited step's seeded result (if any)
    cache-hits, and the frontier re-runs under the edited templates."""
    key: RunKey = (run, slot_id, model_alias)
    blog = _branch_logs[key]
    rlog.bind(blog)
    # One LLM call at a time: bind the step gate IN this task's context so
    # only the branch pauses; the main pipeline tasks never block. A relaunch
    # (step / step-until / run-rest of a paused branch) carries its intent here.
    intent = _branch_gate_intents.pop(key, {})
    _b_model = intent.get("model")
    gate = CellGate(blog, budget=int(intent.get("budget", 0) or 0), model_override=str(_b_model) if _b_model else None)
    gate.auto = bool(intent.get("auto", False))
    _b_until = intent.get("until")
    gate.until_step = str(_b_until) if _b_until else None
    _branch_gates[key] = gate
    llm.set_step_gate(gate.wait)
    prompt = blog.state["prompt"]
    model = blog.state["model"]
    brun_id = blog.slot_id  # composite branch run_id (run/slot/model/_branch)
    try:
        base = prompt_store.load_run_prompts(_run_dir(run))
        prompt_store.bind(base.with_overrides(_branch_overrides.get(key, {})))
        await divider.run(run_id=brun_id, prompt=prompt, model=model, runs_dir=RUNS_DIR)
    except asyncio.CancelledError:
        generation.cancel_pending(brun_id)
        raise
    except Exception as e:
        generation.cancel_pending(brun_id)
        blog.log("run.error", message=f"{type(e).__name__}: {e}")
        return
    await generation.await_pending(brun_id)
    blog.finish_run()


async def _start_cell(run: str, slot_id: str, model_alias: str) -> None:
    """Cancel any in-flight task for this cell, then (re)start its pipeline.
    A fresh cell emits run.start; a paused/errored cell drops its terminal
    sentinel and resumes. Callers own any other precondition guards (e.g.
    blocking a completed run)."""
    slot = _require_slot(slot_id)
    slot_log = _require_slot_log(run, slot_id, model_alias)
    # The run's prompt snapshot is the only prompt source — without one
    # (legacy run) nothing can launch.
    _require_run_prompts(run)
    status = slot_log.state.get("status")
    await _cancel_task(run, slot_id, model_alias)
    events = slot_log.state["events"]
    model_id = MODELS[model_alias]
    if status == "idle" and not events:
        slot_log.start_run(slot.prompt, model_id)
    else:
        if events and events[-1].get("kind") in ("run.error", "run.paused"):
            slot_log.truncate_events_to(len(events) - 1)
        slot_log.state["model"] = model_id
        slot_log.state["status"] = "running"
    _tasks[(run, slot_id, model_alias)] = asyncio.create_task(_run(run, slot_id, model_alias))


_MESH_BUNDLE_MAGIC = b"SMB1"


def _committed_mesh_ids(events: list[dict[str, object]]) -> set[str]:
    """Node ids with a committed `model` event — the meshes that are actually
    part of the scene. A branch hardlinks the WHOLE source `objects/` dir up
    front (so the replayed prefix doesn't re-bill Trellis), so its bundle must
    exclude files for nodes the re-run dropped (no `model` event in the branch
    log) — otherwise those obsoleted source meshes render as ghosts the branch's
    bboxes/scene-projection don't include."""
    return {
        e["id"]  # type: ignore[misc]
        for e in events
        if e.get("kind") == "model" and isinstance(e.get("id"), str)
    }


async def _mesh_bundle(objects_dir: Path, ids: set[str] | None = None) -> AsyncIterator[bytes]:
    """Stream every finished GLB under `objects_dir` as one length-prefixed
    binary bundle, so a client can fetch an entire scene in a single request
    instead of one HTTP round-trip per mesh. Framing (little-endian):

        b"SMB1"
        repeat: <uint32 id_len><id utf-8><uint32 glb_len><glb bytes>

    The id is the file stem, which matches the `model` event's `id`. Files are
    read off the event loop via a thread so a large scene doesn't stall it. When
    `ids` is given, only those stems are streamed — the branch passes the set of
    its committed `model` ids so stale hardlinked prefix meshes aren't served.
    """
    yield _MESH_BUNDLE_MAGIC
    if not objects_dir.is_dir():
        return
    for path in sorted(objects_dir.glob("*.glb")):
        # Skip pre-processing intermediates (`<id>.raw.glb`); only the finished
        # `<id>.glb` is what the client renders.
        if path.name.endswith(".raw.glb"):
            continue
        node_id = path.name[: -len(".glb")]
        if ids is not None and node_id not in ids:
            continue  # not part of this scene (e.g. a branch's obsoleted object)
        data = await asyncio.to_thread(path.read_bytes)
        id_bytes = node_id.encode("utf-8")
        yield struct.pack("<I", len(id_bytes)) + id_bytes
        yield struct.pack("<I", len(data)) + data


def _scene_projection(events: list[dict[str, object]]) -> dict[str, object]:
    """Fold the event log into the minimal renderable scene state — the read
    model the client paints directly instead of replaying every event.

    Mirrors the *scene-building* cases of the client's `dispatch()` (bbox,
    divider.*, step, mesh.*, image, model). Deliberately ignores `cache.llm`
    and the other log/observability-only kinds — those stay in the full event
    log, which the client backfills into the side panels separately.

    Returns `{ "nodes": [...], "last_index": N }` where N is the highest event
    index folded, so the client can subscribe to the live tail at index > N.
    """
    nodes: dict[str, dict[str, object]] = {}
    last_index = -1

    def node(node_id: str) -> dict[str, object]:
        n = nodes.get(node_id)
        if n is None:
            n = {"id": node_id}
            nodes[node_id] = n
        return n

    for e in events:
        idx = e.get("index")
        if isinstance(idx, int):
            last_index = max(last_index, idx)
        kind = e.get("kind")
        nid = e.get("id")
        if kind == "bbox" and isinstance(nid, str):
            n = node(nid)
            n["parent_id"] = e.get("parent_id")
            n["prompt"] = e.get("prompt")
            n["node_kind"] = e.get("node_kind", "zone")
            n["origin"] = e.get("origin")
            n["dimensions"] = e.get("dimensions")
            n["proxy_shape"] = e.get("proxy_shape")
        elif kind in ("divider.decompose", "divider.zone_decompose"):
            children = e.get("children")
            if isinstance(children, list):
                for c in children:
                    if not isinstance(c, dict):
                        continue
                    cid = c.get("id")
                    if not isinstance(cid, str):
                        continue
                    n = node(cid)
                    n.setdefault("parent_id", c.get("parent") or e.get("node"))
                    n.setdefault("prompt", c.get("prompt"))
                    n.setdefault("node_kind", "zone")
        elif kind == "divider.zone_plan":
            anchor = e.get("node")
            if isinstance(anchor, str) and isinstance(e.get("plan"), str):
                node(anchor)["plan"] = e["plan"]
        elif kind == "step":
            anchor = e.get("node")
            if isinstance(anchor, str):
                node(anchor)["phase"] = e.get("phase")
        elif kind == "mesh.submit" and isinstance(nid, str):
            node(nid)["phase"] = "generating_mesh"
        elif kind == "image" and isinstance(nid, str):
            n = node(nid)
            n["image_url"] = e.get("url")
            if isinstance(e.get("prompt"), str):
                n["image_prompt"] = e.get("prompt")
        elif kind == "model" and isinstance(nid, str):
            n = node(nid)
            n["mesh_url"] = e.get("url")
            n["phase"] = "done"
            n.pop("error", None)
        elif kind == "mesh.error" and isinstance(nid, str):
            n = node(nid)
            n["phase"] = "error"
            n["error"] = e.get("message", "unknown error")
        elif kind == "mesh.retry" and isinstance(nid, str):
            n = node(nid)
            n["phase"] = "generating_mesh"
            n.pop("error", None)

    return {"nodes": list(nodes.values()), "last_index": last_index}


async def _sse(
    slot_log: SlotLog,
    q: asyncio.Queue[dict[str, object]],
    snapshot: list[dict[str, object]],
) -> AsyncIterator[str]:
    # All events ride the default SSE "message" type; the client dispatches
    # by `event.kind` internally. Keeps the client listener table flat.
    #
    # Snapshot terminal events do NOT close the stream — only live ones
    # do. That way a client re-subscribing to a finished slot (to drive a
    # standalone mesh retry) gets the historical timeline and then waits
    # on the live queue for the retry's new events, instead of being
    # disconnected the instant the past `run.done` replays.
    try:
        for event in snapshot:
            yield f"data: {json.dumps(event)}\n\n"
        while True:
            event = await q.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["kind"] in {"run.done", "run.error", "run.paused"}:
                return
    finally:
        slot_log.unsubscribe(q)


async def _run(run: str, slot_id: str, model_alias: str) -> None:
    key: RunKey = (run, slot_id, model_alias)
    slot_log = _slot_logs[key]
    rlog.bind(slot_log)
    # Stepped mode: gate this cell so it pauses before every frontier LLM
    # call (the per-step prompt-iteration workflow). The next-gate intent
    # (set by "step" / "run rest" / rerun-step) tunes the first launch.
    if key in _stepped_cells:
        intent = _gate_intents.pop(key, {})
        gate = CellGate(slot_log, budget=int(intent.get("budget", 0) or 0))
        gate.auto = bool(intent.get("auto", False))
        _until = intent.get("until")
        gate.until_step = str(_until) if _until else None
        _cell_gates[key] = gate
        llm.set_step_gate(gate.wait)
    prompt = slot_log.state["prompt"]
    model = slot_log.state["model"]
    run_id = _run_id(run, slot_id, model_alias)
    try:
        # Bind the run's prompt snapshot inside the try so a broken/missing
        # snapshot surfaces as a clean run.error instead of a dead task.
        prompt_store.bind(prompt_store.load_run_prompts(_run_dir(run)))
        await divider.run(
            run_id=run_id,
            prompt=prompt,
            model=model,
            runs_dir=RUNS_DIR,
        )
    except asyncio.CancelledError:
        generation.cancel_pending(run_id)
        raise
    except Exception as e:
        generation.cancel_pending(run_id)
        # OpenRouter SDK errors only stringify to the top-level "Provider
        # returned error" message; the actually useful detail (upstream
        # provider's complaint, the request body that tripped it) lives on
        # `data.error.metadata` and on `e.body` (the response body the SDK
        # already read; `raw_response.text` would re-trigger a read on a
        # closed streaming response). Pull both into the logged message so
        # the run.error event tells us what went wrong.
        details = []
        data = getattr(e, "data", None)
        err = getattr(data, "error", None) if data is not None else None
        if err is not None:
            metadata = getattr(err, "metadata", None)
            if metadata:
                details.append(f"metadata={metadata}")
        body = getattr(e, "body", None)
        if body:
            details.append(f"body={body[:2000]}")
        suffix = (" | " + " | ".join(details)) if details else ""
        slot_log.log("run.error", message=f"{type(e).__name__}: {e}{suffix}")
        return
    finally:
        # A gate only lives for the duration of its task — drop it so a paused
        # or finished cell reports no `pending`. (A live pause is still
        # awaiting inside divider.run, so this runs only once the task ends.)
        _cell_gates.pop(key, None)
    # Pipeline tree is fully resolved; meshes may still be in flight.
    # Hold the run open until they all land so `run.done` truly means done.
    await generation.await_pending(run_id)
    slot_log.finish_run()
