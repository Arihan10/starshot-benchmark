"""File-based prompt versioning.

`VERSIONS_DIR` holds the SOURCE prompt versions: each subfolder is one version
(the folder name is the version name) containing `<step>.system.txt` +
`<step>.user.txt` for every pipeline step (see versions/TEMPLATE.md).

A run never reads the sources after birth: creating a run copies the chosen
version folder into `runs/<run>/prompts/` (the run's SNAPSHOT), and every
pipeline task renders from that snapshot — so editing a source version later
never changes a started run's prompts, cache keys, or resumes.

Templates reference runtime scene state with variables written as a brace
token wrapped in backticks — backtick, `{UPPER_SNAKE_CASE}`, backtick —
resolved from the dicts built in `scene_context`. The backticks are consumed
by the substitution. Unknown variables fail the render loudly; anything that
is not exactly that shape (plain `{...}` braces, or backticked brace text with
a non-UPPER_SNAKE body such as the literal `{target, kind}`) passes through
untouched.

The active `PromptSet` is task-local (ContextVar), mirroring how the slot log
and LLM model are bound: each pipeline task binds its run's snapshot at entry.
"""

from __future__ import annotations

import os
import re
import shutil
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

# Repo root (this file is server/app/core/prompt_store.py). Source versions
# live in <repo>/versions next to <repo>/runs, regardless of the directory
# the server happens to be launched from.
_REPO_ROOT = Path(__file__).resolve().parents[3]

VERSIONS_DIR = Path(os.environ.get("STARSHOT_VERSIONS_DIR", _REPO_ROOT / "versions"))

# Subfolder of a run dir holding its prompt snapshot.
RUN_PROMPTS_SUBDIR = "prompts"

# Every step a version must provide, as `<step>.system.txt` + `<step>.user.txt`.
STEPS: list[str] = [
    "zone_plan_root",
    "zone_plan",
    "overall_bbox",
    "zone_decompose_root",
    "zone_decompose",
    "child_bbox_batch",
    "encapsulating_decompose",
    "anchor_decompose",
    "negative_space_decompose",
    "object_bbox_batch",
    "next_object",
    "image_prompt",
]

# The full variable vocabulary, in display order (scene-wide, target-zone,
# step-specific). EVERY render receives EVERY variable: ones whose backing
# state doesn't exist at that step resolve to "" (or the canonical
# empty-scene placeholder), so any variable can be injected into any
# template without failing the run.
ALL_VARIABLES: list[str] = [
    "ROOT_PROMPT", "ROOT_PLAN", "ROOT_DIMENSIONS", "ROOT_ORIGIN", "ROOT_HEADER",
    "ROOT_OBJECTS", "SCENE_CONTEXT",
    "ZONE_ID", "ZONE_PROMPT", "ZONE_PLAN", "ZONE_PLACEMENT", "ZONE_DIMENSIONS",
    "ZONE_ORIGIN", "ZONE_OBJECTS", "PARENT_ZONE_ID", "PARENT_ZONE_PLAN", "PARENT_ZONE_ORIGIN",
    "TO_PLACE", "RETRY_BLOCK", "ADJACENT_ZONES",
    "OBJECT_PROMPT", "OBJECT_DIMENSIONS", "PROXY_SHAPE", "IMAGE_TEMPLATE_FRONT",
    "IMAGE_TEMPLATE_SIDE", "IMAGE_TEMPLATE_TOP", "PRIOR_SUBJECTS",
]

# Tokens that USED to be variables. They resolve to "" so a run snapshot
# created before their removal keeps rendering; they are no longer offered
# in the editor or docs. DEEPSEEK_SUFFIX became a runtime injection at the
# LLM call boundary (see services/llm.apply_model_quirks).
LEGACY_VARIABLES: list[str] = ["DEEPSEEK_SUFFIX"]

# The variables each step NATIVELY populates with real values (see
# versions/TEMPLATE.md) — everything else renders empty/placeholder there.
# Drives the editor's chip styling and the docs; the runtime provides the
# full ALL_VARIABLES set everywhere.
_ZONE_VARIABLES = [
    "ROOT_PROMPT", "ROOT_PLAN", "ROOT_DIMENSIONS", "ROOT_ORIGIN", "ROOT_HEADER",
    "ROOT_OBJECTS", "SCENE_CONTEXT", "ZONE_ID", "ZONE_PROMPT", "ZONE_PLAN",
    "ZONE_PLACEMENT", "ZONE_DIMENSIONS", "ZONE_ORIGIN", "ZONE_OBJECTS", "PARENT_ZONE_ID",
    "PARENT_ZONE_PLAN", "PARENT_ZONE_ORIGIN",
]
STEP_VARIABLES: dict[str, list[str]] = {
    "zone_plan_root": ["ZONE_ID", "ZONE_PROMPT", "ROOT_PROMPT", "SCENE_CONTEXT", "ROOT_OBJECTS"],
    "zone_plan": _ZONE_VARIABLES,
    "overall_bbox": ["ROOT_PROMPT", "ROOT_PLAN", "ZONE_ID", "ZONE_PROMPT", "ZONE_PLAN", "SCENE_CONTEXT", "ROOT_OBJECTS"],
    "zone_decompose_root": _ZONE_VARIABLES,
    "zone_decompose": _ZONE_VARIABLES,
    "child_bbox_batch": _ZONE_VARIABLES + ["TO_PLACE"],
    "encapsulating_decompose": _ZONE_VARIABLES + ["RETRY_BLOCK", "ADJACENT_ZONES"],
    "anchor_decompose": _ZONE_VARIABLES + ["RETRY_BLOCK"],
    "negative_space_decompose": _ZONE_VARIABLES + ["RETRY_BLOCK"],
    "object_bbox_batch": _ZONE_VARIABLES + ["TO_PLACE"],
    "next_object": _ZONE_VARIABLES + ["RETRY_BLOCK"],
    "image_prompt": _ZONE_VARIABLES + [
        "OBJECT_PROMPT", "OBJECT_DIMENSIONS", "PROXY_SHAPE", "IMAGE_TEMPLATE_FRONT",
        "IMAGE_TEMPLATE_SIDE", "IMAGE_TEMPLATE_TOP", "PRIOR_SUBJECTS",
    ],
}

_ROLES = ("system", "user")

# A variable is exactly `{NAME}` wrapped in backticks with an UPPER_SNAKE
# name. Both constraints are load-bearing: the backticks free plain-brace
# prose from ever being a variable, and the name shape keeps backticked
# lowercase brace text (the literal `{target, kind}` in several prompts)
# literal too.
_VAR_RE = re.compile(r"`\{([A-Z][A-Z0-9_]*)\}`")


class PromptTemplateError(ValueError):
    """A template set is incomplete or referenced an unavailable variable."""


def resolve(template: str, variables: dict[str, str], *, where: str) -> str:
    def _sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in variables:
            raise PromptTemplateError(
                f"{where} references `{{{name}}}`, which is not available to this step "
                f"(available: {', '.join(sorted(variables))})"
            )
        return variables[name]

    return _VAR_RE.sub(_sub, template)


@dataclass(frozen=True)
class PromptSet:
    name: str
    path: Path
    templates: dict[tuple[str, str], str]

    def _render(self, step: str, role: str, variables: dict[str, str]) -> str:
        text = self.templates.get((step, role))
        if text is None:
            raise PromptTemplateError(f"prompt set {self.name!r} has no {step}.{role}.txt")
        return resolve(text, variables, where=f"{self.name}/{step}.{role}.txt")

    def system(self, step: str, variables: dict[str, str]) -> str:
        return self._render(step, "system", variables)

    def user(self, step: str, variables: dict[str, str]) -> str:
        return self._render(step, "user", variables)

    def template(self, step: str, role: str) -> str:
        text = self.templates.get((step, role))
        if text is None:
            raise PromptTemplateError(f"prompt set {self.name!r} has no {step}.{role}.txt")
        return text

    def with_overrides(self, overrides: dict[str, dict[str, str]]) -> PromptSet:
        """A copy of this set with `{step: {"system": text, "user": text}}`
        template overrides applied — the prompt-lab's in-memory edit, used by
        downstream-simulation branches without touching the run's snapshot."""
        templates = dict(self.templates)
        for step, roles in overrides.items():
            for role, text in roles.items():
                if (step, role) not in templates:
                    raise PromptTemplateError(f"unknown template {step}.{role}")
                templates[(step, role)] = text
        return PromptSet(name=f"{self.name}+edit", path=self.path, templates=templates)


def _load_dir(path: Path, name: str) -> PromptSet:
    templates: dict[tuple[str, str], str] = {}
    missing: list[str] = []
    for step in STEPS:
        for role in _ROLES:
            f = path / f"{step}.{role}.txt"
            if not f.is_file():
                missing.append(f.name)
                continue
            templates[(step, role)] = f.read_text(encoding="utf-8")
    if missing:
        raise PromptTemplateError(
            f"prompt set at {path} is missing: {', '.join(missing)}"
        )
    return PromptSet(name=name, path=path, templates=templates)


# --- source versions ----------------------------------------------------------


def list_versions() -> list[str]:
    if not VERSIONS_DIR.is_dir():
        return []
    return sorted(p.name for p in VERSIONS_DIR.iterdir() if p.is_dir())


def version_exists(name: str) -> bool:
    return name in list_versions()


def validate_version(name: str) -> None:
    """Raise PromptTemplateError when the source version is missing files."""
    _load_dir(VERSIONS_DIR / name, name)


def snapshot_into_run(version: str, run_dir: Path) -> None:
    """Copy the source version folder into the run as its immutable snapshot."""
    shutil.copytree(VERSIONS_DIR / version, run_dir / RUN_PROMPTS_SUBDIR)


def save_snapshot_with_overrides(
    *, base_dir: Path, dest: Path, overrides: dict[str, dict[str, str]],
) -> None:
    """Copy the template set at `base_dir` to `dest`, overwriting the
    overridden step templates — how a prompt-lab edit becomes a concrete
    folder (a new source version, or a new run's snapshot). Validates the
    override keys and the completeness of the result."""
    for step, roles in overrides.items():
        if step not in STEPS:
            raise PromptTemplateError(f"unknown step: {step}")
        for role in roles:
            if role not in _ROLES:
                raise PromptTemplateError(f"unknown template role: {role}")
    shutil.copytree(base_dir, dest)
    for step, roles in overrides.items():
        for role, text in roles.items():
            (dest / f"{step}.{role}.txt").write_text(text, encoding="utf-8")
    _load_dir(dest, dest.name)


def save_version(
    name: str, *, base_dir: Path, overrides: dict[str, dict[str, str]],
) -> None:
    """Persist `base_dir` + overrides as the new source version `name`."""
    save_snapshot_with_overrides(
        base_dir=base_dir, dest=VERSIONS_DIR / name, overrides=overrides,
    )


def fork_version(name: str, *, base: str) -> None:
    """Copy source version `base` to a new source version `name` — the
    "start a fresh version to iterate on" entry point."""
    save_snapshot_with_overrides(
        base_dir=VERSIONS_DIR / base, dest=VERSIONS_DIR / name, overrides={},
    )


def write_overrides(target_dir: Path, overrides: dict[str, dict[str, str]]) -> None:
    """Overwrite step templates inside an EXISTING template folder (a run's
    snapshot being iterated in place, or its source version being kept in
    sync). Validates keys and the resulting set, and drops any cached load so
    the next pipeline (re)launch renders the new bytes."""
    for step, roles in overrides.items():
        if step not in STEPS:
            raise PromptTemplateError(f"unknown step: {step}")
        for role, text in roles.items():
            if role not in _ROLES:
                raise PromptTemplateError(f"unknown template role: {role}")
            (target_dir / f"{step}.{role}.txt").write_text(text, encoding="utf-8")
    _load_dir(target_dir, target_dir.name)
    _snapshot_cache.pop(target_dir.resolve(), None)


def sync_templates(dest_dir: Path, src_dir: Path) -> None:
    """Hard-replace EVERY step template in `dest_dir` with `src_dir`'s. Used in
    both directions: to push a run's FULL prompt snapshot back onto its source
    version (`dest_dir` = the version), and to RESTORE a run's snapshot from its
    base version (`dest_dir` = the run snapshot). Unlike `write_overrides` (which
    only touches the steps you just edited), this copies all steps, so edits
    applied to the run EARLIER without syncing still land, not just the latest
    override. Validates both sides and drops the cached load of the
    destination."""
    src = _load_dir(src_dir, src_dir.name)  # validates src is a complete set
    for (step, role), text in src.templates.items():
        (dest_dir / f"{step}.{role}.txt").write_text(text, encoding="utf-8")
    _load_dir(dest_dir, dest_dir.name)  # validate the result
    _snapshot_cache.pop(dest_dir.resolve(), None)


# --- run snapshots --------------------------------------------------------------

_snapshot_cache: dict[Path, PromptSet] = {}


def has_run_prompts(run_dir: Path) -> bool:
    return (run_dir / RUN_PROMPTS_SUBDIR).is_dir()


def load_run_prompts(run_dir: Path) -> PromptSet:
    """The run's snapshot PromptSet. Snapshots are immutable after creation,
    so loads are cached for the life of the process."""
    path = (run_dir / RUN_PROMPTS_SUBDIR).resolve()
    cached = _snapshot_cache.get(path)
    if cached is not None:
        return cached
    ps = _load_dir(path, name=run_dir.name)
    _snapshot_cache[path] = ps
    return ps


# --- task-local binding ---------------------------------------------------------

_current: ContextVar[PromptSet | None] = ContextVar("current_prompt_set", default=None)


def bind(ps: PromptSet) -> None:
    """Bind the current asyncio task to a prompt set. Called at the top of
    every pipeline task — subsequent awaits inherit the binding."""
    _current.set(ps)


def current() -> PromptSet:
    ps = _current.get()
    if ps is None:
        raise RuntimeError("no prompt set bound — prompt_store.bind() must run at task entry")
    return ps
