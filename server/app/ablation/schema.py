"""Scene-context SCHEMA ablation axis (`schema_mode`).

Render the scene context the treated step SEES as soft-JSON (baseline), XML, or
prose — holding the information constant so only *structure* varies (see
ablation-experiments.md §6, §9). This is an INPUT-only axis (the bbox solvers'
OUTPUT stays JSON), and it is GATED on the bound ablation treatment: a normal run
(no treatment) — or the `baseline` level — renders soft-JSON byte-identically to
before. Only a schema variant's single re-inferred step renders XML/prose.

Both the rendered string AND the per-attribute role span-map come from the SAME
pass of the SHARED emitter (`app.core.scene_render`), so the string the model
sees and the spans the attention worker scores on can never drift — this is the
"log the span-map via the shared emitter" design (§9.5, Option 1). The span-map
rides in the step's `variables` dict under `ROLES_VAR` (already logged verbatim
in the `cache.llm` event; never a template variable), and
`app.services.teacher_forcing` reads it back instead of scraping soft-JSON.
"""

from __future__ import annotations

import json
from typing import Any

from app.ablation import context as _ctx
from app.core import scene_render

# baseline == the current soft-JSON; xml / prose are the treated renderings.
SCHEMA_MODES: tuple[str, ...] = ("baseline", "xml", "prose")
DEFAULT_MODE = "baseline"
_FMT: dict[str, str] = {"baseline": "json", "xml": "xml", "prose": "prose"}

# Scene-bearing variables that share the entry grammar `scene_render.parse_scene`
# reads. `SCENE_CONTEXT` is the one that matters on the later steps we test; the
# others are usually empty there, and empty/placeholder values are left untouched.
SCENE_VARS: tuple[str, ...] = (
    "SCENE_CONTEXT", "SCENE_CONTEXT_COMPACT", "ADJACENT_ZONES",
    "ROOT_OBJECTS", "ZONE_OBJECTS", "TO_PLACE",
)

# Synthetic `variables` key the per-var span-map rides in — logged with the
# cache.llm event and read back by teacher_forcing. It is NOT a template variable
# (prompt_store.resolve only substitutes `{UPPER}` names present in a template),
# and its leading underscores can't match `_VAR_RE`, so it never affects a render.
ROLES_VAR = "__SCENE_ROLES__"


def _norm(mode: str | None) -> str:
    return mode if mode in SCHEMA_MODES else DEFAULT_MODE


def current_mode() -> str:
    """The bound variant's schema_mode, or `baseline` on a normal run."""
    rt = _ctx.current()
    if rt is None:
        return DEFAULT_MODE
    return _norm(getattr(rt.treatment, "schema_mode", DEFAULT_MODE))


def current_fmt() -> str:
    """`json` | `xml` | `prose` for the bound mode."""
    return _FMT[current_mode()]


def is_active() -> bool:
    """True when a schema treatment asks for a non-JSON rendering."""
    return current_fmt() != "json"


def apply_to_vars(variables: dict[str, Any]) -> None:
    """If a schema ablation is bound, RE-RENDER each scene-bearing var in
    `variables` into the chosen format IN-PLACE and accumulate the per-var role
    span-map under `ROLES_VAR` (JSON). No-op for baseline/json or a normal run,
    so non-ablation renders are byte-identical.

    Idempotent + additive: a var already recorded in the span-map is skipped
    (a rendered XML/prose blob can't be re-parsed as soft-JSON), so this can be
    called again after later vars (e.g. `TO_PLACE`) are added to the dict."""
    fmt = current_fmt()
    if fmt == "json":
        return
    try:
        recorded: dict[str, Any] = json.loads(variables.get(ROLES_VAR) or "{}")
    except Exception:
        recorded = {}
    changed = False
    for var in SCENE_VARS:
        soft = variables.get(var)
        if not soft or var in recorded:
            continue
        if "{" not in soft or soft.strip().startswith("{(none"):
            continue  # empty / placeholder scene var — leave verbatim
        try:
            entities = scene_render.parse_scene(soft)
            if not entities:
                continue
            text, smap = scene_render.build_scene_map(fmt, soft, entities)
        except Exception:
            # Never break a render on a parse hiccup — fall back to soft-JSON for
            # this var (attribution then scrapes it as JSON, still correct).
            continue
        variables[var] = text
        recorded[var] = smap
        changed = True
    if changed:
        variables[ROLES_VAR] = json.dumps(recorded)
