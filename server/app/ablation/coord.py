"""Coordinate-frame ablation axis (`coord_mode`).

A SINGLE enumerated experiment axis (never a cross of two toggles) controlling,
for the two bbox-solver steps, how coordinates are SHOWN in scene context (the
INPUT representation) and which frame the solver must EMIT (the OUTPUT frame).

Levels (L = local / parent-relative, G = global / world; the L->G case is
intentionally excluded — local-only input gives the model no absolute anchor to
produce a global answer):

    id        condition   input   output   note
    baseline  L/G -> L    both    local    current production behaviour
    lg2g      L/G -> G     both    global
    l2l       L   -> L     local   local
    g2g       G   -> G     global  global
    g2l       G   -> L     global  local

Because it is ONE axis, the wizard fans it out to exactly its own conditions and
it never cross-multiplies with the shuffle / xml axes.

Runtime seam
------------
The active level rides the SAME task-local ablation ContextVar as every other
treatment (`app.ablation.context`). Two consumption points read it:

  * scene-context rendering — INPUT: which of the per-entity ``Global origin
    corner`` / ``Local origin corner`` lines are emitted (see `show_global` /
    `show_local`, read at render time in `app.core.scene_context`).
  * the two bbox solvers — OUTPUT: the local->world conversion is skipped when
    the solver already emits world coordinates (see `emit_global`, read in
    `app.pipeline.generation` / `app.pipeline.divider`).

Prompt wording
--------------
The instruction wording (system `<output>` bullet + the user-prompt "bounding
boxes you give" paragraph) and the scene-context prose are rewritten ONCE, at
pipeline bind, via `prompt_overrides` + `PromptSet.with_overrides` — so a variant
run's treated bbox firing renders the right instructions while every non-bbox
step (and every replayed prefix step) is untouched. All rewrites are best-effort
literal replacements against the base (minglun) templates: an absent anchor is a
no-op, so a different prompt version simply keeps its own wording.
"""

from __future__ import annotations

from typing import Any

from app.ablation import context as _ctx

# The two steps that both READ coordinates from scene context and EMIT them —
# the only steps the OUTPUT axis is meaningful for.
BBOX_STEPS: tuple[str, ...] = ("object_bbox_batch", "child_bbox_batch")

DEFAULT_MODE = "baseline"

# level id -> (input_repr, output_frame); input_repr in {both, local, global},
# output_frame in {local, global}.
_MODES: dict[str, tuple[str, str]] = {
    "baseline": ("both", "local"),
    "lg2g": ("both", "global"),
    "l2l": ("local", "local"),
    "g2g": ("global", "global"),
    "g2l": ("global", "local"),
}


def is_mode(coord_mode: str | None) -> bool:
    return coord_mode in _MODES


def _norm(coord_mode: str | None) -> str:
    return coord_mode if coord_mode in _MODES else DEFAULT_MODE


def current_mode() -> str:
    """The active variant's coord_mode, or `baseline` on a normal run."""
    rt = _ctx.current()
    if rt is None:
        return DEFAULT_MODE
    return _norm(getattr(rt.treatment, "coord_mode", DEFAULT_MODE))


def input_repr(coord_mode: str | None = None) -> str:
    """`both` | `local` | `global` — for `coord_mode` if given, else the bound one."""
    m = _norm(coord_mode) if coord_mode is not None else current_mode()
    return _MODES[m][0]


def output_frame(coord_mode: str | None = None) -> str:
    """`local` | `global` — for `coord_mode` if given, else the bound one."""
    m = _norm(coord_mode) if coord_mode is not None else current_mode()
    return _MODES[m][1]


def show_global() -> bool:
    """Emit each entity's world (`Global origin corner`) line in scene context?"""
    return input_repr() in ("both", "global")


def show_local() -> bool:
    """Emit each entity's parent-relative (`Local origin corner`) line in scene context?"""
    return input_repr() in ("both", "local")


def emit_global() -> bool:
    """Does the solver emit world-frame boxes (so the local->world conversion is skipped)?"""
    return output_frame() == "global"


# --- prompt rewrites ---------------------------------------------------------
# Exact anchors from the minglun base templates. Replacements are literal, so an
# unexpected template (or an already-edited one) is left untouched.

# OUTPUT frame: the system `<output>` bullet (local -> world).
_OBJ_SYS_LOCAL = (
    "the box's MINIMUM corner, expressed in the LOCAL frame of the object's "
    "declared `parent` — (0, 0, 0) is the parent's own minimum corner, with axes "
    "parallel to the world axes (when the parent is the region itself, this is "
    "the region's minimum corner)."
)
_OBJ_SYS_GLOBAL = (
    "the box's MINIMUM corner in the GLOBAL (world) frame — the SAME absolute "
    'frame in which the overall scene bounding box and every "Global origin '
    'corner" in the scene context are expressed. (0, 0, 0) is the world origin, '
    "with axes parallel to the world axes. Emit the object's absolute world "
    "position directly; do NOT measure it relative to the parent."
)
_CHILD_SYS_LOCAL = (
    "the box's MINIMUM corner, expressed in the LOCAL frame of the parent region "
    "`{ZONE_ID}` — (0, 0, 0) is the parent region's own minimum corner, with axes "
    "parallel to the world axes."
)
_CHILD_SYS_GLOBAL = (
    "the box's MINIMUM corner in the GLOBAL (world) frame — the SAME absolute "
    'frame in which the overall scene bounding box and every "Global origin '
    'corner" in the scene context are expressed. (0, 0, 0) is the world origin, '
    "with axes parallel to the world axes. Emit each subregion's absolute world "
    "position directly; do NOT measure it relative to the parent region."
)

# OUTPUT frame: the user-prompt "bounding boxes you give" paragraph.
_OBJ_USER_LOCAL = (
    "The minimum corner of each object's bounding box should be given relative to "
    "the bounding box of its parent, treating the minimum corner of the parent's "
    "bounding box as the origin (0, 0, 0). think about how the object is placed "
    "relatively to its parent, and the semantic meanings behind that."
)
_OBJ_USER_GLOBAL = (
    "The minimum corner of each object's bounding box should be given relative to "
    "the bounding box of the scene, treating the minimum corner of the scene's "
    "bounding box as the origin (0, 0, 0). think about how the object is placed, "
    "and the semantic meanings behind that."
)
_CHILD_USER_LOCAL = (
    "The minimum corner of each subregion's bounding box should be given relative "
    "to the bounding box of the parent region, treating the minimum corner of "
    "`{ZONE_ID}` as the origin (0, 0, 0)."
)
_CHILD_USER_GLOBAL = (
    "The minimum corner of each subregion's bounding box should be given relative "
    "to the bounding box of the scene, treating the minimum corner of the scene "
    "as the origin (0, 0, 0)."
)

_OUTPUT_GLOBAL_REPLACEMENTS: dict[str, tuple[tuple[str, str], ...]] = {
    "object_bbox_batch": (
        (_OBJ_SYS_LOCAL, _OBJ_SYS_GLOBAL),
        (_OBJ_USER_LOCAL, _OBJ_USER_GLOBAL),
    ),
    "child_bbox_batch": (
        (_CHILD_SYS_LOCAL, _CHILD_SYS_GLOBAL),
        (_CHILD_USER_LOCAL, _CHILD_USER_GLOBAL),
    ),
}

# INPUT representation: the object solver's "two sets of coordinates" description
# (object system only — the child solver carries no such sentence).
_INPUT_DESC_BOTH = (
    "In the scene context, you are provided two sets of coordinates for every "
    "object/region: the global coordinates and the local coordinates. the local "
    "coordinates are expressed relative to the parent object/region, with the "
    "parent object/region's bounding box's minimum corner being (0, 0, 0). you "
    "should use the local coordinates to understand how an object is placed "
    "relative to its parent, and the global coordinates to understand how an "
    "object is placed overall."
)
_INPUT_DESC_GLOBAL = (
    "In the scene context, you are provided the global (world) coordinates of "
    "every object/region — its position in the single absolute world frame the "
    "entire scene shares. Use these global coordinates to understand where each "
    "object/region sits in the scene."
)
_INPUT_DESC_LOCAL = (
    "In the scene context, you are provided the local coordinates of every "
    "object/region, expressed relative to its parent object/region, with the "
    "parent object/region's bounding box's minimum corner being (0, 0, 0). Use "
    "these local coordinates to understand how each object/region is placed "
    "relative to its parent."
)

# INPUT representation: the shared scene-context explainer sentence (present in
# both bbox solvers' user prompts, verbatim).
_EXPLAINER_BOTH = (
    "Additionally, each subregion and object mentioned will also have a set of "
    "local coordinates that define its position relative to its parent (not to be "
    "confused with parent_region for objects), where the origin is the actual "
    "minimum corner of the parent's bounding box."
)
_EXPLAINER_GLOBAL = (
    "Additionally, each subregion and object mentioned is shown with its global "
    "(world) coordinates — its origin corner in the single absolute world frame "
    "the entire scene shares."
)
_EXPLAINER_LOCAL = (
    "Additionally, each subregion and object mentioned is shown with its local "
    "coordinates that define its position relative to its parent (not to be "
    "confused with parent_region for objects), where the origin is the actual "
    "minimum corner of the parent's bounding box."
)


def _rewrite_output_global(text: str, step: str) -> str:
    for local, world in _OUTPUT_GLOBAL_REPLACEMENTS.get(step, ()):  # noqa: B007
        text = text.replace(local, world)
    return text


def _rewrite_input(text: str, inp: str) -> str:
    if inp == "global":
        text = text.replace(_INPUT_DESC_BOTH, _INPUT_DESC_GLOBAL)
        text = text.replace(_EXPLAINER_BOTH, _EXPLAINER_GLOBAL)
    elif inp == "local":
        text = text.replace(_INPUT_DESC_BOTH, _INPUT_DESC_LOCAL)
        text = text.replace(_EXPLAINER_BOTH, _EXPLAINER_LOCAL)
    return text


def prompt_overrides(ps: Any, coord_mode: str | None) -> dict[str, dict[str, str]]:
    """`{step: {"system": text, "user": text}}` overrides for the two bbox solvers
    under `coord_mode`: rewrite the OUTPUT-frame instruction (local -> world) when
    the mode emits global, and the INPUT scene-context prose (global-only /
    local-only) when the mode shows a single representation. Returns only the
    steps that actually change — empty for `baseline` (and any mode whose text
    anchors aren't present, i.e. a non-minglun prompt set)."""
    mode = _norm(coord_mode)
    if mode == "baseline":
        return {}
    inp = input_repr(mode)
    out = output_frame(mode)
    overrides: dict[str, dict[str, str]] = {}
    for step in BBOX_STEPS:
        try:
            system = ps.template(step, "system")
            user = ps.template(step, "user")
        except Exception:
            continue
        new_system, new_user = system, user
        if out == "global":
            new_system = _rewrite_output_global(new_system, step)
            new_user = _rewrite_output_global(new_user, step)
        if inp != "both":
            new_system = _rewrite_input(new_system, inp)
            new_user = _rewrite_input(new_user, inp)
        if new_system != system or new_user != user:
            overrides[step] = {"system": new_system, "user": new_user}
    return overrides
