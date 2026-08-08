"""The one-shot pipeline: a scene is designed in one (or two) LLM calls, then meshed.

Three selectable pipeline VERSIONS (a run pins one at creation; see routes):
  * v1-bbox    — one structured JSON call returning objects placed by
                 world-frame `bbox` and yaw. Like every version, the name
                 is the sole semantic payload.
  * v2-spatial — spatial-only: one PLAIN-TEXT call returning a CSV object
                 list (`name,[x_min,y_min,z_min],[w,h,d],yaw` — world-frame
                 min-corner boxes). The name is the sole semantic payload.
  * v3-room    — one plain-text call against a HARNESS-BUILT room shell
                 (floor + four perimeter walls, sized by the slot's
                 `canvas_ft` — the only version whose canvas scales with
                 the brief): the model designs the interior as center+size
                 rows (`name,(x, y),(width, length, height)`), which the
                 harness turns into boxes standing on the floor — the
                 top-view center plus a footprint, never a y coordinate.
  * v4-grid-only — two plain-text calls: the floor plan as 16
                 comma-separated rows whose cells carry object NAMES
                 directly (0 = empty floor) — names are the plan's sole
                 currency, committed and embedded as-is — then a conversion
                 call that reads the name-celled matrix verbatim (axis
                 markers, no legend) as a ROUGH SEMANTIC GUIDE and emits
                 the same center+size rows as v3-room, boxed by the
                 harness onto the fixed canvas.
  * v5-sidescroller — ONE plain-text call for 2D platformer-style levels:
                 a 32 wide x 8 tall name-celled SIDE VIEW with 2 ft cells,
                 i.e. a 64 ft x 16 ft level (model x = scroll axis, model
                 y = level vertical -> world z; row 1 is the level's top).
                 There is no conversion call — a deterministic solver maps
                 cells 1:1 onto the level plane as V5_THICKNESS_FT-thin
                 slabs on a harness BACKGROUND WALL, fusing same-name
                 neighbours by object KIND: unit items (bricks, blocks,
                 coins) stay one object per cell, structures (pipes,
                 ground, platforms) merge per straight rectangular run.
                 After all LLM calls, a deterministic post-pass rebuilds
                 WALLS from the plan's wall patches as individual thin
                 frames (`_normalize_v4_walls`) for consistent viewing.

Every version builds inside a canvas square centered on the origin, stated
in the prompts — fixed at CANVAS_SIDE_FT except v3-room, whose side comes
from the slot's `canvas_ft`. No model authors the scene bounds (the
benchmark evaluates interior layout only); run() pins the root bbox to the
canvas footprint, y hugging the content.

v1 keeps provider-side structured output; the rest are deliberately
schema-free (`llm.call_text` + the parsers below). Every design call is a
single attempt — an output violation fails the call outright with a precise
validator message (no auto retry / resample), logged as `llm.failed` with
the exact prompts and verbatim output. v4's name-grid parse is deliberately
lenient instead: structural damage is repaired best-effort and each repair
is recorded as a warning on the committed output (surfaced by the info
panel), so the layout still flows into conversion; only a response with no
grid rows at all fails.

Each version's prompt templates live in `prompts/<version>/` and are re-read
on every launch. All versions converge on the same placement currency
(`_PlacedObject`), so dedup, events, library matching, and the GLB bake are
shared verbatim.

Resume is optimistic and log-driven: each design call replays its committed
`cache.llm` output verbatim keyed by step identity (`oneshot_grid` for the
grid calls, `oneshot_scene` for v1-v4's final scene call; v5's only call
is the grid — so a prompt-file edit between pause and resume can't fork a
half-built scene), library matches cache by prompt text, and
`bbox`/`image`/`model`/`oneshot.grid`/`oneshot.solver` events dedup on
re-emission; already-placed GLBs are skipped via `path.exists()`.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.core import prompt_store
from app.core.types import BoundingBox, Orientation
from app.oneshot import llm
from app.oneshot.slots import DllmModel
from app.services import library
from app.utils import glb_place, logging

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

VERSIONS: list[str] = [
    "v1-bbox", "v2-spatial", "v3-room", "v4-grid-only", "v5-sidescroller",
]
DEFAULT_VERSION = "v2-spatial"

# The track's world unit is FEET (a 50 ft canvas reads as a hall; 50 m read
# as a stadium). Numbers flow verbatim from model output to world coords —
# the mesh bake stretches assets by ratio, so the unit only has to be
# consistent between the prompts and these constants.
#
# EVERY version shares one fixed canvas — a CANVAS_SIDE_FT square centered
# on the world origin, stated in the prompts via the `{CANVAS_FT}` family of
# variables (v3-room overrides the side per slot). No model authors the
# scene bounds (v1's scene_bbox is gone); the benchmark evaluates interior
# layout only, and run() pins the root bbox to this footprint.
CANVAS_SIDE_FT = 50.0
GRID_SIZE = 16
# v4's grid stays GRID_SIZE cells per side regardless of canvas size (wide
# name-celled rows are where dLLMs drift), so a cell now spans several feet.
# The conversion step never chooses the scene footprint (a model-picked
# scale was observed double-transforming walls and doll-sizing furniture).
CELL_FT = CANVAS_SIDE_FT / GRID_SIZE
# The ground slab is pre-placed by the harness (walkable top at y = 0,
# spanning the whole canvas) so no design call has to invent its own ground
# geometry.
FLOOR_THICKNESS_FT = 1.0
# v5's side-scroller level: a 32-wide x 8-tall front-on grid (model x =
# scroll axis, model y = level vertical -> world z) with its own 2 ft cell,
# so the level spans 64 ft x 16 ft. World y is degenerate in 2D — every
# object is a V5_THICKNESS_FT-thin slab standing proud of the harness
# background wall.
V5_GRID_COLS = 32
V5_GRID_ROWS = 8
V5_CELL_FT = 2.0
V5_THICKNESS_FT = 1.0
# v5's side view lays the level's vertical onto world z, but library assets
# stand up along world y. Bake a fixed +Z->+Y reorientation (a -90° turn
# about x) into every v5 placement so assets stand upright on the level
# plane instead of lying flat across it.
V5_UPRIGHT_ROTATION = glb_place.quat_x(-90.0)
# How v5's deterministic solver fuses same-name neighbours depends on the
# KIND of object. Names carrying one of these tokens are UNIT items — a row
# of bricks is N individual blocks, like Mario's — while every other name
# (pipe, ground, platform, stair, ...) merges into one object per straight
# rectangular run. Tuned against live grid logs (gemma/opus/qwen/inception
# mario + hollow knight runs).
V5_UNIT_TOKENS = ("block", "brick", "coin", "crate", "box", "question")

T = TypeVar("T", bound=BaseModel)


# --- v1-bbox output schema ------------------------------------------------------


class OneShotObjectV1(BaseModel):
    """Same `name` contract as every version — the only semantic payload,
    library matching runs on it verbatim. v1 differs only in transport:
    structured JSON with the box nested under `bbox`, instead of CSV
    columns."""

    name: str
    bbox: BoundingBox
    orientation: Orientation = 0

    @field_validator("name", mode="after")
    @classmethod
    def _named(cls, v: str) -> str:
        # "root" is reserved for the scene canvas and an empty name has
        # nothing for the mesh generator to work from.
        if not v or v == "root":
            raise ValueError('name must be a non-empty identifier other than "root"')
        return v

    @field_validator("bbox", mode="after")
    @classmethod
    def _canonicalize(cls, v: BoundingBox) -> BoundingBox:
        return v.canonical()


class OneShotSceneOutputV1(BaseModel):
    # Pre-canvas-era outputs carried a model-authored `scene_bbox`; pydantic
    # ignores the extra key, so old cells still replay through this schema.
    objects: list[OneShotObjectV1] = Field(min_length=1)


# --- v2-spatial output schema -----------------------------------------------------


class OneShotObject(BaseModel):
    """One placed object, spatial fields only. `name` doubles as the node id
    and the ONLY semantic payload — library matching runs on it verbatim.
    `origin` + `dimensions` are the FINAL world-frame AABB (v1 semantics:
    `origin` is the box's minimum corner; the mesh is turned to
    `orientation` first, then stretched to fill the box). The base-center
    anchor this replaced made even frontier models float walls at y = h/2 —
    box-form outputs in the v1 logs never did."""

    name: str
    origin: tuple[float, float, float]
    dimensions: tuple[float, float, float]
    orientation: Orientation = 0

    @field_validator("name", mode="after")
    @classmethod
    def _named(cls, v: str) -> str:
        # "root" is the derived scene canvas's reserved id; an empty name has
        # nothing for the mesh generator to work from. Rejecting here fails
        # the call with an explicit message.
        if not v or v == "root":
            raise ValueError('name must be a non-empty identifier other than "root"')
        return v

    @field_validator("dimensions", mode="after")
    @classmethod
    def _positive(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if min(v) <= 0:
            raise ValueError("every dimension must be a positive number of feet")
        return v

    def world_bbox(self) -> BoundingBox:
        """The model's box IS the world AABB — no anchor math."""
        return BoundingBox(origin=self.origin, dimensions=self.dimensions).canonical()


class OneShotSceneOutput(BaseModel):
    objects: list[OneShotObject] = Field(min_length=1)


_OBJECTS_CSV_HEADER = "name,origin,dimensions,orientation"
# Confused models occasionally echo the previous era's header; both skip.
_HEADER_VARIANTS = {_OBJECTS_CSV_HEADER, "name,position,dimensions,orientation"}


def _split_outside_brackets(line: str) -> list[str]:
    fields: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in line:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            fields.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    fields.append("".join(buf).strip())
    return fields


def _vec3(field: str, label: str, row: int) -> tuple[float, float, float]:
    s = field.strip().strip("\"'").strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p for p in re.split(r"[,\s]+", s.strip()) if p]
    if len(parts) != 3:
        raise ValueError(
            f"object row {row}: {label} must be a [x,y,z] triple of 3 numbers, got {field!r}"
        )
    try:
        x, y, z = (float(p) for p in parts)
    except ValueError:
        raise ValueError(
            f"object row {row}: {label} has a non-numeric value in {field!r}"
        ) from None
    return (x, y, z)


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_ROW_NAME_RE = re.compile(r"\s*[\"']?([A-Za-z_][A-Za-z0-9_ -]*)")


def _parse_object_fields(s: str, row: int) -> tuple[str, list[float]]:
    """(name, [x, y, z, w, h, d, yaw]) from one row. The strict 4-column form
    is tried first; on any structural miss a REPAIR pass runs — dLLMs drop a
    '[' or the comma after the name in otherwise-perfect rows (seen live from
    diffusiongemma), and since a row is unambiguous as a leading name plus
    exactly 7 numbers, that damage is recoverable without failing the call."""
    fields = _split_outside_brackets(s)
    if len(fields) == 4:
        try:
            name = fields[0].strip().strip("\"'")
            nums = [
                *_vec3(fields[1], "origin", row),
                *_vec3(fields[2], "dimensions", row),
                float(fields[3].strip().strip("\"'")),
            ]
            return name, nums
        except ValueError:
            pass
    m = _ROW_NAME_RE.match(s)
    found = _NUM_RE.findall(s[m.end():]) if m else []
    if m is None or len(found) != 7:
        raise ValueError(
            f"object row {row} must be `name,[x_min,y_min,z_min],[w,h,d],yaw` — "
            f"could not recover a name plus exactly 7 numbers from {s!r}"
        )
    return m.group(1).strip(), [float(t) for t in found]


def parse_objects_csv(content: str) -> OneShotSceneOutput:
    """The v2 object-list contract: an optional header line, then one
    `name,[x_min,y_min,z_min],[w,h,d],yaw` row per object (strict form
    first, per-row repair for bracket/comma damage). Code fences, blank
    lines, header variants, and digit-less prose are ignored — they cannot
    carry an object. Raises ValueError with row-precise messages that surface
    on the failed call; semantic violations (bad yaw, "root", non-positive
    dims) are never repaired."""
    objects: list[OneShotObject] = []
    row = 0
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("```"):
            continue
        if s.replace(" ", "").lower() in _HEADER_VARIANTS:
            continue
        if "[" not in s and not any(ch.isdigit() for ch in s):
            continue
        row += 1
        name, nums = _parse_object_fields(s, row)
        x, y, z, w, h, d, yaw = nums
        try:
            objects.append(
                OneShotObject(
                    name=name,
                    origin=(x, y, z),
                    dimensions=(w, h, d),
                    orientation=int(yaw),
                )
            )
        except ValidationError as e:
            raise ValueError(f"object row {row} ({name!r}): {e}") from None
    if not objects:
        raise ValueError(
            "no object rows found — emit one `name,[x_min,y_min,z_min],[w,h,d],yaw` "
            "line per object"
        )
    return OneShotSceneOutput(objects=objects)


# --- grid-plan currency (v4 stage 1) ------------------------------------------------


class GridPlanOutput(BaseModel):
    """The floor plan exactly as the model authored it: GRID_SIZE rows of
    GRID_SIZE cells, each cell the object's name ("" = open floor). Names
    are the sole currency — there is no legend or numeric indirection
    anywhere. The parser repairs violations instead of failing, recording
    each on `warnings`, which rides the committed output so the dashboard's
    info panel can surface it (and replay keeps it)."""

    grid: list[list[str]]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> GridPlanOutput:
        # Per-version dimensions (16x16 top-down, 32x16 side view) are
        # enforced by the parser; the committed currency only has to be a
        # non-empty rectangle.
        if not self.grid or not self.grid[0]:
            raise ValueError("grid must have at least one row of cells")
        w = len(self.grid[0])
        for r, row in enumerate(self.grid, start=1):
            if len(row) != w:
                raise ValueError(f"grid row {r} has {len(row)} cells, expected {w}")
        return self


def render_grid_plan(plan: GridPlanOutput) -> str:
    """The floorplan with the model's own object names in the cells (0 =
    open floor) — no legend, no numeric indirection — which is exactly what
    the conversion call reads and the dashboard overlay displays. The
    header carries axis markers for the 2D-minded conversion step: +x runs
    rightward across a row and +y runs downward through the rows (the
    plan's y is world z). Shape-agnostic, so it serves both the square
    top-down plans and v5's wide side-view levels."""
    header = "Floorplan (+x -> right, +y -> down the rows):"
    rows = [",".join(c if c else "0" for c in row) for row in plan.grid]
    return "\n".join([header, *rows])


# --- v4-grid-only output contract ---------------------------------------------------


def parse_name_grid(
    content: str, *, cols: int = GRID_SIZE, n_rows: int = GRID_SIZE
) -> GridPlanOutput:
    """Best-effort parse of the name-grid contract: `n_rows` comma-separated
    rows of `cols` cells carrying object names directly (0/empty = open
    floor, brackets tolerated) — 16x16 for v4's top-down plan, 32x16 for
    v5's side view. Lines that don't split into enough cells are prose and
    skipped. Structural damage does NOT fail the call — each violation is
    repaired and recorded on the plan's `warnings` (the dashboard's info
    panel shows them): short/long rows pad/truncate to `cols` cells,
    surplus rows keep the LAST `n_rows` (models echo the prompt's example
    grid before answering), missing rows pad with empty floor. Only a
    response with no grid rows at all raises."""
    rows: list[list[str]] = []
    warnings: list[str] = []
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("```"):
            continue
        cells = [c.strip().strip("\"'").strip() for c in s.strip("[]").split(",")]
        if len(cells) < cols // 2:
            continue
        if len(cells) != cols:
            warnings.append(
                f"row {len(rows) + 1} has {len(cells)} cells, expected "
                f"{cols} — padded/truncated to fit"
            )
            cells = (cells + ["0"] * cols)[:cols]
        rows.append(["" if c == "0" else c for c in cells])
    if not rows:
        raise ValueError(
            f"no grid rows found — emit {n_rows} lines of {cols} "
            "comma-separated cells (0 = empty floor, or the object's name)"
        )
    if len(rows) > n_rows:
        warnings.append(
            f"found {len(rows)} candidate rows, expected {n_rows} — kept "
            f"the last {n_rows}"
        )
        rows = rows[-n_rows:]
    elif len(rows) < n_rows:
        warnings.append(
            f"found only {len(rows)} rows, expected {n_rows} — padded the "
            "rest with empty floor"
        )
        rows += [[""] * cols for _ in range(n_rows - len(rows))]
    if not any(c for row in rows for c in row):
        warnings.append(
            "the matrix is all zeros — the scene will only contain the floor"
        )
    return GridPlanOutput(grid=rows, warnings=warnings)


# --- v3-room / v4-convert output schema (center + size boxes) ----------------------


class RoomObject(BaseModel):
    """One object in the top-view center+size schema shared by v3-room's
    single call and v4's conversion call: `position` is the object's CENTER
    in the top view — model (x, y) maps to world (x, z) — and `size` is
    `(width, length, height)`: width spans world x, length spans world z
    (the plan's y axis), height spans world y. The box stands on the floor
    (y spans 0..height, x/z span center ± half the footprint), so nothing
    dips below y = 0."""

    name: str
    position: tuple[float, float]
    size: tuple[float, float, float]

    @field_validator("name", mode="after")
    @classmethod
    def _named(cls, v: str) -> str:
        # Degenerate output normalizes instead of failing the call: a blank
        # name becomes a visible "placeholder". "root" stays reserved for
        # the scene canvas.
        v = v.strip() or "placeholder"
        if v == "root":
            raise ValueError('name must be an identifier other than "root"')
        return v

    @field_validator("size", mode="after")
    @classmethod
    def _positive(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        # Zero (or negative) components flatten the box and used to hard-fail
        # the call; nudge them to a visible 0.1 ft sliver instead.
        return tuple(max(c, 0.1) for c in v)

    def world_bbox(self) -> BoundingBox:
        x, z = self.position
        w, length, h = self.size
        return BoundingBox(
            origin=(x - w / 2, 0.0, z - length / 2),
            dimensions=(w, h, length),
        )


class RoomSceneOutput(BaseModel):
    objects: list[RoomObject] = Field(min_length=1)


# `position:` / `size:` (or `dimensions:`) field labels in the row form.
# Digit-free, so number extraction never sees them — but when a dLLM drops
# the comma before a label, the label word bleeds into the name capture and
# must be trimmed off its tail.
_LABEL_TAIL_RE = re.compile(r"(?:[\s,;]*\b(?:position|dimensions|size))+\s*$", re.IGNORECASE)


def parse_room_csv(content: str) -> RoomSceneOutput:
    """The center+size contract: one row per object in the labeled form
    `name, position: (x, y), size: (w,l,h)` (`dimensions:` is tolerated as
    a size synonym, and bare unlabeled tuples still parse — the labels are
    digit-free, so extraction is label-blind). Mirrors parse_objects_csv's
    dLLM-tolerant stance — a row is unambiguous as a leading name plus
    exactly 5 numbers, so paren/comma damage is recoverable; fences,
    headers, and digit-less prose are skipped (they cannot carry an
    object). Degenerate values normalize rather than fail (nameless rows
    become "placeholder", zero-size components become 0.1 ft); a
    digit-bearing line with the wrong number count still raises
    row-precisely, as does the reserved name "root"."""
    form = "`name, position: (x, y), size: (width, length, height)`"
    objects: list[RoomObject] = []
    row = 0
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("```"):
            continue
        if not any(ch.isdigit() for ch in s):
            continue
        row += 1
        m = _ROW_NAME_RE.match(s)
        nums = _NUM_RE.findall(s[m.end():] if m else s)
        if len(nums) != 5:
            raise ValueError(
                f"object row {row} must be {form} — could not recover "
                f"exactly 5 numbers from {s!r}"
            )
        name = _LABEL_TAIL_RE.sub("", m.group(1) if m else "").strip() or "placeholder"
        x, y, w, length, h = (float(t) for t in nums)
        try:
            objects.append(
                RoomObject(name=name, position=(x, y), size=(w, length, h))
            )
        except ValidationError as e:
            raise ValueError(f"object row {row} ({name!r}): {e}") from None
    if not objects:
        raise ValueError(f"no object rows found — emit one {form} line per object")
    return RoomSceneOutput(objects=objects)


def _from_room(out: RoomSceneOutput) -> list[_PlacedObject]:
    """Center+size rows to floor-standing boxes. The schema has no
    uniqueness rule — three chairs are three `chair` rows — so ids get
    instance suffixes while the raw name stays the library match text."""
    counts: dict[str, int] = {}
    placed: list[_PlacedObject] = []
    for o in out.objects:
        k = counts.get(o.name, 0) + 1
        counts[o.name] = k
        placed.append(
            _PlacedObject(o.name if k == 1 else f"{o.name} {k}", o.name, o.world_bbox(), 0)
        )
    return placed


# --- common placement currency ----------------------------------------------------


@dataclass(frozen=True)
class _PlacedObject:
    """What every version's design call reduces to: a node id, the text the
    library matcher sees (the object's id/name), the final world AABB, and
    the yaw for the bbox event."""

    id: str
    match_text: str
    bbox: BoundingBox
    orientation: Orientation


def _artifact_url(out_root: Path, path: Path) -> str:
    return f"/oneshot/artifacts/{path.relative_to(out_root).as_posix()}"


def render_step(
    version: str,
    stem: str,
    prompt: str,
    *,
    grid_plan: str = "",
    canvas_ft: float = CANVAS_SIDE_FT,
) -> tuple[str, str]:
    """Resolve one step's editable templates (`prompts/<version>/<stem>.*.txt`)
    against the scene prompt. Read fresh on every call so prompt-file edits
    apply to the next launch without a server restart."""
    cell = V5_CELL_FT if version == "v5-sidescroller" else CELL_FT
    variables = {
        "SCENE_PROMPT": prompt,
        "GRID_PLAN": grid_plan,
        "CANVAS_FT": f"{canvas_ft:g}",
        "CANVAS_HALF_FT": f"{canvas_ft / 2:g}",
        "CELL_FT": f"{cell:g}",
    }
    base = _PROMPTS_DIR / version
    system = prompt_store.resolve(
        (base / f"{stem}.system.txt").read_text(encoding="utf-8"),
        variables,
        where=f"{version}/{stem}.system.txt",
    )
    user = prompt_store.resolve(
        (base / f"{stem}.user.txt").read_text(encoding="utf-8"),
        variables,
        where=f"{version}/{stem}.user.txt",
    )
    return system, user


async def _structured_step(
    model_cfg: DllmModel,
    *,
    version: str,
    stem: str,
    step: str,
    schema: type[T],
    prompt: str,
    grid_plan: str = "",
) -> T:
    # Structural replay: a committed output is reused verbatim on resume,
    # keyed by step identity rather than prompt bytes — editing the prompt
    # files between pause and resume must not fork a half-built scene.
    e = logging.find_event("cache.llm", step=step)
    if e is not None and isinstance(e.get("output"), dict):
        try:
            return schema.model_validate(e["output"])
        except Exception:
            pass
    system, user = render_step(version, stem, prompt, grid_plan=grid_plan)
    return await llm.call_structured(
        model_cfg,
        system=system,
        user=user,
        output_schema=schema,
        node_id="root",
        step=step,
    )


async def _text_step(
    model_cfg: DllmModel,
    *,
    version: str,
    stem: str,
    step: str,
    schema: type[T],
    parse: Callable[[str], T],
    prompt: str,
    grid_plan: str = "",
    canvas_ft: float = CANVAS_SIDE_FT,
) -> T:
    """`_structured_step`'s plain-text twin: same step-identity replay (the
    logged `output` is the canonical parsed model, so old JSON-era cells
    replay through the same path), then a schema-free completion parsed by
    `parse`."""
    e = logging.find_event("cache.llm", step=step)
    if e is not None and isinstance(e.get("output"), dict):
        try:
            return schema.model_validate(e["output"])
        except Exception:
            pass
    system, user = render_step(
        version, stem, prompt, grid_plan=grid_plan, canvas_ft=canvas_ft
    )
    return await llm.call_text(
        model_cfg,
        system=system,
        user=user,
        schema=schema,
        parse=parse,
        node_id="root",
        step=step,
    )


def _from_spatial(out: OneShotSceneOutput) -> list[_PlacedObject]:
    return [
        _PlacedObject(o.name, o.name, o.world_bbox(), o.orientation)
        for o in out.objects
    ]


def _canvas_root(
    objects: list[_PlacedObject], width: float, depth: float
) -> BoundingBox:
    """The shared root: x/z pinned to the canvas footprint (square
    everywhere except v5's wide side-scroller level; v3-room's square
    scales with the slot), y hugging the built content (clamped to include
    the ground plane)."""
    lo_y = min(0.0, min(o.bbox.origin[1] for o in objects))
    hi_y = max(o.bbox.max_corner[1] for o in objects)
    return BoundingBox(
        origin=(-width / 2, lo_y, -depth / 2),
        dimensions=(width, round(hi_y - lo_y, 6), depth),
    )


def _canvas_floor(prompt: str, side: float = CANVAS_SIDE_FT) -> _PlacedObject:
    """The harness-placed ground slab: walkable top exactly at y = 0,
    spanning the whole canvas."""
    half = side / 2
    return _PlacedObject(
        "floor",
        f"flat rectangular ground floor slab for: {prompt}",
        BoundingBox(
            origin=(-half, -FLOOR_THICKNESS_FT, -half),
            dimensions=(side, FLOOR_THICKNESS_FT, side),
        ),
        0,
    )


ROOM_WALL_HEIGHT_FT = 8.0
ROOM_WALL_THICKNESS_FT = 1.0


def _level_backdrop(prompt: str) -> _PlacedObject:
    """v5's harness-placed backing slab — the side-scroller analog of the
    floor: a background wall spanning the whole level, top face at y = 0,
    that every flat object stands proud of."""
    w, d = V5_GRID_COLS * V5_CELL_FT, V5_GRID_ROWS * V5_CELL_FT
    return _PlacedObject(
        "background",
        f"flat 2d side-scroller level background wall for: {prompt}",
        BoundingBox(
            origin=(-w / 2, -FLOOR_THICKNESS_FT, -d / 2),
            dimensions=(w, FLOOR_THICKNESS_FT, d),
        ),
        0,
    )


def _room_shell(prompt: str, side: float) -> list[_PlacedObject]:
    """v3-room's harness-built shell: the ground slab plus four perimeter
    walls hugging the canvas edges (side walls inset between back and front
    so the corners don't overlap). The model only designs the interior."""
    half = side / 2
    t, h = ROOM_WALL_THICKNESS_FT, ROOM_WALL_HEIGHT_FT
    wall = f"flat plain interior wall of: {prompt}"

    def seg(wall_id: str, origin: tuple[float, float, float], dims: tuple[float, float, float]) -> _PlacedObject:
        return _PlacedObject(wall_id, wall, BoundingBox(origin=origin, dimensions=dims), 0)

    return [
        _canvas_floor(prompt, side),
        seg("wall back", (-half, 0.0, -half), (side, h, t)),
        seg("wall front", (-half, 0.0, half - t), (side, h, t)),
        seg("wall left", (-half, 0.0, -half + t), (t, h, side - 2 * t)),
        seg("wall right", (half - t, 0.0, -half + t), (t, h, side - 2 * t)),
    ]


def _is_wall(name: str) -> bool:
    return "wall" in name.lower()


def _segment_rects(cells: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    """Partition a set of grid cells into straight rectangular runs instead
    of one bounding box: maximal horizontal runs per row, merged vertically
    while consecutive rows share the exact same column span. A wall ring
    becomes its four sides, a stair diagonal one rect per step, a solid
    rectangular patch merges back into a single rect, and disjoint patches
    of the same name come out as separate rects."""
    by_row: dict[int, list[int]] = {}
    for r, c in cells:
        by_row.setdefault(r, []).append(c)
    rects: list[list[int]] = []  # [r0, c0, r1, c1]
    prev_row: dict[tuple[int, int], list[int]] = {}
    for r in sorted(by_row):
        spans: list[tuple[int, int]] = []
        cols = sorted(by_row[r])
        start = prev = cols[0]
        for c in cols[1:]:
            if c != prev + 1:
                spans.append((start, prev))
                start = c
            prev = c
        spans.append((start, prev))
        current: dict[tuple[int, int], list[int]] = {}
        for span in spans:
            rect = prev_row.get(span)
            if rect is not None and rect[2] == r - 1:
                rect[2] = r
            else:
                rect = [r, span[0], r, span[1]]
                rects.append(rect)
            current[span] = rect
        prev_row = current
    return [(r0, c0, r1, c1) for r0, c0, r1, c1 in rects]


def _normalize_walls(
    plan: GridPlanOutput, placed: list[_PlacedObject]
) -> list[_PlacedObject]:
    """Post-LLM deterministic normalization (v4, walls only, for ease of
    viewing): models reliably emit walls as a few giant blocks rather than
    proper sides. Drop every wall placement the conversion call produced
    and rebuild walls from the plan's wall patches as individual thin
    frames — runs straight from the matrix (`_segment_rects`), with the
    short axis thinned to the harness wall thickness (centered on the
    painted cells, since a cell spans several feet) and standing on the
    floor at the harness wall height. When the plan painted no walls there
    is nothing to rebuild from, and placements pass through untouched."""
    half_x = len(plan.grid[0]) * CELL_FT / 2
    half_z = len(plan.grid) * CELL_FT / 2
    by_name: dict[str, list[tuple[int, int]]] = {}
    for r, grid_row in enumerate(plan.grid):
        for c, name in enumerate(grid_row):
            if name and _is_wall(name):
                by_name.setdefault(name, []).append((r, c))
    counts: dict[str, int] = {}
    frames: list[_PlacedObject] = []
    for name, cells in by_name.items():
        for r0, c0, r1, c1 in _segment_rects(cells):
            k = counts.get(name, 0) + 1
            counts[name] = k
            x0, z0 = -half_x + c0 * CELL_FT, -half_z + r0 * CELL_FT
            w, d = (c1 - c0 + 1) * CELL_FT, (r1 - r0 + 1) * CELL_FT
            t = ROOM_WALL_THICKNESS_FT
            if w <= d:
                x0, w = x0 + (w - t) / 2, t
            else:
                z0, d = z0 + (d - t) / 2, t
            frames.append(
                _PlacedObject(
                    name if k == 1 else f"{name} {k}",
                    name,
                    BoundingBox(
                        origin=(x0, 0.0, z0),
                        dimensions=(w, ROOM_WALL_HEIGHT_FT, d),
                    ),
                    0,
                )
            )
    if not frames:
        return placed
    return [p for p in placed if not _is_wall(p.match_text)] + frames


def _is_unit(name: str) -> bool:
    n = name.lower()
    return any(t in n for t in V5_UNIT_TOKENS)


def render_placements(objects: list[_PlacedObject]) -> str:
    """A placement list in the object-CSV currency — what the deterministic
    solver hands downstream, shown verbatim by the dashboard's info panel."""
    lines = [_OBJECTS_CSV_HEADER]
    for o in objects:
        x, y, z = o.bbox.origin
        w, h, d = o.bbox.dimensions
        lines.append(f"{o.id},[{x:g},{y:g},{z:g}],[{w:g},{h:g},{d:g}],{o.orientation}")
    return "\n".join(lines)


def _v5_objects(plan: GridPlanOutput) -> list[_PlacedObject]:
    """v5's deterministic solver — the plan IS the level, with no second
    LLM call: cells map 1:1 onto the level plane (column -> x, row -> world
    z, V5_CELL_FT each) and every object is a V5_THICKNESS_FT-thin slab.
    How same-name neighbours fuse depends on the object's KIND: unit items
    (V5_UNIT_TOKENS — bricks, blocks, coins...) become one object PER CELL,
    everything else merges into one object per straight rectangular run
    (`_segment_rects`) — so a pipe patch is a single pipe, a ground sheet
    splits only at its pits, and a stair diagonal is one slab per step."""
    half_x = len(plan.grid[0]) * V5_CELL_FT / 2
    half_z = len(plan.grid) * V5_CELL_FT / 2
    by_name: dict[str, list[tuple[int, int]]] = {}
    for r, grid_row in enumerate(plan.grid):
        for c, name in enumerate(grid_row):
            if name:
                by_name.setdefault(name, []).append((r, c))
    counts: dict[str, int] = {}
    objects: list[_PlacedObject] = []
    for name, cells in by_name.items():
        rects = (
            [(r, c, r, c) for r, c in cells]
            if _is_unit(name)
            else _segment_rects(cells)
        )
        for r0, c0, r1, c1 in rects:
            k = counts.get(name, 0) + 1
            counts[name] = k
            objects.append(
                _PlacedObject(
                    name if k == 1 else f"{name} {k}",
                    name,
                    BoundingBox(
                        origin=(
                            -half_x + c0 * V5_CELL_FT,
                            0.0,
                            -half_z + r0 * V5_CELL_FT,
                        ),
                        dimensions=(
                            (c1 - c0 + 1) * V5_CELL_FT,
                            V5_THICKNESS_FT,
                            (r1 - r0 + 1) * V5_CELL_FT,
                        ),
                    ),
                    0,
                )
            )
    return objects


async def _design_scene(
    prompt: str, model_cfg: DllmModel, version: str, canvas_ft: float
) -> list[_PlacedObject]:
    """Run the version's design call(s) and reduce them to placements. No
    version authors the scene bounds — run() pins the root to the canvas
    (fixed everywhere except v3-room, which scales with the slot)."""
    if version == "v1-bbox":
        out_v1 = await _structured_step(
            model_cfg, version=version, stem="oneshot", step="oneshot_scene",
            schema=OneShotSceneOutputV1, prompt=prompt,
        )
        return [
            _PlacedObject(o.name, o.name, o.bbox, o.orientation)
            for o in out_v1.objects
        ]
    if version == "v2-spatial":
        out = await _text_step(
            model_cfg, version=version, stem="oneshot", step="oneshot_scene",
            schema=OneShotSceneOutput, parse=parse_objects_csv, prompt=prompt,
        )
        return _from_spatial(out)
    if version == "v3-room":
        out_room = await _text_step(
            model_cfg, version=version, stem="oneshot", step="oneshot_scene",
            schema=RoomSceneOutput, parse=parse_room_csv,
            prompt=prompt, canvas_ft=canvas_ft,
        )
        # Shell first: id collisions with model-emitted floor/walls dedup
        # in the harness's favor.
        return [*_room_shell(prompt, canvas_ft), *_from_room(out_room)]
    if version == "v4-grid-only":
        plan = await _text_step(
            model_cfg, version=version, stem="grid", step="oneshot_grid",
            schema=GridPlanOutput, parse=parse_name_grid, prompt=prompt,
        )
        grid_text = render_grid_plan(plan)
        # Surface the plan as its own event (the dashboard overlay shows it);
        # log_once keeps a grid-replayed resume from duplicating it.
        logging.log_once("oneshot.grid", match_fields=(), content=grid_text)
        out_room = await _text_step(
            model_cfg, version=version, stem="convert", step="oneshot_scene",
            schema=RoomSceneOutput, parse=parse_room_csv,
            prompt=prompt, grid_plan=grid_text,
        )
        placed = [_canvas_floor(prompt), *_from_room(out_room)]
        return _normalize_walls(plan, placed)
    if version == "v5-sidescroller":
        plan = await _text_step(
            model_cfg, version=version, stem="grid", step="oneshot_grid",
            schema=GridPlanOutput,
            parse=lambda c: parse_name_grid(
                c, cols=V5_GRID_COLS, n_rows=V5_GRID_ROWS
            ),
            prompt=prompt,
        )
        grid_text = render_grid_plan(plan)
        logging.log_once("oneshot.grid", match_fields=(), content=grid_text)
        placed = [_level_backdrop(prompt), *_v5_objects(plan)]
        # The deterministic solver is a real pipeline step — surface its
        # exact input (the plan) and output (the placements it hands to
        # meshing, backdrop included) for the info panel.
        logging.log_once(
            "oneshot.solver",
            match_fields=(),
            input=grid_text,
            output=render_placements(placed),
        )
        return placed
    raise ValueError(f"unknown oneshot version: {version}")


async def _place_one(
    obj: _PlacedObject,
    *,
    objs_dir: Path,
    out_root: Path,
    model_rotation: list[float] | None = None,
) -> None:
    """Materialize one object from the asset library: match its `match_text`
    to the closest catalog item (cheap retrieval LLM — not part of the
    benchmark surface), then bake the placement transform into the pre-built
    GLB so the compressed bytes are preserved. Mirrors the main pipeline's
    library path. `model_rotation` is a fixed reorientation baked into every
    placement (v5 stands assets upright)."""
    try:
        path = objs_dir / f"{obj.id}.glb"
        match = await library.match(obj.match_text, node_id=obj.id, zone_id="root")
        logging.log(
            "library.match",
            id=obj.id,
            prompt=obj.match_text,
            library_id=match.library_id,
        )
        asset = library.asset_path(match.library_id)
        # The catalog thumbnail doubles as the object's reference image so the
        # assets panel shows what was matched.
        ref_image = asset.with_suffix(".png")
        if ref_image.exists():
            dest_image = objs_dir / f"{obj.id}.png"
            await asyncio.to_thread(shutil.copy2, ref_image, dest_image)
            logging.log(
                "image",
                id=obj.id,
                url=_artifact_url(out_root, dest_image),
                prompt=obj.match_text,
            )
        if not asset.exists():
            logging.log(
                "mesh.error",
                id=obj.id,
                message=f"library asset missing on disk: {match.library_id}",
            )
            return
        bounds = library.asset_rotated_bounds(match.library_id, obj.orientation)
        if bounds is not None:
            rotated_min, rotated_max = bounds
            if model_rotation is not None:
                # Fold the fixed reorientation into the fill bounds so the
                # per-axis stretch still fills the box after the extra turn.
                rotated_min, rotated_max = glb_place.rotate_aabb(
                    rotated_min, rotated_max, model_rotation
                )
            await asyncio.to_thread(
                glb_place.place_glb,
                src=asset,
                dst=path,
                bbox=obj.bbox,
                orientation=obj.orientation,
                rotated_min=rotated_min,
                rotated_max=rotated_max,
                model_rotation=model_rotation,
            )
        else:
            # Asset missing from the bounds manifest: copy through unscaled so
            # the placement still renders rather than vanishing.
            await asyncio.to_thread(shutil.copyfile, asset, path)
            logging.log("library.bounds_missing", id=obj.id, library_id=match.library_id)
        logging.emit_model(obj.id, artifact_kind="object", url=_artifact_url(out_root, path))
    except Exception as e:
        logging.log("mesh.error", id=obj.id, message=f"{type(e).__name__}: {e}")


_pending: dict[str, list[asyncio.Task[None]]] = {}


async def await_pending(run_id: str) -> None:
    """Block until every background mesh task for this cell has finished.
    `_place_one` logs + swallows its own errors, so this only waits."""
    tasks = _pending.pop(run_id, [])
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def cancel_pending(run_id: str) -> None:
    for t in _pending.pop(run_id, []):
        t.cancel()


async def run(
    *,
    prompt: str,
    model_cfg: DllmModel,
    version: str,
    run_id: str,
    cell_dir: Path,
    out_root: Path,
    canvas_ft: float = CANVAS_SIDE_FT,
) -> None:
    placed = await _design_scene(prompt, model_cfg, version, canvas_ft)

    seen: set[str] = {"root"}
    objects: list[_PlacedObject] = []
    for obj in placed:
        if obj.id in seen:
            logging.log("oneshot.dedup_drop", id=obj.id)
            continue
        seen.add(obj.id)
        objects.append(obj)

    if version == "v3-room":
        width = depth = canvas_ft
    elif version == "v5-sidescroller":
        width = V5_GRID_COLS * V5_CELL_FT
        depth = V5_GRID_ROWS * V5_CELL_FT
    else:
        width = depth = CANVAS_SIDE_FT
    logging.emit_bbox(
        "root",
        _canvas_root(objects, width, depth),
        parent_id=None,
        prompt=prompt,
        kind="zone",
    )
    for obj in objects:
        logging.emit_bbox(
            obj.id,
            obj.bbox,
            parent_id="root",
            prompt=obj.match_text,
            kind="object",
            orientation=obj.orientation,
        )

    model_rotation = V5_UPRIGHT_ROTATION if version == "v5-sidescroller" else None
    objs_dir = cell_dir / "objects"
    objs_dir.mkdir(parents=True, exist_ok=True)
    pending = _pending.setdefault(run_id, [])
    for obj in objects:
        path = objs_dir / f"{obj.id}.glb"
        if path.exists():
            logging.emit_model(obj.id, artifact_kind="object", url=_artifact_url(out_root, path))
            continue
        logging.log("mesh.submit", id=obj.id, prompt=obj.match_text)
        pending.append(
            asyncio.create_task(
                _place_one(
                    obj, objs_dir=objs_dir, out_root=out_root,
                    model_rotation=model_rotation,
                ),
            ),
        )
