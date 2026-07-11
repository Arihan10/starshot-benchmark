"""Ablation experiment configuration + auto-naming.

An ablation is anchored to an existing BASE run (a completed cell). The harness
spawns auto-named VARIANT runs that INHERIT the base's prompt snapshot + scene
history and diverge only at the last-N firings of a target step kind, under a
treatment (scene-context shuffle method, XML tags on/off, prompt section order,
distractor injection).

This module is deliberately pure config — no server / pipeline / FastAPI
imports — so it can be unit-tested and shared by both the `/ablations` endpoint
and any headless orchestration script. Runtime application of a treatment lives
behind the `context` seam and the scene renderer (a later phase).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from itertools import product

# The four scene-item shuffling methods compared in the study. "order" is the
# current (insertion-order) baseline; the rest reorder the scene items fed to
# the model for the treated steps.
SHUFFLE_METHODS: tuple[str, ...] = ("order", "random", "distance", "raytrace")

# Coordinate-frame axis levels (a SINGLE enumerated axis — the input->output
# representation for the two bbox solvers; see app.ablation.coord). "baseline"
# is the current L/G->L behaviour and carries no run-name tag. The tags mirror
# ablationcore.js `coordTagOf` verbatim so a variant's name is identical whether
# it was launched from the wizard (JS) or this module. Duplicated here (not
# imported from `coord`) to keep this config module import-cycle-free.
COORD_MODES: tuple[str, ...] = ("baseline", "lg2g", "l2l", "g2g", "g2l")
_COORD_TAGS: dict[str, str] = {
    "lg2g": "crd-LG2G",
    "l2l": "crd-L2L",
    "g2g": "crd-G2G",
    "g2l": "crd-G2L",
}

# Scene-context SCHEMA axis levels (a SINGLE enumerated axis — how the treated
# step's scene context is SERIALIZED; see app.ablation.schema). "baseline" is the
# current soft-JSON and carries no run-name tag. Tags mirror ablationcore.js
# `schemaTagOf`. Duplicated here (not imported) to keep this config module
# import-cycle-free.
SCHEMA_MODES: tuple[str, ...] = ("baseline", "xml", "prose")
_SCHEMA_TAGS: dict[str, str] = {
    "xml": "sch-XML",
    "prose": "sch-PROSE",
}

# XML-gravity axis levels (a SINGLE enumerated axis — where the neutral <prompt>
# closing tag sits within the treated step's instruction block; see
# app.ablation.gravity). "baseline" is the untouched base cell (real
# VERY_IMPORTANT_INSTRUCTIONS tags) and carries no tag; "none" strips all tags
# (the experiment's own comparison anchor); q1..q4 place the closing tag after
# each word-count quarter. Tags mirror ablationcore.js `gravityTagOf`. Duplicated
# here (not imported) to keep this config module import-cycle-free.
GRAVITY_MODES: tuple[str, ...] = ("baseline", "none", "q1", "q2", "q3", "q4")
_GRAVITY_TAGS: dict[str, str] = {
    "none": "grav-none",
    "q1": "grav-Q1",
    "q2": "grav-Q2",
    "q3": "grav-Q3",
    "q4": "grav-Q4",
}


@dataclass(frozen=True)
class Treatment:
    """One point in the ablation matrix — what is applied to the divergent tail
    (the last-N firings of the target step kind) of an inherited variant run."""

    shuffle_method: str = "order"  # order | random | distance | raytrace
    xml_tags: bool = True          # keep vs strip prompt XML section tags
    section_order: str = "default"  # named permutation of prompt sections
    distractors: int = 0           # count of irrelevant objects injected
    # Coordinate-frame axis (single enumerated level; see app.ablation.coord):
    # baseline | lg2g | l2l | g2g | g2l. Drives the two bbox solvers' scene-context
    # INPUT representation (local / global / both) and OUTPUT frame (local / global).
    # A treatment identity axis (part of the run-name tag), NOT crossed with the rest.
    coord_mode: str = "baseline"
    # Scene-context SCHEMA axis (single enumerated level; see app.ablation.schema):
    # baseline (soft-JSON) | xml | prose. Re-renders the treated step's scene
    # context in that format (INPUT-only; the bbox output stays JSON). Also a
    # treatment identity axis, part of the run-name tag, NOT crossed with the rest.
    schema_mode: str = "baseline"
    # XML-gravity axis (single enumerated level; see app.ablation.gravity):
    # baseline (untouched) | none (tags stripped) | q1..q4 (neutral closing tag
    # after that word-count quarter). Rewrites the treated step's instruction
    # block at bind; a treatment identity axis, part of the run-name tag.
    gravity_mode: str = "baseline"
    seed: int = 0                  # RNG seed (random shuffle / distractor pick / sampling)
    # Sampling temperature for the RE-INFERRED treated step. None → the provider's
    # default. A positive value + a per-replicate `seed` is what makes duplicate
    # re-runs (replicates) INDEPENDENT draws — else a deterministic decode would
    # reproduce the same output and only fake-narrow the confidence interval. It's
    # NOT part of the run-name tag (a sampling knob, not a treatment identity).
    temperature: float | None = None

    def tag(self) -> str:
        """A short, filename-safe descriptor used in the auto-generated run name."""
        parts = [self.shuffle_method]
        if self.coord_mode and self.coord_mode != "baseline":
            parts.append(_COORD_TAGS.get(self.coord_mode, self.coord_mode))
        if self.schema_mode and self.schema_mode != "baseline":
            parts.append(_SCHEMA_TAGS.get(self.schema_mode, self.schema_mode))
        if self.gravity_mode and self.gravity_mode != "baseline":
            parts.append(_GRAVITY_TAGS.get(self.gravity_mode, self.gravity_mode))
        if not self.xml_tags:
            parts.append("noxml")
        if self.section_order != "default":
            parts.append(f"ord-{self.section_order}")
        if self.distractors:
            parts.append(f"d{self.distractors}")
        if self.seed:
            parts.append(f"s{self.seed}")
        return "_".join(parts)


@dataclass(frozen=True)
class AblationVariant:
    """A single variant run to launch: which base cell it inherits from, which
    step kind it treats, and the treatment."""

    base_run: str
    slot: str
    model: str
    target_step_kind: str
    last_n: int
    treatment: Treatment

    def run_name(self) -> str:
        return auto_name(self.base_run, self.target_step_kind, self.treatment)

    def runtime(self) -> "AblationRuntime":
        return AblationRuntime(target_step_kind=self.target_step_kind, treatment=self.treatment)


@dataclass(frozen=True)
class AblationRuntime:
    """The task-local slice the pipeline reads while a variant run executes: for
    calls whose step kind == target_step_kind, apply `treatment` to the scene it
    builds. `last_n` is not needed at run time — the inheritance CUT already
    isolates the treated tail (only post-cut target-kind calls run fresh)."""

    target_step_kind: str
    treatment: Treatment


@dataclass
class AblationSpec:
    """The full matrix: a base cell + the treatment axes to cross into variants."""

    base_run: str
    slot: str
    model: str
    target_step_kind: str
    last_n: int = 3
    shuffle_methods: list[str] = field(default_factory=lambda: list(SHUFFLE_METHODS))
    xml_variants: list[bool] = field(default_factory=lambda: [True])
    section_orders: list[str] = field(default_factory=lambda: ["default"])
    distractor_counts: list[int] = field(default_factory=lambda: [0])
    seeds: list[int] = field(default_factory=lambda: [0])

    def variants(self) -> list[AblationVariant]:
        """Cross the axes into concrete variants, de-duped by their auto name.

        A seed only matters when it actually drives randomness (the random
        shuffle or distractor sampling), so it is collapsed to 0 otherwise — that
        way `order`/`distance`/`raytrace` with no distractors don't fan out into
        identical-but-differently-named runs."""
        out: list[AblationVariant] = []
        for method, xml, order, dcount, seed in product(
            self.shuffle_methods,
            self.xml_variants,
            self.section_orders,
            self.distractor_counts,
            self.seeds,
        ):
            eff_seed = seed if (method == "random" or dcount) else 0
            treatment = Treatment(
                shuffle_method=method,
                xml_tags=xml,
                section_order=order,
                distractors=dcount,
                seed=eff_seed,
            )
            out.append(
                AblationVariant(
                    base_run=self.base_run,
                    slot=self.slot,
                    model=self.model,
                    target_step_kind=self.target_step_kind,
                    last_n=self.last_n,
                    treatment=treatment,
                )
            )
        deduped: dict[str, AblationVariant] = {}
        for v in out:
            deduped.setdefault(v.run_name(), v)
        return list[AblationVariant](deduped.values())


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text: str, limit: int) -> str:
    cleaned = _UNSAFE.sub("-", str(text)).strip("-.")
    return (cleaned[:limit] or "x").strip("-.") or "x"


def auto_name(base_run: str, target_step_kind: str, treatment: Treatment) -> str:
    """Deterministic, filesystem-safe variant run name that reads at a glance and
    can't collide across treatments — shape ``<base>__abl-<kind>-<tag>-<hash>``.

    `create_run` rejects names containing ``/``/``\\`` or starting with ``.``;
    `_slug` guarantees none of those, so the name is always a valid run id."""
    kind = _slug(target_step_kind, 24)
    tag = _slug(treatment.tag(), 32)
    digest = hashlib.sha1(f"{base_run}|{target_step_kind}|{tag}".encode()).hexdigest()[:6]
    return f"{_slug(base_run, 40)}__abl-{kind}-{tag}-{digest}"


def target_firings(steps: list[tuple[int, str]], target_step_kind: str, last_n: int) -> tuple[list[int], int | None]:
    """From an ordered ``[(event_index, step_kind)]`` list (e.g. built from the
    `tf-steps` timeline), return the last ``last_n`` event indices whose kind
    matches, plus the CUT event index (the first of those) to rewind before.
    ``([], None)`` when the kind never fires."""
    hits = [ev for ev, kind in steps if kind == target_step_kind]
    if not hits:
        return [], None
    chosen = hits[-max(1, last_n):]
    return chosen, chosen[0]


def variant_to_meta(variant: AblationVariant) -> dict:
    """Serialize a variant for persistence in the variant run's ``run.json``
    (under an ``ablation`` key) so a run is self-describing and re-collectable."""
    return {
        "base_run": variant.base_run,
        "slot": variant.slot,
        "model": variant.model,
        "target_step_kind": variant.target_step_kind,
        "last_n": variant.last_n,
        "treatment": asdict(variant.treatment),
    }


def treatment_from_meta(meta: dict) -> Treatment:
    t = dict((meta or {}).get("treatment") or {})
    temp = t.get("temperature")
    return Treatment(
        shuffle_method=t.get("shuffle_method", "order"),
        xml_tags=bool(t.get("xml_tags", True)),
        section_order=t.get("section_order", "default"),
        distractors=int(t.get("distractors", 0) or 0),
        coord_mode=str(t.get("coord_mode", "baseline") or "baseline"),
        schema_mode=str(t.get("schema_mode", "baseline") or "baseline"),
        gravity_mode=str(t.get("gravity_mode", "baseline") or "baseline"),
        seed=int(t.get("seed", 0) or 0),
        temperature=(float(temp) if temp is not None else None),
    )


# --- nested layout (runs/<base>/ablations/<experiment>/<variant>/) -----------
# A variant used to be a FLAT top-level run (`<base>__abl-...`); it now lives
# under its base run. These pure helpers derive the two path segments from the
# variant's `ablation` meta block (run.json) — the single source of truth (the
# old folder name was a lossy encoding). Kept here, import-cycle-free, so the
# `/runs/<base>/ablations` endpoint, `from-branches`, and the migration script
# all compute identical paths.

ABLATIONS_SUBDIR = "ablations"

# The treatment axes in priority order — MIRRORS ablationcore.js ABLATION_AXES so
# a variant's experiment bucket is identical whether derived here or in the UI.
# `field` reads the stored treatment; a value != `baseline` means "this axis is
# the one this variant varies". `id` is the experiment-folder name.
_EXPERIMENT_AXES: tuple[tuple[str, str, object], ...] = (
    ("coord", "coord_mode", "baseline"),
    ("schema", "schema_mode", "baseline"),
    ("gravity", "gravity_mode", "baseline"),
    ("shuffle", "shuffle_method", "order"),
    ("distractors", "distractors", 0),
    ("attend", "attend_target", ""),
)


def experiment_id(abl: dict) -> str:
    """The `<experiment>` folder for a variant: its explicit ``label`` if set,
    else the study inferred from the single axis its treatment varies (coord /
    schema / gravity / shuffle / distractors / attend). An all-baseline launched
    variant (e.g. the shuffle study's ``order`` / ``noxml`` arms — coord & schema
    baselines are the un-forked cell, so they never launch all-baseline) buckets
    under ``shuffle``. The manifest still carries the full treatment, so a UI can
    always regroup regardless of this human-facing folder."""
    label = str((abl or {}).get("label") or "").strip()
    if label:
        return _slug(label, 40)
    t = dict((abl or {}).get("treatment") or {})
    for exp_id, field, baseline in _EXPERIMENT_AXES:
        val = t.get(field)
        if val is not None and val != baseline:
            return exp_id
    return "shuffle"


def variant_id(abl: dict) -> str:
    """The `<variant>` folder: ``<target_step_kind>@<cut>-<treatment.tag>[-r<rep>]``
    (replicate omitted when 1). Reproduces today's flat-name suffix (minus the
    ``<base>__abl-`` prefix) because it reuses ``Treatment.tag()``."""
    kind = _slug(str((abl or {}).get("target_step_kind") or "step"), 24)
    tag = _slug(treatment_from_meta(abl).tag(), 40)
    cut = (abl or {}).get("cut")
    stem = f"{kind}@{cut}-{tag}" if cut is not None else f"{kind}-{tag}"
    rep = int((abl or {}).get("replicate") or 1)
    return f"{stem}-r{rep}" if rep > 1 else stem


def ablation_rel_path(abl: dict) -> str:
    """The variant's path RELATIVE to its base run dir:
    ``ablations/<experiment>/<variant>``."""
    return f"{ABLATIONS_SUBDIR}/{experiment_id(abl)}/{variant_id(abl)}"


def nested_run_id(abl: dict) -> str | None:
    """The variant's full logical run id (also its subpath under RUNS_DIR and its
    /artifacts URL segment): ``<base>/ablations/<experiment>/<variant>``. ``None``
    if the meta lacks a ``base_run`` (not a well-formed ablation)."""
    base = str((abl or {}).get("base_run") or "").strip()
    if not base:
        return None
    return f"{base}/{ablation_rel_path(abl)}"
