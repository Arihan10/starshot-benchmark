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


@dataclass(frozen=True)
class Treatment:
    """One point in the ablation matrix — what is applied to the divergent tail
    (the last-N firings of the target step kind) of an inherited variant run."""

    shuffle_method: str = "order"  # order | random | distance | raytrace
    xml_tags: bool = True          # keep vs strip prompt XML section tags
    section_order: str = "default"  # named permutation of prompt sections
    distractors: int = 0           # count of irrelevant objects injected
    seed: int = 0                  # RNG seed (random shuffle / distractor pick)

    def tag(self) -> str:
        """A short, filename-safe descriptor used in the auto-generated run name."""
        parts = [self.shuffle_method]
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
        return list(deduped.values())


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
    return Treatment(
        shuffle_method=t.get("shuffle_method", "order"),
        xml_tags=bool(t.get("xml_tags", True)),
        section_order=t.get("section_order", "default"),
        distractors=int(t.get("distractors", 0) or 0),
        seed=int(t.get("seed", 0) or 0),
    )
