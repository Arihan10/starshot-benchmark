"""Attention ablation harness: config + auto-naming + the task-local treatment seam.

An ablation inherits from an existing base run and spawns auto-named variant runs
that diverge only at the last-N firings of a target step kind, under a treatment.
"""

from __future__ import annotations

from .config import (
    SHUFFLE_METHODS,
    AblationRuntime,
    AblationSpec,
    AblationVariant,
    Treatment,
    auto_name,
    target_firings,
    treatment_from_meta,
    variant_to_meta,
)
from .context import AblationComplete, clear, current, set_runtime, stop_if_treated_step

__all__ = [
    "SHUFFLE_METHODS",
    "AblationRuntime",
    "AblationSpec",
    "AblationVariant",
    "Treatment",
    "auto_name",
    "target_firings",
    "treatment_from_meta",
    "variant_to_meta",
    "AblationComplete",
    "clear",
    "current",
    "set_runtime",
    "stop_if_treated_step",
    "set_treatment",
]

# Back-compat alias in case callers use the verb-noun form.
set_treatment = set_runtime
