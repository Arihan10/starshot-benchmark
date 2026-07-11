"""Per-base-run ablation manifest — ``runs/<base>/ablations/manifest.json``.

An index of every folded variant so the board / tf drawer discover ablations with
ONE read instead of scanning the whole top-level run list and filtering by name.

The manifest is a CACHE derived from each variant's ``run.json`` ``ablation`` block
(the source of truth) — never authoritative. ``read(..., rebuild_on_miss=True)``
re-derives it from disk when absent/stale, so it self-heals. Shared by the
``/runs/<base>/ablations`` endpoint, the ``from-branches`` writer, and the migration
script so all three agree on shape + paths.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from . import config as _cfg

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
RUN_META_NAME = "run.json"


def ablations_root(base_dir: Path) -> Path:
    return base_dir / _cfg.ABLATIONS_SUBDIR


def manifest_path(base_dir: Path) -> Path:
    return ablations_root(base_dir) / MANIFEST_NAME


def _read_meta(run_dir: Path) -> dict:
    try:
        data = json.loads((run_dir / RUN_META_NAME).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def entry(abl: dict, rel_path: str) -> dict:
    """One variant's manifest row: static, self-describing fields (dynamic status
    like done/running + attention-computed count is filled by the read endpoint,
    not persisted here). ``rel_path`` is relative to the base run dir, e.g.
    ``ablations/shuffle/child_bbox_batch@2107-distance``."""
    return {
        "id": _cfg.variant_id(abl),
        "path": rel_path,
        "run_id": f"{abl.get('base_run')}/{rel_path}" if abl.get("base_run") else rel_path,
        "experiment": _cfg.experiment_id(abl),
        "slot": abl.get("slot"),
        "model": abl.get("model"),
        "target_step_kind": abl.get("target_step_kind"),
        "cut": abl.get("cut"),
        "last_n": abl.get("last_n"),
        "replicate": abl.get("replicate", 1),
        "label": abl.get("label"),
        "treatment": abl.get("treatment") or {},
    }


def build(base_dir: Path) -> dict:
    """Re-derive the manifest by scanning the base's nested variant run.json files."""
    root = ablations_root(base_dir)
    variants: list[dict] = []
    if root.is_dir():
        for meta_path in sorted(root.rglob(RUN_META_NAME)):
            abl = _read_meta(meta_path.parent).get("ablation")
            if isinstance(abl, dict):
                rel = meta_path.parent.relative_to(base_dir).as_posix()
                variants.append(entry(abl, rel))
    return {
        "version": MANIFEST_VERSION,
        "base_run": base_dir.name,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "variants": variants,
    }


def write(base_dir: Path, manifest: dict | None = None) -> dict:
    manifest = manifest if manifest is not None else build(base_dir)
    root = ablations_root(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def refresh(base_dir: Path) -> dict:
    """Rebuild + persist — call after a variant is added or removed."""
    return write(base_dir, build(base_dir))


def read(base_dir: Path, *, rebuild_on_miss: bool = True) -> dict:
    """Load the manifest; rebuild from disk when absent/stale (self-healing)."""
    path = manifest_path(base_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("version") == MANIFEST_VERSION:
            return data
    except (OSError, json.JSONDecodeError):
        pass
    if rebuild_on_miss and ablations_root(base_dir).is_dir():
        return refresh(base_dir)
    return {"version": MANIFEST_VERSION, "base_run": base_dir.name, "updated_at": None, "variants": []}
