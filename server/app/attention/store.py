"""Build the analysis input from a tf-export, and store the SPARSE result.

Only sparse summaries are persisted (per-token top-k entities/attributes +
scalars) — never dense attention tensors — under the cell dir at
`attention/<event_index>.json`, served through the normal /artifacts route.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from app.attention.schema import (
    ANALYSIS_VERSION,
    BBOX_TEMPLATES,
    TO_PLACE_VERSION,
    AnalysisResult,
    GenerationTrace,
)


def _needs_to_place_recompute(meta: dict[str, Any]) -> bool:
    """True when a step CARRIES a to-place batch but its stored to-place readout is
    missing or from an older TO_PLACE_VERSION — the TARGETED recompute trigger, so
    only to-place-bearing steps go stale when the to-place analysis changes.

    Works on either a full `meta` dict or the cheap `_meta_flags` dict (same keys).
    Legacy results (pre-to-place, no `to_place_present`) fall back to the template
    heuristic so existing bbox-batch results recompute once to gain the readout."""
    present = meta.get("to_place_present")
    if present is None:  # legacy result — infer from the step template
        present = meta.get("template") in BBOX_TEMPLATES
    return bool(present) and meta.get("to_place_version") != TO_PLACE_VERSION


def is_fresh(cached: dict[str, Any] | None, input_key: Any, *, min_heads: int = 0) -> bool:
    """Whether a stored analysis can be REUSED as-is: it must be REAL (not mock),
    match the step's current content (`input_key`), and be from the current
    analysis version. Mock or stale-version results are recomputed.

    `min_heads` lets a caller demand at least that many per-token instrumented
    (layer, head) pairs: a result computed with fewer heads than now requested is
    treated as stale so raising the top-N recomputes (lowering it reuses). Results
    predating head-count tracking are assumed to hold the legacy default (4).

    A to-place-bearing step (bbox batch) is also stale when its to-place readout is
    missing/old — so only those steps recompute for the to-place feature."""
    if not cached:
        return False
    m = cached.get("meta", {})
    if m.get("mock") or m.get("input_key") != input_key or m.get("analysis_version") != ANALYSIS_VERSION:
        return False
    if min_heads:
        stored = m.get("max_heads")
        stored = 4 if stored is None else stored  # legacy results held top-4
        if stored < min_heads:
            return False
    if _needs_to_place_recompute(m):
        return False
    return True


def trace_from_export(
    export: dict[str, Any],
    *,
    remote_logprobs: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> GenerationTrace:
    """Wrap a `teacher_forcing.build_export` payload as the worker's input.
    Everything the worker needs (native sequence, completion boundary, frames,
    scene/output char-span maps) is already in the export; the remote logprobs
    sidecar rides along for the round-trip comparison."""
    return GenerationTrace(
        model_id=export["meta"]["model_id"],
        full_text=export["text"]["full"],
        completion_start=export["boundaries"]["completion_start"],
        frames=export.get("frames", {}),
        scene_map=export.get("scene_map", []),
        output_map=export.get("output_map", []),
        to_place_map=export.get("to_place_map", []),
        variables_map=export.get("variables_map", []),
        remote_logprobs=remote_logprobs,
        meta={
            "run": export["meta"].get("run"),
            "slot": export["meta"].get("slot"),
            "model": export["meta"].get("model"),
            "event_index": export["meta"].get("event_index"),
            "step": export["meta"].get("step"),
            "template": export["meta"].get("template"),
            "node": export["meta"].get("node"),
            "schema": export["meta"].get("schema"),
            **(extra_meta or {}),
        },
    )


def analysis_path(cell_dir: Path, event_index: int) -> Path:
    return cell_dir / "attention" / f"{event_index}.json"


# Cache of (path -> (mtime, flags)) so the status endpoint doesn't re-read files
# on every poll — only re-reads when a file changes. `flags` carries the mtime-
# stable freshness inputs (mock, version, template, to-place version/presence).
_META_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_MOCK_RE = re.compile(r'"mock"\s*:\s*(true|false)')
_VER_RE = re.compile(r'"analysis_version"\s*:\s*(-?\d+)')
_TPL_RE = re.compile(r'"template"\s*:\s*("(?:[^"\\]|\\.)*"|null)')
_TPV_RE = re.compile(r'"to_place_version"\s*:\s*(-?\d+|null)')
_TPP_RE = re.compile(r'"to_place_present"\s*:\s*(true|false)')
_MH_RE = re.compile(r'"max_heads"\s*:\s*(-?\d+|null)')


def _meta_flags(path: Path) -> dict[str, Any]:
    """Freshness inputs read cheaply from the file head (meta is the first object)
    and mtime-cached: `{mock, version, template, to_place_version, to_place_present}`.
    Falls back to a full parse only when the head regex misses (older files)."""
    empty = {"mock": False, "version": None, "template": None, "to_place_version": None, "to_place_present": None, "max_heads": None}
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return empty
    cached = _META_CACHE.get(str(path))
    if cached is not None and cached[0] == mtime:
        return cached[1]
    flags = dict(empty)
    try:
        with path.open("r", encoding="utf-8") as f:
            head = f.read(8192)
        mm, vm = _MOCK_RE.search(head), _VER_RE.search(head)
        if mm is not None and vm is not None:
            flags["mock"] = mm.group(1) == "true"
            flags["version"] = int(vm.group(1))
            tpl, tpv, tpp, mh = _TPL_RE.search(head), _TPV_RE.search(head), _TPP_RE.search(head), _MH_RE.search(head)
            if tpl is not None:
                flags["template"] = None if tpl.group(1) == "null" else json.loads(tpl.group(1))
            if tpv is not None and tpv.group(1) != "null":
                flags["to_place_version"] = int(tpv.group(1))
            if tpp is not None:
                flags["to_place_present"] = tpp.group(1) == "true"
            if mh is not None and mh.group(1) != "null":
                flags["max_heads"] = int(mh.group(1))
        else:  # pre-version / edge files — parse fully (cached after)
            meta = json.loads(path.read_text()).get("meta", {})
            flags["mock"] = bool(meta.get("mock"))
            flags["version"] = meta.get("analysis_version") if isinstance(meta.get("analysis_version"), int) else None
            flags["template"] = meta.get("template")
            flags["to_place_version"] = meta.get("to_place_version")
            flags["to_place_present"] = meta.get("to_place_present")
            mh = meta.get("max_heads")
            flags["max_heads"] = mh if isinstance(mh, int) else None
    except (OSError, json.JSONDecodeError):
        flags = dict(empty)
    _META_CACHE[str(path)] = (mtime, flags)
    return flags


def list_status(cell_dir: Path, *, min_heads: int = 0) -> dict[str, list[int]]:
    """Partition stored analyses into `fresh` (REAL + current analysis version, and
    — for to-place-bearing steps — current TO_PLACE_VERSION) and `stale` (mock, an
    older version, or a bbox-batch step missing the current to-place readout).
    When `min_heads` > 0, results computed with fewer instrumented heads count as
    stale so raising the UI head budget shows them as needing recompute."""
    d = cell_dir / "attention"
    if not d.is_dir():
        return {"fresh": [], "stale": []}
    fresh: list[int] = []
    stale: list[int] = []
    for p in d.glob("*.json"):
        try:
            ev = int(p.stem)
        except ValueError:
            continue
        f = _meta_flags(p)
        ok = (not f["mock"]) and f["version"] == ANALYSIS_VERSION and not _needs_to_place_recompute(f)
        if min_heads:
            stored = f.get("max_heads")
            stored = 4 if stored is None else stored  # legacy results held top-4
            if stored < min_heads:
                ok = False
        (fresh if ok else stale).append(ev)
    return {"fresh": sorted(fresh), "stale": sorted(stale)}


def list_computed(cell_dir: Path) -> list[int]:
    """Event indices with a FRESH (current-version, real) analysis."""
    return list_status(cell_dir)["fresh"]


def save(cell_dir: Path, event_index: int, result: AnalysisResult) -> Path:
    return save_dict(cell_dir, event_index, result.to_dict())


def save_dict(cell_dir: Path, event_index: int, data: dict[str, Any]) -> Path:
    """Persist an already-serialized result (e.g. one returned by the Modal
    worker). A RECOMPUTE atomically REPLACES the old version: we write a temp
    file then rename over the target, so there's never a partial/mixed file or a
    stale remnant of the previous version — and we drop the cached mock/version
    flags so the status re-reads the new file on the next poll (no confusion
    between the old and new analysis)."""
    path = analysis_path(cell_dir, event_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)  # atomic swap of old -> new
    finally:
        tmp.unlink(missing_ok=True)  # no-op after a successful replace
    _META_CACHE.pop(str(path), None)  # force a fresh mock/version read next status poll
    return path


def load(cell_dir: Path, event_index: int) -> dict[str, Any] | None:
    path = analysis_path(cell_dir, event_index)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def delete(cell_dir: Path, event_index: int) -> None:
    """Remove a stored analysis. Used to drop a STALE result before recomputing
    it, so a worker failure can't resurrect the outdated version — the step then
    reads as not-computed/failed rather than silently reverting to old data."""
    path = analysis_path(cell_dir, event_index)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    _META_CACHE.pop(str(path), None)
