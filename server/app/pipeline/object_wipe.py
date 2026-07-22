"""Complete, permanent removal of one object from a cell's source of truth.

Erases every reference to a node id from ALL event logs (the library build's
`events.jsonl` and EVERY generated version's `events.generated.jsonl`), reindexes
each so `index == line position` holds again, deletes the object's mesh / image
artifacts from every build directory (each version included), and repairs the two
structural invariants a bare delete would break:

  * ORPHANED CHILDREN — any object that was anchored to the wiped one is
    re-parented onto the wiped object's owning region (a zone). Placement bboxes
    are stored in WORLD coordinates in their `bbox` events, so re-anchoring is a
    pure metadata change: nothing moves.

  * PREFAB CANONICAL — in the generated build, if the wiped object was a prefab
    canonical (other objects reuse its mesh), the canonical role is handed to one
    of its reuses: that reuse inherits the shared raw Trellis mesh (cloned on
    disk) and the canonical's symmetry, and the remaining reuses are repointed to
    it. The flat prefab star is preserved and every member keeps its mesh.

The edit is destructive and deterministic — no LLM calls, no mesh work. The
caller MUST have already torn down the cell's pipeline / generate / regen tasks
and discarded its branches (branch fork indices are absolute and a reindex
invalidates them) before invoking `wipe_object`. This module is pure file +
log surgery and binds no SlotLog, so it never emits its own events.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from app.pipeline import generation
from app.services import prefabs
from app.utils.logging import SlotLog

# Keys whose value is a list of object/subregion spec dicts across the committed
# (`generation.decompose`, retry/`emitted` diagnostics) and observability
# (`cache.llm.output`) events. Each item may carry an `id` (drop when it IS the
# wiped node), a `parent` (re-point off the wiped node onto its region), and a
# relationships list (drop secondary edges to the wiped node).
_SPEC_LISTS = ("objects", "subregions", "children", "assignments", "emitted")
# `relationships` is the serialized alias; `referenced_ids` is the attribute name
# legacy logs were dumped under (ChildNodeSpec/SubregionSpec use populate_by_name).
_REL_KEYS = ("relationships", "referenced_ids")
_PARENT_KIND_KEYS = ("parent_relationship_kind", "parent_kind")

# Artifact name suffixes a node owns in any build dir: the served + raw + image,
# plus the atomic-write temporaries a crashed build can strand.
_ARTIFACT_SUFFIXES = (
    ".glb", ".raw.glb", ".png",
    ".glb.part", ".raw.glb.part", ".opt-tmp.glb",
)


def _scrub_spec(spec: Any, *, node_id: str, region_id: str | None) -> Any | None:
    """Returns None when `spec` IS the wiped node (caller drops it from its list),
    else a copy with edges to the wiped node severed: a `parent` that pointed at
    it is repointed onto `region_id` (anchored IN the owning zone), and any
    secondary relationship targeting it is dropped."""
    if not isinstance(spec, dict):
        return spec
    if spec.get("id") == node_id:
        return None
    out = dict(spec)
    if out.get("parent") == node_id and region_id is not None:
        out["parent"] = region_id
        for k in _PARENT_KIND_KEYS:
            if k in out:
                out[k] = "IN"
    for k in _REL_KEYS:
        rels = out.get(k)
        if isinstance(rels, list):
            out[k] = [
                r for r in rels
                if not (isinstance(r, dict) and r.get("target") == node_id)
            ]
    return out


def _scrub_output(output: Any, *, node_id: str, region_id: str | None) -> Any:
    """Strip the wiped node out of an `cache.llm` structured output: drop it from
    every id-bearing list and scrub the surviving siblings' edges, and drop a
    single `object` field that named it."""
    if not isinstance(output, dict):
        return output
    out = dict(output)
    for key in _SPEC_LISTS:
        lst = out.get(key)
        if isinstance(lst, list):
            scrubbed = (_scrub_spec(s, node_id=node_id, region_id=region_id) for s in lst)
            out[key] = [s for s in scrubbed if s is not None]
    # `prefab_match` output is a bare id list (`matches`), not spec dicts — drop
    # the wiped node from any sibling seed's match set.
    matches = out.get("matches")
    if isinstance(matches, list):
        out["matches"] = [m for m in matches if m != node_id]
    single = out.get("object")
    if isinstance(single, dict):
        if single.get("id") == node_id:
            out.pop("object", None)
        else:
            out["object"] = _scrub_spec(single, node_id=node_id, region_id=region_id)
    return out


def _edit_event(
    e: dict[str, Any], *, node_id: str, region_id: str | None, new_canonical: str | None,
) -> dict[str, Any]:
    """A SURVIVING event, with every reference to the wiped node repaired:
    embedded spec lists scrubbed, an orphaned child's `bbox.parent_id` repointed
    onto the region, and (generated log) a reuse of the wiped canonical repointed
    onto its successor."""
    kind = e.get("kind")
    out = dict(e)
    if kind == "cache.llm":
        out["output"] = _scrub_output(out.get("output"), node_id=node_id, region_id=region_id)
    else:
        for key in _SPEC_LISTS:
            lst = out.get(key)
            if isinstance(lst, list):
                scrubbed = (_scrub_spec(s, node_id=node_id, region_id=region_id) for s in lst)
                out[key] = [s for s in scrubbed if s is not None]
        single = out.get("object")
        if isinstance(single, dict) and single.get("id") != node_id:
            out["object"] = _scrub_spec(single, node_id=node_id, region_id=region_id)
    if kind == "bbox" and out.get("parent_id") == node_id and region_id is not None:
        out["parent_id"] = region_id
    if kind == "prefab.match" and new_canonical is not None and out.get("reuse_id") == node_id:
        # The successor's own row becomes a canonical (reuse_id ""); the other
        # reuses of the wiped canonical repoint onto the successor.
        out["reuse_id"] = "" if out.get("id") == new_canonical else new_canonical
    return out


def _transform(
    events: list[dict[str, Any]], *, node_id: str, region_id: str | None, new_canonical: str | None,
) -> list[dict[str, Any]]:
    """Drop every event that is ABOUT the wiped node, then repair the survivors.

    Dropped: the node's own per-node events (`id == node_id`), its resumable
    provider bookkeeping (`job_id == node_id`), its own LLM calls
    (`cache.llm` with `node == node_id`) plus their `llm.cost` rows, its `step`
    markers, and warnings that name it as a `source`.

    An `llm.cost` joins its call by `generation_id` (OpenRouter) or by the
    content-hash `key` (a token-priced compat backend), so both are collected
    from the dropped calls and both forms of cost row are removed."""
    dropped_gen_ids: set[str] = set()
    dropped_keys: set[str] = set()
    survivors: list[dict[str, Any]] = []
    for e in events:
        kind = e.get("kind")
        if kind == "cache.llm":
            drop = e.get("node") == node_id
        else:
            drop = (
                e.get("id") == node_id
                or e.get("job_id") == node_id
                or (kind == "step" and e.get("node") == node_id)
                or e.get("source") == node_id
            )
        if drop:
            if kind == "cache.llm":
                gid = e.get("generation_id")
                if isinstance(gid, str):
                    dropped_gen_ids.add(gid)
                key = e.get("key")
                if isinstance(key, str):
                    dropped_keys.add(key)
            continue
        survivors.append(e)

    def _is_dropped_cost(e: dict[str, Any]) -> bool:
        if e.get("kind") != "llm.cost":
            return False
        return e.get("generation_id") in dropped_gen_ids or e.get("key") in dropped_keys

    return [
        _edit_event(e, node_id=node_id, region_id=region_id, new_canonical=new_canonical)
        for e in survivors
        if not _is_dropped_cost(e)
    ]


def _region_of(events: list[dict[str, Any]], node_id: str) -> str | None:
    """The zone that OWNS the object — the `zone` of the generation pass that
    emitted it (`generation.decompose` listing it, or its `generation.next`).
    Orphaned children re-anchor here. Falls back to the object's own structural
    parent from its `bbox` event for legacy logs that lack the emitting event."""
    for e in events:
        kind = e.get("kind")
        if kind == "generation.decompose":
            objs = e.get("objects")
            if isinstance(objs, list) and any(
                isinstance(o, dict) and o.get("id") == node_id for o in objs
            ):
                zone = e.get("zone")
                if isinstance(zone, str):
                    return zone
        elif kind == "generation.next" and e.get("id") == node_id:
            zone = e.get("zone")
            if isinstance(zone, str):
                return zone
    for e in events:
        if e.get("kind") == "bbox" and e.get("id") == node_id:
            parent = e.get("parent_id")
            if isinstance(parent, str):
                return parent
    return None


def _latest_symmetry(events: list[dict[str, Any]], node_id: str) -> dict[str, Any] | None:
    """The node's CURRENT mirror state from its `symmetry.applied` history (latest
    wins): `{cut_plane, keep_positive?}` when mirrored, else None (a trailing
    `none` from an un-symmetrize means not mirrored). Carried onto a prefab
    successor so it re-derives the same mirror its served mesh already shows."""
    state: dict[str, Any] | None = None
    for e in events:
        if e.get("kind") != "symmetry.applied" or e.get("id") != node_id:
            continue
        cut_plane = e.get("cut_plane")
        if cut_plane in ("xy", "xz"):
            state = {"cut_plane": cut_plane}
            if isinstance(e.get("keep_positive"), bool):
                state["keep_positive"] = e["keep_positive"]
        else:
            state = None
    return state


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn tail line mid-flush
    return events


def _write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    """Reindex (`index == position`) and write atomically (temp + replace) so a
    crash can't leave the generated log half-rewritten."""
    for i, e in enumerate(events):
        e["index"] = i
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    os.replace(tmp, path)


def _clone_raw_to_successor(
    runs_dir: Path, run_id: str, version: str, old_id: str, new_id: str,
) -> bool:
    """Copy the wiped canonical's raw Trellis mesh (and its reference image when
    the successor lacks one) onto the successor WITHIN one generated version, so it
    becomes a self-sufficient canonical the version's group can still be re-derived
    from. Returns whether the raw was cloned (a missing source raw — e.g. an
    interrupted build, or a version where the node was a reuse — is skipped)."""
    raw_dir, _opt_dir = generation.generated_dirs(runs_dir, run_id, version)
    src_raw = raw_dir / f"{old_id}.raw.glb"
    if not src_raw.exists():
        return False
    shutil.copyfile(src_raw, raw_dir / f"{new_id}.raw.glb")
    src_png = raw_dir / f"{old_id}.png"
    dst_png = raw_dir / f"{new_id}.png"
    if src_png.exists() and not dst_png.exists():
        shutil.copyfile(src_png, dst_png)
    return True


def _delete_object_files(runs_dir: Path, run_id: str, node_id: str) -> list[str]:
    """Unlink the node's served/raw/image artifacts (and stranded temporaries)
    from every build dir: the library `objects/` (+ migrated `objects-optimized/`),
    a pre-versioning non-versioned generated build (defensive — normally already
    folded into v1), and EVERY generated version under `generated/<n>/`."""
    base = runs_dir / run_id
    dirs = [
        base / "objects",
        base / "objects-optimized",
        base / generation.GENERATED_RAW_SUBDIR,
        base / generation.GENERATED_OPT_SUBDIR,
    ]
    gen_root = base / generation.GENERATED_DIR
    if gen_root.is_dir():
        for version in gen_root.iterdir():
            if version.is_dir():
                dirs.append(version / generation.GENERATED_RAW_SUBDIR)
                dirs.append(version / generation.GENERATED_OPT_SUBDIR)
    deleted: list[str] = []
    for d in dirs:
        for suffix in _ARTIFACT_SUFFIXES:
            path = d / f"{node_id}{suffix}"
            try:
                if path.is_file():
                    path.unlink()
                    deleted.append(path.relative_to(runs_dir).as_posix())
            except OSError:
                pass
    return deleted


def wipe_object(
    *, node_id: str, library_log: SlotLog, runs_dir: Path, run_id: str,
) -> dict[str, Any] | None:
    """Permanently remove `node_id` from the cell. Returns a summary, or None when
    the cell holds no such object (no `bbox` event — the caller raises 404).

    The library log gates rendering (its `model` events drive both builds'
    `/meshes` bundles), so it is wiped first; EVERY generated version's log is wiped
    next, each handing off its own prefab canonical role if needed (versions group
    independently); files are deleted last (after each successor has cloned the raw
    it inherits)."""
    # Read the FULL events from disk (not the slim in-memory buffer, which drops
    # the cache.llm prompt/variables/reasoning bytes) so the rewrite preserves every
    # surviving event's heavy bytes; `replace_events` re-slims the resident copy.
    lib_events = _read_jsonl(library_log.events_path)
    if not any(e.get("kind") == "bbox" and e.get("id") == node_id for e in lib_events):
        return None

    region_id = _region_of(lib_events, node_id)

    # Library build never groups (no prefab.match), so no canonical handoff here.
    library_log.replace_events(
        _transform(lib_events, node_id=node_id, region_id=region_id, new_canonical=None)
    )

    # Wipe the node from EACH generated version's log independently — every version
    # owns its own prefab grouping, so the canonical handoff is resolved + applied
    # per version. Fold any pre-versioning build into v1 first so it's covered.
    generation.migrate_legacy_generated(runs_dir, run_id)
    new_canonicals: dict[str, str] = {}
    for version in generation.list_generated_versions(runs_dir, run_id):
        gen_path = generation.generated_events_path(runs_dir, run_id, version)
        if not gen_path.exists():
            continue
        gen_events = _read_jsonl(gen_path)
        canonical_id, reuse_ids = prefabs.resolve_group(gen_events, node_id)
        new_canonical: str | None = None
        if canonical_id == node_id and reuse_ids:
            new_canonical = reuse_ids[0]
            if _clone_raw_to_successor(runs_dir, run_id, version, node_id, new_canonical):
                new_canonicals[version] = new_canonical
        new_gen = _transform(
            gen_events, node_id=node_id, region_id=region_id, new_canonical=new_canonical,
        )
        if new_canonical is not None:
            mirror = _latest_symmetry(gen_events, node_id)
            if mirror is not None:
                new_gen.append({"kind": "symmetry.applied", "id": new_canonical, **mirror})
        _write_jsonl(gen_path, new_gen)

    deleted = _delete_object_files(runs_dir, run_id, node_id)
    return {
        "node_id": node_id,
        "region": region_id,
        "new_canonicals": new_canonicals,
        "library_events": len(library_log.state["events"]),
        "deleted_files": deleted,
    }
