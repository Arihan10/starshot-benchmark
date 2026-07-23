"""Stage 1 — Scene assembler (the converter).

Turns one generated cell on disk into a normalized, validated **scene manifest**
that the rest of the splat pipeline (Stage 2 surfel sampler onward) consumes.

It is deliberately a *pure library*: `assemble_cell` takes explicit paths — the
server (`app.api.routes`) resolves a `(run, slot, model)` cell to those paths
(handling the legacy `generated/<n>/` layout) and calls in. Nothing here imports
the server.

Meshes are consumed AS-IS through `splat.assets.load_geoms`: vanilla glTF via
trimesh (full materials), KTX2/Meshopt sets via the in-process decoder (no
de-optimization step — geometry decodes natively; textures stay undecoded and
materials degrade to alphaMode/baseColorFactor stubs carrying the KTX2 pixel
size read from the image header).

Why no monolithic composed mesh: generation bakes every mesh into world space
(`rescale_mesh_to_bbox`), verified here — a mesh's world AABB equals its bbox
event. So "compose" is just load-at-identity + union, and we can process one
object at a time (peak ~0.1 GB) instead of holding the whole ~5.5 GB cell.

The manifest records the node tree, the derived scene AABB (the free-fly play
volume), and per-object geometry + material facts, plus warnings for anything
that doesn't line up (missing meshes, AABB drift). It is written as
`scene.json` under the cell's `splat/` dir and is safe to re-run (overwrites).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from splat.assets import load_geoms

MANIFEST_VERSION = 2
MANIFEST_NAME = "scene.json"

# A placed mesh's world AABB should equal its bbox event to within this (metres);
# beyond it we flag drift. Oblique yaws (±45/±135) deliberately inscribe a
# non-square footprint, so their under-fill is expected and not flagged.
AABB_TOL = 1e-2

# progress(done, total, current_id) — called after each object is processed.
ProgressCb = Callable[[int, int, str], None]

# Node kinds that carry a mesh (zones are pure containers — never meshed).
_MESHED_KINDS = ("frame", "object")


def read_scene_tree(events_path: Path) -> dict[str, dict[str, Any]]:
    """`{node_id: bbox_event}` from a cell's divider log (`events.jsonl`), in
    file order. Tolerates a torn final line written mid-flush."""
    nodes: dict[str, dict[str, Any]] = {}
    with events_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn tail line, or a non-JSON record
            if event.get("kind") == "bbox" and "id" in event:
                nodes[event["id"]] = event
    return nodes


def read_zone_meta(events_path: Path) -> tuple[dict[str, bool], dict[str, str]]:
    """Per-zone atomicity + per-node zone OWNERSHIP from the divider log.

    `zone_atomic` maps zone id → the divider's `is_atomic` verdict
    (`divider.zone_plan` events): True = a leaf place populated with objects,
    False = a container decomposed into sub-zones. `owner` maps a generated
    node id → the zone whose generation pass emitted it
    (`generation.decompose` / `generation.next` events) — the semantic
    "which place does this belong to" that the structural `parent_id` chain
    does not encode (a bed's parent is the carpet floor, which belongs to
    the shell zone, not the bedroom). Tolerates a torn final line."""
    zone_atomic: dict[str, bool] = {}
    owner: dict[str, str] = {}
    with events_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn tail line, or a non-JSON record
            kind = event.get("kind")
            if kind == "divider.zone_plan" and event.get("node"):
                zone_atomic[str(event["node"])] = bool(event.get("is_atomic"))
            elif kind == "generation.decompose" and event.get("zone"):
                zone = str(event["zone"])
                for obj in event.get("objects") or []:
                    oid = obj.get("id") if isinstance(obj, dict) else obj
                    if oid:
                        owner[str(oid)] = zone
            elif kind == "generation.next" and event.get("zone") and event.get("id"):
                owner[str(event["id"])] = str(event["zone"])
    return zone_atomic, owner


def placed_object_ids(raw_dir: Path) -> set[str]:
    """Ids with a placed, world-space mesh in `raw_dir` — the served `<id>.glb`,
    excluding the `<id>.raw.glb` pre-placement intermediates."""
    return {
        p.name[: -len(".glb")]
        for p in raw_dir.glob("*.glb")
        if not p.name.endswith(".raw.glb")
    }


def _bbox_corners(node: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """`(min_corner, max_corner)` for a bbox event. `dimensions` can be SIGNED —
    the divider sometimes anchors `origin` at a corner other than the minimum and
    lets a component go negative — so we order per-axis rather than assume
    `origin` is the min (which otherwise reads as spurious placement drift)."""
    origin = node.get("origin")
    dims = node.get("dimensions")
    if origin is None or dims is None:
        return None
    a = np.asarray(origin, dtype=float)
    b = a + np.asarray(dims, dtype=float)
    return np.minimum(a, b), np.maximum(a, b)


def _material_info(geoms: list[Any]) -> dict[str, Any] | None:
    """PBR facts from the first geometry that has a material — read WITHOUT
    decoding texels: vanilla files expose a PIL image whose `.size` reads only
    the header; KTX2 stubs carry `ktx2_texture_size` from the KTX2 header.
    `alpha_mode` drives later opacity init (OPAQUE → 1)."""
    for geom in geoms:
        material = getattr(getattr(geom, "visual", None), "material", None)
        if material is None:
            continue
        base = getattr(material, "baseColorTexture", None)
        ktx2 = getattr(material, "ktx2_texture_size", None)
        return {
            "alpha_mode": getattr(material, "alphaMode", None),
            "base_color": base is not None or ktx2 is not None,
            "normal": getattr(material, "normalTexture", None) is not None,
            "metallic_roughness": (
                getattr(material, "metallicRoughnessTexture", None) is not None
            ),
            "texture_size": (
                list(base.size) if base is not None
                else list(ktx2) if ktx2 is not None
                else None
            ),
        }
    return None


def _object_record(
    node_id: str, node: dict[str, Any], mesh_path: Path
) -> dict[str, Any]:
    """Load one placed mesh and collect geometry + material facts + AABB drift
    against its bbox event. Raises on an unreadable GLB (the caller records it as
    a warning and moves on)."""
    geoms = load_geoms(mesh_path)
    try:
        vertices = int(sum(len(g.vertices) for g in geoms))
        faces = int(sum(len(g.faces) for g in geoms))
        gb = np.array([g.bounds for g in geoms], dtype=float)  # world AABBs (baked)
        amin, amax = gb[:, 0].min(axis=0), gb[:, 1].max(axis=0)

        orientation = int(node.get("orientation", 0) or 0)
        corners = _bbox_corners(node)
        bbox_dev: float | None = None
        placed_ok = True
        if corners is not None:
            dev = max(
                float(np.abs(amin - corners[0]).max()),
                float(np.abs(amax - corners[1]).max()),
            )
            bbox_dev = round(dev, 5)
            # Oblique yaws inscribe a non-square footprint on purpose, so their
            # under-fill isn't drift; only flag axis-aligned placements.
            placed_ok = dev <= AABB_TOL or orientation % 90 != 0

        return {
            "id": node_id,
            "kind": node.get("node_kind"),
            "mesh": mesh_path.name,
            "vertices": vertices,
            "faces": faces,
            "aabb_min": [round(v, 5) for v in amin.tolist()],
            "aabb_max": [round(v, 5) for v in amax.tolist()],
            "bbox_dev": bbox_dev,
            "placed_ok": placed_ok,
            "orientation": orientation,
            "material": _material_info(geoms),
        }
    finally:
        # loaded geometries hold decoded arrays (+ PIL images on the vanilla
        # path); drop them before the next object.
        del geoms


def assemble_cell(
    *,
    run: str,
    slot: str,
    model: str,
    raw_dir: Path,
    events_path: Path,
    out_path: Path,
    runs_dir: Path | None = None,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Convert one cell → a scene manifest written to `out_path`, returning a
    compact summary (also embedded in the manifest as `summary`).

    Streams the placed meshes one at a time (peak ~one object in RAM). `progress`
    is called after each object with `(done, total, current_id)`.
    """
    if not events_path.exists():
        raise FileNotFoundError(f"scene tree not found: {events_path}")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"placed-mesh dir not found: {raw_dir}")

    nodes = read_scene_tree(events_path)
    zone_atomic, owner = read_zone_meta(events_path)
    placed = placed_object_ids(raw_dir)

    def _owning_zone(nid: str) -> str | None:
        """The zone a node belongs to: generation-event provenance first
        (authoritative — objects structurally parent to whatever supports
        them, across zones), else the nearest zone ancestor via `parent_id`.
        Zone nodes resolve to their ENCLOSING zone (None for root)."""
        got = owner.get(nid)
        if got is not None:
            return got
        seen: set[str] = set()
        cur = (nodes.get(nid) or {}).get("parent_id")
        while cur and cur not in seen:
            seen.add(cur)
            n = nodes.get(cur)
            if n is None:
                return None
            if n.get("node_kind") == "zone":
                return cur
            cur = n.get("parent_id")
        return None

    # Which nodes are meshable (frame/object) and actually have a mesh on disk.
    meshable = {
        nid for nid, n in nodes.items() if n.get("node_kind") in _MESHED_KINDS
    }
    have_mesh = sorted(meshable & placed, key=lambda nid: nodes[nid].get("index", 0))
    missing = sorted(meshable - placed)
    orphan_meshes = sorted(placed - set(nodes))  # a GLB with no bbox node

    # A node is "internal" if some other node lists it as its parent — it was
    # decomposed, so its children carry the geometry and it isn't meant to have
    # its own mesh. So a missing LEAF is a true hole (generation/optimization
    # failed for that item); a missing INTERNAL node is just a container.
    parent_ids = {n.get("parent_id") for n in nodes.values() if n.get("parent_id")}
    missing_holes = [nid for nid in missing if nid not in parent_ids]
    missing_containers = [nid for nid in missing if nid in parent_ids]

    total = len(have_mesh)
    if progress is not None:
        progress(0, total, "")

    warnings: list[str] = []
    objects: list[dict[str, Any]] = []
    total_vertices = 0
    total_faces = 0

    for done, node_id in enumerate(have_mesh, start=1):
        node = nodes[node_id]
        try:
            record = _object_record(node_id, node, raw_dir / f"{node_id}.glb")
            objects.append(record)
            total_vertices += record["vertices"]
            total_faces += record["faces"]
            if record["placed_ok"] is False:
                warnings.append(
                    f"{node_id}: world AABB drifts {record['bbox_dev']}m from its bbox"
                )
        except Exception as exc:  # a corrupt/unreadable GLB — keep going
            warnings.append(f"{node_id}: failed to load mesh ({type(exc).__name__}: {exc})")
        if progress is not None:
            progress(done, total, node_id)

    for node_id in missing_holes:
        warnings.append(f"{node_id}: {nodes[node_id].get('node_kind')} leaf has no mesh (hole)")
    for node_id in orphan_meshes:
        warnings.append(f"{node_id}: placed mesh has no bbox node")

    kind_counts = {"zone": 0, "frame": 0, "object": 0}
    for node in nodes.values():
        kind = node.get("node_kind")
        if kind in kind_counts:
            kind_counts[kind] += 1

    # Scene AABB (the free-fly play volume) from the ACTUAL placed-mesh world
    # AABBs — placement is baked into the vertices, so this is ground truth and
    # immune to bbox-metadata quirks. Falls back to the bbox union if, somehow,
    # nothing loaded.
    if objects:
        obj_min = np.min([o["aabb_min"] for o in objects], axis=0)
        obj_max = np.max([o["aabb_max"] for o in objects], axis=0)
        aabb = (obj_min.tolist(), obj_max.tolist())
    else:
        lo = np.array([np.inf, np.inf, np.inf])
        hi = np.array([-np.inf, -np.inf, -np.inf])
        for nid in have_mesh:
            corners = _bbox_corners(nodes[nid])
            if corners is not None:
                lo, hi = np.minimum(lo, corners[0]), np.maximum(hi, corners[1])
        aabb = (lo.tolist(), hi.tolist()) if np.isfinite(lo).all() else None

    root = _bbox_corners(nodes["root"]) if "root" in nodes else None
    have_mesh_set = set(have_mesh)

    tree = []
    for nid, n in nodes.items():
        corners = _bbox_corners(n)
        kind = n.get("node_kind")
        entry = {
            "id": nid,
            "kind": kind,
            "parent_id": n.get("parent_id"),
            "zone": _owning_zone(nid),
            "bbox_min": [round(v, 5) for v in corners[0].tolist()] if corners else None,
            "bbox_max": [round(v, 5) for v in corners[1].tolist()] if corners else None,
            "orientation": int(n.get("orientation", 0) or 0),
            "has_mesh": nid in have_mesh_set,
            "is_leaf": nid not in parent_ids,
        }
        if kind == "zone":
            # The divider's own verdict: True = an atomic place populated
            # with objects (Stage 4 gives it a panorama station), False = a
            # container of sub-zones (establishing views only). None = the
            # zone never got a plan event (very old logs).
            entry["is_atomic"] = zone_atomic.get(nid)
        tree.append(entry)

    manifest_rel = None
    if runs_dir is not None:
        try:
            manifest_rel = out_path.resolve().relative_to(runs_dir.resolve()).as_posix()
        except ValueError:
            manifest_rel = None

    summary = {
        "counts": {
            **kind_counts,
            "meshable": len(meshable),
            "placed": len(have_mesh),
            "missing_holes": len(missing_holes),
            "missing_containers": len(missing_containers),
            "orphan_meshes": len(orphan_meshes),
        },
        "scene_aabb": {"min": aabb[0], "max": aabb[1]} if aabb else None,
        "totals": {"vertices": total_vertices, "faces": total_faces},
        "warnings": len(warnings),
        "manifest": manifest_rel,
    }

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": run,
        "slot": slot,
        "model": model,
        "source_raw_dir": (
            raw_dir.resolve().relative_to(runs_dir.resolve()).as_posix()
            if runs_dir is not None and raw_dir.resolve().is_relative_to(runs_dir.resolve())
            else str(raw_dir)
        ),
        "units": {"length": "meter", "up": "Y", "handedness": "right"},
        "scene_aabb": summary["scene_aabb"],
        "root_bbox": (
            {"min": root[0].tolist(), "max": root[1].tolist()} if root else None
        ),
        "counts": summary["counts"],
        "totals": summary["totals"],
        "nodes": tree,
        "objects": objects,
        "warnings": warnings,
        "summary": summary,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    tmp.replace(out_path)

    return summary
