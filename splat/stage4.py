"""Stage 4 — Camera planner (zone-driven dense field; no coverage optimization).

Plans the single-shot reference cameras Stage 5 renders and the splat trainer
consumes. The design replicates the Atlux capture workflow (dense, fixed-count,
heavily-overlapping DSLR shots) with the human artist replaced by the scene's
own construction record: THE ZONE TREE PROPOSES, THE GEOMETRY DISPOSES.

  * ZONES PROPOSE. The divider's zone tree (Stage-1 `scene.json`, enriched
    with per-zone `is_atomic`) says where content lives and how it nests:
      - every zone contributes a HALTON FILL of its bbox — positions spread
        low-discrepancy through the zone's air, camera count proportional to
        the VISIBLE SURFACE (skin bricks) inside its bbox, so budget follows
        content, not empty volume;
      - every ATOMIC zone (a leaf place populated with objects) additionally
        gets outward-facing FIBONACCI STATIONS — the panorama rig an artist
        would drop in each room — at its highest-clearance interior points,
        gated by ENCLOSURE (a zone that doesn't wrap the camera, e.g. a small
        open field, takes no station: its content is convex and the inward
        fill already covers it), with station count growing with zone extent;
      - the ROOT gets an inward-facing Fibonacci SHELL outside the scene —
        the establishing/orbit layer. Interior scenes occlude most of it into
        harmless exterior views of the building shell; open scenes get the
        far standpoints a zoomed-out viewer renders from.
  * GEOMETRY DISPOSES. Every proposal passes the same scene-agnostic filters
    against the Stage-2 grid: position in reachable EMPTY air, standoff from
    cover (probe shells), near-surface band (no cameras floating in dead air
    far from all content), and a global lattice dedupe so overlapping zone
    bboxes (they are NOT a partition) can't double-spend. Directions come
    from a global low-discrepancy sphere sequence — a smooth, non-directed
    field (content-aimed cameras were deliberately rejected: they starve the
    connective tissue and shatter view overlap) — with a POINT-BLANK CULL
    that re-aims or drops frames staring into nearby cover.

There is deliberately NO greedy set-cover, NO per-surface demand accounting,
NO cube faces, and NO CUDA: redundancy is the coverage guarantee (that is
what makes the Atlux recipe work), each camera is ONE pinhole shot, and every
count is a fixed feed-forward function of the scene. Offline coverage
verification lives in scenebench, not here.

Consumes `freespace.npz` (Stage 2) + `scene.json` (Stage 1). Emits
`cameras.json` (plan_version 2): shared pinhole `intrinsics` + a flat
`cameras` list, each `{pos, forward, up, kind: fill|station|shell, zone}`.
Stage 5 renders one image per entry; poses convert via `opencv_c2w`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from splat.stage2 import FreeSpace, load_free_space

logging.getLogger("trimesh").setLevel(logging.ERROR)

CAMERAS_NAME = "cameras.json"
PLAN_VERSION = 2

# Brick edge of the Stage-2 skin/zone tiers, in fine voxels (stage2._BRICK).
_BRICK = 8

# Fill-camera standoff probes: a position is valid when `occupied` is clear at
# the point and on two probe shells (26 directions × {r/2, r}) — a cheap,
# grid-native "at least ~r from any surface" test.
_CLEAR_DIRS = np.array(
    [o for o in np.ndindex(3, 3, 3) if o != (1, 1, 1)], dtype=np.float64
) - 1.0
_CLEAR_DIRS /= np.linalg.norm(_CLEAR_DIRS, axis=1, keepdims=True)

# Point-blank cull: a frame whose central ray hits opaque cover closer than
# this is a wall close-up; the planner re-aims it (antipodal, then two
# orthogonal directions) and drops the camera if nothing survives.
_MIN_VIEW_M = 0.35
# Ray-march step, × pitch (1.0 = every voxel; cover is gap-free by build).
_MARCH_STEP = 1.0

# Station (outward Fibonacci ball) shape: directions per station, and the
# enclosure gate — the fraction of UPPER-HEMISPHERE probe rays that must hit
# cover within 1.2 × the zone diagonal for the zone to count as "wrapping the
# camera". Upper hemisphere only: near any standpoint the ground itself wraps
# the bottom half of the sphere, so a full-sphere fan scores ≥ 0.5 in an open
# field and the gate would never fire — walls and ceilings are what make a
# place concave, and they live above the horizon.
_STATION_DIRS_DEFAULT = 48
_ENCLOSURE_DIRS = 64
_ENCLOSURE_MIN = 0.35
_ENCLOSURE_MIN_Y = -0.05  # keep a thin below-horizon band (distant walls)
# Enclosure is a LOCAL property: a wall must sit within panorama range to
# wrap a standpoint. Without the cap, huge open zones (a 250 m desert) count
# horizon terrain as "walls" and every dune field earns a panorama.
_ENCLOSURE_REACH_MAX = 25.0
# One station per this much zone extent (max axis, metres), capped.
_STATION_EVERY_M = 8.0
_STATION_MAX = 4
# Stations must be able to stand somewhere at least this clear.
_STATION_MIN_CLEAR = 0.45

# Shell (root establishing orbit): positions are Fibonacci directions
# projected onto the scene AABB inflated by a PER-AXIS standoff — the shell
# adapts to the scene's shape (a flat platformer gets broad-side coverage at
# a sane distance instead of a sphere at its 110 m diagonal). The standoff
# is 0.6× the axis's own half-extent, floored by BOTH an absolute minimum
# (must exceed the Stage-2 grid margin, 1.5 m default, so every position is
# provably outside the grid — open air, no filters needed) and a fraction of
# the LARGEST half-extent: a thin axis still needs establishing range — a
# camera 3 m off a 110 m level's broad face frames a ~4% sliver and the
# level-scale supervision the shell exists for disappears (measured: the
# platformer's unseen jumped 0.85%→5.7% without this floor).
# `_SHELL_RADIUS_MULT` survives only in the far-plane bound. Below-floor
# positions are culled (no under-terrain backface views).
_SHELL_STANDOFF_FRAC = 0.6
_SHELL_STANDOFF_MAXAXIS_FRAC = 0.3
_SHELL_STANDOFF_MIN = 3.0
_SHELL_RADIUS_MULT = 0.9
_SHELL_FLOOR_FRAC = 0.05

ProgressCb = Callable[[int, int, str], None]


@dataclass(frozen=True)
class PlanParams:
    """Stage-4 knobs — every count is feed-forward from these.

    `fill_per_skin_brick` is THE density dial: fill cameras per Stage-2 skin
    brick (a 24 cm cube touching visible surface), i.e. cameras per unit of
    visible content. ~0.05 lands a hotel room around 500 fill views. Scene-
    size independent and content-proportional by construction. `min_fill` /
    `max_fill` clamp the total (the cost dial); allocation across zones stays
    proportional to each zone's skin share."""

    fill_per_skin_brick: float = 0.05
    min_fill: int = 96
    # Benchmarked across the taxonomy: 1600 binds on every scene ≥ ~30 m and
    # measurably starves coverage (shooter seen≥3 77→82%, desert 79→86% at
    # 4000). 4000 leaves room-scale scenes untouched (they want ~500) and
    # keeps the largest plans at a few minutes of render time.
    max_fill: int = 4000
    # Fill positions must sit in air at least this clear of any surface …
    clear_m: float = 0.25
    # … and within this distance of SOME visible surface (kills dead-air
    # cameras floating high above open scenes; generous enough to keep
    # room-center views everywhere indoors).
    near_surface_m: float = 4.0
    # Global dedupe lattice for fill positions (≈ the old candidate spacing).
    fill_spacing: float = 0.45
    # Station / shell counts.
    station_dirs: int = _STATION_DIRS_DEFAULT
    shell_views: int = 128
    # Shared pinhole intrinsics: one FOV for every camera (DSLR-like, better
    # pixel utilization than the old 90° cube faces), square frames.
    fov_deg: float = 70.0
    render_resolution: int = 1024
    seed: int = 0

    def as_summary(self) -> dict[str, Any]:
        return {
            "fill_per_skin_brick": self.fill_per_skin_brick,
            "min_fill": self.min_fill,
            "max_fill": self.max_fill,
            "clear_m": self.clear_m,
            "near_surface_m": self.near_surface_m,
            "fill_spacing": self.fill_spacing,
            "station_dirs": self.station_dirs,
            "shell_views": self.shell_views,
            "fov_deg": self.fov_deg,
            "render_resolution": self.render_resolution,
            "seed": self.seed,
        }


# --- low-discrepancy sequences ---------------------------------------------------


def _radical_inverse(idx: np.ndarray, base: int) -> np.ndarray:
    """Van der Corput radical inverse of integer `idx` in `base` (float64)."""
    idx = idx.astype(np.int64).copy()
    out = np.zeros(idx.shape, dtype=np.float64)
    f = 1.0 / base
    while (idx > 0).any():
        out += f * (idx % base)
        idx //= base
        f /= base
    return out


def _halton3(n: int, start: int = 0) -> np.ndarray:
    """(n,3) Halton points in the unit cube (bases 2, 3, 5), offset by `start`
    so distinct consumers draw disjoint, still low-discrepancy runs."""
    idx = np.arange(start + 1, start + n + 1, dtype=np.int64)
    return np.stack(
        [_radical_inverse(idx, 2), _radical_inverse(idx, 3), _radical_inverse(idx, 5)],
        axis=1,
    )


def _fib_sphere(n: int) -> np.ndarray:
    """(n,3) near-uniform unit directions — the golden-angle spiral."""
    i = np.arange(n, dtype=np.float64)
    phi = (1.0 + 5.0**0.5) / 2.0
    z = 1.0 - (2.0 * i + 1.0) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = 2.0 * np.pi * i / phi
    return np.stack([r * np.cos(theta), z, r * np.sin(theta)], axis=1)


# --- grid-native geometric filters ------------------------------------------------


def _clear_at(fs: FreeSpace, pts: np.ndarray, r: float) -> np.ndarray:
    """True per point when it sits in reachable EMPTY air with no surface
    (opaque OR glass) within ~`r`: the centre cell must be EMPTY and three
    probe shells (26 dirs × {r/3, 2r/3, r}) must be unoccupied — three shells
    so a thin slab (a 10 cm ceiling) can't slip between consecutive radii."""
    ok = fs.empty_at(pts)
    for rad in (r / 3.0, 2.0 * r / 3.0, r):
        for k in range(0, len(_CLEAR_DIRS), 13):  # chunk: 13 dirs × N points
            d = _CLEAR_DIRS[k : k + 13]
            if not ok.any():
                return ok
            live = np.nonzero(ok)[0]
            probes = pts[live][:, None, :] + d[None, :, :] * rad
            hit = fs.occupied(probes.reshape(-1, 3)).reshape(len(live), -1)
            ok[live[hit.any(axis=1)]] = False
    return ok


def _first_hit(
    fs: FreeSpace, origins: np.ndarray, dirs: np.ndarray, t_max: float
) -> np.ndarray:
    """Distance to the first OPAQUE cover cell along each ray (∞ = clear to
    `t_max`). Voxel-stepped march over the sparse occlusion set; only samples
    inside the grid can hit (outside the grid is provably open air)."""
    n = len(origins)
    out = np.full(n, np.inf, dtype=np.float64)
    if n == 0:
        return out
    step = fs.pitch * _MARCH_STEP
    n_steps = max(1, int(ceil(t_max / step)))
    alive = np.arange(n)
    # Chunk the step axis so the transient point buffer stays bounded.
    t = step
    for _ in range(n_steps):
        if not len(alive):
            break
        pts = origins[alive] + dirs[alive] * t
        hit = fs.occluding(pts)
        if hit.any():
            out[alive[hit]] = t
            alive = alive[~hit]
        t += step
    return out


def _skin_centers(fs: FreeSpace) -> np.ndarray:
    """(S,3) world centres of the Stage-2 skin bricks — the 24 cm cubes that
    touch visible surface. The planner's proxy for WHERE CONTENT IS: fill
    budget allocation, the near-surface band, and station scoring all read it."""
    bd = fs.bdims
    lin = fs.skin_lin
    bx = lin // (bd[1] * bd[2])
    by = (lin // bd[2]) % bd[1]
    bz = lin % bd[2]
    cells = np.stack([bx, by, bz], axis=1).astype(np.float64) * _BRICK + _BRICK / 2.0
    return fs.origin + cells * fs.pitch


def _thin_lattice(points: np.ndarray, spacing: float) -> np.ndarray:
    """Indices of one winner per `spacing` lattice cell (nearest cell centre) —
    the global dedupe that keeps overlapping zone fills from double-spending."""
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    cell = np.floor(points / spacing).astype(np.int64)
    loc = cell - cell.min(axis=0)
    ny = int(loc[:, 1].max()) + 1
    nz = int(loc[:, 2].max()) + 1
    gid = (loc[:, 0] * ny + loc[:, 1]) * nz + loc[:, 2]
    center = (cell.astype(np.float64) + 0.5) * spacing
    d2 = ((points - center) ** 2).sum(axis=1)
    order = np.lexsort((d2, gid))
    gid_s = gid[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = gid_s[1:] != gid_s[:-1]
    return order[first]


# --- the zone tree ----------------------------------------------------------------


@dataclass(frozen=True)
class Zone:
    id: str
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    is_atomic: bool

    @property
    def diag(self) -> float:
        return float(np.linalg.norm(self.bbox_max - self.bbox_min))

    @property
    def max_extent(self) -> float:
        return float((self.bbox_max - self.bbox_min).max())


def _read_zones(scene_path: Path) -> tuple[list[Zone], np.ndarray, np.ndarray]:
    """Zones (with atomicity) + the scene AABB from a Stage-1 manifest.
    `is_atomic` falls back to "has no child zones" for pre-enrichment
    manifests. A scene with no zone nodes degrades to one atomic root zone
    over the scene AABB — the planner never NEEDS the tree, it only spends
    better with it."""
    doc = json.loads(Path(scene_path).read_text(encoding="utf-8"))
    aabb = doc.get("scene_aabb") or {}
    lo = np.asarray(aabb.get("min"), dtype=np.float64)
    hi = np.asarray(aabb.get("max"), dtype=np.float64)
    nodes = doc.get("nodes") or []
    zone_nodes = [n for n in nodes if n.get("kind") == "zone" and n.get("bbox_min")]
    zone_ids = {n["id"] for n in zone_nodes}
    has_child_zone = {
        p for n in zone_nodes if (p := n.get("parent_id")) in zone_ids
    }
    zones: list[Zone] = []
    for n in zone_nodes:
        atomic = n.get("is_atomic")
        if atomic is None:
            atomic = n["id"] not in has_child_zone
        zones.append(
            Zone(
                id=n["id"],
                bbox_min=np.asarray(n["bbox_min"], dtype=np.float64),
                bbox_max=np.asarray(n["bbox_max"], dtype=np.float64),
                is_atomic=bool(atomic),
            )
        )
    if not zones:
        zones = [Zone(id="root", bbox_min=lo, bbox_max=hi, is_atomic=True)]
    return zones, lo, hi


# --- rig synthesis ----------------------------------------------------------------


def _fill_positions(
    fs: FreeSpace,
    zones: list[Zone],
    skin: np.ndarray,
    skin_tree: cKDTree,
    params: PlanParams,
) -> tuple[np.ndarray, list[str]]:
    """The Halton fill: per-zone low-discrepancy positions, count ∝ the zone's
    skin share, filtered to clear reachable air within the near-surface band,
    then globally lattice-deduped. Returns (positions, owning zone ids)."""
    # Zone weights: skin bricks inside each zone's bbox (overlaps double-count
    # here, but the TOTAL is normalized and the dedupe removes double-spend).
    weights = np.zeros(len(zones), dtype=np.float64)
    for zi, z in enumerate(zones):
        inside = np.all((skin >= z.bbox_min) & (skin <= z.bbox_max), axis=1)
        weights[zi] = float(inside.sum())
    total_w = float(weights.sum())
    n_total = int(
        np.clip(
            params.fill_per_skin_brick * len(skin), params.min_fill, params.max_fill
        )
    )
    if total_w <= 0:
        weights[:] = 1.0
        total_w = float(len(zones))

    pos_parts: list[np.ndarray] = []
    zone_parts: list[str] = []
    halton_cursor = params.seed * 7919  # disjoint deterministic runs per seed
    for zi, z in enumerate(zones):
        share = weights[zi] / total_w
        n_zone = int(round(n_total * share))
        if n_zone <= 0:
            continue
        size = np.maximum(z.bbox_max - z.bbox_min, 1e-6)
        # Oversample: validity rejection (walls, furniture, dead air) plus the
        # global dedupe eat a large fraction of raw Halton draws.
        n_draw = max(n_zone * 6, 64)
        u = _halton3(n_draw, start=halton_cursor)
        halton_cursor += n_draw
        pts = z.bbox_min + u * size
        ok = _clear_at(fs, pts, params.clear_m)
        if not ok.any():
            continue
        pts = pts[ok]
        near = skin_tree.query(pts, k=1)[0] <= params.near_surface_m
        pts = pts[near]
        if not len(pts):
            continue
        # Per-zone thin toward its own budget before the global dedupe, so one
        # huge sloppy bbox can't flood the lattice before small zones draw.
        if len(pts) > n_zone:
            stride = np.linspace(0, len(pts) - 1, n_zone).astype(np.int64)
            pts = pts[np.unique(stride)]
        pos_parts.append(pts)
        zone_parts.extend([z.id] * len(pts))
    if not pos_parts:
        return np.zeros((0, 3), dtype=np.float64), []
    pos = np.concatenate(pos_parts, axis=0)
    zone_ids = np.asarray(zone_parts)
    keep = _thin_lattice(pos, params.fill_spacing)
    keep = np.sort(keep)
    return pos[keep], [str(zone_ids[i]) for i in keep]


def _fill_directions(
    fs: FreeSpace, pos: np.ndarray, params: PlanParams
) -> tuple[np.ndarray, np.ndarray]:
    """Directions for the fill: an exactly-uniform Fibonacci set over the
    sphere, PERMUTED (seeded) so direction is uncorrelated with camera index —
    the raw golden spiral is z-monotone, and consecutive cameras would
    otherwise all stare at the ceiling. Point-blank cull: a frame whose
    central ray hits opaque cover within `_MIN_VIEW_M` tries the antipode,
    then two orthogonal re-aims, and is dropped if all four stare into
    nearby cover. Returns (keep_mask, directions)."""
    n = len(pos)
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros((0, 3))
    table = _fib_sphere(max(n, 64))
    perm = np.random.default_rng(params.seed).permutation(len(table))
    dirs = table[perm[:n]].copy()
    keep = np.zeros(n, dtype=bool)
    pending = np.arange(n)
    for attempt in range(4):
        if not len(pending):
            break
        d = dirs[pending]
        if attempt == 1:
            d = -d
        elif attempt >= 2:
            # A deterministic orthogonal: rotate about Y then flip.
            d = np.stack([d[:, 2], d[:, 1], -d[:, 0]], axis=1)
            if attempt == 3:
                d = -d
        hit = _first_hit(fs, pos[pending], d, _MIN_VIEW_M)
        good = ~np.isfinite(hit)
        idx = pending[good]
        dirs[idx] = d[good]
        keep[idx] = True
        pending = pending[~good]
    return keep, dirs


def _station_points(
    fs: FreeSpace,
    zone: Zone,
    fill_pos: np.ndarray,
    fill_zone: list[str],
    params: PlanParams,
) -> list[np.ndarray]:
    """Station standpoints for one atomic zone: among the zone's own validated
    fill positions (plus its bbox centre), score local clearance by an
    expanding probe ladder and pick up to `k` far-apart maxima, `k` scaling
    with zone extent. Empty when nowhere is clear enough to stand."""
    cand = [p for p, zid in zip(fill_pos, fill_zone) if zid == zone.id]
    center = (zone.bbox_min + zone.bbox_max) / 2.0
    cand.append(center)
    pts = np.asarray(cand, dtype=np.float64)
    inside = np.all((pts >= zone.bbox_min) & (pts <= zone.bbox_max), axis=1)
    pts = pts[inside]
    # Prefer the zone's mid-height band: clearance maxima in furnished rooms
    # sit just under the ceiling, but a panorama wants to stand where a
    # viewer's eye flies. Fall back to all candidates when the band is empty.
    h = max(float(zone.bbox_max[1] - zone.bbox_min[1]), 1e-6)
    band = (pts[:, 1] >= zone.bbox_min[1] + 0.25 * h) & (
        pts[:, 1] <= zone.bbox_min[1] + 0.75 * h
    )
    if band.any():
        pts = pts[band]
    if not len(pts):
        return []
    # Clearance ladder: the largest probe radius that stays clear (0 if even
    # the smallest fails). Coarse but cheap and monotone — enough to rank.
    ladder = np.array([0.3, 0.45, 0.7, 1.0, 1.4])
    score = np.zeros(len(pts), dtype=np.float64)
    for r in ladder:
        ok = _clear_at(fs, pts, float(r))
        score[ok] = r
    good = score >= _STATION_MIN_CLEAR
    if not good.any():
        return []
    pts, score = pts[good], score[good]
    k = int(np.clip(round(zone.max_extent / _STATION_EVERY_M), 1, _STATION_MAX))
    # Separation scales with the ZONE, not the global station pitch — a 3 m
    # bathroom's station must be allowed to exist 2 m from the bedroom's.
    sep = 0.5 * min(zone.max_extent, _STATION_EVERY_M)
    picked: list[np.ndarray] = []
    order = np.argsort(-score, kind="stable")
    for i in order:
        p = pts[i]
        if any(np.linalg.norm(p - q) < sep for q in picked):
            continue
        picked.append(p)
        if len(picked) >= k:
            break
    return picked


def _enclosure(fs: FreeSpace, point: np.ndarray, reach: float) -> float:
    """Fraction of an UPPER-HEMISPHERE Fibonacci fan of rays from `point`
    that hit opaque cover within `reach` — how much walls/ceiling WRAP a
    camera standing there. The ground is deliberately excluded (constants
    block): open terrain always covers the down-facing half."""
    dirs = _fib_sphere(_ENCLOSURE_DIRS * 2)
    dirs = dirs[dirs[:, 1] >= _ENCLOSURE_MIN_Y][:_ENCLOSURE_DIRS]
    origins = np.broadcast_to(point, (len(dirs), 3)).copy()
    hit = _first_hit(fs, origins, dirs, reach)
    return float(np.isfinite(hit).mean())


def _station_views(
    fs: FreeSpace, point: np.ndarray, n_dirs: int
) -> list[np.ndarray]:
    """One station's outward Fibonacci directions, minus the point-blank ones
    (embedded against furniture on one side keeps the open arc)."""
    dirs = _fib_sphere(n_dirs)
    origins = np.broadcast_to(point, (len(dirs), 3)).copy()
    hit = _first_hit(fs, origins, dirs, _MIN_VIEW_M)
    return [d for d, h in zip(dirs, hit) if not np.isfinite(h)]


def _shell_cameras(
    lo: np.ndarray, hi: np.ndarray, params: PlanParams
) -> tuple[np.ndarray, np.ndarray]:
    """The root establishing shell, SHAPE-ADAPTIVE: Fibonacci directions from
    the scene centre projected onto the scene AABB inflated by a per-axis
    standoff (constants block), so every face is orbited at a standoff
    proportional to its own extent — not at the whole-scene diagonal. Each
    position lies ON a face of the inflated box, i.e. beyond the grid margin
    on its exit axis: provably open air, no validity filters needed. Aims at
    Halton-jittered targets spread over the scene's middle (× its shape), so
    elongated scenes get sectional establishing shots along their length.
    Below-floor positions are culled (no under-terrain backface views)."""
    center = (lo + hi) / 2.0
    half = np.maximum((hi - lo) / 2.0, 1e-6)
    standoff = np.maximum(
        _SHELL_STANDOFF_FRAC * half,
        max(_SHELL_STANDOFF_MAXAXIS_FRAC * float(half.max()), _SHELL_STANDOFF_MIN),
    )
    box = half + standoff
    dirs = _fib_sphere(max(params.shell_views * 2, 8))  # oversample: floor cull
    t = (box / np.maximum(np.abs(dirs), 1e-9)).min(axis=1)  # ray-box exit
    pos = center + dirs * t[:, None]
    floor_y = lo[1] + _SHELL_FLOOR_FRAC * max(hi[1] - lo[1], 1e-6)
    pos = pos[pos[:, 1] >= floor_y]
    if len(pos) > params.shell_views:
        # Even subsample (the Fibonacci order is z-monotone: a prefix
        # truncation would keep only the top of the sphere).
        sel = np.unique(
            np.linspace(0, len(pos) - 1, params.shell_views).astype(np.int64)
        )
        pos = pos[sel]
    jitter = (_halton3(len(pos), start=911) - 0.5) * (0.7 * half)
    fwd = (center + jitter) - pos
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    return pos, fwd


def _up_for(forward: np.ndarray) -> np.ndarray:
    """A non-degenerate image-up per forward direction: world +Y, except near
    the poles where +Z takes over (matches the old cube-face convention)."""
    up = np.tile(np.array([0.0, 1.0, 0.0]), (len(forward), 1))
    polar = np.abs(forward[:, 1]) > 0.98
    up[polar] = np.array([0.0, 0.0, 1.0])
    up[polar & (forward[:, 1] < 0)] = np.array([0.0, 0.0, -1.0])
    return up


# --- the planner -------------------------------------------------------------------


def plan_cameras(
    *,
    run: str,
    slot: str,
    model: str,
    freespace_path: Path,
    scene_path: Path,
    out_path: Path,
    params: PlanParams = PlanParams(),
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Plan a cell's cameras (module docstring) and write `cameras.json`.
    Deterministic (Halton + Fibonacci + seeded assignment; no RNG state), pure
    CPU, and independent of the surfel cloud. Returns a summary."""
    fs = load_free_space(Path(freespace_path))
    zones, lo, hi = _read_zones(Path(scene_path))
    if progress is not None:
        progress(0, 0, "load")

    skin = _skin_centers(fs)
    skin_tree = cKDTree(skin)

    # 1) FILL — the backbone field.
    if progress is not None:
        progress(0, 0, "fill")
    fill_pos, fill_zone = _fill_positions(fs, zones, skin, skin_tree, params)
    keep, fill_dir = _fill_directions(fs, fill_pos, params)
    culled_fill = int((~keep).sum())
    fill_pos = fill_pos[keep]
    fill_dir = fill_dir[keep]
    fill_zone = [z for z, k in zip(fill_zone, keep) if k]

    # 2) STATIONS — outward panoramas in enclosed atomic zones.
    if progress is not None:
        progress(0, 0, "stations")
    station_cams: list[dict[str, Any]] = []
    placed_points: list[np.ndarray] = []  # global: overlapping zone bboxes
    stations_placed = 0
    stations_gated = 0
    for z in zones:
        if not z.is_atomic:
            continue
        for point in _station_points(fs, z, fill_pos, fill_zone, params):
            # Overlapping atomic zones (bboxes are not a partition) converge
            # on the same clearance maxima — one panorama there is enough.
            # Separation scales with the smaller zone so a small room beside
            # a large one keeps its own station.
            sep = 0.5 * min(z.max_extent, _STATION_EVERY_M)
            if any(np.linalg.norm(point - q) < sep for q in placed_points):
                continue
            reach = min(1.2 * z.diag, _ENCLOSURE_REACH_MAX)
            if _enclosure(fs, point, reach) < _ENCLOSURE_MIN:
                stations_gated += 1
                continue
            views = _station_views(fs, point, params.station_dirs)
            if not views:
                continue
            placed_points.append(point)
            stations_placed += 1
            for d in views:
                station_cams.append(
                    {"pos": point, "forward": d, "kind": "station", "zone": z.id}
                )

    # 3) SHELL — the root establishing orbit.
    if progress is not None:
        progress(0, 0, "shell")
    shell_pos, shell_fwd = _shell_cameras(lo, hi, params)

    # Assemble the flat plan.
    cams: list[dict[str, Any]] = []
    for p, d, zid in zip(fill_pos, fill_dir, fill_zone):
        cams.append({"pos": p, "forward": d, "kind": "fill", "zone": zid})
    cams.extend(station_cams)
    for p, d in zip(shell_pos, shell_fwd):
        cams.append({"pos": p, "forward": d, "kind": "shell", "zone": None})

    fwd_all = np.asarray([c["forward"] for c in cams], dtype=np.float64)
    up_all = _up_for(fwd_all) if len(cams) else np.zeros((0, 3))

    diag = float(np.linalg.norm(hi - lo))
    near = min(0.05, fs.pitch * 0.5)
    far = (1.0 + _SHELL_RADIUS_MULT) * diag + 2.0

    doc = {
        "plan_version": PLAN_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": run,
        "slot": slot,
        "model": model,
        "intrinsics": {
            "resolution": int(params.render_resolution),
            "fov_deg": float(params.fov_deg),
            "near": near,
            "far": far,
        },
        "cameras": [
            {
                "pos": [round(float(v), 5) for v in c["pos"]],
                "forward": [round(float(v), 6) for v in fwd_all[i]],
                "up": [round(float(v), 6) for v in up_all[i]],
                "kind": c["kind"],
                "zone": c["zone"],
            }
            for i, c in enumerate(cams)
        ],
    }

    counts = {
        "fill": int(len(fill_pos)),
        "station": int(len(station_cams)),
        "shell": int(len(shell_pos)),
    }
    summary = {
        "run": run,
        "slot": slot,
        "model": model,
        "cameras": len(cams),
        "views": len(cams),  # one shot per camera — kept for status parity
        "kinds": counts,
        "zones": {
            "total": len(zones),
            "atomic": int(sum(z.is_atomic for z in zones)),
            "stations_placed": stations_placed,
            "stations_gated": stations_gated,
        },
        "culled_point_blank": culled_fill,
        "skin_bricks": int(len(skin)),
        "scene_aabb": {"min": lo.tolist(), "max": hi.tolist()},
        "intrinsics": doc["intrinsics"],
        "params": params.as_summary(),
    }
    doc["summary"] = summary

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    tmp.replace(out_path)
    summary["bytes"] = out_path.stat().st_size
    if progress is not None:
        progress(0, 0, "write")
    return summary
