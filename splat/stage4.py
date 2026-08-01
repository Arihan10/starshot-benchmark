"""Stage 4 — Camera planner (inward object shells + residual interior cubemap stations).

Plans the single-shot reference cameras Stage 5 renders and the splat trainer
consumes. WE OWN THE GEOMETRY, so the plan is built object-by-object from the
scene's own construction record instead of scattering undirected views:

  * BALLS, TIGHT AGAINST THE SHAPE. Every placed mesh (props AND architecture
    — geometry decides, not node kind) gets cameras by SURFACE DARTS: pick one
    of the object's own cover bricks (the ~8-voxel cubes of the Stage-2 grid
    its surface actually occupies, ∝ area), pick a direction, step out the
    standoff, aim back at the brick. The sampling domain is therefore the
    offset surface of the REAL shape, not of its bounding box: a tree gets
    cameras under its canopy aiming up and rings around its trunk, a tub gets
    in-bowl views, an L-shaped sofa never wastes draws on its AABB's empty
    corner. Per-object bricks are attributed by AABB overlap against the
    global cover-brick set (no mesh loading; exact per-object attribution via
    a Stage-2 sidecar is the upgrade path if overlapping-AABB smear ever
    measures as a problem).
  * SCALE-MATCHED STANDOFF, WITH DESCENT. The base standoff frames the whole
    object (`standoff_frac`·diag, floored by `standoff_min`, ceilinged only by
    the scene-diagonal cap). When the free space cannot FIT that distance — an
    interior wall cannot be photographed from 5.7 m inside a 7 m room — the
    planner DESCENDS a fixed ladder (1, ½, ¼ ×, floored near the grid scale)
    and accepts the first rung where enough of the inner ring survives,
    re-deriving the count at the accepted distance (closer frames cover less
    surface each — the 1/d² growth is the physics, bounded by the ladder's
    fixed depth). A SALVAGE re-walk with escalated draws runs before any
    target is declared starved; what still fails is genuinely buried.
  * DETAIL RING (geometry density earns close-ups). Whole-object framing
    resolves an object's TEXTURE at ~1 px/texel regardless of size, but not
    sub-texel GEOMETRY — leaves, rock relief, gratings. Each object's feature
    scale is estimated from facts we already have (ℓ ≈ √(voxel surface area /
    mesh faces)); when ℓ needs a camera meaningfully closer than the base
    ring to span `detail_px` pixels, the object earns an extra dart ring at
    that distance, count derived by the same coverage formula there. A
    500k-face cypress pays for leaf-scale views; the 2-triangle backdrop and
    water planes never do — this is the principled replacement for the old
    fixed 1.6 m section cap (which over-covered flats and was in turn capped
    by `ball_max`, both now gone).
  * RING LADDER (the band-limit). One close shell alone trains a splat that
    shimmers from farther away (sub-pixel splats alias; the delivered viewer
    has no mip filter). Each object also gets its shell at `ring_mults` × the
    accepted standoff with `ring_frac` of its budget, so the optimizer is
    graded at multiple scales.
  * SHELL — the root establishing orbit outside the scene AABB (unchanged):
    the far octave the ladders top out at, and the only supervision of
    whole-scene framings.
  * FLOOD — legacy free-space lattice, now OFF by default. ORB v2 subsumes its
    interior job with complete cube coverage, station-aware accounting, and
    residual selection. Retained only as an explicit A/B switch for old plans.
  * SWEEP — a targeted tiling of LARGE FLAT surfaces (ceiling / floor / walls /
    big panels) from their interior side. A big flat's AABB diagonal is huge, so
    its per-object standoff can't fit the room and its room-side darts get culled
    — it would otherwise train into a cloud. The sweep grids the face and views
    each cell perpendicular + oblique (parallax) at a room-fitted standoff.
  * ORBS v2 — the DUAL of the per-object balls and the owner of interior free
    space. A ball sits around a SURFACE and looks IN; an orb is a free-space
    STATION proposing six 90° cube faces. Geometry, not semantics, decides what
    is interior: sampled escape solid angle rejects open exterior positions.
    The existing inward plan is marched first, observations are grouped by
    distinct optical-centre cells, and cubemap stations are selected only where
    they reduce the residual interior deficit. Six non-overlapping faces replace
    the old 24 overlapping Fibonacci views, spending the same frame budget on
    roughly four times as many spatial baselines. Faces that see only sky,
    under-scene geometry, or already-saturated patches have zero marginal value
    and are omitted.

BUDGET BY RESIDUAL (the orb layer only). A surface patch asks for distinct
stations, not raw images. Zero-baseline patches receive exponentially more gain
than patches missing only their final station, so an enclosed pocket beats an
overcovered hall wall without either being named a "room". Selection stops when
no enclosed candidate face can reduce the residual; `orb_max_views` remains a
safety ceiling rather than the ordinary source of scene-size invariance.

GEOMETRY STILL DISPOSES. Every candidate passes scene-agnostic filters
against the Stage-2 grid: POSITION (outside the grid is provably open air;
inside must be reachable EMPTY air with standoff-scaled clearance), BLOCKED
(first OPAQUE hit well before the aimed brick = another object owns the frame
centre; dropped, never re-aimed; glass doesn't block, so framing through
panes works), PRESENT (some cover — opaque or glass — must lie near the aim
distance: no air shots), and a global lattice × aim-octant DEDUPE that
collapses clustered objects' overlapping shells while never stripping a
target's last camera.

Counts are DERIVED, never capped per object: views = `coverage` × the shell's
surface area (measured from the object's own cover bricks) ÷ the area one
frame covers at the standoff ((2·tan(fov/2)·d)²) — dimensionless and
self-normalizing. `ball_min` floors the per-object guarantee;
`min_ball_views`/`max_ball_views` bound the SCENE total by proportional
rescale (a budget guard, not a modeling statement). Deterministic (Halton +
seeded cursors, no RNG state), pure CPU (numpy + one scipy distance transform
for the orb layer's free-space field), independent of the surfel cloud.

Consumes `freespace.npz` (Stage 2) + `scene.json` (Stage 1). Emits
`cameras.json` (plan_version 3): shared pinhole `intrinsics` + a flat
`cameras` list, each `{pos, forward, up, kind: ball|shell|flood|sweep|orb,
zone}`. Stage 5 renders one image per entry (it accepts plan_version ≥ 2);
poses convert via `opencv_c2w`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from splat.stage2 import FreeSpace, load_free_space

CAMERAS_NAME = "cameras.json"
PLAN_VERSION = 3

# Fine voxels per brick edge (matches stage2._BRICK).
_BRICK = 8

# Position-validity probes: a position is clear when `occupied` is empty at the
# point and on three probe shells (26 directions × {r/3, 2r/3, r}) — a cheap,
# grid-native "at least ~r from any surface" test (three shells so a thin slab
# can't slip between consecutive radii).
_CLEAR_DIRS = np.array(
    [o for o in np.ndindex(3, 3, 3) if o != (1, 1, 1)], dtype=np.float64
) - 1.0
_CLEAR_DIRS /= np.linalg.norm(_CLEAR_DIRS, axis=1, keepdims=True)

# Clearance radius rides the ring standoff (`clear = _CLEAR_FRAC × d`) between
# hard bounds: a 35 cm close-up may stand 6 cm off a shelf, a 4 m room view
# keeps real clearance. (The near plane is ~1.5 cm — 6 cm never clips.)
_CLEAR_FRAC = 0.2
_CLEAR_MIN = 0.06
_CLEAR_MAX = 0.25

# Candidate oversampling per ring: validity rejection (walls, clutter, self-
# occlusion) eats a large share of raw draws; the surviving PREFIX of a Halton
# sequence is itself low-discrepancy, so we keep the first `n` valid.
_OVERSAMPLE = 4
# The DESCENT ladder (module docstring): base-standoff multipliers tried in
# order until enough of the inner ring survives. A rung is accepted when the
# inner ring keeps ≥ `_ACCEPT_FRAC` of its derived count (the last rung
# accepts anything it finds).
_DESCENT_MULTS = (1.0, 0.5, 0.25)
_DESCENT_DRAW_FLOOR = (48, 48, 64)
_MIN_STANDOFF = 0.12
_ACCEPT_FRAC = 0.6
# SALVAGE: a target that failed the whole ladder re-walks it once with this
# many draws per rung before being declared starved. Only zero-view targets
# pay for it, and it reliably finds pockets down to ~1–2% of a shell (a
# wine-fridge bottle visible through a slit); anything it still misses is
# genuinely invisible in practice.
_SALVAGE_DRAWS = 160

# Aim-window fuzz around the expected hit distance: the aimed brick centre is
# within half a brick diagonal of real surface, plus voxel-march quantization.
# BLOCKED = opaque cover strictly before the window; PRESENT = any cover
# inside it.
def _aim_pad(fs: FreeSpace) -> float:
    return float(0.5 * np.sqrt(3.0) * _BRICK * fs.pitch + 3.0 * fs.pitch)

# Ring standoffs are capped at this fraction of the scene diagonal — beyond it
# a "ring" is just a worse establishing shell, and the root shell owns that.
_RING_SCENE_CAP = 0.5

# Shell (root establishing orbit) — unchanged construction: Fibonacci
# directions projected onto the scene AABB inflated by a per-axis standoff,
# aims Halton-jittered over the scene middle, below-floor culled.
_SHELL_STANDOFF_FRAC = 0.6
_SHELL_STANDOFF_MAXAXIS_FRAC = 0.3
_SHELL_STANDOFF_MIN = 3.0
_SHELL_RADIUS_MULT = 0.9
_SHELL_FLOOR_FRAC = 0.05

# ORB v2 cube rig. Faces are ordinary 90° pinhole images, so Stage 5 / COLMAP /
# Brush / gsplat stay unchanged; "cubemap" is only the planner's station unit.
_ORB_CUBE_DIRS = np.asarray(
    [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
     [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)
# Marginal gain is represented with small integers: a patch with zero distinct
# baseline cells is worth much more than one missing only its final station.
_ORB_DEFICIT_WEIGHTS = np.asarray([16, 4, 1], dtype=np.int32)

ProgressCb = Callable[[int, int, str], None]


@dataclass(frozen=True)
class PlanParams:
    """Stage-4 knobs — every count is DERIVED from these (module docstring).

    `coverage` is THE density dial, dimensionless: a target's views =
    coverage × its shell area ÷ one frame's footprint at the chosen standoff
    — self-normalizing across object sizes, so there is no per-object cap.
    `detail_px` sets how many pixels an object's typical geometric feature
    should span in its DETAIL ring (smaller = closer detail cameras; the
    ring only exists where geometry is finer than the base ring resolves).
    `ball_min` floors the per-object guarantee; `min_ball_views` /
    `max_ball_views` bound the SCENE total by proportional rescale (floor 2 —
    the guarantee survives the clamp). `ring_mults`/`ring_frac` are the
    multi-scale ring ladder — the train-time band-limit."""

    coverage: float = 0.6
    detail_px: float = 6.0
    ball_min: int = 8
    min_ball_views: int = 128
    max_ball_views: int = 3000
    # Base standoff: frame the object (`frac` × diag), floored so tiny props
    # aren't macro-photographed past their texture information. No upper
    # bound beyond the scene cap — when the space can't fit the distance the
    # DESCENT ladder finds the one it can (constants block).
    standoff_frac: float = 0.71
    standoff_min: float = 0.35
    # The multi-scale ring ladder (see class docstring).
    ring_mults: tuple[float, ...] = (1.0, 2.5, 6.25)
    ring_frac: tuple[float, ...] = (0.6, 0.25, 0.15)
    # Global dedupe lattice (cell edge, metres) × forward octant.
    dedupe_spacing: float = 0.4
    # Root establishing orbit.
    shell_views: int = 128
    # INTERIOR FLOOD-FILL (stacked on the balls): fill the reachable free space
    # with look-around cameras for the wall / floor / ceiling + baked-shadow
    # coverage the object-centric darts miss. The lattice cell scales with the
    # scene (`flood_spacing_frac`·diag, floored by `flood_spacing_min`) so the
    # cell COUNT is scene-size-stable; `flood_dirs` Fibonacci looks per cell;
    # `flood_clear` is the min standoff from any surface (no jammed views); the
    # two caps bound the layer on big scenes by deterministic subsample.
    flood: bool = False
    flood_spacing_frac: float = 0.05
    flood_spacing_min: float = 0.6
    flood_dirs: int = 24
    flood_clear: float = 0.3
    flood_max_positions: int = 512
    flood_max_views: int = 512
    # LARGE-FLAT SURFACE SWEEP (targeted, NOT a global density bump): a big flat
    # target (ceiling / floor / wall / panel) has a huge AABB diagonal, so its
    # per-object standoff can't fit the room and the room-side (looking-AT-it)
    # darts get culled — it trains into a cloud. This tiles such a target's
    # interior face from a room-fitted standoff at a perpendicular + two oblique
    # angles (the parallax that triangulates floaters away). A target qualifies
    # when its thinnest AABB axis ≤ `sweep_thin_max` and its face area ≥
    # `sweep_area_min`; `sweep_max_views` bounds the layer (deterministic subsample).
    sweep: bool = True
    sweep_thin_max: float = 0.6
    sweep_area_min: float = 3.0
    sweep_spacing: float = 1.5
    sweep_standoff: float = 1.5
    sweep_oblique: float = 0.6
    sweep_max_views: int = 512
    # OUTWARD ORBS v2 — residual INTERIOR coverage, not another global view fill.
    # An orb is one FREE-SPACE STATION proposing the six 90° cube faces. Six faces
    # tile 4π without the overlap of the old 24-view Fibonacci fan, so the same
    # image budget buys ~4× as many optical centres — real parallax, not repeated
    # angles from one point. Individual faces whose marginal residual gain is zero
    # are dropped (sky / under-scene / already-saturated directions).
    #
    # INTERIOR is geometric: `orb_escape_dirs` rays estimate the point's escape
    # solid angle. `enclosure = 1 − escaped/dirs`; only candidates above
    # `orb_enclosure_min` participate. A courtyard / exterior stays the job of the
    # inward-facing balls, while a room, corridor or furnished pocket scores high
    # without any semantic "room" annotation.
    #
    # COVERAGE is station-aware. A surface patch wants `orb_target_stations`
    # DISTINCT selected origins, not N overlapping images from one origin. The
    # existing BALL/SHELL/SWEEP plan is marched first; patches already seen from
    # distinct baseline cells get little or no ORB value. Candidate stations are
    # selected greedily by marginal deficit reduction, normalized automatically
    # through diminishing returns — a zero-baseline bathroom patch outranks the
    # twentieth observation of a living-room wall regardless of either region's
    # volume. `orb_candidate_spacing` only thins the candidate pool; it is not a
    # placement lattice and carries no final camera guarantee.
    orb: bool = True
    orb_min_clear: float = 0.12
    orb_candidate_spacing: float = 1.0
    orb_candidate_cap: int = 4096
    orb_enclosure_min: float = 0.85
    orb_escape_dirs: int = 12
    orb_target_stations: int = 3
    orb_min_face_gain: float = 1.0
    orb_max_views: int = 8000
    orb_measure_rays: int = 8
    # PER-CAMERA FOV. One angle for a whole plan is a compromise between two
    # opposed needs: a wide frame sees more of a room per render (and its rays
    # cross more of the air that has to be proven empty), while a narrow frame
    # spends more pixels on a distant surface. Measured on real cells, 110° beat
    # 70° outright on an all-interior suite — resolved views per patch 13 → 20 at
    # equal camera count — and LOST on a half-outdoor house, 3 → 2, because there
    # a 24 cm patch stays above the resolution floor only to 5.4 m at 110° versus
    # 11.0 m at 70°. Subject distances vary as much WITHIN a scene as between
    # scenes, so a single global choice is wrong either way and the angle is
    # DERIVED per camera instead: hold a grid-scale feature at `fov_ref_px`
    # pixels on the surface that camera actually frames, and solve for the angle.
    # `fov_deg` remains the value used when derivation is off, and the emitted
    # `intrinsics.fov_deg` is the plan's median so a consumer that ignores the
    # per-camera field still gets a representative one.
    fov_derive: bool = True
    fov_min_deg: float = 50.0
    fov_max_deg: float = 110.0
    fov_ref_px: float = 16.0
    fov_deg: float = 70.0
    render_resolution: int = 1024
    seed: int = 0

    def as_summary(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage,
            "detail_px": self.detail_px,
            "ball_min": self.ball_min,
            "min_ball_views": self.min_ball_views,
            "max_ball_views": self.max_ball_views,
            "standoff_frac": self.standoff_frac,
            "standoff_min": self.standoff_min,
            "ring_mults": list(self.ring_mults),
            "ring_frac": list(self.ring_frac),
            "dedupe_spacing": self.dedupe_spacing,
            "shell_views": self.shell_views,
            "flood": self.flood,
            "flood_spacing_frac": self.flood_spacing_frac,
            "flood_spacing_min": self.flood_spacing_min,
            "flood_dirs": self.flood_dirs,
            "flood_clear": self.flood_clear,
            "flood_max_positions": self.flood_max_positions,
            "flood_max_views": self.flood_max_views,
            "sweep": self.sweep,
            "sweep_thin_max": self.sweep_thin_max,
            "sweep_area_min": self.sweep_area_min,
            "sweep_spacing": self.sweep_spacing,
            "sweep_standoff": self.sweep_standoff,
            "sweep_oblique": self.sweep_oblique,
            "sweep_max_views": self.sweep_max_views,
            "orb": self.orb,
            "orb_min_clear": self.orb_min_clear,
            "orb_candidate_spacing": self.orb_candidate_spacing,
            "orb_candidate_cap": self.orb_candidate_cap,
            "orb_enclosure_min": self.orb_enclosure_min,
            "orb_escape_dirs": self.orb_escape_dirs,
            "orb_target_stations": self.orb_target_stations,
            "orb_min_face_gain": self.orb_min_face_gain,
            "orb_max_views": self.orb_max_views,
            "orb_measure_rays": self.orb_measure_rays,
            "fov_derive": self.fov_derive,
            "fov_min_deg": self.fov_min_deg,
            "fov_max_deg": self.fov_max_deg,
            "fov_ref_px": self.fov_ref_px,
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
    (opaque OR glass) within ~`r` (constants block)."""
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


def _ray_probe(
    fs: FreeSpace, origins: np.ndarray, dirs: np.ndarray, t_max: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per ray: (first OPAQUE hit, first ANY-cover hit) distances along `dirs`
    (∞ = none within that ray's `t_max`). One voxel-stepped march answers both
    culls — BLOCKED reads the opaque set (glass doesn't block sight), PRESENT
    reads the full cover set (a window pane counts as its object)."""
    n = len(origins)
    opaque = np.full(n, np.inf, dtype=np.float64)
    any_hit = np.full(n, np.inf, dtype=np.float64)
    if n == 0:
        return opaque, any_hit
    step = fs.pitch
    t = step
    t_top = float(t_max.max())
    alive = np.nonzero(t_max >= t)[0]
    while len(alive) and t <= t_top + 1e-9:
        pts = origins[alive] + dirs[alive] * t
        occ = fs.occupied(pts)
        if occ.any():
            hit = alive[occ]
            first = ~np.isfinite(any_hit[hit])
            any_hit[hit[first]] = t
            opq = fs.occluding(pts[occ])
            if opq.any():
                # A ray is answered once its opaque hit lands (its any-cover
                # hit is then ≤ it); glass-only hits march on for the opaque.
                opaque[hit[opq]] = t
                dead = np.zeros(n, dtype=bool)
                dead[hit[opq]] = True
                alive = alive[~dead[alive]]
        t += step
        alive = alive[t_max[alive] >= t]
    return opaque, any_hit


def _thin_lattice_aim(
    points: np.ndarray, dirs: np.ndarray, spacing: float
) -> np.ndarray:
    """Indices of one winner (nearest cell centre) per (`spacing` lattice cell
    × forward OCTANT) — the global dedupe that collapses overlapping object
    shells into shared frames while keeping genuinely different aims."""
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    cell = np.floor(points / spacing).astype(np.int64)
    loc = cell - cell.min(axis=0)
    ny = int(loc[:, 1].max()) + 1
    nz = int(loc[:, 2].max()) + 1
    octant = (
        (dirs[:, 0] > 0).astype(np.int64) * 4
        + (dirs[:, 1] > 0).astype(np.int64) * 2
        + (dirs[:, 2] > 0).astype(np.int64)
    )
    gid = (((loc[:, 0] * ny + loc[:, 1]) * nz) + loc[:, 2]) * 8 + octant
    center = (cell.astype(np.float64) + 0.5) * spacing
    d2 = ((points - center) ** 2).sum(axis=1)
    order = np.lexsort((d2, gid))
    gid_s = gid[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = gid_s[1:] != gid_s[:-1]
    return order[first]


# --- the targets (placed meshes) ---------------------------------------------------


@dataclass(frozen=True)
class Target:
    id: str
    lo: np.ndarray
    hi: np.ndarray
    zone: str | None
    faces: int

    @property
    def diag(self) -> float:
        return float(np.linalg.norm(self.hi - self.lo))


def _read_targets(scene_path: Path) -> tuple[list[Target], np.ndarray, np.ndarray]:
    """Ball targets + the scene AABB from a Stage-1 manifest. Targets are the
    PLACED meshes (`objects[]` — ground-truth world AABBs + mesh face counts,
    the geometry-density signal), each attributed to its owning zone via
    `nodes[]`. Degenerate AABBs (< 2 cm diagonal) are skipped. A manifest
    with no placed objects degrades to ONE target over the scene AABB."""
    doc = json.loads(Path(scene_path).read_text(encoding="utf-8"))
    aabb = doc.get("scene_aabb") or {}
    lo = np.asarray(aabb.get("min"), dtype=np.float64)
    hi = np.asarray(aabb.get("max"), dtype=np.float64)
    zone_of = {n["id"]: n.get("zone") for n in doc.get("nodes") or []}
    targets: list[Target] = []
    for obj in doc.get("objects") or []:
        t_lo = np.asarray(obj["aabb_min"], dtype=np.float64)
        t_hi = np.asarray(obj["aabb_max"], dtype=np.float64)
        if float(np.linalg.norm(t_hi - t_lo)) < 0.02:
            continue
        targets.append(
            Target(
                id=obj["id"], lo=t_lo, hi=t_hi,
                zone=zone_of.get(obj["id"]), faces=int(obj.get("faces") or 0),
            )
        )
    if not targets:
        targets = [Target(id="scene", lo=lo, hi=hi, zone=None, faces=0)]
    return targets, lo, hi


# --- the tight shells: cover bricks + surface darts --------------------------------


def _cover_brick_centers(fs: FreeSpace) -> np.ndarray:
    """(B,3) world centres of every brick containing COVER (opaque or glass)
    — the voxelized surface the darts originate from, and the area measure
    the counts derive from. Computed once per plan from the sorted fine cover
    (chunked; no dense array)."""
    bd = fs.bdims
    nyz = int(fs.dims[1]) * int(fs.dims[2])
    nz = int(fs.dims[2])
    parts: list[np.ndarray] = []
    for i in range(0, fs.occ_lin.size, 1 << 24):
        lin = fs.occ_lin[i : i + (1 << 24)]
        x = lin // nyz
        rem = lin % nyz
        y, z = rem // nz, rem % nz
        blin = ((x >> 3) * bd[1] + (y >> 3)) * bd[2] + (z >> 3)
        parts.append(np.unique(blin))
    blin = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
    bx = blin // (bd[1] * bd[2])
    by = (blin // bd[2]) % bd[1]
    bz = blin % bd[2]
    cells = np.stack([bx, by, bz], axis=1).astype(np.float64) * _BRICK + _BRICK / 2.0
    return fs.origin + cells * fs.pitch


def _dart_samples(
    centers: np.ndarray, d: float, n: int, start: int
) -> tuple[np.ndarray, np.ndarray]:
    """(positions (n,3), forwards (n,3)) by SURFACE DARTS: Halton-pick one of
    the target's own cover bricks (∝ area — bricks are equal-area patches of
    voxelized surface), a uniform sphere direction, step out `d`, aim back at
    the brick. The sampling domain is the offset surface of the REAL shape:
    concavities (under-canopy air, tub bowls, wardrobe interiors) are reached
    exactly like convex faces."""
    u = _halton3(n, start=start)
    idx = np.minimum((u[:, 0] * len(centers)).astype(np.int64), len(centers) - 1)
    z = 2.0 * u[:, 1] - 1.0
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = 2.0 * np.pi * u[:, 2]
    dirs = np.stack([r * np.cos(phi), z, r * np.sin(phi)], axis=1)
    pos = centers[idx] + dirs * d
    return pos, -dirs


def _valid_ball_views(
    fs: FreeSpace,
    d: float,
    pos: np.ndarray,
    fwd: np.ndarray,
    culls: dict[str, int],
) -> np.ndarray:
    """Indices of the candidates that pass every filter, in draw order:
      * POSITION — outside the grid is provably open air; inside needs
        reachable EMPTY air + a standoff-scaled clearance;
      * BLOCKED / PRESENT — one voxel march against the aim window around
        the expected hit distance `d` (module docstring).
    Cull counters accumulate into `culls`."""
    g_lo = fs.origin
    g_hi = fs.origin + fs.dims.astype(np.float64) * fs.pitch
    pad = _aim_pad(fs)

    inb = np.all((pos >= g_lo) & (pos <= g_hi), axis=1)
    ok = ~inb
    if inb.any():
        r_clear = float(np.clip(_CLEAR_FRAC * d, _CLEAR_MIN, _CLEAR_MAX))
        ok_in = _clear_at(fs, pos[inb], r_clear)
        ok[np.nonzero(inb)[0][ok_in]] = True
    culls["position"] += int(inb.sum() - ok[inb].sum())
    if not ok.any():
        return np.zeros(0, dtype=np.int64)
    idx = np.nonzero(ok)[0]

    t_max = np.full(len(idx), d + pad, dtype=np.float64)
    opaque_hit, any_hit = _ray_probe(fs, pos[idx], fwd[idx], t_max)
    blocked = opaque_hit < (d - pad)
    present = np.isfinite(any_hit)
    culls["blocked"] += int(blocked.sum())
    culls["air"] += int((~blocked & ~present).sum())
    return idx[~blocked & present]


# --- the ball planner ---------------------------------------------------------------


def _frame_count(area_m2: float, d: float, params: PlanParams, floor: int) -> int:
    """DERIVED view count at standoff `d`: coverage × shell area ÷ one
    frame's footprint ((2·tan(fov/2)·d)²) — the self-normalizing allocator
    (module docstring): frames and shell grow with the same d², so no object
    size can diverge."""
    footprint = (2.0 * np.tan(np.radians(params.fov_deg) / 2.0) * d) ** 2
    n = round(params.coverage * area_m2 / max(footprint, 1e-12))
    return max(floor, int(n))


def _detail_standoff(
    area_m2: float, faces: int, d_base: float, params: PlanParams
) -> float | None:
    """The DETAIL ring's distance for one target, or None when its geometry
    doesn't earn one. Feature scale ℓ ≈ √(area / faces); the ring sits where
    ℓ spans `detail_px` pixels. Gated to objects whose features need a camera
    meaningfully closer than the base ring (< 0.5 × base) — a prop's base
    ring IS its detail ring, and a low-poly flat never qualifies. Floored at
    half `standoff_min`: features finer than that are beyond the render/
    texture information limit anyway."""
    if faces <= 0 or area_m2 <= 0:
        return None
    focal_px = params.render_resolution / (2.0 * np.tan(np.radians(params.fov_deg) / 2.0))
    ell = float(np.sqrt(area_m2 / faces))
    d = ell * focal_px / max(params.detail_px, 1e-6)
    d = max(d, 0.5 * params.standoff_min, _MIN_STANDOFF)
    return d if d < 0.5 * d_base else None


def _ball_budgets(
    targets: list[Target],
    brick_counts: np.ndarray,
    brick_area: float,
    params: PlanParams,
    scene_diag: float,
) -> tuple[list[float], float]:
    """Per-target base standoff (scale-matched, scene-capped) + the GLOBAL
    budget factor: base + gated-detail requests are estimated at the base
    standoff and the scene total proportionally rescaled into
    [`min_ball_views`, `max_ball_views`]. A descended target re-derives its
    count at the accepted rung, so the realized total can exceed the clamp
    modestly — it is a budget guard, not an exact quota."""
    standoffs: list[float] = []
    total = 0
    for t, nb in zip(targets, brick_counts):
        d0 = max(params.standoff_frac * t.diag, params.standoff_min)
        d0 = min(d0, max(_RING_SCENE_CAP * scene_diag, params.standoff_min))
        standoffs.append(float(d0))
        area = float(nb) * brick_area
        total += _frame_count(area, d0, params, params.ball_min)
        d_det = _detail_standoff(area, t.faces, d0, params)
        if d_det is not None:
            total += _frame_count(area, d_det, params, 0)
    scale = 1.0
    if total > params.max_ball_views:
        scale = params.max_ball_views / total
    elif 0 < total < params.min_ball_views:
        scale = params.min_ball_views / total
    return standoffs, scale


def _plan_balls(
    fs: FreeSpace,
    targets: list[Target],
    params: PlanParams,
    scene_diag: float,
    progress: ProgressCb | None,
) -> tuple[list[dict[str, Any]], dict[str, int], list[float], dict[str, int]]:
    """All surviving ball cameras (pre-dedupe) + cull counters + the CHOSEN
    per-framed-target standoffs + counters {descended, detail_targets,
    detail_views}. Per target: attribute its cover bricks (AABB overlap
    against the global cover-brick set), walk the DESCENT ladder on the base
    ring, then the outer rings and the gated DETAIL ring at the accepted
    standoff. Every ring keeps the first `n` valid draws."""
    if progress is not None:
        progress(0, 0, "bricks")
    centers = _cover_brick_centers(fs)
    brick_edge = _BRICK * fs.pitch
    brick_area = brick_edge * brick_edge

    # Per-target brick attribution: cover bricks whose centre falls in the
    # target's AABB inflated by one brick. Overlapping AABBs share bricks
    # (a spoon's set includes table bricks beneath it) — the aim fuzz this
    # causes is bounded by the validity culls; exact attribution is the
    # Stage-2-sidecar upgrade path.
    tbricks: list[np.ndarray] = []
    for t in targets:
        m = np.all(
            (centers >= t.lo - brick_edge) & (centers <= t.hi + brick_edge), axis=1
        )
        tbricks.append(np.nonzero(m)[0])
    standoffs, scale = _ball_budgets(
        targets, np.array([len(b) for b in tbricks]), brick_area, params, scene_diag
    )

    culls = {"position": 0, "blocked": 0, "air": 0}
    cams: list[dict[str, Any]] = []
    chosen: list[float] = []
    stats = {"descended": 0, "detail_targets": 0, "detail_views": 0}
    cursor = params.seed * 7919  # disjoint deterministic Halton runs per seed

    def emit(target: Target, pos: np.ndarray, fwd: np.ndarray, idx: np.ndarray) -> int:
        for k in idx:
            cams.append(
                {
                    "pos": pos[k],
                    "forward": fwd[k],
                    "kind": "ball",
                    "zone": target.zone,
                    "target": target.id,
                }
            )
        return int(len(idx))

    for ti, target in enumerate(targets):
        if progress is not None and ti % 16 == 0:
            progress(ti, len(targets), "balls")
        bidx = tbricks[ti]
        if not len(bidx):
            continue  # no voxelized surface in its AABB — nothing to frame
        tc = centers[bidx]
        area = float(len(bidx)) * brick_area
        d0 = standoffs[ti]
        ladder: list[float] = []
        for m in _DESCENT_MULTS:
            d = max(d0 * m, _MIN_STANDOFF)
            if d not in ladder:
                ladder.append(d)

        def try_rung(d_base: float, n_draw: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
            nonlocal cursor
            n_total = max(2, round(_frame_count(area, d_base, params, params.ball_min) * scale))
            pos, fwd = _dart_samples(tc, d_base, n_draw, cursor)
            cursor += n_draw
            good = _valid_ball_views(fs, d_base, pos, fwd, culls)
            return pos, fwd, good, n_total

        accepted: tuple[float, int] | None = None
        for rung, d_base in enumerate(ladder):
            n_probe = max(2, round(_frame_count(area, d_base, params, params.ball_min) * scale))
            n_inner = max(1, round(n_probe * params.ring_frac[0]))
            n_draw = max(
                n_inner * _OVERSAMPLE,
                _DESCENT_DRAW_FLOOR[min(rung, len(_DESCENT_DRAW_FLOOR) - 1)],
            )
            pos, fwd, good, n_total = try_rung(d_base, n_draw)
            need = max(1, int(np.ceil(_ACCEPT_FRAC * n_inner)))
            if len(good) >= need or (rung == len(ladder) - 1 and len(good) > 0):
                emit(target, pos, fwd, good[:n_inner])
                accepted = (d_base, n_total)
                if rung > 0:
                    stats["descended"] += 1
                break
        if accepted is None:
            # SALVAGE (constants block): one escalated re-walk of the ladder
            # for the rare-pocket targets the standard draws missed.
            for d_base in ladder:
                pos, fwd, good, n_total = try_rung(d_base, _SALVAGE_DRAWS)
                if len(good):
                    n_inner = max(1, round(n_total * params.ring_frac[0]))
                    emit(target, pos, fwd, good[:n_inner])
                    accepted = (d_base, n_total)
                    stats["descended"] += 1
                    break
        if accepted is None:
            continue  # starved — no reachable viewpoint at any rung

        d_base, n_total = accepted
        chosen.append(d_base)
        for mult, frac in zip(params.ring_mults[1:], params.ring_frac[1:]):
            n_ring = max(1, round(n_total * frac))
            d = min(d_base * mult, max(_RING_SCENE_CAP * scene_diag, d_base))
            n_draw = n_ring * _OVERSAMPLE
            pos, fwd = _dart_samples(tc, d, n_draw, cursor)
            cursor += n_draw
            good = _valid_ball_views(fs, d, pos, fwd, culls)
            emit(target, pos, fwd, good[:n_ring])

        # DETAIL ring (module docstring): geometry finer than the base ring
        # resolves earns close darts at its own derived distance and count.
        d_det = _detail_standoff(area, target.faces, d_base, params)
        if d_det is not None:
            n_det = round(_frame_count(area, d_det, params, 0) * scale)
            if n_det > 0:
                n_draw = max(n_det * _OVERSAMPLE, 32)
                pos, fwd = _dart_samples(tc, d_det, n_draw, cursor)
                cursor += n_draw
                good = _valid_ball_views(fs, d_det, pos, fwd, culls)
                kept = emit(target, pos, fwd, good[:n_det])
                if kept:
                    stats["detail_targets"] += 1
                    stats["detail_views"] += kept
    return cams, culls, chosen, stats


# --- the root shell (unchanged construction) ---------------------------------------


def _shell_cameras(
    lo: np.ndarray, hi: np.ndarray, params: PlanParams
) -> tuple[np.ndarray, np.ndarray]:
    """The root establishing shell, SHAPE-ADAPTIVE: Fibonacci directions from
    the scene centre projected onto the scene AABB inflated by a per-axis
    standoff (constants block), aims Halton-jittered over the scene's middle,
    below-floor positions culled. Positions lie beyond the Stage-2 grid margin
    — provably open air, no validity filters needed."""
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
        sel = np.unique(
            np.linspace(0, len(pos) - 1, params.shell_views).astype(np.int64)
        )
        pos = pos[sel]
    jitter = (_halton3(len(pos), start=911) - 0.5) * (0.7 * half)
    fwd = (center + jitter) - pos
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    return pos, fwd


def _flood_cameras(
    fs: FreeSpace,
    lo: np.ndarray,
    hi: np.ndarray,
    params: PlanParams,
    scene_diag: float,
) -> tuple[np.ndarray, np.ndarray]:
    """INTERIOR FLOOD-FILL, stacked on the per-object balls: fill the reachable
    free space with look-around cameras — the wall / floor / ceiling coverage
    (and the BAKED shadows on them) the object-centric darts never frame. A
    lattice over the scene AABB is culled to cells sitting in reachable EMPTY air
    at least `flood_clear` from any surface (so no view is jammed against a wall);
    each survivor fires `flood_dirs` Fibonacci directions, kept when the ray sees
    cover somewhere in the scene (PRESENT — no BLOCKED test: a flood camera just
    renders whatever it faces). The lattice cell scales with the scene so the cell
    COUNT is scene-size-stable, and two caps bound the layer by deterministic
    subsample. Returns (positions, forwards)."""
    if not params.flood:
        return np.zeros((0, 3)), np.zeros((0, 3))
    spacing = max(params.flood_spacing_min, params.flood_spacing_frac * scene_diag)
    axes = [np.arange(lo[i] + spacing / 2.0, hi[i], spacing) for i in range(3)]
    if any(a.size == 0 for a in axes):
        return np.zeros((0, 3)), np.zeros((0, 3))
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    # Positions in reachable EMPTY air, clear of any surface by `flood_clear` — so
    # every emitted view frames content from at least that far (never jammed).
    pos0 = grid[_clear_at(fs, grid, params.flood_clear)]
    if pos0.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    if pos0.shape[0] > params.flood_max_positions:
        sel = np.unique(
            np.linspace(0, pos0.shape[0] - 1, params.flood_max_positions).astype(np.int64)
        )
        pos0 = pos0[sel]

    dirs = _fib_sphere(max(int(params.flood_dirs), 1))
    pos = np.repeat(pos0, len(dirs), axis=0)
    fwd = np.tile(dirs, (pos0.shape[0], 1))

    # PRESENT: some cover (opaque or glass) along the ray within the scene.
    t_max = np.full(pos.shape[0], scene_diag + 2.0, dtype=np.float64)
    _, any_hit = _ray_probe(fs, pos, fwd, t_max)
    keep = np.isfinite(any_hit)
    pos, fwd = pos[keep], fwd[keep]
    if pos.shape[0] > params.flood_max_views:
        sel = np.unique(
            np.linspace(0, pos.shape[0] - 1, params.flood_max_views).astype(np.int64)
        )
        pos, fwd = pos[sel], fwd[sel]
    return pos, fwd


def _surface_sweep_cameras(
    fs: FreeSpace,
    targets: list[Target],
    lo: np.ndarray,
    hi: np.ndarray,
    params: PlanParams,
    scene_diag: float,
    culls: dict[str, int],
) -> list[dict[str, Any]]:
    """Tile LARGE FLAT targets (ceiling / floor / walls / big panels) from their
    INTERIOR side — the coverage the omnidirectional per-object darts fail to
    place on a big overhead or vertical plane. Such a target's AABB diagonal is
    huge, so `standoff_frac`·diag can't fit the room and every room-side
    (looking-AT-the-surface) dart is culled, leaving the plane to train into a
    cloud. A target qualifies when its thinnest axis ≤ `sweep_thin_max` and its
    face area ≥ `sweep_area_min`; each air-facing side is gridded at
    `sweep_spacing` and every cell viewed from `sweep_standoff` at a perpendicular
    + two oblique angles (`sweep_oblique` gives the parallax that triangulates the
    floaters away). Validated by the shared ball filter (culls fold into the
    shared tally); capped by deterministic subsample."""
    if not params.sweep:
        return []
    d = min(params.sweep_standoff, max(_RING_SCENE_CAP * scene_diag, params.standoff_min))
    ob = params.sweep_oblique
    cams: list[dict[str, Any]] = []
    for t in targets:
        size = t.hi - t.lo
        a = int(np.argmin(size))
        b, c = [i for i in range(3) if i != a]
        if size[a] > params.sweep_thin_max or size[b] * size[c] < params.sweep_area_min:
            continue
        nb = max(2, int(np.ceil(size[b] / params.sweep_spacing)))
        nc = max(2, int(np.ceil(size[c] / params.sweep_spacing)))
        gb = t.lo[b] + (np.arange(nb) + 0.5) * (size[b] / nb)
        gc = t.lo[c] + (np.arange(nc) + 0.5) * (size[c] / nc)
        GB, GC = np.meshgrid(gb, gc, indexing="ij")
        surf = np.zeros((GB.size, 3))
        surf[:, a] = (t.lo[a] + t.hi[a]) / 2.0
        surf[:, b] = GB.ravel()
        surf[:, c] = GC.ravel()
        eb = np.zeros(3); eb[b] = 1.0
        ec = np.zeros(3); ec[c] = 1.0
        # A BOUNDARY flat (ceiling / floor / outer wall — its face sits at the
        # scene AABB bound on axis `a`) is swept only from its INWARD side; the
        # outward side is the exterior roof / sub-floor / outside that no interior
        # viewer sees (and, being open air, would otherwise pass the reachability
        # gate and waste budget). An INTERIOR flat (a freestanding partition) is
        # swept from both air-facing sides.
        margin = max(0.5, size[a] * 2.0)
        near_hi = bool(t.hi[a] >= hi[a] - margin)
        near_lo = bool(t.lo[a] <= lo[a] + margin)
        if near_hi and not near_lo:
            sides = (-1.0,)
        elif near_lo and not near_hi:
            sides = (1.0,)
        else:
            sides = (1.0, -1.0)
        for s in sides:
            n = np.zeros(3); n[a] = s
            # skip a side with no reachable air at the camera standoff distance
            if not fs.empty_at(surf + n * (size[a] * 0.5 + d)).any():
                continue
            base = surf + n * (size[a] * 0.5)  # the visible face points
            for off_b, off_c in ((0.0, 0.0), (ob, 0.0), (0.0, ob)):
                pos = base + n * d + eb * (off_b * d) + ec * (off_c * d)
                fwd = base - pos
                fwd = fwd / np.linalg.norm(fwd, axis=1, keepdims=True)
                d_eff = d * float(np.sqrt(1.0 + off_b * off_b + off_c * off_c))
                good = _valid_ball_views(fs, d_eff, pos, fwd, culls)
                for k in good:
                    cams.append({
                        "pos": pos[k], "forward": fwd[k],
                        "kind": "sweep", "zone": t.zone, "target": t.id,
                    })
    if len(cams) > params.sweep_max_views:
        sel = np.unique(
            np.linspace(0, len(cams) - 1, params.sweep_max_views).astype(np.int64)
        )
        cams = [cams[i] for i in sel]
    return cams


# --- ORB v2: residual enclosed-space cubemap stations -----------------------------


def _brick_fields(fs: FreeSpace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(cover, opaque, empty) dense flat bool arrays over the brick grid, plus
    `bdims`. The brick grid is volume/512, so it always holds dense even when
    the fine grid cannot — which is what makes a free-space distance field and
    a brick-resolution ray march affordable at any scene size."""
    bd = fs.bdims
    n = int(bd[0] * bd[1] * bd[2])
    nyz = int(fs.dims[1]) * int(fs.dims[2])
    nz = int(fs.dims[2])

    def to_bricks(lin: np.ndarray) -> np.ndarray:
        out = np.zeros(n, dtype=bool)
        for i in range(0, lin.size, 1 << 24):
            s = lin[i : i + (1 << 24)]
            x = s // nyz
            rem = s % nyz
            out[((x >> 3) * bd[1] + ((rem // nz) >> 3)) * bd[2] + ((rem % nz) >> 3)] = True
        return out

    empty = np.zeros(n, dtype=bool)
    if fs.skin_lin.size:
        # A skin brick is part air, part surface; it is only a plausible
        # VIEWPOINT if it is mostly air (a brick that is 90% solid is not a
        # place a camera stands, but it is still air a ray may cross).
        frac = np.unpackbits(np.asarray(fs.skin_empty), axis=1).sum(axis=1)
        empty[fs.skin_lin[frac > 0]] = True
    if fs.zone_lin.size:
        empty[fs.zone_lin] = True
    return to_bricks(fs.occ_lin), to_bricks(fs.occ_lin_opaque), empty, bd


def _clearance_field(cover: np.ndarray, empty: np.ndarray, bd: np.ndarray, edge: float) -> np.ndarray:
    """Metres from every cover-free reachable brick to the nearest brick that is
    not one — the free-space distance field the orb placement reads. This is
    what replaces a lattice pitch: it is a property of the SCENE's own geometry,
    so a 0.9 m stall and a 12 m hall are both described in the same units and
    neither needs a scene-diagonal parameter to be found."""
    from scipy import ndimage

    shape = (int(bd[0]), int(bd[1]), int(bd[2]))
    free = (empty & ~cover).reshape(shape)
    return ndimage.distance_transform_edt(free, sampling=edge).ravel().astype(np.float32)


def _brick_centers_at(blin: np.ndarray, bd: np.ndarray, fs: FreeSpace) -> np.ndarray:
    """(N,3) world centres of brick linear ids."""
    bx = blin // (bd[1] * bd[2])
    by = (blin // bd[2]) % bd[1]
    bz = blin % bd[2]
    cells = np.stack([bx, by, bz], axis=1).astype(np.float64) * _BRICK + _BRICK / 2.0
    return fs.origin + cells * fs.pitch


def _camera_hits(
    fs: FreeSpace, bd: np.ndarray, opaque: np.ndarray,
    pos: np.ndarray, fwd: np.ndarray, up: np.ndarray,
    fov: np.ndarray, k: int,
) -> list[np.ndarray]:
    """Opaque brick ids each camera sees (deduped per camera).

    This is the common visibility primitive for ORB v2. The old planner collapsed
    it immediately to raw image counts, which made 24 overlapping frames at one
    origin look like 24 independent geometric observations. Returning incidence
    lists preserves the station identity: six cube faces are unioned before one
    distinct-station contribution is applied."""
    out = [np.zeros(0, dtype=np.int64) for _ in range(len(pos))]
    if not len(pos):
        return out
    step = _BRICK * fs.pitch * 0.5
    n_steps = int(float(np.linalg.norm(fs.dims.astype(np.float64) * fs.pitch)) / step) + 2
    s1, s2 = int(bd[1]), int(bd[2])
    fov = np.broadcast_to(np.asarray(fov, dtype=np.float64), (len(pos),))
    base = (np.arange(k) + 0.5) / k * 2.0 - 1.0
    chunk = max(1, int(1_200_000 / (k * k)))
    for c0 in range(0, len(pos), chunk):
        c1 = min(c0 + chunk, len(pos))
        d_parts = []
        for ci in range(c0, c1):
            z = fwd[ci] / np.linalg.norm(fwd[ci])
            y = -up[ci] / np.linalg.norm(up[ci])
            x = np.cross(y, z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            a = base * np.tan(np.radians(fov[ci]) / 2.0)
            uu, vv = np.meshgrid(a, a, indexing="ij")
            d = z[None, :] + uu.reshape(-1, 1) * x[None, :] + vv.reshape(-1, 1) * y[None, :]
            d_parts.append(d / np.linalg.norm(d, axis=1, keepdims=True))
        org = np.repeat(pos[c0:c1], k * k, axis=0)
        dr = np.concatenate(d_parts)
        local = np.repeat(np.arange(c1 - c0, dtype=np.int64), k * k)
        alive = np.ones(len(dr), dtype=bool)
        hit = np.full(len(dr), -1, dtype=np.int64)
        t = step
        for _ in range(n_steps):
            live = np.nonzero(alive)[0]
            if not live.size:
                break
            cell = np.floor((org[live] + dr[live] * t) / fs.pitch).astype(np.int64) - fs.imin
            inb = np.all((cell >= 0) & (cell < fs.dims), axis=1)
            alive[live[~inb]] = False
            live = live[inb]
            if live.size:
                b = cell[inb] >> 3
                blin = (b[:, 0] * s1 + b[:, 1]) * s2 + b[:, 2]
                h = opaque[blin]
                hit[live[h]] = blin[h]
                alive[live[h]] = False
            t += step
        got = hit >= 0
        for li in range(c1 - c0):
            out[c0 + li] = np.unique(hit[got & (local == li)])
    return out


def _plan_orbs(
    fs: FreeSpace, lo: np.ndarray, hi: np.ndarray, placed: list[dict[str, Any]],
    params: PlanParams, progress: ProgressCb | None,
    fields: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """ORB v2: enclosure-weighted cubemap stations selected by residual gain.

    No semantic rooms. Candidate positions are reachable free-space cells whose
    spherical escape fraction says "enclosed". The existing inward-facing plan
    establishes a station-aware baseline, then each candidate's six cube faces
    are scored only against the remaining interior deficit. Selected faces are
    ordinary 90° perspective cameras; faces with no marginal value are omitted."""
    if not params.orb:
        return [], {}
    from scipy import ndimage

    cover, opaque, empty, bd = fields
    edge = _BRICK * fs.pitch
    shape = (int(bd[0]), int(bd[1]), int(bd[2]))
    clear = _clearance_field(cover, empty, bd, edge)
    near_clear = ndimage.maximum_filter(clear.reshape(shape), size=3).ravel()
    reachable_surface = cover & (near_clear >= params.orb_min_clear)
    n_surface = int(reachable_surface.sum())
    if not n_surface:
        return [], {"stations": 0, "views": 0, "target_bricks": 0}

    # Candidate pool: camera-clear, inside the real scene AABB, coarsened only to
    # bound planning work. The final locations come from gain, not this lattice.
    free_ids = np.nonzero((empty & ~cover) & (clear >= params.orb_min_clear))[0]
    xyz = _brick_centers_at(free_ids, bd, fs)
    inside = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    free_ids, xyz = free_ids[inside], xyz[inside]
    if not len(free_ids):
        return [], {"stations": 0, "views": 0, "target_bricks": n_surface}
    cell = np.floor(xyz / max(params.orb_candidate_spacing, edge)).astype(np.int64)
    _, take = np.unique(cell, axis=0, return_index=True)
    take.sort()
    free_ids, xyz = free_ids[take], xyz[take]
    if len(free_ids) > params.orb_candidate_cap:
        sel = np.unique(
            np.linspace(0, len(free_ids) - 1, params.orb_candidate_cap).astype(np.int64)
        )
        free_ids, xyz = free_ids[sel], xyz[sel]

    # Enclosure = one minus escape solid angle. A sampled ray escapes when it
    # leaves the Stage-2 grid before an opaque hit. Outdoors score near zero;
    # rooms / corridors / furnished pockets score near one.
    esc_dirs = _fib_sphere(max(int(params.orb_escape_dirs), 6))
    esc_pos = np.repeat(xyz, len(esc_dirs), axis=0)
    esc_fwd = np.tile(esc_dirs, (len(xyz), 1))
    t_max = np.full(len(esc_fwd), float(np.linalg.norm(fs.dims * fs.pitch)) + edge)
    hit, _ = _ray_probe(fs, esc_pos, esc_fwd, t_max)
    enclosed = np.isfinite(hit).reshape(len(xyz), len(esc_dirs)).mean(axis=1)
    keep = enclosed >= params.orb_enclosure_min
    free_ids, xyz, enclosed = free_ids[keep], xyz[keep], enclosed[keep]
    if not len(free_ids):
        return [], {
            "stations": 0, "views": 0, "target_bricks": n_surface,
            "candidates": 0, "enclosure_min": params.orb_enclosure_min,
        }

    k = max(4, int(params.orb_measure_rays))
    # Existing cameras count by DISTINCT baseline cells, not images. Hash one
    # metre cells; overlapping frames from one optical centre contribute once.
    station_count = np.zeros(len(cover), dtype=np.int16)
    if placed:
        ppos = np.asarray([c["pos"] for c in placed], dtype=np.float64)
        pfwd = np.asarray([c["forward"] for c in placed], dtype=np.float64)
        pup = _up_for(pfwd)
        pfov = np.asarray(
            [float(c.get("fov") or params.fov_deg) for c in placed], dtype=np.float64
        )
        phits = _camera_hits(fs, bd, opaque, ppos, pfwd, pup, pfov, k)
        pcell = np.floor(ppos / 1.0).astype(np.int64)
        _, groups = np.unique(pcell, axis=0, return_inverse=True)
        base_groups: list[np.ndarray] = []
        for gid in range(int(groups.max()) + 1 if len(groups) else 0):
            ids = [phits[i] for i in np.nonzero(groups == gid)[0] if len(phits[i])]
            if ids:
                union = np.unique(np.concatenate(ids))
                base_groups.append(union)
                station_count[union] += 1
    else:
        base_groups = []

    def weights() -> np.ndarray:
        idx = np.clip(station_count, 0, len(_ORB_DEFICIT_WEIGHTS) - 1)
        w = _ORB_DEFICIT_WEIGHTS[idx].copy()
        w[station_count >= params.orb_target_stations] = 0
        w[~reachable_surface] = 0
        return w

    cube_up = _up_for(_ORB_CUBE_DIRS)
    # A patch is an ORB target only when at least one ENCLOSED candidate can see
    # it. This is the mathematical exterior gate: outdoor surface already served
    # by inward-facing cameras never enters the residual objective merely because
    # a distant indoor ray can graze it.
    interior_surface = np.zeros(len(cover), dtype=bool)

    # Precompute each candidate's six face-hit sets once. This is the planning
    # cost; greedy selection below is then sparse integer arithmetic.
    face_hits: list[list[np.ndarray]] = []
    batch = 256
    if progress is not None:
        progress(0, len(xyz), "orb-candidates")
    for c0 in range(0, len(xyz), batch):
        q = xyz[c0 : c0 + batch]
        pos = np.repeat(q, 6, axis=0)
        fwd = np.tile(_ORB_CUBE_DIRS, (len(q), 1))
        up = np.tile(cube_up, (len(q), 1))
        got = _camera_hits(fs, bd, opaque, pos, fwd, up, np.full(len(pos), 90.0), k)
        for h in got:
            interior_surface[h] = True
        face_hits.extend([got[i : i + 6] for i in range(0, len(got), 6)])
        if progress is not None:
            progress(min(c0 + batch, len(xyz)), len(xyz), "orb-candidates")

    reachable_surface &= interior_surface
    n_surface = int(reachable_surface.sum())
    station_count[:] = 0
    for union in base_groups:
        station_count[union] += 1
    before = station_count.copy()

    selected = np.zeros(len(xyz), dtype=bool)
    cams: list[dict[str, Any]] = []
    n_stations = 0

    # Bucketed residual selector. Re-evaluate all candidates once per batch, then
    # commit a spatially diverse top set before refreshing deficits. This preserves
    # marginal-value behavior without the O(candidates × stations) full rescore
    # that made large scenes impractical.
    select_batch = 32
    while len(cams) < params.orb_max_views:
        w = weights()
        if not w.any():
            break
        scored: list[tuple[float, int, list[int]]] = []
        for i, faces in enumerate(face_hits):
            if selected[i]:
                continue
            gains = [float(w[h].sum()) if len(h) else 0.0 for h in faces]
            useful = [fi for fi, gain in enumerate(gains) if gain >= params.orb_min_face_gain]
            if useful:
                union = np.unique(np.concatenate([faces[fi] for fi in useful]))
                gain = float(w[union].sum()) * float(enclosed[i])
                if gain >= params.orb_min_face_gain:
                    scored.append((gain, i, useful))
        if not scored:
            break
        scored.sort(key=lambda x: (-x[0], x[1]))
        accepted = 0
        batch_cells: list[np.ndarray] = []
        for _gain, best_i, best_faces in scored:
            # Don't take several near-identical winners from the same pocket in
            # one stale-deficit batch; refresh first after a station-sized spread.
            if batch_cells:
                d2 = [float(np.sum((xyz[best_i] - p) ** 2)) for p in batch_cells]
                if min(d2) < params.orb_candidate_spacing**2:
                    continue
            room = params.orb_max_views - len(cams)
            best_faces = best_faces[:room]
            if not best_faces:
                break
            selected[best_i] = True
            station_union = np.unique(
                np.concatenate([face_hits[best_i][fi] for fi in best_faces])
            )
            station_count[station_union] += 1
            for fi in best_faces:
                cams.append({
                    "pos": xyz[best_i], "forward": _ORB_CUBE_DIRS[fi],
                    "kind": "orb", "zone": None, "fov": 90.0,
                    "station": n_stations, "face": fi,
                })
            batch_cells.append(xyz[best_i])
            n_stations += 1
            accepted += 1
            if accepted >= select_batch or len(cams) >= params.orb_max_views:
                break
        if not accepted:
            break
        if progress is not None:
            progress(len(cams), params.orb_max_views, "orb-select")

    target = reachable_surface
    before_ok = (
        float((before[target] >= params.orb_target_stations).mean()) if n_surface else 1.0
    )
    after_ok = (
        float((station_count[target] >= params.orb_target_stations).mean())
        if n_surface else 1.0
    )
    return cams, {
        "stations": n_stations,
        "views": len(cams),
        "candidates": int(len(xyz)),
        "target_bricks": n_surface,
        "target_stations": params.orb_target_stations,
        "reached_before": round(before_ok, 4),
        "reached_after": round(after_ok, 4),
        "median_stations": int(np.median(station_count[target])) if n_surface else 0,
        "enclosure_min": params.orb_enclosure_min,
        "measure_rays": k,
    }


def _subject_distance(
    fs: FreeSpace, bd: np.ndarray, opaque: np.ndarray,
    pos: np.ndarray, fwd: np.ndarray,
) -> np.ndarray:
    """Distance from each camera to the first opaque surface along its axis
    (∞ when it faces open space). One march for the whole plan, so every layer
    is measured the same way instead of each threading its own notion of a
    standoff down from wherever it was chosen."""
    s1, s2 = int(bd[1]), int(bd[2])
    step = _BRICK * fs.pitch * 0.5
    far = float(np.linalg.norm(fs.dims.astype(np.float64) * fs.pitch))
    out = np.full(len(pos), np.inf)
    alive = np.ones(len(pos), dtype=bool)
    t = step
    while alive.any() and t <= far:
        live = np.nonzero(alive)[0]
        cell = np.floor((pos[live] + fwd[live] * t) / fs.pitch).astype(np.int64) - fs.imin
        inb = np.all((cell >= 0) & (cell < fs.dims), axis=1)
        alive[live[~inb]] = False
        live = live[inb]
        if live.size:
            b = cell[inb] >> 3
            blin = (b[:, 0] * s1 + b[:, 1]) * s2 + b[:, 2]
            h = opaque[blin]
            out[live[h]] = t
            alive[live[h]] = False
        t += step
    return out


def _derive_fovs(dist: np.ndarray, edge: float, params: PlanParams) -> np.ndarray:
    """Per-camera FOV in degrees from each camera's own subject distance.

    Solve for the angle that puts a feature of the grid's own scale (`edge`, one
    brick) across `fov_ref_px` pixels at distance `d`:

        px = edge · f / d,   f = (R/2) / tan(fov/2)
        ⇒ tan(fov/2) = (R/2) · edge / (fov_ref_px · d)

    So the angle opens up as the subject gets closer and closes as it recedes,
    which is exactly the trade the two cells measured in opposite directions.
    Clamped to [`fov_min_deg`, `fov_max_deg`]: past the wide end the pixels stop
    buying anything a splat can represent, and past the narrow end a frame sees
    too little context to register against its neighbours. A camera facing open
    space has no subject, so it takes the narrow end — it exists to prove that
    volume empty, and a tight frame does that at the best angular resolution."""
    if not params.fov_derive:
        return np.full(len(dist), float(params.fov_deg))
    d = np.where(np.isfinite(dist), np.maximum(dist, 1e-3), np.inf)
    tan_half = (params.render_resolution / 2.0) * edge / (max(params.fov_ref_px, 1e-6) * d)
    fov = 2.0 * np.degrees(np.arctan(tan_half))
    return np.clip(fov, params.fov_min_deg, params.fov_max_deg)


def _up_for(forward: np.ndarray) -> np.ndarray:
    """A non-degenerate image-up per forward direction: world +Y, except near
    the poles where ±Z takes over (straight-down lawn views stay well-posed)."""
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
    Deterministic (Halton + seeded cursors; no RNG state), pure CPU, and
    independent of the surfel cloud. Returns a summary."""
    fs = load_free_space(Path(freespace_path))
    targets, lo, hi = _read_targets(Path(scene_path))
    scene_diag = float(np.linalg.norm(hi - lo))
    if progress is not None:
        progress(0, 0, "load")

    # 1) BALLS — the per-object tight shells (the guarantee layer).
    ball_cams, culls, standoffs, stats = _plan_balls(
        fs, targets, params, scene_diag, progress
    )
    n_before = len(ball_cams)
    if n_before:
        pos = np.asarray([c["pos"] for c in ball_cams])
        fwd = np.asarray([c["forward"] for c in ball_cams])
        keep = set(_thin_lattice_aim(pos, fwd, params.dedupe_spacing).tolist())
        # The dedupe must not strip a target's LAST camera (clustered props'
        # shells coincide, and the shared winner belongs to one id): re-admit
        # the first surviving view of any target thinned to zero.
        have = {ball_cams[i]["target"] for i in keep}
        for i, c in enumerate(ball_cams):
            if c["target"] not in have:
                keep.add(i)
                have.add(c["target"])
        ball_cams = [ball_cams[i] for i in sorted(keep)]
    culls["dedupe"] = n_before - len(ball_cams)

    framed = {c["target"] for c in ball_cams}
    starved = sorted(t.id for t in targets if t.id not in framed)

    # 2) SHELL — the root establishing orbit.
    if progress is not None:
        progress(0, 0, "shell")
    shell_pos, shell_fwd = _shell_cameras(lo, hi, params)

    cams: list[dict[str, Any]] = list(ball_cams)
    for p, d in zip(shell_pos, shell_fwd):
        cams.append({"pos": p, "forward": d, "kind": "shell", "zone": None})

    # 3) FLOOD — legacy A/B layer, off by default. ORB v2 owns interior free
    # space; enabling this deliberately stacks the old lattice for comparison.
    flood_pos = np.zeros((0, 3))
    if params.flood:
        if progress is not None:
            progress(0, 0, "flood")
        flood_pos, flood_fwd = _flood_cameras(fs, lo, hi, params, scene_diag)
        for p, d in zip(flood_pos, flood_fwd):
            cams.append({"pos": p, "forward": d, "kind": "flood", "zone": None})

    # 4) SWEEP — tile LARGE FLAT surfaces (ceiling / floor / walls / panels) from
    # their interior side: the coverage the omnidirectional per-object darts fail
    # to place on a big overhead / vertical plane (module docstring). Reuses the
    # ball validity filter; its culls fold into the shared tally.
    if progress is not None and params.sweep:
        progress(0, 0, "sweep")
    sweep_cams = _surface_sweep_cameras(fs, targets, lo, hi, params, scene_diag, culls)
    cams.extend(sweep_cams)

    # 5) ORBS v2 — enclosed-space cubemap stations selected against the residual
    # left by every layer above. Runs last because only marginal value matters.
    fields = _brick_fields(fs)
    orb_cams, orb_stats = _plan_orbs(fs, lo, hi, cams, params, progress, fields)
    cams.extend(orb_cams)

    fwd_all = np.asarray([c["forward"] for c in cams], dtype=np.float64)
    up_all = _up_for(fwd_all) if len(cams) else np.zeros((0, 3))

    # 6) PER-CAMERA FOV, derived from each camera's own subject distance
    # (`_derive_fovs`). Done here rather than inside each layer so every camera
    # is measured the same way — one march, against the geometry, instead of
    # five different notions of "how far away is what I am pointing at".
    if progress is not None:
        progress(0, 0, "fov")
    pos_all = np.asarray([c["pos"] for c in cams], dtype=np.float64)
    subj = (
        _subject_distance(fs, fields[3], fields[1], pos_all, fwd_all)
        if len(cams) else np.zeros(0)
    )
    fov_all = _derive_fovs(subj, _BRICK * fs.pitch, params)
    fov_shared = float(np.median(fov_all)) if len(cams) else float(params.fov_deg)

    near = min(0.05, fs.pitch * 0.5)
    far = (1.0 + _SHELL_RADIUS_MULT) * scene_diag + 2.0

    doc = {
        "plan_version": PLAN_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": run,
        "slot": slot,
        "model": model,
        "intrinsics": {
            "resolution": int(params.render_resolution),
            # The plan's MEDIAN angle. Every camera also carries its own `fov`;
            # this is what a consumer that ignores that field should use, and it
            # is a median rather than the nominal so such a consumer is at least
            # representative of the plan rather than of a default.
            "fov_deg": round(fov_shared, 4),
            "fov_per_camera": bool(params.fov_derive),
            "fov_deg_min": round(float(fov_all.min()), 4) if len(cams) else fov_shared,
            "fov_deg_max": round(float(fov_all.max()), 4) if len(cams) else fov_shared,
            "near": near,
            "far": far,
        },
        "cameras": [
            {
                "pos": [round(float(v), 5) for v in c["pos"]],
                "forward": [round(float(v), 6) for v in fwd_all[i]],
                "up": [round(float(v), 6) for v in up_all[i]],
                "fov": round(float(fov_all[i]), 4),
                "kind": c["kind"],
                "zone": c["zone"],
            }
            for i, c in enumerate(cams)
        ],
    }

    d_arr = np.asarray(standoffs) if standoffs else np.zeros(1)
    summary = {
        "run": run,
        "slot": slot,
        "model": model,
        "cameras": len(cams),
        "views": len(cams),  # one shot per camera — kept for status parity
        "kinds": {
            "ball": len(ball_cams),
            "shell": int(len(shell_pos)),
            "flood": int(len(flood_pos)),
            "sweep": len(sweep_cams),
            "orb": len(orb_cams),
        },
        "orb": orb_stats,
        "targets": {
            "total": len(targets),
            "framed": len(framed),
            "descended": stats["descended"],
            "detail_ringed": stats["detail_targets"],
            "starved": len(starved),
            "starved_ids": starved[:8],
        },
        "detail_views": stats["detail_views"],
        "culls": culls,
        "standoff_m": {
            "min": round(float(d_arr.min()), 3),
            "median": round(float(np.median(d_arr)), 3),
            "max": round(float(d_arr.max()), 3),
        },
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
