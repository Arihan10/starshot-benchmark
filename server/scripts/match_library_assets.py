"""Match every object in a run collection to the asset library and bake the
placed meshes to disk — the post-hoc twin of the live `_match_library_assets`.

The live pipeline, per object, distills a visual subject, matches it to the
optimized asset library on gemini-3.1-flash-lite, then bakes the chosen asset
into `objects/<id>.glb` (yaw + a bbox-fill transform, via app.utils.glb_place)
and copies its reference `objects/<id>.png`. That build is what a cell renders.

Runs shared as a bare `events.jsonl` (no `objects/` dir) still carry every
flash-lite decision the run made — one `library.match` event per object — plus
the `bbox` that places it. This script walks each cell and rebuilds the
`objects/` set from those events + the library alone (the optimized GLBs are
Meshopt/KTX2-compressed, so the placement is baked as a node transform without
decoding geometry). An object with no recorded match is matched fresh on
gemini-3.1-flash-lite here, and its `library.match`/`image`/`model` events are
appended so the cell renders and a re-run is idempotent.

A "cell" is one `<scene>/<model>/` dir (anything with an events.jsonl).
"Objects" are the `bbox` events whose `node_kind` is not "zone" (i.e. the
"object"/"frame" leaves generation emits), matching the set the live pipeline
matches. Assets missing from the library, or lacking augmented bounds, mirror
the live fallbacks (skip / copy-through unscaled).

Idempotent: an object whose `objects/<id>.glb` already exists is skipped unless
--force. Re-baking always starts from the pristine source asset.

Usage (from server/):
  uv run python scripts/match_library_assets.py --dry-run       # audit, no writes
  uv run python scripts/match_library_assets.py                 # bake every cell
  uv run python scripts/match_library_assets.py --cell modern-house/gemini-flash
  uv run python scripts/match_library_assets.py --force         # re-bake existing
  uv run python scripts/match_library_assets.py --rematch       # re-run flash-lite for every object
  uv run python scripts/match_library_assets.py --runs-dir ../runs/collection-gemini-flash
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SERVER_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_SERVER_DIR / ".env")
load_dotenv()

from app.core.types import BoundingBox  # noqa: E402
from app.services import library  # noqa: E402
from app.utils import glb_place, logging  # noqa: E402

_REPO_ROOT = _SERVER_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from starshot_paths import runs_root  # noqa: E402

DEFAULT_RUNS_DIR = runs_root() / "collection-gemini-flash"
OBJECTS_SUBDIR = "objects"

# Match the live fan-out cap (generation.LIBRARY_MATCH_CONCURRENCY) so a
# --rematch bounds concurrent flash-lite calls the same way.
MATCH_CONCURRENCY = 12


@dataclass
class ObjectSpec:
    node_id: str
    bbox: BoundingBox
    orientation: int
    seed_prompt: str


@dataclass
class CellData:
    objects: list[ObjectSpec]
    match_by_id: dict[str, str]
    subject_by_id: dict[str, str]
    model_ids: set[str]


@dataclass
class CellResult:
    baked: int = 0
    reused: int = 0
    llm_matched: int = 0
    already: int = 0
    bounds_missing: int = 0
    asset_missing: int = 0
    unmatched_skipped: int = 0
    would_match: int = 0
    errors: list[str] = field(default_factory=list)

    def add(self, other: CellResult) -> None:
        for k in (
            "baked", "reused", "llm_matched", "already",
            "bounds_missing", "asset_missing", "unmatched_skipped", "would_match",
        ):
            setattr(self, k, getattr(self, k) + getattr(other, k))


def _parse_cell(events: list[dict]) -> CellData:
    """Recover the object set + recorded matches/subjects from a cell's events.

    Objects keep first-seen (bbox emission) order. The last `library.match` /
    `image` for an id wins, mirroring `logging.find_event`'s latest-wins read."""
    objects: list[ObjectSpec] = []
    seen: set[str] = set()
    match_by_id: dict[str, str] = {}
    subject_by_id: dict[str, str] = {}
    model_ids: set[str] = set()
    for e in events:
        kind = e.get("kind")
        if kind == "bbox":
            node_id = e.get("id")
            if not node_id or node_id in seen or e.get("node_kind") == "zone":
                continue
            origin, dims = e.get("origin"), e.get("dimensions")
            if not origin or not dims:
                continue
            seen.add(node_id)
            objects.append(
                ObjectSpec(
                    node_id=node_id,
                    bbox=BoundingBox(origin=tuple(origin), dimensions=tuple(dims)),
                    orientation=int(e.get("orientation") or 0),
                    seed_prompt=e.get("prompt") or "",
                )
            )
        elif kind == "library.match":
            node_id, lib = e.get("id"), e.get("library_id")
            if node_id and lib:
                match_by_id[node_id] = lib
        elif kind == "image":
            node_id, prompt = e.get("id"), e.get("prompt")
            if node_id and prompt:
                subject_by_id[node_id] = prompt
        elif kind == "model":
            node_id = e.get("id")
            if node_id:
                model_ids.add(node_id)
    return CellData(objects, match_by_id, subject_by_id, model_ids)


def _artifact_url(runs_dir: Path, path: Path) -> str:
    return f"/artifacts/{path.relative_to(runs_dir).as_posix()}"


def _classify(library_id: str, orientation: int) -> str:
    """Bake outcome WITHOUT writing — for --dry-run. "baked" | "bounds_missing"
    | "asset_missing", same branches `_bake` takes."""
    if not library.asset_path(library_id).exists():
        return "asset_missing"
    if library.asset_rotated_bounds(library_id, orientation) is None:
        return "bounds_missing"
    return "baked"


def _bake(
    *, objs_dir: Path, node_id: str, library_id: str, bbox: BoundingBox, orientation: int
) -> str:
    """Copy the matched asset's reference PNG and bake its placed GLB into
    `objs_dir`, mirroring `_match_library_assets`. Returns "baked" |
    "bounds_missing" (copied through unscaled) | "asset_missing"."""
    asset = library.asset_path(library_id)
    glb = objs_dir / f"{node_id}.glb"

    ref_png = asset.with_suffix(".png")
    if ref_png.exists():
        dest_png = objs_dir / f"{node_id}.png"
        dest_png.unlink(missing_ok=True)
        shutil.copy2(ref_png, dest_png)

    if not asset.exists():
        return "asset_missing"

    bounds = library.asset_rotated_bounds(library_id, orientation)
    glb.unlink(missing_ok=True)
    if bounds is None:
        shutil.copyfile(asset, glb)
        return "bounds_missing"
    glb_place.place_glb(
        src=asset,
        dst=glb,
        bbox=bbox,
        orientation=orientation,
        rotated_min=bounds[0],
        rotated_max=bounds[1],
    )
    return "baked"


def _record_fresh_match(
    slot_log: logging.SlotLog,
    *,
    runs_dir: Path,
    objs_dir: Path,
    node_id: str,
    subject: str,
    library_id: str,
    outcome: str,
) -> None:
    """Append the events the live match writes for a freshly matched object, so
    the cell renders it and a re-run reuses the decision. The flash-lite call's
    `cache.llm` was already appended inside `library.match`."""
    slot_log.log("library.match", id=node_id, prompt=subject, library_id=library_id)
    png = objs_dir / f"{node_id}.png"
    if png.exists():
        slot_log.log("image", id=node_id, url=_artifact_url(runs_dir, png), prompt=subject)
    if outcome != "asset_missing":
        glb = objs_dir / f"{node_id}.glb"
        slot_log.log("model", id=node_id, artifact_kind="object", url=_artifact_url(runs_dir, glb))


def _tally(result: CellResult, *, from_llm: bool, outcome: str) -> None:
    if from_llm:
        result.llm_matched += 1
    else:
        result.reused += 1
    if outcome == "asset_missing":
        result.asset_missing += 1
    else:
        if outcome == "bounds_missing":
            result.bounds_missing += 1
        result.baked += 1


async def process_cell(
    cell_dir: Path, runs_dir: Path, *, force: bool, rematch: bool, dry_run: bool, have_key: bool
) -> CellResult | None:
    events_path = cell_dir / "events.jsonl"
    if not events_path.exists():
        return None

    slot_id = cell_dir.relative_to(runs_dir).as_posix()
    slot_log = logging.SlotLog(slot_id, events_path)
    slot_log.hydrate_from_disk()
    logging.bind(slot_log)

    cell = _parse_cell(slot_log.state["events"])
    result = CellResult()
    if not cell.objects:
        return result

    objs_dir = cell_dir / OBJECTS_SUBDIR
    if not dry_run:
        objs_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(MATCH_CONCURRENCY)

    async def _one(spec: ObjectSpec) -> None:
        glb = objs_dir / f"{spec.node_id}.glb"
        if glb.exists() and not force and not rematch:
            result.already += 1
            return

        committed = cell.match_by_id.get(spec.node_id)
        want_fresh = rematch or committed is None
        can_llm = have_key and not dry_run

        if want_fresh and can_llm:
            try:
                subject = cell.subject_by_id.get(spec.node_id) or spec.seed_prompt
                async with sem:
                    library_id = (await library.match(subject)).library_id
                from_llm = True
            except Exception as e:  # noqa: BLE001
                result.errors.append(f"{spec.node_id}: match failed: {type(e).__name__}: {e}")
                return
        elif committed is not None:
            library_id, from_llm = committed, False
            if want_fresh and dry_run:
                result.would_match += 1  # --rematch would re-run flash-lite here
        else:
            # No recorded match and no way (or no intent) to make one now.
            if dry_run:
                result.would_match += 1
            else:
                result.unmatched_skipped += 1
            return

        if dry_run:
            _tally(result, from_llm=from_llm, outcome=_classify(library_id, spec.orientation))
            return

        try:
            outcome = _bake(
                objs_dir=objs_dir,
                node_id=spec.node_id,
                library_id=library_id,
                bbox=spec.bbox,
                orientation=spec.orientation,
            )
        except Exception as e:  # noqa: BLE001
            result.errors.append(f"{spec.node_id}: bake failed: {type(e).__name__}: {e}")
            return

        if from_llm:
            subject = cell.subject_by_id.get(spec.node_id) or spec.seed_prompt
            _record_fresh_match(
                slot_log, runs_dir=runs_dir, objs_dir=objs_dir,
                node_id=spec.node_id, subject=subject, library_id=library_id, outcome=outcome,
            )
        _tally(result, from_llm=from_llm, outcome=outcome)

    await asyncio.gather(*(_one(s) for s in cell.objects))
    slot_log.close()
    return result


def _discover_cells(runs_dir: Path, cell: str | None) -> list[Path]:
    if cell:
        return [runs_dir / cell]
    return sorted(
        {p.parent for p in runs_dir.rglob("events.jsonl") if "_branches" not in p.parts}
    )


def _fmt(result: CellResult, *, dry_run: bool, label: str) -> str:
    extra = ""
    if result.would_match:
        extra += f" would_match={result.would_match}"
    if result.unmatched_skipped:
        extra += f" unmatched_skipped={result.unmatched_skipped}"
    if result.errors:
        extra += f" ERRORS={len(result.errors)}"
    verb = "dry" if dry_run else "bake"
    return (
        f"[{verb}] {label}: baked={result.baked} reused={result.reused} "
        f"llm={result.llm_matched} already={result.already} "
        f"bounds_missing={result.bounds_missing} asset_missing={result.asset_missing}{extra}"
    )


async def _amain() -> None:
    parser = argparse.ArgumentParser(
        description="Match objects to the library and bake objects/ for existing runs."
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--cell", type=str, default=None, help="<scene>/<model> under runs-dir; omit for all")
    parser.add_argument("--force", action="store_true", help="re-bake even if objects/<id>.glb exists")
    parser.add_argument(
        "--rematch", action="store_true",
        help="re-run gemini-3.1-flash-lite for every object (ignore recorded matches)",
    )
    parser.add_argument("--dry-run", action="store_true", help="audit only; no LLM calls, no writes")
    args = parser.parse_args()

    runs_dir = args.runs_dir.resolve()
    if not runs_dir.is_dir():
        print(f"runs dir not found: {runs_dir}")
        sys.exit(1)

    have_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    if args.rematch and not args.dry_run and not have_key:
        print("[warn] --rematch set but OPENROUTER_API_KEY is unset — falling back to recorded matches\n")

    cells = _discover_cells(runs_dir, args.cell)
    grand = CellResult()
    n_cells = 0
    for cell_dir in cells:
        rel = cell_dir.relative_to(runs_dir).as_posix()
        result = await process_cell(
            cell_dir, runs_dir,
            force=args.force, rematch=args.rematch, dry_run=args.dry_run, have_key=have_key,
        )
        if result is None:
            print(f"[skip] {rel}: no events.jsonl")
            continue
        n_cells += 1
        grand.add(result)
        grand.errors.extend(f"{rel}/{msg}" for msg in result.errors)
        print(_fmt(result, dry_run=args.dry_run, label=rel), flush=True)

    print("\n" + _fmt(grand, dry_run=args.dry_run, label=f"{n_cells} cell(s)"), flush=True)
    if grand.errors:
        print(f"[errors] {len(grand.errors)} object(s) failed:", flush=True)
        for msg in grand.errors:
            print(f"  - {msg}", flush=True)


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
