"""One-time migration: fold flat top-level ablation VARIANT runs under their base.

Ablation variants used to be flat top-level runs named
``<base>__abl-[<slot>-<model>-][<label>-]<kind>@<cut>-<tag>[-r<rep>]``. They now
live under their base run:

    runs/<base>/ablations/<experiment>/<variant>/<slot>/<model>/...

where ``<experiment>`` = ``ablation.label`` (or the study inferred from the varying
treatment axis) and ``<variant>`` = ``<kind>@<cut>-<tag>[-r<rep>]`` — both derived
from the variant's ``run.json`` ``ablation`` block (the source of truth; the old
folder name was a lossy encoding). See ``app.ablation.config`` for the pure helpers.

The move is a metadata-only ``os.rename`` on the same filesystem (NO copy of the
~100 GB of data). A per-base ``ablations/manifest.json`` index is (re)built so the
board / tf drawer can discover variants with one read instead of scanning `/runs`.

SAFETY:
  * DEFAULT IS DRY-RUN — nothing moves until you pass ``--apply``.
  * The server must be IDLE during ``--apply`` (pause launches) so nothing writes
    into a directory mid-rename.

Usage (from server/):
    uv run python scripts/migrate_ablation_layout.py                 # dry-run report
    uv run python scripts/migrate_ablation_layout.py --apply         # do the moves
    uv run python scripts/migrate_ablation_layout.py --manifest-only # just (re)build manifests
    uv run python scripts/migrate_ablation_layout.py --runs-dir ../foo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ablation import config as abl_config  # noqa: E402
from app.ablation import manifest as abl_manifest  # noqa: E402

RUN_META_NAME = "run.json"


def _repo_root() -> Path:
    # server/scripts/x.py -> sb-new/
    return Path(__file__).resolve().parent.parent.parent


def _read_meta(run_dir: Path) -> dict:
    try:
        data = json.loads((run_dir / RUN_META_NAME).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:6]


def _plan_moves(runs_dir: Path) -> tuple[list[tuple[Path, Path, dict]], list[str], list[str]]:
    """Return (moves, skips, warnings). A `move` is (source_dir, target_dir, abl_meta).
    Only FLAT top-level dirs whose run.json has an `ablation.base_run` are variants."""
    moves: list[tuple[Path, Path, dict]] = []
    skips: list[str] = []
    warnings: list[str] = []
    claimed: dict[Path, Path] = {}  # target -> source (collision detection)

    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        # Already-nested variants (inside a base's ablations/) are skipped — we only
        # scan the top level; a base dir is itself iterated but has no ablation meta.
        meta = _read_meta(d)
        abl = meta.get("ablation")
        if not isinstance(abl, dict) or not abl.get("base_run"):
            continue  # a base run or a non-ablation run — leave it alone
        base = str(abl["base_run"]).strip()
        base_dir = runs_dir / base
        if not base_dir.is_dir():
            warnings.append(f"SKIP {d.name}: base run '{base}' not found on disk")
            skips.append(d.name)
            continue
        rel = abl_config.ablation_rel_path(abl)  # ablations/<exp>/<variant>
        target = base_dir / rel
        # Collision: two distinct variants derive the same nested path — disambiguate
        # with a short hash of the ORIGINAL flat name (stable + unique).
        if target in claimed or target.exists():
            suffixed = target.parent / f"{target.name}-{_short_hash(d.name)}"
            warnings.append(
                f"COLLISION {d.name}: {rel} already claimed → {suffixed.relative_to(base_dir)}"
            )
            target = suffixed
        claimed[target] = d
        # Sanity cross-check: for the short flat form <base>__abl-<suffix>, the
        # recomputed <variant> should equal <suffix> (informational only).
        prefix = f"{base}__abl-"
        if d.name.startswith(prefix):
            old_suffix = d.name[len(prefix):]
            if abl_config.variant_id(abl) not in old_suffix:
                warnings.append(
                    f"NOTE {d.name}: recomputed variant '{abl_config.variant_id(abl)}' "
                    f"not in flat suffix '{old_suffix}' (older naming — using run.json)"
                )
        moves.append((d, target, abl))
    return moves, skips, warnings


def _write_manifest(base_dir: Path, *, apply: bool) -> int:
    """(Re)build a base's ablations manifest (shared helper). Writes only on apply."""
    n = len(abl_manifest.build(base_dir).get("variants", []))
    if apply and n:
        abl_manifest.refresh(base_dir)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_runs = Path(os.environ.get("STARSHOT_RUNS_DIR", _repo_root() / "runs"))
    ap.add_argument("--runs-dir", type=Path, default=default_runs, help=f"runs root (default: {default_runs})")
    ap.add_argument("--apply", action="store_true", help="actually move dirs (default: dry-run report only)")
    ap.add_argument("--manifest-only", action="store_true", help="skip moves; only (re)build ablations manifests")
    args = ap.parse_args()

    runs_dir: Path = args.runs_dir.resolve()
    if not runs_dir.is_dir():
        print(f"runs dir not found: {runs_dir}", file=sys.stderr)
        return 2
    apply = bool(args.apply) and not args.manifest_only
    mode = "MANIFEST-ONLY" if args.manifest_only else ("APPLY" if apply else "DRY-RUN")
    print(f"[migrate-ablation-layout] {mode} · runs={runs_dir}")

    if not args.manifest_only:
        moves, skips, warnings = _plan_moves(runs_dir)
        by_base: dict[str, int] = {}
        for _src, tgt, abl in moves:
            by_base[str(abl["base_run"])] = by_base.get(str(abl["base_run"]), 0) + 1
        print(f"\n{len(moves)} variant(s) to fold; {len(skips)} skipped; {len(warnings)} warning(s)")
        for base, n in sorted(by_base.items()):
            print(f"  {base}: {n} variant(s) -> {base}/{abl_config.ABLATIONS_SUBDIR}/")
        for w in warnings:
            print(f"  ! {w}")
        # Show a small sample of the move mapping so the plan is reviewable.
        for src, tgt, _abl in moves[:12]:
            print(f"    {src.name}\n      -> {tgt.relative_to(runs_dir).as_posix()}")
        if len(moves) > 12:
            print(f"    … {len(moves) - 12} more")

        if apply:
            done = 0
            for src, tgt, _abl in moves:
                tgt.parent.mkdir(parents=True, exist_ok=True)
                os.rename(src, tgt)  # same-filesystem: instant, metadata-only
                done += 1
            print(f"\nmoved {done} variant(s).")
        else:
            print("\n(dry-run — nothing moved; pass --apply to execute)")

    # Rebuild manifests for every base that now has an ablations/ subtree.
    bases = sorted({p.parent.name for p in runs_dir.glob(f"*/{abl_config.ABLATIONS_SUBDIR}") if p.is_dir()})
    base_dirs = [runs_dir / b for b in bases]
    if base_dirs:
        print("\nmanifests:")
        for bd in base_dirs:
            n = _write_manifest(bd, apply=apply or args.manifest_only)
            verb = "wrote" if (apply or args.manifest_only) else "would write"
            print(f"  {verb} {bd.name}/{abl_config.ABLATIONS_SUBDIR}/{abl_manifest.MANIFEST_NAME} · {n} variant(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
