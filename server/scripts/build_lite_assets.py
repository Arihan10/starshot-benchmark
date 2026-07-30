#!/usr/bin/env python3
"""Offline per-cell asset-tier builder for starshot-benchmark runs.

Ported from the Unity replay's raw-lite tier. Runs each raw GLB through the
shared optimizer (server/tools/optimize-assets/optimize.mjs) under a chosen
`--preset` (default `lite`), producing a drop-in twin directory:

  lite   presentation tier — weld + dedup + error-bounded (near-lossless)
         simplify + prune + Meshopt geometry compression + KTX2/UASTC textures.
         Visually identical to raw at ~5x smaller on disk.
  splat  splat-pipeline sample/render tier — optimized-grade decimation with
         1024px KTX2/ETC1S base color, all other PBR maps stripped (the splat
         pipeline is unlit). The single asset source every splat stage reads.

On-disk contract (matches the Unity replay's AssetSource.ResolveMeshPath, so the
Starshot Replay window loads the output without any further wiring):

    library   cell:  <cell>/objects            ->  <cell>/objects-lite
    generated cell:  <cell>/generated/<v>/objects-generated
                                          ->  <cell>/generated/<v>/objects-generated-lite

Each <id>.glb (excluding the pristine <id>.raw.glb) is built, and the <id>.png
reference is copied across so the output is a drop-in set. Existing outputs are
skipped unless missing, older than their source, or --force is given, so the run
is safe to cancel and resume.

Usage:
    # one cell's generated version 2
    python server/scripts/build_lite_assets.py --runs runs \\
        --filter good_opus_new_hotel2/hotel-room/opus-new --version 2

    # one cell's library set
    python server/scripts/build_lite_assets.py --runs runs \\
        --filter <run>/<slot>/<model> --version library

    # a batch from the Unity picker's targets file ({"targets":[{run,slot,model,version}]})
    python server/scripts/build_lite_assets.py --runs runs --targets-file targets.json

For the shared asset LIBRARY (assets/ -> assets-lite/), run the optimizer directly:
    node optimize.mjs --preset lite --input <assets> --output <assets-lite>

Scope note: this ports the geometry+texture "lite" tier only. The Unity replay's
--masks-only (HDRP mask KTX2 sidecars) and --rtproxy (ray-tracing proxies) tiers
are HDRP-runtime specific and are intentionally NOT part of this port.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]  # server/scripts -> server -> repo
sys.path.insert(0, str(_REPO_ROOT))

from starshot_paths import runs_root  # noqa: E402

_OPTIMIZE_DIR = _REPO_ROOT / "server" / "tools" / "optimize-assets"
_OPTIMIZE_SCRIPT = _OPTIMIZE_DIR / "optimize.mjs"
_NODE_BIN = os.environ.get("STARSHOT_NODE_BIN", "node")
_DEFAULT_RUNS = str(runs_root())


@dataclass
class Target:
    """One (cell, variant-set) whose raw GLBs get a lite twin."""

    cell: str  # run/slot/model, for logging
    label: str  # "library" or "v<n>"
    src_dir: Path
    out_dir: Path


def _fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"[lite] {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def _tier_dirs(cell_dir: Path, version: str) -> tuple[Path, Path]:
    """Source (raw) and destination (lite) dirs for a cell's variant set."""
    if version == "library":
        return cell_dir / "objects", cell_dir / "objects-lite"
    return (
        cell_dir / "generated" / version / "objects-generated",
        cell_dir / "generated" / version / "objects-generated-lite",
    )


def _source_glbs(src_dir: Path) -> list[Path]:
    """Placed source meshes to build from — the pristine <id>.raw.glb is excluded."""
    return sorted(p for p in src_dir.glob("*.glb") if not p.name.endswith(".raw.glb"))


def _is_fresh(src: Path, out: Path) -> bool:
    try:
        return out.exists() and out.stat().st_mtime >= src.stat().st_mtime
    except OSError:
        return False


def _copy_png(src_glb: Path, out_dir: Path, force: bool) -> None:
    """Mirror the <id>.png reference so the lite dir is a drop-in set."""
    png = src_glb.with_suffix(".png")
    if not png.exists():
        return
    dst = out_dir / png.name
    if force or not dst.exists() or _is_fresh(png, dst) is False:
        try:
            shutil.copy2(png, dst)
        except OSError:
            pass


def _build_one(target: Target, src: Path, force: bool, preset: str = "lite") -> str:
    """Build one GLB twin via `optimize.mjs --preset <preset>`. Returns done|skip."""
    out = target.out_dir / src.name
    if not force and _is_fresh(src, out):
        _copy_png(src, target.out_dir, force)
        return "skip"

    target.out_dir.mkdir(parents=True, exist_ok=True)
    # Atomic write. The temp MUST keep the .glb extension: gltf-transform's
    # NodeIO.write picks binary-GLB vs glTF+external-resources off the path
    # suffix, so a ".part" temp would spill a JSON stub + loose .bin/.ktx2.
    tmp = out.with_name(out.stem + ".opt-tmp.glb")
    try:
        proc = subprocess.run(
            [
                _NODE_BIN, str(_OPTIMIZE_SCRIPT), "--preset", preset,
                "--file", str(src), "--out-file", str(tmp),
            ],
            cwd=str(_OPTIMIZE_DIR),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not tmp.exists():
            detail = (proc.stderr or proc.stdout or "optimize.mjs failed").strip()
            # Skip node's "    at ..." stack frames — keep the real Error message.
            lines = [
                ln.strip() for ln in detail.splitlines()
                if ln.strip() and not ln.strip().startswith("at ")
            ]
            raise RuntimeError(lines[-1] if lines else "optimize.mjs failed")
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    _copy_png(src, target.out_dir, force)
    return "done"


def _gather_targets(args: argparse.Namespace) -> list[Target]:
    # Explicit source/output dirs (how the server's build-lite endpoint drives
    # us): bypass the run/filter/version layout resolution entirely.
    if args.src_dir or args.out_dir:
        if not (args.src_dir and args.out_dir):
            _fail("--src-dir and --out-dir must be given together")
        # Resolve to absolute: optimize.mjs runs with cwd=optimize-assets, so a
        # relative --file/--out-file would resolve against the wrong dir.
        src_dir = Path(args.src_dir).resolve()
        out_dir = Path(args.out_dir).resolve()
        if not src_dir.is_dir():
            _fail(f"no source dir {src_dir}")
        return [Target(cell=src_dir.parent.name, label="explicit",
                       src_dir=src_dir, out_dir=out_dir)]

    runs = Path(args.runs).resolve()
    requested: list[tuple[Path, str]] = []  # (cell_dir, version)

    if args.targets_file:
        try:
            data = json.loads(Path(args.targets_file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _fail(f"could not read --targets-file: {exc}")
        for entry in data.get("targets", []):
            try:
                cell = runs / entry["run"] / entry["slot"] / entry["model"]
            except (KeyError, TypeError):
                continue
            requested.append((cell, str(entry.get("version") or "library")))
    elif args.filter:
        parts = [p for p in re.split(r"[\\/]+", args.filter.strip("\\/")) if p]
        if len(parts) != 3:
            _fail(f"--filter must be <run>/<slot>/<model>, got: {args.filter!r}")
        requested.append((runs.joinpath(*parts), args.version))
    else:
        _fail("pass --filter <run>/<slot>/<model> (with --version), or --targets-file <json>")

    targets: list[Target] = []
    for cell_dir, version in requested:
        src_dir, out_dir = _tier_dirs(cell_dir, version)
        label = "library" if version == "library" else f"v{version}"
        cell = f"{cell_dir.parent.parent.name}/{cell_dir.parent.name}/{cell_dir.name}"
        if not src_dir.is_dir():
            print(f"[lite] skip {cell} ({label}): no source dir {src_dir}",
                  file=sys.stderr, flush=True)
            continue
        targets.append(Target(cell=cell, label=label, src_dir=src_dir, out_dir=out_dir))
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the lite asset tier for runs.")
    parser.add_argument(
        "--runs", default=_DEFAULT_RUNS,
        help="runs root (default: $STARSHOT_RUNS_DIR, else repo/runs)",
    )
    parser.add_argument("--filter", help="single cell as <run>/<slot>/<model>")
    parser.add_argument("--version", default="library",
                        help='asset set: "library" or a generated version number (default: library)')
    parser.add_argument("--targets-file", help='JSON {"targets":[{run,slot,model,version}]}')
    parser.add_argument("--src-dir", help="explicit raw source dir (bypasses --runs/--filter/--version)")
    parser.add_argument("--out-dir", help="explicit output dir (used with --src-dir)")
    parser.add_argument("--preset", default="lite", choices=("lite", "splat"),
                        help="optimizer preset for the twin (default: lite)")
    parser.add_argument("--force", action="store_true", help="rebuild even if the lite GLB is up to date")
    parser.add_argument("--limit", type=int, default=0, help="build at most N files (0 = all; for smoke tests)")
    parser.add_argument("--concurrency", type=int, default=4, help="parallel optimizer processes")
    # Accepted for CLI-compatibility with the Unity replay window; out of scope here.
    parser.add_argument("--masks-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rtproxy", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checks", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.masks_only or args.rtproxy:
        _fail("--masks-only / --rtproxy (HDRP mask + RT-proxy tiers) are not part of "
              "this port — only the geometry+texture lite tier is built.")
    if not _OPTIMIZE_SCRIPT.exists():
        _fail(f"optimizer not found at {_OPTIMIZE_SCRIPT}")

    targets = _gather_targets(args)
    work = [(t, src) for t in targets for src in _source_glbs(t.src_dir)]
    if args.limit and args.limit > 0:
        work = work[: args.limit]
    total = len(work)
    if total == 0:
        print("[lite] nothing to build (no source .glb files)", flush=True)
        return 0

    print(f"[lite] {total} file(s) across {len(targets)} set(s) "
          f"-> preset {args.preset}, concurrency {args.concurrency}", flush=True)

    done = built = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {pool.submit(_build_one, t, src, args.force, args.preset): (t, src) for t, src in work}
        for future in as_completed(futures):
            target, src = futures[future]
            done += 1
            try:
                status = future.result()
            except Exception as exc:  # noqa: BLE001 - report per-file, keep going
                failed += 1
                print(f"[{done}/{total}] {src.stem} ({target.label}) FAILED: {exc}",
                      file=sys.stderr, flush=True)
                continue
            if status == "done":
                built += 1
            else:
                skipped += 1
            print(f"[{done}/{total}] {src.stem} ({target.label}) {status}", flush=True)

    print(f"[lite] finished: {built} built, {skipped} skipped, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
