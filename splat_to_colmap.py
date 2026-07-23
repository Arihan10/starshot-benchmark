"""CLI: export a splat cell (Stage-5 refs + Stage-3 surfels) to a COLMAP model.

Thin wrapper over `splat.colmap.export_colmap` — writes cameras.txt / images.txt /
points3D.txt + decoded RGB PNGs into `<SPLAT_DIR>/colmap` (or an explicit OUT_DIR),
ready to drag into Postshot (Camera Poses -> Import).

    python splat_to_colmap.py [SPLAT_DIR] [OUT_DIR] [-j JOBS]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splat import colmap, stage5

DEFAULT_SPLAT = Path("runs/ahhhhhhhh/test-SH/gemini-flash-lite/splat")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("splat_dir", nargs="?", type=Path, default=DEFAULT_SPLAT)
    ap.add_argument("out_dir", nargs="?", type=Path, default=None)
    ap.add_argument("-j", "--jobs", type=int, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or (args.splat_dir / "colmap")
    summary = colmap.export_colmap(
        args.splat_dir / stage5.REFS_DIRNAME,
        args.splat_dir / "cloud.ply",
        out_dir,
        jobs=args.jobs,
    )
    print(
        f"cameras: {summary['cameras']}  images: {summary['images']}  "
        f"points: {summary['points']}\n-> {summary['dir']}"
    )


if __name__ == "__main__":
    main()
