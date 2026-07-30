"""CLI: decode Stage-5 SZF reference frames to RGBA PNGs (coverage alpha kept, depth dropped).

Thin wrapper over `splat.colmap.decode_frames_to_png`, which reuses the canonical
SZF codec in `splat.stage5`.

    python szf_to_png.py [SRC_FRAMES_DIR] [DST_DIR] [-j JOBS]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splat import colmap
from starshot_paths import runs_root

_REFS = runs_root() / "ahhhhhhhh/test-SH/gemini-flash-lite/splat/refs"
DEFAULT_SRC = _REFS / "frames"
DEFAULT_DST = _REFS / "frames-1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", type=Path, default=DEFAULT_SRC)
    ap.add_argument("dst", nargs="?", type=Path, default=DEFAULT_DST)
    ap.add_argument("-j", "--jobs", type=int, default=None)
    args = ap.parse_args()

    n = colmap.decode_frames_to_png(args.src, args.dst, jobs=args.jobs)
    if n == 0:
        raise SystemExit(f"no SZF frames in {args.src}")
    print(f"wrote {n} PNGs -> {args.dst}")


if __name__ == "__main__":
    main()
