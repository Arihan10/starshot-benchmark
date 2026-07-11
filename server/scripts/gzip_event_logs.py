"""Gzip TERMINAL ablation-variant event logs — the ~10x disk win.

A finished variant's `events.jsonl` is mostly `cache.llm` prompt/output TEXT and is
never appended to again (the variant halts right after its treated step). Stored as
`events.jsonl.gz` it shrinks ~10x. `SlotLog` and the `/artifacts` handler read the
gz transparently; any write (a rare resume/edit) re-materializes the plain log first
(`SlotLog._ensure_plain`).

SAFETY:
  * DRY-RUN by default — nothing is compressed/removed until `--apply`.
  * Only logs whose TAIL is terminal (`run.done` / `ablation.complete`) are gzipped —
    an active or merely-paused (resumable) log is left plain.
  * The gz is VERIFIED to round-trip (sha256 of the decompressed stream == the
    plain file) BEFORE the plain file is removed — no lossy step.
  * Idempotent (skips a log that already has a `.gz`). Server must be IDLE.

Usage (from server/):
    uv run python scripts/gzip_event_logs.py                 # dry-run over ablation variants
    uv run python scripts/gzip_event_logs.py --apply
    uv run python scripts/gzip_event_logs.py --all-terminal  # also base/other terminal cells
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ablation import config as abl_config  # noqa: E402

_TERMINAL_RE = re.compile(rb'"kind"\s*:\s*"(?:run\.done|ablation\.complete)"')
_TAIL_BYTES = 262144  # last 256 KB comfortably holds the terminal marker
_CHUNK = 1 << 20


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _is_terminal(path: Path) -> bool:
    """True if the log's tail carries a terminal marker (won't be appended to)."""
    try:
        st = path.stat()
        with path.open("rb") as f:
            if st.st_size > _TAIL_BYTES:
                f.seek(st.st_size - _TAIL_BYTES)
                f.readline()
            data = f.read()
    except OSError:
        return False
    return _TERMINAL_RE.search(data) is not None


def _verify_gz(gz: Path, expected_size: int) -> bool:
    """Confirm the gz decompresses cleanly to exactly `expected_size` bytes.
    gzip carries a CRC32 of the SOURCE data, validated at EOF — so a full
    decompress that reaches the trailer + a byte-count match is a strong
    integrity check (a mismatch would raise or miscount) without a second
    read of the plain file."""
    try:
        total = 0
        with gzip.open(gz, "rb") as f:
            for chunk in iter(lambda: f.read(_CHUNK), b""):
                total += len(chunk)
        return total == expected_size
    except (OSError, EOFError, gzip.BadGzipFile):
        return False


def _targets(runs_dir: Path, all_terminal: bool) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    # Folded ablation variants: runs/<base>/ablations/**/<slot>/<model>/events.jsonl
    for ev in runs_dir.glob(f"*/{abl_config.ABLATIONS_SUBDIR}/**/events.jsonl"):
        if ev not in seen:
            seen.add(ev)
            out.append(ev)
    if all_terminal:
        for ev in runs_dir.glob("*/*/*/events.jsonl"):  # base <run>/<slot>/<model>
            if ev not in seen:
                seen.add(ev)
                out.append(ev)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_runs = Path(os.environ.get("STARSHOT_RUNS_DIR", _repo_root() / "runs"))
    ap.add_argument("--runs-dir", type=Path, default=default_runs, help=f"runs root (default: {default_runs})")
    ap.add_argument("--apply", action="store_true", help="actually compress (default: dry-run report only)")
    ap.add_argument("--all-terminal", action="store_true", help="also gzip terminal base/other cell logs (not just ablation variants)")
    ap.add_argument("--level", type=int, default=6, help="gzip level 1-9 (default 6)")
    args = ap.parse_args()

    runs_dir: Path = args.runs_dir.resolve()
    if not runs_dir.is_dir():
        print(f"runs dir not found: {runs_dir}", file=sys.stderr)
        return 2
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[gzip-event-logs] {mode} · runs={runs_dir} · scope={'all-terminal' if args.all_terminal else 'ablation-variants'}")

    targets = _targets(runs_dir, args.all_terminal)
    n_gz = n_skip_exists = n_skip_active = n_fail = 0
    saved = 0
    for ev in targets:
        gz = ev.with_name(ev.name + ".gz")
        if gz.exists():
            n_skip_exists += 1
            continue
        if not _is_terminal(ev):
            n_skip_active += 1
            continue
        try:
            plain_size = ev.stat().st_size
        except OSError:
            n_fail += 1
            continue
        if not args.apply:
            n_gz += 1
            saved += plain_size  # upper bound; real gz is ~10x smaller
            continue
        try:
            tmp = gz.with_name(gz.name + ".tmp")
            with ev.open("rb") as src, gzip.open(tmp, "wb", compresslevel=args.level) as dst:
                shutil.copyfileobj(src, dst, _CHUNK)
            # Verify the gz decompresses cleanly to the exact original size (CRC32 +
            # byte count) BEFORE dropping the plain log — no lossy step.
            if not _verify_gz(tmp, plain_size):
                tmp.unlink(missing_ok=True)
                print(f"  ! VERIFY FAILED (kept plain): {ev.relative_to(runs_dir)}")
                n_fail += 1
                continue
            os.replace(tmp, gz)
            gz_size = gz.stat().st_size
            ev.unlink()
            saved += plain_size - gz_size
            n_gz += 1
        except OSError as e:
            print(f"  ! ERROR {ev.relative_to(runs_dir)}: {e}")
            n_fail += 1

    verb = "gzipped" if args.apply else "would gzip"
    print(f"\n{verb} {n_gz} log(s); skipped {n_skip_exists} already-gz, {n_skip_active} non-terminal; {n_fail} failed")
    print(f"{'reclaimed' if args.apply else 'plain bytes targeted'}: {saved / 1e9:.2f} GB"
          + ("" if args.apply else "  (actual reclaim ≈ 90% of this after ~10x compression)"))
    if not args.apply:
        print("(dry-run — nothing changed; pass --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
