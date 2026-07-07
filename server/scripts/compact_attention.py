"""Shrink stored attention dumps on disk — the /tf attention caches.

Attention results are stored per cell under `.../attention/`:

  * `{ev}.json`            small COMPACT view (served directly) — keep as-is
  * `{ev}.full.json`       heavy per-token/per-head source (~100MB each), re-pullable
                           from Modal on demand — the biggest hog
  * `.derived/{ev}.vN.*`   compact/present/tokens sidecars DERIVED from the full —
                           fully rebuildable on demand

The heavy `full.json` and the derived `tokens.ndjson` are (a) the same per-token
detail stored twice and (b) plain JSON, which is ~10x gzip-compressible. This
script exploits both:

  1. legacy heal   — a `{ev}.json` that is actually the giant INLINE full blob is
                     down-projected to the compact and its heavy source preserved
                     GZIPPED beside it (`{ev}.full.json.gz`).
  2. compress      — every plain `{ev}.full.json` is gzipped to `{ev}.full.json.gz`
                     (~10x) and the plain copy dropped. The server reads either form.
  3. drop derived  — the whole `.derived/` cache is removed; it rebuilds on demand
                     in the new (compressed) format the first time a step's
                     present/token detail is opened.

Every heavy source ends up at least as new as its compact so the server treats it
as current and never needlessly re-pulls it. Idempotent + safe to re-run.

Usage (from server/):
    uv run python scripts/compact_attention.py --dry-run          # report only
    uv run python scripts/compact_attention.py                    # convert all runs
    uv run python scripts/compact_attention.py --runs-dir ../foo  # a specific tree
    uv run python scripts/compact_attention.py --keep-derived     # don't drop .derived
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.attention import derive as attn_derive  # build_compact / source_paths (pure)

_COMPACT_RE = re.compile(r'"compact"\s*:\s*true')
_CHUNK = 1 << 20


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _is_compact_file(path: Path) -> bool:
    """True if `{ev}.json` already leads with the compact marker (cheap head peek —
    never parses a possibly-huge legacy file)."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return _COMPACT_RE.search(f.read(4096)) is not None
    except OSError:
        return False


def _gzip_file(src: Path, dst_gz: Path) -> None:
    """Stream-compress src -> dst_gz atomically (never holds the blob in RAM)."""
    tmp = dst_gz.with_name(f".{dst_gz.name}.{os.getpid()}.tmp")
    try:
        with src.open("rb") as fin, gzip.open(tmp, "wb", compresslevel=6) as fout:
            shutil.copyfileobj(fin, fout, length=_CHUNK)
        os.replace(tmp, dst_gz)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_gz_text(dst_gz: Path, text: str) -> None:
    tmp = dst_gz.with_name(f".{dst_gz.name}.{os.getpid()}.tmp")
    try:
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as f:
            f.write(text)
        os.replace(tmp, dst_gz)
    finally:
        tmp.unlink(missing_ok=True)


def _touch_newer(path: Path, ref: Path) -> None:
    """Ensure `path` mtime >= `ref` mtime (so the server sees the source as current
    and never re-pulls it). No-op if `ref` is missing."""
    try:
        rm = ref.stat().st_mtime
    except OSError:
        return
    try:
        if path.stat().st_mtime < rm:
            now = time.time()
            os.utime(path, (now, now))
    except OSError:
        pass


class Stats:
    def __init__(self) -> None:
        self.healed = 0
        self.compressed = 0
        self.derived_removed = 0
        self.warmed = 0
        self.freed = 0          # bytes reclaimed (net)
        self.errors = 0


def _source_evs(attn: Path) -> list[int]:
    """Event indices under `attn/` that have a heavy source (gz or plain)."""
    evs: set[int] = set()
    for pat, strip in (("*.full.json.gz", len(".full.json.gz")), ("*.full.json", len(".full.json"))):
        for p in attn.glob(pat):
            try:
                evs.add(int(p.name[:-strip]))
            except ValueError:
                continue
    return sorted(evs)


def _warm_attention_dir(attn: Path, *, dry_run: bool, tokens: bool, st: Stats) -> None:
    """Pre-build the cheap present/compact caches (so present/overview is a fast
    cache hit instead of a heavy source parse). `tokens` also builds the big
    per-token ndjson. Skips steps already warm."""
    evs = _source_evs(attn)
    if not evs:
        return
    print(f"    warm {len(evs)} step(s){' (+tokens)' if tokens else ''}")
    if dry_run:
        st.warmed += len(evs)
        return
    for ev in evs:
        try:
            if attn_derive.warm(attn.parent, ev, tokens=tokens) is not None:
                st.warmed += 1
        except Exception:  # noqa: BLE001 — a bad source must not abort the whole warm
            st.errors += 1


def _process_attention_dir(attn: Path, *, dry_run: bool, drop_derived: bool, st: Stats) -> None:
    # 1. Legacy heal: any `{ev}.json` that is the inline full blob → compact + gz source.
    for path in sorted(attn.glob("*.json")):
        try:
            ev = int(path.stem)
        except ValueError:
            continue  # not an `{ev}.json`
        if _is_compact_file(path):
            continue  # already small
        gz, plain = attn_derive.source_paths(attn.parent, ev)
        before = _size(path)
        try:
            full = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            st.errors += 1
            continue
        if not isinstance(full, dict) or full.get("compact"):
            continue
        print(f"    heal legacy inline {path.name} ({_human(before)})")
        if dry_run:
            st.healed += 1
            st.freed += before - before // 10  # compact is tiny; gz source ~10x smaller
            continue
        try:
            compact = attn_derive.build_compact(full)
            if not gz.is_file() and not plain.is_file():
                _write_gz_text(gz, json.dumps(full))  # preserve heavy source, compressed
            _atomic_write(path, json.dumps(compact))  # replace giant inline with compact
            _touch_newer(gz, path)
            st.healed += 1
            st.freed += before - _size(path) - _size(gz)
        except (OSError, ValueError, KeyError):
            st.errors += 1
        finally:
            del full

    # 2. Compress every plain `{ev}.full.json` → `.gz`, then drop the plain copy.
    for plain in sorted(attn.glob("*.full.json")):
        try:
            ev = int(plain.name[: -len(".full.json")])
        except ValueError:
            continue
        gz, _ = attn_derive.source_paths(attn.parent, ev)  # {ev}.full.json.gz
        compact = attn / f"{ev}.json"
        before = _size(plain)
        # A newer .gz already covers it — just drop the redundant plain copy.
        if gz.is_file() and _size(gz) > 0 and gz.stat().st_mtime >= plain.stat().st_mtime:
            print(f"    drop redundant plain {plain.name} ({_human(before)}; .gz present)")
            if dry_run:
                st.freed += before
            else:
                plain.unlink(missing_ok=True)
                _touch_newer(gz, compact)
                st.freed += before
            st.compressed += 1
            continue
        print(f"    gzip {plain.name} ({_human(before)})")
        if dry_run:
            st.compressed += 1
            st.freed += before - before // 10  # ~10x on attention JSON
            continue
        try:
            _gzip_file(plain, gz)
            if _size(gz) > 0:
                plain.unlink(missing_ok=True)
                _touch_newer(gz, compact)
                st.freed += before - _size(gz)
                st.compressed += 1
            else:
                st.errors += 1
        except OSError:
            st.errors += 1

    # 3. Drop the derived cache — it rebuilds on demand in the compressed v2 format.
    derived = attn / ".derived"
    if drop_derived and derived.is_dir():
        dsize = sum(_size(p) for p in derived.rglob("*") if p.is_file())
        if dsize:
            print(f"    drop .derived/ cache ({_human(dsize)}, rebuilds on demand)")
            if dry_run:
                st.freed += dsize
            else:
                try:
                    shutil.rmtree(derived)
                    st.freed += dsize
                except OSError:
                    st.errors += 1
            st.derived_removed += 1


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]  # sb-new/ (matches routes._REPO_ROOT)
    default_runs = Path(os.environ.get("STARSHOT_RUNS_DIR", repo_root / "runs"))
    ap = argparse.ArgumentParser(description="Compress + dedupe stored attention dumps.")
    ap.add_argument("--runs-dir", type=Path, default=default_runs, help="root holding runs (default: sb-new/runs)")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    ap.add_argument("--keep-derived", action="store_true", help="don't drop the rebuildable .derived/ cache")
    ap.add_argument("--warm", action="store_true",
                    help="pre-build the cheap present/compact caches so present/overview is fast (small)")
    ap.add_argument("--warm-tokens", action="store_true",
                    help="with --warm, also build the big per-token ndjson (fast token detail, more disk)")
    ap.add_argument("--no-compact", action="store_true",
                    help="skip compression/dedupe; only warm (use with --warm)")
    args = ap.parse_args()

    runs_dir: Path = args.runs_dir
    if not runs_dir.is_dir():
        print(f"runs dir not found: {runs_dir}", file=sys.stderr)
        return 2
    warm = args.warm or args.warm_tokens

    attn_dirs = sorted({p for p in runs_dir.rglob("attention") if p.is_dir()})
    print(f"scanning {runs_dir} — {len(attn_dirs)} attention dir(s){'  [DRY RUN]' if args.dry_run else ''}\n")
    st = Stats()
    for attn in attn_dirs:
        rel = attn.relative_to(runs_dir)
        print(f"  {rel}")
        if not args.no_compact:
            _process_attention_dir(attn, dry_run=args.dry_run, drop_derived=not args.keep_derived, st=st)
        if warm:
            _warm_attention_dir(attn, dry_run=args.dry_run, tokens=args.warm_tokens, st=st)

    print(
        f"\n{'would ' if args.dry_run else ''}heal {st.healed} legacy · "
        f"compress {st.compressed} full.json · drop {st.derived_removed} derived cache(s)"
        + (f" · warm {st.warmed} step(s)" if warm else "")
    )
    print(f"{'estimated ' if args.dry_run else ''}reclaimed: {_human(max(st.freed, 0))}"
          + ("  (run without --dry-run to apply)" if args.dry_run else ""))
    if st.errors:
        print(f"errors: {st.errors} (left in place — safe to re-run)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
