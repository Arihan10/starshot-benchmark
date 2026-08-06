"""Replay the generate gate's failure fold over every generated build on disk.

The fold (`routes._generated_failures`) is what the client's failure panel shows,
so this is the cheapest way to confirm it agrees with reality: for each build it
prints the derived failures next to the objects that genuinely have no served
mesh, and flags any object missing from disk that the fold did not account for.

Usage (from server/):
    uv run python scripts/check_generated_failures.py
    uv run python scripts/check_generated_failures.py --runs-dir ../runs --limit 12
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from app.api import routes
from app.pipeline import generation

DEFAULT_RUNS_DIR = Path(
    os.environ.get(
        "STARSHOT_RUNS_DIR", str(Path(__file__).resolve().parent.parent / "runs"),
    )
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--limit", type=int, default=15, help="builds to print in detail",
    )
    args = parser.parse_args()
    runs_dir: Path = args.runs_dir.resolve()
    if not runs_dir.is_dir():
        print(f"no such runs dir: {runs_dir}", file=sys.stderr)
        return 2

    kinds: collections.Counter[str] = collections.Counter()
    unaccounted_total = 0
    builds: list[tuple[int, str, list[dict[str, object]], list[str]]] = []

    for events_path in sorted(runs_dir.rglob("events.generated.jsonl")):
        version_dir = events_path.parent
        opt_dir = version_dir / generation.GENERATED_OPT_SUBDIR
        canonical_of = routes._generated_prefab(events_path)
        if not canonical_of:
            continue
        rows = routes._generated_failure_rows(
            events_path, canonical_of, opt_dir, set(), include_unbuilt=True,
        )
        for row in rows:
            kinds[str(row["kind"])] += 1
        # Ground truth: an object with no served mesh that the fold never named.
        served = {p.name[: -len(".glb")] for p in opt_dir.glob("*.glb")} if opt_dir.is_dir() else set()
        named = {str(r["id"]) for r in rows}
        unaccounted = sorted(set(canonical_of) - served - named)
        unaccounted_total += len(unaccounted)
        if rows or unaccounted:
            builds.append((len(rows), str(events_path.relative_to(runs_dir)), rows, unaccounted))

    builds.sort(reverse=True)
    for _n, name, rows, unaccounted in builds[: args.limit]:
        print(f"\n{name}")
        by_kind: collections.Counter[str] = collections.Counter(str(r["kind"]) for r in rows)
        print(f"    derived: {dict(by_kind)}")
        for row in rows[:4]:
            stale = "  [prior asset still on disk]" if row.get("stale") else ""
            print(f"      {row['kind']:9s} {row['id']}: {str(row['message'])[:88]}{stale}")
        if len(rows) > 4:
            print(f"      … {len(rows) - 4} more")
        if unaccounted:
            print(f"    UNACCOUNTED (no served mesh, not named by the fold): {len(unaccounted)}")
            print(f"      {', '.join(unaccounted[:8])}")

    print("\n" + "=" * 80)
    print("derived failure kinds across every build:")
    for kind, n in kinds.most_common():
        print(f"  {n:6d}  {kind}")
    print(f"\nobjects with no served mesh that the fold did NOT explain: {unaccounted_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
