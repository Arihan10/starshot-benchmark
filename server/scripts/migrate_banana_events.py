"""One-shot migration: rewrite legacy Banana completion events as
`google.banana.done` so the new resume path finds the on-disk cache.

Pre-migration runs logged one of:
  * `nano_banana.done` (very old)   — fields: `node_id`, `remote_url`, `saved`
  * `runware.banana.done` (recent)  — fields: `job_id`,  `remote_url`, `saved`

The new resume code (`app.services.nano_banana.generate_resumable`)
looks for `google.banana.done` events keyed by `job_id`. Without this
migration, every node in an existing run looks un-done to the new
code and re-bills Google for the image even though the PNG is still
on disk.

This script walks every `events.jsonl` under a runs directory and,
for each legacy event whose `saved` path still exists on disk,
APPENDS a new `google.banana.done` event with that same `saved`
path. Idempotent: nodes that already carry a `google.banana.done`
are left alone. The legacy events are kept verbatim — append-only.

PAUSE the slot in the originating pipeline first — both this script
and the live server would otherwise append to the same events.jsonl.

Usage (from server/):
    uv run python scripts/migrate_banana_events.py
    uv run python scripts/migrate_banana_events.py --dry-run
    uv run python scripts/migrate_banana_events.py --runs-dir ../runs-bboxes
    uv run python scripts/migrate_banana_events.py some_slot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starshot_paths import runs_root

LEGACY_KINDS = ("nano_banana.done", "runware.banana.done")
NEW_KIND = "google.banana.done"

DEFAULT_RUNS_DIR = runs_root()

_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _mime_for(path: Path) -> str:
    return _MIME_BY_EXT.get(path.suffix.lower(), "image/png")


def _job_id_of(event: dict[str, Any]) -> str | None:
    """Both event shapes carry the per-node identity, but under
    different field names: very-old `nano_banana.done` uses `node_id`,
    `runware.banana.done` uses `job_id`. Normalize to a single string."""
    jid = event.get("job_id")
    if isinstance(jid, str):
        return jid
    nid = event.get("node_id")
    if isinstance(nid, str):
        return nid
    return None


def migrate_slot(
    events_path: Path, *, dry_run: bool,
) -> tuple[int, int, int]:
    """Migrate one slot's events.jsonl in place.

    Returns (added, already_migrated, skipped_missing_file).
    """
    if not events_path.exists():
        return (0, 0, 0)

    events: list[dict[str, Any]] = []
    with events_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"  {events_path}: skipping malformed line",
                    file=sys.stderr,
                )

    already: set[str] = set()
    for e in events:
        if e.get("kind") == NEW_KIND:
            jid = _job_id_of(e)
            if jid:
                already.add(jid)

    # If a node was retried, multiple legacy `.done` events exist —
    # the latest one reflects the surviving artifact.
    legacy_latest: dict[str, dict[str, Any]] = {}
    for e in events:
        if e.get("kind") in LEGACY_KINDS:
            jid = _job_id_of(e)
            if jid:
                legacy_latest[jid] = e

    new_events: list[dict[str, Any]] = []
    missing = 0
    next_index = len(events)

    for jid, legacy in legacy_latest.items():
        if jid in already:
            continue
        saved = legacy.get("saved")
        if not isinstance(saved, str):
            missing += 1
            continue
        saved_path = Path(saved)
        if not saved_path.exists():
            missing += 1
            continue
        migrated: dict[str, Any] = {
            "index": next_index,
            "kind": NEW_KIND,
            "job_id": jid,
            "saved": saved,
            "mime_type": _mime_for(saved_path),
            "migrated_from": legacy.get("kind"),
        }
        # Preserve the Runware-hosted image URL so replay_stuck_meshes.py
        # can reattach to in-flight Trellis tasks: passing the same
        # remote URL keeps Trellis's input_hash stable, while encoding
        # the same image as a fresh base64 data URI would not.
        remote_url = legacy.get("remote_url")
        if isinstance(remote_url, str):
            migrated["remote_url"] = remote_url
        new_events.append(migrated)
        next_index += 1

    if new_events and not dry_run:
        with events_path.open("a") as f:
            for e in new_events:
                f.write(json.dumps(e) + "\n")

    return (len(new_events), len(already), missing)


def _discover_slots(runs_dir: Path) -> list[str]:
    if not runs_dir.is_dir():
        return []
    return sorted(
        p.name for p in runs_dir.iterdir()
        if p.is_dir() and (p / "events.jsonl").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite legacy Banana .done events as google.banana.done.",
    )
    parser.add_argument(
        "slot_id",
        nargs="?",
        default=None,
        help="slot to migrate. Omit to migrate every slot in --runs-dir.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="runs directory to migrate. Defaults to $STARSHOT_RUNS_DIR or <repo>/runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing.",
    )
    args = parser.parse_args()
    runs_dir = args.runs_dir.resolve()

    print(f"runs dir: {runs_dir}")
    print(f"mode:     {'dry-run' if args.dry_run else 'write'}")

    if args.slot_id is None:
        slots = _discover_slots(runs_dir)
        if not slots:
            print(f"no slots with events.jsonl under {runs_dir}", file=sys.stderr)
            sys.exit(2)
        print(f"slots:    {', '.join(slots)}")
    else:
        slots = [args.slot_id]

    total_added = total_already = total_missing = 0
    for slot in slots:
        events_path = runs_dir / slot / "events.jsonl"
        added, already, missing = migrate_slot(
            events_path, dry_run=args.dry_run,
        )
        verb = "would add" if args.dry_run else "added"
        print(
            f"  {slot}: {verb} {added}, "
            f"already-migrated {already}, "
            f"missing-file {missing}"
        )
        total_added += added
        total_already += already
        total_missing += missing

    verb = "would add" if args.dry_run else "added"
    print(
        f"done: {verb} {total_added} new events across {len(slots)} slot(s), "
        f"{total_already} already migrated, "
        f"{total_missing} legacy entries skipped (file missing)"
    )


if __name__ == "__main__":
    main()
