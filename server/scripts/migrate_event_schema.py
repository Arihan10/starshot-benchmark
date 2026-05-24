"""One-time migration: translate pre-namespacing event logs to the
new schema (`runware.<stage>.{submit,reattach,retry,done}`,
`google.banana.{submit,retry,done}`, `job_id`/`task_id`).

Pre-migration events used flat names (`runware.submit`,
`nano_banana.done`, `cache.artifact`) keyed by `node_id`/`task_uuid`.
The resumable wrapper added in app.utils.resumable looks for the
namespaced kinds with `job_id`/`task_id`; without translation the
recovery script's `find_stuck` returns 0 and the wrapper can't
reattach to already-billed Runware tasks.

Trellis `input_hash` is also recomputed in the new format
(flat dict over the trellis_args, no outer `arguments` wrapper) so
the wrapper's `find_prior_submit` matches and reattach fires.
The hash uses the banana remote_url as the `image` field — exactly
what the new generate_mesh sees when called with that URL.

Each events.jsonl is rewritten in place; a `.bak` is dropped beside
it on first run so re-running the migration is safe.

Usage (from server/):
    uv run python scripts/migrate_event_schema.py <slot_dir>
    uv run python scripts/migrate_event_schema.py --runs-dir ../zone_plan_iter_3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

# Mirror the production defaults so we recompute the same hash the
# new threed.generate_mesh will produce when it sees the same image URL.
TRELLIS_MODEL = os.environ.get("TRELLIS_MODEL", "microsoft:trellis-2@4b")
TRELLIS_REMESH = False
TRELLIS_RESOLUTION = 512
TRELLIS_TEXTURE_SIZE = 1024


def _hash_new(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _new_trellis_input_hash(image_url: str) -> str:
    """Reproduces `resumable.hash_input(trellis_args)` from threed.py for
    the case where the image is a remote URL."""
    return _hash_new({
        "model": TRELLIS_MODEL,
        "image": image_url,
        "remesh": TRELLIS_REMESH,
        "resolution": TRELLIS_RESOLUTION,
        "textureSize": TRELLIS_TEXTURE_SIZE,
        "outputFormat": "GLB",
        "outputType": "URL",
    })


def translate(event: dict, banana_urls: dict[str, str]) -> dict:
    """Return the new-schema form of `event`. Pass-through for kinds
    that don't need translation."""
    kind = event.get("kind")

    if kind == "runware.submit":
        stage = event.get("stage")
        if stage not in ("banana", "trellis"):
            return event
        base = {
            "index": event.get("index"),
            "kind": f"runware.{stage}.submit",
            "job_id": event["node_id"],
            "task_id": event["task_uuid"],
            "attempt": 0,
        }
        if stage == "trellis":
            url = banana_urls.get(event["node_id"])
            if url:
                base["input_hash"] = _new_trellis_input_hash(url)
            else:
                # Banana never completed for this node — leave the old
                # hash so we at least preserve audit; reattach won't
                # match anyway because there's no image to feed Trellis.
                base["input_hash"] = event.get("input_hash", "")
        else:
            base["input_hash"] = event.get("input_hash", "")
        return base

    if kind == "runware.reattach":
        stage = event.get("stage")
        if stage not in ("banana", "trellis"):
            return event
        out = {
            "index": event.get("index"),
            "kind": f"runware.{stage}.reattach",
            "scope": f"runware.{stage}",
            "job_id": event["node_id"],
            "task_id": event["task_uuid"],
            "outcome": event["outcome"],
        }
        if "reason" in event:
            out["reason"] = event["reason"]
        return out

    if kind == "nano_banana.done":
        # Banana is google-direct in the new world; the saved PNG bytes
        # are still valid as Trellis input regardless of provenance.
        # Preserve remote_url — the recovery script feeds it back to
        # Trellis so the input_hash matches the migrated submit.
        out = {
            "index": event.get("index"),
            "kind": "google.banana.done",
            "job_id": event["node_id"],
            "saved": event["saved"],
            "mime_type": "image/png",
        }
        if event.get("remote_url"):
            out["remote_url"] = event["remote_url"]
        return out

    if kind == "cache.artifact":
        return {
            "index": event.get("index"),
            "kind": "runware.trellis.done",
            "job_id": event["node_id"],
            "saved": event["raw_glb_path"],
        }

    if kind == "banana.retry":
        return {**event, "kind": "google.banana.retry"}
    if kind == "trellis.retry":
        return {**event, "kind": "runware.trellis.retry"}
    if kind == "nano_banana.retry":
        return {**event, "kind": "google.banana.retry"}

    # cache.artifact.hit, nano_banana.skip, run.start, run.done, etc.
    # all pass through unchanged.
    return event


def migrate_file(path: Path) -> tuple[int, int]:
    """Rewrite `path` in place. Returns (translated_count, total)."""
    lines = path.read_text().splitlines()
    events: list[dict] = []
    for line in lines:
        if not line.strip():
            continue
        events.append(json.loads(line))

    # First pass: build node_id -> banana remote_url map. Needed by the
    # second pass to recompute trellis input_hash.
    banana_urls: dict[str, str] = {}
    for e in events:
        if e.get("kind") == "nano_banana.done":
            url = e.get("remote_url")
            if url:
                banana_urls[e["node_id"]] = url

    # Backup before touching (only on first run — re-runs preserve the
    # original .bak so we don't overwrite the pre-migration state).
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        bak.write_bytes(path.read_bytes())

    translated = 0
    with path.open("w") as f:
        for e in events:
            new = translate(e, banana_urls)
            if new is not e:
                translated += 1
            f.write(json.dumps(new) + "\n")
    return translated, len(events)


def main(target: Path) -> None:
    if target.is_file():
        slot_files = [target]
    elif target.is_dir():
        # Single slot dir vs. runs dir containing many slots.
        if (target / "events.jsonl").exists():
            slot_files = [target / "events.jsonl"]
        else:
            slot_files = sorted(
                p / "events.jsonl"
                for p in target.iterdir()
                if p.is_dir() and (p / "events.jsonl").exists()
            )
    else:
        print(f"no such path: {target}", file=sys.stderr)
        sys.exit(2)

    if not slot_files:
        print(f"no events.jsonl under {target}", file=sys.stderr)
        sys.exit(2)

    print(f"migrating {len(slot_files)} file(s):")
    total_translated = total_events = 0
    for p in slot_files:
        translated, total = migrate_file(p)
        total_translated += translated
        total_events += total
        print(f"  {p.relative_to(target.parent if target.is_dir() else target.parent)}"
              f": {translated}/{total} translated")
    print(f"done: {total_translated}/{total_events} events translated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        nargs="?",
        type=Path,
        default=None,
        help="slot dir or single events.jsonl. Mutually exclusive with --runs-dir.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="runs directory; every immediate subdir with an events.jsonl is migrated.",
    )
    args = parser.parse_args()
    target = args.target or args.runs_dir
    if target is None:
        parser.error("provide a slot dir or --runs-dir")
    main(target.resolve())
