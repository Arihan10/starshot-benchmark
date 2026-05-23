"""Recover stuck Trellis tasks for a slot.

When the shared Runware WebSocket wedges, Trellis tasks pile up with a
`runware.submit` logged but no `cache.artifact` landing. The
Runware-side task itself usually finished — we just never received the
response. This script re-invokes `generate_mesh` for each stuck node so
`_submit_trellis`'s task-UUID reattach pulls the result back without
re-billing (and falls through to a fresh submit when the task expired).

Banana now runs directly against Google's GenAI API, so a banana-side
hang is just a hung HTTP request — there's no resumable taskUUID for
that stage, and the Banana-skip gate (`nano_banana.done`) covers the
re-bill window once an image has landed on disk.

PAUSE the slot in the originating pipeline first — both this script
and the live server would otherwise append to the same events.jsonl.

Concurrency is capped at 6 to avoid re-wedging the WS.

If you're running multiple pipeline instances (normal + bboxes-only +
etc.), each has its own STARSHOT_RUNS_DIR. Point this script at the
right one by either exporting the same STARSHOT_RUNS_DIR before
running, or passing --runs-dir.

Usage (from server/):
    uv run python scripts/replay_stuck_meshes.py <slot_id>
    uv run python scripts/replay_stuck_meshes.py <slot_id> --runs-dir ../runs-bboxes
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.prompts import wrap_image_prompt
from app.core.types import BoundingBox, ProxyShape
from app.services import threed
from app.utils import logging as rlog
from app.utils.logging import SlotLog

CONCURRENCY = 6
DEFAULT_RUNS_DIR = Path(
    os.environ.get(
        "STARSHOT_RUNS_DIR",
        str(Path(__file__).resolve().parent.parent / "runs"),
    )
)


def find_stuck(
    events: list[dict],
) -> list[tuple[str, str, BoundingBox, ProxyShape | None]]:
    """Return [(node_id, subject_prompt, bbox, proxy_shape), ...] for nodes
    that have at least one `runware.submit` but no terminal completion
    (`cache.artifact` for a fully-finished mesh, or `mesh.error` for a
    permanent failure)."""
    submitted: set[str] = set()
    artifact_done: set[str] = set()
    mesh_err: set[str] = set()
    bboxes: dict[str, dict] = {}
    mesh_submits: dict[str, dict] = {}
    for e in events:
        k = e.get("kind")
        if k == "runware.submit":
            submitted.add(e["node_id"])
        elif k == "cache.artifact":
            artifact_done.add(e["node_id"])
        elif k == "mesh.error":
            mesh_err.add(e["id"])
        elif k == "bbox":
            bboxes[e["id"]] = e
        elif k == "mesh.submit":
            mesh_submits[e["id"]] = e

    pending: list[tuple[str, str, BoundingBox, ProxyShape | None]] = []
    for nid in submitted:
        if nid in artifact_done or nid in mesh_err:
            continue
        bbox_evt = bboxes.get(nid)
        ms = mesh_submits.get(nid)
        if bbox_evt is None or ms is None:
            print(
                f"skip {nid}: missing bbox or mesh.submit event",
                file=sys.stderr,
            )
            continue
        bbox = BoundingBox(
            origin=tuple(bbox_evt["origin"]),
            dimensions=tuple(bbox_evt["dimensions"]),
        )
        ps = bbox_evt.get("proxy_shape")
        proxy = ProxyShape(ps) if ps else None
        pending.append((nid, ms["prompt"], bbox, proxy))
    return pending


async def main(slot_id: str, runs_dir: Path) -> None:
    if not os.environ.get("RUNWARE_API_KEY"):
        print("RUNWARE_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    run_dir = runs_dir / slot_id
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        print(f"no events at {events_path}", file=sys.stderr)
        sys.exit(2)
    print(f"runs dir: {runs_dir}")

    slot_log = SlotLog(slot_id, events_path)
    slot_log.hydrate_from_disk()
    rlog.bind(slot_log)

    stuck = find_stuck(slot_log.state["events"])
    print(f"{slot_id}: {len(stuck)} stuck mesh tasks to replay")
    if not stuck:
        return

    objs_dir = run_dir / "objects"
    objs_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    ok = 0
    err = 0

    async def replay(
        nid: str,
        subject_prompt: str,
        bbox: BoundingBox,
        proxy: ProxyShape | None,
    ) -> None:
        nonlocal ok, err
        async with sem:
            # wrap_image_prompt is deterministic in (subject_prompt,
            # proxy_shape, bbox.size). The Banana-skip gate reuses the
            # already-saved image, so Trellis sees the same base64 input
            # and `_submit_trellis` reattaches to the prior task_uuid.
            banana_prompt = wrap_image_prompt(subject_prompt, proxy, bbox.size)
            raw = objs_dir / f"{nid}.raw.glb"
            image_stem = objs_dir / nid
            try:
                await threed.generate_mesh(
                    banana_prompt, output_path=raw, image_stem=image_stem,
                )
                ok += 1
                print(f"  ok  {nid}")
            except Exception as e:  # noqa: BLE001
                err += 1
                print(f"  err {nid}: {type(e).__name__}: {e}")

    try:
        await asyncio.gather(*(replay(*s) for s in stuck))
    finally:
        await threed.disconnect_runware()
    print(f"done: {ok} recovered, {err} failed, {len(stuck)} total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("slot_id")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=(
            "runs directory of the pipeline instance to recover. "
            "Defaults to $STARSHOT_RUNS_DIR or server/runs."
        ),
    )
    args = parser.parse_args()
    asyncio.run(main(args.slot_id, args.runs_dir.resolve()))
