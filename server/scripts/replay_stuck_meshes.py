"""Recover stuck mesh tasks for a slot.

A "stuck" node is one that has a `mesh.submit` or upstream submit
(`google.banana.submit`, `trellis.submit`) recorded in events.jsonl
but no terminal `trellis.done` and no `mesh.error`. Re-invoking
`generate_mesh` lets the resumable wrapper re-poll the prior server
job by id (no rebill) and fall through to a fresh `POST /generate`
when the server job is gone.

PAUSE the slot in the originating pipeline first — both this script
and the live server would otherwise append to the same events.jsonl.

Nodes are recovered one at a time. Serial work surfaces a hang
immediately and lets the next node proceed unaffected.

If you're running multiple pipeline instances (normal + bboxes-only +
etc.), each has its own STARSHOT_RUNS_DIR. Point this script at the
right one by either exporting the same STARSHOT_RUNS_DIR before
running, or passing --runs-dir.

Usage (from server/):
    uv run python scripts/replay_stuck_meshes.py <slot_id>
    uv run python scripts/replay_stuck_meshes.py <slot_id> --runs-dir ../runs-bboxes
    uv run python scripts/replay_stuck_meshes.py --runs-dir ../runs-bboxes   # all slots
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import find_dotenv, load_dotenv

# Load .env before any module-level code reads from os.environ
# (DEFAULT_RUNS_DIR below).
load_dotenv(find_dotenv(usecwd=True))

from app.core.prompts import wrap_image_prompt
from app.core.types import BoundingBox, ProxyShape
from app.services import nano_banana, threed
from app.utils import logging as rlog, resumable
from app.utils.logging import SlotLog

DEFAULT_RUNS_DIR = Path(
    os.environ.get(
        "STARSHOT_RUNS_DIR",
        str(Path(__file__).resolve().parent.parent / "runs"),
    )
)


_SUBMIT_KINDS = ("google.banana.submit", "trellis.submit", "mesh.submit")
_TERMINAL_DONE_KIND = "trellis.done"


def find_stuck(
    events: list[dict],
) -> list[tuple[str, str, BoundingBox, ProxyShape | None]]:
    """Return [(node_id, subject_prompt, bbox, proxy_shape), ...] for nodes
    that have at least one upstream submit but no terminal completion
    (`trellis.done` for a fully-finished mesh, or `mesh.error` for a
    permanent failure)."""
    submitted: set[str] = set()
    artifact_done: set[str] = set()
    mesh_err: set[str] = set()
    bboxes: dict[str, dict] = {}
    mesh_submits: dict[str, dict] = {}
    for e in events:
        k = e.get("kind")
        if k in _SUBMIT_KINDS:
            submitted.add(e.get("job_id") or e.get("id"))
        elif k == _TERMINAL_DONE_KIND:
            artifact_done.add(e["job_id"])
        elif k == "mesh.error":
            mesh_err.add(e["id"])
        elif k == "bbox":
            bboxes[e["id"]] = e
        elif k == "mesh.submit":
            mesh_submits[e["id"]] = e

    pending: list[tuple[str, str, BoundingBox, ProxyShape | None]] = []
    for nid in submitted:
        if not nid or nid in artifact_done or nid in mesh_err:
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


async def replay_slot(
    slot_id: str, runs_dir: Path, *, per_node_timeout: float,
) -> tuple[int, int, int]:
    """Recover one slot. Returns (ok, err, total_stuck)."""
    run_dir = runs_dir / slot_id
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        print(f"  skip {slot_id}: no events at {events_path}", file=sys.stderr)
        return (0, 0, 0)

    slot_log = SlotLog(slot_id, events_path)
    slot_log.hydrate_from_disk()
    rlog.bind(slot_log)

    stuck = find_stuck(slot_log.state["events"])
    print(f"{slot_id}: {len(stuck)} stuck mesh tasks to replay")
    if not stuck:
        return (0, 0, 0)

    objs_dir = run_dir / "objects"
    objs_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    err = 0

    # Serial, one node at a time, with a per-node hard timeout.
    # Reattach should return quickly (the server's status probe is
    # cheap); a hang past `per_node_timeout` usually means the server
    # job is gone and we need a fresh submit. The fresh-submit retry
    # gets 3× the budget since it actually runs the model.
    fresh_submit_timeout = per_node_timeout * 3
    for i, (nid, subject_prompt, bbox, proxy) in enumerate(stuck, start=1):
        banana_prompt = wrap_image_prompt(subject_prompt, proxy, bbox.size)
        raw = objs_dir / f"{nid}.raw.glb"
        image_path = objs_dir / f"{nid}.png"
        print(f"  [{i}/{len(stuck)}] {nid}", flush=True)

        async def _replay(nid: str = nid, *, skip_reattach: bool = False) -> None:
            # Prefer the hosted banana URL when we have it (migrated
            # runs carry it on google.banana.done). The migrated
            # trellis.submit's input_hash was recomputed assuming
            # image=remote_url, so passing the same URL here makes
            # the wrapper's input_hash match and the prior server
            # job is reattached — no Trellis re-bill. Passing
            # image_bytes instead would hash different content and
            # miss the prior submit, forcing a fresh paid generation.
            banana_done = resumable.find_done(
                scope="google.banana", job_id=nid,
            )
            image_arg: bytes | str
            image_mime = "image/png"
            if banana_done is not None and banana_done.get("remote_url"):
                image_arg = str(banana_done["remote_url"])
            else:
                image = await nano_banana.generate_resumable(
                    banana_prompt, job_id=nid, save_to=image_path,
                )
                image_arg = image.image_bytes
                image_mime = image.mime_type
            await threed.generate_mesh(
                image_arg,
                output_path=raw,
                job_id=nid,
                image_mime=image_mime,
                skip_reattach=skip_reattach,
            )

        try:
            await asyncio.wait_for(_replay(), timeout=per_node_timeout)
            ok += 1
            print(f"    ok  {nid}", flush=True)
        except asyncio.TimeoutError:
            # Reattach hung — the server job is almost certainly
            # gone. Retry with a fresh submit, accepting the re-bill.
            print(
                f"    timeout {nid} after {per_node_timeout:.0f}s "
                f"— resubmitting fresh",
                flush=True,
            )
            try:
                await asyncio.wait_for(
                    _replay(skip_reattach=True), timeout=fresh_submit_timeout,
                )
                ok += 1
                print(f"    ok  {nid} (fresh submit)", flush=True)
            except asyncio.TimeoutError:
                err += 1
                print(
                    f"    err {nid}: fresh submit also timed out "
                    f"after {fresh_submit_timeout:.0f}s",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                err += 1
                print(
                    f"    err {nid} (fresh submit): {type(e).__name__}: {e}",
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001
            err += 1
            print(f"    err {nid}: {type(e).__name__}: {e}", flush=True)

    return (ok, err, len(stuck))


def _discover_slots(runs_dir: Path) -> list[str]:
    """Subdirectories of runs_dir that contain an events.jsonl, sorted."""
    if not runs_dir.is_dir():
        return []
    return sorted(
        p.name for p in runs_dir.iterdir()
        if p.is_dir() and (p / "events.jsonl").exists()
    )


async def main(
    slot_id: str | None, runs_dir: Path, per_node_timeout: float,
) -> None:
    print(f"runs dir: {runs_dir}")
    print(f"per-node timeout: {per_node_timeout:.0f}s")

    if slot_id is None:
        slots = _discover_slots(runs_dir)
        if not slots:
            print(f"no slots with events.jsonl under {runs_dir}", file=sys.stderr)
            sys.exit(2)
        print(f"recovering {len(slots)} slot(s): {', '.join(slots)}")
    else:
        slots = [slot_id]

    total_ok = total_err = total_stuck = 0
    try:
        for sid in slots:
            ok, err, stuck = await replay_slot(
                sid, runs_dir, per_node_timeout=per_node_timeout,
            )
            total_ok += ok
            total_err += err
            total_stuck += stuck
    finally:
        await threed.disconnect_http()
    print(
        f"done: {total_ok} recovered, {total_err} failed, "
        f"{total_stuck} total across {len(slots)} slot(s)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "slot_id",
        nargs="?",
        default=None,
        help=(
            "slot to recover. Omit to recover every slot in --runs-dir "
            "that has an events.jsonl."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=(
            "runs directory of the pipeline instance to recover. "
            "Defaults to $STARSHOT_RUNS_DIR or server/runs."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help=(
            "per-node hard timeout in seconds. On expiry the node is "
            "retried with a fresh submit before being marked failed. "
            "Default: 60."
        ),
    )
    args = parser.parse_args()
    asyncio.run(main(args.slot_id, args.runs_dir.resolve(), args.timeout))
