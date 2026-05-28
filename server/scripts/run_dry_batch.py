"""Batch dry-run: run the full divider pipeline (no mesh generation) for
multiple prompts in parallel and collect all emitted objects into a single
JSON file for analysis.

Usage:  cd server && uv run python scripts/run_dry_batch.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

_SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SERVER_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()
load_dotenv(dotenv_path=_SERVER_DIR.parent / ".env")

# Monkey-patch _spawn_meshes BEFORE anything touches the generation module
# through divider. This replicates what main_nomesh.py does.
from app.core.types import Node  # noqa: E402
from app.pipeline import generation  # noqa: E402
from app.utils import logging as rlog  # noqa: E402


async def _bbox_only_spawn(
    *,
    resolved: list[Node],
    runs_dir: Path,  # noqa: ARG001
    run_id: str,  # noqa: ARG001
    scenario: Literal["anchor", "encapsulating", "negative-space"],  # noqa: ARG001
) -> list[Node]:
    out: list[Node] = []
    for node in resolved:
        rlog.log("nomesh.skip", id=node.id, prompt=node.prompt)
        out.append(node.model_copy(update={"mesh_url": f"bbox://{node.id}"}))
    return out


generation._spawn_meshes = _bbox_only_spawn  # type: ignore[attr-defined]

from app.pipeline import divider  # noqa: E402
from app.utils.logging import SlotLog  # noqa: E402

PROMPTS = [
    "A campsite in the middle of a forest",
    "A swamp with islands, designed as a top-down arcade level where a frog can jump from island to island"
    
]
RUNS_PER_PROMPT = 5
MODEL = "google/gemini-3.5-flash"
MAX_CONCURRENT = 5

_semaphore = asyncio.Semaphore(MAX_CONCURRENT)


async def _run_one(prompt: str, run_index: int, runs_dir: Path) -> dict[str, object]:
    slug = "".join(c if c.isalnum() or c == "-" else "-" for c in prompt.lower())[:40]
    run_id = f"{slug}-{run_index}"

    async with _semaphore:
        slot_dir = runs_dir / run_id
        slot_dir.mkdir(parents=True, exist_ok=True)
        slot_log = SlotLog(run_id, slot_dir / "events.jsonl")
        slot_log.start_run(prompt, MODEL)
        rlog.bind(slot_log)

        print(f"[dry-batch] START {run_id}", flush=True)
        status = "done"
        try:
            await divider.run(
                run_id=run_id, prompt=prompt, model=MODEL, runs_dir=runs_dir,
            )
            await generation.await_pending(run_id)
            slot_log.finish_run()
        except Exception as e:
            generation.cancel_pending(run_id)
            slot_log.log("run.error", message=f"{type(e).__name__}: {e}")
            status = "error"
            print(f"[dry-batch] FAIL {run_id}: {e}", flush=True)

        if status == "done":
            print(f"[dry-batch] DONE {run_id}", flush=True)

    events = slot_log.state["events"]
    objects: list[dict[str, object]] = []
    zones: list[dict[str, object]] = []
    for ev in events:
        kind = ev.get("kind")
        if kind == "bbox":
            objects.append({
                "id": ev.get("id"),
                "prompt": ev.get("prompt"),
                "parent_id": ev.get("parent_id"),
                "node_kind": ev.get("node_kind"),
                "proxy_shape": ev.get("proxy_shape"),
                "orientation": ev.get("orientation"),
                "origin": ev.get("origin"),
                "dimensions": ev.get("dimensions"),
            })
        elif kind == "divider.zone_plan":
            zones.append({
                "id": ev.get("node"),
                "plan": ev.get("plan"),
                "is_atomic": ev.get("is_atomic"),
            })

    return {
        "prompt": prompt,
        "run_index": run_index,
        "run_id": run_id,
        "model": MODEL,
        "status": status,
        "object_count": len([o for o in objects if o.get("node_kind") == "object"]),
        "objects": objects,
        "zones": zones,
    }


async def main() -> None:
    runs_dir = (_SERVER_DIR / "dry_runs").resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        asyncio.create_task(_run_one(prompt, i, runs_dir))
        for prompt in PROMPTS
        for i in range(RUNS_PER_PROMPT)
    ]
    total = len(tasks)
    print(
        f"[dry-batch] launching {total} runs "
        f"({RUNS_PER_PROMPT}x each of {len(PROMPTS)} prompts, "
        f"{MAX_CONCURRENT} concurrent)",
        flush=True,
    )
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful: list[dict[str, object]] = []
    errors = 0
    for r in results:
        if isinstance(r, Exception):
            print(f"[dry-batch] uncaught: {r}", flush=True)
            errors += 1
        else:
            successful.append(r)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "prompts": PROMPTS,
        "runs_per_prompt": RUNS_PER_PROMPT,
        "total_runs": len(successful),
        "errors": errors,
        "runs": successful,
    }
    output_path = runs_dir / "dry_runs_results.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\n[dry-batch] saved {len(successful)}/{total} runs to {output_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
