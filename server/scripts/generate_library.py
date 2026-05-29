"""Generate library assets from a CSV catalog.

Reads a CSV and runs each object through the EXACT same generation
pipeline as the main scene builder:
  1. _build_image_prompt  — LLM noun phrase + wrap_image_prompt (with
     orthographic/perspective instructions, dimensions, proxy shape)
  2. nano_banana.generate_resumable — Gemini image generation
  3. threed.generate_mesh — Trellis 2 mesh generation (raw, no rescale)

Outputs land in {output_dir}/assets/{slug}.png and .glb.
Resumable: re-running skips items whose image and mesh are already done
(via the SlotLog event cache).

Usage:
  cd server && uv run python scripts/generate_library.py
  cd server && uv run python scripts/generate_library.py \\
    --csv scripts/platformer_asset_library_v2.csv \\
    --output-dir app/assets_library_platformer \\
    --style "for a colorful Super Mario Bros-style 3D platformer game — chunky, rounded, toylike proportions with bright saturated colors and clean cartoon surfaces"
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SERVER_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()
load_dotenv(dotenv_path=_SERVER_DIR.parent / ".env")

from app.core.prompts import ImageView  # noqa: E402
from app.core.types import BoundingBox  # noqa: E402
from app.pipeline.generation import _build_image_prompt  # noqa: E402

_THREE_QUARTER_CATEGORIES = {
    "STRUCTURAL — Building Shell",
    "STRUCTURAL — Openings & Transitions",
    "OUTDOOR — Terrain & Land Forms",
    "OUTDOOR — Hardscape & Pathways",
    "OUTDOOR — Fencing & Boundaries",
    "TERRAIN — Ground Blocks",
    "TERRAIN — Platforms",
    "TERRAIN — Walls & Ceilings",
    "TERRAIN — Cliffs & Slopes",
    "SPECIAL BLOCKS",
    "PIPES & PORTALS",
}
from app.services import llm, nano_banana, threed  # noqa: E402
from app.utils import logging as rlog  # noqa: E402
from app.utils.logging import SlotLog  # noqa: E402

_DEFAULT_CSV = _SERVER_DIR / "dry_runs" / "asset_library_v3.csv"
_DEFAULT_OUTPUT_DIR = _SERVER_DIR / "app" / "assets_library"
MODEL = "google/gemini-3.1-flash-lite"
MAX_CONCURRENT = 30
MAX_RETRIES = 3
RETRY_BACKOFF_S = 5.0

_DEFAULT_BBOX = BoundingBox(origin=(0.0, 0.0, 0.0), dimensions=(1.0, 1.0, 1.0))
_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

_done_count = 0
_skip_count = 0
_fail_count = 0
_total = 0


def _slugify(name: str) -> str:
    slug = "-".join(
        part for part in
        ("".join(c if c.isalnum() else " " for c in name.lower()).split())
        if part
    )
    return slug[:80]


async def _generate_one(
    item_id: str,
    csv_id: str,
    name: str,
    category: str,
    *,
    assets_dir: Path,
    style: str | None = None,
) -> dict[str, object]:
    """Run a single CSV item through the exact same pipeline as
    generation._build_image_prompt -> generation._generate_one."""
    global _done_count, _skip_count, _fail_count

    glb_path = assets_dir / f"{item_id}.glb"
    image_path = assets_dir / f"{item_id}.png"

    if glb_path.exists() and image_path.exists():
        _skip_count += 1
        return {
            "csv_id": csv_id, "id": item_id, "name": name,
            "category": category, "status": "skipped",
        }

    async with _semaphore:
        view: ImageView = (
            "three-quarter" if category in _THREE_QUARTER_CATEGORIES
            else "front"
        )
        prompt = f"{name} — {style}" if style else name
        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                subject, image_prompt = await _build_image_prompt(
                    spec_id=item_id,
                    prompt=prompt,
                    bbox=_DEFAULT_BBOX,
                    proxy_shape=None,
                    prior_prompts=[],
                    view=view,
                    include_dimensions=False,
                )

                image = await nano_banana.generate_resumable(
                    image_prompt, job_id=item_id, save_to=image_path,
                )
                rlog.log("image", id=item_id, prompt=subject)

                await threed.generate_mesh(
                    image.image_bytes,
                    output_path=glb_path,
                    job_id=item_id,
                    image_mime=image.mime_type,
                )

                rlog.emit_model(item_id, artifact_kind="object", url=str(glb_path))

                _done_count += 1
                _progress = _done_count + _fail_count + _skip_count
                print(
                    f"[generate-library] ({_progress}/{_total}) "
                    f"DONE {item_id}",
                    flush=True,
                )
                return {
                    "csv_id": csv_id,
                    "id": item_id,
                    "name": name,
                    "category": category,
                    "subject": subject,
                    "image_prompt": image_prompt,
                    "status": "done",
                }
            except Exception as e:
                last_err = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BACKOFF_S * (2 ** attempt)
                    print(
                        f"[generate-library] RETRY {item_id} "
                        f"(attempt {attempt + 1}/{MAX_RETRIES}): "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                    for artifact in (image_path, glb_path):
                        artifact.unlink(missing_ok=True)
                    await asyncio.sleep(delay)

        _fail_count += 1
        _progress = _done_count + _fail_count + _skip_count
        print(
            f"[generate-library] ({_progress}/{_total}) "
            f"FAIL {item_id}: {type(last_err).__name__}: {last_err}",
            flush=True,
        )
        return {
            "csv_id": csv_id,
            "id": item_id,
            "name": name,
            "category": category,
            "status": "error",
            "error": f"{type(last_err).__name__}: {last_err}",
        }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate library assets from a CSV catalog.")
    p.add_argument("--csv", type=Path, default=_DEFAULT_CSV, help="Path to the CSV catalog")
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR, help="Output directory (assets/ subdir created inside)")
    p.add_argument("--style", type=str, default=None, help="Style context appended to every object prompt")
    p.add_argument("--trellis-concurrency", type=int, default=None, help="Override Trellis in-flight job cap (default: 10)")
    p.add_argument("--concurrent", type=int, default=MAX_CONCURRENT, help="Max concurrent items in pipeline (default: 30)")
    return p.parse_args()


async def main() -> None:
    global _total, _semaphore
    args = _parse_args()
    csv_path = args.csv if args.csv.is_absolute() else _SERVER_DIR / args.csv
    output_dir = args.output_dir if args.output_dir.is_absolute() else _SERVER_DIR / args.output_dir
    assets_dir = output_dir / "assets"
    events_path = output_dir / "generate_events.jsonl"
    style: str | None = args.style

    if args.trellis_concurrency is not None:
        threed.GENERATE_CONCURRENCY = args.trellis_concurrency
        threed._inflight_sem = asyncio.Semaphore(args.trellis_concurrency)
    if args.concurrent != MAX_CONCURRENT:
        _semaphore = asyncio.Semaphore(args.concurrent)

    assets_dir.mkdir(parents=True, exist_ok=True)

    items: list[tuple[str, str, str, str]] = []
    seen_slugs: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_id = row["ID"]
            name = row["Object Name"]
            category = row.get("Category", "")
            slug = _slugify(name)
            if slug in seen_slugs:
                slug = f"{slug}-{csv_id}"
            seen_slugs.add(slug)
            items.append((slug, csv_id, name, category))

    _total = len(items)

    slot_log = SlotLog("library-gen", events_path)
    slot_log.hydrate_from_disk()
    if not slot_log.state["events"]:
        slot_log.start_run("library generation", MODEL)
    else:
        slot_log.state["status"] = "running"
    rlog.bind(slot_log)
    llm.set_model(MODEL)

    already = sum(
        1 for slug, _, _, _ in items
        if (assets_dir / f"{slug}.glb").exists()
        and (assets_dir / f"{slug}.png").exists()
    )
    remaining = _total - already
    print(
        f"[generate-library] {_total} items total, "
        f"{already} already done, {remaining} to generate "
        f"(max {MAX_CONCURRENT} concurrent)",
        flush=True,
    )
    t0 = time.monotonic()

    tasks = [
        asyncio.create_task(
            _generate_one(slug, csv_id, name, cat, assets_dir=assets_dir, style=style),
        )
        for slug, csv_id, name, cat in items
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.monotonic() - t0
    manifest: list[dict[str, object]] = []
    for r in results:
        if isinstance(r, Exception):
            print(f"[generate-library] uncaught: {r}", flush=True)
        else:
            manifest.append(r)

    manifest_path = output_dir / "generate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    slot_log.finish_run()
    print(
        f"\n[generate-library] finished in {elapsed:.0f}s  "
        f"done={_done_count} skipped={_skip_count} failed={_fail_count}  "
        f"manifest -> {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
