"""Merge all asset libraries into a single unified library.

Reads generate_manifest.json from each assets_library_* directory,
copies all .glb and .png assets into assets_library/assets/, and
writes a unified library.json with categories from the manifests.

Usage:  cd server && uv run python scripts/merge_libraries.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
TARGET_DIR = APP_DIR / "assets_library"
TARGET_ASSETS = TARGET_DIR / "assets"

SOURCE_DIRS = [
    ("assets_library", None),
    ("assets_library_platformer", "PLATFORMER"),
    ("assets_library_swamp", "SWAMP ARCADE"),
    ("assets_library_campsite", "CAMPSITE"),
]


def main() -> None:
    TARGET_ASSETS.mkdir(parents=True, exist_ok=True)

    catalog: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    copied = 0
    skipped_dup = 0

    for dir_name, fallback_category in SOURCE_DIRS:
        source = APP_DIR / dir_name
        manifest_path = source / "generate_manifest.json"
        assets_dir = source / "assets"

        if not manifest_path.exists():
            print(f"[merge] SKIP {dir_name}: no manifest")
            continue

        manifest = json.load(manifest_path.open())
        items = [x for x in manifest if x.get("status") in ("done", "skipped")]
        print(f"[merge] {dir_name}: {len(items)} items")

        for item in items:
            item_id = item["id"]
            if item_id in seen_ids:
                skipped_dup += 1
                continue
            seen_ids.add(item_id)

            category = item.get("category", "")
            if not category and fallback_category:
                category = fallback_category

            glb_src = assets_dir / f"{item_id}.glb"
            png_src = assets_dir / f"{item_id}.png"
            glb_dst = TARGET_ASSETS / f"{item_id}.glb"
            png_dst = TARGET_ASSETS / f"{item_id}.png"

            if not glb_src.exists():
                continue

            # Only copy if source != target (skip assets_library's own assets)
            if glb_src.resolve() != glb_dst.resolve():
                shutil.copy2(glb_src, glb_dst)
                if png_src.exists():
                    shutil.copy2(png_src, png_dst)
                copied += 1

            catalog.append({
                "id": item_id,
                "description": item.get("name", item_id),
                "category": category,
            })

    # Sort by category then description for readability
    catalog.sort(key=lambda x: (x["category"], x["description"]))

    library_path = TARGET_DIR / "library.json"
    library_path.write_text(json.dumps(catalog, indent=2))

    print(
        f"\n[merge] done: {len(catalog)} items in library.json, "
        f"{copied} files copied, {skipped_dup} duplicates skipped"
    )


if __name__ == "__main__":
    main()
