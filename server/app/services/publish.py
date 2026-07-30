"""Publish one cell's preview + tour to R2 and record it in the D1 catalog.

The prod client reads four top-level folders in the bucket (panoramas/, previews/,
proxies/, tours/). Assets are keyed per cell — run/slot/model, NOT versioned:

  previews/<run>/<slot>/<model>/scene-lite.glb
  tours/<run>/<slot>/<model>/tour.json
  proxies/<run>/<slot>/<model>/proxy.glb
  panoramas/<run>/<slot>/<model>/<anchor>.jpg

so re-publishing a cell overwrites its objects (and its D1 row) in place. The
dollhouse is baked from whichever raw build the cell's source preference selects
(the generated Trellis build OR the asset-library build — the same choice the
splat/tour pipeline resolves, so the preview matches the meshes the tour was
captured against), cached under the cell so a re-publish only re-bakes when a
source mesh or the bake script changes.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.services import d1, r2, scene_lite

# Mirrors the on-disk layout the pipeline writes (app.pipeline.generation): the
# generated build's raw, PNG-textured meshes live at <cell>/objects-generated/
# (older cells nested under <cell>/generated/<n>/objects-generated/), and the
# asset-library build's raw meshes at <cell>/objects/. Kept as local constants so
# this stays off the heavy generation import path.
_RAW_SUBDIR = "objects-generated"
_LIBRARY_SUBDIR = "objects"
_LEGACY_GENERATED_DIR = "generated"
_PUBLISHED_DIR = "published"  # baked preview cached under the cell
# The SOG-encoded trained splat the viewer loads, in PREFERENCE ORDER.
#
# `trained.web.sog` is the delivery encode (client/tools/splat-to-web-sog.mjs): it
# carries any baked frame correction and is sized for streaming, so it wins whenever
# it exists. A bare `trained.sog` is the trainer's own export — usable, but it is
# whatever frame that trainer chose to write, which is not necessarily the world the
# rest of the cell lives in.
_SPLAT_NAMES = ("trained.web.sog", "trained.sog")

# The cell's mesh-source choice (persisted in splat/source.json by the pipeline).
# 'auto' = generated build else library; an explicit value pins exactly one set.
_SPLAT_SOURCES = ("auto", "generated", "library")

# Public origin the bucket is served from — only used to return convenience URLs
# in the publish response; D1 stores bare keys.
_PUBLIC_BASE = "https://benchmark.tryflopilot.com"

_CONTENT_TYPES = {
    ".glb": "model/gltf-binary",
    ".json": "application/json",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def _content_type(path: Path) -> str:
    return _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _scene_glbs(objects_dir: Path) -> list[Path]:
    """Finished per-object GLBs (excludes the `.raw.glb` Trellis intermediates)."""
    if not objects_dir.is_dir():
        return []
    return [p for p in objects_dir.glob("*.glb") if not p.name.endswith(".raw.glb")]


def _source_pref(cell: Path) -> str:
    """The cell's persisted mesh-source choice ('auto' when unset/invalid) — the
    same splat/source.json the splat + tour pipeline reads, so the dollhouse bakes
    from whatever set the tour was captured against."""
    try:
        pref = json.loads(
            (cell / "splat" / "source.json").read_text(encoding="utf-8")
        ).get("source")
    except Exception:
        return "auto"
    return pref if pref in _SPLAT_SOURCES else "auto"


def _generated_objects_dir(cell: Path) -> Path | None:
    """The generated build's RAW (PNG-textured) meshes — the current single build
    (<cell>/objects-generated/), else the newest legacy
    <cell>/generated/<n>/objects-generated/. None when neither has rendered
    meshes."""
    single = cell / _RAW_SUBDIR
    if _scene_glbs(single):
        return single
    legacy_root = cell / _LEGACY_GENERATED_DIR
    if legacy_root.is_dir():
        versions = sorted(
            (p.name for p in legacy_root.iterdir() if p.is_dir() and p.name.isdigit()),
            key=int,
        )
        for v in reversed(versions):
            d = legacy_root / v / _RAW_SUBDIR
            if _scene_glbs(d):
                return d
    return None


def _library_objects_dir(cell: Path) -> Path | None:
    """The asset-library build's RAW (PNG-textured) meshes (<cell>/objects/). None
    when absent or migrated to the KTX2-only `objects-optimized/` twin (which the
    vertex-color bake can't decode)."""
    d = cell / _LIBRARY_SUBDIR
    return d if _scene_glbs(d) else None


def _bake_objects_dir(cell: Path) -> Path | None:
    """The RAW mesh set the dollhouse preview bakes from, honouring the cell's
    source preference — the same generated-or-library choice the splat/tour
    pipeline resolves: 'generated'/'library' pin one set, 'auto' prefers the
    generated build then falls back to the library. None when the chosen set has
    no PNG-textured meshes on disk. (scene_lite decodes PNG textures, so this is
    always a raw set, never a KTX2/Basis optimized twin.)"""
    pref = _source_pref(cell)
    if pref == "generated":
        return _generated_objects_dir(cell)
    if pref == "library":
        return _library_objects_dir(cell)
    return _generated_objects_dir(cell) or _library_objects_dir(cell)


def has_publishable_meshes(cell: Path) -> bool:
    """Whether the cell has a raw mesh set (generated or library, per its source
    preference) the dollhouse preview can bake from."""
    return _bake_objects_dir(cell) is not None


def _stale(dst: Path, *sources: Path) -> bool:
    """True when `dst` is missing or older than any existing source."""
    if not dst.is_file():
        return True
    floor = max((s.stat().st_mtime for s in sources if s.is_file()), default=0.0)
    return dst.stat().st_mtime < floor


async def _ensure_preview(cell: Path) -> Path:
    """Bake (or reuse the cached) vertex-colored dollhouse and return its path.
    Bakes from whichever raw set the cell's source preference selects (generated
    or library). Cached under the cell, invalidated when a source mesh or the bake
    script changes."""
    raw_dir = _bake_objects_dir(cell)
    if raw_dir is None:
        raise FileNotFoundError("cell has no meshes to publish")
    sources = _scene_glbs(raw_dir)
    out = cell / _PUBLISHED_DIR / "scene-lite.glb"
    if _stale(out, *sources, scene_lite.BAKE_SCRIPT):
        await scene_lite.build_scene_vcolor(raw_dir, out)
    return out


def scene_keys(run: str, slot: str, model: str) -> dict[str, str]:
    """The R2 key (and pano prefix) layout for one published cell — unversioned,
    so re-publishing overwrites in place."""
    prefix = f"{run}/{slot}/{model}"
    return {
        "preview_key": f"previews/{prefix}/scene-lite.glb",
        "tour_key": f"tours/{prefix}/tour.json",
        "proxy_key": f"proxies/{prefix}/proxy.glb",
        "pano_prefix": f"panoramas/{prefix}/",
    }


# --- local (no-Cloudflare) publish -------------------------------------------
# The alternate path for local testing: bake the dollhouse under the cell but skip
# the R2 upload + D1 catalog entirely. The tour artifacts (panos, proxy, tour.json,
# minimaps) already sit in the cell's tour/ dir from capture, and the dollhouse
# lands in published/ — all under RUNS_DIR, so the orchestrator's /artifacts route
# serves them directly. The catalog is derived from disk on demand (no D1). Point
# the prod client at this server (NEXT_PUBLIC_LOCAL_API) to run fully offline.


def local_scene_keys(run: str, slot: str, model: str) -> dict[str, str]:
    """Artifact-relative keys (paths under RUNS_DIR) for a locally-served cell —
    the local twin of `scene_keys`. The prod client resolves each against the
    orchestrator's /artifacts route (via its /r2 → /artifacts rewrite)."""
    prefix = f"{run}/{slot}/{model}"
    return {
        "preview_key": f"{prefix}/{_PUBLISHED_DIR}/scene-lite.glb",
        "tour_key": f"{prefix}/tour/tour.json",
        "proxy_key": f"{prefix}/tour/proxy.glb",
        "pano_prefix": f"{prefix}/tour/",
    }


def local_splat_key(cell: Path, run: str, slot: str, model: str) -> str | None:
    """The artifact key for the cell's Gaussian splat, or None when it has none.

    Lives at the cell ROOT rather than under a stage folder because it is delivered
    ALONGSIDE the pipeline rather than produced by it — an external trainer writes
    it, so no stage directory owns it. Most cells have never been trained, so None
    is the common answer and the viewer treats a splat as an optional upgrade."""
    for name in _SPLAT_NAMES:
        if (cell / name).is_file():
            return f"{run}/{slot}/{model}/{name}"
    return None


def local_scene_row(runs_dir: Path, run: str, slot: str, model: str) -> dict[str, Any] | None:
    """The catalog row for one locally-published cell, built from disk — the local
    twin of a D1 `scenes` row (same snake_case fields). None until the dollhouse is
    baked (published/scene-lite.glb), which is what marks a cell publishable."""
    cell = runs_dir / run / slot / model
    preview = cell / _PUBLISHED_DIR / "scene-lite.glb"
    if not preview.is_file():
        return None
    keys = local_scene_keys(run, slot, model)
    tour_dir = cell / "tour"
    tour_json = tour_dir / "tour.json"
    proxy_glb = tour_dir / "proxy.glb"
    panos = sorted(tour_dir.glob("*.jpg")) if tour_dir.is_dir() else []
    stamp = tour_json if tour_json.is_file() else preview
    return {
        "run": run,
        "slot": slot,
        "model": model,
        "preview_key": keys["preview_key"],
        "tour_key": keys["tour_key"] if tour_json.is_file() else None,
        "proxy_key": keys["proxy_key"] if proxy_glb.is_file() else None,
        "pano_prefix": keys["pano_prefix"] if panos else None,
        "splat_key": local_splat_key(cell, run, slot, model),
        "pano_count": len(panos),
        "published_at": datetime.fromtimestamp(stamp.stat().st_mtime, UTC).isoformat(),
    }


def list_local_scenes(runs_dir: Path) -> list[dict[str, Any]]:
    """Every locally-published cell (dollhouse baked), newest first — the on-disk
    twin of the D1 catalog. Scans RUNS_DIR/<run>/<slot>/<model>."""
    if not runs_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        for slot_dir in run_dir.iterdir():
            if not slot_dir.is_dir():
                continue
            for model_dir in slot_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                row = local_scene_row(runs_dir, run_dir.name, slot_dir.name, model_dir.name)
                if row is not None:
                    rows.append(row)
    rows.sort(key=lambda r: r["published_at"], reverse=True)
    return rows


async def publish_cell_local(
    runs_dir: Path, run: str, slot: str, model: str
) -> dict[str, Any]:
    """Local publish: bake (or reuse) the cell's dollhouse; the tour artifacts are
    already on disk from capture. No R2 upload, no D1 write — the catalog is read
    from disk (`list_local_scenes`). Raises FileNotFoundError when there's nothing
    to publish."""
    cell = runs_dir / run / slot / model
    if not cell.is_dir():
        raise FileNotFoundError(f"no such cell: {run}/{slot}/{model}")
    if not has_publishable_meshes(cell):
        raise FileNotFoundError("cell has no meshes to publish")
    await _ensure_preview(cell)
    row = local_scene_row(runs_dir, run, slot, model)
    return row or {"run": run, "slot": slot, "model": model, "pano_count": 0}


async def publish_cell(
    runs_dir: Path,
    run: str,
    slot: str,
    model: str,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Upload a cell's preview + tour to R2 under per-cell keys and upsert the D1
    catalog row. Raises FileNotFoundError when there's nothing to publish.

    With `dry_run`, resolves the planned keys without baking, uploading, or
    touching D1 — so the layout can be previewed without creds."""
    cell = runs_dir / run / slot / model
    if not cell.is_dir():
        raise FileNotFoundError(f"no such cell: {run}/{slot}/{model}")
    if not has_publishable_meshes(cell):
        raise FileNotFoundError("cell has no meshes to publish")

    tour_dir = cell / "tour"
    tour_json = tour_dir / "tour.json"
    proxy_glb = tour_dir / "proxy.glb"
    panos = sorted(tour_dir.glob("*.jpg")) if tour_dir.is_dir() else []
    # Bird's-eye minimap slices (one PNG per Y level) ride under the same pano
    # prefix, so the manifest's `minimaps[].file` resolves like a pano filename.
    minimaps = sorted(tour_dir.glob("minimap-*.png")) if tour_dir.is_dir() else []

    keys = scene_keys(run, slot, model)
    preview_key = keys["preview_key"]
    tour_key = keys["tour_key"] if tour_json.is_file() else None
    proxy_key = keys["proxy_key"] if proxy_glb.is_file() else None
    pano_prefix = keys["pano_prefix"] if panos else None

    if dry_run:
        return {
            "run": run,
            "slot": slot,
            "model": model,
            "preview_key": preview_key,
            "tour_key": tour_key,
            "proxy_key": proxy_key,
            "pano_prefix": pano_prefix,
            "pano_count": len(panos),
            "minimap_count": len(minimaps),
            "base_url": _PUBLIC_BASE,
            "dry_run": True,
        }

    preview = await _ensure_preview(cell)

    uploads = [r2.put_file(preview_key, preview, _content_type(preview))]
    if tour_key:
        uploads.append(r2.put_file(tour_key, tour_json, _content_type(tour_json)))
    if proxy_key:
        uploads.append(r2.put_file(proxy_key, proxy_glb, _content_type(proxy_glb)))
    for p in panos:
        uploads.append(r2.put_file(f"{pano_prefix}{p.name}", p, _content_type(p)))
    if pano_prefix:
        for p in minimaps:
            uploads.append(r2.put_file(f"{pano_prefix}{p.name}", p, _content_type(p)))
    await asyncio.gather(*uploads)

    published_at = datetime.now(UTC).isoformat()
    await d1.upsert_scene(
        run=run,
        slot=slot,
        model=model,
        preview_key=preview_key,
        tour_key=tour_key,
        proxy_key=proxy_key,
        pano_prefix=pano_prefix,
        pano_count=len(panos),
        published_at=published_at,
    )

    return {
        "run": run,
        "slot": slot,
        "model": model,
        "preview_key": preview_key,
        "tour_key": tour_key,
        "proxy_key": proxy_key,
        "pano_prefix": pano_prefix,
        "pano_count": len(panos),
        "minimap_count": len(minimaps),
        "published_at": published_at,
        "base_url": _PUBLIC_BASE,
    }
