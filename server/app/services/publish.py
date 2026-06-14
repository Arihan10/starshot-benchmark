"""Publish one cell's preview + tour to R2 and record it in the D1 catalog.

The prod client reads four top-level folders in the bucket (panoramas/, previews/,
proxies/, tours/). Assets are keyed per cell — run/slot/model, NOT versioned:

  previews/<run>/<slot>/<model>/scene-lite.glb
  tours/<run>/<slot>/<model>/tour.json
  proxies/<run>/<slot>/<model>/proxy.glb
  panoramas/<run>/<slot>/<model>/<anchor>.jpg

so re-publishing a cell overwrites its objects (and its D1 row) in place. The
dollhouse is still baked from a specific generated build — the one the tour was
captured against (else the latest with rendered meshes) — but that build number
is an internal bake detail, never part of the published keys or catalog.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.services import d1, r2, scene_lite

# Mirrors the generated layout the gate writes (app.pipeline.generation): each
# cell holds generated/<version>/objects-generated/ raw meshes. Kept as local
# constants so this stays off the heavy generation import path.
_GENERATED_DIR = "generated"
_RAW_SUBDIR = "objects-generated"
_PUBLISHED_DIR = "published"  # per-version baked previews cached under the cell

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


def _list_versions(cell: Path) -> list[str]:
    gen_root = cell / _GENERATED_DIR
    if not gen_root.is_dir():
        return []
    return sorted(
        (p.name for p in gen_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=int,
    )


def _resolve_version(cell: Path, version: str | None) -> str:
    """The version to publish: the requested one (validated, must have meshes),
    else the latest generated version that actually rendered meshes."""
    versions = _list_versions(cell)
    if version is not None:
        if not version.isdigit():
            raise ValueError(f"invalid version: {version!r}")
        version = str(int(version))  # normalize leading zeros
        if version not in versions:
            raise FileNotFoundError(f"no generated version {version!r} for this cell")
        if not _scene_glbs(cell / _GENERATED_DIR / version / _RAW_SUBDIR):
            raise FileNotFoundError(f"generated version {version} has no rendered meshes")
        return version
    for v in reversed(versions):
        if _scene_glbs(cell / _GENERATED_DIR / v / _RAW_SUBDIR):
            return v
    raise FileNotFoundError("cell has no generated meshes to publish")


def latest_rendered_version(cell: Path) -> str | None:
    """The newest generated version with rendered meshes, or None — the version a
    fresh capture corresponds to. Used to stamp the tour at capture time."""
    try:
        return _resolve_version(cell, None)
    except FileNotFoundError:
        return None


def recorded_tour_version(cell: Path) -> str | None:
    """The generation a cell's tour was captured against, stamped into the tour
    manifest by the /tour/manifest route. None for tours written before this was
    recorded (callers then fall back to the latest rendered version)."""
    tour_json = cell / "tour" / "tour.json"
    if not tour_json.is_file():
        return None
    try:
        data = json.loads(tour_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    v = data.get("generated_version") if isinstance(data, dict) else None
    return str(v) if isinstance(v, (str, int)) and str(v).isdigit() else None


def _stale(dst: Path, *sources: Path) -> bool:
    """True when `dst` is missing or older than any existing source."""
    if not dst.is_file():
        return True
    floor = max((s.stat().st_mtime for s in sources if s.is_file()), default=0.0)
    return dst.stat().st_mtime < floor


async def _ensure_preview(cell: Path, version: str) -> Path:
    """Bake (or reuse the cached) vertex-colored dollhouse for `version` and
    return its path. Cached per version under the cell, invalidated when a source
    mesh or the bake script changes."""
    raw_dir = cell / _GENERATED_DIR / version / _RAW_SUBDIR
    sources = _scene_glbs(raw_dir)
    if not sources:
        raise FileNotFoundError(f"generated version {version} has no rendered meshes")
    out = cell / _PUBLISHED_DIR / version / "scene-lite.glb"
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


async def publish_cell(
    runs_dir: Path,
    run: str,
    slot: str,
    model: str,
    version: str | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Upload a cell's preview + tour to R2 under per-version keys and upsert the
    D1 catalog row. Raises FileNotFoundError when there's nothing to publish.

    With `dry_run`, resolves the version and the planned keys without baking,
    uploading, or touching D1 — so the layout can be previewed without creds."""
    cell = runs_dir / run / slot / model
    if not cell.is_dir():
        raise FileNotFoundError(f"no such cell: {run}/{slot}/{model}")
    # Pair the dollhouse (baked per generated version) with the SAME generation
    # the tour (proxy + panos, captured once into the unversioned tour/ dir) was
    # shot against — otherwise a later regenerate publishes a fresh dollhouse over
    # a stale walkthrough, so the overview and the interior show different scenes.
    # An explicit request wins; a tour with no recorded version falls back to the
    # latest (legacy tours, and the dollhouse-only case).
    if version is None:
        recorded = recorded_tour_version(cell)
        if recorded is not None:
            try:
                version = _resolve_version(cell, recorded)
            except (FileNotFoundError, ValueError):
                version = _resolve_version(cell, None)
        else:
            version = _resolve_version(cell, None)
    else:
        version = _resolve_version(cell, version)

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

    preview = await _ensure_preview(cell, version)

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
