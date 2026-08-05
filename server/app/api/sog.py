"""HTTP delivery for the SOG splats the pipeline produces — everything under `/sog`.

A trained splat leaves the pipeline as a PLY and is compressed by PlayCanvas'
own encoder (`client/tools/ply-to-sog.mjs` / `ply-to-lod-sog.mjs`) into one of
two SOG shapes, both of which the PlayCanvas gsplat loader reads natively:

  * a SINGLE `.sog` — a zip holding `meta.json` + the quantized WebP textures
    (means_l/means_u/quats/scales/sh0). One request, whole model, no LOD.
  * a STREAMED-SOG bundle — a `lod-meta.json` octree manifest beside per-LOD
    chunk directories (`0_0/meta.json` + its WebPs, …). The engine loads the
    coarse chunks first and refines per octree node by camera distance under a
    splat budget, so this is the shape that actually streams.

This module is the delivery half of that. A streamed bundle is HUNDREDS of small
files (a 10.5M-splat scene is 37 chunks of 6 files each), fetched on their own by
the engine as the camera moves, so the transport levers matter as much as the
renderer's: byte range support, strong validators so a revisit costs a 304
instead of a re-download, immutable caching on the content that never changes,
and `Timing-Allow-Origin` so the browser will even tell the playground how many
bytes crossed the wire (cross-origin `transferSize` reads 0 without it).

Three routes:

  GET /sog/catalog                    — every discoverable bundle + its stats
  GET /sog/f/{root}/{path}            — bytes, default delivery policy
  GET /sog/n/{shape}/{root}/{path}    — bytes, with a network profile applied

The `{shape}` segment is deliberately a PATH segment, not a query string: the
engine derives every chunk URL by joining a relative name onto the manifest's
DIRECTORY (`path.getDirectory`, which drops any query), so only a prefix
survives the walk from `lod-meta.json` down to `3_0/scales.webp`. Pointing the
viewer at a shaped manifest URL therefore shapes the whole bundle — which is the
only way to feel the LOD levers on a localhost SSD, where every chunk would
otherwise arrive in under a millisecond.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
import zipfile
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from email.utils import formatdate, parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

# This file is server/app/api/sog.py, so the repo root (holding `starshot_paths`
# and the top-level `splat` package) is four levels up. Mirrors routes.py so the
# module imports cleanly no matter which one loads first.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from starshot_paths import runs_root  # noqa: E402

router = APIRouter(prefix="/sog")

# The runtime REQUIRES this exact basename for a streamed bundle — the gsplat
# handler dispatches to the octree parser on it (see the engine's
# framework/handlers/gsplat.js), and on `meta.json` / `.sog` for the others.
MANIFEST_NAME = "lod-meta.json"

# Extra discovery roots: `STARSHOT_SOG_ROOTS=id=/path{sep}/other/path`.
_ROOTS_ENV = "STARSHOT_SOG_ROOTS"

# Only these ever leave the server — the three file kinds a SOG bundle is made
# of. Anything else under a root (PLYs, checkpoints, logs) stays private.
_MEDIA_TYPES = {
    ".sog": "application/octet-stream",
    ".json": "application/json",
    ".webp": "image/webp",
}

_IMMUTABLE = "public, max-age=31536000, immutable"
_REVALIDATE = "public, max-age=0, must-revalidate"
_NOSTORE = "no-store"

# Chunk textures and single `.sog` zips are rewritten only by a fresh encode
# (which also moves the URL, since the bundle dir is wiped first), so they cache
# forever. The top-level manifest is the one file a rebuild replaces in place, so
# it revalidates — one conditional request that 304s, then the chunks come from
# cache.
_CACHE_POLICIES = {"immutable": _IMMUTABLE, "revalidate": _REVALIDATE, "nostore": _NOSTORE}

# Directories a bounded walk never descends into — bulky, and none of them ever
# holds a SOG bundle.
_SKIP_DIRS = {
    "node_modules", "__pycache__", ".next", ".git", "ckpt", "objects",
    "objects-optimized", "refs", "tour", "prompts", "_branches", "patches",
    "patch_views", "samples",
}
_WALK_DIR_BUDGET = 4000

# Long enough that an incidental poll never re-walks the tree; the client's rescan
# passes `?refresh=1` when it needs the authoritative answer (right after a build).
_CATALOG_TTL_S = 20.0
_ROOTS_TTL_S = 2.0
_READ_CHUNK = 256 * 1024


@dataclass(frozen=True)
class _Root:
    """One discovery root.

    `kind` picks the search strategy: a `runs` tree has a fixed cell shape
    (`<run>/<slot>/<model>/splat/`), while a `tree` root is free-form and needs a
    bounded walk. `cells` marks the ONE tree the `/runs/...` endpoints address —
    they resolve every path against `STARSHOT_RUNS_DIR` by construction, so a splat
    discovered anywhere else can be streamed but not rebuilt."""

    id: str
    path: Path
    kind: str
    cells: bool = False


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text).strip("-") or "root"


def _discover_roots() -> list[_Root]:
    found: dict[Path, _Root] = {}

    def add(rid: str, path: Path, kind: str, cells: bool = False) -> None:
        resolved = path.expanduser()
        try:
            resolved = resolved.resolve()
        except OSError:
            return
        if resolved in found or not resolved.is_dir():
            return
        found[resolved] = _Root(rid, resolved, kind, cells)

    add("runs", runs_root(), "runs", cells=True)
    add("repo", _REPO_ROOT / "runs", "runs")
    add("assets", _REPO_ROOT / "client" / "public" / "assets", "tree")
    for index, spec in enumerate(p for p in os.environ.get(_ROOTS_ENV, "").split(os.pathsep) if p):
        rid, sep, raw = spec.partition("=")
        add(_slug(rid) if sep else f"extra{index}", Path(raw if sep else rid), "tree")
    return list(found.values())


_roots_cache: tuple[float, list[_Root]] | None = None


def roots() -> list[_Root]:
    """The live root list, briefly cached. Every delivered file resolves its root
    first, so a streamed bundle asks hundreds of times a second; re-probing the
    filesystem on each one buys nothing, while expiring quickly still lets a root
    created after boot show up on its own."""
    global _roots_cache
    now = time.monotonic()
    if _roots_cache is not None and now - _roots_cache[0] < _ROOTS_TTL_S:
        return _roots_cache[1]
    discovered = _discover_roots()
    _roots_cache = (now, discovered)
    return discovered


def cell_root() -> _Root | None:
    """The tree the `/runs` cell endpoints address, if it exists on disk."""
    return next((r for r in roots() if r.cells), None)


def _root_by_id(root_id: str) -> _Root:
    for root in roots():
        if root.id == root_id:
            return root
    raise HTTPException(status_code=404, detail=f"unknown sog root: {root_id}")


# --- discovery ------------------------------------------------------------------


def _subdirs(directory: Path) -> list[Path]:
    """Child directories worth descending into, from one `scandir`."""
    try:
        with os.scandir(directory) as entries:
            return [
                Path(e.path)
                for e in entries
                if e.is_dir() and not e.name.startswith(".") and e.name not in _SKIP_DIRS
            ]
    except OSError:
        return []


def _cell_splat_dirs(root: _Root) -> Iterator[Path]:
    """`<run>/<slot>/<model>/splat` for every cell in a runs tree.

    Walked level by level rather than globbed: a runs tree can hold hundreds of
    cells, and each `*/*/*/splat/<pattern>` glob re-walks all three levels, so
    matching artifacts pattern-by-pattern cost seconds of `scandir` on a large
    tree. One pass down to each `splat/` and one listing of it answers every
    pattern at once."""
    for run in _subdirs(root.path):
        for slot in _subdirs(run):
            for model in _subdirs(slot):
                splat = model / "splat"
                if splat.is_dir():
                    yield splat


def _iter_tree_candidates(root: _Root) -> Iterator[Path]:
    """SOG entry points under a free-form root: manifests of streamed bundles and
    standalone `.sog` files. A bundle directory is never descended into — its chunk
    dirs each carry their own `meta.json` and are not separate assets."""
    stack = [root.path]
    budget = _WALK_DIR_BUDGET
    while stack and budget > 0:
        current = stack.pop()
        budget -= 1
        if (current / MANIFEST_NAME).is_file():
            yield current / MANIFEST_NAME
            continue
        try:
            entries = sorted(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if not entry.name.startswith(".") and entry.name not in _SKIP_DIRS:
                    stack.append(entry)
            elif entry.suffix.lower() == ".sog":
                yield entry


def _cell_of(root: _Root, rel: str) -> dict[str, str] | None:
    """The (run, slot, model, which) cell a runs-tree path belongs to. `which` is
    the model the artifact came from — `trained` (raw stage 6) or `healed`
    (delivered stage 7) — read off the stem of `splat/<which>.{ply,sog,lodsog}`."""
    parts = rel.split("/")
    if root.kind != "runs" or len(parts) < 5 or parts[3] != "splat":
        return None
    stem = (
        "splat" if parts[4] == MANIFEST_NAME
        else parts[4].removesuffix(".sog").removesuffix(".lodsog").removesuffix(".ply")
    )
    return {"run": parts[0], "slot": parts[1], "model": parts[2], "which": stem}


def _display_name(root: _Root, rel: str, cell: dict[str, str] | None) -> str:
    if cell:
        return f"{cell['run']} · {cell['slot']} · {cell['model']} · {cell['which']}"
    return f"{root.id}: {rel.removesuffix('/' + MANIFEST_NAME).removesuffix('.sog')}"


_dir_byte_cache: dict[tuple[str, int], int] = {}


def _dir_bytes(directory: Path, version: int) -> int:
    """Total bytes under `directory`, cached against `version` (the manifest's mtime
    — a rebuild always rewrites it). A streamed bundle is hundreds of files, and the
    catalog is polled, so this is worth not repeating."""
    key = (str(directory), version)
    cached = _dir_byte_cache.get(key)
    if cached is not None:
        return cached
    total = 0
    for entry in directory.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    _dir_byte_cache[key] = total
    return total


_sog_counts: dict[tuple[str, int, int], int | None] = {}


def _sog_splat_count(path: Path, stat: os.stat_result) -> int | None:
    """Gaussian count of a single `.sog`, read from the zip's `meta.json` through
    the central directory (no full decompress). Cached on identity so repeated
    catalog polls never reopen the archive."""
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key in _sog_counts:
        return _sog_counts[key]
    count: int | None = None
    try:
        with zipfile.ZipFile(path) as archive:
            count = json.loads(archive.read("meta.json")).get("count")
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        count = None
    _sog_counts[key] = count
    return count


def file_url(root_id: str, rel: str, shape: str | None = None) -> str:
    """The URL that serves `rel` under `root_id`, optionally through a network
    profile. Handed out by the catalog so nobody has to hand-assemble one."""
    quoted = quote(rel, safe="/")
    if shape:
        return f"{router.prefix}/n/{quote(shape, safe='')}/{quote(root_id, safe='')}/{quoted}"
    return f"{router.prefix}/f/{quote(root_id, safe='')}/{quoted}"


def url_for_path(path: Path) -> str | None:
    """The `/sog` URL for a file on disk, or None when it sits under no root. Lets
    the splat endpoints in routes.py hand back delivery URLs without duplicating
    the root table."""
    try:
        resolved = path.resolve()
    except OSError:
        return None
    if not resolved.is_file():
        return None
    for root in roots():
        if resolved.is_relative_to(root.path):
            return file_url(root.id, resolved.relative_to(root.path).as_posix())
    return None


def _describe(root: _Root, path: Path) -> dict[str, Any] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    rel = path.relative_to(root.path).as_posix()
    cell = _cell_of(root, rel)
    common = {
        "id": f"{root.id}/{rel}",
        "root": root.id,
        "path": rel,
        "url": file_url(root.id, rel),
        "name": _display_name(root, rel, cell),
        "cell": cell,
        "modified_at": stat.st_mtime,
    }
    if path.name != MANIFEST_NAME:
        return {
            **common,
            "kind": "sog",
            "bytes": stat.st_size,
            "splats": _sog_splat_count(path, stat),
            "lod_levels": 1,
            "counts": None,
            "chunk_files": 1,
        }
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    counts = meta.get("counts") if isinstance(meta.get("counts"), list) else None
    filenames = meta.get("filenames") if isinstance(meta.get("filenames"), list) else None
    return {
        **common,
        "kind": "lod",
        "bytes": _dir_bytes(path.parent, stat.st_mtime_ns),
        # `counts[0]` is LOD 0 (full detail); `count` is every level summed, which
        # is what the bundle stores but not what a full-detail view renders.
        "splats": counts[0] if counts else meta.get("count"),
        "stored_splats": meta.get("count"),
        "lod_levels": meta.get("lodLevels"),
        "counts": counts,
        "chunk_files": len(filenames) if filenames is not None else None,
    }


# --- compilable sources ---------------------------------------------------------
# A bundle has to be built before it can be streamed, and the thing it is built
# from is a cell's trained/healed PLY. Listing those alongside the finished assets
# is what lets the playground compile one without leaving the page.

_COMPILE_STEMS = ("trained", "healed")


def _describe_source(
    root: _Root, path: Path, which: str, ladder_levels: int, bundle: Path | None
) -> dict[str, Any] | None:
    """A PLY a streamed bundle can be compiled from: a cell's raw stage-6
    `trained.ply` or its delivered stage-7 `healed.ply`. Everything about what
    already exists beside it comes from the caller's single directory listing."""
    try:
        stat = path.stat()
    except OSError:
        return None
    rel = path.relative_to(root.path).as_posix()
    cell = _cell_of(root, rel)
    if cell is None:
        return None
    single = path.with_suffix(".sog")
    return {
        "id": f"{root.id}/{rel}",
        "root": root.id,
        "path": rel,
        "name": _display_name(root, rel, cell),
        "cell": {k: cell[k] for k in ("run", "slot", "model")},
        "which": which,
        "bytes": stat.st_size,
        "modified_at": stat.st_mtime,
        # A trainer-built octave ladder beside the PLY (stage 7's `_export_lod`)
        # beats decimating one at compile time, and the bundler prefers it — so
        # this is the single best predictor of the LOD quality you'll get.
        "ladder_levels": ladder_levels,
        # What already exists for this source, so the client can offer
        # build-vs-rebuild and jump straight to the result.
        "bundle_url": (
            file_url(root.id, bundle.relative_to(root.path).as_posix()) if bundle else None
        ),
        "single_url": (
            file_url(root.id, single.relative_to(root.path).as_posix())
            if single.is_file() else None
        ),
        # Only the cell root can be BUILT from: the /runs endpoints resolve a
        # (run, slot, model) against STARSHOT_RUNS_DIR, so a PLY found in any other
        # root has no address there. It still streams — it just can't be recompiled
        # without being moved into that tree.
        "build_path": (
            f"/runs/{quote(cell['run'], safe='')}/splat/lodsog"
            f"/{quote(cell['slot'], safe='')}/{quote(cell['model'], safe='')}"
        ) if root.cells else None,
        "build_note": None if root.cells else (
            f"outside the server's runs root ({runs_root()}), so the cell endpoints "
            "can't address it — set STARSHOT_RUNS_DIR to this tree to compile from it"
        ),
    }


def _scan_cell(
    root: _Root, splat: Path, assets: list[dict[str, Any]], sources: list[dict[str, Any]]
) -> None:
    """Everything one cell's `splat/` dir offers, from a single listing: finished
    SOGs (a bundled `.sog`, a bare manifest, or a `<which>.lodsog/` bundle) and the
    PLYs a bundle can still be compiled from."""
    try:
        with os.scandir(splat) as it:
            entries = list(it)
    except OSError:
        return
    files = {e.name for e in entries if e.is_file()}
    dirs = {e.name for e in entries if e.is_dir()}

    def add_asset(path: Path) -> None:
        record = _describe(root, path)
        if record is not None:
            assets.append(record)

    for name in sorted(files):
        if name.lower().endswith(".sog") or name == MANIFEST_NAME:
            add_asset(splat / name)
    bundles: dict[str, Path] = {}
    for name in sorted(dirs):
        if not name.endswith(".lodsog"):
            continue
        manifest = splat / name / MANIFEST_NAME
        if manifest.is_file():
            bundles[name] = manifest
            add_asset(manifest)

    for which in _COMPILE_STEMS:
        if f"{which}.ply" not in files:
            continue
        source = _describe_source(
            root,
            splat / f"{which}.ply",
            which,
            sum(1 for n in files if n.startswith(f"{which}.lod") and n.endswith(".ply")),
            bundles.get(f"{which}.lodsog"),
        )
        if source is not None:
            sources.append(source)


_Catalog = dict[str, list[dict[str, Any]]]
_catalog_cache: tuple[float, _Catalog] | None = None


def _catalog(refresh: bool = False) -> _Catalog:
    global _catalog_cache
    now = time.monotonic()
    if not refresh and _catalog_cache is not None and now - _catalog_cache[0] < _CATALOG_TTL_S:
        return _catalog_cache[1]
    assets: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for root in roots():
        if root.kind == "runs":
            for splat in _cell_splat_dirs(root):
                _scan_cell(root, splat, assets, sources)
            continue
        for candidate in _iter_tree_candidates(root):
            record = _describe(root, candidate)
            if record is not None:
                assets.append(record)
    assets.sort(key=lambda a: (-a["modified_at"], a["id"]))
    sources.sort(key=lambda s: (-s["modified_at"], s["id"]))
    result: _Catalog = {"assets": assets, "sources": sources}
    _catalog_cache = (now, result)
    return result


# --- network shaping ------------------------------------------------------------


@dataclass(frozen=True)
class _Shape:
    """A delivery profile parsed out of the URL's `{shape}` segment.

    `delay_ms` is charged before the RESPONSE HEADERS of every file, 304s and 416s
    included, so it lands in the browser's TTFB exactly where a real round trip
    would and a 40-chunk refine pays 40 of them. `jitter_ms` adds a uniform
    0..jitter on top, so `d100-j300` is a 100-400ms link. `kbps` caps throughput
    through a bucket SHARED by every response at that rate — a LINK cap, not a
    per-request one, so raising the engine's fetch concurrency divides the same
    pipe instead of multiplying it. `cache` overrides the cache policy, which is
    how you replay a cold load without clearing the browser cache."""

    delay_ms: int = 0
    kbps: int = 0
    jitter_ms: int = 0
    cache: str = ""

    @property
    def token(self) -> str:
        parts = [f"d{self.delay_ms}" if self.delay_ms else ""]
        parts += [f"k{self.kbps}" if self.kbps else "", f"j{self.jitter_ms}" if self.jitter_ms else ""]
        parts += [self.cache]
        joined = "-".join(p for p in parts if p)
        return joined or "direct"


_NEUTRAL = _Shape()
_SHAPE_LIMITS = {"d": 30_000, "k": 10_000_000, "j": 10_000}


def _parse_shape(token: str) -> _Shape:
    """`direct` (or `off`) for untouched delivery, else dash-joined parts in any
    order: `d<ms>` round-trip delay, `k<kbps>` link cap, `j<ms>` added jitter, and
    one of `immutable` / `revalidate` / `nostore` to override the cache policy —
    e.g. `d80-k12000-j20`."""
    lowered = token.lower()
    if lowered in ("direct", "off", "none"):
        return _NEUTRAL
    values = {"d": 0, "k": 0, "j": 0}
    cache = ""
    for part in lowered.split("-"):
        if not part:
            continue
        if part in _CACHE_POLICIES:
            cache = part
            continue
        head, rest = part[0], part[1:]
        if head not in values or not rest.isdigit():
            raise HTTPException(
                status_code=400,
                detail=f"bad shape part {part!r} — expected d<ms>/k<kbps>/j<ms>, a cache "
                       f"policy ({'/'.join(_CACHE_POLICIES)}), or 'direct'",
            )
        values[head] = min(int(rest), _SHAPE_LIMITS[head])
    return _Shape(delay_ms=values["d"], kbps=values["k"], jitter_ms=values["j"], cache=cache)


class _Bucket:
    """Token bucket metering one shared byte rate across every concurrent
    response. The lock serializes takers so they queue in arrival order rather
    than all waking to race for the same tokens."""

    def __init__(self, bytes_per_s: float) -> None:
        self.rate = bytes_per_s
        self.capacity = max(bytes_per_s * 0.25, 8192.0)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self.lock = asyncio.Lock()

    async def take(self, amount: int) -> None:
        async with self.lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                await asyncio.sleep((amount - self.tokens) / self.rate)


_buckets: dict[int, _Bucket] = {}


def _bucket(kbps: int) -> _Bucket:
    bucket = _buckets.get(kbps)
    if bucket is None:
        bucket = _Bucket(kbps * 1000.0 / 8.0)
        _buckets[kbps] = bucket
    return bucket


# --- byte delivery --------------------------------------------------------------


def _resolve(root: _Root, rel: str) -> tuple[Path, os.stat_result]:
    target = (root.path / rel).resolve()
    if not target.is_relative_to(root.path):
        raise HTTPException(status_code=404)
    if target.suffix.lower() not in _MEDIA_TYPES:
        raise HTTPException(
            status_code=404,
            detail=f"not a SOG file kind ({'/'.join(sorted(_MEDIA_TYPES))})",
        )
    try:
        stat = target.stat()
    except OSError:
        raise HTTPException(status_code=404) from None
    if not target.is_file():
        raise HTTPException(status_code=404)
    return target, stat


def _cache_control(path: Path, shape: _Shape) -> str:
    if shape.cache:
        return _CACHE_POLICIES[shape.cache]
    return _REVALIDATE if path.name == MANIFEST_NAME else _IMMUTABLE


def _fresh(request: Request, etag: str, mtime: float) -> bool:
    """Whether the client's copy is still good, per RFC 9110: `If-None-Match`
    wins outright when present, `If-Modified-Since` is the fallback."""
    inm = request.headers.get("if-none-match")
    if inm:
        return inm.strip() == "*" or etag in {tag.strip() for tag in inm.split(",")}
    ims = request.headers.get("if-modified-since")
    if not ims:
        return False
    try:
        since = parsedate_to_datetime(ims).timestamp()
    except (TypeError, ValueError):
        return False
    return int(mtime) <= int(since)


_UNSATISFIABLE = object()


def _parse_range(header: str | None, size: int) -> tuple[int, int] | None | object:
    """A single `bytes=` range as an inclusive (start, end), None when the header
    is absent or not a form we honour (multi-range falls back to the full body,
    which is always a legal answer), or `_UNSATISFIABLE` for a 416."""
    if not header or not header.startswith("bytes=") or "," in header:
        return None
    spec = header[len("bytes="):].strip()
    first, sep, last = spec.partition("-")
    if not sep:
        return None
    try:
        if not first:
            length = int(last)
            if length <= 0:
                return _UNSATISFIABLE
            return max(0, size - length), size - 1
        start = int(first)
        end = int(last) if last else size - 1
    except ValueError:
        return None
    end = min(end, size - 1)
    if start > end or start >= size:
        return _UNSATISFIABLE
    return start, end


async def _body(path: Path, start: int, length: int, shape: _Shape) -> AsyncIterator[bytes]:
    bucket = _bucket(shape.kbps) if shape.kbps else None
    # Small enough that a throttled stream paces smoothly (~20 slices/second)
    # rather than arriving in visible bursts.
    chunk_size = _READ_CHUNK if bucket is None else int(max(8192, min(_READ_CHUNK, bucket.rate / 20)))
    remaining = length
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            data = handle.read(min(chunk_size, remaining))
            if not data:
                return
            remaining -= len(data)
            if bucket is not None:
                await bucket.take(len(data))
            yield data


async def _serve(request: Request, root_id: str, rel: str, shape: _Shape) -> Response:
    # Charged before anything is written, so the wait shows up as time-to-first-byte
    # rather than as a stall midway through a body that already has its headers.
    if shape.delay_ms or shape.jitter_ms:
        await asyncio.sleep(
            (shape.delay_ms + random.uniform(0.0, shape.jitter_ms)) / 1000.0
        )
    root = _root_by_id(root_id)
    path, stat = _resolve(root, rel)
    etag = f'"{stat.st_size:x}-{stat.st_mtime_ns:x}"'
    headers = {
        "ETag": etag,
        "Last-Modified": formatdate(stat.st_mtime, usegmt=True),
        "Cache-Control": _cache_control(path, shape),
        "Accept-Ranges": "bytes",
        # The playground lives on the client origin and the splats on this one,
        # so without these the browser hides both the transfer size (Resource
        # Timing zeroes `transferSize` cross-origin) and the range/length headers
        # from the page — and the streaming HUD silently reads zero.
        "Timing-Allow-Origin": "*",
        "Access-Control-Expose-Headers": (
            "Content-Length, Content-Range, Accept-Ranges, ETag, Last-Modified, Server-Timing"
        ),
        "Server-Timing": f'sog;desc="{shape.token}";dur={shape.delay_ms}',
    }
    if _fresh(request, etag, stat.st_mtime):
        return Response(status_code=304, headers=headers)

    media_type = _MEDIA_TYPES[path.suffix.lower()]
    span = _parse_range(request.headers.get("range"), stat.st_size)
    if span is _UNSATISFIABLE:
        return Response(
            status_code=416, headers={**headers, "Content-Range": f"bytes */{stat.st_size}"}
        )
    if isinstance(span, tuple):
        start, end = span
        status, length = 206, end - start + 1
        headers["Content-Range"] = f"bytes {start}-{end}/{stat.st_size}"
    else:
        start, status, length = 0, 200, stat.st_size
    headers["Content-Length"] = str(length)

    if request.method == "HEAD":
        return Response(status_code=status, headers=headers, media_type=media_type)
    return StreamingResponse(
        _body(path, start, length, shape),
        status_code=status,
        headers=headers,
        media_type=media_type,
    )


# --- routes ---------------------------------------------------------------------


@router.get("/catalog")
async def catalog(refresh: bool = False) -> dict[str, object]:
    """What this server can deliver and what can still be built, newest first.

    `assets` is every streamable SOG, each with the `url` to hand the viewer plus
    what it costs to stream: `bytes` on the wire, `splats` at full detail,
    `lod_levels`, per-level `counts`, and `chunk_files` (how many separate fetches
    a full refine takes). `sources` is every cell PLY a streamed bundle can be
    compiled FROM, with its `build_path` to POST to, whether a bundle already
    exists, and how many trainer-built ladder levels sit beside it. Cached for a
    few seconds; `?refresh=1` rescans."""
    found = _catalog(refresh)
    return {
        "roots": [
            {"id": r.id, "path": str(r.path), "kind": r.kind, "cells": r.cells}
            for r in roots()
        ],
        "assets": found["assets"],
        "sources": found["sources"],
        "shape": {
            # Swap an asset's `url` prefix for this to stream it through a profile;
            # the shaping rides ahead of the path so the manifest's chunks inherit it.
            "template": f"{router.prefix}/n/{{shape}}/{{root}}/{{path}}",
            "grammar": "direct | dash-joined d<ms> k<kbps> j<ms> "
                       f"{' '.join(sorted(_CACHE_POLICIES))}",
            "example": "d80-k12000-j20",
        },
    }


@router.api_route("/f/{root_id}/{rel:path}", methods=["GET", "HEAD"])
async def sog_file(request: Request, root_id: str, rel: str) -> Response:
    """SOG bytes under the default policy: byte ranges, strong validators, and
    immutable caching on everything but the manifest."""
    return await _serve(request, root_id, rel, _NEUTRAL)


@router.api_route("/n/{shape}/{root_id}/{rel:path}", methods=["GET", "HEAD"])
async def sog_file_shaped(request: Request, shape: str, root_id: str, rel: str) -> Response:
    """The same bytes with a network profile applied (see `_parse_shape`). Because
    `{shape}` precedes the path, a shaped manifest URL shapes every chunk the
    engine derives from it."""
    return await _serve(request, root_id, rel, _parse_shape(shape))
