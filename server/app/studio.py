"""Standalone "phrase studio" — pick a noun phrase, wrap it, watch it
become an image (Nano Banana) and then a 3D object (Trellis).

A separate single-page client for exercising the real production
asset pipeline one object at a time:

  * `GET  /phrases`  — the noun phrases extracted into
                       `object_noun_phrases.json`, flattened + faceted
                       (scene / model) for the picker.
  * `POST /wrap`     — preview the exact `wrap_image_prompt` directive a
                       phrase + proxy_shape + view (+ optional dims)
                       turns into. Lets the UI show what Nano Banana sees
                       before spending a call.
  * `POST /banana`   — wrap the phrase, then call the PRODUCTION Nano
                       Banana client (`app.services.nano_banana`, Google
                       GenAI). Saves the PNG and returns its URL.
  * `POST /trellis`  — feed a previously generated image to the
                       PRODUCTION Trellis client (`app.services.threed`,
                       the hosted Modal endpoint) and serve the resulting
                       textured GLB.

Unlike the pipeline's `generation._generate_one`, this serves Trellis'
RAW textured GLB directly — no trimesh symmetrize/rescale pass — so the
viewer needs neither a bounding box to fit into nor scipy installed; the
embedded-texture GLB drops straight into three.js' GLTFLoader.

Run via `./scripts/run_studio.py` (or, from `server/`:
`uv run uvicorn app.studio:app --port 8770`).
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.core.prompts import wrap_image_prompt
from app.core.types import BoundingBox, ProxyShape
from app.services import nano_banana, symmetry, threed
from app.services import publish as publish_svc
from app.services import scene_lite as scene_lite_svc
from app.utils import logging as rlog
from app.utils.logging import SlotLog

load_dotenv()

# Python's mimetypes doesn't reliably know these; without them the static
# mounts would hand GLBs/modules/wasm to the browser with wrong types — ES
# modules and `WebAssembly` (the KTX2 Basis transcoder) refuse to load unless
# the MIME is right.
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("model/gltf+json", ".gltf")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")

# studio.py lives at server/app/studio.py:
#   parents[0] = server/app   parents[1] = server   parents[2] = repo root
_APP_DIR = Path(__file__).resolve().parent
_SERVER_DIR = _APP_DIR.parent
_REPO_ROOT = _SERVER_DIR.parent

PHRASES_PATH = _REPO_ROOT / "object_noun_phrases.json"
STATIC_DIR = _APP_DIR / "studio_static"

# Generated artifacts (images + meshes + the studio's resumable event log)
# land here. Served read-only at /artifacts; gitignored.
OUT_DIR = _SERVER_DIR / "studio-artifacts"
IMAGES_DIR = OUT_DIR / "images"
MESHES_DIR = OUT_DIR / "meshes"
# Symmetric-asset outputs (raw + optimized twins), served at /artifacts/symmetric.
SYM_DIR = OUT_DIR / "symmetric"

# The pre-built asset library: raw (trimesh-readable) GLBs are the symmetrize
# source; the optimized twins are what the viewer renders for the "before" pane.
LIBRARY_DIR = _APP_DIR / "assets_library"
RAW_ASSETS_DIR = LIBRARY_DIR / "assets"
OPT_ASSETS_DIR = LIBRARY_DIR / "assets-optimized"
CATALOG_PATH = LIBRARY_DIR / "library.json"

# The optimized library GLBs are Meshopt + KTX2/Basis, so the viewer needs the
# Basis transcoder and Meshopt decoder. Serve three.js locally (from the client's
# install) instead of a CDN — fetching the 500KB transcoder wasm per worker over
# unpkg is what made the symmetry view crawl, while the client loads it instantly.
VENDOR_THREE_DIR = _REPO_ROOT / "client" / "node_modules" / "three"

# The pipeline's runs tree — shared with the main API server, which WRITES the
# persisted walkthrough tours under <run>/<slot>/<model>/tour/. Honor the same
# STARSHOT_RUNS_DIR env var the API server uses (default: the repo's runs/), so
# the /pano client can list (GET /tours) and load (served read-only at /runs)
# every rendered tour same-origin — no cross-origin or main-server-URL guessing.
RUNS_DIR = Path(os.environ.get("STARSHOT_RUNS_DIR", str(_REPO_ROOT / "runs"))).resolve()

# Maps the UI's proxy-shape strings to the pipeline's ProxyShape enum.
# `rectangular_prism` is the bare AABB (None), matching the rest of the
# pipeline's convention (see app.core.types.ProxyShape).
_PROXY_SHAPES: dict[str, ProxyShape | None] = {
    "rectangular_prism": None,
    "sphere": ProxyShape.SPHERE,
    "capsule": ProxyShape.CAPSULE,
    "hemisphere": ProxyShape.HEMISPHERE,
}
_VIEWS = frozenset({"front", "side", "top", "three-quarter"})

# One persistent SlotLog bound around every Trellis call. threed.generate_mesh
# emits trellis.submit/.done and drives its resumable completion cache through
# the bound log, so binding a stable log makes a repeat Trellis on the same
# image short-circuit to the cached GLB instead of re-billing Modal.
_studio_log: SlotLog | None = None


def _phrase_log() -> SlotLog:
    global _studio_log
    if _studio_log is None:
        _studio_log = SlotLog("studio", OUT_DIR / "events.jsonl")
        _studio_log.hydrate_from_disk()
    return _studio_log


# --- noun phrases -----------------------------------------------------------

_phrases_cache: dict[str, Any] | None = None


def _load_phrases() -> dict[str, Any]:
    """Flatten object_noun_phrases.json into a flat list plus scene/model
    facets for the picker. Cached after first read (the file is a static
    export; restart the studio to pick up regenerated phrases)."""
    global _phrases_cache
    if _phrases_cache is not None:
        return _phrases_cache
    if not PHRASES_PATH.exists():
        _phrases_cache = {"phrases": [], "scenes": [], "models": [], "source": None}
        return _phrases_cache
    raw = json.loads(PHRASES_PATH.read_text(encoding="utf-8"))
    phrases: list[dict[str, str]] = []
    scenes: set[str] = set()
    models: set[str] = set()
    for log in raw.get("logs", []):
        scene = str(log.get("scene", ""))
        model = str(log.get("model", ""))
        scenes.add(scene)
        models.add(model)
        for obj in log.get("objects", []):
            noun = obj.get("noun_phrase")
            if not isinstance(noun, str) or not noun.strip():
                continue
            phrases.append(
                {
                    "name": str(obj.get("name", "")),
                    "noun_phrase": noun.strip(),
                    "source": str(obj.get("source", "")),
                    "scene": scene,
                    "model": model,
                }
            )
    _phrases_cache = {
        "phrases": phrases,
        "scenes": sorted(s for s in scenes if s),
        "models": sorted(m for m in models if m),
        "source": raw.get("source_dir"),
        "count": len(phrases),
    }
    return _phrases_cache


# --- asset library ----------------------------------------------------------

_library_cache: dict[str, Any] | None = None


def _load_library() -> dict[str, Any]:
    """Catalog of library assets that have a raw GLB on disk (the symmetrize
    source), plus the category facets for the picker. Cached after first read."""
    global _library_cache
    if _library_cache is not None:
        return _library_cache
    items: list[dict[str, str]] = []
    if CATALOG_PATH.exists():
        for it in json.loads(CATALOG_PATH.read_text(encoding="utf-8")):
            asset_id = str(it.get("id", ""))
            if asset_id and (RAW_ASSETS_DIR / f"{asset_id}.glb").exists():
                items.append(
                    {
                        "id": asset_id,
                        "description": str(it.get("description", "")),
                        "category": str(it.get("category", "")),
                    }
                )
    items.sort(key=lambda a: (a["category"], a["description"]))
    categories = sorted({a["category"] for a in items if a["category"]})
    _library_cache = {"assets": items, "categories": categories, "count": len(items)}
    return _library_cache


# --- request models ---------------------------------------------------------


class WrapRequest(BaseModel):
    phrase: str
    proxy_shape: str = "rectangular_prism"
    view: str = "front"
    width: float | None = None
    height: float | None = None
    depth: float | None = None


class BananaRequest(WrapRequest):
    # Optional human-readable handle (the object name) used only to build a
    # friendly artifact filename; the id is always uniquified server-side.
    name: str | None = None


class TrellisRequest(BaseModel):
    # The image_id returned by /banana (the artifact stem).
    image_id: str


class SymmetrizeRequest(BaseModel):
    library_id: str
    axis: int = 2  # 0=X (left/right), 1=Y (up/down), 2=Z (front/back)
    keep_positive: bool = True  # keep the +axis half (the +Z front by default)
    orientation: int = 0  # yaw applied during rescale (one of the 8 allowed steps)
    # Optional target box (meters). All three or none; none = keep natural extents.
    width: float | None = None
    height: float | None = None
    depth: float | None = None


# --- helpers ----------------------------------------------------------------


def _resolve_wrap(req: WrapRequest) -> str:
    phrase = req.phrase.strip()
    if not phrase:
        raise HTTPException(400, "phrase is empty")
    if req.proxy_shape not in _PROXY_SHAPES:
        raise HTTPException(400, f"unknown proxy_shape: {req.proxy_shape!r}")
    if req.view not in _VIEWS:
        raise HTTPException(400, f"unknown view: {req.view!r}")
    proxy = _PROXY_SHAPES[req.proxy_shape]
    dims = (req.width, req.height, req.depth)
    if all(d is not None for d in dims):
        if any(d is not None and d <= 0 for d in dims):
            raise HTTPException(400, "dimensions must be positive")
        dimensions = (float(req.width), float(req.height), float(req.depth))  # type: ignore[arg-type]
    elif any(d is not None for d in dims):
        raise HTTPException(400, "provide all of width/height/depth or none")
    else:
        dimensions = None
    return wrap_image_prompt(phrase, proxy, dimensions, view=req.view)  # type: ignore[arg-type]


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return (s[:48] or "object").strip("-")


# --- scene-lite (vertex-color web export) cell resolution -------------------
#
# A "cell" is one run/slot/model dir under the runs tree. The /lite page bakes a
# cell's raw generated meshes into one small, textureless vertex-colored GLB.

SCENE_LITE_NAME = "scene-lite.glb"


def _scene_object_glbs(objects_dir: Path) -> list[Path]:
    """The finished per-object GLBs in `objects_dir`, excluding the `.raw.glb`
    Trellis intermediates."""
    return [p for p in objects_dir.glob("*.glb") if not p.name.endswith(".raw.glb")]


def _cell_raw_objects_dir(run: str, slot: str, model: str) -> Path | None:
    """Newest generated version's RAW objects (PNG-textured, sharp-decodable) —
    the source for the vertex-color bake. The optimized twins are KTX2/Basis,
    which can't be decoded to sample colors. None when the cell has no raw
    generated meshes (e.g. a library-only cell)."""
    gen_root = RUNS_DIR / run / slot / model / "generated"
    if not gen_root.is_dir():
        return None
    versions = sorted(
        (p for p in gen_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
        reverse=True,
    )
    for v in versions:
        raw = v / "objects-generated"
        if raw.is_dir() and _scene_object_glbs(raw):
            return raw
    return None


def _safe_cell_part(part: str) -> str:
    if not part or "/" in part or "\\" in part or part.startswith("."):
        raise HTTPException(400, f"invalid cell path component: {part!r}")
    return part


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await threed.disconnect_http()
            if _studio_log is not None:
                _studio_log.close()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MESHES_DIR.mkdir(parents=True, exist_ok=True)
    SYM_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/artifacts", StaticFiles(directory=str(OUT_DIR)), name="artifacts")
    # The pre-built library, served read-only for the symmetry tool's "before"
    # pane. check_dir=False so the studio still boots if a set isn't present.
    app.mount(
        "/library-optimized",
        StaticFiles(directory=str(OPT_ASSETS_DIR), check_dir=False),
        name="library-optimized",
    )
    app.mount(
        "/library-raw",
        StaticFiles(directory=str(RAW_ASSETS_DIR), check_dir=False),
        name="library-raw",
    )
    # three.js (build + addons + Basis transcoder), served locally for parity
    # with the client viewer. check_dir=False so the studio still boots if the
    # client hasn't been `npm install`ed yet.
    app.mount(
        "/vendor/three",
        StaticFiles(directory=str(VENDOR_THREE_DIR), check_dir=False),
        name="vendor-three",
    )
    # The runs tree (read-only), so the /pano client can fetch persisted tour
    # bundles (tour.json + panos + proxy.glb) same-origin. check_dir=False so the
    # studio still boots when no runs exist yet.
    app.mount(
        "/runs",
        StaticFiles(directory=str(RUNS_DIR), check_dir=False),
        name="runs",
    )

    @app.get("/")
    async def index() -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/studio.js")
    async def studio_js() -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        return FileResponse(STATIC_DIR / "studio.js", media_type="text/javascript")

    @app.get("/symmetry")
    async def symmetry_page() -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        return FileResponse(STATIC_DIR / "symmetry.html", media_type="text/html")

    @app.get("/symmetry.js")
    async def symmetry_js() -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        return FileResponse(STATIC_DIR / "symmetry.js", media_type="text/javascript")

    @app.get("/pano")
    async def pano_page() -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        return FileResponse(STATIC_DIR / "pano.html", media_type="text/html")

    @app.get("/pano.js")
    async def pano_js() -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        return FileResponse(STATIC_DIR / "pano.js", media_type="text/javascript")

    @app.get("/lite")
    async def lite_page() -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        return FileResponse(STATIC_DIR / "lite.html", media_type="text/html")

    @app.get("/lite.js")
    async def lite_js() -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        return FileResponse(STATIC_DIR / "lite.js", media_type="text/javascript")

    @app.get("/orbit")
    async def orbit_page() -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        return FileResponse(STATIC_DIR / "orbit.html", media_type="text/html")

    @app.get("/orbit.js")
    async def orbit_js() -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        return FileResponse(STATIC_DIR / "orbit.js", media_type="text/javascript")

    @app.get("/scenes")
    async def scenes() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Every generated cell (run/slot/model) with raw meshes to bake, newest
        first, flagged with whether its scene-lite.glb has been built."""
        items: list[dict[str, Any]] = []
        if RUNS_DIR.is_dir():
            for cell in RUNS_DIR.glob("*/*/*"):
                if not cell.is_dir():
                    continue
                run, slot, model = cell.relative_to(RUNS_DIR).parts
                raw_dir = _cell_raw_objects_dir(run, slot, model)
                if raw_dir is None:
                    continue
                lite = cell / SCENE_LITE_NAME
                built = lite.is_file()
                items.append(
                    {
                        "run": run,
                        "slot": slot,
                        "model": model,
                        "objects": len(_scene_object_glbs(raw_dir)),
                        "built": built,
                        "url": "/runs/" + "/".join(quote(p) for p in (run, slot, model, SCENE_LITE_NAME))
                        if built
                        else None,
                        "bytes": lite.stat().st_size if built else None,
                        "mtime": cell.stat().st_mtime,
                    }
                )
        items.sort(key=lambda x: x["mtime"], reverse=True)
        return {"scenes": items}

    @app.post("/scene-lite/{run}/{slot}/{model}")
    async def scene_lite(run: str, slot: str, model: str, force: bool = False) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Bake a cell's raw generated meshes into one small, textureless
        vertex-colored scene-lite.glb (cached under the cell) and return its
        /runs URL + stats. Re-runs only when forced or a source GLB / the bake
        script is newer than the cached build."""
        run, slot, model = _safe_cell_part(run), _safe_cell_part(slot), _safe_cell_part(model)
        raw_dir = _cell_raw_objects_dir(run, slot, model)
        if raw_dir is None:
            raise HTTPException(404, "no raw generated meshes to bake for this cell")
        sources = _scene_object_glbs(raw_dir)
        dst = RUNS_DIR / run / slot / model / SCENE_LITE_NAME
        # Stale when a source GLB or the bake script is newer than the cached build.
        floor_mtime = scene_lite_svc.BAKE_SCRIPT.stat().st_mtime if scene_lite_svc.BAKE_SCRIPT.is_file() else 0.0
        for s in sources:
            floor_mtime = max(floor_mtime, s.stat().st_mtime)
        fresh = dst.is_file() and dst.stat().st_mtime >= floor_mtime
        if force or not fresh:
            try:
                stats = await scene_lite_svc.build_scene_vcolor(raw_dir, dst)
            except Exception as e:  # surface bake failures to the client as 502
                raise HTTPException(502, f"{type(e).__name__}: {e}") from e
        else:
            stats = {"objects": len(sources), "outBytes": dst.stat().st_size}
        return {
            "url": "/runs/" + "/".join(quote(p) for p in (run, slot, model, SCENE_LITE_NAME)),
            "bytes": dst.stat().st_size,
            **stats,
        }

    @app.post("/publish/{run}/{slot}/{model}")
    async def publish(run: str, slot: str, model: str, version: str | None = None, dry_run: bool = False) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Publish this cell's dollhouse preview + capture tour to the R2 bucket
        under per-(run/slot/model/version) keys and upsert its D1 catalog row.
        `version` defaults to the latest generated build with rendered meshes;
        re-publishing the same version overwrites the objects in place. `dry_run`
        returns the planned keys without baking, uploading, or writing D1."""
        run, slot, model = _safe_cell_part(run), _safe_cell_part(slot), _safe_cell_part(model)
        try:
            return await publish_svc.publish_cell(RUNS_DIR, run, slot, model, version, dry_run=dry_run)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:  # R2/D1 failures (creds, network) surface as 502
            raise HTTPException(502, f"{type(e).__name__}: {e}") from e

    @app.get("/tours")
    async def tours() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Every persisted walkthrough tour (<run>/<slot>/<model>/tour/tour.json)
        under the runs tree, newest first, each with a same-origin /runs/ URL the
        /pano client loads directly."""
        items: list[dict[str, Any]] = []
        if RUNS_DIR.is_dir():
            for manifest in RUNS_DIR.glob("*/*/*/tour/tour.json"):
                try:
                    parts = manifest.relative_to(RUNS_DIR).parts
                    if len(parts) != 5:
                        continue
                    data = json.loads(manifest.read_text(encoding="utf-8"))
                    panos = data.get("panos")
                    items.append(
                        {
                            "run": parts[0],
                            "slot": parts[1],
                            "model": parts[2],
                            "url": "/runs/" + "/".join(quote(p) for p in parts),
                            "panos": len(panos) if isinstance(panos, list) else 0,
                            "has_proxy": bool(data.get("proxy")),
                            "mtime": manifest.stat().st_mtime,
                        }
                    )
                except (OSError, ValueError):
                    continue
        items.sort(key=lambda x: x["mtime"], reverse=True)
        return {"tours": items}

    @app.get("/phrases")
    async def phrases() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return _load_phrases()

    @app.post("/wrap")
    async def wrap(req: WrapRequest) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        return {"wrapped_prompt": _resolve_wrap(req)}

    @app.post("/banana")
    async def banana(req: BananaRequest) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        wrapped = _resolve_wrap(req)
        stem = f"{_slug(req.name or req.phrase)}-{uuid.uuid4().hex[:8]}"
        out_path = IMAGES_DIR / f"{stem}.png"
        rlog.bind(_phrase_log())
        try:
            result = await nano_banana.generate(wrapped)
        except Exception as e:  # surface provider failures to the client as 502
            raise HTTPException(502, f"{type(e).__name__}: {e}") from e
        nano_banana.save(result, out_path)
        return {
            "image_id": stem,
            "image_url": f"/artifacts/images/{stem}.png",
            "wrapped_prompt": wrapped,
        }

    @app.post("/trellis")
    async def trellis(req: TrellisRequest) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        stem = Path(req.image_id).name  # defend against path traversal
        image_path = IMAGES_DIR / f"{stem}.png"
        if not image_path.is_file():
            raise HTTPException(404, f"no image for id: {stem}")
        out_path = MESHES_DIR / f"{stem}.glb"
        rlog.bind(_phrase_log())
        try:
            await threed.generate_mesh(
                image_path.read_bytes(),
                output_path=out_path,
                job_id=stem,
                image_mime="image/png",
            )
        except Exception as e:  # surface provider failures to the client as 502
            raise HTTPException(502, f"{type(e).__name__}: {e}") from e
        return {"glb_url": f"/artifacts/meshes/{stem}.glb"}

    @app.get("/library")
    async def library_assets() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        return _load_library()

    _ORIENTATIONS = frozenset({-180, -135, -90, -45, 0, 45, 90, 135, 180})

    @app.post("/symmetrize")
    async def symmetrize(req: SymmetrizeRequest) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        stem = Path(req.library_id).name  # defend against path traversal
        src = RAW_ASSETS_DIR / f"{stem}.glb"
        if not src.is_file():
            raise HTTPException(404, f"no raw library asset for id: {stem}")
        if req.axis not in (0, 1, 2):
            raise HTTPException(400, "axis must be 0, 1, or 2")
        if req.orientation not in _ORIENTATIONS:
            raise HTTPException(400, f"orientation must be one of {sorted(_ORIENTATIONS)}")

        dims = (req.width, req.height, req.depth)
        bbox: BoundingBox | None = None
        if all(d is not None for d in dims):
            if any(d is not None and d <= 0 for d in dims):
                raise HTTPException(400, "dimensions must be positive")
            bbox = BoundingBox.from_center_size(
                (0.0, 0.0, 0.0),
                (float(req.width), float(req.height), float(req.depth)),  # type: ignore[arg-type]
            )
        elif any(d is not None for d in dims):
            raise HTTPException(400, "provide all of width/height/depth or none")

        params = f"{stem}|a{req.axis}|k{int(req.keep_positive)}|o{req.orientation}|d{dims}"
        key = f"{stem[:40]}-{hashlib.sha1(params.encode()).hexdigest()[:8]}"
        raw_out = SYM_DIR / f"{key}.raw.glb"
        opt_out = SYM_DIR / f"{key}.glb"
        try:
            stats = await symmetry.build_symmetric_glb(
                src=src,
                raw_out=raw_out,
                opt_out=opt_out,
                axis=req.axis,
                keep_positive=req.keep_positive,
                orientation=req.orientation,  # type: ignore[arg-type]
                bbox=bbox,
            )
        except Exception as e:  # surface processing/optimize failures as 502
            raise HTTPException(502, f"{type(e).__name__}: {e}") from e

        has_opt = (OPT_ASSETS_DIR / f"{stem}.glb").exists()
        original_url = (
            f"/library-optimized/{stem}.glb" if has_opt else f"/library-raw/{stem}.glb"
        )
        return {
            "original_url": original_url,
            "symmetric_url": f"/artifacts/symmetric/{opt_out.name}",
            "reference_url": f"/library-optimized/{stem}.png"
            if (OPT_ASSETS_DIR / f"{stem}.png").exists()
            else None,
            **stats,
        }

    return app


app = create_app()
