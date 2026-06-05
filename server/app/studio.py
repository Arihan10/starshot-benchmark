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
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.core.prompts import wrap_image_prompt
from app.core.types import BoundingBox, ProxyShape
from app.services import nano_banana, symmetry, threed
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
