"""Standalone dev playground: prompt → Nano Banana → Trellis 2 → GLB.

A minimal FastAPI app with:
  * `GET  /`         — single-page UI for prompt experimentation.
  * `GET  /presets`  — preset subject phrases + proxy shapes + bboxes
                       that previously produced cropped renders.
  * `POST /banana`   — text → image. Returns the Runware image URL.
  * `POST /trellis`  — image → 3D. Returns the Runware GLB URL.

Both stages run on Runware so the playground can exercise both
nano-banana-2 vs nano-banana-pro side-by-side. The production
pipeline now uses Google GenAI Nano Banana (see
`app.services.nano_banana`) and Runware Trellis (see
`app.services.threed`); this playground stays Runware-only for the
image-gen comparison view. Model IDs and per-model settings are
inlined below since `app.services.threed` no longer carries them.

Run via `enx playground` (or `uv run scripts/run_playground.py`).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from runware import (
    I3dInference,
    I3dInputs,
    IAsyncTaskResponse,
    IImageInference,
    ISettings,
    Runware,
)

from app.core.prompts import wrap_image_prompt
from app.core.types import ProxyShape

load_dotenv()

TRELLIS_MODEL = os.environ.get("TRELLIS_MODEL", "microsoft:trellis-2@4b")

# Runware Nano Banana model IDs. Inlined here (previously imported from
# app.services.threed) because the production pipeline no longer goes
# through Runware for image-gen; threed.py is now Trellis-only.
NANO_BANANA_2 = "google:4@3"
NANO_BANANA_PRO = "google:4@2"

_BANANA_MODELS: dict[str, str] = {
    "nano-banana-2": NANO_BANANA_2,
    "nano-banana-pro": NANO_BANANA_PRO,
}


def banana_settings_for(model: str) -> dict[str, Any]:
    """Per-model production settings for the Runware imageInference call.

    Runware rejects `resolution` for text-to-image (the preset is only
    meaningful when `referenceImages` is set, where it matches the input
    aspect ratio). For pure text-to-image, `width`/`height` are required.
    nano-banana-2 (`google:4@3`) runs at 512×512 with thinking=MINIMAL;
    nano-banana-pro (`google:4@2`) runs at 1024×1024.
    """
    if model == NANO_BANANA_2:
        return {"width": 512, "height": 512, "thinking": "MINIMAL"}
    if model == NANO_BANANA_PRO:
        return {"width": 1024, "height": 1024}
    return {}


_PROXY_SHAPES: dict[str, ProxyShape | None] = {
    "rectangular_prism": None,
    "sphere": ProxyShape.SPHERE,
    "capsule": ProxyShape.CAPSULE,
    "hemisphere": ProxyShape.HEMISPHERE,
}


# Preset subject phrases — pulled from runs that produced cropped
# images under the dimensions-less wrapper. proxy_shape and
# dimensions match the values from the original run so the with-dims
# wrapper is exercised under the same conditions.
_PRESETS: list[dict[str, object]] = [
    {
        "id": "banner_pole",
        "label": "banner pole (capsule, pole clipped at bottom)",
        "proxy_shape": "capsule",
        "dimensions": [0.40, 3.00, 0.40],
        "subject": (
            "a tall slender wrought-iron banner pole with rust-pitted "
            "shaft, forged spear-point finial, bronze rope-ring crossbar, "
            "flying a long tattered ox-blood crimson silk banner with "
            "battle-frayed gold fringe and embroidered bronze "
            "laurel-and-gladius device"
        ),
    },
    {
        "id": "swallow_pair",
        "label": "barn swallow pair (prism, wings & tails clipped)",
        "proxy_shape": "rectangular_prism",
        "dimensions": [3.00, 0.50, 2.20],
        "subject": (
            "a pair of barn swallows in mid-glide with deeply forked "
            "tails, glossy steel-blue backs, rust-orange throats, creamy "
            "bellies, and outstretched wings banking in a shallow turn"
        ),
    },
    {
        "id": "ringed_gas_giant",
        "label": "ringed gas giant (prism, rings clipped left/right)",
        "proxy_shape": "rectangular_prism",
        "dimensions": [40.00, 18.00, 40.00],
        "subject": (
            "a massive banded gas giant with cream, butterscotch and "
            "ochre horizontal cloud belts, a swirling dark storm vortex, "
            "encircled by wide flat pale icy rings with cassini-style "
            "gaps casting a thin shadow across its upper hemisphere"
        ),
    },
    {
        "id": "spanish_moss_curtain",
        "label": "spanish moss curtain (prism, full-bleed)",
        "proxy_shape": "rectangular_prism",
        "dimensions": [1.50, 2.50, 0.30],
        "subject": (
            "a long draping curtain of pale silver-green spanish moss "
            "in ragged matted tendrils with dust-soft wispy strands, "
            "thicker rope-like veils, and a few dead twigs and curled "
            "brown leaves caught within"
        ),
    },
    {
        "id": "plane_tree_row",
        "label": "London plane tree row (prism, canopy clipped at top)",
        "proxy_shape": "rectangular_prism",
        "dimensions": [250.00, 12.00, 5.00],
        "subject": (
            "a long linear row of seven mature London plane street trees "
            "in a granite-curbed planter strip with mottled cream-and-olive "
            "exfoliating bark, gnarled buttressed root flares, broad "
            "palmate canopies with golden-edged leaves, dangling spiky "
            "seed-balls, cast-iron tree grates, slim stainless trunk cages, "
            "festoon string lights, mulched boxwood understory, "
            "polished-granite bench coping, and dusk-glowing uplighter spikes"
        ),
    },
    {
        "id": "floating_staircase",
        "label": "cantilevered staircase (prism, top & bottom clipped)",
        "proxy_shape": "rectangular_prism",
        "dimensions": [1.40, 5.00, 4.00],
        "subject": (
            "a tall sculptural floating staircase with twelve thick warm "
            "walnut treads cantilevering from a board-formed concrete "
            "spine wall, slim matte-bronze rod-and-flat-bar handrail, and "
            "warm brushed-bronze under-tread stringer lights"
        ),
    },
    {
        "id": "ceiling_pendant",
        "label": "ceiling pendant (capsule, cord clipped at top)",
        "proxy_shape": "capsule",
        "dimensions": [0.30, 0.80, 0.30],
        "subject": (
            "a slim matte-black cylindrical pendant ceiling lamp with "
            "braided fabric cord and warm soft glow at its lower opening"
        ),
    },
    {
        "id": "earthlike_planet",
        "label": "earthlike planet (sphere, full-bleed)",
        "proxy_shape": "sphere",
        "dimensions": [10.00, 10.00, 10.00],
        "subject": (
            "a vibrant earthlike ocean planet with deep sapphire-blue "
            "seas, ochre-and-green continents, swirling white cloud "
            "bands, a hemispheric cyclone and a thin pale-blue "
            "atmospheric halo"
        ),
    },
    {
        "id": "broken_trident",
        "label": "broken trident (prism, shaft clipped at bottom)",
        "proxy_shape": "rectangular_prism",
        "dimensions": [1.80, 0.12, 0.35],
        "subject": (
            "a snapped ash-wood trident with rust-pitted barbed iron "
            "head crusted in dried gore, splintered shaft exposing pale "
            "fresh wood, and unraveling leather hand-wrap"
        ),
    },
    {
        "id": "north_end_wall",
        "label": "board-formed concrete wall (prism, full-bleed)",
        "proxy_shape": "rectangular_prism",
        "dimensions": [14.00, 11.65, 0.30],
        "subject": (
            "a tall monolithic board-formed pale warm grey concrete "
            "wall with crisp horizontal timber-grain seams and a "
            "regular grid of small recessed tie-rod holes"
        ),
    },
]


class BananaRequest(BaseModel):
    prompt: str
    proxy_shape: str = "rectangular_prism"
    model: Literal["nano-banana-2", "nano-banana-pro"] = "nano-banana-pro"
    width: float | None = None
    height: float | None = None
    depth: float | None = None


class TrellisRequest(BaseModel):
    image_url: str


_client: Runware | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> Runware:
    global _client
    async with _client_lock:
        if _client is None:
            _client = Runware(
                api_key=os.environ["RUNWARE_API_KEY"],
                timeout=180,
                max_retries=0,
            )
            await _client.connect()
        else:
            await _client.ensureConnection()
        return _client


async def _disconnect_client() -> None:
    global _client
    async with _client_lock:
        if _client is not None:
            with contextlib.suppress(Exception):
                await _client.disconnect()
            _client = None


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await _disconnect_client()

    app = FastAPI(
        docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:  # pyright: ignore[reportUnusedFunction]
        return _PAGE

    @app.get("/presets")
    async def presets() -> dict[str, list[dict[str, object]]]:  # pyright: ignore[reportUnusedFunction]
        return {"presets": _PRESETS}

    @app.post("/banana")
    async def banana(req: BananaRequest) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        if not req.prompt.strip():
            raise HTTPException(400, "prompt is empty")
        if req.proxy_shape not in _PROXY_SHAPES:
            raise HTTPException(
                400, f"unknown proxy_shape: {req.proxy_shape!r}"
            )
        if req.width is None or req.height is None or req.depth is None:
            raise HTTPException(
                400, "width, height, depth (subject dimensions) are required",
            )
        proxy = _PROXY_SHAPES[req.proxy_shape]
        final_prompt = wrap_image_prompt(
            req.prompt, proxy, (req.width, req.height, req.depth),
        )
        model_id = _BANANA_MODELS[req.model]
        settings_dict = banana_settings_for(model_id)
        gen_settings = (
            ISettings(thinking=settings_dict["thinking"])
            if "thinking" in settings_dict
            else None
        )
        try:
            client = await _get_client()
            task_uuid = str(uuid.uuid4())
            request = IImageInference(
                taskUUID=task_uuid,
                model=model_id,
                positivePrompt=final_prompt,
                width=settings_dict.get("width"),
                height=settings_dict.get("height"),
                resolution=settings_dict.get("resolution"),
                settings=gen_settings,
                outputFormat="PNG",
                outputType="URL",
                deliveryMethod="async",
                numberResults=1,
            )
            ack = await client.imageInference(requestImage=request)
            results = (
                await client.getResponse(taskUUID=task_uuid, numberResults=1)
                if isinstance(ack, IAsyncTaskResponse)
                else ack
            )
            if not results:
                raise RuntimeError(f"empty result list for task {task_uuid}")
            image = results[0]
        except Exception as e:
            raise HTTPException(502, f"{type(e).__name__}: {e}") from e
        url = getattr(image, "imageURL", None) or ""
        return {
            "image_url": url,
            "wrapped_prompt": final_prompt,
            "model_id": model_id,
            "settings": settings_dict,
        }

    @app.post("/trellis")
    async def trellis(req: TrellisRequest) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        if not req.image_url.strip():
            raise HTTPException(400, "image_url is empty")
        try:
            client = await _get_client()
            task_uuid = str(uuid.uuid4())
            request = I3dInference(
                taskUUID=task_uuid,
                model=TRELLIS_MODEL,
                inputs=I3dInputs(image=req.image_url),
                settings=ISettings(
                    remesh=False, resolution=512, textureSize=1024,
                ),
                outputFormat="GLB",
                outputType="URL",
                deliveryMethod="async",
                numberResults=1,
            )
            ack = await client.inference3d(request3d=request)
            results = (
                await client.getResponse(taskUUID=task_uuid, numberResults=1)
                if isinstance(ack, IAsyncTaskResponse)
                else ack
            )
            if not results:
                raise RuntimeError(f"empty result list for task {task_uuid}")
            mesh = results[0]
        except Exception as e:
            raise HTTPException(502, f"{type(e).__name__}: {e}") from e
        outputs = getattr(mesh, "outputs", None)
        files = getattr(outputs, "files", None) if outputs else None
        if not files:
            raise HTTPException(502, f"trellis result missing files: {mesh!r}")
        first = files[0]
        url = first.get("url") if isinstance(first, dict) else getattr(first, "url", None)
        if not url:
            raise HTTPException(502, f"trellis result missing url: {first!r}")
        return {"glb_url": url}

    return app


app = create_app()


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>nano banana → trellis playground</title>
<style>
  html, body { margin: 0; height: 100%; background: #101114; color: #e6e6e6;
    font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; }
  #app { display: grid; grid-template-columns: 380px 1fr; height: 100%; }
  #panel { padding: 12px; overflow-y: auto; border-right: 1px solid #2a2d35;
    display: flex; flex-direction: column; gap: 10px; background: #16181d; }
  #stage { display: grid; grid-template-columns: 1fr 1fr; min-width: 0; min-height: 0; }
  .col { display: grid; grid-template-rows: auto 1fr 1fr; min-width: 0; min-height: 0;
    border-right: 1px solid #2a2d35; }
  .col:last-child { border-right: none; }
  .col-header { padding: 8px 12px; background: #16181d; color: #b8bcc4;
    border-bottom: 1px solid #2a2d35; font-size: 12px;
    text-transform: uppercase; letter-spacing: 0.06em; display: flex;
    justify-content: space-between; align-items: center; gap: 8px; }
  .col-header.old { color: #c89a9a; }
  .col-header.new { color: #9ad4ff; }
  .stage-pane { position: relative; border-bottom: 1px solid #2a2d35;
    display: flex; align-items: center; justify-content: center; background: #0c0d10;
    min-height: 0; overflow: hidden; }
  .stage-pane:last-child { border-bottom: none; }
  .pane-label { position: absolute; top: 6px; left: 8px; font-size: 11px;
    color: #8a8f99; text-transform: uppercase; letter-spacing: 0.06em; pointer-events: none; }
  .pane-help { position: absolute; top: 6px; right: 8px; font-size: 10px;
    color: #5a5e68; pointer-events: none; }
  .stage-pane:focus { outline: 1px solid #2a4a78; outline-offset: -1px; }
  textarea { width: 100%; box-sizing: border-box; min-height: 100px;
    background: #0c0d10; color: #e6e6e6; border: 1px solid #2a2d35;
    border-radius: 4px; padding: 8px; font: inherit; resize: vertical; }
  button { padding: 8px 12px; border-radius: 4px; border: 1px solid #2a2d35;
    background: #1f2229; color: #e6e6e6; font: inherit; cursor: pointer; }
  button:hover:not(:disabled) { background: #2a4a78; border-color: #4a8fd8; }
  button:disabled { opacity: 0.5; cursor: default; }
  button.primary { background: #2a4a78; border-color: #4a8fd8; }
  button.small { padding: 3px 8px; font-size: 11px; }
  label.row { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #b8bcc4; }
  label.row select { background: #0c0d10; color: #e6e6e6; border: 1px solid #2a2d35;
    border-radius: 3px; padding: 3px 6px; font: inherit; flex: 1; min-width: 0; }
  fieldset { border: 1px solid #2a2d35; border-radius: 4px; padding: 8px;
    display: flex; flex-direction: column; gap: 6px; }
  legend { padding: 0 6px; color: #8a8f99; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.06em; }
  #status { color: #9ad4ff; min-height: 1.4em; white-space: pre-wrap; word-break: break-word; }
  #status.err { color: #ff8080; }
  #status.ok { color: #8bd17c; }
  .stage-pane img { max-width: 100%; max-height: 100%; object-fit: contain; }
  .stage-pane .empty { color: #5a5e68; font-size: 12px; }
  .stage-pane canvas { display: block; }
  .pane-status { position: absolute; bottom: 6px; right: 8px; font-size: 11px;
    color: #5a5e68; pointer-events: none; }
  .pane-status.ok { color: #8bd17c; }
  .pane-status.err { color: #ff8080; }
  .image-url-link { color: #6ac2c2; text-decoration: none; font-size: 10px; }
  .image-url-link:hover { text-decoration: underline; }
  .urls { padding: 6px 12px; background: #16181d; border-top: 1px solid #2a2d35;
    font-size: 10px; color: #5a5e68; word-break: break-all;
    display: flex; flex-direction: column; gap: 4px; }
</style>
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.171.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.171.0/examples/jsm/"
  }
}
</script>
</head>
<body>
<div id="app">
  <div id="panel">
    <fieldset>
      <legend>preset (auto-fills the fields below)</legend>
      <label class="row">
        pick
        <select id="preset"></select>
      </label>
    </fieldset>

    <label style="display:flex;flex-direction:column;gap:4px;">
      <span style="color:#8a8f99;font-size:11px;text-transform:uppercase;letter-spacing:0.06em;">subject phrase</span>
      <textarea id="prompt" placeholder="a weathered cypress log with bleached bark and patches of moss"></textarea>
    </label>

    <fieldset>
      <legend>proxy shape</legend>
      <label class="row">
        proxy_shape
        <select id="opt-proxy">
          <option value="rectangular_prism" selected>rectangular_prism (None)</option>
          <option value="sphere">sphere</option>
          <option value="capsule">capsule</option>
          <option value="hemisphere">hemisphere</option>
        </select>
      </label>
    </fieldset>

    <fieldset>
      <legend>subject dimensions (m) — appended to prompt</legend>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;">
        <label style="display:flex;flex-direction:column;gap:2px;font-size:11px;color:#b8bcc4;">
          width
          <input type="number" id="opt-width" step="0.01" min="0" style="background:#0c0d10;color:#e6e6e6;border:1px solid #2a2d35;border-radius:3px;padding:4px 6px;font:inherit;width:100%;box-sizing:border-box;">
        </label>
        <label style="display:flex;flex-direction:column;gap:2px;font-size:11px;color:#b8bcc4;">
          height
          <input type="number" id="opt-height" step="0.01" min="0" style="background:#0c0d10;color:#e6e6e6;border:1px solid #2a2d35;border-radius:3px;padding:4px 6px;font:inherit;width:100%;box-sizing:border-box;">
        </label>
        <label style="display:flex;flex-direction:column;gap:2px;font-size:11px;color:#b8bcc4;">
          depth
          <input type="number" id="opt-depth" step="0.01" min="0" style="background:#0c0d10;color:#e6e6e6;border:1px solid #2a2d35;border-radius:3px;padding:4px 6px;font:inherit;width:100%;box-sizing:border-box;">
        </label>
      </div>
    </fieldset>

    <button id="btn-compare" type="button" class="primary">Compare nano-banana-2 vs nano-banana-pro image (parallel)</button>
    <button id="btn-compare-3d" type="button" disabled>Compare nano-banana-2 vs nano-banana-pro 3D (parallel)</button>

    <div style="font-size:10px;color:#5a5e68;border:1px solid #2a2d35;border-radius:4px;padding:6px 8px;">
      Production settings (locked, identical to <code>app.services.threed</code>):<br>
      • <b>nano-banana-2</b> (<code>google:4@3</code>): 512×512, thinking=MINIMAL<br>
      • <b>nano-banana-pro</b> (<code>google:4@2</code>): 1024×1024<br>
      • <b>trellis</b>: remesh=off, resolution=512, textureSize=1024
    </div>

    <div id="status"></div>

    <details>
      <summary style="cursor:pointer;color:#8a8f99;font-size:11px;text-transform:uppercase;letter-spacing:0.06em;">wrapped prompts (debug)</summary>
      <div style="margin-top:6px;display:flex;flex-direction:column;gap:8px;">
        <div>
          <div style="color:#c89a9a;font-size:10px;text-transform:uppercase;">nano-banana-2 (left)</div>
          <div id="wrapped-old" style="font-size:10px;color:#8a8f99;background:#0c0d10;border:1px solid #2a2d35;border-radius:3px;padding:6px;white-space:pre-wrap;word-break:break-word;max-height:140px;overflow-y:auto;"></div>
        </div>
        <div>
          <div style="color:#9ad4ff;font-size:10px;text-transform:uppercase;">nano-banana-pro (right)</div>
          <div id="wrapped-new" style="font-size:10px;color:#8a8f99;background:#0c0d10;border:1px solid #2a2d35;border-radius:3px;padding:6px;white-space:pre-wrap;word-break:break-word;max-height:140px;overflow-y:auto;"></div>
        </div>
      </div>
    </details>
  </div>

  <div id="stage">
    <div class="col" data-side="old">
      <div class="col-header old">
        <span>nano-banana-2 (google:4@3, 512×512, thinking=MINIMAL)</span>
      </div>
      <div class="stage-pane" data-pane="image-old">
        <span class="pane-label">image</span>
        <span class="empty">no image yet</span>
      </div>
      <div class="stage-pane" data-pane="model-old">
        <span class="pane-label">3d model</span>
        <span class="pane-help">L-drag orbit · M/R-drag pan · scroll dolly · F frame</span>
        <span class="empty">no model yet</span>
      </div>
    </div>
    <div class="col" data-side="new">
      <div class="col-header new">
        <span>nano-banana-pro (google:4@2, 1024×1024)</span>
      </div>
      <div class="stage-pane" data-pane="image-new">
        <span class="pane-label">image</span>
        <span class="empty">no image yet</span>
      </div>
      <div class="stage-pane" data-pane="model-new">
        <span class="pane-label">3d model</span>
        <span class="pane-help">L-drag orbit · M/R-drag pan · scroll dolly · F frame</span>
        <span class="empty">no model yet</span>
      </div>
    </div>
  </div>
</div>

<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const promptEl = $("#prompt");
const statusEl = $("#status");
const optProxy = $("#opt-proxy");
const optWidth = $("#opt-width");
const optHeight = $("#opt-height");
const optDepth = $("#opt-depth");
const presetEl = $("#preset");
const btnCompare = $("#btn-compare");
const btnCompare3d = $("#btn-compare-3d");
const wrappedOldEl = $("#wrapped-old");
const wrappedNewEl = $("#wrapped-new");

// Production trellis settings — locked to match server/app/services/threed.py
const TRELLIS_PROD_SETTINGS = {
  remesh: false,
  resolution: 512,
  texture_size: 1024,
};

const PROMPT_KEY = "playground.prompt";
const PROXY_KEY = "playground.proxy";

promptEl.value = localStorage.getItem(PROMPT_KEY) ?? "";
promptEl.addEventListener("input", () => localStorage.setItem(PROMPT_KEY, promptEl.value));
optProxy.value = localStorage.getItem(PROXY_KEY) ?? "rectangular_prism";
optProxy.addEventListener("change", () => localStorage.setItem(PROXY_KEY, optProxy.value));

const sideState = {
  old: { imageUrl: null, glbUrl: null },
  new: { imageUrl: null, glbUrl: null },
};

function setStatus(text, cls = "") {
  statusEl.textContent = text;
  statusEl.className = cls;
}

function setPaneStatus(pane, text, cls = "") {
  let badge = pane.querySelector(".pane-status");
  if (!badge) {
    badge = document.createElement("span");
    badge.className = "pane-status";
    pane.appendChild(badge);
  }
  badge.textContent = text;
  badge.className = `pane-status ${cls}`;
}

function clearPaneStatus(pane) {
  const badge = pane.querySelector(".pane-status");
  if (badge) badge.remove();
}

function refreshCompare3dGate() {
  btnCompare3d.disabled = !(sideState.old.imageUrl && sideState.new.imageUrl);
}

function showImage(side, url) {
  sideState[side].imageUrl = url;
  const pane = $(`[data-pane="image-${side}"]`);
  pane.innerHTML = '<span class="pane-label">image</span>';
  const img = document.createElement("img");
  img.src = url;
  pane.appendChild(img);
  const link = document.createElement("a");
  link.className = "image-url-link pane-status";
  link.style.right = "8px";
  link.style.bottom = "6px";
  link.style.position = "absolute";
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = "open ↗";
  pane.appendChild(link);
  refreshCompare3dGate();
}

function showImageEmpty(side, text) {
  sideState[side].imageUrl = null;
  const pane = $(`[data-pane="image-${side}"]`);
  pane.innerHTML = '<span class="pane-label">image</span>';
  const span = document.createElement("span");
  span.className = "empty";
  span.textContent = text;
  pane.appendChild(span);
  refreshCompare3dGate();
}

// ---- presets ---------------------------------------------------------------

let presets = [];

async function loadPresets() {
  const r = await fetch("/presets");
  if (!r.ok) { setStatus("preset load failed", "err"); return; }
  const data = await r.json();
  presets = data.presets;
  presetEl.innerHTML = "";
  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = "— pick a preset —";
  presetEl.appendChild(blank);
  for (const p of presets) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = p.label;
    presetEl.appendChild(opt);
  }
}

const DIMS_KEY = "playground.dims";

function persistDims() {
  localStorage.setItem(DIMS_KEY, JSON.stringify({
    w: optWidth.value, h: optHeight.value, d: optDepth.value,
  }));
}

(() => {
  const saved = JSON.parse(localStorage.getItem(DIMS_KEY) ?? "{}");
  if (saved.w !== undefined) optWidth.value = saved.w;
  if (saved.h !== undefined) optHeight.value = saved.h;
  if (saved.d !== undefined) optDepth.value = saved.d;
})();
for (const el of [optWidth, optHeight, optDepth]) {
  el.addEventListener("input", persistDims);
}

function applyPreset(p) {
  promptEl.value = p.subject;
  optProxy.value = p.proxy_shape;
  if (Array.isArray(p.dimensions) && p.dimensions.length === 3) {
    optWidth.value = p.dimensions[0];
    optHeight.value = p.dimensions[1];
    optDepth.value = p.dimensions[2];
    persistDims();
  }
  localStorage.setItem(PROMPT_KEY, p.subject);
  localStorage.setItem(PROXY_KEY, p.proxy_shape);
  setStatus(`loaded preset: ${p.label}`, "ok");
}

presetEl.addEventListener("change", () => {
  const p = presets.find(p => p.id === presetEl.value);
  if (p) applyPreset(p);
});

loadPresets();

// ---- compare ---------------------------------------------------------------

btnCompare.addEventListener("click", async () => {
  const prompt = promptEl.value.trim();
  if (!prompt) { setStatus("subject is empty", "err"); return; }
  btnCompare.disabled = true;
  for (const side of ["old", "new"]) {
    showImageEmpty(side, "generating…");
    setPaneStatus($(`[data-pane="image-${side}"]`), "generating…");
  }
  setStatus("generating nano-banana-2 and nano-banana-pro in parallel…");
  wrappedOldEl.textContent = "";
  wrappedNewEl.textContent = "";
  const t0 = performance.now();
  const proxy = optProxy.value;
  const w = parseFloat(optWidth.value);
  const h = parseFloat(optHeight.value);
  const d = parseFloat(optDepth.value);
  if (!Number.isFinite(w) || !Number.isFinite(h) || !Number.isFinite(d) || w <= 0 || h <= 0 || d <= 0) {
    setStatus("dimensions must be positive numbers (W, H, D)", "err");
    btnCompare.disabled = false;
    for (const side of ["old", "new"]) showImageEmpty(side, "no image yet");
    return;
  }
  // Both columns use the production wrapper with the same dimensions —
  // only the model + per-model production settings differ.
  const sharedBody = { prompt, proxy_shape: proxy, width: w, height: h, depth: d };
  const bodyBySide = {
    old: { ...sharedBody, model: "nano-banana-2" },
    new: { ...sharedBody, model: "nano-banana-pro" },
  };
  const tasks = ["old", "new"].map(side =>
    fetch("/banana", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(bodyBySide[side]),
    }).then(async r => {
      if (!r.ok) throw new Error(`${side}: ${r.status}: ${await r.text()}`);
      return [side, await r.json()];
    })
  );
  const results = await Promise.allSettled(tasks);
  for (const res of results) {
    if (res.status === "fulfilled") {
      const [side, data] = res.value;
      showImage(side, data.image_url);
      clearPaneStatus($(`[data-pane="image-${side}"]`));
      (side === "old" ? wrappedOldEl : wrappedNewEl).textContent = data.wrapped_prompt;
    } else {
      const msg = String(res.reason?.message ?? res.reason);
      const side = msg.startsWith("old:") ? "old" : "new";
      showImageEmpty(side, "failed");
      setPaneStatus($(`[data-pane="image-${side}"]`), msg, "err");
    }
  }
  const dt = ((performance.now() - t0) / 1000).toFixed(1);
  const failed = results.filter(r => r.status === "rejected").length;
  setStatus(failed ? `done (${dt}s) — ${failed} failed` : `done (${dt}s)`, failed ? "err" : "ok");
  btnCompare.disabled = false;
});

// ---- parallel trellis ------------------------------------------------------

btnCompare3d.addEventListener("click", async () => {
  if (!sideState.old.imageUrl || !sideState.new.imageUrl) {
    setStatus("need images on both sides first", "err");
    return;
  }
  btnCompare3d.disabled = true;
  for (const side of ["old", "new"]) {
    const pane = $(`[data-pane="model-${side}"]`);
    setPaneStatus(pane, "generating 3D…");
  }
  setStatus("generating both 3D meshes in parallel…");
  const t0 = performance.now();
  const tasks = ["old", "new"].map(side =>
    fetch("/trellis", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        image_url: sideState[side].imageUrl,
        ...TRELLIS_PROD_SETTINGS,
      }),
    }).then(async r => {
      if (!r.ok) throw new Error(`${side}: ${r.status}: ${await r.text()}`);
      const data = await r.json();
      return [side, data];
    })
  );
  const results = await Promise.allSettled(tasks);
  for (const res of results) {
    if (res.status === "fulfilled") {
      const [side, data] = res.value;
      sideState[side].glbUrl = data.glb_url;
      clearPaneStatus($(`[data-pane="model-${side}"]`));
      loadModel(side, data.glb_url);
    } else {
      const msg = String(res.reason?.message ?? res.reason);
      const side = msg.startsWith("old:") ? "old" : "new";
      setPaneStatus($(`[data-pane="model-${side}"]`), msg, "err");
    }
  }
  const dt = ((performance.now() - t0) / 1000).toFixed(1);
  const failed = results.filter(r => r.status === "rejected").length;
  setStatus(failed ? `3D done (${dt}s) — ${failed} failed` : `3D done (${dt}s)`, failed ? "err" : "ok");
  btnCompare3d.disabled = false;
});

// ---- 3D viewer (one per side) ---------------------------------------------

const viewers = {};

function ensureViewer(side) {
  if (viewers[side]) return viewers[side];
  const pane = $(`[data-pane="model-${side}"]`);
  pane.innerHTML = '<span class="pane-label">3d model</span>';
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  pane.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0c0d10);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x202028, 0.9));
  const dir = new THREE.DirectionalLight(0xffffff, 0.7);
  dir.position.set(3, 5, 4);
  scene.add(dir);
  scene.add(new THREE.AxesHelper(0.5));
  const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 100);
  camera.position.set(2, 1.5, 2.5);
  const controls = new OrbitControls(camera, renderer.domElement);
  // Unity-Scene-view-feel: pan parallel to screen at constant on-screen
  // speed; scroll-zoom slides the orbit pivot toward the cursor, so
  // rotation always pivots around what you're actually looking at — no
  // more "world swings around because I zoomed in past the centre".
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.screenSpacePanning = true;
  controls.zoomToCursor = true;
  controls.rotateSpeed = 0.9;
  controls.panSpeed = 1.0;
  controls.zoomSpeed = 1.1;
  // Mouse mapping: left = orbit, middle = pan, right = pan (Unity Q tool).
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.PAN,
    RIGHT: THREE.MOUSE.PAN,
  };
  // Suppress the browser's right-click menu so right-drag pan works.
  renderer.domElement.addEventListener("contextmenu", (e) => e.preventDefault());

  const fitSize = () => {
    const w = pane.clientWidth, h = pane.clientHeight;
    if (w === 0 || h === 0) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  fitSize();
  window.addEventListener("resize", fitSize);

  const v = { renderer, scene, camera, controls, pane, model: null, fitSize, frame: null };
  // F key reframes the model when this pane is hovered.
  pane.tabIndex = 0;
  pane.addEventListener("keydown", (e) => {
    if (e.key === "f" || e.key === "F") {
      if (v.frame) v.frame();
    }
  });
  pane.addEventListener("pointerenter", () => pane.focus({ preventScroll: true }));

  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
  viewers[side] = v;
  return v;
}

function loadModel(side, url) {
  const v = ensureViewer(side);
  v.fitSize();
  if (v.model) {
    v.scene.remove(v.model);
    v.model = null;
  }
  const loader = new GLTFLoader();
  loader.load(
    url,
    (gltf) => {
      const root = gltf.scene;
      const box = new THREE.Box3().setFromObject(root);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      root.position.sub(center);
      v.scene.add(root);
      v.model = root;
      const maxDim = Math.max(size.x, size.y, size.z) || 1;
      v.frame = () => {
        const dist = maxDim * 2.4;
        v.camera.position.set(dist * 0.7, dist * 0.5, dist * 0.9);
        v.camera.near = maxDim * 0.01;
        v.camera.far = maxDim * 50;
        v.camera.updateProjectionMatrix();
        v.controls.target.set(0, 0, 0);
        v.controls.update();
      };
      v.frame();
    },
    undefined,
    (err) => setStatus(`GLB load failed (${side}): ${err.message ?? err}`, "err"),
  );
}
</script>
</body>
</html>
"""
