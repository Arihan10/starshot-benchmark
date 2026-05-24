"""Standalone Nano Banana -> Trellis 2 orientation-fidelity test client.

The page generates ten orthographic three-quarter source images of varied
objects in one consistent default orientation, then sends each image to
Trellis 2 and shows the image beside the returned GLB. The GLB viewer
recenters the model for framing only; it does not apply any corrective
rotation.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple

import httpx
import trimesh
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
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
from app.services.threed import (
    NANO_BANANA_2,
    NANO_BANANA_PRO,
    banana_settings_for,
)

load_dotenv()

TRELLIS_MODEL = os.environ.get("TRELLIS_MODEL", "microsoft:trellis-2@4b")
ORIENTATION_RUNS_DIR = Path("./runs/orientation-test")
TEST_OBJECTS_PATH = ORIENTATION_RUNS_DIR / "test_objects.json"
_RUN_ID_PATTERN = r"^[A-Za-z0-9_.-]{1,80}$"

BananaModelName = Literal["nano-banana-2", "nano-banana-pro"]
ViewName = Literal["quarter", "front"]
_VIEWS: tuple[ViewName, ...] = ("quarter", "front")

_BANANA_MODELS: dict[BananaModelName, str] = {
    "nano-banana-2": NANO_BANANA_2,
    "nano-banana-pro": NANO_BANANA_PRO,
}

_HITBOX_TERMS: dict[ProxyShape | None, tuple[str, str]] = {
    None: ("rectangular prism", "rectangular prism"),
    ProxyShape.SPHERE: ("ellipsoid", "sphere"),
    ProxyShape.CAPSULE: ("vertical capsule", "capsule"),
    ProxyShape.HEMISPHERE: ("upright hemisphere", "hemisphere"),
}


class ObjectCase(NamedTuple):
    id: str
    label: str
    prompt: str
    proxy_shape: ProxyShape | None
    dimensions: tuple[float, float, float]


_DEFAULT_CASES: tuple[ObjectCase, ...] = (
    ObjectCase(
        "arrow_rover",
        "arrow rover",
        "an asymmetric orientation calibration rover shaped like a wedge arrow, "
        "with a bright red pointed front nose, a yellow flat rear tailgate, a "
        "blue material panel on the object's left side, a green material panel "
        "on the object's right side, one tall black antenna mounted only on the "
        "front-right corner, and two small rear wheels larger than the front wheel",
        None,
        (2.4, 1.2, 1.6),
    ),
    ObjectCase(
        "delivery_van",
        "delivery van",
        "a compact white delivery van with a red front grille and windshield, "
        "a large green sliding cargo door on the right side, a yellow rear "
        "cargo door, black roof rails, and mismatched side mirrors",
        None,
        (4.8, 2.4, 2.0),
    ),
    ObjectCase(
        "warehouse_forklift",
        "forklift",
        "an orange warehouse forklift with black twin lifting forks projecting "
        "from the front, a protective overhead cage, a visible right-side "
        "propane tank, small front wheels, larger rear steering wheels, and a "
        "single amber beacon on the roof",
        None,
        (2.6, 2.2, 1.4),
    ),
    ObjectCase(
        "patrol_boat",
        "patrol boat",
        "a small navy patrol boat with a sharp red bow at the front, white cabin "
        "windows, green starboard-side stripe, yellow stern transom, roof radar "
        "mast, and twin outboard motors mounted only at the rear",
        None,
        (5.5, 2.0, 1.8),
    ),
    ObjectCase(
        "film_camera",
        "film camera",
        "a vintage black film camera body with a large circular glass lens on "
        "the front, a raised silver viewfinder on top-left, a red shutter button "
        "on the top-right, leather texture panels, and a hinged yellow film door "
        "on the back",
        None,
        (1.2, 0.8, 0.7),
    ),
    ObjectCase(
        "robot_dog",
        "robot dog",
        "a quadruped robot dog with a red sensor head at the front, yellow battery "
        "pack at the rear, blue panels on its left legs, green panels on its right "
        "legs, exposed black joints, and an asymmetric antenna on the front-right shoulder",
        None,
        (1.4, 0.9, 0.5),
    ),
    ObjectCase(
        "racing_motorcycle",
        "motorcycle",
        "a racing motorcycle with a red pointed front fairing and headlight, "
        "green right-side exhaust pipe, yellow rear tail cowl, black wheels, "
        "low handlebars, and a visible kickstand only on the left side",
        None,
        (2.1, 1.2, 0.65),
    ),
    ObjectCase(
        "espresso_machine",
        "espresso machine",
        "a chrome commercial espresso machine with a red front control panel, "
        "two protruding black portafilter handles on the front, a green steam "
        "wand on the right side, yellow rear service panel, cup warmer rails on top, "
        "and four short metal feet",
        None,
        (1.1, 0.8, 0.7),
    ),
    ObjectCase(
        "treasure_chest",
        "treasure chest",
        "an old wooden treasure chest with a curved lid, large red front lock plate, "
        "yellow rear hinge band, green right-side iron handle, blue left-side iron "
        "handle, brass corner protectors, and one front corner visibly chipped",
        None,
        (1.6, 1.0, 1.0),
    ),
    ObjectCase(
        "telescope_mount",
        "telescope",
        "a brass astronomical telescope on a tripod, with the large glass objective "
        "lens at the red-marked front, yellow eyepiece at the rear, green focus knob "
        "on the right side, blue finder scope offset on the left-top, and three uneven "
        "tripod legs",
        None,
        (1.8, 1.5, 0.8),
    ),
)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:56].strip("_") or "object"


def _load_cases_from_file(path: Path) -> tuple[ObjectCase, ...] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    cases: list[ObjectCase] = []
    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path} item {i} must be an object")
        prompt = item.get("prompt")
        bbox = item.get("bbox_m")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path} item {i} has invalid prompt")
        if not isinstance(bbox, list) or len(bbox) != 3:
            raise ValueError(f"{path} item {i} has invalid bbox_m")
        dimensions = (float(bbox[0]), float(bbox[1]), float(bbox[2]))
        case_id = f"{i:03d}_{_slugify(prompt)}"
        cases.append(ObjectCase(case_id, f"{i:03d}", prompt.strip(), None, dimensions))
    return tuple(cases)


_CASES = _load_cases_from_file(TEST_OBJECTS_PATH) or _DEFAULT_CASES


class OrientationBananaRequest(BaseModel):
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    model: BananaModelName = "nano-banana-pro"
    case_id: str
    view: ViewName = "quarter"


class TrellisRequest(BaseModel):
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    case_id: str
    view: ViewName = "quarter"
    image_url: str = Field(min_length=1)


class OrientationResultRequest(BaseModel):
    run_id: str = Field(pattern=_RUN_ID_PATTERN)
    case_id: str
    view: ViewName = "quarter"
    glb_bbox: tuple[float, float, float]


_client: Runware | None = None
_client_lock = asyncio.Lock()
_results_lock = asyncio.Lock()


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


def _case_by_id(case_id: str) -> ObjectCase:
    for case in _CASES:
        if case.id == case_id:
            return case
    raise HTTPException(status_code=400, detail=f"unknown orientation case: {case_id!r}")


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _case_proxy_name(case: ObjectCase) -> str:
    return (
        case.proxy_shape.value
        if case.proxy_shape is not None
        else "rectangular_prism"
    )


def _run_dir(run_id: str) -> Path:
    return ORIENTATION_RUNS_DIR / run_id


def _case_dir(run_id: str, case_id: str) -> Path:
    _case_by_id(case_id)
    path = _run_dir(run_id) / case_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _view_dir(run_id: str, case_id: str, view: ViewName) -> Path:
    path = _case_dir(run_id, case_id) / view
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_url(path: Path) -> str:
    return f"/orientation-artifacts/{path.relative_to(ORIENTATION_RUNS_DIR).as_posix()}"


def _results_url(run_id: str) -> str:
    return _artifact_url(_run_dir(run_id) / "results.json")


def _ext_from_url(url: str, *, default: str) -> str:
    lower = url.lower().split("?", 1)[0]
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".glb"):
        if lower.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return default


async def _download_to_path(url: str, path: Path) -> None:
    async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as http:
        resp = await http.get(url)
        resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)


def _merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _view_result_seed(case: ObjectCase) -> dict[str, Any]:
    return {view: {} for view in _VIEWS}


async def _update_results(
    run_id: str,
    case: ObjectCase,
    patch: dict[str, Any],
) -> dict[str, Any]:
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "results.json"
    async with _results_lock:
        if path.exists():
            loaded = json.loads(path.read_text())
            data: dict[str, Any] = loaded if isinstance(loaded, dict) else {}
        else:
            data = {"run_id": run_id, "created_at": _timestamp(), "cases": {}}
        data["updated_at"] = _timestamp()
        cases_obj = data.get("cases")
        cases: dict[str, Any] = cases_obj if isinstance(cases_obj, dict) else {}
        data["cases"] = cases
        current_obj = cases.get(case.id)
        current: dict[str, Any]
        if isinstance(current_obj, dict):
            current = current_obj
        else:
            current = {
                "id": case.id,
                "label": case.label,
                "prompt": case.prompt,
                "proxy_shape": _case_proxy_name(case),
                "target_bbox_m": list(case.dimensions),
                "views": _view_result_seed(case),
            }
            cases[case.id] = current
        if not isinstance(current.get("views"), dict):
            current["views"] = _view_result_seed(case)
        _merge_dict(current, patch)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        return data


def _compare_bboxes(
    reference_bbox: tuple[float, float, float],
    candidate_bbox: tuple[float, float, float],
    *,
    reference_label: str = "target",
    candidate_label: str = "candidate",
) -> dict[str, Any]:
    dimension_order = ("shortest", "middle", "longest")
    reference_sorted = tuple(sorted(reference_bbox))
    candidate_sorted = tuple(sorted(candidate_bbox))
    reference_shortest = reference_sorted[0]
    candidate_shortest = candidate_sorted[0]
    if reference_shortest <= 0 or candidate_shortest <= 0:
        raise HTTPException(status_code=400, detail="bbox dimensions must be positive")
    scale = reference_shortest / candidate_shortest
    scaled_candidate_sorted = tuple(v * scale for v in candidate_sorted)
    reference_ratios = tuple(v / reference_shortest for v in reference_sorted)
    candidate_ratios = tuple(v / candidate_shortest for v in candidate_sorted)
    ratio_delta = tuple(candidate_ratios[i] - reference_ratios[i] for i in range(3))
    percent_delta = tuple(
        (ratio_delta[i] / reference_ratios[i]) * 100
        if reference_ratios[i] else 0
        for i in range(3)
    )
    delta = tuple(scaled_candidate_sorted[i] - reference_sorted[i] for i in range(3))
    abs_delta = tuple(abs(v) for v in delta)
    matched_dimension_percent_delta = {
        dimension_order[i]: percent_delta[i]
        for i in range(1, 3)
    }
    return {
        "comparison_basis": "sorted_internal_proportions",
        "dimension_order": list(dimension_order),
        "axis_order": ["width_x", "height_y", "depth_z"],
        "reference_label": reference_label,
        "candidate_label": candidate_label,
        "reference_bbox": list(reference_bbox),
        "candidate_bbox": list(candidate_bbox),
        "reference_sorted_bbox": list(reference_sorted),
        "candidate_sorted_bbox": list(candidate_sorted),
        "matched_edge_alignment": {
            "matched_dimension": "shortest",
            "matched_dimension_index": 0,
            "reference_edge": reference_shortest,
            "candidate_edge": candidate_shortest,
            "scale_candidate_to_reference": scale,
        },
        "scaled_candidate_sorted_bbox": list(scaled_candidate_sorted),
        "reference_internal_ratios": list(reference_ratios),
        "candidate_internal_ratios": list(candidate_ratios),
        "ratio_delta_candidate_minus_reference": list(ratio_delta),
        "delta_scaled_candidate_minus_reference": list(delta),
        "absolute_delta": list(abs_delta),
        "percent_delta": list(percent_delta),
        "matched_dimension_percent_delta": matched_dimension_percent_delta,
    }


def measure_glb_bbox(path: Path) -> tuple[float, float, float]:
    loaded = trimesh.load(path, force="scene")
    bounds = getattr(loaded, "bounds", None)
    if bounds is None:
        raise RuntimeError(f"GLB has no bounds: {path}")
    return (
        float(bounds[1][0] - bounds[0][0]),
        float(bounds[1][1] - bounds[0][1]),
        float(bounds[1][2] - bounds[0][2]),
    )


def _comparison_from_case_views(case_data: dict[str, Any]) -> dict[str, Any] | None:
    views = case_data.get("views")
    if not isinstance(views, dict):
        return None
    quarter = views.get("quarter")
    front = views.get("front")
    if not isinstance(quarter, dict) or not isinstance(front, dict):
        return None
    quarter_bbox = quarter.get("glb_bbox_raw")
    front_bbox = front.get("glb_bbox_raw")
    if not isinstance(quarter_bbox, list) or not isinstance(front_bbox, list):
        return None
    if len(quarter_bbox) != 3 or len(front_bbox) != 3:
        return None
    quarter_tuple = (float(quarter_bbox[0]), float(quarter_bbox[1]), float(quarter_bbox[2]))
    front_tuple = (float(front_bbox[0]), float(front_bbox[1]), float(front_bbox[2]))
    return _compare_bboxes(
        quarter_tuple,
        front_tuple,
        reference_label="quarter_glb",
        candidate_label="front_glb",
    )


def _wrap_orthographic_prompt(
    description: str,
    proxy_shape: ProxyShape | None,
    dimensions: tuple[float, float, float],
    case: ObjectCase,
) -> str:
    hitbox, silhouette = _HITBOX_TERMS[proxy_shape]
    width, height, depth = dimensions
    return (
        "Generate a clean orthographic three-quarter product render of "
        f"{description} that roughly can be captured within a {hitbox} "
        "hitbox without bending or deforming the object's natural proportions. "
        f"The object should not fully be in a {silhouette} shape unless its "
        "dimensions and nature dictate it is naturally that shape. Prioritize "
        "realism over confinement to the hitbox shape. "
        "This is an orientation-fidelity source image for a 3D reconstruction "
        "test, so the object's heading must be unambiguous. Use one consistent "
        "default orientation for every object: an orthographic front-right "
        "three-quarter view with parallel projection, about 35 degrees above "
        "the object, so the front, right side, and top are all visible. The "
        "object's front must face toward the lower-right of the image every "
        "time, with the rear pointing toward the upper-left. Do not rotate the "
        "object to a different heading, do not use a side view, and do not mirror "
        "left/right markings. "
        f"The object's dimensions are exactly {width:.2f}m by {height:.2f}m by "
        f"{depth:.2f}m (width by height by depth). Capture the entire model in "
        "the image. Render against a clean, empty white background with no "
        "other objects, dimension markings, labels, axes, or graphics."
    )


def _wrap_prompt_for_view(case: ObjectCase, view: ViewName) -> str:
    if view == "quarter":
        return _wrap_orthographic_prompt(
            case.prompt,
            case.proxy_shape,
            case.dimensions,
            case,
        )
    return wrap_image_prompt(
        case.prompt,
        case.proxy_shape,
        case.dimensions,
        view="front",
    )


async def _await_runware_result(client: Runware, ack: Any, task_uuid: str) -> list[Any]:
    results = (
        await client.getResponse(taskUUID=task_uuid, numberResults=1)
        if isinstance(ack, IAsyncTaskResponse)
        else ack
    )
    if not results:
        raise RuntimeError(f"empty result list for task {task_uuid}")
    return list(results)


def _unwrap_trellis_url(mesh: Any) -> str:
    outputs = getattr(mesh, "outputs", None)
    files = getattr(outputs, "files", None) if outputs else None
    if not files:
        raise RuntimeError(f"trellis result missing files: {mesh!r}")
    first = files[0]
    url = first.get("url") if isinstance(first, dict) else getattr(first, "url", None)
    if not url:
        raise RuntimeError(f"trellis result missing url: {first!r}")
    return str(url)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await _disconnect_client()

    app = FastAPI(
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    ORIENTATION_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/orientation-artifacts",
        StaticFiles(directory=ORIENTATION_RUNS_DIR),
        name="orientation-artifacts",
    )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:  # pyright: ignore[reportUnusedFunction]
        return _PAGE

    @app.get("/orientation/cases")
    async def orientation_cases() -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        return {
            "cases": [
                {
                    "id": case.id,
                    "label": case.label,
                    "prompt": case.prompt,
                    "proxy_shape": _case_proxy_name(case),
                    "dimensions": list(case.dimensions),
                }
                for case in _CASES
            ],
            "models": {
                "nano-banana-2": NANO_BANANA_2,
                "nano-banana-pro": NANO_BANANA_PRO,
                "trellis": TRELLIS_MODEL,
            },
        }

    @app.post("/orientation/banana")
    async def orientation_banana(req: OrientationBananaRequest) -> dict[str, object]:  # pyright: ignore[reportUnusedFunction]
        case = _case_by_id(req.case_id)
        final_prompt = _wrap_prompt_for_view(case, req.view)
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
            image = (await _await_runware_result(client, ack, task_uuid))[0]
        except Exception as e:
            raise HTTPException(502, f"{type(e).__name__}: {e}") from e
        image_url = getattr(image, "imageURL", None)
        if not image_url:
            raise HTTPException(502, f"banana result missing imageURL: {image!r}")
        image_ext = _ext_from_url(str(image_url), default=".png")
        image_path = _view_dir(req.run_id, case.id, req.view) / f"{case.id}_{req.view}{image_ext}"
        try:
            await _download_to_path(str(image_url), image_path)
        except Exception as e:
            raise HTTPException(
                502, f"image download failed: {type(e).__name__}: {e}"
            ) from e
        saved_image_url = _artifact_url(image_path)
        await _update_results(
            req.run_id,
            case,
            {
                "views": {
                    req.view: {
                        "image": {
                            "remote_url": str(image_url),
                            "saved_path": str(image_path),
                            "saved_url": saved_image_url,
                            "wrapped_prompt": final_prompt,
                            "model_id": model_id,
                            "settings": settings_dict,
                        }
                    }
                }
            },
        )
        return {
            "case_id": case.id,
            "view": req.view,
            "image_url": str(image_url),
            "saved_image_url": saved_image_url,
            "saved_image_path": str(image_path),
            "wrapped_prompt": final_prompt,
            "model_id": model_id,
            "settings": settings_dict,
            "results_url": _results_url(req.run_id),
        }

    @app.post("/orientation/trellis")
    async def orientation_trellis(req: TrellisRequest) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        case = _case_by_id(req.case_id)
        try:
            client = await _get_client()
            task_uuid = str(uuid.uuid4())
            request = I3dInference(
                taskUUID=task_uuid,
                model=TRELLIS_MODEL,
                inputs=I3dInputs(image=req.image_url),
                settings=ISettings(remesh=False, resolution=512, textureSize=1024),
                outputFormat="GLB",
                outputType="URL",
                deliveryMethod="async",
                numberResults=1,
            )
            ack = await client.inference3d(request3d=request)
            mesh = (await _await_runware_result(client, ack, task_uuid))[0]
            glb_url = _unwrap_trellis_url(mesh)
        except Exception as e:
            raise HTTPException(502, f"{type(e).__name__}: {e}") from e
        glb_path = _view_dir(req.run_id, case.id, req.view) / f"{case.id}_{req.view}.glb"
        try:
            await _download_to_path(glb_url, glb_path)
        except Exception as e:
            raise HTTPException(
                502, f"GLB download failed: {type(e).__name__}: {e}"
            ) from e
        saved_glb_url = _artifact_url(glb_path)
        await _update_results(
            req.run_id,
            case,
            {
                "views": {
                    req.view: {
                        "glb": {
                            "remote_url": glb_url,
                            "saved_path": str(glb_path),
                            "saved_url": saved_glb_url,
                            "model_id": TRELLIS_MODEL,
                            "settings": {
                                "remesh": False,
                                "resolution": 512,
                                "textureSize": 1024,
                            },
                        }
                    }
                }
            },
        )
        return {
            "case_id": case.id,
            "view": req.view,
            "glb_url": glb_url,
            "saved_glb_url": saved_glb_url,
            "saved_glb_path": str(glb_path),
            "results_url": _results_url(req.run_id),
        }

    @app.post("/orientation/result")
    async def orientation_result(req: OrientationResultRequest) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        case = _case_by_id(req.case_id)
        target_comparison = _compare_bboxes(
            case.dimensions,
            req.glb_bbox,
            reference_label="target_bbox_m",
            candidate_label=f"{req.view}_glb_raw",
        )
        data = await _update_results(
            req.run_id,
            case,
            {
                "views": {
                    req.view: {
                        "glb_bbox_raw": list(req.glb_bbox),
                        "comparison_to_target": target_comparison,
                    }
                }
            },
        )
        case_data = data.get("cases", {}).get(case.id, {})
        view_comparison = (
            _comparison_from_case_views(case_data)
            if isinstance(case_data, dict)
            else None
        )
        if view_comparison is not None:
            await _update_results(
                req.run_id,
                case,
                {"view_dimension_comparison": view_comparison},
            )
        return {
            "case_id": case.id,
            "view": req.view,
            "comparison": target_comparison,
            "view_dimension_comparison": view_comparison,
            "results_url": _results_url(req.run_id),
        }

    return app


app = create_app()


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Trellis orientation fidelity test</title>
<style>
  html, body {
    margin: 0;
    height: 100%;
    background: #101114;
    color: #e6e6e6;
    font: 13px ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  #app {
    display: grid;
    grid-template-columns: 360px 1fr;
    height: 100%;
  }
  #panel {
    padding: 12px;
    overflow-y: auto;
    border-right: 1px solid #2a2d35;
    background: #16181d;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  #grid {
    overflow-y: auto;
    padding: 12px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
    gap: 12px;
    align-content: start;
  }
  .card {
    min-height: 420px;
    border: 1px solid #2a2d35;
    border-radius: 6px;
    background: #16181d;
    display: grid;
    grid-template-rows: auto 1fr auto;
    overflow: hidden;
  }
  .card-header {
    padding: 8px 10px;
    border-bottom: 1px solid #2a2d35;
    display: flex;
    justify-content: space-between;
    gap: 10px;
    color: #9ad4ff;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 11px;
  }
  .card-body {
    display: grid;
    grid-template-rows: 1fr 1fr;
    min-height: 320px;
  }
  .variant {
    display: grid;
    grid-template-rows: auto 1fr;
    min-height: 0;
  }
  .variant + .variant {
    border-top: 1px solid #2a2d35;
  }
  .variant-title {
    padding: 4px 8px;
    background: #101114;
    color: #8a8f99;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .variant-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    min-height: 0;
  }
  .pane {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 0;
    min-height: 0;
    background: #0c0d10;
    overflow: hidden;
  }
  .pane + .pane {
    border-left: 1px solid #2a2d35;
  }
  .pane img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
  }
  .pane canvas {
    display: block;
  }
  .pane-label {
    position: absolute;
    top: 6px;
    left: 8px;
    color: #8a8f99;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    pointer-events: none;
    z-index: 2;
  }
  .pane-help {
    position: absolute;
    top: 6px;
    right: 8px;
    color: #5a5e68;
    font-size: 10px;
    pointer-events: none;
    z-index: 2;
  }
  .empty {
    color: #5a5e68;
    font-size: 12px;
  }
  .case-status {
    color: #8a8f99;
  }
  .case-status.ok {
    color: #8bd17c;
  }
  .case-status.err {
    color: #ff8080;
  }
  .prompt {
    max-height: 70px;
    overflow: auto;
    padding: 8px 10px;
    color: #6f7682;
    font-size: 10px;
    line-height: 1.35;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .meta {
    border-top: 1px solid #2a2d35;
    padding: 8px 10px 0;
    color: #8a8f99;
    font-size: 10px;
    line-height: 1.45;
  }
  .model-dims {
    color: #e0c271;
  }
  .comparison {
    color: #8bd17c;
  }
  .view-comparison {
    color: #9ad4ff;
  }
  .model-dims.pending,
  .comparison.pending,
  .view-comparison.pending {
    color: #5a5e68;
  }
  label.stack {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  label.stack span,
  legend {
    color: #8a8f99;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  textarea, input, select {
    box-sizing: border-box;
    width: 100%;
    background: #0c0d10;
    color: #e6e6e6;
    border: 1px solid #2a2d35;
    border-radius: 4px;
    padding: 7px 8px;
    font: inherit;
  }
  textarea {
    min-height: 150px;
    resize: vertical;
  }
  fieldset {
    border: 1px solid #2a2d35;
    border-radius: 4px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  button {
    padding: 8px 12px;
    border-radius: 4px;
    border: 1px solid #2a2d35;
    background: #1f2229;
    color: #e6e6e6;
    font: inherit;
    cursor: pointer;
  }
  button.primary {
    background: #2a4a78;
    border-color: #4a8fd8;
  }
  button:hover:not(:disabled) {
    background: #2a4a78;
    border-color: #4a8fd8;
  }
  button:disabled {
    opacity: 0.5;
    cursor: default;
  }
  #status {
    min-height: 1.4em;
    color: #9ad4ff;
    white-space: pre-wrap;
    word-break: break-word;
  }
  #status.ok {
    color: #8bd17c;
  }
  #status.err {
    color: #ff8080;
  }
  a.open {
    position: absolute;
    right: 8px;
    bottom: 6px;
    color: #6ac2c2;
    font-size: 10px;
    text-decoration: none;
    z-index: 2;
  }
  a.open:hover {
    text-decoration: underline;
  }
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
      <legend>generation settings</legend>
      <label class="stack">
        <span>nano banana model</span>
        <select id="model">
          <option value="nano-banana-pro" selected>nano-banana-pro (google:4@2, 1024x1024)</option>
          <option value="nano-banana-2">nano-banana-2 (google:4@3, 512x512, thinking=MINIMAL)</option>
        </select>
      </label>
      <label class="stack">
        <span>parallel cases</span>
        <input id="concurrency" type="number" min="1" max="4" step="1" value="2">
      </label>
    </fieldset>
    <button id="run" class="primary" type="button">Generate 10 prompts x 2 views</button>
    <button id="clear" type="button">Clear cards</button>
    <div id="status">loading cases...</div>
    <div style="font-size:10px;color:#5a5e68;line-height:1.45;border:1px solid #2a2d35;border-radius:4px;padding:8px;">
      Each prompt generates both the current orthographic front-right three-quarter view and the production-style direct orthographic front view. Model panes load Trellis GLBs in their returned orientation and only recenter each model so the camera can frame it.
    </div>
  </div>
  <div id="grid"></div>
</div>

<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (sel, root = document) => root.querySelector(sel);

const modelEl = $("#model");
const concurrencyEl = $("#concurrency");
const runEl = $("#run");
const clearEl = $("#clear");
const statusEl = $("#status");
const gridEl = $("#grid");

const STORAGE_KEY = "orientation-test.settings";
const VIEWS = ["quarter", "front"];
const VIEW_LABELS = {
  quarter: "3/4 view",
  front: "front flat-face view",
};
const viewers = new Map();
let cases = [];
let currentRunId = null;

function setStatus(text, cls = "") {
  statusEl.textContent = text;
  statusEl.className = cls;
}

function savedSettings() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "{}");
  } catch {
    return {};
  }
}

function persistSettings() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({
    model: modelEl.value,
    concurrency: concurrencyEl.value,
  }));
}

for (const el of [modelEl, concurrencyEl]) {
  el.addEventListener("input", persistSettings);
  el.addEventListener("change", persistSettings);
}

async function loadCases() {
  const res = await fetch("/orientation/cases");
  if (!res.ok) throw new Error(`case load failed: ${res.status}`);
  const data = await res.json();
  cases = data.cases;
  const saved = savedSettings();
  modelEl.value = saved.model ?? "nano-banana-pro";
  concurrencyEl.value = saved.concurrency ?? "2";
  renderCards();
  persistSettings();
  setStatus("ready");
}

function renderCards() {
  disposeViewers();
  gridEl.innerHTML = "";
  viewers.clear();
  for (const c of cases) {
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.caseId = c.id;
    card.dataset.originalPrompt = c.prompt;
    card.innerHTML = `
      <div class="card-header">
        <span>${c.id} / ${c.label}</span>
        <span class="case-status">pending</span>
      </div>
      <div class="card-body">
        <div class="variant" data-view="quarter">
          <div class="variant-title">3/4 view</div>
          <div class="variant-grid">
            <div class="pane" data-pane="image-quarter">
              <span class="pane-label">source image</span>
              <span class="empty">no image yet</span>
            </div>
            <div class="pane" data-pane="model-quarter">
              <span class="pane-label">trellis GLB</span>
              <span class="pane-help">unrotated GLB / F frames</span>
              <span class="empty">no model yet</span>
            </div>
          </div>
        </div>
        <div class="variant" data-view="front">
          <div class="variant-title">front flat-face view</div>
          <div class="variant-grid">
            <div class="pane" data-pane="image-front">
              <span class="pane-label">source image</span>
              <span class="empty">no image yet</span>
            </div>
            <div class="pane" data-pane="model-front">
              <span class="pane-label">trellis GLB</span>
              <span class="pane-help">unrotated GLB / F frames</span>
              <span class="empty">no model yet</span>
            </div>
          </div>
        </div>
      </div>
      <div class="meta">
        <div>target prompt bbox: ${formatDims(c.dimensions)} m / proxy: ${c.proxy_shape}</div>
        <div class="model-dims pending" data-view="quarter">3/4 GLB bbox: pending</div>
        <div class="comparison pending" data-view="quarter">3/4 vs target: pending</div>
        <div class="model-dims pending" data-view="front">front GLB bbox: pending</div>
        <div class="comparison pending" data-view="front">front vs target: pending</div>
        <div class="view-comparison pending">front vs 3/4 dimension ratio: pending</div>
      </div>
      <div class="prompt">${c.prompt}

3/4 wrapped prompt: pending

front wrapped prompt: pending</div>
    `;
    gridEl.appendChild(card);
  }
}

function formatDims(values) {
  return values.map((v) => Number(v).toFixed(3)).join(" x ");
}

function formatSignedDims(values) {
  return values.map((v) => {
    const n = Number(v);
    return `${n >= 0 ? "+" : ""}${n.toFixed(3)}`;
  }).join(" x ");
}

function makeRunId() {
  const stamp = new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-");
  return `${stamp}_${Math.random().toString(36).slice(2, 8)}`;
}

function disposeViewers() {
  for (const v of viewers.values()) {
    v.renderer.setAnimationLoop(null);
    v.renderer.dispose();
    v.model?.traverse?.((n) => {
      if (n.isMesh) {
        n.geometry?.dispose?.();
        const mats = Array.isArray(n.material) ? n.material : n.material ? [n.material] : [];
        for (const m of mats) m.dispose?.();
      }
    });
  }
}

function cardFor(caseId) {
  return gridEl.querySelector(`[data-case-id="${CSS.escape(caseId)}"]`);
}

function setCaseStatus(caseId, text, cls = "") {
  const card = cardFor(caseId);
  if (!card) return;
  const el = $(".case-status", card);
  el.textContent = text;
  el.className = `case-status ${cls}`;
}

function showImage(caseId, view, url) {
  const card = cardFor(caseId);
  if (!card) return;
  const pane = $(`[data-pane="image-${view}"]`, card);
  pane.innerHTML = `<span class="pane-label">source image</span>`;
  const img = document.createElement("img");
  img.src = url;
  pane.appendChild(img);
  const link = document.createElement("a");
  link.className = "open";
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = "open";
  pane.appendChild(link);
}

function showWrappedPrompt(caseId, view, prompt) {
  const card = cardFor(caseId);
  if (!card) return;
  const otherView = view === "quarter" ? "front" : "quarter";
  const promptEl = $(".prompt", card);
  const current = promptEl.dataset[otherView] ?? "pending";
  promptEl.dataset[view] = prompt;
  promptEl.textContent = `${card.dataset.originalPrompt}

3/4 wrapped prompt: ${view === "quarter" ? prompt : current}

front wrapped prompt: ${view === "front" ? prompt : current}`;
}

function showModelDimensions(caseId, view, size) {
  const card = cardFor(caseId);
  if (!card) return;
  const el = $(`.model-dims[data-view="${view}"]`, card);
  el.textContent = `${VIEW_LABELS[view]} GLB bbox: ${formatDims([size.x, size.y, size.z])} raw units`;
  el.classList.remove("pending");
}

function showTargetComparison(caseId, view, comparison) {
  const card = cardFor(caseId);
  if (!card) return;
  const el = $(`.comparison[data-view="${view}"]`, card);
  const align = comparison.matched_edge_alignment;
  const pct = Object.entries(comparison.matched_dimension_percent_delta)
    .map(([dim, value]) => `${dim} ${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(3)}%`)
    .join(" / ");
  el.textContent = `${VIEW_LABELS[view]} vs target: sorted ratios, match ${align.matched_dimension}, scale ${align.scale_candidate_to_reference.toFixed(6)} / ratio delta ${pct}`;
  el.classList.remove("pending");
}

function showViewComparison(caseId, comparison) {
  if (!comparison) return;
  const card = cardFor(caseId);
  if (!card) return;
  const el = $(".view-comparison", card);
  const align = comparison.matched_edge_alignment;
  const pct = Object.entries(comparison.matched_dimension_percent_delta)
    .map(([dim, value]) => `${dim} ${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(3)}%`)
    .join(" / ");
  el.textContent = `front vs 3/4 dimension ratio: sorted ratios, match ${align.matched_dimension}, scale ${align.scale_candidate_to_reference.toFixed(6)} / ratio delta ${pct}`;
  el.classList.remove("pending");
}

function viewerKey(caseId, view) {
  return `${caseId}:${view}`;
}

function ensureViewer(caseId, view) {
  const key = viewerKey(caseId, view);
  if (viewers.has(key)) return viewers.get(key);
  const card = cardFor(caseId);
  const pane = $(`[data-pane="model-${view}"]`, card);
  pane.innerHTML = '<span class="pane-label">trellis GLB</span><span class="pane-help">unrotated GLB / F frames</span>';
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  pane.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0c0d10);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x202028, 0.9));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(3, 5, 4);
  scene.add(dir);
  scene.add(new THREE.AxesHelper(0.5));

  const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 100);
  camera.position.set(2, 1.4, 2.2);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.screenSpacePanning = true;
  controls.zoomToCursor = true;
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.PAN,
    RIGHT: THREE.MOUSE.PAN,
  };
  renderer.domElement.addEventListener("contextmenu", (e) => e.preventDefault());

  const fitSize = () => {
    const w = pane.clientWidth;
    const h = pane.clientHeight;
    if (w === 0 || h === 0) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  fitSize();
  window.addEventListener("resize", fitSize);

  const viewer = { renderer, scene, camera, controls, pane, model: null, fitSize, frame: null };
  pane.tabIndex = 0;
  pane.addEventListener("pointerenter", () => pane.focus({ preventScroll: true }));
  pane.addEventListener("keydown", (e) => {
    if ((e.key === "f" || e.key === "F") && viewer.frame) viewer.frame();
  });
  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
  viewers.set(key, viewer);
  return viewer;
}

function loadModel(caseId, view, url) {
  const v = ensureViewer(caseId, view);
  v.fitSize();
  if (v.model) {
    v.scene.remove(v.model);
    v.model = null;
  }
  const loader = new GLTFLoader();
  return new Promise((resolve, reject) => {
    loader.load(
      url,
      (gltf) => {
      const root = gltf.scene;
      root.traverse((child) => {
        if (child.isMesh && child.material) {
          const mats = Array.isArray(child.material) ? child.material : [child.material];
          for (const m of mats) m.side = THREE.DoubleSide;
        }
      });
      const box = new THREE.Box3().setFromObject(root);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      showModelDimensions(caseId, view, size);
      root.position.sub(center);
      v.scene.add(root);
      v.model = root;
      const maxDim = Math.max(size.x, size.y, size.z) || 1;
      v.frame = () => {
        const dist = maxDim * 2.4;
        v.camera.position.set(dist * 0.75, dist * 0.55, dist * 0.95);
        v.camera.near = maxDim * 0.01;
        v.camera.far = maxDim * 50;
        v.camera.updateProjectionMatrix();
        v.controls.target.set(0, 0, 0);
        v.controls.update();
      };
      v.frame();
      resolve(size);
      },
      undefined,
      (err) => {
        setCaseStatus(caseId, `GLB load failed: ${err.message ?? err}`, "err");
        reject(err);
      },
    );
  });
}

function inputPayload(caseId, view) {
  if (!currentRunId) throw new Error("missing run id");
  return {
    run_id: currentRunId,
    model: modelEl.value,
    case_id: caseId,
    view,
  };
}

async function runView(c, view) {
  setCaseStatus(c.id, `${view} banana...`);
  const bananaRes = await fetch("/orientation/banana", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(inputPayload(c.id, view)),
  });
  if (!bananaRes.ok) throw new Error(`${c.id} ${view} banana: ${bananaRes.status}: ${await bananaRes.text()}`);
  const banana = await bananaRes.json();
  showImage(c.id, view, banana.saved_image_url ?? banana.image_url);
  showWrappedPrompt(c.id, view, banana.wrapped_prompt);

  setCaseStatus(c.id, `${view} trellis...`);
  const trellisRes = await fetch("/orientation/trellis", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      run_id: currentRunId,
      case_id: c.id,
      view,
      image_url: banana.image_url,
    }),
  });
  if (!trellisRes.ok) throw new Error(`${c.id} ${view} trellis: ${trellisRes.status}: ${await trellisRes.text()}`);
  const trellis = await trellisRes.json();
  const size = await loadModel(c.id, view, trellis.saved_glb_url ?? trellis.glb_url);
  setCaseStatus(c.id, `${view} comparing...`);
  const resultRes = await fetch("/orientation/result", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      run_id: currentRunId,
      case_id: c.id,
      view,
      glb_bbox: [size.x, size.y, size.z],
    }),
  });
  if (!resultRes.ok) throw new Error(`${c.id} ${view} result: ${resultRes.status}: ${await resultRes.text()}`);
  const result = await resultRes.json();
  showTargetComparison(c.id, view, result.comparison);
  showViewComparison(c.id, result.view_dimension_comparison);
}

async function runCase(c) {
  for (const view of VIEWS) {
    await runView(c, view);
  }
  setCaseStatus(c.id, "done", "ok");
}

async function runWithLimit(items, limit, worker) {
  let next = 0;
  const workerCount = Math.max(1, Math.min(limit, items.length));
  const workers = Array.from({ length: workerCount }, async () => {
    while (next < items.length) {
      const item = items[next++];
      try {
        await worker(item);
      } catch (e) {
        setCaseStatus(item.id, e.message ?? String(e), "err");
      }
    }
  });
  await Promise.all(workers);
}

runEl.addEventListener("click", async () => {
  runEl.disabled = true;
  persistSettings();
  currentRunId = makeRunId();
  renderCards();
  const t0 = performance.now();
  try {
    const limit = Math.max(1, Math.min(4, parseInt(concurrencyEl.value || "2", 10)));
    setStatus(`running ${cases.length} cases with concurrency ${limit}...\nsaving to server/runs/orientation-test/${currentRunId}/`);
    await runWithLimit(cases, limit, runCase);
    const failed = Array.from(gridEl.querySelectorAll(".case-status.err")).length;
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    const resultsUrl = `/orientation-artifacts/${currentRunId}/results.json`;
    setStatus(
      failed
        ? `done in ${dt}s; ${failed} failed\nresults: ${resultsUrl}`
        : `done in ${dt}s\nresults: ${resultsUrl}`,
      failed ? "err" : "ok",
    );
  } catch (e) {
    setStatus(e.message ?? String(e), "err");
  } finally {
    runEl.disabled = false;
  }
});

clearEl.addEventListener("click", () => {
  renderCards();
  setStatus("cleared");
});

loadCases().catch((e) => setStatus(e.message ?? String(e), "err"));
</script>
</body>
</html>
"""
