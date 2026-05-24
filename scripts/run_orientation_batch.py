#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Run all orientation-test objects through Nano Banana -> Trellis.

Loads `server/runs/orientation-test/test_objects.json` when present, otherwise
uses the fallback objects in `app.orientation_test`. Saves artifacts and
`results.json` under `server/runs/orientation-test/<run_id>/`.
"""
# pyright: reportPrivateUsage=false, reportMissingParameterType=false, reportArgumentType=false

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from runware import I3dInference, I3dInputs, IImageInference, ISettings

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = REPO_ROOT / "server"
sys.path.insert(0, str(SERVER_DIR))

from app.orientation_test import (  # noqa: E402
    _CASES,
    NANO_BANANA_PRO,
    TRELLIS_MODEL,
    _await_runware_result,
    _case_proxy_name,
    _compare_bboxes,
    _comparison_from_case_views,
    _download_to_path,
    _ext_from_url,
    _get_client,
    _update_results,
    _view_dir,
    _wrap_prompt_for_view,
    banana_settings_for,
    measure_glb_bbox,
)

VIEWS = ("quarter", "front")


def _run_id() -> str:
    stamp = datetime.now(UTC).isoformat(timespec="milliseconds")
    return stamp.replace(":", "-").replace(".", "-").replace("+00-00", "Z")


async def _banana(
    *,
    run_id: str,
    case,
    view: str,
    model: str,
) -> str:
    final_prompt = _wrap_prompt_for_view(case, view)
    settings_dict = banana_settings_for(model)
    gen_settings = (
        ISettings(thinking=settings_dict["thinking"])
        if "thinking" in settings_dict
        else None
    )
    client = await _get_client()
    task_uuid = str(uuid.uuid4())
    request = IImageInference(
        taskUUID=task_uuid,
        model=model,
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
    image_url = getattr(image, "imageURL", None)
    if not image_url:
        raise RuntimeError(f"banana result missing imageURL: {image!r}")

    image_ext = _ext_from_url(str(image_url), default=".png")
    image_path = _view_dir(run_id, case.id, view) / f"{case.id}_{view}{image_ext}"
    await _download_to_path(str(image_url), image_path)
    await _update_results(
        run_id,
        case,
        {
            "views": {
                view: {
                    "image": {
                        "remote_url": str(image_url),
                        "saved_path": str(image_path),
                        "saved_url": str(image_path),
                        "wrapped_prompt": final_prompt,
                        "model_id": model,
                        "settings": settings_dict,
                    }
                }
            }
        },
    )
    return str(image_url)


async def _trellis(
    *,
    run_id: str,
    case,
    view: str,
    image_url: str,
) -> tuple[float, float, float]:
    client = await _get_client()
    task_uuid = str(uuid.uuid4())
    request = I3dInference(
        taskUUID=task_uuid,
        model=TRELLIS_MODEL,
        inputs=I3dInputs(image=image_url),
        settings=ISettings(remesh=False, resolution=512, textureSize=1024),
        outputFormat="GLB",
        outputType="URL",
        deliveryMethod="async",
        numberResults=1,
    )
    ack = await client.inference3d(request3d=request)
    mesh = (await _await_runware_result(client, ack, task_uuid))[0]
    outputs = getattr(mesh, "outputs", None)
    files = getattr(outputs, "files", None) if outputs else None
    if not files:
        raise RuntimeError(f"trellis result missing files: {mesh!r}")
    first = files[0]
    glb_url = first.get("url") if isinstance(first, dict) else getattr(first, "url", None)
    if not glb_url:
        raise RuntimeError(f"trellis result missing url: {first!r}")

    glb_path = _view_dir(run_id, case.id, view) / f"{case.id}_{view}.glb"
    await _download_to_path(str(glb_url), glb_path)
    bbox = measure_glb_bbox(glb_path)
    comparison = _compare_bboxes(
        case.dimensions,
        bbox,
        reference_label="target_bbox_m",
        candidate_label=f"{view}_glb_raw",
    )
    data = await _update_results(
        run_id,
        case,
        {
            "proxy_shape": _case_proxy_name(case),
            "views": {
                view: {
                    "glb": {
                        "remote_url": str(glb_url),
                        "saved_path": str(glb_path),
                        "saved_url": str(glb_path),
                        "model_id": TRELLIS_MODEL,
                        "settings": {
                            "remesh": False,
                            "resolution": 512,
                            "textureSize": 1024,
                        },
                    },
                    "glb_bbox_raw": list(bbox),
                    "comparison_to_target": comparison,
                }
            },
        },
    )
    case_data = data.get("cases", {}).get(case.id, {})
    if isinstance(case_data, dict):
        view_comparison = _comparison_from_case_views(case_data)
        if view_comparison is not None:
            await _update_results(
                run_id,
                case,
                {"view_dimension_comparison": view_comparison},
            )
    return bbox


async def _run_case(run_id: str, case, model: str) -> None:
    print(f"[{case.id}] start", flush=True)
    for view in VIEWS:
        print(f"[{case.id}] {view} banana", flush=True)
        image_url = await _banana(run_id=run_id, case=case, view=view, model=model)
        print(f"[{case.id}] {view} trellis", flush=True)
        bbox = await _trellis(run_id=run_id, case=case, view=view, image_url=image_url)
        print(f"[{case.id}] {view} bbox={bbox}", flush=True)
    print(f"[{case.id}] done", flush=True)


async def _run_all(run_id: str, model: str, concurrency: int) -> None:
    queue = asyncio.Queue()
    for case in _CASES:
        queue.put_nowait(case)

    async def worker() -> None:
        while not queue.empty():
            case = await queue.get()
            try:
                await _run_case(run_id, case, model)
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await queue.join()
    for worker_task in workers:
        worker_task.cancel()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=_run_id())
    parser.add_argument("--model", default=NANO_BANANA_PRO)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()

    print(f"[orientation-batch] cases={len(_CASES)} run_id={args.run_id}", flush=True)
    asyncio.run(_run_all(args.run_id, args.model, max(1, args.concurrency)))
    print(
        f"[orientation-batch] results: server/runs/orientation-test/{args.run_id}/results.json",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
