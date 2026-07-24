"""Modal app for the GPU half of the splat pipeline (stages 4-7).

Deploy with `modal deploy splat/modal_app.py` from the repo root; drive it with
`splat/modal_sync.py` (CLI or from the server). The split:

  * LOCAL (Mac): stages 1-3 — scene manifest, free-space grid, surfel cloud.
  * MODAL (one A100-40GB container): stage 4 (camera plan — zone-driven
    single-shot field, pure CPU; lives here only to keep the stage chain in
    one place), stage 5 (reference renders — headless Chromium against the
    loopback host in `splat/modal_capture.py`), stage 6 (gsplat 2DGS
    fine-tune → RAW `trained.ply`), stage 7 (delete + heal + final-prune →
    delivered `healed.ply` + LOD ladder, then `splat/quantize.py` →
    `healed.sqz`).

DATA MODEL — one `modal.Volume` ("starshot-splat-cells"):

    objects/{sha[:2]}/{sha}.glb                 content-addressed GLB store (the
                                                mesh tier; deduped across cells,
                                                re-runs and hardlinked branches)
    cells/{run}/{slot}/{model}/inputs/          tier.json ({node_id: sha}),
                                                freespace.npz,
                                                freespace.npz.skin.npy.zst,
                                                cloud.ply.zst, scene.json,
                                                manifest.json
    cells/{run}/{slot}/{model}/splat/           cameras.json, refs/, ckpt/,
                                                trained.ply (RAW, stage 6),
                                                healed.ply (+ LODs, stage 7),
                                                healed.sqz, status.json

Inputs are compressed by the CLIENT where it pays (cloud.ply ~1.6-2x, skin
bitmasks ~3-5x; GLBs/npz are already entropy-coded and upload as-is) and are
hashed UNCOMPRESSED so signatures are transport-independent.

IDEMPOTENT / RESUMABLE: `run_cell` skips any stage whose recorded input
signature (content hashes + params) matches and whose artifacts exist —
re-running with new train params re-uses the refs; a preempted container
resumes stage 5 from the frames on the Volume and stage 6 from its checkpoint
(written straight to the Volume). Live progress streams through a
`modal.Dict` heartbeat ("starshot-splat-status", keyed `{run}/{slot}/{model}`).

PYTHON PIN: the image runs Python 3.10 because the pinned `gsplat 1.5.3`
prebuilt wheel (pt24cu124) is cp310-only — no CUDA toolchain in the image, no
JIT compile on first import. The `splat/` package is 3.10-compatible today; if
it adopts 3.12-only syntax, switch to a `nvidia/cuda:*-devel` base and build
gsplat from source with TORCH_CUDA_ARCH_LIST=8.0.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "starshot-splat"
VOLUME_NAME = "starshot-splat-cells"
STATUS_DICT_NAME = "starshot-splat-status"
VOL = "/vol"
_SCRATCH = Path("/tmp/cells")
_ASSETS = "/assets"                      # capture page + three.js (baked below)

_REPO = Path(__file__).resolve().parent.parent

app = modal.App(APP_NAME)
# Volumes v2 (Beta): no file-count limit and faster commits/reloads — the right
# fit for our many-small-files workload (per-cell refs are thousands of SZF
# frames, and the CAS accumulates one file per unique mesh), and its "not for
# mission-critical data" caveat doesn't bite because everything here is
# regenerable from the local stages 1-3.
volume = modal.Volume.from_name(VOLUME_NAME, version=2, create_if_missing=True)
status_dict = modal.Dict.from_name(STATUS_DICT_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    # Renderer for stage 5: Debian's real chromium package (not a snap), the
    # Vulkan loader + glvnd GL/EGL loaders so Chrome's ANGLE/Vulkan backend can
    # reach the NVIDIA device for HARDWARE WebGL, and vulkan-tools for the
    # `vulkaninfo` diagnostic. SwiftShader ships inside chromium as the fallback.
    .apt_install(
        "chromium", "libvulkan1", "vulkan-tools",
        "libglvnd0", "libgl1", "libegl1", "libgles2", "fonts-liberation",
    )
    # Register the NVIDIA Vulkan ICD + EGL vendor. `NVIDIA_DRIVER_CAPABILITIES=all`
    # makes Modal MOUNT the graphics driver libs (libEGL_nvidia / libGLX_nvidia /
    # libnvidia-glcore …, confirmed by probe_gpu_stack), but the driver does NOT
    # ship the loader registration JSONs — so the Vulkan/EGL loaders never see the
    # A100 and Chrome falls back to Mesa llvmpipe (software), which then crash-loops
    # Skia. We write the JSONs ourselves, pointing at libEGL_nvidia.so.0 (the ICD
    # to use headless, where X11 libs are absent — per the NVIDIA driver README).
    # The library soname resolves at runtime against the mounted driver.
    .run_commands(
        "mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d",
        "printf '%s' '{\"file_format_version\":\"1.0.0\",\"ICD\":"
        "{\"library_path\":\"libEGL_nvidia.so.0\",\"api_version\":\"1.3.277\"}}' "
        "> /usr/share/vulkan/icd.d/nvidia_icd.json",
        "printf '%s' '{\"file_format_version\":\"1.0.0\",\"ICD\":"
        "{\"library_path\":\"libEGL_nvidia.so.0\"}}' "
        "> /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    )
    .env({
        # Expose the GRAPHICS driver components (GL/EGL/Vulkan), not just compute
        # — the prerequisite for headless-Chrome hardware WebGL. Superset of
        # compute, so CUDA/training is unaffected.
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "NVIDIA_VISIBLE_DEVICES": "all",
        # Point the loaders at ONLY the NVIDIA ICD/vendor we wrote, so Vulkan
        # enumerates the A100 (not Mesa llvmpipe). SwiftShader is unaffected (it's
        # Chrome's own, not a system Vulkan ICD), so the CPU fallback still works.
        "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
        "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
        "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        # TEMP (debugging the stage-5 stall): stream Chromium's console + logs
        # into `modal app logs`. Remove once stage 5 renders cleanly.
        "STARSHOT_CAPTURE_DEBUG": "1",
    })
    # torch pinned to the cu124 build matching the gsplat wheel tag below.
    .pip_install(
        "torch==2.4.1+cu124",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # ALL non-torch runtime deps + gsplat's own deps (numpy / jaxtyping / rich /
    # ninja / typing_extensions) from PyPI, installed BEFORE gsplat. gsplat's
    # wheel index hosts ONLY the gsplat wheel, so if gsplat had to resolve numpy
    # itself against that index the build fails ("No matching distribution for
    # numpy"); pre-satisfying its deps means the gsplat step only fetches gsplat.
    .pip_install(
        "numpy>=2.0,<3",
        "scipy>=1.11",
        "trimesh>=4.5",
        "pillow>=11.0",
        "zstandard>=0.23",
        "aiohttp>=3.10",
        "ninja",
        "jaxtyping",
        "rich>=13.9",
        "typing_extensions",
    )
    # The prebuilt gsplat wheel for torch 2.4 / CUDA 12.4 (cp310). extra_index_url
    # (NOT index_url) keeps PyPI as a fallback for any dependency; the
    # +pt24cu124 local-version wheel is served only by the gsplat index.
    .pip_install(
        "gsplat==1.5.3+pt24cu124",
        extra_index_url="https://docs.gsplat.studio/whl",
    )
    # Capture page assets, served by the loopback host: the page + workers
    # verbatim, and the SAME three.js the debug viewer runs (importmap paths
    # /vendor/three/... map here).
    .add_local_file(
        _REPO / "client/public/splatcapture.html", f"{_ASSETS}/splatcapture.html"
    )
    .add_local_file(
        _REPO / "client/public/js/splatcapture.js", f"{_ASSETS}/js/splatcapture.js"
    )
    # splatcapture.js imports the shared weighted-blended OIT engine (glass); the
    # loopback host serves it from /js, so it MUST be baked in or the page's ES
    # module graph 404s and no frames render.
    .add_local_file(
        _REPO / "client/public/js/oit.js", f"{_ASSETS}/js/oit.js"
    )
    .add_local_file(
        _REPO / "client/public/js/splatlight.js", f"{_ASSETS}/js/splatlight.js"
    )
    .add_local_file(
        _REPO / "client/public/js/splatcapture-worker.js",
        f"{_ASSETS}/js/splatcapture-worker.js",
    )
    .add_local_dir(_REPO / "client/node_modules/three", f"{_ASSETS}/vendor/three")
    .add_local_python_source("splat")
)


# --- small helpers (in-container) ----------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _sig(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _src_sig(*paths: Path) -> str:
    """A hash of the deployed SOURCE that implements a stage, folded into that
    stage's cache signature. Without this the idempotency key is only inputs +
    params, so editing the pipeline (planner / renderer / trainer) and
    redeploying would silently REUSE artifacts the old code produced — a changed
    stage would never re-run. Missing files are skipped (best effort)."""
    h = hashlib.sha256()
    for p in paths:
        with contextlib.suppress(Exception):
            h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def _zstd_decompress(src: Path, dst: Path) -> None:
    import zstandard

    with src.open("rb") as fi, dst.open("wb") as fo:
        zstandard.ZstdDecompressor().copy_stream(fi, fo)


def _ensure_input(inputs: Path, name: str, sha: str, scratch: Path) -> Path:
    """Materialize one input file into local scratch, decompressing `.zst`
    transport when present. Keyed by content hash, so a warm container reuses
    its copy and a changed upload replaces it."""
    out = scratch / name
    marker = scratch / f"{name}.sha"
    if out.is_file() and marker.is_file() and marker.read_text() == sha:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    zst = inputs / f"{name}.zst"
    if zst.is_file():
        _zstd_decompress(zst, out)
    else:
        shutil.copyfile(inputs / name, out)
    marker.write_text(sha)
    return out


def _sync_dir(src: Path, dst: Path) -> int:
    """Copy files missing (or size-changed) from `src` into `dst`, recursively.
    Returns the number of files copied. Both sides keep their extras."""
    copied = 0
    if not src.is_dir():
        return copied
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        q = dst / p.relative_to(src)
        if q.is_file() and q.stat().st_size == p.stat().st_size:
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, q)
        copied += 1
    return copied


class _Heartbeat:
    """Throttled live-progress publisher: stage progress callbacks → the status
    Dict (for pollers) + container stdout (for `modal app logs`). With
    `commit=True` (Volume-resident long stages: training writes checkpoints
    beside its out_path) it also commits the Volume every `_COMMIT_EVERY_S`,
    bounding what a preemption can lose to the last few minutes."""

    _COMMIT_EVERY_S = 300.0

    def __init__(self, key: str) -> None:
        self.key = key
        self._last = 0.0
        self._last_commit = time.monotonic()

    def stage(self, stage: str, commit: bool = False):
        def cb(done: int, total: int, msg: str) -> None:
            now = time.monotonic()
            if commit and now - self._last_commit > self._COMMIT_EVERY_S:
                self._last_commit = now
                with contextlib.suppress(Exception):
                    volume.commit()
            if now - self._last < 2.0 and done not in (0, total):
                return
            self._last = now
            entry = {
                "stage": stage,
                "done": int(done),
                "total": int(total),
                "msg": str(msg)[:300],
                "t": time.time(),
            }
            # Best-effort: a heartbeat hiccup must never fail the run.
            with contextlib.suppress(Exception):
                status_dict[self.key] = entry
            # Cell-prefixed so lines from concurrent containers stay attributable
            # in the aggregated `modal app logs` stream.
            print(f"[{self.key}] [{stage}] {done}/{total} {msg}", flush=True)

        return cb


def _load_status(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"stages": {}}


def _save_status(path: Path, status: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=1), encoding="utf-8")
    tmp.replace(path)


def _fresh(status: dict[str, Any], stage: str, sig: str, artifacts: list[Path]) -> bool:
    rec = status["stages"].get(stage)
    return (
        rec is not None
        and rec.get("sig") == sig
        and all(p.exists() for p in artifacts)
    )


def _record(
    status: dict[str, Any], status_path: Path, stage: str, sig: str, summary: Any
) -> None:
    status["stages"][stage] = {
        "sig": sig,
        "summary": summary,
        "done_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save_status(status_path, status)
    volume.commit()


# --- the pipeline function -------------------------------------------------------


@app.function(
    image=image,
    gpu="A100-40GB",
    # COST NOTE: Modal bills CPU + memory on max(reservation, actual use) for the
    # WHOLE container lifetime, ON TOP of the GPU. This run is GPU-bound — stage 6
    # training dominates the wall time and uses only ~1-2 cores — so we RESERVE
    # little and let the CPU-heavy stage-5 SZF encode BURST (the soft CPU limit is
    # reservation + 16 cores; memory can grow past the reservation when the worker
    # has room, still billed for actual use). Reserving 16 cores + 50 GiB instead
    # would pay for them idle through the whole train (~$3.25/hr vs ~$2.4/hr here,
    # against the A100-40GB floor of $2.10/hr).
    cpu=4.0,
    memory=8192,            # 8 GiB guaranteed floor; bursts higher on demand, billed for use
    timeout=12 * 3600,
    volumes={VOL: volume},
    # Preemptible (the default, 1x price — non-preemptible is 3x). A preemption
    # just resumes from the Volume checkpoint, so cheap + safe; `retries` re-runs.
    retries=modal.Retries(max_retries=2, initial_delay=10.0),
)
def run_cell(spec: dict[str, Any]) -> dict[str, Any]:
    """Run stages 4-7 for one cell against the Volume, skipping stages whose
    recorded input signature matches (idempotent; `force` re-runs everything in
    the window).

    spec = {
      run, slot, model: str,
      stages: [4,5,6,7] (any contiguous or sparse subset),
      input_sha: {freespace, skin, cloud, scene, tier: str},  # uncompressed
      plan / train / quant: {} param overrides (dataclass field names),
      force: bool,
    }
    """
    from splat import colmap as splat_colmap, modal_capture, quantize
    from splat.stage4 import PlanParams, plan_cameras
    from splat.stage6 import TrainParams, heal_splat, train_splat

    run, slot, model = spec["run"], spec["slot"], spec["model"]
    stages = set(spec.get("stages") or [4, 5, 6, 7])
    force = bool(spec.get("force"))
    key = f"{run}/{slot}/{model}"
    heart = _Heartbeat(key)

    # Identify this run up front so concurrent containers are distinguishable in
    # the aggregated `modal app logs` stream (every heartbeat line is also cell-
    # prefixed — see _Heartbeat). Includes the param overrides so runs that
    # differ only by plan/train config are told apart too.
    print(
        f"=== run_cell START {key} | stages={sorted(stages)} force={force} "
        f"| plan={spec.get('plan') or {}} train={spec.get('train') or {}} "
        f"quant={spec.get('quant') or {}} ===",
        flush=True,
    )

    volume.reload()  # a warm container must see files the client just pushed
    cell = Path(VOL) / "cells" / run / slot / model
    inputs = cell / "inputs"
    out = cell / "splat"
    out.mkdir(parents=True, exist_ok=True)
    scratch = _SCRATCH / run / slot / model
    scratch.mkdir(parents=True, exist_ok=True)

    in_sha: dict[str, str] = spec["input_sha"]
    tier: dict[str, str] = json.loads((inputs / "tier.json").read_text("utf-8"))
    status_path = out / "status.json"
    status = _load_status(status_path)
    status["spec"] = {"run": run, "slot": slot, "model": model,
                      "stages": sorted(stages)}
    summary: dict[str, Any] = {"cell": key, "stages_run": [], "stages_skipped": []}

    # Heavy inputs land in local scratch (fast mmap/reads); artifacts publish to
    # the Volume at stage boundaries with a commit each.
    freespace = _ensure_input(inputs, "freespace.npz", in_sha["freespace"], scratch)
    _ensure_input(inputs, "freespace.npz.skin.npy", in_sha["skin"], scratch)
    cloud = _ensure_input(inputs, "cloud.ply", in_sha["cloud"], scratch)  # stage-6 init
    scene = _ensure_input(inputs, "scene.json", in_sha["scene"], scratch)

    # Per-stage CODE versions (see _src_sig): fold the DEPLOYED pipeline source
    # into each stage's cache signature, so editing a stage + redeploying re-runs
    # THAT stage (and, by output cascade, the stages after it) instead of
    # silently reusing an artifact the old code produced. This is what makes a
    # changed planner / renderer / trainer actually take effect on the next run.
    import splat.stage4 as _stage4
    import splat.stage5 as _stage5
    import splat.stage6 as _stage6

    _js = Path(_ASSETS) / "js"
    code4 = _src_sig(Path(_stage4.__file__))
    code5 = _src_sig(
        Path(_stage5.__file__), Path(modal_capture.__file__),
        _js / "splatcapture.js", _js / "splatcapture-worker.js",
        # splatlight.js defines the bake rig + the (now matte, view-independent)
        # material prep, so it too determines every captured pixel — fold it in
        # so a change there re-renders stage 5 instead of silently reusing frames.
        _js / "splatlight.js",
    )
    # Stage 6 now trains from the COLMAP export, so its code sig folds in colmap.py
    # too — editing the exporter re-runs stage 6 rather than reusing a stale model.
    code6 = _src_sig(Path(_stage6.__file__), Path(splat_colmap.__file__))
    # (stage 7 = heal + quantize folds its own source hash inline — both stage6.py
    # and quantize.py — so no standalone code7 here.)

    # ---- stage 4: camera plan (object-shell single-shot field; CPU) -------------
    plan_params = PlanParams(**(spec.get("plan") or {}))
    sig4 = _sig({"in": [in_sha["freespace"], in_sha["skin"], in_sha["scene"]],
                 "params": plan_params.as_summary(), "code": code4})
    cameras = out / "cameras.json"
    art4 = [cameras]
    if 4 in stages:
        if not force and _fresh(status, "4", sig4, art4):
            summary["stages_skipped"].append(4)
        else:
            s4 = plan_cameras(
                run=run, slot=slot, model=model,
                freespace_path=freespace,
                scene_path=scene,
                out_path=scratch / "cameras.json",
                params=plan_params,
                progress=heart.stage("plan"),
            )
            shutil.copyfile(scratch / "cameras.json", cameras)
            _record(status, status_path, "4", sig4, s4)
            summary["stages_run"].append(4)
            summary["plan"] = s4

    # ---- stage 5: reference renders (loopback capture) -------------------------
    transforms = out / "refs" / "transforms.json"
    if 5 in stages:
        if not cameras.is_file():
            raise FileNotFoundError("cameras.json missing — run stage 4 first")
        cam_sha = _sha256(cameras)
        sig5 = _sig({"cameras": cam_sha, "tier": in_sha["tier"], "code": code5})
        if not force and _fresh(status, "5", sig5, [transforms]):
            summary["stages_skipped"].append(5)
        else:
            refs_scratch = scratch / "refs"
            # RESUME vs REPLAN. Frames are keyed by POSITIONAL view id
            # (cam00042), so frames rendered under a DIFFERENT camera plan
            # collide with the new plan's ids — blindly "resuming" across a replan
            # would skip re-rendering every colliding view and hand Stage 6 the
            # OLD plan's pixels against the NEW plan's poses (silently corrupt
            # supervision). Resume is therefore gated on a plan marker
            # (`refs/plan.sha` = the cameras.json hash): only a preempted render of
            # THIS EXACT plan syncs its prior frames back; anything else clears
            # both copies and renders fresh. This also fixes the invisible stall
            # the unconditional sync caused after a replan — pulling the entire
            # stale reference set (many GB) from the Volume with NO heartbeat, so
            # it looked like stage 4 was hung "writing cameras.json".
            marker = out / "refs" / "plan.sha"
            same_plan = False
            with contextlib.suppress(Exception):
                same_plan = marker.read_text() == cam_sha
            if force or not same_plan:
                heart.stage("refs")(
                    0, 0,
                    "fresh render — dropping any stale frames" if not same_plan and not force
                    else "forced re-render — dropping frames",
                )
                shutil.rmtree(refs_scratch, ignore_errors=True)
                shutil.rmtree(out / "refs", ignore_errors=True)
            else:
                # Same plan, resuming a preempted render: syncing the prior frames
                # from the Volume can take minutes — emit a heartbeat so this step
                # is VISIBLE (not mistaken for a hung earlier stage).
                heart.stage("refs")(0, 0, "resuming — syncing prior frames from volume")
                _sync_dir(out / "refs", refs_scratch)
            # Stamp the plan marker on both copies before rendering; the capture's
            # periodic sample commits carry the Volume copy, so a preemption of
            # THIS plan resumes instead of re-rendering from scratch.
            for d in (out / "refs", refs_scratch):
                d.mkdir(parents=True, exist_ok=True)
                (d / "plan.sha").write_text(cam_sha)
            s5 = modal_capture.capture_refs(
                run=run, slot=slot, model=model,
                cameras_path=cameras,
                tier=tier,
                cas_dir=Path(VOL) / "objects",
                refs_dir=refs_scratch,
                assets_dir=Path(_ASSETS),
                # Debug samples go straight to the Volume (+ commit) as frames
                # render, so the client can pull a few mid-run to eyeball the
                # picture-taking without waiting for the whole render.
                sample_dir=out / "refs" / "samples",
                commit=volume.commit,
                progress=heart.stage("refs"),
            )
            _sync_dir(refs_scratch, out / "refs")
            _record(status, status_path, "5", sig5, s5)
            summary["stages_run"].append(5)
            summary["refs"] = s5

    # ---- stage 6: gsplat 2DGS fine-tune -----------------------------------------
    trained = out / "trained.ply"
    if 6 in stages:
        if not transforms.is_file():
            raise FileNotFoundError("refs/transforms.json missing — run stage 5 first")
        train_params = TrainParams(**(spec.get("train") or {}))
        sig6 = _sig({"cloud": in_sha["cloud"], "refs": _sha256(transforms),
                     "params": train_params.as_summary(), "code": code6})
        if not force and _fresh(status, "6", sig6, [trained]):
            summary["stages_skipped"].append(6)
        else:
            refs_scratch = scratch / "refs"
            _sync_dir(out / "refs", refs_scratch)  # export reads refs from local disk
            # Stage 6 now trains from a COLMAP model (the Postshot-style point cloud
            # + poses + images) — build it once from this cell's Stage-3 cloud +
            # Stage-5 refs into local scratch, rebuilt on force or when absent.
            colmap_scratch = scratch / "colmap"
            # Rebuild on force, when absent, or when a warm container's export
            # predates the SZF supervision sidecar the refs could provide.
            colmap_stale = not (colmap_scratch / splat_colmap.CAMERAS_TXT).is_file() or (
                (refs_scratch / _stage5.FRAMES_DIRNAME).is_dir()
                and not (colmap_scratch / splat_colmap.SIDECAR_NAME).is_file()
            )
            if force or colmap_stale:
                heart.stage("train")(0, 0, "building COLMAP model (points3D + cameras + images)")
                splat_colmap.export_colmap(refs_scratch, cloud, colmap_scratch)
            # out_path lives ON the Volume: trained.ply + LODs land durably and
            # stage-6 checkpoints (splat/ckpt beside it) survive preemption, so
            # Modal's retry resumes mid-training. A fresh (non-resume) run must
            # drop stale checkpoints or train_splat would resume a dead run.
            if force:
                shutil.rmtree(out / "ckpt", ignore_errors=True)
            s6 = train_splat(
                run=run, slot=slot, model=model,
                colmap_dir=colmap_scratch,
                out_path=trained,
                # Stage-3 surfel cloud: the default init (params.init="surfels")
                # seeds every Gaussian at the 2DGS solution; from-points A/Bs
                # pass train={"init": "points"} and ignore it.
                init_ply=cloud,
                params=train_params,
                resume=not force,
                progress=heart.stage("train", commit=True),
            )
            _record(status, status_path, "6", sig6, s6)
            summary["stages_run"].append(6)
            summary["train"] = s6

    # ---- stage 7: heal (delete + heal + final-prune) -> healed.ply, then the
    # LOD ladder + near-lossless quantize of the DELIVERED (healed) model. Stage 6
    # emits the raw trained.ply; this is where the cleaned, shippable healed.ply /
    # healed.lodK.ply / healed.sqz are produced, so the two are viewable apart.
    healed = out / "healed.ply"
    if 7 in stages:
        if not trained.is_file():
            raise FileNotFoundError("trained.ply missing — run stage 6 first")
        if not transforms.is_file():
            raise FileNotFoundError("refs/transforms.json missing — run stage 5 first")
        train_params = TrainParams(**(spec.get("train") or {}))
        quant = quantize.QuantConfig(**(spec.get("quant") or {}))
        # Keyed on the raw trained.ply + the surfels (surface prior) + the refs
        # (heal supervision) + train/quant params + BOTH source files (heal lives
        # in stage6.py, quantize in quantize.py).
        sig7 = _sig({
            "trained": _sha256(trained), "cloud": in_sha["cloud"],
            "refs": _sha256(transforms), "params": train_params.as_summary(),
            "quant": quant.as_summary(),
            "code": _src_sig(Path(_stage6.__file__), Path(quantize.__file__)),
        })
        sqz = out / "healed.sqz"
        if not force and _fresh(status, "7", sig7, [healed, sqz]):
            summary["stages_skipped"].append(7)
        else:
            if force:
                for old in out.glob("healed*"):     # drop stale healed artifacts
                    old.unlink()
            refs_scratch = scratch / "refs"
            _sync_dir(out / "refs", refs_scratch)   # heal reads refs from local disk
            s7: dict[str, Any] = {"heal": heal_splat(
                run=run, slot=slot, model=model,
                trained_path=trained, cloud_path=cloud, refs_dir=refs_scratch,
                out_path=healed, params=train_params,
                progress=heart.stage("heal", commit=True),
            )}
            # Quantize the delivered healed model + its LOD ladder.
            s7["quantize"] = {"main": quantize.quantize_ply(healed, sqz, quant)}
            for lod in sorted(out.glob("healed.lod*.ply")):
                s7["quantize"][lod.stem] = quantize.quantize_ply(
                    lod, lod.with_suffix(".sqz"), quant
                )
            _record(status, status_path, "7", sig7, s7)
            summary["stages_run"].append(7)
            summary["heal"] = s7

    heart.stage("done")(1, 1, "")
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=4.0,
    memory=8192,
    timeout=3600,
    volumes={VOL: volume},
)
def eval_cross(spec: dict[str, Any]) -> dict[str, Any]:
    """Cross-evaluate trained splats on OTHER cells' reference view sets — the
    fair fidelity readout for a planner A/B: each model is scored against every
    listed ref set (PSNR/L1/depth over an `eval_max_views` random subset), so a
    baseline plan's model and a new plan's model face the SAME held-out views
    (near-band views from one plan, mid/far views from the other).

    spec = {
      "models": {label: "run/slot/model[/ply_name]"},   # default trained.ply
      "refs":   {label: "run/slot/model"},              # each cell's splat/refs
      "eval_max_views": 96,
    }
    Returns {model_label: {ref_label: metrics}}.
    """
    import torch
    from gsplat import rasterization_2dgs

    from splat.stage6 import TrainParams, _evaluate, _load_cloud, _load_scene

    volume.reload()
    device = torch.device("cuda")
    params = TrainParams(eval_max_views=int(spec.get("eval_max_views", 96)))

    scenes: dict[str, Any] = {}
    for label, key in spec["refs"].items():
        refs_dir = Path(VOL) / "cells" / key / "splat" / "refs"
        views, K, width, height, _ = _load_scene(torch, refs_dir, device)
        scenes[label] = (views, K, width, height)
        print(f"refs[{label}] = {key}: {len(views)} views @{width}", flush=True)

    out: dict[str, Any] = {}
    for label, key in spec["models"].items():
        parts = key.split("/")
        ply = parts[3] if len(parts) > 3 else "trained.ply"
        ply_path = Path(VOL) / "cells" / "/".join(parts[:3]) / "splat" / ply
        arrays = _load_cloud(ply_path)
        splats = {k: torch.from_numpy(v).to(device) for k, v in arrays.items()}
        out[label] = {"splats": int(arrays["means"].shape[0])}
        for rlabel, (views, K, width, height) in scenes.items():
            m = _evaluate(
                torch, rasterization_2dgs, splats, views, K, width, height,
                params, device,
            )
            out[label][rlabel] = m
            print(f"eval {label} on {rlabel}: {m}", flush=True)
        del splats
        torch.cuda.empty_cache()
    return out


# --- deploy-time probe -------------------------------------------------------------

# Runs from a real file:// document (headless Chromium won't execute inline
# scripts from a `data:` URL). It doesn't just read the renderer string — it
# reproduces the CAPTURE'S workload: a three.js-style WebGL2 context (same
# attributes), a render + readPixels loop against BOTH the canvas default
# framebuffer and an offscreen FBO, and an async `webglcontextlost` listener.
# That's what catches an immediate context LOSS (which a trivial getContext
# misses), reported on body attributes for a robust parse.
_PROBE_HTML = """<!doctype html><html><body><script>
(function () {
  var b = document.body;
  b.setAttribute('data-lost', '0');
  try {
    var c = document.createElement('canvas');
    c.width = 256; c.height = 256; b.appendChild(c);
    c.addEventListener('webglcontextlost', function () { b.setAttribute('data-lost', '1'); });
    var gl = c.getContext('webgl2', {
      alpha: false, antialias: false, powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
    });
    if (!gl) { b.setAttribute('data-r', 'NO-WEBGL2'); return; }
    var e = gl.getExtension('WEBGL_debug_renderer_info');
    b.setAttribute('data-r', String(
      (e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)) || '?'));
    var px = new Uint8Array(4), frames = 0;
    function pass() {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);       // canvas default framebuffer
      gl.viewport(0, 0, 256, 256);
      gl.clearColor(0.2, 0.4, 0.6, 1); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      var tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 256, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      var fb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      gl.clearColor(0.6, 0.2, 0.4, 1); gl.clear(gl.COLOR_BUFFER_BIT);   // offscreen FBO
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      gl.deleteFramebuffer(fb); gl.deleteTexture(tex); frames++;
    }
    for (var i = 0; i < 30 && !gl.isContextLost(); i++) pass();
    if (gl.isContextLost()) b.setAttribute('data-lost', '1');
    b.setAttribute('data-frames', String(frames));
    b.setAttribute('data-pixel', Array.prototype.join.call(px, ','));
  } catch (err) { b.setAttribute('data-r', 'ERR:' + err.message); b.setAttribute('data-lost', 'err'); }
})();
</script></body></html>
"""


@app.function(image=image, gpu="A100-40GB", timeout=300)
def probe_webgl() -> dict[str, Any]:
    """Report the WebGL renderer headless Chromium gets on this GPU class, for
    both launch modes — run after deploy to learn whether stage 5 renders on the
    GPU (ANGLE/Vulkan against the NVIDIA device) or falls back to SwiftShader
    (Google's CPU rasterizer: correct pixels, much slower). Each mode reports the
    renderer string, a hardware/software classification, and — when nothing came
    back — a tail of Chromium's stderr to diagnose missing Vulkan/EGL plumbing."""
    import re
    import subprocess
    import tempfile

    from splat.modal_capture import _COMMON_FLAGS, _GPU_FLAGS, _SW_FLAGS, _chromium_binary

    html = Path(tempfile.mkstemp(suffix=".html")[1])
    html.write_text(_PROBE_HTML, encoding="utf-8")
    url = f"file://{html}"
    out: dict[str, Any] = {}

    def attr(text: str, name: str) -> str:
        m = re.search(rf'{name}="([^"]*)"', text)
        return m.group(1) if m else ""

    for mode, flags in (("gpu", _GPU_FLAGS), ("swiftshader", _SW_FLAGS)):
        res = subprocess.run(
            [_chromium_binary(), *_COMMON_FLAGS, *flags,
             f"--user-data-dir=/tmp/probe-{mode}",
             "--virtual-time-budget=8000", "--dump-dom", url],
            capture_output=True, text=True, timeout=120,
        )
        renderer = attr(res.stdout, "data-r")
        software = bool(re.search(r"swiftshader|llvmpipe|software|swangle", renderer, re.I))
        out[mode] = {
            "renderer": renderer or "(no output)",
            "hardware": bool(renderer) and renderer not in ("NO-WEBGL2", "") and not software,
            # The decisive fields: did the context SURVIVE a render+readback loop?
            "context_lost": attr(res.stdout, "data-lost") or "?",
            "frames": attr(res.stdout, "data-frames") or "0",
            "pixel": attr(res.stdout, "data-pixel"),
            "stderr_tail": "" if renderer else res.stderr[-400:],
        }
    print(json.dumps(out, indent=1))
    return out


@app.function(image=image, gpu="A100-40GB", timeout=300)
def probe_gpu_stack() -> dict[str, Any]:
    """Inventory the container's GRAPHICS stack (not just CUDA), so we know
    whether headless Chrome can reach the NVIDIA device for hardware WebGL and,
    if not, exactly what's missing: the driver version, the Vulkan ICDs + EGL
    vendor files present (and the library each points at), the NVIDIA GL/Vulkan
    libs on the loader path, and the `vulkaninfo` device summary (does an NVIDIA
    device appear, or only llvmpipe?). Run after `modal:up`; the output tells us
    whether `NVIDIA_DRIVER_CAPABILITIES=all` sufficed or a version-matched
    `libnvidia-gl` install is needed."""
    import glob
    import os
    import subprocess

    def run(cmd: list[str]) -> str:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return (r.stdout + r.stderr).strip()
        except Exception as exc:  # diagnostic, never fatal
            return f"<{type(exc).__name__}: {exc}>"

    icds: dict[str, Any] = {}
    for d in ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"):
        for f in sorted(glob.glob(f"{d}/*.json")):
            try:
                icds[f] = json.loads(Path(f).read_text())
            except Exception:
                icds[f] = "(unreadable)"

    out: dict[str, Any] = {
        "driver": run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
        ),
        "caps_env": os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "(unset)"),
        "vulkan_icds": icds,
        "egl_vendor": sorted(glob.glob("/usr/share/glvnd/egl_vendor.d/*.json")),
        "nvidia_libs": run(
            ["bash", "-lc", "ldconfig -p | grep -iE 'nvidia|libGLX|libEGL|vulkan' || true"]
        ),
        # The decisive line: which physical devices the Vulkan loader enumerates.
        # "NVIDIA A100" here = hardware WebGL is reachable; only "llvmpipe" = still
        # software.
        "vulkan_devices": run(
            ["bash", "-lc",
             "vulkaninfo 2>/dev/null | grep -iE 'deviceName|driverName' | head -n 12 "
             "|| echo none"]
        ),
        "vulkaninfo": run(
            ["bash", "-lc", "vulkaninfo --summary 2>&1 | sed -n '1,70p' || echo missing"]
        ),
    }
    print(json.dumps(out, indent=1))
    return out
