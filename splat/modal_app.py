"""Modal app for the GPU half of the splat pipeline (stages 4-7).

Deploy with `modal deploy splat/modal_app.py` from the repo root; drive it with
`splat/modal_sync.py` (CLI or from the server). The split:

  * LOCAL (Mac): stages 1-3 — scene manifest, free-space grid, surfel cloud.
  * MODAL (one L40S container): stage 4 (camera plan — zone-driven
    single-shot field, pure CPU; lives here only to keep the stage chain in
    one place), stage 5 (reference renders — headless Chromium against the
    loopback host in `splat/modal_capture.py`), stage 6 (gsplat fine-tune in
    whichever representation `spec["train"]["representation"]` selects — 2DGS
    surfels by default, 3DGS with `"3dgs"` — → RAW `trained.ply`), stage 7
    (delete + heal + final-prune → delivered `healed.ply` + LOD ladder, then
    `splat/quantize.py` → `healed.sqz`).

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

# The GPU every function in this app (and modal_compact's) runs on — one constant
# so the fleet can't drift apart.
#
# L40S (Ada, sm_89) rather than the A100-40GB this started on: cheaper per hour,
# 48 GB instead of 40, and a much better fit for the WORK. Gaussian splatting is
# bound by FP32 shading, the per-tile radix sort and the alpha-blend rasterizer —
# none of which touch tensor cores or saturate HBM, which is where the A100's
# advantage lives. The A100 stays the better card only for a tensor-core or
# bandwidth-bound job; this pipeline is neither.
#
# ARCH NOTE: the pinned `gsplat` wheel is PREBUILT, so the gsplat trainer only
# works here if that wheel carries sm_89 SASS (or PTX to JIT from) — verified by
# the `gsplat` section of `probe_gpu_stack`, not assumed. Brush is wgpu/Vulkan
# and is architecture-agnostic by construction.
GPU = "L40S"
VOL = "/vol"
_SCRATCH = Path("/tmp/cells")
_ASSETS = "/assets"                      # capture page + three.js (baked below)

# Brush (github.com/ArthurBrussee/brush) — the wgpu/Vulkan splat trainer stage 6
# can run INSTEAD of the in-house gsplat loop (`train.trainer`). Pinned to a
# release tarball + the digest GitHub publishes beside it, so the image is
# reproducible and a tampered/truncated download fails the build rather than the
# run. The v0.3.0 CLI is what `splat/brush.py` targets — its flags differ from
# the repo's main branch (`--total-steps`, not `--total-train-iters`), so the
# version and the flag set move together.
BRUSH_VERSION = "v0.3.0"
_BRUSH_URL = (
    "https://github.com/ArthurBrussee/brush/releases/download/"
    f"{BRUSH_VERSION}/brush-app-x86_64-unknown-linux-gnu.tar.xz"
)
_BRUSH_SHA256 = "4f0f9a8785d1951c62df26aae247c02c5bba32b00f40b06df4e1c9b867399e20"
BRUSH_BIN = "/usr/local/bin/brush"

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
    # GPU and Chrome falls back to Mesa llvmpipe (software), which then crash-loops
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
        # enumerates the L40S (not Mesa llvmpipe). SwiftShader is unaffected (it's
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
    # --- Brush (the alternative stage-6 trainer) --------------------------------
    # A single static Rust binary, so nothing here can perturb the torch/gsplat
    # resolution above. It talks to the GPU through wgpu -> Vulkan, which the
    # NVIDIA ICD written near the top of this image already makes reachable
    # (`probe_gpu_stack` enumerates the L40S as the only Vulkan device), so no
    # extra driver plumbing is needed — only the loader (`libvulkan1`, installed
    # with chromium above) and the tools to unpack an `.xz` release archive.
    #
    # APPENDED AFTER the heavy pip layers on purpose: Modal rebuilds every layer
    # from the first change onward, so bumping Brush must not re-resolve torch.
    .apt_install("curl", "xz-utils")
    .run_commands(
        f"curl -fsSL {_BRUSH_URL} -o /tmp/brush.tar.xz",
        f'echo "{_BRUSH_SHA256}  /tmp/brush.tar.xz" | sha256sum -c -',
        "mkdir -p /tmp/brush-dl && tar -xJf /tmp/brush.tar.xz -C /tmp/brush-dl"
        " && find /tmp/brush-dl -type f -printf '%M %10s %p\\n'",
        # cargo-dist nests the payload under an archive-named dir beside the
        # README/LICENSE, and the bin target has been renamed across releases —
        # so select the one EXECUTABLE file rather than guessing either.
        'install -m 0755 "$(find /tmp/brush-dl -type f -perm -u+x'
        " ! -iname '*.md' ! -iname 'LICENSE*' | head -n1)\" "
        f"{BRUSH_BIN}",
        f"rm -rf /tmp/brush.tar.xz /tmp/brush-dl && {BRUSH_BIN} --version",
    )
    .env({
        # Pin wgpu to Vulkan. The loader is already restricted to the NVIDIA ICD
        # (VK_DRIVER_FILES above), but Mesa's lavapipe/intel/radeon ICDs are on
        # disk as chromium dependencies — an unset backend plus a future loader
        # change could silently land training on a CPU rasterizer, which reads as
        # a ~40x slowdown rather than an error.
        "WGPU_BACKEND": "vulkan",
        # Brush reports progress (`Refine iter N, M splats.`, eval PSNR/SSIM)
        # through `log`, which env_logger gates at `error` by default — without
        # this the heartbeat in splat/brush.py has nothing to parse.
        "RUST_LOG": "info",
    })
    # Capture page assets, served by the loopback host: the page + workers
    # verbatim, and the SAME three.js the debug viewer runs (importmap paths
    # /vendor/three/... map here).
    .add_local_file(
        _REPO / "client/public/splatcapture.html", f"{_ASSETS}/splatcapture.html"
    )
    .add_local_file(
        _REPO / "client/public/js/splatcapture.js", f"{_ASSETS}/js/splatcapture.js"
    )
    # splatcapture.js's whole module graph must be baked in — the loopback host
    # serves these from /js, so a missing one 404s the page and NO frames render.
    # capturecore.js is the render pipeline itself (renderer config, material prep,
    # the light/shadow/reflection bakes, and the OIT + ACES present), shared with the
    # matterport tour capture; it pulls in the weighted-blended OIT engine (oit.js),
    # the per-object scene-reflection baker (reflections.js), the emissive-light rig
    # (emissive.js — glowing lamps/screens + their point lights), the bake rig
    # (splatlight.js), and the reflective/matte discriminator (reflective.js).
    .add_local_file(
        _REPO / "client/public/js/capturecore.js", f"{_ASSETS}/js/capturecore.js"
    )
    .add_local_file(
        _REPO / "client/public/js/reflective.js", f"{_ASSETS}/js/reflective.js"
    )
    .add_local_file(
        _REPO / "client/public/js/oit.js", f"{_ASSETS}/js/oit.js"
    )
    .add_local_file(
        _REPO / "client/public/js/reflections.js", f"{_ASSETS}/js/reflections.js"
    )
    .add_local_file(
        _REPO / "client/public/js/emissive.js", f"{_ASSETS}/js/emissive.js"
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
    gpu=GPU,
    # COST NOTE: Modal bills CPU + memory on max(reservation, actual use) for the
    # WHOLE container lifetime, ON TOP of the GPU. This run is GPU-bound — stage 6
    # training dominates the wall time and uses only ~1-2 cores — so we RESERVE
    # little and let the CPU-heavy stage-5 SZF encode BURST (the soft CPU limit is
    # reservation + 16 cores; memory can grow past the reservation when the worker
    # has room, still billed for actual use). Reserving 16 cores + 50 GiB instead
    # would pay for them idle through the whole train, adding over a dollar an
    # hour on top of the L40S floor this sits just above.
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
      trainer: "brush" | "gsplat",     # which stage-6 back-end (default brush)
      plan / train / brush / quant: {} param overrides (dataclass field names),
      force: bool,
    }

    TRAINER. Stage 6 has two interchangeable back-ends behind one seam (both
    consume the COLMAP model and emit `trained.ply`): "brush" runs the upstream
    Brush binary (`splat/brush.py`, params from `spec["brush"]`), "gsplat" runs
    the in-house loop (`splat.stage6.train_splat`, params from `spec["train"]`).
    They are kept side by side because comparing them IS the benchmark; the
    trainer and its params are part of the stage-6 signature, so switching
    re-trains instead of reusing the other back-end's model.

    Brush owns delivered size itself (`--max-splats`) and its output has had no
    gsplat pass over it, so the Brush path runs stages 4-6 and quantizes inside
    stage 6; stage 7 (heal/compact, which re-optimizes through the gsplat
    rasterizer) applies to the gsplat path only and is skipped, not silently
    applied, if it lands in the window.
    """
    from splat import brush as splat_brush, colmap as splat_colmap, modal_capture, quantize
    from splat.stage4 import PlanParams, plan_cameras
    from splat.stage6 import TrainParams, heal_splat, train_splat

    run, slot, model = spec["run"], spec["slot"], spec["model"]
    stages = set(spec.get("stages") or [4, 5, 6, 7])
    force = bool(spec.get("force"))
    trainer = spec.get("trainer") or "brush"
    if trainer not in ("brush", "gsplat"):
        raise ValueError(f"unknown trainer {trainer!r} (expected 'brush' or 'gsplat')")
    key = f"{run}/{slot}/{model}"
    heart = _Heartbeat(key)

    # Identify this run up front so concurrent containers are distinguishable in
    # the aggregated `modal app logs` stream (every heartbeat line is also cell-
    # prefixed — see _Heartbeat). Includes the param overrides so runs that
    # differ only by plan/train config are told apart too.
    print(
        f"=== run_cell START {key} | stages={sorted(stages)} force={force} "
        f"trainer={trainer} | plan={spec.get('plan') or {}} "
        f"train={spec.get('train') or {}} brush={spec.get('brush') or {}} "
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
        # capturecore.js (the shared render pipeline), splatlight.js (bake rig),
        # oit.js (weighted-blended OIT + the all-layers shadow bake), reflections.js
        # (per-object scene reflections), reflective.js (which surfaces stay
        # reflective vs. forced matte), and emissive.js (emissive glow + point
        # lights) all determine every captured pixel — fold them in so a change
        # there re-renders stage 5 instead of silently reusing frames.
        _js / "capturecore.js", _js / "splatlight.js", _js / "oit.js",
        _js / "reflections.js", _js / "reflective.js", _js / "emissive.js",
    )
    # Stage 6 now trains from the COLMAP export, so its code sig folds in colmap.py
    # too — editing the exporter re-runs stage 6 rather than reusing a stale model.
    # BOTH back-ends' sources go in regardless of which one runs, so the signature
    # is a property of the deployed pipeline rather than of this invocation.
    code6 = _src_sig(
        Path(_stage6.__file__), Path(splat_colmap.__file__),
        Path(splat_brush.__file__),
    )
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

    # ---- stage 6: fine-tune — Brush, or the in-house gsplat loop (2DGS | 3DGS
    # per train.representation). Both read the SAME COLMAP model built below and
    # write the SAME trained.ply; see the `TRAINER` note in the docstring.
    trained = out / "trained.ply"
    trained_sqz = out / "trained.sqz"
    if 6 in stages:
        if not transforms.is_file():
            raise FileNotFoundError("refs/transforms.json missing — run stage 5 first")
        train_params = TrainParams(**(spec.get("train") or {}))
        brush_params = splat_brush.BrushParams(**(spec.get("brush") or {}))
        quant = quantize.QuantConfig(**(spec.get("quant") or {}))
        # The trainer, ITS params and (for Brush) the pinned binary version are all
        # in the key: switching back-end, changing max-splats, or bumping the Brush
        # release each has to re-train rather than serve the previous model.
        sig6 = _sig({
            "cloud": in_sha["cloud"], "refs": _sha256(transforms),
            "trainer": trainer, "code": code6,
            "params": (
                brush_params.as_summary() if trainer == "brush"
                else train_params.as_summary()
            ),
            "brush_version": BRUSH_VERSION if trainer == "brush" else None,
            # Brush quantizes inside this stage (it has no stage 7), so the .sqz
            # config belongs to the key and the artifact list.
            "quant": quant.as_summary() if trainer == "brush" else None,
        })
        art6 = [trained, trained_sqz] if trainer == "brush" else [trained]
        if not force and _fresh(status, "6", sig6, art6):
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
                (out / splat_brush.PROGRESS_NAME).unlink(missing_ok=True)
            if trainer == "brush":
                s6 = splat_brush.train_brush(
                    run=run, slot=slot, model=model,
                    colmap_dir=colmap_scratch,
                    # Brush writes the Volume copy DIRECTLY, every export_every
                    # steps — so a preempted retry finds a partial model to warm
                    # restart from, and the client can pull one mid-run.
                    out_path=trained,
                    params=brush_params,
                    brush_bin=BRUSH_BIN,
                    resume=not force,
                    progress=heart.stage("train", commit=True),
                )
                # No stage 7 on this path, so the shippable .sqz is produced here.
                s6["quantize"] = quantize.quantize_ply(trained, trained_sqz, quant)
                # A cell trained with gsplat BEFORE this run still has that path's
                # stage-7 deliverables on the Volume, and nothing here overwrites
                # them — so drop them, or `pull_cell` brings a stale healed.ply
                # back to sit beside the fresh trained.ply as if it belonged to it.
                for old in out.glob("healed*"):
                    old.unlink()
            else:
                s6 = train_splat(
                    run=run, slot=slot, model=model,
                    colmap_dir=colmap_scratch,
                    out_path=trained,
                    # Stage-3 surfel cloud: the default init (params.init="surfels")
                    # seeds every Gaussian on the mesh surface (a 3DGS run gets the
                    # same seed with a real thickness axis); from-points A/Bs pass
                    # train={"init": "points"} and ignore it. Brush ignores this
                    # entirely — it seeds from the COLMAP points3D.
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
    if 7 in stages and trainer == "brush":
        # Heal/compact MEASURES each Gaussian's contribution and re-optimizes the
        # survivors through the gsplat rasterizer — the loop Brush was chosen over.
        # Brush also bounds delivered size itself (`--max-splats`), so the pass has
        # nothing to add here and could undo the quality it was picked for. Skipped
        # explicitly (and visibly in the summary) rather than quietly applied.
        heart.stage("heal")(1, 1, "skipped — stage 7 compaction is the gsplat path's")
        summary["stages_skipped"].append(7)
    elif 7 in stages:
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
    gpu=GPU,
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

    from splat.stage6 import (
        TrainParams,
        _evaluate,
        _load_cloud,
        _load_scene,
        _ply_representation,
        _rasterizer,
    )

    volume.reload()
    device = torch.device("cuda")
    eval_max_views = int(spec.get("eval_max_views", 96))

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
        # Per-MODEL representation, read off each `.ply`: a cross-eval is exactly the
        # place where a 2DGS baseline and a 3DGS candidate get scored on the same
        # views, so each must be rendered through its own rasterizer.
        params = TrainParams(
            eval_max_views=eval_max_views,
            representation=_ply_representation(ply_path),
        )
        raster = _rasterizer(params)
        out[label] = {
            "splats": int(arrays["means"].shape[0]),
            "representation": params.representation,
        }
        for rlabel, (views, K, width, height) in scenes.items():
            m = _evaluate(
                torch, raster, splats, views, K, width, height,
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


@app.function(image=image, gpu=GPU, timeout=300)
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


@app.function(image=image, gpu=GPU, timeout=900)
def probe_brush() -> dict[str, Any]:
    """END-TO-END smoke test of the Brush trainer on this image: synthesize a tiny
    COLMAP model (8 posed 64px views around a point cube), train it for a handful
    of steps, and report what happened.

    It answers, in one run, every question that can only be settled inside the
    container: does the pinned release binary LINK here (its glibc vs the image's
    — the one real risk of shipping a prebuilt), does wgpu land on the GPU rather
    than a Mesa CPU rasterizer, does Brush parse the FLAT COLMAP layout
    `splat.colmap.export_colmap` writes (no `sparse/0`, capital-D `points3D.txt`),
    and does it export a `.ply` where we point it. `train.device` is the decisive
    field: `train_brush` refuses to return a model trained on a CPU rasterizer, so
    an `ok: true` here means the GPU path really ran."""
    import subprocess
    import tempfile

    import numpy as np
    from PIL import Image

    from splat import brush
    from splat.colmap import rotmat2qvec

    root = Path(tempfile.mkdtemp(prefix="brush-probe-"))
    scene, out = root / "scene", root / "out"
    scene.mkdir(parents=True, exist_ok=True)

    w = h = 64
    focal = 60.0
    (scene / "cameras.txt").write_text(
        f"1 PINHOLE {w} {h} {focal:.10g} {focal:.10g} {w / 2:.10g} {h / 2:.10g}\n",
        encoding="utf-8",
    )

    # 8 views on a ring, each looking at the origin, in the OpenCV camera frame
    # (+x right, +y down, +z forward) COLMAP and `export_colmap` both use.
    img_lines: list[str] = []
    for i in range(8):
        th = 2.0 * np.pi * i / 8.0
        eye = np.array([3.0 * np.cos(th), 1.0, 3.0 * np.sin(th)])
        fwd = -eye / np.linalg.norm(eye)
        right = np.cross(fwd, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        c2w = np.eye(4)
        c2w[:3, :3] = np.stack([right, np.cross(fwd, right), fwd], axis=1)
        c2w[:3, 3] = eye
        w2c = np.linalg.inv(c2w)
        q, t = rotmat2qvec(w2c[:3, :3]), w2c[:3, 3]
        name = f"cam{i:03d}.png"
        # A per-view gradient: enough signal that the optimizer has something to
        # fit, and RGBA so the alpha path (`match_alpha_weight`) is exercised too.
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) / max(w - 1, 1)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = (255 * xx).astype(np.uint8)
        rgba[..., 1] = (255 * yy).astype(np.uint8)
        rgba[..., 2] = np.uint8(255 * i / 8.0)
        rgba[..., 3] = 255
        Image.fromarray(rgba).save(scene / name)
        img_lines.append(
            f"{i + 1} {q[0]:.10g} {q[1]:.10g} {q[2]:.10g} {q[3]:.10g} "
            f"{t[0]:.10g} {t[1]:.10g} {t[2]:.10g} 1 {name}\n\n"
        )
    (scene / "images.txt").write_text("".join(img_lines), encoding="utf-8")

    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.5, 0.5, size=(512, 3))
    (scene / "points3D.txt").write_text(
        "".join(
            f"{j} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 200 180 160 1.0\n"
            for j, p in enumerate(pts, 1)
        ),
        encoding="utf-8",
    )

    # Drive the REAL trainer rather than a hand-rolled subprocess, so the probe
    # covers what production runs: the argv, the log parsing, the preflight
    # guards and the export contract. These 64px frames encode to well under
    # Brush's 16 KB header-sniff buffer, so they also exercise `_pad_small_images`
    # — without it this load dies with "early eof".
    ver = subprocess.run(
        [BRUSH_BIN, "--version"], capture_output=True, text=True, timeout=60
    )
    steps = 60
    params = brush.BrushParams(
        iterations=steps, export_every=steps, max_resolution=w, sh_degree=1
    )
    result: dict[str, Any] = {
        "binary": BRUSH_BIN,
        "pinned": BRUSH_VERSION,
        "version": (ver.stdout or ver.stderr).strip(),
    }
    ply = out / "probe.ply"
    try:
        summary = brush.train_brush(
            run="probe", slot="probe", model="probe",
            colmap_dir=scene, out_path=ply,
            params=params, brush_bin=BRUSH_BIN, resume=False,
            progress=lambda d, t, m: print(f"[probe] {d}/{t} {m}", flush=True),
        )
        # Brush's `.ply` has to survive the DELIVERY path too, which is the other
        # half of "does this trainer drop in": `_ply_representation` must read it
        # as 3dgs (it carries a real `scale_2`), and the quantizer must parse it
        # into the `.sqz` the client fetches. Failing here would mean a model that
        # trains fine and can't be shipped.
        from splat import quantize
        from splat.stage6 import _ply_representation

        result |= {
            "ok": True,
            "train": summary,
            "representation": _ply_representation(ply),
            "quantize": quantize.quantize_ply(ply, out / "probe.sqz"),
        }

        # The warm-restart leg. It only ever fires after a preemption, so a bug
        # here would never show up in a normal run and would then break every
        # retry — and it is the one path that puts a `.ply` INSIDE the dataset
        # dir, which is otherwise a hard error. Fake the marker a killed run
        # leaves behind and check the relaunch picks up from it and cleans up.
        resume_params = brush.BrushParams(
            iterations=120, export_every=60, max_resolution=w, sh_degree=1
        )
        (out / brush.PROGRESS_NAME).write_text(
            json.dumps({"iter": 60, "total": 120, "params": resume_params.as_summary()}),
            encoding="utf-8",
        )
        resumed = brush.train_brush(
            run="probe", slot="probe", model="probe",
            colmap_dir=scene, out_path=ply,
            params=resume_params, brush_bin=BRUSH_BIN, resume=True,
            progress=lambda d, t, m: print(f"[probe:resume] {d}/{t} {m}", flush=True),
        )
        result["resume"] = {
            "resumed_from": resumed["resumed_from"],
            "ok": resumed["resumed_from"] == 60,
            "init_cleaned": not (scene / "init.ply").exists(),
            "marker_cleared": not (out / brush.PROGRESS_NAME).exists(),
        }
    except Exception as exc:  # a probe reports failures, it doesn't raise them
        result |= {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, indent=1))
    return result


@app.function(image=image, gpu=GPU, timeout=300)
def probe_gpu_stack() -> dict[str, Any]:
    """Inventory the container's GRAPHICS stack (not just CUDA), so we know
    whether headless Chrome can reach the NVIDIA device for hardware WebGL and,
    if not, exactly what's missing: the driver version, the Vulkan ICDs + EGL
    vendor files present (and the library each points at), the NVIDIA GL/Vulkan
    libs on the loader path, and the `vulkaninfo` device summary (does an NVIDIA
    device appear, or only llvmpipe?). Run after `modal:up`; the output tells us
    whether `NVIDIA_DRIVER_CAPABILITIES=all` sufficed or a version-matched
    `libnvidia-gl` install is needed.

    It also reports a `gsplat` section, because the OTHER thing a GPU change can
    break is silent: the pinned gsplat wheel is PREBUILT, so it only runs on
    architectures it was compiled for (plus any it can JIT from embedded PTX). A
    card whose compute capability is missing from the wheel fails at the first
    kernel launch with "no kernel image is available for execution on the device"
    — deep inside a paid training run. This launches both rasterizers on a
    two-Gaussian scene up front so that shows up in a 30-second probe instead."""
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
        # The NVIDIA card here = the GPU is reachable for Brush (and for hardware
        # WebGL); only "llvmpipe" = still software.
        "vulkan_devices": run(
            ["bash", "-lc",
             "vulkaninfo 2>/dev/null | grep -iE 'deviceName|driverName' | head -n 12 "
             "|| echo none"]
        ),
        "vulkaninfo": run(
            ["bash", "-lc", "vulkaninfo --summary 2>&1 | sed -n '1,70p' || echo missing"]
        ),
        "gsplat": _probe_gsplat(),
    }
    print(json.dumps(out, indent=1))
    # Round-trip through JSON before returning. The caller is a LOCAL machine with
    # no torch installed, and torch leaks str/tuple SUBCLASSES (`torch.__version__`,
    # `torch.Size`) that pickle by reference — they serialize to JSON invisibly but
    # blow up on unpickle with "No module named 'torch'". This makes the payload
    # plain by construction instead of relying on casting every field.
    return json.loads(json.dumps(out, default=str))


def _probe_gsplat() -> dict[str, Any]:
    """Launch both gsplat rasterizers on a trivial scene and report what happened
    — the compiled-arch check described in `probe_gpu_stack`. Returns the device's
    compute capability alongside a per-rasterizer ok/error, so a wheel that lacks
    this card's arch is named as such rather than surfacing mid-train.

    Every value is coerced to a plain str/int/bool: the caller is the LOCAL
    machine, which has no torch, and a `torch.__version__` (a str subclass) or a
    `torch.Size` in the payload fails to unpickle there."""
    out: dict[str, Any] = {}
    try:
        import torch

        out["torch"] = str(torch.__version__)
        out["capability"] = "sm_%d%d" % torch.cuda.get_device_capability()
        out["device_name"] = str(torch.cuda.get_device_name())
        out["arch_list"] = [str(a) for a in torch.cuda.get_arch_list()]
    except Exception as exc:
        return {"ok": False, "error": f"torch unavailable: {type(exc).__name__}: {exc}"}

    try:
        import gsplat
        from gsplat import rasterization, rasterization_2dgs

        out["gsplat"] = str(getattr(gsplat, "__version__", "?"))
    except Exception as exc:
        return {**out, "ok": False, "error": f"gsplat import: {type(exc).__name__}: {exc}"}

    dev = torch.device("cuda")
    n = 2
    means = torch.zeros(n, 3, device=dev)
    means[:, 2] = 2.0
    quats = torch.zeros(n, 4, device=dev)
    quats[:, 0] = 1.0
    scales = torch.full((n, 3), 0.1, device=dev)
    opacities = torch.full((n,), 0.9, device=dev)
    colors = torch.ones(n, 1, 3, device=dev)          # SH degree 0
    viewmats = torch.eye(4, device=dev)[None]
    k = torch.tensor([[[64.0, 0, 32.0], [0, 64.0, 32.0], [0, 0, 1.0]]], device=dev)

    for name, fn in (("3dgs", rasterization), ("2dgs", rasterization_2dgs)):
        try:
            res = fn(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=colors, viewmats=viewmats, Ks=k, width=64, height=64,
                sh_degree=0,
            )
            img = res[0]
            torch.cuda.synchronize()
            out[name] = {"ok": True, "shape": [int(d) for d in img.shape]}
        except Exception as exc:
            out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    out["ok"] = all(out[k].get("ok") for k in ("3dgs", "2dgs") if isinstance(out.get(k), dict))
    return out
