"""Stage-5 reference capture INSIDE the Modal container (see `splat/modal_app.py`).

Runs the exact same renderer as the local pipeline — the WebGL capture page
(`client/public/js/splatcapture.js`) in a headless Chromium — against a
loopback HTTP host that replays the five endpoints the page speaks
(`server/app/api/routes.py` stage-5 protocol):

  GET  /splatcapture.html, /js/*, /vendor/three/*      static (baked into the image)
  GET  /runs/{run}/splat/stage5/{slot}/{model}/manifest?token=
  GET  /cameras.json                                   the Stage-4 plan
  GET  /bundle                                         SMB1 mesh stream from the CAS
  POST /runs/{run}/splat/stage5/{slot}/{model}/frames?token=   SRF1 batches
  POST /runs/{run}/splat/stage5/{slot}/{model}/finish?token=

The page and the SZF/pose contract (`splat.stage5`) are reused VERBATIM, so the
render conventions locked to the debug-viewer stack never fork. Frames are
encoded on a thread pool via `stage5.write_reference_frame` into a LOCAL
scratch dir (the caller syncs them to the Volume afterwards).

WIRE: raw SRF1, NOT gzip-wrapped — the opposite of the local server's choice, for
a reason specific to this host (see `_WIRE_GZIP`). Set STARSHOT_CAPTURE_GZIP=1 to
put the transport codec back for an A/B.

THROUGHPUT is reported live in the render heartbeat (`N img/s (avg M) · eta …`,
see `_phase_msg`) and recorded in the returned summary as `img_per_s`. Stage 5
runs for minutes per cell, so a capture that has halved in speed is otherwise
invisible: the watchdog only notices a full stop.

RENDERER: stage 5 is a WebGL/three.js workload (NOT CUDA), so it needs the
GRAPHICS half of the driver, not just compute. It now gets it: `modal_app`
registers the NVIDIA Vulkan ICD + EGL vendor itself (the driver ships the libs
but not the loader JSONs), and `modal_app.probe_webgl` measures HARDWARE WebGL
on the L40S — `ANGLE (NVIDIA, Vulkan ..., NVIDIA)`, no context loss.

That was not always true. On the earlier A100 image the ANGLE/Vulkan path
resolved to Mesa `llvmpipe` and the fallback to SwiftShader, both CPU
rasterizers, and this stage rendered every frame on the CPU. The pipeline is
written to be indifferent to which it gets: for an UNLIT render (albedo +
planar-Z depth + coverage alpha, antialias off) the CPU output is functionally
identical — same page, same stack, same locked conventions — so we simply take
the fastest backend available. We try ANGLE/Vulkan first and fall back to
SwiftShader. `&force=1` rides along because the capture page refuses software
WebGL otherwise; on a hardware backend it is a no-op. The achieved renderer
string is reported in the summary, so whether a run was hardware- or CPU-rendered
is always visible — check it before attributing a slow stage 5 to anything else.

Single-tenant by construction: one job, one fixed token, loopback only.
"""

from __future__ import annotations

import asyncio
import contextlib
import glob
import gzip
import mimetypes
import os
import shutil
import struct
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from splat import stage5

# The capture page is ES modules (three.js via an importmap) + a wasm transcoder.
# The browser refuses a module served with a non-JS MIME (strict module MIME
# checks), and `WebAssembly.instantiateStreaming` needs `application/wasm` — and
# Python's mimetypes doesn't know `.wasm`/`.mjs` — so a mis-typed static file
# silently breaks the whole page before any capture runs. Register them.
for _ext, _mt in ((".js", "text/javascript"), (".mjs", "text/javascript"),
                  (".wasm", "application/wasm")):
    mimetypes.add_type(_mt, _ext)

_TOKEN = "local"
_PORT = 8765
# When set, stream Chromium's stdout/stderr (incl. the capture page's console)
# into the container log and enable Chrome's own logging — the only window into
# a browser that isn't posting frames. Off → stdout/stderr go to DEVNULL.
_CAPTURE_DEBUG = bool(os.environ.get("STARSHOT_CAPTURE_DEBUG"))
_SMB1_MAGIC = b"SMB1"
_SRF1_MAGIC = b"SRF1"
_SZC1_MAGIC = b"SZC1"  # gzip-wrapped SRF1 (client-side wire compression; see refcapture)
# Encode workers / backlog mirror the server's capture ingest (routes.py):
# zstd + the numpy filter release the GIL, so threads parallelize fine.
_ENCODE_WORKERS = 8
# Backlog is the FLOW-CONTROL depth, in POST batches held while their frames are
# encoded/written. It is what makes a brief encode hiccup visible to the renderer:
# the frames handler parks in `asyncio.wait` once the backlog is full, which delays
# its response, which trips the page's `renderedViews - postedViews < maxOutstanding`
# gate and stalls the render loop. At 8 batches × 4 views that was only ~32 views of
# slack against a 42-view gate — tight enough that ordinary jitter propagated all
# the way back to the GPU. 16 decouples them; the cost is bounded memory (each held
# batch is one POST body, so ~16 × the batch size).
_ENCODE_BACKLOG = 16
# WATCHDOG. There is no single "is it stuck?" deadline, because the capture goes
# through three phases with completely different expected durations, and only the
# host knows which one it is in. Timing all of them from one clock started at
# browser launch is what makes a big scene indistinguishable from a hung one: a
# 2 GB / 800-mesh tier can legitimately spend many minutes streaming and parsing
# before frame 1, and a flat grace kills it mid-download — then reports the same
# "renderer stalled" as a genuinely dead browser.
#
# So each phase is timed from its own last sign of life:
#   boot    — launched, hasn't asked for /bundle yet. Page + three.js modules +
#             manifest + cameras.json. Size-independent, so a fixed budget is right.
#   load    — /bundle is streaming. The liveness signal is BYTES MOVING, not
#             frames; aiohttp's write backpressure means the timestamp tracks what
#             the client actually consumed. Unbounded in total, bounded in silence.
#   build   — bundle fully delivered; the client is finishing GLB parses, KTX2
#             transcodes and the one-off shadow bake. NOW a first-frame grace is
#             meaningful, because it measures work of a knowable size.
# After the first frame, a long silence is a wedged renderer as before.
_BOOT_GRACE_S = 180.0
# Once the page's assets have been served, it asks for its work order almost
# immediately — the manifest fetch is the second thing `runCapture` does. So a page
# that has loaded and then gone quiet for this long is WEDGED, not slow, and the
# most likely reason is that `getContext("webgl2")` never returned: that call is
# synchronous, so when the GPU process cannot produce a context the page neither
# proceeds nor throws, and no amount of page-side error handling can report it.
# Failing fast here is what turns a 3-minute silent boot into a named error and lets
# the next backend be tried while it still might help.
_PAGE_SILENT_S = 45.0
_BUNDLE_STALL_S = 120.0
_FIRST_FRAME_GRACE_S = 420.0
_STALL_S = 180.0
_POLL_S = 2.0
# Trailing window for the live images/second readout in the render heartbeat
# (mirrors the local server's `_stage5_capture_rate`). Short enough that the number
# reacts to a stall, long enough not to jitter between two-second polls.
_RATE_WINDOW_S = 5.0
# WIRE COMPRESSION IS OFF HERE, deliberately, and this is NOT the same trade the
# local server makes. The page can gzip each SRF1 batch before POSTing it
# (splatcapture-worker.js), which is a clear win over a real socket: it was added
# because the local server's single asyncio event loop — on Windows, so no uvloop —
# was measured saturating a core just ingesting the raw ~6 MB/view bodies.
#
# In THIS container the browser and this host are the same machine talking over
# 127.0.0.1, so the wire is a memory copy and the bytes were never the constraint.
# What gzip buys is nothing; what it costs is a compress in the page's post workers
# plus an inflate here, for every view, on top of the sub-left+zstd that actually
# produces the .szf (~12 ms/view). Measured across three cells the capture floors
# at ~53 views/s even when the GPU is idle (render 0.23 ms/view), and 4 post workers
# × ~80 MB/s of deflate ≈ 320 MB/s is exactly 53 views/s × 6 MB — i.e. the transport
# codec, not the encode it wraps, is what the floor is made of. So skip it and let
# the loopback carry raw SRF1. Both hosts still ACCEPT either framing (see the
# `frames` handler's SZC1 check), so this is a per-run choice, not a format change.
_WIRE_GZIP = bool(os.environ.get("STARSHOT_CAPTURE_GZIP"))
# Render-backend attempts, in order, each tried until one renders a frame. All run
# with force=1 (see the module docstring), which is a no-op on a hardware backend.
#
#   vulkan            ANGLE/Vulkan against the NVIDIA device — the fast path.
#   vulkan-nosurface  the same, minus the Vulkan WSI surface extensions.
#   swiftshader       Google's CPU rasterizer. Correct pixels, ~100x slower.
#
# The middle mode exists because of a real failure. `vkCreateInstance() failed: -7`
# is VK_ERROR_EXTENSION_NOT_PRESENT — NOT a missing driver (that is -9). The driver
# was healthy throughout: `probe_gpu_stack` enumerated the L40S with
# DRIVER_ID_NVIDIA_PROPRIETARY and `vulkaninfo` built an instance happily. What
# Chromium could not get was an INSTANCE EXTENSION, and the ones it asks for depend
# on its ozone platform — in a container with no DISPLAY and no XDG_RUNTIME_DIR, the
# platform surface extensions (VK_KHR_xcb_surface / VK_KHR_wayland_surface) are the
# obvious candidates. `--disable-vulkan-surface` makes Chromium skip the swapchain
# path and bit-blit instead, which is what the Chrome team's own headless-GPU recipe
# uses and is harmless under `--headless=new`, where nothing reaches a screen.
#
# It is a separate MODE rather than an extra flag on `vulkan` because the plain mode
# demonstrably worked on an earlier Chromium and this diagnosis is an inference: keep
# both and let whichever works win. `renderer` in the summary records which one did.
_RENDER_MODES = ("vulkan", "vulkan-nosurface", "swiftshader")
# Debug: how many rendered frames to decode to PNG and publish mid-run, so the
# picture-taking can be eyeballed WHILE stage 5 (and 6) continue.
_SAMPLE_COUNT = 8

ProgressCb = Callable[[int, int, str], None]


def _decompress_frame_batch(body: bytes) -> bytes:
    """Inflate a gzip-wrapped (SZC1) batch to its raw SRF1 bytes — mirrors
    `server/app/services/refcapture.decompress_frame_batch` (the SZC1 wire contract
    in splatcapture-worker.js). Byte-identical output, so the encode path is unchanged."""
    return gzip.decompress(body[len(_SZC1_MAGIC):])


def _parse_frame_batch(body: bytes) -> list[tuple[str, int, int, int]]:
    """One capture-page SRF1 POST body → [(view id, resolution, rgba_offset,
    depth_offset)] descriptors, no pixel copies. Mirrors
    `server/app/services/refcapture.parse_frame_batch` (kept in sync by the
    SRF1 framing contract in splatcapture-worker.js)."""
    if body[:4] != _SRF1_MAGIC:
        raise ValueError("frame batch: bad magic (want SRF1)")
    off = 4
    (count,) = struct.unpack_from("<I", body, off)
    off += 4
    frames: list[tuple[str, int, int, int]] = []
    for _ in range(count):
        id_len, w, h = struct.unpack_from("<III", body, off)
        off += 12
        vid = body[off : off + id_len].decode("utf-8")
        off += id_len
        if w != h:
            raise ValueError(f"frame batch: {vid} is {w}x{h}, want square")
        rgba_len, depth_len = w * h * 4, w * h * 2
        if off + rgba_len + depth_len > len(body):
            raise ValueError(f"frame batch: truncated at {vid}")
        frames.append((vid, int(w), off, off + rgba_len))
        off += rgba_len + depth_len
    if off != len(body):
        raise ValueError(f"frame batch: {len(body) - off} trailing bytes")
    return frames


def _encode_batch(
    refs_dir: Path, body: bytes, frames: list[tuple[str, int, int, int]]
) -> list[str]:
    """Pool worker: slice each view's planes out of the POST body (zero-copy
    memoryview) and persist its SZF frame."""
    view = memoryview(body)
    out: list[str] = []
    for vid, resolution, rgba_off, depth_off in frames:
        rgba = view[rgba_off : rgba_off + resolution * resolution * 4]
        depth = view[depth_off : depth_off + resolution * resolution * 2]
        stage5.write_reference_frame(refs_dir, vid, resolution, rgba, depth)
        out.append(vid)
    return out


# --- loopback host ---------------------------------------------------------------


def _print_capture_stats(tag: str, stats: dict[str, Any]) -> None:
    """Print the page's end-of-capture accounting to the container log.

    Without this the numbers only exist inside the returned summary, which is the
    wrong place to read them from when the question is "which pass do I fix next".
    The per-pass line is the important one: `cpu/gpu ms · draws` per pass says
    whether a capture is submission-bound (high cpu, high draws, low gpu → merge
    geometry or cut passes) or fill-bound (high gpu → fewer pixels), and those two
    want opposite changes."""
    per_view = stats.get("per_view_ms") or {}
    if per_view:
        buckets = " / ".join(f"{k} {v}" for k, v in per_view.items())
        print(
            f"{tag} capture: {stats.get('views_per_s')} views/s "
            f"@ {stats.get('resolution')}² · {buckets} ms/view",
            flush=True,
        )
    passes = stats.get("passes") or {}
    names = [k[: -len("_cpu_ms")] for k in passes if k.endswith("_cpu_ms")]
    if names:
        parts = []
        for n in names:
            gpu = passes.get(f"{n}_gpu_ms")
            parts.append(
                f"{n} {passes[f'{n}_cpu_ms']}/{'-' if gpu is None else gpu}"
                f" x{passes.get(f'{n}_draws')}"
            )
        print(
            f"{tag} passes: {' · '.join(parts)}  (cpu/gpu ms · draws, "
            f"{passes.get('gpu_timing')}, {passes.get('gpu_sampled_views')} gpu samples)",
            flush=True,
        )
    if stats.get("fence_ms_per_segment") is not None:
        print(
            f"{tag} readback: fence signaled {stats['fence_ms_per_segment']} ms after "
            f"seal ({stats.get('views_per_segment')} views/segment) — a value near zero "
            f"means the GPU was already done and the wall is on the CPU side",
            flush=True,
        )


def _make_app(state: dict[str, Any]):
    from aiohttp import web

    def _check_token(request) -> None:
        if request.query.get("token", "") != _TOKEN:
            raise web.HTTPConflict(text="stale capture token")

    async def manifest(request):
        _check_token(request)
        # The boot phase covers two very different states — "the browser hasn't
        # loaded the page yet" and "the page is up but hasn't started work" — and
        # only the second implicates our JS. This is the marker that separates them.
        state["manifest_at"] = time.monotonic()
        return web.json_response(
            {
                "run": state["run"],
                "slot": state["slot"],
                "model": state["model"],
                "resolution": state["resolution"],
                "near": state["near"],
                "far": state["far"],
                "fov_deg": state["fov_deg"],
                "background": list(stage5.BACKGROUND_RGB),
                "cameras_url": "/cameras.json",
                "bundle_url": "/bundle",
                "pending": sorted(state["pending"]),
                "total": state["total"],
            }
        )

    async def cameras(request):
        return web.FileResponse(state["cameras_path"])

    async def bundle(request):
        """Stream the tier as one SMB1 bundle straight from the CAS files —
        the same framing as the server's `_mesh_bundle`.

        Records delivery progress as it goes. That is what lets the watchdog tell
        "still loading a large scene" from "hung": every successful write is proof
        the client is alive and consuming, because aiohttp's write backpressures
        against the transport rather than buffering the whole bundle."""
        resp = web.StreamResponse()
        resp.content_type = "application/octet-stream"
        await resp.prepare(request)
        await resp.write(_SMB1_MAGIC)
        cas: Path = state["cas_dir"]
        # Authoritative reset: a re-fetch (page reload / retried request) starts
        # the delivery over, so the counters must too or the watchdog would read a
        # half-sent bundle as further along than it is.
        now = time.monotonic()
        state["bundle_bytes"] = 0
        state["bundle_meshes"] = 0
        state["bundle_done_at"] = None
        state["bundle_last_write_at"] = now
        state["bundle_started_at"] = now
        state["bundle_samples"].clear()
        for node_id, sha in sorted(state["tier"].items()):
            data = await asyncio.to_thread(
                (cas / sha[:2] / f"{sha}.glb").read_bytes
            )
            id_bytes = node_id.encode("utf-8")
            await resp.write(struct.pack("<I", len(id_bytes)) + id_bytes)
            await resp.write(struct.pack("<I", len(data)))
            await resp.write(data)
            state["bundle_bytes"] += len(data)
            state["bundle_meshes"] += 1
            state["bundle_last_write_at"] = time.monotonic()
            # One (time, cumulative bytes, cumulative meshes) sample per mesh — the
            # series `_load_rate` differentiates for the live delivery speed.
            state["bundle_samples"].append(
                (state["bundle_last_write_at"], state["bundle_bytes"], state["bundle_meshes"])
            )
        await resp.write_eof()
        state["bundle_done_at"] = time.monotonic()
        return resp

    async def frames(request):
        _check_token(request)
        body = await request.read()
        if body[:4] == _SZC1_MAGIC:
            body = await asyncio.to_thread(_decompress_frame_batch, body)
        try:
            parsed = _parse_frame_batch(body)
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        state["last_frame_at"] = time.monotonic()
        batch = [f for f in parsed if f[0] in state["pending"]]
        if batch:
            vids = [f[0] for f in batch]
            fut = asyncio.wrap_future(
                state["pool"].submit(_encode_batch, state["refs_dir"], body, batch)
            )
            state["outstanding"].add(fut)

            def _encoded(f: asyncio.Future, vids: list[str] = vids) -> None:
                state["outstanding"].discard(f)
                exc = f.exception()
                if exc is not None:
                    state["encode_errors"].append(f"{vids[0]}…{vids[-1]}: {exc}")
                    return
                # A view is DONE when its frame is on disk, so time the completions
                # here — that is what the heartbeat's img/s and the summary's
                # `img_per_s` are both measured from.
                now = time.monotonic()
                state["last_frame_at"] = now
                if state["first_frame_at"] is None:
                    state["first_frame_at"] = now
                for vid in vids:
                    state["pending"].discard(vid)
                    state["frame_times"].append(now)
                state["done"] += len(vids)

            fut.add_done_callback(_encoded)
        # Flow control: a full encode backlog backpressures the renderer.
        while len(state["outstanding"]) > _ENCODE_BACKLOG:
            await asyncio.wait(
                list(state["outstanding"]), return_when=asyncio.FIRST_COMPLETED
            )
        return web.json_response({"done": state["done"], "total": state["total"]})

    async def finish(request):
        _check_token(request)
        payload = await request.json() if request.can_read_body else {}
        if payload.get("renderer"):
            state["renderer"] = payload["renderer"]
        if payload.get("stats"):
            state["stats"] = payload["stats"]
            _print_capture_stats("[refs]", payload["stats"])
        if payload.get("error"):
            state["client_error"] = str(payload["error"])
        while state["outstanding"]:
            await asyncio.wait(
                list(state["outstanding"]), return_when=asyncio.ALL_COMPLETED
            )
        state["finished"] = True
        return web.json_response({"missing": len(state["pending"])})

    assets: Path = state["assets_dir"]

    async def page(request):
        return web.FileResponse(assets / "splatcapture.html")

    @web.middleware
    async def _log_requests(request, handler):
        """Print every loopback request → status to the container log, so the
        request SEQUENCE (page → three modules → manifest → bundle → frames)
        pinpoints where a stalled capture got stuck. Also stamps the last time the
        page asked for ANYTHING, which is what `_PAGE_SILENT_S` measures from."""
        state["last_http_at"] = time.monotonic()
        try:
            resp = await handler(request)
        except web.HTTPException as exc:
            print(f"[host] {request.method} {request.path} -> {exc.status}", flush=True)
            raise
        except Exception as exc:
            print(f"[host] {request.method} {request.path} -> 500 {exc}", flush=True)
            raise
        # Skip the high-frequency frame POSTs once flowing; log everything else.
        if not request.path.endswith("/frames"):
            print(f"[host] {request.method} {request.path} -> {resp.status}", flush=True)
        return resp

    stage = f"/runs/{state['run']}/splat/stage5/{state['slot']}/{state['model']}"
    app = web.Application(
        middlewares=[_log_requests], client_max_size=256 * 1024 * 1024
    )
    app.router.add_get(f"{stage}/manifest", manifest)
    app.router.add_post(f"{stage}/frames", frames)
    app.router.add_post(f"{stage}/finish", finish)
    app.router.add_get("/cameras.json", cameras)
    app.router.add_get("/bundle", bundle)
    app.router.add_get("/splatcapture.html", page)
    app.router.add_static("/js", assets / "js")
    app.router.add_static("/vendor/three", assets / "vendor" / "three")
    return app


def _serve(state: dict[str, Any]) -> Callable[[], None]:
    """Run the loopback host on a daemon thread; returns a stopper."""
    from aiohttp import web

    loop = asyncio.new_event_loop()
    runner = web.AppRunner(_make_app(state))

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", _PORT)
        loop.run_until_complete(site.start())
        loop.run_forever()

    thread = threading.Thread(target=_run, name="capture-host", daemon=True)
    thread.start()

    def _stop() -> None:
        async def _teardown() -> None:
            await runner.cleanup()
            loop.stop()

        asyncio.run_coroutine_threadsafe(_teardown(), loop)
        thread.join(timeout=10)

    return _stop


# --- chromium --------------------------------------------------------------------


def _chromium_binary() -> str:
    override = os.environ.get("STARSHOT_CHROME_BIN")
    if override:
        return override
    for name in ("chromium", "chromium-browser", "google-chrome", "chrome"):
        hit = shutil.which(name)
        if hit:
            return hit
    hits = glob.glob(
        os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux/chrome")
    )
    if hits:
        return sorted(hits)[-1]
    raise RuntimeError("no Chromium binary (set STARSHOT_CHROME_BIN)")

_COMMON_FLAGS = [
    "--headless=new",
    "--no-sandbox",                    # container root
    "--disable-dev-shm-usage",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-extensions",
    "--mute-audio",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--window-size=480,360",
    # Exposes EXT_disjoint_timer_query_webgl2, which the capture page uses for
    # per-pass GPU time (see `renderProfile` in capturecore.js). Desktop Chrome
    # already enables it via a driver-bug workaround, so this is belt-and-braces —
    # the page reports `gpu_timing: unavailable` and falls back to CPU-only pass
    # accounting if it is missing.
    "--enable-webgl-developer-extensions",
]
# GPU attempt: ANGLE over Vulkan against the NVIDIA driver. Translate GL ES →
# Vulkan, enable the Vulkan backend, bypass Chrome's GPU + GPU-process-sandbox
# blocklists in the container, and DISABLE THE GPU WATCHDOG (headless server GPU
# init can trip it → context loss). On a container without the NVIDIA graphics libs
# this resolves to Mesa llvmpipe (CPU); with them it uses the GPU.
_GPU_FLAGS = [
    "--use-angle=vulkan",
    "--enable-features=Vulkan",
    "--ignore-gpu-blocklist",
    "--disable-gpu-sandbox",
    "--disable-gpu-watchdog",
]
# Drop the Vulkan WSI surface extensions (see `_RENDER_MODES`). Presentation falls
# back to a bit blit, which is all a headless capture needs — it reads pixels out of
# its own framebuffers and never presents to a screen.
_GPU_NOSURFACE_FLAGS = [*_GPU_FLAGS, "--disable-vulkan-surface"]
# CPU attempt: conformant software WebGL — identical pixels, slower. The page
# rejects software GL unless the URL carries &force=1 (added by the caller).
#
# `--enable-unsafe-swiftshader` is REQUIRED, not optional. Chromium deprecated the
# automatic SwiftShader fallback for WebGL: without this flag a modern build will
# refuse to hand a software context to `getContext("webgl2")` even when
# `--use-angle=swiftshader` is set, and the request does not fail cleanly — the page
# blocks in context creation and simply never starts. That is indistinguishable in
# the logs from a slow scene load, and it is why the SwiftShader retry stopped
# rescuing runs whose Vulkan init had failed: BOTH backends produced no context, and
# each burned the full boot grace in silence.
_SW_FLAGS = [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
]
# The flag set each `_RENDER_MODES` entry launches with.
_MODE_FLAGS = {
    "vulkan": _GPU_FLAGS,
    "vulkan-nosurface": _GPU_NOSURFACE_FLAGS,
    "swiftshader": _SW_FLAGS,
}


def chromium_version() -> str:
    """`chromium --version`, or a reason it could not be read.

    Worth recording: the capture's GL backend is entirely at the mercy of which
    Chromium the image's unpinned `apt_install` happened to fetch, and a silent
    version bump has already broken it once — a newer build both asked Vulkan for an
    instance extension the container could not supply AND began refusing SwiftShader
    without `--enable-unsafe-swiftshader`, so both backends died at once. With the
    version in the summary, "it worked yesterday" becomes checkable."""
    try:
        r = subprocess.run(
            [_chromium_binary(), "--version"],
            capture_output=True, text=True, timeout=30,
        )
        return (r.stdout or r.stderr).strip() or "(no output)"
    except Exception as exc:  # diagnostic only — never fail a capture over it
        return f"<{type(exc).__name__}: {exc}>"


def _launch(url: str, mode: str, profile_dir: Path) -> subprocess.Popen:
    args = [
        _chromium_binary(),
        *_COMMON_FLAGS,
        *_MODE_FLAGS[mode],
        f"--user-data-dir={profile_dir}",
    ]
    if _CAPTURE_DEBUG:
        # Route the page's console + Chrome warnings/errors to stderr, which we
        # let flow to the container log (below) — visibility into a stuck browser.
        # (No `--v=1`: verbose GPU-internal spam would bury the signal over a
        # multi-thousand-view render.)
        args.append("--enable-logging=stderr")
    args.append(url)
    # Debug: inherit the container's stdout/stderr so the browser's output lands
    # in `modal app logs`. Otherwise discard it.
    io = None if _CAPTURE_DEBUG else subprocess.DEVNULL
    return subprocess.Popen(args, stdout=io, stderr=io)


# --- orchestrator ------------------------------------------------------------------


def _tier_bytes(cas_dir: Path, tier: dict[str, str]) -> int:
    """Total size of the tier's CAS blobs — the denominator for the load
    heartbeat, and a number worth having in the log BEFORE a render starts: it is
    the single best predictor of how long the client will sit loading, and an
    outlier tier is then visible immediately rather than as a timeout."""
    total = 0
    for sha in tier.values():
        with contextlib.suppress(OSError):
            total += (cas_dir / sha[:2] / f"{sha}.glb").stat().st_size
    return total


def _phase(state: dict[str, Any]) -> str:
    """Which of the watchdog's phases the capture is in (see the constants)."""
    if state["last_frame_at"] is not None or state["done"]:
        return "render"
    if state["bundle_done_at"] is not None:
        return "build"
    if state["bundle_last_write_at"] is not None:
        return "load"
    return "boot"


def _capture_rate(state: dict[str, Any]) -> tuple[float, float]:
    """(live images/second over the trailing window, session mean since the first
    frame). Computed at read time so the window is always current, and the live
    figure decays to zero within one window after frames stop — which is what makes
    it useful next to the watchdog rather than merely decorative."""
    now = time.monotonic()
    buf: deque[float] = state["frame_times"]
    while buf and now - buf[0] > _RATE_WINDOW_S:
        buf.popleft()
    first = state["first_frame_at"]
    if first is None:
        return 0.0, 0.0
    # During warm-up divide by the elapsed time, not the full window, so the first
    # couple of polls aren't understated.
    window = min(_RATE_WINDOW_S, max(now - first, 1e-3))
    elapsed = now - first
    avg = state["done"] / elapsed if elapsed > 1e-3 else 0.0
    return len(buf) / window, avg


def _load_rate(state: dict[str, Any]) -> tuple[float, float, float]:
    """Bundle-delivery speed: (live MB/s, mean MB/s, live meshes/s). Differentiates
    the per-mesh sample series the `/bundle` handler records, so the live figures
    reflect what the CLIENT is consuming — aiohttp's write backpressures against the
    transport, so those writes complete at the rate the browser drains them, not at
    the rate we can read the CAS.

    The load phase is where a big cell spends minutes before frame 1, and a stalled
    download and a merely slow one look identical on a `133/807 meshes` counter.

    The live window is measured against NOW, not against the newest sample, which is
    the whole point: if delivery stopped twenty seconds ago the numerator is zero and
    the rate reads 0, whereas differencing the last two samples would keep reporting
    the healthy rate it had before it died."""
    buf: deque[tuple[float, int, int]] = state["bundle_samples"]
    now = time.monotonic()
    # Keep the newest sample that has already fallen out of the window: it is the
    # baseline the window's delta is measured from.
    while len(buf) >= 2 and now - buf[1][0] > _RATE_WINDOW_S:
        buf.popleft()
    base_t, base_b, base_m = buf[0] if buf else (now, state["bundle_bytes"], state["bundle_meshes"])
    dt = now - base_t
    live_mb = (state["bundle_bytes"] - base_b) / dt / 1e6 if dt > 1e-3 else 0.0
    live_mesh = (state["bundle_meshes"] - base_m) / dt if dt > 1e-3 else 0.0
    started = state.get("bundle_started_at")
    elapsed = (now - started) if started else 0.0
    avg_mb = state["bundle_bytes"] / elapsed / 1e6 if elapsed > 1e-3 else 0.0
    return live_mb, avg_mb, live_mesh


def _fmt_eta(seconds: float) -> str:
    """Compact ETA: `48s`, `12m30s`, `2h05m`."""
    s = int(max(seconds, 0))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _phase_msg(state: dict[str, Any], mode: str) -> str:
    """Heartbeat text for the current phase. Before the first frame the useful
    number is how much of the SCENE has arrived — `0/N views` is constant through
    the entire load and reads as a hang. Once frames flow the useful number is the
    THROUGHPUT: a capture that has slowed by half is otherwise indistinguishable
    from one that is merely large, and the watchdog only fires on a full stop."""
    phase = _phase(state)
    if phase == "load":
        got, total = state["bundle_bytes"], state["bundle_total_bytes"]
        span = (
            f"{got / 1e9:.2f}/{total / 1e9:.2f} GB" if total else f"{got / 1e9:.2f} GB"
        )
        live_mb, avg_mb, live_mesh = _load_rate(state)
        # ETA off the MEAN rather than the live rate: mesh sizes vary by orders of
        # magnitude, so a live-rate ETA swings wildly on a series this coarse.
        eta = ""
        if total and avg_mb > 1e-6:
            eta = f" · eta {_fmt_eta(max(total - got, 0) / (avg_mb * 1e6))}"
        return (
            f"load[{mode}] {state['bundle_meshes']}/{len(state['tier'])} meshes · {span}"
            f" · {live_mb:.0f} MB/s (avg {avg_mb:.0f}) · {live_mesh:.1f} mesh/s{eta}"
        )
    if phase == "build":
        return f"build[{mode}] scene delivered, parsing + baking"
    if phase == "boot":
        # Naming which half of boot we are in turns a silent 3-minute timeout into
        # an immediate diagnosis: still "launching" means the browser or the page
        # never came up (missing asset, dead GPU process), while "page up" means our
        # JS is running and stuck somewhere before it asked for the bundle.
        return (
            f"boot[{mode}] page up, fetching plan + bundle"
            if state.get("manifest_at")
            else f"boot[{mode}] launching browser, page has not requested the manifest yet"
        )
    if phase == "render":
        rate, avg = _capture_rate(state)
        # `pending` is the authoritative outstanding set — on a resumed render
        # `total - done` would count views that were already on disk before this
        # session started.
        left = len(state["pending"])
        eta = f" · eta {_fmt_eta(left / avg)}" if avg > 1e-6 else ""
        # "ref/s" to match the stage's own vocabulary — these are reference frames,
        # and the heartbeat prefix is already `[refs]`. The summary field stays
        # `img_per_s`, which is what the local server records and what every
        # existing status.json already carries, so runs remain comparable.
        return f"render[{mode}] {rate:.1f} ref/s (avg {avg:.1f}){eta}"
    return f"{phase}[{mode}]"


def _publish_samples(
    state: dict[str, Any], sample_dir: Path, commit: Callable[[], None] | None
) -> None:
    """Decode the first `_SAMPLE_COUNT` completed SZF frames to PNG and write
    them to `sample_dir`, committing once per new batch — a live debug window
    into the render. Frames are written atomically, so any `.szf` present is
    complete; a still-undecodable one is marked seen and skipped."""
    published: set[str] = state["sampled"]
    if len(published) >= _SAMPLE_COUNT:
        return
    frames_dir = state["refs_dir"] / stage5.FRAMES_DIRNAME
    new = 0
    for f in sorted(frames_dir.glob("*.szf")):
        if len(published) >= _SAMPLE_COUNT:
            break
        if f.name in published:
            continue
        published.add(f.name)
        try:
            png = stage5.frame_preview_png(f)
        except Exception:
            continue  # torn/partial — skip (already marked seen)
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / f"{f.stem}.png").write_bytes(png)
        new += 1
    if new and commit is not None:
        commit()


def capture_refs(
    *,
    run: str,
    slot: str,
    model: str,
    cameras_path: Path,
    tier: dict[str, str],
    cas_dir: Path,
    refs_dir: Path,
    assets_dir: Path,
    sample_dir: Path | None = None,
    commit: Callable[[], None] | None = None,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Render every pending view of the Stage-4 plan into `refs_dir` (SZF
    frames + transforms.json), resuming from whatever is already on disk.
    `tier` maps node_id → CAS sha256 (the SMB1 bundle source). Returns a
    summary (views, rendered count, renderer string, seconds)."""
    plan = stage5.load_camera_plan(cameras_path)
    views = stage5.enumerate_views(plan)
    # Capture-settings guard (same mechanism the server's local path uses): if the
    # frames already on disk were rendered under different settings (lighting or
    # COLOR_PIPELINE — e.g. the reflective→matte switch), drop them so the resumed
    # render re-does every view under one consistent look. Keyed to
    # stage5.capture_meta(), so a COLOR_PIPELINE bump self-invalidates the frames
    # the caller synced in from the Volume.
    stage5.reconcile_capture_meta(refs_dir)
    pending = stage5.pending_views(refs_dir, views)
    intr = plan["intrinsics"]
    t0 = time.perf_counter()
    rendered = 0
    renderer = None
    stats = None
    img_per_s = None
    browser = None

    if pending:
        (refs_dir / stage5.FRAMES_DIRNAME).mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "run": run, "slot": slot, "model": model,
            "resolution": int(intr["resolution"]),
            "near": float(intr["near"]),
            "far": float(intr["far"]),
            "fov_deg": float(intr["fov_deg"]),
            "cameras_path": cameras_path,
            "tier": tier,
            "cas_dir": cas_dir,
            "refs_dir": refs_dir,
            "assets_dir": assets_dir,
            "pending": {v["id"] for v in pending},
            "total": len(views),
            "done": 0,
            "outstanding": set(),
            "encode_errors": [],
            "pool": ThreadPoolExecutor(
                max_workers=_ENCODE_WORKERS, thread_name_prefix="szf-encode"
            ),
            "last_frame_at": None,
            # Set when the page asks for its work order — the signal that its module
            # graph evaluated and its JS is running (see `_phase_msg`).
            "manifest_at": None,
            # Last loopback request of any kind, from the logging middleware. Paired
            # with `manifest_at` it separates "page still fetching its modules" from
            # "page loaded and then went silent" (see `_PAGE_SILENT_S`).
            "last_http_at": None,
            # Throughput tracking (see `_capture_rate`): a trailing-window deque of
            # frame-completion times for the live rate, plus the first-frame stamp
            # that `done` is averaged over for the session mean.
            "frame_times": deque(),
            "first_frame_at": None,
            "finished": False,
            "renderer": None,
            "client_error": None,
            "stats": None,
            "sampled": set(),  # frame filenames already published as PNG samples
            # Bundle delivery, written by the /bundle handler and read by the
            # watchdog — the difference between "loading a big scene" and "hung".
            "bundle_total_bytes": _tier_bytes(cas_dir, tier),
            "bundle_bytes": 0,
            "bundle_meshes": 0,
            "bundle_last_write_at": None,
            "bundle_done_at": None,
            # Delivery-speed series for `_load_rate`: one (time, cumulative bytes,
            # cumulative meshes) sample per mesh, pruned to the rate window.
            "bundle_samples": deque(),
            "bundle_started_at": None,
            "timeout": None,
        }
        if progress is not None:
            progress(
                0, len(views),
                f"tier: {len(tier)} meshes · "
                f"{state['bundle_total_bytes'] / 1e9:.2f} GB",
            )
        browser = chromium_version()
        print(f"[host] browser: {browser}", flush=True)
        stop = _serve(state)
        try:
            for mode in _RENDER_MODES:
                if not state["pending"]:
                    break
                state["finished"] = False
                state["client_error"] = None
                # Each launch fetches the bundle again, so the delivery clock and
                # counters restart with it.
                state["bundle_bytes"] = 0
                state["bundle_meshes"] = 0
                state["bundle_last_write_at"] = None
                state["bundle_done_at"] = None
                state["bundle_samples"].clear()
                state["bundle_started_at"] = None
                state["manifest_at"] = None
                state["last_http_at"] = None
                state["timeout"] = None
                # force=1 always: the page refuses a software backend otherwise,
                # which would break the SwiftShader fallback. A no-op on the
                # hardware path (see the docstring). nozip=1 unless
                # STARSHOT_CAPTURE_GZIP is set: on loopback the transport codec is
                # pure cost (see `_WIRE_GZIP`).
                url = (
                    f"http://127.0.0.1:{_PORT}/splatcapture.html"
                    f"?api=http://127.0.0.1:{_PORT}&run={run}&slot={slot}"
                    f"&model={model}&token={_TOKEN}&force=1"
                    + ("" if _WIRE_GZIP else "&nozip=1")
                )
                profile = Path(f"/tmp/splatcap-profile-{mode}")
                shutil.rmtree(profile, ignore_errors=True)
                started = time.monotonic()
                proc = _launch(url, mode, profile)
                try:
                    while True:
                        time.sleep(_POLL_S)
                        if progress is not None:
                            progress(
                                state["done"], state["total"], _phase_msg(state, mode)
                            )
                        if sample_dir is not None:
                            _publish_samples(state, sample_dir, commit)
                        if state["encode_errors"]:
                            raise RuntimeError(
                                f"frame encode failed: {state['encode_errors'][0]}"
                            )
                        if state["finished"] or not state["pending"]:
                            break
                        if state["client_error"] is not None:
                            break  # page bailed (e.g. software GL refusal) — next mode
                        if proc.poll() is not None:
                            break  # browser died — next mode
                        # Time the CURRENT phase from its own last sign of life
                        # (see the watchdog constants). While the bundle streams,
                        # every write is proof of a live client, so a big scene can
                        # take as long as it needs as long as bytes keep moving.
                        phase = _phase(state)
                        since, limit = {
                            "render": (state["last_frame_at"], _STALL_S),
                            "build": (state["bundle_done_at"], _FIRST_FRAME_GRACE_S),
                            "load": (state["bundle_last_write_at"], _BUNDLE_STALL_S),
                            "boot": (started, _BOOT_GRACE_S),
                        }[phase]
                        why = f"{phase} phase"
                        # A page that fetched its assets and then asked for nothing
                        # else is wedged rather than slow — almost always because
                        # `getContext("webgl2")` never returned (a synchronous call,
                        # so the page can neither continue nor report). Cut it short
                        # and say so, instead of spending the whole boot grace.
                        if (
                            phase == "boot"
                            and state["manifest_at"] is None
                            and state["last_http_at"] is not None
                        ):
                            since, limit = state["last_http_at"], _PAGE_SILENT_S
                            why = (
                                "page loaded its modules but never requested the "
                                "manifest — no WebGL context (check the gpu-process "
                                "lines above)"
                            )
                        idle = time.monotonic() - (since if since else started)
                        if idle > limit:
                            state["timeout"] = (
                                f"{why} made no progress for "
                                f"{idle:.0f}s (limit {limit:.0f}s)"
                            )
                            print(
                                f"[host] watchdog: {mode} — {state['timeout']}",
                                flush=True,
                            )
                            break  # wedged — next mode
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                    shutil.rmtree(profile, ignore_errors=True)
        finally:
            state["pool"].shutdown(wait=True)
            stop()
        renderer = state["renderer"]
        rendered = state["done"]
        stats = state["stats"]
        # The HOST's own throughput measurement (frames landed on disk ÷ time since
        # the first one), recorded beside the page's self-reported `capture_stats`.
        # Two independent numbers are worth having: the page's excludes whatever it
        # was blocked on before its first frame, and a shard that dies mid-run still
        # leaves this one behind.
        first = state["first_frame_at"]
        if first is not None and state["done"]:
            elapsed = time.monotonic() - first
            if elapsed > 1e-3:
                img_per_s = round(state["done"] / elapsed, 2)
        if state["pending"]:
            # Name the PHASE that gave up and how much of the scene had landed.
            # "stalled on both backends" alone reads like a capacity wall no
            # matter which of boot / load / build actually ran out of patience.
            why = (
                state["client_error"]
                or state["timeout"]
                or "renderer died before finishing"
            )
            raise RuntimeError(
                f"stage 5: {len(state['pending'])} view(s) still missing — {why} "
                f"[tier {len(tier)} meshes / "
                f"{state['bundle_total_bytes'] / 1e9:.2f} GB, "
                f"{state['bundle_meshes']} delivered on the last attempt]"
            )

    # Deterministic from the plan alone (mirrors the server's finalize).
    K = stage5.intrinsics_matrix(int(intr["resolution"]), float(intr["fov_deg"]))
    stage5.write_transforms(
        refs_dir, K, int(intr["resolution"]),
        float(intr["near"]), float(intr["far"]),
        stage5.reference_frames(views, int(intr["resolution"])),
    )
    return {
        "views": len(views),
        "rendered": rendered,
        "resumed": len(views) - len(pending),
        "renderer": renderer,
        "browser": browser,
        "capture_stats": stats,
        "img_per_s": img_per_s,
        "wire_gzip": _WIRE_GZIP,
        "seconds": round(time.perf_counter() - t0, 1),
    }
