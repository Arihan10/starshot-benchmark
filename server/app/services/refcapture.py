"""Stage-5 reference-capture plumbing (see splat/stage5.py for the contract).

The renderer is a browser page (client/public/js/splatcapture.js) running the
debug viewer's WebGL stack against the cell's splat asset tier. This module is
the server half the stage-5 routes lean on:

  * **browser** — find a local Chromium-based browser (Chrome/Edge/Chromium)
    and launch it headless at the capture URL with an isolated profile. The
    page does all the work over plain HTTP, so nothing here speaks CDP; the
    process handle only exists for supervision (alive? kill on job end).
  * **frame batches** — parse the page's binary `SRF1` POST bodies into
    (view id, resolution, rgba, depth-codes) tuples.
  * **encode pool** — a process pool running splat.stage5.write_reference_frame
    (numpy + zstd, pure CPU, ~3-12 ms/view) so frame encoding never blocks the
    event loop or the GPU: the browser renders, the pool encodes, in parallel.
"""

from __future__ import annotations

import gzip
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

# splat/ lives at the repo root, outside the server's `app` package (mirrors
# routes.py).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FRAME_BATCH_MAGIC = b"SRF1"
# The capture page gzip-wraps the SRF1 batch (SZC1 marker) so the ~6 MB/view POST
# payload doesn't pace the single-threaded ingest; `decompress_frame_batch` inflates
# it back to the byte-identical SRF1 body, so parse + encode stay unchanged.
FRAME_BATCH_GZIP_MAGIC = b"SZC1"

# Frame encode workers — THREADS, deliberately not processes: zstd releases the
# GIL during compression (both the stdlib module and python-zstandard) and the
# numpy filter ops are C-level, so threads parallelize the encode while sharing
# the POST body by REFERENCE. A ProcessPoolExecutor here was profiled spending
# ~0.7 cores in its feeder thread pickling frame bytes into the worker pipe —
# the pipe, not the encoding, was the pipeline's ~100 views/s wall.
_ENCODE_WORKERS = max(
    2, int(os.environ.get("STARSHOT_CAPTURE_ENCODERS", min(8, (os.cpu_count() or 4) - 2)))
)

_pool: ThreadPoolExecutor | None = None


# --- browser ---------------------------------------------------------------------

# Chromium-based binaries to try, in order: an explicit override always wins,
# then PATH names, then the standard Windows/macOS install locations (the GPU
# box is Windows; Edge ships with it, so a bare box still works).
_BROWSER_ENV = "STARSHOT_CHROME_BIN"
_BROWSER_NAMES = ("chrome", "google-chrome", "chromium", "chromium-browser", "msedge")


def _windows_candidates() -> list[str]:
    roots = [
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        os.environ.get("LocalAppData", ""),
    ]
    rels = [
        r"Google\Chrome\Application\chrome.exe",
        r"Microsoft\Edge\Application\msedge.exe",
    ]
    return [os.path.join(root, rel) for root in roots if root for rel in rels]


def find_browser() -> str | None:
    """Path to a launchable Chromium-based browser, or None."""
    override = os.environ.get(_BROWSER_ENV)
    if override:
        return override if Path(override).is_file() else None
    for name in _BROWSER_NAMES:
        hit = shutil.which(name)
        if hit:
            return hit
    for cand in _windows_candidates():
        if Path(cand).is_file():
            return cand
    if sys.platform == "darwin":
        mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if Path(mac).is_file():
            return mac
    return None


def launch_capture_browser(url: str) -> tuple[subprocess.Popen, Path]:
    """Launch a headless Chromium at `url` with an isolated throwaway profile.
    Returns (process, profile_dir) — the caller supervises the process and calls
    `terminate_browser` when the job ends. Raises with a set-STARSHOT_CHROME_BIN /
    open-manually hint when no browser is found."""
    binary = find_browser()
    if binary is None:
        raise RuntimeError(
            f"no Chromium-based browser found (set {_BROWSER_ENV}), or open the "
            f"capture URL in any browser manually: {url}"
        )
    profile_dir = Path(tempfile.mkdtemp(prefix="splatcap-profile-"))
    args = [
        binary,
        "--headless=new",  # new headless keeps the real GPU (old mode is software-only)
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--mute-audio",
        # The capture loop must not be deprioritised the way background tabs are.
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows",
        "--window-size=480,360",
        url,
    ]
    if sys.platform == "win32":
        args.insert(2, "--use-angle=d3d11")  # the GPU-backed ANGLE path on Windows
    proc = subprocess.Popen(
        args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return proc, profile_dir


def terminate_browser(proc: subprocess.Popen | None, profile_dir: Path | None) -> None:
    """Best-effort teardown of a capture browser + its throwaway profile."""
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    if profile_dir is not None:
        shutil.rmtree(profile_dir, ignore_errors=True)


# --- frame batches -----------------------------------------------------------------


def decompress_frame_batch(body: bytes) -> bytes:
    """Inflate a gzip-wrapped (SZC1) capture batch back to its raw SRF1 bytes.

    The page compresses the POST payload client-side (splatcapture-worker.js) so the
    single-threaded ingest isn't the wall; the inflated bytes are byte-identical to
    the page's SRF1 buffer, so `parse_frame_batch` + the encode pool are unchanged.
    Meant to run off the event loop (gzip releases the GIL); raw SRF1 posts skip it."""
    return gzip.decompress(body[len(FRAME_BATCH_GZIP_MAGIC):])


def parse_frame_batch(body: bytes) -> list[tuple[str, int, int, int]]:
    """Parse one capture-page `SRF1` POST body → [(view id, resolution,
    rgba_offset, depth_offset)] — DESCRIPTORS ONLY, no pixel copies. Framing
    (little-endian), mirroring splatcapture-worker.js packBatch:

        b"SRF1" <u32 count>
        repeat: <u32 id_len><id utf-8><u32 w><u32 h><rgba w*h*4><depth u16le w*h>

    The caller hands the untouched `body` + descriptors to the encode pool
    (`submit_encode_batch`), where the worker does the slicing — this parse runs
    on the server's single event-loop thread, which profiling showed saturating
    a core on byte copies at ~100 views/s. Raises ValueError on any structural
    mismatch (torn body, wrong magic)."""
    if body[:4] != FRAME_BATCH_MAGIC:
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


# --- encode pool ---------------------------------------------------------------------


def _encode_batch(
    refs_dir: str, body: bytes, frames: list[tuple[str, int, int, int]]
) -> list[str]:
    """Pool worker: slice each view's planes out of the raw POST `body` (by the
    descriptors from `parse_frame_batch`, zero-copy via memoryview) and encode +
    atomically persist its SZF frame. The event loop never touches pixels."""
    from splat import stage5

    view = memoryview(body)
    out: list[str] = []
    for vid, resolution, rgba_off, depth_off in frames:
        rgba = view[rgba_off : rgba_off + resolution * resolution * 4]
        depth = view[depth_off : depth_off + resolution * resolution * 2]
        stage5.write_reference_frame(Path(refs_dir), vid, resolution, rgba, depth)
        out.append(vid)
    return out


def _ensure_pool() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(
            max_workers=_ENCODE_WORKERS, thread_name_prefix="szf-encode"
        )
    return _pool


def submit_encode_batch(
    refs_dir: Path, body: bytes, frames: list[tuple[str, int, int, int]]
) -> Future:
    """Queue one POST batch of RAW captured views for SZF encoding on the shared
    thread pool (body shared by reference — no pickling, no pipes). Returns a
    future resolving to the written view ids."""
    return _ensure_pool().submit(_encode_batch, str(refs_dir), body, frames)
