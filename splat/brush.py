"""Stage 6 (alternative trainer) - Brush, driven as a subprocess.

`splat.stage6.train_splat` and this module are interchangeable back-ends for the
same seam: both take the COLMAP model `splat.colmap.export_colmap` writes and
produce `trained.ply`. Which one runs is `spec["train"]["trainer"]` in
`splat/modal_app.py`; everything downstream (quantize, the SOG/ksplat encoders,
the viewers) keys off the `.ply` alone and is unchanged.

WHY A SUBPROCESS: Brush (github.com/ArthurBrussee/brush) is a Rust/wgpu trainer
with no Python API. The pinned release binary is baked into the image
(`modal_app.BRUSH_BIN`) and reaches the GPU through Vulkan - `probe_brush`
confirms wgpu selects the NVIDIA card with `backend: Vulkan`, not a Mesa CPU
rasterizer, and `train_brush` refuses to return a model if it ever does land on
one. Progress comes back by parsing its `log` output, which is why the image
sets `RUST_LOG=info`.

VERSION-COUPLED: the flags here are v0.3.0's and differ from the repo's main
branch (v0.3.0 spells the step count `--total-steps`; main renamed it
`--total-train-iters`), so `modal_app.BRUSH_VERSION` and this module move
together and the version is folded into the stage-6 cache signature.

INIT: Brush seeds from the COLMAP `points3D.txt` - position + colour only, with
its own KNN scale init. It does NOT read the Stage-3 surfel orientations,
opacities or glass alphas the way `stage6.train_splat`'s "surfels" init does.
That is deliberate (it reproduces the validated standalone Brush result), and it
is why `_assert_no_ply` below is load-bearing rather than paranoia.

TWO SHARP EDGES in v0.3.0, both handled as preflight rather than found mid-run:

  * ANY `.ply` inside the dataset tree silently BECOMES the init, overriding
    points3D (`formats/mod.rs`: a lone `.ply` wins outright, otherwise a file
    literally named `init.ply` does). Brush's `--export-path` even defaults to
    the CWD, so an export written into the dataset dir would quietly seed the
    NEXT run from the previous run's output. `_assert_no_ply` refuses to start
    in that state; exports always go to `out_path`'s dir, never the model's.
  * Any image UNDER 16,387 bytes aborts the whole dataset load with "early eof".
    `get_image_data` sizes a sniff buffer at 16,387 and fills it with
    `read_exact`, which errors at EOF instead of settling for a short read - so
    one small frame kills the run, and the error names neither the file nor the
    cause. Real reference frames are far larger (measured min ~42 KB on a
    1024px plan), but a shell view facing empty space compresses to almost
    nothing, so `_pad_small_images` pads those with an inert PNG text chunk.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# progress(done, total, message) - same contract as splat.stage6.
ProgressCb = Callable[[int, int, str], None]

# Brush's image-header sniff buffer (`brush-dataset/src/scene.rs::get_image_data`).
# A file smaller than this fails `read_exact` with UnexpectedEof and takes the
# whole dataset load with it, so every image must be at least this many bytes.
_MIN_IMAGE_BYTES = 16_387

# Records how far a run got, so a preempted container can warm-restart instead of
# throwing the training away. Lives beside `trained.ply` on the Volume.
PROGRESS_NAME = "brush-progress.json"

# The init Brush adopts on a warm restart (see `_resume_from`). Written INTO the
# COLMAP dir, which is exactly the hijack `_assert_no_ply` guards - so it is
# placed deliberately, immediately before launch, and cleared otherwise.
_INIT_PLY = "init.ply"

_RE_VIEWS = re.compile(r"Loaded dataset with (\d+) training, (\d+) eval views")
_RE_POINTS = re.compile(r"Starting from colmap points (\d+)")
_RE_REFINE = re.compile(r"Refine iter (\d+), (\d+) splats")
_RE_EVAL = re.compile(r"Eval iter (\d+): PSNR ([\d.eE+-]+), ssim ([\d.eE+-]+)")
_RE_ADAPTER = re.compile(r'AdapterInfo \{ name: "([^"]+)"')
_RE_SOFTWARE = re.compile(r"(?i)llvmpipe|lavapipe|swiftshader")


@dataclass(frozen=True)
class BrushParams:
    """Brush v0.3.0 knobs. Only the ones the pipeline actually steers are exposed;
    everything else keeps Brush's own defaults, deliberately - the whole point of
    this back-end is the upstream training recipe, so this is not the place to
    re-tune learning rates."""

    # Optimizer steps -> `--total-steps`. Brush's own default, and the same
    # quantity `stage6.TrainParams.iterations` and PostShot's step box mean.
    iterations: int = 30_000

    # Gaussian ceiling -> `--max-splats`. Brush's MCMC-style refinement treats it
    # as an upper bound and may finish under it.
    #
    # IT ALSO SUBSAMPLES THE INIT: `train_stream.rs` runs `data.subsample(max_splats)`
    # on the point cloud BEFORE the first step, so a value below the Stage-3
    # surfel count silently discards part of the mesh-exact surface prior. The
    # client warns at that boundary rather than the number being a free knob.
    max_splats: int = 10_000_000

    # View-dependent colour. Capped at 3 because the downstream reader
    # (`stage6._load_cloud`) decodes at most 15 higher-order coefficients -
    # degree 4 would emit 24 and be silently truncated on the way back in.
    sh_degree: int = 3

    # Long-edge cap on loaded images. Above the reference plan's 1024px, so it is
    # a no-op unless a plan renders larger.
    max_resolution: int = 1920

    # Intermediate `.ply` cadence. Every export overwrites `out_path` (the export
    # name carries no `{iter}` placeholder), so this doubles as the warm-restart
    # granularity: a preemption loses at most this many steps.
    export_every: int = 2_000

    # Hold out every Nth view for PSNR/SSIM. None (the default) trains on
    # everything, matching a plain `brush <dir>` run; set it to get the eval
    # numbers the benchmark dashboard reads, at the cost of those views.
    eval_split_every: int | None = None
    eval_every: int = 2_000

    # Keep every Nth init point. Thins a dense Stage-3 cloud without touching
    # `max_splats` (which also caps the trained model).
    subsample_points: int | None = None

    seed: int = 42

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if self.max_splats < 1:
            raise ValueError(f"max_splats must be >= 1, got {self.max_splats}")
        if not 0 <= self.sh_degree <= 3:
            raise ValueError(
                f"sh_degree must be within 0..3 (the downstream .ply reader decodes "
                f"degree 3 at most), got {self.sh_degree}"
            )

    def as_summary(self) -> dict[str, Any]:
        return asdict(self)


def _assert_no_ply(colmap_dir: Path) -> None:
    """Refuse to train when the dataset tree holds a `.ply`. Brush would adopt it
    as the init in place of `points3D.txt` without failing, so the run would
    quietly train from the wrong seed - the kind of thing only noticed later, as
    an unexplained quality difference between two supposedly identical configs."""
    strays = sorted(p.name for p in colmap_dir.rglob("*.ply"))
    if strays:
        raise RuntimeError(
            f"{colmap_dir} contains {strays} - Brush adopts a .ply in the dataset "
            f"tree as its init, overriding points3D.txt. Exports belong beside "
            f"trained.ply, not in the COLMAP model."
        )


def _pad_small_images(colmap_dir: Path) -> int:
    """Pad any image below Brush's 16,387-byte header-sniff buffer, returning how
    many were rewritten. The padding is an inert PNG `tEXt` chunk, so the decoded
    pixels are bit-identical - this buys past an upstream `read_exact` that treats
    a short read as a fatal error."""
    from PIL import Image, PngImagePlugin

    padded = 0
    for p in sorted(colmap_dir.glob("*.png")):
        size = p.stat().st_size
        if size >= _MIN_IMAGE_BYTES:
            continue
        meta = PngImagePlugin.PngInfo()
        meta.add_text("pad", "0" * (_MIN_IMAGE_BYTES - size + 64))
        with Image.open(p) as img:
            img.load()
            img.save(p, format="PNG", compress_level=1, pnginfo=meta)
        padded += 1
    return padded


def _progress_path(out_path: Path) -> Path:
    return out_path.parent / PROGRESS_NAME


def _ply_splat_count(path: Path) -> int:
    """Vertex count from a `.ply` header. The log only reports a splat count on
    refinement steps (every `refine_every`), so a short run can finish without
    ever printing one - read it off the artifact instead of reporting zero."""
    with Path(path).open("rb") as f:
        head = f.read(1 << 16)
    cut = head.find(b"end_header")
    for line in head[: cut if cut >= 0 else len(head)].split(b"\n"):
        if line.startswith(b"element vertex"):
            return int(line.split()[2])
    return 0


def _resume_from(out_path: Path, colmap_dir: Path, params: BrushParams) -> int:
    """Set up a warm restart after a preemption, returning the step to resume at
    (0 for a cold start, which also clears any stale init).

    Brush v0.3.0 checkpoints no optimizer state, so `--start-iter` on its own
    would just fast-forward the schedule while still seeding from `points3D` -
    strictly worse than starting over. A real resume needs the partially trained
    model back in as the init, which is what Brush's `init.ply` override is for:
    the last export becomes the seed and the schedule picks up where it stopped.
    Only the Adam moments are lost.

    Guarded on the params signature so a resume can never splice a run under one
    configuration onto a run under another."""
    marker = _progress_path(out_path)
    if not (marker.is_file() and out_path.is_file()):
        return 0
    try:
        rec = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if rec.get("params") != params.as_summary():
        return 0
    step = int(rec.get("iter", 0))
    if step <= 0 or step >= params.iterations:
        return 0
    # Copy rather than link: Brush walks the dataset dir, and the export keeps
    # being rewritten at `out_path` while training continues.
    (colmap_dir / _INIT_PLY).write_bytes(out_path.read_bytes())
    return step


def train_brush(
    *,
    run: str,
    slot: str,
    model: str,
    colmap_dir: Path,
    out_path: Path,
    params: BrushParams = BrushParams(),
    brush_bin: str = "/usr/local/bin/brush",
    resume: bool = True,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Train `colmap_dir` with Brush into `out_path`, streaming progress.

    `out_path` is written directly by Brush (`--export-path` / `--export-name`),
    every `export_every` steps and again on the final step, each export
    overwriting the last - so a run that dies late still leaves a usable model,
    and the file that survives is the final one."""
    colmap_dir, out_path = Path(colmap_dir), Path(out_path)
    if not (colmap_dir / "cameras.txt").is_file():
        raise FileNotFoundError(f"no COLMAP model at {colmap_dir} (cameras.txt missing)")
    if not Path(brush_bin).is_file():
        raise FileNotFoundError(f"brush binary not found at {brush_bin}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(done: int, total: int, msg: str) -> None:
        if progress is not None:
            progress(done, total, msg)

    # Drop our own init BEFORE the stray check: a preempted container is killed
    # outright, so the copy `_resume_from` placed can outlive the run that made
    # it, and the retry would otherwise trip the guard below and never start.
    # Everything after this point is a genuinely foreign `.ply`.
    (colmap_dir / _INIT_PLY).unlink(missing_ok=True)
    _assert_no_ply(colmap_dir)
    if padded := _pad_small_images(colmap_dir):
        emit(0, params.iterations, f"padded {padded} undersized reference image(s)")

    start_iter = _resume_from(out_path, colmap_dir, params) if resume else 0
    if not resume:
        _progress_path(out_path).unlink(missing_ok=True)
    if start_iter:
        emit(start_iter, params.iterations, f"resuming from step {start_iter}")

    cmd = [
        brush_bin, str(colmap_dir),
        "--total-steps", str(params.iterations),
        "--max-splats", str(params.max_splats),
        "--sh-degree", str(params.sh_degree),
        "--max-resolution", str(params.max_resolution),
        "--export-path", str(out_path.parent),
        "--export-name", out_path.name,
        "--export-every", str(params.export_every),
        "--eval-every", str(params.eval_every),
        "--seed", str(params.seed),
    ]
    if start_iter:
        cmd += ["--start-iter", str(start_iter)]
    if params.eval_split_every is not None:
        cmd += ["--eval-split-every", str(params.eval_split_every)]
    if params.subsample_points is not None:
        cmd += ["--subsample-points", str(params.subsample_points)]
    print(f"$ {' '.join(cmd)}", flush=True)

    state: dict[str, Any] = {
        "views": 0, "eval_views": 0, "init_points": 0,
        "splats": 0, "psnr": None, "ssim": None, "device": "", "software": False,
    }
    last_export = start_iter
    t0 = time.monotonic()
    tail: list[str] = []

    # cwd is deliberately NOT the dataset dir (see the module docstring on
    # `--export-path` defaulting to the CWD).
    proc = subprocess.Popen(
        cmd, cwd=str(out_path.parent), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    try:
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            tail.append(line)
            del tail[:-80]

            if not state["device"] and (m := _RE_ADAPTER.search(line)):
                state["device"] = m.group(1)
                state["software"] = bool(_RE_SOFTWARE.search(m.group(1)))
                emit(start_iter, params.iterations, f"gpu: {state['device']}")
            if m := _RE_VIEWS.search(line):
                state["views"], state["eval_views"] = int(m.group(1)), int(m.group(2))
                emit(
                    start_iter, params.iterations,
                    f"{state['views']} train / {state['eval_views']} eval views",
                )
            if m := _RE_POINTS.search(line):
                state["init_points"] = int(m.group(1))
            if m := _RE_EVAL.search(line):
                state["psnr"], state["ssim"] = float(m.group(2)), float(m.group(3))
            if m := _RE_REFINE.search(line):
                step, splats = int(m.group(1)), int(m.group(2))
                state["splats"] = splats
                its = (step - start_iter) / max(time.monotonic() - t0, 1e-6)
                psnr = f" | psnr={state['psnr']:.2f}" if state["psnr"] else ""
                emit(step, params.iterations, f"{splats:,} splats | {its:.1f} it/s{psnr}")
                # Exports fire on multiples of export_every, so everything up to
                # the last multiple below `step` is already on disk. Recording it
                # is what lets a preempted retry pick up instead of restarting.
                done = (step // params.export_every) * params.export_every
                if done > last_export and out_path.is_file():
                    last_export = done
                    _progress_path(out_path).write_text(
                        json.dumps(
                            {"iter": done, "total": params.iterations,
                             "params": params.as_summary()},
                            indent=1,
                        ),
                        encoding="utf-8",
                    )
    finally:
        proc.stdout.close()
        code = proc.wait()

    (colmap_dir / _INIT_PLY).unlink(missing_ok=True)
    if code != 0:
        raise RuntimeError(
            f"brush exited with code {code}\n--- last output ---\n" + "\n".join(tail)
        )
    if not out_path.is_file():
        raise RuntimeError(
            f"brush exited 0 but wrote no {out_path.name} - expected an export at "
            "the final step\n--- last output ---\n" + "\n".join(tail)
        )
    if state["software"]:
        raise RuntimeError(
            f"brush trained on a SOFTWARE rasterizer ({state['device']}) - the "
            "Vulkan loader is not reaching the GPU; check VK_DRIVER_FILES and "
            "NVIDIA_DRIVER_CAPABILITIES in the image"
        )

    # The run completed, so the warm-restart marker has nothing left to resume.
    _progress_path(out_path).unlink(missing_ok=True)
    elapsed = time.monotonic() - t0
    splats = state["splats"] or _ply_splat_count(out_path)
    summary = {
        "trainer": "brush",
        "cell": f"{run}/{slot}/{model}",
        "iterations": params.iterations,
        "resumed_from": start_iter,
        "splats": splats,
        "views": state["views"],
        "eval_views": state["eval_views"],
        "init_points": state["init_points"],
        "psnr": state["psnr"],
        "ssim": state["ssim"],
        "device": state["device"],
        "seconds": round(elapsed, 1),
        "it_per_s": round((params.iterations - start_iter) / max(elapsed, 1e-6), 2),
        "out": str(out_path),
        "bytes": out_path.stat().st_size,
    }
    emit(
        params.iterations, params.iterations,
        f"done - {splats:,} splats in {elapsed / 60:.1f} min",
    )
    print(json.dumps(summary, indent=1), flush=True)
    return summary
