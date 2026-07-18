"""Stage 6 — Splat fine-tune (the training run) via gsplat 2DGS.

Takes the **Stage-3 surfel cloud** (`cloud.ply`) as the initialization and
optimizes it against the **Stage-5 unlit reference renders** (`refs/` — per-view
RGB + depth + alpha + exact OpenCV poses in `transforms.json`) into an optimized
2DGS splat (`trained.ply`). This is where the raw mesh-sampled surfels become a
clean, crisp splat: densification adds Gaussians where the render disagrees with
the reference (appearance detail on geometrically-flat surfaces), pruning removes
redundant ones, and the depth loss suppresses floaters.

WHAT THE FINE-TUNE FIXES (all lighting-independent, so it runs unlit): render-
operator errors that only appear once flat disks are depth-sorted + alpha-blended
through the real rasterizer — silhouettes, thin geometry, zoom-in detail, alpha
edges — plus floaters. The surfel init is ~90% there; this polishes the rest.

CONTRACT (locked, shared with Stages 4/5 — see overview §12):
  * Init map (`cloud.ply`, a Stage-3 2DGS `.ply`, SH degree 0):
      means = xyz; quats = rot_0..3 (wxyz); opacity = logit (pre-sigmoid);
      scale_0/1 = log tangent radii (isotropic); f_dc_0..2 = SH0 colour coeffs.
      gsplat 2DGS wants 3 scales, so a third log-scale = the tangent one is
      synthesized (rendering uses only the two in-plane axes; the third only
      feeds the strategy's scene-scale-normalized grow/prune thresholds).
  * Poses: `transform_matrix` is OpenCV camera-to-world → `viewmats = inv(c2w)`.
    Intrinsics: pinhole `K = [[fl_x,0,cx],[0,fl_y,cy],[0,0,1]]`.
  * Depth: planar camera-space Z (metres), decoded from the SZF frame's log-uint16
    codes via the shared [near, far] (legacy 16-bit PNG / float32 `.npy` sets still
    read as-is). The loss compares the reference against the splat's MEDIAN depth
    (`depth_mode="median"`, the transmittance-0.5 crossing) — the same "nearest
    depth-WRITING surface" statistic Stage 5 stores, which BLEND glass (α ≈ 0.065)
    never shifts; expected (ED) depth would mix the pane into every window pixel
    and steadily push glass opacity toward zero.
  * Colour: sRGB albedo compared directly (no sRGB↔linear); SH degree a config
    flag (0 = unlit, the decided default), raisable later for shiny surfaces.

LOSSES (per view):
  * photometric — full-frame L1 + D-SSIM on RGB vs the unlit reference. Both
    sides are premultiplied-over-black (the reference alpha-blends over a black
    clear colour; gsplat composites with no background), so glass/MASK pixels
    compare like-with-like and background pixels directly penalize floater
    energy;
  * alpha (mask) — L1(render α, reference α): the renderer's exact coverage
    masks make empty space stay empty and glass stay see-through;
  * depth — alpha-gated L1 on the median depth (floater suppression that cannot
    fade glass);
  * 2DGS regularizers — normal consistency (render normals vs normals-from-depth)
    and optional depth distortion.

RESUMABLE: every `ckpt_every` steps the full training state (params + per-param
Adam + means LR schedule + densification-strategy accumulators + step) is written
to `splat/ckpt/` (atomic, most-recent-`ckpt_keep` kept). An interrupted run resumes
from the latest checkpoint and continues to `iterations`; the checkpoints are
deleted once trained.ply is written. Pass `resume=False` to force a fresh run.

CUDA-ONLY: torch + gsplat compile/require CUDA, so this runs on the GPU box, NOT
Apple Silicon. Both are imported LAZILY inside the trainer so the server (which
imports this module for the route + the torch-free PLY/pose IO) stays importable
on a machine without them.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from splat.stage5 import (
    TRANSFORMS_NAME,
    decode_depth_u16,
    load_depth_png,
    load_reference_frame,
)

logging.getLogger("PIL").setLevel(logging.ERROR)

# The optimized splat written under a cell's `splat/` dir (a 2DGS `.ply`, the
# same layout Stage 3 emits so Stage 7/8 + the viewer read it identically).
TRAINED_NAME = "trained.ply"

# progress(done, total, message) — called periodically during training.
ProgressCb = Callable[[int, int, str], None]


@dataclass(frozen=True)
class TrainParams:
    """Stage-6 knobs. Learning rates follow the gsplat/3DGS defaults; the means LR
    is scaled by the scene extent at runtime and decayed exponentially.

    `refine_stop_iter` defaults to None → resolved at runtime to 50% of
    `iterations` (the gsplat standard), so densification scales automatically
    with training length. Pass an explicit int to override."""

    iterations: int = 30_000
    sh_degree: int = 0                 # active SH bands (0 = unlit albedo, decided default)
    sh_degree_interval: int = 1000     # raise the active degree every N steps (if sh_degree > 0)

    # Loss weights.
    ssim_lambda: float = 0.2           # photometric = (1-λ)·L1 + λ·(1-SSIM)
    alpha_lambda: float = 0.5          # L1(render α, reference α)
    depth_lambda: float = 0.5          # alpha-gated depth L1 (metres)
    alpha_gate: float = 0.5            # reference α above this = opaque pixel (depth/normal masks)
    normal_lambda: float = 0.05        # 2DGS normal consistency
    dist_lambda: float = 0.0           # 2DGS depth distortion (off by default; over-flattens bounded scenes)
    normal_start_iter: int = 2000      # let geometry settle before the regularizers bite
    dist_start_iter: int = 1000

    # Learning rates (means_lr is × scene_scale at runtime).
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20.0

    # Densification (gsplat DefaultStrategy, 2DGS gradient key).
    refine: bool = True
    refine_start_iter: int = 500
    refine_stop_iter: int | None = None  # None → int(iterations * 0.5) at runtime
    refine_every: int = 100
    # 0 DISABLES periodic opacity resets (the default here): a reset clamps every
    # opacity to 2·prune_opa, but surfels no training view covers (occluded
    # regions kept at their mesh-true init) receive no gradient and would stay
    # dimmed forever. The pinned gsplat 1.5.3 never fires resets anyway (its
    # trigger `step % reset_every == 0 & step > 0` parses as `… and (0 > 0)`);
    # this makes that behaviour deliberate and upgrade-proof.
    reset_every: int = 0
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    # Keep pruning BELOW the glass init: glass.py panes seed opacity at
    # GLASS_ALPHA = 0.065 (logit −2.67), and the old 0.05 threshold left only
    # ~0.3 logits of drift before a pane surfel was permanently pruned.
    prune_opa: float = 0.03
    absgrad: bool = False
    cap_max: int = 3_000_000           # freeze densification past this many Gaussians (VRAM guard)

    # Runtime. (2DGS renders non-packed: gsplat's packed depth path is broken —
    # its SH branch mis-broadcasts and its precomputed-colour path skips the
    # visible-subset gather, so RGB+ED crashes once a view sees only part of the
    # cloud. Non-packed is also cheap at our per-cell Gaussian counts.)
    near_plane: float = 0.01
    far_plane: float = 1e10
    # Depth statistic the depth loss (and normals-from-depth) compares: "median"
    # = the transmittance-0.5 crossing — the same "nearest depth-WRITING surface"
    # the references store, which BLEND glass (α ≈ 0.065) never shifts, so depth
    # supervision cannot fade panes; "expected" = the alpha-weighted mean (ED) —
    # denser gradients, but it mixes the pane into every window pixel and pushes
    # glass opacity toward 0.
    depth_mode: str = "median"
    seed: int = 0
    eval_max_views: int = 128          # cap final-metric renders (plans can have thousands of views)
    log_every: int = 50                # emit a progress line every N steps
    batch: int = 1                     # views rendered per step — >1 fills the GPU + VRAM headroom
    prefetch: bool = True              # decode/stack the next batch on a background thread (hide disk I/O)

    # Resumable checkpoints: every `ckpt_every` steps write the full training state
    # (params + per-param Adam + densification strategy + step) to `splat/ckpt/`,
    # keeping the most recent `ckpt_keep`. An interrupted run resumes from the latest
    # and continues to `iterations`; they're deleted once trained.ply is written.
    ckpt_every: int = 2000             # 0 disables checkpointing
    ckpt_keep: int = 2

    @property
    def resolved_refine_stop(self) -> int:
        return self.refine_stop_iter if self.refine_stop_iter is not None else int(self.iterations * 0.5)

    def as_summary(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "sh_degree": self.sh_degree,
            "ssim_lambda": self.ssim_lambda,
            "alpha_lambda": self.alpha_lambda,
            "depth_lambda": self.depth_lambda,
            "depth_mode": self.depth_mode,
            "alpha_gate": self.alpha_gate,
            "normal_lambda": self.normal_lambda,
            "dist_lambda": self.dist_lambda,
            "refine": self.refine,
            "refine_stop_iter": self.resolved_refine_stop,
            "reset_every": self.reset_every,
            "grow_grad2d": self.grow_grad2d,
            "prune_opa": self.prune_opa,
            "batch": self.batch,
            "ckpt_every": self.ckpt_every,
        }


# --- torch-free layer (PLY / pose / image IO — runs anywhere) ------------------


def _load_cloud(path: Path) -> dict[str, np.ndarray]:
    """Parse a Stage-3 binary-little-endian float `.ply` into gsplat 2DGS init
    arrays: means [N,3], quats [N,4] (wxyz), opacities [N] (logit), sh0 [N,1,3]
    (SH0 coeffs), scales [N,3] (log; a synthesized third axis when the cloud is
    the 2-scale 2DGS format)."""
    raw = Path(path).read_bytes()
    marker = b"end_header\n"
    cut = raw.find(marker)
    if cut < 0:
        raise ValueError(f"{path}: not a PLY (no end_header)")
    header = raw[:cut].decode("ascii", "replace").splitlines()
    if not header or header[0].strip() != "ply":
        raise ValueError(f"{path}: missing 'ply' magic")
    if not any(l.startswith("format") and "binary_little_endian" in l for l in header):
        raise ValueError(f"{path}: only binary_little_endian is supported")

    count: int | None = None
    props: list[str] = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            count, in_vertex = int(parts[2]), True
        elif parts[:1] == ["element"]:
            in_vertex = False
        elif in_vertex and parts[:1] == ["property"]:
            if parts[1] != "float":
                raise ValueError(f"{path}: non-float property '{parts[2]}' unsupported")
            props.append(parts[2])
    if count is None:
        raise ValueError(f"{path}: no vertex element")

    body = np.frombuffer(raw, dtype="<f4", count=count * len(props), offset=cut + len(marker))
    table = body.reshape(count, len(props))
    col = {name: table[:, i] for i, name in enumerate(props)}

    def stack(*names: str) -> np.ndarray:
        return np.stack([col[n] for n in names], axis=1).astype(np.float32)

    means = stack("x", "y", "z")
    quats = stack("rot_0", "rot_1", "rot_2", "rot_3")
    opacities = col["opacity"].astype(np.float32)
    sh0 = stack("f_dc_0", "f_dc_1", "f_dc_2").reshape(-1, 1, 3)
    if "scale_2" in col:
        scales = stack("scale_0", "scale_1", "scale_2")
    else:
        # 2DGS cloud: synthesize a TINY third log-scale. Rendering ignores it and
        # the strategy's grow/prune thresholds take max() over scales (unchanged),
        # but DefaultStrategy's split() displaces children along ALL three scaled
        # axes — a normal-axis scale equal to the tangent radius would eject every
        # child a full radius off the surface. 1% of the smaller tangent radius
        # keeps splits in-plane.
        two = stack("scale_0", "scale_1")
        third = (two.min(axis=1, keepdims=True) + np.log(0.01)).astype(np.float32)
        scales = np.concatenate([two, third], axis=1)
    return {"means": means, "quats": quats, "opacities": opacities, "sh0": sh0, "scales": scales}


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_alpha(path: Path) -> np.ndarray:
    from PIL import Image

    return (np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0)[..., None]


def _read_depth(view: dict[str, Any]) -> np.ndarray | None:
    """LEGACY reference depth for a view as [H,W,1] float32 planar-Z metres, or
    None: the log-uint16 depth PNG of pre-SZF reference sets (decoded via
    `load_depth_png` with the view's [near, far]), or the even older float32
    `.npy`. Current sets carry depth inside the SZF frame (`_view_arrays`)."""
    dp = view["depth"]
    if dp is None:
        return None
    if dp.suffix == ".png":
        near, far = view.get("depth_near"), view.get("depth_far")
        if near is None or far is None:
            raise ValueError(f"{dp}: PNG depth needs 'near'/'far' in transforms.json")
        return load_depth_png(dp, near, far)[..., None]
    return np.load(dp).astype(np.float32)[..., None]


def _view_arrays(view: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """One view's supervision as numpy: (rgb [H,W,3] in [0,1], alpha [H,W,1] in
    [0,1], depth [H,W,1] planar-Z metres | None). The current Stage-5 contract is
    ONE SZF frame per view (all three planes, one open + zstd decode — ~4x faster
    than the PNG triple it replaced); the legacy branch keeps pre-SZF reference
    sets (PNG triple, `.npy` depth) trainable without regeneration."""
    frame = view.get("frame")
    if frame is not None:
        rgba, codes = load_reference_frame(frame)
        rgb = rgba[..., :3].astype(np.float32) / 255.0
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        near, far = view.get("depth_near"), view.get("depth_far")
        if near is None or far is None:
            raise ValueError(f"{frame}: SZF depth needs 'near'/'far' in transforms.json")
        return rgb, alpha, decode_depth_u16(codes, near, far)[..., None]
    rgb = _load_rgb(view["rgb"])
    alpha = (
        _load_alpha(view["alpha"])
        if view["alpha"] is not None
        else np.ones((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)
    )
    return rgb, alpha, _read_depth(view)


def _fmt_hms(seconds: float) -> str:
    s = int(max(seconds, 0.0))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _quats_to_normals(quats: np.ndarray) -> np.ndarray:
    """Surfel normal = third column of the (wxyz) quaternion's rotation matrix —
    the axis Stage 3 aligned to the surface normal."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    n = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def _encode_trained_ply(
    means: np.ndarray,
    quats: np.ndarray,
    f_dc: np.ndarray,
    opacity: np.ndarray,
    scales2: np.ndarray,
    out_path: Path,
) -> None:
    """Write the trained model as a Stage-3-compatible 2DGS `.ply` (two tangent
    log-scales, SH degree 0). Values are stored raw (opacity as logit, scales as
    log, colour as `f_dc`) exactly as the viewers/Stage 7 expect."""
    n = means.shape[0]
    normals = _quats_to_normals(quats)
    cols = [
        means[:, 0], means[:, 1], means[:, 2],
        normals[:, 0], normals[:, 1], normals[:, 2],
        f_dc[:, 0], f_dc[:, 1], f_dc[:, 2],
        opacity,
        scales2[:, 0], scales2[:, 1],
        quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3],
    ]
    data = np.stack(cols, axis=1).astype("<f4")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n" "property float y\n" "property float z\n"
        "property float nx\n" "property float ny\n" "property float nz\n"
        "property float f_dc_0\n" "property float f_dc_1\n" "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n" "property float scale_1\n"
        "property float rot_0\n" "property float rot_1\n"
        "property float rot_2\n" "property float rot_3\n"
        "end_header\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    tmp.replace(out_path)


# --- CUDA layer (torch + gsplat imported lazily) -------------------------------


def _require_cuda_trainer():  # noqa: ANN202 - returns (torch, torch.nn.functional, gsplat)
    """Import torch + gsplat and assert a CUDA device, or raise a clear error.
    Kept out of module import so the server (Apple Silicon) can import this file."""
    try:
        import torch
        import torch.nn.functional as F
        import gsplat
    except Exception as exc:
        raise RuntimeError(
            "Stage 6 needs torch + gsplat (CUDA). These are CUDA-only and not "
            "installed here — run Stage 6 on the GPU box, not on Apple Silicon. "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 6 needs a CUDA GPU; torch reports none available.")
    return torch, F, gsplat


def _gaussian_window(torch, ksize: int, sigma: float, device, channels: int):  # noqa: ANN001
    coords = torch.arange(ksize, dtype=torch.float32, device=device) - (ksize - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    win2d = (g[:, None] @ g[None, :])[None, None]        # [1,1,k,k]
    return win2d.expand(channels, 1, ksize, ksize).contiguous()


def _ssim(F, x, y, window):  # noqa: ANN001
    """Mean SSIM of two [B,C,H,W] images in [0,1] (11×11 Gaussian window)."""
    ch = x.shape[1]
    pad = window.shape[-1] // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=ch)
    mu_y = F.conv2d(y, window, padding=pad, groups=ch)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x = F.conv2d(x * x, window, padding=pad, groups=ch) - mu_x2
    sig_y = F.conv2d(y * y, window, padding=pad, groups=ch) - mu_y2
    sig_xy = F.conv2d(x * y, window, padding=pad, groups=ch) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2)
    )
    return ssim_map.mean()


def _read_transforms(refs_dir: Path) -> dict[str, Any]:
    tp = Path(refs_dir) / TRANSFORMS_NAME
    if not tp.is_file():
        raise FileNotFoundError(f"reference renders not found: {tp} (run Stage 5)")
    return json.loads(tp.read_text(encoding="utf-8"))


def _render_inputs(torch, splats, active_sh: int):  # noqa: ANN001
    """SH coefficients [N,K,3] + active `sh_degree` for `rasterization_2dgs`.
    Always the SH path (degree 0 = just the DC term) rendered non-packed — the
    only combination that produces depth correctly here: gsplat's packed SH branch
    mis-broadcasts, and its precomputed-colour path never gathers colours to the
    visible subset, so both crash on RGB+ED once a view sees only part of the
    cloud (nnz < N). gsplat applies the +0.5 offset, matching Stage-3's f_dc."""
    sh = splats["sh0"] if "shN" not in splats else torch.cat([splats["sh0"], splats["shN"]], dim=1)
    return sh, active_sh


def _load_view(torch, view: dict[str, Any], device):  # noqa: ANN001
    """Stream one view's supervision to the GPU: rgb [1,H,W,3], alpha [1,H,W,1],
    depth [1,H,W,1] | None. Pixels are read from disk on demand so plans with
    thousands of 512² views never have to fit in host RAM."""
    rgb_np, alpha_np, d = _view_arrays(view)
    rgb = torch.from_numpy(np.ascontiguousarray(rgb_np)).to(device).unsqueeze(0)
    alpha = torch.from_numpy(np.ascontiguousarray(alpha_np)).to(device).unsqueeze(0)
    depth = None
    if d is not None:
        depth = torch.from_numpy(np.ascontiguousarray(d)).to(device).unsqueeze(0)
    return rgb, alpha, depth


def _view_stream(torch, views, device, batch: int, prefetch: bool, seed: int):  # noqa: ANN001
    """Infinite stream of training batches — `(viewmats [B,4,4], rgb [B,H,W,3],
    alpha [B,H,W,1], depth [B,H,W,1] | None)` on `device`, B random views each.

    With `prefetch`, a daemon thread does the expensive part (SZF zstd / legacy
    PNG decode + stacking) for the NEXT batch while the GPU trains on the current
    one, so the render loop never stalls on disk. The host→device copy stays
    synchronous (the decode is the real cost), which keeps it correct with no
    tensor-lifetime traps."""
    n = len(views)

    def decode_batch(rng) -> tuple:  # noqa: ANN001 - CPU tensors, stacked to [B,...]
        vms, rgbs, alphas, depths = [], [], [], []
        for _ in range(batch):
            v = views[int(rng.integers(0, n))]
            rgb, alpha, d = _view_arrays(v)
            vms.append(v["viewmat"])
            rgbs.append(torch.from_numpy(np.ascontiguousarray(rgb)))
            alphas.append(torch.from_numpy(np.ascontiguousarray(alpha)))
            if d is not None:
                depths.append(torch.from_numpy(np.ascontiguousarray(d)))
        depth = torch.stack(depths) if len(depths) == batch else None
        return torch.stack(vms), torch.stack(rgbs), torch.stack(alphas), depth

    def to_dev(b) -> tuple:  # noqa: ANN001
        vm, rgb, alpha, depth = b
        return (
            vm.to(device), rgb.to(device), alpha.to(device),
            depth.to(device) if depth is not None else None,
        )

    if not prefetch:
        rng = np.random.default_rng(seed)
        while True:
            yield to_dev(decode_batch(rng))

    import queue
    import threading

    q: queue.Queue = queue.Queue(maxsize=3)

    def worker() -> None:
        rng = np.random.default_rng(seed)
        try:
            while True:
                q.put(decode_batch(rng))  # blocks when the queue is full
        except Exception as exc:  # surface to the main thread rather than deadlock
            q.put(exc)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if isinstance(item, Exception):
            raise item
        yield to_dev(item)


CKPT_DIRNAME = "ckpt"


def _ckpt_dir(out_path: Path) -> Path:
    """Where Stage-6 resumable checkpoints live (`splat/ckpt/`, beside trained.ply)."""
    return out_path.parent / CKPT_DIRNAME


def _strat_state_to(state: dict[str, Any], mover: Callable[[Any], Any]) -> dict[str, Any]:
    """Copy a gsplat strategy-state dict, moving its tensors via `mover` (`.cpu()` for
    save, `.to(device)` for load) and passing scalars / None through untouched."""
    return {k: mover(v) if hasattr(v, "to") else v for k, v in state.items()}


def _save_checkpoint(  # noqa: ANN001
    torch, ckpt_dir, step, splats, optimizers, means_sched, strat_state, meta, keep
) -> None:
    """Atomically write a full training checkpoint (params + per-param Adam state +
    means scheduler + densification-strategy state + step), then prune to the most
    recent `keep`. Everything is stored on the CPU so it reloads on any device."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "params": {k: v.detach().cpu() for k, v in splats.items()},
        "optimizers": {n: o.state_dict() for n, o in optimizers.items()},
        "means_sched": means_sched.state_dict(),
        "strat_state": _strat_state_to(strat_state, lambda v: v.detach().cpu()),
        "meta": meta,
    }
    path = ckpt_dir / f"step_{int(step):06d}.pt"
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    if keep > 0:
        for old in sorted(ckpt_dir.glob("step_*.pt"))[:-keep]:
            old.unlink(missing_ok=True)


def _load_checkpoint(torch, ckpt_dir: Path, device):  # noqa: ANN001
    """The newest valid checkpoint in `ckpt_dir` (loaded to `device`), or None. Tries
    newest-first, skipping any that fail to load (tolerating a torn tail file)."""
    if not ckpt_dir.is_dir():
        return None
    for path in sorted(ckpt_dir.glob("step_*.pt"), reverse=True):
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except Exception:
            continue
    return None


def train_splat(
    *,
    run: str,
    slot: str,
    model: str,
    cloud_path: Path,
    refs_dir: Path,
    out_path: Path,
    params: TrainParams = TrainParams(),
    resume: bool = True,
    progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Fine-tune the Stage-3 surfel cloud at `cloud_path` against the Stage-5
    references in `refs_dir`, writing the optimized 2DGS splat to `out_path`.
    Requires a CUDA GPU + gsplat (raises a clear error otherwise). Returns a
    compact summary (init/final splat counts, final metrics, bytes).

    With `resume` (default), continues from the latest `splat/ckpt/` checkpoint when
    one is present and compatible (same SH config); otherwise starts from the surfel
    init. Pass `resume=False` to ignore any checkpoint and train from scratch."""
    torch, F, gsplat = _require_cuda_trainer()
    from gsplat import DefaultStrategy, rasterization_2dgs

    cloud_path, refs_dir, out_path = Path(cloud_path), Path(refs_dir), Path(out_path)
    if not cloud_path.is_file():
        raise FileNotFoundError(f"surfel cloud not found: {cloud_path} (run Stage 3)")

    doc = _read_transforms(refs_dir)
    frames = doc.get("frames", [])
    if not frames:
        raise RuntimeError(f"{refs_dir/TRANSFORMS_NAME} has no frames (run Stage 5)")

    device = torch.device("cuda")
    torch.manual_seed(params.seed)

    width, height = int(doc["w"]), int(doc["h"])
    K = torch.tensor(
        [[doc["fl_x"], 0.0, doc["cx"]], [0.0, doc["fl_y"], doc["cy"]], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )

    # Index the supervision set (paths + poses only; pixels stream from disk per
    # step). Current sets carry ONE SZF frame per view (`frame_path`); legacy
    # PNG-triple sets (`file_path`/`alpha_path`/`depth_path`) stay trainable.
    # `near`/`far` (shared across the plan) decode the log-uint16 depth codes
    # back to metric metres.
    depth_near = float(doc["near"]) if "near" in doc else None
    depth_far = float(doc["far"]) if "far" in doc else None
    views: list[dict[str, Any]] = []
    cam_centers: list[np.ndarray] = []
    for i, fr in enumerate(frames):
        c2w = np.asarray(fr["transform_matrix"], dtype=np.float64)
        cam_centers.append(c2w[:3, 3])
        frame_rel = fr.get("frame_path")
        rgb_rel = fr.get("file_path")
        if frame_rel is None and rgb_rel is None:
            raise ValueError(
                f"{refs_dir / TRANSFORMS_NAME}: frame {i} has neither 'frame_path' "
                "(SZF) nor 'file_path' (legacy PNG) — re-run Stage 5"
            )
        views.append(
            {
                "viewmat": torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)),
                "frame": refs_dir / frame_rel if frame_rel else None,
                "rgb": refs_dir / rgb_rel if rgb_rel else None,
                "alpha": refs_dir / fr["alpha_path"] if fr.get("alpha_path") else None,
                "depth": refs_dir / fr["depth_path"] if fr.get("depth_path") else None,
                "depth_near": depth_near,
                "depth_far": depth_far,
            }
        )
    n_views = len(views)

    # Surfel init (also the scene-scale fallback source when there's ≤1 camera).
    init = _load_cloud(cloud_path)
    n_init = int(init["means"].shape[0])

    # Scene scale = camera-cloud radius (× 1.1), the 3DGS spatial LR / density unit.
    centers = np.stack(cam_centers, axis=0)
    scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) if n_views > 1 else 0.0
    if scene_scale <= 1e-6:
        scene_scale = float(np.linalg.norm(init["means"].max(0) - init["means"].min(0))) * 0.5
    scene_scale = max(scene_scale * 1.1, 1e-3)

    bands = (params.sh_degree + 1) ** 2

    # Resume from the latest checkpoint when present + compatible (same SH config),
    # else build from the surfel init. `meta` guards against reloading a checkpoint
    # whose model shape no longer matches this cloud's SH bands.
    ckpt_dir = _ckpt_dir(out_path)
    meta = {"sh_degree": params.sh_degree, "n_init": n_init}
    ckpt = _load_checkpoint(torch, ckpt_dir, device) if resume else None
    if ckpt is not None and ckpt.get("meta", {}).get("sh_degree") != params.sh_degree:
        ckpt = None

    if ckpt is not None:
        splats = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v.to(device)) for k, v in ckpt["params"].items()}
        )
    else:
        splats = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.from_numpy(init["means"])),
                "scales": torch.nn.Parameter(torch.from_numpy(init["scales"])),
                "quats": torch.nn.Parameter(torch.from_numpy(init["quats"])),
                "opacities": torch.nn.Parameter(torch.from_numpy(init["opacities"])),
                "sh0": torch.nn.Parameter(torch.from_numpy(init["sh0"])),
            }
        ).to(device)
        if bands > 1:
            splats["shN"] = torch.nn.Parameter(
                torch.zeros((n_init, bands - 1, 3), dtype=torch.float32, device=device)
            )

    lrs = {
        "means": params.means_lr * scene_scale,
        "scales": params.scales_lr,
        "quats": params.quats_lr,
        "opacities": params.opacities_lr,
        "sh0": params.sh0_lr,
        "shN": params.shN_lr,
    }
    optimizers = {
        name: torch.optim.Adam(
            [{"params": [splats[name]], "lr": lrs[name], "name": name}], eps=1e-15
        )
        for name in splats.keys()
    }
    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / max(params.iterations, 1))
    )

    # reset_every=0 disables opacity resets by pushing the trigger past the last
    # step (see TrainParams). Accepted side effect: the strategy's scale-based
    # prune (gated on `step > reset_every`) also never fires — our init has no
    # giant blobs and the alpha/depth losses punish oversized splats directly.
    strategy = DefaultStrategy(
        prune_opa=params.prune_opa,
        grow_grad2d=params.grow_grad2d,
        grow_scale3d=params.grow_scale3d,
        refine_start_iter=params.refine_start_iter,
        refine_stop_iter=params.resolved_refine_stop,
        reset_every=params.reset_every if params.reset_every > 0 else params.iterations + 1,
        refine_every=params.refine_every,
        absgrad=params.absgrad,
        key_for_gradient="gradient_2dgs",
        verbose=False,
    )
    strategy.check_sanity(splats, optimizers)
    strat_state = strategy.initialize_state(scene_scale=scene_scale)

    # Restore optimizer / scheduler / densification state so a resumed run continues
    # exactly where it stopped (Adam moments, LR-decay position, grow/prune counters).
    # The scheduler is created above from the base (undecayed) LR, then its state is
    # loaded, so base_lrs stay correct while the current LR follows the checkpoint.
    start_step = 0
    if ckpt is not None:
        for name, opt in optimizers.items():
            sd = ckpt["optimizers"].get(name)
            if sd is not None:
                opt.load_state_dict(sd)
        means_sched.load_state_dict(ckpt["means_sched"])
        strat_state = _strat_state_to(ckpt["strat_state"], lambda v: v.to(device))
        start_step = int(ckpt["step"]) + 1

    window = _gaussian_window(torch, 11, 1.5, device, channels=3)
    dist_on = params.dist_lambda > 0.0

    n_last = int(splats["means"].shape[0])
    if progress is not None:
        where = f"resume@{start_step} n={n_last}" if start_step else f"init={n_init}"
        progress(start_step, params.iterations, f"{where} views={n_views} scale={scene_scale:.2f}")

    t_start = t_last = time.perf_counter()
    done_last = start_step
    log_every = max(params.log_every, 1)
    stream = _view_stream(torch, views, device, max(params.batch, 1), params.prefetch, params.seed)

    for step in range(start_step, params.iterations):
        viewmats, gt_rgb, gt_alpha, gt_depth = next(stream)

        active_sh = min(step // max(params.sh_degree_interval, 1), params.sh_degree)
        colors, sh_deg = _render_inputs(torch, splats, active_sh)

        renders, pred_alpha, normals, normals_from_depth, distort, median_depth, info = rasterization_2dgs(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=viewmats,
            Ks=K.unsqueeze(0).expand(viewmats.shape[0], -1, -1),
            width=width,
            height=height,
            sh_degree=sh_deg,
            packed=False,
            near_plane=params.near_plane,
            far_plane=params.far_plane,
            render_mode="RGB+ED",
            distloss=dist_on,
            depth_mode=params.depth_mode,
            absgrad=params.absgrad,
        )
        pred_rgb = renders[..., :3]
        pred_depth = median_depth if params.depth_mode == "median" else renders[..., 3:4]

        if params.refine:
            strategy.step_pre_backward(splats, optimizers, strat_state, step, info)

        # Photometric: full-frame L1 + D-SSIM. Both sides are premultiplied-over-
        # black (the reference alpha-blends over a black clear colour; gsplat
        # composites with no background), so glass/MASK pixels compare like-with-
        # like and background pixels directly penalize floater energy.
        l1 = (pred_rgb - gt_rgb).abs().mean()
        ssim = _ssim(F, pred_rgb.clamp(0, 1).permute(0, 3, 1, 2), gt_rgb.permute(0, 3, 1, 2), window)
        loss = (1.0 - params.ssim_lambda) * l1 + params.ssim_lambda * (1.0 - ssim)

        # Alpha (coverage/opacity) — the renderer's exact coverage masks.
        if params.alpha_lambda > 0.0:
            loss = loss + params.alpha_lambda * (pred_alpha - gt_alpha).abs().mean()

        # Depth — alpha-gated L1 (metres) on the `depth_mode` statistic.
        if gt_depth is not None and params.depth_lambda > 0.0:
            gate = (gt_alpha > params.alpha_gate) & (gt_depth > 0)
            if gate.any():
                dl = ((pred_depth - gt_depth).abs() * gate).sum() / (gate.sum() + 1e-8)
                loss = loss + params.depth_lambda * dl

        # 2DGS normal consistency (render normals vs normals-from-depth), fg only.
        if params.normal_lambda > 0.0 and step >= params.normal_start_iter:
            nerr = 1.0 - (normals * normals_from_depth).sum(dim=-1)
            mask = gt_alpha.squeeze(-1) > params.alpha_gate
            if mask.any():
                loss = loss + params.normal_lambda * nerr[mask].mean()

        # 2DGS depth distortion.
        if dist_on and step >= params.dist_start_iter:
            loss = loss + params.dist_lambda * distort.mean()

        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()

        if params.refine:
            strategy.step_post_backward(splats, optimizers, strat_state, step, info, packed=False)
            if len(splats["means"]) > params.cap_max:
                strategy.refine_stop_iter = step  # freeze densification (VRAM guard)

        if progress is not None and (step % log_every == 0 or step == params.iterations - 1):
            now = time.perf_counter()
            done = step + 1
            n_now = len(splats["means"])
            rate = (done - done_last) / max(now - t_last, 1e-6)
            eta = (params.iterations - done) / max(rate, 1e-6)
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            rsv_gb = torch.cuda.memory_reserved() / 1e9
            progress(
                done,
                params.iterations,
                f"loss={float(loss):.4f} n={n_now} ({n_now - n_last:+d}) | "
                f"{rate:.2f} it/s ({rate * params.batch:.0f} views/s) | "
                f"elapsed {_fmt_hms(now - t_start)} ETA {_fmt_hms(eta)} | "
                f"vram {rsv_gb:.1f}GB used / {free_gb:.1f}GB free",
            )
            t_last, done_last, n_last = now, done, n_now

        if (
            params.ckpt_every > 0
            and (step + 1) % params.ckpt_every == 0
            and step + 1 < params.iterations
        ):
            _save_checkpoint(
                torch, ckpt_dir, step, splats, optimizers, means_sched,
                strat_state, meta, params.ckpt_keep,
            )

    # Final metrics over a capped subset + write the splat.
    if progress is not None:
        progress(
            params.iterations,
            params.iterations,
            f"training done in {_fmt_hms(time.perf_counter() - t_start)} — "
            f"evaluating {min(n_views, params.eval_max_views)} views + writing {out_path.name}",
        )
    metrics = _evaluate(
        torch, rasterization_2dgs, splats, views, K, width, height, params, device
    )

    with torch.no_grad():
        quats = torch.nn.functional.normalize(splats["quats"].detach(), dim=-1).cpu().numpy()
        _encode_trained_ply(
            splats["means"].detach().cpu().numpy(),
            quats,
            splats["sh0"].detach().cpu().numpy().reshape(-1, 3),
            splats["opacities"].detach().cpu().numpy(),
            splats["scales"].detach().cpu().numpy()[:, :2],
            out_path,
        )
    # Training finished + the final splat is on disk → the periodic checkpoints are
    # obsolete; drop them so a later resume doesn't re-enter a completed run.
    import shutil

    shutil.rmtree(ckpt_dir, ignore_errors=True)

    n_final = int(splats["means"].shape[0])

    return {
        "run": run,
        "slot": slot,
        "model": model,
        "splats_init": n_init,
        "splats_final": n_final,
        "iterations": params.iterations,
        "views": n_views,
        "resolution": width,
        "scene_scale": round(scene_scale, 4),
        "metrics": metrics,
        "params": params.as_summary(),
        "bytes": out_path.stat().st_size,
        "out_path": str(out_path),
    }


def _evaluate(torch, rasterization_2dgs, splats, views, K, width, height, params, device):  # noqa: ANN001
    """Mean PSNR (foreground), RGB L1, and alpha-gated depth L1 over a random
    subset of views (capped by `eval_max_views` — plans can have thousands)."""
    colors, sh_deg = _render_inputs(torch, splats, params.sh_degree)
    n_eval = min(len(views), params.eval_max_views)
    sel = (
        np.random.default_rng(params.seed).choice(len(views), size=n_eval, replace=False)
        if n_eval < len(views)
        else np.arange(len(views))
    )
    psnr_sum = l1_sum = depth_sum = 0.0
    rgb_views = depth_views = 0
    with torch.no_grad():
        for i in sel:
            v = views[int(i)]
            gt_rgb, gt_alpha, gt_depth = _load_view(torch, v, device)
            renders, _alpha, _normals, _nfd, _distort, median_depth, _info = rasterization_2dgs(
                means=splats["means"],
                quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=colors,
                viewmats=v["viewmat"].to(device).unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=width,
                height=height,
                sh_degree=sh_deg,
                packed=False,
                near_plane=params.near_plane,
                far_plane=params.far_plane,
                render_mode="RGB+ED",
                depth_mode=params.depth_mode,
            )
            pred_rgb = renders[..., :3].clamp(0, 1)
            pred_depth = median_depth if params.depth_mode == "median" else renders[..., 3:4]
            fg = gt_alpha > params.alpha_gate
            fg3 = fg.expand_as(gt_rgb)
            if fg3.any():
                mse = ((pred_rgb - gt_rgb)[fg3] ** 2).mean()
                psnr_sum += float(-10.0 * torch.log10(mse + 1e-10))
                l1_sum += float((pred_rgb - gt_rgb)[fg3].abs().mean())
                rgb_views += 1
            if gt_depth is not None:
                gate = fg & (gt_depth > 0)
                if gate.any():
                    depth_sum += float((pred_depth - gt_depth).abs()[gate].mean())
                    depth_views += 1
    return {
        "psnr": round(psnr_sum / max(rgb_views, 1), 3),
        "rgb_l1": round(l1_sum / max(rgb_views, 1), 5),
        "depth_l1": round(depth_sum / max(depth_views, 1), 5) if depth_views else None,
        "eval_views": int(n_eval),
    }


def _main() -> None:
    """Standalone CLI for the GPU box: fine-tune one cell's surfel cloud against
    its Stage-5 references."""
    import argparse

    ap = argparse.ArgumentParser(description="Stage 6 — gsplat 2DGS splat fine-tune")
    ap.add_argument("--cloud", required=True, type=Path, help="Stage-3 cloud.ply (init)")
    ap.add_argument("--refs", required=True, type=Path, help="Stage-5 refs/ dir")
    ap.add_argument("--out", required=True, type=Path, help="output trained.ply")
    ap.add_argument("--iterations", type=int, default=TrainParams.iterations)
    ap.add_argument("--sh-degree", type=int, default=TrainParams.sh_degree)
    ap.add_argument(
        "--refine-stop-iter", type=int, default=None,
        help="stop densification at this step (default: 50%% of iterations)",
    )
    ap.add_argument(
        "--batch", type=int, default=TrainParams.batch,
        help="views rendered per step; >1 fills the GPU (use ~1/batch the iterations)",
    )
    ap.add_argument(
        "--ckpt-every", type=int, default=TrainParams.ckpt_every,
        help="write a resumable checkpoint every N steps (0 disables)",
    )
    ap.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="resume from the latest checkpoint beside --out (splat/ckpt) if present",
    )
    ap.add_argument("--run", default="?")
    ap.add_argument("--slot", default="?")
    ap.add_argument("--model", default="?")
    args = ap.parse_args()

    def _log(done: int, total: int, msg: str) -> None:
        pct = 100.0 * done / max(total, 1)
        print(f"[stage6 {time.strftime('%H:%M:%S')}] {done}/{total} ({pct:4.1f}%) {msg}", flush=True)

    summary = train_splat(
        run=args.run,
        slot=args.slot,
        model=args.model,
        cloud_path=args.cloud,
        refs_dir=args.refs,
        out_path=args.out,
        params=TrainParams(
            iterations=args.iterations,
            sh_degree=args.sh_degree,
            refine_stop_iter=args.refine_stop_iter,
            batch=args.batch,
            ckpt_every=args.ckpt_every,
        ),
        resume=args.resume,
        progress=_log,
    )
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    _main()
