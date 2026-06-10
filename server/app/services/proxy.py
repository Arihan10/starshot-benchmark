"""Low-poly projection proxy for the /pano walkthrough.

The matterport-style walkthrough interpolates between 360° capture points by
projecting the captured panoramas onto a coarse 3D stand-in of the scene. This
builds that stand-in: a heavily decimated, geometry-only GLB.

The viewer hands us the scene's geometry already merged + baked into world space
(every placed mesh flattened into one GLB), so this is a single-file decimation
pass — `optimize.mjs --proxy` (gltf-transform `weld → simplify → strip materials
→ prune`), the same meshoptimizer simplifier the asset library uses, tuned for a
much lower triangle budget and no textures. Output is a plain GLB (no KTX2, no
Meshopt) so /pano loads it with a bare GLTFLoader.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

# optimize.mjs lives at server/tools/optimize-assets/optimize.mjs:
#   parents[2] of this file = server/ ; tools/ sits beside app/.
_SERVER_DIR = Path(__file__).resolve().parents[2]
_OPTIMIZE_DIR = _SERVER_DIR / "tools" / "optimize-assets"
_OPTIMIZE_SCRIPT = _OPTIMIZE_DIR / "optimize.mjs"
_NODE_BIN = os.environ.get("STARSHOT_NODE_BIN", "node")


async def build_proxy(
    src: Path,
    dst: Path,
    *,
    target_tris: int = 8000,
    error: float = 0.5,
) -> None:
    """Decimate `src` (a merged, world-space scene GLB) into a geometry-only
    proxy at `dst`. `target_tris` is the whole-scene budget that the meshoptimizer
    simplifier targets; `error` is a loose ceiling so the budget is what binds
    (proxy mode strips attribute seams, so the ratio actually reaches the target)."""
    src, dst = src.resolve(), dst.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        _NODE_BIN,
        str(_OPTIMIZE_SCRIPT),
        "--file", str(src),
        "--out-file", str(dst),
        "--proxy",
        "--target-tris", str(int(target_tris)),
        "--error", str(float(error)),
        cwd=str(_OPTIMIZE_DIR),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        detail = stderr.decode(errors="replace")[:500] if stderr else f"node exit {proc.returncode}"
        raise RuntimeError(f"optimize.mjs --proxy failed: {detail}")
