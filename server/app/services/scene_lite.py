"""Vertex-color scene bake for the /lite studio page.

A benchmarked scene is generated as many Trellis meshes whose fragmented
auto-UVs make them impossible to decimate while keeping a texture. This bakes
each mesh's texture into per-vertex colors, drops the textures + UVs, decimates
the geometry hard, and merges everything into one small, textureless GLB — a
web-shippable preview. Reads the RAW (PNG-textured) meshes, since the optimized
twins are KTX2/Basis (no transcoder here to decode them). Shells the bake to the
Node toolchain like the proxy/library passes.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[2]
_OPTIMIZE_DIR = _SERVER_DIR / "tools" / "optimize-assets"
_NODE_BIN = os.environ.get("STARSHOT_NODE_BIN", "node")

# Public so the caller's build cache can invalidate when the bake logic changes.
BAKE_SCRIPT = _OPTIMIZE_DIR / "bake-vertex-color.mjs"


async def build_scene_vcolor(raw_dir: Path, dst: Path) -> dict:
    """Bake the RAW objects in `raw_dir` into one textureless, vertex-colored
    scene GLB at `dst`. Returns the bake's stats dict (objects, srcTris, outTris,
    outBytes); the quality knobs live in bake-vertex-color.mjs."""
    raw_dir, dst = raw_dir.resolve(), dst.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        _NODE_BIN,
        str(BAKE_SCRIPT),
        "--inputs-dir", str(raw_dir),
        "--out-file", str(dst),
        cwd=str(_OPTIMIZE_DIR),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        detail = stderr.decode(errors="replace")[:500] if stderr else f"node exit {proc.returncode}"
        raise RuntimeError(f"bake-vertex-color.mjs failed: {detail}")
    # Stats are the last JSON line on stdout; tolerate any logging before it.
    for line in reversed(stdout.decode(errors="replace").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}
