"""Window / glass transparency — an optional, self-contained mesh post-process.

Two entry points, sharing one transform:

  * `apply_alpha_from_white(mesh)` — the UNCONDITIONAL core. Scans every
    base-color texture for white / near-white texels (the bright glass panes,
    versus the darker frame / mullions) and pushes those texels to
    near-transparent on the texture's alpha channel, flipping the material to
    alpha blending so a renderer honors it. The frontend "make transparent"
    control calls this directly to force the effect onto any chosen object.

  * `apply_window_glass_transparency(mesh, ...)` — the GATED wrapper the
    generation pipeline uses. Runs the core only when BOTH gates pass: the
    object reads as glass (its seed prompt or distilled noun phrase mentions a
    `_GLASS_KEYWORDS` token) AND it was decided to be symmetrized (a flat panel
    — windows / glass panels are exactly what the symmetry step mirrors, so the
    decision is a good proxy for "this is a flat glazed surface, not a glass
    figurine"). Non-glass or un-symmetrized objects are returned untouched.

It runs ALONGSIDE the initial transform pass (load -> symmetrize -> rescale ->
export) in `pipeline.generation._generate_one`, right before the transformed
GLB is written, and lives in its own module behind these entry points so the
whole feature can be pulled out by deleting the call sites plus the import.

The mesh is mutated in place. Logging is deliberately left to the caller: this
is invoked inside `asyncio.to_thread`, off the event-loop thread, where the slot
log's asyncio.Queue fan-out is not safe to touch. The returned stats dict — or
None when nothing changed — is what the caller records.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh
from PIL import Image

# Seed prompt / noun phrase tokens that mark an object as glazed. Matched as
# case-insensitive substrings, so "windows", "stained glass", "glass railing"
# all trigger.
_GLASS_KEYWORDS = ("window", "glass")

# A texel is "white / near-white" when its DARKEST rgb channel is at least this
# fraction of full white. Keying off the min channel couples brightness with
# desaturation, so a bright-but-tinted reflection (high in one channel, low in
# another) stays opaque while true white/grey glass is caught.
WHITE_THRESHOLD = 0.8

# Opacity given to the white texels (0 = invisible, 1 = opaque).
GLASS_ALPHA = 0.065


def looks_like_glass(*texts: str | None) -> str | None:
    """The first `_GLASS_KEYWORDS` token found in any of `texts`, else None —
    returned (not a bool) so the caller can record which word matched."""
    for text in texts:
        if not text:
            continue
        lowered = text.lower()
        for keyword in _GLASS_KEYWORDS:
            if keyword in lowered:
                return keyword
    return None


def _iter_geometries(mesh: trimesh.Trimesh | trimesh.Scene) -> list[Any]:
    if isinstance(mesh, trimesh.Scene):
        return list(mesh.geometry.values())
    return [mesh]


def _whiten_alpha(material: Any) -> float | None:
    """Drive the white/near-white texels of `material`'s base-color texture to
    `GLASS_ALPHA` and switch the material to alpha blending. Returns the fraction
    of texels made transparent (0.0 if none were white), or None when there is
    no editable base-color texture."""
    texture = getattr(material, "baseColorTexture", None)
    if texture is None:
        return None

    # Writable RGBA copy. Rebuilding the image from this array (below) gives it
    # format=None, so trimesh re-encodes it as PNG on export and the alpha
    # survives — a texture that came in as JPEG would otherwise be re-saved as
    # JPEG, silently dropping the channel.
    rgba = np.array(texture.convert("RGBA"), dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        return None

    cutoff = round(WHITE_THRESHOLD * 255)
    white = rgba[..., :3].min(axis=2) >= cutoff
    if not white.any():
        return 0.0

    rgba[white, 3] = round(GLASS_ALPHA * 255)
    material.baseColorTexture = Image.fromarray(rgba, mode="RGBA")
    material.alphaMode = "BLEND"
    return float(white.mean())


def apply_alpha_from_white(
    mesh: trimesh.Trimesh | trimesh.Scene,
) -> dict[str, Any] | None:
    """Unconditionally bake texel transparency into every base-color texture on
    `mesh`: white / near-white texels go to `GLASS_ALPHA` and the material flips
    to alpha blending (see the module docstring).

    `mesh` is mutated in place. Returns a stats dict when at least one texture was
    edited; None otherwise (no editable base-color texture, or no white texels to
    cut). This is the gate-free core — the forced frontend path calls it directly;
    `apply_window_glass_transparency` is the gated pipeline wrapper around it."""
    edited: list[float] = []
    for geom in _iter_geometries(mesh):
        material = getattr(getattr(geom, "visual", None), "material", None)
        fraction = _whiten_alpha(material)
        if fraction is not None and fraction > 0.0:
            edited.append(fraction)

    if not edited:
        return None
    return {
        "textures": len(edited),
        "white_threshold": WHITE_THRESHOLD,
        "alpha": GLASS_ALPHA,
        "transparent_fraction": round(sum(edited) / len(edited), 4),
    }


def apply_window_glass_transparency(
    mesh: trimesh.Trimesh | trimesh.Scene,
    *,
    noun_phrase: str | None,
    prompt: str | None,
    symmetrized: bool,
) -> dict[str, Any] | None:
    """Gated entry for the generation pipeline: run `apply_alpha_from_white` only
    when the object reads as glass (keyword in `noun_phrase` / `prompt`) AND it was
    decided to be symmetrized (`symmetrized` — `Node.symmetry_cut_plane != "none"`,
    so a flat glazed panel rather than an arbitrary glass prop).

    `mesh` is mutated in place. Returns a stats dict (carrying the matched
    `keyword`) when both gates passed AND a texture was edited; None otherwise."""
    keyword = looks_like_glass(noun_phrase, prompt)
    if keyword is None or not symmetrized:
        return None
    stats = apply_alpha_from_white(mesh)
    if stats is not None:
        stats["keyword"] = keyword
    return stats
