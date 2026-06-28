#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["usd-core", "trimesh>=4.5", "numpy>=2.0", "pillow>=11.0"]
# ///
"""Export one generated scene to a single USD file, headless (no client/browser).

Selection walks the on-disk run tree — run -> scene (slot) -> model -> version —
and exports the chosen cell's `objects-generated/` directory: the UNOPTIMIZED,
world-placed meshes (raw Trellis geometry already rotated + per-axis scaled +
translated into each object's world bbox by `rescale_mesh_to_bbox`; see
server/app/pipeline/generation.py). Every `<id>.glb` there is already in world
space, so the scene is just those meshes composed under one USD root — the same
set the viewer renders.

Each source GLB is decoded with trimesh and re-authored as a `UsdGeomMesh` with
a UsdPreviewSurface material; the baked-in texture is written out once per unique
image (content-deduplicated, so prefab reuses share one file). By default the
result is a single self-contained `.usdz` (geometry + textures bundled); pass an
`--out` ending in `.usd`/`.usdc`/`.usda` to get an uncompressed stage with a
sibling `<stem>.textures/` folder instead.

USD has no GLB-style 4 GiB ceiling, so the multi-GB unoptimized scenes that
couldn't fit in one GLB export fine here. Work files are written on the OUTPUT
volume (never the system temp), and sources are decoded one at a time, so peak
memory stays at ~one mesh+texture regardless of scene size.

Run with uv so the dependencies resolve (any missing selection is prompted):
  uv run scripts/export_scene_usd.py
  uv run scripts/export_scene_usd.py --run decomp-before-frames --slot battle-arena \
      --model opus --version 3 --out battle-arena.usdz
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdShade, UsdUtils, Vt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = Path(os.environ.get("STARSHOT_RUNS_DIR", str(REPO_ROOT / "runs")))

# The unoptimized, world-placed generated meshes (generation.GENERATED_RAW_SUBDIR).
# `<id>.raw.glb` siblings are the pre-rescale Trellis intermediates — never placed.
OBJECTS_SUBDIR = "objects-generated"
RAW_SUFFIX = ".raw.glb"
_USD_EXTS = (".usdz", ".usdc", ".usda", ".usd")


# --- selection ---------------------------------------------------------------


def _placed_glbs(objects_dir: Path) -> list[Path]:
    """The world-placed `<id>.glb` set the viewer renders — `<id>.raw.glb`
    intermediates and `<id>.png` reference images excluded, sorted like the
    server's mesh bundle."""
    return sorted(
        p for p in objects_dir.glob("*.glb") if not p.name.endswith(RAW_SUFFIX)
    )


def _discover(runs_dir: Path) -> list[Path]:
    """Every `objects-generated/` dir under runs that holds at least one placed
    GLB. run/slot/model are always its first three path segments; the bit
    between model and `objects-generated` is the generated version (e.g.
    `generated/3`, or empty for a pre-versioning legacy build)."""
    return sorted(
        d
        for d in runs_dir.rglob(OBJECTS_SUBDIR)
        if d.is_dir() and _placed_glbs(d)
    )


def _rel_parts(objects_dir: Path, runs_dir: Path) -> tuple[str, ...]:
    return objects_dir.relative_to(runs_dir).parts


def _version_label(objects_dir: Path, runs_dir: Path) -> str:
    """The segments between the model dir and `objects-generated`, e.g.
    `generated/3`; `(root)` when the set sits directly under the model
    (legacy, unversioned)."""
    middle = _rel_parts(objects_dir, runs_dir)[3:-1]
    return "/".join(middle) if middle else "(root)"


def _pick(kind: str, options: list[str], provided: str | None) -> str:
    """Resolve one selection level. Honors a provided value (matched exactly,
    or as a `generated/<n>` shorthand for a version); otherwise prompts when
    interactive, or errors with the candidate list when not."""
    if provided is not None:
        norm = provided.replace("\\", "/")
        if norm in options:
            return norm
        if kind == "version" and f"generated/{norm}" in options:
            return f"generated/{norm}"
        sys.exit(f"[export] no {kind} {provided!r}; available: {', '.join(options)}")
    if len(options) == 1:
        print(f"[export] {kind}: {options[0]}")
        return options[0]
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        sys.exit(
            f"[export] --{kind} not given and no TTY to prompt; "
            f"available: {', '.join(options)}"
        )
    print(f"\nSelect {kind}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input(f"{kind} [1-{len(options)}]: ").strip()
        if raw in options:
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("  invalid choice")


def _select(runs_dir: Path, args: argparse.Namespace) -> Path:
    valid = _discover(runs_dir)
    if not valid:
        sys.exit(f"[export] no generated scenes with placed assets under {runs_dir}")

    def options(depth: int) -> list[str]:
        return sorted({_rel_parts(d, runs_dir)[depth] for d in valid})

    run = _pick("run", options(0), args.run)
    valid = [d for d in valid if _rel_parts(d, runs_dir)[0] == run]
    slot = _pick("scene", options(1), args.slot)
    valid = [d for d in valid if _rel_parts(d, runs_dir)[1] == slot]
    model = _pick("model", options(2), args.model)
    valid = [d for d in valid if _rel_parts(d, runs_dir)[2] == model]

    by_label = {_version_label(d, runs_dir): d for d in valid}
    label = _pick("version", sorted(by_label), args.version)
    return by_label[label]


# --- USD authoring -----------------------------------------------------------


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted smooth vertex normals (numpy only — avoids trimesh's
    scipy-backed path, which is absent here and noisy)."""
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    vn = np.zeros(verts.shape, dtype=np.float64)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (vn / norm).astype(np.float32)


def _diffuse_color(material) -> Gf.Vec3f:
    """A constant fallback diffuse for meshes with no baseColor texture."""
    src = None
    if material is not None:
        src = getattr(material, "baseColorFactor", None)
        if src is None:
            src = getattr(material, "main_color", None)
    if src is None:
        return Gf.Vec3f(0.6, 0.6, 0.6)
    arr = np.asarray(src, dtype=float).reshape(-1)
    if arr.size >= 3 and arr.max() > 1.0:
        arr = arr / 255.0
    return Gf.Vec3f(float(arr[0]), float(arr[1]), float(arr[2]))


def _make_texture_writer(tex_dir: Path):
    """Returns a `write(pil_image) -> filename` that saves each UNIQUE image
    once (keyed by pixel content), so prefab reuses sharing a texture don't
    re-encode or duplicate it. `.cache` exposes the seen set for a count."""
    cache: dict[str, str] = {}

    def write(img) -> str:
        key = hashlib.md5(img.tobytes()).hexdigest()
        name = cache.get(key)
        if name is None:
            name = f"{key}.png"
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            img.save(tex_dir / name)
            cache[key] = name
        return name

    write.cache = cache  # type: ignore[attr-defined]
    return write


def _author_mesh(stage: Usd.Stage, prim_path: str, geom, write_tex, tex_ref) -> None:
    verts = np.asarray(geom.vertices, dtype=np.float64)
    faces = np.asarray(geom.faces, dtype=np.int64)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    # USD meshes default to Catmull-Clark subdivision, which would round off the
    # polygons; pin to none so the triangles render exactly as authored.
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(_vertex_normals(verts, faces)))

    visual = getattr(geom, "visual", None)
    uv = getattr(visual, "uv", None)
    has_uv = uv is not None and len(uv) == len(verts)
    if has_uv:
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        st.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, dtype=np.float32)))

    material = UsdShade.Material.Define(stage, prim_path + "/mat")
    surface = UsdShade.Shader.Define(stage, prim_path + "/mat/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")

    mat = getattr(visual, "material", None)
    texture = getattr(mat, "baseColorTexture", None)
    if texture is not None and has_uv:
        filename = write_tex(texture)
        reader = UsdShade.Shader.Define(stage, prim_path + "/mat/stReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, prim_path + "/mat/diffuseTex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_ref(filename))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
    else:
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(_diffuse_color(mat))

    UsdShade.MaterialBindingAPI(mesh).Bind(material)


def _author_stage(stage: Usd.Stage, sources: list[Path], write_tex, tex_ref) -> int:
    used: set[str] = set()
    meshes = 0
    for idx, path in enumerate(sources, 1):
        name = Tf.MakeValidIdentifier(path.stem)
        unique = name
        n = 2
        while unique in used:
            unique = f"{name}_{n}"
            n += 1
        used.add(unique)

        loaded = trimesh.load(path, process=False)
        geoms = loaded.dump(concatenate=False) if isinstance(loaded, trimesh.Scene) else [loaded]
        UsdGeom.Xform.Define(stage, f"/Scene/{unique}")
        sub = 0
        for geom in geoms:
            if getattr(geom, "faces", None) is None or len(geom.faces) == 0:
                continue
            _author_mesh(stage, f"/Scene/{unique}/mesh_{sub}", geom, write_tex, tex_ref)
            sub += 1
            meshes += 1
        del loaded, geoms
        print(f"[export] ({idx}/{len(sources)}) {unique}", flush=True)
    return meshes


def export_usd(sources: list[Path], out_path: Path) -> dict:
    """Compose every world-placed source mesh under one USD `/Scene` root and
    write `out_path`. `.usdz` is packaged self-contained; `.usd/.usdc/.usda`
    write the stage plus a sibling `<stem>.textures/` folder. Work files live on
    the output volume (the system temp may be a different, full disk)."""
    ext = out_path.suffix.lower()
    if ext not in _USD_EXTS:
        raise ValueError(f"output must end in {' / '.join(_USD_EXTS)}; got {out_path.name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    packaged = ext == ".usdz"
    work: Path | None = None
    if packaged:
        work = Path(tempfile.mkdtemp(prefix=".usdexport-", dir=out_path.parent))
        stage_path = work / "scene.usdc"
        tex_dir = work / "textures"
        tex_dir.mkdir()

        def tex_ref(fn: str) -> str:
            return f"./textures/{fn}"
    else:
        stage_path = out_path
        tex_dir = out_path.parent / f"{out_path.stem}.textures"
        if tex_dir.exists():
            shutil.rmtree(tex_dir)
        tex_dir.mkdir(parents=True)
        rel = tex_dir.name

        def tex_ref(fn: str) -> str:
            return f"./{rel}/{fn}"

    write_tex = _make_texture_writer(tex_dir)
    try:
        stage = Usd.Stage.CreateNew(str(stage_path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        scene_root = UsdGeom.Xform.Define(stage, "/Scene")
        stage.SetDefaultPrim(scene_root.GetPrim())
        meshes = _author_stage(stage, sources, write_tex, tex_ref)
        stage.GetRootLayer().Save()
        del stage
        if packaged and not UsdUtils.CreateNewUsdzPackage(str(stage_path), str(out_path)):
            raise ValueError("USDZ packaging failed")
    finally:
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)

    return {"meshes": meshes, "textures": len(write_tex.cache), "bytes": out_path.stat().st_size}


# --- entrypoint --------------------------------------------------------------


def _default_out(objects_dir: Path, runs_dir: Path) -> Path:
    parts = _rel_parts(objects_dir, runs_dir)[:-1]
    stem = "_".join(parts).replace(" ", "-").replace("/", "-")
    return Path.cwd() / f"{stem}.usdz"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a generated cell's objects-generated/ to a USD scene."
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run", help="top-level run dir under runs/")
    parser.add_argument("--slot", help="scene id (e.g. battle-arena)")
    parser.add_argument("--model", help="model alias (e.g. opus)")
    parser.add_argument("--version", help="generated version (e.g. 3 or generated/3)")
    parser.add_argument(
        "--out", type=Path, help="output .usdz/.usdc/.usda/.usd path (or dir); default <stem>.usdz"
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir.resolve()
    if not runs_dir.is_dir():
        sys.exit(f"[export] runs dir not found: {runs_dir}")

    objects_dir = _select(runs_dir, args)
    sources = _placed_glbs(objects_dir)
    rel = objects_dir.relative_to(runs_dir).as_posix()

    out = args.out
    if out is None:
        out = _default_out(objects_dir, runs_dir)
    elif out.is_dir():
        out = out / _default_out(objects_dir, runs_dir).name
    out = out.resolve()

    print(f"[export] {rel}: {len(sources)} assets -> {out}")
    try:
        summary = export_usd(sources, out)
    except ValueError as e:
        sys.exit(f"[export] failed: {e}")
    print(
        f"[export] wrote {out} ({summary['bytes'] / 2**20:.1f} MiB, "
        f"{summary['meshes']} meshes, {summary['textures']} textures)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
