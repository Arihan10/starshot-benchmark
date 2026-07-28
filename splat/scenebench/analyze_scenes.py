"""Scene-composition ground truth for the Stage 3/4 redesign analysis.

For each cell (runs/<run>/<slot>/<model>), measures per object:
  * world surface area (m²)
  * AABB diagonal → CURRENT Stage-3 spacing (base · max(1, diag/2)) + surfel count
  * texel density from the UV jacobian: s_tex = sqrt(world_area / texel_count)
    per face, area-weighted percentiles (m per texel)
  * d_view: distance from surface samples to the nearest Stage-2 camera
    candidate (needs freespace.npz; skipped otherwise)

Output: one JSON per cell under splat/scenebench/out/.

Run:  splat/.venv/bin/python -m splat.scenebench.analyze_scenes <cell_dir> ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

from splat.assets import load_geoms
from splat.stage3 import _BASE_SPACING, _VIEW_REF_M

_SAMPLES_PER_OBJ = 4000
_FACE_CAP = 60_000  # subsample faces for the UV-jacobian stats


def _texel_stats(geom: trimesh.Trimesh) -> dict | None:
    """Area-weighted s_tex percentiles (m/texel) from the UV jacobian."""
    visual = getattr(geom, "visual", None)
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    tex = getattr(material, "baseColorTexture", None)
    if uv is None or tex is None or len(uv) != len(geom.vertices):
        return None
    w, h = tex.size
    faces = np.asarray(geom.faces)
    if len(faces) > _FACE_CAP:
        idx = np.random.default_rng(0).choice(len(faces), _FACE_CAP, replace=False)
        faces = faces[idx]
        area_w = geom.area_faces[idx]
    else:
        area_w = geom.area_faces
    fuv = np.asarray(uv, dtype=np.float64)[faces]  # (F,3,2)
    e1 = fuv[:, 1] - fuv[:, 0]
    e2 = fuv[:, 2] - fuv[:, 0]
    area_uv = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    texels = area_uv * w * h
    ok = (texels > 1e-12) & (area_w > 1e-12)
    if not ok.any():
        return None
    s = np.sqrt(area_w[ok] / texels[ok])  # m per texel, per face
    aw = area_w[ok]
    order = np.argsort(s)
    cum = np.cumsum(aw[order]) / aw.sum()

    def pct(q: float) -> float:
        return float(s[order][np.searchsorted(cum, q)])

    return {
        "tex_size": [w, h],
        "s_tex_p10": pct(0.10),
        "s_tex_p50": pct(0.50),
        "s_tex_p90": pct(0.90),
        "texels_used": float(texels[ok].sum()),
    }


def _load_candidates(cell: Path) -> np.ndarray | None:
    npz = cell / "splat" / "freespace.npz"
    if not npz.is_file():
        return None
    with np.load(npz) as z:
        pos = z["cand_pos"].astype(np.float32)
        clear = z["cand_clear"].astype(np.float32)
        cm = float(z["clearance_m"])
    return pos[clear >= cm - 2.5e-4]


def analyze_cell(cell: Path) -> dict:
    from scipy.spatial import cKDTree

    tier = next(
        (cell / t for t in (
            "objects-generated-lite", "objects-generated-optimized",
            "objects-generated", "objects", "objects-optimized",
        ) if (cell / t).is_dir() and list((cell / t).glob("*.glb"))),
        None,
    )
    if tier is None:
        raise FileNotFoundError(f"no mesh tier under {cell}")
    cands = _load_candidates(cell)
    tree = cKDTree(cands) if cands is not None and len(cands) else None

    objects = []
    for glb in sorted(tier.glob("*.glb")):
        if glb.name.endswith(".raw.glb"):
            continue
        oid = glb.name[:-4]
        area = 0.0
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        tex_parts = []
        pts_parts = []
        for geom in load_geoms(glb):
            if len(geom.faces) == 0 or geom.area <= 0:
                continue
            area += float(geom.area)
            b = np.asarray(geom.bounds)
            lo = np.minimum(lo, b[0])
            hi = np.maximum(hi, b[1])
            t = _texel_stats(geom)
            if t is not None:
                t["area"] = float(geom.area)
                tex_parts.append(t)
            if tree is not None:
                n = max(64, int(_SAMPLES_PER_OBJ * geom.area / max(area, 1e-9)))
                pts, _ = trimesh.sample.sample_surface(geom, min(n, _SAMPLES_PER_OBJ), seed=0)
                pts_parts.append(np.asarray(pts))
        if area <= 0:
            continue
        diag = float(np.linalg.norm(hi - lo))
        spacing_cur = _BASE_SPACING * max(diag / _VIEW_REF_M, 1.0)
        rec: dict = {
            "id": oid,
            "area": round(area, 2),
            "diag": round(diag, 2),
            "spacing_cur": round(spacing_cur, 4),
            "surfels_cur": int(area / spacing_cur**2),
        }
        if tex_parts:
            wsum = sum(t["area"] for t in tex_parts)
            rec["s_tex_p50"] = round(
                sum(t["s_tex_p50"] * t["area"] for t in tex_parts) / wsum, 5
            )
            rec["s_tex_p10"] = round(min(t["s_tex_p10"] for t in tex_parts), 5)
            rec["tex_size"] = tex_parts[0]["tex_size"]
            rec["texels_used"] = int(sum(t["texels_used"] for t in tex_parts))
        if pts_parts:
            d, _ = tree.query(np.concatenate(pts_parts), workers=-1)
            rec["d_view_p10"] = round(float(np.percentile(d, 10)), 3)
            rec["d_view_p50"] = round(float(np.percentile(d, 50)), 3)
            rec["d_view_p90"] = round(float(np.percentile(d, 90)), 3)
        objects.append(rec)

    return {
        "cell": str(cell),
        "tier": tier.name,
        "n_objects": len(objects),
        "candidates": None if cands is None else int(len(cands)),
        "objects": objects,
    }


def main() -> None:
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    for arg in sys.argv[1:]:
        cell = Path(arg)
        rec = analyze_cell(cell)
        name = "_".join(cell.parts[-3:]).replace(" ", "-") + ".json"
        (out_dir / name).write_text(json.dumps(rec, indent=1))
        total_area = sum(o["area"] for o in rec["objects"])
        total_surf = sum(o["surfels_cur"] for o in rec["objects"])
        print(f"{cell}: {rec['n_objects']} objects, area={total_area:.0f} m², "
              f"surfels_cur≈{total_surf}", flush=True)


if __name__ == "__main__":
    main()
