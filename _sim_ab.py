"""A/B replay of the stage-4 plan on the REAL platformer cell.

A = current semantics (d_min = 8·feat).
B = context-scaled (d_min_eff = max(8·feat, 0.5·ctx), ctx = local open-air depth
    from the candidate clearance field).

Uses the run's own patches.bin, the real stratified candidates + clearances, the
real opaque occupancy for the ray-march (numpy replica of _occluded), and the
REAL _greedy_cover from splat.stage4. Same oct_top + tie-noise for both runs.
Rescue pass omitted (identical for both sides). Temporary script.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from splat.stage2 import load_free_space
from splat.stage4 import (
    PlanParams, _band_of, _bin_of, _candidates, _face_of, _greedy_cover,
    _hemisphere_dirs, _tangent_frames,
)

D = Path("runs/good_opus_new_hotel2/platformer-level/opus-new/splat")
t0 = time.time()

# --- real inputs -------------------------------------------------------------
P = np.frombuffer((D / "patches.bin").read_bytes(), dtype="<f4").reshape(-1, 8)
ppos = P[:, 0:3].astype(np.float64)
pnrm = P[:, 3:6].astype(np.float64)
pfeat = P[:, 6].astype(np.float64)
n_patch = len(ppos)
t1v, t2v = _tangent_frames(pnrm.astype(np.float32))
t1v = t1v.astype(np.float64); t2v = t2v.astype(np.float64)

fs = load_free_space(D / "freespace.npz")
params = PlanParams()
cand, cclear = _candidates(fs, params)
cand = cand.astype(np.float64); cclear = cclear.astype(np.float64)
occ_op = fs.occ_lin_opaque
origin = fs.origin; pitch = float(fs.pitch); dims = fs.dims
ny, nz = int(dims[1]), int(dims[2])
reach = max(2.0 * float(cclear.max()), 4.0 * params.candidate_spacing)
print(f"[{time.time()-t0:.0f}s] patches={n_patch} candidates={len(cand)} "
      f"(run said 713121) occ_opaque={occ_op.size} reach={reach:.1f}")

focal, finest, n_oct, b = 512.0, 64.0, 6, 3
d_min_ladder = pfeat * (focal / finest)
d_max = pfeat * focal
rng = np.random.default_rng(0)
u = rng.random(n_patch)
oct_top = np.minimum(np.floor(-np.log(np.maximum(u, 1e-12)) / np.log(4.0)).astype(np.int64), n_oct - 1)

# --- ctx: local open-air depth from clearance tiers --------------------------
base = params.candidate_spacing
lev = np.maximum(0, np.floor(np.log2(np.maximum(cclear * 0.5, base) / base))).astype(np.int64)
ctx = np.zeros(n_patch)
for L in np.unique(lev):
    m = lev == L
    tree = cKDTree(cand[m])
    dq, iq = tree.query(ppos, k=1, distance_upper_bound=4.0, workers=-1)
    ok = np.isfinite(dq)
    if ok.any():
        cl = cclear[m][np.minimum(iq[ok], m.sum() - 1)]
        ctx[ok] = np.maximum(ctx[ok], cl)
print(f"[{time.time()-t0:.0f}s] ctx p10/50/90 = "
      f"{np.percentile(ctx,10):.2f}/{np.percentile(ctx,50):.2f}/{np.percentile(ctx,90):.2f} m")

# --- pair build (replica of _build_coverage, minus rescue) -------------------
ctree = cKDTree(cand)
DL = _hemisphere_dirs(int(max(8, 3 * b)))


def march_clear(ci, gi):
    out = np.zeros(len(ci), dtype=bool)
    if not len(ci):
        return out
    dist = np.linalg.norm(cand[ci] - ppos[gi], axis=1)
    order = np.argsort(dist, kind="stable")
    ci_s, gi_s, d_s = ci[order], gi[order], dist[order]
    SAMP = 4_000_000
    s0 = 0
    vis = np.zeros(len(ci), dtype=bool)
    while s0 < len(ci_s):
        # slice sized so pairs*steps stays bounded
        n_steps = int(np.ceil(min(float(d_s[min(s0 + 20000, len(d_s) - 1)]), 220.0) / pitch)) + 2
        take = max(1024, SAMP // n_steps)
        s1 = min(s0 + take, len(ci_s))
        n_steps = int(np.ceil(float(d_s[s1 - 1]) / pitch)) + 2
        cam = cand[ci_s[s0:s1]]; pat = ppos[gi_s[s0:s1]]
        dd = np.maximum(d_s[s0:s1], 1e-9)
        t = np.linspace(0.0, 1.0, n_steps)
        pts = cam[:, None, :] + t[None, :, None] * (pat - cam)[:, None, :]
        idx = np.floor((pts - origin) / pitch).astype(np.int64)
        inb = np.all((idx >= 0) & (idx < dims), axis=2)
        tv = (t[None, :] > (pitch / dd)[:, None]) & (t[None, :] < 1.0 - (1.5 * pitch / dd)[:, None])
        res = ~tv.any(axis=1)  # fail closed
        need = inb & tv
        nzr, nzc = np.nonzero(need)
        if len(nzr):
            ii = idx[nzr, nzc]
            lin = (ii[:, 0] * ny + ii[:, 1]) * nz + ii[:, 2]
            pp_ = np.clip(np.searchsorted(occ_op, lin), 0, occ_op.size - 1)
            hit = occ_op[pp_] == lin
            res[nzr[hit]] = True
        vis[s0:s1] = ~res
        s0 = s1
    out[order] = vis
    return out


def build_pairs(d_min):
    tgt_parts, par_parts = [], []
    rad_ap = np.minimum(1.5 * d_min, reach)
    all_idx = np.arange(n_patch)
    uw = (DL[:, 2][None, :, None] * pnrm[:, None, :]
          + DL[:, 0][None, :, None] * t1v[:, None, :]
          + DL[:, 1][None, :, None] * t2v[:, None, :])
    tgt_parts.append((ppos[:, None, :] + rad_ap[:, None, None] * uw).reshape(-1, 3))
    par_parts.append(np.repeat(all_idx, DL.shape[0]))
    coarse = DL[:3]
    for o in range(1, n_oct):
        r_o = (1.5 * (2.0 ** o)) * d_min
        selo = np.nonzero((oct_top >= o) & (r_o <= reach))[0]
        if selo.size:
            uwc = (coarse[:, 2][None, :, None] * pnrm[selo][:, None, :]
                   + coarse[:, 0][None, :, None] * t1v[selo][:, None, :]
                   + coarse[:, 1][None, :, None] * t2v[selo][:, None, :])
            tgt_parts.append((ppos[selo][:, None, :] + r_o[selo][:, None, None] * uwc).reshape(-1, 3))
            par_parts.append(np.repeat(selo, 3))
    tgt = np.concatenate(tgt_parts); par = np.concatenate(par_parts)
    _, ci = ctree.query(tgt, k=1, workers=-1)
    ci = np.asarray(ci, dtype=np.int64)
    # near probes (k=8)
    _, cknn = ctree.query(ppos, k=8, workers=-1)
    ci = np.concatenate([ci, np.asarray(cknn, dtype=np.int64).ravel()])
    par = np.concatenate([par, np.repeat(all_idx, 8)])
    vd = cand[ci] - ppos[par]
    dist = np.linalg.norm(vd, axis=1)
    ok = dist > 1e-6
    ci, par, vd, dist = ci[ok], par[ok], vd[ok], dist[ok]
    cos = np.abs(np.einsum("mc,mc->m", vd, pnrm[par])) / dist
    d_eff = dist / np.maximum(cos, 1e-12)
    ok = d_eff <= d_max[par]
    ci, par, dist, d_eff = ci[ok], par[ok], dist[ok], d_eff[ok]
    order = np.argsort(dist, kind="stable")
    ci, par, d_eff = ci[order], par[order], d_eff[order]
    _, uidx = np.unique(par * np.int64(len(cand)) + ci, return_index=True)
    ci, par, d_eff = ci[uidx], par[uidx], d_eff[uidx]
    vis = march_clear(ci, par)
    ci, par, d_eff = ci[vis], par[vis], d_eff[vis]
    vd = cand[ci] - ppos[par]
    vd /= np.linalg.norm(vd, axis=1, keepdims=True) + 1e-12
    z_ = np.abs(np.einsum("mc,mc->m", vd, pnrm[par]))
    az = np.arctan2(np.einsum("mc,mc->m", vd, t2v[par]), np.einsum("mc,mc->m", vd, t1v[par]))
    binv = _bin_of(z_, az, b)
    band = _band_of(d_eff, d_min[par], n_oct)
    owed = oct_top[par] >= band
    return ci, par, binv, band, owed


# --- scene classes -----------------------------------------------------------
scene = json.load(open(D / "scene.json")); objs = scene["objects"]
BG = [o for o in objs if o["id"].startswith(("sky_backdrop", "bg_"))]
PROP = [o for o in objs if max(np.array(o["aabb_max"]) - np.array(o["aabb_min"])) < 2.0]
def inside_any(pts, oo, pad=0.15):
    m = np.zeros(len(pts), bool)
    for o in oo:
        lo = np.array(o["aabb_min"]) - pad; hi = np.array(o["aabb_max"]) + pad
        m |= np.all((pts >= lo) & (pts <= hi), axis=1)
    return m
bg_m = inside_any(ppos, BG); prop_m = inside_any(ppos, PROP)
blocks = [o for o in objs if "block" in o["id"] and max(np.array(o["aabb_max"]) - np.array(o["aabb_min"])) < 2.0]

n_units = len(cand) * 6
tie = (np.random.default_rng(0).random(n_units) * 0.5).astype(np.float64)


def run_plan(tag, d_min):
    ci, par, binv, band, owed = build_pairs(d_min)
    print(f"[{time.time()-t0:.0f}s] {tag}: visible pairs={len(ci)}")
    face = _face_of(ppos[par] - cand[ci])
    unit = ci * 6 + face
    chosen, binmask, octmask, gains = _greedy_cover(
        unit, par, binv, band, owed, n_units, n_patch, 2, None, tie, None)
    chosen = np.asarray(chosen, dtype=np.int64)
    cset = set(chosen.tolist())
    inpick = np.isin(unit, chosen)
    cu, pu = ci[inpick], par[inpick]
    dists = np.linalg.norm(cand[cu] - ppos[pu], axis=1)
    campos = cand[np.unique(chosen // 6)]
    lut = np.array([bin(i).count("1") for i in range(1 << b)], dtype=np.int16)
    bins_cov = lut[binmask]
    res = {"images": len(chosen), "positions": len(campos)}
    for name, m in (("prop", prop_m), ("bg", bg_m)):
        pm = m[pu]
        img_ids = np.unique(np.concatenate([chosen[np.isin(chosen, unit[np.isin(par, np.nonzero(m)[0]) & inpick])]])) if pm.any() else []
        res[f"{name}_cover_pairs"] = int(pm.sum())
        res[f"{name}_images"] = len(np.unique(cu[pm] * 6 + 0)) and len(np.unique((cu[pm] * 6 + face[inpick][pm])))
        res[f"{name}_standoff_p50"] = float(np.median(dists[pm])) if pm.any() else 0.0
        res[f"{name}_standoff_p90"] = float(np.percentile(dists[pm], 90)) if pm.any() else 0.0
        res[f"{name}_bins_mean"] = float(bins_cov[m].mean())
        res[f"{name}_bins0_pct"] = float(100 * (bins_cov[m] == 0).mean())
    # per-block camera density: distinct chosen camera POSITIONS covering each block
    per_block = []
    for o in blocks:
        pm = inside_any(ppos, [o])
        if not pm.any():
            continue
        sel = np.isin(pu, np.nonzero(pm)[0])
        per_block.append(len(np.unique(cu[sel])))
    res["cams_per_block_mean"] = float(np.mean(per_block)) if per_block else 0.0
    res["cams_per_block_max"] = int(np.max(per_block)) if per_block else 0
    res["n_blocks"] = len(per_block)
    return res


rA = run_plan("A(current)", d_min_ladder)
d_min_B = np.maximum(d_min_ladder, np.minimum(0.5 * ctx, reach / 1.5))
rB = run_plan("B(ctx-scaled)", d_min_B)

print(f"\n{'metric':32s} {'A(current)':>12s} {'B(ctx)':>12s}")
for k in sorted(set(rA) | set(rB)):
    print(f"{k:32s} {rA.get(k,0):>12.2f} {rB.get(k,0):>12.2f}" if isinstance(rA.get(k), float)
          else f"{k:32s} {rA.get(k,0):>12} {rB.get(k,0):>12}")
print(f"\nreal modal run: images=11768 positions=9990 (sanity anchor for A)")
print(f"[{time.time()-t0:.0f}s] done")
