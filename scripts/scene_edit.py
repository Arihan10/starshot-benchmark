#!/usr/bin/env python3
"""Reusable geometry editor for a cell's event log + baked library GLBs.

A node's world position is stored in TWO places that must stay in sync:
  * the `bbox` event in events.jsonl  -> the wireframe + /scene projection
  * the baked GLB's `place_glb` outer node (translation+scale) -> the rendered
    mesh (the /meshes bundle streams these GLBs byte-for-byte).

This tool edits both. Library GLBs carry a single scene-root node with a
translation + non-uniform scale; for any new target bbox we recompute that node
(deriving the asset's intrinsic post-yaw AABB from the current transform) and
rewrite only the JSON chunk, copying the Meshopt/KTX2 BIN chunk byte-for-byte.
A GLB not in that form (e.g. a Trellis vertex-baked mesh) supports translation
via a wrapper node; scaling such a mesh raises.

CLI:
  python scripts/scene_edit.py report  [--cell RUN/SLOT/MODEL] [--emit fixes.json]
  python scripts/scene_edit.py apply --edits fixes.json [--cell RUN/SLOT/MODEL]
      fixes.json: [{"id": "x", "translate": [dx,dy,dz]},
                   {"id": "y", "bbox": [[ox,oy,oz],[dx,dy,dz]]}]

Library use:
  s = Scene.load(cell_dir)
  for c in s.wall_collisions(): print(c)
  s.translate("vanity_mirror", 0, 0, -0.1)
  s.set_bbox("foo", origin, dims)
  s.save()                       # backs up, rewrites events.jsonl + GLBs
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from starshot_paths import runs_root

DEFAULT_CELL = "v3-iter5/modern-house/opus-new"
RUNS = runs_root()

# wall classification + collision tolerances (meters)
WALL_THIN = 0.6        # a "wall" frame is thin (<this) on X or Z ...
WALL_MIN_H = 1.0       # ... and taller than this on Y
PEN_TOL = 0.05         # embedded deeper than this on the wall's thin axis = collision
FACE_OVERLAP = 0.05    # require this much overlap on the wall's two broad axes
GAP = 0.02             # leave this clearance when pushing an object to a wall face

# ---------------------------------------------------------------- GLB surgery
_GLB_MAGIC = 0x46546C67
_CJSON = 0x4E4F534A
_CBIN = 0x004E4942


def _glb_parse(data: bytes):
    _, _, length = struct.unpack_from("<III", data, 0)
    off, j, b = 12, None, None
    while off < length:
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk = data[off + 8 : off + 8 + clen]
        if ctype == _CJSON:
            j = json.loads(chunk)
        elif ctype == _CBIN:
            b = chunk
        off += 8 + clen
    return j, b


def _glb_write(j: dict, b: bytes | None, dst: Path) -> None:
    jb = json.dumps(j, separators=(",", ":")).encode()
    jb += b" " * ((4 - len(jb) % 4) % 4)
    chunks = [(_CJSON, jb)]
    if b is not None:
        chunks.append((_CBIN, b + b"\x00" * ((4 - len(b) % 4) % 4)))
    total = 12 + sum(8 + len(d) for _, d in chunks)
    out = bytearray(struct.pack("<III", _GLB_MAGIC, 2, total))
    for ct, d in chunks:
        out += struct.pack("<II", len(d), ct) + d
    dst.write_bytes(bytes(out))


def _ctr_ext(origin, dims):
    return ([origin[i] + dims[i] / 2 for i in range(3)], [abs(dims[i]) for i in range(3)])


def _quat_y(deg: float):
    """glTF quaternion [x,y,z,w] for a right-handed yaw about +Y (matches place_glb)."""
    h = math.radians(deg) / 2.0
    return [0.0, math.sin(h), 0.0, math.cos(h)]


def _rot_aabb_y90(bmin, bmax, deg):
    """AABB of the box [bmin,bmax] after a yaw of `deg` (a multiple of 90) about +Y.
    Exact integer axis permutation — no float dust. R_y(90): (x,z)->(z,-x)."""
    k = (round(deg / 90)) % 4
    xs, zs = [], []
    for x in (bmin[0], bmax[0]):
        for z in (bmin[2], bmax[2]):
            xx, zz = x, z
            for _ in range(k):
                xx, zz = zz, -xx
            xs.append(xx)
            zs.append(zz)
    return ([min(xs), bmin[1], min(zs)], [max(xs), bmax[1], max(zs)])


def _recompute_outer(scale, trans, cur_o, cur_d, new_o, new_d):
    """New (translation, scale) for the place_glb outer node so the SAME asset
    fills `new` instead of `cur`. Derives the asset's post-yaw AABB from the
    current (scale, trans, cur bbox); pure-translation collapses to trans+Δ."""
    cc, ce = _ctr_ext(cur_o, cur_d)
    nc, ne = _ctr_ext(new_o, new_d)
    nt, ns = [], []
    for i in range(3):
        if scale[i] == 0 or ce[i] == 0:  # flat axis: place_glb left scale 1 -> translate only
            ns.append(scale[i])
            nt.append(trans[i] + (nc[i] - cc[i]))
            continue
        aext = ce[i] / scale[i]
        actr = (cc[i] - trans[i]) / scale[i]
        s = ne[i] / aext
        ns.append(s)
        nt.append(nc[i] - s * actr)
    return nt, ns


# ---------------------------------------------------------------- Scene model
class Scene:
    def __init__(self, cell_dir: Path):
        self.cell = cell_dir
        self.events = cell_dir / "events.jsonl"
        self.objects = cell_dir / "objects"
        self.nodes: dict[str, dict] = {}
        self.edited: set[str] = set()
        self.orient_edits: dict[str, dict] = {}

    @classmethod
    def load(cls, cell_dir: Path) -> "Scene":
        s = cls(cell_dir)
        model_ids: set[str] = set()
        for line in s.events.open():
            if '"kind": "model"' in line:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("kind") == "model" and isinstance(e.get("id"), str):
                    model_ids.add(e["id"])
            if '"kind": "bbox"' not in line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("kind") != "bbox":
                continue
            o, d = list(e["origin"]), list(e["dimensions"])
            s.nodes[e["id"]] = {
                "id": e["id"], "kind": e.get("node_kind"), "parent": e.get("parent_id"),
                "prompt": e.get("prompt", ""), "ref_o": list(o), "ref_d": list(d),
                "origin": o, "dims": d, "orient": e.get("orientation", 0),
                "mesh": False, "glb": None, "form": None,
            }
        for nid in model_ids:
            n = s.nodes.get(nid)
            p = s.objects / f"{nid}.glb"
            if not n or not p.exists():
                continue
            n["mesh"] = True
            j, _ = _glb_parse(p.read_bytes())
            roots = j["scenes"][j.get("scene", 0)]["nodes"]
            if len(roots) == 1 and j["nodes"][roots[0]].get("translation") and j["nodes"][roots[0]].get("scale"):
                rn = j["nodes"][roots[0]]
                n["glb"] = (list(rn["translation"]), list(rn["scale"]))
                n["form"] = "trs"
            else:
                n["form"] = "other"
        return s

    # geometry helpers ----------------------------------------------------
    def minc(self, nid): n = self.nodes[nid]; return [min(n["origin"][i], n["origin"][i] + n["dims"][i]) for i in range(3)]
    def maxc(self, nid): n = self.nodes[nid]; return [max(n["origin"][i], n["origin"][i] + n["dims"][i]) for i in range(3)]
    def center(self, nid): n = self.nodes[nid]; return [n["origin"][i] + n["dims"][i] / 2 for i in range(3)]
    def ext(self, nid): n = self.nodes[nid]; return [abs(n["dims"][i]) for i in range(3)]

    # edit ops (stage; written on save) ----------------------------------
    def translate(self, nid, dx, dy, dz):
        n = self.nodes[nid]
        for i, d in enumerate((dx, dy, dz)):
            n["origin"][i] = round(n["origin"][i] + d, 6)
        self.edited.add(nid)

    def set_bbox(self, nid, origin, dims):
        n = self.nodes[nid]
        n["origin"], n["dims"] = [round(x, 6) for x in origin], [round(x, 6) for x in dims]
        self.edited.add(nid)

    def set_orientation(self, nid, new_orient):
        """Re-bake a node's yaw. The bbox is unchanged; the GLB's inner yaw quat
        is reset and the outer place_glb scale/translation are recomputed for the
        asset's new post-yaw AABB (a 180 flip leaves scale and mirrors translation
        about the bbox center). Only 90-degree-multiple changes are exact."""
        n = self.nodes[nid]
        if not n["mesh"] or n["form"] != "trs":
            raise NotImplementedError(f"{nid}: orientation re-bake needs a place_glb (trs) mesh")
        d = round(new_orient - n["orient"])
        if d % 90 != 0:
            raise ValueError(f"{nid}: only 90-degree-multiple yaw changes are exact (Δ={d})")
        C, E = self.center(nid), self.ext(nid)
        trans0, scale0 = n["glb"]
        R0 = [E[i] / scale0[i] for i in range(3)]
        Rc0 = [(C[i] - trans0[i]) / scale0[i] for i in range(3)]
        bmin = [Rc0[i] - R0[i] / 2 for i in range(3)]
        bmax = [Rc0[i] + R0[i] / 2 for i in range(3)]
        r1min, r1max = _rot_aabb_y90(bmin, bmax, d)
        R1 = [r1max[i] - r1min[i] for i in range(3)]
        Rc1 = [(r1min[i] + r1max[i]) / 2 for i in range(3)]
        scale1 = [E[i] / R1[i] if R1[i] else scale0[i] for i in range(3)]
        trans1 = [C[i] - scale1[i] * Rc1[i] for i in range(3)]
        n["orient"] = new_orient
        self.orient_edits[nid] = {"orient": new_orient, "scale": scale1, "trans": trans1}

    # ---------------- orientation issues (objects facing into a backed wall) ---
    # id tokens whose object has no meaningful "front" (radially symmetric / flat /
    # directional-by-design) — flipping is pointless or unwanted. Matched as WHOLE
    # underscore tokens so "cabinet"/"matte" don't false-match "bin"/"mat".
    _AGNOSTIC = {"plant", "rug", "mat", "lamp", "bin", "basket", "hamper", "stack",
                 "tray", "cushion", "pillow", "throw", "parasol", "stool", "bowl",
                 "vase", "candle", "planter", "umbrella", "curtain", "runner", "clump",
                 "grass", "shrub", "boulder", "tree", "hedge", "rock", "stone", "cart",
                 "lantern", "pendant", "sconce", "centerpiece", "bottle", "jar", "pot",
                 "books", "side"}

    def _facing(self, deg):
        th = math.radians(deg)
        return (math.sin(th), 0.0, math.cos(th))

    def orientation_issues(self):
        """Objects flush against a wall but facing INTO it. Returns dicts with the
        corrected orientation (facing away from the wall). `agnostic`/`diagonal`
        flag cases to skip from an auto-flip."""
        walls = self.walls()
        out = []
        seen = set()
        for oid, n in self.nodes.items():
            if n["kind"] != "object" or oid in seen:
                continue
            f = self._facing(n["orient"])
            omn, omx, oc = self.minc(oid), self.maxc(oid), self.center(oid)
            for wid in walls:
                wmn, wmx, wc = self.minc(wid), self.maxc(wid), self.center(wid)
                t = 0 if (wmx[0] - wmn[0]) <= (wmx[2] - wmn[2]) else 2
                s = 1 if oc[t] >= wc[t] else -1
                gap = (omn[t] - wmx[t]) if s > 0 else (wmn[t] - omx[t])
                if not (-0.10 <= gap <= 0.40):
                    continue
                perp = 2 if t == 0 else 0
                if min(omx[perp], wmx[perp]) - max(omn[perp], wmn[perp]) < 0.3:
                    continue
                if min(omx[1], wmx[1]) - max(omn[1], wmn[1]) < 0.3:
                    continue
                if f[t] * s >= -0.3:   # not facing into the wall
                    continue
                away = (90 * s) if t == 0 else (0 if s > 0 else 180)  # face away from wall
                d = round(away - n["orient"])
                out.append({"obj": oid, "wall": wid, "axis": "xyz"[t], "cur": n["orient"],
                            "fix": away, "agnostic": bool(set(oid.lower().split("_")) & self._AGNOSTIC),
                            "diagonal": d % 90 != 0})
                seen.add(oid)
                break
        return out

    # wall collision detection -------------------------------------------
    def walls(self):
        out = []
        for nid, n in self.nodes.items():
            if n["kind"] != "frame":
                continue
            ex = self.ext(nid)
            if min(ex[0], ex[2]) < WALL_THIN and ex[1] > WALL_MIN_H:
                out.append(nid)
        return out

    def wall_collisions(self):
        """Every (object, wall) pair where the object is embedded into the wall
        deeper than PEN_TOL on the wall's thin axis (with real face overlap)."""
        walls = self.walls()
        objs = [nid for nid, n in self.nodes.items() if n["kind"] == "object"]
        res = []
        for oid in objs:
            omn, omx, oc = self.minc(oid), self.maxc(oid), self.center(oid)
            for wid in walls:
                wmn, wmx, wc = self.minc(wid), self.maxc(wid), self.center(wid)
                ov = [min(omx[i], wmx[i]) - max(omn[i], wmn[i]) for i in range(3)]
                if min(ov) <= 0:
                    continue  # not a true 3D intersection
                t = 0 if (wmx[0] - wmn[0]) <= (wmx[2] - wmn[2]) else 2  # wall thin axis
                broad = [a for a in (0, 1, 2) if a != t]
                if ov[t] < PEN_TOL or any(ov[a] < FACE_OVERLAP for a in broad):
                    continue
                side = 1 if oc[t] >= wc[t] else -1
                if side > 0:
                    shift = (wmx[t] + GAP) - omn[t]
                else:
                    shift = (wmn[t] - GAP) - omx[t]
                through = omn[t] < wmn[t] - 1e-6 and omx[t] > wmx[t] + 1e-6
                centered = abs(oc[t] - wc[t]) < 0.10
                res.append({
                    "obj": oid, "wall": wid, "axis": "xyz"[t], "depth": round(ov[t], 3),
                    "shift": round(shift, 3), "through": through, "centered": centered,
                    "obj_thin": self.ext(oid)[t] < 0.4,
                })
        res.sort(key=lambda r: -r["depth"])
        return res

    def suggest_fixes(self):
        """One translate per object: the deepest push per horizontal axis."""
        per = {}
        for c in self.wall_collisions():
            d = per.setdefault(c["obj"], {"x": [], "z": [], "flags": set()})
            d[c["axis"]].append(c["shift"])
            if c["through"]:
                d["flags"].add(f"through:{c['wall']}")
            if c["centered"]:
                d["flags"].add(f"centered:{c['wall']}")
        fixes = []
        for oid, d in per.items():
            dx = max(d["x"], key=abs) if d["x"] else 0.0
            dz = max(d["z"], key=abs) if d["z"] else 0.0
            fixes.append({"id": oid, "translate": [round(dx, 3), 0.0, round(dz, 3)],
                          "flags": sorted(d["flags"])})
        fixes.sort(key=lambda f: -max(abs(f["translate"][0]), abs(f["translate"][2])))
        return fixes

    # persistence ---------------------------------------------------------
    def _new_outer(self, nid):
        n = self.nodes[nid]
        if n["form"] == "trs":
            return _recompute_outer(n["glb"][1], n["glb"][0], n["ref_o"], n["ref_d"], n["origin"], n["dims"])
        return None  # handled separately for "other"

    def save(self, dry_run=False):
        if not self.edited and not self.orient_edits:
            print("no staged edits")
            return
        assert not (self.edited & set(self.orient_edits)), "node both moved and reoriented in one pass"
        touched = self.edited | set(self.orient_edits)
        mesh_ids = [i for i in touched if self.nodes[i]["mesh"]]
        for i in self.edited:
            if self.nodes[i]["mesh"] and self.nodes[i]["form"] == "other" and self.nodes[i]["ref_d"] != self.nodes[i]["dims"]:
                raise NotImplementedError(f"{i}: cannot resize a non-place_glb mesh")
        if dry_run:
            for i in sorted(self.edited):
                print(f"  move {i}: {self.nodes[i]['ref_o']} -> {self.nodes[i]['origin']}")
            for i in sorted(self.orient_edits):
                print(f"  reorient {i}: -> {self.orient_edits[i]['orient']}deg")
            return
        # backups (preserve earliest pristine .bak)
        bak = self.events.with_suffix(".jsonl.bak")
        if not bak.exists():
            bak.write_bytes(self.events.read_bytes())
        for i in mesh_ids:
            gb = self.objects / f"{i}.glb.bak"
            if not gb.exists():
                gb.write_bytes((self.objects / f"{i}.glb").read_bytes())
        # rewrite events.jsonl atomically (only touched bbox lines change)
        tmp = self.events.with_suffix(".jsonl.tmp")
        seen = set()
        with self.events.open() as fin, tmp.open("w") as fout:
            for line in fin:
                if '"kind": "bbox"' in line:
                    try:
                        e = json.loads(line)
                    except Exception:
                        e = None
                    if e and e.get("kind") == "bbox" and e["id"] in touched:
                        if e["id"] in self.edited:
                            e["origin"], e["dimensions"] = self.nodes[e["id"]]["origin"], self.nodes[e["id"]]["dims"]
                        if e["id"] in self.orient_edits:
                            e["orientation"] = self.orient_edits[e["id"]]["orient"]
                        line = json.dumps(e) + "\n"
                        seen.add(e["id"])
                fout.write(line)
        tmp.replace(self.events)
        # rewrite GLBs
        for i in mesh_ids:
            n = self.nodes[i]
            p = self.objects / f"{i}.glb"
            j, b = _glb_parse(p.read_bytes())
            roots = j["scenes"][j.get("scene", 0)]["nodes"]
            if i in self.orient_edits:  # re-bake yaw: outer scale/trans + inner yaw quat
                oe = self.orient_edits[i]
                rn = j["nodes"][roots[0]]
                rn["translation"], rn["scale"] = oe["trans"], oe["scale"]
                inner = j["nodes"][rn["children"][0]]
                inner["rotation"] = _quat_y(oe["orient"])
            elif n["form"] == "trs":
                nt, ns = self._new_outer(i)
                rn = j["nodes"][roots[0]]
                rn["translation"], rn["scale"] = nt, ns
            else:  # translation-only fallback: wrap roots under a delta node
                dc = [self.center(i)[k] - _ctr_ext(n["ref_o"], n["ref_d"])[0][k] for k in range(3)]
                j["nodes"].append({"translation": dc, "children": list(roots)})
                j["scenes"][j.get("scene", 0)]["nodes"] = [len(j["nodes"]) - 1]
            _glb_write(j, b, p)
        print(f"applied {len(touched)} node edits "
              f"({len(self.edited)} moved/resized, {len(self.orient_edits)} reoriented; {len(mesh_ids)} meshes); "
              f"{len(touched - seen)} had no bbox line: {sorted(touched - seen)}")


# ---------------------------------------------------------------- CLI
def _resolve_cell(cell: str) -> Path:
    p = RUNS / cell
    if not (p / "events.jsonl").exists():
        sys.exit(f"no events.jsonl under {p}")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["report", "orient", "apply"])
    ap.add_argument("--cell", default=DEFAULT_CELL, help="RUN/SLOT/MODEL under runs/")
    ap.add_argument("--emit", help="(report) write suggested fixes JSON here")
    ap.add_argument("--edits", help="(apply) fixes JSON to apply")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cell = _resolve_cell(a.cell)
    s = Scene.load(cell)

    if a.cmd == "report":
        cols = s.wall_collisions()
        print(f"{len(s.walls())} walls, {sum(1 for n in s.nodes.values() if n['kind']=='object')} objects, "
              f"{len(cols)} (object,wall) collisions deeper than {PEN_TOL}m:\n")
        for c in cols:
            tags = " ".join(t for t in ("through", "centered", "obj_thin") if c[t])
            print(f"  {c['obj']:30s} in {c['wall']:28s} {c['axis']} depth={c['depth']:.2f} "
                  f"push={c['shift']:+.2f}  {tags}")
        fixes = s.suggest_fixes()
        print(f"\nsuggested per-object fixes ({len(fixes)}):")
        for f in fixes:
            print(f"  {f['id']:30s} translate {f['translate']}  {' '.join(f['flags'])}")
        if a.emit:
            Path(a.emit).write_text(json.dumps([{"id": f["id"], "translate": f["translate"]} for f in fixes], indent=2))
            print(f"\nwrote {a.emit}")
        return

    if a.cmd == "orient":
        issues = s.orientation_issues()
        fixable = [i for i in issues if not i["agnostic"] and not i["diagonal"]]
        skip = [i for i in issues if i["agnostic"] or i["diagonal"]]
        print(f"{len(issues)} objects facing INTO a wall they back against "
              f"({len(fixable)} auto-fixable flips, {len(skip)} skipped):\n")
        for i in fixable:
            print(f"  {i['obj']:30s} {i['cur']:>4} -> {i['fix']:>4}  (into {i['wall']} on {i['axis']})")
        print("\n  skipped (agnostic shape / diagonal — review manually):")
        for i in skip:
            why = "agnostic" if i["agnostic"] else "diagonal"
            print(f"    {i['obj']:30s} {i['cur']:>4} ({why}, into {i['wall']})")
        if a.emit:
            Path(a.emit).write_text(json.dumps([{"id": i["obj"], "orientation": i["fix"]} for i in fixable], indent=2))
            print(f"\nwrote {a.emit}")
        return

    if a.cmd == "apply":
        if not a.edits:
            sys.exit("apply needs --edits")
        for e in json.loads(Path(a.edits).read_text()):
            if "translate" in e:
                s.translate(e["id"], *e["translate"])
            elif "bbox" in e:
                s.set_bbox(e["id"], e["bbox"][0], e["bbox"][1])
            elif "orientation" in e:
                s.set_orientation(e["id"], e["orientation"])
        s.save(dry_run=a.dry_run)


if __name__ == "__main__":
    main()
