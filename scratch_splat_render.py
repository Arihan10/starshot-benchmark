"""Throwaway diagnostic: CPU 2DGS rasterizer, renders cloud/trained/healed .ply
from a Stage-5 reference pose and builds a side-by-side sheet with the reference.
Delete this file (and scratch_out/) when done."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, ".")
from splat.stage5 import load_reference_frame

BASE = Path("runs/good_opus_new_hotel2/hotel-room/gemini-pro/splat")
OUT = Path("scratch_out")
C0 = 0.28209479177387814


def load(path):
    raw = Path(path).read_bytes()
    cut = raw.find(b"end_header\n")
    count, props, in_vertex = 0, [], False
    for line in raw[:cut].decode("ascii", "replace").splitlines():
        t = line.split()
        if t[:1] == ["element"]:
            in_vertex = t[1] == "vertex"
            if in_vertex:
                count = int(t[2])
        elif in_vertex and t[:1] == ["property"]:
            props.append(t[2])
    tab = np.frombuffer(raw, dtype="<f4", count=count * len(props),
                        offset=cut + 11).reshape(count, len(props))
    return {n: tab[:, i] for i, n in enumerate(props)}, count


def sh_basis(d):
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    return np.stack([np.full_like(x, C0),
                     -0.4886025119 * y, 0.4886025119 * z, -0.4886025119 * x,
                     1.0925484306 * x * y, -1.0925484306 * y * z,
                     0.3153915653 * (2 * zz - xx - yy),
                     -1.0925484306 * x * z, 0.5462742153 * (xx - yy),
                     -0.5900435899 * y * (3 * xx - yy), 2.8906114426 * x * y * z,
                     -0.4570457995 * y * (4 * zz - xx - yy),
                     0.3731763326 * z * (2 * zz - 3 * xx - 3 * yy),
                     -0.4570457995 * x * (4 * zz - xx - yy),
                     1.4453057213 * z * (xx - yy),
                     -0.5900435899 * x * (xx - 3 * yy)], 1)


def quat_to_R(q):
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def render(path, c2w, K, res):
    col, n = load(path)
    means = np.stack([col["x"], col["y"], col["z"]], 1).astype(np.float32)
    quats = np.stack([col[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)
    s = np.exp(np.stack([col["scale_0"], col["scale_1"]], 1)).astype(np.float32)
    opa = 1 / (1 + np.exp(-col["opacity"].astype(np.float32)))
    dc = np.stack([col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]], 1).astype(np.float32)
    rest = sorted((k for k in col if k.startswith("f_rest_")), key=lambda t: int(t.rsplit("_", 1)[-1]))
    R_rest = (np.stack([col[k] for k in rest], 1).reshape(n, 3, -1).astype(np.float32)
              if rest else np.zeros((n, 3, 0), np.float32))

    w2c = np.linalg.inv(c2w)
    Rw, tw = w2c[:3, :3].astype(np.float32), w2c[:3, 3].astype(np.float32)
    pc = means @ Rw.T + tw
    z = np.maximum(pc[:, 2], 1e-6)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * pc[:, 0] / z + cx
    v = fy * pc[:, 1] / z + cy

    Rg = quat_to_R(quats)
    def proj(t):
        t = t @ Rw.T
        return np.stack([fx * (t[:, 0] / z - pc[:, 0] * t[:, 2] / z ** 2),
                         fy * (t[:, 1] / z - pc[:, 1] * t[:, 2] / z ** 2)], 1)
    a1, a2 = proj(Rg[:, :, 0] * s[:, 0:1]), proj(Rg[:, :, 1] * s[:, 1:2])
    cov = (a1[:, :, None] * a1[:, None, :] + a2[:, :, None] * a2[:, None, :]
           + np.eye(2, dtype=np.float32)[None] * 0.25)
    det = np.maximum(cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] ** 2, 1e-9)
    rad = 3.0 * np.sqrt(np.maximum(cov[:, 0, 0], cov[:, 1, 1]))
    keep = ((pc[:, 2] > 0.05) & (rad < res) & (u + rad > 0) & (u - rad < res)
            & (v + rad > 0) & (v - rad < res))

    d = means - c2w[:3, 3].astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    rgb = 0.5 + C0 * dc
    if R_rest.shape[2]:
        rgb = rgb + np.einsum("nk,nck->nc", sh_basis(d)[:, 1:1 + R_rest.shape[2]], R_rest)
    rgb = np.clip(rgb, 0, None)

    inv = np.empty_like(cov)
    inv[:, 0, 0], inv[:, 1, 1] = cov[:, 1, 1] / det, cov[:, 0, 0] / det
    inv[:, 0, 1] = inv[:, 1, 0] = -cov[:, 0, 1] / det

    idx = np.nonzero(keep)[0]
    idx = idx[np.argsort(z[idx])]
    img = np.zeros((res, res, 3), np.float32)
    T = np.ones((res, res), np.float32)
    grid = np.arange(res, dtype=np.float32)
    for i in idx:
        r = rad[i]
        x0, x1 = max(int(u[i] - r), 0), min(int(u[i] + r) + 1, res)
        y0, y1 = max(int(v[i] - r), 0), min(int(v[i] + r) + 1, res)
        if x1 <= x0 or y1 <= y0:
            continue
        Tl = T[y0:y1, x0:x1]
        if Tl.max() < 3e-3:
            continue
        dx, dy = grid[x0:x1] - u[i], grid[y0:y1] - v[i]
        m = inv[i]
        q = (m[0, 0] * dx[None, :] ** 2 + 2 * m[0, 1] * dx[None, :] * dy[:, None]
             + m[1, 1] * dy[:, None] ** 2)
        a = opa[i] * np.exp(-0.5 * np.minimum(q, 60.0))
        img[y0:y1, x0:x1] += (a * Tl)[..., None] * rgb[i]
        T[y0:y1, x0:x1] = Tl * (1 - a)
    return img


def main(frames):
    doc = json.loads((BASE / "refs" / "transforms.json").read_text())
    res = 512
    sc = res / doc["w"]
    K = np.array([[doc["fl_x"] * sc, 0, doc["cx"] * sc],
                  [0, doc["fl_y"] * sc, doc["cy"] * sc], [0, 0, 1]])
    OUT.mkdir(exist_ok=True)
    labelled = [("reference (stage 5 capture)", None),
                ("cloud.ply  (stage 3 init)", "cloud.ply"),
                ("trained.ply  (stage 6)", "trained.ply"),
                ("healed.ply  (stage 7)", "healed.ply")]
    rows = []
    for fi in frames:
        fr = doc["frames"][fi]
        c2w = np.asarray(fr["transform_matrix"], np.float64)
        row = Image.new("RGB", (res * len(labelled), res))
        for j, (label, name) in enumerate(labelled):
            if name is None:
                rgba, _ = load_reference_frame(BASE / "refs" / fr["frame_path"])
                im = Image.fromarray(rgba[..., :3]).resize((res, res), Image.BILINEAR)
            else:
                img = render(BASE / name, c2w, K, res)
                print(f"[{fi}] {name}: mean luminance {img.mean():.3f}")
                im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, res, 20], fill=(0, 0, 0))
            dr.text((6, 5), label, fill=(255, 235, 120))
            row.paste(im, (res * j, 0))
        rows.append(row)
    sheet = Image.new("RGB", (res * len(labelled), res * len(rows)))
    for i, r in enumerate(rows):
        sheet.paste(r, (0, res * i))
    sheet = sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS)
    sheet.save(OUT / "comparison.png")
    print("wrote", (OUT / "comparison.png").resolve())


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [30, 2900])
