import json, struct, collections, pathlib

def glb_chunks(path):
    with open(path, "rb") as f:
        data = f.read()
    length = struct.unpack("<I", data[8:12])[0]
    off = 12
    js = None
    bin_off = bin_len = None
    while off < length:
        clen, ctype = struct.unpack("<II", data[off:off+8])
        off += 8
        if ctype == 0x4E4F534A:
            js = json.loads(data[off:off+clen])
        elif ctype == 0x004E4942:
            bin_off, bin_len = off, clen
        off += clen
    return js, data, bin_off

d = pathlib.Path(r"runs/good_opus_new_hotel2/hotel-room/gemini-pro/generated/1/objects-generated-optimized")

alphamode = collections.Counter()
alpha_lt = 0
exts = collections.Counter()
mimes = collections.Counter()
files = sorted(d.glob("*.glb"))
for glb in files:
    j, _, _ = glb_chunks(glb)
    for e in j.get("extensionsUsed", []): exts[e] += 1
    for m in j.get("materials", []):
        am = m.get("alphaMode", "OPAQUE")
        alphamode[am] += 1
        a = m.get("pbrMetallicRoughness", {}).get("baseColorFactor", [1,1,1,1])[3]
        if a < 0.999: alpha_lt += 1
    for im in j.get("images", []):
        mimes[im.get("mimeType", "?")] += 1

print(f"files: {len(files)}")
print("\nalphaMode:", dict(alphamode))
print("materials with baseColorFactor.a < 1:", alpha_lt)
print("extensionsUsed:", dict(exts))
print("image mimeTypes:", dict(mimes))

# Now decode PNG base-color textures for a few objects and measure the alpha channel.
# Pull the first image per glb, check if PNG has alpha and its mean/min.
import zlib
def png_alpha_stats(raw):
    # minimal PNG parse: find IHDR (color type) and note if alpha present
    if raw[:8] != b"\x89PNG\r\n\x1a\n": return None
    off = 8
    ctype = None
    while off < len(raw):
        ln = struct.unpack(">I", raw[off:off+4])[0]
        typ = raw[off+4:off+8]
        if typ == b"IHDR":
            ctype = raw[off+8+9]
            break
        off += 12 + ln
    # color type 6 = RGBA, 4 = grey+alpha; else no alpha
    return {"colorType": ctype, "hasAlpha": ctype in (4, 6)}

print("\nper-object first-image alpha (sample of 16):")
for glb in files[:16]:
    j, data, bin_off = glb_chunks(glb)
    imgs = j.get("images", [])
    bvs = j.get("bufferViews", [])
    if not imgs:
        print(f"  {glb.name}: no images"); continue
    bv = bvs[imgs[0]["bufferView"]]
    start = bin_off + bv.get("byteOffset", 0)
    raw = data[start:start+bv["byteLength"]]
    st = png_alpha_stats(raw)
    print(f"  {glb.name}: mime={imgs[0].get('mimeType')} {st}")
