import json, struct, collections

from starshot_paths import runs_root

def glb_chunks(path):
    b = open(path, "rb").read()
    L = struct.unpack("<I", b[8:12])[0]
    off = 12; js=None; bin_off=None
    while off < L:
        cl, ct = struct.unpack("<II", b[off:off+8]); off += 8
        if ct == 0x4E4F534A: js = json.loads(b[off:off+cl])
        elif ct == 0x004E4942: bin_off = off
        off += cl
    return js, b, bin_off

def ktx2_info(raw):
    # header
    if raw[:12] != bytes([0xAB,0x4B,0x54,0x58,0x20,0x32,0x30,0xBB,0x0D,0x0A,0x1A,0x0A]):
        return None
    supercomp = struct.unpack("<I", raw[44:48])[0]
    dfdOff = struct.unpack("<I", raw[48:52])[0]
    # basic block
    word1 = struct.unpack("<I", raw[dfdOff+8:dfdOff+12])[0]
    blockSize = (word1 >> 16) & 0xFFFF
    colorModel = raw[dfdOff+12]
    nSamples = (blockSize - 24) // 16
    # sample channel ids (byte at sample+3 low nibble = channelType)
    chans = []
    for i in range(nSamples):
        s = dfdOff + 4 + 24 + i*16
        chanType = raw[s+3] & 0x0F
        chans.append(chanType)
    return {"supercomp": supercomp, "colorModel": colorModel, "nSamples": nSamples, "chans": chans}

# ETC1S channel ids: 0=RGB, 15=AAA(alpha). UASTC: 3=RGBA. colorModel 163=ETC1S,166=UASTC
d = runs_root() / "good_opus_new_hotel2/hotel-room/gemini-pro/generated/1/objects-generated-optimized"
base_alpha = collections.Counter()
models = collections.Counter()
samples_dist = collections.Counter()
n = 0
for glb in sorted(d.glob("*.glb")):
    j, b, bin_off = glb_chunks(glb)
    bvs = j.get("bufferViews", [])
    texs = j.get("textures", [])
    imgs = j.get("images", [])
    for m in j.get("materials", []):
        bct = m.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if not bct: continue
        t = texs[bct["index"]]
        src = t.get("extensions", {}).get("KHR_texture_basisu", {}).get("source", t.get("source"))
        if src is None: continue
        bv = bvs[imgs[src]["bufferView"]]
        start = bin_off + bv.get("byteOffset", 0)
        raw = b[start:start+bv["byteLength"]]
        info = ktx2_info(raw)
        if not info: continue
        n += 1
        models[info["colorModel"]] += 1
        samples_dist[info["nSamples"]] += 1
        has_alpha = (15 in info["chans"]) or (info["colorModel"] == 166 and 3 in info["chans"]) or info["nSamples"] >= 2
        base_alpha["alpha" if has_alpha else "no-alpha"] += 1

print(f"base-color KTX2 textures inspected: {n}")
print("colorModel (163=ETC1S,166=UASTC):", dict(models))
print("nSamples distribution:", dict(samples_dist))
print("base-color alpha channel:", dict(base_alpha))
