// =============================================================================
// tourcapture.js — the HEADLESS matterport tour-capture worker.
// =============================================================================
//
// This is the server-side replacement for the old client-driven capture (the
// prod-client's client/public/tourCapture.js, which an operator drove inside the
// interactive viewer). It runs the SAME way Stage 5 renders its references — a
// headless browser (launched by server/app/api/routes.py `_run_tour_capture` via
// services/refcapture.py) running the shared LIT WebGL stack against the cell's
// mesh tier — so capture is server-side, GPU-fast, and high-resolution, with no
// operator. For one cell it produces the matterport walkthrough's artifacts:
//
//   • 360° panoramas   one equirectangular JPEG per LLM-planned anchor point
//   • projection proxy  a merged, world-space, material-free GLB the server
//                        decimates; the walkthrough projects the panos onto it
//   • bird's-eye minimaps  one top-down slice per storey (Y level)
//   • tour.json manifest    positions + filenames tying the above together
//
// Rendering goes through capturecore.js — the SAME pipeline Stage 5 uses, not a
// lookalike: the same renderer configuration, the same per-object material prep
// (reflective.js's matte discriminator + oit.js's transparent-mesh patch), the
// same emissive / sun / IBL / shadow / reflection-probe bakes, and the same
// weighted-blended OIT + ACES-filmic present. A pano is therefore the same pixels
// a Stage-5 reference would be from that position; only the camera pattern (cube
// faces → equirect) differs.
//
// Protocol (server = server/app/api/routes.py):
//   1. GET  {api}/runs/{run}/tour/capture/{slot}/{model}/manifest?token=…
//        → { bundle_url, lighting, background, anchors, connectors,
//            planner_model, namer_model, planner_reasoning, pano, minimap }
//   2. fetch bundle_url (the SMB1 mesh bundle) → build the lit scene
//   3. per anchor: render 6 cube faces (90° fov) → CPU-stitch to an equirect
//      JPEG → PUT …/tour/pano/{id}?run=…
//   4. build the storey minimaps → PUT …/tour/minimap/{id}?run=…
//   5. bake + POST the merged proxy → POST …/tour/proxy?run=…
//   6. POST …/tour/capture/{slot}/{model}/finish?token=… with the tour manifest
//      → the server writes tour.json and publishes to R2/D1.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { normalizeLighting } from "./splatlight.js";
// The render pipeline is SHARED with Stage 5 (splatcapture.js) — see
// capturecore.js. The ONLY difference between the two captures is the camera
// pattern: six 90° cube faces per anchor here, the planned directed views there.
import {
    createCapture,
    createCaptureRenderer,
    prepareCaptureObject,
    readTargetTopDown,
    setupCaptureLighting,
} from "./capturecore.js";

const DECODE_WORKERS = Math.min(4, navigator.hardwareConcurrency || 2);
const MAX_PARSE_INFLIGHT = 4; // concurrent GLB parses while streaming the bundle
const MESH_BUNDLE_MAGIC = "SMB1";

// --- capture tunables (defaults; the server manifest overrides) --------------
const DEFAULT_FACE_SIZE = 1024; // device px per cube face
const DEFAULT_PANO_WIDTH = 4096; // equirect output width cap (height = width / 2)
const DEFAULT_JPEG_QUALITY = 0.92;
// Bird's-eye minimap knobs. These are FALLBACKS only: the manifest carries the
// values the server actually used, and `readMinimapOpts` prefers those. In
// particular the level gap is the server's own FLOOR_LEVEL_EPS (anchors.py), so
// the floors it named and the slices we cut here can no longer drift apart —
// which is what the old hardcoded twin of that constant risked every time either
// side was retuned.
const DEFAULT_MINIMAP_LEVEL_EPS = 1.5; // metres; anchors within this Y gap share a level
const DEFAULT_MINIMAP_RES = 1024; // longest output side in device px
const DEFAULT_MINIMAP_PAD_FRAC = 0.04; // breathing room around the scene footprint
const DEFAULT_MINIMAP_SLICE_BELOW = 2; // metres below the level's lowest anchor for the floor cut

// One numeric manifest knob, with its fallback. Deliberately not `||`: a
// legitimate 0 (slice_below: 0 cuts exactly at the lowest anchor) must survive,
// while a missing or non-numeric field from a partial manifest still falls back.
function num(value, fallback) {
    return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

// --- the map's frame ---------------------------------------------------------
//
// The bird's-eye map used to be exactly that: look straight down, keep X and Z,
// throw away Y. That is right for a building and wrong for anything shaped
// differently — a side-on level is 200 m wide, 60 m tall and 10 m deep, and
// flattening it from above discards the only axis that says where you are.
//
// So the capture profile (anchors.py `choose_profile`) names WHERE THE MAP CAMERA
// STANDS — `view_from`, the axis it sits on looking back — and WHICH WAY IS DOWN
// the page. Everything else follows: the third direction is fixed by those two, the
// axis named by `view_from` is the one flattened away, and it is also the axis
// storeys are stacked along.
//
// A manifest without a profile gets the plan view every scene had before this,
// so nothing about an existing tour changes.
const AXIS_OF = { X: 0, Y: 1, Z: 2 };

function axisVec(a) {
    const v = [0, 0, 0];
    v[AXIS_OF[a.slice(-1).toUpperCase()]] = a.trim().startsWith("-") ? -1 : 1;
    return v;
}

function readBasis(profile) {
    const viewFrom =
        typeof profile?.view_from === "string" ? profile.view_from : "+Y";
    const imageDown =
        typeof profile?.image_down === "string" ? profile.image_down : "+Z";
    let from;
    let down;
    try {
        from = axisVec(viewFrom);
        down = axisVec(imageDown);
    } catch {
        from = [0, 1, 0];
        down = [0, 0, 1];
    }
    const forward = from.map((v) => -v); // the camera looks back at the scene
    const up = down.map((v) => -v);
    const right = [
        forward[1] * up[2] - forward[2] * up[1],
        forward[2] * up[0] - forward[0] * up[2],
        forward[0] * up[1] - forward[1] * up[0],
    ];
    const axis = from[0] !== 0 ? 0 : from[1] !== 0 ? 1 : 2;
    return {
        viewFrom,
        imageDown,
        axis, // world component the map flattens away
        sign: from[axis], // which side of it the camera stands on
        right,
        down,
        up,
        forward,
    };
}

// The manifest's `minimap` block, resolved against the defaults above.
function readMinimapOpts(manifest) {
    const m = manifest.minimap || {};
    return {
        levelEps: num(m.level_eps, DEFAULT_MINIMAP_LEVEL_EPS),
        res: num(m.res, DEFAULT_MINIMAP_RES),
        padFrac: num(m.pad_frac, DEFAULT_MINIMAP_PAD_FRAC),
        sliceBelow: num(m.slice_below, DEFAULT_MINIMAP_SLICE_BELOW),
    };
}

// forward/up per face; right = cross(forward, up) — matches what lookAt builds,
// so the analytic projection in the stitch agrees with the render exactly.
// Order: +X, -X, +Y, -Y, +Z, -Z.
const PANO_FACES = [
    { f: [1, 0, 0], up: [0, 1, 0] },
    { f: [-1, 0, 0], up: [0, 1, 0] },
    { f: [0, 1, 0], up: [0, 0, 1] },
    { f: [0, -1, 0], up: [0, 0, -1] },
    { f: [0, 0, 1], up: [0, 1, 0] },
    { f: [0, 0, -1], up: [0, 1, 0] },
];
const PANO_FACE_BASIS = PANO_FACES.map(({ f, up }) => ({
    f,
    up,
    right: [
        f[1] * up[2] - f[2] * up[1],
        f[2] * up[0] - f[0] * up[2],
        f[0] * up[1] - f[1] * up[0],
    ],
}));

const params = new URLSearchParams(location.search);
const API = (params.get("api") || "").replace(/\/$/, "");
const RUN = params.get("run");
const SLOT = params.get("slot");
const MODEL = params.get("model");
const TOKEN = params.get("token") || "";

const statusEl = document.getElementById("status");
const lines = [];
function status(text, cls = "") {
    lines.push(cls ? `<span class="${cls}">${text}</span>` : text);
    if (lines.length > 30) lines.shift();
    statusEl.innerHTML = lines.join("\n");
    console.log(`[tour] ${text}`);
}
function progress(text) {
    statusEl.innerHTML = lines.join("\n") + "\n" + text;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// The capture-session protocol (token-authed) vs. the tour persistence sinks
// (run-scoped, shared with any manual flow).
function stagePath(tail) {
    return (
        `${API}/runs/${encodeURIComponent(RUN)}/tour/capture/` +
        `${encodeURIComponent(SLOT)}/${encodeURIComponent(MODEL)}${tail}` +
        `?token=${encodeURIComponent(TOKEN)}`
    );
}
function cellPath(tail) {
    return (
        `${API}/slots/${encodeURIComponent(SLOT)}/${encodeURIComponent(MODEL)}` +
        `${tail}?run=${encodeURIComponent(RUN)}`
    );
}

// Heartbeat so the server's stall timer only fires on a genuinely dead session.
// Uploads already beat; this covers the upload-free phases (scene load, the
// shadow/reflection bakes, and the projection-proxy bake).
async function beat() {
    try {
        await fetch(stagePath("/beat"), { method: "POST" });
    } catch {
        /* transient — the next upload or beat will refresh it */
    }
}

// A steady background heartbeat for the WHOLE capture. Any phase that yields to
// the event loop (even briefly) stays alive without sprinkling beat() everywhere;
// a phase that blocks the main thread (e.g. the monolithic GLB serialize) still
// needs a fresh beat right before it + a generous server stall window. Big scenes
// (hundreds of objects) spend real time in the proxy bake, and a silent capture
// there is exactly what used to trip the stall.
let _keepalive = null;
function startKeepalive() {
    stopKeepalive();
    _keepalive = setInterval(() => void beat(), 5000);
}
function stopKeepalive() {
    if (_keepalive) {
        clearInterval(_keepalive);
        _keepalive = null;
    }
}

// --- scene loading (SMB1 bundle → GLTFLoader.parse) --------------------------
// Mirrors splatcapture.js: one streamed request for the whole scene, each GLB
// parsed as it arrives and handed to capturecore's shared per-object prep, which
// also stamps its id so the proxy keeps per-object identity.

function byteStreamReader(reader) {
    const chunks = [];
    let avail = 0;
    let ended = false;
    async function readExact(n) {
        while (avail < n) {
            if (ended) return null;
            const { done, value } = await reader.read();
            if (done) {
                ended = true;
                continue;
            }
            if (value && value.length) {
                chunks.push(value);
                avail += value.length;
            }
        }
        const out = new Uint8Array(n);
        let filled = 0;
        while (filled < n) {
            const c = chunks[0];
            const take = Math.min(c.length, n - filled);
            out.set(c.subarray(0, take), filled);
            filled += take;
            if (take === c.length) chunks.shift();
            else chunks[0] = c.subarray(take);
            avail -= take;
        }
        return out;
    }
    return { readExact };
}

async function loadScene(renderer, bundleUrl) {
    const ktx2 = new KTX2Loader()
        .setTranscoderPath("/vendor/three/examples/jsm/libs/basis/")
        .setWorkerLimit(DECODE_WORKERS)
        .detectSupport(renderer);
    MeshoptDecoder.useWorkers(DECODE_WORKERS);
    const loader = new GLTFLoader()
        .setKTX2Loader(ktx2)
        .setMeshoptDecoder(MeshoptDecoder);
    const parseGlb = (buf) =>
        new Promise((resolve, reject) => loader.parse(buf, "", resolve, reject));

    const scene = new THREE.Scene();
    const res = await fetch(bundleUrl, { cache: "no-store" });
    if (!res.ok || !res.body) throw new Error(`mesh bundle HTTP ${res.status}`);
    const r = byteStreamReader(res.body.getReader());
    const dec = new TextDecoder();
    const magic = await r.readExact(4);
    if (!magic || dec.decode(magic) !== MESH_BUNDLE_MAGIC) {
        throw new Error("mesh bundle: bad magic (not SMB1)");
    }
    let loaded = 0;
    let failed = 0;
    const inflight = new Set();
    while (true) {
        const idLenB = await r.readExact(4);
        if (!idLenB) break;
        const idB = await r.readExact(new DataView(idLenB.buffer).getUint32(0, true));
        if (!idB) break;
        const id = dec.decode(idB);
        const glbLenB = await r.readExact(4);
        if (!glbLenB) break;
        const glbB = await r.readExact(new DataView(glbLenB.buffer).getUint32(0, true));
        if (!glbB) break;
        const p = (async () => {
            try {
                const gltf = await parseGlb(glbB.buffer);
                // id stamp (emissive.js / reflective.js match on it) + the matte
                // discriminator + the OIT patch — the same prep Stage 5 does, so
                // the panos shade identically. See capturecore.js.
                prepareCaptureObject(gltf.scene, id);
                scene.add(gltf.scene);
                loaded += 1;
            } catch (e) {
                failed += 1;
                console.warn(`[tour] ${id}: parse failed`, e);
            }
            progress(`loading scene… ${loaded + failed} objects`);
        })().finally(() => inflight.delete(p));
        inflight.add(p);
        if (inflight.size >= MAX_PARSE_INFLIGHT) await Promise.race(inflight);
    }
    await Promise.allSettled(inflight);
    return { scene, loaded, failed };
}

// --- renderer ----------------------------------------------------------------

function rendererName(renderer) {
    const gl = renderer.getContext();
    const info = gl.getExtension("WEBGL_debug_renderer_info");
    return info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : "unknown";
}

// Read the current drawing buffer as a top-down {data,width,height} (readPixels
// is bottom-up; flip rows so it matches the equirect stitch's image convention).
// --- 360° panorama capture ---------------------------------------------------

// Render the six cube faces (90° fov) from `pos` through the SHARED capture
// pipeline and read each back top-down. Each face is one `renderView` — the exact
// call Stage 5 makes for a reference view — so glass, reflections, shadows and the
// ACES present are identical. `opaque` forces alpha to 255: rtColor's alpha is OIT
// coverage (0 over background), and a JPEG needs solid pixels.
function renderPanoFaces(renderer, capture, scene, pos, faceSize) {
    const faces = [];
    for (const { f, up } of PANO_FACES) {
        capture.renderView(scene, pos, { forward: f, up });
        faces.push(readTargetTopDown(renderer, capture.rtColor, faceSize, faceSize, true));
    }
    return faces;
}

// Stitch six top-down face images into one equirect ImageData (bilinear
// sampling), then JPEG-encode. Chunked by rows so the page stays responsive.
async function stitchPanoBlob(faces, faceSize, widthCap, quality, onProgress) {
    const W = Math.min(widthCap, faceSize * 4);
    const H = W / 2;
    const out = new ImageData(W, H);
    const o = out.data;
    const S = faceSize;
    const maxIdx = S - 1;

    for (let row = 0; row < H; row++) {
        const v = 1 - (row + 0.5) / H;
        const phi = (v - 0.5) * Math.PI;
        const dy = Math.sin(phi);
        const cosPhi = Math.cos(phi);
        let oi = row * W * 4;
        for (let col = 0; col < W; col++, oi += 4) {
            const az = ((col + 0.5) / W - 0.5) * 2 * Math.PI;
            const dx = cosPhi * Math.cos(az);
            const dz = cosPhi * Math.sin(az);

            const ax = Math.abs(dx);
            const ay = Math.abs(dy);
            const az2 = Math.abs(dz);
            let faceIdx;
            if (ax >= ay && ax >= az2) faceIdx = dx > 0 ? 0 : 1;
            else if (ay >= az2) faceIdx = dy > 0 ? 2 : 3;
            else faceIdx = dz > 0 ? 4 : 5;

            const { f, up, right } = PANO_FACE_BASIS[faceIdx];
            const t = dx * f[0] + dy * f[1] + dz * f[2];
            const u2 = (dx * right[0] + dy * right[1] + dz * right[2]) / t;
            const v2 = (dx * up[0] + dy * up[1] + dz * up[2]) / t;

            const px = (u2 * 0.5 + 0.5) * S - 0.5;
            const py = (0.5 - v2 * 0.5) * S - 0.5;
            let x0 = Math.floor(px);
            let y0 = Math.floor(py);
            const fx = px - x0;
            const fy = py - y0;
            x0 = x0 < 0 ? 0 : x0 > maxIdx ? maxIdx : x0;
            y0 = y0 < 0 ? 0 : y0 > maxIdx ? maxIdx : y0;
            const x1 = x0 < maxIdx ? x0 + 1 : maxIdx;
            const y1 = y0 < maxIdx ? y0 + 1 : maxIdx;

            const d = faces[faceIdx].data;
            const i00 = (y0 * S + x0) * 4;
            const i10 = (y0 * S + x1) * 4;
            const i01 = (y1 * S + x0) * 4;
            const i11 = (y1 * S + x1) * 4;
            const w00 = (1 - fx) * (1 - fy);
            const w10 = fx * (1 - fy);
            const w01 = (1 - fx) * fy;
            const w11 = fx * fy;

            o[oi] = d[i00] * w00 + d[i10] * w10 + d[i01] * w01 + d[i11] * w11;
            o[oi + 1] =
                d[i00 + 1] * w00 + d[i10 + 1] * w10 + d[i01 + 1] * w01 + d[i11 + 1] * w11;
            o[oi + 2] =
                d[i00 + 2] * w00 + d[i10 + 2] * w10 + d[i01 + 2] * w01 + d[i11 + 2] * w11;
            o[oi + 3] = 255;
        }
        if (row % 128 === 127) {
            onProgress?.(row / H);
            await sleep(0);
        }
    }

    const outCanvas = document.createElement("canvas");
    outCanvas.width = W;
    outCanvas.height = H;
    outCanvas.getContext("2d").putImageData(out, 0, 0);
    return new Promise((resolve, reject) =>
        outCanvas.toBlob(
            (blob) => (blob ? resolve(blob) : reject(new Error("JPEG encode failed"))),
            "image/jpeg",
            quality,
        ),
    );
}

// --- bird's-eye minimap slices (one per Y level) -----------------------------

// Cluster captures into levels by gap ALONG THE MAP'S FLATTENED AXIS; returns
// [{ y, minY, indices }] low→high. `levelEps` is the server's own clustering gap
// (see readMinimapOpts), so these slices match the floors it named.
//
// Only the fallback path reaches this — a plan with floors ships its own grouping
// — but a plan rebuilt from a captured tour.json loses the membership while KEEPING
// the profile, so the axis has to be honoured here too or such a re-capture would
// cluster a diorama by height.
function groupAnchorLevels(positions, levelEps, axis = 1) {
    const order = positions
        .map((_, i) => i)
        .sort((a, b) => positions[a][axis] - positions[b][axis]);
    const groups = [];
    let cur = null;
    for (const i of order) {
        const y = positions[i][axis];
        if (!cur || y - cur.lastY > levelEps) {
            cur = { indices: [], ys: [], lastY: y };
            groups.push(cur);
        }
        cur.indices.push(i);
        cur.ys.push(y);
        cur.lastY = y;
    }
    return groups.map((g) => {
        const ys = g.ys.slice().sort((a, b) => a - b);
        return { y: ys[(ys.length - 1) >> 1], minY: ys[0], indices: g.indices };
    });
}

// Render one orthographic slice of the scene into a PNG blob: a slab bounded by
// two planes perpendicular to the map's flattened axis, seen from `basis`.
//
// For a plan view that is the familiar thing — a horizontal slab at camera height
// with the roof cut open. For an elevation it is a slab of DEPTH with the near
// face cut away, which is the same idea pointed sideways: remove what is between
// the camera and the scene so you can see in.
//
// `bounds` is in the map's own (u, v) frame — u across the page, v down it — so
// the stored rectangle maps world positions to image positions without the viewer
// needing to know which world axes those were.
async function captureMinimapBlob(
    renderer,
    scene,
    bounds,
    basis,
    cutNear,
    cutFar,
    sliceLo,
    sliceHi,
    background,
    exposure,
    res,
) {
    const W = bounds.maxU - bounds.minU;
    const D = bounds.maxV - bounds.minV;
    const cu = (bounds.minU + bounds.maxU) / 2;
    const cv = (bounds.minV + bounds.maxV) / 2;

    let pw;
    let ph;
    if (W >= D) {
        pw = res;
        ph = Math.max(1, Math.round((res * D) / W));
    } else {
        ph = res;
        pw = Math.max(1, Math.round((res * W) / D));
    }

    // Stand the camera clear of the near face, looking back along the flattened
    // axis. An orthographic projection doesn't care how far back it is, so the
    // margin only has to keep the near plane off the geometry.
    const span = Math.max(1, sliceHi - sliceLo);
    const margin = Math.max(2, span * 0.02);
    const depth = span + 2 * margin;
    const camSlice = (basis.sign > 0 ? sliceHi : sliceLo) + basis.sign * margin;
    const centre = new THREE.Vector3();
    for (let i = 0; i < 3; i++) {
        centre.setComponent(
            i,
            centre.getComponent(i) + basis.right[i] * cu + basis.down[i] * cv,
        );
    }
    centre.setComponent(basis.axis, (sliceLo + sliceHi) / 2);

    const cam = new THREE.OrthographicCamera(-W / 2, W / 2, D / 2, -D / 2, 0.1, depth);
    cam.position.copy(centre);
    cam.position.setComponent(basis.axis, camSlice);
    cam.up.set(basis.up[0], basis.up[1], basis.up[2]);
    cam.lookAt(centre);
    cam.updateProjectionMatrix();

    // The slab, as two world clip planes perpendicular to the flattened axis. A
    // three.js plane keeps the half-space where `normal . p + constant >= 0`, so
    // these read as: no nearer to the camera than `cutNear`, no further than
    // `cutFar`, both measured along the axis in the direction the camera stands.
    const n = new THREE.Vector3();
    n.setComponent(basis.axis, -basis.sign);
    const planeNear = new THREE.Plane(n.clone(), basis.sign * cutNear);
    const planeFar = new THREE.Plane(
        n.clone().negate(),
        -basis.sign * cutFar,
    );
    const prevClip = renderer.clippingPlanes;
    // The slice needs its own capture instance (it isn't square like a cube face),
    // but it runs the SAME pipeline — so the slab is lit, glazed and tone-mapped
    // exactly like the panos. Shadows stay ENABLED on purpose: the map is baked and
    // frozen, and toggling shadowMap.enabled would recompile every material and
    // could leave a stale map for the passes that follow.
    const mini = createCapture(
        renderer,
        pw,
        ph,
        0.1,
        Math.max(1, depth),
        50,
        background,
        exposure,
    );
    let data;
    try {
        renderer.clippingPlanes = [planeNear, planeFar];
        mini.renderCamera(scene, cam);
        data = readTargetTopDown(renderer, mini.rtColor, pw, ph, true);
    } finally {
        renderer.clippingPlanes = prevClip;
        mini.dispose();
    }

    const crop = document.createElement("canvas");
    crop.width = pw;
    crop.height = ph;
    crop.getContext("2d").putImageData(new ImageData(data.data, pw, ph), 0, 0);
    return new Promise((resolve, reject) =>
        crop.toBlob(
            (blob) => (blob ? resolve(blob) : reject(new Error("minimap encode failed"))),
            "image/png",
        ),
    );
}

// The storey's published fields — its name and world-space volume — pulled off one
// of the plan's floors. Empty when the plan has none (an older plan) or the model
// skipped it; the viewer then falls back to its nearest-anchor reading.
function floorFields(floor) {
    const out = {};
    if (!floor) return out;
    if (typeof floor.name === "string" && floor.name) out.name = floor.name;
    const v = floor.volume;
    if (v && Array.isArray(v.origin) && Array.isArray(v.dimensions)) {
        out.volume = { origin: v.origin, dimensions: v.dimensions };
    }
    return out;
}

// The plan's storey nearest height `y`. Only needed on the FALLBACK path, where we
// had to cluster the anchors ourselves and so have no direct link from a slice back
// to the floor it came from. Matching on height means a drift degrades to one
// mislabelled slice rather than silently shifting every floor by one.
function floorSpecFor(floors, y) {
    if (!Array.isArray(floors) || floors.length === 0) return {};
    let best = null;
    let bestD = Infinity;
    for (const f of floors) {
        const d = Math.abs((f?.coord ?? f?.y ?? 0) - y);
        if (d < bestD) {
            bestD = d;
            best = f;
        }
    }
    return floorFields(best);
}

// The levels to slice, TAKEN FROM THE PLAN. The server decides where the scene
// divides into storeys (anchors.py `plan_floors`), assigns every capture to one,
// and picks the two planes each slice is cut between — so there is one grouping in
// the system instead of the same rule implemented twice and kept in step by hand.
//
// Returns null when the plan can't drive this: it has no floors, or it was rebuilt
// from an old tour.json and so carries no membership. The caller then clusters, as
// it always did.
function planLevels(floors, positions, opts, basis) {
    if (!Array.isArray(floors) || floors.length === 0) return null;
    const out = [];
    for (const f of floors) {
        if (!f || !Array.isArray(f.anchors)) return null;
        const indices = f.anchors.filter(
            (i) => Number.isInteger(i) && i >= 0 && i < positions.length,
        );
        if (indices.length === 0) return null; // a storey with no captures can't be sliced
        const cs = indices
            .map((i) => positions[i][basis.axis])
            .sort((a, b) => a - b);
        const coord = num(f.coord, num(f.y, cs[(cs.length - 1) >> 1]));
        out.push({
            coord,
            // Where to cut. `cut` is the near plane, the one that opens the scene
            // up; `cut_far` is the back of the slab. Both come from the plan, which
            // is the only place that knows whether this is a floor being opened at
            // head height or a diorama being opened at its front face. The
            // fallbacks are the pre-profile behaviour, for a plan that predates it.
            cutNear: num(f.cut, coord),
            cutFar: num(f.cut_far, cs[0] - opts.sliceBelow),
            indices,
            floor: f,
        });
    }
    // Ordered along the flattened axis in the direction the camera looks, so level
    // 0 is the far side — the bottom floor of a building, the back of a diorama.
    out.sort((a, b) => (a.coord - b.coord) * basis.sign);
    return out;
}

// The scene box's extent along one signed unit axis.
function axisRange(box, vec) {
    const i = vec[0] !== 0 ? 0 : vec[1] !== 0 ? 1 : 2;
    const a = vec[i] * box.min.getComponent(i);
    const b = vec[i] * box.max.getComponent(i);
    return a <= b ? [a, b] : [b, a];
}

// Render + upload one slice per storey and return the manifest `minimaps` array
// (empty on any failure — the tour stays valid). Also stamps each pano with the
// level it stands on, so the viewer reads the grouping rather than re-deriving it.
async function buildMinimaps(
    renderer,
    scene,
    panos,
    background,
    exposure,
    floors,
    opts,
    basis,
    onLevel,
) {
    if (panos.length === 0) return [];
    const positions = panos.map((p) => p.position);
    const box = new THREE.Box3().setFromObject(scene);
    if (box.isEmpty()) return [];
    // The footprint in the MAP's own frame: u across the page, v down it. Stored
    // that way so the viewer can place a capture on the image without knowing
    // which world axes those turned out to be.
    const [u0, u1] = axisRange(box, basis.right);
    const [v0, v1] = axisRange(box, basis.down);
    const pad = opts.padFrac * Math.max(u1 - u0, v1 - v0, 1);
    const bounds = {
        minU: u0 - pad,
        maxU: u1 + pad,
        minV: v0 - pad,
        maxV: v1 + pad,
    };
    const sliceLo = box.min.getComponent(basis.axis);
    const sliceHi = box.max.getComponent(basis.axis);

    const levels =
        planLevels(floors, positions, opts, basis) ??
        groupAnchorLevels(positions, opts.levelEps, basis.axis).map((g) => ({
            coord: g.y,
            cutNear: g.y,
            cutFar: g.minY - opts.sliceBelow,
            indices: g.indices,
            floor: null,
        }));
    const minimaps = [];
    for (let li = 0; li < levels.length; li++) {
        onLevel?.(li, levels.length);
        const lv = levels[li];
        const blob = await captureMinimapBlob(
            renderer,
            scene,
            bounds,
            basis,
            lv.cutNear,
            lv.cutFar,
            sliceLo,
            sliceHi,
            background,
            exposure,
            opts.res,
        );
        await uploadMinimap(`minimap-${li}`, blob);
        for (const i of lv.indices) panos[i].level = li;
        minimaps.push({
            level: li,
            // Where this storey sits along the flattened axis. `y` is kept as an
            // alias so a viewer written before the map could look any way but down
            // still reads it.
            coord: lv.coord,
            y: lv.coord,
            file: `minimap-${li}.png`,
            bounds,
            // The frame this image was drawn in, so the viewer places captures and
            // labels on it with the same mapping that produced it.
            basis: { view_from: basis.viewFrom, image_down: basis.imageDown },
            // The planes it was actually cut between, so a plan rebuilt from this
            // tour re-cuts where this one did (routes.py `_plan_from_tour`).
            cut: lv.cutNear,
            cut_far: lv.cutFar,
            // The floor's name + world-space volume, so the walkthrough can title a
            // storey rather than calling it "floor 2", and can decide which floor an
            // arbitrary point belongs to (see the planner in anchors.py).
            ...(lv.floor ? floorFields(lv.floor) : floorSpecFor(floors, lv.coord)),
        });
    }
    return minimaps;
}

// --- projection proxy bake ---------------------------------------------------

// Bake one placed mesh's geometry into world-space, float32, position-only
// geometry. Reading every vertex through `fromBufferAttribute` DENORMALIZES
// quantized attributes (placed GLBs are Meshopt/KHR_mesh_quantization), so
// writing world-space floats into a FRESH Float32 array avoids truncating onto
// the source's integer grid.
function bakeWorldGeometry(mesh) {
    const src = mesh.geometry.getAttribute("position");
    if (!src) return null;
    const count = src.count;
    const positions = new Float32Array(count * 3);
    const v = new THREE.Vector3();
    const m = mesh.matrixWorld;
    for (let i = 0; i < count; i++) {
        v.fromBufferAttribute(src, i).applyMatrix4(m);
        positions[i * 3] = v.x;
        positions[i * 3 + 1] = v.y;
        positions[i * 3 + 2] = v.z;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const idx = mesh.geometry.getIndex();
    if (idx) g.setIndex(new THREE.BufferAttribute(idx.array.slice(), 1));
    return g;
}

// The owning object's id for a mesh: walk up to the node loadScene tagged with
// `objectId`. Carries object identity into the proxy so the walkthrough can
// highlight individual objects after decimation.
function pickIdOf(obj) {
    for (let cur = obj; cur; cur = cur.parent) {
        if (cur.userData?.objectId) return cur.userData.objectId;
    }
    return null;
}

// Bake the live scene into one material-free, world-space GLB (geometry only),
// each source object's meshes under their own node named with the object id, so
// the proxy keeps per-object identity through the server's decimation. Returns
// the binary GLB ArrayBuffer, or null when the scene has no meshes.
async function buildMergedSceneGlbBuffer(scene) {
    scene.updateMatrixWorld(true);
    // Gather the meshes first (cheap), then bake world geometry in a loop that
    // YIELDS periodically — a several-hundred-object scene would otherwise block
    // the main thread long enough to starve the keepalive heartbeat. Group the
    // baked position (+index) arrays by object id so encodeGlb keeps per-object
    // nodes (the server's decimation preserves those for object highlighting).
    const meshes = [];
    scene.traverse((o) => {
        if (o.isMesh && o.geometry) meshes.push(o);
    });
    const groups = new Map(); // id -> [{ pos: Float32Array, idx: Uint32Array|null }]
    for (let i = 0; i < meshes.length; i++) {
        const o = meshes[i];
        const g = bakeWorldGeometry(o);
        if (g) {
            const pos = g.getAttribute("position").array;
            const index = g.getIndex();
            const idx = index
                ? index.array instanceof Uint32Array
                    ? index.array
                    : new Uint32Array(index.array)
                : null;
            const id = pickIdOf(o) ?? "";
            const list = groups.get(id) ?? groups.set(id, []).get(id);
            list.push({ pos, idx });
        }
        if ((i & 63) === 63) await sleep(0); // let the keepalive flush
    }
    if (groups.size === 0) return null;
    await beat(); // fresh stall window before the (fast) serialize
    return encodeGlb(groups);
}

// A minimal position(+index)-only binary GLB writer. GLTFExporter is far too slow
// for a several-hundred-object scene (its dedup / material / per-accessor
// validation passes dominate — minutes), and the projection proxy is pure
// geometry the server decimates anyway. This packs the baked world-space positions
// straight into one binary chunk: one node per object id (name kept so the server
// keeps per-object identity through decimation), one primitive per source mesh.
function encodeGlb(groups) {
    const GL_FLOAT = 5126;
    const GL_UINT = 5125;
    const ARRAY_BUFFER = 34962;
    const ELEMENT_ARRAY_BUFFER = 34963;
    const align4 = (n) => (n + 3) & ~3;

    const accessors = [];
    const bufferViews = [];
    const meshes = [];
    const nodes = [];
    const chunks = []; // typed arrays, in bufferView order
    let byteOffset = 0;

    const pushView = (arr, target) => {
        bufferViews.push({ buffer: 0, byteOffset, byteLength: arr.byteLength, target });
        chunks.push(arr);
        byteOffset = align4(byteOffset + arr.byteLength);
        return bufferViews.length - 1;
    };

    for (const [id, prims] of groups) {
        const primitives = [];
        for (const { pos, idx } of prims) {
            let mnx = Infinity;
            let mny = Infinity;
            let mnz = Infinity;
            let mxx = -Infinity;
            let mxy = -Infinity;
            let mxz = -Infinity;
            for (let i = 0; i < pos.length; i += 3) {
                const x = pos[i];
                const y = pos[i + 1];
                const z = pos[i + 2];
                if (x < mnx) mnx = x;
                if (y < mny) mny = y;
                if (z < mnz) mnz = z;
                if (x > mxx) mxx = x;
                if (y > mxy) mxy = y;
                if (z > mxz) mxz = z;
            }
            const posAcc = accessors.length;
            accessors.push({
                bufferView: pushView(pos, ARRAY_BUFFER),
                componentType: GL_FLOAT,
                count: pos.length / 3,
                type: "VEC3",
                min: [mnx, mny, mnz],
                max: [mxx, mxy, mxz],
            });
            const prim = { attributes: { POSITION: posAcc }, mode: 4 };
            if (idx) {
                prim.indices = accessors.length;
                accessors.push({
                    bufferView: pushView(idx, ELEMENT_ARRAY_BUFFER),
                    componentType: GL_UINT,
                    count: idx.length,
                    type: "SCALAR",
                });
            }
            primitives.push(prim);
        }
        nodes.push({ name: id || undefined, mesh: meshes.length });
        meshes.push({ primitives });
    }

    const gltf = {
        asset: { version: "2.0", generator: "tourcapture" },
        scene: 0,
        scenes: [{ nodes: nodes.map((_, i) => i) }],
        nodes,
        meshes,
        accessors,
        bufferViews,
        buffers: [{ byteLength: byteOffset }],
    };

    let jsonBytes = new TextEncoder().encode(JSON.stringify(gltf));
    const jsonPadded = align4(jsonBytes.length);
    const HEADER = 12;
    const CHUNK_HEADER = 8;
    const binStart = HEADER + CHUNK_HEADER + jsonPadded + CHUNK_HEADER;
    const total = binStart + byteOffset;
    const out = new ArrayBuffer(total);
    const dv = new DataView(out);
    const u8 = new Uint8Array(out);
    let p = 0;
    dv.setUint32(p, 0x46546c67, true); // "glTF"
    dv.setUint32((p += 4), 2, true);
    dv.setUint32((p += 4), total, true);
    dv.setUint32((p += 4), jsonPadded, true);
    dv.setUint32((p += 4), 0x4e4f534a, true); // "JSON"
    p += 4;
    u8.set(jsonBytes, p);
    u8.fill(0x20, p + jsonBytes.length, p + jsonPadded); // pad JSON with spaces
    p += jsonPadded;
    dv.setUint32(p, byteOffset, true);
    dv.setUint32((p += 4), 0x004e4942, true); // "BIN\0"
    p += 4;
    for (let k = 0; k < chunks.length; k++) {
        const arr = chunks[k];
        u8.set(
            new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength),
            binStart + bufferViews[k].byteOffset,
        );
    }
    return out;
}

// --- uploads -----------------------------------------------------------------

async function uploadPano(panoId, blob) {
    const res = await fetch(cellPath(`/tour/pano/${encodeURIComponent(panoId)}`), {
        method: "PUT",
        headers: { "Content-Type": "image/jpeg" },
        body: blob,
    });
    if (!res.ok) throw new Error(`upload ${panoId} → ${res.status}`);
}

async function uploadMinimap(minimapId, blob) {
    const res = await fetch(cellPath(`/tour/minimap/${encodeURIComponent(minimapId)}`), {
        method: "PUT",
        headers: { "Content-Type": "image/png" },
        body: blob,
    });
    if (!res.ok) throw new Error(`upload ${minimapId} → ${res.status}`);
}

async function uploadProxy(glbBuffer) {
    const res = await fetch(cellPath("/tour/proxy"), {
        method: "POST",
        headers: { "Content-Type": "model/gltf-binary" },
        body: glbBuffer,
    });
    return res.ok;
}

// --- finish / fail -----------------------------------------------------------

async function postFinish(payload) {
    const res = await fetch(stagePath("/finish"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`finish POST failed: HTTP ${res.status}`);
    return res.json();
}

let failed = false;
async function fail(err) {
    if (failed) return;
    failed = true;
    const message = String(err && err.message ? err.message : err);
    // Capture is headless, so this /finish error is the ONLY window into where it
    // broke — attach the phase we died in (the last status line) + the top stack
    // frames, both of which the server surfaces through the job error.
    const phase = lines.length
        ? lines[lines.length - 1].replace(/<[^>]*>/g, "").slice(0, 100)
        : "";
    const frames =
        err && err.stack
            ? String(err.stack)
                  .split("\n")
                  .map((s) => s.trim())
                  .filter(Boolean)
                  .slice(1, 3)
                  .join(" ← ")
            : "";
    const detail = [message, phase && `after: ${phase}`, frames].filter(Boolean).join(" · ");
    status(message, "err");
    if (TOKEN) {
        try {
            await postFinish({ error: detail });
        } catch {
            /* server gone — nothing else to report to */
        }
    }
}

// --- capture job -------------------------------------------------------------

async function runCapture() {
    const mres = await fetch(stagePath("/manifest"), { cache: "no-store" });
    if (!mres.ok) throw new Error(`manifest HTTP ${mres.status} (stale job token?)`);
    const manifest = await mres.json();
    const anchors = Array.isArray(manifest.anchors) ? manifest.anchors : [];
    if (anchors.length === 0) throw new Error("manifest carries no anchors");
    const faceSize = manifest.pano?.face_size || DEFAULT_FACE_SIZE;
    const panoWidth = manifest.pano?.width || DEFAULT_PANO_WIDTH;
    const quality = manifest.pano?.quality || DEFAULT_JPEG_QUALITY;
    const minimapOpts = readMinimapOpts(manifest);
    const mapBasis = readBasis(manifest.profile);
    status(`job: ${RUN}/${SLOT}/${MODEL} — ${anchors.length} anchors @ ${faceSize}² faces`);

    const renderer = createCaptureRenderer({ onContextLost: fail });
    const glName = rendererName(renderer);
    status(`WebGL: ${glName}`);
    if (/swiftshader|software|llvmpipe/i.test(glName) && params.get("force") !== "1") {
        throw new Error(
            `software WebGL (${glName}) — headless browser has no GPU; open this URL ` +
                "in a normal browser, or relaunch with a GPU-capable headless Chrome/Edge",
        );
    }

    const { scene, loaded, failed: badObjects } = await loadScene(
        renderer,
        new URL(manifest.bundle_url, API),
    );
    status(`scene: ${loaded} objects loaded${badObjects ? `, ${badObjects} failed` : ""}`);
    if (loaded === 0) throw new Error("no objects loaded from the mesh bundle");
    await beat();

    // The exact Stage-5 bake sequence: emissive fixtures (glow + shadow-casting
    // point lights) → sun/hemi/IBL rig fitted to the scene → ONE shadow bake over
    // both layers → per-object scene reflection probes.
    const lighting = normalizeLighting(manifest.lighting || {});
    const background = manifest.background || [0, 0, 0];
    const { emissive, refl, sceneBox } = setupCaptureLighting(renderer, scene, {
        lighting,
        background,
    });
    if (emissive.count > 0) status(`emissive: ${emissive.count} light source(s)`);
    status(
        `lighting: env ${lighting.env} · sun ${lighting.key} @ ${lighting.azimuth}°/` +
            `${lighting.elevation}° · exposure ${lighting.exposure}`,
    );
    if (refl.probes > 0) status(`reflections: ${refl.probes} scene probe(s) baked`);

    // Camera clip range from the scene bounds: near tight, far past the far wall.
    const diag = sceneBox.isEmpty() ? 50 : sceneBox.getSize(new THREE.Vector3()).length();
    const near = 0.05;
    const far = Math.max(50, diag * 3);
    // One capture for every cube face: square, 90° fov — the same pipeline object
    // Stage 5 drives, just pointed six ways per anchor instead of along a plan.
    const capture = createCapture(
        renderer,
        faceSize,
        faceSize,
        near,
        far,
        90,
        background,
        lighting.exposure ?? 1.0,
    );
    await beat();

    // --- panos, one per anchor ---
    const panoMeta = [];
    const t0 = performance.now();
    for (let i = 0; i < anchors.length; i++) {
        const a = anchors[i];
        const pos = Array.isArray(a.position) ? a.position : [0, 0, 0];
        const id = typeof a.id === "string" && a.id ? a.id : `anchor-${String(i).padStart(3, "0")}`;
        progress(`capturing pano ${i + 1}/${anchors.length} (${id})…`);
        const faces = renderPanoFaces(renderer, capture, scene, pos, faceSize);
        const blob = await stitchPanoBlob(faces, faceSize, panoWidth, quality);
        await uploadPano(id, blob);
        panoMeta.push({
            id,
            file: `${id}.jpg`,
            position: pos,
            forward: [0, 0, -1], // each capture is a full 360; forward only seeds /pano's view
            name: typeof a.name === "string" ? a.name : undefined,
            zone: typeof a.zone === "string" ? a.zone : undefined,
        });
    }
    const rate = (anchors.length / Math.max(0.001, (performance.now() - t0) / 1000)).toFixed(2);
    status(`panos: ${panoMeta.length} captured (${rate} panos/s)`);

    // --- bird's-eye minimaps ---
    let minimaps = [];
    try {
        minimaps = await buildMinimaps(
            renderer,
            scene,
            panoMeta, // stamped with each pano's level on the way through
            background,
            lighting.exposure ?? 1.0,
            Array.isArray(manifest.floors) ? manifest.floors : [],
            minimapOpts,
            mapBasis,
            (li, n) => progress(`rendering minimap ${li + 1}/${n}…`),
        );
        // Report the cut heights, not just the count. Capture is headless, so this
        // log is the only view into it — and a cut that landed above a ceiling
        // produces a blank slice that looks like a render failure until you can see
        // the number it was taken at.
        status(
            `minimaps: ${minimaps.length} level(s) @ ${minimapOpts.res}px, ` +
                `viewed from ${mapBasis.viewFrom} (down ${mapBasis.imageDown}), cut at ` +
                minimaps.map((m) => `${m.cut.toFixed(2)}`).join(" / ") +
                "m",
        );
    } catch (e) {
        minimaps = [];
        status(`minimaps failed (tour saved without them): ${e.message}`, "err");
    }
    await beat();

    // --- projection proxy ---
    let hasProxy = false;
    try {
        progress("building projection proxy…");
        const merged = await buildMergedSceneGlbBuffer(scene);
        if (merged) hasProxy = await uploadProxy(merged);
        status(hasProxy ? "proxy: built + uploaded" : "proxy: skipped (no meshes)");
    } catch (e) {
        status(`proxy failed (tour saved without it): ${e.message}`, "err");
    }
    await beat();

    // --- write tour.json + publish ---
    const tourManifest = {
        version: 1,
        proxy: hasProxy ? "proxy.glb" : null,
        planner_model: manifest.planner_model ?? null,
        namer_model: manifest.namer_model ?? null,
        planner_reasoning: manifest.planner_reasoning ?? null,
        panos: panoMeta,
        minimaps,
        connectors: Array.isArray(manifest.connectors) ? manifest.connectors : [],
        // Ids of the scene's DISCRETE objects (node_kind "object"), as opposed to
        // the zone shells / ground geometry the encapsulating pass produced. The
        // walkthrough only offers to inspect something on this list.
        objects: Array.isArray(manifest.objects) ? manifest.objects : [],
        // Zone names for the bird's-eye map, already pruned so none nests inside
        // another (see anchors.py label_map_zones).
        map_labels: Array.isArray(manifest.map_labels) ? manifest.map_labels : [],
        // What kind of place this is: the map's frame, the inhabitant's size, the
        // word its storeys go by. The viewer reads all three.
        profile: manifest.profile ?? null,
    };
    const fin = await postFinish({ manifest: tourManifest, renderer: glName });
    if (fin.ok) {
        status(`tour capture complete — ${panoMeta.length} panos, tour.json written`, "ok");
    } else {
        status("finish reported an error", "err");
    }
}

// --- entry -------------------------------------------------------------------

(async () => {
    try {
        if (!API || !RUN || !SLOT || !MODEL || !TOKEN) {
            throw new Error(
                "missing query params (?api=&run=&slot=&model=&token=) — this page is " +
                    "launched by the server's tour-capture job",
            );
        }
        startKeepalive();
        await runCapture();
    } catch (e) {
        console.error(e);
        await fail(e);
    } finally {
        stopKeepalive();
    }
})();
