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
const MINIMAP_LEVEL_EPS = 1.5; // metres; anchors within this Y gap share a level
const MINIMAP_RES = 1024; // longest output side in device px
const MINIMAP_PAD_FRAC = 0.04; // breathing room around the scene footprint
const MINIMAP_SLICE_BELOW = 2; // metres below the level's lowest anchor for the floor cut

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

// Cluster anchor Ys into levels by gap; returns [{ y, minY, indices }] low→high.
function groupAnchorLevels(positions) {
    const order = positions
        .map((_, i) => i)
        .sort((a, b) => positions[a][1] - positions[b][1]);
    const groups = [];
    let cur = null;
    for (const i of order) {
        const y = positions[i][1];
        if (!cur || y - cur.lastY > MINIMAP_LEVEL_EPS) {
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

// Render one top-down orthographic slice (a horizontal slab at camera level, roof
// cut open) into a PNG blob. The ortho camera looks straight down with -Z "up" in
// the image, so the stored `bounds` map world (x,z) → image (left,top).
async function captureMinimapBlob(
    renderer,
    scene,
    bounds,
    cutTop,
    cutBottom,
    yTop,
    yBot,
    background,
    exposure,
) {
    const W = bounds.maxX - bounds.minX;
    const D = bounds.maxZ - bounds.minZ;
    const cx = (bounds.minX + bounds.maxX) / 2;
    const cz = (bounds.minZ + bounds.maxZ) / 2;

    let pw;
    let ph;
    if (W >= D) {
        pw = MINIMAP_RES;
        ph = Math.max(1, Math.round((MINIMAP_RES * D) / W));
    } else {
        ph = MINIMAP_RES;
        pw = Math.max(1, Math.round((MINIMAP_RES * W) / D));
    }

    const cam = new THREE.OrthographicCamera(-W / 2, W / 2, D / 2, -D / 2, 0.1, yTop - yBot + 4);
    cam.position.set(cx, yTop + 2, cz);
    cam.up.set(0, 0, -1);
    cam.lookAt(cx, yBot, cz);
    cam.updateProjectionMatrix();

    // World clip planes bounding a horizontal SLAB: keep cutBottom <= y <= cutTop.
    const planeTop = new THREE.Plane(new THREE.Vector3(0, -1, 0), cutTop);
    const planeBottom = new THREE.Plane(new THREE.Vector3(0, 1, 0), -cutBottom);
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
        Math.max(1, yTop - yBot + 4),
        50,
        background,
        exposure,
    );
    let data;
    try {
        renderer.clippingPlanes = [planeTop, planeBottom];
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

// Group anchors by level, render + upload one slice per level, and return the
// manifest `minimaps` array (empty on any failure — the tour stays valid).
async function buildMinimaps(renderer, scene, positions, background, exposure, onLevel) {
    if (positions.length === 0) return [];
    const box = new THREE.Box3().setFromObject(scene);
    if (box.isEmpty()) return [];
    const pad = MINIMAP_PAD_FRAC * Math.max(box.max.x - box.min.x, box.max.z - box.min.z, 1);
    const bounds = {
        minX: box.min.x - pad,
        maxX: box.max.x + pad,
        minZ: box.min.z - pad,
        maxZ: box.max.z + pad,
    };
    const levels = groupAnchorLevels(positions);
    const minimaps = [];
    for (let li = 0; li < levels.length; li++) {
        onLevel?.(li, levels.length);
        const blob = await captureMinimapBlob(
            renderer,
            scene,
            bounds,
            levels[li].y,
            levels[li].minY - MINIMAP_SLICE_BELOW,
            box.max.y,
            box.min.y,
            background,
            exposure,
        );
        await uploadMinimap(`minimap-${li}`, blob);
        minimaps.push({ level: li, y: levels[li].y, file: `minimap-${li}.png`, bounds });
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
            panoMeta.map((p) => p.position),
            background,
            lighting.exposure ?? 1.0,
            (li, n) => progress(`rendering minimap ${li + 1}/${n}…`),
        );
        status(`minimaps: ${minimaps.length} level(s)`);
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
