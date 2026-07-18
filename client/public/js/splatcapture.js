// Stage-5 reference-capture worker — the splat pipeline's renderer.
//
// Runs the SAME WebGL stack as the debug viewer (three.js GLTFLoader +
// KTX2Loader + MeshoptDecoder) against the cell's SPLAT asset tier, so KTX2/
// ETC1S textures stay GPU-compressed and Meshopt geometry stays quantized —
// the whole scene fits comfortably in VRAM and renders at hardware speed.
//
// Protocol (server = server/app/api/routes.py + services/refcapture.py):
//   1. GET  {api}/runs/{run}/splat/stage5/{slot}/{model}/manifest?token=…
//        → { resolution, near, far, fov_deg, background, bundle_url,
//            cameras_url, pending: [view ids], total }
//   2. fetch cameras_url (the Stage-4 plan) + bundle_url (SMB1 mesh bundle)
//   3. for each pending view: render → async-read RGBA8 + packed log-u16
//      depth → POST binary SRF1 batches to …/frames?token=…
//   4. POST …/finish?token=… → the server verifies + writes transforms.json.
//
// Per view, two passes:
//   * scene pass  → rtColor (RGBA8 + float depth texture): unlit albedo,
//     alpha-blended in stored-sRGB space (background under the remainder),
//     alpha channel = accumulated coverage. Opaque + alphaMode=MASK write
//     depth; alphaMode=BLEND glass does not (its depth is the surface behind).
//   * pack pass   → rtDepth (RGBA8): fullscreen triangle converting the depth
//     texture to planar view-space Z metres, log-quantized to the EXACT uint16
//     codes splat/stage5.py's encode_depth_u16 defines (RG = hi/lo byte).
//
// Readback is a PBO + fence ring, so the GPU renders view N+1 while view N's
// bytes drain — no rAF, no sync stalls. The loop yields via MessageChannel
// (immune to background-timer throttling in headless Chrome).
//
// ?selftest=1 renders a synthetic UV-encoding quad through the identical path
// and asserts the pose/flip/planar-depth/alpha conventions in-page (the WebGL
// twin of the old nvdiffrast GPU smoke). No job endpoints needed.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";

const DEPTH_CODE_MAX = 65535; // matches splat/stage5.py _DEPTH_CODE_MAX
const DECODE_WORKERS = Math.min(4, navigator.hardwareConcurrency || 2);
const MAX_PARSE_INFLIGHT = 4; // concurrent GLB parses while streaming the bundle
const READ_SEGMENTS = 3; // readback segments in flight (ring depth)
const SEGMENT_VIEWS = 8; // views per segment → ONE getBufferSubData per 8 views
const POST_BATCH_BYTES = 24 * 1024 * 1024; // ~4 views/batch at 1024² (worker-side)
const MESH_BUNDLE_MAGIC = "SMB1";
// Frame batches go over the wire as SRF1 — built + POSTed by the post worker
// (splatcapture-worker.js), which owns everything after the GPU readback.

const params = new URLSearchParams(location.search);
const API = (params.get("api") || "").replace(/\/$/, "");
const RUN = params.get("run");
const SLOT = params.get("slot");
const MODEL = params.get("model");
const TOKEN = params.get("token") || "";
const SELFTEST = params.get("selftest") === "1";

const statusEl = document.getElementById("status");
const lines = [];
function status(text, cls = "") {
    lines.push(cls ? `<span class="${cls}">${text}</span>` : text);
    if (lines.length > 30) lines.shift();
    statusEl.innerHTML = lines.join("\n");
    console.log(`[capture] ${text}`);
}
// The live progress line rewrites in place instead of appending.
function progress(text) {
    statusEl.innerHTML = lines.join("\n") + "\n" + text;
}

// MessageChannel yield — keeps the render loop scheduling even where headless/
// background pages throttle setTimeout.
const _mc = new MessageChannel();
let _mcWake = null;
_mc.port1.onmessage = () => {
    const w = _mcWake;
    _mcWake = null;
    if (w) w();
};
const yieldLoop = () =>
    new Promise((r) => {
        _mcWake = r;
        _mc.port2.postMessage(0);
    });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Per-view averages of the loop's wall-time buckets — the live/final answer to
// "where does each view's time actually go".
function perViewMs(buckets, n) {
    const out = {};
    for (const [k, v] of Object.entries(buckets)) out[k] = Number((v / n).toFixed(2));
    return out;
}
function bucketLine(buckets, n) {
    return (
        Object.entries(buckets)
            .map(([k, v]) => `${k} ${(v / n).toFixed(1)}`)
            .join(" / ") + " ms/view"
    );
}

function stagePath(tail) {
    return (
        `${API}/runs/${encodeURIComponent(RUN)}/splat/stage5/` +
        `${encodeURIComponent(SLOT)}/${encodeURIComponent(MODEL)}${tail}?token=${encodeURIComponent(TOKEN)}`
    );
}

// --- renderer + capture pipeline ---------------------------------------------

function createRenderer() {
    const renderer = new THREE.WebGLRenderer({
        antialias: false,
        alpha: false,
        preserveDrawingBuffer: false,
        powerPreference: "high-performance",
    });
    if (!renderer.capabilities.isWebGL2) throw new Error("WebGL2 unavailable");
    // Raw byte passthrough: no tone mapping anywhere; render targets are raw by
    // construction (three only encodes to the canvas), and every base-color map
    // is forced to NoColorSpace below, so texels blend in stored-sRGB space —
    // exactly the statistic gsplat trains against.
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    renderer.setSize(256, 256); // preview canvas only; targets carry the real res
    document.body.appendChild(renderer.domElement);
    renderer.domElement.addEventListener("webglcontextlost", (e) => {
        e.preventDefault();
        fail(new Error("WebGL context lost"));
    });
    return renderer;
}

function rendererName(renderer) {
    const gl = renderer.getContext();
    const info = gl.getExtension("WEBGL_debug_renderer_info");
    return info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : "unknown";
}

// One material class for the whole capture: unlit base color. Copies exactly
// the glTF inputs the refs contract reads (map, baseColorFactor, opacity,
// alphaMode via transparent/alphaTest) and drops everything else. DoubleSide
// because Trellis winding is unreliable (matches the pipeline's assumption);
// BLEND glass keeps depthWrite OFF so depth is the surface behind the pane.
function toUnlit(orig) {
    const m = new THREE.MeshBasicMaterial();
    m.map = orig.map || null;
    if (m.map) m.map.colorSpace = THREE.NoColorSpace; // raw texels, no shader decode
    if (orig.color) m.color.copy(orig.color); // baseColorFactor
    m.opacity = orig.opacity ?? 1;
    m.transparent = orig.transparent === true;
    m.alphaTest = orig.alphaTest || 0;
    m.side = THREE.DoubleSide;
    m.depthWrite = !m.transparent;
    m.toneMapped = false;
    m.vertexColors = orig.vertexColors === true;
    return m;
}

function prepareUnlit(root) {
    root.traverse((o) => {
        if (!o.isMesh || !o.material) return;
        const orig = o.material;
        o.material = Array.isArray(orig) ? orig.map(toUnlit) : toUnlit(orig);
        for (const m of Array.isArray(orig) ? orig : [orig]) m.dispose();
    });
}

// Capture pipeline: scene target (color + float depth texture) and the
// depth-pack target + fullscreen pass converting window-space depth to the
// contract's log-uint16 codes (RG = hi/lo byte; 0 = background).
function createCapture(renderer, resolution, near, far, fovDeg, background) {
    const depthTexture = new THREE.DepthTexture(resolution, resolution);
    depthTexture.type = THREE.FloatType;
    const targetOpts = {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        generateMipmaps: false,
        stencilBuffer: false,
    };
    const rtColor = new THREE.WebGLRenderTarget(resolution, resolution, {
        ...targetOpts,
        depthBuffer: true,
        depthTexture,
    });
    // Depth pack target is RG8 (two bytes/pixel), NOT RGBA8: the pack shader writes
    // the log-u16 code's low byte to R and high byte to G, so the readback bytes
    // ARE the little-endian uint16 code — the CPU swizzle (a 1M-element/view JS
    // loop, the pipeline's per-view wall at 1024²) is gone, and depth readback is
    // halved (2 B/px vs 4). RG8 is a plain color-renderable format (no integer-RT
    // fragility); the ?selftest validates the codes on this GPU/driver.
    const rtDepth = new THREE.WebGLRenderTarget(resolution, resolution, {
        ...targetOpts,
        depthBuffer: false,
        format: THREE.RGFormat,
        type: THREE.UnsignedByteType,
    });

    const packMaterial = new THREE.ShaderMaterial({
        uniforms: {
            tDepth: { value: depthTexture },
            uNear: { value: near },
            uFar: { value: far },
        },
        vertexShader: /* glsl */ `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = vec4(position.xy, 0.0, 1.0);
            }
        `,
        fragmentShader: /* glsl */ `
            precision highp float;
            uniform sampler2D tDepth;
            uniform float uNear;
            uniform float uFar;
            varying vec2 vUv;
            void main() {
                float d = texture2D(tDepth, vUv).r;             // window-space [0,1]
                if (d >= 1.0) {                                  // nothing wrote depth
                    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);     // code 0 = background
                    return;
                }
                float zndc = d * 2.0 - 1.0;
                float z = 2.0 * uNear * uFar / (uFar + uNear - zndc * (uFar - uNear));
                // splat/stage5.py encode_depth_u16: log-spaced code 1..65535.
                float t = clamp(log(z / uNear) / log(uFar / uNear), 0.0, 1.0);
                float code = floor(t * ${(DEPTH_CODE_MAX - 1).toFixed(1)} + 0.5) + 1.0;
                float hi = floor(code / 256.0);
                float lo = code - hi * 256.0;
                // LOW byte -> R, HIGH byte -> G: on an RG8 target the readback bytes
                // are [lo, hi] = the code as a little-endian uint16, so no swizzle.
                gl_FragColor = vec4(lo / 255.0, hi / 255.0, 0.0, 1.0);
            }
        `,
        depthTest: false,
        depthWrite: false,
    });
    // Fullscreen triangle (no index, no camera transform — clip-space verts).
    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
        "position",
        new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3),
    );
    geo.setAttribute(
        "uv",
        new THREE.BufferAttribute(new Float32Array([0, 0, 2, 0, 0, 2]), 2),
    );
    const packScene = new THREE.Scene();
    const tri = new THREE.Mesh(geo, packMaterial);
    tri.frustumCulled = false;
    packScene.add(tri);
    const packCamera = new THREE.Camera();

    const camera = new THREE.PerspectiveCamera(fovDeg, 1, near, far);
    const bg = new THREE.Color(background[0], background[1], background[2]);

    const lookTarget = new THREE.Vector3();
    function renderView(scene, pos, face) {
        camera.position.set(pos[0], pos[1], pos[2]);
        camera.up.set(face.up[0], face.up[1], face.up[2]);
        lookTarget
            .set(face.forward[0], face.forward[1], face.forward[2])
            .add(camera.position);
        camera.lookAt(lookTarget);
        renderer.setClearColor(bg, 0); // alpha 0 = empty coverage
        renderer.setRenderTarget(rtColor);
        renderer.render(scene, camera);
        renderer.setRenderTarget(rtDepth);
        renderer.render(packScene, packCamera);
    }

    return { rtColor, rtDepth, renderView, camera };
}

// PBO + fence async readback, SEGMENTED: each view's color + packed-depth RGBA
// land at offsets inside one large per-segment PBO, and a whole segment (up to
// SEGMENT_VIEWS views) drains with ONE getBufferSubData. Profiling showed
// getBufferSubData costs ~3 ms of GPU-process round-trip latency PER CALL
// (regardless of size) — two calls per view was the pipeline's wall at ~100
// views/s. Amortizing the round trip over a segment makes readback ~0.4 ms/view.
function createReadbackRing(renderer, resolution) {
    const gl = renderer.getContext();
    const n = resolution * resolution;
    const colorBytes = n * 4; // RGBA8
    const depthBytes = n * 2; // RG8 (= u16 code, little-endian)
    const viewBytes = colorBytes + depthBytes;
    const segments = [];
    for (let i = 0; i < READ_SEGMENTS; i++) {
        const pbo = gl.createBuffer();
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
        gl.bufferData(gl.PIXEL_PACK_BUFFER, SEGMENT_VIEWS * viewBytes, gl.STREAM_READ);
        segments.push({ pbo, views: [], sync: null });
    }
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    let open = 0; // segment currently accumulating views

    function seal() {
        const seg = segments[open];
        if (seg.views.length === 0 || seg.sync) return;
        seg.sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
        gl.flush();
        open = (open + 1) % segments.length;
    }

    function canEnqueue() {
        return segments[open].sync === null; // else every segment is in flight
    }

    function enqueue(view, rtColor, rtDepth) {
        const seg = segments[open];
        const base = seg.views.length * viewBytes;
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, seg.pbo);
        renderer.setRenderTarget(rtColor);
        gl.readPixels(0, 0, resolution, resolution, gl.RGBA, gl.UNSIGNED_BYTE, base);
        renderer.setRenderTarget(rtDepth);
        gl.readPixels(0, 0, resolution, resolution, gl.RG, gl.UNSIGNED_BYTE, base + colorBytes);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        seg.views.push(view);
        if (seg.views.length === SEGMENT_VIEWS) seal();
    }

    // Oldest in-flight segment whose fence has signaled (segments seal + drain
    // in ring order, so scanning from `open` finds the oldest first).
    function readySealed() {
        for (let i = 0; i < segments.length; i++) {
            const seg = segments[(open + i) % segments.length];
            if (!seg.sync) continue;
            const s = gl.clientWaitSync(seg.sync, 0, 0);
            if (s === gl.ALREADY_SIGNALED || s === gl.CONDITION_SATISFIED) return seg;
            return null; // oldest not ready — later ones can't be either
        }
        return null;
    }

    // → { buffer: ArrayBuffer, ids: [view ids] } — ONE mapped copy for the whole
    // segment; the post worker slices per-view planes out of it (zero-copy
    // transfer). The segment is immediately reusable.
    function collect(seg) {
        gl.deleteSync(seg.sync);
        seg.sync = null;
        const used = seg.views.length * viewBytes;
        const buffer = new Uint8Array(used);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, seg.pbo);
        gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, buffer, 0, used);
        gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
        const ids = seg.views.map((v) => v.id);
        seg.views = [];
        return { buffer: buffer.buffer, ids };
    }

    return {
        enqueue,
        canEnqueue,
        seal, // flush a partial segment (tail of the queue / drain points)
        readySealed,
        collect,
        inFlight: () => segments.some((s) => s.sync !== null) || segments[open].views.length > 0,
    };
}

// --- scene loading (SMB1 bundle → GLTFLoader.parse) ----------------------------

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
                prepareUnlit(gltf.scene);
                scene.add(gltf.scene);
                loaded += 1;
            } catch (e) {
                failed += 1;
                console.warn(`[capture] ${id}: parse failed`, e);
            }
            progress(`loading scene… ${loaded + failed} objects`);
        })().finally(() => inflight.delete(p));
        inflight.add(p);
        if (inflight.size >= MAX_PARSE_INFLIGHT) await Promise.race(inflight);
    }
    await Promise.allSettled(inflight);
    return { scene, loaded, failed };
}

// --- frame post-processing (the worker owns swizzle + SRF1 pack + POST) ----------

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
    status(String(err && err.message ? err.message : err), "err");
    if (TOKEN) {
        try {
            await postFinish({ error: String(err && err.message ? err.message : err) });
        } catch {
            /* server gone — nothing else to report to */
        }
    }
}

// --- capture job -----------------------------------------------------------------

async function runCapture() {
    const renderer = createRenderer();
    const glName = rendererName(renderer);
    status(`WebGL: ${glName}`);
    if (/swiftshader|software|llvmpipe/i.test(glName) && params.get("force") !== "1") {
        throw new Error(
            `software WebGL (${glName}) — headless browser has no GPU; open this ` +
                "URL in a normal browser, or relaunch with a GPU-capable headless " +
                "Chrome/Edge (append &force=1 to render anyway)",
        );
    }

    const mres = await fetch(stagePath("/manifest"), { cache: "no-store" });
    if (!mres.ok) throw new Error(`manifest HTTP ${mres.status} (stale job token?)`);
    const manifest = await mres.json();
    const R = manifest.resolution;
    status(
        `job: ${RUN}/${SLOT}/${MODEL} — ${manifest.pending.length}/${manifest.total} ` +
            `views pending at ${R}²`,
    );

    const plan = await (await fetch(new URL(manifest.cameras_url, API), { cache: "no-store" })).json();
    const { scene, loaded, failed: badObjects } = await loadScene(
        renderer,
        new URL(manifest.bundle_url, API),
    );
    status(`scene: ${loaded} objects loaded${badObjects ? `, ${badObjects} failed` : ""}`);
    if (loaded === 0) throw new Error("no objects loaded from the mesh bundle");

    const capture = createCapture(
        renderer,
        R,
        manifest.near,
        manifest.far,
        manifest.fov_deg,
        manifest.background,
    );
    const ring = createReadbackRing(renderer, R);

    // Pending ids → render entries via the plan (id = cam{index:05d}_{face}).
    const queue = [];
    for (const vid of manifest.pending) {
        const m = /^cam(\d+)_(.+)$/.exec(vid);
        if (!m) throw new Error(`unparseable view id from server: ${vid}`);
        const cam = plan.cameras[Number(m[1])];
        const face = plan.cube_faces[m[2]];
        if (!cam || !face) throw new Error(`view ${vid} not in the camera plan`);
        queue.push({ id: vid, pos: cam.pos, face });
    }

    // The render thread does ONLY GPU work + PBO copies; the depth swizzle, SRF1
    // packing, and POSTs live in a dedicated worker (splatcapture-worker.js) fed
    // by zero-copy buffer transfers — the profiled ~6 ms/view of post-processing
    // came off this thread, which was the pipeline's wall.
    const bytesPerView = R * R * 4 + R * R * 2; // wire size (rgba + u16 codes)
    const batchViews = Math.max(1, Math.floor(POST_BATCH_BYTES / bytesPerView));
    // Bound the raw frames buffered in the worker (rgba + RG8 depth per view =
    // 6·n bytes) so a slow server backpressures the renderer, not memory.
    const maxOutstanding = Math.max(
        2 * batchViews,
        Math.floor((256 * 1024 * 1024) / (R * R * 6)),
    );
    const worker = new Worker("/js/splatcapture-worker.js");
    let postedViews = 0;
    let postError = null; // a failed batch is fatal — silence would strand views
    let workerDrained = false;
    worker.onmessage = (ev) => {
        if (ev.data.posted) postedViews += ev.data.posted;
        else if (ev.data.error) postError = new Error(ev.data.error);
        else if (ev.data.drained) workerDrained = true;
    };
    worker.postMessage({
        cfg: { framesUrl: stagePath("/frames"), resolution: R, batchViews },
    });

    let renderedViews = 0;
    let lastProgress = 0;
    // Per-stage wall-time buckets (ms) — the answer to "what is the render thread
    // actually doing"; shown live, logged, and shipped in the finish stats.
    const buckets = { render: 0, collect: 0, spin: 0 };
    const t0 = performance.now();

    while (queue.length || ring.inFlight()) {
        if (postError) throw postError;
        let progressed = false;
        if (queue.length && ring.canEnqueue() && renderedViews - postedViews < maxOutstanding) {
            const v = queue.shift();
            const tr = performance.now();
            capture.renderView(scene, v.pos, v.face);
            ring.enqueue(v, capture.rtColor, capture.rtDepth);
            buckets.render += performance.now() - tr;
            renderedViews += 1;
            progressed = true;
        }
        if (queue.length === 0) ring.seal(); // flush the partial tail segment
        let seg;
        while ((seg = ring.readySealed())) {
            const tc = performance.now();
            const r = ring.collect(seg);
            worker.postMessage({ segment: r.buffer, ids: r.ids }, [r.buffer]);
            buckets.collect += performance.now() - tc;
            progressed = true;
        }
        const now = performance.now();
        if (now - lastProgress > 500) {
            lastProgress = now;
            const rate = postedViews / Math.max(0.001, (now - t0) / 1000);
            progress(
                `rendering… ${postedViews}/${manifest.pending.length} posted ` +
                    `(${renderedViews} rendered, ${rate.toFixed(1)} views/s) · ` +
                    bucketLine(buckets, Math.max(renderedViews, 1)),
            );
        }
        // Fences land ~1-3 ms after issue, but a setTimeout wait is clamped to
        // ~4 ms once nested — timer-pacing the whole loop. So while readbacks
        // are in flight, spin on the (unclamped, throttle-immune) MessageChannel
        // yield and re-poll; sleep only when truly starved (worker backpressure).
        const ty = performance.now();
        if (progressed || ring.inFlight()) await yieldLoop();
        else await sleep(2);
        buckets.spin += performance.now() - ty;
    }
    worker.postMessage({ flush: true });
    while (!workerDrained) {
        if (postError) throw postError;
        await sleep(20);
    }
    worker.terminate();

    const secs = (performance.now() - t0) / 1000;
    const rate = postedViews / Math.max(secs, 0.001);
    const statsLine = bucketLine(buckets, Math.max(postedViews, 1));
    console.log(`[capture] ${rate.toFixed(1)} views/s · per-view ${statsLine}`);
    status(
        `rendered ${postedViews} views in ${secs.toFixed(1)}s ` +
            `(${rate.toFixed(1)} views/s · ${statsLine}) — finishing…`,
    );
    const fin = await postFinish({
        renderer: glName,
        stats: {
            views_per_s: Number(rate.toFixed(2)),
            per_view_ms: perViewMs(buckets, Math.max(postedViews, 1)),
            resolution: R,
        },
    });
    if (fin.missing && fin.missing > 0) {
        status(`finish: ${fin.missing} views still missing (POST stage 5 again to resume)`, "err");
    } else {
        status("capture complete — transforms.json written", "ok");
    }
}

// --- self-test -------------------------------------------------------------------
// Synthetic scene through the REAL capture path: a UV-encoding quad (glTF UV
// convention) fronto-parallel at z = 2, camera at the origin looking +Z. Checks
// the conventions that would silently corrupt training data if wrong:
// orientation/flips, planar (not ray) depth, the exact log-u16 code mapping,
// alpha coverage, and the background contract.

function selftestChecks(rgba, codes, R, near, far) {
    const results = [];
    const check = (name, ok, detail = "") =>
        results.push({ name, ok, detail });
    // Server-side row flip emulated here: report in top-down coordinates.
    const px = (fr, fc) => {
        const row = Math.round(fr * (R - 1));
        const col = Math.round(fc * (R - 1));
        const glRow = R - 1 - row;
        return glRow * R + col;
    };
    const rgb = (i) => [rgba[i * 4] / 255, rgba[i * 4 + 1] / 255, rgba[i * 4 + 2] / 255];
    const alpha = (i) => rgba[i * 4 + 3] / 255;
    const encode = (z) =>
        Math.round((Math.log(z / near) / Math.log(far / near)) * (DEPTH_CODE_MAX - 1)) + 1;

    const c = px(0.5, 0.5);
    check(
        "depth at center == planar Z0 (log-u16 code)",
        Math.abs(codes[c] - encode(2.0)) <= 2,
        `code=${codes[c]} want≈${encode(2.0)}`,
    );
    const off = px(0.28, 0.28); // on-quad but well off-axis
    const rayZ = 2.0 * Math.hypot(1, 0.44, 0.44); // what ray-distance depth would read
    check(
        "off-axis depth is planar, NOT ray distance",
        Math.abs(codes[off] - encode(2.0)) <= 2 && Math.abs(codes[off] - encode(rayZ)) > 8,
        `code=${codes[off]} planar≈${encode(2.0)} ray≈${encode(rayZ)}`,
    );
    const [tl, tr, bl, br] = [px(0.3, 0.3), px(0.3, 0.7), px(0.7, 0.3), px(0.7, 0.7)].map(rgb);
    check("top-left dark (u≈0, v≈0)", tl[0] < 0.4 && tl[1] < 0.4, `rgb=${tl.map((x) => x.toFixed(2))}`);
    check("top-right red (u≈1, v≈0)", tr[0] > 0.6 && tr[1] < 0.4, `rgb=${tr.map((x) => x.toFixed(2))}`);
    check("bottom-left green (u≈0, v≈1)", bl[0] < 0.4 && bl[1] > 0.6, `rgb=${bl.map((x) => x.toFixed(2))}`);
    check("bottom-right yellow (u≈1, v≈1)", br[0] > 0.6 && br[1] > 0.6, `rgb=${br.map((x) => x.toFixed(2))}`);
    check("alpha == 1 on quad", alpha(c) > 0.99, `α=${alpha(c).toFixed(3)}`);
    const corner = px(0.02, 0.02); // beyond the quad's ±1.3 extent at 90°
    check("alpha == 0 on background", alpha(corner) < 0.01, `α=${alpha(corner).toFixed(3)}`);
    check("background RGB == black", Math.max(...rgb(corner)) < 0.02);
    check("background depth code == 0", codes[corner] === 0, `code=${codes[corner]}`);
    return results;
}

// Synthetic UV-encoding quad at z = 2 — the selftest's convention probe and the
// bench's render load. Built like a loaded glTF (flipY=false, raw glTF UVs).
function buildUvQuadScene() {
    // UV-encoding texture in glTF convention: row 0 of the DATA is texture v=0.
    // GL samples v=0 at data row 0, and glTF UVs put v=0 at the image TOP — so
    // fill row y with v = y (v grows down the image), exactly like a glTF PNG.
    const TW = 256;
    const data = new Uint8Array(TW * TW * 4);
    for (let y = 0; y < TW; y++) {
        for (let x = 0; x < TW; x++) {
            const i = (y * TW + x) * 4;
            data[i] = Math.round((255 * x) / (TW - 1)); // R = u
            data[i + 1] = Math.round((255 * y) / (TW - 1)); // G = v
            data[i + 2] = 128;
            data[i + 3] = 255;
        }
    }
    const tex = new THREE.DataTexture(data, TW, TW, THREE.RGBAFormat);
    tex.needsUpdate = true;
    tex.colorSpace = THREE.NoColorSpace;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;

    // Quad at z=2 with glTF UVs (v=0 along the world-top edge), like the old
    // nvdiffrast smoke: A(+1.3,+1.3) B(−1.3,+1.3) C(−1.3,−1.3) D(+1.3,−1.3).
    const HALF = 1.3;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
        "position",
        new THREE.BufferAttribute(
            new Float32Array([
                HALF, HALF, 2, -HALF, HALF, 2, -HALF, -HALF, 2,
                HALF, HALF, 2, -HALF, -HALF, 2, HALF, -HALF, 2,
            ]),
            3,
        ),
    );
    geo.setAttribute(
        "uv",
        new THREE.BufferAttribute(
            new Float32Array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]),
            2,
        ),
    );
    // No UV rewrite: with flipY=false (DataTexture default, and what GLTFLoader
    // sets on every glTF texture) uv v=0 samples data row 0, which IS the glTF
    // "image top" — the raw glTF UVs above are already in-convention.

    const scene = new THREE.Scene();
    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide }));
    mesh.material.toneMapped = false;
    scene.add(mesh);
    return scene;
}

// Throughput bench (?selftest=1&bench=N[&res=512][&mode=pbo|sync]) — the capture
// pipeline with the server and scene loading factored OUT: renders the synthetic
// quad N times through the real render + readback path, no POSTs. Isolates
// where the per-view time goes (GPU/render vs fence+readback vs swizzle) and
// what the page-side ceiling actually is on this browser/GPU/driver.
async function runBench(n, mode) {
    const renderer = createRenderer();
    status(`WebGL: ${rendererName(renderer)}`);
    const R = Number(params.get("res")) || 512;
    const near = 0.05;
    const far = 10.0;
    const scene = buildUvQuadScene();
    const capture = createCapture(renderer, R, near, far, 90.0, [0, 0, 0]);
    const view = { pos: [0, 0, 0], face: { forward: [0, 0, 1], up: [0, 1, 0] } };
    const buckets = { render: 0, collect: 0, spin: 0 };
    status(`bench: ${n} views at ${R}² (${mode} readback)…`);
    const t0 = performance.now();

    if (mode === "sync") {
        // Blocking readback baseline: no PBOs, no fences — full pipeline stall
        // per view, plus the same swizzle cost as the real path.
        const rgba = new Uint8Array(R * R * 4);
        const depthRG = new Uint8Array(R * R * 2); // RG8 = u16 codes (LE)
        for (let i = 0; i < n; i++) {
            const tr = performance.now();
            capture.renderView(scene, view.pos, view.face);
            buckets.render += performance.now() - tr;
            const tc = performance.now();
            renderer.readRenderTargetPixels(capture.rtColor, 0, 0, R, R, rgba);
            renderer.readRenderTargetPixels(capture.rtDepth, 0, 0, R, R, depthRG);
            buckets.collect += performance.now() - tc;
            if (i % 128 === 0) {
                progress(`bench… ${i}/${n}`);
                await yieldLoop();
            }
        }
    } else {
        // Segmented-PBO path, matching runCapture (readback measured WITHOUT the
        // depth swizzle — that lives in the post worker in the real pipeline).
        const ring = createReadbackRing(renderer, R);
        let issued = 0;
        let done = 0;
        let lastProgress = 0;
        while (done < n) {
            let progressed = false;
            if (issued < n && ring.canEnqueue()) {
                const tr = performance.now();
                capture.renderView(scene, view.pos, view.face);
                ring.enqueue({ id: `b${issued}` }, capture.rtColor, capture.rtDepth);
                buckets.render += performance.now() - tr;
                issued += 1;
                progressed = true;
            }
            if (issued === n) ring.seal();
            let seg;
            while ((seg = ring.readySealed())) {
                const tc = performance.now();
                done += ring.collect(seg).ids.length;
                buckets.collect += performance.now() - tc;
                progressed = true;
            }
            const now = performance.now();
            if (now - lastProgress > 500) {
                lastProgress = now;
                progress(`bench… ${done}/${n} (${(done / ((now - t0) / 1000)).toFixed(1)} views/s)`);
            }
            const ty = performance.now();
            if (progressed || ring.inFlight()) await yieldLoop();
            else await sleep(2);
            buckets.spin += performance.now() - ty;
        }
    }

    const secs = (performance.now() - t0) / 1000;
    const line = `bench(${mode}) ${R}²: ${(n / secs).toFixed(1)} views/s · ${bucketLine(buckets, n)}`;
    console.log(`[capture] ${line}`);
    status(line, "ok");
}

async function runSelftest() {
    const renderer = createRenderer();
    status(`WebGL: ${rendererName(renderer)}`);
    const R = 512;
    const near = 0.05;
    const far = 10.0;
    const scene = buildUvQuadScene();

    const capture = createCapture(renderer, R, near, far, 90.0, [0, 0, 0]);
    capture.renderView(scene, [0, 0, 0], { forward: [0, 0, 1], up: [0, 1, 0] });

    const rgba = new Uint8Array(R * R * 4);
    renderer.readRenderTargetPixels(capture.rtColor, 0, 0, R, R, rgba);
    // RG8 depth readback: the bytes are the log-u16 codes little-endian, so a
    // Uint16Array view over them IS the code array (no swizzle) — the same
    // reinterpret the capture worker relies on.
    const depthRG = new Uint8Array(R * R * 2);
    renderer.readRenderTargetPixels(capture.rtDepth, 0, 0, R, R, depthRG);
    const codes = new Uint16Array(depthRG.buffer);

    const results = selftestChecks(rgba, codes, R, near, far);
    for (const r of results) {
        status(`[${r.ok ? "PASS" : "FAIL"}] ${r.name}${r.detail ? ` — ${r.detail}` : ""}`, r.ok ? "ok" : "err");
    }
    const bad = results.filter((r) => !r.ok).length;
    status(bad ? `${bad} CHECK(S) FAILED` : "ALL CAPTURE SELF-TESTS PASSED", bad ? "err" : "ok");

    // Leave the preview on the canvas for eyeballing.
    renderer.setRenderTarget(null);
    capture.renderView(scene, [0, 0, 0], { forward: [0, 0, 1], up: [0, 1, 0] });
    renderer.setRenderTarget(null);
    renderer.render(scene, capture.camera);
}

// --- entry -------------------------------------------------------------------------

(async () => {
    try {
        if (SELFTEST) {
            const bench = Number(params.get("bench") || 0);
            if (bench > 0) await runBench(bench, params.get("mode") || "pbo");
            else await runSelftest();
        } else {
            if (!API || !RUN || !SLOT || !MODEL || !TOKEN) {
                throw new Error(
                    "missing query params (?api=&run=&slot=&model=&token=) — this page " +
                        "is launched by the server's stage-5 job (or add &selftest=1)",
                );
            }
            await runCapture();
        }
    } catch (e) {
        console.error(e);
        await fail(e);
    }
})();
