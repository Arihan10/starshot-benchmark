// =============================================================================
// tourview.js — inspector for a captured tour's panos + object-ID masks.
// =============================================================================
//
// A deliberately small stand-in for the prod walkthrough (prod_client/): pick a
// captured tour, page through every 360 pano, and hover the image to see the
// object-ID mask do its job. It reads the SAME artifacts the real viewer will —
// tour.json, `{anchor}.jpg`, `{anchor}.sid` — straight off the orchestrator's
// /artifacts route, with no publish step involved.
//
// It exists to answer three questions about a capture, which is why it shows what
// it shows:
//   • does the mask REGISTER with the image?   → "mask view" paints the raw ID
//     plane in false colour through the same equirect material as the pano, so
//     any misalignment is immediately visible as boundaries sliding off edges.
//   • is the hover right?                       → the highlight is driven purely by
//     sampling the mask along the cursor ray (idmask.js `sampleIdMask`), never by
//     raycasting geometry — exactly what the prod viewer will do.
//   • what does it cost?                        → the stats panel reports each
//     mask's bytes against its own pano's.
//
// The camera sits at the origin looking around, which is legitimate rather than a
// simplification: a walkthrough only ever stands ON a capture point, so the pano
// is an exact skybox from there and screen pixel → ray direction → mask texel is
// the whole of the lookup.

import * as THREE from "three";
import { SERVER_URL, api } from "./api.js";
import {
    ID_BACKGROUND,
    createMaskTexture,
    decodeIdMask,
    sampleIdMask,
} from "./idmask.js";

const el = {
    tour: document.getElementById("tour"),
    list: document.getElementById("list"),
    view: document.getElementById("view"),
    hud: document.getElementById("hud"),
    stats: document.getElementById("stats"),
    msg: document.getElementById("msg"),
    prev: document.getElementById("prev"),
    next: document.getElementById("next"),
    highlight: document.getElementById("highlight"),
    maskview: document.getElementById("maskview"),
};

let tours = [];
let tour = null; // { row, base, manifest, panos, objects }
let index = -1;
let mask = null; // decoded SID1 container for the current anchor
let maskStats = null;
let hoverGlobal = ID_BACKGROUND;
let showHighlight = true;
let showMaskView = false;
let loadToken = 0;
let panoTexture = null; // the photo, kept aside while false colour is showing
let maskPreview = null; // the ID plane painted in false colour (built on demand)

let msgSeq = 0;
function note(text, cls = "", clearAfter = 0) {
    const seq = ++msgSeq;
    el.msg.textContent = text;
    el.msg.className = `overlay ${cls}`;
    el.msg.hidden = !text;
    if (clearAfter) setTimeout(() => seq === msgSeq && note(""), clearAfter);
}

const assetUrl = (file) => new URL(tour.base + file, SERVER_URL).href;
const prettyId = (id) => id.replace(/_/g, " ");

// --- viewer ------------------------------------------------------------------

// Equirect backdrop, byte-for-byte the convention idmask.js documents and
// prod_client's makePanoMaterial uses: u = atan2(z,x)/2pi + 0.5, v = asin(y)/pi + 0.5.
function makePanoMaterial() {
    return new THREE.ShaderMaterial({
        uniforms: { map: { value: null } },
        vertexShader: /* glsl */ `
            varying vec3 vDir;
            void main() {
                vDir = position;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: /* glsl */ `
            uniform sampler2D map;
            varying vec3 vDir;
            void main() {
                vec3 d = normalize(vDir);
                vec2 uv = vec2(
                    atan(d.z, d.x) / 6.28318530718 + 0.5,
                    asin(clamp(d.y, -1.0, 1.0)) / 3.14159265359 + 0.5
                );
                gl_FragColor = vec4(texture2D(map, uv).rgb, 1.0);
            }
        `,
        side: THREE.BackSide,
        depthWrite: false,
        depthTest: false,
    });
}

// The highlight: for each screen pixel, turn the view ray into an equirect texel
// and ask how much of it belongs to the hovered object.
//
// "How much", not "whether", is the whole difference between a smooth edge and a
// staircase. A binary answer quantizes the boundary to the mask grid no matter how
// fine that grid is, and a walkthrough magnifies enough to show it. So the mask
// carries the sub-texel coverage its supersampled raster measured (idmask.js), and
// this reconstructs a continuous field from it: each of the four neighbouring
// texels contributes its share of the hovered object — its own coverage where the
// texel is ours, the winner's leftover where it is a texel we bleed into — and
// those are bilerped into a continuous field, which `fwidth` then converts into a
// screen-space distance so the stroke holds its weight at any zoom.
//
// The field is computed with no branching above it on purpose: derivatives are
// undefined in non-uniform control flow, so an early `discard` for fragments
// outside the object would corrupt `fwidth` for the surviving fragments in the same
// quad — which are exactly the ones the outline is drawn on.
//
// THICKNESS comes from a ring, not from that field. Coverage only carries signal
// within about a texel of the boundary, so a stroke derived from its gradient
// saturates at roughly a pixel wide and then varies with zoom. Instead the ring
// asks the mask directly: step `uWidth` screen pixels out in twelve directions and
// count how many land outside the object. That count is a smooth function of the
// distance to the boundary — (1/pi)·acos(t/R) for a straight edge — so it gives a
// stroke of any requested width, feathered inward, at a fixed weight on screen.
// The steps are taken in DIRECTION space (rotate the view ray, then re-project),
// which is seam-safe and pole-safe; doing it in uv would blow up where u wraps and
// where the equirect stretches.
//
// Sampling at `1 - v`: the ID plane is stored top-down while the DataTexture is
// flipY = false, so texture t = 0 is the plane's first row (straight up).
function makeHighlightMaterial() {
    return new THREE.ShaderMaterial({
        uniforms: {
            uMask: { value: null },
            uTexel: { value: new THREE.Vector2(1, 1) },
            uLocal: { value: 0 },
            uWide: { value: false },
            uEdge: { value: new THREE.Color(0xbfe8ff) },
            uWidth: { value: 3.0 }, // stroke width in SCREEN pixels
        },
        vertexShader: /* glsl */ `
            varying vec3 vDir;
            void main() {
                vDir = position;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: /* glsl */ `
            uniform sampler2D uMask;
            uniform vec2 uTexel;
            uniform float uLocal;
            uniform bool uWide;
            uniform vec3 uEdge;
            uniform float uWidth;
            varying vec3 vDir;

            // (local id, coverage) in one tap — the mask is interleaved so four
            // neighbours cost four samples, not eight.
            vec2 tap(vec2 uv) {
                vec4 t = texture2D(uMask, uv);
                return uWide ? vec2(t.r * 255.0 + t.g * 65280.0, t.b) : vec2(t.r * 255.0, t.g);
            }
            float ours(vec2 s) {
                return abs(s.x - uLocal) < 0.5 ? 1.0 : 0.0;
            }
            float share(vec2 s, float near) {
                return mix(near * (1.0 - s.y), s.y, ours(s));
            }
            vec2 dirToUv(vec3 d) {
                return vec2(
                    atan(d.z, d.x) / 6.28318530718 + 0.5,
                    1.0 - (asin(clamp(d.y, -1.0, 1.0)) / 3.14159265359 + 0.5)
                );
            }
            // Sub-texel coverage of the hovered object, bilerped from the four
            // neighbouring texels.
            float coverageAt(vec2 uv) {
                vec2 t = uv / uTexel - 0.5;
                vec2 f = fract(t);
                vec2 b = (floor(t) + 0.5) * uTexel;
                vec2 s00 = tap(b);
                vec2 s10 = tap(b + vec2(uTexel.x, 0.0));
                vec2 s01 = tap(b + vec2(0.0, uTexel.y));
                vec2 s11 = tap(b + uTexel);
                // Claiming a neighbour's leftover is only meaningful next to us.
                float near = max(max(ours(s00), ours(s10)), max(ours(s01), ours(s11)));
                return mix(
                    mix(share(s00, near), share(s10, near), f.x),
                    mix(share(s01, near), share(s11, near), f.x),
                    f.y
                );
            }
            void main() {
                if (uLocal < 0.5) discard;
                vec3 d = normalize(vDir);
                float c = coverageAt(dirToUv(d));

                // Radians per screen pixel, measured on the direction itself so it
                // stays finite at the seam and at the poles.
                float angPerPx = length(fwidth(d));
                vec3 axis = abs(d.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
                vec3 tx = normalize(cross(axis, d));
                vec3 ty = cross(d, tx);
                float r = uWidth * angPerPx;

                float outside = 0.0;
                for (int i = 0; i < 12; i++) {
                    float a = float(i) * 0.5235987756; // 30°
                    vec3 s = normalize(d + (tx * cos(a) + ty * sin(a)) * r);
                    outside += 1.0 - ours(tap(dirToUv(s)));
                }
                // Half the ring is outside when we sit on the boundary, none of it
                // once we are a full width inside — so doubling gives a stroke that
                // is solid at the edge and fades to nothing uWidth pixels in.
                float band = clamp((outside / 12.0) * 2.0, 0.0, 1.0);

                // Drawn from the boundary INWARD. A stroke centred on the boundary
                // spills half its width onto whatever lies beyond, which paints over
                // the neighbour (or the occluder) the mask says owns those pixels.
                // This term is the sub-pixel-accurate outer edge, band is the body.
                float inside = clamp((c - 0.5) / max(fwidth(c), 1e-5) + 0.5, 0.0, 1.0);

                float a = inside * band;
                if (a <= 0.004) discard;
                gl_FragColor = vec4(uEdge, a);
            }
        `,
        side: THREE.BackSide,
        transparent: true,
        depthWrite: false,
        depthTest: false,
    });
}

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
el.view.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 200);
const sphere = new THREE.SphereGeometry(50, 96, 48);
const panoMat = makePanoMaterial();
const highlightMat = makeHighlightMaterial();
const panoMesh = new THREE.Mesh(sphere, panoMat);
const highlightMesh = new THREE.Mesh(sphere, highlightMat);
highlightMesh.renderOrder = 10;
highlightMesh.visible = false;
scene.add(panoMesh, highlightMesh);

let lon = 0;
let lat = 0;
let dragging = false;
let lastX = 0;
let lastY = 0;
let pointerX = 0;
let pointerY = 0;
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
const forward = new THREE.Vector3();

function resize() {
    const w = el.view.clientWidth;
    const h = el.view.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(el.view);
resize();

renderer.setAnimationLoop(() => {
    forward.set(
        Math.cos(lat) * Math.cos(lon),
        Math.sin(lat),
        Math.cos(lat) * Math.sin(lon),
    );
    camera.lookAt(forward);
    renderer.render(scene, camera);
});

const canvas = renderer.domElement;
canvas.addEventListener("pointerdown", (ev) => {
    dragging = true;
    lastX = ev.clientX;
    lastY = ev.clientY;
    canvas.setPointerCapture(ev.pointerId);
    canvas.style.cursor = "grabbing";
});
canvas.addEventListener("pointerup", (ev) => {
    dragging = false;
    canvas.releasePointerCapture(ev.pointerId);
    canvas.style.cursor = "";
});
canvas.addEventListener("pointermove", (ev) => {
    pointerX = ev.clientX;
    pointerY = ev.clientY;
    if (dragging) {
        const k = ((camera.fov * Math.PI) / 180 / el.view.clientHeight) * 1.0;
        lon -= (ev.clientX - lastX) * k;
        lat = Math.max(-1.5, Math.min(1.5, lat + (ev.clientY - lastY) * k));
        lastX = ev.clientX;
        lastY = ev.clientY;
    }
    updateHover();
});
canvas.addEventListener("pointerleave", () => {
    hoverGlobal = ID_BACKGROUND;
    highlightMat.uniforms.uLocal.value = 0;
    renderHud();
});
canvas.addEventListener(
    "wheel",
    (ev) => {
        ev.preventDefault();
        camera.fov = Math.max(20, Math.min(110, camera.fov + ev.deltaY * 0.05));
        camera.updateProjectionMatrix();
        updateHover();
    },
    { passive: false },
);

// The whole pick: screen pixel → world ray → equirect texel. No geometry, no
// depth test, no proxy. The 1-texel majority stops a sliver under the cursor
// flickering the label between two objects.
function updateHover() {
    if (!mask) return;
    const rect = canvas.getBoundingClientRect();
    ndc.set(
        ((pointerX - rect.left) / rect.width) * 2 - 1,
        -((pointerY - rect.top) / rect.height) * 2 + 1,
    );
    raycaster.setFromCamera(ndc, camera);
    const d = raycaster.ray.direction;
    const g = sampleIdMask(mask, d.x, d.y, d.z, 1);
    if (g === hoverGlobal) return;
    hoverGlobal = g;
    highlightMat.uniforms.uLocal.value = mask.localOf.get(g) ?? 0;
    renderHud();
}

// --- data --------------------------------------------------------------------

async function loadTours() {
    const res = await api.tours();
    tours = (res.tours ?? []).filter((t) => t.captured && t.tour_url);
    el.tour.innerHTML = "";
    if (tours.length === 0) {
        el.tour.innerHTML = "<option>no captured tours</option>";
        note("no captured tours on this server — run a tour capture first", "err");
        return;
    }
    for (const [i, t] of tours.entries()) {
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = `${t.run} · ${t.slot} · ${t.model} — ${t.panos} panos`;
        el.tour.appendChild(opt);
    }
    await loadTour(0);
}

async function loadTour(i) {
    const row = tours[i];
    const token = ++loadToken;
    note(`loading ${row.slot}/${row.model}…`);
    const manifest = await (
        await fetch(new URL(row.tour_url, SERVER_URL), { cache: "no-store" })
    ).json();
    if (token !== loadToken) return;
    tour = {
        row,
        base: row.tour_url.slice(0, row.tour_url.lastIndexOf("/") + 1),
        manifest,
        panos: manifest.panos ?? [],
        objects: manifest.objects ?? [],
    };
    index = -1;
    renderList();
    if (tour.panos.length === 0) {
        note("this tour has no panos", "err");
        return;
    }
    const withMask = tour.panos.filter((p) => p.mask).length;
    if (withMask === 0) {
        note("captured before object-ID masks — re-capture this tour to add them", "err");
    } else {
        note(
            `${tour.objects.length} objects · masks on ${withMask}/${tour.panos.length} anchors`,
            "ok",
            4000,
        );
    }
    await showAnchor(0);
}

function renderList() {
    el.list.innerHTML = "";
    tour.panos.forEach((p, i) => {
        const row = document.createElement("div");
        row.className = `row${p.mask ? "" : " nomask"}${i === index ? " sel" : ""}`;
        row.dataset.index = String(i);
        row.innerHTML =
            `<span class="n">${String(i + 1).padStart(3, "0")}</span>` +
            `<span class="id">${p.name ?? p.id}</span>` +
            `<span class="zone">${p.zone ?? ""}</span>`;
        row.addEventListener("click", () => showAnchor(i));
        el.list.appendChild(row);
    });
}

function markSelected() {
    for (const row of el.list.children) {
        row.classList.toggle("sel", Number(row.dataset.index) === index);
    }
    el.list.children[index]?.scrollIntoView({ block: "nearest" });
}

const textureLoader = new THREE.TextureLoader();

// Fetched as a blob rather than handed straight to the loader, so its exact size
// is known — the stats panel weighs the mask against it. (A HEAD probe would be
// the obvious way, but FastAPI doesn't add HEAD to its GET routes, so /artifacts
// answers 405 and the "size" you get back is the length of the error body.)
// Texture settings match prod's loadPanoTexture: no mips, since they break at the
// equirect seam where the u-derivative jumps a full wrap, and repeat wrapping so
// the seam stays continuous.
async function loadPano(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`pano HTTP ${res.status}`);
    const blob = await res.blob();
    const objectUrl = URL.createObjectURL(blob);
    try {
        const tex = await textureLoader.loadAsync(objectUrl);
        tex.generateMipmaps = false;
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        tex.wrapS = THREE.RepeatWrapping;
        return { tex, bytes: blob.size };
    } finally {
        URL.revokeObjectURL(objectUrl);
    }
}

async function showAnchor(i) {
    if (!tour || i < 0 || i >= tour.panos.length) return;
    const token = ++loadToken;
    const p = tour.panos[i];
    index = i;
    // Drop the previous anchor's mask before the fetch, not after: none of it is
    // true here, and the pano is big enough that a stale highlight would be
    // visible (and its false-colour preview is about to be disposed).
    highlightMat.uniforms.uMask.value?.dispose();
    highlightMat.uniforms.uMask.value = null;
    highlightMat.uniforms.uLocal.value = 0;
    highlightMesh.visible = false;
    maskPreview?.dispose();
    maskPreview = null;
    mask = null;
    maskStats = null;
    hoverGlobal = ID_BACKGROUND;
    applyMaskView();
    markSelected();
    renderStats();
    renderHud();

    try {
        const { tex: panoTex, bytes: panoBytes } = await loadPano(assetUrl(p.file));
        if (token !== loadToken) {
            panoTex.dispose();
            return;
        }
        panoTexture?.dispose();
        panoTexture = panoTex;

        if (p.mask) {
            const res = await fetch(assetUrl(p.mask), { cache: "no-store" });
            if (!res.ok) throw new Error(`mask HTTP ${res.status}`);
            const buf = await res.arrayBuffer();
            if (token !== loadToken) return;
            const t0 = performance.now();
            const decoded = await decodeIdMask(buf);
            const decodeMs = performance.now() - t0;
            if (token !== loadToken) return;
            mask = decoded;
            maskStats = { bytes: buf.byteLength, decodeMs, panoBytes };
            highlightMat.uniforms.uMask.value = createMaskTexture(decoded);
            highlightMat.uniforms.uTexel.value.set(1 / decoded.width, 1 / decoded.height);
            highlightMat.uniforms.uWide.value = decoded.indexBytes === 2;
            highlightMesh.visible = showHighlight;
            updateHover();
        }
        applyMaskView();
        renderStats();
        renderHud();
    } catch (e) {
        if (token === loadToken) note(`${p.id}: ${e.message}`, "err");
    }
}

// --- false-colour ID plane ---------------------------------------------------

// Local index → a distinct colour, hue walked by the golden ratio so neighbouring
// ids never land on similar shades. Written straight as sRGB bytes (no three.js
// colour management anywhere in the path) into a canvas the PANO material samples,
// so the false-colour plane and the photo it labels line up pixel for pixel —
// which is the point of the view: any registration error shows up as a boundary
// sliding off an edge.
function hslBytes(h, s, l, out, o) {
    const a = s * Math.min(l, 1 - l);
    const f = (n) => {
        const k = (n + h * 12) % 12;
        return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
    };
    out[o] = f(0);
    out[o + 1] = f(8);
    out[o + 2] = f(4);
}

function buildMaskPreview() {
    const { width, height, plane, palette } = mask;
    const lut = new Uint8Array((palette.length + 1) * 3);
    for (let i = 1; i <= palette.length; i++) {
        hslBytes((i * 0.6180339887) % 1, 0.62, 0.56, lut, i * 3);
    }
    const img = new ImageData(width, height);
    const px = img.data;
    // Dimmed by coverage, so the antialiasing band reads directly: a solid block of
    // colour is a texel one object owns outright, a darker fringe is a boundary
    // texel carrying its sub-texel share.
    for (let i = 0, o = 0; i < plane.length; i++, o += 4) {
        const l = plane[i];
        const cov = (85 + (mask.coverage[i] * 170) / 255) / 255;
        px[o] = lut[l * 3] * cov;
        px[o + 1] = lut[l * 3 + 1] * cov;
        px[o + 2] = lut[l * 3 + 2] * cov;
        px[o + 3] = 255;
    }
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(canvas);
    tex.generateMipmaps = false;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.NearestFilter;
    tex.wrapS = THREE.RepeatWrapping;
    tex.colorSpace = THREE.NoColorSpace;
    return tex;
}

function applyMaskView() {
    if (showMaskView && mask) {
        if (!maskPreview) maskPreview = buildMaskPreview();
        panoMat.uniforms.map.value = maskPreview;
    } else {
        panoMat.uniforms.map.value = panoTexture;
    }
}

// --- chrome ------------------------------------------------------------------

function renderHud() {
    if (!tour || index < 0) {
        el.hud.textContent = "";
        return;
    }
    const p = tour.panos[index];
    const head =
        `<span class="dim">${index + 1}/${tour.panos.length}</span> ` +
        `${p.name ?? p.id}${p.zone ? ` <span class="dim">· ${p.zone}</span>` : ""}`;
    let body;
    if (!mask) {
        body = `<span class="dim">no mask for this anchor</span>`;
    } else if (hoverGlobal === ID_BACKGROUND) {
        body = `<span class="dim">background</span>`;
    } else {
        const obj = tour.objects[hoverGlobal - 1];
        body = obj
            ? `<span class="obj">${prettyId(obj.id)}</span> <span class="dim">#${hoverGlobal}</span>`
            : `<span class="obj">#${hoverGlobal}</span> <span class="dim">(not in directory)</span>`;
    }
    el.hud.innerHTML = `${head}<br />${body}`;
}

function renderStats() {
    if (!mask || !maskStats) {
        el.stats.textContent = "";
        return;
    }
    const kb = (n) => `${(n / 1024).toFixed(1)} KB`;
    const share = maskStats.panoBytes
        ? ` (${((maskStats.bytes / maskStats.panoBytes) * 100).toFixed(1)}% of pano)`
        : "";
    const ss = mask.supersample || 1;
    el.stats.textContent = [
        `${mask.width}×${mask.height} · ${mask.indexBytes * 8}-bit · ${mask.filter === 1 ? "sub-left" : "raw"}`,
        `${ss}×${ss} samples/texel · ${mask.palette.length} objects visible`,
        `${kb(maskStats.bytes)}${share} · pano ${kb(maskStats.panoBytes)}`,
        `decoded in ${maskStats.decodeMs.toFixed(0)} ms`,
    ].join("\n");
}

el.tour.addEventListener("change", () => {
    loadTour(Number(el.tour.value)).catch((e) => note(e.message, "err"));
});
el.prev.addEventListener("click", () => showAnchor(index - 1));
el.next.addEventListener("click", () => showAnchor(index + 1));
el.highlight.addEventListener("click", () => {
    showHighlight = !showHighlight;
    el.highlight.classList.toggle("on", showHighlight);
    highlightMesh.visible = showHighlight && !!mask;
});
el.maskview.addEventListener("click", () => {
    showMaskView = !showMaskView;
    el.maskview.classList.toggle("on", showMaskView);
    applyMaskView();
});

window.addEventListener("keydown", (ev) => {
    if (ev.target instanceof HTMLSelectElement) return;
    if (ev.code === "ArrowLeft") showAnchor(index - 1);
    else if (ev.code === "ArrowRight") showAnchor(index + 1);
    else if (ev.code === "KeyH") el.highlight.click();
    else if (ev.code === "KeyM") el.maskview.click();
});

loadTours().catch((e) => note(`failed to load tours: ${e.message}`, "err"));
