import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (sel, root = document) => root.querySelector(sel);

const els = {
    scene: $("#filter-scene"),
    model: $("#filter-model"),
    search: $("#filter-search"),
    count: $("#phrase-count"),
    list: $("#phrase-list"),
    empty: $("#phrase-empty"),
    phrase: $("#phrase-input"),
    proxy: $("#proxy"),
    view: $("#view"),
    useDims: $("#use-dims"),
    dimW: $("#dim-w"),
    dimH: $("#dim-h"),
    dimD: $("#dim-d"),
    btnBanana: $("#btn-banana"),
    btnTrellis: $("#btn-trellis"),
    status: $("#status"),
    wrapped: $("#wrapped-preview"),
    imagePane: $("#image-pane"),
    modelPane: $("#model-pane"),
};

const MAX_RENDER = 400;

const state = {
    phrases: [],
    selectedName: null,
    imageId: null,
    busy: false,
};

function setStatus(text, cls = "") {
    els.status.textContent = text;
    els.status.className = cls;
}

// --- pane helpers -----------------------------------------------------------

function clearPane(pane) {
    for (const child of [...pane.children]) {
        if (!child.classList.contains("pane-label") && !child.classList.contains("pane-help")) {
            child.remove();
        }
    }
}

function paneMessage(pane, text, { spinner = false, cls = "" } = {}) {
    clearPane(pane);
    if (spinner) {
        const sp = document.createElement("div");
        sp.className = "spinner";
        pane.appendChild(sp);
    }
    const span = document.createElement("span");
    span.className = cls === "err" ? "empty" : "empty";
    span.style.marginTop = spinner ? "10px" : "0";
    span.textContent = text;
    if (cls === "err") span.style.color = "var(--red)";
    pane.appendChild(span);
}

function paneStatus(pane, text, cls = "") {
    let badge = pane.querySelector(".pane-status");
    if (!badge) {
        badge = document.createElement("span");
        badge.className = "pane-status";
        pane.appendChild(badge);
    }
    badge.textContent = text;
    badge.className = `pane-status ${cls}`;
}

// --- phrases ----------------------------------------------------------------

async function loadPhrases() {
    let data;
    try {
        const r = await fetch("/phrases");
        if (!r.ok) throw new Error(`${r.status}`);
        data = await r.json();
    } catch (e) {
        els.empty.textContent = `failed to load phrases: ${e.message ?? e}`;
        return;
    }
    state.phrases = data.phrases ?? [];
    for (const s of data.scenes ?? []) {
        const opt = document.createElement("option");
        opt.value = s;
        opt.textContent = s;
        els.scene.appendChild(opt);
    }
    for (const m of data.models ?? []) {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        els.model.appendChild(opt);
    }
    renderList();
}

function filteredPhrases() {
    const scene = els.scene.value;
    const model = els.model.value;
    const q = els.search.value.trim().toLowerCase();
    return state.phrases.filter((p) => {
        if (scene && p.scene !== scene) return false;
        if (model && p.model !== model) return false;
        if (q && !p.name.toLowerCase().includes(q) && !p.noun_phrase.toLowerCase().includes(q)) {
            return false;
        }
        return true;
    });
}

function renderList() {
    const matches = filteredPhrases();
    els.list.innerHTML = "";
    if (matches.length === 0) {
        const empty = document.createElement("div");
        empty.id = "phrase-empty";
        empty.textContent = "no phrases match your filters.";
        els.list.appendChild(empty);
        els.count.textContent = "0 phrases";
        return;
    }
    const shown = matches.slice(0, MAX_RENDER);
    els.count.textContent =
        matches.length > shown.length
            ? `showing ${shown.length} of ${matches.length} phrases`
            : `${matches.length} phrase${matches.length === 1 ? "" : "s"}`;
    const frag = document.createDocumentFragment();
    for (const p of shown) {
        frag.appendChild(phraseCard(p));
    }
    els.list.appendChild(frag);
}

function phraseCard(p) {
    const card = document.createElement("div");
    card.className = "phrase-card";
    if (p.name === state.selectedName) card.classList.add("active");

    const head = document.createElement("div");
    head.className = "pc-head";
    const name = document.createElement("span");
    name.className = "pc-name";
    name.textContent = p.name || "(unnamed)";
    head.appendChild(name);
    if (p.source) {
        const src = document.createElement("span");
        src.className = "pc-source";
        src.textContent = p.source;
        head.appendChild(src);
    }

    const phrase = document.createElement("div");
    phrase.className = "pc-phrase";
    phrase.textContent = p.noun_phrase;

    const meta = document.createElement("div");
    meta.className = "pc-meta";
    meta.textContent = [p.scene, p.model].filter(Boolean).join(" · ");

    card.append(head, phrase, meta);
    card.addEventListener("click", () => selectPhrase(p, card));
    return card;
}

function selectPhrase(p, card) {
    state.selectedName = p.name;
    els.phrase.value = p.noun_phrase;
    for (const c of els.list.querySelectorAll(".phrase-card.active")) {
        c.classList.remove("active");
    }
    card.classList.add("active");
    setStatus(`selected "${p.name}". generate an image when ready.`);
    refreshWrap();
}

// --- wrap preview -----------------------------------------------------------

function dims() {
    if (!els.useDims.checked) return { width: null, height: null, depth: null };
    const w = parseFloat(els.dimW.value);
    const h = parseFloat(els.dimH.value);
    const d = parseFloat(els.dimD.value);
    const ok = [w, h, d].every((v) => Number.isFinite(v) && v > 0);
    return ok ? { width: w, height: h, depth: d } : { width: null, height: null, depth: null };
}

function requestBody() {
    return {
        phrase: els.phrase.value.trim(),
        proxy_shape: els.proxy.value,
        view: els.view.value,
        ...dims(),
    };
}

let wrapTimer = null;
function refreshWrap() {
    clearTimeout(wrapTimer);
    const body = requestBody();
    if (!body.phrase) {
        els.wrapped.textContent = "—";
        return;
    }
    wrapTimer = setTimeout(async () => {
        try {
            const r = await fetch("/wrap", {
                method: "POST",
                headers: { "content-type": "application/json" },
                body: JSON.stringify(body),
            });
            if (!r.ok) {
                els.wrapped.textContent = `(${r.status}) ${await r.text()}`;
                return;
            }
            const data = await r.json();
            els.wrapped.textContent = data.wrapped_prompt;
        } catch (e) {
            els.wrapped.textContent = `wrap failed: ${e.message ?? e}`;
        }
    }, 250);
}

// --- generation -------------------------------------------------------------

function setBusy(busy) {
    state.busy = busy;
    els.btnBanana.disabled = busy;
    els.btnTrellis.disabled = busy || !state.imageId;
}

async function generateImage() {
    const body = requestBody();
    if (!body.phrase) {
        setStatus("phrase is empty.", "err");
        return;
    }
    if (els.useDims.checked && body.width === null) {
        setStatus("dimensions are on but not all valid (need positive W, H, D).", "err");
        return;
    }
    setBusy(true);
    state.imageId = null;
    els.btnTrellis.disabled = true;
    paneMessage(els.imagePane, "generating…", { spinner: true });
    paneStatus(els.imagePane, "nano banana…");
    setStatus("generating image with Nano Banana…");
    const t0 = performance.now();
    try {
        const r = await fetch("/banana", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ ...body, name: state.selectedName }),
        });
        if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
        const data = await r.json();
        state.imageId = data.image_id;
        els.wrapped.textContent = data.wrapped_prompt;
        showImage(data.image_url);
        const dt = ((performance.now() - t0) / 1000).toFixed(1);
        setStatus(`image ready (${dt}s) — generate 3D when ready.`, "ok");
        paneStatus(els.imagePane, "done", "ok");
    } catch (e) {
        paneMessage(els.imagePane, "failed", { cls: "err" });
        paneStatus(els.imagePane, "error", "err");
        setStatus(`image generation failed: ${e.message ?? e}`, "err");
    } finally {
        setBusy(false);
    }
}

function showImage(url) {
    clearPane(els.imagePane);
    const img = document.createElement("img");
    img.src = url;
    els.imagePane.appendChild(img);
    const link = document.createElement("a");
    link.className = "open";
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = "open ↗";
    els.imagePane.appendChild(link);
}

async function generateModel() {
    if (!state.imageId) {
        setStatus("generate an image first.", "err");
        return;
    }
    setBusy(true);
    paneMessage(els.modelPane, "generating… (Trellis can take ~30–90s)", { spinner: true });
    paneStatus(els.modelPane, "trellis…");
    setStatus("generating 3D mesh with Trellis…");
    const t0 = performance.now();
    try {
        const r = await fetch("/trellis", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ image_id: state.imageId }),
        });
        if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
        const data = await r.json();
        const dt = ((performance.now() - t0) / 1000).toFixed(1);
        await loadModel(data.glb_url);
        setStatus(`3D mesh ready (${dt}s).`, "ok");
        paneStatus(els.modelPane, "done", "ok");
    } catch (e) {
        paneMessage(els.modelPane, "failed", { cls: "err" });
        paneStatus(els.modelPane, "error", "err");
        setStatus(`3D generation failed: ${e.message ?? e}`, "err");
    } finally {
        setBusy(false);
    }
}

// --- three.js viewer --------------------------------------------------------

let viewer = null;

function ensureViewer() {
    if (viewer) return viewer;
    const pane = els.modelPane;
    clearPane(pane);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    pane.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c0d10);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x202028, 1.0));
    const dir = new THREE.DirectionalLight(0xffffff, 1.1);
    dir.position.set(3, 5, 4);
    scene.add(dir);
    const dir2 = new THREE.DirectionalLight(0xffffff, 0.5);
    dir2.position.set(-3, 2, -2);
    scene.add(dir2);

    const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
    camera.position.set(2, 1.5, 2.5);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.screenSpacePanning = true;
    controls.zoomToCursor = true;
    controls.mouseButtons = {
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.PAN,
        RIGHT: THREE.MOUSE.PAN,
    };
    renderer.domElement.addEventListener("contextmenu", (e) => e.preventDefault());

    const fitSize = () => {
        const w = pane.clientWidth;
        const h = pane.clientHeight;
        if (w === 0 || h === 0) return;
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    };
    fitSize();
    new ResizeObserver(fitSize).observe(pane);

    const v = { renderer, scene, camera, controls, pane, model: null, fitSize, frame: null };
    pane.tabIndex = 0;
    pane.addEventListener("keydown", (e) => {
        if ((e.key === "f" || e.key === "F") && v.frame) v.frame();
    });
    pane.addEventListener("pointerenter", () => pane.focus({ preventScroll: true }));

    renderer.setAnimationLoop(() => {
        controls.update();
        renderer.render(scene, camera);
    });
    viewer = v;
    return v;
}

function loadModel(url) {
    const v = ensureViewer();
    v.fitSize();
    if (v.model) {
        v.scene.remove(v.model);
        v.model = null;
    }
    return new Promise((resolve, reject) => {
        new GLTFLoader().load(
            url,
            (gltf) => {
                const root = gltf.scene;
                const box = new THREE.Box3().setFromObject(root);
                const size = box.getSize(new THREE.Vector3());
                const center = box.getCenter(new THREE.Vector3());
                root.position.sub(center);
                v.scene.add(root);
                v.model = root;
                const maxDim = Math.max(size.x, size.y, size.z) || 1;
                v.frame = () => {
                    const dist = maxDim * 2.2;
                    v.camera.position.set(dist * 0.7, dist * 0.5, dist * 0.9);
                    v.camera.near = maxDim * 0.01;
                    v.camera.far = maxDim * 100;
                    v.camera.updateProjectionMatrix();
                    v.controls.target.set(0, 0, 0);
                    v.controls.update();
                };
                v.frame();
                resolve();
            },
            undefined,
            (err) => reject(new Error(err?.message ?? "GLB load failed")),
        );
    });
}

// --- wiring -----------------------------------------------------------------

els.scene.addEventListener("change", renderList);
els.model.addEventListener("change", renderList);
els.search.addEventListener("input", renderList);

els.phrase.addEventListener("input", refreshWrap);
els.proxy.addEventListener("change", refreshWrap);
els.view.addEventListener("change", refreshWrap);
for (const el of [els.dimW, els.dimH, els.dimD]) {
    el.addEventListener("input", refreshWrap);
}
els.useDims.addEventListener("change", () => {
    const on = els.useDims.checked;
    for (const el of [els.dimW, els.dimH, els.dimD]) el.disabled = !on;
    refreshWrap();
});

els.btnBanana.addEventListener("click", generateImage);
els.btnTrellis.addEventListener("click", generateModel);

loadPhrases();
