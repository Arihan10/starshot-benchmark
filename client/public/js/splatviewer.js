// Full-screen Gaussian-splat viewer for Stage-2 clouds, on @mkkellogg/gaussian-
// splats-3d (orbit controls + WASM sort).
//
// LOD streaming: lazy-loads the ~10 MB base cloud (cloud.ply) for a fast first
// paint, then streams the denser detail LOD (cloud.detail.ply) in the background
// and swaps it in (removeSplatScene(base)) once it's ready.
//
// Live controls: a panel exposes the sampler's global knobs (density / overlap /
// flatness / adaptive / detail) and re-splats the open cell — POST the knobs,
// poll progress, reload in place (camera preserved) — so tuning needs no server
// restart and no manual command.

import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { api } from "./api.js";
import { el } from "./ui.js";

let overlay = null;
let canvasEl = null;
let subEl = null;
let statusEl = null;
let controlsEl = null;
let viewer = null;
let openSeq = 0; // bumps on open/close to cancel in-flight loads/polls
let pollTimer = null;
let current = null; // { run, slot, model, source }
let inputs = null; // control input elements, by key
let voxelPoints = null; // Stage-3 free-space overlay (THREE.Points), lazy
let meshGroup = null; // original-mesh overlay (THREE.Group), lazy
let mode = "splat"; // "splat" | "mesh" — the view switch (not an overlay)

// WASD free-fly state (see movementTick). mkkellogg owns mouse orbit/zoom; this
// adds keyboard walk by translating the camera + orbit target together per frame.
const MOVE_CODES = new Set([
    "KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE",
    "Space", "ShiftLeft", "ShiftRight",
]);
const pressed = new Set();
let moveRaf = 0;
let moveLast = 0;
let moveSpeed = 3; // metres/sec, sized per scene from its AABB on load

const POLL_MS = 1200;
const BYTES_PER_SPLAT = 68; // 17 float32 — for the density→size estimate

function setStatus(text, color) {
    if (!statusEl) return;
    statusEl.textContent = text || "";
    statusEl.style.color = color || "";
}

function stopPoll() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

// A 3/4 exterior view framed on the scene AABB (fallback pose when no AABB).
function framing(aabb) {
    if (!aabb || !aabb.min || !aabb.max) {
        return { position: [4, 3, 6], lookAt: [0, 1, 0] };
    }
    const { min, max } = aabb;
    const c = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
    const size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    const r = Math.max(0.5, Math.hypot(size[0], size[1], size[2]) / 2);
    const d = r * 1.5;
    return { position: [c[0] + d, c[1] + d * 0.6, c[2] + d], lookAt: c };
}

// ---- WASD free-fly ----------------------------------------------------------

// Base fly speed scaled to the scene: ~a third of its diagonal per second (Shift
// sprints 3×), clamped so small props aren't glacial and big scenes aren't wild.
function speedFor(aabb) {
    if (!aabb || !aabb.min || !aabb.max) return 3;
    const d = Math.hypot(
        aabb.max[0] - aabb.min[0],
        aabb.max[1] - aabb.min[1],
        aabb.max[2] - aabb.min[2],
    );
    return Math.min(25, Math.max(2, d * 0.35));
}

const _fwd = new THREE.Vector3();
const _right = new THREE.Vector3();
const _move = new THREE.Vector3();
const _WORLD_UP = new THREE.Vector3(0, 1, 0);

function movementTick(now) {
    moveRaf = requestAnimationFrame(movementTick);
    const cam = viewer && viewer.camera;
    const controls = viewer && viewer.controls;
    const dt = Math.min(0.05, (now - moveLast) / 1000 || 0); // clamp big frame gaps
    moveLast = now;
    if (!cam || !controls || pressed.size === 0) return;
    cam.updateMatrixWorld();
    // Horizontal forward/right (drop the pitch → walk, don't dive); Q/E/Space are
    // world-vertical. This keeps W from drifting up/down when the view is tilted.
    cam.getWorldDirection(_fwd);
    _fwd.y = 0;
    if (_fwd.lengthSq() < 1e-8) {
        // looking straight up/down → use the screen-up direction as forward
        _fwd.setFromMatrixColumn(cam.matrixWorld, 1).setY(0);
    }
    _fwd.normalize();
    _right.setFromMatrixColumn(cam.matrixWorld, 0).setY(0).normalize();
    _move.set(0, 0, 0);
    if (pressed.has("KeyW")) _move.add(_fwd);
    if (pressed.has("KeyS")) _move.sub(_fwd);
    if (pressed.has("KeyD")) _move.add(_right);
    if (pressed.has("KeyA")) _move.sub(_right);
    if (pressed.has("KeyE") || pressed.has("Space")) _move.add(_WORLD_UP);
    if (pressed.has("KeyQ")) _move.sub(_WORLD_UP);
    if (_move.lengthSq() === 0) return;
    const sprint = pressed.has("ShiftLeft") || pressed.has("ShiftRight") ? 3 : 1;
    _move.normalize().multiplyScalar(moveSpeed * sprint * dt);
    cam.position.add(_move);
    controls.target.add(_move); // move the orbit target too → look direction preserved
}

function startMovement() {
    if (moveRaf) return;
    moveLast = performance.now();
    moveRaf = requestAnimationFrame(movementTick);
}

function stopMovement() {
    if (moveRaf) cancelAnimationFrame(moveRaf);
    moveRaf = 0;
    pressed.clear();
}

// Read the live camera so a re-splat reload can keep the same view.
function captureCamera() {
    try {
        const p = viewer.camera.position;
        const t = viewer.controls.target;
        return { position: [p.x, p.y, p.z], lookAt: [t.x, t.y, t.z] };
    } catch {
        return null;
    }
}

function disposeObj(obj) {
    obj?.traverse?.((o) => {
        o.geometry?.dispose?.();
        const mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
        for (const m of mats) {
            for (const k in m) {
                const v = m[k];
                if (v && v.isTexture) v.dispose?.();
            }
            m.dispose?.();
        }
    });
}

async function teardown() {
    stopMovement();
    // Overlays live in the viewer's threeScene; free their GPU resources and
    // reset their toggles before the viewer (and its GL context) goes away.
    for (const key of ["voxelPoints", "meshGroup"]) {
        const obj = key === "voxelPoints" ? voxelPoints : meshGroup;
        if (obj) {
            viewer?.threeScene?.remove(obj);
            disposeObj(obj);
        }
    }
    voxelPoints = null;
    meshGroup = null;
    // A rebuild (re-splat) returns to splat mode; the splat is visible by default
    // on the new viewer, and the overlays are gone.
    mode = "splat";
    syncModeButtons();
    if (inputs && inputs._fsRow) inputs._fsRow.style.display = "none";
    const v = viewer;
    viewer = null;
    if (v) {
        try {
            await v.dispose();
        } catch {
            /* disposing a partially-built viewer can throw; ignore */
        }
    }
    if (canvasEl) canvasEl.replaceChildren();
}

// Build a fresh viewer, load the base cloud, then stream the detail LOD behind it.
async function loadClouds(seq, url, detailUrl, summary, camera) {
    await teardown();
    if (seq !== openSeq) return;
    const view = camera || framing(summary && summary.scene_aabb);
    const v = new GaussianSplats3D.Viewer({
        rootElement: canvasEl,
        selfDrivenMode: true,
        useBuiltInControls: true,
        sharedMemoryForWorkers: false, // no COOP/COEP needed on the static server
        dynamicScene: false,
        sphericalHarmonicsDegree: 0, // our .ply is f_dc only (unlit)
        cameraUp: [0, 1, 0],
        initialCameraPosition: view.position,
        initialCameraLookAt: view.lookAt,
    });
    viewer = v;
    try {
        await v.addSplatScene(url, {
            showLoadingUI: true,
            splatAlphaRemovalThreshold: 1,
        });
    } catch (e) {
        if (seq === openSeq) setStatus(`failed: ${e && e.message ? e.message : e}`, "var(--red)");
        return;
    }
    if (seq !== openSeq) return;
    v.start();
    moveSpeed = speedFor(summary && summary.scene_aabb);
    startMovement();
    const baseN = summary && summary.splats;
    setStatus(
        (baseN ? `${baseN.toLocaleString()} splats` : "loaded") +
            (detailUrl ? " · streaming detail…" : ""),
        "var(--green)",
    );
    if (!detailUrl) return;
    // Stream the denser LOD in the background; when it lands, drop the base scene.
    v.addSplatScene(detailUrl, { showLoadingUI: false, splatAlphaRemovalThreshold: 1 })
        .then(() => {
            if (seq !== openSeq || viewer !== v) return;
            try {
                if (v.getSceneCount() > 1) v.removeSplatScene(0);
            } catch {
                /* keep both if removal fails — just a touch heavier */
            }
            const dN = summary && summary.detail && summary.detail.splats;
            setStatus(
                dN ? `${dN.toLocaleString()} splats (detail LOD)` : "detail LOD loaded",
                "var(--green)",
            );
        })
        .catch(() => {
            if (seq === openSeq) setStatus("detail LOD failed — showing base", "");
        });
}

// ---- controls panel ---------------------------------------------------------

function sliderRow(key, label, min, max, step, value, fmt) {
    const val = el("span", { class: "svc-val", text: fmt(value) });
    const input = el("input", {
        type: "range",
        class: "svc-range",
        min,
        max,
        step,
        value,
    });
    input.addEventListener("input", () => {
        val.textContent = fmt(Number(input.value));
    });
    inputs[key] = input;
    return el(
        "div",
        { class: "svc-row" },
        el("span", { class: "svc-lab", text: label }),
        input,
        val,
    );
}

function checkRow(key, label, checked) {
    const input = el("input", { type: "checkbox" });
    input.checked = !!checked;
    inputs[key] = input;
    return el("label", { class: "svc-check" }, input, el("span", { text: label }));
}

// ---- overlays (Stage-3 free space + original mesh) --------------------------

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function setOverlay(text) {
    if (inputs && inputs._overlay) inputs._overlay.textContent = text || "";
}

// Clearance → colour: 0 (at a surface) warm red, 1 (most open) cool blue.
function clearanceColor(t) {
    const x = Math.max(0, Math.min(1, t));
    return [1 - 0.8 * x, 0.25 + 0.45 * x, 0.2 + 0.8 * x];
}

// Build the free-voxel point cloud from voxels.bin ([x,y,z,clearance] float32).
async function buildVoxels(url, summary) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const arr = new Float32Array(await res.arrayBuffer());
    const n = (arr.length / 4) | 0;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const cmax = (summary && summary.clearance_max) || 1;
    for (let i = 0; i < n; i++) {
        pos[i * 3] = arr[i * 4];
        pos[i * 3 + 1] = arr[i * 4 + 1];
        pos[i * 3 + 2] = arr[i * 4 + 2];
        const [r, g, b] = clearanceColor(arr[i * 4 + 3] / cmax);
        col[i * 3] = r;
        col[i * 3 + 1] = g;
        col[i * 3 + 2] = b;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    const mat = new THREE.PointsMaterial({
        size: ((summary && summary.pitch) || 0.12) * 0.5,
        vertexColors: true,
        transparent: true,
        opacity: 0.5,
        sizeAttenuation: true,
        depthWrite: false,
        // X-ray: the free-space field is a VOLUMETRIC overlay filling the room
        // interior, so it must draw THROUGH the (opaque) wall splats — otherwise
        // the interior cloud is hidden behind the walls from an outside camera,
        // which is exactly the "I see no free space inside" symptom.
        depthTest: false,
    });
    const pts = new THREE.Points(geo, mat);
    pts.renderOrder = 999; // composite the overlay last, over the splats
    return pts;
}

// Ensure Stage 3 has run for the open cell; compute + poll if not. Returns status.
async function ensureStage3() {
    const c = current;
    let st = await api.splatStage3Status(c.run, c.slot, c.model);
    if (st.status === "done" && st.url) return st;
    setOverlay("computing free space…");
    await api.splatStage3Start(c.run, c.slot, c.model, {});
    const seq = openSeq;
    while (seq === openSeq) {
        await sleep(1000);
        st = await api.splatStage3Status(c.run, c.slot, c.model);
        if (!st.running) break;
        setOverlay(`free space: ${st.phase || ""} ${st.total ? `${st.done}/${st.total}` : ""}`);
    }
    if (st.status !== "done" || !st.url) throw new Error(st.error || "voxelize failed");
    return st;
}

function syncModeButtons() {
    if (!inputs) return;
    inputs._splatBtn?.classList.toggle("on", mode === "splat");
    inputs._meshBtn?.classList.toggle("on", mode === "mesh");
}

function splatVisible(on) {
    const sm = viewer && viewer.getSplatMesh && viewer.getSplatMesh();
    if (sm) sm.visible = on;
}

// Lazily build + add the free-space points (added hidden); returns success.
async function ensureVoxels() {
    if (voxelPoints) return true;
    const seq = openSeq;
    try {
        const st = await ensureStage3();
        if (seq !== openSeq || !viewer) return false;
        const pts = await buildVoxels(api.absUrl(st.url + `?t=${Date.now()}`), st.summary);
        if (seq !== openSeq || !viewer) {
            disposeObj(pts);
            return false;
        }
        voxelPoints = pts;
        voxelPoints.visible = false;
        viewer.threeScene.add(voxelPoints);
        const fv = (st.summary && st.summary.free_voxels) || 0;
        setOverlay(`free space: ${fv.toLocaleString()} voxels`);
        return true;
    } catch (e) {
        setOverlay(`free space failed: ${e.message}`);
        return false;
    }
}

// Free-space voxels are only shown alongside the mesh (they share its exact world
// frame), via the checkbox that appears in mesh mode.
async function setFreeSpace(on) {
    if (mode !== "mesh") return;
    if (on) {
        const ok = await ensureVoxels();
        if (ok && voxelPoints && mode === "mesh") voxelPoints.visible = true;
    } else if (voxelPoints) {
        voxelPoints.visible = false;
    }
}

// Load the cell's original mesh (SMB1 bundle) into the splat scene, coincident
// with the cloud, for a splat-vs-mesh look comparison.
async function buildMeshGroup() {
    const res = await fetch(
        api.meshesUrl(current.run, current.slot, current.model, {}),
        { cache: "no-store" },
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const ab = await res.arrayBuffer();
    const buf = new Uint8Array(ab);
    const dv = new DataView(ab);
    const dec = new TextDecoder();
    if (buf.length < 4 || dec.decode(buf.subarray(0, 4)) !== "SMB1") {
        throw new Error("not an SMB1 bundle");
    }
    const ktx2 = new KTX2Loader()
        .setTranscoderPath("/vendor/three/examples/jsm/libs/basis/")
        .detectSupport(viewer.renderer);
    MeshoptDecoder.useWorkers?.(2);
    const loader = new GLTFLoader().setKTX2Loader(ktx2).setMeshoptDecoder(MeshoptDecoder);
    const group = new THREE.Group();
    let off = 4; // past the "SMB1" magic
    while (off + 4 <= buf.length) {
        const idLen = dv.getUint32(off, true);
        off += 4 + idLen; // skip the id string
        if (off + 4 > buf.length) break;
        const glbLen = dv.getUint32(off, true);
        off += 4;
        if (off + glbLen > buf.length) break;
        const glb = ab.slice(off, off + glbLen);
        off += glbLen;
        try {
            const gltf = await loader.parseAsync(glb, "");
            group.add(gltf.scene);
        } catch {
            /* skip a bad object */
        }
    }
    ktx2.dispose?.();
    return group.children.length ? group : null;
}

// Lazily build + add the original mesh (added hidden); returns success.
async function ensureMesh() {
    if (meshGroup) return true;
    setOverlay("loading original mesh…");
    const seq = openSeq;
    try {
        const g = await buildMeshGroup();
        if (seq !== openSeq || !viewer) {
            disposeObj(g);
            return false;
        }
        if (!g) throw new Error("no mesh bundle");
        meshGroup = g;
        meshGroup.visible = false;
        viewer.threeScene.add(meshGroup);
        setOverlay("");
        return true;
    } catch (e) {
        setOverlay(`mesh failed: ${e.message}`);
        return false;
    }
}

// Switch the view between the Gaussian splat and the original mesh (same world
// frame → identical pose). Splat mode hides the mesh + free space; mesh mode
// hides the splat, shows the mesh, and turns the free-space field on by default.
async function setMode(next) {
    mode = next;
    syncModeButtons();
    if (inputs && inputs._fsRow) {
        inputs._fsRow.style.display = next === "mesh" ? "" : "none";
    }
    if (next === "splat") {
        splatVisible(true);
        if (meshGroup) meshGroup.visible = false;
        if (voxelPoints) voxelPoints.visible = false;
        setOverlay("");
        return;
    }
    // mesh mode: hide the splat, show the mesh, free space on by default
    splatVisible(false);
    const ok = await ensureMesh();
    if (mode !== "mesh") return; // switched back while the mesh was loading
    if (ok && meshGroup) meshGroup.visible = true;
    if (inputs && inputs.freespace && inputs.freespace.checked) {
        await setFreeSpace(true);
    }
}

const kFmt = (v) => `${Math.round(v / 1000)}k · ~${((v * BYTES_PER_SPLAT) / 1e6).toFixed(0)}MB`;

function buildControls(summary) {
    if (!controlsEl) return;
    const p = (summary && summary.params) || {};
    const detail = summary && summary.detail;
    inputs = {};
    controlsEl.replaceChildren();

    const density = sliderRow(
        "density", "density", 20000, 1200000, 10000,
        p.target_splats || 150000, kFmt,
    );
    const radius = sliderRow(
        "radius", "overlap", 0.5, 1.6, 0.05,
        p.radius_frac != null ? p.radius_frac : 0.9, (v) => v.toFixed(2),
    );
    const flat = sliderRow(
        "flatness", "flatness", 0.02, 0.4, 0.01,
        p.flatness != null ? p.flatness : 0.1, (v) => v.toFixed(2),
    );
    const adaptive = checkRow("adaptive", "adaptive density", p.adaptive !== false);
    const detailOn = checkRow("detail", "stream detail LOD", !!detail);
    const detailDensity = sliderRow(
        "detailDensity", "detail", 100000, 2000000, 50000,
        (detail && detail.splats) || 600000, kFmt,
    );

    const btn = el("button", {
        class: "svc-resplat",
        text: "re-splat",
        onclick: () => resplat(),
    });
    inputs._btn = btn;
    const actual = el("div", { class: "svc-actual" });
    inputs._actual = actual;
    if (summary && summary.splats) {
        actual.textContent = actualText(summary);
    }

    // View mode: Gaussian splat vs. the original mesh — a switch, not an overlay.
    // Free space is drawn WITH the mesh (same world frame), on by default there.
    mode = "splat";
    const splatBtn = el("button", {
        class: "svc-seg-btn on",
        text: "splat",
        onclick: () => setMode("splat"),
    });
    const meshBtn = el("button", {
        class: "svc-seg-btn",
        text: "mesh",
        onclick: () => setMode("mesh"),
    });
    inputs._splatBtn = splatBtn;
    inputs._meshBtn = meshBtn;
    const seg = el("div", { class: "svc-seg" }, splatBtn, meshBtn);
    const fsRow = checkRow("freespace", "free space", true);
    inputs.freespace.addEventListener("change", () =>
        setFreeSpace(inputs.freespace.checked),
    );
    inputs._fsRow = fsRow;
    fsRow.style.display = "none"; // shown only in mesh mode
    const overlay = el("div", { class: "svc-actual" });
    inputs._overlay = overlay;

    controlsEl.append(
        el("div", { class: "svc-title", text: "view" }),
        seg,
        fsRow,
        el("div", { class: "svc-title", text: "sampler knobs" }),
        density,
        radius,
        flat,
        adaptive,
        detailOn,
        detailDensity,
        el("div", { class: "svc-actions" }, btn),
        actual,
        overlay,
    );
}

function actualText(summary) {
    const mb = (summary.bytes / 1e6).toFixed(1);
    let t = `actual: ${summary.splats.toLocaleString()} · ${mb} MB`;
    if (summary.detail) {
        t += ` (detail ${summary.detail.splats.toLocaleString()} · ${(summary.detail.bytes / 1e6).toFixed(0)} MB)`;
    }
    return t;
}

function readParams() {
    const body = {
        target_splats: Math.round(Number(inputs.density.value)),
        radius_frac: Number(inputs.radius.value),
        flatness: Number(inputs.flatness.value),
        adaptive: inputs.adaptive.checked,
    };
    if (inputs.detail.checked) {
        body.detail_splats = Math.round(Number(inputs.detailDensity.value));
    }
    return body;
}

async function resplat() {
    if (!current) return;
    const body = readParams();
    if (inputs._btn) inputs._btn.disabled = true;
    setStatus("re-splatting…", "var(--purple)");
    try {
        await api.splatStage2Start(current.run, current.slot, current.model, body);
    } catch (e) {
        setStatus(`re-splat failed: ${e.message}`, "var(--red)");
        if (inputs._btn) inputs._btn.disabled = false;
        return;
    }
    pollResplat();
}

function pollResplat() {
    stopPoll();
    const seq = openSeq;
    pollTimer = setInterval(async () => {
        if (seq !== openSeq || !current) return stopPoll();
        let st;
        try {
            st = await api.splatStage2Status(current.run, current.slot, current.model);
        } catch {
            return; // transient — retry next tick
        }
        if (seq !== openSeq) return stopPoll();
        if (st.running) {
            const phase = st.phase || "sampling";
            const prog = st.total ? ` ${st.done}/${st.total}` : "";
            setStatus(`${phase}${prog}…`, "var(--purple)");
            return;
        }
        stopPoll();
        if (inputs && inputs._btn) inputs._btn.disabled = false;
        if (st.status === "error") {
            setStatus(`re-splat failed: ${st.error || ""}`, "var(--red)");
            return;
        }
        if (st.status === "done" && st.url) {
            if (inputs && inputs._actual && st.summary) {
                inputs._actual.textContent = actualText(st.summary);
            }
            const bust = `?t=${Date.now()}`;
            loadClouds(
                openSeq,
                api.absUrl(st.url + bust),
                st.detail_url ? api.absUrl(st.detail_url + bust) : null,
                st.summary,
                captureCamera(), // keep the current view across the re-splat
            );
        }
    }, POLL_MS);
}

// ---- open / close -----------------------------------------------------------

export async function openSplatViewer(opts) {
    if (!overlay || !opts || !opts.url) return;
    const seq = ++openSeq;
    stopPoll();
    current = {
        run: opts.run,
        slot: opts.slot,
        model: opts.model,
        source: opts.source,
    };
    overlay.classList.add("open");
    subEl.textContent = opts.label || "";
    setStatus("loading…", "var(--purple)");
    buildControls(opts.summary);
    await loadClouds(seq, opts.url, opts.detailUrl || null, opts.summary, null);
}

export function isSplatViewerOpen() {
    return !!overlay && overlay.classList.contains("open");
}

export function closeSplatViewer() {
    if (!isSplatViewerOpen()) return;
    openSeq++; // cancel any in-flight load/poll
    stopPoll();
    overlay.classList.remove("open");
    setStatus("");
    subEl.textContent = "";
    current = null;
    void teardown();
}

export function initSplatViewer() {
    overlay = document.getElementById("splat-viewer");
    canvasEl = document.getElementById("splat-viewer-canvas");
    subEl = document.getElementById("splat-viewer-sub");
    statusEl = document.getElementById("splat-viewer-status");
    controlsEl = document.getElementById("splat-viewer-controls");
    document
        .getElementById("splat-viewer-close")
        .addEventListener("click", closeSplatViewer);
    // Capture-phase Esc: close the viewer (topmost) and swallow the event so the
    // splat screen's own Esc handler underneath doesn't also close.
    document.addEventListener(
        "keydown",
        (ev) => {
            if (ev.key === "Escape" && isSplatViewerOpen()) {
                ev.stopImmediatePropagation();
                closeSplatViewer();
            }
        },
        true,
    );
    // WASD/QE/Shift walk — only while the viewer is open, and not while typing in
    // the controls panel. keyup always clears so a key can't stick.
    window.addEventListener("keydown", (ev) => {
        if (!isSplatViewerOpen() || !MOVE_CODES.has(ev.code)) return;
        const tag = ev.target && ev.target.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        pressed.add(ev.code);
        if (ev.code === "Space") ev.preventDefault(); // don't scroll the page
    });
    window.addEventListener("keyup", (ev) => pressed.delete(ev.code));
    window.addEventListener("blur", () => pressed.clear());
}
