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
let voxelPoints = null; // Stage-2 free-space overlay (THREE.Points), lazy
let meshGroup = null; // original-mesh overlay (THREE.Group), lazy
let mode = "splat"; // "splat" | "mesh" — the view switch (not an overlay)

// Patch-selection debug feature (needs Stage 4 + Stage 5). A selectable points
// overlay of the coverage patches; clicking one opens its Stage-5 reference images
// in a right-side modal. Lazy-loaded when the "patches" toggle is first enabled.
let patchPoints = null; // THREE.Points of patch centres (the raycast target)
let patchViews = null; // per-patch [[camera_index, face_index], …]
let patchFaces = null; // face-name lookup (index → "+x" …)
let refsBase = null; // /artifacts base URL of the Stage-5 refs/ dir, or null
let patchesOn = false;
let patchModalEl = null; // the right-side image modal (created lazily)
let _downXY = null; // pointer-down pos, to tell a click from an orbit drag

// Pipeline stepper state (the side panel drives the whole splat pipeline).
let cellStatus = null; // last-fetched per-stage status of the open cell
let cloudLoaded = false; // whether a surfel cloud is currently in the canvas

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
    for (const obj of [voxelPoints, meshGroup, patchPoints]) {
        if (obj) {
            viewer?.threeScene?.remove(obj);
            disposeObj(obj);
        }
    }
    voxelPoints = null;
    meshGroup = null;
    patchPoints = null;
    patchViews = null;
    patchFaces = null;
    refsBase = null;
    patchesOn = false;
    hidePatchModal();
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
            // Set the format explicitly: the cache-buster query (`?t=…`) defeats
            // mkkellogg's extension sniffing ("File format not supported").
            format: GaussianSplats3D.SceneFormat.Ply,
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
    v.addSplatScene(detailUrl, {
        showLoadingUI: false,
        splatAlphaRemovalThreshold: 1,
        format: GaussianSplats3D.SceneFormat.Ply,
    })
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

// Ensure Stage 2 (free-space) has run for the open cell; compute + poll if not.
async function ensureFreeSpace() {
    const c = current;
    let st = await api.splatStage2Status(c.run, c.slot, c.model);
    if (st.status === "done" && st.url) return st;
    setOverlay("computing free space…");
    await api.splatStage2Start(c.run, c.slot, c.model, {});
    const seq = openSeq;
    while (seq === openSeq) {
        await sleep(1000);
        st = await api.splatStage2Status(c.run, c.slot, c.model);
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
        const st = await ensureFreeSpace();
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
    // Patch inspector (needs Stage 4 + Stage 5): select a coverage patch on the
    // splat and open its reference images in the right-side modal.
    const patchRow = checkRow("patches", "patches (click to inspect)", false);
    inputs.patches.addEventListener("change", () => setPatches(inputs.patches.checked));
    const overlay = el("div", { class: "svc-actual" });
    inputs._overlay = overlay;
    inputs._stepper = el("div", { class: "svc-stepper" });
    inputs._coverage = el("div", { class: "svc-coverage" });

    controlsEl.append(
        el("div", { class: "svc-title", text: "pipeline" }),
        inputs._stepper,
        inputs._coverage,
        el("div", { class: "svc-title", text: "view" }),
        seg,
        fsRow,
        patchRow,
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
        await api.splatStage3Start(current.run, current.slot, current.model, body);
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
            st = await api.splatStage3Status(current.run, current.slot, current.model);
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

// ---- patch selection (Stage-4 patches → Stage-5 reference images) -----------

const pad5 = (n) => String(n).padStart(5, "0");
const _pv = new THREE.Vector3(); // scratch for screen-space patch picking

// A soft round sprite so patches render as DISKS, not the default square points.
let _patchTex = null;
function patchSprite() {
    if (_patchTex) return _patchTex;
    const s = 64;
    const c = document.createElement("canvas");
    c.width = c.height = s;
    const ctx = c.getContext("2d");
    ctx.beginPath();
    ctx.arc(s / 2, s / 2, s / 2 - 3, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    _patchTex = new THREE.CanvasTexture(c);
    return _patchTex;
}

// Selectable patch points from patches.bin ([x,y,z, nx,ny,nz, feature_scale,
// sectors_seen] float32 × N). Coloured by distinct angles seen (0 = occlusion-
// culled red → covered green), so coverage gaps are visible at a glance.
function buildPatches(buf) {
    const arr = new Float32Array(buf);
    const n = (arr.length / 8) | 0;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
        pos[i * 3] = arr[i * 8];
        pos[i * 3 + 1] = arr[i * 8 + 1];
        pos[i * 3 + 2] = arr[i * 8 + 2];
        const t = Math.max(0, Math.min(1, arr[i * 8 + 7] / 6)); // sectors_seen / ~K
        col[i * 3] = 1 - t;
        col[i * 3 + 1] = 0.25 + 0.65 * t;
        col[i * 3 + 2] = 0.2;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    const mat = new THREE.PointsMaterial({
        size: 0.09, // world-space disk size (bigger than before so patches are pickable)
        vertexColors: true,
        sizeAttenuation: true,
        map: patchSprite(),
        alphaTest: 0.5,
        transparent: true,
        // X-ray so every patch is visible + selectable through the (opaque) splat;
        // the picker resolves the front-most patch under the cursor.
        depthTest: false,
        depthWrite: false,
    });
    const pts = new THREE.Points(geo, mat);
    pts.renderOrder = 1000;
    return pts;
}

// Fetch Stage-4 (patches + patch→views index) and Stage-5 (refs base) for the open
// cell and build the overlay. Returns true on success.
async function ensurePatches() {
    if (patchPoints) return true;
    const c = current;
    setOverlay("loading patches…");
    const s4 = await api.splatStage4Status(c.run, c.slot, c.model);
    if (s4.status !== "done" || !s4.patches_url || !s4.patch_views_url) {
        setOverlay("plan cameras first (Stage 4)");
        return false;
    }
    const seq = openSeq;
    const [pbuf, pv] = await Promise.all([
        fetch(api.absUrl(s4.patches_url + `?t=${Date.now()}`), { cache: "no-store" }).then((r) => r.arrayBuffer()),
        fetch(api.absUrl(s4.patch_views_url + `?t=${Date.now()}`), { cache: "no-store" }).then((r) => r.json()),
    ]);
    if (seq !== openSeq || !viewer) return false;
    patchViews = pv.views || [];
    patchFaces = pv.faces || ["+x", "-x", "+y", "-y", "+z", "-z"];
    patchPoints = buildPatches(pbuf);
    patchPoints.visible = false;
    viewer.threeScene.add(patchPoints);
    // Stage-5 refs for the images (optional — modal says so if not rendered yet).
    const s5 = await api.splatStage5Status(c.run, c.slot, c.model);
    refsBase =
        s5.status === "done" && s5.url
            ? api.absUrl(s5.url.replace(/transforms\.json$/, ""))
            : null;
    setOverlay(`patches: ${(patchViews.length || 0).toLocaleString()} (click one)`);
    return true;
}

async function setPatches(on) {
    patchesOn = on;
    if (on) {
        const ok = await ensurePatches();
        if (ok && patchPoints) patchPoints.visible = true;
        else patchesOn = false;
    } else {
        if (patchPoints) patchPoints.visible = false;
        hidePatchModal();
    }
}

// Pick the front-most patch under the cursor in SCREEN space (project every patch
// to pixels, take the nearest within a small radius, closest to camera). This is
// robust where a Points raycaster's world-space threshold is finicky.
function pickPatch(ev) {
    if (!patchesOn || !patchPoints || !viewer) return -1;
    const rect = canvasEl.getBoundingClientRect();
    const cx = ev.clientX - rect.left;
    const cy = ev.clientY - rect.top;
    const cam = viewer.camera;
    cam.updateMatrixWorld();
    const pos = patchPoints.geometry.getAttribute("position");
    const r2 = 16 * 16; // pick radius² in pixels
    let best = -1;
    let bestZ = Infinity;
    for (let i = 0; i < pos.count; i++) {
        _pv.fromBufferAttribute(pos, i).project(cam);
        if (_pv.z < -1 || _pv.z > 1) continue; // behind camera / clipped
        const sx = (_pv.x * 0.5 + 0.5) * rect.width;
        const sy = (-_pv.y * 0.5 + 0.5) * rect.height;
        const d2 = (sx - cx) ** 2 + (sy - cy) ** 2;
        if (d2 <= r2 && _pv.z < bestZ) {
            bestZ = _pv.z;
            best = i;
        }
    }
    return best;
}

function hidePatchModal() {
    if (patchModalEl) patchModalEl.style.display = "none";
}

// Right-side modal: every Stage-5 reference image that sees patch `index`.
function showPatchModal(index) {
    if (!patchModalEl) return;
    const views = (patchViews && patchViews[index]) || [];
    patchModalEl.replaceChildren();

    const head = el("div");
    Object.assign(head.style, {
        display: "flex", justifyContent: "space-between",
        alignItems: "center", marginBottom: "8px",
    });
    head.appendChild(
        el("span", { text: `patch #${index} · ${views.length} view${views.length === 1 ? "" : "s"}` }),
    );
    const x = el("button", { text: "✕", onclick: hidePatchModal });
    Object.assign(x.style, {
        background: "transparent", color: "#fff", border: "none",
        cursor: "pointer", fontSize: "15px", lineHeight: "1",
    });
    head.appendChild(x);
    patchModalEl.appendChild(head);

    if (!views.length) {
        patchModalEl.appendChild(el("div", { class: "muted", text: "no camera covers this patch (occlusion-culled)" }));
    } else if (!refsBase) {
        patchModalEl.appendChild(el("div", { class: "muted", text: "render references (Stage 5) to see images" }));
    } else {
        for (const [cam, faceIdx] of views) {
            const id = `cam${pad5(cam)}_${patchFaces[faceIdx]}`;
            const fig = el("figure");
            Object.assign(fig.style, { margin: "0 0 10px" });
            const img = el("img");
            img.src = `${refsBase}rgb/${id}.png`;
            img.loading = "lazy";
            img.alt = id;
            Object.assign(img.style, {
                width: "100%", display: "block", borderRadius: "4px", background: "#000",
            });
            const cap = el("figcaption", { text: id });
            Object.assign(cap.style, { fontSize: "11px", opacity: "0.7", marginTop: "2px" });
            fig.appendChild(img);
            fig.appendChild(cap);
            patchModalEl.appendChild(fig);
        }
    }
    patchModalEl.style.display = "block";
}

// ---- pipeline stepper (the side panel: run/re-run each stage, gated) ---------

// The 5 splat stages in dependency order. Stage 1's status is the cell itself;
// Stages 2–5 live on `cell.stageN`. A stage is runnable only once the previous is
// done; re-running a done stage REVERTS everything after it (server-side).
const STAGES = [
    { n: 1, label: "assemble", verb: "convert" },
    { n: 2, label: "free space", verb: "voxelize" },
    { n: 3, label: "surfels", verb: "sample" },
    { n: 4, label: "cameras", verb: "plan" },
    { n: 5, label: "references", verb: "render" },
];
const STAGE_START = {
    1: (r, s, m) => api.splatStage1Start(r, s, m),
    2: (r, s, m) => api.splatStage2Start(r, s, m),
    3: (r, s, m) => api.splatStage3Start(r, s, m),
    4: (r, s, m) => api.splatStage4Start(r, s, m),
    5: (r, s, m) => api.splatStage5Start(r, s, m),
};

function stageState(cell, n) {
    if (!cell) return {};
    return n === 1 ? cell : cell[`stage${n}`] || {};
}

function stageDone(cell, n) {
    return stageState(cell, n).status === "done";
}

function anyStageRunning(cell) {
    return STAGES.some((s) => {
        const st = stageState(cell, s.n).status;
        return st === "running" || st === "pending";
    });
}

async function runStage(n) {
    const c = current;
    if (!c) return;
    // Re-running at/before surfels reverts the cloud → drop it from the canvas.
    if (n <= 3) cloudLoaded = false;
    setStatus(`starting stage ${n}…`, "var(--purple)");
    try {
        await STAGE_START[n](c.run, c.slot, c.model);
    } catch (e) {
        setStatus(`stage ${n} failed: ${e.message}`, "var(--red)");
        return;
    }
    pollStages();
}

function renderStepper() {
    const box = inputs && inputs._stepper;
    if (!box) return;
    box.replaceChildren();
    const cell = cellStatus;
    for (const stage of STAGES) {
        const st = stageState(cell, stage.n);
        const done = st.status === "done";
        const running = st.status === "running" || st.status === "pending";
        const gated = stage.n > 1 && !stageDone(cell, stage.n - 1);
        const row = el("div", { class: "svc-step" });
        row.appendChild(el("span", { class: "svc-step-n muted", text: `${stage.n}` }));
        row.appendChild(el("span", { class: "svc-step-label", text: stage.label }));
        let btn;
        if (running) {
            const prog = st.total ? `${st.done}/${st.total}` : st.phase || "…";
            btn = el("button", { class: "splat-stage2-btn", disabled: true, text: prog });
        } else if (gated) {
            btn = el("button", {
                class: "splat-stage2-btn",
                disabled: true,
                text: stage.verb,
                title: `run stage ${stage.n - 1} first`,
            });
        } else {
            btn = el("button", {
                class: `splat-stage2-btn${done ? " view" : ""}`,
                text: done ? "re-run" : stage.verb,
                title: done
                    ? "re-run — discards every later stage"
                    : `run stage ${stage.n}`,
                onclick: () => runStage(stage.n),
            });
        }
        if (st.status === "error") {
            btn.classList.add("err");
            btn.title = st.error || "failed — click to retry";
        }
        row.appendChild(btn);
        box.appendChild(row);
    }
    renderCoverage();
}

const fmtInt = (n) => (n == null ? "?" : Number(n).toLocaleString());

// Coverage readout for the planned camera set (Stage 4's summary): how well the
// greedy covered the surface, so we can tell at a glance if cameras/greedy are
// healthy. Shown only once cameras are planned.
function renderCoverage() {
    const box = inputs && inputs._coverage;
    if (!box) return;
    box.replaceChildren();
    const s4 = stageState(cellStatus, 4);
    const sum = s4.status === "done" ? s4.summary : null;
    if (!sum) return;
    const cov = sum.coverage || {};
    const patches = sum.patches || 0;
    const seen = cov.seen_at_least_once || 0;
    const sat = cov.satisfied || 0;
    const satPct = cov.satisfied_pct != null ? cov.satisfied_pct : (patches ? Math.round((100 * sat) / patches) : 0);
    const seenPct = patches ? Math.round((100 * seen) / patches) : 0;
    const satColor = satPct >= 90 ? "var(--green)" : satPct >= 70 ? "#d0a020" : "var(--red)";

    box.appendChild(el("div", { class: "svc-title", text: "coverage" }));
    const rows = [
        ["patches", fmtInt(patches), null],
        ["cameras · candidates", `${fmtInt(sum.cameras)} · ${fmtInt(sum.candidates)}`, null],
        ["satisfied (≥K angles)", `${fmtInt(sat)} · ${satPct}%`, satColor],
        ["under-covered", fmtInt(Math.max(0, seen - sat)), null],
        ["occlusion-culled", fmtInt(cov.occlusion_culled), cov.occlusion_culled ? "var(--red)" : null],
        ["reached (≥1 view)", `${fmtInt(seen)} · ${seenPct}%`, null],
        ["mean angles/patch", cov.mean_angles_seen != null ? String(cov.mean_angles_seen) : "?", null],
        ["target K / sectors", `${sum.angles_per_patch ?? "?"} / ${sum.angular_sectors ?? "?"}`, null],
    ];
    for (const [k, v, color] of rows) {
        const row = el("div");
        Object.assign(row.style, {
            display: "flex",
            justifyContent: "space-between",
            gap: "10px",
            fontSize: "12px",
            padding: "1px 0",
        });
        row.appendChild(el("span", { class: "muted", text: k }));
        const val = el("span", { text: String(v) });
        if (color) val.style.color = color;
        row.appendChild(val);
        box.appendChild(row);
    }
}

// Fetch this cell's per-stage status; re-render the stepper; load (or drop) the
// surfel cloud as Stage 3 completes / is reverted. Polls while any stage runs.
function pollStages() {
    stopPoll();
    const seq = openSeq;
    const tick = async () => {
        if (seq !== openSeq || !current) return stopPoll();
        let cell;
        try {
            const payload = await api.splatStageCells(current.run);
            cell = (payload.cells || []).find(
                (c) => c.slot === current.slot && c.model === current.model,
            );
        } catch {
            return; // transient — retry next tick
        }
        if (seq !== openSeq) return stopPoll();
        cellStatus = cell;
        renderStepper();
        const s3 = cell && cell.stage3;
        if (s3 && s3.status === "done" && s3.url && !cloudLoaded) {
            cloudLoaded = true;
            const bust = `?t=${Date.now()}`;
            loadClouds(
                openSeq,
                api.absUrl(s3.url + bust),
                s3.detail_url ? api.absUrl(s3.detail_url + bust) : null,
                s3.summary,
                captureCamera(),
            );
        } else if ((!s3 || s3.status !== "done") && cloudLoaded) {
            cloudLoaded = false; // surfels reverted → clear the canvas
            void teardown();
            setStatus("no splat yet — run surfels (Stage 3)", "");
        }
        if (!anyStageRunning(cell)) stopPoll();
    };
    void tick();
    pollTimer = setInterval(tick, POLL_MS);
}

async function refreshStepper(seq) {
    let cell;
    try {
        const payload = await api.splatStageCells(current.run);
        cell = (payload.cells || []).find(
            (c) => c.slot === current.slot && c.model === current.model,
        );
    } catch {
        cell = null;
    }
    if (seq !== openSeq) return;
    cellStatus = cell;
    renderStepper();
    // If a surfel cloud already exists and nothing was pre-loaded, show it.
    const s3 = cell && cell.stage3;
    if (s3 && s3.status === "done" && s3.url && !cloudLoaded) {
        cloudLoaded = true;
        setStatus("loading splat…", "var(--purple)");
        await loadClouds(
            seq,
            api.absUrl(s3.url),
            s3.detail_url ? api.absUrl(s3.detail_url) : null,
            s3.summary,
            null,
        );
    }
    if (anyStageRunning(cell)) pollStages();
}

// ---- open / close -----------------------------------------------------------

export async function openSplatViewer(opts) {
    if (!overlay || !opts || !opts.run) return;
    const seq = ++openSeq;
    stopPoll();
    current = {
        run: opts.run,
        slot: opts.slot,
        model: opts.model,
        source: opts.source,
    };
    cloudLoaded = false;
    cellStatus = null;
    overlay.classList.add("open");
    subEl.textContent = opts.label || `${opts.slot || ""} · ${opts.model || ""}`;
    buildControls(opts.summary || null);
    renderStepper();
    if (opts.url) {
        cloudLoaded = true;
        setStatus("loading…", "var(--purple)");
        await loadClouds(seq, opts.url, opts.detailUrl || null, opts.summary, null);
        void refreshStepper(seq);
    } else {
        setStatus("no splat yet — run surfels (Stage 3)", "");
        await refreshStepper(seq);
    }
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
    cellStatus = null;
    cloudLoaded = false;
    void teardown();
}

export function initSplatViewer() {
    overlay = document.getElementById("splat-viewer");
    canvasEl = document.getElementById("splat-viewer-canvas");
    subEl = document.getElementById("splat-viewer-sub");
    statusEl = document.getElementById("splat-viewer-status");
    controlsEl = document.getElementById("splat-viewer-controls");
    // Right-side patch-image modal (created once, appended to the viewer overlay).
    patchModalEl = el("div", { class: "spv-modal" });
    Object.assign(patchModalEl.style, {
        position: "absolute", top: "0", right: "0", height: "100%", width: "340px",
        boxSizing: "border-box", background: "rgba(18,18,22,0.94)",
        borderLeft: "1px solid rgba(255,255,255,0.12)", overflowY: "auto",
        padding: "12px", display: "none", zIndex: "30", color: "#fff",
    });
    overlay.appendChild(patchModalEl);
    // Click-to-pick a patch — distinguish a click from an orbit drag by movement.
    // Capture phase, so mkkellogg's built-in controls can't swallow the events.
    canvasEl.addEventListener(
        "pointerdown",
        (ev) => {
            _downXY = [ev.clientX, ev.clientY];
        },
        true,
    );
    canvasEl.addEventListener(
        "pointerup",
        (ev) => {
            if (!_downXY) return;
            const moved = Math.hypot(ev.clientX - _downXY[0], ev.clientY - _downXY[1]);
            _downXY = null;
            if (moved > 5 || !patchesOn) return;
            const idx = pickPatch(ev);
            if (idx >= 0) showPatchModal(idx);
        },
        true,
    );
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
