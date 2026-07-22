// Full-screen Gaussian-splat viewer for Stage-2 clouds, on @mkkellogg/gaussian-
// splats-3d (orbit controls + WASM sort).
//
// Live controls: the panel exposes the sampler's ONE quality knob (`detail`, a
// density multiplier — everything else is derived server-side) and re-splats
// the open cell — POST the knob, poll progress, reload in place (camera
// preserved) — so tuning needs no server restart and no manual command.

import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { api } from "./api.js";
import { el } from "./ui.js";
import {
    closeSogView,
    isSogOpen,
    openSogView,
    setSogVisible,
} from "./sogviewer.js";
import { openRefsViewer } from "./refsviewer.js";

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
let voxelPoints = null; // Stage-2 voxel overlay (THREE.Group: green cover + red garbage shells), lazy
let freePoints = null; // Stage-2 FREE-volume overlay (THREE.Group of blue shells), lazy
let freeShells = null; // [{t, cells, mesh}] per pre-meshed clearance threshold (slider index)
let voxelLights = null; // scene lights (THREE.Group) for solid voxel-only mode, lazy
let meshGroup = null; // original-mesh overlay (THREE.Group), lazy
let mode = "splat"; // "splat" | "mesh" | "sog" — the view switch (not an overlay)
let splatSource = "surfels"; // "surfels" (Stage-3 cloud.ply) | "trained" (Stage-6 trained.ply)
let trainedUrl = null; // /artifacts URL of the Stage-6 trained.ply, when it exists
let sogUrl = null; // /artifacts URL of trained.sog (from stage6.sog_url), rendered via PlayCanvas

// Patch-selection debug feature (needs Stage 4 + Stage 5). A selectable points
// overlay of the coverage patches; clicking one opens its Stage-5 reference images
// in a right-side modal. Lazy-loaded when the "patches" toggle is first enabled.
let patchPoints = null; // THREE.Points of patch centres (the raycast target)
let patchViews = null; // per-patch [[camera_index, face_index], …]
let patchFaces = null; // face-name lookup (index → "+x" …)
let refsBase = null; // /artifacts base URL of the Stage-5 refs/ dir, or null
let patchesOn = false;
let cameraPoints = null; // THREE.Points of Stage-4 camera POSITIONS (view-only overlay)
let camPlaying = false; // greedy-order playback of the camera overlay is running
let camPlayReq = null; // requestAnimationFrame handle for playback
let camRevealed = 0; // camera points currently revealed (greedy order == array order)
let camPlayLast = 0; // previous playback frame timestamp (ms), for dt
let patchModalEl = null; // the right-side image modal (created lazily)
let _downXY = null; // pointer-down pos, to tell a click from an orbit drag

// Pipeline stepper state (the side panel drives the whole splat pipeline).
let cellStatus = null; // last-fetched per-stage status of the open cell
let cloudLoaded = false; // whether a surfel cloud is currently in the canvas
let runningAll = false; // the "run all (local stages 1-5)" sequential driver is active
let assetSource = null; // {source, available, active_kind} — which assets feed the pipeline
let modalPrev = null; // previous poll's cell.modal.status (to catch the done transition)
let modalLog = []; // accumulated remote-train heartbeat lines (the live log pane)
let modalPlanShown = false; // one-time: revealed the camera overlay when the plan arrived mid-run
// Client-owned Stage-6 training length (the server injects NO defaults — see
// server ModalTrainRequest / splat stage6 resolve_schedule). `epochs` = passes
// over the view set → iterations = epochs × n_views (scene-size independent,
// since a bigger scene has proportionally more views); `batch` is the GPU-fill
// speed knob (steps = iterations // batch, constant work). Edited in the "train
// on modal" panel; sent with every start.
let modalTrainCfg = {
    epochs: 12,
    batch: 3,
};

// WASD free-fly state (see movementTick). mkkellogg owns mouse orbit/zoom; this
// adds keyboard walk by translating the camera + orbit target together per frame.
const MOVE_CODES = new Set([
    "KeyW",
    "KeyA",
    "KeyS",
    "KeyD",
    "KeyQ",
    "KeyE",
    "Space",
    "ShiftLeft",
    "ShiftRight",
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
    const c = [
        (min[0] + max[0]) / 2,
        (min[1] + max[1]) / 2,
        (min[2] + max[2]) / 2,
    ];
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
    if (mode === "sog") return; // the PlayCanvas view owns WASD while visible
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
    const sprint =
        pressed.has("ShiftLeft") || pressed.has("ShiftRight") ? 3 : 1;
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
        const mats = Array.isArray(o.material)
            ? o.material
            : o.material
              ? [o.material]
              : [];
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
    closeSogView(); // the PlayCanvas canvas lives in canvasEl too (replaceChildren below)
    // Overlays live in the viewer's threeScene; free their GPU resources and
    // reset their toggles before the viewer (and its GL context) goes away.
    for (const obj of [
        voxelPoints,
        freePoints,
        voxelLights,
        meshGroup,
        patchPoints,
        cameraPoints,
    ]) {
        if (obj) {
            viewer?.threeScene?.remove(obj);
            disposeObj(obj);
        }
    }
    voxelPoints = null;
    freePoints = null;
    freeShells = null;
    voxelLights = null;
    meshGroup = null;
    patchPoints = null;
    patchViews = null;
    patchFaces = null;
    refsBase = null;
    patchesOn = false;
    stopCamPlay();
    cameraPoints = null;
    camRevealed = 0;
    hidePatchModal();
    // A rebuild (re-splat) returns to splat mode; the splat is visible by default
    // on the new viewer, and the overlays are gone.
    mode = "splat";
    syncViewButtons();
    if (inputs && inputs.voxels) inputs.voxels.checked = false;
    if (inputs && inputs.garbage) inputs.garbage.checked = false;
    if (inputs && inputs.freevox) inputs.freevox.checked = false;
    if (inputs && inputs.voxpoints) inputs.voxpoints.checked = false;
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

// One viewer construction shared by the splat and mesh-first paths.
function makeViewer(view) {
    return new GaussianSplats3D.Viewer({
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
}

// A 1-splat PLY (INRIA 3DGS layout) as a blob URL: sub-micron scale, near-zero
// alpha, parked far below any scene. mkkellogg's self-driven render loop gates
// on "splat render ready", which only flips once a splat scene has been ADDED —
// with zero scenes it never renders threeScene at all, so a mesh-only view
// needs this invisible seed scene to make the canvas draw.
function dummySplatPlyUrl() {
    const header =
        "ply\nformat binary_little_endian 1.0\nelement vertex 1\n" +
        "property float x\nproperty float y\nproperty float z\n" +
        "property float nx\nproperty float ny\nproperty float nz\n" +
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n" +
        "property float opacity\n" +
        "property float scale_0\nproperty float scale_1\nproperty float scale_2\n" +
        "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n" +
        "end_header\n";
    // opacity logit −4 → α ≈ 0.018 (survives alpha-removal, invisible);
    // log-scales −14 → ~8e-7 m; y = −9999 keeps it out of any framing.
    const vals = new Float32Array([
        0, -9999, 0, 0, 0, 1, 0, 0, 0, -4, -14, -14, -14, 1, 0, 0, 0,
    ]);
    const head = new TextEncoder().encode(header);
    const buf = new Uint8Array(head.length + vals.byteLength);
    buf.set(head, 0);
    buf.set(new Uint8Array(vals.buffer), head.length);
    return URL.createObjectURL(
        new Blob([buf], { type: "application/octet-stream" }),
    );
}

// MESH-FIRST view: the stage-2 voxel grid is measured off the ORIGINAL meshes,
// so the viewer opens on them (the cell's SELECTED splat asset set) before any
// surfel cloud exists — the voxel/free overlays render over the exact geometry
// they were computed from, no Stage 3 required. When Stage 3 later completes,
// loadClouds tears this down and replaces it with the splat view as usual.
async function openMeshView(seq) {
    await teardown();
    if (seq !== openSeq || !current) return;
    const aabb =
        cellStatus?.stage2?.summary?.scene_aabb ||
        cellStatus?.summary?.scene_aabb ||
        null;
    const v = makeViewer(framing(aabb));
    viewer = v;
    try {
        // Seed the invisible dummy scene FIRST (see dummySplatPlyUrl) so the
        // library's render loop actually draws threeScene.
        const dummy = dummySplatPlyUrl();
        try {
            await v.addSplatScene(dummy, {
                showLoadingUI: false,
                splatAlphaRemovalThreshold: 1,
                format: GaussianSplats3D.SceneFormat.Ply,
            });
        } finally {
            URL.revokeObjectURL(dummy);
        }
        if (seq !== openSeq || viewer !== v) return;
        v.start();
    } catch (e) {
        if (seq === openSeq)
            setStatus(
                `viewer failed: ${e && e.message ? e.message : e}`,
                "var(--red)",
            );
        return;
    }
    moveSpeed = speedFor(aabb);
    startMovement();
    mode = "mesh";
    syncViewButtons();
    const ok = await ensureMesh();
    if (seq !== openSeq || viewer !== v) return;
    if (ok && meshGroup) meshGroup.visible = true;
    setStatus(
        "original mesh — voxel overlays available; run surfels (Stage 3) for the splat",
        "",
    );
}

// Build a fresh viewer and load the cell's surfel cloud.
async function loadClouds(seq, url, summary, camera) {
    await teardown();
    if (seq !== openSeq) return;
    const view = camera || framing(summary && summary.scene_aabb);
    const v = makeViewer(view);
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
        if (seq === openSeq)
            setStatus(
                `failed: ${e && e.message ? e.message : e}`,
                "var(--red)",
            );
        return;
    }
    if (seq !== openSeq) return;
    v.start();
    moveSpeed = speedFor(summary && summary.scene_aabb);
    startMovement();
    const baseN = summary && summary.splats;
    setStatus(
        baseN ? `${baseN.toLocaleString()} splats` : "loaded",
        "var(--green)",
    );
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
    return el(
        "label",
        { class: "svc-check" },
        input,
        el("span", { text: label }),
    );
}

// ---- overlays (Stage-2 voxels + original mesh) --------------------------------

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function setOverlay(text) {
    if (inputs && inputs._overlay) inputs._overlay.textContent = text || "";
}

// One translucent VOLUME layer: a merged boundary shell rendered as a
// non-depth-writing transparent surface, so same-class voxel regions read as
// one collective shape over the scene (mesh or splat view). x-ray layers draw
// faintly THROUGH everything (garbage is inside objects — without it, sealed
// interiors are undiscoverable from outside).
// Translucent, unlit OVERLAY material — the shell drawn as a colored haze over
// a splat/mesh view (non-depth-writing so it doesn't occlude the scene).
function voxelOverlayMat(color, opacity, xray) {
    return new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity,
        depthWrite: false,
        depthTest: !xray,
        side: THREE.DoubleSide,
    });
}

// Solid, lit voxel material (MeshLambert): opaque + depth-writing, so the shells
// CONSTRUCT the scene as shaded blocks — a color-coded silhouette. Scene lights
// (ensureVoxelLights) give it form; DoubleSide keeps it correct from inside a
// hollow.
function voxelSolidMat(color) {
    return new THREE.MeshLambertMaterial({ color, side: THREE.DoubleSide });
}

// One voxel shell layer. Starts in the overlay look; `applyVoxelStyle` swaps it
// to the solid lit look for voxel-only mode. Style metadata rides on the mesh
// so a single pass can restyle (and dispose) every layer.
function volumeLayer(geo, color, opacity, xray) {
    const mesh = new THREE.Mesh(geo, voxelOverlayMat(color, opacity, xray));
    mesh.userData.voxColor = color;
    mesh.userData.voxOpacity = opacity;
    mesh.userData.voxXray = xray;
    mesh.userData.voxSolid = false;
    mesh.userData.voxKind = "shell";
    if (xray) mesh.renderOrder = 998; // over the splats, under the patch overlay
    return mesh;
}

// Expand SVX3 quad records (u16 x,y,z · u8 face · u8 pad · u16 run — one
// run-merged exposed voxel face each) into an indexed BufferGeometry. face =
// axis*2 (+side: plane at the cell's max corner on that axis) or axis*2+1
// (−side); the run extends along z for x/y faces, y for z faces.
function quadGeometry(dv, byteOff, count, origin, pitch) {
    const pos = new Float32Array(count * 12);
    const idx = new Uint32Array(count * 6);
    const ax = [0, 0], // run/width axis lookup per face axis
        o = [0, 0, 0];
    for (let q = 0; q < count; q++) {
        const b = byteOff + q * 10;
        o[0] = dv.getUint16(b, true);
        o[1] = dv.getUint16(b + 2, true);
        o[2] = dv.getUint16(b + 4, true);
        const face = dv.getUint8(b + 6);
        const run = dv.getUint16(b + 8, true);
        const axis = face >> 1;
        const positive = (face & 1) === 0;
        const runAxis = axis === 2 ? 1 : 2;
        const widthAxis = 3 - axis - runAxis;
        // Quad corner in world space: cell min corner, plane offset on `axis`.
        const px = origin[0] + o[0] * pitch;
        const py = origin[1] + o[1] * pitch;
        const pz = origin[2] + o[2] * pitch;
        const base = [px, py, pz];
        if (positive) base[axis] += pitch;
        const ru = [0, 0, 0];
        ru[runAxis] = run * pitch;
        const wu = [0, 0, 0];
        wu[widthAxis] = pitch;
        const p = q * 12;
        pos[p] = base[0];
        pos[p + 1] = base[1];
        pos[p + 2] = base[2];
        pos[p + 3] = base[0] + ru[0];
        pos[p + 4] = base[1] + ru[1];
        pos[p + 5] = base[2] + ru[2];
        pos[p + 6] = base[0] + ru[0] + wu[0];
        pos[p + 7] = base[1] + ru[1] + wu[1];
        pos[p + 8] = base[2] + ru[2] + wu[2];
        pos[p + 9] = base[0] + wu[0];
        pos[p + 10] = base[1] + wu[1];
        pos[p + 11] = base[2] + wu[2];
        const v = q * 4;
        const i = q * 6;
        idx[i] = v;
        idx[i + 1] = v + 1;
        idx[i + 2] = v + 2;
        idx[i + 3] = v;
        idx[i + 4] = v + 2;
        idx[i + 5] = v + 3;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    // Face normals for the lit solid material (voxel-only mode). Quad verts
    // aren't shared between faces, so this stays flat per-face — the crisp
    // blocky look, and it's ignored by the unlit overlay material.
    geo.computeVertexNormals();
    return geo;
}

// Expand the SAME SVX3 quads into DEDUPED per-cell CENTER points — a point cloud
// instead of a merged surface. Points don't form an occluding skin, so a cavity
// (a fridge interior: is it garbage or free?) is legible from outside by orbiting.
// `collapse` folds the fine coords onto the class's native cell — 1 for a fine
// class, 8 (one brick) for a brick-resolution class — so a brick-tiled shell
// reads as ONE point per brick, not 64 per face. Cell coords sit in [0, D).
function cellPointsGeometry(dv, byteOff, count, origin, pitch, collapse) {
    const seen = new Set();
    const xs = [];
    const D = 32768; // key radix per axis (~1 km at 3 cm fine); packs into a f64
    const step = pitch * collapse;
    const half = step * 0.5;
    for (let q = 0; q < count; q++) {
        const b = byteOff + q * 10;
        const c = [
            dv.getUint16(b, true),
            dv.getUint16(b + 2, true),
            dv.getUint16(b + 4, true),
        ];
        const run = dv.getUint16(b + 8, true);
        const axis = dv.getUint8(b + 6) >> 1;
        const runAxis = axis === 2 ? 1 : 2; // z-runs for x/y faces, y-runs for z
        const start = c[runAxis];
        for (let r = 0; r < run; r++) {
            c[runAxis] = start + r;
            const gx = Math.floor(c[0] / collapse);
            const gy = Math.floor(c[1] / collapse);
            const gz = Math.floor(c[2] / collapse);
            const key = (gx * D + gy) * D + gz;
            if (seen.has(key)) continue;
            seen.add(key);
            xs.push(
                origin[0] + gx * step + half,
                origin[1] + gy * step + half,
                origin[2] + gz * step + half,
            );
        }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
        "position",
        new THREE.BufferAttribute(new Float32Array(xs), 3),
    );
    return geo;
}

// Point-cloud voxel look: world-sized square dots (~0.6 of a cell) that shrink
// with distance, so the cloud reads as discrete voxels you can orbit through.
function voxelPointsMat(color, cellSize) {
    return new THREE.PointsMaterial({
        color,
        size: cellSize * 0.6,
        sizeAttenuation: true,
    });
}

// One point-cloud layer for a class (sibling of its `volumeLayer` shell, tagged
// `voxKind: "points"` so `syncVoxelVisibility` shows shells OR points, never both).
function pointLayer(geo, color, cellSize) {
    const pts = new THREE.Points(geo, voxelPointsMat(color, cellSize));
    pts.userData.voxColor = color;
    pts.userData.voxKind = "points";
    pts.visible = false;
    return pts;
}

// Build the voxel overlays from voxels.bin (SVX3): merged VOLUMETRIC boundary
// shells per class — cover (green), garbage (red, solid + x-ray pair), and the
// free volume pre-meshed at a ladder of clearance thresholds (blue; the slider
// swaps shells). Returns { group, shells: [{t, cells, mesh}] } — shell meshes
// live in their own group (freePoints) with only the active one visible.
async function buildVoxels(url, summary) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const ab = await res.arrayBuffer();
    const dv = new DataView(ab);
    // Pre-SVX3 packs (point layouts) — ask for a recompute instead of
    // misparsing positions.
    if (ab.byteLength < 16 || dv.getUint32(0, false) !== 0x53565833) {
        throw new Error("stale voxel pack — recompute free space (stage 2)");
    }
    const nCover = dv.getUint32(4, true);
    const nGar = dv.getUint32(8, true);
    const nShells = dv.getUint32(12, true);
    const pitch = (summary && summary.pitch) || 0.03;
    const origin = (summary && summary.origin) || [0, 0, 0];
    // Per-class point-cloud resolution: a brick-tiled shell (foliage-scale packs)
    // collapses to one point per 8³ brick; a fine shell stays per-cell.
    const vizRes = (summary && summary.viz && summary.viz.resolution) || {};
    const collapseFor = (cls) => (vizRes[cls] === "brick" ? 8 : 1);
    let off = 16;
    const group = new THREE.Group();
    if (nCover) {
        const geo = quadGeometry(dv, off, nCover, origin, pitch);
        const cover = volumeLayer(geo, 0x3ddc6a, 0.3, false);
        cover.userData.voxClass = "cover";
        group.add(cover);
        const cc = collapseFor("cover");
        const pts = pointLayer(
            cellPointsGeometry(dv, off, nCover, origin, pitch, cc), 0x3ddc6a, pitch * cc,
        );
        pts.userData.voxClass = "cover";
        group.add(pts);
    }
    off += nCover * 10;
    if (nGar) {
        const geo = quadGeometry(dv, off, nGar, origin, pitch);
        const garSolid = volumeLayer(geo, 0xff4438, 0.4, false);
        garSolid.userData.voxClass = "garbage";
        group.add(garSolid);
        const garXray = volumeLayer(geo, 0xff4438, 0.12, true);
        garXray.userData.voxClass = "garbage";
        group.add(garXray);
        const gc = collapseFor("garbage");
        const pts = pointLayer(
            cellPointsGeometry(dv, off, nGar, origin, pitch, gc), 0xff4438, pitch * gc,
        );
        pts.userData.voxClass = "garbage";
        group.add(pts);
    }
    off += nGar * 10;
    const shells = [];
    const fc = collapseFor("free");
    for (let s = 0; s < nShells; s++) {
        const t = dv.getFloat32(off, true);
        const quads = dv.getUint32(off + 4, true);
        const cells = dv.getUint32(off + 8, true);
        off += 12;
        const geo = quadGeometry(dv, off, quads, origin, pitch);
        const pts = pointLayer(
            cellPointsGeometry(dv, off, quads, origin, pitch, fc), 0x3d8bff, pitch * fc,
        );
        pts.userData.voxClass = "free";
        off += quads * 10;
        const mesh = volumeLayer(geo, 0x3d8bff, 0.22, false);
        mesh.userData.voxClass = "free";
        mesh.userData.voxTranslucentOnly = true; // filling volume — never opaque
        mesh.visible = false;
        shells.push({ t, cells, mesh, points: pts });
    }
    if (off > ab.byteLength) throw new Error("truncated voxel pack");
    return { group, shells };
}

// Show only the slider's shell of the blue FREE volume — an instant geometry
// swap between pre-meshed clearance thresholds, nothing touches the server.
function updateFreeFilter() {
    if (!inputs || !inputs.freeclear) return;
    if (!freeShells || !freeShells.length) {
        if (inputs._freeVal) inputs._freeVal.textContent = "";
        return;
    }
    const i = Math.max(
        0,
        Math.min(
            freeShells.length - 1,
            Math.round(Number(inputs.freeclear.value)),
        ),
    );
    // Only the slider's shell is visible, drawn as the merged surface OR the
    // point cloud per the "points" style toggle (never both).
    const pointsOn = !!(inputs && inputs.voxpoints && inputs.voxpoints.checked);
    for (let s = 0; s < freeShells.length; s++) {
        const active = s === i;
        freeShells[s].mesh.visible = active && !pointsOn;
        if (freeShells[s].points) freeShells[s].points.visible = active && pointsOn;
    }
    if (inputs._freeVal) {
        inputs._freeVal.textContent = `${freeShells[i].t.toFixed(2)}m · ${freeShells[i].cells.toLocaleString()}`;
    }
    viewer?.forceRenderNextFrame?.();
}

// The shell the slider currently points at (or null pre-load).
function activeShell() {
    if (!freeShells || !freeShells.length || !inputs || !inputs.freeclear) {
        return null;
    }
    const i = Math.max(
        0,
        Math.min(
            freeShells.length - 1,
            Math.round(Number(inputs.freeclear.value)),
        ),
    );
    return freeShells[i];
}

// POST the active shell's threshold to the server — re-bakes clearance_m in
// freespace.npz (no re-voxelization) and invalidates stages 4+; until then the
// slider is a client-side preview only.
async function applyClearance() {
    const shell = activeShell();
    if (!current || !shell) return;
    const btn = inputs._freeApply;
    if (btn) btn.disabled = true;
    try {
        const st = await api.splatStage2Clearance(
            current.run,
            current.slot,
            current.model,
            { clearance: shell.t },
        );
        const n = (st.summary && st.summary.free_voxels) || 0;
        setOverlay(
            `clearance ${shell.t.toFixed(2)}m applied — ${n.toLocaleString()} free vox · stages 4+ invalidated`,
        );
    } catch (e) {
        setOverlay(`clearance apply failed: ${e.message}`);
    } finally {
        if (btn) btn.disabled = false;
    }
}

// Ensure Stage 2 (the voxel grid) has run for the open cell; compute + poll if not.
async function ensureFreeSpace() {
    const c = current;
    let st = await api.splatStage2Status(c.run, c.slot, c.model);
    if (st.status === "done" && st.url) return st;
    setOverlay("voxelizing…");
    await api.splatStage2Start(c.run, c.slot, c.model, {});
    const seq = openSeq;
    while (seq === openSeq) {
        await sleep(1000);
        st = await api.splatStage2Status(c.run, c.slot, c.model);
        if (!st.running) break;
        setOverlay(
            `voxelizing: ${st.phase || ""} ${st.total ? `${st.done}/${st.total}` : ""}`,
        );
    }
    if (st.status !== "done" || !st.url)
        throw new Error(st.error || "voxelize failed");
    return st;
}

// Highlight the active view in the 4-way switch. "original" is mesh mode, "sog"
// is the PlayCanvas view; the two splat sources map to whichever .ply is loaded.
function syncViewButtons() {
    if (!inputs) return;
    const active =
        mode === "mesh"
            ? "original"
            : mode === "sog"
              ? "sog"
              : mode === "voxels"
                ? "voxels"
                : splatSource;
    inputs._surfBtn?.classList.toggle("on", active === "surfels");
    inputs._trainBtn?.classList.toggle("on", active === "trained");
    inputs._sogBtn?.classList.toggle("on", active === "sog");
    inputs._origBtn?.classList.toggle("on", active === "original");
    inputs._voxViewBtn?.classList.toggle("on", active === "voxels");
}

function syncSogBtn() {
    const btn = inputs && inputs._sogBtn;
    if (!btn) return;
    btn.disabled = !sogUrl;
    btn.title = sogUrl
        ? "the SOG-encoded trained splat, rendered with PlayCanvas"
        : "encode it first: node client/tools/ply-to-sog.mjs …/trained.ply …/trained.sog";
}

// Enable the "trained" and "sog" toggles from the cell poll's Stage-6 status
// (Stage 6 is a backend script, so trained.ply / trained.sog just appear on disk
// and the server surfaces both URLs — no client-side existence probe).
function updateSourceAvail() {
    const s6 = (cellStatus && cellStatus.stage6) || {};
    trainedUrl = s6.url || null;
    const btn = inputs && inputs._trainBtn;
    if (btn) {
        btn.disabled = !trainedUrl;
        btn.title = trainedUrl
            ? "the Stage-6 fine-tuned splat (trained.ply)"
            : "run Stage 6 (the training script) to produce trained.ply";
    }
    if (!trainedUrl && splatSource === "trained") splatSource = "surfels";

    sogUrl = s6.sog_url || null;
    syncSogBtn();
    syncViewButtons();
}

// Switch to the SOG (PlayCanvas) view: hide the three.js splat/mesh/overlays and
// reveal the PlayCanvas canvas, opening it (and loading trained.sog) on first
// use. The camera pose carries over from the live mkkellogg camera.
async function setSogMode() {
    if (!sogUrl || mode === "sog") {
        syncViewButtons();
        return;
    }
    const cam = captureCamera();
    mode = "sog";
    syncViewButtons();
    splatVisible(false);
    if (meshGroup) meshGroup.visible = false;
    if (voxelPoints) voxelPoints.visible = false; // PlayCanvas owns the canvas
    if (freePoints) freePoints.visible = false;
    hidePatchModal();
    try {
        if (!isSogOpen()) {
            setStatus("loading SOG (PlayCanvas)…", "var(--purple)");
            const s3 = (cellStatus && cellStatus.stage3) || {};
            await openSogView({
                container: canvasEl,
                url: api.absUrl(sogUrl),
                view: cam || framing(s3.summary && s3.summary.scene_aabb),
                speed: moveSpeed,
            });
        }
        if (mode !== "sog") return; // switched away while loading
        setSogVisible(true);
        setStatus("SOG · PlayCanvas", "var(--green)");
    } catch (e) {
        setStatus(
            `SOG failed: ${e && e.message ? e.message : e}`,
            "var(--red)",
        );
        mode = "splat";
        splatVisible(true);
        syncViewButtons();
    }
}

// Switch which splat is loaded (surfels ↔ trained). Both share the cell's world
// frame, so the live camera is preserved across the reload; the fresh viewer
// starts in splat view (mesh / voxel / patch overlays reset).
async function setSource(next) {
    if (next === splatSource) return;
    const s3 = (cellStatus && cellStatus.stage3) || {};
    if (next === "trained" && !trainedUrl) return;
    if (next === "surfels" && !s3.url) return;
    splatSource = next;
    mode = "splat";
    syncViewButtons();
    const cam = captureCamera();
    const bust = `?t=${Date.now()}`;
    cloudLoaded = true;
    if (next === "trained") {
        setStatus("loading trained splat…", "var(--purple)");
        await loadClouds(
            openSeq,
            api.absUrl(trainedUrl + bust),
            { scene_aabb: s3.summary && s3.summary.scene_aabb },
            cam,
        );
    } else {
        setStatus("loading surfels…", "var(--purple)");
        await loadClouds(openSeq, api.absUrl(s3.url + bust), s3.summary, cam);
    }
}

// The unified view switch. surfels ↔ trained reload the splat scene (a
// different .ply); original flips to the mesh overlay; sog flips to the
// PlayCanvas view. Reuses setSource (reload) and setMode (visibility), so
// flipping between an already-loaded splat and the mesh is instant — no reload
// — which is what makes the side-by-side snappy.
async function setView(next) {
    if (next === "sog") {
        await setSogMode();
        return;
    }
    if (isSogOpen()) setSogVisible(false); // leaving sog: reveal the three.js canvas
    if (next === "original") {
        await setMode("mesh");
        return;
    }
    if (next === "voxels") {
        // Works with or without a splat (it hides whatever's underneath); the
        // mesh-first dummy scene keeps the render loop alive pre-Stage-3.
        await setMode("voxels");
        return;
    }
    // Pre-Stage-3 (mesh-first view): there is no splat to reveal yet.
    if (next === "surfels" && !cloudLoaded) {
        setStatus("no splat yet — run surfels (Stage 3)", "");
        return;
    }
    if (splatSource !== next) {
        await setSource(next); // loads the other .ply and returns to splat view
        return;
    }
    if (mode !== "splat") await setMode("splat"); // already loaded — just reveal it
}

function splatVisible(on) {
    const sm = viewer && viewer.getSplatMesh && viewer.getSplatMesh();
    if (sm) sm.visible = on;
}

// Lazily build + add the voxel overlays (added hidden); returns success. One
// fetch builds BOTH the occupied/garbage group and the blue FREE layer; the
// clearance slider is initialized from the grid's baked threshold + observed
// clearance range on first build.
async function ensureVoxels() {
    if (voxelPoints) return true;
    const seq = openSeq;
    try {
        const st = await ensureFreeSpace();
        if (seq !== openSeq || !viewer) return false;
        const built = await buildVoxels(
            api.absUrl(st.url + `?t=${Date.now()}`),
            st.summary,
        );
        if (seq !== openSeq || !viewer) {
            disposeObj(built.group);
            for (const s of built.shells) disposeObj(s.mesh);
            return false;
        }
        voxelPoints = built.group;
        voxelPoints.visible = false;
        viewer.threeScene.add(voxelPoints);
        freeShells = built.shells;
        freePoints = new THREE.Group();
        for (const s of freeShells) {
            freePoints.add(s.mesh);
            if (s.points) freePoints.add(s.points);
        }
        freePoints.visible = false;
        viewer.threeScene.add(freePoints);
        // The slider indexes the pre-meshed shell ladder; default to the shell
        // carrying the BAKED threshold (always in the ladder — what stage 4 uses).
        const p = (st.summary && st.summary.params) || {};
        if (inputs && inputs.freeclear && freeShells.length) {
            inputs.freeclear.min = 0;
            inputs.freeclear.max = freeShells.length - 1;
            inputs.freeclear.step = 1;
            let best = 0;
            for (let i = 0; i < freeShells.length; i++) {
                if (
                    Math.abs(
                        freeShells[i].t - (p.clearance ?? freeShells[0].t),
                    ) <
                    Math.abs(
                        freeShells[best].t - (p.clearance ?? freeShells[0].t),
                    )
                ) {
                    best = i;
                }
            }
            inputs.freeclear.value = best;
        }
        updateFreeFilter();
        const sv = (st.summary && st.summary.solid_voxels) || 0;
        const gv = (st.summary && st.summary.garbage_voxels) || 0;
        const fv = (st.summary && st.summary.free_voxels) || 0;
        setOverlay(
            `occupied: ${sv.toLocaleString()} · garbage: ${gv.toLocaleString()} · free@${(p.clearance ?? 0.35).toFixed(2)}m: ${fv.toLocaleString()}`,
        );
        return true;
    } catch (e) {
        setOverlay(`voxels failed: ${e.message}`);
        return false;
    }
}

// The voxel overlay — Stage 2's occupied cells (green) + garbage cells (red)
// over whichever scene is showing (splat or mesh; same world frame). A plain
// view-only toggle, exactly like the camera-position overlay.
async function setVoxels(on) {
    if (on) {
        const ok = await ensureVoxels();
        if (!ok) return;
    }
    // Central chokepoint: applies overlay-vs-solid style, x-ray gating, and
    // visibility from the checkboxes + current mode.
    syncVoxelVisibility();
    viewer?.forceRenderNextFrame?.();
}

// The FREE-voxel overlay (blue) — the camera-placeable subset of empty space at
// the slider's clearance threshold (NOT all empty space). Same lazy build as
// the cover overlay (one shared fetch).
async function setFreeVoxels(on) {
    if (on) {
        const ok = await ensureVoxels();
        if (!ok) return;
        updateFreeFilter();
    }
    syncVoxelVisibility();
    viewer?.forceRenderNextFrame?.();
}

// UNLIT swap for one glTF material: MeshBasicMaterial keeping the base-color
// map/factor + alpha semantics. The splat viewer's scene has NO lights (splats
// don't need them), so GLTFLoader's PBR materials would render pure black.
// Unlit is also the truer debug view: the whole pipeline is unlit albedo
// (Stage 5 renders exactly this), and MeshBasic ignores TRELLIS's unreliable
// normals; DoubleSide keeps wrong-winding faces visible.
function unlitMaterial(m) {
    const u = new THREE.MeshBasicMaterial({
        map: m.map || null,
        color: m.color ? m.color.clone() : new THREE.Color(0xffffff),
        transparent: !!m.transparent,
        opacity: m.opacity != null ? m.opacity : 1,
        alphaTest: m.alphaTest || 0,
        side: THREE.DoubleSide,
        vertexColors: !!m.vertexColors,
    });
    // The old material's textures stay owned by `u.map`; drop only the program.
    m.dispose();
    return u;
}

function makeUnlit(root) {
    root.traverse((o) => {
        if (!o.isMesh || !o.material) return;
        o.material = Array.isArray(o.material)
            ? o.material.map(unlitMaterial)
            : unlitMaterial(o.material);
    });
}

// Load the cell's original mesh (SMB1 bundle) into the splat scene, coincident
// with the cloud, for a splat-vs-mesh look comparison. Streams the cell's
// SELECTED splat asset set (`variant=splat` follows the assets picker — served
// AS-IS, no tier build), so the mesh on screen is exactly the geometry Stage 2
// voxelized and Stage 3 samples. Falls back to the default serving set if the
// selected source has nothing streamable.
async function buildMeshGroup() {
    let res = await fetch(
        api.meshesUrl(current.run, current.slot, current.model, {
            variant: "splat",
        }),
        { cache: "no-store" },
    );
    if (!res.ok) {
        res = await fetch(
            api.meshesUrl(current.run, current.slot, current.model, {}),
            { cache: "no-store" },
        );
        if (res.ok)
            setOverlay("showing default assets (splat tier not built yet)");
    }
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
    const loader = new GLTFLoader()
        .setKTX2Loader(ktx2)
        .setMeshoptDecoder(MeshoptDecoder);
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
            makeUnlit(gltf.scene); // lightless scene — PBR would render black
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

// The voxel overlays follow their checkboxes in every three.js view (splat AND
// mesh — same world frame); only the PlayCanvas SOG view hides them.
function syncVoxelVisibility() {
    const solo = mode === "voxels";
    if (solo) ensureVoxelLights();
    // Voxel-only mode renders the shells SOLID + lit (they construct the scene);
    // every other view draws them as the translucent overlay.
    applyVoxelStyle(solo);
    if (voxelLights) voxelLights.visible = solo;
    const notSog = mode !== "sog";
    const coverOn = !!(inputs && inputs.voxels && inputs.voxels.checked);
    const garbageOn = !!(inputs && inputs.garbage && inputs.garbage.checked);
    // Style toggle: draw each class as its merged shell (default) or as a per-cell
    // POINT cloud (see-through — for reading a cavity's interior classification).
    const pointsOn = !!(inputs && inputs.voxpoints && inputs.voxpoints.checked);
    if (voxelPoints) {
        // The group rides along; each layer is gated by its own class toggle, so
        // cover (green) and garbage (red) are independent controls.
        voxelPoints.visible = notSog && (coverOn || garbageOn);
        for (const m of voxelPoints.children) {
            const cls = m.userData && m.userData.voxClass;
            let on = cls === "garbage" ? garbageOn : coverOn;
            // Style gate: a class has both a shell and a point layer — show only
            // the one the toggle selects.
            const isPts = m.userData && m.userData.voxKind === "points";
            if (isPts !== pointsOn) on = false;
            // The x-ray garbage duplicate is an OVERLAY affordance (see sealed
            // interiors through walls); it only z-fights the solid pass in
            // voxel-only mode, so drop it there.
            if (m.userData && m.userData.voxXray && solo) on = false;
            m.visible = on;
        }
    }
    if (freePoints) {
        freePoints.visible =
            notSog && !!(inputs && inputs.freevox && inputs.freevox.checked);
        // Re-apply the shell/points split for the active clearance shell.
        updateFreeFilter();
    }
}

// Every voxel shell mesh across both groups (cover, garbage solid + x-ray, and
// each free-clearance shell).
function collectVoxelMeshes() {
    const out = [];
    if (voxelPoints) voxelPoints.traverse((o) => o.isMesh && out.push(o));
    if (freeShells) for (const s of freeShells) out.push(s.mesh);
    return out;
}

// Swap every voxel shell between the translucent OVERLAY look (over a splat or
// mesh) and the SOLID lit look (voxel-only mode). Idempotent per mesh — only
// rebuilds a material when the style actually flips — and disposes the replaced
// one so repeated view switches don't leak GPU programs.
function applyVoxelStyle(solid) {
    for (const m of collectVoxelMeshes()) {
        const d = m.userData;
        // FREE is a filling volume (camera-placeable space) — opaque it would
        // paint over the whole scene, so it stays translucent even in solid mode.
        const wantSolid = solid && !d.voxTranslucentOnly;
        if (d.voxSolid === wantSolid) continue;
        const old = m.material;
        m.material = wantSolid
            ? voxelSolidMat(d.voxColor)
            : voxelOverlayMat(d.voxColor, d.voxOpacity, d.voxXray);
        old?.dispose?.();
        m.renderOrder = !wantSolid && d.voxXray ? 998 : 0;
        d.voxSolid = wantSolid;
    }
}

// Lazily light the scene for solid voxel-only mode (hemisphere + key/fill). The
// other views are unlit (splats have their own shader; the mesh overlay is
// MeshBasic), so these only ever shade the solid voxel material — and they're
// toggled off (voxelLights.visible) outside voxel-only mode.
function ensureVoxelLights() {
    if (voxelLights || !viewer) return;
    const g = new THREE.Group();
    g.add(new THREE.HemisphereLight(0xffffff, 0x2a2a33, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 1.1);
    key.position.set(0.5, 1.0, 0.35);
    g.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.45);
    fill.position.set(-0.55, 0.35, -0.6);
    g.add(fill);
    viewer.threeScene.add(g);
    voxelLights = g;
}

// Switch the view between the Gaussian splat and the original mesh (same world
// frame → identical pose). Splat mode hides the mesh; mesh mode hides the splat
// and shows the mesh. Overlays (voxels / patches / cameras) ride along.
async function setMode(next) {
    mode = next;
    syncViewButtons();
    if (next === "splat") {
        splatVisible(true);
        if (meshGroup) meshGroup.visible = false;
        syncVoxelVisibility();
        setOverlay("");
        return;
    }
    if (next === "voxels") {
        // Voxel-only: nothing underneath — the shells ARE the scene. Hide the
        // splat + mesh, ensure the grid is built, and default the cover layer on
        // so the view isn't black on entry (the free volume keeps its own
        // toggle/slider). syncVoxelVisibility switches the shells to solid+lit.
        splatVisible(false);
        if (meshGroup) meshGroup.visible = false;
        const ok = await ensureVoxels();
        if (mode !== "voxels") return; // switched away while building
        if (
            ok &&
            inputs &&
            inputs.voxels &&
            !inputs.voxels.checked &&
            (!inputs.garbage || !inputs.garbage.checked) &&
            (!inputs.freevox || !inputs.freevox.checked)
        ) {
            inputs.voxels.checked = true; // don't open on a black screen
        }
        syncVoxelVisibility();
        setOverlay(
            ok
                ? "voxel-only — solid shaded shells (toggle cover/garbage/free below)"
                : "",
        );
        viewer?.forceRenderNextFrame?.();
        return;
    }
    // mesh mode: hide the splat, show the mesh
    splatVisible(false);
    const ok = await ensureMesh();
    if (mode !== "mesh") return; // switched back while the mesh was loading
    if (ok && meshGroup) meshGroup.visible = true;
    syncVoxelVisibility();
}

// ---- splat asset-source selector (which meshes feed the pipeline) -----------

function syncAssetSourceUI() {
    if (!inputs || !inputs._srcBtns) return;
    const st = assetSource;
    for (const [kind, btn] of Object.entries(inputs._srcBtns)) {
        const n = st && st.available ? st.available[kind] || 0 : 0;
        btn.classList.toggle("on", !!st && st.source === kind);
        btn.disabled = !st || (kind !== "auto" && n === 0);
        btn.title =
            kind === "auto"
                ? "generated build when present, else the library build"
                : `${n} ${kind} meshes on disk`;
    }
    if (inputs._srcVal) {
        inputs._srcVal.textContent = st ? `→ ${st.active_dir || "none"}` : "";
    }
}

async function refreshAssetSource() {
    if (!current) return;
    try {
        assetSource = await api.splatSourceGet(
            current.run,
            current.slot,
            current.model,
        );
    } catch {
        assetSource = null;
    }
    syncAssetSourceUI();
}

// Pin a different asset set: the server wipes every stage output (the meshes
// changed), so the stepper resets to idle and the pipeline is re-run from
// stage 1 against the newly selected source.
async function setAssetSource(kind) {
    if (!current || !assetSource || assetSource.source === kind) return;
    try {
        assetSource = await api.splatSourceSet(
            current.run,
            current.slot,
            current.model,
            { source: kind },
        );
        const cell = await fetchCellStatus().catch(() => null);
        if (cell) {
            cellStatus = cell;
            renderStepper();
        }
        // Every stage output was reverted server-side and the meshes changed:
        // drop the stale cloud/overlays and land on the mesh-first view of the
        // newly selected asset set (voxel overlays recompute from there).
        cloudLoaded = false;
        await openMeshView(openSeq);
        setStatus(
            `splat assets: ${kind} — stages reset, re-run the pipeline`,
            "var(--purple)",
        );
    } catch (e) {
        setStatus(`asset switch failed: ${e.message}`, "var(--red)");
    }
    syncAssetSourceUI();
}

function buildControls(summary) {
    if (!controlsEl) return;
    const p = (summary && summary.params) || {};
    inputs = {};
    controlsEl.replaceChildren();

    // THE quality knob: `detail` multiplies surfel density around the calibrated
    // default look (1 = default, 2 = twice the surfels/m², 0.5 = half). Spacing,
    // disk radius, feature refinement, culling are all derived server-side. The
    // count/size estimate scales the last run's actual output by the multiplier
    // ratio (needs one completed sample; before that it shows the spacing only).
    const lastDetail = p.detail || 1;
    const lastSampled = (summary && summary.sampled) || 0;
    // Mirrors stage3's _BASE_SPACING (2.95 cm at detail=1), display only.
    const spacingCm = (v) => (2.95 / Math.sqrt(v)).toFixed(2);
    const dFmt = (v) => {
        const s = `${v.toFixed(2)}× · ${spacingCm(v)}cm`;
        if (!lastSampled) return s;
        const est = (lastSampled * v) / lastDetail;
        return `${s} · ~${Math.round(est / 1000)}k · ${((est * BYTES_PER_SPLAT) / 1e6).toFixed(0)}MB`;
    };
    const detail = sliderRow(
        "detail",
        "detail",
        0.25,
        4,
        0.05,
        lastDetail,
        dFmt,
    );

    // Forced per-object pool size — a RESOURCE knob (not part of SampleParams):
    // 0 = auto (min(cores, 8) server-side); >=1 pins the pool (re-clamped to the
    // object count + a memory cap in stage3). `summary.workers` is the RESOLVED
    // count of the last run, shown as a hint; the slider defaults to auto so a
    // re-splat never silently pins the pool after one run.
    const lastWorkers = (summary && summary.workers) || 0;
    const workers = sliderRow(
        "workers",
        "workers",
        0,
        32,
        1,
        0,
        (v) =>
            v < 1 ? (lastWorkers ? `auto (last ${lastWorkers})` : "auto") : `${Math.round(v)}`,
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

    // One 3-way view switch: the pre-fine-tune surfels (Stage 3), the trained
    // splat (Stage 6), or the original meshes — flip between all three in place
    // for a side-by-side (same world frame → identical pose). surfels ↔ trained
    // reload the splat; "original" reuses the mesh overlay (setMode). "trained"
    // stays disabled until trained.ply exists.
    mode = "splat";
    splatSource = "surfels";
    const surfBtn = el("button", {
        class: "svc-seg-btn on",
        text: "surfels",
        title: "the pre-fine-tuning surfel cloud (Stage 3)",
        onclick: () => setView("surfels"),
    });
    const trainBtn = el("button", {
        class: "svc-seg-btn",
        text: "trained",
        disabled: true, // enabled by updateSourceAvail once trained.ply exists
        onclick: () => setView("trained"),
    });
    const sogBtn = el("button", {
        class: "svc-seg-btn",
        text: "sog",
        disabled: true, // enabled by updateSourceAvail once trained.sog exists
        onclick: () => setView("sog"),
    });
    const origBtn = el("button", {
        class: "svc-seg-btn",
        text: "original",
        title: "the original generated meshes (same world frame)",
        onclick: () => setView("original"),
    });
    const voxViewBtn = el("button", {
        class: "svc-seg-btn",
        text: "voxels",
        title: "voxel-only: the stage-2 shells construct the scene as solid shaded blocks (green cover · red garbage · blue free, per their toggles below)",
        onclick: () => setView("voxels"),
    });
    inputs._surfBtn = surfBtn;
    inputs._trainBtn = trainBtn;
    inputs._sogBtn = sogBtn;
    inputs._origBtn = origBtn;
    inputs._voxViewBtn = voxViewBtn;
    const seg = el(
        "div",
        { class: "svc-seg" },
        surfBtn,
        trainBtn,
        sogBtn,
        origBtn,
        voxViewBtn,
    );

    // Stage-2 voxel overlay: occupied cells as green disks over the scene,
    // garbage cells (sealed interiors) as red x-ray disks — one toggle, like
    // the camera-position overlay, just denser.
    const voxRow = checkRow("voxels", "solid voxels (green cover)", false);
    inputs.voxels.addEventListener("change", () =>
        setVoxels(inputs.voxels.checked),
    );
    const garbageRow = checkRow(
        "garbage",
        "garbage voxels (red · sealed interiors)",
        false,
    );
    inputs.garbage.addEventListener("change", () =>
        setVoxels(inputs.garbage.checked),
    );
    // FREE-voxel overlay (blue): the camera-placeable subset of empty space at
    // the clearance slider's threshold. The slider indexes the pack's
    // pre-meshed shell ladder (an instant geometry swap — no server call);
    // "apply" re-bakes the threshold into freespace.npz so stage 4 plans
    // against it (and invalidates stages 4+).
    const freeRow = checkRow(
        "freevox",
        "free volume (blue · camera-placeable)",
        false,
    );
    inputs.freevox.addEventListener("change", () =>
        setFreeVoxels(inputs.freevox.checked),
    );
    inputs._freeVal = el("span", { class: "svc-val", text: "" });
    const freeClearInput = el("input", {
        type: "range",
        class: "svc-range",
        min: 0, // reconfigured to the shell ladder once stage 2 loads
        max: 1,
        step: 1,
        value: 0,
    });
    inputs.freeclear = freeClearInput;
    freeClearInput.addEventListener("input", () => updateFreeFilter());
    const freeClearRow = el(
        "div",
        { class: "svc-row" },
        el("span", { class: "svc-lab", text: "clearance" }),
        freeClearInput,
        inputs._freeVal,
    );
    const freeApplyBtn = el("button", {
        class: "splat-stage2-btn",
        text: "apply",
        title: "bake this clearance into freespace.npz (stage 4 plans against it; stages 4+ are invalidated)",
        onclick: () => applyClearance(),
    });
    inputs._freeApply = freeApplyBtn;
    const freeApplyRow = el(
        "div",
        { class: "svc-row" },
        el("span", { class: "svc-lab", text: "" }),
        freeApplyBtn,
    );
    // Render style for every voxel class (cover / garbage / free): merged
    // translucent shells (default) or a per-cell POINT cloud. Points don't form
    // an occluding skin, so you can orbit and read a cavity's INTERIOR — e.g.
    // whether a fridge interior is garbage (red = sealed) or free (blue).
    const voxPointsRow = checkRow(
        "voxpoints",
        "point voxels (see inside cavities)",
        false,
    );
    inputs.voxpoints.addEventListener("change", () => {
        syncVoxelVisibility();
        viewer?.forceRenderNextFrame?.();
    });
    // Patch inspector (needs Stage 4 + Stage 5): select a coverage patch on the
    // splat and open its reference images in the right-side modal.
    const patchRow = checkRow("patches", "patches (click to inspect)", false);
    inputs.patches.addEventListener("change", () =>
        setPatches(inputs.patches.checked),
    );
    // Browse the Stage-5 reference frames (RGB · alpha · depth) as a lazy
    // thumbnail matrix + full-res inspector, decoded from the SZF containers
    // client-side (opens its own overlay; resolves the refs via Stage-5 status).
    const refsBtn = el("button", {
        class: "splat-stage2-btn",
        text: "view refs",
        title: "browse the Stage-5 reference frames (RGB · alpha · depth)",
        onclick: () => {
            if (!current) return;
            openRefsViewer({
                run: current.run,
                slot: current.slot,
                model: current.model,
                label: subEl?.textContent,
            });
        },
    });
    const refsRow = el(
        "div",
        { class: "svc-row" },
        el("span", { class: "svc-lab", text: "refs" }),
        refsBtn,
    );
    // Every point where Stage 4 placed a camera (cameras.json → positions), view-only.
    const camRow = checkRow("cameras", "camera positions", false);
    inputs.cameras.addEventListener("change", () =>
        setCameras(inputs.cameras.checked),
    );
    // Greedy-order playback: reveal the cameras in the order Stage 4 chose them,
    // at an adjustable rate (points/sec). The count column tracks the current pick.
    const camPlayBtn = el("button", {
        class: "splat-stage2-btn",
        text: "▶ play",
        title: "reveal camera positions in the order Stage 4 greedily chose them",
        onclick: () => toggleCamPlay(),
    });
    inputs._camPlayBtn = camPlayBtn;
    inputs._camPlayVal = el("span", { class: "svc-val", text: "" });
    const camPlayRow = el(
        "div",
        { class: "svc-row" },
        el("span", { class: "svc-lab", text: "greedy" }),
        camPlayBtn,
        inputs._camPlayVal,
    );
    const camSpeedRow = sliderRow(
        "camspeed",
        "speed",
        5,
        5000,
        5,
        200,
        (v) => `${Math.round(v)}/s`,
    );
    const overlay = el("div", { class: "svc-actual" });
    inputs._overlay = overlay;
    inputs._stepper = el("div", { class: "svc-stepper" });
    inputs._coverage = el("div", { class: "svc-coverage" });
    inputs._modal = el("div", { class: "svc-coverage" });
    inputs._log = el("pre", { class: "svc-log" });
    Object.assign(inputs._log.style, {
        maxHeight: "180px",
        overflowY: "auto",
        margin: "4px 0 0",
        padding: "6px 8px",
        background: "#0b0b0e",
        border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: "4px",
        font: "11px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace",
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
        color: "#cfcfe0",
    });

    // Asset-source selector: which meshes the WHOLE pipeline samples/renders.
    // Populated by refreshAssetSource() once the server state arrives.
    inputs._srcBtns = {
        auto: el("button", {
            class: "splat-stage2-btn",
            text: "auto",
            disabled: true,
            onclick: () => setAssetSource("auto"),
        }),
        generated: el("button", {
            class: "splat-stage2-btn",
            text: "generated",
            disabled: true,
            onclick: () => setAssetSource("generated"),
        }),
        library: el("button", {
            class: "splat-stage2-btn",
            text: "library",
            disabled: true,
            onclick: () => setAssetSource("library"),
        }),
    };
    inputs._srcVal = el("span", { class: "svc-val", text: "" });
    const srcRow = el(
        "div",
        { class: "svc-row" },
        el("span", { class: "svc-lab", text: "assets" }),
        inputs._srcBtns.auto,
        inputs._srcBtns.generated,
        inputs._srcBtns.library,
        inputs._srcVal,
    );

    controlsEl.append(
        el("div", { class: "svc-title", text: "pipeline" }),
        srcRow,
        inputs._stepper,
        inputs._coverage,
        el("div", { class: "svc-title", text: "train on modal (stages 4–6)" }),
        inputs._modal,
        inputs._log,
        el("div", { class: "svc-title", text: "view" }),
        seg,
        voxRow,
        garbageRow,
        freeRow,
        freeClearRow,
        freeApplyRow,
        voxPointsRow,
        patchRow,
        refsRow,
        camRow,
        camPlayRow,
        camSpeedRow,
        el("div", { class: "svc-title", text: "sampler" }),
        detail,
        workers,
        el("div", { class: "svc-actions" }, btn),
        actual,
        overlay,
    );
}

function actualText(summary) {
    const mb = (summary.bytes / 1e6).toFixed(1);
    return `actual: ${summary.splats.toLocaleString()} · ${mb} MB`;
}

function readParams() {
    return {
        detail: Number(inputs.detail.value),
        workers: Math.round(Number(inputs.workers.value)),
    };
}

async function resplat() {
    if (!current) return;
    const body = readParams();
    if (inputs._btn) inputs._btn.disabled = true;
    setStatus("re-splatting…", "var(--purple)");
    try {
        await api.splatStage3Start(
            current.run,
            current.slot,
            current.model,
            body,
        );
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
            st = await api.splatStage3Status(
                current.run,
                current.slot,
                current.model,
            );
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
            // A re-splat regenerates the base cloud → back to the surfels source.
            splatSource = "surfels";
            syncViewButtons();
            const bust = `?t=${Date.now()}`;
            loadClouds(
                openSeq,
                api.absUrl(st.url + bust),
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
        fetch(api.absUrl(s4.patches_url + `?t=${Date.now()}`), {
            cache: "no-store",
        }).then((r) => r.arrayBuffer()),
        fetch(api.absUrl(s4.patch_views_url + `?t=${Date.now()}`), {
            cache: "no-store",
        }).then((r) => r.json()),
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
    setOverlay(
        `patches: ${(patchViews.length || 0).toLocaleString()} (click one)`,
    );
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

// ---- camera-position overlay (Stage-4 cameras.json → placed positions) ------

// One cyan disk per placed camera. View-only (no picking), X-rayed through the
// splat like the patch overlay so the whole coverage rig reads at a glance.
function buildCameras(cams) {
    const n = cams.length;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
        const p = cams[i].pos;
        pos[i * 3] = p[0];
        pos[i * 3 + 1] = p[1];
        pos[i * 3 + 2] = p[2];
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    // Opaque disks (alphaTest cutout) that WRITE depth. The viewer renders the
    // scene first, then the splats depth-test against it — writing depth is what
    // lets a point in front of the splat survive the splat pass instead of being
    // overpainted, while a point genuinely behind geometry is correctly occluded.
    const mat = new THREE.PointsMaterial({
        size: 0.3,
        color: 0x5ce6ff,
        sizeAttenuation: true,
        map: patchSprite(),
        alphaTest: 0.5,
    });
    return new THREE.Points(geo, mat);
}

// Fetch the Stage-4 plan (cameras.json) for the open cell and build the overlay.
async function ensureCameras() {
    if (cameraPoints) return true;
    const c = current;
    setOverlay("loading cameras…");
    const s4 = await api.splatStage4Status(c.run, c.slot, c.model);
    if (s4.status !== "done" || !s4.url) {
        setOverlay("plan cameras first (Stage 4)");
        return false;
    }
    const seq = openSeq;
    const plan = await fetch(api.absUrl(s4.url + `?t=${Date.now()}`), {
        cache: "no-store",
    }).then((r) => r.json());
    if (seq !== openSeq || !viewer) return false;
    const cams = plan.cameras || [];
    cameraPoints = buildCameras(cams);
    cameraPoints.visible = false;
    viewer.threeScene.add(cameraPoints);
    setCamReveal(camTotal()); // start fully revealed; playback resets to 0
    setOverlay(`cameras: ${cams.length.toLocaleString()} placed`);
    return true;
}

async function setCameras(on) {
    if (on) {
        const ok = await ensureCameras();
        if (ok && cameraPoints) {
            cameraPoints.visible = true;
            if (!camPlaying) setCamReveal(camTotal()); // static toggle = show all
        }
    } else {
        stopCamPlay();
        if (cameraPoints) cameraPoints.visible = false;
    }
}

// ---- greedy-order playback (reveal camera points in Stage-4 selection order) --

function camTotal() {
    const a = cameraPoints && cameraPoints.geometry.getAttribute("position");
    return a ? a.count : 0;
}

// Reveal the first `k` cameras (greedy order) via the draw range, force a viewer
// frame (it only auto-renders on camera motion), and update the count readout.
function setCamReveal(k) {
    const total = camTotal();
    camRevealed = Math.max(0, Math.min(total, k));
    const n = Math.floor(camRevealed);
    if (cameraPoints) {
        cameraPoints.geometry.setDrawRange(0, n);
        viewer?.forceRenderNextFrame?.();
    }
    if (inputs && inputs._camPlayVal) {
        inputs._camPlayVal.textContent = !total
            ? ""
            : n >= 1000
              ? `${(n / 1000).toFixed(1)}k`
              : String(n);
    }
}

function updateCamPlayBtn() {
    if (inputs && inputs._camPlayBtn) {
        inputs._camPlayBtn.textContent = camPlaying ? "⏸ pause" : "▶ play";
    }
}

function stopCamPlay() {
    camPlaying = false;
    if (camPlayReq) cancelAnimationFrame(camPlayReq);
    camPlayReq = null;
    updateCamPlayBtn();
}

function camPlayTick(now) {
    if (!camPlaying || !cameraPoints || !viewer) {
        stopCamPlay();
        return;
    }
    const dt = camPlayLast ? Math.min(0.1, (now - camPlayLast) / 1000) : 0;
    camPlayLast = now;
    const speed =
        Number(inputs && inputs.camspeed && inputs.camspeed.value) || 200;
    setCamReveal(camRevealed + speed * dt); // points/sec, framerate-independent
    if (camRevealed >= camTotal()) {
        stopCamPlay(); // reached the end — leave every point revealed
        return;
    }
    camPlayReq = requestAnimationFrame(camPlayTick);
}

// Play/pause the greedy-order reveal. Ensures the overlay is loaded + visible,
// restarts from the beginning when already at the end, else resumes.
async function toggleCamPlay() {
    if (camPlaying) {
        stopCamPlay();
        return;
    }
    const ok = await ensureCameras();
    if (!ok || !cameraPoints) return;
    cameraPoints.visible = true;
    if (inputs && inputs.cameras) inputs.cameras.checked = true;
    if (camRevealed >= camTotal()) setCamReveal(0); // at the end → restart
    camPlaying = true;
    camPlayLast = 0;
    updateCamPlayBtn();
    camPlayReq = requestAnimationFrame(camPlayTick);
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
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: "8px",
    });
    head.appendChild(
        el("span", {
            text: `patch #${index} · ${views.length} view${views.length === 1 ? "" : "s"}`,
        }),
    );
    const x = el("button", { text: "✕", onclick: hidePatchModal });
    Object.assign(x.style, {
        background: "transparent",
        color: "#fff",
        border: "none",
        cursor: "pointer",
        fontSize: "15px",
        lineHeight: "1",
    });
    head.appendChild(x);
    patchModalEl.appendChild(head);

    if (!views.length) {
        patchModalEl.appendChild(
            el("div", {
                class: "muted",
                text: "no camera covers this patch (occlusion-culled)",
            }),
        );
    } else if (!refsBase) {
        patchModalEl.appendChild(
            el("div", {
                class: "muted",
                text: "render references (Stage 5) to see images",
            }),
        );
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
                width: "100%",
                display: "block",
                borderRadius: "4px",
                background: "#000",
            });
            const cap = el("figcaption", { text: id });
            Object.assign(cap.style, {
                fontSize: "11px",
                opacity: "0.7",
                marginTop: "2px",
            });
            fig.appendChild(img);
            fig.appendChild(cap);
            patchModalEl.appendChild(fig);
        }
    }
    patchModalEl.style.display = "block";
}

// ---- pipeline stepper (the side panel: run/re-run each stage, gated) ---------

// The LOCAL splat stages in dependency order (1-5): they all run on this machine.
// Stage 1's status is the cell itself; Stages 2-5 live on `cell.stageN`. A stage is
// runnable only once the previous is done; re-running a done stage REVERTS
// everything after it (server-side). Stage 4 (cameras) + Stage 5 (references) run
// locally here so the pipeline can be driven step-by-step and its refs inspected
// before committing to a train. Stage 6 (the fine-tune) still runs REMOTELY via the
// "train on modal" job (see renderModal), which re-plans/re-renders 4-5 on the A100
// as part of that one-shot remote run.
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
    3: (r, s, m) => api.splatStage3Start(r, s, m, readParams()),
    4: (r, s, m) => api.splatStage4Start(r, s, m),
    // Stage 5 resumes by default (renders only the views still missing on disk); an
    // explicit re-run of a DONE stage passes restart so the server wipes refs/ and
    // re-renders every view from scratch.
    5: (r, s, m, restart) => api.splatStage5Start(r, s, m, { restart }),
};

function stageState(cell, n) {
    if (!cell) return {};
    return n === 1 ? cell : cell[`stage${n}`] || {};
}

function stageDone(cell, n) {
    return stageState(cell, n).status === "done";
}

function anyStageRunning(cell) {
    // The remote train job (stages 4-6) keeps the poll alive so its live phase +
    // training heartbeat stream, and so completion is caught for the auto-switch.
    if (cell && cell.modal && cell.modal.status === "running") return true;
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
    // Clicking a DONE stage is an explicit re-run: wipe its outputs and start fresh
    // (Stage 5's restart). A not-done stage runs, or for Stage 5 resumes from the
    // views already on disk after an unexpected stop.
    const restart = stageDone(cellStatus, n);
    setStatus(`starting stage ${n}…`, "var(--purple)");
    try {
        await STAGE_START[n](c.run, c.slot, c.model, restart);
    } catch (e) {
        setStatus(`stage ${n} failed: ${e.message}`, "var(--red)");
        return;
    }
    pollStages();
}

// Latest per-stage status object for the open cell (or null on transient failure).
async function fetchCellStatus() {
    if (!current) return null;
    const payload = await api.splatStageCells(current.run);
    return (payload.cells || []).find(
        (c) => c.slot === current.slot && c.model === current.model,
    );
}

// Show the surfel cloud once Stage 3 is done (mirrors pollStages' cloud load).
async function maybeLoadCloud(seq) {
    const s3 = cellStatus && cellStatus.stage3;
    if (s3 && s3.status === "done" && s3.url && !cloudLoaded) {
        cloudLoaded = true;
        const bust = `?t=${Date.now()}`;
        await loadClouds(
            seq,
            api.absUrl(s3.url + bust),
            s3.summary,
            captureCamera(),
        );
    }
}

// Poll one stage until it leaves the running state, re-rendering each tick.
// Resolves on "done", throws on "error", returns early if the viewer switched.
async function waitStageDone(n, seq) {
    while (seq === openSeq && current) {
        await sleep(POLL_MS);
        if (seq !== openSeq || !current) return;
        let cell;
        try {
            cell = await fetchCellStatus();
        } catch {
            continue; // transient — retry next tick
        }
        if (seq !== openSeq) return;
        cellStatus = cell;
        renderStepper();
        const st = stageState(cell, n);
        if (st.status === "done") return;
        if (st.status === "error")
            throw new Error(st.error || `stage ${n} failed`);
    }
}

// Run stages 1→5 in order, from the first not-yet-done through Stage 5, waiting for
// each before the next. Skips completed stages; stops (and reports) on the first
// error. Drives its own polling, so the interval poll is paused meanwhile.
async function runAll() {
    if (!current || runningAll) return;
    const seq = openSeq;
    runningAll = true;
    stopPoll();
    renderStepper();
    try {
        for (const stage of STAGES) {
            if (seq !== openSeq || !current) return;
            let cell;
            try {
                cell = await fetchCellStatus();
            } catch {
                throw new Error("cell status unavailable");
            }
            if (seq !== openSeq) return;
            cellStatus = cell;
            if (stageDone(cell, stage.n)) {
                renderStepper();
                continue;
            }
            if (stage.n > 1 && !stageDone(cell, stage.n - 1)) {
                throw new Error(`stage ${stage.n - 1} did not complete`);
            }
            if (stage.n <= 3) cloudLoaded = false; // re-running ≤3 reverts the cloud
            setStatus(
                `run all: stage ${stage.n} (${stage.label})…`,
                "var(--purple)",
            );
            await STAGE_START[stage.n](
                current.run,
                current.slot,
                current.model,
            );
            await waitStageDone(stage.n, seq);
            if (stage.n >= 3) await maybeLoadCloud(seq);
        }
        if (seq !== openSeq || !current) return;
        // All LOCAL stages (1–5) done — including the camera plan + reference
        // renders, ready to inspect. Stage 6 (the fine-tune) stays on the Modal
        // A100: launch it from the "train on modal" panel when you want the
        // trained splat (that remote run re-plans/re-renders 4–5 itself).
        setStatus("run all: local stages 1–5 done", "var(--green)");
    } catch (e) {
        if (seq === openSeq)
            setStatus(`run all stopped: ${e.message}`, "var(--red)");
    } finally {
        runningAll = false;
        if (seq === openSeq) {
            const cell = await fetchCellStatus().catch(() => null);
            if (cell) {
                cellStatus = cell;
                await maybeLoadCloud(seq);
                renderStepper();
                if (anyStageRunning(cell)) pollStages();
            } else {
                renderStepper();
            }
        }
    }
}

// The splat pipeline always samples + renders the per-cell SPLAT asset tier
// (built on demand by the stage jobs — optimize.mjs --preset splat), so there
// is no source picker: the stepper row shows the tier build as a stage phase.
function renderSourceRow() {
    const row = el("div", { class: "svc-step" });
    row.appendChild(el("span", { class: "svc-step-n muted", text: "◆" }));
    row.appendChild(el("span", { class: "svc-step-label", text: "source" }));
    row.appendChild(
        el("span", {
            class: "muted",
            text: "splat tier (decimated · KTX2/ETC1S)",
        }),
    );
    return row;
}

// Format a capture throughput (images/second) compactly, or null when there's
// nothing meaningful to show yet.
function fmtRate(r) {
    if (r == null || !isFinite(r) || r <= 0) return null;
    return r >= 10 ? String(Math.round(r)) : r.toFixed(1);
}

// A rough ETA string from the remaining view count and the live rate.
function fmtEta(remaining, rate) {
    if (!rate || rate <= 0 || remaining <= 0) return null;
    const s = remaining / rate;
    if (s < 90) return `${Math.round(s)}s`;
    if (s < 5400) return `${Math.round(s / 60)}m`;
    return `${(s / 3600).toFixed(1)}h`;
}

// Per-stage wall-clock for the CURRENT running step, so the number-less marker
// steps (Stage 2's reduce/fill/clearance/write, Stage 4's load/patches/write —
// single bulk passes with no per-item count to stream) show a live elapsed
// timer instead of a frozen "…". Keyed by stage number → {phase, t0}; resets
// when the step changes or the stage stops running (see renderStepper).
const _stepClocks = new Map();

function _stepElapsedSecs(stageN, phase) {
	const key = phase || "run";
	const rec = _stepClocks.get(stageN);
	if (!rec || rec.phase !== key) {
		_stepClocks.set(stageN, { phase: key, t0: Date.now() });
		return 0;
	}
	return Math.round((Date.now() - rec.t0) / 1000);
}

function fmtStepSecs(s) {
	return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
}

function renderStepper() {
	const box = inputs && inputs._stepper;
	if (!box) return;
	box.replaceChildren();
	box.appendChild(renderSourceRow());
	const cell = cellStatus;
	for (const stage of STAGES) {
		const st = stageState(cell, stage.n);
		const done = st.status === "done";
		const running = st.status === "running" || st.status === "pending";
		const gated = stage.n > 1 && !stageDone(cell, stage.n - 1);
		if (!running) _stepClocks.delete(stage.n); // reset the step timer between runs
		const row = el("div", { class: "svc-step" });
		row.appendChild(
			el("span", { class: "svc-step-n muted", text: `${stage.n}` }),
		);
		const labelEl = el("span", {
			class: "svc-step-label",
			text: stage.label,
		});
		row.appendChild(labelEl);
		let btn;
		let extra = null; // trailing muted info (e.g. Stage-5 live throughput)
		if (running) {
			// `current_id` is the live sub-step: Stage 4's phase (load / patches /
			// coverage / select / write) or the object/view id the others are on. Show
			// it next to the label so the long coverage phase isn't an opaque wait.
			// Stage 5 during `render` shows the throughput instead of the (noisy)
			// per-view id — its named warm-up phases (plan / launch) still show.
			const phase =
				stage.n === 5
					? st.phase &&
						!["render", "done", "pending", "plan"].includes(st.phase)
						? st.phase
						: ""
					: st.current_id && st.current_id !== "plan"
						? String(st.current_id)
						: "";
			if (phase)
				labelEl.textContent = `${stage.label} · ${phase.slice(0, 22)}`;
			const prog = st.total
				? `${st.done}/${st.total}`
				: st.done
					? String(st.done)
					: "…";
			btn = el("button", {
				class: "splat-stage2-btn",
				disabled: true,
				text: prog,
				title:
					`${phase} ${st.total ? `${st.done}/${st.total}` : ""}`.trim() ||
					"running",
			});
			// Live server-computed throughput next to the running step. Stage 5 =
			// images/second (with session avg); Stages 2 & 4 = the current step's
			// actions/second — objects voxelized / shells extracted (Stage 2),
			// candidates ray-marched / cameras picked (Stage 4). Single-pass marker
			// steps don't stream a count, so no rate shows for them.
			const rate =
				stage.n === 5 || stage.n === 4 || stage.n === 2
					? fmtRate(st.rate)
					: null;
			if (rate) {
				const remaining = (st.total || 0) - (st.done || 0);
				const eta = fmtEta(remaining, st.rate);
				if (stage.n === 5) {
					const avg = fmtRate(st.rate_avg);
					extra = el("span", {
						class: "muted",
						text: eta ? `${rate} img/s · ETA ${eta}` : `${rate} img/s`,
						title: avg
							? `live ${rate} img/s · session avg ${avg} img/s`
							: `${rate} img/s`,
					});
				} else {
					extra = el("span", {
						class: "muted",
						text: eta ? `${rate}/s · ETA ${eta}` : `${rate}/s`,
						title: `${phase || "step"} · ${rate} actions/s`,
					});
				}
			} else {
				// A number-less running step (e.g. Stage 2 reduce / fill / clearance /
				// write): a single bulk pass with no per-item count to stream, so show
				// a live elapsed timer — it reads as working, not frozen at "…".
				const secs = _stepElapsedSecs(stage.n, phase || st.phase || "run");
				extra = el("span", {
					class: "muted",
					text: fmtStepSecs(secs),
					title: `${phase || st.phase || "step"} · ${fmtStepSecs(secs)} elapsed`,
				});
			}
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
				disabled: runningAll,
				text: done ? "re-run" : stage.verb,
				title: done
					? stage.n === 5
						? "re-run — wipe all references and re-render from scratch"
						: "re-run — discards every later stage"
					: `run stage ${stage.n}`,
				onclick: () => runStage(stage.n),
			});
		}
		// Planned reference-image count (image files Stage 5 will create), shown before capture.
		if (stage.n === 5 && !running && !extra) {
			const planned = stageState(cell, 4).summary?.views;
			if (planned != null)
				extra = el("span", {
					class: "muted",
					text: `${fmtInt(planned)} views`,
					title: "reference images this render will create (from the Stage-4 plan)",
				});
		}
		if (st.status === "error") {
			btn.classList.add("err");
			btn.title = st.error || "failed — click to retry";
		}
		row.appendChild(btn);
		if (extra) row.appendChild(extra);
		box.appendChild(row);
	}
	box.appendChild(renderRunAll(cell));
	renderModal(cell);
	renderCoverage();
	updateSourceAvail();
}

// "run all" control row: runs local stages 1→5 in order from the first
// not-yet-done (assemble → free space → surfels → cameras → references). Stage 6
// (the fine-tune) is launched separately from the "train on modal" panel.
// Disabled while anything is running, or once all local stages are done.
function renderRunAll(cell) {
    const busy = runningAll || anyStageRunning(cell);
    const allDone = STAGES.every((s) => stageDone(cell, s.n));
    const row = el("div", { class: "svc-step" });
    row.appendChild(el("span", { class: "svc-step-n muted", text: "▶" }));
    row.appendChild(el("span", { class: "svc-step-label", text: "run all" }));
    row.appendChild(
        el("button", {
            class: "splat-stage2-btn",
            disabled: busy || allDone,
            text: runningAll ? "running…" : allDone ? "all done" : "run 1–5",
            title: "run local stages 1–5 in order (assemble → free space → surfels → cameras → references)",
            onclick: () => runAll(),
        }),
    );
    return row;
}

// The client-owned training-policy controls (convergence bounds + plateau
// sensitivity + batch) for the "train on modal" panel. Each input writes STRAIGHT
// back to `modalTrainCfg`, so values persist across the panel's re-renders and
// are read by `startModalTrain`; defaults live here (client), never on the
// server. `disabled` greys them out until Stage 3 is ready.
function modalCfgControls(disabled) {
    const numRow = (label, key, step, min, max, title) => {
        const input = el("input", {
            type: "number", value: modalTrainCfg[key], step, min, max,
            class: "svc-num", title, disabled,
            oninput: () => {
                const v = Number(input.value);
                if (!Number.isNaN(v)) modalTrainCfg[key] = v;
            },
        });
        input.style.width = "60px";
        return el(
            "div",
            { class: "svc-row" },
            el("span", { class: "svc-lab", text: label, title }),
            input,
        );
    };
    return el(
        "div",
        { class: "svc-modal-cfg" },
        numRow("epochs", "epochs", 1, 1, 200,
            "passes over the view set → iterations = epochs × number of views"),
        numRow("batch", "batch", 1, 1, 32,
            "views per optimizer step (fills the GPU); speed knob at constant work"),
    );
}

// Remote-train row + live log. One "train on modal" button runs stages 4-6 on
// the A100 (cameras → references → fine-tune) and pulls trained.ply back; its
// live phase (push / spawn / plan / refs / train / pull) + the training
// heartbeat (`stage · step/total · loss=… it/s`) arrive through the same cells
// poll, and on completion the "trained" view auto-opens.
function renderModal(cell) {
    const box = inputs && inputs._modal;
    if (!box) return;
    const st = (cell && cell.modal) || {};
    const status = st.status || "idle";
    const running = status === "running";
    const s3done = stageDone(cell, 3);
    box.replaceChildren();
    const row = el("div", { class: "svc-step" });
    row.appendChild(el("span", { class: "svc-step-n muted", text: "△" }));
    row.appendChild(
        el("span", {
            class: "svc-step-label",
            text: running ? `modal · ${st.phase || "run"}` : "train on modal",
        }),
    );
    let btn;
    if (running) {
        // During training the heartbeat carries step/total; other phases (push /
        // spawn / refs / pull) show the phase word as the live progress.
        const prog =
            st.stage === "train" && st.total
                ? `${fmtInt(st.done)}/${fmtInt(st.total)}`
                : st.phase || "…";
        btn = el("button", {
            class: "splat-stage2-btn",
            disabled: true,
            text: prog,
            title: "training on the A100 — see the live log below",
        });
    } else {
        btn = el("button", {
            class: `splat-stage2-btn${status === "done" ? " view" : ""}`,
            disabled: !s3done || runningAll,
            text: status === "done" ? "re-train" : "train on modal",
            title: s3done
                ? "plan cameras + render references + fine-tune on the Modal A100 (stages 4–6), then pull trained.ply"
                : "sample the surfel cloud first (Stage 3)",
            onclick: () => startModalTrain(status === "done"),
        });
    }
    if (status === "error") {
        btn.classList.add("err");
        btn.title = st.error || "remote train failed — click to retry";
    }
    row.appendChild(btn);
    box.appendChild(row);

    // Client-owned training policy — editable while idle (hidden mid-run). These
    // values are the ONLY source of training-length defaults; the server forwards
    // them untouched.
    if (!running) box.appendChild(modalCfgControls(!s3done || runningAll));

    const info =
        status === "done"
            ? "trained.ply ready — switch to the “trained” view"
            : status === "error"
              ? st.error || "remote train failed"
              : running
                ? `${st.phase || "running"}${st.msg ? ` — ${st.msg}` : ""}`
                : s3done
                  ? "ready — runs stages 4–6 on the A100"
                  : "needs the surfel cloud (Stage 3)";
    const sub = el("div", { class: "muted", text: info });
    sub.style.fontSize = "12px";
    box.appendChild(sub);

    // The camera plan is pulled back the moment stage 4 finishes (while stages
    // 5-6 keep running remotely). Reveal the camera-position overlay ONCE when
    // it arrives, so the rig is viewable over the surfel cloud mid-run without
    // waiting for training. One-shot + guarded, so a manual toggle-off sticks.
    if (running && st.plan_pulled && !modalPlanShown) {
        modalPlanShown = true;
        if (inputs && inputs.cameras && !inputs.cameras.checked) {
            inputs.cameras.checked = true;
            void setCameras(true);
        }
        setStatus(
            "camera plan ready — showing camera positions (training continues)",
            "var(--green)",
        );
    }

    // Debug sample images pulled from the render mid-run — a quick visual check
    // that the picture-taking is correct. Click a thumbnail to open it full-size.
    const samples = st.sample_urls || [];
    if (samples.length) {
        const strip = el("div");
        Object.assign(strip.style, {
            display: "flex",
            flexWrap: "wrap",
            gap: "3px",
            marginTop: "5px",
        });
        for (const u of samples) {
            const img = el("img");
            img.src = api.absUrl(u);
            img.loading = "lazy";
            img.title = u.split("/").pop();
            Object.assign(img.style, {
                width: "46px",
                height: "46px",
                objectFit: "cover",
                borderRadius: "3px",
                background: "#000",
                cursor: "pointer",
            });
            img.onclick = () => window.open(api.absUrl(u), "_blank");
            strip.appendChild(img);
        }
        box.appendChild(
            el("div", {
                class: "muted",
                text: `render samples (${samples.length})`,
                style: "font-size:11px;margin-top:4px",
            }),
        );
        box.appendChild(strip);
    }

    // Accumulate distinct heartbeat lines into the live log pane (a scrolling
    // history, so training-step lines read as progress, not a single value).
    const line = running && st.msg ? `[${st.stage || st.phase}] ${st.msg}` : null;
    if (line && modalLog[modalLog.length - 1] !== line) {
        modalLog.push(line);
        if (modalLog.length > 400) modalLog.shift();
    }
    const log = inputs._log;
    if (log) {
        const following =
            Math.abs(log.scrollHeight - log.clientHeight - log.scrollTop) < 40;
        log.textContent = modalLog.length
            ? modalLog.join("\n")
            : running
              ? "starting…"
              : "";
        if (following) log.scrollTop = log.scrollHeight;
    }

    // Catch the running → done transition: trained.ply has been pulled (the
    // supervisor sets 'done' only after the pull), so stage6.url is now set —
    // refresh the toggles and open the trained view for an immediate compare.
    if (modalPrev === "running" && status === "done") {
        updateSourceAvail();
        if (trainedUrl) void setView("trained");
    }
    modalPrev = status;
}

// Launch the Modal remote train (stages 4-6) for the open cell, then poll — the
// cells status streams its live phase + training heartbeat.
async function startModalTrain(restart = false) {
    if (!current) return;
    modalLog = [];
    modalPlanShown = false;
    setStatus(
        restart ? "restarting remote train…" : "starting remote train…",
        "var(--purple)",
    );
    try {
        await api.splatModalStart(current.run, current.slot, current.model, {
            restart,
            epochs: modalTrainCfg.epochs,
            batch: modalTrainCfg.batch,
        });
    } catch (e) {
        setStatus(`remote train failed to start: ${e.message}`, "var(--red)");
        return;
    }
    pollStages();
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
    const satPct =
        cov.satisfied_pct != null
            ? cov.satisfied_pct
            : patches
              ? Math.round((100 * sat) / patches)
              : 0;
    const seenPct = patches ? Math.round((100 * seen) / patches) : 0;
    const satColor =
        satPct >= 90 ? "var(--green)" : satPct >= 70 ? "#d0a020" : "var(--red)";

	box.appendChild(el("div", { class: "svc-title", text: "coverage" }));
	const rows = [
		["patches", fmtInt(patches), null],
		[
			"cameras · candidates",
			`${fmtInt(sum.cameras)} · ${fmtInt(sum.candidates)}`,
			null,
		],
		["views (image files)", fmtInt(sum.views), null],
		["satisfied (≥K angles)", `${fmtInt(sat)} · ${satPct}%`, satColor],
		["under-covered", fmtInt(Math.max(0, seen - sat)), null],
		[
			"occlusion-culled",
			fmtInt(cov.occlusion_culled),
			cov.occlusion_culled ? "var(--red)" : null,
		],
		["reached (≥1 view)", `${fmtInt(seen)} · ${seenPct}%`, null],
		[
			"mean angles/patch",
			cov.mean_angles_seen != null ? String(cov.mean_angles_seen) : "?",
			null,
		],
		[
			"target K / sectors",
			`${sum.angles_per_patch ?? "?"} / ${sum.angular_sectors ?? "?"}`,
			null,
		],
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
                s3.summary,
                captureCamera(),
            );
        } else if ((!s3 || s3.status !== "done") && cloudLoaded) {
            cloudLoaded = false; // surfels reverted → fall back to the mesh view
            void openMeshView(openSeq);
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
        await loadClouds(seq, api.absUrl(s3.url), s3.summary, null);
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
    runningAll = false;
    cellStatus = null;
    splatSource = "surfels";
    trainedUrl = null;
    sogUrl = null;
    modalPrev = null;
    modalLog = [];
    modalPlanShown = false;
    overlay.classList.add("open");
    subEl.textContent =
        opts.label || `${opts.slot || ""} · ${opts.model || ""}`;
    assetSource = null;
    buildControls(opts.summary || null);
    renderStepper();
    void refreshAssetSource();
    if (opts.url) {
        cloudLoaded = true;
        setStatus("loading…", "var(--purple)");
        await loadClouds(seq, opts.url, opts.summary, null);
        void refreshStepper(seq);
    } else {
        // MESH-FIRST: no surfel cloud yet, but the stage-2 voxel grid is
        // measured off the original meshes — open on them so the voxel/free
        // overlays are inspectable before Stage 3 ever runs.
        setStatus("no splat yet — showing original mesh", "");
        await refreshStepper(seq); // may load the cloud if Stage 3 is done
        if (seq === openSeq && !cloudLoaded) await openMeshView(seq);
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
        position: "absolute",
        top: "0",
        right: "0",
        height: "100%",
        width: "340px",
        boxSizing: "border-box",
        background: "rgba(18,18,22,0.94)",
        borderLeft: "1px solid rgba(255,255,255,0.12)",
        overflowY: "auto",
        padding: "12px",
        display: "none",
        zIndex: "30",
        color: "#fff",
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
            const moved = Math.hypot(
                ev.clientX - _downXY[0],
                ev.clientY - _downXY[1],
            );
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
