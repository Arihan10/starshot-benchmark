// One-shot bench viewer — an N-way live comparison grid for the /oneshot
// track. Pick a scene (slot) and a set of models; "launch scene" starts every
// checked model in parallel, and each model renders into its own tile so the
// builds can be watched side by side.
//
// Architecture notes:
//   * ONE WebGL context for the whole grid (browsers cap contexts well below
//     our model count): a fullscreen canvas sits behind the tile grid and
//     each tile is scissor-rendered onto it every frame with its own
//     THREE.Scene + camera + OrbitControls (the three.js multi-view pattern).
//   * ONE multiplexed SSE per (run, slot): /oneshot/slots/{slot}/events-all
//     tags every event with its model alias — N separate EventSources would
//     exhaust the browser's ~6-connections-per-origin budget.
//   * Per tile: bbox/proxy wireframes, placed-library GLBs (KTX2+Meshopt),
//     camera fit, hover tooltip + click selection, right-click hide, and an
//     info panel with every design call's inputs/outputs (per step for v4).
//     All ported from the proven single-view build.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";

const SERVER_URL = document
  .querySelector('meta[name="server-url"]')
  .getAttribute("content");

const SLOT_KEY = "starshot.oneshot.slot";
const RUN_KEY = "starshot.oneshot.run";
const CHECKED_KEY = "starshot.oneshot.checkedModels";
const BBOX_KEY = "starshot.oneshot.bboxes";
const MESH_KEY = "starshot.oneshot.meshes";
const SOLID_KEY = "starshot.oneshot.solids";

const hostEl = document.getElementById("canvas-host");
const statusEl = document.getElementById("status");
const runPickerEl = document.getElementById("run-picker");
const runVersionEl = document.getElementById("run-version");
const runNewEl = document.getElementById("run-new");
const slotBarEl = document.getElementById("slot-bar");
const modelBarEl = document.getElementById("model-bar");
const launchEl = document.getElementById("launch");
const bboxToggleEl = document.getElementById("bbox-toggle");
const meshToggleEl = document.getElementById("mesh-toggle");
const solidToggleEl = document.getElementById("solid-toggle");
const fitEl = document.getElementById("fit");
const tilesEl = document.getElementById("tiles");

function setStatus(text, kind = "") {
  statusEl.textContent = text;
  statusEl.className = kind;
}

// --- shared renderer + loader ---------------------------------------------------

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x101114);
hostEl.appendChild(renderer.domElement);

window.addEventListener("resize", () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// KTX2/Basis + Meshopt decoders: placed library assets are compressed GLBs.
const DECODE_WORKERS = Math.min(4, navigator.hardwareConcurrency || 2);
const ktx2 = new KTX2Loader()
  .setTranscoderPath("/vendor/three/examples/jsm/libs/basis/")
  .setWorkerLimit(DECODE_WORKERS)
  .detectSupport(renderer);
MeshoptDecoder.useWorkers(DECODE_WORKERS);
const loader = new GLTFLoader().setKTX2Loader(ktx2).setMeshoptDecoder(MeshoptDecoder);

// --- shared tooltip --------------------------------------------------------------

const TOOLTIP_KIND_COLOR = { zone: "#9ad4ff", object: "#8bd17c", frame: "#7fb3d5" };
const tooltip = document.createElement("div");
tooltip.style.cssText = [
  "position: fixed",
  "display: none",
  "z-index: 40",
  "pointer-events: none",
  "background: rgba(12, 13, 16, 0.95)",
  "border: 1px solid #2a2d35",
  "border-radius: 5px",
  "padding: 7px 9px",
  "font: 11px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace",
  "color: #e6e6e6",
  "max-width: 360px",
  "white-space: pre-wrap",
].join("; ");
document.body.appendChild(tooltip);

function placeTooltip(clientX, clientY) {
  tooltip.style.display = "block";
  tooltip.style.left = "0px";
  tooltip.style.top = "0px";
  const w = tooltip.offsetWidth;
  const h = tooltip.offsetHeight;
  const pad = 12;
  let x = clientX + pad;
  let y = clientY + pad;
  if (x + w > window.innerWidth) x = clientX - pad - w;
  if (y + h > window.innerHeight) y = clientY - pad - h;
  tooltip.style.left = `${Math.max(0, x)}px`;
  tooltip.style.top = `${Math.max(0, y)}px`;
}

// --- global view state ------------------------------------------------------------

let bboxesShown = localStorage.getItem(BBOX_KEY) !== "0";
let meshesShown = localStorage.getItem(MESH_KEY) !== "0";
let solidsShown = localStorage.getItem(SOLID_KEY) === "1";

const BBOX_COLOR = { zone: 0xff3b3b, object: 0x6bd96e, frame: 0x7fb3d5 };
const BBOX_COLOR_HOVER = 0xffe14a;
const BBOX_COLOR_SELECTED = 0x4af0e0;
const BBOX_DIM_OPACITY = 0.35;
const PROXY_BASE_OPACITY = 0.5;
const PROXY_DIM_OPACITY = 0.2;
const CLICK_MAX_MOVE_PX = 4;
const CLICK_MAX_DURATION_MS = 400;

// Stable, well-separated colors for any count of distinct floor-plan names:
// golden-angle hue stepping with a small lightness wobble between neighbours.
// The floor-plan panel (canvas) and the solid filled boxes (three.js) both
// derive from planHSL(i), so the same name gets the SAME color in both.
function planHSL(i) {
  return { h: ((i * 137.508) % 360) / 360, s: 0.62, l: (52 + (i % 3) * 8) / 100 };
}
function planColor(i) {
  const { h, s, l } = planHSL(i);
  return `hsl(${(h * 360).toFixed(2)} ${(s * 100).toFixed(0)}% ${(l * 100).toFixed(0)}%)`;
}
function planColorThree(i) {
  const { h, s, l } = planHSL(i);
  return new THREE.Color().setHSL(h, s, l);
}

// The design calls shown in the info panel. Cells also stream library_match
// calls (one per object) — cheap retrieval, not part of the benchmark surface.
const DESIGN_STEPS = new Set(["oneshot_grid", "oneshot_scene"]);

// --- per-model cell view -----------------------------------------------------------

const STATUS_LIFECYCLE = {
  idle: ["start", "paused", "Run this cell"],
  paused: ["resume", "paused", "Resume the interrupted cell"],
  error: ["retry", "error", "Retry the failed cell"],
  running: ["pause", "running", "Pause this cell"],
  done: [null],
};

function createCellView(alias) {
  // --- tile DOM ---
  const tile = document.createElement("div");
  tile.className = "tile";
  const head = document.createElement("div");
  head.className = "tile-head";
  const dot = document.createElement("span");
  dot.className = "slot-dot status-idle";
  const aliasEl = document.createElement("span");
  aliasEl.className = "alias";
  aliasEl.textContent = alias;
  const metaEl = document.createElement("span");
  metaEl.className = "meta";
  const lifecycleBtn = document.createElement("button");
  lifecycleBtn.className = "tile-btn";
  const resetBtn = document.createElement("button");
  resetBtn.className = "tile-btn danger";
  resetBtn.textContent = "reset";
  resetBtn.title = "Wipe this cell and re-run from scratch";
  const infoBtn = document.createElement("button");
  infoBtn.className = "tile-btn";
  infoBtn.textContent = "info";
  infoBtn.title = "Toggle the design-call panel: the exact system/user prompts sent, reasoning, and the exact model output per step — plus the deterministic solver's exact input/output on grid-only runs";
  head.append(dot, aliasEl, metaEl, lifecycleBtn, resetBtn, infoBtn);
  const viewport = document.createElement("div");
  viewport.className = "tile-viewport";
  // v4 floor-plan companion panel, beside the 3D view so plan and build can
  // be compared at a glance. Stays display:none until a grid plan arrives.
  const planPanel = document.createElement("div");
  planPanel.className = "tile-plan";
  const planNote = document.createElement("div");
  planNote.className = "plan-note";
  planNote.textContent = "stage-1 floor plan · top = back";
  const planCanvas = document.createElement("canvas");
  const planLegend = document.createElement("div");
  planLegend.className = "plan-legend";
  planPanel.append(planNote, planCanvas, planLegend);
  const body = document.createElement("div");
  body.className = "tile-body";
  body.append(viewport, planPanel);
  const overlay = document.createElement("div");
  overlay.className = "tile-overlay";
  tile.append(head, body, overlay);

  // --- three scene ---
  const scene = new THREE.Scene();
  const sceneRoot = new THREE.Group();
  const bboxRoot = new THREE.Group();
  scene.add(sceneRoot, bboxRoot);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x202028, 0.9));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
  dirLight.position.set(8, 12, 6);
  scene.add(dirLight);
  const grid = new THREE.GridHelper(300, 300, 0x44485a, 0x1c1f26);
  grid.material.transparent = true;
  grid.material.opacity = 0.4;
  scene.add(grid);

  const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 5000);
  camera.position.set(10, 7, 12);
  const controls = new OrbitControls(camera, viewport);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 1, 0);

  // --- per-tile state ---
  const bboxes = new Map();   // id -> Box3Helper
  const proxies = new Map();  // id -> proxy wireframe
  const solids = new Map();   // id -> solid filled box mesh
  const models = new Map();   // id -> gltf scene
  // Object-name -> color index, seeded in floor-plan reading order so the
  // solid boxes match the plan panel; names absent from the plan (harness
  // floor/background, non-grid versions) append distinct colors.
  const nameColorIndex = new Map();
  const nodeInfo = new Map(); // id -> {kind, prompt, origin, dimensions, proxyShape, orientation, libraryId}
  const hiddenIds = new Set();
  const objectIds = new Set();
  let gen = 0;
  let highestIndex = -1;
  let fitPending = false;
  let cameraUserMoved = false;
  let controlsInteracting = false;
  let hoveredId = null;
  let selectedId = null;
  let gridText = null;
  let planData = null;
  let errorMessage = null;
  // Design-call payloads by step, in arrival order (v4: grid then scene):
  // the committed cache.llm, or the llm.failed record when the single
  // attempt was rejected (a later commit for the step overwrites it after a
  // manual cell retry). library_match calls share the stream but are
  // retrieval noise, not design.
  const llmCalls = new Map();
  // v4's deterministic grid→boxes step (oneshot.solver event): exact plan
  // consumed + exact placements produced, shown after the LLM steps.
  let solverData = null;

  controls.addEventListener("start", () => {
    cameraUserMoved = true;
    controlsInteracting = true;
    setHovered(null);
  });
  controls.addEventListener("end", () => {
    controlsInteracting = false;
    pointerDirty = true;
  });

  // --- visuals ---
  function disposeMaterial(material) {
    if (!material) return;
    for (const value of Object.values(material)) {
      if (value && value.isTexture) value.dispose();
    }
    material.dispose?.();
  }

  function disposeObject3D(root) {
    root?.traverse?.((n) => {
      if (!n.isMesh) return;
      n.geometry?.dispose?.();
      const mats = Array.isArray(n.material) ? n.material : n.material ? [n.material] : [];
      for (const m of mats) disposeMaterial(m);
    });
  }

  function applyBboxColor(id) {
    if (id === null) return;
    const helper = bboxes.get(id);
    if (!helper) return;
    const base = BBOX_COLOR[nodeInfo.get(id)?.kind ?? "zone"] ?? BBOX_COLOR.zone;
    const color =
      id === selectedId ? BBOX_COLOR_SELECTED
      : id === hoveredId ? BBOX_COLOR_HOVER
      : base;
    helper.material.color.setHex(color);
    const proxy = proxies.get(id);
    if (proxy) proxy.material.color.setHex(color);
  }

  function applyBboxVisibility(id) {
    const visible = id === selectedId || id === hoveredId || bboxesShown;
    const dim = selectedId !== null && id !== selectedId && id !== hoveredId;
    const helper = bboxes.get(id);
    if (helper) {
      helper.visible = visible;
      helper.material.opacity = dim ? BBOX_DIM_OPACITY : 1;
    }
    const proxy = proxies.get(id);
    if (proxy) {
      proxy.visible = visible;
      proxy.material.opacity = dim ? PROXY_DIM_OPACITY : PROXY_BASE_OPACITY;
    }
  }

  function refreshAllBboxVisibility() {
    for (const id of bboxes.keys()) applyBboxVisibility(id);
  }

  function applyModelVisibility(id) {
    const model = models.get(id);
    if (model) model.visible = meshesShown && !hiddenIds.has(id);
  }

  function refreshAllModelVisibility() {
    for (const id of models.keys()) applyModelVisibility(id);
  }

  function applySolidVisibility(id) {
    const solid = solids.get(id);
    if (solid) solid.visible = solidsShown && !hiddenIds.has(id);
  }

  function refreshAllSolidVisibility() {
    for (const id of solids.keys()) applySolidVisibility(id);
  }

  function colorIndexForName(name) {
    const key = name || "";
    if (!nameColorIndex.has(key)) nameColorIndex.set(key, nameColorIndex.size);
    return nameColorIndex.get(key);
  }

  function colorForName(name) {
    return planColorThree(colorIndexForName(name));
  }

  function refreshSolidColors() {
    for (const [id, solid] of solids) {
      solid.material.color.copy(colorForName(nodeInfo.get(id)?.prompt ?? ""));
    }
  }

  function buildSolidBox(origin, dimensions, color) {
    const sx = Math.abs(dimensions[0]) || 0.01;
    const sy = Math.abs(dimensions[1]) || 0.01;
    const sz = Math.abs(dimensions[2]) || 0.01;
    const geom = new THREE.BoxGeometry(sx, sy, sz);
    const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.85, metalness: 0 });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(
      origin[0] + dimensions[0] / 2,
      origin[1] + dimensions[1] / 2,
      origin[2] + dimensions[2] / 2,
    );
    return mesh;
  }

  function buildProxyWireframe(proxyShape, origin, dimensions, color) {
    if (!proxyShape) return null;
    const sx = Math.abs(dimensions[0]);
    const sy = Math.abs(dimensions[1]);
    const sz = Math.abs(dimensions[2]);
    if (sx === 0 || sy === 0 || sz === 0) return null;
    const cx = origin[0] + dimensions[0] / 2;
    const cy = origin[1] + dimensions[1] / 2;
    const cz = origin[2] + dimensions[2] / 2;
    const yMin = Math.min(origin[1], origin[1] + dimensions[1]);
    let geom;
    let anchorY;
    if (proxyShape === "SPHERE") {
      geom = new THREE.SphereGeometry(0.5, 24, 16);
      geom.scale(sx, sy, sz);
      anchorY = cy;
    } else if (proxyShape === "HEMISPHERE") {
      geom = new THREE.SphereGeometry(0.5, 24, 16, 0, Math.PI * 2, 0, Math.PI / 2);
      geom.scale(sx, sy * 2, sz);
      anchorY = yMin;
    } else if (proxyShape === "CAPSULE") {
      const r = Math.min(sx, sz) / 2;
      geom = new THREE.CapsuleGeometry(r, Math.max(0, sy - 2 * r), 8, 24);
      anchorY = cy;
    } else {
      return null;
    }
    const mat = new THREE.MeshBasicMaterial({
      color, wireframe: true, transparent: true, opacity: PROXY_BASE_OPACITY,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(cx, anchorY, cz);
    return mesh;
  }

  function loadBbox(event) {
    const { id, origin, dimensions } = event;
    if (!Array.isArray(origin) || !Array.isArray(dimensions)) return;
    for (const map of [bboxes, proxies]) {
      const prev = map.get(id);
      if (prev) {
        bboxRoot.remove(prev);
        prev.geometry?.dispose?.();
        prev.material?.dispose?.();
        map.delete(id);
      }
    }
    const prevSolid = solids.get(id);
    if (prevSolid) {
      sceneRoot.remove(prevSolid);
      prevSolid.geometry?.dispose?.();
      prevSolid.material?.dispose?.();
      solids.delete(id);
    }
    const kind = event.node_kind ?? "zone";
    const prevInfo = nodeInfo.get(id);
    nodeInfo.set(id, {
      kind,
      prompt: event.prompt ?? "",
      origin,
      dimensions,
      proxyShape: event.proxy_shape ?? null,
      orientation: event.orientation ?? 0,
      libraryId: prevInfo?.libraryId ?? null,
    });
    if (kind === "object") {
      objectIds.add(id);
      updateHeader();
    }
    const fx = origin[0] + dimensions[0];
    const fy = origin[1] + dimensions[1];
    const fz = origin[2] + dimensions[2];
    const box3 = new THREE.Box3(
      new THREE.Vector3(Math.min(origin[0], fx), Math.min(origin[1], fy), Math.min(origin[2], fz)),
      new THREE.Vector3(Math.max(origin[0], fx), Math.max(origin[1], fy), Math.max(origin[2], fz)),
    );
    const helper = new THREE.Box3Helper(box3, BBOX_COLOR[kind] ?? BBOX_COLOR.zone);
    helper.material.transparent = true;
    helper.userData.pickId = id;
    bboxRoot.add(helper);
    bboxes.set(id, helper);
    const proxy = buildProxyWireframe(
      event.proxy_shape, origin, dimensions, BBOX_COLOR[kind] ?? BBOX_COLOR.zone,
    );
    if (proxy) {
      proxy.userData.pickId = id;
      bboxRoot.add(proxy);
      proxies.set(id, proxy);
    }
    if (kind === "object") {
      const solid = buildSolidBox(origin, dimensions, colorForName(event.prompt ?? ""));
      solid.userData.pickId = id;
      sceneRoot.add(solid);
      solids.set(id, solid);
      applySolidVisibility(id);
    }
    applyBboxColor(id);
    applyBboxVisibility(id);
    fitPending = true;
  }

  async function loadModel(event) {
    const myGen = gen;
    try {
      const gltf = await loader.loadAsync(new URL(event.url, SERVER_URL).toString());
      if (myGen !== gen) { disposeObject3D(gltf.scene); return; }
      gltf.scene.traverse((child) => {
        if (child.isMesh && child.material) {
          const mats = Array.isArray(child.material) ? child.material : [child.material];
          for (const m of mats) m.side = THREE.DoubleSide;
        }
      });
      gltf.scene.name = `mesh:${event.id}`;
      gltf.scene.userData.pickId = event.id;
      const prev = models.get(event.id);
      if (prev) {
        sceneRoot.remove(prev);
        disposeObject3D(prev);
      }
      sceneRoot.add(gltf.scene);
      models.set(event.id, gltf.scene);
      applyModelVisibility(event.id);
      fitPending = true;
    } catch (e) {
      console.warn(`[${alias}] mesh load failed for ${event.id}:`, e.message);
    }
  }

  function fitToScene(force = false) {
    if (cameraUserMoved && !force) return;
    let box = new THREE.Box3().setFromObject(sceneRoot);
    if (box.isEmpty()) box = new THREE.Box3().setFromObject(bboxRoot);
    if (box.isEmpty()) return;
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const radius = 0.5 * Math.max(size.x, size.y, size.z);
    if (!isFinite(radius) || radius === 0) return;
    controls.target.copy(center);
    const dist = radius / Math.tan((camera.fov * Math.PI) / 360);
    const dirVec = new THREE.Vector3(1, 0.7, 1).normalize();
    camera.position.copy(center).addScaledVector(dirVec, dist * 1.6);
    camera.near = Math.max(0.01, radius / 100);
    camera.far = Math.max(100, radius * 100);
    camera.updateProjectionMatrix();
    controls.update();
  }

  // --- picking / hover / selection ---
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let pointerDirty = false;
  let pointerInside = false;
  let lastPointerX = 0;
  let lastPointerY = 0;
  const _pickRoots = [];
  const _boxHit = new THREE.Vector3();
  const _boxSize = new THREE.Vector3();

  function setHovered(id) {
    if (id === hoveredId) return;
    const prev = hoveredId;
    hoveredId = id;
    applyBboxColor(prev);
    applyBboxColor(id);
    if (prev !== null) applyBboxVisibility(prev);
    if (id !== null) applyBboxVisibility(id);
    if (id === null && tooltipOwner === view) tooltip.style.display = "none";
  }

  function setSelected(id) {
    if (id === selectedId) return;
    const prev = selectedId;
    selectedId = id;
    applyBboxColor(prev);
    applyBboxColor(id);
    refreshAllBboxVisibility();
  }

  function pickAtPointer() {
    raycaster.setFromCamera(pointer, camera);
    _pickRoots.length = 0;
    for (const model of models.values()) {
      if (model.visible) _pickRoots.push(model);
    }
    for (const solid of solids.values()) {
      if (solid.visible) _pickRoots.push(solid);
    }
    if (_pickRoots.length > 0) {
      for (const hit of raycaster.intersectObjects(_pickRoots, true)) {
        let node = hit.object;
        while (node) {
          if (node.userData?.pickId != null) return node.userData.pickId;
          node = node.parent;
        }
      }
    }
    for (const wantZone of [false, true]) {
      let bestId = null;
      let bestVol = Infinity;
      for (const [id, helper] of bboxes) {
        const isZone = (nodeInfo.get(id)?.kind ?? "zone") === "zone";
        if (isZone !== wantZone) continue;
        if (!wantZone && models.has(id) && meshesShown && !hiddenIds.has(id)) continue;
        if (!helper.visible) continue;
        if (!raycaster.ray.intersectBox(helper.box, _boxHit)) continue;
        helper.box.getSize(_boxSize);
        const vol = _boxSize.x * _boxSize.y * _boxSize.z;
        if (vol < bestVol) {
          bestVol = vol;
          bestId = id;
        }
      }
      if (bestId !== null) return bestId;
    }
    return null;
  }

  function fillTooltip(id, clientX, clientY) {
    const info = nodeInfo.get(id);
    if (!info) {
      tooltip.style.display = "none";
      return;
    }
    tooltip.textContent = "";
    const headEl = document.createElement("div");
    const kindEl = document.createElement("span");
    kindEl.textContent = `[${info.kind}]`;
    kindEl.style.color = TOOLTIP_KIND_COLOR[info.kind] ?? "#e6e6e6";
    headEl.appendChild(kindEl);
    headEl.appendChild(document.createTextNode(` ${id} · ${alias}`));
    tooltip.appendChild(headEl);
    const d = info.dimensions;
    const o = info.origin;
    const rows = [];
    if (info.prompt) rows.push(["prompt", info.prompt]);
    if (info.libraryId) rows.push(["asset", info.libraryId]);
    if (Array.isArray(d)) rows.push(["size", `${d[0].toFixed(2)} × ${d[1].toFixed(2)} × ${d[2].toFixed(2)} ft`]);
    if (Array.isArray(o)) rows.push(["origin", `(${o[0].toFixed(2)}, ${o[1].toFixed(2)}, ${o[2].toFixed(2)}) ft`]);
    if (info.proxyShape) rows.push(["proxy", info.proxyShape]);
    if (info.orientation) rows.push(["yaw", `${info.orientation}°`]);
    if (hiddenIds.has(id)) rows.push(["hidden", "right-click to show"]);
    for (const [label, text] of rows) {
      const row = document.createElement("div");
      row.style.marginTop = "4px";
      row.style.color = "#bdbdbd";
      const lbl = document.createElement("span");
      lbl.textContent = `${label}: `;
      lbl.style.color = "#7a8190";
      row.appendChild(lbl);
      row.appendChild(document.createTextNode(text));
      tooltip.appendChild(row);
    }
    placeTooltip(clientX, clientY);
  }

  function updatePointer(ev) {
    const rect = viewport.getBoundingClientRect();
    pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  }

  viewport.addEventListener("pointermove", (ev) => {
    updatePointer(ev);
    lastPointerX = ev.clientX;
    lastPointerY = ev.clientY;
    pointerInside = true;
    pointerDirty = true;
  });

  viewport.addEventListener("pointerleave", () => {
    pointerInside = false;
    setHovered(null);
  });

  let _downX = 0;
  let _downY = 0;
  let _downT = 0;
  let _downButton = -1;

  viewport.addEventListener("pointerdown", (ev) => {
    _downX = ev.clientX;
    _downY = ev.clientY;
    _downT = performance.now();
    _downButton = ev.button;
  });

  viewport.addEventListener("pointerup", (ev) => {
    if (_downButton !== 0 || ev.button !== 0) return;
    if (Math.hypot(ev.clientX - _downX, ev.clientY - _downY) > CLICK_MAX_MOVE_PX) return;
    if (performance.now() - _downT > CLICK_MAX_DURATION_MS) return;
    updatePointer(ev);
    const id = pickAtPointer();
    setSelected(id === selectedId ? null : id);
  });

  viewport.addEventListener("contextmenu", (ev) => {
    ev.preventDefault();
    if (_downButton === 2) {
      if (Math.hypot(ev.clientX - _downX, ev.clientY - _downY) > CLICK_MAX_MOVE_PX) return;
      if (performance.now() - _downT > CLICK_MAX_DURATION_MS) return;
    }
    updatePointer(ev);
    const id = pickAtPointer();
    if (id !== null && (nodeInfo.get(id)?.kind ?? "zone") !== "zone") {
      if (hiddenIds.has(id)) hiddenIds.delete(id);
      else hiddenIds.add(id);
      applyModelVisibility(id);
      pointerDirty = true;
    }
  });

  // --- v4 floor-plan panel ---
  function renderPlan() {
    const ok = planData && Array.isArray(planData.grid);
    planPanel.classList.toggle("on", !!ok);
    planLegend.textContent = "";
    if (!ok) return;
    // Cells carry the object names directly. Plans committed before the
    // number system was removed carried legend + int cells — map them back
    // to names so one rendering path serves both eras.
    const grid = Array.isArray(planData.legend)
      ? planData.grid.map((row) => row.map((v) => (v > 0 ? planData.legend[v - 1] : "")))
      : planData.grid;
    const rows = grid.length;
    const cols = grid[0]?.length ?? 0;
    if (!rows || !cols) return;
    // Seed the shared name->color index in reading order so the solid boxes
    // (colored by the same index) match this panel exactly.
    const names = [];
    for (const row of grid) {
      for (const c of row) {
        const name = c && c !== "0" ? c : "";
        if (name && !names.includes(name)) {
          colorIndexForName(name);
          names.push(name);
        }
      }
    }
    const colorOf = (name) => planColor(colorIndexForName(name));
    const cell = 20;
    planCanvas.width = cols * cell;
    planCanvas.height = rows * cell;
    const ctx = planCanvas.getContext("2d");
    ctx.fillStyle = "#1c1f26";
    ctx.fillRect(0, 0, planCanvas.width, planCanvas.height);
    // Row 1 is the scene's BACK edge — drawn at the top, per the panel note.
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < (grid[r]?.length ?? 0); c++) {
        const name = grid[r][c] && grid[r][c] !== "0" ? grid[r][c] : "";
        ctx.fillStyle = name ? colorOf(name) : "#101114";
        ctx.fillRect(c * cell + 1, r * cell + 1, cell - 1, cell - 1);
      }
    }
    names.forEach((name) => {
      const row = document.createElement("div");
      row.className = "pl-row";
      const swatch = document.createElement("span");
      swatch.className = "pl-swatch";
      swatch.style.background = colorOf(name);
      const label = document.createElement("span");
      label.className = "pl-name";
      label.textContent = name;
      label.title = name;
      row.append(swatch, label);
      planLegend.appendChild(row);
    });
    // Solids may have been built before the plan arrived (or replayed out of
    // order) — re-color them to the now-seeded plan colors.
    refreshSolidColors();
  }

  // --- header / overlay ---
  function status() {
    return slotSummaries.find((s) => s.id === currentSlot)?.runs?.[alias]?.status ?? "idle";
  }

  function updateHeader() {
    const st = status();
    dot.className = `slot-dot status-${st}`;
    metaEl.textContent = `${st} · ${objectIds.size} obj`;
    tile.classList.toggle("err", st === "error");
    const cfg = STATUS_LIFECYCLE[st] ?? STATUS_LIFECYCLE.idle;
    if (cfg[0] === null) {
      lifecycleBtn.style.display = "none";
    } else {
      lifecycleBtn.style.display = "";
      lifecycleBtn.textContent = cfg[0];
      lifecycleBtn.className = `tile-btn ${cfg[1]}`;
      lifecycleBtn.title = cfg[2];
    }
  }

  function renderOverlay() {
    // Re-renders happen while the panel is open (next step lands, error) —
    // keep whichever sections the user already expanded.
    const openKeys = new Set(
      [...overlay.querySelectorAll("details[open]")].map((d) => d.dataset.k),
    );
    overlay.textContent = "";
    if (errorMessage) {
      const err = document.createElement("div");
      err.className = "ov-err";
      err.textContent = `run.error: ${errorMessage}`;
      overlay.appendChild(err);
    }
    if (llmCalls.size === 0) {
      const meta = document.createElement("div");
      meta.className = "ov-meta";
      meta.textContent = "(no committed design call yet)";
      overlay.appendChild(meta);
      return;
    }
    const addSection = (key, label, text) => {
      const det = document.createElement("details");
      det.dataset.k = key;
      if (openKeys.has(key)) det.open = true;
      const sum = document.createElement("summary");
      sum.textContent = label;
      const body = document.createElement("div");
      body.className = "ov-body";
      body.textContent = text;
      det.append(sum, body);
      overlay.appendChild(det);
    };
    for (const [step, call] of llmCalls) {
      const failed = call.kind === "llm.failed";
      const head = document.createElement("div");
      head.className = "ov-step";
      const parts = [step];
      if (call.tokens_in != null) parts.push(`${call.tokens_in} in`);
      if (call.tokens_out != null) parts.push(`${call.tokens_out} out`);
      if (failed) parts.push("FAILED");
      head.textContent = parts.join(" · ");
      overlay.appendChild(head);
      if (failed) {
        const why = document.createElement("div");
        why.className = "ov-err";
        why.textContent = `validator: ${call.reason ?? "(no reason)"}`;
        overlay.appendChild(why);
      }
      // Relaxed-validation repairs (v4 grid): committed, but worth seeing.
      const warns = call.output?.warnings;
      if (Array.isArray(warns) && warns.length) {
        const warn = document.createElement("div");
        warn.className = "ov-warn";
        warn.textContent = warns.map((m) => `warning: ${m}`).join("\n");
        overlay.appendChild(warn);
      }
      // `raw` is the verbatim model text on plain-text steps (CSV / grid
      // plan) and on failed calls; structured commits only have the parsed
      // `output`.
      addSection(`${step}:system prompt`, "exact input · system", call.system ?? "");
      addSection(`${step}:user prompt`, "exact input · user", call.user ?? "");
      addSection(`${step}:reasoning`, "reasoning", call.reasoning || "(model emitted no reasoning trace)");
      addSection(
        `${step}:output`,
        failed ? "exact output (rejected)" : "exact output",
        call.raw ?? (call.output == null ? "(none)" : JSON.stringify(call.output, null, 2)),
      );
      if (step === "oneshot_grid" && gridText) {
        addSection(`${step}:plan`, "floor plan (rendered, row 1 = back)", gridText);
      }
    }
    if (solverData) {
      const head = document.createElement("div");
      head.className = "ov-step";
      head.textContent = "solver · grid → placed boxes (deterministic, no LLM)";
      overlay.appendChild(head);
      addSection("solver:input", "exact input · grid plan", solverData.input ?? "");
      addSection("solver:output", "exact output · placed objects", solverData.output ?? "");
    }
  }

  infoBtn.addEventListener("click", () => {
    const open = !overlay.classList.contains("open");
    overlay.classList.toggle("open", open);
    infoBtn.classList.toggle("active", open);
    if (open) renderOverlay();
  });

  lifecycleBtn.addEventListener("click", async () => {
    const st = status();
    lifecycleBtn.disabled = true;
    try {
      await post(cellPath(alias, st === "running" ? "pause" : "resume"));
      await refreshSlots();
      subscribeAll();
    } catch (e) {
      setStatus(`${alias} ${st === "running" ? "pause" : "start"} failed: ${e.message}`, "err");
    } finally {
      lifecycleBtn.disabled = false;
    }
  });

  resetBtn.addEventListener("click", async () => {
    if (!window.confirm(`Wipe ${currentRun} :: ${currentSlot} · ${alias} and re-run from scratch?`)) return;
    resetBtn.disabled = true;
    try {
      await post(cellPath(alias, "reset"));
      await refreshSlots();
      subscribeAll();
    } catch (e) {
      setStatus(`${alias} reset failed: ${e.message}`, "err");
    } finally {
      resetBtn.disabled = false;
    }
  });

  // --- event dispatch ---
  function dispatch(event) {
    // A run.start at an index we've already passed means the cell's log was
    // rewound under us (reset / re-run delivered over a reconnected stream).
    // Wipe the view and rebuild from this run's events — otherwise the index
    // guard below silently drops the new run's early events (the bboxes)
    // while its late model events still land, leaving stale wireframes
    // misaligned with fresh meshes.
    if (
      event.kind === "run.start" &&
      typeof event.index === "number" &&
      event.index <= highestIndex
    ) {
      clear();
    }
    if (typeof event.index === "number") {
      if (event.index <= highestIndex) return;
      highestIndex = event.index;
    }
    switch (event.kind) {
      case "run.start":
        errorMessage = null;
        break;
      case "bbox":
        loadBbox(event);
        break;
      case "library.match": {
        const info = nodeInfo.get(event.id);
        if (info) info.libraryId = event.library_id;
        break;
      }
      case "oneshot.grid":
        gridText = event.content ?? null;
        if (overlay.classList.contains("open")) renderOverlay();
        break;
      case "oneshot.solver":
        solverData = event;
        if (overlay.classList.contains("open")) renderOverlay();
        break;
      case "model":
        loadModel(event);
        break;
      case "cache.llm":
      case "llm.failed":
        if (DESIGN_STEPS.has(event.step)) {
          llmCalls.set(event.step, event);
          if (event.step === "oneshot_grid" && event.output) {
            planData = event.output;
            renderPlan();
          }
          if (overlay.classList.contains("open")) renderOverlay();
        }
        break;
      case "run.error":
        errorMessage = event.message ?? "unknown";
        overlay.classList.add("open");
        infoBtn.classList.add("active");
        renderOverlay();
        refreshSlots();
        break;
      case "run.done":
      case "run.paused":
        refreshSlots();
        break;
    }
  }

  function clear() {
    gen += 1;
    for (const map of [bboxes, proxies]) {
      for (const obj of map.values()) {
        bboxRoot.remove(obj);
        obj.geometry?.dispose?.();
        obj.material?.dispose?.();
      }
      map.clear();
    }
    for (const m of models.values()) {
      sceneRoot.remove(m);
      disposeObject3D(m);
    }
    models.clear();
    for (const s of solids.values()) {
      sceneRoot.remove(s);
      s.geometry?.dispose?.();
      s.material?.dispose?.();
    }
    solids.clear();
    nameColorIndex.clear();
    nodeInfo.clear();
    hiddenIds.clear();
    objectIds.clear();
    highestIndex = -1;
    hoveredId = null;
    selectedId = null;
    cameraUserMoved = false;
    llmCalls.clear();
    solverData = null;
    gridText = null;
    planData = null;
    renderPlan();
    errorMessage = null;
    overlay.classList.remove("open");
    infoBtn.classList.remove("active");
    updateHeader();
  }

  function destroy() {
    clear();
    controls.dispose();
    tile.remove();
  }

  // Per-frame work driven by the shared loop: hover resolution, damping,
  // pending camera fits, then a scissored render into the tile's rect.
  function renderInto() {
    const rect = viewport.getBoundingClientRect();
    if (rect.width < 8 || rect.height < 8) return;
    if (rect.bottom < 0 || rect.top > window.innerHeight) return;
    controls.update();
    if (fitPending) {
      fitPending = false;
      fitToScene();
    }
    if (pointerDirty && !controlsInteracting) {
      pointerDirty = false;
      if (pointerInside) {
        const id = pickAtPointer();
        setHovered(id);
        if (id !== null) {
          tooltipOwner = view;
          fillTooltip(id, lastPointerX, lastPointerY);
        }
      }
    }
    const aspect = rect.width / rect.height;
    if (Math.abs(camera.aspect - aspect) > 1e-3) {
      camera.aspect = aspect;
      camera.updateProjectionMatrix();
    }
    const bottom = window.innerHeight - rect.bottom;
    renderer.setViewport(rect.left, bottom, rect.width, rect.height);
    renderer.setScissor(rect.left, bottom, rect.width, rect.height);
    renderer.render(scene, camera);
  }

  const view = {
    alias,
    tile,
    dispatch,
    clear,
    destroy,
    renderInto,
    updateHeader,
    deselect: () => setSelected(null),
    fit: () => { cameraUserMoved = false; fitToScene(true); },
    applyToggles: () => {
      refreshAllBboxVisibility();
      refreshAllModelVisibility();
      refreshAllSolidVisibility();
    },
  };
  return view;
}

let tooltipOwner = null;

// --- render loop -----------------------------------------------------------------

function animate() {
  requestAnimationFrame(animate);
  renderer.setScissorTest(false);
  renderer.clear();
  renderer.setScissorTest(true);
  for (const view of cells.values()) view.renderInto();
}
requestAnimationFrame(animate);

// --- data layer --------------------------------------------------------------------

let availableRuns = [];
let availableModels = [];
let availableVersions = [];
let slotSummaries = [];
let currentRun = null;
let currentSlot = null;
let checkedModels = new Set();
const cells = new Map(); // alias -> view
let source = null;

function api(path, withRun = false) {
  const url = new URL(path, SERVER_URL);
  if (withRun && currentRun) url.searchParams.set("run", currentRun);
  return url;
}

function cellPath(alias, action) {
  const suffix = action === "reset" ? "reset?start=true" : action;
  return `/oneshot/slots/${encodeURIComponent(currentSlot)}/${encodeURIComponent(alias)}/${suffix}`;
}

async function post(path) {
  const res = await fetch(api(path, true), { method: "POST" });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { detail = (await res.json()).detail ?? detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

// --- runs ---------------------------------------------------------------------------

function renderRunPicker() {
  runPickerEl.innerHTML = "";
  for (const r of availableRuns) {
    const opt = document.createElement("option");
    opt.value = r.name;
    opt.textContent = r.version ? `${r.name} · ${r.version}` : r.name;
    runPickerEl.appendChild(opt);
  }
  if (currentRun) runPickerEl.value = currentRun;
}

function renderVersionPicker(versions) {
  if (versions.join() === availableVersions.join()) return;
  availableVersions = versions;
  const prev = runVersionEl.value;
  runVersionEl.innerHTML = "";
  for (const v of versions) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = `new: ${v}`;
    runVersionEl.appendChild(opt);
  }
  // Keep a manual choice across refreshes; default to the newest version.
  runVersionEl.value = versions.includes(prev) ? prev : versions[versions.length - 1];
}

async function refreshRuns() {
  try {
    const res = await fetch(api("/oneshot/runs"));
    if (!res.ok) return;
    const payload = await res.json();
    availableRuns = payload.runs ?? [];
    if (!currentRun) {
      const saved = localStorage.getItem(RUN_KEY);
      currentRun = availableRuns.some((r) => r.name === saved)
        ? saved
        : (payload.current || null);
    }
    renderRunPicker();
  } catch {
    // Transient; next tick retries.
  }
}

async function switchRun(name) {
  if (!name || name === currentRun) return;
  currentRun = name;
  try { localStorage.setItem(RUN_KEY, name); } catch {}
  renderRunPicker();
  await refreshSlots();
  subscribeAll();
}

async function newRun() {
  const name = window.prompt("New one-shot run name:", "");
  if (!name) return;
  runNewEl.disabled = true;
  try {
    const body = { name: name.trim() };
    if (runVersionEl.value) body.version = runVersionEl.value;
    const res = await fetch(api("/oneshot/runs"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try { detail = (await res.json()).detail ?? detail; } catch {}
      setStatus(`create run failed: ${detail}`, "err");
      return;
    }
    const payload = await res.json();
    currentRun = payload.current;
    try { localStorage.setItem(RUN_KEY, currentRun); } catch {}
    setStatus(`run created :: ${currentRun}`);
    await refreshRuns();
    await refreshSlots();
    subscribeAll();
  } finally {
    runNewEl.disabled = false;
  }
}

runPickerEl.addEventListener("change", () => switchRun(runPickerEl.value));
runNewEl.addEventListener("click", newRun);

// --- slots + models -------------------------------------------------------------------

const STATUS_PRIORITY = ["error", "running", "paused", "done", "idle"];

function slotAggregateStatus(slot) {
  // The most attention-worthy status across the CHECKED models.
  let best = "idle";
  for (const alias of checkedModels) {
    const st = slot.runs?.[alias]?.status ?? "idle";
    if (STATUS_PRIORITY.indexOf(st) < STATUS_PRIORITY.indexOf(best)) best = st;
  }
  return best;
}

function renderSlotBar() {
  for (const el of Array.from(slotBarEl.querySelectorAll(".slot-tab"))) el.remove();
  for (const s of slotSummaries) {
    const tab = document.createElement("button");
    tab.type = "button";
    tab.className = "slot-tab" + (s.id === currentSlot ? " active" : "");
    tab.title = s.prompt ?? "";
    const dot = document.createElement("span");
    dot.className = `slot-dot status-${slotAggregateStatus(s)}`;
    const label = document.createElement("span");
    label.textContent = s.id;
    tab.append(dot, label);
    tab.addEventListener("click", () => switchSlot(s.id));
    slotBarEl.appendChild(tab);
  }
}

function renderModelBar() {
  for (const el of Array.from(modelBarEl.querySelectorAll(".model-check"))) el.remove();
  // Insert checkboxes between the bar label and the launch button.
  for (const alias of [...availableModels].reverse()) {
    const label = document.createElement("label");
    label.className = "model-check";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = checkedModels.has(alias);
    cb.addEventListener("change", () => {
      if (cb.checked) checkedModels.add(alias);
      else checkedModels.delete(alias);
      try { localStorage.setItem(CHECKED_KEY, JSON.stringify([...checkedModels])); } catch {}
      renderTiles();
      renderSlotBar();
      subscribeAll();
    });
    const text = document.createElement("span");
    text.textContent = alias;
    label.append(cb, text);
    modelBarEl.insertBefore(label, modelBarEl.children[1]);
  }
}

async function refreshSlots() {
  try {
    const res = await fetch(api("/oneshot/slots", true));
    if (!res.ok) return;
    const payload = await res.json();
    const hadModels = availableModels.length > 0;
    availableModels = payload.models ?? [];
    slotSummaries = payload.slots ?? [];
    renderVersionPicker(payload.versions ?? []);
    if (!hadModels) {
      // First load: restore checked set (default: every model).
      let saved = null;
      try { saved = JSON.parse(localStorage.getItem(CHECKED_KEY) ?? "null"); } catch {}
      checkedModels = new Set(
        Array.isArray(saved) && saved.length
          ? saved.filter((a) => availableModels.includes(a))
          : availableModels,
      );
      if (checkedModels.size === 0) checkedModels = new Set(availableModels);
      renderModelBar();
    }
    renderSlotBar();
    for (const view of cells.values()) view.updateHeader();
  } catch {
    // Transient; next tick retries.
  }
}

function switchSlot(slotId) {
  if (!slotId || slotId === currentSlot) return;
  currentSlot = slotId;
  try { localStorage.setItem(SLOT_KEY, slotId); } catch {}
  renderSlotBar();
  subscribeAll();
}

// --- tiles ---------------------------------------------------------------------------

function gridColumns(n) {
  if (n <= 1) return 1;
  if (n <= 4) return 2;
  if (n <= 9) return 3;
  return 4;
}

function renderTiles() {
  // Destroy views for unchecked models, create for newly checked, keep order.
  for (const [alias, view] of [...cells]) {
    if (!checkedModels.has(alias)) {
      view.destroy();
      cells.delete(alias);
    }
  }
  for (const alias of availableModels) {
    if (!checkedModels.has(alias) || cells.has(alias)) continue;
    const view = createCellView(alias);
    cells.set(alias, view);
  }
  // Re-append in registry order so the grid is stable.
  for (const alias of availableModels) {
    const view = cells.get(alias);
    if (view) tilesEl.appendChild(view.tile);
  }
  const n = cells.size;
  tilesEl.style.gridTemplateColumns = `repeat(${gridColumns(n)}, 1fr)`;
  let emptyEl = document.getElementById("tiles-empty");
  if (n === 0) {
    if (!emptyEl) {
      emptyEl = document.createElement("div");
      emptyEl.id = "tiles-empty";
      emptyEl.textContent = "no models checked — pick models above to build the comparison grid";
      tilesEl.appendChild(emptyEl);
    }
  } else {
    emptyEl?.remove();
  }
}

// --- multiplexed event stream -----------------------------------------------------------

function closeSource() {
  if (source) {
    source.close();
    source = null;
  }
}

function subscribeAll() {
  closeSource();
  for (const view of cells.values()) view.clear();
  if (!currentRun) {
    setStatus("no run yet — create one with “+ new run” to begin", "warn");
    return;
  }
  if (!currentSlot || cells.size === 0) return;
  const slot = slotSummaries.find((s) => s.id === currentSlot);
  const runVersion = availableRuns.find((r) => r.name === currentRun)?.version;
  setStatus(
    `${currentRun}${runVersion ? ` (${runVersion})` : ""} :: ${currentSlot} · ${cells.size} model${cells.size === 1 ? "" : "s"}\n"${slot?.prompt ?? ""}"`,
  );
  const url = api(`/oneshot/slots/${encodeURIComponent(currentSlot)}/events-all`, true);
  url.searchParams.set("models", [...cells.keys()].join(","));
  source = new EventSource(url);
  source.onmessage = (ev) => {
    try {
      const frame = JSON.parse(ev.data);
      cells.get(frame.model)?.dispatch(frame.event);
    } catch {
      // Malformed frame — skip.
    }
  };
  source.onerror = () => {
    // EventSource auto-reconnects; the server replays snapshots and the
    // per-tile index dedup makes the replay idempotent.
  };
}

// --- launch ----------------------------------------------------------------------------

async function launchAll() {
  if (!currentRun) {
    setStatus("create a run first (“+ new run”)", "warn");
    return;
  }
  if (!currentSlot || checkedModels.size === 0) {
    setStatus("pick a slot and at least one model first", "warn");
    return;
  }
  const slot = slotSummaries.find((s) => s.id === currentSlot);
  const withData = [...checkedModels].filter(
    (a) => (slot?.runs?.[a]?.events_count ?? 0) > 0,
  );
  if (withData.length > 0) {
    const ok = window.confirm(
      `${withData.join(", ")} already ${withData.length === 1 ? "has" : "have"} data on ${currentRun} :: ${currentSlot}.\n\nReset and relaunch fresh for all ${checkedModels.size} checked models?`,
    );
    if (!ok) return;
  }
  launchEl.disabled = true;
  try {
    const results = await Promise.all(
      [...checkedModels].map((alias) =>
        post(cellPath(alias, "reset"))
          .then(() => ({ alias, ok: true }))
          .catch((e) => ({ alias, ok: false, error: e.message })),
      ),
    );
    const failed = results.filter((r) => !r.ok);
    if (failed.length > 0) {
      setStatus(
        `launched ${results.length - failed.length}/${results.length} — failed: ` +
          failed.map((f) => `${f.alias} (${f.error})`).join(", "),
        "err",
      );
    } else {
      setStatus(`launched ${results.length} model${results.length === 1 ? "" : "s"} on ${currentSlot} — watching live`);
    }
    await refreshSlots();
    subscribeAll();
  } finally {
    launchEl.disabled = false;
  }
}

launchEl.addEventListener("click", launchAll);

// --- global view controls -----------------------------------------------------------------

function applyBboxToggle() {
  bboxToggleEl.textContent = `bboxes: ${bboxesShown ? "on" : "off"}`;
  bboxToggleEl.classList.toggle("off", !bboxesShown);
}
function applyMeshToggle() {
  meshToggleEl.textContent = `meshes: ${meshesShown ? "on" : "off"}`;
  meshToggleEl.classList.toggle("off", !meshesShown);
}
function applySolidToggle() {
  solidToggleEl.textContent = `solids: ${solidsShown ? "on" : "off"}`;
  solidToggleEl.classList.toggle("off", !solidsShown);
}
bboxToggleEl.addEventListener("click", () => {
  bboxesShown = !bboxesShown;
  localStorage.setItem(BBOX_KEY, bboxesShown ? "1" : "0");
  applyBboxToggle();
  for (const view of cells.values()) view.applyToggles();
});
meshToggleEl.addEventListener("click", () => {
  meshesShown = !meshesShown;
  localStorage.setItem(MESH_KEY, meshesShown ? "1" : "0");
  applyMeshToggle();
  for (const view of cells.values()) view.applyToggles();
});
solidToggleEl.addEventListener("click", () => {
  solidsShown = !solidsShown;
  localStorage.setItem(SOLID_KEY, solidsShown ? "1" : "0");
  applySolidToggle();
  for (const view of cells.values()) view.applyToggles();
});
fitEl.addEventListener("click", () => {
  for (const view of cells.values()) view.fit();
});
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") {
    for (const view of cells.values()) view.deselect();
  }
});
applyBboxToggle();
applyMeshToggle();
applySolidToggle();

// --- boot -----------------------------------------------------------------------------

(async () => {
  await refreshRuns();
  await refreshSlots();
  if (slotSummaries.length === 0) {
    setStatus("no oneshot slots reported — is the server running?", "err");
    return;
  }
  const savedSlot = localStorage.getItem(SLOT_KEY);
  currentSlot = slotSummaries.some((s) => s.id === savedSlot)
    ? savedSlot
    : slotSummaries[0].id;
  renderSlotBar();
  renderTiles();
  subscribeAll();
})();

setInterval(refreshSlots, 2000);
setInterval(refreshRuns, 10000);
