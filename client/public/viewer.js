import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";

const SERVER_URL = document
  .querySelector('meta[name="server-url"]')
  .getAttribute("content");

const SLOT_STORAGE_KEY = "starshot.selectedSlot";
const BBOX_VISIBLE_STORAGE_KEY = "starshot.bboxesVisible";
const FRAMES_VISIBLE_STORAGE_KEY = "starshot.framesVisible";

const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");
const slotsEl = document.getElementById("slots");
const resetEl = document.getElementById("slot-reset");
const bboxToggleEl = document.getElementById("bbox-toggle");
const framesToggleEl = document.getElementById("frames-toggle");
const exportGlbEl = document.getElementById("export-glb");
const assetsEl = document.getElementById("assets");
const assetsBodyEl = document.getElementById("assets-body");
const assetsCountEl = document.getElementById("assets-count");
const assetsHeaderEl = document.getElementById("assets-header");
const assetsToggleEl = document.getElementById("assets-toggle");
const treeEl = document.getElementById("tree");
const treeBodyEl = document.getElementById("tree-body");
const treeHeaderEl = document.getElementById("tree-header");
const treeToggleEl = document.getElementById("tree-toggle");
const treeActiveEl = document.getElementById("tree-active");
const treeSearchEl = document.getElementById("tree-search");
const treeSearchClearEl = document.getElementById("tree-search-clear");
const treeSearchCountEl = document.getElementById("tree-search-count");

// --- log panel --------------------------------------------------------------

const KIND_COLOR = {
  "run.start": "#9ad4ff",
  "run.done": "#8bd17c",
  "run.error": "#ff8080",
  "bbox": "#e0c271",
  "image": "#f6a96a",
  "model": "#c586d1",
  "step": "#4a8fd8",
  "divider.decompose": "#e0c271",
  "generation.decompose": "#c586d1",
};

function setStatus(text, cls = "hdr") {
  statusEl.innerHTML = "";
  const p = document.createElement("p");
  p.className = `line ${cls}`;
  p.textContent = text;
  statusEl.appendChild(p);
}

function fmtValue(v) {
  if (Array.isArray(v)) return "[" + v.map(fmtValue).join(", ") + "]";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2);
  if (v && typeof v === "object")
    return "{" + Object.entries(v).map(([k, x]) => `${k}=${fmtValue(x)}`).join(", ") + "}";
  if (typeof v === "string") return v;
  return String(v);
}

function appendEvent(event) {
  const { kind, index, ...fields } = event;
  const p = document.createElement("p");
  p.className = "line";
  if (typeof index === "number") p.dataset.eventIndex = String(index);

  if (typeof index === "number") {
    const btn = document.createElement("button");
    btn.className = "rewind";
    btn.type = "button";
    btn.textContent = "↶ rewind";
    btn.title = `Rewind to event ${index} (discards this event and everything after)`;
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      rewindTo(index);
    });
    p.appendChild(btn);
  }

  if (typeof index === "number") {
    const idx = document.createElement("span");
    idx.className = "idx";
    idx.textContent = `#${index}`;
    p.appendChild(idx);
  }

  const tag = document.createElement("span");
  tag.className = "step";
  tag.textContent = `[${kind}]`;
  tag.style.color = KIND_COLOR[kind] ?? "#8bd17c";
  p.appendChild(tag);

  const entries = Object.entries(fields);
  if (entries.length === 0) {
    p.appendChild(document.createTextNode(""));
  } else {
    for (const [k, v] of entries) {
      const kv = document.createElement("span");
      kv.className = "kv";
      const label = document.createElement("span");
      label.className = "k";
      label.textContent = ` ${k}=`;
      kv.appendChild(label);
      kv.appendChild(document.createTextNode(fmtValue(v)));
      p.appendChild(kv);
    }
  }
  logEl.appendChild(p);
  logEl.scrollTop = logEl.scrollHeight;
}

function clearLog() {
  logEl.innerHTML = "";
}

// --- asset browser ----------------------------------------------------------

// id -> { imageUrl, prompt, modelUrl, status: "pending" | "loaded" | "error", errorMessage }
const assets = new Map();

function assetStatus(a) {
  return a.status ?? "pending";
}

function upsertAsset(id, patch) {
  const cur = assets.get(id) ?? { imageUrl: null, prompt: null, modelUrl: null, status: "pending" };
  assets.set(id, { ...cur, ...patch });
  renderAsset(id);
  assetsCountEl.textContent = `(${assets.size})`;
}

function renderAsset(id) {
  const a = assets.get(id);
  if (!a) return;
  let card = assetsBodyEl.querySelector(`[data-id="${CSS.escape(id)}"]`);
  if (!card) {
    card = document.createElement("div");
    card.className = "asset-card";
    card.dataset.id = id;
    card.innerHTML = `
      <a class="asset-thumb-link" target="_blank" rel="noopener">
        <div class="asset-thumb placeholder">no image</div>
      </a>
      <div class="asset-body">
        <div class="asset-id"></div>
        <div class="asset-status pending">pending</div>
        <div class="asset-prompt"></div>
      </div>
    `;
    assetsBodyEl.appendChild(card);
    const promptEl = card.querySelector(".asset-prompt");
    promptEl.addEventListener("click", () => {
      promptEl.classList.toggle("expanded");
    });
  }
  card.querySelector(".asset-id").textContent = id;

  const status = assetStatus(a);
  card.className = `asset-card status-${status}`;
  const statusTag = card.querySelector(".asset-status");
  statusTag.className = `asset-status ${status}`;
  statusTag.textContent = status === "error" && a.errorMessage
    ? `error: ${a.errorMessage}`
    : status;

  const link = card.querySelector(".asset-thumb-link");
  const thumb = card.querySelector(".asset-thumb");
  if (a.imageUrl) {
    const absImg = new URL(a.imageUrl, SERVER_URL).toString();
    link.href = absImg;
    if (thumb.tagName !== "IMG") {
      const img = document.createElement("img");
      img.className = "asset-thumb";
      img.loading = "lazy";
      img.alt = id;
      img.src = absImg;
      thumb.replaceWith(img);
    } else if (thumb.src !== absImg) {
      thumb.src = absImg;
    }
  }

  const promptEl = card.querySelector(".asset-prompt");
  promptEl.textContent = a.prompt ?? "";
}

function clearAssets() {
  assets.clear();
  assetsBodyEl.innerHTML = "";
  assetsCountEl.textContent = "(0)";
}

assetsHeaderEl.addEventListener("click", () => {
  const collapsed = assetsEl.classList.toggle("collapsed");
  assetsToggleEl.textContent = collapsed ? "▸" : "▾";
});

// --- tree view --------------------------------------------------------------

// Mirror of the server-side recursion. Nodes are upserted by `bbox` (when
// placed) or by `divider.decompose` (announces children before their bboxes
// are resolved, so the tree shows pending placeholders). The `step` event
// drives the per-node phase badge and the global "active" highlight.
const treeNodes = new Map(); // id -> { id, parentId, prompt, kind, phase, order }
const treeChildren = new Map(); // parentId -> [childIds] in insertion order
let treeRootId = null;
let treeActiveId = null;
let treeOrderCounter = 0;

// Per-node visibility overrides driven from the tree. A node in this set
// hides its own bbox/proxy/mesh; hiding a parent transitively hides every
// descendant in the scene (computed lazily via ancestor walk). State is
// per-run — cleared on slot switch / rewind / reset alongside the rest of
// the tree, and not persisted.
const hiddenIds = new Set();

function treeUpsert(id, patch) {
  const cur = treeNodes.get(id);
  if (!cur) {
    treeNodes.set(id, {
      id, parentId: null, prompt: null, kind: "zone",
      phase: "pending", order: treeOrderCounter++,
      ...patch,
    });
    const parentId = patch.parentId ?? null;
    if (parentId !== null) {
      const arr = treeChildren.get(parentId) ?? [];
      if (!arr.includes(id)) arr.push(id);
      treeChildren.set(parentId, arr);
    } else if (treeRootId === null) {
      treeRootId = id;
    }
    return;
  }
  // Existing node: merge patch, but if parentId changes from null to a real
  // one, wire it into the children index lazily.
  const prevParent = cur.parentId;
  Object.assign(cur, patch);
  if (prevParent === null && cur.parentId) {
    const arr = treeChildren.get(cur.parentId) ?? [];
    if (!arr.includes(id)) arr.push(id);
    treeChildren.set(cur.parentId, arr);
    if (treeRootId === id) treeRootId = null; // was mis-rooted
  }
}

function treeSetPhase(id, phase) {
  const cur = treeNodes.get(id);
  if (!cur) {
    // Step fired before any bbox / decompose — stash the phase so it renders
    // as soon as we have the node.
    treeUpsert(id, { phase });
  } else {
    cur.phase = phase;
  }
  if (phase !== "done") {
    treeActiveId = id;
  } else if (treeActiveId === id) {
    // A node finishing doesn't move the focus elsewhere on its own; leave
    // the highlight on it until the next step event moves it.
  }
  renderTree();
}

function treeClear() {
  treeNodes.clear();
  treeChildren.clear();
  treeRootId = null;
  treeActiveId = null;
  treeOrderCounter = 0;
  hiddenIds.clear();
  treeBodyEl.innerHTML = "";
  treeActiveEl.textContent = "";
}

// True if `id` itself or any ancestor is in `hiddenIds`. The descendant
// case is what makes hiding a zone hide everything underneath it without
// having to mark each child individually.
function effectivelyHidden(id) {
  let cur = treeNodes.get(id);
  while (cur) {
    if (hiddenIds.has(cur.id)) return true;
    cur = cur.parentId ? treeNodes.get(cur.parentId) : null;
  }
  return false;
}

function toggleNodeHidden(id) {
  if (hiddenIds.has(id)) hiddenIds.delete(id);
  else hiddenIds.add(id);
  // Effective-hidden status changed for this node and (potentially) every
  // descendant, so re-evaluate visibility for the whole subtree.
  refreshSubtreeVisibility(id);
  renderTree();
}

function refreshSubtreeVisibility(rootId) {
  const stack = [rootId];
  while (stack.length) {
    const cur = stack.pop();
    applyModelVisibility(cur);
    applyBboxVisibility(cur);
    const kids = treeChildren.get(cur) ?? [];
    for (const k of kids) stack.push(k);
  }
}

function truncate(s, n = 60) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function renderTreeNode(id, ctx) {
  const node = treeNodes.get(id);
  if (!node) return null;
  if (ctx && !ctx.visible.has(id)) return null;
  const wrap = document.createElement("div");
  const classes = ["tree-node"];
  if (id === treeActiveId) classes.push("active");
  if (id === selectedBboxId) classes.push("selected");
  if (ctx && ctx.matches.has(id)) classes.push("matched", "match-highlight");
  if (effectivelyHidden(id)) classes.push("vis-hidden");
  wrap.className = classes.join(" ");
  wrap.dataset.id = id;

  const row = document.createElement("div");
  row.className = "tree-row";
  // Click the row (not a nested child-tree row) to select this node.
  row.addEventListener("click", (ev) => {
    ev.stopPropagation();
    selectTreeNode(id);
  });

  // Per-node visibility toggle. Reflects only the self-hidden state — a
  // descendant of a hidden ancestor still shows ● because its own bit is
  // off; the row's `vis-hidden` class communicates effective state.
  const visBtn = document.createElement("button");
  visBtn.type = "button";
  const selfHidden = hiddenIds.has(id);
  visBtn.className = `tree-vis-toggle${selfHidden ? " hidden" : ""}`;
  visBtn.textContent = selfHidden ? "○" : "●";
  visBtn.title = selfHidden ? "Show node" : "Hide node";
  visBtn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    toggleNodeHidden(id);
  });
  row.appendChild(visBtn);

  const idEl = document.createElement("span");
  idEl.className = `tree-id ${node.kind}`;
  idEl.textContent = node.id;
  row.appendChild(idEl);

  const promptEl = document.createElement("span");
  promptEl.className = "tree-prompt";
  promptEl.textContent = truncate(node.prompt, 80);
  promptEl.title = node.prompt ?? "";
  row.appendChild(promptEl);

  const phaseEl = document.createElement("span");
  phaseEl.className = `tree-phase phase-${node.phase}`;
  phaseEl.textContent = node.phase;
  row.appendChild(phaseEl);

  wrap.appendChild(row);

  const childIds = treeChildren.get(id) ?? [];
  if (childIds.length > 0) {
    const kidsEl = document.createElement("div");
    kidsEl.className = "tree-children";
    for (const cid of childIds) {
      const cEl = renderTreeNode(cid, ctx);
      if (cEl) kidsEl.appendChild(cEl);
    }
    if (kidsEl.childNodes.length > 0) wrap.appendChild(kidsEl);
  }
  return wrap;
}

let treeSearchQuery = "";
// Index of the currently-focused match within `orderedMatches`. Repeated
// Enter advances; Shift+Enter goes back. -1 = nothing selected yet for
// this query (next Enter selects [0]).
let treeMatchIndex = -1;

// A node is a `match` if its id or prompt contains the query.
// `visible` = matches ∪ ancestors-of-matches ∪ descendants-of-matches.
// Returns null when no query (everything visible, no match highlighting).
function computeTreeFilter() {
  const q = treeSearchQuery.trim().toLowerCase();
  if (!q) return null;
  const matches = new Set();
  for (const [id, node] of treeNodes) {
    const idHit = id.toLowerCase().includes(q);
    const promptHit = (node.prompt ?? "").toLowerCase().includes(q);
    if (idHit || promptHit) matches.add(id);
  }
  const visible = new Set(matches);
  // Walk ancestors up.
  for (const id of matches) {
    let cur = treeNodes.get(id);
    while (cur && cur.parentId) {
      if (visible.has(cur.parentId)) break;
      visible.add(cur.parentId);
      cur = treeNodes.get(cur.parentId);
    }
  }
  // Walk descendants down so users can see the matched subtree expanded.
  const stack = [...matches];
  while (stack.length) {
    const id = stack.pop();
    const kids = treeChildren.get(id) ?? [];
    for (const cid of kids) {
      if (!visible.has(cid)) {
        visible.add(cid);
        stack.push(cid);
      }
    }
  }
  return { matches, visible };
}

// Match order for "Enter selects first match" — use insertion order so
// the first match in the rendered tree is what gets selected.
function orderedMatches(filter) {
  if (!filter) return [];
  const sorted = [...treeNodes.values()]
    .filter((n) => filter.matches.has(n.id))
    .sort((a, b) => a.order - b.order);
  return sorted.map((n) => n.id);
}

function renderTree() {
  treeBodyEl.innerHTML = "";
  const filter = computeTreeFilter();
  if (treeRootId !== null) {
    const el = renderTreeNode(treeRootId, filter);
    if (el) treeBodyEl.appendChild(el);
  }
  if (treeActiveId !== null) {
    const n = treeNodes.get(treeActiveId);
    if (n) treeActiveEl.textContent = `${n.phase} · ${n.id}`;
  }
  if (filter) {
    const n = filter.matches.size;
    if (n === 0) {
      treeSearchCountEl.textContent = "no matches";
    } else if (treeMatchIndex >= 0) {
      treeSearchCountEl.textContent = `${(treeMatchIndex % n) + 1}/${n}`;
    } else {
      treeSearchCountEl.textContent = `${n} match${n === 1 ? "" : "es"}`;
    }
  } else {
    treeSearchCountEl.textContent = "";
  }
}

treeHeaderEl.addEventListener("click", () => {
  const collapsed = treeEl.classList.toggle("collapsed");
  treeToggleEl.textContent = collapsed ? "▸" : "▾";
});

treeSearchEl.addEventListener("input", () => {
  treeSearchQuery = treeSearchEl.value;
  treeSearchClearEl.classList.toggle("visible", treeSearchQuery.length > 0);
  treeMatchIndex = -1;
  renderTree();
});

treeSearchEl.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") {
    ev.preventDefault();
    const ordered = orderedMatches(computeTreeFilter());
    if (ordered.length === 0) return;
    if (treeMatchIndex < 0) {
      treeMatchIndex = ev.shiftKey ? ordered.length - 1 : 0;
    } else {
      const delta = ev.shiftKey ? -1 : 1;
      treeMatchIndex = (treeMatchIndex + delta + ordered.length) % ordered.length;
    }
    const target = ordered[treeMatchIndex];
    // Bypass selectTreeNode's toggle behaviour — re-pressing Enter on the
    // same node should cycle, not deselect.
    if (selectedBboxId !== target) selectTreeNode(target);
    const row = treeBodyEl.querySelector(`.tree-node[data-id="${CSS.escape(target)}"]`);
    if (row) row.scrollIntoView({ block: "nearest", behavior: "smooth" });
    renderTree();
  } else if (ev.key === "Escape") {
    treeSearchEl.value = "";
    treeSearchEl.dispatchEvent(new Event("input"));
  }
});

treeSearchClearEl.addEventListener("click", () => {
  treeSearchEl.value = "";
  treeSearchEl.dispatchEvent(new Event("input"));
  treeSearchEl.focus();
});

// --- three.js scene ---------------------------------------------------------

const host = document.getElementById("canvas-host");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x101114);
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const sceneRoot = new THREE.Group();
scene.add(sceneRoot);

// Bboxes live in a sibling group so they don't participate in fit-to-scene,
// and so clearScene can nuke them independently.
const bboxRoot = new THREE.Group();
scene.add(bboxRoot);
let bboxesShown = localStorage.getItem(BBOX_VISIBLE_STORAGE_KEY) !== "0";
function applyBboxToggleLabel() {
  bboxToggleEl.textContent = `bboxes: ${bboxesShown ? "on" : "off"}`;
  bboxToggleEl.classList.toggle("off", !bboxesShown);
}
applyBboxToggleLabel();
bboxToggleEl.addEventListener("click", () => {
  bboxesShown = !bboxesShown;
  localStorage.setItem(BBOX_VISIBLE_STORAGE_KEY, bboxesShown ? "1" : "0");
  applyBboxToggleLabel();
  refreshAllBboxVisibility();
});

let framesShown = localStorage.getItem(FRAMES_VISIBLE_STORAGE_KEY) !== "0";
function applyFramesToggleLabel() {
  framesToggleEl.textContent = `frames: ${framesShown ? "on" : "off"}`;
  framesToggleEl.classList.toggle("off", !framesShown);
}
applyFramesToggleLabel();
framesToggleEl.addEventListener("click", () => {
  framesShown = !framesShown;
  localStorage.setItem(FRAMES_VISIBLE_STORAGE_KEY, framesShown ? "1" : "0");
  applyFramesToggleLabel();
  refreshAllFrameModelVisibility();
});
const bboxes = new Map(); // id -> THREE.Box3Helper
const proxies = new Map(); // id -> THREE.Mesh (wireframe proxy silhouette)
const modelsById = new Map(); // id -> THREE.Object3D (the loaded gltf.scene)
let hoveredBboxId = null;

const BBOX_COLOR_DEFAULT = 0xff3b3b;
const BBOX_COLOR_OBJECT = 0x6bd96e;
const BBOX_COLOR_FRAME = 0x7fb3d5;
const BBOX_COLOR_PROXY = 0xb46aff;
const BBOX_COLOR_HOVER = 0xffe14a;
const BBOX_COLOR_SELECTED = 0x4af0e0;
let selectedBboxId = null;

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const bboxPickPoint = new THREE.Vector3();
const bboxPickSize = new THREE.Vector3();
let pointerDirty = false;
let lastPointerClientX = 0;
let lastPointerClientY = 0;
let controlsInteracting = false;

const tooltip = document.createElement("div");
tooltip.id = "bbox-tooltip";
tooltip.style.cssText = [
  "position: fixed",
  "padding: 3px 8px",
  "background: rgba(22, 24, 29, 0.92)",
  "color: #e6e6e6",
  "border: 1px solid #2a2d35",
  "border-radius: 4px",
  "font: 12px ui-monospace, SFMono-Regular, Menlo, monospace",
  "pointer-events: none",
  "display: none",
  "z-index: 10",
].join("; ");
document.body.appendChild(tooltip);

const camera = new THREE.PerspectiveCamera(
  50,
  window.innerWidth / window.innerHeight,
  0.05,
  5000,
);
camera.position.set(8, 6, 10);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 1, 0);
controls.update();

// Once the user drags the camera, stop auto-fitting so subsequent runs
// preserve their chosen angle. The flag deliberately survives clearScene.
let cameraUserMoved = false;
controls.addEventListener("start", () => {
  cameraUserMoved = true;
  controlsInteracting = true;
  setHoveredBbox(null);
  tooltip.style.display = "none";
});
controls.addEventListener("end", () => {
  controlsInteracting = false;
  pointerDirty = true;
});

// --- WASD fly controls (complementary to OrbitControls) --------------------
// WASD strafes on the horizontal plane relative to the camera direction;
// Q/E moves world-down/up; Shift multiplies speed. Translates camera and
// target together so OrbitControls' pivot follows the camera.
const pressedKeys = new Set();
let _lastMoveT = performance.now();
const _MOVE_KEYS = new Set(["w", "a", "s", "d", "q", "e", "r", "f"]);

function _isTypingTarget(t) {
  return t instanceof HTMLElement &&
    (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
}

window.addEventListener("keydown", (ev) => {
  if (_isTypingTarget(ev.target)) return;
  const k = ev.key.toLowerCase();
  if (_MOVE_KEYS.has(k)) {
    pressedKeys.add(k);
    ev.preventDefault();
  } else if (k === "shift") {
    pressedKeys.add("shift");
  }
});

window.addEventListener("keyup", (ev) => {
  pressedKeys.delete(ev.key.toLowerCase());
});

// Alt-tab / focus-loss: drop held keys so they don't stick on.
window.addEventListener("blur", () => pressedKeys.clear());

const _fwd = new THREE.Vector3();
const _right = new THREE.Vector3();
const _worldUp = new THREE.Vector3(0, 1, 0);
const _move = new THREE.Vector3();

function _applyKeyboardMove(dt) {
  if (pressedKeys.size === 0) return;
  const shifted = pressedKeys.has("shift");
  const speed = 2 * (shifted ? 3 : 1) * dt;

  _fwd.subVectors(controls.target, camera.position);
  _fwd.y = 0;
  if (_fwd.lengthSq() === 0) return;
  _fwd.normalize();
  _right.crossVectors(_fwd, _worldUp).normalize();

  _move.set(0, 0, 0);
  if (pressedKeys.has("w")) _move.addScaledVector(_fwd, speed);
  if (pressedKeys.has("s")) _move.addScaledVector(_fwd, -speed);
  if (pressedKeys.has("d")) _move.addScaledVector(_right, speed);
  if (pressedKeys.has("a")) _move.addScaledVector(_right, -speed);
  if (pressedKeys.has("e")) _move.addScaledVector(_worldUp, speed);
  if (pressedKeys.has("q")) _move.addScaledVector(_worldUp, -speed);

  if (_move.lengthSq() !== 0) {
    camera.position.add(_move);
    controls.target.add(_move);
    cameraUserMoved = true;
  }

  // Dolly toward / away from the orbit target. Held key = continuous zoom;
  // ~1.5x per second baseline, 4x with shift.
  if (pressedKeys.has("r") || pressedKeys.has("f")) {
    const rate = shifted ? 4 : 1.5;
    let factor = 1;
    if (pressedKeys.has("r")) factor *= Math.pow(1 / rate, dt);
    if (pressedKeys.has("f")) factor *= Math.pow(rate, dt);
    _dolly(factor);
  }
}

function _dolly(factor) {
  // factor < 1 zooms in, factor > 1 zooms out. Implemented as scaling the
  // camera->target distance so OrbitControls' pivot semantics stay intact.
  const offset = camera.position.clone().sub(controls.target);
  const dist = offset.length();
  if (dist === 0) return;
  const minDist = 0.05;
  const maxDist = 4000;
  const newDist = Math.max(minDist, Math.min(maxDist, dist * factor));
  offset.multiplyScalar(newDist / dist);
  camera.position.copy(controls.target).add(offset);
  cameraUserMoved = true;
}

scene.add(new THREE.HemisphereLight(0xffffff, 0x202028, 0.9));
const dir = new THREE.DirectionalLight(0xffffff, 0.9);
dir.position.set(8, 12, 6);
scene.add(dir);
scene.add(new THREE.AxesHelper(1));

// Infinite ground grid: a huge plane with a procedural grid shader. Lines
// antialias via screen-space derivatives and fade with distance so the plane
// never looks like it has an edge. Fade distance is driven from camera
// distance each frame so detail scales naturally as the user zooms.
const gridGeom = new THREE.PlaneGeometry(100000, 100000);
gridGeom.rotateX(-Math.PI / 2);
const gridMat = new THREE.ShaderMaterial({
  uniforms: {
    uCameraPos: { value: new THREE.Vector3() },
    uMinorColor: { value: new THREE.Color(0x202020) },
    uMajorColor: { value: new THREE.Color(0x505050) },
    uMinorSpacing: { value: 1.0 },
    uMajorSpacing: { value: 10.0 },
    uFadeStart: { value: 20.0 },
    uFadeEnd: { value: 200.0 },
  },
  vertexShader: `
    varying vec3 vWorldPos;
    void main() {
      vec4 wp = modelMatrix * vec4(position, 1.0);
      vWorldPos = wp.xyz;
      gl_Position = projectionMatrix * viewMatrix * wp;
    }
  `,
  fragmentShader: `
    uniform vec3 uCameraPos;
    uniform vec3 uMinorColor;
    uniform vec3 uMajorColor;
    uniform float uMinorSpacing;
    uniform float uMajorSpacing;
    uniform float uFadeStart;
    uniform float uFadeEnd;
    varying vec3 vWorldPos;

    float gridLine(vec2 p, float spacing) {
      vec2 q = p / spacing;
      vec2 g = abs(fract(q - 0.5) - 0.5) / fwidth(q);
      return 1.0 - min(min(g.x, g.y), 1.0);
    }

    void main() {
      float minor = gridLine(vWorldPos.xz, uMinorSpacing);
      float major = gridLine(vWorldPos.xz, uMajorSpacing);
      float d = distance(vWorldPos.xz, uCameraPos.xz);
      float fade = 1.0 - smoothstep(uFadeStart, uFadeEnd, d);
      float alpha = max(minor * 0.5, major) * fade;
      if (alpha < 0.002) discard;
      vec3 col = mix(uMinorColor, uMajorColor, major);
      gl_FragColor = vec4(col, alpha);
    }
  `,
  transparent: true,
  depthWrite: false,
  side: THREE.DoubleSide,
});
const groundGrid = new THREE.Mesh(gridGeom, gridMat);
groundGrid.renderOrder = -1;
scene.add(groundGrid);

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = Math.min(0.1, (now - _lastMoveT) / 1000);
  _lastMoveT = now;
  _applyKeyboardMove(dt);
  controls.update();

  gridMat.uniforms.uCameraPos.value.copy(camera.position);
  const camDist = Math.max(1, camera.position.distanceTo(controls.target));
  gridMat.uniforms.uFadeStart.value = camDist * 0.5;
  gridMat.uniforms.uFadeEnd.value = camDist * 6.0;

  if (pointerDirty && !controlsInteracting) {
    pointerDirty = false;
    raycaster.setFromCamera(pointer, camera);
    const hoveredId = pickHoveredBboxId();
    setHoveredBbox(hoveredId);
    if (hoveredId !== null) {
      positionTooltip(lastPointerClientX, lastPointerClientY, hoveredId);
    } else {
      tooltip.style.display = "none";
    }
  }

  renderer.render(scene, camera);
}
animate();

function clearScene() {
  resetModelQueue();
  while (sceneRoot.children.length > 0) {
    const child = sceneRoot.children[0];
    sceneRoot.remove(child);
    child.traverse?.((n) => {
      if (n.isMesh) {
        n.geometry?.dispose?.();
        const mats = Array.isArray(n.material) ? n.material : n.material ? [n.material] : [];
        for (const m of mats) m.dispose?.();
      }
    });
  }
  for (const helper of bboxes.values()) {
    bboxRoot.remove(helper);
    helper.geometry?.dispose?.();
    helper.material?.dispose?.();
  }
  bboxes.clear();
  for (const mesh of proxies.values()) {
    bboxRoot.remove(mesh);
    mesh.geometry?.dispose?.();
    mesh.material?.dispose?.();
  }
  proxies.clear();
  modelsById.clear();
  hoveredBboxId = null;
  selectedBboxId = null;
  tooltip.style.display = "none";
}

// Fit controls to the union of all loaded models after each addition.
// Skipped once the user has manually adjusted the camera.
function fitToScene() {
  if (cameraUserMoved) return;
  const box = new THREE.Box3().setFromObject(sceneRoot);
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

// --- model loading ----------------------------------------------------------

const loader = new GLTFLoader();

// Concurrent loads — each GLB fetches/parses/uploads independently. The
// browser's per-origin connection limit (~6) naturally throttles the network
// side. `sceneGen` invalidates loads still in flight when the scene is reset
// (rewind / fresh snapshot on reconnect).
let sceneGen = 0;

function resetModelQueue() {
  sceneGen += 1;
}

function loadModel(event) {
  _loadModelNow(event, sceneGen);
}

async function _loadModelNow(event, gen) {
  if (gen !== sceneGen) return;
  const absUrl = new URL(event.url, SERVER_URL).toString();
  // Skip a re-load when this id already errored on the *same URL* during
  // this scene generation. The server occasionally emits `model` more
  // than once for one id (anchor completion loop, cached replay), and
  // re-running GLTFLoader on a known-bad GLB just spams the same parse
  // error. A new URL or a new sceneGen still goes through.
  const prior = assets.get(event.id);
  if (prior?.status === "error" && prior.modelUrl === event.url) return;
  upsertAsset(event.id, { modelUrl: event.url });
  try {
    const gltf = await loader.loadAsync(absUrl);
    if (gen !== sceneGen) return;
    gltf.scene.traverse((child) => {
      if (child.isMesh && child.material) {
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        for (const m of mats) m.side = THREE.DoubleSide;
      }
    });
    gltf.scene.name = `${event.artifact_kind}:${event.id}`;
    const prevModel = modelsById.get(event.id);
    if (prevModel) {
      sceneRoot.remove(prevModel);
      prevModel.traverse?.((n) => {
        if (n.isMesh) {
          n.geometry?.dispose?.();
          const mats = Array.isArray(n.material) ? n.material : n.material ? [n.material] : [];
          for (const m of mats) m.dispose?.();
        }
      });
    }
    sceneRoot.add(gltf.scene);
    modelsById.set(event.id, gltf.scene);
    applyModelVisibility(event.id);
    fitToScene();
    upsertAsset(event.id, { status: "loaded" });
  } catch (e) {
    appendEvent({ kind: "model.error", id: event.id, message: e.message });
    upsertAsset(event.id, { status: "error", errorMessage: e.message });
  }
}

// --- bbox overlays ----------------------------------------------------------

// `{ id, origin: [x,y,z], dimensions: [dx,dy,dz], proxy_shape?: ... }` —
// matches the Python BoundingBox+Node serialization. Signed and
// zero-valued dimensions are allowed (walls/floors are flat). If a proxy
// shape is set, we draw its wireframe silhouette in addition to the AABB
// wireframe so the user can see what the LLM and surface-snap are
// actually reasoning about.
function loadBbox(event) {
  const { id, origin, dimensions } = event;
  if (bboxes.has(id)) {
    const prev = bboxes.get(id);
    bboxRoot.remove(prev);
    prev.geometry?.dispose?.();
    prev.material?.dispose?.();
    if (hoveredBboxId === id) hoveredBboxId = null;
  }
  if (proxies.has(id)) {
    const prev = proxies.get(id);
    bboxRoot.remove(prev);
    prev.geometry?.dispose?.();
    prev.material?.dispose?.();
    proxies.delete(id);
  }
  const ox = origin[0], oy = origin[1], oz = origin[2];
  const fx = ox + dimensions[0], fy = oy + dimensions[1], fz = oz + dimensions[2];
  const box3 = new THREE.Box3(
    new THREE.Vector3(Math.min(ox, fx), Math.min(oy, fy), Math.min(oz, fz)),
    new THREE.Vector3(Math.max(ox, fx), Math.max(oy, fy), Math.max(oz, fz)),
  );
  const helper = new THREE.Box3Helper(box3, BBOX_COLOR_DEFAULT);
  helper.userData.bboxId = id;
  helper.userData.nodeKind = event.node_kind ?? "zone";
  helper.userData.proxyShape = event.proxy_shape ?? null;
  bboxRoot.add(helper);
  bboxes.set(id, helper);

  const proxyMesh = buildProxyWireframe(event.proxy_shape, origin, dimensions);
  if (proxyMesh !== null) {
    bboxRoot.add(proxyMesh);
    proxies.set(id, proxyMesh);
  }
  // If this id is already selected (user clicked before bbox arrived, or a
  // bbox is being replaced), reapply the selection color.
  applyBboxColor(id);
  applyBboxVisibility(id);
}

function buildProxyWireframe(proxyShape, origin, dimensions) {
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
    // Ellipsoid inscribed in the AABB: unit sphere (diameter 1) scaled
    // to each AABB extent.
    geom = new THREE.SphereGeometry(0.5, 24, 16);
    geom.scale(sx, sy, sz);
    anchorY = cy;
  } else if (proxyShape === "HEMISPHERE") {
    // Top hemisphere with equatorial disk on the AABB's bottom face.
    // thetaLength = PI/2 starting at the north pole gives the upper half.
    geom = new THREE.SphereGeometry(
      0.5, 24, 16,
      0, Math.PI * 2,
      0, Math.PI / 2,
    );
    // The unit hemisphere spans y in [0, 0.5]; scale y by (sy / 0.5) so
    // the apex reaches +sy above the equator.
    geom.scale(sx, sy * 2, sz);
    anchorY = yMin;
  } else if (proxyShape === "CAPSULE") {
    const r = Math.min(sx, sz) / 2;
    const cylHeight = Math.max(0, sy - 2 * r);
    geom = new THREE.CapsuleGeometry(r, cylHeight, 8, 24);
    anchorY = cy;
  } else {
    return null;
  }

  const mat = new THREE.MeshBasicMaterial({
    color: BBOX_COLOR_DEFAULT, wireframe: true, transparent: true, opacity: 0.55,
  });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.position.set(cx, anchorY, cz);
  mesh.renderOrder = 1;
  return mesh;
}

function applyBboxColor(id) {
  const helper = id !== null ? bboxes.get(id) : null;
  if (!helper) return;
  const base =
    helper.userData.proxyShape ? BBOX_COLOR_PROXY
    : helper.userData.nodeKind === "object" ? BBOX_COLOR_OBJECT
    : helper.userData.nodeKind === "frame" ? BBOX_COLOR_FRAME
    : BBOX_COLOR_DEFAULT;
  const color =
    id === selectedBboxId ? BBOX_COLOR_SELECTED
    : id === hoveredBboxId ? BBOX_COLOR_HOVER
    : base;
  helper.material.color.setHex(color);
  const proxy = proxies.get(id);
  if (proxy) proxy.material.color.setHex(color);
}

function applyBboxVisibility(id) {
  // Tree-driven hide always wins — hovering or selecting a hidden node
  // doesn't sneak its bbox back into view.
  const userVisible = !effectivelyHidden(id);
  const visible =
    userVisible &&
    (bboxesShown || id === hoveredBboxId || id === selectedBboxId);
  const helper = bboxes.get(id);
  if (helper) helper.visible = visible;
  const proxy = proxies.get(id);
  if (proxy) proxy.visible = visible;
}

function refreshAllBboxVisibility() {
  for (const id of bboxes.keys()) applyBboxVisibility(id);
}

function applyModelVisibility(id) {
  const model = modelsById.get(id);
  if (!model) return;
  const isFrame = treeNodes.get(id)?.kind === "frame";
  const frameOk = isFrame ? framesShown : true;
  model.visible = frameOk && !effectivelyHidden(id);
}

function refreshAllFrameModelVisibility() {
  for (const id of modelsById.keys()) applyModelVisibility(id);
}

function setHoveredBbox(id) {
  if (id === hoveredBboxId) return;
  const prev = hoveredBboxId;
  hoveredBboxId = id;
  applyBboxColor(prev);
  applyBboxColor(id);
  if (prev !== null) applyBboxVisibility(prev);
  if (id !== null) applyBboxVisibility(id);
}

// Fit the camera to a single Box3 — parameterised variant of fitToScene.
// Used by tree-click selection.
function frameBbox(box) {
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(0.5 * Math.max(size.x, size.y, size.z), 0.5);
  controls.target.copy(center);
  const dist = radius / Math.tan((camera.fov * Math.PI) / 360);
  const dirVec = new THREE.Vector3(1, 0.7, 1).normalize();
  camera.position.copy(center).addScaledVector(dirVec, dist * 1.8);
  camera.near = Math.max(0.01, radius / 100);
  camera.far = Math.max(100, radius * 100);
  camera.updateProjectionMatrix();
  controls.update();
}

function selectTreeNode(id) {
  const prev = selectedBboxId;
  // Toggle off if re-clicking the same node.
  selectedBboxId = prev === id ? null : id;
  if (prev !== null) {
    applyBboxColor(prev);
    applyBboxVisibility(prev);
  }
  if (selectedBboxId !== null) {
    applyBboxColor(selectedBboxId);
    applyBboxVisibility(selectedBboxId);
  }
  renderTree();
  if (selectedBboxId !== null) {
    const helper = bboxes.get(selectedBboxId);
    if (helper) {
      // User took explicit camera control — stop auto-fit from later snapping
      // the view back to the full scene when new meshes land.
      cameraUserMoved = true;
      frameBbox(helper.box);
    }
  }
}

function positionTooltip(clientX, clientY, text) {
  tooltip.textContent = text;
  tooltip.style.left = `${clientX + 12}px`;
  tooltip.style.top = `${clientY + 12}px`;
  tooltip.style.display = "block";
}

function pickHoveredBboxId() {
  let bestId = null;
  let bestKindRank = Infinity;
  let bestDistance = Infinity;
  let bestSize = Infinity;
  for (const [id, model] of modelsById) {
    if (!model.visible) continue;
    const helper = bboxes.get(id);
    if (!helper) continue;
    if (!raycaster.ray.intersectBox(helper.box, bboxPickPoint)) continue;

    const kind = treeNodes.get(id)?.kind;
    const kindRank = kind === "object" ? 0 : kind === "frame" ? 1 : 2;
    const distance = bboxPickPoint.distanceToSquared(camera.position);
    helper.box.getSize(bboxPickSize);
    const sizeTieBreaker = bboxPickSize.lengthSq();

    if (
      kindRank < bestKindRank ||
      (kindRank === bestKindRank && distance < bestDistance) ||
      (kindRank === bestKindRank && distance === bestDistance && sizeTieBreaker < bestSize)
    ) {
      bestId = id;
      bestKindRank = kindRank;
      bestDistance = distance;
      bestSize = sizeTieBreaker;
    }
  }
  return bestId;
}

renderer.domElement.addEventListener("pointermove", (ev) => {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  lastPointerClientX = ev.clientX;
  lastPointerClientY = ev.clientY;
  pointerDirty = true;
});

renderer.domElement.addEventListener("pointerleave", () => {
  setHoveredBbox(null);
  tooltip.style.display = "none";
});

// --- event dispatch ---------------------------------------------------------

// Event-index high-water mark. The server re-replays the entire snapshot
// from index 0 on every SSE (re)connect. EventSource auto-reconnects on its
// own (server idle, network blip), so without this guard we'd wipe and
// reload every model on every reconnect — which is exactly the "models keep
// reloading" behaviour the user reports when inspecting a finished run. We
// dedupe by index instead and let already-processed events fall through.
// Reset to -1 only on explicit user-driven state wipes (slot switch, reset,
// rewind), where a fresh replay genuinely needs to be re-applied.
let highestEventIndex = -1;

function dispatch(event) {
  if (typeof event.index === "number") {
    if (event.index <= highestEventIndex) return;
    highestEventIndex = event.index;
  }
  appendEvent(event);
  switch (event.kind) {
    case "run.start":
      setStatus(`run :: ${event.model}`);
      break;
    case "run.done":
      setStatus("run complete");
      if (currentSource) { currentSource.close(); currentSource = null; }
      refreshSlots();
      break;
    case "run.error":
      setStatus(`error: ${event.message}`, "err");
      if (currentSource) { currentSource.close(); currentSource = null; }
      refreshSlots();
      break;
    case "bbox":
      loadBbox(event);
      treeUpsert(event.id, {
        parentId: event.parent_id ?? null,
        prompt: event.prompt ?? null,
        kind: event.node_kind ?? "zone",
      });
      renderTree();
      break;
    case "divider.decompose":
      // Pre-declare children so the tree shows them (in pending state) before
      // their bboxes resolve. `children` ships as [{id, prompt}, ...].
      for (const c of event.children ?? []) {
        treeUpsert(c.id, { parentId: event.node, prompt: c.prompt, kind: "zone" });
      }
      renderTree();
      break;
    case "step":
      treeSetPhase(event.node, event.phase);
      break;
    case "mesh.submit":
      // Object mesh generation kicked off — show it on the tree.
      treeSetPhase(event.id, "generating_mesh");
      break;
    case "image":
      upsertAsset(event.id, { imageUrl: event.url, prompt: event.prompt });
      break;
    case "model":
      loadModel(event);
      treeSetPhase(event.id, "done");
      break;
    // Everything else is already shown as a log line above.
  }
}

// --- slot picker + run lifecycle --------------------------------------------

// All seven pipelines run in the background on the server. The client
// chooses which one to view — switching closes the active SSE, clears the
// scene, and reconnects to the selected slot's stream.

let currentSource = null;
let currentSlotId = null;
let slotSummaries = [];  // latest /slots payload, for tab rendering

function renderSlotTabs() {
  // Wipe any existing .slot-tab children; keep the #slot-reset button.
  for (const child of Array.from(slotsEl.querySelectorAll(".slot-tab"))) {
    child.remove();
  }
  for (const s of slotSummaries) {
    const tab = document.createElement("button");
    tab.type = "button";
    tab.className = "slot-tab" + (s.id === currentSlotId ? " active" : "");
    tab.dataset.slotId = s.id;
    tab.title = s.prompt ?? "";

    const dot = document.createElement("span");
    dot.className = `slot-dot status-${s.status ?? "idle"}`;
    tab.appendChild(dot);

    const label = document.createElement("span");
    label.textContent = s.id;
    tab.appendChild(label);

    tab.addEventListener("click", () => switchSlot(s.id));
    slotsEl.insertBefore(tab, resetEl);
  }
}

async function refreshSlots() {
  try {
    const res = await fetch(new URL("/slots", SERVER_URL));
    if (!res.ok) return;
    slotSummaries = await res.json();
    renderSlotTabs();
  } catch {
    // Transient; next tick will retry.
  }
}

function switchSlot(id) {
  if (id === currentSlotId) return;
  if (currentSource) {
    currentSource.close();
    currentSource = null;
  }
  clearScene();
  clearLog();
  clearAssets();
  treeClear();
  highestEventIndex = -1;
  currentSlotId = id;
  try { localStorage.setItem(SLOT_STORAGE_KEY, id); } catch {}
  renderSlotTabs();
  setStatus(`slot :: ${id}`);
  subscribe(slotEventsUrl(id));
}

function slotEventsUrl(id) {
  return new URL(`/slots/${encodeURIComponent(id)}/events`, SERVER_URL).toString();
}

async function resetSlot(id) {
  const ok = window.confirm(
    `Wipe runs/${id}/ and restart the pipeline for this slot?`,
  );
  if (!ok) return;
  resetEl.disabled = true;
  try {
    const res = await fetch(
      new URL(`/slots/${encodeURIComponent(id)}/reset`, SERVER_URL),
      { method: "POST" },
    );
    if (!res.ok) {
      setStatus(`reset failed: HTTP ${res.status}`, "err");
      return;
    }
    if (currentSource) {
      currentSource.close();
      currentSource = null;
    }
    clearScene();
    clearLog();
    clearAssets();
    treeClear();
    highestEventIndex = -1;
    setStatus(`slot ${id} reset — streaming events…`);
    subscribe(slotEventsUrl(id));
    refreshSlots();
  } catch (e) {
    setStatus(`reset failed: ${e.message}`, "err");
  } finally {
    resetEl.disabled = false;
  }
}

function subscribe(url) {
  const es = new EventSource(url);
  currentSource = es;
  es.onmessage = (ev) => {
    let data;
    try {
      data = JSON.parse(ev.data);
    } catch {
      return;
    }
    dispatch(data);
  };
  es.onerror = () => {
    // EventSource auto-reconnects on transient failures; only surface a hard close.
    if (es.readyState === EventSource.CLOSED && currentSource === es) {
      setStatus("stream closed", "err");
      currentSource = null;
    }
  };
}

async function rewindTo(index) {
  if (currentSlotId === null) return;
  if (currentSource) {
    currentSource.close();
    currentSource = null;
  }
  clearScene();
  clearLog();
  clearAssets();
  treeClear();
  highestEventIndex = -1;
  setStatus(`POST /slots/${currentSlotId}/rewind to ${index} …`);

  let res;
  try {
    res = await fetch(
      new URL(`/slots/${encodeURIComponent(currentSlotId)}/rewind`, SERVER_URL),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ to_event_index: index }),
      },
    );
  } catch (e) {
    setStatus(`rewind failed: ${e.message}`, "err");
    return;
  }
  if (!res.ok) {
    setStatus(`HTTP ${res.status}: ${await res.text()}`, "err");
    return;
  }
  setStatus(`rewound to ${index} — streaming events…`);
  subscribe(slotEventsUrl(currentSlotId));
  refreshSlots();
}

resetEl.addEventListener("click", () => {
  if (currentSlotId !== null) resetSlot(currentSlotId);
});

document.getElementById("zoom-in").addEventListener("click", () => _dolly(0.8));
document.getElementById("zoom-out").addEventListener("click", () => _dolly(1.25));

// --- GLB export --------------------------------------------------------------

exportGlbEl.addEventListener("click", async () => {
  if (modelsById.size === 0) return;
  exportGlbEl.disabled = true;
  exportGlbEl.textContent = "exporting…";
  try {
    // Temporarily force all models visible so hidden frames are included.
    const wasHidden = [];
    sceneRoot.traverse((obj) => {
      if (!obj.visible) {
        wasHidden.push(obj);
        obj.visible = true;
      }
    });

    const exporter = new GLTFExporter();
    const glb = await exporter.parseAsync(sceneRoot, { binary: true });

    for (const obj of wasHidden) obj.visible = false;

    const blob = new Blob([glb], { type: "model/gltf-binary" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${currentSlotId ?? "scene"}.glb`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    appendEvent({ kind: "run.error", message: `GLB export failed: ${e.message}` });
  } finally {
    exportGlbEl.disabled = false;
    exportGlbEl.textContent = "export .glb";
  }
});

// Boot: load slot list, pick the remembered slot (or the first), subscribe.
(async () => {
  await refreshSlots();
  if (slotSummaries.length === 0) {
    setStatus("no slots reported by server", "err");
    return;
  }
  let saved = null;
  try { saved = localStorage.getItem(SLOT_STORAGE_KEY); } catch {}
  const pick = slotSummaries.find((s) => s.id === saved)?.id ?? slotSummaries[0].id;
  switchSlot(pick);
})();

// Keep tab status dots fresh for slots the user isn't viewing.
setInterval(refreshSlots, 2000);
