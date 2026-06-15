// The 3D cell viewer — extracted from the proven pre-revamp viewer: bbox /
// proxy wireframes, GLB meshes (KTX2 + Meshopt capable), the one-connection
// SMB1 mesh bundle stream, shader ground grid, orbit controls, fit-to-scene,
// and the full interaction layer: raycast hover with the info tooltip,
// click-to-select with select/dim highlighting + camera framing, Shift
// zones-only picking, and WASD/QE + R/F fly-and-dolly keyboard controls.
// One instance for the whole app; the overlay mounts/unmounts its canvas.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";

const BBOX_COLOR_DEFAULT = 0xff3b3b;  // zones
const BBOX_COLOR_OBJECT = 0x6bd96e;
const BBOX_COLOR_FRAME = 0x7fb3d5;
const BBOX_COLOR_PROXY = 0xb46aff;
const BBOX_COLOR_HOVER = 0xffe14a;
const BBOX_COLOR_SELECTED = 0x4af0e0;
const BBOX_COLOR_OVERLAY = 0xff3df5;   // prompt-lab "after" (proposed) boxes
const BBOX_DIM_OPACITY = 0.35;
const PROXY_BASE_OPACITY = 0.55;
const PROXY_DIM_OPACITY = 0.2;
const TOOLTIP_KIND_COLOR = { zone: "#9ad4ff", object: "#8bd17c", frame: "#7fb3d5" };

const DECODE_WORKERS = Math.min(4, navigator.hardwareConcurrency || 2);
const MESH_BUNDLE_MAGIC = "SMB1";
const MAX_INFLIGHT = 20;
const CLICK_MAX_MOVE_PX = 4;
const CLICK_MAX_DURATION_MS = 400;

export function createViewer(host, { keyboard = true } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x101114);
  host.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  const sceneRoot = new THREE.Group(); // meshes (drives fit-to-scene)
  const bboxRoot = new THREE.Group();  // wireframes (excluded from fit)
  // Proposed-placement overlay (the prompt-lab's "after" boxes): magenta
  // wireframes drawn on top of the current scene, excluded from fit.
  const overlayRoot = new THREE.Group();
  scene.add(sceneRoot, bboxRoot, overlayRoot);

  const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 5000);
  camera.position.set(14, 10, 14);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 1, 0);
  controls.update();

  let cameraUserMoved = false;
  let controlsInteracting = false;
  controls.addEventListener("start", () => {
    cameraUserMoved = true;
    controlsInteracting = true;
    setHovered(null);
    tooltip.style.display = "none";
  });
  controls.addEventListener("end", () => {
    controlsInteracting = false;
    pointerDirty = true;
  });

  scene.add(new THREE.HemisphereLight(0xffffff, 0x202028, 0.9));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
  dirLight.position.set(20, 30, 14);
  scene.add(dirLight);

  // Distance-faded shader grid on the ground plane (ported verbatim; fade
  // window tracks camera distance so detail scales as the user zooms).
  const gridMat = new THREE.ShaderMaterial({
    uniforms: {
      uCameraPos: { value: new THREE.Vector3() },
      uMinorColor: { value: new THREE.Color(0x23262e) },
      uMajorColor: { value: new THREE.Color(0x3a3f4c) },
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
  const grid = new THREE.Mesh(new THREE.PlaneGeometry(100000, 100000).rotateX(-Math.PI / 2), gridMat);
  grid.renderOrder = -1;
  scene.add(grid);

  // KTX2/Basis + Meshopt so optimized-library GLBs parse.
  const ktx2 = new KTX2Loader()
    .setTranscoderPath("/vendor/three/examples/jsm/libs/basis/")
    .setWorkerLimit(DECODE_WORKERS)
    .detectSupport(renderer);
  MeshoptDecoder.useWorkers(DECODE_WORKERS);
  const loader = new GLTFLoader().setKTX2Loader(ktx2).setMeshoptDecoder(MeshoptDecoder);

  const bboxes = new Map();   // id -> Box3Helper
  const proxies = new Map();  // id -> wireframe proxy mesh
  const models = new Map();   // id -> gltf scene
  const kinds = new Map();    // id -> node_kind ("zone"/"object"/"frame")
  const show = { bboxes: true, meshes: true, frames: true, grid: true };
  let gen = 0;
  let fitPending = false;
  let bundleAbort = null;
  // The overlay slides off-screen with translateX, which keeps layout size —
  // so visibility is an explicit flag, gating draws AND keyboard capture.
  let active = false;

  // --- interaction state -----------------------------------------------------

  let hoveredId = null;
  let selectedId = null;
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let pointerDirty = false;
  let pointerInsideCanvas = false;
  let lastPointerClientX = 0;
  let lastPointerClientY = 0;
  const pressedKeys = new Set();
  let lastMoveT = performance.now();
  const MOVE_KEYS = new Set(["w", "a", "s", "d", "q", "e", "r", "f"]);

  // Overlay-provided callbacks: node info for the tooltip + hidden-state
  // ancestor walks, selection / hidden-set sync to the observability tree.
  let nodeInfo = () => null;
  let onSelectCb = () => {};
  let onHiddenChangeCb = () => {};

  // Per-node hiding (right-click, shared with the tree's eye buttons). A
  // hidden node hides its MESH only — the wireframe bbox stays visible as a
  // volumetric reference and as the right-click handle for un-hiding. Hiding
  // a ZONE hides every descendant's mesh; hiding an object hides just it.
  const hiddenIds = new Set();

  function effectivelyHidden(id) {
    let cur = nodeInfo(id);
    let isSelf = true;
    let hops = 0;
    while (cur && hops < 64) {
      if (hiddenIds.has(cur.id) && (isSelf || cur.kind === "zone")) return true;
      cur = cur.parentId ? nodeInfo(cur.parentId) : null;
      isSelf = false;
      hops += 1;
    }
    return isSelf ? hiddenIds.has(id) : false;
  }

  function toggleHidden(id) {
    if (hiddenIds.has(id)) hiddenIds.delete(id);
    else hiddenIds.add(id);
    // Effective-hidden status changed for this node and (potentially) every
    // descendant; the scene is small enough to just re-apply everywhere.
    refreshAllVisibility();
    onHiddenChangeCb(id, hiddenIds.has(id));
  }

  const tooltip = document.createElement("div");
  tooltip.style.cssText = [
    "position: fixed",
    "padding: 5px 9px",
    "background: rgba(22, 24, 29, 0.94)",
    "color: #e6e6e6",
    "border: 1px solid #2a2d35",
    "border-radius: 4px",
    "font: 12px ui-monospace, SFMono-Regular, Menlo, monospace",
    "pointer-events: none",
    "display: none",
    "z-index: 110", // above the overlay chrome, below modals
    "max-width: 360px",
    "white-space: pre-wrap",
    "line-height: 1.35",
  ].join("; ");
  document.body.appendChild(tooltip);

  // --- shared helpers ----------------------------------------------------------

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

  // --- colors / visibility (select + hover + dim semantics, ported) -------------

  function applyBboxColor(id) {
    const helper = id !== null ? bboxes.get(id) : null;
    if (!helper) return;
    const base =
      helper.userData.proxyShape ? BBOX_COLOR_PROXY
      : helper.userData.nodeKind === "object" ? BBOX_COLOR_OBJECT
      : helper.userData.nodeKind === "frame" ? BBOX_COLOR_FRAME
      : BBOX_COLOR_DEFAULT;
    const color =
      id === selectedId ? BBOX_COLOR_SELECTED
      : id === hoveredId ? BBOX_COLOR_HOVER
      : base;
    helper.material.color.setHex(color);
    const proxy = proxies.get(id);
    if (proxy) proxy.material.color.setHex(color);
  }

  function applyBboxVisibility(id) {
    // When something is selected, every OTHER bbox is dimmed (not hidden) so
    // the selected one stands out without losing the rest of the scene as
    // spatial reference. Hover gets full opacity so the user can see what
    // they're about to pick.
    const visible = id === selectedId || id === hoveredId || show.bboxes;
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

  function applyModelVisibility(id) {
    const model = models.get(id);
    if (!model) return;
    const isFrame = (kinds.get(id) ?? "zone") === "frame";
    model.visible = show.meshes && (isFrame ? show.frames : true) && !effectivelyHidden(id);
  }

  function refreshAllVisibility() {
    for (const id of bboxes.keys()) applyBboxVisibility(id);
    for (const id of models.keys()) applyModelVisibility(id);
    grid.visible = show.grid;
  }

  function setHovered(id) {
    if (id === hoveredId) return;
    const prev = hoveredId;
    hoveredId = id;
    applyBboxColor(prev);
    applyBboxColor(id);
    if (prev !== null) applyBboxVisibility(prev);
    if (id !== null) applyBboxVisibility(id);
  }

  // --- picking -------------------------------------------------------------------

  const _pickRoots = [];
  const _zoneHit = new THREE.Vector3();
  const _zoneSize = new THREE.Vector3();

  // Smallest visible zone bbox under the ray — the Shift override, and the
  // fallback when no meshes are pickable. Skips hidden nodes (their bbox is
  // only a reference + right-click unhide handle, not a hover target).
  function pickSmallestBoxId({ zonesOnly }) {
    let bestId = null;
    let bestVol = Infinity;
    for (const [id, helper] of bboxes) {
      if (zonesOnly && helper.userData.nodeKind !== "zone") continue;
      if (!helper.visible) continue;
      if (effectivelyHidden(id)) continue;
      if (!raycaster.ray.intersectBox(helper.box, _zoneHit)) continue;
      helper.box.getSize(_zoneSize);
      const vol = _zoneSize.x * _zoneSize.y * _zoneSize.z;
      if (vol < bestVol) {
        bestVol = vol;
        bestId = id;
      }
    }
    return bestId;
  }

  // Right-click target: whatever the hover picker would name, else the
  // NEAREST bbox face under the ray — hidden nodes included, since their
  // bbox is exactly the handle for un-hiding.
  const _rightClickBoxHit = new THREE.Vector3();
  function pickRightClickId() {
    const meshId = pickHoveredId();
    if (meshId !== null) return meshId;
    const zonesOnly = pressedKeys.has("shift");
    let bestId = null;
    let bestDist = Infinity;
    for (const [id, helper] of bboxes) {
      if (!helper.visible) continue;
      if (zonesOnly && helper.userData.nodeKind !== "zone") continue;
      if (!raycaster.ray.intersectBox(helper.box, _rightClickBoxHit)) continue;
      const dist = _rightClickBoxHit.distanceToSquared(camera.position);
      if (dist < bestDist) {
        bestDist = dist;
        bestId = id;
      }
    }
    return bestId;
  }

  function pickHoveredId() {
    if (pressedKeys.has("shift")) return pickSmallestBoxId({ zonesOnly: true });
    _pickRoots.length = 0;
    for (const model of models.values()) {
      if (model.visible) _pickRoots.push(model);
    }
    if (_pickRoots.length > 0) {
      const hits = raycaster.intersectObjects(_pickRoots, true);
      for (const hit of hits) {
        let node = hit.object;
        while (node) {
          const pid = node.userData?.pickId;
          if (pid != null) return pid;
          node = node.parent;
        }
      }
    }
    // Bbox-only view (or a miss past every mesh): pick wireframes directly.
    if (_pickRoots.length === 0) return pickSmallestBoxId({ zonesOnly: false });
    return null;
  }

  // --- tooltip -------------------------------------------------------------------

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

  function positionTooltip(clientX, clientY, id) {
    const info = nodeInfo(id) ?? {};
    const kind = info.kind ?? kinds.get(id) ?? "zone";
    // DOM nodes rather than innerHTML so LLM-authored text can't smuggle markup.
    tooltip.textContent = "";
    const head = document.createElement("div");
    const kindEl = document.createElement("span");
    kindEl.textContent = `[${kind}]`;
    kindEl.style.color = TOOLTIP_KIND_COLOR[kind] ?? "#e6e6e6";
    head.appendChild(kindEl);
    head.appendChild(document.createTextNode(` ${id}`));
    tooltip.appendChild(head);
    const sections = [];
    if (info.prompt) sections.push(["seed", info.prompt]);
    if (kind === "zone" && info.plan) sections.push(["plan", info.plan]);
    if (kind !== "zone" && info.imagePrompt && info.imagePrompt !== info.prompt) {
      sections.push(["image", info.imagePrompt]);
    }
    for (const [label, text] of sections) {
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

  // --- selection + camera framing ---------------------------------------------------

  // Fit the camera to a single Box3 — parameterised variant of fitToScene.
  function frameBox(box) {
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

  // Toggle selection (re-selecting clears, like the old tree click). Framing
  // marks the camera user-moved so auto-fit stops fighting the user.
  function select(id, { frame = true, notify = true } = {}) {
    const prev = selectedId;
    selectedId = prev === id ? null : id;
    if (prev !== null) applyBboxColor(prev);
    if (selectedId !== null) applyBboxColor(selectedId);
    for (const bid of bboxes.keys()) applyBboxVisibility(bid);
    if (selectedId !== null && frame) {
      const helper = bboxes.get(selectedId);
      if (helper) {
        cameraUserMoved = true;
        frameBox(helper.box);
      }
    }
    if (notify) onSelectCb(selectedId);
    return selectedId;
  }

  function clearSelection({ notify = true } = {}) {
    if (selectedId === null) return;
    const prev = selectedId;
    selectedId = null;
    applyBboxColor(prev);
    for (const bid of bboxes.keys()) applyBboxVisibility(bid);
    if (notify) onSelectCb(null);
  }

  // --- pointer + keyboard wiring -----------------------------------------------------

  renderer.domElement.addEventListener("pointermove", (ev) => {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    lastPointerClientX = ev.clientX;
    lastPointerClientY = ev.clientY;
    pointerInsideCanvas = true;
    pointerDirty = true;
  });

  renderer.domElement.addEventListener("pointerleave", () => {
    pointerInsideCanvas = false;
    setHovered(null);
    tooltip.style.display = "none";
  });

  // Click-to-select: distinguish a click from the end of an orbit drag by
  // distance + duration, then reuse the hover picker so selection matches
  // whatever the tooltip was showing.
  let _downX = 0;
  let _downY = 0;
  let _downT = 0;
  let _downButton = -1;
  renderer.domElement.addEventListener("pointerdown", (ev) => {
    _downX = ev.clientX;
    _downY = ev.clientY;
    _downT = performance.now();
    _downButton = ev.button;
  });
  renderer.domElement.addEventListener("pointerup", (ev) => {
    if (_downButton !== 0 || ev.button !== 0) return;
    const dx = ev.clientX - _downX;
    const dy = ev.clientY - _downY;
    if (Math.hypot(dx, dy) > CLICK_MAX_MOVE_PX) return;
    if (performance.now() - _downT > CLICK_MAX_DURATION_MS) return;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const id = pickHoveredId();
    if (id !== null) select(id);
    else clearSelection();
  });

  // Right-click toggles per-node hide for the picked id. The mesh disappears,
  // the bbox stays as a volumetric reference and as the click target for
  // un-hiding. Suppresses the browser's default context menu.
  renderer.domElement.addEventListener("contextmenu", (ev) => {
    ev.preventDefault();
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const id = pickRightClickId();
    if (id !== null) toggleHidden(id);
  });

  function isTypingTarget(t) {
    return t instanceof HTMLElement &&
      (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
  }

  // Keyboard is global (window-level), so grid/thumbnail viewers opt OUT
  // (`keyboard:false`) — otherwise WASD would drive every visible canvas at
  // once. Handlers are named so `dispose()` can detach them.
  const onKeyDown = (ev) => {
    if (!active) return;
    if (isTypingTarget(ev.target)) return;
    const k = ev.key.toLowerCase();
    if (MOVE_KEYS.has(k)) {
      pressedKeys.add(k);
      ev.preventDefault();
    } else if (k === "shift" && !pressedKeys.has("shift")) {
      // Shift flips picking to zones-only; refresh hover without a mouse move.
      pressedKeys.add("shift");
      if (pointerInsideCanvas) pointerDirty = true;
    }
  };
  const onKeyUp = (ev) => {
    const k = ev.key.toLowerCase();
    pressedKeys.delete(k);
    if (k === "shift" && pointerInsideCanvas) pointerDirty = true;
  };
  const onBlur = () => {
    pressedKeys.clear();
    if (pointerInsideCanvas) pointerDirty = true;
  };
  if (keyboard) {
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
  }

  // WASD strafes on the horizontal plane relative to the camera direction;
  // Q/E moves world-down/up; R/F dollies; Shift multiplies speed. Camera and
  // target translate together so OrbitControls' pivot follows.
  const _fwd = new THREE.Vector3();
  const _right = new THREE.Vector3();
  const _worldUp = new THREE.Vector3(0, 1, 0);
  const _move = new THREE.Vector3();

  function applyKeyboardMove(dt) {
    if (pressedKeys.size === 0) return;
    const shifted = pressedKeys.has("shift");
    const camDist = Math.max(1, camera.position.distanceTo(controls.target));
    const speed = Math.max(2, camDist * 0.6) * (shifted ? 3 : 1) * dt;
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
    if (pressedKeys.has("r") || pressedKeys.has("f")) {
      const rate = shifted ? 4 : 1.5;
      let factor = 1;
      if (pressedKeys.has("r")) factor *= Math.pow(1 / rate, dt);
      if (pressedKeys.has("f")) factor *= Math.pow(rate, dt);
      const offset = camera.position.clone().sub(controls.target);
      const dist = offset.length();
      if (dist > 0) {
        offset.multiplyScalar(Math.max(0.05, Math.min(4000, dist * factor)) / dist);
        camera.position.copy(controls.target).add(offset);
        cameraUserMoved = true;
      }
    }
  }

  // --- scene population --------------------------------------------------------------

  function fitToScene(force = false) {
    if (cameraUserMoved && !force) return;
    const box = new THREE.Box3();
    if (sceneRoot.children.length > 0) box.setFromObject(sceneRoot);
    if (box.isEmpty()) {
      for (const helper of bboxes.values()) box.union(helper.box);
    }
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
      color: PROXY_COLOR_FOR(proxyShape), wireframe: true, transparent: true, opacity: PROXY_BASE_OPACITY,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(cx, anchorY, cz);
    mesh.renderOrder = 1;
    return mesh;
  }
  const PROXY_COLOR_FOR = () => BBOX_COLOR_PROXY;

  function loadBbox(event) {
    const { id, origin, dimensions } = event;
    if (!Array.isArray(origin) || !Array.isArray(dimensions)) return;
    const prev = bboxes.get(id);
    if (prev) {
      bboxRoot.remove(prev);
      prev.geometry?.dispose?.();
      prev.material?.dispose?.();
      if (hoveredId === id) hoveredId = null;
    }
    const prevProxy = proxies.get(id);
    if (prevProxy) {
      bboxRoot.remove(prevProxy);
      prevProxy.geometry?.dispose?.();
      prevProxy.material?.dispose?.();
      proxies.delete(id);
    }
    const kind = event.node_kind ?? "zone";
    kinds.set(id, kind);
    const fx = origin[0] + dimensions[0];
    const fy = origin[1] + dimensions[1];
    const fz = origin[2] + dimensions[2];
    const box3 = new THREE.Box3(
      new THREE.Vector3(Math.min(origin[0], fx), Math.min(origin[1], fy), Math.min(origin[2], fz)),
      new THREE.Vector3(Math.max(origin[0], fx), Math.max(origin[1], fy), Math.max(origin[2], fz)),
    );
    const helper = new THREE.Box3Helper(box3, BBOX_COLOR_DEFAULT);
    helper.material.transparent = true;
    helper.material.opacity = 1;
    helper.userData.nodeKind = kind;
    helper.userData.proxyShape = event.proxy_shape ?? null;
    bboxRoot.add(helper);
    bboxes.set(id, helper);
    const proxy = buildProxyWireframe(event.proxy_shape, origin, dimensions);
    if (proxy) {
      bboxRoot.add(proxy);
      proxies.set(id, proxy);
    }
    applyBboxColor(id);
    applyBboxVisibility(id);
    scheduleFit();
  }

  // Proposed-placement overlay — the prompt-lab's "after" boxes (a tested
  // step's output) drawn on top of the current scene (the "before"). Modeled
  // on the old tune sandbox's magenta overlay; never replaces scene bboxes.
  function clearOverlayBoxes() {
    while (overlayRoot.children.length > 0) {
      const child = overlayRoot.children[0];
      overlayRoot.remove(child);
      child.geometry?.dispose?.();
      child.material?.dispose?.();
    }
  }

  function setOverlayBoxes(boxes) {
    clearOverlayBoxes();
    for (const b of boxes ?? []) {
      const { origin, dimensions } = b;
      if (!Array.isArray(origin) || !Array.isArray(dimensions)) continue;
      const fx = origin[0] + dimensions[0];
      const fy = origin[1] + dimensions[1];
      const fz = origin[2] + dimensions[2];
      const box3 = new THREE.Box3(
        new THREE.Vector3(Math.min(origin[0], fx), Math.min(origin[1], fy), Math.min(origin[2], fz)),
        new THREE.Vector3(Math.max(origin[0], fx), Math.max(origin[1], fy), Math.max(origin[2], fz)),
      );
      const helper = new THREE.Box3Helper(box3, BBOX_COLOR_OVERLAY);
      // Draw on top of everything so the proposal reads against the scene.
      helper.material.depthTest = false;
      helper.material.transparent = true;
      helper.renderOrder = 999;
      overlayRoot.add(helper);
    }
  }

  function attachGltf(id, gltfScene, kind) {
    gltfScene.traverse((child) => {
      if (child.isMesh && child.material) {
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        for (const m of mats) m.side = THREE.DoubleSide;
      }
    });
    gltfScene.name = `mesh:${id}`;
    gltfScene.userData.pickId = id;
    const prev = models.get(id);
    if (prev) {
      sceneRoot.remove(prev);
      disposeObject3D(prev);
    }
    sceneRoot.add(gltfScene);
    models.set(id, gltfScene);
    if (kind) kinds.set(id, kind);
    applyModelVisibility(id);
    scheduleFit();
  }

  const failedUrls = new Set(); // `${gen}|${url}` — don't re-parse known-bad GLBs
  async function loadModel(event, absUrl) {
    const myGen = gen;
    const k = `${myGen}|${absUrl}`;
    if (failedUrls.has(k) || models.has(event.id)) return;
    try {
      const gltf = await loader.loadAsync(absUrl);
      if (myGen !== gen) { disposeObject3D(gltf.scene); return; }
      attachGltf(event.id, gltf.scene, kinds.get(event.id));
    } catch (e) {
      failedUrls.add(k);
      console.warn(`[scene3d] mesh load failed for ${event.id}:`, e.message);
    }
  }

  function byteStreamReader(reader) {
    const chunks = [];
    let avail = 0;
    let ended = false;
    async function readExact(n) {
      while (avail < n) {
        if (ended) return null;
        const { done, value } = await reader.read();
        if (done) { ended = true; continue; }
        if (value && value.length) { chunks.push(value); avail += value.length; }
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

  function parseGlb(arrayBuffer) {
    return new Promise((resolve, reject) => loader.parse(arrayBuffer, "", resolve, reject));
  }

  // Pull the whole cell's GLBs over ONE connection and attach progressively.
  async function prefetchBundle(meshesUrl) {
    bundleAbort?.abort?.();
    const abort = new AbortController();
    bundleAbort = abort;
    const myGen = gen;
    try {
      const res = await fetch(meshesUrl, { cache: "no-store", signal: abort.signal });
      if (!res.ok || !res.body) return;
      const r = byteStreamReader(res.body.getReader());
      const dec = new TextDecoder();
      const magic = await r.readExact(4);
      if (!magic || dec.decode(magic) !== MESH_BUNDLE_MAGIC) return;
      const inflight = new Set();
      while (true) {
        if (myGen !== gen || abort.signal.aborted) break;
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
            if (myGen !== gen) { disposeObject3D(gltf.scene); return; }
            attachGltf(id, gltf.scene, kinds.get(id));
          } catch { /* model-event fallback will fetch it individually */ }
        })().finally(() => inflight.delete(p));
        inflight.add(p);
        if (inflight.size >= MAX_INFLIGHT) await Promise.race(inflight);
      }
      await Promise.allSettled(inflight);
    } catch { /* aborted / network — fallbacks cover it */ }
    if (myGen === gen) fitToScene();
  }

  function clear() {
    gen += 1;
    bundleAbort?.abort?.();
    while (sceneRoot.children.length > 0) {
      const child = sceneRoot.children[0];
      sceneRoot.remove(child);
      disposeObject3D(child);
    }
    for (const helper of bboxes.values()) {
      bboxRoot.remove(helper);
      helper.geometry?.dispose?.();
      helper.material?.dispose?.();
    }
    for (const proxy of proxies.values()) {
      bboxRoot.remove(proxy);
      proxy.geometry?.dispose?.();
      proxy.material?.dispose?.();
    }
    clearOverlayBoxes();
    bboxes.clear();
    proxies.clear();
    models.clear();
    kinds.clear();
    failedUrls.clear();
    hiddenIds.clear();
    hoveredId = null;
    selectedId = null;
    tooltip.style.display = "none";
    cameraUserMoved = false;
  }

  function scheduleFit() { fitPending = true; }

  function resize() {
    const w = host.clientWidth || 1;
    const h = host.clientHeight || 1;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(host);
  resize();

  let disposed = false;
  (function animate() {
    if (disposed) return; // viewer torn down — stop the rAF loop entirely
    requestAnimationFrame(animate);
    if (!active || !host.isConnected) return; // hidden — skip draws + input
    const now = performance.now();
    const dt = Math.min(0.1, (now - lastMoveT) / 1000);
    lastMoveT = now;
    applyKeyboardMove(dt);
    controls.update();

    gridMat.uniforms.uCameraPos.value.copy(camera.position);
    const camDist = Math.max(1, camera.position.distanceTo(controls.target));
    gridMat.uniforms.uFadeStart.value = camDist * 0.5;
    gridMat.uniforms.uFadeEnd.value = camDist * 6.0;

    if (pointerDirty && !controlsInteracting) {
      pointerDirty = false;
      if (pointerInsideCanvas) {
        raycaster.setFromCamera(pointer, camera);
        const id = pickHoveredId();
        setHovered(id);
        if (id !== null) positionTooltip(lastPointerClientX, lastPointerClientY, id);
        else tooltip.style.display = "none";
      }
    }

    if (fitPending) { fitPending = false; fitToScene(); }
    renderer.render(scene, camera);
  })();

  return {
    loadBbox,
    loadModel,
    prefetchBundle,
    setOverlayBoxes,
    clearOverlayBoxes,
    setOverlayVisible: (v) => { overlayRoot.visible = v; },
    setBboxesVisible: (v) => { show.bboxes = v; refreshAllVisibility(); },
    clear,
    hasModel: (id) => models.has(id),
    hasBbox: (id) => bboxes.has(id),
    setKind: (id, kind) => { if (kind) kinds.set(id, kind); },
    fit: () => fitToScene(true),
    toggles: show,
    refreshVisibility: refreshAllVisibility,
    select,
    clearSelection,
    getSelected: () => selectedId,
    toggleHidden,
    isHidden: (id) => hiddenIds.has(id),
    setNodeInfo: (fn) => { nodeInfo = fn; },
    onSelect: (fn) => { onSelectCb = fn; },
    onHiddenChange: (fn) => { onHiddenChangeCb = fn; },
    setActive: (v) => {
      active = v;
      if (!v) {
        pressedKeys.clear();
        setHovered(null);
        tooltip.style.display = "none";
      } else {
        resize();
      }
    },
    // Camera read/write + change hook — lets the compare view keep two
    // side-by-side viewers locked to the same vantage for honest A/B.
    getView: () => ({ position: camera.position.toArray(), target: controls.target.toArray() }),
    setView: (v) => {
      if (!v) return;
      camera.position.fromArray(v.position);
      controls.target.fromArray(v.target);
      camera.updateProjectionMatrix();
      controls.update();
      cameraUserMoved = true;
    },
    onCameraChange: (cb) => controls.addEventListener("change", cb),
    // Tear the viewer down completely — stop the loop, detach observers /
    // global listeners, free the GLB scene, and release the WebGL context.
    // Needed because the review grid creates one viewer per slot card and
    // must reclaim contexts (browsers cap simultaneous WebGL contexts).
    dispose: () => {
      if (disposed) return;
      disposed = true;
      active = false;
      bundleAbort?.abort?.();
      if (keyboard) {
        window.removeEventListener("keydown", onKeyDown);
        window.removeEventListener("keyup", onKeyUp);
        window.removeEventListener("blur", onBlur);
      }
      resizeObserver.disconnect();
      clear();
      controls.dispose();
      tooltip.remove();
      renderer.dispose();
      renderer.forceContextLoss?.();
      renderer.domElement.remove();
    },
    get gen() { return gen; },
  };
}
