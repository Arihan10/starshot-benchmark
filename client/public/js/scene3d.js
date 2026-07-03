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
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const BBOX_COLOR_DEFAULT = 0xff3b3b; // zones
// Objects are colored by the decomposition step that emitted them, so the
// benchmark reader can see at a glance which pass produced each object:
// anchor_decompose (the defining objects) keeps the canonical object green,
// next_object (the completion-loop additions) is purple, and
// negative_space_decompose (interstitial fill) is light brown. An object whose
// origin step is unknown (no provenance folded yet) falls back to the green.
const BBOX_COLOR_OBJECT = 0x6bd96e; // anchor_decompose + default object
const BBOX_COLOR_NEXT_OBJECT = 0xb46aff; // next_object (purple)
const BBOX_COLOR_NEGATIVE_SPACE = 0xc8a06a; // negative_space_decompose (light brown)
const BBOX_COLOR_FRAME = 0x7fb3d5;
const BBOX_COLOR_HOVER = 0xffe14a;
const BBOX_COLOR_SELECTED = 0x4af0e0;
const BBOX_COLOR_OVERLAY = 0xff3df5; // prompt-lab "after" (proposed) boxes
// Facing arrow drawn at the selected object/frame, pointing the way its
// `orientation` yaw faces. Warm so it reads against the cyan selected bbox.
const ORIENTATION_ARROW_COLOR = 0xffa033;

// Object bbox (and its proxy wireframe) color keyed by the decomposition step
// recorded as the node's `emitted_by` provenance. The overlay feeds the step
// via `setOriginOf`; viewers without that wiring fall back to the object green.
const OBJECT_COLOR_BY_STEP = {
	anchor_decompose: BBOX_COLOR_OBJECT,
	next_object: BBOX_COLOR_NEXT_OBJECT,
	negative_space_decompose: BBOX_COLOR_NEGATIVE_SPACE,
};
const BBOX_DIM_OPACITY = 0.35;
const PROXY_BASE_OPACITY = 0.55;
const PROXY_DIM_OPACITY = 0.2;
// Zone-layers view: a faint translucent fill on each zone box so overlapping
// subregions read as a denser patch (two fills stack). Kept low so a lone zone
// is barely tinted; isolate a single depth layer to see sibling overlaps clean.
const ZONE_FILL_OPACITY = 0.08;
const ZONE_FILL_DIM_OPACITY = 0.03;
const TOOLTIP_KIND_COLOR = {
	zone: "#9ad4ff",
	object: "#8bd17c",
	frame: "#7fb3d5",
};

// Zone-layers view: each nesting depth in the zone tree gets its own hue so the
// decomposition's shells are visually separable. Derived from the depth at
// runtime (a fixed hue rotation) — it only has to tell adjacent layers apart,
// not be a stable palette.
function zoneLayerColorHex(depth) {
	const hue = (((depth * 57) % 360) + 360) % 360;
	return new THREE.Color().setHSL(hue / 360, 0.7, 0.58).getHex();
}

const DECODE_WORKERS = Math.min(4, navigator.hardwareConcurrency || 2);
const MESH_BUNDLE_MAGIC = "SMB1";
const MAX_INFLIGHT = 20;
const CLICK_MAX_MOVE_PX = 4;
const CLICK_MAX_DURATION_MS = 400;

// Defaults for the main viewer's lighting engine (exposed so the lighting panel
// can reset to them). Tuned so the directional KEY light shapes the scene while
// the IBL environment + hemisphere are fill/reflections rather than washing it
// flat. azimuth 0°=+Z (front), 90°=+X (right); elevation in degrees.
const LIGHTING_DEFAULTS = {
	exposure: 1.0,
	key: 3.5,
	fill: 0.2,
	env: 0.35,
	shadow: 0.4,
	azimuth: 34,
	elevation: 48,
	shadows: true,
};

export function createViewer(host, { keyboard = true, lighting = false } = {}) {
	const renderer = new THREE.WebGLRenderer({ antialias: true });
	renderer.setPixelRatio(window.devicePixelRatio);
	renderer.setClearColor(0x101114);
	// Physically-based tone mapping + shadows for the MAIN viewer only; the mini /
	// compare viewers keep flat linear shading (lighting=false), so the engine
	// here never regresses them or pays for shadow maps they don't display.
	if (lighting) {
		renderer.toneMapping = THREE.ACESFilmicToneMapping;
		renderer.toneMappingExposure = 1.0;
		renderer.shadowMap.enabled = true;
		renderer.shadowMap.type = THREE.PCFSoftShadowMap;
	}
	host.prepend(renderer.domElement);

	const scene = new THREE.Scene();
	const sceneRoot = new THREE.Group(); // meshes (drives fit-to-scene)
	const bboxRoot = new THREE.Group(); // wireframes (excluded from fit)
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

	// Shared shadow-reception gate (see prepareLoadedScene) — defined even for the
	// simple-lit viewers, where the injected uniform is just unused.
	const _forceReceiveShadow = { value: true };
	const lightingRig = lighting ? setupLightingRig() : null;
	if (!lighting) {
		// Flat fill lighting for the mini / compare viewers.
		const hemi = new THREE.HemisphereLight(0xffffff, 0x202028, 0.9);
		hemi.layers.enableAll(); // also light the OIT layer
		scene.add(hemi);
		const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
		dirLight.position.set(20, 30, 14);
		dirLight.layers.enableAll();
		scene.add(dirLight);
	}

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
	const grid = new THREE.Mesh(
		new THREE.PlaneGeometry(100000, 100000).rotateX(-Math.PI / 2),
		gridMat,
	);
	grid.renderOrder = -1;
	scene.add(grid);

	// KTX2/Basis + Meshopt so optimized-library GLBs parse.
	const ktx2 = new KTX2Loader()
		.setTranscoderPath("/vendor/three/examples/jsm/libs/basis/")
		.setWorkerLimit(DECODE_WORKERS)
		.detectSupport(renderer);
	MeshoptDecoder.useWorkers(DECODE_WORKERS);
	const loader = new GLTFLoader()
		.setKTX2Loader(ktx2)
		.setMeshoptDecoder(MeshoptDecoder);

	const bboxes = new Map(); // id -> Box3Helper
	const proxies = new Map(); // id -> wireframe proxy mesh
	const fills = new Map(); // id -> translucent zone fill mesh (zone-layers view)
	const models = new Map(); // id -> gltf scene
	const kinds = new Map(); // id -> node_kind ("zone"/"object"/"frame")
	// Visibility layers. The object/frame/zone layers are per-CATEGORY toggles —
	// switching one off removes that kind entirely, mesh AND bbox. Objects are
	// split by the decomposition step that emitted them (anchor_decompose →
	// anchors, next_object → next, negative_space_decompose → negativeSpace) so
	// the objects dropdown can filter by origin; an object of unknown origin
	// buckets with anchors. `frames` is the encapsulating-shell kind. meshes/
	// bboxes are cross-cutting: meshes off mutes every mesh but keeps the bboxes;
	// bboxes off mutes every wireframe box but keeps the meshes. proxies is the
	// collision-proxy wireframe layer — an opt-in debug overlay, hidden by
	// default. grid is the floor.
	const show = {
		anchors: true,
		next: true,
		negativeSpace: true,
		frames: true,
		zones: true,
		meshes: true,
		bboxes: true,
		proxies: false,
		grid: true,
	};
	// Zone-layers view (an alternative read of the scene): when on, only zone
	// bboxes draw — each colored by its depth in the decomposition tree — and any
	// combination of depths can be isolated for inspection/picking. Persists
	// across cell swaps like `show`; the isolated set resets per scene (`clear`).
	let zoneLayersMode = false;
	// The depths to show when isolating; empty = every layer. A Set so multiple
	// layers can be active at once (e.g. L0 + L1 together).
	const activeZoneLayers = new Set();
	let gen = 0;
	let fitPending = false;
	let bundleAbort = null;
	// The overlay slides off-screen with translateX, which keeps layout size —
	// so visibility is an explicit flag, gating draws AND keyboard capture.
	let active = false;
	// Optional canonical-axes gizmo (X=red, Y=green, Z=blue, drawn on top) placed
	// at a node's center — the trace panel's mini preview toggles it on to make
	// the baked orientation legible. Added to `scene`, so clear() leaves it be.
	let axesGroup = null;
	// Facing arrow for the selected object (see applyOrientationArrow). Added to
	// `scene` like the axes gizmo, so select()/clear() manage it directly.
	let orientationArrow = null;

	// --- interaction state -----------------------------------------------------

	let hoveredId = null;
	let selectedId = null;
	const raycaster = new THREE.Raycaster();
	raycaster.layers.enableAll(); // OIT meshes live on a non-default layer
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
	// Overlay-provided: the decomposition step that emitted a node (its
	// `emitted_by` provenance), used to color objects by origin. Returns null
	// when unknown — the obs model isn't wired, or provenance hasn't folded yet.
	let originOf = () => null;

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
			if (hiddenIds.has(cur.id) && (isSelf || cur.kind === "zone"))
				return true;
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
			const mats = Array.isArray(n.material)
				? n.material
				: n.material
					? [n.material]
					: [];
			for (const m of mats) disposeMaterial(m);
		});
	}

	// --- colors / visibility (select + hover + dim semantics, ported) -------------

	// Depth of a zone in the decomposition tree (root = 0), by walking parent
	// ids — zones only ever nest under zones, so the ancestor count IS the zone
	// layer. Mirrors `effectivelyHidden`'s guarded walk over the same
	// overlay-provided `nodeInfo`. 0 when the parent chain isn't known yet (the
	// projection paints before history folds; `recolorAll` repaints once it has).
	function zoneDepth(id) {
		let depth = 0;
		let cur = nodeInfo(id);
		let hops = 0;
		while (cur && cur.parentId && hops < 64) {
			const parent = nodeInfo(cur.parentId);
			if (!parent) break;
			depth += 1;
			cur = parent;
			hops += 1;
		}
		return depth;
	}

	// Base (unselected, unhovered) bbox color. Objects key off the decomposition
	// step that emitted them (`originOf`); frames and zones key off node kind.
	// The zone-layers view overrides zones to their depth color.
	function baseBboxColor(id, nodeKind) {
		if (zoneLayersMode && nodeKind === "zone")
			return zoneLayerColorHex(zoneDepth(id));
		if (nodeKind === "object")
			return OBJECT_COLOR_BY_STEP[originOf(id)] ?? BBOX_COLOR_OBJECT;
		if (nodeKind === "frame") return BBOX_COLOR_FRAME;
		return BBOX_COLOR_DEFAULT;
	}

	function applyBboxColor(id) {
		const helper = id !== null ? bboxes.get(id) : null;
		if (!helper) return;
		const color =
			id === selectedId
				? BBOX_COLOR_SELECTED
				: id === hoveredId
					? BBOX_COLOR_HOVER
					: baseBboxColor(id, helper.userData.nodeKind);
		helper.material.color.setHex(color);
		// The proxy wireframe always tracks its object's bbox color.
		const proxy = proxies.get(id);
		if (proxy) proxy.material.color.setHex(color);
		// The zone fill tracks the same color (incl. the select/hover highlight).
		const fill = fills.get(id);
		if (fill) fill.material.color.setHex(color);
	}

	// Whether a node's CATEGORY layer is on. Objects are split by the step that
	// emitted them (`originOf`) so the dropdown filters anchors / next /
	// negative-space independently; an object of unknown origin buckets with
	// anchors (matching its default green). Frames + zones key off node kind.
	function categoryOn(kind, id) {
		if (kind === "frame") return show.frames;
		if (kind === "object") {
			const step = originOf(id);
			if (step === "next_object") return show.next;
			if (step === "negative_space_decompose") return show.negativeSpace;
			return show.anchors; // anchor_decompose + unknown-origin objects
		}
		return show.zones; // "zone" (and any unknown default)
	}

	function applyBboxVisibility(id) {
		// When something is selected, every OTHER bbox is dimmed (not hidden) so
		// the selected one stands out without losing the rest of the scene as
		// spatial reference. Hover gets full opacity so the user can see what
		// they're about to pick. A node's CATEGORY toggle (objects/frames/zones)
		// off removes the whole node, so its box goes too; the bboxes toggle only
		// mutes the wireframe layer — selection/hover still reveal a box so it
		// stays pickable while the layer is off.
		const helper = bboxes.get(id);
		const proxy = proxies.get(id);
		const fill = fills.get(id);
		const kind = helper?.userData?.nodeKind ?? kinds.get(id) ?? "zone";
		// Zone-layers view: only zone wireframes + their fills draw — isolated to
		// one nesting depth when a layer is picked — so the bare decomposition
		// structure reads on its own. Objects/frames and every proxy are dropped.
		if (zoneLayersMode) {
			const inLayer =
				kind === "zone" &&
				(activeZoneLayers.size === 0 ||
					activeZoneLayers.has(zoneDepth(id)));
			const dim =
				selectedId !== null && id !== selectedId && id !== hoveredId;
			if (helper) {
				helper.visible =
					inLayer &&
					(id === selectedId || id === hoveredId || show.bboxes);
				helper.material.opacity = dim ? BBOX_DIM_OPACITY : 1;
			}
			if (proxy) proxy.visible = false;
			// The fill rides with the layer (independent of the wireframe toggle,
			// so `bboxes` off gives a clean fills-only overlap view), dimming
			// alongside the box when another zone is selected.
			if (fill) {
				fill.visible = inLayer;
				fill.material.opacity = dim
					? ZONE_FILL_DIM_OPACITY
					: ZONE_FILL_OPACITY;
			}
			return;
		}
		const visible =
			categoryOn(kind, id) &&
			(id === selectedId || id === hoveredId || show.bboxes);
		const dim =
			selectedId !== null && id !== selectedId && id !== hoveredId;
		if (helper) {
			helper.visible = visible;
			helper.material.opacity = dim ? BBOX_DIM_OPACITY : 1;
		}
		if (proxy) {
			// Proxies are an opt-in debug layer (hidden by default): shown only when
			// their category is on AND the proxies toggle is on, independent of the
			// bbox layer. Selection still dims the non-selected ones.
			proxy.visible = categoryOn(kind, id) && show.proxies;
			proxy.material.opacity = dim
				? PROXY_DIM_OPACITY
				: PROXY_BASE_OPACITY;
		}
		// Zone fills are a zone-layers-only overlay — never shown in the normal
		// view, where they'd just tint the meshes.
		if (fill) fill.visible = false;
	}

	function applyModelVisibility(id) {
		const model = models.get(id);
		if (!model) return;
		// The zone-layers view is a pure wireframe structure view — no meshes.
		if (zoneLayersMode) {
			model.visible = false;
			return;
		}
		// Category off hides this kind's mesh; the meshes layer mutes ALL meshes
		// (bboxes stay); per-node hide (right-click / eye) is independent.
		model.visible =
			categoryOn(kinds.get(id) ?? "zone", id) &&
			show.meshes &&
			!effectivelyHidden(id);
	}

	function refreshAllVisibility() {
		for (const id of bboxes.keys()) applyBboxVisibility(id);
		for (const id of models.keys()) applyModelVisibility(id);
		grid.visible = show.grid;
	}

	// Re-derive every bbox's base color — called after the obs model's provenance
	// folds in (the overlay wires `originOf` to it), since objects painted before
	// their decomposition step was known defaulted to the object green.
	function recolorAll() {
		for (const id of bboxes.keys()) applyBboxColor(id);
	}

	// --- zone-layers view --------------------------------------------------------
	// Hide everything but the zone bboxes and color each by its depth in the
	// decomposition tree, so the nesting reads at a glance. A single depth can be
	// isolated; picking then naturally scopes to it (it respects bbox visibility).
	function setZoneLayers(on) {
		zoneLayersMode = !!on;
		if (!zoneLayersMode) activeZoneLayers.clear();
		recolorAll();
		refreshAllVisibility();
	}

	// Toggle a depth in/out of the isolated set (multi-select). When the set
	// empties out, every layer shows again.
	function toggleZoneLayer(depth) {
		if (activeZoneLayers.has(depth)) activeZoneLayers.delete(depth);
		else activeZoneLayers.add(depth);
		refreshAllVisibility();
	}

	function clearZoneLayers() {
		activeZoneLayers.clear();
		refreshAllVisibility();
	}

	// Canonical-axes gizmo. `setAxes(true, {center, size})` draws the world X/Y/Z
	// axes (Three's default red/green/blue) from `center`, sized to `size`,
	// depth-test off so they read through any mesh; `setAxes(false)` removes it.
	// The axes are world-aligned (there is one global canonical front view), so
	// they're placed AT the inspected node to show how its geometry sits in it.
	function setAxes(on, { center = [0, 0, 0], size = 1 } = {}) {
		if (axesGroup) {
			scene.remove(axesGroup);
			axesGroup.traverse((o) => {
				o.geometry?.dispose?.();
				o.material?.dispose?.();
			});
			axesGroup = null;
		}
		if (!on) return;
		const helper = new THREE.AxesHelper(Math.max(size, 0.1));
		helper.material.depthTest = false;
		helper.material.transparent = true;
		helper.renderOrder = 998;
		axesGroup = new THREE.Group();
		axesGroup.add(helper);
		axesGroup.position.set(center[0] ?? 0, center[1] ?? 0, center[2] ?? 0);
		scene.add(axesGroup);
	}

	function clearOrientationArrow() {
		if (!orientationArrow) return;
		scene.remove(orientationArrow);
		orientationArrow.line?.geometry?.dispose?.();
		orientationArrow.line?.material?.dispose?.();
		orientationArrow.cone?.geometry?.dispose?.();
		orientationArrow.cone?.material?.dispose?.();
		orientationArrow = null;
	}

	// Facing arrow: from the selected node's bbox center, pointing the way its
	// `orientation` yaw faces. Trellis bakes each mesh's front along +Z and the
	// node's orientation yaws that front about +Y (matching rescale_mesh_to_bbox),
	// so the world facing is (sin θ, 0, cos θ). Drawn on top (depth-test off) like
	// the axes gizmo. Only for concrete nodes — zones have no meaningful facing.
	function applyOrientationArrow(id) {
		clearOrientationArrow();
		if (id === null) return;
		const helper = bboxes.get(id);
		const kind = kinds.get(id) ?? helper?.userData?.nodeKind ?? "zone";
		if (!helper || kind === "zone") return;
		// Needs the resolved yaw — absent (no nodeInfo wired, e.g. the mini /
		// compare viewers, or a zone) means no meaningful facing, so no arrow.
		const deg = nodeInfo(id)?.orientation;
		if (typeof deg !== "number") return;
		const theta = (deg * Math.PI) / 180;
		const dir = new THREE.Vector3(Math.sin(theta), 0, Math.cos(theta));
		const center = helper.box.getCenter(new THREE.Vector3());
		const size = helper.box.getSize(new THREE.Vector3());
		const length = Math.max(Math.max(size.x, size.y, size.z) * 0.6, 0.4);
		const arrow = new THREE.ArrowHelper(
			dir,
			center,
			length,
			ORIENTATION_ARROW_COLOR,
			length * 0.3,
			length * 0.16,
		);
		for (const part of [arrow.line, arrow.cone]) {
			part.material.depthTest = false;
			part.material.transparent = true;
			part.renderOrder = 997;
		}
		scene.add(arrow);
		orientationArrow = arrow;
	}

	// The depths currently present among zone bboxes + each one's runtime color,
	// for the overlay's layer legend. Layers are empty unless the mode is on.
	function getZoneLayers() {
		const depths = new Set();
		if (zoneLayersMode) {
			for (const [id, helper] of bboxes) {
				if ((helper.userData?.nodeKind ?? "zone") === "zone")
					depths.add(zoneDepth(id));
			}
		}
		return {
			enabled: zoneLayersMode,
			active: [...activeZoneLayers].sort((a, b) => a - b),
			layers: [...depths]
				.sort((a, b) => a - b)
				.map((d) => ({ depth: d, colorHex: zoneLayerColorHex(d) })),
		};
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
			if (!raycaster.ray.intersectBox(helper.box, _rightClickBoxHit))
				continue;
			const dist = _rightClickBoxHit.distanceToSquared(camera.position);
			if (dist < bestDist) {
				bestDist = dist;
				bestId = id;
			}
		}
		return bestId;
	}

	function pickHoveredId() {
		if (pressedKeys.has("shift"))
			return pickSmallestBoxId({ zonesOnly: true });
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
		if (_pickRoots.length === 0)
			return pickSmallestBoxId({ zonesOnly: false });
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
		if (
			kind !== "zone" &&
			info.imagePrompt &&
			info.imagePrompt !== info.prompt
		) {
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
		applyOrientationArrow(selectedId);
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
		clearOrientationArrow();
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
		return (
			t instanceof HTMLElement &&
			(t.tagName === "INPUT" ||
				t.tagName === "TEXTAREA" ||
				t.tagName === "SELECT" ||
				t.isContentEditable)
		);
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
		const camDist = Math.max(
			1,
			camera.position.distanceTo(controls.target),
		);
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
				offset.multiplyScalar(
					Math.max(0.05, Math.min(4000, dist * factor)) / dist,
				);
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
			geom = new THREE.SphereGeometry(
				0.5,
				24,
				16,
				0,
				Math.PI * 2,
				0,
				Math.PI / 2,
			);
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
			// Placeholder color — applyBboxColor recolors it to match the object's
			// bbox as soon as the proxy is registered (loadBbox, right after this).
			color: BBOX_COLOR_OBJECT,
			wireframe: true,
			transparent: true,
			opacity: PROXY_BASE_OPACITY,
		});
		const mesh = new THREE.Mesh(geom, mat);
		mesh.position.set(cx, anchorY, cz);
		mesh.renderOrder = 1;
		return mesh;
	}

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
		const prevFill = fills.get(id);
		if (prevFill) {
			bboxRoot.remove(prevFill);
			prevFill.geometry?.dispose?.();
			prevFill.material?.dispose?.();
			fills.delete(id);
		}
		const kind = event.node_kind ?? "zone";
		kinds.set(id, kind);
		const fx = origin[0] + dimensions[0];
		const fy = origin[1] + dimensions[1];
		const fz = origin[2] + dimensions[2];
		const box3 = new THREE.Box3(
			new THREE.Vector3(
				Math.min(origin[0], fx),
				Math.min(origin[1], fy),
				Math.min(origin[2], fz),
			),
			new THREE.Vector3(
				Math.max(origin[0], fx),
				Math.max(origin[1], fy),
				Math.max(origin[2], fz),
			),
		);
		const helper = new THREE.Box3Helper(box3, BBOX_COLOR_DEFAULT);
		helper.material.transparent = true;
		helper.material.opacity = 1;
		helper.userData.nodeKind = kind;
		bboxRoot.add(helper);
		bboxes.set(id, helper);
		const proxy = buildProxyWireframe(
			event.proxy_shape,
			origin,
			dimensions,
		);
		if (proxy) {
			bboxRoot.add(proxy);
			proxies.set(id, proxy);
		}
		// Zones get a faint solid fill (in addition to the wireframe) so
		// overlapping subregions show up as a denser patch in the zone-layers
		// view. A unit box scaled to the bbox; depthWrite off + double-sided so
		// it reads from any camera position and STACKS with other fills instead
		// of occluding them. Visibility/color are applied below like the box.
		if (kind === "zone") {
			const fill = new THREE.Mesh(
				new THREE.BoxGeometry(1, 1, 1),
				new THREE.MeshBasicMaterial({
					color: BBOX_COLOR_DEFAULT,
					transparent: true,
					opacity: ZONE_FILL_OPACITY,
					depthWrite: false,
					side: THREE.DoubleSide,
				}),
			);
			fill.position.copy(box3.getCenter(new THREE.Vector3()));
			fill.scale.copy(box3.getSize(new THREE.Vector3()));
			bboxRoot.add(fill);
			fills.set(id, fill);
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
				new THREE.Vector3(
					Math.min(origin[0], fx),
					Math.min(origin[1], fy),
					Math.min(origin[2], fz),
				),
				new THREE.Vector3(
					Math.max(origin[0], fx),
					Math.max(origin[1], fy),
					Math.max(origin[2], fz),
				),
			);
			const helper = new THREE.Box3Helper(box3, BBOX_COLOR_OVERLAY);
			// Draw on top of everything so the proposal reads against the scene.
			helper.material.depthTest = false;
			helper.material.transparent = true;
			helper.renderOrder = 999;
			overlayRoot.add(helper);
		}
	}

	// --- lighting engine (main viewer only) ----------------------------------
	// Image-based lighting (a prefiltered RoomEnvironment) + one shadow-casting
	// key light + a hemisphere fill, with a transparent ground shadow catcher.
	// The shadow frustum refits to the scene bounds on geometry change so contact
	// shadows stay crisp at any scale. Returns the handle the viewer exposes to
	// the lighting panel; only ever called when `lighting` is on.
	function setupLightingRig() {
		{
			const pmrem = new THREE.PMREMGenerator(renderer);
			const envScene = new RoomEnvironment(renderer);
			scene.environment = pmrem.fromScene(envScene, 0.04).texture;
			envScene.dispose();
			pmrem.dispose();
		}
		const hemi = new THREE.HemisphereLight(
			0xffffff,
			0x202028,
			LIGHTING_DEFAULTS.fill,
		);
		hemi.layers.enableAll(); // also light the OIT layer
		scene.add(hemi);
		const SHADOW_MAP_SIZE = 4096;
		const key = new THREE.DirectionalLight(0xffffff, LIGHTING_DEFAULTS.key);
		key.castShadow = true;
		key.layers.enableAll();
		key.shadow.mapSize.set(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
		key.shadow.bias = -0.0001;
		scene.add(key, key.target);
		const catcher = new THREE.Mesh(
			new THREE.PlaneGeometry(1, 1),
			new THREE.ShadowMaterial({
				opacity: LIGHTING_DEFAULTS.shadow,
				depthWrite: false,
			}),
		);
		catcher.rotation.x = -Math.PI / 2;
		catcher.receiveShadow = true;
		catcher.renderOrder = -1;
		scene.add(catcher);

		const lightDir = new THREE.Vector3(4, 8, 6).normalize();
		const center = new THREE.Vector3();
		const dim = new THREE.Vector3();
		let lastBox = null;
		const state = { ...LIGHTING_DEFAULTS };

		function fitShadow(box) {
			const hasGeom = !!box && !box.isEmpty();
			lastBox = hasGeom ? box : null;
			if (hasGeom) {
				box.getCenter(center);
				box.getSize(dim);
			} else {
				center.set(0, 0, 0);
				dim.set(20, 20, 20);
			}
			// Bounding-sphere radius (half the diagonal) so the ortho frustum
			// encloses the whole scene, depth precision spent tightly on it.
			const radius = Math.max(0.5, 0.5 * Math.hypot(dim.x, dim.y, dim.z));
			const minY = hasGeom ? box.min.y : 0;
			const dist = radius * 3;
			key.position.copy(center).addScaledVector(lightDir, dist);
			key.target.position.copy(center);
			key.target.updateMatrixWorld();
			const cam = key.shadow.camera;
			const extent = radius * 1.05;
			cam.left = -extent;
			cam.right = extent;
			cam.top = extent;
			cam.bottom = -extent;
			cam.near = Math.max(0.01, dist - radius * 1.1);
			cam.far = dist + radius * 1.1;
			cam.updateProjectionMatrix();
			// Normal-offset bias scaled to the shadow texel's world size — the main
			// defense against self-shadow acne, kept consistent across scene scales.
			key.shadow.normalBias = ((2 * extent) / SHADOW_MAP_SIZE) * 2.0;
			catcher.position.set(center.x, minY + radius * 0.003, center.z);
			catcher.scale.set(radius * 6, radius * 6, 1);
		}

		function applyAngles() {
			const az = THREE.MathUtils.degToRad(state.azimuth);
			const el = THREE.MathUtils.degToRad(state.elevation);
			const cosEl = Math.cos(el);
			lightDir
				.set(Math.sin(az) * cosEl, Math.sin(el), Math.cos(az) * cosEl)
				.normalize();
			fitShadow(lastBox);
		}

		function applyScalars() {
			renderer.toneMappingExposure = state.exposure;
			key.intensity = state.key;
			hemi.intensity = state.fill;
			scene.environmentIntensity = state.env;
			catcher.material.opacity = state.shadow;
			// Caster stays on; reception is gated through the shared uniform so
			// toggling needs no shader recompile.
			_forceReceiveShadow.value = state.shadows;
			catcher.visible = state.shadows;
		}

		applyScalars();
		fitShadow(null);

		return {
			refit() {
				const box = new THREE.Box3();
				if (sceneRoot.children.length > 0) box.setFromObject(sceneRoot);
				if (box.isEmpty())
					for (const helper of bboxes.values()) box.union(helper.box);
				fitShadow(box.isEmpty() ? null : box);
			},
			setLighting(partial) {
				Object.assign(state, partial);
				applyScalars();
				applyAngles();
			},
			getLighting: () => ({ ...state }),
		};
	}

	// Re-gate a streamed PBR material's shadow term on our shared
	// `_forceReceiveShadow` uniform instead of three's per-object `receiveShadow`
	// (which doesn't reach these materials). Inert where there's no shadow map.
	function patchMaterialReceiveShadow(m) {
		if (!m || m.userData.__recvPatched) return;
		m.userData.__recvPatched = true;
		const prev = m.onBeforeCompile;
		m.onBeforeCompile = (shader, rndr) => {
			if (prev) prev(shader, rndr);
			shader.uniforms.uForceReceiveShadow = _forceReceiveShadow;
			shader.fragmentShader = shader.fragmentShader
				.replace(
					"#include <common>",
					"#include <common>\nuniform bool uForceReceiveShadow;",
				)
				.replace(
					"#include <lights_fragment_begin>",
					THREE.ShaderChunk.lights_fragment_begin.replace(
						/\(\s*directLight\.visible\s*&&\s*receiveShadow\s*\)/g,
						"( directLight.visible && uForceReceiveShadow )",
					),
				);
		};
		const prevKey = m.customProgramCacheKey?.bind(m);
		m.customProgramCacheKey = () =>
			"recvForce1|" + (prevKey ? prevKey() : "");
		m.needsUpdate = true;
	}

	// Every loaded GLB: double-sided (Trellis shells are often single-sided),
	// shadow cast + receive, smooth normals computed when absent (generated meshes
	// ship without them, which otherwise breaks shadow reception), and the
	// receive-shadow re-gate. Transparent meshes are routed to the weighted-blended
	// OIT layer so double-sided alpha composites correctly on concave geometry. All
	// shadow bits are no-ops in the flat mini viewers.
	function prepareLoadedScene(root) {
		root.traverse((child) => {
			if (!child.isMesh) return;
			child.castShadow = true;
			child.receiveShadow = true;
			if (child.geometry && !child.geometry.getAttribute("normal")) {
				child.geometry.computeVertexNormals();
			}
			if (!child.material) return;
			const mats = Array.isArray(child.material)
				? child.material
				: [child.material];
			let oit = false;
			for (const m of mats) {
				m.side = THREE.DoubleSide;
				m.shadowSide = THREE.BackSide;
				patchMaterialReceiveShadow(m);
				if (m.transparent) oit = true;
			}
			if (oit) {
				for (const m of mats) patchMaterialOIT(m);
				child.layers.set(OIT_LAYER);
				child.userData.__oit = true;
			}
		});
	}

	function attachGltf(id, gltfScene, kind) {
		prepareLoadedScene(gltfScene);
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
	async function loadModel(
		event,
		absUrl,
		{ replace = false, onLoaded = null, onError = null } = {},
	) {
		const myGen = gen;
		const k = `${myGen}|${absUrl}`;
		// `replace` lets a changed mesh (same id, new bytes) re-attach over the old
		// one; without it a loaded id is left untouched (attachGltf replaces by id).
		if (failedUrls.has(k) || (!replace && models.has(event.id))) return;
		try {
			const gltf = await loader.loadAsync(absUrl);
			if (myGen !== gen) {
				disposeObject3D(gltf.scene);
				return;
			}
			attachGltf(event.id, gltf.scene, kinds.get(event.id));
			// Hand the parsed scene (+ its world bounds) back so a caller can read
			// the loaded geometry — the per-object trace preview frames the raw mesh
			// on its own bounds and reads its PBR textures for the per-map view.
			if (onLoaded) {
				const box = new THREE.Box3().setFromObject(gltf.scene);
				let bounds = null;
				if (!box.isEmpty()) {
					const c = box.getCenter(new THREE.Vector3());
					const s = box.getSize(new THREE.Vector3());
					bounds = {
						center: [c.x, c.y, c.z],
						size: Math.max(s.x, s.y, s.z),
					};
				}
				onLoaded(gltf.scene, bounds);
			}
		} catch (e) {
			failedUrls.add(k);
			// A superseded load (the scene was cleared / focus moved on) must not
			// fire callbacks — its fallback would attach the wrong node's mesh.
			if (myGen !== gen) return;
			if (onError) {
				onError(e);
				return;
			}
			console.warn(
				`[scene3d] mesh load failed for ${event.id}:`,
				e.message,
			);
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

	function parseGlb(arrayBuffer) {
		return new Promise((resolve, reject) =>
			loader.parse(arrayBuffer, "", resolve, reject),
		);
	}

	// Wrap an already-in-memory byte array as a one-shot stream reader, so the
	// SMB1 parse loop can run over fetched-and-held bytes (the compare view's
	// captured candidate bundles) the same way it does over a live response body.
	function singleChunkReader(uint8) {
		let sent = false;
		return {
			read: async () =>
				sent
					? { done: true, value: undefined }
					: ((sent = true), { done: false, value: uint8 }),
		};
	}

	// Parse the SMB1 framing off `r` and attach each GLB, bounded-concurrency.
	// `only` (a Set of node ids) restricts which meshes are attached — the compare
	// view passes a candidate's NEW mesh ids so the shared prefix isn't re-parsed
	// on every candidate switch (no flicker). Aborts when a newer load supersedes
	// it or the viewer's generation advances.
	async function drainBundle(r, myGen, abort, only) {
		const dec = new TextDecoder();
		const magic = await r.readExact(4);
		if (!magic || dec.decode(magic) !== MESH_BUNDLE_MAGIC) return;
		const inflight = new Set();
		while (true) {
			if (myGen !== gen || abort.signal.aborted) break;
			const idLenB = await r.readExact(4);
			if (!idLenB) break;
			const idB = await r.readExact(
				new DataView(idLenB.buffer).getUint32(0, true),
			);
			if (!idB) break;
			const id = dec.decode(idB);
			const glbLenB = await r.readExact(4);
			if (!glbLenB) break;
			const glbB = await r.readExact(
				new DataView(glbLenB.buffer).getUint32(0, true),
			);
			if (!glbB) break;
			if (only && !only.has(id)) continue; // a prefix mesh we already hold — skip
			const p = (async () => {
				try {
					const gltf = await parseGlb(glbB.buffer);
					if (myGen !== gen || abort.signal.aborted) {
						disposeObject3D(gltf.scene);
						return;
					}
					attachGltf(id, gltf.scene, kinds.get(id));
				} catch {
					/* model-event fallback will fetch it individually */
				}
			})().finally(() => inflight.delete(p));
			inflight.add(p);
			if (inflight.size >= MAX_INFLIGHT) await Promise.race(inflight);
		}
		await Promise.allSettled(inflight);
	}

	// Pull the whole cell's GLBs over ONE connection and attach progressively.
	async function prefetchBundle(meshesUrl) {
		bundleAbort?.abort?.();
		const abort = new AbortController();
		bundleAbort = abort;
		const myGen = gen;
		try {
			const res = await fetch(meshesUrl, {
				cache: "no-store",
				signal: abort.signal,
			});
			if (!res.ok || !res.body) return;
			await drainBundle(
				byteStreamReader(res.body.getReader()),
				myGen,
				abort,
				null,
			);
		} catch {
			/* aborted / network — fallbacks cover it */
		}
		if (myGen === gen) fitToScene();
	}

	// Attach meshes from an SMB1 bundle whose bytes are ALREADY in hand (no fetch).
	// The compare view snapshots each candidate model's bundle when it runs — the
	// branch's objects dir only ever holds the latest model's meshes, so replaying
	// captured bytes is the only way to show an earlier candidate's meshes. Forces
	// a re-attach (attachGltf replaces by id) so a candidate's geometry/placement
	// actually swaps in even though the id is unchanged.
	async function loadBundleBuffer(arrayBuffer, onlyIds = null) {
		if (!arrayBuffer || arrayBuffer.byteLength === 0) return;
		bundleAbort?.abort?.();
		const abort = new AbortController();
		bundleAbort = abort;
		const myGen = gen;
		const only = onlyIds ? new Set(onlyIds) : null;
		try {
			await drainBundle(
				byteStreamReader(
					singleChunkReader(new Uint8Array(arrayBuffer)),
				),
				myGen,
				abort,
				only,
			);
		} catch {
			/* malformed bytes — leave what loaded */
		}
	}

	function clear({ keepCamera = false } = {}) {
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
		for (const fill of fills.values()) {
			bboxRoot.remove(fill);
			fill.geometry?.dispose?.();
			fill.material?.dispose?.();
		}
		clearOverlayBoxes();
		clearOrientationArrow();
		bboxes.clear();
		proxies.clear();
		fills.clear();
		models.clear();
		kinds.clear();
		failedUrls.clear();
		hiddenIds.clear();
		hoveredId = null;
		selectedId = null;
		// The zone-layers MODE persists across cell swaps (a view preference, like
		// `show`); the isolated depths do not — depths differ per scene, so an
		// isolated layer from the previous cell could blank the next one.
		activeZoneLayers.clear();
		tooltip.style.display = "none";
		// keepCamera leaves the camera exactly where it is and marks it
		// user-controlled, which suppresses the incoming scene's auto-fit — so
		// swapping the scene in place (a different LLM/branch of the same view)
		// doesn't yank the user's vantage. A plain clear re-arms the auto-fit so a
		// freshly-opened scene frames itself.
		cameraUserMoved = keepCamera;
	}

	// Drop every loaded GLB mesh while KEEPING the bbox/proxy structure (and the
	// camera + selection). Used when the overlay swaps which build's meshes it
	// shows (asset library ↔ generated, or optimized ↔ raw) so the incoming set
	// replaces the old rather than layering over it. Bumping `gen` aborts any
	// in-flight mesh load so a stale one can't re-attach after the swap.
	function clearMeshes() {
		gen += 1;
		bundleAbort?.abort?.();
		while (sceneRoot.children.length > 0) {
			const child = sceneRoot.children[0];
			sceneRoot.remove(child);
			disposeObject3D(child);
		}
		models.clear();
		failedUrls.clear();
	}

	// Reconcile the scene DOWN to an exact id set: drop every bbox/proxy/mesh no
	// longer present. Painting a /scene projection was purely additive, so moving
	// the cut BACKWARD (scrubbing the compare's original to an earlier step, or
	// reverting a branch) left stale geometry behind. `applySceneProjection` calls
	// this so a projection paint means "show EXACTLY this", in both directions.
	function pruneTo({ bboxIds, meshIds }) {
		for (const id of [...bboxes.keys()]) {
			if (bboxIds.has(id)) continue;
			const helper = bboxes.get(id);
			bboxRoot.remove(helper);
			helper.geometry?.dispose?.();
			helper.material?.dispose?.();
			bboxes.delete(id);
			if (hoveredId === id) hoveredId = null;
			if (selectedId === id) selectedId = null;
		}
		for (const id of [...proxies.keys()]) {
			if (bboxIds.has(id)) continue;
			const proxy = proxies.get(id);
			bboxRoot.remove(proxy);
			proxy.geometry?.dispose?.();
			proxy.material?.dispose?.();
			proxies.delete(id);
		}
		for (const id of [...fills.keys()]) {
			if (bboxIds.has(id)) continue;
			const fill = fills.get(id);
			bboxRoot.remove(fill);
			fill.geometry?.dispose?.();
			fill.material?.dispose?.();
			fills.delete(id);
		}
		for (const id of [...models.keys()]) {
			if (meshIds.has(id)) continue;
			const m = models.get(id);
			sceneRoot.remove(m);
			disposeObject3D(m);
			models.delete(id);
		}
		scheduleFit();
	}

	function scheduleFit() {
		fitPending = true;
	}

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

	// --- weighted-blended OIT (transparent meshes) -------------------------------
	// Transparent GLBs render order-independently: an accumulation pass (additive
	// premultiplied colour x weight) and a revealage pass (product of 1-alpha),
	// both depth-tested against the opaque depth with no depth writes, composited
	// over the opaque image. Keeps double-sided alpha correct on concave meshes
	// (windows, pools, the mostly-opaque car) with no per-triangle sorting. Only
	// engaged while transparent meshes are loaded; otherwise the plain path runs.
	const OIT_LAYER = 1;
	// Depth pre-pass cutoff: fragments at/above this occlude (write depth) so solid
	// surfaces and window frames don't bleed the background; genuine glass below it
	// blends. Sits just above typical window alpha (~0.6-0.78) and below the
	// frame/edge transition (~0.8-1.0) so windows see through but their rims stay solid.
	const OIT_OPAQUE = 0.8;
	const oitPass = { value: 0 };
	const _oitSize = new THREE.Vector2();
	const _oitColor = new THREE.Color();
	let opaqueTarget = null;
	let accumTarget = null;
	let revealTarget = null;

	function patchMaterialOIT(m) {
		if (m.userData.__oitPatched) return;
		m.userData.__oitPatched = true;
		m.transparent = true;
		m.depthWrite = false;
		m.depthTest = true;
		m.blending = THREE.CustomBlending;
		m.blendEquation = THREE.AddEquation;
		m.blendEquationAlpha = THREE.AddEquation;
		const prev = m.onBeforeCompile;
		m.onBeforeCompile = (shader, rndr) => {
			if (prev) prev(shader, rndr);
			shader.uniforms.uOITPass = oitPass;
			shader.fragmentShader = shader.fragmentShader
				.replace(
					"#include <common>",
					"#include <common>\nuniform float uOITPass;",
				)
				.replace(
					"#include <dithering_fragment>",
					`#include <dithering_fragment>
					float _a = gl_FragColor.a;
					if (uOITPass > 1.5) { if (_a < ${OIT_OPAQUE}) discard; }
					else {
						// occluders cover fully so frames/rims stay solid (no partial-
						// alpha bleed of the background); genuine glass keeps its alpha.
						float _ac = _a >= ${OIT_OPAQUE} ? 1.0 : _a;
						if (uOITPass < 0.5) {
							float _w = clamp(pow(min(1.0, _ac * 10.0) + 0.01, 3.0) * 1e8 * pow(1.0 - gl_FragCoord.z * 0.9, 3.0), 1e-2, 3e3);
							gl_FragColor = vec4(gl_FragColor.rgb * _ac * _w, _ac * _w);
						} else {
							gl_FragColor = vec4(_ac);
						}
					}`,
				);
		};
		const prevKey = m.customProgramCacheKey?.bind(m);
		m.customProgramCacheKey = () => "oit1|" + (prevKey ? prevKey() : "");
		m.needsUpdate = true;
	}

	// accum: additive (ONE, ONE); revealage: dst *= (1 - srcAlpha).
	function setOITBlend(m, accum) {
		m.blendSrc = accum ? THREE.OneFactor : THREE.ZeroFactor;
		m.blendDst = accum ? THREE.OneFactor : THREE.OneMinusSrcAlphaFactor;
		m.blendSrcAlpha = m.blendSrc;
		m.blendDstAlpha = m.blendDst;
	}

	// Render-target passes are linear + un-tone-mapped (three only tone-maps to the
	// canvas), so the composite resolves OIT over opaque in linear, then applies
	// the viewer's tone mapping (ACES when lit) + sRGB — matching the plain path.
	const oitCompose = new THREE.ShaderMaterial({
		uniforms: {
			uOpaque: { value: null },
			uAccum: { value: null },
			uReveal: { value: null },
			uExposure: { value: 1 },
			uToneMap: { value: lighting ? 1 : 0 },
		},
		vertexShader:
			"varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
		fragmentShader: `
			uniform sampler2D uOpaque;
			uniform sampler2D uAccum;
			uniform sampler2D uReveal;
			uniform float uExposure;
			uniform float uToneMap;
			varying vec2 vUv;
			vec3 RRTAndODTFit(vec3 v){ vec3 a = v * (v + 0.0245786) - 0.000090537; vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081; return a / b; }
			vec3 aces(vec3 color){
				const mat3 IN = mat3(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777);
				const mat3 OUT = mat3(1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602);
				color *= uExposure / 0.6;
				color = OUT * RRTAndODTFit(IN * color);
				return clamp(color, 0.0, 1.0);
			}
			vec3 enc(vec3 c){ c = clamp(c, 0.0, 1.0); return mix(1.055 * pow(c, vec3(0.4166667)) - 0.055, c * 12.92, step(c, vec3(0.0031308))); }
			void main(){
				vec4 accum = texture2D(uAccum, vUv);
				float reveal = texture2D(uReveal, vUv).r;
				vec3 oit = accum.rgb / max(accum.a, 1e-5);
				vec3 lin = mix(oit, texture2D(uOpaque, vUv).rgb, reveal);
				gl_FragColor = vec4(enc(uToneMap > 0.5 ? aces(lin) : lin), 1.0);
			}
		`,
		depthTest: false,
		depthWrite: false,
		toneMapped: false,
	});
	const oitQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), oitCompose);
	oitQuad.frustumCulled = false;
	const oitQuadScene = new THREE.Scene();
	oitQuadScene.add(oitQuad);
	const oitQuadCamera = new THREE.Camera();

	function ensureOITTargets() {
		renderer.getDrawingBufferSize(_oitSize);
		const w = Math.max(1, _oitSize.x);
		const h = Math.max(1, _oitSize.y);
		if (opaqueTarget && opaqueTarget.width === w && opaqueTarget.height === h)
			return;
		disposeOITTargets();
		// accum + reveal share the opaque depth (depth-test only, no writes).
		const depthTexture = new THREE.DepthTexture(w, h);
		const opts = {
			type: THREE.HalfFloatType,
			minFilter: THREE.NearestFilter,
			magFilter: THREE.NearestFilter,
			depthTexture,
		};
		opaqueTarget = new THREE.WebGLRenderTarget(w, h, opts);
		accumTarget = new THREE.WebGLRenderTarget(w, h, opts);
		revealTarget = new THREE.WebGLRenderTarget(w, h, opts);
		oitCompose.uniforms.uOpaque.value = opaqueTarget.texture;
		oitCompose.uniforms.uAccum.value = accumTarget.texture;
		oitCompose.uniforms.uReveal.value = revealTarget.texture;
	}

	function disposeOITTargets() {
		if (!opaqueTarget) return;
		const depth = opaqueTarget.depthTexture; // shared across the three targets
		opaqueTarget.dispose();
		accumTarget.dispose();
		revealTarget.dispose();
		depth?.dispose();
		opaqueTarget = accumTarget = revealTarget = null;
	}

	const _oitMats = [];
	function renderFrame() {
		_oitMats.length = 0;
		sceneRoot.traverse((o) => {
			if (!o.userData.__oit || !o.material) return;
			if (Array.isArray(o.material)) _oitMats.push(...o.material);
			else _oitMats.push(o.material);
		});
		if (_oitMats.length === 0) {
			renderer.setRenderTarget(null);
			renderer.render(scene, camera);
			return;
		}
		ensureOITTargets();
		renderer.getClearColor(_oitColor);
		const prevAlpha = renderer.getClearAlpha();
		const prevShadowAuto = renderer.shadowMap.autoUpdate;

		// opaque (layer 0) -> opaqueTarget, full clear + shadow map.
		camera.layers.set(0);
		renderer.autoClear = true;
		renderer.setRenderTarget(opaqueTarget);
		renderer.setClearColor(_oitColor, 1);
		renderer.render(scene, camera);

		// transparent (layer 1) -> two OIT passes reusing the opaque depth + shadows.
		renderer.autoClear = false;
		renderer.shadowMap.autoUpdate = false;
		camera.layers.set(OIT_LAYER);

		// depth pre-pass: opaque-ish pixels (alpha >= 0.5) write the nearest depth
		// into the shared buffer so accumulation only sees the front layer.
		// Without it WBOIT averages every layer of a solid mesh into an x-ray wash;
		// genuinely transparent pixels (alpha < 0.5) skip it and still blend.
		oitPass.value = 2;
		for (const m of _oitMats) {
			m.depthWrite = true;
			m.colorWrite = false;
		}
		renderer.setRenderTarget(accumTarget);
		renderer.render(scene, camera);

		oitPass.value = 0;
		for (const m of _oitMats) {
			m.depthWrite = false;
			m.colorWrite = true;
			setOITBlend(m, true);
		}
		renderer.setRenderTarget(accumTarget);
		renderer.setClearColor(0x000000, 0);
		renderer.clear(true, false, false);
		renderer.render(scene, camera);

		oitPass.value = 1;
		for (const m of _oitMats) setOITBlend(m, false);
		renderer.setRenderTarget(revealTarget);
		renderer.setClearColor(0xffffff, 1);
		renderer.clear(true, false, false);
		renderer.render(scene, camera);

		// composite to screen (tone-map + sRGB happen here, see oitCompose).
		camera.layers.set(0);
		renderer.shadowMap.autoUpdate = prevShadowAuto;
		renderer.setClearColor(_oitColor, prevAlpha);
		renderer.autoClear = true;
		renderer.setRenderTarget(null);
		oitCompose.uniforms.uExposure.value = renderer.toneMappingExposure;
		renderer.render(oitQuadScene, oitQuadCamera);
	}

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
		const camDist = Math.max(
			1,
			camera.position.distanceTo(controls.target),
		);
		gridMat.uniforms.uFadeStart.value = camDist * 0.5;
		gridMat.uniforms.uFadeEnd.value = camDist * 6.0;

		if (pointerDirty && !controlsInteracting) {
			pointerDirty = false;
			if (pointerInsideCanvas) {
				raycaster.setFromCamera(pointer, camera);
				const id = pickHoveredId();
				setHovered(id);
				if (id !== null)
					positionTooltip(lastPointerClientX, lastPointerClientY, id);
				else tooltip.style.display = "none";
			}
		}

		if (fitPending) {
			fitPending = false;
			fitToScene();
			lightingRig?.refit();
		}
		renderFrame();
	})();

	return {
		loadBbox,
		loadModel,
		prefetchBundle,
		loadBundleBuffer,
		setOverlayBoxes,
		clearOverlayBoxes,
		setOverlayVisible: (v) => {
			overlayRoot.visible = v;
		},
		setBboxesVisible: (v) => {
			show.bboxes = v;
			refreshAllVisibility();
		},
		clear,
		clearMeshes,
		pruneTo,
		// Lighting engine (main viewer only; no-ops + null elsewhere). The panel
		// (lighting.js) drives setLighting; lightingDefaults seeds its reset.
		setLighting: (partial) => lightingRig?.setLighting(partial),
		getLighting: () => lightingRig?.getLighting() ?? null,
		lightingDefaults: lightingRig ? LIGHTING_DEFAULTS : null,
		hasModel: (id) => models.has(id),
		hasBbox: (id) => bboxes.has(id),
		setKind: (id, kind) => {
			if (kind) kinds.set(id, kind);
		},
		fit: () => fitToScene(true),
		toggles: show,
		refreshVisibility: refreshAllVisibility,
		select,
		clearSelection,
		getSelected: () => selectedId,
		toggleHidden,
		unhideAll: () => {
			if (!hiddenIds.size) return;
			hiddenIds.clear();
			refreshAllVisibility();
			onHiddenChangeCb();
		},
		isHidden: (id) => hiddenIds.has(id),
		setNodeInfo: (fn) => {
			nodeInfo = fn;
		},
		setOriginOf: (fn) => {
			originOf = fn;
		},
		recolorAll,
		setZoneLayers,
		toggleZoneLayer,
		clearZoneLayers,
		getZoneLayers,
		setAxes,
		onSelect: (fn) => {
			onSelectCb = fn;
		},
		onHiddenChange: (fn) => {
			onHiddenChangeCb = fn;
		},
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
		getView: () => ({
			position: camera.position.toArray(),
			target: controls.target.toArray(),
		}),
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
			disposeOITTargets();
			oitCompose.dispose();
			oitQuad.geometry.dispose();
			controls.dispose();
			tooltip.remove();
			renderer.dispose();
			renderer.forceContextLoss?.();
			renderer.domElement.remove();
		},
		get gen() {
			return gen;
		},
	};
}
