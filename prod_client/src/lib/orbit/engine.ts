import {
	Box3,
	Color,
	DirectionalLight,
	Group,
	type Intersection,
	type Material,
	MathUtils,
	Mesh,
	MOUSE,
	type Object3D,
	PerspectiveCamera,
	Plane,
	HemisphereLight,
	type Quaternion,
	Raycaster,
	Scene,
	ShaderMaterial,
	SphereGeometry,
	SRGBColorSpace,
	Vector2,
	Vector3,
	WebGLRenderer,
	WebGLRenderTarget,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { CopyShader } from "three/examples/jsm/shaders/CopyShader.js";
import { groundColor } from "@/lib/ink";
import { loadGLB } from "./loaders";
import {
	DUMMY_TEX,
	makePanoMaterial,
	makePolyMaterial,
	SPHERE_RADIUS,
} from "./materials";
import {
	CURSOR_CLEAR,
	NAV_COLORS,
	PEEK_ROTATE_SPEED,
	WASD_DIR_COS,
} from "./markers";
import {
	DEFAULT_METRICS,
	DEFAULT_SCALE,
	describeScale,
	measureSceneScale,
	type NavMetrics,
	navMetrics,
	type SceneScale,
} from "./scale";
import { SurfaceCursor } from "./cursor";
import { LightRig } from "./lighting";
import { prepareLitScene } from "./prepare";
import { MarkerLayer } from "./markerLayer";
import {
	SplatLayer,
	IDENTITY_TRANSFORM,
	type SplatTransform,
} from "./splatLayer";
import { collectObjects, ObjectAddressing } from "./objectAddressing";
import { type PanoEntry, PanoStreamer } from "./panoTextures";
import { Projection } from "./projection";
import {
	buildMinimapState,
	levelForPosition,
	readBasis,
	toMap,
	type MapLabel,
	type MinimapSlice,
} from "./minimap";
import {
	angleDelta,
	buildNavGraph,
	type EdgeType,
	edgeVerb,
	type NavEdge,
	type NavGraph,
	type NavNode,
} from "./navGraph";
import { PASS_DUR_SCALE, planZoneTour, TourDirector } from "./tourDirector";
import {
	applyLook,
	cursorRayDir,
	forwardToLonLat,
	lookTargetFrom,
	MAX_PITCH,
	pinLook,
} from "./look";
import type {
	Chapter,
	Connector,
	NodeDir,
	OrbitMode,
	OrbitState,
	TourManifest,
	TourSource,
} from "./types";

const v3 = (a: [number, number, number]) => new Vector3().fromArray(a);

const describeObject = (o: Object3D): string => {
	for (let n: Object3D | null = o; n; n = n.parent) {
		const label = n.userData?.objLabel as string | undefined;
		if (n.name)
			return label && label !== n.name ? `${n.name} [${label}]` : n.name;
		if (label) return `[${label}]`;
	}
	return o.type;
};
const fmt3 = (v: ArrayLike<number>) =>
	`(${v[0].toFixed(2)}, ${v[1].toFixed(2)}, ${v[2].toFixed(2)})`;
const easeInOut = (t: number) =>
	t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;

const DUR: Record<EdgeType, number> = {
	walk: 700,
	portal: 900,
	phase: 1200,
	vertical: 1100,
	far: 2400,
};
const REDUCED_DUR = 260;
const DWELL_MS = 8500;

const TOUR_PAN_MS = 10000;
const TOUR_PAN_MS_REDUCED = 15000;

const OVERVIEW_FOV = 55;
const INTERIOR_FOV = 75;
const ENTER_CROSSFADE_MS = 450;
const HANDOVER_WAIT_MS = 1000;

const SPLAT_REVEAL_MS = 320;
const REVEAL_FOV_MS_PER_DEG = 10;
const SPLAT_REVEAL_MAX_MS = 700;
const FREEFLY_FOV = INTERIOR_FOV;

const CENTER_CURSOR = true;
const _aim = { x: 0, y: 0 };

const RETICLE_ARM = 5;
const RETICLE_GAP = 3;
const RETICLE_THICK = 1;
const RETICLE_ON_SURFACE = 0.9;
const RETICLE_IN_VOID = 0.32;

const LOOK_SENSITIVITY = 0.002;

const ZOOM_MIN_FOV = 45;
const ZOOM_MAX_FOV = 90;
const ZOOM_PER_NOTCH = 0.12;

const DOLLY_NEAR = 0.44;
const DOLLY_FAR = 1.32;
const DOLLY_PER_STEP = 0.19;
const ZOOM_SLACK = 1.002;
const ZOOM_EASE = 7.5;
const ZOOM_SETTLED = 0.0015;
const FREEFLY_SPEED_MIN = 0.5;
const FREEFLY_SPEED_MAX = 2;
const FREEFLY_RETURN_MS = 1150;
const DISSOLVE_START = 0.32;
const FREEFLY_SPEED_FRAC = 0.18;
const FREEFLY_VEL_TAU = 90;

const DOCK_STILL_SPEED_FRAC = 0.01;
const DOCK_STILL_MS = 500;
const DOCK_DY_WEIGHT = 1.5;
const DOCK_SEEK_GAIN = 2.6;
const DOCK_REVEAL_TAU = 150;

const LOOK_GLIDE_TAU = 420;
const LOOK_SAMPLE_TAU = 90;
const LOOK_VEL_MIN = 0.012;
const FREEFLY_MOVE_KEYS = new Set([
	"KeyW",
	"KeyA",
	"KeyS",
	"KeyD",
	"KeyQ",
	"KeyE",
	"Space",
	"Shift",
]);
const FREEFLY_ENTER_KEYS = new Set([
	"KeyW",
	"KeyA",
	"KeyS",
	"KeyD",
	"Space",
	"Shift",
]);

function buildReticle(): HTMLDivElement {
	const el = document.createElement("div");
	Object.assign(el.style, {
		position: "absolute",
		left: "50%",
		top: "50%",
		width: "0",
		height: "0",
		pointerEvents: "none",
		zIndex: "4",
		opacity: "0",
		filter: "drop-shadow(0 0 1px rgb(var(--ground-rgb) / 0.9))",
		transition: "opacity 120ms linear",
	});
	const off = RETICLE_GAP + RETICLE_ARM / 2;
	const ticks: Array<[number, number, number, number]> = [
		[RETICLE_THICK, RETICLE_ARM, 0, -off],
		[RETICLE_THICK, RETICLE_ARM, 0, off],
		[RETICLE_ARM, RETICLE_THICK, -off, 0],
		[RETICLE_ARM, RETICLE_THICK, off, 0],
	];
	for (const [w, h, x, y] of ticks) {
		const tick = document.createElement("div");
		Object.assign(tick.style, {
			position: "absolute",
			width: `${w}px`,
			height: `${h}px`,
			background: "rgb(var(--mark-rgb))",
			transform: `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`,
		});
		el.appendChild(tick);
	}
	return el;
}

function freeflyKey(code: string): string | null {
	if (code === "ShiftLeft" || code === "ShiftRight") return "Shift";
	return FREEFLY_MOVE_KEYS.has(code) ? code : null;
}

const INSPECT_DWELL_MS = 1750;
const INSPECT_SIZE = 190;
const INSPECT_GAP = 18;
const INSPECT_MARGIN = 12;
const INSPECT_SPIN = 0.55;

const _cursorNdc = new Vector2();
const _bez = new Vector3();
const _flyDir = new Vector3();
const _moveWish = new Vector3();
const _prevClear = new Color();
const _ghostFloor = new Vector3();
const _ovTravel = new Vector3();
const _wpDir = new Vector3();
const _wpOut = new Vector3();
const _wpEye = new Vector3();
const _dropFrom = new Vector3();
const _walkDir = new Vector3();
const _walkPt = new Vector3();
const _walkOut = new Vector3();
const _walkNrm = new Vector3();
const _walkFrom = new Vector3();
const _walkAlt = new Vector3();
const WALK_HEIGHTS = [0.15, 0.5, 0.9] as const;
function fitDistance(
	hull: Vector3[],
	centre: Vector3,
	dir: Vector3,
	fovDeg: number,
	aspect: number,
): number {
	const forward = _fitFwd.copy(dir).normalize();
	const seed = Math.abs(forward.y) > 0.95 ? _fitAltUp : _fitUp;
	const right = _fitRight.crossVectors(seed, forward).normalize();
	const up = _fitCamUp.crossVectors(forward, right).normalize();
	const tanFull = Math.tan((fovDeg * Math.PI) / 360);
	const tanH = tanFull * (aspect > 0 ? aspect : 1);
	const tanV = tanFull * ((SAFE_HI - SAFE_LO) / 2);

	let need = 0;
	for (const point of hull) {
		_fitCorner.copy(point).sub(centre);
		const depth = _fitCorner.dot(forward);
		const x = Math.abs(_fitCorner.dot(right));
		const y = Math.abs(_fitCorner.dot(up));
		need = Math.max(need, depth + x / tanH, depth + y / tanV);
	}
	return need;
}

function framingHull(box: Box3, centre: Vector3): Vector3[] {
	const r = Math.max(
		Math.hypot(box.max.x - centre.x, box.max.z - centre.z),
		Math.hypot(box.max.x - centre.x, box.min.z - centre.z),
		Math.hypot(box.min.x - centre.x, box.max.z - centre.z),
		Math.hypot(box.min.x - centre.x, box.min.z - centre.z),
	);
	const hull: Vector3[] = [];
	for (let i = 0; i < HULL_SEGMENTS; i++) {
		const a = (i / HULL_SEGMENTS) * Math.PI * 2;
		const x = centre.x + r * Math.cos(a);
		const z = centre.z + r * Math.sin(a);
		hull.push(new Vector3(x, box.min.y, z), new Vector3(x, box.max.y, z));
	}
	return hull;
}

const HULL_SEGMENTS = 24;

function groundAnchor(box: Box3, centre: Vector3): Vector3 {
	return new Vector3(centre.x, box.min.y, centre.z);
}

function centrePan(
	camera: PerspectiveCamera,
	hull: Vector3[],
	pos: Vector3,
	target: Vector3,
	up: number,
	upAxis: Vector3,
): number {
	camera.position.copy(pos).addScaledVector(upAxis, up);
	camera.lookAt(_panAim.copy(target).addScaledVector(upAxis, up));
	camera.updateMatrixWorld(true);
	const right = _centreRight
		.setFromMatrixColumn(camera.matrixWorld, 0)
		.normalize();
	let pan = 0;

	for (let pass = 0; pass < 4; pass++) {
		camera.position
			.copy(pos)
			.addScaledVector(upAxis, up)
			.addScaledVector(right, pan);
		camera.lookAt(
			_panAim
				.copy(target)
				.addScaledVector(upAxis, up)
				.addScaledVector(right, pan),
		);
		camera.updateMatrixWorld(true);

		let lo = Infinity;
		let hi = -Infinity;
		for (const point of hull) {
			const x = _panCorner.copy(point).project(camera).x;
			if (x < lo) lo = x;
			if (x > hi) hi = x;
		}
		const off = (lo + hi) / 2;
		if (Math.abs(off) < 0.002) break;

		const half =
			camera.position.distanceTo(target) *
			Math.tan((camera.fov * Math.PI) / 360) *
			camera.aspect;
		pan += off * half;
	}
	return pan;
}

// A RATE PER SECOND, not per frame — see the update(dt) in tick. OrbitControls
// turns `2π / 60 * speed` radians a second, so this is one orbit every 28.6s on
// every display. The number went back UP when the clock changed: at 1.05 it was
// reading double on a 120Hz screen, which is the speed that was signed off.
const BROWSE_SPIN_SPEED = 2.1;

const _centreRight = new Vector3();
const _fitFwd = new Vector3();
const _fitRight = new Vector3();
const _fitCamUp = new Vector3();
const _fitCorner = new Vector3();
const _fitUp = new Vector3(0, 1, 0);
const _fitAltUp = new Vector3(0, 0, 1);

const BROWSE_DIR = new Vector3(0.7, 0.5, 0.9).normalize();
const BROWSE_MARGIN = 0.78;

export const GROUND_LINE = 0.58;

const SAFE_TOP = 0.14;
const SAFE_BOTTOM = 0.15;
const SAFE_HI = 1 - 2 * SAFE_TOP;
const SAFE_LO = -1 + 2 * SAFE_BOTTOM;

function groundLinePan(
	camera: PerspectiveCamera,
	hull: Vector3[],
	anchor: Vector3,
	pos: Vector3,
	target: Vector3,
): number {
	const want = 1 - 2 * GROUND_LINE;

	camera.position.copy(pos);
	camera.lookAt(target);
	camera.updateMatrixWorld(true);
	const up = _panUp.setFromMatrixColumn(camera.matrixWorld, 1).normalize();
	let pan = 0;

	for (let pass = 0; pass < 6; pass++) {
		camera.position.copy(pos).addScaledVector(up, pan);
		camera.lookAt(_panAim.copy(target).addScaledVector(up, pan));
		camera.updateMatrixWorld(true);

		let lo = Infinity;
		let hi = -Infinity;
		for (const point of hull) {
			const y = _panCorner.copy(point).project(camera).y;
			if (y < lo) lo = y;
			if (y > hi) hi = y;
		}
		const stand = _panCorner.copy(anchor).project(camera).y;

		const owed = want - stand;
		const step = Math.max(SAFE_LO - lo, Math.min(SAFE_HI - hi, owed));
		if (Math.abs(step) < 0.002) break;

		const half =
			camera.position.distanceTo(target) *
			Math.tan((camera.fov * Math.PI) / 360);
		pan -= step * half;
	}
	return pan;
}

const _panCorner = new Vector3();
const _panAim = new Vector3();
const _panUp = new Vector3();
const _panShift = new Vector3();
const _dolly = new Vector3();

const _DOWN = new Vector3(0, -1, 0);
const _losFrom = new Vector3();
const _losDir = new Vector3();
const quadBezier = (
	a: Vector3,
	c: Vector3,
	b: Vector3,
	t: number,
	out: Vector3,
) => {
	const u = 1 - t;
	return out
		.copy(a)
		.multiplyScalar(u * u)
		.addScaledVector(c, 2 * u * t)
		.addScaledVector(b, t * t);
};

type Transition = {
	fromPos: Vector3;
	toPos: Vector3;
	fromQuat: Quaternion;
	toQuat: Quaternion;
	fromFov: number;
	toFov: number;
	start: number;
	dur: number;
	crossfade: boolean;
	dissolveInterior: boolean;
	onMid?: () => void;
	onEnd?: () => void;
	midDone: boolean;
};

type Crossfade = {
	armed: number;
	deadline: number;
	dur: number;
	direction: "in" | "out";
	onEnd?: () => void;
};
type Move = {
	fromPos: Vector3;
	toPos: Vector3;
	ctrl: Vector3 | null;
	start: number;
	dur: number;
	index: number;
	type: EdgeType;
	dy: number;
	sphere: boolean;
	pass: boolean;
};
type ReachTarget = { index: number; level: number; levelDelta: number };

type SavedInterior = {
	pos: Vector3;
	lon: number;
	lat: number;
	index: number;
	fov: number;
};

const yieldFrame = () => new Promise<void>((r) => setTimeout(r, 0));

export type PreparedTour = {
	entries: PanoEntry[];
	proxyRoot: Group | null;
	lite: Group | null;
	connectors: Connector[];
	objectIds: string[];
	minimaps: MinimapSlice[];
	minimapPrefetch: HTMLImageElement[];
	mapLabels: MapLabel[];
	mapBasis: ReturnType<typeof readBasis>;
	levelWord: string;
	panoLevel: number[];
	splatReady: boolean;
	splatTransform?: SplatTransform;
	box: Box3 | null;
	center?: Vector3 | null;
	sceneScale: SceneScale;
	metrics: NavMetrics;
	navGraph: NavGraph | null;
};

export class OrbitEngine {
	private readonly host: HTMLElement;
	private readonly onState: (s: OrbitState) => void;
	private readonly onHold?: (held: boolean) => void;

	private readonly onInside?: (inside: boolean) => void;
	private inside = false;

	private setInside(next: boolean) {
		if (this.inside === next) return;
		this.inside = next;
		this.onInside?.(next);
	}

	private readonly renderer: WebGLRenderer;
	private readonly canvas: HTMLCanvasElement;
	private captureWaiting: ((c: HTMLCanvasElement | null) => void) | null =
		null;
	private readonly travelFade: HTMLDivElement;
	private readonly iris: HTMLDivElement;
	private readonly reticle: HTMLDivElement;
	private readonly sonarLabels: HTMLDivElement[] = [];
	private readonly scene: Scene;
	private readonly camera: PerspectiveCamera;
	private readonly controls: OrbitControls;
	private readonly rig: LightRig;
	private readonly ro: ResizeObserver;

	private readonly composer: EffectComposer;

	private readonly sphereA: Mesh;
	private readonly sphereAMat: ShaderMaterial;
	private readonly sphereB: Mesh;
	private readonly sphereBMat: ShaderMaterial;
	private readonly polyMaterial = makePolyMaterial();

	private readonly streamer: PanoStreamer;
	private readonly projection = new Projection();
	private readonly markers: MarkerLayer;
	private readonly addressing: ObjectAddressing;
	private readonly director: TourDirector;
	private readonly requestPano = (i: number) => this.streamer.request(i);

	private readonly dummyCam = new PerspectiveCamera();

	private readonly cursor: SurfaceCursor;
	private readonly cursorRay = new Raycaster();
	private readonly occluder = new Raycaster();
	private readonly dropRay = new Raycaster();
	private readonly walkRay = new Raycaster();
	private aimBlock: {
		source: string;
		object: string;
		face: number;
		dist: number;
		point: [number, number, number];
		planNormal: [number, number, number] | null;
	} | null = null;
	private locked = false;
	private lockClickPending = false;
	private pointerClientX = 0;
	private pointerClientY = 0;
	private pointerInside = false;
	private pointerDown = false;

	private overviewTarget = -1;
	private dollyGoal: number | null = null;
	private fovGoal: number | null = null;
	private lockYielded = false;
	private overviewHit: Intersection | null = null;
	private overviewAimX = -1;
	private overviewAimY = -1;
	private readonly overviewCam = new Vector3();
	private readonly overviewPivot = new Vector3();

	private currentIndex = -1;
	private flyTarget = -1;
	private projectionMode = false;
	private minimaps: MinimapSlice[] = [];
	private mapBasis = readBasis(undefined);
	private levelWord = "floor";
	private mapLabels: MapLabel[] = [];
	private panoLevel: number[] = [];
	private minimapPrefetch: HTMLImageElement[] = [];
	private liteRoot: Group | null = null;
	private proxyGroup: Group | null = null;
	private sharedOverview = false;
	private proxyView = false;
	private proxyColorMats: Material[] = [];
	private connectors: Connector[] = [];

	private readonly splat: SplatLayer;
	private splatView = true;
	private splatReveal = 0;
	private splatRevealing = false;
	private splatRevealMs = SPLAT_REVEAL_MS;
	private revealFovFrom = FREEFLY_FOV;
	private freeflySpeed = 1;
	private readonly freeflyKeys = new Set<string>();
	private readonly freeflyVel = new Vector3();
	private readonly lookVel = { lon: 0, lat: 0 };
	private lookSampledAt = 0;
	private dockTarget = -1;
	private dockReveal = 0;
	private freeflyFrom = -1;
	private dockStaged = false;
	private dockStillSince = 0;
	private dockDelayMs = DOCK_STILL_MS;
	private inspectable = new Set<string>();
	private liteByLabel = new Map<string, Object3D>();
	private inspectScene: Scene | null = null;
	private inspectCam = new PerspectiveCamera(45, 1, 0.01, 100);
	private inspectPivot: Group | null = null;
	private inspect: OrbitState["inspect"] = null;
	private hoverLabel: string | null = null;
	private hoverSince = 0;
	private rcDownX = 0;
	private rcDownY = 0;

	private navGraph: NavGraph | null = null;
	private warmed: { source: TourSource; prepared: PreparedTour } | null =
		null;
	private warming: {
		source: TourSource;
		promise: Promise<PreparedTour | null>;
	} | null = null;
	private nodeDir: NodeDir[] = [];
	private chapters: Chapter[] = [];

	private history: number[] = [];
	private visited = new Set<number>();
	private pendingTravel: number | null = null;
	private arrival: OrbitState["arrival"] = null;

	private mode: OrbitMode = "empty";
	private readonly bgColor: Color | null = null;
	private readonly sceneCenter = new Vector3();
	private sceneMaxDim = 1;
	private sceneTopY = 0;
	private sceneBottomY = 0;
	private sceneScale: SceneScale = DEFAULT_SCALE;
	private metrics: NavMetrics = DEFAULT_METRICS;
	private framingHull: Vector3[] = [];
	private readonly groundAnchor = new Vector3();
	private readonly browsePos = new Vector3();
	private readonly browseTarget = new Vector3();

	private transition: Transition | null = null;
	private move: Move | null = null;
	private crossfade: Crossfade | null = null;

	private lon = 0;
	private lat = 0;
	private dragging = false;
	private dragMoved = 0;
	private downX = 0;
	private downY = 0;
	private readonly grabDir = new Vector3();
	private highlightEnabled = true;

	private interiorBusy = false;
	private exitWhenLanded = false;
	private savedInterior: SavedInterior | null = null;
	private peekHeld = false;
	private readonly locateClip = new Plane(new Vector3(0, -1, 0), 0);

	private hoveredNavIndex = -1;
	private cursorReach: ReachTarget | null = null;
	private arrowReach: ReachTarget | null = null;
	private lastInputAt = 0;
	private dwellPulsed = false;
	private readonly reducedMotion =
		typeof window !== "undefined" &&
		window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

	private autoRotateTimer: ReturnType<typeof setTimeout> | null = null;
	private lastFrame = 0;
	private overlay: OrbitState["overlay"] = null;

	private loadToken = 0;
	private disposed = false;

	constructor(
		host: HTMLElement,
		onState: (s: OrbitState) => void,
		onHold?: (held: boolean) => void,
		onInside?: (inside: boolean) => void,
	) {
		this.host = host;
		this.onState = onState;
		this.onHold = onHold;
		this.onInside = onInside;

		this.renderer = new WebGLRenderer({ antialias: false, alpha: true });
		this.renderer.setPixelRatio(window.devicePixelRatio);
		this.canvas = this.renderer.domElement;
		Object.assign(this.canvas.style, {
			display: "block",
			width: "100%",
			height: "100%",
			position: "relative",
			zIndex: "1",
		});
		host.appendChild(this.canvas);
		this.splat = new SplatLayer(host);

		this.travelFade = document.createElement("div");
		Object.assign(this.travelFade.style, {
			position: "absolute",
			inset: "0",
			background: "rgb(var(--ground-rgb))",
			opacity: "0",
			pointerEvents: "none",
			zIndex: "1",
		});
		host.appendChild(this.travelFade);

		this.iris = document.createElement("div");
		Object.assign(this.iris.style, {
			position: "absolute",
			inset: "0",
			opacity: "0",
			pointerEvents: "none",
			zIndex: "2",
		});
		host.appendChild(this.iris);

		this.reticle = buildReticle();
		host.appendChild(this.reticle);

		this.scene = new Scene();
		this.scene.background = this.bgColor;
		this.rig = new LightRig(this.renderer, this.scene);

		this.camera = new PerspectiveCamera(60, 1, 0.05, 2000);
		this.camera.position.set(4, 3, 5);

		this.controls = new OrbitControls(this.camera, this.canvas);
		this.controls.enableDamping = true;
		this.controls.dampingFactor = 0.12;
		this.controls.screenSpacePanning = true;
		this.controls.zoomToCursor = true;
		this.controls.autoRotate = true;
		this.controls.autoRotateSpeed = BROWSE_SPIN_SPEED;
		this.controls.mouseButtons = {
			LEFT: MOUSE.ROTATE,
			MIDDLE: MOUSE.PAN,
			RIGHT: MOUSE.PAN,
		};
		this.controls.enabled = false;
		this.controls.addEventListener("start", this.onControlsStart);
		this.controls.addEventListener("end", this.onControlsEnd);

		this.sphereAMat = makePanoMaterial();
		this.sphereBMat = makePanoMaterial();
		this.sphereAMat.uniforms.map.value = DUMMY_TEX;
		this.sphereBMat.uniforms.map.value = DUMMY_TEX;
		this.sphereA = new Mesh(
			new SphereGeometry(SPHERE_RADIUS, 64, 32),
			this.sphereAMat,
		);
		this.sphereB = new Mesh(this.sphereA.geometry, this.sphereBMat);
		this.sphereA.renderOrder = 0;
		this.sphereB.renderOrder = 1;
		this.sphereA.visible = false;
		this.sphereB.visible = false;
		this.scene.add(this.sphereA, this.sphereB);

		this.markers = new MarkerLayer(this.scene);
		this.cursor = new SurfaceCursor(this.scene);
		this.streamer = new PanoStreamer(
			() => this.loadToken,
			(i) => this.onPanoReady(i),
		);
		this.addressing = new ObjectAddressing(
			this.scene,
			this.camera,
			this.canvas,
		);
		this.director = new TourDirector(
			{
				busy: () => this.interiorBusy,
				hop: (index, pass) => this.traverse(index, false, pass),
				getLook: () => ({ lon: this.lon, lat: this.lat }),
				setLook: (lon, lat) => {
					this.lon = lon;
					this.lat = lat;
				},
				onProgress: () => this.emit(),
			},
			this.reducedMotion ? TOUR_PAN_MS_REDUCED : TOUR_PAN_MS,
		);

		const composerRT = new WebGLRenderTarget(1, 1, { samples: 4 });
		composerRT.texture.colorSpace = SRGBColorSpace;
		this.composer = new EffectComposer(this.renderer, composerRT);
		this.composer.setPixelRatio(this.renderer.getPixelRatio());
		this.composer.addPass(new RenderPass(this.scene, this.camera));
		this.composer.addPass(this.addressing.fillPass);
		this.composer.addPass(this.addressing.selectPass);
		this.composer.addPass(this.addressing.hoverPass);
		this.composer.addPass(new ShaderPass(CopyShader));

		this.canvas.addEventListener("contextmenu", this.onContextMenu);
		this.canvas.addEventListener("pointerdown", this.onPointerDown);
		this.canvas.addEventListener("pointermove", this.onPointerMove);
		this.canvas.addEventListener("pointerleave", this.onPointerLeave);
		this.canvas.addEventListener("pointerup", this.onPointerUp);
		this.canvas.addEventListener("wheel", this.onWheel, { passive: false });
		this.canvas.addEventListener("click", this.onClick);
		window.addEventListener("pointerup", this.onWindowPointerUp);
		window.addEventListener("keydown", this.onKeyDown);
		window.addEventListener("keyup", this.onKeyUp);
		window.addEventListener("blur", this.onWindowBlur);
		document.addEventListener(
			"pointerlockchange",
			this.onPointerLockChange,
		);
		document.addEventListener("mousemove", this.onLockedMouseMove);

		this.ro = new ResizeObserver(() => {
			this.resizePending = true;
		});
		this.ro.observe(host);
		this.resize();
		this.renderer.setAnimationLoop(this.tick);
		this.emit();
	}

	dispose() {
		this.disposed = true;
		this.loadToken++;
		this.captureWaiting?.(null);
		this.captureWaiting = null;
		this.renderer.setAnimationLoop(null);
		this.ro.disconnect();
		this.controls.removeEventListener("start", this.onControlsStart);
		this.controls.removeEventListener("end", this.onControlsEnd);
		this.controls.dispose();
		this.canvas.removeEventListener("contextmenu", this.onContextMenu);
		this.canvas.removeEventListener("pointerdown", this.onPointerDown);
		this.canvas.removeEventListener("pointermove", this.onPointerMove);
		this.canvas.removeEventListener("pointerleave", this.onPointerLeave);
		this.canvas.removeEventListener("pointerup", this.onPointerUp);
		this.canvas.removeEventListener("wheel", this.onWheel);
		this.canvas.removeEventListener("click", this.onClick);
		window.removeEventListener("pointerup", this.onWindowPointerUp);
		window.removeEventListener("keydown", this.onKeyDown);
		window.removeEventListener("keyup", this.onKeyUp);
		window.removeEventListener("blur", this.onWindowBlur);
		document.removeEventListener(
			"pointerlockchange",
			this.onPointerLockChange,
		);
		document.removeEventListener("mousemove", this.onLockedMouseMove);
		this.releaseLock();
		if (this.autoRotateTimer) clearTimeout(this.autoRotateTimer);
		this.clearScene();
		this.splat.dispose();
		this.cursor.dispose();
		this.markers.dispose();
		this.rig.dispose();
		for (const pass of this.composer.passes) pass.dispose();
		this.composer.dispose();
		this.renderer.dispose();
		this.canvas.remove();
		this.travelFade.remove();
		this.iris.remove();
		this.reticle.remove();
		for (const l of this.sonarLabels) l.remove();
	}

	private get panos(): PanoEntry[] {
		return this.streamer.list;
	}

	private navNode(i: number): NavNode | null {
		return this.navGraph && i >= 0
			? (this.navGraph.nodes[i] ?? null)
			: null;
	}

	private edgeBetween(from: number, to: number): NavEdge | null {
		const node = this.navNode(from);
		return node?.all.find((e) => e.to === to) ?? null;
	}

	private noteInput() {
		this.lastInputAt = performance.now();
		this.dwellPulsed = false;
	}

	private requestLock() {
		if (this.locked) return;
		const pending = this.canvas.requestPointerLock?.() as unknown;
		if (pending instanceof Promise) pending.catch(() => {});
	}

	private releaseLock() {
		if (!this.locked) return;
		this.lockYielded = true;
		document.exitPointerLock?.();
	}

	// Escape while the pointer is locked is swallowed by the browser to release
	// the lock, so the keydown never lands. An unlock we did not ask for, with
	// the tab still focused, is that keypress — leave on it, or Escape reads as
	// needing two presses.
	private onPointerLockChange = () => {
		const locked = document.pointerLockElement === this.canvas;
		if (locked === this.locked) return;
		this.locked = locked;
		if (locked) {
			this.dragging = false;
			this.canvas.style.cursor = "";
		}
		this.stopLookInertia();
		this.emit();
		if (locked) return;
		if (this.lockYielded) {
			this.lockYielded = false;
			return;
		}
		if (!document.hasFocus()) return;
		if (this.mode === "freefly") {
			this.returnToInterior(this.nearestPanoTo(this.camera.position));
			return;
		}
		if (this.mode === "interior") this.leaveInterior();
	};

	private leaveInterior() {
		this.markers.hideSonar();
		this.yieldTour();
		if (this.interiorBusy) {
			this.exitWhenLanded = true;
			this.hurryMove(160);
			return;
		}
		this.exit();
	}

	private onLockedMouseMove = (ev: MouseEvent) => {
		if (!this.locked || !this.isLookMode || this.interiorBusy) return;
		this.lon += (ev.movementX || 0) * LOOK_SENSITIVITY;
		this.lat -= (ev.movementY || 0) * LOOK_SENSITIVITY;
		this.noteInput();
	};

	private aim(): { x: number; y: number } {
		if (!CENTER_CURSOR) {
			_aim.x = this.pointerClientX;
			_aim.y = this.pointerClientY;
			return _aim;
		}
		const r = this.canvas.getBoundingClientRect();
		_aim.x = r.left + r.width / 2;
		_aim.y = r.top + r.height / 2;
		return _aim;
	}

	private get isLookMode(): boolean {
		return this.mode === "interior" || this.mode === "freefly";
	}

	private stopLookInertia() {
		this.lookVel.lon = 0;
		this.lookVel.lat = 0;
		this.lookSampledAt = 0;
	}

	private tickLook(dt: number) {
		if (this.dragging) {
			const stale = Math.exp(-(dt * 1000) / LOOK_SAMPLE_TAU);
			this.lookVel.lon *= stale;
			this.lookVel.lat *= stale;
		} else if (!this.reducedMotion && !this.director.active) {
			if (Math.hypot(this.lookVel.lon, this.lookVel.lat) > LOOK_VEL_MIN) {
				this.lon += this.lookVel.lon * dt;
				this.lat += this.lookVel.lat * dt;
				const decay = Math.exp(-(dt * 1000) / LOOK_GLIDE_TAU);
				this.lookVel.lon *= decay;
				this.lookVel.lat *= decay;
			} else {
				this.lookVel.lon = 0;
				this.lookVel.lat = 0;
			}
		}
		const wanted = this.lat;
		this.lat = applyLook(this.camera, this.lon, this.lat);
		if (this.lat !== wanted) this.lookVel.lat = 0;
	}

	private emit() {
		if (this.mode === "transition") return;
		const cur =
			this.currentIndex >= 0 ? this.panos[this.currentIndex] : null;
		const node = this.navNode(this.currentIndex);
		let hover: OrbitState["hover"] = null;
		if (this.mode === "interior" && this.hoveredNavIndex >= 0) {
			const p = this.panos[this.hoveredNavIndex];
			const e = this.edgeBetween(this.currentIndex, this.hoveredNavIndex);
			hover = {
				id: p.id,
				name: p.name,
				occluded: e ? e.type !== "walk" : false,
			};
		}
		const exits =
			this.mode === "interior" && node
				? node.rendered.map((e) => ({
						index: e.to,
						type: e.type,
						name: this.panos[e.to]?.name ?? null,
						dist: e.dist,
						bearingDeg: (e.bearing * 180) / Math.PI,
					}))
				: [];
		const state: OrbitState = {
			mode: this.mode,
			panoCount: this.panos.length,
			currentId: cur ? cur.id : null,
			currentName: cur ? (cur.name ?? null) : null,
			currentIndex: this.currentIndex,
			hover,
			objectHover: this.addressing.hoverLabel,
			proxyView: this.proxyView,
			canProxyView: this.canToggleProxyView(),
			mouseLook: this.locked,
			zoom: this.zoomRoom,
			dockDelayMs: this.dockDelayMs,
			freeflySpeed: this.freeflySpeed,
			contextMenu: this.addressing.menu,
			busy: this.interiorBusy,
			overlay: this.overlay,
			exits,
			preview: this.mode === "interior" ? this.buildPreview() : null,
			reachPreview:
				this.mode === "interior" ? this.buildReachPreview() : null,
			arrival: this.mode === "interior" ? this.arrival : null,
			sonarActive: this.markers.sonarActive,
			inspect: this.mode === "interior" ? this.inspect : null,
			tour: this.director.progress,
			canGoBack: this.mode === "interior" && this.history.length > 0,
			trapped: !!node?.trapped,
			currentZone: cur?.zone ?? null,
			visited: [...this.visited],
			nodes: this.nodeDir,
			chapters: this.chapters,
			levelWord: this.levelWord,
			minimap: buildMinimapState({
				minimaps: this.minimaps,
				panos: this.panos,
				panoLevel: this.panoLevel,
				currentIndex: this.currentIndex,
				mode: this.mode,
				labels: this.mapLabels,
				step: this.sceneScale.step,
			}),
		};
		this.onState(state);
	}

	private buildPreview(): OrbitState["preview"] {
		if (this.hoveredNavIndex < 0) return null;
		const p = this.panos[this.hoveredNavIndex];
		if (!p) return null;
		const e = this.edgeBetween(this.currentIndex, this.hoveredNavIndex);
		const headingU = (((this.lon / (2 * Math.PI) + 0.5) % 1) + 1) % 1;
		this.camera.updateMatrixWorld();
		const s = v3(p.position).project(this.camera);
		const rect = this.canvas.getBoundingClientRect();
		return {
			index: this.hoveredNavIndex,
			id: p.id,
			name: p.name ?? null,
			type: e?.type ?? "walk",
			dist: e?.dist ?? 0,
			screenX: rect.left + (s.x * 0.5 + 0.5) * rect.width,
			screenY: rect.top + (-s.y * 0.5 + 0.5) * rect.height,
			thumbUrl: p.placeholderUrl,
			headingU,
		};
	}

	private buildReachPreview(): OrbitState["reachPreview"] {
		const arrow = this.arrowReach;
		const target = arrow ?? this.cursorReach;
		if (!target) return null;
		const p = this.panos[target.index];
		if (!p) return null;
		return {
			index: target.index,
			name: p.name ?? null,
			url: p.url,
			placeholderUrl: p.placeholderUrl,
			dist: this.camera.position.distanceTo(v3(p.position)),
			level: target.level,
			levelDelta: target.levelDelta,
		};
	}

	private showOverlay(msg: string, { spinner = true, err = false } = {}) {
		this.overlay = { msg, spinner, err };
		this.emit();
	}
	private hideOverlay() {
		this.overlay = null;
		this.emit();
	}

	private setFx(type: EdgeType, t: number) {
		const m = Math.sin(Math.PI * MathUtils.clamp(t, 0, 1));
		if (this.reducedMotion) {
			this.canvas.style.filter = "none";
			this.travelFade.style.background = "rgb(var(--ground-rgb))";
			this.travelFade.style.opacity = (m * 0.55).toFixed(3);
			this.iris.style.opacity = "0";
			return;
		}
		const blurPx =
			type === "phase"
				? m * 9
				: type === "vertical"
					? m * 5
					: type === "far"
						? m * 8
						: m * 7;
		this.canvas.style.filter =
			blurPx > 0.002 ? `blur(${blurPx.toFixed(2)}px)` : "none";
		// Phase keeps a cool wash; everything else is the page ground.
		const tint =
			type === "phase"
				? "color-mix(in srgb, rgb(var(--accent-deep-rgb)) 55%, rgb(var(--ground-rgb)))"
				: "rgb(var(--ground-rgb))";
		this.travelFade.style.background = tint;
		const fadeAmp =
			type === "phase"
				? 0.6
				: type === "vertical"
					? 0.3
					: type === "far"
						? 0.6
						: 0.5;
		this.travelFade.style.opacity = (m * fadeAmp).toFixed(3);
		if (type === "vertical") {
			const gap =
				Math.abs(Math.cos(Math.PI * MathUtils.clamp(t, 0, 1))) * 130;
			this.iris.style.background = `radial-gradient(circle at 50% 50%, transparent ${gap.toFixed(1)}%, rgb(var(--ground-rgb)) ${(gap + 7).toFixed(1)}%)`;
			this.iris.style.opacity = "1";
		} else {
			this.iris.style.opacity = "0";
		}
	}
	private clearFx() {
		this.canvas.style.filter = "none";
		this.travelFade.style.opacity = "0";
		this.iris.style.opacity = "0";
	}

	private resizePending = false;

	private resize() {
		const w = this.host.clientWidth;
		const h = this.host.clientHeight;
		if (w === 0 || h === 0) return;
		this.renderer.setSize(w, h, false);
		this.composer.setSize(w, h);
		this.splat.resize();
		this.camera.aspect = w / h;
		this.camera.updateProjectionMatrix();

		if (
			this.mode === "overview" &&
			!this.move &&
			this.controls.autoRotate
		) {
			this.frameOverview();
			this.camera.position.copy(this.browsePos);
			this.controls.target.copy(this.browseTarget);
			this.camera.updateProjectionMatrix();
			// Zero elapsed: this is flushing a camera change, not a frame of time, and
			// a bare update() here would nudge the spin on every resize event.
			this.controls.update(0);
		}
	}

	private onContextMenu = (ev: MouseEvent) => {
		ev.preventDefault();
		if (this.mode !== "overview") return;
		if (
			Math.hypot(ev.clientX - this.rcDownX, ev.clientY - this.rcDownY) > 6
		)
			return;
		this.addressing.openMenu(
			ev.clientX,
			ev.clientY,
			this.activeObjectRoot(),
		);
		this.emit();
	};

	private onControlsStart = () => {
		if (this.mode !== "overview") return;
		this.dollyGoal = null;
		this.controls.autoRotate = false;
		if (this.autoRotateTimer) clearTimeout(this.autoRotateTimer);
	};
	private onControlsEnd = () => {
		if (this.mode !== "overview") return;
		this.emit();
		if (this.autoRotateTimer) clearTimeout(this.autoRotateTimer);
		this.autoRotateTimer = setTimeout(() => {
			if (this.mode === "overview") this.controls.autoRotate = true;
		}, 2500);
	};

	private onPointerDown = (ev: PointerEvent) => {
		this.pointerDown = true;
		if (ev.button === 2) {
			this.rcDownX = ev.clientX;
			this.rcDownY = ev.clientY;
		}
		this.downX = ev.clientX;
		this.downY = ev.clientY;
		if (!this.isLookMode) return;
		if (!this.locked) {
			this.requestLock();
			this.lockClickPending = true;
		}
		this.yieldTour();
		if (this.interiorBusy) return;
		this.noteInput();
		this.stopLookInertia();
		this.dragging = true;
		this.dragMoved = 0;
		this.grabDir.copy(
			cursorRayDir(
				this.camera,
				this.canvas,
				this.cursorRay,
				ev.clientX,
				ev.clientY,
			),
		);
		this.canvas.style.cursor = "grabbing";
		this.canvas.setPointerCapture(ev.pointerId);
	};

	private onPointerMove = (ev: PointerEvent) => {
		this.pointerClientX = ev.clientX;
		this.pointerClientY = ev.clientY;
		this.pointerInside = true;
		if (this.mode === "overview") {
			if (ev.buttons !== 0) return;
			const obj = this.highlightEnabled
				? this.addressing.pickAt(
						ev.clientX,
						ev.clientY,
						this.activeObjectRoot(),
					)
				: null;
			if (this.addressing.setHover(obj)) this.emit();
			return;
		}
		if (!this.isLookMode) return;
		if (this.dragging && !this.locked) {
			this.noteInput();
			const look = pinLook(
				this.camera,
				this.canvas,
				ev.clientX,
				ev.clientY,
				this.grabDir,
			);
			const at = performance.now();
			const step = (at - this.lookSampledAt) / 1000;
			if (this.lookSampledAt > 0 && step > 0.001) {
				const k = 1 - Math.exp(-(step * 1000) / LOOK_SAMPLE_TAU);
				this.lookVel.lon +=
					(angleDelta(this.lon, look.lon) / step - this.lookVel.lon) *
					k;
				this.lookVel.lat +=
					((look.lat - this.lat) / step - this.lookVel.lat) * k;
			}
			this.lookSampledAt = at;
			this.lon = look.lon;
			this.lat = look.lat;
			this.dragMoved = Math.max(
				this.dragMoved,
				Math.hypot(ev.clientX - this.downX, ev.clientY - this.downY),
			);
			if (this.hoveredNavIndex !== -1) {
				this.hoveredNavIndex = -1;
				this.markers.setNavHover(null);
				this.emit();
			}
		} else if (
			this.mode === "interior" &&
			!this.interiorBusy &&
			!CENTER_CURSOR
		) {
			this.updateHover(ev.clientX, ev.clientY);
		}
	};

	private onPointerUp = () => {
		this.pointerDown = false;
		if (!this.isLookMode) return;
		this.dragging = false;
		this.canvas.style.cursor = "";
		if (this.lockClickPending) {
			this.lockClickPending = false;
			return;
		}
		if (this.dragMoved >= 5) return;
		this.noteInput();
		const at = this.aim();
		if (this.mode === "freefly") {
			this.clickFromFreefly(at.x, at.y);
			return;
		}
		const arrow = this.markers.pickFloorArrow(
			at.x,
			at.y,
			this.camera,
			this.canvas,
		);
		if (arrow) {
			this.traverse(arrow.userData.to as number);
			return;
		}
		const spot = this.markers.pickNav(at.x, at.y, this.camera, this.canvas);
		if (spot) {
			this.traverse(spot.userData.to as number);
			return;
		}
		this.clickAnywhere(at.x, at.y);
	};

	private setFov(deg: number) {
		this.camera.fov = deg;
		this.camera.updateProjectionMatrix();
	}

	private wheelNotches(ev: WheelEvent): number {
		const perNotch =
			ev.deltaMode === 1
				? 3
				: ev.deltaMode === 2
					? 1
					: Math.abs(ev.deltaY) >= 40
						? 100
						: 400;
		return MathUtils.clamp(ev.deltaY / perNotch, -3, 3);
	}

	private onWheel = (ev: WheelEvent) => {
		if (!this.isLookMode) return;
		ev.preventDefault();
		this.fovGoal = null;
		const notches = this.wheelNotches(ev);
		if (this.mode === "freefly") {
			this.freeflySpeed = MathUtils.clamp(
				this.freeflySpeed * Math.exp(-notches * 0.18),
				FREEFLY_SPEED_MIN,
				FREEFLY_SPEED_MAX,
			);
			this.emit();
			return;
		}
		const half = Math.tan((this.camera.fov * Math.PI) / 360);
		const next = half * Math.exp(notches * ZOOM_PER_NOTCH);
		this.setFov(
			MathUtils.clamp(
				(Math.atan(next) * 360) / Math.PI,
				ZOOM_MIN_FOV,
				ZOOM_MAX_FOV,
			),
		);
	};

	private get dollying(): boolean {
		return this.mode === "overview" || this.mode === "peek";
	}

	private get zoomRoom(): { in: boolean; out: boolean } {
		if (this.isLookMode) {
			const fov = this.fovGoal ?? this.camera.fov;
			return {
				in: fov > ZOOM_MIN_FOV + 0.01,
				out: fov < ZOOM_MAX_FOV - 0.01,
			};
		}
		if (!this.dollying) return { in: false, out: false };
		const dist =
			this.dollyGoal ??
			this.camera.position.distanceTo(this.controls.target);
		return {
			in: dist > this.controls.minDistance * ZOOM_SLACK,
			out: dist * ZOOM_SLACK < this.controls.maxDistance,
		};
	}

	zoom(step: number) {
		if (this.isLookMode) {
			const from = this.fovGoal ?? this.camera.fov;
			const half = Math.tan((from * Math.PI) / 360);
			const next = half * Math.exp(-step * ZOOM_PER_NOTCH * 2.5);
			this.fovGoal = MathUtils.clamp(
				(Math.atan(next) * 360) / Math.PI,
				ZOOM_MIN_FOV,
				ZOOM_MAX_FOV,
			);
			this.emit();
			return;
		}
		if (!this.dollying) return;
		const from =
			this.dollyGoal ??
			this.camera.position.distanceTo(this.controls.target);
		if (!from) return;
		this.dollyGoal = MathUtils.clamp(
			from * Math.exp(-step * DOLLY_PER_STEP),
			this.controls.minDistance,
			this.controls.maxDistance,
		);
		this.emit();
	}

	private easeZoom(dt: number) {
		if (!dt) return;
		const k = 1 - Math.exp(-ZOOM_EASE * dt);

		if (this.fovGoal !== null) {
			if (!this.isLookMode) this.fovGoal = null;
			else {
				const goal = this.fovGoal;
				const next = this.camera.fov + (goal - this.camera.fov) * k;
				const done = Math.abs(goal - next) < goal * ZOOM_SETTLED;
				this.setFov(done ? goal : next);
				if (done) this.fovGoal = null;
			}
		}

		if (this.dollyGoal === null) return;
		if (!this.dollying) {
			this.dollyGoal = null;
			return;
		}
		_dolly.subVectors(this.camera.position, this.controls.target);
		const dist = _dolly.length();
		if (!dist) {
			this.dollyGoal = null;
			return;
		}
		const goal = this.dollyGoal;
		const next = dist + (goal - dist) * k;
		const done = Math.abs(goal - next) < goal * ZOOM_SETTLED;
		this.camera.position
			.copy(this.controls.target)
			.addScaledVector(_dolly.divideScalar(dist), done ? goal : next);
		if (done) {
			this.dollyGoal = null;
			this.emit();
		}
	}

	private onClick = (ev: MouseEvent) => {
		if (this.mode !== "overview") return;
		if (Math.hypot(ev.clientX - this.downX, ev.clientY - this.downY) > 6)
			return;
		const target =
			this.overviewTarget >= 0
				? this.overviewTarget
				: this.panoAtPointer(ev.clientX, ev.clientY);
		if (target >= 0) this.enter(target);
	};

	private panoAtPointer(clientX: number, clientY: number): number {
		const hit = this.raycastOverview(clientX, clientY);
		return hit ? this.nearestPanoTo(hit.point) : -1;
	}

	private onPointerLeave = () => {
		this.pointerInside = false;
		if (!CENTER_CURSOR) this.cursor.hide();
	};
	private onWindowPointerUp = () => {
		this.pointerDown = false;
		this.peekUp();
	};

	private onKeyDown = (ev: KeyboardEvent) => {
		const focused = document.activeElement;
		if (
			focused instanceof HTMLInputElement ||
			focused instanceof HTMLTextAreaElement ||
			(focused instanceof HTMLElement && focused.isContentEditable)
		)
			return;
		const spaceFlies =
			this.mode === "freefly" ||
			(this.mode === "interior" && this.canEnterFreefly());
		if (ev.code === "Space" && !ev.repeat && !spaceFlies) {
			ev.preventDefault();
			this.peekDown();
			return;
		}
		if (this.mode === "freefly") {
			if (ev.code === "BracketLeft" || ev.code === "BracketRight") {
				ev.preventDefault();
				this.dockDelayMs = MathUtils.clamp(
					this.dockDelayMs + (ev.code === "BracketRight" ? 50 : -50),
					0,
					3000,
				);
				this.emit();
				return;
			}
			if (ev.code === "Escape") {
				ev.preventDefault();
				this.returnToInterior(this.nearestPanoTo(this.camera.position));
				return;
			}
			this.requestLock();
			this.trackFreeflyKey(ev, true);
			return;
		}
		if (this.mode !== "interior") return;
		if (ev.code === "Tab" && !ev.repeat) {
			ev.preventDefault();
			this.toggleSonar();
			return;
		}
		if (ev.code === "Escape") {
			this.leaveInterior();
			return;
		}
		if (this.interiorBusy || ev.repeat) return;
		if (ev.code === "KeyL") {
			ev.preventDefault();
			this.logAim();
			return;
		}
		if (ev.code === "Backspace") {
			ev.preventDefault();
			this.goBack();
			return;
		}
		if (ev.code.startsWith("Digit")) {
			const n = Number(ev.code.slice(5));
			if (n >= 1) {
				ev.preventDefault();
				this.jumpToLevel(n - 1);
			}
			return;
		}
		const flyKey = freeflyKey(ev.code);
		if (
			flyKey !== null &&
			FREEFLY_ENTER_KEYS.has(flyKey) &&
			this.canEnterFreefly()
		) {
			ev.preventDefault();
			this.requestLock();
			this.freeflyKeys.add(flyKey);
			this.enterFreefly();
			return;
		}
		const fx = Math.cos(this.lon);
		const fz = Math.sin(this.lon);
		const rx = -Math.sin(this.lon);
		const rz = Math.cos(this.lon);
		switch (ev.code) {
			case "KeyW":
				ev.preventDefault();
				this.stepToward(fx, fz);
				break;
			case "KeyS":
				ev.preventDefault();
				this.stepToward(-fx, -fz);
				break;
			case "KeyD":
				ev.preventDefault();
				this.stepToward(rx, rz);
				break;
			case "KeyA":
				ev.preventDefault();
				this.stepToward(-rx, -rz);
				break;
		}
	};
	private onKeyUp = (ev: KeyboardEvent) => {
		if (ev.code === "Space") this.peekUp();
		this.trackFreeflyKey(ev, false);
	};

	private trackFreeflyKey(ev: KeyboardEvent, down: boolean) {
		const code = freeflyKey(ev.code);
		if (code === null) return;
		if (!down) {
			this.freeflyKeys.delete(code);
			return;
		}
		ev.preventDefault();
		this.freeflyKeys.add(code);
		this.noteInput();
	}

	private onWindowBlur = () => {
		this.freeflyKeys.clear();
		this.freeflyVel.set(0, 0, 0);
		this.stopLookInertia();
	};

	private updateHover(aimX: number, aimY: number) {
		if (this.currentIndex < 0) return;
		const arrow = this.markers.pickFloorArrow(
			aimX,
			aimY,
			this.camera,
			this.canvas,
		);
		this.markers.setArrowHover(arrow);
		if (arrow) {
			this.markers.setNavHover(null);
			this.canvas.style.cursor = "pointer";
			let changedArrow = this.hoveredNavIndex !== -1;
			const to = arrow.userData.to as number;
			if (!this.arrowReach || this.arrowReach.index !== to) {
				const level = this.panoLevel[to] ?? -1;
				const cur = this.panoLevel[this.currentIndex] ?? level;
				this.arrowReach = { index: to, level, levelDelta: level - cur };
				this.requestPano(to);
				changedArrow = true;
			}
			this.hoveredNavIndex = -1;
			if (this.addressing.setHover(null) || changedArrow) this.emit();
			return;
		}
		if (this.arrowReach) {
			this.arrowReach = null;
			this.emit();
		}
		const spot = this.markers.pickNav(aimX, aimY, this.camera, this.canvas);
		this.markers.setNavHover(spot);
		const idx = spot ? (spot.userData.to as number) : -1;
		const obj = this.highlightEnabled
			? this.addressing.pickAt(aimX, aimY, this.activeObjectRoot())
			: null;
		if (idx >= 0) {
			const rendered = this.navNode(this.currentIndex)?.rendered;
			const isVertical = rendered?.some(
				(e) => e.to === idx && e.type === "vertical",
			);
			const clear =
				isVertical || this.isTargetClear(v3(this.panos[idx].position));
			this.canvas.style.cursor = clear ? "pointer" : "crosshair";
		} else if (obj) {
			this.canvas.style.cursor = "pointer";
		} else {
			this.canvas.style.cursor = this.raycastInterior(aimX, aimY)
				? ""
				: "zoom-out";
		}
		const changed = idx !== this.hoveredNavIndex;
		this.hoveredNavIndex = idx;
		if (this.addressing.setHover(obj) || changed) this.emit();
	}

	private interiorTargets(): Object3D[] {
		const targets: Object3D[] = [];
		if (this.projectionMode) {
			if (this.proxyGroup) targets.push(this.proxyGroup);
			if (this.projection.proxyBase)
				targets.push(this.projection.proxyBase);
		} else {
			this.sphereA.updateMatrixWorld();
			targets.push(this.sphereA);
		}
		return targets;
	}

	private raycastInteriorAll(
		clientX: number,
		clientY: number,
	): Intersection[] {
		const targets = this.interiorTargets();
		if (targets.length === 0) return [];
		const rect = this.canvas.getBoundingClientRect();
		_cursorNdc.set(
			((clientX - rect.left) / rect.width) * 2 - 1,
			-((clientY - rect.top) / rect.height) * 2 + 1,
		);
		this.camera.updateMatrixWorld();
		this.cursorRay.setFromCamera(_cursorNdc, this.camera);
		return this.cursorRay
			.intersectObjects(targets, true)
			.filter((h) => this.hitIsPickable(h));
	}

	private hitIsPickable(h: Intersection): boolean {
		const splat = this.splat.isActive;
		for (let o: Object3D | null = h.object; o; o = o.parent) {
			if (
				splat &&
				(o === this.proxyGroup || o === this.projection.proxyBase)
			)
				return true;
			if (!o.visible) return false;
		}
		return true;
	}

	private raycastInterior(
		clientX: number,
		clientY: number,
	): Intersection | null {
		return this.raycastInteriorAll(clientX, clientY)[0] ?? null;
	}

	private raycastOverview(
		clientX: number,
		clientY: number,
	): Intersection | null {
		const root = this.activeObjectRoot();
		if (!root) return null;
		const standingIn = this.splat.isActive;
		if (!root.visible && !standingIn) return null;
		const rect = this.canvas.getBoundingClientRect();
		_cursorNdc.set(
			((clientX - rect.left) / rect.width) * 2 - 1,
			-((clientY - rect.top) / rect.height) * 2 + 1,
		);
		this.camera.updateMatrixWorld();
		this.cursorRay.setFromCamera(_cursorNdc, this.camera);
		for (const h of this.cursorRay.intersectObject(root, true)) {
			let visible = true;
			for (
				let o: Object3D | null = h.object;
				o && o !== root;
				o = o.parent
			)
				if (!o.visible) {
					visible = false;
					break;
				}
			if (visible) return h;
		}
		return null;
	}

	private get hasFloorVolumes(): boolean {
		return this.minimaps.some((m) => !!m.volume);
	}

	private floorAt(p: Vector3): number {
		let best = -1;
		let bestVolume = Infinity;
		for (const mm of this.minimaps) {
			const v = mm.volume;
			if (!v) continue;
			const [ox, oy, oz] = v.origin;
			const [dx, dy, dz] = v.dimensions;
			if (p.x < ox || p.x > ox + dx) continue;
			if (p.y < oy || p.y > oy + dy) continue;
			if (p.z < oz || p.z > oz + dz) continue;
			const volume = dx * dy * dz;
			if (volume < bestVolume) {
				bestVolume = volume;
				best = mm.level;
			}
		}
		return best;
	}

	private autoHomeTarget(
		hit: Intersection,
		floor = -1,
		exclude: number = this.currentIndex,
	): number {
		const cam = this.camera.position;
		const clickBearing = Math.atan2(
			hit.point.z - cam.z,
			hit.point.x - cam.x,
		);
		const directional =
			Math.hypot(hit.point.x - cam.x, hit.point.z - cam.z) > 0.5;
		let best = -1;
		let bestCost = Infinity;
		for (let i = 0; i < this.panos.length; i++) {
			if (i === exclude) continue;
			if (floor >= 0 && this.panoLevel[i] !== floor) continue;
			const pp = v3(this.panos[i].position);
			const d = pp.distanceTo(hit.point);
			const bearing = Math.atan2(pp.z - cam.z, pp.x - cam.x);
			const ang = Math.abs(angleDelta(clickBearing, bearing));
			const cost = d + (directional ? ang * 3 : 0);
			if (cost < bestCost) {
				bestCost = cost;
				best = i;
			}
		}
		return best < 0 && floor >= 0
			? this.autoHomeTarget(hit, -1, exclude)
			: best;
	}

	private aimQuery(
		hits: Intersection[],
	): { point: Vector3; through: Vector3 | null } | null {
		if (hits.length === 0) return null;
		const near = hits[0];
		const m = this.metrics;
		const lookingUp = near.point.y > this.camera.position.y;
		const probeY = near.point.y + (lookingUp ? -m.probeEps : m.probeEps);
		let step = 0;
		_wpDir.set(0, 0, 0);
		if (near.face) {
			_wpDir
				.copy(near.face.normal)
				.transformDirection(near.object.matrixWorld);
			const c = this.camera.position;
			if (
				_wpDir.x * (near.point.x - c.x) +
					_wpDir.y * (near.point.y - c.y) +
					_wpDir.z * (near.point.z - c.z) <
				0
			)
				_wpDir.negate();
			_wpDir.y = 0;
			const lateral = Math.min(1, _wpDir.length());
			step = lateral * m.wpStandoff;
			if (lateral > 1e-4) _wpDir.divideScalar(_wpDir.length());
			else _wpDir.set(0, 0, 0);
		}
		_wpOut.set(
			near.point.x,
			this.settleAt(near.point.x, near.point.z, probeY),
			near.point.z,
		);
		return { point: _wpOut, through: step > 1e-4 ? _wpDir : null };
	}

	private settleAt(
		x: number,
		z: number,
		probeY: number,
		known?: ReturnType<OrbitEngine["columnAt"]>,
	): number {
		const spans = known ?? this.columnAt(x, z);
		const inside = spans.find(
			(sp) => probeY > sp.bottom && probeY < sp.top,
		);
		if (inside) return inside.top;
		for (const sp of spans)
			if (sp.standable && sp.top <= probeY) return sp.top;
		let above: number | null = null;
		for (const sp of spans) if (sp.standable) above = sp.top;
		return above ?? probeY;
	}

	private resolveAim(hits: Intersection[]): {
		marker: Vector3;
		occluded: boolean;
		index: number;
	} | null {
		if (hits.length === 0) return null;
		const query = this.aimQuery(hits);
		if (!query) return null;
		const eye = this.standingEye(query.point);
		let index = this.nearestPanoTo(eye, this.currentIndex);
		if (index >= 0 && query.through) {
			const far = this.nearestPanoBeyond(
				eye,
				query.point,
				query.through,
				this.currentIndex,
			);
			if (
				far >= 0 &&
				eye.distanceTo(v3(this.panos[far].position)) <=
					eye.distanceTo(v3(this.panos[index].position)) +
						this.metrics.wpThrough
			)
				index = far;
		}
		if (index < 0) return null;
		let clear = this.isTargetClear(v3(this.panos[index].position));

		const curLevel =
			this.currentIndex >= 0
				? (this.panoLevel[this.currentIndex] ?? -1)
				: -1;
		if (
			!clear &&
			curLevel >= 0 &&
			(this.panoLevel[index] ?? -1) !== curLevel
		) {
			const pinned = this.nearestPanoTo(eye, this.currentIndex, curLevel);
			if (pinned >= 0) {
				index = pinned;
				clear = this.isTargetClear(v3(this.panos[index].position));
			}
		}
		return {
			marker: this.walkBackFrom(index, hits[0]),
			occluded: !clear,
			index,
		};
	}

	private planNormal(h: Intersection, dir: Vector3, out: Vector3): boolean {
		if (!h.face) return false;
		out.copy(h.face.normal).transformDirection(h.object.matrixWorld);
		if (out.dot(dir) > 0) out.negate();
		out.y = 0;
		if (out.lengthSq() < 1e-8) return false;
		out.normalize();
		return true;
	}

	private walkBackFrom(index: number, cursorHit: Intersection): Vector3 {
		const m = this.metrics;
		const from = v3(this.panos[index].position);
		const ground = from.y - m.eyeHeight;
		_walkDir.copy(cursorHit.point).sub(from);
		_walkDir.y = 0;
		const dist = _walkDir.length();
		if (dist < 1e-6) return _walkOut.copy(from).setY(ground);
		_walkDir.divideScalar(dist);

		const targets = this.interiorTargets();
		let blocked: Intersection | undefined;
		for (let i = 0; targets.length > 0 && i < WALK_HEIGHTS.length; i++) {
			_walkFrom.set(
				from.x,
				ground + m.eyeHeight * WALK_HEIGHTS[i],
				from.z,
			);
			this.walkRay.set(_walkFrom, _walkDir);
			this.walkRay.near = 0;
			this.walkRay.far = dist;
			const h = this.walkRay
				.intersectObjects(targets, true)
				.find((x) => this.hitIsPickable(x));
			if (h && (!blocked || h.distance < blocked.distance)) blocked = h;
		}
		const pushable =
			!!blocked && this.planNormal(blocked, _walkDir, _walkNrm);
		_walkPt.copy(from).addScaledVector(_walkDir, dist);
		let planNormal: [number, number, number] | null = null;
		if (pushable) {
			const q = _walkNrm.dot((blocked as Intersection).point);
			const destDepth = _walkNrm.dot(from) - q;
			if (destDepth < m.wpClearance) {
				_walkPt.copy(from);
			} else {
				const depth = _walkNrm.dot(_walkPt) - q;
				_walkPt.addScaledVector(
					_walkNrm,
					Math.max(0, m.wpClearance - depth),
				);
				_walkAlt.set(_walkPt.x - from.x, 0, _walkPt.z - from.z);
				const span = _walkAlt.length();
				if (span > 1e-6) {
					_walkAlt.divideScalar(span);
					let stop = span;
					for (
						let i = 0;
						targets.length > 0 && i < WALK_HEIGHTS.length;
						i++
					) {
						_walkFrom.set(
							from.x,
							ground + m.eyeHeight * WALK_HEIGHTS[i],
							from.z,
						);
						this.walkRay.set(_walkFrom, _walkAlt);
						this.walkRay.near = 0;
						this.walkRay.far = span;
						const h = this.walkRay
							.intersectObjects(targets, true)
							.find((x) => this.hitIsPickable(x));
						if (h && h.distance < stop) stop = h.distance;
					}
					if (stop < span) {
						const t = Math.max(0, stop - m.wpClearance);
						_walkPt.set(
							from.x + _walkAlt.x * t,
							_walkPt.y,
							from.z + _walkAlt.z * t,
						);
					}
				}
			}
			planNormal = _walkNrm.toArray() as [number, number, number];
		}
		this.aimBlock = blocked
			? {
					source: "walk",
					object: describeObject(blocked.object),
					face: blocked.faceIndex ?? -1,
					dist: blocked.distance,
					point: blocked.point.toArray() as [number, number, number],
					planNormal,
				}
			: null;
		return _walkOut.set(_walkPt.x, ground, _walkPt.z);
	}

	private recordAim(
		cursor: Intersection,
		aim: { marker: Vector3; occluded: boolean; index: number },
	): Record<string, unknown> {
		const pano = this.panos[aim.index];
		const b = this.aimBlock;
		const cursorAt = cursor.point.toArray() as [number, number, number];
		const markerAt = aim.marker.toArray() as [number, number, number];
		const chest = markerAt[1] + this.metrics.eyeHeight * 0.5;
		const buried = this.columnAt(markerAt[0], markerAt[2]).some(
			(sp) => chest > sp.bottom && chest < sp.top,
		);
		const toDest = pano
			? Math.hypot(
					markerAt[0] - pano.position[0],
					markerAt[2] - pano.position[2],
				)
			: -1;
		return {
			summary:
				`cursor ${fmt3(cursorAt)} on ${describeObject(cursor.object)}` +
				` face ${cursor.faceIndex ?? -1}\n` +
				`      marker ${fmt3(markerAt)} — ${aim.occluded ? "SHOWN (destination out of view)" : "hidden (destination in view)"}\n` +
				`      dest   #${aim.index} ${pano?.id ?? "?"} ${fmt3(pano?.position ?? [0, 0, 0])}` +
				` level ${this.panoLevel[aim.index] ?? -1}\n` +
				`      walk   ${b ? `clears ${b.object} face ${b.face} (${b.source}) at ${b.dist.toFixed(2)}m, plan normal ${b.planNormal ? fmt3(b.planNormal) : "none — level face, no push"}` : "nothing to clear"}\n` +
				`      marker ${buried ? "IS BURIED in geometry" : "stands in open air"}` +
				` · ${toDest.toFixed(2)} m from the capture in plan\n` +
				`      camera ${fmt3(this.camera.position.toArray())}`,
			cursor: cursorAt,
			cursorObject: describeObject(cursor.object),
			cursorFace: cursor.faceIndex ?? -1,
			markerBuried: buried,
			markerToDestPlan: toDest,
			marker: markerAt,
			occluded: aim.occluded,
			destIndex: aim.index,
			destId: pano?.id ?? null,
			destName: pano?.name ?? null,
			destPosition: pano?.position ?? null,
			destLevel: this.panoLevel[aim.index] ?? -1,
			currentIndex: this.currentIndex,
			currentLevel: this.panoLevel[this.currentIndex] ?? -1,
			blocker: b,
			camera: this.camera.position.toArray(),
			metrics: {
				eye: this.metrics.eyeHeight,
				wpClearance: this.metrics.wpClearance,
				wpStandoff: this.metrics.wpStandoff,
			},
		};
	}

	logAim() {
		const at = this.aim();
		const hits = this.raycastInteriorAll(at.x, at.y);
		const aim = hits.length > 0 ? this.resolveAim(hits) : null;
		if (!aim) {
			console.log(
				`[aim] nothing resolved — ${hits.length === 0 ? "no scene geometry under the aim point" : "no capture to travel to"}`,
			);
			return;
		}
		const record = this.recordAim(hits[0], aim);
		console.log(`[aim] ${record.summary}`, record);
	}

	private standingEye(waypoint: Vector3): Vector3 {
		return _wpEye.copy(waypoint).setY(waypoint.y + this.metrics.eyeHeight);
	}

	private columnAt(
		x: number,
		z: number,
	): Array<{ top: number; bottom: number; standable: boolean }> {
		const targets = this.interiorTargets();
		if (targets.length === 0) return [];
		const from = this.sceneTopY + 1;
		this.dropRay.set(_dropFrom.set(x, from, z), _DOWN);
		this.dropRay.near = 0;
		this.dropRay.far = Math.max(1, from - this.sceneBottomY) + 1;
		const ys = this.dropRay
			.intersectObjects(targets, true)
			.map((h) => h.point.y);
		const out: Array<{ top: number; bottom: number; standable: boolean }> =
			[];
		for (let i = 0; i < ys.length; i += 2) {
			const top = ys[i];
			const bottom = i + 1 < ys.length ? ys[i + 1] : top;
			const air =
				out.length === 0 ? Infinity : out[out.length - 1].bottom - top;
			out.push({
				top,
				bottom,
				standable: air >= this.metrics.standHeadroom,
			});
		}
		return out;
	}

	private destinationFloor(targetIdx: number): Vector3 {
		const p = this.panos[targetIdx].position;
		return _ghostFloor.set(p[0], p[1] - this.metrics.floorDrop, p[2]);
	}

	private clickAnywhere(clientX: number, clientY: number) {
		const hits = this.raycastInteriorAll(clientX, clientY);
		if (hits.length === 0) {
			this.exit();
			return;
		}
		const aim = this.resolveAim(hits);
		if (aim) this.traverse(aim.index);
	}

	private reskinProxy(mat: Material) {
		if (!this.proxyGroup) return;
		const matte = mat === this.polyMaterial;
		this.proxyGroup.traverse((o) => {
			const m = o as Mesh;
			if (!m.isMesh) return;
			m.material = matte
				? ((m.userData.colorMat as Material) ?? mat)
				: mat;
		});
		this.projection.setBaseMaterial(mat);
	}

	private colorProxyObjects() {
		if (!this.proxyGroup) return;
		collectObjects(this.proxyGroup).forEach((obj, i) => {
			const mat = this.polyMaterial.clone();
			mat.color.setHSL((i * 0.6180339887) % 1, 0.6, 0.55, SRGBColorSpace);
			this.proxyColorMats.push(mat);
			obj.traverse((o) => {
				if ((o as Mesh).isMesh) o.userData.colorMat = mat;
			});
		});
	}

	private proxyAsDollhouse(): boolean {
		return this.sharedOverview || (this.proxyView && !!this.proxyGroup);
	}

	private get splatEnabled(): boolean {
		return this.splat.ready && this.splatView;
	}

	private setSplatShowing(on: boolean) {
		this.splat.setActive(on);
		this.scene.background = on ? null : this.bgColor;
	}

	private setOverviewView() {
		const useSplat = this.splatEnabled;
		const proxyDoll = !useSplat && this.proxyAsDollhouse();
		if (this.liteRoot) this.liteRoot.visible = !useSplat && !proxyDoll;
		if (this.proxyGroup) {
			if (proxyDoll) {
				this.reskinProxy(this.polyMaterial);
				this.proxyGroup.visible = true;
			} else {
				this.proxyGroup.visible = false;
			}
		}
		this.setSplatShowing(useSplat);
		this.sphereA.visible = false;
		this.sphereB.visible = false;
		this.markers.navGroup.visible = false;
		this.markers.arrowGroup.visible = false;
		this.markers.hideSonar();
		this.markers.you.group.visible = false;
		this.projection.syncBase(!!this.proxyGroup?.visible);
	}

	private setFreeflyView() {
		const useSplat = this.splatEnabled;
		if (this.liteRoot) this.liteRoot.visible = false;
		if (this.proxyGroup) {
			if (!useSplat) this.reskinProxy(this.polyMaterial);
			this.proxyGroup.visible = !useSplat;
		}
		this.setSplatShowing(useSplat);
		this.sphereA.visible = false;
		this.markers.navGroup.visible = false;
		this.markers.arrowGroup.visible = false;
		this.markers.hideSonar();
		this.markers.you.group.visible = false;
		this.projection.syncBase(false);
	}

	private setInteriorProxyView() {
		if (!this.proxyGroup || !this.projectionMode) return;
		this.reskinProxy(
			this.proxyView ? this.polyMaterial : this.projection.material,
		);
		this.sphereA.visible = !this.proxyView;
		if (!this.proxyView) this.updateProjection();
	}

	private setInteriorView() {
		this.setSplatShowing(false);
		if (this.liteRoot) this.liteRoot.visible = false;
		if (this.proxyGroup) this.proxyGroup.visible = this.projectionMode;
		if (this.projectionMode) {
			this.setInteriorProxyView();
		} else {
			this.sphereA.visible = true;
			this.sphereA.position.copy(this.camera.position);
		}
		this.markers.navGroup.visible = true;
		this.markers.arrowGroup.visible = true;
		this.markers.you.group.visible = false;
		this.projection.syncBase(!!this.proxyGroup?.visible);
	}

	private setPeekView() {
		this.setSplatShowing(false);
		const proxyDoll = this.proxyAsDollhouse();
		if (this.liteRoot) this.liteRoot.visible = !proxyDoll;
		if (this.proxyGroup) {
			if (proxyDoll) {
				this.reskinProxy(this.polyMaterial);
				this.proxyGroup.visible = true;
			} else {
				this.proxyGroup.visible = false;
			}
		}
		this.sphereA.visible = false;
		this.sphereB.visible = false;
		this.markers.navGroup.visible = false;
		this.markers.arrowGroup.visible = false;
		this.markers.hideSonar();
		this.markers.you.group.visible = true;
		this.projection.syncBase(!!this.proxyGroup?.visible);
	}

	private canToggleProxyView(): boolean {
		if (this.splat.isActive) return false;
		if (this.mode === "overview")
			return !!this.liteRoot && !!this.proxyGroup;
		if (this.mode === "interior") return this.projectionMode;
		return false;
	}

	toggleProxyView() {
		if (!this.canToggleProxyView()) return;
		this.proxyView = !this.proxyView;
		this.addressing.setHover(null);
		this.addressing.closeMenu();
		this.canvas.style.cursor = "";
		if (this.mode === "overview") this.setOverviewView();
		else if (this.mode === "interior") this.setInteriorProxyView();
		this.emit();
	}

	private activeObjectRoot(): Object3D | null {
		if (this.mode === "overview" || this.mode === "peek") {
			if (this.proxyView && this.proxyGroup) return this.proxyGroup;
			return this.liteRoot ?? this.proxyGroup;
		}
		if (this.mode === "interior") return this.proxyGroup;
		return null;
	}

	closeMenu() {
		if (!this.addressing.hasMenu) return;
		this.addressing.closeMenu();
		this.emit();
	}
	toggleMenuTargetHidden() {
		this.addressing.toggleMenuTargetHidden();
		this.emit();
	}
	toggleMenuTargetOutline() {
		this.addressing.toggleMenuTargetOutline();
		this.emit();
	}
	showAllHidden() {
		this.addressing.showAllHidden();
		this.emit();
	}
	clearOutlines() {
		this.addressing.clearOutlines();
		this.emit();
	}

	private updateProjection() {
		this.sphereA.visible = !this.proxyView && !this.move;
		this.projection.project(
			this.panos,
			this.activeCaptures(),
			this.requestPano,
			this.sphereA,
			this.camera.position,
		);
	}

	private activeCaptures(): Array<[number, number]> {
		if (this.move) {
			const to = this.move.index;
			const from = this.currentIndex;
			if (from < 0 || from === to) return [[to, 1]];
			const t = Math.min(
				1,
				Math.max(
					0,
					(performance.now() - this.move.start) / this.move.dur,
				),
			);
			const e = easeInOut(t);
			return [
				[from, 1 - e],
				[to, e],
			];
		}
		if (this.flyTarget >= 0) return [[this.flyTarget, 1]];
		if (this.currentIndex >= 0) return [[this.currentIndex, 1]];
		return [];
	}

	private startFly(
		toPos: Vector3,
		lookTarget: Vector3,
		dur: number,
		cbs: {
			toFov?: number;
			crossfade?: boolean;
			dissolveInterior?: boolean;
			onMid?: () => void;
			onEnd?: () => void;
		} = {},
	) {
		this.dummyCam.up.copy(this.camera.up);
		this.dummyCam.position.copy(toPos);
		this.dummyCam.lookAt(lookTarget);
		this.dummyCam.updateMatrixWorld();
		this.transition = {
			fromPos: this.camera.position.clone(),
			toPos: toPos.clone(),
			fromQuat: this.camera.quaternion.clone(),
			toQuat: this.dummyCam.quaternion.clone(),
			fromFov: this.camera.fov,
			toFov: cbs.toFov ?? this.camera.fov,
			start: performance.now(),
			dur: this.reducedMotion ? REDUCED_DUR : dur,
			crossfade: !!cbs.crossfade,
			dissolveInterior: !!cbs.dissolveInterior,
			onMid: cbs.onMid,
			onEnd: cbs.onEnd,
			midDone: false,
		};
		if (cbs.crossfade) this.travelFade.style.opacity = "0";
		this.mode = "transition";
		this.stopLookInertia();
		this.closeInspect();
		this.arrowReach = null;
		this.markers.setArrowHover(null);
		this.controls.enabled = false;
		this.controls.autoRotate = false;
		this.hoveredNavIndex = -1;
		this.markers.setNavHover(null);
		this.markers.hideSonar();
		this.addressing.setHover(null);
		this.addressing.closeMenu();
		this.canvas.style.cursor = "";
		this.emit();
	}

	private traverse(index: number, reverse = false, pass = false) {
		if (index === this.currentIndex || !this.panos[index]) return;
		if (this.interiorBusy) {
			this.pendingTravel = index;
			return;
		}
		const edge = this.edgeBetween(this.currentIndex, index);
		const type: EdgeType = edge?.type ?? "far";
		const dy =
			edge?.dy ??
			this.panos[index].position[1] -
				this.panos[this.currentIndex].position[1];
		if (!reverse && this.currentIndex >= 0)
			this.history.push(this.currentIndex);
		this.beginMove(index, type, dy, pass);
	}

	private beginMove(index: number, type: EdgeType, dy: number, pass = false) {
		this.interiorBusy = true;
		this.stopLookInertia();
		this.closeInspect();
		this.arrowReach = null;
		this.cursorReach = null;
		this.markers.setArrowHover(null);
		this.hoveredNavIndex = -1;
		this.markers.setNavHover(null);
		this.markers.navGroup.visible = false;
		this.markers.arrowGroup.visible = false;
		this.markers.hideSonar();
		this.emit();
		this.requestPano(index);
		const fromPos = this.camera.position.clone();
		const toPos = v3(this.panos[index].position);
		const ctrl = this.reducedMotion
			? null
			: this.pathControl(fromPos, toPos, type);
		const dur =
			(this.reducedMotion ? REDUCED_DUR : DUR[type]) *
			(pass ? PASS_DUR_SCALE : 1);
		if (this.projectionMode) {
			this.move = {
				fromPos,
				toPos,
				ctrl,
				start: performance.now(),
				dur,
				index,
				type,
				dy,
				sphere: false,
				pass,
			};
			return;
		}
		const token = this.loadToken;
		void this.streamer.ensure(index).then(() => {
			if (this.disposed || token !== this.loadToken) return;
			const target = this.panos[index];
			this.sphereBMat.uniforms.map.value = target.texture ?? DUMMY_TEX;
			this.sphereBMat.uniforms.opacity.value = 0;
			this.sphereB.visible = true;
			this.move = {
				fromPos,
				toPos,
				ctrl,
				start: performance.now(),
				dur,
				index,
				type,
				dy,
				sphere: true,
				pass,
			};
		});
	}

	private pathControl(
		from: Vector3,
		to: Vector3,
		type: EdgeType,
	): Vector3 | null {
		if (type === "vertical") {
			const c = from.clone().add(to).multiplyScalar(0.5);
			c.y =
				Math.max(from.y, to.y) +
				Math.max(
					this.metrics.eyeHeight * 0.3,
					Math.abs(to.y - from.y) * 0.3,
				);
			return c;
		}
		if (type === "far") {
			const c = from.clone().add(to).multiplyScalar(0.5);
			c.y += Math.min(this.sceneMaxDim * 0.6, from.distanceTo(to) * 0.45);
			return c;
		}
		return null;
	}

	private hurryMove(ms = 240) {
		const mv = this.move;
		if (!mv) return;
		const now = performance.now();
		const t = Math.min(1, (now - mv.start) / mv.dur);
		if (t > 0.95) return;
		mv.start = now - (t * ms) / (1 - t);
		mv.dur = ms / (1 - t);
	}

	private finishMove(mv: Move) {
		this.clearFx();
		if (mv.sphere) {
			const target = this.panos[mv.index];
			this.sphereAMat.uniforms.map.value = target.texture ?? DUMMY_TEX;
			this.sphereAMat.uniforms.opacity.value = 1;
			this.sphereB.visible = false;
		}
		this.interiorBusy = false;
		const p = this.panos[mv.index];
		if (!mv.pass) {
			this.arrival = {
				name: p.name ?? p.id,
				verb: edgeVerb(mv.type, mv.dy),
				ts: performance.now(),
			};
		}
		this.activate(mv.index);
		if (this.exitWhenLanded) {
			this.exitWhenLanded = false;
			this.pendingTravel = null;
			this.exit();
			return;
		}
		const next = this.pendingTravel;
		this.pendingTravel = null;
		if (next != null && next !== this.currentIndex) this.traverse(next);
	}

	private activate(index: number) {
		this.currentIndex = index;
		this.flyTarget = -1;
		this.visited.add(index);
		this.requestPano(index);
		if (!this.projectionMode) {
			this.sphereAMat.uniforms.map.value =
				this.panos[index].texture ?? DUMMY_TEX;
			this.sphereAMat.uniforms.opacity.value = 1;
		}
		const node = this.navNode(index);
		this.markers.buildNav(node, this.panos);
		this.markers.navGroup.visible = this.mode === "interior";
		this.refreshFloorArrows();
		if (node?.trapped) this.markers.pulseExits(performance.now(), 2200);
		this.noteInput();
		this.emit();
	}

	private onPanoReady(i: number) {
		if (!this.projectionMode && i === this.currentIndex) {
			this.sphereAMat.uniforms.map.value =
				this.panos[i].texture ?? DUMMY_TEX;
			this.sphereAMat.uniforms.opacity.value = 1;
		}
	}

	private nearestPanoBeyond(
		point: Vector3,
		at: Vector3,
		through: Vector3,
		exclude: number,
	): number {
		let best = -1;
		let bestD = Number.POSITIVE_INFINITY;
		for (let i = 0; i < this.panos.length; i++) {
			if (i === exclude) continue;
			const p = this.panos[i].position;
			if ((p[0] - at.x) * through.x + (p[2] - at.z) * through.z <= 0)
				continue;
			const d = point.distanceToSquared(v3(p));
			if (d < bestD) {
				bestD = d;
				best = i;
			}
		}
		return best;
	}

	private nearestPanoTo(
		point: Vector3,
		exclude = -1,
		onlyLevel = -1,
	): number {
		let best = -1;
		let bestD = Infinity;
		for (let i = 0; i < this.panos.length; i++) {
			if (i === exclude) continue;
			if (onlyLevel >= 0 && this.panoLevel[i] !== onlyLevel) continue;
			const d = point.distanceToSquared(v3(this.panos[i].position));
			if (d < bestD) {
				bestD = d;
				best = i;
			}
		}
		return best < 0 && exclude < 0 ? 0 : best;
	}

	private currentUserWorldPos(): Vector3 {
		return this.projectionMode
			? this.camera.position.clone()
			: v3(this.panos[this.currentIndex].position);
	}

	enter(index: number | null = null) {
		if (this.mode !== "overview" || this.move || this.panos.length === 0)
			return;
		const idx = index ?? this.nearestPanoTo(this.controls.target);
		this.setInside(true);
		this.history = [];
		this.requestLock();
		this.flyIntoInterior(idx, 1100);
	}

	private flyIntoInterior(
		idx: number,
		dur: number,
		{ dissolve = false }: { dissolve?: boolean } = {},
	) {
		this.requestPano(idx);
		const toPos = v3(this.panos[idx].position);
		const dir = this.camera.getWorldDirection(_flyDir);
		const look = forwardToLonLat([dir.x, dir.y, dir.z]);
		const lon = look.lon;
		const lat = MathUtils.clamp(look.lat, -MAX_PITCH, MAX_PITCH);
		this.flyTarget = idx;
		this.startFly(toPos, lookTargetFrom(toPos, lon, lat), dur, {
			toFov: INTERIOR_FOV,
			crossfade: !dissolve,
			dissolveInterior: dissolve,
			onEnd: () => {
				this.mode = "interior";
				this.lon = lon;
				this.lat = lat;
				this.arrival = null;
				this.setInteriorView();
				this.activate(idx);
			},
		});
	}

	private canEnterFreefly(): boolean {
		return this.splatEnabled && this.projectionMode && !this.interiorBusy;
	}

	private enterFreefly() {
		if (this.mode !== "interior" || !this.canEnterFreefly()) return;
		this.yieldTour();
		this.closeInspect();
		this.hoveredNavIndex = -1;
		this.arrowReach = null;
		this.cursorReach = null;
		this.markers.setNavHover(null);
		this.markers.setArrowHover(null);
		this.markers.hideGhost();
		this.markers.hideSonar();
		this.addressing.setHover(null);
		this.addressing.closeMenu();
		this.freeflyVel.set(0, 0, 0);
		this.dockTarget = -1;
		this.dockReveal = 0;
		this.dockStaged = false;
		this.dockStillSince = 0;
		this.freeflySpeed = 1;
		this.freeflyFrom = this.currentIndex;

		const tex =
			this.currentIndex >= 0
				? this.panos[this.currentIndex]?.texture
				: null;
		this.mode = "freefly";
		this.setFreeflyView();
		if (tex && !this.reducedMotion) {
			this.sphereBMat.uniforms.map.value = tex;
			this.sphereBMat.uniforms.opacity.value = 1;
			this.sphereBMat.depthTest = false;
			this.sphereB.renderOrder = 20;
			this.sphereB.visible = true;
			this.sphereB.position.copy(this.camera.position);
			this.splatReveal = 0;
			this.splatRevealing = true;
			this.revealFovFrom = this.camera.fov;
			this.splatRevealMs = Math.min(
				SPLAT_REVEAL_MAX_MS,
				SPLAT_REVEAL_MS +
					Math.abs(this.camera.fov - FREEFLY_FOV) *
						REVEAL_FOV_MS_PER_DEG,
			);
		} else {
			this.clearPanoOverlay();
			this.splatReveal = 1;
			this.splatRevealing = false;
			this.setFov(FREEFLY_FOV);
		}
		this.noteInput();
		this.emit();
	}

	private returnToInterior(index: number) {
		if (this.mode !== "freefly" || !this.panos[index]) return;
		if (this.dockTarget === index) return;
		this.dockTarget = -1;
		this.dockReveal = 0;
		this.dockStaged = false;
		this.freeflyKeys.clear();
		this.freeflyVel.set(0, 0, 0);
		this.splatRevealing = false;
		this.clearPanoOverlay();
		this.cursor.hide();
		this.markers.hideGhost();

		const resident = !!this.panos[index].texture;
		if (!resident || !this.projectionMode || this.reducedMotion) {
			this.flyIntoInterior(index, FREEFLY_RETURN_MS);
			return;
		}
		this.reskinProxy(this.projection.material);
		if (this.proxyGroup) this.proxyGroup.visible = true;
		this.projection.syncBase(true);
		this.canvas.style.opacity = "0";
		this.flyIntoInterior(index, FREEFLY_RETURN_MS, { dissolve: true });
	}

	private get dockRadius(): number {
		return this.metrics.dockRadius;
	}

	private dockCandidate(): number {
		const cam = this.camera.position;
		const camLevel = this.hasFloorVolumes ? this.floorAt(cam) : -1;
		const from = this.freeflyFrom;
		const holdingOff =
			from >= 0 &&
			cam.distanceTo(v3(this.panos[from].position)) <= this.dockRadius;
		let best = -1;
		let bestD = this.dockRadius;
		for (let i = 0; i < this.panos.length; i++) {
			if (holdingOff && i === from) continue;
			const p = v3(this.panos[i].position);
			const dy = p.y - cam.y;
			if (Math.abs(dy) > this.metrics.dockMaxDy) continue;
			if (camLevel >= 0 && this.panoLevel[i] !== camLevel) continue;
			const d = Math.hypot(p.x - cam.x, dy * DOCK_DY_WEIGHT, p.z - cam.z);
			if (d >= bestD) continue;
			if (!this.panos[i].texture) {
				this.requestPano(i);
				continue;
			}
			if (!this.isTargetClear(p)) continue;
			bestD = d;
			best = i;
		}
		return best;
	}

	private applyDockReveal() {
		const on = this.dockReveal > 0.001;
		if (on !== this.dockStaged) {
			this.dockStaged = on;
			if (on) {
				this.reskinProxy(this.projection.material);
				if (this.proxyGroup) this.proxyGroup.visible = true;
				this.projection.syncBase(true);
			} else {
				this.setFreeflyView();
			}
		}
		if (on) this.updateProjection();
		this.canvas.style.opacity = on
			? easeInOut(MathUtils.clamp(this.dockReveal, 0, 1)).toFixed(3)
			: "1";
	}

	private cancelDock() {
		if (this.dockTarget < 0) return;
		this.dockTarget = -1;
		this.flyTarget = -1;
	}

	private commitDock(index: number) {
		this.dockTarget = -1;
		this.dockReveal = 0;
		this.freeflyKeys.clear();
		this.freeflyVel.set(0, 0, 0);
		this.splatRevealing = false;
		this.clearPanoOverlay();
		this.canvas.style.opacity = "1";
		this.cursor.hide();
		this.markers.hideGhost();
		this.mode = "interior";
		this.arrival = null;
		this.setInteriorView();
		this.activate(index);
	}

	private clickFromFreefly(clientX: number, clientY: number) {
		const hit = this.raycastInterior(clientX, clientY);
		const best = hit
			? this.autoHomeTarget(hit, this.floorAt(hit.point), -1)
			: this.nearestPanoTo(this.camera.position);
		if (best >= 0) this.returnToInterior(best);
	}

	exit() {
		if (this.mode !== "interior" || this.interiorBusy) return;
		this.setInside(false);
		this.releaseLock();
		this.director.abort();
		this.hoveredNavIndex = -1;
		this.markers.setNavHover(null);
		this.markers.hideSonar();
		this.addressing.setHover(null);
		this.addressing.closeMenu();
		this.canvas.style.cursor = "";
		this.clearFx();
		this.cursor.hide();
		this.markers.hideGhost();

		const tex =
			this.currentIndex >= 0
				? this.panos[this.currentIndex]?.texture
				: null;
		this.setOverviewView();
		this.mode = "transition";
		this.controls.enabled = false;
		this.controls.autoRotate = false;
		this.emit();

		const flyOut = () => {
			this.clearPanoOverlay();
			this.startFly(
				this.browsePos.clone(),
				this.browseTarget.clone(),
				1000,
				{
					toFov: OVERVIEW_FOV,
					onEnd: () => {
						this.mode = "overview";
						this.controls.target.copy(this.browseTarget);
						this.camera.position.copy(this.browsePos);
						this.controls.enabled = true;
						this.controls.update(0);
						this.controls.autoRotate = true;
						this.emit();
					},
				},
			);
		};

		if (!tex || this.reducedMotion) {
			flyOut();
			return;
		}
		this.sphereBMat.uniforms.map.value = tex;
		this.sphereBMat.uniforms.opacity.value = 1;
		this.sphereBMat.depthTest = false;
		this.sphereB.renderOrder = 20;
		this.sphereB.visible = true;
		this.sphereB.position.copy(this.camera.position);
		this.crossfade = {
			armed: performance.now(),
			deadline: 0,
			dur: ENTER_CROSSFADE_MS,
			direction: "out",
			onEnd: flyOut,
		};
	}

	traverseTo(index: number) {
		if (this.mode !== "interior" || !this.panos[index]) return;
		this.yieldTour();
		this.traverse(index);
	}

	goBack() {
		if (this.mode !== "interior" || this.interiorBusy) return;
		this.yieldTour();
		const prev = this.history.pop();
		if (prev == null) {
			this.exit();
			return;
		}
		this.traverse(prev, true);
	}

	toggleTour() {
		if (this.mode !== "interior") return;
		if (this.director.active) {
			this.yieldTour();
			return;
		}
		if (!this.navGraph || this.currentIndex < 0) return;
		this.stopLookInertia();
		this.director.start(
			planZoneTour(
				this.navGraph,
				(i) => this.panos[i]?.zone ?? "",
				this.currentIndex,
			),
		);
	}

	private yieldTour() {
		if (!this.director.active) return;
		this.director.stop();
		if (this.director.active) this.hurryMove();
	}

	toggleSonar() {
		if (this.mode !== "interior") return;
		this.noteInput();
		if (this.markers.sonarActive) {
			this.markers.hideSonar();
		} else {
			this.markers.buildSonar(
				this.navNode(this.currentIndex),
				this.panos,
				this.currentIndex,
			);
			this.markers.startSonar(performance.now(), this.camera);
		}
		this.emit();
	}

	jumpToLevel(level: number) {
		if (this.mode !== "interior" || this.interiorBusy) return;
		if (this.panoLevel[this.currentIndex] === level) return;
		this.yieldTour();
		const cur = v3(this.panos[this.currentIndex].position);
		let best = -1;
		let bestD = Infinity;
		for (let i = 0; i < this.panos.length; i++) {
			if (this.panoLevel[i] !== level) continue;
			const d = cur.distanceToSquared(v3(this.panos[i].position));
			if (d < bestD) {
				bestD = d;
				best = i;
			}
		}
		if (best >= 0) this.traverse(best);
	}

	getFacingDeg(): number {
		this.camera.getWorldDirection(_flyDir);
		const f = toMap(this.mapBasis, [_flyDir.x, _flyDir.y, _flyDir.z]);
		if (Math.hypot(f.u, f.v) < 1e-4) return (this.lon * 180) / Math.PI;
		return (Math.atan2(f.v, f.u) * 180) / Math.PI;
	}

	private stepToward(dirX: number, dirZ: number) {
		if (
			this.mode !== "interior" ||
			this.interiorBusy ||
			this.currentIndex < 0
		)
			return;
		this.yieldTour();
		const cur = this.panos[this.currentIndex].position;
		let best = -1;
		let bestDist2 = Infinity;
		for (let i = 0; i < this.panos.length; i++) {
			if (i === this.currentIndex) continue;
			const p = this.panos[i].position;
			if (Math.abs(p[1] - cur[1]) > this.metrics.wasdRise) continue;
			const dx = p[0] - cur[0];
			const dz = p[2] - cur[2];
			const dist2 = dx * dx + dz * dz;
			const stride = this.metrics.wasdStep;
			if (dist2 < 1e-6 || dist2 > stride * stride) continue;
			if ((dx * dirX + dz * dirZ) / Math.sqrt(dist2) < WASD_DIR_COS)
				continue;
			if (dist2 < bestDist2) {
				bestDist2 = dist2;
				best = i;
			}
		}
		if (best >= 0) this.traverse(best);
	}

	private peekStart() {
		if (this.mode !== "interior" || this.interiorBusy) return;
		this.releaseLock();
		this.savedInterior = {
			pos: this.camera.position.clone(),
			lon: this.lon,
			lat: this.lat,
			index: this.currentIndex,
			fov: this.camera.fov,
		};
		const userPos = this.currentUserWorldPos();
		this.markers.positionYouMarker(userPos);
		const { axis, sign } = this.mapBasis;
		this.locateClip.normal.set(0, 0, 0).setComponent(axis, -sign);
		this.locateClip.constant =
			sign *
			(userPos.getComponent(axis) + sign * this.metrics.sliceAboveEye);
		this.renderer.clippingPlanes = [this.locateClip];
		const flat = userPos.clone().sub(this.sceneCenter);
		flat.y = 0;
		if (flat.lengthSq() < 1e-6) flat.set(0, 0, 1);
		flat.normalize();
		const toPos = this.sceneCenter
			.clone()
			.addScaledVector(flat, this.sceneMaxDim * 1.5);
		toPos.y += this.sceneMaxDim * 1.2;
		this.startFly(toPos, this.sceneCenter.clone(), 850, {
			toFov: OVERVIEW_FOV,
			onMid: () => {
				this.setPeekView();
			},
			onEnd: () => {
				this.mode = "peek";
				this.emit();
				if (!this.peekHeld) this.peekEnd();
			},
		});
	}

	private peekEnd() {
		if (this.mode !== "peek" || !this.savedInterior) return;
		this.renderer.clippingPlanes = [];
		const s = this.savedInterior;
		this.startFly(s.pos.clone(), lookTargetFrom(s.pos, s.lon, s.lat), 800, {
			toFov: s.fov,
			onMid: () => {
				this.setInteriorView();
			},
			onEnd: () => {
				this.mode = "interior";
				this.lon = s.lon;
				this.lat = s.lat;
				this.currentIndex = s.index;
				this.activate(s.index);
			},
		});
	}

	peekDown() {
		if (this.mode !== "interior" || this.interiorBusy) return;
		this.yieldTour();
		this.peekHeld = true;
		this.onHold?.(true);
		this.peekStart();
	}
	peekUp() {
		if (!this.peekHeld) return;
		this.peekHeld = false;
		this.onHold?.(false);
		this.requestLock();
		if (this.mode === "peek") this.peekEnd();
	}

	private disposeObject(obj: Object3D) {
		obj.traverse((o) => {
			const m = o as Mesh;
			if (!m.isMesh && !(o as { isLine?: boolean }).isLine) return;
			m.geometry?.dispose();
			const mats = Array.isArray(m.material) ? m.material : [m.material];
			for (const mat of mats) {
				if (
					mat &&
					mat !== this.projection.material &&
					mat !== this.polyMaterial
				)
					mat.dispose();
			}
		});
	}

	private clearScene() {
		this.director.abort();
		if (this.liteRoot) {
			this.scene.remove(this.liteRoot);
			this.disposeObject(this.liteRoot);
			this.liteRoot = null;
		}
		if (this.proxyGroup) {
			this.scene.remove(this.proxyGroup);
			this.disposeObject(this.proxyGroup);
			this.proxyGroup = null;
		}
		for (const m of this.proxyColorMats) m.dispose();
		this.proxyColorMats = [];
		this.projection.clearBase(this.scene);
		this.streamer.reset();
		this.connectors = [];
		this.navGraph = null;
		this.nodeDir = [];
		this.chapters = [];
		this.history = [];
		this.visited.clear();
		this.pendingTravel = null;
		this.arrival = null;
		this.sceneScale = DEFAULT_SCALE;
		this.metrics = DEFAULT_METRICS;
		this.minimaps = [];
		this.mapLabels = [];
		this.mapBasis = readBasis(undefined);
		this.levelWord = "floor";
		this.panoLevel = [];
		this.minimapPrefetch = [];
		this.sphereAMat.uniforms.map.value = DUMMY_TEX;
		this.sphereBMat.uniforms.map.value = DUMMY_TEX;
		this.sphereA.material = this.sphereAMat;
		this.sphereA.scale.setScalar(1);
		this.currentIndex = -1;
		this.flyTarget = -1;
		this.markers.clear();
		this.sphereA.visible = false;
		this.sphereB.visible = false;
		this.transition = null;
		this.move = null;
		this.crossfade = null;
		this.clearPanoOverlay();
		this.interiorBusy = false;
		this.exitWhenLanded = false;
		this.peekHeld = false;
		this.savedInterior = null;
		this.renderer.clippingPlanes = [];
		this.hoveredNavIndex = -1;
		this.cursorReach = null;
		this.proxyView = false;
		this.splat.clear();
		this.splatView = true;
		this.splatReveal = 0;
		this.splatRevealing = false;
		this.splatRevealMs = SPLAT_REVEAL_MS;
		this.revealFovFrom = FREEFLY_FOV;
		this.freeflySpeed = 1;
		this.freeflyKeys.clear();
		this.freeflyVel.set(0, 0, 0);
		this.dockTarget = -1;
		this.dockReveal = 0;
		this.dockStaged = false;
		this.freeflyFrom = -1;
		this.dockStillSince = 0;
		this.stopLookInertia();
		this.scene.background = this.bgColor;
		this.canvas.style.opacity = "1";
		this.addressing.reset();
		this.releaseLock();
		this.lockClickPending = false;
		this.canvas.style.cursor = "";
		this.clearFx();
		for (const l of this.sonarLabels) l.style.display = "none";
	}

	async prepareTour(
		source: TourSource,
		token: number,
	): Promise<PreparedTour | null> {
		{
			let manifest: TourManifest | null = null;
			if (source.manifestUrl) {
				const res = await fetch(source.manifestUrl, {
					cache: "no-store",
				});
				if (token !== this.loadToken || this.disposed) return null;
				if (res.ok) manifest = (await res.json()) as TourManifest;
			}
			if (token !== this.loadToken || this.disposed) return null;

			const mmList =
				manifest && Array.isArray(manifest.minimaps)
					? manifest.minimaps
					: [];
			const minimaps: MinimapSlice[] = mmList.map((m) => ({
				...m,
				url: source.resolveMinimap(m.file),
			}));
			const minimapPrefetch = minimaps.map((m) => {
				const img = new Image();
				img.src = m.url;
				return img;
			});

			const list =
				manifest && Array.isArray(manifest.panos) ? manifest.panos : [];
			const entries: PanoEntry[] = list.map((p) => {
				const { url, placeholderUrl } = source.resolvePano(p.file);
				return {
					id: p.id,
					name: p.name,
					zone: p.zone,
					level: p.level,
					position: p.position,
					forward: p.forward,
					url,
					placeholderUrl,
					texture: null,
					placeholderTexture: null,
					hasFull: false,
					requested: false,
				};
			});

			const connectors =
				manifest && Array.isArray(manifest.connectors)
					? manifest.connectors
					: [];
			const objectIds =
				manifest && Array.isArray(manifest.objects)
					? manifest.objects
					: [];
			const mapLabels =
				manifest && Array.isArray(manifest.map_labels)
					? manifest.map_labels
					: [];
			const mapBasis = readBasis(minimaps[0]?.basis);
			const word = manifest?.profile?.level_word;
			const levelWord = typeof word === "string" && word ? word : "floor";

			let proxyRoot: Group | null = null;
			if (manifest?.proxy) {
				try {
					proxyRoot = await loadGLB(
						source.resolveProxy(manifest.proxy),
					);
				} catch {
					proxyRoot = null;
				}
			}
			let lite: Group | null = null;
			if (source.dollhouseUrl) {
				try {
					lite = await loadGLB(source.dollhouseUrl);
				} catch {
					lite = null;
				}
			}
			const splatReady = source.splatUrl
				? await this.splat.prepare(source.splatUrl)
				: false;
			if (token !== this.loadToken || this.disposed) {
				if (splatReady) this.splat.discardStaged();
				return null;
			}

			const breathe = this.isEmpty ? null : yieldFrame;
			if (breathe) await breathe();

			const panoLevel = entries.map((p) =>
				typeof p.level === "number" &&
				p.level >= 0 &&
				p.level < minimaps.length
					? p.level
					: levelForPosition(minimaps, p.position),
			);

			if (lite) prepareLitScene(lite);
			if (proxyRoot) prepareLitScene(proxyRoot);

			const framed = lite ?? proxyRoot;
			if (!framed) {
				return {
					entries,
					proxyRoot,
					lite,
					connectors,
					objectIds,
					minimaps,
					minimapPrefetch,
					mapLabels,
					mapBasis,
					levelWord,
					panoLevel,
					splatReady,
					splatTransform: source.splatTransform,
					box: null,
					sceneScale: DEFAULT_SCALE,
					metrics: DEFAULT_METRICS,
					navGraph: null,
				};
			}

			framed.updateMatrixWorld(true);
			if (proxyRoot && proxyRoot !== framed)
				proxyRoot.updateMatrixWorld(true);
			if (breathe) await breathe();

			const box = new Box3().setFromObject(framed);
			const size = box.getSize(new Vector3());
			const center = box.getCenter(new Vector3());
			const sceneMaxDim = Math.max(size.x, size.y, size.z) || 1;

			const sceneScale = measureSceneScale(
				sceneMaxDim,
				entries.map((p) => p.position),
				proxyRoot,
			);
			const metrics = navMetrics(sceneScale);
			console.info(`[orbit] scene scale — ${describeScale(sceneScale)}`);
			if (breathe) await breathe();

			const navGraph = await buildNavGraph(
				entries.map((p) => ({ position: p.position, zone: p.zone })),
				panoLevel,
				(a, b) => this.segmentBlocked(a, b, proxyRoot, metrics),
				metrics,
				breathe ?? undefined,
			);

			if (token !== this.loadToken || this.disposed) {
				if (splatReady) this.splat.discardStaged();
				return null;
			}

			return {
				entries,
				proxyRoot,
				lite,
				connectors,
				objectIds,
				minimaps,
				minimapPrefetch,
				mapLabels,
				mapBasis,
				levelWord,
				panoLevel,
				splatReady,
				box,
				center,
				sceneScale,
				metrics,
				navGraph,
				splatTransform: source.splatTransform,
			};
		}
	}

	commitTour(prepared: PreparedTour) {
		this.setInside(false);
		this.mode = "loading";
		this.controls.enabled = false;
		this.clearScene();
		if (prepared.splatReady) this.splat.commit();
		else this.splat.discardStaged();
		this.splat.setTransform(prepared.splatTransform ?? IDENTITY_TRANSFORM);
		this.minimaps = prepared.minimaps;
		this.minimapPrefetch = prepared.minimapPrefetch;
		this.mapLabels = prepared.mapLabels;
		this.mapBasis = prepared.mapBasis;
		this.levelWord = prepared.levelWord;
		this.panoLevel = prepared.panoLevel;
		this.sceneScale = prepared.sceneScale;
		this.metrics = prepared.metrics;
		this.navGraph = prepared.navGraph;
		this.applyScene(prepared);
	}

	warmTour(source: TourSource) {
		if (this.disposed) return;
		if (this.warmed?.source === source || this.warming?.source === source)
			return;
		const token = this.loadToken;
		const promise = this.prepareTour(source, token)
			.then((prepared) => {
				if (this.disposed || token !== this.loadToken) return null;
				if (prepared) this.warmed = { source, prepared };
				return prepared;
			})
			.catch(() => null)
			.finally(() => {
				if (this.warming?.source === source) this.warming = null;
			});
		this.warming = { source, promise };
	}

	async loadTour(
		source: TourSource,
		commitVia?: (commit: () => void) => void,
	) {
		const token = ++this.loadToken;
		if (this.isEmpty) this.showOverlay("loading scene…");
		try {
			const prepared = await this.takePrepared(source, token);
			if (!prepared || !this.isCurrentLoad(token)) return;
			const commit = () => {
				if (!this.isCurrentLoad(token)) return;
				this.commitTour(prepared);
			};
			if (commitVia) commitVia(commit);
			else commit();
		} catch (e) {
			if (!this.isCurrentLoad(token)) return;
			this.failLoad(e);
		}
	}

	private async takePrepared(source: TourSource, token: number) {
		if (this.warmed?.source === source) {
			const { prepared } = this.warmed;
			this.warmed = null;
			return prepared;
		}
		if (this.warming?.source === source) {
			const prepared = await this.warming.promise;
			if (prepared) {
				this.warmed = null;
				return prepared;
			}
		}
		return this.prepareTour(source, token);
	}

	get isEmpty(): boolean {
		return !this.liteRoot && !this.proxyGroup;
	}

	nextLoadToken(): number {
		return ++this.loadToken;
	}

	isCurrentLoad(token: number): boolean {
		return token === this.loadToken && !this.disposed;
	}

	failLoad(e: unknown) {
		this.splat.discardStaged();
		this.clearScene();
		this.mode = "empty";
		this.showOverlay(
			`failed to load scene: ${e instanceof Error ? e.message : String(e)}`,
			{ spinner: false, err: true },
		);
	}

	private applyScene(prepared: PreparedTour) {
		const { entries, proxyRoot, lite, connectors, objectIds } = prepared;
		this.connectors = connectors;
		this.inspectable = new Set(objectIds);
		this.streamer.reset(entries);
		this.projectionMode = !!proxyRoot;
		this.sharedOverview = !lite && !!proxyRoot;

		if (!lite && !proxyRoot) {
			this.mode = "empty";
			this.showOverlay("nothing to show for this scene", {
				spinner: false,
				err: true,
			});
			return;
		}

		if (lite) {
			this.liteRoot = lite;
			this.scene.add(lite);
		}
		if (proxyRoot) {
			this.projection.setup(proxyRoot, this.sphereA);
			this.proxyGroup = proxyRoot;
			this.scene.add(proxyRoot);
		}

		if (this.liteRoot) {
			this.addressing.register(this.liteRoot);
			for (const o of collectObjects(this.liteRoot)) {
				const label = o.userData.objLabel as string | undefined;
				if (label) this.liteByLabel.set(label, o);
			}
		}
		if (this.proxyGroup) {
			this.addressing.register(this.proxyGroup);
			this.colorProxyObjects();
			this.projection.buildBase(this.proxyGroup, this.scene);
		}

		const box = prepared.box!;
		this.sceneCenter.copy(prepared.center!);
		const size = box.getSize(new Vector3());
		this.sceneMaxDim = Math.max(size.x, size.y, size.z) || 1;
		this.sceneTopY = box.max.y;
		this.sceneBottomY = box.min.y;
		this.rig.fit(box);

		this.camera.near = this.metrics.cameraNear;
		this.camera.far = this.metrics.cameraFar;

		this.buildSceneDirectory(entries);

		this.markers.build(this.sceneMaxDim, this.metrics);
		this.framingHull = framingHull(box, this.sceneCenter);
		this.groundAnchor.copy(groundAnchor(box, this.sceneCenter));
		this.frameOverview();
		this.camera.position.copy(this.browsePos);
		this.camera.lookAt(this.browseTarget);
		this.camera.updateProjectionMatrix();
		this.controls.target.copy(this.browseTarget);
		this.controls.enabled = true;
		this.controls.update(0);
		this.controls.autoRotate = true;

		this.setOverviewView();
		this.mode = "overview";
		this.hideOverlay();
	}

	private frameOverview() {
		if (this.framingHull.length === 0) return;

		this.browsePos
			.copy(this.sceneCenter)
			.addScaledVector(
				BROWSE_DIR,
				fitDistance(
					this.framingHull,
					this.sceneCenter,
					BROWSE_DIR,
					OVERVIEW_FOV,
					this.camera.aspect,
				) * BROWSE_MARGIN,
			);
		this.browseTarget.copy(this.sceneCenter);

		this.camera.fov = OVERVIEW_FOV;
		this.camera.updateProjectionMatrix();
		const pan = groundLinePan(
			this.camera,
			this.framingHull,
			this.groundAnchor,
			this.browsePos,
			this.browseTarget,
		);
		_panShift.setFromMatrixColumn(this.camera.matrixWorld, 1).normalize();
		this.browsePos.addScaledVector(_panShift, pan);
		this.browseTarget.addScaledVector(_panShift, pan);

		const across = centrePan(
			this.camera,
			this.framingHull,
			this.browsePos,
			this.browseTarget,
			0,
			_panShift,
		);
		_centreRight
			.setFromMatrixColumn(this.camera.matrixWorld, 0)
			.normalize();
		this.browsePos.addScaledVector(_centreRight, across);
		this.browseTarget.addScaledVector(_centreRight, across);

		const framed = this.browsePos.distanceTo(this.browseTarget);
		this.controls.minDistance = framed * DOLLY_NEAR;
		this.controls.maxDistance = framed * DOLLY_FAR;
	}

	private segmentBlocked(
		a: [number, number, number],
		b: [number, number, number],
		proxy: Group | null = this.proxyGroup,
		metrics: NavMetrics = this.metrics,
	): boolean {
		if (!proxy) return false;
		const from = v3(a);
		const d = v3(b).sub(from);
		const dist = d.length();
		if (dist < 1e-3) return false;
		d.divideScalar(dist);
		const trim = metrics.losTrim;
		this.occluder.set(from, d);
		this.occluder.near = trim;
		this.occluder.far = dist - trim;
		if (this.occluder.far <= this.occluder.near) return false;
		return this.occluder.intersectObject(proxy, true).length > 0;
	}

	private isTargetClear(target: Vector3): boolean {
		if (!this.proxyGroup) return true;
		const cx = this.camera.position.x;
		const cy = this.camera.position.y;
		const cz = this.camera.position.z;
		const spread = this.metrics.aimSpread;
		const trim = this.metrics.aimTrim;
		for (const [ox, oz] of [
			[0, 0],
			[spread, 0],
			[-spread, 0],
			[0, spread],
			[0, -spread],
		]) {
			_losFrom.set(cx + ox, cy, cz + oz);
			_losDir.copy(target).sub(_losFrom);
			const dist = _losDir.length();
			if (dist < this.metrics.aimMinDist) return true;
			_losDir.divideScalar(dist);
			this.occluder.set(_losFrom, _losDir);
			this.occluder.near = trim;
			this.occluder.far = dist - trim;
			if (this.occluder.far <= this.occluder.near) continue;
			if (
				this.occluder.intersectObject(this.proxyGroup, true).length ===
				0
			)
				return true;
		}
		return false;
	}

	private buildSceneDirectory(entries: PanoEntry[]) {
		this.nodeDir = entries.map((p, i) => ({
			index: i,
			name: p.name ?? null,
			zone: p.zone ?? null,
			level: this.panoLevel[i],
		}));
		const chapters: Chapter[] = [];
		for (let i = 0; i < entries.length; i++) {
			const zone = entries[i].zone ?? "";
			const found = chapters.find((c) => c.zone === zone);
			if (found) found.count++;
			else chapters.push({ zone, count: 1, firstIndex: i });
		}
		this.chapters = chapters;
	}

	private tickInspect(now: number) {
		const obj = this.addressing.hoveredObject;
		const label = obj ? ((obj.userData.objLabel as string) ?? null) : null;
		if (label !== this.hoverLabel) {
			this.hoverLabel = label;
			this.hoverSince = now;
			if (this.inspect) this.closeInspect();
		}
		if (
			!label ||
			this.inspect ||
			!this.inspectable.has(label) ||
			!this.liteByLabel.has(label) ||
			now - this.hoverSince < INSPECT_DWELL_MS
		)
			return;
		this.openInspect(label);
	}

	private openInspect(label: string) {
		const src = this.liteByLabel.get(label);
		if (!src) return;
		const scene = new Scene();
		scene.add(new HemisphereLight(0xffffff, 0x2a2f38, 1.5));
		const key = new DirectionalLight(0xffffff, 1.8);
		key.position.set(2, 3, 2.5);
		scene.add(key);
		const clone = src.clone(true);
		clone.visible = true;
		clone.traverse((o) => {
			o.visible = true;
		});
		clone.updateMatrixWorld(true);
		const box = new Box3().setFromObject(clone);
		if (box.isEmpty()) return;
		const centre = box.getCenter(new Vector3());
		const radius = Math.max(
			1e-3,
			box.getSize(new Vector3()).length() * 0.5,
		);
		clone.position.sub(centre);
		const pivot = new Group();
		pivot.add(clone);
		scene.add(pivot);
		this.inspectCam.position
			.set(0.62, 0.42, 1)
			.normalize()
			.multiplyScalar(
				(radius / Math.sin((this.inspectCam.fov * Math.PI) / 360)) *
					1.12,
			);
		this.inspectCam.near = Math.max(0.01, radius * 0.05);
		this.inspectCam.far = radius * 20;
		this.inspectCam.lookAt(0, 0, 0);
		this.inspectCam.updateProjectionMatrix();
		this.inspectScene = scene;
		this.inspectPivot = pivot;
		const rect = this.canvas.getBoundingClientRect();
		const m = INSPECT_MARGIN;
		const at = this.aim();
		this.inspect = {
			label,
			x: Math.min(
				Math.max(at.x + INSPECT_GAP, rect.left + m),
				rect.right - INSPECT_SIZE - m,
			),
			y: Math.min(
				Math.max(at.y - INSPECT_SIZE - INSPECT_GAP, rect.top + m),
				rect.bottom - INSPECT_SIZE - m,
			),
			w: INSPECT_SIZE,
			h: INSPECT_SIZE,
		};
		this.emit();
	}

	private closeInspect() {
		if (!this.inspect) return;
		this.inspectScene = null;
		this.inspectPivot = null;
		this.inspect = null;
		this.emit();
	}

	private renderInspect(dt: number) {
		const ins = this.inspect;
		if (!ins || !this.inspectScene || !this.inspectPivot) return;
		this.inspectPivot.rotation.y += dt * INSPECT_SPIN;
		const rect = this.canvas.getBoundingClientRect();
		const x = ins.x - rect.left;
		const y = rect.height - (ins.y - rect.top) - ins.h;
		const prevAutoClear = this.renderer.autoClear;
		this.renderer.getClearColor(_prevClear);
		const prevClearAlpha = this.renderer.getClearAlpha();
		this.renderer.autoClear = false;
		this.renderer.setScissorTest(true);
		this.renderer.setViewport(x, y, ins.w, ins.h);
		this.renderer.setScissor(x, y, ins.w, ins.h);
		this.renderer.setClearColor(new Color(groundColor()), 1);
		this.renderer.clear(true, true, false);
		this.renderer.render(this.inspectScene, this.inspectCam);
		this.renderer.setScissorTest(false);
		this.renderer.setViewport(0, 0, rect.width, rect.height);
		this.renderer.setClearColor(_prevClear, prevClearAlpha);
		this.renderer.autoClear = prevAutoClear;
	}

	private refreshFloorArrows() {
		const cur =
			this.currentIndex >= 0 ? this.panoLevel[this.currentIndex] : -1;
		if (cur < 0) {
			this.markers.clearFloorArrows();
			return;
		}
		const here = v3(this.panos[this.currentIndex].position);
		const items: Array<{ index: number; up: boolean; pos: Vector3 }> = [];
		for (const step of [1, -1]) {
			const level = cur + step;
			let index = -1;
			let best = Infinity;
			for (let i = 0; i < this.panos.length; i++) {
				if (this.panoLevel[i] !== level) continue;
				const d = here.distanceToSquared(v3(this.panos[i].position));
				if (d < best) {
					best = d;
					index = i;
				}
			}
			if (index < 0) continue;
			items.push({
				index,
				up: step > 0,
				pos: here.clone().setY(here.y + step * this.metrics.arrowDist),
			});
		}
		this.markers.buildFloorArrows(items);
		this.markers.arrowGroup.visible = this.mode === "interior";
	}

	private tick = (time: number) => {
		const now = performance.now();
		const dt = this.lastFrame
			? Math.min(0.05, (time - this.lastFrame) / 1000)
			: 0;
		this.lastFrame = time;

		if (this.resizePending) {
			this.resizePending = false;
			this.resize();
		}

		this.easeZoom(dt);

		if (this.transition) {
			const tr = this.transition;
			const t = Math.min(1, (now - tr.start) / tr.dur);
			const e = easeInOut(t);
			this.camera.position.lerpVectors(tr.fromPos, tr.toPos, e);
			this.camera.quaternion.slerpQuaternions(tr.fromQuat, tr.toQuat, e);
			if (tr.toFov !== tr.fromFov) {
				this.camera.fov = tr.fromFov + (tr.toFov - tr.fromFov) * e;
				this.camera.updateProjectionMatrix();
			}
			this.camera.updateMatrixWorld();
			if (tr.dissolveInterior) {
				this.updateProjection();
				const d = MathUtils.clamp(
					(t - DISSOLVE_START) / (1 - DISSOLVE_START),
					0,
					1,
				);
				this.canvas.style.opacity = easeInOut(d).toFixed(3);
			}
			if (!tr.crossfade && !tr.dissolveInterior)
				this.travelFade.style.opacity = (
					Math.sin(Math.PI * t) * 0.5
				).toFixed(3);
			if (!tr.midDone && t >= 0.5) {
				tr.midDone = true;
				tr.onMid?.();
			}
			if (t >= 1) {
				const cb = tr.onEnd;
				const crossfade = tr.crossfade;
				this.transition = null;
				if (crossfade) {
					this.crossfade = {
						armed: 0,
						deadline: now + HANDOVER_WAIT_MS,
						dur: this.reducedMotion
							? REDUCED_DUR
							: ENTER_CROSSFADE_MS,
						direction: "in",
						onEnd: cb,
					};
				} else {
					this.travelFade.style.opacity = "0";
					this.canvas.style.opacity = "1";
					cb?.();
				}
			}
		} else if (this.crossfade) {
			this.tickCrossfade(now);
		} else if (this.mode === "overview") {
			// SECONDS, NOT FRAMES. Given no argument OrbitControls advances the spin by
			// a fixed angle PER CALL — its own docs quote the speed "at 60fps" — so the
			// scene turned 2.4x fast on a 144Hz display and slowed with the frame rate
			// through a hitch. Handed the frame's own dt it turns at a rate per second.
			this.controls.update(dt);
		} else if (this.mode === "interior") {
			if (this.move) {
				const mv = this.move;
				const t = Math.min(1, (now - mv.start) / mv.dur);
				const e = easeInOut(t);
				if (mv.ctrl) quadBezier(mv.fromPos, mv.ctrl, mv.toPos, e, _bez);
				else _bez.lerpVectors(mv.fromPos, mv.toPos, e);
				this.camera.position.copy(_bez);
				if (mv.sphere) {
					this.sphereBMat.uniforms.opacity.value = e;
					this.sphereA.position.copy(this.camera.position);
					this.sphereB.position.copy(this.camera.position);
				}
				this.setFx(mv.type, t);
				if (t >= 1) {
					this.move = null;
					this.finishMove(mv);
				}
			}
			if (this.projectionMode) {
				if (!this.proxyView) this.updateProjection();
			} else if (!this.move) {
				this.sphereA.position.copy(this.camera.position);
			}
			this.director.tick(now);
			this.tickLook(dt);
			if (!this.interiorBusy) {
				if (CENTER_CURSOR) {
					const at = this.aim();
					this.updateHover(at.x, at.y);
				}
				this.markers.updateNav(
					this.camera,
					this.lon,
					now,
					this.host.clientHeight,
				);
				this.markers.updateFloorArrows(now);
				if (this.markers.sonarActive) {
					this.markers.updateSonar(
						now,
						this.camera,
						this.host.clientHeight,
					);
					this.updateSonarLabels();
					if (!this.markers.sonarActive) this.emit();
				} else if (
					this.sonarLabels.some((l) => l.style.display !== "none")
				) {
					for (const l of this.sonarLabels) l.style.display = "none";
				}
				this.tickInspect(now);
				if (!this.dwellPulsed && now - this.lastInputAt > DWELL_MS) {
					this.dwellPulsed = true;
					this.markers.pulseExits(now, 1600);
				}
			}
		} else if (this.mode === "freefly") {
			this.tickFreefly(now, dt);
		} else if (this.mode === "peek") {
			const off = this.camera.position.clone().sub(this.sceneCenter);
			const a = PEEK_ROTATE_SPEED * dt;
			const c = Math.cos(a);
			const s = Math.sin(a);
			this.camera.position.x = this.sceneCenter.x + off.x * c - off.z * s;
			this.camera.position.z = this.sceneCenter.z + off.x * s + off.z * c;
			this.camera.lookAt(this.sceneCenter);
		}

		this.updateCursorRing();
		this.addressing.updateOutlines();
		this.splat.render(this.camera);
		this.composer.render();
		this.renderInspect(dt);

		if (this.captureWaiting) {
			const done = this.captureWaiting;
			this.captureWaiting = null;
			done(this.composite());
		}
	};

	capture(): Promise<HTMLCanvasElement | null> {
		if (this.disposed) return Promise.resolve(null);
		return new Promise((resolve) => {
			this.captureWaiting?.(null);
			this.captureWaiting = resolve;
		});
	}

	private composite(): HTMLCanvasElement | null {
		const w = this.canvas.width;
		const h = this.canvas.height;
		if (!w || !h) return null;
		const out = document.createElement("canvas");
		out.width = w;
		out.height = h;
		const ctx = out.getContext("2d");
		if (!ctx) return null;
		const splatCanvas = this.splat.canvasEl;
		if (splatCanvas && this.splatEnabled && splatCanvas.width > 0) {
			ctx.drawImage(splatCanvas, 0, 0, w, h);
		}
		ctx.drawImage(this.canvas, 0, 0, w, h);
		return out;
	}

	private tickFreefly(now: number, dt: number) {
		const cl = Math.cos(this.lat);
		const fx = cl * Math.cos(this.lon);
		const fy = Math.sin(this.lat);
		const fz = cl * Math.sin(this.lon);
		const rx = -Math.sin(this.lon);
		const rz = Math.cos(this.lon);
		const keys = this.freeflyKeys;
		_moveWish.set(0, 0, 0);
		if (keys.has("KeyW")) _moveWish.set(fx, fy, fz);
		if (keys.has("KeyS")) _moveWish.set(-fx, -fy, -fz);
		if (keys.has("KeyD")) {
			_moveWish.x += rx;
			_moveWish.z += rz;
		}
		if (keys.has("KeyA")) {
			_moveWish.x -= rx;
			_moveWish.z -= rz;
		}
		if (keys.has("Space") || keys.has("KeyE")) _moveWish.y += 1;
		if (keys.has("Shift") || keys.has("KeyQ")) _moveWish.y -= 1;
		const asking = _moveWish.lengthSq() > 0;
		if (asking) _moveWish.normalize();
		_moveWish.multiplyScalar(
			this.sceneMaxDim * FREEFLY_SPEED_FRAC * this.freeflySpeed,
		);
		const topSpeed = this.sceneMaxDim * FREEFLY_SPEED_FRAC;
		const still =
			!asking &&
			this.dockTarget < 0 &&
			this.freeflyVel.length() < topSpeed * DOCK_STILL_SPEED_FRAC;
		if (!still) this.dockStillSince = 0;
		else if (this.dockStillSince === 0) this.dockStillSince = now;

		if (asking) {
			this.cancelDock();
		} else if (
			this.dockTarget < 0 &&
			this.projectionMode &&
			this.splat.isActive &&
			this.dockStillSince > 0 &&
			now - this.dockStillSince >= this.dockDelayMs
		) {
			const cand = this.dockCandidate();
			if (cand >= 0) {
				this.dockTarget = cand;
				this.flyTarget = cand;
			}
		}
		let dockDist = Infinity;
		if (this.dockTarget >= 0) {
			const to = v3(this.panos[this.dockTarget].position).sub(
				this.camera.position,
			);
			dockDist = to.length();
			_moveWish.copy(to).multiplyScalar(DOCK_SEEK_GAIN);
			if (_moveWish.length() > topSpeed) _moveWish.setLength(topSpeed);
		}

		this.freeflyVel.lerp(
			_moveWish,
			1 - Math.exp(-(dt * 1000) / FREEFLY_VEL_TAU),
		);
		this.camera.position.addScaledVector(this.freeflyVel, dt);
		this.tickLook(dt);

		const wanted =
			this.dockTarget >= 0
				? MathUtils.clamp(1 - dockDist / this.metrics.dockReveal, 0, 1)
				: 0;
		if (wanted > 0 || this.dockReveal > 0.001) {
			this.dockReveal +=
				(wanted - this.dockReveal) *
				(1 - Math.exp(-(dt * 1000) / DOCK_REVEAL_TAU));
			this.applyDockReveal();
		}
		if (this.dockTarget >= 0 && dockDist < this.metrics.dockArrive) {
			this.commitDock(this.dockTarget);
			return;
		}

		if (!this.splatRevealing) return;
		this.splatReveal = Math.min(
			1,
			this.splatReveal + (dt * 1000) / this.splatRevealMs,
		);
		const e = easeInOut(this.splatReveal);
		this.sphereB.position.copy(this.camera.position);
		this.sphereBMat.uniforms.opacity.value = 1 - e;
		if (this.revealFovFrom !== FREEFLY_FOV) {
			this.setFov(
				this.revealFovFrom + (FREEFLY_FOV - this.revealFovFrom) * e,
			);
		}
		if (this.splatReveal >= 1) {
			this.splatRevealing = false;
			this.clearPanoOverlay();
		}
	}

	private tickCrossfade(now: number) {
		const cf = this.crossfade;
		if (!cf) return;
		if (cf.armed === 0) {
			const tex =
				this.flyTarget >= 0
					? this.panos[this.flyTarget]?.texture
					: null;
			if (!tex && now < cf.deadline) return;
			if (tex) {
				this.sphereBMat.uniforms.map.value = tex;
				this.sphereBMat.uniforms.opacity.value = 0;
				this.sphereBMat.depthTest = false;
				this.sphereB.renderOrder = 20;
				this.sphereB.visible = true;
			}
			cf.armed = now;
		}
		const t = Math.min(1, (now - cf.armed) / cf.dur);
		const e = easeInOut(t);
		if (this.sphereB.visible) {
			this.sphereB.position.copy(this.camera.position);
			this.sphereBMat.uniforms.opacity.value =
				cf.direction === "out" ? 1 - e : e;
		}
		if (t < 1) return;
		this.crossfade = null;
		this.clearPanoOverlay();
		cf.onEnd?.();
	}

	private clearPanoOverlay() {
		this.sphereB.visible = false;
		this.sphereBMat.uniforms.opacity.value = 0;
		this.sphereBMat.depthTest = true;
		this.sphereB.renderOrder = 1;
	}

	private updateOverviewCursor() {
		const aimX = this.pointerClientX;
		const aimY = this.pointerClientY;
		const active =
			this.pointerInside &&
			!this.pointerDown &&
			!this.move &&
			this.panos.length > 0;
		if (!active) {
			this.overviewHit = null;
			this.overviewTarget = -1;
			this.cursor.hide();
			if (!this.pointerDown) this.canvas.style.cursor = "";
			return;
		}
		const moved =
			aimX !== this.overviewAimX ||
			aimY !== this.overviewAimY ||
			!this.overviewCam.equals(this.camera.position) ||
			!this.overviewPivot.equals(this.controls.target);
		if (moved) {
			this.overviewAimX = aimX;
			this.overviewAimY = aimY;
			this.overviewCam.copy(this.camera.position);
			this.overviewPivot.copy(this.controls.target);
			this.overviewHit = this.raycastOverview(aimX, aimY);
			this.overviewTarget = this.overviewHit
				? this.nearestPanoTo(this.overviewHit.point)
				: -1;
		}
		const hit = this.overviewHit;
		this.canvas.style.cursor = hit ? "pointer" : "";
		if (!hit) {
			this.cursor.hide();
			return;
		}
		const travel =
			this.overviewTarget >= 0
				? _ovTravel
						.fromArray(this.panos[this.overviewTarget].position)
						.sub(hit.point)
				: null;
		this.cursor.setColor(CURSOR_CLEAR);
		this.cursor.update(hit, this.camera, this.host.clientHeight, travel);
	}

	private updateCursorRing() {
		if (this.mode === "overview") {
			this.updateOverviewCursor();
			this.setReticle(false, false);
			return;
		}
		const active =
			this.isLookMode &&
			!this.interiorBusy &&
			(CENTER_CURSOR || this.pointerInside) &&
			!this.markers.hoveredNav &&
			!this.markers.hoveredArrow;
		const at = this.aim();
		const hits = active ? this.raycastInteriorAll(at.x, at.y) : [];
		const hit = hits[0] ?? null;
		if (CENTER_CURSOR && this.isLookMode) {
			this.canvas.style.cursor = this.locked ? "none" : "";
		}
		let ghosted = false;
		let reach: ReachTarget | null = null;
		let travel: Vector3 | null = null;
		if (this.mode === "freefly" && this.dockTarget >= 0) {
			this.markers.showGhost(
				this.destinationFloor(this.dockTarget),
				{ to: this.dockTarget, type: "walk", dy: 0 },
				this.camera,
				this.host.clientHeight,
			);
			ghosted = true;
		} else if (hit && this.mode === "freefly") {
			const targetIdx = this.autoHomeTarget(
				hit,
				this.floorAt(hit.point),
				-1,
			);
			if (targetIdx >= 0) {
				this.cursor.setColor(CURSOR_CLEAR);
				this.markers.showGhost(
					this.destinationFloor(targetIdx),
					{ to: targetIdx, type: "walk", dy: 0 },
					this.camera,
					this.host.clientHeight,
				);
				ghosted = true;
				reach = {
					index: targetIdx,
					level: this.panoLevel[targetIdx] ?? -1,
					levelDelta: 0,
				};
			}
		} else if (hit && this.currentIndex >= 0) {
			const curLevel = this.panoLevel[this.currentIndex] ?? -1;
			const aim = this.resolveAim(hits);
			if (aim) {
				const { marker, occluded, index: targetIdx } = aim;
				const destLevel = this.panoLevel[targetIdx] ?? -1;
				if (occluded) {
					this.cursor.setColor(NAV_COLORS.portal);
					reach = {
						index: targetIdx,
						level: destLevel,
						levelDelta: 0,
					};
					this.markers.showGhost(
						marker,
						{ to: targetIdx, type: "portal", dy: 0 },
						this.camera,
						this.host.clientHeight,
					);
					ghosted = true;
				} else {
					const crossesLevel =
						curLevel >= 0 &&
						destLevel >= 0 &&
						destLevel !== curLevel;
					this.cursor.setColor(
						crossesLevel ? NAV_COLORS.vertical : CURSOR_CLEAR,
					);
					travel = v3(this.panos[targetIdx].position).sub(
						this.camera.position,
					);
				}
			} else {
				this.cursor.setColor(CURSOR_CLEAR);
			}
		}
		if (!ghosted) this.markers.hideGhost();
		const changed =
			(this.cursorReach?.index ?? -1) !== (reach?.index ?? -1);
		this.cursorReach = reach;
		if (changed) {
			if (reach) this.requestPano(reach.index);
			this.emit();
		}
		this.cursor.update(hit, this.camera, this.host.clientHeight, travel);
		this.setReticle(this.isLookMode && !this.interiorBusy, !!hit);
	}

	private setReticle(show: boolean, onSurface: boolean) {
		const want =
			!CENTER_CURSOR || !show
				? "0"
				: onSurface
					? String(RETICLE_ON_SURFACE)
					: String(RETICLE_IN_VOID);
		if (this.reticle.style.opacity !== want)
			this.reticle.style.opacity = want;
	}

	private updateSonarLabels() {
		const targets = this.markers.sonarLabelTargets(
			this.camera,
			this.canvas,
			8,
		);
		const rect = this.canvas.getBoundingClientRect();
		for (let i = 0; i < this.sonarLabels.length; i++) {
			const el = this.sonarLabels[i];
			const t = targets[i];
			if (!t) {
				el.style.display = "none";
				continue;
			}
			el.textContent = t.name;
			el.style.left = `${t.x - rect.left}px`;
			el.style.top = `${t.y - rect.top}px`;
			el.style.display = "block";
		}
		while (this.sonarLabels.length < targets.length) {
			const el = document.createElement("div");
			Object.assign(el.style, {
				position: "absolute",
				transform: "translate(-50%, -140%)",
				padding: "1px 6px",
				borderRadius: "5px",
				background: "rgb(var(--ground-rgb) / 0.82)",
				color: "rgb(var(--accent-rgb))",
				font: "600 10px ui-sans-serif, system-ui, sans-serif",
				whiteSpace: "nowrap",
				pointerEvents: "none",
				zIndex: "3",
			});
			this.host.appendChild(el);
			this.sonarLabels.push(el);
		}
	}
}
