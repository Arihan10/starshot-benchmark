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
	type ShaderMaterial,
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
import { loadGLB } from "./loaders";
import {
	DUMMY_TEX,
	makePanoMaterial,
	makePolyMaterial,
	SPHERE_RADIUS,
} from "./materials";
import {
	CURSOR_CLEAR,
	HOTSPOT_FLOOR_DROP,
	FLOOR_ARROW_DIST,
	FLOOR_ARROW_PITCH,
	LOCATE_SLICE_ABOVE_EYE,
	NAV_COLORS,
	PEEK_ROTATE_SPEED,
	WASD_DIR_COS,
	WASD_MAX_STEP,
	WASD_MAX_Y_STEP,
} from "./markers";
import { SurfaceCursor } from "./cursor";
import { LightRig } from "./lighting";
import { prepareLitScene } from "./prepare";
import { MarkerLayer } from "./markerLayer";
import { IDENTITY_TRANSFORM, SplatLayer, type SplatTransform } from "./splatLayer";
import { collectObjects, ObjectAddressing } from "./objectAddressing";
import { type PanoEntry, PanoStreamer } from "./panoTextures";
import { Projection } from "./projection";
import {
	buildMinimapState,
	levelForY,
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
	MapEdge,
	NodeDir,
	OrbitMode,
	OrbitState,
	TourManifest,
	TourSource,
} from "./types";

const v3 = (a: [number, number, number]) => new Vector3().fromArray(a);
const easeInOut = (t: number) =>
	t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;

// Duration constancy: a hop's length is felt through speed, not time — a 2 m and
// a 20 m walk both take the same beat. Phase is deliberately slower (a narrative
// "we are taking you through anyway"); far is a skippable flight.
const DUR: Record<EdgeType, number> = {
	walk: 700,
	portal: 900,
	phase: 1200,
	vertical: 1100,
	far: 2400,
};
const REDUCED_DUR = 260;
const DWELL_MS = 8500; // idle this long in a node → pulse the exits once

// One revolution at a tour's centrepoint. Slow is the point — it's a look around
// the room, not a spin. Under reduced motion it's slower still: discomfort tracks
// angular rate, so stretching the same turn is the gentler knob than cutting it.
const TOUR_PAN_MS = 10000;
const TOUR_PAN_MS_REDUCED = 15000;

// The two framings the viewer lives in. Every flight LERPS between them rather
// than snapping at the halfway mark, so the dollhouse framing opens continuously
// into the walkthrough's.
const OVERVIEW_FOV = 55;
const INTERIOR_FOV = 75;
// Once the fly-in has parked the camera exactly on the capture point, the pano
// dissolves in over the dollhouse across this long. Generous on purpose: both are
// the same view of the same room by then, so the only thing changing is shading
// and detail, and a slow blend reads as the room resolving rather than a cut.
const ENTER_CROSSFADE_MS = 450;
// Ceiling on how long the dissolve waits for the pano to stream in before going
// ahead anyway — the camera is parked, so waiting is invisible, but never hang.
const HANDOVER_WAIT_MS = 1000;

// --- free flight ------------------------------------------------------------
//
// Leaving the walkthrough for the splat is a DISSOLVE IN PLACE, not a flight: the
// camera is already exactly where it belongs, and the splat and the panorama are
// two renderings of the same room from the same point. So the departing pano is
// ramped off and the splat is simply already behind it — the same parked-dissolve
// reasoning `enter()` uses for the dollhouse handover. Short, because unlike that
// handover there is nothing to stream and waiting would just read as lag.
const SPLAT_REVEAL_MS = 320;
// Coming back IS a flight (you are in open space, the destination is a capture
// point), so it reuses the enter() path wholesale — arc, FOV opening, arrival
// crossfade and all. A touch quicker than entering from the dollhouse, since the
// distances involved are usually a room rather than a whole scene.
const FREEFLY_RETURN_MS = 1150;
// Where in the return flight the interior starts asserting itself, as a fraction of
// the move. Not a taste knob: before this the camera is still far enough from the
// anchor that its panorama projects onto the proxy badly stretched, so showing it
// early would trade one artefact for another. Landing the dissolve in the final
// stretch means it is only ever visible while it is nearly correct.
const DISSOLVE_START = 0.32;
// Flight speed as a fraction of the scene's largest dimension per second, so a
// cathedral and a bathroom both take a sensible time to cross.
const FREEFLY_SPEED_FRAC = 0.18;
const FREEFLY_SPRINT = 3;
// Velocity easing. Enough to take the edge off starting and stopping without
// feeling like ice — the movement should read as deliberate, not floaty.
const FREEFLY_ACCEL_TAU = 90; // ms
// What flies the camera once you are out there. Q/E are vertical here, where in
// the walkthrough they snap-turn — a rig pinned to an anchor has nowhere to rise
// to, and a rig in open space has no need to turn in 45° steps.
const FREEFLY_MOVE_KEYS = new Set([
	"KeyW",
	"KeyA",
	"KeyS",
	"KeyD",
	"KeyQ",
	"KeyE",
]);
// ...and which of them, pressed inside the walkthrough, mean "let me fly". Only
// the four horizontal ones: Q/E still belong to snap-turning until you have left.
const FREEFLY_ENTER_KEYS = new Set(["KeyW", "KeyA", "KeyS", "KeyD"]);

// There is deliberately NO default splat transform here.
//
// The pipeline hands Postshot a COLMAP model that is already in the repo-native
// world frame — stage5 locks it ("World: Y-up, right-handed, metres"), colmap.py
// writes `w2c = inv(c2w)` with no flip and copies cloud.ply's xyz verbatim — and
// Postshot trains from those exact poses and that exact point cloud. A splat built
// that way lands in world space, so the viewer's job is to draw it where it says it
// is, not to guess a correction.
//
// A fitted offset used to live here. It was wrong on three counts: it belonged to
// the ASSET rather than to one renderer, it was eyeballed from bounding boxes
// (which cannot even distinguish a translation from an axis flip), and it hid a
// defect in the handoff instead of fixing it. If a splat ever genuinely needs
// moving, the transform is computable in closed form against the point cloud it was
// initialized from and belongs baked into the file
// (tools/splat-to-web-sog.mjs --translate).

// Look up more steeply than this at something over your head and a click reads as
// "take me through it" — see the engine's targetFloorFor.
const CEILING_PITCH = (40 * Math.PI) / 180;

// ...and how close to your own feet the ground has to be for a click on it to read
// as "down through here" instead of "walk over there". Kept tight: this overrides
// the floor you are standing on, so it has to be somewhere you would never aim
// while picking a spot to walk to.
const UNDERFOOT_RADIUS = 1.6;

// Rest the cursor on one object this long and the walkthrough offers a proper look
// at it. Long enough that sweeping the room never triggers it, short enough to feel
// like an answer to a question you were already asking.
const INSPECT_DWELL_MS = 1750;
const INSPECT_SIZE = 190; // px — the inset's square edge
const INSPECT_GAP = 18; // px between the cursor and the inset
const INSPECT_MARGIN = 12; // px it keeps clear of the viewport edges
const INSPECT_SPIN = 0.55; // rad/s — a slow turn, not a spin

const _cursorNdc = new Vector2();
const _bez = new Vector3();
const _flyDir = new Vector3();
const _moveWish = new Vector3();
const _prevClear = new Color();
const _ghostFloor = new Vector3();
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
	// Dissolve into the destination instead of swapping mid-air behind a dip to
	// black. Only worth it when the flight lands somewhere both representations
	// agree on, i.e. a capture point. See tickCrossfade.
	crossfade: boolean;
	// Dissolve the interior in DURING the flight instead of parking at the end to do
	// it. Only legal when the two representations sit on DIFFERENT canvases — the
	// splat below, three.js above — because the dissolve is then a compositing
	// operation between two independently parallax-correct images. The dollhouse
	// paths cannot use it: their departure image is drawn by three.js too, so fading
	// that canvas would fade away the very thing being dissolved from.
	dissolveInterior: boolean;
	onMid?: () => void;
	onEnd?: () => void;
	midDone: boolean;
};

// A parked dissolve between the dollhouse and the capture pano. With the camera
// ON the capture point at the walkthrough's FOV, the equirect is the same view as
// the projected interior — so ramping it either way changes only shading/detail,
// never framing. "in" (enter) fades the pano over the dollhouse; "out" (exit)
// reveals the dollhouse before the fly-out, so the capture image never rides
// along as the camera pulls away.
type Crossfade = {
	armed: number; // when the ramp began; 0 while still waiting on the texture
	deadline: number; // stop waiting for the pano and hand over regardless
	dur: number;
	direction: "in" | "out";
	onEnd?: () => void;
};
// A typed interior traversal (one edge of the nav graph). `ctrl` bends the path
// (an arc for far/vertical hops); `sphere` crossfades the backdrop in sphere-only
// tours; `dy` is the height change, which names the arrival ("up/down a level").
// `pass` marks an anchor the auto tour is only walking through, which shortens the
// hop and skips the arrival narration.
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
// A destination the cursor can reach but the eye cannot see: the capture it lands
// on, the storey that capture is on, and how many storeys that is from here (0 =
// this floor, so the move is through geometry rather than up or down).
type ReachTarget = { index: number; level: number; levelDelta: number };

type SavedInterior = {
	pos: Vector3;
	lon: number;
	lat: number;
	index: number;
	fov: number;
};

// A combined dollhouse + interior walkthrough with a TYPED navigation grammar.
// OVERVIEW orbits the vertex-colored lite scene; stepping INSIDE drops into the
// pano walkthrough, where every reachable neighbour is classified into one of
// five edge types (walk / portal / vertical / phase / far), each with its own
// affordance and its own transition. All per-frame work mutates three.js / the
// canvas directly (never React), so the UI only re-renders on discrete changes.
export class OrbitEngine {
	private readonly host: HTMLElement;
	private readonly onState: (s: OrbitState) => void;
	private readonly onHold?: (held: boolean) => void;

	private readonly renderer: WebGLRenderer;
	private readonly canvas: HTMLCanvasElement;
	private readonly travelFade: HTMLDivElement;
	private readonly iris: HTMLDivElement; // vertical-shaft "hatch" wipe overlay
	private readonly sonarLabels: HTMLDivElement[] = []; // pooled x-ray name tags
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
	private readonly occluder = new Raycaster(); // LOS tests for the nav graph
	private pointerClientX = 0;
	private pointerClientY = 0;
	private pointerInside = false;

	private currentIndex = -1;
	// The capture the fly-in is heading to, projected during enter() before the
	// arrival is `activate`d (currentIndex is still -1 then). Cleared on arrival.
	private flyTarget = -1;
	private projectionMode = false;
	private minimaps: MinimapSlice[] = [];
	private mapLabels: MapLabel[] = [];
	private panoLevel: number[] = [];
	private minimapPrefetch: HTMLImageElement[] = [];
	private liteRoot: Group | null = null;
	private proxyGroup: Group | null = null;
	private sharedOverview = false;
	private proxyView = false;
	private proxyColorMats: Material[] = [];
	private connectors: Connector[] = []; // parsed but not surfaced (highlights hidden for now)

	// --- Gaussian splat ---
	// The splat renders on its OWN canvas beneath this one (see splatLayer.ts), so
	// nothing here shares a context with it — the engine only feeds it a camera and
	// decides when it is on screen. `splatView` is the user-facing switch; when it
	// is off every mesh view behaves exactly as it did before the splat existed.
	private readonly splat: SplatLayer;
	private splatView = true;
	// Where the splat sits in world space while its true placement is still being
	// established — see SplatLayer.setTransform. Baked away once confirmed.
	private splatTransform: SplatTransform = { ...IDENTITY_TRANSFORM };
	// The pano-to-splat dissolve: 0 = the walkthrough's panorama still covers the
	// view, 1 = the splat is fully uncovered. Runs alongside free flight rather
	// than blocking it, so movement responds from the first frame.
	private splatReveal = 0;
	private splatRevealing = false;
	// Held movement keys + the eased velocity they drive, in world units/sec.
	private readonly freeflyKeys = new Set<string>();
	private readonly freeflyVel = new Vector3();
	// --- dwell inspection ---
	// Which ids are discrete objects worth looking at (from the manifest), the
	// dollhouse copy of each (the only per-object geometry that is BOTH published
	// and coloured — the proxy is untextured and position-only), and the live inset.
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

	// The typed navigation graph (built at scene load) + per-scene directory data
	// the chrome reads for chapters / search / the minimap overlay.
	private navGraph: NavGraph | null = null;
	private nodeDir: NodeDir[] = [];
	private chapters: Chapter[] = [];
	private mapEdges: MapEdge[] = [];

	// Invariants: a back-stack that retraces the exact path (never blocked), the
	// set of nodes stood on (minimap fill), and a one-slot input buffer so
	// chained clicks queue instead of blocking.
	private history: number[] = [];
	private visited = new Set<number>();
	private pendingTravel: number | null = null;
	private arrival: OrbitState["arrival"] = null;

	private mode: OrbitMode = "empty";
	// Restored whenever the splat is not behind this layer; matches the host's CSS
	// backdrop so dropping to a transparent background never changes what you see.
	private readonly bgColor = new Color(0x0c0d10);
	private readonly sceneCenter = new Vector3();
	private sceneMaxDim = 1;
	private readonly browsePos = new Vector3();

	private transition: Transition | null = null;
	private move: Move | null = null;
	// The arrival dissolve a crossfading flight hands off to when it lands.
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
	private savedInterior: SavedInterior | null = null;
	private peekHeld = false;
	private readonly locateClip = new Plane(new Vector3(0, -1, 0), 0);

	private hoveredNavIndex = -1;
	// What the cursor currently REACHES: the destination a click would take you to
	// whenever you cannot see it from here — behind geometry, or on another floor.
	// Recomputed every frame in updateCursorRing; null whenever the destination is
	// in plain sight, which is the state that needs no explaining.
	private cursorReach: ReachTarget | null = null;
	// The floor arrow under the cursor. Its preview docks in the same corner as the
	// cursor's, so all this needs to carry is the destination.
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
	) {
		this.host = host;
		this.onState = onState;
		this.onHold = onHold;

		// `alpha` so this canvas can be drawn OVER the splat's: when the splat is
		// showing, the scene background is dropped and everything three.js renders
		// — markers, cursor, the pano dissolve — composites onto it. With no splat
		// the opaque background is restored and this is byte-for-byte the old path.
		this.renderer = new WebGLRenderer({ antialias: false, alpha: true });
		this.renderer.setPixelRatio(window.devicePixelRatio);
		// The display transform (ACES + sRGB + shadows) is owned by LightRig below.
		this.canvas = this.renderer.domElement;
		Object.assign(this.canvas.style, {
			display: "block",
			width: "100%",
			height: "100%",
			// Explicit stacking: the splat canvas is absolutely positioned, and a
			// positioned element would otherwise paint over this one whatever the
			// DOM order.
			position: "relative",
			zIndex: "1",
		});
		host.appendChild(this.canvas);
		this.splat = new SplatLayer(host);

		this.travelFade = document.createElement("div");
		Object.assign(this.travelFade.style, {
			position: "absolute",
			inset: "0",
			background: "#0e0f12",
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

		this.scene = new Scene();
		this.scene.background = this.bgColor;
		// Neutral IBL + hemisphere fill + a shadow-casting sun, on the same numbers
		// the panos were baked with (see lighting.ts) so the dollhouse and the
		// interior agree.
		this.rig = new LightRig(this.renderer, this.scene);

		this.camera = new PerspectiveCamera(60, 1, 0.05, 2000);
		this.camera.position.set(4, 3, 5);

		this.controls = new OrbitControls(this.camera, this.canvas);
		this.controls.enableDamping = true;
		this.controls.dampingFactor = 0.12;
		this.controls.screenSpacePanning = true;
		this.controls.zoomToCursor = true;
		this.controls.autoRotate = true;
		this.controls.autoRotateSpeed = 0.6;
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

		this.ro = new ResizeObserver(() => this.resize());
		this.ro.observe(host);
		this.resize();
		this.renderer.setAnimationLoop(this.tick);
		this.emit();
	}

	dispose() {
		this.disposed = true;
		this.loadToken++;
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

	// The two first-person modes. Both are a yaw/pitch rig driven by the same
	// drag-to-look and both resolve a click through the same surface raycast — they
	// differ only in whether the camera is pinned to a capture point.
	private get isLookMode(): boolean {
		return this.mode === "interior" || this.mode === "freefly";
	}

	// --- state emission (gated so chrome holds through camera flights) --------

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
			// The EFFECTIVE state, not the raw switch: with no splat loaded the
			// control has nothing to turn on and should not read as active.
			splatView: this.splatEnabled,
			canSplatView: this.canToggleSplatView(),
			splatTransform: this.splat.ready ? this.splatTransform : null,
			highlightEnabled: this.highlightEnabled,
			canHighlight:
				(this.mode === "overview" || this.mode === "interior") &&
				!!this.activeObjectRoot(),
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
			mapEdges: this.mapEdges,
			minimap: buildMinimapState({
				minimaps: this.minimaps,
				panos: this.panos,
				panoLevel: this.panoLevel,
				currentIndex: this.currentIndex,
				mode: this.mode,
				labels: this.mapLabels,
			}),
		};
		this.onState(state);
	}

	// The hover preview card payload, floated at the affordance's projected screen
	// point. Your heading carries across the hop, so the thumbnail is panned to the
	// direction you're facing RIGHT NOW — the card shows what you'll actually see
	// when you land, not some other view of the room.
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

	// The preview payload for an out-of-sight destination. The chrome pans it
	// through a full 360 rather than showing the equirect flat, so what you read is
	// the room and not a warped strip.
	// The out-of-sight destination to preview: a hovered floor waypoint if there is
	// one (it names an exact capture), else whatever the cursor currently reaches.
	//
	// Titled with that CAPTURE's own name. It used to carry a floor-wide name
	// instead, to stop the title churning as the auto-home target changed under the
	// cursor — but the panel now dissolves between destinations instead of cutting,
	// so it can track the actual destination and stay readable.
	private buildReachPreview(): OrbitState["reachPreview"] {
		// A hovered arrow names one exact destination, so it outranks the cursor's
		// rolling guess at what you are pointing past.
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

	// --- travel FX: the per-type transition "look" ----------------------------
	// A ground-glide blurs + dims (motion hides proxy warp). A phase tints the
	// screen blueprint-blue and runs slower — clearly synthetic, never pretending
	// the wall wasn't there. A vertical shaft irises through a hatch. Reduced-
	// motion collapses all of it to a quick dip.
	private setFx(type: EdgeType, t: number) {
		const m = Math.sin(Math.PI * MathUtils.clamp(t, 0, 1));
		if (this.reducedMotion) {
			this.canvas.style.filter = "none";
			this.travelFade.style.background = "#0e0f12";
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
		const tint =
			type === "phase"
				? "#0b2a44"
				: type === "far"
					? "#0a0c14"
					: "#0e0f12";
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
			// Close-then-open iris = passing up/down through a hatch.
			const gap =
				Math.abs(Math.cos(Math.PI * MathUtils.clamp(t, 0, 1))) * 130;
			this.iris.style.background = `radial-gradient(circle at 50% 50%, transparent ${gap.toFixed(1)}%, #05070d ${(gap + 7).toFixed(1)}%)`;
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

	private resize() {
		const w = this.host.clientWidth;
		const h = this.host.clientHeight;
		if (w === 0 || h === 0) return;
		this.renderer.setSize(w, h, false);
		this.composer.setSize(w, h);
		this.splat.resize(); // its canvas is a sibling, not a child — size it too
		this.camera.aspect = w / h;
		this.camera.updateProjectionMatrix();
	}

	// --- input handlers -------------------------------------------------------

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
		this.controls.autoRotate = false;
		if (this.autoRotateTimer) clearTimeout(this.autoRotateTimer);
	};
	private onControlsEnd = () => {
		if (this.mode !== "overview") return;
		if (this.autoRotateTimer) clearTimeout(this.autoRotateTimer);
		this.autoRotateTimer = setTimeout(() => {
			if (this.mode === "overview") this.controls.autoRotate = true;
		}, 2500);
	};

	private onPointerDown = (ev: PointerEvent) => {
		if (ev.button === 2) {
			this.rcDownX = ev.clientX;
			this.rcDownY = ev.clientY;
		}
		// Track the press origin in every mode so a click handler can tell a genuine
		// tap from the tail of a drag (overview enter-on-click; interior look-drag).
		this.downX = ev.clientX;
		this.downY = ev.clientY;
		if (!this.isLookMode) return;
		this.yieldTour(); // before the busy gate, so a click lands mid-hop too
		if (this.interiorBusy) return;
		this.noteInput();
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
			// The whole dollhouse is the "enter" affordance now: clicking anywhere on
			// it drops inside at the nearest capture point, so any hover over it reads
			// as clickable. Object highlight (when enabled) still layers on top.
			const overScene = !!this.raycastOverview(ev.clientX, ev.clientY);
			const obj = this.highlightEnabled
				? this.addressing.pickAt(
						ev.clientX,
						ev.clientY,
						this.activeObjectRoot(),
					)
				: null;
			this.canvas.style.cursor = overScene ? "pointer" : "";
			if (this.addressing.setHover(obj)) this.emit();
			return;
		}
		if (!this.isLookMode) return;
		if (this.dragging) {
			this.noteInput();
			const look = pinLook(
				this.camera,
				this.canvas,
				ev.clientX,
				ev.clientY,
				this.grabDir,
			);
			this.lon = look.lon;
			this.lat = look.lat;
			this.dragMoved = Math.max(
				this.dragMoved,
				Math.hypot(ev.clientX - this.downX, ev.clientY - this.downY),
			);
			// Drop any stale hover preview once the look starts moving.
			if (this.hoveredNavIndex !== -1) {
				this.hoveredNavIndex = -1;
				this.markers.setNavHover(null);
				this.emit();
			}
		} else if (this.mode === "interior" && !this.interiorBusy) {
			// Free flight has no standing affordances to hover; its cursor is
			// resolved per-frame in updateCursorRing instead.
			this.updateHover(ev);
		}
	};

	private onPointerUp = (ev: PointerEvent) => {
		if (!this.isLookMode) return;
		this.dragging = false;
		this.canvas.style.cursor = "";
		if (this.dragMoved >= 5) return;
		this.noteInput();
		if (this.mode === "freefly") {
			this.clickFromFreefly(ev.clientX, ev.clientY);
			return;
		}
		// A floor arrow takes the click first: it is the only thing under the cursor
		// that changes storey. Then an affordance traverses its edge; else
		// click-anywhere routing snaps to the node minimizing graph cost + angular
		// deviation.
		const arrow = this.markers.pickFloorArrow(
			ev.clientX,
			ev.clientY,
			this.camera,
			this.canvas,
		);
		if (arrow) {
			this.traverse(arrow.userData.to as number);
			return;
		}
		const spot = this.markers.pickNav(
			ev.clientX,
			ev.clientY,
			this.camera,
			this.canvas,
		);
		if (spot) {
			this.traverse(spot.userData.to as number);
			return;
		}
		this.clickAnywhere(ev.clientX, ev.clientY);
	};

	private onWheel = (ev: WheelEvent) => {
		if (!this.isLookMode) return;
		ev.preventDefault();
		this.camera.fov = MathUtils.clamp(
			this.camera.fov + ev.deltaY * 0.05,
			25,
			120,
		);
		this.camera.updateProjectionMatrix();
	};

	private onClick = (ev: MouseEvent) => {
		if (this.mode !== "overview") return;
		// Ignore the click that ends an orbit-drag; only a genuine tap enters.
		if (Math.hypot(ev.clientX - this.downX, ev.clientY - this.downY) > 6)
			return;
		const point = this.raycastOverview(ev.clientX, ev.clientY);
		if (point) this.enter(this.nearestPanoTo(point));
	};

	private onPointerLeave = () => {
		this.pointerInside = false;
		this.cursor.hide();
	};
	private onWindowPointerUp = () => this.peekUp();

	private onKeyDown = (ev: KeyboardEvent) => {
		if (ev.code === "Space" && !ev.repeat) {
			ev.preventDefault();
			this.peekDown();
			return;
		}
		if (this.mode === "freefly") {
			if (ev.code === "Escape") {
				ev.preventDefault();
				// Always a way back to the walkthrough, whatever the cursor is over.
				this.returnToInterior(this.nearestPanoTo(this.camera.position));
				return;
			}
			this.trackFreeflyKey(ev, true);
			return;
		}
		if (this.mode !== "interior") return;
		// Sonar ping (hold Tab): reveal every node through walls for a few seconds.
		if (ev.code === "Tab" && !ev.repeat) {
			ev.preventDefault();
			this.toggleSonar();
			return;
		}
		if (ev.code === "Escape") {
			this.yieldTour();
			if (this.markers.sonarActive) {
				this.markers.hideSonar();
				this.emit();
			}
			return;
		}
		if (this.interiorBusy || ev.repeat) return;
		if (ev.code === "Backspace") {
			ev.preventDefault();
			this.goBack();
			return;
		}
		if (ev.code === "KeyQ") {
			ev.preventDefault();
			this.snapTurn(-45);
			return;
		}
		if (ev.code === "KeyE") {
			ev.preventDefault();
			this.snapTurn(45);
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
		// A movement key is how you leave the walkthrough for the splat. With no
		// splat to fly through it keeps its original meaning below: step to the
		// neighbouring capture point along the look bearing.
		if (FREEFLY_ENTER_KEYS.has(ev.code) && this.canEnterFreefly()) {
			ev.preventDefault();
			this.freeflyKeys.add(ev.code);
			this.enterFreefly();
			return;
		}
		// WASD walks the graph edge nearest the look bearing.
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
		// Released in EVERY mode, not just free flight: a key held across the
		// transition out would otherwise stay latched and fly the camera on its own.
		this.trackFreeflyKey(ev, false);
	};

	// The held-key set behind free flight. Shift is folded to one name so the two
	// physical keys can't leave each other latched.
	private trackFreeflyKey(ev: KeyboardEvent, down: boolean) {
		const shift = ev.code === "ShiftLeft" || ev.code === "ShiftRight";
		if (!shift && !FREEFLY_MOVE_KEYS.has(ev.code)) return;
		const code = shift ? "Shift" : ev.code;
		if (!down) {
			this.freeflyKeys.delete(code);
			return;
		}
		ev.preventDefault();
		this.freeflyKeys.add(code);
		this.noteInput();
	}

	// Focus loss can't be seen as a key-up, so anything held becomes stuck down.
	private onWindowBlur = () => {
		this.freeflyKeys.clear();
		this.freeflyVel.set(0, 0, 0);
	};

	// Interior hover: light the affordance under the cursor + surface its preview.
	private updateHover(ev: PointerEvent) {
		if (this.currentIndex < 0) return;
		const arrow = this.markers.pickFloorArrow(
			ev.clientX,
			ev.clientY,
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
		const spot = this.markers.pickNav(
			ev.clientX,
			ev.clientY,
			this.camera,
			this.canvas,
		);
		this.markers.setNavHover(spot);
		const idx = spot ? (spot.userData.to as number) : -1;
		const obj = this.highlightEnabled
			? this.addressing.pickAt(
					ev.clientX,
					ev.clientY,
					this.activeObjectRoot(),
				)
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
			// Nothing to aim at here. If the ray leaves the scene entirely, this click
			// pulls back out to the orbit (see clickAnywhere) — advertise that instead
			// of letting it happen by surprise.
			this.canvas.style.cursor = this.raycastInterior(ev.clientX, ev.clientY)
				? ""
				: "zoom-out";
		}
		const changed = idx !== this.hoveredNavIndex;
		this.hoveredNavIndex = idx;
		if (this.addressing.setHover(obj) || changed) this.emit();
	}

	// Every visible interior surface under a screen point, nearest first. The
	// intersect call computes and sorts the whole list anyway, so handing it all
	// back costs nothing over returning just the first.
	private raycastInteriorAll(
		clientX: number,
		clientY: number,
	): Intersection[] {
		const targets: Object3D[] = [];
		if (this.projectionMode) {
			if (this.proxyGroup) targets.push(this.proxyGroup);
			if (this.projection.proxyBase)
				targets.push(this.projection.proxyBase);
		} else {
			this.sphereA.updateMatrixWorld();
			targets.push(this.sphereA);
		}
		if (targets.length === 0) return [];
		const rect = this.canvas.getBoundingClientRect();
		_cursorNdc.set(
			((clientX - rect.left) / rect.width) * 2 - 1,
			-((clientY - rect.top) / rect.height) * 2 + 1,
		);
		this.camera.updateMatrixWorld();
		this.cursorRay.setFromCamera(_cursorNdc, this.camera);
		// While the splat provides the picture, the proxy is hidden — but the cursor
		// still needs it, because it is the only thing in the scene that knows where
		// the surfaces are. The visibility walk therefore STOPS at the roots we hid
		// ourselves, which keeps an object the USER hid inside the proxy correctly
		// skipped: that check still runs, it just never reaches the group above it.
		// Keyed on the splat being ON SCREEN, not merely switched on: inside the
		// walkthrough the proxy is genuinely visible and carries the projection, and
		// exempting it there would quietly stop honouring a proxy the user hid.
		const forced = new Set<Object3D>();
		if (this.splat.isActive) {
			if (this.proxyGroup) forced.add(this.proxyGroup);
			if (this.projection.proxyBase) forced.add(this.projection.proxyBase);
		}
		return this.cursorRay.intersectObjects(targets, true).filter((h) => {
			for (let o: Object3D | null = h.object; o && !forced.has(o); o = o.parent)
				if (!o.visible) return false;
			return true;
		});
	}

	// Interior geometry under a screen point (proxy + floor base, or the sphere).
	private raycastInterior(
		clientX: number,
		clientY: number,
	): Intersection | null {
		return this.raycastInteriorAll(clientX, clientY)[0] ?? null;
	}

	// The dollhouse surface point under a screen pixel (overview): raycast the
	// visible scene root and return the first hit on a shown mesh, or null over
	// empty space. Enter-on-click homes to the nearest capture point to this.
	private raycastOverview(clientX: number, clientY: number): Vector3 | null {
		const root = this.activeObjectRoot();
		if (!root || !root.visible) return null;
		const rect = this.canvas.getBoundingClientRect();
		_cursorNdc.set(
			((clientX - rect.left) / rect.width) * 2 - 1,
			-((clientY - rect.top) / rect.height) * 2 + 1,
		);
		this.camera.updateMatrixWorld();
		this.cursorRay.setFromCamera(_cursorNdc, this.camera);
		for (const h of this.cursorRay.intersectObject(root, true)) {
			let visible = true;
			for (let o: Object3D | null = h.object; o; o = o.parent)
				if (!o.visible) {
					visible = false;
					break;
				}
			if (visible) return h.point.clone();
		}
		return null;
	}

	// Which storey a click on `hit` should go to.
	//
	// Normally that is simply the storey the geometry belongs to. The exception is
	// looking UP: a ceiling belongs to the room beneath it, so testing the hit point
	// alone answers "your own floor" and scopes the click back onto the storey you
	// are already standing on — which is exactly what stopped clicking the ceiling
	// from taking you upstairs. Pointing steeply above your own head is a request to
	// go through the thing over your head, so it resolves to the storey above.
	//
	// Decided on the ray's PITCH rather than the surface normal: proxy normals are
	// recomputed after decimation and the source winding is unreliable (the surface
	// cursor already flips them toward the camera for exactly that reason), so
	// "is this face pointing down at me" is not a question the geometry can answer.
	// How steeply you are looking is not in doubt.
	private targetFloorFor(hit: Intersection): number {
		if (!this.hasFloorVolumes) return -1;
		const cur = this.panoLevel[this.currentIndex] ?? -1;
		if (cur >= 0) {
			const plan = Math.hypot(
				hit.point.x - this.camera.position.x,
				hit.point.z - this.camera.position.z,
			);
			const pitch = Math.atan2(hit.point.y - this.camera.position.y, plan);
			if (pitch > CEILING_PITCH && this.panoLevel.includes(cur + 1))
				return cur + 1;
			// The mirror of that, and the reason it needs its own rule rather than
			// falling out of the geometry: looking DOWN, the thing you hit is the
			// floor you are standing on, which belongs to your own storey — so the
			// honest answer is "you are already here" and there is no way to ask for
			// the storey below by pointing.
			//
			// Aiming at the ground WITHIN ARM'S REACH of where you stand is that ask.
			// Radius rather than pitch does the work: past a step or two away, looking
			// down means walking over there, and only right at your feet does it mean
			// going through. (Inside this radius the pitch is steep anyway — the floor
			// sits ~1.3m below the eye, so the far edge is already past 40°.)
			if (
				hit.point.y < this.camera.position.y &&
				plan <= UNDERFOOT_RADIUS &&
				this.panoLevel.includes(cur - 1)
			)
				return cur - 1;
		}
		return this.floorAt(hit.point);
	}

	// Whether this tour's floors carry described volumes at all. Older captures
	// don't, and everything that reads a floor from geometry falls back to the
	// nearest-capture-point reading when they don't.
	private get hasFloorVolumes(): boolean {
		return this.minimaps.some((m) => !!m.volume);
	}

	// Which floor a world point is ON, by testing it against the floors' described
	// volumes (smallest wins where they overlap, so a mezzanine inside a taller
	// storey claims its own space). -1 means the point is on NO floor — terrain, a
	// cliff face, scenery, the slab between two storeys — which is a real answer,
	// not a failure: those things belong to no storey and the walkthrough must not
	// offer to travel to one on their behalf.
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

	// The pano that a floor click would snap to: the node minimizing (distance to
	// the hit point + angular deviation from the click bearing). Shared by
	// clickAnywhere (the actual traversal) and updateCursorRing (the live preview).
	//
	// `floor`, when given, SCOPES the search to anchors on that storey — the floor
	// the geometry under the cursor actually belongs to. Without it, clicking the
	// ground of the storey below through a gap could still resolve to an anchor on
	// your own floor (or vice versa) purely because it was closer in space, so the
	// preview and the destination could name different floors. A storey with no
	// eligible anchor of its own falls back to the whole set rather than making the
	// click do nothing.
	// `exclude` is the anchor a click may not resolve to — the one you are standing
	// on, since "travel to where you already are" is not an answer. Free flight
	// passes -1: you have left that anchor behind, and flying back to it is a
	// perfectly reasonable thing to ask for.
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
		// Straight up or straight down, the hit point sits on top of the eye in plan
		// and that bearing is atan2 of two near-zeroes — noise. Weighting it then
		// silently drags the choice toward whichever anchor happens to lie at bearing
		// zero, which is what made looking at the ceiling feel erratic. Fall back to
		// pure distance when there is no horizontal direction to read.
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
			const cost = d + (directional ? ang * 3 : 0); // 1 rad off ≈ 3 m of detour
			if (cost < bestCost) {
				bestCost = cost;
				best = i;
			}
		}
		return best < 0 && floor >= 0
			? this.autoHomeTarget(hit, -1, exclude)
			: best;
	}

	// The floor point of a destination capture — where its affordance is drawn, and
	// the same placement buildNav uses for a standing marker. The waypoint sits HERE,
	// on the anchor a click actually lands on, and the tether from the cursor to it
	// carries the "if I click here" part (see MarkerLayer.showGhost).
	//
	// It used to be drawn on the cursor's own bearing instead — keeping the pointer's
	// screen column and pinning only the height to the destination's floor — so that
	// it always answered "what happens if I click HERE". But that throws away
	// everything perpendicular to the bearing, and autoHomeTarget prices a radian of
	// deviation at only 3 m, so the marker routinely stood metres from the anchor it
	// promised (and, when the anchor fell behind the cursor plane, was clamped flat
	// against the obstruction instead — maximally wrong exactly where it mattered).
	// A marker that lies about its destination is worse than one you have to trace a
	// line to.
	private destinationFloor(targetIdx: number): Vector3 {
		// Scratch: showGhost copies this straight into the marker, never retains it.
		const p = this.panos[targetIdx].position;
		return _ghostFloor.set(p[0], p[1] - HOTSPOT_FLOOR_DROP, p[2]);
	}

	// Click-anywhere floor routing: raycast into the scene, then travel to the
	// auto-homed node — so the floor itself is the button. A click that reaches NO
	// geometry at all isn't aiming at a destination: it has gone out past the edge
	// of the scene into the void, which reads as "back out of here" — so it pulls
	// up to the orbit rather than quietly doing nothing.
	private clickAnywhere(clientX: number, clientY: number) {
		const hit = this.raycastInterior(clientX, clientY);
		if (!hit) {
			this.exit();
			return;
		}
		// Scoped by the floor the clicked geometry is on, exactly as the live preview
		// scopes it — so what the cursor promised is what the click delivers.
		const best = this.autoHomeTarget(hit, this.targetFloorFor(hit));
		if (best >= 0) this.traverse(best);
	}

	// --- view toggles (which geometry each mode shows) ------------------------

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

	// A splat is loaded AND switched on — i.e. it should stand in for the scene's
	// appearance wherever a mode chooses to show it. Not the same as being on
	// screen: the walkthrough turns it off regardless (see setInteriorView), so
	// anything asking "is it visible right now" wants `splat.isActive` instead.
	private get splatEnabled(): boolean {
		return this.splat.ready && this.splatView;
	}

	// Put the splat on or off screen. The background has to move with it: the splat
	// renders on the canvas BEHIND this one, so it is only visible while this layer
	// clears to transparent.
	private setSplatShowing(on: boolean) {
		this.splat.setActive(on);
		this.scene.background = on ? null : this.bgColor;
	}

	private setOverviewView() {
		// The splat IS the dollhouse when the cell has one. The lite mesh stays
		// loaded underneath it — it is the addressable geometry and the fallback —
		// but nothing renders it while the splat is doing that job.
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

	// Free flight: the splat carries the whole picture, and the proxy is hidden but
	// NOT removed — it is the only geometry that knows where the surfaces are, so
	// the cursor keeps raycasting it (see raycastInteriorAll) and a click still
	// resolves to a real place. With the splat switched off it renders as bare
	// polygons instead, which is also how you check the two are in register.
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
		// Inside the walkthrough the panoramas ARE the picture — they are a
		// higher-fidelity rendering of the same room than the splat is, and showing
		// both would just double-expose it. The splat context idles until free
		// flight or the overview asks for it again.
		this.setSplatShowing(false);
		if (this.liteRoot) this.liteRoot.visible = false;
		if (this.proxyGroup) this.proxyGroup.visible = this.projectionMode;
		if (this.projectionMode) {
			this.setInteriorProxyView();
		} else {
			this.sphereA.visible = true;
			// The backdrop rides the camera, so seed it here: the first interior frame
			// is drawn before the interior branch of the tick ever runs.
			this.sphereA.position.copy(this.camera.position);
		}
		this.markers.navGroup.visible = true;
		this.markers.arrowGroup.visible = true;
		this.markers.you.group.visible = false;
		this.projection.syncBase(!!this.proxyGroup?.visible);
	}

	private setPeekView() {
		// Hold-to-locate slices the roof off with a renderer clipping plane, which
		// belongs to three.js and means nothing to the splat's own context — a splat
		// here would keep its ceiling and bury the "you are here" pin under it. So
		// locating is always done on the mesh.
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
		// While the splat is ON SCREEN the mesh views are not, so the proxy/lite swap
		// has nothing to swap — turn the splat off first. Deliberately not keyed on
		// the splat merely being loaded: inside the walkthrough the splat is off
		// screen and the interior's own proxy/projection toggle must keep working.
		if (this.splat.isActive) return false;
		if (this.mode === "overview")
			return !!this.liteRoot && !!this.proxyGroup;
		if (this.mode === "interior") return this.projectionMode;
		return false;
	}

	// Only where the switch actually changes what is on screen. It deliberately
	// EXCLUDES the walkthrough: the panoramas are the picture in there and the splat
	// is off by design, so offering the control would put a live-looking button in
	// front of you that does nothing when pressed — which reads as the splat being
	// broken rather than as the control being inapplicable.
	private canToggleSplatView(): boolean {
		return (
			this.splat.ready &&
			(this.mode === "overview" || this.mode === "freefly")
		);
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

	// Swap between the Gaussian splat and the mesh views. Off is the escape hatch:
	// it restores the dollhouse/proxy exactly as they behaved before a splat
	// existed, which is both the fallback for a broken splat and the way to reach
	// the addressable per-object geometry.
	toggleSplatView() {
		if (!this.canToggleSplatView()) return;
		this.splatView = !this.splatView;
		this.addressing.setHover(null);
		this.addressing.closeMenu();
		this.canvas.style.cursor = "";
		if (this.mode === "overview") this.setOverviewView();
		else if (this.mode === "freefly") this.setFreeflyView();
		this.emit();
	}

	/**
	 * Move the splat in world space. This exists to settle a splat whose trainer
	 * renormalized the scene: nudge it until it registers against the proxy, then
	 * bake the numbers into the asset (tools/splat-to-web-sog.mjs --translate) and
	 * drop this back to identity. Deliberately not persisted — a correction that
	 * lives only in the viewer is one every other consumer gets wrong.
	 */
	setSplatTransform(patch: Partial<SplatTransform>) {
		this.splatTransform = { ...this.splatTransform, ...patch };
		this.splat.setTransform(this.splatTransform);
		this.emit();
	}

	getSplatTransform(): SplatTransform {
		return this.splatTransform;
	}

	toggleHighlight() {
		this.highlightEnabled = !this.highlightEnabled;
		if (!this.highlightEnabled) this.addressing.setHover(null);
		this.canvas.style.cursor = "";
		this.emit();
	}

	// --- per-object addressing ------------------------------------------------

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

	// --- projection -----------------------------------------------------------

	private updateProjection() {
		// The proxy has depth, so a projected capture parallaxes and two captures
		// cross-dissolve anchored to the same surface points — clean. The backdrop is
		// a depthless camera-centred sphere, so mid-glide the departure and
		// destination skyboxes land at slightly different angles in the void and smear
		// against each other ("old images leaking" where there's no geometry to pin
		// them). So hide it WHILE gliding: the parallax-correct proxy carries the move
		// and the void reads as clean background, then the backdrop returns — single
		// and exactly aligned — the moment we settle on the destination capture.
		this.sphereA.visible = !this.proxyView && !this.move;
		this.projection.project(
			this.panos,
			this.activeCaptures(),
			this.requestPano,
			this.sphereA,
			this.camera.position,
		);
	}

	// The capture(s) to project right now: the from/to pair while gliding a hop
	// (time-weighted so proxy + backdrop cross-dissolve together), else just the
	// capture you're standing at (an exact skybox — no offset ghosts). During the
	// overview→interior fly-in the arrival isn't `activate`d yet, so fall back to
	// the fly target. The walkthrough never free-roams, so this set is exact.
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
		// The fly target OUTRANKS where we currently stand. It is only ever set while
		// a fly-in is committed to an arrival, and it is the more specific answer:
		// stepping in from the dollhouse leaves `currentIndex` at -1 so either works,
		// but landing out of free flight leaves it pointing at the anchor we
		// DEPARTED — and reading that projects the wrong capture onto the proxy for
		// the frame between setInteriorView() and activate().
		if (this.flyTarget >= 0) return [[this.flyTarget, 1]];
		if (this.currentIndex >= 0) return [[this.currentIndex, 1]];
		return [];
	}

	// --- camera flight (mode changes) -----------------------------------------

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
		// A crossfading flight never dims, so clear any dip left by an earlier one.
		if (cbs.crossfade) this.travelFade.style.opacity = "0";
		this.mode = "transition";
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

	// --- typed traversal (interior) -------------------------------------------

	// Traverse to a node by its graph edge type. Chained clicks queue (input
	// buffering) instead of blocking; the back-stack is pushed unless retracing.
	private traverse(index: number, reverse = false, pass = false) {
		if (index === this.currentIndex || !this.panos[index]) return;
		if (this.interiorBusy) {
			this.pendingTravel = index; // latest click wins
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
		this.closeInspect();
		this.arrowReach = null;
		this.cursorReach = null;
		this.markers.setArrowHover(null);
		this.hoveredNavIndex = -1;
		this.markers.setNavHover(null);
		this.markers.navGroup.visible = false;
		this.markers.arrowGroup.visible = false;
		this.markers.hideSonar();
		// Push the cleared hover state out NOW. Nothing else emits between here and
		// the arrival at the far end, so without this the tooltip you just clicked
		// would sit there for the whole flight, previewing a place you are already on
		// your way to. The panel fades on its own once the state says it is gone.
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
		// Sphere-only tour: wait for a texture (placeholder is enough), then
		// crossfade the backdrop while the camera drifts onto the destination.
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

	// Bend the camera path: vertical shafts rise up-and-over; far flights pull
	// back toward a dollhouse vantage before pushing in. Walks stay straight.
	private pathControl(
		from: Vector3,
		to: Vector3,
		type: EdgeType,
	): Vector3 | null {
		if (type === "vertical") {
			const c = from.clone().add(to).multiplyScalar(0.5);
			c.y =
				Math.max(from.y, to.y) +
				Math.max(0.5, Math.abs(to.y - from.y) * 0.3);
			return c;
		}
		if (type === "far") {
			const c = from.clone().add(to).multiplyScalar(0.5);
			c.y += Math.min(this.sceneMaxDim * 0.6, from.distanceTo(to) * 0.45);
			return c;
		}
		return null;
	}

	// Bring the hop in flight to its end almost immediately, keeping the eased
	// position continuous: re-base the timeline so progress resumes from exactly
	// where it is and reaches 1 in `ms`. Used when the tour is stopped mid-hop —
	// the eye can't just freeze between two capture points (the projection, the
	// affordances and the exits all assume you're standing at one), so instead of
	// gliding on for up to another 2.4s it lands right away.
	private hurryMove(ms = 240) {
		const mv = this.move;
		if (!mv) return;
		const now = performance.now();
		const t = Math.min(1, (now - mv.start) / mv.dur);
		if (t > 0.95) return; // already landing — let it
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
		// Input buffering: run the most recent queued click as one journey.
		const next = this.pendingTravel;
		this.pendingTravel = null;
		if (next != null && next !== this.currentIndex) this.traverse(next);
	}

	// Land on a node. The look direction is deliberately left alone: heading
	// persistence across a hop is what keeps the mental map intact — you arrive
	// facing exactly where you were facing when you left, so the world appears to
	// slide past you rather than cutting to a new orientation. (The capture
	// `forward` is a fixed compass direction, not a per-edge "best view", so
	// snapping to it would just yank the camera back on every move.)
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

	// --- mode transitions -----------------------------------------------------

	private nearestPanoTo(point: Vector3): number {
		let best = 0;
		let bestD = Infinity;
		for (let i = 0; i < this.panos.length; i++) {
			const d = point.distanceToSquared(v3(this.panos[i].position));
			if (d < bestD) {
				bestD = d;
				best = i;
			}
		}
		return best;
	}

	private currentUserWorldPos(): Vector3 {
		return this.projectionMode
			? this.camera.position.clone()
			: v3(this.panos[this.currentIndex].position);
	}

	// Step inside. The flight carries the dollhouse's own framing all the way in —
	// same heading, same pitch, and a FOV that opens from the orbit framing to the
	// walkthrough's — and nothing is swapped mid-air. The scene only changes hands
	// once the camera is parked exactly on the capture point, where the dollhouse
	// render and the pano are the same view and can simply dissolve (tickCrossfade).
	enter(index: number | null = null) {
		if (this.mode !== "overview" || this.panos.length === 0) return;
		const idx = index ?? this.nearestPanoTo(this.controls.target);
		this.history = []; // a fresh interior session
		this.flyIntoInterior(idx, 1100);
	}

	// Fly from wherever the camera happens to be onto a capture point, then hand
	// over to the walkthrough. Shared by stepping in from the dollhouse and by
	// landing out of free flight, because they are the same journey: you are in
	// open space, and a capture point is where the walkthrough can take over.
	//
	// The walkthrough is a yaw/pitch rig, so the live look direction is read back
	// as lon/lat and the pitch pre-clamped to what applyLook enforces. The pose the
	// flight lands on is then exactly the pose the rig holds afterwards, so the
	// handover doesn't snap the view a single degree — and because the heading
	// carries across, the room you were looking at is the room you arrive facing.
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
		this.flyTarget = idx; // project the arrival during the fly-in (pre-activate)
		this.startFly(toPos, lookTargetFrom(toPos, lon, lat), dur, {
			toFov: INTERIOR_FOV,
			// One or the other: dissolve DURING the move, or park at the end and
			// crossfade there. Never both — they are two answers to the same handover
			// and would fight over the same panorama.
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

	// --- free flight ----------------------------------------------------------

	private canEnterFreefly(): boolean {
		// The proxy is the requirement, not a nicety: without it the cursor has
		// nothing to raycast, and a mode you can fly into but not click your way
		// out of is a trap.
		return this.splatEnabled && this.projectionMode && !this.interiorBusy;
	}

	// Leave the walkthrough for the splat WITHOUT moving. The camera already stands
	// exactly where the splat says the room is, so the two renderings agree at this
	// pose and the handover is a dissolve rather than a transition — the same
	// reasoning enter() uses to hand the dollhouse over to a panorama.
	//
	// Movement is live from the first frame: the ramp is cosmetic and never gates
	// input, so the keypress that asked for free flight is already moving you.
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

		const tex =
			this.currentIndex >= 0 ? this.panos[this.currentIndex]?.texture : null;
		this.mode = "freefly";
		this.setFreeflyView();
		if (tex && !this.reducedMotion) {
			// Stage the panorama we are standing in as a camera-locked overlay at
			// full strength; the tick ramps it away to uncover the splat behind.
			this.sphereBMat.uniforms.map.value = tex;
			this.sphereBMat.uniforms.opacity.value = 1;
			this.sphereBMat.depthTest = false;
			this.sphereB.renderOrder = 20;
			this.sphereB.visible = true;
			this.sphereB.position.copy(this.camera.position);
			this.splatReveal = 0;
			this.splatRevealing = true;
		} else {
			this.clearPanoOverlay();
			this.splatReveal = 1;
			this.splatRevealing = false;
		}
		this.noteInput();
		this.emit();
	}

	// Land out of free flight onto a capture point and give the walkthrough back.
	//
	// The interior is brought up BEFORE the flight and dissolved into during it, so
	// the move and the handover are one gesture rather than a glide followed by a
	// swap. That is only possible here: the departure image is the splat, on its own
	// canvas, so the two can be composited while both are moving. Stepping in from
	// the dollhouse cannot do this — see the note on Transition.dissolveInterior.
	//
	// It needs the destination panorama ALREADY resident, because the projection
	// shader renders black with no texture bound and dissolving into black is worse
	// than the old behaviour. The cursor pre-warms it while you aim, so this is the
	// normal case; when it isn't ready we fall back to the parked crossfade, which
	// knows how to wait.
	private returnToInterior(index: number) {
		if (this.mode !== "freefly" || !this.panos[index]) return;
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
		// Stage the interior invisible, then let the tick ramp it up over the flight.
		this.reskinProxy(this.projection.material);
		if (this.proxyGroup) this.proxyGroup.visible = true;
		this.projection.syncBase(true);
		this.canvas.style.opacity = "0";
		this.flyIntoInterior(index, FREEFLY_RETURN_MS, { dissolve: true });
	}

	// A click in free flight means what a click means everywhere else here: take me
	// there. It runs the walkthrough's own resolution — the surface under the
	// cursor, the floor that surface belongs to, the anchor that best answers it —
	// so both modes agree about where a click lands. The floor comes straight from
	// the geometry rather than from targetFloorFor's look-up/look-down heuristics,
	// which are about a visitor rooted at one anchor and mean nothing in flight.
	//
	// Nothing under the cursor (aimed past the scene) falls back to the nearest
	// capture, so a click always has somewhere to put you.
	private clickFromFreefly(clientX: number, clientY: number) {
		const hit = this.raycastInterior(clientX, clientY);
		const best = hit
			? this.autoHomeTarget(hit, this.floorAt(hit.point), -1)
			: this.nearestPanoTo(this.camera.position);
		if (best >= 0) this.returnToInterior(best);
	}

	// Step back out. The capture image is dissolved away WHILE still parked at the
	// capture point — same pose, same FOV — so the dollhouse is the only thing left
	// when the fly-out begins. Flying with the pano still glued on was the
	// duplicated-room look: the equirect (and the projected proxy) rode along as
	// the camera pulled away, then the dollhouse appeared underneath it.
	exit() {
		if (this.mode !== "interior" || this.interiorBusy) return;
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
		// Dollhouse underneath; keep the capture as a camera-locked overlay so we
		// can ramp it out without the projected proxy stretching as we leave.
		this.setOverviewView();
		this.mode = "transition";
		this.controls.enabled = false;
		this.controls.autoRotate = false;
		this.emit();

		const flyOut = () => {
			this.clearPanoOverlay();
			this.startFly(
				this.browsePos.clone(),
				this.sceneCenter.clone(),
				1000,
				{
					toFov: OVERVIEW_FOV,
					onEnd: () => {
						this.mode = "overview";
						this.controls.target.copy(this.sceneCenter);
						this.camera.position.copy(this.browsePos);
						this.controls.enabled = true;
						this.controls.update();
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

	// Travel to any node from the chrome (minimap / chapters / search): typed if
	// it's a direct edge, else a far flight.
	traverseTo(index: number) {
		if (this.mode !== "interior" || !this.panos[index]) return;
		this.yieldTour();
		this.traverse(index);
	}

	// Retrace the back-stack one hop (never blocked); empty stack → out to overview.
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

	snapTurn(deg: number) {
		if (this.mode !== "interior") return;
		this.yieldTour();
		this.noteInput();
		this.lon += (deg * Math.PI) / 180;
	}

	// Start / stop the zone-by-zone auto tour. Stopping leaves the camera exactly
	// where it is — the itinerary is simply dropped.
	toggleTour() {
		if (this.mode !== "interior") return;
		if (this.director.active) {
			this.yieldTour();
			return;
		}
		if (!this.navGraph || this.currentIndex < 0) return;
		this.director.start(
			planZoneTour(
				this.navGraph,
				(i) => this.panos[i]?.zone ?? "",
				this.currentIndex,
			),
		);
	}

	// Any deliberate navigation hands the view back: the tour writes the look
	// angles every frame while sweeping, so it has to let go the moment the user
	// takes over rather than fight them for the camera. Stopping mid-sweep is
	// instant and leaves the camera untouched; stopping mid-hop can only let go
	// once the hop has landed, so hurry that landing along.
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

	// Jump to a floor level (number keys / floor chips): nearest node on it.
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
		return (this.lon * 180) / Math.PI;
	}

	// WASD: nearest graph neighbour inside a forward cone, one floor only.
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
			if (Math.abs(p[1] - cur[1]) > WASD_MAX_Y_STEP) continue;
			const dx = p[0] - cur[0];
			const dz = p[2] - cur[2];
			const dist2 = dx * dx + dz * dz;
			if (dist2 < 1e-6 || dist2 > WASD_MAX_STEP * WASD_MAX_STEP) continue;
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
		this.savedInterior = {
			pos: this.camera.position.clone(),
			lon: this.lon,
			lat: this.lat,
			index: this.currentIndex,
			fov: this.camera.fov,
		};
		const userPos = this.currentUserWorldPos();
		this.markers.positionYouMarker(userPos);
		this.locateClip.constant = userPos.y + LOCATE_SLICE_ABOVE_EYE;
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
		if (this.mode === "peek") this.peekEnd();
	}

	// --- cell loading ---------------------------------------------------------

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
		this.loadToken++;
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
		this.mapEdges = [];
		this.history = [];
		this.visited.clear();
		this.pendingTravel = null;
		this.arrival = null;
		this.minimaps = [];
		this.mapLabels = [];
		this.panoLevel = [];
		this.minimapPrefetch = [];
		this.sphereAMat.uniforms.map.value = DUMMY_TEX;
		this.sphereBMat.uniforms.map.value = DUMMY_TEX;
		// A projection tour skins the backdrop with the VDTM material and scales it to
		// the scene — hand it back to the plain equirect material so a sphere-only tour
		// loaded next still renders its pano.
		this.sphereA.material = this.sphereAMat;
		this.sphereA.scale.setScalar(1);
		this.currentIndex = -1;
		this.flyTarget = -1;
		this.markers.clear();
		this.sphereA.visible = false;
		this.sphereB.visible = false;
		this.transition = null;
		this.move = null;
		this.crossfade = null; // a dissolve queued for a scene that's now gone
		this.clearPanoOverlay();
		this.interiorBusy = false;
		this.peekHeld = false;
		this.savedInterior = null;
		this.renderer.clippingPlanes = [];
		this.hoveredNavIndex = -1;
		this.cursorReach = null;
		this.proxyView = false;
		this.splat.clear();
		this.splatView = true; // a scene that ships a splat leads with it
		this.splatTransform = { ...IDENTITY_TRANSFORM };
		this.splatReveal = 0;
		this.splatRevealing = false;
		this.freeflyKeys.clear();
		this.freeflyVel.set(0, 0, 0);
		this.scene.background = this.bgColor;
		// A scene swapped in mid-dissolve would otherwise inherit a part-transparent
		// canvas and render washed out for the rest of the session.
		this.canvas.style.opacity = "1";
		this.addressing.reset();
		this.canvas.style.cursor = "";
		this.clearFx();
		for (const l of this.sonarLabels) l.style.display = "none";
	}

	async loadTour(source: TourSource) {
		this.mode = "loading";
		this.controls.enabled = false;
		this.clearScene();
		const token = this.loadToken;
		this.showOverlay("loading scene…");

		try {
			let manifest: TourManifest | null = null;
			if (source.manifestUrl) {
				// no-store, always: tour.json is rewritten in place by a
				// re-publish or a metadata backfill, so a cached copy silently
				// serves a scene that is missing whatever was just added to it.
				const res = await fetch(source.manifestUrl, { cache: "no-store" });
				if (token !== this.loadToken || this.disposed) return;
				if (res.ok) manifest = (await res.json()) as TourManifest;
			}
			if (token !== this.loadToken || this.disposed) return;

			const mmList =
				manifest && Array.isArray(manifest.minimaps)
					? manifest.minimaps
					: [];
			this.minimaps = mmList.map((m) => ({
				...m,
				url: source.resolveMinimap(m.file),
			}));
			this.minimapPrefetch = this.minimaps.map((m) => {
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
				manifest && Array.isArray(manifest.objects) ? manifest.objects : [];
			this.mapLabels =
				manifest && Array.isArray(manifest.map_labels)
					? manifest.map_labels
					: [];

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
			// Loaded BEFORE the first frame rather than popped in afterwards: the
			// splat IS the scene's appearance when it has one, and showing the
			// dollhouse only to swap it out a second later reads as a glitch.
			// Failure is non-fatal — the dollhouse and the walkthrough are each
			// complete without it, so a broken splat costs the feature, not the scene.
			// Drawn exactly where the file says it is — see the note by
			// IDENTITY_TRANSFORM for why no correction is applied here.
			if (source.splatUrl) await this.splat.load(source.splatUrl);
			if (token !== this.loadToken || this.disposed) return;
			this.applyScene(entries, proxyRoot, lite, connectors, objectIds);
		} catch (e) {
			if (token !== this.loadToken || this.disposed) return;
			this.mode = "empty";
			this.showOverlay(
				`failed to load scene: ${e instanceof Error ? e.message : String(e)}`,
				{ spinner: false, err: true },
			);
		}
	}

	private applyScene(
		entries: PanoEntry[],
		proxyRoot: Group | null,
		lite: Group | null,
		connectors: Connector[],
		objectIds: string[] = [],
	) {
		this.connectors = connectors;
		this.inspectable = new Set(objectIds);
		this.streamer.reset(entries);
		// Which storey each capture stands on. Taken from the manifest when it says
		// — the floor planner decided the split and assigned every capture to one,
		// so re-deriving it here could only disagree, and a capture near a boundary
		// is exactly where it would. Matching the nearest slice by height is the
		// fallback for tours captured before the split was planned.
		this.panoLevel = entries.map((p) =>
			typeof p.level === "number" &&
			p.level >= 0 &&
			p.level < this.minimaps.length
				? p.level
				: levelForY(this.minimaps, p.position[1]),
		);
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

		// Make both roots shadeable BEFORE anything re-skins them: generate the
		// missing normals (without which the standard material shades to black),
		// force the matte splat-tier look, and put falsely-BLEND opaque geometry
		// back in the opaque queue so depth — not centroid sorting — decides what
		// occludes what. See prepare.ts.
		if (lite) {
			prepareLitScene(lite);
			this.liteRoot = lite;
			this.scene.add(lite);
		}
		if (proxyRoot) {
			prepareLitScene(proxyRoot);
			this.projection.setup(proxyRoot, this.sphereA);
			this.proxyGroup = proxyRoot;
			this.scene.add(proxyRoot);
		}

		// Object addressing on both roots. Connector highlights are intentionally
		// NOT pinned (hidden for now) — travel is driven entirely by the nav graph.
		if (this.liteRoot) {
			this.addressing.register(this.liteRoot);
			// The hovered object is a PROXY node, but the proxy is untextured
			// geometry; the dollhouse is the only published per-object mesh carrying
			// colour. Both name their nodes with the same pipeline id, which is what
			// lets one stand in for the other.
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

		const framed = lite ?? proxyRoot!;
		const box = new Box3().setFromObject(framed);
		const size = box.getSize(new Vector3());
		box.getCenter(this.sceneCenter);
		this.sceneMaxDim = Math.max(size.x, size.y, size.z) || 1;
		this.rig.fit(box); // spend the shadow frustum's precision on this scene

		this.camera.near = Math.max(0.02, this.sceneMaxDim * 0.002);
		this.camera.far = Math.max(500, this.sceneMaxDim * 60);

		// Build the typed navigation graph now that geometry + panos are placed.
		this.navGraph = buildNavGraph(
			entries.map((p) => ({ position: p.position, zone: p.zone })),
			this.panoLevel,
			(a, b) => this.segmentBlocked(a, b),
		);
		this.buildSceneDirectory(entries);

		this.markers.build(this.sceneMaxDim);

		const dist = this.sceneMaxDim * 1.6;
		this.browsePos
			.copy(this.sceneCenter)
			.add(new Vector3(dist * 0.7, dist * 0.5, dist * 0.9));
		this.camera.position.copy(this.browsePos);
		this.camera.fov = OVERVIEW_FOV;
		this.camera.lookAt(this.sceneCenter);
		this.camera.updateProjectionMatrix();
		this.controls.target.copy(this.sceneCenter);
		this.controls.enabled = true;
		this.controls.update();
		this.controls.autoRotate = true;

		this.setOverviewView();
		this.mode = "overview";
		this.hideOverlay();
	}

	// Line-of-sight test for the nav graph: is the straight segment between two
	// capture points blocked by the proxy? (No proxy → nothing occludes, so every
	// same-level pair reads as a clear walk.) Trimmed at both ends so hugging a
	// wall doesn't read as a block.
	private segmentBlocked(
		a: [number, number, number],
		b: [number, number, number],
	): boolean {
		if (!this.proxyGroup) return false;
		const from = v3(a);
		const d = v3(b).sub(from);
		const dist = d.length();
		if (dist < 1e-3) return false;
		d.divideScalar(dist);
		this.occluder.set(from, d);
		this.occluder.near = 0.2;
		this.occluder.far = dist - 0.2;
		if (this.occluder.far <= this.occluder.near) return false;
		return this.occluder.intersectObject(this.proxyGroup, true).length > 0;
	}

	// Live LOS from the camera to a target pano position. Shoots one direct ray
	// plus four slightly-offset origin rays (a small cross in the XZ plane) so a
	// ray that barely clips a wall corner doesn't mis-report the whole path as
	// blocked. Returns true when ANY ray reaches the target unobstructed.
	private isTargetClear(target: Vector3): boolean {
		if (!this.proxyGroup) return true;
		const cx = this.camera.position.x;
		const cy = this.camera.position.y;
		const cz = this.camera.position.z;
		for (const [ox, oz] of [
			[0, 0],
			[0.2, 0],
			[-0.2, 0],
			[0, 0.2],
			[0, -0.2],
		] as const) {
			_losFrom.set(cx + ox, cy, cz + oz);
			_losDir.copy(target).sub(_losFrom);
			const dist = _losDir.length();
			if (dist < 0.5) return true;
			_losDir.divideScalar(dist);
			this.occluder.set(_losFrom, _losDir);
			this.occluder.near = 0.15;
			this.occluder.far = dist - 0.15;
			if (this.occluder.far <= this.occluder.near) continue;
			if (
				this.occluder.intersectObject(this.proxyGroup, true).length ===
				0
			)
				return true;
		}
		return false;
	}

	// Stable per-scene directory + zone chapters + undirected map edges (for the
	// minimap overlay, chapters drawer, and "take me to" search).
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
		const seen = new Set<string>();
		const edges: MapEdge[] = [];
		if (this.navGraph) {
			for (const node of this.navGraph.nodes) {
				for (const e of node.all) {
					if (e.type === "far") continue;
					const a = Math.min(node.index, e.to);
					const b = Math.max(node.index, e.to);
					const key = `${a}-${b}`;
					if (seen.has(key)) continue;
					seen.add(key);
					edges.push({ a, b, type: e.type });
				}
			}
		}
		this.mapEdges = edges;
	}

	// --- dwell inspection ------------------------------------------------------

	// Watch how long the cursor has rested on one object. Tracked here rather than
	// in the pointer handler because dwelling is precisely the absence of pointer
	// events — the hover is already resolved, what we are timing is the stillness.
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

	// Build the inset: a clone of the dollhouse copy of this object, recentred on a
	// pivot so it turns about itself, framed by its own bounding sphere, lit by a
	// small rig of its own. The clone shares geometry and materials with the scene
	// copy — only the transform is ours — so opening one costs no upload.
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
		const radius = Math.max(1e-3, box.getSize(new Vector3()).length() * 0.5);
		clone.position.sub(centre);
		const pivot = new Group();
		pivot.add(clone);
		scene.add(pivot);
		// Pull back far enough that the whole silhouette fits at every angle of the
		// turn — the bounding SPHERE, not the box, since the box's footprint changes
		// as it rotates and the object must never clip its own frame.
		this.inspectCam.position.set(0.62, 0.42, 1).normalize().multiplyScalar(
			(radius / Math.sin((this.inspectCam.fov * Math.PI) / 360)) * 1.12,
		);
		this.inspectCam.near = Math.max(0.01, radius * 0.05);
		this.inspectCam.far = radius * 20;
		this.inspectCam.lookAt(0, 0, 0);
		this.inspectCam.updateProjectionMatrix();
		this.inspectScene = scene;
		this.inspectPivot = pivot;
		const rect = this.canvas.getBoundingClientRect();
		const m = INSPECT_MARGIN;
		this.inspect = {
			label,
			x: Math.min(
				Math.max(this.pointerClientX + INSPECT_GAP, rect.left + m),
				rect.right - INSPECT_SIZE - m,
			),
			y: Math.min(
				Math.max(this.pointerClientY - INSPECT_SIZE - INSPECT_GAP, rect.top + m),
				rect.bottom - INSPECT_SIZE - m,
			),
			w: INSPECT_SIZE,
			h: INSPECT_SIZE,
		};
		this.emit();
	}

	private closeInspect() {
		if (!this.inspect) return;
		// Geometry and materials belong to the dollhouse; only the wrapper is ours.
		this.inspectScene = null;
		this.inspectPivot = null;
		this.inspect = null;
		this.emit();
	}

	// Draw the inset into its own rectangle of the main canvas, AFTER the composer
	// has presented the frame. A scissored viewport rather than a second canvas: a
	// third and fourth WebGL context (the workspace runs two engines side by side)
	// to spin one small object is not a trade worth making.
	private renderInspect(dt: number) {
		const ins = this.inspect;
		if (!ins || !this.inspectScene || !this.inspectPivot) return;
		this.inspectPivot.rotation.y += dt * INSPECT_SPIN;
		const rect = this.canvas.getBoundingClientRect();
		const x = ins.x - rect.left;
		// GL's viewport origin is bottom-left; the rect is measured from the top.
		const y = rect.height - (ins.y - rect.top) - ins.h;
		const prevAutoClear = this.renderer.autoClear;
		// The clear state has to be PUT BACK, not just overwritten. The inset needs
		// an opaque backdrop of its own, but the main pass clears to transparent so
		// the splat layer behind this canvas can show through — and leaving alpha at
		// 1 here turns the whole canvas opaque for every subsequent frame, hiding the
		// splat behind a wall of flat colour while the markers drawn on top of it
		// carry on working. That reads as "the splat stopped loading" and survives
		// until reload, which is exactly as confusing as it sounds.
		this.renderer.getClearColor(_prevClear);
		const prevClearAlpha = this.renderer.getClearAlpha();
		this.renderer.autoClear = false;
		this.renderer.setScissorTest(true);
		this.renderer.setViewport(x, y, ins.w, ins.h);
		this.renderer.setScissor(x, y, ins.w, ins.h);
		this.renderer.setClearColor(0x0b0d12, 1);
		this.renderer.clear(true, true, false);
		this.renderer.render(this.inspectScene, this.inspectCam);
		this.renderer.setScissorTest(false);
		this.renderer.setViewport(0, 0, rect.width, rect.height);
		this.renderer.setClearColor(_prevClear, prevClearAlpha);
		this.renderer.autoClear = prevAutoClear;
	}

	// --- floor arrows ----------------------------------------------------------

	// Place the floor arrows for the node just arrived at, ON THE HEADING YOU
	// ARRIVED FACING — one ahead and above for the storey up, one ahead and below
	// for the storey down.
	//
	// Every earlier version put this marker at the destination, which is the one
	// place you are guaranteed not to be looking: it is on another floor, behind
	// the ceiling or under your feet. So the arrow had to be hunted for, which is
	// the opposite of what a way out should ask of you. Putting it on the arrival
	// heading means it is simply in front of you when you land — and since heading
	// carries across a hop, "the way you arrived facing" is exactly where your
	// attention already is.
	//
	// Placed once per arrival, in world space: turning around leaves it behind you,
	// where it belongs, rather than dragging it along like a HUD element. Clicking
	// one snaps to the nearest capture on that storey.
	private refreshFloorArrows() {
		const cur =
			this.currentIndex >= 0 ? this.panoLevel[this.currentIndex] : -1;
		if (cur < 0) {
			this.markers.clearFloorArrows();
			return;
		}
		const here = v3(this.panos[this.currentIndex].position);
		const ahead = new Vector3(Math.cos(this.lon), 0, Math.sin(this.lon));
		// Out toward the top / bottom of the frame, from the angle rather than a
		// hard-coded height, so the two stay put if the distance is ever retuned.
		const rise = FLOOR_ARROW_DIST * Math.tan(FLOOR_ARROW_PITCH);
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
				pos: here
					.clone()
					.addScaledVector(ahead, FLOOR_ARROW_DIST)
					.setY(here.y + step * rise),
			});
		}
		this.markers.buildFloorArrows(items);
		this.markers.arrowGroup.visible = this.mode === "interior";
	}

	// --- render loop ----------------------------------------------------------

	private tick = (time: number) => {
		const now = performance.now();
		const dt = this.lastFrame
			? Math.min(0.05, (time - this.lastFrame) / 1000)
			: 0;
		this.lastFrame = time;

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
			// Dissolve the interior in WHILE moving. Both layers parallax correctly —
			// the splat on its own canvas, the panorama projected onto the proxy on
			// this one — so there is nothing to smear and no reason to stop first.
			//
			// The weight ramps LATE on purpose. A capture projected from far off its
			// own vantage is badly stretched, and that error shrinks to nothing as the
			// camera converges on the anchor. So the splat carries the opening of the
			// move and the interior asserts itself exactly as it becomes correct: the
			// dissolve is scheduled by fidelity, not by the clock.
			if (tr.dissolveInterior) {
				this.updateProjection();
				const d = MathUtils.clamp(
					(t - DISSOLVE_START) / (1 - DISSOLVE_START),
					0,
					1,
				);
				this.canvas.style.opacity = easeInOut(d).toFixed(3);
			}
			// A crossfading flight stays fully visible the whole way in — there is
			// nothing to hide, because the swap happens at the far end where the two
			// renders already agree.
			// A dissolving flight must not dip either: the whole point is that the
			// picture never goes away, it only changes hands.
			if (!tr.crossfade && !tr.dissolveInterior)
				this.travelFade.style.opacity = (
					Math.sin(Math.PI * t) * 0.5
				).toFixed(3);
			if (!tr.midDone && t >= 0.5) {
				tr.midDone = true;
				tr.onMid?.();
			}
			// Never project during a flight, EXCEPT a dissolving one (handled above).
			// The enter path is still on the dollhouse and the exit path has already
			// dissolved the capture away, so projecting on either would re-glue the
			// pano to the proxy and ride it out with the camera — the duplicated-room
			// look. A dissolving flight is the one case where projecting is the point:
			// it lands on a capture point and its departure image is a different canvas.
			if (t >= 1) {
				const cb = tr.onEnd;
				const crossfade = tr.crossfade;
				this.transition = null;
				if (crossfade) {
					// Landed on the capture point. Park and dissolve rather than cut.
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
					// Defensive: a dissolve leaves this mid-ramp, and a canvas stuck
					// part-transparent would quietly wash out every later frame.
					this.canvas.style.opacity = "1";
					cb?.();
				}
			}
		} else if (this.crossfade) {
			this.tickCrossfade(now);
		} else if (this.mode === "overview") {
			this.controls.update();
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
			// The tour drives the same yaw/pitch drag-look writes, so it has to run
			// before the look is applied.
			this.director.tick(now);
			this.lat = applyLook(this.camera, this.lon, this.lat);
			if (!this.interiorBusy) {
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
					if (!this.markers.sonarActive) this.emit(); // just expired
				} else if (
					this.sonarLabels.some((l) => l.style.display !== "none")
				) {
					for (const l of this.sonarLabels) l.style.display = "none";
				}
				this.tickInspect(now);
				// Never let stillness become stuckness: pulse the exits once on dwell.
				if (!this.dwellPulsed && now - this.lastInputAt > DWELL_MS) {
					this.dwellPulsed = true;
					this.markers.pulseExits(now, 1600);
				}
			}
		} else if (this.mode === "freefly") {
			this.tickFreefly(dt);
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
		// The splat draws FIRST, from the camera this frame just settled on, so the
		// two canvases present the same pose. Anything three.js puts on top — the
		// cursor, a waypoint, the dissolving panorama — is then glued to it rather
		// than trailing it by a frame. A no-op whenever the splat is off screen.
		this.splat.render(this.camera);
		this.composer.render();
		this.renderInspect(dt);
	};

	// One frame of free flight. Velocity EASES toward what the held keys ask for
	// rather than snapping to it, which is most of what separates flying from
	// teleporting; the ramp is short enough to still feel deliberate.
	private tickFreefly(dt: number) {
		const cl = Math.cos(this.lat);
		const fx = cl * Math.cos(this.lon);
		const fy = Math.sin(this.lat);
		const fz = cl * Math.sin(this.lon);
		const rx = -Math.sin(this.lon);
		const rz = Math.cos(this.lon);
		const keys = this.freeflyKeys;
		_moveWish.set(0, 0, 0);
		// W/S fly along the FULL look direction, pitch included — looking up and
		// pressing forward should climb, which is the difference between flying and
		// walking. Q/E stay on world up, so you can rise without changing where you
		// are looking.
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
		if (keys.has("KeyE")) _moveWish.y += 1;
		if (keys.has("KeyQ")) _moveWish.y -= 1;
		if (_moveWish.lengthSq() > 0) _moveWish.normalize();
		_moveWish.multiplyScalar(
			this.sceneMaxDim *
				FREEFLY_SPEED_FRAC *
				(keys.has("Shift") ? FREEFLY_SPRINT : 1),
		);
		// Frame-rate independent approach, so the feel is the same at 60 and 144.
		this.freeflyVel.lerp(_moveWish, 1 - Math.exp(-(dt * 1000) / FREEFLY_ACCEL_TAU));
		this.camera.position.addScaledVector(this.freeflyVel, dt);
		this.lat = applyLook(this.camera, this.lon, this.lat);

		if (!this.splatRevealing) return;
		// The departing panorama rides the camera while it fades, so the dissolve
		// changes only opacity — a backdrop left behind would parallax against the
		// splat and read as two rooms sliding apart.
		this.splatReveal = Math.min(1, this.splatReveal + (dt * 1000) / SPLAT_REVEAL_MS);
		this.sphereB.position.copy(this.camera.position);
		this.sphereBMat.uniforms.opacity.value = 1 - easeInOut(this.splatReveal);
		if (this.splatReveal >= 1) {
			this.splatRevealing = false;
			this.clearPanoOverlay();
		}
	}

	// Parked dissolve between dollhouse and capture pano, entirely on the GPU.
	// "in" (enter): wait for the texture, then ramp the equirect over the dollhouse.
	// "out" (exit): the caller has already staged the sphere at full opacity over the
	// dollhouse; ramp it down so the capture is gone before the fly-out begins.
	private tickCrossfade(now: number) {
		const cf = this.crossfade;
		if (!cf) return;
		if (cf.armed === 0) {
			// Enter path only — exit arms itself before parking the Crossfade.
			const tex =
				this.flyTarget >= 0
					? this.panos[this.flyTarget]?.texture
					: null;
			if (!tex && now < cf.deadline) return; // parked, still streaming
			if (tex) {
				this.sphereBMat.uniforms.map.value = tex;
				this.sphereBMat.uniforms.opacity.value = 0;
				this.sphereBMat.depthTest = false; // sit OVER the dollhouse, not behind it
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

	// Tear down the dissolve sphere so a later hop crossfade finds its defaults.
	private clearPanoOverlay() {
		this.sphereB.visible = false;
		this.sphereBMat.uniforms.opacity.value = 0;
		this.sphereBMat.depthTest = true;
		this.sphereB.renderOrder = 1;
	}

	private updateCursorRing() {
		const active =
			this.isLookMode &&
			!this.interiorBusy &&
			this.pointerInside &&
			!this.markers.hoveredNav &&
			!this.markers.hoveredArrow;
		const hit = active
			? this.raycastInterior(this.pointerClientX, this.pointerClientY)
			: null;
		let ghosted = false;
		let reach: ReachTarget | null = null;
		// Where a click from here would carry the eye. The cursor turns this into its
		// direction arrow — but only where the surface it is lying on can express it
		// (see SurfaceCursor.aimArrow).
		let travel: Vector3 | null = null;
		if (hit && this.mode === "freefly") {
			// In flight the only question a cursor can answer is "where would this
			// put me down", so it answers exactly that: a waypoint standing on the
			// capture a click would land on, tethered back to the surface under the
			// pointer. None of the interior's floor scoping or occlusion tinting
			// applies — those describe a visitor rooted at one anchor, and you are
			// not rooted at one.
			const targetIdx = this.autoHomeTarget(hit, this.floorAt(hit.point), -1);
			if (targetIdx >= 0) {
				this.cursor.setColor(CURSOR_CLEAR);
				this.markers.showGhost(
					this.destinationFloor(targetIdx),
					{ to: targetIdx, type: "walk", dy: 0 },
					this.camera,
					this.host.clientHeight,
					hit.point,
				);
				ghosted = true;
				// Recorded so the shared change-detection below STREAMS this pano
				// while you are still deciding. Without it the first request happens
				// at click time, and the arrival then parks at the anchor waiting up
				// to a second for a 4k equirect to decode before the dissolve can
				// even begin — which is the flight landing and then visibly
				// re-settling. The walkthrough pre-warms the same way on hover; free
				// flight has to as well, and for the same reason.
				//
				// It drives nothing else here: the 360 preview panel this feeds in
				// the walkthrough is gated to interior mode (see emit).
				reach = {
					index: targetIdx,
					level: this.panoLevel[targetIdx] ?? -1,
					levelDelta: 0,
				};
			}
		} else if (hit && this.currentIndex >= 0) {
			const curLevel = this.panoLevel[this.currentIndex] ?? -1;
			// Ask the GEOMETRY which floor it is on, by testing the point against the
			// floors' described volumes — not the nearest capture point, which is what
			// this used to do and why pointing at a cliff face offered to send you to
			// whichever storey happened to have an anchor near the rock. Terrain and
			// the slab between storeys land in no volume at all, and -1 (on no floor)
			// is the right answer for them. Tours captured before floors carried
			// volumes keep the old nearest-anchor reading.
			const hitLevel = this.targetFloorFor(hit);
			// The floor under the cursor also SCOPES the destination, so the click
			// cannot land on a storey the preview didn't name.
			const targetIdx = this.autoHomeTarget(hit, hitLevel);
			if (targetIdx >= 0) {
				// Read the destination's OWN storey, never the storey of the geometry
				// under the cursor. They agree whenever the hit resolved to a floor,
				// because that floor then scoped the search — but they part company in
				// two cases, and both were reachable:
				//   • the hit is on no floor at all (terrain, a cliff, the slab between
				//     storeys — 56% of this scene's volume), so the search ran
				//     unscoped and could land anywhere;
				//   • the scoped search found nothing on that floor and fell back to
				//     the unscoped one.
				// In both, `hitLevel` is -1 or stale, `crossesLevel` reads false, and a
				// hop to another storey came up amber with no floor preview. What the
				// colour promises is where you END UP, so it has to be read from there.
				const destLevel = this.panoLevel[targetIdx] ?? -1;
				const crossesLevel =
					curLevel >= 0 && destLevel >= 0 && destLevel !== curLevel;
				const occluded = !this.isTargetClear(
					v3(this.panos[targetIdx].position),
				);
				// Out of sight is out of sight: BOTH cases wear the amber ring, so the
				// cursor answers one question only — can you see where this click puts
				// you? What differs is the second thing shown, and that follows from
				// how far the answer is from you.
				//
				// Behind geometry on THIS floor gets a waypoint ON the destination
				// anchor, tethered back to the cursor — the marker is the place you
				// land, the line is the "if I click here" that summoned it.
				//
				// Changing storey gets the same treatment plus the 360 preview, and its
				// waypoint is green with an up/down chevron. Placing the marker on the
				// destination is what makes this work across floors: it lands where you
				// will actually be, over your head or under your feet, and the tether
				// running to it through the slab is the honest picture of the move.
				//
				// In plain sight needs neither: a quiet grey ring, and the directional
				// arrow wherever the surface can carry one.
				if (crossesLevel) {
					// Green, matching the floor arrows: changing storey is one idea and
					// it should wear one colour wherever it is offered, whether you
					// reached it by pointing or by hovering an arrow.
					this.cursor.setColor(NAV_COLORS.vertical);
					const dy = destLevel - curLevel;
					reach = {
						index: targetIdx,
						level: destLevel,
						levelDelta: dy,
					};
					this.markers.showGhost(
						this.destinationFloor(targetIdx),
						// `vertical` gives it the green ring and the taller chevron, and
						// `dy` points that chevron the way you are going.
						{ to: targetIdx, type: "vertical", dy },
						this.camera,
						this.host.clientHeight,
						hit.point,
					);
					ghosted = true;
				} else if (occluded) {
					this.cursor.setColor(NAV_COLORS.portal);
					// Same storey, but still out of sight — so it previews too. The
					// waypoint says WHERE the click lands; the panel says what is
					// there. levelDelta 0 is what tells the panel this is a hop
					// through geometry rather than a change of floor.
					reach = {
						index: targetIdx,
						level: destLevel,
						levelDelta: 0,
					};
					this.markers.showGhost(
						this.destinationFloor(targetIdx),
						{ to: targetIdx, type: "portal", dy: 0 },
						this.camera,
						this.host.clientHeight,
						hit.point,
					);
					ghosted = true;
				} else {
					this.cursor.setColor(CURSOR_CLEAR);
					travel = v3(this.panos[targetIdx].position).sub(
						this.camera.position,
					);
				}
			} else {
				this.cursor.setColor(CURSOR_CLEAR);
			}
		}
		if (!ghosted) this.markers.hideGhost();
		// Emit only when the DESTINATION changes, never per pixel of pointer travel —
		// the panel tracks the cursor itself (see OrbitViewer's ReachPreviewPanel), so
		// moving within one destination's catchment costs nothing. Re-targeting is
		// frequent by design now, but the panel cross-dissolves rather than remounting,
		// so a re-render here is a change of contents, not a rebuild of the window.
		const changed = (this.cursorReach?.index ?? -1) !== (reach?.index ?? -1);
		this.cursorReach = reach;
		if (changed) {
			if (reach) this.requestPano(reach.index); // warm the pano it pans through
			this.emit();
		}
		this.cursor.update(hit, this.camera, this.host.clientHeight, travel);
	}

	// X-ray name tags for the nearest sonar nodes (engine-owned DOM pool, so the
	// reveal labels track look without churning React).
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
		// Grow the pool on demand.
		while (this.sonarLabels.length < targets.length) {
			const el = document.createElement("div");
			Object.assign(el.style, {
				position: "absolute",
				transform: "translate(-50%, -140%)",
				padding: "1px 6px",
				borderRadius: "5px",
				background: "rgba(10,12,20,0.82)",
				color: "#cfe6ff",
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
