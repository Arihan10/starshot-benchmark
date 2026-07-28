import {
	Box3,
	Color,
	Group,
	type Intersection,
	type Material,
	MathUtils,
	Mesh,
	MOUSE,
	type Object3D,
	PerspectiveCamera,
	Plane,
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
	HOTSPOT_FLOOR_DROP,
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
import { collectObjects, ObjectAddressing } from "./objectAddressing";
import { type PanoEntry, PanoStreamer } from "./panoTextures";
import { Projection } from "./projection";
import { buildMinimapState, levelForY, type MinimapSlice } from "./minimap";
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

// Reading the cursor ray through an obstruction. A gap larger than this between
// two consecutive surfaces means the ray has come out the far side into open
// space…
const PORTAL_SPAN_GAP = 1.2;
// …and the waypoint then stands this far clear of it, so it reads as being beyond
// the wall rather than buried in it.
const PORTAL_CLEARANCE = 0.4;

const _cursorNdc = new Vector2();
const _bez = new Vector3();
const _flyDir = new Vector3();
const _wpDir = new Vector3();
const _wpOut = new Vector3();
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
	private panoLevel: number[] = [];
	private minimapPrefetch: HTMLImageElement[] = [];
	private liteRoot: Group | null = null;
	private proxyGroup: Group | null = null;
	private sharedOverview = false;
	private proxyView = false;
	private proxyColorMats: Material[] = [];
	private connectors: Connector[] = []; // parsed but not surfaced (highlights hidden for now)
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
	// The floor waypoint under the cursor (its destination capture + which floor it
	// leads to), plus one representative capture per floor to stand them on.
	private hoveredLevel: { index: number; level: number; up: boolean } | null =
		null;
	private levelReps: Array<{ level: number; index: number }> = [];
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

		this.renderer = new WebGLRenderer({ antialias: false });
		this.renderer.setPixelRatio(window.devicePixelRatio);
		// The display transform (ACES + sRGB + shadows) is owned by LightRig below.
		this.canvas = this.renderer.domElement;
		Object.assign(this.canvas.style, {
			display: "block",
			width: "100%",
			height: "100%",
		});
		host.appendChild(this.canvas);

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
		this.scene.background = new Color(0x0c0d10);
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
		if (this.autoRotateTimer) clearTimeout(this.autoRotateTimer);
		this.clearScene();
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
			highlightEnabled: this.highlightEnabled,
			canHighlight:
				(this.mode === "overview" || this.mode === "interior") &&
				!!this.activeObjectRoot(),
			contextMenu: this.addressing.menu,
			busy: this.interiorBusy,
			overlay: this.overlay,
			exits,
			preview: this.mode === "interior" ? this.buildPreview() : null,
			levelPreview:
				this.mode === "interior" ? this.buildLevelPreview() : null,
			arrival: this.mode === "interior" ? this.arrival : null,
			sonarActive: this.markers.sonarActive,
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

	// The floor-waypoint preview payload: the capture the monolith will drop you
	// into. The chrome pans it through a full 360 rather than showing the equirect
	// flat, so what you read is the room and not a warped strip.
	private buildLevelPreview(): OrbitState["levelPreview"] {
		const lvl = this.hoveredLevel;
		if (!lvl) return null;
		const p = this.panos[lvl.index];
		if (!p) return null;
		return {
			index: lvl.index,
			level: lvl.level,
			up: lvl.up,
			name: p.name ?? null,
			url: p.url,
			placeholderUrl: p.placeholderUrl,
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
		if (this.mode !== "interior") return;
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
		if (this.mode !== "interior") return;
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
			if (this.hoveredNavIndex !== -1 || this.hoveredLevel) {
				this.hoveredNavIndex = -1;
				this.hoveredLevel = null;
				this.markers.setNavHover(null);
				this.markers.setLevelHover(null);
				this.emit();
			}
		} else if (!this.interiorBusy) {
			this.updateHover(ev);
		}
	};

	private onPointerUp = (ev: PointerEvent) => {
		if (this.mode !== "interior") return;
		this.dragging = false;
		this.canvas.style.cursor = "";
		if (this.dragMoved >= 5) return;
		this.noteInput();
		// A floor monolith takes the click first (it's the big overlay target and it
		// changes storey); then an affordance traverses its edge; else click-anywhere
		// routing snaps to the node minimizing graph cost + angular deviation.
		const slab = this.markers.pickLevel(
			ev.clientX,
			ev.clientY,
			this.camera,
			this.canvas,
		);
		if (slab) {
			this.traverse(slab.userData.to as number);
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
		if (this.mode !== "interior") return;
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
		if (this.hoveredLevel) {
			this.hoveredLevel = null;
			this.markers.setLevelHover(null);
			this.emit();
		}
	};
	private onWindowPointerUp = () => this.peekUp();

	private onKeyDown = (ev: KeyboardEvent) => {
		if (ev.code === "Space" && !ev.repeat) {
			ev.preventDefault();
			this.peekDown();
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
	};

	// Interior hover: light the affordance under the cursor + surface its preview.
	private updateHover(ev: PointerEvent) {
		if (this.currentIndex < 0) return;
		// The floor monolith is a large, deliberate target drawn over the scene, so
		// it claims the hover ahead of the smaller affordances behind it.
		const slab = this.markers.pickLevel(
			ev.clientX,
			ev.clientY,
			this.camera,
			this.canvas,
		);
		this.markers.setLevelHover(slab);
		const level = slab
			? {
					index: slab.userData.to as number,
					level: slab.userData.level as number,
					up: slab.userData.up as boolean,
				}
			: null;
		let changed = (this.hoveredLevel?.index ?? -1) !== (level?.index ?? -1);
		this.hoveredLevel = level;
		if (level) {
			this.requestPano(level.index); // warm the pano the preview pans through
			this.markers.setNavHover(null);
			this.canvas.style.cursor = "pointer";
			changed = changed || this.hoveredNavIndex !== -1;
			this.hoveredNavIndex = -1;
			if (this.addressing.setHover(null) || changed) this.emit();
			return;
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
		changed = changed || idx !== this.hoveredNavIndex;
		this.hoveredNavIndex = idx;
		if (this.addressing.setHover(obj) || changed) this.emit();
	}

	// Every visible interior surface under a screen point, nearest first. The
	// intersect call computes and sorts the whole list anyway, so handing it all
	// back costs nothing over returning just the first — and the waypoint math needs
	// the hits BEHIND the first one to know where the obstruction ends.
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
		return this.cursorRay.intersectObjects(targets, true).filter((h) => {
			for (let o: Object3D | null = h.object; o; o = o.parent)
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

	// The pano that a floor click would snap to: the node minimizing (distance to
	// the hit point + angular deviation from the click bearing). Shared by
	// clickAnywhere (the actual traversal) and updateCursorRing (the live preview).
	private autoHomeTarget(hit: Intersection): number {
		const cam = this.camera.position;
		const clickBearing = Math.atan2(
			hit.point.z - cam.z,
			hit.point.x - cam.x,
		);
		let best = -1;
		let bestCost = Infinity;
		for (let i = 0; i < this.panos.length; i++) {
			if (i === this.currentIndex) continue;
			const pp = v3(this.panos[i].position);
			const d = pp.distanceTo(hit.point);
			const bearing = Math.atan2(pp.z - cam.z, pp.x - cam.x);
			const ang = Math.abs(angleDelta(clickBearing, bearing));
			const cost = d + ang * 3; // 1 rad off ≈ 3 m of detour
			if (cost < bestCost) {
				bestCost = cost;
				best = i;
			}
		}
		return best;
	}

	// Where to DRAW the waypoint for a hop that cuts through geometry.
	//
	// The real destination is behind the wall and usually off to one side, so a
	// marker sitting on it answers the wrong question — what you want to know is
	// what happens if you click HERE. So the marker follows the pointer on ONE axis
	// only: it keeps the cursor's world bearing (which is what screen-x reads off,
	// so it stays lined up horizontally), while its height is pinned to the
	// destination's floor. That keeps it lying flat on the ground like every other
	// puck instead of floating at whatever height the wall happened to be hit.
	//
	// Along that floor bearing we then slide it to the point nearest the true
	// destination — the least misleading spot available: lined up with the pointer,
	// smallest possible error against where the click actually lands. Finally it is
	// pushed past the obstruction, so it reads as "through this, out the other side".
	private portalWaypoint(hits: Intersection[], targetIdx: number): Vector3 {
		const origin = this.camera.position;
		const dest = this.panos[targetIdx].position;
		// The pointer's bearing, flattened — only the horizontal part sets screen-x.
		_wpDir.copy(hits[0].point).sub(origin);
		_wpDir.y = 0;
		// Aiming (near enough) straight up or down leaves no bearing to read; fall
		// back to the way the camera itself faces.
		if (_wpDir.lengthSq() < 1e-6) {
			this.camera.getWorldDirection(_wpDir);
			_wpDir.y = 0;
			if (_wpDir.lengthSq() < 1e-6) _wpDir.set(0, 0, 1);
		}
		_wpDir.normalize();
		// Ground distance that lands nearest the destination, never nearer than the
		// far side of whatever we're pointing through.
		const wanted =
			(dest[0] - origin.x) * _wpDir.x + (dest[2] - origin.z) * _wpDir.z;
		const depth = Math.max(this.exitPlanDepth(hits), wanted);
		// Scratch: showGhost copies this straight into the marker, never retains it.
		return _wpOut.set(
			origin.x + _wpDir.x * depth,
			dest[1] - HOTSPOT_FLOOR_DROP,
			origin.z + _wpDir.z * depth,
		);
	}

	// How far across the GROUND the obstruction under the pointer ends. The proxy is
	// double-sided, so a wall comes back as a front AND a back face, and a doorway
	// jamb can add several more — step through consecutive hits while they are still
	// the same obstruction, then clear the far side of it. Measured horizontally,
	// because the waypoint slides along the floor rather than down the cursor ray.
	private exitPlanDepth(hits: Intersection[]): number {
		let exit = 0; // index of the last surface still part of this obstruction
		for (let i = 1; i < hits.length; i++) {
			if (hits[i].distance - hits[exit].distance > PORTAL_SPAN_GAP) break;
			exit = i;
		}
		const p = hits[exit].point;
		const origin = this.camera.position;
		return (
			Math.hypot(p.x - origin.x, p.z - origin.z) + PORTAL_CLEARANCE
		);
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
		const best = this.autoHomeTarget(hit);
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

	private setOverviewView() {
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
		this.markers.hideSonar();
		this.markers.levelGroup.visible = false;
		this.markers.you.group.visible = false;
		this.projection.syncBase(!!this.proxyGroup?.visible);
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
		this.markers.levelGroup.visible = true;
		this.markers.you.group.visible = false;
		this.projection.syncBase(!!this.proxyGroup?.visible);
	}

	private setPeekView() {
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
		this.markers.hideSonar();
		this.markers.levelGroup.visible = false;
		this.markers.you.group.visible = true;
		this.projection.syncBase(!!this.proxyGroup?.visible);
	}

	private canToggleProxyView(): boolean {
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
		if (this.currentIndex >= 0) return [[this.currentIndex, 1]];
		if (this.flyTarget >= 0) return [[this.flyTarget, 1]];
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
			onMid: cbs.onMid,
			onEnd: cbs.onEnd,
			midDone: false,
		};
		// A crossfading flight never dims, so clear any dip left by an earlier one.
		if (cbs.crossfade) this.travelFade.style.opacity = "0";
		this.mode = "transition";
		this.controls.enabled = false;
		this.controls.autoRotate = false;
		this.hoveredNavIndex = -1;
		this.hoveredLevel = null;
		this.markers.setNavHover(null);
		this.markers.setLevelHover(null);
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
		this.hoveredNavIndex = -1;
		this.hoveredLevel = null;
		this.markers.setNavHover(null);
		this.markers.setLevelHover(null);
		this.markers.navGroup.visible = false;
		this.markers.levelGroup.visible = false;
		this.markers.hideSonar();
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
		this.refreshLevelWaypoints();
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
		this.requestPano(idx);
		const toPos = v3(this.panos[idx].position);
		// The walkthrough is a yaw/pitch rig, so read the live orbit direction back
		// as lon/lat and pre-clamp the pitch to the limit applyLook enforces. The
		// pose the flight lands on is then exactly the pose the rig holds afterwards,
		// so the handover doesn't snap the view a single degree.
		const dir = this.camera.getWorldDirection(_flyDir);
		const look = forwardToLonLat([dir.x, dir.y, dir.z]);
		const lon = look.lon;
		const lat = MathUtils.clamp(look.lat, -MAX_PITCH, MAX_PITCH);
		this.history = []; // a fresh interior session
		this.flyTarget = idx; // project the arrival during the fly-in (pre-activate)
		this.startFly(toPos, lookTargetFrom(toPos, lon, lat), 1100, {
			toFov: INTERIOR_FOV,
			crossfade: true,
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

	// Step back out. The capture image is dissolved away WHILE still parked at the
	// capture point — same pose, same FOV — so the dollhouse is the only thing left
	// when the fly-out begins. Flying with the pano still glued on was the
	// duplicated-room look: the equirect (and the projected proxy) rode along as
	// the camera pulled away, then the dollhouse appeared underneath it.
	exit() {
		if (this.mode !== "interior" || this.interiorBusy) return;
		this.director.abort();
		this.hoveredNavIndex = -1;
		this.hoveredLevel = null;
		this.markers.setNavHover(null);
		this.markers.setLevelHover(null);
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
		this.hoveredLevel = null;
		this.levelReps = [];
		this.proxyView = false;
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
				const res = await fetch(source.manifestUrl);
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
			if (token !== this.loadToken || this.disposed) return;
			this.applyScene(entries, proxyRoot, lite, connectors);
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
	) {
		this.connectors = connectors;
		this.streamer.reset(entries);
		this.panoLevel = entries.map((p) =>
			levelForY(this.minimaps, p.position[1]),
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
		if (this.liteRoot) this.addressing.register(this.liteRoot);
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
		this.buildLevelReps();

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

	// --- level waypoints ------------------------------------------------------

	// One representative capture per floor: the anchor nearest that floor's own
	// centroid. The raw centroid itself is no good — on an L-shaped floor or one
	// wrapped around a stairwell it lands inside a wall — and this marker has to
	// name a spot you can actually stand, because it is BOTH the thing you click
	// and the thing the hover preview shows. Floors come from `panoLevel`, so a
	// tour captured without minimap slices simply has no storeys and no waypoints.
	private buildLevelReps() {
		const byLevel = new Map<number, number[]>();
		for (let i = 0; i < this.panoLevel.length; i++) {
			const lv = this.panoLevel[i];
			if (lv < 0) continue;
			const members = byLevel.get(lv);
			if (members) members.push(i);
			else byLevel.set(lv, [i]);
		}
		const reps: Array<{ level: number; index: number }> = [];
		for (const [level, members] of byLevel) {
			const centroid = new Vector3();
			for (const i of members) centroid.add(v3(this.panos[i].position));
			centroid.divideScalar(members.length);
			let best = members[0];
			let bestD = Infinity;
			for (const i of members) {
				const d = centroid.distanceToSquared(
					v3(this.panos[i].position),
				);
				if (d < bestD) {
					bestD = d;
					best = i;
				}
			}
			reps.push({ level, index: best });
		}
		reps.sort((a, b) => a.level - b.level);
		this.levelReps = reps;
	}

	// Stand a monolith on the floor directly above and the floor directly below —
	// the two moves the waypoint promises. Floors further off are reached one
	// storey at a time, which is what keeps a tall scene from filling with slabs.
	private refreshLevelWaypoints() {
		const cur =
			this.currentIndex >= 0 ? this.panoLevel[this.currentIndex] : -1;
		if (cur < 0 || this.levelReps.length < 2) {
			this.markers.clearLevels();
			return;
		}
		this.markers.buildLevels(
			this.levelReps
				.filter((r) => Math.abs(r.level - cur) === 1)
				.map((r) => {
					const floorPos = v3(this.panos[r.index].position);
					floorPos.y -= HOTSPOT_FLOOR_DROP;
					return {
						level: r.level,
						index: r.index,
						floorPos,
						up: r.level > cur,
					};
				}),
		);
		this.markers.levelGroup.visible = this.mode === "interior";
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
			// A crossfading flight stays fully visible the whole way in — there is
			// nothing to hide, because the swap happens at the far end where the two
			// renders already agree.
			if (!tr.crossfade)
				this.travelFade.style.opacity = (
					Math.sin(Math.PI * t) * 0.5
				).toFixed(3);
			if (!tr.midDone && t >= 0.5) {
				tr.midDone = true;
				tr.onMid?.();
			}
			// Never project during a flight. The enter path is still on the dollhouse,
			// and the exit path has already dissolved the capture away — projecting
			// here would re-glue the pano to the proxy and ride it out with the camera
			// (the duplicated-room look).
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
				this.markers.updateLevels(
					this.camera,
					now,
					this.host.clientHeight,
				);
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
				// Never let stillness become stuckness: pulse the exits once on dwell.
				if (!this.dwellPulsed && now - this.lastInputAt > DWELL_MS) {
					this.dwellPulsed = true;
					this.markers.pulseExits(now, 1600);
				}
			}
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
		this.composer.render();
	};

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
			this.mode === "interior" &&
			!this.interiorBusy &&
			this.pointerInside &&
			!this.markers.hoveredNav &&
			!this.markers.hoveredLevel;
		const hits = active
			? this.raycastInteriorAll(this.pointerClientX, this.pointerClientY)
			: [];
		const hit = hits[0] ?? null;
		let ghosted = false;
		if (hit && this.currentIndex >= 0) {
			const targetIdx = this.autoHomeTarget(hit);
			if (targetIdx >= 0) {
				// One classification drives BOTH the cursor tint and the waypoint: a
				// visible level change reads vertical (green); otherwise a live LOS
				// test splits a clear walk (blue) from a hop that cuts through
				// geometry (orange).
				const rendered = this.navNode(this.currentIndex)?.rendered;
				const isVisibleVertical = rendered?.some(
					(e) => e.to === targetIdx && e.type === "vertical",
				);
				const targetPos = v3(this.panos[targetIdx].position);
				const type: EdgeType = isVisibleVertical
					? "vertical"
					: this.isTargetClear(targetPos)
						? "walk"
						: "portal";
				this.cursor.setColor(NAV_COLORS[type]);
				// Only a through-geometry hop grows a waypoint. A clear walk needs none
				// (you can already see the spot) and a level change has its monolith —
				// so the pointer sprouts an affordance exactly when the click would
				// take you somewhere you CAN'T see, and it's drawn under the cursor
				// past the obstruction rather than out at the hidden destination.
				if (type === "portal") {
					this.markers.showGhost(
						this.portalWaypoint(hits, targetIdx),
						{ to: targetIdx, type, dy: 0 },
						this.camera,
						this.host.clientHeight,
					);
					ghosted = true;
				}
			} else {
				this.cursor.setColor(NAV_COLORS.walk);
			}
		}
		if (!ghosted) this.markers.hideGhost();
		this.cursor.update(hit, this.camera, this.host.clientHeight);
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
