import {
	Box3,
	Color,
	DirectionalLight,
	DoubleSide,
	Group,
	HemisphereLight,
	type Intersection,
	type Material,
	MathUtils,
	Matrix3,
	Mesh,
	MeshBasicMaterial,
	MOUSE,
	type Object3D,
	NoToneMapping,
	PerspectiveCamera,
	PlaneGeometry,
	type Quaternion,
	Raycaster,
	RingGeometry,
	Scene,
	type ShaderMaterial,
	Sphere,
	SphereGeometry,
	SRGBColorSpace,
	type Texture,
	Vector2,
	Vector3,
	WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { loadGLB, loadPanoTexture } from "./loaders";
import {
	DUMMY_TEX,
	makePanoMaterial,
	makePolyMaterial,
	makeProjectionMaterial,
	PROJ_K,
	SPHERE_RADIUS,
} from "./materials";
import {
	ANCHOR_RING_INNER,
	ANCHOR_RING_OCCLUDED_COLOR,
	ANCHOR_RING_OCCLUDED_OPACITY,
	ANCHOR_RING_OCCLUDED_SCALE,
	ANCHOR_RING_OPACITY,
	ANCHOR_RING_OUTER,
	AUTO_AIM_PX,
	CAPTURE_EYE_HEIGHT,
	ENTRY_AIM_PX,
	ENTRY_TARGET_PX,
	HOTSPOT_FLOOR_DROP,
	HOTSPOT_MAX_OCCLUDED,
	HOTSPOT_OCCLUDE_EPS,
	HOTSPOT_REACH,
	hotspotScaleForDistance,
	makeDisc,
	makeYouMarker,
	PEEK_ROTATE_SPEED,
	pickByScreen,
	WASD_DIR_COS,
	WASD_MAX_STEP,
	WASD_MAX_Y_STEP,
	type YouMarker,
} from "./markers";
import type {
	MinimapLevel,
	OrbitMode,
	OrbitState,
	TourManifest,
	TourSource,
} from "./types";

const v3 = (a: [number, number, number]) => new Vector3().fromArray(a);
const easeInOut = (t: number) =>
	t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
const _ndc = new Vector2();
// Scratch for the surface-riding ring cursor (interior). RingGeometry lies in the
// XY plane facing +Z, so orienting +Z onto a hit normal lays the ring flat on
// that surface. RING_OUTER_PX keeps the ring a constant on-screen radius.
const _cursorNdc = new Vector2();
const _cursorNormal = new Vector3();
const _cursorNormalMat = new Matrix3();
const _cursorToCam = new Vector3();
const Z_AXIS = new Vector3(0, 0, 1);
const RING_OUTER_PX = 18;

type Transition = {
	fromPos: Vector3;
	toPos: Vector3;
	fromQuat: Quaternion;
	toQuat: Quaternion;
	start: number;
	dur: number;
	onMid?: () => void;
	onEnd?: () => void;
	midDone: boolean;
};
type Glide = {
	fromPos: Vector3;
	toPos: Vector3;
	start: number;
	dur: number;
	index: number;
};
type SavedInterior = {
	pos: Vector3;
	lon: number;
	lat: number;
	index: number;
	fov: number;
};

// A capture point. Textures are loaded lazily (on enter / on movement): a low-res
// blurred placeholder shows first, then the full image swaps in. `texture` is the
// current best (placeholder or full); `placeholderTexture` is kept only so it can
// be disposed at teardown without racing a bound shader uniform.
type PanoEntry = {
	id: string;
	position: [number, number, number];
	forward?: [number, number, number];
	url: string;
	placeholderUrl: string;
	texture: Texture | null;
	placeholderTexture: Texture | null;
	hasFull: boolean;
	requested: boolean;
	ready?: Promise<void>;
	resolveReady?: () => void;
};

// A combined dollhouse + interior walkthrough. OVERVIEW orbits the cell's
// vertex-colored lite scene with a free-pan camera; stepping INSIDE drops into
// the pano projection walkthrough. Both live in the SAME world frame, so a "you
// are here" marker dropped at the interior camera maps onto the dollhouse with
// no coordinate fixup. All per-frame work mutates three.js / the canvas directly
// (never React), so the UI only re-renders on discrete state changes.
export class OrbitEngine {
	private readonly host: HTMLElement;
	private readonly onState: (s: OrbitState) => void;
	private readonly onHold?: (held: boolean) => void;

	private readonly renderer: WebGLRenderer;
	private readonly canvas: HTMLCanvasElement;
	private readonly travelFade: HTMLDivElement;
	private readonly scene: Scene;
	private readonly camera: PerspectiveCamera;
	private readonly controls: OrbitControls;
	private readonly ro: ResizeObserver;

	private readonly sphereA: Mesh;
	private readonly sphereAMat: ShaderMaterial;
	private readonly sphereB: Mesh;
	private readonly sphereBMat: ShaderMaterial;
	private readonly projMaterial: ShaderMaterial;
	private readonly polyMaterial = makePolyMaterial();
	private backdropRadius = SPHERE_RADIUS;

	private readonly hotspotGroup = new Group();
	private readonly entryGroup = new Group();
	// Every anchor as a small white ring laid flat on the floor. Depth-tested
	// (depthTest defaults on) so scene geometry obstructs it, and never
	// screen-scaled in tick() so it keeps a world-fixed size — smaller far, larger
	// near, like a real object. One shared geometry + material across all anchors.
	private readonly anchorRingGroup = new Group();
	private readonly anchorRingGeo = new RingGeometry(
		ANCHOR_RING_INNER,
		ANCHOR_RING_OUTER,
		40,
	);
	private readonly anchorRingMat = new MeshBasicMaterial({
		color: 0xffffff,
		transparent: true,
		opacity: ANCHOR_RING_OPACITY,
		side: DoubleSide,
		depthWrite: false,
	});
	// The X closest obstructed anchors reuse the ring geometry in warm gold, drawn
	// over everything (depthTest off) so they show through walls as reachable.
	private readonly anchorRingOccludedMat = new MeshBasicMaterial({
		color: ANCHOR_RING_OCCLUDED_COLOR,
		transparent: true,
		opacity: ANCHOR_RING_OCCLUDED_OPACITY,
		side: DoubleSide,
		depthWrite: false,
		depthTest: false,
	});
	private readonly you: YouMarker;
	private readonly occluder = new Raycaster();
	private readonly dummyCam = new PerspectiveCamera();

	// Surface-adhering ring cursor (interior only): a flat ring laid on the world
	// point under the native cursor, oriented to the hit surface's normal so it
	// sits flush on floors / walls / objects. The OS cursor is never hidden — this
	// rides on top of it.
	private readonly cursorRing: Mesh;
	private readonly cursorRay = new Raycaster();
	private pointerClientX = 0;
	private pointerClientY = 0;
	private pointerInside = false;

	private panos: PanoEntry[] = [];
	private currentIndex = -1;
	private projectionMode = false;
	// Bird's-eye minimap slices (one per Y level) + the level each pano sits on
	// (its nearest minimap by capture height). Empty when the tour has no slices.
	private minimaps: Array<MinimapLevel & { url: string }> = [];
	private panoLevel: number[] = [];
	// Held image prefetchers for the slices, so every floor is cached up front and
	// the floor switcher is instant. Kept referenced so the in-flight loads aren't
	// GC'd; dropped on the next clearScene.
	private minimapPrefetch: HTMLImageElement[] = [];
	private liteRoot: Group | null = null;
	private proxyGroup: Group | null = null;
	private proxyBase: Mesh | null = null; // floor slab under the proxy (mirrors proxy material)
	private sharedOverview = false;
	private proxyView = false; // overview shows the proxy mesh instead of the lite dollhouse

	// Per-object addressing (lite + proxy). Each placed object is tagged at load so
	// the user can hover (cyan) and right-click (hide / persist orange) it. The
	// highlight is a translucent fill overlay — a copy of the mesh — parented to
	// each mesh, so reskinProxy() (which rewrites proxy mesh materials on view
	// changes) can't clobber it. It's depth-tested (with a slight negative polygon
	// offset so it doesn't z-fight the coincident source surface), so nearer
	// geometry occludes the tint — the object reads as highlighted in place, not
	// floated on top. depthTest rejects the back faces, so the DoubleSide fill is
	// single-coverage; opacity is set accordingly.
	private readonly picker = new Raycaster();
	private readonly hiddenObjects = new Set<Object3D>();
	private readonly outlinedObjects = new Set<Object3D>();
	private hoveredObj: Object3D | null = null;
	private menuTarget: Object3D | null = null;
	private contextMenu: OrbitState["contextMenu"] = null;
	private rcDownX = 0;
	private rcDownY = 0;
	private readonly hoverFillMat = new MeshBasicMaterial({
		color: 0x66e0ff,
		transparent: true,
		opacity: 0.42,
		side: DoubleSide,
		depthTest: true,
		depthWrite: false,
		polygonOffset: true,
		polygonOffsetFactor: -1,
		polygonOffsetUnits: -1,
	});
	private readonly selectFillMat = new MeshBasicMaterial({
		color: 0xffa23a,
		transparent: true,
		opacity: 0.6,
		side: DoubleSide,
		depthTest: true,
		depthWrite: false,
		polygonOffset: true,
		polygonOffsetFactor: -1,
		polygonOffsetUnits: -1,
	});

	private mode: OrbitMode = "empty";
	private readonly sceneCenter = new Vector3();
	private sceneMaxDim = 1;
	private readonly browsePos = new Vector3();

	private transition: Transition | null = null;
	private glide: Glide | null = null;

	private lon = 0;
	private lat = 0;
	private dragging = false;
	private dragMoved = 0;
	private downX = 0;
	private downY = 0;
	private downLon = 0;
	private downLat = 0;
	// Hover-highlight (cyan fill on the object under the cursor) is on by default;
	// the toolbar toggle flips this. Persistent right-click selections ignore it.
	private highlightEnabled = true;

	private interiorBusy = false;
	private savedInterior: SavedInterior | null = null;
	private peekHeld = false;

	private hoveredTargetIndex = -1;
	private hoveredEntryIndex = -1;
	private hoveredOccluded = false;

	private autoRotateTimer: ReturnType<typeof setTimeout> | null = null;
	private lastFrame = 0;
	private overlay: OrbitState["overlay"] = null;
	private readonly camDist2: number[] = [];

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

		this.renderer = new WebGLRenderer({ antialias: true });
		this.renderer.setPixelRatio(window.devicePixelRatio);
		this.renderer.toneMapping = NoToneMapping;
		this.renderer.outputColorSpace = SRGBColorSpace;
		this.canvas = this.renderer.domElement;
		// CSS-size the canvas to fill the host (buffer stays at w*dpr for crisp
		// HiDPI); setSize(w,h,false) below avoids overriding these styles.
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

		this.scene = new Scene();
		this.scene.background = new Color(0x0c0d10);
		this.scene.add(new HemisphereLight(0xffffff, 0x202028, 1.0));
		const dir1 = new DirectionalLight(0xffffff, 1.1);
		dir1.position.set(3, 5, 4);
		const dir2 = new DirectionalLight(0xffffff, 0.5);
		dir2.position.set(-3, 2, -2);
		this.scene.add(dir1, dir2);

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

		this.projMaterial = makeProjectionMaterial();
		this.sphereAMat = makePanoMaterial();
		this.sphereBMat = makePanoMaterial();
		// Valid sampler before any pano texture loads (panos load lazily now).
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

		this.scene.add(this.hotspotGroup, this.entryGroup, this.anchorRingGroup);
		this.you = makeYouMarker();
		this.scene.add(this.you.group);

		// Base outer radius 1; scaled per-frame to a constant pixel size. Drawn over
		// everything (depthTest off), like the other markers, so it stays visible.
		this.cursorRing = new Mesh(
			new RingGeometry(0.72, 1, 48),
			new MeshBasicMaterial({
				color: 0x7fe9ff,
				transparent: true,
				opacity: 0.9,
				side: DoubleSide,
				depthTest: false,
				depthWrite: false,
			}),
		);
		this.cursorRing.renderOrder = 1000;
		this.cursorRing.frustumCulled = false;
		this.cursorRing.visible = false;
		this.scene.add(this.cursorRing);

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
		this.scene.remove(this.cursorRing);
		this.cursorRing.geometry.dispose();
		(this.cursorRing.material as Material).dispose();
		this.scene.remove(this.anchorRingGroup);
		this.anchorRingGeo.dispose();
		this.anchorRingMat.dispose();
		this.anchorRingOccludedMat.dispose();
		this.hoverFillMat.dispose();
		this.selectFillMat.dispose();
		this.renderer.dispose();
		this.canvas.remove();
		this.travelFade.remove();
	}

	// --- state emission (gated so chrome holds through camera flights) --------

	private emit() {
		if (this.mode === "transition") return;
		const cur =
			this.currentIndex >= 0 ? this.panos[this.currentIndex] : null;
		let hover: OrbitState["hover"] = null;
		if (this.mode === "overview" && this.hoveredEntryIndex >= 0) {
			hover = {
				id: this.panos[this.hoveredEntryIndex].id,
				occluded: false,
			};
		} else if (this.mode === "interior" && this.hoveredTargetIndex >= 0) {
			hover = {
				id: this.panos[this.hoveredTargetIndex].id,
				occluded: this.hoveredOccluded,
			};
		}
		const state: OrbitState = {
			mode: this.mode,
			panoCount: this.panos.length,
			currentId: cur ? cur.id : null,
			currentIndex: this.currentIndex,
			hover,
			objectHover: this.hoveredObj
				? (this.hoveredObj.userData.objLabel as string)
				: null,
			proxyView: this.proxyView,
			canProxyView: this.canToggleProxyView(),
			highlightEnabled: this.highlightEnabled,
			canHighlight:
				(this.mode === "overview" || this.mode === "interior") &&
				!!this.activeObjectRoot(),
			contextMenu: this.contextMenu,
			busy: this.interiorBusy,
			overlay: this.overlay,
			minimap: this.buildMinimapState(),
		};
		this.onState(state);
	}

	private showOverlay(msg: string, { spinner = true, err = false } = {}) {
		this.overlay = { msg, spinner, err };
		this.emit();
	}
	private hideOverlay() {
		this.overlay = null;
		this.emit();
	}

	// --- minimap (bird's-eye slice for the current level) ---------------------

	// Nearest slice to a capture height. Levels are Y-separated, so argmin-|Δy|
	// reproduces the grouping the capturer used.
	private levelForY(y: number): number {
		let best = -1;
		let bestD = Infinity;
		for (let i = 0; i < this.minimaps.length; i++) {
			const d = Math.abs(this.minimaps[i].y - y);
			if (d < bestD) {
				bestD = d;
				best = i;
			}
		}
		return best;
	}

	// The minimap for whatever level the user is on right now — the matching
	// slice plus every same-level anchor placed onto it. Only while walking the
	// interior (or peeking); the overview already shows the whole dollhouse.
	private buildMinimapState(): OrbitState["minimap"] {
		if (this.minimaps.length === 0 || this.currentIndex < 0) return null;
		if (this.mode !== "interior" && this.mode !== "peek") return null;
		const currentLevel = this.panoLevel[this.currentIndex];
		if (currentLevel < 0) return null;
		const pct = (n: number) => Math.max(0, Math.min(100, n * 100));
		// Surface every floor's slice + its anchors so the chrome can browse other
		// levels without moving the camera; the live capture lights up on its level.
		const levels = this.minimaps.map((mm, idx) => {
			const w = mm.bounds.maxX - mm.bounds.minX;
			const d = mm.bounds.maxZ - mm.bounds.minZ;
			const points: {
				index: number;
				id: string;
				leftPct: number;
				topPct: number;
				current: boolean;
			}[] = [];
			for (let i = 0; i < this.panos.length; i++) {
				if (this.panoLevel[i] !== idx) continue;
				const p = this.panos[i].position;
				points.push({
					index: i,
					id: this.panos[i].id,
					leftPct: w > 0 ? pct((p[0] - mm.bounds.minX) / w) : 50,
					topPct: d > 0 ? pct((p[2] - mm.bounds.minZ) / d) : 50,
					current: i === this.currentIndex,
				});
			}
			return {
				level: idx,
				url: mm.url,
				aspect: d > 0 ? w / d : 1,
				points,
			};
		});
		return { currentLevel, levels };
	}

	// --- travel mask: blur the canvas + dip to bg, peaking mid-move -----------

	private setTravelMask(t: number) {
		const m = Math.sin(Math.PI * MathUtils.clamp(t, 0, 1));
		this.canvas.style.filter =
			m > 0.002 ? `blur(${(m * 7).toFixed(2)}px)` : "none";
		this.travelFade.style.opacity = (m * 0.5).toFixed(3);
	}
	private clearTravelMask() {
		this.canvas.style.filter = "none";
		this.travelFade.style.opacity = "0";
	}

	private resize() {
		const w = this.host.clientWidth;
		const h = this.host.clientHeight;
		if (w === 0 || h === 0) return;
		this.renderer.setSize(w, h, false);
		this.camera.aspect = w / h;
		this.camera.updateProjectionMatrix();
	}

	// --- input handlers (bound fields so dispose can detach them) -------------

	// Right-click an object → per-object menu (hide / outline). A right-DRAG still
	// pans the overview (OrbitControls RIGHT = PAN), so only a near-stationary
	// right-click counts. Addressing is an overview activity, so the menu is gated
	// to it.
	private onContextMenu = (ev: MouseEvent) => {
		ev.preventDefault();
		if (this.mode !== "overview") return;
		if (
			Math.hypot(ev.clientX - this.rcDownX, ev.clientY - this.rcDownY) > 6
		)
			return;
		this.openObjectMenu(ev.clientX, ev.clientY);
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
		if (this.mode !== "interior" || this.interiorBusy) return;
		this.dragging = true;
		this.dragMoved = 0;
		this.downX = ev.clientX;
		this.downY = ev.clientY;
		this.downLon = this.lon;
		this.downLat = this.lat;
		this.canvas.style.cursor = "grabbing";
		this.canvas.setPointerCapture(ev.pointerId);
	};

	private onPointerMove = (ev: PointerEvent) => {
		// Keep the last cursor position for the ring (updated every frame in tick so
		// it tracks look-drag rotation too, not just raw movement).
		this.pointerClientX = ev.clientX;
		this.pointerClientY = ev.clientY;
		this.pointerInside = true;
		if (this.mode === "overview") {
			if (ev.buttons !== 0) return; // skip mid-orbit drag
			const spot = pickByScreen(
				ev.clientX,
				ev.clientY,
				this.entryGroup,
				ENTRY_AIM_PX,
				this.camera,
				this.canvas,
			);
			const entryIdx = spot ? (spot.userData.targetIndex as number) : -1;
			// Entry discs are the primary overview action and win the hover; only
			// when none is under the cursor (and highlighting is on) do we offer the
			// object beneath it.
			const obj =
				entryIdx >= 0 || !this.highlightEnabled
					? null
					: this.pickObjectAt(ev.clientX, ev.clientY);
			this.canvas.style.cursor = entryIdx >= 0 || obj ? "pointer" : "";
			const hoverChanged = this.setObjectHover(obj);
			const entryChanged = entryIdx !== this.hoveredEntryIndex;
			this.hoveredEntryIndex = entryIdx;
			if (entryChanged || hoverChanged) this.emit();
			return;
		}
		if (this.mode !== "interior") return;
		if (this.dragging) {
			const k = (0.0032 * this.camera.fov) / 75;
			this.lon = this.downLon + (this.downX - ev.clientX) * k;
			this.lat = this.downLat + (ev.clientY - this.downY) * k;
			this.dragMoved = Math.max(
				this.dragMoved,
				Math.hypot(ev.clientX - this.downX, ev.clientY - this.downY),
			);
		} else if (!this.interiorBusy) {
			this.updateHover(ev);
		}
	};

	private onPointerUp = (ev: PointerEvent) => {
		if (this.mode !== "interior") return;
		this.dragging = false;
		this.canvas.style.cursor = "";
		if (this.dragMoved >= 5 || this.interiorBusy) return;
		// Click a ring to go straight there: a gold behind-wall marker, or a visible
		// white ring (both draw over / sit in the scene, so a screen-space pick is
		// what matches what you see). Off the rings, fall back to auto-aim's
		// nearest-anchor-to-the-clicked-surface pick.
		const spot = this.pickAnchorMarker(ev.clientX, ev.clientY);
		if (spot) this.travelTo(spot.userData.targetIndex as number);
		else this.autoAimTravel(ev.clientX, ev.clientY);
	};

	private onWheel = (ev: WheelEvent) => {
		if (this.mode !== "interior") return;
		ev.preventDefault();
		this.camera.fov = MathUtils.clamp(
			this.camera.fov + ev.deltaY * 0.05,
			25,
			100,
		);
		this.camera.updateProjectionMatrix();
	};

	private onClick = (ev: MouseEvent) => {
		if (this.mode !== "overview") return;
		const spot = pickByScreen(
			ev.clientX,
			ev.clientY,
			this.entryGroup,
			ENTRY_AIM_PX,
			this.camera,
			this.canvas,
		);
		if (spot) this.enter(spot.userData.targetIndex as number);
	};

	private onPointerLeave = () => {
		this.pointerInside = false;
		this.cursorRing.visible = false;
	};
	private onWindowPointerUp = () => this.peekUp();
	private onKeyDown = (ev: KeyboardEvent) => {
		if (ev.code === "Space" && !ev.repeat) {
			ev.preventDefault();
			this.peekDown();
			return;
		}
		// WASD walks the interior: forward is where you're looking (XZ of the look
		// azimuth), right is 90° off it. One step per press (auto-repeat ignored).
		if (this.mode !== "interior" || this.interiorBusy || ev.repeat) return;
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

	// Interior hover: navigation discs are the primary action and win the hover; the
	// proxy object beneath is highlighted only when no disc is under the cursor and
	// the hover-highlight toggle is on. The fill overlay is depth-test-off, so it
	// tints the object on top of the projected pano — it reads as picking the object
	// straight out of the 360 image.
	private updateHover(ev: PointerEvent) {
		if (this.currentIndex < 0) return;
		const spot = this.pickAnchorMarker(ev.clientX, ev.clientY);
		const idx = spot ? (spot.userData.targetIndex as number) : -1;
		const occ = spot ? !!spot.userData.occluded : false;
		const obj =
			idx >= 0 || !this.highlightEnabled
				? null
				: this.pickObjectAt(ev.clientX, ev.clientY);
		this.canvas.style.cursor = idx >= 0 || obj ? "pointer" : "";
		const hotspotChanged =
			idx !== this.hoveredTargetIndex || occ !== this.hoveredOccluded;
		this.hoveredTargetIndex = idx;
		this.hoveredOccluded = occ;
		if (this.setObjectHover(obj) || hotspotChanged) this.emit();
	}

	// Nearest clickable anchor marker under the cursor (screen-space magnetism): a
	// gold obstructed marker (drawn over everything, always reachable), else a white
	// ring that scene geometry isn't hiding — an occluded white ring is invisible,
	// so clicking the wall in front of it must NOT teleport through. Returns the
	// marker (read userData.targetIndex / .occluded) or null. Shared by hover +
	// click, so what lights up is exactly what a click travels to.
	private pickAnchorMarker(clientX: number, clientY: number): Object3D | null {
		const occluded = pickByScreen(
			clientX,
			clientY,
			this.hotspotGroup,
			AUTO_AIM_PX,
			this.camera,
			this.canvas,
		);
		if (occluded) return occluded;
		const ring = pickByScreen(
			clientX,
			clientY,
			this.anchorRingGroup,
			AUTO_AIM_PX,
			this.camera,
			this.canvas,
		);
		if (!ring) return null;
		const i = ring.userData.targetIndex as number;
		if (i === this.currentIndex) return null;
		const cur = this.panos[this.currentIndex].position;
		return this.anchorOccluded(cur, this.panos[i].position) ? null : ring;
	}

	// Interior geometry under a screen point: the projection proxy plus its floor
	// base, or the pano sphere when there's no proxy. Returns the nearest visible
	// hit (raycasting doesn't skip hidden objects, so walk the parent chain), or
	// null. Shared by the surface cursor ring and click auto-aim.
	private raycastInterior(
		clientX: number,
		clientY: number,
	): Intersection | null {
		const targets: Object3D[] = [];
		if (this.projectionMode) {
			if (this.proxyGroup) targets.push(this.proxyGroup);
			if (this.proxyBase) targets.push(this.proxyBase);
		} else {
			this.sphereA.updateMatrixWorld();
			targets.push(this.sphereA);
		}
		if (targets.length === 0) return null;
		const rect = this.canvas.getBoundingClientRect();
		_cursorNdc.set(
			((clientX - rect.left) / rect.width) * 2 - 1,
			-((clientY - rect.top) / rect.height) * 2 + 1,
		);
		this.camera.updateMatrixWorld();
		this.cursorRay.setFromCamera(_cursorNdc, this.camera);
		return (
			this.cursorRay.intersectObjects(targets, true).find((h) => {
				for (let o: Object3D | null = h.object; o; o = o.parent)
					if (!o.visible) return false;
				return true;
			}) ?? null
		);
	}

	// Auto-aim travel: world-raycast the click into the interior geometry and walk
	// to the anchor closest (in world space) to where it lands. Clicking any
	// surface snaps to the nearest capture point — no need to hit a marker.
	private autoAimTravel(clientX: number, clientY: number) {
		const hit = this.raycastInterior(clientX, clientY);
		if (hit) this.travelTo(this.nearestPanoTo(hit.point));
	}

	// The surface-riding ring cursor. Lays a flat ring on the first interior surface
	// under the native cursor, oriented to that surface's normal so it foreshortens
	// with floors / walls. Kept a constant on-screen size and drawn over everything
	// (depthTest off), like the other markers. Runs every frame so it follows both
	// cursor motion and look-drag rotation. The OS cursor is never touched.
	private updateCursorRing() {
		if (
			this.mode !== "interior" ||
			this.interiorBusy ||
			!this.pointerInside
		) {
			this.cursorRing.visible = false;
			return;
		}
		const hit = this.raycastInterior(
			this.pointerClientX,
			this.pointerClientY,
		);
		if (!hit) {
			this.cursorRing.visible = false;
			return;
		}
		// Local→world surface normal (fall back to facing the camera if the hit
		// carried none), flipped camera-ward so the ring's plane hugs the surface and
		// a hair of offset keeps it off the skin.
		if (hit.face) {
			_cursorNormalMat.getNormalMatrix(hit.object.matrixWorld);
			_cursorNormal
				.copy(hit.face.normal)
				.applyMatrix3(_cursorNormalMat)
				.normalize();
		} else {
			_cursorNormal.copy(this.camera.position).sub(hit.point).normalize();
		}
		_cursorToCam.copy(this.camera.position).sub(hit.point);
		if (_cursorNormal.dot(_cursorToCam) < 0) _cursorNormal.negate();
		this.cursorRing.position
			.copy(hit.point)
			.addScaledVector(_cursorNormal, hit.distance * 0.01);
		this.cursorRing.quaternion.setFromUnitVectors(Z_AXIS, _cursorNormal);
		const tan = Math.tan((this.camera.fov * Math.PI) / 360);
		this.cursorRing.scale.setScalar(
			(RING_OUTER_PX * 2 * hit.distance * tan) /
				(this.host.clientHeight || 1),
		);
		this.cursorRing.visible = true;
	}

	// --- view toggles (which geometry each mode shows) ------------------------

	private reskinProxy(mat: Material) {
		if (!this.proxyGroup) return;
		this.proxyGroup.traverse((o) => {
			const m = o as Mesh;
			if (m.isMesh) m.material = mat;
		});
		// The base mirrors the proxy: panos project onto it in the walkthrough (so
		// it blends into the floor instead of showing through as a flat fill), and
		// it goes matte in proxy view.
		if (this.proxyBase) this.proxyBase.material = mat;
	}

	// The proxy stands in for the dollhouse when there's no lite export, or when
	// the user flipped on "proxy view" to inspect/address the low-poly geometry.
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
		this.hotspotGroup.visible = false;
		this.entryGroup.visible = true;
		this.anchorRingGroup.visible = false;
		this.you.group.visible = false;
		this.syncProxyBase();
	}

	// Swap the interior proxy in place between the bare low-poly mesh (proxy view:
	// flat matte, no pano) and the captured panos projected onto it. The backdrop
	// sphere is the projected sky, so it's off in proxy view. Safe to call
	// mid-walkthrough — it leaves the navigation markers alone.
	private setInteriorProxyView() {
		if (!this.proxyGroup || !this.projectionMode) return;
		this.reskinProxy(
			this.proxyView ? this.polyMaterial : this.projMaterial,
		);
		this.sphereA.visible = !this.proxyView;
		// Refresh the projection uniforms after a proxy-view spell (updateProjection
		// is skipped while it's on) so the first textured frame isn't stale.
		if (!this.proxyView) this.updateProjection();
	}

	private setInteriorView() {
		if (this.liteRoot) this.liteRoot.visible = false;
		if (this.proxyGroup) this.proxyGroup.visible = this.projectionMode;
		if (this.projectionMode) {
			this.setInteriorProxyView();
		} else {
			this.sphereA.visible = true; // sphere-only tour: the pano sphere IS the view
		}
		this.hotspotGroup.visible = true;
		this.entryGroup.visible = false;
		this.anchorRingGroup.visible = true;
		this.you.group.visible = false;
		this.syncProxyBase();
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
		this.hotspotGroup.visible = false;
		this.entryGroup.visible = false;
		this.anchorRingGroup.visible = false;
		this.you.group.visible = true;
		this.syncProxyBase();
	}

	// Where the bare-proxy / textured swap is offered: the overview needs a lite
	// scene to swap the proxy in for (sharedOverview tours already show the proxy);
	// the interior needs a proxy to project onto (i.e. projection mode).
	private canToggleProxyView(): boolean {
		if (this.mode === "overview")
			return !!this.liteRoot && !!this.proxyGroup;
		if (this.mode === "interior") return this.projectionMode;
		return false;
	}

	// Flip between the textured scene (lite dollhouse / projected panos) and the
	// bare low-poly proxy — in the overview AND the first-person interior.
	toggleProxyView() {
		if (!this.canToggleProxyView()) return;
		this.proxyView = !this.proxyView;
		this.setObjectHover(null);
		this.contextMenu = null;
		this.menuTarget = null;
		this.canvas.style.cursor = "";
		if (this.mode === "overview") this.setOverviewView();
		else if (this.mode === "interior") this.setInteriorProxyView();
		this.emit();
	}

	// Turn the on-hover object highlight on/off (persistent right-click selections
	// are unaffected). Clearing the live hover on the way off; the next pointer move
	// re-picks when turned back on.
	toggleHighlight() {
		this.highlightEnabled = !this.highlightEnabled;
		if (!this.highlightEnabled) this.setObjectHover(null);
		this.canvas.style.cursor = "";
		this.emit();
	}

	// --- per-object addressing (pick / hide / outline) ------------------------

	// The addressable objects of a loaded root. The exporter/loader can wrap the
	// real objects under a single node, so unwrap single unnamed non-mesh wrappers,
	// then take that container's mesh-bearing children. Fall back to "every mesh"
	// when the structure is flat or collapses to one node.
	private collectObjects(root: Object3D): Object3D[] {
		let container = root;
		while (
			container.children.length === 1 &&
			!(container.children[0] as Mesh).isMesh &&
			!container.children[0].name &&
			container.children[0].children.length > 0
		) {
			container = container.children[0];
		}
		const hasMesh = (o: Object3D) => {
			let found = false;
			o.traverse((c) => {
				if ((c as Mesh).isMesh) found = true;
			});
			return found;
		};
		let objs = container.children.filter(hasMesh);
		if (objs.length <= 1) {
			objs = [];
			root.traverse((o) => {
				if ((o as Mesh).isMesh) objs.push(o);
			});
		}
		return objs;
	}

	private registerObjects(root: Object3D) {
		this.collectObjects(root).forEach((o, i) => {
			o.userData.objId = i;
			o.userData.objLabel =
				o.name && o.name.trim() ? o.name.trim() : `object ${i + 1}`;
		});
	}

	// Lazily build the highlight overlay: a translucent copy of each sub-mesh
	// (sharing the source geometry) parented to it, so it tracks the object's
	// transform and survives reskinProxy() (which only rewrites the base mesh
	// materials). The fill material is depth-tested, so nearer geometry occludes the
	// tint. Collect first, then attach — adding children mid-traverse would
	// otherwise re-visit (and re-overlay) the overlays.
	private ensureFillOverlay(obj: Object3D): Mesh[] {
		const ud = obj.userData as { fillOverlay?: Mesh[] };
		if (ud.fillOverlay) return ud.fillOverlay;
		const meshes: Mesh[] = [];
		obj.traverse((node) => {
			const mesh = node as Mesh;
			if (mesh.isMesh && mesh.geometry && !mesh.userData.isOutline)
				meshes.push(mesh);
		});
		ud.fillOverlay = meshes.map((mesh) => {
			const fill = new Mesh(mesh.geometry, this.hoverFillMat);
			fill.userData.isOutline = true; // never a pick target / never itself overlaid
			fill.raycast = () => {};
			fill.renderOrder = 6;
			fill.frustumCulled = false;
			fill.visible = false;
			mesh.add(fill);
			return fill;
		});
		return ud.fillOverlay;
	}

	// Selection (persistent, orange) beats hover (transient, cyan) beats none.
	private refreshOutline(obj: Object3D) {
		const kind = this.outlinedObjects.has(obj)
			? "select"
			: obj === this.hoveredObj
				? "hover"
				: "none";
		const ud = obj.userData as { fillOverlay?: Mesh[] };
		if (kind === "none" && !ud.fillOverlay) return;
		const mat = kind === "select" ? this.selectFillMat : this.hoverFillMat;
		for (const fill of this.ensureFillOverlay(obj)) {
			fill.visible = kind !== "none";
			fill.material = mat;
		}
	}

	private setObjectHover(obj: Object3D | null): boolean {
		if (obj === this.hoveredObj) return false;
		const prev = this.hoveredObj;
		this.hoveredObj = obj;
		if (prev) this.refreshOutline(prev);
		if (obj) this.refreshOutline(obj);
		return true;
	}

	private setObjectHidden(obj: Object3D, hidden: boolean) {
		obj.visible = !hidden;
		if (hidden) {
			this.hiddenObjects.add(obj);
			if (obj === this.hoveredObj) this.hoveredObj = null; // can't hover what's gone
		} else {
			this.hiddenObjects.delete(obj);
		}
	}

	private toggleObjectOutline(obj: Object3D) {
		if (this.outlinedObjects.has(obj)) this.outlinedObjects.delete(obj);
		else this.outlinedObjects.add(obj);
		this.refreshOutline(obj);
	}

	// Which loaded root the cursor addresses right now: the dollhouse in overview /
	// peek (lite, or the proxy when "proxy view" is on or there's no lite), and the
	// projected proxy in the interior (hover-highlight — see updateHover).
	private activeObjectRoot(): Object3D | null {
		if (this.mode === "overview" || this.mode === "peek") {
			if (this.proxyView && this.proxyGroup) return this.proxyGroup;
			return this.liteRoot ?? this.proxyGroup;
		}
		if (this.mode === "interior") return this.proxyGroup;
		return null;
	}

	private findObjectRoot(node: Object3D, root: Object3D): Object3D | null {
		let cur: Object3D | null = node;
		while (cur && cur !== root) {
			if (cur.userData.objId !== undefined) return cur;
			cur = cur.parent;
		}
		return null;
	}

	private pickObjectAt(clientX: number, clientY: number): Object3D | null {
		const root = this.activeObjectRoot();
		if (!root || !root.visible) return null;
		const rect = this.canvas.getBoundingClientRect();
		_ndc.set(
			((clientX - rect.left) / rect.width) * 2 - 1,
			-((clientY - rect.top) / rect.height) * 2 + 1,
		);
		this.picker.setFromCamera(_ndc, this.camera);
		// Raycaster doesn't skip invisible objects, so skip hidden ones explicitly —
		// otherwise a hidden object would still shadow the geometry behind it.
		for (const h of this.picker.intersectObject(root, true)) {
			if ((h.object.userData as { isOutline?: boolean }).isOutline)
				continue;
			const obj = this.findObjectRoot(h.object, root);
			if (obj && obj.visible) return obj;
		}
		return null;
	}

	// --- right-click object menu ----------------------------------------------

	private openObjectMenu(clientX: number, clientY: number) {
		const obj = this.pickObjectAt(clientX, clientY);
		// Right-clicking empty space still surfaces the recovery actions (you can't
		// re-pick an object once it's hidden), but only if there's something to recover.
		if (
			!obj &&
			this.hiddenObjects.size === 0 &&
			this.outlinedObjects.size === 0
		) {
			this.closeMenu();
			return;
		}
		this.menuTarget = obj;
		this.contextMenu = {
			x: clientX,
			y: clientY,
			label: obj ? (obj.userData.objLabel as string) : null,
			hidden: !!obj && this.hiddenObjects.has(obj),
			outlined: !!obj && this.outlinedObjects.has(obj),
			hiddenCount: this.hiddenObjects.size,
			outlinedCount: this.outlinedObjects.size,
		};
		this.emit();
	}

	closeMenu() {
		if (!this.contextMenu) return;
		this.contextMenu = null;
		this.menuTarget = null;
		this.emit();
	}

	toggleMenuTargetHidden() {
		const obj = this.menuTarget;
		if (obj) this.setObjectHidden(obj, !this.hiddenObjects.has(obj));
		this.closeMenu();
	}

	toggleMenuTargetOutline() {
		const obj = this.menuTarget;
		if (obj) this.toggleObjectOutline(obj);
		this.closeMenu();
	}

	showAllHidden() {
		for (const o of this.hiddenObjects) o.visible = true;
		this.hiddenObjects.clear();
		this.closeMenu();
	}

	clearOutlines() {
		const all = [...this.outlinedObjects];
		this.outlinedObjects.clear();
		for (const o of all) this.refreshOutline(o);
		this.closeMenu();
	}

	// --- markers --------------------------------------------------------------

	// Is the straight line between two capture points blocked by the proxy? Our
	// "behind a wall/floor" test, trimmed at both ends so a hugged wall doesn't read
	// as occlusion.
	private anchorOccluded(
		fromPos: [number, number, number],
		toPos: [number, number, number],
	): boolean {
		if (!this.proxyGroup) return false;
		const from = v3(fromPos);
		const d = v3(toPos).sub(from);
		const dist = d.length();
		if (dist < 1e-3) return false;
		d.divideScalar(dist);
		this.occluder.set(from, d);
		this.occluder.near = HOTSPOT_OCCLUDE_EPS;
		this.occluder.far = dist - HOTSPOT_OCCLUDE_EPS;
		if (this.occluder.far <= this.occluder.near) return false;
		return this.occluder.intersectObject(this.proxyGroup, true).length > 0;
	}

	private neighborsByDistance(): number[] {
		const cur = v3(this.panos[this.currentIndex].position);
		const out: Array<[number, number]> = [];
		for (let i = 0; i < this.panos.length; i++) {
			if (i === this.currentIndex) continue;
			const d2 = cur.distanceToSquared(v3(this.panos[i].position));
			if (this.projectionMode && d2 > HOTSPOT_REACH * HOTSPOT_REACH)
				continue;
			out.push([i, d2]);
		}
		out.sort((a, b) => a[1] - b[1]);
		return out.map((o) => o[0]);
	}

	// The X closest obstructed (behind-wall) anchors. Same ring as the white anchor
	// rings but warm gold, larger, and drawn over everything (depthTest off via
	// anchorRingOccludedMat) so they read as reachable through walls. Every anchor
	// also has a depth-tested white ring (buildAnchorRings); this layers the
	// see-through gold ring onto the nearest few that geometry hides. Rebuilt on
	// travel (occlusion is per position).
	private rebuildHotspots() {
		this.hotspotGroup.clear();
		if (this.currentIndex < 0) return;
		const cur = this.panos[this.currentIndex];
		let nOccluded = 0;
		for (const i of this.neighborsByDistance()) {
			if (!this.anchorOccluded(cur.position, this.panos[i].position))
				continue;
			const ring = new Mesh(this.anchorRingGeo, this.anchorRingOccludedMat);
			ring.rotation.x = -Math.PI / 2;
			ring.scale.setScalar(ANCHOR_RING_OCCLUDED_SCALE);
			ring.position.fromArray(this.panos[i].position);
			ring.position.y -= HOTSPOT_FLOOR_DROP;
			ring.userData.targetIndex = i;
			ring.userData.occluded = true;
			this.hotspotGroup.add(ring);
			if (++nOccluded >= HOTSPOT_MAX_OCCLUDED) break;
		}
	}

	// One white ring per anchor, laid flat on the floor — built once per scene
	// (every anchor, always present). Shares anchorRingGeo/anchorRingMat so it's
	// depth-tested (geometry obstructs it) and world-fixed-size; never touched by
	// tick()'s screen-scaling, so it shrinks far / grows near like a real object.
	private buildAnchorRings() {
		this.anchorRingGroup.clear();
		for (let i = 0; i < this.panos.length; i++) {
			const ring = new Mesh(this.anchorRingGeo, this.anchorRingMat);
			ring.rotation.x = -Math.PI / 2;
			ring.position.fromArray(this.panos[i].position);
			ring.position.y -= HOTSPOT_FLOOR_DROP;
			ring.userData.targetIndex = i;
			ring.userData.occluded = false;
			this.anchorRingGroup.add(ring);
		}
	}

	private buildEntryMarkers() {
		this.entryGroup.clear();
		for (let i = 0; i < this.panos.length; i++) {
			const spot = makeDisc(i, 0x9ad4ff, 0x4a8fd8);
			spot.position.fromArray(this.panos[i].position);
			spot.position.y -= HOTSPOT_FLOOR_DROP;
			this.entryGroup.add(spot);
		}
	}

	private sizeYouMarker() {
		const r = Math.max(0.05, this.sceneMaxDim * 0.014);
		this.you.sphere.geometry.dispose();
		this.you.sphere.geometry = new SphereGeometry(r, 24, 16);
		this.you.ring.geometry.dispose();
		this.you.ring.geometry = new RingGeometry(r * 1.6, r * 2.2, 40);
	}

	// Floor directly beneath the user (panos sit at eye height), not the global
	// scene minimum — so the base lands on the level you're standing on.
	private positionYouMarker(p: Vector3) {
		const floorY = p.y - CAPTURE_EYE_HEIGHT;
		this.you.sphere.position.copy(p);
		this.you.ring.position.set(p.x, floorY, p.z);
		this.you.line.geometry.setFromPoints([
			new Vector3(p.x, floorY, p.z),
			p.clone(),
		]);
	}

	// --- projection (view-dependent texture mapping, per frame) ---------------

	private updateProjection() {
		if (this.panos.length === 0) return;
		const u = this.projMaterial.uniforms;
		const cam = this.camera.position;
		this.camDist2.length = this.panos.length;
		for (let i = 0; i < this.panos.length; i++) {
			const p = this.panos[i].position;
			const dx = cam.x - p[0];
			const dy = cam.y - p[1];
			const dz = cam.z - p[2];
			this.camDist2[i] = dx * dx + dy * dy + dz * dz;
		}
		const order = this.panos
			.map((_, i) => i)
			.sort((a, b) => this.camDist2[a] - this.camDist2[b]);
		const K = Math.min(PROJ_K, this.panos.length);
		// Load on movement: kick off the K nearest captures, but project only the
		// ones already loaded (blurred placeholder counts) so we never block.
		const ready: number[] = [];
		for (let k = 0; k < K; k++) {
			this.requestPano(order[k]);
			if (this.panos[order[k]].texture) ready.push(order[k]);
		}
		let wsum = 0;
		const w: number[] = [];
		for (let k = 0; k < ready.length; k++) {
			const ww = 1 / (this.camDist2[ready[k]] + 0.25);
			w.push(ww);
			wsum += ww;
		}
		for (let k = 0; k < PROJ_K; k++) {
			if (k < ready.length) {
				const idx = ready[k];
				u.uTex.value[k] = this.panos[idx].texture;
				(u.uCenter.value[k] as Vector3).fromArray(
					this.panos[idx].position,
				);
				u.uWeight.value[k] = w[k] / wsum;
			} else {
				u.uTex.value[k] = DUMMY_TEX;
				u.uWeight.value[k] = 0;
			}
		}
		u.uCount.value = ready.length;
		this.sphereAMat.uniforms.map.value = ready.length
			? this.panos[ready[0]].texture
			: DUMMY_TEX;
		this.sphereA.position.copy(cam);
	}

	// Re-skin proxy meshes with the projection shader, recompute normals (the
	// decimation dropped usable ones), and size the backdrop to the scene extent.
	private setupProjection(root: Group) {
		root.traverse((o) => {
			const m = o as Mesh;
			if (!m.isMesh || !m.geometry) return;
			m.geometry.computeVertexNormals();
			m.material = this.projMaterial;
			m.frustumCulled = false;
		});
		const box = new Box3().setFromObject(root);
		const sph = box.getBoundingSphere(new Sphere());
		this.backdropRadius = Math.max(80, sph.radius * 4);
		this.sphereA.scale.setScalar(this.backdropRadius / SPHERE_RADIUS);
		this.sphereA.renderOrder = -1;
		this.sphereAMat.uniforms.opacity.value = 1;
		this.sphereAMat.depthTest = true; // let the opaque proxy occlude the backdrop
	}

	// A floor slab spanning the proxy's footprint, sat just under its lowest point,
	// backing "proxy leaks" (gaps in the proxy floor). It shares the proxy's
	// material (see reskinProxy), so the panos project onto it just like the floor:
	// at a capture point the projection equals the backdrop image, so the slab is
	// invisible, and it picks up the live projected floor as you move — rather than
	// showing through holes as a flat fill. Kept out of proxyGroup so it isn't
	// registered/picked as an addressable object; visibility is synced to the proxy.
	private buildProxyBase() {
		if (!this.proxyGroup) return;
		const box = new Box3().setFromObject(this.proxyGroup, true);
		if (box.isEmpty()) return;
		const size = box.getSize(new Vector3());
		const center = box.getCenter(new Vector3());
		const base = new Mesh(
			new PlaneGeometry(size.x, size.z),
			this.projMaterial,
		);
		base.rotation.x = -Math.PI / 2; // lie flat, normal up
		// A hair below the lowest vertex so it never z-fights a coincident floor.
		base.position.set(
			center.x,
			box.min.y - Math.max(0.01, size.y * 0.002),
			center.z,
		);
		base.frustumCulled = false;
		base.visible = false;
		this.scene.add(base);
		this.proxyBase = base;
	}

	// The base belongs to the proxy, so it shows exactly when the proxy does.
	private syncProxyBase() {
		if (this.proxyBase) this.proxyBase.visible = !!this.proxyGroup?.visible;
	}

	// --- look controls (interior: lon/lat drag) -------------------------------

	private setLookFromForward(f: [number, number, number]) {
		const v = v3(f).normalize();
		this.lon = Math.atan2(v.z, v.x);
		this.lat = Math.asin(MathUtils.clamp(v.y, -1, 1));
	}
	private lookTargetFrom(pos: Vector3, lo: number, la: number): Vector3 {
		return pos
			.clone()
			.add(
				new Vector3(
					Math.cos(la) * Math.cos(lo),
					Math.sin(la),
					Math.cos(la) * Math.sin(lo),
				),
			);
	}
	private applyLook() {
		this.lat = MathUtils.clamp(this.lat, -1.55, 1.55);
		this.camera.lookAt(
			this.lookTargetFrom(this.camera.position, this.lon, this.lat),
		);
	}

	// --- camera flight (mode changes: slerp orientation + lerp position) ------

	private startFly(
		toPos: Vector3,
		lookTarget: Vector3,
		dur: number,
		cbs: { onMid?: () => void; onEnd?: () => void } = {},
	) {
		// A camera (not a bare Object3D) so lookAt orients -Z toward the target,
		// matching how the real camera faces.
		this.dummyCam.up.copy(this.camera.up);
		this.dummyCam.position.copy(toPos);
		this.dummyCam.lookAt(lookTarget);
		this.dummyCam.updateMatrixWorld();
		this.transition = {
			fromPos: this.camera.position.clone(),
			toPos: toPos.clone(),
			fromQuat: this.camera.quaternion.clone(),
			toQuat: this.dummyCam.quaternion.clone(),
			start: performance.now(),
			dur,
			onMid: cbs.onMid,
			onEnd: cbs.onEnd,
			midDone: false,
		};
		this.mode = "transition";
		this.controls.enabled = false;
		this.controls.autoRotate = false;
		this.hoveredEntryIndex = -1;
		this.hoveredTargetIndex = -1;
		this.setObjectHover(null);
		this.contextMenu = null;
		this.menuTarget = null;
		this.canvas.style.cursor = "";
		this.emit(); // gated: holds the chrome while mode === "transition"
	}

	// --- travel between anchors (interior) ------------------------------------

	private travelTo(index: number) {
		if (
			this.interiorBusy ||
			index === this.currentIndex ||
			!this.panos[index]
		)
			return;
		this.hoveredTargetIndex = -1;
		this.interiorBusy = true;
		this.hotspotGroup.visible = false;

		if (this.projectionMode) {
			// Glide through world space; the projection re-blends live (loading the
			// captures we pass) so geometry + textures interpolate with parallax.
			this.requestPano(index);
			this.glide = {
				fromPos: this.camera.position.clone(),
				toPos: v3(this.panos[index].position),
				start: performance.now(),
				dur: 900,
				index,
			};
			return;
		}

		// Sphere mode: wait for the target's texture (a placeholder is enough), then
		// crossfade the backdrop to it while drifting the camera onto its position.
		const token = this.loadToken;
		void this.ensurePano(index).then(() => {
			if (this.disposed || token !== this.loadToken) return;
			const target = this.panos[index];
			this.sphereBMat.uniforms.map.value = target.texture ?? DUMMY_TEX;
			this.sphereBMat.uniforms.opacity.value = 0;
			this.sphereB.visible = true;
			const fromPos = this.camera.position.clone();
			const toPos = v3(target.position);
			const start = performance.now();
			const dur = 700;
			const step = (now: number) => {
				if (this.disposed || token !== this.loadToken) return;
				const t = Math.min(1, (now - start) / dur);
				const e = easeInOut(t);
				this.sphereBMat.uniforms.opacity.value = e;
				this.camera.position.lerpVectors(fromPos, toPos, e);
				this.setTravelMask(t);
				if (t < 1) {
					requestAnimationFrame(step);
					return;
				}
				this.sphereAMat.uniforms.map.value =
					target.texture ?? DUMMY_TEX;
				this.sphereAMat.uniforms.opacity.value = 1;
				this.sphereB.visible = false;
				this.clearTravelMask();
				this.interiorBusy = false;
				this.hotspotGroup.visible = true;
				this.activate(index);
			};
			requestAnimationFrame(step);
		});
	}

	private activate(index: number) {
		this.currentIndex = index;
		this.requestPano(index);
		if (!this.projectionMode) {
			this.sphereAMat.uniforms.map.value =
				this.panos[index].texture ?? DUMMY_TEX;
			this.sphereAMat.uniforms.opacity.value = 1;
		}
		this.rebuildHotspots();
		this.emit();
	}

	// --- lazy pano textures (load on enter / on movement) ---------------------

	// Fire-and-forget trigger (per-frame safe): start loading pano `i` if needed.
	private requestPano(i: number) {
		const p = this.panos[i];
		if (!p || p.texture || p.requested) return;
		this.startPanoLoad(i);
	}

	// Resolves once a texture (placeholder or full) is set — for paths that need
	// something to show before animating (sphere-mode travel).
	private ensurePano(i: number): Promise<void> {
		const p = this.panos[i];
		if (!p || p.texture) return Promise.resolve();
		if (!p.ready) {
			p.ready = new Promise<void>((res) => {
				p.resolveReady = res;
			});
		}
		const ready = p.ready;
		this.requestPano(i);
		return ready;
	}

	private startPanoLoad(i: number) {
		const p = this.panos[i];
		if (p.requested) return;
		p.requested = true;
		const token = this.loadToken;
		// Low-res blurred preview first (streams in fast), then the full image
		// sharpens in place — the panorama page's LQIP→full swap.
		loadPanoTexture(p.placeholderUrl)
			.then((tex) => {
				if (this.disposed || token !== this.loadToken || p.hasFull) {
					tex.dispose();
					return;
				}
				p.placeholderTexture = tex;
				if (!p.texture) {
					p.texture = tex;
					this.onPanoReady(i);
				}
			})
			.catch(() => {});
		loadPanoTexture(p.url)
			.then((tex) => {
				if (this.disposed || token !== this.loadToken) {
					tex.dispose();
					return;
				}
				p.hasFull = true;
				p.texture = tex;
				this.onPanoReady(i);
			})
			.catch(() => {});
	}

	private onPanoReady(i: number) {
		const p = this.panos[i];
		p.resolveReady?.();
		p.resolveReady = undefined;
		// Sphere mode shows one pano on the backdrop; refresh it if this is the
		// current capture. Projection mode re-reads textures every frame.
		if (!this.projectionMode && i === this.currentIndex) {
			this.sphereAMat.uniforms.map.value = p.texture ?? DUMMY_TEX;
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

	enter(index: number | null = null) {
		if (this.mode !== "overview" || this.panos.length === 0) return;
		const idx = index ?? this.nearestPanoTo(this.controls.target);
		this.requestPano(idx); // head start so the texture is ready by the fly-in's end
		const p = this.panos[idx];
		const fwd: [number, number, number] =
			p.forward && p.forward.length ? p.forward : [0, 0, 1];
		const toPos = v3(p.position);
		this.startFly(toPos, toPos.clone().add(v3(fwd)), 1100, {
			onMid: () => {
				this.setInteriorView();
				this.camera.fov = 75;
				this.camera.updateProjectionMatrix();
			},
			onEnd: () => {
				this.mode = "interior";
				this.setLookFromForward(fwd);
				this.activate(idx);
				this.emit();
			},
		});
	}

	exit() {
		if (this.mode !== "interior" || this.interiorBusy) return;
		this.startFly(this.browsePos.clone(), this.sceneCenter.clone(), 1000, {
			onMid: () => {
				this.setOverviewView();
				this.camera.fov = 55;
				this.camera.updateProjectionMatrix();
			},
			onEnd: () => {
				this.mode = "overview";
				this.controls.target.copy(this.sceneCenter);
				this.camera.position.copy(this.browsePos);
				this.controls.enabled = true;
				this.controls.update();
				this.controls.autoRotate = true;
				this.emit();
			},
		});
	}

	// Walk to a capture from the minimap (interior only; overview uses enter()).
	travelToIndex(index: number) {
		if (this.mode !== "interior" || this.interiorBusy) return;
		this.travelTo(index);
	}

	// WASD: step to the nearest anchor along a horizontal direction. "Nearest" is
	// XZ-only (Y ignored); a |Δy| gate keeps the step on the current floor and a
	// reach cap bounds how far one press travels. Only anchors inside a 45° cone of
	// the direction qualify, so the four keys tile the plane into quadrants.
	private stepToward(dirX: number, dirZ: number) {
		if (
			this.mode !== "interior" ||
			this.interiorBusy ||
			this.currentIndex < 0
		)
			return;
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
		if (best >= 0) this.travelTo(best);
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
		this.positionYouMarker(userPos);
		const flat = userPos.clone().sub(this.sceneCenter);
		flat.y = 0;
		if (flat.lengthSq() < 1e-6) flat.set(0, 0, 1);
		flat.normalize();
		const toPos = this.sceneCenter
			.clone()
			.addScaledVector(flat, this.sceneMaxDim * 1.5);
		toPos.y += this.sceneMaxDim * 0.6;
		this.startFly(toPos, this.sceneCenter.clone(), 850, {
			onMid: () => {
				this.setPeekView();
				this.camera.fov = 55;
				this.camera.updateProjectionMatrix();
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
		const s = this.savedInterior;
		this.startFly(
			s.pos.clone(),
			this.lookTargetFrom(s.pos, s.lon, s.lat),
			800,
			{
				onMid: () => {
					this.setInteriorView();
					this.camera.fov = s.fov;
					this.camera.updateProjectionMatrix();
				},
				onEnd: () => {
					this.mode = "interior";
					this.lon = s.lon;
					this.lat = s.lat;
					this.currentIndex = s.index;
					this.activate(s.index);
					this.emit();
				},
			},
		);
	}

	peekDown() {
		if (this.mode !== "interior" || this.interiorBusy) return;
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
			// Highlight overlays share their parent's geometry + a singleton material;
			// the parent disposes the geometry, so skip them here.
			if ((o.userData as { isOutline?: boolean }).isOutline) return;
			const m = o as Mesh;
			if (!m.isMesh && !(o as { isLine?: boolean }).isLine) return;
			m.geometry?.dispose();
			const mats = Array.isArray(m.material) ? m.material : [m.material];
			for (const mat of mats) {
				// Shared singletons (projection/poly fills, highlight fill mats) outlive
				// any one cell, so never dispose them here.
				if (
					mat &&
					mat !== this.projMaterial &&
					mat !== this.polyMaterial &&
					mat !== this.hoverFillMat &&
					mat !== this.selectFillMat
				)
					mat.dispose();
			}
		});
	}

	private clearScene() {
		this.loadToken++; // invalidate in-flight loads / sphere-travel steps
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
		if (this.proxyBase) {
			this.scene.remove(this.proxyBase);
			this.proxyBase.geometry.dispose(); // shares the proj/poly singletons — don't dispose them
			this.proxyBase = null;
		}
		for (const p of this.panos) {
			p.texture?.dispose();
			if (p.placeholderTexture && p.placeholderTexture !== p.texture)
				p.placeholderTexture.dispose();
		}
		this.panos = [];
		this.minimaps = [];
		this.panoLevel = [];
		this.minimapPrefetch = [];
		this.sphereAMat.uniforms.map.value = DUMMY_TEX;
		this.sphereBMat.uniforms.map.value = DUMMY_TEX;
		this.currentIndex = -1;
		this.hotspotGroup.clear();
		this.entryGroup.clear();
		this.anchorRingGroup.clear();
		this.you.group.visible = false;
		this.sphereA.visible = false;
		this.sphereB.visible = false;
		this.transition = null;
		this.glide = null;
		this.interiorBusy = false;
		this.peekHeld = false;
		this.savedInterior = null;
		this.hoveredEntryIndex = -1;
		this.hoveredTargetIndex = -1;
		// Reset per-object addressing; the old nodes are disposed with the roots.
		this.proxyView = false;
		this.hiddenObjects.clear();
		this.outlinedObjects.clear();
		this.hoveredObj = null;
		this.menuTarget = null;
		this.contextMenu = null;
		this.canvas.style.cursor = "";
		this.clearTravelMask();
	}

	// Build the scene from its R2 assets: the dollhouse (overview) GLB plus an
	// optional capture-tour manifest (pano positions + proxy). Only the dollhouse
	// + proxy geometry load here; the pano images load lazily (on enter / on
	// movement). With no manifest the panos can't be placed, so we orbit the
	// dollhouse alone — never inventing positions.
	async loadTour(source: TourSource) {
		this.mode = "loading";
		this.controls.enabled = false;
		this.clearScene(); // bumps loadToken; stale awaits below bail out
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

			// Bird's-eye slices: resolve their URLs and prefetch EVERY floor now
			// (parallel with the GLB loads below), so paging floors on the minimap
			// is instant instead of fetching each slice on first view.
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
			// Panos load lazily (on enter / on movement); just resolve URLs now.
			const entries: PanoEntry[] = list.map((p) => {
				const { url, placeholderUrl } = source.resolvePano(p.file);
				return {
					id: p.id,
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
			this.applyScene(entries, proxyRoot, lite);
		} catch (e) {
			if (token !== this.loadToken || this.disposed) return;
			this.mode = "empty";
			this.showOverlay(
				`failed to load scene: ${e instanceof Error ? e.message : String(e)}`,
				{
					spinner: false,
					err: true,
				},
			);
		}
	}

	private applyScene(
		entries: PanoEntry[],
		proxyRoot: Group | null,
		lite: Group | null,
	) {
		this.panos = entries;
		this.panoLevel = entries.map((p) => this.levelForY(p.position[1]));
		this.projectionMode = !!proxyRoot;
		this.sharedOverview = !lite && !!proxyRoot; // no lite: the proxy doubles as the dollhouse

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
			this.setupProjection(proxyRoot);
			this.proxyGroup = proxyRoot;
			this.scene.add(proxyRoot);
		}

		// Tag each placed object in both roots so they can be hovered / hidden /
		// outlined individually (independently per scene — lite and proxy nodes
		// don't share identity).
		if (this.liteRoot) this.registerObjects(this.liteRoot);
		if (this.proxyGroup) this.registerObjects(this.proxyGroup);

		// Give proxy floor leaks an opaque backing (a base under its footprint).
		this.buildProxyBase();

		const framed = lite ?? proxyRoot!;
		const box = new Box3().setFromObject(framed);
		const size = box.getSize(new Vector3());
		box.getCenter(this.sceneCenter);
		this.sceneMaxDim = Math.max(size.x, size.y, size.z) || 1;

		this.camera.near = Math.max(0.02, this.sceneMaxDim * 0.002);
		this.camera.far = Math.max(500, this.sceneMaxDim * 60);

		this.sizeYouMarker();
		this.buildEntryMarkers();
		this.buildAnchorRings();
		this.rebuildHotspots();

		const dist = this.sceneMaxDim * 1.6;
		this.browsePos
			.copy(this.sceneCenter)
			.add(new Vector3(dist * 0.7, dist * 0.5, dist * 0.9));
		this.camera.position.copy(this.browsePos);
		this.camera.fov = 55;
		this.camera.lookAt(this.sceneCenter);
		this.camera.updateProjectionMatrix();
		this.controls.target.copy(this.sceneCenter);
		this.controls.enabled = true;
		this.controls.update();
		this.controls.autoRotate = true;

		this.setOverviewView();
		this.mode = "overview";
		this.hideOverlay(); // emits the framed overview state
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
			this.camera.updateMatrixWorld();
			this.setTravelMask(t);
			if (!tr.midDone && t >= 0.5) {
				tr.midDone = true;
				tr.onMid?.();
			}
			if (
				this.proxyGroup?.visible &&
				this.projectionMode &&
				!this.proxyView
			)
				this.updateProjection();
			if (t >= 1) {
				this.clearTravelMask();
				const cb = tr.onEnd;
				this.transition = null;
				cb?.();
			}
		} else if (this.mode === "overview") {
			this.controls.update();
		} else if (this.mode === "interior") {
			if (this.glide) {
				const g = this.glide;
				const t = Math.min(1, (now - g.start) / g.dur);
				this.camera.position.lerpVectors(
					g.fromPos,
					g.toPos,
					easeInOut(t),
				);
				this.setTravelMask(t);
				if (t >= 1) {
					this.clearTravelMask();
					const idx = g.index;
					this.glide = null;
					this.interiorBusy = false;
					this.hotspotGroup.visible = true;
					this.activate(idx);
				}
			}
			if (this.projectionMode) {
				if (!this.proxyView) this.updateProjection(); // proxy view shows bare geometry, no panos
			} else {
				this.sphereA.position.copy(this.camera.position);
			}
			this.applyLook();
		} else if (this.mode === "peek") {
			// Slowly orbit the dollhouse so locating gives a 360 view.
			const off = this.camera.position.clone().sub(this.sceneCenter);
			const a = PEEK_ROTATE_SPEED * dt;
			const c = Math.cos(a);
			const s = Math.sin(a);
			this.camera.position.x = this.sceneCenter.x + off.x * c - off.z * s;
			this.camera.position.z = this.sceneCenter.z + off.x * s + off.z * c;
			this.camera.lookAt(this.sceneCenter);
		}

		this.updateCursorRing();

		// Overview entry discs render at a constant on-screen size + pulse; the
		// interior anchor rings (white + gold) are world-fixed, so they're left be.
		if (this.entryGroup.visible) {
			const pulse = 1 + 0.07 * Math.sin(time * 0.004);
			for (const spot of this.entryGroup.children) {
				const hovered =
					spot.userData.targetIndex === this.hoveredEntryIndex;
				const d = this.camera.position.distanceTo(spot.position);
				spot.scale.setScalar(
					hotspotScaleForDistance(
						d,
						ENTRY_TARGET_PX,
						this.camera.fov,
						this.host.clientHeight,
					) * (hovered ? 1.35 : 1),
				);
				const disc = spot.children[0] as Mesh;
				const ring = spot.children[1] as Mesh;
				ring.scale.setScalar(pulse);
				(disc.material as MeshBasicMaterial).opacity = hovered ? 0.9 : 0.55;
				(ring.material as MeshBasicMaterial).opacity = hovered ? 1.0 : 0.85;
			}
		}

		this.renderer.render(this.scene, this.camera);
	};
}
