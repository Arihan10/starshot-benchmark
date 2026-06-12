import {
	Box3,
	Color,
	DirectionalLight,
	Group,
	HemisphereLight,
	type Material,
	MathUtils,
	Mesh,
	type MeshBasicMaterial,
	MOUSE,
	type Object3D,
	NoToneMapping,
	PerspectiveCamera,
	type Quaternion,
	Raycaster,
	RingGeometry,
	Scene,
	type ShaderMaterial,
	Sphere,
	SphereGeometry,
	SRGBColorSpace,
	type Texture,
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
	AUTO_AIM_PX,
	CAPTURE_EYE_HEIGHT,
	ENTRY_AIM_PX,
	ENTRY_TARGET_PX,
	HOTSPOT_FLOOR_DROP,
	HOTSPOT_MAX_OCCLUDED,
	HOTSPOT_MAX_VISIBLE,
	HOTSPOT_OCCLUDE_EPS,
	HOTSPOT_REACH,
	HOTSPOT_TARGET_PX,
	hotspotScaleForDistance,
	makeDisc,
	makeYouMarker,
	PEEK_ROTATE_SPEED,
	pickByScreen,
	type YouMarker,
} from "./markers";
import type { OrbitMode, OrbitState, TourManifest, TourSource } from "./types";

const v3 = (a: [number, number, number]) => new Vector3().fromArray(a);
const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

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
type Glide = { fromPos: Vector3; toPos: Vector3; start: number; dur: number; index: number };
type SavedInterior = { pos: Vector3; lon: number; lat: number; index: number; fov: number };

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
	private readonly you: YouMarker;
	private readonly occluder = new Raycaster();
	private readonly dummyCam = new PerspectiveCamera();

	private panos: PanoEntry[] = [];
	private currentIndex = -1;
	private projectionMode = false;
	private liteRoot: Group | null = null;
	private proxyGroup: Group | null = null;
	private sharedOverview = false;

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

	constructor(host: HTMLElement, onState: (s: OrbitState) => void, onHold?: (held: boolean) => void) {
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
		Object.assign(this.canvas.style, { display: "block", width: "100%", height: "100%" });
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
		this.controls.mouseButtons = { LEFT: MOUSE.ROTATE, MIDDLE: MOUSE.PAN, RIGHT: MOUSE.PAN };
		this.controls.enabled = false;
		this.controls.addEventListener("start", this.onControlsStart);
		this.controls.addEventListener("end", this.onControlsEnd);

		this.projMaterial = makeProjectionMaterial();
		this.sphereAMat = makePanoMaterial();
		this.sphereBMat = makePanoMaterial();
		// Valid sampler before any pano texture loads (panos load lazily now).
		this.sphereAMat.uniforms.map.value = DUMMY_TEX;
		this.sphereBMat.uniforms.map.value = DUMMY_TEX;
		this.sphereA = new Mesh(new SphereGeometry(SPHERE_RADIUS, 64, 32), this.sphereAMat);
		this.sphereB = new Mesh(this.sphereA.geometry, this.sphereBMat);
		this.sphereA.renderOrder = 0;
		this.sphereB.renderOrder = 1;
		this.sphereA.visible = false;
		this.sphereB.visible = false;
		this.scene.add(this.sphereA, this.sphereB);

		this.scene.add(this.hotspotGroup, this.entryGroup);
		this.you = makeYouMarker();
		this.scene.add(this.you.group);

		this.canvas.addEventListener("contextmenu", this.onContextMenu);
		this.canvas.addEventListener("pointerdown", this.onPointerDown);
		this.canvas.addEventListener("pointermove", this.onPointerMove);
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
		this.canvas.removeEventListener("pointerup", this.onPointerUp);
		this.canvas.removeEventListener("wheel", this.onWheel);
		this.canvas.removeEventListener("click", this.onClick);
		window.removeEventListener("pointerup", this.onWindowPointerUp);
		window.removeEventListener("keydown", this.onKeyDown);
		window.removeEventListener("keyup", this.onKeyUp);
		if (this.autoRotateTimer) clearTimeout(this.autoRotateTimer);
		this.clearScene();
		this.renderer.dispose();
		this.canvas.remove();
		this.travelFade.remove();
	}

	// --- state emission (gated so chrome holds through camera flights) --------

	private emit() {
		if (this.mode === "transition") return;
		const cur = this.currentIndex >= 0 ? this.panos[this.currentIndex] : null;
		let hover: OrbitState["hover"] = null;
		if (this.mode === "overview" && this.hoveredEntryIndex >= 0) {
			hover = { id: this.panos[this.hoveredEntryIndex].id, occluded: false };
		} else if (this.mode === "interior" && this.hoveredTargetIndex >= 0) {
			hover = { id: this.panos[this.hoveredTargetIndex].id, occluded: this.hoveredOccluded };
		}
		const state: OrbitState = {
			mode: this.mode,
			panoCount: this.panos.length,
			currentId: cur ? cur.id : null,
			currentIndex: this.currentIndex,
			hover,
			busy: this.interiorBusy,
			overlay: this.overlay,
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

	// --- travel mask: blur the canvas + dip to bg, peaking mid-move -----------

	private setTravelMask(t: number) {
		const m = Math.sin(Math.PI * MathUtils.clamp(t, 0, 1));
		this.canvas.style.filter = m > 0.002 ? `blur(${(m * 7).toFixed(2)}px)` : "none";
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

	private onContextMenu = (e: Event) => e.preventDefault();

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
		if (this.mode === "overview") {
			if (ev.buttons !== 0) return; // skip mid-orbit drag
			const spot = pickByScreen(ev.clientX, ev.clientY, this.entryGroup, ENTRY_AIM_PX, this.camera, this.canvas);
			const idx = spot ? (spot.userData.targetIndex as number) : -1;
			this.canvas.style.cursor = spot ? "pointer" : "";
			if (idx !== this.hoveredEntryIndex) {
				this.hoveredEntryIndex = idx;
				this.emit();
			}
			return;
		}
		if (this.mode !== "interior") return;
		if (this.dragging) {
			const k = (0.0032 * this.camera.fov) / 75;
			this.lon = this.downLon + (this.downX - ev.clientX) * k;
			this.lat = this.downLat + (ev.clientY - this.downY) * k;
			this.dragMoved = Math.max(this.dragMoved, Math.hypot(ev.clientX - this.downX, ev.clientY - this.downY));
		} else if (!this.interiorBusy) {
			this.updateHover(ev);
		}
	};

	private onPointerUp = (ev: PointerEvent) => {
		if (this.mode !== "interior") return;
		this.dragging = false;
		this.canvas.style.cursor = "";
		if (this.dragMoved < 5 && !this.interiorBusy) {
			const spot = pickByScreen(ev.clientX, ev.clientY, this.hotspotGroup, AUTO_AIM_PX, this.camera, this.canvas);
			if (spot) this.travelTo(spot.userData.targetIndex as number);
		}
	};

	private onWheel = (ev: WheelEvent) => {
		if (this.mode !== "interior") return;
		ev.preventDefault();
		this.camera.fov = MathUtils.clamp(this.camera.fov + ev.deltaY * 0.05, 25, 100);
		this.camera.updateProjectionMatrix();
	};

	private onClick = (ev: MouseEvent) => {
		if (this.mode !== "overview") return;
		const spot = pickByScreen(ev.clientX, ev.clientY, this.entryGroup, ENTRY_AIM_PX, this.camera, this.canvas);
		if (spot) this.enter(spot.userData.targetIndex as number);
	};

	private onWindowPointerUp = () => this.peekUp();
	private onKeyDown = (ev: KeyboardEvent) => {
		if (ev.code === "Space" && !ev.repeat) {
			ev.preventDefault();
			this.peekDown();
		}
	};
	private onKeyUp = (ev: KeyboardEvent) => {
		if (ev.code === "Space") this.peekUp();
	};

	private updateHover(ev: PointerEvent) {
		if (this.currentIndex < 0) return;
		const spot = pickByScreen(ev.clientX, ev.clientY, this.hotspotGroup, AUTO_AIM_PX, this.camera, this.canvas);
		const idx = spot ? (spot.userData.targetIndex as number) : -1;
		const occ = spot ? !!spot.userData.occluded : false;
		this.canvas.style.cursor = spot ? "pointer" : "";
		if (idx !== this.hoveredTargetIndex || occ !== this.hoveredOccluded) {
			this.hoveredTargetIndex = idx;
			this.hoveredOccluded = occ;
			this.emit();
		}
	}

	// --- view toggles (which geometry each mode shows) ------------------------

	private reskinProxy(mat: Material) {
		if (!this.proxyGroup) return;
		this.proxyGroup.traverse((o) => {
			const m = o as Mesh;
			if (m.isMesh) m.material = mat;
		});
	}

	private setOverviewView() {
		if (this.liteRoot) this.liteRoot.visible = true;
		if (this.proxyGroup) {
			if (this.sharedOverview) {
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
		this.you.group.visible = false;
	}

	private setInteriorView() {
		if (this.liteRoot) this.liteRoot.visible = false;
		if (this.proxyGroup) {
			if (this.sharedOverview) this.reskinProxy(this.projMaterial);
			this.proxyGroup.visible = this.projectionMode;
		}
		this.sphereA.visible = true;
		this.hotspotGroup.visible = true;
		this.entryGroup.visible = false;
		this.you.group.visible = false;
	}

	private setPeekView() {
		if (this.liteRoot) this.liteRoot.visible = true;
		if (this.proxyGroup) {
			if (this.sharedOverview) {
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
		this.you.group.visible = true;
	}

	// --- markers --------------------------------------------------------------

	// Is the straight line between two capture points blocked by the proxy? Our
	// "behind a wall/floor" test, trimmed at both ends so a hugged wall doesn't read
	// as occlusion.
	private anchorOccluded(fromPos: [number, number, number], toPos: [number, number, number]): boolean {
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
			if (this.projectionMode && d2 > HOTSPOT_REACH * HOTSPOT_REACH) continue;
			out.push([i, d2]);
		}
		out.sort((a, b) => a[1] - b[1]);
		return out.map((o) => o[0]);
	}

	private rebuildHotspots() {
		this.hotspotGroup.clear();
		if (this.currentIndex < 0) return;
		const cur = this.panos[this.currentIndex];
		let nVisible = 0;
		let nOccluded = 0;
		for (const i of this.neighborsByDistance()) {
			const occluded = this.anchorOccluded(cur.position, this.panos[i].position);
			if (occluded) {
				if (nOccluded >= HOTSPOT_MAX_OCCLUDED) continue;
				nOccluded++;
			} else {
				if (nVisible >= HOTSPOT_MAX_VISIBLE) continue;
				nVisible++;
			}
			const spot = makeDisc(i, occluded ? 0xe0c271 : 0xffffff, occluded ? 0xe0c271 : 0x9ad4ff);
			spot.userData.occluded = occluded;
			spot.position.fromArray(this.panos[i].position);
			spot.position.y -= HOTSPOT_FLOOR_DROP;
			this.hotspotGroup.add(spot);
			if (nVisible >= HOTSPOT_MAX_VISIBLE && nOccluded >= HOTSPOT_MAX_OCCLUDED) break;
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
		this.you.line.geometry.setFromPoints([new Vector3(p.x, floorY, p.z), p.clone()]);
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
		const order = this.panos.map((_, i) => i).sort((a, b) => this.camDist2[a] - this.camDist2[b]);
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
				(u.uCenter.value[k] as Vector3).fromArray(this.panos[idx].position);
				u.uWeight.value[k] = w[k] / wsum;
			} else {
				u.uTex.value[k] = DUMMY_TEX;
				u.uWeight.value[k] = 0;
			}
		}
		u.uCount.value = ready.length;
		this.sphereAMat.uniforms.map.value = ready.length ? this.panos[ready[0]].texture : DUMMY_TEX;
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

	// --- look controls (interior: lon/lat drag) -------------------------------

	private setLookFromForward(f: [number, number, number]) {
		const v = v3(f).normalize();
		this.lon = Math.atan2(v.z, v.x);
		this.lat = Math.asin(MathUtils.clamp(v.y, -1, 1));
	}
	private lookTargetFrom(pos: Vector3, lo: number, la: number): Vector3 {
		return pos
			.clone()
			.add(new Vector3(Math.cos(la) * Math.cos(lo), Math.sin(la), Math.cos(la) * Math.sin(lo)));
	}
	private applyLook() {
		this.lat = MathUtils.clamp(this.lat, -1.55, 1.55);
		this.camera.lookAt(this.lookTargetFrom(this.camera.position, this.lon, this.lat));
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
		this.canvas.style.cursor = "";
		this.emit(); // gated: holds the chrome while mode === "transition"
	}

	// --- travel between anchors (interior) ------------------------------------

	private travelTo(index: number) {
		if (this.interiorBusy || index === this.currentIndex || !this.panos[index]) return;
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
				this.sphereAMat.uniforms.map.value = target.texture ?? DUMMY_TEX;
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
			this.sphereAMat.uniforms.map.value = this.panos[index].texture ?? DUMMY_TEX;
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
		const fwd: [number, number, number] = p.forward && p.forward.length ? p.forward : [0, 0, 1];
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
		const toPos = this.sceneCenter.clone().addScaledVector(flat, this.sceneMaxDim * 1.5);
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
		this.startFly(s.pos.clone(), this.lookTargetFrom(s.pos, s.lon, s.lat), 800, {
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
		});
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
			const m = o as Mesh;
			if (!m.isMesh && !(o as { isLine?: boolean }).isLine) return;
			m.geometry?.dispose();
			const mats = Array.isArray(m.material) ? m.material : [m.material];
			for (const mat of mats) {
				if (mat && mat !== this.projMaterial && mat !== this.polyMaterial) mat.dispose();
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
		for (const p of this.panos) {
			p.texture?.dispose();
			if (p.placeholderTexture && p.placeholderTexture !== p.texture) p.placeholderTexture.dispose();
		}
		this.panos = [];
		this.sphereAMat.uniforms.map.value = DUMMY_TEX;
		this.sphereBMat.uniforms.map.value = DUMMY_TEX;
		this.currentIndex = -1;
		this.hotspotGroup.clear();
		this.entryGroup.clear();
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

			const list = manifest && Array.isArray(manifest.panos) ? manifest.panos : [];
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
					proxyRoot = await loadGLB(source.resolveProxy(manifest.proxy));
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
			this.showOverlay(`failed to load scene: ${e instanceof Error ? e.message : String(e)}`, {
				spinner: false,
				err: true,
			});
		}
	}

	private applyScene(entries: PanoEntry[], proxyRoot: Group | null, lite: Group | null) {
		this.panos = entries;
		this.projectionMode = !!proxyRoot;
		this.sharedOverview = !lite && !!proxyRoot; // no lite: the proxy doubles as the dollhouse

		if (!lite && !proxyRoot) {
			this.mode = "empty";
			this.showOverlay("nothing to show for this scene", { spinner: false, err: true });
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

		const framed = lite ?? proxyRoot!;
		const box = new Box3().setFromObject(framed);
		const size = box.getSize(new Vector3());
		box.getCenter(this.sceneCenter);
		this.sceneMaxDim = Math.max(size.x, size.y, size.z) || 1;

		this.camera.near = Math.max(0.02, this.sceneMaxDim * 0.002);
		this.camera.far = Math.max(500, this.sceneMaxDim * 60);

		this.sizeYouMarker();
		this.buildEntryMarkers();
		this.rebuildHotspots();

		const dist = this.sceneMaxDim * 1.6;
		this.browsePos.copy(this.sceneCenter).add(new Vector3(dist * 0.7, dist * 0.5, dist * 0.9));
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
		const dt = this.lastFrame ? Math.min(0.05, (time - this.lastFrame) / 1000) : 0;
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
			if (this.proxyGroup && this.proxyGroup.visible && this.projectionMode) this.updateProjection();
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
				this.camera.position.lerpVectors(g.fromPos, g.toPos, easeInOut(t));
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
			if (this.projectionMode) this.updateProjection();
			else this.sphereA.position.copy(this.camera.position);
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

		// Constant on-screen sizing + pulse for whichever marker group is showing.
		const pulse = 1 + 0.07 * Math.sin(time * 0.004);
		for (const group of [this.hotspotGroup, this.entryGroup]) {
			if (!group.visible) continue;
			const targetPx = group === this.entryGroup ? ENTRY_TARGET_PX : HOTSPOT_TARGET_PX;
			const hoverIdx = group === this.entryGroup ? this.hoveredEntryIndex : this.hoveredTargetIndex;
			for (const spot of group.children) {
				const occluded = !!spot.userData.occluded;
				const hovered = spot.userData.targetIndex === hoverIdx;
				const d = this.camera.position.distanceTo(spot.position);
				spot.scale.setScalar(
					hotspotScaleForDistance(d, targetPx, this.camera.fov, this.host.clientHeight) *
						(hovered ? 1.35 : 1) *
						(occluded ? 0.82 : 1),
				);
				const disc = spot.children[0] as Mesh;
				const ring = spot.children[1] as Mesh;
				ring.scale.setScalar(pulse);
				const discMat = disc.material as MeshBasicMaterial;
				const ringMat = ring.material as MeshBasicMaterial;
				if (occluded) {
					discMat.opacity = hovered ? 0.5 : 0.22;
					ringMat.opacity = hovered ? 0.8 : 0.45;
				} else {
					discMat.opacity = hovered ? 0.9 : 0.55;
					ringMat.opacity = hovered ? 1.0 : 0.85;
				}
			}
		}

		this.renderer.render(this.scene, this.camera);
	};
}
