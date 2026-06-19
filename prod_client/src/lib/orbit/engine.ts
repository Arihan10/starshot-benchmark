import {
	Box3,
	Color,
	DirectionalLight,
	Group,
	HemisphereLight,
	type Intersection,
	type Material,
	MathUtils,
	Mesh,
	MOUSE,
	type Object3D,
	NoToneMapping,
	PerspectiveCamera,
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
	ENTRY_AIM_PX,
	PEEK_ROTATE_SPEED,
	pickByScreen,
	WASD_DIR_COS,
	WASD_MAX_STEP,
	WASD_MAX_Y_STEP,
} from "./markers";
import { SurfaceCursor } from "./cursor";
import { MarkerLayer } from "./markerLayer";
import { collectObjects, ObjectAddressing } from "./objectAddressing";
import { type PanoEntry, PanoStreamer } from "./panoTextures";
import { Projection } from "./projection";
import { buildMinimapState, levelForY, type MinimapSlice } from "./minimap";
import {
	applyLook,
	cursorRayDir,
	forwardToLonLat,
	lookTargetFrom,
	pinLook,
} from "./look";
import type { OrbitMode, OrbitState, TourManifest, TourSource } from "./types";

const v3 = (a: [number, number, number]) => new Vector3().fromArray(a);
const easeInOut = (t: number) =>
	t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
const _cursorNdc = new Vector2(); // scratch: native cursor in NDC for raycastInterior
const _labelPos = new Vector3(); // scratch: project a hovered ring's anchor to screen

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
	private readonly destLabel: HTMLDivElement; // floating name tag above a hovered gold ring
	private readonly scene: Scene;
	private readonly camera: PerspectiveCamera;
	private readonly controls: OrbitControls;
	private readonly ro: ResizeObserver;

	// Post-processing: the beauty pass plus two OutlinePasses (owned by
	// ObjectAddressing) that silhouette the selected (orange) / hovered (cyan)
	// objects, then a copy to screen.
	private readonly composer: EffectComposer;

	private readonly sphereA: Mesh;
	private readonly sphereAMat: ShaderMaterial;
	private readonly sphereB: Mesh;
	private readonly sphereBMat: ShaderMaterial;
	private readonly polyMaterial = makePolyMaterial();

	// Subsystems: pano texture streaming, the projection backdrop, the navigation
	// marker layer, and per-object addressing. The engine wires them, routes
	// input, and runs the camera / render loop.
	private readonly streamer: PanoStreamer;
	private readonly projection = new Projection();
	private readonly markers: MarkerLayer;
	private readonly addressing: ObjectAddressing;
	private readonly requestPano = (i: number) => this.streamer.request(i);

	private readonly dummyCam = new PerspectiveCamera();

	// Surface-adhering ring cursor (interior only); see SurfaceCursor. The raycast
	// that finds the point under the cursor is shared with click auto-aim, so it
	// (cursorRay) stays here and the resulting hit is fed to the cursor each frame.
	private readonly cursor: SurfaceCursor;
	private readonly cursorRay = new Raycaster();
	private pointerClientX = 0;
	private pointerClientY = 0;
	private pointerInside = false;

	private currentIndex = -1;
	private projectionMode = false;
	// Bird's-eye minimap slices (one per Y level) + the level each pano sits on
	// (its nearest minimap by capture height). Empty when the tour has no slices.
	private minimaps: MinimapSlice[] = [];
	private panoLevel: number[] = [];
	// Held image prefetchers for the slices, so every floor is cached up front and
	// the floor switcher is instant. Kept referenced so the in-flight loads aren't
	// GC'd; dropped on the next clearScene.
	private minimapPrefetch: HTMLImageElement[] = [];
	private liteRoot: Group | null = null;
	private proxyGroup: Group | null = null;
	private sharedOverview = false;
	private proxyView = false; // overview shows the proxy mesh instead of the lite dollhouse
	// One matte material per proxy object so the bare proxy reads as distinct
	// parts; reskinProxy's matte path swaps these in, clearScene disposes them.
	private proxyColorMats: Material[] = [];
	private rcDownX = 0;
	private rcDownY = 0;

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
	// The world direction under the cursor when a look-drag begins. pinLook turns
	// the camera so this stays welded under the cursor as you drag (see pinLook).
	private readonly grabDir = new Vector3();
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

		// AA comes from the composer's multisampled buffer, so the default
		// framebuffer doesn't need it (the final pass is a fullscreen blit).
		this.renderer = new WebGLRenderer({ antialias: false });
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

		// Floating tag shown above a hovered gold (behind-wall) ring; positioned every
		// frame from the anchor's world position (positionDestLabel) so it tracks look.
		this.destLabel = document.createElement("div");
		Object.assign(this.destLabel.style, {
			position: "absolute",
			display: "none",
			transform: "translate(-50%, -100%)",
			padding: "2px 8px",
			borderRadius: "6px",
			border: "1px solid rgba(255,206,115,0.35)",
			background: "rgba(12,13,16,0.78)",
			color: "#ffd98a",
			font: "600 11px ui-sans-serif, system-ui, sans-serif",
			whiteSpace: "nowrap",
			pointerEvents: "none",
			zIndex: "2",
		});
		host.appendChild(this.destLabel);

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

		// Outline pipeline. The working buffer is sRGB + multisampled so the
		// composite matches the direct-render look: the pano / projection shaders
		// aren't colour-managed, so a linear buffer + OutputPass would re-encode
		// (shift) their colours — an sRGB buffer copied verbatim avoids that while
		// keeping MSAA. RenderPass → select outline → hover outline → copy to screen.
		const composerRT = new WebGLRenderTarget(1, 1, { samples: 4 });
		composerRT.texture.colorSpace = SRGBColorSpace;
		this.composer = new EffectComposer(this.renderer, composerRT);
		this.composer.setPixelRatio(this.renderer.getPixelRatio());
		this.composer.addPass(new RenderPass(this.scene, this.camera));
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
		for (const pass of this.composer.passes) pass.dispose();
		this.composer.dispose();
		this.renderer.dispose();
		this.canvas.remove();
		this.travelFade.remove();
		this.destLabel.remove();
	}

	// The pano list is owned by the streamer; the engine reads positions / ids /
	// textures straight off it.
	private get panos(): PanoEntry[] {
		return this.streamer.list;
	}

	// --- state emission (gated so chrome holds through camera flights) --------

	private emit() {
		if (this.mode === "transition") return;
		const cur =
			this.currentIndex >= 0 ? this.panos[this.currentIndex] : null;
		let hover: OrbitState["hover"] = null;
		if (this.mode === "overview" && this.hoveredEntryIndex >= 0) {
			const p = this.panos[this.hoveredEntryIndex];
			hover = { id: p.id, name: p.name, occluded: false };
		} else if (this.mode === "interior" && this.hoveredTargetIndex >= 0) {
			const p = this.panos[this.hoveredTargetIndex];
			hover = { id: p.id, name: p.name, occluded: this.hoveredOccluded };
		}
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
		this.composer.setSize(w, h);
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
		if (this.mode !== "interior" || this.interiorBusy) return;
		this.dragging = true;
		this.dragMoved = 0;
		this.downX = ev.clientX;
		this.downY = ev.clientY;
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
				this.markers.entryGroup,
				ENTRY_AIM_PX,
				this.camera,
				this.canvas,
			);
			const entryIdx = spot ? (spot.userData.targetIndex as number) : -1;
			// The object highlight is independent of the entry discs: a disc under the
			// cursor no longer suppresses it, so the object beneath still highlights
			// (when the toggle is on).
			const obj = this.highlightEnabled
				? this.addressing.pickAt(
						ev.clientX,
						ev.clientY,
						this.activeObjectRoot(),
					)
				: null;
			this.canvas.style.cursor = entryIdx >= 0 || obj ? "pointer" : "";
			const hoverChanged = this.addressing.setHover(obj);
			const entryChanged = entryIdx !== this.hoveredEntryIndex;
			this.hoveredEntryIndex = entryIdx;
			if (entryChanged || hoverChanged) this.emit();
			return;
		}
		if (this.mode !== "interior") return;
		if (this.dragging) {
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
		const spot = this.markers.pickAnchorMarker(
			ev.clientX,
			ev.clientY,
			this.camera,
			this.canvas,
			this.panos,
			this.currentIndex,
			this.proxyGroup,
		);
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
			this.markers.entryGroup,
			ENTRY_AIM_PX,
			this.camera,
			this.canvas,
		);
		if (spot) this.enter(spot.userData.targetIndex as number);
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

	// Interior hover: the anchor markers and the object highlight are independent — a
	// marker under the cursor no longer suppresses the highlight, so the proxy object
	// beneath still tints (when the toggle is on). The fill overlay is depth-test-off,
	// so it reads as picking the object straight out of the projected 360 image.
	private updateHover(ev: PointerEvent) {
		if (this.currentIndex < 0) return;
		const spot = this.markers.pickAnchorMarker(
			ev.clientX,
			ev.clientY,
			this.camera,
			this.canvas,
			this.panos,
			this.currentIndex,
			this.proxyGroup,
		);
		this.markers.setRingHover(spot as Mesh | null);
		const idx = spot ? (spot.userData.targetIndex as number) : -1;
		const occ = spot ? !!spot.userData.occluded : false;
		const obj = this.highlightEnabled
			? this.addressing.pickAt(
					ev.clientX,
					ev.clientY,
					this.activeObjectRoot(),
				)
			: null;
		this.canvas.style.cursor = idx >= 0 || obj ? "pointer" : "";
		const hotspotChanged =
			idx !== this.hoveredTargetIndex || occ !== this.hoveredOccluded;
		this.hoveredTargetIndex = idx;
		this.hoveredOccluded = occ;
		if (this.addressing.setHover(obj) || hotspotChanged) this.emit();
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
			if (this.projection.proxyBase)
				targets.push(this.projection.proxyBase);
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

	// Place the surface cursor each frame (runs in tick, so it follows both pointer
	// motion and look-drag rotation). Shown only in the interior when not busy, the
	// pointer's inside, and it isn't already on an anchor ring (hoveredRing) — that
	// ring lights up instead, so two rings never stack. The interior raycast is
	// shared with click auto-aim; the hit (or null) is handed to the cursor.
	private updateCursorRing() {
		const active =
			this.mode === "interior" &&
			!this.interiorBusy &&
			this.pointerInside &&
			!this.markers.hoveredRing;
		const hit = active
			? this.raycastInterior(this.pointerClientX, this.pointerClientY)
			: null;
		this.cursor.update(hit, this.camera, this.host.clientHeight);
	}

	// The destination tag floats above a hovered gold (behind-wall) ring: project the
	// anchor's eye-height world position (above the floor ring) to screen and place
	// the tag there. Only gold rings get it — the room they lead to is hidden, so
	// naming it helps. Hidden otherwise.
	private positionDestLabel() {
		const ring = this.markers.hoveredRing;
		if (this.mode !== "interior" || !ring || !ring.userData.occluded) {
			this.destLabel.style.display = "none";
			return;
		}
		const p = this.panos[ring.userData.targetIndex as number];
		if (!p) {
			this.destLabel.style.display = "none";
			return;
		}
		this.camera.updateMatrixWorld();
		_labelPos.fromArray(p.position).project(this.camera);
		if (_labelPos.z > 1) {
			this.destLabel.style.display = "none";
			return;
		}
		this.destLabel.textContent = p.name ?? p.id;
		this.destLabel.style.left = `${(_labelPos.x * 0.5 + 0.5) * this.host.clientWidth}px`;
		this.destLabel.style.top = `${(-_labelPos.y * 0.5 + 0.5) * this.host.clientHeight}px`;
		this.destLabel.style.display = "block";
	}

	// --- view toggles (which geometry each mode shows) ------------------------

	private reskinProxy(mat: Material) {
		if (!this.proxyGroup) return;
		// Matte skin: each object wears its own color (colorProxyObjects); the
		// projection skin is one shared shader for all.
		const matte = mat === this.polyMaterial;
		this.proxyGroup.traverse((o) => {
			const m = o as Mesh;
			if (!m.isMesh) return;
			m.material = matte
				? ((m.userData.colorMat as Material) ?? mat)
				: mat;
		});
		// The base mirrors the proxy: panos project onto it in the walkthrough (so
		// it blends into the floor instead of showing through as a flat fill), and
		// it goes matte in proxy view. It's a neutral backing slab (not an
		// addressable object), so it keeps the shared material, not a per-object color.
		this.projection.setBaseMaterial(mat);
	}

	// Give each proxy object its own matte color so the bare proxy reads as
	// distinct parts instead of one gray blob. Clones inherit polyMaterial's
	// flat-shaded look; hues are spread by golden-ratio stepping so neighbors
	// never collide. The colors ride in via reskinProxy's matte path; clearScene
	// disposes the clones.
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
		this.markers.hotspotGroup.visible = false;
		this.markers.entryGroup.visible = true;
		this.markers.anchorRingGroup.visible = false;
		this.markers.you.group.visible = false;
		this.projection.syncBase(!!this.proxyGroup?.visible);
	}

	// Swap the interior proxy in place between the bare low-poly mesh (proxy view:
	// flat matte, no pano) and the captured panos projected onto it. The backdrop
	// sphere is the projected sky, so it's off in proxy view. Safe to call
	// mid-walkthrough — it leaves the navigation markers alone.
	private setInteriorProxyView() {
		if (!this.proxyGroup || !this.projectionMode) return;
		this.reskinProxy(
			this.proxyView ? this.polyMaterial : this.projection.material,
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
		this.markers.hotspotGroup.visible = true;
		this.markers.entryGroup.visible = false;
		this.markers.anchorRingGroup.visible = true;
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
		this.markers.hotspotGroup.visible = false;
		this.markers.entryGroup.visible = false;
		this.markers.anchorRingGroup.visible = false;
		this.markers.you.group.visible = true;
		this.projection.syncBase(!!this.proxyGroup?.visible);
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
		this.addressing.setHover(null);
		this.addressing.closeMenu();
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
		if (!this.highlightEnabled) this.addressing.setHover(null);
		this.canvas.style.cursor = "";
		this.emit();
	}

	// --- per-object addressing (pick / hide / outline) ------------------------

	// Which loaded root the cursor addresses right now: the dollhouse in overview /
	// peek (lite, or the proxy when "proxy view" is on or there's no lite), and the
	// projected proxy in the interior (hover-highlight — see updateHover). The
	// hover / hide / outline mechanics live in ObjectAddressing.
	private activeObjectRoot(): Object3D | null {
		if (this.mode === "overview" || this.mode === "peek") {
			if (this.proxyView && this.proxyGroup) return this.proxyGroup;
			return this.liteRoot ?? this.proxyGroup;
		}
		if (this.mode === "interior") return this.proxyGroup;
		return null;
	}

	// --- right-click object menu (delegated to ObjectAddressing) --------------

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

	// --- projection (view-dependent texture mapping) -------------------------

	// Per-frame VDTM blend, delegated to Projection (which owns the shader and the
	// backdrop sphere sizing). Skipped while proxy view shows the bare geometry.
	private updateProjection() {
		this.projection.update(
			this.camera,
			this.panos,
			this.requestPano,
			this.sphereA,
			this.sphereAMat,
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
		this.markers.setRingHover(null);
		this.addressing.setHover(null);
		this.addressing.closeMenu();
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
		this.markers.setRingHover(null);
		this.interiorBusy = true;
		this.markers.hotspotGroup.visible = false;

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
		void this.streamer.ensure(index).then(() => {
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
				this.markers.hotspotGroup.visible = true;
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
		this.markers.rebuildHotspots(
			this.panos,
			this.currentIndex,
			this.proxyGroup,
			this.projectionMode,
		);
		this.emit();
	}

	// The streamer owns lazy LQIP→full loading; this fires when the current
	// capture's texture lands so sphere mode can refresh its backdrop (projection
	// mode re-reads textures every frame).
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
				const look = forwardToLonLat(fwd);
				this.lon = look.lon;
				this.lat = look.lat;
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
		this.markers.positionYouMarker(userPos);
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
			lookTargetFrom(s.pos, s.lon, s.lat),
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
			const m = o as Mesh;
			if (!m.isMesh && !(o as { isLine?: boolean }).isLine) return;
			m.geometry?.dispose();
			const mats = Array.isArray(m.material) ? m.material : [m.material];
			for (const mat of mats) {
				// Shared singletons (projection / poly fills) outlive any one cell, so
				// never dispose them here.
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
		// Per-object proxy colors are detached in projection mode (so disposeObject
		// can't reach them), so drop them explicitly.
		for (const m of this.proxyColorMats) m.dispose();
		this.proxyColorMats = [];
		this.projection.clearBase(this.scene);
		this.streamer.reset();
		this.minimaps = [];
		this.panoLevel = [];
		this.minimapPrefetch = [];
		this.sphereAMat.uniforms.map.value = DUMMY_TEX;
		this.sphereBMat.uniforms.map.value = DUMMY_TEX;
		this.currentIndex = -1;
		this.markers.clear();
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
		this.addressing.reset();
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
					name: p.name,
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
		this.streamer.reset(entries);
		this.panoLevel = entries.map((p) =>
			levelForY(this.minimaps, p.position[1]),
		);
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
			this.projection.setup(proxyRoot, this.sphereA, this.sphereAMat);
			this.proxyGroup = proxyRoot;
			this.scene.add(proxyRoot);
		}

		// Tag each placed object in both roots so they can be hovered / hidden /
		// outlined individually (independently per scene — lite and proxy nodes
		// don't share identity).
		if (this.liteRoot) this.addressing.register(this.liteRoot);
		if (this.proxyGroup) {
			this.addressing.register(this.proxyGroup);
			this.colorProxyObjects();
			// Give proxy floor leaks an opaque backing (a base under its footprint).
			this.projection.buildBase(this.proxyGroup, this.scene);
		}

		const framed = lite ?? proxyRoot!;
		const box = new Box3().setFromObject(framed);
		const size = box.getSize(new Vector3());
		box.getCenter(this.sceneCenter);
		this.sceneMaxDim = Math.max(size.x, size.y, size.z) || 1;

		this.camera.near = Math.max(0.02, this.sceneMaxDim * 0.002);
		this.camera.far = Math.max(500, this.sceneMaxDim * 60);

		this.markers.build(this.panos, this.sceneMaxDim);
		this.markers.rebuildHotspots(
			this.panos,
			this.currentIndex,
			this.proxyGroup,
			this.projectionMode,
		);

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
					this.markers.hotspotGroup.visible = true;
					this.activate(idx);
				}
			}
			if (this.projectionMode) {
				if (!this.proxyView) this.updateProjection(); // proxy view shows bare geometry, no panos
			} else {
				this.sphereA.position.copy(this.camera.position);
			}
			this.lat = applyLook(this.camera, this.lon, this.lat);
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
		this.positionDestLabel();
		this.markers.updateEntryDiscs(
			this.camera,
			this.host.clientHeight,
			this.hoveredEntryIndex,
			time,
		);
		this.addressing.updateOutlines();
		this.composer.render();
	};
}
