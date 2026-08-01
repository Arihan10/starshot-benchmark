import type { PerspectiveCamera } from "three";

// The Gaussian-splat layer: a PlayCanvas canvas sitting UNDERNEATH the three.js
// one, rendering the cell's SOG-encoded trained splat as the scene's appearance.
//
// WHY A SECOND CANVAS. PlayCanvas' gsplat renderer needs its own GraphicsDevice,
// and a WebGL context cannot be shared across engines — so the splat cannot be a
// three.js object no matter how much tidier that would be. Two stacked canvases
// is the only shape available, and it turns out to be a good one: three.js keeps
// owning everything it is already good at (markers, cursor, outlines, the pano
// projection) and simply renders over a transparent background, with the splat
// showing through wherever nothing was drawn.
//
// FRAME LOCKSTEP IS THE WHOLE TRICK. PlayCanvas would happily run its own
// requestAnimationFrame loop, but then the two layers would present on
// independent schedules: the splat would show the camera pose from one frame
// while the cursor drawn over it showed the pose from another, and during any
// movement the overlay would visibly shear against the scene it is supposed to be
// glued to. So `autoRender` stays OFF for the life of the app and the engine
// calls `render()` from inside its own tick, right after copying the camera
// across. One camera, one frame, two canvases.
//
// COST WHEN UNUSED IS ZERO. The module is dynamically imported on first load, so
// a cell with no splat never downloads the PlayCanvas engine at all, and the
// canvas/context are never created. When the splat exists but is hidden (interior
// walkthrough), `render()` simply isn't called — the context idles.

export type SplatTransform = {
	position: [number, number, number];
	rotation: [number, number, number]; // Euler degrees, XYZ
	scale: number;
};

export const IDENTITY_TRANSFORM: SplatTransform = {
	position: [0, 0, 0],
	rotation: [0, 0, 0],
	scale: 1,
};

// Pure black, matching the host's CSS backdrop so the splat's empty space and the
// page agree. Four surfaces have to hold the same value for the seam to stay
// invisible — this canvas, the three.js scene background and inspect clear
// (engine.ts), and the viewer's own CSS (OrbitViewer / LoadingOverlay). Change one
// and you change them all.
const CLEAR = [0, 0, 0] as const;

type Pc = typeof import("playcanvas");

export class SplatLayer {
	private pc: Pc | null = null;
	private app: import("playcanvas").Application | null = null;
	private canvas: HTMLCanvasElement | null = null;
	private camera: import("playcanvas").Entity | null = null;
	private entity: import("playcanvas").Entity | null = null;
	private assetUrl: string | null = null;
	private transform: SplatTransform = { ...IDENTITY_TRANSFORM };
	private active = false;
	// Invalidated on every load/clear so an in-flight import or asset fetch that
	// lands after the scene changed is dropped instead of attaching to it.
	private token = 0;

	constructor(private readonly host: HTMLElement) {}

	/** Whether a splat is loaded and can be shown. */
	get ready(): boolean {
		return !!this.entity;
	}

	/**
	 * The layer's own canvas, for compositing a still of the panel.
	 *
	 * Readable ONLY inside the frame that drew it: the context is created without
	 * `preserveDrawingBuffer`, so the backbuffer is undefined once the browser has
	 * composited. See OrbitEngine.capture, which is why this is exposed at all.
	 */
	get canvasEl(): HTMLCanvasElement | null {
		return this.canvas;
	}

	get url(): string | null {
		return this.assetUrl;
	}

	/**
	 * Load `url` as the layer's splat, creating the PlayCanvas app on first use.
	 * Resolves false when the load was superseded or failed — the caller carries
	 * on without a splat rather than failing the scene, since the dollhouse and
	 * the walkthrough are both perfectly usable on their own.
	 */
	async load(url: string): Promise<boolean> {
		this.clear();
		const token = ++this.token;
		try {
			if (!this.pc) this.pc = await import("playcanvas");
			if (token !== this.token) return false;
			const pc = this.pc;
			if (!this.app) this.create(pc);
			const app = this.app;
			if (!app) return false;

			const asset = await new Promise<import("playcanvas").Asset>(
				(resolve, reject) => {
					app.assets.loadFromUrl(url, "gsplat", (err, a) =>
						err || !a ? reject(new Error(String(err))) : resolve(a),
					);
				},
			);
			if (token !== this.token) {
				asset.unload();
				return false;
			}
			// `app` EXPLICITLY. `new pc.Entity(name)` defaults its owning app to
			// PlayCanvas' module-global "current application" — the last one
			// constructed anywhere on the page — and `addComponent` resolves the
			// component system off THAT app. With two viewers side by side the
			// awaits above interleave: both apps exist by the time either asset
			// finishes, so one splat ends up with a gsplat component belonging to
			// the other viewer's app while sitting in this one's scene graph, and
			// silently never draws. Naming the app removes the global from the
			// path entirely. Same reason the camera below passes it.
			const entity = new pc.Entity("splat", app);
			entity.addComponent("gsplat", { asset });
			app.root.addChild(entity);
			this.entity = entity;
			this.assetUrl = url;
			this.applyTransform();
			this.resize();
			return true;
		} catch (e) {
			console.warn("[splat] load failed", e);
			return false;
		}
	}

	private create(pc: Pc) {
		const canvas = document.createElement("canvas");
		Object.assign(canvas.style, {
			position: "absolute",
			inset: "0",
			width: "100%",
			height: "100%",
			// Under the three.js canvas (which the engine pins to z-index 1) but
			// above the host's own background.
			zIndex: "0",
			// VISIBILITY, not display. PlayCanvas re-derives its backbuffer size from
			// the canvas's clientWidth/clientHeight at the top of every render (see
			// AppBase.render → updateCanvasSize under RESOLUTION_AUTO), and a
			// `display: none` canvas reports 0×0 — which would hand the graphics
			// device a zero-sized backbuffer. `visibility: hidden` keeps the element
			// laid out, so those measurements stay honest whether it is on screen or
			// not.
			visibility: "hidden",
			pointerEvents: "none", // input belongs to the three.js canvas above
		});
		// Prepended so it can never sit above the overlays the engine appends.
		this.host.insertBefore(canvas, this.host.firstChild);
		this.canvas = canvas;

		const app = new pc.Application(canvas, {
			graphicsDeviceOptions: { antialias: false },
		});
		app.setCanvasFillMode(pc.FILLMODE_NONE);
		app.setCanvasResolution(pc.RESOLUTION_AUTO);
		app.graphicsDevice.maxPixelRatio = window.devicePixelRatio || 1;
		app.start();
		// See the lockstep note at the top: the engine drives render() itself.
		app.autoRender = false;

		const camera = new pc.Entity("splat-camera", app);
		camera.addComponent("camera", {
			clearColor: new pc.Color(CLEAR[0], CLEAR[1], CLEAR[2], 1),
			fov: 75,
			nearClip: 0.05,
			farClip: 2000,
		});
		app.root.addChild(camera);

		this.app = app;
		this.camera = camera;
	}

	/**
	 * The splat's placement in world space. Trainers that renormalize the scene on
	 * ingest (Postshot does) hand back a splat whose origin is nowhere near the
	 * world the walkthrough lives in; this is where that gets corrected while the
	 * numbers are still being found. Once a correction is confirmed it belongs
	 * baked into the asset (tools/splat-to-web-sog.mjs --translate) so every
	 * consumer gets it, and this returns to identity.
	 */
	setTransform(t: SplatTransform) {
		this.transform = t;
		this.applyTransform();
	}

	getTransform(): SplatTransform {
		return this.transform;
	}

	private applyTransform() {
		const e = this.entity;
		if (!e) return;
		const [px, py, pz] = this.transform.position;
		const [rx, ry, rz] = this.transform.rotation;
		e.setLocalPosition(px, py, pz);
		e.setLocalEulerAngles(rx, ry, rz);
		e.setLocalScale(this.transform.scale, this.transform.scale, this.transform.scale);
	}

	/** Show or hide the layer. Hidden costs nothing: `render` becomes a no-op. */
	setActive(on: boolean) {
		const next = on && !!this.entity;
		if (next === this.active) return;
		this.active = next;
		if (this.canvas) this.canvas.style.visibility = next ? "visible" : "hidden";
	}

	get isActive(): boolean {
		return this.active;
	}

	/**
	 * Copy the three.js camera across and draw one frame. Called from the engine's
	 * tick BEFORE it renders its own layer, so both present the same pose.
	 * PlayCanvas and three.js share a convention here — right-handed, Y-up,
	 * looking down local -Z — so the quaternion transfers without correction.
	 */
	render(cam: PerspectiveCamera) {
		if (!this.active || !this.app || !this.camera) return;
		// A render with no layout size would resize the backbuffer to 0×0 and throw
		// the frame away — and the panel this lives in genuinely reaches zero size
		// (the A/B workspace collapses one side outright). Skipping is free; the next
		// frame with real dimensions picks straight back up.
		if (!this.canvas?.clientWidth || !this.canvas.clientHeight) return;
		const c = this.camera.camera;
		if (c) {
			c.fov = cam.fov; // three's fov is vertical; so is PlayCanvas' default
			c.nearClip = cam.near;
			c.farClip = cam.far;
		}
		this.camera.setPosition(cam.position.x, cam.position.y, cam.position.z);
		this.camera.setRotation(
			cam.quaternion.x,
			cam.quaternion.y,
			cam.quaternion.z,
			cam.quaternion.w,
		);
		// `render()` is marked @ignore in PlayCanvas' typings — internal, not part of
		// the supported surface — so the version is pinned exactly and this degrades
		// rather than throws if a future one drops it. It is used anyway because it
		// is the only way to draw SYNCHRONOUSLY, here, with the camera we just set.
		// The public alternative (`renderNextFrame = true`) defers to PlayCanvas'
		// own requestAnimationFrame callback, which may run either side of ours —
		// and a splat drawn from last frame's pose with this frame's cursor on top
		// shears visibly the moment anything moves, which is the one artefact this
		// whole layer exists to avoid.
		const app = this.app as { render?: () => void; renderNextFrame?: boolean };
		if (typeof app.render === "function") app.render();
		else app.renderNextFrame = true;
	}

	resize() {
		if (!this.app) return;
		const w = this.host.clientWidth;
		const h = this.host.clientHeight;
		if (w > 0 && h > 0) this.app.resizeCanvas(w, h);
	}

	/** Drop the current splat, keeping the app alive for the next scene. */
	clear() {
		this.token++;
		// Routed through setActive rather than reaching for the style directly, so
		// exactly ONE place decides how this canvas is hidden. Two places disagreeing
		// is not hypothetical: this method used to hide it with `display: none` while
		// setActive showed it again with `visibility`, which left the element
		// undisplayed forever and the splat permanently invisible.
		this.setActive(false);
		if (this.entity) {
			this.entity.destroy();
			this.entity = null;
		}
		if (this.app) this.app.assets.list().forEach((a) => a.unload());
		this.assetUrl = null;
	}

	dispose() {
		this.clear();
		const app = this.app;
		this.app = null;
		this.camera = null;
		if (app) {
			try {
				app.destroy(); // tears down entities, assets and the GL context
			} catch {
				/* destroying a partially-built app can throw; nothing left to save */
			}
		}
		this.canvas?.remove();
		this.canvas = null;
		this.pc = null;
	}
}
