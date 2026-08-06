import type { PerspectiveCamera } from "three";

export type SplatTransform = {
	position: [number, number, number];
	rotation: [number, number, number];
	scale: number;
};

export const IDENTITY_TRANSFORM: SplatTransform = {
	position: [0, 0, 0],
	rotation: [0, 0, 0],
	scale: 1,
};

const CLEAR = [0, 0, 0, 0] as const;

type Pc = typeof import("playcanvas");

type Staged = {
	entity: import("playcanvas").Entity;
	asset: import("playcanvas").Asset;
	url: string;
};

export class SplatLayer {
	private pc: Pc | null = null;
	private app: import("playcanvas").Application | null = null;
	private canvas: HTMLCanvasElement | null = null;
	private camera: import("playcanvas").Entity | null = null;
	private entity: import("playcanvas").Entity | null = null;
	private asset: import("playcanvas").Asset | null = null;
	private staged: Staged | null = null;
	private assetUrl: string | null = null;
	private transform: SplatTransform = { ...IDENTITY_TRANSFORM };
	private active = false;
	private token = 0;

	constructor(private readonly host: HTMLElement) {}

	get ready(): boolean {
		return !!this.entity;
	}

	get canvasEl(): HTMLCanvasElement | null {
		return this.canvas;
	}

	get url(): string | null {
		return this.assetUrl;
	}

	async prepare(url: string): Promise<boolean> {
		this.discardStaged();
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
			if (!asset.resource) {
				app.assets.remove(asset);
				asset.unload();
				console.warn("[splat] no resource for", url);
				return false;
			}
			const entity = new pc.Entity("splat", app);
			entity.addComponent("gsplat", { asset });
			entity.enabled = false;
			app.root.addChild(entity);
			this.staged = { entity, asset, url };
			return true;
		} catch (e) {
			console.warn("[splat] load failed", e);
			return false;
		}
	}

	commit() {
		const next = this.staged;
		this.staged = null;
		this.dropActive();
		if (!next) return;
		this.entity = next.entity;
		this.asset = next.asset;
		this.assetUrl = next.url;
		next.entity.enabled = true;
		this.applyTransform();
		this.resize();
	}

	discardStaged() {
		const s = this.staged;
		this.staged = null;
		if (!s) return;
		s.entity.destroy();
		s.asset.unload();
	}

	private dropActive() {
		this.setActive(false);
		if (this.entity) {
			this.entity.destroy();
			this.entity = null;
		}
		if (this.asset) {
			this.asset.unload();
			this.asset = null;
		}
		this.assetUrl = null;
	}

	private create(pc: Pc) {
		const canvas = document.createElement("canvas");
		Object.assign(canvas.style, {
			position: "absolute",
			inset: "0",
			width: "100%",
			height: "100%",
			zIndex: "0",
			visibility: "hidden",
			pointerEvents: "none",
		});
		this.host.insertBefore(canvas, this.host.firstChild);
		this.canvas = canvas;

		const app = new pc.Application(canvas, {
			graphicsDeviceOptions: { antialias: false, alpha: true },
		});
		app.setCanvasFillMode(pc.FILLMODE_NONE);
		app.setCanvasResolution(pc.RESOLUTION_AUTO);
		app.graphicsDevice.maxPixelRatio = window.devicePixelRatio || 1;
		app.start();
		app.autoRender = false;

		const camera = new pc.Entity("splat-camera", app);
		camera.addComponent("camera", {
			clearColor: new pc.Color(...CLEAR),
			fov: 75,
			nearClip: 0.05,
			farClip: 2000,
		});
		app.root.addChild(camera);

		this.app = app;
		this.camera = camera;
	}

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

	setActive(on: boolean) {
		const next = on && !!this.entity;
		if (next === this.active) return;
		this.active = next;
		if (this.canvas) this.canvas.style.visibility = next ? "visible" : "hidden";
	}

	get isActive(): boolean {
		return this.active;
	}

	render(cam: PerspectiveCamera) {
		if (!this.active || !this.app || !this.camera) return;
		if (!this.canvas?.clientWidth || !this.canvas.clientHeight) return;
		const c = this.camera.camera;
		if (c) {
			c.fov = cam.fov;
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

	clear() {
		this.token++;
		this.dropActive();
	}

	dispose() {
		this.discardStaged();
		this.clear();
		const app = this.app;
		this.app = null;
		this.camera = null;
		if (app) {
			try {
				app.destroy();
			} catch {
			}
		}
		this.canvas?.remove();
		this.canvas = null;
		this.pc = null;
	}
}
