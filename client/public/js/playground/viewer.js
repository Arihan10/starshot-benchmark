// SOG-LOD playground viewer — a PlayCanvas app that renders a Gaussian splat and,
// when handed a streamed-SOG bundle (lod-meta.json), lets the engine's native LOD
// system STREAM per-octree-node level-of-detail over the network: coarse chunks
// load first, then refine by camera distance under a splat budget, with unused
// chunks unloaded after a cooldown. A single .sog still loads (no LOD).
//
// The engine does the streaming; this module owns the app, the birdseye camera,
// the LOD/streaming knobs (surfaced to the UI), and the live stats HUD feed.

import {
    Application,
    Color,
    Entity,
    FILLMODE_NONE,
    RESOLUTION_AUTO,
    GSPLAT_DEBUG_NONE,
    GSPLAT_DEBUG_LOD,
    GSPLAT_DEBUG_AABBS,
    GSPLAT_DEBUG_NODE_AABBS,
    GSPLAT_DEBUG_HEATMAP,
} from "playcanvas";

import { OrbitFlyControls } from "./controls.js";

export const DEBUG_MODES = {
    none: GSPLAT_DEBUG_NONE,
    lod: GSPLAT_DEBUG_LOD,
    aabbs: GSPLAT_DEBUG_AABBS,
    nodes: GSPLAT_DEBUG_NODE_AABBS,
    heatmap: GSPLAT_DEBUG_HEATMAP,
};

export class SplatViewer {
    constructor(canvas) {
        this.canvas = canvas;
        this.splat = null; // the gsplat Entity
        this.asset = null;
        this.octree = null; // GSplatOctree when streaming LOD, else null
        this.center = null; // scene AABB centre (for the HUD's camera-distance)
        this.frameMs = 16; // smoothed
        this.bytes = 0; // network bytes for the ACTIVE asset (via PerformanceObserver)
        this._assetBase = null;

        const app = new Application(canvas, {
            graphicsDeviceOptions: { antialias: false },
        });
        app.setCanvasFillMode(FILLMODE_NONE);
        app.setCanvasResolution(RESOLUTION_AUTO);
        app.graphicsDevice.maxPixelRatio = window.devicePixelRatio || 1;
        app.start();
        this.app = app;

        const cam = new Entity("camera");
        cam.addComponent("camera", {
            clearColor: new Color(0.043, 0.05, 0.063, 1),
            fov: 55,
            nearClip: 0.05,
            farClip: 5000,
        });
        app.root.addChild(cam);
        this.cam = cam;

        this.controls = new OrbitFlyControls(canvas, cam, { speed: 4 });
        this.controls.attach();

        app.on("update", (dt) => {
            this.controls.update(dt);
            this.frameMs += ((dt * 1000) - this.frameMs) * 0.1; // ~EMA smoothing
        });

        const fit = () => {
            const r = canvas.parentElement.getBoundingClientRect();
            if (r.width > 0 && r.height > 0) app.resizeCanvas(r.width, r.height);
        };
        fit();
        this._resizeObs = new ResizeObserver(fit);
        this._resizeObs.observe(canvas.parentElement);

        // Network accounting for the active asset: sum transfer sizes of resource
        // fetches under the asset's base path — this is what makes streaming
        // visible (bytes climb as you move closer and finer chunks page in).
        this._perf = new PerformanceObserver((list) => {
            for (const e of list.getEntries()) {
                if (this._assetBase && e.name.includes(this._assetBase)) {
                    this.bytes += e.transferSize || e.encodedBodySize || 0;
                }
            }
        });
        try {
            this._perf.observe({ type: "resource", buffered: true });
        } catch {
            this._perf = null; // Safari <16 etc. — HUD just omits the byte counter
        }
    }

    // Load a splat by URL. `lod-meta.json` → streamed LOD; anything else → single
    // splat. Resolves with a summary once the resource is ready + framed.
    async load(url) {
        this._clearSplat();
        this.bytes = 0;
        // Base path used to attribute streamed chunk fetches to this asset.
        this._assetBase = url.replace(/[^/]*$/, ""); // dir of the manifest/sog
        if (!/\/$/.test(this._assetBase)) this._assetBase = url;

        const asset = await new Promise((resolve, reject) => {
            this.app.assets.loadFromUrl(url, "gsplat", (err, a) =>
                err ? reject(new Error(String(err))) : resolve(a),
            );
        });
        this.asset = asset;

        const splat = new Entity("splat");
        splat.addComponent("gsplat", { asset });
        this.app.root.addChild(splat);
        this.splat = splat;

        const res = asset.resource;
        this.octree = res.octree ?? null;
        const aabb = res.aabb;
        this.center = aabb.center.clone();
        this._applyLodConfig();
        this._frameCamera(aabb);

        return {
            type: this.octree ? "lod" : "single",
            lodLevels: this.octree ? this.octree.lodLevels : 1,
            splats: res.numSplats ?? 0,
            filesTotal: this.octree ? this.octree.files.length : 1,
        };
    }

    // Fit near/far to the scene and drop the camera into the birdseye entry pose.
    _frameCamera(aabb, opts) {
        const { dist, radius } = this.controls.frame(aabb, opts);
        this.cam.camera.nearClip = Math.max(0.02, radius * 0.01);
        this.cam.camera.farClip = (dist + radius) * 6;
    }

    reframe(opts) {
        if (this.asset) this._frameCamera(this.asset.resource.aabb, opts);
    }

    // --- LOD + streaming configuration (mirrors the UI) -----------------------

    // Per-splat LOD selection (component) + global streaming behaviour (scene).
    config = {
        lodBaseDistance: 5,
        lodMultiplier: 3,
        lodRangeMin: 0,
        lodRangeMax: 99,
        splatBudget: 0, // 0 = uncapped
        lodUpdateDistance: 1,
        lodUpdateAngle: 0,
        lodBehindPenalty: 1,
        debug: "none",
    };

    set(key, value) {
        this.config[key] = value;
        this._applyLodConfig();
    }

    _applyLodConfig() {
        const g = this.app.scene.gsplat;
        const c = this.config;
        g.splatBudget = c.splatBudget;
        g.lodUpdateDistance = c.lodUpdateDistance;
        g.lodUpdateAngle = c.lodUpdateAngle;
        g.lodBehindPenalty = c.lodBehindPenalty;
        g.debug = DEBUG_MODES[c.debug] ?? GSPLAT_DEBUG_NONE;
        g.dirty = true; // force LOD re-evaluation this frame, even without movement
        const comp = this.splat && this.splat.gsplat;
        if (comp) {
            comp.lodBaseDistance = c.lodBaseDistance;
            comp.lodMultiplier = c.lodMultiplier;
            comp.lodRangeMin = c.lodRangeMin;
            comp.lodRangeMax = c.lodRangeMax;
        }
    }

    setPixelRatio(r) {
        this.app.graphicsDevice.maxPixelRatio = r;
    }

    // Live numbers for the HUD.
    stats() {
        const camPos = this.cam.getPosition();
        const camDist = this.center ? camPos.distance(this.center) : 0;
        const filesLoaded = this.octree ? this.octree.fileResources.size : this.asset ? 1 : 0;
        const filesTotal = this.octree ? this.octree.files.length : this.asset ? 1 : 0;
        return {
            frameMs: this.frameMs,
            fps: this.frameMs > 0 ? 1000 / this.frameMs : 0,
            camDist,
            filesLoaded,
            filesTotal,
            bytes: this.bytes,
            pixelRatio: this.app.graphicsDevice.maxPixelRatio,
        };
    }

    _clearSplat() {
        if (this.splat) {
            this.splat.destroy();
            this.splat = null;
        }
        if (this.asset) {
            this.app.assets.remove(this.asset);
            this.asset.unload();
            this.asset = null;
        }
        this.octree = null;
        this.center = null;
    }

    dispose() {
        this._perf?.disconnect();
        this._resizeObs?.disconnect();
        this.controls.detach();
        this._clearSplat();
        this.app.destroy();
    }
}
