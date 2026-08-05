// SOG-LOD playground viewer — a PlayCanvas app that renders a Gaussian splat and,
// when handed a streamed-SOG bundle (lod-meta.json), lets the engine's native LOD
// system STREAM per-octree-node level-of-detail over the network: coarse chunks
// load first, then refine by camera distance under a splat budget, with unused
// chunks unloaded after a cooldown. A single .sog still loads (no LOD).
//
// The engine does the streaming; this module owns the app, the birdseye camera,
// and the plumbing that makes every lever in levers.js reach the thing it
// controls — the scene's gsplat params, the splat entity's LOD selection, the
// octree's asset loader (fetch concurrency), or the device itself. It also feeds
// the HUD, including the two numbers the engine alone can answer: whether the
// current viewpoint is fully resolved, and how many fetches are still out.

import {
    Application,
    Color,
    Entity,
    FILLMODE_NONE,
    RESOLUTION_AUTO,
    createGraphicsDevice,
    DEVICETYPE_WEBGPU,
    DEVICETYPE_WEBGL2,
    GSPLAT_DEBUG_NONE,
    GSPLAT_DEBUG_LOD,
    GSPLAT_DEBUG_AABBS,
    GSPLAT_DEBUG_NODE_AABBS,
    GSPLAT_DEBUG_SH_UPDATE,
    GSPLAT_RENDERER_AUTO,
    GSPLAT_RENDERER_RASTER_CPU_SORT,
    GSPLAT_RENDERER_RASTER_GPU_SORT,
    GSPLATDATA_COMPACT,
    GSPLATDATA_LARGE,
} from "playcanvas";

import { OrbitFlyControls } from "./controls.js";
import { LEVERS, LEVERS_BY_KEY, defaults } from "./levers.js";

// Levers whose UI value is a token; the engine wants its constant.
const ENUMS = {
    debug: {
        none: GSPLAT_DEBUG_NONE,
        lod: GSPLAT_DEBUG_LOD,
        nodes: GSPLAT_DEBUG_NODE_AABBS,
        aabbs: GSPLAT_DEBUG_AABBS,
        shUpdate: GSPLAT_DEBUG_SH_UPDATE,
    },
    renderer: {
        auto: GSPLAT_RENDERER_AUTO,
        cpu: GSPLAT_RENDERER_RASTER_CPU_SORT,
        gpu: GSPLAT_RENDERER_RASTER_GPU_SORT,
    },
    dataFormat: { compact: GSPLATDATA_COMPACT, large: GSPLATDATA_LARGE },
};

// Scene-level LOD/streaming knobs the engine stores as plain fields, so nothing
// re-evaluates until the camera happens to move. Flagging the scene dirty makes a
// slider drag visible on a parked camera, which is the whole point of a panel.
const FORCE_REEVALUATE = new Set(["stream"]);

const MOVE_EPSILON = 1e-4;
const DRAWN_POLL_MS = 400;

export class SplatViewer {
    // Frustum culling only exists on the GPU-sort renderer, and that renderer only
    // exists on WebGPU: the cull is a compute pass (`GSplatIntervalCull`) reading
    // storage buffers, so on WebGL the engine silently falls back to CPU sorting
    // with no culling at all. That makes the DEVICE the real switch, and a device
    // has to be built before the Application rather than by it — hence this
    // factory. `deviceType` forces one ("webgpu" / "webgl2"); by default WebGPU is
    // preferred and `createGraphicsDevice` falls back to WebGL2 on its own.
    //
    // glslangUrl/twgslUrl are deliberately omitted: those transpile GLSL for
    // WebGPU, and the engine's gsplat path ships native WGSL chunks, so the wasm
    // payload buys nothing here.
    static async create(canvas, { deviceType } = {}) {
        const deviceTypes =
            deviceType === "webgl2" ? [DEVICETYPE_WEBGL2]
            : deviceType === "webgpu" ? [DEVICETYPE_WEBGPU]
            : [DEVICETYPE_WEBGPU, DEVICETYPE_WEBGL2];
        const device = await createGraphicsDevice(canvas, { deviceTypes, antialias: false });
        return new SplatViewer(canvas, device);
    }

    constructor(canvas, device) {
        this.canvas = canvas;
        this.splat = null; // the gsplat Entity
        this.asset = null;
        this.octree = null; // GSplatOctree when streaming LOD, else null
        this.center = null; // scene AABB centre (for the HUD's camera-distance)
        this.frameMs = 16; // smoothed
        this.config = defaults();

        // Streaming progress, straight from the engine's own frame:ready signal.
        this.resolved = false;
        this.pendingLoads = 0;

        // Network accounting for the ACTIVE asset (see _observeNetwork).
        this.bytes = 0;
        this.requests = 0;
        this.cacheHits = 0;
        this.latencyMs = 0;
        this._assetBase = null;

        this._needsRender = false;
        this._lastCamPos = null;
        this._lastCamFwd = null;

        // Post-cull drawn count, sampled off the GPU (see _pollDrawnSplats).
        this.drawnSplats = null;
        this._drawnAt = 0;
        this._drawnPending = false;

        // Application only builds its own WebGL device when none is handed in, so
        // passing one keeps every default component system and resource handler.
        const app = new Application(canvas, {
            graphicsDevice: device,
            graphicsDeviceOptions: { antialias: false },
        });
        this.device = app.graphicsDevice;
        app.setCanvasFillMode(FILLMODE_NONE);
        app.setCanvasResolution(RESOLUTION_AUTO);
        app.graphicsDevice.maxPixelRatio = this.config.pixelRatio;
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

        // Both handlers run before the engine decides whether to draw this frame
        // (update → framerender → render), so an on-demand request is honoured
        // without a frame of latency.
        app.systems.gsplat.on("frame:request", () => {
            if (this.config.onDemandRender) app.renderNextFrame = true;
        });
        app.systems.gsplat.on("frame:ready", (_camera, _layer, ready, loading) => {
            this.resolved = ready;
            this.pendingLoads = loading;
        });

        app.on("update", (dt) => {
            this.controls.update(dt);
            this.frameMs += ((dt * 1000) - this.frameMs) * 0.1; // ~EMA smoothing
            if (this.config.onDemandRender && this._cameraMoved()) app.renderNextFrame = true;
            this._pollDrawnSplats(performance.now());
        });

        const fit = () => {
            const r = canvas.parentElement.getBoundingClientRect();
            if (r.width > 0 && r.height > 0) app.resizeCanvas(r.width, r.height);
        };
        fit();
        this._resizeObs = new ResizeObserver(fit);
        this._resizeObs.observe(canvas.parentElement);

        this._observeNetwork();
        this.applyAll();
    }

    // Per-resource transfer stats for the active asset: what streaming actually
    // costs on the wire, and how much of it the HTTP cache absorbed. Requires the
    // server to send `Timing-Allow-Origin` — the splats come from the API origin
    // and the browser zeroes cross-origin sizes without it (see server/app/api/sog.py).
    _observeNetwork() {
        this._perf = new PerformanceObserver((list) => {
            for (const e of list.getEntries()) {
                if (!this._assetBase || !e.name.includes(this._assetBase)) continue;
                const transferred = e.transferSize || 0;
                this.requests += 1;
                this.bytes += transferred;
                // A served-from-cache entry reports a body but no transfer.
                if (transferred === 0 && (e.decodedBodySize || 0) > 0) this.cacheHits += 1;
                this.latencyMs += (e.duration - this.latencyMs) / this.requests;
            }
        });
        try {
            this._perf.observe({ type: "resource", buffered: true });
        } catch {
            this._perf = null; // Safari <16 etc. — the HUD just omits the counters
        }
    }

    _cameraMoved() {
        const p = this.cam.getPosition();
        const f = this.cam.forward;
        const moved =
            !this._lastCamPos ||
            this._lastCamPos.distance(p) > MOVE_EPSILON ||
            this._lastCamFwd.distance(f) > MOVE_EPSILON;
        this._lastCamPos = p.clone();
        this._lastCamFwd = f.clone();
        return moved;
    }

    // Load a splat by URL. `lod-meta.json` → streamed LOD; anything else → single
    // splat. Resolves with a summary once the resource is ready + framed.
    async load(url) {
        this._clearSplat();
        this.bytes = 0;
        this.requests = 0;
        this.cacheHits = 0;
        this.latencyMs = 0;
        this.resolved = false;
        this.pendingLoads = 0;
        // Directory of the manifest/sog — every chunk the engine derives from it
        // shares this prefix, which is what attributes their fetches to this asset.
        this._assetBase = url.replace(/[^/]*$/, "");

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
        // The entity and the octree's loader only exist now, so re-push every
        // lever that targets them.
        this.applyAll();
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

    // --- levers ---------------------------------------------------------------

    set(key, value) {
        this.config[key] = value;
        this._apply(LEVERS_BY_KEY.get(key));
    }

    applyAll() {
        for (const lever of LEVERS) this._apply(lever);
    }

    _apply(lever) {
        if (!lever) return;
        const { key, target } = lever;
        const value = this.config[key];
        switch (target) {
            case "scene": {
                const gsplat = this.app.scene.gsplat;
                gsplat[key] = ENUMS[key] ? ENUMS[key][value] : value;
                if (FORCE_REEVALUATE.has(lever.group)) gsplat.dirty = true;
                break;
            }
            case "entity":
                // The component forwards to its placement, which flags itself
                // lodDirty — no scene-wide invalidation needed.
                if (this.splat?.gsplat) this.splat.gsplat[key] = value;
                break;
            case "net":
                if (this.octree?.assetLoader) this.octree.assetLoader[key] = value;
                break;
            case "app":
                if (key === "pixelRatio") this.app.graphicsDevice.maxPixelRatio = value;
                break;
            case "viewer":
                if (key === "onDemandRender") {
                    this.app.autoRender = !value;
                    this.app.renderNextFrame = true;
                }
                break;
        }
    }

    // --- HUD ------------------------------------------------------------------

    // The manager owning this camera's splats. Reached through the director because
    // the engine exposes no public handle, and defensively so an engine change
    // degrades the readout instead of throwing. The map is keyed on the scene
    // Camera, which is one level below the component: `entity.camera` is the
    // CameraComponent and `entity.camera.camera` is the Camera the director sees.
    _gsplatManager() {
        const cameras = this.app.renderer?.gsplatDirector?.camerasMap;
        if (!cameras) return null;
        const cameraData =
            cameras.get(this.cam.camera?.camera) ??
            (cameras.size === 1 ? cameras.values().next().value : null);
        for (const layerData of cameraData?.layersMap?.values() ?? []) {
            if (layerData.gsplatManager) return layerData.gsplatManager;
        }
        return null;
    }

    // The surviving splat count after the cull, read back from the buffer that
    // drives the indirect draw. Async and rate-limited: it is a GPU-to-CPU copy, so
    // it is sampled a few times a second and `stats()` reports the last answer.
    // Without this, culling has no observable effect on this machine at all — the
    // reduction happens entirely on the GPU after the CPU's last count.
    _pollDrawnSplats(now) {
        if (this._drawnPending || now - this._drawnAt < DRAWN_POLL_MS) return;
        const buffer = this._gsplatManager()?.renderer?.intervalCompaction?.numSplatsBuffer;
        if (!buffer) {
            this.drawnSplats = null;
            return;
        }
        this._drawnAt = now;
        this._drawnPending = true;
        Promise.resolve(buffer.read(0, 4))
            .then((bytes) => {
                if (bytes?.buffer) {
                    this.drawnSplats = new Uint32Array(bytes.buffer, bytes.byteOffset, 1)[0];
                }
            })
            .catch(() => {
                this.drawnSplats = null; // readback unsupported — HUD just omits it
            })
            .finally(() => {
                this._drawnPending = false;
            });
    }

    // Whether per-node frustum culling is running, and if not, what is missing.
    // There is no engine flag to report: culling is a consequence of the GPU-sort
    // renderer, which needs WebGPU. `boundsEntries` is how many bounding spheres
    // the compute pass tests each frame — one per resident octree node for a
    // streamed bundle, or 1 for a single .sog (whose only cull is all-or-nothing).
    culling() {
        const api = this.device.isWebGPU ? "WebGPU" : "WebGL 2";
        if (!this.device.isWebGPU) {
            return { active: false, api, reason: "needs WebGPU", boundsEntries: 0 };
        }
        if (this.app.scene.gsplat.currentRenderer !== GSPLAT_RENDERER_RASTER_GPU_SORT) {
            return { active: false, api, reason: "needs the GPU-sort renderer", boundsEntries: 0 };
        }
        const world = this._gsplatManager()?.world;
        if (!world) {
            return { active: false, api, reason: "no splat resident", boundsEntries: 0 };
        }
        return {
            active: world.hasBounds,
            api,
            reason: world.hasBounds ? "per-node compute cull" : "waiting for bounds",
            boundsEntries: world.workBuffer?.frustumCuller?.totalBoundsEntries ?? 0,
        };
    }

    stats() {
        const camDist = this.center ? this.cam.getPosition().distance(this.center) : 0;
        const filesLoaded = this.octree
            ? this.octree.fileResources.size
            : this.asset ? 1 : 0;
        const filesTotal = this.octree ? this.octree.files.length : this.asset ? 1 : 0;
        return {
            frameMs: this.frameMs,
            fps: this.frameMs > 0 ? 1000 / this.frameMs : 0,
            camDist,
            filesLoaded,
            filesTotal,
            // Chunks past their last reference, still resident on the cooldown —
            // what a turn-around gets back for free.
            filesCooling: this.octree ? this.octree.cooldowns.size : 0,
            bytes: this.bytes,
            requests: this.requests,
            cacheHits: this.cacheHits,
            latencyMs: this.latencyMs,
            resolved: this.resolved,
            pendingLoads: this.pendingLoads,
            culling: this.culling(),
            // Gaussians baked into the work buffer this frame — the number the splat
            // budget is enforced against, and the population the cull then filters.
            splatsRendered: this.app.renderer._gsplatCount ?? 0,
            // What actually reached the draw after culling, or null when the readback
            // isn't available (no culling, or an unsupported device).
            splatsDrawn: this.drawnSplats,
            // Share of the work buffer re-uploaded, i.e. how much churn the current
            // LOD settings are causing.
            bufferChurn: this.app.renderer._gsplatBufferCopy ?? 0,
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
