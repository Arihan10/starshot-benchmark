// The arena's scene layer: each panel renders one cell's TRAINED GAUSSIAN SPLAT —
// the same `trained.web.sog` files the production client ships under
// prod_client/public/scenes — rather than a procedural stand-in built in three.js.
//
// PORTED FROM prod_client/src/lib/orbit/splatLayer.ts, minus the parts that only make
// sense there. Prod stacks this canvas UNDER a three.js one that draws the markers,
// cursor and walkthrough over the splat, so it runs PlayCanvas with `autoRender` off
// and renders from three's own tick to keep both layers on the same frame. Here the
// splat is the whole picture — but the render stays externally driven for the same
// reason one remove out: both panels are stepped by the single loop in
// SceneBench.dc.html, so a side-by-side comparison never shows A a frame ahead of B.
//
// ONE APPLICATION PER PANEL, also as prod does. PlayCanvas' gsplat renderer needs its
// own GraphicsDevice and a WebGL context cannot be shared, so two panels means two
// apps — which is fine, and the alternative (one app, two cameras with viewport rects
// carved out of a canvas spanning both panels) has to track the panels' boxes by hand.

// Pinned to the version prod pins (prod_client/package.json), because `app.render()`
// is marked @ignore in PlayCanvas' typings — internal, not part of the supported
// surface. See the call site at the bottom of `frame`.
const ENGINE = 'https://cdn.jsdelivr.net/npm/playcanvas@2.20.6/build/playcanvas.mjs';

let pcPromise = null;
const getPc = () => (pcPromise = pcPromise || import(ENGINE));

/* ---------- where the .sog files are ---------- */

// The splats are NOT copied into this folder. They live in the production client's
// public/ tree, and duplicating 12 MB per cell to preview a design component is how
// the two silently drift apart. What that tree looks like from here depends entirely
// on where the previewing server is rooted, which this file cannot know — so it asks.
// The first candidate that answers for a real splat wins and every later URL resolves
// against it.
const BASES = [
  './scenes/',                     // this folder (the `scenes` symlink beside this file)
  '../prod_client/public/scenes/', // a server rooted at the repo
  '/scenes/',                      // a server rooted at prod_client/public (next dev)
  '/prod_client/public/scenes/',
];

const splatFile = (dir) => `${dir}/trained.web.sog`;

let basePromise = null;
function resolveBase(probe) {
  if (basePromise) return basePromise;
  const override = typeof window !== 'undefined' && window.SCENEBENCH_SPLAT_BASE;
  if (override) return (basePromise = Promise.resolve(String(override).replace(/\/*$/, '/')));
  basePromise = (async () => {
    for (const b of BASES) {
      try {
        // A one-byte range, not HEAD: HEAD is optional in static servers, and a plain
        // GET would download 12 MB from every candidate that answers. `cancel` covers
        // the servers that ignore Range and start sending the whole file anyway.
        const res = await fetch(b + probe, { headers: { Range: 'bytes=0-0' } });
        res.body?.cancel?.();
        if (res.ok) return b;
      } catch {
        /* wrong root, or blocked — try the next */
      }
    }
    const err = new Error('no reachable splat base — set window.SCENEBENCH_SPLAT_BASE');
    err.code = 'no-base';
    throw err;
  })();
  return basePromise;
}

/** Absolute URL of a cell's trained splat, e.g. splatUrl('modern-house-gemini-flash'). */
export async function splatUrl(dir) {
  const file = splatFile(dir);
  return (await resolveBase(file)) + file;
}

/* ---------- viewport ---------- */

// How long after a drag/zoom before the panel resumes orbiting on its own.
const IDLE_MS = 1400;

export class Viewport {
  constructor(el, opts = {}) {
    this.el = el;
    this.autoSpeed = opts.orbitSpeed ?? 0.14;
    this.frameZoom = opts.frameZoom ?? 1.9;

    // `view` is where the input says the camera should be, `cam` is where it has got
    // to. Everything is chased rather than assigned, so a drag, the reveal swing and
    // the auto-orbit can all be in flight at once without fighting over one number.
    this.home = { az: opts.az0 ?? -0.7, pol: 1.15, rad: 8 };
    this.view = { ...this.home };
    this.cam = { ...this.home, rad: this.home.rad * 1.35 };
    this.target = { x: 0, y: 0, z: 0 };
    this.ext = this.home.rad / this.frameZoom;

    this.dragging = false;
    this.vel = 0;
    this.swingV = 0;
    this.idleAt = 0;
    this._pulse = 1;
    // Bumped on every load and on dispose, so an in-flight import or asset fetch that
    // lands after the round changed is dropped instead of attaching to it.
    this.token = 0;
    // Kept, not unloaded, unlike prod's `clear()`. The arena cycles through a fixed
    // handful of cells and shows each one repeatedly, so holding their resources turns
    // every later round into an instant swap; prod's catalog is unbounded, so it can't.
    this.assets = new Map();

    el.style.touchAction = 'none';
    el.style.cursor = 'grab';
    this._down = (e) => {
      if (e.target.closest && e.target.closest('button, a')) return;
      this.dragging = true;
      this._px = e.clientX;
      this._py = e.clientY;
      el.setPointerCapture?.(e.pointerId);
      el.style.cursor = 'grabbing';
    };
    this._move = (e) => {
      if (!this.dragging) return;
      const dx = e.clientX - this._px;
      const dy = e.clientY - this._py;
      this._px = e.clientX;
      this._py = e.clientY;
      this.view.az -= dx * 0.006;
      this.view.pol = Math.max(0.25, Math.min(1.5, this.view.pol - dy * 0.005));
      this.vel = -dx * 0.16;
    };
    this._up = () => {
      if (!this.dragging) return;
      this.dragging = false;
      this.idleAt = performance.now();
      el.style.cursor = 'grab';
    };
    this._wheel = (e) => {
      e.preventDefault();
      const r = this.home.rad;
      this.view.rad = Math.max(r * 0.25, Math.min(r * 2.2, this.view.rad * (1 + e.deltaY * 0.0013)));
      this.idleAt = performance.now();
    };
    el.addEventListener('pointerdown', this._down);
    el.addEventListener('pointermove', this._move);
    el.addEventListener('pointerup', this._up);
    el.addEventListener('pointercancel', this._up);
    el.addEventListener('wheel', this._wheel, { passive: false });

    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(el);

    // A panel that cannot get a WebGL context has not failed to load a scene — the
    // arena cannot render at all — so the failure is remembered and re-thrown out of
    // `load` for the host to put on screen, rather than swallowed into a black panel.
    this.ready = this._init().catch((e) => {
      this.dead = e;
      throw e;
    });
    // `load` is what reports it; this only keeps it off the unhandled-rejection channel
    // in the window before anything asks to load.
    this.ready.catch(() => {});
  }

  async _init() {
    const pc = (this.pc = await getPc());
    if (this.disposed) return;

    const canvas = document.createElement('canvas');
    canvas.style.cssText =
      'position:absolute;inset:0;width:100%;height:100%;display:block;pointer-events:none';
    // Prepended so it can never sit above the panel's own overlays (the A/B letter,
    // the reveal card) if the host ever appends any.
    this.el.insertBefore(canvas, this.el.firstChild);
    this.canvas = canvas;

    const app = new pc.Application(canvas, { graphicsDeviceOptions: { antialias: false } });
    app.setCanvasFillMode(pc.FILLMODE_NONE);
    app.setCanvasResolution(pc.RESOLUTION_AUTO);
    // Two 690k-splat panels side by side: full retina on both costs far more than it
    // shows at this size.
    app.graphicsDevice.maxPixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    app.start();
    app.autoRender = false; // see the lockstep note at the top — `frame` renders

    // `app` EXPLICITLY, here and on every entity below. `new pc.Entity(name)` defaults
    // its owning app to PlayCanvas' module-global "current application" — the last one
    // constructed anywhere on the page — and `addComponent` resolves the component
    // system off THAT app. With two panels the awaits interleave: both apps exist by
    // the time either asset lands, so one splat ends up with a gsplat component
    // belonging to the other panel's app while sitting in this one's scene graph, and
    // silently never draws. (Straight from prod's splatLayer.ts, which learned it the
    // hard way.)
    const camera = new pc.Entity('splat-camera', app);
    camera.addComponent('camera', {
      // Pure black, matching the page behind it, so the splat's empty space and the
      // arena's backdrop agree.
      clearColor: new pc.Color(0, 0, 0, 1),
      fov: 42,
      nearClip: 0.05,
      farClip: 2000,
    });
    app.root.addChild(camera);

    this.app = app;
    this.camera = camera;
    this.resize();
  }

  /**
   * Show the cell whose splat lives in `dir` under the scenes root. Resolves false if
   * the load was superseded by a later one or the asset could not be fetched — the
   * panel then simply keeps showing black rather than taking the arena down with it.
   */
  async load(dir) {
    const token = ++this.token;
    if (this.entity) {
      this.entity.destroy();
      this.entity = null;
    }
    try {
      const url = await splatUrl(dir);
      await this.ready;
      if (token !== this.token || !this.app) return false;
      const pc = this.pc;
      const app = this.app;

      let asset = this.assets.get(url);
      if (!asset) {
        asset = await new Promise((resolve, reject) => {
          app.assets.loadFromUrl(url, 'gsplat', (err, a) =>
            err || !a ? reject(new Error(String(err))) : resolve(a));
        });
        if (token !== this.token) {
          asset.unload();
          return false;
        }
        this.assets.set(url, asset);
      }

      const entity = new pc.Entity('splat', app);
      entity.addComponent('gsplat', { asset });
      app.root.addChild(entity);
      this.entity = entity;
      this.frameTo(asset);
      return true;
    } catch (e) {
      if (e && e.code === 'no-base') throw e; // the host reports this one to the user
      console.warn('[splat] load failed', dir, e);
      return false;
    }
  }

  /**
   * Park the camera around the splat that just arrived.
   *
   * Scan AABBs are dragged wide open by stray floaters — a handful of splats behind
   * the rig, or out past a wall — so framing on the raw box parks the camera far too
   * far back and the scene reads as a speck. Frame on the 3rd–97th percentile of splat
   * centres instead: that is the scene a person would say they are looking at.
   */
  frameTo(asset) {
    const res = asset && asset.resource;
    if (!res) return;
    const centers = res.hasCenters ? res.centers : null;
    let cx, cy, cz, ext;
    if (centers && centers.length > 8) {
      const n = centers.length / 3;
      const span = (axis) => {
        const want = Math.min(n, 50000);
        const step = Math.max(1, Math.floor(n / want));
        const v = [];
        for (let i = 0; i < n && v.length < want; i += step) v.push(centers[i * 3 + axis]);
        v.sort((a, b) => a - b);
        return [v[Math.floor(v.length * 0.03)], v[Math.floor(v.length * 0.97)]];
      };
      const X = span(0), Y = span(1), Z = span(2);
      cx = (X[0] + X[1]) / 2;
      cy = (Y[0] + Y[1]) / 2;
      cz = (Z[0] + Z[1]) / 2;
      ext = Math.hypot(X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]) / 2;
    } else {
      const box = res.aabb;
      cx = box.center.x; cy = box.center.y; cz = box.center.z;
      ext = box.halfExtents.length();
    }
    this.target = { x: cx, y: cy, z: cz };
    this.ext = ext;
    this.home.rad = Math.max(1.2, ext * this.frameZoom);
    // Back to the opening pose for the new scene, and far enough out that the intro
    // below is a push-in rather than a cut.
    this.view = { ...this.home };
    this.cam = { ...this.home, rad: this.home.rad * 1.35 };
  }

  /** Framing tightness, from the host's `frameZoom` prop. */
  setFrameZoom(z) {
    if (!(z > 0) || z === this.frameZoom) return;
    const k = z / this.frameZoom;
    this.frameZoom = z;
    this.home.rad *= k;
    this.view.rad *= k;
  }

  /** A shove of angular momentum, on both panels, the moment a vote lands. */
  swing() {
    this.swingV += 1.2;
  }

  /** A short push-in, on the winner only. */
  pulse() {
    this._pulse = 0;
  }

  frame(dt, t) {
    const app = this.app;
    if (!app || !this.camera) return;
    // A render with no layout size resizes the backbuffer to 0×0 and throws the frame
    // away — and these panels genuinely reach zero (the arena collapses one side
    // outright at narrow widths). Skipping is free; the next real frame picks up.
    if (!this.el.clientWidth || !this.el.clientHeight) return;

    if (!this.dragging && performance.now() - this.idleAt > IDLE_MS) {
      this.view.az += (this.autoSpeed + this.vel) * dt;
    }
    this.vel *= Math.pow(0.03, dt);
    this.view.az += this.swingV * dt;
    this.swingV *= Math.pow(0.008, dt);

    const c = this.cam, v = this.view;
    c.az += (v.az - c.az) * Math.min(1, dt * 7);
    c.pol += (v.pol - c.pol) * Math.min(1, dt * 7);
    c.rad += (v.rad - c.rad) * Math.min(1, dt * 4.5);

    this._pulse = Math.min(1, this._pulse + dt / 0.55);
    const push = 1 - Math.sin(this._pulse * Math.PI) * 0.06;

    const r = c.rad * push, tg = this.target;
    this.camera.setPosition(
      tg.x + r * Math.sin(c.pol) * Math.sin(c.az),
      tg.y + r * Math.cos(c.pol),
      tg.z + r * Math.sin(c.pol) * Math.cos(c.az));
    this.camera.lookAt(tg.x, tg.y, tg.z);

    // `render()` is @ignore in PlayCanvas' typings, so the version is pinned exactly
    // and this degrades rather than throws if a future one drops it. It is used anyway
    // because it is the only way to draw SYNCHRONOUSLY, here, with the camera just set
    // — the public alternative (`renderNextFrame`) defers to PlayCanvas' own rAF,
    // which may run either side of the host's, letting the two panels present poses
    // from different frames.
    if (typeof app.render === 'function') app.render();
    else app.renderNextFrame = true;
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (this.app && w > 0 && h > 0) this.app.resizeCanvas(w, h);
  }

  dispose() {
    this.disposed = true;
    this.token++;
    this._ro.disconnect();
    const el = this.el;
    el.removeEventListener('pointerdown', this._down);
    el.removeEventListener('pointermove', this._move);
    el.removeEventListener('pointerup', this._up);
    el.removeEventListener('pointercancel', this._up);
    el.removeEventListener('wheel', this._wheel);
    this.entity = null;
    this.assets.clear();
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
  }
}

export function startLoop(vps) {
  let last = performance.now(), stop = false;
  function tick(now) {
    if (stop) return;
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    for (const v of vps) v.frame(dt, now / 1000);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
  return () => { stop = true; };
}
