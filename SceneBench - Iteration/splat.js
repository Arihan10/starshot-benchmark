// Orbit-only gaussian-splat layer for the editorial arena.
// Adapted from "SceneBench spatial benchmark(1)/splat.js" in the repo (itself a port
// of prod_client/src/lib/orbit/splatLayer.ts). Panels sharing a stageHost share one
// PlayCanvas app (each view = camera + layer into a viewport rect of the shared
// canvas); distinct stageHosts get their own app/canvas, so a host inside a clipped
// panel renders a clippable view. Orbit + zoom only — no walkthrough logic.
const ENGINE = 'https://cdn.jsdelivr.net/npm/playcanvas@2.20.6/build/playcanvas.mjs';
const SPLATS = {
  A: 'client/public/assets/hotel-room-lod/1_0/meta.json',
  B: 'client/public/assets/hotel-room-lod/2_0/meta.json',
};

let pcPromise = null;
const getPc = () => (pcPromise = pcPromise || import(ENGINE));

const stages = new Map();

function makeStage(pc, host) {
  const canvas = document.createElement('canvas');
  Object.assign(canvas.style, { position: 'absolute', display: 'block', pointerEvents: 'none', zIndex: '0' });
  const parent = host.parentElement || host;
  if (getComputedStyle(parent).position === 'static') parent.style.position = 'relative';
  parent.appendChild(canvas);

  const s = { pc, app: null, canvas, host, parent, views: [] };

  const app = new pc.Application(canvas, {
    graphicsDeviceOptions: { antialias: false, alpha: true, preserveDrawingBuffer: true },
  });
  app.setCanvasFillMode(pc.FILLMODE_NONE);
  app.setCanvasResolution(pc.RESOLUTION_AUTO);
  app.graphicsDevice.maxPixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  app.start();
  s.app = app;

  const size = () => {
    const r = host.getBoundingClientRect();
    const p = parent.getBoundingClientRect();
    if (!r.width || !r.height) return;
    canvas.style.left = (r.left - p.left) + 'px';
    canvas.style.top = (r.top - p.top) + 'px';
    canvas.style.width = r.width + 'px';
    canvas.style.height = r.height + 'px';
    app.resizeCanvas(Math.round(r.width), Math.round(r.height));
  };
  size();
  const ro = new ResizeObserver(size);
  ro.observe(host);
  ro.observe(parent);
  s.ro = ro;
  s.reposition = size;

  app.on('update', (dt) => {
    const hr = canvas.getBoundingClientRect();
    if (!hr.width || !hr.height) return;
    for (const v of s.views) v.frame(dt, hr);
  });

  return s;
}

export function disposeStage() {
  for (const s of stages.values()) {
    s.ro.disconnect();
    try { s.app.destroy(); } catch { /* nothing left to save */ }
    s.canvas.remove();
  }
  stages.clear();
}

export function mount(container, opts = {}) {
  const HOME = { theta: 0.7, phi: 1.15, radius: 30 };
  const view = { ...HOME };
  const cam = { ...HOME };
  let spin = true, dragging = false, px = 0, py = 0, idleAt = 0;
  let variant = opts.variant === 'B' ? 'B' : 'A';
  let pc = null, app = null, camera = null, entity = null, layer = null;
  let target = null, disposed = false, self = null, stage = null;

  const loader = document.createElement('div');
  loader.textContent = 'LOADING SPLAT';
  Object.assign(loader.style, {
    position: 'absolute', inset: '0', display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontFamily: "'IBM Plex Mono', monospace", fontSize: '10px', letterSpacing: '0.24em',
    color: 'rgba(236,233,226,0.34)', pointerEvents: 'none', transition: 'opacity 420ms ease',
  });
  container.appendChild(loader);

  container.style.cursor = 'grab';
  container.style.touchAction = 'none';

  const down = (e) => {
    if (e.target.closest('button, a')) return;
    e.stopPropagation(); e.preventDefault();
    dragging = true; px = e.clientX; py = e.clientY;
    container.setPointerCapture(e.pointerId); container.style.cursor = 'grabbing';
  };
  const move = (e) => {
    if (!dragging) return;
    e.stopPropagation();
    view.theta -= (e.clientX - px) * 0.007;
    view.phi = Math.max(0.2, Math.min(2.4, view.phi - (e.clientY - py) * 0.006));
    px = e.clientX; py = e.clientY;
  };
  const end = () => { if (!dragging) return; dragging = false; idleAt = performance.now(); container.style.cursor = 'grab'; };
  const wheel = (e) => {
    e.stopPropagation(); e.preventDefault();
    view.radius = Math.max(HOME.radius * 0.25, Math.min(HOME.radius * 2.2, view.radius * (1 + e.deltaY * 0.0013)));
    idleAt = performance.now();
  };
  container.addEventListener('pointerdown', down);
  container.addEventListener('pointermove', move);
  container.addEventListener('pointerup', end);
  container.addEventListener('pointercancel', end);
  container.addEventListener('wheel', wheel, { passive: false });

  // Scan AABBs include stray outliers — frame on the 3rd-97th percentile of centres.
  function frameTo(asset) {
    const res = asset && asset.resource;
    if (!res || !entity) return;
    const centers = (res.hasCenters && res.centers) || res._centers || res.centers;
    let c, ext;
    if (centers && centers.length > 8) {
      const n = centers.length / 3;
      const span = (axis) => {
        const len = Math.min(n, 50000);
        const step = Math.max(1, Math.floor(n / len));
        const v = [];
        for (let i = 0; i < n && v.length < len; i += step) v.push(centers[i * 3 + axis]);
        v.sort((a, b) => a - b);
        return [v[Math.floor(v.length * 0.03)], v[Math.floor(v.length * 0.97)]];
      };
      const X = span(0), Y = span(1), Z = span(2);
      c = new pc.Vec3((X[0] + X[1]) / 2, (Y[0] + Y[1]) / 2, (Z[0] + Z[1]) / 2);
      ext = Math.hypot(X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]) / 2;
    } else {
      c = res.aabb.center.clone();
      ext = res.aabb.halfExtents.length();
    }
    target = c;
    HOME.radius = Math.max(1.2, ext * (opts.frameZoom ?? 1.6));
    view.radius = HOME.radius;
    cam.radius = HOME.radius * 1.35;
  }

  async function load() {
    const url = SPLATS[variant];
    const asset = new pc.Asset('splat-' + variant + '-' + Math.random().toString(36).slice(2, 7), 'gsplat', { url });
    app.assets.add(asset);
    app.assets.load(asset);
    await new Promise((res) => (asset.resource ? res() : asset.ready(res)));
    if (disposed) return;
    if (entity) entity.destroy();
    entity = new pc.Entity('splat');
    entity.addComponent('gsplat', { asset, layers: [layer.id] });
    app.root.addChild(entity);
    frameTo(asset);
    loader.style.opacity = '0';
  }

  getPc().then(async (mod) => {
    if (disposed) return;
    pc = mod;
    const host = opts.stageHost || container.parentElement;
    stage = stages.get(host);
    if (!stage) { stage = makeStage(pc, host); stages.set(host, stage); }
    app = stage.app;

    layer = new pc.Layer({ name: 'splat-' + stage.views.length + '-' + Math.random().toString(36).slice(2, 5) });
    app.scene.layers.push(layer);

    camera = new pc.Entity('camera');
    camera.addComponent('camera', {
      clearColor: new pc.Color(0, 0, 0, 0),
      fov: 42,
      farClip: 5000,
      layers: [layer.id],
      priority: stage.views.length,
    });
    app.root.addChild(camera);
    target = new pc.Vec3();

    self = {
      frame(dt, hr) {
        const r = container.getBoundingClientRect();
        const visible = r.width > 2 && r.height > 2 && getComputedStyle(container).visibility !== 'hidden';
        camera.enabled = visible;
        if (!visible) return;
        camera.camera.rect = new pc.Vec4(
          (r.left - hr.left) / hr.width,
          1 - (r.bottom - hr.top) / hr.height,
          r.width / hr.width,
          r.height / hr.height
        );
        if (spin && !dragging && performance.now() - idleAt > 1400) view.theta += dt * 0.16;
        cam.theta += (view.theta - cam.theta) * 0.12;
        cam.phi += (view.phi - cam.phi) * 0.12;
        cam.radius += (view.radius - cam.radius) * 0.08;
        const t = target;
        camera.setPosition(
          t.x + cam.radius * Math.sin(cam.phi) * Math.sin(cam.theta),
          t.y + cam.radius * Math.cos(cam.phi),
          t.z + cam.radius * Math.sin(cam.phi) * Math.cos(cam.theta)
        );
        camera.lookAt(t);
      },
    };
    stage.views.push(self);

    await load();
  }).catch((e) => {
    loader.textContent = 'SPLAT UNAVAILABLE';
    console.warn('[splat] engine failed', e);
  });

  return {
    setSpin(v) { spin = !!v; },
    setVariant(v) {
      const next = v === 'B' ? 'B' : 'A';
      if (next === variant || !app) return;
      variant = next;
      loader.style.opacity = '1';
      load();
    },
    reset() {
      view.theta = HOME.theta; view.phi = HOME.phi; view.radius = HOME.radius;
      idleAt = performance.now();
    },
    // Pixel copy of just this panel's region, for the shatter effect.
    snapshot() {
      if (!stage) return null;
      const c = stage.canvas, hr = c.getBoundingClientRect(), r = container.getBoundingClientRect();
      if (!hr.width || !r.width) return null;
      const sx = c.width / hr.width, sy = c.height / hr.height;
      const out = document.createElement('canvas');
      out.width = Math.max(1, Math.round(r.width * sx));
      out.height = Math.max(1, Math.round(r.height * sy));
      try {
        out.getContext('2d').drawImage(c,
          Math.round((r.left - hr.left) * sx), Math.round((r.top - hr.top) * sy),
          out.width, out.height, 0, 0, out.width, out.height);
        return out;
      } catch (err) { return null; }
    },
    dispose() {
      disposed = true;
      loader.remove();
      container.removeEventListener('pointerdown', down);
      container.removeEventListener('pointermove', move);
      container.removeEventListener('pointerup', end);
      container.removeEventListener('pointercancel', end);
      container.removeEventListener('wheel', wheel);
      if (entity) { entity.destroy(); entity = null; }
      if (camera) { camera.destroy(); camera = null; }
      if (stage && self) {
        const i = stage.views.indexOf(self);
        if (i > -1) stage.views.splice(i, 1);
      }
    },
  };
}
