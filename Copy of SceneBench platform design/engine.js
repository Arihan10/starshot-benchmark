// SceneBench — dual-viewport render engine
import * as THREE from 'https://unpkg.com/three@0.184.0/build/three.module.js';
import { buildScene } from './scenes.js';

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const outCubic = t => 1 - Math.pow(1 - t, 3);
const outBack = t => { const c = 2.0, u = t - 1; return 1 + (c + 1) * u * u * u + c * u * u; };
const inBack = t => { const c = 1.9; return (c + 1) * t * t * t - c * t * t; };

const ENVS = {
  day:   { sky: 0x94a9d1, gr: 0x1a1613, hemi: .75, key: [0xffe9c4, 3.1, [6, 10, 4]],  fill: [0x7f92c9, .6, [-7, 4, -2]],  rim: [0xaebfff, 1, [-2, 6, -9]],   exp: 1.12 },
  dusk:  { sky: 0x5e6ea8, gr: 0x181310, hemi: .4,  key: [0xffb583, 1.7, [7, 5.5, 5]], fill: [0x4b5b96, .45, [-7, 4, -2]], rim: [0x8fa8ff, 1.1, [-3, 6, -9]], exp: 1.12 },
  night: { sky: 0x39435f, gr: 0x0c0c10, hemi: .4,  key: [0xa9bef2, 1.35, [-5, 9, -3]], fill: [0x2c3554, .45, [6, 3, 2]], rim: [0x7d8fc4, 1.15, [3, 5, -8]],   exp: 1.15 },
};

class Viewport {
  constructor(wrap, canvas) {
    this.wrap = wrap; this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: true });
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x000000, 11, 26);
    this.camera = new THREE.PerspectiveCamera(38, 1, .1, 120);
    this.hemi = new THREE.HemisphereLight(0x94a9d1, 0x141210, .6);
    this.key = new THREE.DirectionalLight(0xffe9c4, 2.6);
    this.key.castShadow = true;
    this.key.shadow.mapSize.set(2048, 2048);
    const sc = this.key.shadow.camera;
    sc.left = -9.5; sc.right = 9.5; sc.top = 9.5; sc.bottom = -9.5; sc.far = 45;
    this.key.shadow.bias = -.0004; this.key.shadow.normalBias = .035;
    this.fill = new THREE.DirectionalLight(0x7f92c9, .55);
    this.rim = new THREE.DirectionalLight(0xaebfff, .9);
    this.scene.add(this.hemi, this.key, this.key.target, this.fill, this.rim);
    this.az = -.62; this.pol = 1.12; this.radius = 8.6; this.targetY = 1.2;
    this.pulse = 0; this.exp = 1.1; this.expT = 1.1; this.radScale = 1; this.radScaleT = 1;
    this.objects = []; this.group = null; this.outMode = false; this.w = 0; this.h = 0;
    this._drag = null;
    canvas.style.cursor = 'grab';
    canvas.addEventListener('pointerdown', e => {
      this._drag = { x: e.clientX, y: e.clientY };
      canvas.setPointerCapture(e.pointerId); canvas.style.cursor = 'grabbing';
    });
    canvas.addEventListener('pointermove', e => {
      if (!this._drag) return;
      this.az -= (e.clientX - this._drag.x) * .005;
      this.pol = clamp(this.pol - (e.clientY - this._drag.y) * .004, .82, 1.42);
      this._drag = { x: e.clientX, y: e.clientY };
    });
    const end = () => { this._drag = null; canvas.style.cursor = 'grab'; };
    canvas.addEventListener('pointerup', end); canvas.addEventListener('pointercancel', end);
  }
  resize() {
    const w = this.wrap.clientWidth, h = this.wrap.clientHeight;
    if (!w || !h || (w === this.w && h === this.h)) return;
    this.w = w; this.h = h;
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.75));
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }
  applyEnv(e) {
    this.hemi.color.set(e.sky); this.hemi.groundColor.set(e.gr); this.hemi.intensity = e.hemi;
    for (const [l, d] of [[this.key, e.key], [this.fill, e.fill], [this.rim, e.rim]]) {
      l.color.set(d[0]); l.intensity = d[1]; l.position.set(...d[2]);
    }
    this.envExp = e.exp; this.expT = e.exp;
  }
  clear() {
    if (!this.group) return;
    this.scene.remove(this.group);
    this.group.traverse(n => {
      if (n.geometry) n.geometry.dispose();
      if (n.isSprite && n.material) n.material.dispose();
    });
    this.group = null; this.objects = [];
  }
  load(key, now) {
    this.clear();
    const def = buildScene(key);
    this.def = def; this.group = def.group;
    this.scene.add(def.group);
    this._dimmed = false;
    this.sprites = [];
    def.group.traverse(n => { if (n.isSprite) this.sprites.push({ m: n.material, base: n.material.opacity, ph: Math.random() * 6 }); });
    this.radius = def.radius; this.targetY = def.targetY;
    this.az = def.az ?? -.55;
    this.applyEnv(ENVS[def.env] || ENVS.day);
    const n = def.objects.length, stag = Math.min(.06, 1.15 / Math.max(1, n));
    def.objects.forEach((o, i) => {
      o.t0 = now + (o.ground ? .12 : .34 + i * stag + Math.random() * .05);
      o.dur = o.ground ? .6 : .42;
      o.node.scale.set(0, o.ground ? 1 : 0, 0);
      o.node.visible = false;
    });
    this.objects = def.objects;
    this.outMode = false; this.radScaleT = 1; this.pulse = 0;
  }
  out(now) {
    if (!this.objects.length) return Promise.resolve();
    this.outMode = true;
    const n = this.objects.length;
    this.objects.forEach((o, i) => { o.t1 = now + (n - 1 - i) * .008 + Math.random() * .04; });
    return new Promise(r => setTimeout(r, (n * 8 + 340) | 0));
  }
  reveal(dim) {
    this.pulse = 1; this.radScaleT = .92;
    this._dimmed = true;
    this.expT = dim ? .34 : (this.envExp || 1.1) * 1.05;
  }
  preview(onn) {
    if (this._dimmed) return;
    this.radScaleT = onn ? .955 : 1;
    this.expT = (this.envExp || 1.1) * (onn ? 1.28 : 1);
    this.pulse = Math.max(this.pulse, onn ? .22 : 0);
  }
  update(now, dt, speed) {
    if (this.w < 4) return;
    for (const o of this.objects) {
      let s, popY = 0;
      if (this.outMode) {
        const q = clamp((now - o.t1) / .3, 0, 1);
        s = Math.max(0, 1 - inBack(q));
      } else {
        const q = clamp((now - o.t0) / o.dur, 0, 1);
        s = q <= 0 ? 0 : outBack(q);
        popY = (1 - outCubic(q)) * .3;
      }
      o.node.visible = s > .001;
      if (o.ground) o.node.scale.set(s, 1, s);
      else o.node.scale.setScalar(s);
      let y = o.baseY + (o.ground ? 0 : popY);
      if (o.bob) y += Math.sin(now * o.bob.sp + (o.bob.ph || 0)) * o.bob.amp;
      o.node.position.y = y;
      if (o.spin) o.node.rotation.y = o.rot0 + now * o.spin;
    }
    this.pulse *= Math.exp(-dt * 2.6);
    if (this.sprites) for (let i = 0; i < this.sprites.length; i++) {
      const sp = this.sprites[i];
      sp.m.opacity = sp.base * (.82 + .18 * Math.sin(now * 2.2 + sp.ph));
    }
    this.exp += (this.expT - this.exp) * Math.min(1, dt * 5);
    this.renderer.toneMappingExposure = this.exp;
    this.radScale += (this.radScaleT - this.radScale) * Math.min(1, dt * 4);
    if (!this._drag) this.az += dt * speed * (1 + this.pulse * 9);
    const fit = clamp(Math.pow(1.15 / Math.max(.4, this.camera.aspect), .72), 1, 2);
    const r = this.radius * this.radScale * fit, sp = Math.sin(this.pol);
    this.scene.fog.near = r * 1.15 + 3; this.scene.fog.far = r * 2.3 + 3;
    this.camera.position.set(Math.sin(this.az) * r * sp, this.targetY + Math.cos(this.pol) * r, Math.cos(this.az) * r * sp);
    this.camera.lookAt(0, this.targetY, 0);
    this.renderer.render(this.scene, this.camera);
  }
}

export function createArena(wrapA, wrapB, canvasA, canvasB) {
  const A = new Viewport(wrapA, canvasA), Bv = new Viewport(wrapB, canvasB);
  let speed = .05, running = true, last = performance.now() / 1000;
  const ro = new ResizeObserver(() => { A.resize(); Bv.resize(); });
  ro.observe(wrapA); ro.observe(wrapB);
  A.resize(); Bv.resize();
  const frame = () => {
    A.resize(); Bv.resize();
    const now = performance.now() / 1000, dt = Math.min(.05, now - last); last = now;
    A.update(now, dt, speed); Bv.update(now, dt, speed * 1.08);
  };
  (function loop() {
    if (!running) return;
    requestAnimationFrame(loop);
    frame();
  })();
  // keep rendering (assembly, counts) even if rAF is starved, e.g. hidden iframe
  const iv = setInterval(() => { if (running && performance.now() / 1000 - last > .3) frame(); }, 250);
  const t = () => performance.now() / 1000;
  return {
    load(a, b) { const n = t(); A.load(a, n); Bv.load(b, n); },
    loadSingle(k) { Bv.clear(); A.load(k, t()); },
    reveal(dimA, dimB) { A.reveal(dimA); Bv.reveal(dimB); },
    preview(side) { A.preview(side === 'a' || side === 'tie'); Bv.preview(side === 'b' || side === 'tie'); },
    out() { const n = t(); return Promise.all([A.out(n), Bv.out(n)]); },
    setSpeed(v) { speed = v; },
    dispose() { running = false; clearInterval(iv); ro.disconnect(); A.clear(); Bv.clear(); A.renderer.dispose(); Bv.renderer.dispose(); },
  };
}
