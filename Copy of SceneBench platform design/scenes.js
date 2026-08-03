// SceneBench — procedural multi-object scene builders
import * as THREE from 'https://unpkg.com/three@0.184.0/build/three.module.js';

const PI = Math.PI, TAU = PI * 2;
const mats = new Map();
export function mat(c, o = {}) {
  const key = [c, o.r ?? .88, o.m ?? 0, o.e ?? 0, o.ei ?? 1, o.flat ? 1 : 0].join('|');
  if (!mats.has(key)) mats.set(key, new THREE.MeshStandardMaterial({
    color: c, roughness: o.r ?? .88, metalness: o.m ?? 0, flatShading: !!o.flat,
    emissive: o.e ?? 0x000000, emissiveIntensity: o.ei ?? 1
  }));
  return mats.get(key);
}
let _glowTex;
function glowTexture() {
  if (_glowTex) return _glowTex;
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const x = c.getContext('2d');
  const g = x.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(.32, 'rgba(255,255,255,.4)'); g.addColorStop(1, 'rgba(255,255,255,0)');
  x.fillStyle = g; x.fillRect(0, 0, 128, 128);
  _glowTex = new THREE.CanvasTexture(c);
  return _glowTex;
}

class B {
  constructor() { this.group = new THREE.Group(); this.objects = []; }
  reg(node, o = {}) {
    this.group.add(node);
    this.objects.push({ node, ground: !!o.ground, bob: o.bob, spin: o.spin, baseY: node.position.y, rot0: node.rotation.y });
    return node;
  }
  make(geo, m, x, y, z, o = {}) {
    const ms = new THREE.Mesh(geo, m);
    ms.position.set(x, y, z); ms.rotation.set(o.rx || 0, o.ry || 0, o.rz || 0);
    if (o.s) ms.scale.setScalar(o.s);
    ms.castShadow = o.cast !== false; ms.receiveShadow = o.recv !== false;
    if (o.to) { o.to.add(ms); return ms; }
    return this.reg(ms, o);
  }
  box(w, h, d, m, x, y, z, o) { return this.make(new THREE.BoxGeometry(w, h, d), m, x, y, z, o); }
  cyl(rt, rb, h, m, x, y, z, o = {}) { return this.make(new THREE.CylinderGeometry(rt, rb, h, o.seg || 20), m, x, y, z, o); }
  sph(r, m, x, y, z, o = {}) { return this.make(new THREE.SphereGeometry(r, o.seg || 18, Math.max(8, ((o.seg || 18) * .7) | 0)), m, x, y, z, o); }
  ico(r, det, m, x, y, z, o) { return this.make(new THREE.IcosahedronGeometry(r, det), m, x, y, z, o); }
  cone(r, h, m, x, y, z, o = {}) { return this.make(new THREE.ConeGeometry(r, h, o.seg || 16), m, x, y, z, o); }
  torus(R, r, m, x, y, z, o = {}) { return this.make(new THREE.TorusGeometry(R, r, 12, 40), m, x, y, z, o); }
  glow(c, s, x, y, z, op = .45, o = {}) {
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(), color: c, transparent: true, opacity: op, depthWrite: false, blending: THREE.AdditiveBlending }));
    sp.position.set(x, y, z); sp.scale.set(s, s, 1);
    if (o.to) { o.to.add(sp); return sp; }
    return this.reg(sp, o);
  }
  ground(r, c) {
    const ms = new THREE.Mesh(new THREE.CircleGeometry(r, 64), mat(c, { r: .97 }));
    ms.rotation.x = -PI / 2; ms.receiveShadow = true;
    return this.reg(ms, { ground: true });
  }
  sub(x, y, z, o = {}) {
    const g = new THREE.Group(); g.position.set(x, y, z); if (o.ry) g.rotation.y = o.ry;
    return this.reg(g, o);
  }
}

function tree(b, x, z, s, leaf = 0x2f4d30, o = {}) {
  const g = b.sub(x, 0, z, o);
  b.cyl(.07 * s, .11 * s, .95 * s, mat(0x4a3a2c), 0, .47 * s, 0, { to: g, seg: 8 });
  b.ico(.6 * s, 1, mat(leaf, { flat: true }), 0, 1.25 * s, 0, { to: g });
  b.ico(.38 * s, 1, mat(leaf, { flat: true }), .3 * s, .95 * s, .12 * s, { to: g });
  return g;
}
function coin(b, x, y, z, c = 0xffc23d, ph = 0) {
  const g = b.sub(x, y, z, { spin: 2.4, bob: { amp: .05, sp: 1.6, ph } });
  b.cyl(.23, .23, .055, mat(0x7a5410, { e: c, ei: 1.5, r: .5, m: .4 }), 0, 0, 0, { rx: PI / 2, to: g, seg: 22 });
  return g;
}
function cloud(b, x, y, z, c = 0xe6eaf1, ph = 0) {
  const g = b.sub(x, y, z, { bob: { amp: .09, sp: .5, ph } });
  b.sph(.5, mat(c, { flat: true }), 0, 0, 0, { to: g, cast: false, seg: 10 });
  b.sph(.34, mat(c, { flat: true }), .5, -.06, .1, { to: g, cast: false, seg: 10 });
  b.sph(.3, mat(c, { flat: true }), -.48, -.08, -.06, { to: g, cast: false, seg: 10 });
  return g;
}
function island(b, x, y, z, r, h, grass, rock) {
  const g = b.sub(x, y, z);
  b.cyl(r, r * .92, .5, mat(grass, { flat: true }), 0, -.25, 0, { to: g, seg: 9 });
  b.cone(r * .94, h, mat(rock, { flat: true }), 0, -.5 - h / 2 + .01, 0, { to: g, rx: PI, seg: 9 });
  return g;
}
function lantern(b, x, y, z, c = 0xffc46b, s = .1, ph = 0) {
  const g = b.sub(x, y, z, { bob: { amp: .03, sp: 1.1, ph } });
  b.sph(s, mat(0x552e10, { e: c, ei: 1.7 }), 0, 0, 0, { to: g, cast: false, seg: 10 });
  b.glow(c, s * 6.5, 0, 0, 0, .4, { to: g });
  return g;
}

// ————— A modern house with a pool —————
function house_a(b) {
  const white = mat(0xe8e4da, { r: .82 }), charcoal = mat(0x232327, { r: .8 }), slab = mat(0x121215),
    wood = mat(0x8a6a4c), stone = mat(0x6e6862, { flat: true }),
    warmGlass = mat(0x1c1610, { e: 0xffc98a, ei: .95, r: .4 });
  b.ground(6.8, 0x191816);
  b.box(7.2, .22, 5, mat(0x22201e), 0, .11, .3);
  b.box(2.5, .07, 1.7, mat(0x25311e), -2.15, .25, 1.6);
  b.box(3.8, 1.5, 2.7, white, -1.1, .97, -.7);
  b.box(3.3, 1.05, .08, warmGlass, -1.1, .8, .67);
  for (let i = 0; i < 4; i++) b.box(.05, 1.05, .1, charcoal, -2.3 + i * .8, .8, .68);
  b.box(4.2, .1, 3.1, slab, -1.1, 1.77, -.55);
  b.box(4.8, 1.25, 2.3, charcoal, -.2, 2.45, -.95);
  b.box(2.6, .7, .08, warmGlass, .35, 2.42, .22);
  b.box(5.2, .12, 2.6, slab, -.2, 3.13, -.95);
  b.box(.38, 3, 1.5, stone, .95, 1.5, -1.35);
  b.box(.3, .45, .3, charcoal, .95, 3.35, -1.35);
  b.box(.5, .16, .4, charcoal, -1.6, 3.25, -1.3);
  b.box(3, .16, 1.7, mat(0x33312e), 2.15, .3, 1.3);
  b.box(2.7, .09, 1.42, mat(0x06343c, { e: 0x3fd9e8, ei: 1.15, r: .3 }), 2.15, .35, 1.3);
  b.glow(0x54e2f0, 2.3, 2.15, .62, 1.3, .3);
  for (let i = 0; i < 5; i++) b.box(.56, .05, 2.1, wood, -2.5 + i * .62, .26, 2.9 - 1.4, { cast: false });
  for (const dx of [.35, 1.35]) {
    const g = b.sub(1.3 + dx, .36, 2.35, { ry: -.25 });
    b.box(.8, .07, .3, wood, 0, 0, 0, { to: g });
    b.box(.34, .07, .3, wood, -.5, .12, 0, { rz: .6, to: g });
  }
  tree(b, -3.1, -1.7, 1.5, 0x2e4a2e); tree(b, 3.2, -1.5, 1.1, 0x33502f); tree(b, 4.3, .6, .8, 0x2e4a2e);
  for (const [x, z, s] of [[-.4, 2.6, .26], [.4, 2.75, .2], [-3, 1.9, .3]]) b.ico(s, 1, mat(0x33502f, { flat: true }), x, .28 + s * .4, z);
  for (let i = 0; i < 5; i++) b.box(.52, .05, .78, mat(0x3c3a37), -.55 - i * .18, .25, 2.7 - i * .55, { cast: false });
  for (let i = 0; i < 3; i++) {
    b.box(.06, .12, .06, mat(0x333, { e: 0xffd9a3, ei: 2 }), -.95 - i * .18, .32, 2.55 - i * .55, { cast: false });
    b.glow(0xffd9a3, .5, -.95 - i * .18, .36, 2.55 - i * .55, .38);
  }
  b.box(2, .42, .14, mat(0x2a2826), -3.1, .43, 1.1, { ry: .12 });
  b.glow(0xffc98a, 1.9, -1.1, .85, 1.05, .34);
  b.glow(0xffc98a, 1.5, .35, 2.42, .6, .3);
  return { radius: 11.3, targetY: 1.15, env: 'dusk', az: -.55 };
}

// ————— courtyard timber house (model B take) —————
function house_b(b) {
  const woodA = mat(0xa8875f, { r: .85 }), woodB = mat(0x96754f), dark = mat(0x141416),
    warmGlass = mat(0x241a12, { e: 0xffd9a8, ei: .85, r: .4 });
  b.ground(6.8, 0x171614);
  b.box(6.8, .2, 4.8, mat(0x21201e), .1, .1, .1);
  b.cyl(1.85, 1.85, .07, mat(0x2e2b27, { r: .98 }), .2, .23, .5, { seg: 40, cast: false });
  b.box(4.4, 1.35, 1.9, woodA, -.5, .9, -1.45);
  b.box(1.8, 1.35, 2.9, woodB, 2.35, .9, .5);
  b.box(3.6, .88, .08, warmGlass, -.5, .72, -.46);
  b.box(.08, .88, 2.3, warmGlass, 1.41, .72, .5);
  for (let i = 0; i < 5; i++) b.box(.06, .88, .09, dark, -2 + i * .75, .72, -.45);
  b.box(5.6, .14, 3.9, mat(0x3f3f46), .5, 1.72, -.35);
  for (const [x, z] of [[-2.5, 1.6], [-2.5, -2.35], [3.5, 1.9]]) b.cyl(.05, .05, 1.44, dark, x, .92, z, { seg: 8 });
  b.box(1.5, .14, 1.1, mat(0x33312e), -2.3, .27, 1.5);
  b.box(1.3, .07, .9, mat(0x0a3a40, { e: 0x49cfd9, ei: .85, r: .3 }), -2.3, .31, 1.5);
  b.glow(0x62d9e4, 1.4, -2.3, .5, 1.5, .25);
  const t = b.sub(3.6, 0, -1.7);
  b.cyl(.11, .16, 1.3, mat(0x4a3a2c), 0, .65, 0, { to: t, seg: 8 });
  b.ico(.78, 1, mat(0x39512e, { flat: true }), 0, 1.7, 0, { to: t });
  b.ico(.5, 1, mat(0x33502f, { flat: true }), .45, 1.25, .25, { to: t });
  for (const [x, z, s] of [[-1.5, 1.9, .3], [-1.1, 2.2, .2], [-3.2, .4, .26], [1.5, 1.8, .22]]) b.ico(s, 0, mat(0x4a4642, { flat: true }), x, .2 + s * .5, z);
  for (let i = 0; i < 3; i++) b.box(.9, .07, .5, mat(0x3c3a37), .2, .21 - i * 0, 2.55 + i * .42, { cast: false });
  for (let i = 0; i < 7; i++) b.box(.09, .66, .03, mat(0x5c4a38), -3.3 + i * .32, .53, -2.5);
  for (const [x, z] of [[1.3, 2.3], [-1, -2.6]]) {
    b.cyl(.03, .03, .85, dark, x, .62, z, { seg: 6 });
    b.box(.15, .18, .15, mat(0x3a2c14, { e: 0xffc98a, ei: 1.6 }), x, 1.1, z, { cast: false });
    b.glow(0xffc98a, .8, x, 1.1, z, .4);
  }
  b.box(1.1, .08, .34, woodB, .4, .4, 1.9);
  b.box(.08, .18, .3, dark, 0, .3, 1.9); b.box(.08, .18, .3, dark, .8, .3, 1.9);
  return { radius: 11, targetY: 1.05, env: 'day', az: .5 };
}

// ————— floating island platformer (lush) —————
function plat_a(b) {
  const grass = 0x55a04a, rock = 0x6d4c36;
  island(b, 0, 0, 0, 2.25, 2.6, grass, rock);
  island(b, -2.55, .65, .3, 1.2, 1.7, grass, rock);
  island(b, 2.85, 1.5, -.3, 1.3, 1.8, grass, rock);
  b.reg(new THREE.Group()); // spacer keeps stagger rhythm
  for (const [x, y, z, r, ph] of [[1.15, .6, .5, .48, 0], [1.95, 1.05, .1, .4, 2]]) {
    const g = b.sub(x, y, z, { bob: { amp: .06, sp: .9, ph } });
    b.cyl(r, r * .88, .28, mat(grass, { flat: true }), 0, 0, 0, { to: g, seg: 8 });
    b.cone(r * .9, .55, mat(rock, { flat: true }), 0, -.4, 0, { to: g, rx: PI, seg: 8 });
  }
  for (let i = 0; i < 7; i++) {
    const t = i / 6, x = -1.05 - t * 1.35, y = .12 + Math.sin(t * PI) * .3, z = .28 + t * .1;
    b.box(.38, .05, .17, mat(0x8a6444), x, y, z, { rz: Math.cos(t * PI) * .3 });
  }
  coin(b, -.5, 1, .7, 0xffc23d, 0); coin(b, .1, 1.15, .6, 0xffc23d, 1); coin(b, .7, 1.05, .5, 0xffc23d, 2);
  coin(b, 1.5, 1.7, .3, 0xffc23d, 3); coin(b, 2.85, 2.75, -.3, 0xffc23d, 4);
  b.glow(0xffd875, .6, .1, 1.15, .6, .35);
  for (const [x, z, r, h, tl] of [[-3.1, .7, .15, .5, .2], [-2.6, .2, .11, .34, -.25], [-2.85, .9, .09, .3, .4]])
    b.cone(r, h, mat(0x0d4f46, { e: 0x53e6d0, ei: 1.2, flat: true }), x, 1 + h / 2 - .05, z, { rz: tl, seg: 6 });
  b.glow(0x53e6d0, 1.3, -2.85, 1.25, .55, .4);
  const flag = b.sub(2.85, 1.5, -.3);
  b.cyl(.035, .035, 1.15, mat(0xe8e4da), 0, .57, 0, { to: flag, seg: 8 });
  b.cone(.14, .42, mat(0xff5c47, { flat: true }), .2, 1.02, 0, { rz: -PI / 2, to: flag, seg: 4 });
  b.cyl(.14, .18, .07, mat(rock, { flat: true }), 0, .03, 0, { to: flag, seg: 8 });
  const spring = b.sub(.85, 0, .95);
  b.cyl(.3, .34, .12, mat(0xd94a3d), 0, .06, 0, { to: spring });
  b.cyl(.2, .2, .1, mat(0xf2f1ed), 0, .2, 0, { to: spring });
  tree(b, -.9, -.9, 1.15, 0x2f5c33); tree(b, .4, -1.3, .85, 0x357040);
  b.ico(.26, 1, mat(0x33502f, { flat: true }), -1.4, .12, .8);
  cloud(b, -3.4, 2.7, -1.6, 0xe6eaf1, 0); cloud(b, 3.1, 3.2, -2, 0xe6eaf1, 2); cloud(b, .2, 3.6, -3, 0xe6eaf1, 4);
  for (const [x, y, z, s, ph] of [[-1.6, 1.9, -1, .2, 1], [3.6, .7, 1, .16, 3], [-3.6, 2, .8, .13, 5]])
    b.ico(s, 0, mat(0x6d5a48, { flat: true }), x, y, z, { bob: { amp: .08, sp: .7, ph } });
  return { radius: 12, targetY: 1.2, env: 'day', az: -.5 };
}

// ————— floating platformer (dusk tower) —————
function plat_b(b) {
  const grass = 0x3f7d6d, rock = 0x584a70;
  island(b, 0, 0, 0, 2.5, 3, grass, rock);
  const spots = [[1.9, .95, .6, .62], [.4, 1.75, 1.75, .55], [-1.55, 2.5, .9, .5], [-1.1, 3.25, -.85, .48]];
  for (const [x, y, z, r] of spots) {
    const g = b.sub(x, y, z, { bob: { amp: .04, sp: .8, ph: x * 3 } });
    b.cyl(r, r * .88, .26, mat(grass, { flat: true }), 0, 0, 0, { to: g, seg: 8 });
    b.cone(r * .88, .5, mat(rock, { flat: true }), 0, -.38, 0, { to: g, rx: PI, seg: 8 });
  }
  island(b, .35, 4.1, -1.6, 1.05, 1.4, grass, rock);
  const portal = b.sub(.35, 4.1, -1.6);
  b.box(.9, .12, .9, mat(rock, { flat: true }), 0, .06, 0, { to: portal });
  b.torus(.58, .065, mat(0x3d1038, { e: 0xe560c8, ei: 1.6 }), 0, .85, 0, { to: portal });
  b.glow(0xe560c8, 2, 0, .85, 0, .45, { to: portal });
  coin(b, 1.9, 1.75, .6, 0xffa03d, 0); coin(b, .4, 2.55, 1.75, 0xffa03d, 1.5);
  coin(b, -1.55, 3.3, .9, 0xffa03d, 3); coin(b, -1.1, 4.05, -.85, 0xffa03d, 4.5);
  for (const [x, z, s] of [[-1.2, .6, 1], [.9, -1.1, .8]]) {
    const p = b.sub(x, 0, z);
    b.cyl(.06 * s, .09 * s, .5 * s, mat(0x3a2c22), 0, .25 * s, 0, { to: p, seg: 6 });
    b.cone(.42 * s, .7 * s, mat(0x2a4a44, { flat: true }), 0, .75 * s, 0, { to: p, seg: 7 });
    b.cone(.3 * s, .55 * s, mat(0x315850, { flat: true }), 0, 1.15 * s, 0, { to: p, seg: 7 });
  }
  lantern(b, 1.1, 1.9, -.9, 0xffd9a3, .09, 0); lantern(b, -2.2, 2.9, .2, 0xffd9a3, .08, 2); lantern(b, 1.7, 3.4, -1.7, 0xffd9a3, .07, 4);
  for (const [x, z, r, h, tl] of [[1.3, 1.4, .16, .55, .2], [1.7, 1, .11, .36, -.3]])
    b.cone(r, h, mat(0x3d1038, { e: 0xc44fd4, ei: 1.1, flat: true }), x, h / 2 - .02, z, { rz: tl, seg: 6 });
  const fall = b.sub(-2.1, -.6, .9);
  b.box(.44, 1.9, .07, mat(0x0e3540, { e: 0x6fd8e8, ei: .55, r: .3 }), 0, -.6, 0, { to: fall, cast: false });
  b.glow(0x6fd8e8, .9, 0, .38, .05, .3, { to: fall });
  cloud(b, -3.3, 1.6, -1.8, 0x8b93ad, 1); cloud(b, 3.4, 4, -2.2, 0x8b93ad, 3);
  for (const [x, y, z, s, ph] of [[2.9, 2.4, .8, .18, 0], [-2.9, 4.2, -1.2, .14, 2], [2.2, 5, -2, .12, 4]])
    b.ico(s, 0, mat(rock, { flat: true }), x, y, z, { bob: { amp: .09, sp: .6, ph } });
  return { radius: 12.4, targetY: 2.3, env: 'dusk', az: .6 };
}

// ————— tiny ramen shop at night (street corner) —————
function ramen_a(b) {
  const hull = mat(0x352c25), dark = mat(0x15151a), wood = mat(0xa57c50);
  b.ground(6.8, 0x141416);
  b.box(5.4, .14, 2.8, mat(0x232327), -.2, .07, 1.1, { cast: false });
  b.box(2.7, 2.7, 2.1, hull, -.8, 1.49, -.8);
  b.box(3, .12, 2.4, dark, -.8, 2.9, -.8);
  b.cyl(.28, .28, .5, mat(0x4a4a52), -1.6, 3.2, -1.2, { seg: 12 });
  b.box(.5, .35, .45, mat(0x3c3c44), .1, 3.13, -1.3);
  b.box(.7, .5, .07, mat(0x241a10, { e: 0xffc98a, ei: 1 }), -1.45, 2.25, .27, { cast: false });
  b.box(.5, .5, .07, mat(0x241a10, { e: 0xffc98a, ei: .35 }), -.1, 2.25, .27, { cast: false });
  b.box(2.4, .5, .14, mat(0x6e2a26), -.8, 1.62, .3);
  for (let i = 0; i < 3; i++) b.box(.56, .52, .025, mat(i === 1 ? 0x9e3434 : 0x8e3030), -1.42 + i * .62, 1.1, .33, { rz: (i - 1) * .05, cast: false });
  b.box(2.3, 1.1, .07, mat(0x2a1a0e, { e: 0xffb066, ei: 1.3 }), -.8, .85, .1, { cast: false });
  b.glow(0xffb066, 2.6, -.8, .95, .5, .5);
  b.box(2.4, .11, .5, wood, -.8, .74, .62);
  b.box(2.4, .55, .09, mat(0x2d2620), -.8, .44, .62);
  for (const dx of [-.5, .1]) {
    b.cyl(.09, .06, .08, mat(0xe8e4da), -.8 + dx, .84, .58, { seg: 10, cast: false });
    b.glow(0xffffff, .35, -.8 + dx, 1.05, .58, .18, { bob: { amp: .05, sp: .7, ph: dx * 9 } });
  }
  for (let i = 0; i < 3; i++) {
    b.cyl(.03, .03, .42, dark, -1.5 + i * .7, .35, 1.25, { seg: 8 });
    b.cyl(.17, .17, .07, mat(0xc23b30), -1.5 + i * .7, .6, 1.25, { seg: 12 });
  }
  b.box(.34, 1.5, .14, mat(0x2a0f0d, { e: 0xff4a3d, ei: 1.35 }), .72, 1.95, .5);
  for (let i = 0; i < 3; i++) b.box(.2, .07, .03, mat(0x431410, { e: 0xffe9dc, ei: 1.7 }), .72, 2.42 - i * .34, .58, { cast: false });
  b.glow(0xff4a3d, 1.8, .72, 1.95, .62, .5);
  b.cyl(.04, .05, 2.4, mat(0x1c1c20), 2.5, 1.27, .9, { seg: 8 });
  for (let i = 0; i < 5; i++) {
    const t = i / 4, x = 2.5 - t * 1.9, y = 2.42 + t * .35 - Math.sin(t * PI) * .28, z = .9 - t * 1.1;
    lantern(b, x, y, z, 0xffc46b, .095, i * 1.3);
  }
  b.box(.62, 1.45, .6, mat(0x2a4d8f), 1.85, .87, -1);
  b.box(.5, 1, .06, mat(0x10263f, { e: 0x9fd4ff, ei: 1.1 }), 1.85, 1, -.68, { cast: false });
  b.glow(0x9fd4ff, 1.1, 1.85, 1, -.6, .4);
  for (const [x, y, z, r] of [[-2.5, .29, .3, 0], [-2.5, .59, .32, .3], [-2.05, .29, .5, .5]]) b.box(.42, .3, .42, mat(0x5c4a38), x, y, z, { ry: r });
  b.cyl(.19, .16, .5, mat(0x2e2e34), -2.7, .39, 1.4, { seg: 12 });
  b.cyl(.14, .11, .22, mat(0x703a2c), 1.55, .25, .8, { seg: 10 });
  b.ico(.2, 1, mat(0x33502f, { flat: true }), 1.55, .45, .8);
  b.cyl(.06, .07, 3.4, mat(0x232326), -3, 1.77, -1.7, { seg: 8 });
  b.box(.95, .07, .07, mat(0x232326), -3, 3.2, -1.7);
  b.cyl(.15, .15, .38, mat(0x2e2e34), -2.8, 2.75, -1.7, { seg: 10 });
  b.box(.45, .3, .35, mat(0x35353c), .65, 2.5, -1.86);
  return { radius: 11.8, targetY: 1.45, env: 'night', az: .35 };
}

// ————— ramen alley (model B take) —————
function ramen_b(b) {
  const bldgL = mat(0x26221f), bldgR = mat(0x211f24), dark = mat(0x15151a);
  b.ground(6.8, 0x131316);
  b.box(2.6, .09, 5.2, mat(0x1d1d22), 0, .05, .5, { cast: false });
  b.box(2.2, 3.6, 3.4, bldgL, -2.2, 1.8, -.6);
  b.box(2.2, 3.1, 3, bldgR, 2.2, 1.55, -.7);
  b.box(2.4, .1, 3.6, dark, -2.2, 3.65, -.6); b.box(2.4, .1, 3.2, dark, 2.2, 3.65 - .5, -.7);
  const win = (x, y, z, ry, c, ei) => b.box(.34, .44, .05, mat(0x1c150e, { e: c, ei }), x, y, z, { ry, cast: false });
  win(-1.06, 2.6, .1, 0, 0xffc98a, .9); win(-1.06, 1.9, -.9, 0, 0xffc98a, .3); win(-1.06, 3, -1.3, 0, 0x7de8dc, .8);
  win(1.06, 2.4, -.2, 0, 0xffc98a, .75); win(1.06, 1.7, -1.1, 0, 0xffc98a, .25); win(1.06, 2.9, .4, 0, 0xffc98a, .5);
  win(-1.7, 2.9, 1.11, 0, 0x7de8dc, .6); win(2.6, 2.5, .81, 0, 0xffc98a, .45);
  b.box(1.9, 1.7, 1.4, mat(0x3a2e26), 0, .94, -2.1);
  b.box(2.1, .1, 1.6, dark, 0, 1.84, -2.1);
  b.box(1.6, .95, .07, mat(0x2a1a0e, { e: 0xffb066, ei: 1.45 }), 0, .75, -1.36, { cast: false });
  b.glow(0xffb066, 2.8, 0, .9, -1, .55);
  for (let i = 0; i < 2; i++) b.box(.6, .45, .025, mat(0x355a8e), -.35 + i * .7, 1.32, -1.32, { rz: (i - .5) * .07, cast: false });
  b.box(1.7, .1, .4, mat(0xa57c50), 0, .68, -1.15);
  for (let i = 0; i < 2; i++) {
    b.cyl(.03, .03, .4, dark, -.3 + i * .6, .32, -.75, { seg: 8 });
    b.cyl(.16, .16, .06, mat(0x2e6e8e), -.3 + i * .6, .55, -.75, { seg: 12 });
  }
  const signs = [[-1, 2.5, .9, .3, .95, 0xff4a3d, 1.4], [-1, 1.55, 1.3, .3, .6, 0x7de8dc, 1.25], [1.02, 2.1, .6, .28, .5, 0xffe9dc, 1.1]];
  for (const [x, y, z, w, h, c, ei] of signs) {
    b.box(w, h, .1, mat(0x1a0c0a, { e: c, ei }), x, y, z);
    b.glow(c, h * 1.9, x, y, z + .15 * Math.sign(-x), .42);
  }
  for (let i = 0; i < 4; i++) lantern(b, -.85 + i * .58, 2.5 + Math.sin(i / 3 * PI) * -.18, .4, 0xffc46b, .09, i);
  for (let i = 0; i < 4; i++) lantern(b, -.85 + i * .58, 2.95 + Math.sin(i / 3 * PI) * -.15, -.7, 0xffb066, .08, i + 2);
  for (const [x, y, z] of [[-1.04, 2.05, -.35], [1.04, 2.2, .15], [1.04, 1.5, -.7]]) b.box(.4, .28, .3, mat(0x35353c), x, y, z);
  b.cyl(.05, .05, 3, mat(0x1b1b1f), 1.04, 1.5, -1.5, { seg: 8 });
  b.box(.4, .05, .05, mat(0x1b1b1f), .85, 2.9, -1.5);
  const lad = b.sub(-1.06, 0, 1.3);
  for (const dx of [-.14, .14]) b.cyl(.025, .025, 2.2, dark, dx, 1.4, 0, { to: lad, seg: 6 });
  for (let i = 0; i < 5; i++) b.cyl(.02, .02, .28, dark, 0, .6 + i * .42, 0, { to: lad, rz: PI / 2, seg: 6 });
  for (const [x, z, ry] of [[1.5, .9, .2], [1.15, 1.25, .6], [1.45, 1.6, 0]]) b.box(.4, .32, .4, mat(0x5c4a38), x, .25, z, { ry });
  b.cyl(.21, .19, .5, mat(0x3f3a33), -1.6, .34, 1.9, { seg: 12 });
  b.glow(0xdddddd, .5, 0, 2.1, -2.1, .12, { bob: { amp: .06, sp: .5, ph: 1 } });
  return { radius: 12, targetY: 1.7, env: 'night', az: .12 };
}

const REG = { house_a, house_b, plat_a, plat_b, ramen_a, ramen_b };
export const SCENE_KEYS = Object.keys(REG);
export function buildScene(key) {
  const b = new B();
  const def = REG[key] ? REG[key](b) : REG.house_a(b);
  return { group: b.group, objects: b.objects, ...def };
}
