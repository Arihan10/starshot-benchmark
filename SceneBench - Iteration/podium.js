// Port of prod_client/src/components/leaderboard/{podiumLayout.ts,loose.ts,sweep.ts,Podium.tsx}
// (Starshot-Labs/starshot-benchmark @ jace/prod-client) to plain three.js.
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js";

/* ================= podiumLayout ================= */
export const CUBE = 1;
const SEAM = 0.07;
const STRIDE = CUBE + SEAM;
export const PLATE = 2 * STRIDE;
export const PLATE_H = 0.9;

export const ACROSS = 0.7071;
export const DEPTH = 0.4082;
export const RISE = 0.8165;

function spot(p, s = 0) { return [p + s / 2, -p + s / 2]; }
function skyline(s, y) { return y * RISE - s * DEPTH; }

function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const SEED = 0x5cebe4;
function site(kind, a, b) {
  let h = SEED ^ Math.imul(kind + 1, 0x27d4eb2d);
  h = Math.imul(h ^ (a + 8192), 0x9e3779b1);
  h = Math.imul(h ^ (b + 8192), 0x85ebca6b);
  const rand = mulberry32(h >>> 0);
  rand();
  return rand;
}

const FLIGHT = 0.16;
function piece(rest, rank, [from, to], rand) {
  const start = from + rank * Math.max(0, to - from - FLIGHT);
  const [x, , z] = rest;
  const len = Math.hypot(x, z) || 1;
  const ox = (x || rand() - 0.5) / len;
  const oz = (z || rand() - 0.5) / len;
  const out = 3 + rand() * 4;
  return {
    rest,
    from: [rest[0] + ox * out, rest[1] + 4 + rand() * 6, rest[2] + oz * out],
    spin: [(rand() - 0.5) * 1.7, (rand() - 0.5) * 2.4, (rand() - 0.5) * 1.7],
    start,
    end: start + FLIGHT,
  };
}

const REACH_D = 13;
const REACH_W = 9;
const COAST = 1.35;
function shore(w) {
  const u = w / REACH_W;
  if (Math.abs(u) >= 1) return 0;
  const t = Math.atan2(w * 1.6, REACH_W) * 2.2;
  const wander = (Math.sin(t * 3.1 + 0.7) * 0.6 + Math.sin(t * 7.3 + 2.2) * 0.4) * COAST;
  return REACH_D * Math.sqrt(1 - u * u) + wander;
}

const KEEL = 8;
function draught(k) { return (1 - k / (KEEL + 1)) ** 0.62; }

export function onGround(x, z) {
  const d = (x - z) / PLATE;
  const w = (x + z) / PLATE;
  return Math.abs(d) <= shore(w) + 0.5;
}

function ground() {
  const cells = [];
  for (let k = 0; k <= KEEL; k++) {
    const shrink = k === 0 ? 1 : draught(k);
    for (let w = -REACH_W; w <= REACH_W; w++) {
      const span = shore(w) * shrink;
      if (span < 0.6) continue;
      for (let d = -Math.round(span); d <= Math.round(span); d++) {
        if (((w + d) & 1) !== 0) continue;
        if (Math.abs(d) > span) continue;
        const i = (w + d) / 2;
        const j = (w - d) / 2;
        cells.push({
          x: i * PLATE,
          y: -PLATE_H / 2 - k * PLATE_H,
          z: j * PLATE,
          away: Math.hypot(d, w * 1.4) + k * 26,
        });
      }
    }
  }
  cells.sort((a, b) => a.away - b.away);
  const last = Math.max(1, cells.length - 1);
  return cells.map((c, k) =>
    piece([c.x, c.y, c.z], k / last, [0.0, 0.42], site(0, Math.round(c.x), Math.round(c.z * 7 + c.y))),
  );
}

const GROUND = 0;

function slab(x0, x1, z0, z1, y) {
  const out = [];
  for (let x = x0; x <= x1; x++) for (let z = z0; z <= z1; z++) out.push([x, y, z]);
  return out;
}

function place(cells, p, s, span, rand) {
  const [ox, oz] = spot(p, s);
  const order = [...cells].sort((a, b) => a[1] - b[1]);
  const last = Math.max(1, order.length - 1);
  return order.map(([dx, dy, dz], k) =>
    piece([ox + dx * STRIDE, GROUND + CUBE / 2 + dy * STRIDE, oz + dz * STRIDE], k / last, span, rand),
  );
}

function building(p, s, storeys, w, d, span, rand) {
  const cells = [];
  let x0 = -((w - 1) >> 1);
  let x1 = x0 + w - 1;
  let z0 = -((d - 1) >> 1);
  let z1 = z0 + d - 1;
  for (let y = 0; y < storeys; y++) {
    if (y > 0 && y % 3 === 0) {
      if (x1 - x0 > 0 && rand() < 0.7) {
        if (rand() < 0.5) x0 += 1; else x1 -= 1;
      } else if (z1 - z0 > 0) {
        if (rand() < 0.5) z0 += 1; else z1 -= 1;
      }
    }
    cells.push(...slab(x0, x1, z0, z1, y));
  }
  return place(cells, p, s, span, rand);
}

const ZIGGURAT = [...slab(-1, 1, -1, 1, 0), ...slab(-1, 1, -1, 1, 1), ...slab(-1, 0, -1, 0, 2), [0, 3, 0]];
const GATE = [[-1, 0, 0], [-1, 1, 0], [-1, 2, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0], ...slab(-1, 1, 0, 0, 3), [0, 4, 0]];
const MAST = [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0], ...slab(-1, 1, -1, 1, 5), [0, 6, 0]];
const BRIDGE = [[-2, 0, 0], [-2, 1, 0], [-2, 2, 0], [2, 0, 0], [2, 1, 0], [2, 2, 0], ...slab(-2, 2, 0, 0, 3)];
const COURT = [
  ...slab(-1, 1, -1, 1, 0).filter(([x, , z]) => x !== 0 || z !== 0),
  ...slab(-1, 1, -1, 1, 1).filter(([x, , z]) => x !== 0 || z !== 0),
  [-1, 2, -1], [1, 2, 1],
];
const TURN = [0, 1, 2, 3, 4, 5].flatMap((y) => (y % 2 === 0 ? slab(-1, 0, 0, 0, y) : slab(0, 0, -1, 0, y)));
const LANDMARKS = [ZIGGURAT, GATE, MAST, BRIDGE, COURT, TURN];

const LOT = 2.25;
const DRIFT = 0.9;
const ROWS = [
  { s: 8.6, clear: 0, max: 2, span: [0.2, 0.4] },
  { s: 6.0, clear: 0, max: 2, span: [0.22, 0.42] },
  { s: 3.4, clear: 4.2, max: 3, span: [0.24, 0.44] },
  { s: 0.8, clear: 6.4, max: 3, span: [0.26, 0.46] },
  { s: -1.8, clear: 5.2, max: 5, span: [0.28, 0.48] },
  { s: -4.4, clear: 0, max: 6, span: [0.3, 0.5] },
  { s: -7.0, clear: 0, max: 7, span: [0.33, 0.53] },
  { s: -9.6, clear: 0, max: 8, span: [0.36, 0.56] },
  { s: -12.2, clear: 0, max: 9, span: [0.39, 0.59] },
  { s: -14.8, clear: 0, max: 9, span: [0.42, 0.62] },
  { s: -17.4, clear: 0, max: 8, span: [0.45, 0.65] },
];
function ceiling(s) { return 9.6 + 0.34 * -s; }
function storeys(row) {
  const budget = ceiling(row.s) - skyline(row.s, 0);
  return Math.max(2, Math.min(row.max, Math.floor(budget / (STRIDE * RISE))));
}

const GAP = 5.6;
const PILLAR_W = 3.1;
export const CHALLENGER_P = 2 * GAP;
const BERTHS = [-GAP, 0, GAP, CHALLENGER_P];
function shadowed(p, s) {
  if (s > 0 || -s > 11) return false;
  const x = 2 * p * ACROSS;
  return BERTHS.some((c) => Math.abs(x - 2 * c * ACROSS) < PILLAR_W * ACROSS + 1.3);
}

function city() {
  const out = [];
  ROWS.forEach((row, r) => {
    const tallest = storeys(row);
    const span = [row.span[0], row.span[1]];
    const room = (shore(row.s / PLATE) * PLATE) / 2 - 1.6;
    const lots = Math.floor(room / LOT);
    for (let n = -lots; n <= lots; n++) {
      const rand = site(1, r, n);
      const p = n * LOT + (rand() - 0.5) * 1.2;
      const s = row.s + (rand() - 0.5) * 2 * DRIFT;
      if (Math.abs(p) > room) continue;
      if (Math.abs(p) < row.clear) continue;
      if (shadowed(p, s)) continue;
      if (rand() < 0.08) continue;
      if (row.s < -5 && rand() < 0.28) {
        out.push(...place(LANDMARKS[Math.floor(rand() * LANDMARKS.length)], p, s, span, rand));
        continue;
      }
      const tall = 1 + Math.round(rand() ** 1.05 * (tallest - 1));
      const r2 = rand();
      const w = r2 < 0.1 ? 1 : r2 < 0.5 ? 2 : 3;
      const d = rand() < 0.2 ? 1 : rand() < 0.6 ? 2 : 3;
      out.push(...building(p, s, tall, w, d, span, rand));
    }
  });
  return out;
}

function footing(p, span) {
  return place(slab(-1, 1, -1, 1, 0), p, 0, span, site(2, Math.round(p * 10), 0));
}

const FOOT_TOP = GROUND + CUBE;

export const DECK = ground();
const CITY = city();
const FOOT = [
  { rank: 2, cells: footing(-GAP, [0.58, 0.68]) },
  { rank: 1, cells: footing(0, [0.58, 0.68]) },
  { rank: 3, cells: footing(GAP, [0.58, 0.68]) },
  { rank: 0, cells: footing(CHALLENGER_P, [0.58, 0.68]) },
];
export const BLOCKS = [...CITY, ...FOOT.flatMap((f) => f.cells)];
export const FOOTINGS = FOOT.map((f, i) => ({
  rank: f.rank,
  from: CITY.length + FOOT.slice(0, i).reduce((n, g) => n + g.cells.length, 0),
  count: f.cells.length,
}));

const FLOOR = 1000;
const STUB = 2.4;
export const TALL = 31;
const CURVE = 7;
const HEADROOM = 28;
const FOOTROOM = 34;

export function heightScale(best, challenger) {
  const racing = challenger != null;
  const lo = racing ? challenger - FOOTROOM : FLOOR;
  const hi = racing ? best + HEADROOM : best;
  const span = Math.max(1, hi - lo);
  return (elo) => {
    const u = Math.max(0, Math.min(1, (elo - lo) / span));
    return STUB + (TALL - STUB) * (racing ? u : u ** CURVE);
  };
}

export const PILLARS = [
  { rank: 3, p: GAP, start: 0.66, end: 0.81 },
  { rank: 2, p: -GAP, start: 0.73, end: 0.88 },
  { rank: 1, p: 0, start: 0.8, end: 0.96 },
].map(({ p, ...rest }) => {
  const [x, z] = spot(p);
  return { ...rest, x, z, width: PILLAR_W, depth: PILLAR_W, base: FOOT_TOP };
});

export const CHALLENGER = (() => {
  const [x, z] = spot(CHALLENGER_P);
  return { x, z, width: PILLAR_W, depth: PILLAR_W, base: FOOT_TOP };
})();

export const LABELS = [0.9, 1.0];

const BOXES = [];
for (const d of DECK) BOXES.push([d.rest[0], d.rest[1], d.rest[2], PLATE / 2, PLATE_H / 2, PLATE / 2]);
for (const b of BLOCKS) BOXES.push([b.rest[0], b.rest[1], b.rest[2], CUBE / 2, CUBE / 2, CUBE / 2]);
for (const p of [...PILLARS, CHALLENGER]) BOXES.push([p.x, p.base + TALL / 2, p.z, p.width / 2, TALL / 2, p.depth / 2]);

function spanAt(yaw) {
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [cx, cy, cz, hx, hy, hz] of BOXES) {
    const rx = cx * c + cz * s;
    const rz = -cx * s + cz * c;
    const ex = Math.abs(hx * c) + Math.abs(hz * s);
    const ez = Math.abs(hx * s) + Math.abs(hz * c);
    const x = (rx - rz) * ACROSS;
    const y = cy * RISE - (rx + rz) * DEPTH;
    const spreadX = (ex + ez) * ACROSS;
    const spreadY = hy * RISE + (ex + ez) * DEPTH;
    if (x - spreadX < minX) minX = x - spreadX;
    if (x + spreadX > maxX) maxX = x + spreadX;
    if (y - spreadY < minY) minY = y - spreadY;
    if (y + spreadY > maxY) maxY = y + spreadY;
  }
  return { w: maxX - minX, h: maxY - minY, mid: (maxY + minY) / 2 };
}
const SPAN0 = spanAt(0);
export const VIEW = { w: SPAN0.w, h: SPAN0.h };
export const GROUP_Y = -SPAN0.mid / RISE;

/* ================= loose (physics) ================= */
const RADIUS = CUBE * 0.5;
const GRAVITY = -30;
const BOUNCE = 0.36;
const SKID = 0.72;
const SPIN_DAMP = 0.6;
const ASLEEP = 0.5;
export const MAX_THROW = 40;

const CELL = 3;
const cellKey = (x, z) => (Math.floor(x / CELL) + 128) * 512 + (Math.floor(z / CELL) + 128);
let grid = null;
function addBox(box) {
  for (let x = box.min.x; x <= box.max.x + CELL; x += CELL) {
    for (let z = box.min.z; z <= box.max.z + CELL; z += CELL) {
      const key = cellKey(Math.min(x, box.max.x), Math.min(z, box.max.z));
      const cell = grid.get(key);
      if (cell) { if (!cell.includes(box)) cell.push(box); }
      else grid.set(key, [box]);
    }
  }
}
function field() {
  if (grid) return grid;
  grid = new Map();
  const half = CUBE / 2;
  for (let i = 0; i < BLOCKS.length; i++) {
    const [x, y, z] = BLOCKS[i].rest;
    addBox({ i, min: new THREE.Vector3(x - half, y - half, z - half), max: new THREE.Vector3(x + half, y + half, z + half) });
  }
  for (const p of PILLARS) {
    addBox({
      i: -1,
      min: new THREE.Vector3(p.x - p.width / 2, p.base, p.z - p.depth / 2),
      max: new THREE.Vector3(p.x + p.width / 2, p.base + TALL, p.z + p.depth / 2),
    });
  }
  return grid;
}

const normal = new THREE.Vector3();
const tangent = new THREE.Vector3();

function overlap(p, box) {
  const cx = Math.max(box.min.x, Math.min(p.x, box.max.x));
  const cy = Math.max(box.min.y, Math.min(p.y, box.max.y));
  const cz = Math.max(box.min.z, Math.min(p.z, box.max.z));
  const dx = p.x - cx, dy = p.y - cy, dz = p.z - cz;
  const d2 = dx * dx + dy * dy + dz * dz;
  if (d2 >= RADIUS * RADIUS) return 0;
  if (d2 > 1e-8) {
    const d = Math.sqrt(d2);
    normal.set(dx / d, dy / d, dz / d);
    return RADIUS - d;
  }
  const ox = Math.min(p.x - box.min.x, box.max.x - p.x);
  const oy = Math.min(p.y - box.min.y, box.max.y - p.y);
  const oz = Math.min(p.z - box.min.z, box.max.z - p.z);
  if (ox <= oy && ox <= oz) { normal.set(p.x < (box.min.x + box.max.x) / 2 ? -1 : 1, 0, 0); return ox + RADIUS; }
  if (oy <= oz) { normal.set(0, p.y < (box.min.y + box.max.y) / 2 ? -1 : 1, 0); return oy + RADIUS; }
  normal.set(0, 0, p.z < (box.min.z + box.max.z) / 2 ? -1 : 1);
  return oz + RADIUS;
}

function respond(b, depth) {
  b.pos.addScaledVector(normal, depth);
  const into = b.vel.dot(normal);
  if (into >= 0) return;
  tangent.copy(b.vel).addScaledVector(normal, -into);
  b.vel.copy(normal).multiplyScalar(-into * BOUNCE).addScaledVector(tangent, SKID);
  b.spin.multiplyScalar(0.65);
}

function groundHit(b, bounce) {
  if (b.pos.y >= RADIUS || !onGround(b.pos.x, b.pos.z)) return false;
  if (bounce) { normal.set(0, 1, 0); respond(b, RADIUS - b.pos.y); }
  else b.pos.y = RADIUS;
  return true;
}

function cityHit(b, boxes, gone) {
  let landed = false;
  for (let cx = -1; cx <= 1; cx++) {
    for (let cz = -1; cz <= 1; cz++) {
      const cell = boxes.get(cellKey(b.pos.x + cx * CELL, b.pos.z + cz * CELL));
      if (!cell) continue;
      for (const box of cell) {
        if (box.i >= 0 && gone.has(box.i)) continue;
        const depth = overlap(b.pos, box);
        if (depth > 0) { respond(b, depth); if (normal.y > 0.5) landed = true; }
      }
    }
  }
  return landed;
}

const turnQ = new THREE.Quaternion();
const axisV = new THREE.Vector3();

export function step(loose, dt, dropped) {
  const boxes = field();
  const gone = new Set();
  for (const b of loose) gone.add(b.i);
  if (dropped) for (const i of dropped) gone.add(i);
  for (const b of loose) {
    if (b.held) { groundHit(b, false); continue; }
    if (b.asleep) continue;
    const rate = b.spin.length();
    if (rate > 1e-4) {
      axisV.copy(b.spin).divideScalar(rate);
      turnQ.setFromAxisAngle(axisV, rate * dt);
      b.quat.premultiply(turnQ);
      b.spin.multiplyScalar(Math.max(0, 1 - SPIN_DAMP * dt));
    }
    const reach = b.vel.length() * dt;
    const steps = Math.min(8, Math.max(1, Math.ceil(reach / (RADIUS * 0.6))));
    const h = dt / steps;
    let landed = false;
    for (let k = 0; k < steps; k++) {
      b.vel.y += GRAVITY * h;
      b.pos.addScaledVector(b.vel, h);
      const floor = groundHit(b, true);
      const wall = cityHit(b, boxes, gone);
      landed = landed || floor || wall;
    }
    if (landed && b.vel.lengthSq() < ASLEEP * ASLEEP) {
      b.vel.set(0, 0, 0);
      b.spin.set(0, 0, 0);
      b.asleep = true;
    }
  }
}

/* ================= sweep (plinth shader) ================= */
const AT = [0, 0.24, 0.46, 0.68, 0.88, 1];
const NAMES = ["--sweep-0", "--sweep-1", "--sweep-2", "--sweep-3", "--sweep-4", "--sweep-0"];
const S_ANGLE = (104 * Math.PI) / 180;
const SWEEP_AXIS = new THREE.Vector3(1, 0, -1)
  .multiplyScalar(Math.sin(S_ANGLE) / (2 * ACROSS))
  .addScaledVector(new THREE.Vector3(0, 1, 0), Math.cos(S_ANGLE) / RISE)
  .normalize();
const SWEEP_SPAN = Math.abs(SWEEP_AXIS.x) + Math.abs(SWEEP_AXIS.y) + Math.abs(SWEEP_AXIS.z);
export const PLINTH_ATTRIBUTE = "aPlinth";

export function sweepStops() {
  const probe = document.createElement("span");
  probe.style.display = "none";
  document.body.appendChild(probe);
  try {
    return NAMES.map((name) => {
      probe.style.color = `var(${name})`;
      return new THREE.Color().setStyle(getComputedStyle(probe).color);
    });
  } finally {
    probe.remove();
  }
}

export function paintSweep(material) {
  const stops = { value: AT.map(() => new THREE.Color(1, 1, 1)) };
  const at = { value: AT.slice() };
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uSweepStops = stops;
    shader.uniforms.uSweepAt = at;
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", `#include <common>
        attribute float ${PLINTH_ATTRIBUTE};
        varying float vPlinth;
        varying float vSweep;`)
      .replace("#include <begin_vertex>", `#include <begin_vertex>
        vPlinth = ${PLINTH_ATTRIBUTE};
        vSweep = clamp(
          dot(position, vec3(${SWEEP_AXIS.x}, ${SWEEP_AXIS.y}, ${SWEEP_AXIS.z}))
            / ${SWEEP_SPAN} + 0.5,
          0.0, 1.0
        );`);
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", `#include <common>
        uniform vec3 uSweepStops[${AT.length}];
        uniform float uSweepAt[${AT.length}];
        varying float vPlinth;
        varying float vSweep;
        vec3 sweepAt(float t) {
          vec3 c = uSweepStops[0];
          for (int i = 1; i < ${AT.length}; i++) {
            c = mix(c, uSweepStops[i], clamp(
              (t - uSweepAt[i - 1]) / max(uSweepAt[i] - uSweepAt[i - 1], 1e-5),
              0.0, 1.0
            ));
          }
          return c;
        }`)
      .replace("#include <color_fragment>", `#include <color_fragment>
        diffuseColor.rgb = mix(diffuseColor.rgb, sweepAt(vSweep), vPlinth);`);
  };
  return { stops };
}

/* ================= Stage ================= */
const INTRO = 3.2;
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const easeOutCubic = (u) => 1 - (1 - u) ** 3;
const easeOutBack = (u) => { const c = 1.34; return 1 + (c + 1) * (u - 1) ** 3 + c * (u - 1) ** 2; };

const MARGIN = 1.08;
const ZOOM = 50;
const CROWN = 40;
const BEARING = 62;
const PLATE_RISE = (px) => ((PILLARS[0].width + PILLARS[0].depth) / 2) * DEPTH * px + 10;
const ETCH = "rgb(var(--ground-rgb) / 0.42)";
const ETCH_INSET = 0.95;
const SWELL = 1.075;
const LIFT = 1.03;
const SANS = "'Manrope', sans-serif";
const OUTLINE = { webkitTextStrokeWidth: "3px", webkitTextStrokeColor: "rgb(var(--ground-rgb))", paintOrder: "stroke fill" };
// Rest-pose face matrices (island does not turn): sidePlate(+Z) and topPlate.
const SIDE_MATRIX = "matrix(0.7071, 0.4082, 0, 0.8165, 0, 0)";
const TOP_MATRIX = "matrix(0.7071, -0.4082, 0.7071, 0.4082, 0, 0)";

const scratch = new THREE.Object3D();
const spotV = new THREE.Vector3();
const tumble = new THREE.Euler();

function sample(s, t) {
  const u = clamp01((t - s.start) / (s.end - s.start));
  if (u <= 0) return 0;
  const e = easeOutCubic(u);
  spotV.set(
    s.from[0] + (s.rest[0] - s.from[0]) * e,
    s.from[1] + (s.rest[1] - s.from[1]) * e,
    s.from[2] + (s.rest[2] - s.from[2]) * e,
  );
  const left = 1 - e;
  tumble.set(s.spin[0] * left, s.spin[1] * left, s.spin[2] * left);
  return easeOutCubic(clamp01(u / 0.5));
}

function fly(mesh, specs, t) {
  if (!mesh) return;
  for (let i = 0; i < specs.length; i++) {
    const scale = sample(specs[i], t);
    if (scale <= 0) { scratch.position.set(0, 0, 0); scratch.rotation.set(0, 0, 0); scratch.scale.setScalar(0); }
    else { scratch.position.copy(spotV); scratch.rotation.copy(tumble); scratch.scale.setScalar(scale); }
    scratch.updateMatrix();
    mesh.setMatrixAt(i, scratch.matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
}

function hide(mesh, gone) {
  if (!mesh || gone.size === 0) return;
  scratch.position.set(0, 0, 0); scratch.rotation.set(0, 0, 0); scratch.scale.setScalar(0);
  scratch.updateMatrix();
  for (const i of gone) mesh.setMatrixAt(i, scratch.matrix);
  mesh.instanceMatrix.needsUpdate = true;
}

function paint(mesh, list) {
  if (!mesh) return;
  for (const b of list) {
    scratch.position.copy(b.pos);
    scratch.quaternion.copy(b.quat);
    scratch.scale.setScalar(1);
    scratch.updateMatrix();
    mesh.setMatrixAt(b.i, scratch.matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
}

function label(parent, inner) {
  const outer = document.createElement("div");
  outer.style.cssText = "position:absolute;left:0;top:0;pointer-events:none;will-change:transform;";
  const center = document.createElement("div");
  center.style.cssText = "position:absolute;translate:-50% -50%;";
  center.appendChild(inner);
  outer.appendChild(center);
  parent.appendChild(outer);
  return outer;
}

export class Podium {
  constructor(host, { rows, foot = 76, tone, loadMark }) {
    this.host = host;
    this.rows = rows;
    this.foot = foot;
    this.tone = tone;
    this.reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.compare = null;
    this.best = rows.reduce((n, r) => Math.max(n, r.elo), 1);
    const scale = heightScale(this.best, undefined);
    this.heights = PILLARS.map((p) => { const row = rows.find((r) => r.rank === p.rank); return row ? scale(row.elo) : 0; });
    this.rival = 0;
    this.intro = this.reduced ? 1 : 0;
    this.settled = false;
    this.drawn = -1;
    this.loose = [];
    this.dropped = new Set();
    this.dragging = null;
    this.dragPlane = new THREE.Plane();
    this.dragLast = new THREE.Vector3();
    this.dragTime = 0;
    this.over = PILLARS.map(() => false);
    this.swell = PILLARS.map(() => 0);
    this.risen = PILLARS.map(() => 0);
    this.grown = 0;
    this.live = true;
    this.disposed = false;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.toneMapping = THREE.NoToneMapping; // r3f `flat`
    renderer.domElement.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none;";
    host.appendChild(renderer.domElement);
    this.renderer = renderer;

    this.overlay = document.createElement("div");
    this.overlay.setAttribute("aria-hidden", "true");
    this.overlay.style.cssText = "position:absolute;inset:0;overflow:hidden;pointer-events:none;z-index:7;";
    host.appendChild(this.overlay);

    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 200);
    this.camera.position.set(18, 18, 18);
    this.camera.zoom = ZOOM;
    this.camera.lookAt(0, 0, 0);

    const scene = new THREE.Scene();
    this.scene = scene;
    scene.add(new THREE.AmbientLight(0xffffff, 0.4));
    const key = new THREE.DirectionalLight(0xffffff, 0.72);
    key.position.set(9, 16, 7);
    scene.add(key);
    const back = new THREE.DirectionalLight(0xffffff, 0.2);
    back.position.set(-11, 5, -9);
    scene.add(back);

    const ink = getComputedStyle(document.body).getPropertyValue("--ink-rgb").trim() || "237 237 237";
    const inkCss = `rgb(${ink})`;

    this.root = new THREE.Group();
    scene.add(this.root);
    this.frame = new THREE.Group();
    this.root.add(this.frame);
    this.model = new THREE.Group();
    this.model.position.set(0, GROUP_Y, 0);
    this.frame.add(this.model);

    const plateGeo = new THREE.BoxGeometry(PLATE, PLATE_H, PLATE);
    const cubeGeo = new THREE.BoxGeometry(CUBE, CUBE, CUBE);
    const postGeo = new THREE.BoxGeometry(1, 1, 1);
    postGeo.translate(0, 0.5, 0);
    this.geos = [plateGeo, cubeGeo, postGeo];

    const skin = new THREE.MeshStandardMaterial({ color: inkCss, roughness: 0.74, metalness: 0 });
    const stoneMat = new THREE.MeshStandardMaterial({ color: inkCss, roughness: 0.74, metalness: 0 });
    this.stone = { material: stoneMat, ...paintSweep(stoneMat) };
    const read = sweepStops();
    for (let i = 0; i < this.stone.stops.value.length; i++) this.stone.stops.value[i].copy(read[i]);
    this.mats = [skin, stoneMat];

    const flags = new Float32Array(BLOCKS.length);
    for (const f of FOOTINGS) flags.fill(1, f.from, f.from + f.count);
    cubeGeo.setAttribute(PLINTH_ATTRIBUTE, new THREE.InstancedBufferAttribute(flags, 1));

    this.plates = new THREE.InstancedMesh(plateGeo, skin, DECK.length);
    this.plates.frustumCulled = false;
    this.model.add(this.plates);
    this.blocks = new THREE.InstancedMesh(cubeGeo, stoneMat, BLOCKS.length);
    this.blocks.frustumCulled = false;
    let far = 0;
    for (const b of BLOCKS) far = Math.max(far, Math.hypot(...b.rest), Math.hypot(...b.from));
    this.blocks.boundingSphere = new THREE.Sphere(new THREE.Vector3(), far + CUBE);
    this.model.add(this.blocks);

    this.pillars = PILLARS.map((p) => {
      const row = rows.find((r) => r.rank === p.rank);
      const mat = new THREE.MeshStandardMaterial({ color: row ? tone(row.lab) : inkCss, roughness: 0.68, metalness: 0 });
      this.mats.push(mat);
      const mesh = new THREE.Mesh(postGeo, mat);
      mesh.position.set(p.x, p.base, p.z);
      mesh.visible = false;
      this.model.add(mesh);
      return mesh;
    });

    this.rivalTone = new THREE.MeshStandardMaterial({ color: inkCss, roughness: 0.68, metalness: 0 });
    this.mats.push(this.rivalTone);
    this.challenger = new THREE.Mesh(postGeo, this.rivalTone);
    this.challenger.position.set(CHALLENGER.x, CHALLENGER.base, CHALLENGER.z);
    this.challenger.visible = false;
    this.model.add(this.challenger);

    // Label rigs: real anchors parented under the model so projection follows transforms.
    this.rigs = PILLARS.map((p) => {
      const rig = new THREE.Group();
      rig.position.set(p.x, p.base, p.z);
      this.model.add(rig);
      const etchAnchor = new THREE.Object3D();
      etchAnchor.position.set(0, -ETCH_INSET, p.depth / 2); // front (+Z) face
      rig.add(etchAnchor);
      const topAnchor = new THREE.Object3D();
      rig.add(topAnchor);
      return { rig, etchAnchor, topAnchor };
    });
    this.markAnchor = new THREE.Object3D();
    this.markAnchor.position.set(CHALLENGER.x, CHALLENGER.base, CHALLENGER.z);
    this.model.add(this.markAnchor);

    // Label DOM.
    this.labels = PILLARS.map((p) => {
      const row = rows.find((r) => r.rank === p.rank);
      const etchIn = document.createElement("div");
      etchIn.textContent = String(row.rank).padStart(2, "0");
      Object.assign(etchIn.style, {
        opacity: "0", transform: SIDE_MATRIX, lineHeight: "1", letterSpacing: "-0.03em",
        color: ETCH, fontFamily: SANS, fontWeight: "800", fontVariantNumeric: "tabular-nums",
      });
      const etch = label(this.overlay, etchIn);

      const markIn = document.createElement("div");
      Object.assign(markIn.style, { opacity: "0", transform: TOP_MATRIX, filter: "brightness(0)" });
      const markHold = document.createElement("span");
      markHold.style.cssText = "display:flex;";
      markIn.appendChild(markHold);
      const mark = label(this.overlay, markIn);
      loadMark(row.lab).then((svg) => {
        if (this.disposed) return;
        if (svg) markHold.innerHTML = svg;
        else {
          markHold.textContent = row.lab.slice(0, 2).toUpperCase();
          Object.assign(markHold.style, { fontFamily: SANS, fontWeight: "800", letterSpacing: "0.06em" });
        }
        this.sizeMark(markHold);
      });

      const plateIn = document.createElement("div");
      plateIn.textContent = row.name;
      Object.assign(plateIn.style, {
        opacity: "0", fontFamily: SANS, fontSize: "15px", lineHeight: "1", fontWeight: "800",
        letterSpacing: "-0.01em", whiteSpace: "nowrap", color: "rgb(var(--ink-rgb))",
        WebkitTextStrokeWidth: OUTLINE.webkitTextStrokeWidth, WebkitTextStrokeColor: OUTLINE.webkitTextStrokeColor, paintOrder: OUTLINE.paintOrder,
      });
      const plate = label(this.overlay, plateIn);
      return { etch, etchIn, mark, markIn, markHold, plate, plateIn };
    });

    const rname = document.createElement("div");
    Object.assign(rname.style, {
      opacity: "0", fontFamily: SANS, fontSize: "15px", lineHeight: "1", fontWeight: "800",
      letterSpacing: "-0.01em", whiteSpace: "nowrap", color: "rgb(var(--ink-rgb))",
      WebkitTextStrokeWidth: OUTLINE.webkitTextStrokeWidth, WebkitTextStrokeColor: OUTLINE.webkitTextStrokeColor, paintOrder: OUTLINE.paintOrder,
    });
    const rrank = document.createElement("div");
    Object.assign(rrank.style, {
      opacity: "0", fontFamily: SANS, fontSize: "10px", lineHeight: "1", letterSpacing: "0.2em",
      whiteSpace: "nowrap", textTransform: "uppercase", color: "rgb(var(--ink-rgb) / 0.64)", fontWeight: "800",
      WebkitTextStrokeWidth: OUTLINE.webkitTextStrokeWidth, WebkitTextStrokeColor: OUTLINE.webkitTextStrokeColor, paintOrder: OUTLINE.paintOrder,
    });
    this.rivalName = label(this.overlay, rname);
    this.rivalRank = label(this.overlay, rrank);
    this.rivalNameIn = rname;
    this.rivalRankIn = rrank;

    this.caster = new THREE.Raycaster();
    this.ndc = new THREE.Vector2();
    this.markPx = 0;

    this.onMove = (ev) => this.pointerMove(ev);
    this.onDown = (ev) => this.pointerDown(ev);
    this.onUp = () => this.release();
    renderer.domElement.addEventListener("pointermove", this.onMove);
    renderer.domElement.addEventListener("pointerdown", this.onDown);
    renderer.domElement.addEventListener("pointerup", this.onUp);
    renderer.domElement.addEventListener("pointercancel", this.onUp);
    renderer.domElement.addEventListener("pointerleave", () => { if (!this.dragging) this.over.fill(false); });

    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(host);
    // Pause only when the host is genuinely scrolled out of view. At mount the
    // column can measure 0x0 (layout not settled) and that report must not park
    // the loop — a zero-size host costs nothing to keep rendering.
    this.io = new IntersectionObserver((entries) => {
      const e = entries[entries.length - 1];
      const zero = e.boundingClientRect.width === 0 && e.boundingClientRect.height === 0;
      this.live = e.isIntersecting || zero;
    }, { threshold: 0 });
    this.io.observe(host);

    this.resize();
    window.__podium = this; // debug handle
    this.clock = new THREE.Clock();
    this.lastFrame = performance.now();
    const loop = () => {
      if (this.disposed) return;
      this.raf = requestAnimationFrame(loop);
      const delta = this.clock.getDelta();
      this.lastFrame = performance.now();
      if (!this.live) return;
      this.tick(delta);
    };
    loop();
    // rAF is throttled hard in background/offscreen contexts; keep the city
    // settling (and thrown blocks falling) at a low rate rather than freezing.
    this.pulse = setInterval(() => {
      if (this.disposed || !this.live) return;
      if (performance.now() - this.lastFrame < 200) return;
      this.clock.getDelta();
      this.tick(1 / 30);
    }, 120);
  }

  sizeMark(hold) {
    const px = Math.round(1.05 * this.markPx || 0);
    if (!px) return;
    const svg = hold.querySelector("svg");
    if (svg) { svg.setAttribute("width", px); svg.setAttribute("height", px); svg.style.display = "block"; }
    else hold.style.fontSize = Math.max(9, px * 0.34) + "px";
  }

  resize() {
    const w = this.host.clientWidth || 1;
    const h = this.host.clientHeight || 1;
    this.w = w; this.h = h;
    this.renderer.setSize(w, h, false);
    const cam = this.camera;
    cam.left = -w / 2; cam.right = w / 2; cam.top = h / 2; cam.bottom = -h / 2;
    cam.updateProjectionMatrix();
  }

  setCompare(row) {
    this.compare = row;
    const scale = heightScale(this.best, row ? row.elo : undefined);
    this.heights = PILLARS.map((p) => {
      const r = this.rows.find((x) => x.rank === p.rank);
      return r ? scale(r.elo) : 0;
    });
    this.rival = row ? scale(row.elo) : 0;
    if (row) {
      this.rivalTone.color.set(this.tone(row.lab));
      this.rivalNameIn.textContent = row.name;
      this.rivalRankIn.textContent = `Rank ${String(row.rank).padStart(2, "0")}`;
    }
  }

  project(obj, el) {
    obj.getWorldPosition(spotV);
    spotV.project(this.camera);
    const x = (spotV.x * 0.5 + 0.5) * this.w;
    const y = (-spotV.y * 0.5 + 0.5) * this.h;
    el.style.transform = `translate(${x.toFixed(2)}px, ${y.toFixed(2)}px)`;
  }

  pointerMove(ev) {
    const el = this.renderer.domElement;
    const rect = el.getBoundingClientRect();
    this.ndc.set(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);

    const b = this.dragging;
    if (b) {
      this.caster.setFromCamera(this.ndc, this.camera);
      const at = new THREE.Vector3();
      if (!this.caster.ray.intersectPlane(this.dragPlane, at)) return;
      this.model.worldToLocal(at);
      const gap = Math.max(4, ev.timeStamp - this.dragTime) / 1000;
      this.dragTime = ev.timeStamp;
      const flick = new THREE.Vector3().subVectors(at, this.dragLast).divideScalar(gap);
      this.dragLast.copy(at);
      b.vel.lerp(flick, 0.65);
      b.pos.copy(at);
      b.asleep = false;
      return;
    }

    this.caster.setFromCamera(this.ndc, this.camera);
    const pillarHits = this.caster.intersectObjects([...this.pillars], false);
    const overIdx = pillarHits.length ? this.pillars.indexOf(pillarHits[0].object) : -1;
    for (let i = 0; i < this.over.length; i++) this.over[i] = i === overIdx;
    const blockHit = overIdx < 0 && this.settled && this.caster.intersectObject(this.blocks, false).length > 0;
    el.style.cursor = blockHit ? "grab" : "";
  }

  pointerDown(ev) {
    if (ev.pointerType === "touch") return;
    if (!this.settled) return;
    const el = this.renderer.domElement;
    const rect = el.getBoundingClientRect();
    this.ndc.set(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
    this.caster.setFromCamera(this.ndc, this.camera);
    const pillarHits = this.caster.intersectObjects([...this.pillars, this.challenger], false);
    const blockHits = this.caster.intersectObject(this.blocks, false);
    if (!blockHits.length) return;
    if (pillarHits.length && pillarHits[0].distance < blockHits[0].distance) return;
    const hit = blockHits[0];
    const id = hit.instanceId;
    if (id == null) return;

    let b = this.loose.find((l) => l.i === id);
    if (!b) {
      b = {
        i: id,
        pos: new THREE.Vector3(...BLOCKS[id].rest),
        vel: new THREE.Vector3(),
        quat: new THREE.Quaternion(),
        spin: new THREE.Vector3(),
        held: true,
        asleep: false,
      };
      this.loose.push(b);
    }
    b.held = true;
    b.asleep = false;
    b.vel.set(0, 0, 0);
    this.camera.getWorldDirection(spotV);
    this.dragPlane.setFromNormalAndCoplanarPoint(spotV, hit.point);
    this.dragging = b;
    this.dragLast.copy(b.pos);
    this.dragTime = ev.timeStamp;
    el.style.cursor = "grabbing";
    el.setPointerCapture(ev.pointerId);
  }

  release() {
    const b = this.dragging;
    this.dragging = null;
    this.renderer.domElement.style.cursor = "";
    if (!b) return;
    b.held = false;
    b.vel.clampLength(0, MAX_THROW);
    const speed = b.vel.length();
    b.spin.set(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).multiplyScalar(speed * 0.5 + 1);
  }

  tick(delta) {
    const dt = Math.min(delta, 1 / 30);
    if (this.intro < 1) this.intro = Math.min(1, this.intro + dt / INTRO);
    const t = this.intro;
    this.settled = t > 0.999;

    const crown = CROWN / ZOOM;
    const sole = this.foot / ZOOM;
    const vw = this.w / ZOOM;
    const vh = this.h / ZOOM;
    const solved = Math.max(1e-4, Math.min(
      (vw - (2 * BEARING) / ZOOM) / (VIEW.w * MARGIN),
      (vh - crown - sole) / (VIEW.h * MARGIN),
    ));
    const lift = (sole - crown) / 2;
    const floor = -vh / 2 - lift;
    this.root.position.y = lift;
    this.frame.scale.setScalar(solved);
    const px = solved * ZOOM;
    if (Math.abs(px - this.markPx) > 0.5) {
      this.markPx = px;
      for (const l of this.labels) {
        l.etchIn.style.fontSize = `${0.82 * px}px`;
        this.sizeMark(l.markHold);
      }
    }

    const named = clamp01((t - LABELS[0]) / (LABELS[1] - LABELS[0]));

    if (Math.abs(t - this.drawn) >= 0.0002) {
      this.drawn = t;
      fly(this.plates, DECK, t);
      fly(this.blocks, BLOCKS, t);
      hide(this.blocks, this.dropped);
    }

    for (let i = 0; i < PILLARS.length; i++) {
      const mesh = this.pillars[i];
      const p = PILLARS[i];
      const u = clamp01((t - p.start) / (p.end - p.start));
      const L = this.labels[i];
      if (u <= 0) { mesh.visible = false; L.etchIn.style.opacity = "0"; L.markIn.style.opacity = "0"; L.plateIn.style.opacity = "0"; continue; }
      mesh.visible = true;
      this.swell[i] = THREE.MathUtils.damp(this.swell[i], this.over[i] ? 1 : 0, 9, dt);
      const s = this.swell[i];
      const wide = 1 + (SWELL - 1) * s;
      const tall = 1 + (LIFT - 1) * s;
      const rise = Math.max(1e-3, easeOutBack(u) * tall);
      this.risen[i] = THREE.MathUtils.damp(this.risen[i] || this.heights[i], this.heights[i], 9, dt);
      mesh.scale.set(p.width * wide, this.risen[i] * rise, p.depth * wide);

      const { rig, etchAnchor, topAnchor } = this.rigs[i];
      rig.position.y = p.base + this.risen[i] * rise;
      rig.scale.set(wide, rise, wide);

      L.etchIn.style.opacity = String(named);
      L.etchIn.style.scale = String(wide);
      L.markIn.style.opacity = String(named);
      L.markIn.style.scale = String(wide);
      L.plateIn.style.opacity = String(named);
      L.plateIn.style.translate = `0 calc(-50% - ${PLATE_RISE(px)}px)`;
      this.project(etchAnchor, L.etch);
      this.project(topAnchor, L.mark);
      this.project(topAnchor, L.plate);
    }

    this.grown = THREE.MathUtils.damp(this.grown, this.rival, 9, dt);
    const post = this.challenger;
    const up = this.grown;
    post.visible = up > 0.02 && t > LABELS[0];
    post.scale.set(CHALLENGER.width, Math.max(1e-3, up), CHALLENGER.depth);
    this.markAnchor.position.y = CHALLENGER.base + up;
    const shown = this.rival > 0 ? clamp01((this.grown / this.rival) * 3 - 2) : 0;
    this.rivalNameIn.style.opacity = String(shown);
    this.rivalRankIn.style.opacity = String(shown);
    this.rivalNameIn.style.translate = `0 calc(-50% - ${PLATE_RISE(px)}px)`;
    this.rivalRankIn.style.translate = `0 calc(-50% - ${PLATE_RISE(px) - 17}px)`;
    this.project(this.markAnchor, this.rivalName);
    this.project(this.markAnchor, this.rivalRank);

    const list = this.loose;
    if (list.length) {
      let lost = false;
      for (const b of list) {
        const below = (b.pos.y * RISE - (b.pos.x + b.pos.z) * DEPTH - SPAN0.mid) * solved;
        if (below < floor - CUBE * solved) {
          if (this.dragging === b) this.dragging = null;
          this.dropped.add(b.i);
          lost = true;
        }
      }
      if (lost) {
        this.loose = list.filter((b) => !this.dropped.has(b.i));
        hide(this.blocks, this.dropped);
      }
      step(this.loose, dt, this.dropped);
      paint(this.blocks, this.loose);
    }

    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    clearInterval(this.pulse);
    this.ro.disconnect();
    this.io.disconnect();
    const el = this.renderer.domElement;
    el.removeEventListener("pointermove", this.onMove);
    el.removeEventListener("pointerdown", this.onDown);
    el.removeEventListener("pointerup", this.onUp);
    el.removeEventListener("pointercancel", this.onUp);
    for (const g of this.geos) g.dispose();
    for (const m of this.mats) m.dispose();
    this.renderer.dispose();
    el.remove();
    this.overlay.remove();
  }
}
