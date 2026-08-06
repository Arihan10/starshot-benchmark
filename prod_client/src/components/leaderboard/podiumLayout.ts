
export const CUBE = 1;

const SEAM = 0.07;
const STRIDE = CUBE + SEAM;

export const PLATE = 2 * STRIDE;
export const PLATE_H = 0.9;

type Vec3 = [number, number, number];

export type BlockSpec = {
	rest: Vec3;
	from: Vec3;
	spin: Vec3;
	start: number;
	end: number;
};

// THE THREE NUMBERS THE WHOLE ISLAND IS DRAWN IN — how far a unit of model space
// carries across the screen, down it, and up it. Exported together now: anything
// that has to place a flat label on a face of this thing needs the same
// projection, and a second copy of 0.7071 somewhere else is a copy that will
// eventually disagree with this one.
export const ACROSS = 0.7071;
export const DEPTH = 0.4082;
export const RISE = 0.8165;

function spot(p: number, s = 0): [number, number] {
	return [p + s / 2, -p + s / 2];
}

function skyline(s: number, y: number): number {
	return y * RISE - s * DEPTH;
}

function mulberry32(seed: number) {
	return () => {
		seed = (seed + 0x6d2b79f5) | 0;
		let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

const SEED = 0x5cebe4;

// A STREAM PER SITE, not one stream for the whole city — and this is the piece
// that makes a resizable city possible at all.
//
// Drawing every block from one sequential PRNG means the layout is a function of
// HOW MANY blocks were asked for: widen the city by one lot and every draw after
// it shifts by however many numbers that lot consumed, so the whole skyline
// re-scatters. On a page that rebuilds the city when the window grows, that is a
// city that visibly reshuffles itself while you drag the corner of the browser.
//
// Seeded on the site's own coordinates instead, a lot's building is a pure
// function of WHERE it is. Extension becomes purely additive: everything already
// standing is untouched and new lots simply appear beyond it. The same holds for
// the deck, keyed on its own cell.
//
// The first draw is discarded. Adjacent seeds are adjacent integers, and
// mulberry32's first output has not been through enough mixing to be independent
// across them — undiscarded, neighbouring lots came out visibly alike.
function site(kind: number, a: number, b: number) {
	let h = SEED ^ Math.imul(kind + 1, 0x27d4eb2d);
	h = Math.imul(h ^ (a + 8192), 0x9e3779b1);
	h = Math.imul(h ^ (b + 8192), 0x85ebca6b);
	const rand = mulberry32(h >>> 0);
	rand();
	return rand;
}

const FLIGHT = 0.16;

function piece(
	rest: Vec3,
	rank: number,
	[from, to]: [number, number],
	rand: () => number,
): BlockSpec {
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

// ---------------------------------------------------------------------------
// THE ISLAND
//
// The city used to be a STRIP: a deck that ran off both edges of the page and
// was built as wide as the window happened to be. It is an OBJECT now — a
// bounded landmass floating in the column with a city on its back — and that
// changes what every measurement in this file is for.
//
// A strip is fitted on height and extended sideways to cover whatever it must.
// An island is fitted whole, both axes, like any other model: it has a size, it
// has edges you are meant to see, and the machinery that used to grow it to
// order is gone. What survives is the per-site PRNG above, which now earns its
// keep differently — the rim is irregular, and an irregular rim has to be the
// same irregular rim every time it is drawn.
// ---------------------------------------------------------------------------

// THE PLAN, in plate units: how far the land reaches across (`d`) and how deep
// it runs (`w`). Wider than it is deep, because the projection already halves
// depth — `d` carries ACROSS at 0.7071 and `w` carries DEPTH at 0.4082, so equal
// numbers would draw an island nearly twice as wide as tall. These two put it at
// a little over 2:1 on screen, which reads as land rather than as a plate.
const REACH_D = 13;
const REACH_W = 9;

// HOW RAGGED THE COAST IS. A perfect ellipse reads as a token — a game piece,
// not a place. The rim is pushed in and out by up to this many plates, sampled
// off the direction it sits in so the coast wanders slowly rather than
// alternating cell by cell.
const COAST = 1.35;

// The rim at a given depth, in plates across. Zero once past either end.
function shore(w: number): number {
	const u = w / REACH_W;
	if (Math.abs(u) >= 1) return 0;
	const t = Math.atan2(w * 1.6, REACH_W) * 2.2;
	const wander =
		(Math.sin(t * 3.1 + 0.7) * 0.6 + Math.sin(t * 7.3 + 2.2) * 0.4) * COAST;
	return REACH_D * Math.sqrt(1 - u * u) + wander;
}

// HOW FAR THE ISLAND GOES DOWN, and how fast it closes. Layers of plate beneath
// the surface, each drawn in from the one above — a keel, so the thing reads as
// a piece of land torn out of somewhere rather than as a tabletop. The taper is
// deliberately not linear: land undercuts sharply just below the waterline and
// then tails off to a point.
const KEEL = 8;

function draught(k: number): number {
	return (1 - k / (KEEL + 1)) ** 0.62;
}

/**
 * Is this point over the island's surface?
 *
 * Used by the physics for blocks a visitor has thrown: past the coast there is
 * no longer any ground to land on, and they fall past the island and out of the
 * scene — which is the whole reason an island is more fun to throw things off
 * than a deck that ran to the horizon.
 */
export function onGround(x: number, z: number): boolean {
	const d = (x - z) / PLATE;
	const w = (x + z) / PLATE;
	return Math.abs(d) <= shore(w) + 0.5;
}

function ground(): BlockSpec[] {
	const cells: { x: number; y: number; z: number; away: number }[] = [];

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
					// The surface lands first and from the middle out; the keel
					// follows, deepest last, so the island assembles downward.
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

type Cell = [number, number, number];

function slab(x0: number, x1: number, z0: number, z1: number, y: number): Cell[] {
	const out: Cell[] = [];
	for (let x = x0; x <= x1; x++) for (let z = z0; z <= z1; z++) out.push([x, y, z]);
	return out;
}

function place(
	cells: Cell[],
	p: number,
	s: number,
	span: [number, number],
	rand: () => number,
): BlockSpec[] {
	const [ox, oz] = spot(p, s);
	const order = [...cells].sort((a, b) => a[1] - b[1]);
	const last = Math.max(1, order.length - 1);
	return order.map(([dx, dy, dz], k) =>
		piece(
			[ox + dx * STRIDE, GROUND + CUBE / 2 + dy * STRIDE, oz + dz * STRIDE],
			k / last,
			span,
			rand,
		),
	);
}

function building(
	p: number,
	s: number,
	storeys: number,
	w: number,
	d: number,
	span: [number, number],
	rand: () => number,
): BlockSpec[] {
	const cells: Cell[] = [];
	let x0 = -((w - 1) >> 1);
	let x1 = x0 + w - 1;
	let z0 = -((d - 1) >> 1);
	let z1 = z0 + d - 1;
	for (let y = 0; y < storeys; y++) {
		if (y > 0 && y % 3 === 0) {
			if (x1 - x0 > 0 && rand() < 0.7) {
				if (rand() < 0.5) x0 += 1;
				else x1 -= 1;
			} else if (z1 - z0 > 0) {
				if (rand() < 0.5) z0 += 1;
				else z1 -= 1;
			}
		}
		cells.push(...slab(x0, x1, z0, z1, y));
	}
	return place(cells, p, s, span, rand);
}

const ZIGGURAT: Cell[] = [
	...slab(-1, 1, -1, 1, 0),
	...slab(-1, 1, -1, 1, 1),
	...slab(-1, 0, -1, 0, 2),
	[0, 3, 0],
];

const GATE: Cell[] = [
	[-1, 0, 0],
	[-1, 1, 0],
	[-1, 2, 0],
	[1, 0, 0],
	[1, 1, 0],
	[1, 2, 0],
	...slab(-1, 1, 0, 0, 3),
	[0, 4, 0],
];

const MAST: Cell[] = [
	[0, 0, 0],
	[0, 1, 0],
	[0, 2, 0],
	[0, 3, 0],
	[0, 4, 0],
	...slab(-1, 1, -1, 1, 5),
	[0, 6, 0],
];

const BRIDGE: Cell[] = [
	[-2, 0, 0],
	[-2, 1, 0],
	[-2, 2, 0],
	[2, 0, 0],
	[2, 1, 0],
	[2, 2, 0],
	...slab(-2, 2, 0, 0, 3),
];

const COURT: Cell[] = [
	...slab(-1, 1, -1, 1, 0).filter(([x, , z]) => x !== 0 || z !== 0),
	...slab(-1, 1, -1, 1, 1).filter(([x, , z]) => x !== 0 || z !== 0),
	[-1, 2, -1],
	[1, 2, 1],
];

const TURN: Cell[] = [0, 1, 2, 3, 4, 5].flatMap((y) =>
	y % 2 === 0 ? slab(-1, 0, 0, 0, y) : slab(0, 0, -1, 0, y),
);

const LANDMARKS = [ZIGGURAT, GATE, MAST, BRIDGE, COURT, TURN];

// The tallest storey any landmark reaches. Landmarks are placed WHOLE and so
// ignore the row's own storey cap, which is why the headroom below has to know
// this number rather than trusting `ceiling`.
const LANDMARK_TOP = LANDMARKS.reduce(
	(top, cells) => cells.reduce((n, [, y]) => Math.max(n, y), top),
	0,
);

const LOT = 2.25;

const DRIFT = 0.9;

// THE STREETS RUN THE DEPTH OF THE ISLAND. Each row is given whatever width the
// coast leaves it at that depth, so the plan is island-shaped without any row
// having to know the island's outline — the front and back streets come out
// short, the middle ones run the full breadth, and nothing is ever built out
// over the water.
//
// `max` grades upward toward the back: low at the front so nothing stands
// between the reader and the podium, tall behind it so the city gives the
// pillars something to be tall against.
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
] as const;

function ceiling(s: number): number {
	return 9.6 + 0.34 * -s;
}

function storeys(row: (typeof ROWS)[number]): number {
	const budget = ceiling(row.s) - skyline(row.s, 0);
	return Math.max(2, Math.min(row.max, Math.floor(budget / (STRIDE * RISE))));
}

// ---------------------------------------------------------------------------
// THE PODIUM AND THE CHALLENGER
//
// FOUR BERTHS, NOT THREE. The fourth stands empty until a row of the standings
// is pointed at, and then it carries that model — so the plan has to reserve it
// whether or not anything is standing in it. Left to the city, a challenger
// would rise straight through somebody's rooftops.
// ---------------------------------------------------------------------------

const GAP = 5.6;
const PILLAR_W = 3.1;

// Beyond third place, on the same line and at the same spacing, so the four read
// as one row of posts and the comparison is a matter of looking along it.
export const CHALLENGER_P = 2 * GAP;

const BERTHS = [-GAP, 0, GAP, CHALLENGER_P];

function shadowed(p: number, s: number): boolean {
	if (s > 0 || -s > 11) return false;
	const x = 2 * p * ACROSS;
	return BERTHS.some(
		(c) => Math.abs(x - 2 * c * ACROSS) < PILLAR_W * ACROSS + 1.3,
	);
}

function city(): BlockSpec[] {
	const out: BlockSpec[] = [];
	ROWS.forEach((row, r) => {
		const tallest = storeys(row);
		const span: [number, number] = [row.span[0], row.span[1]];
		// The breadth the coast leaves this street, converted from plates across
		// into the `p` the lots are counted in, and held back from the very edge
		// so no frontage overhangs the water.
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
				out.push(
					...place(LANDMARKS[Math.floor(rand() * LANDMARKS.length)], p, s, span, rand),
				);
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

function footing(p: number, span: [number, number]): BlockSpec[] {
	return place(slab(-1, 1, -1, 1, 0), p, 0, span, site(2, Math.round(p * 10), 0));
}

const FOOT_TOP = GROUND + CUBE;

export const DECK: BlockSpec[] = ground();

const CITY = city();
const FOOT: { rank: number; cells: BlockSpec[] }[] = [
	{ rank: 2, cells: footing(-GAP, [0.58, 0.68]) },
	{ rank: 1, cells: footing(0, [0.58, 0.68]) },
	{ rank: 3, cells: footing(GAP, [0.58, 0.68]) },
	// THE FOURTH PLINTH IS ALWAYS BUILT, and stands empty until a row of the
	// standings is pointed at. It is what makes the challenger read as arriving
	// somewhere rather than materialising in the middle of a street: the berth is
	// visibly waiting, so raising a post into it is an answer to a question the
	// island was already asking. Rank 0 — it belongs to no one until it does.
	{ rank: 0, cells: footing(CHALLENGER_P, [0.58, 0.68]) },
];

export const BLOCKS: BlockSpec[] = [...CITY, ...FOOT.flatMap((f) => f.cells)];

export const FOOTINGS: { rank: number; from: number; count: number }[] = FOOT.map(
	(f, i) => ({
		rank: f.rank,
		from: CITY.length + FOOT.slice(0, i).reduce((n, g) => n + g.cells.length, 0),
		count: f.cells.length,
	}),
);

// ---------------------------------------------------------------------------
// ONE HEIGHT SCALE, FOR THE PODIUM AND FOR THE CHALLENGER ALIKE
//
// This is the part that makes the hover mean anything. The three podium pillars
// used to be authored heights — 31 / 21 / 13, chosen because they looked like a
// podium — and a fourth pillar raised beside them from a model's rating would
// have been measured against nothing at all. Two models could not be compared by
// looking, which is the only thing the picture is for.
//
// So every pillar on the island, the three included, is now the same function of
// the same number. The three come out at roughly 31 / 22 / 17, which is within a
// couple of units of the authored figures — the podium looks the way it always
// did — and rank four comes out at 9, visibly short of third, on a scale that
// says so honestly rather than by decoration.
//
// WHY A POWER CURVE AND NOT A STRAIGHT LINE. Elo at the top is close: first and
// second here are 25 points apart in 463, and a linear scale draws them within a
// few percent of each other — a podium of three equal posts, which tells a
// reader nothing. The exponent stretches the top of the range apart and packs
// the bottom, so the differences that decide the podium are the ones you can
// see. The floor sits below the lowest rating on the board so nothing lands at
// zero: the tail of the standings raises a short tower in the city rather than
// no tower at all.
const FLOOR = 1000;
const STUB = 2.4;

export const TALL = 31;

const CURVE = 7;

export function pillarHeight(elo: number, best: number): number {
	const u = Math.max(0, Math.min(1, (elo - FLOOR) / Math.max(1, best - FLOOR)));
	return STUB + (TALL - STUB) * u ** CURVE;
}

export type PillarSpec = {
	rank: number;
	x: number;
	z: number;
	width: number;
	depth: number;
	base: number;
	start: number;
	end: number;
};

export const PILLARS: PillarSpec[] = [
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

export const LABELS: [number, number] = [0.9, 1.0];

// HOW BIG THE WHOLE THING IS — measured, now that there is a whole thing to
// measure. While the city was a strip built to order this had to be derived from
// what the generator was ALLOWED to produce, because the measurement would
// otherwise creep with the width and chase its own tail. An island has one fixed
// extent, so the honest bounding box is available again and is used.
//
// THE PILLARS ARE COUNTED AT THEIR CEILING, not at the heights the current
// standings give them. A hovered challenger can be any height up to `TALL`, and
// a box measured on today's data would let tomorrow's tallest post grow straight
// out of the top of the column.
// EVERY BOX, KEPT — because the island turns now, and a bounding box measured at
// one yaw does not survive being spun. Each entry is a centre and its half-extents.
const BOXES: [number, number, number, number, number, number][] = [];

for (const d of DECK) {
	BOXES.push([
		d.rest[0],
		d.rest[1],
		d.rest[2],
		PLATE / 2,
		PLATE_H / 2,
		PLATE / 2,
	]);
}
for (const b of BLOCKS) {
	BOXES.push([b.rest[0], b.rest[1], b.rest[2], CUBE / 2, CUBE / 2, CUBE / 2]);
}
for (const p of [...PILLARS, CHALLENGER]) {
	BOXES.push([p.x, p.base + TALL / 2, p.z, p.width / 2, TALL / 2, p.depth / 2]);
}

export type Span = { w: number; h: number; mid: number };

// THE ISLAND'S SCREEN FOOTPRINT AT A GIVEN YAW.
//
// Each box is rotated about the model's vertical axis, then projected — and the
// rotated box is bounded by its own AABB rather than by its eight corners, which
// is what lets this stay a loop over centres instead of a loop over vertices. The
// bound is loose only for boxes turned off-axis, and every one of these is a cube
// or near-cube where the difference is a fraction of a unit against a 42-unit
// island.
function spanAt(yaw: number): Span {
	const c = Math.cos(yaw);
	const s = Math.sin(yaw);
	let minX = Number.POSITIVE_INFINITY;
	let maxX = Number.NEGATIVE_INFINITY;
	let minY = Number.POSITIVE_INFINITY;
	let maxY = Number.NEGATIVE_INFINITY;
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

// SAMPLED ONCE AND READ FOREVER. `spanAt` is 1,755 boxes; at sixty frames a second
// that is a hundred thousand boxes a second to answer a question whose answer
// changes smoothly and is already known. A degree of yaw moves the footprint by
// well under a tenth of a percent, so the table is sampled every two degrees and
// read with a lerp between neighbours.
const SPAN_STEP = 2;
const SPANS: Span[] = [];
for (let deg = 0; deg <= 360; deg += SPAN_STEP) {
	SPANS.push(spanAt((deg * Math.PI) / 180));
}

/**
 * The island's screen size and vertical centre at a given yaw, in model units.
 *
 * BOTH NUMBERS MOVE, and the second one is the easy one to miss. Turning the
 * island changes how TALL it draws — by up to 18% — but it also changes where the
 * middle of it is, because the near and far coasts trade places. Fit to the height
 * alone and the island stays inside the column while sliding up and down it.
 */
export function viewAt(yaw: number): Span {
	const deg = ((((yaw * 180) / Math.PI) % 360) + 360) % 360;
	const at = deg / SPAN_STEP;
	const i = Math.floor(at);
	const f = at - i;
	const a = SPANS[i];
	const b = SPANS[i + 1];
	return {
		w: a.w + (b.w - a.w) * f,
		h: a.h + (b.h - a.h) * f,
		mid: a.mid + (b.mid - a.mid) * f,
	};
}

// The rest pose, which is still what the composition is designed around and what
// every fixed pixel reservation was measured against.
export const VIEW = { w: SPANS[0].w, h: SPANS[0].h };

export const GROUP_Y = -SPANS[0].mid / RISE;
