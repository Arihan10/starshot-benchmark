
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

export type PillarSpec = {
	rank: number;
	x: number;
	z: number;
	width: number;
	depth: number;
	height: number;
	base: number;
	start: number;
	end: number;
};

const ACROSS = 0.7071;
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

const rand = mulberry32(0x5cebe4);

const FLIGHT = 0.16;

function piece(rest: Vec3, rank: number, [from, to]: [number, number]): BlockSpec {
	const start = from + rank * Math.max(0, to - from - FLIGHT);
	const [x, , z] = rest;
	const len = Math.hypot(x, z) || 1;
	const ox = (x || rand() - 0.5) / len;
	const oz = (z || rand() - 0.5) / len;
	const reach = 3 + rand() * 4;
	return {
		rest,
		from: [
			rest[0] + ox * reach,
			rest[1] + 4 + rand() * 6,
			rest[2] + oz * reach,
		],
		spin: [(rand() - 0.5) * 1.7, (rand() - 0.5) * 2.4, (rand() - 0.5) * 1.7],
		start,
		end: start + FLIGHT,
	};
}

const FRONT = 5;
const BACK = 9;
const RUN = 18;

function ground(): BlockSpec[] {
	const cells: { x: number; z: number; away: number }[] = [];
	for (let w = -BACK; w <= FRONT; w++) {
		for (let d = -RUN; d <= RUN; d++) {
			if (((w + d) & 1) !== 0) continue;
			const i = (w + d) / 2;
			const j = (w - d) / 2;
			cells.push({
				x: i * PLATE,
				z: j * PLATE,
				away: Math.hypot(d, w * 1.4),
			});
		}
	}
	cells.sort((a, b) => a.away - b.away);
	const last = Math.max(1, cells.length - 1);
	return cells.map((c, k) =>
		piece([c.x, -PLATE_H / 2, c.z], k / last, [0.0, 0.34]),
	);
}

const GROUND = 0;

export function onGround(x: number, z: number): boolean {
	const d = (x - z) / PLATE;
	const w = (x + z) / PLATE;
	return Math.abs(d) <= RUN + 0.5 && w >= -BACK - 0.5 && w <= FRONT + 0.5;
}

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
): BlockSpec[] {
	const [ox, oz] = spot(p, s);
	const order = [...cells].sort((a, b) => a[1] - b[1]);
	const last = Math.max(1, order.length - 1);
	return order.map(([dx, dy, dz], k) =>
		piece(
			[ox + dx * STRIDE, GROUND + CUBE / 2 + dy * STRIDE, oz + dz * STRIDE],
			k / last,
			span,
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
	return place(cells, p, s, span);
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

const LOT = 3.5;
const LOTS = 5;

const DRIFT = 1.3;

const ROWS = [
	{ s: 8.2, clear: 0, max: 2, span: [0.2, 0.4] },
	{ s: 4.6, clear: 2.6, max: 3, span: [0.23, 0.43] },
	{ s: -3.2, clear: 7.0, max: 7, span: [0.27, 0.47] },
	{ s: -6.6, clear: 4.5, max: 7, span: [0.31, 0.51] },
	{ s: -10.0, clear: 0, max: 7, span: [0.35, 0.55] },
	{ s: -13.4, clear: 0, max: 6, span: [0.39, 0.59] },
	{ s: -16.8, clear: 0, max: 6, span: [0.43, 0.63] },
] as const;

function ceiling(s: number): number {
	return 6.9 + 0.34 * -s;
}

function shadowed(p: number, s: number): boolean {
	if (s > 0 || -s > 11) return false;
	const x = 2 * p * ACROSS;
	return [0, GAP, -GAP].some(
		(c) => Math.abs(x - 2 * c * ACROSS) < PILLAR_W * ACROSS + 1.3,
	);
}

function city(): BlockSpec[] {
	const out: BlockSpec[] = [];
	for (const row of ROWS) {
		const budget = ceiling(row.s) - skyline(row.s, 0);
		const tallest = Math.max(
			2,
			Math.min(row.max, Math.floor(budget / (STRIDE * RISE))),
		);
		const span: [number, number] = [row.span[0], row.span[1]];
		for (let n = -LOTS; n <= LOTS; n++) {
			const p = n * LOT + (rand() - 0.5) * 1.6;
			const s = row.s + (rand() - 0.5) * 2 * DRIFT;
			if (Math.abs(p) < row.clear) continue;
			if (shadowed(p, s)) continue;
			if (rand() < 0.2) continue;
			if (row.s < -5 && rand() < 0.28) {
				out.push(
					...place(
						LANDMARKS[Math.floor(rand() * LANDMARKS.length)],
						p,
						s,
						span,
					),
				);
				continue;
			}
			const tall = 1 + Math.round(rand() ** 1.05 * (tallest - 1));
			const r = rand();
			const w = r < 0.2 ? 1 : r < 0.65 ? 2 : 3;
			const d = rand() < 0.35 ? 1 : rand() < 0.8 ? 2 : 3;
			out.push(...building(p, s, tall, w, d, span));
		}
	}
	return out;
}

const GAP = 6.5;
const PILLAR_W = 2.3;

function footing(p: number, span: [number, number]): BlockSpec[] {
	return place(slab(-1, 1, -1, 1, 0), p, 0, span);
}

const FOOT_TOP = GROUND + CUBE;

export const DECK: BlockSpec[] = ground();

const CITY = city();
const FOOT: { rank: number; cells: BlockSpec[] }[] = [
	{ rank: 2, cells: footing(-GAP, [0.58, 0.68]) },
	{ rank: 1, cells: footing(0, [0.58, 0.68]) },
	{ rank: 3, cells: footing(GAP, [0.58, 0.68]) },
];

export const BLOCKS: BlockSpec[] = [...CITY, ...FOOT.flatMap((f) => f.cells)];

export const FOOTINGS: { rank: number; from: number; count: number }[] = FOOT.map(
	(f, i) => ({
		rank: f.rank,
		from:
			CITY.length +
			FOOT.slice(0, i).reduce((n, g) => n + g.cells.length, 0),
		count: f.cells.length,
	}),
);

export const PILLARS: PillarSpec[] = [
	{ rank: 3, p: GAP, height: 7.0, start: 0.66, end: 0.81 },
	{ rank: 2, p: -GAP, height: 10.4, start: 0.73, end: 0.88 },
	{ rank: 1, p: 0, height: 14.4, start: 0.8, end: 0.96 },
].map(({ p, ...rest }) => {
	const [x, z] = spot(p);
	return { ...rest, x, z, width: PILLAR_W, depth: PILLAR_W, base: FOOT_TOP };
});

export const LABELS: [number, number] = [0.9, 1.0];

// Measured off every box actually drawn — deck, city and pillars — projected
// into screen space. The estimate this replaced counted only the pillars for
// width, so the fit ran large and the model sat low in whatever box it was
// given.
const seen = {
	minX: Number.POSITIVE_INFINITY,
	maxX: Number.NEGATIVE_INFINITY,
	minY: Number.POSITIVE_INFINITY,
	maxY: Number.NEGATIVE_INFINITY,
};

function cover(
	cx: number,
	cy: number,
	cz: number,
	hx: number,
	hy: number,
	hz: number,
): void {
	const x = (cx - cz) * ACROSS;
	const y = cy * RISE - (cx + cz) * DEPTH;
	const spreadX = (hx + hz) * ACROSS;
	const spreadY = hy * RISE + (hx + hz) * DEPTH;
	if (x - spreadX < seen.minX) seen.minX = x - spreadX;
	if (x + spreadX > seen.maxX) seen.maxX = x + spreadX;
	if (y - spreadY < seen.minY) seen.minY = y - spreadY;
	if (y + spreadY > seen.maxY) seen.maxY = y + spreadY;
}

for (const d of DECK) {
	cover(d.rest[0], d.rest[1], d.rest[2], PLATE / 2, PLATE_H / 2, PLATE / 2);
}
for (const b of BLOCKS) {
	cover(b.rest[0], b.rest[1], b.rest[2], CUBE / 2, CUBE / 2, CUBE / 2);
}
for (const p of PILLARS) {
	cover(p.x, p.base + p.height / 2, p.z, p.width / 2, p.height / 2, p.depth / 2);
}

export const VIEW = {
	w: seen.maxX - seen.minX,
	h: seen.maxY - seen.minY,
};

export const GROUP_Y = -((seen.maxY + seen.minY) / 2) / RISE;
