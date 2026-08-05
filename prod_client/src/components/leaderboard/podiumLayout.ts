/**
 * THE CITY THE PODIUM STANDS IN, and when each piece of it arrives.
 *
 * PURE DATA, no React and no three. Every position, every fly-in path and every
 * moment in the timeline is decided here, and the scene component's only job is to
 * write these numbers onto instances each frame. That split is what makes the
 * choreography legible: the whole build order is one file you can read top to
 * bottom, rather than something you reconstruct by chasing refs through a render.
 *
 * ONE SCALAR DRIVES ALL OF IT. Every piece carries a `start` and `end` in the same
 * 0..1 space, so "assembled" is a single number the page can hand the scene —
 * ticked forward by a clock on first paint and pulled back down by scroll. There is
 * no second timeline to keep in step, and running it backwards is free, which is
 * exactly what scrolling up has to do.
 *
 * DETERMINISTIC RANDOMNESS. The city is generated rather than hand-placed, but it
 * is generated from a SEED: this module is evaluated once on the server while the
 * client component is prerendered and again in the browser, and two different
 * cities would be a hydration mismatch. It also means the skyline is the same every
 * visit, which is the difference between a designed composition and a lava lamp.
 */

/** Cube edge, in world units. Everything else is expressed in terms of it. */
export const CUBE = 1;

/** The seam left between neighbours — what makes a run of blocks read as MASONRY
 *  rather than as one slab. Small enough that a wall still reads as solid. */
const SEAM = 0.07;
const STRIDE = CUBE + SEAM;

/** The ground's paving: two cubes square, and flatter, so the city floor reads as
 *  ground rather than as another course of blocks. */
export const PLATE = 2 * STRIDE;
export const PLATE_H = 0.9;

type Vec3 = [number, number, number];

export type BlockSpec = {
	/** Where it ends up. */
	rest: Vec3;
	/** Where it comes in from — off the board and above it. */
	from: Vec3;
	/** Tumble it carries on the way in, unwound to zero as it lands. */
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
	/** World Y of the pillar's foot — the top of its own plinth. */
	base: number;
	start: number;
	end: number;
};

// --- the projection -------------------------------------------------------
//
// THE THREE CONSTANTS THE WHOLE COMPOSITION IS PLANNED IN. Under a true isometric
// camera a world offset (x, y, z) lands at screen (x − z)·cos45 across, and
// (x + z)·sin45·sin35.264 down, and y·cos35.264 up. They are the projection — not
// tuning — and having them here is what lets this file reason about what the
// picture will look like rather than only about where things are in the world.
const ACROSS = 0.7071;
const DEPTH = 0.4082;
const RISE = 0.8165;

/**
 * THE TWO AXES THE COMPOSITION IS ACTUALLY BUILT ON, and neither of them is X.
 *
 * The SCREEN's horizontal is the world direction (1, 0, −1): move along it and a
 * thing slides sideways without changing its height on screen OR its distance from
 * the camera. The screen's vertical is a combination of world Y — height — and
 * (1, 0, 1), which is DEPTH: something further from the camera sits HIGHER in the
 * picture.
 *
 * That second fact is the whole reason this file was rewritten. The composition
 * was a shallow band across the middle of the stage with black above it, and the
 * way to fill that space in an isometric projection is not taller buildings, it is
 * a DEEPER ground: every row you add behind the last climbs the screen by
 * PLATE·sin45·sin35 whether anything is standing on it or not.
 *
 * `p` runs along the platform, `s` steps toward the viewer (positive, lower on
 * screen) or away from them (negative, higher up and further off).
 */
function spot(p: number, s = 0): [number, number] {
	return [p + s / 2, -p + s / 2];
}

/** Where a thing at depth `s` and height `y` lands on the screen's vertical, in
 *  world units above the plaza floor. The city is planned against this, so that a
 *  building four rows back and one three rows back can be given heights that read
 *  as a coherent skyline rather than as a random comb. */
function skyline(s: number, y: number): number {
	return y * RISE - s * DEPTH;
}

// A small deterministic PRNG. Thirty-odd lines of a library, one line here.
function mulberry32(seed: number) {
	return () => {
		seed = (seed + 0x6d2b79f5) | 0;
		let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

const rand = mulberry32(0x5cebe4);

/** How long any one piece spends in flight, in timeline units. */
const FLIGHT = 0.16;

/**
 * One piece's spec.
 *
 * IT COMES IN FROM ITS OWN DIRECTION — outward from the composition's centre and
 * well above it — so the swarm converges rather than raining straight down. A
 * single shared entry vector reads as a curtain; radial entry reads as assembly.
 *
 * `rank` is 0..1 within the piece's group and decides WHERE IN THE GROUP'S WINDOW
 * it flies, so each part lays itself down in a deliberate order rather than all at
 * once.
 */
function piece(rest: Vec3, rank: number, [from, to]: [number, number]): BlockSpec {
	const start = from + rank * Math.max(0, to - from - FLIGHT);
	const [x, , z] = rest;
	// Straight up the middle has no outward direction to take, so it gets one.
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

// --- the ground -----------------------------------------------------------
//
// A DEEP FIELD, CUT OFF AT THE FRONT. Generated directly in the projected axes:
// `d` counts along the screen's horizontal and `w` back into the picture. Two rows
// in front of the plaza, so the platform has a near edge to end on, and eleven
// behind it, which is what carries the city up the stage.
//
// The parity guard is what keeps the tiles on the world grid — `d` and `w` are the
// sum and difference of two integer indices, so they have to share a parity for
// the cell they name to exist.
// FAR ENOUGH TO CARRY THE OUTERMOST ROW OF BUILDINGS AND NO FURTHER, at both ends.
// Every plate past the back is empty white field where the horizon should be, and
// the front now has to carry real ground too — the city runs on BOTH sides of the
// plaza, so there are buildings nearer the viewer than the podium is.
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
	// Laid from the plaza outward, so the ground unrolls toward the edges of the
	// screen and away into the distance rather than arriving as a rectangle.
	cells.sort((a, b) => a.away - b.away);
	const last = Math.max(1, cells.length - 1);
	return cells.map((c, k) =>
		piece([c.x, -PLATE_H / 2, c.z], k / last, [0.0, 0.34]),
	);
}

/** The ground's top face — what everything else stands on. */
const GROUND = 0;

/**
 * IS THERE FLOOR UNDER THIS POINT? The ground is a band, not an infinite plane, so
 * a block thrown past its edge has nothing to land on and falls into the dark —
 * which is the whole reason the edge is worth knowing about (see `loose`).
 *
 * Tested in the same projected axes the ground was generated in, so it is the
 * actual footprint rather than a bounding box drawn around it.
 */
export function onGround(x: number, z: number): boolean {
	const d = (x - z) / PLATE;
	const w = (x + z) / PLATE;
	return Math.abs(d) <= RUN + 0.5 && w >= -BACK - 0.5 && w <= FRONT + 0.5;
}

// --- building blocks ------------------------------------------------------

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
	// Bottom course first: a stack that assembles top-down looks like it is being
	// hung rather than built.
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

/**
 * ONE BUILDING, WITH SETBACKS. Every few storeys the footprint loses a cell, which
 * is the one move that makes a stack of cubes read as ARCHITECTURE rather than as
 * a column — it is what a voxel skyscraper is, and it gives the silhouette the
 * steps that catch the light differently on each face.
 *
 * The setback is taken off alternating sides so the tower leans a little rather
 * than tapering symmetrically, which would read as a spire.
 */
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

// --- the city -------------------------------------------------------------
//
// FOUR ROWS RECEDING, one in front, laid on a grid of lots. Two rules shape it and
// they are both about the PICTURE rather than about the world:
//
// A FLAT SKYLINE CAP. A building four rows back starts higher up the screen than
// one in the front row, so giving every row the same height budget in world units
// would build a staircase climbing out of the top of the frame. Each row's budget
// is worked out from `skyline` instead, against one screen-space ceiling — so the
// city reads as an even skyline and the far rows come out low, which is also how
// distance looks.
//
// A PLAZA. Lots too close to the podium are skipped, so the three pillars stand in
// a clearing rather than being crowded by their own city, and the eye has
// somewhere to rest at the middle of the picture.
// LANDMARKS. Plain setback towers make a skyline; they do not make a city you
// want to look at. These are the shapes that only a stack of cubes can be — a
// gateway with a hole through it, a mast under an oversailing cap, a span between
// two legs, a hollow court — and each one is chosen to be legible from an
// isometric angle, where a silhouette is all you get.

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

/** A thin shaft under a cap that oversails it on every side — the one thing a
 *  stack of blocks can do that a solid tower cannot. */
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

/** A hollow ring: four walls and a yard, which reads as a courtyard block from
 *  above and as a notched mass from the side. */
const COURT: Cell[] = [
	...slab(-1, 1, -1, 1, 0).filter(([x, , z]) => x !== 0 || z !== 0),
	...slab(-1, 1, -1, 1, 1).filter(([x, , z]) => x !== 0 || z !== 0),
	[-1, 2, -1],
	[1, 2, 1],
];

/** Courses that alternate direction as they climb, so the tower turns as it goes
 *  up — the corners step out on every second storey. */
const TURN: Cell[] = [0, 1, 2, 3, 4, 5].flatMap((y) =>
	y % 2 === 0 ? slab(-1, 0, 0, 0, y) : slab(0, 0, -1, 0, y),
);

const LANDMARKS = [ZIGGURAT, GATE, MAST, BRIDGE, COURT, TURN];

/** Lot pitch along the run, and how many lots either side of centre.
 *
 *  WIDE, AND THAT IS THE POINT. At the pitch this replaces the buildings stood
 *  shoulder to shoulder and the city read as one crenellated mass — clutter rather
 *  than density. Density is not how much of the picture is white, it is how many
 *  distinct THINGS you can count in it, and that needs air between them. Fewer,
 *  bigger, further apart. */
const LOT = 3.5;
/** Lots either side of centre.
 *
 *  SPENT WHERE THEY ARE SEEN. At seven, the outer four columns each side landed
 *  past ±24 screen units — the frame is about ±16 — so two fifths of the city was
 *  generated, animated and never once visible. Five leaves one column of bleed at
 *  each edge, which is the part that reads as the city continuing past the frame. */
const LOTS = 5;

/** How far a lot may wander off its row, so the rows do not read as rows. */
const DRIFT = 1.3;

const ROWS = [
	// `clear` is how much of the plaza this row keeps out of; `max` is its own
	// ceiling in storeys.
	//
	// THE FIRST TWO ROWS ARE IN FRONT OF THE PODIUM — nearer the viewer, lower in
	// the picture, and drawn over the plaza floor. They are what stops the city
	// being a backdrop the podium is pasted onto: with buildings on both sides of
	// it, the three pillars sit IN the city rather than in front of a picture of
	// one. Held low, because a tall building this near the camera is a wall.
	// CLEAR OF NOTHING, the near one. It sits well below the pillars' feet in the
	// picture, so a building dead in front of the winner reads as foreground, not as
	// something in the way.
	{ s: 8.2, clear: 0, max: 2, span: [0.2, 0.4] },
	{ s: 4.6, clear: 2.6, max: 3, span: [0.23, 0.43] },
	{ s: -3.2, clear: 7.0, max: 7, span: [0.27, 0.47] },
	{ s: -6.6, clear: 4.5, max: 7, span: [0.31, 0.51] },
	{ s: -10.0, clear: 0, max: 7, span: [0.35, 0.55] },
	{ s: -13.4, clear: 0, max: 6, span: [0.39, 0.59] },
	{ s: -16.8, clear: 0, max: 6, span: [0.43, 0.63] },
] as const;

/**
 * THE SKYLINE'S CEILING, in screen units, for a row this far back.
 *
 * IT CLIMBS WITH DISTANCE, and slightly SLOWER than the ground does. That single
 * relationship is the whole profile of the city: the ground rises 0.408 screen
 * units per unit of depth, so a ceiling climbing at 0.34 leaves each row a little
 * less height budget than the row in front of it — buildings get shorter as they
 * recede, exactly as distance looks, while their ROOFS still climb the picture and
 * fill it.
 *
 * The flat cap this replaces was the reason the back of the city was empty: held
 * at one screen height while the ground climbed underneath it, the budget ran out
 * four rows back and every lot past that got a single cube.
 */
function ceiling(s: number): number {
	return 6.9 + 0.34 * -s;
}

/**
 * WOULD A BUILDING HERE JUST BE HIDDEN BEHIND A PILLAR? The podium is three broad
 * white towers standing in the foreground, and anything close behind one of them
 * at the same screen position is work nobody will ever see.
 *
 * Only the near rows are tested. Far enough back the ground has climbed high
 * enough that a building's roof clears the pillar tops on its own, which is what
 * makes the deep city worth building in the first place.
 */
function shadowed(p: number, s: number): boolean {
	// BEHIND ONLY. Testing the front rows as well is what emptied the middle of the
	// foreground: it banished every near building out past the pillars' screen
	// bands, which is off the sides of the frame, and left the one part of the
	// picture the eye actually rests on bare.
	//
	// It was also wrong. A pillar is eleven units tall and a front-row building is
	// two, and the near row sits a long way DOWN the picture — so a building in
	// front of a pillar lands below its foot and hides nothing. Occlusion in an
	// isometric view is not about sharing a screen column, it is about sharing one
	// at the same height, and nothing in the front rows ever gets that high.
	if (s > 0 || -s > 11) return false;
	const x = 2 * p * ACROSS;
	return [0, GAP, -GAP].some(
		(c) => Math.abs(x - 2 * c * ACROSS) < PILLAR_W * ACROSS + 1.3,
	);
}

function city(): BlockSpec[] {
	const out: BlockSpec[] = [];
	for (const row of ROWS) {
		// TWO CEILINGS, WHICHEVER BITES FIRST. The row's own limit is a composition
		// choice; the screen ceiling is arithmetic — a building this far back already
		// starts partway up the picture, so what is left of the budget is what it can
		// spend on height.
		const budget = ceiling(row.s) - skyline(row.s, 0);
		const tallest = Math.max(
			2,
			Math.min(row.max, Math.floor(budget / (STRIDE * RISE))),
		);
		const span: [number, number] = [row.span[0], row.span[1]];
		for (let n = -LOTS; n <= LOTS; n++) {
			const p = n * LOT + (rand() - 0.5) * 1.6;
			// OFF THE ROW, TOO. Jittering depth as well as position is what turns a
			// grid into a plan: with only the run jittered the city still resolved
			// into ranks the moment you looked at it, because everything in a row
			// shares a screen height and the eye finds that instantly.
			const s = row.s + (rand() - 0.5) * 2 * DRIFT;
			if (Math.abs(p) < row.clear) continue;
			if (shadowed(p, s)) continue;
			// Gaps in the grid: a city with a building on every lot is a wall.
			if (rand() < 0.2) continue;
			// A QUARTER OF THE BACKGROUND IS LANDMARKS. Not the front row, which has
			// to stay low, and not so many that they stop being landmarks.
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
			// Only mildly skewed, so a row is mostly mid-rise with real towers in it —
			// the heavy skew this replaces made every lot a one-storey box.
			const tall = 1 + Math.round(rand() ** 1.05 * (tallest - 1));
			// BIGGER FOOTPRINTS to go with the wider pitch. Single-cube buildings at
			// close spacing are what made the city read as gravel; at this pitch they
			// would just read as gravel with gaps in it.
			const r = rand();
			const w = r < 0.2 ? 1 : r < 0.65 ? 2 : 3;
			const d = rand() < 0.35 ? 1 : rand() < 0.8 ? 2 : 3;
			out.push(...building(p, s, tall, w, d, span));
		}
	}
	return out;
}

// --- the podium -----------------------------------------------------------

/** How far apart the pillars stand along the run.
 *
 *  MEASURED ON SCREEN, not in the world. Two points `g` apart along `p` land
 *  `2g·cos45` = 1.41·g apart horizontally in the projection, and a pillar is
 *  1.41·width wide on screen — so at this spacing there is a clear band of black
 *  between neighbouring faces instead of one silhouette running into the next. */
const GAP = 4.8;
const PILLAR_W = 2.3;

/** Each pillar stands on its own three-by-three footing, which is the other half
 *  of telling them apart: three separate plinths read as three objects in a
 *  square, where one shared step reads as a single block with notches in it. */
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

/**
 * WHICH INSTANCES BELONG TO WHICH PODIUM PLACE.
 *
 * The footings are made of the site's accent gradient rather than of plain white,
 * and they live in the same instanced mesh as the nine hundred city blocks — so
 * the shader needs to know which instances they are before it can paint them.
 * Derived from the arrays that built it rather than written down, because a
 * hand-counted offset into an instance buffer is a bug waiting for someone to add
 * a district.
 */
export const FOOTINGS: { rank: number; from: number; count: number }[] = FOOT.map(
	(f, i) => ({
		rank: f.rank,
		from:
			CITY.length +
			FOOT.slice(0, i).reduce((n, g) => n + g.cells.length, 0),
		count: f.cells.length,
	}),
);

/**
 * THE THREE, in podium order: the winner centre and tallest, second to its left,
 * third to its right. Listed in the order they RISE, which is the reverse — third
 * first, so the sequence climbs.
 *
 * TALL ENOUGH TO BE LANDMARKS. In a city of blocks the podium has to be the thing
 * the skyline is measured against, or it is just three more buildings; every one
 * of these clears the city's cap, and the winner clears it by half again.
 */
export const PILLARS: PillarSpec[] = [
	{ rank: 3, p: GAP, height: 6.6, start: 0.66, end: 0.81 },
	{ rank: 2, p: -GAP, height: 8.6, start: 0.73, end: 0.88 },
	{ rank: 1, p: 0, height: 11.0, start: 0.8, end: 0.96 },
].map(({ p, ...rest }) => {
	const [x, z] = spot(p);
	return { ...rest, x, z, width: PILLAR_W, depth: PILLAR_W, base: FOOT_TOP };
});

/** When the names on the pillars fade up — after the last one has landed. */
export const LABELS: [number, number] = [0.9, 1.0];

// --- framing --------------------------------------------------------------

/** What must always fit ACROSS: the podium and its footings, and no more. The city
 *  is meant to run off both edges, so it is deliberately not in this. */
const CORE_HALF_W = GAP * 2 * ACROSS + PILLAR_W * ACROSS + 1.6;

/**
 * WHAT MUST FIT UP AND DOWN, MEASURED OFF THE COMPOSITION ITSELF rather than
 * asserted.
 *
 * The top used to be hard-coded as "the winner's pillar plus room for its name",
 * which was true only while the city was capped below it. Now that the skyline
 * climbs into the distance, the tallest thing in the picture is a roof somewhere at
 * the back — and which roof that is depends on a seeded generator. So it is read
 * out of the generated blocks, and the framing can never disagree with what was
 * built.
 */
const roof = (b: BlockSpec) => skyline(b.rest[0] + b.rest[2], b.rest[1] + CUBE / 2);

const CORE_UP = Math.max(
	skyline(0, FOOT_TOP + PILLARS[2].height) + 1.7,
	...BLOCKS.map(roof),
);

/** The near edge of the ground, which is the lowest thing on screen. */
const CORE_DOWN = skyline(FRONT * PLATE, 0) * -1 + PLATE_H * RISE;

export const VIEW = {
	w: CORE_HALF_W * 2,
	h: CORE_UP + CORE_DOWN,
};

/**
 * WHERE THE MODEL SITS RELATIVE TO THE CAMERA, which is a single number because
 * the camera never moves: it looks at the origin down a true isometric axis, so
 * framing the composition means sliding the composition, not aiming anything.
 */
export const GROUP_Y = -(CORE_UP - CORE_DOWN) / 2 / RISE;
