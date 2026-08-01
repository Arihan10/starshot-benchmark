import type { PanoEntry } from "./panoTextures";
import type {
	MinimapBasis,
	MinimapBounds,
	MinimapLevel,
	OrbitMode,
	OrbitState,
} from "./types";

export type MinimapSlice = MinimapLevel & { url: string };

// A zone name to print on the map, with the world centre of the zone it names.
export type MapLabel = {
	id: string;
	label: string;
	center: [number, number, number];
};

// --- the map's frame ---------------------------------------------------------
//
// The slice images are drawn by the capture worker from a basis it was handed: an
// axis the camera stood on, and a direction that points down the page. To place a
// capture point on one of those images the viewer has to reconstruct the same
// frame — so this is the twin of `readBasis` in client/public/js/tourcapture.js
// and has to agree with it. Both derive the third direction rather than carrying
// it, which is what keeps them from being able to disagree about handedness.
//
// A slice with no basis is a plan view, which is what every tour captured before
// the map could look any other way actually is.

type Basis = { axis: number; sign: number; right: Axis3; down: Axis3 };
type Axis3 = [number, number, number];

const AXIS_OF: Record<string, number> = { X: 0, Y: 1, Z: 2 };

function axisVec(a: string): Axis3 | null {
	const i = AXIS_OF[a.slice(-1).toUpperCase()];
	if (i === undefined) return null;
	const v: Axis3 = [0, 0, 0];
	v[i] = a.trim().startsWith("-") ? -1 : 1;
	return v;
}

const PLAN_BASIS: Basis = {
	axis: 1,
	sign: 1,
	right: [1, 0, 0],
	down: [0, 0, 1],
};

export function readBasis(b: MinimapBasis | undefined): Basis {
	const from = b ? axisVec(b.view_from) : null;
	const down = b ? axisVec(b.image_down) : null;
	if (!from || !down) return PLAN_BASIS;
	const axis = from[0] !== 0 ? 0 : from[1] !== 0 ? 1 : 2;
	const dAxis = down[0] !== 0 ? 0 : down[1] !== 0 ? 1 : 2;
	if (axis === dAxis) return PLAN_BASIS; // the two must name different axes
	const forward: Axis3 = [-from[0], -from[1], -from[2]];
	const up: Axis3 = [-down[0], -down[1], -down[2]];
	const right: Axis3 = [
		forward[1] * up[2] - forward[2] * up[1],
		forward[2] * up[0] - forward[0] * up[2],
		forward[0] * up[1] - forward[1] * up[0],
	];
	return { axis, sign: from[axis], right, down };
}

/** A signed unit axis, so the dot product is one multiply. */
function along(v: Axis3, p: readonly [number, number, number]): number {
	const i = v[0] !== 0 ? 0 : v[1] !== 0 ? 1 : 2;
	return v[i] * p[i];
}

/** A world point in the map's own frame: across the page, then down it. */
export function toMap(
	basis: Basis,
	p: readonly [number, number, number],
): { u: number; v: number } {
	return { u: along(basis.right, p), v: along(basis.down, p) };
}

/** The image's rectangle, accepting the u/v keys or an older tour's x/z ones. */
function readBounds(b: MinimapBounds): {
	u0: number;
	u1: number;
	v0: number;
	v1: number;
} {
	if (typeof b.minU === "number" && typeof b.minV === "number") {
		return { u0: b.minU, u1: b.maxU ?? b.minU, v0: b.minV, v1: b.maxV ?? b.minV };
	}
	return {
		u0: b.minX ?? 0,
		u1: b.maxX ?? 0,
		v0: b.minZ ?? 0,
		v1: b.maxZ ?? 0,
	};
}

/** Where a slice sits along the flattened axis (`coord`, or an older tour's `y`). */
function sliceCoord(mm: MinimapLevel): number {
	return typeof mm.coord === "number" ? mm.coord : mm.y;
}

// Nearest slice to a world position, measured along the axis the map flattens.
// The slices are separated along that axis, so argmin reproduces the grouping the
// capture used. Only a fallback now — the capture stamps each pano with its level
// — but map labels have no level of their own and still resolve this way.
export function levelForPosition(
	minimaps: MinimapSlice[],
	p: readonly [number, number, number],
): number {
	let best = -1;
	let bestD = Infinity;
	for (let i = 0; i < minimaps.length; i++) {
		const basis = readBasis(minimaps[i].basis);
		const d = Math.abs(sliceCoord(minimaps[i]) - p[basis.axis]);
		if (d < bestD) {
			bestD = d;
			best = i;
		}
	}
	return best;
}

// --- the moving window -------------------------------------------------------
//
// The map does NOT have to show the whole storey. On a house it does and should —
// the plan fits, and seeing all of it at once is the point. On a 200 m level it
// cannot: fitted to the panel, a metre is under half a pixel, capture points a
// couple of metres apart land on top of each other, and the map becomes a strip
// that says nothing about where you are.
//
// So past a certain size the map stops fitting and starts FOLLOWING: it shows a
// window of the storey centred on where you stand, and slides as you move. The
// window is measured in capture-spacings rather than metres, so it holds the same
// number of reachable points whatever the scene is built at — which is the thing
// that actually decides whether a map is readable.
const WINDOW_STEPS = 15;

/** Narrow `[lo, hi]` to `span` around `centre`, without leaving the original. */
function windowAround(
	lo: number,
	hi: number,
	centre: number,
	span: number,
): [number, number] {
	if (!(span > 0) || hi - lo <= span) return [lo, hi];
	let a = centre - span / 2;
	if (a < lo) a = lo;
	if (a + span > hi) a = hi - span;
	return [a, a + span];
}

// The minimap for whatever level the user is on right now — the matching slice
// plus every same-level anchor placed onto it. Only while walking the interior
// (or peeking); the overview already shows the whole dollhouse. Surfaces every
// floor's slice + anchors so the chrome can browse other levels without moving
// the camera; the live capture lights up on its level.
export function buildMinimapState(args: {
	minimaps: MinimapSlice[];
	panos: PanoEntry[];
	panoLevel: number[];
	currentIndex: number;
	mode: OrbitMode;
	labels: MapLabel[];
	/** The scene's measured capture spacing — the window's unit (see scale.ts). */
	step: number;
}): OrbitState["minimap"] {
	const { minimaps, panos, panoLevel, currentIndex, mode, labels, step } = args;
	if (minimaps.length === 0 || currentIndex < 0) return null;
	if (mode !== "interior" && mode !== "peek") return null;
	const currentLevel = panoLevel[currentIndex];
	if (currentLevel < 0) return null;
	const windowSpan = step > 0 ? step * WINDOW_STEPS : Infinity;

	const levels = minimaps.map((mm, idx) => {
		const basis = readBasis(mm.basis);
		const b = readBounds(mm.bounds);
		const spanU = b.u1 - b.u0;
		const spanV = b.v1 - b.v0;
		// Scope the map to the storey. Every slice is rendered over the whole scene
		// footprint, so an upper floor occupying one wing sat marooned in the middle
		// of ground it has nothing to do with. Its described volume is its real
		// extent, so clip the image to that and express everything below — aspect,
		// anchors, labels — in the clipped frame. No volume (an older tour, or a
		// scene whose floors were clustered rather than described) falls back to the
		// full slice, unchanged.
		let u0 = b.u0;
		let u1 = b.u1;
		let v0 = b.v0;
		let v1 = b.v1;
		const vol = mm.volume;
		if (vol && spanU > 0 && spanV > 0) {
			const lo = vol.origin;
			const hi: [number, number, number] = [
				lo[0] + vol.dimensions[0],
				lo[1] + vol.dimensions[1],
				lo[2] + vol.dimensions[2],
			];
			// A box's extent in the map frame is its two corners projected and
			// ordered — the projection may flip an axis, so neither corner is
			// reliably the smaller one.
			const a = toMap(basis, lo);
			const c = toMap(basis, hi);
			const cu0 = Math.max(b.u0, Math.min(a.u, c.u));
			const cu1 = Math.min(b.u1, Math.max(a.u, c.u));
			const cv0 = Math.max(b.v0, Math.min(a.v, c.v));
			const cv1 = Math.min(b.v1, Math.max(a.v, c.v));
			if (cu1 - cu0 > 0 && cv1 - cv0 > 0) {
				u0 = cu0;
				u1 = cu1;
				v0 = cv0;
				v1 = cv1;
			}
		}

		// Then follow. The current storey centres on the capture you are standing
		// at; the others centre on their own captures, so paging between floors on
		// the rail doesn't change the zoom under you.
		const members: number[] = [];
		for (let i = 0; i < panos.length; i++) if (panoLevel[i] === idx) members.push(i);
		let focus: { u: number; v: number } | null = null;
		if (idx === currentLevel) {
			focus = toMap(basis, panos[currentIndex].position);
		} else if (members.length > 0) {
			let su = 0;
			let sv = 0;
			for (const i of members) {
				const m = toMap(basis, panos[i].position);
				su += m.u;
				sv += m.v;
			}
			focus = { u: su / members.length, v: sv / members.length };
		}
		if (focus) {
			[u0, u1] = windowAround(u0, u1, focus.u, windowSpan);
			[v0, v1] = windowAround(v0, v1, focus.v, windowSpan);
		}

		const w = u1 - u0;
		const d = v1 - v0;
		const crop = {
			u0: spanU > 0 ? (u0 - b.u0) / spanU : 0,
			v0: spanV > 0 ? (v0 - b.v0) / spanV : 0,
			u1: spanU > 0 ? (u1 - b.u0) / spanU : 1,
			v1: spanV > 0 ? (v1 - b.v0) / spanV : 1,
		};
		// Percentages of the WINDOW, deliberately unclamped: a point outside it is
		// dropped rather than pinned to an edge. Clamping was harmless while the map
		// always showed the whole storey; with a window it would pile every capture
		// you have walked past into a heap along the border.
		const pct = (n: number) => n * 100;
		const inside = (l: number, t: number) =>
			l >= -2 && l <= 102 && t >= -2 && t <= 102;

		const points: {
			index: number;
			id: string;
			name?: string;
			leftPct: number;
			topPct: number;
			current: boolean;
		}[] = [];
		for (const i of members) {
			const m = toMap(basis, panos[i].position);
			// Against the WINDOW's origin, not the slice's — the anchors, the labels
			// and the image all have to be measured in the same frame or the
			// you-are-here dot drifts off the room it is standing in.
			const leftPct = w > 0 ? pct((m.u - u0) / w) : 50;
			const topPct = d > 0 ? pct((m.v - v0) / d) : 50;
			if (!inside(leftPct, topPct)) continue;
			points.push({
				index: i,
				id: panos[i].id,
				name: panos[i].name,
				leftPct,
				topPct,
				current: i === currentIndex,
			});
		}

		// Zone names land on whichever storey their centre is nearest, placed by the
		// same world→image mapping as the anchors. Each carries the closest capture
		// to that centre, so clicking the name travels there.
		const placed: Array<{
			id: string;
			label: string;
			leftPct: number;
			topPct: number;
			index: number;
		}> = [];
		for (const lab of labels) {
			if (levelForPosition(minimaps, lab.center) !== idx) continue;
			const m = toMap(basis, lab.center);
			const leftPct = w > 0 ? pct((m.u - u0) / w) : 50;
			const topPct = d > 0 ? pct((m.v - v0) / d) : 50;
			if (!inside(leftPct, topPct)) continue;
			let index = -1;
			let best = Infinity;
			for (const i of members) {
				const q = toMap(basis, panos[i].position);
				const dd = (q.u - m.u) ** 2 + (q.v - m.v) ** 2;
				if (dd < best) {
					best = dd;
					index = i;
				}
			}
			if (index < 0) continue;
			placed.push({ id: lab.id, label: lab.label, leftPct, topPct, index });
		}

		return {
			level: idx,
			// The planner's name for this storey, so the floor control can say
			// "Living & Pool Terrace" rather than "2" (see anchors.py). Null on tours
			// captured before floors were named.
			name: mm.name ?? null,
			url: mm.url,
			aspect: d > 0 ? w / d : 1,
			crop,
			points,
			labels: placed,
		};
	});
	return { currentLevel, levels };
}
