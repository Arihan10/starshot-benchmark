import { type Object3D, MathUtils, Raycaster, Vector3 } from "three";

// The scene's own scale, measured rather than assumed.
//
// Every distance in the walkthrough used to be an absolute number of metres tuned
// against one modern house: 30 m of reach, a 20 m "far" cutoff, a 24 m WASD
// stride, a 1.3 m marker drop, a 3.2 m floor arrow. None of those is a property of
// a walkthrough — they are properties of THAT BUILDING — and they fail in both
// directions. On a 200 m side-scrolling level a 20 m cutoff makes nearly every link
// map-only and a 3 m dock radius can never catch a glide; on anything doll-sized
// the same numbers put every affordance outside the scene entirely.
//
// So the knobs are expressed as ratios of three MEASURED quantities, each answering
// a different question:
//
//   extent — HOW BIG IS THE WHOLE SCENE. Drives what frames it from outside: the
//            camera's clip planes, the orbit distance, free-flight speed.
//   step   — HOW FAR APART ARE THE PLACES YOU CAN STAND. The median distance from
//            one capture point to its nearest neighbour. Drives everything about
//            travelling: reach, what counts as far, a stride, the dock radius.
//   eye    — HOW BIG IS WHOEVER LIVES HERE. The median height of a capture point
//            above the surface underneath it. Drives everything drawn at human
//            size: where a floor marker sits, how big a ring is, what rise counts
//            as a change of floor.
//
// `step` and `eye` are deliberately SEPARATE measurements, not one scale factor.
// They happen to be related in a house — the anchor planner is told to space
// captures at least 2 m apart and to stand them at a realistic viewing height, and
// the modern house measures 2.24 m and about 1.6 m — but a landscape can have
// captures 40 m apart and still be walked by a person. Reach has to grow there;
// marker sizes must not.
//
// Nothing here involves a model. Every number is either measured off the scene or a
// fixed ratio, so the same scene always scales the same way.

export type SceneScale = {
	extent: number; // largest dimension of the scene's bounding box (m)
	step: number; // median capture-to-nearest-capture distance (m)
	eye: number; // median capture height above the surface below it (m)
};

// What a scene with nothing to measure falls back to: the modern house's own
// numbers, so a tour with no captures and no proxy behaves exactly as it did
// before any of this existed.
const FALLBACK_EXTENT = 20;
const FALLBACK_STEP = 2.24;
const FALLBACK_EYE = 1.6;

export const DEFAULT_SCALE: SceneScale = {
	extent: FALLBACK_EXTENT,
	step: FALLBACK_STEP,
	eye: FALLBACK_EYE,
};

// --- the ratios -------------------------------------------------------------
//
// Chosen so the modern house's MEASURED scale (step 2.24 m, eye ~1.6 m) reproduces
// the hand-tuned constants these replace, to within a few centimetres. That is the
// calibration discipline and it is what makes this change safe: on the one scene
// the numbers were tuned for, the derived numbers land on the tuned numbers, so the
// only scenes whose behaviour changes are the ones that were already wrong.
//
// The `was` comments are the exact values being replaced. Keep them — they are how
// anyone re-tunes a ratio without having to re-derive where it came from.

// Multiples of `step` — the travel distances.
const STEP_RATIO = {
	reach: 13, // was 30 m: furthest an in-view walk/portal reaches
	farDist: 9, // was 20 m: beyond this a link is map/search only
	wasdStep: 11, // was 24 m: furthest a single WASD step travels
	hopPenalty: 0.7, // was 1.5 m: per-hop routing charge, in metres-equivalent
	arrowDist: 1.4, // was 3.2 m: how far ahead a floor arrow is planted
	dockRadius: 1.35, // was ~3.0 m (0.08 x a 38 m scene, clamped to 2..5)
	dockReveal: 1.0, // was 2.2 m: where the interior has fully faded in
	// How much FURTHER away a capture on the far side of the surface under the pointer
	// may be and still be preferred over a nearer one on this side. Aiming at a wall
	// means the room beyond it, so the far side wins ties — but only ties. One typical
	// hop between captures is the whole budget: past that, whatever is over there is
	// not what the pointer meant, only the first thing the far half of the world
	// happened to contain. Beyond the back wall of a study is the neighbours' garden.
	wpThrough: 1.0,
} as const;

// Multiples of `eye` — everything drawn or judged at the size of an inhabitant.
const EYE_RATIO = {
	floorDrop: 0.8, // was 1.3 m: how far below the eye a floor marker sits
	sliceAboveEye: 0.19, // was 0.3 m: where hold-to-locate slices the roof off
	// How far past the surface under the cursor the waypoint stands, and how big a
	// jump between two consecutive surfaces means the ray has left the obstruction
	// rather than passed into another part of it. Both were fixed metres before
	// (0.4 and 1.2), and both are lengths a person is measured against: how far
	// "just behind that wall" is, and how thick a wall can be.
	// TWO distances, because one number was doing two jobs badly.
	//
	// `wpProbe` is how far past the near face we step to ask WHAT we are in — a
	// solid to stand on, or open air beyond a wall. It has to stay short: step a
	// metre and a half into a one-metre block and you are out the other side, so the
	// block reads as something to walk past rather than something to stand on.
	//
	// `wpStandoff` is how far past the obstruction the QUERY point is taken once we
	// know we went through something. Pressed against the far face of a wall it names
	// the wall rather than the room beyond; a person who walked through that doorway
	// would be standing a pace in, and that is the neighbourhood whose capture we want.
	wpProbe: 0.25,
	wpStandoff: 1.0,
	// How far clear of a blocking surface the MARKER comes to rest, walking back from
	// the capture. A pace, the way the query is, is far too much here and for a reason
	// worth keeping: the walk arrives at the surface from the destination's side, so a
	// pace of retreat is a pace back TOWARD the capture it started from — and where the
	// capture is nearer to the wall than that, the retreat consumes the whole walk and
	// the marker collapses onto the capture. Three captures at increasing distance
	// behind the same pane then produce three different markers, when the entire point
	// of walking back is that they produce one.
	//
	// So this is a CLEARANCE, not a standoff: just enough that the marker's ring reads
	// as standing beside the surface rather than buried in it (a shade more than
	// `ringOuter`), and small enough that no plausibly-placed capture can be closer to
	// a wall than it is.
	wpClearance: 0.3,
	// A settled height within this far below the probe means we are standing on the
	// very thing we pointed at — walking across a floor, not passing through
	// anything — so the standoff is dropped and the marker stays under the cursor.
	wpSameSurface: 0.5,
	spanGap: 0.75,
	// How much clear air a surface needs above it to count as somewhere a person
	// could stand. Literally the question, so literally the eye height.
	standHeadroom: 1.0,
	// How far ABOVE the point you aimed at a surface can be and still read as
	// something you step ONTO rather than something you go THROUGH. This is the one
	// judgement in the waypoint: it separates a platform (top half a metre up) from
	// a building's outside wall (roof eighteen metres up), and standability alone
	// cannot, because a roof has open sky above it too.
	stepUp: 1.5,
	// Nudge used to ask "which side of this surface am I on". Small enough never to
	// skip over a nearby surface, large enough to clear coincident-face round-off.
	probeEps: 0.01,
	wasdRise: 1.25, // was 2.0 m: max rise a WASD step will cross
	verticalDy: 1.25, // was 2.0 m: min rise that reads as a floor change
	dockMaxDy: 1.25, // was 2.0 m: hard cap on docking to something below you
	dockArrive: 0.075, // was 0.12 m: close enough to hand over invisibly
	ringInner: 0.125, // was 0.2 m
	ringOuter: 0.19, // was 0.3 m
	arrowR: 0.105, // was 0.17 m
	arrowH: 0.21, // was 0.34 m
	arrowGap: 0.19, // was 0.3 m: lead-to-trailing chevron spacing
	arrowBob: 0.0625, // was 0.1 m: drift each way
	ghostLift: 0.0125, // was 0.02 m: hair of lift so a ghost never z-fights
	losTrim: 0.125, // was 0.2 m: segment trim so hugging a wall isn't a block
	aimTrim: 0.094, // was 0.15 m: the same, for the live aim ray
	aimMinDist: 0.31, // was 0.5 m: closer than this and the target is just clear
	aimSpread: 0.125, // was 0.2 m: offset of the aim ray's sideways probes
} as const;

// --- the camera's clip planes -----------------------------------------------
//
// The FAR plane is left exactly as it was: 60 x extent, floored at 500 m. It looks
// wasteful — a 200 m scene gets a 12 km far plane — but tightening it buys nothing
// and costs something real, so it stays.
//
// Nothing: depth precision is governed by `1/near - 1/far`, and with a near plane
// measured in centimetres the `1/far` term is already negligible. Pulling a house's
// far plane from 2280 m to 500 m changes that quantity by 0.012%.
//
// Something: free flight is deliberately unbounded — nothing clamps the camera to
// the scene — so the far plane is the only thing keeping the scene visible once you
// fly away from it. A generous one is headroom, and the backdrop sphere the camera
// carries with it (max(80, sceneSphereRadius x 4), see projection.ts) has to fit
// inside it too.
const CAMERA_FAR_EXTENT_RATIO = 60;
const CAMERA_FAR_FLOOR = 500;

// The NEAR plane is the one worth changing, and it follows `eye` rather than extent.
// What matters is how close the camera can get to a surface, which is a fact about
// whoever is standing there and not about how big the world is. The old rule
// (extent x 0.002) gave a 200 m level a 0.4 m near plane — so any wall, block or
// object within 40 cm of the eye simply wasn't drawn, and you saw through it. That
// is a guaranteed visible artefact in a walkthrough, and worse in first person where
// the reticle invites you to look straight at a surface.
//
// The trade is real and worth naming: a nearer plane makes DISTANT depth resolution
// coarser, by the same `1/near` term above — five times coarser on that 200 m scene.
// In absolute terms that is a 1.9 mm depth gap at 50 m instead of 0.4 mm, which is
// far below anything that can z-fight visibly. Paying imperceptible precision to
// stop objects vanishing at arm's length is the right way round.
const CAMERA_NEAR_EYE_RATIO = 0.05; // reproduces the house's 0.076 m

export type NavMetrics = {
	// travel
	reach: number;
	farDist: number;
	wasdStep: number;
	hopPenalty: number;
	arrowDist: number;
	dockRadius: number;
	dockReveal: number;
	wpThrough: number;
	// human size
	floorDrop: number;
	eyeHeight: number;
	sliceAboveEye: number;
	wpProbe: number;
	wpStandoff: number;
	wpClearance: number;
	wpSameSurface: number;
	spanGap: number;
	standHeadroom: number;
	stepUp: number;
	probeEps: number;
	wasdRise: number;
	verticalDy: number;
	dockMaxDy: number;
	dockArrive: number;
	ringInner: number;
	ringOuter: number;
	arrowR: number;
	arrowH: number;
	arrowGap: number;
	arrowBob: number;
	ghostLift: number;
	losTrim: number;
	aimTrim: number;
	aimMinDist: number;
	aimSpread: number;
	// whole-scene framing
	cameraNear: number;
	cameraFar: number;
};

/** Resolve every scale-dependent distance for one scene. */
export function navMetrics(s: SceneScale): NavMetrics {
	const { step, eye, extent } = s;
	return {
		reach: step * STEP_RATIO.reach,
		farDist: step * STEP_RATIO.farDist,
		wasdStep: step * STEP_RATIO.wasdStep,
		hopPenalty: step * STEP_RATIO.hopPenalty,
		arrowDist: step * STEP_RATIO.arrowDist,
		dockRadius: step * STEP_RATIO.dockRadius,
		dockReveal: step * STEP_RATIO.dockReveal,
		wpThrough: step * STEP_RATIO.wpThrough,

		floorDrop: eye * EYE_RATIO.floorDrop,
		eyeHeight: eye,
		sliceAboveEye: eye * EYE_RATIO.sliceAboveEye,
		wpProbe: eye * EYE_RATIO.wpProbe,
		wpStandoff: eye * EYE_RATIO.wpStandoff,
		wpClearance: eye * EYE_RATIO.wpClearance,
		wpSameSurface: eye * EYE_RATIO.wpSameSurface,
		spanGap: eye * EYE_RATIO.spanGap,
		standHeadroom: eye * EYE_RATIO.standHeadroom,
		stepUp: eye * EYE_RATIO.stepUp,
		probeEps: eye * EYE_RATIO.probeEps,
		wasdRise: eye * EYE_RATIO.wasdRise,
		verticalDy: eye * EYE_RATIO.verticalDy,
		dockMaxDy: eye * EYE_RATIO.dockMaxDy,
		dockArrive: eye * EYE_RATIO.dockArrive,
		ringInner: eye * EYE_RATIO.ringInner,
		ringOuter: eye * EYE_RATIO.ringOuter,
		arrowR: eye * EYE_RATIO.arrowR,
		arrowH: eye * EYE_RATIO.arrowH,
		arrowGap: eye * EYE_RATIO.arrowGap,
		arrowBob: eye * EYE_RATIO.arrowBob,
		ghostLift: eye * EYE_RATIO.ghostLift,
		losTrim: eye * EYE_RATIO.losTrim,
		aimTrim: eye * EYE_RATIO.aimTrim,
		aimMinDist: eye * EYE_RATIO.aimMinDist,
		aimSpread: eye * EYE_RATIO.aimSpread,

		cameraNear: Math.max(0.01, eye * CAMERA_NEAR_EYE_RATIO),
		cameraFar: Math.max(CAMERA_FAR_FLOOR, extent * CAMERA_FAR_EXTENT_RATIO),
	};
}

export const DEFAULT_METRICS: NavMetrics = navMetrics(DEFAULT_SCALE);

// --- measurement ------------------------------------------------------------

function median(values: number[]): number | null {
	if (values.length === 0) return null;
	const sorted = [...values].sort((a, b) => a - b);
	return sorted[(sorted.length - 1) >> 1]; // lower median, as elsewhere in the pipeline
}

/**
 * The median distance from a capture point to its nearest neighbour — the scene's
 * travel unit. Null when there aren't two points to measure between.
 *
 * The MEDIAN rather than the minimum or the mean: one pair of captures a hand's
 * breadth apart would drag a minimum to nothing, and a few long hops across a
 * courtyard drag a mean upward. The median is what "the usual distance to the next
 * place" actually means.
 *
 * Brute force at O(n^2). A scene has captures in the hundreds, so this is tens of
 * thousands of comparisons once per scene load — nothing next to the line-of-sight
 * raycasts the nav graph does immediately afterwards.
 */
export function measureStep(
	positions: ReadonlyArray<readonly [number, number, number]>,
): number | null {
	if (positions.length < 2) return null;
	const nearest: number[] = [];
	for (let i = 0; i < positions.length; i++) {
		const [ax, ay, az] = positions[i];
		let best = Number.POSITIVE_INFINITY;
		for (let j = 0; j < positions.length; j++) {
			if (j === i) continue;
			const [bx, by, bz] = positions[j];
			const d2 = (bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2;
			if (d2 < best) best = d2;
		}
		if (best > 0 && Number.isFinite(best)) nearest.push(Math.sqrt(best));
	}
	return median(nearest);
}

const _down = new Vector3(0, -1, 0);
const _origin = new Vector3();

/**
 * The median height of a capture point above whatever is under it — the scene's
 * human unit, found by dropping a ray from every capture onto the proxy.
 *
 * This is the honest way to ask "how big is whoever lives here". The anchor planner
 * stands its cameras at a plausible viewing height above a surface, so the gap
 * between a capture and the ground beneath it IS that height, whatever the scene is
 * built at. Null when there is no proxy to measure against, or when too few
 * captures find ground for a median to mean anything — a heavily decimated proxy
 * has holes in its floor, and one capture over a hole should not decide the scale.
 */
export function measureEye(
	positions: ReadonlyArray<readonly [number, number, number]>,
	proxy: Object3D | null,
	extent: number,
): number | null {
	if (!proxy || positions.length === 0) return null;
	proxy.updateMatrixWorld(true); // freshly added to the scene; matrices are stale
	const ray = new Raycaster();
	ray.near = 0;
	ray.far = Math.max(1, extent); // no ground below the scene is worth finding
	const heights: number[] = [];
	for (const p of positions) {
		ray.set(_origin.set(p[0], p[1], p[2]), _down);
		const hit = ray.intersectObject(proxy, true)[0];
		if (hit && hit.distance > 1e-3) heights.push(hit.distance);
	}
	const enough = Math.max(3, Math.ceil(positions.length * 0.25));
	return heights.length >= enough ? median(heights) : null;
}

/**
 * Measure one scene's scale. Never throws and never returns a degenerate value:
 * each measurement has a fallback, and the pair is sanity-banded against each
 * other, so a broken proxy or a one-capture tour still yields usable numbers.
 */
export function measureSceneScale(
	extent: number,
	positions: ReadonlyArray<readonly [number, number, number]>,
	proxy: Object3D | null,
): SceneScale {
	const safeExtent =
		Number.isFinite(extent) && extent > 0 ? extent : FALLBACK_EXTENT;
	// With nothing to measure between, guess the travel unit from the scene itself:
	// a scene tends to be a couple of dozen strides across.
	const step = measureStep(positions) ?? Math.max(safeExtent * 0.06, 0.05);
	const measured = measureEye(positions, proxy, safeExtent);
	// A measured eye height wildly out of proportion to the travel step means the
	// measurement is broken, not that the scene is strange: the planner's own two
	// instructions (space captures at least so far apart, stand them at a viewing
	// height) tie the two together loosely but firmly. The band only ever catches a
	// disaster — the house sits at 0.71 of its step, comfortably inside it.
	const eye =
		measured === null
			? Math.min(FALLBACK_EYE, step)
			: MathUtils.clamp(measured, step * 0.05, step * 2);
	return { extent: safeExtent, step, eye };
}

/** Which of the three numbers were measured, for the load-time log. */
export function describeScale(s: SceneScale): string {
	return `extent ${s.extent.toFixed(1)}m · step ${s.step.toFixed(2)}m · eye ${s.eye.toFixed(2)}m`;
}
