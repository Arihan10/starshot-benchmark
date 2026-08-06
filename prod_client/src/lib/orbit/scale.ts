import { type Object3D, MathUtils, Raycaster, Vector3 } from "three";

export type SceneScale = {
	extent: number;
	step: number;
	eye: number;
};

const FALLBACK_EXTENT = 20;
const FALLBACK_STEP = 2.24;
const FALLBACK_EYE = 1.6;

export const DEFAULT_SCALE: SceneScale = {
	extent: FALLBACK_EXTENT,
	step: FALLBACK_STEP,
	eye: FALLBACK_EYE,
};

const STEP_RATIO = {
	reach: 13,
	farDist: 9,
	wasdStep: 11,
	hopPenalty: 0.7,
	arrowDist: 1.4,
	dockRadius: 1.35,
	dockReveal: 1.0,
	wpThrough: 1.0,
} as const;

const EYE_RATIO = {
	floorDrop: 0.8,
	sliceAboveEye: 0.19,
	wpProbe: 0.25,
	wpStandoff: 1.0,
	wpClearance: 0.3,
	wpSameSurface: 0.5,
	spanGap: 0.75,
	standHeadroom: 1.0,
	stepUp: 1.5,
	probeEps: 0.01,
	wasdRise: 1.25,
	verticalDy: 1.25,
	dockMaxDy: 1.25,
	dockArrive: 0.075,
	ringInner: 0.125,
	ringOuter: 0.19,
	arrowR: 0.105,
	arrowH: 0.21,
	arrowGap: 0.19,
	arrowBob: 0.0625,
	ghostLift: 0.0125,
	losTrim: 0.125,
	aimTrim: 0.094,
	aimMinDist: 0.31,
	aimSpread: 0.125,
} as const;

const CAMERA_FAR_EXTENT_RATIO = 60;
const CAMERA_FAR_FLOOR = 500;

const CAMERA_NEAR_EYE_RATIO = 0.05;

export type NavMetrics = {
	reach: number;
	farDist: number;
	wasdStep: number;
	hopPenalty: number;
	arrowDist: number;
	dockRadius: number;
	dockReveal: number;
	wpThrough: number;
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
	cameraNear: number;
	cameraFar: number;
};

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

function median(values: number[]): number | null {
	if (values.length === 0) return null;
	const sorted = [...values].sort((a, b) => a - b);
	return sorted[(sorted.length - 1) >> 1];
}

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

export function measureEye(
	positions: ReadonlyArray<readonly [number, number, number]>,
	proxy: Object3D | null,
	extent: number,
): number | null {
	if (!proxy || positions.length === 0) return null;
	proxy.updateMatrixWorld(true);
	const ray = new Raycaster();
	ray.near = 0;
	ray.far = Math.max(1, extent);
	const heights: number[] = [];
	for (const p of positions) {
		ray.set(_origin.set(p[0], p[1], p[2]), _down);
		const hit = ray.intersectObject(proxy, true)[0];
		if (hit && hit.distance > 1e-3) heights.push(hit.distance);
	}
	const enough = Math.max(3, Math.ceil(positions.length * 0.25));
	return heights.length >= enough ? median(heights) : null;
}

export function measureSceneScale(
	extent: number,
	positions: ReadonlyArray<readonly [number, number, number]>,
	proxy: Object3D | null,
): SceneScale {
	const safeExtent =
		Number.isFinite(extent) && extent > 0 ? extent : FALLBACK_EXTENT;
	const step = measureStep(positions) ?? Math.max(safeExtent * 0.06, 0.05);
	const measured = measureEye(positions, proxy, safeExtent);
	const eye =
		measured === null
			? Math.min(FALLBACK_EYE, step)
			: MathUtils.clamp(measured, step * 0.05, step * 2);
	return { extent: safeExtent, step, eye };
}

export function describeScale(s: SceneScale): string {
	return `extent ${s.extent.toFixed(1)}m · step ${s.step.toFixed(2)}m · eye ${s.eye.toFixed(2)}m`;
}
