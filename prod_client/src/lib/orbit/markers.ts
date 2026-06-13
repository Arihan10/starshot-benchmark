import {
	BufferGeometry,
	CircleGeometry,
	DoubleSide,
	Group,
	Line,
	LineBasicMaterial,
	MathUtils,
	Mesh,
	MeshBasicMaterial,
	type Object3D,
	type PerspectiveCamera,
	RingGeometry,
	SphereGeometry,
	Vector3,
} from "three";

export const HOTSPOT_FLOOR_DROP = 1.3; // meters below eye level (markers sit on the floor)
export const HOTSPOT_REACH = 30; // furthest an interior anchor can be and still show
export const HOTSPOT_MAX_VISIBLE = 10; // line-of-sight anchors shown
export const HOTSPOT_MAX_OCCLUDED = 6; // behind-wall anchors shown (as ghosts)
export const HOTSPOT_OCCLUDE_EPS = 0.2; // trim both ends so a hugged wall isn't a block
export const HOTSPOT_TARGET_PX = 24; // interior hotspot radius on screen, in CSS px
export const RETICLE_TARGET_PX = 30; // on-surface cursor radius on screen, in CSS px
export const ENTRY_TARGET_PX = 12; // overview entry discs render smaller
export const AUTO_AIM_PX = 42; // interior pick/hover magnetism radius
export const ENTRY_AIM_PX = 26; // tighter pick radius for the smaller entry discs
export const HOTSPOT_BASE_RADIUS = 0.16; // the disc geometry's world radius
export const CAPTURE_EYE_HEIGHT = 1.6; // panos are shot at eye height; floor sits this far below
export const PEEK_ROTATE_SPEED = 0.5; // rad/s the dollhouse spins while locating
export const STEP_CONE_HALF_ANGLE = MathUtils.degToRad(40); // half-angle of each WASD pick cone
export const CLICK_CONE_HALF_ANGLE = MathUtils.degToRad(18); // half-angle of the click direction pick

// A flat floor disc + ring. Lies flat (normal up) so it reads as a spot on the
// floor; screen-space auto-aim (see pickByScreen) handles forgiving picking.
export function makeDisc(targetIndex: number, color: number, ringColor: number): Group {
	const group = new Group();
	const disc = new Mesh(
		new CircleGeometry(HOTSPOT_BASE_RADIUS, 40),
		new MeshBasicMaterial({
			color,
			transparent: true,
			opacity: 0.55,
			side: DoubleSide,
			depthTest: false,
			depthWrite: false,
		}),
	);
	const ring = new Mesh(
		new RingGeometry(HOTSPOT_BASE_RADIUS * 1.38, HOTSPOT_BASE_RADIUS * 1.69, 48),
		new MeshBasicMaterial({
			color: ringColor,
			transparent: true,
			opacity: 0.9,
			side: DoubleSide,
			depthTest: false,
			depthWrite: false,
		}),
	);
	disc.rotation.x = -Math.PI / 2;
	ring.rotation.x = -Math.PI / 2;
	disc.renderOrder = 2;
	ring.renderOrder = 2;
	group.add(disc, ring);
	group.renderOrder = 2;
	group.userData.targetIndex = targetIndex;
	return group;
}

export type YouMarker = { group: Group; sphere: Mesh; ring: Mesh; line: Line };

// "You are here": a red pin (eye-height sphere + floor ring + connector) drawn
// over everything (depthTest off) so the dollhouse never hides it.
export function makeYouMarker(): YouMarker {
	const group = new Group();
	group.visible = false;
	group.renderOrder = 999;
	const sphere = new Mesh(
		new SphereGeometry(1, 24, 16),
		new MeshBasicMaterial({ color: 0xff3030, depthTest: false, depthWrite: false }),
	);
	const ring = new Mesh(
		new RingGeometry(1.4, 1.9, 40),
		new MeshBasicMaterial({
			color: 0xff3030,
			transparent: true,
			opacity: 0.85,
			side: DoubleSide,
			depthTest: false,
			depthWrite: false,
		}),
	);
	ring.rotation.x = -Math.PI / 2;
	const line = new Line(
		new BufferGeometry().setFromPoints([new Vector3(), new Vector3()]),
		new LineBasicMaterial({ color: 0xff3030, transparent: true, opacity: 0.8, depthTest: false }),
	);
	for (const m of [sphere, ring, line]) m.renderOrder = 999;
	group.add(sphere, ring, line);
	return { group, sphere, ring, line };
}

// The on-surface cursor: a ring + center dot lying in the XY plane (normal +Z),
// so the engine can lay it flat on whatever proxy face is under the pointer by
// aligning +Z with the hit normal. Outer radius matches HOTSPOT_BASE_RADIUS so
// hotspotScaleForDistance keeps it a constant pixel size. Drawn over everything
// (depthTest off) so it reads as a crisp reticle gliding across the surface.
export function makeReticle(): Group {
	const group = new Group();
	group.visible = false;
	group.renderOrder = 6;
	const ring = new Mesh(
		new RingGeometry(HOTSPOT_BASE_RADIUS * 0.78, HOTSPOT_BASE_RADIUS, 48),
		new MeshBasicMaterial({
			color: 0x9ad4ff,
			transparent: true,
			opacity: 0.95,
			side: DoubleSide,
			depthTest: false,
			depthWrite: false,
		}),
	);
	const dot = new Mesh(
		new CircleGeometry(HOTSPOT_BASE_RADIUS * 0.16, 24),
		new MeshBasicMaterial({
			color: 0x9ad4ff,
			transparent: true,
			opacity: 0.6,
			side: DoubleSide,
			depthTest: false,
			depthWrite: false,
		}),
	);
	for (const m of [ring, dot]) m.renderOrder = 6;
	group.add(ring, dot);
	return group;
}

// World scale that renders a disc at ~targetPx on screen at distance `d` —
// perspective shrinks worldSize/distance, so scaling by distance keeps the click
// target a constant pixel size no matter how far the anchor is (or the fov).
export function hotspotScaleForDistance(
	d: number,
	targetPx: number,
	fov: number,
	stageHeight: number,
): number {
	const h = stageHeight || 1;
	const worldRadius = (targetPx * 2 * d * Math.tan((fov * Math.PI) / 360)) / h;
	return MathUtils.clamp(worldRadius / HOTSPOT_BASE_RADIUS, 0.15, 14);
}

const _aimWorld = new Vector3();

// Project each marker's center to the screen and lock onto the NEAREST within
// `maxPx` of the cursor. Forgiving targeting that keeps the discs glued flat.
export function pickByScreen(
	clientX: number,
	clientY: number,
	group: Group,
	maxPx: number,
	camera: PerspectiveCamera,
	canvas: HTMLCanvasElement,
): Object3D | null {
	const rect = canvas.getBoundingClientRect();
	const cx = clientX - rect.left;
	const cy = clientY - rect.top;
	let best: Object3D | null = null;
	let bestPx = maxPx;
	for (const spot of group.children) {
		spot.getWorldPosition(_aimWorld).project(camera);
		if (_aimWorld.z > 1) continue;
		const sx = (_aimWorld.x * 0.5 + 0.5) * rect.width;
		const sy = (-_aimWorld.y * 0.5 + 0.5) * rect.height;
		const px = Math.hypot(sx - cx, sy - cy);
		if (px < bestPx) {
			bestPx = px;
			best = spot;
		}
	}
	return best;
}
