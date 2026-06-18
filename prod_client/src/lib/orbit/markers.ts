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
export const HOTSPOT_MAX_OCCLUDED = 6; // X closest behind-wall anchors kept as yellow ghosts
export const HOTSPOT_OCCLUDE_EPS = 0.2; // trim both ends so a hugged wall isn't a block
export const ENTRY_TARGET_PX = 5; // overview entry discs render as small dots
export const AUTO_AIM_PX = 42; // interior pick/hover magnetism radius
export const ENTRY_AIM_PX = 26; // tighter pick radius for the smaller entry discs
export const HOTSPOT_BASE_RADIUS = 0.16; // the disc geometry's world radius
// Every anchor shows a white ring laid flat on the floor: world-fixed size
// (scales with distance like a real object) and depth-tested (scene geometry hides it).
export const ANCHOR_RING_INNER = 0.2; // world-space inner radius of the anchor ring
export const ANCHOR_RING_OUTER = 0.3; // world-space outer radius of the anchor ring
export const ANCHOR_RING_OPACITY = 0.4; // anchor rings are faint / transparent
// The X closest behind-wall anchors reuse the anchor-ring look in a warm gold:
// brighter, larger, and drawn over everything so they read as reachable through walls.
export const ANCHOR_RING_OCCLUDED_COLOR = 0xffce73; // warm gold
export const ANCHOR_RING_OCCLUDED_OPACITY = 0.6; // translucent so the hover-to-opaque pop reads clearly (still bolder than white)
export const ANCHOR_RING_OCCLUDED_SCALE = 1.5; // larger than the white rings, for visibility
export const CAPTURE_EYE_HEIGHT = 1.6; // panos are shot at eye height; floor sits this far below
export const PEEK_ROTATE_SPEED = 0.5; // rad/s the dollhouse spins while locating
export const WASD_MAX_Y_STEP = 2.0; // m: max |Δy| a WASD step will cross (blocks floor hops)
export const WASD_MAX_STEP = 24.0; // m: furthest a single WASD step will travel (XZ distance)
export const WASD_DIR_COS = Math.SQRT1_2; // cos(45°): WASD takes the nearest anchor in a quadrant-wide cone

// A flat floor disc + ring. Lies flat (normal up) so it reads as a spot on the
// floor; screen-space auto-aim (see pickByScreen) handles forgiving picking.
export function makeDisc(
	targetIndex: number,
	color: number,
	ringColor: number,
): Group {
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
		new RingGeometry(
			HOTSPOT_BASE_RADIUS * 1.38,
			HOTSPOT_BASE_RADIUS * 1.69,
			48,
		),
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
		new MeshBasicMaterial({
			color: 0xff3030,
			depthTest: false,
			depthWrite: false,
		}),
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
		new LineBasicMaterial({
			color: 0xff3030,
			transparent: true,
			opacity: 0.8,
			depthTest: false,
		}),
	);
	for (const m of [sphere, ring, line]) m.renderOrder = 999;
	group.add(sphere, ring, line);
	return { group, sphere, ring, line };
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
	const worldRadius =
		(targetPx * 2 * d * Math.tan((fov * Math.PI) / 360)) / h;
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
