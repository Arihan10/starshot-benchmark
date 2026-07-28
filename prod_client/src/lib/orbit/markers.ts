import {
	BufferGeometry,
	ConeGeometry,
	DoubleSide,
	Group,
	Line,
	LineBasicMaterial,
	LineDashedMaterial,
	LineLoop,
	MathUtils,
	Mesh,
	MeshBasicMaterial,
	type Object3D,
	type PerspectiveCamera,
	RingGeometry,
	SphereGeometry,
	Vector3,
} from "three";
import type { EdgeType } from "./navGraph";

export const HOTSPOT_FLOOR_DROP = 1.3; // meters below eye level (markers sit on the floor)
export const CAPTURE_EYE_HEIGHT = 1.6; // panos are shot at eye height; floor sits this far below
export const PEEK_ROTATE_SPEED = 0.5; // rad/s the dollhouse spins while locating
export const LOCATE_SLICE_ABOVE_EYE = 0.3; // m above the eye where hold-to-locate slices the roof off
export const NAV_AIM_PX = 44; // interior affordance pick/hover magnetism radius
export const HOTSPOT_BASE_RADIUS = 0.16; // default world radius for on-screen size scaling

// WASD graph-walk gates (nearest edge inside a forward cone).
export const WASD_MAX_Y_STEP = 2.0; // m: max |Δy| a WASD step will cross (keeps you on-floor)
export const WASD_MAX_STEP = 24.0; // m: furthest a single WASD step travels (XZ)
export const WASD_DIR_COS = Math.SQRT1_2; // cos(45°): quadrant-wide cone per key

// --- typed navigation affordances -------------------------------------------
// One shape + hue per edge type, held constant across every scene so the grammar
// transfers between wildly different environments (that consistency is the brand):
//   walk → cyan floor puck · portal → warm chevron · vertical → teal up/down
//   chevron · phase → violet dashed ghost ring.
export const NAV_COLORS: Record<EdgeType, number> = {
	walk: 0x8fd0ff,
	portal: 0xffc46b,
	vertical: 0x7ef2c2,
	phase: 0xc9a6ff,
	far: 0x9aa7b4,
};
// The surface cursor's colour when the destination is in PLAIN SIGHT — a neutral
// light grey rather than the walk hue. The ring is on screen constantly, tracking
// every surface under the pointer, so the one state that means "nothing to explain,
// you can see the place you're about to go" should be the quietest thing in the
// frame. Reaching an occluded destination doesn't recolour it: the ring is hidden
// outright and the reach preview takes over (see the engine's updateCursorRing).
export const CURSOR_CLEAR = 0xc4ccd6;

export const NAV_RING_INNER = 0.2;
export const NAV_RING_OUTER = 0.3;
export const NAV_REST_OPACITY = 0.62; // affordances rest here, brightening toward the gaze — high enough to be findable off-axis
export const NAV_GAZE_RAD = (55 * Math.PI) / 180; // within this bearing of the gaze → full brightness
export const NAV_TARGET_PX = 15; // affordances never shrink below ~this on-screen size (distant points stay visible)
export const SONAR_DURATION = 3200; // ms the reveal front takes to sweep + fade

export type YouMarker = { group: Group; sphere: Mesh; ring: Mesh; line: Line };

// "You are here": a red pin drawn over everything (peek/locate).
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

// A solid floor ring for one affordance. `overlay` (portal/vertical/phase) draws
// it over the scene since the destination isn't visible; walk rings are depth-
// tested so furniture correctly hides them (that's what the proxy is for).
function makeFloorRing(color: number, overlay: boolean): Mesh {
	const ring = new Mesh(
		new RingGeometry(NAV_RING_INNER, NAV_RING_OUTER, 40),
		new MeshBasicMaterial({
			color,
			transparent: true,
			opacity: NAV_REST_OPACITY,
			side: DoubleSide,
			depthWrite: false,
			depthTest: !overlay,
		}),
	);
	ring.rotation.x = -Math.PI / 2;
	return ring;
}

// A dashed floor ring — the ghost puck for a phase edge, so "the map never lies":
// a sealed connection is drawn differently from a walk. Line dashes need
// per-vertex distances computed.
function makeDashedRing(color: number): LineLoop {
	// Built in the XY plane like RingGeometry, so makeNavMarker's -90° X rotation
	// lays it flat on the floor (a ring built in XZ would stand up vertically).
	const pts: Vector3[] = [];
	for (let i = 0; i <= 48; i++) {
		const a = (i / 48) * Math.PI * 2;
		pts.push(new Vector3(Math.cos(a) * NAV_RING_OUTER, Math.sin(a) * NAV_RING_OUTER, 0));
	}
	const loop = new LineLoop(
		new BufferGeometry().setFromPoints(pts),
		new LineDashedMaterial({
			color,
			dashSize: 0.06,
			gapSize: 0.05,
			transparent: true,
			opacity: NAV_REST_OPACITY,
			depthTest: false,
			depthWrite: false,
		}),
	);
	return loop;
}

// An upward glyph (cone). Portals point up like a beacon at the doorway; vertical
// shafts flip to point down when the destination is below and stand TALLER, so a
// floor-change reads as a distinct beacon (not just another floor ring) from afar.
function makeChevron(color: number, down: boolean, overlay = true, size = 1): Mesh {
	const cone = new Mesh(
		new ConeGeometry(NAV_RING_OUTER * 0.7 * size, NAV_RING_OUTER * 1.6 * size, 4),
		new MeshBasicMaterial({
			color,
			transparent: true,
			opacity: NAV_REST_OPACITY,
			depthWrite: false,
			depthTest: !overlay,
			side: DoubleSide,
		}),
	);
	cone.rotation.y = Math.PI / 4; // face a flat toward the viewer
	if (down) cone.rotation.z = Math.PI;
	cone.position.y = NAV_RING_OUTER * 1.1 * size;
	return cone;
}

// Build the affordance group for one typed edge, anchored at the destination's
// floor. `userData.edge` carries the edge so hover/click read straight off it.
// `forceChevron` gives a type that normally goes bare (the walk puck) the same
// upward arrow a portal wears — the auto-home ghost sets it so the destination
// always reads as a pointer rather than a ring lying flat on the floor.
export function makeNavMarker(
	edge: { to: number; type: EdgeType; dy: number },
	floorPos: Vector3,
	forceChevron = false,
): Group {
	const group = new Group();
	const color = NAV_COLORS[edge.type];
	const overlay = edge.type !== "walk"; // walk pucks depth-test; the rest draw over
	const parts: Object3D[] = [];
	if (edge.type === "phase") {
		const ring = makeDashedRing(color);
		ring.rotation.x = -Math.PI / 2;
		ring.computeLineDistances?.();
		parts.push(ring);
	} else {
		parts.push(makeFloorRing(color, overlay));
	}
	if (edge.type === "vertical") {
		parts.push(makeChevron(color, edge.dy < 0, true, 1.6));
	} else if (edge.type === "portal" || forceChevron) {
		// Match the ring's depth behaviour so a walk ghost's arrow doesn't float
		// through furniture while its ring is correctly occluded.
		parts.push(makeChevron(color, false, overlay));
	}
	for (const p of parts) p.renderOrder = overlay ? 6 : 3;
	group.add(...parts);
	group.position.copy(floorPos);
	group.renderOrder = overlay ? 6 : 3;
	group.userData.to = edge.to;
	group.userData.type = edge.type;
	group.userData.overlay = overlay;
	return group;
}

// A single sonar dot (x-ray, always over the scene) colored by its edge type.
export function makeSonarDot(color: number): Mesh {
	const dot = new Mesh(
		new SphereGeometry(1, 12, 8),
		new MeshBasicMaterial({
			color,
			transparent: true,
			opacity: 0,
			depthTest: false,
			depthWrite: false,
		}),
	);
	dot.renderOrder = 998;
	return dot;
}

// World scale that renders a marker at ~targetPx on screen at distance `d`.
export function screenScaleForDistance(
	d: number,
	targetPx: number,
	fov: number,
	stageHeight: number,
	baseRadius = HOTSPOT_BASE_RADIUS,
): number {
	const h = stageHeight || 1;
	const worldRadius = (targetPx * 2 * d * Math.tan((fov * Math.PI) / 360)) / h;
	return MathUtils.clamp(worldRadius / baseRadius, 0.15, 14);
}

const _aimWorld = new Vector3();

// Project each child's center to screen and lock onto the NEAREST within `maxPx`
// of the cursor. Forgiving targeting that keeps the affordances glued flat.
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
		if (!spot.visible) continue;
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
