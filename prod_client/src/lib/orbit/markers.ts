import {
	AdditiveBlending,
	BufferGeometry,
	ConeGeometry,
	DoubleSide,
	GreaterDepth,
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
import type { NavMetrics } from "./scale";

// Every DISTANCE this module used to define — the marker drop, the eye height, the
// ring radii, the arrow's size and stand-off, the WASD stride — now comes from
// scale.ts, measured per scene. What stays here is what has no length: angles,
// on-screen pixel sizes, opacities, rates, and the colour/shape grammar.
export const PEEK_ROTATE_SPEED = 0.5; // rad/s the dollhouse spins while locating
export const NAV_AIM_PX = 44; // interior affordance pick/hover magnetism radius

// WASD graph-walk gate. The stride and the rise it will cross are scale-derived
// (NavMetrics.wasdStep / wasdRise); the cone is an angle, so it is not.
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

// A solid floor ring for one affordance. Always depth-tested now: an overlay type
// gets its see-through-geometry behaviour from `depthSplit` instead, which shows it
// FAINT where something is in front rather than pretending nothing is.
function makeFloorRing(color: number, m: NavMetrics): Mesh {
	const ring = new Mesh(
		new RingGeometry(m.ringInner, m.ringOuter, 40),
		new MeshBasicMaterial({
			color,
			transparent: true,
			opacity: NAV_REST_OPACITY,
			side: DoubleSide,
			depthWrite: false,
		}),
	);
	ring.rotation.x = -Math.PI / 2;
	return ring;
}

// A dashed floor ring — the ghost puck for a phase edge, so "the map never lies":
// a sealed connection is drawn differently from a walk. Line dashes need
// per-vertex distances computed.
function makeDashedRing(color: number, m: NavMetrics): LineLoop {
	// Built in the XY plane like RingGeometry, so makeNavMarker's -90° X rotation
	// lays it flat on the floor (a ring built in XZ would stand up vertically).
	const pts: Vector3[] = [];
	for (let i = 0; i <= 48; i++) {
		const a = (i / 48) * Math.PI * 2;
		pts.push(new Vector3(Math.cos(a) * m.ringOuter, Math.sin(a) * m.ringOuter, 0));
	}
	const loop = new LineLoop(
		new BufferGeometry().setFromPoints(pts),
		new LineDashedMaterial({
			color,
			// Dash lengths are world distances along the ring, so they have to follow
			// the ring: a fixed 6 cm dash on a ring scaled down to 8 cm across is one
			// solid loop, and on a ring scaled up it is two long arcs.
			dashSize: m.ringOuter * 0.2,
			gapSize: m.ringOuter * 0.167,
			transparent: true,
			opacity: NAV_REST_OPACITY,
			depthWrite: false,
		}),
	);
	return loop;
}

// An upward glyph (cone). Portals point up like a beacon at the doorway; vertical
// shafts flip to point down when the destination is below and stand TALLER, so a
// floor-change reads as a distinct beacon (not just another floor ring) from afar.
function makeChevron(
	color: number,
	down: boolean,
	m: NavMetrics,
	size = 1,
): Mesh {
	const cone = new Mesh(
		new ConeGeometry(m.ringOuter * 0.7 * size, m.ringOuter * 1.6 * size, 4),
		new MeshBasicMaterial({
			color,
			transparent: true,
			opacity: NAV_REST_OPACITY,
			depthWrite: false,
			side: DoubleSide,
		}),
	);
	cone.rotation.y = Math.PI / 4; // face a flat toward the viewer
	if (down) cone.rotation.z = Math.PI;
	cone.position.y = m.ringOuter * 1.1 * size;
	return cone;
}

// How much of an overlay marker survives where the scene is in front of it. Not
// zero — a destination behind a wall still has to be findable — and not full,
// which is what it was: a marker drawn with the depth test OFF is pasted onto the
// screen, so a waypoint on the FAR side of a wall looked exactly like one on the
// near side, and the one thing the marker exists to say was the one thing it could
// not say. Matches the floor arrows, which had the same problem and this fix.
//
// TUNING. This multiplies whatever brightness the marker already has, so the two
// knobs interact: the ghost rests at GHOST_OPACITY 0.4, which puts its behind pass
// near 0.16 — close to the floor arrows' 0.19, faint enough to read as "behind
// that" and solid enough to still find. Lower it for a sharper distinction, at the
// cost of a waypoint you can lose behind a thick wall.
export const NAV_OCCLUDED = 0.4;

// Draw one part TWICE, split by depth test, so occlusion reads PER FRAGMENT: the
// part standing in open air stays solid while the part behind something goes faint,
// cut on the real silhouette of whatever is in the way.
//
// `GreaterDepth` draws ONLY where something is nearer than the marker, and the
// normal pass only where nothing is — each fragment is drawn exactly once, so there
// is no double-blended seam along the boundary. With nothing writing depth (sphere
// mode) the solid pass covers everything and the marker simply reads as unoccluded.
//
// The faint copy carries `occludedFactor` so the opacity machinery — gaze fade,
// ghost translucency, the dwell pulse — keeps scaling both passes together instead
// of flattening them back to one brightness (see markerLayer's setGroupOpacity).
function depthSplit(part: Mesh | LineLoop): Object3D[] {
	const behind = part.clone();
	const mat = (part.material as MeshBasicMaterial).clone();
	mat.depthFunc = GreaterDepth;
	mat.opacity = (part.material as MeshBasicMaterial).opacity * NAV_OCCLUDED;
	mat.userData.occludedFactor = NAV_OCCLUDED;
	behind.material = mat;
	return [part, behind];
}

// Build the affordance group for one typed edge, anchored at the destination's
// floor. `userData.edge` carries the edge so hover/click read straight off it.
// `forceChevron` gives a type that normally goes bare (the walk puck) the same
// upward arrow a portal wears — the auto-home ghost sets it so the destination
// always reads as a pointer rather than a ring lying flat on the floor.
export function makeNavMarker(
	edge: { to: number; type: EdgeType; dy: number },
	floorPos: Vector3,
	m: NavMetrics,
	forceChevron = false,
): Group {
	const group = new Group();
	const color = NAV_COLORS[edge.type];
	// An overlay type marks somewhere you may not be able to see, so it stays
	// findable through geometry — but drawn so it reads as being behind it.
	const overlay = edge.type !== "walk";
	const parts: Object3D[] = [];
	if (edge.type === "phase") {
		const ring = makeDashedRing(color, m);
		ring.rotation.x = -Math.PI / 2;
		ring.computeLineDistances?.();
		parts.push(ring);
	} else {
		parts.push(makeFloorRing(color, m));
	}
	if (edge.type === "vertical") {
		parts.push(makeChevron(color, edge.dy < 0, m, 1.6));
	} else if (edge.type === "portal" || forceChevron) {
		parts.push(makeChevron(color, false, m));
	}
	// An overlay type (portal / vertical / phase) marks somewhere you cannot
	// necessarily see, so it must stay findable through geometry — but it has to
	// LOOK like it is behind that geometry rather than in front of it.
	const drawn = overlay ? parts.flatMap((p) => depthSplit(p as Mesh)) : parts;
	for (const p of drawn) p.renderOrder = overlay ? 6 : 3;
	group.add(...drawn);
	group.position.copy(floorPos);
	group.renderOrder = overlay ? 6 : 3;
	group.userData.to = edge.to;
	group.userData.type = edge.type;
	group.userData.overlay = overlay;
	return group;
}

// --- floor arrows -------------------------------------------------------------
// The way to the storey above or below, drawn as a pair of drifting chevrons
// placed IN FRONT OF YOU rather than at the destination.
//
// Placing it at the destination was the problem with every earlier attempt: that
// point is on another floor, which is by definition the one place you cannot see,
// so the marker had to be hunted for. Placing it on the arrival heading means it
// is simply there when you land — no looking around — and clicking it snaps to the
// nearest capture on that floor.
//
// Two chevrons rather than one — a lead and a smaller, fainter trailing one — so
// the glyph has a direction built into its silhouette. They drift as a RIGID PAIR:
// the two sit only ~1cm apart, so bobbing them on offset phases (which is what the
// first version did, to suggest flow) pulled them up to 7cm apart and back, and an
// arrow that comes apart and reassembles reads as two objects rather than one.
// Overlay-drawn (depthTest off), because a way out of the room must not be hidden
// by the room.
export const FLOOR_ARROW_COLOR = NAV_COLORS.vertical;
// The arrow's distance from the eye, its radius, its height, the gap between the
// two chevrons and how far it drifts are all lengths, so they are scale-derived
// (NavMetrics.arrowDist / arrowR / arrowH / arrowGap / arrowBob).
//
// There is no placement ANGLE any more. The arrows used to be pitched out toward
// the top and bottom edges of the frame so they were in view without being looked
// for; they now sit straight up and straight down from where you stand, which is
// where the way out of a storey actually is (see the engine's refreshFloorArrows).
export const FLOOR_ARROW_RATE = 0.0018; // rad/ms — a drift, not a bounce
export const FLOOR_ARROW_REST = 0.85; // resting opacity of the lead chevron
// What survives of an arrow where the scene is in front of it. Not zero: a way out
// of the room has to stay findable through the floor between you and it. Not full
// either, which is what it was — an arrow as bright through a ceiling as in open
// air reads as pasted onto the screen rather than standing in the world, and tells
// you nothing about whether the way there is clear.
export const FLOOR_ARROW_OCCLUDED = 0.22;
// Hover glow. A halo of the same shape, scaled up and blended ADDITIVELY, so it
// reads as the arrow giving off light rather than as a bigger arrow behind it.
// Drawn without a depth test on purpose, unlike the body: hover feedback has to
// arrive even when what you are pointing at is behind a slab, or pointing at it
// feels broken.
//
// It has to hug the SILHOUETTE. Scaled well up it stops being light and becomes a
// second, larger arrow behind the first — which is what 1.35x was. Just over 1
// leaves a thin rim standing proud of the body, and additive light on that rim is
// what reads as glow.
export const FLOOR_ARROW_GLOW_SCALE = 1.12;
// Kept DELIBERATELY faint. Additive light on an already-bright teal climbs fast,
// and it is confirming attention you have already given — you only ever see this
// because you pointed at the thing.
export const FLOOR_ARROW_GLOW = 0.16; // peak additive alpha
export const FLOOR_ARROW_PULSE_RATE = 0.0021; // rad/ms — a ~3s breath
export const FLOOR_ARROW_HOVER_LIFT = 0.18; // how much the body itself brightens
// Time constant for the glow rising and falling. Without it the whole effect
// switched on at full strength the instant the cursor crossed the arrow, so what
// you noticed was the STEP, not the light — a slow pulse that arrives by snapping
// on is still a snap.
export const FLOOR_ARROW_GLOW_TAU = 260; // ms

export function makeFloorArrow(up: boolean, m: NavMetrics): Group {
	const group = new Group();
	for (let i = 0; i < 2; i++) {
		const scale = 1 - i * 0.3;
		const opacity = i === 0 ? FLOOR_ARROW_REST : FLOOR_ARROW_REST * 0.45;
		const geometry = new ConeGeometry(
			m.arrowR * scale,
			m.arrowH * scale,
			4,
		);
		// Two draws of the SAME geometry, split by depth test, so occlusion reads
		// PER FRAGMENT: the part of the arrow standing in open air stays solid while
		// the part buried in a slab goes faint, cut on the real silhouette of
		// whatever is in the way. `GreaterDepth` makes the faint pass draw ONLY where
		// something is nearer than the arrow, and the solid pass only where nothing
		// is — each fragment is drawn exactly once, so there is no double-blending
		// seam along the boundary. With nothing writing depth (sphere mode) the
		// solid pass covers everything and the arrow simply reads as unoccluded.
		for (const occluded of [false, true]) {
			const alpha = occluded ? opacity * FLOOR_ARROW_OCCLUDED : opacity;
			const cone = new Mesh(
				geometry,
				new MeshBasicMaterial({
					color: FLOOR_ARROW_COLOR,
					transparent: true,
					opacity: alpha,
					side: DoubleSide,
					depthWrite: false,
					...(occluded ? { depthFunc: GreaterDepth } : {}),
				}),
			);
			cone.rotation.y = Math.PI / 4; // present a flat face, not an edge
			if (!up) cone.rotation.z = Math.PI;
			// The trailing chevron sits BEHIND the lead one along the travel
			// direction, so the pair points the way it drifts. Fixed relative to its
			// partner — only the GROUP drifts, so the pair holds its shape.
			cone.position.y = (up ? -1 : 1) * i * m.arrowGap;
			cone.userData.baseOpacity = alpha;
			cone.renderOrder = 7;
			group.add(cone);
		}
		const halo = new Mesh(
			geometry,
			new MeshBasicMaterial({
				color: FLOOR_ARROW_COLOR,
				transparent: true,
				opacity: 0, // invisible until hovered
				side: DoubleSide,
				depthWrite: false,
				depthTest: false,
				blending: AdditiveBlending,
			}),
		);
		halo.rotation.y = Math.PI / 4;
		if (!up) halo.rotation.z = Math.PI;
		halo.position.y = (up ? -1 : 1) * i * m.arrowGap;
		halo.scale.setScalar(FLOOR_ARROW_GLOW_SCALE);
		halo.userData.halo = true;
		halo.userData.baseOpacity = 0;
		halo.renderOrder = 6; // behind the body it surrounds
		group.add(halo);
	}
	group.renderOrder = 7;
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
// `baseRadius` is the marker's own world size and is now REQUIRED: it used to
// default to a fixed 0.16 m, which silently mis-scaled anything whose real radius
// came from the scene's measured scale rather than that constant.
export function screenScaleForDistance(
	d: number,
	targetPx: number,
	fov: number,
	stageHeight: number,
	baseRadius: number,
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
