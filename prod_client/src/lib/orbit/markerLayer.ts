import {
	Group,
	type Material,
	MathUtils,
	Mesh,
	type MeshBasicMaterial,
	type Object3D,
	type PerspectiveCamera,
	Raycaster,
	RingGeometry,
	type Scene,
	SphereGeometry,
	Vector2,
	Vector3,
} from "three";
import {
	CAPTURE_EYE_HEIGHT,
	HOTSPOT_FLOOR_DROP,
	LEVEL_HEIGHT,
	LEVEL_MAX_SCALE,
	LEVEL_REST_FACTOR,
	LEVEL_TARGET_PX,
	makeLevelWaypoint,
	makeNavMarker,
	makeSonarDot,
	makeYouMarker,
	NAV_AIM_PX,
	NAV_COLORS,
	NAV_GAZE_RAD,
	NAV_REST_OPACITY,
	NAV_RING_OUTER,
	NAV_TARGET_PX,
	pickByScreen,
	screenScaleForDistance,
	SONAR_DURATION,
	type YouMarker,
} from "./markers";
import { angleDelta, type EdgeType, type NavNode } from "./navGraph";
import type { PanoEntry } from "./panoTextures";

// The auto-home ghost rests fainter than a live affordance, so it reads as a
// preview of where a click would land rather than another button to press.
const GHOST_OPACITY = 0.4;

// --- suppressed STANDING affordances -----------------------------------------
//
// An edge type in this set renders no standing marker in buildNav(). It says
// nothing about the on-demand waypoint the cursor grows while homing — whether and
// where to draw that is the engine's call (updateCursorRing / portalWaypoint).
//
// TO RE-ENABLE a type, delete its entry — nothing else needs touching. The
// builders in markers.ts still handle all five types, so it comes straight back.
//
// WHY THESE ARE OFF:
//   walk (blue) — a walk edge is BY DEFINITION a spot you can already SEE from
//     where you stand, so its puck spends attention on what the view already told
//     you. With navGraph's nearest-3 guarantee most nodes drew several at once and
//     the floor filled with near-identical blue rings.
//   vertical (green) — level changes are the red monolith waypoints now
//     (levelGroup / the engine's refreshLevelWaypoints), so a green chevron is a
//     second, competing wayfinding layer saying the same thing.
//   portal (orange) — a STANDING marker for a destination behind a wall sits
//     somewhere you cannot see, and rarely where you are actually pointing. The
//     engine draws it on demand instead, on the cursor ray just past the surface
//     you are aiming at, so it answers "what happens if I click HERE".
// That leaves phase (violet), and only on a sealed node with no other way out.
//
// NOTHING BECOMES UNREACHABLE. Suppressed destinations keep every other route in:
// clicking the floor still auto-homes onto them (engine's clickAnywhere →
// autoHomeTarget scores every pano and ignores edge type), the surface cursor
// still tints by type while one is the homing target, and they stay in the exits
// panel, the minimap and the sonar ping. This hides scene geometry only — the nav
// graph, routing costs and traversal FX are all untouched.
const HIDDEN_AFFORDANCES: ReadonlySet<EdgeType> = new Set<EdgeType>([
	"walk",
	"vertical",
	"portal",
]);

const v3 = (a: [number, number, number]) => new Vector3().fromArray(a);
const _ndc = new Vector2();

// The interior navigation overlay. Renders the CURRENT node's typed affordances
// (every type except those in HIDDEN_AFFORDANCES), fades them by gaze bearing,
// drives the sonar reveal of every node, previews the auto-home destination, holds
// the red level waypoints, and keeps the peek "you are here" pin. The engine flips
// group visibility per mode and feeds in the live nav graph / camera each frame.
export class MarkerLayer {
	readonly navGroup = new Group(); // interior: the current node's typed affordances
	readonly sonarGroup = new Group(); // interior: x-ray reveal of every node
	readonly ghostGroup = new Group(); // interior: the auto-home destination preview
	readonly levelGroup = new Group(); // interior: one red monolith per adjacent floor
	readonly you: YouMarker = makeYouMarker();

	private hovered: Object3D | null = null;
	private hoveredLevelMarker: Object3D | null = null;
	private readonly levelRay = new Raycaster();
	private ghostType: EdgeType | null = null; // rebuild key: the ghost's current edge type
	private pulseUntil = 0; // dwell/never-trapped: briefly boost every affordance
	private sonarStart = 0;
	private sonarReach = 1;

	constructor(private readonly scene: Scene) {
		scene.add(
			this.navGroup,
			this.sonarGroup,
			this.ghostGroup,
			this.levelGroup,
			this.you.group,
		);
	}

	// Per-scene build: size the you-marker to the scene extent.
	build(sceneMaxDim: number) {
		this.sizeYouMarker(sceneMaxDim);
	}

	// --- typed interior affordances -----------------------------------------

	// Rebuild the affordances for the node the user is standing on. Each rendered
	// edge becomes one marker at the destination's floor, except the types listed in
	// HIDDEN_AFFORDANCES — see that set for why they're off and how to restore them.
	// In practice that leaves portal spots plus a trapped node's phase ring.
	buildNav(node: NavNode | null, panos: PanoEntry[]) {
		this.clearNav();
		if (!node) return;
		for (const edge of node.rendered) {
			if (HIDDEN_AFFORDANCES.has(edge.type)) continue;
			const floor = v3(panos[edge.to].position);
			floor.y -= HOTSPOT_FLOOR_DROP;
			const marker = makeNavMarker(edge, floor);
			marker.userData.bearing = edge.bearing;
			marker.userData.pulse = edge.type === "phase"; // trapped ghost never stops pulsing
			this.navGroup.add(marker);
		}
	}

	clearNav() {
		for (const m of this.navGroup.children) disposeGroup(m);
		this.navGroup.clear();
		this.hovered = null;
	}

	// The auto-home waypoint: a translucent affordance marking what a click would
	// do. The caller decides both WHETHER to draw one and WHERE — a through-geometry
	// hop is placed on the cursor ray rather than at its hidden destination. Rebuilt
	// only when the edge type changes (cheap); re-placed and screen-scaled every
	// frame so it tracks the pointer exactly like a live marker.
	showGhost(
		floorPos: Vector3,
		edge: { to: number; type: EdgeType; dy: number },
		camera: PerspectiveCamera,
		viewportHeight: number,
	) {
		if (this.ghostType !== edge.type) {
			for (const m of this.ghostGroup.children) disposeGroup(m);
			this.ghostGroup.clear();
			const marker = makeNavMarker(edge, floorPos, true);
			setGroupOpacity(marker, GHOST_OPACITY);
			this.ghostGroup.add(marker);
			this.ghostType = edge.type;
		}
		const marker = this.ghostGroup.children[0];
		if (!marker) return;
		marker.position.copy(floorPos);
		marker.position.y += 0.02; // lift a hair so a ghost never z-fights a coincident marker
		const d = camera.position.distanceTo(marker.position);
		marker.scale.setScalar(
			Math.max(
				1,
				screenScaleForDistance(d, NAV_TARGET_PX, camera.fov, viewportHeight, NAV_RING_OUTER),
			),
		);
		this.ghostGroup.visible = true;
	}

	hideGhost() {
		if (this.ghostGroup.visible) this.ghostGroup.visible = false;
	}

	// Gaze-contingent disclosure: affordances rest bright enough to FIND off-axis
	// and go full as the look swings toward them; phase ghosts + a dwell/never-
	// trapped pulse breathe. Each marker also gets a minimum on-screen size so a
	// far (or another-floor) point never shrinks away — near ones keep world size,
	// distant ones scale UP to stay visible + clickable.
	updateNav(
		camera: PerspectiveCamera,
		cameraAzimuth: number,
		now: number,
		viewportHeight: number,
	) {
		if (!this.navGroup.visible) return;
		const pulsing = now < this.pulseUntil;
		const breathe = 0.7 + 0.3 * Math.sin(now * 0.006);
		for (const marker of this.navGroup.children) {
			const bearing = marker.userData.bearing as number;
			const gaze = Math.max(
				0,
				1 - Math.abs(angleDelta(cameraAzimuth, bearing)) / NAV_GAZE_RAD,
			);
			let op = NAV_REST_OPACITY + (1 - NAV_REST_OPACITY) * gaze;
			if (marker === this.hovered) op = 1;
			else if (marker.userData.pulse || pulsing) op = Math.max(op, breathe);
			setGroupOpacity(marker, op);
			const d = camera.position.distanceTo(marker.position);
			marker.scale.setScalar(
				Math.max(
					1,
					screenScaleForDistance(d, NAV_TARGET_PX, camera.fov, viewportHeight, NAV_RING_OUTER),
				),
			);
		}
	}

	// Nearest affordance under the cursor (screen-space magnetism). Returns the
	// marker group; its userData carries the edge target + type.
	pickNav(
		clientX: number,
		clientY: number,
		camera: PerspectiveCamera,
		canvas: HTMLCanvasElement,
	): Object3D | null {
		return pickByScreen(clientX, clientY, this.navGroup, NAV_AIM_PX, camera, canvas);
	}

	setNavHover(marker: Object3D | null) {
		this.hovered = marker;
	}

	get hoveredNav(): Object3D | null {
		return this.hovered;
	}

	// Briefly pulse every exit (idle-dwell nudge, or a trapped node on arrival).
	pulseExits(now: number, ms = 1400) {
		this.pulseUntil = now + ms;
	}

	// --- level waypoints -----------------------------------------------------

	// Rebuild the per-floor monoliths. Each one stands at the capture its click
	// will actually land on, so the hover preview and the destination can never
	// disagree about where you are going.
	buildLevels(
		items: Array<{ level: number; index: number; floorPos: Vector3; up: boolean }>,
	) {
		this.clearLevels();
		for (const it of items) {
			const marker = makeLevelWaypoint(it.up);
			marker.position.copy(it.floorPos);
			marker.userData.level = it.level;
			marker.userData.to = it.index;
			marker.userData.up = it.up;
			this.levelGroup.add(marker);
		}
	}

	clearLevels() {
		for (const m of this.levelGroup.children) disposeGroup(m);
		this.levelGroup.clear();
		this.hoveredLevelMarker = null;
	}

	// Yaw each slab toward the eye so its face (and its up/down glyph) always reads,
	// breathe it gently so it announces itself as interactive, and hold a minimum
	// on-screen size so a floor across a large scene stays findable.
	updateLevels(camera: PerspectiveCamera, now: number, viewportHeight: number) {
		if (!this.levelGroup.visible) return;
		const breathe = LEVEL_REST_FACTOR + 0.12 * Math.sin(now * 0.0022);
		for (const marker of this.levelGroup.children) {
			marker.rotation.y = Math.atan2(
				camera.position.x - marker.position.x,
				camera.position.z - marker.position.z,
			);
			setRelativeOpacity(marker, marker === this.hoveredLevelMarker ? 1 : breathe);
			const d = camera.position.distanceTo(marker.position);
			marker.scale.setScalar(
				MathUtils.clamp(
					screenScaleForDistance(
						d,
						LEVEL_TARGET_PX,
						camera.fov,
						viewportHeight,
						LEVEL_HEIGHT * 0.5,
					),
					1,
					LEVEL_MAX_SCALE,
				),
			);
		}
	}

	// A monolith is a large, deliberate target drawn over the scene, so it's picked
	// by a real raycast rather than screen-space magnetism: clicking anywhere on the
	// slab counts, and its scaled world matrix means a bigger slab is a bigger hit.
	pickLevel(
		clientX: number,
		clientY: number,
		camera: PerspectiveCamera,
		canvas: HTMLCanvasElement,
	): Object3D | null {
		if (!this.levelGroup.visible || this.levelGroup.children.length === 0) return null;
		const rect = canvas.getBoundingClientRect();
		_ndc.set(
			((clientX - rect.left) / rect.width) * 2 - 1,
			-((clientY - rect.top) / rect.height) * 2 + 1,
		);
		camera.updateMatrixWorld();
		this.levelRay.setFromCamera(_ndc, camera);
		for (const h of this.levelRay.intersectObject(this.levelGroup, true)) {
			for (let o: Object3D | null = h.object; o; o = o.parent)
				if (o.userData.level !== undefined) return o;
		}
		return null;
	}

	setLevelHover(marker: Object3D | null) {
		this.hoveredLevelMarker = marker;
	}

	get hoveredLevel(): Object3D | null {
		return this.hoveredLevelMarker;
	}

	// --- sonar reveal --------------------------------------------------------

	// Build one x-ray dot per OTHER node, colored by its edge type from the current
	// node (far when it has no direct edge). Sized/positioned each frame.
	buildSonar(node: NavNode | null, panos: PanoEntry[], currentIndex: number) {
		for (const d of this.sonarGroup.children) disposeMesh(d as Mesh);
		this.sonarGroup.clear();
		if (!node) return;
		const typeOf = new Map<number, string>();
		for (const e of node.all) typeOf.set(e.to, e.type);
		for (let i = 0; i < panos.length; i++) {
			if (i === currentIndex) continue;
			const type = (typeOf.get(i) ?? "far") as keyof typeof NAV_COLORS;
			const dot = makeSonarDot(NAV_COLORS[type]);
			dot.position.fromArray(panos[i].position);
			dot.userData.to = i;
			dot.userData.type = type;
			dot.userData.name = panos[i].name ?? panos[i].id;
			this.sonarGroup.add(dot);
		}
	}

	startSonar(now: number, camera: PerspectiveCamera) {
		this.sonarStart = now;
		let reach = 1;
		for (const d of this.sonarGroup.children)
			reach = Math.max(reach, camera.position.distanceTo(d.position));
		this.sonarReach = reach * 1.05 + 1;
		this.sonarGroup.visible = true;
	}

	get sonarActive(): boolean {
		return this.sonarGroup.visible;
	}

	// Expanding reveal front sweeps outward from the camera; dots light as it
	// passes them, then everything fades. Returns false once finished.
	updateSonar(now: number, camera: PerspectiveCamera, viewportHeight: number): boolean {
		if (!this.sonarGroup.visible) return false;
		const t = (now - this.sonarStart) / SONAR_DURATION;
		if (t >= 1) {
			this.hideSonar();
			return false;
		}
		const front = Math.min(1, t / 0.55) * this.sonarReach;
		const globalFade = t < 0.8 ? 1 : 1 - (t - 0.8) / 0.2;
		for (const dot of this.sonarGroup.children) {
			const d = camera.position.distanceTo(dot.position);
			const reveal = Math.max(0, Math.min(1, (front - d) / 2 + 0.5));
			const mat = (dot as Mesh).material as MeshBasicMaterial;
			mat.opacity = reveal * globalFade * 0.95;
			dot.scale.setScalar(
				screenScaleForDistance(
					Math.max(0.2, d),
					6,
					camera.fov,
					viewportHeight,
					1,
				),
			);
		}
		return true;
	}

	hideSonar() {
		this.sonarGroup.visible = false;
	}

	// Screen placement of the nearest sonar dots, for the engine's HTML labels.
	sonarLabelTargets(
		camera: PerspectiveCamera,
		canvas: HTMLCanvasElement,
		max: number,
	): Array<{ x: number; y: number; name: string; type: string }> {
		if (!this.sonarGroup.visible) return [];
		const rect = canvas.getBoundingClientRect();
		const out: Array<{ x: number; y: number; name: string; type: string; d: number }> = [];
		const p = new Vector3();
		for (const dot of this.sonarGroup.children) {
			const mat = (dot as Mesh).material as MeshBasicMaterial;
			if (mat.opacity < 0.25) continue;
			dot.getWorldPosition(p);
			const d = camera.position.distanceTo(p);
			p.project(camera);
			if (p.z > 1) continue;
			out.push({
				x: rect.left + (p.x * 0.5 + 0.5) * rect.width,
				y: rect.top + (-p.y * 0.5 + 0.5) * rect.height,
				name: dot.userData.name as string,
				type: dot.userData.type as string,
				d,
			});
		}
		out.sort((a, b) => a.d - b.d);
		return out.slice(0, max);
	}

	// --- peek "you are here" -------------------------------------------------

	positionYouMarker(p: Vector3) {
		const floorY = p.y - CAPTURE_EYE_HEIGHT;
		this.you.sphere.position.copy(p);
		this.you.ring.position.set(p.x, floorY, p.z);
		this.you.line.geometry.setFromPoints([new Vector3(p.x, floorY, p.z), p.clone()]);
	}

	clear() {
		this.clearNav();
		for (const d of this.sonarGroup.children) disposeMesh(d as Mesh);
		this.sonarGroup.clear();
		this.sonarGroup.visible = false;
		for (const m of this.ghostGroup.children) disposeGroup(m);
		this.ghostGroup.clear();
		this.ghostGroup.visible = false;
		this.ghostType = null;
		this.clearLevels();
		this.levelGroup.visible = false;
		this.you.group.visible = false;
		this.pulseUntil = 0;
	}

	dispose() {
		this.clear();
		this.scene.remove(
			this.navGroup,
			this.sonarGroup,
			this.ghostGroup,
			this.levelGroup,
			this.you.group,
		);
	}

	private sizeYouMarker(sceneMaxDim: number) {
		const r = Math.max(0.05, sceneMaxDim * 0.014);
		this.you.sphere.geometry.dispose();
		this.you.sphere.geometry = new SphereGeometry(r, 24, 16);
		this.you.ring.geometry.dispose();
		this.you.ring.geometry = new RingGeometry(r * 1.6, r * 2.2, 40);
	}
}

// Scale every tagged part of a marker by one factor, preserving the relative
// weights it was designed with (low-fill body, crisp edges, bright glyph) instead
// of flattening them to a single alpha the way setGroupOpacity does.
function setRelativeOpacity(marker: Object3D, factor: number) {
	marker.traverse((o) => {
		const base = o.userData.baseOpacity as number | undefined;
		if (base === undefined) return;
		const m = (o as Mesh).material as Material | Material[] | undefined;
		if (!m) return;
		for (const mat of Array.isArray(m) ? m : [m])
			(mat as MeshBasicMaterial).opacity = base * factor;
	});
}

// Set a uniform opacity across every material in an affordance group.
function setGroupOpacity(marker: Object3D, opacity: number) {
	marker.traverse((o) => {
		const m = (o as Mesh).material as Material | Material[] | undefined;
		if (!m) return;
		for (const mat of Array.isArray(m) ? m : [m])
			(mat as MeshBasicMaterial).opacity = opacity;
	});
}

function disposeMesh(m: Mesh) {
	m.geometry?.dispose();
	const mats = Array.isArray(m.material) ? m.material : [m.material];
	for (const mat of mats) mat?.dispose();
}

function disposeGroup(g: Object3D) {
	g.traverse((o) => {
		const m = o as Mesh;
		if (m.isMesh || (o as { isLine?: boolean }).isLine) disposeMesh(m);
	});
}
