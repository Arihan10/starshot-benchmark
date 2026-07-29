import {
	type BufferAttribute,
	Group,
	LineBasicMaterial,
	type Material,
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
	FLOOR_ARROW_BOB,
	FLOOR_ARROW_GLOW,
	FLOOR_ARROW_GLOW_TAU,
	FLOOR_ARROW_HOVER_LIFT,
	FLOOR_ARROW_PULSE_RATE,
	FLOOR_ARROW_RATE,
	HOTSPOT_FLOOR_DROP,
	makeFloorArrow,
	makeGhostTether,
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

// --- the ghost tether ---------------------------------------------------------
//
// The ghost stands on the anchor a click actually lands on, which is the only
// placement that can't lie about the destination — but the anchor is behind
// geometry and usually off to one side, so on its own it is a second mark on
// screen with nothing connecting it to the pointer that summoned it. The tether
// is that connection: cursor at one end, destination at the other, so the pair
// reads as one sentence — click HERE, arrive THERE.
//
// It doubles as the off-screen cue. When the anchor falls outside the frustum the
// line still leaves the frame pointing at it, which is why no edge indicator is
// needed (see makeGhostTether for why it survives being partly off-camera).
//
// Below TETHER_MIN_DIST the ghost is already sitting on the cursor and the line
// would be a nub inside the ring, so it's dropped.
const TETHER_MIN_DIST = 0.4;

// --- suppressed STANDING affordances -----------------------------------------
//
// An edge type in this set renders no standing marker in buildNav(). It says
// nothing about the on-demand waypoint the cursor grows while homing — whether to
// draw that is the engine's call (updateCursorRing / destinationFloor).
//
// TO RE-ENABLE a type, delete its entry — nothing else needs touching. The
// builders in markers.ts still handle all five types, so it comes straight back.
//
// WHY THESE ARE OFF:
//   walk (blue) — a walk edge is BY DEFINITION a spot you can already SEE from
//     where you stand, so its puck spends attention on what the view already told
//     you. With navGraph's nearest-3 guarantee most nodes drew several at once and
//     the floor filled with near-identical blue rings.
//   portal (orange) — a STANDING marker for every destination behind a wall fills
//     the room with rings for places you cannot see and are not asking about. The
//     engine draws ONE on demand instead, on the destination the cursor currently
//     resolves to, tethered back to the cursor so it still answers "what happens if
//     I click HERE" without having to lie about where that is.
// That leaves phase (violet), and only on a sealed node with no other way out.
//
// NOTHING BECOMES UNREACHABLE. Suppressed destinations keep every other route in:
// clicking the floor still auto-homes onto them (engine's clickAnywhere →
// autoHomeTarget scores every pano and ignores edge type), the surface cursor
// still tints by type while one is the homing target, and they stay in the exits
// panel, the minimap and the sonar ping. This hides scene geometry only — the nav
// graph, routing costs and traversal FX are all untouched.
//   vertical (green) — still very much ON, just not from here: floor changes are
//     drawn by the dedicated arrow layer below, which places them in your view on
//     arrival instead of at the destination (a point on another storey, i.e. the
//     one place you are guaranteed not to be looking).
const HIDDEN_AFFORDANCES: ReadonlySet<EdgeType> = new Set<EdgeType>([
	"walk",
	"portal",
	"vertical",
]);

const v3 = (a: [number, number, number]) => new Vector3().fromArray(a);
const _ndc = new Vector2();

// The interior navigation overlay. Renders the CURRENT node's typed affordances
// (every type except those in HIDDEN_AFFORDANCES), fades them by gaze bearing,
// drives the sonar reveal of every node, previews the auto-home destination, and
// keeps the peek "you are here" pin. The engine flips
// group visibility per mode and feeds in the live nav graph / camera each frame.
export class MarkerLayer {
	readonly navGroup = new Group(); // interior: the current node's typed affordances
	readonly sonarGroup = new Group(); // interior: x-ray reveal of every node
	readonly ghostGroup = new Group(); // interior: the auto-home destination preview
	readonly arrowGroup = new Group(); // interior: the in-view floor-change arrows
	readonly you: YouMarker = makeYouMarker();

	private hovered: Object3D | null = null;
	private hoveredArrowMarker: Object3D | null = null;
	private readonly arrowRay = new Raycaster();
	private lastArrowTick = 0;
	private ghostType: EdgeType | null = null; // rebuild key: the ghost's current edge type
	// The ghost's ring/chevron, held rather than read back off ghostGroup.children:
	// the tether shares that group and is never rebuilt, so an index would be
	// whichever of the two happened to be added first.
	private ghostMarker: Object3D | null = null;
	private readonly ghostTether = makeGhostTether();
	private pulseUntil = 0; // dwell/never-trapped: briefly boost every affordance
	private sonarStart = 0;
	private sonarReach = 1;

	constructor(private readonly scene: Scene) {
		this.ghostGroup.add(this.ghostTether);
		scene.add(
			this.navGroup,
			this.sonarGroup,
			this.ghostGroup,
			this.arrowGroup,
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
	// In practice that leaves a trapped node's phase ring.
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
	// do, standing ON the destination anchor — `floorPos` is that capture's floor
	// point, so the marker and the place you land are the same spot and cannot drift
	// apart. `from` is the surface point under the cursor; the tether runs between
	// the two, which is what keeps a marker you may not be pointing at legible as an
	// answer to where you ARE pointing. Pass null for no tether.
	//
	// Rebuilt only when the edge type changes (cheap); re-placed and screen-scaled
	// every frame, so the ring holds its on-screen size as the destination's distance
	// changes under a moving cursor.
	showGhost(
		floorPos: Vector3,
		edge: { to: number; type: EdgeType; dy: number },
		camera: PerspectiveCamera,
		viewportHeight: number,
		from: Vector3 | null = null,
	) {
		if (this.ghostType !== edge.type) {
			this.clearGhostMarker();
			const marker = makeNavMarker(edge, floorPos, true);
			setGroupOpacity(marker, GHOST_OPACITY);
			this.ghostGroup.add(marker);
			this.ghostMarker = marker;
			(this.ghostTether.material as LineBasicMaterial).color.setHex(
				NAV_COLORS[edge.type],
			);
			this.ghostType = edge.type;
		}
		const marker = this.ghostMarker;
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
		const tethered =
			!!from && from.distanceTo(marker.position) > TETHER_MIN_DIST;
		if (tethered && from) {
			const pts = this.ghostTether.geometry.getAttribute(
				"position",
			) as BufferAttribute;
			pts.setXYZ(0, from.x, from.y, from.z);
			pts.setXYZ(1, marker.position.x, marker.position.y, marker.position.z);
			pts.needsUpdate = true;
		}
		this.ghostTether.visible = tethered;
		this.ghostGroup.visible = true;
	}

	hideGhost() {
		if (this.ghostGroup.visible) this.ghostGroup.visible = false;
	}

	// Drop the ghost's marker while KEEPING the tether, which lives in the same group
	// but is built once and reused.
	private clearGhostMarker() {
		if (!this.ghostMarker) return;
		disposeGroup(this.ghostMarker);
		this.ghostGroup.remove(this.ghostMarker);
		this.ghostMarker = null;
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

	// --- floor arrows --------------------------------------------------------

	// Rebuild the pair of floor arrows for the node just arrived at. `pos` is
	// already placed by the engine, on your arrival heading — see refreshFloorArrows
	// for why that, and not the destination, is where these live.
	buildFloorArrows(items: Array<{ index: number; up: boolean; pos: Vector3 }>) {
		this.clearFloorArrows();
		for (const it of items) {
			const marker = makeFloorArrow(it.up);
			marker.position.copy(it.pos);
			marker.userData.to = it.index;
			marker.userData.up = it.up;
			marker.userData.baseY = it.pos.y;
			// The up and down arrows drift out of step with each other — they are
			// separate objects and reading as such costs nothing, whereas the two
			// halves of ONE arrow must never do that.
			marker.userData.phase = it.up ? 0 : Math.PI / 2;
			this.arrowGroup.add(marker);
		}
	}

	clearFloorArrows() {
		for (const m of this.arrowGroup.children) disposeGroup(m);
		this.arrowGroup.clear();
		this.hoveredArrowMarker = null;
	}

	// Drift each arrow as ONE rigid object. The chevrons inside it never move
	// relative to each other — that was what made a single arrow read as two.
	updateFloorArrows(now: number) {
		if (!this.arrowGroup.visible) return;
		// Clamped so a backgrounded tab returning after seconds eases rather than
		// teleporting; seeded so the first frame is a sane step, not a jump.
		const dt = this.lastArrowTick ? Math.min(100, now - this.lastArrowTick) : 16;
		this.lastArrowTick = now;
		const k = 1 - Math.exp(-dt / FLOOR_ARROW_GLOW_TAU);
		for (const marker of this.arrowGroup.children) {
			const baseY = marker.userData.baseY as number;
			const phase = marker.userData.phase as number;
			marker.position.y =
				baseY + Math.sin(now * FLOOR_ARROW_RATE + phase) * FLOOR_ARROW_BOB;
			// Hover level EASES toward its target rather than jumping, so the glow
			// arrives and leaves instead of switching. Exponential approach, driven by
			// real elapsed time, so it looks the same at any frame rate.
			const target = marker === this.hoveredArrowMarker ? 1 : 0;
			const level =
				((marker.userData.glow as number) ?? 0) +
				(target - ((marker.userData.glow as number) ?? 0)) * k;
			marker.userData.glow = level;
			// One scalar drives the whole hover response, so the rim and the body
			// swell together instead of beating against each other.
			const glow =
				level * (0.72 + 0.28 * Math.sin(now * FLOOR_ARROW_PULSE_RATE));
			for (const part of marker.children) {
				const base = part.userData.baseOpacity as number;
				const mat = (part as Mesh).material as MeshBasicMaterial;
				mat.opacity = part.userData.halo
					? FLOOR_ARROW_GLOW * glow
					: Math.min(1, base * (1 + FLOOR_ARROW_HOVER_LIFT * glow));
			}
		}
	}

	// Picked by a REAL raycast first, then by screen magnetism as a fallback.
	//
	// Magnetism alone measured from the arrow's centre, so anywhere past its radius
	// stopped counting as "on the arrow" even though the arrow was plainly under the
	// cursor — and the proxy surface behind it took over instead. An arrow is drawn
	// over everything, so if the ray meets its geometry you ARE pointing at it, and
	// it wins. The magnetism stays underneath to keep near-misses forgiving.
	pickFloorArrow(
		clientX: number,
		clientY: number,
		camera: PerspectiveCamera,
		canvas: HTMLCanvasElement,
	): Object3D | null {
		if (!this.arrowGroup.visible || this.arrowGroup.children.length === 0)
			return null;
		const rect = canvas.getBoundingClientRect();
		_ndc.set(
			((clientX - rect.left) / rect.width) * 2 - 1,
			-((clientY - rect.top) / rect.height) * 2 + 1,
		);
		camera.updateMatrixWorld();
		this.arrowRay.setFromCamera(_ndc, camera);
		for (const h of this.arrowRay.intersectObject(this.arrowGroup, true)) {
			for (let o: Object3D | null = h.object; o; o = o.parent) {
				if (o.parent === this.arrowGroup) return o;
			}
		}
		return pickByScreen(clientX, clientY, this.arrowGroup, NAV_AIM_PX, camera, canvas);
	}

	setArrowHover(marker: Object3D | null) {
		this.hoveredArrowMarker = marker;
	}

	get hoveredArrow(): Object3D | null {
		return this.hoveredArrowMarker;
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
		this.clearGhostMarker();
		this.ghostGroup.visible = false;
		this.ghostTether.visible = false;
		this.ghostType = null;
		this.clearFloorArrows();
		this.arrowGroup.visible = false;
		this.you.group.visible = false;
		this.pulseUntil = 0;
	}

	dispose() {
		this.clear();
		// clear() keeps the tether alive for the next scene; teardown owns it.
		disposeGroup(this.ghostTether);
		this.ghostGroup.remove(this.ghostTether);
		this.scene.remove(
			this.navGroup,
			this.sonarGroup,
			this.ghostGroup,
			this.arrowGroup,
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
