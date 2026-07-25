import {
	Group,
	type Material,
	Mesh,
	type MeshBasicMaterial,
	type Object3D,
	type PerspectiveCamera,
	RingGeometry,
	type Scene,
	SphereGeometry,
	Vector3,
} from "three";
import {
	CAPTURE_EYE_HEIGHT,
	ENTRY_TARGET_PX,
	HOTSPOT_FLOOR_DROP,
	makeDisc,
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
import { angleDelta, type NavNode } from "./navGraph";
import type { PanoEntry } from "./panoTextures";

const v3 = (a: [number, number, number]) => new Vector3().fromArray(a);

// The interior navigation overlay. Renders the CURRENT node's typed affordances
// (walk pucks, portal/vertical chevrons, phase ghost rings), fades them by gaze
// bearing, drives the sonar reveal of every node, and keeps the overview "enter"
// discs + the peek "you are here" pin. The engine flips group visibility per mode
// and feeds in the live nav graph / camera each frame.
export class MarkerLayer {
	readonly entryGroup = new Group(); // overview: one disc per capture ("enter here")
	readonly navGroup = new Group(); // interior: the current node's typed affordances
	readonly sonarGroup = new Group(); // interior: x-ray reveal of every node
	readonly you: YouMarker = makeYouMarker();

	private hovered: Object3D | null = null;
	private pulseUntil = 0; // dwell/never-trapped: briefly boost every affordance
	private sonarStart = 0;
	private sonarReach = 1;

	constructor(private readonly scene: Scene) {
		scene.add(this.entryGroup, this.navGroup, this.sonarGroup, this.you.group);
	}

	// Per-scene build: the overview entry discs + the you-marker sized to the scene.
	build(panos: PanoEntry[], sceneMaxDim: number) {
		this.sizeYouMarker(sceneMaxDim);
		this.entryGroup.clear();
		for (let i = 0; i < panos.length; i++) {
			const spot = makeDisc(i, 0x9ad4ff, 0x4a8fd8);
			spot.position.fromArray(panos[i].position);
			spot.position.y -= HOTSPOT_FLOOR_DROP;
			this.entryGroup.add(spot);
		}
	}

	// --- typed interior affordances -----------------------------------------

	// Rebuild the affordances for the node the user is standing on. Each rendered
	// edge (walk/portal/vertical, plus phase when the node is trapped) becomes one
	// marker at the destination's floor.
	buildNav(node: NavNode | null, panos: PanoEntry[]) {
		this.clearNav();
		if (!node) return;
		for (const edge of node.rendered) {
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

	// Overview entry discs render at a constant on-screen size + pulse.
	updateEntryDiscs(
		camera: PerspectiveCamera,
		viewportHeight: number,
		hoveredEntryIndex: number,
		time: number,
	) {
		if (!this.entryGroup.visible) return;
		const pulse = 1 + 0.07 * Math.sin(time * 0.004);
		for (const spot of this.entryGroup.children) {
			const hovered = spot.userData.targetIndex === hoveredEntryIndex;
			const d = camera.position.distanceTo(spot.position);
			spot.scale.setScalar(
				screenScaleForDistance(d, ENTRY_TARGET_PX, camera.fov, viewportHeight) *
					(hovered ? 1.35 : 1),
			);
			const disc = spot.children[0] as Mesh;
			const ring = spot.children[1] as Mesh;
			ring.scale.setScalar(pulse);
			(disc.material as MeshBasicMaterial).opacity = hovered ? 0.9 : 0.55;
			(ring.material as MeshBasicMaterial).opacity = hovered ? 1.0 : 0.85;
		}
	}

	clear() {
		this.clearNav();
		for (const d of this.sonarGroup.children) disposeMesh(d as Mesh);
		this.sonarGroup.clear();
		this.sonarGroup.visible = false;
		this.entryGroup.clear();
		this.you.group.visible = false;
		this.pulseUntil = 0;
	}

	dispose() {
		this.clear();
		this.scene.remove(this.entryGroup, this.navGroup, this.sonarGroup, this.you.group);
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
