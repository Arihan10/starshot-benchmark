import {
	DoubleSide,
	Group,
	Mesh,
	MeshBasicMaterial,
	type Object3D,
	type PerspectiveCamera,
	Raycaster,
	RingGeometry,
	type Scene,
	SphereGeometry,
	Vector3,
} from "three";
import {
	ANCHOR_RING_INNER,
	ANCHOR_RING_OCCLUDED_COLOR,
	ANCHOR_RING_OCCLUDED_OPACITY,
	ANCHOR_RING_OCCLUDED_SCALE,
	ANCHOR_RING_OPACITY,
	ANCHOR_RING_OUTER,
	AUTO_AIM_PX,
	CAPTURE_EYE_HEIGHT,
	ENTRY_TARGET_PX,
	HOTSPOT_FLOOR_DROP,
	HOTSPOT_MAX_OCCLUDED,
	HOTSPOT_OCCLUDE_EPS,
	HOTSPOT_REACH,
	hotspotScaleForDistance,
	makeDisc,
	makeYouMarker,
	pickByScreen,
	type YouMarker,
} from "./markers";
import type { PanoEntry } from "./panoTextures";

const v3 = (a: [number, number, number]) => new Vector3().fromArray(a);

// The interior navigation overlay: white anchor rings on every capture point,
// warm-gold rings layered onto the nearest behind-wall anchors, the overview
// entry discs, and the "you are here" pin used while locating. Owns the marker
// groups + their shared geometry/materials; the engine drives group visibility
// per mode and feeds in the live panos / camera.
export class MarkerLayer {
	readonly hotspotGroup = new Group();
	readonly entryGroup = new Group();
	// Every anchor as a small white ring laid flat on the floor. Depth-tested
	// (depthTest defaults on) so scene geometry obstructs it, and never
	// screen-scaled so it keeps a world-fixed size — smaller far, larger near,
	// like a real object. One shared geometry + material across all anchors.
	readonly anchorRingGroup = new Group();
	readonly you: YouMarker = makeYouMarker();

	private readonly anchorRingGeo = new RingGeometry(
		ANCHOR_RING_INNER,
		ANCHOR_RING_OUTER,
		40,
	);
	private readonly anchorRingMat = new MeshBasicMaterial({
		color: 0xffffff,
		transparent: true,
		opacity: ANCHOR_RING_OPACITY,
		side: DoubleSide,
		depthWrite: false,
	});
	// The X closest obstructed anchors reuse the ring geometry in warm gold, drawn
	// over everything (depthTest off) so they show through walls as reachable.
	private readonly anchorRingOccludedMat = new MeshBasicMaterial({
		color: ANCHOR_RING_OCCLUDED_COLOR,
		transparent: true,
		opacity: ANCHOR_RING_OCCLUDED_OPACITY,
		side: DoubleSide,
		depthWrite: false,
		depthTest: false,
	});
	// Full-opacity twins of the two ring materials, swapped onto the single ring the
	// cursor is over (setRingHover). Rings share the faint base materials, so this
	// lights just the hovered one — matching its color + depthTest, only at opacity 1.
	private readonly anchorRingHoverMat = new MeshBasicMaterial({
		color: 0xffffff,
		transparent: true,
		opacity: 1,
		side: DoubleSide,
		depthWrite: false,
	});
	private readonly anchorRingOccludedHoverMat = new MeshBasicMaterial({
		color: ANCHOR_RING_OCCLUDED_COLOR,
		transparent: true,
		opacity: 1,
		side: DoubleSide,
		depthWrite: false,
		depthTest: false,
	});
	private hovered: Mesh | null = null;
	private readonly occluder = new Raycaster();

	constructor(private readonly scene: Scene) {
		scene.add(this.hotspotGroup, this.entryGroup, this.anchorRingGroup);
		scene.add(this.you.group);
	}

	get hoveredRing(): Mesh | null {
		return this.hovered;
	}

	// Per-scene build: entry discs, white anchor rings, and the "you" marker sized
	// to the scene. Hotspots are rebuilt separately (they depend on position).
	build(panos: PanoEntry[], sceneMaxDim: number) {
		this.sizeYouMarker(sceneMaxDim);
		this.buildEntryMarkers(panos);
		this.buildAnchorRings(panos);
	}

	// The X closest obstructed (behind-wall) anchors. Same ring as the white anchor
	// rings but warm gold, larger, and drawn over everything so they read as
	// reachable through walls. Rebuilt on travel (occlusion is per position).
	rebuildHotspots(
		panos: PanoEntry[],
		currentIndex: number,
		proxyGroup: Group | null,
		projectionMode: boolean,
	) {
		this.hotspotGroup.clear();
		if (currentIndex < 0) return;
		const cur = panos[currentIndex];
		let nOccluded = 0;
		for (const i of this.neighborsByDistance(
			panos,
			currentIndex,
			projectionMode,
		)) {
			if (!this.anchorOccluded(proxyGroup, cur.position, panos[i].position))
				continue;
			const ring = new Mesh(this.anchorRingGeo, this.anchorRingOccludedMat);
			ring.rotation.x = -Math.PI / 2;
			ring.scale.setScalar(ANCHOR_RING_OCCLUDED_SCALE);
			ring.position.fromArray(panos[i].position);
			ring.position.y -= HOTSPOT_FLOOR_DROP;
			ring.userData.targetIndex = i;
			ring.userData.occluded = true;
			this.hotspotGroup.add(ring);
			if (++nOccluded >= HOTSPOT_MAX_OCCLUDED) break;
		}
	}

	// Nearest clickable anchor marker under the cursor (screen-space magnetism): a
	// gold obstructed marker (drawn over everything, always reachable), else a white
	// ring that scene geometry isn't hiding — an occluded white ring is invisible,
	// so clicking the wall in front of it must NOT teleport through. Shared by hover
	// + click, so what lights up is exactly what a click travels to.
	pickAnchorMarker(
		clientX: number,
		clientY: number,
		camera: PerspectiveCamera,
		canvas: HTMLCanvasElement,
		panos: PanoEntry[],
		currentIndex: number,
		proxyGroup: Group | null,
	): Object3D | null {
		const occluded = pickByScreen(
			clientX,
			clientY,
			this.hotspotGroup,
			AUTO_AIM_PX,
			camera,
			canvas,
		);
		if (occluded) return occluded;
		const ring = pickByScreen(
			clientX,
			clientY,
			this.anchorRingGroup,
			AUTO_AIM_PX,
			camera,
			canvas,
		);
		if (!ring) return null;
		const i = ring.userData.targetIndex as number;
		if (i === currentIndex) return null;
		const cur = panos[currentIndex].position;
		return this.anchorOccluded(proxyGroup, cur, panos[i].position)
			? null
			: ring;
	}

	// Light the one ring the cursor is over to full opacity, reverting the one it
	// left. Rings share faint base materials, so we swap just this mesh's material
	// to its bolder twin (same color + depthTest, opacity 1).
	setRingHover(mesh: Mesh | null) {
		if (mesh === this.hovered) return;
		if (this.hovered) {
			this.hovered.material = this.hovered.userData.occluded
				? this.anchorRingOccludedMat
				: this.anchorRingMat;
		}
		this.hovered = mesh;
		if (mesh) {
			mesh.material = mesh.userData.occluded
				? this.anchorRingOccludedHoverMat
				: this.anchorRingHoverMat;
		}
	}

	// Floor directly beneath the user (panos sit at eye height), not the global
	// scene minimum — so the base lands on the level you're standing on.
	positionYouMarker(p: Vector3) {
		const floorY = p.y - CAPTURE_EYE_HEIGHT;
		this.you.sphere.position.copy(p);
		this.you.ring.position.set(p.x, floorY, p.z);
		this.you.line.geometry.setFromPoints([
			new Vector3(p.x, floorY, p.z),
			p.clone(),
		]);
	}

	// Overview entry discs render at a constant on-screen size + pulse; the
	// interior anchor rings (white + gold) are world-fixed, so they're left be.
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
				hotspotScaleForDistance(
					d,
					ENTRY_TARGET_PX,
					camera.fov,
					viewportHeight,
				) * (hovered ? 1.35 : 1),
			);
			const disc = spot.children[0] as Mesh;
			const ring = spot.children[1] as Mesh;
			ring.scale.setScalar(pulse);
			(disc.material as MeshBasicMaterial).opacity = hovered ? 0.9 : 0.55;
			(ring.material as MeshBasicMaterial).opacity = hovered ? 1.0 : 0.85;
		}
	}

	// Empty the marker groups for a scene swap (the discs/rings are cheap clones
	// over shared geometry; the shared geo/materials live until dispose()).
	clear() {
		this.hotspotGroup.clear();
		this.entryGroup.clear();
		this.anchorRingGroup.clear();
		this.hovered = null;
		this.you.group.visible = false;
	}

	dispose() {
		this.scene.remove(this.anchorRingGroup);
		this.anchorRingGeo.dispose();
		this.anchorRingMat.dispose();
		this.anchorRingOccludedMat.dispose();
		this.anchorRingHoverMat.dispose();
		this.anchorRingOccludedHoverMat.dispose();
	}

	// One white ring per anchor, laid flat on the floor — built once per scene.
	// Shares anchorRingGeo/anchorRingMat so it's depth-tested (geometry obstructs
	// it) and world-fixed-size; it shrinks far / grows near like a real object.
	private buildAnchorRings(panos: PanoEntry[]) {
		this.anchorRingGroup.clear();
		for (let i = 0; i < panos.length; i++) {
			const ring = new Mesh(this.anchorRingGeo, this.anchorRingMat);
			ring.rotation.x = -Math.PI / 2;
			ring.position.fromArray(panos[i].position);
			ring.position.y -= HOTSPOT_FLOOR_DROP;
			ring.userData.targetIndex = i;
			ring.userData.occluded = false;
			this.anchorRingGroup.add(ring);
		}
	}

	private buildEntryMarkers(panos: PanoEntry[]) {
		this.entryGroup.clear();
		for (let i = 0; i < panos.length; i++) {
			const spot = makeDisc(i, 0x9ad4ff, 0x4a8fd8);
			spot.position.fromArray(panos[i].position);
			spot.position.y -= HOTSPOT_FLOOR_DROP;
			this.entryGroup.add(spot);
		}
	}

	private sizeYouMarker(sceneMaxDim: number) {
		const r = Math.max(0.05, sceneMaxDim * 0.014);
		this.you.sphere.geometry.dispose();
		this.you.sphere.geometry = new SphereGeometry(r, 24, 16);
		this.you.ring.geometry.dispose();
		this.you.ring.geometry = new RingGeometry(r * 1.6, r * 2.2, 40);
	}

	// Is the straight line between two capture points blocked by the proxy? Our
	// "behind a wall/floor" test, trimmed at both ends so a hugged wall doesn't
	// read as occlusion.
	private anchorOccluded(
		proxyGroup: Group | null,
		fromPos: [number, number, number],
		toPos: [number, number, number],
	): boolean {
		if (!proxyGroup) return false;
		const from = v3(fromPos);
		const d = v3(toPos).sub(from);
		const dist = d.length();
		if (dist < 1e-3) return false;
		d.divideScalar(dist);
		this.occluder.set(from, d);
		this.occluder.near = HOTSPOT_OCCLUDE_EPS;
		this.occluder.far = dist - HOTSPOT_OCCLUDE_EPS;
		if (this.occluder.far <= this.occluder.near) return false;
		return this.occluder.intersectObject(proxyGroup, true).length > 0;
	}

	private neighborsByDistance(
		panos: PanoEntry[],
		currentIndex: number,
		projectionMode: boolean,
	): number[] {
		const cur = v3(panos[currentIndex].position);
		const out: Array<[number, number]> = [];
		for (let i = 0; i < panos.length; i++) {
			if (i === currentIndex) continue;
			const d2 = cur.distanceToSquared(v3(panos[i].position));
			if (projectionMode && d2 > HOTSPOT_REACH * HOTSPOT_REACH) continue;
			out.push([i, d2]);
		}
		out.sort((a, b) => a[1] - b[1]);
		return out.map((o) => o[0]);
	}
}
