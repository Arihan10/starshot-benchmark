import {
	Group,
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
	FLOOR_ARROW_GLOW,
	FLOOR_ARROW_GLOW_TAU,
	FLOOR_ARROW_HOVER_LIFT,
	FLOOR_ARROW_PULSE_RATE,
	FLOOR_ARROW_RATE,
	makeFloorArrow,
	makeNavMarker,
	makeSonarDot,
	makeYouMarker,
	NAV_AIM_PX,
	NAV_COLORS,
	NAV_GAZE_RAD,
	NAV_REST_OPACITY,
	NAV_TARGET_PX,
	pickByScreen,
	screenScaleForDistance,
	SONAR_DURATION,
	type YouMarker,
} from "./markers";
import { angleDelta, type EdgeType, type NavNode } from "./navGraph";
import type { PanoEntry } from "./panoTextures";
import { DEFAULT_METRICS, type NavMetrics } from "./scale";

const GHOST_OPACITY = 0.4;

const HIDDEN_AFFORDANCES: ReadonlySet<EdgeType> = new Set<EdgeType>([
	"walk",
	"portal",
	"vertical",
]);

const v3 = (a: [number, number, number]) => new Vector3().fromArray(a);
const _ndc = new Vector2();

export class MarkerLayer {
	readonly navGroup = new Group();
	readonly sonarGroup = new Group();
	readonly ghostGroup = new Group();
	readonly arrowGroup = new Group();
	readonly you: YouMarker = makeYouMarker();

	private metrics: NavMetrics = DEFAULT_METRICS;

	private hovered: Object3D | null = null;
	private hoveredArrowMarker: Object3D | null = null;
	private readonly arrowRay = new Raycaster();
	private lastArrowTick = 0;
	private ghostType: EdgeType | null = null;
	private ghostMarker: Object3D | null = null;
	private pulseUntil = 0;
	private sonarStart = 0;
	private sonarReach = 1;

	constructor(private readonly scene: Scene) {
		scene.add(
			this.navGroup,
			this.sonarGroup,
			this.ghostGroup,
			this.arrowGroup,
			this.you.group,
		);
	}

	build(sceneMaxDim: number, metrics: NavMetrics) {
		this.metrics = metrics;
		this.sizeYouMarker(sceneMaxDim);
	}

	buildNav(node: NavNode | null, panos: PanoEntry[]) {
		this.clearNav();
		if (!node) return;
		for (const edge of node.rendered) {
			if (HIDDEN_AFFORDANCES.has(edge.type)) continue;
			const floor = v3(panos[edge.to].position);
			floor.y -= this.metrics.floorDrop;
			const marker = makeNavMarker(edge, floor, this.metrics);
			marker.userData.bearing = edge.bearing;
			marker.userData.pulse = edge.type === "phase";
			this.navGroup.add(marker);
		}
	}

	clearNav() {
		for (const m of this.navGroup.children) disposeGroup(m);
		this.navGroup.clear();
		this.hovered = null;
	}

	showGhost(
		floorPos: Vector3,
		edge: { to: number; type: EdgeType; dy: number },
		camera: PerspectiveCamera,
		viewportHeight: number,
	) {
		if (this.ghostType !== edge.type) {
			this.clearGhostMarker();
			const marker = makeNavMarker(edge, floorPos, this.metrics, true);
			setGroupOpacity(marker, GHOST_OPACITY);
			this.ghostGroup.add(marker);
			this.ghostMarker = marker;
			this.ghostType = edge.type;
		}
		const marker = this.ghostMarker;
		if (!marker) return;
		marker.position.copy(floorPos);
		marker.position.y += this.metrics.ghostLift;
		const d = camera.position.distanceTo(marker.position);
		marker.scale.setScalar(
			Math.max(
				1,
				screenScaleForDistance(d, NAV_TARGET_PX, camera.fov, viewportHeight, this.metrics.ringOuter),
			),
		);
		this.ghostGroup.visible = true;
	}

	hideGhost() {
		if (this.ghostGroup.visible) this.ghostGroup.visible = false;
	}

	private clearGhostMarker() {
		if (!this.ghostMarker) return;
		disposeGroup(this.ghostMarker);
		this.ghostGroup.remove(this.ghostMarker);
		this.ghostMarker = null;
	}

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
					screenScaleForDistance(d, NAV_TARGET_PX, camera.fov, viewportHeight, this.metrics.ringOuter),
				),
			);
		}
	}

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

	pulseExits(now: number, ms = 1400) {
		this.pulseUntil = now + ms;
	}

	buildFloorArrows(items: Array<{ index: number; up: boolean; pos: Vector3 }>) {
		this.clearFloorArrows();
		for (const it of items) {
			const marker = makeFloorArrow(it.up, this.metrics);
			marker.position.copy(it.pos);
			marker.userData.to = it.index;
			marker.userData.up = it.up;
			marker.userData.baseY = it.pos.y;
			marker.userData.phase = it.up ? 0 : Math.PI / 2;
			this.arrowGroup.add(marker);
		}
	}

	clearFloorArrows() {
		for (const m of this.arrowGroup.children) disposeGroup(m);
		this.arrowGroup.clear();
		this.hoveredArrowMarker = null;
	}

	updateFloorArrows(now: number) {
		if (!this.arrowGroup.visible) return;
		const dt = this.lastArrowTick ? Math.min(100, now - this.lastArrowTick) : 16;
		this.lastArrowTick = now;
		const k = 1 - Math.exp(-dt / FLOOR_ARROW_GLOW_TAU);
		for (const marker of this.arrowGroup.children) {
			const baseY = marker.userData.baseY as number;
			const phase = marker.userData.phase as number;
			marker.position.y =
				baseY + Math.sin(now * FLOOR_ARROW_RATE + phase) * this.metrics.arrowBob;
			const target = marker === this.hoveredArrowMarker ? 1 : 0;
			const level =
				((marker.userData.glow as number) ?? 0) +
				(target - ((marker.userData.glow as number) ?? 0)) * k;
			marker.userData.glow = level;
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

	positionYouMarker(p: Vector3) {
		const floorY = p.y - this.metrics.eyeHeight;
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
		this.ghostType = null;
		this.clearFloorArrows();
		this.arrowGroup.visible = false;
		this.you.group.visible = false;
		this.pulseUntil = 0;
	}

	dispose() {
		this.clear();
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

function setGroupOpacity(marker: Object3D, opacity: number) {
	marker.traverse((o) => {
		const m = (o as Mesh).material as Material | Material[] | undefined;
		if (!m) return;
		for (const mat of Array.isArray(m) ? m : [m]) {
			const factor = (mat.userData?.occludedFactor as number) ?? 1;
			(mat as MeshBasicMaterial).opacity = opacity * factor;
		}
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
