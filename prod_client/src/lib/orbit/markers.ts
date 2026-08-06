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

export const PEEK_ROTATE_SPEED = 0.5;
export const NAV_AIM_PX = 44;

export const WASD_DIR_COS = Math.SQRT1_2;

export const NAV_COLORS: Record<EdgeType, number> = {
	walk: 0x8fd0ff,
	portal: 0xffc46b,
	vertical: 0x7ef2c2,
	phase: 0xc9a6ff,
	far: 0x9aa7b4,
};
export const CURSOR_CLEAR = 0xc4ccd6;

export const NAV_REST_OPACITY = 0.62;
export const NAV_GAZE_RAD = (55 * Math.PI) / 180;
export const NAV_TARGET_PX = 15;
export const SONAR_DURATION = 3200;

export type YouMarker = { group: Group; sphere: Mesh; ring: Mesh; line: Line };

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

function makeDashedRing(color: number, m: NavMetrics): LineLoop {
	const pts: Vector3[] = [];
	for (let i = 0; i <= 48; i++) {
		const a = (i / 48) * Math.PI * 2;
		pts.push(new Vector3(Math.cos(a) * m.ringOuter, Math.sin(a) * m.ringOuter, 0));
	}
	const loop = new LineLoop(
		new BufferGeometry().setFromPoints(pts),
		new LineDashedMaterial({
			color,
			dashSize: m.ringOuter * 0.2,
			gapSize: m.ringOuter * 0.167,
			transparent: true,
			opacity: NAV_REST_OPACITY,
			depthWrite: false,
		}),
	);
	return loop;
}

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
	cone.rotation.y = Math.PI / 4;
	if (down) cone.rotation.z = Math.PI;
	cone.position.y = m.ringOuter * 1.1 * size;
	return cone;
}

export const NAV_OCCLUDED = 0.4;

function depthSplit(part: Mesh | LineLoop): Object3D[] {
	const behind = part.clone();
	const mat = (part.material as MeshBasicMaterial).clone();
	mat.depthFunc = GreaterDepth;
	mat.opacity = (part.material as MeshBasicMaterial).opacity * NAV_OCCLUDED;
	mat.userData.occludedFactor = NAV_OCCLUDED;
	behind.material = mat;
	return [part, behind];
}

export function makeNavMarker(
	edge: { to: number; type: EdgeType; dy: number },
	floorPos: Vector3,
	m: NavMetrics,
	forceChevron = false,
): Group {
	const group = new Group();
	const color = NAV_COLORS[edge.type];
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

export const FLOOR_ARROW_COLOR = NAV_COLORS.vertical;
export const FLOOR_ARROW_RATE = 0.0018;
export const FLOOR_ARROW_REST = 0.85;
export const FLOOR_ARROW_OCCLUDED = 0.22;
export const FLOOR_ARROW_GLOW_SCALE = 1.12;
export const FLOOR_ARROW_GLOW = 0.16;
export const FLOOR_ARROW_PULSE_RATE = 0.0021;
export const FLOOR_ARROW_HOVER_LIFT = 0.18;
export const FLOOR_ARROW_GLOW_TAU = 260;

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
			cone.rotation.y = Math.PI / 4;
			if (!up) cone.rotation.z = Math.PI;
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
				opacity: 0,
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
		halo.renderOrder = 6;
		group.add(halo);
	}
	group.renderOrder = 7;
	return group;
}

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
