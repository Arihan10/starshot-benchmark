import * as THREE from "three";
import { BLOCKS, CUBE, onGround, PILLARS, TALL } from "./podiumLayout";

const RADIUS = CUBE * 0.5;

const GRAVITY = -30;

const BOUNCE = 0.36;
const SKID = 0.72;

const SPIN_DAMP = 0.6;
const ASLEEP = 0.5;

export const MAX_THROW = 40;

export type Loose = {
	i: number;
	pos: THREE.Vector3;
	vel: THREE.Vector3;
	quat: THREE.Quaternion;
	spin: THREE.Vector3;
	held: boolean;
	asleep: boolean;
};

type Box = {
	i: number;
	min: THREE.Vector3;
	max: THREE.Vector3;
};

const CELL = 3;
const cellKey = (x: number, z: number) =>
	(Math.floor(x / CELL) + 128) * 512 + (Math.floor(z / CELL) + 128);

let grid: Map<number, Box[]> | null = null;

function add(box: Box): void {
	if (!grid) return;
	for (let x = box.min.x; x <= box.max.x + CELL; x += CELL) {
		for (let z = box.min.z; z <= box.max.z + CELL; z += CELL) {
			const key = cellKey(Math.min(x, box.max.x), Math.min(z, box.max.z));
			const cell = grid.get(key);
			if (cell) {
				if (!cell.includes(box)) cell.push(box);
			} else {
				grid.set(key, [box]);
			}
		}
	}
}

function field(): Map<number, Box[]> {
	if (grid) return grid;
	grid = new Map();
	const half = CUBE / 2;
	const blocks = BLOCKS;
	for (let i = 0; i < blocks.length; i++) {
		const [x, y, z] = blocks[i].rest;
		add({
			i,
			min: new THREE.Vector3(x - half, y - half, z - half),
			max: new THREE.Vector3(x + half, y + half, z + half),
		});
	}
	for (const p of PILLARS) {
		add({
			i: -1,
			min: new THREE.Vector3(
				p.x - p.width / 2,
				p.base,
				p.z - p.depth / 2,
			),
			// AT FULL HEIGHT, not at today's. A pillar's height is a function of a
			// rating that changes, and the collision grid is built once — measured
			// against the ceiling the scale can produce, a block can never pass
			// through a post that happens to be taller than the grid remembers.
			max: new THREE.Vector3(
				p.x + p.width / 2,
				p.base + TALL,
				p.z + p.depth / 2,
			),
		});
	}
	return grid;
}

const normal = new THREE.Vector3();
const tangent = new THREE.Vector3();

function overlap(p: THREE.Vector3, box: Box): number {
	const cx = Math.max(box.min.x, Math.min(p.x, box.max.x));
	const cy = Math.max(box.min.y, Math.min(p.y, box.max.y));
	const cz = Math.max(box.min.z, Math.min(p.z, box.max.z));
	const dx = p.x - cx;
	const dy = p.y - cy;
	const dz = p.z - cz;
	const d2 = dx * dx + dy * dy + dz * dz;
	if (d2 >= RADIUS * RADIUS) return 0;

	if (d2 > 1e-8) {
		const d = Math.sqrt(d2);
		normal.set(dx / d, dy / d, dz / d);
		return RADIUS - d;
	}

	const ox = Math.min(p.x - box.min.x, box.max.x - p.x);
	const oy = Math.min(p.y - box.min.y, box.max.y - p.y);
	const oz = Math.min(p.z - box.min.z, box.max.z - p.z);
	if (ox <= oy && ox <= oz) {
		normal.set(p.x < (box.min.x + box.max.x) / 2 ? -1 : 1, 0, 0);
		return ox + RADIUS;
	}
	if (oy <= oz) {
		normal.set(0, p.y < (box.min.y + box.max.y) / 2 ? -1 : 1, 0);
		return oy + RADIUS;
	}
	normal.set(0, 0, p.z < (box.min.z + box.max.z) / 2 ? -1 : 1);
	return oz + RADIUS;
}

function respond(b: Loose, depth: number): void {
	b.pos.addScaledVector(normal, depth);
	const into = b.vel.dot(normal);
	if (into >= 0) return;
	tangent.copy(b.vel).addScaledVector(normal, -into);
	b.vel.copy(normal).multiplyScalar(-into * BOUNCE).addScaledVector(tangent, SKID);
	b.spin.multiplyScalar(0.65);
}

function ground(b: Loose, bounce: boolean): boolean {
	if (b.pos.y >= RADIUS || !onGround(b.pos.x, b.pos.z)) return false;
	if (bounce) {
		normal.set(0, 1, 0);
		respond(b, RADIUS - b.pos.y);
	} else {
		b.pos.y = RADIUS;
	}
	return true;
}

function city(b: Loose, boxes: Map<number, Box[]>, gone: Set<number>): boolean {
	let landed = false;
	for (let cx = -1; cx <= 1; cx++) {
		for (let cz = -1; cz <= 1; cz++) {
			const cell = boxes.get(cellKey(b.pos.x + cx * CELL, b.pos.z + cz * CELL));
			if (!cell) continue;
			for (const box of cell) {
				if (box.i >= 0 && gone.has(box.i)) continue;
				const depth = overlap(b.pos, box);
				if (depth > 0) {
					respond(b, depth);
					if (normal.y > 0.5) landed = true;
				}
			}
		}
	}
	return landed;
}

const turn = new THREE.Quaternion();
const axis = new THREE.Vector3();

export function step(
	loose: Loose[],
	dt: number,
	dropped?: ReadonlySet<number>,
): void {
	const boxes = field();
	const gone = new Set<number>();
	for (const b of loose) gone.add(b.i);
	if (dropped) for (const i of dropped) gone.add(i);

	for (const b of loose) {
		if (b.held) {
			ground(b, false);
			continue;
		}
		if (b.asleep) continue;

		const rate = b.spin.length();
		if (rate > 1e-4) {
			axis.copy(b.spin).divideScalar(rate);
			turn.setFromAxisAngle(axis, rate * dt);
			b.quat.premultiply(turn);
			b.spin.multiplyScalar(Math.max(0, 1 - SPIN_DAMP * dt));
		}

		const reach = b.vel.length() * dt;
		const steps = Math.min(8, Math.max(1, Math.ceil(reach / (RADIUS * 0.6))));
		const h = dt / steps;

		let landed = false;
		for (let k = 0; k < steps; k++) {
			b.vel.y += GRAVITY * h;
			b.pos.addScaledVector(b.vel, h);
			const floor = ground(b, true);
			const wall = city(b, boxes, gone);
			landed = landed || floor || wall;
		}

		if (landed && b.vel.lengthSq() < ASLEEP * ASLEEP) {
			b.vel.set(0, 0, 0);
			b.spin.set(0, 0, 0);
			b.asleep = true;
		}
	}
}
