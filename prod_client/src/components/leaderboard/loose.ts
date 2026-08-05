import * as THREE from "three";
import { BLOCKS, CUBE, onGround, PILLARS } from "./podiumLayout";

/**
 * BLOCKS YOU CAN PULL OUT OF THE CITY AND THROW.
 *
 * NO PHYSICS ENGINE, and that is a consequence of the brief rather than a corner
 * cut. What makes a rigid-body engine necessary is bodies pushing EACH OTHER
 * around — the solver exists to resolve a stack of things all shoving at once, and
 * that is where the cost, the WASM payload and the instability all come from. Here
 * the city never moves: a thrown cube bounces off it exactly as it would bounce off
 * a wall, and the only things being integrated are the handful of cubes in the air.
 *
 * So this is one sphere against a set of fixed boxes, which is a page of arithmetic
 * and costs nothing measurable — a dozen cubes in flight is a dozen position
 * updates and a few grid lookups a frame, against Rapier's several hundred KB of
 * WASM and a solver running over nine hundred bodies.
 *
 * A SPHERE, NOT A BOX, for the moving side. A cube tumbling end over end into a
 * corner needs a full contact manifold to look right and needs it fifty times a
 * second; a sphere of the same radius needs one closest-point test, and at the size
 * these are on screen nobody can tell which one bounced. The cube still SPINS —
 * that is what sells it — the spin simply does not take part in the collision.
 */

/** The collision radius of a block. Its half-diagonal would be more honest and
 *  would make blocks bounce off things they visibly missed; the half-EDGE lets them
 *  clip a corner, which reads better. */
const RADIUS = CUBE * 0.5;

/** Heavier than earth, because the whole composition is only a dozen units tall
 *  and real gravity at this scale reads as slow motion. */
const GRAVITY = -30;

/** How much of the approach speed survives a bounce, and how much of the sideways
 *  speed survives the scrape. */
const BOUNCE = 0.36;
const SKID = 0.72;

/** Spin bled off per second, and the speed below which a block on the ground is
 *  declared asleep. Without the second one a block never quite stops — it creeps
 *  and jitters forever on a surface it is already resting on. */
const SPIN_DAMP = 0.6;
const ASLEEP = 0.5;

/** Fall past this and the city takes the block back. */
export const ABYSS = -30;

/** The hardest a block can be thrown, however fast the pointer was moving. */
export const MAX_THROW = 40;

export type Loose = {
	/** Its index in the BLOCKS instance buffer. */
	i: number;
	pos: THREE.Vector3;
	vel: THREE.Vector3;
	quat: THREE.Quaternion;
	/** Axis-angle per second, packed as a vector. */
	spin: THREE.Vector3;
	held: boolean;
	asleep: boolean;
	/**
	 * 0 while the block is its own; climbs to 1 as the timeline takes it back —
	 * after a fall into the dark, or because the reader scrolled and the whole city
	 * is coming apart. The blend is what stops it teleporting to its slot.
	 */
	home: number;
	homeFrom: THREE.Vector3;
	homeQuat: THREE.Quaternion;
};

// --- what a thrown block can hit ------------------------------------------
//
// A UNIFORM GRID over the ground plane, built once. The city is a few hundred
// boxes and a moving block only ever touches the two or three nearest, so the
// alternative — testing all of them, every frame, per block — is a thousand times
// the work for the same answer.

type Box = {
	/** Which instance this is, so a block never collides with itself. */
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
	for (let i = 0; i < BLOCKS.length; i++) {
		const [x, y, z] = BLOCKS[i].rest;
		add({
			i,
			min: new THREE.Vector3(x - half, y - half, z - half),
			max: new THREE.Vector3(x + half, y + half, z + half),
		});
	}
	// The pillars are solid too, and they are the one thing worth bouncing off.
	for (const p of PILLARS) {
		add({
			i: -1,
			min: new THREE.Vector3(
				p.x - p.width / 2,
				p.base,
				p.z - p.depth / 2,
			),
			max: new THREE.Vector3(
				p.x + p.width / 2,
				p.base + p.height,
				p.z + p.depth / 2,
			),
		});
	}
	return grid;
}

const normal = new THREE.Vector3();
const tangent = new THREE.Vector3();

/** Closest-point test. Returns how deep the sphere is, and leaves the direction to
 *  push it in `normal`. Zero means no contact. */
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

	// DEAD CENTRE INSIDE THE BOX, where there is no closest point to push away
	// from. Leave along whichever face is nearest, which is the only answer that
	// does not send the block through the far side.
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

/** Push out of one surface and take the energy off the bounce. */
function respond(b: Loose, depth: number): void {
	b.pos.addScaledVector(normal, depth);
	const into = b.vel.dot(normal);
	if (into >= 0) return;
	// Split the velocity about the contact: the part going INTO the surface comes
	// back reduced, the part sliding along it is scrubbed.
	tangent.copy(b.vel).addScaledVector(normal, -into);
	b.vel.copy(normal).multiplyScalar(-into * BOUNCE).addScaledVector(tangent, SKID);
	b.spin.multiplyScalar(0.65);
}

/** The floor, where there is one. Returns whether it caught the block.
 *
 *  `bounce` false just lifts it clear without touching its velocity — that is the
 *  block on the end of the pointer, which has no momentum of its own. */
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

/** Everything the block is currently inside. Returns whether any of it was
 *  something to stand on. */
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

/**
 * Advance every block that is in the air. Held blocks are the pointer's business
 * and reclaimed ones are the timeline's, so both are skipped here.
 */
export function step(loose: Loose[], dt: number): void {
	const boxes = field();
	// THE HOLES A THROW LEAVES ARE HOLES. The grid was built from the city at rest,
	// so a block that has been pulled out still has a box sitting in its slot —
	// without this, a cube thrown through the gap bounces off the empty air where
	// its neighbour used to be.
	const gone = new Set<number>();
	for (const b of loose) gone.add(b.i);

	for (const b of loose) {
		if (b.home > 0) continue;

		// THE HELD BLOCK IS THE POINTER'S, and its position is simply wherever the
		// cursor put it — but it still may not be inside the floor. Dragging one down
		// used to sink it through the ground and out of sight, because collision was
		// skipped entirely while held; lifting it clear costs one comparison and is
		// the difference between "solid" and "the physics is broken".
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

		// SUB-STEPPED, AND THAT IS THE WHOLE FIX FOR PHASING THROUGH THINGS.
		//
		// Collision here is a test of where the block IS, not of the path it took to
		// get there — so a block that moves further in one frame than the thing it
		// should have hit is thick simply arrives on the far side, having never been
		// inside it on any frame anyone tested. At a throw of forty units a second
		// and a frame of a thirtieth that is over a unit of travel against a
		// half-unit radius: straight through the floor.
		//
		// So the faster it is going, the more often it is asked. Only the handful of
		// blocks in the air ever pay for it, and a block at rest takes one step as
		// before.
		const reach = b.vel.length() * dt;
		const steps = Math.min(8, Math.max(1, Math.ceil(reach / (RADIUS * 0.6))));
		const h = dt / steps;

		let landed = false;
		for (let k = 0; k < steps; k++) {
			b.vel.y += GRAVITY * h;
			b.pos.addScaledVector(b.vel, h);
			// THE FLOOR, but only where there is one. Past the edge of the band the
			// block keeps going, which is what makes the edge worth throwing off.
			const floor = ground(b, true);
			const wall = city(b, boxes, gone);
			landed = landed || floor || wall;
		}

		// SETTLE. A block resting on a surface still has gravity added to it every
		// frame, so without this it shivers in place for as long as the page is open.
		if (landed && b.vel.lengthSq() < ASLEEP * ASLEEP) {
			b.vel.set(0, 0, 0);
			b.spin.set(0, 0, 0);
			b.asleep = true;
		}
	}
}
