"use client";

import { Html } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { markColor } from "@/lib/ink";
import BrandMark, { brandTone } from "./BrandMark";
import { useHeroProgress } from "./heroProgress";
import { ABYSS, type Loose, MAX_THROW, step } from "./loose";
import { paintSweep, PLINTH_ATTRIBUTE, sweepStops } from "./sweep";
import {
	BLOCKS,
	type BlockSpec,
	CUBE,
	DECK,
	FOOTINGS,
	GROUP_Y,
	LABELS,
	PILLARS,
	PLATE,
	PLATE_H,
	VIEW,
} from "./podiumLayout";

/** Seconds the podium takes to build itself on arrival. */
const INTRO = 3.2;

/**
 * How hard the scene chases the scroll. Lower is looser.
 *
 * THIS IS WHERE "SMOOTHLY" LIVES. A mandatory snap is a fast, abrupt scroll by
 * design — the browser covers a whole viewport in a few hundred milliseconds — and
 * driving the model straight off `scrollTop` would make the podium snap apart just
 * as hard. Damping toward the scroll instead lets the page arrive while the podium
 * is still coming apart, so a flick and a slow drag produce the same unhurried
 * disassembly. It is also frame-rate independent, which a plain lerp per frame is
 * not: the same number gives the same curve at 60Hz and 120Hz.
 *
 * TIGHTENED, not loosened, and that is the counter-intuitive half of the fix.
 *
 * The instinct when an animation feels rushed is to slow it down — but slack here
 * delays the START, because the model only ever moves in proportion to how far it
 * is from its target. What makes the exit unhurried is the CURVE in front of this,
 * which is steepest at the very beginning and eases out; what makes it begin at
 * once is a short time constant, so the model actually goes where that curve says.
 *
 * Measured against a 300ms snap: the city is visibly in motion 100ms in — six
 * frames — and is still settling at 800ms, long after the reader has arrived. At
 * the loose constant this replaces, nothing but the pillars moved for the first
 * fifth of a second and the whole thing then happened at once.
 */
const CHASE = 7;

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const easeOutCubic = (u: number) => 1 - (1 - u) ** 3;
// A little past the mark and back — the "pop" a block makes as it seats itself.
const easeOutBack = (u: number) => {
	const c = 1.34;
	return 1 + (c + 1) * (u - 1) ** 3 + c * (u - 1) ** 2;
};

function useReducedMotion() {
	const [reduced, setReduced] = useState(
		() =>
			typeof window !== "undefined" &&
			window.matchMedia("(prefers-reduced-motion: reduce)").matches,
	);
	useEffect(() => {
		const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
		const sync = () => setReduced(mq.matches);
		mq.addEventListener("change", sync);
		return () => mq.removeEventListener("change", sync);
	}, []);
	return reduced;
}

/** Air left around the podium once it is fitted to the stage. */
const MARGIN = 1.08;

/** The camera's zoom — pixels per world unit before the stage's own fit is applied.
 *  Named because the etched type has to size itself in world units, and that
 *  conversion needs the same number the camera was built with. */
const ZOOM = 50;

/**
 * HOW TO LIE A FLAT THING ON A FACE OF A BOX, without leaving the DOM.
 *
 * The camera never moves, so every face of an axis-aligned box projects to the same
 * parallelogram forever — which means "painted on that face" is a fixed 2D shear,
 * not a 3D transform. A CSS matrix maps the element's own right and down axes onto
 * the two world directions the face is spanned by:
 *
 *   world +X lands at (cos45, sin45·sin35.26) = ( 0.707,  0.408)
 *   world +Z lands at its mirror              = (-0.707,  0.408)
 *   world +Y lands straight up                = ( 0,     -0.816)
 *
 * Which is the whole trick: no CSS3D, no textures, no second text renderer — the
 * type stays real DOM in the site's own faces and logos stay the same components
 * the table uses, and they still read as cut into the stone.
 *
 * `matrix(a,b,c,d,…)` is column-major: (a,b) is where the element's RIGHT goes and
 * (c,d) is where its DOWN goes.
 */
/** The face turned to the left of screen — its normal is world +Z. Reads across
 *  world +X, down world −Y. */
const LEFT_FACE = "matrix(0.7071, 0.4082, 0, 0.8165, 0, 0)";
/** The face turned to the right — normal world +X. Reads across world −Z. */
const RIGHT_FACE = "matrix(0.7071, -0.4082, 0, 0.8165, 0, 0)";
/** The top — normal world +Y — with its BOTTOM toward the right-hand face, so the
 *  logo sits the way you would read it standing on that side of the pillar. */
const TOP_FACE = "matrix(0.7071, -0.4082, 0.7071, 0.4082, 0, 0)";

/** Etched, not printed: black at low opacity darkens whatever brand colour is under
 *  it, which is what a cut into a surface does. Bright ink laid on top would read as
 *  a sticker. */
const ETCH = "rgb(0 0 0 / 0.42)";

/** How far up the pillar the rank and rating sit. Near the foot, so the shaft above
 *  them is left clear and the eye still reads the pillar as a column. */
const ETCH_HEIGHT = 0.95;

/**
 * How much a pillar swells under the pointer.
 *
 * WIDER MORE THAN TALLER, and the difference is not arbitrary: the name floats a
 * fixed clearance above each pillar's top, so growth in HEIGHT eats that gap. Held
 * to a few percent it stays comfortably clear, while the width carries most of the
 * gesture — which also reads better, because a block that swells toward you is
 * responding and one that stretches upward is just getting bigger.
 */
const SWELL = 1.075;
const LIFT = 1.03;

/**
 * One scratch transform, reused for every instance of every frame.
 *
 * `Object3D` is only being borrowed for its compose-a-matrix maths here; allocating
 * one per instance per frame would be four hundred objects sixty times a second,
 * which is the kind of garbage that shows up as a stutter rather than as a number.
 */
const scratch = new THREE.Object3D();
const spot = new THREE.Vector3();
const tumble = new THREE.Euler();
const scripted = new THREE.Quaternion();

/**
 * Where a piece is at time `t` — position into `spot`, rotation into `tumble`,
 * scale returned. A scale of zero means it has not set off yet.
 *
 * PULLED OUT OF `fly` because the throw needs it too: a block that was in the air
 * when the reader scrolled has to be blended back to wherever the timeline says its
 * slot is RIGHT NOW, and that is this same sample, taken from two places.
 */
function sample(s: BlockSpec, t: number): number {
	const u = clamp01((t - s.start) / (s.end - s.start));
	if (u <= 0) return 0;
	const e = easeOutCubic(u);
	spot.set(
		s.from[0] + (s.rest[0] - s.from[0]) * e,
		s.from[1] + (s.rest[1] - s.from[1]) * e,
		s.from[2] + (s.rest[2] - s.from[2]) * e,
	);
	// The tumble unwinds as it flies, so every piece is square with the world by the
	// time it seats — nothing lands crooked and corrects itself.
	const left = 1 - e;
	tumble.set(s.spin[0] * left, s.spin[1] * left, s.spin[2] * left);
	// FULL SIZE BEFORE IT LANDS. Scale runs on the first half of the flight so a
	// piece reads as a solid thing travelling, not as one inflating into place at
	// the moment it touches down.
	return easeOutCubic(clamp01(u / 0.5));
}

/**
 * Fly one group of pieces to where the timeline says they should be.
 *
 * INSTANCED, because the city is four hundred blocks and they are all the same two
 * boxes. As individual meshes that is four hundred React elements, four hundred
 * scene-graph nodes and four hundred draw calls to say one thing; as two
 * `InstancedMesh`es it is two of each, and adding another district to the city
 * costs a matrix rather than a component.
 *
 * WRITTEN ONCE FOR BOTH the ground and the blocks, because the motion is the same
 * motion — only the geometry differs, and that is decided at the mesh.
 */
function fly(
	mesh: THREE.InstancedMesh | null,
	specs: BlockSpec[],
	t: number,
): void {
	if (!mesh) return;
	for (let i = 0; i < specs.length; i++) {
		const scale = sample(specs[i], t);
		if (scale <= 0) {
			// AN INSTANCE CANNOT BE HIDDEN, only shrunk — there is no per-instance
			// `visible` — so a piece that has not left yet is scaled to nothing.
			scratch.position.set(0, 0, 0);
			scratch.rotation.set(0, 0, 0);
			scratch.scale.setScalar(0);
		} else {
			scratch.position.copy(spot);
			scratch.rotation.copy(tumble);
			scratch.scale.setScalar(scale);
		}
		scratch.updateMatrix();
		mesh.setMatrixAt(i, scratch.matrix);
	}
	mesh.instanceMatrix.needsUpdate = true;
}

/**
 * Draw the blocks that are no longer the timeline's — the ones in the air, the one
 * on the end of the pointer, and any being handed back.
 *
 * RUN AFTER `fly`, and only over the handful of blocks that are loose, so it
 * overwrites their slots in the same buffer the city was just written into.
 */
function paint(
	mesh: THREE.InstancedMesh | null,
	list: Loose[],
	t: number,
): void {
	if (!mesh) return;
	for (const b of list) {
		if (b.home > 0) {
			// GOING BACK, and the target is moving: if the reader is scrolling, this
			// block's slot is itself flying off the board, so the blend runs toward
			// wherever the timeline has it at this instant rather than toward its
			// resting place. A block caught mid-throw rejoins the disassembly instead
			// of snapping home and then leaving again.
			const e = easeOutCubic(clamp01(b.home));
			const scale = sample(BLOCKS[b.i], t);
			scratch.position.lerpVectors(b.homeFrom, spot, e);
			scratch.quaternion
				.copy(b.homeQuat)
				.slerp(scripted.setFromEuler(tumble), e);
			scratch.scale.setScalar(Math.max(1e-4, scale));
		} else {
			scratch.position.copy(b.pos);
			scratch.quaternion.copy(b.quat);
			scratch.scale.setScalar(1);
		}
		scratch.updateMatrix();
		mesh.setMatrixAt(b.i, scratch.matrix);
	}
	mesh.instanceMatrix.needsUpdate = true;
}

type Row = { rank: number; name: string; lab: string; elo: number };

function Stage({
	rows,
	reduced,
}: {
	rows: Row[];
	reduced: boolean;
}) {
	const progress = useHeroProgress();

	// HOW BIG THE PODIUM IS ON THIS STAGE, and it is a SCALE rather than a camera
	// zoom. Mutating the camera each time the canvas resizes is the usual way and it
	// reaches into an object React handed us; scaling the model is the same
	// projection, computed during render from a value R3F already derives, and it
	// stays declarative — resize the window and this is simply a different number.
	//
	// drei's `<Bounds fit observe>` would do the job by MEASURING the scene, which
	// is the one thing that cannot work here: the model is in pieces for most of the
	// animation, so a measured box would breathe as the blocks fly in and the
	// framing would ride it. The podium's assembled size is a constant this codebase
	// already knows (see podiumLayout), so the fit is computed from that and holds
	// still while the thing inside it moves.
	const viewport = useThree((s) => s.viewport);
	const camera = useThree((s) => s.camera);
	const gl = useThree((s) => s.gl);

	// THE CANVAS, HELD IN A REF rather than reached for through `gl` at the moment
	// it is needed. The drag sets a cursor and captures a pointer, and both of those
	// are mutations of something React handed us — legal through a ref, which is
	// what a ref is for, and a lint error through the hook's own value.
	const canvas = useRef<HTMLCanvasElement | null>(null);
	useEffect(() => {
		canvas.current = gl.domElement;
	}, [gl]);

	/** Our own, so the drag never disturbs the one the event system is using. */
	const caster = useMemo(() => new THREE.Raycaster(), []);
	const fit = Math.min(
		viewport.width / (VIEW.w * MARGIN),
		viewport.height / (VIEW.h * MARGIN),
	);

	const plates = useRef<THREE.InstancedMesh>(null);
	const blocks = useRef<THREE.InstancedMesh>(null);
	const pillars = useRef<(THREE.Mesh | null)[]>([]);
	/** Which pillar the pointer is on, and how far each has swelled toward it.
	 *  Refs, because a hover that re-rendered the scene would rebuild the city to
	 *  animate three boxes. */
	const over = useRef(PILLARS.map(() => false));
	const swell = useRef(PILLARS.map(() => 0));
	/** Every piece of type on the podium — four per pillar, in one list, because
	 *  they all arrive on the same beat. */
	const labels = useRef<(HTMLDivElement | null)[]>([]);
	/** The frame each pillar's writing hangs in, scaled with the stone it is on. */
	const stones = useRef<(THREE.Group | null)[]>([]);
	/** The inner group, so screen positions can be brought into ITS space — the
	 *  whole model is scaled to fit the stage, and a throw has to happen in the
	 *  coordinates the city was built in rather than in the window's. */
	const model = useRef<THREE.Group>(null);

	// --- the blocks that are nobody's but their own -------------------------
	const loose = useRef<Loose[]>([]);
	const dragging = useRef<Loose | null>(null);
	const dragPlane = useRef(new THREE.Plane());
	const dragLast = useRef(new THREE.Vector3());
	const dragTime = useRef(0);

	// The last value the scene was drawn at. Once the city is assembled and nobody
	// is scrolling, `t` stops moving — and rewriting four hundred identical matrices
	// every frame to prove it is the one avoidable cost in here.
	const drawn = useRef(-1);
	/** Whether the city is standing still and finished — the only state a block may
	 *  be pulled out of. Kept apart from `drawn`, which is a drawing cache and gets
	 *  invalidated for reasons that have nothing to do with whether the city is up. */
	const settled = useRef(false);

	// The two inputs to the one scalar: a clock that runs once, and a damped copy
	// of the scroll. Refs, because nothing about them belongs in a render.
	const intro = useRef(reduced ? 1 : 0);
	const chased = useRef(1);

	// THREE GEOMETRIES AND ONE MATERIAL for a hundred and fifty meshes. Sharing them
	// is what keeps this cheap: the meshes are separate objects because each one
	// moves on its own path, but they are all drawing the same two boxes, and a
	// material compiled once is a shader compiled once.
	const plate = useMemo(
		() => new THREE.BoxGeometry(PLATE, PLATE_H, PLATE),
		[],
	);
	const cube = useMemo(() => new THREE.BoxGeometry(CUBE, CUBE, CUBE), []);
	const post = useMemo(() => {
		const g = new THREE.BoxGeometry(1, 1, 1);
		// ORIGIN AT THE FOOT, so scaling Y grows the pillar UPWARD out of the deck.
		// Scaled about its centre it would sink into the platform as it grew, which
		// is the one thing "rising out of it" must not look like.
		g.translate(0, 0.5, 0);
		return g;
	}, []);
	const skin = useMemo(
		() =>
			new THREE.MeshStandardMaterial({
				color: markColor(),
				roughness: 0.74,
				metalness: 0,
			}),
		[],
	);

	/**
	 * THE SAME WHITE, BUT ABLE TO PAINT THE ACCENT ON THE CUBES THAT ASK FOR IT.
	 *
	 * A separate material from `skin` only because the two are drawn from different
	 * geometries: the patched shader declares a per-instance attribute, and the deck
	 * plates have no such attribute to give it. One extra compile, and the deck is
	 * left on a shader with nothing in it that it does not use.
	 */
	const stone = useMemo(() => {
		const m = new THREE.MeshStandardMaterial({
			color: markColor(),
			roughness: 0.74,
			metalness: 0,
		});
		return { material: m, ...paintSweep(m) };
	}, []);

	/**
	 * WHICH CUBES ARE PLINTH, as one float per instance.
	 *
	 * The alternative was a second mesh for the twenty-seven of them, which would
	 * have meant a second copy of every path that flies, throws, or returns a block
	 * — all to express something the shader can be told in one attribute. Set once:
	 * a cube's membership never changes, only its matrix does.
	 */
	useEffect(() => {
		const flags = new Float32Array(BLOCKS.length);
		for (const foot of FOOTINGS) flags.fill(1, foot.from, foot.from + foot.count);
		cube.setAttribute(
			PLINTH_ATTRIBUTE,
			new THREE.InstancedBufferAttribute(flags, 1),
		);
	}, [cube]);

	/** And the ramp itself, once the document exists to be asked. */
	useEffect(() => {
		const read = sweepStops();
		for (let i = 0; i < stone.stops.value.length; i++) {
			stone.stops.value[i].copy(read[i]);
		}
	}, [stone]);

	// THE THREE PLACES, EACH IN ITS MODEL'S OWN COLOUR — the one thing in the scene
	// that is not white. Three materials rather than one, because each is a
	// different colour and a material is the only place a colour can live for a
	// plain mesh; the city stays on the shared white `skin` behind them, which is
	// what makes three coloured objects read as THE subject rather than as three
	// more buildings.
	//
	// `rows` arrives from a server component and holds its identity across renders,
	// so this rebuilds when the standings change and not when the window resizes.
	const tones = useMemo(
		() =>
			PILLARS.map((p) => {
				const row = rows.find((r) => r.rank === p.rank);
				return new THREE.MeshStandardMaterial({
					color: row ? brandTone(row.lab) : "#ededed",
					roughness: 0.68,
					metalness: 0,
				});
			}),
		[rows],
	);

	useEffect(
		() => () => {
			plate.dispose();
			cube.dispose();
			post.dispose();
			skin.dispose();
			stone.material.dispose();
			for (const m of tones) m.dispose();
		},
		[plate, cube, post, skin, stone, tones],
	);

	/**
	 * A BOUNDING SPHERE BIG ENOUGH TO HOLD THE WHOLE ANIMATION, set by hand.
	 *
	 * `InstancedMesh.raycast` rejects a ray against this sphere before it tests a
	 * single instance, and if it is null three computes one — ONCE, lazily, from
	 * whatever the instance matrices happen to say at that moment. The first pointer
	 * move over the canvas is usually during the intro, when every block is scaled
	 * to nothing at the origin, so it would cache a sphere of almost no radius and
	 * then silently reject every ray for the rest of the session. Which is exactly
	 * what "clicking blocks does nothing" looks like.
	 *
	 * Measured off the specs — the resting city AND the scatter positions the blocks
	 * fly in from, since they are real positions the mesh occupies — so it stays
	 * right if the city grows.
	 */
	useEffect(() => {
		const mesh = blocks.current;
		if (!mesh) return;
		let far = 0;
		for (const b of BLOCKS) {
			far = Math.max(
				far,
				Math.hypot(b.rest[0], b.rest[1], b.rest[2]),
				Math.hypot(b.from[0], b.from[1], b.from[2]),
			);
		}
		mesh.boundingSphere = new THREE.Sphere(new THREE.Vector3(), far + CUBE);
	}, []);


	/**
	 * DRAG AND RELEASE, on the canvas rather than on the block.
	 *
	 * A pointer handler on the mesh only fires while the pointer is OVER it, and a
	 * throw is precisely the gesture where it stops being: you pull the block away
	 * faster than the hit test can follow, and halfway through the flick the events
	 * would go to whatever is underneath. Listening on the canvas — with the pointer
	 * captured — means the drag belongs to the block until it is let go.
	 */
	useEffect(() => {
		const el = gl.domElement;
		const ndc = new THREE.Vector2();
		const at = new THREE.Vector3();
		const flick = new THREE.Vector3();

		const move = (ev: PointerEvent) => {
			const b = dragging.current;
			const group = model.current;
			if (!b || !group) return;
			const rect = el.getBoundingClientRect();
			ndc.set(
				((ev.clientX - rect.left) / rect.width) * 2 - 1,
				-((ev.clientY - rect.top) / rect.height) * 2 + 1,
			);
			caster.setFromCamera(ndc, camera);
			// ONTO A PLANE FACING THE CAMERA, through wherever the block was picked
			// up. A pointer gives two numbers and a position needs three, so the third
			// has to be decided: holding the grab depth is what makes the block track
			// the cursor exactly rather than sliding away up the ground plane.
			if (!caster.ray.intersectPlane(dragPlane.current, at)) return;
			group.worldToLocal(at);

			// THE THROW IS MEASURED HERE, off the POINTER's own clock.
			//
			// It used to be measured once a frame, from how far the block had moved
			// since the last one — and a flick almost always ends with the hand
			// slowing before the button comes up, so the frame that happened to be
			// last read as barely moving and the block was released at a standstill.
			// It looked like the throw was being thrown away, because it was.
			//
			// Event timestamps are independent of when frames happen, and the running
			// average keeps the speed of the flick alive across the last few samples
			// instead of trusting whichever one arrived last. It still decays: pause
			// with the button down and you set the block down rather than throw it.
			const gap = Math.max(4, ev.timeStamp - dragTime.current) / 1000;
			dragTime.current = ev.timeStamp;
			flick.subVectors(at, dragLast.current).divideScalar(gap);
			dragLast.current.copy(at);
			b.vel.lerp(flick, 0.65);

			b.pos.copy(at);
			b.asleep = false;
		};

		const release = () => {
			const b = dragging.current;
			dragging.current = null;
			el.style.cursor = "";
			if (!b) return;
			b.held = false;
			// The velocity was measured over the last frame while it was held; capping
			// it stops a flick across the whole window from launching a block into the
			// next county.
			b.vel.clampLength(0, MAX_THROW);
			const speed = b.vel.length();
			b.spin
				.set(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5)
				.multiplyScalar(speed * 0.5 + 1);
		};

		el.addEventListener("pointermove", move);
		el.addEventListener("pointerup", release);
		el.addEventListener("pointercancel", release);
		return () => {
			el.removeEventListener("pointermove", move);
			el.removeEventListener("pointerup", release);
			el.removeEventListener("pointercancel", release);
		};
	}, [gl, camera, caster]);

	useFrame((_, delta) => {
		// A tab left in the background hands back one enormous delta on return;
		// capped, the podium resumes rather than teleporting through its own build.
		const dt = Math.min(delta, 1 / 30);

		if (intro.current < 1) intro.current = Math.min(1, intro.current + dt / INTRO);

		// FRONT-LOADED, AND THAT IS THE WHOLE POINT.
		//
		// This was a `smoothstep` with a few percent of dead zone in front of it, and
		// it read as an animation that started late and then panicked. Smoothstep is
		// flat at BOTH ends: the first fifth of the scroll moved the target by about
		// one percent, so nothing appeared to happen at all, and then the whole
		// disassembly was crammed into the middle of a snap that only lasts a few
		// hundred milliseconds.
		//
		// The curve below is the opposite shape — steepest at the very start, easing
		// out. A fiftieth of a scroll already pulls the target down past 0.78, and a
		// twentieth past 0.69, which matters because of how the timeline is laid out:
		// the pillars occupy the top of it and the CITY sits between 0.2 and 0.7, so
		// the target has to fall that far before there is anything to see. Now it
		// gets there in the first few percent of the gesture.
		const q = clamp01(progress.current / 0.78);
		chased.current = THREE.MathUtils.damp(
			chased.current,
			1 - q ** 0.42,
			CHASE,
			dt,
		);

		// THE MINIMUM, not the product: whichever of "has it finished building" and
		// "has it been scrolled away" is further from assembled wins. That is what
		// lets a reader scroll during the intro and get a disassembly from wherever
		// the build had got to, rather than the two fighting over the same meshes.
		const t = Math.min(intro.current, chased.current);
		settled.current = t > 0.999;

		// THE CITY IS ONLY REDRAWN WHEN IT HAS MOVED. Once it is assembled and nobody
		// is scrolling, `t` stops — and rewriting nine hundred identical matrices
		// every frame to prove it is the one avoidable cost in here. The loose blocks
		// below are not covered by this: they keep moving after the timeline stops.
		if (Math.abs(t - drawn.current) >= 0.0002) {
			drawn.current = t;
			fly(plates.current, DECK, t);
			fly(blocks.current, BLOCKS, t);
		}

		// THE PODIUM AND ITS NAMES, EVERY FRAME AND UNCONDITIONALLY.
		//
		// These used to sit below an early return that skipped the rest of the frame
		// when no block was loose — which is the normal state — so the pillars never
		// got their height and stood as unit cubes, and the names never faded up. A
		// guard for one feature is not a guard for the frame.
		for (let i = 0; i < PILLARS.length; i++) {
			const mesh = pillars.current[i];
			if (!mesh) continue;
			const p = PILLARS[i];
			const u = clamp01((t - p.start) / (p.end - p.start));
			if (u <= 0) {
				mesh.visible = false;
				continue;
			}
			mesh.visible = true;
			// Damped rather than switched, so a pointer crossing the podium leaves a
			// swell behind it rather than three blocks snapping to attention.
			swell.current[i] = THREE.MathUtils.damp(
				swell.current[i],
				over.current[i] ? 1 : 0,
				9,
				dt,
			);
			const s = swell.current[i];
			const wide = 1 + (SWELL - 1) * s;
			const tall = 1 + (LIFT - 1) * s;
			const rise = Math.max(1e-3, easeOutBack(u) * tall);
			mesh.scale.set(p.width * wide, p.height * rise, p.depth * wide);

			// AND ITS WRITING WITH IT. The marks are placed in the pillar's own
			// coordinates, so giving their frame the pillar's transform carries all
			// four: the rank and rating slide out to stay on the faces, the logo and
			// the name ride up on the top. Widths only for the type itself — a letter
			// stretched by a different amount along each axis stops being the letter,
			// and at a swell of a few percent the pair are within a percent anyway.
			stones.current[i]?.scale.set(wide, rise, wide);
			for (let k = 0; k < 4; k++) {
				const el = labels.current[i * 4 + k];
				// A STANDALONE `scale`, not a transform — three of these four already
				// carry the shear that lies them on a face, and writing over it would
				// stand them back up. The individual properties compose with it, and a
				// uniform scale commutes, so the order does not matter either.
				if (el) el.style.scale = String(wide);
			}
		}

		// The names above and the placings below arrive together, so the podium is
		// captioned in one moment rather than in two.
		const named = clamp01((t - LABELS[0]) / (LABELS[1] - LABELS[0]));
		// OPACITY ONLY. The etched marks carry a transform that puts them on their
		// face, and a rise written into the same property would peel them off it.
		for (const el of labels.current) {
			if (el) el.style.opacity = String(named);
		}

		// --- and then whatever is in the air ---------------------------------
		const list = loose.current;
		if (list.length === 0) return;

		// WHAT TAKES A BLOCK BACK: the reader scrolling, so the whole city is coming
		// apart and a stray cube cannot stay behind; or a fall past the edge of the
		// world, which would otherwise cost the city a block every time someone
		// missed. Both hand it to the same blend.
		for (const b of list) {
			if (b.home === 0 && (t < 0.999 || b.pos.y < ABYSS)) {
				if (dragging.current === b) dragging.current = null;
				b.held = false;
				b.home = 1e-4;
				b.homeFrom.copy(b.pos);
				b.homeQuat.copy(b.quat);
			}
			if (b.home > 0) b.home = Math.min(1, b.home + dt / 0.55);
		}

		step(list, dt);
		paint(blocks.current, list, t);

		if (list.some((b) => b.home >= 1)) {
			loose.current = list.filter((b) => b.home < 1);
			// Their slots belong to the timeline again, and the timeline has not
			// written them since they were pulled out. Force one full pass.
			drawn.current = -1;
		}
	});

	return (
		// TWO GROUPS, and the nesting is what keeps the maths honest: the outer one
		// scales about the origin, the inner one slides the model's centre of mass
		// onto that origin. Combined on one group the offset would be scaled too, and
		// the podium would drift up the stage as the window widened.
		<group scale={fit}>
			<group ref={model} position={[0, GROUP_Y, 0]}>
				{/* THE GROUND, and it is what fills the picture. Every row laid behind
				    the plaza climbs the screen, so the depth of this field is what turns
				    a band across the middle of the stage into a city seen from above —
				    see podiumLayout, where it is generated in the PROJECTED axes for
				    exactly that reason.

				    `frustumCulled={false}` because an instanced mesh is culled on the
				    bounding sphere of its GEOMETRY — one plate — and not on where its
				    instances actually are, so the whole city can blink out when the
				    origin leaves the frustum. */}
				<instancedMesh
					ref={plates}
					args={[plate, skin, DECK.length]}
					frustumCulled={false}
				/>

				{/* AND THESE ONES COME OUT. Every block of every building is grabbable
				    — not the ground, which would leave holes for things to fall
				    through, and not the pillars, which are the subject.

				    TOUCH IS EXCLUDED, deliberately. Inside a snap-scrolling hero a drag
				    IS the scroll gesture, and there is no way to claim it for throwing
				    without taking the page's own scrolling away from a phone. On a
				    pointer with a wheel behind it the two gestures are already
				    separate. */}
				<instancedMesh
					ref={blocks}
					args={[cube, stone.material, BLOCKS.length]}
					frustumCulled={false}
					onPointerOver={() => {
						if (settled.current && canvas.current) canvas.current.style.cursor = "grab";
					}}
					onPointerOut={() => {
						if (!dragging.current && canvas.current) canvas.current.style.cursor = "";
					}}
					onPointerDown={(e) => {
						const group = model.current;
						const id = e.instanceId;
						if (id == null || !group) return;
						if (e.nativeEvent.pointerType === "touch") return;
						// Nothing comes loose while the city is still building itself or
						// already coming apart — the timeline owns every block then, and
						// two owners is how a block ends up in two places.
						if (!settled.current) return;
						e.stopPropagation();

						let b = loose.current.find((l) => l.i === id);
						if (!b) {
							b = {
								i: id,
								pos: new THREE.Vector3(...BLOCKS[id].rest),
								vel: new THREE.Vector3(),
								quat: new THREE.Quaternion(),
								spin: new THREE.Vector3(),
								held: true,
								asleep: false,
								home: 0,
								homeFrom: new THREE.Vector3(),
								homeQuat: new THREE.Quaternion(),
							};
							loose.current.push(b);
						}
						b.held = true;
						b.asleep = false;
						b.home = 0;
						b.vel.set(0, 0, 0);

						// A PLANE FACING THE CAMERA, THROUGH THE BLOCK. Everything the
						// pointer does from here is read against this one surface.
						camera.getWorldDirection(spot);
						dragPlane.current.setFromNormalAndCoplanarPoint(spot, e.point);
						dragging.current = b;
						dragLast.current.copy(b.pos);
						dragTime.current = e.nativeEvent.timeStamp;
						canvas.current?.style.setProperty("cursor", "grabbing");
						canvas.current?.setPointerCapture(e.nativeEvent.pointerId);

						// A HANDFUL AT A TIME. Left unbounded, a determined reader ends up
						// with the whole city in the air and every one of them integrating
						// every frame; past the cap the longest-loose block goes home.
						if (loose.current.length > 24) {
							const oldest = loose.current.find(
								(l) => l !== b && l.home === 0,
							);
							if (oldest) {
								oldest.home = 1e-4;
								oldest.homeFrom.copy(oldest.pos);
								oldest.homeQuat.copy(oldest.quat);
							}
						}
					}}
				/>

				{/* SOLID TO THE POINTER, WHICH THEY WERE NOT. With no handlers of their
				    own a pillar was invisible to the event system: the ray went straight
				    through it and grabbed whatever block of the city stood behind, so
				    the podium could be reached through. Claiming the event here — the
				    pillars are nearer the camera, and R3F walks the intersections in
				    distance order — stops it at the surface it looks like it should
				    stop at. */}
				{PILLARS.map((p, i) => (
					<mesh
						key={p.rank}
						ref={(m) => {
							pillars.current[i] = m;
						}}
						geometry={post}
						material={tones[i]}
						position={[p.x, p.base, p.z]}
						visible={false}
						onPointerOver={(e) => {
							e.stopPropagation();
							over.current[i] = true;
						}}
						onPointerOut={() => {
							over.current[i] = false;
						}}
						onPointerDown={(e) => {
							e.stopPropagation();
						}}
					/>
				))}

				{/* --- what is written on each pillar --------------------------------
				    THREE OF THE FOUR ARE CUT INTO IT rather than floating over it. A
				    podium is a monument, and a monument carries its numbers on its own
				    stone — the placing on the face turned left, the rating on the face
				    turned right, the maker's mark on the top. Only the NAME is left
				    hovering, because a name has to be read against the sky from across
				    the plaza and a face of a pillar is not big enough to carry it.

				    They are still DOM — type and logos are the two things a canvas is
				    worse at than the page around it — but sheared onto the faces by the
				    projection's own constants, so they belong to the stone rather than
				    sitting in front of it. Sized in WORLD units: the font is multiplied
				    by the stage's fit, so an etched numeral keeps its proportion to the
				    pillar at every window size, which fixed pixels would not. */}
				{PILLARS.map((p, i) => {
					const row = rows.find((r) => r.rank === p.rank);
					if (!row) return null;
					// World units to screen pixels, at this stage's scale.
					const px = fit * ZOOM;
					return (
						/* ONE GROUP PER PILLAR, STANDING ON ITS BASE, and carrying the same
						   transform the stone does — which is what makes the writing part of
						   the pillar rather than something hanging in front of it. Every mark
						   is placed in the pillar's OWN coordinates, so the frame loop moves
						   all four by scaling one object: the rank and rating ride out as it
						   widens, the top mark rides up as it grows.

						   The group's scale reaches the anchors but not the type — an `Html`
						   in screen space takes a position from the scene and nothing else —
						   so the frame loop scales the elements themselves to match. */
						<group
							key={p.rank}
							ref={(g) => {
								stones.current[i] = g;
							}}
							position={[p.x, p.base, p.z]}
						>
							{/* THE NAME, above the pillar and against the sky — the one thing not
							    cut into the stone, because a name has to be read from across the
							    plaza and no face of a pillar is wide enough to carry it.

							    OUTLINED RATHER THAN BACKED. It cannot simply be white: an
							    isometric city puts its far towers HIGH in the picture, so this
							    type crosses white roofs as often as black sky. A stroke holds it
							    against both without laying a dark card over the skyline the way
							    the pool it replaces did. `paint-order` is the whole reason it
							    works — by default a stroke is painted OVER the fill and eats half
							    its width out of the letterforms, which at this weight closes the
							    counters; sent behind, it grows outward and the glyphs keep their
							    shape. */}
							<Html
								position={[0, p.height + 0.85, 0]}
								center
								zIndexRange={[8, 0]}
								style={{ pointerEvents: "none" }}
							>
								<div
									ref={(el) => {
										labels.current[i * 4 + 0] = el;
									}}
									style={{
										opacity: 0,
										WebkitTextStrokeWidth: "4px",
										WebkitTextStrokeColor: "#000",
										paintOrder: "stroke fill",
									}}
									className="w-[260px] text-center text-[21px] leading-tight font-extrabold text-ink"
								>
									{row.name}
								</div>
							</Html>

							{/* THE PLACING, cut into the left-hand face. */}
							<Html
								position={[0, ETCH_HEIGHT, p.depth / 2]}
								center
								zIndexRange={[7, 0]}
								style={{ pointerEvents: "none" }}
							>
								<div
									ref={(el) => {
										labels.current[i * 4 + 1] = el;
									}}
									className="font-sans font-black tabular-nums"
									style={{
										opacity: 0,
										transform: LEFT_FACE,
										fontSize: `${0.82 * px}px`,
										lineHeight: 1,
										letterSpacing: "-0.03em",
										color: ETCH,
									}}
								>
									{String(row.rank).padStart(2, "0")}
								</div>
							</Html>

							{/* THE RATING, cut into the right-hand face — under its own heading,
							    because four digits on a wall are not self-explanatory. A reader
							    who has not met the leaderboard below yet has no reason to know
							    that 1487 is a rating rather than a score, a count of wins, or a
							    year.

							    ONE SHEAR AROUND BOTH LINES, not one each. The face matrix is on
							    the wrapper, so "above" is just the previous line of ordinary DOM
							    flow — which the shear then carries UP THE PILLAR, in the same
							    plane, with the leading between them scaling like everything else
							    on the stone. Two separately-placed anchors would have to agree
							    about that by arithmetic instead.

							    Both lines are cut at the same depth. A caption set smaller is
							    quieter for that reason alone, and a chisel does not press
							    lighter for small letters. */}
							<Html
								position={[p.width / 2, ETCH_HEIGHT, 0]}
								center
								zIndexRange={[7, 0]}
								style={{ pointerEvents: "none" }}
							>
								<div
									ref={(el) => {
										labels.current[i * 4 + 2] = el;
									}}
									className="text-center font-mono font-bold whitespace-nowrap tabular-nums"
									style={{
										opacity: 0,
										transform: RIGHT_FACE,
										lineHeight: 1,
										color: ETCH,
									}}
								>
									<div
										style={{
											fontSize: `${0.19 * px}px`,
											// Wide, because a three-letter word at this size needs
											// width to read as a label rather than as a smudge.
											letterSpacing: "0.42em",
											// The tracking is added to the RIGHT of the last letter
											// too, so the word hangs left of centre without this.
											textIndent: "0.42em",
											marginBottom: `${0.13 * px}px`,
										}}
									>
										ELO
									</div>
									<div
										style={{
											fontSize: `${0.42 * px}px`,
											letterSpacing: "0.12em",
											textIndent: "0.12em",
										}}
									>
										{row.elo}
									</div>
								</div>
							</Html>

							{/* THE MARK, cut into the top. Darkened to black rather than left in
							    the lab's colours: a coloured logo lying on a lit face reads as a
							    decal stuck to it, and every one of these marks — flat or
							    gradient, dark or light — goes to the same engraved grey through
							    one filter. */}
							<Html
								position={[0, p.height, 0]}
								center
								zIndexRange={[7, 0]}
								style={{ pointerEvents: "none" }}
							>
								<div
									ref={(el) => {
										labels.current[i * 4 + 3] = el;
									}}
									style={{
										opacity: 0,
										transform: TOP_FACE,
										filter: "brightness(0)",
									}}
								>
									<BrandMark lab={row.lab} size={Math.round(1.05 * px)} />
								</div>
							</Html>
						</group>
					);
				})}
			</group>
		</group>
	);
}

/**
 * The podium: the top three, built out of white isometric blocks.
 *
 * WHY THREE.JS AND NOT CSS. Isometric block art is drawable in CSS transforms or
 * flat SVG, and both make you paint the shading by hand — three faces per cube,
 * every one a colour you picked — which is a maintenance problem the moment the
 * light or the angle changes. A real orthographic camera down the (1,1,1) axis IS
 * isometric projection, and a real light gives every face of every block its own
 * value for free. The dependency is already here for the scene viewers, so this
 * costs no new bytes in the tree.
 *
 * WHY NOT drei's <ScrollControls>. It is the obvious fit and the wrong one: it
 * builds its own scrolling element and expects the scene to own the page's scroll,
 * which would put the podium in charge of a route that also has a leaderboard, a
 * footer and its own snapping. Reading a scroll position the page already has is
 * one `addEventListener` — see SnapScroller — and leaves the page in charge.
 *
 * IT STOPS WHEN IT IS OFF SCREEN. Once the reader has snapped past the hero the
 * canvas has nothing to say, and `frameloop="never"` means it stops asking for
 * frames entirely rather than re-rendering an invisible scene sixty times a
 * second. It resumes on the way back up.
 */
export default function Podium({ rows }: { rows: Row[] }) {
	const host = useRef<HTMLDivElement>(null);
	const [live, setLive] = useState(true);
	const reduced = useReducedMotion();

	useEffect(() => {
		const el = host.current;
		if (!el) return;
		const io = new IntersectionObserver(
			([entry]) => setLive(entry.isIntersecting),
			{ threshold: 0 },
		);
		io.observe(el);
		return () => io.disconnect();
	}, []);

	return (
		<div ref={host} className="absolute inset-0">
			<Canvas
				// `flat` is NoToneMapping. The scene is white geometry on black and the
				// filmic curve exists to tame highlights — here it just greys the tops
				// of the blocks, which are the brightest thing the page has.
				flat
				dpr={[1, 2]}
				frameloop={live ? "always" : "never"}
				// TRUE ISOMETRIC: equal components put the camera on the (1,1,1) axis,
				// which is the definition of the projection rather than an approximation
				// of it. Orthographic, so there is no perspective to break it.
				orthographic
				camera={{ position: [18, 18, 18], zoom: ZOOM, near: 0.1, far: 200 }}
				gl={{ antialias: true }}
			>
				{/* Key from above and one side so each cube shows three distinct values
				    — top, near face, far face — which is what makes a white box read as
				    a solid rather than as a silhouette. The fill keeps the dark side off
				    pure black so the far edges do not dissolve into the page. */}
				<ambientLight intensity={0.45} />
				<directionalLight position={[9, 16, 7]} intensity={1.9} />
				<directionalLight position={[-11, 5, -9]} intensity={0.42} />
					<Stage rows={rows} reduced={reduced} />
			</Canvas>
		</div>
	);
}
