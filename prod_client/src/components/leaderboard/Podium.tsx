"use client";

import { Html } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
	useEffect,
	useMemo,
	useRef,
	useState,
	useSyncExternalStore,
} from "react";
import * as THREE from "three";
import { inkColor } from "@/lib/ink";
import BrandMark, { brandTone } from "./BrandMark";
import { type Loose, MAX_THROW, step } from "./loose";
import { paintSweep, PLINTH_ATTRIBUTE, sweepStops } from "./sweep";
import {
	ACROSS,
	BLOCKS,
	type BlockSpec,
	CHALLENGER,
	CUBE,
	DECK,
	DEPTH,
	FOOTINGS,
	GROUP_Y,
	LABELS,
	PILLARS,
	pillarHeight,
	PLATE,
	PLATE_H,
	RISE,
	VIEW,
	viewAt,
} from "./podiumLayout";

const INTRO = 3.2;

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const easeOutCubic = (u: number) => 1 - (1 - u) ** 3;
const easeOutBack = (u: number) => {
	const c = 1.34;
	return 1 + (c + 1) * (u - 1) ** 3 + c * (u - 1) ** 2;
};

// Re-read on theme changes; the palette is editable at runtime and a colour
// captured once at mount would keep the city on whatever ink it booted with.
function useInk(): string {
	return useSyncExternalStore(
		(onChange) => {
			const mo = new MutationObserver(onChange);
			mo.observe(document.documentElement, {
				attributes: true,
				attributeFilter: ["style", "class"],
			});
			return () => mo.disconnect();
		},
		inkColor,
		inkColor,
	);
}

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

const MARGIN = 1.08;

const ZOOM = 50;

// HEADROOM FOR THE NAME PLATES, in pixels of screen rather than in model units.
//
// The plates are captions, not paint: everything drawn ON the pillars — the rank
// etched into the front face, the lab's mark laid on the top — is sized off the
// model and shrinks with it, because those are surfaces of the object. A model's
// NAME is the page talking about the object, so it holds one size the way the
// standings below it do, and the room reserved for it is therefore a pixel
// measurement too.
//
// Taken off the seat BEFORE the fit is solved rather than hoped for afterwards.
// The alternative — fit the model, then discover the plates want space the band
// has already spent — puts the winner's name behind the masthead.
const CROWN = 40;

// ROOM AT THE SIDES FOR THE SAME PLATES, for exactly the same reason and in the
// same units. A name plate is centred on its pillar and is wider than the pillar
// is — which costs nothing for the three in the middle of the island, and
// everything for the challenger, whose berth is the outermost thing on the
// model. Fitted to the island's own width, its name ran off the column.
//
// Sized on the longest name the board can raise out there rather than on a round
// number: about 120px of 15px extrabold, half of which hangs past the post.
const BEARING = 62;

// HOW FAR THE PLATE FLOATS above the point it is anchored to, which is the
// CENTRE of the pillar's top face. The face is a rhombus and its back corner
// stands proud of that centre by half the plate's own diagonal, so clearing the
// pillar means clearing that first — hence a term in model units, converted at
// the scale in force, plus a fixed gap of air above it.
const PLATE_RISE = (px: number) =>
	((PILLARS[0].width + PILLARS[0].depth) / 2) * DEPTH * px + 10;

// THE PLATES ARE OUTLINED, because there is no telling what they will be over.
//
// A name floats above its pillar in whatever the city happens to leave behind
// it — open sky at the top of the column, but rooftops the moment a back-row
// tower rises past the crown, and light ink on light ink is a name that
// disappears exactly where the skyline is busiest. A drop shadow would not fix
// it: a shadow falls on ONE side and the letters go on touching the city on the
// other three.
//
// `paint-order: stroke` is what makes a stroke usable on type at all. By default
// the stroke is painted OVER the fill and centred on the glyph's edge, so half
// of it eats into the letterform and 3px of it at this size closes up the
// counters. Painted underneath, the fill lands back on top at full weight and
// what shows is only the half that fell outside — a true outline, and the type
// keeps its shape.
const PLATE_OUTLINE = {
	WebkitTextStrokeWidth: "3px",
	WebkitTextStrokeColor: "rgb(var(--ground-rgb))",
	paintOrder: "stroke fill",
} as React.CSSProperties;

// THE FOUR SIDE FACES, as outward normals in the pillar's own frame. `+Z` is the
// one the rank was etched into when the island could not turn, and it stays the
// one it starts on.
const SIDES: [number, number][] = [
	[0, 1],
	[1, 0],
	[0, -1],
	[-1, 0],
];

// HOW MANY COPIES OF THE RANK EACH PILLAR CARRIES: one per side face. Spin the
// island and the face the number was cut into turns away — for half of every
// revolution, if there is only one of it. Cut into all four, the rank is on
// whichever side you are looking at, which is also what an etched number on a
// stone block actually does.
//
// Only ONE is ever at full strength. They cross-fade at the corners, so what a
// viewer sees is a single number that hands itself from face to face as the stone
// comes round, not four numbers taking turns.
const RANKS = SIDES.length;

// HOW MANY LABELS EACH PILLAR CARRIES — the rank cut into each of its four sides,
// the lab's mark laid on its top, and the name plate floating above it. They share
// one flat ref list, so the stride has to be named; the ranks come first, the mark
// is the second-to-last of every group and the plate the last.
const FACES = RANKS + 2;
const MARK_SLOT = RANKS;
const PLATE_SLOT = RANKS + 1;

// HOW MUCH BETTER A NEIGHBOURING FACE HAS TO BE BEFORE THE RANK MOVES TO IT.
//
// Without this the number changes face the instant the island leaves its rest
// pose: at rest the two front faces are turned through exactly the same angle, so
// a single degree either way makes one of them technically the better face and the
// rank would set off around the corner on the smallest nudge.
//
// Half a unit of facing buys about fourteen degrees of slack, which puts every
// hand-over near the middle of a quarter turn — far from where the island sits when
// nobody is touching it, and at an angle where BOTH faces are still square enough to
// the camera that a number crossing between them stays legible the whole way.
const FACE_HOLD = 0.5;

// PROJECTING A DIRECTION ONTO THE SCREEN, by hand, because these labels are HTML
// and the camera cannot help them. Everything drawn on a face of the island is a
// flat <div> skewed to lie in that face's plane, and the skew is only correct for
// the orientation it was written for — so once the island turns, the matrix has to
// be re-derived from the yaw on every frame.
//
// This is the same projection the layout module is built on, one direction at a
// time: an orthographic camera on [18,18,18] carries world X and Z ACROSS the
// screen and DOWN it in equal measure, and world Y straight up it. Screen Y counts
// DOWNWARD here, as CSS does, which is why the RISE term is negated.
const proj = (
	vx: number,
	vy: number,
	vz: number,
	c: number,
	s: number,
): [number, number] => {
	const wx = vx * c + vz * s;
	const wz = -vx * s + vz * c;
	return [(wx - wz) * ACROSS, (wx + wz) * DEPTH - vy * RISE];
};

// A face's CSS matrix is just its two in-plane axes, projected: the first pair is
// where the label's own X ends up, the second where its Y (downward) does.
const facePlate = (
	ax: [number, number, number],
	ay: [number, number, number],
	c: number,
	s: number,
): string => {
	const [a, b] = proj(ax[0], ax[1], ax[2], c, s);
	const [d, e] = proj(ay[0], ay[1], ay[2], c, s);
	return `matrix(${a.toFixed(5)}, ${b.toFixed(5)}, ${d.toFixed(5)}, ${e.toFixed(5)}, 0, 0)`;
};

// A SIDE FACE runs along the horizontal perpendicular of its own normal, and drops
// straight down the world's Y. For `+Z` at rest this comes out as the matrix that
// used to be written here by hand — (0.7071, 0.4082, 0, 0.8165) — which is the
// check that the derivation above is the same projection the island is drawn in.
const sidePlate = (nx: number, nz: number, c: number, s: number): string =>
	facePlate([nz, 0, -nx], [0, -1, 0], c, s);

// THE TOP FACE has no vertical component at all: both its axes are horizontal, so
// it is the one plate that never squashes as the island turns — it only spins.
const topPlate = (c: number, s: number): string =>
	facePlate([0, 0, -1], [1, 0, 0], c, s);

// HOW FACE-ON A SIDE IS, as the dot of its turned normal with the direction the
// camera looks from. Positive means you can see it; `SQRT2` is dead-on. Used both
// to pick the face the rank rides and to fade the ones behind it.
const facing = (nx: number, nz: number, c: number, s: number): number =>
	nx * (c - s) + nz * (c + s);

// HOW FAR THE ISLAND TURNS PER PIXEL DRAGGED. A shade under half a degree, so a
// full revolution is roughly the width of the column it sits in — the island keeps
// up with the hand without spinning away from it.
const YAW_PER_PX = 0.0075;

// WHAT IS LEFT OF A THROW AFTER A SECOND. The island keeps turning when released
// and slows on its own; there is no snap back to the isometric, so where you leave
// it is where it stays.
const SPIN_DECAY = 0.12;

// Below this the island is not visibly moving and the residue is dropped, so a
// finished throw stops asking for frames.
const SPIN_STOP = 0.02;

const ETCH = "rgb(0 0 0 / 0.42)";

// HOW FAR THE RANK SITS FROM THE TOP OF ITS PILLAR, in world units — measured DOWN
// from the top face rather than up from the base, which is the whole change.
//
// Cut near the footing, the number was the one thing on the podium that did not
// move: every pillar starts on the same deck, so the three ranks sat in a row at
// the same height whatever the standings did, while the marks they belong to were
// up at three different elevations. The rank is a property of the pillar, and the
// end of the pillar that MEANS anything is the top — it is where the height is
// read off, where the mark sits and where the eye already is. Hung off the top the
// three ranks step down with their own pillars and the number arrives next to the
// thing it labels.
//
// The value is unchanged: 0.95 was the inset from the base and is now the inset
// from the top, so the digits keep exactly the clearance from their end of the
// stone that they were drawn with.
const ETCH_INSET = 0.95;

const SWELL = 1.075;
const LIFT = 1.03;

const scratch = new THREE.Object3D();
const spot = new THREE.Vector3();
const tumble = new THREE.Euler();

function sample(s: BlockSpec, t: number): number {
	const u = clamp01((t - s.start) / (s.end - s.start));
	if (u <= 0) return 0;
	const e = easeOutCubic(u);
	spot.set(
		s.from[0] + (s.rest[0] - s.from[0]) * e,
		s.from[1] + (s.rest[1] - s.from[1]) * e,
		s.from[2] + (s.rest[2] - s.from[2]) * e,
	);
	const left = 1 - e;
	tumble.set(s.spin[0] * left, s.spin[1] * left, s.spin[2] * left);
	return easeOutCubic(clamp01(u / 0.5));
}

function fly(
	mesh: THREE.InstancedMesh | null,
	specs: BlockSpec[],
	t: number,
): void {
	if (!mesh) return;
	for (let i = 0; i < specs.length; i++) {
		const scale = sample(specs[i], t);
		if (scale <= 0) {
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

function hide(mesh: THREE.InstancedMesh | null, gone: Set<number>): void {
	if (!mesh || gone.size === 0) return;
	scratch.position.set(0, 0, 0);
	scratch.rotation.set(0, 0, 0);
	scratch.scale.setScalar(0);
	scratch.updateMatrix();
	for (const i of gone) mesh.setMatrixAt(i, scratch.matrix);
	mesh.instanceMatrix.needsUpdate = true;
}

function paint(mesh: THREE.InstancedMesh | null, list: Loose[]): void {
	if (!mesh) return;
	for (const b of list) {
		scratch.position.copy(b.pos);
		scratch.quaternion.copy(b.quat);
		scratch.scale.setScalar(1);
		scratch.updateMatrix();
		mesh.setMatrixAt(b.i, scratch.matrix);
	}
	mesh.instanceMatrix.needsUpdate = true;
}

type Row = { rank: number; name: string; lab: string; elo: number };

function Stage({
	rows,
	compare,
	reduced,
	foot,
}: {
	rows: Row[];
	compare: Row | null;
	reduced: boolean;
	foot: number;
}) {
	const ink = useInk();
	const viewport = useThree((s) => s.viewport);
	const camera = useThree((s) => s.camera);
	const gl = useThree((s) => s.gl);

	const canvas = useRef<HTMLCanvasElement | null>(null);
	useEffect(() => {
		canvas.current = gl.domElement;
	}, [gl]);

	const caster = useMemo(() => new THREE.Raycaster(), []);

	// EVERY POST ON THE ISLAND IS THE SAME FUNCTION OF THE SAME NUMBER, which is
	// what lets a challenger raised beside the podium mean anything at all. The
	// best rating on the board anchors the top of the scale, so the winner's
	// pillar is always full height and everything else is read against it.
	const best = useMemo(
		() => rows.reduce((n, r) => Math.max(n, r.elo), 1),
		[rows],
	);
	const heights = useMemo(
		() =>
			PILLARS.map((p) => {
				const row = rows.find((r) => r.rank === p.rank);
				return row ? pillarHeight(row.elo, best) : 0;
			}),
		[rows, best],
	);
	const rival = compare ? pillarHeight(compare.elo, best) : 0;

	// THE CANVAS IS THE COLUMN, so the viewport IS the room available and there is
	// nothing left to measure — an orthographic camera's viewport is the element's
	// own pixels over the zoom. The band this used to be given was a strip across
	// the top of a full-width canvas and had to be measured in the DOM and passed
	// down; a column the city owns outright reports its own size.
	const crown = CROWN / ZOOM;
	const sole = foot / ZOOM;

	// A CONTAIN FIT, both axes, because the island is an OBJECT.
	//
	// While the city was a strip this was height-only on purpose: a strip has no
	// natural width and is meant to run off the page, so fitting it to the column
	// would have shrunk it to a sliver. An island is the opposite case — it has
	// edges, and they are the point. Its coastline has to be inside the column on
	// every side or it stops reading as land floating in space and starts reading
	// as a texture that got cropped.
	//
	// So the smaller of the two ratios wins, which is the ordinary answer for
	// fitting a thing in a box, and the machinery that used to build the city out
	// sideways to order is gone with the reason for it.
	const fit = Math.max(
		1e-4,
		Math.min(
			(viewport.width - (2 * BEARING) / ZOOM) / (VIEW.w * MARGIN),
			(viewport.height - crown - sole) / (VIEW.h * MARGIN),
		),
	);

	// THE MODEL CENTRES BETWEEN ITS TWO CLEARANCES, not in the column: the crown
	// is reserved at the top for the name plates and the sole at the bottom for
	// the exit bar laid over the foot of the page. Two unequal reservations mean
	// the centre moves by half their difference.
	const lift = (sole - crown) / 2;
	const floor = -viewport.height / 2 - lift;

	const plates = useRef<THREE.InstancedMesh>(null);
	const blocks = useRef<THREE.InstancedMesh>(null);
	const pillars = useRef<(THREE.Mesh | null)[]>([]);
	const over = useRef(PILLARS.map(() => false));
	const swell = useRef(PILLARS.map(() => 0));
	const labels = useRef<(HTMLDivElement | null)[]>([]);
	const stones = useRef<(THREE.Group | null)[]>([]);
	const frame = useRef<THREE.Group>(null);
	const model = useRef<THREE.Group>(null);
	const challenger = useRef<THREE.Mesh>(null);
	const mark = useRef<THREE.Group>(null);
	const rivalLabels = useRef<(HTMLDivElement | null)[]>([]);
	const grown = useRef(0);

	const loose = useRef<Loose[]>([]);
	const dropped = useRef(new Set<number>());
	const dragging = useRef<Loose | null>(null);
	const dragPlane = useRef(new THREE.Plane());
	const dragLast = useRef(new THREE.Vector3());
	const dragTime = useRef(0);

	// WHERE THE ISLAND IS FACING, and how fast it is still turning. Refs rather
	// than state: this changes every frame while a hand is on it, and re-rendering
	// three pillars and two thousand instanced boxes to move a rotation is work
	// the render loop is already doing for free.
	const yaw = useRef(0);
	const spin = useRef(0);
	// The x the orbit gesture is measuring from, or null when nothing is turning
	// the island by hand.
	const turn = useRef<number | null>(null);
	// Which side face each pillar's rank is currently riding, and how far each of
	// the four has faded. The chosen face holds until another is clearly better —
	// see the frame loop.
	const worn = useRef(PILLARS.map(() => 0));
	const shownFace = useRef<number[][]>(
		PILLARS.map(() => SIDES.map((_, k) => (k === 0 ? 1 : 0))),
	);
	// The fit in force this frame, as a multiple of the rest fit. Labels are sized
	// in the render pass from the rest fit, so this is what keeps them in scale
	// with an island that is resizing itself as it turns.
	const fitNow = useRef(1);

	const drawn = useRef(-1);
	const settled = useRef(false);

	const intro = useRef(reduced ? 1 : 0);

	const plate = useMemo(
		() => new THREE.BoxGeometry(PLATE, PLATE_H, PLATE),
		[],
	);
	const cube = useMemo(() => new THREE.BoxGeometry(CUBE, CUBE, CUBE), []);
	const post = useMemo(() => {
		const g = new THREE.BoxGeometry(1, 1, 1);
		g.translate(0, 0.5, 0);
		return g;
	}, []);
	const skin = useMemo(
		() =>
			new THREE.MeshStandardMaterial({
				color: ink,
				roughness: 0.74,
				metalness: 0,
			}),
		[ink],
	);

	const stone = useMemo(() => {
		const m = new THREE.MeshStandardMaterial({
			color: ink,
			roughness: 0.74,
			metalness: 0,
		});
		return { material: m, ...paintSweep(m) };
	}, [ink]);

	// KEYED ON THE BUILD as well as the geometry. The footings are the last blocks
	// in the list, so a wider city moves where they start — a flag array left over
	// from a narrower one would light the sweep on whatever ordinary buildings now
	// occupy those indices, and leave the plinths flat.
	useEffect(() => {
		const flags = new Float32Array(BLOCKS.length);
		for (const foot of FOOTINGS) {
			flags.fill(1, foot.from, foot.from + foot.count);
		}
		cube.setAttribute(
			PLINTH_ATTRIBUTE,
			new THREE.InstancedBufferAttribute(flags, 1),
		);
	}, [cube]);

	// A REBUILD IS A NEW SET OF BLOCKS, so everything indexed against the old one
	// has to go with it. `drawn` is the important one: it short-circuits the
	// per-frame paint once the intro has settled, so a mesh replaced after that
	// point would never be written to at all and the city would simply vanish.
	// Anything a visitor has thrown is dropped for the same reason — a `Loose`
	// holds an instance id, and those have been renumbered.
	useEffect(() => {
		drawn.current = -1;
		loose.current = [];
		dropped.current.clear();
	}, []);

	useEffect(() => {
		const read = sweepStops();
		for (let i = 0; i < stone.stops.value.length; i++) {
			stone.stops.value[i].copy(read[i]);
		}
	}, [stone]);

	// ITS OWN MATERIAL, rebuilt when the lab changes. Sharing one of the podium's
	// would have tinted a podium pillar every time the pointer moved down the
	// standings.
	const rivalTone = useMemo(
		() =>
			new THREE.MeshStandardMaterial({
				color: compare ? brandTone(compare.lab) : "#ededed",
				roughness: 0.68,
				metalness: 0,
			}),
		[compare],
	);
	useEffect(() => () => rivalTone.dispose(), [rivalTone]);

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

	useEffect(() => {
		const el = gl.domElement;
		const ndc = new THREE.Vector2();
		const at = new THREE.Vector3();
		const flick = new THREE.Vector3();

		// THE ISLAND TURNS WHEN THE GRAB MISSED A BLOCK — but the press alone does not
		// decide that, and cannot.
		//
		// A block answers R3F's own pointer events, which are bound to this very same
		// canvas, so both handlers are listening for one `pointerdown` and which of
		// them runs first is a question about effect ordering inside the renderer
		// rather than about anything this file controls. Ask `dragging` here and the
		// answer is right only if R3F happened to register first.
		//
		// So the press only ARMS the gesture. By the time the pointer has actually
		// moved, the press is over and every handler that was going to claim it has
		// run — `dragging` is then a settled fact rather than a race. It also reads
		// better: a click that never travels no longer takes hold of the island.
		// Plain locals: the effect runs once and these listeners close over them for
		// as long as they are attached, which is exactly the lifetime wanted.
		const armed = { x: 0, id: -1, live: false };
		const down = (ev: PointerEvent) => {
			if (!settled.current) return;
			armed.x = ev.clientX;
			armed.id = ev.pointerId;
			armed.live = true;
		};

		const move = (ev: PointerEvent) => {
			if (armed.live) {
				armed.live = false;
				// A block took the press: this is a throw, not a turn.
				if (!dragging.current) {
					turn.current = armed.x;
					spin.current = 0;
					dragTime.current = ev.timeStamp;
					el.setPointerCapture(armed.id);
					el.style.cursor = "grabbing";
				}
			}

			const from = turn.current;
			if (from !== null) {
				const by = (ev.clientX - from) * YAW_PER_PX;
				turn.current = ev.clientX;
				yaw.current += by;
				// THE THROW IS THE LAST MOVEMENT, not an average of the drag. A
				// pointer that travelled a long way and stopped before letting go
				// should leave the island still, and a running mean says otherwise.
				const gap = Math.max(4, ev.timeStamp - dragTime.current) / 1000;
				dragTime.current = ev.timeStamp;
				spin.current = by / gap;
				return;
			}

			const b = dragging.current;
			const group = model.current;
			if (!b || !group) return;
			const rect = el.getBoundingClientRect();
			ndc.set(
				((ev.clientX - rect.left) / rect.width) * 2 - 1,
				-((ev.clientY - rect.top) / rect.height) * 2 + 1,
			);
			caster.setFromCamera(ndc, camera);
			if (!caster.ray.intersectPlane(dragPlane.current, at)) return;
			group.worldToLocal(at);

			const gap = Math.max(4, ev.timeStamp - dragTime.current) / 1000;
			dragTime.current = ev.timeStamp;
			flick.subVectors(at, dragLast.current).divideScalar(gap);
			dragLast.current.copy(at);
			b.vel.lerp(flick, 0.65);

			b.pos.copy(at);
			b.asleep = false;
		};

		const release = () => {
			armed.live = false;
			if (turn.current !== null) {
				turn.current = null;
				el.style.cursor = "";
				return;
			}
			const b = dragging.current;
			dragging.current = null;
			el.style.cursor = "";
			if (!b) return;
			b.held = false;
			b.vel.clampLength(0, MAX_THROW);
			const speed = b.vel.length();
			b.spin
				.set(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5)
				.multiplyScalar(speed * 0.5 + 1);
		};

		el.addEventListener("pointerdown", down);
		el.addEventListener("pointermove", move);
		el.addEventListener("pointerup", release);
		el.addEventListener("pointercancel", release);
		return () => {
			el.removeEventListener("pointerdown", down);
			el.removeEventListener("pointermove", move);
			el.removeEventListener("pointerup", release);
			el.removeEventListener("pointercancel", release);
		};
	}, [gl, camera, caster]);

	useFrame((_, delta) => {
		const dt = Math.min(delta, 1 / 30);

		if (intro.current < 1) intro.current = Math.min(1, intro.current + dt / INTRO);

		const t = intro.current;
		settled.current = t > 0.999;

		// --- the island's heading, and everything that hangs off it -------------
		if (turn.current === null && spin.current !== 0) {
			yaw.current += spin.current * dt;
			spin.current *= SPIN_DECAY ** dt;
			if (Math.abs(spin.current) < SPIN_STOP) spin.current = 0;
		}
		const cos = Math.cos(yaw.current);
		const sin = Math.sin(yaw.current);

		// THE FIT IS RE-SOLVED EVERY FRAME, because turning the island changes how
		// much room it needs. It draws up to 18% taller at three-quarters on than it
		// does at rest, which a fit solved once for the rest pose would simply push
		// through the top and bottom of the column — and the crown and the sole are
		// reserved for the name plates and the exit bar, so what it would push
		// through is the two things that must not be covered.
		//
		// The vertical CENTRE moves with it and is taken from the same table: the
		// near and far coasts trade places as the island comes round, and fitting
		// the height alone would keep it inside the column while sliding it up and
		// down inside it.
		const span = viewAt(yaw.current);
		const solved = Math.max(
			1e-4,
			Math.min(
				(viewport.width - (2 * BEARING) / ZOOM) / (span.w * MARGIN),
				(viewport.height - crown - sole) / (span.h * MARGIN),
			),
		);
		fitNow.current = solved;
		frame.current?.scale.setScalar(solved);
		if (model.current) {
			model.current.rotation.y = yaw.current;
			model.current.position.y = -span.mid / RISE;
		}
		// Labels are sized in the render pass against the REST fit, so this is the
		// correction that keeps a number cut into the stone the same size as the
		// stone while the stone is resizing.
		const grade = solved / fit;

		// HOW FAR THE LETTERING HAS ARRIVED. Every label on a pillar now carries its
		// own opacity — the ranks multiply this by how far their face has turned into
		// view — so it is read once here rather than swept over the whole list at the
		// end, which would have overwritten the cross-fade a frame after setting it.
		const named = clamp01((t - LABELS[0]) / (LABELS[1] - LABELS[0]));

		if (Math.abs(t - drawn.current) >= 0.0002) {
			drawn.current = t;
			fly(plates.current, DECK, t);
			fly(blocks.current, BLOCKS, t);
			hide(blocks.current, dropped.current);
		}

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
			mesh.scale.set(p.width * wide, heights[i] * rise, p.depth * wide);

			stones.current[i]?.scale.set(wide, rise, wide);

			// WHICH SIDE THE RANK IS ON. The best-presented face wins, but only once
			// it is clearly better than the one already carrying the number — see
			// FACE_HOLD. Ties go to the lower index, which is why the rest pose keeps
			// the number on `+Z`, the face it was cut into before any of this turned.
			let pick = 0;
			let picked = Number.NEGATIVE_INFINITY;
			for (let k = 0; k < RANKS; k++) {
				const g = facing(SIDES[k][0], SIDES[k][1], cos, sin);
				if (g > picked) {
					picked = g;
					pick = k;
				}
			}
			const held = worn.current[i];
			const heldG = facing(SIDES[held][0], SIDES[held][1], cos, sin);
			if (picked - heldG > FACE_HOLD) worn.current[i] = pick;
			const on = worn.current[i];

			// ONLY THE PAINTED ONES SWELL WITH THE PILLAR. The rank and the lab's
			// mark are on its faces and have to grow with them; the name plate is
			// a caption floating above and holds its size. It still RIDES the
			// swell, because its anchor is the pillar's top and that has moved —
			// what it must not do is change type size when a pointer crosses it.
			const worn4 = shownFace.current[i];
			for (let k = 0; k < RANKS; k++) {
				worn4[k] = THREE.MathUtils.damp(worn4[k], k === on ? 1 : 0, 12, dt);
				const el = labels.current[i * FACES + k];
				if (!el) continue;
				el.style.opacity = String(named * worn4[k]);
				// Nothing to draw and nothing to compose: a face at zero is left out
				// of the frame entirely rather than being composited transparently.
				el.style.visibility = worn4[k] < 0.004 ? "hidden" : "visible";
				el.style.scale = String(wide * grade);
				el.style.transform = sidePlate(SIDES[k][0], SIDES[k][1], cos, sin);
			}

			const face = labels.current[i * FACES + MARK_SLOT];
			if (face) {
				face.style.opacity = String(named);
				face.style.scale = String(wide * grade);
				face.style.transform = topPlate(cos, sin);
			}

			const plate = labels.current[i * FACES + PLATE_SLOT];
			if (plate) {
				plate.style.opacity = String(named);
				plate.style.transform = `translateY(calc(-50% - ${PLATE_RISE(solved * ZOOM)}px))`;
			}
		}

		// THE CHALLENGER RISES AND FALLS RATHER THAN APPEARING. Its height is
		// damped toward whatever the pointed-at model asks for — zero when the
		// pointer is off the list — so moving down the standings pumps one post up
		// and down through the skyline instead of teleporting a new one into place
		// on every row. One damped number covers all three cases: coming up out of
		// nothing, going back down to nothing, and travelling between two models.
		grown.current = THREE.MathUtils.damp(grown.current, rival, 9, dt);
		const post = challenger.current;
		if (post) {
			const up = grown.current;
			// Below a hair of height there is nothing to see and a zero scale is
			// a degenerate matrix, so it leaves rather than flattens.
			post.visible = up > 0.02 && t > LABELS[0];
			post.scale.set(CHALLENGER.width, Math.max(1e-3, up), CHALLENGER.depth);
		}
		if (mark.current) mark.current.position.y = grown.current;
		// Fades on the LAST of its travel, so the name arrives with a post that is
		// already most of the way up rather than floating ahead of it.
		const shown = rival > 0 ? clamp01((grown.current / rival) * 3 - 2) : 0;
		for (const el of rivalLabels.current) {
			if (el) el.style.opacity = String(shown);
		}

		const list = loose.current;
		if (list.length === 0) return;

		// Over the bottom edge is one way. Nothing brings these back short of a
		// remount, so they leave `loose` and stay zeroed in the instanced mesh.
		//
		// THE BLOCK IS TURNED BEFORE IT IS PROJECTED, because "how far down the
		// screen is this" stopped being a question about its model coordinates the
		// moment the island could spin. Left unrotated, a block resting on the far
		// side of a turned island reads as being metres below the frame and is
		// deleted while it is still plainly on the page.
		let lost = false;
		for (const b of list) {
			const bx = b.pos.x * cos + b.pos.z * sin;
			const bz = -b.pos.x * sin + b.pos.z * cos;
			const below =
				(b.pos.y * RISE - (bx + bz) * DEPTH - span.mid) * solved;
			if (below < floor - CUBE * solved) {
				if (dragging.current === b) dragging.current = null;
				dropped.current.add(b.i);
				lost = true;
			}
		}

		if (lost) {
			loose.current = list.filter((b) => !dropped.current.has(b.i));
			hide(blocks.current, dropped.current);
		}

		step(loose.current, dt, dropped.current);
		paint(blocks.current, loose.current);
	});

	return (
		<group position={[0, lift, 0]}>
		{/* THE FIT IS DRIVEN FROM THE FRAME LOOP, not from this render. It changes
		    with the island's heading, and re-rendering the scene sixty times a second
		    to write one scalar would cost more than the whole rotation does. The
		    value here is the rest fit, which is what it holds until something turns
		    it — and the number every label is sized against. */}
		<group ref={frame} scale={fit}>
			<group ref={model} position={[0, GROUP_Y, 0]}>
				<instancedMesh
					ref={plates}
					args={[plate, skin, DECK.length]}
					frustumCulled={false}
				/>

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
							};
							loose.current.push(b);
						}
						b.held = true;
						b.asleep = false;
						b.vel.set(0, 0, 0);

						camera.getWorldDirection(spot);
						dragPlane.current.setFromNormalAndCoplanarPoint(spot, e.point);
						dragging.current = b;
						dragLast.current.copy(b.pos);
						dragTime.current = e.nativeEvent.timeStamp;
						canvas.current?.style.setProperty("cursor", "grabbing");
						canvas.current?.setPointerCapture(e.nativeEvent.pointerId);
					}}
				/>

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

				{/* THE CHALLENGER'S BERTH — fourth on the same line, at the same
				    spacing, standing on a plinth the island always carries whether
				    anything is in it or not. Beyond the podium rather than among it:
				    the three are a result and this is a question being asked of them,
				    and putting it in the row would have made it look like fourth place
				    had been promoted.

				    It is EXCLUDED from the pointer handlers the three carry. Those
				    swell a pillar when the mouse crosses it, which would be a second,
				    conflicting answer to "what is being compared" — this post is
				    driven from the standings and nowhere else. */}
				<mesh
					ref={challenger}
					geometry={post}
					material={rivalTone}
					position={[CHALLENGER.x, CHALLENGER.base, CHALLENGER.z]}
					visible={false}
				/>

				<group
					ref={mark}
					position={[CHALLENGER.x, CHALLENGER.base, CHALLENGER.z]}
				>
					<Html
						position={[0, 0, 0]}
						center
						zIndexRange={[7, 0]}
						style={{ pointerEvents: "none" }}
					>
						<div
							ref={(el) => {
								rivalLabels.current[0] = el;
							}}
							className="font-sans text-[15px] leading-none font-extrabold tracking-[-0.01em] whitespace-nowrap text-ink"
							style={{
								opacity: 0,
								transform: `translateY(calc(-50% - ${PLATE_RISE(fit * ZOOM)}px))`,
								...PLATE_OUTLINE,
							}}
						>
							{compare?.name ?? ""}
						</div>
					</Html>

					{/* THE RANK, UNDER THE NAME, and the reason the comparison reads at
					    a glance: the height says how far off the podium this model is
					    and the number says exactly where it stands. Set in the board's
					    own readout voice so the two halves of the page are plainly
					    quoting the same list. */}
					<Html
						position={[0, 0, 0]}
						center
						zIndexRange={[7, 0]}
						style={{ pointerEvents: "none" }}
					>
						<div
							ref={(el) => {
								rivalLabels.current[1] = el;
							}}
							className="font-mono text-[10px] leading-none tracking-[0.2em] whitespace-nowrap uppercase text-ink-64"
							style={{
								opacity: 0,
								transform: `translateY(calc(-50% - ${PLATE_RISE(fit * ZOOM) - 17}px))`,
								...PLATE_OUTLINE,
							}}
						>
							{compare ? `Rank ${String(compare.rank).padStart(2, "0")}` : ""}
						</div>
					</Html>
				</group>

				{PILLARS.map((p, i) => {
					const row = rows.find((r) => r.rank === p.rank);
					if (!row) return null;
					const px = fit * ZOOM;
					return (
						<group
							key={p.rank}
							ref={(g) => {
								stones.current[i] = g;
							}}
							position={[p.x, p.base, p.z]}
						>
							{/* THE RANK, ON EVERY SIDE — anchored at the middle of each face,
							    half the pillar's own width or depth out from its axis, so
							    each copy rides its own face as the island turns. The frame
							    loop skews each to its face and fades all but the one you are
							    looking at; see FACE_HOLD. */}
							{SIDES.map(([nx, nz], k) => (
								<Html
									key={`${nx},${nz}`}
									position={[
										(nx * p.width) / 2,
										heights[i] - ETCH_INSET,
										(nz * p.depth) / 2,
									]}
									center
									zIndexRange={[7, 0]}
									style={{ pointerEvents: "none" }}
								>
									<div
										ref={(el) => {
											labels.current[i * FACES + k] = el;
										}}
										className="font-sans font-black tabular-nums"
										style={{
											opacity: 0,
											visibility: k === 0 ? "visible" : "hidden",
											transform: sidePlate(nx, nz, 1, 0),
											fontSize: `${0.82 * px}px`,
											lineHeight: 1,
											letterSpacing: "-0.03em",
											color: ETCH,
										}}
									>
										{String(row.rank).padStart(2, "0")}
									</div>
								</Html>
							))}

							<Html
								position={[0, heights[i], 0]}
								center
								zIndexRange={[7, 0]}
								style={{ pointerEvents: "none" }}
							>
								<div
									ref={(el) => {
										labels.current[i * FACES + MARK_SLOT] = el;
									}}
									style={{
										opacity: 0,
										transform: topPlate(1, 0),
										filter: "brightness(0)",
									}}
								>
									<BrandMark lab={row.lab} size={Math.round(1.05 * px)} />
								</div>
							</Html>

							{/* THE NAME PLATE. Anchored to the same point as the mark —
							    the centre of the pillar's top face — and floated clear
							    of it in screen pixels, so it rides every movement of
							    the pillar without being drawn into the isometric.

							    UPRIGHT, WHERE EVERYTHING ELSE ON THE PILLAR IS SKEWED.
							    The rank is laid on the front face and the lab's mark on
							    the top, both sheared into the projection because they
							    are painted ON the object. A name sheared the same way
							    would be unreadable at this size and, worse, would be
							    claiming to be part of the model. It is not: it is the
							    page naming what the model stands for, which is the same
							    job the standings do below — so it is set in the same
							    voice and stands square to the reader.

							    `-50%` PLUS THE RISE puts the plate's BOTTOM edge that
							    far above the anchor whatever the name's own height
							    turns out to be. Offsetting the anchor in 3D instead
							    would have tied the gap to the model's scale and closed
							    it up on a short window, which is exactly where the
							    clearance is tightest. */}
							<Html
								position={[0, heights[i], 0]}
								center
								zIndexRange={[7, 0]}
								style={{ pointerEvents: "none" }}
							>
								<div
									ref={(el) => {
										labels.current[i * FACES + PLATE_SLOT] = el;
									}}
									className="font-sans text-[15px] leading-none font-extrabold tracking-[-0.01em] whitespace-nowrap text-ink"
									style={{
										opacity: 0,
										transform: `translateY(calc(-50% - ${PLATE_RISE(px)}px))`,
										...PLATE_OUTLINE,
									}}
								>
									{row.name}
								</div>
							</Html>
						</group>
					);
				})}
			</group>
		</group>
		</group>
	);
}

export default function Podium({
	rows,
	compare = null,
	foot,
}: {
	rows: Row[];
	compare?: Row | null;
	foot: number;
}) {
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
				flat
				dpr={[1, 2]}
				frameloop={live ? "always" : "never"}
				orthographic
				camera={{ position: [18, 18, 18], zoom: ZOOM, near: 0.1, far: 200 }}
				gl={{ antialias: true }}
			>
				{/* LIT TO LAND ON INK, NOT PAST IT. The city is drawn in the theme's
				    own ink and was arriving on screen as pure white, which is not a
				    colour anyone chose: the canvas is `flat`, so nothing tone-maps
				    what comes out of the shader, and a key at 1.9 over an ambient of
				    0.45 drives a face to more than twice full brightness. Everything
				    above 1 is simply clipped, so every lit face in the city clamped
				    to #ffffff and the ink token stopped meaning anything.

				    The three now sum to about 1 on a face turned to the key, which
				    puts the brightest thing in the city exactly ON ink and leaves
				    every other face reading as shade of it rather than as grey paint.
				    A theme that moves ink now moves the city with it, which is what
				    reading the token was supposed to buy in the first place. */}
				<ambientLight intensity={0.4} />
				<directionalLight position={[9, 16, 7]} intensity={0.72} />
				<directionalLight position={[-11, 5, -9]} intensity={0.2} />
				<Stage rows={rows} compare={compare} reduced={reduced} foot={foot} />
			</Canvas>
		</div>
	);
}
