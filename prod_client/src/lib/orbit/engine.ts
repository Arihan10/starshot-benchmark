import {
	Box3,
	Color,
	DirectionalLight,
	Group,
	type Intersection,
	type Material,
	MathUtils,
	Mesh,
	MOUSE,
	type Object3D,
	PerspectiveCamera,
	Plane,
	HemisphereLight,
	type Quaternion,
	Raycaster,
	Scene,
	type ShaderMaterial,
	SphereGeometry,
	SRGBColorSpace,
	Vector2,
	Vector3,
	WebGLRenderer,
	WebGLRenderTarget,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { CopyShader } from "three/examples/jsm/shaders/CopyShader.js";
import { loadGLB } from "./loaders";
import {
	DUMMY_TEX,
	makePanoMaterial,
	makePolyMaterial,
	SPHERE_RADIUS,
} from "./materials";
import {
	CURSOR_CLEAR,
	NAV_COLORS,
	PEEK_ROTATE_SPEED,
	WASD_DIR_COS,
} from "./markers";
import {
	DEFAULT_METRICS,
	DEFAULT_SCALE,
	describeScale,
	measureSceneScale,
	type NavMetrics,
	navMetrics,
	type SceneScale,
} from "./scale";
import { SurfaceCursor } from "./cursor";
import { LightRig } from "./lighting";
import { prepareLitScene } from "./prepare";
import { MarkerLayer } from "./markerLayer";
import { SplatLayer } from "./splatLayer";
import { collectObjects, ObjectAddressing } from "./objectAddressing";
import { type PanoEntry, PanoStreamer } from "./panoTextures";
import { Projection } from "./projection";
import {
	buildMinimapState,
	levelForPosition,
	readBasis,
	toMap,
	type MapLabel,
	type MinimapSlice,
} from "./minimap";
import {
	angleDelta,
	buildNavGraph,
	type EdgeType,
	edgeVerb,
	type NavEdge,
	type NavGraph,
	type NavNode,
} from "./navGraph";
import { PASS_DUR_SCALE, planZoneTour, TourDirector } from "./tourDirector";
import {
	applyLook,
	cursorRayDir,
	forwardToLonLat,
	lookTargetFrom,
	MAX_PITCH,
	pinLook,
} from "./look";
import type {
	Chapter,
	Connector,
	NodeDir,
	OrbitMode,
	OrbitState,
	TourManifest,
	TourSource,
} from "./types";

const v3 = (a: [number, number, number]) => new Vector3().fromArray(a);

// TEMPORARY (debug). Whatever identifies a mesh well enough to find it again in the
// proxy: its own name, else the pipeline object id the addressing layer stamps on it,
// else the nearest named ancestor. Used only by `logAim`.
const describeObject = (o: Object3D): string => {
	for (let n: Object3D | null = o; n; n = n.parent) {
		const label = n.userData?.objLabel as string | undefined;
		if (n.name) return label && label !== n.name ? `${n.name} [${label}]` : n.name;
		if (label) return `[${label}]`;
	}
	return o.type;
};
const fmt3 = (v: ArrayLike<number>) =>
	`(${v[0].toFixed(2)}, ${v[1].toFixed(2)}, ${v[2].toFixed(2)})`;
const easeInOut = (t: number) =>
	t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;

// Duration constancy: a hop's length is felt through speed, not time — a 2 m and
// a 20 m walk both take the same beat. Phase is deliberately slower (a narrative
// "we are taking you through anyway"); far is a skippable flight.
const DUR: Record<EdgeType, number> = {
	walk: 700,
	portal: 900,
	phase: 1200,
	vertical: 1100,
	far: 2400,
};
const REDUCED_DUR = 260;
const DWELL_MS = 8500; // idle this long in a node → pulse the exits once

// One revolution at a tour's centrepoint. Slow is the point — it's a look around
// the room, not a spin. Under reduced motion it's slower still: discomfort tracks
// angular rate, so stretching the same turn is the gentler knob than cutting it.
const TOUR_PAN_MS = 10000;
const TOUR_PAN_MS_REDUCED = 15000;

// The two framings the viewer lives in. Every flight LERPS between them rather
// than snapping at the halfway mark, so the dollhouse framing opens continuously
// into the walkthrough's.
const OVERVIEW_FOV = 55;
const INTERIOR_FOV = 75;
// Once the fly-in has parked the camera exactly on the capture point, the pano
// dissolves in over the dollhouse across this long. Generous on purpose: both are
// the same view of the same room by then, so the only thing changing is shading
// and detail, and a slow blend reads as the room resolving rather than a cut.
const ENTER_CROSSFADE_MS = 450;
// Ceiling on how long the dissolve waits for the pano to stream in before going
// ahead anyway — the camera is parked, so waiting is invisible, but never hang.
const HANDOVER_WAIT_MS = 1000;

// --- free flight ------------------------------------------------------------
//
// Leaving the walkthrough for the splat is a DISSOLVE IN PLACE, not a flight: the
// camera is already exactly where it belongs, and the splat and the panorama are
// two renderings of the same room from the same point. So the departing pano is
// ramped off and the splat is simply already behind it — the same parked-dissolve
// reasoning `enter()` uses for the dollhouse handover. Short, because unlike that
// handover there is nothing to stream and waiting would just read as lag.
const SPLAT_REVEAL_MS = 320;
// ...but stretched when there is a ZOOM to unwind as well.
//
// Every other handover in the engine animates the field of view through startFly —
// entering the interior, exiting to the overview, and both halves of hold-to-locate
// all lerp it. This dissolve is the one that cannot, because it is the only handover
// with no flight to carry it: the camera is already exactly where it belongs.
//
// So the dissolve absorbs the lens change instead, and buys itself the room to do it
// by lasting longer in proportion to how far the lens has to travel. Unzoomed, there
// is nothing to reconcile and it stays as snappy as it is now; zoomed right in, it
// takes about twice as long — the transition costs time only when it has work to do.
const REVEAL_FOV_MS_PER_DEG = 10;
const SPLAT_REVEAL_MAX_MS = 700;
// The lens free flight settles at. Equal to INTERIOR_FOV on purpose: matching them
// means the common case (no zoom applied) needs no reconciliation at all.
const FREEFLY_FOV = INTERIOR_FOV;

// --- EXPERIMENTAL: where the cursor aims ------------------------------------
//
// true  — FIRST PERSON. The reticle is pinned to the centre of the viewport and you
//         aim it by turning the view, like a head-mounted sight. What you are looking
//         at and what you would act on are the same thing by construction, so the
//         "where will this click send me" question the ghost exists to answer
//         stops needing to be asked at all.
// false — POINTING, the original: the reticle tracks the pointer and you aim it by
//         moving the mouse, Google-Maps style.
//
// This is a single switch on purpose. Four separate things read the aim point — the
// surface reticle, the click routing, the affordance hover, and the dwell inset — and
// they have to agree or the thing you are looking at and the thing you would click
// are different. Flip this one constant and all four move together.
//
// Note the structural consequence: with the reticle centred, hover becomes a function
// of the CAMERA rather than of pointer events, so it has to be resolved every frame
// instead of on pointermove. See the interior branch of the tick.
const CENTER_CURSOR = true;
const _aim = { x: 0, y: 0 };

// --- aim reticle ------------------------------------------------------------
//
// A small crosshair at the aim point, drawn WHATEVER the ray finds — including
// nothing. The surface ring only exists where there is a surface for it to lie on, so
// sighting on open sky, or out past the edge of the scene, left the screen with no
// indication of where "here" even was.
//
// DOM rather than a three.js object, because what is being marked is a SCREEN
// position and not a place in the world: it needs no projection, stays pixel-crisp at
// any device ratio, and cannot be occluded by geometry or pushed through the
// composer's tone mapping. It also costs nothing per frame — centred by CSS, so
// nothing is written unless its opacity changes.
const RETICLE_ARM = 5; // px per tick
const RETICLE_GAP = 3; // px of clear space each side of centre
const RETICLE_THICK = 1;
// Full strength when the ray lands on something clickable, faint when it is out in
// the void — so the reticle answers "where am I pointing" and "is there anything
// there" with one mark instead of appearing and disappearing.
const RETICLE_ON_SURFACE = 0.9;
const RETICLE_IN_VOID = 0.32;

// --- mouse look (pointer lock) ----------------------------------------------
//
// Looking should not cost a held button. That needs POINTER LOCK: without it the
// pointer runs out of window after a quarter-turn, which is why the drag rig existed
// in the first place.
//
// The lock is an ENHANCEMENT, not a requirement. Held, the mouse turns the view
// directly and the centred reticle is the sight. Released, everything falls back to
// exactly the old drag-to-look — which is what keeps the chrome usable, because a
// locked pointer has no cursor to click a minimap or a floor rail with. Esc therefore
// means "give me my cursor back", not "leave", and it takes a second Esc to actually
// go anywhere.
const LOOK_SENSITIVITY = 0.002; // rad per pixel of locked movement — matches sogviewer.js
// Esc is the browser's OWN pointer-lock exit, so by the time our keydown runs the lock
// may already be gone and `locked` already false. Without a grace window that Esc
// would fall through and also leave free flight — one keypress doing two things. Same
// trick sogviewer.js uses for the same reason.
const ESCAPE_GRACE_MS = 250;

// --- zoom -------------------------------------------------------------------
//
// Zoom is MULTIPLICATIVE on the half-angle tangent — i.e. on focal length — because
// that is how zoom is perceived. The old code added degrees linearly, which is
// imperceptible at the wide end and violent at the narrow end of the same range.
//
// The range is bounded by what the panoramas can actually support, not by what the
// projection matrix allows. They are 4096x2048, so 180 degrees vertical is 2048 px:
// past roughly 45 degrees you are magnifying under 600 pano pixels across the whole
// viewport and zooming IN makes the image softer, which reads as a bug. 90 degrees is
// the other end, where rectilinear stretch at the frame edges starts to look wrong.
const ZOOM_MIN_FOV = 45;
const ZOOM_MAX_FOV = 90;
const ZOOM_PER_NOTCH = 0.12;
// Flight-speed range for the wheel in free flight. The wheel does NOT dolly there:
// W already flies forward, so a dolly would be a slower duplicate of a key the user
// is already holding. Speed is the axis a free camera actually wants, and keeping FOV
// out of free flight is what leaves exactly one place where zoom has to be reconciled.
// Kept NARROW on purpose. A 16x span (the first attempt was 0.25-4) means the floor
// is a crawl, and a control that can quietly leave the camera four times slower than
// its default is one that reads as a performance bug rather than a setting. Within
// this range even the slow end is usable.
const FREEFLY_SPEED_MIN = 0.5;
const FREEFLY_SPEED_MAX = 2;
// Coming back IS a flight (you are in open space, the destination is a capture
// point), so it reuses the enter() path wholesale — arc, FOV opening, arrival
// crossfade and all. A touch quicker than entering from the dollhouse, since the
// distances involved are usually a room rather than a whole scene.
const FREEFLY_RETURN_MS = 1150;
// Where in the return flight the interior starts asserting itself, as a fraction of
// the move. Not a taste knob: before this the camera is still far enough from the
// anchor that its panorama projects onto the proxy badly stretched, so showing it
// early would trade one artefact for another. Landing the dissolve in the final
// stretch means it is only ever visible while it is nearly correct.
const DISSOLVE_START = 0.32;
// Flight speed as a fraction of the scene's largest dimension per second, so a
// cathedral and a bathroom both take a sensible time to cross.
const FREEFLY_SPEED_FRAC = 0.18;
// There is no sprint modifier: Shift now descends. At this fraction a building-scale
// scene crosses in a few seconds anyway, so the multiplier was a convenience rather
// than a necessity — but if it is wanted back it needs a key that is safe to hold
// alongside WASD, which rules out Ctrl (Ctrl+W closes the tab).
// Velocity easing — ONE time constant, deliberately short, for both pressing and
// releasing.
//
// A long release tail was tried and removed: it gives the camera a pleasant weight
// but makes it impossible to put the eye exactly where you want it, because the
// camera keeps travelling after you have stopped asking. In a tool whose whole
// purpose is choosing a vantage, precision beats momentum. Short enough to feel
// direct, long enough not to be a hard edge.
//
// Docking therefore triggers on a genuine STOP rather than on a slow glide — with
// this curve, "stopped" is a real state that arrives promptly instead of something
// asymptotic you have to threshold.
const FREEFLY_VEL_TAU = 90; // ms

// --- docking ----------------------------------------------------------------
//
// Catching the glide as it SLOWS, not once it stops: a visitor almost never comes to
// a genuine halt before setting off again, so a zero-velocity trigger would nearly
// never fire. Slowing down near a viewpoint is the real signal that someone wants to
// look at something rather than travel past it.
//
// It is a DRIFT, not a flight. The dock only replaces the velocity TARGET the
// free-flight integrator is already easing toward, so the camera curves into the
// anchor with its velocity unbroken. Triggering a scripted flight instead would zero
// the glide and restart from a standstill — a hitch precisely where the motion is
// supposed to be seamless. It also means a keypress cancels by simply becoming the
// target again: there is no animation to interrupt.
// What counts as STOPPED, as a fraction of base top speed. With the short velocity
// tau above, the camera crosses this within a few hundred ms of the last keypress,
// so it reads as "the user has actually finished moving" rather than "the user is
// still travelling slowly".
const DOCK_STILL_SPEED_FRAC = 0.01;
// How long that stillness has to HOLD before the settle begins. This is the number
// that decides whether docking feels attentive or impatient, and it can only be
// judged by feel — so it is live-adjustable in free flight with [ and ] while a
// value is being chosen (see `dockDelayMs`). Once settled, lock it here and remove
// the keys.
const DOCK_STILL_MS = 500;
// How near an anchor has to be to catch a glide is now `metrics.dockRadius` — a
// multiple of the distance between neighbouring captures (see scale.ts). It used to
// be a fraction of the whole scene's extent, clamped to 2..5 m, and the clamp did
// all the work at both ends: a bathroom got the 2 m floor and a 200 m level got the
// 5 m ceiling, so neither actually scaled. What "near a viewpoint" means depends on
// how far apart the viewpoints are, which is the thing now being measured.
// Vertical offset is treated as EXPENSIVE, not as a cliff.
//
// It was a hard clamp at 1 m, and that was wrong in a way that read as broken
// distance detection: half a metre horizontally from an anchor — visually right on
// it — was rejected outright for being 1.1 m above, at a total distance of 1.2 m
// inside a 2.8 m radius. Nothing about that rejection is a distance, so no amount of
// standing in the right place fixed it.
//
// So vertical offset now costs WEIGHT inside the distance instead: being level with
// an anchor buys you the full radius horizontally, and rising above it spends that
// budget smoothly rather than falling off an edge. Anchors you are level with still
// win over ones you are above, which is the behaviour the clamp was reaching for.
//
// The hard cap survives, but only as a backstop against the original worry — being
// dragged down out of real mid-air. At weight 1.5 the metric rejects 2 m of rise on
// its own, so the cap almost never decides anything.
const DOCK_DY_WEIGHT = 1.5;
// The hard cap on docking to something above or below you is `metrics.dockMaxDy`.
const DOCK_SEEK_GAIN = 2.6; // 1/s — the pull toward the anchor; the integrator damps it
// `metrics.dockArrive` — close enough to hand over without visible motion.
// `metrics.dockReveal` — the distance by which the interior has fully faded in.
const DOCK_REVEAL_TAU = 150; // ms — eases the reveal BOTH ways, so a cancel fades out

// --- look inertia -----------------------------------------------------------
//
// The look rig is DIRECT MANIPULATION: pinLook solves for the angles that keep the
// grabbed point under the cursor, so lon/lat track the pointer exactly and stop the
// instant it does. That is right while dragging and wrong on release — a flick should
// coast. So the drag's angular rate is estimated as it happens and, once released,
// integrated with an exponential decay.
//
// Two time constants, because they answer different questions. GLIDE is how long a
// released flick coasts. SAMPLE is how fast the rate ESTIMATE goes stale while still
// dragging, and it is what separates a flick from a drag-and-hold: hold the pointer
// still for a moment before letting go and the estimate has already decayed to
// nothing, so the camera stays exactly where you put it.
const LOOK_GLIDE_TAU = 420; // ms
const LOOK_SAMPLE_TAU = 90; // ms
const LOOK_VEL_MIN = 0.012; // rad/s — below this, stop rather than creep forever
// What flies the camera once you are out there. Q/E are vertical here, where in
// the walkthrough they snap-turn — a rig pinned to an anchor has nowhere to rise
// to, and a rig in open space has no need to turn in 45° steps.
const FREEFLY_MOVE_KEYS = new Set([
	"KeyW",
	"KeyA",
	"KeyS",
	"KeyD",
	"KeyQ",
	"KeyE",
	"Space",
	"Shift",
]);
// ...and which of them, pressed inside the walkthrough, mean "let me fly". The four
// horizontal ones plus the primary vertical pair: asking to move in any direction is
// asking to leave. Q/E are deliberately NOT here — they are aliases for people
// already on the left of the keyboard, and they do nothing at all in the walkthrough.
const FREEFLY_ENTER_KEYS = new Set([
	"KeyW",
	"KeyA",
	"KeyS",
	"KeyD",
	"Space",
	"Shift",
]);

// Four ticks around a clear centre, so the exact pixel being aimed at is left
// visible rather than covered by the mark describing it. Centred by CSS at 50%/50%,
// which is where the aim point is in first-person mode and costs nothing to maintain.
//
// The dark drop-shadow is not decoration: a white reticle on a white wall is invisible
// otherwise, and this scene has plenty of both.
function buildReticle(): HTMLDivElement {
	const el = document.createElement("div");
	Object.assign(el.style, {
		position: "absolute",
		left: "50%",
		top: "50%",
		width: "0",
		height: "0",
		pointerEvents: "none",
		zIndex: "4", // above the travel fades, below the React chrome
		opacity: "0",
		filter: "drop-shadow(0 0 1px rgba(0,0,0,0.9))",
		transition: "opacity 120ms linear",
	});
	const off = RETICLE_GAP + RETICLE_ARM / 2; // centre of each tick, from the middle
	const ticks: Array<[number, number, number, number]> = [
		[RETICLE_THICK, RETICLE_ARM, 0, -off],
		[RETICLE_THICK, RETICLE_ARM, 0, off],
		[RETICLE_ARM, RETICLE_THICK, -off, 0],
		[RETICLE_ARM, RETICLE_THICK, off, 0],
	];
	for (const [w, h, x, y] of ticks) {
		const tick = document.createElement("div");
		Object.assign(tick.style, {
			position: "absolute",
			width: `${w}px`,
			height: `${h}px`,
			background: "#fff",
			transform: `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`,
		});
		el.appendChild(tick);
	}
	return el;
}

// One name per control, so the two physical Shift keys cannot leave each other
// latched down, and so the enter/track paths agree on what a key is called.
// Returns null for anything that is not a movement control.
function freeflyKey(code: string): string | null {
	if (code === "ShiftLeft" || code === "ShiftRight") return "Shift";
	return FREEFLY_MOVE_KEYS.has(code) ? code : null;
}

// There is deliberately NO default splat transform here.
//
// The pipeline hands Postshot a COLMAP model that is already in the repo-native
// world frame — stage5 locks it ("World: Y-up, right-handed, metres"), colmap.py
// writes `w2c = inv(c2w)` with no flip and copies cloud.ply's xyz verbatim — and
// Postshot trains from those exact poses and that exact point cloud. A splat built
// that way lands in world space, so the viewer's job is to draw it where it says it
// is, not to guess a correction.
//
// A fitted offset used to live here. It was wrong on three counts: it belonged to
// the ASSET rather than to one renderer, it was eyeballed from bounding boxes
// (which cannot even distinguish a translation from an axis flip), and it hid a
// defect in the handoff instead of fixing it. If a splat ever genuinely needs
// moving, the transform is computable in closed form against the point cloud it was
// initialized from and belongs baked into the file
// (tools/splat-to-web-sog.mjs --translate).

// Which storey a click resolves to is no longer a judgement call, because the cursor
// no longer decides it. It used to try — looking steeply UP meant "take me through
// the ceiling", looking at the ground within arm's reach meant "take me down through
// the floor" — and both were guesses at an intent nobody had expressed, read off a
// pitch angle. Floor changes belong to the arrows overhead and underfoot, which say
// what they do. The cursor moves you around the floor you are on, and crosses to
// another only when you can SEE where you would land. See `resolveAim`.

// Rest the cursor on one object this long and the walkthrough offers a proper look
// at it. Long enough that sweeping the room never triggers it, short enough to feel
// like an answer to a question you were already asking.
const INSPECT_DWELL_MS = 1750;
const INSPECT_SIZE = 190; // px — the inset's square edge
const INSPECT_GAP = 18; // px between the cursor and the inset
const INSPECT_MARGIN = 12; // px it keeps clear of the viewport edges
const INSPECT_SPIN = 0.55; // rad/s — a slow turn, not a spin

const _cursorNdc = new Vector2();
const _bez = new Vector3();
const _flyDir = new Vector3();
const _moveWish = new Vector3();
const _prevClear = new Color();
const _ghostFloor = new Vector3();
const _ovTravel = new Vector3();
const _wpDir = new Vector3();
const _wpOut = new Vector3();
const _wpEye = new Vector3();
const _dropFrom = new Vector3();
const _walkDir = new Vector3();
const _walkPt = new Vector3();
const _walkOut = new Vector3();
const _walkNrm = new Vector3();
const _walkFrom = new Vector3();
const _walkAlt = new Vector3();
// The heights the walk is sampled at, as fractions of the eye height above the
// destination's floor: shin, waist, brow. Enough to tell a doorway from a door and a
// counter from a wall, few enough to stay cheap. See `walkBackFrom`.
const WALK_HEIGHTS = [0.15, 0.5, 0.9] as const;
const _DOWN = new Vector3(0, -1, 0);
const _losFrom = new Vector3();
const _losDir = new Vector3();
const quadBezier = (
	a: Vector3,
	c: Vector3,
	b: Vector3,
	t: number,
	out: Vector3,
) => {
	const u = 1 - t;
	return out
		.copy(a)
		.multiplyScalar(u * u)
		.addScaledVector(c, 2 * u * t)
		.addScaledVector(b, t * t);
};

type Transition = {
	fromPos: Vector3;
	toPos: Vector3;
	fromQuat: Quaternion;
	toQuat: Quaternion;
	fromFov: number;
	toFov: number;
	start: number;
	dur: number;
	// Dissolve into the destination instead of swapping mid-air behind a dip to
	// black. Only worth it when the flight lands somewhere both representations
	// agree on, i.e. a capture point. See tickCrossfade.
	crossfade: boolean;
	// Dissolve the interior in DURING the flight instead of parking at the end to do
	// it. Only legal when the two representations sit on DIFFERENT canvases — the
	// splat below, three.js above — because the dissolve is then a compositing
	// operation between two independently parallax-correct images. The dollhouse
	// paths cannot use it: their departure image is drawn by three.js too, so fading
	// that canvas would fade away the very thing being dissolved from.
	dissolveInterior: boolean;
	onMid?: () => void;
	onEnd?: () => void;
	midDone: boolean;
};

// A parked dissolve between the dollhouse and the capture pano. With the camera
// ON the capture point at the walkthrough's FOV, the equirect is the same view as
// the projected interior — so ramping it either way changes only shading/detail,
// never framing. "in" (enter) fades the pano over the dollhouse; "out" (exit)
// reveals the dollhouse before the fly-out, so the capture image never rides
// along as the camera pulls away.
type Crossfade = {
	armed: number; // when the ramp began; 0 while still waiting on the texture
	deadline: number; // stop waiting for the pano and hand over regardless
	dur: number;
	direction: "in" | "out";
	onEnd?: () => void;
};
// A typed interior traversal (one edge of the nav graph). `ctrl` bends the path
// (an arc for far/vertical hops); `sphere` crossfades the backdrop in sphere-only
// tours; `dy` is the height change, which names the arrival ("up/down a level").
// `pass` marks an anchor the auto tour is only walking through, which shortens the
// hop and skips the arrival narration.
type Move = {
	fromPos: Vector3;
	toPos: Vector3;
	ctrl: Vector3 | null;
	start: number;
	dur: number;
	index: number;
	type: EdgeType;
	dy: number;
	sphere: boolean;
	pass: boolean;
};
// A destination the cursor can reach but the eye cannot see: the capture it lands
// on, the storey that capture is on, and how many storeys that is from here (0 =
// this floor, so the move is through geometry rather than up or down).
type ReachTarget = { index: number; level: number; levelDelta: number };

type SavedInterior = {
	pos: Vector3;
	lon: number;
	lat: number;
	index: number;
	fov: number;
};

// A combined dollhouse + interior walkthrough with a TYPED navigation grammar.
// OVERVIEW orbits the vertex-colored lite scene; stepping INSIDE drops into the
// pano walkthrough, where every reachable neighbour is classified into one of
// five edge types (walk / portal / vertical / phase / far), each with its own
// affordance and its own transition. All per-frame work mutates three.js / the
// canvas directly (never React), so the UI only re-renders on discrete changes.
export class OrbitEngine {
	private readonly host: HTMLElement;
	private readonly onState: (s: OrbitState) => void;
	private readonly onHold?: (held: boolean) => void;

	// --- "am I in the scene?" ---------------------------------------------------
	// Reported on its OWN channel rather than through the state stream, because the
	// state stream is deliberately frozen for the whole of a transition (see emit)
	// — and a flight is exactly when this answer changes. Anything that has to move
	// WITH the camera rather than after it reads this: the comparison page sizes its
	// panels from it, so the panel opens as the fly-in starts instead of snapping
	// open 1.1 seconds later, when the journey is already over.
	//
	// True from the moment a flight inward BEGINS until the moment a flight outward
	// begins — not from the arrivals, which is the whole point.
	private readonly onInside?: (inside: boolean) => void;
	private inside = false;

	private setInside(next: boolean) {
		if (this.inside === next) return;
		this.inside = next;
		this.onInside?.(next);
	}

	private readonly renderer: WebGLRenderer;
	private readonly canvas: HTMLCanvasElement;
	// Pending `capture()` request, served at the end of the next drawn frame.
	private captureWaiting: ((c: HTMLCanvasElement | null) => void) | null = null;
	private readonly travelFade: HTMLDivElement;
	private readonly iris: HTMLDivElement; // vertical-shaft "hatch" wipe overlay
	private readonly reticle: HTMLDivElement; // the aim crosshair
	private readonly sonarLabels: HTMLDivElement[] = []; // pooled x-ray name tags
	private readonly scene: Scene;
	private readonly camera: PerspectiveCamera;
	private readonly controls: OrbitControls;
	private readonly rig: LightRig;
	private readonly ro: ResizeObserver;

	private readonly composer: EffectComposer;

	private readonly sphereA: Mesh;
	private readonly sphereAMat: ShaderMaterial;
	private readonly sphereB: Mesh;
	private readonly sphereBMat: ShaderMaterial;
	private readonly polyMaterial = makePolyMaterial();

	private readonly streamer: PanoStreamer;
	private readonly projection = new Projection();
	private readonly markers: MarkerLayer;
	private readonly addressing: ObjectAddressing;
	private readonly director: TourDirector;
	private readonly requestPano = (i: number) => this.streamer.request(i);

	private readonly dummyCam = new PerspectiveCamera();

	private readonly cursor: SurfaceCursor;
	private readonly cursorRay = new Raycaster();
	private readonly occluder = new Raycaster(); // LOS tests for the nav graph
	private readonly dropRay = new Raycaster(); // settles a waypoint onto the floor
	private readonly walkRay = new Raycaster(); // destination -> cursor, for the marker
	// TEMPORARY (debug). The last aim the walkthrough resolved, kept so a jerk can be
	// reported with the exact numbers and the exact triangles that produced it — see
	// `logAim`, the L key, and the "log aim" button. Delete all four together.
	private aimBlock: {
		source: string;
		object: string;
		face: number;
		dist: number;
		point: [number, number, number];
		planNormal: [number, number, number] | null;
	} | null = null;
	// Pointer-lock state. `unlockedAt` timestamps the release so the Esc that caused
	// it can be told apart from a later, deliberate one — see ESCAPE_GRACE_MS.
	private locked = false;
	private unlockedAt = 0;
	// This click was spent taking hold of the pointer, so it must not also travel.
	private lockClickPending = false;
	private pointerClientX = 0;
	private pointerClientY = 0;
	private pointerInside = false;
	// Whether a button is currently held. The overview reads it to stand its cursor
	// down during an orbit-drag: the pointer is not aiming then, it is holding the
	// scene, and a ring skating across the geometry as the model turns under a
	// stationary hand reads as the cursor having come loose.
	private pointerDown = false;

	// --- the overview cursor ---------------------------------------------------
	// What a click on the dollhouse would open, resolved once per frame from the
	// surface under the pointer and reused by the click itself, so what is drawn and
	// where you land cannot disagree. -1 = the pointer is not over the scene.
	private overviewTarget = -1;
	// The surface hit the ring is currently lying on, kept so a frame in which
	// neither the pointer nor the camera moved can skip the raycast entirely.
	private overviewHit: Intersection | null = null;
	private overviewAimX = -1;
	private overviewAimY = -1;
	private readonly overviewCam = new Vector3();
	private readonly overviewPivot = new Vector3();

	private currentIndex = -1;
	// The capture the fly-in is heading to, projected during enter() before the
	// arrival is `activate`d (currentIndex is still -1 then). Cleared on arrival.
	private flyTarget = -1;
	private projectionMode = false;
	private minimaps: MinimapSlice[] = [];
	// The frame this scene's map is drawn in, and the word its storeys go by. Both
	// come from the capture profile; a tour without one is a plan view of a
	// building, which is what every tour was before the profile existed.
	private mapBasis = readBasis(undefined);
	private levelWord = "floor";
	private mapLabels: MapLabel[] = [];
	private panoLevel: number[] = [];
	private minimapPrefetch: HTMLImageElement[] = [];
	private liteRoot: Group | null = null;
	private proxyGroup: Group | null = null;
	private sharedOverview = false;
	private proxyView = false;
	private proxyColorMats: Material[] = [];
	private connectors: Connector[] = []; // parsed but not surfaced (highlights hidden for now)

	// --- Gaussian splat ---
	// The splat renders on its OWN canvas beneath this one (see splatLayer.ts), so
	// nothing here shares a context with it — the engine only feeds it a camera and
	// decides when it is on screen. `splatView` is the user-facing switch; when it
	// is off every mesh view behaves exactly as it did before the splat existed.
	private readonly splat: SplatLayer;
	private splatView = true;
	// The pano-to-splat dissolve: 0 = the walkthrough's panorama still covers the
	// view, 1 = the splat is fully uncovered. Runs alongside free flight rather
	// than blocking it, so movement responds from the first frame.
	private splatReveal = 0;
	private splatRevealing = false;
	// Duration and starting lens of the current reveal. Both are per-transition
	// because the dissolve stretches to absorb however much zoom has to be unwound.
	private splatRevealMs = SPLAT_REVEAL_MS;
	private revealFovFrom = FREEFLY_FOV;
	// Flight-speed multiplier, driven by the wheel in free flight. Deliberately reset
	// on every entry rather than persisted: it used to carry over, which meant a scroll
	// made while looking at something else left every later excursion slower with
	// nothing on screen to explain why. A speed you did not set on this trip is
	// indistinguishable from the viewer being broken.
	private freeflySpeed = 1;
	// Held movement keys + the eased velocity they drive, in world units/sec.
	private readonly freeflyKeys = new Set<string>();
	private readonly freeflyVel = new Vector3();
	// Angular velocity of the look rig, rad/sec, shared by BOTH first-person modes —
	// they are one yaw/pitch rig, so a flick coasts the same way inside the
	// walkthrough as it does in flight. Sampled from the drag, integrated on release.
	private readonly lookVel = { lon: 0, lat: 0 };
	private lookSampledAt = 0;
	// The anchor the glide is currently settling onto (-1 = none), how far the
	// interior has faded in for it, and whether docking is allowed to fire at all.
	//
	// `freeflyFrom` is the anchor free flight began on, and it is excluded from docking
	// until the camera has left its radius.
	//
	// This replaces a global "armed" flag that disabled docking entirely until you had
	// travelled more than one dock radius from where you started. That flag solved the
	// right problem — free flight begins standing ON an anchor with every dock
	// condition already satisfied, so without something you would be snapped straight
	// back and could never leave — but it solved it far too broadly: adjusting your
	// position slightly never docked you to ANY anchor, including ones you had moved
	// deliberately towards. Excluding one anchor is the narrow version of the same
	// guarantee.
	private dockTarget = -1;
	private dockReveal = 0;
	private freeflyFrom = -1;
	private dockStaged = false; // the interior is mounted behind the fade
	// When the camera came to rest (0 = it hasn't). Docking waits for this to have
	// held for `dockDelayMs`, so a momentary pause between two moves doesn't grab you.
	private dockStillSince = 0;
	// TEMPORARY tuning knob: [ and ] adjust it in free flight so the delay can be
	// judged by feel rather than guessed. Fold the chosen value into DOCK_STILL_MS
	// and delete this along with the key handling.
	private dockDelayMs = DOCK_STILL_MS;
	// --- dwell inspection ---
	// Which ids are discrete objects worth looking at (from the manifest), the
	// dollhouse copy of each (the only per-object geometry that is BOTH published
	// and coloured — the proxy is untextured and position-only), and the live inset.
	private inspectable = new Set<string>();
	private liteByLabel = new Map<string, Object3D>();
	private inspectScene: Scene | null = null;
	private inspectCam = new PerspectiveCamera(45, 1, 0.01, 100);
	private inspectPivot: Group | null = null;
	private inspect: OrbitState["inspect"] = null;
	private hoverLabel: string | null = null;
	private hoverSince = 0;
	private rcDownX = 0;
	private rcDownY = 0;

	// The typed navigation graph (built at scene load) + per-scene directory data
	// the chrome reads for chapters / search / the minimap overlay.
	private navGraph: NavGraph | null = null;
	private nodeDir: NodeDir[] = [];
	private chapters: Chapter[] = [];

	// Invariants: a back-stack that retraces the exact path (never blocked), the
	// set of nodes stood on (minimap fill), and a one-slot input buffer so
	// chained clicks queue instead of blocking.
	private history: number[] = [];
	private visited = new Set<number>();
	private pendingTravel: number | null = null;
	private arrival: OrbitState["arrival"] = null;

	private mode: OrbitMode = "empty";
	// Restored whenever the splat is not behind this layer; matches the host's CSS
	// backdrop so dropping to a transparent background never changes what you see.
	// See the note by SplatLayer's CLEAR — the same black is spelled out in four
	// places and they have to agree.
	private readonly bgColor = new Color(0x000000);
	private readonly sceneCenter = new Vector3();
	private sceneMaxDim = 1;
	// Vertical bounds of the loaded scene, so the waypoint's column cast starts
	// above everything and reaches below it.
	private sceneTopY = 0;
	private sceneBottomY = 0;
	// The scene's measured scale and every distance derived from it. Replaced on each
	// scene load; the fallback keeps a not-yet-loaded engine self-consistent.
	private sceneScale: SceneScale = DEFAULT_SCALE;
	private metrics: NavMetrics = DEFAULT_METRICS;
	private readonly browsePos = new Vector3();

	private transition: Transition | null = null;
	private move: Move | null = null;
	// The arrival dissolve a crossfading flight hands off to when it lands.
	private crossfade: Crossfade | null = null;

	private lon = 0;
	private lat = 0;
	private dragging = false;
	private dragMoved = 0;
	private downX = 0;
	private downY = 0;
	private readonly grabDir = new Vector3();
	private highlightEnabled = true;

	private interiorBusy = false;
	private savedInterior: SavedInterior | null = null;
	private peekHeld = false;
	// Hold-to-locate cuts the scene open along the same axis the map does — the
	// roof off a building, the front off a diorama — so what you see while locating
	// is the view the map is a drawing of. Its normal is set per scene, since the
	// axis is not known until one is loaded.
	private readonly locateClip = new Plane(new Vector3(0, -1, 0), 0);

	private hoveredNavIndex = -1;
	// What the cursor currently REACHES: the destination a click would take you to
	// whenever you cannot see it from here — behind geometry, or on another floor.
	// Recomputed every frame in updateCursorRing; null whenever the destination is
	// in plain sight, which is the state that needs no explaining.
	private cursorReach: ReachTarget | null = null;
	// The floor arrow under the cursor. Its preview docks in the same corner as the
	// cursor's, so all this needs to carry is the destination.
	private arrowReach: ReachTarget | null = null;
	private lastInputAt = 0;
	private dwellPulsed = false;
	private readonly reducedMotion =
		typeof window !== "undefined" &&
		window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

	private autoRotateTimer: ReturnType<typeof setTimeout> | null = null;
	private lastFrame = 0;
	private overlay: OrbitState["overlay"] = null;

	private loadToken = 0;
	private disposed = false;

	constructor(
		host: HTMLElement,
		onState: (s: OrbitState) => void,
		onHold?: (held: boolean) => void,
		onInside?: (inside: boolean) => void,
	) {
		this.host = host;
		this.onState = onState;
		this.onHold = onHold;
		this.onInside = onInside;

		// `alpha` so this canvas can be drawn OVER the splat's: when the splat is
		// showing, the scene background is dropped and everything three.js renders
		// — markers, cursor, the pano dissolve — composites onto it. With no splat
		// the opaque background is restored and this is byte-for-byte the old path.
		this.renderer = new WebGLRenderer({ antialias: false, alpha: true });
		this.renderer.setPixelRatio(window.devicePixelRatio);
		// The display transform (ACES + sRGB + shadows) is owned by LightRig below.
		this.canvas = this.renderer.domElement;
		Object.assign(this.canvas.style, {
			display: "block",
			width: "100%",
			height: "100%",
			// Explicit stacking: the splat canvas is absolutely positioned, and a
			// positioned element would otherwise paint over this one whatever the
			// DOM order.
			position: "relative",
			zIndex: "1",
		});
		host.appendChild(this.canvas);
		this.splat = new SplatLayer(host);

		this.travelFade = document.createElement("div");
		Object.assign(this.travelFade.style, {
			position: "absolute",
			inset: "0",
			background: "#0e0f12",
			opacity: "0",
			pointerEvents: "none",
			zIndex: "1",
		});
		host.appendChild(this.travelFade);

		this.iris = document.createElement("div");
		Object.assign(this.iris.style, {
			position: "absolute",
			inset: "0",
			opacity: "0",
			pointerEvents: "none",
			zIndex: "2",
		});
		host.appendChild(this.iris);

		this.reticle = buildReticle();
		host.appendChild(this.reticle);

		this.scene = new Scene();
		this.scene.background = this.bgColor;
		// Neutral IBL + hemisphere fill + a shadow-casting sun, on the same numbers
		// the panos were baked with (see lighting.ts) so the dollhouse and the
		// interior agree.
		this.rig = new LightRig(this.renderer, this.scene);

		this.camera = new PerspectiveCamera(60, 1, 0.05, 2000);
		this.camera.position.set(4, 3, 5);

		this.controls = new OrbitControls(this.camera, this.canvas);
		this.controls.enableDamping = true;
		this.controls.dampingFactor = 0.12;
		this.controls.screenSpacePanning = true;
		this.controls.zoomToCursor = true;
		this.controls.autoRotate = true;
		this.controls.autoRotateSpeed = 0.6;
		this.controls.mouseButtons = {
			LEFT: MOUSE.ROTATE,
			MIDDLE: MOUSE.PAN,
			RIGHT: MOUSE.PAN,
		};
		this.controls.enabled = false;
		this.controls.addEventListener("start", this.onControlsStart);
		this.controls.addEventListener("end", this.onControlsEnd);

		this.sphereAMat = makePanoMaterial();
		this.sphereBMat = makePanoMaterial();
		this.sphereAMat.uniforms.map.value = DUMMY_TEX;
		this.sphereBMat.uniforms.map.value = DUMMY_TEX;
		this.sphereA = new Mesh(
			new SphereGeometry(SPHERE_RADIUS, 64, 32),
			this.sphereAMat,
		);
		this.sphereB = new Mesh(this.sphereA.geometry, this.sphereBMat);
		this.sphereA.renderOrder = 0;
		this.sphereB.renderOrder = 1;
		this.sphereA.visible = false;
		this.sphereB.visible = false;
		this.scene.add(this.sphereA, this.sphereB);

		this.markers = new MarkerLayer(this.scene);
		this.cursor = new SurfaceCursor(this.scene);
		this.streamer = new PanoStreamer(
			() => this.loadToken,
			(i) => this.onPanoReady(i),
		);
		this.addressing = new ObjectAddressing(
			this.scene,
			this.camera,
			this.canvas,
		);
		this.director = new TourDirector(
			{
				busy: () => this.interiorBusy,
				hop: (index, pass) => this.traverse(index, false, pass),
				getLook: () => ({ lon: this.lon, lat: this.lat }),
				setLook: (lon, lat) => {
					this.lon = lon;
					this.lat = lat;
				},
				onProgress: () => this.emit(),
			},
			this.reducedMotion ? TOUR_PAN_MS_REDUCED : TOUR_PAN_MS,
		);

		const composerRT = new WebGLRenderTarget(1, 1, { samples: 4 });
		composerRT.texture.colorSpace = SRGBColorSpace;
		this.composer = new EffectComposer(this.renderer, composerRT);
		this.composer.setPixelRatio(this.renderer.getPixelRatio());
		this.composer.addPass(new RenderPass(this.scene, this.camera));
		this.composer.addPass(this.addressing.fillPass);
		this.composer.addPass(this.addressing.selectPass);
		this.composer.addPass(this.addressing.hoverPass);
		this.composer.addPass(new ShaderPass(CopyShader));

		this.canvas.addEventListener("contextmenu", this.onContextMenu);
		this.canvas.addEventListener("pointerdown", this.onPointerDown);
		this.canvas.addEventListener("pointermove", this.onPointerMove);
		this.canvas.addEventListener("pointerleave", this.onPointerLeave);
		this.canvas.addEventListener("pointerup", this.onPointerUp);
		this.canvas.addEventListener("wheel", this.onWheel, { passive: false });
		this.canvas.addEventListener("click", this.onClick);
		window.addEventListener("pointerup", this.onWindowPointerUp);
		window.addEventListener("keydown", this.onKeyDown);
		window.addEventListener("keyup", this.onKeyUp);
		window.addEventListener("blur", this.onWindowBlur);
		document.addEventListener("pointerlockchange", this.onPointerLockChange);
		document.addEventListener("mousemove", this.onLockedMouseMove);

		// NOTE THE FLAG. The observer records that the size changed; the TICK acts on
		// it, at the top of the frame, before anything draws.
		//
		// Resizing straight from this callback is a frame too late, and the browser's
		// frame order is why: animation callbacks run FIRST, then resize observations
		// are delivered, then the page is painted. So a resize done here reallocated
		// both backbuffers immediately after the frame had been drawn into them, and
		// what got painted was the fresh, empty one. A single resize (a window drag,
		// the old instant panel swap) lost one frame and nobody could see it. Give the
		// panel an ANIMATED width and it happens every frame for the length of the
		// animation: the scene goes black for the whole transition and reappears the
		// moment the size settles.
		this.ro = new ResizeObserver(() => {
			this.resizePending = true;
		});
		this.ro.observe(host);
		this.resize();
		this.renderer.setAnimationLoop(this.tick);
		this.emit();
	}

	dispose() {
		this.disposed = true;
		this.loadToken++;
		// No further frames will run, so anything awaiting a capture would hang.
		this.captureWaiting?.(null);
		this.captureWaiting = null;
		this.renderer.setAnimationLoop(null);
		this.ro.disconnect();
		this.controls.removeEventListener("start", this.onControlsStart);
		this.controls.removeEventListener("end", this.onControlsEnd);
		this.controls.dispose();
		this.canvas.removeEventListener("contextmenu", this.onContextMenu);
		this.canvas.removeEventListener("pointerdown", this.onPointerDown);
		this.canvas.removeEventListener("pointermove", this.onPointerMove);
		this.canvas.removeEventListener("pointerleave", this.onPointerLeave);
		this.canvas.removeEventListener("pointerup", this.onPointerUp);
		this.canvas.removeEventListener("wheel", this.onWheel);
		this.canvas.removeEventListener("click", this.onClick);
		window.removeEventListener("pointerup", this.onWindowPointerUp);
		window.removeEventListener("keydown", this.onKeyDown);
		window.removeEventListener("keyup", this.onKeyUp);
		window.removeEventListener("blur", this.onWindowBlur);
		document.removeEventListener("pointerlockchange", this.onPointerLockChange);
		document.removeEventListener("mousemove", this.onLockedMouseMove);
		this.releaseLock();
		if (this.autoRotateTimer) clearTimeout(this.autoRotateTimer);
		this.clearScene();
		this.splat.dispose();
		this.cursor.dispose();
		this.markers.dispose();
		this.rig.dispose();
		for (const pass of this.composer.passes) pass.dispose();
		this.composer.dispose();
		this.renderer.dispose();
		this.canvas.remove();
		this.travelFade.remove();
		this.iris.remove();
		this.reticle.remove();
		for (const l of this.sonarLabels) l.remove();
	}

	private get panos(): PanoEntry[] {
		return this.streamer.list;
	}

	private navNode(i: number): NavNode | null {
		return this.navGraph && i >= 0
			? (this.navGraph.nodes[i] ?? null)
			: null;
	}

	private edgeBetween(from: number, to: number): NavEdge | null {
		const node = this.navNode(from);
		return node?.all.find((e) => e.to === to) ?? null;
	}

	private noteInput() {
		this.lastInputAt = performance.now();
		this.dwellPulsed = false;
	}

	// --- pointer lock ---------------------------------------------------------

	// Requested only from real user gestures (a click, a movement key, a peek
	// release), because browsers refuse it otherwise. A refusal is not an error: the
	// drag rig is still there, so the worst case is the old behaviour.
	// Deliberately does NOT gate on the mode. The gestures that buy a lock all fire
	// while the mode is still something else — stepping inside is requested from the
	// overview, releasing a peek from `peek` — so a mode guard here would reject
	// exactly the calls that matter. onLockedMouseMove gates the effect instead, so a
	// lock held across a flight moves nothing.
	private requestLock() {
		if (this.locked) return;
		const pending = this.canvas.requestPointerLock?.() as unknown;
		if (pending instanceof Promise) pending.catch(() => {});
	}

	private releaseLock() {
		if (this.locked) document.exitPointerLock?.();
	}

	private onPointerLockChange = () => {
		const locked = document.pointerLockElement === this.canvas;
		if (locked === this.locked) return;
		this.locked = locked;
		if (locked) {
			// A drag cannot survive into a locked look: clientX/Y freeze the moment the
			// lock takes, so pinLook would keep solving against a stale pointer.
			this.dragging = false;
			this.canvas.style.cursor = "";
		} else {
			this.unlockedAt = performance.now();
		}
		this.stopLookInertia(); // 1:1 mouse look has no coast, and neither does taking hold
		this.emit();
	};

	// Turn the view by raw pointer deltas. On `document` rather than the canvas
	// because that is where the browser delivers movement for a locked pointer, and
	// mirroring PointerLockControls here keeps it working across browsers.
	//
	// Signs follow the rig: lon increases toward the "right" basis vector used by the
	// movement code, and lat is elevation, so pushing the mouse down looks down.
	private onLockedMouseMove = (ev: MouseEvent) => {
		if (!this.locked || !this.isLookMode || this.interiorBusy) return;
		this.lon += (ev.movementX || 0) * LOOK_SENSITIVITY;
		this.lat -= (ev.movementY || 0) * LOOK_SENSITIVITY;
		this.noteInput();
	};

	// Whether this Escape was spent releasing the pointer. Returns true when it was,
	// so the caller leaves the mode alone — the first Esc buys back the cursor, and
	// only a second one actually goes anywhere.
	private consumeEscape(): boolean {
		if (this.locked) {
			this.releaseLock();
			return true;
		}
		return performance.now() - this.unlockedAt < ESCAPE_GRACE_MS;
	}

	// Where the reticle sits and what a click acts on. Centred, it is the middle of
	// the viewport regardless of the pointer; otherwise it is the pointer itself.
	// Returns a shared scratch object — read it, don't keep it.
	private aim(): { x: number; y: number } {
		if (!CENTER_CURSOR) {
			_aim.x = this.pointerClientX;
			_aim.y = this.pointerClientY;
			return _aim;
		}
		const r = this.canvas.getBoundingClientRect();
		_aim.x = r.left + r.width / 2;
		_aim.y = r.top + r.height / 2;
		return _aim;
	}

	// The two first-person modes. Both are a yaw/pitch rig driven by the same
	// drag-to-look and both resolve a click through the same surface raycast — they
	// differ only in whether the camera is pinned to a capture point.
	private get isLookMode(): boolean {
		return this.mode === "interior" || this.mode === "freefly";
	}

	// Drop the coast. Called wherever the SYSTEM takes the camera — a flight, a hop,
	// a snap turn, the auto tour — because a leftover flick would either fight the
	// pose being animated to or, worse, wait out the flight frozen and then rotate
	// the view a moment after arrival, which reads as the scene twitching on its own.
	private stopLookInertia() {
		this.lookVel.lon = 0;
		this.lookVel.lat = 0;
		this.lookSampledAt = 0;
	}

	// Advance the look rig one frame and aim the camera.
	//
	// While DRAGGING, lon/lat are already exact (pinLook solved them from the
	// pointer), so nothing is integrated — but the rate estimate is decayed, which is
	// what makes holding still before release let go cleanly instead of flicking.
	// Once released the stored rate is integrated and decayed, so the view coasts.
	private tickLook(dt: number) {
		if (this.dragging) {
			const stale = Math.exp(-(dt * 1000) / LOOK_SAMPLE_TAU);
			this.lookVel.lon *= stale;
			this.lookVel.lat *= stale;
		} else if (!this.reducedMotion && !this.director.active) {
			// The tour writes these angles itself, so coasting during one would be two
			// authors fighting over the same rig.
			if (Math.hypot(this.lookVel.lon, this.lookVel.lat) > LOOK_VEL_MIN) {
				this.lon += this.lookVel.lon * dt;
				this.lat += this.lookVel.lat * dt;
				const decay = Math.exp(-(dt * 1000) / LOOK_GLIDE_TAU);
				this.lookVel.lon *= decay;
				this.lookVel.lat *= decay;
			} else {
				this.lookVel.lon = 0;
				this.lookVel.lat = 0;
			}
		}
		const wanted = this.lat;
		this.lat = applyLook(this.camera, this.lon, this.lat);
		// Pressed against the pitch limit: kill that component so the coast doesn't
		// sit there straining against a clamp it can never pass.
		if (this.lat !== wanted) this.lookVel.lat = 0;
	}

	// --- state emission (gated so chrome holds through camera flights) --------

	private emit() {
		if (this.mode === "transition") return;
		const cur =
			this.currentIndex >= 0 ? this.panos[this.currentIndex] : null;
		const node = this.navNode(this.currentIndex);
		let hover: OrbitState["hover"] = null;
		if (this.mode === "interior" && this.hoveredNavIndex >= 0) {
			const p = this.panos[this.hoveredNavIndex];
			const e = this.edgeBetween(this.currentIndex, this.hoveredNavIndex);
			hover = {
				id: p.id,
				name: p.name,
				occluded: e ? e.type !== "walk" : false,
			};
		}
		const exits =
			this.mode === "interior" && node
				? node.rendered.map((e) => ({
						index: e.to,
						type: e.type,
						name: this.panos[e.to]?.name ?? null,
						dist: e.dist,
						bearingDeg: (e.bearing * 180) / Math.PI,
					}))
				: [];
		const state: OrbitState = {
			mode: this.mode,
			panoCount: this.panos.length,
			currentId: cur ? cur.id : null,
			currentName: cur ? (cur.name ?? null) : null,
			currentIndex: this.currentIndex,
			hover,
			objectHover: this.addressing.hoverLabel,
			proxyView: this.proxyView,
			canProxyView: this.canToggleProxyView(),
			mouseLook: this.locked,
			// TEMPORARY, surfaced only so the settle delay can be read while tuning it.
			dockDelayMs: this.dockDelayMs,
			freeflySpeed: this.freeflySpeed,
			contextMenu: this.addressing.menu,
			busy: this.interiorBusy,
			overlay: this.overlay,
			exits,
			preview: this.mode === "interior" ? this.buildPreview() : null,
			reachPreview:
				this.mode === "interior" ? this.buildReachPreview() : null,
			arrival: this.mode === "interior" ? this.arrival : null,
			sonarActive: this.markers.sonarActive,
			inspect: this.mode === "interior" ? this.inspect : null,
			tour: this.director.progress,
			canGoBack: this.mode === "interior" && this.history.length > 0,
			trapped: !!node?.trapped,
			currentZone: cur?.zone ?? null,
			visited: [...this.visited],
			nodes: this.nodeDir,
			chapters: this.chapters,
			levelWord: this.levelWord,
			minimap: buildMinimapState({
				minimaps: this.minimaps,
				panos: this.panos,
				panoLevel: this.panoLevel,
				currentIndex: this.currentIndex,
				mode: this.mode,
				labels: this.mapLabels,
				// The window's unit: past a certain size the map stops fitting the
				// whole storey and starts following you, measured in capture
				// spacings so it holds the same number of reachable points on any
				// scene. See buildMinimapState.
				step: this.sceneScale.step,
			}),
		};
		this.onState(state);
	}

	// The hover preview card payload, floated at the affordance's projected screen
	// point. Your heading carries across the hop, so the thumbnail is panned to the
	// direction you're facing RIGHT NOW — the card shows what you'll actually see
	// when you land, not some other view of the room.
	private buildPreview(): OrbitState["preview"] {
		if (this.hoveredNavIndex < 0) return null;
		const p = this.panos[this.hoveredNavIndex];
		if (!p) return null;
		const e = this.edgeBetween(this.currentIndex, this.hoveredNavIndex);
		const headingU = (((this.lon / (2 * Math.PI) + 0.5) % 1) + 1) % 1;
		this.camera.updateMatrixWorld();
		const s = v3(p.position).project(this.camera);
		const rect = this.canvas.getBoundingClientRect();
		return {
			index: this.hoveredNavIndex,
			id: p.id,
			name: p.name ?? null,
			type: e?.type ?? "walk",
			dist: e?.dist ?? 0,
			screenX: rect.left + (s.x * 0.5 + 0.5) * rect.width,
			screenY: rect.top + (-s.y * 0.5 + 0.5) * rect.height,
			thumbUrl: p.placeholderUrl,
			headingU,
		};
	}

	// The preview payload for an out-of-sight destination. The chrome pans it
	// through a full 360 rather than showing the equirect flat, so what you read is
	// the room and not a warped strip.
	// The out-of-sight destination to preview: a hovered floor waypoint if there is
	// one (it names an exact capture), else whatever the cursor currently reaches.
	//
	// Titled with that CAPTURE's own name. It used to carry a floor-wide name
	// instead, to stop the title churning as the auto-home target changed under the
	// cursor — but the panel now dissolves between destinations instead of cutting,
	// so it can track the actual destination and stay readable.
	private buildReachPreview(): OrbitState["reachPreview"] {
		// A hovered arrow names one exact destination, so it outranks the cursor's
		// rolling guess at what you are pointing past.
		const arrow = this.arrowReach;
		const target = arrow ?? this.cursorReach;
		if (!target) return null;
		const p = this.panos[target.index];
		if (!p) return null;
		return {
			index: target.index,
			name: p.name ?? null,
			url: p.url,
			placeholderUrl: p.placeholderUrl,
			dist: this.camera.position.distanceTo(v3(p.position)),
			level: target.level,
			levelDelta: target.levelDelta,
		};
	}

	private showOverlay(msg: string, { spinner = true, err = false } = {}) {
		this.overlay = { msg, spinner, err };
		this.emit();
	}
	private hideOverlay() {
		this.overlay = null;
		this.emit();
	}

	// --- travel FX: the per-type transition "look" ----------------------------
	// A ground-glide blurs + dims (motion hides proxy warp). A phase tints the
	// screen blueprint-blue and runs slower — clearly synthetic, never pretending
	// the wall wasn't there. A vertical shaft irises through a hatch. Reduced-
	// motion collapses all of it to a quick dip.
	private setFx(type: EdgeType, t: number) {
		const m = Math.sin(Math.PI * MathUtils.clamp(t, 0, 1));
		if (this.reducedMotion) {
			this.canvas.style.filter = "none";
			this.travelFade.style.background = "#0e0f12";
			this.travelFade.style.opacity = (m * 0.55).toFixed(3);
			this.iris.style.opacity = "0";
			return;
		}
		const blurPx =
			type === "phase"
				? m * 9
				: type === "vertical"
					? m * 5
					: type === "far"
						? m * 8
						: m * 7;
		this.canvas.style.filter =
			blurPx > 0.002 ? `blur(${blurPx.toFixed(2)}px)` : "none";
		const tint =
			type === "phase"
				? "#0b2a44"
				: type === "far"
					? "#0a0c14"
					: "#0e0f12";
		this.travelFade.style.background = tint;
		const fadeAmp =
			type === "phase"
				? 0.6
				: type === "vertical"
					? 0.3
					: type === "far"
						? 0.6
						: 0.5;
		this.travelFade.style.opacity = (m * fadeAmp).toFixed(3);
		if (type === "vertical") {
			// Close-then-open iris = passing up/down through a hatch.
			const gap =
				Math.abs(Math.cos(Math.PI * MathUtils.clamp(t, 0, 1))) * 130;
			this.iris.style.background = `radial-gradient(circle at 50% 50%, transparent ${gap.toFixed(1)}%, #05070d ${(gap + 7).toFixed(1)}%)`;
			this.iris.style.opacity = "1";
		} else {
			this.iris.style.opacity = "0";
		}
	}
	private clearFx() {
		this.canvas.style.filter = "none";
		this.travelFade.style.opacity = "0";
		this.iris.style.opacity = "0";
	}

	// Set by the ResizeObserver, consumed at the top of the next tick.
	private resizePending = false;

	private resize() {
		const w = this.host.clientWidth;
		const h = this.host.clientHeight;
		if (w === 0 || h === 0) return;
		this.renderer.setSize(w, h, false);
		this.composer.setSize(w, h);
		this.splat.resize(); // its canvas is a sibling, not a child — size it too
		this.camera.aspect = w / h;
		this.camera.updateProjectionMatrix();
	}

	// --- input handlers -------------------------------------------------------

	private onContextMenu = (ev: MouseEvent) => {
		ev.preventDefault();
		if (this.mode !== "overview") return;
		if (
			Math.hypot(ev.clientX - this.rcDownX, ev.clientY - this.rcDownY) > 6
		)
			return;
		this.addressing.openMenu(
			ev.clientX,
			ev.clientY,
			this.activeObjectRoot(),
		);
		this.emit();
	};

	private onControlsStart = () => {
		if (this.mode !== "overview") return;
		this.controls.autoRotate = false;
		if (this.autoRotateTimer) clearTimeout(this.autoRotateTimer);
	};
	private onControlsEnd = () => {
		if (this.mode !== "overview") return;
		if (this.autoRotateTimer) clearTimeout(this.autoRotateTimer);
		this.autoRotateTimer = setTimeout(() => {
			if (this.mode === "overview") this.controls.autoRotate = true;
		}, 2500);
	};

	private onPointerDown = (ev: PointerEvent) => {
		this.pointerDown = true;
		if (ev.button === 2) {
			this.rcDownX = ev.clientX;
			this.rcDownY = ev.clientY;
		}
		// Track the press origin in every mode so a click handler can tell a genuine
		// tap from the tail of a drag (overview enter-on-click; interior look-drag).
		this.downX = ev.clientX;
		this.downY = ev.clientY;
		if (!this.isLookMode) return;
		// Clicking the view is how you take hold of it again after Esc handed the cursor
		// back. That click is spent on the lock and nothing else — travelling as well
		// would send you somewhere you were only trying to look.
		if (!this.locked) {
			this.requestLock();
			this.lockClickPending = true;
		}
		this.yieldTour(); // before the busy gate, so a click lands mid-hop too
		if (this.interiorBusy) return;
		this.noteInput();
		// Grabbing catches the coast, the way putting a hand on a spinning globe
		// stops it. Continuing to drift under a held pointer would fight pinLook,
		// which is solving for the angles that keep the grabbed point exactly under
		// the cursor.
		this.stopLookInertia();
		this.dragging = true;
		this.dragMoved = 0;
		this.grabDir.copy(
			cursorRayDir(
				this.camera,
				this.canvas,
				this.cursorRay,
				ev.clientX,
				ev.clientY,
			),
		);
		this.canvas.style.cursor = "grabbing";
		this.canvas.setPointerCapture(ev.pointerId);
	};

	private onPointerMove = (ev: PointerEvent) => {
		this.pointerClientX = ev.clientX;
		this.pointerClientY = ev.clientY;
		this.pointerInside = true;
		if (this.mode === "overview") {
			if (ev.buttons !== 0) return;
			// Whether the pointer is over the scene — and so whether this reads as
			// clickable — is resolved once a frame in updateOverviewCursor, which has
			// to run anyway because the model turns under a stationary pointer. Asking
			// again here would be a second raycast through the dollhouse per mouse
			// event to answer a question already on hand. Object highlight (when
			// enabled) is genuinely pointer-driven and still layers on top.
			const obj = this.highlightEnabled
				? this.addressing.pickAt(
						ev.clientX,
						ev.clientY,
						this.activeObjectRoot(),
					)
				: null;
			if (this.addressing.setHover(obj)) this.emit();
			return;
		}
		if (!this.isLookMode) return;
		// A locked pointer reports frozen clientX/Y — only movementX/Y move — so pinLook
		// would solve against a stale position forever. onLockedMouseMove owns the look
		// while the lock is held.
		if (this.dragging && !this.locked) {
			this.noteInput();
			const look = pinLook(
				this.camera,
				this.canvas,
				ev.clientX,
				ev.clientY,
				this.grabDir,
			);
			// Angular rate of the drag, kept for the coast on release. angleDelta
			// keeps the yaw honest across pinLook's atan2 wrap: a raw subtraction
			// there reads as a ~2π-per-frame flick and would hurl the camera.
			const at = performance.now();
			const step = (at - this.lookSampledAt) / 1000;
			if (this.lookSampledAt > 0 && step > 0.001) {
				const k = 1 - Math.exp(-(step * 1000) / LOOK_SAMPLE_TAU);
				this.lookVel.lon +=
					(angleDelta(this.lon, look.lon) / step - this.lookVel.lon) * k;
				this.lookVel.lat +=
					((look.lat - this.lat) / step - this.lookVel.lat) * k;
			}
			this.lookSampledAt = at;
			this.lon = look.lon;
			this.lat = look.lat;
			this.dragMoved = Math.max(
				this.dragMoved,
				Math.hypot(ev.clientX - this.downX, ev.clientY - this.downY),
			);
			// Drop any stale hover preview once the look starts moving.
			if (this.hoveredNavIndex !== -1) {
				this.hoveredNavIndex = -1;
				this.markers.setNavHover(null);
				this.emit();
			}
		} else if (
			this.mode === "interior" &&
			!this.interiorBusy &&
			!CENTER_CURSOR
		) {
			// Pointer-driven hover only. With a centred reticle what you are sighted on
			// changes when the CAMERA turns, not when the mouse moves, so resolving it
			// here would leave it stale through every look — the tick owns it instead.
			// Free flight has no standing affordances to hover either way.
			this.updateHover(ev.clientX, ev.clientY);
		}
	};

	private onPointerUp = () => {
		this.pointerDown = false;
		if (!this.isLookMode) return;
		this.dragging = false;
		this.canvas.style.cursor = "";
		if (this.lockClickPending) {
			this.lockClickPending = false;
			return;
		}
		if (this.dragMoved >= 5) return;
		this.noteInput();
		// Everything a click resolves against comes from the same aim point the reticle
		// is drawn at, so you can never act on something other than what you are sighted
		// on. With CENTER_CURSOR that is the middle of the frame; otherwise the pointer.
		const at = this.aim();
		if (this.mode === "freefly") {
			this.clickFromFreefly(at.x, at.y);
			return;
		}
		// A floor arrow takes the click first: it is the only thing under the cursor
		// that changes storey. Then an affordance traverses its edge; else
		// click-anywhere routing snaps to the node minimizing graph cost + angular
		// deviation.
		const arrow = this.markers.pickFloorArrow(
			at.x,
			at.y,
			this.camera,
			this.canvas,
		);
		if (arrow) {
			this.traverse(arrow.userData.to as number);
			return;
		}
		const spot = this.markers.pickNav(at.x, at.y, this.camera, this.canvas);
		if (spot) {
			this.traverse(spot.userData.to as number);
			return;
		}
		this.clickAnywhere(at.x, at.y);
	};

	private setFov(deg: number) {
		this.camera.fov = deg;
		this.camera.updateProjectionMatrix();
	}

	// Wheel deltas are not comparable between devices: a mouse notch is ~100 px, a
	// Firefox line-mode notch is ~3, a trackpad emits a continuous stream of small
	// fractions, and a page-mode notch is ~1. Normalizing to "notches" first is what
	// makes the same gesture zoom the same amount everywhere, and the clamp stops one
	// violent trackpad flick from crossing the whole range in a single event.
	private wheelNotches(ev: WheelEvent): number {
		// Normalizing the UNITS is not enough — the densities differ too. A mouse emits
		// ONE large discrete delta per detent (~100-120 px); a trackpad emits a stream
		// of small ones, so the same physical gesture arrives as roughly ten times the
		// total. Treating them alike is what let a single swipe cross an entire range.
		// Magnitude is the only signal that separates them, since both report
		// deltaMode 0.
		const perNotch =
			ev.deltaMode === 1
				? 3
				: ev.deltaMode === 2
					? 1
					: Math.abs(ev.deltaY) >= 40
						? 100 // discrete wheel detent
						: 400; // continuous trackpad stream
		return MathUtils.clamp(ev.deltaY / perNotch, -3, 3);
	}

	private onWheel = (ev: WheelEvent) => {
		if (!this.isLookMode) return;
		ev.preventDefault();
		const notches = this.wheelNotches(ev);
		if (this.mode === "freefly") {
			// Flight speed, not zoom. See the note by FREEFLY_SPEED_MIN for why the
			// wheel does not dolly here.
			this.freeflySpeed = MathUtils.clamp(
				this.freeflySpeed * Math.exp(-notches * 0.18),
				FREEFLY_SPEED_MIN,
				FREEFLY_SPEED_MAX,
			);
			this.emit();
			return;
		}
		// Multiplicative on the half-angle tangent — that is focal length, which is
		// what zoom actually is. Adding degrees linearly (the old behaviour) changes
		// almost nothing at the wide end and lurches at the narrow end.
		const half = Math.tan((this.camera.fov * Math.PI) / 360);
		const next = half * Math.exp(notches * ZOOM_PER_NOTCH);
		this.setFov(
			MathUtils.clamp(
				(Math.atan(next) * 360) / Math.PI,
				ZOOM_MIN_FOV,
				ZOOM_MAX_FOV,
			),
		);
	};

	private onClick = (ev: MouseEvent) => {
		if (this.mode !== "overview") return;
		// Ignore the click that ends an orbit-drag; only a genuine tap enters.
		if (Math.hypot(ev.clientX - this.downX, ev.clientY - this.downY) > 6)
			return;
		// The destination the cursor already resolved this frame, so the click opens
		// the capture the ring was standing on rather than re-deriving one that could
		// differ. The fallback is for a pointer that never hovered — a touch tap,
		// where there is no hover to have happened.
		const target =
			this.overviewTarget >= 0
				? this.overviewTarget
				: this.panoAtPointer(ev.clientX, ev.clientY);
		if (target >= 0) this.enter(target);
	};

	/** The capture a click at this screen point would open; -1 over empty space. */
	private panoAtPointer(clientX: number, clientY: number): number {
		const hit = this.raycastOverview(clientX, clientY);
		return hit ? this.nearestPanoTo(hit.point) : -1;
	}

	private onPointerLeave = () => {
		this.pointerInside = false;
		// A centred reticle is not the pointer's, so the pointer leaving says nothing
		// about whether it should be shown — updateCursorRing keeps owning it.
		if (!CENTER_CURSOR) this.cursor.hide();
	};
	// A release OUTSIDE the canvas ends the drag too — the button is up wherever it
	// happened, and only tracking it on our own element leaves the overview cursor
	// stood down until the next press.
	private onWindowPointerUp = () => {
		this.pointerDown = false;
		this.peekUp();
	};

	private onKeyDown = (ev: KeyboardEvent) => {
		// These listeners are on `window`, so a focused text field would otherwise have
		// its keystrokes flown into the camera — the "take me to" search box is one, and
		// with Shift now a movement control every capital letter typed there would break
		// the user out into free flight.
		const focused = document.activeElement;
		if (
			focused instanceof HTMLInputElement ||
			focused instanceof HTMLTextAreaElement ||
			(focused instanceof HTMLElement && focused.isContentEditable)
		)
			return;
		// Space is hold-to-locate, EXCEPT where it now means "fly up" — either already
		// in flight, or in the walkthrough where it breaks out into flight. On a scene
		// with no splat to fly through there is nothing to break out to, so the shortcut
		// survives untouched there; where it is taken, the ⤢ locate button still carries
		// peek.
		const spaceFlies =
			this.mode === "freefly" ||
			(this.mode === "interior" && this.canEnterFreefly());
		if (ev.code === "Space" && !ev.repeat && !spaceFlies) {
			ev.preventDefault();
			this.peekDown();
			return;
		}
		if (this.mode === "freefly") {
			// TEMPORARY: tune the settle delay by feel. Remove with `dockDelayMs`.
			if (ev.code === "BracketLeft" || ev.code === "BracketRight") {
				ev.preventDefault();
				this.dockDelayMs = MathUtils.clamp(
					this.dockDelayMs + (ev.code === "BracketRight" ? 50 : -50),
					0,
					3000,
				);
				this.emit();
				return;
			}
			if (ev.code === "Escape") {
				ev.preventDefault();
				// Two-stage exit: the first Esc hands the cursor back so the chrome is
				// reachable, and only a second one leaves free flight. Without this the
				// browser's own lock exit and our handler would both fire on one keypress.
				if (this.consumeEscape()) return;
				// Always a way back to the walkthrough, whatever the cursor is over.
				this.returnToInterior(this.nearestPanoTo(this.camera.position));
				return;
			}
			// Asking to move is asking to look: a movement key re-takes the pointer if
			// Esc had released it.
			this.requestLock();
			this.trackFreeflyKey(ev, true);
			return;
		}
		if (this.mode !== "interior") return;
		// Sonar ping (hold Tab): reveal every node through walls for a few seconds.
		if (ev.code === "Tab" && !ev.repeat) {
			ev.preventDefault();
			this.toggleSonar();
			return;
		}
		if (ev.code === "Escape") {
			// Escape unwinds the walkthrough one layer at a time: the pointer lock
			// first (same two-stage rule as free flight), then whatever transient
			// thing is running, and only once nothing is left, the walkthrough
			// itself.
			//
			// That last step is what makes the interior two-way. Clicking the scene
			// is the way IN and this is the only way OUT — the chrome that used to
			// carry an "overview" button is gone, and no other key leaves the mode,
			// so without this an entered scene is a trap.
			if (this.consumeEscape()) return;
			const sonar = this.markers.sonarActive;
			const touring = !!this.director.progress;
			this.yieldTour();
			if (sonar) {
				this.markers.hideSonar();
				this.emit();
			}
			if (!sonar && !touring) this.exit();
			return;
		}
		if (this.interiorBusy || ev.repeat) return;
		// TEMPORARY (debug). The same dump the "log aim" button does, reachable while the
		// pointer is captured by mouse-look. See `logAim`.
		if (ev.code === "KeyL") {
			ev.preventDefault();
			this.logAim();
			return;
		}
		if (ev.code === "Backspace") {
			ev.preventDefault();
			this.goBack();
			return;
		}
		// Q/E are unbound in the walkthrough. They used to snap-turn 45°, which the
		// drag-look rig makes redundant, and they are vertical-movement aliases in
		// flight — so leaving them bound here meant one key doing two unrelated things
		// either side of a mode change.
		if (ev.code.startsWith("Digit")) {
			const n = Number(ev.code.slice(5));
			if (n >= 1) {
				ev.preventDefault();
				this.jumpToLevel(n - 1);
			}
			return;
		}
		// A movement key is how you leave the walkthrough for the splat — horizontal or
		// vertical, since asking to move in any direction is asking to fly. With no
		// splat to fly through, WASD keeps its original meaning below: step to the
		// neighbouring capture point along the look bearing.
		const flyKey = freeflyKey(ev.code);
		if (flyKey !== null && FREEFLY_ENTER_KEYS.has(flyKey) && this.canEnterFreefly()) {
			ev.preventDefault();
			this.requestLock(); // this keypress is the gesture that buys the lock
			this.freeflyKeys.add(flyKey);
			this.enterFreefly();
			return;
		}
		// WASD walks the graph edge nearest the look bearing.
		const fx = Math.cos(this.lon);
		const fz = Math.sin(this.lon);
		const rx = -Math.sin(this.lon);
		const rz = Math.cos(this.lon);
		switch (ev.code) {
			case "KeyW":
				ev.preventDefault();
				this.stepToward(fx, fz);
				break;
			case "KeyS":
				ev.preventDefault();
				this.stepToward(-fx, -fz);
				break;
			case "KeyD":
				ev.preventDefault();
				this.stepToward(rx, rz);
				break;
			case "KeyA":
				ev.preventDefault();
				this.stepToward(-rx, -rz);
				break;
		}
	};
	private onKeyUp = (ev: KeyboardEvent) => {
		if (ev.code === "Space") this.peekUp();
		// Released in EVERY mode, not just free flight: a key held across the
		// transition out would otherwise stay latched and fly the camera on its own.
		this.trackFreeflyKey(ev, false);
	};

	// The held-key set behind free flight, keyed by `freeflyKey`'s canonical names.
	private trackFreeflyKey(ev: KeyboardEvent, down: boolean) {
		const code = freeflyKey(ev.code);
		if (code === null) return;
		if (!down) {
			this.freeflyKeys.delete(code);
			return;
		}
		// Space in particular MUST be swallowed, or the page scrolls under the viewer
		// every time you rise.
		ev.preventDefault();
		this.freeflyKeys.add(code);
		this.noteInput();
	}

	// Focus loss can't be seen as a key-up, so anything held becomes stuck down.
	private onWindowBlur = () => {
		this.freeflyKeys.clear();
		this.freeflyVel.set(0, 0, 0);
		this.stopLookInertia();
	};

	// Interior hover: light the affordance under the aim point + surface its preview.
	// Takes coordinates rather than an event because the aim point has two possible
	// sources — the pointer, or the centre of the frame (see `aim`).
	private updateHover(aimX: number, aimY: number) {
		if (this.currentIndex < 0) return;
		const arrow = this.markers.pickFloorArrow(
			aimX,
			aimY,
			this.camera,
			this.canvas,
		);
		this.markers.setArrowHover(arrow);
		if (arrow) {
			this.markers.setNavHover(null);
			this.canvas.style.cursor = "pointer";
			let changedArrow = this.hoveredNavIndex !== -1;
			const to = arrow.userData.to as number;
			if (!this.arrowReach || this.arrowReach.index !== to) {
				const level = this.panoLevel[to] ?? -1;
				const cur = this.panoLevel[this.currentIndex] ?? level;
				this.arrowReach = { index: to, level, levelDelta: level - cur };
				this.requestPano(to);
				changedArrow = true;
			}
			this.hoveredNavIndex = -1;
			if (this.addressing.setHover(null) || changedArrow) this.emit();
			return;
		}
		if (this.arrowReach) {
			this.arrowReach = null;
			this.emit();
		}
		const spot = this.markers.pickNav(
			aimX,
			aimY,
			this.camera,
			this.canvas,
		);
		this.markers.setNavHover(spot);
		const idx = spot ? (spot.userData.to as number) : -1;
		const obj = this.highlightEnabled
			? this.addressing.pickAt(
					aimX,
					aimY,
					this.activeObjectRoot(),
				)
			: null;
		if (idx >= 0) {
			const rendered = this.navNode(this.currentIndex)?.rendered;
			const isVertical = rendered?.some(
				(e) => e.to === idx && e.type === "vertical",
			);
			const clear =
				isVertical || this.isTargetClear(v3(this.panos[idx].position));
			this.canvas.style.cursor = clear ? "pointer" : "crosshair";
		} else if (obj) {
			this.canvas.style.cursor = "pointer";
		} else {
			// Nothing to aim at here. If the ray leaves the scene entirely, this click
			// pulls back out to the orbit (see clickAnywhere) — advertise that instead
			// of letting it happen by surprise.
			this.canvas.style.cursor = this.raycastInterior(aimX, aimY)
				? ""
				: "zoom-out";
		}
		const changed = idx !== this.hoveredNavIndex;
		this.hoveredNavIndex = idx;
		if (this.addressing.setHover(obj) || changed) this.emit();
	}

	// Every visible interior surface under a screen point, nearest first. The
	// intersect call computes and sorts the whole list anyway, so handing it all
	// back costs nothing over returning just the first.
	// What the interior rays are cast against: the proxy (plus the floor slab that
	// backs its leaks) while projecting, or the backdrop sphere in a sphere-only
	// tour. Shared by the cursor ray and the waypoint's downward drop so the two can
	// never disagree about what counts as a surface.
	private interiorTargets(): Object3D[] {
		const targets: Object3D[] = [];
		if (this.projectionMode) {
			if (this.proxyGroup) targets.push(this.proxyGroup);
			if (this.projection.proxyBase)
				targets.push(this.projection.proxyBase);
		} else {
			this.sphereA.updateMatrixWorld();
			targets.push(this.sphereA);
		}
		return targets;
	}

	private raycastInteriorAll(
		clientX: number,
		clientY: number,
	): Intersection[] {
		const targets = this.interiorTargets();
		if (targets.length === 0) return [];
		const rect = this.canvas.getBoundingClientRect();
		_cursorNdc.set(
			((clientX - rect.left) / rect.width) * 2 - 1,
			-((clientY - rect.top) / rect.height) * 2 + 1,
		);
		this.camera.updateMatrixWorld();
		this.cursorRay.setFromCamera(_cursorNdc, this.camera);
		return this.cursorRay
			.intersectObjects(targets, true)
			.filter((h) => this.hitIsPickable(h));
	}

	// Shared by every ray that asks "what is in the way": the cursor, and the walk that
	// places the marker. They have to agree — a surface the user hid is one the cursor
	// sees straight through, and a marker that stopped at it would be resting against
	// something invisible.
	private hitIsPickable(h: Intersection): boolean {
		const splat = this.splat.isActive;
		for (let o: Object3D | null = h.object; o; o = o.parent) {
			// While the splat provides the picture, the proxy is hidden — but the
			// cursor still needs it, because it is the only thing in the scene that
			// knows where the surfaces are. The walk STOPS at the roots we hid
			// ourselves, which keeps an object the USER hid inside the proxy correctly
			// skipped: that check still runs, it just never reaches the group above it.
			// Keyed on the splat being ON SCREEN, not merely switched on: inside the
			// walkthrough the proxy is genuinely visible and carries the projection,
			// and exempting it there would quietly stop honouring a proxy the user hid.
			if (splat && (o === this.proxyGroup || o === this.projection.proxyBase))
				return true;
			if (!o.visible) return false;
		}
		return true;
	}

	// Interior geometry under a screen point (proxy + floor base, or the sphere).
	private raycastInterior(
		clientX: number,
		clientY: number,
	): Intersection | null {
		return this.raycastInteriorAll(clientX, clientY)[0] ?? null;
	}

	// The dollhouse surface under a screen pixel (overview): raycast the scene root
	// and return the first hit on a shown mesh, or null over empty space. Enter-on-
	// click homes to the nearest capture point to it.
	//
	// The whole Intersection, not just the point: the cursor ring lies IN the
	// surface, so it needs the face to orient against and the distance to size
	// itself from — the same three things the interior's raycast hands back, so one
	// SurfaceCursor can serve both.
	//
	// THE MESH IS RAYCAST EVEN WHEN IT IS INVISIBLE, on exactly the terms
	// `hitIsPickable` uses inside the walkthrough: while a splat is providing the
	// picture, `setOverviewView` hides the lite mesh under it — and that mesh is
	// still the only thing in the scene that knows where the surfaces ARE. Honouring
	// its visibility here meant every cell that ships a splat, which is every cell
	// on the comparison page, answered "no geometry" to every hover and every click:
	// the dollhouse looked solid and was not clickable anywhere.
	private raycastOverview(
		clientX: number,
		clientY: number,
	): Intersection | null {
		const root = this.activeObjectRoot();
		if (!root) return null;
		// Only the SPLAT earns the exemption. A root hidden for any other reason is
		// hidden because nothing should be picking it.
		const standingIn = this.splat.isActive;
		if (!root.visible && !standingIn) return null;
		const rect = this.canvas.getBoundingClientRect();
		_cursorNdc.set(
			((clientX - rect.left) / rect.width) * 2 - 1,
			-((clientY - rect.top) / rect.height) * 2 + 1,
		);
		this.camera.updateMatrixWorld();
		this.cursorRay.setFromCamera(_cursorNdc, this.camera);
		for (const h of this.cursorRay.intersectObject(root, true)) {
			let visible = true;
			// The walk STOPS AT the root, so a mesh hidden by the splat swap does not
			// veto its own descendants — while anything the USER hid inside the scene
			// still does, because that check sits below the root and runs as before.
			for (let o: Object3D | null = h.object; o && o !== root; o = o.parent)
				if (!o.visible) {
					visible = false;
					break;
				}
			if (visible) return h;
		}
		return null;
	}

	// Whether this tour's floors carry described volumes at all. Older captures
	// don't, and everything that reads a floor from geometry falls back to the
	// nearest-capture-point reading when they don't.
	private get hasFloorVolumes(): boolean {
		return this.minimaps.some((m) => !!m.volume);
	}

	// Which floor a world point is ON, by testing it against the floors' described
	// volumes (smallest wins where they overlap, so a mezzanine inside a taller
	// storey claims its own space). -1 means the point is on NO floor — terrain, a
	// cliff face, scenery, the slab between two storeys — which is a real answer,
	// not a failure: those things belong to no storey and the walkthrough must not
	// offer to travel to one on their behalf.
	private floorAt(p: Vector3): number {
		let best = -1;
		let bestVolume = Infinity;
		for (const mm of this.minimaps) {
			const v = mm.volume;
			if (!v) continue;
			const [ox, oy, oz] = v.origin;
			const [dx, dy, dz] = v.dimensions;
			if (p.x < ox || p.x > ox + dx) continue;
			if (p.y < oy || p.y > oy + dy) continue;
			if (p.z < oz || p.z > oz + dz) continue;
			const volume = dx * dy * dz;
			if (volume < bestVolume) {
				bestVolume = volume;
				best = mm.level;
			}
		}
		return best;
	}

	// The pano that a floor click would snap to: the node minimizing (distance to
	// the hit point + angular deviation from the click bearing). Shared by
	// clickAnywhere (the actual traversal) and updateCursorRing (the live preview).
	//
	// `floor`, when given, SCOPES the search to anchors on that storey — the floor
	// the geometry under the cursor actually belongs to. Without it, clicking the
	// ground of the storey below through a gap could still resolve to an anchor on
	// your own floor (or vice versa) purely because it was closer in space, so the
	// preview and the destination could name different floors. A storey with no
	// eligible anchor of its own falls back to the whole set rather than making the
	// click do nothing.
	// `exclude` is the anchor a click may not resolve to — the one you are standing
	// on, since "travel to where you already are" is not an answer. Free flight
	// passes -1: you have left that anchor behind, and flying back to it is a
	// perfectly reasonable thing to ask for.
	private autoHomeTarget(
		hit: Intersection,
		floor = -1,
		exclude: number = this.currentIndex,
	): number {
		const cam = this.camera.position;
		const clickBearing = Math.atan2(
			hit.point.z - cam.z,
			hit.point.x - cam.x,
		);
		// Straight up or straight down, the hit point sits on top of the eye in plan
		// and that bearing is atan2 of two near-zeroes — noise. Weighting it then
		// silently drags the choice toward whichever anchor happens to lie at bearing
		// zero, which is what made looking at the ceiling feel erratic. Fall back to
		// pure distance when there is no horizontal direction to read.
		const directional =
			Math.hypot(hit.point.x - cam.x, hit.point.z - cam.z) > 0.5;
		let best = -1;
		let bestCost = Infinity;
		for (let i = 0; i < this.panos.length; i++) {
			if (i === exclude) continue;
			if (floor >= 0 && this.panoLevel[i] !== floor) continue;
			const pp = v3(this.panos[i].position);
			const d = pp.distanceTo(hit.point);
			const bearing = Math.atan2(pp.z - cam.z, pp.x - cam.x);
			const ang = Math.abs(angleDelta(clickBearing, bearing));
			const cost = d + (directional ? ang * 3 : 0); // 1 rad off ≈ 3 m of detour
			if (cost < bestCost) {
				bestCost = cost;
				best = i;
			}
		}
		return best < 0 && floor >= 0
			? this.autoHomeTarget(hit, -1, exclude)
			: best;
	}

	// WHICH CAPTURE DID THEY MEAN. Two things, and neither is a guess: the point the
	// pointer is resting on, settled onto the floor, and WHICH WAY IS THROUGH — the
	// surface's own plan normal turned away from the camera, or null where the surface
	// is level and has no through.
	//
	// It used to displace the point instead: a stride along that normal, on the theory
	// that aiming at a wall means the room beyond it. The theory is right and the
	// method was not. Moving the measuring point throws away the one fact that
	// discriminates between captures — how near each is to what the pointer is on — and
	// replaces it with proximity to a spot a stride inside the masonry, where
	// everything within a metre becomes a tie. Measured on this house: displaced, "Upper
	// Hall Gallery" sat at 1.043 m and "Upper Circulation Spine" at 1.064 m, ELEVEN
	// MILLIMETRES apart, and a two-centimetre twitch of the pointer threw the
	// destination two metres across the house. From the pointer itself those two are
	// 0.87 m and 2.63 m away — a margin nothing can twitch across.
	//
	// So the direction is handed back AS a direction, and `resolveAim` uses it to
	// prefer the far side rather than to move the ruler.
	private aimQuery(
		hits: Intersection[],
	): { point: Vector3; through: Vector3 | null } | null {
		if (hits.length === 0) return null;
		const near = hits[0];
		const m = this.metrics;
		// Which side of the surface we are asking about. Looking UP at something means
		// we want to be UNDER it — aim at a carport roof and the answer is the floor
		// beneath, not the roof — so the probe goes below the hit; anything else stays
		// on top of what was aimed at.
		const lookingUp = near.point.y > this.camera.position.y;
		const probeY = near.point.y + (lookingUp ? -m.probeEps : m.probeEps);
		// The surface's own normal, flattened into plan and turned to point AWAY from
		// the camera — the direction "through" it. Its length is how lateral the
		// surface is, and so how much of a stride the step is worth.
		let step = 0;
		_wpDir.set(0, 0, 0);
		if (near.face) {
			_wpDir
				.copy(near.face.normal)
				.transformDirection(near.object.matrixWorld);
			// Turned to point along the view rather than back at it, using the ray that
			// actually found this surface — not the camera's forward axis, which is only
			// the same thing while the aim sits dead centre.
			const c = this.camera.position;
			if (
				_wpDir.x * (near.point.x - c.x) +
					_wpDir.y * (near.point.y - c.y) +
					_wpDir.z * (near.point.z - c.z) <
				0
			)
				_wpDir.negate();
			_wpDir.y = 0;
			const lateral = Math.min(1, _wpDir.length());
			step = lateral * m.wpStandoff;
			if (lateral > 1e-4) _wpDir.divideScalar(_wpDir.length());
			else _wpDir.set(0, 0, 0);
		}
		_wpOut.set(
			near.point.x,
			this.settleAt(near.point.x, near.point.z, probeY),
			near.point.z,
		);
		return { point: _wpOut, through: step > 1e-4 ? _wpDir : null };
	}

	// The height a waypoint settles to at one spot: the surface a person would be
	// standing on there, read off the column of solids above and below it.
	private settleAt(
		x: number,
		z: number,
		probeY: number,
		known?: ReturnType<OrbitEngine["columnAt"]>,
	): number {
		const spans = known ?? this.columnAt(x, z);
		// Buried in something — stand on top of it rather than inside it.
		const inside = spans.find((sp) => probeY > sp.bottom && probeY < sp.top);
		if (inside) return inside.top;
		// Otherwise the highest surface you could stand on at or below the probe.
		// `probeY` is already nudged to the correct side, so one comparison covers
		// both looking up and looking down.
		for (const sp of spans) if (sp.standable && sp.top <= probeY) return sp.top;
		// Nothing beneath: take the LOWEST standable surface above instead, and
		// failing even that leave the point floating — the honest answer off the
		// edge of a level.
		let above: number | null = null;
		for (const sp of spans) if (sp.standable) above = sp.top;
		return above ?? probeY;
	}

	// Everything one aim resolves to: which capture a click lands on, whether you can
	// see it, and where to draw the marker. ONE function, because the live cursor and
	// the click have to agree exactly — two ways of choosing a destination is two
	// answers that can differ, and the one on screen would be the wrong one.
	//
	// THE ORDER IS THE WHOLE DESIGN. The destination is chosen FIRST, and the marker is
	// then derived from it. Every version of this that drew the marker independently —
	// at the point past the surface under the pointer — eventually put it somewhere the
	// click did not go, because two things chosen separately drift apart no matter how
	// many tests are stacked on top. Resting on a bed marked one storey while the click
	// crossed a room on another. Derivation is not a stricter test; it removes the
	// second decision entirely.
	//
	// So: pick the capture, then WALK BACK from it toward the cursor and stop where the
	// geometry does. See `walkBackFrom`. The marker still tracks the pointer, because
	// the walk aims at the pointer, but it can only ever come to rest in the
	// destination's own space — it started there and never crossed a surface to leave.
	//
	// Nothing here can refuse. A marker that is not drawn and a click that does nothing
	// is the worst answer available: the pointer is over real geometry, the user has
	// asked to go there, and silence reads as broken. Every aim at a surface resolves
	// to some capture; when the first choice is disallowed the rule that disallowed it
	// picks the replacement.
	//
	// `occluded` — the amber cursor, and whether the marker is drawn — means exactly
	// one thing: the capture this click lands on is not currently in view.
	//
	// The returned marker is shared scratch — valid until the next aim is resolved,
	// never to be retained.
	private resolveAim(hits: Intersection[]): {
		marker: Vector3;
		occluded: boolean;
		index: number;
	} | null {
		if (hits.length === 0) return null;
		// WHICH CAPTURE. Measured from the pointer's own position — see `aimQuery` for
		// why nothing is displaced — at the height a visitor standing there would be,
		// since captures are recorded at eye height and measuring from a floor point
		// handicaps your own storey by that height while the one below pays nothing.
		//
		// AIMING AT A SURFACE MEANS THE FAR SIDE OF IT, so a capture through the surface
		// is preferred over a nearer one on this side. Preferred, not forced: it may be
		// at most one typical hop further away (`wpThrough`), because past that it is
		// not what the pointer meant, only the first thing the far half of the world
		// happened to contain. Aiming at the back wall of a study, the far side holds
		// the garden — 4.33 m out and a storey down — against 1.64 m for the landing
		// behind you, and the landing is the honest answer.
		const query = this.aimQuery(hits);
		if (!query) return null;
		const eye = this.standingEye(query.point);
		let index = this.nearestPanoTo(eye, this.currentIndex);
		if (index >= 0 && query.through) {
			const far = this.nearestPanoBeyond(
				eye,
				query.point,
				query.through,
				this.currentIndex,
			);
			if (
				far >= 0 &&
				eye.distanceTo(v3(this.panos[far].position)) <=
					eye.distanceTo(v3(this.panos[index].position)) +
						this.metrics.wpThrough
			)
				index = far;
		}
		if (index < 0) return null;
		let clear = this.isTargetClear(v3(this.panos[index].position));

		// THE FLOOR RULE. Cross a storey only when you can SEE the spot you would land
		// on — an open mezzanine below, a gallery above. Otherwise this is the arrows'
		// job, and the click stays on the floor you are standing on. Re-picked, never
		// refused. A floor holding no capture but the one under your feet keeps the
		// original answer rather than leaving the click inert.
		const curLevel =
			this.currentIndex >= 0 ? (this.panoLevel[this.currentIndex] ?? -1) : -1;
		if (!clear && curLevel >= 0 && (this.panoLevel[index] ?? -1) !== curLevel) {
			const pinned = this.nearestPanoTo(
				eye,
				this.currentIndex,
				curLevel,
			);
			if (pinned >= 0) {
				index = pinned;
				clear = this.isTargetClear(v3(this.panos[index].position));
			}
		}
		return {
			marker: this.walkBackFrom(index, hits[0]),
			occluded: !clear,
			index,
		};
	}

	// A hit's face normal, flattened into plan and turned to face the walker, written
	// into `out`. False when the face is level — it has no plan normal, so there is
	// nothing for it to push the marker along.
	private planNormal(h: Intersection, dir: Vector3, out: Vector3): boolean {
		if (!h.face) return false;
		out.copy(h.face.normal).transformDirection(h.object.matrixWorld);
		if (out.dot(dir) > 0) out.negate();
		out.y = 0;
		if (out.lengthSq() < 1e-8) return false;
		out.normalize();
		return true;
	}

	// WHERE THE MARKER STANDS: walk from the destination toward the cursor and stop
	// where the geometry stops you.
	//
	// This is what keeps the marker honest without pinning it to the capture and losing
	// the tracking. It begins inside the destination's own space and moves toward the
	// pointer, so it can only leave that space by passing through a surface — and it
	// never does, because a surface is exactly what ends the walk. Whatever it settles
	// on is somewhere a person standing at that capture could walk to in a straight
	// line, which is the strongest sense in which a marker can be "where you are going"
	// while still following your cursor.
	//
	// It also solves the case that broke the old placement outright. Your cursor is on
	// the SIDE of the next platform. Stepping past that side face lands the marker
	// inside the platform, or out the far end of it twenty metres away. Walking back
	// from the capture — which stands on TOP — the first thing met is the platform's
	// top surface, near the edge you are pointing at. Which is the answer a person
	// would give.
	//
	// THE MARKER MUST NOT KNOW WHICH CAPTURE IT CAME FROM. The walk starts at one, but
	// where it comes to rest has to be a fact about the geometry and the pointer alone,
	// because the capture underneath changes discretely — three captures on a carpet
	// behind one glass pane, and the nearest one flips as the pointer moves. Anything
	// in the answer that depends on which capture is current becomes a jump with
	// nothing on screen to explain it. Only a real edge in the geometry may move the
	// marker abruptly: the end of a wall, the lip of a platform, a doorway.
	//
	// The capture therefore decides WHICH surface stops the walk, and the surface
	// decides everything after that. See the clearance below, which is where this was
	// getting broken.
	private walkBackFrom(index: number, cursorHit: Intersection): Vector3 {
		const m = this.metrics;
		const from = v3(this.panos[index].position);
		const ground = from.y - m.eyeHeight; // the destination's own floor
		// THE WALK IS TAKEN IN PLAN, at the destination's own height. Following the
		// line to the cursor in 3D lets the walk climb or dive: aiming down through a
		// stairwell opening, nothing stops a descending ray and the marker comes to
		// rest a storey below the capture it was supposed to be describing. Held level,
		// the walk can only ever end somewhere on the destination's own floor, which is
		// the property the marker needs. (`aimQuery` steps in plan for the same reason;
		// it is the same mistake in both places.)
		_walkDir.copy(cursorHit.point).sub(from);
		_walkDir.y = 0;
		const dist = _walkDir.length();
		if (dist < 1e-6) return _walkOut.copy(from).setY(ground);
		_walkDir.divideScalar(dist);

		// A PERSON, NOT A LINE. One ray at the destination's eye asks whether a pencil
		// could reach the pointer, which is not the question — and it answers
		// differently from one pixel to the next. A doorway is open at head height and
		// shut at waist height; a counter is the reverse. Sampled at one height only,
		// detection flickers as the pointer creeps along a surface, and the whole
		// clearance flickers with it: 8.6 cm of pointer travel measured 47.5 cm of
		// marker. Sampling the height a walker actually occupies and taking the NEAREST
		// hit makes the answer a property of the geometry rather than of which slice of
		// it one ray happened to catch.
		const targets = this.interiorTargets();
		let blocked: Intersection | undefined;
		for (let i = 0; targets.length > 0 && i < WALK_HEIGHTS.length; i++) {
			_walkFrom.set(from.x, ground + m.eyeHeight * WALK_HEIGHTS[i], from.z);
			this.walkRay.set(_walkFrom, _walkDir);
			this.walkRay.near = 0;
			this.walkRay.far = dist;
			const h = this.walkRay
				.intersectObjects(targets, true)
				.find((x) => this.hitIsPickable(x));
			if (h && (!blocked || h.distance < blocked.distance)) blocked = h;
		}
		// The blocking surface's normal, FLATTENED INTO PLAN and pointed back the way
		// the walk came. Flattened because the walk is level and the height is locked,
		// so only the horizontal part of a surface can push the marker anywhere; a
		// sloped face otherwise contributes a shortened sideways push and a vertical
		// one that is thrown away, which quietly under-clears the surface. A face with
		// no horizontal part at all — the walk grazing a floor along its length — is
		// not a barrier in plan and does not move the marker.
		// WHAT THE MARKER HAS TO STAND CLEAR OF. Whatever stopped the walk — and
		// failing that, the surface under the pointer itself, which is a barrier every
		// bit as real: point at a door and the marker belongs on the destination's side
		// of it, whether or not a level ray happened to catch it on the way.
		//
		// That fallback is the whole fix for a jerk that was trivial to trigger. The
		// walk is one thin ray at the DESTINATION's height, while the pointer is
		// somewhere else on the same surface — low on a door, high on a wall — so at
		// head height it can sail clean through a doorway the pointer is nowhere near.
		// Detection then flickers on and off between adjacent pixels and the whole
		// clearance flickers with it: a measured 8.6 cm of pointer travel threw the
		// marker 47.5 cm, which is the clearance exactly. With the fallback both
		// answers describe the SAME barrier, so they differ by that surface's thickness
		// — centimetres — instead of by all or nothing.
		// ONLY WHAT THE WALK MET may move the marker. Treating the surface under the
		// pointer as a barrier in its own right — a stopgap for the flickering above —
		// is wrong, and worse than the flicker it was patching: the pointer lands on
		// whatever happens to be under it, and most of that is not a barrier at all. A
		// studio area rug two centimetres off the floor and a tray on a hall console
		// each threw the marker a full clearance along the arbitrary horizontal normal
		// of a sliver on their edge, planting it in front of the pointer while the
		// click carried on into the next room. You walk over a rug. Sampling the walk
		// properly is what that stopgap was standing in for.
		const pushable = !!blocked && this.planNormal(blocked, _walkDir, _walkNrm);
		// The pointer's own plan position, which is where the marker goes when nothing
		// stands in the way.
		_walkPt.copy(from).addScaledVector(_walkDir, dist);
		let planNormal: [number, number, number] | null = null;
		if (pushable) {
			// Dropped perpendicularly onto the blocking surface and held a clearance
			// clear of it. Measured off the POINTER rather than off the spot the walk
			// happened to cross, because those differ by the surface's thickness and
			// two captures on different bearings cross it at different points — which
			// is the marker jumping sideways for no reason you can see.
			// How far the pointer sits on the walker's side of the barrier, and how far
			// the DESTINATION does. Same measure, same plane, directly comparable.
			const q = _walkNrm.dot((blocked as Intersection).point);
			const destDepth = _walkNrm.dot(from) - q;
			if (destDepth < m.wpClearance) {
				// THE CLEARANCE IS A PROMISE THAT CANNOT ALWAYS BE KEPT, and when it
				// cannot, the destination itself is the only honest answer.
				//
				// The marker is meant to stand a clearance clear of the barrier, on the
				// path from the destination to the pointer. If the DESTINATION is nearer
				// to that barrier than the clearance is, no point on that path qualifies
				// — walk the segment and the furthest point that clears the barrier sits
				// at a NEGATIVE distance. There is nothing to place.
				//
				// This is not a rare corner; it is what a barrier crossed at a shallow
				// angle looks like. A capture 0.74 m from the pointer, in open hallway,
				// measures only 0.20 m from a wall its path grazes at 18 degrees — and
				// 0.20 is less than the 0.48 promised. Pushing perpendicular anyway put
				// the marker 0.70 m from that capture, hugging the wall it was meant to
				// be clear of, while the click went to the capture. The perpendicular
				// push assumes there is room to retreat into; here there is none.
				//
				// So it goes to the capture: the one position known to be valid, where
				// the click lands, and what the marker claims to mark.
				_walkPt.copy(from);
			} else {
				// Room to place it. Push the pointer perpendicular onto the clearance
				// line — and only ever push, so a pointer already clear is left where it
				// is rather than dragged back toward the barrier.
				const depth = _walkNrm.dot(_walkPt) - q;
				_walkPt.addScaledVector(
					_walkNrm,
					Math.max(0, m.wpClearance - depth),
				);
				// AND THEN WALK TO IT, because the push moved the marker sideways off the
				// line the first walk proved clear, and nothing has vouched for where it
				// landed. Clearing one surface says nothing about the rest of the room: a
				// pointer on the near face of a wall, pushed half a metre off it, can come
				// to rest inside the wardrobe behind. Measured on this house, exactly that
				// — 0.49 m of clearance, and the marker buried in geometry.
				//
				// The same walk as before, at the same heights, now aimed at the marker
				// rather than the pointer. If anything stands in the way the marker comes
				// back down that line to a clearance short of it — a place the walk has
				// just shown can be reached.
				_walkAlt.set(_walkPt.x - from.x, 0, _walkPt.z - from.z);
				const span = _walkAlt.length();
				if (span > 1e-6) {
					_walkAlt.divideScalar(span);
					let stop = span;
					for (let i = 0; targets.length > 0 && i < WALK_HEIGHTS.length; i++) {
						_walkFrom.set(
							from.x,
							ground + m.eyeHeight * WALK_HEIGHTS[i],
							from.z,
						);
						this.walkRay.set(_walkFrom, _walkAlt);
						this.walkRay.near = 0;
						this.walkRay.far = span;
						const h = this.walkRay
							.intersectObjects(targets, true)
							.find((x) => this.hitIsPickable(x));
						if (h && h.distance < stop) stop = h.distance;
					}
					if (stop < span) {
						const t = Math.max(0, stop - m.wpClearance);
						_walkPt.set(
							from.x + _walkAlt.x * t,
							_walkPt.y,
							from.z + _walkAlt.z * t,
						);
					}
				}
			}
			planNormal = _walkNrm.toArray() as [number, number, number];
		}
		// THE HEIGHT IS THE DESTINATION'S, FULL STOP. Not settled onto whatever happens
		// to lie under the walk's end, which is where the last of the jerk was coming
		// from: a drop is discrete, so sliding the pointer over the lip of a step, a
		// table, or a hole in a decimated proxy snapped the marker up or down half a
		// metre with nothing to justify it. Worse, it could put the marker on a surface
		// at a different height from the one the click actually lands on — the marker
		// standing on the bed while you arrive on the floor beside it.
		//
		// A capture stands an eye height above its own floor; that is what `eye`
		// measures. So the floor it stands on is exactly this, and it is the only
		// height the marker is ever allowed to have.
		this.aimBlock = blocked
			? {
					source: "walk",
					object: describeObject(blocked.object),
					face: blocked.faceIndex ?? -1,
					dist: blocked.distance,
					point: blocked.point.toArray() as [number, number, number],
					planNormal,
				}
			: null;
		return _walkOut.set(_walkPt.x, ground, _walkPt.z);
	}

	// TEMPORARY (debug). Freeze one aim into a plain object, flat enough to read in a
	// console and complete enough to go looking for the triangle afterwards.
	private recordAim(
		cursor: Intersection,
		aim: { marker: Vector3; occluded: boolean; index: number },
	): Record<string, unknown> {
		const pano = this.panos[aim.index];
		const b = this.aimBlock;
		const cursorAt = cursor.point.toArray() as [number, number, number];
		const markerAt = aim.marker.toArray() as [number, number, number];
		// Is the marker standing INSIDE something? Tested at chest height in the column
		// under it, which is the one thing the numbers alone never reveal and the first
		// thing to know when a marker "looks like it is in front of the door".
		const chest = markerAt[1] + this.metrics.eyeHeight * 0.5;
		const buried = this.columnAt(markerAt[0], markerAt[2]).some(
			(sp) => chest > sp.bottom && chest < sp.top,
		);
		const toDest = pano
			? Math.hypot(
					markerAt[0] - pano.position[0],
					markerAt[2] - pano.position[2],
				)
			: -1;
		return {
			summary:
				`cursor ${fmt3(cursorAt)} on ${describeObject(cursor.object)}` +
				` face ${cursor.faceIndex ?? -1}\n` +
				`      marker ${fmt3(markerAt)} — ${aim.occluded ? "SHOWN (destination out of view)" : "hidden (destination in view)"}\n` +
				`      dest   #${aim.index} ${pano?.id ?? "?"} ${fmt3(pano?.position ?? [0, 0, 0])}` +
				` level ${this.panoLevel[aim.index] ?? -1}\n` +
				`      walk   ${b ? `clears ${b.object} face ${b.face} (${b.source}) at ${b.dist.toFixed(2)}m, plan normal ${b.planNormal ? fmt3(b.planNormal) : "none — level face, no push"}` : "nothing to clear"}\n` +
				`      marker ${buried ? "IS BURIED in geometry" : "stands in open air"}` +
				` · ${toDest.toFixed(2)} m from the capture in plan\n` +
				`      camera ${fmt3(this.camera.position.toArray())}`,
			cursor: cursorAt,
			cursorObject: describeObject(cursor.object),
			cursorFace: cursor.faceIndex ?? -1,
			markerBuried: buried,
			markerToDestPlan: toDest,
			marker: markerAt,
			occluded: aim.occluded,
			destIndex: aim.index,
			destId: pano?.id ?? null,
			destName: pano?.name ?? null,
			destPosition: pano?.position ?? null,
			destLevel: this.panoLevel[aim.index] ?? -1,
			currentIndex: this.currentIndex,
			currentLevel: this.panoLevel[this.currentIndex] ?? -1,
			blocker: b,
			camera: this.camera.position.toArray(),
			metrics: {
				eye: this.metrics.eyeHeight,
				wpClearance: this.metrics.wpClearance,
				wpStandoff: this.metrics.wpStandoff,
			},
		};
	}

	// TEMPORARY (debug). Print everything behind the marker currently on screen: what
	// the pointer is on, which capture the click resolves to, and — the part that
	// matters for a jerk — the exact triangle that stopped the walk.
	//
	// Resolves the aim fresh at the moment it is called rather than reading a record
	// kept per frame, so nothing is formatted or allocated until you ask. The answer is
	// identical either way: the aim point is the viewport centre (`CENTER_CURSOR`), so
	// it does not move when you reach for the button.
	//
	// Bound to L as well, because in mouse-look the pointer is captured and there is no
	// cursor to click a button with.
	logAim() {
		const at = this.aim();
		const hits = this.raycastInteriorAll(at.x, at.y);
		const aim = hits.length > 0 ? this.resolveAim(hits) : null;
		if (!aim) {
			console.log(
				`[aim] nothing resolved — ${hits.length === 0 ? "no scene geometry under the aim point" : "no capture to travel to"}`,
			);
			return;
		}
		const record = this.recordAim(hits[0], aim);
		console.log(`[aim] ${record.summary}`, record);
	}

	// Where a visitor STANDING at `waypoint` would have their eyes.
	//
	// This is the frame every capture is recorded in — the anchor planner stands its
	// cameras an eye height above a surface — so it is the frame a waypoint has to be
	// compared in. A waypoint is a place to stand, i.e. a point on a floor, and
	// matching a floor point against a set of eye-height points quietly handicaps
	// your OWN storey by that eye height while the storey below pays nothing: its
	// captures are simply below you, not offset. Near a stairwell that is enough to
	// tip the answer downstairs.
	//
	// Fills a shared scratch vector; callers must not retain it.
	private standingEye(waypoint: Vector3): Vector3 {
		return _wpEye
			.copy(waypoint)
			.setY(waypoint.y + this.metrics.eyeHeight);
	}

	// The solids stacked in one vertical column, top-down, by casting a ray straight
	// down through the whole scene.
	//
	// Faces arrive in pairs — entering a solid, then leaving it — so pairing them
	// gives spans, and the gap between one span's bottom and the next span's top is
	// clear air. `standable` is that gap measured against a person's height: a floor
	// has a room above it, the underside of a slab has 40 cm of slab.
	//
	// Deliberately uses no surface normals. The proxy is decimated and its winding is
	// unreliable — the surface cursor already flips normals toward the camera for
	// exactly that reason — so "is this face pointing up" is not a question this
	// geometry can answer, while "how much room is above it" is.
	private columnAt(
		x: number,
		z: number,
	): Array<{ top: number; bottom: number; standable: boolean }> {
		const targets = this.interiorTargets();
		if (targets.length === 0) return [];
		const from = this.sceneTopY + 1;
		this.dropRay.set(_dropFrom.set(x, from, z), _DOWN);
		this.dropRay.near = 0;
		this.dropRay.far = Math.max(1, from - this.sceneBottomY) + 1;
		const ys = this.dropRay
			.intersectObjects(targets, true)
			.map((h) => h.point.y);
		const out: Array<{ top: number; bottom: number; standable: boolean }> = [];
		for (let i = 0; i < ys.length; i += 2) {
			const top = ys[i];
			// An unpaired trailing face is an open shell (a single-sided ground
			// plane); treat it as a surface with nothing under it rather than
			// discarding it, since it is usually the floor.
			const bottom = i + 1 < ys.length ? ys[i + 1] : top;
			const air = out.length === 0 ? Infinity : out[out.length - 1].bottom - top;
			out.push({ top, bottom, standable: air >= this.metrics.standHeadroom });
		}
		return out;
	}

	// The floor point of a destination capture — where its affordance is drawn, and
	// the same placement buildNav uses for a standing marker. Free flight still uses
	// this: out there the question is "where would this put me down", and a landing
	// spot genuinely is a capture point rather than a spot on a wall.
	//
	// It used to be drawn on the cursor's own bearing instead — keeping the pointer's
	// screen column and pinning only the height to the destination's floor — so that
	// it always answered "what happens if I click HERE". But that throws away
	// everything perpendicular to the bearing, and autoHomeTarget prices a radian of
	// deviation at only 3 m, so the marker routinely stood metres from the anchor it
	// promised (and, when the anchor fell behind the cursor plane, was clamped flat
	// against the obstruction instead — maximally wrong exactly where it mattered).
	// A marker that lies about its destination is worse than one you have to trace a
	// line to.
	private destinationFloor(targetIdx: number): Vector3 {
		// Scratch: showGhost copies this straight into the marker, never retains it.
		const p = this.panos[targetIdx].position;
		return _ghostFloor.set(p[0], p[1] - this.metrics.floorDrop, p[2]);
	}

	// Click-anywhere floor routing: raycast into the scene, then travel to the
	// auto-homed node — so the floor itself is the button. A click that reaches NO
	// geometry at all isn't aiming at a destination: it has gone out past the edge
	// of the scene into the void, which reads as "back out of here" — so it pulls
	// up to the orbit rather than quietly doing nothing.
	private clickAnywhere(clientX: number, clientY: number) {
		const hits = this.raycastInteriorAll(clientX, clientY);
		if (hits.length === 0) {
			this.exit();
			return;
		}
		// The SAME function the live cursor drew with, floor rule and all, so the click
		// lands exactly where the marker said it would. Sharing it is the point: any
		// second way of picking a destination is a second answer that can disagree with
		// the one on screen.
		const aim = this.resolveAim(hits);
		if (aim) this.traverse(aim.index);
	}

	// --- view toggles (which geometry each mode shows) ------------------------

	private reskinProxy(mat: Material) {
		if (!this.proxyGroup) return;
		const matte = mat === this.polyMaterial;
		this.proxyGroup.traverse((o) => {
			const m = o as Mesh;
			if (!m.isMesh) return;
			m.material = matte
				? ((m.userData.colorMat as Material) ?? mat)
				: mat;
		});
		this.projection.setBaseMaterial(mat);
	}

	private colorProxyObjects() {
		if (!this.proxyGroup) return;
		collectObjects(this.proxyGroup).forEach((obj, i) => {
			const mat = this.polyMaterial.clone();
			mat.color.setHSL((i * 0.6180339887) % 1, 0.6, 0.55, SRGBColorSpace);
			this.proxyColorMats.push(mat);
			obj.traverse((o) => {
				if ((o as Mesh).isMesh) o.userData.colorMat = mat;
			});
		});
	}

	private proxyAsDollhouse(): boolean {
		return this.sharedOverview || (this.proxyView && !!this.proxyGroup);
	}

	// A splat is loaded AND switched on — i.e. it should stand in for the scene's
	// appearance wherever a mode chooses to show it. Not the same as being on
	// screen: the walkthrough turns it off regardless (see setInteriorView), so
	// anything asking "is it visible right now" wants `splat.isActive` instead.
	private get splatEnabled(): boolean {
		return this.splat.ready && this.splatView;
	}

	// Put the splat on or off screen. The background has to move with it: the splat
	// renders on the canvas BEHIND this one, so it is only visible while this layer
	// clears to transparent.
	private setSplatShowing(on: boolean) {
		this.splat.setActive(on);
		this.scene.background = on ? null : this.bgColor;
	}

	private setOverviewView() {
		// The splat IS the dollhouse when the cell has one. The lite mesh stays
		// loaded underneath it — it is the addressable geometry and the fallback —
		// but nothing renders it while the splat is doing that job.
		const useSplat = this.splatEnabled;
		const proxyDoll = !useSplat && this.proxyAsDollhouse();
		if (this.liteRoot) this.liteRoot.visible = !useSplat && !proxyDoll;
		if (this.proxyGroup) {
			if (proxyDoll) {
				this.reskinProxy(this.polyMaterial);
				this.proxyGroup.visible = true;
			} else {
				this.proxyGroup.visible = false;
			}
		}
		this.setSplatShowing(useSplat);
		this.sphereA.visible = false;
		this.sphereB.visible = false;
		this.markers.navGroup.visible = false;
		this.markers.arrowGroup.visible = false;
		this.markers.hideSonar();
		this.markers.you.group.visible = false;
		this.projection.syncBase(!!this.proxyGroup?.visible);
	}

	// Free flight: the splat carries the whole picture, and the proxy is hidden but
	// NOT removed — it is the only geometry that knows where the surfaces are, so
	// the cursor keeps raycasting it (see raycastInteriorAll) and a click still
	// resolves to a real place. With the splat switched off it renders as bare
	// polygons instead, which is also how you check the two are in register.
	private setFreeflyView() {
		const useSplat = this.splatEnabled;
		if (this.liteRoot) this.liteRoot.visible = false;
		if (this.proxyGroup) {
			if (!useSplat) this.reskinProxy(this.polyMaterial);
			this.proxyGroup.visible = !useSplat;
		}
		this.setSplatShowing(useSplat);
		this.sphereA.visible = false;
		this.markers.navGroup.visible = false;
		this.markers.arrowGroup.visible = false;
		this.markers.hideSonar();
		this.markers.you.group.visible = false;
		this.projection.syncBase(false);
	}

	private setInteriorProxyView() {
		if (!this.proxyGroup || !this.projectionMode) return;
		this.reskinProxy(
			this.proxyView ? this.polyMaterial : this.projection.material,
		);
		this.sphereA.visible = !this.proxyView;
		if (!this.proxyView) this.updateProjection();
	}

	private setInteriorView() {
		// Inside the walkthrough the panoramas ARE the picture — they are a
		// higher-fidelity rendering of the same room than the splat is, and showing
		// both would just double-expose it. The splat context idles until free
		// flight or the overview asks for it again.
		this.setSplatShowing(false);
		if (this.liteRoot) this.liteRoot.visible = false;
		if (this.proxyGroup) this.proxyGroup.visible = this.projectionMode;
		if (this.projectionMode) {
			this.setInteriorProxyView();
		} else {
			this.sphereA.visible = true;
			// The backdrop rides the camera, so seed it here: the first interior frame
			// is drawn before the interior branch of the tick ever runs.
			this.sphereA.position.copy(this.camera.position);
		}
		this.markers.navGroup.visible = true;
		this.markers.arrowGroup.visible = true;
		this.markers.you.group.visible = false;
		this.projection.syncBase(!!this.proxyGroup?.visible);
	}

	private setPeekView() {
		// Hold-to-locate slices the roof off with a renderer clipping plane, which
		// belongs to three.js and means nothing to the splat's own context — a splat
		// here would keep its ceiling and bury the "you are here" pin under it. So
		// locating is always done on the mesh.
		this.setSplatShowing(false);
		const proxyDoll = this.proxyAsDollhouse();
		if (this.liteRoot) this.liteRoot.visible = !proxyDoll;
		if (this.proxyGroup) {
			if (proxyDoll) {
				this.reskinProxy(this.polyMaterial);
				this.proxyGroup.visible = true;
			} else {
				this.proxyGroup.visible = false;
			}
		}
		this.sphereA.visible = false;
		this.sphereB.visible = false;
		this.markers.navGroup.visible = false;
		this.markers.arrowGroup.visible = false;
		this.markers.hideSonar();
		this.markers.you.group.visible = true;
		this.projection.syncBase(!!this.proxyGroup?.visible);
	}

	private canToggleProxyView(): boolean {
		// While the splat is ON SCREEN the mesh views are not, so the proxy/lite swap
		// has nothing to swap — turn the splat off first. Deliberately not keyed on
		// the splat merely being loaded: inside the walkthrough the splat is off
		// screen and the interior's own proxy/projection toggle must keep working.
		if (this.splat.isActive) return false;
		if (this.mode === "overview")
			return !!this.liteRoot && !!this.proxyGroup;
		if (this.mode === "interior") return this.projectionMode;
		return false;
	}

	// Only where the switch actually changes what is on screen. It deliberately
	// EXCLUDES the walkthrough: the panoramas are the picture in there and the splat
	// is off by design, so offering the control would put a live-looking button in
	// front of you that does nothing when pressed — which reads as the splat being
	// broken rather than as the control being inapplicable.
	toggleProxyView() {
		if (!this.canToggleProxyView()) return;
		this.proxyView = !this.proxyView;
		this.addressing.setHover(null);
		this.addressing.closeMenu();
		this.canvas.style.cursor = "";
		if (this.mode === "overview") this.setOverviewView();
		else if (this.mode === "interior") this.setInteriorProxyView();
		this.emit();
	}

	// --- per-object addressing ------------------------------------------------

	private activeObjectRoot(): Object3D | null {
		if (this.mode === "overview" || this.mode === "peek") {
			if (this.proxyView && this.proxyGroup) return this.proxyGroup;
			return this.liteRoot ?? this.proxyGroup;
		}
		if (this.mode === "interior") return this.proxyGroup;
		return null;
	}

	closeMenu() {
		if (!this.addressing.hasMenu) return;
		this.addressing.closeMenu();
		this.emit();
	}
	toggleMenuTargetHidden() {
		this.addressing.toggleMenuTargetHidden();
		this.emit();
	}
	toggleMenuTargetOutline() {
		this.addressing.toggleMenuTargetOutline();
		this.emit();
	}
	showAllHidden() {
		this.addressing.showAllHidden();
		this.emit();
	}
	clearOutlines() {
		this.addressing.clearOutlines();
		this.emit();
	}

	// --- projection -----------------------------------------------------------

	private updateProjection() {
		// The proxy has depth, so a projected capture parallaxes and two captures
		// cross-dissolve anchored to the same surface points — clean. The backdrop is
		// a depthless camera-centred sphere, so mid-glide the departure and
		// destination skyboxes land at slightly different angles in the void and smear
		// against each other ("old images leaking" where there's no geometry to pin
		// them). So hide it WHILE gliding: the parallax-correct proxy carries the move
		// and the void reads as clean background, then the backdrop returns — single
		// and exactly aligned — the moment we settle on the destination capture.
		this.sphereA.visible = !this.proxyView && !this.move;
		this.projection.project(
			this.panos,
			this.activeCaptures(),
			this.requestPano,
			this.sphereA,
			this.camera.position,
		);
	}

	// The capture(s) to project right now: the from/to pair while gliding a hop
	// (time-weighted so proxy + backdrop cross-dissolve together), else just the
	// capture you're standing at (an exact skybox — no offset ghosts). During the
	// overview→interior fly-in the arrival isn't `activate`d yet, so fall back to
	// the fly target. The walkthrough never free-roams, so this set is exact.
	private activeCaptures(): Array<[number, number]> {
		if (this.move) {
			const to = this.move.index;
			const from = this.currentIndex;
			if (from < 0 || from === to) return [[to, 1]];
			const t = Math.min(
				1,
				Math.max(
					0,
					(performance.now() - this.move.start) / this.move.dur,
				),
			);
			const e = easeInOut(t);
			return [
				[from, 1 - e],
				[to, e],
			];
		}
		// The fly target OUTRANKS where we currently stand. It is only ever set while
		// a fly-in is committed to an arrival, and it is the more specific answer:
		// stepping in from the dollhouse leaves `currentIndex` at -1 so either works,
		// but landing out of free flight leaves it pointing at the anchor we
		// DEPARTED — and reading that projects the wrong capture onto the proxy for
		// the frame between setInteriorView() and activate().
		if (this.flyTarget >= 0) return [[this.flyTarget, 1]];
		if (this.currentIndex >= 0) return [[this.currentIndex, 1]];
		return [];
	}

	// --- camera flight (mode changes) -----------------------------------------

	private startFly(
		toPos: Vector3,
		lookTarget: Vector3,
		dur: number,
		cbs: {
			toFov?: number;
			crossfade?: boolean;
			dissolveInterior?: boolean;
			onMid?: () => void;
			onEnd?: () => void;
		} = {},
	) {
		this.dummyCam.up.copy(this.camera.up);
		this.dummyCam.position.copy(toPos);
		this.dummyCam.lookAt(lookTarget);
		this.dummyCam.updateMatrixWorld();
		this.transition = {
			fromPos: this.camera.position.clone(),
			toPos: toPos.clone(),
			fromQuat: this.camera.quaternion.clone(),
			toQuat: this.dummyCam.quaternion.clone(),
			fromFov: this.camera.fov,
			toFov: cbs.toFov ?? this.camera.fov,
			start: performance.now(),
			dur: this.reducedMotion ? REDUCED_DUR : dur,
			crossfade: !!cbs.crossfade,
			dissolveInterior: !!cbs.dissolveInterior,
			onMid: cbs.onMid,
			onEnd: cbs.onEnd,
			midDone: false,
		};
		// A crossfading flight never dims, so clear any dip left by an earlier one.
		if (cbs.crossfade) this.travelFade.style.opacity = "0";
		this.mode = "transition";
		this.stopLookInertia(); // the flight owns the pose from here
		this.closeInspect();
		this.arrowReach = null;
		this.markers.setArrowHover(null);
		this.controls.enabled = false;
		this.controls.autoRotate = false;
		this.hoveredNavIndex = -1;
		this.markers.setNavHover(null);
		this.markers.hideSonar();
		this.addressing.setHover(null);
		this.addressing.closeMenu();
		this.canvas.style.cursor = "";
		this.emit();
	}

	// --- typed traversal (interior) -------------------------------------------

	// Traverse to a node by its graph edge type. Chained clicks queue (input
	// buffering) instead of blocking; the back-stack is pushed unless retracing.
	private traverse(index: number, reverse = false, pass = false) {
		if (index === this.currentIndex || !this.panos[index]) return;
		if (this.interiorBusy) {
			this.pendingTravel = index; // latest click wins
			return;
		}
		const edge = this.edgeBetween(this.currentIndex, index);
		const type: EdgeType = edge?.type ?? "far";
		const dy =
			edge?.dy ??
			this.panos[index].position[1] -
				this.panos[this.currentIndex].position[1];
		if (!reverse && this.currentIndex >= 0)
			this.history.push(this.currentIndex);
		this.beginMove(index, type, dy, pass);
	}

	private beginMove(index: number, type: EdgeType, dy: number, pass = false) {
		this.interiorBusy = true;
		// Heading carries across a hop by design, but a leftover flick is not heading
		// — it would keep turning you through the traversal and land you somewhere you
		// never aimed.
		this.stopLookInertia();
		this.closeInspect();
		this.arrowReach = null;
		this.cursorReach = null;
		this.markers.setArrowHover(null);
		this.hoveredNavIndex = -1;
		this.markers.setNavHover(null);
		this.markers.navGroup.visible = false;
		this.markers.arrowGroup.visible = false;
		this.markers.hideSonar();
		// Push the cleared hover state out NOW. Nothing else emits between here and
		// the arrival at the far end, so without this the tooltip you just clicked
		// would sit there for the whole flight, previewing a place you are already on
		// your way to. The panel fades on its own once the state says it is gone.
		this.emit();
		this.requestPano(index);
		const fromPos = this.camera.position.clone();
		const toPos = v3(this.panos[index].position);
		const ctrl = this.reducedMotion
			? null
			: this.pathControl(fromPos, toPos, type);
		const dur =
			(this.reducedMotion ? REDUCED_DUR : DUR[type]) *
			(pass ? PASS_DUR_SCALE : 1);
		if (this.projectionMode) {
			this.move = {
				fromPos,
				toPos,
				ctrl,
				start: performance.now(),
				dur,
				index,
				type,
				dy,
				sphere: false,
				pass,
			};
			return;
		}
		// Sphere-only tour: wait for a texture (placeholder is enough), then
		// crossfade the backdrop while the camera drifts onto the destination.
		const token = this.loadToken;
		void this.streamer.ensure(index).then(() => {
			if (this.disposed || token !== this.loadToken) return;
			const target = this.panos[index];
			this.sphereBMat.uniforms.map.value = target.texture ?? DUMMY_TEX;
			this.sphereBMat.uniforms.opacity.value = 0;
			this.sphereB.visible = true;
			this.move = {
				fromPos,
				toPos,
				ctrl,
				start: performance.now(),
				dur,
				index,
				type,
				dy,
				sphere: true,
				pass,
			};
		});
	}

	// Bend the camera path: vertical shafts rise up-and-over; far flights pull
	// back toward a dollhouse vantage before pushing in. Walks stay straight.
	private pathControl(
		from: Vector3,
		to: Vector3,
		type: EdgeType,
	): Vector3 | null {
		if (type === "vertical") {
			const c = from.clone().add(to).multiplyScalar(0.5);
			// The floor on the arc's height keeps a short hop from reading as a
			// straight line. Scale-derived: a fixed half-metre is a shallow bump in a
			// cathedral and a vault over the ceiling in a doll's house.
			c.y =
				Math.max(from.y, to.y) +
				Math.max(
					this.metrics.eyeHeight * 0.3,
					Math.abs(to.y - from.y) * 0.3,
				);
			return c;
		}
		if (type === "far") {
			const c = from.clone().add(to).multiplyScalar(0.5);
			c.y += Math.min(this.sceneMaxDim * 0.6, from.distanceTo(to) * 0.45);
			return c;
		}
		return null;
	}

	// Bring the hop in flight to its end almost immediately, keeping the eased
	// position continuous: re-base the timeline so progress resumes from exactly
	// where it is and reaches 1 in `ms`. Used when the tour is stopped mid-hop —
	// the eye can't just freeze between two capture points (the projection, the
	// affordances and the exits all assume you're standing at one), so instead of
	// gliding on for up to another 2.4s it lands right away.
	private hurryMove(ms = 240) {
		const mv = this.move;
		if (!mv) return;
		const now = performance.now();
		const t = Math.min(1, (now - mv.start) / mv.dur);
		if (t > 0.95) return; // already landing — let it
		mv.start = now - (t * ms) / (1 - t);
		mv.dur = ms / (1 - t);
	}

	private finishMove(mv: Move) {
		this.clearFx();
		if (mv.sphere) {
			const target = this.panos[mv.index];
			this.sphereAMat.uniforms.map.value = target.texture ?? DUMMY_TEX;
			this.sphereAMat.uniforms.opacity.value = 1;
			this.sphereB.visible = false;
		}
		this.interiorBusy = false;
		const p = this.panos[mv.index];
		if (!mv.pass) {
			this.arrival = {
				name: p.name ?? p.id,
				verb: edgeVerb(mv.type, mv.dy),
				ts: performance.now(),
			};
		}
		this.activate(mv.index);
		// Input buffering: run the most recent queued click as one journey.
		const next = this.pendingTravel;
		this.pendingTravel = null;
		if (next != null && next !== this.currentIndex) this.traverse(next);
	}

	// Land on a node. The look direction is deliberately left alone: heading
	// persistence across a hop is what keeps the mental map intact — you arrive
	// facing exactly where you were facing when you left, so the world appears to
	// slide past you rather than cutting to a new orientation. (The capture
	// `forward` is a fixed compass direction, not a per-edge "best view", so
	// snapping to it would just yank the camera back on every move.)
	private activate(index: number) {
		this.currentIndex = index;
		this.flyTarget = -1;
		this.visited.add(index);
		this.requestPano(index);
		if (!this.projectionMode) {
			this.sphereAMat.uniforms.map.value =
				this.panos[index].texture ?? DUMMY_TEX;
			this.sphereAMat.uniforms.opacity.value = 1;
		}
		const node = this.navNode(index);
		this.markers.buildNav(node, this.panos);
		this.markers.navGroup.visible = this.mode === "interior";
		this.refreshFloorArrows();
		if (node?.trapped) this.markers.pulseExits(performance.now(), 2200);
		this.noteInput();
		this.emit();
	}

	private onPanoReady(i: number) {
		if (!this.projectionMode && i === this.currentIndex) {
			this.sphereAMat.uniforms.map.value =
				this.panos[i].texture ?? DUMMY_TEX;
			this.sphereAMat.uniforms.opacity.value = 1;
		}
	}

	// --- mode transitions -----------------------------------------------------

	// The nearest capture on the FAR side of the plane through `at` with normal
	// `through` — the half of the world the pointer is aiming into. -1 when that side
	// holds nothing, which is the ordinary answer for the outward face of an exterior
	// wall.
	private nearestPanoBeyond(
		point: Vector3,
		at: Vector3,
		through: Vector3,
		exclude: number,
	): number {
		let best = -1;
		let bestD = Number.POSITIVE_INFINITY;
		for (let i = 0; i < this.panos.length; i++) {
			if (i === exclude) continue;
			const p = this.panos[i].position;
			if ((p[0] - at.x) * through.x + (p[2] - at.z) * through.z <= 0)
				continue;
			const d = point.distanceToSquared(v3(p));
			if (d < bestD) {
				bestD = d;
				best = i;
			}
		}
		return best;
	}

	private nearestPanoTo(
		point: Vector3,
		exclude = -1,
		onlyLevel = -1,
	): number {
		let best = -1;
		let bestD = Infinity;
		for (let i = 0; i < this.panos.length; i++) {
			if (i === exclude) continue;
			if (onlyLevel >= 0 && this.panoLevel[i] !== onlyLevel) continue;
			const d = point.distanceToSquared(v3(this.panos[i].position));
			if (d < bestD) {
				bestD = d;
				best = i;
			}
		}
		// Callers that pass no exclusion always have at least one capture to find, so
		// falling back to 0 keeps their old contract; an excluding caller genuinely
		// can come up empty (a one-capture scene) and is checked for -1.
		return best < 0 && exclude < 0 ? 0 : best;
	}

	private currentUserWorldPos(): Vector3 {
		return this.projectionMode
			? this.camera.position.clone()
			: v3(this.panos[this.currentIndex].position);
	}

	// Step inside. The flight carries the dollhouse's own framing all the way in —
	// same heading, same pitch, and a FOV that opens from the orbit framing to the
	// walkthrough's — and nothing is swapped mid-air. The scene only changes hands
	// once the camera is parked exactly on the capture point, where the dollhouse
	// render and the pano are the same view and can simply dissolve (tickCrossfade).
	enter(index: number | null = null) {
		// `this.move` is the flight already in progress: the mode stays "overview"
		// for the whole fly-in, so without this a second click part-way there would
		// start the journey again from wherever the camera had got to.
		if (this.mode !== "overview" || this.move || this.panos.length === 0) return;
		const idx = index ?? this.nearestPanoTo(this.controls.target);
		// Announced at the START of the journey, not on arrival. The camera is going
		// inside from this moment, and anything sizing itself around that has 1.1
		// seconds of flight to do it in — which is what makes the panel opening and
		// the camera diving one movement rather than two.
		this.setInside(true);
		this.history = []; // a fresh interior session
		// Requested HERE, inside the click or button press that asked to step inside,
		// because that is the gesture the browser will honour — by the time the fly-in
		// lands there is no activation left to spend.
		this.requestLock();
		this.flyIntoInterior(idx, 1100);
	}

	// Fly from wherever the camera happens to be onto a capture point, then hand
	// over to the walkthrough. Shared by stepping in from the dollhouse and by
	// landing out of free flight, because they are the same journey: you are in
	// open space, and a capture point is where the walkthrough can take over.
	//
	// The walkthrough is a yaw/pitch rig, so the live look direction is read back
	// as lon/lat and the pitch pre-clamped to what applyLook enforces. The pose the
	// flight lands on is then exactly the pose the rig holds afterwards, so the
	// handover doesn't snap the view a single degree — and because the heading
	// carries across, the room you were looking at is the room you arrive facing.
	private flyIntoInterior(
		idx: number,
		dur: number,
		{ dissolve = false }: { dissolve?: boolean } = {},
	) {
		this.requestPano(idx);
		const toPos = v3(this.panos[idx].position);
		const dir = this.camera.getWorldDirection(_flyDir);
		const look = forwardToLonLat([dir.x, dir.y, dir.z]);
		const lon = look.lon;
		const lat = MathUtils.clamp(look.lat, -MAX_PITCH, MAX_PITCH);
		this.flyTarget = idx; // project the arrival during the fly-in (pre-activate)
		this.startFly(toPos, lookTargetFrom(toPos, lon, lat), dur, {
			toFov: INTERIOR_FOV,
			// One or the other: dissolve DURING the move, or park at the end and
			// crossfade there. Never both — they are two answers to the same handover
			// and would fight over the same panorama.
			crossfade: !dissolve,
			dissolveInterior: dissolve,
			onEnd: () => {
				this.mode = "interior";
				this.lon = lon;
				this.lat = lat;
				this.arrival = null;
				this.setInteriorView();
				this.activate(idx);
			},
		});
	}

	// --- free flight ----------------------------------------------------------

	private canEnterFreefly(): boolean {
		// The proxy is the requirement, not a nicety: without it the cursor has
		// nothing to raycast, and a mode you can fly into but not click your way
		// out of is a trap.
		return this.splatEnabled && this.projectionMode && !this.interiorBusy;
	}

	// Leave the walkthrough for the splat WITHOUT moving. The camera already stands
	// exactly where the splat says the room is, so the two renderings agree at this
	// pose and the handover is a dissolve rather than a transition — the same
	// reasoning enter() uses to hand the dollhouse over to a panorama.
	//
	// Movement is live from the first frame: the ramp is cosmetic and never gates
	// input, so the keypress that asked for free flight is already moving you.
	private enterFreefly() {
		if (this.mode !== "interior" || !this.canEnterFreefly()) return;
		this.yieldTour();
		this.closeInspect();
		this.hoveredNavIndex = -1;
		this.arrowReach = null;
		this.cursorReach = null;
		this.markers.setNavHover(null);
		this.markers.setArrowHover(null);
		this.markers.hideGhost();
		this.markers.hideSonar();
		this.addressing.setHover(null);
		this.addressing.closeMenu();
		this.freeflyVel.set(0, 0, 0);
		// Free flight begins standing ON an anchor, which satisfies every dock
		// condition at once — so docking stays disarmed until the camera has actually
		// left, or you could never get out of the anchor you just left.
		this.dockTarget = -1;
		this.dockReveal = 0;
		this.dockStaged = false;
		this.dockStillSince = 0;
		this.freeflySpeed = 1; // every excursion starts at a speed you can predict
		// The anchor being left. Held off from docking only while still beside it.
		this.freeflyFrom = this.currentIndex;

		const tex =
			this.currentIndex >= 0 ? this.panos[this.currentIndex]?.texture : null;
		this.mode = "freefly";
		this.setFreeflyView();
		if (tex && !this.reducedMotion) {
			// Stage the panorama we are standing in as a camera-locked overlay at
			// full strength; the tick ramps it away to uncover the splat behind.
			this.sphereBMat.uniforms.map.value = tex;
			this.sphereBMat.uniforms.opacity.value = 1;
			this.sphereBMat.depthTest = false;
			this.sphereB.renderOrder = 20;
			this.sphereB.visible = true;
			this.sphereB.position.copy(this.camera.position);
			this.splatReveal = 0;
			this.splatRevealing = true;
			// Buy room for the lens change in proportion to its size, so an unzoomed
			// exit stays as immediate as it is today and only a zoomed one slows down.
			this.revealFovFrom = this.camera.fov;
			this.splatRevealMs = Math.min(
				SPLAT_REVEAL_MAX_MS,
				SPLAT_REVEAL_MS +
					Math.abs(this.camera.fov - FREEFLY_FOV) * REVEAL_FOV_MS_PER_DEG,
			);
		} else {
			this.clearPanoOverlay();
			this.splatReveal = 1;
			this.splatRevealing = false;
			// No dissolve to hide it in (no texture, or reduced motion): take the lens
			// straight there rather than leaving free flight on a borrowed zoom.
			this.setFov(FREEFLY_FOV);
		}
		this.noteInput();
		this.emit();
	}

	// Land out of free flight onto a capture point and give the walkthrough back.
	//
	// The interior is brought up BEFORE the flight and dissolved into during it, so
	// the move and the handover are one gesture rather than a glide followed by a
	// swap. That is only possible here: the departure image is the splat, on its own
	// canvas, so the two can be composited while both are moving. Stepping in from
	// the dollhouse cannot do this — see the note on Transition.dissolveInterior.
	//
	// It needs the destination panorama ALREADY resident, because the projection
	// shader renders black with no texture bound and dissolving into black is worse
	// than the old behaviour. The cursor pre-warms it while you aim, so this is the
	// normal case; when it isn't ready we fall back to the parked crossfade, which
	// knows how to wait.
	private returnToInterior(index: number) {
		if (this.mode !== "freefly" || !this.panos[index]) return;
		// Already settling onto exactly this anchor: let the drift finish rather than
		// abandoning it for a flight to the same place.
		if (this.dockTarget === index) return;
		this.dockTarget = -1;
		this.dockReveal = 0;
		this.dockStaged = false;
		this.freeflyKeys.clear();
		this.freeflyVel.set(0, 0, 0);
		this.splatRevealing = false;
		this.clearPanoOverlay();
		this.cursor.hide();
		this.markers.hideGhost();

		const resident = !!this.panos[index].texture;
		if (!resident || !this.projectionMode || this.reducedMotion) {
			this.flyIntoInterior(index, FREEFLY_RETURN_MS);
			return;
		}
		// Stage the interior invisible, then let the tick ramp it up over the flight.
		this.reskinProxy(this.projection.material);
		if (this.proxyGroup) this.proxyGroup.visible = true;
		this.projection.syncBase(true);
		this.canvas.style.opacity = "0";
		this.flyIntoInterior(index, FREEFLY_RETURN_MS, { dissolve: true });
	}

	private get dockRadius(): number {
		return this.metrics.dockRadius;
	}

	// The anchor the glide should settle onto, or -1 for none.
	//
	// Two exclusions matter more than proximity. An OCCLUDED anchor would pull the
	// camera through whatever is between — a wall, a slab — which is the one thing
	// free flight must never appear to do on its own. And an anchor on ANOTHER STOREY
	// would drop you through the floor you are looking at; height is tested directly
	// rather than through the floor plan so it still holds on tours whose floors carry
	// no volumes.
	//
	// The line-of-sight raycast is deliberately LAST: it is the only expensive test,
	// and by then at most a couple of anchors are still in the running.
	private dockCandidate(): number {
		const cam = this.camera.position;
		const camLevel = this.hasFloorVolumes ? this.floorAt(cam) : -1;
		// The anchor free flight started on is off-limits only while you are still
		// standing around it. Step outside its radius and it becomes an ordinary
		// candidate again, so flying away and coming back does settle.
		const from = this.freeflyFrom;
		const holdingOff =
			from >= 0 &&
			cam.distanceTo(v3(this.panos[from].position)) <= this.dockRadius;
		let best = -1;
		let bestD = this.dockRadius;
		for (let i = 0; i < this.panos.length; i++) {
			if (holdingOff && i === from) continue;
			const p = v3(this.panos[i].position);
			const dy = p.y - cam.y;
			if (Math.abs(dy) > this.metrics.dockMaxDy) continue;
			if (camLevel >= 0 && this.panoLevel[i] !== camLevel) continue;
			// Vertical offset costs more than horizontal — see DOCK_DY_WEIGHT. This is
			// the value ranked on as well as gated on, so being level with an anchor
			// beats being above a nearer one.
			const d = Math.hypot(p.x - cam.x, dy * DOCK_DY_WEIGHT, p.z - cam.z);
			if (d >= bestD) continue;
			// No texture yet means the projection would fade in black, so leave it be;
			// requesting it now makes the next pass eligible.
			if (!this.panos[i].texture) {
				this.requestPano(i);
				continue;
			}
			if (!this.isTargetClear(p)) continue;
			bestD = d;
			best = i;
		}
		return best;
	}

	// Stage or unstage the interior behind the dock, and set how much of it shows.
	//
	// The proxy re-skin only happens on the transition, not every frame — it walks the
	// whole scene graph reassigning materials, which is not a per-frame cost worth
	// paying for a value that changes nothing about the staging.
	private applyDockReveal() {
		const on = this.dockReveal > 0.001;
		if (on !== this.dockStaged) {
			this.dockStaged = on;
			if (on) {
				this.reskinProxy(this.projection.material);
				if (this.proxyGroup) this.proxyGroup.visible = true;
				this.projection.syncBase(true);
			} else {
				// Back to splat-only. setFreeflyView owns exactly this state, including
				// hiding the projection backdrop that would otherwise sit over the splat.
				this.setFreeflyView();
			}
		}
		if (on) this.updateProjection();
		this.canvas.style.opacity = on
			? easeInOut(MathUtils.clamp(this.dockReveal, 0, 1)).toFixed(3)
			: "1";
	}

	// Give the glide back to the user. Nothing is animated out: the dock only ever
	// changed the velocity TARGET, so releasing it hands the camera straight back to
	// whatever the keys are asking for, and the interior fades out on its own tau.
	private cancelDock() {
		if (this.dockTarget < 0) return;
		this.dockTarget = -1;
		this.flyTarget = -1;
	}

	// Arrived. There is no motion left to play — the drift has already put the camera
	// on the anchor and faded the interior all the way in — so this is bookkeeping.
	private commitDock(index: number) {
		this.dockTarget = -1;
		this.dockReveal = 0;
		this.freeflyKeys.clear();
		this.freeflyVel.set(0, 0, 0);
		this.splatRevealing = false;
		this.clearPanoOverlay();
		this.canvas.style.opacity = "1";
		this.cursor.hide();
		this.markers.hideGhost();
		this.mode = "interior";
		this.arrival = null;
		this.setInteriorView();
		this.activate(index);
	}

	// A click in free flight means what a click means everywhere else here: take me
	// there. It runs the walkthrough's own resolution — the surface under the
	// cursor, the floor that surface belongs to, the anchor that best answers it —
	// so both modes agree about where a click lands. The floor comes straight from
	// the geometry rather than from any look-up/look-down heuristic,
	// which are about a visitor rooted at one anchor and mean nothing in flight.
	//
	// Nothing under the cursor (aimed past the scene) falls back to the nearest
	// capture, so a click always has somewhere to put you.
	private clickFromFreefly(clientX: number, clientY: number) {
		const hit = this.raycastInterior(clientX, clientY);
		const best = hit
			? this.autoHomeTarget(hit, this.floorAt(hit.point), -1)
			: this.nearestPanoTo(this.camera.position);
		if (best >= 0) this.returnToInterior(best);
	}

	// Step back out. The capture image is dissolved away WHILE still parked at the
	// capture point — same pose, same FOV — so the dollhouse is the only thing left
	// when the fly-out begins. Flying with the pano still glued on was the
	// duplicated-room look: the equirect (and the projected proxy) rode along as
	// the camera pulled away, then the dollhouse appeared underneath it.
	exit() {
		if (this.mode !== "interior" || this.interiorBusy) return;
		// Same moment, other direction: the way out starts here, so the row starts
		// re-forming as the camera pulls back rather than once it has landed.
		this.setInside(false);
		// The overview is orbited and clicked, so it needs a cursor.
		this.releaseLock();
		this.director.abort();
		this.hoveredNavIndex = -1;
		this.markers.setNavHover(null);
		this.markers.hideSonar();
		this.addressing.setHover(null);
		this.addressing.closeMenu();
		this.canvas.style.cursor = "";
		this.clearFx();
		this.cursor.hide();
		this.markers.hideGhost();

		const tex =
			this.currentIndex >= 0
				? this.panos[this.currentIndex]?.texture
				: null;
		// Dollhouse underneath; keep the capture as a camera-locked overlay so we
		// can ramp it out without the projected proxy stretching as we leave.
		this.setOverviewView();
		this.mode = "transition";
		this.controls.enabled = false;
		this.controls.autoRotate = false;
		this.emit();

		const flyOut = () => {
			this.clearPanoOverlay();
			this.startFly(
				this.browsePos.clone(),
				this.sceneCenter.clone(),
				1000,
				{
					toFov: OVERVIEW_FOV,
					onEnd: () => {
						this.mode = "overview";
						this.controls.target.copy(this.sceneCenter);
						this.camera.position.copy(this.browsePos);
						this.controls.enabled = true;
						this.controls.update();
						this.controls.autoRotate = true;
						this.emit();
					},
				},
			);
		};

		if (!tex || this.reducedMotion) {
			flyOut();
			return;
		}
		this.sphereBMat.uniforms.map.value = tex;
		this.sphereBMat.uniforms.opacity.value = 1;
		this.sphereBMat.depthTest = false;
		this.sphereB.renderOrder = 20;
		this.sphereB.visible = true;
		this.sphereB.position.copy(this.camera.position);
		this.crossfade = {
			armed: performance.now(),
			deadline: 0,
			dur: ENTER_CROSSFADE_MS,
			direction: "out",
			onEnd: flyOut,
		};
	}

	// Travel to any node from the chrome (minimap / chapters / search): typed if
	// it's a direct edge, else a far flight.
	traverseTo(index: number) {
		if (this.mode !== "interior" || !this.panos[index]) return;
		this.yieldTour();
		this.traverse(index);
	}

	// Retrace the back-stack one hop (never blocked); empty stack → out to overview.
	goBack() {
		if (this.mode !== "interior" || this.interiorBusy) return;
		this.yieldTour();
		const prev = this.history.pop();
		if (prev == null) {
			this.exit();
			return;
		}
		this.traverse(prev, true);
	}

	// Start / stop the zone-by-zone auto tour. Stopping leaves the camera exactly
	// where it is — the itinerary is simply dropped.
	toggleTour() {
		if (this.mode !== "interior") return;
		if (this.director.active) {
			this.yieldTour();
			return;
		}
		if (!this.navGraph || this.currentIndex < 0) return;
		this.stopLookInertia(); // the director writes these angles itself
		this.director.start(
			planZoneTour(
				this.navGraph,
				(i) => this.panos[i]?.zone ?? "",
				this.currentIndex,
			),
		);
	}

	// Any deliberate navigation hands the view back: the tour writes the look
	// angles every frame while sweeping, so it has to let go the moment the user
	// takes over rather than fight them for the camera. Stopping mid-sweep is
	// instant and leaves the camera untouched; stopping mid-hop can only let go
	// once the hop has landed, so hurry that landing along.
	private yieldTour() {
		if (!this.director.active) return;
		this.director.stop();
		if (this.director.active) this.hurryMove();
	}

	toggleSonar() {
		if (this.mode !== "interior") return;
		this.noteInput();
		if (this.markers.sonarActive) {
			this.markers.hideSonar();
		} else {
			this.markers.buildSonar(
				this.navNode(this.currentIndex),
				this.panos,
				this.currentIndex,
			);
			this.markers.startSonar(performance.now(), this.camera);
		}
		this.emit();
	}

	// Jump to a floor level (number keys / floor chips): nearest node on it.
	jumpToLevel(level: number) {
		if (this.mode !== "interior" || this.interiorBusy) return;
		if (this.panoLevel[this.currentIndex] === level) return;
		this.yieldTour();
		const cur = v3(this.panos[this.currentIndex].position);
		let best = -1;
		let bestD = Infinity;
		for (let i = 0; i < this.panos.length; i++) {
			if (this.panoLevel[i] !== level) continue;
			const d = cur.distanceToSquared(v3(this.panos[i].position));
			if (d < bestD) {
				bestD = d;
				best = i;
			}
		}
		if (best >= 0) this.traverse(best);
	}

	// Which way the you-are-here cone points ON THE MAP — an angle in the map's own
	// frame, measured from "across the page" toward "down the page".
	//
	// It used to return the yaw straight out, which is right only while the map is
	// a plan view: there, across the page IS world +X and down the page IS world
	// +Z, so the yaw already was the map angle. On an elevation the same number
	// would spin the cone about an axis the map does not have.
	getFacingDeg(): number {
		this.camera.getWorldDirection(_flyDir);
		const f = toMap(this.mapBasis, [_flyDir.x, _flyDir.y, _flyDir.z]);
		// Looking straight along the flattened axis leaves nothing to point at, so
		// hold the last readable heading rather than snapping to an arbitrary one.
		if (Math.hypot(f.u, f.v) < 1e-4) return (this.lon * 180) / Math.PI;
		return (Math.atan2(f.v, f.u) * 180) / Math.PI;
	}

	// WASD: nearest graph neighbour inside a forward cone, one floor only.
	private stepToward(dirX: number, dirZ: number) {
		if (
			this.mode !== "interior" ||
			this.interiorBusy ||
			this.currentIndex < 0
		)
			return;
		this.yieldTour();
		const cur = this.panos[this.currentIndex].position;
		let best = -1;
		let bestDist2 = Infinity;
		for (let i = 0; i < this.panos.length; i++) {
			if (i === this.currentIndex) continue;
			const p = this.panos[i].position;
			if (Math.abs(p[1] - cur[1]) > this.metrics.wasdRise) continue;
			const dx = p[0] - cur[0];
			const dz = p[2] - cur[2];
			const dist2 = dx * dx + dz * dz;
			const stride = this.metrics.wasdStep;
			if (dist2 < 1e-6 || dist2 > stride * stride) continue;
			if ((dx * dirX + dz * dirZ) / Math.sqrt(dist2) < WASD_DIR_COS)
				continue;
			if (dist2 < bestDist2) {
				bestDist2 = dist2;
				best = i;
			}
		}
		if (best >= 0) this.traverse(best);
	}

	private peekStart() {
		if (this.mode !== "interior" || this.interiorBusy) return;
		// Locating is an orbit view; hand the cursor back for it.
		this.releaseLock();
		this.savedInterior = {
			pos: this.camera.position.clone(),
			lon: this.lon,
			lat: this.lat,
			index: this.currentIndex,
			fov: this.camera.fov,
		};
		const userPos = this.currentUserWorldPos();
		this.markers.positionYouMarker(userPos);
		// Cut along the map's own axis: the roof off a building, the front off a
		// diorama. `sign` says which side the map camera stands on, and the plane
		// keeps everything on the far side of the cut from it.
		const { axis, sign } = this.mapBasis;
		this.locateClip.normal.set(0, 0, 0).setComponent(axis, -sign);
		this.locateClip.constant =
			sign * (userPos.getComponent(axis) + sign * this.metrics.sliceAboveEye);
		this.renderer.clippingPlanes = [this.locateClip];
		const flat = userPos.clone().sub(this.sceneCenter);
		flat.y = 0;
		if (flat.lengthSq() < 1e-6) flat.set(0, 0, 1);
		flat.normalize();
		const toPos = this.sceneCenter
			.clone()
			.addScaledVector(flat, this.sceneMaxDim * 1.5);
		toPos.y += this.sceneMaxDim * 1.2;
		this.startFly(toPos, this.sceneCenter.clone(), 850, {
			toFov: OVERVIEW_FOV,
			onMid: () => {
				this.setPeekView();
			},
			onEnd: () => {
				this.mode = "peek";
				this.emit();
				if (!this.peekHeld) this.peekEnd();
			},
		});
	}

	private peekEnd() {
		if (this.mode !== "peek" || !this.savedInterior) return;
		this.renderer.clippingPlanes = [];
		const s = this.savedInterior;
		this.startFly(s.pos.clone(), lookTargetFrom(s.pos, s.lon, s.lat), 800, {
			toFov: s.fov,
			onMid: () => {
				this.setInteriorView();
			},
			onEnd: () => {
				this.mode = "interior";
				this.lon = s.lon;
				this.lat = s.lat;
				this.currentIndex = s.index;
				this.activate(s.index);
			},
		});
	}

	peekDown() {
		if (this.mode !== "interior" || this.interiorBusy) return;
		this.yieldTour();
		this.peekHeld = true;
		this.onHold?.(true);
		this.peekStart();
	}
	peekUp() {
		if (!this.peekHeld) return;
		this.peekHeld = false;
		this.onHold?.(false);
		// The release is the gesture, so take the pointer back on the way in.
		this.requestLock();
		if (this.mode === "peek") this.peekEnd();
	}

	// --- cell loading ---------------------------------------------------------

	private disposeObject(obj: Object3D) {
		obj.traverse((o) => {
			const m = o as Mesh;
			if (!m.isMesh && !(o as { isLine?: boolean }).isLine) return;
			m.geometry?.dispose();
			const mats = Array.isArray(m.material) ? m.material : [m.material];
			for (const mat of mats) {
				if (
					mat &&
					mat !== this.projection.material &&
					mat !== this.polyMaterial
				)
					mat.dispose();
			}
		});
	}

	private clearScene() {
		this.loadToken++;
		this.director.abort();
		if (this.liteRoot) {
			this.scene.remove(this.liteRoot);
			this.disposeObject(this.liteRoot);
			this.liteRoot = null;
		}
		if (this.proxyGroup) {
			this.scene.remove(this.proxyGroup);
			this.disposeObject(this.proxyGroup);
			this.proxyGroup = null;
		}
		for (const m of this.proxyColorMats) m.dispose();
		this.proxyColorMats = [];
		this.projection.clearBase(this.scene);
		this.streamer.reset();
		this.connectors = [];
		this.navGraph = null;
		this.nodeDir = [];
		this.chapters = [];
		this.history = [];
		this.visited.clear();
		this.pendingTravel = null;
		this.arrival = null;
		// Back to the fallback scale until the next scene is measured. Without this a
		// load that bails out early ("nothing to show for this scene") would leave the
		// previous scene's distances in force for whatever is shown next.
		this.sceneScale = DEFAULT_SCALE;
		this.metrics = DEFAULT_METRICS;
		this.minimaps = [];
		this.mapLabels = [];
		this.mapBasis = readBasis(undefined);
		this.levelWord = "floor";
		this.panoLevel = [];
		this.minimapPrefetch = [];
		this.sphereAMat.uniforms.map.value = DUMMY_TEX;
		this.sphereBMat.uniforms.map.value = DUMMY_TEX;
		// A projection tour skins the backdrop with the VDTM material and scales it to
		// the scene — hand it back to the plain equirect material so a sphere-only tour
		// loaded next still renders its pano.
		this.sphereA.material = this.sphereAMat;
		this.sphereA.scale.setScalar(1);
		this.currentIndex = -1;
		this.flyTarget = -1;
		this.markers.clear();
		this.sphereA.visible = false;
		this.sphereB.visible = false;
		this.transition = null;
		this.move = null;
		this.crossfade = null; // a dissolve queued for a scene that's now gone
		this.clearPanoOverlay();
		this.interiorBusy = false;
		this.peekHeld = false;
		this.savedInterior = null;
		this.renderer.clippingPlanes = [];
		this.hoveredNavIndex = -1;
		this.cursorReach = null;
		this.proxyView = false;
		this.splat.clear();
		this.splatView = true; // a scene that ships a splat leads with it
		this.splatReveal = 0;
		this.splatRevealing = false;
		this.splatRevealMs = SPLAT_REVEAL_MS;
		this.revealFovFrom = FREEFLY_FOV;
		this.freeflySpeed = 1;
		this.freeflyKeys.clear();
		this.freeflyVel.set(0, 0, 0);
		this.dockTarget = -1;
		this.dockReveal = 0;
		this.dockStaged = false;
		this.freeflyFrom = -1;
		this.dockStillSince = 0;
		this.stopLookInertia(); // a new scene must not inherit the last one's spin
		this.scene.background = this.bgColor;
		// A scene swapped in mid-dissolve would otherwise inherit a part-transparent
		// canvas and render washed out for the rest of the session.
		this.canvas.style.opacity = "1";
		this.addressing.reset();
		this.releaseLock(); // a new scene starts in the overview, which needs a cursor
		this.lockClickPending = false;
		this.canvas.style.cursor = "";
		this.clearFx();
		for (const l of this.sonarLabels) l.style.display = "none";
	}

	async loadTour(source: TourSource) {
		this.mode = "loading";
		// A new scene always starts outside it, and the swap can happen from
		// anywhere — including from inside the old one, which would otherwise leave
		// the layout held open around a scene that no longer exists.
		this.setInside(false);
		this.controls.enabled = false;
		this.clearScene();
		const token = this.loadToken;
		this.showOverlay("loading scene…");

		try {
			let manifest: TourManifest | null = null;
			if (source.manifestUrl) {
				// no-store, always: tour.json is rewritten in place by a
				// re-publish or a metadata backfill, so a cached copy silently
				// serves a scene that is missing whatever was just added to it.
				const res = await fetch(source.manifestUrl, { cache: "no-store" });
				if (token !== this.loadToken || this.disposed) return;
				if (res.ok) manifest = (await res.json()) as TourManifest;
			}
			if (token !== this.loadToken || this.disposed) return;

			const mmList =
				manifest && Array.isArray(manifest.minimaps)
					? manifest.minimaps
					: [];
			this.minimaps = mmList.map((m) => ({
				...m,
				url: source.resolveMinimap(m.file),
			}));
			this.minimapPrefetch = this.minimaps.map((m) => {
				const img = new Image();
				img.src = m.url;
				return img;
			});

			const list =
				manifest && Array.isArray(manifest.panos) ? manifest.panos : [];
			const entries: PanoEntry[] = list.map((p) => {
				const { url, placeholderUrl } = source.resolvePano(p.file);
				return {
					id: p.id,
					name: p.name,
					zone: p.zone,
					level: p.level,
					position: p.position,
					forward: p.forward,
					url,
					placeholderUrl,
					texture: null,
					placeholderTexture: null,
					hasFull: false,
					requested: false,
				};
			});

			const connectors =
				manifest && Array.isArray(manifest.connectors)
					? manifest.connectors
					: [];
			const objectIds =
				manifest && Array.isArray(manifest.objects) ? manifest.objects : [];
			this.mapLabels =
				manifest && Array.isArray(manifest.map_labels)
					? manifest.map_labels
					: [];
			// The map's frame comes from the slices themselves (the capture stamps
			// each with the basis it drew that image in), so the viewer places
			// captures with exactly the mapping that produced the picture.
			this.mapBasis = readBasis(this.minimaps[0]?.basis);
			const word = manifest?.profile?.level_word;
			this.levelWord = typeof word === "string" && word ? word : "floor";

			let proxyRoot: Group | null = null;
			if (manifest?.proxy) {
				try {
					proxyRoot = await loadGLB(
						source.resolveProxy(manifest.proxy),
					);
				} catch {
					proxyRoot = null;
				}
			}
			let lite: Group | null = null;
			if (source.dollhouseUrl) {
				try {
					lite = await loadGLB(source.dollhouseUrl);
				} catch {
					lite = null;
				}
			}
			// Loaded BEFORE the first frame rather than popped in afterwards: the
			// splat IS the scene's appearance when it has one, and showing the
			// dollhouse only to swap it out a second later reads as a glitch.
			// Failure is non-fatal — the dollhouse and the walkthrough are each
			// complete without it, so a broken splat costs the feature, not the scene.
			// Drawn exactly where the file says it is — see the note by
			// IDENTITY_TRANSFORM for why no correction is applied here.
			if (source.splatUrl) await this.splat.load(source.splatUrl);
			if (token !== this.loadToken || this.disposed) return;
			this.applyScene(entries, proxyRoot, lite, connectors, objectIds);
		} catch (e) {
			if (token !== this.loadToken || this.disposed) return;
			this.mode = "empty";
			this.showOverlay(
				`failed to load scene: ${e instanceof Error ? e.message : String(e)}`,
				{ spinner: false, err: true },
			);
		}
	}

	private applyScene(
		entries: PanoEntry[],
		proxyRoot: Group | null,
		lite: Group | null,
		connectors: Connector[],
		objectIds: string[] = [],
	) {
		this.connectors = connectors;
		this.inspectable = new Set(objectIds);
		this.streamer.reset(entries);
		// Which storey each capture stands on. Taken from the manifest when it says
		// — the floor planner decided the split and assigned every capture to one,
		// so re-deriving it here could only disagree, and a capture near a boundary
		// is exactly where it would. Matching the nearest slice by height is the
		// fallback for tours captured before the split was planned.
		this.panoLevel = entries.map((p) =>
			typeof p.level === "number" &&
			p.level >= 0 &&
			p.level < this.minimaps.length
				? p.level
				: levelForPosition(this.minimaps, p.position),
		);
		this.projectionMode = !!proxyRoot;
		this.sharedOverview = !lite && !!proxyRoot;

		if (!lite && !proxyRoot) {
			this.mode = "empty";
			this.showOverlay("nothing to show for this scene", {
				spinner: false,
				err: true,
			});
			return;
		}

		// Make both roots shadeable BEFORE anything re-skins them: generate the
		// missing normals (without which the standard material shades to black),
		// force the matte splat-tier look, and put falsely-BLEND opaque geometry
		// back in the opaque queue so depth — not centroid sorting — decides what
		// occludes what. See prepare.ts.
		if (lite) {
			prepareLitScene(lite);
			this.liteRoot = lite;
			this.scene.add(lite);
		}
		if (proxyRoot) {
			prepareLitScene(proxyRoot);
			this.projection.setup(proxyRoot, this.sphereA);
			this.proxyGroup = proxyRoot;
			this.scene.add(proxyRoot);
		}

		// Object addressing on both roots. Connector highlights are intentionally
		// NOT pinned (hidden for now) — travel is driven entirely by the nav graph.
		if (this.liteRoot) {
			this.addressing.register(this.liteRoot);
			// The hovered object is a PROXY node, but the proxy is untextured
			// geometry; the dollhouse is the only published per-object mesh carrying
			// colour. Both name their nodes with the same pipeline id, which is what
			// lets one stand in for the other.
			for (const o of collectObjects(this.liteRoot)) {
				const label = o.userData.objLabel as string | undefined;
				if (label) this.liteByLabel.set(label, o);
			}
		}
		if (this.proxyGroup) {
			this.addressing.register(this.proxyGroup);
			this.colorProxyObjects();
			this.projection.buildBase(this.proxyGroup, this.scene);
		}

		const framed = lite ?? proxyRoot!;
		const box = new Box3().setFromObject(framed);
		const size = box.getSize(new Vector3());
		box.getCenter(this.sceneCenter);
		this.sceneMaxDim = Math.max(size.x, size.y, size.z) || 1;
		this.sceneTopY = box.max.y;
		this.sceneBottomY = box.min.y;
		this.rig.fit(box); // spend the shadow frustum's precision on this scene

		// MEASURE THE SCENE BEFORE ANYTHING READS A DISTANCE FROM IT. Every metre
		// value below — the clip planes, the nav graph's reach, the marker sizes, the
		// line-of-sight trims — is derived from these three numbers, so this has to
		// land first. See scale.ts for what is measured and why.
		//
		// The proxy is passed because the eye height is found by dropping a ray from
		// each capture onto it; a tour without one falls back, and both are handled
		// inside measureSceneScale.
		this.sceneScale = measureSceneScale(
			this.sceneMaxDim,
			entries.map((p) => p.position),
			this.proxyGroup,
		);
		this.metrics = navMetrics(this.sceneScale);
		console.info(`[orbit] scene scale — ${describeScale(this.sceneScale)}`);

		this.camera.near = this.metrics.cameraNear;
		this.camera.far = this.metrics.cameraFar;

		// Build the typed navigation graph now that geometry + panos are placed.
		// `segmentBlocked` reads `metrics.losTrim`, so the measurement above is a
		// hard prerequisite rather than a nicety.
		this.navGraph = buildNavGraph(
			entries.map((p) => ({ position: p.position, zone: p.zone })),
			this.panoLevel,
			(a, b) => this.segmentBlocked(a, b),
			this.metrics,
		);
		this.buildSceneDirectory(entries);

		this.markers.build(this.sceneMaxDim, this.metrics);

		const dist = this.sceneMaxDim * 1.6;
		this.browsePos
			.copy(this.sceneCenter)
			.add(new Vector3(dist * 0.7, dist * 0.5, dist * 0.9));
		this.camera.position.copy(this.browsePos);
		this.camera.fov = OVERVIEW_FOV;
		this.camera.lookAt(this.sceneCenter);
		this.camera.updateProjectionMatrix();
		this.controls.target.copy(this.sceneCenter);
		this.controls.enabled = true;
		this.controls.update();
		this.controls.autoRotate = true;

		this.setOverviewView();
		this.mode = "overview";
		this.hideOverlay();
	}

	// Line-of-sight test for the nav graph: is the straight segment between two
	// capture points blocked by the proxy? (No proxy → nothing occludes, so every
	// same-level pair reads as a clear walk.) Trimmed at both ends so hugging a
	// wall doesn't read as a block.
	private segmentBlocked(
		a: [number, number, number],
		b: [number, number, number],
	): boolean {
		if (!this.proxyGroup) return false;
		const from = v3(a);
		const d = v3(b).sub(from);
		const dist = d.length();
		if (dist < 1e-3) return false;
		d.divideScalar(dist);
		const trim = this.metrics.losTrim;
		this.occluder.set(from, d);
		this.occluder.near = trim;
		this.occluder.far = dist - trim;
		if (this.occluder.far <= this.occluder.near) return false;
		return this.occluder.intersectObject(this.proxyGroup, true).length > 0;
	}

	// Live LOS from the camera to a target pano position. Shoots one direct ray
	// plus four slightly-offset origin rays (a small cross in the XZ plane) so a
	// ray that barely clips a wall corner doesn't mis-report the whole path as
	// blocked. Returns true when ANY ray reaches the target unobstructed.
	private isTargetClear(target: Vector3): boolean {
		if (!this.proxyGroup) return true;
		const cx = this.camera.position.x;
		const cy = this.camera.position.y;
		const cz = this.camera.position.z;
		const spread = this.metrics.aimSpread;
		const trim = this.metrics.aimTrim;
		for (const [ox, oz] of [
			[0, 0],
			[spread, 0],
			[-spread, 0],
			[0, spread],
			[0, -spread],
		]) {
			_losFrom.set(cx + ox, cy, cz + oz);
			_losDir.copy(target).sub(_losFrom);
			const dist = _losDir.length();
			if (dist < this.metrics.aimMinDist) return true;
			_losDir.divideScalar(dist);
			this.occluder.set(_losFrom, _losDir);
			this.occluder.near = trim;
			this.occluder.far = dist - trim;
			if (this.occluder.far <= this.occluder.near) continue;
			if (
				this.occluder.intersectObject(this.proxyGroup, true).length ===
				0
			)
				return true;
		}
		return false;
	}

	// Stable per-scene directory + zone chapters + undirected map edges (for the
	// minimap overlay, chapters drawer, and "take me to" search).
	private buildSceneDirectory(entries: PanoEntry[]) {
		this.nodeDir = entries.map((p, i) => ({
			index: i,
			name: p.name ?? null,
			zone: p.zone ?? null,
			level: this.panoLevel[i],
		}));
		const chapters: Chapter[] = [];
		for (let i = 0; i < entries.length; i++) {
			const zone = entries[i].zone ?? "";
			const found = chapters.find((c) => c.zone === zone);
			if (found) found.count++;
			else chapters.push({ zone, count: 1, firstIndex: i });
		}
		this.chapters = chapters;
	}

	// --- dwell inspection ------------------------------------------------------

	// Watch how long the cursor has rested on one object. Tracked here rather than
	// in the pointer handler because dwelling is precisely the absence of pointer
	// events — the hover is already resolved, what we are timing is the stillness.
	private tickInspect(now: number) {
		const obj = this.addressing.hoveredObject;
		const label = obj ? ((obj.userData.objLabel as string) ?? null) : null;
		if (label !== this.hoverLabel) {
			this.hoverLabel = label;
			this.hoverSince = now;
			if (this.inspect) this.closeInspect();
		}
		if (
			!label ||
			this.inspect ||
			!this.inspectable.has(label) ||
			!this.liteByLabel.has(label) ||
			now - this.hoverSince < INSPECT_DWELL_MS
		)
			return;
		this.openInspect(label);
	}

	// Build the inset: a clone of the dollhouse copy of this object, recentred on a
	// pivot so it turns about itself, framed by its own bounding sphere, lit by a
	// small rig of its own. The clone shares geometry and materials with the scene
	// copy — only the transform is ours — so opening one costs no upload.
	private openInspect(label: string) {
		const src = this.liteByLabel.get(label);
		if (!src) return;
		const scene = new Scene();
		scene.add(new HemisphereLight(0xffffff, 0x2a2f38, 1.5));
		const key = new DirectionalLight(0xffffff, 1.8);
		key.position.set(2, 3, 2.5);
		scene.add(key);
		const clone = src.clone(true);
		clone.visible = true;
		clone.traverse((o) => {
			o.visible = true;
		});
		clone.updateMatrixWorld(true);
		const box = new Box3().setFromObject(clone);
		if (box.isEmpty()) return;
		const centre = box.getCenter(new Vector3());
		const radius = Math.max(1e-3, box.getSize(new Vector3()).length() * 0.5);
		clone.position.sub(centre);
		const pivot = new Group();
		pivot.add(clone);
		scene.add(pivot);
		// Pull back far enough that the whole silhouette fits at every angle of the
		// turn — the bounding SPHERE, not the box, since the box's footprint changes
		// as it rotates and the object must never clip its own frame.
		this.inspectCam.position.set(0.62, 0.42, 1).normalize().multiplyScalar(
			(radius / Math.sin((this.inspectCam.fov * Math.PI) / 360)) * 1.12,
		);
		this.inspectCam.near = Math.max(0.01, radius * 0.05);
		this.inspectCam.far = radius * 20;
		this.inspectCam.lookAt(0, 0, 0);
		this.inspectCam.updateProjectionMatrix();
		this.inspectScene = scene;
		this.inspectPivot = pivot;
		const rect = this.canvas.getBoundingClientRect();
		const m = INSPECT_MARGIN;
		// Offset from the AIM point, so the inset appears beside whatever summoned it —
		// the pointer when pointing, the reticle when sighted.
		const at = this.aim();
		this.inspect = {
			label,
			x: Math.min(
				Math.max(at.x + INSPECT_GAP, rect.left + m),
				rect.right - INSPECT_SIZE - m,
			),
			y: Math.min(
				Math.max(at.y - INSPECT_SIZE - INSPECT_GAP, rect.top + m),
				rect.bottom - INSPECT_SIZE - m,
			),
			w: INSPECT_SIZE,
			h: INSPECT_SIZE,
		};
		this.emit();
	}

	private closeInspect() {
		if (!this.inspect) return;
		// Geometry and materials belong to the dollhouse; only the wrapper is ours.
		this.inspectScene = null;
		this.inspectPivot = null;
		this.inspect = null;
		this.emit();
	}

	// Draw the inset into its own rectangle of the main canvas, AFTER the composer
	// has presented the frame. A scissored viewport rather than a second canvas: a
	// third and fourth WebGL context (the workspace runs two engines side by side)
	// to spin one small object is not a trade worth making.
	private renderInspect(dt: number) {
		const ins = this.inspect;
		if (!ins || !this.inspectScene || !this.inspectPivot) return;
		this.inspectPivot.rotation.y += dt * INSPECT_SPIN;
		const rect = this.canvas.getBoundingClientRect();
		const x = ins.x - rect.left;
		// GL's viewport origin is bottom-left; the rect is measured from the top.
		const y = rect.height - (ins.y - rect.top) - ins.h;
		const prevAutoClear = this.renderer.autoClear;
		// The clear state has to be PUT BACK, not just overwritten. The inset needs
		// an opaque backdrop of its own, but the main pass clears to transparent so
		// the splat layer behind this canvas can show through — and leaving alpha at
		// 1 here turns the whole canvas opaque for every subsequent frame, hiding the
		// splat behind a wall of flat colour while the markers drawn on top of it
		// carry on working. That reads as "the splat stopped loading" and survives
		// until reload, which is exactly as confusing as it sounds.
		this.renderer.getClearColor(_prevClear);
		const prevClearAlpha = this.renderer.getClearAlpha();
		this.renderer.autoClear = false;
		this.renderer.setScissorTest(true);
		this.renderer.setViewport(x, y, ins.w, ins.h);
		this.renderer.setScissor(x, y, ins.w, ins.h);
		this.renderer.setClearColor(0x000000, 1);
		this.renderer.clear(true, true, false);
		this.renderer.render(this.inspectScene, this.inspectCam);
		this.renderer.setScissorTest(false);
		this.renderer.setViewport(0, 0, rect.width, rect.height);
		this.renderer.setClearColor(_prevClear, prevClearAlpha);
		this.renderer.autoClear = prevAutoClear;
	}

	// --- floor arrows ----------------------------------------------------------

	// Place the floor arrows for the node just arrived at, ON THE HEADING YOU
	// ARRIVED FACING — one ahead and above for the storey up, one ahead and below
	// for the storey down.
	//
	// Every earlier version put this marker at the destination, which is the one
	// place you are guaranteed not to be looking: it is on another floor, behind
	// the ceiling or under your feet. So the arrow had to be hunted for, which is
	// the opposite of what a way out should ask of you. Putting it on the arrival
	// heading means it is simply in front of you when you land — and since heading
	// carries across a hop, "the way you arrived facing" is exactly where your
	// attention already is.
	//
	// Placed once per arrival, in world space: turning around leaves it behind you,
	// where it belongs, rather than dragging it along like a HUD element. Clicking
	// one snaps to the nearest capture on that storey.
	private refreshFloorArrows() {
		const cur =
			this.currentIndex >= 0 ? this.panoLevel[this.currentIndex] : -1;
		if (cur < 0) {
			this.markers.clearFloorArrows();
			return;
		}
		const here = v3(this.panos[this.currentIndex].position);
		const items: Array<{ index: number; up: boolean; pos: Vector3 }> = [];
		for (const step of [1, -1]) {
			const level = cur + step;
			let index = -1;
			let best = Infinity;
			for (let i = 0; i < this.panos.length; i++) {
				if (this.panoLevel[i] !== level) continue;
				const d = here.distanceToSquared(v3(this.panos[i].position));
				if (d < best) {
					best = d;
					index = i;
				}
			}
			if (index < 0) continue;
			// DIRECTLY OVERHEAD AND UNDERFOOT, on your own spot.
			//
			// They used to be planted out on the arrival heading, near the top and
			// bottom edges of the frame, so they were in view without having to look
			// for them. The cost was that they sat somewhere you were not, and turning
			// around left them behind you.
			//
			// Straight up and straight down is where the way out of a storey actually
			// is, and it is also exactly where the old look-up / look-at-your-feet
			// gesture used to point — that gesture was invisible and had to be
			// discovered, and this is the same idea made into something you can see
			// and click. Drawn through the ceiling or the floor by the depth split, so
			// "the way up is through there" reads directly.
			items.push({
				index,
				up: step > 0,
				pos: here
					.clone()
					.setY(here.y + step * this.metrics.arrowDist),
			});
		}
		this.markers.buildFloorArrows(items);
		this.markers.arrowGroup.visible = this.mode === "interior";
	}

	// --- render loop ----------------------------------------------------------

	private tick = (time: number) => {
		const now = performance.now();
		const dt = this.lastFrame
			? Math.min(0.05, (time - this.lastFrame) / 1000)
			: 0;
		this.lastFrame = time;

		// FIRST, so both layers are drawn at the size they are about to be presented
		// at. The observer only says THAT the panel changed size; this is where it
		// gets acted on. See the ResizeObserver for what doing it there cost.
		if (this.resizePending) {
			this.resizePending = false;
			this.resize();
		}

		if (this.transition) {
			const tr = this.transition;
			const t = Math.min(1, (now - tr.start) / tr.dur);
			const e = easeInOut(t);
			this.camera.position.lerpVectors(tr.fromPos, tr.toPos, e);
			this.camera.quaternion.slerpQuaternions(tr.fromQuat, tr.toQuat, e);
			if (tr.toFov !== tr.fromFov) {
				this.camera.fov = tr.fromFov + (tr.toFov - tr.fromFov) * e;
				this.camera.updateProjectionMatrix();
			}
			this.camera.updateMatrixWorld();
			// Dissolve the interior in WHILE moving. Both layers parallax correctly —
			// the splat on its own canvas, the panorama projected onto the proxy on
			// this one — so there is nothing to smear and no reason to stop first.
			//
			// The weight ramps LATE on purpose. A capture projected from far off its
			// own vantage is badly stretched, and that error shrinks to nothing as the
			// camera converges on the anchor. So the splat carries the opening of the
			// move and the interior asserts itself exactly as it becomes correct: the
			// dissolve is scheduled by fidelity, not by the clock.
			if (tr.dissolveInterior) {
				this.updateProjection();
				const d = MathUtils.clamp(
					(t - DISSOLVE_START) / (1 - DISSOLVE_START),
					0,
					1,
				);
				this.canvas.style.opacity = easeInOut(d).toFixed(3);
			}
			// A crossfading flight stays fully visible the whole way in — there is
			// nothing to hide, because the swap happens at the far end where the two
			// renders already agree.
			// A dissolving flight must not dip either: the whole point is that the
			// picture never goes away, it only changes hands.
			if (!tr.crossfade && !tr.dissolveInterior)
				this.travelFade.style.opacity = (
					Math.sin(Math.PI * t) * 0.5
				).toFixed(3);
			if (!tr.midDone && t >= 0.5) {
				tr.midDone = true;
				tr.onMid?.();
			}
			// Never project during a flight, EXCEPT a dissolving one (handled above).
			// The enter path is still on the dollhouse and the exit path has already
			// dissolved the capture away, so projecting on either would re-glue the
			// pano to the proxy and ride it out with the camera — the duplicated-room
			// look. A dissolving flight is the one case where projecting is the point:
			// it lands on a capture point and its departure image is a different canvas.
			if (t >= 1) {
				const cb = tr.onEnd;
				const crossfade = tr.crossfade;
				this.transition = null;
				if (crossfade) {
					// Landed on the capture point. Park and dissolve rather than cut.
					this.crossfade = {
						armed: 0,
						deadline: now + HANDOVER_WAIT_MS,
						dur: this.reducedMotion
							? REDUCED_DUR
							: ENTER_CROSSFADE_MS,
						direction: "in",
						onEnd: cb,
					};
				} else {
					this.travelFade.style.opacity = "0";
					// Defensive: a dissolve leaves this mid-ramp, and a canvas stuck
					// part-transparent would quietly wash out every later frame.
					this.canvas.style.opacity = "1";
					cb?.();
				}
			}
		} else if (this.crossfade) {
			this.tickCrossfade(now);
		} else if (this.mode === "overview") {
			this.controls.update();
		} else if (this.mode === "interior") {
			if (this.move) {
				const mv = this.move;
				const t = Math.min(1, (now - mv.start) / mv.dur);
				const e = easeInOut(t);
				if (mv.ctrl) quadBezier(mv.fromPos, mv.ctrl, mv.toPos, e, _bez);
				else _bez.lerpVectors(mv.fromPos, mv.toPos, e);
				this.camera.position.copy(_bez);
				if (mv.sphere) {
					this.sphereBMat.uniforms.opacity.value = e;
					this.sphereA.position.copy(this.camera.position);
					this.sphereB.position.copy(this.camera.position);
				}
				this.setFx(mv.type, t);
				if (t >= 1) {
					this.move = null;
					this.finishMove(mv);
				}
			}
			if (this.projectionMode) {
				if (!this.proxyView) this.updateProjection();
			} else if (!this.move) {
				this.sphereA.position.copy(this.camera.position);
			}
			// The tour drives the same yaw/pitch drag-look writes, so it has to run
			// before the look is applied.
			this.director.tick(now);
			this.tickLook(dt);
			if (!this.interiorBusy) {
				// A centred reticle is aimed by TURNING, so what it is sighted on changes
				// on camera motion rather than on pointer events — which makes hover a
				// per-frame question. (In pointer mode onPointerMove still owns it, and
				// resolving it again here would be wasted work every frame.)
				if (CENTER_CURSOR) {
					const at = this.aim();
					this.updateHover(at.x, at.y);
				}
				this.markers.updateNav(
					this.camera,
					this.lon,
					now,
					this.host.clientHeight,
				);
				this.markers.updateFloorArrows(now);
				if (this.markers.sonarActive) {
					this.markers.updateSonar(
						now,
						this.camera,
						this.host.clientHeight,
					);
					this.updateSonarLabels();
					if (!this.markers.sonarActive) this.emit(); // just expired
				} else if (
					this.sonarLabels.some((l) => l.style.display !== "none")
				) {
					for (const l of this.sonarLabels) l.style.display = "none";
				}
				this.tickInspect(now);
				// Never let stillness become stuckness: pulse the exits once on dwell.
				if (!this.dwellPulsed && now - this.lastInputAt > DWELL_MS) {
					this.dwellPulsed = true;
					this.markers.pulseExits(now, 1600);
				}
			}
		} else if (this.mode === "freefly") {
			this.tickFreefly(now, dt);
		} else if (this.mode === "peek") {
			const off = this.camera.position.clone().sub(this.sceneCenter);
			const a = PEEK_ROTATE_SPEED * dt;
			const c = Math.cos(a);
			const s = Math.sin(a);
			this.camera.position.x = this.sceneCenter.x + off.x * c - off.z * s;
			this.camera.position.z = this.sceneCenter.z + off.x * s + off.z * c;
			this.camera.lookAt(this.sceneCenter);
		}

		this.updateCursorRing();
		this.addressing.updateOutlines();
		// The splat draws FIRST, from the camera this frame just settled on, so the
		// two canvases present the same pose. Anything three.js puts on top — the
		// cursor, a waypoint, the dissolving panorama — is then glued to it rather
		// than trailing it by a frame. A no-op whenever the splat is off screen.
		this.splat.render(this.camera);
		this.composer.render();
		this.renderInspect(dt);

		// Served here, at the END of the frame that drew it, because that is the
		// only moment the pixels exist to be read. Neither context is created with
		// `preserveDrawingBuffer`, so a caller reading the canvas at an arbitrary
		// time gets an undefined (usually blank) buffer — and turning that flag on
		// to make reads work anywhere costs every frame to serve the rare one.
		if (this.captureWaiting) {
			const done = this.captureWaiting;
			this.captureWaiting = null;
			done(this.composite());
		}
	};

	/**
	 * A still of the panel exactly as drawn — both layers flattened, splat under
	 * three.js — resolved on the next frame. Used by the arena's shatter, which
	 * needs real pixels to break into shards; without it the effect degrades to
	 * anonymous grey tiles.
	 *
	 * Resolves null if the engine is disposed before the frame lands, so a caller
	 * awaiting it during teardown gets an answer rather than hanging.
	 */
	capture(): Promise<HTMLCanvasElement | null> {
		if (this.disposed) return Promise.resolve(null);
		return new Promise((resolve) => {
			this.captureWaiting?.(null); // a superseded request never resolves otherwise
			this.captureWaiting = resolve;
		});
	}

	private composite(): HTMLCanvasElement | null {
		const w = this.canvas.width;
		const h = this.canvas.height;
		if (!w || !h) return null;
		const out = document.createElement("canvas");
		out.width = w;
		out.height = h;
		const ctx = out.getContext("2d");
		if (!ctx) return null;
		// The splat sits UNDER the three.js canvas on screen, so it is drawn first
		// here; three.js renders on a transparent background and lands on top. Its
		// canvas is sized independently, so both are stretched to one box rather
		// than blitted 1:1.
		const splatCanvas = this.splat.canvasEl;
		if (splatCanvas && this.splatEnabled && splatCanvas.width > 0) {
			ctx.drawImage(splatCanvas, 0, 0, w, h);
		}
		ctx.drawImage(this.canvas, 0, 0, w, h);
		return out;
	}

	// One frame of free flight. Velocity EASES toward what the held keys ask for
	// rather than snapping to it, which is most of what separates flying from
	// teleporting; the ramp is short enough to still feel deliberate.
	private tickFreefly(now: number, dt: number) {
		const cl = Math.cos(this.lat);
		const fx = cl * Math.cos(this.lon);
		const fy = Math.sin(this.lat);
		const fz = cl * Math.sin(this.lon);
		const rx = -Math.sin(this.lon);
		const rz = Math.cos(this.lon);
		const keys = this.freeflyKeys;
		_moveWish.set(0, 0, 0);
		// W/S fly along the FULL look direction, pitch included — looking up and
		// pressing forward should climb, which is the difference between flying and
		// walking. The vertical controls stay on world up, so you can rise and fall
		// without changing where you are looking: Space/Shift as the primary pair,
		// Q/E as aliases for anyone already holding the left of the keyboard.
		if (keys.has("KeyW")) _moveWish.set(fx, fy, fz);
		if (keys.has("KeyS")) _moveWish.set(-fx, -fy, -fz);
		if (keys.has("KeyD")) {
			_moveWish.x += rx;
			_moveWish.z += rz;
		}
		if (keys.has("KeyA")) {
			_moveWish.x -= rx;
			_moveWish.z -= rz;
		}
		if (keys.has("Space") || keys.has("KeyE")) _moveWish.y += 1;
		if (keys.has("Shift") || keys.has("KeyQ")) _moveWish.y -= 1;
		const asking = _moveWish.lengthSq() > 0;
		if (asking) _moveWish.normalize();
		_moveWish.multiplyScalar(
			this.sceneMaxDim * FREEFLY_SPEED_FRAC * this.freeflySpeed,
		);
		// --- docking ---------------------------------------------------------
		// No global arming gate: the only anchor that has to be held off is the one this
		// excursion started on, and dockCandidate excludes just that one, just while you
		// are still beside it.
		const topSpeed = this.sceneMaxDim * FREEFLY_SPEED_FRAC;
		// Stillness is tracked as a HELD state, not sampled at an instant: the clock
		// starts when the camera comes to rest and is thrown away the moment it moves
		// again, so a pause on the way between two places never counts as arriving.
		const still =
			!asking &&
			this.dockTarget < 0 &&
			this.freeflyVel.length() < topSpeed * DOCK_STILL_SPEED_FRAC;
		if (!still) this.dockStillSince = 0;
		else if (this.dockStillSince === 0) this.dockStillSince = now;

		if (asking) {
			// Any request to move outranks a dock in progress. This is the whole veto:
			// nothing is animating, so releasing the target hands the camera straight
			// back with its velocity intact.
			this.cancelDock();
		} else if (
			this.dockTarget < 0 &&
			this.projectionMode &&
			this.splat.isActive &&
			this.dockStillSince > 0 &&
			now - this.dockStillSince >= this.dockDelayMs
		) {
			const cand = this.dockCandidate();
			if (cand >= 0) {
				this.dockTarget = cand;
				this.flyTarget = cand; // so activeCaptures projects THIS anchor
			}
		}
		let dockDist = Infinity;
		if (this.dockTarget >= 0) {
			const to = v3(this.panos[this.dockTarget].position).sub(
				this.camera.position,
			);
			dockDist = to.length();
			// Replace only the velocity TARGET. The integrator below is unchanged, so
			// the glide curves into the anchor without its velocity ever breaking.
			_moveWish.copy(to).multiplyScalar(DOCK_SEEK_GAIN);
			if (_moveWish.length() > topSpeed) _moveWish.setLength(topSpeed);
		}

		// One short time constant for press, release and dock alike. A dock needs no
		// special easing: its target SHRINKS as the camera closes on the anchor, so the
		// approach decelerates itself. Frame-rate independent, so the feel is the same
		// at 60 and 144.
		this.freeflyVel.lerp(
			_moveWish,
			1 - Math.exp(-(dt * 1000) / FREEFLY_VEL_TAU),
		);
		this.camera.position.addScaledVector(this.freeflyVel, dt);
		this.tickLook(dt);

		// Fidelity fades in with PROXIMITY, not with a clock, and the ease runs both
		// ways so a cancelled dock fades back out instead of snapping off.
		const wanted =
			this.dockTarget >= 0
				? MathUtils.clamp(1 - dockDist / this.metrics.dockReveal, 0, 1)
				: 0;
		if (wanted > 0 || this.dockReveal > 0.001) {
			this.dockReveal +=
				(wanted - this.dockReveal) *
				(1 - Math.exp(-(dt * 1000) / DOCK_REVEAL_TAU));
			this.applyDockReveal();
		}
		if (this.dockTarget >= 0 && dockDist < this.metrics.dockArrive) {
			this.commitDock(this.dockTarget);
			return;
		}

		if (!this.splatRevealing) return;
		// The departing panorama rides the camera while it fades, so the dissolve
		// changes only opacity — a backdrop left behind would parallax against the
		// splat and read as two rooms sliding apart.
		this.splatReveal = Math.min(
			1,
			this.splatReveal + (dt * 1000) / this.splatRevealMs,
		);
		const e = easeInOut(this.splatReveal);
		this.sphereB.position.copy(this.camera.position);
		this.sphereBMat.uniforms.opacity.value = 1 - e;
		// The lens travels on the same curve as the dissolve, so any zoom applied
		// inside the walkthrough is unwound by the time free flight has the picture.
		// This is the handover that startFly cannot do for us — see REVEAL_FOV_MS_PER_DEG.
		if (this.revealFovFrom !== FREEFLY_FOV) {
			this.setFov(
				this.revealFovFrom + (FREEFLY_FOV - this.revealFovFrom) * e,
			);
		}
		if (this.splatReveal >= 1) {
			this.splatRevealing = false;
			this.clearPanoOverlay();
		}
	}

	// Parked dissolve between dollhouse and capture pano, entirely on the GPU.
	// "in" (enter): wait for the texture, then ramp the equirect over the dollhouse.
	// "out" (exit): the caller has already staged the sphere at full opacity over the
	// dollhouse; ramp it down so the capture is gone before the fly-out begins.
	private tickCrossfade(now: number) {
		const cf = this.crossfade;
		if (!cf) return;
		if (cf.armed === 0) {
			// Enter path only — exit arms itself before parking the Crossfade.
			const tex =
				this.flyTarget >= 0
					? this.panos[this.flyTarget]?.texture
					: null;
			if (!tex && now < cf.deadline) return; // parked, still streaming
			if (tex) {
				this.sphereBMat.uniforms.map.value = tex;
				this.sphereBMat.uniforms.opacity.value = 0;
				this.sphereBMat.depthTest = false; // sit OVER the dollhouse, not behind it
				this.sphereB.renderOrder = 20;
				this.sphereB.visible = true;
			}
			cf.armed = now;
		}
		const t = Math.min(1, (now - cf.armed) / cf.dur);
		const e = easeInOut(t);
		if (this.sphereB.visible) {
			this.sphereB.position.copy(this.camera.position);
			this.sphereBMat.uniforms.opacity.value =
				cf.direction === "out" ? 1 - e : e;
		}
		if (t < 1) return;
		this.crossfade = null;
		this.clearPanoOverlay();
		cf.onEnd?.();
	}

	// Tear down the dissolve sphere so a later hop crossfade finds its defaults.
	private clearPanoOverlay() {
		this.sphereB.visible = false;
		this.sphereBMat.uniforms.opacity.value = 0;
		this.sphereBMat.depthTest = true;
		this.sphereB.renderOrder = 1;
	}

	// The dollhouse's cursor: the same ring the walkthrough uses, laid on the mesh
	// under the pointer, with its arrow pointing at the capture a click would open.
	//
	// PER FRAME, not per pointer event. The overview turns on its own (autoRotate)
	// and under a drag, so the surface beneath a stationary pointer changes with no
	// input to hang the work off — a hover resolved on pointermove alone would sit
	// frozen on geometry that had rotated out from under it. The raycast is skipped
	// on any frame where neither the pointer nor the camera has actually moved,
	// which is most of them once the model settles.
	private updateOverviewCursor() {
		const aimX = this.pointerClientX;
		const aimY = this.pointerClientY;
		// A scene with no captures has nowhere to send a click, so it gets no cursor
		// that says otherwise — and neither does one already being flown into
		// (`this.move`), where the answer has been given and the ring would just ride
		// the geometry down.
		const active =
			this.pointerInside &&
			!this.pointerDown &&
			!this.move &&
			this.panos.length > 0;
		if (!active) {
			this.overviewHit = null;
			this.overviewTarget = -1;
			this.cursor.hide();
			// The held pointer keeps whatever cursor it grabbed the scene with; a
			// drag is not the moment to change what the hand looks like.
			if (!this.pointerDown) this.canvas.style.cursor = "";
			return;
		}
		const moved =
			aimX !== this.overviewAimX ||
			aimY !== this.overviewAimY ||
			!this.overviewCam.equals(this.camera.position) ||
			!this.overviewPivot.equals(this.controls.target);
		if (moved) {
			this.overviewAimX = aimX;
			this.overviewAimY = aimY;
			this.overviewCam.copy(this.camera.position);
			this.overviewPivot.copy(this.controls.target);
			this.overviewHit = this.raycastOverview(aimX, aimY);
			this.overviewTarget = this.overviewHit
				? this.nearestPanoTo(this.overviewHit.point)
				: -1;
		}
		const hit = this.overviewHit;
		// Aiming past the scene is not aiming at anything: no ring, and the plain
		// cursor back, so the emptiness around the model reads as what it is rather
		// than as somewhere that could be clicked.
		this.canvas.style.cursor = hit ? "pointer" : "";
		if (!hit) {
			this.cursor.hide();
			return;
		}
		// From the RING to the destination, not from the eye: the ring is the thing
		// the arrow is drawn on, so "that way" has to be measured from where it lies.
		// (The interior asks from the eye because there the two are a stride apart.)
		// Straight down onto the capture under your feet leaves nothing in the
		// surface to point along, and SurfaceCursor drops the arrow — correctly, that
		// click does not move you across this surface, it moves you into it.
		const travel =
			this.overviewTarget >= 0
				? _ovTravel
						.fromArray(this.panos[this.overviewTarget].position)
						.sub(hit.point)
				: null;
		this.cursor.setColor(CURSOR_CLEAR);
		this.cursor.update(hit, this.camera, this.host.clientHeight, travel);
	}

	private updateCursorRing() {
		// The overview is not a look mode and shares none of what follows — no
		// reticle, no affordances, no floor scoping — so it is answered first and in
		// full.
		if (this.mode === "overview") {
			this.updateOverviewCursor();
			this.setReticle(false, false);
			return;
		}
		const active =
			this.isLookMode &&
			!this.interiorBusy &&
			// A centred reticle belongs to the camera, not the mouse, so it must stay up
			// whether or not the pointer happens to be over the canvas — the alternative
			// is a sight that blinks out when you move your hand off the viewport.
			(CENTER_CURSOR || this.pointerInside) &&
			!this.markers.hoveredNav &&
			!this.markers.hoveredArrow;
		const at = this.aim();
		// The whole list, not just the nearest surface: the waypoint has to know where
		// the thing under the cursor ENDS, which takes the hits behind the first one.
		const hits = active ? this.raycastInteriorAll(at.x, at.y) : [];
		const hit = hits[0] ?? null;
		// One reticle, not two — but ONLY while the pointer is locked, where there is no
		// OS cursor to conflict with anyway. Released, the cursor must come back: Esc's
		// whole job is handing it over so the chrome can be reached, and hiding it here
		// would take back the thing that keypress just gave. (The pointer-relative
		// cursor hints updateHover sets are meaningless with a centred aim point, so
		// clearing them is the honest default.)
		if (CENTER_CURSOR && this.isLookMode) {
			this.canvas.style.cursor = this.locked ? "none" : "";
		}
		let ghosted = false;
		let reach: ReachTarget | null = null;
		// Where a click from here would carry the eye. The cursor turns this into its
		// direction arrow — but only where the surface it is lying on can express it
		// (see SurfaceCursor.aimArrow).
		let travel: Vector3 | null = null;
		if (this.mode === "freefly" && this.dockTarget >= 0) {
			// A dock in progress OWNS the waypoint. Showing where the glide is settling
			// is what makes it read as a settle rather than a snatch — the destination
			// is on screen, growing, before the camera ever gets there.
			this.markers.showGhost(
				this.destinationFloor(this.dockTarget),
				{ to: this.dockTarget, type: "walk", dy: 0 },
				this.camera,
				this.host.clientHeight,
			);
			ghosted = true;
		} else if (hit && this.mode === "freefly") {
			// In flight the only question a cursor can answer is "where would this
			// put me down", so it answers exactly that: a waypoint standing on the
			// capture a click would land on. None of the interior's floor scoping or
			// occlusion tinting applies — those describe a visitor rooted at one
			// anchor, and you are not rooted at one.
			const targetIdx = this.autoHomeTarget(hit, this.floorAt(hit.point), -1);
			if (targetIdx >= 0) {
				this.cursor.setColor(CURSOR_CLEAR);
				this.markers.showGhost(
					this.destinationFloor(targetIdx),
					{ to: targetIdx, type: "walk", dy: 0 },
					this.camera,
					this.host.clientHeight,
				);
				ghosted = true;
				// Recorded so the shared change-detection below STREAMS this pano
				// while you are still deciding. Without it the first request happens
				// at click time, and the arrival then parks at the anchor waiting up
				// to a second for a 4k equirect to decode before the dissolve can
				// even begin — which is the flight landing and then visibly
				// re-settling. The walkthrough pre-warms the same way on hover; free
				// flight has to as well, and for the same reason.
				//
				// It drives nothing else here: the 360 preview panel this feeds in
				// the walkthrough is gated to interior mode (see emit).
				reach = {
					index: targetIdx,
					level: this.panoLevel[targetIdx] ?? -1,
					levelDelta: 0,
				};
			}
		} else if (hit && this.currentIndex >= 0) {
			const curLevel = this.panoLevel[this.currentIndex] ?? -1;
			// `resolveAim` picks the destination, places the marker ON it, and decides
			// whether to draw it at all. The click path calls the very same function, so
			// what you are shown and where you go cannot drift apart.
			const aim = this.resolveAim(hits);
			if (aim) {
				const { marker, occluded, index: targetIdx } = aim;
				const destLevel = this.panoLevel[targetIdx] ?? -1;
				if (occluded) {
					// The click carries you somewhere you cannot see, so draw the marker —
					// through whatever is hiding it, which is what the markers' two-pass
					// depth split is for. It is on the floor you are already on, because
					// an unseen storey change is re-picked back onto it in `resolveAim`,
					// so there is nothing to announce and the marker is the amber portal
					// rather than the green chevron.
					this.cursor.setColor(NAV_COLORS.portal);
					reach = { index: targetIdx, level: destLevel, levelDelta: 0 };
					this.markers.showGhost(
						marker,
						{ to: targetIdx, type: "portal", dy: 0 },
						this.camera,
						this.host.clientHeight,
					);
					ghosted = true;
				} else {
					// You can already see where this lands, so there is nothing a marker
					// would add — you are looking straight at the spot. Pointing at the
					// near wall of a small room and being sent to the capture beside you
					// arrives here too, and honestly: no passage, so no promise of one.
					//
					// This is also the ONE path that may change storey, and the reason
					// visibility is what licenses it — an open mezzanine you can see down
					// onto. The cursor wears the green of the floor arrows to say so.
					const crossesLevel =
						curLevel >= 0 && destLevel >= 0 && destLevel !== curLevel;
					this.cursor.setColor(
						crossesLevel ? NAV_COLORS.vertical : CURSOR_CLEAR,
					);
					travel = v3(this.panos[targetIdx].position).sub(
						this.camera.position,
					);
				}
			} else {
				// A tour of one capture, with nowhere else to go. Every other aim at real
				// geometry resolves; `resolveAim` has no refusal in it.
				this.cursor.setColor(CURSOR_CLEAR);
			}
		}
		if (!ghosted) this.markers.hideGhost();
		// Emit only when the DESTINATION changes, never per pixel of pointer travel —
		// the panel tracks the cursor itself (see OrbitViewer's ReachPreviewPanel), so
		// moving within one destination's catchment costs nothing. Re-targeting is
		// frequent by design now, but the panel cross-dissolves rather than remounting,
		// so a re-render here is a change of contents, not a rebuild of the window.
		const changed = (this.cursorReach?.index ?? -1) !== (reach?.index ?? -1);
		this.cursorReach = reach;
		if (changed) {
			if (reach) this.requestPano(reach.index); // warm the pano it pans through
			this.emit();
		}
		this.cursor.update(hit, this.camera, this.host.clientHeight, travel);
		// Deliberately NOT gated on `active`: that also goes false when an affordance is
		// hovered, and since affordances are picked at the aim point too, the reticle
		// would blink out exactly when it is sighted on something worth clicking.
		this.setReticle(this.isLookMode && !this.interiorBusy, !!hit);
	}

	// Show the crosshair, and say whether it found anything. Writes only on change, so
	// a per-frame call costs nothing and can never thrash layout against the
	// getBoundingClientRect reads elsewhere in the frame.
	private setReticle(show: boolean, onSurface: boolean) {
		const want =
			!CENTER_CURSOR || !show
				? "0"
				: onSurface
					? String(RETICLE_ON_SURFACE)
					: String(RETICLE_IN_VOID);
		if (this.reticle.style.opacity !== want) this.reticle.style.opacity = want;
	}

	// X-ray name tags for the nearest sonar nodes (engine-owned DOM pool, so the
	// reveal labels track look without churning React).
	private updateSonarLabels() {
		const targets = this.markers.sonarLabelTargets(
			this.camera,
			this.canvas,
			8,
		);
		const rect = this.canvas.getBoundingClientRect();
		for (let i = 0; i < this.sonarLabels.length; i++) {
			const el = this.sonarLabels[i];
			const t = targets[i];
			if (!t) {
				el.style.display = "none";
				continue;
			}
			el.textContent = t.name;
			el.style.left = `${t.x - rect.left}px`;
			el.style.top = `${t.y - rect.top}px`;
			el.style.display = "block";
		}
		// Grow the pool on demand.
		while (this.sonarLabels.length < targets.length) {
			const el = document.createElement("div");
			Object.assign(el.style, {
				position: "absolute",
				transform: "translate(-50%, -140%)",
				padding: "1px 6px",
				borderRadius: "5px",
				background: "rgba(10,12,20,0.82)",
				color: "#cfe6ff",
				font: "600 10px ui-sans-serif, system-ui, sans-serif",
				whiteSpace: "nowrap",
				pointerEvents: "none",
				zIndex: "3",
			});
			this.host.appendChild(el);
			this.sonarLabels.push(el);
		}
	}
}
