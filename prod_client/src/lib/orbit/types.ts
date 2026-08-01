// What the orbit page needs to build one scene: the dollhouse (overview) GLB,
// plus an optional capture-tour manifest. URLs are R2 assets (same-origin via
// the /r2 proxy). A null manifest means dollhouse-only — no panos to place.
// The manifest names panos/proxy by bare filename; resolvePano/resolveProxy map
// those to URLs, so the engine never needs to know the bucket's layout.
export type TourSource = {
	dollhouseUrl: string | null;
	manifestUrl: string | null;
	// The cell's SOG-encoded Gaussian splat, or null when it has none. When present
	// it REPLACES the dollhouse as the scene's appearance (overview + free flight);
	// the dollhouse stays loaded as the fallback and as the addressable geometry.
	splatUrl: string | null;
	resolvePano: (file: string) => { url: string; placeholderUrl: string };
	resolveProxy: (file: string) => string;
	resolveMinimap: (file: string) => string;
};

// The rectangle a slice image spans, in the MAP's own frame: u runs across the
// page, v runs down it. Which world axes those are is the basis's business, so
// nothing here has to care. Tours captured before the map could look any way but
// down carry `minX/maxX/minZ/maxZ` instead, which is the same rectangle for a plan
// view; `readBounds` in minimap.ts accepts either.
export type MinimapBounds = {
	minU?: number;
	maxU?: number;
	minV?: number;
	maxV?: number;
	minX?: number;
	maxX?: number;
	minZ?: number;
	maxZ?: number;
};
// Where the map camera stood and which way up the picture is. `view_from` names
// the axis it sat on looking back at the scene — so that axis is the one flattened
// away, and also the one storeys are stacked along. Absent on older tours, which
// were all plan views.
export type MinimapBasis = { view_from: string; image_down: string };
export type MinimapLevel = {
	level: number;
	// Where this storey sits along the flattened axis. `y` is the old name for the
	// same number, kept because every tour captured so far carries it.
	coord?: number;
	y: number;
	file: string;
	bounds: MinimapBounds;
	basis?: MinimapBasis;
	// The two planes this slice was cut between, along the flattened axis.
	cut?: number;
	cut_far?: number;
	// The floor describer's name for this storey (server/app/services/anchors.py).
	// Deliberately characterizes the WHOLE floor — travel to a floor auto-homes to
	// whichever anchor is nearest the cursor, so a name that promised one room
	// would be a lie. Absent on tours captured before floors were described; the
	// chrome falls back to a plain ordinal, which is honest.
	name?: string;
	// The world-space volume the storey occupies, as a minimum corner + extent.
	// This is what makes "which floor is this piece of geometry on?" answerable at
	// all: floors are otherwise a 1-D clustering of camera heights with no extent,
	// so the question could only be answered indirectly, via the nearest capture
	// point — which happily assigned a cliff face to whichever storey had an anchor
	// near it. Geometry inside NO floor volume (terrain, scenery, the void between
	// storeys) is on no floor, which is a correct answer. Absent on older tours.
	volume?: { origin: [number, number, number]; dimensions: [number, number, number] };
};

// A cross-zone connector (door / stair / ...). `id` matches a proxy object's
// name; clicking that object traverses between `starting_zone` and `target_zone`.
export type Connector = {
	id: string;
	target_zone: string;
	starting_zone: string;
};

// tour.json: capture points (panos) in the shared world frame, plus an optional
// low-poly proxy GLB the panos project onto. Paths are relative to tour.json.
export type TourManifest = {
	panos?: Array<{
		id: string;
		position: [number, number, number];
		forward?: [number, number, number];
		file: string;
		name?: string; // POI label from the anchor namer; absent for unnamed anchors
		zone?: string; // the zone this capture point belongs to (assigned server-side)
		// Which storey this capture stands on, decided once by the floor planner
		// (anchors.py) and carried through the capture. Absent on tours captured
		// before the split was planned, where the engine falls back to matching the
		// nearest slice by height.
		level?: number;
	}>;
	proxy?: string;
	minimaps?: MinimapLevel[];
	// Ids of the scene's DISCRETE objects — `node_kind: "object"` in the pipeline
	// log, as opposed to the `frame` nodes the encapsulating pass produces (zone
	// shells, ground, cliff backdrops). An allow-list: an id absent from it is
	// never offered for inspection, so an unclassifiable node fails closed.
	objects?: string[];
	// What kind of place this is (server/app/services/anchors.py `choose_profile`):
	// which way its map looks, how tall whoever moves through it is, and the word
	// its storeys go by. Absent on tours captured before it existed, where every
	// scene was assumed to be a building seen from above.
	profile?: {
		form?: string;
		view_from?: string;
		image_down?: string;
		inhabitant_height_m?: number;
		level_word?: string;
	};
	// Zone names to print on the bird's-eye map, chosen by the map labeller and
	// already pruned so no label sits inside another (see anchors.py
	// `label_map_zones`). `center` is the world centre of the zone's bbox.
	map_labels?: Array<{ id: string; label: string; center: [number, number, number] }>;
	// Cross-zone connectors from the anchor planner (doors, stairs, ...): each
	// names an object `id`, the `starting_zone` it sits in, and the `target_zone`
	// it leads into. The matching proxy object is highlighted + click-to-traverse.
	connectors?: Connector[];
};

import type { EdgeType } from "./navGraph";

export type OrbitMode =
	| "empty"
	| "loading"
	| "overview"
	| "interior"
	// Free flight through the Gaussian splat. Reached by pressing a movement key
	// inside the walkthrough, left by clicking anywhere — which lands on a capture
	// point and hands control back to the walkthrough's own traversal.
	| "freefly"
	| "peek"
	| "transition";

// One exit affordance of the node you're standing on, surfaced to the "exits"
// panel + screen reader. `bearingDeg` is the world azimuth to it; the panel
// rotates an arrow by (bearingDeg − live facing) so it always points true.
export type NavExit = {
	index: number;
	type: EdgeType;
	name: string | null;
	dist: number;
	bearingDeg: number;
};

// Everything the hover preview card needs: the destination thumbnail pre-rotated
// to its arrival heading (headingU is the equirect u fraction to center), its
// name/distance, the edge verb, and where to float (screen px).
export type HoverPreview = {
	index: number;
	id: string;
	name: string | null;
	type: EdgeType;
	dist: number;
	screenX: number;
	screenY: number;
	thumbUrl: string;
	headingU: number;
};

// The big preview for a destination you cannot see from here — either behind
// geometry on this storey, or on another one.
//
// It was briefly restricted to floor changes alone, on the argument that not being
// able to see round a wall is a small question you can answer by looking. In
// practice the amber cursor answers WHERE the click lands and this answers WHAT is
// there, and both questions are worth answering in both cases; `levelDelta` is what
// tells them apart (0 means a hop through geometry rather than a change of
// storey).
//
// It follows the nearest anchor to the cursor, so it re-targets continuously as the
// pointer moves; the panel itself never unmounts and cross-dissolves between
// destinations (see ReachPreviewPanel). The capture is panned continuously through
// its 360 rather than squashed into one frame, so the destination reads as a room
// instead of a warped strip.
export type ReachPreview = {
	index: number; // destination pano
	name: string | null; // that capture's own point-of-interest name
	url: string; // full equirect — the layer that pans
	placeholderUrl: string; // instant blurred backdrop while the full image loads
	dist: number; // metres from the eye to the destination
	level: number; // destination floor, 0-based (-1 when unknown)
	levelDelta: number; // storeys crossed, signed; 0 for a hop through geometry
	// There is deliberately no record of WHAT opened this. Everything docks in the
	// same corner, and the colour follows `levelDelta` — the kind of MOVE — rather
	// than the affordance, so a floor change is green whether you found it by
	// pointing or by hovering an arrow, and a hop through a wall is amber either
	// way. One idea, one colour.
};

// A dwell-revealed look at one object, rendered as a slowly orbiting inset so you
// can read a shape the room only ever shows you one side of. The engine draws the
// 3D into this exact rectangle of its own canvas (a scissored viewport, so no
// second WebGL context); the chrome frames it and captions it. Rect is in viewport
// CSS px.
export type ObjectInspect = {
	label: string;
	x: number;
	y: number;
	w: number;
	h: number;
};

// The node directory (stable per scene) that powers chapters + "take me to".
export type NodeDir = {
	index: number;
	name: string | null;
	zone: string | null;
	level: number;
};
export type Chapter = { zone: string; count: number; firstIndex: number };

// The right-click per-object menu, positioned in viewport coords. `label` is
// null when opened over empty space (only the recovery actions show then).
export type ObjectMenu = {
	x: number;
	y: number;
	label: string | null;
	hidden: boolean;
	outlined: boolean;
	hiddenCount: number;
	outlinedCount: number;
};

// What the engine surfaces is fed straight into the on-screen chrome.
export type OrbitState = {
	mode: OrbitMode;
	panoCount: number;
	currentId: string | null;
	currentName: string | null; // POI name of the current pano; null falls back to the id
	currentIndex: number;
	hover: { id: string; name?: string; occluded: boolean } | null;
	objectHover: string | null; // label of the object under the cursor (overview)
	proxyView: boolean; // overview shows the low-poly proxy instead of the lite dollhouse
	canProxyView: boolean; // the proxy/lite swap is available (overview + both loaded)
	// TEMPORARY. How long free flight must sit still before it settles onto a nearby
	// viewpoint, surfaced only so the value can be read while it is tuned by feel
	// ([ and ] adjust it). Delete once a number is locked into DOCK_STILL_MS.
	dockDelayMs: number;
	// Free flight's speed multiplier, driven by the wheel there (the wheel zooms in the
	// walkthrough instead, where the camera is pinned and cannot travel). Surfaced so
	// the HUD can show what the wheel just changed.
	freeflySpeed: number;
	// Whether the pointer is locked, i.e. the mouse turns the view directly. Drives the
	// hint, because Escape means something different either side of it: released, it
	// leaves the mode; held, it only hands the cursor back.
	mouseLook: boolean;
	contextMenu: ObjectMenu | null;
	busy: boolean;
	overlay: { msg: string; spinner: boolean; err: boolean } | null;
	// --- typed navigation (interior) ---
	// The current node's in-view exits (the "exits" panel + a11y list).
	exits: NavExit[];
	// Rich destination preview for the affordance the cursor is over.
	preview: HoverPreview | null;
	// The 360-panning preview of where the cursor would take you, shown whenever
	// that destination is out of sight — behind geometry or on another storey.
	reachPreview: ReachPreview | null;
	// Arrival narration ("Archive · phased through the wall"); a rising `ts` lets
	// the toast re-fire for repeat arrivals at the same node.
	arrival: { name: string; verb: string; ts: number } | null;
	sonarActive: boolean;
	// Set once the cursor has rested on a discrete object long enough to ask what
	// it is; null the moment the cursor moves off it.
	inspect: ObjectInspect | null;
	// Auto tour progress — which zone's centrepoint is being shown, out of how
	// many. Null whenever the tour isn't running.
	tour: { stop: number; stops: number; zone: string } | null;
	canGoBack: boolean;
	trapped: boolean; // the current node is a sealed room (only phase exits)
	currentZone: string | null;
	visited: number[]; // pano indices the user has stood on (minimap fill)
	// Stable-per-scene directory + groupings for search / chapters / map lines.
	nodes: NodeDir[];
	chapters: Chapter[];
	// The word this scene's storeys go by — "floor", "deck", "section" (see the
	// capture profile in anchors.py). The chrome writes "deck 2" from it, so a ship
	// stops being described as having floors.
	levelWord: string;
	// The bird's-eye minimap (interior / peek only). Every captured floor level is
	// surfaced so the chrome can page between floors without moving the camera;
	// `currentLevel` is the level the character is actually on. Per level, `points`
	// are that floor's capture anchors placed onto the slice as 0–100% offsets,
	// and `aspect` is the slice's width/height for sizing the box.
	minimap: {
		currentLevel: number;
		levels: Array<{
			level: number;
			name: string | null;
			url: string;
			// Aspect and every percentage below are relative to the CROP, not to the
			// whole slice — see `crop`.
			aspect: number;
			// The sub-rectangle of the slice image this storey actually occupies, as
			// image fractions. Slices are all rendered over the whole scene footprint,
			// so a floor that fills one wing was drawn adrift in a sea of the storeys
			// around it (the top floor here used 26% of its own map). The floor's
			// described volume gives its real extent, so the chrome shows just that.
			crop: { u0: number; v0: number; u1: number; v1: number };
			points: Array<{
				index: number;
				id: string;
				name?: string;
				leftPct: number;
				topPct: number;
				current: boolean;
			}>;
			// Zone names printed on this storey's slice, placed as 0-100% offsets.
			// `index` is the nearest capture to that zone's centre, so a label is
			// also the way to travel there.
			labels: Array<{
				id: string;
				label: string;
				leftPct: number;
				topPct: number;
				index: number;
			}>;
		}>;
	} | null;
};

export const INITIAL_ORBIT_STATE: OrbitState = {
	mode: "empty",
	panoCount: 0,
	currentId: null,
	currentName: null,
	currentIndex: -1,
	hover: null,
	objectHover: null,
	proxyView: false,
	canProxyView: false,
	dockDelayMs: 500,
	freeflySpeed: 1,
	mouseLook: false,
	contextMenu: null,
	busy: false,
	overlay: null,
	exits: [],
	preview: null,
	reachPreview: null,
	arrival: null,
	sonarActive: false,
	inspect: null,
	tour: null,
	canGoBack: false,
	trapped: false,
	currentZone: null,
	visited: [],
	nodes: [],
	chapters: [],
	levelWord: "floor",
	minimap: null,
};
