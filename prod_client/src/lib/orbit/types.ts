import type { SplatTransform } from "./splatLayer";
export type TourSource = {
	dollhouseUrl: string | null;
	manifestUrl: string | null;
	splatUrl: string | null;
	/** #TODO temporary offset for one bad SOG encode; omit (= identity) once fixed. */
	splatTransform?: SplatTransform;
	resolvePano: (file: string) => { url: string; placeholderUrl: string };
	resolveProxy: (file: string) => string;
	resolveMinimap: (file: string) => string;
};

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
export type MinimapBasis = { view_from: string; image_down: string };
export type MinimapLevel = {
	level: number;
	coord?: number;
	y: number;
	file: string;
	bounds: MinimapBounds;
	basis?: MinimapBasis;
	cut?: number;
	cut_far?: number;
	name?: string;
	volume?: { origin: [number, number, number]; dimensions: [number, number, number] };
};

export type Connector = {
	id: string;
	target_zone: string;
	starting_zone: string;
};

export type TourManifest = {
	panos?: Array<{
		id: string;
		position: [number, number, number];
		forward?: [number, number, number];
		file: string;
		name?: string;
		zone?: string;
		level?: number;
	}>;
	proxy?: string;
	minimaps?: MinimapLevel[];
	objects?: string[];
	profile?: {
		form?: string;
		view_from?: string;
		image_down?: string;
		inhabitant_height_m?: number;
		level_word?: string;
	};
	map_labels?: Array<{ id: string; label: string; center: [number, number, number] }>;
	connectors?: Connector[];
};

import type { EdgeType } from "./navGraph";

export type OrbitMode =
	| "empty"
	| "loading"
	| "overview"
	| "interior"
	| "freefly"
	| "peek"
	| "transition";

export type NavExit = {
	index: number;
	type: EdgeType;
	name: string | null;
	dist: number;
	bearingDeg: number;
};

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

export type ReachPreview = {
	index: number;
	name: string | null;
	url: string;
	placeholderUrl: string;
	dist: number;
	level: number;
	levelDelta: number;
};

export type ObjectInspect = {
	label: string;
	x: number;
	y: number;
	w: number;
	h: number;
};

export type NodeDir = {
	index: number;
	name: string | null;
	zone: string | null;
	level: number;
};
export type Chapter = { zone: string; count: number; firstIndex: number };

export type ObjectMenu = {
	x: number;
	y: number;
	label: string | null;
	hidden: boolean;
	outlined: boolean;
	hiddenCount: number;
	outlinedCount: number;
};

export type OrbitState = {
	mode: OrbitMode;
	panoCount: number;
	currentId: string | null;
	currentName: string | null;
	currentIndex: number;
	hover: { id: string; name?: string; occluded: boolean } | null;
	objectHover: string | null;
	proxyView: boolean;
	canProxyView: boolean;
	dockDelayMs: number;
	freeflySpeed: number;
	mouseLook: boolean;
	zoom: { in: boolean; out: boolean };
	contextMenu: ObjectMenu | null;
	busy: boolean;
	overlay: { msg: string; spinner: boolean; err: boolean } | null;
	exits: NavExit[];
	preview: HoverPreview | null;
	reachPreview: ReachPreview | null;
	arrival: { name: string; verb: string; ts: number } | null;
	sonarActive: boolean;
	inspect: ObjectInspect | null;
	tour: { stop: number; stops: number; zone: string } | null;
	canGoBack: boolean;
	trapped: boolean;
	currentZone: string | null;
	visited: number[];
	nodes: NodeDir[];
	chapters: Chapter[];
	levelWord: string;
	minimap: {
		currentLevel: number;
		levels: Array<{
			level: number;
			name: string | null;
			url: string;
			aspect: number;
			crop: { u0: number; v0: number; u1: number; v1: number };
			points: Array<{
				index: number;
				id: string;
				name?: string;
				leftPct: number;
				topPct: number;
				current: boolean;
			}>;
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
	zoom: { in: false, out: false },
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
