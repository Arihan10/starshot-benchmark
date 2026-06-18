// What the orbit page needs to build one scene: the dollhouse (overview) GLB,
// plus an optional capture-tour manifest. URLs are R2 assets (same-origin via
// the /r2 proxy). A null manifest means dollhouse-only — no panos to place.
// The manifest names panos/proxy by bare filename; resolvePano/resolveProxy map
// those to URLs, so the engine never needs to know the bucket's layout.
export type TourSource = {
	dollhouseUrl: string | null;
	manifestUrl: string | null;
	resolvePano: (file: string) => { url: string; placeholderUrl: string };
	resolveProxy: (file: string) => string;
	resolveMinimap: (file: string) => string;
};

// One bird's-eye slice: the scene rendered top-down, cut at a Y "level" (storey)
// the panos were captured on. `bounds` is the world-space XZ rectangle the image
// spans, so a pano at world (x,z) maps to image fractions ((x-minX)/W, (z-minZ)/D)
// — image +x is world +X, image +y (downward) is world +Z.
export type MinimapBounds = {
	minX: number;
	maxX: number;
	minZ: number;
	maxZ: number;
};
export type MinimapLevel = {
	level: number;
	y: number; // the level's representative camera height (its match key)
	file: string;
	bounds: MinimapBounds;
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
	}>;
	proxy?: string;
	minimaps?: MinimapLevel[];
};

export type OrbitMode =
	| "empty"
	| "loading"
	| "overview"
	| "interior"
	| "peek"
	| "transition";

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
	highlightEnabled: boolean; // hover-highlight the object under the cursor (toggleable)
	canHighlight: boolean; // hover-highlight applies in this mode (overview / interior w/ objects)
	contextMenu: ObjectMenu | null;
	busy: boolean;
	overlay: { msg: string; spinner: boolean; err: boolean } | null;
	// The bird's-eye minimap (interior / peek only). Every captured floor level is
	// surfaced so the chrome can page between floors without moving the camera;
	// `currentLevel` is the level the character is actually on. Per level, `points`
	// are that floor's capture anchors placed onto the slice as 0–100% offsets,
	// and `aspect` is the slice's width/height for sizing the box.
	minimap: {
		currentLevel: number;
		levels: Array<{
			level: number;
			url: string;
			aspect: number;
			points: Array<{
				index: number;
				id: string;
				name?: string;
				leftPct: number;
				topPct: number;
				current: boolean;
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
	highlightEnabled: true,
	canHighlight: false,
	contextMenu: null,
	busy: false,
	overlay: null,
	minimap: null,
};
