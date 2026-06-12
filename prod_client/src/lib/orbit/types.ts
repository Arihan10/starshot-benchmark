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
};

// tour.json: capture points (panos) in the shared world frame, plus an optional
// low-poly proxy GLB the panos project onto. Paths are relative to tour.json.
export type TourManifest = {
	panos?: Array<{
		id: string;
		position: [number, number, number];
		forward?: [number, number, number];
		file: string;
	}>;
	proxy?: string;
};

export type OrbitMode =
	| "empty"
	| "loading"
	| "overview"
	| "interior"
	| "peek"
	| "transition";

// What the engine surfaces is fed straight into the on-screen chrome.
export type OrbitState = {
	mode: OrbitMode;
	panoCount: number;
	currentId: string | null;
	currentIndex: number;
	hover: { id: string; occluded: boolean } | null;
	busy: boolean;
	overlay: { msg: string; spinner: boolean; err: boolean } | null;
};

export const INITIAL_ORBIT_STATE: OrbitState = {
	mode: "empty",
	panoCount: 0,
	currentId: null,
	currentIndex: -1,
	hover: null,
	busy: false,
	overlay: null,
};
