// The cells the comparison canvas puts side by side. A fixed pairing, so it is
// not resolved from the D1 catalog the way a scene the user PICKS is
// (lib/scenes.ts) — but it is assembled out of the same two places those assets
// actually live.
//
// SPLIT BY WEIGHT, not by kind. What the page needs before it can show anything —
// the dollhouse and the splat — is committed under public/ and served from our own
// origin, so the comparison stands up with no catalog, no bucket and no
// orchestrator running. The walkthrough's assets are the other 128 MB: 224 panos
// across the two cells, plus their proxies and floor slices. Those are pulled from
// the published bucket through the /r2 proxy, on demand, as the engine asks for
// them — a pano at a time, only once someone points at it.
import type { TourSource } from "./orbit/types";
import { assetUrl, cfImageUrl } from "./r2";

// A cell publishes its dollhouse and its splat; that pair is all the orbit engine
// needs to stand a scene up.
//
// The DOLLHOUSE IS NOT OPTIONAL, even though the splat is what we came to look
// at: the splat is only an appearance layer, and `applyScene` gives up with
// "nothing to show for this scene" unless some real mesh arrived to be measured,
// framed and addressed. The splat then replaces it as what you actually see
// (`loadTour` sets splatView on any scene that ships one).
//
// `dir` is the folder under public/; `cell` is the published cell prefix
// (run/slot/model) its tour hangs under in the bucket. The two are separate
// arguments because they are separate facts — the same build can be checked in
// under any name — and they have to be kept pointing at the SAME build by hand.
function cellSource(dir: string, cell: string): TourSource {
	const base = `/scenes/${dir}`;
	// Every walkthrough asset is named by the manifest as a bare filename relative
	// to tour.json, and publish.py puts the whole set under one prefix — so one
	// resolver serves panos, proxy and minimap slices alike.
	const tour = (file: string) => assetUrl(`${cell}/tour/${file}`);
	return {
		dollhouseUrl: `${base}/scene-lite.glb`,
		splatUrl: `${base}/trained.web.sog`,
		// WITHOUT THIS THERE IS NO WALKTHROUGH. The manifest is what carries the
		// capture points, and with none of them the scene has nowhere to put you:
		// the dollhouse is inert, clicking it does nothing, and the cursor stands
		// down rather than advertise a door that isn't there.
		manifestUrl: tour("tour.json"),
		resolvePano: (file) => ({
			url: tour(file),
			// The same blurred stand-in the catalog path uses, so a capture appears
			// instantly and sharpens rather than arriving as a held blank. Falls back
			// to the full image wherever the image-transform edge isn't there.
			placeholderUrl: cfImageUrl(`${cell}/tour/${file}`, {
				width: 640,
				quality: 50,
				blur: 16,
			}),
		}),
		resolveProxy: (file) => tour(file),
		resolveMinimap: (file) => tour(file),
	};
}

export type LocalCell = {
	id: string;
	/** The prompt slot, e.g. "modern-house". */
	slot: string;
	/** The LLM that orchestrated this build — the thing under comparison. */
	model: string;
	/**
	 * The model's rating going into this matchup, revealed only after a vote.
	 * #TODO: hard-coded alongside the cells. Ratings are a property of the
	 * leaderboard, not of a scene, and they move every time anyone votes — so this
	 * belongs to whatever serves the pairing, and the vote has to post back to it.
	 */
	elo: number;
	source: TourSource;
};

// Built ONCE at module load, never per render. OrbitViewer reloads its engine
// whenever the source's identity changes, so rebuilding these inside a component
// would restart both scenes on every render.
export const LOCAL_CELLS: readonly LocalCell[] = [
	{
		id: "modern-house-gemini-flash",
		slot: "modern-house",
		model: "Gemini Flash",
		elo: 2091,
		// #TODO: the run id is pinned here, so re-running "ahhhhhhhh" under a new
		// name silently costs both cells their walkthrough (the dollhouse and splat
		// keep working, which is what makes it silent). The pairing should arrive
		// from the server carrying its own cell keys — same #TODO as `elo`.
		source: cellSource(
			"modern-house-gemini-flash",
			"ahhhhhhhh/modern-house/gemini-flash",
		),
	},
	{
		id: "platformer-level-opus-new",
		slot: "platformer-level",
		model: "Claude Opus",
		elo: 2108,
		source: cellSource(
			"platformer-level-opus-new",
			"ahhhhhhhh/platformer-level/opus-new",
		),
	},
];

/**
 * What share of previous voters picked the LEFT build, revealed after you vote.
 * #TODO: hard-coded, and the same shape of lie as `elo` — it comes from the vote
 * tally for this pairing and should arrive with it.
 */
export const LEFT_VOTE_SHARE = 38;
