// The cells the comparison canvas puts side by side, served straight out of
// public/ rather than through the D1 catalog + R2.
//
// This is deliberately NOT the published path (lib/scenes.ts). That one resolves
// every asset from a catalog row's stored keys through the /r2 proxy, which is
// right for anything a user picks — but these two cells are a fixed experiment
// checked into the repo, so they have no catalog row to resolve from and their
// URLs are just paths under public/.
import type { TourSource } from "./orbit/types";

// A cell publishes its dollhouse and its splat; that pair is all the orbit engine
// needs to stand a scene up.
//
// The DOLLHOUSE IS NOT OPTIONAL, even though the splat is what we came to look
// at: the splat is only an appearance layer, and `applyScene` gives up with
// "nothing to show for this scene" unless some real mesh arrived to be measured,
// framed and addressed. The splat then replaces it as what you actually see
// (`loadTour` sets splatView on any scene that ships one).
function cellSource(dir: string): TourSource {
	const base = `/scenes/${dir}`;
	return {
		dollhouseUrl: `${base}/scene-lite.glb`,
		// No tour manifest, on purpose. The panos for these two cells run to 128 MB
		// together, which does not belong in the repo, and without a manifest the
		// engine loads no panos, no minimap slices and no proxy and settles in
		// orbit mode — an orbitable splat, which is the whole of what an A/B
		// comparison needs. The walkthrough is simply not published here.
		manifestUrl: null,
		splatUrl: `${base}/trained.web.sog`,
		// Unreachable while `manifestUrl` is null — every pano, slice and proxy
		// filename comes off the manifest — but the shape has to be complete, and
		// these are where those files would live if a tour were ever added.
		resolvePano: (file) => ({
			url: `${base}/tour/${file}`,
			placeholderUrl: `${base}/tour/${file}`,
		}),
		resolveProxy: (file) => `${base}/${file}`,
		resolveMinimap: (file) => `${base}/tour/${file}`,
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
		source: cellSource("modern-house-gemini-flash"),
	},
	{
		id: "platformer-level-opus-new",
		slot: "platformer-level",
		model: "Claude Opus",
		elo: 2108,
		source: cellSource("platformer-level-opus-new"),
	},
];

/**
 * What share of previous voters picked the LEFT build, revealed after you vote.
 * #TODO: hard-coded, and the same shape of lie as `elo` — it comes from the vote
 * tally for this pairing and should arrive with it.
 */
export const LEFT_VOTE_SHARE = 38;
