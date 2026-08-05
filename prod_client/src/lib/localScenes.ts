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
// `cell` is the published cell prefix (run/slot/model) everything hangs under in
// the bucket. `dir` is optional and names a folder under public/ holding a
// CHECKED-IN COPY of that same cell's dollhouse and splat — the two files the page
// cannot show anything without.
//
// Only the opening pair is worth checking in. It is what a first visit waits on,
// so it is served from our own origin and the comparison stands up with no
// catalog, no bucket and no orchestrator; every later round is a round the viewer
// is already watching something during, and can stream. Cells with no `dir`
// resolve all four asset classes from the bucket.
//
// The two are separate arguments because they are separate facts — the same build
// can be checked in under any name — and they have to be kept pointing at the SAME
// build by hand.
function cellSource(
	cell: string,
	dir?: string,
	splatTransform?: TourSource["splatTransform"],
): TourSource {
	// Every walkthrough asset is named by the manifest as a bare filename relative
	// to tour.json, and publish.py puts the whole set under one prefix — so one
	// resolver serves panos, proxy and minimap slices alike.
	const tour = (file: string) => assetUrl(`${cell}/tour/${file}`);
	return {
		splatTransform,
		dollhouseUrl: dir
			? `/scenes/${dir}/scene-lite.glb`
			: assetUrl(`${cell}/published/scene-lite.glb`),
		splatUrl: dir
			? `/scenes/${dir}/trained.web.sog`
			: assetUrl(`${cell}/splat/trained.web.sog`),
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

/** One matchup: a prompt, the two builds of it, and how the crowd voted. */
export type LocalRound = {
	id: string;
	/**
	 * The prompt both builds were given. It lives on the ROUND rather than beside
	 * the masthead, because it is the question this pair is an answer to — a
	 * prompt held anywhere else can drift from the scenes under it silently.
	 */
	prompt: string;
	/** Share of previous voters who picked the LEFT build. */
	leftShare: number;
	cells: readonly [LocalCell, LocalCell];
};

// Built ONCE at module load, never per render. OrbitViewer reloads its engine
// whenever the source's identity changes, so rebuilding these inside a component
// would restart every scene on every render.
//
// #TODO: the run ids are pinned here, so re-running "ahhhhhhhh" under a new name
// silently costs those cells their walkthrough (the dollhouse and splat keep
// working, which is what makes it silent). Rounds should arrive from the server
// carrying their own cell keys — same #TODO as `elo` and `leftShare`.
export const LOCAL_ROUNDS: readonly LocalRound[] = [
	{
		id: "modern-house",
		// THE FIRST ROUND IS A REAL MATCHUP, and it is the first one that has ever
		// been able to be: one prompt, two models, nothing else different. Both cells
		// are `modern-house`, so the question the page asks — who built it better —
		// is finally the question the pair answers.
		prompt: "A modern house",
		leftShare: 44,
		// BOTH CHECKED IN, which is what makes this the opener. A first visit waits
		// on these four files and nothing else; every later round is one the viewer
		// is already watching something during, and can stream.
		cells: [
			{
				id: "modern-house-gemini-flash",
				slot: "modern-house",
				model: "Gemini Flash",
				elo: 2091,
				source: cellSource(
					"ahhhhhhhh/modern-house/gemini-flash",
					"modern-house-gemini-flash",
				),
			},
			{
				id: "modern-house-opus-new",
				slot: "modern-house",
				model: "Claude Opus",
				elo: 2108,
				// The splat here was trained under the `compare` run while the mesh
				// comes from `ahhhhhhhh` — different runs of the same slot and model.
				// They are the same build: measured, the splat spans 35.914 m against
				// the dollhouse's 36.0 m and its root bbox is exactly the scene's
				// [-13,0,-18]→[13,9,18]. Brush exports in the scene's own frame (unlike
				// Postshot, which recentres), so it needs no transform — see
				// IDENTITY_TRANSFORM in splatLayer.ts.
				source: cellSource(
					"ahhhhhhhh/modern-house/opus-new",
					"modern-house-opus-new",
					// #TODO TEMPORARY — DELETE WITH A FIXED ENCODE. This bundle's means
					// decode to a centre near (-55, +52, +17) while its mesh sits at the
					// origin, so without this the splat renders off in the middle
					// distance and the panel looks empty. Numbers are the measured
					// offset, negated; see TourSource.splatTransform.
					{ position: [55.6, -52.4, -16.9], rotation: [0, 0, 0], scale: 1 },
				),
			},
		],
	},
	{
		id: "platformer-hotel",
		// #TODO: STILL TWO DIFFERENT PROMPTS, and the prompt shown belongs to neither
		// of them cleanly. The round above is what a matchup should look like; this
		// one stays only because these are the other cells that exist with tours and
		// splats captured, and the swap has to be exercised against something. It
		// goes the moment a second slot has two models built.
		prompt: "A super mario style platformer level",
		leftShare: 61,
		// Streamed rather than checked in: by the time this comes up the viewer has
		// been looking at the opener for a while, which is the budget these load
		// against — and the warm-up starts a round early besides (see page.tsx).
		cells: [
			{
				id: "platformer-level-opus-new",
				slot: "platformer-level",
				model: "Claude Opus",
				elo: 2108,
				source: cellSource(
					"ahhhhhhhh/platformer-level/opus-new",
					"platformer-level-opus-new",
				),
			},
			{
				id: "hotel-room-gemini-pro",
				slot: "hotel-room",
				model: "Gemini Pro",
				elo: 2143,
				source: cellSource("good_opus_new_hotel2/hotel-room/gemini-pro"),
			},
		],
	},
];

// The blanket "both rounds pair two different prompts" note that used to sit here
// is gone: the opener is a genuine matchup now. What remains of it is stated on the
// second round, which is the only one it is still true of.
