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
function cellSource(cell: string, dir?: string): TourSource {
	// Every walkthrough asset is named by the manifest as a bare filename relative
	// to tour.json, and publish.py puts the whole set under one prefix — so one
	// resolver serves panos, proxy and minimap slices alike.
	const tour = (file: string) => assetUrl(`${cell}/tour/${file}`);
	return {
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
		id: "platformer",
		prompt: "A super mario style platformer level",
		leftShare: 38,
		cells: [
			{
				id: "modern-house-gemini-flash",
				slot: "modern-house",
				model: "Gemini Flash",
				elo: 2091,
				// The checked-in pair, and the reason it is the OPENING round: these
				// two are the only assets a first visit waits on.
				source: cellSource(
					"ahhhhhhhh/modern-house/gemini-flash",
					"modern-house-gemini-flash",
				),
			},
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
		],
	},
	{
		id: "hotel-room",
		prompt: "A hotel room",
		leftShare: 61,
		// Streamed in full — no checked-in copy. By the time this round comes up the
		// viewer has been looking at the previous one for a while, which is the
		// budget these two are loaded against.
		cells: [
			{
				id: "hotel-room-gemini-pro",
				slot: "hotel-room",
				model: "Gemini Pro",
				elo: 2143,
				source: cellSource("good_opus_new_hotel2/hotel-room/gemini-pro"),
			},
			{
				id: "modern-house-opus-new",
				slot: "modern-house",
				model: "Claude Opus",
				elo: 2108,
				source: cellSource("ahhhhhhhh/modern-house/opus-new"),
			},
		],
	},
];

// #TODO: BOTH ROUNDS PAIR TWO DIFFERENT PROMPTS AGAINST EACH OTHER, which is not
// what the page claims to be showing — a matchup is supposed to be one prompt
// built twice. These are the cells that exist with tours and splats captured, so
// they are what the swap can be exercised against; the moment a slot has two
// models captured, a round should be built from that instead.
