import type { TourSource } from "./orbit/types";
import { assetUrl, cfImageUrl } from "./r2";

function cellSource(
	cell: string,
	dir?: string,
	splatTransform?: TourSource["splatTransform"],
): TourSource {
	const tour = (file: string) => assetUrl(`${cell}/tour/${file}`);
	return {
		splatTransform,
		dollhouseUrl: dir
			? `/scenes/${dir}/scene-lite.glb`
			: assetUrl(`${cell}/published/scene-lite.glb`),
		splatUrl: dir
			? `/scenes/${dir}/trained.web.sog`
			: assetUrl(`${cell}/splat/trained.web.sog`),
		manifestUrl: tour("tour.json"),
		resolvePano: (file) => ({
			url: tour(file),
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
	slot: string;
	model: string;
	/** #TODO: hard-coded; should come from the pairing/leaderboard service. */
	elo: number;
	source: TourSource;
};

export type LocalRound = {
	id: string;
	prompt: string;
	leftShare: number;
	cells: readonly [LocalCell, LocalCell];
};

// #TODO: pinned run ids — rounds should arrive from the server with their own keys.
// #TODO temporary — one pair, so every round is the two modern houses.
export const LOCAL_ROUNDS: readonly LocalRound[] = [
	{
		id: "modern-house",
		prompt: "A modern house",
		leftShare: 44,
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
				// #TODO temporary — both sides render the Gemini Flash scene until
				// the opus-new SOG encode is fixed.
				source: cellSource(
					"ahhhhhhhh/modern-house/gemini-flash",
					"modern-house-gemini-flash",
				),
			},
		],
	},
];
