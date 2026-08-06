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
				source: cellSource(
					"ahhhhhhhh/modern-house/opus-new",
					"modern-house-opus-new",
					// #TODO temporary — delete once the SOG encode is fixed.
					{ position: [55.6, -52.4, -16.9], rotation: [0, 0, 0], scale: 1 },
				),
			},
		],
	},
	{
		id: "platformer-hotel",
		// #TODO: still two different prompts; shown prompt belongs to neither cleanly.
		prompt: "A super mario style platformer level",
		leftShare: 61,
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
