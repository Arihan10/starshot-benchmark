/**
 * Standings for the leaderboard.
 *
 * #TODO: PLACEHOLDER DATA. These are made-up ratings for real model names, so the
 * screen can be built and laid out against something the right shape. Twenty-two of
 * them, which is more than any window fits — the point of the length is that "show
 * more" is a real state to look at rather than a button that never appears. When the
 * server owns votes this becomes a fetch — which is why the page that renders it is
 * a server component and this module exports a plain array rather than a hook:
 * swapping the source should not reach into the UI.
 *
 * ELO ORDER AND NOTHING ELSE, for now. A scene filter, a search and a three-way
 * sort all lived in this module and are gone: the first thing this page has to do
 * is be a ranking, and every control above the table is a reader deciding something
 * before being shown anything. They come back when the data behind them is real —
 * scene splits especially, which needs per-scene ratings the server does not yet
 * produce. `rank` is the row's own and survives whatever ordering arrives later.
 */
export type Standing = {
	rank: number;
	name: string;
	/** Who makes it. Set beside the name at a lower rank, never in place of it. */
	lab: string;
	elo: number;
	/** Points moved in the last seven days; negative is a fall. */
	delta: number;
	/** Share of head-to-heads won, 0–100. */
	winRate: number;
	votes: number;
};

export const STANDINGS: Standing[] = [
	{ rank: 1, name: "Claude Opus 4.5", lab: "Anthropic", elo: 1487, delta: 23, winRate: 64.2, votes: 12840 },
	{ rank: 2, name: "Gemini 3 Pro", lab: "Google", elo: 1462, delta: 11, winRate: 61.8, votes: 11930 },
	{ rank: 3, name: "GPT-5.2", lab: "OpenAI", elo: 1441, delta: -8, winRate: 59.4, votes: 13102 },
	{ rank: 4, name: "Claude Sonnet 4.5", lab: "Anthropic", elo: 1398, delta: 16, winRate: 55.1, votes: 9874 },
	{ rank: 5, name: "Grok 4", lab: "xAI", elo: 1362, delta: -14, winRate: 51.7, votes: 8216 },
	{ rank: 6, name: "Gemini 3 Flash", lab: "Google", elo: 1330, delta: 5, winRate: 48.9, votes: 7405 },
	{ rank: 7, name: "GPT-5.2 mini", lab: "OpenAI", elo: 1294, delta: -3, winRate: 45.2, votes: 6988 },
	{ rank: 8, name: "DeepSeek V4", lab: "DeepSeek", elo: 1271, delta: 27, winRate: 43.8, votes: 6410 },
	{ rank: 9, name: "Llama 4 Maverick", lab: "Meta", elo: 1251, delta: 9, winRate: 41.6, votes: 5312 },
	{ rank: 10, name: "Qwen 3 Max", lab: "Alibaba", elo: 1233, delta: 4, winRate: 39.9, votes: 5087 },
	{ rank: 11, name: "Mistral Large 3", lab: "Mistral", elo: 1218, delta: -6, winRate: 38.4, votes: 4760 },
	{ rank: 12, name: "Kimi K2", lab: "Moonshot", elo: 1201, delta: 18, winRate: 36.7, votes: 4402 },
	{ rank: 13, name: "GLM-5", lab: "Zhipu", elo: 1184, delta: -2, winRate: 35.1, votes: 3961 },
	{ rank: 14, name: "Command R+ 2", lab: "Cohere", elo: 1166, delta: 7, winRate: 33.4, votes: 3617 },
	{ rank: 15, name: "Nova Pro 2", lab: "Nova", elo: 1149, delta: -11, winRate: 31.8, votes: 3344 },
	{ rank: 16, name: "Doubao Pro 2", lab: "Doubao", elo: 1132, delta: 3, winRate: 30.2, votes: 3105 },
	{ rank: 17, name: "Phi-5", lab: "Microsoft", elo: 1114, delta: -5, winRate: 28.6, votes: 2870 },
	{ rank: 18, name: "MiniMax M2", lab: "Minimax", elo: 1097, delta: 12, winRate: 27.1, votes: 2588 },
	{ rank: 19, name: "Nemotron 5", lab: "Nvidia", elo: 1078, delta: -9, winRate: 25.4, votes: 2301 },
	{ rank: 20, name: "Yi Lightning 2", lab: "Yi", elo: 1061, delta: 1, winRate: 23.8, votes: 2064 },
	{ rank: 21, name: "Step-3", lab: "Stepfun", elo: 1042, delta: -4, winRate: 22.1, votes: 1795 },
	{ rank: 22, name: "Jamba 2 Large", lab: "Ai21", elo: 1024, delta: 6, winRate: 20.5, votes: 1533 },
];

// `MIN_ROWS` lived here, for a table that measured how many rows fitted the window
// and hid the rest behind a "show more". The board scrolls inside its own box now,
// so every row is always reachable and there is no floor to enforce.