"use client";

import { useState } from "react";
import Fade from "@/components/site/Fade";
import Podium from "./Podium";
import StandingsTable from "./StandingsTable";
import type { Standing } from "@/lib/leaderboard";

/**
 * The two columns, and the one piece of state they share.
 *
 * WHY THIS COMPONENT EXISTS AT ALL: the board and the island are otherwise
 * strangers — one is a table of numbers, the other a WebGL canvas — and neither
 * should have to know about the other to do its own job. What joins them is a
 * single fact, WHICH MODEL IS BEING LOOKED AT, and this is the smallest place
 * that can hold it: the nearest ancestor of both.
 *
 * IT IS A CLIENT ISLAND INSIDE A SERVER PAGE. The route stays a server component
 * so its metadata and its data-loading stay on the server; only the part that
 * genuinely needs a hover — this — ships to the browser. Lifting the state into
 * the page would have turned the whole route client-side to carry one string.
 */
export default function LeaderboardStage({
	rows,
	top,
	foot,
}: {
	rows: Standing[];
	top: Standing[];
	foot: number;
}) {
	const [compare, setCompare] = useState<Standing | null>(null);

	// THE PODIUM'S OWN THREE ARE NEVER CHALLENGERS. Pointing at first place would
	// otherwise raise a fourth post identical in height to the one already
	// standing in the middle of the island, which reads as a duplicate rather than
	// as a comparison. Their pillars are already the answer.
	const rival =
		compare && top.some((r) => r.rank === compare.rank) ? null : compare;

	return (
		<div className="relative z-10 grid min-h-0 flex-1 grid-cols-[minmax(0,1.12fr)_minmax(0,1fr)]">
			{/* The clearance is a style rather than a utility so `foot` stays the one
			    place the exit bar's height is written. Interpolated into a class name
			    it would not survive the build — Tailwind reads source text, and a
			    class assembled at runtime is a class it never sees. */}
			<div
				className="flex min-h-0 flex-col pt-sm pr-md pl-lg"
				style={{ paddingBottom: foot }}
			>
				<Fade enter={640} delay={200} className="flex min-h-0 flex-1 flex-col">
					<StandingsTable rows={rows} onCompare={setCompare} />
				</Fade>
			</div>

			{/* THE CANVAS RUNS THE FULL COLUMN rather than stopping short of the exit
			    bar, so a block thrown off the coast stays drawn all the way down and
			    out. The clearance the bar needs is taken inside the fit instead. */}
			<div className="relative min-h-0">
				<Fade enter={640} delay={120} className="absolute inset-0">
					<Podium rows={top} compare={rival} foot={foot} />
				</Fade>
			</div>
		</div>
	);
}
