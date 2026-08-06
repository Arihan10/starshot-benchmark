"use client";

import { useEffect, useRef, useState } from "react";
import Fade from "@/components/site/Fade";
import Podium from "./Podium";
import StandingsTable from "./StandingsTable";
import type { Standing } from "@/lib/leaderboard";

// Where the two columns stop fitting beside each other and become two screens.
// Tailwind's `md`, written out because matchMedia cannot read a variant.
const WIDE = "(min-width: 48rem)";

const HINT =
	"font-mono text-[9.5px] font-bold tracking-[0.26em] uppercase text-ink-40 transition-colors duration-quick hover:text-ink";

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
 *
 * ON A PHONE THE COLUMNS BECOME SCREENS, and the board is the one you land on.
 * Side by side they would each get half of 390px, which is not a board and not a
 * city; stacked in a scroller they would each get half the height, which is the
 * squeezed strip the side-by-side layout was built to escape. So the page snaps:
 * the standings own the first screen outright and the island is a swipe away.
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

	const board = useRef<HTMLElement>(null);
	const city = useRef<HTMLElement>(null);

	// THE CITY IS NOT BUILT UNTIL IT IS ASKED FOR. It is a WebGL context and a few
	// thousand instanced blocks, and on a phone it starts a screen below the fold
	// — paid for in full before the board that is actually on screen has settled.
	// Opting in is the whole point of putting it on its own screen, so the mount
	// waits for the scroll. A wide window is opted in from the start: there both
	// columns are visible at once and there is nothing to opt into.
	const [armed, setArmed] = useState(false);
	useEffect(() => {
		const el = city.current;
		if (!el) return;

		const wide = window.matchMedia(WIDE);
		const sync = () => {
			if (wide.matches) setArmed(true);
		};
		sync();
		wide.addEventListener("change", sync);

		const io = new IntersectionObserver(
			([entry]) => {
				if (entry.isIntersecting) setArmed(true);
			},
			{ threshold: 0.2 },
		);
		io.observe(el);

		return () => {
			wide.removeEventListener("change", sync);
			io.disconnect();
		};
	}, []);

	return (
		<div className="relative z-10 min-h-0 flex-1 max-md:snap-y max-md:snap-mandatory max-md:overflow-y-auto max-md:overscroll-contain md:grid md:grid-cols-[minmax(0,1.12fr)_minmax(0,1fr)]">
			{/* The clearance is a style rather than a utility so `foot` stays the one
			    place the exit bar's height is written. Interpolated into a class name
			    it would not survive the build — Tailwind reads source text, and a
			    class assembled at runtime is a class it never sees. */}
			<section
				ref={board}
				className="flex min-h-0 flex-col pt-sm pr-md pl-lg max-md:h-full max-md:snap-start max-md:px-md"
				style={{ paddingBottom: foot }}
			>
				<Fade enter={640} delay={200} className="flex min-h-0 flex-1 flex-col">
					<StandingsTable rows={rows} onCompare={setCompare} />
				</Fade>

				{/* BOTH THE HINT AND THE WAY THERE. A label that only says a screen
				    exists leaves the reader to guess the gesture; one that also takes
				    them costs nothing more and answers it. */}
				<button
					type="button"
					onClick={() =>
						city.current?.scrollIntoView({ behavior: "smooth" })
					}
					className={`${HINT} flex flex-none cursor-pointer items-center justify-center gap-xs pt-sm md:hidden`}
				>
					The podium
					<span
						aria-hidden
						className="animate-[about-nudge_2.4s_ease-in-out_infinite] text-[11px]"
					>
						↓
					</span>
				</button>
			</section>

			{/* THE CANVAS RUNS THE FULL COLUMN rather than stopping short of the exit
			    bar, so a block thrown off the coast stays drawn all the way down and
			    out. The clearance the bar needs is taken inside the fit instead. */}
			<section
				ref={city}
				className="relative min-h-0 max-md:h-full max-md:snap-start"
			>
				{armed && (
					<Fade enter={640} delay={120} className="absolute inset-0">
						<Podium rows={top} compare={rival} foot={foot} />
					</Fade>
				)}

				{/* Off to one side, clear of the crown the pillars' name plates sit in. */}
				<button
					type="button"
					onClick={() =>
						board.current?.scrollIntoView({ behavior: "smooth" })
					}
					className={`${HINT} absolute top-xs left-md z-10 flex cursor-pointer items-center gap-xs md:hidden`}
				>
					<span aria-hidden className="text-[11px]">
						↑
					</span>
					Standings
				</button>
			</section>
		</div>
	);
}
