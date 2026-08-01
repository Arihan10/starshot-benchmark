"use client";

import type { LocalCell } from "@/lib/localScenes";
import RevealCard from "./RevealCard";
import VoteButton from "./VoteButton";

/**
 * The strip under the two scenes, in both of its states.
 *
 * ONE BAR SPANNING BOTH PANELS rather than a footer inside each. Before the vote
 * the difference is invisible — two halves, a button in each, sitting under their
 * own scene. After it, it is the whole point: the result reads as a single
 * sentence with "next pair" as its full stop, and per-panel footers could only
 * ever put that control beside the pair or beneath it, never between them.
 *
 * The cards also swap which way they face. The percentage in each panel hugs the
 * OUTER corner, away from the divide; the cards hug the INNER edge, so the
 * reading collapses toward the middle where the next action is.
 */
export default function VoteBar({
	cells,
	vote,
	swing,
	onVote,
	onNext,
}: {
	cells: readonly LocalCell[];
	vote: "a" | "b" | null;
	/** Rating points the winner gains and the loser drops. */
	swing: number;
	onVote: (side: "a" | "b") => void;
	onNext: () => void;
}) {
	const [left, right] = cells;
	// No horizontal inset: the scenes above run edge to edge, so the strip under
	// them does too — an inset bar would draw a margin the rest of the page does
	// not have.
	const gutter = "pb-[clamp(10px,1.5vh,22px)] pt-[clamp(8px,1.1vh,16px)]";

	if (vote === null) {
		return (
			<div className={`flex flex-none gap-[clamp(8px,1vw,18px)] ${gutter}`}>
				<div className="min-w-0 flex-1">
					<VoteButton label="A wins" onVote={() => onVote("a")} />
				</div>
				<div className="min-w-0 flex-1">
					<VoteButton label="B wins" onVote={() => onVote("b")} />
				</div>
			</div>
		);
	}

	return (
		<div
			className={`grid flex-none grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-[clamp(10px,1.6vw,30px)] ${gutter}`}
		>
			<div className="flex min-w-0">
				<RevealCard
					model={left.model}
					elo={left.elo}
					delta={vote === "a" ? swing : -swing}
					won={vote === "a"}
					align="left"
				/>
			</div>

			{/* Emerges between the two results once they are in — the one thing left
			    to do, in the one place the eye is already looking. */}
			<button
				type="button"
				onClick={onNext}
				className="cursor-pointer rounded-xs border border-foreground bg-foreground px-[clamp(18px,2.2vw,40px)] py-[clamp(10px,1.4vh,19px)] font-sans text-[clamp(10px,0.8vw,14px)] font-medium tracking-[0.18em] whitespace-nowrap text-background transition-colors duration-200 hover:bg-[#d8dae0] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/50 focus-visible:ring-offset-2 focus-visible:ring-offset-black"
				style={{ animation: "arena-rise 420ms cubic-bezier(0.25,0.8,0.3,1) 260ms both" }}
			>
				NEXT PAIR
			</button>

			<div className="flex min-w-0">
				<RevealCard
					model={right.model}
					elo={right.elo}
					delta={vote === "b" ? swing : -swing}
					won={vote === "b"}
					align="right"
				/>
			</div>
		</div>
	);
}
