"use client";

import type { ReactNode } from "react";
import type { LocalCell } from "@/lib/localScenes";
import NextTimer from "./NextTimer";
import RevealCard from "./RevealCard";
import VoteButton from "./VoteButton";

// THE SIDES GROW, THE MIDDLE NEVER MOVES. Both halves of the bar are given an
// explicit width and the middle keeps the one it had, so a centred row can only
// resolve the extra width by pushing its two ends OUTWARD — the vote you cast
// opens into what it cost, away from the centre, while the thing between them
// holds still. A row of `auto` widths would slide everything sideways as the
// content changed, and read as a swap rather than as an opening.
const SIDE_VOTING = "clamp(150px,15.5vw,248px)";
const SIDE_REVEALED = "clamp(236px,27vw,428px)";
const MIDDLE = "clamp(78px,7.4vw,118px)";
const EXPAND = "620ms cubic-bezier(0.25,0.8,0.3,1)";

// MODULE SCOPE, not a closure inside the bar. Declared during render this is a new
// component type on every pass, so React tears the subtree down and builds it
// again each time — and the card inside counts a rating up from zero when it
// mounts. It would have restarted that count on every render of the page.
function Side({ expanded, children }: { expanded: boolean; children: ReactNode }) {
	return (
		<div
			className="flex min-w-0"
			style={{
				width: expanded ? SIDE_REVEALED : SIDE_VOTING,
				transition: `width ${EXPAND}`,
			}}
		>
			{children}
		</div>
	);
}

/**
 * The control the whole page is asking you to use, floating over the scenes at
 * the bottom of the frame.
 *
 * ONE BAR, TWO STATES, THE SAME THREE BOXES. Before the vote it is a segmented
 * control: the two builds, and a way past them. After it, each side has become
 * the result for the build it voted on and the middle has become the wait for the
 * next pair. Nothing is torn down in between, which is what keeps the winner's
 * ground from blinking — it was the site's foreground as a button and stays that
 * colour as a card, while the loser's fades out from under it.
 */
export default function VoteBar({
	cells,
	vote,
	swing,
	onVote,
	onNext,
	paused = false,
}: {
	cells: readonly LocalCell[];
	vote: "a" | "b" | null;
	/** Rating points the winner gains and the loser drops. */
	swing: number;
	onVote: (side: "a" | "b") => void;
	onNext: () => void;
	/** A scene is being toured — hold the countdown. See NextTimer. */
	paused?: boolean;
}) {
	const [left, right] = cells;
	const voted = vote !== null;

	return (
		// FLUSH, no gutters: the three segments are one slab, and the black middle is
		// the only seam it needs. `items-stretch` so that middle is exactly as tall
		// as whatever is beside it, in both states, without either being told a
		// height.
		<div className="flex items-stretch">
			<Side expanded={voted}>
				{voted ? (
					<RevealCard
						model={left.model}
						elo={left.elo}
						delta={vote === "a" ? swing : -swing}
						won={vote === "a"}
						align="left"
					/>
				) : (
					<VoteButton label="A wins" side="a" onVote={() => onVote("a")} />
				)}
			</Side>

			{/* The centre always means "move on" — before the vote a way to decline
			    it, after it the wait for the next one. Only the sides change what
			    they are. */}
			<div className="flex" style={{ width: MIDDLE }}>
				{voted ? (
					<NextTimer onNext={onNext} paused={paused} />
				) : (
					// INVERTED, not dimmed. The three segments are one slab of cream and
					// this one is cut out of it in black — which is what makes the middle
					// read as a different KIND of answer rather than a quieter version of
					// the two beside it. Hovering spreads its letters instead of changing
					// its colour, since there is no colour left to go to.
					<button
						type="button"
						onClick={onNext}
						className="w-full cursor-pointer bg-background px-[clamp(6px,0.7vw,12px)] font-sans text-[clamp(11px,0.95vw,16px)] font-black tracking-[0.06em] whitespace-nowrap text-foreground transition-[letter-spacing] duration-150 hover:tracking-[0.12em] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/50 focus-visible:ring-offset-2 focus-visible:ring-offset-black"
					>
						SKIP
					</button>
				)}
			</div>

			<Side expanded={voted}>
				{voted ? (
					<RevealCard
						model={right.model}
						elo={right.elo}
						delta={vote === "b" ? swing : -swing}
						won={vote === "b"}
						align="right"
					/>
				) : (
					<VoteButton label="B wins" side="b" onVote={() => onVote("b")} />
				)}
			</Side>
		</div>
	);
}
