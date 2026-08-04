"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
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

// A ceiling, so a pathological model name cannot push the row off the screen.
// Two of these plus the middle still leaves a margin at either edge.
const SIDE_MAX = "44vw";

// MODULE SCOPE, not a closure inside the bar. Declared during render this is a new
// component type on every pass, so React tears the subtree down and builds it
// again each time — and the card inside counts a rating up from zero when it
// mounts. It would have restarted that count on every render of the page.
function Side({
	expanded,
	width,
	align,
	children,
}: {
	expanded: boolean;
	/** Measured content width, once the bar knows it. */
	width: number | null;
	align: "left" | "right";
	children: ReactNode;
}) {
	return (
		<div
			// CLIPPED, and anchored to the OUTER edge. While the row is opening, the
			// card inside is already at its full width and wider than this box — so
			// something has to give, and what gives is the end facing the middle. The
			// result reads as the card being uncovered from the outside in, rather
			// than as text squeezing itself into a growing gap.
			className={`flex overflow-hidden ${align === "right" ? "justify-end" : ""}`}
			style={{
				// `max()` so a measurement can only ever make the row wider than the
				// buttons it replaced. A pair of short model names would otherwise
				// close the bar on the vote, which is the opposite motion to the one
				// the reveal is supposed to make.
				width: expanded
					? width
						? `max(${SIDE_VOTING}, ${width}px)`
						: SIDE_REVEALED
					: SIDE_VOTING,
				maxWidth: SIDE_MAX,
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
	/** `skip` is an answer too — it reveals the round without moving any rating. */
	vote: "a" | "b" | "skip" | null;
	/** Rating points the winner gains and the loser drops. */
	swing: number;
	onVote: (choice: "a" | "b" | "skip") => void;
	onNext: () => void;
	/** A scene is being toured — hold the countdown. See NextTimer. */
	paused?: boolean;
}) {
	const [left, right] = cells;
	const voted = vote !== null;
	// SKIPPING MOVES NOTHING. Both models keep the rating they came in with, and
	// the cards say so with ±0 rather than hiding — "no opinion" is a result about
	// the pair, and the reader has still earned the answer to who built them.
	const swingFor = (side: "a" | "b") =>
		vote === "skip" ? 0 : vote === side ? swing : -swing;

	// BOTH SIDES ARE AS WIDE AS THE WIDER RESULT. The cards refuse to be narrower
	// than their own content (`min-w-max` in RevealCard), so measuring one tells us
	// what it needs; taking the larger of the two and giving it to BOTH is what
	// keeps the row symmetrical about the middle while still fitting the longer
	// model name. A width guessed in CSS cannot do this — it does not know how wide
	// "Claude Sonnet 4.5" sets in Archivo at this viewport.
	const cardA = useRef<HTMLDivElement>(null);
	const cardB = useRef<HTMLDivElement>(null);
	const [revealWidth, setRevealWidth] = useState<number | null>(null);

	// Cleared during RENDER rather than from an effect, so no frame is ever painted
	// with the last pair's width — the next round is not these two models, and it
	// measures itself.
	const [wasVoted, setWasVoted] = useState(voted);
	if (voted !== wasVoted) {
		setWasVoted(voted);
		if (!voted) setRevealWidth(null);
	}

	useEffect(() => {
		if (!voted) return;
		const cards = [cardA.current, cardB.current].filter(Boolean) as HTMLElement[];
		if (cards.length === 0) return;

		// GROWS, NEVER SHRINKS, within one round. The card is `max(box, content)`, so
		// once the box has been set to the measurement the card reports the box back
		// — and a rule that also shrank would sit in a loop with its own output. It
		// also means the late arrival of the "from" rating, which widens the row a
		// second after the vote, simply widens it again.
		// `offsetWidth`, NOT a bounding rect. The card arrives on an entrance
		// animation that scales it to 0.96, and a bounding rect is the TRANSFORMED
		// box — so measuring that way reads a card 4% narrower than the one that will
		// be sitting there a moment later, and the row settles a few pixels short
		// with the end of the card clipped off. A ResizeObserver watches the border
		// box, which a transform does not touch, so nothing fires again to correct
		// it. `offsetWidth` is that same untransformed box.
		const measure = () => {
			const w = Math.ceil(Math.max(...cards.map((c) => c.offsetWidth)));
			if (w > 0) setRevealWidth((cur) => (cur === null || w > cur ? w : cur));
		};
		// No opening call: a ResizeObserver delivers the element's current size as
		// soon as it is observed, so the first measurement arrives on its own.
		const ro = new ResizeObserver(measure);
		for (const c of cards) ro.observe(c);
		return () => ro.disconnect();
	}, [voted]);

	return (
		// FLUSH, no gutters: the three segments are one slab, and the black middle is
		// the only seam it needs. `items-stretch` so that middle is exactly as tall
		// as whatever is beside it, in both states, without either being told a
		// height.
		<div className="flex items-stretch">
			<Side expanded={voted} width={revealWidth} align="left">
				{voted ? (
					<RevealCard
						ref={cardA}
						model={left.model}
						elo={left.elo}
						delta={swingFor("a")}
						won={vote === "a"}
						align="left"
					/>
				) : (
					<VoteButton label="A wins" side="a" onVote={() => onVote("a")} />
				)}
			</Side>

			{/* The centre always means "move on" — before the answer a way to
			    decline giving one, after it the wait for the next pair.

			    DECLINING IS STILL AN ANSWER, and it ends the round exactly as a vote
			    does: the models are named, the crowd's split is shown, and the same
			    clock runs. Skipping used to cut straight to the next pair, which
			    quietly punished the honest reply — you learned nothing about a pair
			    you had just spent time on, purely because you could not separate them.
			    The only difference is that the ratings do not move. */}
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
						onClick={() => onVote("skip")}
						className="w-full cursor-pointer bg-background px-[clamp(6px,0.7vw,12px)] font-sans text-[clamp(11px,0.95vw,16px)] font-black tracking-[0.06em] whitespace-nowrap text-foreground transition-[letter-spacing] duration-150 hover:tracking-[0.12em] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/50 focus-visible:ring-offset-2 focus-visible:ring-offset-black"
					>
						SKIP
					</button>
				)}
			</div>

			<Side expanded={voted} width={revealWidth} align="right">
				{voted ? (
					<RevealCard
						ref={cardB}
						model={right.model}
						elo={right.elo}
						delta={swingFor("b")}
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
