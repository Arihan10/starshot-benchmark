"use client";

import type { Ref } from "react";

/** How long the card takes to rise into place. Read by VoteBar, which derives the
 *  moment the whole reveal has stopped moving from it. */
export const RISE_MS = 400;

/**
 * Who built it — shown under a panel once the comparison is over.
 *
 * THE RATING IS STATED, NOT PERFORMED. This used to count the number up to its new
 * value, then slide the old one in behind it with a signed chip beside it: three
 * staged movements, about 1.7 s, to say something that does not change while you
 * read it. The count was also why everything else on the page was held back — the
 * crowd's percentage waited 1.65 s and the countdown 2.7 s rather than talk over
 * it — so taking it out is what let both of those start at once. What is left
 * answers the only question the reveal is asked: which model made this.
 *
 * `won` drives the whole treatment. The winner inverts to the site's foreground
 * the way the CTA does, so the eye lands on it first; the loser drops to a third
 * of its contrast rather than turning red, because it did not fail at anything —
 * it came second.
 */
export default function RevealCard({
	model,
	elo,
	won,
	align,
	ref,
}: {
	model: string;
	elo: number;
	won: boolean;
	align: "left" | "right";
	/** The bar measures this card to size both of its sides. See VoteBar. */
	ref?: Ref<HTMLDivElement>;
}) {
	return (
		<div
			ref={ref}
			// `min-w-max` IS THE NO-TRUNCATION RULE. The card fills its half of the bar
			// (`w-full`) but is never allowed narrower than its own content, so a long
			// model name pushes the card wider instead of losing its tail — and the bar
			// reads that width back off it to size both sides (see VoteBar). Between
			// them, the two mean the row fits whatever it is given rather than cropping
			// it to fit.
			//
			// Full width of its half, and its content pushed APART: the model name sits
			// at the outer edge of the screen and the rating toward the middle, so the
			// two cards frame the control between them instead of floating in the gap.
			// `align` is the side of the SCREEN this card is on.
			//
			// OPAQUE, both ways round. The bar floats over the scenes, and the winning
			// scene is lit from its own edges — so a card you can see through is a card
			// the winner's glow and the seam down the middle read straight across,
			// which puts the light in front of the control instead of behind it. The
			// loser's card still recedes; it does it by dropping its contrast, not by
			// letting the picture through.
			className={`flex w-full min-w-max items-center justify-between gap-md border px-md py-sm transition-colors duration-500 ${
				won
					? "border-mark bg-mark text-background"
					: "border-mark-8 bg-background text-ink-40"
			} ${align === "right" ? "flex-row-reverse" : ""}`}
			style={{ animation: `content-swap ${RISE_MS}ms ease both` }}
		>
			<div className={`flex flex-col gap-1 ${align === "right" ? "items-end" : ""}`}>
				<span className="font-label text-2xs opacity-60">BUILT BY</span>
				{/* Archivo at its heaviest, NOT the display face: `font-display` is the
				    wordmark's Anton, which has one weight and a poster's proportions —
				    a model name set in it would announce itself louder than the site
				    does. This is a result, so it is the interface face, set hard. */}
				<span className="font-sans text-base font-extrabold tracking-[-0.015em] whitespace-nowrap uppercase">
					{model}
				</span>
			</div>

			{/* LABELLED, now that nothing else says what it is. While the card carried
			    a signed chip beside it the number read as a rating by context; alone it
			    is just a figure. The label is the same voice as BUILT BY opposite, so
			    the card reads as two captioned facts rather than a name and a number.

			    Aligned toward the MIDDLE — the mirror of the model name, which sits at
			    the outer edge — so both cards point their rating at the control between
			    them. Tabular figures because the two are read against each other. */}
			<div
				className={`flex flex-col gap-1 ${
					align === "left" ? "items-end" : "items-start"
				}`}
			>
				<span className="font-label text-2xs opacity-60">ELO</span>
				<span className="font-sans text-sm tabular-nums">{elo}</span>
			</div>
		</div>
	);
}
