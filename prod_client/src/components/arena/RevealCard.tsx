"use client";

import { useEffect, useState } from "react";
import { easeExpo, useProgress } from "./useProgress";

// The reveal is staged rather than shown at once: the rating counts to its new
// value, and only once it settles does the number it came FROM slide in behind it.
// Showing both from the start turns a movement into a table.
const COUNT_DELAY_MS = 620;
const COUNT_MS = 1050;
const HISTORY_DELAY_MS = COUNT_DELAY_MS + COUNT_MS + 300;

/**
 * Who built it, and what the vote cost them — shown under a panel once the
 * comparison is over.
 *
 * `won` drives the whole treatment. The winner inverts to the site's foreground
 * the way the CTA does, so the eye lands on it first; the loser drops to a third
 * of its contrast rather than turning red, because it did not fail at anything —
 * it came second.
 */
export default function RevealCard({
	model,
	elo,
	delta,
	won,
	align,
}: {
	model: string;
	elo: number;
	delta: number;
	won: boolean;
	align: "left" | "right";
}) {
	const t = useProgress(COUNT_MS, COUNT_DELAY_MS, easeExpo);
	const [showHistory, setShowHistory] = useState(false);

	useEffect(() => {
		const timer = window.setTimeout(() => setShowHistory(true), HISTORY_DELAY_MS);
		return () => clearTimeout(timer);
	}, []);

	const live = Math.round(delta * t);
	const sign = live > 0 ? `+${live}` : live < 0 ? String(live) : "±0";

	const chip = (
		<span
			className={`rounded-xs border px-2 py-1 font-sans text-[clamp(10px,0.85vw,15px)] font-medium tabular-nums ${
				delta > 0
					? "border-background/25 bg-background text-foreground"
					: "border-foreground/25 bg-transparent"
			}`}
		>
			{sign}
		</span>
	);

	return (
		<div
			// Full width of its half, and its content pushed APART: the model name
			// sits at the outer edge of the screen and the rating toward the middle,
			// so the two cards frame the control between them instead of floating in
			// the gap. `align` is the side of the SCREEN this card is on.
			className={`flex w-full items-center justify-between gap-[clamp(12px,1.6vw,28px)] rounded-xs border px-[clamp(14px,1.8vw,30px)] py-[clamp(10px,1.3vh,18px)] transition-colors duration-500 ${
				won
					? "border-foreground bg-foreground text-background"
					: "border-white/10 bg-transparent text-foreground/40"
			} ${align === "right" ? "flex-row-reverse" : ""}`}
			style={{ animation: "arena-rise 320ms ease both" }}
		>
			<div className={`flex min-w-0 flex-col gap-1 ${align === "right" ? "items-end" : ""}`}>
				<span className="font-sans text-[clamp(9px,0.7vw,12px)] font-medium tracking-[0.2em] opacity-60">
					BUILT BY
				</span>
				<span className="truncate font-display text-[clamp(14px,1.35vw,24px)] font-semibold tracking-[-0.015em]">
					{model}
				</span>
			</div>

			{/* THE NUMBERS NEVER MIRROR, THE CHIP DOES. `old → new` is a sentence and
			    reads one way on both sides; reversing it would leave the arrow
			    pointing back at where the rating came from. The chip is not a
			    sentence — it is the headline — so it hops to whichever end faces the
			    middle, putting both results' outcomes either side of the control
			    between them rather than at the far edges of the screen. */}
			<div className="flex items-center gap-[clamp(6px,0.8vw,12px)]">
				{align === "right" && chip}
				{/* The rating it came from, revealed after the count lands. Width is
				    what animates, not opacity alone — the row grows to admit it, so
				    nothing jumps. */}
				<span
					className="flex items-center gap-1.5 overflow-hidden whitespace-nowrap tabular-nums opacity-55 transition-all duration-500"
					style={{
						maxWidth: showHistory ? "7em" : "0em",
						opacity: showHistory ? 0.55 : 0,
					}}
				>
					<span className="font-sans text-[clamp(11px,1vw,18px)]">{elo}</span>
					<span className="text-[clamp(11px,1vw,18px)] opacity-60">→</span>
				</span>
				<span className="font-sans text-[clamp(13px,1.15vw,21px)] tabular-nums">
					{elo + live}
				</span>
				{align === "left" && chip}
			</div>
		</div>
	);
}
