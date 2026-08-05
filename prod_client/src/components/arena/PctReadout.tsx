"use client";

import { easeOutCubic, useProgress } from "./useProgress";

// STRAIGHT AWAY. It was held 1.65 s so as not to talk over the rating counting up
// — what you chose, then what it cost, then what everyone else chose. There is no
// count any more (see RevealCard), so the only thing the delay was doing was
// holding a zero on screen and then animating it, which reads as the number being
// slow rather than as an order being kept.
const DELAY_MS = 0;
const COUNT_MS = 950;

/**
 * "VOTERS PICKED THIS — 62%", set into the corner of a panel.
 *
 * Sized off the PANEL rather than the viewport (the caller sets `--arena-pct`),
 * so the number stays proportionate when a panel is collapsed or expanded.
 */
export default function PctReadout({
	share,
	align,
}: {
	share: number;
	/** Which corner: the reading runs outward from the centre divide. */
	align: "left" | "right";
}) {
	const t = useProgress(COUNT_MS, DELAY_MS, easeOutCubic);

	return (
		<div
			// LIFTED CLEAR OF THE VOTE BAR. The bar floats over the scenes now rather
			// than sitting in a strip below them, and once a vote lands its two halves
			// open outward into exactly these corners — so a number parked at the
			// bottom of the panel ended up sliced in half by the card that came to
			// meet it. This is that bar's own offset plus its height, which is the
			// least it can be raised and still clear it.
			className={`pointer-events-none absolute bottom-[calc(var(--spacing-xl)*2.6)] z-20 flex flex-col gap-0.5 ${
				align === "left"
					? "left-lg items-start"
					: "right-lg items-end"
			}`}
			style={{ animation: "arena-rise 520ms cubic-bezier(0.25,0.8,0.3,1) both" }}
		>
			<span className="font-label text-2xs text-ink-40">
				VOTERS PICKED THIS
			</span>
			<span
				// Archivo at its heaviest. It used to be the display face light, which
				// now resolves to Anton — one weight, and a condensed poster face that
				// would set a two-digit number as a monument. The number is big
				// already; it does not also need to be loud in its letterforms.
				className="font-sans font-black leading-[0.86] tracking-[-0.045em] text-foreground tabular-nums"
				style={{
					fontSize: "var(--arena-pct, 64px)",
					textShadow: "0 4px 26px rgba(0,0,0,0.55)",
				}}
			>
				{Math.round(share * t)}%
			</span>
		</div>
	);
}
