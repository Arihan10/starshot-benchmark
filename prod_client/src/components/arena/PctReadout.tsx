"use client";

import { easeOutCubic, useProgress } from "./useProgress";

// Held back until the shatter has played and the Elo has moved. The order is the
// point: what YOU chose, then what it cost, then what everyone else chose. Arriving
// first, the crowd's answer would colour your reading of your own.
const DELAY_MS = 1650;
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
			className={`pointer-events-none absolute bottom-[clamp(10px,1.6vh,26px)] z-20 flex flex-col gap-0.5 ${
				align === "left"
					? "left-[clamp(18px,1.8vw,34px)] items-start"
					: "right-[clamp(18px,1.8vw,34px)] items-end"
			}`}
			style={{ animation: "arena-rise 520ms cubic-bezier(0.25,0.8,0.3,1) both" }}
		>
			<span className="font-sans text-[clamp(9px,0.72vw,13px)] font-medium tracking-[0.22em] text-foreground/55">
				VOTERS PICKED THIS
			</span>
			<span
				className="font-display font-light leading-[0.86] tracking-[-0.045em] text-foreground tabular-nums"
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
