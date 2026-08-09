"use client";

import { buildStep } from "./buildSequence";

/**
 * THE JOIN BETWEEN THE TWO SCENES.
 *
 * Absolutely centred on the arena, not a flex column and not a child of either
 * panel. An empty stretched flex item has no intrinsic height and was collapsing
 * to nothing; a line nested in the left panel is half-owned by the right panel's
 * later paint. This box has a real `top` + `bottom` (so it always has height) and
 * sits above both panels in DOM order with a z-index, so the canvases cannot
 * composite through it.
 */
export default function ArenaSeam({
	built,
	untuck,
	roundKey,
	shown,
}: {
	built: boolean;
	untuck: boolean;
	roundKey: string;
	shown: boolean;
}) {
	if (!shown) return null;

	return (
		<div
			key={roundKey}
			aria-hidden
			className="pointer-events-none absolute top-0 left-1/2 z-10 hidden origin-top -translate-x-1/2 bg-mark transition-[scale] md:block"
			style={{
				width: "var(--seam-width, 3px)",
				// Lands on the vote bar's top edge — same clearance the line had when
				// it lived on the left panel.
				bottom: "calc(var(--seam-drop, 0px) + var(--seam-break, 0px))",
				willChange: "scale",
				scale: built ? "1 1" : "1 0",
				...buildStep("ray", built),
				...(untuck
					? {
							animation: `seam-untuck ${buildStep("ray", true).transitionDuration} cubic-bezier(0.16,0.84,0.28,1) ${buildStep("ray", true).transitionDelay} backwards`,
						}
					: null),
			}}
		/>
	);
}
