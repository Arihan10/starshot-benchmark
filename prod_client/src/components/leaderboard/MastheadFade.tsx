"use client";

import type { ReactNode } from "react";
import { useBoard } from "./heroProgress";

/**
 * Shows a masthead only while the page is settled on the section it belongs to.
 *
 * WHY THE MASTHEAD GOES AWAY AT ALL. It is the one piece of furniture that is in
 * the same place on both sections, so scrolling past it would slide one moon out
 * of the top of the window while an identical one slid in underneath — which reads
 * as a stutter rather than as a page turning. Taking it out for the length of the
 * journey and bringing it back on arrival is what makes the two sections feel like
 * two places.
 *
 * TWO OF THEM, ONE PER SECTION, and each in its section's own flow — which is what
 * reserves the space it occupies, with no measuring and no magic numbers. The cost
 * is that the navbar exists twice in the document, so the hidden one is `inert`:
 * without it a keyboard would tab into an invisible copy of the site navigation,
 * and a screen reader would find two of every link.
 */
export default function MastheadFade({
	on,
	children,
}: {
	/** Which resting place this masthead belongs to. */
	on: "hero" | "board";
	children: ReactNode;
}) {
	const { phase } = useBoard();
	const shown = phase === on;

	return (
		<div
			inert={!shown}
			// AND THE MOON IS TOLD TO IGNORE THIS ONE. Both sections carry a masthead,
			// so both carry an anchor; without this the moon would chase whichever came
			// first in the document — including the band that has scrolled a whole
			// screen away. Marked idle, the hidden one is skipped, and while NEITHER is
			// shown the moon simply holds its place and dims, which is what it did when
			// it was part of the band.
			{...(shown ? {} : { "data-moon-idle": true })}
			className={`flex-none transition-opacity duration-settle ${
				shown ? "opacity-100" : "opacity-0"
			}`}
		>
			{children}
		</div>
	);
}
