"use client";

import { useEffect, useState } from "react";

// THE ARC IS A TRUE CIRCLE, AND IT IS ENORMOUS. What follows is the comp's own
// construction rather than an approximation of it.
//
// SAG is the whole design. It is the depth of the arc at the ends of its chord —
// how far the limb falls from there to its lowest point — and at 32px it is the
// single number that decides how curved the masthead reads. Everything else here
// is derived from it.
//
// THE CHORD RUNS PAST THE WINDOW, 60px beyond each edge. That is why the limb is
// still descending where it leaves the frame instead of levelling off into it: an
// arc that ended its chord exactly at the window would meet the edge at its own
// flattest, which reads as a band with rounded corners rather than as the top of
// something round. The visible curve is the MIDDLE of a slightly wider arc.
const SAG = 32;
const OVERHANG = 60;

// THE CIRCLE THROUGH A CHORD, from the sagitta: for a half-chord `hw` and a sag
// `s`, the radius is (hw² + s²) / 2s, so the DIAMETER is that without the 2. The
// element is then simply a square of that size with a 50% radius, and its bottom
// cap is the limb.
//
// A TRUE CIRCLE RATHER THAN AN ELLIPSE, and the difference is visible even this
// shallow. The shape that was here — a box twice the window wide with elliptical
// bottom corners — has its curvature INCREASING toward the window edges, because
// the window sees out to half the ellipse's own radius. A circle this large is
// working within 8% of its radius, where it is indistinguishable from a parabola:
// the drop at the quarter-point measures 0.250 of the drop at the edge against the
// ellipse's 0.237. Small numbers, but they are the difference between a limb that
// curves evenly across the frame and one that turns down as it reaches the corners.
const diameter = (width: number) => {
	const hw = width / 2 + OVERHANG;
	return Math.round((hw * hw + SAG * SAG) / SAG);
};

/**
 * The paper the masthead is printed on: the bottom cap of a very large disc.
 *
 * THE SIZE HAS TO BE MEASURED, because CSS cannot do it. The diameter is a chord
 * SQUARED over a sag, and `calc()` has no way to multiply one length by another —
 * so the width is read from the window and written here, exactly as the comp does
 * it. 1440 is the fallback the comp uses server-side and is kept for the same
 * reason: it has to render to something before there is a window to ask.
 *
 * THE SURFACE IS FLAT. It carried an engraved one: four glints, twelve craters
 * and four maria, laid out as a 1600px tile and walked sideways over 140 seconds
 * so the moon appeared to turn. It is gone, and not because it was expensive —
 * at this scale the marks read as blotches on the paper rather than as relief on
 * a body, and the masthead is a surface the site PRINTS ON. Anything mottled
 * behind a wordmark and a prompt competes with them for the same attention and
 * loses, twice: the texture is too faint to be legible as a moon and too present
 * to ignore.
 *
 * The `moon-drift` keyframes in globals.css are now unused and can go with it.
 */
export default function MoonLimb() {
	const [d, setD] = useState(() => diameter(1440));

	useEffect(() => {
		const sync = () => setD(diameter(window.innerWidth));
		sync();
		window.addEventListener("resize", sync);
		return () => window.removeEventListener("resize", sync);
	}, []);

	return (
		<div aria-hidden className="absolute inset-0 overflow-hidden">
			<div
				className="absolute bottom-0 left-1/2 -translate-x-1/2 rounded-full bg-paper"
				style={{
					width: d,
					height: d,
					// THE LIMB IS SHADED, not just cut. An inset shadow hugging the bottom
					// edge follows the circle all the way round, so the paper darkens as it
					// turns away — which is what stops a very shallow arc reading as a flat
					// band that happens to have a curved bottom.
					boxShadow: "inset 0 -26px 48px -22px rgba(9,11,16,0.32)",
				}}
			/>
		</div>
	);
}
