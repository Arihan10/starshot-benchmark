"use client";

import { useEffect, useRef, useState } from "react";

// THE ARC IS A TRUE CIRCLE, AND IT IS ENORMOUS. What follows is the comp's own
// construction rather than an approximation of it.
//
// SAG is the whole design. It is the depth of the arc at the ends of its chord —
// how far the limb falls from there to its lowest point — and at 32px it is the
// single number that decides how curved the masthead reads. Everything else here
// is derived from it.
//
// THE CHORD RUNS PAST THE HOST, 60px beyond each edge. That is why the limb is
// still descending where it leaves the frame instead of levelling off into it: an
// arc that ended its chord exactly at the frame would meet the edge at its own
// flattest, which reads as a band with rounded corners rather than as the top of
// something round. The visible curve is the MIDDLE of a slightly wider arc.
//
// WHICH IS WHY THE CHORD IS AN ARGUMENT and the overhang only its default: a chord
// NARROWER than the host puts both its ends inside the frame, and the same
// construction then draws a body bulging out of whatever is above it.
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
const diameter = (chord: number, sag: number) => {
	const hw = chord / 2;
	return Math.round((hw * hw + sag * sag) / sag);
};

// THE FEATHER'S OWN CURVE, and it cannot be a straight ramp. Alpha interpolates
// LINEARLY, and the paper is 237 against a page of 0 — so a straight ramp is still
// carrying a mid grey a couple of pixels from where it lands and reads as a soft
// band with a hard line under it, which is the one thing this figure exists to
// avoid. These are (1 − t)^2.2 at the quarter points: the light is mostly gone by
// the halfway mark and the last quarter is a tail you cannot find the end of.
const FEATHER = [1, 0.52, 0.22, 0.055, 0];

/**
 * The paper the masthead is printed on: the bottom cap of a very large disc.
 *
 * THE SIZE HAS TO BE MEASURED, because CSS cannot do it. The diameter is a chord
 * SQUARED over a sag, and `calc()` has no way to multiply one length by another —
 * so the box is read from the DOM and written here, exactly as the comp does it.
 * 1440 is the fallback the comp uses server-side and is kept for the same reason:
 * it has to render to something before there is a box to ask.
 *
 * BOTH DIMENSIONS COME OFF THE HOST rather than off the window, which is what lets
 * the same construction draw a tongue a fraction of the page wide. The surface
 * needs both numbers — it is drawn on the host's own footprint so that a
 * percentage across or up it means what it says — and neither is knowable ahead of
 * layout: the height comes off the spacing tokens and the bar's own content. A
 * ResizeObserver on this element is the only thing that sees them.
 */
export default function MoonLimb({
	sag = SAG,
	chord = (width) => width + 2 * OVERHANG,
	fade,
	shade = true,
}: {
	/**
	 * How far the arc falls below its chord.
	 *
	 * `"host"` makes it the host's full height, which lands the chord's ENDS on the
	 * host's top edge. Pair that with a chord equal to the host and the limb is a
	 * moon segment from corner to corner; pair it with a narrower chord and the
	 * same construction draws a body bulging out of whatever sits above it. Same
	 * circle either way — the only difference is where the chord is.
	 */
	sag?: number | "host";
	/** The chord's width, given the host's. */
	chord?: (hostWidth: number) => number;
	/**
	 * Feather the rim over this many pixels instead of ending it on a line.
	 *
	 * STRUCK ON THE CIRCLE ITSELF — same centre, same radius, so what softens is
	 * the limb rather than a horizontal band laid across it. A linear fade would
	 * be level while the edge it is supposed to be dissolving is not, and would
	 * therefore be widest at the window's edges and narrowest in the middle.
	 */
	fade?: number;
	shade?: boolean;
}) {
	const hostRef = useRef<HTMLDivElement>(null);
	const [box, setBox] = useState({ w: 1440, h: 112 });

	useEffect(() => {
		const host = hostRef.current;
		if (!host) return;
		const sync = () => {
			const w = host.offsetWidth;
			const h = host.offsetHeight;
			setBox((prev) => (prev.w === w && prev.h === h ? prev : { w, h }));
		};
		sync();
		const observer = new ResizeObserver(sync);
		observer.observe(host);
		return () => observer.disconnect();
	}, []);

	const d = diameter(chord(box.w), sag === "host" ? box.h : sag);
	const r = d / 2;

	// The centre sits a radius above the foot of the host, which is where the disc
	// is anchored. Absolute stops are lengths along the ray, so they describe the
	// ring in the circle's own terms and owe nothing to the gradient's box.
	const veil = fade
		? `radial-gradient(circle at 50% ${Math.round(box.h - r)}px, ${FEATHER.map(
				(alpha, i) =>
					`rgba(0,0,0,${alpha}) ${Math.round(r - fade + (fade * i) / (FEATHER.length - 1))}px`,
			).join(", ")})`
		: undefined;

	return (
		<div
			ref={hostRef}
			aria-hidden
			className="absolute inset-0 overflow-hidden"
			style={veil ? { maskImage: veil, WebkitMaskImage: veil } : undefined}
		>
			<div
				className="absolute bottom-0 left-1/2 -translate-x-1/2 rounded-full bg-paper"
				style={{
					width: d,
					height: d,
					// THE LIMB IS SHADED, not just cut. An inset shadow hugging the bottom
					// edge follows the circle all the way round, so the paper darkens as it
					// turns away — which is what stops a very shallow arc reading as a flat
					// band that happens to have a curved bottom. A feathered rim has no
					// edge to turn on and takes none of it.
					boxShadow: shade
						? "inset 0 -26px 48px -22px rgba(9,11,16,0.32)"
						: undefined,
				}}
			/>
		</div>
	);
}
