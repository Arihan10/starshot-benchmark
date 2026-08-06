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

// ---------------------------------------------------------------------------
// THE SURFACE, AND WHY IT IS BACK.
//
// An engraved surface was here once and was taken off for reading as blotches on
// the paper rather than as relief on a body. That diagnosis was right and the
// cause was not the marks being too strong — it was that they were FLAT. A disc of
// shadow with no lit side is a stain; the thing that makes the same disc read as a
// crater is that one wall of it catches the light and the opposite wall does not.
//
// So the marks are modelled now, in pairs, and BECAUSE they are modelled they can
// be much fainter than the ones that were removed and still be legible as a
// surface. A crater is about 14 levels of shadow against 9 of highlight on 237
// paper — a dimple you have to be looking for, rather than a mark on the page.
//
// LIT FROM THE UPPER LEFT, which is where `--moon-surface` puts its highlight and
// so where the light on this site comes from. A bowl lit from that corner is dark
// on its upper-left inner wall — the one facing away — and bright on its
// lower-right one. Reverse the pair and every crater turns into a dome.
//
// DRAWN ON THE MASTHEAD'S OWN BOX, which is why that box has to be measured. The
// disc is tens of thousands of pixels across, so nothing can be placed against IT
// — a percentage down its face lands a mile above the window and a pixel offset up
// from its foot is a different fraction of the bar at every width. The masthead is
// 112px tall at 1440 and about 68 at a narrow window, so marks pinned in pixels
// simply leave the top of the bar as it narrows. Both coordinates are percentages
// of the measured box instead: `up` from the limb, `x` across the window.
//
// THE FRINGES FALL PAST THE LIMB and the disc clips them — see `overflow-hidden`
// on it, which is what keeps a mare from smudging the black page below the paper.
//
// THE TEXTURE DRIFTS LEFT TO RIGHT while the lighting remains fixed. Two identical
// tiles trade places in a marquee, so the limb stays covered through every frame
// and the loop is invisible.
// ---------------------------------------------------------------------------

// The offset of each wall from the crater's centre, as a fraction of its radius.
const TILT = 0.2;

const HIGHLIGHT = 0.48;

const CRATERS: { x: number; up: number; r: number; a: number }[] = [
	{ x: 6, up: 70, r: 24, a: 0.059 },
	{ x: 15, up: 49, r: 14, a: 0.07 },
	{ x: 26, up: 79, r: 32, a: 0.048 },
	{ x: 37, up: 54, r: 18, a: 0.064 },
	{ x: 57, up: 57, r: 27, a: 0.05 },
	{ x: 68, up: 80, r: 16, a: 0.067 },
	{ x: 79, up: 52, r: 22, a: 0.056 },
	{ x: 96, up: 55, r: 19, a: 0.06 },
];

// The broad low ground between the craters. Far too faint to find on their own —
// they are what stops the paper reading as evenly white between the marks.
const MARIA: { x: number; up: number; rx: number; ry: number; a: number }[] = [
	{ x: 20, up: 64, rx: 250, ry: 76, a: 0.027 },
	{ x: 62, up: 77, rx: 310, ry: 64, a: 0.022 },
	{ x: 90, up: 61, rx: 210, ry: 72, a: 0.025 },
];

// The walls are offset in PIXELS off a percentage. A crater's two faces are a
// fixed distance apart — that is its radius, which does not change with the bar —
// so the offset cannot be a share of a box that does.
const wall = (c: (typeof CRATERS)[number]) =>
	`radial-gradient(circle ${c.r}px at calc(${c.x}% - ${(c.r * TILT).toFixed(1)}px) calc(${100 - c.up}% - ${(c.r * TILT).toFixed(1)}px), rgba(9,11,16,${c.a}) 0%, rgba(9,11,16,${(c.a * 0.45).toFixed(3)}) 58%, transparent 100%)`;

const lit = (c: (typeof CRATERS)[number]) =>
	`radial-gradient(circle ${Math.round(c.r * 0.86)}px at calc(${c.x}% + ${(c.r * TILT).toFixed(1)}px) calc(${100 - c.up}% + ${(c.r * TILT).toFixed(1)}px), rgba(255,255,255,${HIGHLIGHT}) 0%, transparent 100%)`;

const SURFACE = [
	...CRATERS.flatMap((c) => [lit(c), wall(c)]),
	...MARIA.map(
		(m) =>
			`radial-gradient(ellipse ${m.rx}px ${m.ry}px at ${m.x}% ${100 - m.up}%, rgba(11,14,20,${m.a}) 0%, rgba(11,14,20,${(m.a * 0.5).toFixed(3)}) 52%, transparent 100%)`,
	),
].join(", ");
/**
 * The paper the masthead is printed on: the bottom cap of a very large disc.
 *
 * THE SIZE HAS TO BE MEASURED, because CSS cannot do it. The diameter is a chord
 * SQUARED over a sag, and `calc()` has no way to multiply one length by another —
 * so the width is read from the window and written here, exactly as the comp does
 * it. 1440 is the fallback the comp uses server-side and is kept for the same
 * reason: it has to render to something before there is a window to ask.
 *
 * THE BOX IS MEASURED TOO, and the diameter derived from its width rather than
 * held in state. The surface needs both numbers — it is drawn on the masthead's
 * own footprint so that a percentage across or up it means what it says — and the
 * height is not knowable from the window: it comes off the spacing tokens and the
 * bar's own content. A ResizeObserver on this element is the only thing that sees
 * it. Holding the diameter instead would leave the surface with no way back to the
 * width it came from.
 */
export default function MoonLimb() {
	const hostRef = useRef<HTMLDivElement>(null);
	const [box, setBox] = useState({ w: 1440, h: 112 });

	useEffect(() => {
		const host = hostRef.current;
		if (!host) return;
		const sync = () => {
			const w = window.innerWidth;
			const h = host.offsetHeight;
			setBox((prev) => (prev.w === w && prev.h === h ? prev : { w, h }));
		};
		sync();
		const observer = new ResizeObserver(sync);
		observer.observe(host);
		return () => observer.disconnect();
	}, []);

	const d = diameter(box.w);

	return (
		<div ref={hostRef} aria-hidden className="absolute inset-0 overflow-hidden">
			<div
				className="absolute bottom-0 left-1/2 -translate-x-1/2 overflow-hidden rounded-full bg-paper"
				style={{
					width: d,
					height: d,
					// THE LIMB IS SHADED, not just cut. An inset shadow hugging the bottom
					// edge follows the circle all the way round, so the paper darkens as it
					// turns away — which is what stops a very shallow arc reading as a flat
					// band that happens to have a curved bottom.
					boxShadow: "inset 0 -26px 48px -22px rgba(9,11,16,0.32)",
				}}
			>
				<div
					className="absolute bottom-0 left-1/2 -translate-x-1/2 overflow-hidden"
					style={{ width: box.w, height: box.h }}
				>
					<div className="absolute inset-y-0 right-0 flex w-[200%] motion-safe:animate-[moon-texture-marquee_260s_linear_infinite]">
						{[0, 1].map((tile) => (
							<div
								key={tile}
								className="h-full w-1/2 flex-none"
								style={{ backgroundImage: SURFACE }}
							/>
						))}
					</div>
				</div>
			</div>
		</div>
	);
}
