"use client";

import { useId, useLayoutEffect, useRef } from "react";

// The prompt, set on a circle CONCENTRIC WITH THE MOON.
//
// SVG `textPath` rather than per-character transforms: the browser handles letter
// spacing along the curve, kerning survives, and the whole thing stays one text
// node for selection and screen readers.
//
// THE VIEWBOX IS THE MOON. It is square, 1000 units across, and the caller sizes
// the SVG to the disc's own diameter and anchors it the same way the disc is
// anchored — so viewBox (500,500) IS the moon's centre and 500 units IS its
// radius, at every viewport, with no measuring. The baseline is then simply a
// circle of radius 500 − INSET about that point: not approximately concentric,
// not concentric at one width, but the same circle struck from the same centre.
//
// This replaces a quadratic Bézier whose radius was tuned to match the disc's.
// Matching the RADIUS is not enough — the arc hung from the top of the masthead
// while the disc hung from the bottom, so the two curves were struck from
// different centres and the line drifted away from the limb as it crossed the
// middle. Anchoring both to the same box is what makes the two curves the same
// curve.
const VB = 1000;
const CENTRE = VB / 2;

// How far inside the limb the baseline sits, in viewBox units — i.e. as a
// fraction of the moon, so it holds at every size.
//
// SMALL ON PURPOSE. Every unit of inset lifts the line off the bottom of the disc
// (a concentric circle's lowest point is exactly INSET above the disc's lowest
// point), and the masthead band is only as tall as the navbar — so a generous
// inset walks the prompt straight up into the label above it. This is 2% of the
// diameter, and the floor is set by the DESCENDERS, not the baseline: at FONT_MAX
// they hang roughly 8 units below it, which still leaves a clear dozen units of
// moon under the tail of a "y" before the limb runs out.
const INSET = 20;

// How much of the circle the line may use, each side of the bottom. The limit is
// not the moon — it is the BAND: the disc is clipped to the masthead, so past this
// the baseline climbs out of the visible cap and the ends of the prompt would be
// set on black rather than on the moon. It also has to stop short of the label
// above it, which is what sets this rather than the geometry.
const HALF_SPAN_DEG = 24;

// The size range, in viewBox units — so, as thousandths of the moon's diameter.
//
// LENGTH IS WHAT SIZES THE PROMPT, not the moon. The arc is a fixed run of a fixed
// circle, so a longer prompt has exactly one way to fit on it: set smaller. That
// is the whole of the rule — the disc never grows to accommodate a sentence, and
// the masthead stays the same object whatever anyone types.
//
// A short prompt gets FONT_MAX and stops there; without a ceiling "A house" would
// inflate to fill the arc and dwarf everything around it. A long one shrinks to
// FONT_MIN and no further, below which the prompt is no longer the thing you read
// first; past that the tail clips, which is the honest failure and better than
// type nobody can read.
//
// THE CEILING IS THE SPREAD. Only short prompts ever reach it — anything past
// about six words fits at its own natural size well below — so this number is not
// "how big is the prompt", it is "how much bigger is a two-word prompt than a
// twelve-word one". Set high, the masthead changed scale every round; the moon and
// the label around it stayed put while the one line between them swung, and the
// page looked like it was zooming rather than turning. Lowering it tightens that
// spread and leaves the long prompts, which never touched it, exactly as they were.
const FONT_MAX = 38;
const FONT_MIN = 17;

// Leave the arc's last few percent empty at both ends. Text run to the very tip
// sits where the curve is steepest and reads as falling off the edge.
const FIT_MARGIN = 0.94;

const rad = (deg: number) => (deg * Math.PI) / 180;

/** A point on the baseline circle, `deg` away from the bottom of the moon. */
function at(radius: number, deg: number): [number, number] {
	const a = rad(deg);
	return [CENTRE + radius * Math.sin(a), CENTRE + radius * Math.cos(a)];
}

/**
 * `diameter` is a CSS length and must be the SAME expression the moon is sized
 * with; the caller is responsible for anchoring this box exactly as the disc is
 * anchored (bottom edge on the bottom of the masthead, horizontally centred).
 * Everything else follows from that.
 *
 * THE TYPE FITS ITSELF TO THE ARC. A path has a finite length and `textPath`
 * simply stops drawing at the end of it, so a prompt past a certain length would
 * silently lose its tail — and prompts are user-supplied, so "long enough" is not
 * a case that can be designed away. The measurement is done on a hidden copy held
 * at FONT_MAX, never on the visible text: measuring the visible node would feed
 * its own shrunken width back into the next calculation and ratchet the size down
 * every time the prompt changed.
 */
export default function CurvedPrompt({
	text,
	diameter,
	className,
}: {
	text: string;
	/** The moon's diameter — the same CSS length the disc itself is given. */
	diameter: string;
	className?: string;
}) {
	const pathRef = useRef<SVGPathElement>(null);
	const textRef = useRef<SVGTextElement>(null);
	const gaugeRef = useRef<SVGTextElement>(null);

	const radius = CENTRE - INSET;
	const [x0, y0] = at(radius, -HALF_SPAN_DEG);
	const [x1, y1] = at(radius, HALF_SPAN_DEG);
	// A TRUE CIRCULAR ARC, not a Bézier approximating one. `sweep-flag` 0 runs it
	// left to right along the BOTTOM of the circle, which is the direction the text
	// reads and the side of the moon that is on screen.
	const arc = `M ${x0.toFixed(2)},${y0.toFixed(2)} A ${radius},${radius} 0 0,0 ${x1.toFixed(2)},${y1.toFixed(2)}`;

	// `useId`, not a random string: the id has to be identical on the server and
	// the client or the href breaks on hydration, and two viewers on one page must
	// not collide on it.
	const pathId = `prompt-arc-${useId()}`;

	// Written straight to the DOM rather than held in state: nothing else needs to
	// know the fitted size, and a state round-trip would repaint the whole masthead
	// to change one attribute.
	useLayoutEffect(() => {
		const path = pathRef.current;
		const target = textRef.current;
		const gauge = gaugeRef.current;
		if (!path || !target || !gauge) return;

		const fit = () => {
			// The gauge is held at FONT_MAX, so this is the width the prompt WANTS.
			const needed = gauge.getComputedTextLength();
			if (!needed) return;
			const usable = path.getTotalLength() * FIT_MARGIN;
			const wanted = FONT_MAX * (usable / needed);
			target.style.fontSize = `${Math.max(FONT_MIN, Math.min(FONT_MAX, wanted))}px`;
		};

		fit();
		// Measured again once the webfont lands. The first pass runs against the
		// fallback face, whose widths differ enough to leave the fitted size visibly
		// wrong — usually too small, since the fallback is wider.
		document.fonts?.ready.then(fit).catch(() => {});
	}, [text]);

	// THE ONE SERIF ON THE PAGE, and italic. Everything else here is the product
	// talking — grotesque for the interface, monospace for its labels — and this is
	// the only line that came from a person, so it is set the way a person writes
	// rather than the way a system reports.
	const typeStyle = {
		fontFamily: "var(--font-instrument-serif), serif",
		fontStyle: "italic",
		fontWeight: 400,
		letterSpacing: "0.01em",
	} as const;

	return (
		<svg
			viewBox={`0 0 ${VB} ${VB}`}
			// Square, and sized to the moon. The box is mostly empty — only the strip
			// near the bottom of the circle ever carries ink — but that is what buys
			// the exact registration with the disc.
			style={{ width: diameter, height: diameter, overflow: "visible" }}
			className={className}
			role="img"
			aria-label={text}
		>
			<defs>
				<path ref={pathRef} id={pathId} d={arc} fill="none" />
			</defs>
			{/* The gauge: same string, same face, always at FONT_MAX, never drawn. */}
			<text
				ref={gaugeRef}
				fontSize={FONT_MAX}
				style={typeStyle}
				visibility="hidden"
				aria-hidden
			>
				{text}
			</text>
			<text ref={textRef} fontSize={FONT_MAX} fill="currentColor" style={typeStyle}>
				<textPath href={`#${pathId}`} startOffset="50%" textAnchor="middle">
					{text}
				</textPath>
			</text>
		</svg>
	);
}
