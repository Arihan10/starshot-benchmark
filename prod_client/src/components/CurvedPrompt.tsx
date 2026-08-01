"use client";

import { useId, useLayoutEffect, useRef } from "react";

// The prompt, set on an arc concentric with the moon behind it.
//
// SVG `textPath` rather than per-character transforms: the browser handles letter
// spacing along the curve, kerning survives, and the whole thing stays one text
// node for selection and screen readers. It is also what the Arena design uses for
// the moon's own label, so the two curves are produced the same way.
//
// EVERYTHING IS IN viewBox UNITS, and the SVG is sized in CSS. That is what makes
// it responsive without measuring the viewport: scale the box and the arc, the
// type and the bow all scale together, so the curvature stays matched to the moon
// at every width instead of drifting as the disc grows.

const VB_WIDTH = 1000;

// Where the baseline starts and ends, and how far the middle of it drops below
// them. SAGITTA is the whole curvature control — it is the depth of the arc's bow,
// and the radius it implies is r = (a² + s²) / 2s for half-span a. Tuned so that
// r, once scaled to CSS pixels, lands near the moon's own radius at the height the
// text sits at; the eye reads "concentric", not "identical", so this does not have
// to be exact and must not be recomputed per frame.
const PATH_X0 = 90;
const PATH_X1 = 910;
const BASELINE = 88;
const HALF_SPAN = (PATH_X1 - PATH_X0) / 2;

// THE CURVATURE IS AN INPUT, NOT A CONSTANT. The caller passes the radius the arc
// should sit on — measured off the moon, in the same viewBox units — and the bow
// falls out of it: s = r − √(r² − a²). Hand-tuning a sagitta instead looks right at
// exactly one viewport and one prompt length, and there is no way to tell from the
// number whether it still matches the disc.
const sagittaFor = (radius: number) =>
	radius - Math.sqrt(Math.max(0, radius * radius - HALF_SPAN * HALF_SPAN));

// THE SIZE RANGE, in viewBox units. The CSS width the caller gives the SVG divided
// by VB_WIDTH is the scale factor, so rendered px = size * (cssWidth / 1000).
//
// A short prompt gets FONT_MAX and stops there — without a ceiling, "A house"
// would inflate to fill the arc and dwarf everything around it. A long one shrinks
// to fit and stops at FONT_MIN, below which the prompt is no longer the thing you
// read first and the masthead has lost its point; past that the tail clips, which
// is the honest failure and better than type nobody can read.
const FONT_MAX = 84;
const FONT_MIN = 44;

// Leave the arc's last few percent empty at both ends. Text that runs to the very
// tip sits where the curve is steepest and reads as falling off the edge.
const FIT_MARGIN = 0.95;

// Breathing room left around the ink once the box is cropped to it.
const INK_PAD = 6;

/**
 * `width` is a CSS length (clamp() is expected) and drives everything else — the
 * height follows from the viewBox's aspect ratio, so the caller only picks one
 * number.
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
	width,
	radius,
	className,
}: {
	text: string;
	width: string;
	/** Arc radius in viewBox units — see the caller for how it comes off the moon. */
	radius: number;
	className?: string;
}) {
	const svgRef = useRef<SVGSVGElement>(null);
	const pathRef = useRef<SVGPathElement>(null);
	const textRef = useRef<SVGTextElement>(null);
	const gaugeRef = useRef<SVGTextElement>(null);
	const sagitta = sagittaFor(radius);
	// A quadratic Bézier passes through the midpoint of its control offset, so the
	// control sits at twice the sagitta to make the curve dip exactly that far.
	const arc = `M ${PATH_X0},${BASELINE} Q ${VB_WIDTH / 2},${BASELINE + sagitta * 2} ${PATH_X1},${BASELINE}`;
	// Only a starting box. The real one is cropped to the ink below, once the
	// fitted size is known.
	const initialHeight = BASELINE + sagitta + 30;
	// `useId`, not a random string: the id has to be identical on the server and
	// the client or the href breaks on hydration, and two viewers on one page must
	// not collide on it.
	const pathId = `prompt-arc-${useId()}`;

	// Written straight to the DOM rather than held in state: nothing else needs to
	// know the fitted size, and a state round-trip would repaint the whole masthead
	// to change one attribute.
	useLayoutEffect(() => {
		const svg = svgRef.current;
		const path = pathRef.current;
		const target = textRef.current;
		const gauge = gaugeRef.current;
		if (!svg || !path || !target || !gauge) return;

		const fit = () => {
			// The gauge is held at FONT_MAX, so this is the width the prompt WANTS.
			const needed = gauge.getComputedTextLength();
			if (!needed) return;
			const usable = path.getTotalLength() * FIT_MARGIN;
			const wanted = FONT_MAX * (usable / needed);
			target.style.fontSize = `${Math.max(FONT_MIN, Math.min(FONT_MAX, wanted))}px`;

			// CROP THE BOX TO THE INK. A box sized for the arc's full extent is
			// mostly empty: the ends of the curve carry no glyphs above them, and a
			// prompt that fitted below FONT_MAX leaves the reserved cap height
			// unused too. That emptiness is not free — the SVG is a flow element, so
			// every unused unit becomes real space between the label and the prompt,
			// and pushes the moon's limb (anchored to the band's bottom) further
			// down the page. Measured AFTER the size settles, so it crops what is
			// actually drawn rather than what might be.
			const ink = target.getBBox();
			if (ink.height > 0) {
				svg.setAttribute(
					"viewBox",
					`0 ${ink.y - INK_PAD} ${VB_WIDTH} ${ink.height + INK_PAD * 2}`,
				);
			}
		};

		fit();
		// Measured again once the webfont lands. The first pass runs against the
		// fallback face, whose widths differ enough to leave the fitted size visibly
		// wrong — usually too small, since the fallback is wider.
		document.fonts?.ready.then(fit).catch(() => {});
	}, [text, radius]);

	const typeStyle = {
		fontFamily: "var(--font-archivo)",
		fontWeight: 600,
		letterSpacing: "-0.02em",
	} as const;

	return (
		<svg
			ref={svgRef}
			viewBox={`0 0 ${VB_WIDTH} ${initialHeight}`}
			style={{ width, height: "auto", overflow: "visible" }}
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
