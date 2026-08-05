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

// How much of the circle the line may use, each side of the bottom.
//
// THE CEILING IS THE CAP'S DEPTH, and that is the real constraint. The arc's ends
// climb R - (R-INSET)*cos(theta) above the disc's lowest point, so on a large disc
// showing only a shallow sliver, a wide span walks the ends of the line off the top
// of what is visible.
//
// AND THE ASCENDERS GO HIGHER THAN THE BASELINE, which is what caught this twice:
// the arc's ends sat inside the cap by the geometry and the TYPE on them clipped off
// the top of the window anyway. The span has to leave room for the letters, not just
// for the path — and the taller the type, the more room, so raising FONT_MAX means
// bringing this down with it. The two constants are one adjustment.
//
// It also MOVES WITH THE RADIUS, because the span is angular and the arc's length is
// therefore a fraction of the disc. The moon is now small — it has to fit the gap
// between the nav's inner pairs — so the same degrees buy far fewer pixels, and the
// span has to open up to keep the prompt a readable length. It went to 12 while the
// disc was briefly enormous and is back up now that it is not. The limit is
// not the moon — it is the BAND: the disc is clipped to the masthead, so past this
// the baseline climbs out of the visible cap and the ends of the prompt would be
// set on black rather than on the moon. It also has to stop short of the label
// above it, which is what sets this rather than the geometry.
const HALF_SPAN_DEG = 33;

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
// RAISED WHEN THE MOON GREW. The ceiling had been tuned against a much smaller arc,
// and by the time the disc reached 690px even a SEVEN-WORD prompt was pinned at it —
// measuring exactly FONT_MAX, which is the fitter reporting that the arc had room to
// spare and the cap would not let it use any. A ceiling that the longest prompts hit
// is not a ceiling on the spread, it is a cap on the type.
//
// THE CEILING IS THE SPREAD. Only short prompts should ever reach it — anything past
// about six words should fit at its own natural size below — so this number is not
// "how big is the prompt", it is "how much bigger is a two-word prompt than a
// twelve-word one". Set high, the masthead changed scale every round; the moon and
// the label around it stayed put while the one line between them swung, and the
// page looked like it was zooming rather than turning. Lowering it tightens that
// spread and leaves the long prompts, which never touched it, exactly as they were.
// AND IT IS PER VOICE, because the two lines this draws are not the same size on
// the same disc. See VOICE below: the arena's prompt keeps this ceiling, the
// leaderboard's champion takes the comp's, which is a good deal smaller.
const FONT_MAX = 54;
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
/**
 * WHO IS SPEAKING, which is the only thing the two callers disagree about.
 *
 * `prompt` is the arena, and it is THE ONE SERIF ON THE PAGE, italic — everything
 * else on this site is the product talking, and that line came from a person, so it
 * is set the way a person writes rather than the way a system reports.
 *
 * `name` is the leaderboard, and it is not a sentence at all. It is a model's
 * name — a label the benchmark produced — so it takes the interface face, bold, in
 * line with every other name the site sets. An italic serif around a champion read
 * as a quotation of something nobody said.
 */
/**
 * WHAT EACH VOICE MAY GROW TO, in viewBox units, and the champion's is NOT a
 * matter of taste — it is the comp's number, converted.
 *
 * The comp caps the champion at 42 units on a disc of radius 510, which is 42/1020
 * of its DIAMETER. Our box is the diameter, a thousand units across, so the same
 * proportion is 41. Reading the two numbers side by side is misleading: the comp's
 * SVG is a shallow 680-wide crop of a much larger circle, ours is the whole disc,
 * so 42 and 54 are not comparable until both are put over the diameter they sit on.
 *
 * Ours was arc-limited rather than cap-limited — the fitter was landing near 54 for
 * a typical champion because the arc had room and nothing stopped it — which put
 * the name about 30% larger than the design. At 41 the cap binds for short and
 * medium names, exactly as the comp's own formula does, and a very long one still
 * shrinks to fit the arc.
 */
const CEILING = { prompt: FONT_MAX, name: 41 } as const;

const VOICE = {
	prompt: {
		fontFamily: "var(--font-instrument-serif), serif",
		fontStyle: "italic",
		fontWeight: 400,
		letterSpacing: "0.01em",
	},
	name: {
		fontFamily: "var(--font-sans), sans-serif",
		fontStyle: "normal",
		// THE COMP'S OWN WEIGHT AND TRACKING: Archivo 900 at −0.015em. This was 700
		// at +0.02em on the reasoning that a line of capitals needs air — which is
		// true of capitals in a sentence and wrong here. At 900 the counters are
		// already tight and the sidebearings are generous, so positive tracking pulls
		// the word apart into separate letters; the comp closes it up instead and the
		// name reads as one mark struck on the disc. Heavier AND narrower, which
		// sounds contradictory until you set it.
		fontWeight: 900,
		letterSpacing: "-0.015em",
		// CAPITALS, and set in the STYLE rather than by upper-casing the string —
		// which keeps `aria-label` reading as the model is actually written. The
		// hidden gauge shares this object, so the fit is measured on the capitals
		// that will be drawn rather than on the mixed case that will not.
		textTransform: "uppercase",
	},
} as const;

export default function CurvedPrompt({
	text,
	diameter,
	className,
	voice = "prompt",
}: {
	text: string;
	/** The moon's diameter — the same CSS length the disc itself is given. */
	diameter: string;
	className?: string;
	voice?: keyof typeof VOICE;
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
			// The gauge is held at FONT_MAX, so this is the width the line WANTS. It
			// stays at FONT_MAX whatever the ceiling is — it is a ruler, and shrinking
			// the ruler would only cost precision.
			const needed = gauge.getComputedTextLength();
			if (!needed) return;
			const usable = path.getTotalLength() * FIT_MARGIN;
			const wanted = FONT_MAX * (usable / needed);
			// The ceiling is the voice's, the floor is shared: below FONT_MIN nothing
			// is legible whoever is speaking.
			const ceiling = CEILING[voice];
			target.style.fontSize = `${Math.max(FONT_MIN, Math.min(ceiling, wanted))}px`;
		};

		fit();
		// Measured again once the webfont lands. The first pass runs against the
		// fallback face, whose widths differ enough to leave the fitted size visibly
		// wrong — usually too small, since the fallback is wider.
		document.fonts?.ready.then(fit).catch(() => {});
		// `voice` is in here because the two faces set the same string at very
		// different widths: refitting is the whole reason the gauge exists.
	}, [text, voice]);

	const typeStyle = VOICE[voice];

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
