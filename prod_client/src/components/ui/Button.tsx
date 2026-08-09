"use client";

import Link from "next/link";
import { useLayoutEffect, useRef, useState } from "react";

/**
 * The button. There is one, and everything pressable is it.
 *
 * WHY A COMPONENT AND NOT A CLASS STRING: before this there were five buttons on
 * the page and no two agreed. The CTA and the vote buttons were both "off-white
 * ground, black text" at different weights, paddings and radii — the exact failure
 * that makes a page look assembled rather than designed, because the eye reads two
 * things as the same kind of control and then finds they are not quite the same
 * shape. A component makes that impossible rather than merely discouraged.
 *
 * THE FORM IS SHARP, AND SOMETIMES LEANING. It has been a chamfer and it has been
 * a pill; both were one shape applied everywhere, which is why neither said
 * anything about the control it was on. This one carries information: a button's
 * silhouette tells you whether it stands alone or belongs to a group, and where
 * in that group it sits. Corners stay square — a radius on a tessellated row puts
 * a notch at every seam.
 *
 * `shape` is that. A standalone control leans — a parallelogram, both edges raked
 * the same way, which reads as a single object with a direction. A control with
 * NEIGHBOURS instead takes the edge that lets it interlock with them: the ends
 * rake outward and the middle is a trapezoid between them, so a row of buttons
 * tessellates into one bar with slanted seams rather than sitting as three tiles
 * that happen to be adjacent.
 *
 * The rake is small. At a steep angle these stop being buttons and start being
 * chevrons, and a chevron means "forward" — which is a claim none of these
 * controls is making.
 *
 * AND THE EDGE IS LIT RATHER THAN DRAWN. Each control carries a hairline of the
 * accent and a soft bloom of it outside — so a button reads as a thing the moon is
 * catching rather than as a rectangle someone filled in.
 */
/**
 * The rank of a control: how loud it is, and what it does when pointed at.
 *
 * `outline` AND `ghost` ARE THE SAME OBJECT AT REST — the ground with the mark's
 * hairline round it and the mark's ink for type — and they are deliberately
 * identical there. What separates them is what they do when pointed at, which is
 * the whole of what a rank is.
 *
 * GHOST MOVES; OUTLINE DOES NOT. Ghost is SKIP in the arena, and the note at
 * MOTION.ghost is the argument for why its hover has to be a widening rather than
 * a colour: it sits between the two white vote slabs and anything that lightens it
 * reads as a third choice. Outline is the navbar's board — a slab whose hover
 * belongs to the row it stands in, so the control itself stays still and the
 * caller supplies the rule that draws under it.
 *
 * `outline` WAS `fill`, which flooded to solid on hover. That made a second solid
 * control appear beside the only one meant to be solid; the flood is gone and the
 * name went with it, because a variant named for a fill it no longer performs is
 * worse than no name at all.
 *
 * `solid` IS THE VOTE SLAB — white on the black page, no obligation to carry an
 * edge. `cta` IS THE OFFER (Generate, and the same shape on about/faq): filled,
 * edged, and allowed to sweep without the glass erasing its border. Sharing one
 * variant made every solid tweak to the arena silently rewrite the masthead CTA.
 */
type Variant = "solid" | "cta" | "ghost" | "outline" | "quiet";

/** `true` = full perimeter; `"y"` = top and bottom only (SKIP); `false` = none. */
type Edge = boolean | "y";

/**
 * Where this control sits, which decides its silhouette.
 *
 * `square` is the default and the only one with four right angles — for anything
 * that is neither standing alone nor part of a row.
 *
 * The three group shapes INTERLOCK, and only if the boxes overlap by the rake:
 * flush against each other they leave a parallelogram of empty space at every
 * seam, because each one's slanted edge leans away from its neighbour's. That
 * overlap is `-ml-[var(--rake)]` on every member after the first — see VoteBar.
 */
type Shape =
	| "square"
	| "standalone"
	| "upright-start"
	| "upright-end"
	| "start"
	| "middle"
	| "end"
	| "cap-start"
	| "cap-end";

// How far the top edge is displaced from the bottom, in px. The angle this works
// out to depends on the control's height, which is the point: a tall button and a
// short one in the same row get the same LEAN rather than the same angle, so their
// seams stay parallel.
/**
 * HOW FAR THE SLANTED EDGES LEAN, in pixels.
 *
 * Exported as a NUMBER because the arena's build animation has to trace these
 * edges: the beam that draws the vote bar narrows as it descends, and the rate it
 * narrows at is exactly this. A second copy of 13 over there would be a copy that
 * eventually disagrees, and the failure would be silent — the line would simply
 * stop touching the shape it is supposedly drawing.
 */
export const RAKE_PX = 13;
const RAKE = `${RAKE_PX}px`;

// Published so the correction above and any caller that needs to interlock a row
// can read the same number rather than repeating it.
const RAKE_VAR = { "--rake": RAKE } as React.CSSProperties;

/**
 * HOW LONG THE SWEEP TAKES TO ARRIVE.
 *
 * Handed to the control as a variable because TWO things are timed to it — the
 * gradient fading in, and the label crossing to the dark that suits it — and they
 * are set by different utilities in different layers. A second copy of 420 in
 * either would be a copy that eventually disagrees, and the failure is silent:
 * the label simply stops being the right colour for the ground it is on.
 */
const SWEEP_MS = "420ms";

// Edge width as a length the inset polygons can name. Axis-aligned `inset` on the
// INNER layer was wrong for every raked shape: the same polygon on a smaller box
// does not track the outer slants, so the border fattened at the tips and the
// corners refused to meet under zoom. Both layers share the button's box; only
// the clip vertices move in by `--btn-edge`.
const E = "var(--btn-edge)";

/**
 * HOW FAR A FILL IS PAINTED PAST THE CLIP THAT SHAPES IT, in px.
 *
 * A layer whose paint ENDS where its clip does is antialiased twice at that
 * boundary — once for the fill's own edge, once for the clip — and the two
 * coverages MULTIPLY. Along a flat edge both are 1 and nothing shows. At a tip
 * they are both fractional: a vertex sitting on the box corner came out at
 * roughly a third of the coverage it should have, and the point read as bitten
 * off. Painting from a box this much larger keeps the fill's own edge out of
 * the way, so the clip is the only thing drawing the silhouette.
 *
 * Any value past a pixel works; this is comfortably clear of the AA footprint
 * at every zoom and is cut away regardless.
 */
const BLEED = 8;
const BLEED_BOX = { inset: -BLEED } as const;

/**
 * THE INNER SILHOUETTE — the same figure held in by `--btn-edge`, drawn all in
 * CSS.
 *
 * A UNIFORM BAND CANNOT BE MADE BY PULLING VERTICES IN BY `(e, e)`: on a slant
 * the perpendicular is shorter than the horizontal by the rake angle, so that
 * band thins along the rake and the tips miss. The correct horizontal offset for
 * a raked vertex is `e·√(r² + h²)/h` give or take `e·r/h` — larger at the
 * PROTRUDING end of the edge, smaller at the recessed one.
 *
 * NEITHER TERM MENTIONS THE WIDTH. That is the whole reason this can be CSS: the
 * only measured quantity is the control's HEIGHT, which the Leaderboard's unfold
 * does not change, so `%` carries the width and nothing has to be rewritten as
 * the slab grows. Measuring the width instead is what made that edge shimmer —
 * the outer silhouette was `%` and the inner was pixels, and a fraction of a
 * pixel between them is a border of visibly different weights.
 *
 * `--edge-k` and `--edge-t` are those two ratios, published by useRakeGeometry.
 */
const IN_BIG = `calc(${E} * (var(--edge-k) + var(--edge-t)))`;
const IN_SMALL = `calc(${E} * (var(--edge-k) - var(--edge-t)))`;

/**
 * A vertex, named rather than measured: which of the four x positions it sits
 * at, and whether it is on the top or the bottom edge.
 *
 * x — 0: left, 1: left + rake, 2: right − rake, 3: right.
 * y — 0: top, 1: bottom.
 *
 * ONE TABLE, TWO POLYGONS. The outer figure and its inset were separate literal
 * lists and drifted apart every time a shape was touched; both are now read off
 * this.
 */
type Anchor = readonly [0 | 1 | 2 | 3, 0 | 1];

const FIGURE: Record<Shape, readonly Anchor[] | null> = {
	square: null,
	standalone: [
		[1, 0],
		[3, 0],
		[2, 1],
		[0, 1],
	],
	"upright-start": [
		[0, 0],
		[3, 0],
		[2, 1],
		[0, 1],
	],
	"upright-end": [
		[1, 0],
		[3, 0],
		[3, 1],
		[0, 1],
	],
	start: [
		[0, 0],
		[2, 0],
		[3, 1],
		[1, 1],
	],
	middle: [
		[0, 0],
		[3, 0],
		[2, 1],
		[1, 1],
	],
	end: [
		[1, 0],
		[3, 0],
		[2, 1],
		[0, 1],
	],
	"cap-start": [
		[0, 0],
		[3, 0],
		[3, 1],
		[1, 1],
	],
	"cap-end": [
		[0, 0],
		[3, 0],
		[2, 1],
		[0, 1],
	],
};

/** `polygon()` for a figure, either on the silhouette or held in by the edge. */
function figureClip(shape: Shape, inner: boolean): string | undefined {
	const figure = FIGURE[shape];
	if (!figure) return undefined;

	const points = figure.map(([ax, ay], i) => {
		const onLeft = ax < 2;
		const base = ax === 0 ? "0px" : ax === 1 ? RAKE : ax === 2 ? `calc(100% - ${RAKE})` : "100%";
		const y = ay === 0 ? "0px" : "100%";
		if (!inner) return `${base} ${y}`;

		// The other end of this vertex's SIDE edge — the neighbour at the other
		// y. Vertical side (same x) insets by the edge width; a raked one insets
		// by more at whichever end protrudes.
		const [px, py] = figure[(i + 1) % figure.length];
		const partner = py === ay ? figure[(i + figure.length - 1) % figure.length] : ([px, py] as Anchor);
		const dx =
			partner[0] === ax
				? E
				: (onLeft ? ax < partner[0] : ax > partner[0])
					? IN_BIG
					: IN_SMALL;

		const x = onLeft
			? ax === 0
				? dx
				: `calc(${base} + ${dx})`
			: `calc(${base} - ${dx})`;
		return `${x} ${ay === 0 ? E : `calc(100% - ${E})`}`;
	});

	return `polygon(${points.join(", ")})`;
}

/**
 * The two clips, read off the one table.
 *
 * `standalone` rakes both edges the same way: a true parallelogram, leaning
 * right.
 *
 * STANDALONE, BUT WITH ONE EDGE STOOD UP. The parallelogram leans at both ends,
 * which is right for a control floating in the middle of a page and wrong for
 * one that TERMINATES something — the navbar's pair ends the bar, and a slant on
 * its outermost edge reads as the bar having been cut short rather than
 * finished. The suffix names WHICH END IS UPRIGHT: `upright-end` stands its
 * right edge up, `upright-start` its left.
 *
 * THE RAKED EDGE KEEPS THE PARALLELOGRAM'S LEAN, top vertex outboard of bottom —
 * which is why these are their own entries and not the caps. A cap rakes the
 * OTHER way (bottom outboard), so `cap-start` would have stood the CTA's right
 * edge up correctly and then leaned its left edge against the Leaderboard's
 * right instead of alongside it: the gap between the two would have opened into
 * a V rather than staying the constant slanted seam it is.
 *
 * THE ROW WIDENS UPWARD. `start` / `middle` / `end` each put their protruding
 * vertex at the TOP, so the group's outline is an inverted trapezoid — broad
 * along its top edge, drawn in underneath. The other way round it sat like a
 * plinth, which is a shape that wants something standing ON it. `start` leans
 * left and `end` leans right, so the pair opens away from the middle and the row
 * is symmetric about its centre.
 *
 * THE CAPS rake only their OUTER edge and leave the inner one vertical, so a row
 * of cap-start, squares, cap-end has ONE continuous trapezoid for a silhouette
 * rather than a seam at every join.
 *
 * THE TIPS SIT ON THE BOX, and they have to. A clip-path can only take paint
 * away — pushing a vertex past `100%` does not extend the fill, it moves the cut
 * outside it, and what is left is the element's own square edge with no
 * antialiasing at all. Fills are bled past the clip instead; see BLEED.
 */
const SHAPE = Object.fromEntries(
	(Object.keys(FIGURE) as Shape[]).map((s) => [s, figureClip(s, false)]),
) as Record<Shape, string | undefined>;

const SHAPE_INNER = Object.fromEntries(
	(Object.keys(FIGURE) as Shape[]).map((s) => [s, figureClip(s, true)]),
) as Record<Shape, string | undefined>;

/**
 * Publishes the two slant ratios the inner clip is written in terms of.
 *
 * THIS IS THE WHOLE OF THE JAVASCRIPT NOW. Both silhouettes are CSS, so nothing
 * has to be rewritten as a control resizes — which is what the Leaderboard's
 * shimmering right edge was. Its slab widens as the podium mark unfolds; with
 * the outer figure in `%` and the inner one in measured pixels, the two
 * disagreed by whatever the observer was behind by, and the band between them
 * breathed. `--edge-k` and `--edge-t` depend only on the rake and the HEIGHT,
 * and the unfold does not change the height, so they are written once and stay
 * true for every frame of it.
 *
 * MEASURED OFF THE LAYER, NOT THE CONTROL. `ResizeObserver` skips non-replaced
 * inline elements, and an `<a>` — every `href` button, the Leaderboard included
 * — is one. Observing the host silently never called back. The paint host is
 * absolutely positioned, so it is blockified and reports real sizes.
 *
 * `borderBoxSize` is fractional and untransformed; `offsetWidth` rounds, and a
 * rounded height put the ratios slightly off on tall controls.
 */
function useRakeGeometry(shape: Shape, active: boolean) {
	const boxRef = useRef<HTMLSpanElement>(null);
	// THE HEIGHT, ROUNDED TO THE PIXEL — and the rounding is the whole point.
	//
	// State is only safe here because the ratios move with the HEIGHT, so a
	// width animation can observe, compare, and return without touching the
	// tree. Kept RAW that comparison is far too sensitive to hold: `--edge-t`
	// shifts by about 6e-3 per pixel of height, so a hundredth of a pixel of
	// wobble — which a hover transition on a neighbouring box is more than
	// capable of producing — already reads as a change and re-renders. The inner
	// clip then re-derives mid-hover, and because the flat edges inset by a
	// plain `--btn-edge` and only the RAKED one is written in terms of these
	// ratios, the slant is the one edge that visibly shifts. That was the
	// Leaderboard's jitter.
	//
	// Half a pixel of height moves the inset by under a thousandth of one, so
	// rounding costs nothing and makes the value stable. Re-setting the same
	// number is a no-op in React, so hovering re-renders nothing at all.
	const [height, setHeight] = useState(0);

	useLayoutEffect(() => {
		const box = boxRef.current;
		if (!active || shape === "square" || !box) return;

		const read = (h: number) => {
			if (h >= 1) setHeight(Math.round(h));
		};

		read(box.offsetHeight);
		const ro = new ResizeObserver(([entry]) => {
			const size = entry.borderBoxSize?.[0];
			read(size ? size.blockSize : box.offsetHeight);
		});
		ro.observe(box);
		return () => ro.disconnect();
	}, [active, shape]);

	// Unmeasured, the ratios describe a square corner — which makes the inner
	// clip the plain `(e, e)` pull. Wrong on a slant by a fraction of a pixel,
	// right everywhere else, and correct the moment the height is known.
	return {
		boxRef,
		k: height ? Math.hypot(RAKE_PX, height) / height : 1,
		t: height ? RAKE_PX / height : 0,
	};
}

const GROUND: Record<Variant, string> = {
	// Vote slab on the black page — the one thing brighter than the moon.
	//
	// NO HOVER GROUND HERE, deliberately. Every solid button on the site lights up
	// by fading a copy of the mark's sweep over itself, and this used to ALSO swap
	// the ground to the accent underneath it — on a shorter clock. The result was a
	// button that jumped to flat periwinkle in 110ms and then dissolved into the
	// gradient over the next 260, which reads as the hover landing twice. The sweep
	// is the hover; a solid button that wants one supplies the layer.
	solid: "bg-mark",
	// The offer (Generate). Same paint tokens as solid so ON_PAPER still inverts it
	// to a dark slab on the masthead, but a separate rank so arena tweaks cannot
	// silently rewrite it — and so its edge can stay visible under the sweep.
	cta: "bg-mark",
	// BLACK, and it STAYS black. SKIP sits between two white slabs and a grey ground
	// between them reads as a third, muddier state; pure black with a white edge
	// reads as the absence of a choice, which is what it means. It used to lift to
	// `surface` on hover, which was that muddier state arriving at exactly the
	// moment the control was being considered — see MOTION.ghost for what replaced
	// it.
	ghost: "bg-ground",
	// The same ground, and nothing happens to it. Whatever this control does on
	// hover is the row's business, not the slab's.
	outline: "bg-ground",
	// LIGHT ON DARK, and settled. This flipped twice while the moon was changing
	// size; it is fixed now because the geometry fixed it — the disc is sized to the
	// GAP between the nav's inner pairs, so by construction no nav item is ever on
	// the moon. They are always on the page's own black, where dark ink is invisible.
	// NOTHING, EVER. The navbar's ABOUT / FAQ / ARENA are words in a row, not slabs,
	// and their hover is the accent rule that draws in underneath them — see the
	// nav in Navbar, which owns that line because it is shared between the three.
	// A ground arriving behind the word as well was the hover landing twice, and
	// the tinted panel was the half that made three text links look like three
	// small buttons.
	quiet: "bg-transparent",
};

// THE BORDER IS A STROKE, NOT A RING — and for raked shapes it has to be.
//
// `inset-ring` and `border` are drawn against the BOX, and every raked control is
// clipped to a polygon that leaves the box at its slanted edges. The ring ran
// along four sides the shape does not have; the slants had no border at all.
//
// Square controls still use a ground layer inset by `--btn-edge`. Raked controls
// paint an SVG polygon with a centred stroke and clip the host to the silhouette,
// so the visible hairline is uniform on every side — see useRakeGeometry. Most
// controls keep the 1px default; SKIP matches the arena seam at `--seam-width`.
const EDGE: Record<Variant, string> = {
	solid: "bg-mark",
	cta: "bg-mark",
	ghost: "bg-mark group-hover/btn:bg-mark",
	// The hairline is the whole of this control's edge and it is never anything
	// else — the shape it draws is what the hover moves, not the colour.
	outline: "bg-mark",
	// No outline at rest and none on hover: see GROUND.quiet. A hairline appearing
	// round a text link is the same mistake as a ground appearing behind it.
	quiet: "bg-transparent",
};

// THE FACE IS PER-VARIANT, and it has to be declared here rather than shared in the
// shell below. `font-sans` sat in the shell for every control, and `quiet` — the
// navbar's ABOUT / FAQ / ARENA — needs Public Sans, which is what the comp sets
// those in. A second family utility on the same element does not override the first:
// they carry equal specificity, so which one wins is decided by the order Tailwind
// emitted them, not by which was written later. The only way to change one rank's
// face is for each rank to name its own.
const TEXT: Record<Variant, string> = {
	// THE LABEL IS THE GROUND. A solid control is the mark carrying type, so its
	// text is the page it sits on — which is why this is a token and not a colour,
	// and why the palette has three knobs and not four. Briefly white, which on
	// #ffff00 measures ~1.07:1 and simply vanished — worth measuring any pairing
	// before committing to it rather than judging by eye on one background.
	solid: "font-sans text-ground font-black",
	cta: "font-sans text-ground font-black",
	ghost: "font-sans text-ink font-black",
	// GHOST'S TYPE, TO THE LETTER, because the two are one object at rest and the
	// navbar's board has to measure against the offer beside it. It used to cross to
	// `text-ground` on hover to stay legible while the ground flooded; with no flood
	// there is nothing to cross for.
	outline: "font-sans text-ink font-black",
	// WHITE, NOT GREY. The comp sets every nav control at full ink and reserves
	// grey for things that are genuinely secondary; a navbar of grey labels reads
	// as disabled. Grey is now spent only where something IS lesser — a byline, a
	// lab name under a model.
	// THIS IS THE KNOB. `quiet` is the navbar's text links — ABOUT, FAQ, ARENA —
	// and their weight lives here rather than in the shared string below so it can
	// differ from the slabs without touching them. `font-medium` reads as a line of
	// words rather than three small buttons; `font-black` is the comp. One word.
	quiet: "font-display text-ink font-bold",
};

// CENTRING CORRECTION FOR THE SLANTED ENDS.
//
// A parallelogram needs none: both its edges lean the same way, so the box centre
// and the shape's centre are the same point. A CAP is a trapezoid — one edge
// upright, one raked — and its centre of area sits away from the raked side. Text
// centred in the box therefore reads as pushed toward the slant.
//
// Half the rake is the correction: it centres the label on the MIDPOINT of the
// slanted edge rather than on its furthest point, which is where the eye puts the
// edge anyway.
// ADDED TO the rank's padding, not substituted for it. `pl-[…]` alongside `px-*`
// REPLACES that side's padding rather than supplementing it, so it has to restate
// the base — hence `--btn-px`, which each rank sets to its own value.
//
// It was written against a literal `--spacing-md` instead, and then the quiet rank
// dropped to `px-xs` to make room for the moon: the nudge went on setting a
// medium-sized padding on one side of a small button, so About sat visibly right of
// centre and Leaderboard left of it. A correction that names its base by hand stops
// being a correction the moment the base moves.
//
// The sign matters too: a cap's raked edge cuts material AWAY from that side, so
// the label moves TOWARD the upright edge — right on cap-start, left on cap-end.
//
// THE UPRIGHTS TAKE THE SAME CORRECTION AS THE CAP THEY MIRROR, and exactly the
// same — the correction is set by where the raked edge's MIDPOINT falls, and
// mirroring which way that edge leans moves both its ends without moving the
// point halfway between them. So a shape raked on the left is nudged right by the
// same amount whichever way the rake runs.
//
// NOT ON THE CAPS, THOUGH, and that is a deliberate departure from the reasoning
// above rather than an oversight in it. The caps are the navbar's ABOUT / FAQ /
// ARENA, and the comp cuts those to the same trapezoid — `polygon(0 0, 100% 0,
// 100% 100%, 13px 100%)`, the identical 13px rake — while padding them evenly. Its
// labels therefore sit where the box centres them, not where the shape's area does.
//
// It measured as exactly half a rake. About carries the only `cap-start` in the bar,
// so its 6.5px went into its own width and pushed FAQ and ARENA along with it: all
// three sat 6.5px right of the comp, one correction showing up three times. The
// argument for the nudge is a real one and it is kept above for the shapes that
// still take it; on these two the comp is the authority and the comp does not.
const NUDGE: Partial<Record<Shape, string>> = {
	"upright-start": "pr-[calc(var(--btn-px)+var(--rake)/2)]",
	"upright-end": "pl-[calc(var(--btn-px)+var(--rake)/2)]",
};

const SIZING: Record<Variant, string> = {
	solid: "text-sm px-[var(--btn-px)] py-sm",
	cta: "text-sm px-[var(--btn-px)] py-sm",
	ghost: "text-sm px-[var(--btn-px)] py-sm",
	// The navbar pair has to measure as a pair, so this is the CTA's sizing to the
	// letter rather than a copy of ghost's that happens to match today.
	outline: "text-sm px-[var(--btn-px)] py-sm",
	// TIGHT, because the navbar's width is the moon's width. The disc has to fit the
	// gap between the two inner pairs, so every pixel of padding here is a pixel off
	// the moon's diameter — this is the narrowest the ground can arrive on hover
	// without looking cut too close to the word.
	quiet: "text-xs px-[var(--btn-px)] py-sm",
};

/**
 * WHAT A RANK DOES WHEN POINTED AT, in the cases where the answer is movement
 * rather than colour.
 *
 * ON THE SHELL, which is why these are `hover:` and not `group-hover/btn:`. The
 * shell is the element carrying `group/btn` and the group variant compiles to a
 * descendant selector — nothing is its own descendant, so a group variant here
 * would emit, read correctly in devtools and match nothing. Same trap as the
 * sweep's label rule. It also HAS to be the shell: the ground and the edge are
 * separate layers, so scaling either one alone would stretch the fill out from
 * under its own hairline.
 */
// The horizontal padding each rank runs at, published as a variable so NUDGE can
// add to it rather than restate it.
const PAD: Record<Variant, string> = {
	solid: "var(--spacing-md)",
	cta: "var(--spacing-md)",
	ghost: "var(--spacing-md)",
	outline: "var(--spacing-md)",
	quiet: "var(--spacing-xs)",
};

export default function Button({
	variant = "ghost",
	edge = true,
	shape = "square",
	sweep = false,
	href,
	className = "",
	children,
	style: styleProp,
	...rest
}: {
	variant?: Variant;
	/**
	 * How to draw the hairline edge.
	 *
	 * On by default (full perimeter), because a control needs an outline to read as
	 * one. `"y"` is top and bottom only — SKIP sits between two white slabs, so a
	 * left/right edge would draw a second seam on top of theirs; the horizontal
	 * rules are what mark it as a cut. `false` draws none.
	 */
	edge?: Edge;
	shape?: Shape;
	/**
	 * Light up in the mark's own gradient on hover.
	 *
	 * IT LIVES HERE RATHER THAN AT THE CALL SITE, and it has to. It was a `<span>`
	 * each caller dropped in as a child — which worked until the border became two
	 * layers and the children got wrapped in a positioned span of their own: the
	 * overlay then resolved against THAT box instead of the button, so the gradient
	 * appeared behind the label only, in a rectangle the width of the text. As a
	 * layer of the component it is a sibling of the ground, clipped by the same
	 * shape, and cannot be detached from the thing it is supposed to cover.
	 */
	sweep?: boolean;
	/**
	 * Render as a link to this route instead of as a button.
	 *
	 * A CONTROL THAT NAVIGATES IS AN ANCHOR, not a button with an onClick — it has
	 * to be middle-clickable, copyable, crawlable, and it has to say "link" to a
	 * screen reader. Wrapping the button in a <Link> instead would nest interactive
	 * elements, which is invalid and behaves differently in every browser.
	 *
	 * The two branches share every class and every layer below; only the tag and
	 * the handful of attributes that are specific to one of them differ.
	 */
	href?: string;
} & Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, "href">) {
	const edgeAll = edge === true;
	const edgeY = edge === "y";
	// NO PERIMETER EDGE (A/B wins), or rules only (SKIP): the SHELL is the shape.
	// Ground and clip both sit on it, so the silhouette is one paint under one
	// antialiased cover, and the hit area is the polygon rather than the box.
	// A second clipped child over the same tip is what stepped the vote corners.
	const shellFill = (edge === false || edgeY) && shape !== "square";

	const outerClip = SHAPE[shape];
	const squareEdge = edgeAll && shape === "square";
	const rakedEdge = edgeAll && shape !== "square";
	const { boxRef, k, t } = useRakeGeometry(shape, rakedEdge);
	const insetBox = { top: E, right: E, bottom: E, left: E };
	// The figure held in by one edge width. Raked controls clip their ground to
	// it; every sweep insets to it so the border stands under the glass.
	const innerClip = squareEdge ? undefined : SHAPE_INNER[shape];

	const shell = [
				// The type is part of the control, not a choice at the call site: heavy,
				// tracked, capitals.
				// WEIGHT IS PER-VARIANT, not shared. It used to be `font-black` here for
				// every control, which is right for a slab carrying a label and wrong for
				// a text link: ABOUT, FAQ and ARENA are words in a row, not buttons, and
				// at 900 they shouted over the wordmark beside them. See TEXT.
				"tracking-[0.05em] whitespace-nowrap uppercase",
				// NEVER THE SHELL'S OWN BACKGROUND on a clipped control. A
				// background paints to the border box, and the box edge is exactly
				// where the tips are — see BLEED. Shell-filled controls paint from
				// an oversized child instead.
				"group/btn relative cursor-pointer appearance-none rounded-none border-0 bg-transparent",
				SIZING[variant],
				NUDGE[shape] ?? "",
				TEXT[variant],
				// THE LABEL GOES DARK UNDER THE SWEEP, and it has to. The gradient's
				// middle is four near-white stops, so whatever the control was written
				// in, on hover it is type on pale glass. On the black pages `text-ground`
				// is already black and this changes nothing; in an INVERTED bar the
				// ground is the paper, and cream on cream is a label that vanishes at
				// exactly the moment the button is being pointed at.
				//
				// Named as the paper's ink rather than as `mark`, deliberately. This is
				// not the theme's dark — it is the dark that suits the GLASS, and the
				// glass is the same pale gradient in every context, so the colour that
				// reads on it cannot be one that inverts with the subtree.
				//
				// `hover:`, NOT `group-hover/btn:` — and the difference is the whole
				// thing. The group variant compiles to a DESCENDANT selector, roughly
				// `.group\/btn:hover *`, and this class sits on the element that carries
				// `group/btn`: nothing is its own descendant, so the rule was emitted,
				// looked correct in devtools' stylesheet, and matched nothing. The group
				// variant is for the LAYERS below, which really are children.
				sweep ? "hover:text-[rgb(var(--paper-ink-rgb))]" : "",
				// A press moves the button, not the page. One pixel reads as travel and
				// is small enough never to disturb what is beside it.
				"transition-[color,translate] active:translate-y-px",
				// TWO CLOCKS, AND NOT THE SAME ONE. The press stays quick — that is a
				// mechanical response and anything slower reads as lag — while the
				// label's colour crosses on the SWEEP's, because it is crossing to suit
				// the gradient and has to land with it. On `duration-quick` for both,
				// the label finished turning black while the glass was still arriving,
				// which put black type on a still-black button for a beat: the same
				// "hover landing twice" this component already refuses in GROUND.solid.
				//
				// The durations map onto `transition-[color,translate]` in order, and
				// they are written as ONE utility per branch rather than as an override
				// of `duration-quick` — two duration utilities on the same element have
				// equal specificity, so which one won would be decided by the order
				// Tailwind happened to emit them in.
				sweep
					? "[transition-duration:var(--sweep-ms),var(--duration-quick)]"
					: "duration-quick",
				"focus-visible:outline-none",
		className,
	].join(" ");

	const layers = (
		<>
			{/* Square: axis-aligned edge ring via an inset ground box. */}
			{squareEdge && (
				<>
					<span
						aria-hidden
						className={`absolute inset-0 transition-colors duration-quick ${EDGE[variant]}`}
					/>
					<span
						aria-hidden
						className={`absolute transition-colors duration-quick ${GROUND[variant]}`}
						style={insetBox}
					/>
				</>
			)}
			{/* RAKED: TWO CLIPS OFF ONE REFERENCE BOX. The host takes the `%`
			    silhouette and holds the edge colour, bled past it so the clip is
			    the only edge drawn (see BLEED); the ground sits inside on the
			    same figure held in by `--btn-edge`. Both are CSS against the same
			    box, so no width can arrive at one of them before the other — the
			    band between them is the border, and it cannot breathe. */}
			{rakedEdge && (
				<span
					ref={boxRef}
					aria-hidden
					className="absolute inset-0"
					style={{ clipPath: outerClip }}
				>
					<span
						className={`absolute transition-colors duration-quick ${EDGE[variant]}`}
						style={BLEED_BOX}
					/>
					<span
						className={`absolute inset-0 transition-colors duration-quick ${GROUND[variant]}`}
						style={{ clipPath: innerClip }}
					/>
				</span>
			)}
			{/* THE GROUND OF A SHELL-FILLED CONTROL, painted past the silhouette
			    on every side so the shell's clip is the only edge drawn at the
			    tips. See BLEED. */}
			{shellFill && (
				<span
					aria-hidden
					className={`absolute ${GROUND[variant]}`}
					style={BLEED_BOX}
				/>
			)}
			{/* Top/bottom only — children of a clipped shell, so the rules stop at
			    the slants and never stroke the left/right edges. Bled sideways and
			    outward for the same reason as the ground; the overhang is clipped. */}
			{edgeY && (
				<>
					<span
						aria-hidden
						className={`absolute ${EDGE[variant]}`}
						style={
							shellFill
								? {
										top: -BLEED,
										left: -BLEED,
										right: -BLEED,
										height: `calc(${E} + ${BLEED}px)`,
									}
								: { top: 0, left: 0, right: 0, height: E }
						}
					/>
					<span
						aria-hidden
						className={`absolute ${EDGE[variant]}`}
						style={
							shellFill
								? {
										bottom: -BLEED,
										left: -BLEED,
										right: -BLEED,
										height: `calc(${E} + ${BLEED}px)`,
									}
								: { bottom: 0, left: 0, right: 0, height: E }
						}
					/>
				</>
			)}
			{sweep && (
				// A LAYER, not a background swap: `background-image` does not
				// interpolate, so a gradient set on hover would snap in while
				// everything around it eased. Fading a copy over the ground is the
				// only way the two arrive together.
				//
				// Edged: held in to the inner figure so the border stands under the
				// glass. Shell-filled (A/B wins): no clip and bled past the
				// silhouette — the shell is the shape, and a glass edge that
				// stopped ON it would put a second antialiased cover on the tips.
				<span
					aria-hidden
					className={`absolute opacity-0 transition-opacity duration-(--sweep-ms) ease-out group-hover/btn:opacity-100 ${
						squareEdge || shellFill ? "" : "inset-0"
					}`}
					style={{
						backgroundImage: "var(--accent-sweep)",
						...(squareEdge ? insetBox : null),
						...(shellFill ? BLEED_BOX : { clipPath: innerClip }),
					}}
				/>
			)}
			<span className="relative">{children}</span>
		</>
	);

	// `--btn-edge` defaults here; callers (SKIP) override via `style`. `style` is
	// pulled out of `rest` above so a spread cannot clobber the merged object.
	const style = {
		...RAKE_VAR,
		"--sweep-ms": SWEEP_MS,
		"--btn-px": PAD[variant],
		"--btn-edge": "1px",
		// The slant ratios the inner figure is written in terms of, inherited by
		// every layer below. See useRakeGeometry.
		"--edge-k": k,
		"--edge-t": t,
		// A/B wins and SKIP: the shell IS the shape. One clip, one cover, and a
		// hit area that stops at the slant instead of at the box.
		...(shellFill ? { clipPath: outerClip } : null),
		...styleProp,
	} as React.CSSProperties;

	return href ? (
		<Link href={href} className={shell} {...(rest as object)} style={style}>
			{layers}
		</Link>
	) : (
		<button type="button" {...rest} className={shell} style={style}>
			{layers}
		</button>
	);
}
