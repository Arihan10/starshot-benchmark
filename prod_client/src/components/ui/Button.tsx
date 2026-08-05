import Link from "next/link";

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
 * THE FORM IS SQUARE, SLIGHTLY SOFTENED, AND SOMETIMES LEANING. It has been a
 * chamfer and it has been a pill; both were one shape applied everywhere, which is
 * why neither said anything about the control it was on. This one carries
 * information: a button's silhouette tells you whether it stands alone or belongs
 * to a group, and where in that group it sits.
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
type Variant = "solid" | "ghost" | "quiet";

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

const SHAPE: Record<Shape, string | undefined> = {
	square: undefined,
	// Both edges raked the same way: a true parallelogram, leaning right.
	standalone: `polygon(${RAKE} 0, 100% 0, calc(100% - ${RAKE}) 100%, 0 100%)`,
	// THE ROW WIDENS UPWARD. Each shape's protruding vertex is at the TOP, so the
	// group's outline is an inverted trapezoid — broad along its top edge, drawn in
	// underneath. The other way round it sat like a plinth, which is a shape that
	// wants something standing ON it; inverted it reads as a panel the page is
	// looking down at, which is what a control bar under two scenes should be.
	//
	// `start` leans left and `end` leans right, so the pair opens away from the
	// middle and the whole row is symmetric about its centre.
	start: `polygon(0 0, calc(100% - ${RAKE}) 0, 100% 100%, ${RAKE} 100%)`,
	// The inverted trapezoid: the shape left between two parallelograms leaning
	// away from each other, wider along the top than the bottom.
	middle: `polygon(0 0, 100% 0, calc(100% - ${RAKE}) 100%, ${RAKE} 100%)`,
	end: `polygon(${RAKE} 0, 100% 0, calc(100% - ${RAKE}) 100%, 0 100%)`,

	// THE CAPS: only the OUTER edge is raked, the inner one left vertical. A row of
	// cap-start, squares, cap-end has one continuous trapezoid for a silhouette —
	// the ends slope away and everything between them is a plain rectangle — which
	// is a different figure from a row of parallelograms, where every seam leans.
	// Use these when the group should read as ONE shape rather than as segments.
	"cap-start": `polygon(0 0, 100% 0, 100% 100%, ${RAKE} 100%)`,
	"cap-end": `polygon(0 0, 100% 0, calc(100% - ${RAKE}) 100%, 0 100%)`,
};

const GROUND: Record<Variant, string> = {
	// Ink ground, ground-coloured text — the one thing on the page brighter than the
	// moon.
	//
	// NO HOVER GROUND HERE, deliberately. Every solid button on the site lights up
	// by fading a copy of the mark's sweep over itself, and this used to ALSO swap
	// the ground to the accent underneath it — on a shorter clock. The result was a
	// button that jumped to flat periwinkle in 110ms and then dissolved into the
	// gradient over the next 260, which reads as the hover landing twice. The sweep
	// is the hover; a solid button that wants one supplies the layer.
	solid: "bg-mark",
	// BLACK, not a grey tint. SKIP sits between two white slabs and a grey ground
	// between them reads as a third, muddier state; pure black with a white edge
	// reads as the absence of a choice, which is what it means.
	ghost: "bg-ground group-hover/btn:bg-surface",
	// LIGHT ON DARK, and settled. This flipped twice while the moon was changing
	// size; it is fixed now because the geometry fixed it — the disc is sized to the
	// GAP between the nav's inner pairs, so by construction no nav item is ever on
	// the moon. They are always on the page's own black, where dark ink is invisible.
	quiet: "bg-transparent group-hover/btn:bg-surface",
};

// THE BORDER IS A LAYER, NOT A RING — and it has to be.
//
// `inset-ring` and `border` are both drawn against the BOX, and every one of these
// controls is clipped to a polygon that leaves the box at its raked edges. So the
// ring rendered along four sides the shape does not have, and the two slanted edges
// — the ones that make the shape recognisable — had no border at all. It looked
// like the outline had simply failed on those corners, which is exactly what had
// happened.
//
// Drawn instead as a filled layer BEHIND an inset copy of the same shape: the outer
// layer takes the border colour, the inner sits a pixel in and takes the ground, and
// what shows between them is a one-pixel edge that follows the polygon all the way
// round, slants included.
const EDGE: Record<Variant, string> = {
	solid: "bg-mark",
	ghost: "bg-mark group-hover/btn:bg-mark",
	quiet: "bg-transparent group-hover/btn:bg-mark-16",
};

const TEXT: Record<Variant, string> = {
	// THE LABEL IS THE GROUND. A solid control is the mark carrying type, so its
	// text is the page it sits on — which is why this is a token and not a colour,
	// and why the palette has three knobs and not four. Briefly white, which on
	// #ffff00 measures ~1.07:1 and simply vanished — worth measuring any pairing
	// before committing to it rather than judging by eye on one background.
	solid: "text-ground font-black",
	ghost: "text-ink font-black",
	// WHITE, NOT GREY. The comp sets every nav control at full ink and reserves
	// grey for things that are genuinely secondary; a navbar of grey labels reads
	// as disabled. Grey is now spent only where something IS lesser — a byline, a
	// lab name under a model.
	// THIS IS THE KNOB. `quiet` is the navbar's text links — ABOUT, FAQ, ARENA —
	// and their weight lives here rather than in the shared string below so it can
	// differ from the slabs without touching them. `font-medium` reads as a line of
	// words rather than three small buttons; `font-black` is the comp. One word.
	quiet: "text-ink font-black",
};

// CENTRING CORRECTION FOR THE SLANTED ENDS.
//
// A parallelogram needs none: both its edges lean the same way, so the box centre
// and the shape's centre are the same point. A CAP is a trapezoid — one edge
// upright, one raked — and its centre of area sits away from the raked side. Text
// centred in the box therefore reads as pushed toward the slant, which on ARENA and
// LEADERBOARD was plainly visible.
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
const NUDGE: Partial<Record<Shape, string>> = {
	"cap-start": "pl-[calc(var(--btn-px)+var(--rake)/2)]",
	"cap-end": "pr-[calc(var(--btn-px)+var(--rake)/2)]",
};

// WHICH CORNERS GET SOFTENED, and it is only ever the ones on the OUTSIDE of a
// row. A radius on a seam puts a notch between two buttons that are meant to read
// as one cut object — the join has to be a join. Standalone and square controls
// round all four; a group rounds its two outer ends and nothing else.
const ROUND: Record<Shape, string> = {
	square: "rounded-[3px]",
	standalone: "rounded-[3px]",
	start: "rounded-l-[3px]",
	middle: "",
	end: "rounded-r-[3px]",
	"cap-start": "rounded-l-[3px]",
	"cap-end": "rounded-r-[3px]",
};

const SIZING: Record<Variant, string> = {
	solid: "text-sm px-[var(--btn-px)] py-sm",
	ghost: "text-sm px-[var(--btn-px)] py-sm",
	// TIGHT, because the navbar's width is the moon's width. The disc has to fit the
	// gap between the two inner pairs, so every pixel of padding here is a pixel off
	// the moon's diameter — this is the narrowest the ground can arrive on hover
	// without looking cut too close to the word.
	quiet: "text-xs px-[var(--btn-px)] py-sm",
};

// The horizontal padding each rank runs at, published as a variable so NUDGE can
// add to it rather than restate it.
const PAD: Record<Variant, string> = {
	solid: "var(--spacing-md)",
	ghost: "var(--spacing-md)",
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
	...rest
}: {
	variant?: Variant;
	/**
	 * Whether to draw the hairline edge.
	 *
	 * On by default, because a control needs an outline to read as one. SKIP turns it
	 * off: it is the black cut between two white slabs, and the slabs' own edges
	 * already describe it — an outline there drew a box around the gap.
	 */
	edge?: boolean;
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
	const shell = [
				// The type is part of the control, not a choice at the call site: heavy,
				// tracked, capitals.
				// WEIGHT IS PER-VARIANT, not shared. It used to be `font-black` here for
				// every control, which is right for a slab carrying a label and wrong for
				// a text link: ABOUT, FAQ and ARENA are words in a row, not buttons, and
				// at 900 they shouted over the wordmark beside them. See TEXT.
				"font-sans tracking-[0.05em] whitespace-nowrap uppercase",
				"group/btn relative cursor-pointer border-0 bg-transparent",
				SIZING[variant],
				NUDGE[shape] ?? "",
				TEXT[variant],
				// A press moves the button, not the page. One pixel reads as travel and
				// is small enough never to disturb what is beside it.
				"transition-[color,translate] duration-quick active:translate-y-px",
				"focus-visible:outline-none",
		className,
	].join(" ");

	const layers = (
		<>
			{/* The edge, then the ground a pixel inside it. Both carry the SAME clip,
			    so the gap between them is even all the way round — including along the
			    slants, which is the whole reason this is two layers and not a ring.

			    A HAIR OF ROUNDING on each, which survives the clip only at the corners
			    the polygon actually passes through: the outer corners of a row, never
			    its interior seams. That is the right place for it — the outside of a
			    group is an edge and wants softening, the joins inside it are joins. */}
			{edge && (
				<span
					aria-hidden
					className={`absolute inset-0 transition-colors duration-quick ${ROUND[shape]} ${EDGE[variant]}`}
					style={{ clipPath: SHAPE[shape] }}
				/>
			)}
			{/* Sits a pixel inside the edge so the edge shows as a hairline round the
			    whole polygon — or fills the box outright when there is no edge to
			    leave room for. Not one layer with a border: see EDGE. */}
			<span
				aria-hidden
				className={`absolute transition-colors duration-quick ${
					edge ? "inset-[1px]" : "inset-0"
				} ${ROUND[shape]} ${GROUND[variant]}`}
				style={{ clipPath: SHAPE[shape] }}
			/>
			{sweep && (
				// A LAYER, not a background swap: `background-image` does not
				// interpolate, so a gradient set on hover would snap in while
				// everything around it eased. Fading a copy over the ground is the
				// only way the two arrive together.
				<span
					aria-hidden
					className={`absolute opacity-0 transition-opacity duration-[420ms] ease-out group-hover/btn:opacity-100 ${
						edge ? "inset-[1px]" : "inset-0"
					}`}
					style={{
						clipPath: SHAPE[shape],
						backgroundImage: "var(--accent-sweep)",
					}}
				/>
			)}
			<span className="relative">{children}</span>
		</>
	);

	const style = {
		...RAKE_VAR,
		"--btn-px": PAD[variant],
		...rest.style,
	} as React.CSSProperties;

	return href ? (
		<Link href={href} className={shell} style={style} {...(rest as object)}>
			{layers}
		</Link>
	) : (
		<button type="button" {...rest} className={shell} style={style}>
			{layers}
		</button>
	);
}
