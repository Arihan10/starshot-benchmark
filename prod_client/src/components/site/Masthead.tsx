import type { ReactNode } from "react";
import Fade from "./Fade";
import MoonAnchor from "./MoonAnchor";
import Navbar from "./Navbar";

/**
 * HOW BIG THE MOON IS, and it is set by what it must NOT cover. The disc is centred
 * and hangs off the top of the window, so its widest point on screen is the very
 * top — where the nav's inner pairs sit. Those are the ceiling, not taste.
 *
 * THE MOON IS NOT NEGOTIABLE, and it is not sized by the navbar. It was shrunk to
 * 375px to fit the gap the flex nav left, and that had it backwards: the disc is
 * the masthead's ground and the nav is laid ON it. A moon trimmed to whatever the
 * buttons leave over is a moon that changes size every time a label does.
 *
 * Exported because every caller that draws on the arc has to be handed the SAME
 * length: CurvedPrompt anchors a square box of this width the way the disc is
 * anchored, and that shared centre is the whole of the alignment.
 */
export const MOON_DIAMETER = "min(48vw, 690px)";

/**
 * THE BAND'S OWN TWO NUMBERS, and they are exported because they are not only the
 * masthead's business any more.
 *
 * `BAND_PAD` is how far below the navbar the band ends — which IS the depth of the
 * moon's cap, since the disc is anchored to the bottom of it. `MOON_LIFT` is how
 * far above that edge the disc stops.
 *
 * About measures this same band to send the moon home (see AboutStage): the pose it
 * animates to has to be the pose the masthead would have put it in, or the moon
 * lands next to where the next page draws it and the whole illusion goes. Written
 * twice, the two drifted the first time this padding was touched — so now there is
 * one copy, and the page that needs to agree with it reads it from here.
 */
export const BAND_PAD = "calc(var(--spacing-xl) * 0.42)";
export const MOON_LIFT = "var(--spacing-md)";

/**
 * The masthead: the navbar, the moon behind it, and whatever is written on the moon.
 *
 * ONE COMPONENT FOR EVERY PAGE THAT HAS ONE, because the arena and the leaderboard
 * do the same thing here — a small label and a single line struck on the disc's own
 * circle. The arena writes the prompt; the leaderboard writes the champion. Two
 * copies of this would be two sets of numbers to keep in step, and the numbers are
 * the hard part: the diameter, the lift, the band depth and the arc's span are all
 * one interlocking set.
 *
 * OVERLAID OR STACKED, and that is the one thing the two callers disagree about —
 * see `placement`. Either way it takes no pointer events except on its own
 * controls, so a drag that starts on the moon still orbits the scene underneath it.
 *
 * `label` is the small tracked line inside the cap. `children` is the arced line,
 * and it is passed in rather than derived because the two callers animate it
 * differently — the arena's prompt rolls out and settles with the round, the
 * leaderboard's champion simply is.
 */
export default function Masthead({
	label,
	placement = "overlay",
	children,
}: {
	label: string;
	/**
	 * Whether the band floats over the page or takes its own space in it.
	 *
	 * `overlay` is the arena: the canvases run the full height of the window and
	 * the moon hangs over them, so the band must not occupy any of that height.
	 *
	 * `flow` is every other page. A reading page's content starts below the band
	 * rather than under it, and an overlaid band reserves no height — so the first
	 * thing on the page would slide beneath the moon and be read through it. Which
	 * is not a variant of the same idea; it is the same band, stacked instead of
	 * floated, and everything inside it is anchored to the band rather than to the
	 * window, so nothing else has to change.
	 */
	placement?: "overlay" | "flow";
	children: ReactNode;
}) {
	return (
		<div
			className={`pointer-events-none isolate overflow-hidden [&_a]:pointer-events-auto [&_button]:pointer-events-auto ${
				placement === "overlay"
					? "absolute inset-x-0 top-0 z-30"
					: "relative z-30 flex-none"
			}`}
			style={
				{
					// BOTH FROM THE CONSTANTS ABOVE rather than written into a class, so
					// the one other place that has to agree with this band can import them.
					paddingBottom: BAND_PAD,
					// How far the disc — and the line struck on it — sit above the band's
					// bottom edge. One variable read by both, so the two cannot be lifted
					// by different amounts and pull the arc off the limb.
					"--moon-lift": MOON_LIFT,
				} as React.CSSProperties
			}
		>
			{/* NOT A MOON — a marker saying where one belongs. The moon itself lives
			    in the root layout and never unmounts, so it travels between pages
			    instead of being destroyed and rebuilt on each one. See MoonStage.

			    Same anchoring the disc used to have: bottom edge on the band's, lifted,
			    centred. CurvedPrompt hangs its square box from the same corner at the
			    same width, which is what puts the arc on the limb. */}
			<MoonAnchor
				className="absolute bottom-[var(--moon-lift,0px)] left-1/2 -translate-x-1/2"
				style={{ width: MOON_DIAMETER, height: MOON_DIAMETER }}
			/>
			<Navbar />

			{/* THE LABEL, clearly SECONDARY to the line below it — roughly a third its
			    height — or the band has two headlines and neither leads. Dark ink,
			    because it is on the lit disc.

			    OUT OF FLOW, like the arc. Stacked under the navbar it made the band
			    taller, and band height IS cap depth — every pixel of it widens the
			    disc at the top and pushes the limb into the buttons. Absolute, it sits
			    inside the cap, level with the navbar and horizontally in the gap
			    between the two nav pairs, which at that height is empty. */}
			{/* BOTH LINES FADE TOGETHER, because they are one caption on one object —
			    the label in the cap and the line struck on the arc. Wrapped rather than
			    given the animation twice so the two can never drift apart, and the
			    wrapper is static, so both stay anchored to the band exactly as they
			    were: `opacity` makes a stacking context but not a containing block. */}
			<Fade enter={700} delay={180} leave={220}>
				<div className="pointer-events-none absolute inset-x-0 top-[calc(var(--spacing-xs)*1.5)] flex justify-center px-lg">
					{/* MIRRORED FROM THE COMP: mono, a quarter-em of tracking, solid black on
					    the lit disc. The comp sets it at 13.5px inside a 680-unit viewBox
					    rendered about 475px wide — so on screen it is 13.5 x 475/680, about
					    9.5px. Copying the 13.5 literally, as this did, made it half again too
					    big and the label started competing with the prompt. The label voice is Archivo everywhere else on
					    the site; this one line keeps the monospace because it is the
					    machine asking the question, and the comp sets it that way. */}
					<span className="font-mono text-[9.5px] tracking-[0.24em] whitespace-nowrap uppercase text-ground">
						{label}
					</span>
				</div>

				{children}
			</Fade>
		</div>
	);
}

/**
 * The wrapper the arced line goes in.
 *
 * ANCHORED EXACTLY AS THE DISC IS — bottom edge on the bottom of the band,
 * horizontally centred. Two square boxes pinned to the same corner share a centre,
 * so the baseline CurvedPrompt strikes about its viewBox centre is struck about the
 * MOON's centre. Matching radii is not enough; an arc hung from the top of a band
 * while the disc hangs from the bottom drifts off the limb as it crosses the middle.
 *
 * ABSOLUTE, and that is load-bearing rather than stylistic: the SVG is square and as
 * wide as the moon, so in normal flow it contributed its full height to the band and
 * pushed the disc a thousand pixels down the page.
 */
export function MoonArc({
	className = "",
	children,
}: {
	className?: string;
	children: ReactNode;
}) {
	return (
		<div
			className={`pointer-events-none absolute bottom-[var(--moon-lift,0px)] left-1/2 -translate-x-1/2 text-ground ${className}`}
		>
			{children}
		</div>
	);
}
