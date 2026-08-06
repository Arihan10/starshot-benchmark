"use client";

import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import LogoMark from "@/components/LogoMark";
import Button from "@/components/ui/Button";

// ONE DECLARATION, TWO COPIES. The hover glow is the wordmark drawn a second time
// directly over the first, so the two have to set identically — same face, size,
// tracking and leading. Sharing the string is what guarantees it; two lists that
// merely looked alike would drift the moment either was touched, and a glow a pixel
// off its letters reads as a printing error.
//
// SIZED OFF THE BYLINE, not off the type scale. The two lines are one lockup and
// have to end on the same vertical, which is a relationship between two strings
// rather than a size either gets to pick.
//
// MEASURED, and re-measured for every change to either face — a weight change is
// enough to move it, which is how this was caught when the mark was briefly bolded.
//
// Back on Anton: it sets "SCENEBENCH" at 5.4163 em-widths against Archivo's 13.2846
// for "BY STARSHOT LABS" at the label tracking, so the wordmark rides at 13.2846 /
// 5.4163 = 2.4527x the byline for the two to come out flush. Anton is condensed and
// therefore NARROW per em, so it needs far more size to reach the same width than
// the squarish face that sat here in between, which needed 1.734.
//
// Worth noting that the measurement lands on 2.4527 and the note this replaced
// recorded 2.452 for "the old condensed face" — the same number, arrived at twice.
// A multiplier carried over from another typeface is simply a wrong number; one
// measured against the face in use survives being rediscovered.
//
// MEASURED ON THE GLYPHS, not the boxes. The byline is a flex child that stretches
// to the column, so comparing `getBoundingClientRect()` on the two spans reports
// them as equal whatever the type is actually doing — which it did, for three
// different faces. Use a Range over the text nodes.
//
// Expressed as a multiple of `--text-2xs` rather than as a size of its own, which
// is what makes it hold: both lines then ride ONE clamp and the ratio cannot drift.
// Pinned to `--text-sm` instead — a different clamp with different breakpoints —
// they agreed at 1440 and were 8% out at either end of the range.
const WORDMARK_TYPE =
	"font-display text-[length:calc(var(--lockup)*2.4527)] leading-none tracking-[0.08em] whitespace-nowrap text-mark";

// The travelling window, in mask terms: opaque at its centre, feathered to nothing
// well before either end. Everything inside it is lit, so the softness of these
// edges IS the softness of the glow — a hard-edged window would switch letters on
// and off as it passed.
const GLOW_WINDOW =
	"linear-gradient(100deg, transparent 26%, rgba(0,0,0,0.55) 40%, #000 50%, rgba(0,0,0,0.55) 60%, transparent 74%)";

// ARENA FIRST, because it is what the site is for and the one you come back to.
// The others are places you go looking for.
//
// Every item has a route now. The pattern is kept — `href` is optional and an item
// without one renders as a <button> — because that is what a nav item with nowhere
// to go should be: a <Link href="#"> would look identical, take focus, and send a
// viewer to the top of the page they are already on.
// SPLIT ACROSS THE MASTHEAD, one pair each side of the moon, because the middle is
// no longer available — the disc holds the question and the prompt and nothing else
// may sit on it.
//
// The pairs are not arbitrary. LEFT is what the site is ABOUT: reference material, a
// reader going looking. RIGHT is what the site DOES: the arena you are in, the board
// it feeds, and the offer to build your own — three things in one direction of
// travel, ending on the only solid button up here.
// THE BAR RUNS INVERTED, and it does it by flipping the three colour roles rather
// than by recolouring anything. Every control up here is already written in terms
// of `ground`, `ink` and `mark`; swap what those RESOLVE to for the subtree and
// each variant re-derives on its own — solid comes out black-on-white, ghost keeps
// its hairline but in ink, quiet's labels darken. Recolouring the buttons one at a
// time would have been the same change written five times, and the fifth would be
// the one that got missed.
//
// `--color-surface` and `--color-accent` are literals in the theme rather than
// built from the rgb triplets, so they have to be named here too: quiet's hover
// ground and the active underline would otherwise stay tuned for a black bar.
const ON_PAPER = {
	"--ground-rgb": "var(--paper-rgb)",
	"--ink-rgb": "var(--paper-ink-rgb)",
	"--mark-rgb": "var(--paper-ink-rgb)",
	"--surface-rgb": "var(--paper-surface-rgb)",
	"--accent-rgb": "var(--accent-deep-rgb)",
} as React.CSSProperties;

const NAV_LEFT: { label: string; href?: string }[] = [
	{ label: "About", href: "/about" },
	{ label: "FAQ", href: "/faq" },
	// ARENA JOINS THE READING SIDE. The split was "what the site is ABOUT" on the
	// left and "what it DOES" on the right; the right is now a pair of buttons, and
	// a plain text link standing next to two solid controls reads as something that
	// failed to become one. Arena is also the page you come BACK to, which makes it
	// a way of navigating rather than an offer.
	{ label: "Arena", href: "/" },
];
// The right is no longer a list at all: it is two controls, cut to interlock —
// see the pair at the foot of this file.

/**
 * The site's navbar.
 *
 * THREE COLUMNS, AND THE MOON IS THE MIDDLE ONE — passed in rather than laid over.
 * The two `1fr` tracks are equal, which is what keeps the disc centred on the
 * WINDOW while the groups either side of it differ in width, and the caller sizes
 * the middle track to the moon's own berth so the clearance is reserved rather
 * than hoped for.
 *
 * The groups are pinned with `col-start-1` and `col-start-3`. Auto-placement would
 * put the right-hand group in the middle track on any page that renders the bar
 * without a moon — About does — and it would collapse toward the lockup.
 *
 * VERTICAL PADDING SITS ON THE GROUPS, not on the header, so the middle track can
 * stretch the bar's full height and the moon can hang to its very bottom edge.
 */
export default function Navbar({ moon }: { moon?: ReactNode }) {
	const pathname = usePathname();

	// EACH PAIR IS ITS OWN GROUP, so each gets its own cap-start and cap-end and
	// reads as one trapezoid — two of them now rather than one four-wide. Written
	// once and called twice: the two sides differ only in what is in them, and a
	// second copy of this would be a second place for the shapes to drift.
	const pair = (items: { label: string; href?: string }[], side: "left" | "right") =>
		items.map((item, i) => {
			// THE ARENA IS ONLY ACTIVE ON EXACTLY "/". Every path starts with a slash,
			// so a `startsWith` test would light it on every page of the site.
			const active = item.href
				? item.href === "/"
					? pathname === "/"
					: pathname.startsWith(item.href)
				: false;
			// ONLY THE OUTER EDGE OF A PAIR IS RAKED. The moon-facing side stays
			// vertical — FAQ's right edge and Arena's left — because those two edges
			// front the disc, and a slant leaning away from a circle reads as a
			// mis-cut rather than as a shape. Each pair now slopes outward, toward the
			// edges of the window, and presents a clean face to the moon.
			//
			// LEFT PAIR: first item raked, rest square. RIGHT PAIR: last item raked,
			// rest square. Which end that is depends on the side, so it is passed in.
			const shape =
				side === "left"
					? i === 0
						? "cap-start"
						: "square"
					: i === items.length - 1
						? "cap-end"
						: "square";
			const inner = (
				<span className="relative">
					{item.label}
					{active && (
						// IT UNDRAWS ON HOVER, right to left: `origin-left` with
						// `scale-x-0` keeps the left end pinned while the right retreats
						// along it, so the line is taken back the way it was written.
						//
						// The transition names `scale`, NOT `transform` — Tailwind v4's
						// `scale-x-*` utilities write the standalone `scale` property, so
						// `transition-transform` covers nothing they do.
						<span
							aria-hidden
							className="pointer-events-none absolute inset-x-0 -bottom-[0.45em] h-[2px] origin-left scale-x-100 rounded-full bg-accent transition-[scale] duration-settle ease-out group-hover/btn:scale-x-0"
							style={{ boxShadow: "0 0 10px -1px var(--color-accent)" }}
						/>
					)}
				</span>
			);
			return item.href ? (
				<Button
					key={item.label}
					href={item.href}
					aria-current={active ? "page" : undefined}
					variant="quiet"
					shape={shape}
				>
					{inner}
				</Button>
			) : (
				<Button key={item.label} variant="quiet" shape={shape} onClick={() => {}}>
					{inner}
				</Button>
			);
		});

	return (
		<header
			style={ON_PAPER}
			className="relative z-20 grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] px-lg"
		>
			{/* --- the mark ---------------------------------------------------- */}
			<div className="col-start-1 flex items-center gap-md pt-2xs pb-xs">
				<button
					type="button"
					// ONE SIZE FOR THE WHOLE LOCKUP. The byline runs at `--lockup` and the
					// wordmark at 1.734x it, so scaling the pair is one number and the
					// flush relationship cannot be broken by resizing either alone.
					//
					// THE MULTIPLIER IS A PROPERTY OF THE TWO TYPEFACES, not of any
					// screen: Anton sets SCENEBENCH at 5.4169 em-widths and Archivo
					// sets BY STARSHOT LABS at 13.29 with the label voice's tracking, and
					// 13.29/7.663 is this number. Measured at 1280, 1440 and 1920 it comes
					// out 1.7338, 1.7332 and 1.7348 — the same value, because em-widths do
					// not care about viewport. So one constant holds at every resolution,
					// and the only thing that can invalidate it is CHANGING A FACE. It was
					// 1.6398 for Space Grotesk; swapping to Archivo, which sets that string
					// wider, is what left the wordmark short.
					//
					// `text-[length:...]`, not `text-[...]`. A bare `var()` in an arbitrary
					// value is ambiguous — Tailwind cannot tell a length from a colour —
					// so the byline's size was being dropped entirely and it rendered at
					// the browser's default 16px against a wordmark computed from 12.96.
					// That is where the last 38px of misalignment came from.
					style={{ ["--lockup" as string]: "var(--text-2xs)" }}
					// #TODO: no destination yet. Becomes a <Link> to "/" once there is
					// anywhere else to be.
					onClick={() => {}}
					aria-label="SceneBench by Starshot Labs"
					className="group/mark flex flex-none cursor-pointer items-center gap-2xs text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
				>
					<LogoMark className="size-[calc(var(--spacing-xl)*1.45)] flex-none" />
					<div className="flex flex-col justify-center gap-2xs">
						<span className="relative inline-block">
							<span className={WORDMARK_TYPE}>SCENEBENCH</span>
							{/* THE GLOW: the same word again, in white, with a bloom around
							    the letters and a soft window travelling along it — so only
							    the stretch of the name inside the window is lit. Nothing is
							    laid OVER the type: the letters are the light, and the word
							    stays exactly as readable during the pass as before it.
							    `plus-lighter` adds the light to what is already there
							    rather than replacing it, which is the difference between a
							    letter glowing and a letter being repainted. */}
							<span
								aria-hidden
								className={`pointer-events-none absolute inset-0 text-ink opacity-0 mix-blend-plus-lighter group-hover/mark:animate-[wordmark-glow_820ms_cubic-bezier(0.4,0,0.55,1)_60ms] ${WORDMARK_TYPE}`}
								style={{
									textShadow:
										"0 0 5px rgb(var(--ink-rgb) / 0.9), 0 0 13px rgba(195,205,255,0.55), 0 0 26px rgba(122,104,178,0.4)",
									maskImage: GLOW_WINDOW,
									WebkitMaskImage: GLOW_WINDOW,
									// TALLER THAN THE TEXT, on purpose. The bloom spills well
									// above and below the letters, and a mask only the height
									// of the box cut it off flat at both — which drew a lit
									// RECTANGLE around the word instead of a halo on it.
									maskSize: "260% 420%",
									WebkitMaskSize: "260% 420%",
									maskRepeat: "no-repeat",
									WebkitMaskRepeat: "no-repeat",
									maskPosition: "100% 50%",
									WebkitMaskPosition: "100% 50%",
								}}
							>
								SCENEBENCH
							</span>
							{/* THE SPARK, where the gleam runs out: above the last letter,
							    OUTSIDE the word rather than over it, timed to land as the
							    light leaves. It was lost when the lockup moved out of the
							    page and into this component — the wordmark and its glow
							    came across and this did not, so the animation ran and
							    simply ended on nothing. */}
							<span
								aria-hidden
								className="pointer-events-none absolute -top-[0.42em] -right-[0.3em] size-xs text-accent opacity-0 group-hover/mark:animate-[star-twinkle_620ms_cubic-bezier(0.3,1.4,0.4,1)_600ms]"
							>
								<svg viewBox="0 0 24 24" fill="currentColor" className="size-full">
									{/* A four-point spark with concave sides — the shape a
									    point of light makes, not the five-point badge on a
									    sheriff. */}
									<path d="M12 0c0 6.6 5.4 12 12 12-6.6 0-12 5.4-12 12 0-6.6-5.4-12-12-12 6.6 0 12-5.4 12-12z" />
								</svg>
							</span>
						</span>
						{/* NEVER WRAPS, and that was the whole bug. Without this the byline
						    broke over two lines inside a shrinkable flex column — which not
						    only looked wrong but made every measurement of it a lie: the ink
						    width came back as the WIDEST LINE, not the string, so the flush
						    ratio was computed against a number that changed with the wrap
						    point. Adding tracking made it measure NARROWER, because more
						    tracking moved the break earlier. Three tuning passes chased that
						    ghost before the screenshot showed the two lines. */}
						<span className="font-label text-[length:var(--lockup)] leading-none whitespace-nowrap text-mark-40">
							BY STARSHOT LABS
						</span>
					</div>
				</button>
				<nav aria-label="About SceneBench" className="flex items-center">
					{pair(NAV_LEFT, "left")}
				</nav>
			</div>

			{moon}

{/* --- the offer ------------------------------------------------------
			    The one solid button up here, and the only one on the page outside the
			    vote itself.

			    ITS HOVER IS THE MARK'S OWN GLASS. The site is black, white and one
			    accent, and the accent is the logo's — so the control the page most
			    wants pressed lights up in the same material as the name in the
			    opposite corner. It rides OVER the button's own hover rather than
			    replacing it, because `background-image` does not interpolate: a
			    gradient set on hover would snap in while everything else eased.
			    Fading a copy over the ink is the only way the two arrive together. */}
			{/* FLUSH AGAINST THE OFFER, and that is the moon's doing. The disc is
			    centred on the WINDOW and 690px across, so at navbar height its limb
			    reaches roughly 240px either side of centre — and the nav pair, sitting
			    a medium gap to the left of the button, was inside that. Nothing here
			    can move further right than the button itself, so the gap between them
			    is the only room there is, and it has all been given up.

			    THE SEAM IS A JOIN, not a collision. The pair's last cap rakes its
			    bottom-right corner and the offer's parallelogram rakes its top-left by
			    the same amount and in the same direction, so pulling the button back
			    by one rake makes the two slants coincide exactly — see Button, where
			    that overlap is what the group shapes are cut for. The three controls
			    read as one bar with slanted seams rather than as a pair that has
			    drifted into a button. */}
			<div className="col-start-3 flex items-center justify-end pt-2xs pb-xs">

				{/* STILL A PARALLELOGRAM, raked right — the silhouette reserved for the
				    one thing being offered. It has neighbours now, but it is the only
				    control up here leaning at BOTH edges, so the eye still finds it
				    without the button having to be any louder than it already is. */}
				{/* TWO PARALLELOGRAMS, SIDE BY SIDE AND NOT TOUCHING. Both are
				    `standalone`, so each leans at BOTH edges — top-right and
				    bottom-left corners pushed out, the same silhouette the CTA has
				    always had.

				    A GAP, DELIBERATELY. Cut to interlock (cap-start/cap-end, pulled
				    together by one rake) the two slants land on the same line and the
				    pair reads as ONE control split by a slanted seam — which is wrong,
				    because they are two different destinations. Held a hair apart they
				    read as a pair travelling together, which is what they are.

				    THE SAME CONTROL, INVERTED. Leaderboard takes the ground with the
				    mark as its edge and its type, against the CTA's solid mark — same
				    face, same size, same weight, so the two are legible as a pair and
				    the difference between them is only which way round they are. The
				    offer stays the solid one; a board you can go and read is not an
				    offer. */}
				<Button href="/leaderboard" variant="ghost" shape="standalone">
					Leaderboard
				</Button>
				<Button
					variant="solid"
					sweep
					shape="standalone"
					// Close, not joined. One rake of overlap would make the slants
					// coincide and fuse the two into a single silhouette.
					className="ml-2xs"
					// #TODO: no action yet. This should take a prompt from the visitor
					// and queue a build on both models.
					onClick={() => {}}
				>
					Build one yourself
				</Button>
			</div>
		</header>
	);
}
