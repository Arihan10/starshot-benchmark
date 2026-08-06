"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import LogoMark from "@/components/LogoMark";
import Button from "@/components/ui/Button";

// ONE DECLARATION, TWO COPIES. The hover pass is the wordmark drawn a second time
// directly over the first, so the two have to set identically — same face, size,
// tracking, leading AND BASELINE. A pass a pixel off its letters reads as a plate
// out of register.
//
// APPLIED TO THEIR WRAPPER, not to each copy, so the two INHERIT it rather than
// both naming it. Naming it twice keeps the type in step and still lets the copies
// come apart vertically, which is what happened — see the wrapper below. Inherited
// from a common ancestor there is one box, one strut and one baseline, and register
// stops being something the two copies have to agree about.
//
// TAKEN FROM THE COMP, WHOLE. `calc(var(--lockup) * 2.05)` at `0.015em` in Public
// Sans 800 is what the comp sets, and this is that verbatim — it is the lockup's
// spec rather than a value derived here, so it is not something to re-tune.
//
// BOTH NUMBERS OR NEITHER. Size and tracking set the wordmark's width TOGETHER, and
// the comp's are a pair: 2.05 is large, 0.015em is nearly solid, and the width that
// comes out is a product of the two. Take the size and leave the old 0.08em tracking
// and the word runs 6% wide of the comp at every viewport — which is how it is
// possible to copy a number faithfully and still miss.
//
// AND IT OVERHANGS ITS BYLINE, on purpose, which is the part worth flagging. This
// was sized so the two lines ended on the same vertical — 1.5842x, derived by
// measuring "SCENEBENCH" at 7.8790 em-widths against Manrope's 12.4813 for "BY
// STARSHOT LABS" and dividing. That measurement was sound and the rule it served was
// ours: at the comp's numbers the wordmark comes out around 14px wider than the
// byline and hangs past it. Flush was never what the comp drew.
//
// If the lockup is ever cut loose from the comp again, the flush relationship is
// recoverable — it is `bylineEm / wordmarkEm`, measured with a Range over the text
// nodes and NET OF THE TRAILING LETTER-SPACING, which CSS adds after the last
// character of both lines and which is 0.22em on one against a fraction of that on
// the other. Align the raw Range widths and what comes out flush is the two lines'
// trailing air.
//
// Expressed as a multiple of `--text-2xs` rather than as a size of its own, which is
// what makes it hold: both lines ride ONE clamp and the ratio cannot drift. Pinned
// to `--text-sm` instead — a different clamp with different breakpoints — they
// agreed at 1440 and were 8% out at either end of the range.
const WORDMARK_TYPE =
	"font-display font-extrabold text-[length:calc(var(--lockup)*2.05)] leading-none tracking-[0.015em] whitespace-nowrap text-mark";

// The travelling window, in mask terms: opaque at its centre, feathered to nothing
// well before either end. Everything inside it is lit, so the softness of these
// edges IS the softness of the glow — a hard-edged window would switch letters on
// and off as it passed.
const GLOW_WINDOW =
	"linear-gradient(100deg, transparent 26%, rgba(0,0,0,0.55) 40%, #000 50%, rgba(0,0,0,0.55) 60%, transparent 74%)";

// WHERE THE LIGHT WAITS, AND WHERE IT ENDS UP — and both are well outside the
// word, which is the change that lets the whole effect be a transition.
//
// THE OLD REST POSITION WAS NOT CLEAR OF THE TYPE. The mask is drawn at 260% of
// the wordmark's width, and its lit core spans 26%–74% of that — so from the
// mask's own left edge the light runs from 0.676 to 1.924 of a wordmark. At the
// `100%` this used to start from, the mask sits 1.6 wordmarks left, which puts
// that band at −0.924 to +0.324: THE LIGHT WAS ALREADY A THIRD OF THE WAY ACROSS
// THE WORD before the animation had run a frame. `opacity: 0` on the first
// keyframe is what hid it, and that is why the gleam needed an opacity envelope
// at all — it was covering for a mask that never left the type.
//
// Solved rather than nudged. The band clears the word when
// `1.6 · P/100 ≥ 1.924 + bloom/W`, and the bloom — the outermost text-shadow —
// reaches about 26px on a 155px wordmark, so P ≥ 131 leaving and P ≤ −31
// arriving. 135 and −35 take those with a little room.
//
// WHICH BUYS THE OPACITY BACK. With the light genuinely off the word at both
// ends, the mask's own feathered edge is the fade — the gleam grows as the band
// slides onto the letters and dies as it slides off, which is what a highlight
// does. The envelope was fading the whole word in and out uniformly instead.
//
// The two numbers live in the class list rather than up here, and have to: they
// are Tailwind arbitrary properties, and Tailwind reads this file as text to
// decide what to emit. A class built from a constant is a class it never sees.
// This is the working; the values are `[--gleam:135%]` and `[--gleam:-35%]`.

// The sweep is one event and takes about as long as it always did. The distance
// is longer now that it starts and finishes off the type, so the light crosses
// the letters faster than it did — the part you watch is the same length.
const GLEAM_MS = 900;
const GLEAM_EASE = "cubic-bezier(0.4,0,0.55,1)";

// SLOW, AND ON TWO DIFFERENT CLOCKS — `star-turn 9s` against `star-breathe 1.9s`,
// on the spark below. A turn every nine seconds is barely a drift rather than a
// spin, which is what a point of light catching should do and what a loading
// spinner should not; and the pulse runs on a period that does not divide into
// it, so the star never repeats the same combination of angle and size while you
// are watching it.
//
// Behind `motion-safe:`, so a reader who has asked for less gets the star without
// the movement — which is the whole of what it is for. Written into the class list
// for the same reason the gleam's two positions are: Tailwind only emits what it
// can read here as text.

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
export const ON_PAPER = {
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
 * TWO GROUPS, PUSHED APART — and nothing at all in the middle. This was a three
 * column grid with a spacer track sized to a `TITLE_BERTH` of `min(38vw, 520px)`,
 * reserving a berth in the bar for a moon that does not sit in the bar: the disc
 * is drawn behind the whole masthead and the prompt hangs off the BOTTOM of it,
 * a full row below this one. There was nothing in the middle to clear. What the
 * berth actually did was hold the two groups hundreds of pixels apart at every
 * width, which is why the offer sat so far in from the right edge.
 *
 * `space-between` leaves exactly the gap the two groups do not use, and the
 * caption rides in it — see Masthead, which centres the label on the window at
 * this height. `gap-lg` is the floor under that: at a narrow enough window the
 * groups stop separating and hold a large gap instead of touching.
 *
 * VERTICAL PADDING IS THE HEADER'S NOW, not the groups'. It sat on the groups so
 * that the middle track could stretch the bar's full height for the moon to hang
 * into; with no middle track there is nothing to stretch, and two copies of the
 * same padding is one more place for the two sides to drift apart.
 */
export default function Navbar() {
	const pathname = usePathname();

	// THE READING GROUP, AND IT IS SQUARE ALL THE WAY ALONG.
	//
	// ABOUT's leading edge stands up now. It was the group's one raked edge — a
	// `cap-start`, whose whole distinguishing feature IS that slant — so taking the
	// slant off makes it a plain rectangle like the two beside it, and the row reads
	// as three text links rather than as a cut object with one bevelled end. Worth
	// knowing this is a deliberate step away from the comp, which does rake it, by
	// the same 13px: `polygon(0 0, 100% 0, 100% 100%, 13px 100%)`. Nothing MOVES,
	// though — `clip-path` paints, it does not lay out — so the three labels stay
	// exactly where the comp puts them.
	//
	// The `side` argument went with it. It existed to choose which END of a pair was
	// raked, and the right-hand pair became two standalone buttons some time ago, so
	// it was already deciding between one live branch and one dead one; with the
	// left-hand rake gone it decides nothing at all.
	const pair = (items: { label: string; href?: string }[]) =>
		items.map((item) => {
			// THE ARENA IS ONLY ACTIVE ON EXACTLY "/". Every path starts with a slash,
			// so a `startsWith` test would light it on every page of the site.
			const active = item.href
				? item.href === "/"
					? pathname === "/"
					: pathname.startsWith(item.href)
				: false;
			const inner = (
				<span className="relative">
					{item.label}
					{
						// ONE LINE FOR THE THREE OF THEM, and it goes where the pointer is.
						//
						// Every item carries a rule now, not just the active one, and which
						// of them is drawn is decided by three rules whose ORDER OF
						// PRECEDENCE IS THEIR SPECIFICITY — never the order Tailwind
						// happened to emit them in, which is a coin-toss between two
						// variants of equal weight and the way this exact kind of rule
						// silently stops working:
						//
						//   base                                (0,1,0)  the active page's
						//                                                line, at rest
						//   group-hover/nav:                    (0,2,0)  a pointer anywhere
						//                                                in the nav takes it
						//                                                back
						//   group-hover/nav:group-hover/btn:    (0,3,0)  except on the item
						//                                                actually under it
						//
						// So at rest the current page is underlined; the moment the pointer
						// enters the nav that line retreats and the one under the pointer
						// draws in; and moving between items hands it along. Leave, and it
						// returns to the page you are on. Never two lines at once — which
						// is what "whichever button we are hovering" has to mean, and what
						// a second permanent line under the active item would break.
						//
						// IT DRAWS AND UNDRAWS FROM THE LEFT. `origin-left` pins that end
						// so the rule is written and taken back the way a line is written,
						// rather than growing out of its own middle.
						//
						// The transition names `scale`, NOT `transform` — Tailwind v4's
						// `scale-x-*` utilities write the standalone `scale` property, so
						// `transition-transform` covers nothing they do.
						<span
							aria-hidden
							className={`pointer-events-none absolute inset-x-0 -bottom-[0.45em] h-[2px] origin-left rounded-full bg-accent transition-[scale] duration-settle ease-out group-hover/nav:scale-x-0 group-hover/nav:group-hover/btn:scale-x-100 ${
								active ? "scale-x-100" : "scale-x-0"
							}`}
							// THE TRIPLET, not `var(--color-accent)`. The line itself is
							// `bg-accent`, which writes `rgb(var(--accent-rgb))` into the
							// utility and so takes the bar's deep violet; the glow read the
							// `--color-accent` ALIAS, which is declared on `:root` and
							// therefore froze at the black page's periwinkle before ON_PAPER
							// ever ran. A violet line with a pale blue halo — the one thing
							// ON_PAPER names `--accent-rgb` specifically to prevent.
							style={{ boxShadow: "0 0 10px -1px rgb(var(--accent-rgb))" }}
						/>
					}
				</span>
			);
			return item.href ? (
				<Button
					key={item.label}
					href={item.href}
					aria-current={active ? "page" : undefined}
					variant="quiet"
					shape="square"
				>
					{inner}
				</Button>
			) : (
				<Button key={item.label} variant="quiet" shape="square" onClick={() => {}}>
					{inner}
				</Button>
			);
		});

	return (
		<header
			style={ON_PAPER}
			className="relative z-20 flex items-center justify-between gap-lg px-sm pt-[2px] pb-xs"
		>
			{/* --- the mark ---------------------------------------------------- */}
			<div className="flex items-center gap-md">
				{/* THE LOCKUP GOES HOME, and it is an ANCHOR now rather than a button
				    with a handler — the same rule the Button component is written to:
				    a control that navigates has to be middle-clickable, copyable and
				    crawlable, and has to announce itself as a link. An onClick that
				    pushed the route would look identical and be none of those things.

				    IT POINTS AT THE ARENA, which is "/" — the same place the ARENA nav
				    item goes. Two routes to one page is not a duplication to fix: a
				    wordmark that goes home is a convention people arrive already
				    knowing, and the nav item is the one that says so out loud.

				    NOT MARKED `aria-current`, unlike that nav item. The wordmark is the
				    site's name wherever you are standing; announcing the masthead as
				    the current page would say it on every visit to "/" and mean nothing
				    the nav item has not already said. */}
				<Link
					href="/"
					// ONE SIZE FOR THE WHOLE LOCKUP. The byline runs at `--lockup` and the
					// wordmark at a fixed multiple of it, so scaling the pair is one number
					// and the flush relationship cannot be broken by resizing either alone.
					//
					// WHAT THAT MULTIPLE IS, and how it was measured, lives at WORDMARK_TYPE
					// — with the constant, which is the only place it stays true. It was
					// restated here too, and the copy went stale: it still quoted Anton and
					// Archivo and a ratio of 1.734 while the constant above had moved to
					// 2.4527, so the file gave three different accounts of one number and
					// two of them were wrong. A measurement is documentation of a VALUE.
					// It belongs where the value is.
					//
					// `text-[length:...]`, not `text-[...]`. A bare `var()` in an arbitrary
					// value is ambiguous — Tailwind cannot tell a length from a colour —
					// so the byline's size was being dropped entirely and it rendered at
					// the browser's default 16px against a wordmark computed from 12.96.
					// That is where the last 38px of misalignment came from.
					style={{ ["--lockup" as string]: "var(--text-2xs)" }}
					aria-label="SceneBench by Starshot Labs"
					// `text-left` went with the button: it existed only to undo the
					// centring a <button> applies to its own contents, and an anchor
					// never had it to undo.
					className="group/mark flex flex-none cursor-pointer items-center gap-2xs focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
				>
					<LogoMark className="size-[calc(var(--spacing-xl)*1.45)] flex-none" />
					<div className="flex flex-col justify-center gap-2xs">
						{/* THE TYPE IS SET ON THE WRAPPER, not on each copy, and that is
						    what actually holds the two in register. Both copies now
						    INHERIT one declaration from a common ancestor rather than
						    naming the same constant twice, which is a stronger guarantee
						    than sharing a string: the string only ever matched font, size,
						    tracking and leading, and the thing that was out is none of
						    those. It is the BASELINE.

						    The lit copy is `absolute inset-0`, so its box is the wrapper's
						    content box and its text sits at the top of that box. The
						    visible copy is in flow, so it sits on the wrapper's line box —
						    and that line box is as tall as the wrapper's STRUT, which is
						    struck from the wrapper's own font. With the type on the copies
						    the wrapper inherited the page's 16px sans, so the strut was a
						    different face at a different size from the word inside it, the
						    in-flow copy was pushed down to clear it, and the absolute copy
						    was not. The two sat a good eight pixels apart.

						    Anton hid it. Its bloom was white, soft and on black, and
						    misregistration in a blur is just a slightly bigger blur; the
						    pass is a crisp violet repaint now, and a repaint that misses
						    reads exactly like a plate out of register. Moving the type up
						    here makes the strut the wordmark's own, so the line box is the
						    word's own box and the two copies land on one baseline by
						    construction rather than by coincidence. */}
						<span className={`relative inline-block ${WORDMARK_TYPE}`}>
							<span>SCENEBENCH</span>
							{/* THE GLEAM: the same word again, in WHITE, with a soft window
							    travelling along it — so only the stretch of the name inside
							    the window is caught. Light crossing the letters, which is
							    what the comp draws and what this is copied from.

							    IT LIGHTS TO WHITE ON PURPOSE, and the letters do briefly go
							    pale against the cream. That is the effect, not a fault in
							    it: a gleam is a thing passing OVER the type, and the window
							    is narrow enough and quick enough that the word is never more
							    than partly in it — the letters it has left and the letters
							    it has not reached are both still black, so the name reads
							    throughout.

							    This ran in the accent for a while, on the reasoning that
							    white type cannot hold on a light ground. True of type, and
							    beside the point for a highlight: the reasoning silently
							    swapped the goal from "light crosses the word" to "the word
							    stays maximally legible while it does", and the second one is
							    satisfied by not animating at all. What it produced was a
							    violet wash — a colour change rather than a light.

							    NO BLEND MODE, matching the comp. `plus-lighter` was here to
							    make white letters exceed themselves on a black page and adds
							    nothing over black ink, where painting white IS the lightest
							    the pixel can go. */}
							<span
								aria-hidden
								// THE HOVER MOVES ONE NUMBER and the transition does the rest.
								// `--gleam` is the only thing the two states differ by, so
								// there is no second declaration of the mask to keep in step
								// — and no opacity to gate, because the light is genuinely
								// off the word at both ends now.
								//
								// WRITTEN OUT, NOT INTERPOLATED. Tailwind reads these files as
								// TEXT to decide what CSS to emit, so a class assembled from
								// constants is a class it never sees and never generates. The
								// two numbers are derived in GLEAM_REST / GLEAM_PAST above,
								// which is where the working is; they have to appear literally
								// here to exist at all.
								className="pointer-events-none absolute inset-0 [--gleam:135%] group-hover/mark:[--gleam:-35%]"
								style={{
									// SET HERE RATHER THAN AS A UTILITY, because WORDMARK_TYPE
									// ends in `text-mark` and a second colour class on the same
									// element would be a coin-toss: equal specificity, so the
									// winner is whichever Tailwind happened to emit last. That
									// went unnoticed while the two copies shared a colour —
									// `text-ink` and `text-mark` both resolve to the paper's ink
									// in this bar — and would have surfaced as an intermittent
									// dead animation the moment they differed, which is exactly
									// what lighting the pass requires.
									color: "#ffffff",
									// THE BLOOM, as the comp sets it. Written as literals rather
									// than as `rgb(var(--ink-rgb) / 0.9)`, which is what the comp
									// says: the comp's masthead is cream drawn with LITERAL dark
									// type on a page whose ink token is still the black page's
									// near-white, so that token reads 237 there. This bar gets
									// its cream by re-pointing the token instead, so the same
									// expression here would resolve to 17 and paint a black halo
									// round a white gleam. Same intent, and the literal is the
									// only form of it that survives the inversion.
									textShadow:
										"0 0 5px rgba(237,237,237,0.9), 0 0 13px rgba(195,205,255,0.55), 0 0 26px rgba(122,104,178,0.4)",
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
									// READ FROM THE VARIABLE THE HOVER MOVES. The custom property
									// itself is not what animates — an unregistered property has
									// no type to interpolate and simply flips — but the mask
									// position that CONSUMES it is a resolved length, and that is
									// what the transition below is on.
									maskPosition: "var(--gleam) 50%",
									WebkitMaskPosition: "var(--gleam) 50%",
									// BOTH SPELLINGS, because both are set above and a transition
									// naming only one leaves the other to jump. The prefixed
									// property is the one older WebKit is actually reading.
									transition: `mask-position ${GLEAM_MS}ms ${GLEAM_EASE}, -webkit-mask-position ${GLEAM_MS}ms ${GLEAM_EASE}`,
								}}
							>
								SCENEBENCH
							</span>
							{/* THE SPARK, where the gleam runs out: above the last letter,
							    OUTSIDE the word rather than over it, timed to land as the
							    light leaves.

							    IT STAYS FOR AS LONG AS YOU DO. This was a 620ms one-shot that
							    ended on `scale(0.15)` and `opacity: 0` — so the star arrived,
							    shrank and died while the pointer was still sitting on the
							    mark, and the lockup went dead under a hand that had not
							    moved. Now the light crosses once and leaves a star behind it,
							    turning and breathing until you go.

							    PAUSED, NOT REMOVED, when the hover ends. Dropping the
							    animations would snap the star back to zero degrees and its
							    starting size for the length of the fade — a spark that
							    straightens up as it goes out. Paused, it holds the angle and
							    the size it had got to and dims from exactly there, and picks
							    the same motion back up if you return.

							    ONLY THE ENTRANCE IS DELAYED. `delay-0` is the base and the
							    delay is applied by the hover variant, so it governs the way
							    IN and not the way out: the star waits for the light on
							    arrival and leaves the instant the pointer does, rather than
							    hanging on for 700ms after the mark is cold. */}
							<span
								aria-hidden
								className="pointer-events-none absolute -top-[0.42em] -right-[0.3em] size-xs text-accent opacity-0 transition-opacity delay-0 duration-260 ease-out [animation-play-state:paused] group-hover/mark:opacity-100 group-hover/mark:delay-700 group-hover/mark:[animation-play-state:running] motion-safe:animate-[star-turn_9s_linear_infinite,star-breathe_1.9s_ease-in-out_infinite]"
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
						{/* THE BYLINE IS TRACKED OUT UNTIL IT MEASURES THE WORDMARK, which is
						    what makes the lockup a lockup: two lines of different lengths
						    stacked is a name with a caption under it, two lines that end on
						    the same vertical is one mark.

						    TRACKING, NOT `scaleX`. Stretching the glyphs would hit the same
						    width and wreck the face doing it — stems thicken on the
						    horizontal only, round letters go oval, and at 10px that reads as
						    a rendering fault. Letter-spacing adds the width BETWEEN the
						    letters, which is what setting a byline wide has always meant.

						    0.3774em IS SOLVED AGAINST PAINTED PIXELS, and it has to be.
						    Solving it against the DOM's idea of the two widths lands 1.5px
						    long, because `getBoundingClientRect` reports ADVANCE and the eye
						    reads INK: the wordmark's advance runs about 3px past its last
						    black pixel — the H's right side bearing, plus the wordmark's own
						    0.015em trailing letter-space — so tracking the byline out to the
						    same advance visibly overshoots it. Screenshot the lockup, scan
						    for the rightmost dark pixel in each line, and tune until the two
						    agree; 0.3774 is where they do, to the quarter-pixel.

						    In `em`, so it survives the clamp — both lines ride `--lockup`, so
						    every width scales the two together and the relationship holds
						    without a second number to keep in step.

						    THE NEGATIVE MARGIN IS NOT A NUDGE. CSS puts a letter-space after
						    the LAST character too, so the box runs one full step past the
						    final S. Left in, the lockup column measures 4px wider than its own
						    ink and the whole nav — which lands on the comp to the pixel —
						    slides right by that much. Pulling exactly one letter-space back
						    off the end trims the box to the glyphs. */}
						<span className="font-label text-[length:var(--lockup)] leading-none tracking-[0.3774em] -mr-[0.3774em] whitespace-nowrap text-mark-40">
							BY STARSHOT LABS
						</span>
					</div>
				</Link>
				{/* THE GROUP IS THE NAV, not each link, because the rule is shared: an
				    item has to know that some OTHER item is being pointed at in order to
				    give the line up. `group/nav` is what makes that knowable in CSS —
				    the three links are its descendants, so each one can be styled on
				    whether the row is hot as well as on whether it is itself. */}
				<nav
					aria-label="About SceneBench"
					className="group/nav flex items-center"
				>
					{pair(NAV_LEFT)}
				</nav>
			</div>

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
			<div className="flex items-center">

				{/* TWO SLABS, SIDE BY SIDE AND NOT TOUCHING, and the rake is now only on
				    the two edges that FACE EACH OTHER. The pair used to be a pair of
				    parallelograms leaning at all four edges; the outermost two are stood
				    up, so the group reads as a bar that ENDS where it ends — flush to the
				    moon on the left and square to the window on the right. A slant on
				    either outer edge read as the bar having been cut short, and the one
				    on the right in particular left the site's most important control
				    tapering away toward the corner of the page.

				    THE INNER SLANTS ARE UNTOUCHED, and they are what the shapes are for:
				    both still rake the same way, so the space between them stays a seam
				    of constant width rather than a wedge.

				    A GAP, DELIBERATELY. Cut to interlock (pulled together by one rake)
				    the two slants land on the same line and the pair reads as ONE control
				    split by a slanted seam — which is wrong, because they are two
				    different destinations. Held a hair apart they read as a pair
				    travelling together, which is what they are.

				    THE SAME CONTROL, INVERTED. Leaderboard takes the ground with the
				    mark as its edge and its type, against the CTA's solid mark — same
				    face, same size, same weight, so the two are legible as a pair and
				    the difference between them is only which way round they are. The
				    offer stays the solid one; a board you can go and read is not an
				    offer. */}
				{/* IT FILLS IN TO BECOME ITS TWIN. At rest the two are one control
				    printed both ways round — cream with black type against black with
				    cream — and pointing at this one floods it black, so it lands on
				    exactly the colours the offer beside it is already wearing.

				    Which is the argument for it: the pair reads as one thing and its
				    inverse, and the hover says the quieter half is a destination too
				    rather than a caption on the loud one. They do not collide while it
				    happens — the offer is only ever black at REST, and the moment it is
				    the one being pointed at it is under its gradient. */}
				<Button href="/leaderboard" variant="fill" shape="upright-start">
					Leaderboard
				</Button>
				<Button
					variant="solid"
					sweep
					// THE EDGE IS BACK, and the note that removed it had the mechanism
					// right and the conclusion backwards. It is drawn in `mark` and so is
					// this button's ground, so at rest it is black on black and does
					// nothing — and the ONE thing it does is hold the sweep a pixel in
					// and leave a black frame round the gradient on hover.
					//
					// That frame is the point. Under the sweep the button is four pale
					// stops of near-white glass, and in a cream bar a pale slab with no
					// edge has nothing to end it: the control dissolves into the paper at
					// exactly the moment it is being pointed at. The border is what keeps
					// the silhouette — the rake, the uprights, the whole shape the pair
					// is cut to — legible while the fill is at its lightest.
					shape="upright-end"
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
