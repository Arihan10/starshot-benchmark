"use client";

import Link from "next/link";
import LogoMark from "@/components/LogoMark";

// ONE DECLARATION, TWO COPIES. The hover pass is the wordmark drawn a second time
// directly over the first, so the two have to set identically — same face, size,
// tracking, leading AND BASELINE. A pass a pixel off its letters reads as a plate
// out of register.
//
// APPLIED TO THEIR WRAPPER, not to each copy, so the two INHERIT it rather than
// both naming it. Naming it twice keeps the type in step and still lets the copies
// come apart vertically. Inherited from a common ancestor there is one box, one
// strut and one baseline, and register stops being something the two copies have
// to agree about.
//
// TAKEN FROM THE COMP, WHOLE. `calc(var(--lockup) * 2.05)` at `0.015em` in Public
// Sans 800 is what the comp sets, and this is that verbatim — it is the lockup's
// spec rather than a value derived here, so it is not something to re-tune.
const WORDMARK_TYPE =
	"font-display font-extrabold text-[length:calc(var(--lockup)*2.05)] leading-none tracking-[0.015em] whitespace-nowrap text-mark";

// The travelling window, in mask terms: opaque at its centre, feathered to nothing
// well before either end. Softness of these edges IS the softness of the glow.
const GLOW_WINDOW =
	"linear-gradient(100deg, transparent 26%, rgba(0,0,0,0.55) 40%, #000 50%, rgba(0,0,0,0.55) 60%, transparent 74%)";

// Rest / past mask positions — must appear literally in the class list for Tailwind.
// See working in git history; values are `[--gleam:135%]` and `[--gleam:-35%]`.
const GLEAM_MS = 900;
const GLEAM_EASE = "cubic-bezier(0.4,0,0.55,1)";

/**
 * The site lockup: mark + wordmark + byline, one home link.
 *
 * A single box the navbar insets from the bar edge — not a free-floating mark
 * that can sit outside the header's padding.
 */
export default function Lockup({ className = "" }: { className?: string }) {
	return (
		<Link
			href="/"
			// ONE SIZE FOR THE WHOLE LOCKUP. The byline runs at `--lockup` and the
			// wordmark at a fixed multiple of it, so scaling the pair is one number
			// and the flush relationship cannot be broken by resizing either alone.
			//
			// `text-[length:...]`, not `text-[...]`. A bare `var()` in an arbitrary
			// value is ambiguous — Tailwind cannot tell a length from a colour —
			// so the byline's size would be dropped and render at the browser default.
			style={{ ["--lockup" as string]: "var(--text-2xs)" }}
			aria-label="SceneBench by Starshot Labs"
			className={`group/mark flex flex-none cursor-pointer items-center gap-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent ${className}`}
		>
			{/* THE BOX IS THE DISC. logo.png draws the mark right to its own edges, so
			    the mark's layout box is its ink — nothing here has to answer for a
			    transparent border, and the bar's padding is the whole of the margin
			    the eye reads on this side. */}
			<LogoMark className="size-xl flex-none" />
			<div className="flex flex-col justify-center gap-2xs">
				{/* Type on the wrapper so both copies share one strut / baseline. */}
				<span className={`relative inline-block ${WORDMARK_TYPE}`}>
					<span>SCENEBENCH</span>
					{/* Gleam: white pass under a travelling mask window. */}
					<span
						aria-hidden
						className="pointer-events-none absolute inset-0 [--gleam:135%] group-hover/mark:[--gleam:-35%]"
						style={{
							color: "rgb(var(--gleam-rgb))",
							textShadow:
								"0 0 5px rgb(var(--gleam-rgb) / 0.9), 0 0 13px rgb(var(--accent-rgb) / 0.55), 0 0 26px rgb(var(--accent-deep-rgb) / 0.4)",
							maskImage: GLOW_WINDOW,
							WebkitMaskImage: GLOW_WINDOW,
							maskSize: "260% 420%",
							WebkitMaskSize: "260% 420%",
							maskRepeat: "no-repeat",
							WebkitMaskRepeat: "no-repeat",
							maskPosition: "var(--gleam) 50%",
							WebkitMaskPosition: "var(--gleam) 50%",
							transition: `mask-position ${GLEAM_MS}ms ${GLEAM_EASE}, -webkit-mask-position ${GLEAM_MS}ms ${GLEAM_EASE}`,
						}}
					>
						SCENEBENCH
					</span>
					{/* Spark left behind when the gleam exits. */}
					<span
						aria-hidden
						className="pointer-events-none absolute -top-[0.42em] -right-[0.3em] size-xs text-mark opacity-0 transition-opacity delay-0 duration-260 ease-out [animation-play-state:paused] group-hover/mark:opacity-100 group-hover/mark:delay-700 group-hover/mark:[animation-play-state:running] motion-safe:animate-[star-turn_9s_linear_infinite,star-breathe_1.9s_ease-in-out_infinite]"
					>
						<svg
							viewBox="0 0 24 24"
							fill="currentColor"
							className="size-full"
						>
							<path d="M12 0c0 6.6 5.4 12 12 12-6.6 0-12 5.4-12 12 0-6.6-5.4-12-12-12 6.6 0 12-5.4 12-12z" />
						</svg>
					</span>
				</span>
				{/* Tracked to the wordmark; -mr trims trailing letter-space from the box. */}
				<span className="font-label text-[length:var(--lockup)] leading-none tracking-[0.3774em] -mr-[0.3774em] whitespace-nowrap text-mark-40">
					BY STARSHOT LABS
				</span>
			</div>
		</Link>
	);
}
