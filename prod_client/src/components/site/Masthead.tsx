"use client";

import type { ReactNode } from "react";
import { useLayoutEffect, useRef, useState } from "react";
import { useMastheadShape } from "@/lib/mastheadShape";
import Fade from "./Fade";
import MoonLimb from "./MoonLimb";
import Navbar, { ON_PAPER } from "./Navbar";

// ---------------------------------------------------------------------------
// THE PROMPT IS SET TO THE COMP'S OWN SIZE: 48 units of a 1600-wide viewBox laid
// out at `min(96vw, 1560px)`. Vertical place is not a copied number — the title
// is centred between the label and the moon's foot (see the caption well below).
// ---------------------------------------------------------------------------

// One unit of the comp's viewBox, in CSS pixels: 96vw / 1600, capped where its
// width caps.
const UNIT = "min(0.06vw, 0.975px)";

// 48 units. Comes out at 41.5px against `--text-xl`'s 51.8 at 1440.
//
// BOTH VOICES RIDE IT, which is why it is named for the slot and not for the
// prompt. The leaderboard's name was on `--text-xl` — a step of the type scale,
// picked because it was the nearest one — and came out a quarter larger than the
// caption doing the same job one route away. Two titles in the same slot, at the
// same moment in the same masthead, disagreeing by 10px reads as two pages built
// by different hands. The comp states a size for this slot; that is the size.
const TITLE_SIZE = `calc(48 * ${UNIT})`;

// THE PROTRUDING MOON — a shallow segment hung from the navbar's open berth.
// Width is measured from that berth so the arch starts only after the left
// cluster and ends before the right; sag is the visible foot below the bar.
const MOON_SAG = "calc(var(--spacing-xl) * 0.62)";

// HOW FAR THE MIDDLE BUMP HANGS BELOW THE BAR. Same token as the moon's sag.
export const BELLY = MOON_SAG;

// WHERE THE LABEL SITS — the origin's own drop from the top of the masthead.
// The prompt is then centred in whatever room remains below it (see the caption
// well), so neither gap is a number we retune when the lip moves.
const LABEL_TOP = "calc(var(--spacing-2xs) + 12px)";

/**
 * WHERE THE PROMPT PIVOTS: a point far below the page, so the caption ROLLS.
 *
 * Hand this to whatever carries `prompt-roll-out` / `prompt-settle`, which rotate
 * rather than translate. The angle is meaningless without it — about its own
 * centre, 1.7deg is just a caption tilting; about a point 9,600px down, the same
 * 1.7deg is 285px of travel along an almost flat arc, which is the prompt sliding
 * across the face of the moon.
 *
 * A CONSTANT, AND DELIBERATELY NOT THE LIVE RADIUS — which is the comp's own value
 * and worth defending, because the obvious "correction" here is a real regression.
 * The disc is sized off the window, so its radius is not fixed: 7,672px at 1280 and
 * 28,072px at 2560. Pivot on the true centre and the SAME 1.7deg sweeps an arc of
 * whatever length that radius makes it — 228px on a laptop against 833px on a wide
 * monitor, the prompt hurling itself off the side of the screen on the machines
 * most likely to be showing it.
 *
 * Rotation is a fixed angle; arc length is angle TIMES radius. Only one of the two
 * can be held still across a shape that resizes, and the one that has to be held is
 * the one you can see. 9,600 is the radius at about 1440 — so the roll is exact at
 * the width it was drawn for and stays believable either side of it, which is the
 * right trade for a curve this shallow: at 2560 the prompt travels on an arc a
 * little tighter than the limb, and there is no angle at which that is visible.
 */
export const ROLL_ORIGIN = "50% -9600px";

// ---------------------------------------------------------------------------
// THE FIVE FIGURES.
//
// The bar's contents are the same row of controls in every one of them, dead
// straight, and that is the point: what is being compared is only ever what the
// PAPER does around a layout that does not move. See `mastheadShape` for the
// switch and the reasoning behind the set.
// ---------------------------------------------------------------------------

// How far the paper turns as it ends, for the one figure that ends full-bleed on a
// level edge. The limb's own shading follows its circle; a straight edge has to be
// given the same darkening by hand or the paper reads as a sheet of white cut with
// scissors.
//
// NOT ON THE ISLAND, which is the same edge and a different object. A slab floating
// clear of the page has nothing to turn away INTO — the shade there reads as what
// the house style calls a component with an altered background inside it, and the
// thing that separates a bright slab from a black page is that it is bright.
//
// NOR ON THE SPLIT'S BAR, where the edge is a lie: the paper does not end at the
// bar's foot, it carries on down the tongue. Shading it drew a band across the
// tongue's shoulders and stepped the tone at the join.
const EDGE_TURN = "inset 0 -22px 40px -24px rgba(9,11,16,0.34)";

// THE FEATHER, and it is deliberately shallower than it wants to be. The rim can
// dissolve over any distance at all; what bounds it is the prompt, which sits in
// the same paper and has to stay on stock that is still opaque. This is the depth
// the belly below is opened up by, so the fade happens entirely under the type.
const VEIL = 30;
const VEIL_ROOM = "24px";

// THE TONGUE: the arc, cut loose from the bar and given to the arena.
//
// NARROW ENOUGH TO READ AS AN OBJECT rather than as a second bar — at the full
// window it would simply be the limb again with a seam above it. The overhang
// comes down with the width for the reason given at MoonLimb: sixty pixels either
// side of a chord this short flattens the arc out of existence.
const TONGUE = {
	width: "min(58vw, 780px)",
	height: `calc(${TITLE_SIZE} + var(--text-2xs) + var(--spacing-xl) * 0.58)`,
	sag: 18,
	overhang: 16,
} as const;

function Caption({ children }: { children: ReactNode }) {
	return (
		<span className="font-mono text-2xs tracking-[0.24em] whitespace-nowrap uppercase text-ink-40">
			{children}
		</span>
	);
}

export default function Masthead({
	label,
	placement = "overlay",
	children,
}: {
	label: string;
	placement?: "overlay" | "flow";
	children: ReactNode;
}) {
	const shape = useMastheadShape();
	const shellRef = useRef<HTMLDivElement>(null);
	const berthRef = useRef<HTMLDivElement>(null);
	const [moon, setMoon] = useState({ left: 0, width: 0 });

	// The moon's chord is the navbar berth — measured, not guessed, so it stays
	// clear of every control as the lockup and offer change width.
	useLayoutEffect(() => {
		const shell = shellRef.current;
		const berth = berthRef.current;
		if (!shell || !berth) return;

		const sync = () => {
			const s = shell.getBoundingClientRect();
			const b = berth.getBoundingClientRect();
			const left = b.left - s.left;
			const width = b.width;
			setMoon((prev) =>
				prev.left === left && prev.width === width
					? prev
					: { left, width },
			);
		};

		sync();
		const observer = new ResizeObserver(sync);
		observer.observe(shell);
		observer.observe(berth);
		return () => observer.disconnect();
	}, [shape]);

	const frame = `pointer-events-none [&_a]:pointer-events-auto [&_button]:pointer-events-auto ${
		placement === "overlay"
			? "absolute inset-x-0 top-0"
			: "relative flex-none"
	}`;

	// THE SLAB FLOATS FREE OF ALL FOUR EDGES, which is the whole of this figure:
	// with no edge to meet, there is no geometry for a straight row to disagree
	// with. The prompt cannot follow it — a slab sized to the nav has no belly to
	// hang a title in — so it goes on the page's own black, where it is the only
	// bright thing below the bar.
	if (shape === "island")
		return (
			<div className={frame}>
				<div className="px-sm pt-sm">
					<div className="relative overflow-hidden rounded-[14px] bg-paper">
						<Navbar />
						<div className="absolute inset-0" style={ON_PAPER}>
							<Fade
								enter={700}
								delay={180}
								leave={220}
								className="absolute inset-0 flex items-center justify-center px-lg"
							>
								<Caption>{label}</Caption>
							</Fade>
						</div>
					</div>
				</div>

				<Fade
					enter={700}
					delay={240}
					leave={220}
					className="flex justify-center px-lg pt-sm pb-2xs"
				>
					<div className="flex min-w-0 max-w-[64vw] justify-center">
						{children}
					</div>
				</Fade>
			</div>
		);

	// THE BAR STANDS UP AND THE ARC MOVES DOWN A ROW. Navigation is a straight
	// object because it is a row of straight objects; the arena is not, and gets
	// the curve to itself — so the two never have to be the same shape. They are
	// the same paper and meet without a seam, which is what keeps it one masthead
	// rather than two stacked bands.
	if (shape === "split")
		return (
			<div className={frame}>
				<div className="relative bg-paper">
					<Navbar />
				</div>

				<div
					className="relative mx-auto"
					style={{ width: TONGUE.width, height: TONGUE.height }}
				>
					<MoonLimb
						sag={TONGUE.sag}
						chord={(width) => width + 2 * TONGUE.overhang}
					/>
					<div className="absolute inset-0" style={ON_PAPER}>
						<Fade
							enter={700}
							delay={180}
							leave={220}
							className="absolute inset-0"
						>
							{/* Label at the top of the tongue; prompt centred in what
							    remains so the gaps above and below it stay equal. */}
							<div className="absolute inset-x-0 top-2xs bottom-0 flex flex-col px-md">
								<div className="flex flex-none justify-center">
									<Caption>{label}</Caption>
								</div>
								<div className="flex min-h-0 min-w-0 flex-1 items-center justify-center">
									<div className="flex min-w-0 max-w-[52vw] justify-center">
										{children}
									</div>
								</div>
							</div>
						</Fade>
					</div>
				</div>
			</div>
		);

	const veiled = shape === "veil";

	// STRAIGHT BAR + BERTH-SIZED MOON LIP. The bump is only as wide as the open
	// track between the nav clusters, so the bar's foot stays straight under
	// every control and the arch begins only once the buttons have ended.
	return (
		<div data-masthead className={frame}>
			<div ref={shellRef} className="relative">
				<div
					className="relative z-10 bg-paper"
					style={
						shape === "flat" ? { boxShadow: EDGE_TURN } : undefined
					}
				>
					<Navbar berthRef={berthRef} />
				</div>

				{shape !== "flat" && moon.width > 0 && (
					<div
						className="relative z-10 -mt-px"
						style={{
							width: moon.width,
							height: MOON_SAG,
							marginLeft: moon.left,
						}}
					>
						<MoonLimb
							// Chord = berth width, sag = host height → a shallow
							// circular segment whose flat edge is the bar's foot.
							sag="host"
							chord={(width) => width}
							fade={veiled ? VEIL : undefined}
							// Same paper as the bar — no turn-away shade on a lip
							// this shallow, or the bump reads as a second object.
							shade={false}
						/>
					</div>
				)}

				{shape !== "flat" && (
					<div className="absolute inset-0 z-20" style={ON_PAPER}>
						<Fade
							enter={700}
							delay={180}
							leave={220}
							className="absolute inset-0"
						>
							{/* Label pinned; prompt centred in the room below it —
							    origin's layout, so the stack sits in the well rather
							    than hugging the moon's foot. */}
							<div
								className="absolute inset-x-0 flex flex-col px-lg"
								style={{
									top: LABEL_TOP,
									bottom: veiled ? VEIL_ROOM : 0,
								}}
							>
								<div className="flex flex-none justify-center">
									<Caption>{label}</Caption>
								</div>
								<div className="flex min-h-0 min-w-0 flex-1 items-center justify-center">
									<div className="flex min-w-0 max-w-[64vw] justify-center">
										{children}
									</div>
								</div>
							</div>
						</Fade>
					</div>
				)}

				{shape === "flat" && (
					<div className="relative flex justify-center px-lg pt-sm pb-2xs">
						<Fade enter={700} delay={180} leave={220}>
							<div className="flex min-w-0 max-w-[64vw] flex-col items-center gap-2xs">
								<Caption>{label}</Caption>
								{children}
							</div>
						</Fade>
					</div>
				)}
			</div>
		</div>
	);
}

const VOICE = {
	// ITALIC NEEDS A BEARING, upright does not — which is why this is per-voice and
	// not on Title itself. The block below `truncate`s, and `overflow: hidden` cuts
	// at the box edge; an italic face draws its glyphs LEANING PAST their advance
	// width, so the box the browser measures is narrower than the ink it contains
	// and the last letter loses its tail. On the arena that last letter is a closing
	// curly quote — nearly all overhang — so it was the visible casualty.
	//
	// SYMMETRIC, so the centring is untouched. The clip is only ever on the right,
	// but padding one side would walk the prompt half a bearing off centre under a
	// `justify-center` parent, and a title that sits slightly left of the moon is a
	// worse fault than the one being fixed. In `em`, because the overhang scales
	// with the type.
	// TRACKING IS THE COMP'S, not the face's default. It sets this caption at
	// 0.01em, which is small enough to look like nothing and measures ~10px across
	// a short prompt — the whole of the width our version was coming up short by
	// once the size and the baseline agreed.
	prompt: "font-serif italic tracking-[0.01em] px-[0.14em]",
	// NO SIZE HERE — it comes off TITLE_SIZE with the prompt's, below. A `text-*`
	// class alongside that inline `fontSize` would be dead weight at best and a
	// second opinion about the answer at worst.
	name: "font-sans font-black tracking-[-0.015em] uppercase",
} as const;

// SIZE IS SHARED across voices. Both titles take TITLE_SIZE so the slot is the
// same height whichever page you arrive on. The old per-face baseline lift is
// gone: the prompt is centred between the label and the moon's foot by its box
// edges, and a translateY would steal from the gap above and give it to the gap
// below.
const VOICE_STYLE: Partial<Record<keyof typeof VOICE, React.CSSProperties>> = {
	prompt: { fontSize: TITLE_SIZE },
	name: { fontSize: TITLE_SIZE },
};

export function Title({
	voice = "prompt",
	className = "",
	children,
}: {
	voice?: keyof typeof VOICE;
	className?: string;
	children: ReactNode;
}) {
	return (
		<span
			style={VOICE_STYLE[voice]}
			className={`block max-w-full truncate leading-none text-ink ${VOICE[voice]} ${className}`}
		>
			{children}
		</span>
	);
}
