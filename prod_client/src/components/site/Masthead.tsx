"use client";

import type { ReactNode } from "react";
import { useLayoutEffect, useRef, useState } from "react";
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

// THE PROTRUDING MOON — a shallow segment centred on the bar. Half-width is the
// tighter clearance from midline to either cluster (inset a quarter toward the
// centre), so +x and −x match and the longer side simply sits closer to its
// buttons; sag is the visible foot.
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
	const shellRef = useRef<HTMLDivElement>(null);
	const berthRef = useRef<HTMLDivElement>(null);
	const [moon, setMoon] = useState({ left: 0, width: 0 });

	// The moon's chord is centred on the bar, not on the berth. The lockup is
	// wider than the offer, so a berth-relative chord would sit off-centre; both
	// ends are instead the same distance from the midline. Width is capped by the
	// tighter side (the longer cluster) so the arch stays clear of every control
	// and simply lands closer to the longer side.
	useLayoutEffect(() => {
		const shell = shellRef.current;
		const berth = berthRef.current;
		if (!shell || !berth) return;

		const sync = () => {
			const s = shell.getBoundingClientRect();
			const b = berth.getBoundingClientRect();
			const mid = s.left + s.width / 2;
			// 0 = flush with the nearer berth edge; 1 = collapsed to the midline.
			const inset = 0.25;
			const half =
				(1 - inset) * Math.min(mid - b.left, b.right - mid);
			const left = mid - half - s.left;
			const width = 2 * half;
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
	}, []);

	const frame = `pointer-events-none [&_a]:pointer-events-auto [&_button]:pointer-events-auto ${
		placement === "overlay"
			? "absolute inset-x-0 top-0"
			: "relative flex-none"
	}`;

	// STRAIGHT BAR + CENTRED MOON LIP. Fill is `mark` — the same ground the vote
	// slabs use (`Button` solid). Paper aliases mark at :root, so ON_PAPER and
	// the bar stay one light.
	return (
		<div data-masthead className={frame}>
			<div ref={shellRef} className="relative">
				<div className="relative z-10 bg-mark">
					<Navbar berthRef={berthRef} />
				</div>

				{moon.width > 0 && (
					<div
						className="relative z-10 -mt-px"
						style={{
							width: moon.width,
							height: MOON_SAG,
							marginLeft: moon.left,
						}}
					>
						<MoonLimb
							sag="host"
							chord={(width) => width}
							shade={false}
						/>
					</div>
				)}

				<div className="absolute inset-0 z-20" style={ON_PAPER}>
					<Fade
						enter={700}
						delay={180}
						leave={220}
						className="absolute inset-0"
					>
						<div
							className="absolute inset-x-0 bottom-0 flex flex-col px-lg"
							style={{ top: LABEL_TOP }}
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
