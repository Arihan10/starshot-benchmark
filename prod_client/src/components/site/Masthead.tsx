"use client";

import type { ReactNode } from "react";
import { useLayoutEffect, useRef, useState } from "react";
import Fade from "./Fade";
import MoonLimb from "./MoonLimb";
import Navbar, { ON_PAPER } from "./Navbar";

// ---------------------------------------------------------------------------
// THE PROMPT IS SET TO THE COMP'S OWN SIZE: 48 units of a 1600-wide viewBox laid
// out at `min(96vw, 1560px)`. Vertical place is not a copied number — the title
// sits in the well below the label (see the caption well below).
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

// THE PROTRUDING MOON — a shallow segment centred on the bar. Its MAXIMUM half
// is the tighter clearance from the screen mid to either nav cluster, minus a
// small edge gap — so the lip opens as wide as the bar allows without running
// into FAQ or Leaderboard. The arena can pass a tighter `chord` so a short
// prompt gets a shorter smile; long prompts open out to that max, then the type
// shrinks. Sag is the visible foot.
const MOON_SAG = "calc(var(--spacing-xl) * 0.62)";

/** Air between a chord end and the nearest nav control. */
const MOON_EDGE_GAP = 12;

// HOW FAR THE MIDDLE BUMP HANGS BELOW THE BAR. Same token as the moon's sag.
export const BELLY = MOON_SAG;

// WHERE THE LABEL SITS — the origin's own drop from the top of the masthead.
// The prompt lives in the well BELOW this; moving the prompt never retunes it.
const LABEL_TOP = "calc(var(--spacing-2xs) + 12px)";

// Fallback disc radius (~1440-wide window) used before the limb has been measured.
const ROLL_RADIUS_FALLBACK = 9600;

/** Arc length of the prompt roll, in CSS pixels — held still across widths. */
const ROLL_ARC_PX = 285;

export type MastheadPrompt = {
	/** Live moon disc radius, px — curvature for CurvedPrompt and the roll pivot. */
	moonRadius: number;
	/** Live moon chord width, px — prompt is capped to this so type stays on the face. */
	moonWidth: number;
	/** Transform origin that puts the pivot on the disc's centre. */
	rollOrigin: string;
	/** Angle whose arc length is ROLL_ARC_PX on the live radius. */
	rollDeg: number;
};

/**
 * Roll angle (degrees) for a constant arc length on the live moon radius.
 * Holding the ANGLE still made the prompt hurl off-screen on wide monitors;
 * holding the DISTANCE keeps the motion the same size you can see.
 */
export function rollDegrees(moonRadius: number): number {
	const r = Math.max(moonRadius, 1);
	return (ROLL_ARC_PX / r) * (180 / Math.PI);
}

/** @deprecated Prefer MastheadPrompt.rollOrigin from the render-prop context. */
export const ROLL_ORIGIN = `50% ${-ROLL_RADIUS_FALLBACK}px`;

function Caption({
	children,
	ref,
}: {
	children: ReactNode;
	ref?: React.Ref<HTMLSpanElement>;
}) {
	return (
		<span
			ref={ref}
			className="font-mono text-2xs tracking-[0.24em] whitespace-nowrap uppercase text-ink-40"
		>
			{children}
		</span>
	);
}

export default function Masthead({
	label,
	placement = "overlay",
	chord: chordWanted,
	children,
}: {
	label: string;
	placement?: "overlay" | "flow";
	/**
	 * Preferred chord width in px (from the prompt / title). Clamped to the
	 * navbar max and floored at the gray label's width — omit for the full lip.
	 */
	chord?: number;
	children: ReactNode | ((ctx: MastheadPrompt) => ReactNode);
}) {
	const shellRef = useRef<HTMLDivElement>(null);
	const leftClusterRef = useRef<HTMLDivElement>(null);
	const rightClusterRef = useRef<HTMLDivElement>(null);
	const limbRef = useRef<HTMLDivElement>(null);
	const labelRef = useRef<HTMLSpanElement>(null);
	const [moon, setMoon] = useState({
		left: 0,
		width: 0,
		sag: 0,
		radius: ROLL_RADIUS_FALLBACK,
	});

	// Chord centred on the screen. Max half = tighter of (mid → FAQ content end,
	// mid → offer start), minus edge gap — read straight from the clusters, not
	// as a fraction of one side. When `chord` is set, a shorter prompt pulls both
	// ends in; long prompts open to that max, then the type shrinks. Floor is
	// the gray label above the title so the lip never undercuts its own caption.
	useLayoutEffect(() => {
		const shell = shellRef.current;
		const leftCluster = leftClusterRef.current;
		const rightCluster = rightClusterRef.current;
		if (!shell || !leftCluster || !rightCluster) return;

		const sync = () => {
			const s = shell.getBoundingClientRect();
			const mid = s.left + s.width / 2;
			const nav = leftCluster.querySelector("nav");
			const last = nav?.lastElementChild ?? leftCluster;
			const edge = last.getBoundingClientRect();
			const padR = Number.parseFloat(getComputedStyle(last).paddingRight) || 0;
			const leftEnd = edge.right - padR;
			const rightStart = rightCluster.getBoundingClientRect().left;
			const maxHalf = Math.max(
				0,
				Math.min(mid - leftEnd, rightStart - mid) - MOON_EDGE_GAP,
			);
			const maxWidth = 2 * maxHalf;
			const labelWidth = labelRef.current?.getBoundingClientRect().width ?? 0;
			const minWidth = Math.min(maxWidth, labelWidth);
			const width =
				chordWanted != null && chordWanted > 0
					? Math.min(maxWidth, Math.max(minWidth, chordWanted))
					: maxWidth;
			const half = width / 2;
			const left = mid - half - s.left;
			const sag = limbRef.current?.offsetHeight ?? 0;
			// Circle through chord + sagitta — same construction as MoonLimb.
			const radius =
				width > 0 && sag > 0
					? (half * half + sag * sag) / (2 * sag)
					: ROLL_RADIUS_FALLBACK;
			setMoon((prev) =>
				prev.left === left &&
				prev.width === width &&
				prev.sag === sag &&
				Math.abs(prev.radius - radius) < 0.5
					? prev
					: { left, width, sag, radius },
			);
		};

		sync();
		document.fonts?.ready.then(sync).catch(() => {});
		const observer = new ResizeObserver(sync);
		observer.observe(shell);
		observer.observe(leftCluster);
		observer.observe(rightCluster);
		const limb = limbRef.current;
		if (limb) observer.observe(limb);
		const labelEl = labelRef.current;
		if (labelEl) observer.observe(labelEl);
		return () => observer.disconnect();
		// Re-bind when the limb host mounts (it only exists once width > 0).
	}, [moon.width, chordWanted, label]);

	const prompt: MastheadPrompt = {
		moonRadius: moon.radius,
		moonWidth: moon.width,
		rollOrigin: `50% ${-moon.radius}px`,
		rollDeg: rollDegrees(moon.radius),
	};
	const body = typeof children === "function" ? children(prompt) : children;

	const frame = `pointer-events-none [&_a]:pointer-events-auto [&_button]:pointer-events-auto ${
		placement === "overlay"
			? "absolute inset-x-0 top-0"
			: "relative flex-none"
	}`;

	// STRAIGHT BAR + CENTRED MOON LIP. Fill is `mark` — the same ground the vote
	// slabs use (`Button` solid). Paper aliases mark at :root, so ON_PAPER and
	// the bar stay one light.
	//
	// THE LIP HANGS; IT DOES NOT PUSH. In flow it added MOON_SAG to the masthead's
	// height and the arena started below that — so left and right of the disc the
	// page ground showed as a black bar between the white navbar and the scene.
	// Absolute under the bar, the limb paints over the arena and the empty berth
	// stays clear.
	return (
		<div data-masthead className={`${frame} z-20`}>
			<div ref={shellRef} className="relative">
				<div className="relative z-10 bg-mark">
					<Navbar
						leftClusterRef={leftClusterRef}
						rightClusterRef={rightClusterRef}
					/>
				</div>

				{moon.width > 0 && (
					<div
						ref={limbRef}
						className="absolute top-full -mt-px bg-transparent"
						style={{
							width: moon.width,
							height: MOON_SAG,
							left: moon.left,
						}}
					>
						<MoonLimb
							sag="host"
							chord={(width) => width}
							shade={false}
						/>
					</div>
				)}

				{/* Extends into the belly so the prompt can sit on the face of
				    the moon, even though the lip no longer stretches the shell. */}
				<div
					className="absolute inset-x-0 top-0 z-20"
					style={{
						...ON_PAPER,
						bottom: `calc(-1 * ${MOON_SAG})`,
					}}
				>
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
								<Caption ref={labelRef}>{label}</Caption>
							</div>
							<div className="flex min-h-0 min-w-0 flex-1 items-center justify-center">
								{/* Cap to the moon chord — a vw ceiling let the prompt outgrow
								    the face, so the arc math broke and the caption jumped. */}
								<div
									className="flex w-full min-w-0 justify-center"
									style={{
										maxWidth: moon.width > 0 ? moon.width : undefined,
									}}
								>
									{body}
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
