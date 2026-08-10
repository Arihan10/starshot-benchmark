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
// into FAQ or Leaderboard. A caller can pass a tighter `chord` so a long prompt
// opens out to that max and then lets the type shrink; below the floor set out
// underneath, a shorter one no longer pulls the lip in with it.
//
// RADIUS IS FIXED. Sag used to be the constant and radius fell out of
// chord + sag, so short titles (About, FAQ) sat on a tiny circle and looked
// crushed. Radius is pinned to the homepage prompt — "A modern house" at a
// 1440-wide window with the old sag token — and sag now falls out of
// chord + radius. Chord length tracks the caption; curvature does not.
const MOON_RADIUS_PX = 487;

// AND THE LIP HAS A FLOOR, because a fixed radius makes the DIP a consequence of
// the chord: FAQ opened 139px of it and the moon reached five pixels into the
// scene, which is not a moon. This is the homepage prompt's own dip — the other
// half of the measurement the radius is pinned to — so "A modern house" is the
// shallowest the lip is ever drawn, and a shorter title borrows its footprint
// rather than shrinking the moon to its own width.
//
// THE FLOOR IS ON THE DIP, NOT ON THE CHORD, even though the chord is what gets
// clamped. Depth is what the arena sees, so it is the number worth stating; the
// chord that carries it falls out of the circle and re-derives on its own if the
// radius is ever re-pinned. Written the other way round the dip would change
// silently the next time the curvature moved.
const MOON_MIN_SAG_PX = 33.9;

/** Chord that opens MOON_MIN_SAG_PX on the fixed circle: hw = √(s(2r − s)). */
const MOON_MIN_CHORD_PX =
	2 * Math.sqrt(MOON_MIN_SAG_PX * (2 * MOON_RADIUS_PX - MOON_MIN_SAG_PX));

/** Air between a chord end and the nearest nav control. */
const MOON_EDGE_GAP = 12;

// WHERE THE LABEL SITS — the origin's own drop from the top of the masthead.
// The prompt lives in the well BELOW this; moving the prompt never retunes it.
const LABEL_TOP = "calc(var(--spacing-2xs) + 12px)";

// Before the first layout pass — same circle the lip uses.
const ROLL_RADIUS_FALLBACK = MOON_RADIUS_PX;

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
	// as a fraction of one side. When `chord` is set, a longer prompt pushes both
	// ends out to that max and then the type shrinks; a shorter one pulls them in
	// only as far as MOON_MIN_CHORD_PX, which holds the dip at the homepage's.
	// Radius is MOON_RADIUS_PX; only the chord (and thus the sag) moves.
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
			// The lip may neither undercut its own caption nor sit shallower than
			// the homepage's. Capped to the berth last, so a narrow window opens
			// the lip as wide as the bar allows instead of overrunning the nav.
			const minWidth = Math.min(
				maxWidth,
				Math.max(labelWidth, MOON_MIN_CHORD_PX),
			);
			// Chord may not reach a diameter — beyond that the fixed circle cannot
			// host the arc (sag collapses / goes imaginary).
			const chordCeil = Math.min(maxWidth, MOON_RADIUS_PX * 1.98);
			const width =
				chordWanted != null && chordWanted > 0
					? Math.min(chordCeil, Math.max(minWidth, chordWanted))
					: chordCeil;
			const half = width / 2;
			const left = mid - half - s.left;
			const radius = MOON_RADIUS_PX;
			// Sagitta of this chord on the fixed circle — short captions sit on a
			// shallow bite of the same arc the homepage prompt uses.
			const sag =
				width > 0 && half < radius
					? radius - Math.sqrt(radius * radius - half * half)
					: 0;
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
	// THE LIP HANGS; IT DOES NOT PUSH. Absolute under the bar, the limb paints
	// over the arena and the empty berth stays clear — its height is the live
	// sag for this chord on MOON_RADIUS_PX.
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
							height: moon.sag,
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
						bottom: moon.sag > 0 ? -moon.sag : 0,
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
