"use client";

import type { ReactNode } from "react";
import { useLayoutEffect, useRef, useState } from "react";
import { RAKE_PX } from "@/components/ui/Button";
import Fade from "./Fade";
import Navbar, { ON_PAPER } from "./Navbar";

// ---------------------------------------------------------------------------
// THE PROMPT IS SET TO THE COMP'S OWN SIZE: 48 units of a 1600-wide viewBox laid
// out at `min(96vw, 1560px)`. Vertical place is not a copied number — the title
// sits in the well below the label (see the caption well below).
// ---------------------------------------------------------------------------

const UNIT = "min(0.06vw, 0.975px)";
const TITLE_SIZE = `calc(48 * ${UNIT})`;

// How far the white plate hangs below the black bar — a hair, not a tongue.
const PLATE_SAG = "var(--spacing-xs)";

// How far the plate peeks past the bar's top edge. Paired with the drop-shadow,
// that overhang is what makes the plate read as sitting ABOVE the navbar.
const PLATE_OVERHANG = 2;

/** Air between a plate end and the nearest nav control. */
const PLATE_EDGE_GAP = 8;

// Air either side of the prompt on the plate's foot — in title-size ems so the
// foot grows with the type. Without this the plate tracks the words alone and
// reads as a tall stub; the margin is what stretches it long.
const PLATE_END_MARGIN_EMS = 1.5;

// The offer buttons lean RAKE_PX over their own height. The same angle on this
// wider plate reads soft — scale the lean so the cut keeps the buttons' bite.
const PLATE_RAKE_SCALE = 2.25;

const VOICE = {
	// ITALIC NEEDS A BEARING, upright does not — `truncate` cuts at the box edge
	// and an italic face leans past its advance width. Symmetric padding keeps
	// the centre. In `em`, because the overhang scales with the type.
	prompt: "font-serif italic tracking-[0.01em] px-[0.14em]",
	name: "font-sans font-black tracking-[-0.015em] uppercase",
} as const;

type Voice = keyof typeof VOICE;

const VOICE_STYLE: Partial<Record<Voice, React.CSSProperties>> = {
	prompt: { fontSize: TITLE_SIZE },
	name: { fontSize: TITLE_SIZE },
};

/** Off-DOM gauge styles — must match Title so the plate foot tracks the ink. */
function gaugeCss(voice: Voice): string {
	const shared = [
		"position:fixed",
		"left:0",
		"top:0",
		"visibility:hidden",
		"pointer-events:none",
		"white-space:nowrap",
		`font-size:${TITLE_SIZE}`,
	];
	if (voice === "name") {
		return [
			...shared,
			"font-family:var(--font-sans), sans-serif",
			"font-weight:900",
			"letter-spacing:-0.015em",
			"text-transform:uppercase",
		].join(";");
	}
	return [
		...shared,
		"font-family:var(--font-instrument-serif), serif",
		"font-style:italic",
		"font-weight:400",
		"letter-spacing:0.01em",
		"padding:0 0.14em",
	].join(";");
}

export function measureTitleWidth(
	text: string,
	voice: Voice = "prompt",
): number {
	if (typeof document === "undefined" || !text) return 0;
	const id = `title-gauge-${voice}`;
	let gauge = document.getElementById(id) as HTMLSpanElement | null;
	if (!gauge) {
		gauge = document.createElement("span");
		gauge.id = id;
		document.body.appendChild(gauge);
	}
	gauge.style.cssText = gaugeCss(voice);
	gauge.textContent = text;
	return gauge.getBoundingClientRect().width;
}

/** Live width of a Title at the current viewport size. */
export function useTitleWidth(text: string, voice: Voice = "prompt"): number {
	const [width, setWidth] = useState(0);

	useLayoutEffect(() => {
		const sync = () => {
			const next = measureTitleWidth(text, voice);
			setWidth((prev) => (Math.abs(prev - next) < 0.5 ? prev : next));
		};
		sync();
		document.fonts?.ready.then(sync).catch(() => {});
		window.addEventListener("resize", sync);
		return () => window.removeEventListener("resize", sync);
	}, [text, voice]);

	return width;
}

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
	captionWidth,
	children,
}: {
	label: string;
	placement?: "overlay" | "flow";
	/** Prompt/title advance in px — plate foot is this plus end margins. */
	captionWidth?: number;
	children: ReactNode;
}) {
	const shellRef = useRef<HTMLDivElement>(null);
	const leftClusterRef = useRef<HTMLDivElement>(null);
	const rightClusterRef = useRef<HTMLDivElement>(null);
	const plateRef = useRef<HTMLDivElement>(null);
	const probeRef = useRef<HTMLSpanElement>(null);
	const [plate, setPlate] = useState({
		left: 0,
		width: 0,
		bottom: 0,
		slant: 0,
		sag: 0,
	});

	// Bottom edge = caption + end margins. Slant is the offer-button lean
	// (RAKE_PX / btnH) scaled up by PLATE_RAKE_SCALE — same cut, larger face.
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
			const padR =
				Number.parseFloat(getComputedStyle(last).paddingRight) || 0;
			const leftEnd = edge.right - padR;
			const rightStart = rightCluster.getBoundingClientRect().left;
			const berth =
				2 *
				Math.max(
					0,
					Math.min(mid - leftEnd, rightStart - mid) - PLATE_EDGE_GAP,
				);

			const probe = probeRef.current;
			const sag = probe?.getBoundingClientRect().height ?? 0;
			const typePx = probe
				? Number.parseFloat(getComputedStyle(probe).fontSize)
				: 0;
			const endMargin = PLATE_END_MARGIN_EMS * typePx;

			const offer =
				rightCluster.querySelector("a, button") ?? rightCluster;
			const btnH = offer.getBoundingClientRect().height;
			// Plate height before it exists: padding + bar + sag.
			const barH =
				shell.querySelector(":scope > .bg-ground")?.getBoundingClientRect()
					.height ?? 0;
			const plateH =
				plateRef.current?.getBoundingClientRect().height ||
				PLATE_OVERHANG + barH + sag;

			const slant =
				btnH > 0
					? plateH * (RAKE_PX / btnH) * PLATE_RAKE_SCALE
					: RAKE_PX * PLATE_RAKE_SCALE;

			const foot = Math.min(
				(captionWidth ?? 0) + 2 * endMargin,
				berth,
			);
			if (foot <= 0 || berth <= 0) {
				setPlate((prev) =>
					prev.width === 0 ? prev : { left: 0, width: 0, bottom: 0, slant: 0, sag },
				);
				return;
			}

			let width = foot + 2 * slant;
			let bottom = foot;
			if (width > berth) {
				width = berth;
				bottom = Math.max(0, width - 2 * slant);
			}
			const left = mid - width / 2 - s.left;

			setPlate((prev) =>
				prev.left === left &&
				prev.width === width &&
				Math.abs(prev.bottom - bottom) < 0.25 &&
				Math.abs(prev.slant - slant) < 0.25 &&
				Math.abs(prev.sag - sag) < 0.25
					? prev
					: { left, width, bottom, slant, sag },
			);
		};

		sync();
		document.fonts?.ready.then(sync).catch(() => {});
		const observer = new ResizeObserver(sync);
		observer.observe(shell);
		observer.observe(leftCluster);
		observer.observe(rightCluster);
		const plateEl = plateRef.current;
		if (plateEl) observer.observe(plateEl);
		return () => observer.disconnect();
	}, [plate.width, captionWidth]);

	const frame = `pointer-events-none [&_a]:pointer-events-auto [&_button]:pointer-events-auto ${
		placement === "overlay"
			? "absolute inset-x-0 top-0"
			: "relative flex-none"
	}`;

	return (
		<div data-masthead className={`${frame} z-20`}>
			{/* Padding keeps the plate's top overhang in-flow so it is not clipped
			    at the viewport edge; the bar sits below it. */}
			<div
				ref={shellRef}
				className="relative"
				style={{ paddingTop: PLATE_OVERHANG }}
			>
				<span
					ref={probeRef}
					aria-hidden
					className="pointer-events-none invisible absolute w-0"
					style={{ fontSize: TITLE_SIZE, height: PLATE_SAG }}
				/>
				<div className="relative z-10 bg-ground">
					<Navbar
						leftClusterRef={leftClusterRef}
						rightClusterRef={rightClusterRef}
					/>
				</div>

				{plate.width > 0 && plate.sag > 0 && (
					<div
						ref={plateRef}
						aria-hidden
						className="pointer-events-none absolute z-20 bg-mark"
						style={{
							left: plate.left,
							width: plate.width,
							top: 0,
							bottom: -plate.sag,
							clipPath: `polygon(0 0, 100% 0, calc(100% - ${plate.slant}px) 100%, ${plate.slant}px 100%)`,
							filter: "drop-shadow(0 4px 10px rgb(0 0 0 / 0.45))",
						}}
					/>
				)}

				<div
					className="absolute inset-x-0 z-30"
					style={{
						...ON_PAPER,
						top: 0,
						bottom: `calc(-1 * ${PLATE_SAG})`,
					}}
				>
					<Fade
						enter={700}
						delay={180}
						leave={220}
						className="absolute inset-0"
					>
						<div className="absolute inset-0 flex flex-col items-center justify-center gap-2xs px-lg">
							<Caption>{label}</Caption>
							<div
								className="flex min-w-0 justify-center"
								style={{
									width:
										plate.bottom > 0
											? plate.bottom
											: undefined,
									maxWidth:
										plate.bottom > 0
											? plate.bottom
											: undefined,
								}}
							>
								{children}
							</div>
						</div>
					</Fade>
				</div>
			</div>
		</div>
	);
}

export function Title({
	voice = "prompt",
	className = "",
	children,
}: {
	voice?: Voice;
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
