"use client";

import { useId, useLayoutEffect, useRef, useState } from "react";

// Same slot size as Masthead Title — the layout box must stay one line tall so
// flex-centering lands it where the flat caption used to sit. Scales with the
// viewport; fit() may shrink further so a long prompt still rides the moon.
const UNIT = "min(0.06vw, 0.975px)";
const TITLE_SIZE = `calc(48 * ${UNIT})`;

/** Hard floor — only when the moon itself is tiny; prefer shrinking with the slot. */
const FONT_FLOOR = 12;
const FIT_MARGIN = 0.9;
/** Baseline within the 1em line box (approx. where a serif sits). */
const BASELINE = 0.82;

/** Name voice sits a step under the prompt ceiling at the same slot size. */
const NAME_SCALE = 41 / 48;

const VOICE = {
	prompt: {
		fontFamily: "var(--font-instrument-serif), serif",
		fontStyle: "italic" as const,
		fontWeight: 400,
		letterSpacing: "0.01em",
	},
	name: {
		fontFamily: "var(--font-sans), sans-serif",
		fontStyle: "normal" as const,
		fontWeight: 900,
		letterSpacing: "-0.015em",
		textTransform: "uppercase" as const,
	},
};

const GAUGE_STYLE = [
	"position:fixed",
	"left:0",
	"top:0",
	"visibility:hidden",
	"pointer-events:none",
	"white-space:nowrap",
	`font-size:${TITLE_SIZE}`,
	"font-family:var(--font-instrument-serif),serif",
	"font-style:italic",
	"font-weight:400",
	"letter-spacing:0.01em",
].join(";");

/** Flat advance of the prompt at title size → chord the moon should open to. */
export function measurePromptChord(text: string): number {
	if (typeof document === "undefined" || !text) return 0;
	let gauge = document.getElementById(
		"prompt-chord-gauge",
	) as HTMLSpanElement | null;
	if (!gauge) {
		gauge = document.createElement("span");
		gauge.id = "prompt-chord-gauge";
		gauge.style.cssText = GAUGE_STYLE;
		document.body.appendChild(gauge);
	}
	gauge.textContent = text;
	const advance = gauge.getBoundingClientRect().width;
	// Shallow arc length ≈ chord; FIT_MARGIN is the same headroom fit() keeps.
	return advance > 0 ? advance / FIT_MARGIN : 0;
}

/** Live chord needed for `text` at the current viewport title size. */
export function usePromptChord(text: string): number {
	const [chord, setChord] = useState(0);

	useLayoutEffect(() => {
		const sync = () => {
			const next = measurePromptChord(text);
			setChord((prev) => (Math.abs(prev - next) < 0.5 ? prev : next));
		};
		sync();
		document.fonts?.ready.then(sync).catch(() => {});
		window.addEventListener("resize", sync);
		return () => window.removeEventListener("resize", sync);
	}, [text]);

	return chord;
}

/**
 * A caption set on a circular arc that matches the masthead moon.
 *
 * The HOST is one line tall at the masthead title size — same box Title used —
 * so the caption well can centre it. Width is capped to the moon chord by the
 * parent; type then shrinks to the live arc length so it never leaves the face.
 * The arc is lifted by its own sag so middle letters sit in the title slot.
 */
export default function CurvedPrompt({
	text,
	radius,
	className = "",
	voice = "prompt",
}: {
	text: string;
	/** Moon disc radius, in CSS pixels. */
	radius: number;
	className?: string;
	voice?: keyof typeof VOICE;
}) {
	const hostRef = useRef<HTMLDivElement>(null);
	const pathRef = useRef<SVGPathElement>(null);
	const textRef = useRef<SVGTextElement>(null);
	const gaugeRef = useRef<SVGTextElement>(null);
	const [chord, setChord] = useState(0);
	const [em, setEm] = useState(0);
	// useId() includes colons (`:r0:`) which break SVG fragment hrefs.
	const pathId = `prompt-arc-${useId().replace(/:/g, "")}`;

	useLayoutEffect(() => {
		const host = hostRef.current;
		if (!host) return;
		const sync = () => {
			const w = host.clientWidth;
			const nextEm = Number.parseFloat(getComputedStyle(host).fontSize);
			if (w > 0) setChord((prev) => (prev === w ? prev : w));
			if (nextEm > 0)
				setEm((prev) => (Math.abs(prev - nextEm) < 0.25 ? prev : nextEm));
		};
		sync();
		const ro = new ResizeObserver(sync);
		ro.observe(host);
		return () => ro.disconnect();
	}, []);

	useLayoutEffect(() => {
		const path = pathRef.current;
		const target = textRef.current;
		const gauge = gaugeRef.current;
		if (!path || !target || !gauge || chord < 1 || em < 1) return;

		const ceiling = voice === "name" ? em * NAME_SCALE : em;

		const fit = () => {
			gauge.style.fontSize = `${ceiling}px`;
			const needed = gauge.getComputedTextLength();
			if (!needed) return;
			const usable = path.getTotalLength() * FIT_MARGIN;
			const wanted = ceiling * (usable / needed);
			target.style.fontSize = `${Math.max(FONT_FLOOR, Math.min(ceiling, wanted))}px`;
		};

		fit();
		document.fonts?.ready.then(fit).catch(() => {});
	}, [text, voice, chord, radius, em]);

	const r = Math.max(radius, 1);
	// Never span past a semicircle — a chord wider than the disc breaks the arc
	// (sag → 0) and the caption detaches from the moon.
	const span = chord > 0 ? Math.min(chord, r * 1.98) : 0;
	const hw = span / 2;
	const sag = span > 0 && hw < r ? r - Math.sqrt(r * r - hw * hw) : 0;
	const y0 = em > 0 ? em * BASELINE : 0;
	// Tall enough for the sag below the baseline; host stays 1em for layout.
	const vbH = Math.max(em, y0 + sag + em * 0.2);
	// The well centres the 1em host. Without a lift the arc's midpoint sits
	// `sag` below the flat baseline, so the caption reads under the moon.
	// Raise by that sag: middle letters land where Title's ink did.
	const lift = sag;
	const arc =
		span > 0 && y0 > 0
			? `M 0,${y0.toFixed(2)} A ${r.toFixed(2)},${r.toFixed(2)} 0 0 0 ${span.toFixed(2)},${y0.toFixed(2)}`
			: "";
	const xOff = chord > span ? (chord - span) / 2 : 0;

	const typeStyle = VOICE[voice];

	return (
		<div
			ref={hostRef}
			className={`relative w-full min-w-0 leading-none ${className}`}
			style={{ fontSize: TITLE_SIZE }}
		>
			{/* Strut — same 1em line box Title had, so the well centres us alike. */}
			<span aria-hidden className="invisible inline-block">
				&nbsp;
			</span>
			{span > 0 && em > 0 && (
				<svg
					viewBox={`0 0 ${span} ${vbH}`}
					width={span}
					height={vbH}
					className="pointer-events-none absolute overflow-visible text-ink"
					style={{ top: -lift, left: xOff }}
					role="img"
					aria-label={text}
				>
					<defs>
						<path ref={pathRef} id={pathId} d={arc} fill="none" />
					</defs>
					<text
						ref={gaugeRef}
						fontSize={em}
						style={typeStyle}
						visibility="hidden"
						aria-hidden
					>
						{text}
					</text>
					<text
						ref={textRef}
						fontSize={em}
						fill="currentColor"
						style={typeStyle}
					>
						<textPath
							href={`#${pathId}`}
							startOffset="50%"
							textAnchor="middle"
						>
							{text}
						</textPath>
					</text>
				</svg>
			)}
		</div>
	);
}
