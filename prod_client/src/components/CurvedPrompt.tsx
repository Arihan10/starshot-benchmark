"use client";

import { useId, useLayoutEffect, useRef } from "react";

const VB = 1000;
const CENTRE = VB / 2;

const INSET = 20;

const HALF_SPAN_DEG = 33;

const FONT_MAX = 54;
const FONT_MIN = 17;

const FIT_MARGIN = 0.94;

const rad = (deg: number) => (deg * Math.PI) / 180;

function at(radius: number, deg: number): [number, number] {
	const a = rad(deg);
	return [CENTRE + radius * Math.sin(a), CENTRE + radius * Math.cos(a)];
}

const CEILING = { prompt: FONT_MAX, name: 41 } as const;

const VOICE = {
	prompt: {
		fontFamily: "var(--font-instrument-serif), serif",
		fontStyle: "italic",
		fontWeight: 400,
		letterSpacing: "0.01em",
	},
	name: {
		fontFamily: "var(--font-sans), sans-serif",
		fontStyle: "normal",
		fontWeight: 900,
		letterSpacing: "-0.015em",
		textTransform: "uppercase",
	},
} as const;

export default function CurvedPrompt({
	text,
	diameter,
	className,
	voice = "prompt",
}: {
	text: string;
	diameter: string;
	className?: string;
	voice?: keyof typeof VOICE;
}) {
	const pathRef = useRef<SVGPathElement>(null);
	const textRef = useRef<SVGTextElement>(null);
	const gaugeRef = useRef<SVGTextElement>(null);

	const radius = CENTRE - INSET;
	const [x0, y0] = at(radius, -HALF_SPAN_DEG);
	const [x1, y1] = at(radius, HALF_SPAN_DEG);
	const arc = `M ${x0.toFixed(2)},${y0.toFixed(2)} A ${radius},${radius} 0 0,0 ${x1.toFixed(2)},${y1.toFixed(2)}`;

	const pathId = `prompt-arc-${useId()}`;

	useLayoutEffect(() => {
		const path = pathRef.current;
		const target = textRef.current;
		const gauge = gaugeRef.current;
		if (!path || !target || !gauge) return;

		const fit = () => {
			const needed = gauge.getComputedTextLength();
			if (!needed) return;
			const usable = path.getTotalLength() * FIT_MARGIN;
			const wanted = FONT_MAX * (usable / needed);
			const ceiling = CEILING[voice];
			target.style.fontSize = `${Math.max(FONT_MIN, Math.min(ceiling, wanted))}px`;
		};

		fit();
		document.fonts?.ready.then(fit).catch(() => {});
	}, [text, voice]);

	const typeStyle = VOICE[voice];

	return (
		<svg
			viewBox={`0 0 ${VB} ${VB}`}
			style={{ width: diameter, height: diameter, overflow: "visible" }}
			className={className}
			role="img"
			aria-label={text}
		>
			<defs>
				<path ref={pathRef} id={pathId} d={arc} fill="none" />
			</defs>
			<text
				ref={gaugeRef}
				fontSize={FONT_MAX}
				style={typeStyle}
				visibility="hidden"
				aria-hidden
			>
				{text}
			</text>
			<text ref={textRef} fontSize={FONT_MAX} fill="currentColor" style={typeStyle}>
				<textPath href={`#${pathId}`} startOffset="50%" textAnchor="middle">
					{text}
				</textPath>
			</text>
		</svg>
	);
}
