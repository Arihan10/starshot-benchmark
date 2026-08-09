"use client";

import {
	Ai21,
	Anthropic,
	Cohere,
	DeepSeek,
	Doubao,
	Gemini,
	Grok,
	Meta,
	Microsoft,
	Minimax,
	Mistral,
	Moonshot,
	Nova,
	Nvidia,
	OpenAI,
	Qwen,
	Stepfun,
	Yi,
	Zhipu,
} from "@lobehub/icons";
import { useEffect, useRef, useState } from "react";
import { inkVar, markColor } from "@/lib/ink";
import { needsInvertOnPaper } from "./markContrast";

type Mark = {
	Icon: React.ComponentType<{ size?: number; style?: React.CSSProperties }>;
	tint?: string;
	tone: string;
};

/**
 * THE PAGE'S INK, WRITTEN SO IT ACTUALLY FOLLOWS THE PAGE.
 *
 * This was `var(--color-ink)` and it did not. A custom property's own `var()`
 * references are substituted AT THE ELEMENT WHERE IT IS DECLARED, and
 * `--color-ink: rgb(var(--ink-rgb))` is declared once, on `:root`. So it computes
 * to the root's near-white there and INHERITS THAT LITERAL everywhere — a snapshot
 * of the ink, not a reference to it. Re-pointing `--ink-rgb` for a subtree moves
 * `.text-ink` (which writes `rgb(var(--ink-rgb))` straight into the utility, which
 * is what `@theme inline` is for) and leaves every `var(--color-ink)` reader behind.
 *
 * Which is exactly what the standings rows do: hovering one re-points `--ink-rgb`
 * to the paper's ink and lays cream underneath, and every mark tinted this way went
 * on painting itself at 237. Measured on the hovered row, the OpenAI and Grok marks
 * came out at 1.03:1 against the cream — white on off-white, gone.
 *
 * Written as the triplet, the reference survives into the inline style and the mark
 * re-derives with its row like everything else in it. The same trap is waiting in
 * any `var(--color-*)` read from JS or an inline style; the theme's note in
 * globals.css covers the literal-vs-reference half of this, and this is the other
 * half — a reference is only live if it is resolved where it is USED.
 */
const INK = inkVar();

const MARKS: Record<string, Mark> = {
	Anthropic: { Icon: Anthropic, tint: "#d97757", tone: "#d97757" },
	OpenAI: { Icon: OpenAI, tint: INK, tone: "#10a37f" },
	Google: { Icon: Gemini.Color, tone: "#4285f4" },
	xAI: { Icon: Grok, tint: INK, tone: "#8b95a6" },
	Meta: { Icon: Meta.Color, tone: "#1d65c1" },
	Mistral: { Icon: Mistral.Color, tone: "#fa520f" },
	Alibaba: { Icon: Qwen.Color, tone: "#615ced" },
	DeepSeek: { Icon: DeepSeek.Color, tone: "#4d6bfe" },
	Moonshot: { Icon: Moonshot, tint: INK, tone: "#6b5cff" },
	Zhipu: { Icon: Zhipu.Color, tone: "#3859ff" },
	Cohere: { Icon: Cohere.Color, tone: "#d18ee2" },
	Nova: { Icon: Nova.Color, tone: "#ff9900" },
	Doubao: { Icon: Doubao.Color, tone: "#1e37fc" },
	Microsoft: { Icon: Microsoft.Color, tone: "#00a4ef" },
	Minimax: { Icon: Minimax.Color, tone: "#f23f5d" },
	Nvidia: { Icon: Nvidia.Color, tone: "#76b900" },
	Yi: { Icon: Yi.Color, tone: "#00b37e" },
	Stepfun: { Icon: Stepfun, tint: INK, tone: "#8b95a6" },
	Ai21: { Icon: Ai21, tint: "#e91e63", tone: "#e91e63" },
};

export function brandTone(lab: string): string {
	// Resolved mark, not a literal — Three.js materials cannot consume `var()`.
	return MARKS[lab]?.tone ?? markColor();
}

export default function BrandMark({
	lab,
	size = 17,
}: {
	lab: string;
	size?: number;
}) {
	const mark = MARKS[lab];
	const host = useRef<HTMLSpanElement>(null);
	const [invert, setInvert] = useState(false);

	// ASKED ONCE PER LAB, of the artwork itself — see markContrast. The verdict is
	// cached there, so the twenty rows of a sorted board and every keystroke in the
	// search field share one rasterisation.
	//
	// It cannot be answered during render: it needs the mark IN THE DOCUMENT, with
	// its paint resolved, which is only true after commit. So a mark that turns out
	// to need inverting starts its first frame un-inverted — invisible only if the
	// row happens to be hovered within that frame, and correct from then on.
	useEffect(() => {
		const svg = host.current?.querySelector("svg");
		if (!svg) return;
		let live = true;
		needsInvertOnPaper(lab, svg).then((verdict) => {
			if (live) setInvert(verdict);
		});
		return () => {
			live = false;
		};
	}, [lab]);

	if (!mark) {
		return (
			<span className="font-mono text-[9.5px] font-black tracking-[0.06em] text-ink-40">
				{lab.slice(0, 2).toUpperCase()}
			</span>
		);
	}

	const { Icon, tint } = mark;
	return (
		// FLIPPED ONLY UNDER A LIT ROW, which is the only place the paper appears.
		// `group-hover/row` is the standings table's group and simply never matches
		// anywhere else the mark is used — the podium takes the same component and is
		// never on cream, so it needs no exception written for it.
		//
		// IT CROSSES ON THE ROW'S CLOCK. The ground fades over 160ms; a filter that
		// snapped would land ahead of it and read as a flicker rather than as the
		// mark turning with its row. Interpolating `invert()` passes through a flat
		// mid-grey on the way, which at this duration reads as the flip it is.
		<span
			ref={host}
			className={`flex transition-[filter] duration-[160ms] ${
				invert ? "group-hover/row:invert" : ""
			}`}
		>
			<Icon size={size} style={tint ? { color: tint } : undefined} />
		</span>
	);
}
