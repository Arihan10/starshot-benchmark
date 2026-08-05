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

/**
 * A lab's own logo, in its own colours.
 *
 * WHY A LOOKUP AND NOT A FIELD ON THE ROW: the mark belongs to the LAB, not to the
 * model, so it is derived from `lab` rather than stored ten times over. Two
 * Anthropic models cannot end up with different logos, and adding a model from a
 * lab already on the board needs no new data at all.
 *
 * IN COLOUR, which is a reversal — these were mono on a neutral tile on the
 * argument that a column of brand palettes would out-shout the numbers beside it.
 * It does not, because the tile is 30px and the numbers are 21px and 25px: at that
 * ratio the logo is a bullet, not a banner. What colour buys is the thing the
 * column exists for — you find Anthropic's rows, or Google's, by hue at a glance,
 * without reading a single lab name. A column of identical grey glyphs has to be
 * read to be used, which makes it decoration that costs 30px.
 *
 * TWO KINDS OF MARK, because the set is two kinds. Most labs ship a full-colour
 * lockup and take it as-is. The rest — Anthropic, OpenAI, xAI, Moonshot, Stepfun,
 * AI21 — draw in one colour by design, and several of them draw in BLACK, so
 * `colorPrimary` off the icon set would put an invisible logo on a black ground.
 * Those are tinted here instead: the brand's own hue where it has one that carries
 * against black, and white where the brand IS black.
 */
type Mark = {
	Icon: React.ComponentType<{ size?: number; style?: React.CSSProperties }>;
	/** Set only for the single-colour marks; the colour ones bring their own. */
	tint?: string;
	/**
	 * THE LAB'S COLOUR AS A SURFACE, which is not the same thing as its colour as a
	 * logo. The podium is painted in these (see Podium), and a material has to
	 * survive being lit: a mark that is black by design — OpenAI's, xAI's — would
	 * make a pillar that vanishes into the page, and a mark that is a gradient has
	 * no single value to take. So each lab gets ONE hue chosen to read as a lit
	 * solid against black, drawn from the brand rather than from the SVG.
	 */
	tone: string;
};

const MARKS: Record<string, Mark> = {
	Anthropic: { Icon: Anthropic, tint: "#d97757", tone: "#d97757" },
	// Its mark is monochrome by design, so the tone comes from the product's own
	// green rather than from a logo that has no colour to give.
	OpenAI: { Icon: OpenAI, tint: "var(--color-ink)", tone: "#10a37f" },
	Google: { Icon: Gemini.Color, tone: "#4285f4" },
	xAI: { Icon: Grok, tint: "var(--color-ink)", tone: "#8b95a6" },
	Meta: { Icon: Meta.Color, tone: "#1d65c1" },
	Mistral: { Icon: Mistral.Color, tone: "#fa520f" },
	Alibaba: { Icon: Qwen.Color, tone: "#615ced" },
	DeepSeek: { Icon: DeepSeek.Color, tone: "#4d6bfe" },
	Moonshot: { Icon: Moonshot, tint: "var(--color-ink)", tone: "#6b5cff" },
	Zhipu: { Icon: Zhipu.Color, tone: "#3859ff" },
	Cohere: { Icon: Cohere.Color, tone: "#d18ee2" },
	Nova: { Icon: Nova.Color, tone: "#ff9900" },
	Doubao: { Icon: Doubao.Color, tone: "#1e37fc" },
	Microsoft: { Icon: Microsoft.Color, tone: "#00a4ef" },
	Minimax: { Icon: Minimax.Color, tone: "#f23f5d" },
	Nvidia: { Icon: Nvidia.Color, tone: "#76b900" },
	Yi: { Icon: Yi.Color, tone: "#00b37e" },
	Stepfun: { Icon: Stepfun, tint: "var(--color-ink)", tone: "#8b95a6" },
	Ai21: { Icon: Ai21, tint: "#e91e63", tone: "#e91e63" },
};

/**
 * The lab's colour, for anything that is not the logo itself.
 *
 * EXPORTED FROM HERE rather than kept beside the podium, because it belongs to the
 * same fact as the mark: one entry per lab, one place to add the next one. A lab
 * with no entry falls back to the page's own ink, so an unknown model gets a white
 * pillar rather than a black one.
 */
export function brandTone(lab: string): string {
	return MARKS[lab]?.tone ?? "#ededed";
}

export default function BrandMark({
	lab,
	size = 17,
}: {
	lab: string;
	/** The table sets these at row scale; the podium wants them much larger. */
	size?: number;
}) {
	const mark = MARKS[lab];
	// A lab with no icon in the set falls back to its initial rather than to a gap
	// — the tile has a fixed size and an empty one reads as a loading state that
	// never resolves.
	if (!mark) {
		return (
			<span className="font-mono text-[9.5px] font-bold tracking-[0.06em] text-ink-40">
				{lab.slice(0, 2).toUpperCase()}
			</span>
		);
	}
	const { Icon, tint } = mark;
	// `color` rather than a class, because the single-colour marks fill from
	// `currentColor` and the multi-colour ones ignore it — one prop covers both,
	// and the tint stays beside the icon it belongs to instead of in a class list.
	return <Icon size={size} style={tint ? { color: tint } : undefined} />;
}
