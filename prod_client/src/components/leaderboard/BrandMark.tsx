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

type Mark = {
	Icon: React.ComponentType<{ size?: number; style?: React.CSSProperties }>;
	tint?: string;
	tone: string;
};

const MARKS: Record<string, Mark> = {
	Anthropic: { Icon: Anthropic, tint: "#d97757", tone: "#d97757" },
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

export function brandTone(lab: string): string {
	return MARKS[lab]?.tone ?? "#ededed";
}

export default function BrandMark({
	lab,
	size = 17,
}: {
	lab: string;
	size?: number;
}) {
	const mark = MARKS[lab];
	if (!mark) {
		return (
			<span className="font-mono text-[9.5px] font-bold tracking-[0.06em] text-ink-40">
				{lab.slice(0, 2).toUpperCase()}
			</span>
		);
	}
	const { Icon, tint } = mark;
	return <Icon size={size} style={tint ? { color: tint } : undefined} />;
}
