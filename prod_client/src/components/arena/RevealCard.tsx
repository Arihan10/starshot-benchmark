"use client";

import type { Ref } from "react";

export const RISE_MS = 400;

export default function RevealCard({
	model,
	elo,
	won,
	align,
	ref,
}: {
	model: string;
	elo: number;
	won: boolean;
	align: "left" | "right";
	ref?: Ref<HTMLDivElement>;
}) {
	return (
		<div
			ref={ref}
			className={`flex w-full min-w-max items-center justify-between gap-md border px-md py-sm transition-colors duration-500 ${
				won
					? "border-mark bg-mark text-background"
					: "border-mark-8 bg-background text-ink-40"
			} ${align === "right" ? "flex-row-reverse" : ""}`}
			style={{ animation: `content-swap ${RISE_MS}ms ease both` }}
		>
			<div className={`flex flex-col gap-1 ${align === "right" ? "items-end" : ""}`}>
				<span className="font-label text-2xs opacity-60">BUILT BY</span>
				<span className="font-sans text-base font-extrabold tracking-[-0.015em] whitespace-nowrap uppercase">
					{model}
				</span>
			</div>

			<div
				className={`flex flex-col gap-1 ${
					align === "left" ? "items-end" : "items-start"
				}`}
			>
				<span className="font-label text-2xs opacity-60">ELO</span>
				<span className="font-sans text-sm tabular-nums">{elo}</span>
			</div>
		</div>
	);
}
