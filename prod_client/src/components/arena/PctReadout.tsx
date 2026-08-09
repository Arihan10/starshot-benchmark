"use client";

import { easeOutCubic, useProgress } from "./useProgress";

const DELAY_MS = 0;
const COUNT_MS = 950;

export default function PctReadout({
	share,
	align,
}: {
	share: number;
	align: "left" | "right";
}) {
	const t = useProgress(COUNT_MS, DELAY_MS, easeOutCubic);

	return (
		<div
			className={`pointer-events-none absolute bottom-lg z-20 flex flex-col gap-0.5 ${
				align === "left"
					? "left-lg items-start"
					: "right-lg items-end"
			}`}
			style={{ animation: "arena-rise 520ms cubic-bezier(0.25,0.8,0.3,1) both" }}
		>
			<span className="font-label text-2xs text-ink-40">
				VOTERS PICKED THIS
			</span>
			<span
				className="font-sans font-black leading-[0.86] tracking-[-0.045em] text-foreground tabular-nums"
				style={{
					fontSize: "var(--arena-pct, 64px)",
					textShadow: "0 4px 26px rgb(var(--ground-rgb) / 0.55)",
				}}
			>
				{Math.round(share * t)}%
			</span>
		</div>
	);
}
