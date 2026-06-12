"use client";

import { useState, type ReactNode } from "react";
import PanoramaImage from "@/components/PanoramaImage";
import ViewerHeader from "@/components/ViewerHeader";
import { PANORAMA_COUNT } from "@/lib/r2";

export default function PanoramaPage() {
	const [pano, setPano] = useState(0);

	return (
		<main className="relative flex h-dvh flex-col overflow-hidden bg-neutral-950 text-neutral-100">
			<ViewerHeader />

			<div className="relative flex-1">
				<PanoramaImage index={pano} />

				<PanoramaSelector
					index={pano}
					count={PANORAMA_COUNT}
					onSelect={setPano}
				/>

				<p className="pointer-events-none absolute left-4 top-4 text-xs text-neutral-500">
					drag to look around
				</p>
			</div>
		</main>
	);
}

function PanoramaSelector({
	index,
	count,
	onSelect,
}: {
	index: number;
	count: number;
	onSelect: (i: number) => void;
}) {
	const step = (delta: number) => onSelect((index + delta + count) % count);

	return (
		<div className="pointer-events-none absolute inset-x-0 bottom-0 flex flex-col items-center gap-3 p-4">
			<div className="pointer-events-auto flex items-center gap-1 rounded-full border border-white/10 bg-black/50 px-2 py-1.5 backdrop-blur">
				<ArrowButton onClick={() => step(-1)} label="Previous panorama">
					‹
				</ArrowButton>
				<span className="w-20 text-center text-xs tabular-nums text-neutral-300">
					{String(index).padStart(3, "0")} / {String(count - 1).padStart(3, "0")}
				</span>
				<ArrowButton onClick={() => step(1)} label="Next panorama">
					›
				</ArrowButton>
			</div>
			<div className="pointer-events-auto flex max-w-full gap-1 overflow-x-auto rounded-lg border border-white/10 bg-black/50 p-1 backdrop-blur">
				{Array.from({ length: count }, (_, i) => (
					<button
						key={i}
						onClick={() => onSelect(i)}
						className={`h-7 w-7 shrink-0 rounded-md text-xs tabular-nums transition ${
							i === index
								? "bg-white/90 text-neutral-900"
								: "text-neutral-300 hover:bg-white/10"
						}`}
					>
						{i}
					</button>
				))}
			</div>
		</div>
	);
}

function ArrowButton({
	onClick,
	label,
	children,
}: {
	onClick: () => void;
	label: string;
	children: ReactNode;
}) {
	return (
		<button
			onClick={onClick}
			aria-label={label}
			className="flex h-7 w-7 items-center justify-center rounded-full text-lg leading-none text-neutral-200 transition hover:bg-white/10"
		>
			{children}
		</button>
	);
}
