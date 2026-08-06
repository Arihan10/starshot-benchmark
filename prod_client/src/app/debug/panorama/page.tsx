"use client";

import { useState, type ReactNode } from "react";
import PanoramaImage from "@/components/PanoramaImage";
import SceneGate from "@/components/SceneGate";
import ViewerHeader from "../ViewerHeader";
import { panoFiles, panoPlaceholderUrl, panoUrl, sceneId, type Scene } from "@/lib/scenes";

export default function PanoramaPage() {
	return (
		<main className="relative flex h-dvh flex-col overflow-hidden bg-ground text-ink">
			<ViewerHeader />

			<div className="relative flex-1">
				<SceneGate>{(scene) => <PanoramaBrowser key={sceneId(scene)} scene={scene} />}</SceneGate>
			</div>
		</main>
	);
}

function PanoramaBrowser({ scene }: { scene: Scene }) {
	const files = panoFiles(scene);
	const [index, setIndex] = useState(0);

	if (files.length === 0) {
		return (
			<div className="absolute inset-0 flex items-center justify-center">
				<span className="text-sm text-ink-40">this scene has no panoramas</span>
			</div>
		);
	}

	const file = files[index];

	return (
		<>
			<PanoramaImage url={panoUrl(scene, file)} placeholderUrl={panoPlaceholderUrl(scene, file)} />

			<PanoramaSelector index={index} count={files.length} onSelect={setIndex} />

			<p className="pointer-events-none absolute left-4 top-4 text-xs text-ink-40">
				drag to look around
			</p>
		</>
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
			<div className="pointer-events-auto flex items-center gap-1 rounded-full border border-mark-8 bg-ground/50 px-2 py-1.5 backdrop-blur">
				<ArrowButton onClick={() => step(-1)} label="Previous panorama">
					‹
				</ArrowButton>
				<span className="w-20 text-center text-xs tabular-nums text-ink-64">
					{String(index).padStart(3, "0")} / {String(count - 1).padStart(3, "0")}
				</span>
				<ArrowButton onClick={() => step(1)} label="Next panorama">
					›
				</ArrowButton>
			</div>
			<div className="pointer-events-auto flex max-w-full gap-1 overflow-x-auto rounded-lg border border-mark-8 bg-ground/50 p-1 backdrop-blur">
				{Array.from({ length: count }, (_, i) => (
					<button
						key={i}
						onClick={() => onSelect(i)}
						className={`h-7 w-7 shrink-0 rounded-md text-xs tabular-nums transition ${
							i === index
								? "bg-mark text-ground"
								: "text-ink-64 hover:bg-mark-8"
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
			className="flex h-7 w-7 items-center justify-center rounded-full text-lg leading-none text-ink transition hover:bg-mark-8"
		>
			{children}
		</button>
	);
}
