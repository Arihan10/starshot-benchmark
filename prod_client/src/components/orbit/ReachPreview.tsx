"use client";

import { useEffect, useState, type CSSProperties } from "react";
import type { ReachPreview as ReachPreviewState } from "@/lib/orbit/types";
import { storey } from "./labels";

const REACH_ACCENT = "#ffc46b";
const FLOOR_ACCENT = "#7ef2c2";

const PANO_SCREEN_W = "clamp(420px, 36vw, 760px)";
const PANO_SCREEN_ASPECT = 2.4;

const PANO_TILE_RATIO = 4.5;

const PANO_PAN_MS = 28000;
const REACH_DOCK_INSET = 16;
const REACH_SLIDE_MS = 300;
const REACH_XFADE_MS = 260;

type ReachLayer = {
	key: number;
	url: string;
	placeholderUrl: string;
	panPhase: number;
};

export default function ReachPreview({
	preview,
	levelWord,
}: {
	preview: ReachPreviewState | null;
	levelWord: string;
}) {
	const [shown, setShown] = useState<ReachPreviewState | null>(null);
	const [layers, setLayers] = useState<ReachLayer[]>([]);

	if (preview && preview.index !== shown?.index) {
		setShown(preview);
		setLayers((prev) => {
			const last = prev[prev.length - 1];
			if (last && last.url === preview.url) return prev;
			return [
				...prev,
				{
					key: (last?.key ?? 0) + 1,
					url: preview.url,
					placeholderUrl: preview.placeholderUrl,
					panPhase: performance.now() % PANO_PAN_MS,
				},
			].slice(-2);
		});
	}

	useEffect(() => {
		if (layers.length < 2) return;
		const t = setTimeout(() => setLayers((l) => l.slice(-1)), REACH_XFADE_MS + 40);
		return () => clearTimeout(t);
	}, [layers]);

	const open = !!preview;
	const level = shown ? shown.level + 1 : 1;
	const delta = shown?.levelDelta ?? 0;
	const accent = delta !== 0 ? FLOOR_ACCENT : REACH_ACCENT;

	return (
		<div
			aria-hidden={!open}
			className='pointer-events-none fixed z-30 will-change-transform'
			style={{
				width: PANO_SCREEN_W,
				right: REACH_DOCK_INSET,
				top: REACH_DOCK_INSET,
				opacity: open ? 1 : 0,
				transform: open
					? "translateX(0)"
					: `translateX(calc(100% + ${REACH_DOCK_INSET * 2}px))`,
				transition: `transform ${REACH_SLIDE_MS}ms cubic-bezier(0.22, 0.61, 0.36, 1), opacity ${REACH_SLIDE_MS}ms ease-out`,
			}}
		>
			<div
				className='overflow-hidden rounded-xl bg-ground/85 shadow-2xl backdrop-blur'
				style={{
					border: `1px solid ${accent}80`,
					containerType: "inline-size",
				}}
			>
				<div
					className='relative overflow-hidden bg-surface'
					style={{ aspectRatio: PANO_SCREEN_ASPECT }}
				>
					{layers.map((layer) => (
						<div
							key={layer.key}
							className='absolute inset-0'
							style={
								{
									backgroundImage: `url(${layer.url}), url(${layer.placeholderUrl})`,
									backgroundRepeat: "repeat-x",
									backgroundSize: `${PANO_TILE_RATIO * 100}% auto`,
									backgroundPositionY: "50%",
									"--pano-tile": `${PANO_TILE_RATIO * 100}cqw`,
									animationName: "pano-pan, reach-layer-in",
									animationDuration: `${PANO_PAN_MS}ms, ${REACH_XFADE_MS}ms`,
									animationTimingFunction: "linear, ease-out",
									animationIterationCount: "infinite, 1",
									animationFillMode: "none, forwards",
									animationDelay: `-${layer.panPhase}ms, 0ms`,
								} as CSSProperties
							}
						/>
					))}
					<div className='pointer-events-none absolute inset-x-0 top-0 h-14 bg-linear-to-b from-black/70 to-transparent' />
					<span
						className='absolute left-3 top-3 rounded px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-ground'
						style={{ background: accent }}
					>
						{delta === 0
							? "out of sight"
							: `${delta > 0 ? "▲" : "▼"} ${storey(levelWord, level - 1)}`}
					</span>
				</div>
				<div
					className='flex items-center justify-between gap-2 px-3 py-2'
					style={{ fontSize: "var(--text-xs)" }}
				>
					<span className='min-w-0 truncate font-semibold text-ink'>
						{shown?.name ?? "unnamed"}
					</span>
					<span
						className='shrink-0 text-[0.8em] uppercase tracking-wider'
						style={{ color: accent }}
					>
						click to go
						{shown && shown.dist < 100 ? ` · ${shown.dist.toFixed(0)} m` : ""}
					</span>
				</div>
			</div>
		</div>
	);
}
