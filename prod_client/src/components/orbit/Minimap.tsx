"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import type { OrbitEngine } from "@/lib/orbit/engine";
import type { OrbitState } from "@/lib/orbit/types";
import { storey } from "./labels";

function useFacingLoop(
	engineRef: RefObject<OrbitEngine | null>,
	cb: (facingDeg: number) => void,
) {
	useEffect(() => {
		let raf = 0;
		const loop = () => {
			const deg = engineRef.current?.getFacingDeg();
			if (typeof deg === "number") cb(deg);
			raf = requestAnimationFrame(loop);
		};
		raf = requestAnimationFrame(loop);
		return () => cancelAnimationFrame(raf);
	}, [engineRef, cb]);
}

const CONE_HALF_ANGLE = 33;
const CONE_SPAN = 260;
const CONE_CLIP = `polygon(50% 50%, 100% ${
	50 - 50 * Math.tan((CONE_HALF_ANGLE * Math.PI) / 180)
}%, 100% ${50 + 50 * Math.tan((CONE_HALF_ANGLE * Math.PI) / 180)}%)`;

const minimapWidth = (aspect: number, maxW: string, maxH: string) =>
	`min(${maxW}, calc(${maxH} * ${aspect}))`;

const MINIMAP_COMPACT = { w: "calc(var(--spacing-xl) * 7.9)", h: "calc(var(--spacing-xl) * 6.9)" };
const MINIMAP_EXPANDED = { w: "clamp(420px, 56vw, 900px)", h: "clamp(340px, 72vh, 820px)" };

export default function Minimap({
	minimap,
	currentIndex,
	viewedLevel,
	levelWord,
	engine,
}: {
	minimap: NonNullable<OrbitState["minimap"]>;
	currentIndex: number;
	viewedLevel: number;
	levelWord: string;
	engine: RefObject<OrbitEngine | null>;
}) {
	const [expanded, setExpanded] = useState(false);
	const { currentLevel, levels } = minimap;

	const [flash, setFlash] = useState(0);
	const [prevIdx, setPrevIdx] = useState(currentIndex);
	if (currentIndex !== prevIdx) {
		setPrevIdx(currentIndex);
		setFlash((n) => n + 1);
	}
	useEffect(() => {
		if (!flash) return;
		const t = setTimeout(() => setFlash(0), 450);
		return () => clearTimeout(t);
	}, [flash]);

	const coneRef = useRef<HTMLDivElement>(null);
	const spin = useCallback((facingDeg: number) => {
		if (coneRef.current)
			coneRef.current.style.transform = `translate(-50%, -50%) rotate(${facingDeg}deg)`;
	}, []);
	useFacingLoop(engine, spin);

	const view = levels[viewedLevel] ?? levels[currentLevel];
	if (!view) return null;
	const caps = expanded ? MINIMAP_EXPANDED : MINIMAP_COMPACT;
	const onCurrentFloor = view.level === currentLevel;

	const travelToPoint = (e: React.MouseEvent<HTMLDivElement>) => {
		const r = e.currentTarget.getBoundingClientRect();
		if (!r.width || !r.height) return;
		const left = ((e.clientX - r.left) / r.width) * 100;
		const top = ((e.clientY - r.top) / r.height) * 100;
		let index = -1;
		let best = Infinity;
		for (const pt of view.points) {
			const dx = (pt.leftPct - left) * view.aspect;
			const dy = pt.topPct - top;
			const d = dx * dx + dy * dy;
			if (d < best) {
				best = d;
				index = pt.index;
			}
		}
		if (index >= 0) engine.current?.traverseTo(index);
	};

	return (
		<div className='rounded-md border border-mark-8 bg-ground/60 p-1.5 backdrop-blur'>
			<div className='mb-1 flex items-center justify-between gap-2 px-0.5 text-[9px] uppercase tracking-wider text-ink-64'>
				<span className='truncate'>{view.name ?? storey(levelWord, view.level)}</span>
				<button
					type='button'
					onClick={() => setExpanded((v) => !v)}
					title={expanded ? "collapse minimap" : "expand minimap"}
					aria-label={expanded ? "collapse minimap" : "expand minimap"}
					className='rounded px-1 text-[11px] leading-none text-ink-64 transition hover:bg-mark-8 hover:text-ink'
				>
					{expanded ? "✕" : "⤢"}
				</button>
			</div>
			<div
				onClick={travelToPoint}
				title='click anywhere to travel there'
				className='relative cursor-pointer overflow-hidden rounded'
				style={{
					width: minimapWidth(view.aspect, caps.w, caps.h),
					aspectRatio: view.aspect,
					containerType: "inline-size",
				}}
			>
				{/* eslint-disable-next-line @next/next/no-img-element -- runtime R2 slice via /r2 proxy */}
				<img
					src={view.url}
					alt='scene from above'
					draggable={false}
					className='absolute max-w-none'
					style={{
						transition:
							"width 320ms ease-out, height 320ms ease-out, left 320ms ease-out, top 320ms ease-out",
						width: `${100 / (view.crop.u1 - view.crop.u0)}%`,
						height: `${100 / (view.crop.v1 - view.crop.v0)}%`,
						left: `${(-100 * view.crop.u0) / (view.crop.u1 - view.crop.u0)}%`,
						top: `${(-100 * view.crop.v0) / (view.crop.v1 - view.crop.v0)}%`,
					}}
				/>
				{onCurrentFloor && (
					<div
						ref={coneRef}
						className='pointer-events-none absolute'
						style={{
							left: `${view.points.find((p) => p.current)?.leftPct ?? 50}%`,
							top: `${view.points.find((p) => p.current)?.topPct ?? 50}%`,
							width: `${CONE_SPAN}%`,
							aspectRatio: 1,
							clipPath: CONE_CLIP,
							background:
								"radial-gradient(circle at 50% 50%, rgb(var(--mark-rgb) / 0.50) 0%, rgb(var(--mark-rgb) / 0.28) 18%, rgb(var(--mark-rgb) / 0.10) 42%, rgb(var(--mark-rgb) / 0) 72%)",
							mixBlendMode: "screen",
							transform: "translate(-50%, -50%)",
						}}
					/>
				)}
				{view.labels.map((lab) => (
					<button
						key={lab.id}
						type='button'
						title={`go to ${lab.label}`}
						onClick={(e) => {
							e.stopPropagation();
							engine.current?.traverseTo(lab.index);
						}}
						style={{
							left: `${lab.leftPct}%`,
							top: `${lab.topPct}%`,
							fontSize: "var(--text-xs)",
							textShadow:
								"0 1px 3px rgb(var(--ground-rgb) / 0.95), 0 0 8px rgb(var(--ground-rgb) / 0.8)",
						}}
						className='absolute -translate-x-1/2 -translate-y-1/2 whitespace-nowrap rounded px-1 py-0.5 text-center font-medium leading-tight text-ink transition hover:bg-accent/25 hover:text-ink'
					>
						{lab.label}
					</button>
				))}
				{view.points
					.filter((pt) => pt.current)
					.map((pt) => (
						<span
							key={pt.index}
							style={{
								left: `${pt.leftPct}%`,
								top: `${pt.topPct}%`,
								boxShadow: flash
									? "0 0 10px 4px rgb(var(--accent-rgb) / 0.95)"
									: "0 0 6px 2px rgb(var(--accent-rgb) / 0.7)",
							}}
							className={`pointer-events-none absolute -translate-x-1/2 -translate-y-1/2 rounded-full border border-mark bg-accent ${
								expanded ? "h-3.5 w-3.5" : "h-2.5 w-2.5"
							}`}
						/>
					))}
				{!onCurrentFloor && (
					<div className='pointer-events-none absolute bottom-1 left-1 rounded bg-ground/65 px-1 py-0.5 text-[8px] uppercase tracking-wider text-accent'>
						{storey(levelWord, view.level)} · you are on {currentLevel + 1}
					</div>
				)}
			</div>
		</div>
	);
}
