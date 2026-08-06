"use client";

import { useMemo, useState } from "react";
import type { RefObject } from "react";
import type { OrbitEngine } from "@/lib/orbit/engine";
import type { OrbitState } from "@/lib/orbit/types";
import { storey, Storey } from "./labels";

const RAIL_COLLAPSED = "w-[68px]";
const RAIL_EXPANDED = "w-72";

export default function FloorRail({
	minimap,
	visited,
	viewedLevel,
	setViewedLevel,
	levelWord,
	engine,
}: {
	minimap: NonNullable<OrbitState["minimap"]>;
	visited: number[];
	viewedLevel: number;
	setViewedLevel: (level: number) => void;
	levelWord: string;
	engine: RefObject<OrbitEngine | null>;
}) {
	const [open, setOpen] = useState(false);
	const { currentLevel, levels } = minimap;
	const visitedSet = useMemo(() => new Set(visited), [visited]);
	const unvisited = useMemo(() => {
		const out = new Set<number>();
		for (const lv of levels)
			if (!lv.points.some((p) => visitedSet.has(p.index))) out.add(lv.level);
		return out;
	}, [levels, visitedSet]);

	return (
		<div
			onPointerEnter={() => setOpen(true)}
			onPointerLeave={() => setOpen(false)}
			className={`absolute bottom-4 right-4 z-10 overflow-hidden rounded-lg border border-mark-8 bg-ground/60 p-3 backdrop-blur transition-[width] duration-200 ease-out ${
				open ? RAIL_EXPANDED : RAIL_COLLAPSED
			}`}
		>
			<div
				className={`overflow-hidden whitespace-nowrap text-[10px] uppercase tracking-wider text-ink-64 transition-all duration-200 ${
					open ? "mb-1.5 h-4 opacity-100" : "mb-0 h-0 opacity-0"
				}`}
			>
				floors
			</div>
			<div className='flex flex-col-reverse gap-0.5'>
				{levels.map((lv) => {
					const isCurrent = lv.level === currentLevel;
					const isViewed = lv.level === viewedLevel;
					const isUnvisited = unvisited.has(lv.level);
					return (
						<button
							key={lv.level}
							type='button'
							title={
								isCurrent
									? "you are on this floor"
									: `go to ${lv.name ?? storey(levelWord, lv.level)}`
							}
							onPointerEnter={() => setViewedLevel(lv.level)}
							onFocus={() => {
								setOpen(true);
								setViewedLevel(lv.level);
							}}
							onClick={() => engine.current?.jumpToLevel(lv.level)}
							className={`flex items-center gap-2.5 overflow-hidden whitespace-nowrap rounded-md p-1.5 text-left transition ${
								isUnvisited
									? "animate-[floor-unvisited_2s_ease-in-out_infinite] text-amber-100"
									: isCurrent
										? "bg-cyan-500/20 text-cyan-100"
										: "text-ink-64 hover:bg-mark-8 hover:text-ink"
							} ${isViewed && !isCurrent ? "ring-1 ring-inset ring-mark-16" : ""}`}
						>
							<span
								className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-[13px] font-semibold tabular-nums ${
									isCurrent ? "bg-cyan-400 text-ground" : "bg-mark-8"
								}`}
							>
								{lv.level + 1}
							</span>
							<span
								className={`min-w-0 flex-1 truncate text-[12px] transition-opacity duration-150 ${
									open ? "opacity-100" : "opacity-0"
								}`}
							>
								{lv.name ?? Storey(levelWord, lv.level)}
							</span>
							{isCurrent && (
								<span
									className={`shrink-0 text-[9px] uppercase tracking-wider text-cyan-300/80 transition-opacity duration-150 ${
										open ? "opacity-100" : "opacity-0"
									}`}
								>
									here
								</span>
							)}
						</button>
					);
				})}
			</div>
		</div>
	);
}
