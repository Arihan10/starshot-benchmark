"use client";

import { useMemo, useState } from "react";
import type { RefObject } from "react";
import type { OrbitEngine } from "@/lib/orbit/engine";
import type { OrbitState } from "@/lib/orbit/types";
import { storey, Storey } from "./labels";

// The floor control. Inter-floor travel is the one move the scene itself cannot
// offer — the floor you stand on hides everything below it, and a generated scene
// cannot be assumed to model stairs — so the control that CAN offer it gets a
// corner to itself.
//
// It rests as a bare column of storey numbers and opens on hover into the full
// list with names. That split is the point: at rest it costs the 3D almost
// nothing, and it only takes the room needed to read a floor name at the moment
// you are actually reading one.
//
// Laid out HIGHEST STOREY AT THE TOP (flex-col-reverse), so the control is a
// section through the building: moving up the list moves up the scene. Hover a row
// to preview that floor's plan in the minimap, click to go. Unvisited floors
// breathe (the floor-unvisited keyframes) until you have actually stood on them.
//
// Anchored bottom-RIGHT so the pinned edge is the right one and opening grows the
// panel leftward and upward: out of its corner, never across the middle of the
// view.
// Collapsed width is the chip (32) + the button's own padding (12) + the panel's
// (24). That panel padding is deliberately generous: the unvisited pulse is a
// box-shadow reaching ~11px past each button, and the panel clips its overflow to
// keep the names from spilling out while narrow — too tight a padding and the glow
// would be shaved off exactly where it is meant to draw the eye.
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
			className={`absolute bottom-4 right-4 z-10 overflow-hidden rounded-lg border border-white/10 bg-black/60 p-3 backdrop-blur transition-[width] duration-200 ease-out ${
				open ? RAIL_EXPANDED : RAIL_COLLAPSED
			}`}
		>
			<div
				className={`overflow-hidden whitespace-nowrap text-[10px] uppercase tracking-wider text-neutral-400 transition-all duration-200 ${
					open ? "mb-1.5 h-4 opacity-100" : "mb-0 h-0 opacity-0"
				}`}
			>
				floors
			</div>
			<div className='flex flex-col-reverse gap-0.5'>
				{levels.map((lv) => {
					const isCurrent = lv.level === currentLevel;
					const isViewed = lv.level === viewedLevel;
					// Viewing is not visiting: the pulse holds until you go.
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
										: "text-neutral-300 hover:bg-white/10 hover:text-white"
							} ${isViewed && !isCurrent ? "ring-1 ring-inset ring-white/25" : ""}`}
						>
							<span
								className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-[13px] font-semibold tabular-nums ${
									isCurrent ? "bg-cyan-400 text-neutral-950" : "bg-white/10"
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
