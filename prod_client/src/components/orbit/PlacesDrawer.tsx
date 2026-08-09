"use client";

import { useMemo, useState } from "react";
import type { RefObject } from "react";
import ScrollBox from "@/components/site/ScrollBox";
import type { OrbitEngine } from "@/lib/orbit/engine";
import type { Chapter, OrbitState } from "@/lib/orbit/types";

export default function PlacesDrawer({
	state,
	engine,
	onClose,
}: {
	state: OrbitState;
	engine: RefObject<OrbitEngine | null>;
	onClose: () => void;
}) {
	const [query, setQuery] = useState("");
	const results = useMemo(() => {
		const q = query.trim().toLowerCase();
		if (!q) return [];
		return state.nodes
			.filter(
				(n) =>
					(n.name ?? "").toLowerCase().includes(q) ||
					(n.zone ?? "").toLowerCase().includes(q),
			)
			.slice(0, 20);
	}, [query, state.nodes]);

	const go = (index: number) => {
		engine.current?.traverseTo(index);
		onClose();
	};

	return (
		<div className='absolute right-4 top-16 z-30 flex max-h-[70vh] w-72 flex-col rounded-lg border border-mark-8 bg-ground/90 p-2 shadow-2xl backdrop-blur'>
			<input
				autoFocus
				value={query}
				onChange={(e) => setQuery(e.target.value)}
				placeholder='take me to…'
				className='mb-2 w-full rounded-md border border-mark-16 bg-ground/50 px-2.5 py-1.5 text-xs text-ink outline-none transition placeholder:text-ink-40 focus:border-accent'
			/>
			<ScrollBox className='min-h-0 flex-1'>
				{query.trim() ? (
					results.length ? (
						results.map((n) => (
							<button
								key={n.index}
								type='button'
								onClick={() => go(n.index)}
								className='block w-full truncate rounded px-2 py-1.5 text-left text-xs text-ink transition hover:bg-accent/20 hover:text-ink'
							>
								{n.name ?? `node ${n.index + 1}`}
								{n.zone ? <span className='text-ink-40'> · {n.zone}</span> : null}
							</button>
						))
					) : (
						<div className='px-2 py-1.5 text-xs text-ink-40'>no matches</div>
					)
				) : (
					<ChapterList chapters={state.chapters} onGo={go} />
				)}
			</ScrollBox>
		</div>
	);
}

function ChapterList({
	chapters,
	onGo,
}: {
	chapters: Chapter[];
	onGo: (index: number) => void;
}) {
	if (chapters.length === 0)
		return <div className='px-2 py-1.5 text-xs text-ink-40'>no zones</div>;
	return (
		<>
			<div className='mb-1 px-1 text-[9px] uppercase tracking-wider text-ink-40'>
				chapters
			</div>
			{chapters.map((c) => (
				<button
					key={c.zone || c.firstIndex}
					type='button'
					onClick={() => onGo(c.firstIndex)}
					className='flex w-full items-center justify-between gap-2 rounded px-2 py-1.5 text-left text-xs text-ink transition hover:bg-accent/20 hover:text-ink'
				>
					<span className='min-w-0 truncate'>{c.zone || "unzoned"}</span>
					<span className='shrink-0 text-[9px] tabular-nums text-ink-40'>{c.count}</span>
				</button>
			))}
		</>
	);
}
