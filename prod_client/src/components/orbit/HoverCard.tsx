"use client";

import { groundVar, signalVar } from "@/lib/ink";
import type { EdgeType } from "@/lib/orbit/navGraph";
import type { HoverPreview } from "@/lib/orbit/types";

export const EDGE_META: Record<EdgeType, { label: string; verb: string; color: string }> = {
	walk: { label: "Walk", verb: "walk over", color: "#8fd0ff" },
	portal: { label: "Doorway", verb: "step through", color: signalVar() },
	vertical: { label: "Level", verb: "change level", color: "#7ef2c2" },
	phase: { label: "Phase", verb: "phase through wall", color: "#c9a6ff" },
	far: { label: "Travel", verb: "travel across", color: "#9aa7b4" },
};

export default function HoverCard({ preview }: { preview: HoverPreview }) {
	const meta = EDGE_META[preview.type];
	return (
		<div
			className='pointer-events-none absolute z-20 w-48 -translate-x-1/2 -translate-y-[calc(100%+14px)] overflow-hidden rounded-lg border bg-ground/85 shadow-2xl backdrop-blur'
			style={{
				left: preview.screenX,
				top: preview.screenY,
				borderColor: `color-mix(in srgb, ${meta.color} 40%, transparent)`,
			}}
		>
			<div className='relative h-26 w-full bg-surface-lit'>
				{/* eslint-disable-next-line @next/next/no-img-element -- runtime R2 thumbnail via /r2 proxy */}
				<img
					src={preview.thumbUrl}
					alt=''
					draggable={false}
					className='h-full w-full object-cover'
					style={{ objectPosition: `${preview.headingU * 100}% 50%` }}
				/>
				<span
					className='absolute left-2 top-2 rounded px-1.5 py-0.5 text-[8px] font-semibold uppercase tracking-wider'
					style={{ color: groundVar(), background: meta.color }}
				>
					{meta.label}
				</span>
			</div>
			<div className='flex items-center justify-between gap-2 px-2.5 py-1.5'>
				<span className='min-w-0 truncate text-xs font-semibold text-ink'>
					{preview.name ?? "unnamed"}
				</span>
				<span className='shrink-0 text-[10px] tabular-nums text-ink-64'>
					{preview.dist < 100 ? `${preview.dist.toFixed(0)} m` : ""}
				</span>
			</div>
		</div>
	);
}
