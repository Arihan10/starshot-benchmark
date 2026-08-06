"use client";

import type { ObjectInspect } from "@/lib/orbit/types";
import { prettyLabel } from "./labels";

export default function InspectFrame({ inspect }: { inspect: ObjectInspect }) {
	return (
		<div
			className='pointer-events-none fixed z-30'
			style={{ left: inspect.x, top: inspect.y, width: inspect.w, height: inspect.h }}
		>
			<div className='h-full w-full rounded-lg border border-mark-16 shadow-2xl' />
			<div className='absolute inset-x-0 -bottom-6 truncate rounded bg-ground/75 px-2 py-1 text-center text-[10px] font-medium text-ink backdrop-blur'>
				{prettyLabel(inspect.label)}
			</div>
		</div>
	);
}
