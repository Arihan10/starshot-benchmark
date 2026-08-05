"use client";

import type { ObjectInspect } from "@/lib/orbit/types";
import { prettyLabel } from "./labels";

// The dwell inspection: a frame and a caption around the rectangle the ENGINE is
// drawing the orbiting object into (a scissored viewport of the main canvas, see
// renderInspect). Deliberately has no background of its own — anything opaque here
// would paint over the 3D underneath it. Pointer-transparent, so resting the cursor
// to summon it never blocks the click that follows.
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
