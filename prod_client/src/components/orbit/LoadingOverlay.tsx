"use client";

import type { OrbitState } from "@/lib/orbit/types";

/** The engine's own loading / failure notice, covering the canvas while it applies. */
export default function LoadingOverlay({
	overlay,
}: {
	overlay: NonNullable<OrbitState["overlay"]>;
}) {
	return (
		<div className='absolute inset-0 z-20 flex flex-col items-center justify-center gap-2.5 bg-[#0c0d10]/80'>
			{overlay.spinner && (
				<span className='h-5 w-5 animate-spin rounded-full border-2 border-white/15 border-t-cyan-400' />
			)}
			<span className={`text-xs ${overlay.err ? "text-red-400" : "text-neutral-300"}`}>
				{overlay.msg}
			</span>
		</div>
	);
}
