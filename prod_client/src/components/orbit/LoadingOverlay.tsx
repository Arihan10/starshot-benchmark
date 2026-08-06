"use client";

import type { OrbitState } from "@/lib/orbit/types";

export default function LoadingOverlay({
	overlay,
}: {
	overlay: NonNullable<OrbitState["overlay"]>;
}) {
	return (
		<div className='absolute inset-0 z-20 flex flex-col items-center justify-center gap-2.5 bg-ground/80'>
			{overlay.spinner && (
				<span className='h-5 w-5 animate-spin rounded-full border-2 border-mark-16 border-t-cyan-400' />
			)}
			<span className={`text-xs ${overlay.err ? "text-red-400" : "text-ink-64"}`}>
				{overlay.msg}
			</span>
		</div>
	);
}
