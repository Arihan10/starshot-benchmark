"use client";

import { useEffect, useState } from "react";
import type { OrbitState } from "@/lib/orbit/types";

const ENTER_MS = 240;

// LONGER GOING OUT THAN COMING IN. The scene lands under the veil in one frame,
// and a symmetric fade makes that arrival read as a cut; the slower lift lets the
// geometry settle before the last of the cover is gone.
const EXIT_MS = 420;

// Read as tiers of a building going up. Widths and delays are the whole design:
// a common left edge so they stack against one wall, and a shorter course each
// time so the silhouette steps back as it rises.
const TIERS = [
	{ width: "100%", delay: 0 },
	{ width: "74%", delay: 150 },
	{ width: "46%", delay: 300 },
];

export default function LoadingOverlay({
	overlay,
}: {
	overlay: OrbitState["overlay"];
}) {
	// The overlay it was last asked to show, kept after that ask is withdrawn so
	// there is something to fade OUT. Without it the veil unmounts on the frame
	// the scene commits and the cover vanishes rather than lifting.
	const [held, setHeld] = useState(overlay);
	const [entered, setEntered] = useState(false);

	useEffect(() => {
		if (overlay) {
			setHeld(overlay);
			return;
		}
		const done = window.setTimeout(() => setHeld(null), EXIT_MS);
		return () => window.clearTimeout(done);
	}, [overlay]);

	useEffect(() => {
		const frame = requestAnimationFrame(() => setEntered(true));
		return () => cancelAnimationFrame(frame);
	}, []);

	if (!held) return null;

	return (
		<div
			data-loader
			role="status"
			aria-live="polite"
			className="pointer-events-none absolute inset-0 z-20 flex flex-col items-center justify-center gap-md bg-ground/80"
			style={{
				opacity: entered && overlay ? 1 : 0,
				transition: `opacity ${overlay ? ENTER_MS : EXIT_MS}ms ease-out`,
			}}
		>
			{held.spinner && (
				<div className="flex w-[104px] flex-col-reverse items-start gap-[5px]">
					{TIERS.map((tier) => (
						<span
							key={tier.width}
							aria-hidden
							className="h-[3px] origin-left bg-mark"
							style={{
								width: tier.width,
								animation: `lay-in 1.6s var(--ease-out-soft) ${tier.delay}ms infinite`,
							}}
						/>
					))}
				</div>
			)}

			<span
				key={held.msg}
				className={`font-label animate-[veil-in_260ms_ease-out_both] text-2xs ${
					held.err ? "text-fall" : "text-ink-40"
				}`}
			>
				{held.msg}
			</span>
		</div>
	);
}
