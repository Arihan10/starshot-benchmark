"use client";

import { useEffect, useRef } from "react";
import OrbitViewer, { type OrbitViewerHandle } from "@/components/OrbitViewer";
import type { LocalCell } from "@/lib/localScenes";
import PctReadout from "./PctReadout";
import { shatter } from "./shatter";

export type Outcome = "won" | "lost" | null;

/**
 * One side of the comparison: the scene, and everything that happens to it when
 * the vote lands.
 *
 * The panel owns its own reaction. The page decides WHO won; how a winner is
 * crowned and how a loser comes apart is this component's business, which keeps
 * the page a description of the round rather than a pile of animation state.
 *
 * The controls are NOT here. The vote button and the reveal card live in one bar
 * spanning both panels — see VoteBar — because after the vote the two cards have
 * to flank a shared "next pair" control, and a footer owned per-panel has no way
 * to put anything between them.
 */
export default function ScenePanel({
	cell,
	outcome,
	share,
	align,
	dividerRight,
}: {
	cell: LocalCell;
	outcome: Outcome;
	/** Share of previous voters who picked THIS side. */
	share: number;
	align: "left" | "right";
	dividerRight?: boolean;
}) {
	const stageRef = useRef<HTMLDivElement>(null);
	const fxRef = useRef<HTMLDivElement>(null);
	const viewerRef = useRef<OrbitViewerHandle>(null);
	const voted = outcome !== null;

	// The loser shatters. The snapshot has to be taken BEFORE the dim lands, and
	// it resolves a frame later (the engine can only read its own canvases inside
	// the frame that drew them), so this is async by necessity — and guarded, since
	// a re-vote or unmount can land while that frame is still pending.
	useEffect(() => {
		if (outcome !== "lost") return;
		const fx = fxRef.current;
		const stage = stageRef.current;
		if (!fx || !stage) return;

		let live = true;
		let cancel = () => {};
		void viewerRef.current?.capture().then((snapshot) => {
			if (!live) return;
			cancel = shatter(fx, snapshot, stage.clientWidth, stage.clientHeight);
		});
		return () => {
			live = false;
			cancel();
		};
	}, [outcome]);

	return (
		<section
			className={`relative flex min-h-0 min-w-0 flex-1 flex-col transition-transform duration-620 ease-[cubic-bezier(0.25,0.8,0.3,1)] ${
				outcome === "won" ? "-translate-y-2" : ""
			} ${dividerRight ? "md:border-r md:border-white/10" : ""}`}
			// The percentage sizes off the panel, not the window, so it stays in
			// proportion however the row is split.
			style={{ ["--arena-pct" as string]: "clamp(26px, min(34vh, 14vw), 168px)" }}
		>
			<div ref={stageRef} className="relative min-h-0 flex-1">
				<OrbitViewer ref={viewerRef} source={cell.source} />

				{/* Desaturates and darkens whatever is under it, so the losing scene
				    recedes without being covered — an opaque wash would hide the very
				    thing the shards are made of. */}
				<div
					className="pointer-events-none absolute inset-0 z-10 transition-opacity duration-540"
					style={{
						opacity: outcome === "lost" ? 1 : 0,
						backdropFilter: "grayscale(1) brightness(0.42) contrast(0.95)",
						WebkitBackdropFilter: "grayscale(1) brightness(0.42) contrast(0.95)",
					}}
				/>
				{/* Lit from inside the frame rather than outlined: a border would read
				    as a selection control, a glow reads as the thing itself winning. */}
				<div
					className="pointer-events-none absolute inset-0 z-10 transition-opacity duration-520 ease-[cubic-bezier(0.25,0.8,0.3,1)]"
					style={{
						opacity: outcome === "won" ? 1 : 0,
						boxShadow:
							"inset 0 0 0 1px rgba(237,237,237,0.85), inset 0 0 34px rgba(237,237,237,0.3), inset 0 0 90px rgba(237,237,237,0.16)",
					}}
				/>
				{/* Shards live here, above both treatments. */}
				<div
					ref={fxRef}
					data-arena-fx
					className="pointer-events-none absolute inset-0 z-20 overflow-hidden"
				/>

				{voted && <PctReadout share={share} align={align} />}
			</div>
		</section>
	);
}
