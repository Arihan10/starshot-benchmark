"use client";

import { useEffect, useRef } from "react";
import OrbitViewer, { type OrbitViewerHandle } from "@/components/OrbitViewer";
import type { LocalCell } from "@/lib/localScenes";
import PctReadout from "./PctReadout";
import { shatter } from "./shatter";

export type Outcome = "won" | "lost" | "skipped" | null;

export const SOLO_TRANSITION_MS = 1000;

export const SOLO_EASING = "cubic-bezier(0.25,0.8,0.3,1)";

export default function ScenePanel({
	cell,
	outcome,
	share,
	align,
	role = "paired",
	onFocusedChange,
	warm = null,
	commitVia,
}: {
	cell: LocalCell;
	warm?: LocalCell["source"] | null;
	commitVia?: (commit: () => void) => void;
	outcome: Outcome;
	share: number;
	align: "left" | "right";
	role?: "paired" | "expanded" | "pushed";
	onFocusedChange?: (focused: boolean) => void;
}) {
	const stageRef = useRef<HTMLDivElement>(null);
	const fxRef = useRef<HTMLDivElement>(null);
	const viewerRef = useRef<OrbitViewerHandle>(null);
	const voted = outcome !== null;
	const keepSeam =
		role === "paired" && (outcome === null || outcome === "skipped");

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
			inert={role === "pushed"}
			className={`relative flex min-h-0 min-w-0 flex-col ${
				outcome === "won" ? "-translate-y-2" : ""
			}`}
			style={{
				flex: `0 0 ${role === "expanded" ? "100%" : "50%"}`,
				transitionProperty: "flex-basis, translate",
				transitionDuration: `${SOLO_TRANSITION_MS}ms, 620ms`,
				transitionTimingFunction: `${SOLO_EASING}, ${SOLO_EASING}`,
				["--arena-pct" as string]: "calc(var(--text-xl) * 1.85)",
			}}
		>
			<div ref={stageRef} className="relative isolate min-h-0 flex-1">
				{/* Keep the canvas off the centred seam. Half the rule sits in each
				    panel; without this the WebGL layer composites through those
				    pixels and the join reads thinner than `--seam-width`. */}
				<div
					className="absolute inset-0"
					style={
						keepSeam
							? align === "left"
								? { right: "calc(var(--seam-width, 3px) / 2)" }
								: { left: "calc(var(--seam-width, 3px) / 2)" }
							: undefined
					}
				>
					<OrbitViewer
						ref={viewerRef}
						source={cell.source}
						warm={warm}
						commitVia={commitVia}
						onFocusedChange={onFocusedChange}
						align={align}
						controls={!voted}
					/>
				</div>

				<div
					aria-hidden
					className="pointer-events-none absolute inset-0 z-4 duration-540"
					style={{
						backgroundColor:
							outcome === "lost"
								? "rgb(var(--ground-rgb) / 0.62)"
								: "rgb(var(--ground-rgb) / 0)",
						transitionProperty: "background-color",
					}}
				/>
				<div
					className="pointer-events-none absolute inset-0 z-10 transition-opacity duration-520 ease-[cubic-bezier(0.25,0.8,0.3,1)]"
					style={{
						opacity: outcome === "won" ? 1 : 0,
						// The frame states the win on its own. It used to carry a 34px and a
						// 90px bloom inside it as well, which lit the edges of the render
						// rather than the panel and read as the scene itself glowing.
						boxShadow: "inset 0 0 0 1px rgb(var(--mark-rgb) / 0.85)",
					}}
				/>
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
