"use client";

import { Fragment, useEffect, useRef } from "react";
import OrbitViewer, { type OrbitViewerHandle } from "@/components/OrbitViewer";
import { GROUND_LINE } from "@/lib/orbit/engine";
import type { LocalCell } from "@/lib/localScenes";
import { buildStep } from "./buildSequence";
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
	dividerRight,
	role = "paired",
	onFocusedChange,
	built = true,
	untuck = false,
	roundKey,
	warm = null,
	commitVia,
}: {
	cell: LocalCell;
	warm?: LocalCell["source"] | null;
	commitVia?: (commit: () => void) => void;
	outcome: Outcome;
	share: number;
	align: "left" | "right";
	dividerRight?: boolean;
	role?: "paired" | "expanded" | "pushed";
	onFocusedChange?: (focused: boolean) => void;
	built?: boolean;
	untuck?: boolean;
	roundKey?: string;
}) {
	const stageRef = useRef<HTMLDivElement>(null);
	const fxRef = useRef<HTMLDivElement>(null);
	const viewerRef = useRef<OrbitViewerHandle>(null);
	const voted = outcome !== null;
	const skipped = outcome === "skipped";

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
				{dividerRight && role === "paired" && (outcome === null || skipped) && (
					// eslint-disable-next-line react/jsx-key
					<Fragment key={roundKey}>
						<div
							aria-hidden
							className="pointer-events-none absolute right-0 z-10 hidden w-[3px] origin-top translate-x-1/2 bg-mark transition-[scale] md:block"
							style={{
								willChange: "scale",
								// The masthead's belly bottoms out on this panel's top edge, so
								// the seam picks up exactly where the moon's limb leaves off.
								top: 0,
								// Lands on the bar's top edge. The panel now runs to the foot
								// of the page, so that is the bar's drop from the bottom plus
								// the half-height its centre line sits above.
								bottom:
									"calc(var(--seam-drop, 0px) + var(--seam-break, 0px))",
								scale: built ? "1 1" : "1 0",
								...buildStep("ray", built),
								...(untuck
									? {
											animation: `seam-untuck ${buildStep("ray", true).transitionDuration} cubic-bezier(0.16,0.84,0.28,1) ${buildStep("ray", true).transitionDelay} backwards`,
										}
									: null),
							}}
						/>
					</Fragment>
				)}

				<div
					aria-hidden
					className="pointer-events-none absolute inset-0 z-1 mix-blend-screen"
					style={{
						background: [
							// #TODO approximate — should project the scene's ground plane.
							`linear-gradient(to top, rgba(148,163,184,0.05) 0%, rgba(148,163,184,0.018) ${(GROUND_LINE * 100).toFixed(1)}%, transparent 82%)`,
						].join(", "),
					}}
				/>

				<div className="absolute inset-0">
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
							outcome === "lost" ? "rgb(0 0 0 / 0.62)" : "rgb(0 0 0 / 0)",
						transitionProperty: "background-color",
					}}
				/>
				<div
					className="pointer-events-none absolute inset-0 z-10 transition-opacity duration-520 ease-[cubic-bezier(0.25,0.8,0.3,1)]"
					style={{
						opacity: outcome === "won" ? 1 : 0,
						boxShadow:
							"inset 0 0 0 1px rgb(var(--mark-rgb) / 0.85), inset 0 0 34px rgb(var(--mark-rgb) / 0.3), inset 0 0 90px rgb(var(--mark-rgb) / 0.16)",
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
