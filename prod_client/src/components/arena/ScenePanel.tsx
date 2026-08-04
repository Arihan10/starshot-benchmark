"use client";

import { useEffect, useRef } from "react";
import OrbitViewer, { type OrbitViewerHandle } from "@/components/OrbitViewer";
import type { LocalCell } from "@/lib/localScenes";
import PctReadout from "./PctReadout";
import { shatter } from "./shatter";

/**
 * What the round did to this panel.
 *
 * `skipped` is a REVEAL WITHOUT A RESULT: declining to choose still ends the
 * round, so the models are named and the crowd's split is shown — but nothing
 * won and nothing lost, so neither the glow nor the shatter fires. Reading it as
 * `null` instead would have hidden the reveal entirely, and reading it as a loss
 * for both would have shattered two scenes nobody voted against.
 */
export type Outcome = "won" | "lost" | "skipped" | null;

/**
 * How long the row takes to re-form around the panel being entered or left.
 *
 * MATCHED TO THE CAMERA, not chosen for the layout. The engine flies into a scene
 * over 1100 ms and back out over 1000 ms (OrbitEngine.enter / exit), and both
 * journeys now announce themselves as they START — so this runs alongside the
 * flight rather than after it, and the panel opening reads as part of the same
 * movement as the dive. A snappier layout number would land the row early and
 * leave the camera still travelling into a frame that had stopped changing.
 */
export const SOLO_TRANSITION_MS = 1000;

// The row and the panels share one curve, or the two halves of a single movement
// would ease differently.
export const SOLO_EASING = "cubic-bezier(0.25,0.8,0.3,1)";

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
	role = "paired",
	onFocusedChange,
}: {
	cell: LocalCell;
	outcome: Outcome;
	/** Share of previous voters who picked THIS side. */
	share: number;
	align: "left" | "right";
	dividerRight?: boolean;
	/**
	 * How this panel is sharing the row.
	 *
	 * `expanded` — it is the one being toured, and takes the whole row.
	 * `pushed`   — the OTHER one is, so this keeps its size and is carried out of
	 *              frame by its sibling's growth. It KEEPS ITS SIZE on purpose: a
	 *              panel that shrank to nothing would squeeze its scene flat on the
	 *              way out, and a 3D view does not squash, it gets narrower — which
	 *              reads as the render breaking rather than as the panel leaving.
	 *
	 * Never unmounted, and never `display:none` either: the engine, its two GL
	 * contexts and the scene they hold would all be thrown away and rebuilt on the
	 * way back — a blank panel and a second reload for something the user only left
	 * for a moment.
	 */
	role?: "paired" | "expanded" | "pushed";
	/** Fires when the camera enters this scene, and again when it leaves. */
	onFocusedChange?: (focused: boolean) => void;
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
			// `inert` while it is on its way out: the panel is still laid out and
			// still rendering, so without it the leaving scene keeps taking clicks and
			// keyboard focus from off the edge of the screen.
			inert={role === "pushed"}
			className={`relative flex min-h-0 min-w-0 flex-col ${
				outcome === "won" ? "-translate-y-2" : ""
			} ${
				// THE SEAM. It is not a hairline between two cards — it is the join
				// down the middle of one picture, and the two scenes either side of it
				// are the comparison, so it is drawn as a deliberate edge rather than a
				// suggestion. Both scenes render on black and meet with no border of
				// their own, so at 10% white it read as a rendering artefact.
				//
				// The divider separates two panels. Once one of them owns the row there
				// is nothing to separate, and the line is left hanging down the edge of
				// the frame.
				dividerRight && role === "paired"
					? "md:border-r-2 md:border-white/70"
					: ""
			}`}
			style={{
				// THE SIZE IS THE ANIMATION. Both panels hold half the row and the toured
				// one grows to all of it — so the two together come to 150%, and the one
				// that is not growing is simply carried past the edge by the one that is.
				// No shrink, or the pushed panel would give its width back instead of
				// leaving. See the row in page.tsx for how the SECOND panel leaves.
				flex: `0 0 ${role === "expanded" ? "100%" : "50%"}`,
				// TWO CLOCKS. The size follows the camera's flight; the winner's lift is
				// a reaction to a vote and keeps its own quicker beat. One shared
				// duration would have to be wrong for one of them.
				//
				// The lift is named as `translate` because that is the property Tailwind's
				// `-translate-y-2` actually sets — `transform` is a different property and
				// transitioning it would leave the lift snapping into place.
				transitionProperty: "flex-basis, translate",
				transitionDuration: `${SOLO_TRANSITION_MS}ms, 620ms`,
				transitionTimingFunction: `${SOLO_EASING}, ${SOLO_EASING}`,
				// The percentage sizes off the panel, not the window, so it stays in
				// proportion however the row is split.
				["--arena-pct" as string]: "clamp(26px, min(34vh, 14vw), 168px)",
			}}
		>
			<div ref={stageRef} className="relative min-h-0 flex-1">
				<OrbitViewer
					ref={viewerRef}
					source={cell.source}
					onFocusedChange={onFocusedChange}
				/>

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
