"use client";

import { Fragment, useEffect, useRef } from "react";
import OrbitViewer, { type OrbitViewerHandle } from "@/components/OrbitViewer";
import { GROUND_LINE } from "@/lib/orbit/engine";
import type { LocalCell } from "@/lib/localScenes";
import { buildStep } from "./buildSequence";
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
	built = true,
	untuck = false,
	roundKey,
	warm = null,
	commitVia,
}: {
	cell: LocalCell;
	/**
	 * The cell this side will show NEXT, solved during the countdown so pressing
	 * "next" is a swap rather than a load. See OrbitEngine.warmTour.
	 */
	warm?: LocalCell["source"] | null;
	/**
	 * Hands this side's swap to the round's pair gate, so both panels change in the
	 * same frame. See PairGate.
	 */
	commitVia?: (commit: () => void) => void;
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
	/**
	 * Whether the controls are built. False while a prompt is being written, and
	 * false for the first frame after mount so the page assembles itself on load.
	 *
	 * The seam runs from the disc down between the two builds, so retracting it
	 * toward its own origin is the one motion that reads as the moon taking the
	 * structure back rather than as a divider being switched off.
	 */
	built?: boolean;
	/**
	 * Whether this seam should come back out from behind the controls on arrival.
	 *
	 * Set for every round after the first: the previous one left the line tucked
	 * behind the vote bar, so the next one has to bring it back rather than have it
	 * appear already extended.
	 */
	untuck?: boolean;
	/**
	 * Identifies the round on screen.
	 *
	 * The panel now outlives the round — that is the point of it — so the seam can
	 * no longer rely on a fresh mount to replay its entrance. Keyed on this, the two
	 * runs are rebuilt when the round turns and nothing else is.
	 */
	roundKey?: string;
}) {
	const stageRef = useRef<HTMLDivElement>(null);
	const fxRef = useRef<HTMLDivElement>(null);
	const viewerRef = useRef<OrbitViewerHandle>(null);
	const voted = outcome !== null;
	// SKIPPING IS THE ONLY OUTCOME THAT KEEPS THE SEAM. Declining to choose leaves
	// the pair still standing as a pair, so the join between them is still true. A
	// win does not: the winner takes a lit edge all the way round its own frame, and
	// leaving the divider up would put two borders along the same join — one saying
	// something, the other left over from before the question was answered.
	const skipped = outcome === "skipped";

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
				// A STEP OF ITS OWN, above the scale's top. `--text-xl` is the heading
				// size, and this is not a heading — it is the one figure on the panel a
				// viewer reads from across a room, and at the same size as a title it
				// was competing with the model name rather than answering it. Held as a
				// multiple of the token so it still moves with the type scale.
				["--arena-pct" as string]: "calc(var(--text-xl) * 1.85)",
			}}
		>
			{/* `isolate` IS LOAD-BEARING, and its absence is why the beam rendered
			    grey on first load.
			    
			    The pool of light below uses `mix-blend-screen`, and a blended element
			    blends against every layer beneath it in its stacking context. This
			    div is `relative` with `z-index: auto`, which does NOT open a stacking
			    context — so the blending group escaped the panel entirely and the
			    compositor had to resolve that blend against the seam's own layer,
			    which is promoted while its scale animates. On a settled page it
			    resolved the way the z-indexes say it should; during the first load,
			    with two WebGL contexts still coming up and layers still being
			    assigned, it did not, and the beam came out at a fifth of its
			    brightness for the length of its travel.
			    
			    `isolation: isolate` opens the stacking context, which confines the
			    blend to this panel — what the wash was always meant to do — and makes
			    the seam's z-10 an ordering the compositor can honour on any frame
			    rather than one that happens to hold once everything has settled. */}
			<div ref={stageRef} className="relative isolate min-h-0 flex-1">
				{/* --- the shadows the chrome casts --------------------------------
				    The masthead floats over the builds at the top and the control
				    stack at the bottom, so a scene orbited toward either arrives
				    behind type with nothing between the two. These are what it
				    arrives into: the ground at full strength where the furniture
				    sits, gone by the inner edge of each band.

				    THEY LIVE IN HERE, and that is the whole reason this works. The
				    stage `isolate`s, so the seam's `z-10` is scoped to THIS box —
				    which means a fade rendered as a sibling of the panel cannot get
				    underneath it. From out there the panel is one `z-index: auto`
				    element and any positive z beats the lot, seam included; that is
				    exactly what happened, and why the ray kept coming out dimmed. In
				    here they are the seam's own siblings and z-5 loses to it
				    properly.

				    The top one is TWO layers because two things cast it: a linear for
				    the navbar's straight edge, and a radial for the moon, which hangs
				    into the page in the middle and so has to pull the dark down with
				    it. The bottom is one — there is no disc down there. */}
				<div
					aria-hidden
					className="pointer-events-none absolute inset-x-0 top-0 z-5 h-[clamp(150px,26vh,310px)]"
					style={{
						background: [
							"radial-gradient(ellipse 46% 96% at 50% -6%, rgb(var(--ground-rgb)) 0%, rgb(var(--ground-rgb) / 0.74) 44%, rgb(var(--ground-rgb) / 0.28) 72%, transparent 92%)",
							"linear-gradient(to bottom, rgb(var(--ground-rgb)) 0%, rgb(var(--ground-rgb) / 0.66) 26%, rgb(var(--ground-rgb) / 0.22) 58%, transparent 100%)",
						].join(", "),
					}}
				/>
				<div
					aria-hidden
					className="pointer-events-none absolute inset-x-0 bottom-0 z-5 h-[clamp(130px,22vh,260px)]"
					style={{
						background:
							"linear-gradient(to top, rgb(var(--ground-rgb)) 0%, rgb(var(--ground-rgb) / 0.72) 28%, rgb(var(--ground-rgb) / 0.26) 60%, transparent 100%)",
					}}
				/>
				{/* THE SEAM, fading DOWNWARD — away from the light rather than
				    across it, so it is brightest where it meets the top rule and gone
				    by the bottom of the frame.

				    The TOP rule is not here. It was, one per panel, and the gradient
				    then measured across each panel separately — so the line dimmed at
				    both ends of each half and the page got a dip at dead centre,
				    directly under the moon, which is the one place it should have been
				    brightest. It is drawn once across the whole row in page.tsx. */}
				{/* THE SEAM GOES WHEN THE ROUND IS DECIDED. The winner takes an inset
				    white glow all round its own frame, which IS an edge — a brighter and
				    more meaningful one than this — so leaving the divider up puts two
				    borders along the same join, one of them saying something and one of
				    them left over from before the question was answered. */}
				{dividerRight && role === "paired" && (outcome === null || skipped) && (
					// eslint-disable-next-line react/jsx-key
					// THE SEAM, IN TWO RUNS WITH THE CONTROLS IN THE BREAK.
					//
					// Solid white, three pixels, no fade. It is the join down the middle of
					// one picture and the two builds either side of it ARE the comparison,
					// so the edge between them is drawn like a decision — at one pixel it
					// read as a hairline artefact between two renders.
					//
					// The upper run stops at the top of the SKIP button and the lower one
					// starts again at the composer's own underline, so the vertical hands
					// the stroke to the horizontal and the controls sit INSIDE the line
					// rather than across it.
					//
					// BOTH SCALE FROM THEIR TOP EDGE, which is what makes this a beam: the
					// head stays where it was and the foot travels, so the upper run reads
					// as light leaving the moon and reaching down, and the lower one as it
					// carrying on past the composer. They are different parts of the same
					// sequence — see buildSequence — so the tail cannot arrive before the
					// thing it continues from.
					//
					// `scale`, NOT `transform`: Tailwind v4's scale utilities write the
					// standalone property, so a transition naming `transform` animates
					// nothing at all and the line would simply blink out.
					<Fragment key={roundKey}>
						<div
							aria-hidden
							className="pointer-events-none absolute top-0 right-0 z-10 hidden w-[3px] origin-top bg-mark transition-[scale] md:block"
							style={{
								// THE LAYER EXISTS BEFORE THE ANIMATION NEEDS IT.
								//
								// Without this the seam is promoted to its own compositing
								// layer at the moment its scale begins animating, and on first
								// load that handoff lands while two WebGL contexts are still
								// coming up. Measured, the beam painted at exactly 0.2 alpha
								// for the first part of its travel and at full strength after:
								// forced to pure red it drew rgb(51,0,0), which is 255 x 0.2,
								// while its own and every ancestor's computed opacity read 1.
								// Nothing in the styles was transparent — the compositor was
								// mid-handoff, and a half-composited layer is what grey was.
								//
								// Declared up front, the layer is already there on the first
								// frame and there is no handoff left to catch it in.
								willChange: "scale",
								// A SKIPPED ROUND DOES NOT MOVE THIS LINE. It runs on
								// behind the controls instead.
								//
								// It used to be unmounted outright, which is not an exit
								// but a missing frame. Animating it out was no better:
								// the beam receding is what ENTERING THE COMPOSER means —
								// the structure being pulled back into the moon — and
								// borrowing that motion here says the same thing about an
								// event that means the opposite. Nothing is being taken
								// back when a round is answered; the controls underneath
								// have simply become the result.
								//
								// So the line holds still and its END moves out of sight:
								// it stops at the bar's TOP while a question is open,
								// giving the beam a visible point to land on, and at the
								// bar's BOTTOM once one is answered, where the z-30
								// control stack covers it completely. Same line, same
								// place, no motion — it just no longer terminates
								// anywhere you can see.
								// IT TUCKS ON SKIP AND UNTUCKS ON THE NEXT ROUND.
								//
								// The line runs the full height of the controls once the
								// round is skipped, so it has somewhere to go: it
								// collapses to its own BOTTOM edge and the head travels
								// down behind the next-round button, which sits in a z-30
								// stack above this z-10 seam.
								//
								// ORIGIN BOTTOM, and that is the whole difference between
								// this exit and the composer's. Entering the composer
								// sends the beam back where it came from — origin top,
								// foot receding toward the moon. Skipping does not take
								// the structure back; the controls underneath have become
								// the result, so the way out is past them, downward.
								bottom: skipped
									? "var(--seam-under, 0px)"
									: "var(--seam-break, 0px)",
								// ORIGIN TOP ON THE WAY IN, ALWAYS — the head stays at the moon
								// and the foot travels down, which is what makes this a beam
								// rather than a bar being uncovered.
								//
								// A new round used to draw from the BOTTOM, reasoning that the
								// round before it left the line tucked behind the controls, so it
								// should come back out the way it went in. True of the geometry
								// and wrong on screen: growing upward from the vote bar over 400 ms
								// is not read as travel at all — it reads as the line simply being
								// there, which is exactly the complaint. Only a SKIP still uses the
								// bottom, and only to leave by it.
								transformOrigin: skipped ? "bottom" : "top",
								scale: skipped ? "1 0" : built ? "1 1" : "1 0",
								...buildStep("ray", built),
								// A DECISION DOES NOT QUEUE. The collapse chain holds the
								// ray until the beam has closed, which is right when the
								// composer is opening and the structure is being pulled
								// back in order. Answering a round is not that.
								...(skipped ? { transitionDelay: "0ms" } : null),
								// COMING BACK OUT. A fresh round mounts a fresh element,
								// so there is no previous value to transition from — the
								// untuck has to be an animation, not a transition, or the
								// line would simply be there.
								...(untuck
									? {
											animation: `seam-untuck ${buildStep("ray", true).transitionDuration} cubic-bezier(0.16,0.84,0.28,1) ${buildStep("ray", true).transitionDelay} backwards`,
										}
									: null),
							}}
						/>
						<div
							aria-hidden
							className="pointer-events-none absolute right-0 bottom-0 z-10 hidden h-[var(--seam-stop,0px)] w-[3px] origin-top bg-mark transition-[scale] md:block"
							style={{
								willChange: "scale",
								// IT TUCKS WITH THE RUN ABOVE IT. A skip collapsed the upper
								// run behind the controls and left this one standing — a
								// three-pixel stub below a bar with nothing above it, which
								// is the one state that reads as broken rather than as
								// answered. Same origin as its own build (top), so it
								// retracts back toward the rule it came out of.
								scale: skipped || !built ? "1 0" : "1 1",
								...buildStep("tail", built),
								// A DECISION DOES NOT QUEUE — see the run above. The build
								// chain holds this until the rule has opened, which is not a
								// wait that means anything on the way out.
								...(skipped ? { transitionDelay: "0ms" } : null),
								// REPLAYED, NOT RESTORED. A fresh round mounts a fresh
								// element, so `scale: 1 1` is its FIRST value and there is
								// nothing to transition from — the line was simply there. As
								// an animation it draws itself again, on its own beat in the
								// sequence, so the whole stroke replays rather than the
								// bottom half appearing already finished.
								...(untuck
									? {
											animation: `seam-untuck ${buildStep("tail", true).transitionDuration} cubic-bezier(0.16,0.84,0.28,1) ${buildStep("tail", true).transitionDelay} backwards`,
										}
									: null),
							}}
						/>
					</Fragment>
				)}



				{/* THE STAGE — the reason the pair reads as two objects rather than two
				    screenshots.

				    A build rendered on #000 with nothing under it is not lit, it is cut
				    out: there is no surface for it to be standing on, so the eye files
				    it as a picture pasted onto the page. This is the floor. It is a POOL
				    OF LIGHT rather than a contact shadow for the obvious reason that a
				    shadow on black has nothing to darken; on an unlit stage the way you
				    show where the ground is, is to light it.

				    Pinned to GROUND_LINE, imported from the engine rather than typed
				    again here — that constant is where the camera stands every build, so
				    the light lands under the thing standing on it by construction, at
				    any panel size, for any scene. Two numbers that had to agree would
				    eventually stop agreeing.

				    `screen` so it can only ever lighten. It sits above the canvas (the
				    viewer's own background is opaque, so there is no behind to be in),
				    and blending is what keeps that from mattering: black contributes
				    nothing, so the build is untouched and only the emptiness around its
				    base comes up off the page.

				    The wash under it is the same argument at panel scale — a floor is
				    brighter near the ground and falls off upward, and having that
				    gradient at all is what stops the surrounding black reading as a void
				    the object is suspended in. */}
				<div
					aria-hidden
					className="pointer-events-none absolute inset-0 z-1 mix-blend-screen"
					style={{
						background: [
// WIDE ENOUGH TO BE THE FLOOR, not a spotlight on it. It was 58% of
							// the panel, narrower than most builds now that BROWSE_MARGIN lets
							// them overhang the frame — so the pool sat under the middle of a
							// footprint that ran off both sides. At full width it reaches the
							// same edges the build does.
							//
							// #TODO APPROXIMATE. The honest version projects the scene's root
							// bbox to screen space and sizes this from THAT, so the light
							// matches each build's real footprint and sits on its true bottom
							// edge. The engine has the box and the camera, but this element
							// lives outside OrbitViewer so there is no path for it yet.
							// `frameOverview` fits the hull to the safe band, which is what
							// makes the panel a fair stand-in for the footprint meanwhile.
// THE POOL OF LIGHT IS IN THE SCENE NOW, not over it — a quad on the
							// ground plane in three.js (see OrbitEngine's stageGlow), so it
							// foreshortens with the camera and turns with the build instead of
							// sitting on the glass as a flat ellipse. What stays here is the
							// panel-scale wash below, which is a property of the PAGE — the floor
							// being brighter near the ground and falling off upward — and has no
							// business being in the 3D scene.
							`linear-gradient(to top, rgba(148,163,184,0.05) 0%, rgba(148,163,184,0.018) ${(GROUND_LINE * 100).toFixed(1)}%, transparent 82%)`,
						].join(", "),
					}}
				/>

				{/* THE LOSER RECEDES BY FILTERING THE VIEW, NOT ITS BACKDROP.
					
					    This was an overlay carrying `backdrop-filter`, and that is the
					    most expensive way to express it. A backdrop filter makes the
					    compositor snapshot everything beneath the element, run the
					    filter chain over it and composite the result — EVERY FRAME. The
					    thing beneath it here is a WebGL canvas that redraws
					    continuously, so the cost was paid on every one of them for the
					    rest of the round, and the scene it was dimming is exactly the
					    one that then looked like it had stopped orbiting.
					
					    Filtering the viewer itself is one layer processed once per
					    frame instead of a backdrop sampled and re-filtered per frame.
					    It also cannot touch the shards: they live in a sibling layer,
					    so the break stays full-strength over a dimmed scene, which is
					    what it did before. */}
				<div className="absolute inset-0">
					<OrbitViewer
						ref={viewerRef}
						source={cell.source}
						warm={warm}
						commitVia={commitVia}
						onFocusedChange={onFocusedChange}
					/>
				</div>

				{/* THE LOSER DIMS — OVER the scene, not around it.
				
				    This used to WRAP the viewer, and a parent's background-color paints
				    BEHIND its children: the wash was underneath two opaque canvases
				    that redraw every frame, so it was composited perfectly and never
				    visible. Nothing about the colour or the transition was wrong, which
				    is why it looked like the feature had been removed rather than
				    broken — the element was there, animating, doing nothing.
				
				    As a sibling above it, it covers what it is meant to cover. z-4:
				    over the viewer, which is positioned at auto, and under the chrome
				    shadows at z-5 and the seam and win-glow at z-10 — a losing panel
				    still gets the same edges as a winning one, just darker behind them.
				
				    A FLAT WASH, NOT A FILTER. `filter` here is a full-panel
				    post-process the compositor runs on EVERY frame, and the thing under
				    it never stops redrawing — so the cost ran for the whole round, on
				    the panel the reader is most likely watching come apart. Compositing
				    a translucent black over it is free by comparison and says the same
				    thing: this one is out. The desaturation was the expensive half of
				    the look and the least of what it communicated.
				
				    It also cannot touch the shards: they live in a sibling layer, so
				    the break stays full-strength over a dimmed scene. */}
				<div
					aria-hidden
					className="pointer-events-none absolute inset-0 z-4 duration-540"
					style={{
						backgroundColor:
							outcome === "lost" ? "rgb(0 0 0 / 0.62)" : "rgb(0 0 0 / 0)",
						transitionProperty: "background-color",
					}}
				/>
				{/* Lit from inside the frame rather than outlined: a border would read
				    as a selection control, a glow reads as the thing itself winning. */}
				<div
					className="pointer-events-none absolute inset-0 z-10 transition-opacity duration-520 ease-[cubic-bezier(0.25,0.8,0.3,1)]"
					style={{
						opacity: outcome === "won" ? 1 : 0,
						boxShadow:
							"inset 0 0 0 1px rgb(var(--mark-rgb) / 0.85), inset 0 0 34px rgb(var(--mark-rgb) / 0.3), inset 0 0 90px rgb(var(--mark-rgb) / 0.16)",
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
