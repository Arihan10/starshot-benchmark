"use client";

import {
	useCallback,
	useEffect,
	useImperativeHandle,
	useRef,
	useState,
	type Ref,
} from "react";
import type { TourSource } from "@/lib/orbit/types";
import type { Scene } from "@/lib/scenes";
import ArrivalToast from "./orbit/ArrivalToast";
import FloorRail from "./orbit/FloorRail";
import FullscreenButton from "./orbit/FullscreenButton";
import HoverCard from "./orbit/HoverCard";
import InspectFrame from "./orbit/InspectFrame";
import LoadingOverlay from "./orbit/LoadingOverlay";
import Minimap from "./orbit/Minimap";
import ObjectMenu from "./orbit/ObjectMenu";
import PlacesDrawer from "./orbit/PlacesDrawer";
import ReachPreview from "./orbit/ReachPreview";
import { useFullscreen } from "./orbit/useFullscreen";
import { useOrbitEngine } from "./orbit/useOrbitEngine";

/**
 * One scene, orbitable, with a walkthrough inside it.
 *
 * THE ORBIT VIEW CARRIES NO CHROME. A viewer in an A/B comparison is being judged
 * on what it renders, and every badge and button laid over it is something the eye
 * has to discount first — so the only control here is full screen, and everything
 * else is a gesture on the scene itself: drag to orbit, scroll to zoom, click to
 * step inside, Escape to come back out.
 *
 * The walkthrough keeps its navigation aids (minimap, floor rail, previews),
 * because once you are inside a generated building "where am I and what is behind
 * me" stops being answerable from the picture alone. Each one is a sibling under
 * ./orbit and is mounted by exactly one line below, so any of them can be dropped
 * without unpicking anything else.
 *
 * This component is deliberately only lifecycle + layout. The engine wiring lives
 * in useOrbitEngine; every visual piece owns its own behaviour.
 */
export type OrbitViewerHandle = {
	/**
	 * A still of what the panel is currently showing, both layers flattened.
	 * Resolves on the next drawn frame — see OrbitEngine.capture for why it cannot
	 * be synchronous — or null if there is nothing to draw.
	 */
	capture: () => Promise<HTMLCanvasElement | null>;
};

export default function OrbitViewer({
	scene = null,
	source = null,
	warm = null,
	commitVia,
	onFocusedChange,
	ref,
}: {
	ref?: Ref<OrbitViewerHandle>;
	// The published path: a catalog row, whose assets resolve through /r2.
	scene?: Scene | null;
	// The local path: a source the caller assembled itself (see lib/localScenes).
	// Takes precedence, so an explicit override is never silently ignored.
	// Must be referentially STABLE — its identity is what triggers a reload, so a
	// source rebuilt inline on each render would restart the scene every frame.
	source?: TourSource | null;
	/** A source to solve ahead of time — see OrbitEngine.warmTour. Also stable. */
	warm?: TourSource | null;
	/** Hands this panel's swap to a coordinator. See PairGate. */
	commitVia?: (commit: () => void) => void;
	onFocusedChange?: (focused: boolean) => void;
}) {
	const rootRef = useRef<HTMLDivElement>(null);
	const { hostRef, engineRef, state, inside } = useOrbitEngine({
		scene,
		source,
		warm,
		commitVia,
	});
	const {
		isFullscreen,
		supported,
		exit: exitFullscreen,
		toggle,
	} = useFullscreen(rootRef);

	// The one thing the outside can ask the engine to DO. Everything else the
	// viewer exposes is declarative (a scene in, focus out); a still of the current
	// frame is a request, not a state, so it comes through a handle instead.
	useImperativeHandle(
		ref,
		() => ({
			capture: () => engineRef.current?.capture() ?? Promise.resolve(null),
		}),
		[engineRef],
	);
	const [drawer, setDrawer] = useState(false);

	const { mode, minimap, overlay } = state;
	const landed = mode === "interior";

	// ENTERING A SCENE DOES NOT TAKE THE SCREEN.
	//
	// It used to: arriving inside asked for browser fullscreen, so stepping into a
	// build swapped the whole window, hid the browser chrome and left the viewer to
	// find their way out of a mode they had not asked for. Walking through a scene
	// does not need the screen — it needs the SITE out of the way, which is a
	// different thing and is now what happens (see page.tsx, which clears its own
	// furniture while a panel is toured).
	//
	// The ⛶ control is still there for anyone who does want it, and because nothing
	// enters fullscreen behind the viewer's back, Escape reaches the document again
	// and the engine's own handler gets it — which is what the block below had to
	// work around when the browser was consuming it.

	// LEAVING FULLSCREEN LEAVES THE SCENE — because the browser will not let us do
	// this any other way.
	//
	// Escape is the way out of the walkthrough, and the engine listens for it. But
	// while a page is fullscreen, Chrome CONSUMES the Escape that exits fullscreen
	// and never dispatches it to the document: the keydown handler is not called,
	// cannot be called, and no amount of work in it will help. The first press
	// therefore only shrank the window, and a viewer who meant "get me out" was
	// still standing in the scene wondering why nothing happened.
	//
	// So the exit is taken from the fullscreen state itself, which is the one signal
	// that press does produce. `leftDeliberately` is what keeps the ⛶ control
	// meaning what it says: pressing it is a request about the WINDOW, and must not
	// also throw away the place you are standing in.
	const leftDeliberately = useRef(false);
	const toggleFullscreen = useCallback(() => {
		if (isFullscreen) leftDeliberately.current = true;
		toggle();
	}, [isFullscreen, toggle]);

	// A TRANSITION out of fullscreen, not merely the absence of it. Entering a scene
	// sets `inside` a full second before the screen is taken, so a rule that read
	// "not fullscreen while inside" would fire on the way IN and ask the engine to
	// leave the moment the dive began. (It would be refused — `exit` wants the
	// walkthrough to have landed — but a correctness that rests on someone else's
	// guard is one edit away from not holding.) Only a viewer who HAD the screen
	// can hand it back.
	const hadScreen = useRef(false);
	useEffect(() => {
		if (isFullscreen) {
			hadScreen.current = true;
			return;
		}
		if (!inside || !hadScreen.current) return;
		hadScreen.current = false;
		if (leftDeliberately.current) {
			leftDeliberately.current = false;
			return;
		}
		engineRef.current?.exit();
	}, [isFullscreen, inside, engineRef]);

	// Coming out, the screen goes back FIRST: the parent is not told the camera has
	// left until it has, so whatever the parent does about it — the comparison page
	// re-forms its row — happens on a page that is already its normal size again.
	useEffect(() => {
		if (inside) {
			onFocusedChange?.(true);
			return;
		}
		let live = true;
		void exitFullscreen().then(() => {
			if (live) onFocusedChange?.(false);
		});
		return () => {
			live = false;
		};
	}, [inside, onFocusedChange, exitFullscreen]);

	// "M" opens the chapters/search drawer. It has no button any more, so the key
	// is the whole affordance — and it has to ignore keystrokes aimed at the
	// drawer's own search field.
	useEffect(() => {
		if (!landed) return;
		const onKey = (e: KeyboardEvent) => {
			if (e.code !== "KeyM") return;
			if (document.activeElement instanceof HTMLInputElement) return;
			setDrawer((v) => !v);
		};
		window.addEventListener("keydown", onKey);
		return () => window.removeEventListener("keydown", onKey);
	}, [landed]);

	// Leaving the walkthrough leaves the drawer behind with it, so stepping back in
	// starts closed. Adjusted during render rather than in an effect — the same
	// pattern as the viewed level below — so no frame is ever painted with the
	// stale value.
	const [wasInside, setWasInside] = useState(landed);
	if (landed !== wasInside) {
		setWasInside(landed);
		if (!landed) setDrawer(false);
	}

	// Which storey's plan the minimap is showing. It lives here because two
	// siblings share it: hovering a floor on the rail previews its plan on the map.
	//
	// Deliberately NOT reset when the pointer leaves the rail — the two sit apart,
	// so moving from one to the other would snap the plan back before you reached
	// it. A previewed floor sticks until you preview another or arrive somewhere,
	// and the map's own "you are on N" badge covers the mismatch.
	const currentLevel = minimap?.currentLevel ?? -1;
	const [viewedLevel, setViewedLevel] = useState(0);
	const [prevLevel, setPrevLevel] = useState(-1);
	if (currentLevel !== prevLevel) {
		setPrevLevel(currentLevel);
		setViewedLevel(currentLevel);
	}

	// `isolate` KEEPS THIS COMPONENT'S LAYERING INSIDE THIS COMPONENT.
	//
	// Without it the root is `relative` with `z-index: auto`, which creates NO
	// stacking context — so every z-index used in here competes directly with the
	// panel's own siblings instead of staying local. The loading veil below is
	// `z-20`; the seam the arena draws down the panel edge is `z-10`. The veil
	// therefore covered the beam, and being black at 0.8 alpha it rendered it at
	// exactly 1 - 0.8. Measured, the beam painted 47 against its own 237, and
	// forced to pure red it drew rgb(51,0,0) — while its own opacity and every
	// ancestor's read 1, with no filter or blend anywhere in the chain. Nothing
	// about the beam was wrong, which is why nothing done to the beam fixed it.
	//
	// It only ever showed on a cold load, because that is the only time the veil is
	// up, and it stopped at exactly the moment the veil came down.
	return (
		<div
			ref={rootRef}
			// `pointer-events-auto`, ASSERTED RATHER THAN ASSUMED.
			//
			// The viewer used to be rendered INSIDE the panel's dim layer, which is
			// `pointer-events-none` so its tint cannot swallow clicks — and
			// `pointer-events` INHERITS, so the viewer, its host and both canvases
			// silently inherited `none`. The scene stopped taking drag, wheel and click
			// while looking completely normal, because nothing about a canvas looks
			// different when it is not listening.
			//
			// The dim is a sibling above the viewer now (see ScenePanel), so the
			// inheritance is gone. This stays as a floor: whatever a caller wraps this
			// component in, the layer that owns the input says so itself.
			className='group relative isolate h-full w-full bg-ground pointer-events-auto'
		>
			<div ref={hostRef} className='absolute inset-0' />

			{supported && (
				<FullscreenButton isFullscreen={isFullscreen} onToggle={toggleFullscreen} />
			)}

			{/* --- inside the scene ------------------------------------------------
			    The walkthrough's aids, faded out as ONE GROUP the moment the camera
			    starts leaving.

			    They cannot fade themselves, and the reason is worth knowing: each is
			    mounted on `landed`, which comes from the state stream, and that stream
			    is deliberately frozen for the whole of a transition (OrbitEngine.emit).
			    So on the way out they held their last frame — a map of a place the
			    camera was already flying away from — for the full second of the
			    flight, and then vanished between one frame and the next when the state
			    finally moved. That is the stall. `inside` is the channel that is NOT
			    frozen (it flips as the flight begins), so it is what the fade listens
			    to; the unmount still happens on `landed`, a beat later, by which time
			    there has been nothing to see for 700 ms.

			    A plain, unpositioned wrapper. Opacity does not create a containing
			    block, so every child still positions against the viewer root exactly
			    as it did standing alone — the group can be faded without any of them
			    knowing they are in one. */}
			<div
				className={`transition-opacity duration-300 ease-out ${
					// Not clickable on the way out: opacity is not hit-testing, and a map
					// you can still travel with while it dissolves is a map that can send
					// you back into a scene you just left.
					inside ? "opacity-100" : "pointer-events-none opacity-0"
				}`}
			>
				{(landed || mode === "peek") && minimap && (
					<div className='absolute left-4 top-4 z-20'>
						<Minimap
							minimap={minimap}
							currentIndex={state.currentIndex}
							viewedLevel={viewedLevel}
							levelWord={state.levelWord}
							engine={engineRef}
						/>
					</div>
				)}

				{landed && minimap && minimap.levels.length > 1 && (
					<FloorRail
						minimap={minimap}
						visited={state.visited}
						viewedLevel={viewedLevel}
						setViewedLevel={setViewedLevel}
						levelWord={state.levelWord}
						engine={engineRef}
					/>
				)}

				{landed && drawer && (
					<PlacesDrawer
						state={state}
						engine={engineRef}
						onClose={() => setDrawer(false)}
					/>
				)}

				{landed && state.arrival && (
					<ArrivalToast
						key={state.arrival.ts}
						arrival={state.arrival}
						trapped={state.trapped}
					/>
				)}
			</div>

			{/* --- transient feedback ---------------------------------------------- */}
			{/* Always mounted: it fades and re-targets in place, so re-aiming across a
			    doorway edge can never blink the window out and back. */}
			<ReachPreview preview={state.reachPreview} levelWord={state.levelWord} />
			{state.preview && <HoverCard preview={state.preview} />}
			{state.inspect && <InspectFrame inspect={state.inspect} />}
			{state.contextMenu && <ObjectMenu menu={state.contextMenu} engine={engineRef} />}
			{overlay && <LoadingOverlay overlay={overlay} />}
		</div>
	);
}
