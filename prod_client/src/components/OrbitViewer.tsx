"use client";

import { useEffect, useImperativeHandle, useRef, useState, type Ref } from "react";
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
	onFocusedChange?: (focused: boolean) => void;
}) {
	const rootRef = useRef<HTMLDivElement>(null);
	const { hostRef, engineRef, state } = useOrbitEngine({ scene, source, onFocusedChange });
	const { isFullscreen, supported, toggle } = useFullscreen(rootRef);

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
	const inside = mode === "interior";

	// "M" opens the chapters/search drawer. It has no button any more, so the key
	// is the whole affordance — and it has to ignore keystrokes aimed at the
	// drawer's own search field.
	useEffect(() => {
		if (!inside) return;
		const onKey = (e: KeyboardEvent) => {
			if (e.code !== "KeyM") return;
			if (document.activeElement instanceof HTMLInputElement) return;
			setDrawer((v) => !v);
		};
		window.addEventListener("keydown", onKey);
		return () => window.removeEventListener("keydown", onKey);
	}, [inside]);

	// Leaving the walkthrough leaves the drawer behind with it, so stepping back in
	// starts closed. Adjusted during render rather than in an effect — the same
	// pattern as the viewed level below — so no frame is ever painted with the
	// stale value.
	const [wasInside, setWasInside] = useState(inside);
	if (inside !== wasInside) {
		setWasInside(inside);
		if (!inside) setDrawer(false);
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

	return (
		<div ref={rootRef} className='group relative h-full w-full bg-black'>
			<div ref={hostRef} className='absolute inset-0' />

			{supported && <FullscreenButton isFullscreen={isFullscreen} onToggle={toggle} />}

			{/* --- inside the scene ------------------------------------------------ */}
			{(inside || mode === "peek") && minimap && (
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

			{inside && minimap && minimap.levels.length > 1 && (
				<FloorRail
					minimap={minimap}
					visited={state.visited}
					viewedLevel={viewedLevel}
					setViewedLevel={setViewedLevel}
					levelWord={state.levelWord}
					engine={engineRef}
				/>
			)}

			{inside && drawer && (
				<PlacesDrawer
					state={state}
					engine={engineRef}
					onClose={() => setDrawer(false)}
				/>
			)}

			{inside && state.arrival && (
				<ArrivalToast
					key={state.arrival.ts}
					arrival={state.arrival}
					trapped={state.trapped}
				/>
			)}

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
