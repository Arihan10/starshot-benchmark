"use client";

import {
	useEffect,
	useImperativeHandle,
	useState,
	type Ref,
} from "react";
import type { TourSource } from "@/lib/orbit/types";
import type { Scene } from "@/lib/scenes";
import ArrivalToast from "./orbit/ArrivalToast";
import FloorRail from "./orbit/FloorRail";
import HoverCard from "./orbit/HoverCard";
import InspectFrame from "./orbit/InspectFrame";
import LoadingOverlay from "./orbit/LoadingOverlay";
import Minimap from "./orbit/Minimap";
import ObjectMenu from "./orbit/ObjectMenu";
import PlacesDrawer from "./orbit/PlacesDrawer";
import ReachPreview from "./orbit/ReachPreview";
import { useOrbitEngine } from "./orbit/useOrbitEngine";

export type OrbitViewerHandle = {
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
	scene?: Scene | null;
	source?: TourSource | null;
	warm?: TourSource | null;
	commitVia?: (commit: () => void) => void;
	onFocusedChange?: (focused: boolean) => void;
}) {
	const { hostRef, engineRef, state, inside } = useOrbitEngine({
		scene,
		source,
		warm,
		commitVia,
	});

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

	useEffect(() => {
		onFocusedChange?.(inside);
	}, [inside, onFocusedChange]);

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

	const [wasInside, setWasInside] = useState(landed);
	if (landed !== wasInside) {
		setWasInside(landed);
		if (!landed) setDrawer(false);
	}

	const currentLevel = minimap?.currentLevel ?? -1;
	const [viewedLevel, setViewedLevel] = useState(0);
	const [prevLevel, setPrevLevel] = useState(-1);
	if (currentLevel !== prevLevel) {
		setPrevLevel(currentLevel);
		setViewedLevel(currentLevel);
	}

	return (
		<div className="relative h-full w-full">
			<div className="group relative isolate h-full w-full pointer-events-auto">
			<div ref={hostRef} className="absolute inset-0" />

			<div
				className={`transition-opacity duration-300 ease-out ${
					inside ? "opacity-100" : "pointer-events-none opacity-0"
				}`}
			>
				{(landed || mode === "peek") && minimap && (
					<div className="absolute left-4 top-4 z-20">
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

			<ReachPreview preview={state.reachPreview} levelWord={state.levelWord} />
			{state.preview && <HoverCard preview={state.preview} />}
			{state.inspect && <InspectFrame inspect={state.inspect} />}
			{state.contextMenu && <ObjectMenu menu={state.contextMenu} engine={engineRef} />}
			{/* Always mounted: it owns its own exit, and unmounting it on the frame
			    the scene commits is what made the cover vanish instead of lift. */}
			<LoadingOverlay overlay={overlay} />
			</div>
		</div>
	);
}
