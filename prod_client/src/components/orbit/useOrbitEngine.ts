"use client";

import { useEffect, useRef, useState } from "react";
import { OrbitEngine } from "@/lib/orbit/engine";
import { INITIAL_ORBIT_STATE, type OrbitState, type TourSource } from "@/lib/orbit/types";
import { tourSource, type Scene } from "@/lib/scenes";

/** Modes in which the camera has left the orbit and is inside the scene. */
const FOCUSED_MODES = new Set(["interior", "freefly", "transition", "peek"]);

/**
 * Owns one OrbitEngine: its lifetime, the scene it is showing, and the state it
 * publishes back.
 *
 * The engine is imperative and lives for as long as the panel does, so the only
 * thing React needs from it is the state stream — everything else is a method
 * call on the ref. Keeping that wiring here leaves the component free to be
 * nothing but layout.
 */
export function useOrbitEngine({
	scene = null,
	source = null,
	onFocusedChange,
}: {
	scene?: Scene | null;
	source?: TourSource | null;
	onFocusedChange?: (focused: boolean) => void;
}) {
	const hostRef = useRef<HTMLDivElement>(null);
	const engineRef = useRef<OrbitEngine | null>(null);
	const [state, setState] = useState<OrbitState>(INITIAL_ORBIT_STATE);

	// One engine per mount. Empty deps on purpose: a rebuilt engine would drop the
	// loaded scene and the camera with it, so the scene is swapped below instead.
	useEffect(() => {
		const host = hostRef.current;
		if (!host) return;
		const engine = new OrbitEngine(host, setState);
		engineRef.current = engine;
		return () => {
			engine.dispose();
			engineRef.current = null;
		};
	}, []);

	// `tourSource` is derived INSIDE the effect rather than during render: it
	// returns a fresh object every call, so using it as a dependency would reload
	// the scene on every render.
	useEffect(() => {
		const src = source ?? (scene ? tourSource(scene) : null);
		if (src) void engineRef.current?.loadTour(src);
	}, [scene, source]);

	const focused = FOCUSED_MODES.has(state.mode);
	useEffect(() => {
		onFocusedChange?.(focused);
	}, [focused, onFocusedChange]);

	return { hostRef, engineRef, state };
}
