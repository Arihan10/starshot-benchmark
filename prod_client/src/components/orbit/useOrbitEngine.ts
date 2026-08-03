"use client";

import { useEffect, useRef, useState } from "react";
import { OrbitEngine } from "@/lib/orbit/engine";
import { INITIAL_ORBIT_STATE, type OrbitState, type TourSource } from "@/lib/orbit/types";
import { tourSource, type Scene } from "@/lib/scenes";


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
}: {
	scene?: Scene | null;
	source?: TourSource | null;
}) {
	const hostRef = useRef<HTMLDivElement>(null);
	const engineRef = useRef<OrbitEngine | null>(null);
	const [state, setState] = useState<OrbitState>(INITIAL_ORBIT_STATE);
	// Kept apart from `state` deliberately. Whether the camera is in the scene has
	// to be known DURING a flight, and the state stream is frozen for exactly that
	// stretch (OrbitEngine.emit returns early while transitioning), so reading it
	// off `state.mode` could only ever report the two arrivals — which is a beat
	// late at both ends. The engine reports this one the moment each journey
	// starts.
	const [inside, setInside] = useState(false);

	// One engine per mount. Empty deps on purpose: a rebuilt engine would drop the
	// loaded scene and the camera with it, so the scene is swapped below instead.
	useEffect(() => {
		const host = hostRef.current;
		if (!host) return;
		const engine = new OrbitEngine(host, setState, undefined, setInside);
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

	// `inside` is handed back rather than reported onward from here: going in and
	// coming out have a running order now (the screen is taken after the panel has
	// finished opening, and given back before it closes), and the component that
	// owns the fullscreen controls is the one that can sequence it. See OrbitViewer.
	return { hostRef, engineRef, state, inside };
}
