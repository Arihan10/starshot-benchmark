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
	warm = null,
	commitVia,
}: {
	scene?: Scene | null;
	source?: TourSource | null;
	/**
	 * A scene this panel will be asked for LATER — solved now, while the viewer is
	 * still reading the current round's result. See OrbitEngine.warmTour.
	 * Referentially stable, like `source`.
	 */
	warm?: TourSource | null;
	/**
	 * Hands the swap to a coordinator instead of running it on arrival, so two
	 * panels can be made to land in the same frame. See PairGate.
	 */
	commitVia?: (commit: () => void) => void;
}) {
	const hostRef = useRef<HTMLDivElement>(null);
	const engineRef = useRef<OrbitEngine | null>(null);
	// Bumped when the engine is constructed, so the load effect below re-runs
	// against an engine that exists. It used to call `engineRef.current?.loadTour`
	// with no such dependency: if the ref was empty when it ran — the host not yet
	// attached, or the engine effect not yet flushed — the optional chain swallowed
	// the call and NOTHING retried it. That panel then sat black for the rest of
	// the session while its sibling loaded normally, which is exactly the "one
	// scene doesn't load at all" shape.
	const [engineReady, setEngineReady] = useState(0);
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
		setEngineReady((n) => n + 1);
		return () => {
			engine.dispose();
			engineRef.current = null;
		};
	}, []);

	// `tourSource` is derived INSIDE the effect rather than during render: it
	// returns a fresh object every call, so using it as a dependency would reload
	// the scene on every render.
	//
	// `commitVia` is read through a ref rather than depended on: the gate changes
	// identity every round, and depending on it would restart the load of a scene
	// that is already on screen.
	// Synced in an effect declared BEFORE the load below, so by the time a load
	// runs the ref already holds this render's gate. Effects fire in declaration
	// order, which is the only thing making that true.
	const commitViaRef = useRef(commitVia);
	useEffect(() => {
		commitViaRef.current = commitVia;
	}, [commitVia]);
	useEffect(() => {
		const engine = engineRef.current;
		if (!engine) return;
		const src = source ?? (scene ? tourSource(scene) : null);
		if (!src) return;
		void engine.loadTour(src, (commit) => {
			const via = commitViaRef.current;
			if (via) via(commit);
			else commit();
		});
	}, [scene, source, engineReady]);

	// The NEXT scene, solved during the wait for it. Kept apart from the load above
	// so the two can never be confused: this one changes nothing on screen.
	useEffect(() => {
		if (!warm) return;
		engineRef.current?.warmTour(warm);
	}, [warm, engineReady]);

	// `inside` is handed back rather than reported onward from here: going in and
	// coming out have a running order now (the screen is taken after the panel has
	// finished opening, and given back before it closes), and the component that
	// owns the fullscreen controls is the one that can sequence it. See OrbitViewer.
	return { hostRef, engineRef, state, inside };
}
