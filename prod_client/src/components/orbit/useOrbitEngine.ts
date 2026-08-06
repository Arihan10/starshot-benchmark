"use client";

import { useEffect, useRef, useState } from "react";
import { OrbitEngine } from "@/lib/orbit/engine";
import { INITIAL_ORBIT_STATE, type OrbitState, type TourSource } from "@/lib/orbit/types";
import { tourSource, type Scene } from "@/lib/scenes";

export function useOrbitEngine({
	scene = null,
	source = null,
	warm = null,
	commitVia,
}: {
	scene?: Scene | null;
	source?: TourSource | null;
	warm?: TourSource | null;
	commitVia?: (commit: () => void) => void;
}) {
	const hostRef = useRef<HTMLDivElement>(null);
	const engineRef = useRef<OrbitEngine | null>(null);
	const [engineReady, setEngineReady] = useState(0);
	const [state, setState] = useState<OrbitState>(INITIAL_ORBIT_STATE);
	const [inside, setInside] = useState(false);

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

	useEffect(() => {
		if (!warm) return;
		engineRef.current?.warmTour(warm);
	}, [warm, engineReady]);

	return { hostRef, engineRef, state, inside };
}
