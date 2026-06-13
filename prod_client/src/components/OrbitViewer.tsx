"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { OrbitEngine } from "@/lib/orbit/engine";
import { INITIAL_ORBIT_STATE, type OrbitState } from "@/lib/orbit/types";
import { panoFileUrls, proxyFileUrl, scenePreviewUrl, tourManifestUrl } from "@/lib/r2";

export default function OrbitViewer() {
	const hostRef = useRef<HTMLDivElement>(null);
	const engineRef = useRef<OrbitEngine | null>(null);
	const [state, setState] = useState<OrbitState>(INITIAL_ORBIT_STATE);
	const [holding, setHolding] = useState(false);

	// Mount the imperative engine once and load the scene from R2: the dollhouse
	// GLB plus the optional capture-tour manifest (positions + proxy). Dispose
	// fully on unmount so React StrictMode's double-invoke leaves nothing leaked.
	useEffect(() => {
		const host = hostRef.current;
		if (!host) return;
		const engine = new OrbitEngine(host, setState, setHolding);
		engineRef.current = engine;
		void engine.loadTour({
			dollhouseUrl: scenePreviewUrl(),
			manifestUrl: tourManifestUrl(),
			resolvePano: panoFileUrls,
			resolveProxy: proxyFileUrl,
		});
		return () => {
			engine.dispose();
			engineRef.current = null;
		};
	}, []);

	const { mode, overlay } = state;
	const hud = hudContent(state);

	return (
		<div className="relative flex-1">
			<div ref={hostRef} className="absolute inset-0 bg-[#0c0d10]" />

			{mode !== "empty" && mode !== "loading" && (
				<div className="pointer-events-none absolute left-4 top-4 z-10 text-[10px] uppercase tracking-wider text-neutral-400 [&_strong]:font-semibold [&_strong]:text-cyan-200">
					{modeLabel(mode)}
				</div>
			)}

			<div className="absolute right-4 top-4 z-10 flex gap-2">
				{mode === "overview" && (
					<button
						type="button"
						disabled={state.panoCount === 0}
						onClick={() => engineRef.current?.enter()}
						className="rounded-md border border-cyan-500/40 bg-cyan-500/10 px-3 py-2 text-xs text-cyan-100 transition hover:border-cyan-400 hover:bg-cyan-500/20 hover:text-white disabled:cursor-default disabled:opacity-40"
					>
						enter interior ▸
					</button>
				)}
				{mode === "interior" && (
					<button
						type="button"
						disabled={state.busy}
						onClick={() => engineRef.current?.exit()}
						className="rounded-md border border-white/15 bg-black/50 px-3 py-2 text-xs text-neutral-200 backdrop-blur transition hover:border-white/30 hover:text-white disabled:cursor-default disabled:opacity-40"
					>
						◂ overview
					</button>
				)}
				{(mode === "interior" || mode === "peek") && (
					<button
						type="button"
						title="Hold to zoom out to the dollhouse and mark where you are"
						onPointerDown={(e) => {
							e.preventDefault();
							engineRef.current?.peekDown();
						}}
						onPointerUp={() => engineRef.current?.peekUp()}
						className={`rounded-md border px-3 py-2 text-xs transition ${
							holding
								? "border-red-400 bg-red-500/30 text-white"
								: "border-red-500/40 bg-red-500/10 text-red-200 hover:border-red-400/70 hover:text-red-100"
						}`}
					>
						⤢ hold to locate
					</button>
				)}
			</div>

			{hud && (
				<div className="pointer-events-none absolute bottom-4 left-4 z-10 max-w-[60%] rounded-md border border-white/10 bg-black/60 px-2.5 py-1.5 text-[11px] text-neutral-300 backdrop-blur [&_strong]:font-semibold [&_strong]:text-cyan-200">
					{hud}
				</div>
			)}

			{overlay && (
				<div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-2.5 bg-[#0c0d10]/80">
					{overlay.spinner && (
						<span className="h-5 w-5 animate-spin rounded-full border-2 border-white/15 border-t-cyan-400" />
					)}
					<span className={`text-xs ${overlay.err ? "text-red-400" : "text-neutral-300"}`}>
						{overlay.msg}
					</span>
				</div>
			)}
		</div>
	);
}

function modeLabel(mode: OrbitState["mode"]): ReactNode {
	if (mode === "overview") return <><strong>dollhouse</strong> · orbit</>;
	if (mode === "interior") return <><strong>interior</strong> · walkthrough</>;
	if (mode === "peek") return <><strong>locating</strong> · release to return</>;
	return null;
}

function hudContent(state: OrbitState): ReactNode {
	const { mode, hover, currentId, currentIndex, panoCount } = state;
	if (mode === "overview") {
		if (hover) return <>enter at <strong>{hover.id}</strong></>;
		if (panoCount === 0) return <>drag to orbit · scroll to zoom · no walkthrough tour published</>;
		return (
			<>
				drag to orbit · right-drag to pan · scroll to zoom · click a marker or{" "}
				<strong>enter interior</strong>
			</>
		);
	}
	if (mode === "interior") {
		if (hover) {
			return (
				<>
					<strong>{currentId}</strong> · go to <strong>{hover.id}</strong>
					{hover.occluded ? " · behind wall" : ""}
				</>
			);
		}
		if (currentId) {
			return (
				<>
					<strong>{currentId}</strong> · {currentIndex + 1}/{panoCount} · drag to look ·{" "}
					<strong>WASD</strong> to step
				</>
			);
		}
		return null;
	}
	if (mode === "peek") return <>you are <strong>here</strong> · release to drop back in</>;
	return null;
}
