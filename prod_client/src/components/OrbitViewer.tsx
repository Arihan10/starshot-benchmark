"use client";

import { useEffect, useRef, useState, type ReactNode, type RefObject } from "react";
import { useScene } from "@/components/SceneProvider";
import { OrbitEngine } from "@/lib/orbit/engine";
import { INITIAL_ORBIT_STATE, type OrbitState } from "@/lib/orbit/types";
import { tourSource } from "@/lib/scenes";

export default function OrbitViewer() {
	const hostRef = useRef<HTMLDivElement>(null);
	const engineRef = useRef<OrbitEngine | null>(null);
	const menuRef = useRef<HTMLDivElement>(null);
	const [state, setState] = useState<OrbitState>(INITIAL_ORBIT_STATE);
	const [holding, setHolding] = useState(false);
	const { selected, status, error } = useScene();

	// Mount the imperative engine once; dispose fully on unmount so React
	// StrictMode's double-invoke leaves nothing leaked.
	useEffect(() => {
		const host = hostRef.current;
		if (!host) return;
		const engine = new OrbitEngine(host, setState, setHolding);
		engineRef.current = engine;
		return () => {
			engine.dispose();
			engineRef.current = null;
		};
	}, []);

	// Load (or reload) the selected scene's assets from R2 whenever it changes:
	// the dollhouse GLB plus the optional capture-tour manifest (positions +
	// proxy). The engine fetches the manifest and streams panos lazily.
	useEffect(() => {
		if (selected) void engineRef.current?.loadTour(tourSource(selected));
	}, [selected]);

	// Dismiss the object menu on any outside press (capture, so it beats the canvas
	// handlers) or Escape. The contextMenu reference is stable while open, so this
	// only re-subscribes when the menu opens / closes / moves.
	useEffect(() => {
		if (!state.contextMenu) return;
		const onDocPointerDown = (e: PointerEvent) => {
			if (!menuRef.current?.contains(e.target as Node)) engineRef.current?.closeMenu();
		};
		const onKeyDown = (e: KeyboardEvent) => {
			if (e.key === "Escape") engineRef.current?.closeMenu();
		};
		document.addEventListener("pointerdown", onDocPointerDown, true);
		window.addEventListener("keydown", onKeyDown);
		return () => {
			document.removeEventListener("pointerdown", onDocPointerDown, true);
			window.removeEventListener("keydown", onKeyDown);
		};
	}, [state.contextMenu]);

	const { mode, overlay, minimap } = state;
	const hud = hudContent(state);
	const catalogMessage =
		status === "loading"
			? "loading scenes…"
			: status === "error"
				? `failed to load scenes: ${error}`
				: !selected
					? "no scenes published yet"
					: null;

	return (
		<div className="relative flex-1">
			<div ref={hostRef} className="absolute inset-0 bg-[#0c0d10]" />

			{catalogMessage && (
				<div className="absolute inset-0 z-20 flex items-center justify-center bg-[#0c0d10]/80">
					<span className={`text-sm ${status === "error" ? "text-red-400" : "text-neutral-400"}`}>
						{catalogMessage}
					</span>
				</div>
			)}

			{mode !== "empty" && mode !== "loading" && (
				<div className="absolute left-4 top-4 z-10 flex flex-col gap-2">
					{minimap && <Minimap minimap={minimap} engine={engineRef} />}
					<div className="pointer-events-none text-[10px] uppercase tracking-wider text-neutral-400 [&_strong]:font-semibold [&_strong]:text-cyan-200">
						{modeLabel(state)}
					</div>
				</div>
			)}

			<div className="absolute right-4 top-4 z-10 flex gap-2">
				{state.canProxyView && (
					<button
						type="button"
						title="Toggle between the textured scene and the bare low-poly proxy (no panorama)"
						onClick={() => engineRef.current?.toggleProxyView()}
						className={`rounded-md border px-3 py-2 text-xs backdrop-blur transition ${
							state.proxyView
								? "border-violet-400/70 bg-violet-500/20 text-violet-200"
								: "border-white/15 bg-black/50 text-neutral-200 hover:border-white/30 hover:text-white"
						}`}
					>
						proxy view
					</button>
				)}
				{state.canHighlight && (
					<button
						type="button"
						title="Highlight the object under the cursor on hover"
						onClick={() => engineRef.current?.toggleHighlight()}
						className={`rounded-md border px-3 py-2 text-xs backdrop-blur transition ${
							state.highlightEnabled
								? "border-cyan-400/70 bg-cyan-500/20 text-cyan-100"
								: "border-white/15 bg-black/50 text-neutral-300 hover:border-white/30 hover:text-white"
						}`}
					>
						hover highlight
					</button>
				)}
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

			{state.contextMenu && (
				<ObjectMenu menu={state.contextMenu} menuRef={menuRef} engine={engineRef} />
			)}
		</div>
	);
}

function ObjectMenu({
	menu,
	menuRef,
	engine,
}: {
	menu: NonNullable<OrbitState["contextMenu"]>;
	menuRef: RefObject<HTMLDivElement | null>;
	engine: RefObject<OrbitEngine | null>;
}) {
	const hasExtras = menu.hiddenCount > 0 || menu.outlinedCount > 0;
	const left = Math.min(menu.x, window.innerWidth - 200);
	const top = Math.min(menu.y, window.innerHeight - 180);
	return (
		<div
			ref={menuRef}
			style={{ left, top }}
			onContextMenu={(e) => e.preventDefault()}
			className="fixed z-50 min-w-[170px] select-none rounded-md border border-white/10 bg-neutral-900/95 p-1 text-xs shadow-2xl backdrop-blur"
		>
			{menu.label && (
				<div className="mb-1 max-w-[240px] truncate border-b border-white/10 px-2 py-1.5 font-semibold text-cyan-200">
					{menu.label}
				</div>
			)}
			{menu.label && (
				<MenuButton onClick={() => engine.current?.toggleMenuTargetHidden()}>
					{menu.hidden ? "show" : "hide"}
				</MenuButton>
			)}
			{menu.label && (
				<MenuButton onClick={() => engine.current?.toggleMenuTargetOutline()}>
					{menu.outlined ? "remove outline" : "highlight outline"}
				</MenuButton>
			)}
			{menu.label && hasExtras && <div className="my-1 h-px bg-white/10" />}
			{menu.hiddenCount > 0 && (
				<MenuButton onClick={() => engine.current?.showAllHidden()}>
					show all ({menu.hiddenCount})
				</MenuButton>
			)}
			{menu.outlinedCount > 0 && (
				<MenuButton onClick={() => engine.current?.clearOutlines()}>clear outlines</MenuButton>
			)}
		</div>
	);
}

function MenuButton({ onClick, children }: { onClick: () => void; children: ReactNode }) {
	return (
		<button
			type="button"
			onClick={onClick}
			className="block w-full rounded px-2 py-1.5 text-left text-neutral-200 transition hover:bg-cyan-500/20 hover:text-white"
		>
			{children}
		</button>
	);
}

// Top-left bird's-eye minimap. Shows the captured slice for the floor you're
// viewing, with that floor's anchors dotted on (the live one lit). The floor
// switcher pages between captured levels WITHOUT moving the camera; it defaults
// to — and follows — the floor the character is on. Clicking a dot walks there;
// the header toggle expands it. The box always keeps the slice's aspect (so the
// %-placed dots stay aligned), at the largest size that fits the caps on both
// axes — width = min(maxW, maxH * aspect).
const minimapWidth = (aspect: number, maxW: string, maxH: string) =>
	`min(${maxW}, calc(${maxH} * ${aspect}))`;
const MINIMAP_COMPACT = { w: "200px", h: "170px" };
const MINIMAP_EXPANDED = { w: "min(48vw, 640px)", h: "min(66vh, 640px)" };

function Minimap({
	minimap,
	engine,
}: {
	minimap: NonNullable<OrbitState["minimap"]>;
	engine: RefObject<OrbitEngine | null>;
}) {
	const [expanded, setExpanded] = useState(false);
	const { currentLevel, levels } = minimap;
	const [viewedLevel, setViewedLevel] = useState(currentLevel);
	// Default to / follow the character's floor; a manual pick sticks until they
	// change floors. Adjusting state during render on a prop change (vs. an effect)
	// is React's recommended pattern — no extra commit / cascading render.
	const [prevLevel, setPrevLevel] = useState(currentLevel);
	if (currentLevel !== prevLevel) {
		setPrevLevel(currentLevel);
		setViewedLevel(currentLevel);
	}

	const view = levels[viewedLevel] ?? levels[currentLevel];
	if (!view) return null;
	const caps = expanded ? MINIMAP_EXPANDED : MINIMAP_COMPACT;
	const onCurrentFloor = view.level === currentLevel;
	return (
		<div className="rounded-md border border-white/10 bg-black/60 p-1.5 backdrop-blur">
			<div className="mb-1 flex items-center justify-between gap-2 px-0.5 text-[9px] uppercase tracking-wider text-neutral-400">
				<span>minimap</span>
				<div className="flex items-center gap-1">
					{levels.length > 1 && (
						<div className="flex items-center gap-0.5" title="switch floor (doesn't move you)">
							{levels.map((lv) => {
								const isViewed = lv.level === viewedLevel;
								const isCurrent = lv.level === currentLevel;
								return (
									<button
										key={lv.level}
										type="button"
										title={`floor ${lv.level + 1}${isCurrent ? " · you are here" : ""}`}
										onClick={() => setViewedLevel(lv.level)}
										className={`rounded px-1 py-0.5 text-[9px] leading-none tabular-nums transition ${
											isViewed
												? "bg-cyan-500/30 text-cyan-100"
												: "text-neutral-400 hover:bg-white/10 hover:text-white"
										} ${isCurrent ? "ring-1 ring-inset ring-cyan-300/70" : ""}`}
									>
										{lv.level + 1}
									</button>
								);
							})}
						</div>
					)}
					<button
						type="button"
						onClick={() => setExpanded((v) => !v)}
						title={expanded ? "collapse minimap" : "expand minimap"}
						aria-label={expanded ? "collapse minimap" : "expand minimap"}
						className="rounded px-1 text-[11px] leading-none text-neutral-300 transition hover:bg-white/10 hover:text-white"
					>
						{expanded ? "✕" : "⤢"}
					</button>
				</div>
			</div>
			<div
				className="relative overflow-hidden rounded"
				style={{ width: minimapWidth(view.aspect, caps.w, caps.h), aspectRatio: view.aspect }}
			>
				{/* eslint-disable-next-line @next/next/no-img-element -- runtime R2 slice (proxied via /r2), not a static asset; next/image adds no value here */}
				<img
					src={view.url}
					alt="scene from above"
					draggable={false}
					className="absolute inset-0 h-full w-full object-fill"
				/>
				{view.points.map((pt) => (
					<button
						key={pt.index}
						type="button"
						title={pt.id}
						onClick={() => engine.current?.travelToIndex(pt.index)}
						style={{ left: `${pt.leftPct}%`, top: `${pt.topPct}%` }}
						className={`absolute -translate-x-1/2 -translate-y-1/2 rounded-full border transition ${
							pt.current
								? `${expanded ? "h-3.5 w-3.5" : "h-2.5 w-2.5"} border-white bg-cyan-400 shadow-[0_0_6px_2px_rgba(34,211,238,0.7)]`
								: `${expanded ? "h-3 w-3" : "h-2 w-2"} border-white/70 bg-cyan-300/40 hover:scale-125 hover:bg-cyan-300/90`
						}`}
					/>
				))}
				{!onCurrentFloor && (
					<div className="pointer-events-none absolute bottom-1 left-1 rounded bg-black/65 px-1 py-0.5 text-[8px] uppercase tracking-wider text-cyan-200">
						floor {view.level + 1} · you are on {currentLevel + 1}
					</div>
				)}
			</div>
		</div>
	);
}

function modeLabel({ mode, proxyView }: OrbitState): ReactNode {
	if (mode === "overview") {
		return proxyView ? <><strong>proxy</strong> · orbit</> : <><strong>dollhouse</strong> · orbit</>;
	}
	if (mode === "interior") {
		return proxyView ? <><strong>interior</strong> · proxy</> : <><strong>interior</strong> · walkthrough</>;
	}
	if (mode === "peek") return <><strong>locating</strong> · release to return</>;
	return null;
}

function hudContent(state: OrbitState): ReactNode {
	const { mode, hover, objectHover, currentId, currentIndex, panoCount } = state;
	if (mode === "overview") {
		if (hover) return <>enter at <strong>{hover.id}</strong></>;
		if (objectHover) {
			return (
				<>
					object <strong>{objectHover}</strong> · right-click to hide / outline
				</>
			);
		}
		if (panoCount === 0) return <>drag to orbit · scroll to zoom · no walkthrough tour published</>;
		return (
			<>
				drag to orbit · right-drag to pan · scroll to zoom · click a marker or{" "}
				<strong>enter interior</strong>
			</>
		);
	}
	if (mode === "interior") {
		if (objectHover) {
			return (
				<>
					object <strong>{objectHover}</strong>
				</>
			);
		}
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
					<strong>{currentId}</strong> · {currentIndex + 1}/{panoCount} · drag to look
				</>
			);
		}
		return null;
	}
	if (mode === "peek") return <>you are <strong>here</strong> · release to drop back in</>;
	return null;
}
