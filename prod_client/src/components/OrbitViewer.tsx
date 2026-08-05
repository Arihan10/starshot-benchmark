"use client";

import {
	useCallback,
	useEffect,
	useMemo,
	useRef,
	useState,
	type CSSProperties,
	type ReactNode,
	type RefObject,
} from "react";
import { OrbitEngine } from "@/lib/orbit/engine";
import type { EdgeType } from "@/lib/orbit/navGraph";
import {
	INITIAL_ORBIT_STATE,
	type Chapter,
	type HoverPreview,
	type LevelPreview,
	type NavExit,
	type OrbitState,
} from "@/lib/orbit/types";
import { tourSource, type Scene } from "@/lib/scenes";

// One palette + vocabulary for the five edge types, shared by every affordance,
// the exits panel, the preview card, and the minimap — so the grammar reads the
// same everywhere (that consistency is the whole point).
const EDGE_META: Record<
	EdgeType,
	{ label: string; verb: string; color: string }
> = {
	walk: { label: "Walk", verb: "walk over", color: "#8fd0ff" },
	portal: { label: "Doorway", verb: "step through", color: "#ffc46b" },
	vertical: { label: "Level", verb: "change level", color: "#7ef2c2" },
	phase: { label: "Phase", verb: "phase through wall", color: "#c9a6ff" },
	far: { label: "Travel", verb: "travel across", color: "#9aa7b4" },
};

// Drive a callback from the engine's live facing (deg) each frame WITHOUT React
// re-renders — used to spin the minimap cone + exit arrows as the user looks.
function useFacingLoop(
	engineRef: RefObject<OrbitEngine | null>,
	cb: (facingDeg: number) => void,
) {
	useEffect(() => {
		let raf = 0;
		const loop = () => {
			const deg = engineRef.current?.getFacingDeg();
			if (typeof deg === "number") cb(deg);
			raf = requestAnimationFrame(loop);
		};
		raf = requestAnimationFrame(loop);
		return () => cancelAnimationFrame(raf);
	}, [engineRef, cb]);
}

export default function OrbitViewer({
	scene,
	onFocusedChange,
}: {
	scene: Scene | null;
	onFocusedChange?: (focused: boolean) => void;
}) {
	const hostRef = useRef<HTMLDivElement>(null);
	const engineRef = useRef<OrbitEngine | null>(null);
	const menuRef = useRef<HTMLDivElement>(null);
	const [state, setState] = useState<OrbitState>(INITIAL_ORBIT_STATE);
	const [holding, setHolding] = useState(false);
	const [drawer, setDrawer] = useState(false);

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

	useEffect(() => {
		if (scene) void engineRef.current?.loadTour(tourSource(scene));
	}, [scene]);

	const focused =
		state.mode === "interior" ||
		state.mode === "transition" ||
		state.mode === "peek";
	useEffect(() => {
		onFocusedChange?.(focused);
	}, [focused, onFocusedChange]);

	// Dismiss the object menu on any outside press or Escape.
	useEffect(() => {
		if (!state.contextMenu) return;
		const onDocPointerDown = (e: PointerEvent) => {
			if (!menuRef.current?.contains(e.target as Node))
				engineRef.current?.closeMenu();
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

	// "M" toggles the chapters/search drawer while inside.
	useEffect(() => {
		const onKey = (e: KeyboardEvent) => {
			if (e.code === "KeyM" && state.mode === "interior") {
				const el = document.activeElement;
				if (el instanceof HTMLInputElement) return;
				setDrawer((v) => !v);
			}
		};
		window.addEventListener("keydown", onKey);
		return () => window.removeEventListener("keydown", onKey);
	}, [state.mode]);

	const { mode, overlay, minimap } = state;

	return (
		<div className='relative h-full w-full'>
			<div ref={hostRef} className='absolute inset-0 bg-[#0c0d10]' />

			{mode !== "empty" && mode !== "loading" && (
				<div className='absolute left-4 top-4 z-10 flex flex-col gap-2'>
					{minimap && (
						<Minimap
							minimap={minimap}
							mapEdges={state.mapEdges}
							visited={state.visited}
							currentIndex={state.currentIndex}
							engine={engineRef}
						/>
					)}
					<div className='pointer-events-none text-[10px] uppercase tracking-wider text-neutral-400 [&_strong]:font-semibold [&_strong]:text-cyan-200'>
						{modeLabel(state)}
					</div>
				</div>
			)}

			<div className='absolute right-4 top-4 z-10 flex flex-wrap justify-end gap-2'>
				{state.canProxyView && (
					<ToolButton
						active={state.proxyView}
						activeClass='border-violet-400/70 bg-violet-500/20 text-violet-200'
						title='Toggle between the textured scene and the bare low-poly proxy'
						onClick={() => engineRef.current?.toggleProxyView()}
					>
						proxy view
					</ToolButton>
				)}
				{state.canHighlight && (
					<ToolButton
						active={state.highlightEnabled}
						activeClass='border-cyan-400/70 bg-cyan-500/20 text-cyan-100'
						title='Highlight the object under the cursor on hover'
						onClick={() => engineRef.current?.toggleHighlight()}
					>
						hover highlight
					</ToolButton>
				)}
				{mode === "overview" && (
					<button
						type='button'
						disabled={state.panoCount === 0}
						onClick={() => engineRef.current?.enter()}
						className='rounded-md border border-cyan-500/40 bg-cyan-500/10 px-3 py-2 text-xs text-cyan-100 transition hover:border-cyan-400 hover:bg-cyan-500/20 hover:text-white disabled:cursor-default disabled:opacity-40'
					>
						enter interior ▸
					</button>
				)}
				{mode === "interior" && (
					<>
						<ToolButton
							active={false}
							title='Retrace your steps (Backspace)'
							disabled={!state.canGoBack}
							onClick={() => engineRef.current?.goBack()}
						>
							◂ back
						</ToolButton>
						<ToolButton
							active={!!state.tour}
							activeClass='border-emerald-400/70 bg-emerald-500/20 text-emerald-100'
							title={
								state.tour
									? "Stop the tour and stay exactly where you are"
									: "Auto tour: walk to the middle of each zone and look around"
							}
							onClick={() => engineRef.current?.toggleTour()}
						>
							{state.tour ? "■ stop tour" : "▶ auto tour"}
						</ToolButton>
						<ToolButton
							active={state.sonarActive}
							activeClass='border-cyan-300/70 bg-cyan-400/20 text-cyan-100'
							title='Ping: reveal every nearby node through walls (Tab)'
							onClick={() => engineRef.current?.toggleSonar()}
						>
							◎ ping
						</ToolButton>
						<ToolButton
							active={drawer}
							activeClass='border-cyan-400/70 bg-cyan-500/20 text-cyan-100'
							title='Chapters & search (M)'
							onClick={() => setDrawer((v) => !v)}
						>
							☰ places
						</ToolButton>
						<button
							type='button'
							disabled={state.busy}
							onClick={() => engineRef.current?.exit()}
							className='rounded-md border border-white/15 bg-black/50 px-3 py-2 text-xs text-neutral-200 backdrop-blur transition hover:border-white/30 hover:text-white disabled:cursor-default disabled:opacity-40'
						>
							◂ overview
						</button>
					</>
				)}
				{(mode === "interior" || mode === "peek") && (
					<button
						type='button'
						title='Hold to zoom out to the dollhouse and mark where you are'
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
						⤢ locate
					</button>
				)}
			</div>

			{mode === "interior" && state.exits.length > 0 && (
				<ExitsPanel exits={state.exits} engine={engineRef} />
			)}

			{mode === "interior" && drawer && (
				<PlacesDrawer
					state={state}
					engine={engineRef}
					onClose={() => setDrawer(false)}
				/>
			)}

			{state.levelPreview && (
				<LevelPreviewPanel preview={state.levelPreview} />
			)}
			{state.preview && <HoverCard preview={state.preview} />}
			{mode === "interior" && state.arrival && (
				<ArrivalToast
					key={state.arrival.ts}
					arrival={state.arrival}
					trapped={state.trapped}
				/>
			)}

			<Hud state={state} />

			{overlay && (
				<div className='absolute inset-0 z-20 flex flex-col items-center justify-center gap-2.5 bg-[#0c0d10]/80'>
					{overlay.spinner && (
						<span className='h-5 w-5 animate-spin rounded-full border-2 border-white/15 border-t-cyan-400' />
					)}
					<span
						className={`text-xs ${overlay.err ? "text-red-400" : "text-neutral-300"}`}
					>
						{overlay.msg}
					</span>
				</div>
			)}

			{state.contextMenu && (
				<ObjectMenu
					menu={state.contextMenu}
					menuRef={menuRef}
					engine={engineRef}
				/>
			)}
		</div>
	);
}

function ToolButton({
	active,
	activeClass = "border-cyan-400/70 bg-cyan-500/20 text-cyan-100",
	title,
	disabled,
	onClick,
	children,
}: {
	active: boolean;
	activeClass?: string;
	title: string;
	disabled?: boolean;
	onClick: () => void;
	children: ReactNode;
}) {
	return (
		<button
			type='button'
			title={title}
			disabled={disabled}
			onClick={onClick}
			className={`rounded-md border px-3 py-2 text-xs backdrop-blur transition disabled:cursor-default disabled:opacity-40 ${
				active
					? activeClass
					: "border-white/15 bg-black/50 text-neutral-200 hover:border-white/30 hover:text-white"
			}`}
		>
			{children}
		</button>
	);
}

// The exits panel (affordance layer, in text): every in-view exit of the node,
// with a live arrow that always points true (rotates with the camera) and a
// verb — this is also what a screen reader would read on arrival.
function ExitsPanel({
	exits,
	engine,
}: {
	exits: NavExit[];
	engine: RefObject<OrbitEngine | null>;
}) {
	const containerRef = useRef<HTMLDivElement>(null);
	const spin = useCallback((facingDeg: number) => {
		const el = containerRef.current;
		if (!el) return;
		for (const arrow of el.querySelectorAll<HTMLElement>(
			"[data-bearing]",
		)) {
			const bearing = Number(arrow.dataset.bearing);
			arrow.style.transform = `rotate(${bearing - facingDeg}deg)`;
		}
	}, []);
	useFacingLoop(engine, spin);

	return (
		<div
			ref={containerRef}
			className='absolute bottom-4 right-4 z-10 w-56 rounded-lg border border-white/10 bg-black/60 p-2 backdrop-blur'
		>
			<div className='mb-1 px-1 text-[9px] uppercase tracking-wider text-neutral-400'>
				exits
			</div>
			<div className='flex flex-col gap-0.5'>
				{exits.map((e) => {
					const meta = EDGE_META[e.type];
					return (
						<button
							key={e.index}
							type='button'
							onClick={() => engine.current?.traverseTo(e.index)}
							className='group flex items-center gap-2 rounded px-1.5 py-1 text-left text-xs text-neutral-200 transition hover:bg-white/10'
						>
							<span
								className='flex h-4 w-4 shrink-0 items-center justify-center'
								data-bearing={e.bearingDeg}
							>
								<span
									className='text-[11px] leading-none'
									style={{ color: meta.color }}
								>
									▲
								</span>
							</span>
							<span className='min-w-0 flex-1 truncate'>
								{e.name ?? `node ${e.index + 1}`}
							</span>
							<span
								className='shrink-0 rounded px-1 py-0.5 text-[8px] uppercase tracking-wide'
								style={{
									color: meta.color,
									background: `${meta.color}1f`,
								}}
							>
								{meta.label}
							</span>
							<span className='shrink-0 text-[9px] tabular-nums text-neutral-500'>
								{e.dist < 100 ? `${e.dist.toFixed(0)}m` : ""}
							</span>
						</button>
					);
				})}
			</div>
		</div>
	);
}

// The hover preview: destination thumbnail panned to its arrival heading, plus
// the LLM-authored name + distance + the transition verb. Floats at the
// affordance's screen point; pointer-transparent so it never blocks a click.
function HoverCard({ preview }: { preview: HoverPreview }) {
	const meta = EDGE_META[preview.type];
	return (
		<div
			className='pointer-events-none absolute z-20 w-48 -translate-x-1/2 -translate-y-[calc(100%+14px)] overflow-hidden rounded-lg border bg-neutral-950/85 shadow-2xl backdrop-blur'
			style={{
				left: preview.screenX,
				top: preview.screenY,
				borderColor: `${meta.color}66`,
			}}
		>
			<div className='relative h-26 w-full bg-neutral-800'>
				{/* eslint-disable-next-line @next/next/no-img-element -- runtime R2 thumbnail via /r2 proxy */}
				<img
					src={preview.thumbUrl}
					alt=''
					draggable={false}
					className='h-full w-full object-cover'
					style={{ objectPosition: `${preview.headingU * 100}% 50%` }}
				/>
				<span
					className='absolute left-2 top-2 rounded px-1.5 py-0.5 text-[8px] font-semibold uppercase tracking-wider'
					style={{ color: "#0b0d12", background: meta.color }}
				>
					{meta.label}
				</span>
			</div>
			<div className='flex items-center justify-between gap-2 px-2.5 py-1.5'>
				<span className='min-w-0 truncate text-xs font-semibold text-white'>
					{preview.name ?? "unnamed"}
				</span>
				<span className='shrink-0 text-[10px] tabular-nums text-neutral-400'>
					{preview.dist < 100 ? `${preview.dist.toFixed(0)} m` : ""}
				</span>
			</div>
		</div>
	);
}

const PANO_SCREEN_H = 240; // px: the preview "screen" height
// One full 360 is drawn this wide. Being well past the panel's own width means at
// most a third of the panorama is on screen at once, so the frame reads as a view
// into the room rather than a whole equirect flattened out — and the wrap seam is
// never visible twice in the same frame. Height follows the 2:1 aspect (900px) and
// gets cropped to the screen, which lands the visible band on the horizon.
const PANO_TILE_W = 1800;

// The floor-waypoint preview: a wide "screen" that pans continuously through the
// destination panorama rather than squashing a whole equirect into one frame, so
// you read the room you're about to drop into instead of a warped strip. The pano
// is tiled horizontally and shifted by exactly one tile width (see the pano-pan
// keyframes), which is what makes the 360 loop seamlessly. Pointer-transparent, so
// it never swallows the click it is previewing.
function LevelPreviewPanel({ preview }: { preview: LevelPreview }) {
	return (
		<div className='pointer-events-none absolute left-1/2 top-1/2 z-30 w-[min(600px,70vw)] -translate-x-1/2 -translate-y-1/2'>
			<div className='overflow-hidden rounded-xl border border-red-400/50 bg-neutral-950/85 shadow-2xl backdrop-blur'>
				<div
					className='relative overflow-hidden bg-neutral-900'
					style={{ height: PANO_SCREEN_H }}
				>
					<div
						className='absolute inset-0 animate-[pano-pan_28s_linear_infinite]'
						style={
							{
								// Full pano over the blurred placeholder, so the frame is filled
								// the instant it opens and both layers pan together.
								backgroundImage: `url(${preview.url}), url(${preview.placeholderUrl})`,
								backgroundRepeat: "repeat-x",
								backgroundSize: `${PANO_TILE_W}px auto`,
								backgroundPositionY: "50%",
								"--pano-tile": `${PANO_TILE_W}px`,
							} as CSSProperties
						}
					/>
					<div className='pointer-events-none absolute inset-x-0 top-0 h-14 bg-linear-to-b from-black/70 to-transparent' />
					<span className='absolute left-3 top-3 rounded bg-red-500 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-neutral-950'>
						{preview.up ? "▲" : "▼"} floor {preview.level + 1}
					</span>
				</div>
				<div className='flex items-center justify-between gap-3 px-3 py-2'>
					<span className='min-w-0 truncate text-sm font-semibold text-white'>
						{preview.name ?? `floor ${preview.level + 1}`}
					</span>
					<span className='shrink-0 text-[10px] uppercase tracking-wider text-red-300'>
						click to go {preview.up ? "up" : "down"}
					</span>
				</div>
			</div>
		</div>
	);
}

// Arrival narration ("Archive · sealed room, phased through the wall") — invariant
// #4 in text form. The caller keys this on `arrival.ts`, so each arrival mounts a
// fresh toast: the fade-out timer re-arms and the transition replays.
function ArrivalToast({
	arrival,
	trapped,
}: {
	arrival: NonNullable<OrbitState["arrival"]>;
	trapped: boolean;
}) {
	const [shown, setShown] = useState(true);
	useEffect(() => {
		const t = setTimeout(() => setShown(false), 2600);
		return () => clearTimeout(t);
	}, []);
	return (
		<div
			className={`pointer-events-none absolute bottom-20 left-1/2 z-20 -translate-x-1/2 rounded-full border border-white/10 bg-black/75 px-4 py-1.5 text-xs backdrop-blur transition-all duration-300 ${
				shown ? "opacity-100" : "translate-y-1 opacity-0"
			}`}
		>
			<span className='font-semibold text-white'>{arrival.name}</span>
			<span className='text-neutral-400'>
				{trapped ? " · sealed room" : ""} · {arrival.verb}
			</span>
		</div>
	);
}

// Layer 4: chapters (zones by their authored names) + free-text "take me to".
function PlacesDrawer({
	state,
	engine,
	onClose,
}: {
	state: OrbitState;
	engine: RefObject<OrbitEngine | null>;
	onClose: () => void;
}) {
	const [query, setQuery] = useState("");
	const results = useMemo(() => {
		const q = query.trim().toLowerCase();
		if (!q) return [];
		return state.nodes
			.filter(
				(n) =>
					(n.name ?? "").toLowerCase().includes(q) ||
					(n.zone ?? "").toLowerCase().includes(q),
			)
			.slice(0, 20);
	}, [query, state.nodes]);
	const go = (index: number) => {
		engine.current?.traverseTo(index);
		onClose();
	};
	return (
		<div className='absolute right-4 top-16 z-30 flex max-h-[70vh] w-72 flex-col rounded-lg border border-white/10 bg-neutral-950/90 p-2 shadow-2xl backdrop-blur'>
			<input
				autoFocus
				value={query}
				onChange={(e) => setQuery(e.target.value)}
				placeholder='take me to…'
				className='mb-2 w-full rounded-md border border-white/15 bg-black/50 px-2.5 py-1.5 text-xs text-neutral-100 outline-none transition placeholder:text-neutral-500 focus:border-cyan-400'
			/>
			<div className='min-h-0 flex-1 overflow-y-auto'>
				{query.trim() ? (
					results.length ? (
						results.map((n) => (
							<button
								key={n.index}
								type='button'
								onClick={() => go(n.index)}
								className='block w-full truncate rounded px-2 py-1.5 text-left text-xs text-neutral-200 transition hover:bg-cyan-500/20 hover:text-white'
							>
								{n.name ?? `node ${n.index + 1}`}
								{n.zone ? (
									<span className='text-neutral-500'>
										{" "}
										· {n.zone}
									</span>
								) : null}
							</button>
						))
					) : (
						<div className='px-2 py-1.5 text-xs text-neutral-500'>
							no matches
						</div>
					)
				) : (
					<ChapterList chapters={state.chapters} onGo={go} />
				)}
			</div>
		</div>
	);
}

function ChapterList({
	chapters,
	onGo,
}: {
	chapters: Chapter[];
	onGo: (index: number) => void;
}) {
	if (chapters.length === 0)
		return (
			<div className='px-2 py-1.5 text-xs text-neutral-500'>no zones</div>
		);
	return (
		<>
			<div className='mb-1 px-1 text-[9px] uppercase tracking-wider text-neutral-500'>
				chapters
			</div>
			{chapters.map((c) => (
				<button
					key={c.zone || c.firstIndex}
					type='button'
					onClick={() => onGo(c.firstIndex)}
					className='flex w-full items-center justify-between gap-2 rounded px-2 py-1.5 text-left text-xs text-neutral-200 transition hover:bg-cyan-500/20 hover:text-white'
				>
					<span className='min-w-0 truncate'>
						{c.zone || "unzoned"}
					</span>
					<span className='shrink-0 text-[9px] tabular-nums text-neutral-500'>
						{c.count}
					</span>
				</button>
			))}
		</>
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
			className='fixed z-50 min-w-42.5 select-none rounded-md border border-white/10 bg-neutral-900/95 p-1 text-xs shadow-2xl backdrop-blur'
		>
			{menu.label && (
				<div className='mb-1 max-w-60 truncate border-b border-white/10 px-2 py-1.5 font-semibold text-cyan-200'>
					{menu.label}
				</div>
			)}
			{menu.label && (
				<MenuButton
					onClick={() => engine.current?.toggleMenuTargetHidden()}
				>
					{menu.hidden ? "show" : "hide"}
				</MenuButton>
			)}
			{menu.label && (
				<MenuButton
					onClick={() => engine.current?.toggleMenuTargetOutline()}
				>
					{menu.outlined ? "remove outline" : "highlight outline"}
				</MenuButton>
			)}
			{menu.label && hasExtras && (
				<div className='my-1 h-px bg-white/10' />
			)}
			{menu.hiddenCount > 0 && (
				<MenuButton onClick={() => engine.current?.showAllHidden()}>
					show all ({menu.hiddenCount})
				</MenuButton>
			)}
			{menu.outlinedCount > 0 && (
				<MenuButton onClick={() => engine.current?.clearOutlines()}>
					clear outlines
				</MenuButton>
			)}
		</div>
	);
}

function MenuButton({
	onClick,
	children,
}: {
	onClick: () => void;
	children: ReactNode;
}) {
	return (
		<button
			type='button'
			onClick={onClick}
			className='block w-full rounded px-2 py-1.5 text-left text-neutral-200 transition hover:bg-cyan-500/20 hover:text-white'
		>
			{children}
		</button>
	);
}

const minimapWidth = (aspect: number, maxW: string, maxH: string) =>
	`min(${maxW}, calc(${maxH} * ${aspect}))`;
const MINIMAP_COMPACT = { w: "200px", h: "170px" };
const MINIMAP_EXPANDED = { w: "min(48vw, 640px)", h: "min(66vh, 640px)" };

// Layer 3: the bird's-eye minimap. Now with a live you-are-here facing cone,
// visited/unvisited fills, dashed phase edge lines (the map never lies), and a
// re-anchoring flash after every arrival.
function Minimap({
	minimap,
	mapEdges,
	visited,
	currentIndex,
	engine,
}: {
	minimap: NonNullable<OrbitState["minimap"]>;
	mapEdges: OrbitState["mapEdges"];
	visited: number[];
	currentIndex: number;
	engine: RefObject<OrbitEngine | null>;
}) {
	const [expanded, setExpanded] = useState(false);
	const { currentLevel, levels } = minimap;
	const [viewedLevel, setViewedLevel] = useState(currentLevel);
	const [prevLevel, setPrevLevel] = useState(currentLevel);
	if (currentLevel !== prevLevel) {
		setPrevLevel(currentLevel);
		setViewedLevel(currentLevel);
	}

	// Re-anchoring flash: pulse the current dot briefly after each arrival.
	const [flash, setFlash] = useState(0);
	const [prevIdx, setPrevIdx] = useState(currentIndex);
	if (currentIndex !== prevIdx) {
		setPrevIdx(currentIndex);
		setFlash((n) => n + 1);
	}
	useEffect(() => {
		if (!flash) return;
		const t = setTimeout(() => setFlash(0), 450);
		return () => clearTimeout(t);
	}, [flash]);

	const coneRef = useRef<HTMLDivElement>(null);
	const spin = useCallback((facingDeg: number) => {
		if (coneRef.current)
			coneRef.current.style.transform = `translate(-50%, -50%) rotate(${facingDeg}deg)`;
	}, []);
	useFacingLoop(engine, spin);

	const view = levels[viewedLevel] ?? levels[currentLevel];
	const visitedSet = useMemo(() => new Set(visited), [visited]);
	const ptByIndex = useMemo(() => {
		const m = new Map<number, { left: number; top: number }>();
		if (view)
			for (const p of view.points)
				m.set(p.index, { left: p.leftPct, top: p.topPct });
		return m;
	}, [view]);
	if (!view) return null;
	const caps = expanded ? MINIMAP_EXPANDED : MINIMAP_COMPACT;
	const onCurrentFloor = view.level === currentLevel;

	return (
		<div className='rounded-md border border-white/10 bg-black/60 p-1.5 backdrop-blur'>
			<div className='mb-1 flex items-center justify-between gap-2 px-0.5 text-[9px] uppercase tracking-wider text-neutral-400'>
				<span>minimap</span>
				<div className='flex items-center gap-1'>
					{levels.length > 1 && (
						<div
							className='flex items-center gap-0.5'
							title="switch floor (doesn't move you)"
						>
							{levels.map((lv) => {
								const isViewed = lv.level === viewedLevel;
								const isCurrent = lv.level === currentLevel;
								return (
									<button
										key={lv.level}
										type='button'
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
						type='button'
						onClick={() => setExpanded((v) => !v)}
						title={expanded ? "collapse minimap" : "expand minimap"}
						aria-label={
							expanded ? "collapse minimap" : "expand minimap"
						}
						className='rounded px-1 text-[11px] leading-none text-neutral-300 transition hover:bg-white/10 hover:text-white'
					>
						{expanded ? "✕" : "⤢"}
					</button>
				</div>
			</div>
			<div
				className='relative overflow-hidden rounded'
				style={{
					width: minimapWidth(view.aspect, caps.w, caps.h),
					aspectRatio: view.aspect,
				}}
			>
				{/* eslint-disable-next-line @next/next/no-img-element -- runtime R2 slice via /r2 proxy */}
				<img
					src={view.url}
					alt='scene from above'
					draggable={false}
					className='absolute inset-0 h-full w-full object-fill'
				/>
				<svg
					className='pointer-events-none absolute inset-0 h-full w-full'
					viewBox='0 0 100 100'
					preserveAspectRatio='none'
				>
					{mapEdges.map((e, i) => {
						// Portal (orange) edges swamp the minimap into unreadability; the
						// doorway affordance already lives in-scene, so keep it off the map.
						if (e.type === "portal") return null;
						const a = ptByIndex.get(e.a);
						const b = ptByIndex.get(e.b);
						if (!a || !b) return null;
						return (
							<line
								key={i}
								x1={a.left}
								y1={a.top}
								x2={b.left}
								y2={b.top}
								stroke={EDGE_META[e.type].color}
								strokeWidth={0.5}
								strokeOpacity={0.5}
								strokeDasharray={
									e.type === "phase" ? "2 2" : undefined
								}
								vectorEffect='non-scaling-stroke'
							/>
						);
					})}
				</svg>
				{view.points.map((pt) => {
					const seen = visitedSet.has(pt.index);
					return (
						<button
							key={pt.index}
							type='button'
							title={pt.name ?? pt.id}
							onClick={() => engine.current?.traverseTo(pt.index)}
							style={{
								left: `${pt.leftPct}%`,
								top: `${pt.topPct}%`,
							}}
							className={`absolute -translate-x-1/2 -translate-y-1/2 rounded-full border transition ${
								pt.current
									? `${expanded ? "h-3.5 w-3.5" : "h-2.5 w-2.5"} border-white bg-cyan-400 ${
											flash
												? "shadow-[0_0_10px_4px_rgba(34,211,238,0.95)]"
												: "shadow-[0_0_6px_2px_rgba(34,211,238,0.7)]"
										}`
									: seen
										? `${expanded ? "h-3 w-3" : "h-2 w-2"} border-white/70 bg-cyan-300/70 hover:scale-125`
										: `${expanded ? "h-3 w-3" : "h-2 w-2"} border-white/50 bg-transparent hover:scale-125 hover:bg-cyan-300/60`
							}`}
						/>
					);
				})}
				{onCurrentFloor && (
					<div
						ref={coneRef}
						className='pointer-events-none absolute'
						style={{
							left: `${view.points.find((p) => p.current)?.leftPct ?? 50}%`,
							top: `${view.points.find((p) => p.current)?.topPct ?? 50}%`,
							width: 0,
							height: 0,
							borderTop: "7px solid transparent",
							borderBottom: "7px solid transparent",
							borderLeft: "13px solid rgba(34,211,238,0.7)",
						}}
					/>
				)}
				{!onCurrentFloor && (
					<div className='pointer-events-none absolute bottom-1 left-1 rounded bg-black/65 px-1 py-0.5 text-[8px] uppercase tracking-wider text-cyan-200'>
						floor {view.level + 1} · you are on {currentLevel + 1}
					</div>
				)}
			</div>
		</div>
	);
}

function modeLabel({ mode, proxyView }: OrbitState): ReactNode {
	if (mode === "overview")
		return proxyView ? (
			<>
				<strong>proxy</strong> · orbit
			</>
		) : (
			<>
				<strong>dollhouse</strong> · orbit
			</>
		);
	if (mode === "interior")
		return proxyView ? (
			<>
				<strong>interior</strong> · proxy
			</>
		) : (
			<>
				<strong>interior</strong> · walkthrough
			</>
		);
	if (mode === "peek")
		return (
			<>
				<strong>locating</strong> · release to return
			</>
		);
	return null;
}

function Hud({ state }: { state: OrbitState }) {
	const content = hudContent(state);
	if (!content) return null;
	return (
		<div className='pointer-events-none absolute bottom-4 left-4 z-10 max-w-[52%] rounded-md border border-white/10 bg-black/60 px-2.5 py-1.5 text-[11px] text-neutral-300 backdrop-blur [&_strong]:font-semibold [&_strong]:text-cyan-200'>
			{content}
		</div>
	);
}

function hudContent(state: OrbitState): ReactNode {
	const {
		mode,
		hover,
		objectHover,
		currentName,
		currentId,
		panoCount,
		tour,
	} = state;
	if (mode === "overview") {
		if (hover)
			return (
				<>
					enter at <strong>{hover.name ?? hover.id}</strong>
				</>
			);
		if (objectHover)
			return (
				<>
					object <strong>{objectHover}</strong> · right-click to hide
					/ outline
				</>
			);
		if (panoCount === 0)
			return (
				<>drag to orbit · scroll to zoom · no walkthrough published</>
			);
		return (
			<>
				drag to orbit · click the scene to <strong>step inside</strong>
			</>
		);
	}
	if (mode === "interior") {
		if (tour)
			return (
				<>
					touring <strong>{tour.zone || "the scene"}</strong> · zone{" "}
					{tour.stop} of {tour.stops} · move or press{" "}
					<strong>stop tour</strong> to take over
				</>
			);
		if (hover)
			return (
				<>
					go to <strong>{hover.name ?? hover.id}</strong>
					{hover.occluded ? " · out of sight" : ""}
				</>
			);
		// Naming what the cursor is resting on takes precedence over the key hints:
		// the outline is already drawn, and the label is what says which thing it is.
		if (objectHover)
			return (
				<>
					<strong>{objectHover}</strong> ·{" "}
					<span className='text-neutral-400'>
						{currentName ?? currentId}
					</span>
				</>
			);
		return (
			<>
				<strong>{currentName ?? currentId}</strong> ·{" "}
				<strong>WASD</strong> move · <strong>Q/E</strong> turn ·{" "}
				<strong>Tab</strong> ping · <strong>M</strong> places
			</>
		);
	}
	if (mode === "peek")
		return (
			<>
				you are <strong>here</strong> · release to drop back in
			</>
		);
	return null;
}
