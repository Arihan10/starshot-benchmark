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
	type ReachPreview,
	type ObjectInspect,
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
	// Which storey's plan the minimap is showing. Lifted out of the minimap because
	// the floor control now lives apart from it and drives it: hovering a floor over
	// there previews its plan over here.
	// Deliberately NOT reset when the pointer leaves the control: the rail and the
	// minimap sit apart, so moving from one to the other would snap the plan back
	// before you reached it. A previewed floor sticks until you preview another or
	// arrive somewhere, and the map's own "you are on N" badge covers the mismatch.
	const [viewedLevel, setViewedLevel] = useState(0);
	const [prevLevel, setPrevLevel] = useState(-1);
	// The splat alignment nudger, off unless asked for — it exists to find a
	// placement, not to be part of the walkthrough.
	const [aligning, setAligning] = useState(false);

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
		state.mode === "freefly" ||
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
	const currentLevel = minimap?.currentLevel ?? -1;
	if (currentLevel !== prevLevel) {
		setPrevLevel(currentLevel);
		setViewedLevel(currentLevel);
	}

	return (
		<div className='relative h-full w-full'>
			<div ref={hostRef} className='absolute inset-0 bg-[#0c0d10]' />

			{mode !== "empty" && mode !== "loading" && (
				<div className='absolute left-4 top-4 z-20 flex flex-col gap-2'>
					{minimap && (
						<Minimap
							minimap={minimap}
							currentIndex={state.currentIndex}
							viewedLevel={viewedLevel}
							engine={engineRef}
						/>
					)}
					<div className='pointer-events-none text-[10px] uppercase tracking-wider text-neutral-400 [&_strong]:font-semibold [&_strong]:text-cyan-200'>
						{modeLabel(state)}
					</div>
				</div>
			)}

			{/* The row wraps and is far wider than the buttons in it, so its empty
			    space used to swallow clicks aimed at whatever lay underneath —
			    notably the expanded minimap's collapse button. Only the buttons
			    themselves take the pointer now. */}
			<div className='pointer-events-none absolute right-4 top-4 z-10 flex flex-wrap justify-end gap-2 [&>*]:pointer-events-auto'>
				{state.canSplatView && (
					<ToolButton
						active={state.splatView}
						activeClass='border-fuchsia-400/70 bg-fuchsia-500/20 text-fuchsia-200'
						title='Show the Gaussian splat instead of the mesh — press WASD inside to fly through it'
						onClick={() => engineRef.current?.toggleSplatView()}
					>
						splat
					</ToolButton>
				)}
				{state.splatTransform && (
					<ToolButton
						active={aligning}
						activeClass='border-amber-400/70 bg-amber-500/20 text-amber-200'
						title='Nudge the splat into register with the scene'
						onClick={() => setAligning((v) => !v)}
					>
						align
					</ToolButton>
				)}
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

			{mode === "interior" && minimap && minimap.levels.length > 1 && (
				<FloorRail
					minimap={minimap}
					visited={state.visited}
					viewedLevel={viewedLevel}
					setViewedLevel={setViewedLevel}
					engine={engineRef}
				/>
			)}

			{mode === "interior" && drawer && (
				<PlacesDrawer
					state={state}
					engine={engineRef}
					onClose={() => setDrawer(false)}
				/>
			)}

			{/* Always mounted, even with nothing to show: it fades and re-targets in
			    place, so re-aiming across a doorway edge can never blink the window
			    out and back. */}
			{aligning && state.splatTransform && (
				<SplatAlign transform={state.splatTransform} engine={engineRef} />
			)}

			<ReachPreviewPanel preview={state.reachPreview} />
			{state.preview && <HoverCard preview={state.preview} />}
			{state.inspect && <InspectFrame inspect={state.inspect} />}
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

// The splat alignment nudger.
//
// A trainer that renormalizes the scene on ingest (Postshot does) returns a splat
// whose origin is nowhere near the world the walkthrough occupies, and no amount
// of arithmetic settles it from the outside: matching bounding boxes cannot tell a
// translation from an axis flip, because both move an AABB's corners the same way.
// Looking at it can. So this exists to FIND the placement by eye — turn the splat
// off and on against the proxy until they register — and then to hand the numbers
// over to the encoder, which bakes them into the asset so that no viewer has to
// carry a correction at all. It is scaffolding, and it should be deleted with the
// offset it was used to find.
//
// The rotation row is not decoration: a 180° flip about an axis is the single most
// likely thing to be wrong here, and it is the one thing translation alone can
// never fix.
function SplatAlign({
	transform,
	engine,
}: {
	transform: NonNullable<OrbitState["splatTransform"]>;
	engine: RefObject<OrbitEngine | null>;
}) {
	const [step, setStep] = useState(1);
	const { position, rotation, scale } = transform;
	const nudge = (axis: 0 | 1 | 2, dir: number) => {
		const next = [...position] as [number, number, number];
		next[axis] = +(next[axis] + dir * step).toFixed(3);
		engine.current?.setSplatTransform({ position: next });
	};
	const spin = (axis: 0 | 1 | 2, deg: number) => {
		const next = [...rotation] as [number, number, number];
		next[axis] = (((next[axis] + deg) % 360) + 360) % 360;
		engine.current?.setSplatTransform({ rotation: next });
	};
	const reset = () =>
		engine.current?.setSplatTransform({
			position: [0, 0, 0],
			rotation: [0, 0, 0],
			scale: 1,
		});
	// What the found placement becomes. World-space, matching the encoder's flags —
	// which do the coordinate-convention conversion internally, so these numbers go
	// in exactly as they read here.
	const cli =
		`--translate ${position.join(",")}` +
		(rotation.some((r) => r !== 0) ? ` --rotate ${rotation.join(",")}` : "") +
		(scale !== 1 ? ` --scale ${scale}` : "");

	return (
		<div className='absolute right-4 top-16 z-30 w-72 rounded-lg border border-amber-400/30 bg-neutral-950/90 p-3 text-xs shadow-2xl backdrop-blur'>
			<div className='mb-2 flex items-center justify-between'>
				<span className='text-[10px] uppercase tracking-wider text-amber-200/80'>
					splat alignment
				</span>
				<button
					type='button'
					onClick={reset}
					className='rounded px-1.5 py-0.5 text-[10px] text-neutral-400 transition hover:bg-white/10 hover:text-white'
				>
					reset
				</button>
			</div>

			{(["x", "y", "z"] as const).map((label, axis) => (
				<div key={label} className='mb-1 flex items-center gap-2'>
					<span className='w-3 text-neutral-500'>{label}</span>
					<button
						type='button'
						onClick={() => nudge(axis as 0 | 1 | 2, -1)}
						className='h-6 w-6 rounded bg-white/10 text-neutral-200 transition hover:bg-white/20'
					>
						−
					</button>
					<input
						value={position[axis]}
						onChange={(e) => {
							const v = Number(e.target.value);
							if (!Number.isFinite(v)) return;
							const next = [...position] as [number, number, number];
							next[axis] = v;
							engine.current?.setSplatTransform({ position: next });
						}}
						className='w-full min-w-0 rounded border border-white/15 bg-black/50 px-1.5 py-0.5 text-right tabular-nums text-neutral-100 outline-none focus:border-amber-400'
					/>
					<button
						type='button'
						onClick={() => nudge(axis as 0 | 1 | 2, 1)}
						className='h-6 w-6 rounded bg-white/10 text-neutral-200 transition hover:bg-white/20'
					>
						+
					</button>
				</div>
			))}

			<div className='mb-2 mt-2 flex items-center gap-2'>
				<span className='text-[10px] uppercase tracking-wider text-neutral-500'>
					step
				</span>
				{[0.1, 1, 5].map((s) => (
					<button
						key={s}
						type='button'
						onClick={() => setStep(s)}
						className={`rounded px-1.5 py-0.5 tabular-nums transition ${
							step === s
								? "bg-amber-400/25 text-amber-100"
								: "text-neutral-400 hover:bg-white/10"
						}`}
					>
						{s}
					</button>
				))}
			</div>

			<div className='mb-2 flex items-center gap-2'>
				<span className='text-[10px] uppercase tracking-wider text-neutral-500'>
					flip
				</span>
				{(["x", "y", "z"] as const).map((label, axis) => (
					<button
						key={label}
						type='button'
						title={`rotate 180° about ${label}`}
						onClick={() => spin(axis as 0 | 1 | 2, 180)}
						className={`rounded px-1.5 py-0.5 transition ${
							rotation[axis] !== 0
								? "bg-amber-400/25 text-amber-100"
								: "text-neutral-400 hover:bg-white/10"
						}`}
					>
						{label}180
					</button>
				))}
			</div>

			<code className='block w-full overflow-x-auto whitespace-nowrap rounded bg-black/60 px-2 py-1 text-[10px] text-emerald-300/90'>
				{cli}
			</code>
			<p className='mt-1.5 text-[10px] leading-snug text-neutral-500'>
				Bake with{" "}
				<span className='text-neutral-400'>splat-to-web-sog.mjs</span>, then
				clear this offset.
			</p>
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

// The panel takes the hue of whatever opened it, so it always reads as an
// extension of that affordance rather than as a third thing.
const REACH_ACCENT = "#ffc46b"; // amber: out of sight on THIS floor
const FLOOR_ACCENT = "#7ef2c2"; // green: a change of storey
// Viewport-scaled, like the minimap: docked in a corner it can afford real size,
// and a preview of a room you cannot see is the one thing here worth looking at.
const PANO_SCREEN_W = "clamp(420px, 36vw, 760px)";
const PANO_SCREEN_ASPECT = 2.4; // the "screen" is sized from the width, not fixed
// One full 360 is drawn this many times the screen's own width, so only a slice of
// the panorama is visible at once — 1/4.5, about 80° across — and the frame reads as
// a wide view INTO the room rather than a whole equirect flattened out. Expressed as
// a RATIO rather than a pixel width so the visible angle is the same at every panel
// size: scale the panel and you get more pixels, not more panorama. The pan shifts
// by exactly one tile (in cqw, against the card), which is what makes the wrap
// invisible. Equirects are linear in angle, so the slice keeps the screen's aspect
// and nothing is stretched.
const PANO_TILE_RATIO = 4.5;

const PANO_PAN_MS = 28000; // one full revolution of the panning 360
const REACH_DOCK_INSET = 16; // px from the top-right corner when docked
const REACH_SLIDE_MS = 300; // the docked slide in / out
const REACH_XFADE_MS = 260; // dissolve between two destinations

// One panorama layer inside the panel. Layers stack and cross-dissolve, so `key`
// has to be unique per *appearance* rather than per pano — re-aiming back at a
// destination you just left is a new dissolve, not the same element.
type ReachLayer = {
	key: number;
	url: string;
	placeholderUrl: string;
	panPhase: number; // ms into pano-pan when this layer was born
};

// The floor-change preview: a wide "screen" that pans continuously through the
// destination panorama rather than squashing a whole equirect into one frame, so
// you read the room you're about to drop into instead of a warped strip. The pano
// is tiled horizontally and shifted by exactly one tile width (see the pano-pan
// keyframes), which is what makes the 360 loop seamlessly. Pointer-transparent, so
// it never swallows the click it is previewing.
//
// It RIDES THE CURSOR, and it is NEVER UNMOUNTED — it fades to nothing instead.
// Both matter for the same reason: re-aiming the cursor changes the destination
// continuously, so anything that rebuilt the window would make it strobe as you
// swept across a room. Position is written as a transform straight onto the node
// rather than through state (same reason the minimap cone and exit arrows are
// driven that way: a re-render per mousemove would change nothing visually and
// cost everything), and a new destination fades in OVER the outgoing one.
// Positioned in VIEWPORT coords, so it stays correct in the side-by-side
// workspace where the panel is not at the window origin.
function ReachPreviewPanel({ preview }: { preview: ReachPreview | null }) {
	const ref = useRef<HTMLDivElement>(null);
	// The last destination worth showing. The panel keeps rendering it while it
	// fades out, so dismissal is a fade rather than an abrupt blanking.
	const [shown, setShown] = useState<ReachPreview | null>(null);
	const [layers, setLayers] = useState<ReachLayer[]>([]);

	// A new destination pushes a layer that fades in OVER the outgoing one, and
	// `shown` keeps the caption alive through the fade-out. Adjusted during render
	// rather than in an effect (the pattern the minimap's arrival flash uses), so
	// the panel never paints a frame of stale content — and guarded on the
	// destination, so it settles immediately instead of looping.
	if (preview && preview.index !== shown?.index) {
		setShown(preview);
		setLayers((prev) => {
			const last = prev[prev.length - 1];
			if (last && last.url === preview.url) return prev;
			return [
				...prev,
				{
					key: (last?.key ?? 0) + 1,
					url: preview.url,
					placeholderUrl: preview.placeholderUrl,
					// Shared clock, so the incoming layer pans in step with the one it
					// is covering: the dissolve changes the room, not the framing.
					panPhase: performance.now() % PANO_PAN_MS,
				},
			].slice(-2);
		});
	}

	// One place, whoever asked. Both the cursor's out-of-sight previews and the
	// green floor arrows' dock to the same corner and slide in from off-frame: WHERE
	// the panel appears should not depend on which affordance summoned it, or the eye
	// has to re-find it every time. Only the accent differs, tying it back to the
	// thing that opened it.
	//
	// Nothing here writes to the DOM directly, so nothing races React for the
	// transform — the slide is a plain style toggle on `open`.

	// Drop the outgoing layer once it is fully covered.
	useEffect(() => {
		if (layers.length < 2) return;
		const t = setTimeout(
			() => setLayers((l) => l.slice(-1)),
			REACH_XFADE_MS + 40,
		);
		return () => clearTimeout(t);
	}, [layers]);

	const open = !!preview;
	const level = shown ? shown.level + 1 : 1;
	const delta = shown?.levelDelta ?? 0;
	// Keyed on the move, not on what opened the panel: green whenever the click
	// changes storey, amber when it only goes somewhere you cannot see on this one.
	const accent = delta !== 0 ? FLOOR_ACCENT : REACH_ACCENT;

	return (
		<div
			ref={ref}
			aria-hidden={!open}
			className='pointer-events-none fixed z-30 will-change-transform'
			style={{
				width: PANO_SCREEN_W,
				right: REACH_DOCK_INSET,
				top: REACH_DOCK_INSET,
				opacity: open ? 1 : 0,
				// Slides clear of its own corner when closed, so it leaves the way it
				// arrived rather than blinking out.
				transform: open
					? "translateX(0)"
					: `translateX(calc(100% + ${REACH_DOCK_INSET * 2}px))`,
				transition: `transform ${REACH_SLIDE_MS}ms cubic-bezier(0.22, 0.61, 0.36, 1), opacity ${REACH_SLIDE_MS}ms ease-out`,
			}}
		>
			<div
				className='overflow-hidden rounded-xl bg-neutral-950/85 shadow-2xl backdrop-blur'
				style={{
					border: `1px solid ${accent}80`,
					// Everything inside sizes off the card's own width, so the panorama
					// slice and the caption stay in proportion at any panel size.
					containerType: "inline-size",
				}}
			>
				<div
					className='relative overflow-hidden bg-neutral-900'
					style={{ aspectRatio: PANO_SCREEN_ASPECT }}
				>
					{layers.map((layer) => (
						<div
							key={layer.key}
							className='absolute inset-0'
							style={
								{
									// Full pano over the blurred placeholder, so the frame is
									// filled the instant it opens and both pan together.
									backgroundImage: `url(${layer.url}), url(${layer.placeholderUrl})`,
									backgroundRepeat: "repeat-x",
									backgroundSize: `${PANO_TILE_RATIO * 100}% auto`,
									backgroundPositionY: "50%",
									"--pano-tile": `${PANO_TILE_RATIO * 100}cqw`,
									animationName: "pano-pan, reach-layer-in",
									animationDuration: `${PANO_PAN_MS}ms, ${REACH_XFADE_MS}ms`,
									animationTimingFunction: "linear, ease-out",
									animationIterationCount: "infinite, 1",
									animationFillMode: "none, forwards",
									// Negative delay starts pano-pan mid-cycle, matching the
									// layer below it.
									animationDelay: `-${layer.panPhase}ms, 0ms`,
								} as CSSProperties
							}
						/>
					))}
					<div className='pointer-events-none absolute inset-x-0 top-0 h-14 bg-linear-to-b from-black/70 to-transparent' />
					<span
						className='absolute left-3 top-3 rounded px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-neutral-950'
						style={{ background: accent }}
					>
						{delta === 0
							? "out of sight"
							: `${delta > 0 ? "▲" : "▼"} floor ${level}`}
					</span>
				</div>
				<div
					className='flex items-center justify-between gap-2 px-3 py-2'
					style={{ fontSize: "clamp(11px, 2.6cqw, 17px)" }}
				>
					<span className='min-w-0 truncate font-semibold text-white'>
						{shown?.name ?? "unnamed"}
					</span>
					<span
						className='shrink-0 text-[0.8em] uppercase tracking-wider'
						style={{ color: accent }}
					>
						click to go
						{shown && shown.dist < 100
							? ` · ${shown.dist.toFixed(0)} m`
							: ""}
					</span>
				</div>
			</div>
		</div>
	);
}

// The floor control. Inter-floor travel is the one move the scene itself cannot
// offer — the floor you stand on hides everything below it, and a generated scene
// cannot be assumed to model stairs — so the control that CAN offer it gets a
// corner to itself rather than a row of digits tucked into the minimap's header,
// where it read as a view toggle and was easy to never notice.
//
// It rests as a bare column of storey numbers and opens on hover into the full
// list with names. That split is the point: at rest it costs the 3D almost
// nothing, and it only takes the room needed to read a floor name at the moment
// you are actually reading one.
//
// Laid out HIGHEST STOREY AT THE TOP (flex-col-reverse), so the control is a
// section through the building: moving up the list moves up the scene. Hover a row
// to preview that floor's plan in the minimap, click to go. Unvisited floors
// breathe (the floor-unvisited keyframes) until you have actually stood on them.
//
// Anchored bottom-RIGHT (the exits panel's old corner, and the only free one —
// the HUD holds bottom-left), so the pinned edge is the right one and opening
// grows the panel leftward and upward: out of its corner, never across the middle
// of the view.
// Collapsed width is the chip (32) + the button's own padding (12) + the panel's
// (24). That panel padding is deliberately generous: the unvisited pulse is a
// box-shadow reaching ~11px past each button, and the panel clips its overflow to
// keep the names from spilling out while narrow — too tight a padding and the glow
// would be shaved off exactly where it is meant to draw the eye.
const RAIL_COLLAPSED = "w-[68px]";
const RAIL_EXPANDED = "w-72";

function FloorRail({
	minimap,
	visited,
	viewedLevel,
	setViewedLevel,
	engine,
}: {
	minimap: NonNullable<OrbitState["minimap"]>;
	visited: number[];
	viewedLevel: number;
	setViewedLevel: (level: number) => void;
	engine: RefObject<OrbitEngine | null>;
}) {
	const [open, setOpen] = useState(false);
	const { currentLevel, levels } = minimap;
	const visitedSet = useMemo(() => new Set(visited), [visited]);
	const unvisited = useMemo(() => {
		const out = new Set<number>();
		for (const lv of levels)
			if (!lv.points.some((p) => visitedSet.has(p.index))) out.add(lv.level);
		return out;
	}, [levels, visitedSet]);

	return (
		<div
			onPointerEnter={() => setOpen(true)}
			onPointerLeave={() => setOpen(false)}
			className={`absolute bottom-4 right-4 z-10 overflow-hidden rounded-lg border border-white/10 bg-black/60 p-3 backdrop-blur transition-[width] duration-200 ease-out ${
				open ? RAIL_EXPANDED : RAIL_COLLAPSED
			}`}
		>
			<div
				className={`overflow-hidden whitespace-nowrap text-[10px] uppercase tracking-wider text-neutral-400 transition-all duration-200 ${
					open ? "mb-1.5 h-4 opacity-100" : "mb-0 h-0 opacity-0"
				}`}
			>
				floors
			</div>
			<div className='flex flex-col-reverse gap-0.5'>
				{levels.map((lv) => {
					const isCurrent = lv.level === currentLevel;
					const isViewed = lv.level === viewedLevel;
					// Viewing is not visiting: the pulse holds until you go.
					const isUnvisited = unvisited.has(lv.level);
					return (
						<button
							key={lv.level}
							type='button'
							title={
								isCurrent
									? "you are on this floor"
									: `go to ${lv.name ?? `floor ${lv.level + 1}`}`
							}
							onPointerEnter={() => setViewedLevel(lv.level)}
							onFocus={() => {
								setOpen(true);
								setViewedLevel(lv.level);
							}}
							onClick={() => engine.current?.jumpToLevel(lv.level)}
							className={`flex items-center gap-2.5 overflow-hidden whitespace-nowrap rounded-md p-1.5 text-left transition ${
								isUnvisited
									? "animate-[floor-unvisited_2s_ease-in-out_infinite] text-amber-100"
									: isCurrent
										? "bg-cyan-500/20 text-cyan-100"
										: "text-neutral-300 hover:bg-white/10 hover:text-white"
							} ${
								isViewed && !isCurrent
									? "ring-1 ring-inset ring-white/25"
									: ""
							}`}
						>
							<span
								className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-[13px] font-semibold tabular-nums ${
									isCurrent
										? "bg-cyan-400 text-neutral-950"
										: "bg-white/10"
								}`}
							>
								{lv.level + 1}
							</span>
							<span
								className={`min-w-0 flex-1 truncate text-[12px] transition-opacity duration-150 ${
									open ? "opacity-100" : "opacity-0"
								}`}
							>
								{lv.name ?? `Floor ${lv.level + 1}`}
							</span>
							{isCurrent && (
								<span
									className={`shrink-0 text-[9px] uppercase tracking-wider text-cyan-300/80 transition-opacity duration-150 ${
										open ? "opacity-100" : "opacity-0"
									}`}
								>
									here
								</span>
							)}
						</button>
					);
				})}
			</div>
		</div>
	);
}

// The dwell inspection: a frame and a caption around the rectangle the ENGINE is
// drawing the orbiting object into (a scissored viewport of the main canvas, see
// renderInspect). Deliberately has no background of its own — anything opaque here
// would paint over the 3D underneath it. Pointer-transparent, so resting the cursor
// to summon it never blocks the click that follows.
function InspectFrame({ inspect }: { inspect: ObjectInspect }) {
	return (
		<div
			className='pointer-events-none fixed z-30'
			style={{ left: inspect.x, top: inspect.y, width: inspect.w, height: inspect.h }}
		>
			<div className='h-full w-full rounded-lg border border-white/25 shadow-2xl' />
			<div className='absolute inset-x-0 -bottom-6 truncate rounded bg-black/75 px-2 py-1 text-center text-[10px] font-medium text-neutral-100 backdrop-blur'>
				{prettyLabel(inspect.label)}
			</div>
		</div>
	);
}

// `antique_display_sextant` → "Antique display sextant". The ids are authored by
// the pipeline, so they read as words already — they just need unpicking.
function prettyLabel(id: string): string {
	const words = id.replace(/[_-]+/g, " ").trim();
	return words ? words[0].toUpperCase() + words.slice(1) : id;
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
									<span className='text-neutral-500'> · {n.zone}</span>
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
		return <div className='px-2 py-1.5 text-xs text-neutral-500'>no zones</div>;
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
					<span className='min-w-0 truncate'>{c.zone || "unzoned"}</span>
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
				<MenuButton onClick={() => engine.current?.toggleMenuTargetHidden()}>
					{menu.hidden ? "show" : "hide"}
				</MenuButton>
			)}
			{menu.label && (
				<MenuButton onClick={() => engine.current?.toggleMenuTargetOutline()}>
					{menu.outlined ? "remove outline" : "highlight outline"}
				</MenuButton>
			)}
			{menu.label && hasExtras && <div className='my-1 h-px bg-white/10' />}
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

// The you-are-here view cone. Half-angle roughly matches what the walkthrough
// actually shows, and the span is a multiple of the panel width so the gradient
// fades out well before the element does — the cone reads as scattering off into
// the distance rather than stopping at a boundary.
const CONE_HALF_ANGLE = 33; // degrees either side of the facing direction
const CONE_SPAN = 260; // % of the panel width; the wedge is square in px
// Apex at the centre, opening toward +X (rotation 0 = world +X = map right, which
// is the convention `facingDeg` is already in).
const CONE_CLIP = `polygon(50% 50%, 100% ${
	50 - 50 * Math.tan((CONE_HALF_ANGLE * Math.PI) / 180)
}%, 100% ${50 + 50 * Math.tan((CONE_HALF_ANGLE * Math.PI) / 180)}%)`;

// The map is fitted inside a box: it takes the smaller of the width cap and the
// width its own aspect implies from the height cap, so a tall floor plan and a wide
// one both sit inside the same envelope without either being cropped or stretched.
const minimapWidth = (aspect: number, maxW: string, maxH: string) =>
	`min(${maxW}, calc(${maxH} * ${aspect}))`;

// Both envelopes are viewport-relative rather than fixed, so the map keeps its
// share of the frame instead of shrinking into irrelevance on a large display and
// crowding out the scene on a small one. Clamped at both ends: below the floor the
// zone names stop being readable, and above the ceiling a "small" map is no longer
// small. Note vw/vh are WINDOW units, so in the side-by-side workspace both viewers
// size to the window rather than to their own half — which is what you want, since
// entering one expands it to the full width anyway.
const MINIMAP_COMPACT = {
	w: "clamp(240px, 24vw, 460px)",
	h: "clamp(200px, 26vh, 400px)",
};
const MINIMAP_EXPANDED = {
	w: "clamp(420px, 56vw, 900px)",
	h: "clamp(340px, 72vh, 820px)",
};

// Layer 3: the bird's-eye minimap — the storey's slice, the zone names printed on
// it, a live you-are-here facing cone, and a re-anchoring flash after arrival.
function Minimap({
	minimap,
	currentIndex,
	viewedLevel,
	engine,
}: {
	minimap: NonNullable<OrbitState["minimap"]>;
	currentIndex: number;
	// Read-only here: the floor rail owns which storey is being shown.
	viewedLevel: number;
	engine: RefObject<OrbitEngine | null>;
}) {
	const [expanded, setExpanded] = useState(false);
	const { currentLevel, levels } = minimap;

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
	if (!view) return null;
	const caps = expanded ? MINIMAP_EXPANDED : MINIMAP_COMPACT;
	const onCurrentFloor = view.level === currentLevel;

	return (
		<div className='rounded-md border border-white/10 bg-black/60 p-1.5 backdrop-blur'>
			<div className='mb-1 flex items-center justify-between gap-2 px-0.5 text-[9px] uppercase tracking-wider text-neutral-400'>
				<span className='truncate'>
					{view.name ?? `floor ${view.level + 1}`}
				</span>
				<div className='flex items-center gap-1'>
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
			{/* The whole slice is a travel surface: click anywhere and you go to the
			    nearest capture to that spot on the floor being shown. The map already
			    reads as a plan of somewhere you can be, so making only the labels
			    clickable meant most of it looked live and wasn't. */}
			<div
				onClick={(e) => {
					const r = e.currentTarget.getBoundingClientRect();
					if (!r.width || !r.height) return;
					const left = ((e.clientX - r.left) / r.width) * 100;
					const top = ((e.clientY - r.top) / r.height) * 100;
					// Nearest capture in PERCENT space, with the horizontal axis
					// weighted by the slice's aspect. The two axes span different
					// numbers of metres, so comparing raw percentages would bias the
					// pick along whichever axis the floor happens to be longer in;
					// scaling by width/depth restores the true ordering without the
					// state needing to carry world bounds at all.
					let index = -1;
					let best = Infinity;
					for (const pt of view.points) {
						const dx = (pt.leftPct - left) * view.aspect;
						const dy = pt.topPct - top;
						const d = dx * dx + dy * dy;
						if (d < best) {
							best = d;
							index = pt.index;
						}
					}
					if (index >= 0) engine.current?.traverseTo(index);
				}}
				title='click anywhere to travel there'
				className='relative cursor-pointer overflow-hidden rounded'
				style={{
					width: minimapWidth(view.aspect, caps.w, caps.h),
					aspectRatio: view.aspect,
					// Names size off the MAP, not the window: the box is aspect-fitted
					// inside its envelope, so its real width is often the height cap
					// times the floor's aspect and has no fixed relation to vw. A query
					// container lets the labels read that resolved width directly, and
					// keeps them proportional through the expand as well.
					containerType: "inline-size",
				}}
			>
				{/* Blown up and offset so the storey's own crop rect exactly fills the
				    box; the container clips the rest. Same stretch-to-fit semantics as
				    the old object-fill, just applied to a sub-rectangle. */}
				{/* eslint-disable-next-line @next/next/no-img-element -- runtime R2 slice via /r2 proxy */}
				<img
					src={view.url}
					alt='scene from above'
					draggable={false}
					className='absolute max-w-none'
					style={{
						width: `${100 / (view.crop.u1 - view.crop.u0)}%`,
						height: `${100 / (view.crop.v1 - view.crop.v0)}%`,
						left: `${(-100 * view.crop.u0) / (view.crop.u1 - view.crop.u0)}%`,
						top: `${(-100 * view.crop.v0) / (view.crop.v1 - view.crop.v0)}%`,
					}}
				/>
				{onCurrentFloor && (
					<div
						ref={coneRef}
						className='pointer-events-none absolute'
						style={{
							left: `${view.points.find((p) => p.current)?.leftPct ?? 50}%`,
							top: `${view.points.find((p) => p.current)?.topPct ?? 50}%`,
							// A view cone cast from the marker rather than a glyph beside
							// it. The old solid triangle stated a length the camera does
							// not have — sight runs until something stops it — so this is
							// a wedge whose light simply thins out with distance and never
							// resolves into an edge. Sized well past the panel so the
							// falloff, not the element, is what ends it.
							width: `${CONE_SPAN}%`,
							aspectRatio: 1,
							clipPath: CONE_CLIP,
							background:
								"radial-gradient(circle at 50% 50%, rgba(214,222,232,0.50) 0%, rgba(214,222,232,0.28) 18%, rgba(214,222,232,0.10) 42%, rgba(214,222,232,0) 72%)",
							// Screen, so the cone BRIGHTENS the plan under it instead of
							// laying an opaque wash over it — the floor stays readable
							// through the thing describing where you are looking.
							mixBlendMode: "screen",
							transform: "translate(-50%, -50%)",
						}}
					/>
				)}
				{/* Zone NAMES, not a web of dots. The dots were one marker per
				    capture with a line to every neighbour — on a 200px map that is a
				    tangle nobody can read a room out of. A handful of names answers
				    the only question the map is actually asked ("what is over
				    there?"), and clicking one travels to the nearest capture in it.
				    Which zones get named is decided upstream by the map labeller, so
				    no label ever sits inside another (see anchors.py). */}
				{view.labels.map((lab) => (
					<button
						key={lab.id}
						type='button'
						title={`go to ${lab.label}`}
						onClick={(e) => {
							e.stopPropagation(); // the surface below would re-resolve it
							engine.current?.traverseTo(lab.index);
						}}
						style={{
							left: `${lab.leftPct}%`,
							top: `${lab.topPct}%`,
							// Floors so a crowded compact map stays legible, ceiling so an
							// expanded one doesn't turn into signage.
							fontSize: "clamp(9px, 4.2cqw, 17px)",
							// A slice is a lit render, so a name over it needs its own
							// contrast rather than borrowing the map's.
							textShadow: "0 1px 3px rgba(0,0,0,0.95), 0 0 8px rgba(0,0,0,0.8)",
						}}
						className='absolute -translate-x-1/2 -translate-y-1/2 whitespace-nowrap rounded px-1 py-0.5 text-center font-medium leading-tight text-white/95 transition hover:bg-cyan-400/25 hover:text-white'
					>
						{lab.label}
					</button>
				))}
				{/* Only the capture you are standing on keeps a dot: it is the one
				    thing on the map that is about you rather than about the scene. */}
				{view.points
					.filter((pt) => pt.current)
					.map((pt) => (
						<span
							key={pt.index}
							style={{ left: `${pt.leftPct}%`, top: `${pt.topPct}%` }}
							className={`pointer-events-none absolute -translate-x-1/2 -translate-y-1/2 rounded-full border border-white bg-cyan-400 ${
								expanded ? "h-3.5 w-3.5" : "h-2.5 w-2.5"
							} ${
								flash
									? "shadow-[0_0_10px_4px_rgba(34,211,238,0.95)]"
									: "shadow-[0_0_6px_2px_rgba(34,211,238,0.7)]"
							}`}
						/>
					))}
				{!onCurrentFloor && (
					<div className='pointer-events-none absolute bottom-1 left-1 rounded bg-black/65 px-1 py-0.5 text-[8px] uppercase tracking-wider text-cyan-200'>
						floor {view.level + 1} · you are on {currentLevel + 1}
					</div>
				)}
			</div>
		</div>
	);
}

function modeLabel({ mode, proxyView, splatView }: OrbitState): ReactNode {
	if (mode === "freefly")
		return (
			<>
				<strong>free flight</strong> · splat
			</>
		);
	if (mode === "overview")
		return splatView ? (
			<>
				<strong>splat</strong> · orbit
			</>
		) : proxyView ? (
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
		return (
			<>
				<strong>{currentName ?? currentId}</strong> ·{" "}
				<strong>WASD</strong> move · <strong>Space/Shift</strong> fly ·{" "}
				<strong>Tab</strong> ping · <strong>M</strong> places
				{state.mouseLook ? (
					<>
						{" "}
						· <strong>Esc</strong> for cursor
					</>
				) : (
					<>
						{" "}
						· <strong>click</strong> to look
					</>
				)}
			</>
		);
	}
	if (mode === "freefly")
		return (
			<>
				<strong>WASD</strong> fly · <strong>Space/Shift</strong> up/down ·{" "}
				<strong>wheel</strong> speed{" "}
				<strong>{state.freeflySpeed.toFixed(2)}×</strong> ·{" "}
				<strong>click</strong> or <strong>stop</strong> near a viewpoint to land ·{" "}
				<strong>Esc</strong> {state.mouseLook ? "for cursor" : "back"}
				{/* TEMPORARY: readout for the settle delay while it is tuned by feel. */}
				<span className='text-neutral-500'>
					{" "}
					· settle <strong>{state.dockDelayMs}ms</strong> ([ / ])
				</span>
			</>
		);
	if (mode === "peek")
		return (
			<>
				you are <strong>here</strong> · release to drop back in
			</>
		);
	return null;
}
