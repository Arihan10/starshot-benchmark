"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import type { OrbitEngine } from "@/lib/orbit/engine";
import type { OrbitState } from "@/lib/orbit/types";
import { storey } from "./labels";

// Drive a callback from the engine's live facing (deg) each frame WITHOUT React
// re-renders. Only the cone needs this, and it needs it every frame — a re-render
// per mousemove would change nothing visually and cost everything.
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
const MINIMAP_COMPACT = { w: "calc(var(--spacing-xl) * 7.9)", h: "calc(var(--spacing-xl) * 6.9)" };
const MINIMAP_EXPANDED = { w: "clamp(420px, 56vw, 900px)", h: "clamp(340px, 72vh, 820px)" };

// The bird's-eye minimap — the storey's slice, the zone names printed on it, a
// live you-are-here facing cone, and a re-anchoring flash after arrival.
export default function Minimap({
	minimap,
	currentIndex,
	viewedLevel,
	levelWord,
	engine,
}: {
	minimap: NonNullable<OrbitState["minimap"]>;
	currentIndex: number;
	// Read-only here: the floor rail owns which storey is being shown.
	viewedLevel: number;
	levelWord: string;
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

	// The whole slice is a travel surface: click anywhere and you go to the nearest
	// capture to that spot on the floor being shown. The map already reads as a plan
	// of somewhere you can be, so making only the labels clickable meant most of it
	// looked live and wasn't.
	const travelToPoint = (e: React.MouseEvent<HTMLDivElement>) => {
		const r = e.currentTarget.getBoundingClientRect();
		if (!r.width || !r.height) return;
		const left = ((e.clientX - r.left) / r.width) * 100;
		const top = ((e.clientY - r.top) / r.height) * 100;
		// Nearest capture in PERCENT space, with the horizontal axis weighted by the
		// slice's aspect. The two axes span different numbers of metres, so comparing
		// raw percentages would bias the pick along whichever axis the floor happens
		// to be longer in; scaling by width/depth restores the true ordering without
		// the state needing to carry world bounds at all.
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
	};

	return (
		<div className='rounded-md border border-mark-8 bg-ground/60 p-1.5 backdrop-blur'>
			<div className='mb-1 flex items-center justify-between gap-2 px-0.5 text-[9px] uppercase tracking-wider text-ink-64'>
				<span className='truncate'>{view.name ?? storey(levelWord, view.level)}</span>
				<button
					type='button'
					onClick={() => setExpanded((v) => !v)}
					title={expanded ? "collapse minimap" : "expand minimap"}
					aria-label={expanded ? "collapse minimap" : "expand minimap"}
					className='rounded px-1 text-[11px] leading-none text-ink-64 transition hover:bg-mark-8 hover:text-ink'
				>
					{expanded ? "✕" : "⤢"}
				</button>
			</div>
			<div
				onClick={travelToPoint}
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
				    box; the container clips the rest. The crop is recomputed only on
				    arrival, never per frame — but a map that FOLLOWS you would jump on
				    every hop if it cut straight to the new window. Transitioning the four
				    values it is expressed as gives a smooth pan for free, with no extra
				    state and no per-frame work. */}
				{/* eslint-disable-next-line @next/next/no-img-element -- runtime R2 slice via /r2 proxy */}
				<img
					src={view.url}
					alt='scene from above'
					draggable={false}
					className='absolute max-w-none'
					style={{
						transition:
							"width 320ms ease-out, height 320ms ease-out, left 320ms ease-out, top 320ms ease-out",
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
							// it. A solid triangle would state a length the camera does not
							// have — sight runs until something stops it — so this is a
							// wedge whose light simply thins out with distance and never
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
				{/* Zone NAMES, not a web of dots. One marker per capture with a line to
				    every neighbour is, on a 200px map, a tangle nobody can read a room
				    out of. A handful of names answers the only question the map is
				    actually asked ("what is over there?"), and clicking one travels to
				    the nearest capture in it. Which zones get named is decided upstream
				    by the map labeller, so no label ever sits inside another. */}
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
							fontSize: "var(--text-xs)",
							// A slice is a lit render, so a name over it needs its own
							// contrast rather than borrowing the map's.
							textShadow: "0 1px 3px rgba(0,0,0,0.95), 0 0 8px rgba(0,0,0,0.8)",
						}}
						className='absolute -translate-x-1/2 -translate-y-1/2 whitespace-nowrap rounded px-1 py-0.5 text-center font-medium leading-tight text-ink transition hover:bg-cyan-400/25 hover:text-ink'
					>
						{lab.label}
					</button>
				))}
				{/* Only the capture you are standing on keeps a dot: it is the one thing
				    on the map that is about you rather than about the scene. */}
				{view.points
					.filter((pt) => pt.current)
					.map((pt) => (
						<span
							key={pt.index}
							style={{ left: `${pt.leftPct}%`, top: `${pt.topPct}%` }}
							className={`pointer-events-none absolute -translate-x-1/2 -translate-y-1/2 rounded-full border border-mark bg-cyan-400 ${
								expanded ? "h-3.5 w-3.5" : "h-2.5 w-2.5"
							} ${
								flash
									? "shadow-[0_0_10px_4px_rgba(34,211,238,0.95)]"
									: "shadow-[0_0_6px_2px_rgba(34,211,238,0.7)]"
							}`}
						/>
					))}
				{!onCurrentFloor && (
					<div className='pointer-events-none absolute bottom-1 left-1 rounded bg-ground/65 px-1 py-0.5 text-[8px] uppercase tracking-wider text-cyan-200'>
						{storey(levelWord, view.level)} · you are on {currentLevel + 1}
					</div>
				)}
			</div>
		</div>
	);
}
