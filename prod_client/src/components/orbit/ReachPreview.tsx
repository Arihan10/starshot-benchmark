"use client";

import { useEffect, useState, type CSSProperties } from "react";
import type { ReachPreview as ReachPreviewState } from "@/lib/orbit/types";
import { storey } from "./labels";

// The panel takes the hue of whatever opened it, so it always reads as an
// extension of that affordance rather than as a third thing.
const REACH_ACCENT = "#ffc46b"; // amber: out of sight on THIS floor
const FLOOR_ACCENT = "#7ef2c2"; // green: a change of storey

// Viewport-scaled: docked in a corner it can afford real size, and a preview of a
// room you cannot see is the one thing here worth looking at.
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

/**
 * The out-of-sight preview: a wide "screen" that pans continuously through the
 * destination panorama rather than squashing a whole equirect into one frame, so
 * you read the room you're about to drop into instead of a warped strip. The pano
 * is tiled horizontally and shifted by exactly one tile width (see the pano-pan
 * keyframes), which is what makes the 360 loop seamlessly. Pointer-transparent, so
 * it never swallows the click it is previewing.
 *
 * It is NEVER UNMOUNTED — it fades to nothing instead. Re-aiming the cursor
 * changes the destination continuously, so anything that rebuilt the window would
 * make it strobe as you swept across a room; a new destination fades in OVER the
 * outgoing one.
 *
 * It docks to ONE corner whoever asked. Both the cursor's out-of-sight previews
 * and the floor arrows land in the same place and slide in from off-frame: WHERE
 * the panel appears should not depend on which affordance summoned it, or the eye
 * has to re-find it every time. Only the accent differs, tying it back to the
 * thing that opened it. Positioned in VIEWPORT coords, so it stays correct in the
 * side-by-side workspace where the panel is not at the window origin.
 */
export default function ReachPreview({
	preview,
	levelWord,
}: {
	preview: ReachPreviewState | null;
	levelWord: string;
}) {
	// The last destination worth showing. The panel keeps rendering it while it
	// fades out, so dismissal is a fade rather than an abrupt blanking.
	const [shown, setShown] = useState<ReachPreviewState | null>(null);
	const [layers, setLayers] = useState<ReachLayer[]>([]);

	// A new destination pushes a layer that fades in OVER the outgoing one, and
	// `shown` keeps the caption alive through the fade-out. Adjusted during render
	// rather than in an effect, so the panel never paints a frame of stale content —
	// and guarded on the destination, so it settles immediately instead of looping.
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

	// Drop the outgoing layer once it is fully covered.
	useEffect(() => {
		if (layers.length < 2) return;
		const t = setTimeout(() => setLayers((l) => l.slice(-1)), REACH_XFADE_MS + 40);
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
							: `${delta > 0 ? "▲" : "▼"} ${storey(levelWord, level - 1)}`}
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
						{shown && shown.dist < 100 ? ` · ${shown.dist.toFixed(0)} m` : ""}
					</span>
				</div>
			</div>
		</div>
	);
}
