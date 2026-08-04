"use client";

import { useCallback, useEffect, useRef } from "react";
import { linear, useProgress } from "./useProgress";

// The clock does not start with the reveal. The ratings are still counting for the
// first two seconds and the crowd's percentage lands at 2.6 — a bar already
// draining through all of that reads as a deadline on reading the result. It
// starts once the result has finished telling itself.
const LEAD_MS = 2700;
const RUN_MS = 5200;

/**
 * The wait before the next pair, in the slot the SKIP button just left.
 *
 * A COUNTDOWN, NOT A BUTTON — but still clickable. Once you have voted there is
 * exactly one thing left to happen and no decision left to make, so making the
 * viewer ask for it is ceremony; the bar simply shows how long the result stays
 * up. Clicking is for the reader who is already done, and is the same action the
 * clock is going to take anyway.
 */
export default function NextTimer({
	onNext,
	paused = false,
}: {
	onNext: () => void;
	/**
	 * Someone has stepped inside one of the scenes. The clock holds: advancing
	 * would swap the round out from under a viewer who is standing in it, and the
	 * one thing they cannot see right now is this bar.
	 */
	paused?: boolean;
}) {
	const t = useProgress(RUN_MS, LEAD_MS, linear);

	// ONCE PER ROUND, whichever gets there first.
	//
	// The clock and the button are two routes to the same single event, and both
	// were live at the same time: pressing "next" a moment before the countdown
	// ended asked for the next pair, and then the timeout — still pending, because
	// it is only cleared when this unmounts, which is a beat after the change
	// starts — asked for it AGAIN. Two advances land two rounds on, and with a
	// short list that wraps straight back to the pair just left: the result clears,
	// the moon turns, and the same two scenes are still sitting there.
	const fired = useRef(false);
	const advance = useCallback(() => {
		if (fired.current) return;
		fired.current = true;
		onNext();
	}, [onNext]);

	useEffect(() => {
		if (paused) return;
		const timer = window.setTimeout(advance, LEAD_MS + RUN_MS);
		return () => window.clearTimeout(timer);
	}, [advance, paused]);

	return (
		<button
			type="button"
			onClick={advance}
			// Solid black, not a tint: everything in this bar has to be opaque or the
			// winning scene's glow shines through the control that is supposed to be
			// in front of it. Hovering lifts the border rather than the ground.
			className="group/next flex w-full cursor-pointer flex-col items-center justify-center gap-[clamp(6px,0.9vh,11px)] rounded-xs border border-white/15 bg-background px-[clamp(8px,0.9vw,16px)] py-[clamp(10px,1.3vh,18px)] transition-colors duration-200 hover:border-white/45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/50 focus-visible:ring-offset-2 focus-visible:ring-offset-black"
			style={{ animation: "arena-rise 420ms cubic-bezier(0.25,0.8,0.3,1) 260ms both" }}
		>
			<span className="font-sans text-[clamp(9px,0.72vw,12px)] font-medium tracking-[0.2em] whitespace-nowrap text-foreground/55 transition-colors duration-200 group-hover/next:text-foreground">
				NEXT
			</span>
			{/* Drains left to right in real time. `scaleX` on a full-width fill rather
			    than an animated width: it is one composited transform per frame
			    instead of a layout pass, on a page still rendering two scenes. */}
			<span className="block h-px w-full overflow-hidden bg-white/15">
				<span
					className="block h-full w-full origin-left bg-foreground/70"
					style={{ transform: `scaleX(${t})` }}
				/>
			</span>
		</button>
	);
}
