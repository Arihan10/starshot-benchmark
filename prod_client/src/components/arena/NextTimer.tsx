"use client";

import { useCallback, useEffect, useRef } from "react";
import Button from "@/components/ui/Button";
import { linear, useProgress } from "./useProgress";

// IT STARTS WITH THE REVEAL. The 2.7 s lead existed because the ratings counted up
// for the first two seconds and the crowd's percentage landed at 2.6 — a bar
// draining through all of that read as a deadline on reading the result. Neither is
// true now (see RevealCard), so there is nothing left to wait for, and a countdown
// that sits still before it starts is just a longer countdown.
//
// Shorter as well: with the reveal down to a name and a number, 7.9 s of total wait
// was most of a round spent on a result that takes a moment to read.
const LEAD_MS = 0;
const RUN_MS = 3000;

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
		<Button
			onClick={advance}
			// Solid black, not a tint: everything in this bar has to be opaque or the
			// winning scene's glow shines through the control that is supposed to be
			// in front of it. Hovering lifts the border rather than the ground.
			className="group/next flex w-full flex-col items-center justify-center gap-xs"
			style={{ animation: "content-swap 400ms ease both" }}
		>
			<span className="font-label text-2xs whitespace-nowrap">
				NEXT
			</span>
			{/* Drains left to right in real time. `scaleX` on a full-width fill rather
			    than an animated width: it is one composited transform per frame
			    instead of a layout pass, on a page still rendering two scenes. */}
			<span className="block h-px w-full overflow-hidden bg-mark-16">
				<span
					className="block h-full w-full origin-left bg-mark-64"
					style={{ transform: `scaleX(${t})` }}
				/>
			</span>
		</Button>
	);
}
