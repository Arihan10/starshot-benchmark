"use client";

import { useEffect, useState } from "react";

/** Fast out of the gate, long settle — the number lands before it stops moving. */
export const easeExpo = (p: number) => (1 - Math.exp(-4.6 * p)) / (1 - Math.exp(-4.6));

/** Standard decelerate, for the percentage. */
export const easeOutCubic = (p: number) => 1 - (1 - p) ** 3;

/**
 * No easing at all — for a clock. A countdown that slowed as it ran would be
 * lying about how much time was left.
 *
 * Module scope, like the others, and that matters: `useProgress` keys its effect
 * on the ease it is handed, so an arrow function written at the call site is a new
 * ease every render and restarts the ramp each time.
 */
export const linear = (p: number) => p;

/**
 * A 0→1 ramp that starts when the component mounts.
 *
 * DELIBERATELY OWNED BY THE LEAF THAT DISPLAYS IT. Driving these from the page
 * would re-render both 3D panels sixty times a second for two seconds, to animate
 * a number neither of them shows — the engines survive it (they live in refs) but
 * every overlay in the viewer would be rebuilt for nothing. Kept down here, the
 * per-frame render is one span of text.
 */
export function useProgress(
	duration: number,
	delay = 0,
	ease: (p: number) => number = (p) => p,
): number {
	const [t, setT] = useState(0);

	useEffect(() => {
		let raf = 0;
		let started = 0;
		const step = (now: number) => {
			if (!started) started = now;
			const p = Math.min(1, (now - started) / duration);
			setT(ease(p));
			if (p < 1) raf = requestAnimationFrame(step);
		};
		const timer = window.setTimeout(() => {
			raf = requestAnimationFrame(step);
		}, delay);
		return () => {
			clearTimeout(timer);
			cancelAnimationFrame(raf);
		};
	}, [duration, delay, ease]);

	return t;
}
