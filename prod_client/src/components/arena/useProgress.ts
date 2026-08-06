"use client";

import { useEffect, useState } from "react";

export const easeExpo = (p: number) => (1 - Math.exp(-4.6 * p)) / (1 - Math.exp(-4.6));

export const easeOutCubic = (p: number) => 1 - (1 - p) ** 3;

export const linear = (p: number) => p;

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
