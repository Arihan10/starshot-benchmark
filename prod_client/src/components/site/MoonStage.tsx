"use client";

import { useEffect, useRef } from "react";
import Moon from "@/components/Moon";

const CHASE = 3.4;
const FADE = 7;

const SETTLED = 0.2;

export default function MoonStage() {
	const disc = useRef<HTMLDivElement>(null);

	useEffect(() => {
		let frame = 0;
		let last = 0;
		let pose: { x: number; y: number; d: number } | null = null;
		let lit = 0;

		const step = (now: number) => {
			frame = requestAnimationFrame(step);
			const el = disc.current;
			if (!el) return;

			const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
			last = now;

			const marks = document.querySelectorAll<HTMLElement>("[data-moon-anchor]");
			let anchor: HTMLElement | null = null;
			for (const mark of marks) {
				if (!mark.closest("[data-moon-idle]")) {
					anchor = mark;
					break;
				}
			}

			const want = anchor?.getBoundingClientRect();
			const there = want && want.width > 0;

			lit += ((there ? 1 : 0) - lit) * (1 - Math.exp(-FADE * dt));
			el.style.opacity = `${lit}`;
			if (!there || !want) return;


			const target = {
				x: want.left + want.width / 2,
				y: want.top + want.height / 2,
				d: want.width,
			};

			if (!pose) {
				pose = { x: target.x, y: -target.d, d: target.d };
			}

			const k = 1 - Math.exp(-CHASE * dt);
			pose.x += (target.x - pose.x) * k;
			pose.y += (target.y - pose.y) * k;
			pose.d += (target.d - pose.d) * k;

			if (
				Math.abs(target.x - pose.x) < SETTLED &&
				Math.abs(target.y - pose.y) < SETTLED &&
				Math.abs(target.d - pose.d) < SETTLED
			) {
				pose = { ...target };
			}

			el.style.width = `${pose.d}px`;
			el.style.height = `${pose.d}px`;
			el.style.transform = `translate(${pose.x - pose.d / 2}px, ${pose.y - pose.d / 2}px)`;

		};

		frame = requestAnimationFrame(step);
		return () => cancelAnimationFrame(frame);
	}, []);

	return (
		<>
			<div
				ref={disc}
				aria-hidden
				className="pointer-events-none fixed top-0 left-0 z-10 opacity-0 will-change-transform"
			>
				<Moon diameter="100%" />
			</div>
		</>
	);
}
