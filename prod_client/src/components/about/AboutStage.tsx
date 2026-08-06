"use client";

import { type ReactNode, useEffect, useLayoutEffect, useRef } from "react";
import Fade from "@/components/site/Fade";
import MoonAnchor from "@/components/site/MoonAnchor";
import { VoxelDrift } from "@/components/site/VoxelSky";
import Navbar from "@/components/site/Navbar";

const STARS = [
	{ top: "-6%", left: "-14%", width: 210, alpha: 0.85, seconds: 17, delay: 0 },
	{ top: "12%", left: "-22%", width: 150, alpha: 0.55, seconds: 23, delay: 6.5 },
	{ top: "-14%", left: "18%", width: 260, alpha: 0.7, seconds: 29, delay: 13 },
	{ top: "30%", left: "-10%", width: 120, alpha: 0.45, seconds: 21, delay: 18.5 },
	{ top: "-2%", left: "46%", width: 180, alpha: 0.6, seconds: 34, delay: 25 },
];

const useBeforePaint = typeof window === "undefined" ? useEffect : useLayoutEffect;

const HOLD = 0.12;
const GONE = 0.34;

const BUSY_L = [0, 1, 0.32];
const BUSY_R = [1, 0, 0.32];

export default function AboutStage({ children }: { children: ReactNode }) {
	const stage = useRef<HTMLDivElement>(null);
	const scroller = useRef<HTMLDivElement>(null);
	const anchor = useRef<HTMLDivElement>(null);
	const bar = useRef<HTMLDivElement>(null);

	useBeforePaint(() => {
		const box = scroller.current;
		const mark = anchor.current;
		const root = stage.current;
		if (!box || !mark || !root) return;

		let queued = 0;

		const place = () => {
			const W = box.clientWidth;
			const H = box.clientHeight;

			const size = Math.max(240, Math.min(0.46 * H, 0.44 * W, 560));
			const swell = Math.max(1.35 * W, 1.1 * H, 980);

			const stops = [
				{ x: 0.26 * W, y: 0.52 * H, d: size },
				{ x: 0.74 * W, y: 0.5 * H, d: size },
				{ x: 0.5 * W, y: H + swell / 2 - 0.3 * H, d: swell },
			];

			const screens = box.querySelectorAll<HTMLElement>("section");
			const travelled = Math.min(2, Math.max(0, box.scrollTop / Math.max(1, H)));
			const leg = travelled < 1 ? 0 : 1;
			const f = travelled - leg;
			const u = f * f * (3 - 2 * f);
			const a = stops[leg];
			const b = stops[leg + 1];
			const x = a.x + (b.x - a.x) * u;
			const y = a.y + (b.y - a.y) * u;
			const d = a.d + (b.d - a.d) * u;

			mark.style.width = `${d}px`;
			mark.style.height = `${d}px`;
			mark.style.transform = `translate(${x - d / 2}px, ${y - d / 2}px)`;

			for (let i = 0; i < screens.length; i++) {
				const away = Math.abs(travelled - i);
				const t = (away - HOLD) / (GONE - HOLD);
				const o = t <= 0 ? 1 : t >= 1 ? 0 : 1 - t * t * (3 - 2 * t);
				screens[i].style.opacity = o.toFixed(3);
				screens[i].style.pointerEvents = o < 0.05 ? "none" : "";
				// Set once and never cleared: a screen's contents rise on the way in
				// the first time it is reached, and scrolling back past it must not
				// re-arm a cascade the reader has already watched.
				if (o > 0.86) screens[i].dataset.arrived = "";
			}

			const busyL = BUSY_L[leg] + (BUSY_L[leg + 1] - BUSY_L[leg]) * u;
			const busyR = BUSY_R[leg] + (BUSY_R[leg + 1] - BUSY_R[leg]) * u;
			root.style.setProperty("--voxel-left-o", (1 - 0.86 * busyL).toFixed(3));
			root.style.setProperty("--voxel-right-o", (1 - 0.86 * busyR).toFixed(3));
			root.style.setProperty("--voxel-left-x", `${(-104 * busyL).toFixed(1)}px`);
			root.style.setProperty("--voxel-right-x", `${(104 * busyR).toFixed(1)}px`);
		};

		const tick = () => {
			if (queued) return;
			queued = requestAnimationFrame(() => {
				queued = 0;
				place();
			});
		};

		place();
		box.addEventListener("scroll", tick, { passive: true });
		const observer = new ResizeObserver(place);
		observer.observe(box);
		return () => {
			box.removeEventListener("scroll", tick);
			observer.disconnect();
			cancelAnimationFrame(queued);
		};
	}, []);

	return (
		<div
			ref={stage}
			className="relative h-dvh overflow-hidden bg-ground"
		>
			<div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
				{STARS.map((star) => (
					<span
						key={`${star.top}-${star.left}`}
						className="absolute h-px origin-left"
						style={{
							top: star.top,
							left: star.left,
							width: star.width,
							background: `linear-gradient(90deg, rgb(var(--mark-rgb) / 0) 0%, rgb(var(--mark-rgb) / ${star.alpha}) 62%, rgb(var(--mark-rgb) / 0) 100%)`,
							animation: `about-shoot ${star.seconds}s linear ${star.delay}s infinite backwards`,
						}}
					/>
				))}
			</div>

			<VoxelDrift />

			<MoonAnchor
				ref={anchor}
				className="absolute top-0 left-0"
			/>

			<div
				ref={scroller}
				className="relative z-30 h-full snap-y snap-mandatory overflow-y-auto overscroll-contain"
			>
				<Fade enter={null}>{children}</Fade>
			</div>

			{/* The paper the bar runs on. Masthead supplies its own; this page mounts
			    the bar directly, and the inverted palette is unreadable without it. */}
			<div
				ref={bar}
				className="pointer-events-none absolute inset-x-0 top-0 z-30 bg-paper [&_a]:pointer-events-auto [&_button]:pointer-events-auto"
			>
				<Navbar />
			</div>
		</div>
	);
}
