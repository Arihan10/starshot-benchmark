"use client";

import { useEffect, useRef } from "react";

const FIRST_MS = 1200;
const GAP_MS = 2600;
const GAP_SPREAD_MS = 5200;

/**
 * A thin streak crosses a corner of the page every few seconds. Drawn with the
 * Web Animations API, so the sky costs nothing between shots.
 */
export default function StarField() {
	const hostRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		const host = hostRef.current;
		if (!host) return;
		if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

		let timer = 0;
		const shoot = () => {
			const width = host.clientWidth;
			const height = host.clientHeight;
			if (width && height) {
				const fromRight = Math.random() < 0.5;
				const streak = document.createElement("div");
				Object.assign(streak.style, {
					position: "absolute",
					top: `${(Math.random() * 0.55 * height).toFixed(0)}px`,
					left: `${(fromRight ? 0.55 + Math.random() * 0.4 : Math.random() * 0.4) * width}px`,
					width: `${(120 + Math.random() * 220).toFixed(0)}px`,
					height: "1px",
					transformOrigin: "0 50%",
					rotate: `${fromRight ? 155 + Math.random() * 12 : 25 - Math.random() * 12}deg`,
					background:
						"linear-gradient(90deg, rgb(255 255 255 / 0), rgb(255 255 255 / 0.85))",
					opacity: "0",
				});
				host.appendChild(streak);

				const travel = 320 + Math.random() * 340;
				const clear = () => streak.remove();
				streak
					.animate(
						[
							{ transform: `translateX(-${travel}px)`, opacity: 0 },
							{ opacity: 0.9, offset: 0.25 },
							{ opacity: 0.9, offset: 0.7 },
							{ transform: `translateX(${travel}px)`, opacity: 0 },
						],
						{
							duration: 900 + Math.random() * 700,
							easing: "cubic-bezier(0.3, 0, 0.7, 1)",
						},
					)
					.finished.then(clear, clear);
			}
			timer = window.setTimeout(shoot, GAP_MS + Math.random() * GAP_SPREAD_MS);
		};

		timer = window.setTimeout(shoot, FIRST_MS);
		return () => {
			window.clearTimeout(timer);
			host.replaceChildren();
		};
	}, []);

	return <div aria-hidden ref={hostRef} className="arena-sky" />;
}
