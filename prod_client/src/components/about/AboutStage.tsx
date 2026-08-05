"use client";

import { type ReactNode, useEffect, useLayoutEffect, useRef } from "react";
import Fade from "@/components/site/Fade";
import MoonAnchor from "@/components/site/MoonAnchor";
import { VoxelDrift } from "@/components/site/VoxelSky";
import Navbar from "@/components/site/Navbar";

/**
 * The About page's frame: three screens that snap, a sky behind them, and the moon
 * that travels between them.
 *
 * THE MOON IS THE PAGE'S ONE MOVING PART, and it is parked rather than scrolled. It
 * holds three positions — left of the first screen's copy, right of the second's,
 * and then rising into the bottom of the third at many times its own size, so the
 * page ends on a horizon. Between them it is interpolated, so scrolling drags it
 * from one to the next; at rest on any screen it is exactly where that screen wants
 * it.
 *
 * WRITTEN STRAIGHT TO THE DOM, never through state. This runs on every scroll
 * event, and putting a transform through React would re-render three screens of
 * copy to move one disc.
 */

/** The five streaks. Rakes are shared — only where and when differ. */
const STARS = [
	{ top: "-6%", left: "-14%", width: 210, alpha: 0.85, seconds: 17, delay: 0 },
	{ top: "12%", left: "-22%", width: 150, alpha: 0.55, seconds: 23, delay: 6.5 },
	{ top: "-14%", left: "18%", width: 260, alpha: 0.7, seconds: 29, delay: 13 },
	{ top: "30%", left: "-10%", width: 120, alpha: 0.45, seconds: 21, delay: 18.5 },
	{ top: "-2%", left: "46%", width: 180, alpha: 0.6, seconds: 34, delay: 25 },
];

/**
 * BEFORE THE PAINT, NOT AFTER IT.
 *
 * `useEffect` runs once the browser has already drawn the frame — so the moon's
 * first appearance on this page was a box with no size at all, and the disc
 * vanished for a frame at exactly the moment it was meant to be sailing across
 * unbroken. The very first placement has to happen while the frame is still being
 * assembled.
 *
 * Chosen once at module scope rather than per render, because which hook it is
 * cannot change between renders — and `useLayoutEffect` on the server is a warning
 * about a hook that has nothing to lay out.
 */
const useBeforePaint = typeof window === "undefined" ? useEffect : useLayoutEffect;


// HOW LONG THE WORDS STAY, in screens travelled away from their own.
//
// `HOLD` is the dead zone either side of a stop where the copy is simply up —
// without it the text would start dimming on the first pixel of a scroll, which
// reads as a page that cannot decide. `GONE` is where it has finished leaving, and
// it is deliberately well short of the half-way point the moon crosses at.
const HOLD = 0.12;
const GONE = 0.34;

// Which half of each screen the copy is in. See the fade below.
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

			// HOW BIG THE DISC IS, and it is bounded on three sides: never smaller than
			// 240 or it stops reading as a body, never more than 560 or it crowds the
			// copy, and never more than a fraction of either window dimension so it
			// cannot fill a short window or a narrow one.
			const size = Math.max(240, Math.min(0.46 * H, 0.44 * W, 560));
			// What it grows to on the last screen: bigger than the window on both axes,
			// so what shows is a curve rather than a circle.
			const swell = Math.max(1.35 * W, 1.1 * H, 980);

			const stops = [
				{ x: 0.26 * W, y: 0.52 * H, d: size },
				{ x: 0.74 * W, y: 0.5 * H, d: size },
				{ x: 0.5 * W, y: H + swell / 2 - 0.3 * H, d: swell },
			];

			// WHICH PAIR OF STOPS WE ARE BETWEEN, in screens travelled. Clamped at both
			// ends so overscroll — a rubber-band on a trackpad — cannot throw the disc
			// past its last position.
			const screens = box.querySelectorAll<HTMLElement>("section");
			const travelled = Math.min(2, Math.max(0, box.scrollTop / Math.max(1, H)));
			const leg = travelled < 1 ? 0 : 1;
			const f = travelled - leg;
			// Smoothstep, so the moon is at rest at each stop rather than changing
			// direction at speed the instant a screen is reached.
			const u = f * f * (3 - 2 * f);
			const a = stops[leg];
			const b = stops[leg + 1];
			const x = a.x + (b.x - a.x) * u;
			const y = a.y + (b.y - a.y) * u;
			const d = a.d + (b.d - a.d) * u;

			// THE ANCHOR MOVES, NOT A MOON. This page owns where the disc belongs on
			// each of its screens; the disc itself belongs to the layout and eases
			// toward whatever this box says. Which is why there is no arrival animation
			// here any more, and no departure either — coming from the masthead and
			// going back to it are both just the moon following a rectangle that
			// happens to have moved to another page.
			mark.style.width = `${d}px`;
			mark.style.height = `${d}px`;
			mark.style.transform = `translate(${x - d / 2}px, ${y - d / 2}px)`;

			// --- THE COPY GETS OUT OF THE MOON'S WAY -------------------------
			//
			// Held at full strength while the disc is settled on a screen, and gone
			// well before it arrives. The numbers are what make the sequence read
			// the way it is described: at `travelled` = x.5 the moon is halfway
			// across, and both screens' words left at x.34 — so it crosses an empty
			// page rather than sliding under a paragraph. The next screen's words
			// then come back only inside the last tenth, once the disc has stopped.
			//
			// The SECTIONS are faded rather than anything inside them, so this needs
			// no markup and cannot fall out of step with a column that gets renamed.
			// Opacity does not affect layout, so the snap points are untouched.
			for (let i = 0; i < screens.length; i++) {
				const away = Math.abs(travelled - i);
				const t = (away - HOLD) / (GONE - HOLD);
				const o = t <= 0 ? 1 : t >= 1 ? 0 : 1 - t * t * (3 - 2 * t);
				screens[i].style.opacity = o.toFixed(3);
				// Nothing under a fully faded screen should still be clickable.
				screens[i].style.pointerEvents = o < 0.05 ? "none" : "";
			}

			// --- AND THE VOXELS TAKE WHAT IS LEFT ----------------------------
			//
			// How much of each side the copy occupies, per screen: right on the
			// first, left on the second, and on the third it holds the middle — which
			// frees both margins. Interpolated on the SAME easing as the disc, so the
			// blocks clear the way on the moon's clock rather than a second one.
			const busyL = BUSY_L[leg] + (BUSY_L[leg + 1] - BUSY_L[leg]) * u;
			const busyR = BUSY_R[leg] + (BUSY_R[leg + 1] - BUSY_R[leg]) * u;
			// Dimmed AND pushed toward their own edge: fading alone leaves them
			// sitting behind the text, which is where they were least wanted.
			root.style.setProperty("--voxel-left-o", (1 - 0.86 * busyL).toFixed(3));
			root.style.setProperty("--voxel-right-o", (1 - 0.86 * busyR).toFixed(3));
			root.style.setProperty("--voxel-left-x", `${(-104 * busyL).toFixed(1)}px`);
			root.style.setProperty("--voxel-right-x", `${(104 * busyR).toFixed(1)}px`);
		};

		// COALESCED TO ONE PER FRAME. Scroll fires faster than the display refreshes,
		// and every extra call would be a layout read and a style write thrown away.
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
			{/* --- the sky ---------------------------------------------------------
			    Behind everything, and taking no pointer events. */}
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

			{/* --- the voxels ------------------------------------------------------
			    The site's own, shared with the leaderboard and the FAQ. Only the drift:
			    the streaks above are this page's, on their own placement and clock, and
			    two sets of them running at once reads as noise rather than as sky. */}
			<VoxelDrift />

			{/* --- where the moon belongs -----------------------------------------
			    Not a moon: a rectangle. The disc is the layout's and outlives this
			    page; all this does is say where it should be on each screen, and the
			    script above moves this box as the reader scrolls. */}
			<MoonAnchor
				ref={anchor}
				className="absolute top-0 left-0"
			/>

			{/* --- the screens -----------------------------------------------------
			    `z-30` so the copy sits OVER the moon, which is fixed at z-10 in the
			    root layout. Without a layer of its own this would fall behind: the moon
			    is positioned and these screens are not, and a positioned box paints
			    above static ones whatever the document order says. */}
			<div
				ref={scroller}
				className="relative z-30 h-full snap-y snap-mandatory overflow-y-auto overscroll-contain"
			>
				{/* THE COPY FADES, THE MOON DOES NOT. Everything that belongs to this
				    page in particular is inside the scroller; the disc and the sky are
				    siblings outside it, so wrapping here sees the three screens out
				    without touching the one object that is supposed to survive the
				    navigation.

				    No entrance: each screen already stages its own, a beat apart down
				    the column, and a second fade over the top would only flatten it. */}
				<Fade enter={null}>{children}</Fade>
			</div>

			{/* THE NAVBAR IS FIXED OVER ALL OF IT, as it is on the arena — this page
			    scrolls through three screens and the way out has to be on every one. */}
			<div
				ref={bar}
				className="pointer-events-none absolute inset-x-0 top-0 z-30 [&_a]:pointer-events-auto [&_button]:pointer-events-auto"
			>
				<Navbar />
			</div>
		</div>
	);
}
