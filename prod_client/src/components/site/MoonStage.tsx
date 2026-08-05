"use client";

import { useEffect, useRef } from "react";
import Moon from "@/components/Moon";

/**
 * THE MOON. One of it, for the whole site, for the life of the tab.
 *
 * It is rendered by the root layout, which survives every navigation, so it is
 * never destroyed and never rebuilt. Pages do not draw a moon — they place a
 * `MoonAnchor` where one belongs, and this eases toward whichever anchor is on
 * screen. Changing page changes the target; the moon simply carries on.
 *
 * WHAT THIS REPLACED, because the difference is the point. The moon used to be
 * rendered per page, so a navigation destroyed one and made another. Faking
 * continuity across that took a held route change, a clone pinned to the viewport,
 * a baton of state smuggled through module scope, and the arriving page
 * reconstructing how far along the animation should be from a timestamp. Six
 * separate bugs came out of that seam — a double-play, a teleport, an invisible
 * frame, a stale hand-off — and every one of them was the same bug: two objects
 * pretending to be one. There is no seam here to get wrong.
 *
 * IT IS NEVER RE-RENDERED. The pose is written straight to the element from a frame
 * loop; React lays this out once and then has nothing further to do with it.
 */

/**
 * How hard the moon chases its anchor, and how hard it fades.
 *
 * Frame-rate independent — the exponential means the same constant gives the same
 * curve at 60Hz and 120Hz. Loose enough that a change of page reads as the disc
 * TRAVELLING rather than cutting, tight enough that scrolling the About page does
 * not leave it trailing behind the copy it is meant to sit beside.
 */
const CHASE = 3.4;
const FADE = 7;

/** Any further from its target than this and the moon is not eased, it is placed —
 *  see the first frame, and any page that has no anchor at all. */
const SETTLED = 0.4;

export default function MoonStage() {
	const disc = useRef<HTMLDivElement>(null);

	useEffect(() => {
		let frame = 0;
		let last = 0;
		// Null until the first anchor is found, which is what tells the loop it has
		// nowhere to ease FROM yet.
		let pose: { x: number; y: number; d: number } | null = null;
		let lit = 0;

		const step = (now: number) => {
			frame = requestAnimationFrame(step);
			const el = disc.current;
			if (!el) return;

			const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
			last = now;

			// WHICHEVER ANCHOR IS LIVE. A page may hold more than one — the leaderboard
			// has a masthead per section — and the one that is not currently on show
			// marks itself idle rather than being removed, so the moon is never left
			// chasing a band that has scrolled away.
			const marks = document.querySelectorAll<HTMLElement>("[data-moon-anchor]");
			let anchor: HTMLElement | null = null;
			for (const mark of marks) {
				if (!mark.closest("[data-moon-idle]")) {
					anchor = mark;
					break;
				}
			}

			// NOTHING TO SIT ON: hold position and fade. That is exactly the
			// leaderboard mid-scroll, where one band has gone and the next has not
			// arrived — the moon dims in place rather than sliding a whole screen to
			// meet a target that is about to be replaced.
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
				// FIRST SIGHT OF AN ANCHOR: come down into it rather than blinking on.
				// The disc starts a full diameter above the window, so on a page whose
				// moon already hangs off the top this is a short descent, and on About —
				// where it belongs beside the copy — it is the arrival the page was
				// designed around. Costs nothing: it is the same easing as everything
				// else, given a different starting point.
				pose = { x: target.x, y: -target.d, d: target.d };
			}

			const k = 1 - Math.exp(-CHASE * dt);
			pose.x += (target.x - pose.x) * k;
			pose.y += (target.y - pose.y) * k;
			pose.d += (target.d - pose.d) * k;

			// Snap the last fraction of a pixel, so a moon that has arrived stops
			// writing new values and the compositor has nothing to do.
			if (
				Math.abs(target.x - pose.x) < SETTLED &&
				Math.abs(target.y - pose.y) < SETTLED &&
				Math.abs(target.d - pose.d) < SETTLED
			) {
				pose = { ...target };
			}

			// SIZED, NOT SCALED. The moon is built from radial gradients; a transform
			// scale rasterises them once and stretches the bitmap, and About grows it
			// nearly fivefold on its last screen.
			el.style.width = `${pose.d}px`;
			el.style.height = `${pose.d}px`;
			el.style.transform = `translate(${pose.x - pose.d / 2}px, ${pose.y - pose.d / 2}px)`;
		};

		frame = requestAnimationFrame(step);
		return () => cancelAnimationFrame(frame);
	}, []);

	return (
		// FIXED, so it is in viewport coordinates on every page and no page's scroll
		// or overflow can clip it.
		//
		// `z-10` places it above the things it should sit over — the arena's canvases,
		// the leaderboard's sky — and below everything that must be read across it:
		// the navbar and the caption struck on its own face at z-30, the page copy and
		// the standings at z-30, the exit bar at z-40.
		<div
			ref={disc}
			aria-hidden
			className="pointer-events-none fixed top-0 left-0 z-10 opacity-0 will-change-transform"
		>
			<Moon diameter="100%" />
		</div>
	);
}
