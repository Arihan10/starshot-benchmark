"use client";

import {
	type ReactNode,
	useCallback,
	useEffect,
	useMemo,
	useRef,
	useState,
} from "react";
import { Board, HeroProgress } from "./heroProgress";

/**
 * The leaderboard's scroll container: two full-height sections that snap.
 *
 * WHY NOT THE DOCUMENT. The page needs a scroller it can measure and snap without
 * touching how every other route scrolls, so the container is a `h-dvh` box with
 * `overflow-y-auto` and the document itself never moves. That also makes the
 * progress maths trivial: one section is exactly one `clientHeight`, so
 * `scrollTop / clientHeight` IS the hero's progress with no layout queries.
 *
 * NATIVE CSS SNAP, NOT A SCROLL LIBRARY. `scroll-snap-type: y mandatory` is the
 * whole of "you cannot be left in the middle of the animation": the browser refuses
 * to rest between sections, and it does it with the platform's own physics on
 * every input — wheel, trackpad, touch, keyboard, spacebar, find-in-page — none of
 * which a hijacked scroller gets right for free. GSAP's ScrollTrigger and Lenis
 * both do this well and both take over scrolling to do it; for two sections that
 * is a large amount of machinery, and a second animation authority arguing with
 * the render loop, to replace one CSS declaration.
 *
 * The SMOOTHNESS the brief asks for is not the scroll's job — see Podium, where
 * the scene damps its way toward whatever this reports. Snapping decides WHERE the
 * page lands; the damping decides how the podium gets there, and separating the
 * two is why a fast flick still disassembles gracefully instead of cutting.
 *
 * `overscroll-contain` so a flick at either end does not chain out to the browser's
 * own bounce, which on iOS would let a reader hang the page mid-section.
 */
export default function SnapScroller({
	children,
	footer,
}: {
	children: ReactNode;
	/** Pinned over the scroller rather than inside it — see below. */
	footer?: ReactNode;
}) {
	const scroller = useRef<HTMLDivElement>(null);
	const progress = useRef(0);

	// HALFWAY IS THE SWITCH. The bar at the foot of the page says something
	// different depending on which section the reader is on, and past the midpoint
	// of the gesture the snap has effectively committed to the other one.
	const [onBoard, setOnBoard] = useState(false);
	const wasOnBoard = useRef(false);

	// AT REST, OR IN TRANSIT. A hair of tolerance at either end rather than an exact
	// comparison: a snapped scroll position lands on a fractional pixel often enough
	// that `=== 0` would leave the page permanently "between", and the masthead would
	// never come back.
	const [phase, setPhase] = useState<"hero" | "between" | "board">("hero");
	const wasPhase = useRef<"hero" | "between" | "board">("hero");

	const toBoard = useCallback(() => {
		const el = scroller.current;
		// `scrollTo` rather than `scrollIntoView`: the snap will land it exactly, and
		// this way the smooth scroll belongs to the container the reader is in.
		el?.scrollTo({ top: el.clientHeight, behavior: "smooth" });
	}, []);

	const nav = useMemo(
		() => ({ onBoard, phase, toBoard }),
		[onBoard, phase, toBoard],
	);

	useEffect(() => {
		const el = scroller.current;
		if (!el) return;
		const read = () => {
			const page = el.clientHeight || 1;
			progress.current = Math.min(1, Math.max(0, el.scrollTop / page));
			// GUARDED, so a scroll event that changes nothing schedules nothing. This
			// fires at the display's refresh rate; handing React the same boolean
			// sixty times a second is sixty chances to do work for no reason.
			const next = progress.current > 0.5;
			if (next !== wasOnBoard.current) {
				wasOnBoard.current = next;
				setOnBoard(next);
			}

			const where =
				progress.current < 0.02
					? "hero"
					: progress.current > 0.98
						? "board"
						: "between";
			if (where !== wasPhase.current) {
				wasPhase.current = where;
				setPhase(where);
			}
		};
		// Read once on mount as well as on scroll: a reload part-way down the page
		// restores the scroll position without firing an event, and the podium would
		// otherwise assemble itself over a section nobody is looking at.
		read();
		el.addEventListener("scroll", read, { passive: true });
		return () => el.removeEventListener("scroll", read);
	}, []);

	return (
		<HeroProgress.Provider value={progress}>
			<Board.Provider value={nav}>
				<div className="relative h-dvh overflow-hidden bg-ground">
					<div
						ref={scroller}
						className="h-full snap-y snap-mandatory overflow-x-hidden overflow-y-auto overscroll-contain"
					>
						{children}
					</div>

					{/* OVER THE SCROLLER, NOT IN IT. Sticky inside the container would put
					    the bar's own height at the end of the scrollable content — a third
					    resting place 60-odd pixels past the last snap point, which is
					    precisely the "caught between things" state the snapping exists to
					    prevent. Outside it, the bar covers the same strip of the window and
					    contributes no scroll length at all. */}
					{footer && <div className="absolute inset-x-0 bottom-0 z-40">{footer}</div>}
				</div>
			</Board.Provider>
		</HeroProgress.Provider>
	);
}
