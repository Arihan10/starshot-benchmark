"use client";

import { createContext, useContext, type RefObject } from "react";

/**
 * HOW FAR DOWN THE HERO THE READER HAS SCROLLED: 0 at the top, 1 once the next
 * section has arrived.
 *
 * A REF, NOT STATE, and that is the whole point of this module. Scroll fires at
 * the display's refresh rate; putting it in `useState` would re-render the page —
 * and everything under it — sixty times a second to deliver a number that only the
 * render loop reads. A ref lets the producer write and the consumer read without
 * React ever hearing about it, which is the standard shape for handing a scroll
 * position to a canvas.
 *
 * Its own module so the scroller and the scene can both reach it without either
 * importing the other.
 */
export const HeroProgress = createContext<RefObject<number> | null>(null);

export function useHeroProgress(): RefObject<number> {
	const ref = useContext(HeroProgress);
	if (!ref) {
		throw new Error("useHeroProgress must be used inside <SnapScroller>");
	}
	return ref;
}

/**
 * WHICH OF THE TWO SECTIONS THE READER IS ON, and how to get to the other.
 *
 * A SECOND CONTEXT, deliberately, rather than another field on the one above. That
 * one is a ref precisely so the scene can be driven sixty times a second without
 * React hearing about it; this one is STATE, because a control has to re-render to
 * change what it says. Put together, every tick of the scroll would re-render the
 * podium to tell a button what to call itself.
 */
export type BoardNav = {
	/** True once the standings have arrived — past the midpoint of the gesture. */
	onBoard: boolean;
	/**
	 * WHERE THE PAGE HAS COME TO REST, which is not the same question.
	 *
	 * `onBoard` flips halfway through, because a control has to commit to saying one
	 * thing or the other. This says whether the page is actually SETTLED at one end
	 * or still in transit — which is what the masthead needs, since it has to be
	 * gone for the whole of the journey and only return once the reader has arrived.
	 */
	phase: "hero" | "between" | "board";
	/** Take the reader down to the standings. */
	toBoard: () => void;
};

export const Board = createContext<BoardNav | null>(null);

export function useBoard(): BoardNav {
	const nav = useContext(Board);
	if (!nav) {
		throw new Error("useBoard must be used inside <SnapScroller>");
	}
	return nav;
}
