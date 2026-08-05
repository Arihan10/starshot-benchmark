"use client";

import { usePathname, useRouter } from "next/navigation";
import {
	createContext,
	type ReactNode,
	useContext,
	useEffect,
	useState,
} from "react";
import { heldNavigation } from "./navIntercept";

/**
 * ONE PLACE THAT KNOWS THE PAGE IS LEAVING, and everything that wants to see itself
 * out subscribes to it.
 *
 * WHY IT HAD TO BE CENTRAL. Each thing that animated on the way out used to hold
 * the navigation itself — the moon on About, the type on the masthead — and they
 * raced. Two mastheads on the leaderboard meant two listeners on the same click:
 * the first called `preventDefault` and pushed, the second found the event already
 * handled and did nothing, so whichever caption happened to be VISIBLE was the one
 * that might not fade. Worse, each hold was its own duration, so the page left
 * whenever the fastest one finished.
 *
 * Now the click is caught once, everything is told at the same instant, and the
 * route changes when the LONGEST of them is done. Adding another thing that exits
 * costs a `useLeaving()` and no coordination at all.
 */

/**
 * How long the page waits before it goes.
 *
 * THE SLOWEST EXIT SETS IT, and that is now only the type coming off the moon —
 * a fifth of a second. It used to be more than twice this, because the moon had to
 * finish climbing back to the masthead before the route could change, or it would
 * be destroyed mid-flight. The moon no longer travels between pages; it IS between
 * pages, so nothing has to wait for it and the hold is down to what a fade needs.
 */
export const HOLD = 240;

/** Null when nothing is leaving; otherwise the route being gone to — which the
 *  moon needs, because it only travels for one destination. */
const Leaving = createContext<string | null>(null);

/** True from the moment a held navigation is taken until the route changes. */
export function useLeaving(): boolean {
	return useContext(Leaving) !== null;
}

/** Where the page is going, for the few things that behave differently per
 *  destination. Null unless a navigation is under way. */
export function useDestination(): string | null {
	return useContext(Leaving);
}

export default function PageTransition({ children }: { children: ReactNode }) {
	const router = useRouter();
	const here = usePathname();
	// WHICH PAGE IS ON ITS WAY OUT, rather than a bare "leaving" flag — and the
	// difference is what makes the reset free. The flag has to clear once the route
	// changes, or a page would arrive already rendering itself as departing; storing
	// the path being LEFT means that happens by arithmetic, the moment `here` stops
	// matching it. As a boolean it needed an effect to put it back, and an effect
	// that sets state from its own body is a cascading render.
	const [trip, setTrip] = useState<{ from: string; to: string } | null>(null);
	// Cleared by arithmetic the moment the route changes, rather than by an effect
	// that would set state from its own body.
	const going = trip && trip.from === here ? trip.to : null;

	useEffect(() => {
		const onClick = (event: MouseEvent) => {
			const href = heldNavigation(event, here);
			if (!href) return;

			// A reader who has asked for less movement gets the page, not a send-off.
			if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

			event.preventDefault();
			event.stopPropagation();
			setTrip({ from: here, to: href });
			window.setTimeout(() => router.push(href), HOLD);
		};
		/**
		 * FETCHED ON APPROACH, so the click has somewhere to go.
		 *
		 * The hold in front of a navigation is spent animating, and that animation is
		 * only smooth if the next page is ready when it ends. Asked for at the click,
		 * the route is still arriving while the moon is mid-flight — and everything
		 * downstream then has to cope with a page that mounts late.
		 *
		 * A pointer crossing a link is the earliest honest signal that it might be
		 * followed, and it buys a few hundred milliseconds. Deduplicated, because
		 * a pointer crosses a link many times on the way to pressing it.
		 */
		const warmed = new Set<string>();
		const onOver = (event: Event) => {
			const link = (event.target as HTMLElement | null)?.closest?.("a");
			const href = link?.getAttribute("href");
			if (!href?.startsWith("/") || href === here) return;
			if (warmed.has(href)) return;
			warmed.add(href);
			router.prefetch(href);
		};

		// IN THE CAPTURE PHASE. The anchor's own handler is what starts the
		// navigation, so a listener waiting for the bubble would be told after Next
		// had already been asked for the next page.
		document.addEventListener("click", onClick, true);
		document.addEventListener("pointerover", onOver, true);
		return () => {
			document.removeEventListener("click", onClick, true);
			document.removeEventListener("pointerover", onOver, true);
		};
	}, [router, here]);

	return <Leaving.Provider value={going}>{children}</Leaving.Provider>;
}
