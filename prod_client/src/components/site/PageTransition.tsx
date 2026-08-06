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

export const HOLD = 240;

const Leaving = createContext<string | null>(null);

export function useLeaving(): boolean {
	return useContext(Leaving) !== null;
}

export function useDestination(): string | null {
	return useContext(Leaving);
}

export default function PageTransition({ children }: { children: ReactNode }) {
	const router = useRouter();
	const here = usePathname();
	const [trip, setTrip] = useState<{ from: string; to: string } | null>(null);
	const going = trip && trip.from === here ? trip.to : null;

	useEffect(() => {
		const onClick = (event: MouseEvent) => {
			const href = heldNavigation(event, here);
			if (!href) return;

			if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

			event.preventDefault();
			event.stopPropagation();
			setTrip({ from: here, to: href });
			window.setTimeout(() => router.push(href), HOLD);
		};
		const warmed = new Set<string>();
		const onOver = (event: Event) => {
			const link = (event.target as HTMLElement | null)?.closest?.("a");
			const href = link?.getAttribute("href");
			if (!href?.startsWith("/") || href === here) return;
			if (warmed.has(href)) return;
			warmed.add(href);
			router.prefetch(href);
		};

		document.addEventListener("click", onClick, true);
		document.addEventListener("pointerover", onOver, true);
		return () => {
			document.removeEventListener("click", onClick, true);
			document.removeEventListener("pointerover", onOver, true);
		};
	}, [router, here]);

	return <Leaving.Provider value={going}>{children}</Leaving.Provider>;
}
