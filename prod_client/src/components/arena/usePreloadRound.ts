"use client";

import { useEffect } from "react";
import type { LocalRound } from "@/lib/localScenes";

/**
 * Pull the next matchup's heavy assets down while the reader is still looking at
 * this one's result.
 *
 * THE COUNTDOWN IS THE BUDGET. Between a vote landing and the next pair arriving
 * there are about eight seconds in which the page is showing two ratings and a
 * progress bar and asking nothing of the network — and the thing that happens at
 * the end of those eight seconds is two fresh scenes being asked for at once.
 * Spending the wait on the fetch is the difference between a swap and a stall.
 *
 * ONLY THE TWO FILES A SCENE CANNOT APPEAR WITHOUT: the dollhouse mesh and the
 * splat. The tour manifest is deliberately skipped even though it is small —
 * the engine fetches it `no-store` (a re-publish rewrites it in place), so a
 * warmed copy would be thrown away. Panos and the proxy are skipped because
 * nothing needs them until someone steps inside, and pulling them now would
 * compete with the two files that ARE about to be on screen.
 *
 * Fire and forget, and deliberately NOT aborted on cleanup: the point is to leave
 * the response in the HTTP cache, and a viewer who presses "next" early is exactly
 * the viewer whose download must not be cancelled a moment before it is wanted.
 */
export function usePreloadRound(round: LocalRound, active: boolean) {
	useEffect(() => {
		if (!active) return;
		const urls = round.cells
			.flatMap((cell) => [cell.source.dollhouseUrl, cell.source.splatUrl])
			.filter((url): url is string => !!url);

		for (const url of urls) {
			// The body is READ to completion. A response whose body is left dangling
			// is not reliably finished or stored — the fetch has to be drained for
			// the cache entry to exist, which is the entire purpose here. The buffer
			// is dropped immediately; what stays behind is the cache.
			void fetch(url, { priority: "low" } as RequestInit)
				.then((res) => res.arrayBuffer())
				.catch(() => {});
		}
	}, [round, active]);
}
