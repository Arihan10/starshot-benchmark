"use client";

import Link from "next/link";
import { useBoard } from "./heroProgress";

/**
 * The bar across the foot of the page, and it is two controls in one place.
 *
 * ON THE PODIUM it offers the thing the reader came for and cannot yet see — the
 * full board, one section down. On the BOARD it offers the only thing left to do,
 * which is go and vote. One bar rather than two because the foot of the page has
 * room for exactly one, and whichever is the next step should be sitting in it.
 *
 * THE TWO STATES ARE DIFFERENT ELEMENTS ON PURPOSE. "Show the leaderboard" moves
 * the reader within this page and is a button; "enter the arena" goes somewhere
 * else and has to be an anchor, or it cannot be middle-clicked, copied or crawled.
 * Rendering one and faking the other would cost exactly that.
 *
 * The bar around them keeps its identity, so the colour is a transition rather
 * than a cut even though what is inside it is swapped.
 */
export default function ExitBar() {
	const { onBoard, toBoard } = useBoard();

	const type =
		"relative block w-full py-md text-center font-sans text-sm font-bold tracking-[0.07em] uppercase";

	return (
		<div
			className={`relative overflow-hidden transition-colors duration-settle ${
				onBoard
					? "bg-mark"
					: // BLACK ON BLACK NEEDS AN EDGE. Over the bottom of the city the bar
						// would otherwise have no boundary at all — the hairline is what
						// makes it read as a bar rather than as a gap in the scene.
						"border-t border-mark-16 bg-ground"
			}`}
		>
			{onBoard ? (
				<Link href="/" className={`group/exit ${type} text-ground`}>
					{/* THE MARK'S OWN SWEEP, the same layer the CTA uses — a fading copy
					    rather than a background swap, because `background-image` does not
					    interpolate and would snap in while everything around it eased. */}
					<span
						aria-hidden
						className="absolute inset-0 opacity-0 transition-opacity duration-[420ms] ease-out group-hover/exit:opacity-100"
						style={{ backgroundImage: "var(--accent-sweep)" }}
					/>
					<span className="relative">
						Enter the arena
						<span
							aria-hidden
							className="ml-sm inline-block transition-transform duration-quick group-hover/exit:translate-x-1"
						>
							→
						</span>
					</span>
				</Link>
			) : (
				<button
					type="button"
					onClick={toBoard}
					className={`group/more cursor-pointer text-ink transition-colors duration-quick hover:bg-surface ${type}`}
				>
					Show full leaderboard
					{/* Down, because that is where it goes — and it nudges that way on
					    hover, which is the same gesture the reader is about to make. */}
					<span
						aria-hidden
						className="ml-sm inline-block transition-transform duration-quick group-hover/more:translate-y-1"
					>
						↓
					</span>
				</button>
			)}
		</div>
	);
}
