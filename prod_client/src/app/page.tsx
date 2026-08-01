"use client";

import OrbitViewer from "@/components/OrbitViewer";
import { LOCAL_CELLS } from "@/lib/localScenes";

// SceneBench's comparison canvas: the same prompt-shaped task built by two
// different LLMs, side by side, each one orbitable on its own.
//
// The two cells are fixed for now (lib/localScenes) — this is the experiment we
// are looking at, not a picker. Scene selection, the A/B voting UI and the rest
// of the real site get built onto this.
//
// The builds carry NO attribution on screen. Which model made which is the
// question being asked, so naming them next to the render answers it for the
// viewer and poisons the comparison; the cells still know (LocalCell.model), and
// the reveal belongs after a vote rather than before one.
export default function Page() {
	return (
		<main className="flex h-dvh flex-col overflow-hidden bg-black">
			<div className="flex min-h-0 flex-1 flex-col md:flex-row">
				{LOCAL_CELLS.map((cell, i) => (
					<section
						key={cell.id}
						className={`relative min-h-0 min-w-0 flex-1 ${
							i === 0
								? "border-b border-white/10 md:border-b-0 md:border-r"
								: ""
						}`}
					>
						<OrbitViewer source={cell.source} />
					</section>
				))}
			</div>
		</main>
	);
}
