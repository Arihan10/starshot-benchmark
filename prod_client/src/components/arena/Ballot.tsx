"use client";

import type { LocalCell } from "@/lib/localScenes";
import type { Vote } from "./vote";

function Chevron({ back, className }: { back?: boolean; className: string }) {
	return (
		<svg
			aria-hidden
			className={className}
			width="11"
			height="11"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2.6"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<path d={back ? "m15 5-7 7 7 7" : "m9 5 7 7-7 7"} />
		</svg>
	);
}

function Reveal({
	slot,
	cell,
	won,
	shown,
}: {
	slot: string;
	cell: LocalCell;
	won: boolean;
	shown: boolean;
}) {
	return (
		<span className="arena-reveal" data-won={won} aria-hidden={!shown}>
			<span className="arena-reveal__who">
				<span className="arena-reveal__slot">{slot}</span>
				<span className="arena-reveal__model">{cell.model}</span>
			</span>
			<span className="arena-reveal__elo">{cell.elo}</span>
		</span>
	);
}

export default function Ballot({
	cells,
	vote,
	onVote,
	onNext,
}: {
	cells: readonly [LocalCell, LocalCell];
	vote: Vote | null;
	onVote: (choice: Vote) => void;
	onNext: () => void;
}) {
	const voted = vote !== null;

	return (
		<div className="arena-ballot arena-chrome" data-voted={voted}>
			<div className="arena-ballot__row">
				<span className="arena-glow arena-glow--key">
					<button
						type="button"
						className="arena-vote"
						disabled={voted}
						onClick={() => onVote("a")}
					>
						<Chevron back className="arena-vote__arrow arena-vote__arrow--back" />
						<span className="arena-vote__label">A wins</span>
						<Reveal slot="A" cell={cells[0]} won={vote === "a"} shown={voted} />
					</button>
				</span>

				<button
					type="button"
					className="arena-skip"
					onClick={() => (voted ? onNext() : onVote("skip"))}
				>
					<span className="arena-skip__label">SKIP</span>
					<span className="arena-next" aria-hidden={!voted}>
						<span className="arena-next__label">NEXT</span>
						<span aria-hidden className="arena-next__track">
							<span className="arena-next__fill" />
						</span>
					</span>
				</button>

				<span className="arena-glow arena-glow--key">
					<button
						type="button"
						className="arena-vote"
						disabled={voted}
						onClick={() => onVote("b")}
					>
						<span className="arena-vote__label">B wins</span>
						<Chevron className="arena-vote__arrow arena-vote__arrow--forward" />
						<Reveal slot="B" cell={cells[1]} won={vote === "b"} shown={voted} />
					</button>
				</span>
			</div>
		</div>
	);
}
