"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import type { LocalCell } from "@/lib/localScenes";
import NextTimer from "./NextTimer";
import RevealCard, { RISE_MS } from "./RevealCard";
import VoteButton from "./VoteButton";
import Button from "@/components/ui/Button";

const SIDE_VOTING = "calc(var(--spacing-xl) * 4.3)";
const SIDE_SKIP_HOVER = "calc(var(--spacing-xl) * 4.04375)";
const SIDE_REVEALED = "calc(var(--spacing-xl) * 7.4)";
const MIDDLE = "calc(var(--spacing-xl) * 2.05)";
const MIDDLE_SKIP_HOVER = "calc(var(--spacing-xl) * 2.5625)";
const EXPAND_MS = 880;
const EXPAND = `${EXPAND_MS}ms cubic-bezier(0.62,0.02,0.24,1)`;

export const REVEAL_SETTLE_MS = EXPAND_MS + RISE_MS;

const SIDE_MAX = "44vw";

function Side({
	expanded,
	width,
	align,
	children,
	hoverWidth,
}: {
	expanded: boolean;
	width: number | null;
	align: "left" | "right";
	children: ReactNode;
	hoverWidth?: string;
}) {
	return (
		<div
			className={`flex overflow-hidden ${
				align === "right" ? "justify-end" : ""
			}`}
			style={{
				width: expanded
					? width
						? `max(${SIDE_VOTING}, ${width}px)`
						: SIDE_REVEALED
					: hoverWidth ?? SIDE_VOTING,
				maxWidth: SIDE_MAX,
				transition: expanded ? `width ${EXPAND}` : "width var(--duration-settle) ease-out",
			}}
		>
			{children}
		</div>
	);
}

export default function VoteBar({
	cells,
	vote,
	onVote,
	onNext,
	paused = false,
}: {
	cells: readonly LocalCell[];
	vote: "a" | "b" | "skip" | null;
	onVote: (choice: "a" | "b" | "skip") => void;
	onNext: () => void;
	paused?: boolean;
}) {
	const [left, right] = cells;
	const voted = vote !== null;
	const [skipHovered, setSkipHovered] = useState(false);

	const cardA = useRef<HTMLDivElement>(null);
	const cardB = useRef<HTMLDivElement>(null);
	const [revealWidth, setRevealWidth] = useState<number | null>(null);

	const [wasVoted, setWasVoted] = useState(voted);
	if (voted !== wasVoted) {
		setWasVoted(voted);
		if (!voted) setRevealWidth(null);
	}

	useEffect(() => {
		if (!voted) return;
		const cards = [cardA.current, cardB.current].filter(Boolean) as HTMLElement[];
		if (cards.length === 0) return;

		const measure = () => {
			const w = Math.ceil(Math.max(...cards.map((c) => c.offsetWidth)));
			if (w > 0) setRevealWidth((cur) => (cur === null || w > cur ? w : cur));
		};
		const ro = new ResizeObserver(measure);
		for (const c of cards) ro.observe(c);
		return () => ro.disconnect();
	}, [voted]);

	return (
		<div
			className={`flex items-stretch ${
				voted ? "" : "[&>*+*]:-ml-[13px]"
			}`}
		>
			<Side
				expanded={voted}
				width={revealWidth}
				align="left"
				hoverWidth={skipHovered ? SIDE_SKIP_HOVER : undefined}
			>
				{voted ? (
					<RevealCard
							ref={cardA}
							model={left.model}
							elo={left.elo}
							won={vote === "a"}
							align="left"
						/>
				) : (
					<VoteButton label="A wins" side="a" onVote={() => onVote("a")} />
				)}
			</Side>

			<div
				data-vote-middle
				className="flex justify-center"
				style={{
					width: skipHovered && !voted ? MIDDLE_SKIP_HOVER : MIDDLE,
					transition: "width var(--duration-settle) ease-out",
				}}
			>
				{voted ? (
					<NextTimer onNext={onNext} paused={paused} />
				) : (
					<div className="relative z-10 flex w-full shrink-0">
						<Button
							shape="middle"
							onClick={() => {
								setSkipHovered(false);
								onVote("skip");
							}}
							onMouseEnter={() => setSkipHovered(true)}
							onMouseLeave={() => setSkipHovered(false)}
							onFocus={() => setSkipHovered(true)}
							onBlur={() => setSkipHovered(false)}
							className="w-full px-xs"
							style={
								{
									// Same weight as the vertical join between the two scenes —
									// a 1px hairline here read as a different rule from the
									// divider it sits under.
									["--btn-edge" as string]: "var(--seam-width, 3px)",
								}
							}
						>
							Skip
						</Button>
					</div>
				)}
			</div>

			<Side
				expanded={voted}
				width={revealWidth}
				align="right"
				hoverWidth={skipHovered ? SIDE_SKIP_HOVER : undefined}
			>
				{voted ? (
					<RevealCard
							ref={cardB}
							model={right.model}
							elo={right.elo}
							won={vote === "b"}
							align="right"
						/>
				) : (
					<VoteButton label="B wins" side="b" onVote={() => onVote("b")} />
				)}
			</Side>
		</div>
	);
}
