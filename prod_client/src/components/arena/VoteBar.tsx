"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import type { LocalCell } from "@/lib/localScenes";
import NextTimer from "./NextTimer";
import RevealCard, { RISE_MS } from "./RevealCard";
import VoteButton from "./VoteButton";
import Button, { RAKE_PX } from "@/components/ui/Button";

// Pull each member over its neighbour by MORE than the rake. At exactly the rake
// the shared diagonals only kiss: two independently antialiased edges each land
// partway into the boundary pixel, so the white of a slab and the white of
// SKIP's rule sum to about three quarters of one — a grey notch at the very
// corner where they are supposed to meet. One extra pixel lets the whites
// overlap and the seam has nothing left to show.
//
// THE OVERLAP ONLY WORKS IF THE SLABS ARE ON TOP — see the z-order note below.
const SEAM_PULL = RAKE_PX + 1;

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
            // Only clip once the reveal cards are in — while voting, `overflow:
            // hidden` shears the clip-path AA off every outer vertex (the tips sit
            // on the box edge), so the parallelogram corners look open when zoomed.
            className={`flex ${expanded ? "overflow-hidden" : ""} ${
                align === "right" ? "justify-end" : ""
            }`}
            style={{
                width: expanded
                    ? width
                        ? `max(${SIDE_VOTING}, ${width}px)`
                        : SIDE_REVEALED
                    : (hoverWidth ?? SIDE_VOTING),
                maxWidth: SIDE_MAX,
                transition: expanded
                    ? `width ${EXPAND}`
                    : "width var(--duration-settle) ease-out",
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
        const cards = [cardA.current, cardB.current].filter(
            Boolean,
        ) as HTMLElement[];
        if (cards.length === 0) return;

        const measure = () => {
            const w = Math.ceil(Math.max(...cards.map((c) => c.offsetWidth)));
            if (w > 0)
                setRevealWidth((cur) => (cur === null || w > cur ? w : cur));
        };
        const ro = new ResizeObserver(measure);
        for (const c of cards) ro.observe(c);
        return () => ro.disconnect();
    }, [voted]);

    return (
        <div
            className={`flex items-stretch ${voted ? "" : "[&>*+*]:ml-(--vote-seam)"}`}
            style={
                voted
                    ? undefined
                    : { ["--vote-seam" as string]: `-${SEAM_PULL}px` }
            }
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
                    <VoteButton
                        label="A wins"
                        side="a"
                        onVote={() => onVote("a")}
                    />
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
                    // UNDER THE SLABS, not over them. SKIP's ground is the page's
                    // own black, so painting it above the vote buttons showed as
                    // nothing at all EXCEPT at the seam, where its antialiased
                    // diagonal fell across their pointed corners and blunted
                    // them. Below, the slabs' own edge is the only one drawn
                    // there. Hit testing is unaffected: every member is clipped
                    // to its polygon, so the boxes may overlap but the shapes do
                    // not. See SEAM_PULL.
                    <div className="relative z-0 flex w-full shrink-0">
						<Button
							shape="middle"
							edge="y"
							onClick={() => {
								setSkipHovered(false);
								onVote("skip");
							}}
							onMouseEnter={() => setSkipHovered(true)}
							onMouseLeave={() => setSkipHovered(false)}
							onFocus={() => setSkipHovered(true)}
							onBlur={() => setSkipHovered(false)}
							className="w-full px-xs"
							style={{
								// Same weight as the vertical join between the two scenes —
								// a 1px hairline here read as a different rule from the
								// divider it sits under. Top/bottom only: the slabs own
								// the slanted seams.
								["--btn-edge" as string]:
									"var(--seam-width, 3px)",
							}}
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
                    <VoteButton
                        label="B wins"
                        side="b"
                        onVote={() => onVote("b")}
                    />
                )}
            </Side>
        </div>
    );
}
