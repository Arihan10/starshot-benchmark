"use client";

import type { Ref } from "react";
import { usePathname } from "next/navigation";
import Button from "@/components/ui/Button";
import Lockup from "@/components/site/Lockup";

// THE BAR RUNS INVERTED, and it does it by flipping the three colour roles rather
// than by recolouring anything. Every control up here is already written in terms
// of `ground`, `ink` and `mark`; swap what those RESOLVE to for the subtree and
// each variant re-derives on its own — solid comes out black-on-white, ghost keeps
// its hairline but in ink, quiet's labels darken.
//
// `--color-surface` and `--color-accent` are literals in the theme rather than
// built from the rgb triplets, so they have to be named here too: quiet's hover
// ground and the active underline would otherwise stay tuned for a black bar.
export const ON_PAPER = {
	"--ground-rgb": "var(--paper-rgb)",
	"--ink-rgb": "var(--paper-ink-rgb)",
	"--mark-rgb": "var(--paper-ink-rgb)",
	"--surface-rgb": "var(--paper-surface-rgb)",
	"--accent-rgb": "var(--accent-deep-rgb)",
} as React.CSSProperties;

/** What the site is ABOUT: reference material, a reader going looking. */
const READING: { label: string; href: string }[] = [
	{ label: "About", href: "/about" },
	{ label: "FAQ", href: "/faq" },
];

/** Three steps: short, tall, medium — revealed beside Leaderboard on hover. */
function PodiumMark({ className }: { className?: string }) {
	return (
		<svg
			aria-hidden
			viewBox="0 0 24 24"
			fill="currentColor"
			className={className}
		>
			<rect x="1.5" y="13" width="6" height="9" rx="0.5" />
			<rect x="9" y="4" width="6" height="18" rx="0.5" />
			<rect x="16.5" y="9" width="6" height="13" rx="0.5" />
		</svg>
	);
}

/**
 * A reading link, ruled underneath.
 *
 * THE RULE BELONGS TO THE ROW, not to the item. Pointing at one link has to take
 * the line off whichever other one is wearing it, and an item can only know that
 * some OTHER item is hot through `group/nav` on the <nav> above it.
 */
function ReadingLink({
	label,
	href,
	active,
}: {
	label: string;
	href: string;
	active: boolean;
}) {
	return (
		<Button
			href={href}
			aria-current={active ? "page" : undefined}
			variant="quiet"
			shape="square"
		>
			<span className="relative inline-flex">
				{label}
				<span
					aria-hidden
					className={`pointer-events-none absolute left-0 -bottom-0.75 h-[1.5px] bg-mark transition-[width] duration-settle ease-out group-hover/nav:w-0 group-hover/nav:group-hover/btn:w-full ${
						active ? "w-full" : "w-0"
					}`}
				/>
			</span>
		</Button>
	);
}

/**
 * Leaderboard's label, and the podium it gives way to.
 *
 * ONE GRID CELL FOR BOTH, so the slab's width — and the raked edge on it — cannot
 * move as they cross. On the board itself the label stays: the page is already
 * the podium.
 */
function BoardLabel({ onBoard }: { onBoard: boolean }) {
	return (
		<span className="inline-grid place-items-center overflow-hidden">
			<span
				className={`col-start-1 row-start-1 transition-transform duration-settle ease-out ${
					onBoard ? "" : "group-hover/btn:-translate-y-full"
				}`}
			>
				Leaderboard
			</span>
			{!onBoard && (
				<span
					aria-hidden
					className="col-start-1 row-start-1 translate-y-full transition-transform duration-settle ease-out group-hover/btn:translate-y-0"
				>
					<PodiumMark className="size-[1.1em]" />
				</span>
			)}
		</span>
	);
}

/**
 * The site's navbar.
 *
 * TWO CLUSTERS AND THE AIR BETWEEN THEM — the lockup with the reading links, and
 * the offer. `justify-between` is what opens the berth, so the middle is whatever
 * the bar has left over rather than a track of its own, and `gap-lg` is the floor
 * it collapses to before the two can touch.
 *
 * `p-sm` IS THE WHOLE MARGIN, one value on all four sides, and nothing inside
 * corrects for it. That holds because every control's ink runs to its own box
 * edge — the mark is sized to the disc rather than to artwork with a transparent
 * border (see Lockup) — so there is no slack on one side of the bar to answer for
 * on the other, and the height is `p-sm` twice plus the tallest control.
 *
 * THE BERTH IS ALSO THE MOON'S. Masthead measures both clusters and opens the lip
 * between them, so the curve clears the controls and the bar stays straight under
 * all of them.
 */
export default function Navbar({
	leftClusterRef,
	rightClusterRef,
}: {
	/** The reading cluster — Masthead reads its right edge for the moon's chord. */
	leftClusterRef?: Ref<HTMLDivElement>;
	/** The offer cluster — Masthead reads its left edge for the moon's chord. */
	rightClusterRef?: Ref<HTMLDivElement>;
}) {
	const pathname = usePathname();

	// The route itself or one below it. A bare `startsWith` would light ABOUT up on
	// any route that merely begins with those letters.
	const isHere = (href: string) =>
		pathname === href || pathname.startsWith(`${href}/`);

	const onBoard = isHere("/leaderboard");

	return (
		<header
			style={ON_PAPER}
			className="flex items-center justify-between gap-lg p-sm"
		>
			<div ref={leftClusterRef} className="flex items-center gap-md">
				<Lockup />
				<nav
					aria-label="About SceneBench"
					className="group/nav flex items-center"
				>
					{READING.map((item) => (
						<ReadingLink
							key={item.href}
							{...item}
							active={isHere(item.href)}
						/>
					))}
				</nav>
			</div>

			<div ref={rightClusterRef} className="flex items-center gap-2xs">
				<nav aria-label="SceneBench standings" className="flex items-center">
					<Button
						href="/leaderboard"
						aria-current={onBoard ? "page" : undefined}
						variant="outline"
						shape="upright-start"
					>
						<BoardLabel onBoard={onBoard} />
					</Button>
				</nav>
				{/* #TODO: no action yet. This should take a prompt from the visitor
				    and queue a build on both models. */}
				<Button variant="cta" sweep shape="upright-end">
					Generate
				</Button>
			</div>
		</header>
	);
}
