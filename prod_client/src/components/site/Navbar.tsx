"use client";

import type { Ref } from "react";
import { usePathname } from "next/navigation";
import Button from "@/components/ui/Button";
import Lockup from "@/components/site/Lockup";

// Paper inversion for type that sits ON the white masthead plate (caption /
// title). The bar itself stays on the page's black tokens — white ink, white mark.
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

/** Offer-label face height in em — lockup mark tracks the same via `--nav-mark`. */
const OFFER_FACE_EM = 1.65;

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

/** ✨ without the yellow — large + small, mass centred on the viewBox. */
function SparklesMark({ className }: { className?: string }) {
	return (
		<svg
			aria-hidden
			viewBox="0 0 24 24"
			fill="currentColor"
			className={className}
		>
			<path d="M10.5 5c0 4.4 3.6 8 8 8-4.4 0-8 3.6-8 8 0-4.4-3.6-8-8-8 4.4 0 8-3.6 8-8z" />
			<path d="M18 3.2c0 2 1.6 3.6 3.6 3.6-2 0-3.6 1.6-3.6 3.6 0-2-1.6-3.6-3.6-3.6 2 0 3.6-1.6 3.6-3.6z" />
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
		<span
			className="inline-grid place-items-center overflow-hidden"
			style={{ minHeight: `${OFFER_FACE_EM}em` }}
		>
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
					className="col-start-1 row-start-1 flex size-full items-center justify-center translate-y-full transition-transform duration-settle ease-out group-hover/btn:translate-y-0"
				>
					<PodiumMark className="size-[1.1em]" />
				</span>
			)}
		</span>
	);
}

/** Generate's label, and the sparkles it gives way to. */
function GenerateLabel() {
	return (
		<span
			className="inline-grid place-items-center overflow-hidden"
			style={{ minHeight: `${OFFER_FACE_EM}em` }}
		>
			<span className="col-start-1 row-start-1 transition-transform duration-settle ease-out group-hover/btn:-translate-y-full">
				Generate
			</span>
			{/* Wrapper matches the cell so translate-y-full clears the clip —
			    sizing the icon alone left it peeking under the shorter glyph. */}
			<span
				aria-hidden
				className="col-start-1 row-start-1 flex size-full items-center justify-center translate-y-full transition-transform duration-settle ease-out group-hover/btn:translate-y-0"
			>
				<SparklesMark className="size-[1.65em]" />
			</span>
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
 * corrects for it. That holds because every control shares one height — the mark
 * is `--nav-mark`, the offer labels are `OFFER_FACE_EM` of `text-sm`, both equal
 * to `py-sm` × 2 of face — so there is no shorter slab for `items-center` to
 * float in extra vertical air past the sides, and the height is `p-sm` twice
 * plus that one control.
 *
 * THE BERTH IS ALSO THE PLATE'S. Masthead measures both clusters and opens the
 * trapezoid between them, so the plate clears the controls and the bar stays
 * straight under all of them.
 */
export default function Navbar({
	leftClusterRef,
	rightClusterRef,
}: {
	/** The reading cluster — Masthead reads its right edge for the plate's chord. */
	leftClusterRef?: Ref<HTMLDivElement>;
	/** The offer cluster — Masthead reads its left edge for the plate's chord. */
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
			className="flex items-center justify-between gap-lg p-sm"
			style={
				{
					// Match the offer face: py-sm × 2 + OFFER_FACE_EM of text-sm.
					"--nav-mark": `calc(2 * var(--spacing-sm) + ${OFFER_FACE_EM} * var(--text-sm))`,
				} as React.CSSProperties
			}
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
						variant="solid"
						edge={false}
						shape="upright-start"
					>
						<BoardLabel onBoard={onBoard} />
					</Button>
				</nav>
				{/* #TODO: no action yet. This should take a prompt from the visitor
				    and queue a build on both models. */}
				<Button variant="outline" shape="upright-end">
					<GenerateLabel />
				</Button>
			</div>
		</header>
	);
}
