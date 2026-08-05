import type { ReactNode } from "react";

/**
 * The frame every page that is NOT the arena sits in.
 *
 * WHY THIS IS NOT `app/layout.tsx`: the arena is a fixed-height, non-scrolling
 * screen with the navbar inside its own flex column and the moonlight laid over
 * everything including the nav. A shared root layout would have to own that
 * arrangement for every route, and the two reading pages want the opposite — normal
 * document flow, scrolling, a measure. One layout serving both would be a pile of
 * conditionals; two frames that each say what they are is less code and less to get
 * wrong.
 *
 * `eyebrow` is the small tracked label above the title. It is a separate prop
 * rather than markup the caller passes because it is the one piece of furniture the
 * two pages must set identically — it is how a reader knows they are still on the
 * same site after leaving the arena.
 */
export default function PageShell({
	eyebrow,
	title,
	lede,
	masthead,
	footer,
	measure = "900px",
	children,
}: {
	/** Both optional: a page whose own content opens it needs no page header. */
	eyebrow?: string;
	title?: string;
	/** Replaces the navbar with a full masthead — the moon and a line on it. */
	masthead?: ReactNode;
	/**
	 * A full-bleed bar closing the page.
	 *
	 * Rendered OUTSIDE `<main>`, which is what makes it full width — main carries
	 * the measure, and anything inside it inherits that measure whether it wants it
	 * or not. A footer that stops at 900px is a card sitting near the bottom, not a
	 * footer.
	 *
	 * STICKY, not fixed. `position: sticky; bottom: 0` keeps the bar on screen while
	 * the page scrolls AND leaves it holding its normal place in flow — so it
	 * reserves its own height instead of floating over the last row, and anything
	 * sizing itself against the window can treat that height as a plain constant.
	 * `fixed` would take it out of flow and hand every other element the job of
	 * remembering to leave room for it.
	 */
	footer?: ReactNode;
	/** One paragraph under the title. Optional: the leaderboard leads with data. */
	lede?: ReactNode;
	/**
	 * How wide the column is.
	 *
	 * A PAGE OF PROSE AND A PAGE OF TABLE WANT DIFFERENT ANSWERS, and 900px is the
	 * prose one — a measure picked so a paragraph is 60-odd characters long. The
	 * leaderboard has no paragraphs in it; it has six columns, and squeezing them
	 * into a reading measure crushes the model names until they ellipsis. So the
	 * default stays what About needs and the board asks for its own.
	 */
	measure?: string;
	/** Optional: a page mid-rebuild is a frame with nothing in it yet. */
	children?: ReactNode;
}) {
	return (
		// A FLEX COLUMN so the footer is pushed to the bottom of the window even when
		// the content does not reach it. `min-h-dvh` alone only guarantees the SHELL
		// is a screen tall — the bar inside it still sat wherever the content ended,
		// with black underneath, which is a bar near the bottom rather than a footer.
		<div className="flex min-h-dvh flex-col bg-ground">
			{masthead}

			{/* THE MEASURE IS THE POINT. The arena is edge-to-edge because it is
			    showing two pictures; a page of prose that ran the full width of a
			    1440px window would be unreadable at any type size, so everything here
			    lives inside one column and every section shares its edges. */}
			{/* TIGHTER THAN A READING PAGE WANTS, because the leaderboard's job is to
			    get as many models above the fold as it can — every step of padding here
			    is a row that does not fit. */}
			{/* NO BOTTOM PADDING WHEN A FOOTER CLOSES THE PAGE. `flex-1` makes main
			    absorb every spare pixel so the bar can sit at the fold, and main's
			    content is top-aligned — so all of that slack collects at its bottom
			    edge, where it reads as a gap between the last thing said and the bar.
			    Its own padding was adding to that gap on top of the slack. The footer
			    is the page's bottom edge; it does not need a margin above it as well. */}
			<main
				className={`mx-auto w-full flex-1 px-lg pt-lg ${footer ? "" : "pb-lg"}`}
				style={{ maxWidth: measure }}
			>
				{/* NO HEADER AT ALL when a page does not ask for one. The leaderboard
				    opens on its champion, which already says what the page is — a title
				    and a strapline above it would be the page introducing something the
				    reader can already see. */}
				{title && (
					<header className="mb-xl">
						{eyebrow && (
							<p className="font-label text-2xs text-accent">{eyebrow}</p>
						)}
						{/* Tight leading and negative tracking: at this size the default
						    spacing of a display line reads as loose, and a heading that is
						    not set as a heading is the fastest way to look unfinished. */}
						<h1 className="mt-sm font-sans text-xl leading-[1.08] font-bold tracking-[-0.02em] text-ink text-balance">
							{title}
						</h1>
						{lede && (
							<div className="mt-md max-w-[62ch] font-sans text-sm leading-[1.65] text-ink-64">
								{lede}
							</div>
						)}
					</header>
				)}

				{children}
			</main>

			{footer && (
				<div className="sticky bottom-0 z-30">{footer}</div>
			)}
		</div>
	);
}
