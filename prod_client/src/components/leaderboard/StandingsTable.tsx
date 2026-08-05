"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Standing } from "@/lib/leaderboard";
import BrandMark from "./BrandMark";

/**
 * The board: four columns, sortable, and nothing else.
 *
 * NO SEARCH AND NO FILTERS. Every control above a table is a decision a reader has
 * to make before being shown anything, and this one has twenty-two rows in it —
 * fewer than a search box would save anyone reading. Sorting is different: it is
 * not a question the reader has to answer, it is four different views of the same
 * board, and the column heads already name them.
 *
 * A REAL `<table>`. This is tabular data, and a screen reader announcing "row 4,
 * model, Claude Sonnet 4.5, elo 1398" is worth more than the convenience of a grid
 * of divs. It also means `aria-sort` on a column head does the whole job of telling
 * an assistive reader what the board is currently ordered by.
 */

type Key = "name" | "elo" | "winRate" | "votes";

const COLUMNS: {
	key: Key;
	label: string;
	/**
	 * WHICH WAY IT GOES ON THE FIRST PRESS, and it is not the same for every
	 * column. Every figure here is "more is better", so a reader pressing ELO wants
	 * the best at the top — descending. A NAME has no better, only an order, and
	 * the order anyone expects from a list of names is A to Z. Starting both the
	 * same way would mean one of the two always took two presses to do the obvious
	 * thing.
	 */
	first: "asc" | "desc";
}[] = [
	{ key: "name", label: "Model", first: "asc" },
	{ key: "elo", label: "Elo", first: "desc" },
	{ key: "winRate", label: "Win rate", first: "desc" },
	{ key: "votes", label: "Votes", first: "desc" },
];

/** The readout voice: small monospace capitals, tracked wide. */
const READOUT = "font-mono uppercase";

/**
 * The height of the slot the SHOW MORE button sits in.
 *
 * A CONSTANT, AND RESERVED WHETHER OR NOT THE BUTTON IS THERE. If the slot
 * collapsed once every model was on show, the space would come back, another row
 * would fit, and the count would shift under the reader who had just pressed it.
 */
const TAIL = 58;

export default function StandingsTable({ rows }: { rows: Standing[] }) {
	// ELO, HIGHEST FIRST — a leaderboard's one obvious opening state, and the only
	// one that makes the first row mean what the podium above it says.
	const [sort, setSort] = useState<{ key: Key; dir: "asc" | "desc" }>({
		key: "elo",
		dir: "desc",
	});

	const [query, setQuery] = useState("");

	const sorted = useMemo(() => {
		// ACROSS BOTH NAMES, because a reader does not hold the two apart. Half of
		// them will type "anthropic" looking for Claude and half will type "claude"
		// looking for Anthropic, and neither has got it wrong — the model and the lab
		// that built it are one fact with two names on it.
		const q = query.trim().toLowerCase();
		const list = q
			? rows.filter(
					(r) =>
						r.name.toLowerCase().includes(q) || r.lab.toLowerCase().includes(q),
				)
			: [...rows];
		// SORTED ASCENDING ONCE, THEN REVERSED. Two comparators — one per direction —
		// is two places for a tie to be broken differently, and the reversal is exact.
		list.sort((a, b) =>
			sort.key === "name"
				? a.name.localeCompare(b.name)
				: a[sort.key] - b[sort.key],
		);
		return sort.dir === "desc" ? list.reverse() : list;
	}, [rows, sort, query]);

	const press = (column: (typeof COLUMNS)[number]) => {
		setSort((current) =>
			current.key === column.key
				? // ALREADY SORTING BY THIS ONE: turn it round.
					{ key: column.key, dir: current.dir === "asc" ? "desc" : "asc" }
				: { key: column.key, dir: column.first },
		);
	};

	// --- how much of the board is on show ----------------------------------
	//
	// HOW MANY ROWS FIT, MEASURED rather than assumed. A fixed page size is wrong on
	// every window except the one it was picked on: it leaves half a screen empty on
	// a tall display and makes a short one scroll to reach a button that says "show
	// more", which is the worst of both.
	const frame = useRef<HTMLDivElement>(null);
	const find = useRef<HTMLDivElement>(null);
	const head = useRef<HTMLTableSectionElement>(null);
	const body = useRef<HTMLTableSectionElement>(null);
	const [fits, setFits] = useState(7);
	const [pages, setPages] = useState(1);

	useEffect(() => {
		const measure = () => {
			const box = frame.current;
			const search = find.current;
			const thead = head.current;
			const row = body.current?.querySelector("tr");
			if (!box || !search || !thead || !row) return;
			const rowHeight = row.getBoundingClientRect().height;
			if (!rowHeight) return;

			// MEASURED AGAINST THE FRAME, NOT THE SCROLLING BOX. The box now sizes
			// itself to its own rows — that is what closed the gap under the last one —
			// so its height is an OUTPUT of this calculation and reading it as an input
			// would be a feedback loop: fewer rows, shorter box, smaller budget, fewer
			// rows. Those do not converge, they oscillate until the ResizeObserver
			// gives up delivering.
			//
			// The frame is `flex-1` inside a section exactly one screen tall, so its
			// height is free space rather than content. Everything taken off it is
			// chrome that does not move: the search, one row of column heads, and the
			// button's slot.
			const budget =
				box.clientHeight -
				search.getBoundingClientRect().height -
				thead.getBoundingClientRect().height -
				TAIL;

			// A PIXEL OF SLACK. Row heights are fractional and the browser rounds each
			// one as it lays it out, so fifteen of them can come to a pixel more than
			// fifteen times the height measured off the first — and one pixel of
			// overflow is a scrollbar on a view that is meant not to have one.
			setFits(Math.max(1, Math.floor((budget - 1) / rowHeight)));
		};
		// A ResizeObserver rather than a resize listener: it fires once on observe,
		// which covers the first measurement, and again for anything that changes the
		// page's height — including a webfont landing, which a resize listener misses.
		const observer = new ResizeObserver(measure);
		observer.observe(document.documentElement);
		return () => observer.disconnect();
	}, []);

	const limit = Math.min(sorted.length, pages * fits);
	const visible = sorted.slice(0, limit);
	const rest = sorted.length - limit;

	return (
		// THE HEAD STAYS, THE BODY SCROLLS — but only once a reader has ASKED for
		// more than a screenful. Until then the board shows exactly what fits and
		// there is nothing to scroll.
		//
		// NO `overscroll-contain` HERE, and that is deliberate. Contained, a reader
		// who had expanded the board and scrolled back to its top would be stuck:
		// the wheel would keep hitting the end of this box and never chain out to the
		// page behind it, so there would be no way back up to the podium.
		<div ref={frame} className="flex min-h-0 flex-1 flex-col">
			{/* --- the search ------------------------------------------------------
			    A BARE FIELD ON A RULE, which is the same treatment the arena's
			    composer gets, because it is the same kind of object: somewhere you
			    type, on a page that is otherwise all surfaces. A boxed input would be
			    the only bordered control on the section.

			    FULL WIDTH, so its rule is the board's own top edge rather than a
			    stub over one corner. Held to a measure it read as a stray control
			    floating above the table instead of as the head of it. */}
			{/* THE GAP UNDER IT IS PADDING, NOT MARGIN, and it has to be: the budget
			    above measures this element with `getBoundingClientRect`, which counts
			    borders and padding and does NOT count margins. A margin here would be
			    space the row calculation could not see, and the board would overflow
			    by exactly that much. */}
			<div ref={find} className="flex-none pb-xs">
			<div className="flex items-center gap-sm border-b border-mark-16 pb-xs">
				<svg
					aria-hidden
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					strokeWidth="2.2"
					className="size-[15px] flex-none text-ink-40"
				>
					<circle cx="10.5" cy="10.5" r="6.5" />
					<path d="M15.5 15.5 L21 21" strokeLinecap="round" />
				</svg>
				<input
					value={query}
					onChange={(e) => setQuery(e.target.value)}
					// ESCAPE CLEARS IT, because that is what escape does in every search
					// field a reader has ever used, and it costs one line.
					onKeyDown={(e) => {
						if (e.key === "Escape") setQuery("");
					}}
					placeholder="Search model or lab"
					spellCheck={false}
					aria-label="Search models and labs"
					// MONO, like every other label on this board. The column heads, the
					// lab names, the deltas and the Clear button beside it are all the
					// READOUT voice; a grotesque field in the middle of them was the
					// one thing on the section set in a different face.
					className="w-full min-w-0 bg-transparent font-mono text-[13px] tracking-[0.02em] text-ink outline-none placeholder:text-ink-40"
				/>
				{/* Only while there is something to clear. A permanently present clear
				    button on an empty field is a control that does nothing, which
				    teaches a reader to ignore that corner. */}
				{query !== "" && (
					<button
						type="button"
						onClick={() => setQuery("")}
						className={`${READOUT} flex-none cursor-pointer text-[9.5px] font-bold tracking-[0.2em] text-ink-40 transition-colors duration-[160ms] hover:text-ink`}
					>
						Clear
					</button>
				)}
			</div>
			</div>

			{/* THE HEAD STAYS, THE BODY SCROLLS — but only once a reader has ASKED for
			    more than a screenful. Until then the board shows exactly what fits and
			    there is nothing to scroll.

			    NOT `flex-1`, and that is what closed the gap under the last row. Told
			    to fill the frame, this box held the whole of the free space whether or
			    not the rows did — and since the row count is a `floor()`, there was
			    always the better part of a row left over, sitting empty between the
			    last entry and the button. Sized to its content it ends where the rows
			    end, and the slack collects below the button instead, where it reads as
			    margin.

			    NO `overscroll-contain`, deliberately. Contained, a reader who had
			    expanded the board and scrolled back to its top would be stuck: the
			    wheel would keep hitting the end of this box and never chain out to the
			    page behind it, so there would be no way back up to the podium. */}
			<div className="min-h-0 overflow-y-auto">
				<table className="w-full table-fixed border-collapse text-left">
					<caption className="sr-only">
						Model standings, sorted by{" "}
						{COLUMNS.find((c) => c.key === sort.key)?.label},{" "}
						{sort.dir === "desc" ? "highest first" : "lowest first"}. Showing{" "}
						{limit} of {sorted.length}.
					</caption>
					<colgroup>
						<col />
						<col className="w-[150px]" />
						<col className="w-[210px]" />
						<col className="w-[130px]" />
					</colgroup>
					<thead ref={head}>
						{/* STICKY, and its rule is a shadow rather than a border: under
						    `border-collapse` a sticky cell's own border is left behind as
						    the body scrolls under it, which draws the line in the wrong
						    place. An inset shadow travels with the cell. */}
						<tr>
							{COLUMNS.map((column, i) => {
								const active = sort.key === column.key;
								return (
									<th
										key={column.key}
										scope="col"
										aria-sort={
											active
												? sort.dir === "asc"
													? "ascending"
													: "descending"
												: "none"
										}
										className={`sticky top-0 z-10 bg-ground p-0 shadow-[inset_0_-1px_0_var(--color-mark-16)] ${
											i === 0 ? "pl-[22px]" : ""
										} ${i === COLUMNS.length - 1 ? "pr-[22px]" : ""}`}
									>
										{/* CENTRED ON THE COLUMN, and so is everything under it.
										    What matters is that the two AGREE: set to opposite
										    edges, a short head like ELO floated over the right end
										    of its own numbers with nothing beneath its first
										    letter, which is what stopped this reading as a table.
										    The figures are tabular, so digits still line up with
										    each other down the column whatever the column does.

										    MODEL is the exception and stays left: it is a name and
										    a mark, not a measurement, and a centred column of
										    names gives the eye no edge to run down. */}
										<button
											type="button"
											onClick={() => press(column)}
											className={`${READOUT} flex h-[38px] w-full cursor-pointer items-center gap-xs px-[9px] text-[9.5px] font-bold tracking-[0.22em] transition-colors duration-[160ms] hover:text-ink ${
												i === 0 ? "justify-start" : "justify-center"
											} ${active ? "text-ink" : "text-ink-40"}`}
										>
											{column.label}
											{/* THE ARROW ONLY EXISTS ON THE ACTIVE COLUMN. A
											    permanent indicator on every head is three claims
											    that the board is sorted four ways at once. */}
											<span
												aria-hidden
												className={`text-[8px] leading-none transition-opacity duration-[160ms] ${
													active ? "opacity-100" : "opacity-0"
												}`}
											>
												{sort.dir === "desc" ? "▼" : "▲"}
											</span>
										</button>
									</th>
								);
							})}
						</tr>
					</thead>
					<tbody ref={body}>
						{visible.map((row) => (
							// The row lights and gains an edge at the left margin — the wash
							// alone is one step of the ink ramp and reads as an artefact on a
							// black screen. The rule is drawn on the first cell because a
							// row's own box-shadow is a coin flip across browsers under
							// `border-collapse`.
							<tr
								key={row.name}
								className="group/row h-[clamp(56px,5.8vh,68px)] border-b border-mark-8 transition-colors duration-[160ms] hover:bg-mark-8"
							>
								<td className="pr-[9px] pl-[22px] align-middle transition-shadow duration-[160ms] group-hover/row:shadow-[inset_3px_0_0_var(--color-mark)]">
									<div className="flex min-w-0 items-center gap-[14px]">
										{/* NO TILE ROUND IT. The box was doing a job the marks can
										    do themselves — every one of these is a strong, closed
										    silhouette in its own colour, and a hairline square
										    round each turned a column of logos into a column of
										    BOXES with logos in them. The slot stays, at a fixed
										    size and with nothing drawn on it, because that is what
										    keeps the names on one left margin whatever shape the
										    mark inside happens to be. */}
										<span className="grid size-[30px] flex-none place-items-center">
											<BrandMark lab={row.lab} size={28} />
										</span>
										<span className="flex min-w-0 flex-col gap-[5px]">
											<span className="truncate text-[17px] leading-none font-extrabold text-ink">
												{row.name}
											</span>
											<span
												className={`${READOUT} truncate text-[9px] leading-none font-bold tracking-[0.18em] text-ink-40`}
											>
												{row.lab}
											</span>
										</span>
									</div>
								</td>

								{/* ELO AND ITS WEEK, SIDE BY SIDE. The delta had a column of its
								    own and did not earn one: it is not a fourth measurement, it
								    is the second half of this one — where the rating is now, and
								    which way it has been going. Read across, that is one
								    statement instead of two.

								    ON THE BASELINE, not the centre: a small figure set beside a
								    large one belongs on the line the large one sits on, or it
								    reads as floating halfway up it.

								    AND THE DELTA'S SLOT IS FIXED. Its width varies with the
								    figure — one digit or two, and the sign is a glyph — so left
								    to itself it would shift the rating a few pixels sideways on
								    every row, which is the one thing a column of numbers must
								    not do. Given a constant width, the seam between the two
								    never moves. */}
								<td className="px-[9px] text-center align-middle">
									<span className="flex items-baseline justify-center gap-[9px]">
										<span className="text-[21px] leading-none font-extrabold tracking-[-0.02em] tabular-nums text-ink">
											{row.elo}
										</span>
										{/* Sign carried by the glyph AND the colour, never colour
										    alone — a red number and a green number are the same
										    number to a reader who cannot tell them apart. */}
										<span
											className={`${READOUT} w-[42px] flex-none text-left text-[10px] leading-none font-bold tracking-[0.1em] tabular-nums ${
												row.delta >= 0 ? "text-rise" : "text-fall"
											}`}
										>
											{row.delta >= 0 ? "▲" : "▼"} {Math.abs(row.delta)}
										</span>
									</span>
								</td>

								{/* A BAR AND THE NUMBER. The figure is the answer; the bar is
								    what lets you read the whole column's shape without
								    comparing twenty decimals to each other.

								    BOTH FIXED, so the pair is a block of known width and can be
								    centred under its head as one thing. The bar used to take the
								    column's slack, which meant there was no slack left to centre
								    it in — it would have sat wherever the column ended. */}
								<td className="px-[9px] align-middle">
									<div className="flex items-center justify-center gap-[14px]">
										<span
											aria-hidden
											className="h-[3px] w-[86px] flex-none bg-mark-16"
										>
											<span
												className="block h-full bg-mark"
												style={{ width: `${row.winRate}%` }}
											/>
										</span>
										<span
											className={`${READOUT} w-[54px] flex-none text-right text-[12px] font-medium tracking-[0.04em] tabular-nums text-ink`}
										>
											{row.winRate.toFixed(1)}%
										</span>
									</div>
								</td>

								<td className="pr-[22px] pl-[9px] text-center align-middle">
									<span
										className={`${READOUT} text-[12px] font-medium tracking-[0.06em] tabular-nums text-ink-40`}
									>
										{row.votes.toLocaleString("en-US")}
									</span>
								</td>
							</tr>
						))}
					</tbody>
				</table>

				{/* A SEARCH THAT MATCHES NOTHING HAS TO SAY SO. An empty table under a
				    populated set of column heads reads as a page that has failed to
				    load, and a reader's next move is to reload it rather than to fix
				    their spelling. */}
				{sorted.length === 0 && (
					<p
						className={`${READOUT} py-xl text-center text-[10.5px] font-bold tracking-[0.22em] text-ink-40`}
					>
						No model or lab matches “{query.trim()}”
					</p>
				)}
			</div>

			{/* --- the tail -------------------------------------------------------
			    OUTSIDE THE SCROLLING BOX, AND ALWAYS THIS TALL. Both halves of that
			    are load-bearing. Outside, so the button is reachable without first
			    scrolling to the end of the very list it is offering to extend — and
			    so its height is already excluded from the row budget above. Always,
			    so that budget is a constant: if the slot collapsed once every model
			    was on show, the space would come back, more rows would fit, and the
			    count would shift under the reader who had just pressed it. */}
			<div
				className="flex flex-none items-center justify-center"
				style={{ height: TAIL }}
			>
				{rest > 0 && (
					<button
						type="button"
						onClick={() => setPages((p) => p + 1)}
						className={`${READOUT} cursor-pointer border border-mark-16 px-[26px] py-[11px] text-[10.5px] font-bold tracking-[0.22em] text-ink-64 transition-colors duration-[160ms] hover:border-mark hover:text-ink`}
					>
						Show more
					</button>
				)}
			</div>
		</div>
	);
}
