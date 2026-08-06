"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Standing } from "@/lib/leaderboard";
import BrandMark from "./BrandMark";

type Key = "name" | "elo" | "winRate" | "votes";

const COLUMNS: {
	key: Key;
	label: string;
	first: "asc" | "desc";
}[] = [
	{ key: "name", label: "Model", first: "asc" },
	{ key: "elo", label: "Elo", first: "desc" },
	{ key: "winRate", label: "Win rate", first: "desc" },
	{ key: "votes", label: "Votes", first: "desc" },
];

const READOUT = "font-mono uppercase";

// HOW DEEP THE FADE AT THE FOOT RUNS, in px — a little over one row, so the last
// line you can read is whole and the one behind it is the one dissolving. Much
// shorter and it stops reading as a fade and starts reading as a row that failed
// to paint; much longer and it eats a legible row for the sake of the effect.
const TAIL = 72;

// THE FOOT DISSOLVES, IT IS NOT COVERED OVER. The obvious version of this is a
// gradient from the page colour laid over the bottom of the list — and it cannot
// work here, because there is no page colour to fade to: the board sits over a
// live 3D city, and a strip of flat `ground` painted across it would be a grey
// band hanging in front of the skyline. A MASK removes the type instead, so what
// comes through as the rows go is whatever is actually behind them.
//
// THE BAND IS ANCHORED TO THE BOX, NOT THE CONTENT. A mask resolves against the
// element's own padding box, and for a scroller that box is the WINDOW rather
// than the scrolled material — which is what makes rows dissolve as they reach
// the bottom edge rather than a fixed row somewhere in the list being faded.
//
// AND IT RETRACTS AT THE END OF THE LIST. `--tail` runs 1 while there is more
// below and eases to 0 over the last `TAIL` pixels of travel, so the gradient's
// stop walks back to 100% and the final row arrives at full strength. Left at a
// constant, the bottom of the table stays permanently half-erased once you have
// scrolled to it, which reads as the list still continuing — the exact thing a
// scroll hint is supposed to tell you is no longer true.
const TAIL_MASK = `linear-gradient(to bottom, #000 calc(100% - var(--tail, 0) * ${TAIL}px), transparent 100%)`;


export default function StandingsTable({
	rows,
	onCompare,
}: {
	rows: Standing[];
	/**
	 * Called with the row under the pointer, and with null when it leaves.
	 *
	 * THE TABLE DOES NOT KNOW WHAT THIS IS FOR. It reports which model is being
	 * looked at; what the page does with that — raise a pillar for it out in the
	 * city, and stand it next to the podium — is the page's business. Reaching
	 * into the scene from here would tie a list of numbers to a WebGL canvas and
	 * make the board unusable anywhere the canvas is not.
	 */
	onCompare?: (row: Standing | null) => void;
}) {
	const [sort, setSort] = useState<{ key: Key; dir: "asc" | "desc" }>({
		key: "elo",
		dir: "desc",
	});

	const [query, setQuery] = useState("");

	const sorted = useMemo(() => {
		const q = query.trim().toLowerCase();
		const list = q
			? rows.filter(
					(r) =>
						r.name.toLowerCase().includes(q) || r.lab.toLowerCase().includes(q),
				)
			: [...rows];
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
				?
					{ key: column.key, dir: current.dir === "asc" ? "desc" : "asc" }
				: { key: column.key, dir: column.first },
		);
	};

	const frame = useRef<HTMLDivElement>(null);
	const find = useRef<HTMLDivElement>(null);
	const scroller = useRef<HTMLDivElement>(null);
	const sheet = useRef<HTMLTableElement>(null);
	const body = useRef<HTMLTableSectionElement>(null);
	const tail = useRef<HTMLDivElement>(null);

	// HOW MANY ROWS THE COLUMN HOLDS, and how many pages of them have been asked
	// for. The board shows a page at a time again rather than one long scroller:
	// a list that ends where the column ends, with a control that says how to see
	// more, states its own length — a scroller only admits it once you are already
	// dragging. The seed is a plausible count so the first paint is not empty; it
	// is replaced the moment anything can be measured.
	const [fits, setFits] = useState(7);
	const [pages, setPages] = useState(1);

	// MEASURED, NOT ASSUMED. Every part of the frame that is not rows — the search
	// field, the sticky header, the footer holding the control — is subtracted
	// from the frame's own height, and what is left is divided by a row. All four
	// move: the rows carry a `vh` clamp, and the type they hold settles again when
	// the webfont lands, which is why `fonts.ready` gets a measurement of its own.
	useEffect(() => {
		const measure = () => {
			const box = frame.current;
			const search = find.current;
			const head = sheet.current?.tHead;
			const foot = tail.current;
			const row = body.current?.querySelector("tr");
			if (!box || !search || !head || !foot || !row) return;
			const rowHeight = row.getBoundingClientRect().height;
			if (!rowHeight) return;

			const budget =
				box.clientHeight -
				search.getBoundingClientRect().height -
				head.getBoundingClientRect().height -
				foot.getBoundingClientRect().height;

			setFits(Math.max(1, Math.floor((budget - 1) / rowHeight)));
		};
		const observer = new ResizeObserver(measure);
		if (frame.current) observer.observe(frame.current);
		observer.observe(document.documentElement);
		document.fonts?.ready.then(measure).catch(() => {});
		return () => observer.disconnect();
	}, []);

	// A NEW ARRANGEMENT IS A NEW LIST, so it starts at one page. Carrying the
	// count across a search would open the results already expanded, which reads
	// as the filter having failed to narrow anything.
	useEffect(() => setPages(1), [sorted]);

	const limit = Math.min(sorted.length, pages * fits);
	const visible = sorted.slice(0, limit);
	const rest = sorted.length - limit;


	// HOW MUCH IS LEFT BELOW, written to the element as a 0–1 number and read by
	// the mask. Set on the DOM node directly rather than through state: this fires
	// on every scroll event, and a `setState` per frame would re-render 23 rows to
	// change one gradient stop.
	useEffect(() => {
		const box = scroller.current;
		const table = sheet.current;
		if (!box || !table) return;

		const sync = () => {
			const below = box.scrollHeight - box.clientHeight - box.scrollTop;
			box.style.setProperty("--tail", String(Math.min(1, below / TAIL)));
		};

		sync();
		box.addEventListener("scroll", sync, { passive: true });
		// TWO THINGS CHANGE THE ANSWER and only one of them is a scroll. The box
		// resizing is the obvious one; the CONTENT changing height is the other —
		// filtering the list to three rows leaves the box exactly as it was, so an
		// observer watching only the box would keep a fade over a list that no
		// longer overflows. Watching the table catches both, since it lives inside
		// the box, but the box is watched too: it can be resized by the window
		// without the table's own height moving at all.
		const observer = new ResizeObserver(sync);
		observer.observe(box);
		observer.observe(table);
		return () => {
			box.removeEventListener("scroll", sync);
			observer.disconnect();
		};
	}, []);

	// ONE WEIGHT, DECLARED ONCE, FOR THE WHOLE BOARD. `font-weight` inherits, so
	// the root sets it and every cell, header, readout and control below takes it —
	// including the sort buttons and the search field, which Tailwind's preflight
	// gives `font: inherit` precisely so they stop opting out of the page's type.
	//
	// It replaces a per-element spread of medium/bold/extrabold that had no system
	// behind it: the win rate and the vote count sat at 500 while the name beside
	// them sat at 800, which read as three tables stacked rather than one. A board
	// is a single instrument and every number on it carries the same authority.
	//
	// WORTH KNOWING WHAT 900 ACTUALLY BUYS: Manrope is the face for both `sans` and
	// `mono` here and its variable `wght` axis stops at 800, so `font-black` renders
	// AS `font-extrabold` — it does not get heavier than what the name was already
	// set in. The gain is on everything BELOW that ceiling, and the reason to write
	// black rather than extrabold is that it is the heaviest this can be asked for:
	// swap in a face with a true 900 and the board takes it without an edit.
	return (
		<div ref={frame} className="flex min-h-0 flex-1 flex-col font-black">

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
					onKeyDown={(e) => {
						if (e.key === "Escape") setQuery("");
					}}
					placeholder="Search model or lab"
					spellCheck={false}
					aria-label="Search models and labs"
					className="w-full min-w-0 bg-transparent font-mono text-[13px] tracking-[0.02em] text-ink outline-none placeholder:text-ink-40"
				/>
				{query !== "" && (
					<button
						type="button"
						onClick={() => setQuery("")}
						className={`${READOUT} flex-none cursor-pointer text-[9.5px] tracking-[0.2em] text-ink-40 transition-colors duration-[160ms] hover:text-ink`}
					>
						Clear
					</button>
				)}
			</div>
			</div>

			{/* THE WHOLE LIST IS HERE, and scrolling is how you reach it. It used to be
			    cut to whatever fitted the viewport with a SHOW MORE under it, which
			    made the board's own height a thing that had to be measured — row
			    height, header height, search height, footer height, re-measured on
			    every resize and after the webfont landed — to answer a question the
			    scrollbar answers for free.

			    THE FOOT IS MASKED, NOT COVERED — see TAIL_MASK. Both properties are
			    written because the prefixed one is still what older WebKit reads. */}
			<div
				ref={scroller}
				className="min-h-0 overflow-y-auto"
				style={{ maskImage: TAIL_MASK, WebkitMaskImage: TAIL_MASK }}
			>
				<table
					ref={sheet}
					className="w-full table-fixed border-collapse text-left"
				>
					<caption className="sr-only">
						Model standings, sorted by{" "}
						{COLUMNS.find((c) => c.key === sort.key)?.label},{" "}
						{sort.dir === "desc" ? "highest first" : "lowest first"}.{" "}
						Showing {visible.length} of {sorted.length}.
					</caption>
					<colgroup>
						{/* WIDENED FOR THE NUMERALS, and it has to be: the table is
						    `table-fixed`, so a column too narrow for its content does not
						    grow — the digits simply run under the model name beside them.
						    Two 27px figures measure ~33px, and the cell spends 40 of its
						    width on insets before they start. */}
						{/* CUT TO THE COLUMN THE BOARD NOW LIVES IN. These were set
						    against the full width of the page and carried the slack to
						    prove it — a 210px cell for a bar and a percentage, 130 for
						    a four-figure number. In half the width that slack is the
						    difference between a model name that reads and one that
						    truncates, so it has been given to the only column that
						    holds words.

						    THE READINGS KEEP THEIR ORDER OF SIZE, which is what the
						    trimming has to protect: rank and Elo are what the board is
						    FOR and stay unhurried, while votes — the least-read figure
						    here — gives up the most. */}
						<col className="w-[72px]" />
						<col />
						<col className="w-[132px]" />
						<col className="w-[148px]" />
						<col className="w-[96px]" />
					</colgroup>
					<thead>
						<tr>
							<th
								scope="col"
								className="sticky top-0 z-10 bg-ground p-0 pl-[22px] shadow-[inset_0_-1px_0_var(--color-mark-16)]"
							>
								<span
									className={`${READOUT} flex h-[38px] items-center px-[9px] text-[9.5px] tracking-[0.22em] text-ink-40`}
								>
									Rank
								</span>
							</th>
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
											i === COLUMNS.length - 1 ? "pr-[22px]" : ""
										}`}
									>
										<button
											type="button"
											onClick={() => press(column)}
											className={`${READOUT} flex h-[38px] w-full cursor-pointer items-center gap-xs px-[9px] text-[9.5px] tracking-[0.22em] transition-colors duration-[160ms] hover:text-ink ${
												i === 0 ? "justify-start" : "justify-center"
											} ${active ? "text-ink" : "text-ink-40"}`}
										>
											{column.label}
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
						{visible.map((row, i) => (
							<tr
								key={row.name}
								// THE CASCADE RESTARTS PER PAGE, which is why the delay is taken
								// modulo the page length rather than from the row's index in the
								// list. Straight off the index, the first row revealed by SHOW
								// MORE would wait out the whole page above it before appearing.
								//
								// Only ever on the way in: sorting and filtering keep these keys,
								// so React reuses the rows and no animation re-runs on them.
								style={{
									["--lift-delay" as string]: `${(i % Math.max(1, fits)) * 24}ms`,
									["--lift-dur" as string]: "360ms",
									["--lift-from" as string]: "6px",
								}}
								// HOVER IS THE WHOLE INTERACTION, so it is reported from the
								// row itself rather than from any one cell — the reader is
								// pointing at a MODEL, and every part of the line is that
								// model. Cleared on leave rather than on entering the next
								// row, because the pointer can leave the list entirely and
								// nothing else would ever say so.
								onMouseEnter={() => onCompare?.(row)}
								onMouseLeave={() => onCompare?.(null)}
								// THE ROW INVERTS RATHER THAN TINTS. Every value in it is already
							// written against `ink` and `mark`, so re-pointing those two for
							// the row is the whole change — the name, the rank, the elo, the
							// win bar and the rule under it all re-derive on their own. Naming
							// a hover colour per element would be the same edit five times.
							//
							// The spans carry the transition themselves: a transition on the
							// row does not cascade, and without it the type would snap while
							// the ground behind it faded.
							className="lift-in group/row h-[clamp(56px,5.8vh,68px)] border-b border-mark-8 transition-colors duration-[160ms] hover:bg-paper hover:[--ink-rgb:var(--paper-ink-rgb)] hover:[--mark-rgb:var(--paper-ink-rgb)] [&_span]:transition-colors [&_span]:duration-[160ms]"
							>
								{/* THE EDGE MARKER, and it is written as the TRIPLET rather than
								    as `var(--color-mark)` — which is what it was, and it was
								    invisible. `--color-mark` is declared once on `:root`, so its
								    `var(--mark-rgb)` is substituted THERE and it inherits the
								    root's near-white as a literal; the row re-points `--mark-rgb`
								    just above and this never heard about it. A 237-white bar on
								    the 236-cream it lands on is a bar nobody can see. Same trap as
								    the brand marks — see the note at BrandMark's INK. */}
								<td className="pl-[22px] align-middle transition-shadow duration-[160ms] group-hover/row:shadow-[inset_3px_0_0_rgb(var(--mark-rgb))]">
									{/* THE RANK IS THE BIGGEST THING IN THE ROW, and it should be:
									    it is the one number the board exists to state, and at
									    15px in 40% ink it was set as a row label — smaller and
									    fainter than the Elo it was supposedly ordering.

									    FULL INK, not the 40% it sat at. Standings are read down
									    the left edge, so the ranks are a column the eye tracks
									    rather than a value it looks up per row, and a tracked
									    column has to hold at a glance.

									    TRACKING COMES OFF. The 0.04em here was doing what letter
									    spacing does for small readout type — opening it up so it
									    stays legible — and at 27px that same value just pushes
									    two digits apart. Big numerals want to be tight.

									    `leading-none` so the row keeps its measured height: the
									    rows are the unit the visible-count is computed from, and
									    a taller line here silently drops one off the page. */}
									<span
										className={`${READOUT} block px-[9px] text-[27px] leading-none tracking-[-0.01em] text-ink tabular-nums`}
									>
										{String(row.rank).padStart(2, "0")}
									</span>
								</td>
								<td className="pr-[9px] align-middle">
									<div className="flex min-w-0 items-center gap-[14px]">
										<span className="grid size-[30px] flex-none place-items-center">
											<BrandMark lab={row.lab} size={28} />
										</span>
										<span className="flex min-w-0 flex-col gap-[5px]">
											<span className="truncate text-[17px] leading-none text-ink">
												{row.name}
											</span>
											<span
												className={`${READOUT} truncate text-[9px] leading-none tracking-[0.18em] text-ink-40`}
											>
												{row.lab}
											</span>
										</span>
									</div>
								</td>

								<td className="px-[9px] text-center align-middle">
									<span className="flex items-baseline justify-center gap-[9px]">
										<span className="text-[21px] leading-none tracking-[-0.02em] tabular-nums text-ink">
											{row.elo}
										</span>
										<span
											className={`${READOUT} w-[42px] flex-none text-left text-[10px] leading-none tracking-[0.1em] tabular-nums ${
												row.delta >= 0 ? "text-rise" : "text-fall"
											}`}
										>
											{row.delta >= 0 ? "▲" : "▼"} {Math.abs(row.delta)}
										</span>
									</span>
								</td>

								<td className="px-[9px] align-middle">
									{/* THE BAR GIVES UP THE WIDTH, NOT THE FIGURE. Both had to
									    give something back to fit the narrower column, and only
									    one of the two is read for a value — the bar is here to
									    make the column scannable down its length, and it does
									    that at any width so long as every row shares it. */}
									<div className="flex items-center justify-center gap-[10px]">
										<span
											aria-hidden
											className="h-[3px] w-[56px] flex-none bg-mark-16"
										>
											<span
												className="block h-full bg-mark"
												style={{ width: `${row.winRate}%` }}
											/>
										</span>
										<span
											className={`${READOUT} w-[46px] flex-none text-right text-[12px] tracking-[0.04em] tabular-nums text-ink`}
										>
											{row.winRate.toFixed(1)}%
										</span>
									</div>
								</td>

								<td className="pr-[22px] pl-[9px] text-center align-middle">
									<span
										className={`${READOUT} text-[12px] tracking-[0.06em] tabular-nums text-ink-40`}
									>
										{row.votes.toLocaleString("en-US")}
									</span>
								</td>
							</tr>
						))}
					</tbody>
				</table>

			{sorted.length === 0 && (
					<p
						className={`${READOUT} py-xl text-center text-[10.5px] tracking-[0.22em] text-ink-40`}
					>
						No model or lab matches “{query.trim()}”
					</p>
				)}
			</div>

			{/* KEPT IN LAYOUT WHEN THERE IS NOTHING MORE TO SHOW. The footer's height
			    is one of the four measurements the row budget is computed from, so a
			    control that collapsed when it ran out of pages would hand the list a
			    taller budget, which would fit one more row, which could bring the
			    control back — a layout that oscillates. Hidden but still occupying,
			    the budget is the same on every page. */}
			<div
				ref={tail}
				className="flex flex-none items-center justify-center pt-sm"
			>
				<button
					type="button"
					onClick={() => setPages((n) => n + 1)}
					aria-hidden={rest === 0}
					tabIndex={rest === 0 ? -1 : undefined}
					className={`${READOUT} cursor-pointer border border-mark-16 px-[26px] py-[11px] text-[10.5px] tracking-[0.22em] text-ink-64 transition-colors duration-[160ms] hover:border-mark hover:text-ink ${
						rest === 0 ? "invisible" : ""
					}`}
				>
					Show {rest < fits ? rest : fits} more
				</button>
			</div>
		</div>
	);
}
