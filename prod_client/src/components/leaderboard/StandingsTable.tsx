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

export default function StandingsTable({ rows }: { rows: Standing[] }) {
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
	const head = useRef<HTMLTableSectionElement>(null);
	const body = useRef<HTMLTableSectionElement>(null);
	const tail = useRef<HTMLDivElement>(null);
	const [fits, setFits] = useState(7);
	const [pages, setPages] = useState(1);

	useEffect(() => {
		const measure = () => {
			const box = frame.current;
			const search = find.current;
			const thead = head.current;
			const foot = tail.current;
			const row = body.current?.querySelector("tr");
			if (!box || !search || !thead || !foot || !row) return;
			const rowHeight = row.getBoundingClientRect().height;
			if (!rowHeight) return;

			const budget =
				box.clientHeight -
				search.getBoundingClientRect().height -
				thead.getBoundingClientRect().height -
				foot.getBoundingClientRect().height;

			setFits(Math.max(1, Math.floor((budget - 1) / rowHeight)));
		};
		const observer = new ResizeObserver(measure);
		if (frame.current) observer.observe(frame.current);
		observer.observe(document.documentElement);
		document.fonts?.ready.then(measure).catch(() => {});
		return () => observer.disconnect();
	}, []);

	const limit = Math.min(sorted.length, pages * fits);
	const visible = sorted.slice(0, limit);
	const rest = sorted.length - limit;

	return (
		<div ref={frame} className="flex min-h-0 flex-1 flex-col">
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
						className={`${READOUT} flex-none cursor-pointer text-[9.5px] font-bold tracking-[0.2em] text-ink-40 transition-colors duration-[160ms] hover:text-ink`}
					>
						Clear
					</button>
				)}
			</div>
			</div>

			<div className="min-h-0 overflow-y-auto">
				<table className="w-full table-fixed border-collapse text-left">
					<caption className="sr-only">
						Model standings, sorted by{" "}
						{COLUMNS.find((c) => c.key === sort.key)?.label},{" "}
						{sort.dir === "desc" ? "highest first" : "lowest first"}. Showing{" "}
						{limit} of {sorted.length}.
					</caption>
					<colgroup>
						<col className="w-[72px]" />
						<col />
						<col className="w-[150px]" />
						<col className="w-[210px]" />
						<col className="w-[130px]" />
					</colgroup>
					<thead ref={head}>
						<tr>
							<th
								scope="col"
								className="sticky top-0 z-10 bg-ground p-0 pl-[22px] shadow-[inset_0_-1px_0_var(--color-mark-16)]"
							>
								<span
									className={`${READOUT} flex h-[38px] items-center px-[9px] text-[9.5px] font-bold tracking-[0.22em] text-ink-40`}
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
											className={`${READOUT} flex h-[38px] w-full cursor-pointer items-center gap-xs px-[9px] text-[9.5px] font-bold tracking-[0.22em] transition-colors duration-[160ms] hover:text-ink ${
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
						{visible.map((row) => (
							<tr
								key={row.name}
								className="group/row h-[clamp(56px,5.8vh,68px)] border-b border-mark-8 transition-colors duration-[160ms] hover:bg-mark-8"
							>
								<td className="pl-[22px] align-middle transition-shadow duration-[160ms] group-hover/row:shadow-[inset_3px_0_0_var(--color-mark)]">
									<span
										className={`${READOUT} block px-[9px] text-[15px] font-bold tracking-[0.04em] text-ink-40 tabular-nums`}
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

								<td className="px-[9px] text-center align-middle">
									<span className="flex items-baseline justify-center gap-[9px]">
										<span className="text-[21px] leading-none font-extrabold tracking-[-0.02em] tabular-nums text-ink">
											{row.elo}
										</span>
										<span
											className={`${READOUT} w-[42px] flex-none text-left text-[10px] leading-none font-bold tracking-[0.1em] tabular-nums ${
												row.delta >= 0 ? "text-rise" : "text-fall"
											}`}
										>
											{row.delta >= 0 ? "▲" : "▼"} {Math.abs(row.delta)}
										</span>
									</span>
								</td>

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

				{sorted.length === 0 && (
					<p
						className={`${READOUT} py-xl text-center text-[10.5px] font-bold tracking-[0.22em] text-ink-40`}
					>
						No model or lab matches “{query.trim()}”
					</p>
				)}
			</div>

			{/* Kept in layout when there is nothing more to show — a collapsing
			    tail would change the row budget that decides it is needed. */}
			<div
				ref={tail}
				className="flex flex-none items-center justify-center pt-sm"
			>
				<button
					type="button"
					onClick={() => setPages((p) => p + 1)}
					aria-hidden={rest === 0}
					tabIndex={rest === 0 ? -1 : undefined}
					className={`${READOUT} cursor-pointer border border-mark-16 px-[26px] py-[11px] text-[10.5px] font-bold tracking-[0.22em] text-ink-64 transition-colors duration-[160ms] hover:border-mark hover:text-ink ${
						rest === 0 ? "invisible" : ""
					}`}
				>
					Show more
				</button>
			</div>
		</div>
	);
}
