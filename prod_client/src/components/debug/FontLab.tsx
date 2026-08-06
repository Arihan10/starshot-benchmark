"use client";

import {
	useDeferredValue,
	useEffect,
	useMemo,
	useState,
	useSyncExternalStore,
} from "react";
import type { FontFamily } from "@/app/api/debug/fonts/route";

const ROLES = [
	{
		key: "display",
		token: "--font-public-sans",
		label: "Display",
		generic: "sans-serif",
	},
	{
		key: "sans",
		token: "--font-sans-src",
		label: "Sans",
		generic: "sans-serif",
	},
	{
		key: "serif",
		token: "--font-instrument-serif",
		label: "Serif",
		generic: "serif",
	},
	{
		key: "mono",
		token: "--font-mono-src",
		label: "Mono",
		generic: "monospace",
	},
] as const;

type RoleKey = (typeof ROLES)[number]["key"];
type Picks = Partial<Record<RoleKey, string>>;

const STORE = "scenebench:font-lab";
const PANEL_TYPE = "ui-sans-serif, system-ui, sans-serif";
const SITE_WEIGHTS = [400, 500, 600, 700, 900];
const PAGE = 40;

const stock = (role: (typeof ROLES)[number]) =>
	`var(${role.token}), ${role.generic}`;

function familyParam(f: FontFamily, weights: number[], italics: number[]) {
	const name = f.name.replace(/\s+/g, "+");
	if (!weights.length && !italics.length) return `family=${name}`;
	if (!italics.length) return `family=${name}:wght@${weights.join(";")}`;
	const axes = [...weights.map((w) => `0,${w}`), ...italics.map((w) => `1,${w}`)];
	return `family=${name}:ital,wght@${axes.join(";")}`;
}

function servedParam(f: FontFamily) {
	const weights = f.weights.filter((w) => SITE_WEIGHTS.includes(w));
	const italics = f.italics.filter((w) => SITE_WEIGHTS.includes(w));
	return familyParam(f, weights.length ? weights : f.weights, italics);
}

function sampleParam(f: FontFamily) {
	const weights = f.weights.includes(400) ? [400] : f.weights.slice(0, 1);
	return familyParam(f, weights, weights.length ? [] : f.italics.slice(0, 1));
}

function sheetHref(params: string[]) {
	if (!params.length) return null;
	return `https://fonts.googleapis.com/css2?${params.join("&")}&display=swap`;
}

function useSheet(href: string | null) {
	useEffect(() => {
		if (!href) return;
		const link = document.createElement("link");
		link.rel = "stylesheet";
		link.href = href;
		document.head.append(link);
		return () => link.remove();
	}, [href]);
}

const readers = new Set<() => void>();

function subscribe(onChange: () => void) {
	readers.add(onChange);
	return () => void readers.delete(onChange);
}

function stored() {
	return window.localStorage.getItem(STORE) ?? "{}";
}

function store(picks: Picks) {
	window.localStorage.setItem(STORE, JSON.stringify(picks));
	for (const onChange of readers) onChange();
}

function parse(raw: string): Picks {
	try {
		return JSON.parse(raw) as Picks;
	} catch {
		return {};
	}
}

function rank(families: FontFamily[], query: string) {
	const needle = query.trim().toLowerCase();
	if (!needle) return families;

	const opens: FontFamily[] = [];
	const holds: FontFamily[] = [];
	for (const f of families) {
		const name = f.name.toLowerCase();
		if (name.startsWith(needle)) opens.push(f);
		else if (name.includes(needle)) holds.push(f);
	}
	return [...opens, ...holds];
}

export default function FontLab() {
	const [families, setFamilies] = useState<FontFamily[] | null>(null);
	const [open, setOpen] = useState(false);
	const [editing, setEditing] = useState<RoleKey | null>(null);
	const [query, setQuery] = useState("");
	const [shown, setShown] = useState(PAGE);

	const deferred = useDeferredValue(query);

	const raw = useSyncExternalStore(subscribe, stored, () => null);
	const picks = useMemo(() => (raw ? parse(raw) : {}), [raw]);

	useEffect(() => {
		let live = true;
		fetch("/api/debug/fonts")
			.then((res) => res.json())
			.then((data) => live && setFamilies(data.families ?? []))
			.catch(() => live && setFamilies([]));
		return () => {
			live = false;
		};
	}, []);

	useEffect(() => {
		const root = document.documentElement;
		for (const role of ROLES) {
			const pick = picks[role.key];
			if (pick) root.style.setProperty(role.token, `"${pick}", ${role.generic}`);
			else root.style.removeProperty(role.token);
		}
	}, [picks]);

	const byName = useMemo(
		() => new Map((families ?? []).map((f) => [f.name, f])),
		[families],
	);

	useSheet(
		useMemo(() => {
			const params = ROLES.map((role) => picks[role.key])
				.map((name) => (name ? byName.get(name) : undefined))
				.filter((f) => f !== undefined)
				.map(servedParam);
			return sheetHref(params);
		}, [picks, byName]),
	);

	const results = useMemo(
		() => (editing && families ? rank(families, deferred) : []),
		[editing, families, deferred],
	);
	const page = useMemo(() => results.slice(0, shown), [results, shown]);

	useSheet(useMemo(() => sheetHref(page.map(sampleParam)), [page]));

	if (raw === null) return null;

	const role = ROLES.find((r) => r.key === editing);

	const pick = (key: RoleKey, name: string | null) => {
		const next = { ...picks };
		if (name) next[key] = name;
		else delete next[key];
		store(next);
	};

	return (
		<div
			style={{ fontFamily: PANEL_TYPE }}
			className="fixed bottom-3 left-3 z-[999] w-[19rem] text-[12px] text-ink"
		>
			{open ? (
				<div className="flex flex-col bg-ground shadow-[0_0_0_1px_rgb(var(--ink-rgb)/0.16),0_18px_40px_-12px_rgb(0_0_0)]">
					<header className="flex items-center justify-between px-3 py-2">
						<span className="text-[10px] tracking-[0.2em] text-ink-40 uppercase">
							{role ? role.label : "Fonts"}
						</span>
						<div className="flex items-center gap-3 text-[10px] tracking-[0.16em] text-ink-40 uppercase">
							{role ? (
								<button type="button" onClick={() => setEditing(null)} className="hover:text-ink">
									Back
								</button>
							) : (
								<button type="button" onClick={() => store({})} className="hover:text-ink">
									Reset
								</button>
							)}
							<button type="button" onClick={() => setOpen(false)} className="hover:text-ink">
								Close
							</button>
						</div>
					</header>

					{role ? (
						<>
							<input
								autoFocus
								value={query}
								onChange={(e) => {
									setQuery(e.target.value);
									setShown(PAGE);
								}}
								placeholder={
									families ? `Search ${families.length} families` : "Loading families…"
								}
								className="border-y border-ink-8 bg-transparent px-3 py-2 outline-none placeholder:text-ink-40"
							/>

							<div className="max-h-[19rem] overflow-y-auto overscroll-contain">
								{picks[role.key] && (
									<button
										type="button"
										onClick={() => pick(role.key, null)}
										className="block w-full px-3 py-2 text-left text-[10px] tracking-[0.16em] text-ink-40 uppercase hover:bg-surface hover:text-ink"
									>
										Back to stock
									</button>
								)}

								{page.map((f) => (
									<button
										type="button"
										key={f.name}
										onClick={() => pick(role.key, f.name)}
										className="flex w-full items-baseline justify-between gap-3 px-3 py-1.5 text-left hover:bg-surface"
									>
										<span
											className="truncate text-[15px] leading-tight"
											style={{ fontFamily: `"${f.name}", ${role.generic}` }}
										>
											{f.name}
										</span>
										<span className="shrink-0 text-[9px] tracking-[0.14em] text-ink-40 uppercase">
											{f.category}
										</span>
									</button>
								))}

								{results.length > page.length && (
									<button
										type="button"
										onClick={() => setShown((n) => n + PAGE)}
										className="block w-full px-3 py-2 text-left text-[10px] tracking-[0.16em] text-ink-40 uppercase hover:bg-surface hover:text-ink"
									>
										{results.length - page.length} more
									</button>
								)}

								{families && !results.length && (
									<p className="px-3 py-2 text-ink-40">Nothing matches “{deferred}”.</p>
								)}
							</div>
						</>
					) : (
						ROLES.map((r) => (
							<button
								type="button"
								key={r.key}
								onClick={() => {
									setEditing(r.key);
									setQuery("");
									setShown(PAGE);
								}}
								className="flex items-baseline justify-between gap-3 border-t border-ink-8 px-3 py-2 text-left hover:bg-surface"
							>
								<span className="shrink-0 text-[10px] tracking-[0.16em] text-ink-40 uppercase">
									{r.label}
								</span>
								<span
									className="truncate text-[15px] leading-tight"
									style={{
										fontFamily: picks[r.key]
											? `"${picks[r.key]}", ${r.generic}`
											: stock(r),
									}}
								>
									{picks[r.key] ?? "Stock"}
								</span>
							</button>
						))
					)}
				</div>
			) : (
				<button
					type="button"
					onClick={() => setOpen(true)}
					aria-label="Font lab"
					className="size-8 bg-ground text-[13px] text-ink-40 shadow-[0_0_0_1px_rgb(var(--ink-rgb)/0.16)] hover:text-ink"
				>
					Aa
				</button>
			)}
		</div>
	);
}
