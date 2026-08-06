"use client";

import { useEffect, useMemo, useState, useSyncExternalStore } from "react";

const GROUPS: { name: string; tokens: { token: string; label: string }[] }[] = [
	{
		name: "Page",
		tokens: [
			{ token: "--ground-rgb", label: "Ground" },
			{ token: "--ink-rgb", label: "Ink" },
			{ token: "--mark-rgb", label: "Mark" },
			{ token: "--surface-rgb", label: "Surface" },
			{ token: "--surface-lit-rgb", label: "Surface lit" },
		],
	},
	{
		name: "Navbar",
		tokens: [
			{ token: "--paper-rgb", label: "Paper" },
			{ token: "--paper-ink-rgb", label: "Paper ink" },
			{ token: "--paper-surface-rgb", label: "Paper hover" },
		],
	},
	{
		name: "Moon",
		tokens: [
			{ token: "--moon-cool-rgb", label: "Cool" },
			{ token: "--moon-mid-rgb", label: "Mid" },
			{ token: "--moon-warm-rgb", label: "Warm" },
		],
	},
	{
		name: "Accent",
		tokens: [
			{ token: "--accent-rgb", label: "Accent" },
			{ token: "--accent-deep-rgb", label: "Accent deep" },
			{ token: "--rise-rgb", label: "Rise" },
			{ token: "--fall-rgb", label: "Fall" },
		],
	},
];

const STORE = "scenebench:color-lab";
const PANEL_TYPE = "ui-sans-serif, system-ui, sans-serif";

type Picks = Record<string, string>;

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

const hex = (triplet: string) => {
	const parts = triplet.trim().split(/[\s,]+/).map(Number);
	if (parts.length < 3 || parts.some((n) => !Number.isFinite(n))) return "#000000";
	return `#${parts
		.slice(0, 3)
		.map((n) => Math.max(0, Math.min(255, Math.round(n))).toString(16).padStart(2, "0"))
		.join("")}`;
};

const triplet = (value: string) => {
	const n = Number.parseInt(value.slice(1), 16);
	return `${(n >> 16) & 255} ${(n >> 8) & 255} ${n & 255}`;
};

let stockCache: Picks | null = null;

function readStock(): Picks {
	if (stockCache) return stockCache;
	const style = getComputedStyle(document.documentElement);
	const base: Picks = {};
	for (const group of GROUPS)
		for (const { token } of group.tokens)
			base[token] = style.getPropertyValue(token).trim();
	stockCache = base;
	return base;
}

export default function ColorLab() {
	const [open, setOpen] = useState(false);

	const raw = useSyncExternalStore(subscribe, stored, () => null);
	const picks = useMemo(() => (raw ? parse(raw) : {}), [raw]);

	// Read during render rather than from an effect, and cached for the session.
	// The effect below writes overrides onto the same element, so a later read
	// would report an override as the value it replaced — and Reset would then
	// restore the experiment instead of the design.
	const stock = useMemo(() => (raw === null ? null : readStock()), [raw]);

	useEffect(() => {
		const root = document.documentElement;
		for (const group of GROUPS)
			for (const { token } of group.tokens) {
				const pick = picks[token];
				if (pick) root.style.setProperty(token, pick);
				else root.style.removeProperty(token);
			}
	}, [picks]);

	if (raw === null || !stock) return null;

	const set = (token: string, value: string) =>
		store({ ...picks, [token]: triplet(value) });

	return (
		<div
			style={{ fontFamily: PANEL_TYPE }}
			className="fixed bottom-3 left-14 z-[999] w-[16rem] text-[12px] text-ink"
		>
			{open ? (
				<div className="flex flex-col bg-ground shadow-[0_0_0_1px_rgb(var(--ink-rgb)/0.16),0_18px_40px_-12px_rgb(0_0_0)]">
					<header className="flex items-center justify-between px-3 py-2">
						<span className="text-[10px] tracking-[0.2em] text-ink-40 uppercase">
							Colour
						</span>
						<div className="flex items-center gap-3 text-[10px] tracking-[0.16em] text-ink-40 uppercase">
							<button type="button" onClick={() => store({})} className="hover:text-ink">
								Reset
							</button>
							<button type="button" onClick={() => setOpen(false)} className="hover:text-ink">
								Close
							</button>
						</div>
					</header>

					<div className="max-h-[24rem] overflow-y-auto overscroll-contain">
						{GROUPS.map((group) => (
							<section key={group.name} className="border-t border-ink-8">
								<h2 className="px-3 pt-2 text-[9px] tracking-[0.2em] text-ink-40 uppercase">
									{group.name}
								</h2>
								{group.tokens.map(({ token, label }) => {
									const value = hex(picks[token] ?? stock[token] ?? "");
									return (
										<label
											key={token}
											className="flex cursor-pointer items-center justify-between gap-3 px-3 py-1.5 hover:bg-surface"
										>
											<span className="truncate">{label}</span>
											<span className="flex items-center gap-2">
												<span className="font-mono text-[10px] text-ink-40 uppercase">
													{value}
												</span>
												<input
													type="color"
													value={value}
													onChange={(e) => set(token, e.target.value)}
													className="size-5 cursor-pointer bg-transparent"
												/>
											</span>
										</label>
									);
								})}
							</section>
						))}
					</div>
				</div>
			) : (
				<button
					type="button"
					onClick={() => setOpen(true)}
					aria-label="Colour lab"
					className="size-8 bg-ground text-[13px] text-ink-40 shadow-[0_0_0_1px_rgb(var(--ink-rgb)/0.16)] hover:text-ink"
				>
					◑
				</button>
			)}
		</div>
	);
}
