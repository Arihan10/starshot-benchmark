"use client";

import { useEffect, useMemo, useState } from "react";

type FontItem = { family: string; category: string };

type Role = "sans" | "display" | "serif" | "mono";

const ROLES: { id: Role; label: string; cssVar: string; fallback: string }[] = [
	{ id: "sans", label: "Sans", cssVar: "--font-sans-src", fallback: "sans-serif" },
	{ id: "display", label: "Display", cssVar: "--font-display", fallback: "sans-serif" },
	{ id: "serif", label: "Serif", cssVar: "--font-serif", fallback: "serif" },
	{ id: "mono", label: "Mono", cssVar: "--font-mono-src", fallback: "monospace" },
];

const STORAGE_KEY = "scenebench-debug-fonts";

type Picks = Partial<Record<Role, string>>;

function loadPicks(): Picks {
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		return raw ? (JSON.parse(raw) as Picks) : {};
	} catch {
		return {};
	}
}

function savePicks(picks: Picks) {
	try {
		localStorage.setItem(STORAGE_KEY, JSON.stringify(picks));
	} catch {
		/* ignore */
	}
}

function cssFamily(family: string, fallback: string) {
	return `"${family}", ${fallback}`;
}

function ensureStylesheet(family: string) {
	const id = `gf-debug-${family.replace(/[^a-z0-9]+/gi, "-")}`;
	if (document.getElementById(id)) return;
	const link = document.createElement("link");
	link.id = id;
	link.rel = "stylesheet";
	const familyParam = encodeURIComponent(family).replace(/%20/g, "+");
	link.href = `https://fonts.googleapis.com/css2?family=${familyParam}:wght@100;200;300;400;500;600;700;800;900&display=swap`;
	document.head.appendChild(link);
}

function applyRole(role: Role, family: string) {
	const meta = ROLES.find((r) => r.id === role);
	if (!meta) return;
	ensureStylesheet(family);
	document.documentElement.style.setProperty(
		meta.cssVar,
		cssFamily(family, meta.fallback),
	);
}

function bucket(category: string): "sans" | "serif" | "other" {
	if (category === "Sans Serif") return "sans";
	if (category === "Serif") return "serif";
	return "other";
}

export default function FontDebugPicker() {
	const [open, setOpen] = useState(false);
	const [fonts, setFonts] = useState<FontItem[] | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [role, setRole] = useState<Role>("sans");
	const [group, setGroup] = useState<"sans" | "serif" | "other">("sans");
	const [query, setQuery] = useState("");
	const [picks, setPicks] = useState<Picks>({});

	useEffect(() => {
		const saved = loadPicks();
		setPicks(saved);
		for (const r of ROLES) {
			const family = saved[r.id];
			if (family) applyRole(r.id, family);
		}
	}, []);

	useEffect(() => {
		if (!open || fonts) return;
		let cancelled = false;
		fetch("/api/debug/fonts")
			.then(async (res) => {
				if (!res.ok) throw new Error(`catalog ${res.status}`);
				return res.json() as Promise<{ fonts: FontItem[] }>;
			})
			.then((data) => {
				if (!cancelled) setFonts(data.fonts);
			})
			.catch((err: unknown) => {
				if (!cancelled) {
					setError(err instanceof Error ? err.message : "failed to load");
				}
			});
		return () => {
			cancelled = true;
		};
	}, [open, fonts]);

	const filtered = useMemo(() => {
		if (!fonts) return [];
		const q = query.trim().toLowerCase();
		return fonts
			.filter((f) => bucket(f.category) === group)
			.filter((f) => !q || f.family.toLowerCase().includes(q))
			.sort((a, b) => a.family.localeCompare(b.family));
	}, [fonts, group, query]);

	const pick = (family: string) => {
		applyRole(role, family);
		setPicks((prev) => {
			const next = { ...prev, [role]: family };
			savePicks(next);
			return next;
		});
	};

	const reset = () => {
		for (const r of ROLES) {
			document.documentElement.style.removeProperty(r.cssVar);
		}
		setPicks({});
		savePicks({});
	};

	return (
		<div className="pointer-events-none fixed right-3 bottom-3 z-100 flex flex-col items-end gap-2">
			{open && (
				<div className="pointer-events-auto flex max-h-[min(70vh,560px)] w-[min(92vw,340px)] flex-col overflow-hidden rounded-md border border-mark-16 bg-ground/95 text-ink shadow-2xl backdrop-blur-md">
					<div className="flex items-center justify-between gap-2 border-b border-mark-8 px-3 py-2">
						<span className="font-mono text-2xs tracking-[0.18em] uppercase text-ink-40">
							Font debug
						</span>
						<button
							type="button"
							onClick={reset}
							className="text-2xs tracking-wide text-ink-40 uppercase hover:text-ink"
						>
							Reset
						</button>
					</div>

					<div className="flex gap-1 border-b border-mark-8 p-2">
						{ROLES.map((r) => (
							<button
								key={r.id}
								type="button"
								onClick={() => setRole(r.id)}
								className={`flex-1 rounded-sm px-1.5 py-1 text-2xs font-bold tracking-wide uppercase ${
									role === r.id
										? "bg-mark text-ground"
										: "text-ink-64 hover:text-ink"
								}`}
								title={picks[r.id] ?? "site default"}
							>
								{r.label}
							</button>
						))}
					</div>

					<div className="border-b border-mark-8 px-3 py-2 text-2xs text-ink-40">
						Hover applies immediately ·{" "}
						<span className="text-ink">
							{picks[role] ?? "default"}
						</span>
					</div>

					<div className="flex gap-1 border-b border-mark-8 p-2">
						{(
							[
								["sans", "Sans"],
								["serif", "Serif"],
								["other", "Other"],
							] as const
						).map(([id, label]) => (
							<button
								key={id}
								type="button"
								onClick={() => setGroup(id)}
								className={`flex-1 rounded-sm px-1.5 py-1 text-2xs font-bold tracking-wide uppercase ${
									group === id
										? "bg-mark-16 text-ink"
										: "text-ink-40 hover:text-ink"
								}`}
							>
								{label}
							</button>
						))}
					</div>

					<div className="border-b border-mark-8 p-2">
						<input
							value={query}
							onChange={(e) => setQuery(e.target.value)}
							placeholder="Filter…"
							className="w-full rounded-sm border border-mark-8 bg-transparent px-2 py-1.5 text-xs text-ink outline-none placeholder:text-ink-40 focus:border-mark-40"
						/>
					</div>

					<div className="min-h-0 flex-1 overflow-y-auto">
						{error && (
							<p className="px-3 py-4 text-xs text-fall">{error}</p>
						)}
						{!error && !fonts && (
							<p className="px-3 py-4 text-xs text-ink-40">
								Loading Google Fonts…
							</p>
						)}
						{fonts && filtered.length === 0 && (
							<p className="px-3 py-4 text-xs text-ink-40">No matches.</p>
						)}
						<ul className="py-1">
							{filtered.map((f) => {
								const active = picks[role] === f.family;
								return (
									<li key={f.family}>
										<button
											type="button"
											onMouseEnter={() => pick(f.family)}
											onFocus={() => pick(f.family)}
											className={`block w-full truncate px-3 py-1.5 text-left text-xs transition-colors ${
												active
													? "bg-mark text-ground"
													: "text-ink-64 hover:bg-mark-8 hover:text-ink"
											}`}
											style={{ fontFamily: `"${f.family}", sans-serif` }}
										>
											{f.family}
										</button>
									</li>
								);
							})}
						</ul>
					</div>
				</div>
			)}

			<button
				type="button"
				onClick={() => setOpen((v) => !v)}
				className="pointer-events-auto rounded-md border border-mark-16 bg-ground/90 px-3 py-1.5 font-mono text-2xs tracking-[0.16em] uppercase text-ink shadow-lg backdrop-blur hover:border-mark-40"
			>
				{open ? "Close fonts" : "Fonts"}
			</button>
		</div>
	);
}
