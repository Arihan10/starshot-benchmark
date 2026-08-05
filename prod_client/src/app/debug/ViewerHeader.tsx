"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import ScenePicker from "./ScenePicker";

// Local (no-Cloudflare) testing mode reads scenes + assets straight off the
// orchestrator; surface that in the header so it's obvious which source is live.
const LOCAL_API = process.env.NEXT_PUBLIC_LOCAL_API;

export default function ViewerHeader() {
	const pathname = usePathname();

	return (
		<header className="z-10 flex flex-wrap items-center justify-between gap-4 border-b border-mark-8 px-5 py-3">
			<div className="flex items-center gap-4">
				<div>
					<h1 className="text-sm font-semibold tracking-tight">Benchmark viewer</h1>
					<p className="text-xs text-ink-40">
						{LOCAL_API ? "local · orchestrator artifacts" : "benchmark-assets-prod · Cloudflare R2"}
					</p>
				</div>
				<ScenePicker />
			</div>
			<div className="flex rounded-lg border border-mark-8 bg-mark-8 p-0.5 text-sm">
				<NavLink href="/debug" active={pathname === "/debug"}>
					3D scene
				</NavLink>
				<NavLink href="/debug/panorama" active={pathname === "/debug/panorama"}>
					Panoramas
				</NavLink>
				<NavLink href="/debug/orbit" active={pathname === "/debug/orbit"}>
					Orbit
				</NavLink>
			</div>
		</header>
	);
}

function NavLink({
	href,
	active,
	children,
}: {
	href: string;
	active: boolean;
	children: ReactNode;
}) {
	return (
		<Link
			href={href}
			className={`rounded-md px-3 py-1 transition ${
				active
					? "bg-mark text-ground"
					: "text-ink-64 hover:text-ink"
			}`}
		>
			{children}
		</Link>
	);
}
