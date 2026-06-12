"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

export default function ViewerHeader() {
	const pathname = usePathname();

	return (
		<header className="z-10 flex items-center justify-between gap-4 border-b border-white/10 px-5 py-3">
			<div>
				<h1 className="text-sm font-semibold tracking-tight">Benchmark viewer</h1>
				<p className="text-xs text-neutral-500">benchmark-assets-prod · Cloudflare R2</p>
			</div>
			<div className="flex rounded-lg border border-white/10 bg-white/5 p-0.5 text-sm">
				<NavLink href="/" active={pathname === "/"}>
					3D scene
				</NavLink>
				<NavLink href="/panorama" active={pathname === "/panorama"}>
					Panoramas
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
					? "bg-white/90 text-neutral-900"
					: "text-neutral-300 hover:text-white"
			}`}
		>
			{children}
		</Link>
	);
}
