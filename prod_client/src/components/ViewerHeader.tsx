"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import ScenePicker from "@/components/ScenePicker";

// Local (no-Cloudflare) testing mode reads scenes + assets straight off the
// orchestrator; surface that in the header so it's obvious which source is live.
const LOCAL_API = process.env.NEXT_PUBLIC_LOCAL_API;

export default function ViewerHeader() {
	const pathname = usePathname();

	return (
		<header className='z-10 flex flex-wrap items-center justify-between gap-4 border-b border-white/10 px-5 py-3'>
			<div className='flex items-center gap-4'>
				<div>
					<h1 className='text-sm font-semibold tracking-tight'>
						Benchmark viewer
					</h1>
					<p className='text-xs text-neutral-500'>
						{LOCAL_API
							? "local · orchestrator artifacts"
							: "benchmark-assets-prod · Cloudflare R2"}
					</p>
				</div>
				<ScenePicker />
			</div>
			<div className='flex rounded-lg border border-white/10 bg-white/5 p-0.5 text-sm'>
				<NavLink href='/' active={pathname === "/"}>
					3D scene
				</NavLink>
				<NavLink href='/panorama' active={pathname === "/panorama"}>
					Panoramas
				</NavLink>
				<NavLink href='/orbit' active={pathname === "/orbit"}>
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
					? "bg-white/90 text-neutral-900"
					: "text-neutral-300 hover:text-white"
			}`}
		>
			{children}
		</Link>
	);
}
