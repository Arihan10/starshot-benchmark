"use client";

import Link from "next/link";
import type { ArenaMetrics } from "./useArenaMetrics";

export default function ArenaHeader({
	metrics,
	onGenerate,
}: {
	metrics: ArenaMetrics;
	onGenerate: () => void;
}) {
	return (
		<header className="arena-head arena-chrome">
			<div className="arena-head__group arena-head__group--left">
				<Link href="/" className="arena-brand">
					<span aria-hidden className="arena-brand__mark" />
					<span className="arena-brand__stack">
						<span ref={metrics.register("brandName")} className="arena-brand__name">
							SCENEBENCH
						</span>
						<span ref={metrics.register("brandSub")} className="arena-brand__sub">
							BY STARSHOT LABS
						</span>
					</span>
				</Link>

				<span aria-hidden className="arena-divider" />

				<nav aria-label="Site" ref={metrics.register("navEnd")} className="arena-nav">
					<Link href="/about">ABOUT</Link>
					<Link href="/faq">FAQ</Link>
				</nav>
			</div>

			<span ref={metrics.register("eyebrow")} className="arena-eyebrow">
				WHO BUILT IT BETTER?
			</span>

			<div className="arena-head__group arena-head__group--right">
				<Link
					href="/leaderboard"
					ref={metrics.register("keysStart")}
					className="arena-key"
				>
					RANKINGS
					<span aria-hidden className="arena-key__glyph">
						↗
					</span>
				</Link>

				<span className="arena-glow">
					<button
						type="button"
						onClick={onGenerate}
						className="arena-key arena-key--solid"
					>
						GENERATE YOUR OWN
						<span aria-hidden className="arena-key__glyph">
							+
						</span>
					</button>
				</span>
			</div>
		</header>
	);
}
