"use client";

import { useState } from "react";
import {
	SHAPES,
	setMastheadShape,
	STOCK,
	useMastheadShape,
} from "@/lib/mastheadShape";

const PANEL_TYPE = "ui-sans-serif, system-ui, sans-serif";

export default function MastheadLab() {
	const [open, setOpen] = useState(false);
	const shape = useMastheadShape();

	return (
		<div
			style={{ fontFamily: PANEL_TYPE }}
			className="fixed bottom-3 left-[104px] z-[999] w-[17rem] text-[12px] text-ink"
		>
			{open ? (
				<div className="flex flex-col bg-ground shadow-[0_0_0_1px_rgb(var(--ink-rgb)/0.16),0_18px_40px_-12px_rgb(0_0_0)]">
					<header className="flex items-center justify-between px-3 py-2">
						<span className="text-[10px] tracking-[0.2em] text-ink-40 uppercase">
							Masthead
						</span>
						<div className="flex items-center gap-3 text-[10px] tracking-[0.16em] text-ink-40 uppercase">
							<button
								type="button"
								onClick={() => setMastheadShape(STOCK)}
								className="hover:text-ink"
							>
								Reset
							</button>
							<button
								type="button"
								onClick={() => setOpen(false)}
								className="hover:text-ink"
							>
								Close
							</button>
						</div>
					</header>

					{SHAPES.map((figure) => {
						const on = figure.key === shape;
						return (
							<button
								type="button"
								key={figure.key}
								onClick={() => setMastheadShape(figure.key)}
								aria-pressed={on}
								className={`flex items-baseline gap-3 border-t border-ink-8 px-3 py-2 text-left hover:bg-surface ${
									on ? "text-ink" : "text-ink-40 hover:text-ink"
								}`}
							>
								<span
									aria-hidden
									className={`size-1.5 shrink-0 translate-y-[-1px] ${
										on ? "bg-accent" : "bg-ink-16"
									}`}
								/>
								<span className="w-[3.6rem] shrink-0 text-[10px] tracking-[0.16em] uppercase">
									{figure.label}
								</span>
								<span className="truncate text-[11px] text-ink-40">
									{figure.note}
								</span>
							</button>
						);
					})}
				</div>
			) : (
				<button
					type="button"
					onClick={() => setOpen(true)}
					aria-label="Masthead lab"
					className="size-8 bg-ground text-[13px] text-ink-40 shadow-[0_0_0_1px_rgb(var(--ink-rgb)/0.16)] hover:text-ink"
				>
					⌒
				</button>
			)}
		</div>
	);
}
