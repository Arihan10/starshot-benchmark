import type { ReactNode } from "react";
import Fade from "./Fade";
import MoonAnchor from "./MoonAnchor";
import Navbar from "./Navbar";

export const MOON_DIAMETER = "min(48vw, 690px)";

const MOON_BERTH = `calc(${MOON_DIAMETER} * 0.68)`;
const MOON_DROP = "var(--spacing-md)";

export default function Masthead({
	label,
	placement = "overlay",
	children,
}: {
	label: string;
	placement?: "overlay" | "flow";
	children: ReactNode;
}) {
	return (
		<div
			className={`pointer-events-none [&_a]:pointer-events-auto [&_button]:pointer-events-auto ${
				placement === "overlay"
					? "absolute inset-x-0 top-0"
					: "relative flex-none"
			}`}
			style={
				{ "--moon-drop": MOON_DROP, paddingBottom: MOON_DROP } as React.CSSProperties
			}
		>
			{/* THE PAPER, and it has to be a sibling of the bar rather than a layer
			    inside it. The header is `z-20`, which makes it a stacking context, so
			    anything parented to it paints ABOVE the moon — the moon is `fixed` at
			    `z-10` on the root and never enters that context. Out here the band
			    lands under the disc and over the page, which is the order wanted.

			    It stops at the BAR's bottom, not the masthead's. The disc hangs one
			    drop lower and reads against the black from there down, which is the
			    whole silhouette — paper carried to the moon's lowest point instead
			    just squares the band off under it. */}
			<div
				aria-hidden
				className="absolute inset-x-0 top-0 bottom-(--moon-drop) bg-paper"
			/>

			<Navbar
				moon={
					// OPTS BACK OUT OF THE BAR'S INVERSION. The slot is passed into the
					// header and inherits its flip, but the label and the arced prompt sit
					// on the DISC, not on the paper — and the disc did not invert.
					<div
						className="relative flex justify-center pt-xs"
						style={
							{ "--ground-rgb": "0 0 0", width: MOON_BERTH } as React.CSSProperties
						}
					>
						<MoonAnchor
							className="absolute -bottom-(--moon-drop,0) left-1/2 -translate-x-1/2"
							style={{ width: MOON_DIAMETER, height: MOON_DIAMETER }}
						/>

						<Fade enter={700} delay={180} leave={220}>
							<span className="font-mono text-2xs tracking-[0.24em] whitespace-nowrap uppercase text-ground">
								{label}
							</span>

							{children}
						</Fade>
					</div>
				}
			/>

		</div>
	);
}

export function MoonBerth() {
	return <div aria-hidden style={{ width: MOON_BERTH }} />;
}

export function MoonArc({
	className = "",
	children,
}: {
	className?: string;
	children: ReactNode;
}) {
	return (
		<div
			className={`pointer-events-none absolute -bottom-(--moon-drop,0) left-1/2 -translate-x-1/2 text-ground ${className}`}
		>
			{children}
		</div>
	);
}
