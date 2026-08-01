"use client";

/**
 * "A WINS" / "B WINS", sat directly under the build it votes for.
 *
 * The chevron points UP at that panel rather than outward as in the Arena design —
 * there, the two buttons share a row between the panels and left/right is what
 * disambiguates them. Under its own scene the horizontal arrow points at nothing;
 * up points at the thing being voted for.
 *
 * White on black, and the loudest thing on the page after the CTA: at the moment
 * of choosing, this IS the next step.
 */
export default function VoteButton({
	label,
	onVote,
	disabled,
}: {
	label: string;
	onVote: () => void;
	disabled?: boolean;
}) {
	return (
		<button
			type="button"
			onClick={onVote}
			disabled={disabled}
			className="group/vote flex w-full cursor-pointer items-center justify-center gap-[clamp(8px,1vw,18px)] rounded-xs border border-foreground bg-foreground px-[clamp(16px,2vw,36px)] py-[clamp(11px,1.5vh,20px)] font-sans text-[clamp(11px,0.85vw,14px)] font-semibold tracking-[0.18em] whitespace-nowrap text-background transition-[background-color,border-color,box-shadow] duration-200 hover:border-white hover:bg-white hover:shadow-[0_0_0_1px_rgba(237,237,237,0.5),0_0_26px_rgba(237,237,237,0.42),0_0_60px_rgba(237,237,237,0.22)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/50 focus-visible:ring-offset-2 focus-visible:ring-offset-black disabled:cursor-default disabled:opacity-40"
		>
			<svg
				width="15"
				height="15"
				viewBox="0 0 24 24"
				fill="none"
				stroke="currentColor"
				strokeWidth="2.4"
				strokeLinecap="round"
				strokeLinejoin="round"
				aria-hidden="true"
				className="opacity-75 group-hover/vote:animate-[arena-chev-up_720ms_cubic-bezier(0.45,0,0.55,1)_infinite]"
			>
				<path d="m5 15 7-7 7 7" />
			</svg>
			<span>{label}</span>
		</button>
	);
}
