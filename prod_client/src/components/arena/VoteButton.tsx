"use client";

/**
 * One half of the vote, as a segment of the bar floating over the scenes.
 *
 * THE ARROW POINTS AT THE SCENE IT VOTES FOR. The two buttons sit together in the
 * middle of the frame now rather than under their own panel, so left and right is
 * what tells them apart and the arrow says which side of the screen this choice is
 * about. (It used to point UP, from when the button lived under its own scene and
 * horizontal pointed at nothing.)
 *
 * Cream on black, and the loudest thing on the page: at the moment of choosing,
 * this IS the next step. Hovering inverts it and lifts it off the bar rather than
 * lighting it — a glow here would read as the treatment the winning SCENE gets.
 */
export default function VoteButton({
	label,
	side,
	onVote,
	disabled,
}: {
	label: string;
	/** Which half of the screen this votes for; decides which way the arrow faces. */
	side: "a" | "b";
	onVote: () => void;
	disabled?: boolean;
}) {
	const arrow = (
		<svg
			width="13"
			height="13"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2.4"
			strokeLinecap="round"
			strokeLinejoin="round"
			aria-hidden="true"
			className={`flex-none opacity-45 transition-transform duration-200 ${
				side === "a"
					? "group-hover/vote:-translate-x-1"
					: "group-hover/vote:translate-x-1"
			}`}
		>
			<path d={side === "a" ? "m15 5-7 7 7 7" : "m9 5 7 7-7 7"} />
		</svg>
	);

	return (
		<button
			type="button"
			onClick={onVote}
			disabled={disabled}
			className="group/vote flex w-full cursor-pointer items-center justify-center gap-[clamp(7px,0.9vw,14px)] bg-foreground px-[clamp(12px,1.4vw,26px)] py-[clamp(12px,1.6vh,21px)] font-sans text-[clamp(11px,0.95vw,16px)] font-bold tracking-[0.12em] whitespace-nowrap text-background uppercase transition-[background-color,color,transform] duration-200 hover:-translate-y-0.75 hover:bg-background hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/50 focus-visible:ring-offset-2 focus-visible:ring-offset-black active:translate-y-0 disabled:cursor-default disabled:opacity-40"
		>
			{side === "a" && arrow}
			<span className="truncate">{label}</span>
			{side === "b" && arrow}
		</button>
	);
}
