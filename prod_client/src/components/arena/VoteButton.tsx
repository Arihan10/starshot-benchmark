"use client";
import Button from "@/components/ui/Button";

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
		<Button
			// The side is the SHAPE: A raked left, B raked right, so the pair leans
			// away from the SKIP between them and the three interlock into one bar.
			// Colour is no longer part of it — see the sweep below.
			variant="solid"
			sweep
			shape={side === "a" ? "start" : "end"}
			onClick={onVote}
			disabled={disabled}
			className="group/vote relative flex w-full items-center justify-center disabled:cursor-default disabled:opacity-40"
			style={{ animation: "content-swap 400ms ease both" }}
		>
			<span className="relative flex items-center gap-sm">
				{side === "a" && arrow}
				<span className="truncate">{label}</span>
				{side === "b" && arrow}
			</span>
		</Button>
	);
}
