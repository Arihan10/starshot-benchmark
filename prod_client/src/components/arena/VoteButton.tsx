"use client";
import Button from "@/components/ui/Button";

export default function VoteButton({
	label,
	side,
	onVote,
	disabled,
}: {
	label: string;
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
