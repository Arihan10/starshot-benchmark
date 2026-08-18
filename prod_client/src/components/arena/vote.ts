export type Side = "a" | "b";

export type Vote = Side | "skip";

export type Outcome = "won" | "lost" | "skipped" | null;

export function outcomeFor(vote: Vote | null, side: Side): Outcome {
	if (vote === null) return null;
	if (vote === "skip") return "skipped";
	return vote === side ? "won" : "lost";
}
