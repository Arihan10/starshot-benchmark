const INK_FALLBACK = "rgb(237 237 237)";
const MARK_FALLBACK = "rgb(237 237 237)";

function readChannels(name: string, fallback: string): string {
	if (typeof window === "undefined") return fallback;
	const channels = getComputedStyle(document.documentElement)
		.getPropertyValue(name)
		.trim();
	return channels ? `rgb(${channels})` : fallback;
}

export function inkColor(): string {
	return readChannels("--ink-rgb", INK_FALLBACK);
}

export function markColor(): string {
	return readChannels("--mark-rgb", MARK_FALLBACK);
}
