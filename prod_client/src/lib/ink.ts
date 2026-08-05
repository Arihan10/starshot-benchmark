/**
 * The site's ink, for the things that cannot write CSS.
 *
 * WebGL materials take a colour, not a custom property, so the leaderboard's city
 * and anything else drawn in three.js used to carry `#ededed` as a literal — a
 * copy of the one value globals.css exists to hold. Copies do not follow: flipping
 * `--ink-rgb` recoloured every pixel of the site except the nine hundred towers on
 * the leaderboard, which stayed white.
 *
 * Read from the document rather than duplicated as a TS constant, so there is
 * still exactly ONE definition and it is the one the CSS uses.
 */
const INK_FALLBACK = "rgb(237 237 237)";
const MARK_FALLBACK = "rgb(237 237 237)";

function readChannels(name: string, fallback: string): string {
	// Server render has no computed style to read; the fallback is only ever used
	// for a frame that never reaches a screen, since every caller is in a canvas
	// that mounts on the client.
	if (typeof window === "undefined") return fallback;
	const channels = getComputedStyle(document.documentElement)
		.getPropertyValue(name)
		.trim();
	return channels ? `rgb(${channels})` : fallback;
}

/** Type colour, for anything drawn as text in a canvas. */
export function inkColor(): string {
	return readChannels("--ink-rgb", INK_FALLBACK);
}

/** Fill colour, for bodies the site draws — the podium's city, and the moon. */
export function markColor(): string {
	return readChannels("--mark-rgb", MARK_FALLBACK);
}

