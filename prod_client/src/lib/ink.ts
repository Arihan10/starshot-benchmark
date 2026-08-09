// Fallbacks match `:root` in globals.css — used only when the document has no
// computed style yet (SSR / first paint before tokens land).
const INK_FALLBACK = "237 237 237";
const MARK_FALLBACK = "237 237 237";
const GROUND_FALLBACK = "0 0 0";

function readTriplet(name: string, fallback: string): string {
	if (typeof window === "undefined") return fallback;
	const channels = getComputedStyle(document.documentElement)
		.getPropertyValue(name)
		.trim();
	return channels || fallback;
}

function rgb(triplet: string, alpha?: number): string {
	return alpha == null ? `rgb(${triplet})` : `rgb(${triplet} / ${alpha})`;
}

/** Resolved ink for JS / WebGL (Three.js, canvas). */
export function inkColor(alpha?: number): string {
	return rgb(readTriplet("--ink-rgb", INK_FALLBACK), alpha);
}

/** Resolved mark for JS / WebGL. */
export function markColor(alpha?: number): string {
	return rgb(readTriplet("--mark-rgb", MARK_FALLBACK), alpha);
}

/** Resolved ground for JS / WebGL. */
export function groundColor(alpha?: number): string {
	return rgb(readTriplet("--ground-rgb", GROUND_FALLBACK), alpha);
}

/** Live CSS expressions for inline styles — follow token re-points in a subtree. */
export const inkVar = (alpha?: number) =>
	alpha == null ? "rgb(var(--ink-rgb))" : `rgb(var(--ink-rgb) / ${alpha})`;
export const markVar = (alpha?: number) =>
	alpha == null ? "rgb(var(--mark-rgb))" : `rgb(var(--mark-rgb) / ${alpha})`;
export const groundVar = (alpha?: number) =>
	alpha == null
		? "rgb(var(--ground-rgb))"
		: `rgb(var(--ground-rgb) / ${alpha})`;
export const surfaceLitVar = (alpha?: number) =>
	alpha == null
		? "rgb(var(--surface-lit-rgb))"
		: `rgb(var(--surface-lit-rgb) / ${alpha})`;
export const accentVar = (alpha?: number) =>
	alpha == null
		? "rgb(var(--accent-rgb))"
		: `rgb(var(--accent-rgb) / ${alpha})`;
export const accentDeepVar = (alpha?: number) =>
	alpha == null
		? "rgb(var(--accent-deep-rgb))"
		: `rgb(var(--accent-deep-rgb) / ${alpha})`;
export const signalVar = (alpha?: number) =>
	alpha == null
		? "rgb(var(--signal-rgb))"
		: `rgb(var(--signal-rgb) / ${alpha})`;
