// Pure, dependency-free helpers shared across the /tf modules: small numeric
// utilities, a bounded-concurrency map, the shared heat ramp, and HTML/CSS
// escaping. No imports — this is a leaf module.

export const _mean = (arr) => (arr.length ? arr.reduce((s, x) => s + x, 0) / arr.length : 0);

// Bounded-concurrency map over items (keeps the browser's connection pool free).
export async function pool(items, limit, fn) {
	const q = items.slice();
	await Promise.all(Array.from({ length: Math.min(limit, q.length) }, async () => { while (q.length) await fn(q.shift()); }));
}

// Round a value up to a clean chart ceiling (1/2/5 × 10ⁿ) so a bar axis reads
// in nice increments instead of an arbitrary data max.
export function niceMax(v) {
	if (!(v > 0)) return 1;
	const p = Math.pow(10, Math.floor(Math.log10(v)));
	const n = v / p;
	return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
}

// low scene mass → dark blue, high → bright yellow-green.
export function heatColor(s) {
	s = Math.max(0, Math.min(1, s));
	return `hsl(${Math.round(220 - 160 * s)}, 85%, ${Math.round(12 + 46 * s)}%)`;
}

export function escapeHtml(s) {
	return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
}
export function cssEsc(s) {
	return window.CSS?.escape ? CSS.escape(s) : String(s).replace(/"/g, '\\"');
}
