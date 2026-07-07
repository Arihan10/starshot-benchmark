// Reusable ± (plus/minus) toolkit for the /tf analysis graphs.
//
// One canonical "p/m signal" is computed wherever we aggregate several samples
// (steps / heads / layers): a mean and a spread (sample standard deviation).
// Numbers render as "m ± s"; graphs render an error bar aligned to their axis.
// Three edge cases are handled uniformly:
//   - error too large     -> draw an "×" instead of an unreadable giant bar
//   - value near zero      -> never printed as a hard 0 (it isn't statistically
//                             zero); shown with extra precision + an "ns" flag
//   - near-zero on a radial -> its whisker is suppressed so the hub doesn't clutter

// Values whose magnitude is below this are treated as "essentially zero" for
// significance, but are still rendered with real precision (not "0.000").
export const NEAR_ZERO = 1e-4;

// If a symmetric error would span more than this fraction of the axis, we stop
// drawing a bar/whisker and mark the point with an "×" (too uncertain to read).
export const ERR_TOO_BIG = 0.6;

// On a spider/radial, a mean below this fraction of the axis sits basically on
// the hub; drawing its whisker just clutters the center, so we skip it.
export const RADIAL_NEAR_ZERO = 0.04;

export function mean(xs) {
	return xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0;
}

// Sample standard deviation (n-1). Returns 0 for <2 samples.
export function std(xs) {
	const n = xs.length;
	if (n < 2) return 0;
	const m = mean(xs);
	const v = xs.reduce((s, x) => s + (x - m) * (x - m), 0) / (n - 1);
	return Math.sqrt(Math.max(0, v));
}

// The canonical p/m signal for a set of samples: { m, s, n }.
export function pm(xs) {
	return { m: mean(xs), s: std(xs), n: xs.length };
}

// A value is "significant" (distinguishable from zero) when its magnitude
// exceeds its own spread.
export function significant(m, s) {
	return Math.abs(m) > (s || 0) + NEAR_ZERO;
}

// Pearson correlation of paired samples in [-1, 1]; 0 when undefined (fewer than
// two pairs, or no variance on either axis).
export function pearson(xs, ys) {
	const n = Math.min(xs.length, ys.length);
	if (n < 2) return 0;
	const mx = mean(xs.slice(0, n)), my = mean(ys.slice(0, n));
	let sxy = 0, sxx = 0, syy = 0;
	for (let i = 0; i < n; i++) { const dx = xs[i] - mx, dy = ys[i] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
	const den = Math.sqrt(sxx * syy);
	return den > 0 ? sxy / den : 0;
}

// Fractional ranks (0-based); tied values share their average rank.
function ranks(xs) {
	const idx = xs.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
	const r = new Array(xs.length);
	let i = 0;
	while (i < idx.length) {
		let j = i;
		while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
		const avg = (i + j) / 2;
		for (let k = i; k <= j; k++) r[idx[k][1]] = avg;
		i = j + 1;
	}
	return r;
}

// Spearman rank correlation (Pearson on ranks) — robust to monotone nonlinearity,
// so it reads "does order predict order/attention" without assuming a straight line.
export function spearman(xs, ys) {
	const n = Math.min(xs.length, ys.length);
	if (n < 2) return 0;
	return pearson(ranks(xs.slice(0, n)), ranks(ys.slice(0, n)));
}

// Normal CDF approximation (Abramowitz & Stegun 26.2.17).
function normalCdf(z) {
	const x = Math.abs(z);
	const t = 1 / (1 + 0.2316419 * x);
	const d = 0.3989423 * Math.exp(-x * x / 2);
	const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
	return z >= 0 ? 1 - p : p;
}

// Two-tailed p-value for Student's t (df ≥ 1). Uses the normal limit for large df.
export function tPValueTwoTail(t, df) {
	if (!isFinite(t) || df < 1) return 1;
	const z = Math.abs(t) * (df >= 30 ? 1 : Math.sqrt(1 - 1 / (4 * df)));
	return Math.max(0, Math.min(1, 2 * (1 - normalCdf(z))));
}

// OLS slope of y on x with standard error and two-tailed significance.
export function olsTrend(xs, ys) {
	const n = Math.min(xs.length, ys.length);
	if (n < 3) return { slope: 0, se: 0, r: 0, t: 0, p: 1, dir: "flat", n };
	const mx = mean(xs.slice(0, n)), my = mean(ys.slice(0, n));
	let sxx = 0, sxy = 0, syy = 0;
	for (let i = 0; i < n; i++) {
		const dx = xs[i] - mx, dy = ys[i] - my;
		sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
	}
	if (sxx <= 0) return { slope: 0, se: 0, r: 0, t: 0, p: 1, dir: "flat", n };
	const slope = sxy / sxx;
	const r = Math.sqrt(sxx * syy) > 0 ? sxy / Math.sqrt(sxx * syy) : 0;
	let sse = 0;
	for (let i = 0; i < n; i++) {
		const pred = my + slope * (xs[i] - mx);
		sse += (ys[i] - pred) ** 2;
	}
	const se = Math.sqrt(Math.max(0, sse / (n - 2) / sxx));
	const t = se > 0 ? slope / se : 0;
	const p = tPValueTwoTail(t, n - 2);
	const dir = trendDir(slope, p);
	return { slope, se, r, t, p, dir, n };
}

// Spearman ρ trend with two-tailed p (t approximation on ranks).
export function spearmanTrend(xs, ys) {
	const n = Math.min(xs.length, ys.length);
	if (n < 3) return { rho: 0, p: 1, dir: "flat", n };
	const rho = spearman(xs.slice(0, n), ys.slice(0, n));
	const den = 1 - rho * rho;
	const t = den > 1e-12 ? rho * Math.sqrt((n - 2) / den) : (rho >= 0 ? Infinity : -Infinity);
	const p = tPValueTwoTail(t, n - 2);
	const dir = trendDir(rho, p);
	return { rho, p, dir, n };
}

function trendDir(signed, p, alpha = 0.05) {
	if (p >= alpha || Math.abs(signed) < 1e-9) return "flat";
	return signed > 0 ? "increasing" : "decreasing";
}

// Format a scalar without ever collapsing a nonzero value to "0.000".
export function fmtNum(v, digits = 3) {
	if (v == null || !isFinite(v)) return "–";
	if (v === 0) return "0";
	const a = Math.abs(v);
	if (a < 5e-4) return v.toExponential(1);
	return v.toFixed(digits);
}

// "m ± s" (drops the ± when there is no spread to report).
export function fmtPM(m, s, digits = 3) {
	if (s == null || s <= 0) return fmtNum(m, digits);
	return `${fmtNum(m, digits)} ± ${fmtNum(s, digits)}`;
}

// Axis ceiling that leaves headroom for the UPPER whisker. Without this the
// top-ranked item (whose mean equals the raw max) sits exactly on the axis, so
// its +err always spills past the edge and gets flagged as an unreadable "×".
// When `showErr` is false we fall back to the plain max (no artificial slack).
// `errAt` maps an item to its spread; omit it to key off an `.sd`/`.s` field.
export function axisMax(items, valueAt, errAt = null, showErr = true, floor = 1e-9) {
	let m = floor;
	for (const it of items) {
		const v = valueAt(it) || 0;
		const e = showErr ? (errAt ? errAt(it) : (it && (it.sd ?? it.s))) || 0 : 0;
		if (v + e > m) m = v + e;
	}
	return m;
}

// Geometry for a horizontal error bar over a [0..max] track, in percent.
// tooBig => caller should render an "×" at fillPct instead of a whisker.
export function barError({ value, err, max }) {
	const denom = max || 1e-9;
	const fillPct = Math.max(0, Math.min(100, (100 * value) / denom));
	const ePct = (100 * (err || 0)) / denom;
	const loPct = Math.max(0, fillPct - ePct);
	const hiRaw = fillPct + ePct;
	const hiPct = Math.min(100, hiRaw);
	const tooBig = ePct > ERR_TOO_BIG * 100 || hiRaw > 135;
	return { fillPct, loPct, hiPct, ePct, tooBig };
}

// Draw an "×" centered at (x, y) on a canvas.
export function drawCross(ctx, x, y, size = 4, color = "#fff", lw = 1.6) {
	ctx.save();
	ctx.strokeStyle = color;
	ctx.lineWidth = lw;
	ctx.beginPath();
	ctx.moveTo(x - size, y - size); ctx.lineTo(x + size, y + size);
	ctx.moveTo(x + size, y - size); ctx.lineTo(x - size, y + size);
	ctx.stroke();
	ctx.restore();
}

// Draw an error whisker on a spider axis: a segment ALONG the axis direction
// (`ang`) from (mean-err) to (mean+err), with caps perpendicular to the axis.
// `valMean` / `valErr` are already normalized to [0..1] of the axis span.
// Returns { tooBig } — when true it drew an "×" at the mean instead.
export function drawRadialError(ctx, { cx, cy, ang, r0, r1, valMean, valErr, color = "#fff", cap = 3.5, lw = 1.4 }) {
	const span = r1 - r0;
	const rAt = (v) => r0 + span * Math.max(0, Math.min(1, v));
	const ux = Math.cos(ang), uy = Math.sin(ang);
	const rMean = rAt(valMean);
	const mx = cx + ux * rMean, my = cy + uy * rMean;
	// Suppress near-hub whiskers (they collide at the center) and mark grossly
	// over-uncertain points with an "×" rather than a whisker that leaves the web.
	if (valMean < RADIAL_NEAR_ZERO) return { tooBig: false, skipped: true };
	if (valErr > ERR_TOO_BIG || valMean + valErr > 1.15) {
		drawCross(ctx, mx, my, cap + 1.5, color, lw + 0.4);
		return { tooBig: true };
	}
	const rIn = rAt(valMean - valErr), rOut = rAt(valMean + valErr);
	const px = Math.cos(ang + Math.PI / 2), py = Math.sin(ang + Math.PI / 2);
	const ix = cx + ux * rIn, iy = cy + uy * rIn;
	const ox = cx + ux * rOut, oy = cy + uy * rOut;
	ctx.save();
	ctx.strokeStyle = color;
	ctx.lineWidth = lw;
	ctx.beginPath();
	ctx.moveTo(ix, iy); ctx.lineTo(ox, oy);
	ctx.moveTo(ix - px * cap, iy - py * cap); ctx.lineTo(ix + px * cap, iy + py * cap);
	ctx.moveTo(ox - px * cap, oy - py * cap); ctx.lineTo(ox + px * cap, oy + py * cap);
	ctx.stroke();
	ctx.restore();
	return { tooBig: false };
}
