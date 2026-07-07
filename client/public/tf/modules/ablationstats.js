// Pure statistics for the ablation analysis: one-way ANOVA (used both for the
// 4 shuffle methods and for XML yes/no, where 2 groups reduce to a t-test since
// F = t²), with the F-distribution p-value via the regularized incomplete beta.
// No DOM / deps — report.js renders the table from these numbers.

export function mean(xs) {
	return xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0;
}
export function sd(xs) {
	if (xs.length < 2) return 0;
	const m = mean(xs);
	return Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - 1));
}

// Log-gamma (Lanczos) — needed by the incomplete beta.
function gammaln(x) {
	const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
	let y = x;
	let tmp = x + 5.5;
	tmp -= (x + 0.5) * Math.log(tmp);
	let ser = 1.000000000190015;
	for (let j = 0; j < 6; j++) { y += 1; ser += c[j] / y; }
	return -tmp + Math.log(2.5066282746310005 * ser / x);
}

// Continued fraction for the incomplete beta (Numerical Recipes betacf).
function betacf(a, b, x) {
	const MAXIT = 200, EPS = 3e-12, FPMIN = 1e-300;
	const qab = a + b, qap = a + 1, qam = a - 1;
	let c = 1;
	let d = 1 - qab * x / qap;
	if (Math.abs(d) < FPMIN) d = FPMIN;
	d = 1 / d;
	let h = d;
	for (let m = 1; m <= MAXIT; m++) {
		const m2 = 2 * m;
		let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
		d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
		c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
		d = 1 / d; h *= d * c;
		aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
		d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
		c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
		d = 1 / d;
		const del = d * c; h *= del;
		if (Math.abs(del - 1) < EPS) break;
	}
	return h;
}

// Regularized incomplete beta I_x(a,b).
function betai(a, b, x) {
	if (x <= 0) return 0;
	if (x >= 1) return 1;
	const bt = Math.exp(gammaln(a + b) - gammaln(a) - gammaln(b) + a * Math.log(x) + b * Math.log(1 - x));
	return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}

// Upper-tail p-value of the F distribution: P(F_{d1,d2} >= f).
export function fPValue(f, d1, d2) {
	if (!Number.isFinite(f)) return f > 0 ? 0 : 1;
	if (f <= 0 || d1 <= 0 || d2 <= 0) return 1;
	return betai(d2 / 2, d1 / 2, d2 / (d2 + d1 * f));
}

// One-way ANOVA over `groups` (array of numeric arrays, one per level). Returns
// F, p, dfs, the grand mean, and per-group mean/sd/n. `ok` is false when there
// aren't enough groups/replicates to test.
export function oneWayAnova(groups) {
	const g = groups.filter((x) => x && x.length);
	const k = g.length;
	const N = g.reduce((s, x) => s + x.length, 0);
	const groupStats = g.map((x) => ({ n: x.length, mean: mean(x), sd: sd(x) }));
	if (k < 2 || N - k < 1) return { ok: false, F: NaN, p: NaN, dfB: Math.max(0, k - 1), dfW: Math.max(0, N - k), groupStats };
	const grand = mean(g.flat());
	let ssB = 0, ssW = 0;
	for (const x of g) {
		const m = mean(x);
		ssB += x.length * (m - grand) ** 2;
		for (const v of x) ssW += (v - m) ** 2;
	}
	const dfB = k - 1, dfW = N - k;
	const msB = ssB / dfB, msW = ssW / dfW;
	const F = msW > 0 ? msB / msW : (msB > 0 ? Infinity : 0);
	const p = fPValue(F, dfB, dfW);
	// eta² (share of variance explained by the factor) as an effect size.
	const eta2 = (ssB + ssW) > 0 ? ssB / (ssB + ssW) : 0;
	return { ok: true, F, p, dfB, dfW, grand, eta2, groupStats };
}
