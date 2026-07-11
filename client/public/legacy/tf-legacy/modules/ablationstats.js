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

// Two-way ANOVA (factor A × factor B, with replication) — the ISOLATED test.
// `cells[i][j]` is the array of values in row-level i (e.g. method) × col-level j
// (e.g. xml). Unlike two separate one-way ANOVAs (which marginalize over the other
// factor and so bury ITS systematic variance in the error term), this fits both
// factors + their interaction at once, so each main effect is tested against the
// WITHIN-CELL error only — the other factor's variance is partitioned out, which
// is what raises power (lowers p) for the effect you actually care about.
//
// Uses the UNWEIGHTED-MEANS method (marginal means of the cell means + the
// harmonic-mean cell size) so it stays correct-ish on the UNBALANCED cells an
// ablation sweep produces. Requires every A×B cell filled and total replication
// (N > a·b) for a positive error df — that's why replicates matter.
export function twoWayAnova(cells) {
	const a = cells.length;
	const b = a ? cells[0].length : 0;
	if (a < 2 || b < 2) return { ok: false, reason: "need ≥2 levels on each factor" };
	const nij = [], mij = [];
	let N = 0, filled = 0;
	for (let i = 0; i < a; i++) {
		nij[i] = []; mij[i] = [];
		for (let j = 0; j < b; j++) {
			const v = cells[i][j] || [];
			nij[i][j] = v.length;
			mij[i][j] = v.length ? mean(v) : NaN;
			N += v.length;
			if (v.length) filled += 1;
		}
	}
	if (filled < a * b) return { ok: false, reason: "some method×xml cells are empty — compute the missing combinations" };
	const dfError = N - a * b;
	if (dfError < 1) return { ok: false, reason: "no replication (need >1 variant per method×xml cell — raise reps / widen scope)" };
	let inv = 0;
	for (let i = 0; i < a; i++) for (let j = 0; j < b; j++) inv += 1 / nij[i][j];
	const nH = (a * b) / inv; // harmonic mean of cell sizes
	const rowM = mij.map((row) => mean(row));
	const colM = Array.from({ length: b }, (_, j) => mean(mij.map((row) => row[j])));
	const grand = mean(mij.flat());
	let ssA = 0; for (let i = 0; i < a; i++) ssA += (rowM[i] - grand) ** 2; ssA *= nH * b;
	let ssB = 0; for (let j = 0; j < b; j++) ssB += (colM[j] - grand) ** 2; ssB *= nH * a;
	let ssAB = 0; for (let i = 0; i < a; i++) for (let j = 0; j < b; j++) ssAB += (mij[i][j] - rowM[i] - colM[j] + grand) ** 2; ssAB *= nH;
	let ssE = 0; for (let i = 0; i < a; i++) for (let j = 0; j < b; j++) { const m = mij[i][j]; for (const v of cells[i][j]) ssE += (v - m) ** 2; }
	const total = ssA + ssB + ssAB + ssE || 1;
	const msE = ssE / dfError;
	const mk = (ss, df) => {
		const ms = ss / df;
		const F = msE > 0 ? ms / msE : (ms > 0 ? Infinity : 0);
		return { F, p: fPValue(F, df, dfError), df, eta2: ss / total };
	};
	return { ok: true, N, dfError, msError: msE, a, b, A: mk(ssA, a - 1), B: mk(ssB, b - 1), AB: mk(ssAB, (a - 1) * (b - 1)) };
}

// --- two-sample tests: difference (Welch t) + equivalence (TOST) -------------

// Student-t two-sided p via the incomplete beta; one-sided upper tail P(T>t).
export function tTwoSided(t, df) {
	if (!Number.isFinite(t)) return t === 0 ? 1 : 0;
	if (df <= 0) return 1;
	return betai(df / 2, 0.5, df / (df + t * t));
}
function tSf(t, df) { const two = tTwoSided(Math.abs(t), df); return t >= 0 ? two / 2 : 1 - two / 2; }

// Welch two-sample t-test (unequal variances). diff = mean(a) − mean(b), its SE,
// Welch–Satterthwaite df, t, and two-sided p. ok=false when a group has < 2.
export function welchT(a, b) {
	const na = (a || []).length, nb = (b || []).length;
	if (na < 2 || nb < 2) return { ok: false };
	const ma = mean(a), mb = mean(b), va = sd(a) ** 2, vb = sd(b) ** 2;
	const se = Math.sqrt(va / na + vb / nb);
	const diff = ma - mb;
	const df = se > 0 ? (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)) : (na + nb - 2);
	const t = se > 0 ? diff / se : (diff !== 0 ? Infinity : 0);
	return { ok: true, na, nb, ma, mb, diff, se, df, t, p: tTwoSided(t, df) };
}

// TOST equivalence: are mean(a) and mean(b) the SAME within ±margin? Two one-sided
// t-tests (H0: diff ≤ −margin, and H0: diff ≥ +margin); TOST p = max of the two.
// equiv = p < alpha ⇒ the means are statistically within the margin (practically
// equal). ok=false without ≥2 per group, a positive SE, or a positive margin.
export function tost(a, b, margin, alpha = 0.05) {
	const w = welchT(a, b);
	if (!w.ok || !(w.se > 0) || !(margin > 0)) return { ok: false, ...w, margin };
	const p1 = tSf((w.diff + margin) / w.se, w.df);       // reject H0: diff ≤ −margin (upper)
	const p2 = 1 - tSf((w.diff - margin) / w.se, w.df);   // reject H0: diff ≥ +margin (lower)
	const p = Math.max(p1, p2);
	return { ok: true, p, equiv: p < alpha, diff: w.diff, se: w.se, df: w.df, margin, ma: w.ma, mb: w.mb, na: w.na, nb: w.nb };
}
