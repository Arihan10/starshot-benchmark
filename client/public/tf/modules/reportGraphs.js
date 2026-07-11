// Graph rendering protocol for the /tf analysis workspace.
// Each graph chooses its normal or peer-comparison rendering from the same entrypoint.

import { pm, mean, std, fmtNum, fmtPM, drawRadialError, axisMax, spearman, spearmanTrend, olsTrend } from "./uncertainty.js";

let deps = {};

export const COMPARE_COLORS = ["#7aa2f7", "#6bd96e", "#e0a94a", "#b46aff", "#4af0e0", "#ff6b9d", "#ff9e64", "#9ece6a"];
const ATTR_AXIS_ORDER = [
	"name", "noun_phrase", "prompt", "description",
	"placement", "relationships", "dimensions", "orientation", "yaw",
	"proxy_shape", "parent", "parent_region", "global_origin", "local_origin",
];

export function initGraphProtocol(d) { deps = d; }
const d = () => deps;

export function defaultGraphModules() {
	return [
		{ id: "g-pipeline", type: "pipelineMetric", title: "pipeline attention", metric: "sceneMass", scope: "scene", chart: "timeline", wide: true },
		{ id: "g-attrs", type: "attributeProfile", title: "attribute profile", metric: "scene", scope: "step", chart: "bars", wide: true },
	];
}

export function renderGraph(type, opts = {}) {
	const GRAPH_PROTOCOLS = {
		attributeProfile: renderAttributeProfileGraph,
		entityRanking: renderEntityRankingGraph,
		headLayerMap: renderHeadLayerGraph,
		pipelineMetric: renderPipelineMetricGraph,
		placementSplit: renderPlacementSplitGraph,
	};
	return (GRAPH_PROTOCOLS[type] || (() => d().reportEmpty(`unknown graph: ${type}`)))(opts);
}

export function renderModuleGraph(mod, ctx = {}) {
	return renderGraph(mod.type, { ...ctx, title: mod.title, chart: mod.chart, scope: mod.scope });
}

function renderAttributeProfileGraph({ items = null, title = "attribute profile", sub = "", chart = "bars", cmp = null, rows = [], a = null } = {}) {
	const { aggregateAttn, overviewAggregate } = d();
	if (cmp?.active) return comparePeerAttributes(cmp, title || "selected attribute profiles");
	const list = items || (a ? aggregateAttn(a).componentTotals : overviewAggregate(rows).componentTotals);
	return attributeProfileGraph(list, { title, sub, chart });
}

function renderEntityRankingGraph({ title = "entity attention", cmp = null, rows = [] } = {}) {
	const { overviewEntities, reportEmpty } = d();
	if (cmp?.active) return comparePeerEntities(cmp, title || "selected entity attention");
	return rows.length ? overviewEntities(rows) : reportEmpty("no computed rows yet");
}

function renderHeadLayerGraph({ title = "head/layer attention", cmp = null, rows = [], a = null } = {}) {
	const { overviewHeadGrid, headGrid, reportEmpty } = d();
	if (cmp?.active) return comparePeerHeads(cmp, title || "selected head/layer attention");
	return rows.length ? overviewHeadGrid(rows) : (a ? headGrid(a) : reportEmpty("no head data yet"));
}

function renderPipelineMetricGraph({ title = "pipeline attention", cmp = null, rows = [], order = [] } = {}) {
	const { overviewMassPerStep, reportEmpty } = d();
	if (cmp?.active) return comparePeerPipeline(cmp, title || "selected pipeline attention");
	return rows.length ? overviewMassPerStep(rows, order) : reportEmpty("no computed rows yet");
}

function renderPlacementSplitGraph({ title = "placement split", cmp = null, rows = [], a = null } = {}) {
	const { hasToPlace, summaryToPlace, overviewToPlace, reportEmpty } = d();
	if (cmp?.active) return comparePeerPlacement(cmp, title || "selected placement split");
	return a && hasToPlace(a) ? summaryToPlace(a) : overviewToPlace(rows) || reportEmpty("no placement data yet");
}

function comparePeerAttributes(ctx, title) {
	const { overviewAggregate, reportCard, showErr } = d();
	const errOn = !showErr || showErr();
	const comps = pickAxes(ctx.peers.flatMap((p) => overviewAggregate(p.rows).componentTotals.map((c) => [c.component, c.score])));
	const profiles = ctx.peers.map((p) => {
		const totals = overviewAggregate(p.rows).componentTotals;
		return {
			label: p.label, color: p.color,
			map: new Map(totals.map((c) => [c.component, c.score])),
			err: new Map(totals.map((c) => [c.component, c.sd || 0])),
		};
	});
	const max = axisMax(profiles.flatMap((p) => comps.map((c) => ({ v: p.map.get(c) || 0, e: p.err.get(c) || 0 }))), (x) => x.v, (x) => x.e, errOn);
	return reportCard(title, `${ctx.peers.length} selected ${ctx.level}s · spider normalized per item (shape) · matrix = raw mean ± sd`,
		spiderChart(profiles),
		peerMatrix(profiles, comps, max, ctx.level));
}

function comparePeerEntities(ctx, title = "selected entity attention") {
	const { overviewAggregate, reportCard } = d();
	const ents = [...new Set(ctx.peers.flatMap((p) => overviewAggregate(p.rows).entityTotals.slice(0, 8).map((e) => e.id)))].slice(0, 10);
	const profiles = ctx.peers.map((p) => {
		const totals = overviewAggregate(p.rows).entityTotals;
		return {
			label: p.label, color: p.color,
			map: new Map(totals.map((e) => [e.id, e.score])),
			err: new Map(totals.map((e) => [e.id, e.sd || 0])),
		};
	});
	const max = Math.max(...profiles.flatMap((p) => ents.map((e) => p.map.get(e) || 0)), 1e-9);
	return reportCard(title, `${ctx.peers.length} selected ${ctx.level}s · mean ± sd across steps`, peerMatrix(profiles, ents, max, ctx.level));
}

function comparePeerPipeline(ctx, title = "selected pipeline attention") {
	const { el, reportCard, summaryBar, showErr } = d();
	const order = ctx.order || [];
	const overall = ctx.peers.map((p) => { const s = pm(p.rows.map((r) => r.mass || 0)); return { label: p.label, color: p.color, mean: s.m, sd: s.s }; });
	const maxO = axisMax(overall, (o) => o.mean, (o) => o.sd, !showErr || showErr());
	const bars = el("div", { class: "sbars" }, ...overall.map((o) => summaryBar({
		color: o.color, label: o.label, value: o.mean, sd: o.sd, max: maxO,
		title: `${o.label} · mean scene mass ${fmtPM(o.mean, o.sd)}`,
	})));
	const rank = (k) => { const i = order.indexOf(k); return i < 0 ? 1e6 : i; };
	const kinds = [...new Set(ctx.peers.flatMap((p) => p.rows.map((r) => r.template)))].sort((a, b) => rank(a) - rank(b));
	const profiles = ctx.peers.map((p) => {
		const by = new Map();
		for (const r of p.rows) { const c = by.get(r.template) || { s: 0, n: 0 }; c.s += r.mass || 0; c.n++; by.set(r.template, c); }
		return { label: p.label, color: p.color, map: new Map([...by.entries()].map(([k, { s, n }]) => [k, n ? s / n : 0])) };
	});
	const max = Math.max(...profiles.flatMap((p) => kinds.map((k) => p.map.get(k) || 0)), 1e-9);
	return reportCard(title, `${ctx.peers.length} selected ${ctx.level}s · mean scene mass`,
		bars,
		kinds.length ? peerMatrix(profiles, kinds, max, ctx.level) : null);
}

function comparePeerPlacement(ctx, title = "selected placement split") {
	const { reportCard, reportEmpty } = d();
	const cols = ["to-place", "scene"];
	const profiles = ctx.peers.map((p) => {
		const tp = p.rows.filter((r) => r.hasTp);
		const tpMass = tp.length ? tp.reduce((s, r) => s + (r.tpMass || 0), 0) / tp.length : 0;
		const scMass = tp.length ? tp.reduce((s, r) => s + (r.mass || 0), 0) / tp.length : 0;
		return { label: p.label, color: p.color, has: tp.length, map: new Map([["to-place", tpMass], ["scene", scMass]]) };
	});
	if (!profiles.some((p) => p.has)) return reportCard(title, null, reportEmpty("no placement (bbox) steps in selection"));
	const max = Math.max(...profiles.flatMap((p) => cols.map((c) => p.map.get(c) || 0)), 1e-9);
	return reportCard(title, `${ctx.peers.length} selected ${ctx.level}s · to-place vs scene`, peerMatrix(profiles, cols, max, ctx.level));
}

function comparePeerHeads(ctx, title = "selected head/layer attention") {
	const { reportCard } = d();
	const layerTotals = (rows) => {
		const byLayer = new Map();
		for (const r of rows) for (const g of r.a.head_grid || []) {
			const cur = byLayer.get(g.layer) || { s: 0, n: 0 };
			cur.s += g.mean_scale; cur.n++; byLayer.set(g.layer, cur);
		}
		return [...byLayer.entries()].map(([layer, { s, n }]) => [`L${layer}`, n ? s / n : 0]);
	};
	const keys = [...new Set(ctx.peers.flatMap((p) => layerTotals(p.rows).map(([k]) => k)))].sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)));
	const profiles = ctx.peers.map((p) => ({ label: p.label, color: p.color, map: new Map(layerTotals(p.rows)) }));
	const max = Math.max(...profiles.flatMap((p) => keys.map((k) => p.map.get(k) || 0)), 1e-9);
	return reportCard(title, `${ctx.peers.length} selected ${ctx.level}s`, peerMatrix(profiles, keys, max, ctx.level));
}

// ---- ablation research-question graphs ------------------------------------

// ONE "XML existence" card: the attention split across ATTRIBUTES with prompt XML
// kept (✓) vs stripped (✗) — two overlaid profiles (spider = normalized shape,
// matrix = raw mean ± sd). `onRows` / `offRows` are the pooled computed rows for
// each XML state. report.js orchestrates: an "all kinds" card first, then one
// SELECTED step kind (rather than laying every kind out).
function xmlSplitCard(title, sub, onRows, offRows) {
	const { overviewAggregate, reportCard, reportEmpty, showErr } = d();
	const errOn = !showErr || showErr();
	const profileFrom = (rows, label, color) => {
		const totals = overviewAggregate(rows).componentTotals;
		return { label, color, map: new Map(totals.map((c) => [c.component, c.score])), err: new Map(totals.map((c) => [c.component, c.sd || 0])) };
	};
	const profiles = [];
	if (onRows.length) profiles.push(profileFrom(onRows, "xml \u2713", "#6bd96e"));
	if (offRows.length) profiles.push(profileFrom(offRows, "xml \u2717", "#ff6b9d"));
	if (!profiles.length) return reportCard(title, sub, reportEmpty("no computed variants for this selection"));
	const comps = pickAxes(profiles.flatMap((pr) => [...pr.map.entries()]));
	const max = axisMax(profiles.flatMap((pr) => comps.map((c) => ({ v: pr.map.get(c) || 0, e: pr.err.get(c) || 0 }))), (x) => x.v, (x) => x.e, errOn);
	return reportCard(title, sub, spiderChart(profiles), comps.length ? peerMatrix(profiles, comps, max, "xml") : null);
}

// "Scene ordering" scatter: x = an object's position in the (shuffled) scene
// context (0 = first … 1 = last), y = attention to that object (graph 1) or to
// one of its attributes (graph 2). Points colored by shuffle method; the pale
// overlay is the per-position-bin mean (the aggregate trend). The corner shows
// Spearman ρ + p of attention-vs-position — the "does order matter?" readout.
// `rawPoints`: [{ x, y, color, label }]; `meanPoints`: [{ x, y }].
function ablationOrderCard(title, sub, rawPoints, meanPoints, opts = {}) {
	const { reportCard, reportEmpty, el } = d();
	if (!rawPoints.length) return reportCard(title, sub, reportEmpty(opts.empty || "no positioned objects in the selection yet"));
	const pts = rawPoints.map((p) => ({ x: p.x, y: p.y, color: p.color, r: 2.6, label: p.label }));
	for (const m of meanPoints) pts.push({ x: m.x, y: m.y, color: "#e8fcff", r: 4.6, label: `bin mean ${m.y.toFixed(4)}` });
	const tr = spearmanTrend(rawPoints.map((p) => p.x), rawPoints.map((p) => p.y));
	const arrow = tr.rho > 0.05 ? "\u2197" : tr.rho < -0.05 ? "\u2198" : "\u2192";
	const corr = rawPoints.length >= 3 ? `\u03c1=${tr.rho.toFixed(2)} ${arrow} p=${tr.p < 0.001 ? "<0.001" : tr.p.toFixed(3)}` : null;
	const chart = scatterChart(pts, {
		xLabel: "position in context  (0 = first \u2192 1 = last)",
		yFmt: opts.yFmt || ((v) => (Math.abs(v) >= 1 ? v.toFixed(v >= 100 ? 0 : 1) : v.toFixed(3))),
		xMax: 1, corrLabel: corr, legend: opts.legend, height: 230, logY: opts.logY,
	});
	return reportCard(title, sub, chart);
}

// Companion to ablationOrderCard: how the position→attention correlation holds up
// as you FOCUS on the objects that matter. x = focus threshold (min attention
// share), y = Spearman ρ(position, attention) over the objects kept at that
// threshold — one line per shuffle method. A flat line near 0 ⇒ ordering doesn't
// bias attention; a line that dives negative as focus rises ⇒ the FEW objects that
// matter are the early ones. y is fixed to [-1, 1] with a zero baseline (ρ is
// signed, which svgFrame/lineChart don't do). series: [{ method, color,
// points:[{x, y, n}] }].
function rhoCurveCard(title, sub, series, opts = {}) {
	const { reportCard, reportEmpty, el } = d();
	const withPts = (series || []).filter((s) => (s.points || []).length);
	if (!withPts.length) return reportCard(title, sub, reportEmpty(opts.empty || "not enough objects to correlate — widen the scope or lower the focus"));
	const W = 960, H = 250, padL = 46, padR = 14, padT = 12, padB = 34;
	// x = focus threshold. Linear by default; LOG (opts.logX) spreads the crowded
	// small values, with the x=0 ("all objects") point parked in its own left slot
	// since log(0) is undefined.
	const logX = !!opts.logX;
	const allX = withPts.flatMap((s) => s.points.map((p) => p.x));
	const xMax = Math.max(...allX, 0.005);
	const posX = allX.filter((x) => x > 0);
	const minPos = posX.length ? Math.min(...posX) : 0.0005;
	const lgLo = Math.log10(minPos), lgHi = Math.log10(xMax), lgSpan = (lgHi - lgLo) || 1;
	const lgAll = lgLo - lgSpan * 0.08; // where x=0 sits on a log axis
	const tx = (x) => logX ? (x > 0 ? Math.log10(x) : lgAll) : x;
	const tMin = logX ? lgAll : 0, tMax = logX ? lgHi : xMax;
	const X = (x) => padL + ((tx(x) - tMin) / ((tMax - tMin) || 1)) * (W - padL - padR);
	const yMin = opts.yMin ?? -0.5;
	const yMax = opts.yMax ?? 0.5;

	const Y = (y) =>
	padT +
	((yMax - Math.max(yMin, Math.min(yMax, y))) / (yMax - yMin)) *
		(H - padT - padB);
	const pct = (v) => `${(v * 100).toFixed(v > 0 && v < 0.02 ? 2 : 1)}%`;
	const svg = svgEl("svg", { class: "gsvg", viewBox: `0 0 ${W} ${H}`, style: "width:100%;height:auto;display:block;overflow:visible" });
	for (const gy of [-0.5, 0.25, 0, 0.25, 0.5]) {
		const yy = Y(gy);
		svg.appendChild(svgEl("line", { x1: padL, y1: yy.toFixed(1), x2: W - padR, y2: yy.toFixed(1), stroke: gy === 0 ? "rgba(255,255,255,0.24)" : "rgba(255,255,255,0.06)" }));
		svg.appendChild(svgEl("text", { x: padL - 6, y: yy.toFixed(1), fill: "rgba(220,230,245,0.5)", "font-size": 10, "text-anchor": "end", "dominant-baseline": "middle" }, gy.toFixed(1)));
	}
	const xticks = [];
	if (logX) {
		xticks.push({ x: 0, label: "all" });
		for (let k = Math.floor(lgLo); k <= Math.ceil(lgHi); k++) for (const m of [1, 2, 5]) {
			const v = m * Math.pow(10, k);
			if (v >= minPos * 0.999 && v <= xMax * 1.001) xticks.push({ x: v, label: pct(v) });
		}
	} else {
		for (let i = 0; i <= 4; i++) xticks.push({ x: (i / 4) * xMax, label: pct((i / 4) * xMax) });
	}
	for (const t of xticks) {
		const xx = X(t.x);
		svg.appendChild(svgEl("line", { x1: xx.toFixed(1), y1: padT, x2: xx.toFixed(1), y2: (H - padB).toFixed(1), stroke: "rgba(255,255,255,0.05)" }));
		svg.appendChild(svgEl("text", { x: xx.toFixed(1), y: H - padB + 14, fill: "rgba(220,230,245,0.5)", "font-size": 10, "text-anchor": "middle" }, t.label));
	}
	svg.appendChild(svgEl("text", { x: ((padL + W - padR) / 2).toFixed(1), y: H - 4, fill: "rgba(220,230,245,0.82)", "font-size": 11, "text-anchor": "middle" }, (opts.xLabel || "focus  (min attention share)") + (logX ? "  · log" : "")));
	const my = ((padT + (H - padB)) / 2).toFixed(1);
	svg.appendChild(svgEl("text", { x: 12, y: my, fill: "rgba(220,230,245,0.82)", "font-size": 11, "text-anchor": "middle", transform: `rotate(-90 12 ${my})` }, "Spearman \u03c1 (position \u00b7 attention)"));
	// mark where the scatter's currently-selected focus sits on the curve
	if (opts.markX != null && opts.markX > 0 && opts.markX <= xMax) {
		const mx = X(opts.markX);
		svg.appendChild(svgEl("line", { x1: mx.toFixed(1), y1: padT, x2: mx.toFixed(1), y2: (H - padB).toFixed(1), stroke: "rgba(232,193,74,0.6)", "stroke-dasharray": "3 3" }));
		svg.appendChild(svgEl("text", { x: mx.toFixed(1), y: padT + 9, fill: "rgba(232,193,74,0.9)", "font-size": 9.5, "text-anchor": "middle" }, "focus"));
	}
	for (const s of withPts) {
		const pts = [...s.points].sort((a, b) => a.x - b.x);
		let dd = "";
		pts.forEach((p, i) => { dd += (i ? "L" : "M") + X(p.x).toFixed(2) + "," + Y(p.y).toFixed(2) + " "; });
		svg.appendChild(svgEl("path", { d: dd.trim(), fill: "none", stroke: s.color, "stroke-width": 1.8, "stroke-linejoin": "round", "stroke-linecap": "round" }));
		for (const p of pts) {
			const dot = svgEl("circle", { cx: X(p.x).toFixed(2), cy: Y(p.y).toFixed(2), r: 2.8, fill: s.color });
			dot.appendChild(svgEl("title", null, `${s.method} · focus ≥${(p.x * 100).toFixed(2)}% · ρ=${p.y.toFixed(2)} · n=${p.n}`));
			svg.appendChild(dot);
		}
	}
	const wrap = el("div", { class: "gwrap" }, svg);
	wrap.appendChild(chartLegend(withPts.map((s) => ({ label: s.method, color: s.color }))));
	return reportCard(title, sub, wrap);
}

// "Spatial relevance" JOINTPLOT: x = an object's distance-to-target RANK (1 =
// closest), y = its ray-trace visibility RANK (1 = most visible). Points are
// grouped into ATTENTION TIERS (high / mid / low, by tercile) and colored
// categorically, with marginal distributions on the top (x) + right (y). The
// question reads straight off the marginals: if the HIGH-attention tier's top
// marginal skews to low distance rank (and its right marginal to low visibility
// rank), the model attends to what's close + visible. Spearman ρ (attention vs
// each rank, NEGATIVE = attends closer / more visible) rides in the corner.
// `rawPoints`: [{ x, y, attn, label }].
// Joint scatter of distance-rank (x) vs visibility-rank (y). Each point is one
// (variant, object): COLOR = its type (scene-order / step-kind / xml — cats come
// in via opts), SIZE = its attention (bigger circle ⇒ more attention). Marginals
// show how each type is distributed over closeness / visibility.
// Rendering one SVG <circle> per point freezes/crashes the DOM when there are
// thousands (variants × objects × replicates), and `Math.max(...hugeArray)`
// overflows the call stack. Cap the RENDERED dots (stats are still computed on
// the full set) via a deterministic stride, so the sample is uniform and stable
// across the report's frequent re-renders (no flicker).
const MAX_SCATTER_POINTS = 3000;
function downsamplePoints(arr, max = MAX_SCATTER_POINTS) {
	if (arr.length <= max) return arr;
	const stride = arr.length / max, out = [];
	for (let i = 0; out.length < max && i < arr.length; i += stride) out.push(arr[Math.floor(i)]);
	return out;
}

function spatialScatterCard(title, sub, rawPoints, opts = {}) {
	const { reportCard, reportEmpty } = d();
	if (!rawPoints.length) return reportCard(title, sub, reportEmpty(opts.empty || "no positioned objects with attention yet"));
	// reduce, NOT Math.max(...spread): the set can be tens of thousands of points
	// and a spread that large blows the call stack (a crash source here).
	let maxA = 1e-9, maxRank = 1;
	for (const p of rawPoints) { const a = p.attn || 0; if (a > maxA) maxA = a; if (p.x > maxRank) maxRank = p.x; if (p.y > maxRank) maxRank = p.y; }
	// Correlation is over ALL points; only the rendered dots are capped.
	const rd = spearman(rawPoints.map((p) => p.attn), rawPoints.map((p) => p.x));
	const rv = spearman(rawPoints.map((p) => p.attn), rawPoints.map((p) => p.y));
	const corr = rawPoints.length >= 3 ? `attn·dist \u03c1=${rd.toFixed(2)} · attn·vis \u03c1=${rv.toFixed(2)}` : null;
	// attention drives BOTH size and opacity: area-proportional radius (sqrt, so a
	// 4× busier object reads ~2× wider) and alpha ramped from 0.5 → 1.0 so heavier
	// objects read as more solid without hiding the low-attention tail.
	const shown = downsamplePoints(rawPoints);
	const pts = shown.map((p) => {
		const f = Math.sqrt(Math.max(0, p.attn || 0) / maxA);
		return { x: p.x, y: p.y, cat: p.cat, label: p.label, r: 2.6 + 6.4 * f, o: 0.5 + 0.5 * f };
	});
	const sizeLabel = shown.length < rawPoints.length
		? `○ size + opacity = attention · showing ${shown.length.toLocaleString()} of ${rawPoints.length.toLocaleString()} pts`
		: "○ size + opacity = attention";
	const chart = jointScatter(pts, {
		cats: opts.cats, xMax: maxRank, yMax: maxRank, corrLabel: corr, sizeLabel,
		xLabel: "distance-to-target rank  (1 = closest \u2192)",
		yLabel: "ray-visibility rank  (1 = most visible)",
		xFmt: (v) => String(Math.round(v)), yFmt: (v) => String(Math.round(v)),
	});
	return reportCard(title, sub, chart);
}

function peerMatrix(profiles, columns, max, corner = "") {
	const { el, heatColor } = d();
	const cols = `grid-template-columns:200px repeat(${columns.length},minmax(96px,1fr))`;
	const header = el("div", { class: "matrix-row", style: cols },
		el("div", { class: "matrix-head", text: corner }),
		...columns.map((c) => el("div", { class: "matrix-head", text: c, title: c })));
	const body = profiles.map((p) => el("div", { class: "matrix-row", style: cols },
		el("div", { class: "matrix-label", title: p.label },
			el("i", { style: `width:9px;height:9px;border-radius:50%;background:${p.color};display:inline-block;flex:none;margin-right:6px` }),
			el("span", { text: p.label })),
		...columns.map((c) => {
			const v = p.map.get(c) || 0;
			const s = p.err ? (p.err.get(c) || 0) : 0;
			return el("div", { class: "matrix-cell", style: `background:${heatColor(v / (max || 1e-9))}`, title: `${p.label} · ${c} · ${fmtPM(v, s)}`, text: v ? fmtNum(v, 2) : "" });
		})));
	return el("div", { class: "matrix-wrap" }, header, ...body);
}

function attributeProfileGraph(items, { title = "attribute profile", sub = "" } = {}) {
	// Attribute profiles are always rendered as a spider/radial chart (comparison
	// or not); the ranked bars ride along underneath as a readable legend.
	const { el, summaryBar, compHex, reportCard, reportEmpty, showErr } = d();
	const all = items || [];
	if (!all.length) return reportCard(title, sub, reportEmpty("no attributes recorded"));
	// Ranked bars show the strongest attributes; the spider keeps the full profile
	// so its fixed ring always carries every attribute (absent ones sit at the hub).
	const barItems = all.slice(0, MAX_ATTR_AXES);
	const max = axisMax(barItems, (c) => c.score, (c) => c.sd || 0, !showErr || showErr());
	const bars = el("div", { class: "sbars" }, ...barItems.map((c) => summaryBar({
		color: compHex(c.component), label: c.component, value: c.score, sd: c.sd ?? null, max,
		title: `${c.component} · ${fmtPM(c.score, c.sd)}`,
	})));
	const profile = {
		label: sub || title, color: "#7aa2f7",
		map: new Map(all.map((c) => [c.component, c.score])),
		err: new Map(all.map((c) => [c.component, c.sd || 0])),
	};
	return reportCard(title, sub, spiderChart([profile]), el("div", { class: "attr-legend" }, bars));
}

function orderedComponents(components) {
	const have = new Set(components);
	const ordered = ATTR_AXIS_ORDER.filter((c) => have.has(c));
	for (const c of components) if (!ordered.includes(c)) ordered.push(c);
	return ordered;
}

// Cap on the ranked attribute bars and the comparison matrix columns. The spider
// ring itself is independent — it always shows the full canonical ATTR_AXIS_ORDER.
const MAX_ATTR_AXES = 12;

// Top-N attributes by total score, then laid out in the fixed ATTR_AXIS_ORDER.
// Used to choose (and order) the columns of the comparison matrix.
function pickAxes(scorePairs, n = MAX_ATTR_AXES) {
	const totals = new Map();
	for (const [c, s] of scorePairs) totals.set(c, (totals.get(c) || 0) + (s || 0));
	const top = [...totals.entries()].sort((a, b) => b[1] - a[1]).slice(0, n).map(([c]) => c);
	return orderedComponents(top);
}

// Native hover tooltip decoding the abbreviated axis labels ("rel" -> relationships).
// Canvas has no per-glyph DOM, so we hit-test the recorded label anchors and set
// the canvas title as the pointer moves.
function attachAxisTooltip(canvas, spots, ABBR) {
	const onMove = (ev) => {
		const rect = canvas.getBoundingClientRect();
		const sx = canvas.width / (rect.width || 1), sy = canvas.height / (rect.height || 1);
		const mx = (ev.clientX - rect.left) * sx, my = (ev.clientY - rect.top) * sy;
		let hit = null;
		for (const s of spots) {
			const left = s.align === "right" ? s.lx - 64 : s.align === "center" ? s.lx - 30 : s.lx - 6;
			const right = s.align === "left" ? s.lx + 64 : s.align === "center" ? s.lx + 30 : s.lx + 6;
			if (mx >= left && mx <= right && Math.abs(my - s.ly) <= 11) { hit = s; break; }
		}
		const t = hit ? `${ABBR[hit.component] ?? hit.component} = ${hit.component}` : "";
		if (canvas.title !== t) { canvas.title = t; canvas.style.cursor = hit ? "help" : "default"; }
	};
	canvas.addEventListener("mousemove", onMove);
	canvas.addEventListener("mouseleave", () => { canvas.title = ""; canvas.style.cursor = "default"; });
}

// Turn each profile into [0..1] radial coordinates (+ matching error fractions).
// EACH profile is normalized INDIVIDUALLY as the ratio of every attribute to that
// profile's MOST-ATTENDED attribute: the peak MEAN vertex reaches the rim (=1) and
// every other axis shows its plain fraction of that peak. Errors ride the same
// per-axis scale (a whisker may cross the rim on the peak axis — that's honest).
// The rule is per-profile and independent of the error toggle, so a single
// selection and a multi-profile overlay scale identically: the shape you read
// never changes when you add peers or flip error bars, and each polygon simply
// lays on top of the others, self-scaled to its own peak.
function spiderNormalize(profiles, components) {
	// SUM (L1) normalization, not peak-to-peak: each attribute is its SHARE of the
	// profile's total attention (value / Σ over all attributes) — "how much does
	// this field attend relative to the sum". So the shape is a proportion
	// distribution, comparable across profiles regardless of magnitude. A single
	// SHARED display scale (the largest share across every profile) maps to the
	// ring edge, so the shares stay directly comparable AND use the ring — never a
	// per-profile peak rescale that would flatten differences in magnitude.
	const props = profiles.map((p) => {
		const rawVals = components.map((c) => Math.max(0, p.map.get(c) || 0));
		const rawErrs = components.map((c) => (p.err ? Math.max(0, p.err.get(c) || 0) : 0));
		const sum = rawVals.reduce((a, b) => a + b, 0) || 1;
		return { vals: rawVals.map((v) => v / sum), errs: rawErrs.map((e) => e / sum) };
	});
	const scale = Math.max(1e-9, ...props.flatMap((p) => p.vals));
	return props.map((p) => ({
		vals: p.vals.map((v) => v / scale),
		errs: p.errs.map((e) => e / scale),
	}));
}

function spiderChart(profiles) {
	const { el, COMPONENT_ABBR, compHex, showErr } = d();
	const multi = profiles.length > 1;
	const errOn = !showErr || showErr();
	// FIXED ring: every attribute spider shows the same axes in the same order
	// (the canonical ATTR_AXIS_ORDER), so a given attribute sits at the same clock
	// position in every chart — single, kind, scene, or comparison — and adjacent
	// or overlaid spiders line up. Unexpected extra keys are appended stably so
	// nothing present in the data is silently dropped.
	const present = new Set(profiles.flatMap((p) => [...p.map.keys()]));
	const components = [...ATTR_AXIS_ORDER, ...[...present].filter((c) => !ATTR_AXIS_ORDER.includes(c))];
	// Per-vertex coordinates in [0..1]: each profile is normalized to its attribute
	// SHARES (value / its own sum), then a shared scale maps the largest share to
	// the ring (see spiderNormalize). One profile or several, the rule is identical
	// — an overlaid comparison reads as the SHARE distribution, comparable in shape.
	const norm = spiderNormalize(profiles, components);
	const canvas = el("canvas", { class: "attr-radial" });
	canvas.width = 320; canvas.height = 320;
	const ctx = canvas.getContext("2d");
	const cx = 160, cy = 160, r0 = 14, r1 = 112;
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	ctx.strokeStyle = "rgba(255,255,255,0.08)";
	ctx.lineWidth = 1;
	for (const rr of [0.25, 0.5, 0.75, 1]) {
		ctx.beginPath();
		components.forEach((_, i) => {
			const ang = -Math.PI / 2 + (i / components.length) * Math.PI * 2;
			const x = cx + Math.cos(ang) * (r0 + (r1 - r0) * rr);
			const y = cy + Math.sin(ang) * (r0 + (r1 - r0) * rr);
			i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
		});
		ctx.closePath(); ctx.stroke();
	}
	const labelSpots = [];
	components.forEach((component, i) => {
		const ang = -Math.PI / 2 + (i / components.length) * Math.PI * 2;
		const gx = cx + Math.cos(ang) * r1, gy = cy + Math.sin(ang) * r1;
		ctx.strokeStyle = "rgba(255,255,255,0.12)";
		ctx.beginPath(); ctx.moveTo(cx + Math.cos(ang) * r0, cy + Math.sin(ang) * r0); ctx.lineTo(gx, gy); ctx.stroke();
		const lx = cx + Math.cos(ang) * (r1 + 26), ly = cy + Math.sin(ang) * (r1 + 20);
		// Tint each axis label to match its slug-legend chip (and the ranked bars)
		// so the abbreviated axes are decodable at a glance, single or comparison.
		ctx.fillStyle = compHex ? compHex(component) : "rgba(220,230,245,0.78)";
		ctx.font = "12px ui-monospace, Menlo, monospace";
		const align = lx < cx - 8 ? "right" : lx > cx + 8 ? "left" : "center";
		ctx.textAlign = align;
		ctx.textBaseline = "middle";
		ctx.fillText(COMPONENT_ABBR[component] ?? component, lx, ly, 64);
		labelSpots.push({ component, lx, ly, align });
	});
	attachAxisTooltip(canvas, labelSpots, COMPONENT_ABBR);
	profiles.forEach((p, pi) => {
		const color = p.color || COMPARE_COLORS[pi % COMPARE_COLORS.length];
		ctx.beginPath();
		components.forEach((_, i) => {
			const ang = -Math.PI / 2 + (i / components.length) * Math.PI * 2;
			const val = Math.max(0, Math.min(1, norm[pi].vals[i]));
			const x = cx + Math.cos(ang) * (r0 + (r1 - r0) * val);
			const y = cy + Math.sin(ang) * (r0 + (r1 - r0) * val);
			i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
		});
		ctx.closePath();
		ctx.fillStyle = `${color}33`;
		ctx.strokeStyle = color;
		ctx.lineWidth = 2;
		ctx.fill(); ctx.stroke();
	});
	// Error whiskers: one per vertex, aligned along its axis. Near-hub points are
	// skipped (anti-clutter) and grossly uncertain ones are marked with an "×".
	profiles.forEach((p, pi) => {
		if (!p.err || !errOn) return;
		const color = p.color || COMPARE_COLORS[pi % COMPARE_COLORS.length];
		components.forEach((_, i) => {
			const s = norm[pi].errs[i];
			if (s <= 0) return;
			const ang = -Math.PI / 2 + (i / components.length) * Math.PI * 2;
			drawRadialError(ctx, {
				cx, cy, ang, r0, r1,
				valMean: norm[pi].vals[i],
				valErr: s, color, cap: multi ? 3 : 4, lw: multi ? 1.2 : 1.6,
			});
		});
	});
	const legend = el("div", { class: `attr-legend${multi ? " compact" : ""}` }, ...profiles.map((p, i) => el("span", { class: "pin-pill", title: p.label },
		el("i", { style: `width:9px;height:9px;border-radius:50%;background:${p.color || COMPARE_COLORS[i % COMPARE_COLORS.length]};display:inline-block;flex:none` }),
		el("span", { text: p.label }))));
	const wrap = el("div", { class: "attr-radial-wrap" }, canvas, legend);
	// Always decode the abbreviated axes (e.g. "nn" = noun_phrase) beneath the ring
	// so every attribute spider is self-explanatory — single-item and comparison alike.
	const axisLegend = el("div", { class: "attr-axis-legend" }, ...components.map((c) => el("span", { class: "axis-pill", title: c },
		el("i", { style: `background:${compHex ? compHex(c) : "#7aa2f7"}` }),
		el("b", { text: COMPONENT_ABBR[c] ?? c }),
		el("span", { text: c }))));
	return el("div", { class: "attr-radial-block" }, wrap, axisLegend);
}

function compareAttributesGraph(rows, { title = "attribute comparison", group = "kind", order = null } = {}) {
	const { el, reportCard, reportEmpty, heatColor } = d();
	if (!rows.length) return reportCard(title, group, reportEmpty("no computed rows to compare"));
	const groups = groupByCompare(rows, group, order);
	const compTotals = new Map();
	for (const g of groups) for (const c of g.components) compTotals.set(c.component, (compTotals.get(c.component) || 0) + c.score);
	const comps = pickAxes([...compTotals.entries()]);
	if (!comps.length) return reportCard(title, group, reportEmpty("no attributes in selected rows"));
	const val = (g, c) => (g.map.get(c) || 0);
	const max = Math.max(...groups.flatMap((g) => comps.map((c) => val(g, c))), 1e-9);
	const profiles = groups.slice(0, 6).map((g, i) => ({ label: g.label, map: g.map, color: COMPARE_COLORS[i % COMPARE_COLORS.length] }));
	const cols = `grid-template-columns:200px repeat(${comps.length},minmax(96px,1fr))`;
	const header = el("div", { class: "matrix-row", style: cols },
		el("div", { class: "matrix-head", text: group }),
		...comps.map((c) => el("div", { class: "matrix-head", text: c, title: c })));
	const body = groups.map((g) => el("div", { class: "matrix-row", style: cols },
		el("div", { class: "matrix-label", text: g.label, title: g.label }),
		...comps.map((c) => {
			const v = val(g, c);
			return el("div", { class: "matrix-cell", style: `background:${heatColor(v / max)}`, title: `${g.label} · ${c} · ${v.toFixed(4)}`, text: v ? v.toFixed(2) : "" });
		})));
	return reportCard(title, group,
		spiderChart(profiles),
		el("div", { class: "hint", text: (profiles.length < groups.length ? `spider overlays first ${profiles.length} profiles; matrix includes all ${groups.length} · ` : "") + "spider is normalized per profile (compares shape); matrix cells are raw scores" }),
		el("div", { class: "matrix-wrap" }, header, ...body));
}

function groupByCompare(rows, group, order = null) {
	const buckets = new Map();
	for (const r of rows) {
		const label = group === "step" || group === "pinned" ? `#${r.step?.event_index ?? "?"} ${r.template}` : r.template;
		if (!buckets.has(label)) buckets.set(label, []);
		buckets.get(label).push(r);
	}
	const labels = order ? order.filter((x) => buckets.has(x)) : [...buckets.keys()];
	return labels.map((label) => {
		const comp = new Map();
		for (const r of buckets.get(label) || []) for (const c of r.agg.componentTotals) comp.set(c.component, (comp.get(c.component) || 0) + c.score);
		const components = [...comp.entries()].map(([component, score]) => ({ component, score })).sort((a, b) => b.score - a.score);
		return { label, components, map: comp };
	});
}

function placementObjectsGraph(a) {
	const { el, reportCard, reportEmpty, summaryBar, COLORS } = d();
	const outs = ((a.agg && a.agg.outputs) || []).filter((o) => o.to_place);
	if (!outs.length) return reportCard("placed objects", null, reportEmpty("no per-object placement rollups"));
	const cards = outs.slice(0, 12).map((o) => {
		const tp = o.to_place?.mass ?? 0, sc = o.scene?.mass ?? 0, max = Math.max(tp, sc, 1e-9);
		return reportCard(o.entity, `${o.n} token${o.n === 1 ? "" : "s"}`,
			el("div", { class: "sbars" },
				summaryBar({ color: COLORS.to_place, label: "to-place", value: tp, max, title: "attention on objects being placed" }),
				summaryBar({ color: "#4af0e0", label: "scene", value: sc, max, title: "attention on surrounding scene" })));
	});
	return el("div", { class: "graph-grid" }, ...cards);
}

// ===== token ordering ======================================================
// How attention relates to WHERE a token sits (position) and to the ORDER in
// which scene/to-place items appear. Shared position→x transform + generic line
// and scatter primitives, then the four analyses (scene mass vs position,
// context order vs attention, plotting order vs scene mass, input vs output
// order). All read from the compact payload the client already holds.

// Chosen position basis (from the report control): "n" (tokens from step start),
// "ratio" (fraction of the step), or "log" (log(1+n) — RoPE's geometric ladder).
// "feature" is NOT a position basis (it maps attention onto output items, not token
// order) — the progression/trajectory graphs fall back to "n" for it, and only the
// per-feature card reacts via `featureMode()`.
function _rawX() { const f = d().xMode; return f ? f() : "n"; }
function curX() { const m = _rawX(); return m === "feature" ? "n" : m; }
function featureMode() { return _rawX() === "feature"; }
const X_LABEL = { n: "scored token index", ratio: "position (fraction of step)", log: "log(1 + token index)" };
function xLabelFor(mode) { return X_LABEL[mode] || "position"; }
function xFmtFor(mode) {
	if (mode === "ratio") return (v) => v.toFixed(2);
	if (mode === "log") return (v) => String(Math.round(Math.expm1(v))); // ticks read as real token counts
	return (v) => String(Math.round(v));
}
// Map scored-token rank (0…n−1 within the compact list) to chart x. Uses array
// index, NOT absolute token.index — reasoning is subsampled but every output
// token is kept; absolute gaps squash the output band (see summaryTrajectory).
function xformTrajPos(i, tr, mode) {
	const n = tr.toks.length;
	const span = tr.span;
	const out = Math.min(tr.outStart, n);
	if (mode === "n") return i;
	if (mode === "ratio") return span > 0 ? i / span : 0;
	if (mode === "log") {
		// Log spacing within each phase; each phase still owns width ∝ token count.
		const nO = n - out;
		if (i < out) {
			if (out <= 1) return 0;
			return (Math.log1p(i) / Math.log1p(out - 1)) * (out - 1);
		}
		const j = i - out;
		if (nO <= 1) return Math.max(0, out - 1);
		return (out - 1) + (Math.log1p(j) / Math.log1p(nO - 1)) * (nO - 1);
	}
	return i;
}
// Map a context-entity position (order in scene context) — still uses absolute span.
function xformPos(relPos, span, mode) {
	if (mode === "ratio") return span > 0 ? relPos / span : 0;
	if (mode === "log") return Math.log1p(Math.max(0, relPos));
	return relPos;
}
const pctFmt = (v) => `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`;

// Append an alpha byte to a "#rrggbb" color (for faint overlays / bands).
function hexA(hex, a) {
	if (typeof hex !== "string" || hex[0] !== "#" || hex.length < 7) return hex;
	return hex + Math.round(Math.max(0, Math.min(1, a)) * 255).toString(16).padStart(2, "0");
}
function templateColorOf(template, rows) {
	const tc = d().templateColor;
	return tc ? tc(template, [...new Set(rows.map((r) => r.template))]) : "#7aa2f7";
}
function chartLegend(items) {
	const { el } = d();
	return el("div", { class: "seg-legend", style: "font-size:10px;gap:12px;margin-top:6px;flex-wrap:wrap" },
		...items.filter(Boolean).map((it) => el("span", {}, el("span", { class: "frame-sw", style: `background:${it.color}` }), el("span", { text: ` ${it.label}` }))));
}

// --- SVG chart toolkit ------------------------------------------------------
// The token-ordering charts render as real SVG DOM elements (crisp vectors that
// stay sharp at any width, and hoverable) rather than a flat canvas raster.
// svgFrame lays out the shared axis; each chart appends <path>/<circle>/<text>;
// chartHover wires a crosshair + value tooltip. All coordinates stay in the
// logical W×H space and the <svg> scales to the card width via its viewBox.
const SVGNS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs, ...kids) {
	const e = document.createElementNS(SVGNS, tag);
	if (attrs) for (const k in attrs) { const v = attrs[k]; if (v != null) e.setAttribute(k, String(v)); }
	for (const c of kids) if (c != null) e.appendChild(c.nodeType ? c : document.createTextNode(String(c)));
	return e;
}
const escTip = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
// Shared hover/tooltip stylesheet, injected once.
function ensureChartCSS() {
	if (document.getElementById("tf-chart-css")) return;
	const s = document.createElement("style");
	s.id = "tf-chart-css";
	s.textContent =
		".gwrap{position:relative}" +
		".gwrap.square{max-width:460px;margin:0 auto}" +
		".gsvg{touch-action:none}" +
		".gsvg .garea{transition:fill-opacity .12s ease}" +
		".gsvg .garea:hover{fill-opacity:.98}" +
		".gsvg .gpt{cursor:pointer;transition:fill-opacity .12s ease}" +
		".gsvg .gpt:hover{fill-opacity:1;stroke:rgba(255,255,255,.85);stroke-width:1}" +
		".graph-tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .08s ease;z-index:6;" +
		"background:rgba(13,15,20,.96);border:1px solid rgba(255,255,255,.16);border-radius:6px;padding:6px 8px;" +
		"font:11px ui-monospace,Menlo,monospace;color:#dce6f5;white-space:nowrap;box-shadow:0 4px 14px rgba(0,0,0,.5);max-width:72%}" +
		".graph-tip .xh{opacity:.55;margin-bottom:3px}" +
		".graph-tip .r{display:flex;justify-content:space-between;gap:14px;line-height:1.55}" +
		".graph-tip .sw{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px}";
	document.head.appendChild(s);
}
// Shared axis frame as SVG: output shade, y grid + ticks (0..yMax), three x
// ticks, axis label. Returns the <svg>, coordinate mappers, and plot rectangle.
function svgFrame(W, H, { padL, padR, padT, padB, xMin, xMax, yMax, yFmt, xFmt, xLabel, shade, xTicks = true, botGutter = 0, logY = false }) {
	ensureChartCSS();
	const px0 = padL, px1 = W - padR, py0 = padT, py1 = H - padB;
	const xr = (xMax - xMin) || 1; yMax = yMax || 1;
	const X = (x) => px0 + ((x - xMin) / xr) * (px1 - px0);
	// log-y (log1p) spreads a heavy-tailed magnitude (raw attention sums span 0..1000s)
	// so most points don't collapse onto the axis. Gridlines stay evenly spaced;
	// their LABELS carry the (inverse-mapped) data value.
	const _lg = (v) => Math.log10(1 + Math.max(0, v));
	const yTop = logY ? (_lg(yMax) || 1) : yMax;
	const Y = logY
		? (y) => py1 - (_lg(Math.max(0, Math.min(yMax, y))) / yTop) * (py1 - py0)
		: (y) => py1 - (Math.max(0, Math.min(yMax, y)) / yMax) * (py1 - py0);
	const svg = svgEl("svg", { class: "gsvg", viewBox: `0 0 ${W} ${H}`, style: "width:100%;height:auto;display:block;overflow:visible" });
	if (shade && shade.to > shade.from) svg.appendChild(svgEl("rect", { x: X(shade.from), y: py0, width: Math.max(0, X(shade.to) - X(shade.from)), height: py1 - py0, fill: "rgba(122,79,208,0.14)" }));
	for (const f of [0, 0.25, 0.5, 0.75, 1]) {
		const yy = py1 - f * (py1 - py0);
		const val = logY ? (Math.pow(10, f * yTop) - 1) : yMax * f;
		svg.appendChild(svgEl("line", { x1: px0, y1: yy, x2: px1, y2: yy, stroke: "rgba(255,255,255,0.07)" }));
		svg.appendChild(svgEl("text", { x: px0 - 5, y: yy, fill: "rgba(220,230,245,0.5)", "font-size": 10, "text-anchor": "end", "dominant-baseline": "middle" }, yFmt ? yFmt(val) : val.toFixed(2)));
	}
	// A bottom gutter (e.g. the reasoning token-type strip) pushes ticks + label down.
	const tickY = py1 + botGutter;
	if (xTicks) for (const f of [0, 0.5, 1]) {
		const xv = xMin + xr * f;
		svg.appendChild(svgEl("text", { x: X(xv), y: tickY + 6, fill: "rgba(220,230,245,0.5)", "font-size": 10, "text-anchor": "middle", "dominant-baseline": "hanging" }, xFmt ? xFmt(xv) : String(Math.round(xv))));
	}
	if (xLabel) svg.appendChild(svgEl("text", { x: px1, y: tickY + 16, fill: "rgba(220,230,245,0.72)", "font-size": 10, "text-anchor": "end", "dominant-baseline": "hanging" }, xLabel));
	return { svg, X, Y, px0, px1, py0, py1, yMax, W, H, botGutter };
}
// A corner annotation (e.g. a correlation label) in the plot's top-right.
function svgCorner(fr, text) {
	if (text) fr.svg.appendChild(svgEl("text", { x: fr.px1 - 2, y: fr.py0 + 1, fill: "rgba(220,230,245,0.85)", "font-size": 11, "text-anchor": "end", "dominant-baseline": "hanging" }, text));
}
// Vertical segment markers shared by the line + stacked-area charts (e.g. output
// item / attribute boundaries). vlines: [{x (data-space), label, major}]. major =
// item (solid, top label); minor = attribute field (faint dashed, rotated bottom
// label). Labels de-collide so a dense output stays readable.
function _drawVlines(fr, vlines) {
	if (!vlines || !vlines.length) return;
	let lastMaj = -1e9, lastMin = -1e9;
	for (const v of vlines) {
		const px = fr.X(v.x);
		if (px < fr.px0 - 0.5 || px > fr.px1 + 0.5) continue;
		fr.svg.appendChild(svgEl("line", { x1: px.toFixed(1), y1: fr.py0, x2: px.toFixed(1), y2: fr.py1, stroke: v.major ? "rgba(255,255,255,0.3)" : "rgba(160,185,225,0.16)", "stroke-width": v.major ? 1 : 0.6, "stroke-dasharray": v.major ? null : "2 3" }));
		if (!v.label) continue;
		// A dark stroke halo (paint-order:stroke) keeps labels legible over the colored
		// areas beneath them — plain low-opacity text was hard to read.
		if (v.major) { if (px - lastMaj < 26) continue; lastMaj = px;
			fr.svg.appendChild(svgEl("text", { x: (px + 3).toFixed(1), y: fr.py0 + 1, fill: "rgba(238,244,255,0.96)", stroke: "rgba(8,10,18,0.92)", "stroke-width": 3, "paint-order": "stroke", "font-size": 10, "font-weight": 600, "text-anchor": "start", "dominant-baseline": "hanging" }, v.label));
		} else { if (px - lastMin < 9) continue; lastMin = px;
			fr.svg.appendChild(svgEl("text", { x: (px + 1).toFixed(1), y: fr.py1 - 3, fill: "rgba(214,226,246,0.92)", stroke: "rgba(8,10,18,0.9)", "stroke-width": 2.4, "paint-order": "stroke", "font-size": 8.5, "text-anchor": "start", transform: `rotate(-90 ${(px + 1).toFixed(1)} ${(fr.py1 - 3).toFixed(1)})` }, v.label));
		}
	}
}
// Translucent highlight over one data-space x-range [x0,x1] marking the selected
// output item. Drawn ON TOP (with a clear border) so it stays visible over the
// opaque stacked areas as well as the line chart.
function _drawHi(fr, hi) {
	if (!hi || !(hi[1] > hi[0])) return;
	const a = fr.X(hi[0]), b = fr.X(hi[1]);
	const lo = Math.max(fr.px0, Math.min(a, b)), hiX = Math.min(fr.px1, Math.max(a, b));
	if (hiX - lo < 0.5) return;
	fr.svg.appendChild(svgEl("rect", { x: lo.toFixed(1), y: fr.py0, width: (hiX - lo).toFixed(1), height: fr.py1 - fr.py0, fill: "rgba(122,162,247,0.13)", stroke: "rgba(150,185,255,0.72)", "stroke-width": 1.2, "pointer-events": "none" }));
}
// Reasoning token-type strip: a thin stacked mini-histogram drawn in the bottom
// gutter, one column per bin, spanning the reasoning x-range — "what kind of text is
// the model generating here" (the tokens' OWN type, not what they attend to).
function _drawXHist(fr, hist) {
	if (!hist || !hist.bins.length || !fr.botGutter) return;
	const gy1 = fr.py1 + fr.botGutter - 3, gh = fr.botGutter - 6;
	if (gh < 4) return;
	const bx0 = fr.X(hist.x0), bx1 = fr.X(hist.x1), bw = (bx1 - bx0) / hist.nb;
	fr.svg.appendChild(svgEl("line", { x1: bx0.toFixed(1), y1: (gy1 + 1.5).toFixed(1), x2: bx1.toFixed(1), y2: (gy1 + 1.5).toFixed(1), stroke: "rgba(255,255,255,0.12)" }));
	for (let bi = 0; bi < hist.bins.length; bi++) {
		const parts = hist.bins[bi].parts;
		if (!parts.length) continue;
		const x = bx0 + bi * bw; let y = gy1;
		for (const p of parts) { const h = p.frac * gh; y -= h; fr.svg.appendChild(svgEl("rect", { x: (x + 0.4).toFixed(1), y: y.toFixed(1), width: Math.max(0.6, bw - 0.8).toFixed(1), height: Math.max(0.5, h).toFixed(1), fill: TYPE_STYLE[p.t] || "#8a8f98" })); }
	}
}
// Clickable output-item bands: a transparent hit rect per band (on top, catches
// clicks → onBand(id)) plus a highlight behind the data for the selected one. The
// hover crosshair still works (pointermove bubbles from the rects to the <svg>).
function _drawBands(fr, bands, selId, onBand) {
	if (!bands || !bands.length || typeof onBand !== "function") return;
	for (const bd of bands) {
		const a = fr.X(bd.x0), b = fr.X(bd.x1);
		const lo = Math.max(fr.px0, Math.min(a, b)), hiX = Math.min(fr.px1, Math.max(a, b));
		if (hiX - lo < 0.5) continue;
		const on = bd.id === selId;
		if (on) _drawHi(fr, [bd.x0, bd.x1]);
		const hit = svgEl("rect", { x: lo.toFixed(1), y: fr.py0, width: (hiX - lo).toFixed(1), height: fr.py1 - fr.py0, fill: "transparent", style: "cursor:pointer" });
		hit.addEventListener("click", (ev) => { ev.stopPropagation(); onBand(bd.id); });
		hit.appendChild(svgEl("title", {}, on ? `${bd.label} — click to clear` : `${bd.label} — click for breakdown`));
		fr.svg.appendChild(hit);
	}
}
// Crosshair + value tooltip shared by the line and stacked-area charts. Attaches
// to the <svg> so per-element :hover still fires. entries: [{label,color,values[]}]
// aligned to xs (ascending). Values are read at the x nearest the cursor.
function chartHover(wrap, fr, xs, entries, { xFmt, vFmt } = {}) {
	if (!xs || !xs.length || !entries.length) return;
	const { el } = d();
	const svg = fr.svg;
	const guide = svgEl("line", { y1: fr.py0, y2: fr.py1, stroke: "rgba(255,255,255,0.42)", "stroke-width": 1, "stroke-dasharray": "3 3", visibility: "hidden", "pointer-events": "none" });
	svg.appendChild(guide);
	const tip = el("div", { class: "graph-tip" });
	wrap.appendChild(tip);
	const xf = xFmt || ((v) => String(Math.round(v)));
	const vf = vFmt || ((v) => v.toFixed(3));
	const hide = () => { guide.setAttribute("visibility", "hidden"); tip.style.opacity = "0"; };
	svg.addEventListener("pointermove", (ev) => {
		const r = svg.getBoundingClientRect();
		const vx = (ev.clientX - r.left) * (fr.W / r.width);
		const vy = (ev.clientY - r.top) * (fr.H / r.height);
		if (vx < fr.px0 - 2 || vx > fr.px1 + 2 || vy < fr.py0 - 2 || vy > fr.py1 + 2) return hide();
		let idx = 0, best = Infinity;
		for (let i = 0; i < xs.length; i++) { const dd = Math.abs(fr.X(xs[i]) - vx); if (dd < best) { best = dd; idx = i; } }
		const gx = fr.X(xs[idx]);
		guide.setAttribute("x1", gx); guide.setAttribute("x2", gx); guide.setAttribute("visibility", "visible");
		const rowsHtml = entries.map((e) => ({ e, v: +(e.values[idx] || 0) })).filter((o) => o.v > 1e-9).sort((a, b) => b.v - a.v).slice(0, 9)
			.map((o) => `<div class="r"><span><span class="sw" style="background:${o.e.color}"></span>${escTip(o.e.label)}</span><b>${escTip(vf(o.v))}</b></div>`).join("");
		tip.innerHTML = `<div class="xh">${escTip(xf(xs[idx]))}</div>` + (rowsHtml || `<div style="opacity:.5">no attention here</div>`);
		const wr = wrap.getBoundingClientRect();
		const tw = tip.offsetWidth || 150, th = tip.offsetHeight || 60;
		let left = ev.clientX - wr.left + 14;
		if (left + tw > wr.width) left = ev.clientX - wr.left - tw - 14;
		let top = ev.clientY - wr.top + 12;
		if (top + th > wr.height) top = ev.clientY - wr.top - th - 12;
		tip.style.left = Math.max(0, left) + "px";
		tip.style.top = Math.max(0, top) + "px";
		tip.style.opacity = "1";
	});
	svg.addEventListener("pointerleave", hide);
}
// One knob to nudge the whole token-ordering tab's chart height.
const CHART_H = 0.86;

// Generic multi-series SVG line chart. Each series: { points:[[x,y]...] (x
// ascending), color, width?, dash?, area? (fill to baseline), faint?, marker?,
// band?:{hi:[],lo:[]} aligned to points (translucent envelope) }.
function lineChart(series, opts = {}) {
	const { el, niceMax } = d();
	const gutter = opts.xHist ? 22 : 0;   // room for the reasoning token-type strip
	const W = 960, H = Math.round((opts.height || 128) * CHART_H) + gutter, padL = 46, padR = 14, padT = 10, padB = 26 + gutter;
	const allPts = series.flatMap((s) => s.points || []);
	if (!allPts.length) return el("div", { class: "hint", text: "no data to plot" });
	const xMin = opts.xMin != null ? opts.xMin : Math.min(...allPts.map((p) => p[0]));
	const xMax = opts.xMax != null ? opts.xMax : Math.max(...allPts.map((p) => p[0]));
	const hiVals = allPts.map((p) => p[1]).concat(series.flatMap((s) => (s.band ? s.band.hi : [])));
	const yMax = opts.yMax != null ? opts.yMax : (niceMax ? niceMax(Math.max(...hiVals, 1e-9)) : Math.max(...hiVals, 1e-9) * 1.05);
	const fr = svgFrame(W, H, { padL, padR, padT, padB, xMin, xMax, yMax, yFmt: opts.yFmt, xFmt: opts.xFmt, xLabel: opts.xLabel, shade: opts.shade, botGutter: gutter });
	const dOf = (pts, close) => { let s = ""; for (let i = 0; i < pts.length; i++) s += (i ? "L" : "M") + fr.X(pts[i][0]).toFixed(2) + "," + fr.Y(pts[i][1]).toFixed(2) + " "; return close ? s + "Z" : s.trim(); };
	for (const s of series) {
		const pts = s.points || [];
		if (!pts.length) continue;
		const color = s.color || "#7aa2f7";
		if (s.band && s.band.hi && s.band.lo) {
			const up = pts.map((p, i) => [p[0], s.band.hi[i]]);
			const dn = pts.map((p, i) => [p[0], s.band.lo[i]]).reverse();
			fr.svg.appendChild(svgEl("path", { d: dOf(up.concat(dn), true), fill: hexA(color, 0.16), stroke: "none" }));
		}
		if (s.area) {
			const base = [[pts[pts.length - 1][0], 0], [pts[0][0], 0]];
			fr.svg.appendChild(svgEl("path", { d: dOf(pts.concat(base), true), fill: hexA(color, 0.14), stroke: "none" }));
		}
		fr.svg.appendChild(svgEl("path", {
			d: dOf(pts), fill: "none", stroke: s.faint ? hexA(color, 0.32) : color,
			"stroke-width": s.width || (s.faint ? 1 : 1.8), "stroke-dasharray": s.dash ? s.dash.join(" ") : null,
			"stroke-linejoin": "round", "stroke-linecap": "round",
		}));
		if (s.marker) for (const p of pts) fr.svg.appendChild(svgEl("circle", { cx: fr.X(p[0]).toFixed(2), cy: fr.Y(p[1]).toFixed(2), r: 2.6, fill: color }));
	}
	_drawXHist(fr, opts.xHist);
	_drawVlines(fr, opts.vlines);
	_drawBands(fr, opts.bands, opts.selBand, opts.onBand);
	svgCorner(fr, opts.corrLabel);
	const wrap = el("div", { class: "gwrap" }, fr.svg);
	// Hover reads the meaningful (non-faint) curves; label each by matching the
	// legend swatch color the caller already assigned.
	const solid = series.filter((s) => !s.faint && (s.points || []).length);
	if (solid.length) {
		const base = solid.reduce((a, b) => (b.points.length > a.points.length ? b : a), solid[0]);
		const bx = base.points.map((p) => p[0]);
		const sampleAt = (pts, xv) => { let bi = 0, bd = Infinity; for (let i = 0; i < pts.length; i++) { const dd = Math.abs(pts[i][0] - xv); if (dd < bd) { bd = dd; bi = i; } } return pts[bi][1]; };
		const legLabel = (c) => { const it = (opts.legend || []).find((L) => L && L.color === c); return it ? it.label : "value"; };
		const entries = solid.map((s) => ({ label: legLabel(s.color), color: s.color, values: bx.map((xv) => sampleAt(s.points, xv)) }));
		chartHover(wrap, fr, bx, entries, { xFmt: opts.xFmt, vFmt: opts.yFmt });
	}
	if (opts.legend) wrap.appendChild(chartLegend(opts.legend));
	return wrap;
}

// Generic scatter with an optional y=x reference line and a corner correlation
// label. points: { x, y, color?, r? }.
function scatterChart(points, opts = {}) {
	const { el, niceMax } = d();
	const padL = 46, padR = 14, padT = 10, padB = 26;
	// `square` gives a square PLOT area (equal visual scale on both axes) — right
	// when x and y are the same kind of quantity (e.g. two rankings). W/H are
	// picked so (W-padX) === (H-padY); a max-width on the wrap keeps it compact.
	const sq = !!opts.square;
	const W = sq ? 660 : 960;
	const H = sq ? 636 : Math.round((opts.height || 145) * CHART_H);
	if (!points.length) return el("div", { class: "hint", text: "no data to plot" });
	const nice = (v) => (niceMax ? niceMax(v) : v * 1.05);
	const xMax = opts.xMax != null ? opts.xMax : nice(Math.max(...points.map((p) => p.x), 1e-9));
	const rawYMax = Math.max(...points.map((p) => p.y), 1e-9);
	const yMax = opts.yMax != null ? opts.yMax : (opts.logY ? rawYMax * 1.15 : nice(rawYMax));
	const fr = svgFrame(W, H, { padL, padR, padT, padB, xMin: 0, xMax, yMax, yFmt: opts.yFmt, xFmt: opts.xFmt, xLabel: opts.xLabel, logY: opts.logY });
	if (opts.refLine) {
		const m = Math.min(xMax, yMax);
		fr.svg.appendChild(svgEl("line", { x1: fr.X(0), y1: fr.Y(0), x2: fr.X(m), y2: fr.Y(m), stroke: "rgba(255,255,255,0.25)", "stroke-width": 1, "stroke-dasharray": "4 3" }));
	}
	const xf = opts.xFmt || ((v) => v.toFixed(2)), yf = opts.yFmt || ((v) => v.toFixed(2));
	for (const p of points) {
		const dot = svgEl("circle", { class: "gpt", cx: fr.X(p.x).toFixed(2), cy: fr.Y(p.y).toFixed(2), r: p.r || 3.2, fill: p.color || "#7aa2f7", "fill-opacity": 0.85 });
		dot.appendChild(svgEl("title", null, `${p.label ? p.label + "  " : ""}(${xf(p.x)}, ${yf(p.y)})`));
		fr.svg.appendChild(dot);
	}
	svgCorner(fr, opts.corrLabel);
	const wrap = el("div", { class: `gwrap${sq ? " square" : ""}` }, fr.svg);
	if (opts.legend) wrap.appendChild(chartLegend(opts.legend));
	return wrap;
}

// Joint scatter with MARGINAL distributions (a jointplot): a square main scatter,
// colored by CATEGORY, with per-category smoothed histograms along the TOP (x) and
// RIGHT (y) axes + a legend. `points`: [{ x, y, cat, label, r? }]; `cats`:
// [{ key, color, label }] in draw/legend order. The marginals let you read each
// group's x / y distribution at a glance (e.g. "do high-attention objects skew to
// low distance rank?").
function jointScatter(points, opts = {}) {
	const { el } = d();
	if (!points.length) return el("div", { class: "hint", text: "no data to plot" });
	const cats = (opts.cats && opts.cats.length) ? opts.cats : [{ key: "", color: "#7aa2f7", label: "" }];
	const colorOf = new Map(cats.map((c) => [c.key, c.color]));
	const W = 578, H = 560, m = 76, gap = 6, padL = 46, padR = 12, padT = 10, padB = 30;
	const mainL = padL, mainR = W - padR - m - gap, mainT = padT + m + gap, mainB = H - padB; // square main plot
	const xMax = opts.xMax || Math.max(...points.map((p) => p.x), 1);
	const yMax = opts.yMax || Math.max(...points.map((p) => p.y), 1);
	const X = (x) => mainL + (x / (xMax || 1)) * (mainR - mainL);
	const Y = (y) => mainB - (y / (yMax || 1)) * (mainB - mainT);
	const xf = opts.xFmt || ((v) => String(Math.round(v))), yf = opts.yFmt || ((v) => String(Math.round(v)));
	const svg = svgEl("svg", { class: "gsvg", viewBox: `0 0 ${W} ${H}`, style: "width:100%;height:auto;display:block;overflow:visible" });
	for (const f of [0, 0.25, 0.5, 0.75, 1]) {
		const gy = mainB - f * (mainB - mainT), gx = mainL + f * (mainR - mainL);
		svg.appendChild(svgEl("line", { x1: mainL, y1: gy.toFixed(1), x2: mainR, y2: gy.toFixed(1), stroke: "rgba(255,255,255,0.06)" }));
		svg.appendChild(svgEl("line", { x1: gx.toFixed(1), y1: mainT, x2: gx.toFixed(1), y2: mainB, stroke: "rgba(255,255,255,0.06)" }));
		svg.appendChild(svgEl("text", { x: gx.toFixed(1), y: mainB + 14, fill: "rgba(220,230,245,0.5)", "font-size": 10, "text-anchor": "middle" }, xf(f * xMax)));
		svg.appendChild(svgEl("text", { x: mainL - 6, y: gy.toFixed(1), fill: "rgba(220,230,245,0.5)", "font-size": 10, "text-anchor": "end", "dominant-baseline": "middle" }, yf(f * yMax)));
	}
	if (opts.xLabel) svg.appendChild(svgEl("text", { x: ((mainL + mainR) / 2).toFixed(1), y: H - 4, fill: "rgba(220,230,245,0.82)", "font-size": 11, "text-anchor": "middle" }, opts.xLabel));
	if (opts.yLabel) { const my = ((mainT + mainB) / 2).toFixed(1); svg.appendChild(svgEl("text", { x: 12, y: my, fill: "rgba(220,230,245,0.82)", "font-size": 11, "text-anchor": "middle", transform: `rotate(-90 12 ${my})` }, opts.yLabel)); }
	// per-category smoothed marginal histograms (top = x, right = y)
	const nb = Math.min(Math.max(6, Math.round(Math.max(xMax, yMax))), 24);
	const binOf = (vals, mx) => { const b = new Array(nb).fill(0); for (const v of vals) b[Math.min(nb - 1, Math.max(0, Math.floor((v / (mx || 1)) * nb)))] += 1; return b; };
	const smooth = (b) => b.map((_, i) => (b[i - 1] || 0) * 0.25 + b[i] * 0.5 + (b[i + 1] || 0) * 0.25);
	const topB = cats.map((c) => smooth(binOf(points.filter((p) => p.cat === c.key).map((p) => p.x), xMax)));
	const topMax = Math.max(1e-9, ...topB.flat());
	cats.forEach((c, ci) => {
		let dd = `M ${mainL.toFixed(1)} ${(padT + m).toFixed(1)}`;
		topB[ci].forEach((v, i) => { dd += ` L ${(mainL + ((i + 0.5) / nb) * (mainR - mainL)).toFixed(1)} ${((padT + m) - (v / topMax) * (m - 3)).toFixed(1)}`; });
		dd += ` L ${mainR.toFixed(1)} ${(padT + m).toFixed(1)} Z`;
		svg.appendChild(svgEl("path", { d: dd, fill: hexA(c.color, 0.32), stroke: c.color, "stroke-width": 1 }));
	});
	const rgtB = cats.map((c) => smooth(binOf(points.filter((p) => p.cat === c.key).map((p) => p.y), yMax)));
	const rgtMax = Math.max(1e-9, ...rgtB.flat()), rx0 = mainR + gap;
	cats.forEach((c, ci) => {
		let dd = `M ${rx0.toFixed(1)} ${mainB.toFixed(1)}`;
		rgtB[ci].forEach((v, i) => { dd += ` L ${(rx0 + (v / rgtMax) * (m - 3)).toFixed(1)} ${(mainB - ((i + 0.5) / nb) * (mainB - mainT)).toFixed(1)}`; });
		dd += ` L ${rx0.toFixed(1)} ${mainT.toFixed(1)} Z`;
		svg.appendChild(svgEl("path", { d: dd, fill: hexA(c.color, 0.32), stroke: c.color, "stroke-width": 1 }));
	});
	for (const p of points) {
		const dot = svgEl("circle", { class: "gpt", cx: X(p.x).toFixed(2), cy: Y(p.y).toFixed(2), r: p.r || 3.4, fill: colorOf.get(p.cat) || "#7aa2f7", "fill-opacity": p.o ?? 0.68 });
		dot.appendChild(svgEl("title", null, p.label || `(${xf(p.x)}, ${yf(p.y)})`));
		svg.appendChild(dot);
	}
	if (opts.corrLabel) svg.appendChild(svgEl("text", { x: (mainR - 2).toFixed(1), y: mainT + 2, fill: "rgba(220,230,245,0.9)", "font-size": 11, "text-anchor": "end", "dominant-baseline": "hanging" }, opts.corrLabel));
	const wrap = el("div", { class: "gwrap square" }, svg);
	const legendItems = cats.filter((c) => c.label).map((c) => el("span", { class: "pin-pill" },
		el("i", { style: `width:11px;height:11px;border-radius:50%;background:${c.color};display:inline-block;flex:none` }), el("span", { text: c.label })));
	if (opts.sizeLabel) legendItems.push(el("span", { class: "muted", style: "font-size:11px;align-self:center", text: opts.sizeLabel }));
	wrap.appendChild(el("div", { class: "attr-legend compact", style: "justify-content:center;margin-top:6px", }, ...legendItems));
	return wrap;
}

// --- per-step accessors ------------------------------------------------------

// A step's per-token scene mass (+ to-place mass). Positions are scored-token
// ranks (0…n−1) — one equal slot per compact token, matching summaryTrajectory.
function stepTrajectory(a) {
	const toks = (a && a.tokens) || [];
	if (!toks.length) return null;
	const n = toks.length;
	const span = Math.max(1, n - 1);
	const mass = (a.agg && a.agg.mass) || [];
	const tp = (a.agg && a.agg.to_place && a.agg.to_place.mass) || null;
	const os = d().outputStartTok;
	const outStart = a.out_start ?? (os ? os(a) : toks.findIndex((t) => t.output_entity != null));
	return { toks, rel: toks.map((_, i) => i), span, mass, tp, outStart };
}

// "Output zoom": a sub-trajectory of just the output tokens (reasoning trimmed), so
// the per-attribute segmentation fills the width instead of squishing on the right.
// rel is re-based to 0…nOut−1 and outStart→0; `base` records the original offset so
// vline token indices can be mapped in. No-op when there is no output region.
function outputZoomTrajectory(tr) {
	if (!tr) return tr;
	const s = Math.max(0, Math.min(tr.toks.length, tr.outStart | 0));
	if (s <= 0 || s >= tr.toks.length) return tr;
	const toks = tr.toks.slice(s);
	return { toks, rel: toks.map((_, i) => i), span: Math.max(1, toks.length - 1), mass: (tr.mass || []).slice(s), tp: tr.tp ? tr.tp.slice(s) : null, outStart: 0, zoomed: true, base: s };
}

// The to-place objects of an object_bbox step joined to their emission order.
// input rank = order in the scene context (to_place_entities by first token);
// output rank = emission order (agg.outputs is already first-emitted order).
function toPlaceOrdering(a) {
	const { hasToPlace } = d();
	if (!hasToPlace(a)) return null;
	const tpe = a.to_place_entities || [];
	if (!tpe.length) return null;
	const inOrder = [...tpe].sort((x, y) => ((x.token_span && x.token_span[0]) || 0) - ((y.token_span && y.token_span[0]) || 0));
	const inRank = new Map(inOrder.map((e, i) => [e.id, i]));
	const objs = [];
	for (const o of (a.agg && a.agg.outputs) || []) {
		if (!inRank.has(o.entity)) continue; // keep only the to-place objects
		objs.push({ id: o.entity, outRank: objs.length, inRank: inRank.get(o.entity),
			sceneMass: (o.scene && o.scene.mass) || 0, tpMass: o.to_place ? (o.to_place.mass || 0) : null });
	}
	return objs.length ? { objs, n: inOrder.length } : null;
}

// REUSABLE output-feature → token mapping. A step's OUTPUT is a sequence of items
// it emits — objects for a bbox step, zones for a plan step, … — and each output
// token carries its item id in `output_entity`; `agg.outputs` holds the precomputed
// per-item attention rollup in first-emitted order. Grouping tokens by that id gives
// every feature its token span, so ANY per-feature graph can share this basis. It
// customizes per step kind for free (the ids ARE whatever that kind emitted).
// Returns [{ id, label, tokens:[idx…], n, sceneMass, tpMass|null, entityTotals }].
function outputFeatures(a) {
	const toks = (a && a.tokens) || [];
	const mass = (a && a.agg && a.agg.mass) || [];
	const tpArr = (a && a.agg && a.agg.to_place && a.agg.to_place.mass) || null;
	const byId = new Map();
	toks.forEach((t, i) => { const e = t && t.output_entity; if (e == null) return; if (!byId.has(e)) byId.set(e, []); byId.get(e).push(i); });
	return ((a && a.agg && a.agg.outputs) || []).map((o) => {
		const tokens = byId.get(o.entity) || [];
		// Per-item mean ± sd of scene / to-place mass ACROSS the item's tokens — the
		// spread that gives the per-feature histogram a rich, uncertainty-aware y.
		const sv = tokens.map((i) => mass[i] || 0), sp = sv.length ? pm(sv) : null;
		const tv = tpArr ? tokens.map((i) => tpArr[i] || 0) : null, tp = tv && tv.length ? pm(tv) : null;
		return {
			id: o.entity, label: o.entity, tokens, n: o.n ?? tokens.length,
			sceneMass: sp ? sp.m : (o.scene && o.scene.mass) || 0, sceneSd: sp ? sp.s : 0,
			tpMass: o.to_place ? (tp ? tp.m : o.to_place.mass || 0) : null, tpSd: tp ? tp.s : 0,
			entityTotals: (o.scene && o.scene.entityTotals) || [],
			componentTotals: (o.scene && o.scene.componentTotals) || [],
		};
	});
}

// Longest shared prefix (to a separator) across a set of ids — used to shorten
// grouped labels (e.g. show `bench_west_2` under a `south_walkway` group) without
// repeating the shared part on every one.
function _commonPrefix(labels) {
	if (!labels || labels.length < 2) return "";
	let p = labels[0];
	for (const l of labels) { while (p && !l.startsWith(p)) p = p.slice(0, -1); if (!p) break; }
	const cut = Math.max(p.lastIndexOf("_"), p.lastIndexOf("-"), p.lastIndexOf(".")) + 1;
	return cut >= 3 ? p.slice(0, cut) : "";
}

// Segment a step's OUTPUT token stream into items and, WITHIN each item, the JSON
// attribute fields it is emitting — so the by-n trajectory can be sliced by "which
// item / which attribute is being written here". Item boundaries come from runs of
// `output_entity`; attribute boundaries are found by reconstructing the emitted text
// from per-token `text` and locating each `"field":` key (mapped back to its token).
// Frontend-only: no per-token field label is stored server-side. Returns token-index
// markers { items:[{i0,label}], fields:[{i,label}] } (indices into a.tokens).
function outputSegments(a) {
	const toks = (a && a.tokens) || [];
	let outStart = a && a.out_start != null ? a.out_start : toks.findIndex((t) => t && t.output_entity != null);
	if (outStart == null || outStart < 0) return null;
	const items = [];
	for (let i = outStart; i < toks.length; i++) {
		const e = toks[i].output_entity;
		if (e == null) continue;
		if (!items.length || items[items.length - 1].label !== e) items.push({ i0: i, label: e });
	}
	// Reconstruct the emitted text (output tokens) + a char-offset → token index map,
	// then find every `"key":` field start and attribute it to the token it begins in.
	let text = ""; const offTok = [];
	for (let i = outStart; i < toks.length; i++) { const t = toks[i].text || ""; for (let c = 0; c < t.length; c++) offTok.push(i); text += t; }
	const fields = []; const re = /"([A-Za-z_][\w]*)"\s*:/g;
	let m; while ((m = re.exec(text))) { const tok = offTok[m.index]; if (tok != null) fields.push({ i: tok, label: m[1] }); }
	return { items, fields, outStart };
}

// A token fraction → progression x (matches _progXs, used by the stacked breakdowns).
function _progX(f, meanNq, mode) {
	if (mode === "ratio") return f;
	if (mode === "log") return Math.log1p(f * meanNq);
	return f * meanNq;
}
// Build the item + attribute-field vertical markers for a single step, ONCE, shared
// by every within-step breakdown graph so the same boundaries line up everywhere.
// Two x-mappings are returned because the graphs live in two x-spaces: the scene-mass
// LINE uses xformTrajPos (phase-aware), the progression STACKS use _progXs
// (fraction→meanNq). Same boundaries, mapped so they align on each. → { traj, prog, note }.
function _outputVlines(a, mode, zoom) {
	const seg = outputSegments(a);
	if (!seg || (!seg.items.length && !seg.fields.length)) return null;
	let tr = stepTrajectory(a);
	if (!tr) return null;
	// In output-zoom the axis is re-based to the output tokens only, so shift each
	// marker's token index by the trimmed reasoning prefix (tr.base) before mapping.
	if (zoom) tr = outputZoomTrajectory(tr);
	const base = tr.base || 0;
	const span = Math.max(1, tr.toks.length - 1), meanNq = tr.toks.length; // n_query ≈ scored-token count
	const preI = _commonPrefix(seg.items.map((it) => it.label));
	const traj = [], prog = [];
	const push = (i0, label, major) => {
		const i = i0 - base;
		if (i < 0 || i > span) return;
		traj.push({ x: xformTrajPos(i, tr, mode), label, major });
		prog.push({ x: _progX(i / span, meanNq, mode), label, major });
	};
	for (const it of seg.items) push(it.i0, preI && it.label.length > preI.length ? it.label.slice(preI.length) : it.label, true);
	for (const f of seg.fields) push(f.i, f.label, false);
	traj.sort((p, q) => p.x - q.x); prog.sort((p, q) => p.x - q.x);
	const note = `output segmented: ${seg.items.length} items · ${seg.fields.length} attribute fields${preI ? ` · group ${preI.replace(/[_.-]$/, "")}` : ""}`;
	return { traj, prog, note };
}

// Output ITEM bands (contiguous token span per emitted item) mapped into BOTH chart
// x-spaces, so clicking a band on the scene-mass line (traj space) can highlight the
// same span on the progression stacks (prog space). id is the full entity id (for
// selection + breakdown lookup); label is the prefix-shortened display name.
function _outputBands(a, mode, zoom) {
	const seg = outputSegments(a);
	if (!seg || !seg.items.length) return [];
	let tr = stepTrajectory(a);
	if (!tr) return [];
	if (zoom) tr = outputZoomTrajectory(tr);
	const base = tr.base || 0, n = tr.toks.length, span = Math.max(1, n - 1);
	const preI = _commonPrefix(seg.items.map((it) => it.label));
	const out = [];
	for (let k = 0; k < seg.items.length; k++) {
		const s = seg.items[k].i0 - base;
		if (s < 0 || s > span) continue;
		const eTok = k + 1 < seg.items.length ? Math.min(span, seg.items[k + 1].i0 - base) : span;
		const e = Math.max(s, eTok);
		const label = preI && seg.items[k].label.length > preI.length ? seg.items[k].label.slice(preI.length) : seg.items[k].label;
		out.push({ id: seg.items[k].label, label, x0: xformTrajPos(s, tr, mode), x1: xformTrajPos(e, tr, mode), px0: _progX(s / span, n, mode), px1: _progX(e / span, n, mode) });
	}
	return out;
}

// --- reasoning-text token types (the generated tokens' OWN class, not attention) --
// The server classifies key tokens for the attention `type` buckets but never keeps
// the query token's own class, so we recompute it here from a.tokens[i].text. This
// MIRRORS semantic.classify_tokens (keep the word lists in sync). `entity_name` is
// dropped — it needs scene NAME spans that don't cover generated tokens.
const _Q_NUM_RE = /^[-+]?\d+(?:\.\d+)?$/;
const _Q_BRACKET = new Set("{}[]()"), _Q_SEP = new Set(":,"), _Q_QUOTE = new Set("\"'`"), _Q_OP = new Set("=");
const _Q_STRUCT = new Set([..._Q_BRACKET, ..._Q_SEP, ..._Q_QUOTE, ..._Q_OP]);
const _Q_SPATIAL = new Set(["above", "below", "beside", "under", "over", "on", "in", "inside", "into", "onto", "beneath", "behind", "front", "atop", "upon", "adjacent", "near", "against", "top", "bottom", "left", "right", "attached", "between", "around", "within", "outside", "up", "down", "aligned"]);
const _Q_FUNCTION = new Set(["the", "a", "an", "this", "that", "these", "those", "of", "to", "for", "with", "from", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", "will", "would", "should", "shall", "can", "could", "must", "may", "might", "it", "its", "they", "them", "their", "which", "as", "at", "by", "if", "then", "so", "not", "no", "than", "each", "per", "we", "you", "i", "he", "she", "there", "here", "also", "have", "has", "had", "do", "does", "did"]);
// Stack order (matches the word-type legend families) so the strip reads consistently.
const _Q_TYPE_ORDER = ["number", "spatial", "content", "function", "entity_name", "bracket", "separator", "quote", "operator", "other", "whitespace"];
function _classifyText(tx) {
	const s = (tx || "").trim();
	if (s === "") return "whitespace";
	if (_Q_NUM_RE.test(s)) return "number";
	if ([...s].every((c) => _Q_STRUCT.has(c))) { const c = s[0]; return _Q_BRACKET.has(c) ? "bracket" : _Q_SEP.has(c) ? "separator" : _Q_QUOTE.has(c) ? "quote" : "operator"; }
	const low = s.toLowerCase();
	if (_Q_SPATIAL.has(low)) return "spatial";
	if (_Q_FUNCTION.has(low)) return "function";
	if (/[a-z]/i.test(s)) return "content";
	return "other";
}
// Per-x-bin token-type mix of the REASONING text (tokens [0, outStart)), for the
// mini histogram drawn in the x-axis gutter. Whitespace is dropped (pure filler that
// would otherwise dominate). Bins are laid out in the current x-mode so they line up
// with the plot above. Returns { x0, x1, nb, present:[type…], bins:[{parts:[{t,frac}]}] }.
function _reasoningTypeHist(a, mode, zoom, nbins = 24) {
	if (zoom) return null;   // output-zoom trims the reasoning region away entirely
	const tr = stepTrajectory(a);
	if (!tr) return null;
	const out = tr.outStart | 0;
	if (!(out > 1)) return null;
	const x0 = xformTrajPos(0, tr, mode), x1 = xformTrajPos(out, tr, mode);
	if (!(x1 > x0)) return null;
	const nb = Math.max(4, Math.min(nbins, out));
	const counts = Array.from({ length: nb }, () => ({}));
	const present = new Set();
	for (let i = 0; i < out; i++) {
		const t = _classifyText(tr.toks[i] && tr.toks[i].text);
		if (t === "whitespace") continue;
		const bi = Math.max(0, Math.min(nb - 1, Math.floor(((xformTrajPos(i, tr, mode) - x0) / (x1 - x0)) * nb)));
		counts[bi][t] = (counts[bi][t] || 0) + 1; present.add(t);
	}
	if (!present.size) return null;
	const bins = counts.map((c) => {
		const total = Object.values(c).reduce((s, v) => s + v, 0) || 1;
		return { parts: _Q_TYPE_ORDER.filter((t) => c[t]).map((t) => ({ t, frac: c[t] / total })) };
	});
	return { x0, x1, nb, bins, present: _Q_TYPE_ORDER.filter((t) => present.has(t)) };
}

// Resample a step's (ratio, value) polyline onto a shared [0..1] grid so steps
// of different lengths average vertex-to-vertex.
function resampleByRatio(ratios, values, grid) {
	return grid.map((g) => {
		if (g <= ratios[0]) return values[0];
		if (g >= ratios[ratios.length - 1]) return values[values.length - 1];
		let hi = 1;
		while (hi < ratios.length && ratios[hi] < g) hi++;
		const lo = hi - 1, t = (g - ratios[lo]) / ((ratios[hi] - ratios[lo]) || 1);
		return values[lo] + t * (values[hi] - values[lo]);
	});
}

const TRAJ_GRID = 32;
function aggregateTrajectory(rows) {
	const grid = Array.from({ length: TRAJ_GRID }, (_, i) => i / (TRAJ_GRID - 1));
	const perStep = [], spans = [];
	for (const r of rows) {
		const tr = stepTrajectory(r.a);
		if (!tr || tr.span <= 0) continue;
		const ratios = tr.rel.map((x) => x / tr.span);
		perStep.push({ template: r.template, vals: resampleByRatio(ratios, tr.mass, grid),
			outRatio: tr.outStart < tr.toks.length ? tr.outStart / tr.span : 1 });
		spans.push(tr.span);
	}
	if (!perStep.length) return null;
	spans.sort((a, b) => a - b);
	return {
		grid, perStep, n: perStep.length,
		mean: grid.map((_, k) => mean(perStep.map((s) => s.vals[k]))),
		sd: grid.map((_, k) => std(perStep.map((s) => s.vals[k]))),
		medianSpan: spans[Math.floor(spans.length / 2)] || 1,
		outRatio: mean(perStep.map((s) => s.outRatio)),
	};
}

function median(xs) { const a = [...xs].sort((p, q) => p - q); return a.length ? a[Math.floor(a.length / 2)] : 0; }

// Exponential moving average along a series (left→right = token order). Span
// controls smoothness (~number of tokens in the effective window).
function ema(values, span = 5) {
	if (!values.length) return [];
	const alpha = 2 / (Math.max(2, span) + 1);
	const out = [values[0]];
	for (let i = 1; i < values.length; i++) out.push(alpha * values[i] + (1 - alpha) * out[i - 1]);
	return out;
}
function emaSpan(n) { return Math.max(3, Math.min(12, Math.round(n / 6))); }
function emaLineSeries(points, color, span = null) {
	if (!points || points.length < 3) return null;
	span = span ?? emaSpan(points.length);
	const ys = ema(points.map((p) => p[1]), span);
	return { points: points.map((p, i) => [p[0], ys[i]]), color, width: 2.2, dash: [7, 4] };
}

const TREND_ARROW = { increasing: "↗", decreasing: "↘", flat: "→" };

function trajectoryXY(tr, mode) {
	return {
		xs: tr.rel.map((r) => xformTrajPos(r, tr, mode)),
		ys: tr.mass,
	};
}

// Per-step Spearman + OLS on (token position, scene mass). Pooled across steps
// the mean ρ ± sd quantifies whether scene mass rises or falls along the sequence.
function sceneMassTrendSteps(rows, mode) {
	const per = [];
	for (const r of rows) {
		const tr = stepTrajectory(r.a);
		if (!tr || tr.mass.length < 3) continue;
		const { xs, ys } = trajectoryXY(tr, mode);
		per.push({ sp: spearmanTrend(xs, ys), ols: olsTrend(xs, ys) });
	}
	return per;
}

function sceneMassTrendOnSeries(xs, ys) {
	if (!xs?.length || xs.length < 3) return null;
	return { sp: spearmanTrend(xs, ys), ols: olsTrend(xs, ys) };
}

function fmtP(p) {
	if (!isFinite(p)) return "–";
	if (p < 0.001) return "p<0.001";
	if (p < 0.01) return `p=${p.toFixed(3)}`;
	return `p=${p.toFixed(2)}`;
}

function fmtSlopePerStep(slope, se, mode) {
	// Express slope as Δ scene-mass per normalized step position (0→1 span).
	if (mode === "ratio") return `${fmtPM(slope, se)} mass / step`;
	if (mode === "log") return `${fmtPM(slope, se)} mass / log-token`;
	return `${fmtPM(slope, se)} mass / token`;
}

function sceneMassTrendSummary(perStep, aggTrend) {
	if (!perStep.length && !aggTrend) return null;
	const rhos = perStep.map((s) => s.sp.rho);
	const rp = rhos.length ? pm(rhos) : null;
	const sig = perStep.filter((s) => s.sp.p < 0.05).length;
	let dir = "flat";
	if (perStep.length === 1) dir = perStep[0].sp.dir;
	else if (rp && Math.abs(rp.m) >= 0.04) dir = rp.m > 0 ? "increasing" : "decreasing";
	else if (aggTrend) dir = aggTrend.sp.dir;
	return { perStep, rp, sig, nSteps: perStep.length, aggTrend, dir };
}

function sceneMassTrendCorrLabel(summary) {
	if (!summary) return "";
	const { rp, sig, nSteps, aggTrend, dir } = summary;
	const arrow = TREND_ARROW[dir] || "→";
	if (nSteps <= 1 && rp) {
		const s = summary.perStep[0];
		return `${arrow} ${dir} · ρ=${rp.m.toFixed(2)} (${fmtP(s.sp.p)})`;
	}
	const parts = [`${arrow} ${dir}`];
	if (rp) parts.push(`ρ=${fmtPM(rp.m, rp.s)} · ${nSteps} steps`);
	if (sig && nSteps > 1) parts.push(`${sig}/${nSteps} sig`);
	if (aggTrend) parts.push(`curve ρ=${aggTrend.sp.rho.toFixed(2)} (${fmtP(aggTrend.sp.p)})`);
	return parts.join(" · ");
}

function sceneMassTrendNote(summary, mode) {
	const { el } = d();
	if (!summary) return null;
	const { perStep, rp, sig, nSteps, aggTrend, dir } = summary;
	const arrow = TREND_ARROW[dir] || "→";
	const lines = [];
	if (nSteps <= 1 && perStep[0]) {
		const s = perStep[0];
		lines.push(el("span", {},
			el("b", { text: `${arrow} ${dir} ` }),
			el("span", { text: `Spearman ρ=${s.sp.rho.toFixed(2)} (${fmtP(s.sp.p)}) · OLS slope ${fmtSlopePerStep(s.ols.slope, s.ols.se, mode)} (${fmtP(s.ols.p)})` })));
	} else {
		if (rp) {
			lines.push(el("span", {},
				el("b", { text: `${arrow} ${dir} ` }),
				el("span", { text: `mean Spearman ρ=${fmtPM(rp.m, rp.s)} across ${nSteps} steps` }),
				sig ? el("span", { text: ` · significant (p<0.05) in ${sig}/${nSteps} steps` }) : el("span", { text: ` · not significant in most steps` })));
		}
		if (aggTrend) {
			lines.push(el("span", {},
				el("b", { text: "aggregate curve " }),
				el("span", { text: `ρ=${aggTrend.sp.rho.toFixed(2)} (${fmtP(aggTrend.sp.p)}) · slope ${fmtSlopePerStep(aggTrend.ols.slope, aggTrend.ols.se, mode)} (${fmtP(aggTrend.ols.p)})` })));
		}
	}
	return el("div", { class: "hint trend-note", style: "display:flex;flex-direction:column;gap:4px;margin-top:6px" }, ...lines);
}

function trajAtNormalized(g, span, outRatio, mode) {
	const outStart = Math.min(span, Math.round(outRatio * span));
	return xformTrajPos(g * span, { toks: { length: span + 1 }, span, outStart }, mode);
}

// --- (A) scene mass vs token position ---------------------------------------
// "attention of the scene by tokens vs output tokens": scene mass along the
// generated sequence, output region shaded, uncertainty band toggled by ± bars.
function sceneMassVsPositionCard(rows, { title = "scene mass vs token position", vlines = null, segNote = null, zoom = false, bands = null, selBand = null, onBand = null, xHist = null } = {}) {
	const { el, reportCard, reportEmpty, COLORS } = d();
	const mode = curX(), errOn = !d().showErr || d().showErr();
	const tpColor = (COLORS && COLORS.to_place) || "#e0a94a";
	if (rows.length === 1) {
		const a = rows[0].a, tr0 = stepTrajectory(a);
		if (!tr0) return reportCard(title, null, reportEmpty("no tokens in this step"));
		const tr = zoom ? outputZoomTrajectory(tr0) : tr0, base = tr.base || 0;
		const { xs, ys } = trajectoryXY(tr, mode);
		const sd = errOn ? (a.tokens || []).slice(base).map((t) => ((t.hscale || []).length > 1 ? std(t.hscale) : 0)) : null;
		const massPts = xs.map((x, i) => [x, tr.mass[i]]);
		const series = [{
			points: massPts, color: "#4af0e0", area: true,
			band: sd ? { hi: tr.mass.map((m, i) => Math.min(1, m + sd[i])), lo: tr.mass.map((m, i) => Math.max(0, m - sd[i])) } : null,
		}];
		if (tr.tp) series.push({ points: xs.map((x, i) => [x, tr.tp[i]]), color: tpColor, dash: [4, 3] });
		const massEma = emaLineSeries(massPts, "#e8fcff");
		if (massEma) series.push(massEma);
		// No output shade when zoomed — the whole width IS the output region.
		const shadeFrom = (!tr.zoomed && tr.outStart < tr.toks.length) ? xformTrajPos(tr.outStart, tr, mode) : null;
		const trend = sceneMassTrendSummary(sceneMassTrendSteps(rows, mode), sceneMassTrendOnSeries(xs, ys));
		// Output segmentation (item + attribute-field boundaries) is computed once by the
		// caller and shared across every breakdown graph so the lines align everywhere.
		const chart = lineChart(series, {
			height: vlines ? 150 : 118, xLabel: (tr.zoomed ? "output " : "") + xLabelFor(mode), xFmt: xFmtFor(mode), yFmt: pctFmt,
			shade: shadeFrom != null ? { from: shadeFrom, to: Math.max(...xs) } : null,
			corrLabel: sceneMassTrendCorrLabel(trend), vlines, bands, selBand, onBand, xHist,
			legend: [{ label: "scene mass", color: "#4af0e0" }, massEma ? { label: "EMA", color: "#e8fcff" } : null, tr.tp ? { label: "to-place mass", color: tpColor } : null, shadeFrom != null ? { label: "output region", color: "#7a4fd0" } : null],
		});
		const clickHint = (onBand && bands && bands.length) ? (selBand ? " · click the item again to clear" : " · click an item for its breakdown") : "";
		const histNote = xHist ? " · x-axis strip = reasoning token-type mix" : "";
		const sub = (segNote || (tr.zoomed ? "y = scene mass · band = ±sd across heads · output region only" : "y = scene mass · band = ±sd across heads · shaded = output tokens")) + clickHint + histNote;
		const histLegend = xHist ? el("div", { style: "display:flex;gap:10px;flex-wrap:wrap;font-size:10px;margin-top:4px;opacity:.85" },
			el("span", { style: "opacity:.5", text: "reasoning types:" }),
			...xHist.present.map((t) => el("span", { style: "display:inline-flex;align-items:center;gap:4px;cursor:help", title: TYPE_DESC[t] || t },
				el("span", { class: "frame-sw", style: `background:${TYPE_STYLE[t] || "#8a8f98"}` }), el("span", { text: t })))) : null;
		return reportCard(tr.zoomed ? `${title} · output` : title, sub, chart, histLegend, sceneMassTrendNote(trend, mode), reasoningOutputSplit(tr0));
	}
	const agg = aggregateTrajectory(rows);
	if (!agg) return reportCard(title, null, reportEmpty("no multi-token steps to aggregate"));
	const xAt = (g) => trajAtNormalized(g, agg.medianSpan, agg.outRatio, mode);
	const perStep = sceneMassTrendSteps(rows, mode);
	const aggXs = agg.grid.map(xAt);
	const trend = sceneMassTrendSummary(perStep, sceneMassTrendOnSeries(aggXs, agg.mean));
	const series = agg.perStep.map((s) => ({ points: agg.grid.map((g, k) => [xAt(g), s.vals[k]]), color: templateColorOf(s.template, rows), faint: true }));
	const meanPts = agg.grid.map((g, k) => [xAt(g), agg.mean[k]]);
	series.push({
		points: meanPts, color: "#7aa2f7", width: 2,
		band: errOn ? { hi: agg.mean.map((m, i) => m + agg.sd[i]), lo: agg.mean.map((m, i) => Math.max(0, m - agg.sd[i])) } : null,
	});
	const meanEma = emaLineSeries(meanPts, "#e8eefc");
	if (meanEma) series.push(meanEma);
	const chart = lineChart(series, {
		height: 124, xLabel: xLabelFor(mode), xFmt: xFmtFor(mode), yFmt: pctFmt,
		shade: { from: xAt(agg.outRatio), to: xAt(1) },
		corrLabel: sceneMassTrendCorrLabel(trend),
		legend: [{ label: `mean of ${agg.n} steps`, color: "#7aa2f7" }, meanEma ? { label: "EMA", color: "#e8eefc" } : null, { label: "per-step", color: hexA("#7aa2f7", 0.4) }, { label: "output region", color: "#7a4fd0" }],
	});
	return reportCard(title, `mean ± sd across ${agg.n} steps · resampled on a shared position grid`, chart, sceneMassTrendNote(trend, mode));
}

// Grouped-prefix layout for a crowded categorical axis. Object ids are flat
// names whose leading segments repeat (e.g. `kitchen_counter_stool_1`,
// `kitchen_counter_stool_2`, `kitchen_island`). Rather than repeating the long
// shared prefix on every tick, we sort items so shared-prefix runs are
// contiguous and draw the prefix ONCE as a bracket under the axis. Nesting is
// capped at `maxDepth` (default 2): an outer bracket for the 1st id segment and
// an inner one for the 2nd. The per-item label is only the remaining suffix.
//
// Returns { order, labels, groupLevels, grouped, sep }:
//   order       display order (permutation of input indices)
//   labels      per-display-position suffix label (prefix stripped)
//   groupLevels bracket levels, FINEST FIRST: [[{label,i0,i1}...], ...] (≤ maxDepth)
// The separator is auto-detected across the ids so path-like ("/", ".") and
// underscore/hyphen name conventions all work.
function _prefixLayout(ids, maxDepth = 2) {
	const n = ids.length;
	const identity = { order: ids.map((_, i) => i), labels: ids.slice(), groupLevels: [], grouped: false, sep: "" };
	if (n < 2) return identity;
	const SEPS = ["/", ".", ":", ">", "|", "_", "-"];
	let sep = "", best = 0;
	for (const s of SEPS) { const c = ids.reduce((k, id) => k + (String(id).split(s).length > 1 ? 1 : 0), 0); if (c > best) { best = c; sep = s; } }
	if (!sep) return identity; // no shared separator → nothing to group
	const segs = ids.map((id) => String(id).split(sep));
	// Sort so items sharing leading segments sit together (ties keep input order).
	const order = ids.map((_, i) => i).sort((a, b) => {
		const sa = segs[a], sb = segs[b], m = Math.min(sa.length, sb.length);
		for (let k = 0; k < m; k++) if (sa[k] !== sb[k]) return sa[k] < sb[k] ? -1 : 1;
		return sa.length !== sb.length ? sa.length - sb.length : a - b;
	});
	const oSegs = order.map((i) => segs[i]);
	// Maximal contiguous runs (length ≥ 2) that share the first `depth` segments and
	// have a suffix beyond them. Label = the `depth`-th segment (the enclosing level
	// already shows the shallower ones).
	const runsAt = (depth) => {
		const runs = [];
		for (let i = 0; i < n;) {
			if (oSegs[i].length <= depth) { i++; continue; }
			const key = oSegs[i].slice(0, depth).join(sep);
			let j = i + 1;
			while (j < n && oSegs[j].length > depth && oSegs[j].slice(0, depth).join(sep) === key) j++;
			if (j - i >= 2) runs.push({ label: oSegs[i][depth - 1], i0: i, i1: j - 1 });
			i = j;
		}
		return runs;
	};
	const level1 = runsAt(1);
	const level2 = maxDepth >= 2 ? runsAt(2) : [];
	// Deepest bracket covering each position → how many leading segments to strip.
	const depthOf = new Array(n).fill(0);
	for (const r of level1) for (let i = r.i0; i <= r.i1; i++) depthOf[i] = 1;
	for (const r of level2) for (let i = r.i0; i <= r.i1; i++) depthOf[i] = 2;
	const labels = oSegs.map((s) => s.join(sep)).map((full, i) => {
		const dpt = depthOf[i];
		return dpt ? (oSegs[i].slice(dpt).join(sep) || oSegs[i][oSegs[i].length - 1]) : full;
	});
	const groupLevels = [level2, level1].filter((lvl) => lvl.length);
	return { order, labels, groupLevels, grouped: groupLevels.length > 0, sep };
}

// Per OUTPUT ITEM, its attention split across REGIONS (scene_content / to_place /
// organized / free / reasoning / output / …). The region partition only exists in
// the whole-row reduction (`a.buckets.region`, progression-bucketed), so we INFER
// each item's composition by mapping its scored-token range onto those buckets —
// items and buckets are ~1:1, so each item lands on ~its own bucket. Reuses
// `_rollRegionGrid` (→ canonical REGION_SUBS). null when the step carries no buckets.
function _featureRegions(a, feats) {
	const b = _validBuckets(a);
	if (!b || !(b.region || []).length) return null;
	const rolled = _rollRegionGrid(b), B = rolled.length, K = REGION_SUBS.length;
	const toks = a.tokens || [], N = toks.length || 1;
	// Match _bucketize: reasoning ([0,outStart)) and output ([outStart,N)) are bucketed
	// SEPARATELY, output starting at bucket out_bucket. Output items live in [outStart,N),
	// so map their token index into the OUTPUT bucket band.
	let outStart = a.out_start != null ? a.out_start : toks.findIndex((t) => t && t.output_entity != null);
	if (outStart == null || outStart < 0) outStart = 0;
	const outBucket = Math.max(0, Math.min(B - 1, b.out_bucket || 0));
	const nOut = Math.max(1, N - outStart), nOutB = Math.max(1, B - outBucket);
	const bucketOf = (i) => (i < outStart
		? Math.max(0, Math.min(outBucket, Math.floor((i / Math.max(1, outStart)) * outBucket)))
		: Math.min(B - 1, outBucket + Math.floor(((i - outStart) / nOut) * nOutB)));
	return feats.map((f) => {
		if (!f.tokens.length) return new Array(K).fill(0);
		let lo = bucketOf(Math.min(...f.tokens)), hi = bucketOf(Math.max(...f.tokens));
		if (lo > hi) { const t = lo; lo = hi; hi = t; }
		const acc = new Array(K).fill(0); let n = 0;
		for (let bi = lo; bi <= hi; bi++) { rolled[bi].forEach((v, k) => { acc[k] += v; }); n++; }
		return n ? acc.map((v) => v / n) : acc;
	});
}

// --- "per feature" within-step view -----------------------------------------
// The "per feature" x-mode maps each OUTPUT ITEM the step emitted to its attention,
// as a stacked histogram: each bar is one emitted item, stacked by REGION — how much
// of that item's attention went to scene content / to-place / organized / free /
// reasoning / output. Falls back to the scene / to-place split when the step carries
// no region buckets. The shared id prefix is shown once as a bracket under the axis.
function perFeatureCard(a, { title = "attention per output feature" } = {}) {
	const { el, reportCard, reportEmpty, COLORS } = d();
	let feats = outputFeatures(a);
	if (!feats.length) return reportCard(title, null, reportEmpty("this step emits no output items to map attention onto"));
	// Scene-mass trend along EMISSION order, measured before any regrouping.
	const rho = feats.length >= 3 ? spearman(feats.map((_, i) => i), feats.map((f) => f.sceneMass)) : NaN;
	// Group items sharing an id prefix (depth 2) so a long shared prefix becomes a
	// bracket under the axis instead of crowding every tick; this reorders grouped
	// items to be contiguous. `_featureRegions` runs on the reordered feats so the
	// stack columns line up with the labels.
	const layout = _prefixLayout(feats.map((f) => f.id), 2);
	if (layout.grouped) feats = layout.order.map((i) => feats[i]);
	const full = feats.map((f) => f.label), xs = feats.map((_, i) => i);
	const xFmt = (v) => full[Math.round(v)] ?? "";   // hover keeps the full id
	const barOpts = {
		bars: true, catLabels: layout.grouped ? layout.labels : full, catGroups: layout.groupLevels, xFmt,
		xLabel: layout.grouped ? "output item (grouped by shared prefix)" : "output item (emission order)", yFmt: pctFmt,
	};
	const corner = isFinite(rho) ? `scene trend ρ=${rho.toFixed(2)}${layout.grouped ? " (emission)" : ""}` : null;
	const regions = _featureRegions(a, feats);
	if (regions) {
		// Region composition per item (the requested "how much of each output section
		// attended to each region"). REGION_SUBS = the same categories/colors as the
		// main region graph; keep only regions that carry attention here.
		const present = REGION_SUBS.map((_, k) => k).filter((k) => regions.some((r) => r[k] > 1e-9));
		const layers = present.map((k) => ({ label: REGION_SUBS[k][2], color: REGION_SUBS[k][3], values: regions.map((r) => r[k]) }));
		const chart = stackAreaChart(layers, xs, {
			...barOpts, share: true, yMax: 1, height: 220, legend: layers.map((L) => ({ label: L.label, color: L.color })), corner,
		});
		return reportCard(title, `x = output item (${feats.length}) · stack = share of its attention on each region (inferred from region buckets)`, chart);
	}
	// Fallback (no region buckets computed): the two regions available per item — the
	// scene mass and to-place mass (from agg.outputs).
	const sceneC = "#4af0e0", tpColor = (COLORS && COLORS.to_place) || "#b46aff";
	const hasTp = feats.some((f) => f.tpMass != null);
	const layers = [{ label: "scene content", color: sceneC, values: feats.map((f) => f.sceneMass) }];
	if (hasTp) layers.push({ label: "to-place", color: tpColor, values: feats.map((f) => f.tpMass || 0) });
	const top = Math.max(1e-9, ...feats.map((f) => f.sceneMass + (hasTp ? f.tpMass || 0 : 0)));
	const chart = stackAreaChart(layers, xs, {
		...barOpts, share: false, yMax: Math.min(1, top * 1.12), height: 210, legend: layers.map((L) => ({ label: L.label, color: L.color })), corner,
	});
	return reportCard(title, `x = output item (${feats.length}) · scene vs to-place mass · compute region buckets for the full region split`, chart);
}

// Drill-down for a single clicked OUTPUT ITEM: where its attention went (region
// composition), its scene / to-place mass ± sd, and the scene entities it attended
// to most. Rendered when a segment is selected on the scene-mass trajectory; the
// ✕ button clears the selection. Returns null when the id isn't in this step.
function segmentBreakdownCard(a, selId) {
	const { el, reportCard } = d();
	const feats = outputFeatures(a);
	const idx = feats.findIndex((f) => f.id === selId);
	if (idx < 0) return null;
	const f = feats[idx];
	const reg = (_featureRegions(a, feats) || [])[idx] || null;
	const clear = d().selectSegment
		? el("button", { class: "seg-clear", title: "clear selection", style: "margin-left:auto;font-size:11px;padding:2px 8px;border-radius:5px;border:1px solid rgba(255,255,255,0.16);background:rgba(255,255,255,0.05);color:#dce6f5;cursor:pointer", onclick: () => d().selectSegment(selId), text: "✕ clear" })
		: null;
	const stat = (label, v) => el("span", {}, el("b", { text: label + " " }), el("span", { text: v }));
	const head = el("div", { class: "hint", style: "display:flex;align-items:center;gap:18px;flex-wrap:wrap" },
		stat("tokens", String(f.n ?? f.tokens.length)),
		stat("scene mass", fmtPM(f.sceneMass, f.sceneSd)),
		f.tpMass != null ? stat("to-place mass", fmtPM(f.tpMass, f.tpSd)) : null,
		clear);
	// Region composition — a horizontal 100%-stacked bar (same categories/colors as
	// the main region graph) answering "which region did THIS item attend to".
	let regionBlock = null;
	if (reg) {
		const present = REGION_SUBS.map((_, k) => k).filter((k) => reg[k] > 1e-9).sort((x, y) => reg[y] - reg[x]);
		const total = present.reduce((s, k) => s + reg[k], 0) || 1;
		if (present.length) regionBlock = el("div", {},
			el("div", { class: "graph-card-sub", style: "margin:8px 0 3px", text: "attention composition · which region this item attended to" }),
			el("div", { style: "display:flex;height:15px;border-radius:4px;overflow:hidden;border:1px solid rgba(255,255,255,0.1)" },
				...present.map((k) => el("span", { title: `${REGION_SUBS[k][2]} · ${(reg[k] / total * 100).toFixed(1)}%`, style: `background:${REGION_SUBS[k][3]};width:${(reg[k] / total * 100).toFixed(3)}%` }))),
			el("div", { style: "display:flex;gap:12px;flex-wrap:wrap;font-size:10px;margin-top:5px" },
				...present.map((k) => el("span", { style: "display:inline-flex;align-items:center;gap:4px" },
					el("span", { class: "frame-sw", style: `background:${REGION_SUBS[k][3]}` }),
					el("span", { text: `${REGION_SUBS[k][2]} ${(reg[k] / total * 100).toFixed(0)}%` })))));
	}
	// Top scene entities this item attended to (from the per-item entity totals).
	let entBlock = null;
	const ents = (f.entityTotals || []).slice().sort((x, y) => (y.score || 0) - (x.score || 0)).slice(0, 6);
	if (ents.length) {
		const mx = Math.max(1e-9, ...ents.map((e) => e.score || 0));
		entBlock = el("div", {},
			el("div", { class: "graph-card-sub", style: "margin:8px 0 3px", text: "top attended scene entities" }),
			el("div", { style: "display:flex;flex-direction:column;gap:2px" },
				...ents.map((e) => { const p = ((e.score || 0) / mx * 100).toFixed(1); return el("div", { style: "display:flex;align-items:center;gap:8px;font-size:11px" },
					el("span", { style: "flex:0 0 42%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;opacity:.85", title: e.id, text: e.id }),
					el("span", { style: `flex:1;height:8px;border-radius:3px;background:linear-gradient(90deg,#4af0e0 ${p}%,rgba(255,255,255,0.06) ${p}%)` }),
					el("span", { style: "opacity:.6;font-variant-numeric:tabular-nums", text: (e.score || 0).toFixed(3) })); })));
	}
	return reportCard(`selected item · ${f.id}`, "drill-down for the clicked output item", head, regionBlock, entBlock);
}

// Peer-comparison: one mean±sd scene-mass curve per selected step/kind/scene.
function sceneMassCompareCard(peers) {
	const { reportCard, reportEmpty } = d();
	const mode = curX(), errOn = !d().showErr || d().showErr();
	const aggs = peers.map((p) => ({ p, agg: aggregateTrajectory(p.rows) })).filter((x) => x.agg);
	if (!aggs.length) return reportCard("scene mass vs token position", null, reportEmpty("no multi-token steps to aggregate"));
	const ms = median(aggs.map((x) => x.agg.medianSpan)) || 1;
	const outRatio = mean(aggs.map((x) => x.agg.outRatio)) || 1;
	const xAt = (g) => trajAtNormalized(g, ms, outRatio, mode);
	const series = [], legend = [], trendNotes = [];
	for (const { p, agg } of aggs) {
		const aggXs = agg.grid.map(xAt);
		const perStep = sceneMassTrendSteps(p.rows, mode);
		const trend = sceneMassTrendSummary(perStep, sceneMassTrendOnSeries(aggXs, agg.mean));
		const meanPts = agg.grid.map((g, k) => [xAt(g), agg.mean[k]]);
		series.push({ points: meanPts, color: p.color, width: 2,
			band: errOn ? { hi: agg.mean.map((m, i) => m + agg.sd[i]), lo: agg.mean.map((m, i) => Math.max(0, m - agg.sd[i])) } : null });
		const peerEma = emaLineSeries(meanPts, p.color);
		if (peerEma) series.push(peerEma);
		const arrow = TREND_ARROW[trend?.dir] || "→";
		const rhoLab = trend?.rp ? `ρ=${fmtPM(trend.rp.m, trend.rp.s)}` : (trend?.aggTrend ? `ρ=${trend.aggTrend.sp.rho.toFixed(2)}` : "ρ=–");
		legend.push({ label: `${p.label} · ${arrow} ${trend?.dir || "flat"} · ${rhoLab}`, color: p.color });
		if (peerEma) legend.push({ label: `${p.label} EMA`, color: p.color });
		trendNotes.push(el("span", {}, el("b", { text: `${p.label}: ` }), el("span", { text: sceneMassTrendCorrLabel(trend) || "–" })));
	}
	const chart = lineChart(series, { height: 124, xLabel: xLabelFor(mode), xFmt: xFmtFor(mode), yFmt: pctFmt, legend });
	const note = el("div", { class: "hint trend-note", style: "display:flex;flex-direction:column;gap:3px;margin-top:6px" }, ...trendNotes);
	return reportCard("scene mass vs token position", `${aggs.length} compared · mean ± sd · trend = Spearman on position`,
		chart, note);
}

// The "scene by tokens vs output tokens" split as two numbers under the curve.
function reasoningOutputSplit(tr) {
	const { el } = d();
	const rMass = [], oMass = [];
	tr.mass.forEach((m, i) => (i < tr.outStart ? rMass : oMass).push(m));
	const rp = pm(rMass), op = pm(oMass);
	return el("div", { class: "hint", style: "display:flex;gap:18px;flex-wrap:wrap" },
		el("span", {}, el("b", { text: "reasoning tokens " }), el("span", { text: rMass.length ? fmtPM(rp.m, rp.s) : "—" })),
		el("span", {}, el("b", { text: "output tokens " }), el("span", { text: oMass.length ? fmtPM(op.m, op.s) : "—" })));
}

// --- (B) context order vs attention -----------------------------------------
// Does an entity's ORDER in the scene context relate to the attention it draws?
// Per step: point per context entity (x = order, y = per-step normalized score)
// plus a Spearman ρ; pooled across a group with per-step ρ averaged.
function contextOrderData(rows, mode) {
	const pts = [], rhos = [];
	for (const r of rows) {
		const ents = (r.a && r.a.scene_entities) || [];
		const totals = new Map((((r.a.agg || {}).scene || {}).entityTotals || []).map((e) => [e.id, e.score]));
		const withPos = ents.filter((e) => e.token_span && e.token_span.length);
		if (withPos.length < 3) continue;
		const starts = withPos.map((e) => e.token_span[0]);
		const lo = Math.min(...starts), span = Math.max(...starts) - lo;
		const maxScore = Math.max(1e-9, ...withPos.map((e) => totals.get(e.id) || 0));
		const px = [], py = [];
		for (const e of withPos) {
			const rel = e.token_span[0] - lo, score = totals.get(e.id) || 0;
			pts.push({ x: xformPos(rel, span, mode), y: score / maxScore, kind: e.kind });
			px.push(rel); py.push(score);
		}
		rhos.push(spearman(px, py));
	}
	return { pts, rhos };
}
function corrLabelOf(rhos) {
	if (!rhos.length) return "";
	if (rhos.length === 1) return `Spearman ρ = ${rhos[0].toFixed(2)}`;
	const rp = pm(rhos);
	return `Spearman ρ = ${fmtPM(rp.m, rp.s)} · ${rhos.length} steps`;
}
function contextOrderCard(rows, { title = "context order vs attention" } = {}) {
	const { reportCard, reportEmpty } = d();
	const mode = curX();
	const { pts, rhos } = contextOrderData(rows, mode);
	if (pts.length < 3) return reportCard(title, null, reportEmpty("not enough context entities with positions + attention"));
	const points = pts.map((q) => ({ x: q.x, y: q.y, color: q.kind === "zone" ? "#e0a94a" : "#7aa2f7" }));
	return reportCard(title, "dot = a context entity · x = order in scene context · y = attention (per-step normalized)",
		scatterChart(points, {
			height: 132, xLabel: xLabelFor(mode), xFmt: xFmtFor(mode), yMax: 1, yFmt: (v) => v.toFixed(1),
			corrLabel: corrLabelOf(rhos), legend: [{ label: "object", color: "#7aa2f7" }, { label: "zone", color: "#e0a94a" }],
		}));
}
function contextOrderCompareCard(peers) {
	const { reportCard, reportEmpty } = d();
	const mode = curX();
	const points = [], legend = [];
	for (const p of peers) {
		const { pts, rhos } = contextOrderData(p.rows, mode);
		if (!pts.length) continue;
		for (const q of pts) points.push({ x: q.x, y: q.y, color: p.color });
		const r = rhos.length === 1 ? rhos[0] : pm(rhos).m;
		legend.push({ label: `${p.label} · ρ=${isFinite(r) ? r.toFixed(2) : "–"}`, color: p.color });
	}
	if (points.length < 3) return reportCard("context order vs attention", null, reportEmpty("not enough context entities with attention"));
	return reportCard("context order vs attention", "x = order in scene context · y = attention (per-step normalized)",
		scatterChart(points, { height: 132, xLabel: xLabelFor(mode), xFmt: xFmtFor(mode), yMax: 1, yFmt: (v) => v.toFixed(1), legend }));
}

// --- (C) plotting order vs scene mass ---------------------------------------
// For to-place objects: does WHEN an object is emitted (n) relate to the scene
// mass while its bbox is written? Mean scene mass per emission rank across steps.
function toPlaceRankData(rows) {
	const orderings = rows.map((r) => toPlaceOrdering(r.a)).filter(Boolean);
	if (!orderings.length) return null;
	const maxK = Math.max(...orderings.map((o) => o.objs.length));
	const meanArr = [], sdArr = [];
	for (let k = 0; k < maxK; k++) { const p = pm(orderings.map((o) => o.objs[k]?.sceneMass).filter((v) => v != null)); meanArr.push(p.m); sdArr.push(p.s); }
	return { orderings, maxK, mean: meanArr, sd: sdArr };
}
function toPlaceOrderCard(rows, { title = "plotting order vs scene mass" } = {}) {
	const { reportCard, reportEmpty, COLORS } = d();
	const errOn = !d().showErr || d().showErr();
	const tpColor = (COLORS && COLORS.to_place) || "#e0a94a";
	const data = toPlaceRankData(rows);
	if (!data) return reportCard(title, null, reportEmpty("no to-place objects in this selection"));
	if (data.orderings.length === 1) {
		const objs = data.orderings[0].objs;
		const series = [{ points: objs.map((x) => [x.outRank, x.sceneMass]), color: "#4af0e0", marker: true }];
		if (objs.some((x) => x.tpMass != null)) series.push({ points: objs.map((x) => [x.outRank, x.tpMass || 0]), color: tpColor, marker: true, dash: [4, 3] });
		return reportCard(title, "x = order the object is emitted (n) · y = scene mass during its bbox",
			lineChart(series, { height: 120, xMin: 0, xLabel: "output order (n)", yFmt: pctFmt,
				legend: [{ label: "scene mass", color: "#4af0e0" }, series.length > 1 ? { label: "to-place mass", color: tpColor } : null] }));
	}
	const grid = Array.from({ length: data.maxK }, (_, k) => k);
	const series = data.orderings.map((o) => ({ points: o.objs.map((x) => [x.outRank, x.sceneMass]), color: "#7aa2f7", faint: true }));
	series.push({ points: grid.map((k) => [k, data.mean[k]]), color: "#7aa2f7", width: 2, marker: true,
		band: errOn ? { hi: data.mean.map((m, i) => m + data.sd[i]), lo: data.mean.map((m, i) => Math.max(0, m - data.sd[i])) } : null });
	return reportCard(title, `mean ± sd across ${data.orderings.length} bbox steps · x = emission order`,
		lineChart(series, { height: 120, xMin: 0, xLabel: "output order (n)", yFmt: pctFmt,
			legend: [{ label: `mean of ${data.orderings.length} steps`, color: "#7aa2f7" }, { label: "per-step", color: hexA("#7aa2f7", 0.4) }] }));
}
function toPlaceOrderCompareCard(peers) {
	const { reportCard, reportEmpty } = d();
	const series = [], legend = [];
	for (const p of peers) {
		const data = toPlaceRankData(p.rows);
		if (!data) continue;
		const grid = Array.from({ length: data.maxK }, (_, k) => k);
		series.push({ points: grid.map((k) => [k, data.mean[k]]), color: p.color, width: 2, marker: true });
		legend.push({ label: p.label, color: p.color });
	}
	if (!series.length) return reportCard("plotting order vs scene mass", null, reportEmpty("no to-place objects in this selection"));
	return reportCard("plotting order vs scene mass", "mean scene mass by emission order (n)",
		lineChart(series, { height: 120, xMin: 0, xLabel: "output order (n)", yFmt: pctFmt, legend }));
}

// --- (D) input order vs output order ----------------------------------------
// Is the order to-place objects are GIVEN to the model correlated with the order
// it EMITS them? (ranks normalized so steps of different sizes overlay.)
function inOutOrderData(rows) {
	const orderings = rows.map((r) => toPlaceOrdering(r.a)).filter((o) => o && o.objs.length >= 2);
	const pts = [], rhos = [];
	for (const o of orderings) {
		const denom = Math.max(1, o.objs.length - 1);
		rhos.push(spearman(o.objs.map((x) => x.inRank), o.objs.map((x) => x.outRank)));
		for (const x of o.objs) pts.push({ x: x.inRank / denom, y: x.outRank / denom });
	}
	return { orderings, pts, rhos };
}
function inOutOrderCard(rows, { title = "input order vs output order" } = {}) {
	const { reportCard, reportEmpty } = d();
	const { orderings, pts, rhos } = inOutOrderData(rows);
	if (!orderings.length) return reportCard(title, null, reportEmpty("need a bbox step with ≥2 to-place objects"));
	return reportCard(title, "dot = a to-place object · x = order given to model · y = order emitted · dashed = identity",
		scatterChart(pts, { height: 138, xMax: 1, yMax: 1, refLine: true, xLabel: "input order (normalized)", xFmt: (v) => v.toFixed(1), yFmt: (v) => v.toFixed(1), corrLabel: corrLabelOf(rhos) }));
}
function inOutOrderCompareCard(peers) {
	const { reportCard, reportEmpty } = d();
	const points = [], legend = [];
	for (const p of peers) {
		const { orderings, pts, rhos } = inOutOrderData(p.rows);
		if (!orderings.length) continue;
		for (const q of pts) points.push({ x: q.x, y: q.y, color: p.color });
		const r = orderings.length === 1 ? rhos[0] : pm(rhos).m;
		legend.push({ label: `${p.label} · ρ=${isFinite(r) ? r.toFixed(2) : "–"}`, color: p.color });
	}
	if (!points.length) return reportCard("input order vs output order", null, reportEmpty("need bbox steps with ≥2 to-place objects"));
	return reportCard("input order vs output order", "x = order given to model · y = order emitted · dashed = identity",
		scatterChart(points, { height: 138, xMax: 1, yMax: 1, refLine: true, xLabel: "input order (normalized)", xFmt: (v) => v.toFixed(1), yFmt: (v) => v.toFixed(1), legend }));
}

// Assemble the token-ordering cards for a report level. Step = the current step;
// kind/scene = aggregate over the level's rows; when peers are selected it
// overlays them (compare) exactly like the other report tabs. To-place cards
// appear only where object_bbox steps exist. Each card spans the full width with
// a wide/short SVG so a full-width row stays a readable, non-towering height.
// ===== aggregation-expansion stacked-area graphs ===========================
// Consume the server's per-step `a.buckets` (region × word-type mass over
// generation-progression buckets, head-averaged). Graph 1 = attention by
// category (color) × subcategory (area); Graph 2 = word types (aggregate +
// organized/free splits). All reuse the SVG stackAreaChart primitive. This
// version also CROSS-VALIDATES the contract (validateBuckets) so integration/
// shape mismatches surface loudly instead of rendering silently-wrong.

// Canonical region subcategories (disjoint partition), fixed order + color so
// steps/scenes with different leaf sets align. color families: variables=teal,
// text=amber, completion=violet.
const REGION_SUBS = [
	["variables", "scene_content", "scene content", "#4af0e0"],
	["variables", "to_place", "to-place", "#2a9d94"],
	["variables", "other", "other vars", "#17706a"],
	["text", "organized", "organized", "#f0c070"],
	["text", "free", "free text", "#b07820"],
	["completion", "reasoning", "reasoning", "#9a7ae8"],
	["completion", "output", "output", "#6a48c0"],
	["other", "template", "template / control", "#5b6270"],
];
const REGION_CAT_COLOR = { variables: "#4af0e0", text: "#f0c070", completion: "#9a7ae8", other: "#5b6270" };
// Word/token-type colors. The structural family (bracket/separator/quote/operator)
// shares a blue ramp so it reads as one group split into tags; the lexical kinds
// keep their distinct hues.
const TYPE_STYLE = {
	number: "#4af0e0",
	bracket: "#7aa2f7", separator: "#4d7fd6", quote: "#a9c7ff", operator: "#33507e",
	whitespace: "#3a3f4b", spatial: "#e0a94a", function: "#c98bdb",
	entity_name: "#5be584", content: "#f7768e", other: "#9098a6",
};
// Word-type families for the grouped legend (structural tags read together).
const TYPE_GROUPS = [
	["structural", ["bracket", "separator", "quote", "operator"]],
	["lexical", ["number", "spatial", "function", "content", "entity_name"]],
	["layout", ["whitespace", "other"]],
];
// Plain-language definition of every word/token class + group heading — matches the
// backend classifier (semantic.classify_tokens) so the legend explains exactly what
// each bucket counts. Shown as a hover tooltip on the swatch and the group heading.
const TYPE_GROUP_DESC = {
	structural: "JSON / markup scaffolding punctuation the model emits — split into bracket / separator / quote / operator so the structure is resolved, not lumped together.",
	lexical: "Word-level classes — what the words themselves mean (numbers, spatial relations, grammatical glue, content words, entity names).",
	layout: "Non-semantic filler — whitespace and any residue that doesn't fall into a structural or lexical class.",
};
const TYPE_DESC = {
	bracket: "Bracket punctuation { } [ ] ( ) — object / array / grouping scaffolding.",
	separator: "Separators : and , — the key→value and item-to-item delimiters in the emitted JSON.",
	quote: "Quote marks \" ' ` — string delimiters wrapping keys and values.",
	operator: "= and any residual symbol punctuation — assignment / operator characters.",
	number: "Numeric tokens — integers or decimals (coordinates, sizes, counts).",
	spatial: "Spatial-relation words — above, below, beside, on, in, left, right, between, near, …",
	function: "Closed-class function words — the, a, of, to, is, are, and, … (grammatical glue, no content).",
	content: "Open-class content words — nouns / verbs / adjectives that aren't in the function or spatial lists.",
	entity_name: "Tokens inside an entity's NAME span — overlaps the scene layer (proper nouns / object ids).",
	whitespace: "Whitespace-only tokens — the spaces / newlines between other tokens.",
	other: "Residual — sentential punctuation and symbols not covered by any class above.",
};
// Grouped word-type legend: each family sits under a faint heading so the four
// structural tags read as one split group rather than a flat list of swatches.
// Hover any swatch or heading for its definition.
function wordTypeLegend() {
	const { el } = d();
	const group = ([title, keys]) => el("div", { style: "display:inline-flex;align-items:center;gap:8px" },
		el("span", { style: "font-size:9px;letter-spacing:.05em;text-transform:uppercase;opacity:.45;cursor:help;border-bottom:1px dotted rgba(220,230,245,0.35)", title: TYPE_GROUP_DESC[title] || title, text: title }),
		...keys.map((nm) => el("span", { style: "display:inline-flex;align-items:center;gap:4px;cursor:help", title: TYPE_DESC[nm] || nm },
			el("span", { class: "frame-sw", style: `background:${TYPE_STYLE[nm]}` }), el("span", { text: nm }))));
	return el("div", { class: "seg-legend", style: "font-size:10px;gap:18px;margin-top:8px;flex-wrap:wrap;align-items:center" },
		...TYPE_GROUPS.map(group));
}

function _validBuckets(a) {
	const b = a && a.buckets;
	return b && b.region_names && Array.isArray(b.region) ? b : null;
}

// Contract cross-validation — push human-readable issues; also console.warn.
function validateBuckets(a, r, warnings) {
	const b = a && a.buckets;
	const tag = r && r.step ? `#${r.step.event_index}` : "?";
	if (!b || !Object.keys(b).length) return;   // scene-less step legitimately carries no view
	const need = ["region_names", "region", "region_meta", "type_names", "type", "type_organized", "type_free", "n_tokens"];
	for (const k of need) if (!(k in b)) warnings.push(`${tag}: missing buckets.${k}`);
	if (b.region && b.region.length) {
		if (b.region[0].length !== (b.region_names || []).length) warnings.push(`${tag}: region grid width ${b.region[0].length} ≠ region_names ${(b.region_names || []).length}`);
		if ((b.region_meta || []).length !== (b.region_names || []).length) warnings.push(`${tag}: region_meta ${(b.region_meta || []).length} ≠ region_names ${(b.region_names || []).length}`);
		const s = b.region[Math.floor(b.region.length / 2)].reduce((a2, v) => a2 + v, 0);
		if (s > 1e-6 && Math.abs(s - 1) > 0.05) warnings.push(`${tag}: region partition sums ${s.toFixed(2)} ≠ 1`);
	}
	if (b.type && b.type.length && b.type[0].length !== (b.type_names || []).length) warnings.push(`${tag}: type grid width ≠ type_names`);
	if (b.region_tokens && b.region_tokens.length !== (b.region_names || []).length) warnings.push(`${tag}: region_tokens ≠ region_names`);
	for (const m of b.region_meta || []) if (!REGION_SUBS.some((s2) => s2[0] === m.category && s2[1] === m.sub)) warnings.push(`${tag}: unknown region category/sub ${m.category}/${m.sub}`);
}

function _diag(warnings) {
	const { el } = d();
	if (!warnings.length) return null;
	const uniq = [...new Set(warnings)];
	uniq.slice(0, 12).forEach((w) => console.warn("[buckets contract]", w));
	return el("div", { class: "hint", style: "color:#f7768e;margin-top:4px" },
		el("b", { text: `⚠ ${uniq.length} contract issue(s): ` }),
		el("span", { text: uniq.slice(0, 4).join(" · ") + (uniq.length > 4 ? " …(see console)" : "") }));
}

// Roll a step's region leaf grid [B][leaves] onto the canonical [B][7 subs].
function _rollRegionGrid(b) {
	const names = b.region_names || [], meta = b.region_meta || [], grid = b.region || [];
	const subIdx = new Map(REGION_SUBS.map((s, i) => [`${s[0]}/${s[1]}`, i]));
	const colSub = names.map((_, c) => { const m = meta[c] || {}; return subIdx.get(`${m.category}/${m.sub}`); });
	return grid.map((row) => {
		const out = new Array(REGION_SUBS.length).fill(0);
		row.forEach((v, c) => { const si = colSub[c]; if (si != null) out[si] += v; });
		return out;
	});
}

// Resample a [B][K] grid onto [G][K] by linear interp over the bucket fraction.
function _resampleGrid(grid, G) {
	const B = grid.length;
	if (!B) return [];
	return Array.from({ length: G }, (_, g) => {
		const x = (G === 1 ? 0 : g / (G - 1)) * (B - 1);
		const lo = Math.floor(x), hi = Math.min(B - 1, lo + 1), t = x - lo;
		return grid[lo].map((v, k) => v * (1 - t) + grid[hi][k] * t);
	});
}
function _mean1(xs) { return xs.length ? xs.reduce((a, v) => a + v, 0) / xs.length : 0; }

// Output-zoom trim of a step's RAW per-bucket grid: keep only the output buckets
// (from out_bucket on) so the later resample-to-G runs at full resolution over the
// output region. Also rescales the token axis (meanNq → output token count) and
// resets outFrac to 0. No-op unless zoom is on and the step has an output region.
function _stepGridScaled(b, grid, zoom) {
	const nb = b.n_buckets || grid.length || 1, ob = b.out_bucket || 0;
	let meanNq = b.n_query || 0, outFrac = nb ? ob / nb : 0;
	if (zoom && ob > 0 && ob < grid.length) { grid = grid.slice(ob); meanNq *= (1 - outFrac); outFrac = 0; }
	return { grid, meanNq, outFrac };
}

// Aggregate region sub-grid over a selection's steps, on a shared progression grid.
function _regionProgression(rows, { normalize, zoom }) {
	const G = 32, acc = [];
	let n = 0, nqSum = 0, outSum = 0;
	for (const r of rows) {
		const b = _validBuckets(r.a);
		if (!b || !(b.region || []).length) continue;
		let grid = _rollRegionGrid(b);
		if (normalize) { const nt = b.n_tokens || 1; grid = grid.map((row) => row.map((v) => v / nt)); }
		const sc = _stepGridScaled(b, grid, zoom);
		const rg = _resampleGrid(sc.grid, G);
		if (!acc.length) rg.forEach((row) => acc.push(row.slice()));
		else rg.forEach((row, i) => row.forEach((v, k) => { acc[i][k] += v; }));
		n++; nqSum += sc.meanNq; outSum += sc.outFrac;
	}
	if (!n) return null;
	return { grid: acc.map((row) => row.map((v) => v / n)), G, meanNq: nqSum / n || 1, outFrac: outSum / n, n };
}

// --- cross-step grouping ----------------------------------------------------
// A kind/scene selection can COMPARE its steps instead of averaging them into one
// within-step curve: partition the rows by a dimension (one step each, or by step
// KIND / ZONE) and emit one stacked column per group. Groups are ordered by first
// pipeline appearance (min event_index) so the x-axis reads in run order.
const _kindOf = (r) => (r.step && (r.step.template ?? r.step.step)) || "?";
const _zoneOf = (r) => (r.step && r.step.node) || "(scene)";
// `abl` groups by the ablation ITEM: rows are pre-tagged (report.js) with
// _ablGroup (column key), _ablOrd (column order), _ablLabel (display) so the same
// stacks can put one column per ablation factor level on the x-axis.
const _GROUP_KEY = { step: (r) => r.step?.event_index ?? 0, kind: _kindOf, zone: _zoneOf, abl: (r) => r._ablGroup ?? "?" };
function _groupRows(rows, dim) {
	const keyOf = _GROUP_KEY[dim] || _GROUP_KEY.step;
	const abl = dim === "abl";
	const map = new Map();
	for (const r of rows) {
		const k = keyOf(r), ord = abl ? (r._ablOrd ?? 0) : (r.step?.event_index ?? 0);
		if (!map.has(k)) map.set(k, { key: k, rows: [], ord, label: abl ? (r._ablLabel ?? String(k)) : null });
		const g = map.get(k); g.rows.push(r); g.ord = Math.min(g.ord, ord);
	}
	const groups = [...map.values()].sort((a, b) => a.ord - b.ord);
	groups.forEach((g, i) => { if (g.label == null) g.label = dim === "step" ? String(i + 1) : String(g.key); });
	return groups;
}
// Average a per-row vector across a group's steps. `vecOf(row)` returns a step's
// mean-over-buckets vector (region sub-mass, or a type-key mass).
function _groupMean(groupRows, vecOf, K) {
	const acc = new Array(K).fill(0);
	for (const r of groupRows) { const v = vecOf(r); for (let k = 0; k < K; k++) acc[k] += v[k] || 0; }
	return acc.map((v) => v / (groupRows.length || 1));
}
// Region sub-mass per group (dim = step | kind | zone). One column per group.
function _regionByGroup(rows, dim, { normalize }) {
	const valid = (rows || []).filter((r) => _validBuckets(r.a) && (r.a.buckets.region || []).length);
	const groups = _groupRows(valid, dim);
	if (!groups.length) return null;
	const K = REGION_SUBS.length;
	const vecOf = (r) => {
		const b = r.a.buckets, rg = _rollRegionGrid(b), nb = rg.length || 1, nt = normalize ? (b.n_tokens || 1) : 1, m = new Array(K).fill(0);
		rg.forEach((row) => row.forEach((v, k) => { m[k] += v; }));
		return m.map((v) => v / nb / nt);
	};
	return { grid: groups.map((g) => _groupMean(g.rows, vecOf, K)), labels: groups.map((g) => g.label), dim };
}

// --- organized prompt sections (per-XML-tag) --------------------------------
// The region graph rolls every organized <tag> into one "organized" area; these
// helpers keep the tags SEPARATE so we can see whether the model attends to a
// specific section like <output> or <judging_criteria>. A step's region grid
// carries one column per leaf; `region_meta[c].tag` names the organized ones.
const SECTION_COLORS = [
	"#7aa2f7", "#6bd96e", "#e0a94a", "#b46aff", "#4af0e0", "#ff6b9d",
	"#ff9e64", "#9ece6a", "#f7768e", "#7dcfff", "#c98bdb", "#e6db74",
];
// Stable color: index into the (sorted) tag list so a tag keeps its hue across
// steps/scenes of the same template.
function sectionColor(tag, tags) { const i = tags.indexOf(tag); return SECTION_COLORS[(i < 0 ? 0 : i) % SECTION_COLORS.length]; }
// The section breakdown shows each organized <tag> as ONE layer — VII sub-sentences
// (VERY_IMPORTANT_INSTRUCTIONS#NN) collapse back to the whole section (the
// per-instruction split is the dedicated VII view's job, not this overview).
const _collapseSectionTag = (t) => (t || "section").replace(/#\d+$/, "");
// Sorted union of organized-tag names across a selection's steps.
function _sectionTags(rows) {
	const set = new Set();
	for (const r of rows) {
		const b = _validBuckets(r.a);
		if (!b) continue;
		(b.region_meta || []).forEach((m) => { if (m && m.category === "text" && m.sub === "organized") set.add(_collapseSectionTag(m.tag)); });
	}
	return [...set].sort();
}
// One step's per-bucket per-tag organized mass, keyed to `tags` (shared order).
function _sectionRowsFor(b, tags, tagIdx, normalize) {
	const meta = b.region_meta || [], grid = b.region || [];
	const nt = normalize ? (b.n_tokens || 1) : 1;
	return grid.map((row) => {
		const o = new Array(tags.length).fill(0);
		meta.forEach((m, c) => { if (m && m.category === "text" && m.sub === "organized") { const k = tagIdx.get(_collapseSectionTag(m.tag)); if (k != null) o[k] += row[c] / nt; } });
		return o;
	});
}
// Aggregate per-tag organized mass over a selection, on a shared progression grid.
function _sectionProgression(rows, { normalize, tags: tagsIn, zoom }) {
	const G = 32, tags = tagsIn || _sectionTags(rows);
	if (!tags.length) return null;
	const tagIdx = new Map(tags.map((t, i) => [t, i]));
	const acc = []; let n = 0, nqSum = 0, outSum = 0;
	for (const r of rows) {
		const b = _validBuckets(r.a);
		if (!b || !(b.region || []).length) continue;
		const sc = _stepGridScaled(b, _sectionRowsFor(b, tags, tagIdx, normalize), zoom);
		const rg = _resampleGrid(sc.grid, G);
		if (!acc.length) rg.forEach((row) => acc.push(row.slice()));
		else rg.forEach((row, i) => row.forEach((v, k) => { acc[i][k] += v; }));
		n++; nqSum += sc.meanNq; outSum += sc.outFrac;
	}
	if (!n) return null;
	return { grid: acc.map((row) => row.map((v) => v / n)), tags, G, meanNq: nqSum / n || 1, outFrac: outSum / n, n };
}
// Per-group mean per-tag organized mass (dim = step | kind | zone).
function _sectionByGroup(allRows, dim, { normalize }) {
	const valid = (allRows || []).filter((r) => _validBuckets(r.a) && (r.a.buckets.region || []).length);
	const groups = _groupRows(valid, dim);
	const tags = _sectionTags(valid);
	if (!groups.length || !tags.length) return null;
	const tagIdx = new Map(tags.map((t, i) => [t, i]));
	const vecOf = (r) => {
		const rows2 = _sectionRowsFor(r.a.buckets, tags, tagIdx, normalize), nb = rows2.length || 1, m = new Array(tags.length).fill(0);
		rows2.forEach((row) => row.forEach((v, k) => { m[k] += v; }));
		return m.map((v) => v / nb);
	};
	return { grid: groups.map((g) => _groupMean(g.rows, vecOf, tags.length)), tags, labels: groups.map((g) => g.label), dim };
}

function _typeProgression(rows, key, { normalize, zoom }) {
	const G = 32, acc = [];
	let n = 0, nqSum = 0, outSum = 0, names = null;
	for (const r of rows) {
		const b = _validBuckets(r.a);
		if (!b) continue;
		const grid0 = b[key] || [];
		if (!grid0.length) continue;
		names = names || b.type_names;
		let grid = grid0;
		if (normalize) { const nt = b.n_tokens || 1; grid = grid0.map((row) => row.map((v) => v / nt)); }
		const sc = _stepGridScaled(b, grid, zoom);
		const rg = _resampleGrid(sc.grid, G);
		if (!acc.length) rg.forEach((row) => acc.push(row.slice()));
		else rg.forEach((row, i) => row.forEach((v, k) => { acc[i][k] += v; }));
		n++; nqSum += sc.meanNq; outSum += sc.outFrac;
	}
	if (!n) return null;
	return { grid: acc.map((row) => row.map((v) => v / n)), names, G, meanNq: nqSum / n || 1, outFrac: outSum / n };
}
function _typeByGroup(allRows, dim, key, { normalize }) {
	const valid = (allRows || []).filter((r) => _validBuckets(r.a) && (r.a.buckets[key] || []).length);
	const groups = _groupRows(valid, dim);
	if (!groups.length) return null;
	const names = valid[0].a.buckets.type_names || [];
	const K = names.length;
	const vecOf = (r) => {
		const b = r.a.buckets, g0 = b[key] || [], m = new Array(K).fill(0);
		g0.forEach((row) => row.forEach((v, k) => { m[k] += v; }));
		const nb = g0.length || 1, nt = normalize ? (b.n_tokens || 1) : 1;
		return m.map((v) => v / nb / nt);
	};
	return { grid: groups.map((g) => _groupMean(g.rows, vecOf, K)), names, labels: groups.map((g) => g.label), dim };
}

function _progXs(G, meanNq, mode) {
	return Array.from({ length: G }, (_, g) => {
		const f = G === 1 ? 0 : g / (G - 1);
		if (mode === "ratio") return f;
		if (mode === "log") return Math.log1p(f * meanNq);
		return f * meanNq;
	});
}
const _densFmt = (v) => (v >= 0.01 ? v.toFixed(2) : v === 0 ? "0" : v.toExponential(1));

// Axis options for a grouped stacked area (one column per group). Step groups keep
// a numeric ordinal axis; categorical dims (kind/zone) label every column.
const _GROUP_XLABEL = { step: "step (selection order)", kind: "step kind", zone: "zone", abl: "ablation item" };
function _groupXOpts(agg) {
	const xs = agg.grid.map((_, i) => i);
	// step = an ordered sequence → stacked area reads as a progression. kind/zone are
	// CATEGORICAL → a stacked histogram (bars), one labeled column per group.
	if (agg.dim === "step") return { xs, xLabel: _GROUP_XLABEL.step, xFmt: (v) => String(Math.round(v) + 1) };
	const labels = agg.labels;
	return { xs, xLabel: _GROUP_XLABEL[agg.dim] || agg.dim, catLabels: labels, bars: true, xFmt: (v) => labels[Math.round(v)] ?? "" };
}

// Stacked-area primitive: layers drawn bottom→top as SVG <path> areas; color by
// category; optional right-edge labels aligned to each area's mid-height with
// de-collision + leader lines. A crosshair tooltip lists every subcategory's
// value at the hovered x. `catLabels` (one per x) draws a rotated per-column tick
// for categorical group axes instead of the 3 numeric ticks.
function stackAreaChart(layers, xs, opts = {}) {
	const { el, niceMax } = d();
	layers = layers.filter((L) => (L.values || []).some((v) => v > 1e-9));
	if (!layers.length || !(xs || []).length) return el("div", { class: "hint", text: opts.empty || "no data to plot" });
	const bars = !!opts.bars;
	// A single AREA group (e.g. a kind with one step) can't form a band, so widen it
	// into a flat full-width column. Bars don't need this — one bar is fine.
	const labelXs = xs.slice();
	if (xs.length === 1 && !bars) {
		const x0 = xs[0];
		xs = [x0 - 0.4, x0 + 0.4];
		layers = layers.map((L) => ({ ...L, values: [L.values[0], L.values[0]] }));
	}
	const n = xs.length;
	const W = 960, H = Math.round((opts.height || 160) * CHART_H);
	const sideLabels = opts.sideLabels && !bars;   // right-edge leaders make no sense per-bar
	const twoLine = opts.catLabels && opts.catLabels.some((l) => String(l).includes("\n"));
	// Grouping brackets: nested levels, finest first. `catGroup` (single) is still
	// accepted and treated as one level for back-compat.
	const groupLevels = opts.catGroups || (opts.catGroup ? [[opts.catGroup]] : null);
	const nGroupLv = groupLevels && bars ? groupLevels.length : 0;
	const padL = 46, padR = sideLabels ? 118 : 14, padT = 10;
	const padB = (opts.catLabels ? (twoLine ? 64 : 48) : 28) + nGroupLv * 22;
	const cum = xs.map(() => 0);
	const tops = layers.map((L) => L.values.map((v, i) => (cum[i] += v)));
	// Shares (mass fractions) cap the axis at exactly 1.0 → a clean 0–100% that never
	// overshoots to 150/200%. Non-share (per-token density) auto-fits.
	const yMax = opts.yMax != null ? opts.yMax : opts.share ? Math.max(1, ...cum) : (niceMax ? niceMax(Math.max(...cum, 1e-9)) : Math.max(...cum, 1e-9) * 1.05);
	// Histogram: half-unit margins so columns 0…n−1 sit centered with padding.
	const fr = svgFrame(W, H, { padL, padR, padT, padB, xMin: bars ? -0.5 : xs[0], xMax: bars ? n - 0.5 : xs[n - 1], yMax, yFmt: opts.yFmt, xFmt: opts.xFmt, xLabel: opts.xLabel, shade: opts.shade, xTicks: !opts.catLabels });
	// Categorical group axis: a rotated label under each column (kind/zone/item names),
	// anchored at each group's ORIGINAL position (a widened single area stays centered).
	// When there are many columns, thin the labels (every k-th) so they don't overlap —
	// the hover tooltip still names every bar.
	if (opts.catLabels) {
		const stride = Math.max(1, Math.ceil(opts.catLabels.length / 30));
		const clip = (s) => (s.length > 18 ? s.slice(0, 17) + "…" : s);
		opts.catLabels.forEach((lab, i) => {
			if (labelXs[i] == null || i % stride !== 0) return;
			const x = fr.X(labelXs[i]), parts = String(lab).split("\n");
			const t = svgEl("text", { x, y: fr.py1 + 4, fill: "rgba(220,230,245,0.62)", "font-size": 9, "text-anchor": "end", "dominant-baseline": "middle", transform: `rotate(-32 ${x} ${fr.py1 + 4})` });
			t.appendChild(svgEl("tspan", { x, dy: 0 }, clip(parts[0])));            // distinguishing part
			if (parts[1] != null) t.appendChild(svgEl("tspan", { x, dy: "1.05em", fill: "rgba(220,230,245,0.4)", "font-size": 8 }, clip(parts[1]))); // shared group prefix
			fr.svg.appendChild(t);
		});
	}
	const barW = bars ? Math.max(3, ((fr.px1 - fr.px0) / n) * 0.72) : 0;
	let lower = xs.map(() => 0);
	const labelPts = [];
	layers.forEach((L, li) => {
		const upper = tops[li];
		if (bars) {
			for (let i = 0; i < n; i++) {
				if (upper[i] - lower[i] <= 1e-9) continue;
				const yTop = fr.Y(upper[i]), yBot = fr.Y(lower[i]);
				const rect = svgEl("rect", { class: "garea", x: (fr.X(xs[i]) - barW / 2).toFixed(2), y: yTop.toFixed(2), width: barW.toFixed(2), height: Math.max(0, yBot - yTop).toFixed(2), fill: L.color, "fill-opacity": 0.9, stroke: L.color, "stroke-width": 0.4, "stroke-opacity": 0.6 });
				rect.appendChild(svgEl("title", null, L.label));
				fr.svg.appendChild(rect);
				// ±sd whisker on the bar top (single-series bars pass an `err` array).
				const err = L.err && L.err[i];
				if (err) {
					const cx = fr.X(xs[i]), cap = Math.min(5, barW * 0.3);
					const hi = fr.Y(Math.min(fr.yMax, upper[i] + err)), lo = fr.Y(Math.max(0, upper[i] - err));
					for (const seg of [[cx, hi, cx, lo], [cx - cap, hi, cx + cap, hi], [cx - cap, lo, cx + cap, lo]])
						fr.svg.appendChild(svgEl("line", { x1: seg[0], y1: seg[1], x2: seg[2], y2: seg[3], stroke: "rgba(255,255,255,0.7)", "stroke-width": 1 }));
				}
			}
		} else {
			let dstr = "";
			for (let i = 0; i < n; i++) dstr += (i ? "L" : "M") + fr.X(xs[i]).toFixed(2) + "," + fr.Y(upper[i]).toFixed(2) + " ";
			for (let i = n - 1; i >= 0; i--) dstr += "L" + fr.X(xs[i]).toFixed(2) + "," + fr.Y(lower[i]).toFixed(2) + " ";
			const area = svgEl("path", { class: "garea", d: dstr + "Z", fill: L.color, "fill-opacity": 0.85, stroke: L.color, "stroke-width": 0.6, "stroke-opacity": 0.9 });
			area.appendChild(svgEl("title", null, L.label));
			fr.svg.appendChild(area);
			// One label per area, at its RIGHT edge only (two-sided was cramped), skipping
			// slivers; de-collided + leader-lined below.
			if (sideLabels && L.values[n - 1] > 0.02 * (yMax || 1)) labelPts.push({ y0: (lower[n - 1] + upper[n - 1]) / 2, color: L.color, label: L.label });
		}
		lower = upper.slice();
	});
	if (opts.sideLabels && labelPts.length) {
		const pts = labelPts.map((p) => ({ ...p, yy: fr.Y(p.y0) })).sort((a, b) => a.yy - b.yy);
		for (let i = 1; i < pts.length; i++) if (pts[i].yy < pts[i - 1].yy + 12) pts[i].yy = pts[i - 1].yy + 12;
		for (const p of pts) p.yy = Math.max(fr.py0 + 4, Math.min(fr.py1 - 2, p.yy));
		for (const p of pts) {
			fr.svg.appendChild(svgEl("line", { x1: fr.px1, y1: fr.Y(p.y0), x2: fr.px1 + 6, y2: p.yy, stroke: hexA(p.color, 0.5), "stroke-width": 0.7 }));
			fr.svg.appendChild(svgEl("text", { x: fr.px1 + 9, y: p.yy, fill: p.color, "font-size": 10, "text-anchor": "start", "dominant-baseline": "middle" }, p.label));
		}
	}
	// Grouping brackets: shared id prefixes drawn ONCE under the axis spanning their
	// bars (└──────┘), instead of repeated on every bar's label. Levels are nested,
	// finest first (closest to the axis); each level sits 22px lower.
	if (groupLevels && bars) {
		groupLevels.forEach((level, li) => {
			const by = fr.py1 + 34 + li * 22;
			for (const g of level) {
				const i0 = Math.max(0, g.i0 | 0), i1 = Math.min(n - 1, g.i1 | 0);
				if (i1 < i0) continue;
				const xL = fr.X(xs[i0]) - barW / 2, xR = fr.X(xs[i1]) + barW / 2;
				fr.svg.appendChild(svgEl("path", { d: `M${xL.toFixed(1)},${by - 5} L${xL.toFixed(1)},${by} L${xR.toFixed(1)},${by} L${xR.toFixed(1)},${by - 5}`, fill: "none", stroke: "rgba(220,230,245,0.32)", "stroke-width": 1 }));
				// Clip the label to its span so nested labels don't collide.
				const maxChars = Math.max(3, Math.floor((xR - xL) / 6));
				const lab = String(g.label).length > maxChars ? String(g.label).slice(0, Math.max(1, maxChars - 1)) + "…" : g.label;
				fr.svg.appendChild(svgEl("text", { x: ((xL + xR) / 2).toFixed(1), y: by + 3, fill: "rgba(220,230,245,0.6)", "font-size": 10, "text-anchor": "middle", "dominant-baseline": "hanging" }, lab));
			}
		});
	}
	_drawVlines(fr, opts.vlines);
	_drawHi(fr, opts.hi);
	svgCorner(fr, opts.corner);
	const wrap = el("div", { class: "gwrap" }, fr.svg);
	chartHover(wrap, fr, xs, layers.map((L) => ({ label: L.label, color: L.color, values: L.values })), { xFmt: opts.xFmt, vFmt: opts.yFmt || pctFmt });
	if (opts.legend) wrap.appendChild(chartLegend(opts.legend));
	return wrap;
}

// --- Graph 1: attention by category × subcategory ---------------------------
function bucketsRegionCard(rows, allRows, { level, group, vlines = null, zoom = false, hi = null }) {
	const { reportCard, reportEmpty } = d();
	const normalize = d().normalize ? d().normalize() : false;
	const grouped = group && group !== "progression";
	const warnings = [];
	(grouped ? allRows : rows).forEach((r) => validateBuckets(r.a, r, warnings));
	const catLegend = [
		{ label: "variables", color: REGION_CAT_COLOR.variables },
		{ label: "text", color: REGION_CAT_COLOR.text },
		{ label: "completion", color: REGION_CAT_COLOR.completion },
		{ label: "template", color: REGION_CAT_COLOR.other },
	];
	const layersFrom = (grid) => REGION_SUBS.map((s, k) => ({ label: s[2], color: s[3], values: grid.map((row) => row[k]) }));
	let chart;
	let sub = `${normalize ? "per-token density" : "attention share"} · area = subcategory · color = category`;
	if (grouped) {
		const agg = _regionByGroup(allRows, group, { normalize });
		if (!agg) return markWide(reportCard("attention by category", null, reportEmpty("no computed steps carry a bucket view"), _diag(warnings)));
		const gx = _groupXOpts(agg);
		chart = stackAreaChart(layersFrom(agg.grid), gx.xs, {
			xLabel: gx.xLabel, xFmt: gx.xFmt, catLabels: gx.catLabels, bars: gx.bars,
			yFmt: normalize ? _densFmt : pctFmt, share: !normalize, sideLabels: true, height: 176, legend: catLegend,
		});
		sub += ` · one ${gx.bars ? "bar" : "column"} per ${group === "step" ? "step" : group}`;
	} else {
		const agg = _regionProgression(rows, { normalize, zoom });
		if (!agg) return markWide(reportCard("attention by category", null, reportEmpty("compute this selection to see the category breakdown"), _diag(warnings)));
		const mode = curX(), xs = _progXs(agg.G, agg.meanNq, mode);
		const shadeFrom = agg.outFrac ? xs[Math.round(agg.outFrac * (agg.G - 1))] : null; // 0 (zoomed) → no shade
		chart = stackAreaChart(layersFrom(agg.grid), xs, {
			xLabel: (zoom ? "output " : "") + xLabelFor(mode), xFmt: xFmtFor(mode), yFmt: normalize ? _densFmt : pctFmt,
			share: !normalize, sideLabels: true, height: 176, legend: catLegend, vlines, hi,
			shade: shadeFrom != null ? { from: shadeFrom, to: xs[xs.length - 1] } : null,
		});
		if (zoom) sub += " · output region only";
		if (agg.n > 1) sub += ` · mean of ${agg.n} steps`;
	}
	return markWide(reportCard("attention by category × subcategory", sub, chart, _diag(warnings)));
}

// --- Graph 1b: organized prompt sections (per-XML-tag) ----------------------
// Drill-down of Graph 1's "organized" area: which <tag> section (e.g. <output>,
// <judging_criteria>) draws attention. The y-axis AUTO-FITS (not capped at 100%)
// because organized sections are a small slice of total attention — capping would
// squash them to an invisible sliver at the axis.
function bucketsSectionCard(rows, allRows, { level, group, vlines = null, zoom = false, hi = null }) {
	const { reportCard, reportEmpty } = d();
	const normalize = d().normalize ? d().normalize() : false;
	const grouped = group && group !== "progression";
	const agg = grouped ? _sectionByGroup(allRows, group, { normalize }) : _sectionProgression(rows, { normalize, zoom });
	if (!agg || !agg.tags.length) return markWide(reportCard("prompt sections · organized <tags>", null, reportEmpty("no organized <tag> sections carry attention in this selection")));
	// 100%-stacked COMPOSITION: normalize each x-column so its <tag> sections sum to 1.
	// The raw stack summed to the total ORGANIZED attention (which drifts with position
	// / step / normalize toggle), so you couldn't compare the section MIX across columns
	// — every column has a different total height. Sharing to 1 answers "OF the attention
	// organized text draws here, how is it split across <output>/<judging_criteria>/…",
	// which is comparable everywhere. A column with no organized attention stays empty.
	const grid = agg.grid.map((row) => { const s = row.reduce((a, v) => a + v, 0); return s > 1e-12 ? row.map((v) => v / s) : row.map(() => 0); });
	const layers = agg.tags.map((t, k) => ({ label: `<${t}>`, color: sectionColor(t, agg.tags), values: grid.map((row) => row[k]) }));
	const gx = grouped ? _groupXOpts(agg) : null;
	const xs = grouped ? gx.xs : _progXs(agg.G, agg.meanNq, curX());
	const shadeFrom = (!grouped && agg.outFrac) ? xs[Math.round(agg.outFrac * (agg.G - 1))] : null; // 0 (zoomed) → no shade
	const chart = stackAreaChart(layers, xs, {
		xLabel: grouped ? gx.xLabel : (zoom ? "output " : "") + xLabelFor(curX()),
		xFmt: grouped ? gx.xFmt : xFmtFor(curX()), catLabels: grouped ? gx.catLabels : null, bars: grouped ? gx.bars : false,
		yFmt: pctFmt, share: true, yMax: 1, sideLabels: true, height: 176,
		shade: shadeFrom != null ? { from: shadeFrom, to: xs[xs.length - 1] } : null,
		vlines: grouped ? null : vlines, hi: grouped ? null : hi,
		legend: layers.map((L) => ({ label: L.label, color: L.color })),
	});
	let sub = grouped
		? `composition · each ${gx.bars ? "bar" : "column"}'s <tag> sections sum to 100% · per ${group === "step" ? "step" : group}`
		: "composition · <tag> sections sum to 100% at each position (section mix, not magnitude)";
	if (!grouped) { if (zoom) sub += " · output region only"; if (agg.n > 1) sub += ` · mean of ${agg.n} steps`; }
	return markWide(reportCard("prompt sections · organized <tags>", sub, chart));
}

// Peer comparison: one organized-<tag> stack per selection, colors shared across
// peers (union tag order) so the same section reads the same hue everywhere.
function bucketsSectionCompareCard(peers) {
	const { el, reportCard, reportEmpty } = d();
	const normalize = d().normalize ? d().normalize() : false;
	const allTags = [...new Set(peers.flatMap((p) => _sectionTags(p.rows)))].sort();
	const cols = [];
	for (const p of peers) {
		const agg = _sectionProgression(p.rows, { normalize, tags: allTags });
		if (!agg) continue;
		// Composition (column sums to 1), same as the single graph — so the section MIX
		// is comparable across peers, not confounded by each peer's organized total.
		const grid = agg.grid.map((row) => { const s = row.reduce((a, v) => a + v, 0); return s > 1e-12 ? row.map((v) => v / s) : row.map(() => 0); });
		cols.push(el("div", { style: "flex:1;min-width:220px" },
			el("div", { class: "graph-card-sub", style: "margin:0 0 2px 2px", text: p.label },
				el("span", { class: "frame-sw", style: `background:${p.color};margin-left:6px` })),
			stackAreaChart(agg.tags.map((t, k) => ({ label: `<${t}>`, color: sectionColor(t, allTags), values: grid.map((row) => row[k]) })),
				_progXs(agg.G, agg.meanNq, curX()),
				{ xLabel: xLabelFor(curX()), xFmt: xFmtFor(curX()), yFmt: pctFmt, share: true, yMax: 1, height: 150, sideLabels: false })));
	}
	if (!cols.length) return markWide(reportCard("prompt sections · organized <tags> · compared", null, reportEmpty("no compared selection carries organized sections")));
	return markWide(reportCard("prompt sections · organized <tags> · compared",
		"composition · each stack's <tag> sections sum to 100% · one stack per selection",
		el("div", { style: "display:flex;gap:14px;flex-wrap:wrap" }, ...cols),
		chartLegend(allTags.map((t) => ({ label: `<${t}>`, color: sectionColor(t, allTags) })))));
}

// --- Graph 2: word/token types (aggregate + organized/free) -----------------
function bucketsWordTypeCard(rows, allRows, { level, group, vlines = null, zoom = false, hi = null }) {
	const { el, reportCard, reportEmpty } = d();
	const normalize = d().normalize ? d().normalize() : false;
	const grouped = group && group !== "progression";
	const mode = curX();
	const warnings = [];
	(grouped ? allRows : rows).forEach((r) => validateBuckets(r.a, r, warnings));
	// colNorm = 100%-stacked composition (each x column ÷ its own total). The
	// organized/free subgraphs draw a tiny share of total attention, so without it
	// they hug the axis; normalizing makes their word-type MIX fill the height.
	const panel = (key, title, { colNorm = false, height = 176 } = {}) => {
		const agg = grouped ? _typeByGroup(allRows, group, key, { normalize }) : _typeProgression(rows, key, { normalize, zoom });
		if (!agg || !agg.names) return null;
		let grid = agg.grid;
		if (colNorm) grid = grid.map((row) => { const s = row.reduce((a, v) => a + v, 0); return s > 1e-12 ? row.map((v) => v / s) : row.map(() => 0); });
		const layers = agg.names.map((nm, k) => ({ label: nm, color: TYPE_STYLE[nm] || "#8a8f98", values: grid.map((row) => row[k]) }));
		const gx = grouped ? _groupXOpts({ ...agg, grid }) : null;
		const xs = grouped ? gx.xs : _progXs(agg.G, agg.meanNq, mode);
		const chart = stackAreaChart(layers, xs, {
			xLabel: grouped ? gx.xLabel : (zoom ? "output " : "") + xLabelFor(mode),
			xFmt: grouped ? gx.xFmt : xFmtFor(mode), catLabels: grouped ? gx.catLabels : null, bars: grouped ? gx.bars : false,
			yFmt: colNorm ? pctFmt : normalize ? _densFmt : pctFmt,
			share: colNorm ? true : !normalize, yMax: colNorm ? 1 : undefined,
			height, sideLabels: false, vlines: grouped ? null : vlines, hi: grouped ? null : hi,
		});
		const sub = colNorm ? `${title} · composition` : title;
		return el("div", { style: "flex:1;min-width:320px" }, el("div", { class: "graph-card-sub", style: "margin:0 0 3px 2px", text: sub }), chart);
	};
	const agg = panel("type", "all text", { height: 188 });
	if (!agg) return markWide(reportCard("word / token types", null, reportEmpty("compute this selection to see word-type attention"), _diag(warnings)));
	const org = panel("type_organized", "organized text", { colNorm: true, height: 184 });
	const free = panel("type_free", "free text", { colNorm: true, height: 184 });
	const subRow = (org || free) ? el("div", { style: "display:flex;gap:16px;flex-wrap:wrap;margin-top:8px" }, org, free) : null;
	return markWide(reportCard("word / token types",
		`${normalize ? "per-token density" : "attention share"} · area = word type · structural split into tags${grouped ? ` · one column per ${group === "step" ? "step" : group}` : ""}`,
		agg, subRow, wordTypeLegend(), _diag(warnings)));
}

// Peer comparison: one region category×subcategory stacked area per selected
// step/kind/scene, side by side (fair with the normalize toggle).
function bucketsRegionCompareCard(peers) {
	const { el, reportCard, reportEmpty } = d();
	const normalize = d().normalize ? d().normalize() : false;
	const cols = [];
	for (const p of peers) {
		const agg = _regionProgression(p.rows, { normalize });
		if (agg) cols.push(el("div", { style: "flex:1;min-width:220px" },
			el("div", { class: "graph-card-sub", style: "margin:0 0 2px 2px", text: p.label },
				el("span", { class: "frame-sw", style: `background:${p.color};margin-left:6px` })),
			stackAreaChart(REGION_SUBS.map((s, k) => ({ label: s[2], color: s[3], values: agg.grid.map((row) => row[k]) })),
				_progXs(agg.G, agg.meanNq, curX()),
				{ xLabel: xLabelFor(curX()), xFmt: xFmtFor(curX()), yFmt: normalize ? _densFmt : pctFmt, share: !normalize, height: 150, sideLabels: false })));
	}
	if (!cols.length) return markWide(reportCard("attention by category · compared", null, reportEmpty("no compared selection carries a bucket view")));
	return markWide(reportCard("attention by category · compared",
		`${normalize ? "per-token density" : "attention share"} · one stack per selection`,
		el("div", { style: "display:flex;gap:14px;flex-wrap:wrap" }, ...cols),
		chartLegend([{ label: "variables", color: REGION_CAT_COLOR.variables }, { label: "text", color: REGION_CAT_COLOR.text }, { label: "completion", color: REGION_CAT_COLOR.completion }])));
}

export function renderTokenOrdering(rows, { level = "step", cmp = null, allRows = null, group = "progression" } = {}) {
	const { reportCard, reportEmpty, hasToPlace } = d();
	if (cmp && cmp.active) return renderTokenOrderingCompare(cmp).map(markWide);
	rows = (rows || []).filter((r) => r && r.a);
	if (!rows.length) return [markWide(reportCard("token ordering", null, reportEmpty("compute this selection's attention to see token-ordering analyses")))];
	const full = (allRows && allRows.length ? allRows : rows).filter((r) => r && r.a);
	// "per feature" (single step) is its own layout: the category / section / word-type
	// graphs are token-ordered and can't bin by output item, so we lead with the
	// per-feature histogram instead of showing them.
	if (level === "step" && featureMode()) {
		const feat = [perFeatureCard(rows[0].a), contextOrderCard(rows)];
		if (hasToPlace(rows[0].a)) feat.push(toPlaceOrderCard(rows), inOutOrderCard(rows));
		return feat.map(markWide);
	}
	// Output zoom trims the within-step graphs to just the output region (drop the
	// reasoning prefix). Only applies at the single-step level.
	const zoom = level === "step" && (d().outputZoom ? d().outputZoom() : false);
	const opts = { level, group, zoom };
	// Output segmentation overlay — item + attribute-field boundaries computed once
	// (in the current x-mode, re-based when zoomed) and shared by every within-step
	// breakdown graph. Only meaningful at the single-step level, where the x-axis is
	// this step's own token progression; grouped (kind/zone) columns have no such axis.
	let sceneVlines = null, segNote = null, bands = null, selBand = null, xHist = null;
	if (level === "step" && (!d().segmentOutput || d().segmentOutput())) {
		const sv = _outputVlines(rows[0].a, curX(), zoom);
		if (sv) { opts.vlines = sv.prog; sceneVlines = sv.traj; segNote = sv.note; }
		// Clickable item bands (drill-down): compute once, highlight the selected item
		// across every graph (prog-space for the stacks, traj-space for the line).
		bands = _outputBands(rows[0].a, curX(), zoom);
		const sel = d().selectedSegment ? d().selectedSegment() : null;
		const selBd = sel && bands.find((b) => b.id === sel);
		if (selBd) { selBand = sel; opts.hi = [selBd.px0, selBd.px1]; }
		// Reasoning-side x-axis annotation: the reasoning text's own token-type mix.
		xHist = _reasoningTypeHist(rows[0].a, curX(), zoom);
	}
	const cards = [bucketsRegionCard(rows, full, opts), bucketsSectionCard(rows, full, opts), bucketsWordTypeCard(rows, full, opts)];
	if (level === "step") {
		cards.push(sceneMassVsPositionCard(rows, { vlines: sceneVlines, segNote, zoom, bands, selBand, onBand: d().selectSegment, xHist }));
		// Drill-down card for the clicked item, right after the click surface.
		if (selBand) { const bc = segmentBreakdownCard(rows[0].a, selBand); if (bc) cards.push(markWide(bc)); }
		cards.push(contextOrderCard(rows));
		if (hasToPlace(rows[0].a)) cards.push(toPlaceOrderCard(rows), inOutOrderCard(rows));
	} else {
		cards.push(sceneMassVsPositionCard(rows));
		if (level === "scene") cards.push(contextOrderCard(rows));
		const tpRows = rows.filter((r) => hasToPlace(r.a));
		if (tpRows.length) cards.push(toPlaceOrderCard(tpRows), inOutOrderCard(tpRows));
	}
	return cards.map(markWide);
}

// Generic SVG radar/spider: axes = [{label,color?}], series = [{label,color,
// values[]}] (values aligned to axes). Each series is L1-normalized to its SHARE
// distribution (value / Σ) — the same "shape, not magnitude" convention as the
// attribute spider — then a shared scale maps the largest share to the ring. Hover
// a vertex for the exact share.
function radarChart(axes, series, opts = {}) {
	const { el } = d();
	const A = axes.length;
	if (A < 3 || !series.length) return el("div", { class: "hint", text: opts.empty || "not enough axes to plot a radar" });
	const normed = series.map((s) => {
		const vals = axes.map((_, k) => Math.max(0, s.values[k] || 0));
		const sum = vals.reduce((a, b) => a + b, 0) || 1;
		return { label: s.label, color: s.color || "#7aa2f7", share: vals.map((v) => v / sum) };
	});
	const scale = Math.max(1e-9, ...normed.flatMap((s) => s.share));
	const W = 380, H = 360, cx = W / 2, cy = H / 2 + 4, R = 116;
	const ang = (k) => -Math.PI / 2 + (k / A) * Math.PI * 2;
	const pt = (k, r) => [cx + Math.cos(ang(k)) * R * r, cy + Math.sin(ang(k)) * R * r];
	const svg = svgEl("svg", { class: "gsvg", viewBox: `0 0 ${W} ${H}`, style: "width:100%;height:auto;display:block;overflow:visible;max-width:440px;margin:0 auto" });
	for (const rr of [0.25, 0.5, 0.75, 1]) {
		let dd = "";
		for (let k = 0; k <= A; k++) { const [x, y] = pt(k % A, rr); dd += (k ? "L" : "M") + x.toFixed(1) + "," + y.toFixed(1) + " "; }
		svg.appendChild(svgEl("path", { d: dd + "Z", fill: "none", stroke: "rgba(255,255,255,0.07)" }));
	}
	axes.forEach((ax, k) => {
		const [x, y] = pt(k, 1);
		svg.appendChild(svgEl("line", { x1: cx, y1: cy, x2: x.toFixed(1), y2: y.toFixed(1), stroke: "rgba(255,255,255,0.08)" }));
		const [lx, ly] = pt(k, 1.13);
		const anchor = Math.abs(lx - cx) < 8 ? "middle" : lx < cx ? "end" : "start";
		svg.appendChild(svgEl("text", { x: lx.toFixed(1), y: ly.toFixed(1), fill: ax.color || "rgba(220,230,245,0.75)", "font-size": 9.5, "text-anchor": anchor, "dominant-baseline": "middle" }, ax.label));
	});
	for (const s of normed) {
		let dd = "";
		s.share.forEach((v, k) => { const [x, y] = pt(k, Math.min(1, v / scale)); dd += (k ? "L" : "M") + x.toFixed(1) + "," + y.toFixed(1) + " "; });
		svg.appendChild(svgEl("path", { d: dd + "Z", fill: hexA(s.color, 0.12), stroke: s.color, "stroke-width": 1.8, "stroke-linejoin": "round" }));
		s.share.forEach((v, k) => {
			const [x, y] = pt(k, Math.min(1, v / scale));
			const dot = svgEl("circle", { cx: x.toFixed(1), cy: y.toFixed(1), r: 2.4, fill: s.color });
			dot.appendChild(svgEl("title", null, `${s.label} · ${axes[k].label}: ${(v * 100).toFixed(1)}%`));
			svg.appendChild(dot);
		});
	}
	const wrap = el("div", { class: "gwrap" }, svg);
	wrap.appendChild(chartLegend(series.map((s) => ({ label: s.label, color: s.color || "#7aa2f7" }))));
	return wrap;
}

// One polygon per ablation item over the category×subcategory axes. Falls back to
// the stacked-bar card when too few subcategories carry attention to form a radar.
function ablBreakdownRegionSpider(rows, colorMap) {
	const { reportCard, reportEmpty } = d();
	const normalize = d().normalize ? d().normalize() : false;
	const agg = _regionByGroup(rows, "abl", { normalize });
	if (!agg) return reportCard("attention by category \u00d7 subcategory", null, reportEmpty("no computed variants carry a bucket view"));
	const present = REGION_SUBS.map((_, k) => k).filter((k) => agg.grid.some((row) => row[k] > 1e-9));
	if (present.length < 3) return bucketsRegionCard(rows, rows, { level: "kind", group: "abl" });
	const axes = present.map((k) => ({ label: REGION_SUBS[k][2], color: REGION_SUBS[k][3] }));
	const series = agg.grid.map((row, i) => ({ label: agg.labels[i], color: (colorMap && colorMap.get(agg.labels[i])) || "#7aa2f7", values: present.map((k) => row[k]) }));
	return reportCard("attention by category \u00d7 subcategory", "one polygon per ablation item \u00b7 L1-normalized to each item's SHARE across subcategories (shape) \u00b7 axis = subcategory", radarChart(axes, series));
}

// A compact "treatments" legend card: the treatment → color key for the compared
// ablation levels, so every graph below reads unambiguously by color.
function ablTreatmentLegendCard(rows, colorMap) {
	const { reportCard } = d();
	const seen = new Map(); // label -> ord (peer order)
	for (const r of rows) { const l = r._ablLabel; if (l != null && !seen.has(l)) seen.set(l, r._ablOrd ?? 0); }
	if (!seen.size) return null;
	const items = [...seen.entries()].sort((a, b) => a[1] - b[1]).map(([label]) => ({ label, color: (colorMap && colorMap.get(label)) || "#7aa2f7" }));
	return reportCard("treatments", "color key for the compared ablation levels", chartLegend(items));
}

// Ablation breakdown: the SAME token-type breakdown as step/kind/scene, keyed to
// the ablation item. Category × subcategory stays a SPIDER (its SHAPE compares
// directly), while the prompt <tag> sections AND the word / token types are
// STACKED BARS — one column per ablation level — with the word/token-type bars
// LAST. Rows are pre-tagged (report.js) with _ablGroup / _ablOrd / _ablLabel;
// `colorMap` maps each level → its color.
export function renderAblationBreakdown(rows, colorMap) {
	const { reportCard, reportEmpty } = d();
	rows = (rows || []).filter((r) => r && r.a);
	if (!rows.length) return [markWide(reportCard("attention breakdown", null, reportEmpty("compute variants (⚗ ablation tab) — their token-type breakdown then compares here")))];
	return [
		ablTreatmentLegendCard(rows, colorMap),
		ablBreakdownRegionSpider(rows, colorMap),
		bucketsSectionCard(rows, rows, { level: "kind", group: "abl" }),
		bucketsWordTypeCard(rows, rows, { level: "kind", group: "abl" }),
	].filter(Boolean).map(markWide);
}

// STRUCTURE vs CONTENT: aggregate the per-attribute `attr_role` buckets (one
// segment per <attribute>|<role>, role ∈ {context, frame, content}) across each
// ablation group's rows, summing attention MASS and token COUNT separately per
// segment NAME so the density is correct across steps with different scenes.
// density = Σ mass / Σ token-count = the length-normalized mean per-token
// attention on that role's tokens — exactly "how much is it attending to
// frame / content / context, normalized to length of each".
function _attrRoleByGroup(allRows) {
	const valid = (allRows || []).filter((r) => r && r.a && r.a.buckets
		&& Array.isArray(r.a.buckets.attr_role) && (r.a.buckets.attr_role_names || []).length);
	if (!valid.length) return null;
	const groups = new Map(); // ablGroup key -> { label, ord, mass:Map(name→Σ), cnt:Map(name→Σ) }
	for (const r of valid) {
		const key = r._ablGroup ?? "all";
		let g = groups.get(key);
		if (!g) groups.set(key, g = { label: r._ablLabel ?? key, ord: r._ablOrd ?? 0, mass: new Map(), cnt: new Map() });
		const b = r.a.buckets, names = b.attr_role_names || [], counts = b.attr_role_tokens || [], grid = b.attr_role || [];
		const segMass = new Array(names.length).fill(0);
		for (const row of grid) for (let k = 0; k < row.length; k++) segMass[k] += row[k];  // sum over progression buckets
		names.forEach((nm, k) => {
			g.mass.set(nm, (g.mass.get(nm) || 0) + segMass[k]);
			g.cnt.set(nm, (g.cnt.get(nm) || 0) + (counts[k] || 0));
		});
	}
	const allNames = new Set();
	const perGroup = [...groups.values()].sort((a, b) => a.ord - b.ord).map((g) => {
		const density = new Map();
		for (const nm of g.mass.keys()) { allNames.add(nm); const c = g.cnt.get(nm) || 0; density.set(nm, c ? g.mass.get(nm) / c : 0); }
		return { label: g.label, density };
	});
	return { perGroup, names: [...allNames] };
}

// Role stack order (bottom → top) + the opacity SHADE each role is drawn at, so a
// bar's hue reads as its TREATMENT while the three segments read as the roles.
const STRUCT_ROLES = ["context", "frame", "content"];
const STRUCT_ROLE_OP = { context: 0.4, frame: 0.68, content: 1 };

// Grouped STACKED-BAR chart of the per-attribute context/frame/content split:
// x = attributes (grouped), one bar per treatment within a group, each bar stacked
// into its 3 role segments (shaded by STRUCT_ROLE_OP). Bar height = length-
// normalized attention density (mass ÷ tokens); a shared y-scale keeps bars
// comparable. Hover a segment for the exact per-token value.
function attrRoleBars(agg, colorMap) {
	const attrTotal = (attr) => agg.perGroup.reduce((s, g) => s + STRUCT_ROLES.reduce((t, r) => t + (g.density.get(`${attr}|${r}`) || 0), 0), 0);
	const attrs = [...new Set(agg.names.map((n) => n.slice(0, n.lastIndexOf("|"))))]
		.filter((a) => attrTotal(a) > 1e-9)
		.sort((a, b) => attrTotal(b) - attrTotal(a)); // most-attended attribute first
	if (!attrs.length) return null;
	const treats = agg.perGroup;
	const colorOf = (label) => (colorMap && colorMap.get(label)) || "#7aa2f7";
	let yMax = 0;
	for (const a of attrs) for (const g of treats) { const s = STRUCT_ROLES.reduce((t, r) => t + (g.density.get(`${a}|${r}`) || 0), 0); if (s > yMax) yMax = s; }
	yMax = yMax || 1;
	const barW = 14, barGap = 2, groupGap = 18, padL = 46, padR = 12, padT = 12, padB = 72, plotH = 200;
	const groupW = Math.max(barW, treats.length * (barW + barGap) - barGap);
	const plotW = Math.max(1, attrs.length * (groupW + groupGap) - groupGap);
	const W = padL + plotW + padR, H = padT + plotH + padB;
	const y0 = padT, y1 = padT + plotH;
	const Y = (v) => y1 - (Math.max(0, Math.min(yMax, v)) / yMax) * (y1 - y0);
	const svg = svgEl("svg", { class: "gsvg", viewBox: `0 0 ${W} ${H}`, style: `width:100%;max-width:${W}px;height:auto;display:block;overflow:visible` });
	for (const f of [0, 0.25, 0.5, 0.75, 1]) {
		const yy = y1 - f * (y1 - y0);
		svg.appendChild(svgEl("line", { x1: padL, y1: yy, x2: W - padR, y2: yy, stroke: "rgba(255,255,255,0.07)" }));
		svg.appendChild(svgEl("text", { x: padL - 5, y: yy, fill: "rgba(220,230,245,0.5)", "font-size": 9, "text-anchor": "end", "dominant-baseline": "middle" }, (yMax * f).toFixed(3)));
	}
	attrs.forEach((attr, ai) => {
		const gx = padL + ai * (groupW + groupGap);
		treats.forEach((g, ti) => {
			const bx = gx + ti * (barW + barGap);
			const col = colorOf(g.label);
			let acc = 0;
			for (const role of STRUCT_ROLES) {
				const v = g.density.get(`${attr}|${role}`) || 0;
				if (v > 1e-12) {
					const yTop = Y(acc + v), yBot = Y(acc);
					svg.appendChild(svgEl("rect", { class: "gpt", x: bx, y: yTop, width: barW, height: Math.max(0.6, yBot - yTop), fill: col, "fill-opacity": STRUCT_ROLE_OP[role], stroke: "rgba(13,15,20,0.55)", "stroke-width": 0.5 },
						svgEl("title", null, `${attr} · ${role}\n${g.label}: ${v.toFixed(4)} per token`)));
				}
				acc += v;
			}
		});
		svg.appendChild(svgEl("text", { x: gx + groupW / 2, y: y1 + 11, fill: "rgba(220,230,245,0.72)", "font-size": 9.5, "text-anchor": "end", transform: `rotate(-40 ${gx + groupW / 2} ${y1 + 11})` }, attr));
	});
	return svg;
}

// STRUCTURE vs CONTENT: for EACH attribute, one bar per treatment, and each bar is
// split into 3 segments — context / frame / content — by its length-normalized
// attention density. Bar hue = treatment (legend); segment shade = role. Rows are
// pre-tagged (report.js) with _ablGroup / _ablLabel / _ablOrd.
export function renderAblationStructure(rows, colorMap) {
	const { el, reportCard, reportEmpty } = d();
	rows = (rows || []).filter((r) => r && r.a);
	const agg = _attrRoleByGroup(rows);
	if (!agg) {
		return [markWide(reportCard("structure vs content", null,
			reportEmpty("compute variants (⚗ ablation tab) — the per-attribute context / frame / content attention split compares here. Needs recomputed attention (analysis v9+): the attr_role buckets ship only on freshly-computed steps.")))];
	}
	const chart = attrRoleBars(agg, colorMap);
	if (!chart) {
		return [markWide(reportCard("structure vs content", null,
			reportEmpty("not enough attributes with a well-defined context/frame/content split in scope")))];
	}
	const treatLegend = chartLegend(agg.perGroup.map((g) => ({ label: g.label, color: (colorMap && colorMap.get(g.label)) || "#7aa2f7" })));
	const roleLegend = el("div", { class: "seg-legend", style: "font-size:10px;gap:12px;margin-top:4px;flex-wrap:wrap" },
		el("span", { class: "muted", text: "segment (bottom→top):" }),
		...STRUCT_ROLES.map((r) => el("span", {}, el("span", { class: "frame-sw", style: `background:#9aa4b2;opacity:${STRUCT_ROLE_OP[r]}` }), el("span", { text: ` ${r}` }))));
	return [markWide(reportCard("structure vs content — per attribute",
		"one bar per treatment, grouped by attribute · each bar's 3 segments = its context / frame / content attention density (mass ÷ tokens, length-normalized) · bar hue = treatment, segment shade = role",
		chart, treatLegend, roleLegend))];
}

// Peer comparison: overlay each selected step/kind/scene, colored like the rest
// of the report's comparisons. To-place cards appear when any peer has bbox steps.
function renderTokenOrderingCompare(cmp) {
	const { hasToPlace } = d();
	const peers = (cmp.peers || [])
		.map((p) => ({ label: p.label, color: p.color, rows: (p.rows || []).filter((r) => r && r.a) }))
		.filter((p) => p.rows.length);
	const cards = [bucketsRegionCompareCard(peers), bucketsSectionCompareCard(peers), sceneMassCompareCard(peers), contextOrderCompareCard(peers)];
	const tpPeers = peers.map((p) => ({ ...p, rows: p.rows.filter((r) => hasToPlace(r.a)) })).filter((p) => p.rows.length);
	if (tpPeers.length) cards.push(toPlaceOrderCompareCard(tpPeers), inOutOrderCompareCard(tpPeers));
	return cards;
}
function markWide(card) { card.classList && card.classList.add("wide"); return card; }

export {
	comparePeerAttributes,
	comparePeerEntities,
	comparePeerHeads,
	attributeProfileGraph,
	compareAttributesGraph,
	placementObjectsGraph,
	xmlSplitCard,
	ablationOrderCard,
	rhoCurveCard,
	spatialScatterCard,
};
