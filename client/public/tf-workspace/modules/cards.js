// The four Data-view cards. Each takes the loaded `rows` (the selection's `agg`
// analyses) and returns a `.card` element. Charts are mounted responsively
// (chartHost) so every graph fills the actual space it's given and stays crisp.
// The reasoning card manages its own pie-slice → scatter filter (and the scatter
// log-Y toggle) in place.

import { el } from "../../js/ui.js";
import { state, COLORS, compHex, entityHex } from "./state.js";
import { poolComponents, poolKindTotals, contextPoints, poolOutputs, sectionProgression, outputSegments, progXOfToken, viiInstructions, viiScatterModel } from "./data.js";
import { spiderChart, pieChart, scatterChart, stackAreaChart, svgEl, pctFmt, chartHost, repaint, fontScale, escTip, chartLegend } from "./charts.js";
import { rhoFocusSeries, rhoCurveChart, spearmanTrend, FOCUS_THRESHOLDS } from "./ablation.js";
import { runViiReport } from "./viiReport.js";

// p-value formatter + x-binned mean line for the context-order scatter's trend overlay.
const fmtP = (p) => (p == null || !isFinite(p)) ? "" : (p < 0.001 ? "p<0.001" : `p=${p.toFixed(3)}`);
const focusLabel = (t) => {
	if (!t) return "all";
	const pct = t * 100;
	return `≥${pct.toFixed(pct < 0.1 ? 3 : pct < 1 ? 2 : 1)}%`; // finer decimals for tiny shares
};
// Min-focus slider is NONLINEAR: a raw position in [0,1] (1000 steps → very fine) maps
// through an exponential curve to an attention SHARE in [0, FOCUS_MAX]. This spends most
// of the travel/resolution on the crowded low-share range while keeping high shares
// reachable at the far end. focusFromPos / posFromFocus are exact inverses.
const FOCUS_MAX = Math.max(...FOCUS_THRESHOLDS);   // 0.15 — filter's upper bound
const FOCUS_K = 4;                                  // curvature: higher = finer near 0
const focusFromPos = (pos) => {
	const t = Math.max(0, Math.min(1, pos));
	return FOCUS_MAX * (Math.exp(FOCUS_K * t) - 1) / (Math.exp(FOCUS_K) - 1);
};
const posFromFocus = (foc) => {
	const f = Math.max(0, Math.min(FOCUS_MAX, foc || 0));
	return Math.log(1 + (f / FOCUS_MAX) * (Math.exp(FOCUS_K) - 1)) / FOCUS_K;
};
function binnedMeanLine(points, nBins = 10) {
	if (points.length < 2) return [];
	const bins = Array.from({ length: nBins }, () => ({ sx: 0, sy: 0, n: 0 }));
	for (const p of points) { let b = Math.floor(p.x * nBins); if (b >= nBins) b = nBins - 1; if (b < 0) b = 0; bins[b].sx += p.x; bins[b].sy += p.y; bins[b].n += 1; }
	return bins.filter((b) => b.n > 0).map((b) => ({ x: b.sx / b.n, y: b.sy / b.n }));
}
// Variance of attention within each x (order) bin → { x (bin center), v (variance), n }.
function binnedVar(points, nBins = 10) {
	if (points.length < 2) return [];
	const bins = Array.from({ length: nBins }, () => []);
	for (const p of points) { let b = Math.floor(p.x * nBins); if (b >= nBins) b = nBins - 1; if (b < 0) b = 0; bins[b].push(p.y); }
	const out = [];
	bins.forEach((ys, i) => {
		if (ys.length < 2) return;
		const m = ys.reduce((a, b) => a + b, 0) / ys.length;
		const v = ys.reduce((a, b) => a + (b - m) * (b - m), 0) / ys.length;
		out.push({ x: (i + 0.5) / nBins, v, n: ys.length });
	});
	return out;
}
// A tiny variance-over-order sparkline, boxed as its OWN canvas — mounted floating
// at the bottom-right of the scatter (outside the main plot canvas). x = order
// (early→late), y = variance of attention in that order bin.
function varianceInset(varPts) {
	if (varPts.length < 2) return null;
	const W = 158, H = 84, padL = 5, padR = 5, padT = 15, padB = 12;
	const px0 = padL, px1 = W - padR, py0 = padT, py1 = H - padB;
	const maxV = Math.max(...varPts.map((p) => p.v), 1e-9);
	const X = (x) => px0 + x * (px1 - px0);
	const Y = (v) => py1 - (v / maxV) * (py1 - py0);
	const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, style: "display:block" });
	svg.appendChild(svgEl("line", { x1: px0, y1: py1, x2: px1, y2: py1, stroke: "rgba(255,255,255,0.16)" }));
	const s = [...varPts].sort((a, b) => a.x - b.x);
	let dLine = "", dArea = `M${X(s[0].x).toFixed(1)},${py1.toFixed(1)} `;
	s.forEach((p, i) => { const xx = X(p.x).toFixed(1), yy = Y(p.v).toFixed(1); dLine += (i ? "L" : "M") + xx + "," + yy + " "; dArea += "L" + xx + "," + yy + " "; });
	dArea += `L${X(s[s.length - 1].x).toFixed(1)},${py1.toFixed(1)} Z`;
	svg.appendChild(svgEl("path", { d: dArea, fill: "rgba(255,43,43,0.16)", stroke: "none" }));
	svg.appendChild(svgEl("path", { d: dLine.trim(), fill: "none", stroke: "#ff2b2b", "stroke-width": 1.6, "stroke-linejoin": "round", "stroke-linecap": "round" }));
	for (const p of s) { const dot = svgEl("circle", { cx: X(p.x).toFixed(1), cy: Y(p.v).toFixed(1), r: 2, fill: "#ff2b2b" }); dot.appendChild(svgEl("title", null, `order ${(p.x * 100).toFixed(0)}% · var ${p.v.toFixed(4)} · n=${p.n}`)); svg.appendChild(dot); }
	svg.appendChild(svgEl("text", { x: px0, y: 10, fill: "rgba(220,230,245,0.82)", "font-size": 9.5, "font-weight": 600 }, "variance over order"));
	svg.appendChild(svgEl("text", { x: px0, y: H - 2.5, fill: "rgba(220,230,245,0.5)", "font-size": 8 }, "early"));
	svg.appendChild(svgEl("text", { x: px1, y: H - 2.5, fill: "rgba(220,230,245,0.5)", "font-size": 8, "text-anchor": "end" }, "late"));
	return el("div", { style: "display:inline-block;background:rgba(13,15,20,0.55);border:1px solid rgba(255,255,255,0.13);border-radius:7px;padding:3px 5px 1px" }, svg);
}

function card(title, sub, ...body) {
	const head = el("div", { class: "card-head" }, el("span", { class: "card-title", text: title }));
	if (sub) head.appendChild(el("span", { class: "card-sub", text: sub }));
	return el("div", { class: "card" }, head, el("div", { class: "card-body" }, ...body.filter(Boolean)));
}
const empty = (msg) => el("div", { class: "empty", text: msg });
// Chart height as a fraction of the viewport, clamped — so a graph fills the
// screen on a big monitor but never becomes a tiny sliver or an endless column.
const vh = (frac, min, max) => Math.max(min, Math.min(Math.round(window.innerHeight * frac), max));

// --- 1. attribute spider -----------------------------------------------------
export function spiderCard(rows) {
	const comps = poolComponents(rows);
	if (!comps.length) return card("attribute breakdown", null, empty("no attribute attention in this selection"));
	const profile = { label: "attention", color: "#7aa2f7", map: new Map(comps.map((c) => [c.component, c.score])) };
	const sub = rows.length === 1 ? "one step" : `mean across ${rows.length} steps`;
	// canvas ring fills its column, capped so it never dwarfs the viewport height
	const host = chartHost((w) => spiderChart([profile], { size: Math.min(w, vh(0.62, 300, 560)) }), (w) => w);
	return card("attribute breakdown", `${sub} · hover an axis for its value`, host);
}

// --- 2. composition (pie) + context-order scatter ---------------------------
const KIND_ORDER = [
	{ key: "zone", label: "zones", color: COLORS.zone },
	{ key: "object", label: "objects", color: COLORS.object },
	{ key: "frame", label: "frames", color: COLORS.frame },
];
// Which generation region the composition is measured over.
const COMP_REGIONS = [
	{ key: "reasoning", label: "reasoning" },
	{ key: "output", label: "output" },
	{ key: "scene", label: "both" },
];
export function compositionCard(rows) {
	const c = card("composition", rows.length === 1 ? "one step" : `${rows.length} steps`);
	const body = c.querySelector(".card-body");

	function rebuild() {
		const region = state.compRegion;
		const kt = poolKindTotals(rows, region);
		const slices = KIND_ORDER.map((k) => ({ key: k.key, label: k.label, color: k.color, value: kt[k.key] || 0 }));
		const pts = contextPoints(rows, region);

		// reasoning / output / both toggle
		const regTog = el("div", { class: "seg-ctl" }, ...COMP_REGIONS.map((r) => el("button", {
			class: `seg-btn${region === r.key ? " on" : ""}`, text: r.label,
			title: `measure attention during the ${r.key === "scene" ? "whole generation (reasoning + output)" : r.key}`,
			onclick: () => { state.compRegion = r.key; try { localStorage.setItem("tf-comp-region", r.key); } catch { /* ignore */ } rebuild(); },
		})));
		const toolbar = el("div", { class: "comp-toolbar" }, el("span", { class: "muted", text: "attention during" }), regTog);

		const pie = slices.some((s) => s.value > 0)
			? chartHost((w) => pieChart(slices, { active: state.pieFilter, onSlice: (k) => { state.pieFilter = k; rebuild(); }, size: Math.min(w, vh(0.4, 220, 340)) }), (w) => w)
			: empty("no scene attention to compose here");

		// Min-focus SLIDER (nonlinear, continuous): keep only entities that drew ≥ this
		// SHARE of the step's attention. The raw slider value is a position in [0,1] that
		// maps through focusFromPos — fine near 0, coarse near the top. Dragging re-renders
		// only the scatter ZONE below, so the slider isn't recreated mid-drag.
		const focusVal = el("span", { class: "muted", style: "min-width:54px", text: focusLabel(state.compFocus || 0) });
		let focusRaf = 0; // coalesce the scatter re-render to one per frame while dragging
		const slider = el("input", {
			type: "range", min: "0", max: "1", step: "0.001", value: String(posFromFocus(state.compFocus || 0)),
			style: "width:150px;accent-color:#ff2b2b;vertical-align:middle",
			title: "minimum attention share to keep an entity — drag to filter (fine near 0)",
			oninput: (e) => {
				state.compFocus = focusFromPos(Number(e.target.value));
				focusVal.textContent = focusLabel(state.compFocus); // instant label
				if (focusRaf) return;
				focusRaf = requestAnimationFrame(() => { focusRaf = 0; renderScatterZone(); });
			},
			onchange: () => { try { localStorage.setItem("tf-comp-focus", String(state.compFocus)); } catch { /* ignore */ } },
		});
		const logBtn = el("button", {
			class: `mini-toggle${state.scatterLogY ? " on" : ""}`, text: "log y",
			title: "log-scale the attention axis (spreads the many small values)",
			onclick: () => { state.scatterLogY = !state.scatterLogY; try { localStorage.setItem("tf-scatter-logy", state.scatterLogY ? "1" : "0"); } catch { /* ignore */ } renderScatterZone(); },
		});
		const focusRow = el("div", { class: "scatter-cap", style: "flex-wrap:wrap;gap:8px;align-items:center" },
			el("span", { class: "muted", text: "min focus" }), slider, focusVal, logBtn);

		// The scatter + caption + variance inset — the only part that changes as the
		// focus slider / log toggle move (the slider row above stays put).
		const scatterZone = el("div", {});
		function renderScatterZone() {
			const focus = state.compFocus || 0;
			const kindFiltered = state.pieFilter ? pts.filter((p) => p.kind === state.pieFilter) : pts;
			const filtered = focus > 0 ? kindFiltered.filter((p) => p.share >= focus) : kindFiltered;
			const scatterPts = filtered.map((p) => ({
				x: p.x, y: p.y, label: `${p.id} · ${p.kind}`,
				color: p.kind === "zone" ? COLORS.zone : p.kind === "frame" ? COLORS.frame : COLORS.object, r: 3.2,
			}));
			// x-binned mean trend line + Spearman ρ(order, attention) with two-tailed p.
			const binLine = binnedMeanLine(filtered, 10);
			const tr = spearmanTrend(filtered.map((p) => p.x), filtered.map((p) => p.y));
			const corr = filtered.length >= 3 ? `ρ=${tr.rho.toFixed(2)} · ${fmtP(tr.p)} · n=${tr.n}` : null;
			const cap = el("div", { class: "scatter-cap" },
				el("span", { class: "muted", text: `attention vs order${state.pieFilter ? ` · ${state.pieFilter}s` : ""}${focus > 0 ? ` · ${focusLabel(focus)}` : ""} · ${scatterPts.length} entities${corr ? ` · Spearman ${corr}` : ""}` }));
			const yFmt = state.scatterLogY ? (v) => (v >= 0.1 ? v.toFixed(1) : v >= 0.01 ? v.toFixed(2) : v.toFixed(3)) : (v) => v.toFixed(1);
			const scatter = scatterPts.length >= 2
				? chartHost((w, h) => scatterChart(scatterPts, {
					width: w, height: h, yMax: 1, xMax: 1, logY: state.scatterLogY,
					xLabel: "order in scene context  (0 = first → 1 = last)",
					xFmt: (v) => v.toFixed(1), yFmt,
					line: binLine.length >= 2 ? { points: binLine, color: "#ff2b2b", width: 2.5 } : null,
					corrLabel: corr ? `Spearman ${corr}` : null,
					legend: state.pieFilter ? null : KIND_ORDER.map((k) => ({ key: k.key, label: k.label, color: k.color })),
				}), () => vh(0.5, 300, 520))
				: empty("not enough positioned entities for the order scatter");
			// Variance-over-order mini-plot as its OWN boxed canvas at the bottom-right.
			const varInset = scatterPts.length >= 2 ? varianceInset(binnedVar(filtered, 10)) : null;
			const varRow = varInset ? el("div", { style: "display:flex;justify-content:flex-end;margin-top:6px" }, varInset) : null;
			scatterZone.replaceChildren(cap, scatter, ...(varRow ? [varRow] : []));
		}
		renderScatterZone();

		body.replaceChildren(toolbar, el("div", { class: "split" }, el("div", {}, pie), el("div", {}, focusRow, scatterZone)));
	}
	rebuild();
	return c;
}

// --- 2b. ρ vs focus (position↔attention correlation as focus rises) ----------
// Companion to the composition scatter: measured over the whole generation
// ("scene") so it stands independent of the composition region toggle. Uses the
// same builder the ablation view uses, but with a single pooled line.
export function rhoFocusCard(rows) {
	const series = rhoFocusSeries(rows, "scene");
	if (!series.length) return card("\u03c1 vs focus", null, empty("not enough positioned objects to correlate"));
	const body = chartHost((w, h) => rhoCurveChart(series, { width: w, height: h }), () => vh(0.42, 280, 460));
	return card("\u03c1 vs focus", "does earlier-in-context mean more attention? \u03c1 as you keep only higher-attention objects", body);
}

// --- 3. output graph (bins = emitted objects, sub-bins = attributes) ---------
// Segment-output view: one bin per emitted object, stacked by its attributes.
// Hovering a bin fisheye-enlarges it (neighbors compress) so the attribute
// sub-bins are big enough to read/select; clicking a bin pins its full breakdown
// (every attribute's exact value + the scene entities it attended) below.
function buildOutputSvg(items, maxTotal, w, h, foc, hooks) {
	const fs = fontScale(w), F = (b) => +(b * fs).toFixed(1);
	const MINBIN = Math.round(26 + 4 * fs), padL = Math.round(38 + 12 * fs), padR = 14, padT = Math.round(10 + 6 * fs), padB = Math.round(66 + 36 * fs);
	const nb = items.length;
	const svgW = Math.max(w, padL + padR + nb * MINBIN);
	const plotW = svgW - padL - padR;
	let widths;
	if (foc == null || foc < 0 || nb === 1) {
		widths = items.map(() => plotW / nb);
	} else {
		const FW = Math.max(MINBIN, Math.min(plotW * 0.55, 340, plotW - (nb - 1) * 6));
		const others = nb > 1 ? Math.max(6, (plotW - FW) / (nb - 1)) : plotW;
		widths = items.map((_, i) => (i === foc ? FW : others));
	}
	const plotH = Math.max(160, h - padT - padB), H = padT + plotH + padB;
	const svg = svgEl("svg", { class: "out-svg", viewBox: `0 0 ${svgW} ${H}`, width: svgW, height: H });
	for (const f of [0, 0.25, 0.5, 0.75, 1]) {
		const yy = padT + plotH - f * plotH;
		svg.appendChild(svgEl("line", { x1: padL, y1: yy, x2: svgW - padR, y2: yy, stroke: "rgba(255,255,255,0.06)" }));
		svg.appendChild(svgEl("text", { x: padL - 7, y: yy, fill: "rgba(220,230,245,0.62)", "font-size": F(12), "text-anchor": "end", "dominant-baseline": "middle" }, (maxTotal * f).toFixed(maxTotal < 1 ? 2 : maxTotal < 10 ? 1 : 0)));
	}
	svg.appendChild(svgEl("text", { x: 13, y: padT + plotH / 2, fill: "rgba(220,230,245,0.82)", "font-size": F(12.5), "text-anchor": "middle", transform: `rotate(-90 13 ${padT + plotH / 2})` }, "attention"));
	let x = padL;
	items.forEach((it, i) => {
		const bw = widths[i], focused = i === foc;
		const g = svgEl("g", { class: `out-bin${focused ? " foc" : ""}` });
		g.appendChild(svgEl("title", null, `${it.id}${it.step ? ` · ${it.step}` : ""} · ${it.n} tok · total attention ${it.total.toFixed(3)}${focused ? "" : " · hover to enlarge, click to pin"}`));
		// full-column transparent hit area (so gaps above the stack still hover/click)
		g.appendChild(svgEl("rect", { x: x.toFixed(1), y: padT, width: bw.toFixed(1), height: plotH, fill: "transparent" }));
		let yBase = padT + plotH;
		for (const a of it.attrs) {
			const hgt = (a.score / maxTotal) * plotH;
			if (hgt <= 0.2) continue;
			const segY = yBase - hgt;
			const rect = svgEl("rect", { class: "out-seg", x: (x + 1.5).toFixed(1), y: segY.toFixed(1), width: Math.max(1, bw - 3).toFixed(1), height: hgt.toFixed(1), fill: compHex(a.component), "fill-opacity": 0.9 });
			rect.appendChild(svgEl("title", null, `${it.id} · ${a.component} = ${a.score.toFixed(4)} (${((a.score / (it.total || 1)) * 100).toFixed(0)}% of its attention)`));
			g.appendChild(rect);
			// attribute labels only when the bin is enlarged (readable width)
			if (focused && hgt >= F(16) && bw >= F(70)) g.appendChild(svgEl("text", { class: "out-seg-lab", x: (x + bw / 2).toFixed(1), y: (segY + hgt / 2).toFixed(1), fill: "#06070a", "font-size": F(12.5), "font-weight": 700, "text-anchor": "middle", "dominant-baseline": "middle" }, `${a.component} · ${((a.score / (it.total || 1)) * 100).toFixed(0)}%`));
			yBase = segY;
		}
		if (i < nb - 1) g.appendChild(svgEl("line", { x1: (x + bw).toFixed(1), y1: padT, x2: (x + bw).toFixed(1), y2: (padT + plotH).toFixed(1), stroke: "rgba(255,255,255,0.5)", "stroke-width": 1 }));
		if (focused || bw >= F(19)) {
			const nm = it.id.length > 24 ? it.id.slice(0, 23) + "…" : it.id;
			const lx = x + bw / 2, ly = padT + plotH + 7;
			g.appendChild(svgEl("text", { class: "out-name", x: lx.toFixed(1), y: ly.toFixed(1), "font-size": F(12), "text-anchor": "end", transform: `rotate(-38 ${lx.toFixed(1)} ${ly.toFixed(1)})` }, nm));
		}
		g.addEventListener("pointerenter", () => hooks.onEnter(i));
		g.addEventListener("click", () => hooks.onClick(i));
		svg.appendChild(g);
		x += bw;
	});
	return svg;
}
function outputDetail(it) {
	const max = Math.max(1e-9, ...it.attrs.map((a) => a.score));
	const bars = it.attrs.map((a) => {
		const pct = (a.score / (it.total || 1)) * 100;
		return el("div", { class: "od-row" },
			el("span", { class: "od-sw", style: `background:${compHex(a.component)}` }),
			el("span", { class: "od-nm", text: a.component }),
			el("span", { class: "od-bar" }, el("i", { style: `width:${(a.score / max * 100).toFixed(1)}%;background:${compHex(a.component)}` })),
			el("span", { class: "od-val", text: `${a.score.toFixed(3)} · ${pct.toFixed(0)}%` }));
	});
	const ents = (it.entities || []).length
		? el("div", {}, el("div", { class: "od-sub", text: "top attended scene entities" }),
			el("div", { class: "od-ents" }, ...it.entities.slice(0, 8).map((e) => el("span", { class: "od-ent", title: `${e.id} · ${e.score.toFixed(3)}` },
				el("span", { class: "od-ent-sw", style: `background:${entityHex(e.kind, e.id)}` }), el("span", { text: e.id })))))
		: null;
	return el("div", {},
		el("div", { class: "od-head" }, el("b", { text: it.id }), el("span", { class: "muted", text: ` · ${it.step} · ${it.n} tok · Σ ${it.total.toFixed(3)}` })),
		el("div", { class: "od-sub", text: "attention by attribute" }),
		el("div", { class: "od-bars" }, ...bars),
		ents);
}
export function outputCard(rows) {
	const items = poolOutputs(rows);
	if (!items.length) return card("output", null, empty("this selection emits no output items (e.g. a plan step) — pick a decompose / bbox step"));
	const maxTotal = Math.max(1e-9, ...items.map((it) => it.total));
	const keyOf = (it) => `${it.ev}:${it.id}`;
	let focus = null, selKey = null;
	const detail = el("div", { class: "out-detail" });
	const selIdx = () => (selKey ? items.findIndex((it) => keyOf(it) === selKey) : -1);
	const host = chartHost((w, h) => buildOutputSvg(items, maxTotal, w, h, focus != null ? focus : selIdx(), {
		onEnter: (i) => { if (focus !== i) { focus = i; repaint(host); } },
		onClick: (i) => { const k = keyOf(items[i]); selKey = selKey === k ? null : k; renderDetail(); repaint(host); },
	}), () => vh(0.5, 320, 560));
	host.style.overflowX = "auto";
	host.addEventListener("mouseleave", () => { if (focus != null) { focus = null; repaint(host); } });
	function renderDetail() {
		const i = selIdx();
		detail.replaceChildren(i < 0
			? el("div", { class: "out-detail-hint", text: "hover a column to enlarge it and read its attributes · click a column to pin its full breakdown here" })
			: outputDetail(items[i]));
	}
	renderDetail();
	const sub = rows.length === 1 ? `${items.length} output objects` : `${items.length} output objects · ${rows.length} steps`;
	return card("output", sub, host,
		el("div", { class: "out-hint", text: "each bin = one emitted object (output-zoom) · stacked by attribute · height = attention it drew · hover to enlarge, click to pin the breakdown" }),
		detail);
}

// --- 4. prompt tag breakdown (organized <tags>) -----------------------------
const SECTION_COLORS = ["#7aa2f7", "#6bd96e", "#e0a94a", "#b46aff", "#4af0e0", "#ff6b9d", "#ff9e64", "#9ece6a", "#f7768e", "#7dcfff", "#c98bdb", "#e6db74"];
export function tagsCard(rows) {
	const agg = sectionProgression(rows);
	if (!agg || !agg.tags.length) return card("prompt tag breakdown", null, empty("no organized <tag> prompt sections carry attention in this selection"));
	const grid = agg.grid.map((row) => { const s = row.reduce((a, v) => a + v, 0); return s > 1e-12 ? row.map((v) => v / s) : row.map(() => 0); });
	const layers = agg.tags.map((t, k) => ({ label: `<${t}>`, color: SECTION_COLORS[k % SECTION_COLORS.length], values: grid.map((row) => row[k]) }));
	const xs = Array.from({ length: agg.G }, (_, g) => (agg.G === 1 ? 0 : g / (agg.G - 1)) * agg.meanNq);
	// single-step segment-output overlay: object bands (hover the lane to zoom) + item/field vlines
	let bands = null, vlines = null;
	if (rows.length === 1 && (rows[0].a.tokens || []).length) {
		const a = rows[0].a, segs = outputSegments(a);
		if (segs && segs.items.length) {
			const px = progXOfToken(a), N = (a.tokens || []).length;
			bands = segs.items.map((it, k) => ({ id: it.label, label: it.label, x0: px(it.i0), x1: k + 1 < segs.items.length ? px(segs.items[k + 1].i0) : px(N - 1) }));
			vlines = [
				...segs.items.map((it) => ({ x: px(it.i0), label: it.label, major: true })),
				...segs.fields.map((f) => ({ x: px(f.i), label: f.label, major: false })),
			];
		}
	}
	let zoom = null; // { x0, x1, id } while hovering an object band in the lane
	const host = chartHost((w, h) => stackAreaChart(layers, xs, {
		width: w, height: h, share: true, yMax: 1, xLabel: "generated token", xFmt: (v) => String(Math.round(v)), yFmt: pctFmt,
		legend: layers.map((L) => ({ label: L.label, color: L.color })),
		vlines, bands, selBand: zoom ? zoom.id : null, corner: zoom ? `zoomed: ${zoom.id}` : null,
		xMin: zoom ? zoom.x0 : null, xMax: zoom ? zoom.x1 : null,
		onBand: bands ? (id) => {
			const b = id ? bands.find((x) => x.id === id) : null;
			const nz = b ? { x0: b.x0, x1: b.x1, id: b.id } : null;
			if ((nz && zoom && nz.id === zoom.id) || (!nz && !zoom)) return;
			zoom = nz; repaint(host);
		} : null,
	}), () => vh(0.46, 300, 520));
	const sub = bands ? `${bands.length} output objects · segment overlay · hover the lane to zoom` : (agg.n > 1 ? `mean of ${agg.n} steps · pick one step for the segment overlay` : "one step");
	return card("prompt tag breakdown", `${sub} · <tag> section mix over generation`, host);
}

// --- 5. per-instruction attention within VERY_IMPORTANT_INSTRUCTIONS ---------
// The "attention-saving" view: rank each instruction (a sentence/clause of the
// VII prompt block) by the mean attention share it draws, with its token cost, so
// low-share / high-cost instructions — dead weight to trim — are easy to spot. A
// "trim candidates" toggle re-sorts by attention-per-token (ascending) to surface
// them directly. Only populated once a step is recomputed under the splitting
// analysis version (⚗ VII sample in the /tf inspector).
// Dots + rank rows are colored by the STEP KIND the instruction belongs to (its
// attention is already the y axis), decoded by a shared legend. Colors are fixed
// per kind so a kind reads the same hue everywhere; unknown kinds hash into the
// generic palette. Order follows the canonical pipeline order for the legend.
const VII_KIND_ORDER = ["zone_plan", "zone_decompose", "anchor_decompose", "encapsulating_decompose", "negative_space_decompose", "object_bbox_batch", "child_bbox_batch", "next_object"];
const VII_KIND_COLORS = {
	zone_plan: "#7aa2f7", zone_decompose: "#4af0e0", anchor_decompose: "#6bd96e",
	encapsulating_decompose: "#e0a94a", negative_space_decompose: "#ff9e64",
	object_bbox_batch: "#b46aff", child_bbox_batch: "#ff6b9d", next_object: "#f7768e",
};
const VII_KIND_LABEL = {
	zone_plan: "zone plan", zone_decompose: "zone decomp", anchor_decompose: "anchor decomp",
	encapsulating_decompose: "encaps decomp", negative_space_decompose: "neg-space decomp",
	object_bbox_batch: "object bbox", child_bbox_batch: "child bbox", next_object: "next object",
};
function viiKindColor(kind) {
	if (VII_KIND_COLORS[kind]) return VII_KIND_COLORS[kind];
	let h = 0; const s = kind || "?"; for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
	return SECTION_COLORS[h % SECTION_COLORS.length];
}
const viiKindLabel = (k) => VII_KIND_LABEL[k] || (k || "?").replace(/_/g, " ");
// Distinct kinds in the data, in canonical pipeline order (unknowns appended, sorted).
function viiKindsPresent(P) {
	const set = new Set(P.map((p) => p.kind));
	return [...VII_KIND_ORDER.filter((k) => set.has(k)), ...[...set].filter((k) => !VII_KIND_ORDER.includes(k)).sort()];
}
export function viiCard(rows) {
	const TITLE = "instruction attention · VERY_IMPORTANT_INSTRUCTIONS";
	// The model (fit + z-scores) only exists once the section is split into per-
	// instruction leaves. On null, distinguish "no VII attention" from "unsplit
	// (older) result" so we can prompt a recompute in the latter case.
	const model = viiScatterModel(rows);
	if (!model) {
		const any = viiInstructions(rows).length > 0;
		return card(TITLE, null, empty(any
			? "VERY_IMPORTANT_INSTRUCTIONS isn't split into per-instruction attention for this step yet — recompute it with ⚗ VII sample in the /tf inspector."
			: "no VERY_IMPORTANT_INSTRUCTIONS section carries attention in this selection"));
	}
	const { P, max, slope, fit, ranked, marked, xMaxTok, items } = model;
	const sub = `${items.length} instruction${items.length === 1 ? "" : "s"} · ${rows.length === 1 ? "one step" : `${rows.length} steps`} · trend ${slope >= 0 ? "+" : "−"}${Math.abs(slope * 100).toFixed(2)}%/tok · ${marked.size} trim pick${marked.size === 1 ? "" : "s"}`;
	const sideOf = (w) => Math.max(300, Math.min(w, vh(0.62, 320, 600))); // square side, capped
	let query = "", selKey = null;
	const host = chartHost((w) => {
		const q = query.trim().toLowerCase();
		const side = sideOf(w);
		const pts = P.map((p) => ({ ...p, r: 4.5, color: viiKindColor(p.kind), hit: q ? p.label.toLowerCase().includes(q) : false }));
		return scatterChart(pts, {
			width: side, height: side, yMax: 1, xMax: xMaxTok,
			xLabel: "instruction length (tokens)",
			xFmt: (v) => String(Math.round(v)),
			yFmt: (v) => v.toFixed(2),
			empty: "no split instructions to plot",
			line: { points: [{ x: 0, y: fit(0) }, { x: xMaxTok, y: fit(xMaxTok) }], color: "#ff2b2b", width: 2, dash: "5 4" },
			yErr: (p) => p.ey,
			dim: q ? (p) => !p.hit : null,
			hot: (p) => (q && p.hit) || p.key === selKey,
			ring: (p) => (marked.has(p.key) ? "#39d98a" : null),
			tip: (p) => {
				const rel = Math.round(p.resid * 100); // in % of the top instruction (y is ÷ max)
				const trend = p.resid < 0 ? `<span class="below">${-rel}% below trend · ${p.z.toFixed(1)}σ</span>` : `${rel}% above trend`;
				const kindTag = `<span style="display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;background:${viiKindColor(p.kind)}"></span>${escTip(viiKindLabel(p.kind))}${p.kinds && p.kinds.length > 1 ? ` +${p.kinds.length - 1}` : ""}`;
				return `<div class="xh">${(p.share * 100).toFixed(3)}% attention · ${p.tokens} tokens · ±${(p.ey * max * 100).toFixed(3)}%</div>`
					+ `<div class="tip-trend">${kindTag} · ${marked.has(p.key) ? "★ trim candidate · " : ""}${trend}</div>`
					+ `<div class="tip-sentence">${escTip(p.label)}</div>`;
			},
		});
	}, (w) => sideOf(w));
	host.classList.add("vii-plot-host");
	// right-hand ranking by z = distance below the fit ÷ its error. Hover a row to read
	// its full text (it wraps in place); click to spotlight it on the scatter (and keep it
	// expanded via .on). Full label stays in the DOM span, so it's readable + accessible.
	const rankRows = ranked.map((p, i) => {
		const row = el("button", { class: `vii-rk${marked.has(p.key) ? " pick" : ""}`, style: `box-shadow:inset 3px 0 0 ${viiKindColor(p.kind)}` },
			el("span", { class: "vii-rk-n", text: String(i + 1) }),
			el("span", { class: `vii-rk-z${p.z > 0 ? " pos" : ""}`, text: `${p.z >= 0 ? "+" : ""}${p.z.toFixed(1)}σ` }),
			el("span", { class: "vii-rk-txt", text: p.label }));
		row.onclick = () => {
			// if the user just drag-selected text in this row, let them copy it — don't
			// treat the mouseup as a spotlight toggle (a plain click collapses the selection).
			const sel = window.getSelection();
			if (sel && sel.rangeCount && !sel.isCollapsed && row.contains(sel.anchorNode)) return;
			selKey = selKey === p.key ? null : p.key;
			rankRows.forEach((rr) => rr.classList.toggle("on", rr === row && selKey === p.key));
			repaint(host);
		};
		return row;
	});
	const rankBox = el("div", { class: "vii-rank" },
		el("div", { class: "vii-rank-head", text: "worst → best · σ below trend" }),
		el("div", { class: "vii-rank-list" }, ...rankRows));
	const main = el("div", { class: "vii-main" }, host, rankBox);
	// shared step-kind legend for both the dots and the ranked rows' left color bar
	const legend = el("div", { class: "vii-legend" },
		el("span", { class: "vii-legend-lab", text: "step kind" }),
		chartLegend(viiKindsPresent(P).map((k) => ({ key: k, label: viiKindLabel(k), color: viiKindColor(k) }))));
	const count = el("span", { class: "vii-search-n" });
	const search = el("input", { class: "vii-search", type: "search", placeholder: "search instructions to highlight on the plot…", spellcheck: "false" });
	const sync = () => {
		const q = query.trim().toLowerCase();
		const m = q ? P.filter((p) => p.label.toLowerCase().includes(q)).length : 0;
		count.textContent = q ? `${m} match${m === 1 ? "" : "es"}` : "";
		repaint(host);
	};
	search.oninput = (e) => { query = e.target.value; sync(); };
	const ico = svgEl("svg", { viewBox: "0 0 16 16", class: "vii-search-ico", width: 14, height: 14 });
	ico.appendChild(svgEl("path", { d: "M6.5 1a5.5 5.5 0 0 1 4.38 8.82l3.9 3.9-1.06 1.06-3.9-3.9A5.5 5.5 0 1 1 6.5 1zm0 1.5a4 4 0 1 0 0 8 4 4 0 0 0 0-8z", fill: "currentColor" }));
	const reportBtn = el("button", { class: "vii-report-btn", text: "⤓ generate report", title: "build a standalone HTML report — scatter, worst instructions, and full VII text for every step kind" });
	reportBtn.onclick = () => runViiReport(reportBtn);
	const bar = el("div", { class: "vii-searchbar" }, ico, search, count, reportBtn);
	return card(TITLE, sub, bar, legend, main,
		el("div", { class: "vii-hint", text: "each dot = one instruction (sentence). x = its length (tokens) · y = attention drawn, normalized so the most-attended = 1 · whiskers = ±1 standard error · dashed line = linear fit. dots (and each ranked row's left bar) are colored by the STEP KIND the instruction belongs to — see the legend. green rings and the right-hand ranking = the instructions farthest BELOW the trend with the most certainty (score σ = distance below ÷ error) — the best candidates to trim. hover a dot for its score + text; click a ranked row to spotlight it; search to highlight matches." }));
}
