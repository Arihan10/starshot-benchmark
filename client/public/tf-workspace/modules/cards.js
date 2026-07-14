// The four Data-view cards. Each takes the loaded `rows` (the selection's `agg`
// analyses) and returns a `.card` element. Charts are mounted responsively
// (chartHost) so every graph fills the actual space it's given and stays crisp.
// The reasoning card manages its own pie-slice → scatter filter (and the scatter
// log-Y toggle) in place.

import { el } from "../../js/ui.js";
import { emittingRegion } from "../../js/events.js";
import { state, ALL, COLORS, compHex, entityHex, ATTR_AXIS_ORDER } from "./state.js";
import { poolComponents, poolKindTotals, contextPoints, poolOutputs, tagScatterModel, viiInstructions, viiScatterModel, loadAllCellProfiles, overlayKindFilter } from "./data.js";
import { spiderChart, pieChart, scatterChart, barChart, boxPlotChart, svgEl, chartHost, repaint, fontScale, escTip, chartLegend, hexA } from "./charts.js";
import { rhoFocusSeries, rhoCurveChart, spearmanTrend, FOCUS_THRESHOLDS, attrRoleTotals, STRUCT_ROLES, stackedGroupedBarChart } from "./ablation.js";
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
// Overlay mode: compare every run × model cell's attribute profile on ONE spider,
// coloured by MODEL (each model = a hue). Persisted per session; cached by the
// compared step KIND so re-renders (window drags, cell swaps) don't refetch.
let _spiderOverlay = (() => { try { return localStorage.getItem("tf-spider-overlay") === "1"; } catch { return false; } })();
let _overlayCache = null; // { key, data:{profiles,scanned,capped} }

// Evenly-spaced hue per model (max separation), stable for a given model set.
function modelColorMap(models) {
	const uniq = [...new Set(models)].sort();
	const m = new Map();
	uniq.forEach((name, i) => m.set(name, `hsl(${Math.round((i / Math.max(1, uniq.length)) * 360)}, 68%, 60%)`));
	return m;
}

export function spiderCard(rows) {
	const comps = poolComponents(rows);
	const single = comps.length ? { label: "attention", color: "#7aa2f7", map: new Map(comps.map((c) => [c.component, c.score])) } : null;
	const sub = rows.length === 1 ? "one step" : `mean across ${rows.length} steps`;
	const kind = overlayKindFilter();
	const kindTxt = kind || "whole scene";

	const bodyHost = el("div");
	const toggle = el("button", {
		class: `mini-toggle${_spiderOverlay ? " on" : ""}`,
		title: "overlay every run × model cell's attribute profile on this spider, coloured by model (compares the SAME step kind across cells)",
	}, _spiderOverlay ? "◉ compare all runs × models" : "○ compare all runs × models");

	const renderSingle = () => {
		if (!single) { bodyHost.replaceChildren(empty("no attribute attention in this selection")); return; }
		bodyHost.replaceChildren(chartHost((w) => spiderChart([single], { size: Math.min(w, vh(0.62, 300, 560)) }), (w) => w));
	};
	const renderOverlay = (data) => {
		const profiles = (data.profiles || []).filter((p) => p.map && p.map.size);
		if (!profiles.length) { bodyHost.replaceChildren(empty("no computed attention across cells for this step kind")); return; }
		const colorOf = modelColorMap(profiles.map((p) => p.model));
		const series = profiles.map((p) => ({ label: `${p.model} · ${p.run}/${p.slot} (${p.n})`, color: colorOf.get(p.model), map: p.map }));
		// half-transparent, scaled down as more polygons overlap so they stay legible
		const n = series.length;
		const fillOpacity = Math.max(0.08, Math.min(0.5, 2.0 / n));
		const host = chartHost((w) => spiderChart(series, { size: Math.min(w, vh(0.62, 300, 560)), fillOpacity, strokeOpacity: 0.85, lineWidth: 1.6, dots: n <= 8 }), (w) => w);
		const legend = chartLegend([...colorOf.entries()].map(([m, c]) => ({ key: m, label: m, color: c })));
		const note = el("div", { class: "out-hint", style: "margin:6px 0 0", text: `${profiles.length} cells · ${kindTxt} · each polygon = one cell, L1-normalized to its attribute SHARE so shapes are comparable · colour = model${data.capped ? " · capped" : ""}` });
		bodyHost.replaceChildren(host, legend, note);
	};

	const run = () => {
		if (!_spiderOverlay) { renderSingle(); return; }
		if (_overlayCache && _overlayCache.key === kind) { renderOverlay(_overlayCache.data); return; }
		bodyHost.replaceChildren(empty(`loading all cells (${kindTxt})…`));
		const token = state.loadToken;
		loadAllCellProfiles(kind, token).then((data) => {
			if (!data || token !== state.loadToken || !_spiderOverlay) return; // stale / toggled off
			_overlayCache = { key: kind, data };
			renderOverlay(data);
		}).catch(() => { if (token === state.loadToken) bodyHost.replaceChildren(empty("failed to load cells")); });
	};

	const subOf = () => _spiderOverlay ? `all cells · ${kindTxt} · model = hue` : `${sub} · hover an axis for its value`;
	let cardEl = null;
	toggle.onclick = () => {
		_spiderOverlay = !_spiderOverlay;
		try { localStorage.setItem("tf-spider-overlay", _spiderOverlay ? "1" : "0"); } catch { /* ignore */ }
		toggle.classList.toggle("on", _spiderOverlay);
		toggle.textContent = _spiderOverlay ? "◉ compare all runs × models" : "○ compare all runs × models";
		const s = cardEl && cardEl.querySelector(".card-sub");
		if (s) s.textContent = subOf();
		run();
	};

	run();
	cardEl = card("attribute breakdown", subOf(), toggle, bodyHost);
	return cardEl;
}

// --- structure vs content (per attribute) -----------------------------------
// For the MAIN sequence: split each attribute's attention into context (the key
// name) / frame (brackets · quotes · punctuation) / content (the value), pooled
// across the selection. One bar per attribute, each normalized to 100% so the
// context/frame/content PROPORTION compares. Reuses the ablation attr_role split
// (server-classified per each attribute's real serialized form).
export function structureCard(rows) {
	const mass = new Map(), toks = new Map();               // `${attr}|${role}` -> pooled mass / tokens
	for (const r of rows) {
		for (const seg of attrRoleTotals(r.a)) {
			if (String(seg.name).indexOf("|") < 0) continue; // attr_role segments only
			mass.set(seg.name, (mass.get(seg.name) || 0) + (seg.mass || 0));
			toks.set(seg.name, (toks.get(seg.name) || 0) + (seg.tokens || 0));
		}
	}
	const density = (attr, role) => { const mk = `${attr}|${role}`; const t = toks.get(mk) || 0; return t ? (mass.get(mk) || 0) / t : 0; };
	const attrSet = new Set();
	for (const k of mass.keys()) attrSet.add(k.slice(0, k.lastIndexOf("|")));
	const attrTotal = (a) => STRUCT_ROLES.reduce((t, r) => t + density(a, r.key), 0);
	const attrs = [...attrSet].filter((a) => attrTotal(a) > 1e-9).sort((a, b) => attrTotal(b) - attrTotal(a));
	if (!attrs.length) return card("structure vs content", "context / frame / content attention per attribute", empty("no attribute role-split tokens in this selection — recompute this sequence's attention (analysis v9+)"));
	const rawMass = (attr, role) => mass.get(`${attr}|${role}`) || 0;
	const cats = attrs.map((a) => ({ label: a }));
	const scope = rows.length === 1 ? "one step" : `pooled across ${rows.length} steps`;
	let normalized = true; // normalized: each bar = 100% (density proportion) · off: bar height = total attention (mass)
	const capNorm = () => normalized
		? `one bar per attribute · each = 100% of its context / frame / content attention (length-normalized mass ÷ tokens) · ${scope}`
		: `one bar per attribute · bar HEIGHT = total attention (mass) to it, split context / frame / content · ${scope}`;
	const hf = (w) => Math.round(vh(0.5, 280, 500));
	const REF = "#aeb9cc";
	const roleLegend = el("div", { class: "chart-legend", style: "gap:14px;margin-top:5px;font-size:11.5px" },
		el("span", { class: "muted", text: "segment shade (faint→solid):" }),
		...STRUCT_ROLES.map((r) => el("div", { class: "lg" }, el("span", { class: "sw", style: `background:${hexA(REF, r.op)};width:13px;height:13px` }), el("span", { text: r.label.split(" (")[0] }))));
	const host = chartHost((w, h) => {
		const v = normalized ? density : rawMass;
		const series = [{ key: "seq", label: rows.length === 1 ? "step" : `${rows.length} steps`, color: "#7aa2f7",
			values: attrs.map((a) => ({ context: v(a, "context"), frame: v(a, "frame"), content: v(a, "content") })) }];
		return stackedGroupedBarChart(w, cats, series, STRUCT_ROLES, { height: h, normalize: normalized });
	}, hf);
	const cap = el("div", { class: "out-hint", style: "margin:0", text: capNorm() });
	const toggle = el("button", { class: `mini-toggle${normalized ? " on" : ""}`,
		title: "normalized: each bar = 100% (length-normalized density) · off: bar height = total attention (mass) paid to the attribute",
		onclick: (e) => { normalized = !normalized; e.currentTarget.classList.toggle("on", normalized); cap.textContent = capNorm(); repaint(host); } }, "normalize");
	const body = el("div", {},
		el("div", { style: "display:flex;align-items:center;gap:10px;margin:0 0 8px" }, toggle, cap),
		host, roleLegend);
	return card("structure vs content", "context / frame / content attention per attribute", body);
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
// CLICKING a bin fisheye-enlarges it (neighbours compress) AND pins its full
// breakdown below; clicking again collapses. X labels are laid out vertically and
// the chart grows to fit them; a sort control reorders / groups the bins (by
// height, prefix, zone) or filters them (search), with tiered grouping brackets
// drawn below the labels for prefix / zone.

// The grouping PATH for an item under a mode: prefix → underscore tokens minus the
// leaf (its own last segment); zone → the ancestor-zone chain (root → region).
const _prefixPath = (id) => String(id || "").split("_").filter(Boolean).slice(0, -1);
function _zonePath(id) {
	const m = state.obs;
	if (!m || !m.nodes) return [];
	const chain = [];
	let region = emittingRegion(m, id);
	let hops = 0;
	while (region && m.nodes.has(region) && hops < 64) { chain.push(region); region = m.nodes.get(region).parentId; hops += 1; }
	return chain.reverse();
}
const _pathFor = (it, mode) => (mode === "zone" ? (it._zone || []) : mode === "prefix" ? (it._prefix || []) : []);

// Tiered grouping brackets: for the sorted `items`, at each depth L build the
// contiguous runs that share the same path prefix (and actually reach depth L). A
// deeper run whose span duplicates a shallower one is dropped (no redundant single-
// child bracket). Returns [ [ {i0,i1,label} ] per tier ], capped at MAX_TIERS.
const MAX_TIERS = 3;
function computeTiers(items, mode) {
	if (mode !== "prefix" && mode !== "zone") return [];
	const paths = items.map((it) => _pathFor(it, mode));
	const depth = Math.min(MAX_TIERS, paths.reduce((mx, p) => Math.max(mx, p.length), 0));
	const tiers = [];
	const spansAbove = new Set(); // "i0:i1" spans already drawn at a shallower tier
	for (let L = 0; L < depth; L++) {
		const groups = [];
		let start = -1, key = null;
		const flush = (end) => { if (key != null && start >= 0) groups.push({ i0: start, i1: end, label: paths[start][L] }); };
		for (let i = 0; i < items.length; i++) {
			const p = paths[i];
			const k = p.length > L ? p.slice(0, L + 1).join("\u0000") : null;
			if (k !== key) { flush(i - 1); key = k; start = k != null ? i : -1; }
		}
		flush(items.length - 1);
		const kept = groups.filter((grp) => !spansAbove.has(`${grp.i0}:${grp.i1}`)); // drop redundant single-child tiers
		if (kept.length) { tiers.push(kept); for (const grp of kept) spansAbove.add(`${grp.i0}:${grp.i1}`); }
	}
	return tiers;
}

// Chart metrics that BOTH the height function and the SVG builder must agree on —
// the bottom area (padB) grows with the longest vertical label + the grouping
// bracket tiers, so the graph height adapts to the labels.
function outMetrics(items, mode, w) {
	const fs = fontScale(w), F = (b) => +(b * fs).toFixed(1);
	const padT = Math.round(10 + 6 * fs), padL = Math.round(38 + 12 * fs), padR = 14;
	const maxChars = Math.min(46, items.reduce((mx, it) => Math.max(mx, String(it.id || "").length), 3));
	const labelH = maxChars * F(6.6) + F(6);         // vertical label extent
	const tiers = computeTiers(items, mode);
	const TIER_H = F(22);
	const bracketH = tiers.length ? (F(8) + tiers.length * TIER_H) : 0;
	const padB = Math.round(F(8) + labelH + bracketH + F(8));
	return { fs, F, padT, padL, padR, labelH, tiers, TIER_H, padB };
}

function buildOutputSvg(items, maxTotal, w, h, foc, mode, hooks) {
	const { fs, F, padT, padL, padR, labelH, tiers, TIER_H, padB } = outMetrics(items, mode, w);
	const MINBIN = Math.round(26 + 4 * fs);
	const nb = items.length;
	const svgW = Math.max(w, padL + padR + nb * MINBIN);
	const plotW = svgW - padL - padR;
	let widths;
	if (foc == null || foc < 0 || nb === 1) {
		widths = items.map(() => plotW / nb);
	} else {
		const FWbase = Math.max(MINBIN, Math.min(plotW * 0.55, 340, plotW - (nb - 1) * 6));
		// expand-no-shrink: the enlarged bar is never narrower than its uniform width.
		const FW = Math.max(FWbase, plotW / nb);
		const others = nb > 1 ? Math.max(6, (plotW - FW) / (nb - 1)) : plotW;
		widths = items.map((_, i) => (i === foc ? FW : others));
	}
	const plotH = Math.max(150, h - padT - padB), H = padT + plotH + padB;
	const axisY = padT + plotH;
	const svg = svgEl("svg", { class: "out-svg", viewBox: `0 0 ${svgW} ${H}`, width: svgW, height: H });
	for (const f of [0, 0.25, 0.5, 0.75, 1]) {
		const yy = axisY - f * plotH;
		svg.appendChild(svgEl("line", { x1: padL, y1: yy, x2: svgW - padR, y2: yy, stroke: "rgba(255,255,255,0.06)" }));
		svg.appendChild(svgEl("text", { x: padL - 7, y: yy, fill: "rgba(220,230,245,0.62)", "font-size": F(12), "text-anchor": "end", "dominant-baseline": "middle" }, (maxTotal * f).toFixed(maxTotal < 1 ? 2 : maxTotal < 10 ? 1 : 0)));
	}
	svg.appendChild(svgEl("text", { x: 13, y: padT + plotH / 2, fill: "rgba(220,230,245,0.82)", "font-size": F(12.5), "text-anchor": "middle", transform: `rotate(-90 13 ${padT + plotH / 2})` }, "attention"));
	// SEARCH divider geometry: the x of the seam between the matched items (moved to the
	// front) and the rest — derived from the (possibly fisheye) widths so it lands exactly
	// on the boundary. A faint green band tints the matched cohort behind its bars.
	const bnd = hooks && hooks.boundary != null ? hooks.boundary : -1;
	let bndX = null;
	if (bnd > 0 && bnd < nb) {
		bndX = padL; for (let i = 0; i < bnd; i++) bndX += widths[i];
		svg.appendChild(svgEl("rect", { x: padL.toFixed(1), y: padT.toFixed(1), width: (bndX - padL).toFixed(1), height: plotH.toFixed(1), fill: "rgba(57,217,138,0.07)" }));
	}
	const xs = []; // per-item {x0, x1, cx} — brackets span these (respects the fisheye widths)
	let x = padL;
	items.forEach((it, i) => {
		const bw = widths[i], focused = i === foc;
		xs.push({ x0: x, x1: x + bw, cx: x + bw / 2 });
		const g = svgEl("g", { class: `out-bin${focused ? " foc" : ""}` });
		g.appendChild(svgEl("title", null, `${it.id}${it.step ? ` · ${it.step}` : ""} · ${it.n} tok · total attention ${it.total.toFixed(3)} · click to ${focused ? "collapse" : "expand"}`));
		// full-column transparent hit area (so gaps above the stack still click)
		g.appendChild(svgEl("rect", { x: x.toFixed(1), y: padT, width: bw.toFixed(1), height: plotH, fill: "transparent" }));
		let yBase = axisY;
		for (const a of it.attrs) {
			const hgt = (a.score / maxTotal) * plotH;
			if (hgt <= 0.2) continue;
			const segY = yBase - hgt;
			const rect = svgEl("rect", { class: "out-seg", x: (x + 1.5).toFixed(1), y: segY.toFixed(1), width: Math.max(1, bw - 3).toFixed(1), height: hgt.toFixed(1), fill: compHex(a.component), "fill-opacity": 0.9 });
			rect.appendChild(svgEl("title", null, `${it.id} · ${a.component} = ${a.score.toFixed(4)} (${((a.score / (it.total || 1)) * 100).toFixed(0)}% of its attention)`));
			g.appendChild(rect);
			if (focused && hgt >= F(16) && bw >= F(70)) g.appendChild(svgEl("text", { class: "out-seg-lab", x: (x + bw / 2).toFixed(1), y: (segY + hgt / 2).toFixed(1), fill: "#06070a", "font-size": F(12.5), "font-weight": 700, "text-anchor": "middle", "dominant-baseline": "middle" }, `${a.component} · ${((a.score / (it.total || 1)) * 100).toFixed(0)}%`));
			yBase = segY;
		}
		if (i < nb - 1) g.appendChild(svgEl("line", { x1: (x + bw).toFixed(1), y1: padT, x2: (x + bw).toFixed(1), y2: axisY.toFixed(1), stroke: "rgba(255,255,255,0.5)", "stroke-width": 1 }));
		// vertical x-axis label (the chart height was sized to fit these)
		const lx = xs[i].cx, ly = axisY + F(8);
		const nm = it.id.length > 46 ? it.id.slice(0, 45) + "…" : it.id;
		const gLab = svgEl("g", { transform: `translate(${lx.toFixed(1)},${ly.toFixed(1)}) rotate(90)` });
		gLab.appendChild(svgEl("text", { class: "out-name", "text-anchor": "start", "dominant-baseline": "middle", "font-size": F(11) }, nm));
		g.appendChild(gLab);
		g.addEventListener("click", () => hooks.onClick(i));
		svg.appendChild(g);
		x += bw;
	});
	// SEARCH divider: a bold green seam from the plot top down through the labels, tagged
	// with the match count — matched objects are LEFT of it, the rest of the scene continue
	// to the right (same height scale) so the two cohorts are directly comparable.
	if (bndX != null) {
		svg.appendChild(svgEl("line", { x1: bndX.toFixed(1), y1: (padT - F(2)).toFixed(1), x2: bndX.toFixed(1), y2: (axisY + F(6) + labelH).toFixed(1), stroke: "#39d98a", "stroke-width": 2.5 }));
		svg.appendChild(svgEl("text", { x: (bndX + F(5)).toFixed(1), y: (padT + F(10)).toFixed(1), fill: "#39d98a", "font-size": F(11), "font-weight": 700, "text-anchor": "start" }, `◂ ${bnd} match${bnd === 1 ? "" : "es"}`));
	}
	// tiered grouping brackets, below the labels
	if (tiers.length) {
		const top = axisY + F(8) + labelH + F(6);
		tiers.forEach((groups, L) => {
			const ty = top + L * TIER_H + TIER_H * 0.42;
			for (const grp of groups) {
				const gx0 = xs[grp.i0].x0 + 3, gx1 = xs[grp.i1].x1 - 3;
				svg.appendChild(svgEl("path", { d: `M${gx0.toFixed(1)} ${(ty - 4).toFixed(1)} L${gx0.toFixed(1)} ${ty.toFixed(1)} L${gx1.toFixed(1)} ${ty.toFixed(1)} L${gx1.toFixed(1)} ${(ty - 4).toFixed(1)}`, fill: "none", stroke: "rgba(220,230,245,0.32)", "stroke-width": 1 }));
				svg.appendChild(svgEl("text", { x: ((gx0 + gx1) / 2).toFixed(1), y: (ty + F(11)).toFixed(1), fill: "rgba(220,230,245,0.72)", "font-size": F(10.5), "text-anchor": "middle" }, grp.label));
			}
		});
	}
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
const OUT_SORTS = [["height", "height"], ["prefix", "prefix"], ["zone", "zone"], ["search", "search"]];
export function outputCard(rows) {
	const allItems = poolOutputs(rows);
	if (!allItems.length) return card("output", null, empty("this selection emits no output items (e.g. a plan step) — pick a decompose / bbox step"));
	// annotate each item with its grouping paths (for the tiered brackets)
	for (const it of allItems) { it._prefix = _prefixPath(it.id); it._zone = _zonePath(it.id); }
	const maxTotal = Math.max(1e-9, ...allItems.map((it) => it.total));
	const keyOf = (it) => `${it.ev}:${it.id}`;
	let sortMode = "height";      // height | prefix | zone | search
	let query = "";
	let expanded = null;          // key of the clicked/expanded item (fisheye + pinned detail)

	// The displayed (sorted / filtered) item list for the current mode.
	const cmpPath = (a, b) => {
		const pa = _pathFor(a, sortMode), pb = _pathFor(b, sortMode);
		for (let i = 0; i < Math.max(pa.length, pb.length); i++) { const x = pa[i] || "", y = pb[i] || ""; if (x !== y) return x < y ? -1 : 1; }
		return String(a.id) < String(b.id) ? -1 : 1;
	};
	const searchHit = () => { const q = query.trim().toLowerCase(); return q ? ((it) => String(it.id).toLowerCase().includes(q)) : null; };
	function displayed() {
		if (sortMode === "prefix" || sortMode === "zone") return [...allItems].sort(cmpPath);
		const byH = (a, b) => b.total - a.total;
		const hit = sortMode === "search" ? searchHit() : null;
		// SEARCH doesn't hide non-matches: it moves the matched objects to the FRONT
		// (by height), then keeps the rest (by height) after a green divider — so a
		// searched cohort (e.g. every "wall") can be compared against the whole scene.
		if (hit) return [...allItems.filter(hit)].sort(byH).concat([...allItems.filter((it) => !hit(it))].sort(byH));
		return [...allItems].sort(byH); // height (and empty-query search) by height
	}
	// Index of the divider = how many items matched (they sit at the front). -1 when
	// not searching / empty query / all-or-none match (no meaningful split to draw).
	function searchBoundary() {
		const hit = sortMode === "search" ? searchHit() : null;
		if (!hit) return -1;
		let n = 0; for (const it of allItems) if (hit(it)) n++;
		return n > 0 && n < allItems.length ? n : -1;
	}
	const detail = el("div", { class: "out-detail" });
	const host = chartHost((w, h) => {
		const list = displayed();
		const foc = expanded ? list.findIndex((it) => keyOf(it) === expanded) : -1;
		return buildOutputSvg(list, maxTotal, w, h, foc, sortMode, {
			onClick: (i) => { const k = keyOf(list[i]); expanded = expanded === k ? null : k; renderDetail(); repaint(host); },
			boundary: searchBoundary(),
		});
	}, (w) => { const m = outMetrics(displayed(), sortMode, w); return m.padT + vh(0.42, 240, 480) + m.padB; });
	host.style.overflowX = "auto";

	function renderDetail() {
		const it = expanded ? allItems.find((x) => keyOf(x) === expanded) : null;
		detail.replaceChildren(it ? outputDetail(it)
			: el("div", { class: "out-detail-hint", text: "click a column to expand it and pin its full attribute breakdown here" }));
	}
	renderDetail();

	// sort / group control + (search mode) filter box
	const controls = el("div", { class: "out-sortbar" });
	const searchInput = el("input", { class: "vii-search", type: "text", placeholder: "search id — matches ◂ green divider ▸ rest…",
		oninput: (e) => { query = e.target.value; if (sortMode === "search") repaint(host); } });
	function buildControls() {
		const seg = el("div", { class: "seg-ctl" }, ...OUT_SORTS.map(([id, lab]) =>
			el("button", { class: `seg-btn${sortMode === id ? " on" : ""}`, onclick: () => { sortMode = id; expanded = null; renderDetail(); buildControls(); repaint(host); } }, lab)));
		controls.replaceChildren(el("span", { class: "muted", style: "font-size:12px", text: "sort" }), seg,
			...(sortMode === "search" ? [searchInput] : []));
		if (sortMode === "search") setTimeout(() => searchInput.focus(), 0);
	}
	buildControls();

	// attribute colour legend — the stacked segments are coloured by attribute
	// (component); show only the attributes actually present, in canonical order.
	const present = new Set();
	for (const it of allItems) for (const a of it.attrs) if (a.score > 0) present.add(a.component);
	const legendComps = [...ATTR_AXIS_ORDER.filter((c) => present.has(c)), ...[...present].filter((c) => !ATTR_AXIS_ORDER.includes(c))];
	const legend = el("div", { class: "chart-legend", style: "gap:7px 13px;margin-top:8px;font-size:11.5px" },
		el("span", { class: "muted", text: "attribute" }),
		...legendComps.map((c) => el("div", { class: "lg", title: c }, el("span", { class: "sw", style: `background:${compHex(c)}` }), el("span", { text: c }))));

	const sub = rows.length === 1 ? `${allItems.length} output objects` : `${allItems.length} output objects · ${rows.length} steps`;
	return card("output", sub, controls, host, legend,
		el("div", { class: "out-hint", text: "each bin = one emitted object · stacked by attribute · height = attention it drew · click a column to expand + pin its breakdown · sort by height / prefix / zone, or search (matches move left of a green divider, the rest of the scene continue on the right at the same scale so you can compare)" }),
		detail);
}

// --- 4. prompt tag breakdown (organized <tags>) -----------------------------
const SECTION_COLORS = ["#7aa2f7", "#6bd96e", "#e0a94a", "#b46aff", "#4af0e0", "#ff6b9d", "#ff9e64", "#9ece6a", "#f7768e", "#7dcfff", "#c98bdb", "#e6db74"];
const tagCompact = (t) => (t || "").replace(/_/g, " ");
// Attention over organized <tag> blocks, two ways (toggle). BOX (default): for each
// step KIND, a box per TAG — the distribution of that tag section's attention across
// the steps of that kind (Q1–Q3, median, 1.5·IQR whiskers, outliers). SCATTER: the
// VII scatter's twin — each dot = one tag block in one step (x = length in tokens,
// y = attention, normalized so the top block = 1) with a linear fit + σ-below-trend
// ranking. Both color by tag.
export function tagsCard(rows) {
	const TITLE = "prompt tag breakdown";
	const model = tagScatterModel(rows);
	if (!model) return card(TITLE, null, empty("no organized <tag> prompt sections carry attention in this selection"));
	const { P, max, slope, fit, ranked, marked, xMaxTok, items, tags } = model;
	const tagColorMap = new Map(tags.map((t, i) => [t, SECTION_COLORS[i % SECTION_COLORS.length]]));
	const tagColor = (t) => tagColorMap.get(t) || "#7aa2f7";
	const rowText = (p) => `<${p.tag}> · ${p.node}`;
	const nSteps = new Set(P.map((p) => p.ev)).size;
	const sub = `${items.length} tag block${items.length === 1 ? "" : "s"} · ${tags.length} tag${tags.length === 1 ? "" : "s"} · ${nSteps} step${nSteps === 1 ? "" : "s"} · trend ${slope >= 0 ? "+" : "−"}${Math.abs(slope * 100).toFixed(2)}%/tok · ${marked.size} over-length pick${marked.size === 1 ? "" : "s"}`;
	const sideOf = (w) => Math.max(300, Math.min(w, vh(0.62, 320, 600))); // square side, capped
	let query = "", selKey = null;
	const host = chartHost((w) => {
		const q = query.trim().toLowerCase();
		const side = sideOf(w);
		const pts = P.map((p) => ({ ...p, r: 4.5, color: tagColor(p.tag), hit: q ? (p.tag.toLowerCase().includes(q) || String(p.node).toLowerCase().includes(q)) : false }));
		return scatterChart(pts, {
			width: side, height: side, yMax: 1, xMax: xMaxTok,
			xLabel: "section length (tokens)",
			xFmt: (v) => String(Math.round(v)),
			yFmt: (v) => v.toFixed(2),
			empty: "no tag sections to plot",
			line: { points: [{ x: 0, y: fit(0) }, { x: xMaxTok, y: fit(xMaxTok) }], color: "#ff2b2b", width: 2, dash: "5 4" },
			yErr: (p) => p.ey,
			dim: q ? (p) => !p.hit : null,
			hot: (p) => (q && p.hit) || p.key === selKey,
			ring: (p) => (marked.has(p.key) ? "#39d98a" : null),
			tip: (p) => {
				const rel = Math.round(p.resid * 100); // in % of the top block (y is ÷ max)
				const trend = p.resid < 0 ? `<span class="below">${-rel}% below trend · ${p.z.toFixed(1)}σ</span>` : `${rel}% above trend`;
				const sw = `<span style="display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;background:${tagColor(p.tag)}"></span>`;
				return `<div class="xh">${(p.share * 100).toFixed(3)}% attention · ${p.tokens} tokens · ±${(p.ey * max * 100).toFixed(3)}%</div>`
					+ `<div class="tip-trend">${sw}&lt;${escTip(p.tag)}&gt; · ${escTip(tagCompact(p.kind))} · ${marked.has(p.key) ? "★ over-length · " : ""}${trend}</div>`
					+ `<div class="tip-sentence">step node: ${escTip(String(p.node))}</div>`;
			},
		});
	}, (w) => sideOf(w));
	host.classList.add("vii-plot-host");
	// Right panel — Δ chart / ★ attended / σ trim over the SAME (tag, step) blocks.
	const focused = state.step !== ALL;
	let rightMode = focused ? "delta" : "trim";
	let deltaSort = "delta";  // delta | attention | length | tag
	let deltaDesc = true;
	const rightBox = el("div", { class: "vii-rank" });
	const spotlight = (p) => { selKey = selKey === p.key ? null : p.key; renderRight(); repaint(host); };
	const wireRow = (row, p) => {
		row.onclick = () => {
			const sel = window.getSelection();
			if (sel && sel.rangeCount && !sel.isCollapsed && row.contains(sel.anchorNode)) return;
			spotlight(p);
		};
		row._key = p.key;
	};
	const rankRow = (p, i, valText, pos) => {
		const row = el("button", { class: `vii-rk${p.key === selKey ? " on" : ""}${marked.has(p.key) ? " pick" : ""}`, style: `box-shadow:inset 3px 0 0 ${tagColor(p.tag)}` },
			el("span", { class: "vii-rk-n", text: String(i + 1) }),
			el("span", { class: `vii-rk-z${pos ? " pos" : ""}`, text: valText }),
			el("span", { class: "vii-rk-txt", text: rowText(p) }));
		wireRow(row, p);
		return row;
	};
	const sortedForDelta = () => {
		const dir = deltaDesc ? -1 : 1;
		const byResid = (a, b) => (a.resid - b.resid) * dir;
		const cmp = deltaSort === "attention" ? (a, b) => (a.share - b.share) * dir
			: deltaSort === "length" ? (a, b) => (a.tokens - b.tokens) * dir
				: deltaSort === "tag" ? (a, b) => (tags.indexOf(a.tag) - tags.indexOf(b.tag)) || byResid(a, b)
					: byResid;
		return [...P].sort(cmp);
	};
	const deltaTip = (p) => {
		const pos = p.resid >= 0;
		return `<div class="xh">${(p.share * 100).toFixed(3)}% attn · ${p.tokens} tok</div>`
			+ `<div class="tip-trend"><span style="display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;background:${tagColor(p.tag)}"></span>&lt;${escTip(p.tag)}&gt; · ${escTip(String(p.node))} · Δ ${pos ? "+" : "−"}${Math.abs(p.resid * 100).toFixed(1)}% vs fit${marked.has(p.key) ? " · ★ over-length" : ""}</div>`;
	};
	const renderRight = () => {
		const modeBtn = (id, label, title) => {
			const b = el("button", { class: `seg-btn${rightMode === id ? " on" : ""}`, title }, label);
			b.onclick = () => { rightMode = id; renderRight(); };
			return b;
		};
		const seg = el("div", { class: "seg-ctl" },
			modeBtn("delta", "Δ chart", "bar chart: attention minus the length-fit per tag block (x sortable several ways)"),
			modeBtn("attended", "★ attended", "ranked by raw attention — the most-attended tag blocks"),
			modeBtn("trim", "σ trim", "farthest below the trend — long but under-attended tag blocks"));
		const head = el("div", { class: "vii-rank-head vii-rank-tools" }, seg);
		if (rightMode === "delta") {
			const sortSel = el("select", { class: "vii-sort-sel", title: "sort the x-axis" },
				...[["delta", "Δ attn−fit"], ["attention", "attention"], ["length", "length"], ["tag", "tag"]]
					.map(([v, t]) => el("option", { value: v, text: t, ...(v === deltaSort ? { selected: "" } : {}) })));
			sortSel.onchange = (e) => { deltaSort = e.target.value; renderRight(); };
			const dirBtn = el("button", { class: "mini-toggle on", title: "sort direction", text: deltaDesc ? "↓ high" : "↑ low" });
			dirBtn.onclick = () => { deltaDesc = !deltaDesc; renderRight(); };
			head.append(sortSel, dirBtn);
			const bars = sortedForDelta().map((p) => ({ value: p.resid, color: tagColor(p.tag), label: rowText(p), _p: p }));
			const chartWrap = el("div", { class: "vii-delta-chart" });
			chartWrap.appendChild(chartHost(
				(w) => barChart(bars, { width: w, height: 300, yLabel: "attention − fit", yFmt: (v) => `${(v * 100).toFixed(0)}%`, empty: "no tag blocks to plot", tip: (b) => deltaTip(b._p), hot: (b) => b._p.key === selKey, onClick: (b) => spotlight(b._p) }),
				() => 300));
			rightBox.replaceChildren(head, chartWrap);
			return;
		}
		let list;
		if (rightMode === "attended") {
			list = [...P].sort((a, b) => b.share - a.share).map((p, i) => rankRow(p, i, `${(p.share * 100).toFixed(2)}%`, true));
		} else {
			list = ranked.map((p, i) => rankRow(p, i, `${p.z >= 0 ? "+" : ""}${p.z.toFixed(1)}σ`, p.z > 0));
		}
		rightBox.replaceChildren(head, el("div", { class: "vii-rank-list" }, ...list));
	};
	renderRight();
	const main = el("div", { class: "vii-main" }, host, rightBox);
	const legend = el("div", { class: "vii-legend" },
		el("span", { class: "vii-legend-lab", text: "tag" }),
		chartLegend(tags.map((t) => ({ key: t, label: `<${t}>`, color: tagColor(t) }))));
	const count = el("span", { class: "vii-search-n" });
	const search = el("input", { class: "vii-search", type: "search", placeholder: "search tags / step nodes to highlight on the plot…", spellcheck: "false" });
	const sync = () => {
		const q = query.trim().toLowerCase();
		const m = q ? P.filter((p) => p.tag.toLowerCase().includes(q) || String(p.node).toLowerCase().includes(q)).length : 0;
		count.textContent = q ? `${m} match${m === 1 ? "" : "es"}` : "";
		repaint(host);
	};
	search.oninput = (e) => { query = e.target.value; sync(); };
	const ico = svgEl("svg", { viewBox: "0 0 16 16", class: "vii-search-ico", width: 14, height: 14 });
	ico.appendChild(svgEl("path", { d: "M6.5 1a5.5 5.5 0 0 1 4.38 8.82l3.9 3.9-1.06 1.06-3.9-3.9A5.5 5.5 0 1 1 6.5 1zm0 1.5a4 4 0 1 0 0 8 4 4 0 0 0 0-8z", fill: "currentColor" }));
	const bar = el("div", { class: "vii-searchbar" }, ico, search, count);
	const scatterNode = el("div", {}, bar, main,
		el("div", { class: "vii-hint", text: "each dot = one organized <tag> block in one step. x = its length (tokens) · y = attention drawn, normalized so the most-attended block = 1 · whiskers = ±1 standard error · dashed line = linear fit. dots are colored by TAG (see legend). RIGHT panel — Δ vs length: attention minus the fit (sortable bars); ★ attended: ranked by raw attention; σ trim: farthest below the trend (green-ringed = long but under-attended sections). hover a dot for its score; click a row to spotlight it; search to highlight matches." }));

	// BOX view: for each step KIND, one box per TAG — the distribution of that tag
	// section's attention across the steps of that kind (grouped by kind on x).
	const boxHost = (() => {
		const kinds = viiKindsPresent(P);
		const clusters = kinds.map((k) => {
			const byTag = new Map();
			for (const p of P) if (p.kind === k) { let arr = byTag.get(p.tag); if (!arr) { arr = []; byTag.set(p.tag, arr); } arr.push(p.share); }
			const boxes = tags.filter((t) => byTag.has(t)).map((t) => ({ label: t, color: tagColor(t), values: byTag.get(t) }));
			return { label: tagCompact(k), boxes };
		}).filter((c) => c.boxes.length);
		const hAt = () => vh(0.5, 320, 560);
		const h = chartHost((w) => boxPlotChart(clusters, {
			width: w, height: hAt(), yLabel: "attention share", yFmt: (v) => `${(v * 100).toFixed(0)}%`,
			empty: "no tag sections to plot",
			tip: (e) => `<div class="xh">&lt;${escTip(e.label)}&gt; · ${escTip(e.cluster)} · n=${e.st.n}</div>`
				+ `<div class="tip-trend">median ${(e.st.med * 100).toFixed(2)}% · IQR ${(e.st.q1 * 100).toFixed(2)}–${(e.st.q3 * 100).toFixed(2)}% · mean ${(e.st.mean * 100).toFixed(2)}%</div>`
				+ `<div class="tip-sentence">range ${(e.st.min * 100).toFixed(2)}–${(e.st.max * 100).toFixed(2)}%${e.st.outliers.length ? ` · ${e.st.outliers.length} outlier${e.st.outliers.length === 1 ? "" : "s"}` : ""}</div>`,
		}), hAt);
		h.style.overflowX = "auto";
		return h;
	})();
	const boxNode = el("div", {}, boxHost,
		el("div", { class: "vii-hint", text: "each box = one <tag> section's attention across the steps of a kind — grouped by step kind on the x-axis, colored by tag. box = Q1–Q3 · bright line = median · dashed = mean · whiskers = 1.5×IQR · dots = outliers. y = attention share. hover a box for its five-number summary." }));

	// box (default) / scatter view toggle
	let view = "box";
	const body = el("div", { class: "vii-body" }, boxNode);
	const viewSeg = el("div", { class: "seg-ctl" });
	const mkBtns = () => [
		["box", "▧ box plot", "per step kind, a box per tag: its attention distribution (Q1–Q3, median, whiskers)"],
		["scatter", "· scatter", "attention vs length per (tag, step), with the linear fit + ranking"],
	].map(([id, lab, title]) => {
		const b = el("button", { class: `seg-btn${view === id ? " on" : ""}`, title }, lab);
		b.onclick = () => {
			if (view === id) return;
			view = id;
			viewSeg.replaceChildren(...mkBtns());
			body.replaceChildren(view === "box" ? boxNode : scatterNode);
			repaint(view === "box" ? boxHost : host);
		};
		return b;
	});
	viewSeg.replaceChildren(...mkBtns());
	const controls = el("div", { class: "vii-viewbar" }, el("span", { class: "muted", style: "font-size:12px", text: "view" }), viewSeg);
	return card(TITLE, sub, controls, legend, body);
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
	// Right panel — switchable views over the SAME instructions, each row clickable
	// to spotlight it on the scatter (drag-selecting text to copy is preserved):
	//   delta    — attention MINUS the length-fit (residual): a diverging bar list,
	//              sortable, so length-adjusted over/under-attention reads at a glance
	//   attended — ranked by RAW attention (the best / most-attended instructions)
	//   trim     — the original σ-below-trend ranking (trim candidates)
	// Default = delta for a focused (step-kind / single-step) selection, else trim.
	const focused = state.step !== ALL;
	let rightMode = focused ? "delta" : "trim";
	// x-axis sort orders for the Δ bar chart ("sorted in several ways").
	let deltaSort = "delta";  // delta | attention | length | kind
	let deltaDesc = true;
	const rightBox = el("div", { class: "vii-rank" });
	const spotlight = (p) => { selKey = selKey === p.key ? null : p.key; renderRight(); repaint(host); };
	const wireRow = (row, p) => {
		row.onclick = () => {
			const sel = window.getSelection();
			if (sel && sel.rangeCount && !sel.isCollapsed && row.contains(sel.anchorNode)) return;
			spotlight(p);
		};
		row._key = p.key;
	};
	const rankRow = (p, i, valText, pos) => {
		const row = el("button", { class: `vii-rk${p.key === selKey ? " on" : ""}${marked.has(p.key) ? " pick" : ""}`, style: `box-shadow:inset 3px 0 0 ${viiKindColor(p.kind)}` },
			el("span", { class: "vii-rk-n", text: String(i + 1) }),
			el("span", { class: `vii-rk-z${pos ? " pos" : ""}`, text: valText }),
			el("span", { class: "vii-rk-txt", text: p.label }));
		wireRow(row, p);
		return row;
	};
	// Sort the instructions for the Δ bar chart's x-axis by the chosen key.
	const sortedForDelta = () => {
		const dir = deltaDesc ? -1 : 1;
		const ord = viiKindsPresent(P);
		const byResid = (a, b) => (a.resid - b.resid) * dir;
		const cmp = deltaSort === "attention" ? (a, b) => (a.share - b.share) * dir
			: deltaSort === "length" ? (a, b) => (a.tokens - b.tokens) * dir
				: deltaSort === "kind" ? (a, b) => (ord.indexOf(a.kind) - ord.indexOf(b.kind)) || byResid(a, b)
					: byResid;
		return [...P].sort(cmp);
	};
	const deltaTip = (p) => {
		const pos = p.resid >= 0;
		return `<div class="xh">${(p.share * 100).toFixed(3)}% attn · ${p.tokens} tok</div>`
			+ `<div class="tip-trend"><span style="display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;background:${viiKindColor(p.kind)}"></span>${escTip(viiKindLabel(p.kind))} · Δ ${pos ? "+" : "−"}${Math.abs(p.resid * 100).toFixed(1)}% vs fit${marked.has(p.key) ? " · ★ trim" : ""}</div>`
			+ `<div class="tip-sentence">${escTip(p.label)}</div>`;
	};
	const renderRight = () => {
		const modeBtn = (id, label, title) => {
			const b = el("button", { class: `seg-btn${rightMode === id ? " on" : ""}`, title }, label);
			b.onclick = () => { rightMode = id; renderRight(); };
			return b;
		};
		const seg = el("div", { class: "seg-ctl" },
			modeBtn("delta", "Δ chart", "bar chart: attention minus the length-fit per instruction (x sortable several ways)"),
			modeBtn("attended", "★ attended", "ranked by raw attention — the most-attended instructions"),
			modeBtn("trim", "σ trim", "farthest below the trend — the trim candidates"));
		const head = el("div", { class: "vii-rank-head vii-rank-tools" }, seg);
		if (rightMode === "delta") {
			// x-axis sort control (several ways) + direction toggle
			const sortSel = el("select", { class: "vii-sort-sel", title: "sort the x-axis" },
				...[["delta", "Δ attn−fit"], ["attention", "attention"], ["length", "length"], ["kind", "step kind"]]
					.map(([v, t]) => el("option", { value: v, text: t, ...(v === deltaSort ? { selected: "" } : {}) })));
			sortSel.onchange = (e) => { deltaSort = e.target.value; renderRight(); };
			const dirBtn = el("button", { class: "mini-toggle on", title: "sort direction", text: deltaDesc ? "↓ high" : "↑ low" });
			dirBtn.onclick = () => { deltaDesc = !deltaDesc; renderRight(); };
			head.append(sortSel, dirBtn);
			const bars = sortedForDelta().map((p) => ({ value: p.resid, color: viiKindColor(p.kind), label: p.label, _p: p }));
			const chartWrap = el("div", { class: "vii-delta-chart" });
			chartWrap.appendChild(chartHost(
				(w) => barChart(bars, { width: w, height: 300, yLabel: "attention − fit", yFmt: (v) => `${(v * 100).toFixed(0)}%`, empty: "no instructions to plot", tip: (b) => deltaTip(b._p), hot: (b) => b._p.key === selKey, onClick: (b) => spotlight(b._p) }),
				() => 300));
			rightBox.replaceChildren(head, chartWrap);
			return;
		}
		let list;
		if (rightMode === "attended") {
			list = [...P].sort((a, b) => b.share - a.share).map((p, i) => rankRow(p, i, `${(p.share * 100).toFixed(2)}%`, true));
		} else {
			list = ranked.map((p, i) => rankRow(p, i, `${p.z >= 0 ? "+" : ""}${p.z.toFixed(1)}σ`, p.z > 0));
		}
		rightBox.replaceChildren(head, el("div", { class: "vii-rank-list" }, ...list));
	};
	renderRight();
	const main = el("div", { class: "vii-main" }, host, rightBox);
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
		el("div", { class: "vii-hint", text: "each dot = one instruction (variable values like ids / coordinates are masked so the same instruction merges across steps). x = its length (tokens) · y = attention drawn, normalized so the most-attended = 1 · whiskers = ±1 standard error · dashed line = linear fit. dots + rows are colored by STEP KIND (see legend). RIGHT panel — Δ vs length: attention minus the fit (diverging bars, sortable); ★ attended: ranked by raw attention (best-attended); σ trim: farthest below the trend (green-ringed = trim candidates). hover a dot for its score; click a row to spotlight it; search to highlight matches." }));
}
