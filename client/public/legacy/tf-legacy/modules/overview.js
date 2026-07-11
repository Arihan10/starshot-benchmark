// Whole-cell overview: aggregates EVERY computed step's stored analysis (fetched
// lazily + cached in state.stepAnalyses) into cell-wide stats and per-step-kind
// breakdowns, plus the hierarchical attention "badge tree" and the static
// "attention in 3D" launcher.

import { el } from "../../js/ui.js";
import { api } from "../../js/api.js";
import { emittingRegion } from "../../js/events.js";
import { $, state, COLORS, compHex, entityHex, entityKindLabel, stepKind, templateColor, groupByTemplate, treeHoverRegister } from "./state.js";
import { _mean, niceMax, heatColor, pool } from "./util.js";
import { aggregateAttn, hasToPlace, overviewAggregate } from "./aggregate.js";
import { summaryBlock, summaryBar, statCard, wide } from "./widgets.js";
import { pm, fmtPM, fmtNum, axisMax } from "./uncertainty.js";
import { jumpTo } from "./exportPanel.js";
import { applyAttnHighlight } from "./attnPanel.js";
import { gotoStep } from "./render.js";
import { openAttn3DWindow, reapplyAttn3DWindow } from "./present.js";
import { updateReportCtx } from "./tabs.js";
import * as report from "./report.js";

// Expose attnPanel's default (token/head) 3D highlight to the shared hover-restore
// in state.js, which can't import attnPanel without a cycle. Set once on load.
state.applyAttnHighlight = applyAttnHighlight;

// Raw scored entities tagged with their canonical kind (zone|frame|object) — the
// 3D view filters by type and normalizes per visible subset (see present.js
// openAttn3DWindow → computeItems).
function scopeEntities(scope) {
	return (overviewAggregate(scope).entityTotals || [])
		.filter((e) => e.score > 0)
		.map((e) => ({ id: e.id, score: e.score, kind: entityKindLabel(e.kind, e.id) }));
}
function scopeSub(scope, total) {
	if (scope.length === 1) {
		const s = scope[0].step;
		const idx = s ? state.steps.indexOf(s) + 1 : 0;
		return s ? `${stepKind(s)} · step ${idx || "?"}` : "current step";
	}
	return scope.length >= total ? `${scope.length} steps` : `last ${scope.length} of ${total} steps`;
}

// Shade the scene by this tree's aggregated attention in a draggable + resizable
// popup window (present.js openAttn3DWindow), reparenting the live viewer into it
// so it floats ABOVE the report workspace. For a single current step the window
// tracks step navigation via `getEntities` (contents update, window persists).
function viewAttnIn3D(scope, { total = scope.length } = {}) {
	if (!state.viewer || !scope?.length) return;
	const entities = scopeEntities(scope);
	if (!entities.length) {
		$("tf-attn-progress").textContent = "view in 3D: no attended entities in this scope";
		return;
	}
	// Live refresh for the step view: recompute from whatever step is displayed so
	// arrow-key navigation updates the popup in place (kind/scene stay static).
	const cur = state.steps[state.stepIdx];
	const isStepScope = scope.length === 1 && scope[0].step && cur && scope[0].step.event_index === cur.event_index;
	const getEntities = isStepScope ? () => {
		const a = state.attn, step = state.steps[state.stepIdx];
		if (!a || !step || a.meta?.event_index !== step.event_index) return null;
		const sc = [{ step, a, agg: aggregateAttn(a) }];
		return { entities: scopeEntities(sc), sub: scopeSub(sc, 1), key: step.event_index };
	} : null;
	openAttn3DWindow({
		viewer: state.viewer,
		host: $("tf-canvas-host"),
		modelLabel: state.model || state.attn?.meta?.model_id || "the model",
		title: "attention in 3D",
		sub: scopeSub(scope, total),
		entities,
		getEntities,
		kindColors: { zone: COLORS.zone, frame: COLORS.frame, object: COLORS.object },
		onClose: () => { state.restoreAttnHighlight = null; applyAttnHighlight(); },
	});
	// While the popup owns the viewer shading, hover-out restores its aggregate
	// (rather than the token/head default) — see state.setHoverEntity.
	state.restoreAttnHighlight = () => reapplyAttn3DWindow();
}

let _overviewLoading = false;
// Fetch (once) every computed step's stored analysis into the cache, then
// re-render. Guarded so overlapping polls don't double-fetch.
export async function ensureOverviewLoaded() {
	if (_overviewLoading) return;
	const run = state.run, slot = state.slot, model = state.model;
	// Newest first: recent steps are already the small compact ones (they return
	// instantly), so the tab fills in immediately while any older LEGACY blobs are
	// down-projected server-side (a one-time heal) behind them — no more freeze.
	const missing = state.steps.filter((s) =>
		["ready", "stale"].includes(state.attnStatus[s.event_index]) && !state.stepAnalyses.has(s.event_index))
		.sort((a, b) => b.event_index - a.event_index);
	if (!missing.length) return;
	_overviewLoading = true;
	const sameCell = () => state.run === run && state.slot === slot && state.model === model;
	let done = 0, lastPaint = 0;
	// Keep concurrency modest: a legacy step's compact is built from a big parse on
	// the server (serialized there), so flooding it with requests buys nothing.
	await pool(missing, 5, async (s) => {
		try {
			const a = await api.attentionGet(run, slot, model, s.event_index, { view: "compact" });
			if (sameCell()) state.stepAnalyses.set(s.event_index, a);
		} catch { /* skip unreadable */ }
		done++;
		// Repaint progressively (throttled) so computed steps appear as they arrive
		// rather than all at once after the (possibly slow) legacy heals resolve.
		const now = Date.now();
		if (sameCell() && state.reportView && (done === missing.length || now - lastPaint > 500)) {
			lastPaint = now;
			report.renderReportWorkspace();
		}
	});
	_overviewLoading = false;
	if (sameCell()) report.renderReportWorkspace();
}

function renderOverview() {
	if (state.reportView && ["scene", "compare", "board"].includes(state.reportView)) { report.renderReportWorkspace(); return; }
	const host = $("tf-overview-panel");
	if (!host || !report.reportShowing("overview")) return;
	updateReportCtx();
	const scroller = $("tf-report-body");
	const savedTop = scroller ? scroller.scrollTop : 0;
	const paint = (...c) => { host.replaceChildren(...c); if (scroller) scroller.scrollTop = savedTop; };

	const computed = state.steps.filter((s) => ["ready", "stale"].includes(state.attnStatus[s.event_index]));
	if (!computed.length) {
		paint(el("div", { class: "empty-note" }, el("span", { class: "big", text: "🗺" }),
			el("span", { text: "no computed steps yet — run ‘⚡ compute all steps’ to build the cell overview" })));
		return;
	}
	const rows = computed.map((s) => ({ step: s, a: state.stepAnalyses.get(s.event_index) })).filter((r) => r.a);
	const missing = computed.length - rows.length;
	if (missing > 0) ensureOverviewLoaded(); // fill the cache, then this re-renders
	if (!rows.length) {
		paint(el("div", { class: "empty-note" }, el("span", { class: "big", text: "⏳" }), el("span", { text: `loading ${computed.length} step analyses…` })));
		return;
	}
	for (const r of rows) {
		r.agg = aggregateAttn(r.a); r.template = stepKind(r.step); r.mass = _mean(r.agg.mass);
		r.hasTp = hasToPlace(r.a);
		r.tpMass = r.hasTp ? _mean(r.a.agg.to_place.mass || []) : null;
	}
	const order = [...new Set(rows.map((r) => r.template))];
	paint(...[
		el("div", { class: "rep-note",
			text: `${rows.length} of ${state.steps.length} steps computed${missing ? ` · loading ${missing}…` : ""} · ${order.length} step kinds` }),
		wide(overviewKeyStats(rows)),
		wide(overviewMassPerStep(rows, order)), // per-step bar chart reads best full width
		overviewToPlace(rows),        // bbox steps: to-place vs scene attention (null if none)
		overviewByKindMass(rows, order),
		overviewByKindComposition(rows, order),
		overviewByKindAttributes(rows, order),
		overviewEntities(rows),
		overviewLayerDepth(rows),
		wide(overviewHeadGrid(rows)), // layers × heads grid needs the full width
	].filter(Boolean));
}

// Cell-wide to-place readout: across the steps that place a batch, mean attention
// on the placed objects vs. the scene, plus the most-attended placed objects.
export function overviewToPlace(rows) {
	const bbox = rows.filter((r) => r.hasTp);
	if (!bbox.length) return null;
	const tp = pm(bbox.map((r) => r.tpMass));
	const sc = pm(bbox.map((r) => r.mass));
	const smax = axisMax([tp, sc], (x) => x.m, (x) => x.s, state.showErr);
	const split = el("div", { class: "sbars" },
		summaryBar({ color: COLORS.to_place, label: "to-place (self)", value: tp.m, sd: tp.s, max: smax, title: `mean to-place attention across bbox steps · ${fmtPM(tp.m, tp.s)}` }),
		summaryBar({ color: "#4af0e0", label: "scene (context)", value: sc.m, sd: sc.s, max: smax, title: `mean scene attention across bbox steps · ${fmtPM(sc.m, sc.s)}` }),
	);
	const entity = new Map();
	for (const r of bbox) {
		for (const e of (r.a.agg.to_place.entityTotals || [])) {
			const cur = entity.get(e.id) || { id: e.id, kind: e.kind, vals: [] };
			cur.vals.push(e.score); entity.set(e.id, cur);
		}
	}
	const top = [...entity.values()]
		.map((e) => { const p = pm(e.vals.concat(Array(Math.max(0, bbox.length - e.vals.length)).fill(0))); return { id: e.id, kind: e.kind, score: p.m, sd: p.s }; })
		.sort((x, y) => y.score - x.score).slice(0, 12);
	const emax = axisMax(top, (e) => e.score, (e) => e.sd, state.showErr);
	const objs = top.length
		? el("div", { class: "sbars" }, ...top.map((e) => summaryBar({
			color: COLORS.to_place, label: e.id, value: e.score, sd: e.sd, max: emax,
			title: `${e.id} · ${fmtPM(e.score, e.sd)} mean±sd across bbox steps · click → focus in 3D`, onclick: () => jumpTo(e.id),
		})))
		: el("div", { class: "muted", style: "font-size:11px", text: "—" });
	return summaryBlock(`to-place vs scene · ${bbox.length} bbox step${bbox.length > 1 ? "s" : ""}`,
		"steps that place a batch: attention on the placed objects vs. the scene · most-attended placed objects cell-wide · click → focus",
		split, el("div", { class: "exp-lab", style: "margin-top:8px", text: "most-attended placed objects" }), objs);
}

export function overviewKeyStats(rows) {
	const cell = overviewAggregate(rows);
	const massPM = pm(rows.map((r) => r.mass));
	const hi = rows.reduce((b, r) => (r.mass > b.mass ? r : b), rows[0]);
	const topE = cell.entityTotals[0], topC = cell.componentTotals[0];
	return summaryBlock("organization statistics", null, el("div", { class: "stat-grid" },
		statCard("steps computed", `${rows.length}`, `of ${state.steps.length}`),
		statCard("scene mean scene mass", `${(massPM.m * 100).toFixed(1)}%`, `± ${(massPM.s * 100).toFixed(1)}% across steps`),
		statCard("highest-mass step", `${(hi.mass * 100).toFixed(0)}%`, hi.template),
		statCard("entities attended", `${cell.entityTotals.length}`),
		statCard("top entity (cell)", topE ? topE.id : "—", topE ? `${entityKindLabel(topE.kind, topE.id)} · ${fmtPM(topE.score, topE.sd)}` : null),
		statCard("top attribute (cell)", topC ? topC.component : "—", topC ? fmtPM(topC.score, topC.sd) : null),
	));
}

// Scene mass across the pipeline: one bar per computed step (in order), colored
// by step kind. The current step is outlined; click to jump.
export function overviewMassPerStep(rows, order) {
	const H = 72;
	const top = niceMax(Math.max(...rows.map((r) => r.mass), 1e-9)); // absolute, nicely-rounded axis ceiling
	const fmt = (v) => `${(v * 100).toFixed(top < 0.1 ? 1 : 0)}%`;
	const curEv = state.steps[state.stepIdx]?.event_index;
	const bars = rows.map((r) => el("div", {
		class: `ovb${r.step.event_index === curEv ? " cur" : ""}`,
		title: `step ${state.steps.indexOf(r.step) + 1} · ${r.template} · scene mass ${(r.mass * 100).toFixed(1)}%`,
		onclick: () => gotoStep(state.steps.indexOf(r.step)),
	}, el("div", { class: "ovb-fill", style: `height:${(r.mass / top) * 100}%;background:${templateColor(r.template, order)}` })));
	const axis = el("div", { class: "ovaxis", style: `height:${H}px` },
		el("div", { class: "ovtick", text: fmt(top) }),
		el("div", { class: "ovtick", text: fmt(top / 2) }),
		el("div", { class: "ovtick", text: "0%" }),
	);
	const plot = el("div", { class: "ovplot", style: `height:${H}px` },
		el("div", { class: "ovgrid" }),
		el("div", { class: "ovbars" }, ...bars),
	);
	const legend = el("div", { class: "tk-legend" }, ...order.map((t) => el("span", {}, el("i", { style: `background:${templateColor(t, order)}` }), el("span", { text: t }))));
	return summaryBlock("scene mass across the pipeline",
		`one bar per computed step (in order) · y-axis = scene mass (0–${fmt(top)}) · colored by step kind · click to jump`,
		el("div", { class: "ovchart" }, axis, plot), legend);
}

// Cell-wide scene mass by layer depth: mean scene mass per global layer,
// averaged over EVERY head and EVERY computed step — where scene-tracking lives
// across the whole pipeline (the overview counterpart of summaryLayerDepth).
export function overviewLayerDepth(rows) {
	const byLayer = new Map();
	for (const r of rows) for (const g of r.a.head_grid || []) {
		const c = byLayer.get(g.layer) || { vals: [] };
		c.vals.push(g.mean_scale); byLayer.set(g.layer, c);
	}
	const items = [...byLayer.entries()].map(([layer, { vals }]) => { const p = pm(vals); return { layer, mean: p.m, sd: p.s }; }).sort((x, y) => x.layer - y.layer);
	if (items.length < 2) return null;
	const max = axisMax(items, (i) => i.mean, (i) => i.sd, state.showErr);
	const bars = items.map((it) => summaryBar({
		color: heatColor(it.mean), label: `L${it.layer}`, value: it.mean, sd: it.sd, max,
		title: `layer ${it.layer} · cell-mean scene mass ${fmtPM(it.mean, it.sd)} (all heads · all computed steps)`,
	}));
	return summaryBlock("scene mass by layer depth (cell-wide)",
		"mean scene mass per global layer, averaged over every head and every computed step — where scene-tracking concentrates across the pipeline",
		el("div", { class: "sbars" }, ...bars));
}

// Cell-wide heads × depth grid: mean scene mass per (global layer, head)
// averaged over all computed steps — which specific heads track the scene across
// the whole cell (the overview counterpart of the per-step headGrid).
export function overviewHeadGrid(rows) {
	const agg = new Map(); // "layer:head" -> { s, n, layer, head }
	for (const r of rows) for (const g of r.a.head_grid || []) {
		const k = `${g.layer}:${g.head}`;
		const c = agg.get(k) || { s: 0, n: 0, layer: g.layer, head: g.head };
		c.s += g.mean_scale; c.n++; agg.set(k, c);
	}
	if (!agg.size) return null;
	const layers = [...new Set([...agg.values()].map((c) => c.layer))].sort((x, y) => x - y);
	const heads = [...new Set([...agg.values()].map((c) => c.head))].sort((x, y) => x - y);
	const val = new Map([...agg.entries()].map(([k, c]) => [k, c.s / c.n]));
	const cols = `grid-template-columns:30px repeat(${heads.length},1fr)`;
	const header = el("div", { class: "hg-row", style: cols },
		el("div", { class: "hg-corner", text: "L╲H" }),
		...heads.map((hd) => el("div", { class: "hg-h", text: String(hd) })),
	);
	const body = layers.map((ly) => el("div", { class: "hg-row", style: cols },
		el("div", { class: "hg-l", text: `L${ly}` }),
		...heads.map((hd) => {
			const v = val.get(`${ly}:${hd}`);
			return el("div", {
				class: "hg-c",
				style: `background:${v == null ? "var(--panel-2)" : heatColor(v)}`,
				title: `layer ${ly} · head ${hd}${v == null ? "" : ` · cell-mean scene mass ${(v * 100).toFixed(1)}%`}`,
			});
		}),
	));
	return summaryBlock("heads × depth (cell-wide)",
		"mean scene mass per (layer, head) over all computed steps — which heads track the scene across the pipeline",
		el("div", { class: "hg-wrap" }, header, ...body));
}

// Mean scene mass per step kind — which prompt modes consult the scene most.
export function overviewByKindMass(rows, order) {
	const byT = groupByTemplate(rows);
	const items = order.filter((t) => byT.has(t)).map((t) => { const p = pm(byT.get(t).map((r) => r.mass)); return { t, mean: p.m, sd: p.s, n: byT.get(t).length }; }).sort((a, b) => b.mean - a.mean);
	const max = axisMax(items, (i) => i.mean, (i) => i.sd, state.showErr);
	const bars = items.map((it) => summaryBar({
		color: templateColor(it.t, order), label: `${it.t} ×${it.n}`, value: it.mean, sd: it.sd, max,
		title: `${it.t} · ${it.n} steps · mean scene mass ${fmtPM(it.mean, it.sd)}`,
	}));
	return summaryBlock("scene mass by step kind", "mean scene mass per prompt mode", el("div", { class: "sbars" }, ...bars));
}

// A horizontal error whisker overlaid on a stacked bar (percent coordinates).
// Centered at `centerPct` spanning ±`sdPct`; an "×" replaces it when too wide.
function compErrWhisker(centerPct, sdPct) {
	if (2 * sdPct > 60) return el("span", { class: "seg-err-x", style: `left:${Math.max(2, Math.min(96, centerPct))}%`, text: "×" });
	const lo = Math.max(0, centerPct - sdPct), hi = Math.min(100, centerPct + sdPct);
	return el("span", { class: "seg-err", style: `left:${lo}%;width:${Math.max(0, hi - lo)}%` });
}

// Zone / object / frame share of attention within each step kind. Each segment
// is the MEAN share across the kind's steps; an error whisker shows the ± sd of
// that share (equal weight per step), so a lopsided step can't hide in a sum.
export function overviewByKindComposition(rows, order) {
	const byT = groupByTemplate(rows);
	const kinds = [["zone", COLORS.zone], ["object", COLORS.object], ["frame", COLORS.frame]];
	const list = order.filter((t) => byT.has(t)).map((t) => {
		const perStep = byT.get(t).map((r) => {
			const kt = r.agg.kindTotals; const tot = (kt.zone + kt.object + kt.frame) || 1e-9;
			return { zone: kt.zone / tot, object: kt.object / tot, frame: kt.frame / tot };
		});
		const stats = kinds.map(([k, hex]) => { const p = pm(perStep.map((s) => s[k])); return { k, hex, m: p.m, s: p.s }; });
		const msum = stats.reduce((a, x) => a + x.m, 0) || 1e-9;
		const bar = el("div", { class: "kindbar", style: "flex:1" });
		let cum = 0;
		for (const st of stats) {
			const w = (100 * st.m) / msum;
			if (w <= 0) continue;
			bar.appendChild(el("div", { class: "seg", style: `width:${w}%;background:${st.hex}`, title: `${st.k} ${fmtPM(st.m * 100, st.s * 100, 1)}%` }));
			if (state.showErr && st.s > 0) bar.appendChild(compErrWhisker(cum + w / 2, (100 * st.s) / msum));
			cum += w;
		}
		return el("div", { class: "comp-row" }, el("span", { class: "comp-lab", title: t, text: t }), bar);
	});
	const legend = el("div", { class: "kindbar-legend" },
		...kinds.map(([k, hex]) => el("span", {}, el("i", { style: `background:${hex}` }), el("span", { text: k }))));
	return summaryBlock("kind composition by step kind", "zone / object / frame mean share ± sd within each prompt mode", el("div", { class: "comp-list" }, ...list), legend);
}

// Top attributes emphasized by each step kind.
function overviewByKindAttributes(rows, order) {
	const byT = groupByTemplate(rows);
	const list = order.filter((t) => byT.has(t)).map((t) => {
		const steps = byT.get(t);
		const comp = new Map();
		for (const r of steps) for (const c of r.agg.componentTotals) { const cur = comp.get(c.component) || []; cur.push(c.score); comp.set(c.component, cur); }
		const top = [...comp.entries()]
			.map(([component, vals]) => { const p = pm(vals.concat(Array(Math.max(0, steps.length - vals.length)).fill(0))); return { component, m: p.m, s: p.s }; })
			.sort((a, b) => b.m - a.m).slice(0, 5);
		if (!top.length) return null;
		return el("div", { class: "comp-row" },
			el("span", { class: "comp-lab", title: t, text: t }),
			el("div", { class: "out-chips" }, ...top.map((c) => el("span", { class: "chip", style: `background:${compHex(c.component)}`, title: `${c.component} · ${fmtPM(c.m, c.s)} mean±sd across steps`, text: c.component }))));
	}).filter(Boolean);
	return summaryBlock("top attributes by step kind", "which entity attributes each prompt mode emphasizes", el("div", { class: "comp-list" }, ...list));
}

// Most-attended entities across the whole cell.
export function overviewEntities(rows) {
	const items = overviewAggregate(rows).entityTotals.slice(0, 15);
	if (!items.length) return null;
	const max = items[0].score;
	const bars = items.map((e) => summaryBar({
		color: entityHex(e.kind, e.id), label: e.id, value: e.score, max,
		title: `${e.id} · ${fmtNum(e.score)} mean attention across steps — click to focus in 3D`,
		onclick: () => { jumpTo(e.id); applyAttnHighlight([{ id: e.id, weight: 1 }]); },
	}));
	return summaryBlock("most-attended entities (whole cell)", "mean attention across every computed step · click → focus in 3D", el("div", { class: "sbars" }, ...bars));
}

// ---- Attention tree -------------------------------------------------------
// Collapse state (whole-card clamp + per-tree carets) persists across the
// frequent workspace re-renders so a folded tree stays folded.
function treeFoldGet(key) { try { return localStorage.getItem(`tf-treefold-${key}`) === "1"; } catch { return false; } }
function treeFoldSet(key, v) { try { localStorage.setItem(`tf-treefold-${key}`, v ? "1" : "0"); } catch { /* ignore */ } }
// A hierarchical "badge tree" of the scene structure (root → zones/frames →
// objects → …). Each badge's WIDTH encodes that node's own mean attention over
// `rows` on ONE global scale (no error bars) — so within a layer the badge
// widths literally sum to that layer's total attention, and layers are directly
// comparable. Siblings are grouped under their parent (a gap separates groups),
// unattended subtrees are pruned, and thin badges keep their full name on hover.
// Row scope is windowed by the workspace-level slider in report.js (last N steps).
export function attentionTree(rows, { heatOnly = false, windowTotal = rows.length, targetZone = null } = {}) {
	const bodyHost = el("div", { class: "tree-host" });
	const relayout = () => bodyHost.querySelector(".tree-view")?._layout?.();
	const render = () => {
		bodyHost.replaceChildren(buildTreeBody(rows, { heatOnly, targetZone }));
		requestAnimationFrame(relayout);
	};
	if (typeof ResizeObserver !== "undefined" && !bodyHost._ro) {
		let lastW = 0;
		bodyHost._ro = new ResizeObserver(() => {
			const w = bodyHost.clientWidth;
			if (Math.abs(w - lastW) < 1) return;
			lastW = w;
			relayout();
		});
		bodyHost._ro.observe(bodyHost);
	}
	render();
	// Whole-card clamp: one toggle hides both trees (persisted), independent of
	// the per-tree carets, so a large tree can be tucked away.
	const foldBtn = el("button", { class: "tree-foldall", type: "button", title: "collapse or expand the whole attention tree" });
	const applyCardFold = (folded) => {
		bodyHost.style.display = folded ? "none" : "";
		foldBtn.textContent = folded ? "▸ show tree" : "▾ hide tree";
	};
	foldBtn.onclick = () => {
		const folded = bodyHost.style.display !== "none";
		treeFoldSet("card", folded);
		applyCardFold(folded);
		if (!folded) requestAnimationFrame(relayout);
	};
	applyCardFold(treeFoldGet("card"));
	const view3dBtn = el("button", {
		class: "tree-view3d", type: "button", title: "shade scene bboxes by this tree's attention in fullscreen 3D",
		text: "view in 3D",
		onclick: () => viewAttnIn3D(rows, { total: windowTotal }),
	});
	return el("div", { class: "tree-wrap" }, el("div", { class: "tree-toolbar" }, foldBtn, view3dBtn), bodyHost);
}

// Last-N-step window shared by every graph in the analysis workspace.
export function windowedScope(rows) {
	if (!rows?.length) return [];
	const sorted = [...rows].sort((a, b) => (a.step?.event_index ?? 0) - (b.step?.event_index ?? 0));
	if (sorted.length <= 1) return sorted;
	const n = Math.min(Math.max(1, state.reportLastN || 3), sorted.length);
	return sorted.slice(-n);
}

function buildTreeBody(scope, { heatOnly = false, targetZone = null } = {}) {
	const m = state.obs;
	if (!m || !scope.length) return el("div", { class: "muted", style: "font-size:12px", text: "no scene structure / attention yet" });
	const totals = overviewAggregate(scope).entityTotals;
	const score = new Map(totals.map((e) => [e.id, e.score]));
	const own = (id) => Math.max(0, score.get(id) || 0);
	// Zones own a line and hold children; objects/frames are always leaves.
	const kindOf = new Map(totals.map((e) => [e.id, e.kind]));
	const isZoneKind = (id) => ((kindOf.get(id) ?? m.nodes.get(id)?.kind) === "zone");
	// Parent = the ZONE a node was GENERATED IN (its emitting region), NOT the bbox
	// anchor: a lamp placed ON a nightstand still belongs to the bedroom zone. This
	// is the scene-context arrangement — objects live inside zones and are never
	// parents of other objects.
	const regionOf = (id) => { const r = emittingRegion(m, id); return r && r !== id && m.nodes.has(r) ? r : null; };
	const kidsOf = new Map();
	for (const id of m.order) { const p = regionOf(id); if (!p) continue; if (!kidsOf.has(p)) kidsOf.set(p, []); kidsOf.get(p).push(id); }
	const childrenOf = (id) => kidsOf.get(id) || [];
	const roots = m.order.filter((id) => !regionOf(id));
	const rootSet = new Set(roots);
	// Subtree attention (own + descendants) prunes branches nothing attended to.
	const subtotal = new Map(), calcd = new Set();
	const calc = (id) => {
		if (calcd.has(id)) return 0; calcd.add(id);
		let s = own(id);
		for (const c of childrenOf(id)) s += calc(c);
		subtotal.set(id, s); return s;
	};
	roots.forEach(calc);
	const attended = (id) => (subtotal.get(id) || 0) > 0;
	const INDENT = 18;

	// PASS 1 — lay out rows. Each zone gets its OWN line; the line directly beneath
	// holds its objects + frames stacked horizontally; child zones recurse indented.
	// A container (recurses) is a zone by kind OR anything with attended children,
	// so no object subtree is dropped; a leaf is an object/frame with no children.
	const isContainer = (id) => isZoneKind(id) || childrenOf(id).some(attended);
	const rows = [], seen = new Set();
	const walk = (id, depth) => {
		if (seen.has(id)) return; seen.add(id);
		rows.push({ depth, zone: true, ids: [id] });
		const ks = childrenOf(id).filter(attended);
		const leaves = ks.filter((c) => !isContainer(c)).sort((a, b) => own(b) - own(a));
		const zones = ks.filter((c) => isContainer(c)).sort((a, b) => (subtotal.get(b) || 0) - (subtotal.get(a) || 0));
		if (leaves.length) rows.push({ depth: depth + 1, zone: false, ids: leaves });
		for (const c of zones) walk(c, depth + 1);
	};
	roots.filter(attended).sort((a, b) => (subtotal.get(b) || 0) - (subtotal.get(a) || 0)).forEach((r) => walk(r, 0));
	if (!rows.length) return el("div", { class: "muted", style: "font-size:12px", text: "no attended scene entities in this window" });

	// PASS 2 — one global width scale so the WIDEST STACK fits with LEEWAY: normalize
	// every badge against the largest row total (a leaf row's summed attention, or a
	// lone zone badge) times FILL<100%, so a horizontal group of objects fills most
	// of the row but never crowds the edge, and every width stays comparable.
	const FILL = 86; // widest row reaches 86% of the width — the rest is breathing room
	const maxRow = Math.max(...rows.map((r) => r.ids.reduce((s, id) => s + own(id), 0)), 1e-9);
	const maxOne = Math.max(...[...score.values()].map((v) => Math.max(0, v)), 1e-9);
	const REL_MIN = 0.05; // label all but the faintest slivers (below 5% of the top entity)
	const badge = (id, zone) => {
		const n = m.nodes.get(id);
		const isRoot = rootSet.has(id);
		const hex = isRoot ? "#9aa7bd" : entityHex(n?.kind, id);
		const kindLab = isRoot ? "root" : entityKindLabel(n?.kind, id);
		const reg = regionOf(id);
		const total = subtotal.get(id) || 0;
		// The root is structural — it holds no tokens, so it never receives direct
		// attention. Draw it as the FULL-WIDTH scene container so every bit of
		// attention is visibly accounted for beneath it. Everything else keeps
		// width = its OWN attention on the shared maxRow scale.
		const wpct = isRoot ? FILL : (FILL * own(id)) / maxRow;
		const totNote = childrenOf(id).length ? ` · subtree total ${fmtNum(total)}` : "";
		const isTarget = id === targetZone;
		const b = el("div", {
			class: `tree-badge${zone ? " zone" : ""}${isRoot ? " root" : ""}${isTarget ? " target" : ""}`,
			style: `width:${wpct.toFixed(3)}%;background:${hex}2e;border-color:${hex}`,
			title: `${id} · ${kindLab}${reg ? ` · in ${reg}` : ""}${isTarget ? " · current step's zone" : ""} · attention ${fmtNum(own(id))}${totNote}`,
			dataset: { id, rel: isRoot ? "1" : (own(id) / maxOne).toFixed(4) },
			onclick: () => { jumpTo(id); applyAttnHighlight([{ id, weight: 1 }]); },
		}, el("span", { class: "tree-badge-lab", text: id }));
		treeHoverRegister(id, b); // badge <-> zone-plan phrase <-> 3D cross-highlight
		return b;
	};
	const SVGNS = "http://www.w3.org/2000/svg";
	const out = rows.map((r) => {
		const rowEl = el("div", { class: `tree-row ${r.zone ? "zone-row" : "leaf-row"}` }, ...r.ids.map((id) => badge(id, r.zone)));
		const lane = el("div", { class: "tree-labels" });
		const arms = document.createElementNS(SVGNS, "svg");
		arms.setAttribute("class", "tree-arms");
		const g = el("div", { class: `tree-rowgroup${r.zone ? " zone-grp" : ""}`, style: `margin-left:${r.depth * INDENT}px` }, rowEl, lane);
		g.appendChild(arms);
		return g;
	});

	// Post-layout label placement. A badge shows its name INLINE only when the text
	// fits its measured width; otherwise (if the entity is relevant enough) the name
	// moves to a label lane below, joined by a 3-segment leader arm. Faint entities
	// get neither — they stay a thin colored sliver. Widths are measured (not the
	// text hidden) so this is idempotent and safe to re-run on resize.
	const LABEL_H = 15, ARM_DROP = 8, ROW_GAP = 3, LGAP = 8, PAD = 12, MAX_LABELS = 24;
	const mctx = (buildTreeBody._mctx ||= document.createElement("canvas").getContext("2d"));
	const layoutRow = (group) => {
		const rowEl = group.querySelector(".tree-row");
		const lane = group.querySelector(".tree-labels");
		const arms = group.querySelector(".tree-arms");
		lane.replaceChildren();
		while (arms.firstChild) arms.removeChild(arms.firstChild);
		const badges = [...rowEl.children];
		if (!badges.length) return;
		// Clear glow bindings from a prior layout (badges persist across relayouts,
		// so their listeners would otherwise stack and point at stale labels/arms).
		for (const b of badges) {
			if (b._hoverGlowOn) { b.removeEventListener("mouseenter", b._hoverGlowOn); b.removeEventListener("mouseleave", b._hoverGlowOff); b._hoverGlowOn = b._hoverGlowOff = null; }
			b.classList.remove("tree-hot");
		}
		const cs = getComputedStyle(badges[0]);
		mctx.font = `${cs.fontSize} ${cs.fontFamily}`;
		let shorts = [];
		for (const b of badges) {
			const fits = mctx.measureText(b.dataset.id).width <= b.clientWidth - PAD - 2;
			b.classList.toggle("lab-out", !fits);
			if (!fits && Number(b.dataset.rel) >= REL_MIN) shorts.push(b);
		}
		// Cap leader labels per row — keep the most-attended, let the rest stay
		// slivers — so a dense row can't turn the lane into a thicket.
		if (shorts.length > MAX_LABELS) shorts = shorts.sort((a, z) => Number(z.dataset.rel) - Number(a.dataset.rel)).slice(0, MAX_LABELS);
		if (!shorts.length) { lane.style.height = "0px"; arms.style.display = "none"; return; }
		arms.style.display = "";
		const groupW = group.clientWidth;
		// Place every label DIRECTLY under its badge (preferred x = badge center),
		// then first-fit into stacked levels: an overlapping label drops to the next
		// line rather than being shoved sideways. This yields near-vertical arms that
		// don't cross, and labels that still read left-to-right in badge order.
		const items = shorts
			.map((b) => {
				const d = lane.appendChild(el("div", { class: "tree-label", text: b.dataset.id }));
				d.dataset.id = b.dataset.id;
				return { b, d };
			})
			.map((it) => ({ ...it, bc: it.b.offsetLeft + it.b.offsetWidth / 2, lw: it.d.offsetWidth }))
			.sort((a, z) => a.bc - z.bc);
		// Hovering a short badge OR its external label glows both (and the leader
		// arm) together. `it.p` (the arm path) is assigned in the arm loop below;
		// the handlers read it lazily, so it's linked by the time a hover fires.
		for (const it of items) {
			const on = () => { it.b.classList.add("tree-hot"); it.d.classList.add("tree-hot"); it.p?.classList.add("hot"); };
			const off = () => { it.b.classList.remove("tree-hot"); it.d.classList.remove("tree-hot"); it.p?.classList.remove("hot"); };
			it.b._hoverGlowOn = on; it.b._hoverGlowOff = off;
			it.b.addEventListener("mouseenter", on); it.b.addEventListener("mouseleave", off);
			it.d.addEventListener("mouseenter", on); it.d.addEventListener("mouseleave", off);
		}
		const levelRight = [];
		for (const it of items) {
			const px = Math.max(0, Math.min(it.bc - it.lw / 2, Math.max(0, groupW - it.lw)));
			let lvl = levelRight.findIndex((r) => px >= r + LGAP);
			if (lvl === -1) { lvl = levelRight.length; levelRight.push(0); }
			levelRight[lvl] = px + it.lw;
			it.x = px; it.lvl = lvl;
			it.d.style.left = `${px}px`;
			it.d.style.top = `${ARM_DROP + lvl * (LABEL_H + ROW_GAP)}px`;
		}
		const used = levelRight.length || 1;
		lane.style.height = `${ARM_DROP + used * (LABEL_H + ROW_GAP) + 2}px`;
		const gW = group.clientWidth, gH = group.clientHeight;
		arms.setAttribute("width", gW); arms.setAttribute("height", gH);
		arms.setAttribute("viewBox", `0 0 ${gW} ${gH}`);
		const by = rowEl.offsetHeight;
		for (const it of items) {
			const lcx = it.x + it.lw / 2;
			const ly = lane.offsetTop + ARM_DROP + it.lvl * (LABEL_H + ROW_GAP);
			const mid = by + Math.max(3, (ly - by) / 2);
			const p = document.createElementNS(SVGNS, "path");
			p.setAttribute("d", `M${it.bc.toFixed(1)} ${by}L${it.bc.toFixed(1)} ${mid.toFixed(1)}L${lcx.toFixed(1)} ${mid.toFixed(1)}L${lcx.toFixed(1)} ${ly.toFixed(1)}`);
			p.setAttribute("fill", "none");
			p.setAttribute("stroke", "rgba(255,255,255,0.26)");
			p.setAttribute("stroke-width", "1");
			arms.appendChild(p);
			it.p = p; // link the arm so hovering the badge/label can glow it too
		}
	};

	const legend = el("div", { class: "kindbar-legend tree-legend" },
		...[["root", "#9aa7bd"], ["zone", COLORS.zone], ["frame", COLORS.frame], ["object", COLORS.object]]
			.map(([k, hex]) => el("span", {}, el("i", { style: `background:${hex}` }), el("span", { text: k }))));
	const view = el("div", { class: "tree-view" }, ...out, legend);
	view._layout = () => out.forEach(layoutRow);

	// Companion heat tree: the SAME hierarchy and rows, but attention is encoded by
	// COLOR DEPTH (dark blue = low → bright yellow = high) instead of width, and
	// every badge in a row is uniform width. This reads structure and intensity at
	// a glance where the width tree reads magnitude.
	const heat = (t) => {
		t = Math.max(0, Math.min(1, t));
		return `hsl(${(212 - 162 * t).toFixed(0)}, 70%, ${(32 + 28 * t).toFixed(0)}%)`;
	};
	const heatBadge = (id, zone) => {
		const n = m.nodes.get(id);
		const isRoot = rootSet.has(id);
		const t = own(id) / maxOne;
		const kindLab = isRoot ? "root" : entityKindLabel(n?.kind, id);
		const reg = regionOf(id);
		const dark = !isRoot && t > 0.52;
		const isTarget = id === targetZone;
		const b = el("div", {
			class: `tree-badge heat${zone ? " zone" : ""}${isRoot ? " root" : ""}${isTarget ? " target" : ""}`,
			style: `background:${isRoot ? "#464d5b" : heat(t)};border-color:${isRoot ? "#6b7280" : heat(Math.min(1, t + 0.12))};color:${dark ? "#141414" : "#e8eefc"}`,
			title: `${id} · ${kindLab}${reg ? ` · in ${reg}` : ""}${isTarget ? " · current step's zone" : ""} · attention ${fmtNum(own(id))}`,
			onclick: () => { jumpTo(id); applyAttnHighlight([{ id, weight: 1 }]); },
		}, el("span", { class: "tree-badge-lab", text: id }));
		treeHoverRegister(id, b);
		return b;
	};
	const heatOut = rows.map((r) => el("div", { class: `tree-row ${r.zone ? "zone-row" : "leaf-row"}`, style: `margin-left:${r.depth * INDENT}px` }, ...r.ids.map((id) => heatBadge(id, r.zone))));
	const heatLegend = el("div", { class: "tree-heat-legend" },
		el("span", { text: "attention" }), el("span", { class: "thl-lo", text: "low" }),
		el("span", { class: "thl-bar" }), el("span", { class: "thl-hi", text: "high" }));
	const heatView = el("div", { class: "tree-view tree-heat" }, ...heatOut, heatLegend);

	// Each tree collapses on its own via a caret in its caption (persisted). When a
	// tree is re-opened, re-run its label layout (badge widths are only measurable
	// once visible), which is a no-op for the heat tree (uniform width, no arms).
	const subSection = (key, capText, viewEl) => {
		const folded = treeFoldGet(key);
		const caret = el("span", { class: "tree-fold-caret", text: folded ? "▸" : "▾" });
		const sub = el("div", { class: `tree-sub${folded ? " folded" : ""}` },
			el("button", {
				class: "tree-sub-cap", type: "button", title: "collapse or expand this tree",
				onclick: () => {
					const nowFolded = !sub.classList.contains("folded");
					sub.classList.toggle("folded", nowFolded);
					caret.textContent = nowFolded ? "▸" : "▾";
					treeFoldSet(key, nowFolded);
					if (!nowFolded) requestAnimationFrame(() => viewEl._layout?.());
				},
			}, caret, el("span", { text: capText })),
			viewEl);
		return sub;
	};
	// The main/headline tab shows ONLY the colored (heat) tree; the dedicated
	// "tree" tab shows both the width- and color-encoded views.
	return el("div", { class: "tree-dual" },
		...(heatOnly ? [] : [subSection("width", "badge width = attention", view)]),
		subSection("heat", "badge color = attention · uniform width", heatView));
}
