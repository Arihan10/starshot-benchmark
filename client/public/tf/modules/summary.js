// Per-step aggregate analytics. Everything here is derived from the CURRENT
// step's stored analysis (state.attn), aggregated over its generated tokens and
// instrumented (top-k) heads, to answer "what did this step attend to, overall?"
// — which entities, attributes, kinds, and WHEN across the generated sequence.

import { el } from "../../js/ui.js";
import { $, state, COLORS, compHex, entityHex, entityKindLabel } from "./state.js";
import { _mean, heatColor } from "./util.js";
import { aggregateAttn, hasToPlace, outputStartTok } from "./aggregate.js";
import { summaryBlock, summaryBar, statCard, wide } from "./widgets.js";
import { jumpTo } from "./exportPanel.js";
import { applyAttnHighlight, renderAttention } from "./attnPanel.js";
import { updateReportCtx } from "./tabs.js";
import * as report from "./report.js";

function renderSummary() {
	if (report.reportShowing("step")) { report.renderReportWorkspace(); return; }
	const host = $("tf-summary-panel");
	if (!host || !report.reportShowing("summary")) return; // only build when the report shows it (it's heavy)
	updateReportCtx();
	// Preserve scroll position across re-renders (step/token changes) so the user
	// stays on the chart they were reading.
	const scroller = $("tf-report-body");
	const savedTop = scroller ? scroller.scrollTop : 0;
	const paint = (...children) => { host.replaceChildren(...children); if (scroller) scroller.scrollTop = savedTop; };
	const step = state.steps[state.stepIdx];
	const displayed = !!(state.attn && step && state.attn.meta && state.attn.meta.event_index === step.event_index);
	if (!displayed) {
		paint(el("div", { class: "empty-note" },
			el("span", { class: "big", text: "📊" }),
			el("span", { text: "compute this step's attention to see its aggregate summary" })));
		return;
	}
	const a = state.attn;
	const agg = aggregateAttn(a);
	if (!agg.entityTotals.length) {
		paint(el("div", { class: "empty-note" }, el("span", { text: "no scene attention recorded for this step" })));
		return;
	}
	paint(...[
		el("div", { class: "rep-note",
			text: `aggregated over ${a.tokens.length} generated tokens × ${(a.selected_heads || []).length} instrumented heads` }),
		wide(summaryKeyStats(a, agg)),
		wide(summaryTrajectory(a, agg)),      // the sequence-over-time chart reads best full width
		wide(summaryEntityTokenMap(a, agg)),  // ditto the entity × token heatmap
		summaryToPlace(a),           // bbox-batch: what the step attends to while placing (null otherwise)
		summaryEntities(agg),
		summaryReasoningVsOutput(a),
		summaryAttributes(agg),
		summaryKind(agg),
		summaryParents(a, agg),
		summaryHeadEntity(a, agg),
		summaryLayerDepth(a),
	].filter(Boolean));
}

// TO-PLACE readout (bbox-batch steps only): how much the step attends to the
// objects being PLACED vs. the surrounding scene, plus the top placed objects and
// the attributes it leans on. Returns null for steps with no to-place batch.
export function summaryToPlace(a) {
	if (!hasToPlace(a)) return null;
	const tp = a.agg.to_place; // { mass[], entityTotals, componentTotals }
	const tpMass = _mean(tp.mass), scMass = _mean(a.agg.mass || []);
	const splitMax = Math.max(tpMass, scMass, 1e-9);
	const split = el("div", { class: "sbars" },
		summaryBar({ color: COLORS.to_place, label: "to-place (self)", value: tpMass, max: splitMax, title: "mean attention mass on the objects being placed this step" }),
		summaryBar({ color: "#4af0e0", label: "scene (context)", value: scMass, max: splitMax, title: "mean attention mass on the surrounding scene" }),
	);
	const top = tp.entityTotals.slice(0, 12);
	const emax = top[0]?.score ?? 1;
	const objs = top.length
		? el("div", { class: "sbars" }, ...top.map((e) => summaryBar({
			color: COLORS.to_place, label: e.id, value: e.score, max: emax,
			title: `${e.id} · Σ ${e.score.toFixed(4)} — click → focus in 3D`,
			onclick: () => { jumpTo(e.id); applyAttnHighlight([{ id: e.id, weight: 1 }]); },
		})))
		: el("div", { class: "muted", style: "font-size:11px", text: "—" });
	const attrs = tp.componentTotals.slice(0, 12);
	const amax = attrs[0]?.score ?? 1;
	const attrBars = attrs.length
		? el("div", { class: "sbars" }, ...attrs.map((c) => summaryBar({ color: "#8ab4ff", label: c.component, value: c.score, max: amax, title: `${c.component} · Σ ${c.score.toFixed(4)}` })))
		: null;
	return summaryBlock("to-place · objects being placed",
		"attention on the batch being placed vs. the scene · the placed objects it looks at most · click → focus in 3D",
		el("div", { class: "exp-lab", style: "margin-top:2px", text: "self vs scene" }), split,
		el("div", { class: "exp-lab", style: "margin-top:8px", text: "top placed objects" }), objs,
		...(attrBars ? [el("div", { class: "exp-lab", style: "margin-top:8px", text: "attributes leaned on" }), attrBars] : []),
	);
}

// Scalar headline stats — the numbers you'd quote about this step's attention.
export function summaryKeyStats(a, agg) {
	const out = outputStartTok(a);
	const peak = agg.mass.reduce((b, v, i) => (v > b.v ? { v, i } : b), { v: 0, i: 0 });
	const rMass = _mean(agg.mass.slice(0, out)), oMass = _mean(agg.mass.slice(out));
	const topE = agg.entityTotals[0], topC = agg.componentTotals[0];
	const nCtx = a.meta?.n_scene_entities ?? 0;
	const cards = [
		statCard("mean scene mass", `${(_mean(agg.mass) * 100).toFixed(1)}%`),
		statCard("peak scene mass", `${(peak.v * 100).toFixed(1)}%`, `@ token ${peak.i + 1}`),
		statCard("reasoning → output", `${(rMass * 100).toFixed(0)}→${(oMass * 100).toFixed(0)}%`, "mean mass"),
		statCard("mean entropy", _mean(agg.entropy).toFixed(3)),
		statCard("entities attended", `${agg.entityTotals.length}`, nCtx ? `of ${nCtx}` : null),
		statCard("top entity", topE ? topE.id : "—", topE ? entityKindLabel(topE.kind, topE.id) : null),
		statCard("top attribute", topC ? topC.component : "—"),
	];
	return summaryBlock("key statistics", null, el("div", { class: "stat-grid" }, ...cards));
}

// What the model attends to WHILE REASONING vs WHILE EMITTING the answer —
// two ranked entity lists side by side (only when the step has both phases).
export function summaryReasoningVsOutput(a) {
	const out = outputStartTok(a);
	if (out <= 0 || out >= (a.tokens || []).length) return null;
	const col = (title, sub, entityTotals) => {
		const max = entityTotals[0]?.score ?? 1;
		return el("div", { class: "rvo-col" },
			el("div", { class: "rvo-head" }, el("span", { text: title }), el("span", { class: "muted", text: sub })),
			entityTotals.length
				? el("div", { class: "sbars" }, ...entityTotals.slice(0, 6).map((e) => summaryBar({
					color: entityHex(e.kind, e.id), label: e.id, value: e.score, max,
					title: `${e.id} · ${e.score.toFixed(4)}`,
					onclick: () => { jumpTo(e.id); applyAttnHighlight([{ id: e.id, weight: 1 }]); },
				})))
				: el("div", { class: "muted", style: "font-size:11px", text: "—" }),
		);
	};
	const mass = a.agg.mass || [];
	const rMass = _mean(mass.slice(0, out)), oMass = _mean(mass.slice(out));
	const R = (a.agg.reasoning && a.agg.reasoning.entityTotals) || [];
	const O = (a.agg.output && a.agg.output.entityTotals) || [];
	return summaryBlock("reasoning vs output — what each phase attends to",
		"top entities while thinking (left) vs while emitting the answer (right)",
		el("div", { class: "rvo" },
			col("reasoning", `${(rMass * 100).toFixed(0)}% mass`, R),
			col("output", `${(oMass * 100).toFixed(0)}% mass`, O)));
}

// Attention rolled up to each attended entity's PARENT region — structural focus
// (does the step concentrate on one zone's contents?).
export function summaryParents(a, agg) {
	const parentOf = new Map((a.scene_entities || []).map((e) => [e.id, e.parent]));
	const roll = new Map();
	for (const e of agg.entityTotals) { const p = parentOf.get(e.id) || "(root)"; roll.set(p, (roll.get(p) || 0) + e.score); }
	const items = [...roll.entries()].map(([id, score]) => ({ id, score })).sort((x, y) => y.score - x.score).slice(0, 10);
	if (items.length <= 1) return null;
	const max = items[0].score;
	const bars = items.map((it) => summaryBar({
		color: it.id === "(root)" ? "#888" : COLORS.zone, label: it.id, value: it.score, max,
		title: `${it.id} · Σ ${it.score.toFixed(4)}`,
		onclick: it.id === "(root)" ? null : () => { jumpTo(it.id); applyAttnHighlight([{ id: it.id, weight: 1 }]); },
	}));
	return summaryBlock("attention by parent region",
		"attention rolled up to each entity's parent zone — structural focus",
		el("div", { class: "sbars" }, ...bars));
}

// Head specialization: rows = top entities, cols = instrumented heads, cell =
// how much that head attends the entity. Do heads divide the scene between them?
export function summaryHeadEntity(a, agg) {
	const heads = a.selected_heads || [];
	const he = (a.agg && a.agg.head_entity) || { entities: [], M: [] };
	const top = (he.entities || []).slice(0, 10);
	if (heads.length < 2 || !top.length) return null;
	const M = (he.M || []).slice(0, top.length); // [entity][head] — precomputed
	const max = Math.max(...M.flat(), 1e-9);
	// Keep each head column wide enough for its L#H# label; when there are many
	// heads the matrix outgrows the drawer and scrolls horizontally rather than
	// squashing the labels to nothing.
	const colW = 30;
	const cols = `grid-template-columns:96px repeat(${heads.length},minmax(${colW}px,1fr))`;
	const header = el("div", { class: "hg-row", style: cols },
		el("div", { class: "hg-corner", text: "" }),
		...heads.map((h, i) => el("div", { class: `hg-h${i === state.attnHead ? " cur-h" : ""}`, text: `L${h.layer}H${h.head}` })));
	const body = top.map((e, r) => el("div", { class: "hg-row", style: cols },
		el("div", { class: "hg-el", title: e.id, text: e.id }),
		...heads.map((h, hi) => el("div", {
			class: `hg-c sel${hi === state.attnHead ? " cur" : ""}`, style: `background:${heatColor(M[r][hi] / max)}`,
			title: `${e.id} · L${h.layer}H${h.head} · ${M[r][hi].toFixed(4)} — click to select this head`,
			onclick: () => { state.attnHead = hi; renderAttention(); },
		}))));
	return summaryBlock("head × entity — specialization",
		"rows = top entities · cols = instrumented heads · brighter = that head attends it more · click to select a head",
		el("div", { class: "hg-scroll" },
			el("div", { class: "hg-wrap", style: `min-width:${96 + heads.length * colW}px` }, header, ...body)));
}

// Where scene-tracking lives in the network: mean scene mass per global layer
// (averaged over all its heads), in depth order.
export function summaryLayerDepth(a) {
	const byLayer = new Map();
	for (const g of a.head_grid || []) { const c = byLayer.get(g.layer) || { s: 0, n: 0 }; c.s += g.mean_scale; c.n++; byLayer.set(g.layer, c); }
	const items = [...byLayer.entries()].map(([layer, { s, n }]) => ({ layer, mean: n ? s / n : 0 })).sort((x, y) => x.layer - y.layer);
	if (items.length < 2) return null;
	const max = Math.max(...items.map((i) => i.mean), 1e-9);
	const bars = items.map((it) => summaryBar({
		color: heatColor(it.mean), label: `L${it.layer}`, value: it.mean, max,
		title: `layer ${it.layer} · mean scene mass ${(it.mean * 100).toFixed(1)}% across all heads`,
	}));
	return summaryBlock("scene attention by layer depth",
		"mean scene mass per global layer (all heads) — where scene-tracking concentrates",
		el("div", { class: "sbars" }, ...bars));
}

// Which scene entities the step attends to most (Σ over all tokens + heads).
export function summaryEntities(agg) {
	const items = agg.entityTotals.slice(0, 15);
	const max = items[0]?.score ?? 1;
	const bars = items.map((e) => summaryBar({
		color: entityHex(e.kind, e.id), label: e.id, value: e.score, max,
		title: `${e.id} · ${entityKindLabel(e.kind, e.id)} · Σ attention ${e.score.toFixed(4)} — click to focus in 3D`,
		onclick: () => { jumpTo(e.id); applyAttnHighlight([{ id: e.id, weight: 1 }]); },
	}));
	return summaryBlock("scene entities by attention",
		"which parts of the scene this step looks at · Σ over tokens · click → focus in 3D",
		el("div", { class: "sbars" }, ...bars));
}

// Which entity ATTRIBUTES (components) attract attention across the step.
function summaryAttributes(agg) {
	if (!agg.componentTotals.length) return null;
	const items = agg.componentTotals.slice(0, 14);
	const max = items[0]?.score ?? 1;
	const bars = items.map((c) => summaryBar({
		color: compHex(c.component), label: c.component, value: c.score, max,
		title: `${c.component} · Σ attention ${c.score.toFixed(4)}`,
	}));
	return summaryBlock("attributes by attention",
		"which entity attributes drive attention (placement, dimensions, relationships, proxy_shape, …)",
		el("div", { class: "sbars" }, ...bars));
}

// Zone vs object vs frame share of the step's scene attention.
export function summaryKind(agg) {
	const total = (agg.kindTotals.zone + agg.kindTotals.object + agg.kindTotals.frame) || 1e-9;
	const segs = [["zone", COLORS.zone], ["object", COLORS.object], ["frame", COLORS.frame]]
		.map(([k, hex]) => ({ k, hex, pct: (100 * agg.kindTotals[k]) / total }))
		.filter((s) => s.pct > 0);
	const bar = el("div", { class: "kindbar" },
		...segs.map((s) => el("div", { class: "seg", style: `width:${s.pct}%;background:${s.hex}`, title: `${s.k} · ${s.pct.toFixed(1)}%` })));
	const legend = el("div", { class: "kindbar-legend" },
		...segs.map((s) => el("span", {}, el("i", { style: `background:${s.hex}` }), el("span", { text: `${s.k} ${s.pct.toFixed(0)}%` }))));
	return summaryBlock("attention by entity kind", null, bar, legend);
}

// Temporal map: rows = top entities, cols = generated tokens, cell = attention
// (aggregated over heads). Reveals WHICH object is attended WHEN. Click to scrub
// the token (+ focus the row's entity in 3D).
export function summaryEntityTokenMap(a, agg) {
	const tokens = a.tokens || [];
	const et = (a.agg && a.agg.entity_token) || { entities: [], M: [] };
	const top = et.entities || [];
	if (!top.length || !tokens.length) return null;
	const M = et.M || []; // [entity][token] — precomputed (summed over heads)
	const mx = Math.max(...M.flat(), 1e-9);
	const rows = top.length, cols = tokens.length, rh = 15, bandH = 8;
	const cw = Math.max(2, Math.min(14, Math.floor(500 / Math.max(cols, 1))));
	const outStart = outputStartTok(a);
	const band = el("canvas", { class: "attn-heat" }); band.width = cols * cw; band.height = bandH;
	const bx = band.getContext("2d");
	for (let c = 0; c < cols; c++) { bx.fillStyle = c >= outStart ? "#7a4fd0" : "#2f6feb"; bx.fillRect(c * cw, 0, cw, bandH); }
	bx.strokeStyle = "#fff"; bx.strokeRect(state.attnToken * cw + 0.5, 0.5, Math.max(cw - 1, 1), bandH - 1);
	const canvas = el("canvas", { class: "attn-heat" }); canvas.width = cols * cw; canvas.height = rows * rh;
	const ctx = canvas.getContext("2d");
	for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) { ctx.fillStyle = heatColor(M[r][c] / mx); ctx.fillRect(c * cw, r * rh, cw, rh); }
	ctx.strokeStyle = "#fff"; ctx.strokeRect(state.attnToken * cw + 0.5, 0.5, Math.max(cw - 1, 1), rows * rh - 1);
	canvas.onclick = (ev) => {
		const rect = canvas.getBoundingClientRect();
		const c = Math.floor((ev.clientX - rect.left) / cw), r = Math.floor((ev.clientY - rect.top) / rh);
		if (c >= 0 && c < cols) state.attnToken = c;
		if (r >= 0 && r < rows) { jumpTo(top[r].id); applyAttnHighlight([{ id: top[r].id, weight: 1 }]); }
		renderAttention();
	};
	band.onclick = (ev) => { const rect = band.getBoundingClientRect(); const c = Math.floor((ev.clientX - rect.left) / cw); if (c >= 0 && c < cols) { state.attnToken = c; renderAttention(); } };
	const labels = el("div", { class: "attn-heat-labels" },
		el("div", { style: `height:${bandH}px` }),
		...top.map((e) => el("div", { class: "attn-heat-label", title: e.id, style: `height:${rh}px;line-height:${rh}px`, text: e.id })));
	return summaryBlock("entity × token map",
		"rows = top entities · cols = generated tokens → · brighter = more attention · click to scrub + focus",
		el("div", { class: "attn-heat-wrap" }, labels, el("div", { class: "attn-heat-scroll" }, band, canvas)));
}

// Scene-mass (how much of attention lands on the scene) + entropy (how spread
// out) across the generated sequence, mean over heads. Shows WHEN the model
// consults the scene while reasoning/emitting.
export function summaryTrajectory(a, agg) {
	const n = (a.tokens || []).length;
	if (!n) return null;
	const W = Math.min(600, Math.max(220, n * 6)), H = 88, pad = 5;
	const cw = W / n;
	const outStart = outputStartTok(a);
	const canvas = el("canvas", { class: "attn-heat", style: `width:100%;max-width:${W}px` });
	canvas.width = W; canvas.height = H;
	const ctx = canvas.getContext("2d");
	ctx.fillStyle = "#0a0b0e"; ctx.fillRect(0, 0, W, H);
	if (outStart > 0 && outStart < n) { ctx.fillStyle = "rgba(122,79,208,0.14)"; ctx.fillRect(outStart * cw, 0, (n - outStart) * cw, H); }
	const y = (v) => H - pad - Math.max(0, Math.min(1, v)) * (H - 2 * pad);
	ctx.beginPath(); ctx.moveTo(cw / 2, H);
	agg.mass.forEach((v, i) => ctx.lineTo(i * cw + cw / 2, y(v)));
	ctx.lineTo((n - 1) * cw + cw / 2, H); ctx.closePath();
	ctx.fillStyle = "rgba(74,240,224,0.16)"; ctx.fill();
	ctx.beginPath(); agg.mass.forEach((v, i) => { const X = i * cw + cw / 2, Y = y(v); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
	ctx.strokeStyle = "#4af0e0"; ctx.lineWidth = 1.5; ctx.stroke();
	ctx.beginPath(); agg.entropy.forEach((v, i) => { const X = i * cw + cw / 2, Y = y(v); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
	ctx.strokeStyle = "#e0a94a"; ctx.lineWidth = 1; ctx.setLineDash([3, 2]); ctx.stroke(); ctx.setLineDash([]);
	ctx.strokeStyle = "#fff"; ctx.lineWidth = 1;
	ctx.beginPath(); ctx.moveTo(state.attnToken * cw + cw / 2, 0); ctx.lineTo(state.attnToken * cw + cw / 2, H); ctx.stroke();
	const scrub = (ev) => { const rect = canvas.getBoundingClientRect(); const c = Math.floor((ev.clientX - rect.left) / (rect.width / n)); if (c >= 0 && c < n) { state.attnToken = c; renderAttention(); } };
	canvas.onclick = scrub;
	// Drag-scrub too, so moving through tokens updates the readout live.
	canvas.onpointermove = (ev) => { if (ev.buttons & 1) scrub(ev); };
	const legend = el("div", { class: "seg-legend", style: "font-size:10px;gap:12px;margin-top:5px" },
		el("span", {}, el("span", { class: "frame-sw", style: "background:#4af0e0" }), el("span", { text: " scene mass" })),
		el("span", {}, el("span", { class: "frame-sw", style: "background:#e0a94a" }), el("span", { text: " entropy (spread)" })),
	);
	// Live readout of the token under the cursor — the actual generated word as
	// you move through the sequence, with its frame + what it emits.
	const ti = Math.max(0, Math.min(state.attnToken, n - 1));
	const tokRec = a.tokens[ti];
	const isOut = ti >= outStart;
	const readout = el("div", { class: "traj-readout" },
		el("span", { class: `frame-tag ${isOut ? "out" : "reason"}`, text: isOut ? "output" : "reasoning" }),
		el("span", { class: "muted", text: ` token ${ti + 1}/${n} · #${tokRec.index} ` }),
		el("code", { class: "traj-word", text: tokRec.text }),
		tokRec.output_entity ? el("span", { class: "muted", text: ` · emits “${tokRec.output_entity}”` }) : null,
	);
	return summaryBlock("attention over the generated sequence",
		"mean over heads · shaded = output region · white = current token · click/drag to scrub",
		readout, canvas, legend);
}
