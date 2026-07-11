// Attention tab: the per-step map. A compact section head (status + recompute)
// then the body, ordered by importance — the token→scene heatmap, the selected
// token's attended objects, output→objects, and the always-visible heads×depth
// overview. Also owns the 3D cross-highlight for the current (token, head).

import { el } from "../../js/ui.js";
import { $, state, COLORS, COMPONENT_ABBR, compHex, entityHex, entityKindLabel, ATTN_STATUS_LABEL } from "./state.js";
import { heatColor } from "./util.js";
import { outputStartTok } from "./aggregate.js";
import { jumpTo } from "./exportPanel.js";
import { currentTokenHeads } from "./render.js";
import { computeStep } from "./attnQueue.js";
import { renderPlacement } from "./placement.js";
import { longestComputedRun } from "./presentCtx.js";
import { isAttn3DWindowOpen, refreshAttn3DWindow } from "./present.js";
import * as report from "./report.js";

// Fills #tf-attn-panel: a compact section head (status + recompute) then the
// body, resolved deterministically from the step's status (never stuck on a
// generic "loading"). Per-step + queue status now live in the timeline bar.
export function renderAttention() {
	const host = $("tf-attn-panel");
	if (!host) return;
	if (!state.steps.length) { host.replaceChildren(el("div", { class: "empty-note", text: "no steps in this cell" })); return; }
	const step = state.steps[state.stepIdx];
	const ev = step.event_index;
	const st = state.attnStatus[ev] || "none";
	// A failed (re)compute must not leave the old/stale map on screen — drop it so
	// the panel shows the failure, not resurrected outdated data.
	if (st === "error" && state.attn?.meta?.event_index === ev) state.attn = null;
	const e = state.export;
	const displayed = !!(state.attn && state.attn.meta && state.attn.meta.event_index === ev);

	const busy = st === "running" || st === "queued";
	const hasResult = displayed || st === "ready" || st === "stale";
	// Scene-less steps (root plans / overall bbox) have nothing to attend to —
	// don't offer to compute them (matches the compute-all filter).
	const noScene = step.has_scene === false && !hasResult;
	const canCompute = !!e && !busy && !noScene;
	const computeBtn = el("button", {
		id: "tf-attn-recompute",
		text: hasResult ? "↻ recompute" : "▶ compute",
		title: "compute the full attention map for THIS step only (⚡ compute all steps in the top bar does the whole cell)",
		...(canCompute ? {} : { disabled: "" }),
		onclick: () => { if (canCompute) computeStep(ev, hasResult); },
	});
	const head = el("div", { class: "drawer-sec-head" },
		el("span", { text: "attention" }),
		el("span", { id: "tf-attn-stpill", class: `attn-stpill st-${st}`, text: ATTN_STATUS_LABEL[st] }),
		el("span", { style: "margin-left:auto" }),
		computeBtn,
	);

	let body;
	if (displayed) {
		// Show the map; if it's from an old analysis version, flag it above.
		body = st === "stale"
			? el("div", {}, el("div", { class: "stale-banner", text: "⚠ stale — this map is from an older analysis version. Recompute to refresh it." }), attnBody(state.attn))
			: attnBody(state.attn);
	}
	else if (!e) body = el("div", { class: "empty-note", text: `loading step ${state.stepIdx + 1}…` });
	else if (st === "running") body = el("div", { class: "empty-note", text: "computing on Modal GPU…" });
	else if (st === "queued") body = el("div", { class: "empty-note", text: "queued — waiting for a free GPU slot…" });
	else if (st === "ready" || st === "stale") body = el("div", { class: "empty-note", text: "loading result…" });
	else if (st === "error") body = el("div", { class: "empty-note", text: `compute failed${state.attnErrors?.[ev] ? `: ${state.attnErrors[ev]}` : ""} — click ‘recompute’ to retry.` });
	else if (noScene) body = el("div", { class: "empty-note", text: "no scene context — this step has no scene entities to attend to, so it's skipped by ‘compute all’." });
	else body = el("div", { class: "empty-note" }, el("span", { class: "big", text: "▶" }), el("span", { text: "not computed — click ‘compute’ above, or ‘⚡ compute all steps’ for the whole cell." }));

	const wrap = el("div", { class: "attn-body" }, body);
	host.replaceChildren(head, wrap);
	// Present menu is available if THIS step is displayed (single-step) OR there's
	// a computed run to stitch end-to-end.
	const presentBtn = $("tf-present-btn");
	if (presentBtn) presentBtn.disabled = !(displayed || longestComputedRun().length);
	// Cross-highlight the attended scene entities in 3D for the current token/head.
	// While the 3D popup window is open it OWNS the viewer shading (its aggregate),
	// so refresh it in place instead of painting the token/head highlight over it.
	if (displayed) { if (isAttn3DWindowOpen()) refreshAttn3DWindow(); else applyAttnHighlight(); }
	else if (!isAttn3DWindowOpen()) state.viewer?.clearAttnHighlight?.();
	if (state.reportView) { state.reportSelections.token = state.attnToken; state.reportSelections.head = state.attnHead; report.saveReportState(); report.renderReportWorkspace(); }
	renderPlacement(); // and the placement tab (bbox steps)
}

// The attention body, ordered by importance (the point of the tool): the
// token→scene map (heatmap → the selected token's attended objects →
// output→objects), then the always-visible heads×depth overview.
function attnBody(a) {
	const heads = a.selected_heads || [];
	const tokens = a.tokens || [];
	if (!heads.length || !tokens.length) return el("div", { class: "muted", text: "no scene-attending heads / query tokens for this step" });
	state.attnHead = Math.min(state.attnHead, heads.length - 1);
	state.attnToken = Math.min(state.attnToken, tokens.length - 1);
	const metaLine = el("div", { class: "muted", style: "font-size:11px;margin-bottom:8px",
		text: `${tokens.length} generated tokens${a.meta.subsampled ? ` (sampled of ${a.meta.n_query_tokens_total})` : ""} · ${a.meta.n_scene_tokens} scene tokens · ${a.meta.n_scene_entities} entities` });
	return el("div", {}, metaLine, heatmap(a), tokenDetail(a), outputsSection(a), headGrid(a));
}

// Below this per-head scene mass, the token barely attends the scene, so the
// (normalized) top entities are noise — skip the 3D highlight rather than light
// up objects spuriously. Tunable; mirrors present.js.
const MIN_SCENE_MASS = 0.02;

// The scene entities the current (token, head) attends to, weighted 0..1 by
// score relative to the strongest — used for the 3D cross-highlight. Returns
// nothing when the head's scene mass is too low to be meaningful.
function currentAttended() {
	const a = state.attn;
	const tok = a?.tokens?.[state.attnToken];
	if (!tok) return [];
	// Gate on the (compact) per-head scene mass before pulling detail — no point
	// lighting up objects for a token that barely looks at the scene.
	if ((tok.hscale?.[state.attnHead] ?? 0) < MIN_SCENE_MASS) return [];
	const heads = currentTokenHeads(); // lazy — empty until the detail arrives
	const ents = (heads && heads[state.attnHead]?.top_entities) || [];
	if (!ents.length) return [];
	const max = Math.max(...ents.map((e) => e.score), 1e-9);
	return ents.map((e) => ({ id: e.id, weight: e.score / max }));
}

// Push a weighted highlight set into the 3D viewer (defaults to the current
// token/head's attended entities).
export function applyAttnHighlight(items) {
	state.viewer?.setAttnHighlight?.(items ?? currentAttended());
}

// Per-output-assignment view: for each emitted assignment, the scene objects
// its tokens attended to (aggregated over the current head) — the "which
// objects did THIS output rely on" map. Click a row to inspect + highlight it.
export function outputsSection(a) {
	// Token indices per output object (compact tokens carry output_entity) — for
	// the click-to-scrub + "current" highlight; the attended-object chips come
	// from the precomputed per-object rollup (summed over all instrumented heads).
	const idxs = new Map();
	(a.tokens || []).forEach((t, i) => {
		if (t.output_entity == null) return;
		if (!idxs.has(t.output_entity)) idxs.set(t.output_entity, []);
		idxs.get(t.output_entity).push(i);
	});
	const outs = (a.agg && a.agg.outputs) || [];
	if (!outs.length) return null;
	const rows = outs.map((o) => {
		const gi = idxs.get(o.entity) || [];
		const top = (o.scene?.entityTotals || []).slice(0, 12);
		const chips = top.length
			? top.map((e) => el("span", {
				class: "chip clickable", style: `background:${e.kind === "zone" ? COLORS.zone : COLORS.object}`,
				title: `${e.id} · Σ score ${e.score.toFixed(3)} — click to focus in 3D`, text: e.id,
				onclick: (ev) => { ev.stopPropagation(); jumpTo(e.id); },
			}))
			: [el("span", { class: "muted", text: "— no scene attention" })];
		return el("div", {
			class: `out-row${gi.includes(state.attnToken) ? " cur" : ""}`, title: `inspect “${o.entity}” (${o.n} token${o.n === 1 ? "" : "s"})`,
			onclick: () => { if (gi.length) state.attnToken = gi[0]; renderAttention(); jumpTo(o.entity); },
		},
			el("div", { class: "out-name" }, el("span", { class: "map-sw", style: `background:${COLORS.output}` }), el("span", { text: ` ${o.entity}` }), el("span", { class: "muted", text: ` · ${o.n} tok` })),
			el("div", { class: "out-chips" }, ...chips),
		);
	});
	return el("div", { class: "exp-block" },
		el("div", { class: "exp-lab", text: "output → attended scene objects (all instrumented heads · click a row to inspect + highlight)" }),
		el("div", { class: "out-list" }, ...rows),
	);
}

// The across-heads / across-depths overview: rows = global layers (network
// depth ↓), columns = attention heads, cell = MEAN scene mass for that
// (layer, head) over the sampled steps. The instrumented top-k heads (the only
// ones with per-token detail) are outlined — click one to drill into it below.
export function headGrid(a) {
	const grid = a.head_grid || [];
	if (!grid.length) return null;
	const layers = [...new Set(grid.map((g) => g.layer))].sort((x, y) => x - y);
	const heads = [...new Set(grid.map((g) => g.head))].sort((x, y) => x - y);
	const val = new Map(grid.map((g) => [`${g.layer}:${g.head}`, g.mean_scale]));
	const selIdx = new Map((a.selected_heads || []).map((h, i) => [`${h.layer}:${h.head}`, i]));
	const cur = (a.selected_heads || [])[state.attnHead];
	const curKey = cur ? `${cur.layer}:${cur.head}` : null;
	const cols = `grid-template-columns:30px repeat(${heads.length},1fr)`;

	const header = el("div", { class: "hg-row", style: cols },
		el("div", { class: "hg-corner", text: "L╲H" }),
		...heads.map((hd) => el("div", { class: "hg-h", text: String(hd) })),
	);
	const body = layers.map((ly) => el("div", { class: "hg-row", style: cols },
		el("div", { class: "hg-l", text: `L${ly}` }),
		...heads.map((hd) => {
			const key = `${ly}:${hd}`;
			const v = val.get(key);
			const isSel = selIdx.has(key);
			return el("div", {
				class: `hg-c${isSel ? " sel" : ""}${key === curKey ? " cur" : ""}`,
				style: `background:${v == null ? "var(--panel-2)" : heatColor(v)}`,
				title: `layer ${ly} · head ${hd}${v == null ? "" : ` · mean scene mass ${(v * 100).toFixed(1)}%`}${isSel ? " · instrumented (click to inspect)" : ""}`,
				onclick: isSel ? () => { state.attnHead = selIdx.get(key); renderAttention(); } : null,
			});
		}),
	));
	return el("div", { class: "exp-block" },
		el("div", { class: "exp-lab", text: `scene attention across heads × depth — ${layers.length} global layers × ${heads.length} heads, mean scene mass cᵢ` }),
		el("div", { class: "hg-wrap" }, header, ...body),
		el("div", { class: "hint", text: "rows = layer depth ↓ · cols = heads → · outlined = instrumented (click to inspect) · cyan = current" }),
	);
}

// The layerwise 2D graph: rows = selected heads (grouped by layer), columns =
// query steps, cell color = scene mass c_i. Click a cell to inspect that
// (head, step). Canvas so it scales to long completions.
function heatmap(a) {
	const heads = a.selected_heads, tokens = a.tokens;
	const rows = heads.length, cols = tokens.length;
	// Shrink the per-head row as head count grows so a top-32 map stays compact
	// (32 x 15 = 480px is unwieldy) while a top-4 map keeps its comfortable size.
	const rh = rows > 24 ? 9 : rows > 12 ? 12 : 15, bandH = 8;
	const cw = Math.max(2, Math.min(12, Math.floor(560 / Math.max(cols, 1))));
	const outStart = outputStartTok(a);

	// reasoning/output frame band — one strip above the map, per token column.
	const band = el("canvas", { class: "attn-heat" });
	band.width = cols * cw; band.height = bandH;
	const bx = band.getContext("2d");
	for (let c = 0; c < cols; c++) { bx.fillStyle = c >= outStart ? "#7a4fd0" : "#2f6feb"; bx.fillRect(c * cw, 0, cw, bandH); }
	bx.strokeStyle = "#fff"; bx.strokeRect(state.attnToken * cw + 0.5, 0.5, Math.max(cw - 1, 1), bandH - 1);

	const canvas = el("canvas", { class: "attn-heat" });
	canvas.width = cols * cw;
	canvas.height = rows * rh;
	const ctx = canvas.getContext("2d");
	for (let r = 0; r < rows; r++) {
		for (let c = 0; c < cols; c++) {
			ctx.fillStyle = heatColor(tokens[c].hscale?.[r] ?? 0);
			ctx.fillRect(c * cw, r * rh, cw, rh);
		}
	}
	// reasoning→output divider
	if (outStart > 0 && outStart < cols) {
		ctx.strokeStyle = "#b46aff"; ctx.lineWidth = 1;
		ctx.beginPath(); ctx.moveTo(outStart * cw + 0.5, 0); ctx.lineTo(outStart * cw + 0.5, rows * rh); ctx.stroke();
	}
	// highlight current column (step) + row (head)
	ctx.strokeStyle = "#fff"; ctx.lineWidth = 1;
	ctx.strokeRect(state.attnToken * cw + 0.5, 0.5, Math.max(cw - 1, 1), rows * rh - 1);
	ctx.strokeStyle = "#4af0e0";
	ctx.strokeRect(0.5, state.attnHead * rh + 0.5, cols * cw - 1, rh - 1);
	const pick = (ev, canv) => {
		const rect = canv.getBoundingClientRect();
		const c = Math.floor((ev.clientX - rect.left) / cw);
		const r = Math.floor((ev.clientY - rect.top) / rh);
		if (c >= 0 && c < cols) state.attnToken = c;
		if (canv === canvas && r >= 0 && r < rows) state.attnHead = r;
		renderAttention();
	};
	canvas.onclick = (ev) => pick(ev, canvas);
	band.onclick = (ev) => pick(ev, band); // clicking the band scrubs the token too
	const labels = el("div", { class: "attn-heat-labels" },
		el("div", { style: `height:${bandH}px` }), // spacer aligning labels with the map rows (below the band)
		...heads.map((h, i) => el("div", {
			class: `attn-heat-label${i === state.attnHead ? " on" : ""}`, style: `height:${rh}px;line-height:${rh}px;font-size:${Math.min(10, rh)}px`,
			title: `global layer ${h.layer}, head ${h.head} · mean scene scale ${h.mean_scale.toFixed(3)}`,
			text: `L${h.layer}·H${h.head}`, onclick: () => { state.attnHead = i; renderAttention(); },
		})));
	const legend = el("div", { class: "muted", style: "font-size:10px;margin:2px 0 3px;display:flex;gap:10px;align-items:center" },
		el("span", {}, el("span", { class: "frame-sw", style: "background:#2f6feb" }), el("span", { text: " reasoning" })),
		el("span", {}, el("span", { class: "frame-sw", style: "background:#7a4fd0" }), el("span", { text: " output" })),
	);
	return el("div", { class: "exp-block" },
		el("div", { class: "exp-lab", text: "heads × generated tokens — scene mass cᵢ" }),
		legend,
		el("div", { class: "attn-heat-wrap" }, labels, el("div", { class: "attn-heat-scroll" }, band, canvas)),
		el("div", { class: "hint", text: "columns = generated tokens → · white = current token · cyan = current head · click a cell to inspect" }),
	);
}

// Per-(token, head) detail: selectors + scene mass/entropy + a TABULATED
// (no bars) list of top attended objects with the top attribute per object.
function tokenDetail(a) {
	const tokRec = a.tokens[state.attnToken];
	// The attended-object table needs this token's per-head detail, which the
	// compact payload omits — fetch it lazily (null while in flight).
	const heads = currentTokenHeads();
	const hstat = heads ? (heads[state.attnHead] || {}) : null;
	const headBar = el("div", { class: "seg-legend" }, el("span", { class: "muted", text: "head:" }),
		...a.selected_heads.map((h, i) => el("button", {
			class: state.attnHead === i ? "on" : "", text: `L${h.layer}H${h.head}`,
			onclick: () => { state.attnHead = i; renderAttention(); },
		})),
	);
	const slider = el("input", {
		type: "range", min: "0", max: String(a.tokens.length - 1), value: String(state.attnToken),
		style: "width:100%", oninput: (ev) => { state.attnToken = Number(ev.target.value); renderAttention(); },
	});
	const isOutput = state.attnToken >= outputStartTok(a);
	const tokLine = el("div", { class: "muted", style: "margin:3px 0" },
		el("span", { class: `frame-tag ${isOutput ? "out" : "reason"}`, text: isOutput ? "output" : "reasoning" }),
		el("span", { text: ` token ${state.attnToken + 1}/${a.tokens.length} · #${tokRec.index} ` }),
		el("code", { text: JSON.stringify(tokRec.text) }),
		tokRec.output_entity ? el("span", { text: ` · emits “${tokRec.output_entity}”` }) : null,
	);
	// scene mass is a scalar carried in the compact payload; entropy comes from
	// the (lazy) detail.
	const scale = tokRec.hscale?.[state.attnHead] ?? 0;
	const ent = hstat ? (hstat.entropy_ratio ?? 0) : null;
	const metrics = el("div", { class: "metric-row" },
		el("span", { title: "share of this token's attention landing on scene tokens" },
			el("span", { class: "m-k", text: "scene mass " }), el("span", { class: "m-v", text: `${(scale * 100).toFixed(1)}%` })),
		el("span", { title: "how spread-out (1) vs peaked (0) the scene attention is" },
			el("span", { class: "m-k", text: "entropy " }), el("span", { class: "m-v", text: ent == null ? "…" : ent.toFixed(3) })),
	);
	let tableBody;
	if (!heads) {
		tableBody = el("div", { class: "muted", text: "loading token detail…" });
	} else {
		const rows = (hstat?.top_entities || []).map((x) => {
			const topComp = Object.entries(x.components || {}).sort((p, q) => q[1] - p[1])[0];
			return el("tr", { class: "clickable", title: "select in 3D", onclick: () => jumpTo(x.id) },
				el("td", {}, el("span", { class: "map-sw", style: `background:${entityHex(x.kind, x.id)}` }), el("span", { text: ` ${x.id}` })),
				el("td", { text: entityKindLabel(x.kind, x.id) }),
				el("td", { class: "num", text: x.score.toFixed(4) }),
				el("td", {}, topComp ? el("span", { class: "chip", style: `background:${compHex(topComp[0])}`, text: `${COMPONENT_ABBR[topComp[0]] ?? topComp[0]} ${topComp[1].toFixed(4)}` }) : el("span", { class: "muted", text: "—" })),
			);
		});
		const table = el("table", { class: "attn-table" },
			el("thead", {}, el("tr", {}, el("th", { text: "object" }), el("th", { text: "kind" }), el("th", { class: "num", text: "score" }), el("th", { text: "top attribute" }))),
			el("tbody", {}, rows),
		);
		tableBody = rows.length ? table : el("div", { class: "muted", text: "no scene attention from this token / head" });
	}
	return el("div", { class: "exp-block" },
		headBar, slider, tokLine, metrics,
		el("div", { class: "exp-lab", text: "top attended objects (click a row → select in 3D)" }),
		tableBody,
	);
}
