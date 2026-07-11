// Per-step rendering + navigation: the merged timeline/scrubber strip, step
// navigation (gotoStep), the 3D scene render, the scene-tree pipeline tab, the
// export fetch, and the lazy per-token attention detail loaders.

import { el } from "../../js/ui.js";
import { api } from "../../js/api.js";
import { applySceneProjection } from "../../js/events.js";
import { $, state, bumpView, lruSet, COMPACT_CACHE_MAX, TOKEN_DETAIL_CACHE_MAX, ATTN_STATUS_LABEL, entityHex } from "./state.js";
import { escapeHtml } from "./util.js";
import { jumpTo, renderExport } from "./exportPanel.js";
import { renderAttention, applyAttnHighlight } from "./attnPanel.js";
import { updateBatchProgress } from "./attnQueue.js";

// --- timeline (step nav + per-step compute status, merged) -------------------

// One cell per step, colored by attention status (grey none · blue queued ·
// amber running · green ready · red error); click to jump. This IS the scrubber
// and the compute-status strip in one — the whole cell's progress at a glance.
function timelineCells() {
	return state.steps.map((step, i) => {
		const ev = step.event_index;
		const st = state.attnStatus[ev] || "none";
		const noScene = step.has_scene === false;
		const err = st === "error" && state.attnErrors?.[ev] ? ` — ${state.attnErrors[ev]}` : "";
		return el("div", {
			class: `cell st-${st}${noScene ? " no-scene" : ""}${i === state.stepIdx ? " cur" : ""}`,
			title: `step ${i + 1}/${state.steps.length} · ${step.template ?? step.step ?? "?"}${step.node ? ` on ${step.node}` : ""} · ${ATTN_STATUS_LABEL[st]}${noScene ? " · no scene context (skipped)" : ""}${err}`,
			onclick: () => gotoStep(i),
		});
	});
}

// Keep the current step visible by scrolling ONLY the strip horizontally, and
// only when the cell is near/past an edge. We deliberately avoid
// Element.scrollIntoView(): it re-centers on every step change (a jarring "snap"
// once the history is long enough to scroll) AND it scrolls vertical ancestors,
// which yanks the whole panel when the content below is tall.
function centerTimelineCell(strip, idx) {
	const cur = strip.children[idx];
	if (!cur) return;
	const s = strip.getBoundingClientRect();
	const c = cur.getBoundingClientRect();
	const margin = Math.max(c.width * 3, 24); // keep a few cells of lead-in visible
	if (c.left < s.left + margin) strip.scrollLeft -= (s.left + margin) - c.left;
	else if (c.right > s.right - margin) strip.scrollLeft += c.right - (s.right - margin);
}

// Full timeline repaint: cells + nav buttons + step label, and keep the
// current step in view. Called on navigation.
export function renderTimeline() {
	const strip = $("tf-timeline");
	if (strip) {
		strip.replaceChildren(...timelineCells());
		centerTimelineCell(strip, state.stepIdx);
	}
	$("tf-prev").disabled = state.stepIdx === 0;
	$("tf-next").disabled = state.stepIdx >= state.steps.length - 1;
	const step = state.steps[state.stepIdx];
	const line = $("tf-step-line");
	if (step) {
		const tmpl = step.template ?? step.step ?? "?";
		const node = step.node ?? "?";
		const lp = step.has_logprobs ? " · logprobs ✓" : "";
		line.innerHTML = `<b>${state.stepIdx + 1}</b>/${state.steps.length} · <b>${escapeHtml(tmpl)}</b> · ${escapeHtml(node)}${lp}`;
		line.title = `step ${state.stepIdx + 1}/${state.steps.length} · ${tmpl} · ${node}${step.has_logprobs ? " · logprobs" : ""}`;
	} else {
		line.textContent = "—";
		line.title = "";
	}
	updateBatchProgress();
}

// Cheap in-place repaint of just the cells + top-bar progress (poll tick) —
// doesn't move scroll or focus.
export function refreshTimeline() {
	const strip = $("tf-timeline");
	if (strip) strip.replaceChildren(...timelineCells());
	updateBatchProgress();
	// Keep the scene-tree chips' status colors in sync while the queue drains,
	// but only when that tab is actually visible (avoids rebuilding it hidden).
	if (!$("tf-tab-obs")?.hidden) renderPipeline();
}

// --- per-step render ---------------------------------------------------------

export function gotoStep(k) {
	if (!state.steps.length) return;
	k = Math.max(0, Math.min(state.steps.length - 1, k));
	state.stepIdx = k;
	state.attn = null;    // attention is per-step; drop the previous step's analysis
	state.tokenDetail = null; // and its lazily-loaded per-token detail
	state.export = null;  // and its export, so panels don't paint stale data mid-load
	state.segSel = null;  // clear the per-step segment breakdown drill-down
	const token = bumpView();
	const step = state.steps[k];
	state.reportSelections.stepIdx = k;
	renderTimeline();
	renderScene(step);
	renderPipeline();
	renderExportForStep(step, token);
}

export function renderScene(step) {
	const v = state.viewer;
	if (!v) return;
	const opts = step.render_until != null ? { untilIndex: step.render_until } : {};
	v.clear();
	api.scene(state.run, state.slot, state.model, opts)
		.then((proj) => {
			applySceneProjection(v, proj);
			v.prefetchBundle(api.meshesUrl(state.run, state.slot, state.model, opts));
			applyAttnHighlight(); // (re)paint attention highlight once the entities exist
		})
		.catch(() => { /* non-fatal — leave the viewer empty for this step */ });
}

// Pipeline tab: the cell's SCENE TREE (zones ▸ subregions ▸ objects) with each
// node's pipeline steps as clickable chips — pick the steps for a specific zone,
// like the main viewer's tree. Clicking a node focuses it in 3D; clicking a step
// opens it (jumps the timeline). Chip color = attention-compute status. Rebuilt
// on navigation so the current step/node stays highlighted.
export function renderPipeline() {
	const host = $("tf-obs-wrap");
	if (!host) return;
	const model = state.obs;
	if (!model || !state.steps.length) { host.replaceChildren(el("div", { class: "obsm-empty", text: "no pipeline yet" })); return; }
	// Steps grouped by the scene node they operate on (step.node).
	const byNode = new Map();
	state.steps.forEach((s, idx) => {
		const k = s.node ?? "?";
		if (!byNode.has(k)) byNode.set(k, []);
		byNode.get(k).push(idx);
	});
	const curEv = state.steps[state.stepIdx]?.event_index;

	const stepChip = (idx) => {
		const s = state.steps[idx];
		const stt = state.attnStatus[s.event_index] || "none";
		return el("button", {
			class: `pl-step st-${stt}${s.event_index === curEv ? " cur" : ""}${s.has_scene === false ? " no-scene" : ""}`,
			title: `step ${idx + 1}/${state.steps.length} · ${s.template ?? s.step ?? "?"} · ${ATTN_STATUS_LABEL[stt]}${s.has_scene === false ? " · no scene context" : ""}`,
			text: s.template ?? s.step ?? "?",
			onclick: () => gotoStep(idx),
		});
	};

	const seen = new Set();
	const nodeBlock = (id, depth) => {
		if (seen.has(id)) return null;
		seen.add(id);
		const n = model.nodes.get(id);
		const kids = model.order.filter((c) => model.nodes.get(c)?.parentId === id);
		const steps = byNode.get(id) || [];
		const hasCur = steps.some((i) => state.steps[i].event_index === curEv);
		const row = el("div", { class: "pl-node", style: `margin-left:${Math.min(depth, 6) * 11}px` },
			el("div", { class: `pl-node-row${hasCur ? " cur" : ""}`, title: `focus ${id} in 3D`, onclick: () => jumpTo(id) },
				el("span", { class: "map-sw", style: `background:${entityHex(n?.kind, id)}` }),
				el("span", { class: "obsm-id", text: id }),
				steps.length ? el("span", { class: "pl-count", text: String(steps.length) }) : null,
			),
			steps.length ? el("div", { class: "pl-steps" }, ...steps.map(stepChip)) : null,
		);
		const childBlocks = kids.map((c) => nodeBlock(c, depth + 1)).filter(Boolean);
		return childBlocks.length ? el("div", {}, row, ...childBlocks) : row;
	};

	const roots = model.order.filter((id) => { const p = model.nodes.get(id)?.parentId; return !p || !model.nodes.has(p); });
	const blocks = roots.map((r) => nodeBlock(r, 0)).filter(Boolean);
	// Steps whose node isn't a scene entity (root plans / cell-level calls).
	const orphan = [...byNode.entries()].filter(([k]) => !model.nodes.has(k)).flatMap(([, v]) => v);
	const orphanBlock = orphan.length
		? el("div", { class: "pl-node" },
			el("div", { class: "pl-node-row", style: "opacity:.75" }, el("span", { class: "obsm-id", text: "· cell-level steps" })),
			el("div", { class: "pl-steps" }, ...orphan.map(stepChip)))
		: null;
	$("tf-obs-count").textContent = `${state.steps.length} steps · ${model.nodes.size} nodes`;
	host.replaceChildren(
		el("div", { class: "muted", style: "font-size:10px;margin-bottom:6px", text: "scene tree — click a zone/object to focus it in 3D, a step to open it. chip color = attention status." }),
		...blocks,
		orphanBlock,
	);
}

async function renderExportForStep(step, token) {
	$("tf-export-wrap").replaceChildren(el("div", { class: "empty-note", text: "reconstructing…" }));
	renderAttention(); // export is null → shows the step strip + "loading step…"
	let exp;
	try {
		exp = await api.tfExport(state.run, state.slot, state.model, step.event_index);
	} catch (e) {
		if (token !== state.viewToken) return; // navigated away
		$("tf-export-wrap").replaceChildren(el("div", { class: "empty-note", text: `export failed: ${e.message}` }));
		renderAttention();
		return;
	}
	if (token !== state.viewToken) return; // a newer step/cell is showing
	state.export = exp;
	try { renderExport(); } catch (e) { $("tf-export-wrap").replaceChildren(el("div", { class: "empty-note", text: `render failed: ${e.message}` })); }
	// If this step has a stored result (fresh OR stale), load it for display;
	// otherwise renderAttention shows the compute affordance. Either way the panel
	// resolves deterministically — never stuck on a generic "loading".
	const st = state.attnStatus[step.event_index];
	if (st === "ready" || st === "stale") loadAttention(step.event_index, token);
	else renderAttention();
}

// Fetch a stored analysis for display (distinct from computing it). Token-guarded
// so a slow response for an old step never paints the current view. Leaves the
// per-step STATUS to the server-authoritative poll (ready vs stale) — we only
// load the data here.
export function loadAttention(eventIndex, token) {
	// Serve from the cross-step cache instantly (no refetch) UNLESS a force
	// recompute is pending, in which case we must pull the refreshed copy.
	// compactCache is keyed by event_index and cleared per cell (see selectCell).
	const cached = state.compactCache.get(eventIndex);
	if (cached && !state.attnPendingReload.has(eventIndex)) {
		state.attn = cached; state.attnHead = 0; state.attnToken = 0; state.tokenDetail = null;
		renderAttention();
		if (state.export) renderExport();
		return;
	}
	renderAttention(); // shows "loading result…" while the fetch is in flight
	api.attentionGet(state.run, state.slot, state.model, eventIndex, { view: "compact" })
		.then((a) => {
			if (token !== state.viewToken) return;
			state.attnPendingReload.delete(eventIndex); // refreshed copy now shown
			lruSet(state.compactCache, eventIndex, a, COMPACT_CACHE_MAX);
			state.attn = a; state.attnHead = 0; state.attnToken = 0; state.tokenDetail = null;
			renderAttention();
			if (state.export) renderExport(); // refresh the reconstruction tab's validity block
		})
		.catch(() => {
			if (token !== state.viewToken) return;
			state.attnStatus[eventIndex] = "none"; // stored file vanished / unreadable
			renderAttention();
		});
}

// Lazily fetch (and cache) one generated token's full per-head detail — the top
// entities/attributes the token-detail table + 3D highlight need, which the
// compact payload omits. Re-renders when it arrives. Returns the cached heads[]
// immediately when available, else null (caller shows a lightweight placeholder).
function ensureTokenDetail(ev, i) {
	const key = `${ev}:${i}`;
	const hit = state.tokenDetailCache.get(key);
	if (hit) return hit;
	if (state.tokenDetailPending.has(key)) return null;
	state.tokenDetailPending.add(key);
	const token = state.viewToken;
	api.attentionGet(state.run, state.slot, state.model, ev, { view: "token", i })
		.then((rec) => {
			state.tokenDetailPending.delete(key);
			const heads = (rec && rec.heads) || [];
			lruSet(state.tokenDetailCache, key, heads, TOKEN_DETAIL_CACHE_MAX);
			if (token !== state.viewToken) return; // navigated away
			// Only repaint if this is still the token being viewed.
			const step = state.steps[state.stepIdx];
			if (step && step.event_index === ev && state.attnToken === i) {
				state.tokenDetail = { ev, i, heads };
				renderAttention();
			}
		})
		.catch(() => { state.tokenDetailPending.delete(key); });
	return null;
}

// The current (step, token) heads[] detail if loaded, else kick off a fetch and
// return null. Keeps state.tokenDetail in sync as a fast path.
export function currentTokenHeads() {
	const a = state.attn;
	if (!a || !a.meta) return null;
	const ev = a.meta.event_index, i = state.attnToken;
	if (state.tokenDetail && state.tokenDetail.ev === ev && state.tokenDetail.i === i) return state.tokenDetail.heads;
	const cached = state.tokenDetailCache.get(`${ev}:${i}`);
	if (cached) { state.tokenDetail = { ev, i, heads: cached }; return cached; }
	return ensureTokenDetail(ev, i);
}
