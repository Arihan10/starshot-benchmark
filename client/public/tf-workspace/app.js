// Entry for the (new) /tf data view. Boots the header/nav (run · scene · model ·
// region · step · data/ablation), resolves the region/step selection into a set
// of computed attention steps, and renders the four data cards. Reuses only the
// shared data layer (api.js, events.js) — no 3D, no legacy state machinery.

import { el } from "../js/ui.js";
import { $, state, ALL, bumpLoad } from "./modules/state.js";
import {
	loadRuns, loadCell, loadSteps, computedSteps, regionOptions, stepOptions, loadRows, selectedSteps,
} from "./modules/data.js";
import { spiderCard, compositionCard, structureCard, rhoFocusCard, outputCard, tagsCard, viiCard } from "./modules/cards.js";
import { renderAblation } from "./modules/ablation.js";
import { renderContent, closeContentWindows } from "./modules/content.js";
import { openPromptView } from "./modules/promptview.js";

// Open-weight models are the only ones with attention; prefer them when picking a
// default cell so the page opens on data.
const OPEN_HINT = ["gemma", "qwen-122b", "qwen"];

const opt = (value, label, selected) => el("option", { value, text: label, ...(selected ? { selected: "" } : {}) });
function fill(sel, options, current) {
	sel.replaceChildren(...options.map((o) => opt(o.value, o.label, String(o.value) === String(current))));
}
function showLoading(on) { $("dv-loading").classList.toggle("on", on); }

function saveScene() {
	try { if (state.run && state.slot && state.model) localStorage.setItem("tf-new-scene", JSON.stringify({ run: state.run, slot: state.slot, model: state.model })); } catch { /* ignore */ }
}
function loadSavedScene() {
	try { return JSON.parse(localStorage.getItem("tf-new-scene") || "null"); } catch { return null; }
}

// --- nav population ----------------------------------------------------------

function populateRegionStep() {
	const regions = regionOptions();
	const regionOpts = [{ value: ALL, label: `ALL regions (${regions.length})` }, ...regions.map((r) => ({ value: r.id, label: `${r.id} · ${r.count}` }))];
	fill($("sel-region"), regionOpts, state.region);
	populateStep();
}
function populateStep() {
	const steps = stepOptions();
	const label = state.region === ALL ? `ALL steps (${steps.length} kinds)` : `ALL steps (${steps.length})`;
	const stepOpts = [{ value: ALL, label }, ...steps.map((s) => ({ value: s.value, label: `${s.label}` }))];
	fill($("sel-step"), stepOpts, state.step);
	syncPromptBtn();
}

// Render the active view. Region/step/cell changes route through here.
function rerender() {
	if (state.view === "ablation") renderAblation();
	else if (state.view === "content") renderContent();
	else render();
}

// The "prompts" button is live only when the selection resolves to exactly one
// step (so there's a single call to show).
function syncPromptBtn() {
	const btn = $("sel-prompts");
	if (!btn) return;
	const sel = selectedSteps();
	if (sel.length === 1) { btn.disabled = false; btn._step = sel[0]; }
	else { btn.disabled = true; btn._step = null; }
}

// `nShown`/`nTotal` describe the loaded selection: when the last-N window is
// clipping (`nShown < nTotal`) the chip reads "last n of N steps"; otherwise the
// full loaded count. `nShown == null` (pre-load) shows the computed-step count.
function scopeLine(nShown, nTotal = null) {
	const parts = [];
	if (state.run) parts.push(`<b>${state.run}</b>`);
	if (state.slot) parts.push(`<b>${state.slot}</b>`);
	if (state.model) parts.push(`<b>${state.model}</b>`);
	const head = parts.join(" · ");
	const regionTxt = state.region === ALL ? "ALL regions" : state.region;
	const stepTxt = state.step === ALL ? (state.region === ALL ? "ALL steps (whole scene)" : "ALL steps") : state.step;
	const nComputed = computedSteps().length;
	const chip = nShown == null ? `${nComputed} computed steps`
		: (nTotal != null && nShown < nTotal) ? `last ${nShown} of ${nTotal} steps`
			: `${nShown} step${nShown === 1 ? "" : "s"} loaded`;
	$("scope-line").innerHTML =
		`${head} &nbsp;—&nbsp; region <b>${regionTxt}</b> · step <b>${stepTxt}</b>` +
		` <span class="chip${nComputed ? "" : " warn"}">${chip}</span>`;
}

// --- last-N step window (data view) -----------------------------------------
// Clamp the persisted window (state.lastN; 0 = "all") to the current selection's
// step count. Applies to any multi-step selection — the whole scene (ALL/ALL), a
// step kind (ALL region + a template), or a region.
function windowCount(total) {
	const raw = state.lastN || 0;
	return raw <= 0 ? total : Math.min(Math.max(1, raw), total);
}
// Short description of what the selection spans, shown beside the window slider.
function windowScopeHint() {
	if (state.region === ALL && state.step === ALL) return "whole scene";
	if (state.region === ALL) return state.step;   // a step KIND across the scene
	if (state.step === ALL) return state.region;   // a scene region
	return null;
}
// A persistent slider limiting the graphs to the last N steps of the selection.
// It is built ONCE per selection and never recreated while dragging (only its
// label + the cards below update), so a drag isn't dropped. Null for ≤1 step.
function buildWindowBar(total, onChange) {
	if (total <= 1) return null;
	const label = el("span", { class: "dv-win-lab" });
	const setLabel = (nn) => { label.textContent = nn >= total ? `all ${total} steps` : `last ${nn} of ${total} steps`; };
	setLabel(windowCount(total));
	let raf = 0;
	const slider = el("input", {
		type: "range", min: "1", max: String(total), step: "1", value: String(windowCount(total)),
		class: "dv-win-slider", title: "limit the graphs to the last N steps of this selection",
		oninput: (ev) => {
			const v = Number(ev.target.value);
			state.lastN = v >= total ? 0 : v; // dragging to the max is a sticky "all"
			try { localStorage.setItem("tf-lastn", String(state.lastN)); } catch { /* ignore */ }
			setLabel(v);
			if (raf) return; // coalesce to one card re-render per frame while dragging
			raf = requestAnimationFrame(() => { raf = 0; onChange(); });
		},
	});
	const scope = windowScopeHint();
	return el("div", { class: "dv-window" },
		el("span", { class: "dv-win-cap", text: "window" }),
		slider, label,
		scope ? el("span", { class: "dv-win-scope", text: scope }) : null);
}

// --- render ------------------------------------------------------------------

// Render the card-based "data" view for the current selection into `innerId`.
// `view` is the state.view this render belongs to, so a stale async load for a
// view you've since switched away from is dropped.
async function renderCards(view, innerId) {
	if (state.view !== view) return;
	const token = bumpLoad();
	state.pieFilter = null;
	scopeLine(null);
	const inner = $(innerId);
	if (!computedSteps().length) {
		inner.replaceChildren(el("div", { class: "empty" },
			el("span", { class: "big", text: "⚡" }),
			el("div", { text: "no computed attention for this cell." }),
			el("div", { class: "faint", style: "margin-top:6px", text: "pick another run / scene / model, or compute attention for this cell in the legacy inspector." }),
			el("div", { style: "margin-top:10px" }, el("a", { class: "pill", href: "/tf-legacy", text: "open legacy ↗" }))));
		return;
	}
	showLoading(true);
	inner.replaceChildren(el("div", { class: "empty", text: "loading attention…" }));
	let rows;
	try { rows = await loadRows(token); } catch (e) { if (token === state.loadToken) inner.replaceChildren(el("div", { class: "empty", text: `failed: ${e.message}` })); showLoading(false); return; }
	if (token !== state.loadToken) return;
	showLoading(false);
	state.rows = rows; // full loaded set (unwindowed); the window is applied per-render below
	if (!rows.length) { scopeLine(0, null); inner.replaceChildren(el("div", { class: "empty", text: "no attention data for this selection" })); return; }
	// Cards live in their own host so the last-N window slider re-renders them
	// (client-side, no refetch) without recreating the slider itself mid-drag.
	const cardsHost = el("div", { class: "dv-cards" });
	function paintCards() {
		if (token !== state.loadToken) return; // a newer selection superseded this one
		const total = state.rows.length;
		const n = windowCount(total);
		const shown = n >= total ? state.rows : state.rows.slice(-n); // rows are pre-sorted by event_index
		scopeLine(shown.length, total);
		cardsHost.replaceChildren(
			el("div", { class: "dv-row" }, spiderCard(shown), compositionCard(shown)),
			structureCard(shown),
			rhoFocusCard(shown),
			outputCard(shown),
			tagsCard(shown),
			viiCard(shown),
		);
	}
	inner.replaceChildren(...[buildWindowBar(rows.length, paintCards), cardsHost].filter(Boolean));
	paintCards();
}

function render() { return renderCards("data", "dv-inner"); }
// renderContent (the per-step "structure" page: 3D · tree · plan/output/reasoning)
// lives in ./modules/content.js.

// --- selection flow ----------------------------------------------------------

async function selectCell(slot, model) {
	state.slot = slot; state.model = model;
	state.region = ALL; state.step = ALL;
	state.aggCache.clear();
	closeContentWindows(); // drop the previous cell's detached plan/output/reasoning windows
	// drop the previous cell's ablation caches so the ablation view reloads fresh
	state._ablCell = null; state.variants = []; state.variantRows = new Map(); state.ablRun = null;
	saveScene();
	$("scope-line").textContent = "loading cell…";
	try {
		await loadSteps(state.run, slot, model);
	} catch (e) {
		$("scope-line").textContent = `failed to load cell: ${e.message}`;
		return;
	}
	populateRegionStep();
	rerender();
}

async function selectRun(run, prefer = null) {
	state.run = run;
	let cell;
	try { cell = await loadCell(run); } catch (e) { $("scope-line").textContent = `failed to load run: ${e.message}`; return; }
	const { slots, models } = cell;
	fill($("sel-model"), models.map((m) => ({ value: m, label: m })), null);
	fill($("sel-slot"), slots.map((s) => ({ value: s.id, label: s.id })), null);
	// Pick a cell: prefer the saved/explicit one, else the first slot × an open
	// model that has logged events (so attention can exist).
	let pick = null;
	if (prefer && slots.some((s) => s.id === prefer.slot) && models.includes(prefer.model)) pick = { slot: prefer.slot, model: prefer.model };
	const openModels = models.filter((m) => OPEN_HINT.some((h) => m.includes(h)));
	const modelPref = [...openModels, ...models];
	for (const s of slots) {
		if (pick) break;
		for (const m of modelPref) if ((s.runs?.[m]?.events_count ?? 0) > 0) { pick = { slot: s.id, model: m }; break; }
	}
	if (!pick && slots.length && models.length) pick = { slot: slots[0].id, model: modelPref[0] || models[0] };
	if (pick) {
		$("sel-slot").value = pick.slot;
		$("sel-model").value = pick.model;
		await selectCell(pick.slot, pick.model);
	}
}

function setView(v) {
	state.view = v;
	for (const b of document.querySelectorAll("#view-toggle button")) b.classList.toggle("on", b.dataset.view === v);
	$("data-view").classList.toggle("off", v !== "data");
	$("ablation-view").classList.toggle("on", v === "ablation");
	$("content-view").classList.toggle("on", v === "content");
	if (v !== "content") closeContentWindows(); // leaving content → dismiss its 3D + text popouts
	rerender();
}

// --- boot --------------------------------------------------------------------

async function boot() {
	$("sel-run").onchange = () => selectRun($("sel-run").value);
	$("sel-slot").onchange = () => selectCell($("sel-slot").value, $("sel-model").value);
	$("sel-model").onchange = () => selectCell($("sel-slot").value, $("sel-model").value);
	$("sel-region").onchange = () => { state.region = $("sel-region").value; state.step = ALL; populateStep(); rerender(); };
	$("sel-step").onchange = () => { state.step = $("sel-step").value; syncPromptBtn(); rerender(); };
	for (const b of document.querySelectorAll("#view-toggle button")) b.onclick = () => setView(b.dataset.view);
	$("sel-prompts").onclick = () => { const s = $("sel-prompts")._step; if (s) openPromptView(s); };

	let runs;
	try { runs = await loadRuns(); } catch (e) { $("scope-line").textContent = `failed to load runs: ${e.message}`; return; }
	state.runs = runs;
	// Deep link from the /tf viewer's "open workspace" modal: ?run=&slot=&model=&embed=1
	const params = new URLSearchParams(location.search);
	if (params.get("embed") === "1") document.body.classList.add("embedded");
	const pRun = params.get("run");
	const link = pRun && runs.includes(pRun) ? { run: pRun, slot: params.get("slot"), model: params.get("model") } : null;
	const saved = loadSavedScene();
	const prefer = link || (saved && runs.includes(saved.run) ? saved : null);
	const current = prefer ? prefer.run : runs[0];
	fill($("sel-run"), runs.map((r) => ({ value: r, label: r })), current);
	if (current) await selectRun(current, prefer && prefer.run === current ? prefer : null);
}

boot();
