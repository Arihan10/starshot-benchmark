// In-drawer tab switching + the fullscreen analysis-report open/close/sync, plus
// the shared tab-bar highlight and the report header context line. These are the
// cross-cutting UI-sync helpers several feature modules call, kept in their own
// module so nothing has to import the side-effectful entry.

import { $, state, focusedKind } from "./state.js";
import * as report from "./report.js";
import { renderPlacement } from "./placement.js";
import { renderPipeline } from "./render.js";
import { renderAblationDrawer } from "./ablationdrawer.js";

// In-drawer tabs (compact). Summary + overview are NOT here — they open the big
// report window instead (see openReport). "ablation" only appears when the scene
// is ablated.
const DRAWER_TABS = ["attention", "placement", "export", "obs", "ablation"];

export function switchTab(name) {
	// The two analytics views live in the fullscreen report, not the drawer.
	if (name === "summary" || name === "overview") { openReport(name); return; }
	state.drawerTab = name;
	syncTabHighlight();
	for (const id of DRAWER_TABS) { const n = $(`tf-tab-${id}`); if (n) n.hidden = id !== name; }
	if (name === "placement") renderPlacement();
	if (name === "obs") renderPipeline();
	if (name === "ablation") renderAblationDrawer();
}

// Tab-bar highlight: the active drawer tab, plus (while open) the report tab.
export function syncTabHighlight() {
	// The ⚗ ablation drawer tab only shows for scenes that have ablation variants.
	const ablBtn = document.getElementById("tf-tab-ablation-btn");
	if (ablBtn) ablBtn.hidden = !(state.ablatedSlots && state.ablatedSlots.has(state.slot));
	for (const btn of document.querySelectorAll("#tf-tabs button")) {
		const t = btn.dataset.tab;
		btn.classList.toggle("on", t === state.drawerTab && !btn.classList.contains("tf-tab-report"));
		if (btn.classList.contains("tf-tab-report")) {
			const hasAttention = report.sceneAttentionCount(state.slot) > 0;
			btn.classList.toggle("active", report.normalizeReportView(t) === state.reportView);
			btn.disabled = !hasAttention;
			btn.title = hasAttention ? "open the analysis workspace" : "compute attention for this scene before opening the workspace";
		}
	}
}

// Open the large analysis window on a workspace mode. Legacy drawer actions
// still pass "summary"/"overview"; normalize them into "step"/"scene".
export function openReport(view) {
	if (report.sceneAttentionCount(state.slot) <= 0) {
		$("tf-attn-progress").textContent = "compute attention for this scene before opening the workspace";
		return;
	}
	state.reportView = report.normalizeReportView(view || state.lastReportMode);
	state.lastReportMode = state.reportView;
	$("tf-report").hidden = false;
	$("tf-report").setAttribute("aria-hidden", "false");
	syncReportView();
	syncTabHighlight();
}

export function closeReport() {
	if (!state.reportView) return;
	state.lastReportMode = state.reportView;
	state.reportView = null;
	$("tf-report").hidden = true;
	$("tf-report").setAttribute("aria-hidden", "true");
	report.saveReportState();
	syncTabHighlight();
}

// Show the chosen view inside the report + (re)render it.
export function syncReportView() {
	const view = state.reportView;
	for (const btn of document.querySelectorAll("#tf-report-tabs button")) btn.classList.toggle("on", report.normalizeReportView(btn.dataset.rv) === view);
	updateReportCtx();
	report.renderReportWorkspace();
}

// The report header context line, rebuilt (redesign canvas) as a location
// breadcrumb — run › model › scene › {scope tail} — plus a freshness readout
// (how many of the cell's steps are computed). The header's overflow ellipsizes
// a long breadcrumb; the full parts still read left-to-right.
export function updateReportCtx() {
	const host = $("tf-report-ctx");
	if (!host) return;
	const crumbs = [state.run, state.model, state.slot ? `scene ${state.slot}` : null].filter(Boolean);
	let tail = "";
	if (state.reportView === "step") {
		if ((state.pins.steps || []).length >= 2) tail = `${state.pins.steps.length} selected steps`;
		else {
			const step = state.steps[state.stepIdx];
			tail = step ? `${step.template ?? step.step ?? "?"} · step ${state.stepIdx + 1}/${state.steps.length}` : "";
		}
	} else if (state.reportView === "kind") {
		if ((state.pins.kinds || []).length >= 2) tail = `${state.pins.kinds.length} selected kinds`;
		else { const k = focusedKind(); tail = k ? `kind ${k}` : "step kind"; }
	} else if (state.reportView === "scene") {
		if ((state.pins.scenes || []).length >= 2) tail = `${state.pins.scenes.length} selected scenes`;
		else tail = `${report.pinCount()} pins · ${state.graphModules.length} graphs`;
	} else if (state.reportView === "ablation") {
		const nRuns = (state.pins.runs || []).length;
		tail = nRuns ? `${nRuns} selected runs` : "pick variant runs to compare";
	}
	const computed = state.steps.filter((s) => ["ready", "stale"].includes(state.attnStatus[s.event_index])).length;
	const freshness = state.steps.length ? `${computed}/${state.steps.length} computed` : "";
	host.textContent = [crumbs.join(" › "), tail, freshness].filter(Boolean).join("  ·  ");
}
