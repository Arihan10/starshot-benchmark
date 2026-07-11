// In-drawer tab switching + the shared tab-bar highlight. The analysis workspace
// is the separate /tf-workspace iframe (opened via workspace-modal.js), so the
// old in-page report open/close/sync + header context line no longer live here.

import { $, state } from "./state.js";
import { sceneAttentionCount } from "./reportState.js";
import { renderPlacement } from "./placement.js";
import { renderPipeline } from "./render.js";
import { renderAblationDrawer } from "./ablationdrawer.js";

// In-drawer tabs (compact). "ablation" only appears when the scene is ablated.
const DRAWER_TABS = ["attention", "placement", "export", "obs", "ablation"];

export function switchTab(name) {
	// "summary"/"overview" are the workspace button — intercepted by
	// workspace-modal.js (→ iframe); ignore if one ever reaches here.
	if (name === "summary" || name === "overview") return;
	state.drawerTab = name;
	syncTabHighlight();
	for (const id of DRAWER_TABS) { const n = $(`tf-tab-${id}`); if (n) n.hidden = id !== name; }
	if (name === "placement") renderPlacement();
	if (name === "obs") renderPipeline();
	if (name === "ablation") renderAblationDrawer();
}

// Tab-bar highlight: the active drawer tab, plus the ⚗ ablation tab's visibility.
// The "open workspace" button is enabled only once the scene has a computed map.
export function syncTabHighlight() {
	const ablBtn = document.getElementById("tf-tab-ablation-btn");
	if (ablBtn) ablBtn.hidden = !(state.ablatedSlots && state.ablatedSlots.has(state.slot));
	for (const btn of document.querySelectorAll("#tf-tabs button")) {
		const t = btn.dataset.tab;
		btn.classList.toggle("on", t === state.drawerTab && !btn.classList.contains("tf-tab-report"));
		if (btn.classList.contains("tf-tab-report")) {
			const hasAttention = sceneAttentionCount(state.slot) > 0;
			btn.disabled = !hasAttention;
			btn.title = hasAttention ? "open the analysis workspace" : "compute attention for this scene before opening the workspace";
		}
	}
}
