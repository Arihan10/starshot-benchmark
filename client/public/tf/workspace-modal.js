// Bridges the /tf viewer to the analysis workspace (/tf-workspace). The drawer's
// "open workspace" button pops the data view in a fullscreen modal iframe, seeded
// with the viewer's current run/scene/model. The in-page report has been removed
// from /tf, so this is the only workspace entry point.
//
// We read the live cell straight from the shared state module (ES modules are
// singletons, so this is the same object the viewer mutates).

import { state } from "./modules/state.js";

const $ = (id) => document.getElementById(id);

function cellQuery() {
	const p = new URLSearchParams();
	if (state.run) p.set("run", state.run);
	if (state.slot) p.set("slot", state.slot);
	if (state.model) p.set("model", state.model);
	p.set("embed", "1");
	return p.toString();
}

function openWorkspace() {
	const modal = $("tf-ws-modal"), frame = $("tf-ws-frame");
	if (!modal || !frame) return;
	const qs = cellQuery();
	const url = `/tf-workspace?${qs}`;
	// Reload on every open so it always reflects the current cell + latest build
	// (add a cache-buster so the iframe never serves a stale module graph).
	frame.src = `${url}&t=${Date.now()}`;
	const link = $("tf-ws-open"); if (link) link.href = url;
	const ctx = $("tf-ws-ctx"); if (ctx) ctx.textContent = [state.run, state.slot, state.model].filter(Boolean).join(" · ");
	modal.hidden = false; modal.setAttribute("aria-hidden", "false");
}
function closeWorkspace() {
	const modal = $("tf-ws-modal");
	if (modal) { modal.hidden = true; modal.setAttribute("aria-hidden", "true"); }
}

function wire() {
	// Open the analysis workspace from the drawer's "open workspace" button.
	// switchTab("summary"/"overview") is a harmless no-op now, so a plain listener
	// suffices (no capture-phase interception needed).
	for (const btn of document.querySelectorAll("#tf-tabs button.tf-tab-report")) {
		btn.addEventListener("click", openWorkspace);
		btn.title = "open the analysis workspace";
	}
	const closeBtn = $("tf-ws-close");
	if (closeBtn) closeBtn.onclick = closeWorkspace;
	const modal = $("tf-ws-modal");
	if (modal) modal.addEventListener("click", (e) => { if (e.target === modal) closeWorkspace(); });
	document.addEventListener("keydown", (e) => { if (e.key === "Escape" && modal && !modal.hidden) { e.preventDefault(); closeWorkspace(); } });
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", wire);
else wire();
