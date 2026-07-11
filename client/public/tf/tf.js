// Teacher-forcing export inspector. A separate page (served at /tf) that walks
// a cell's LLM steps on a LINEAR time scale: a scrubber selects a step, the
// center reuses the shared scene3d viewer to render the 3D scene AT that step,
// and the right drawer shows the observability tree plus the step's
// teacher-forcing export (the Gemma-faithful reconstructed sequence + the
// id→char-span maps for scene zones/objects, to-place, output, and variables).
//
// It reuses the same renderers as the dashboard (scene3d, events, obsmini) so
// the 3D + obs views stay identical to the main app.
//
// This file is the thin ENTRY: it boots the viewer and wires the run/cell pickers +
// keyboard. Every feature lives in a focused module (state/render/attnPanel/
// attnQueue/…), all sharing the mutable `state` in state.js. The analysis
// workspace is the separate /tf-workspace iframe (opened via workspace-modal.js) —
// the old in-page report (report/reportGraphs/overview/summary) is not loaded here.

import { api } from "../js/api.js";
import { createViewer } from "../js/scene3d.js";
import { el } from "../js/ui.js";

import { state, $, bumpView, obsFromTree } from "./modules/state.js";
import { setSceneAttentionCount } from "./modules/reportState.js";
import { renderAttention } from "./modules/attnPanel.js";
import { gotoStep } from "./modules/render.js";
import {
	computeAllAttention, computeViiSample, syncFromModal, syncAttnStatus, applyServerStatus,
	renderAttnModelBadge, hasPending, startAttnPoll, attnQueueSnapshot,
} from "./modules/attnQueue.js";
import { setAttnProvider } from "../js/attnbus.js";
import { ablQueueSnapshot, syncFromBackend } from "./modules/ablationdrawer.js";
import { switchTab, syncTabHighlight } from "./modules/tabs.js";
import { wirePresentMenu } from "./modules/presentCtx.js";

// --- boot --------------------------------------------------------------------

// Create the 3D viewer, retrying if the browser is momentarily out of WebGL
// contexts (rapid reloads / many tabs — Chrome reclaims old contexts a beat late).
// Gives up with a clear, actionable message instead of a dead page + uncaught error.
async function createViewerResilient(host, opts, tries = 6) {
	for (let i = 0; ; i++) {
		try { return createViewer(host, opts); }
		catch (err) {
			if (i >= tries) throw err;
			await new Promise((r) => setTimeout(r, 250 * (i + 1)));
		}
	}
}

function showViewerError() {
	const host = $("tf-canvas-host");
	if (host) {
		host.innerHTML = '<div style="position:absolute;inset:0;display:flex;flex-direction:column;gap:10px;align-items:center;justify-content:center;text-align:center;color:#9fb0c8;font-size:13px;padding:24px;">'
			+ "<div>Couldn't get a WebGL context — the browser may be out of GPU contexts (too many tabs, or a rapid reload).</div>"
			+ '<button id="tf-webgl-retry" style="padding:7px 14px;border-radius:7px;border:1px solid #2a3550;background:#22304f;color:#dce6f5;cursor:pointer;">Reload</button></div>';
		const btn = document.getElementById("tf-webgl-retry");
		if (btn) btn.onclick = () => location.reload();
	}
	const prog = $("tf-attn-progress");
	if (prog) prog.textContent = "3D viewer unavailable: WebGL context could not be created";
}

// Persist the last-viewed scene (run/slot/model) so a reload reopens it instead
// of the server's default cell.
function saveScene() {
	try {
		if (state.run && state.slot && state.model)
			localStorage.setItem("tf-scene", JSON.stringify({ run: state.run, slot: state.slot, model: state.model }));
	} catch { /* private mode / quota */ }
}
function loadScene() {
	try { return JSON.parse(localStorage.getItem("tf-scene") || "null"); } catch { return null; }
}

async function boot() {
	try {
		state.viewer = await createViewerResilient($("tf-canvas-host"), { keyboard: true, lighting: true });
	} catch (err) {
		console.error("[tf] viewer init failed:", err);
		showViewerError();
		return;
	}
	state.viewer.setActive(true);

	$("tf-run").onchange = () => selectRun($("tf-run").value);
	$("tf-slot").onchange = () => selectCell($("tf-slot").value, $("tf-model").value);
	$("tf-model").onchange = () => selectCell($("tf-slot").value, $("tf-model").value);
	$("tf-prev").onclick = () => gotoStep(state.stepIdx - 1);
	$("tf-next").onclick = () => gotoStep(state.stepIdx + 1);
	for (const btn of document.querySelectorAll("#tf-tabs button")) {
		btn.onclick = () => switchTab(btn.dataset.tab);
	}
	$("tf-attn-all").onclick = computeAllAttention;
	$("tf-attn-vii").onclick = computeViiSample;
	const viiN = $("tf-attn-vii-n");
	if (viiN) {
		viiN.value = String(state.viiSampleN);
		viiN.onchange = () => { state.viiSampleN = Number(viiN.value) || 2; try { localStorage.setItem("tf-vii-n", String(state.viiSampleN)); } catch { /* ignore */ } };
	}
	// The Modal attention queue is now synced from the GLOBAL ops bar (bottom-left),
	// which shows an "attention" segment for this cell. Publish a provider it reads:
	// snapshot = the main-sequence + ablation queues combined; sync = re-pull both.
	setAttnProvider({
		snapshot: () => {
			const m = attnQueueSnapshot();
			const a = ablQueueSnapshot();
			return {
				main: m, abl: a,
				running: m.running + a.running,
				queued: m.queued + a.queued,
				computed: m.computed + a.computed,
				total: m.total + a.total,
				cell: `${state.slot ?? "?"} / ${state.model ?? "?"}`,
			};
		},
		sync: () => Promise.allSettled([syncFromModal(), syncFromBackend()]),
	});
	wirePresentMenu();
	const headsSel = $("tf-attn-heads");
	if (headsSel) {
		headsSel.value = String(state.maxHeads);
		headsSel.onchange = () => {
			state.maxHeads = Number(headsSel.value) || 4;
			syncAttnStatus(); // reclassify stored results vs the new head budget
		};
	}
	// Arrow keys: ↑/↓ step through the pipeline, ←/→ step through query tokens
	// (the attention step). Ignore while a form control is focused so its own
	// arrow behavior (e.g. the token slider) still works.
	window.addEventListener("keydown", (e) => {
		const t = e.target.tagName;
		if (t === "SELECT" || t === "INPUT" || t === "TEXTAREA") return;
		if (e.key === "ArrowUp") { e.preventDefault(); gotoStep(state.stepIdx - 1); }
		else if (e.key === "ArrowDown") { e.preventDefault(); gotoStep(state.stepIdx + 1); }
		else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
			if (state.attn?.tokens?.length) {
				e.preventDefault();
				const n = state.attn.tokens.length;
				state.attnToken = Math.max(0, Math.min(n - 1, state.attnToken + (e.key === "ArrowRight" ? 1 : -1)));
				renderAttention();
			}
		}
	});

	try {
		const data = await api.runs();
		const runs = Array.isArray(data) ? data : (data.runs ?? []);
		const names = runs.map((r) => (typeof r === "string" ? r : r.name));
		const saved = loadScene();
		// Deep link from the ablation board: ?run=&slot=&model=&view=ablation&pins=a,b,c
		// lands on the base cell in the requested view with the family pre-pinned.
		const params = new URLSearchParams(location.search);
		const pRun = params.get("run");
		const deepLink = pRun && names.includes(pRun);
		const pSlot = params.get("slot");
		const pModel = params.get("model");
		const current = deepLink ? pRun : ((saved && names.includes(saved.run)) ? saved.run : ((!Array.isArray(data) && data.current) || names[0]));
		// Collapse ablation variants out of the top-level picker — they're
		// organized under their parent run in the ablation view, not listed as
		// standalone runs. Keep the current run selectable even if it's a variant
		// (e.g. opened via the board's "scene ↗" deep link).
		const pickerNames = names.filter((n) => !n.includes("__abl-") || n === current);
		$("tf-run").replaceChildren(...pickerNames.map((n) => el("option", { value: n, text: n, ...(n === current ? { selected: "" } : {}) })));
		if (current) {
			const prefer = deepLink && pSlot && pModel ? { run: pRun, slot: pSlot, model: pModel } : (saved && saved.run === current ? saved : null);
			await selectRun(current, prefer, data); // reuse the runs payload (skip a 2nd /runs scan)
		}
	} catch (e) {
		$("tf-export-wrap").replaceChildren(el("div", { class: "empty-note", text: `failed to load runs: ${e.message}` }));
	}
}

// Slots that have ablation variants for `run` — used to tag the scene selector
// and drive the ablation drawer. Cheap: reads /runs (which carries each run's
// ablation meta) and buckets by base+slot.
async function ablatedSlotsFor(run, _runsData = null) {
	try {
		// Variants are folded under the base run now, so read the base's ablations
		// index (one call) instead of filtering the top-level /runs list.
		const list = (await api.ablations(run)).variants ?? [];
		const set = new Set();
		for (const r of list) {
			const abl = r && r.ablation;
			if (abl && abl.slot) set.add(abl.slot);
		}
		return set;
	} catch { return new Set(); }
}

async function selectRun(run, prefer = null, runsData = null) {
	state.run = run;
	state.sceneAttentionCounts = new Map();
	state.sceneRowsCache = new Map();
	const data = await api.slots(run);
	const models = data.models ?? [];
	const slots = data.slots ?? [];
	state.models = models;
	state.slots = slots;
	$("tf-model").replaceChildren(...models.map((m) => el("option", { value: m, text: m })));
	// Tag scenes that have ablation variants with ⚗ so it's obvious which are ablated.
	const ablated = await ablatedSlotsFor(run, runsData);
	state.ablatedSlots = ablated;
	$("tf-slot").replaceChildren(...slots.map((s) => el("option", { value: s.id, text: ablated.has(s.id) ? `⚗ ${s.id}` : s.id })));
	// Restore the saved cell when reloading; otherwise prefer a cell that
	// actually has logged steps so the page opens on data.
	let pick = null;
	if (prefer && slots.some((s) => s.id === prefer.slot) && models.includes(prefer.model)) pick = { slot: prefer.slot, model: prefer.model };
	for (const s of slots) {
		if (pick) break;
		for (const m of models) {
			if ((s.runs?.[m]?.events_count ?? 0) > 0) { pick = { slot: s.id, model: m }; break; }
		}
	}
	if (!pick && slots.length && models.length) {
		pick = { slot: slots[0].id, model: data.defaultModel ?? models[0] };
	}
	if (pick) {
		$("tf-slot").value = pick.slot;
		$("tf-model").value = pick.model;
		await selectCell(pick.slot, pick.model);
	}
}

async function selectCell(slot, model) {
	bumpView(); // invalidate any in-flight fetches from the previous cell
	const run = state.run;
	const modelChanged = state.model !== model;
	state.slot = slot;
	state.model = model;
	// Show/refresh the ⚗ ablation drawer tab for this scene (hidden unless ablated).
	if (state.drawerTab === "ablation") switchTab(state.ablatedSlots?.has(slot) ? "ablation" : "attention");
	else syncTabHighlight();
	if (modelChanged) { state.sceneAttentionCounts = new Map(); state.sceneRowsCache = new Map(); }
	state.export = null;
	state.obs = null; // rebuilt from the cell's scene-tree projection below (id→kind for frames)
	state.attn = null;
	state.attnStatus = {}; // reset per-step status for the new cell
	state.attnErrors = {};
	state.attnPendingReload = new Set();
	state.attnPlan = null; // abandon any compute-all window from the previous cell
	state.attnServer = { queued: [], running: [], computed: [] };
	state.stepAnalyses = new Map(); // drop the previous cell's cross-step cache
	state.compactCache = new Map(); // drop cached compact payloads
	state.tokenDetailCache = new Map(); // and per-token detail
	state.tokenDetailPending = new Set();
	state.tokenDetail = null;
	state.pins = { steps: [], entities: [], heads: [], kinds: [], scenes: [], runs: [] };
	state.reportSelections = { stepIdx: 0, token: 0, head: 0 };
	state.openSelectGroups = new Set();
	state.closedSelectGroups = new Set();
	$("tf-attn-progress").textContent = ""; // clear stale global-compute progress
	$("tf-obs-wrap").replaceChildren(el("div", { class: "obsm-empty", text: "loading…" }));
	$("tf-export-wrap").replaceChildren(el("div", { class: "empty-note", text: "loading…" }));
	$("tf-attn-panel").replaceChildren(el("div", { class: "empty-note", text: "loading…" }));
	try {
		// Fetch steps, the compact scene tree, AND which steps already have a stored
		// analysis together, so the per-step indicators are correct on first paint
		// (and gotoStep(0) below can auto-load a computed step with no flicker). The
		// scene tree is a small server projection (api.tfTree) — the browser no
		// longer downloads + folds the whole (hundreds-of-MB) events.jsonl per cell.
		const [stepsResp, tree, computed] = await Promise.all([
			api.tfSteps(state.run, slot, model),
			api.tfTree(state.run, slot, model).catch((e) => { console.warn("[tf] tf-tree failed:", e); return { order: [], nodes: {} }; }),
			api.attentionList(run, slot, model, { maxHeads: state.maxHeads }).catch(() => ({ computed: [] })),
		]);
		state.steps = stepsResp.steps ?? [];
		// Whether attention is available for this cell's model (needs open HF
		// weights). Absent on an older server -> assume open (don't gate).
		state.attnModel = stepsResp.attention || { open: true, hf_url: null, hf_path: null };
		state.obs = obsFromTree(tree); // id→kind/parent + emitting provenance (recovers "frame" entities)
		applyServerStatus(computed); // seed per-step status (computed/running/queued) for first paint
		setSceneAttentionCount(slot, computed);
		renderAttnModelBadge(); // reflect open/closed model in the compute bar
		saveScene(); // remember this cell so a reload returns to it
	} catch (e) {
		$("tf-export-wrap").replaceChildren(el("div", { class: "empty-note", text: `failed: ${e.message}` }));
		return;
	}
	if (!state.steps.length) {
		$("tf-timeline").replaceChildren();
		$("tf-prev").disabled = $("tf-next").disabled = true;
		$("tf-step-line").textContent = "no LLM steps logged for this cell";
		$("tf-attn-progress").textContent = "";
		state.viewer.clear();
		$("tf-obs-wrap").replaceChildren(el("div", { class: "obsm-empty", text: "no pipeline yet" }));
		$("tf-attn-panel").replaceChildren(el("div", { class: "empty-note", text: "this cell has no cache.llm steps yet." }));
		$("tf-export-wrap").replaceChildren(el("div", { class: "empty-note", text: "this cell has no cache.llm steps — start it (optionally with logprob capture) from the dashboard." }));
		return;
	}
	gotoStep(Math.min(state.reportSelections?.stepIdx ?? 0, state.steps.length - 1));
	// If the server reports queued/running computes for this cell (another
	// viewer, or a prior session), keep reconciling until they settle.
	if (hasPending()) startAttnPoll();
}

boot();
