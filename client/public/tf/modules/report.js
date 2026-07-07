import * as graphs from "./reportGraphs.js";
import { COMPARE_COLORS } from "./reportGraphs.js";
import { oneWayAnova, mean } from "./ablationstats.js";
// state.js is the shared leaf every module imports directly; focusedKind reads
// the same singleton `state` that initReport receives via deps, so the scope
// focus stays consistent whether it's set here or read there.
import { focusedKind } from "./state.js";

// Modular analysis workspace for /tf.
// Owns report state, selection drawer, comparison policies, and graph board rendering.

let state, $, el, api, pool;
let stepKind, aggregateAttn, hasToPlace, _mean, templateColor, outputStartTok, niceMax;
let overviewAggregate, overviewKeyStats, overviewMassPerStep, overviewToPlace, overviewByKindComposition, overviewByKindMass, overviewEntities, overviewHeadGrid, overviewLayerDepth, attentionTree;
let summaryKeyStats, summaryTrajectory, summaryParents, summaryReasoningVsOutput, summaryEntityTokenMap, summaryEntities, summaryKind, summaryToPlace, summaryHeadEntity, summaryLayerDepth, outputsSection, headGrid;
let summaryBar, heatColor, compHex, COMPONENT_ABBR, COLORS;
let gotoStep, selectCell, ensureOverviewLoaded;
let updateReportCtx, syncTabHighlight, windowedScope;

const REPORT_MODES = ["step", "kind", "scene", "ablation"];
const REPORT_ALIASES = { summary: "step", overview: "scene", cell: "scene", placement: "step", heads: "scene" };
const REPORT_STATE_VERSION = 1;
const autoSelectSuppressed = { steps: new Set(), kinds: new Set(), scenes: new Set() };
let sceneAttrKind = null;
// The kind highlighted as "focused" in the selector, healed against the kinds
// actually present so it matches the kind the hero renders. Set in
// selectionDrawer (kind scope) just before the rows are built.
let renderedFocusKind = null;
// Pending rAF handle for the window slider, so a drag coalesces to one rebuild
// per frame instead of a full workspace re-render on every intermediate pixel.
let winRenderRaf = 0;

export function initReport(deps) {
	({ state, $, el, api, pool,
		stepKind, aggregateAttn, hasToPlace, _mean, templateColor, outputStartTok, niceMax,
		overviewAggregate, overviewKeyStats, overviewMassPerStep, overviewToPlace, overviewByKindComposition, overviewByKindMass, overviewEntities, overviewHeadGrid, overviewLayerDepth, attentionTree,
		summaryKeyStats, summaryTrajectory, summaryParents, summaryReasoningVsOutput, summaryEntityTokenMap, summaryEntities, summaryKind, summaryToPlace, summaryHeadEntity, summaryLayerDepth, outputsSection, headGrid,
		summaryBar, heatColor, compHex, COMPONENT_ABBR, COLORS,
		gotoStep, selectCell, ensureOverviewLoaded,
		updateReportCtx, syncTabHighlight, windowedScope } = deps);
	graphs.initGraphProtocol({
		el,
		showErr: () => state.showErr,
		xMode: () => state.tokenOrderXMode,
		normalize: () => state.bucketsNormalize,
		segmentOutput: () => state.segmentOutput,
		outputZoom: () => state.outputZoom,
		selectedSegment: () => state.segSel,
		selectSegment: (id) => { state.segSel = state.segSel === id ? null : id; renderReportWorkspace({ scrollSelection: false }); },
		aggregateAttn,
		hasToPlace,
		templateColor,
		outputStartTok,
		niceMax,
		overviewAggregate,
		overviewMassPerStep,
		overviewToPlace,
		overviewEntities,
		overviewHeadGrid,
		summaryToPlace,
		summaryBar,
		heatColor,
		compHex,
		COMPONENT_ABBR,
		COLORS,
		reportCard,
		reportEmpty,
		headGrid,
	});
}

export function normalizeReportView(view) {
	return REPORT_ALIASES[view] || (REPORT_MODES.includes(view) ? view : "step");
}

export function reportShowing(view) {
	return state.reportView === normalizeReportView(view) && !$("tf-report").hidden;
}

function countComputedAttention(r) {
	const ids = new Set([...(r?.computed || []), ...(r?.stale || [])].map(Number));
	return ids.size;
}

function setSceneAttentionCount(slotId, r) {
	if (!slotId) return;
	state.sceneAttentionCounts.set(String(slotId), countComputedAttention(r));
}

function sceneAttentionCount(slotId) {
	return state.sceneAttentionCounts.get(String(slotId)) || 0;
}

async function syncSceneAttentionCounts() {
	const run = state.run, model = state.model;
	if (!run || !model || !state.slots?.length) return;
	const slots = state.slots.map((s) => (typeof s === "string" ? s : s.id)).filter(Boolean);
	await pool(slots, 4, async (slotId) => {
		try {
			const r = await api.attentionList(run, slotId, model, { maxHeads: state.maxHeads });
			if (state.run === run && state.model === model) setSceneAttentionCount(slotId, r);
		} catch {
			if (state.run === run && state.model === model && !state.sceneAttentionCounts.has(String(slotId))) {
				state.sceneAttentionCounts.set(String(slotId), 0);
			}
		}
	});
	if (state.run === run && state.model === model && state.reportView) renderReportWorkspace({ scrollSelection: false });
	if (state.run === run && state.model === model) syncTabHighlight();
}

function defaultGraphModules() {
	return graphs.defaultGraphModules();
}

function reportStorageKey() {
	return state.run && state.slot && state.model ? `tf-report:${REPORT_STATE_VERSION}:${state.run}:${state.slot}:${state.model}` : null;
}

function saveReportState() {
	const key = reportStorageKey();
	if (!key) return;
	const data = {
		version: REPORT_STATE_VERSION,
		open: !!state.reportView,
		mode: state.reportView || state.lastReportMode,
		category: state.reportCategory,
		compareTree: state.compareTree,
		pins: state.pins,
		modules: state.graphModules,
		selections: { stepIdx: state.stepIdx, token: state.attnToken, head: state.attnHead, focusKind: state.focusKind },
	};
	try { localStorage.setItem(key, JSON.stringify(data)); } catch { /* private mode / quota */ }
}

function loadReportState() {
	state.reportRestoreOpen = false;
	const key = reportStorageKey();
	if (!key) return;
	let data = null;
	try { data = JSON.parse(localStorage.getItem(key) || "null"); } catch { data = null; }
	if (!data || data.version !== REPORT_STATE_VERSION) return;
	state.reportRestoreOpen = !!data.open;
	if (data.mode) state.lastReportMode = normalizeReportView(data.mode);
	if (data.category && typeof data.category === "object") state.reportCategory = { ...state.reportCategory, ...data.category };
	if (data.compareTree === "kind" || data.compareTree === "zone") state.compareTree = data.compareTree;
	// Comparison membership (step/kind/scene pins) is intentionally NOT restored:
	// a comparison is a transient exploration, so it starts empty each session and
	// can't silently persist ("stick") across reloads. It lives only in memory,
	// driven by the checkboxes, for the current session.
	if (Array.isArray(data.modules) && data.modules.length) state.graphModules = data.modules;
	state.graphModules = state.graphModules.filter((m) => m?.type !== "compareAttributes");
	state.graphModules = dedupeGraphModules(state.graphModules).map((m) => ({ ...m, wide: true }));
	if (data.selections && typeof data.selections === "object") {
		state.reportSelections = {
			stepIdx: Number(data.selections.stepIdx) || 0,
			token: Number(data.selections.token) || 0,
			head: Number(data.selections.head) || 0,
		};
		// Focus is a view preference (not compare membership), so it's safe to
		// restore; it self-heals in focusedKind() if the kind is gone.
		state.focusKind = data.selections.focusKind ?? null;
	}
}

function dedupeGraphModules(mods) {
	const seen = new Set();
	const out = [];
	for (const m of mods || []) {
		if (!m?.type || seen.has(m.type)) continue;
		seen.add(m.type);
		out.push(m);
	}
	return out;
}

function pinCount() {
	return Object.values(state.pins || {}).reduce((n, arr) => n + (Array.isArray(arr) ? arr.length : 0), 0);
}

function pinKey(kind, val) {
	return kind === "heads" && typeof val === "object" ? `${val.layer}:${val.head}` : String(val);
}

function isPinned(kind, val) {
	return (state.pins[kind] || []).includes(pinKey(kind, val));
}

function togglePin(kind, val) {
	const key = pinKey(kind, val);
	const arr = state.pins[kind] || (state.pins[kind] = []);
	const i = arr.indexOf(key);
	if (i >= 0) {
		arr.splice(i, 1);
		autoSelectSuppressed[kind]?.add(key);
	} else {
		arr.push(key);
		autoSelectSuppressed[kind]?.delete(key);
	}
	saveReportState();
	renderReportWorkspace({ scrollSelection: false });
}

// Set the FOCUSED kind at kind scope (the "primary" you're inspecting),
// independent of the 3D step cursor — clicking a kind here does NOT navigate
// the timeline the way clicking a step does. This is the fix for the primary
// selection being step-granular in the (non-step) kind organization.
function setFocusKind(kind) {
	state.focusKind = kind;
	saveReportState();
	renderReportWorkspace({ scrollSelection: false });
}

function selectionSummary() {
	const level = selectionLevel();
	const selected = level === "scene" ? state.pins.scenes : level === "kind" ? state.pins.kinds : state.pins.steps;
	const n = selected?.length || 0;
	return el("div", { class: "muted", style: "font-size:11px;line-height:1.5" },
		n ? `${n} selected ${level}${n === 1 ? "" : "s"} for comparison` : `check ${level}s to compare them in the graphs`);
}

function categoryControls(mode, choices) {
	const cur = currentCategory(mode, choices);
	return el("div", { class: "rep-controls", role: "tablist", "aria-label": `${mode} analysis category` }, ...choices.map((c) => el("button", {
		text: c.label, class: cur === c.id ? "on" : "",
		role: "tab",
		"aria-selected": cur === c.id ? "true" : "false",
		onclick: () => { state.reportCategory[mode] = c.id; saveReportState(); renderReportWorkspace({ scrollSelection: false }); },
	})));
}

function currentCategory(mode, choices) {
	const ids = new Set(choices.map((c) => c.id));
	const cur = ids.has(state.reportCategory[mode]) ? state.reportCategory[mode] : choices[0]?.id;
	if (cur && state.reportCategory[mode] !== cur) state.reportCategory[mode] = cur;
	return cur;
}

function reportCard(title, sub, ...body) {
	return el("div", { class: "graph-card" },
		el("div", { class: "graph-card-head" },
			el("span", { class: "graph-card-title", text: title }),
			sub ? el("span", { class: "graph-card-sub", text: sub }) : null),
		...body.filter(Boolean));
}

function reportHero(title, sub, ...body) {
	return el("div", { class: "rep-main hero" },
		el("div", { class: "rep-title-row" }, el("h2", { text: title }), sub ? el("span", { class: "muted", text: sub }) : null),
		...body.filter(Boolean));
}

function currentDisplayedAttention() {
	const step = state.steps[state.stepIdx];
	return !!(state.attn && step && state.attn.meta && state.attn.meta.event_index === step.event_index);
}

function overviewRows({ kickLoad = true } = {}) {
	const computed = state.steps.filter((s) => ["ready", "stale"].includes(state.attnStatus[s.event_index]));
	const rows = computed.map((s) => ({ step: s, a: state.stepAnalyses.get(s.event_index) })).filter((r) => r.a);
	const missing = computed.length - rows.length;
	if (kickLoad && missing > 0) ensureOverviewLoaded();
	for (const r of rows) {
		r.agg = aggregateAttn(r.a);
		r.template = stepKind(r.step);
		r.mass = _mean(r.agg.mass);
		r.hasTp = hasToPlace(r.a);
		r.tpMass = r.hasTp ? _mean(r.a.agg.to_place.mass || []) : null;
	}
	return { computed, rows, missing, order: [...new Set(rows.map((r) => r.template))] };
}

function finalizeRows(rows) {
	for (const r of rows) {
		r.agg = aggregateAttn(r.a);
		r.template = stepKind(r.step);
		r.mass = _mean(r.agg.mass);
		r.hasTp = hasToPlace(r.a);
		r.tpMass = r.hasTp ? _mean(r.a.agg.to_place.mass || []) : null;
	}
	return rows;
}

async function ensureSceneRows(slotId) {
	slotId = String(slotId);
	if (slotId === String(state.slot)) return overviewRows().rows;
	const cached = state.sceneRowsCache.get(slotId);
	if (cached?.rows || cached?.loading) return cached?.rows || [];
	const run = state.run, model = state.model;
	state.sceneRowsCache.set(slotId, { loading: true, rows: [] });
	try {
		const [stepsResp, status] = await Promise.all([
			api.tfSteps(run, slotId, model),
			api.attentionList(run, slotId, model, { maxHeads: state.maxHeads }),
		]);
		const steps = stepsResp.steps || [];
		const ready = new Set([...(status.computed || []), ...(status.stale || [])].map(Number));
		const computed = steps.filter((s) => ready.has(Number(s.event_index)));
		const rows = [];
		await pool(computed, 4, async (s) => {
			try {
				const a = await api.attentionGet(run, slotId, model, s.event_index, { view: "compact" });
				rows.push({ step: s, a });
			} catch { /* skip unreadable */ }
		});
		if (state.run === run && state.model === model) {
			finalizeRows(rows);
			state.sceneRowsCache.set(slotId, { loading: false, rows, order: [...new Set(rows.map((r) => r.template))] });
			if (state.reportView === "scene") renderReportWorkspace({ scrollSelection: false });
		}
		return rows;
	} catch {
		if (state.run === run && state.model === model) state.sceneRowsCache.set(slotId, { loading: false, rows: [] });
		return [];
	}
}

// Ablation runs to offer for comparison: the current run (the base/reference)
// plus any auto-named ablation variants (`…__abl-…`), each carrying its ablation
// meta so the drawer can group them into base+label families. Fetched once;
// re-renders when it lands.
let ablationRunItems = []; // [{ name, ablation }]
let ablationRunsLoaded = false;
function ensureAblationRuns() {
	if (ablationRunsLoaded) return;
	ablationRunsLoaded = true;
	Promise.resolve(api.runs()).then((data) => {
		const list = Array.isArray(data) ? data : (data.runs ?? []);
		ablationRunItems = list
			.map((r) => (typeof r === "string" ? { name: r, ablation: null } : { name: r.name, ablation: r.ablation || null }))
			.filter((r) => r.name && (r.name.includes("__abl-") || r.name === state.run));
		if (state.reportView === "ablation") renderReportWorkspace({ scrollSelection: false });
	}).catch(() => { ablationRunsLoaded = false; });
}

// A variant run → its family coordinates (prefers persisted ablation meta, with
// a best-effort name parse for older variants).
function ablationFamilyOf(item) {
	const abl = item.ablation && typeof item.ablation === "object" ? item.ablation : null;
	if (abl) {
		const t = abl.treatment || {};
		const tag = [t.shuffle_method || null, t.xml_tags === false ? "noxml" : null, t.attend_target ? `att-${String(t.attend_target).slice(0, 12)}` : null, t.distractors ? `d${t.distractors}` : null, t.seed ? `s${t.seed}` : null].filter(Boolean).join("_") || "order";
		return { base: abl.base_run || item.name.split("__abl-")[0], label: abl.label || "", kind: abl.target_step_kind || "", cut: abl.cut ?? "", tag };
	}
	const base = item.name.split("__abl-")[0];
	const rest = item.name.split("__abl-")[1] || "";
	const at = rest.indexOf("@");
	return { base, label: "", kind: at >= 0 ? rest.slice(0, at) : rest, cut: "", tag: rest };
}

// Pin / unpin a whole set of variant runs at once (the "select family" toggle).
function pinRunsBulk(names, on) {
	const arr = state.pins.runs || (state.pins.runs = []);
	for (const n of names) {
		const i = arr.indexOf(n);
		if (on && i < 0) arr.push(n);
		else if (!on && i >= 0) arr.splice(i, 1);
	}
	saveReportState();
	renderReportWorkspace({ scrollSelection: false });
}

// The computed attention rows of ANOTHER run's same cell (slot+model) — the
// cross-run analogue of ensureSceneRows, so ablation variants feed the same
// compare graphs. Cached per run; re-renders when the fetch completes.
async function ensureRunRows(runName) {
	runName = String(runName);
	if (runName === String(state.run)) return overviewRows().rows;
	const cached = state.runRowsCache.get(runName);
	if (cached?.rows?.length || cached?.loading) return cached?.rows || [];
	const slot = state.slot, model = state.model;
	state.runRowsCache.set(runName, { loading: true, rows: [] });
	try {
		const [stepsResp, status] = await Promise.all([
			api.tfSteps(runName, slot, model),
			api.attentionList(runName, slot, model, { maxHeads: state.maxHeads }),
		]);
		const steps = stepsResp.steps || [];
		const ready = new Set([...(status.computed || []), ...(status.stale || [])].map(Number));
		const computed = steps.filter((s) => ready.has(Number(s.event_index)));
		const rows = [];
		await pool(computed, 4, async (s) => {
			try { const a = await api.attentionGet(runName, slot, model, s.event_index, { view: "compact" }); rows.push({ step: s, a }); } catch { /* skip unreadable */ }
		});
		if (state.slot === slot && state.model === model) {
			finalizeRows(rows);
			state.runRowsCache.set(runName, { loading: false, rows, order: [...new Set(rows.map((r) => r.template))] });
			if (state.reportView === "ablation") renderReportWorkspace({ scrollSelection: false });
		}
		return rows;
	} catch {
		state.runRowsCache.set(runName, { loading: false, rows: [] });
		return [];
	}
}

function comparisonContext(baseRows = null) {
	const { rows, order } = baseRows ? { rows: baseRows, order: [...new Set(baseRows.map((r) => r.template))] } : overviewRows();
	const level = selectionLevel();
	if (level === "step") {
		const peers = rows
			.filter((r) => state.pins.steps.includes(String(r.step.event_index)))
			.map((r, i) => ({ key: String(r.step.event_index), label: `#${state.steps.indexOf(r.step) + 1} ${r.template}`, rows: [r], color: COMPARE_COLORS[i % COMPARE_COLORS.length] }));
		return { level, active: peers.length >= 2, peers, rows, order };
	}
	if (level === "kind") {
		const selected = state.pins.kinds || [];
		const peers = selected
			.map((kind, i) => ({ key: kind, label: kind, rows: rows.filter((r) => r.template === kind), color: COMPARE_COLORS[i % COMPARE_COLORS.length] }))
			.filter((p) => p.rows.length);
		return { level, active: peers.length >= 2, peers, rows, order };
	}
	if (level === "run") {
		// Ablation compare: peers are variant RUNS (same slot+model, different run),
		// reusing the identical peer machinery + graphs as scene/kind/step compare.
		const selectedRuns = state.pins.runs || [];
		for (const rn of selectedRuns) ensureRunRows(rn);
		const runPeers = selectedRuns.map((rn, i) => {
			const rr = String(rn) === String(state.run) ? rows : (state.runRowsCache.get(String(rn))?.rows || []);
			return { key: String(rn), label: rn, rows: rr, color: COMPARE_COLORS[i % COMPARE_COLORS.length] };
		}).filter((p) => p.rows.length);
		return { level, active: runPeers.length >= 2, peers: runPeers, rows, order, loading: selectedRuns.some((rn) => !state.runRowsCache.get(String(rn))?.rows && String(rn) !== String(state.run)) };
	}
	const selectedScenes = state.pins.scenes || [];
	for (const scene of selectedScenes) ensureSceneRows(scene);
	const peers = selectedScenes.map((scene, i) => {
		const sceneRows = String(scene) === String(state.slot) ? rows : (state.sceneRowsCache.get(String(scene))?.rows || []);
		return { key: String(scene), label: `scene ${scene}`, rows: sceneRows, color: COMPARE_COLORS[i % COMPARE_COLORS.length] };
	}).filter((p) => p.rows.length);
	return { level, active: peers.length >= 2, peers, rows, order, loading: selectedScenes.some((s) => !state.sceneRowsCache.get(String(s))?.rows && String(s) !== String(state.slot)) };
}

function renderReportWorkspace({ scrollSelection = true } = {}) {
	const host = $("tf-report-panel");
	if (!host || !$("tf-report") || $("tf-report").hidden || !state.reportView) return;
	ensureFocusedSelection();
	updateReportCtx();
	// The real scroll containers (main graph column + selection tree) are REBUILT
	// below, so capture their offsets and restore them onto the fresh nodes — else
	// every re-render (notably the status poll) snaps the view back to the top.
	// `tf-report-body` itself is overflow:hidden, so it never held the scroll.
	const prevContent = host.querySelector(".rep-content");
	const prevTree = host.querySelector(".sel-tree");
	const contentTop = prevContent ? prevContent.scrollTop : 0;
	const treeTop = prevTree ? prevTree.scrollTop : 0;
	let body;
	if (state.reportView === "step") body = renderStepWorkspace();
	else if (state.reportView === "kind") body = renderKindWorkspace();
	else if (state.reportView === "scene") body = renderSceneWorkspace();
	else if (state.reportView === "ablation") body = renderAblationWorkspace();
	else body = renderStepWorkspace();
	host.replaceChildren(body);
	const nextContent = host.querySelector(".rep-content");
	if (nextContent && contentTop) nextContent.scrollTop = contentTop;
	const nextTree = host.querySelector(".sel-tree");
	if (nextTree && treeTop) nextTree.scrollTop = treeTop;
	if (scrollSelection) scrollSelectionIntoView();
}

function ensureFocusedSelection() {
	// Comparison membership is driven ONLY by the explicit checkboxes. Navigating
	// (clicking into a step, kind, or scene) no longer auto-adds it to the compare
	// set, so browsing can't silently build — and then stick — a multi-select
	// comparison. Uncheck the boxes to leave comparison. State is still persisted
	// here on each render.
	saveReportState();
}

function workspaceWindowBar(rows) {
	const sorted = [...rows].sort((a, b) => (a.step?.event_index ?? 0) - (b.step?.event_index ?? 0));
	const total = sorted.length;
	if (total <= 1) return null;
	const n = Math.min(Math.max(1, state.reportLastN || 3), total);
	const label = el("span", { class: "rep-win-lab", text: n >= total ? `all ${total} steps` : `last ${n} of ${total}` });
	return el("div", { class: "rep-window" },
		el("span", { class: "rep-win-cap", text: "window" }),
		el("input", {
			type: "range", min: "1", max: String(total), value: String(n), class: "rep-win-slider",
			title: "limit all graphs to the last N computed steps",
			oninput: (ev) => {
				state.reportLastN = Number(ev.target.value);
				try { localStorage.setItem("tf-report-lastn", String(state.reportLastN)); } catch { /* ignore */ }
				// Dragging fires oninput per pixel; coalesce to one rebuild per frame so
				// the whole workspace isn't re-rendered on every intermediate value.
				if (winRenderRaf) return;
				winRenderRaf = requestAnimationFrame(() => { winRenderRaf = 0; renderReportWorkspace({ scrollSelection: false }); });
			},
		}),
		label);
}

function workspaceShell(mode, choices, hero, side, below = [], windowRows = null) {
	const winBar = windowRows?.length > 1 ? workspaceWindowBar(windowRows) : null;
	return el("div", { class: "rep-workspace" },
		categoryControls(mode, choices),
		winBar,
		el("div", { class: "rep-shell" },
			el("aside", { class: "rep-side" }, side),
			el("main", { class: "rep-content" },
				hero,
				...(below.length ? [el("div", { class: "graph-grid" }, ...below.filter(Boolean))] : []))));
}

function reportEmpty(txt, big = "•") {
	return el("div", { class: "rep-empty" }, el("div", { class: "big", text: big }), el("div", { text: txt }));
}

// Token-ordering section: a full-width control card (position-axis basis) plus
// the level-appropriate analyses. `rows` are finalized here so a single displayed
// step and an aggregate selection flow through the same graph functions.
const TOKEN_X_MODES = [{ id: "n", label: "by n" }, { id: "ratio", label: "by ratio" }, { id: "log", label: "log (RoPE)" }, { id: "feature", label: "per feature" }];
// How the stacked-area (category / section / word-type) graphs lay out X for a
// multi-step selection. A kind/scene selection COMPARES ACROSS its steps (one
// stacked column per group) — within-step token order lives at the single-step
// level, so it isn't offered here. First entry is the level default: a kind
// compares its steps, a scene its step-kinds (zone / step one click away).
const BUCKET_GROUPS = {
	kind: [["step", "by step"]],
	scene: [["kind", "by kind"], ["zone", "by zone"], ["step", "by step"]],
};
// Effective grouping for a level: the persisted choice if it's valid here, else
// the level default (first option). Step level is always within-step (one step).
function bucketGroupFor(level) {
	if (level === "step") return "progression";
	const opts = (BUCKET_GROUPS[level] || []).map((o) => o[0]);
	const g = state.bucketsGroup;
	return g && opts.includes(g) ? g : (opts[0] || "step");
}
function _toggleBtn(active, label, title, onclick) {
	return el("button", { class: active ? "on" : "", title, text: label, onclick });
}
function tokenOrderBar(level = "step") {
	const cur = state.tokenOrderXMode || "n";
	const persist = (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } };
	const group = bucketGroupFor(level);
	const bar = [
		el("span", { class: "graph-card-title", style: "font-size:13px;margin-right:2px", text: "attention breakdown" }),
	];
	if (level === "step") {
		// Single step: within-step token-position bases (n / ratio / log) plus the
		// "per feature" mode that maps scene-mass onto each emitted output item.
		for (const m of TOKEN_X_MODES) bar.push(_toggleBtn(cur === m.id, m.label,
			m.id === "feature" ? "map attention onto each output item the step emits (object / zone / …)" : "within-step token position basis", () => {
				state.tokenOrderXMode = m.id; persist("tf-token-x", m.id);
				renderReportWorkspace({ scrollSelection: false });
			}));
		// Output overlays across EVERY within-step breakdown graph (scene mass, category,
		// sections, word types). Not meaningful in "per feature" (already binned by
		// item), so hide the toggles there.
		//   ⊟ segment output — mark item + attribute (JSON field) boundaries.
		//   ⊙ output zoom    — trim the x-axis to just the output region so those
		//                      boundaries fill the width instead of squishing on the right.
		if (cur !== "feature") {
			bar.push(el("span", { class: "graph-tb-sep", style: "opacity:.3;margin:0 1px", text: "·" }));
			bar.push(_toggleBtn(!!state.segmentOutput, "⊟ segment output",
				"annotate the x-axis: output item + attribute (JSON field) boundaries on every breakdown graph, and the reasoning token-type strip under the scene-mass trajectory", () => {
					state.segmentOutput = !state.segmentOutput; persist("tf-seg-output", state.segmentOutput ? "1" : "0");
					renderReportWorkspace({ scrollSelection: false });
				}));
			bar.push(_toggleBtn(!!state.outputZoom, "⊙ output zoom",
				"trim every within-step breakdown graph to just the output region (drop reasoning) so per-attribute segmentation fills the width", () => {
					state.outputZoom = !state.outputZoom; persist("tf-output-zoom", state.outputZoom ? "1" : "0");
					renderReportWorkspace({ scrollSelection: false });
				}));
		}
	} else if (BUCKET_GROUPS[level]) {
		// Kind/scene: COMPARE the selection's steps by step # / kind / zone (one column
		// per group). These aren't token-ordering, so the within-step basis is dropped.
		bar.push(el("span", { class: "graph-tb-cap", style: "opacity:.5;font-size:11px;margin-left:4px", text: "compare" }));
		for (const [id, lab] of BUCKET_GROUPS[level]) {
			bar.push(_toggleBtn(group === id, lab, `one column per ${id === "step" ? "step" : id} — compare the selection's steps by ${id}`, () => {
				state.bucketsGroup = id; persist("tf-buckets-group", id);
				renderReportWorkspace({ scrollSelection: false });
			}));
		}
	}
	bar.push(el("span", { style: "flex:1" }));
	bar.push(_toggleBtn(!!state.bucketsNormalize, "÷ tokens", "normalize each attention mass by the scene's total token count (fair comparison across scenes of different lengths)", () => {
		state.bucketsNormalize = !state.bucketsNormalize; persist("tf-buckets-norm", state.bucketsNormalize ? "1" : "0");
		renderReportWorkspace({ scrollSelection: false });
	}));
	return el("div", { class: "graph-card wide", style: "padding:9px 14px" },
		el("div", { class: "graph-toolbar", style: "gap:8px" }, ...bar));
}

function tokenOrderSection(level, rows, cmp, allRows = null) {
	finalizeRows(rows);
	if (allRows) finalizeRows(allRows);
	return [tokenOrderBar(level), ...graphs.renderTokenOrdering(rows, { level, cmp, allRows, group: bucketGroupFor(level) })];
}

function selectionDrawer(extra = null) {
	const level = selectionLevel();
	const groups = compareTreeGroups(level);
	// Heal the focused kind against the kinds actually shown, so the selector's
	// highlight matches the kind the hero renders even after a stale restore.
	if (level === "kind") renderedFocusKind = focusedKind(groups.map((g) => g.label));
	const toggle = el("span", { class: "sel-toggle" },
		...["kind", "zone"].map((mode) => el("button", {
			text: mode, class: state.compareTree === mode ? "on" : "",
			title: mode === "kind" ? "Group as scene -> kind -> step" : "Group as scene -> zone -> step",
			onclick: () => { state.compareTree = mode; saveReportState(); renderReportWorkspace(); },
		})));
	const sceneRoots = sceneRows(groups, level);
	return el("div", {},
		el("div", { class: "sel-drawer-head" }, el("span", { class: "sel-drawer-title", text: `select ${level}s` }), level === "step" ? toggle : null),
		el("div", { class: "sel-tree" }, ...sceneRoots),
		extra ? el("div", { style: "margin-top:12px" }, extra) : null);
}

function selectionLevel() {
	if (state.reportView === "ablation") return "run";
	if (state.reportView === "scene") return "scene";
	if (state.reportView === "kind") return "kind";
	return "step";
}

function sceneRows(groups, level) {
	const slots = state.slots?.length ? state.slots : (state.slot ? [{ id: state.slot }] : []);
	const curModel = state.model;
	const visible = slots.filter((slot) => {
		const id = typeof slot === "string" ? slot : slot.id;
		return sceneAttentionCount(id) > 0;
	});
	if (!visible.length) {
		return [el("div", { class: "sel-root" },
			el("div", { class: "sel-root-title", text: state.slot ? `scene ${state.slot}` : "scene" }),
			el("div", { class: "muted", style: "font-size:11px;line-height:1.5", text: "no scenes with computed attention yet" }))];
	}
	return visible.map((slot) => {
		const id = typeof slot === "string" ? slot : slot.id;
		const isCur = id === state.slot;
		const count = sceneAttentionCount(id);
		const title = `scene ${id} · ${count} computed`;
		const sceneCheck = selectionCheckbox("scenes", id, `select ${title}`);
		if (!isCur) {
			return el("div", { class: "sel-root scene-closed" },
				el("div", { class: "scene-line" },
					sceneCheck,
					el("button", { class: "sel-root-title scene-switch", title: `open ${title}`, text: title,
						onclick: () => { if (id && curModel) { $("tf-slot").value = id; selectCell(id, curModel); } } })));
		}
		// At scene scope the loaded scene IS the focused scene — highlight it so the
		// primary reads at scene granularity, not as a step.
		return el("div", { class: `sel-root${level === "scene" ? " focused" : ""}` },
			el("div", { class: "scene-line" }, sceneCheck, el("div", { class: "sel-root-title", text: title })),
			...(level === "scene" ? [] : (groups.length ? groups.map((g) => selectionGroup(g, level)) : [el("div", { class: "muted", style: "font-size:11px;line-height:1.5", text: "no computed attention steps in this scene yet" })])));
	});
}

function selectionCheckbox(kind, val, title) {
	const checked = isPinned(kind, val);
	return el("input", {
		class: "sel-check", type: "checkbox", checked, title,
		onclick: (e) => { e.stopPropagation(); },
		onkeydown: (e) => { e.stopPropagation(); },
		onchange: (e) => { e.stopPropagation(); togglePin(kind, val); },
	});
}

function selectionBulkStepCheckbox(steps, title) {
	const evs = steps.map((s) => String(s.step.event_index));
	const selected = new Set(state.pins.steps || []);
	const all = evs.length > 0 && evs.every((ev) => selected.has(ev));
	const any = evs.some((ev) => selected.has(ev));
	const input = el("input", {
		class: "sel-check", type: "checkbox", checked: all, title,
		onclick: (e) => { e.stopPropagation(); },
		onkeydown: (e) => { e.stopPropagation(); },
		onchange: (e) => {
			e.stopPropagation();
			const cur = new Set(state.pins.steps || []);
			if (all) {
				for (const ev of evs) {
					cur.delete(ev);
					autoSelectSuppressed.steps.add(ev);
				}
			} else {
				for (const ev of evs) {
					cur.add(ev);
					autoSelectSuppressed.steps.delete(ev);
				}
			}
			state.pins.steps = [...cur];
			saveReportState();
			renderReportWorkspace({ scrollSelection: false });
		},
	});
	input.indeterminate = any && !all;
	return input;
}

function selectionGroup(g, level) {
	if (level === "kind") {
		// Rebuilt selector semantics (same as steps/scenes): the NAME sets focus,
		// the CHECKBOX toggles comparison. A <label> made the whole row toggle the
		// checkbox, which left no way to focus a kind without pinning it.
		const focused = renderedFocusKind === g.label;
		return el("div", { class: `sel-group pick-only${focused ? " focused" : ""}` },
			selectionCheckbox("kinds", g.label, isPinned("kinds", g.label) ? "remove from comparison" : "add to comparison"),
			el("button", {
				class: "sel-focus-name", title: `focus kind ${g.label}`, "aria-pressed": focused ? "true" : "false",
				onclick: () => setFocusKind(g.label),
			}, el("span", { class: "sel-group-name", title: g.label, text: g.label })),
			el("span", { class: "sel-count", text: `${g.steps.length}` }));
	}
	const key = `${state.slot}:${state.compareTree}:${g.label}`;
	const hasCur = g.steps.some((s) => s.idx === state.stepIdx);
	const hasPinned = g.steps.some((s) => isPinned("steps", s.step.event_index));
	const defaultOpen = hasCur || hasPinned;
	const open = state.openSelectGroups.has(key) || (defaultOpen && !state.closedSelectGroups.has(key));
	return el("div", { class: `sel-group${open ? " open" : ""}` },
		el("div", { class: "sel-group-head", title: open ? "collapse group" : "expand group", role: "button", tabIndex: 0,
			onclick: () => toggleSelectGroup(key, open),
			onkeydown: (e) => {
				if (e.key === "Enter" || e.key === " ") {
					e.preventDefault();
					toggleSelectGroup(key, open);
				}
			} },
			level === "step" ? selectionBulkStepCheckbox(g.steps, `select all steps in ${g.label}`) : null,
			el("span", { class: "sel-caret", text: open ? "▾" : "▸" }),
			el("span", { class: "sel-group-name", title: g.label, text: g.label }),
			el("span", { class: "sel-count", text: `${g.steps.length}` })),
		open ? el("div", { class: "sel-steps" }, ...g.steps.map((s) => selectionStepButton(s, level))) : null);
}

function toggleSelectGroup(key, open) {
	if (open) {
		state.openSelectGroups.delete(key);
		state.closedSelectGroups.add(key);
	} else {
		state.closedSelectGroups.delete(key);
		state.openSelectGroups.add(key);
	}
	renderReportWorkspace({ scrollSelection: false });
}

function compareTreeGroups(level = selectionLevel()) {
	const map = new Map();
	state.steps.forEach((s, idx) => {
		// The compare selector is for attention-bearing steps only. Early root
		// planning / overall bbox calls have no scene context, so they cannot
		// produce scene-attention stats and should not appear here.
		if (s.has_scene === false) return;
		// It is also only useful once an analysis exists. Hide uncomputed steps
		// from the compare selector so the drawer contains selectable data, not
		// pipeline placeholders.
		if (!["ready", "stale"].includes(state.attnStatus[s.event_index])) return;
		const key = level === "kind" ? stepKind(s) : (state.compareTree === "zone" ? (s.node || "(scene)") : stepKind(s));
		if (!map.has(key)) map.set(key, []);
		map.get(key).push({ step: s, idx });
	});
	return [...map.entries()].map(([label, steps]) => ({ label, steps }));
}

function selectionStepButton(rec, level = "step") {
	const ev = rec.step.event_index;
	const pinned = isPinned("steps", ev);
	const cur = rec.idx === state.stepIdx;
	return el("div", {
		class: `sel-step${cur ? " cur" : ""}${pinned ? " pinned" : ""}`,
		title: `step ${rec.idx + 1} · ${rec.step.template ?? rec.step.step ?? "?"} · ${rec.step.node ?? ""}`,
		onclick: () => gotoStep(rec.idx),
		onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); gotoStep(rec.idx); } },
		role: "button",
		tabIndex: 0,
	},
		level === "step" ? selectionCheckbox("steps", ev, pinned ? "remove from comparison" : "add to comparison") : el("span", {}),
		el("span", { class: "sel-step-idx", text: `#${rec.idx + 1}` }),
		el("span", { class: "sel-step-name", text: rec.step.template ?? rec.step.step ?? "?" }));
}

function scrollSelectionIntoView() {
	requestAnimationFrame(() => {
		const side = document.querySelector("#tf-report .sel-tree");
		if (!side) return;
		const target = side.querySelector(".sel-step.cur") || side.querySelector(".sel-step.pinned");
		if (!target) return;
		const s = side.getBoundingClientRect();
		const t = target.getBoundingClientRect();
		// Only move the drawer when the selected row is clipped; avoid fighting
		// normal reading/scrolling when it is already visible.
		if (t.top >= s.top + 8 && t.bottom <= s.bottom - 8) return;
		const delta = (t.top - s.top) - Math.max(42, side.clientHeight * 0.28);
		side.scrollTo({ top: side.scrollTop + delta, behavior: "smooth" });
	});
}

function renderStepWorkspace() {
	// Consistent view vocabulary/order across scopes (redesign canvas): the shared
	// views (overview · structure · attributes · breakdown · board) keep the same
	// names and order at every scope; step-only facets follow.
	const choices = [
		{ id: "main", label: "overview" }, { id: "tree", label: "structure" }, { id: "attributes", label: "attributes" },
		{ id: "tokenOrder", label: "breakdown" }, { id: "regions", label: "regions" }, { id: "output", label: "output" },
		{ id: "token", label: "token map" }, { id: "modules", label: "board" },
	];
	if (!currentDisplayedAttention()) {
		return workspaceShell("step", choices, reportHero("Step story", null, reportEmpty("compute this step's attention to build the step workspace")), selectionDrawer(selectionSummary()), []);
	}
	const a = state.attn, agg = aggregateAttn(a), cat = currentCategory("step", choices);
	const stepCmp = comparisonContext();
	const hasStepSelection = (state.pins.steps || []).length > 0;
	const hero = stepCmp.active
		? reportHero("Step comparison", `${stepCmp.peers.length} selected steps`, graphs.comparePeerAttributes(stepCmp, "attribute profile"))
		: hasStepSelection
			? reportHero("Selected step", `${a.tokens.length} tokens · ${(a.selected_heads || []).length} heads`, summaryTrajectory(a, agg))
			: reportHero("Step story", `${a.tokens.length} tokens · ${(a.selected_heads || []).length} heads`, summaryTrajectory(a, agg));
	const side = selectionDrawer(selectionSummary());
	const entityFocus = stepCmp.active
		? graphs.comparePeerEntities(stepCmp, "selected step entities")
		: reportCard("entity focus", "top attended scene objects", summaryEntities(agg), summaryKind(agg));
	const treeCard = (opts = {}) => reportCard("attention tree",
		`scene structure · badge ${opts.heatOnly ? "color" : "width"} = this step's attention · hover for full name · click → focus`,
		attentionTree([{ step: state.steps[state.stepIdx], a, agg }], opts));
	let below;
	if (cat === "tree") below = [treeCard()];
	else if (cat === "attributes") below = [stepCmp.active ? null : stepAttributeSection(agg)];
	else if (cat === "regions") below = [reportCard("regions and phases", stepCmp.active ? "structure · current step (single-step view)" : "structure", summaryParents(a, agg), summaryReasoningVsOutput(a))];
	else if (cat === "output") below = [reportCard("output reliance", stepCmp.active ? "current step (single-step view)" : "what emitted objects used", summaryReasoningVsOutput(a), outputsSection(a))];
	else if (cat === "token") below = [reportCard("entity × token map", stepCmp.active ? "per-token maps are single-step; select one step to inspect tokens" : "click to scrub", summaryEntityTokenMap(a, agg))];
	else if (cat === "tokenOrder") below = tokenOrderSection("step", [{ step: state.steps[state.stepIdx], a }], stepCmp);
	else if (cat === "modules") below = [renderOrgBoard("step")];
	// main dashboard: organization statistics + the attention tree live here (tree
	// also has its own tab). Not repeated in the other tabs.
	else below = stepCmp.active ? [entityFocus, treeCard({ heatOnly: true })] : [summaryKeyStats(a, agg), entityFocus, treeCard({ heatOnly: true })];
	return workspaceShell("step", choices, hero, side, below);
}

function renderKindWorkspace() {
	const choices = [
		{ id: "main", label: "overview" }, { id: "tree", label: "structure" }, { id: "attributes", label: "attributes" },
		{ id: "heads", label: "heads" }, { id: "tokenOrder", label: "breakdown" }, { id: "placement", label: "placement" },
		{ id: "modules", label: "board" },
	];
	const { computed, rows, missing, order } = overviewRows();
	if (!computed.length) return workspaceShell("kind", choices, reportHero("Step kind", null, reportEmpty("compute steps to compare this kind of step")), selectionDrawer(selectionSummary()), []);
	if (!rows.length) return workspaceShell("kind", choices, reportHero("Step kind", null, reportEmpty(`loading ${computed.length} step analyses…`)), selectionDrawer(selectionSummary()), []);
	// Kind scope focus is scope-granular: use the explicitly focused kind (healed
	// against the kinds actually present), not whatever kind the current 3D step
	// happens to belong to.
	const curKind = focusedKind(order);
	const kindRows = rows.filter((r) => r.template === curKind);
	const useRows = kindRows.length ? kindRows : rows;
	const cat = currentCategory("kind", choices);
	const kindCmp = comparisonContext();
	const scopedRows = kindCmp.active ? useRows : windowedScope(useRows);
	const hasKindSelection = (state.pins.kinds || []).length > 0;
	const hero = kindCmp.active
		? reportHero("Step-kind comparison", `${kindCmp.peers.length} selected kinds`, graphs.comparePeerAttributes(kindCmp, "kind attribute profile"))
		: reportHero(hasKindSelection ? "Selected step kind" : (curKind || "Step kind"), `${scopedRows.length} matching computed step${scopedRows.length === 1 ? "" : "s"}`,
			graphs.attributeProfileGraph(overviewAggregate(scopedRows).componentTotals, { title: "aggregate attribute shape", sub: curKind }));
	const side = selectionDrawer(selectionSummary());
	const entityFocus = kindCmp.active
		? graphs.comparePeerEntities(kindCmp, "selected kind entities")
		: reportCard("entities for this kind", "aggregate across matching steps", overviewEntities(scopedRows), overviewByKindComposition(scopedRows, [curKind]));
	const treeCard = (opts = {}) => reportCard("attention tree",
		`scene structure · badge ${opts.heatOnly ? "color" : "width"} = this kind's attention · hover for full name · click → focus`,
		attentionTree(scopedRows, { windowTotal: useRows.length, ...opts }));
	let below;
	if (cat === "tree") below = [treeCard()];
	else if (cat === "attributes") below = [kindCmp.active ? null : kindAttributeSection(scopedRows)];
	else if (cat === "placement") below = [reportCard("placement for this kind", "bbox/to-place when present", overviewToPlace(scopedRows) || reportEmpty("this kind has no computed to-place readout"))];
	else if (cat === "heads") below = [kindCmp.active ? graphs.comparePeerHeads(kindCmp, "selected kind head/layer attention") : reportCard("heads for this kind", "aggregate across matching steps", overviewHeadGrid(scopedRows), overviewLayerDepth(scopedRows))];
	else if (cat === "tokenOrder") below = tokenOrderSection("kind", scopedRows, kindCmp, useRows);
	else if (cat === "modules") below = [renderOrgBoard("kind", scopedRows)];
	// main dashboard: organization statistics + the attention tree live here (tree
	// also has its own tab). Not repeated in the other tabs.
	else below = kindCmp.active ? [entityFocus, treeCard({ heatOnly: true })] : [overviewKeyStats(scopedRows), entityFocus, treeCard({ heatOnly: true })];
	return workspaceShell("kind", choices, hero, side, below, kindCmp.active ? null : useRows);
}

// The ablation view's selection drawer: the current run (base/reference) plus
// variant runs grouped into base+label FAMILIES. A family header toggles its
// whole set into the comparison at once ("select family"); individual variants
// stay independently checkable. All reuse the shared compare-checkbox (pins.runs)
// machinery, so the peers feed the same graphs as scene/kind/step compare.
function ablationSelectionDrawer() {
	const cur = String(state.run);
	const items = ablationRunItems.length ? ablationRunItems : [{ name: cur, ablation: null }];
	const variants = items.filter((it) => it.name !== cur && it.name.includes("__abl-"));

	const fams = new Map();
	for (const it of variants) {
		const f = ablationFamilyOf(it);
		const key = `${f.base}\u0000${f.label}`;
		if (!fams.has(key)) fams.set(key, { base: f.base, label: f.label, rows: [] });
		fams.get(key).rows.push({ name: it.name, ...f });
	}

	const curRow = el("div", { class: "sel-group pick-only focused" },
		selectionCheckbox("runs", cur, isPinned("runs", cur) ? "remove from comparison" : "add the base run as the reference peer"),
		el("span", { class: "sel-group-name", title: cur, text: `${cur} (current · base)` }),
		el("span", {}));

	const famEls = [];
	for (const fam of fams.values()) {
		const names = fam.rows.map((r) => r.name);
		const allPinned = names.length > 0 && names.every((n) => isPinned("runs", n));
		const title = fam.label || "(unlabeled)";
		const famHead = el("div", { class: "sel-group", style: "gap:6px;margin-top:6px" },
			el("input", { class: "sel-check", type: "checkbox", checked: allPinned, title: "select / clear this whole family",
				onclick: (e) => { e.stopPropagation(); }, onchange: (e) => { e.stopPropagation(); pinRunsBulk(names, e.target.checked); } }),
			el("span", { class: "sel-group-name", title: `family: ${fam.base}${fam.label ? " · " + fam.label : ""}`, style: "font-weight:600", text: title }),
			el("span", { class: "muted", style: "font-size:11px;white-space:nowrap", text: `${names.length}` }));
		const rows = fam.rows.map((r) => el("div", { class: "sel-group pick-only", style: "padding-left:16px", title: r.name },
			selectionCheckbox("runs", r.name, isPinned("runs", r.name) ? "remove from comparison" : "add to comparison"),
			el("span", { class: "sel-group-name", text: r.cut !== "" ? `${r.kind}@${r.cut} · ${r.tag}` : (r.tag || r.kind || r.name) }),
			el("span", {})));
		famEls.push(el("div", { class: "sel-fam" }, famHead, ...rows));
	}
	if (!famEls.length) famEls.push(el("div", { class: "muted", style: "font-size:11px;padding:6px 2px", text: "no ablation variants yet — launch some from the ⚗ board" }));

	return el("div", {},
		el("div", { class: "sel-drawer-head" }, el("span", { class: "sel-drawer-title", text: "ablation families" })),
		el("div", { class: "sel-tree" }, curRow, ...famEls),
		el("div", { class: "muted", style: "font-size:11px;line-height:1.5;margin-top:8px", text: "check the base + a family (or single variants) to overlay them in the same graphs · monitor + launch on the ⚗ board" }));
}

// name -> { method, xml } from the fetched ablation variants' treatments.
function ablTreatmentByName() {
	const m = new Map();
	for (const it of ablationRunItems) {
		const a = it.ablation;
		if (a && a.treatment) m.set(it.name, { method: a.treatment.shuffle_method || "order", xml: a.treatment.xml_tags !== false });
	}
	return m;
}

const _fmtNum = (v) => (v == null || !Number.isFinite(v)) ? "—" : (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3));
const _fmtP = (p) => (p == null || !Number.isFinite(p)) ? "—" : (p < 0.001 ? "<0.001" : p.toFixed(3));

// ANOVA analysis of the compared variants: for each attention attribute, does
// the shuffle METHOD (one-way ANOVA over its levels) or XML on/off (2-group
// ANOVA ≡ t-test) significantly move it? Each variant is one replicate; grouping
// is marginal per factor. Shows per-method + per-xml means so the DIRECTION of
// each effect is legible, not just the p-value.
function renderAblationStats(cmp) {
	const treat = ablTreatmentByName();
	const samples = cmp.peers.map((p) => {
		const t = treat.get(String(p.key));
		const totals = overviewAggregate(p.rows).componentTotals || [];
		return t ? { method: t.method, xml: t.xml, comps: new Map(totals.map((c) => [c.component, c.score])) } : null;
	}).filter((s) => s && s.comps.size);
	if (samples.length < 3) {
		return reportCard("ANOVA — method & XML effects", "needs ≥3 computed variants (with treatments) selected",
			reportEmpty("select more variants + compute their attention (⚗ drawer / board), then they compare here"));
	}
	const methods = [...new Set(samples.map((s) => s.method))];
	const comps = [...new Set(samples.flatMap((s) => [...s.comps.keys()]))];
	const rows = comps.map((comp) => {
		const val = (s) => s.comps.get(comp);
		const has = (s) => val(s) != null;
		const methodScores = methods.map((mth) => samples.filter((s) => s.method === mth && has(s)).map(val));
		const methodMean = new Map(methods.map((mth, i) => [mth, methodScores[i].length ? mean(methodScores[i]) : null]));
		const xmlOn = samples.filter((s) => s.xml && has(s)).map(val);
		const xmlOff = samples.filter((s) => !s.xml && has(s)).map(val);
		return { comp, methodMean, aMethod: oneWayAnova(methodScores), xmlOnM: xmlOn.length ? mean(xmlOn) : null, xmlOffM: xmlOff.length ? mean(xmlOff) : null, aXml: oneWayAnova([xmlOn, xmlOff]) };
	}).sort((a, b) => (a.aMethod.p ?? 1) - (b.aMethod.p ?? 1));

	const sigMethod = rows.filter((r) => r.aMethod.ok && r.aMethod.p < 0.05).map((r) => r.comp);
	const sigXml = rows.filter((r) => r.aXml.ok && r.aXml.p < 0.05).map((r) => r.comp);
	const nOn = samples.filter((s) => s.xml).length, nOff = samples.length - nOn;

	const pCell = (a) => el("td", { class: `anv-p${a.ok && a.p < 0.05 ? " sig" : ""}`, title: a.ok ? `F(${a.dfB},${a.dfW})=${_fmtNum(a.F)} · η²=${_fmtNum(a.eta2)}` : "not enough replicates" }, a.ok ? _fmtP(a.p) : "n/a");
	const head = el("tr", {}, el("th", { text: "attribute" }),
		...methods.map((mth) => el("th", { title: `mean over ${mth} variants`, text: mth })),
		el("th", { text: "F" }), el("th", { text: "p·method" }),
		el("th", { class: "anv-div", text: `xml✓ (${nOn})` }), el("th", { text: `xml✗ (${nOff})` }), el("th", { text: "p·xml" }));
	const body = rows.map((r) => el("tr", {},
		el("td", { class: "anv-attr", text: r.comp }),
		...methods.map((mth) => el("td", { text: _fmtNum(r.methodMean.get(mth)) })),
		el("td", { text: r.aMethod.ok ? _fmtNum(r.aMethod.F) : "—" }), pCell(r.aMethod),
		el("td", { class: "anv-div", text: _fmtNum(r.xmlOnM) }), el("td", { text: _fmtNum(r.xmlOffM) }), pCell(r.aXml)));
	const table = el("table", { class: "abl-anova" }, el("thead", {}, head), el("tbody", {}, ...body));

	const summary = el("div", { class: "anv-summary" },
		el("div", {}, el("b", { text: "method" }), ` significantly moves (p<0.05): ${sigMethod.length ? sigMethod.join(", ") : "—"}`),
		el("div", {}, el("b", { text: "XML on/off" }), ` significantly moves (p<0.05): ${sigXml.length ? sigXml.join(", ") : "—"}`));
	return reportCard("ANOVA — method & XML effects",
		`${samples.length} variants · ${methods.length} methods · one-way ANOVA per attribute (each variant = one replicate)`,
		summary, table);
}

// Ablation scope: reuses the peer-comparison graphs, with the peers being the
// selected variant RUNS. Same graphs as scene/kind/step compare, different axis.
function renderAblationWorkspace() {
	const choices = [
		{ id: "attributes", label: "attributes" }, { id: "stats", label: "ANOVA" },
		{ id: "entities", label: "entities" }, { id: "heads", label: "heads" },
	];
	ensureAblationRuns();
	const cat = currentCategory("ablation", choices);
	const cmp = comparisonContext();
	const side = ablationSelectionDrawer();
	if (!cmp.active) {
		return workspaceShell("ablation", choices,
			reportHero("Ablation comparison", `${cmp.peers.length}/2 runs selected${cmp.loading ? " · loading…" : ""}`,
				reportEmpty("check at least two variant runs on the left to overlay their attention in the same graphs")),
			side, []);
	}
	if (cat === "stats") {
		return workspaceShell("ablation", choices,
			reportHero("Ablation ANOVA", `${cmp.peers.length} runs${cmp.loading ? " · loading…" : ""} · method & XML effects`, renderAblationStats(cmp)),
			side, []);
	}
	const hero = reportHero("Ablation comparison", `${cmp.peers.length} runs${cmp.loading ? " · loading…" : ""}`, graphs.comparePeerAttributes(cmp, "attribute profile across runs"));
	let below;
	if (cat === "entities") below = [graphs.comparePeerEntities(cmp, "entities across runs")];
	else if (cat === "heads") below = [graphs.comparePeerHeads(cmp, "heads / layers across runs")];
	else below = [graphs.comparePeerAttributes(cmp, "attribute profile across runs")];
	return workspaceShell("ablation", choices, hero, side, below);
}

function renderSceneWorkspace() {
	const choices = [
		{ id: "main", label: "overview" }, { id: "tree", label: "structure" }, { id: "entities", label: "entities" },
		{ id: "attributes", label: "attributes" }, { id: "heads", label: "heads" }, { id: "tokenOrder", label: "breakdown" },
		{ id: "modules", label: "board" },
	];
	const { computed, rows, missing, order } = overviewRows();
	if (!computed.length) return workspaceShell("scene", choices, reportHero("Scene organization", null, reportEmpty("compute steps to build the scene-level view")), selectionDrawer(selectionSummary()), []);
	if (!rows.length) return workspaceShell("scene", choices, reportHero("Scene organization", null, reportEmpty(`loading ${computed.length} step analyses…`)), selectionDrawer(selectionSummary()), []);
	const cat = currentCategory("scene", choices);
	const sceneCmp = comparisonContext();
	const scopedRows = sceneCmp.active ? rows : windowedScope(rows);
	const hasSceneSelection = (state.pins.scenes || []).length > 0;
	const hero = sceneCmp.active
		? reportHero("Scene comparison", `${sceneCmp.peers.length} selected scenes`, graphs.comparePeerAttributes(sceneCmp, "scene attribute profile"))
		: reportHero(hasSceneSelection ? "Selected scene" : "Scene organization", `${scopedRows.length}/${state.steps.length} computed${missing ? ` · loading ${missing}` : ""}`,
			renderGraphModule({ type: "pipelineMetric", metric: "sceneMass", title: "scene timeline" }, { rows: scopedRows, order }));
	const side = selectionDrawer(selectionSummary());
	const treeCard = (opts = {}) => reportCard("attention tree",
		`scene structure · badge ${opts.heatOnly ? "color" : "width"} = scene attention · hover for full name · click → focus`,
		attentionTree(scopedRows, { windowTotal: rows.length, ...opts }));
	let below;
	if (cat === "tree") below = [treeCard()];
	else if (cat === "attributes") below = [sceneCmp.active ? null : attributesByKindPicker(scopedRows, order)];
	else if (cat === "entities") below = [sceneCmp.active ? graphs.comparePeerEntities(sceneCmp, "selected scene entities") : reportCard("scene entities", "scene-level totals", overviewEntities(scopedRows))];
	else if (cat === "heads") below = [sceneCmp.active ? graphs.comparePeerHeads(sceneCmp, "selected scene head/layer attention") : reportCard("head depth", "network-level layer view", overviewLayerDepth(scopedRows))];
	else if (cat === "tokenOrder") below = tokenOrderSection("scene", scopedRows, sceneCmp, rows);
	else if (cat === "modules") below = [renderOrgBoard("scene", scopedRows)];
	// main dashboard: the most important scene stats. Organization statistics,
	// step-kind composition, step-kind scene mass, cell heads, and the attention
	// tree live here (tree also has its own tab). When comparing, composition/mass compare.
	else if (sceneCmp.active) below = [graphs.renderGraph("pipelineMetric", { cmp: sceneCmp, title: "step-kind scene mass", rows, order }), treeCard({ heatOnly: true })];
	else below = [
		overviewKeyStats(scopedRows),
		reportCard("composition by step kind", "scene make-up · step counts", overviewByKindComposition(scopedRows, order)),
		reportCard("step-kind scene mass", "which prompt modes consult scene", overviewByKindMass(scopedRows, order), overviewToPlace(scopedRows)),
		reportCard("cell heads", "scene-tracking heads", overviewHeadGrid(scopedRows)),
		treeCard({ heatOnly: true }),
	];
	return workspaceShell("scene", choices, hero, side, below, sceneCmp.active ? null : rows);
}

function renderPlacementWorkspace() {
	const choices = [{ id: "objects", label: "objects" }, { id: "cell", label: "cell bbox" }];
	const { rows } = overviewRows();
	if (state.reportCategory.placement === "cell") {
		const bbox = rows.filter((r) => r.hasTp);
		return workspaceShell("placement", choices,
			reportHero("Placement across cell", `${bbox.length} bbox steps`, bbox.length ? overviewToPlace(rows) : reportEmpty("no computed bbox/to-place analyses yet")),
			selectionSummary(),
			[graphs.compareAttributesGraph(bbox, { title: "bbox attributes", group: "step" })]);
	}
	if (!currentDisplayedAttention() || !hasToPlace(state.attn)) {
		return workspaceShell("placement", choices, reportHero("Placement step", null, reportEmpty("open a computed bbox-batch step to inspect placement attention")), selectionSummary(), []);
	}
	const a = state.attn;
	return workspaceShell("placement", choices,
		reportHero("Placement step", "self/spec versus surrounding scene", summaryToPlace(a)),
		selectionSummary(),
		[graphs.placementObjectsGraph(a)]);
}

function renderHeadsWorkspace() {
	const choices = [{ id: "grid", label: "current step" }, { id: "cell", label: "cell-wide" }];
	if (state.reportCategory.heads === "cell") {
		const { rows } = overviewRows();
		return workspaceShell("heads", choices,
			reportHero("Heads across cell", `${rows.length} computed steps`, rows.length ? overviewHeadGrid(rows) : reportEmpty("compute steps to see cell-wide heads")),
			selectionSummary(),
			[rows.length ? overviewLayerDepth(rows) : null]);
	}
	if (!currentDisplayedAttention()) return workspaceShell("heads", choices, reportHero("Current-step heads", null, reportEmpty("compute this step to inspect heads")), selectionSummary(), []);
	const a = state.attn, agg = aggregateAttn(a);
	return workspaceShell("heads", choices,
		reportHero("Current-step heads", "scene-tracking heads and layers", headGrid(a)),
		selectionSummary(),
		[summaryHeadEntity(a, agg), summaryLayerDepth(a)]);
}

function stepAttributeSection(agg) {
	const ctx = comparisonContext();
	if (ctx.active) return graphs.comparePeerAttributes(ctx, "selected step attributes");
	return graphs.attributeProfileGraph(agg.componentTotals, { title: "attribute profile", sub: "current step", chart: "radial" });
}

function kindAttributeSection(useRows) {
	const ctx = comparisonContext();
	if (ctx.active) return graphs.comparePeerAttributes(ctx, "selected kind attributes");
	// The hero already shows the aggregate attribute spider. Here we give the
	// complementary per-step breakdown (spider overlay + exact-value matrix) so
	// the tab isn't a copy of the hero.
	return graphs.compareAttributesGraph(useRows, { title: "attributes by step", group: "step" });
}

function attributesByKindPicker(rows, order) {
	const kinds = order.filter((k) => rows.some((r) => r.template === k && r.agg.componentTotals.length));
	if (!kinds.length) return reportCard("attributes by step kind", null, reportEmpty("no attributes recorded"));
	if (!sceneAttrKind || !kinds.includes(sceneAttrKind)) sceneAttrKind = kinds[0];
	const selectedRows = rows.filter((r) => r.template === sceneAttrKind);
	const tabs = el("div", { class: "graph-toolbar", style: "margin-bottom:8px" },
		...kinds.map((k) => el("button", {
			text: k,
			class: k === sceneAttrKind ? "on" : "",
			onclick: () => { sceneAttrKind = k; renderReportWorkspace({ scrollSelection: false }); },
		})));
	const profile = graphs.attributeProfileGraph(overviewAggregate(selectedRows).componentTotals, {
		title: "attributes by step kind",
		sub: sceneAttrKind,
		chart: "radial",
	});
	return reportCard("attributes by step kind", "select one kind", tabs, ...profile.childNodes);
}

const GRAPH_MODULE_TYPES = ["pipelineMetric", "attributeProfile", "entityRanking", "placementSplit", "headLayerMap"];

function renderOrgBoard(org, rowScope = null) {
	const base = overviewRows();
	const ctx = { ...base, rows: rowScope || base.rows, a: currentDisplayedAttention() ? state.attn : null };
	const add = (type) => el("button", {
		text: `+ ${type}`,
		disabled: state.graphModules.some((m) => m.type === type),
		onclick: () => addGraphModule(type),
	});
	const toolbar = el("div", { class: "graph-toolbar" }, ...GRAPH_MODULE_TYPES.map(add));
	const cards = state.graphModules.map((m, i) => graphModuleCard(m, ctx, i));
	return el("div", { class: "graph-card wide" },
		el("div", { class: "graph-card-head" },
			el("span", { class: "graph-card-title", text: `${org} modules` }),
			el("span", { class: "graph-card-sub", text: "saved local board" })),
		toolbar,
		el("div", { class: "muted", style: "font-size:11px;margin:8px 0 12px", text: "Modules are saved for this run/slot/model and shown inside each organization." }),
		el("div", { class: "graph-grid" }, ...(cards.length ? cards : [reportEmpty("add a graph module to start composing this organization")])));
}

function addGraphModule(type) {
	if (state.graphModules.some((m) => m.type === type)) return;
	const id = `g-${type}-${Date.now().toString(36)}`;
	const defaults = {
		pipelineMetric: { type, title: "pipeline timeline", metric: "sceneMass", scope: "scene", chart: "timeline", wide: true },
		attributeProfile: { type, title: "attribute profile", metric: "scene", scope: "step", chart: "radial", wide: true },
		entityRanking: { type, title: "entity ranking", metric: "entities", scope: "scene", chart: "bars", wide: true },
		placementSplit: { type, title: "placement split", metric: "toPlace", scope: "step", chart: "bars", wide: true },
		headLayerMap: { type, title: "head/layer map", metric: "sceneMass", scope: "scene", chart: "heatmap", wide: true },
	};
	state.graphModules.push({ id, ...(defaults[type] || defaults.pipelineMetric) });
	saveReportState();
	renderReportWorkspace();
}

function graphModuleCard(mod, ctx, idx) {
	const body = renderGraphModule(mod, ctx);
	const cls = `graph-card${mod.wide ? " wide" : ""}`;
	const content = body?.classList?.contains("graph-card") ? [...body.childNodes] : [body];
	return el("div", { class: cls },
		el("div", { class: "graph-card-head" },
			el("span", { class: "graph-card-title", text: mod.title || mod.type }),
			el("span", { class: "graph-card-sub", text: mod.scope || "" }),
			el("span", { class: "graph-card-actions" },
				el("button", { text: "↑", disabled: idx <= 0, onclick: () => moveGraphModule(idx, -1) }),
				el("button", { text: "↓", disabled: idx >= state.graphModules.length - 1, onclick: () => moveGraphModule(idx, 1) }),
				el("button", { text: "remove", onclick: () => removeGraphModule(mod.id) }))),
		...content.filter(Boolean));
}

function moveGraphModule(idx, delta) {
	const j = idx + delta;
	if (j < 0 || j >= state.graphModules.length) return;
	const arr = state.graphModules;
	[arr[idx], arr[j]] = [arr[j], arr[idx]];
	saveReportState();
	renderReportWorkspace();
}

function removeGraphModule(id) {
	state.graphModules = state.graphModules.filter((m) => m.id !== id);
	saveReportState();
	renderReportWorkspace();
}

function renderGraphModule(mod, ctx = {}) {
	return graphs.renderModuleGraph(mod, { ...ctx, cmp: comparisonContext(ctx.rows || []) });
}


export {
	defaultGraphModules,
	saveReportState,
	loadReportState,
	pinCount,
	setSceneAttentionCount,
	sceneAttentionCount,
	syncSceneAttentionCounts,
	renderReportWorkspace,
};
