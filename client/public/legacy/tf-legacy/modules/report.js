import * as graphs from "./reportGraphs.js";
import { COMPARE_COLORS } from "./reportGraphs.js";
import { spearman } from "./uncertainty.js";
import { mountSpatial3D, resetSpatial3DView } from "./scatter3d.js";
import { oneWayAnova, twoWayAnova, tost, mean } from "./ablationstats.js";
import { hasSceneContext } from "../../js/ablationcore.js";
import { emittingRegion } from "../../js/events.js";
// state.js is the shared leaf every module imports directly; focusedKind reads
// the same singleton `state` that initReport receives via deps, so the scope
// focus stays consistent whether it's set here or read there.
import { focusedKind, entityKindLabel, entityHex, COMPONENT_COLORS, treeHoverReset, treeHoverRegister } from "./state.js";

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
// Ablation selection tree: which kind groups are expanded (persists across the
// status re-renders). Seeded once to open kinds that already have a selection.
const ablExpandKind = new Set();
let ablTreeSeeded = false;
// Ablation comparison controls (module-scoped; a full re-render reads them).
// The comparison is AUTO-DISCOVERED: pick a FACTOR (the axis compared) + a SCOPE
// (which step kind), and the peers are the factor's levels, each aggregating every
// computed variant at that level across steps (marginal over the other factors).
let ablFactor = "method";     // generic-view compare axis: "method" | "xml" (sections override)
let ablScope = "all";         // kind scope: "all" | a specific step kind
let ablLogY = false;          // log-scale the attention (y) axis on the ordering graphs
let ablManual = false;        // true → the collapsible drill-down's pins narrow the comparison
let ablManualOpen = false;    // is the manual drill-down section expanded
let ablXmlKind = null;        // XML-existence: which single step kind the 2nd card shows
let ablFocusR = 0;            // scene-ordering focus: min attention SHARE (of the step's context) to plot an object
let ablOrderAttr = null;      // graph-2 attribute selector
// Spatial-relevance graph: per-firing geometry (distance + ray-trace visibility to
// the target), fetched on the "compute" click. Cache is keyed by base cell + cut.
const ablSpatialCache = new Map(); // `${base}\0${slot}\0${model}\0${cut}` -> Map(objId -> {distance, ray_hits})
let ablSpatialComputed = false;
let ablSpatialLoading = false;
let ablSpatialColorBy = "method"; // color the spatial scatter by "method" (scene order) | "kind" | "xml"
let ablSpatial3D = false;         // spatial scatter: 2D jointplot vs 3D (z = attention)
let ablSpatialLogZ = false;       // 3D: log-scale the attention height
let ablSpatialCoM = true;         // 3D: show each category's attention-weighted center of mass
let ablTostDelta = 0.2;           // TOST equivalence margin, as a fraction of each attribute's grand mean
const orderCache = new Map(); // `${run}\0${slot}\0${model}\0${ev}` -> Map(id->normPos) | "loading"
const ORDER_METHOD_COLORS = { order: "#7aa2f7", random: "#e0a94a", distance: "#6bd96e", raytrace: "#b46aff", attend: "#ff6b9d" };
const XML_LEVEL_COLOR = { "xml \u2713": "#6bd96e", "xml \u2717": "#ff6b9d" };
const ablLevelColor = (factor, level) => factor === "xml" ? (XML_LEVEL_COLOR[level] || "#7aa2f7") : (ORDER_METHOD_COLORS[level] || "#7aa2f7");
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
			// GATE: the base run + only ABLATABLE variants (scene-context kinds).
			// overall_bbox, *_root and other no-scene variants never enter the view.
			.filter((r) => r.name && (r.name === state.run || (r.name.includes("__abl-") && hasSceneContext(ablationFamilyOf(r).kind))));
		_cutRank = null; // variant set changed — recompute cut ranks (color lightness) lazily
		_ablComputedKey = null; // re-scan which variants have attention
		// Drop any stale pins for now-filtered (ineligible) variants so they can't
		// linger as invisible comparison peers with no way to uncheck them.
		const allowed = new Set(ablationRunItems.map((r) => r.name));
		if (Array.isArray(state.pins.runs)) state.pins.runs = state.pins.runs.filter((n) => allowed.has(String(n)));
		if (state.reportView === "ablation") renderReportWorkspace({ scrollSelection: false });
	}).catch(() => { ablationRunsLoaded = false; });
}

// Which of this cell's variants actually HAVE an attention result (fresh OR stale
// — both WERE computed; only never-run / failed ones lack one). Cheap per-variant
// disk-index scans (no hydration, no Modal), so the COMPARE selector lists ONLY
// runs you can overlay — no half-finished / failed / never-computed clutter.
// Rebuilt on cell change / variant reload / an explicit ↻.
let _ablComputed = new Set();
let _ablComputedKey = null;
let _ablComputedLoading = false;
function ensureAblationComputed(force = false) {
	if (!ablationRunItems.length) return; // variant list not loaded yet — don't cache an empty scan
	const key = `${state.slot}\u0000${state.model}`;
	if ((!force && _ablComputedKey === key) || _ablComputedLoading) return;
	const cands = ablationRunItems.filter((it) => it.name !== String(state.run) && it.name.includes("__abl-"));
	if (!cands.length) { _ablComputed = new Set(); _ablComputedKey = key; return; }
	_ablComputedLoading = true;
	const slot = state.slot, model = state.model;
	const found = new Set();
	Promise.resolve(pool(cands, 10, async (it) => {
		try {
			const s = await api.attentionIndex(it.name, slot, model, { maxHeads: state.maxHeads });
			if ((s.fresh || []).length || (s.stale || []).length) found.add(it.name);
		} catch { /* unreadable → treat as not-computed */ }
	})).then(() => {
		_ablComputedLoading = false;
		if (state.slot !== slot || state.model !== model) return; // cell changed mid-scan
		_ablComputed = found;
		_ablComputedKey = key;
		// A pin that turned out to have no attention (failed / never computed) is
		// dropped so it can't linger as an invisible, un-uncheckable peer.
		if (Array.isArray(state.pins.runs)) state.pins.runs = state.pins.runs.filter((n) => String(n) === String(state.run) || found.has(String(n)));
		if (state.reportView === "ablation") renderReportWorkspace({ scrollSelection: false });
	}).catch(() => { _ablComputedLoading = false; });
}

// ---- auto-discovered comparison -------------------------------------------
// The ablation comparison is DISCOVERED, not hand-selected: the computed variants
// for this cell form a factorial design (kind × cut × method × xml), and a view
// compares ONE factor while aggregating everything else across steps.

// The computed variants in the current scope (kind), optionally narrowed to the
// collapsible drill-down's manual pins. The single source the comparison is
// discovered from — no arbitrary per-run selection.
function ablScopedVariants() {
	let items = ablationRunItems.filter((it) => it.name !== String(state.run) && it.name.includes("__abl-") && _ablComputed.has(it.name));
	if (ablScope !== "all") items = items.filter((it) => (ablMeta(it.name)?.kind) === ablScope);
	if (ablManual && (state.pins.runs || []).length) {
		const pins = new Set((state.pins.runs || []).map(String));
		items = items.filter((it) => pins.has(it.name));
	}
	return items;
}

// Tag each row with its source run + shuffle method so the ordering scatter stays
// peer-agnostic (reads r._run / r._method instead of assuming one run per peer).
function tagRows(rows, run, method) {
	for (const r of rows) { r._run = run; r._method = method; }
	return rows;
}

// Per-variant samples (each variant = one replicate: its kind, method, xml, and
// computed rows). Feeds the per-kind XML split + the ANOVA, which need per-variant
// granularity. Kicks row loads; rows fill in as they land.
function ablVariantSamples() {
	const out = [];
	for (const it of ablScopedVariants()) {
		const m = ablMeta(it.name);
		if (!m) continue;
		ensureRunRows(it.name);
		const rows = state.runRowsCache.get(it.name)?.rows || [];
		if (rows.length) out.push({ name: it.name, kind: m.kind, method: m.method, xml: m.xml, rows });
	}
	return out;
}

// Auto peers = the LEVELS of `factor`, each aggregating every scoped variant at
// that level across steps (marginal over the other factors). This replaces manual
// run-picking for the spider / matrix / ordering graphs.
function ablAutoPeers(factor) {
	const byLevel = new Map();
	for (const it of ablScopedVariants()) {
		const m = ablMeta(it.name);
		if (!m) continue;
		const level = factor === "xml" ? (m.xml ? "xml \u2713" : "xml \u2717") : m.method;
		(byLevel.get(level) || byLevel.set(level, []).get(level)).push({ name: it.name, method: m.method });
	}
	const order = factor === "xml" ? ["xml \u2713", "xml \u2717"] : METHOD_ORDER;
	const levels = [...byLevel.keys()].sort((a, b) => (order.indexOf(a) - order.indexOf(b)) || (a < b ? -1 : 1));
	let loading = false;
	const peers = levels.map((level) => {
		const members = byLevel.get(level);
		for (const mem of members) ensureRunRows(mem.name);
		let rows = [];
		for (const mem of members) {
			const c = state.runRowsCache.get(mem.name);
			if (c?.rows?.length) rows = rows.concat(tagRows(c.rows, mem.name, mem.method));
			else if (!c || c.loading) loading = true;
		}
		return { key: level, label: `${level} · ${members.length}`, rows, color: ablLevelColor(factor, level), n: members.length, method: factor === "method" ? level : null };
	}).filter((p) => p.rows.length);
	return { level: "run", factor, active: peers.length >= 1, peers, loading, groups: levels.length };
}

// A variant run → its family coordinates (prefers persisted ablation meta, with
// a best-effort name parse for older variants).
function ablationFamilyOf(item) {
	const abl = item.ablation && typeof item.ablation === "object" ? item.ablation : null;
	if (abl) {
		const t = abl.treatment || {};
		// SEED is the replicate axis (each replicate bumps it), so it's EXCLUDED from
		// the family tag — a cell's replicates group as one treatment, and `rep`
		// carries the index. This is what lets replicates pool as samples in the stats.
		const tag = [t.shuffle_method || null, t.xml_tags === false ? "noxml" : null, t.attend_target ? `att-${String(t.attend_target).slice(0, 12)}` : null, t.distractors ? `d${t.distractors}` : null].filter(Boolean).join("_") || "order";
		return { base: abl.base_run || item.name.split("__abl-")[0], label: abl.label || "", kind: abl.target_step_kind || "", cut: abl.cut ?? "", tag, rep: Number(abl.replicate) || 1 };
	}
	const base = item.name.split("__abl-")[0];
	let rest = item.name.split("__abl-")[1] || "";
	const repM = rest.match(/-r(\d+)$/); // strip a trailing replicate suffix
	const rep = repM ? Number(repM[1]) : 1;
	if (repM) rest = rest.slice(0, -repM[0].length);
	const at = rest.indexOf("@");
	return { base, label: "", kind: at >= 0 ? rest.slice(0, at) : rest, cut: "", tag: rest, rep };
}

// ---- ablation color coding ------------------------------------------------
// A variant's color is a PURE function of its treatment coordinates, so the
// swatch in the left selection is EXACTLY the line color in the graph — never an
// arbitrary per-selection index. It's also semantic: hue encodes the shuffle
// METHOD (the main axis), saturation encodes XML on/off, and lightness steps by
// the firing (cut) rank within the kind, so same-method variants stay a
// recognizable color family while every one is still distinguishable.
const METHOD_HUE = { order: 214, random: 38, distance: 138, raytrace: 276, attend: 330 };
const METHOD_ORDER = ["order", "random", "distance", "raytrace", "attend"];
const ABL_BASE_COLOR = "#dfe8f5"; // the base run's reference line (neutral, bright)

function hslHex(h, s, l) {
	s /= 100; l /= 100;
	const a = s * Math.min(l, 1 - l);
	const f = (n) => {
		const k = (n + h / 30) % 12;
		const c = l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
		return Math.round(255 * c).toString(16).padStart(2, "0");
	};
	return `#${f(0)}${f(8)}${f(4)}`;
}

// Rank each kind's cuts (ascending) so lightness steps stably per firing. Rebuilt
// whenever the variant list reloads (reset in ensureAblationRuns).
let _cutRank = null;
function cutRankMap() {
	if (_cutRank) return _cutRank;
	const byKind = new Map();
	for (const it of ablationRunItems) {
		const f = ablationFamilyOf(it);
		if (!f.kind) continue;
		(byKind.get(f.kind) || byKind.set(f.kind, new Set()).get(f.kind)).add(String(f.cut));
	}
	_cutRank = new Map();
	for (const [kind, cuts] of byKind) {
		[...cuts].sort((a, b) => Number(a) - Number(b)).forEach((c, i) => _cutRank.set(`${kind}\u0000${c}`, i));
	}
	return _cutRank;
}

function ablColor(kind, cut, method, xml) {
	const rank = cutRankMap().get(`${kind}\u0000${String(cut)}`) ?? 0;
	const h = METHOD_HUE[method] ?? 220;
	const s = xml ? 72 : 46;
	const l = Math.min(80, 50 + (rank % 4) * 8 + (xml ? 0 : 12));
	return hslHex(h, s, l);
}

// name -> { kind, cut, method, xml, tag } from the fetched variant metadata.
function ablMeta(name) {
	const it = ablationRunItems.find((x) => x.name === name);
	if (!it || !it.ablation) return null;
	const f = ablationFamilyOf(it);
	const t = it.ablation.treatment || {};
	return { kind: f.kind, cut: f.cut, method: t.shuffle_method || "order", xml: t.xml_tags !== false, tag: f.tag, rep: f.rep };
}
function ablColorForRun(name) {
	if (String(name) === String(state.run)) return ABL_BASE_COLOR;
	const m = ablMeta(name);
	return m ? ablColor(m.kind, m.cut, m.method, m.xml) : "#8a8f99";
}
function ablLabelForRun(name) {
	if (String(name) === String(state.run)) return "base";
	const m = ablMeta(name);
	return m ? `${m.kind}@${m.cut} · ${m.method}${m.xml ? "" : "·noxml"}` : String(name);
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
		// "agg" = the compact MINUS the heavy per-token `tokens` array: the ablation
		// comparison only reads `r.agg` (entity/attribute rollups), so pulling full
		// compacts for every variant is what froze the page (e.g. the spatial scatter).
		await pool(computed, 4, async (s) => {
			try { const a = await api.attentionGet(runName, slot, model, s.event_index, { view: "agg" }); rows.push({ step: s, a }); } catch { /* skip unreadable */ }
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
		// Color + label are DETERMINISTIC per variant (treatment coordinates), not a
		// per-selection index — so a graph line's color is the swatch shown next to
		// it on the left, and its legend reads as the treatment, not the raw run name.
		const runPeers = selectedRuns.map((rn) => {
			const rr = String(rn) === String(state.run) ? rows : (state.runRowsCache.get(String(rn))?.rows || []);
			return { key: String(rn), label: ablLabelForRun(String(rn)), rows: rr, color: ablColorForRun(String(rn)) };
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
	treeHoverReset(); // rebuild the badge<->plan-phrase cross-highlight index for this render
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
	refreshStoryWindows(); // keep any detached plan/output/reasoning windows in sync with the step
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

function workspaceShell(mode, choices, hero, side, below = [], windowRows = null, loading = null) {
	const winBar = windowRows?.length > 1 ? workspaceWindowBar(windowRows) : null;
	return el("div", { class: "rep-workspace" },
		categoryControls(mode, choices),
		loading ? loadingBar(loading) : null,
		winBar,
		el("div", { class: "rep-shell" },
			el("aside", { class: "rep-side" }, side),
			el("main", { class: "rep-content" },
				hero,
				...(below.length ? [el("div", { class: "graph-grid" }, ...below.filter(Boolean))] : []))));
}

// A slim progress bar under the category pills while the workspace loads data.
// `loading`: `true` → indeterminate (animated sweep); `{ done, total, label }` →
// determinate (e.g. how many variants of the ablation aggregation have landed).
function loadingBar(loading) {
	const indet = loading === true;
	const { done = 0, total = 0, label } = indet ? {} : loading;
	const pct = indet ? 0 : (total ? Math.min(100, Math.round((100 * done) / total)) : 0);
	const text = indet ? (label || "loading…") : (label || `loading ${done}/${total}…`);
	return el("div", { class: `rep-loading${indet ? " indeterminate" : ""}`, title: text },
		el("div", { class: "rep-loading-track" }, el("div", { class: "rep-loading-fill", style: indet ? "" : `width:${pct}%` })),
		el("span", { class: "rep-loading-label", text }));
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
	// "View text" button on the right of the badge → popup with this step's exact
	// input / reasoning / output. stopPropagation so it doesn't also select the step.
	const textBtn = el("button", {
		class: "sel-step-text", type: "button", text: "≡",
		title: "view this step's text — input · reasoning · output",
		onclick: (e) => { e.stopPropagation(); openStepTextPanel(rec); },
		onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); openStepTextPanel(rec); } },
	});
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
		el("span", { class: "sel-step-name", text: rec.step.template ?? rec.step.step ?? "?" }),
		textBtn);
}

// Last dragged position of the step-text window (session-remembered) so reopening
// keeps where you parked it; null → the default bottom-left.
let stepPanelPos = null;
// Formatted (highlighted + pretty-printed) vs raw-verbatim step text. Persisted.
let stepPanelFmt = (() => { try { return localStorage.getItem("tf-steptext-fmt") !== "0"; } catch { return true; } })();

const _escapeRe = (s) => String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
// Translucent variant of an hsl()/#hex color for a faint highlight background.
function _alpha(color, a) {
	if (color.startsWith("hsl(")) return color.replace("hsl(", "hsla(").replace(")", `,${a})`);
	const h = color.replace("#", "");
	const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
	return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

// Render `text` into an array of DOM nodes, wrapping every mention of a known scene
// entity (object / zone / frame / to-place) and every JSON attribute key so it reads
// at a glance. Entities are UNDERLINED in their scene-tree attention color — heat
// (heatColor of score ÷ peak) when this step has computed attention, else the flat
// kind color — so the popup mirrors the tree's coloring. Attribute keys take their
// component color. ctx: { ent:Map(id→{score,kind}), max, entities:Set, kindOf:Map }.
function _highlightNodes(text, ctx) {
	const s = String(text || "");
	if (!s) return [];
	const ids = [...ctx.entities].filter(Boolean).sort((a, b) => b.length - a.length);
	const attrs = Object.keys(COMPONENT_COLORS);
	const idAlt = ids.map(_escapeRe).join("|");
	const atAlt = attrs.map(_escapeRe).join("|");
	const src = [idAlt ? `\\b(?:${idAlt})\\b` : null, atAlt ? `"(?:${atAlt})"(?=\\s*:)` : null].filter(Boolean).join("|");
	if (!src) return [document.createTextNode(s)];
	const re = new RegExp(src, "g");
	const out = [];
	let last = 0, m;
	while ((m = re.exec(s))) {
		if (m.index === re.lastIndex) { re.lastIndex++; continue; } // zero-width guard
		if (m.index > last) out.push(document.createTextNode(s.slice(last, m.index)));
		const hit = m[0];
		if (hit.startsWith('"')) {
			const name = hit.slice(1, -1);
			out.push(el("span", { class: "stx-attr", style: `color:${COMPONENT_COLORS[name] || "#888"}`, title: `attribute · ${name}`, text: hit }));
		} else {
			out.push(_entitySpan(hit, ctx));
		}
		last = re.lastIndex;
	}
	if (last < s.length) out.push(document.createTextNode(s.slice(last)));
	return out;
}

function _entitySpan(id, ctx) {
	const info = ctx.ent.get(id);
	const kind = (info && info.kind) || ctx.kindOf.get(id) || "object";
	const kindLabel = entityKindLabel(kind, id);
	let color, title;
	if (info && ctx.max > 0) {
		color = heatColor(info.score / ctx.max);   // same heat ramp as the scene tree
		title = `${id} · ${kindLabel} · attention ${info.score.toFixed(4)}`;
	} else {
		color = entityHex(kind, id);
		title = `${id} · ${kindLabel}${info ? "" : " · no attention on this step"}`;
	}
	const span = el("span", {
		class: "stx-ent", title: `${title} — hover to locate in the tree / 3D`, text: id,
		style: `border-bottom:2px solid ${color};background:${_alpha(color, 0.14)}`,
		onclick: () => { state.viewer?.select?.(id, { frame: true }); state.viewer?.setAttnHighlight?.([{ id, weight: 1 }]); },
	});
	// Join the shared cross-highlight index so hovering an output/reasoning entity
	// glows its tree badge + plan phrase and lights it in 3D (and vice-versa).
	treeHoverRegister(id, span, true);
	return span;
}

// Pretty-print the step output for readability (per whatever JSON shape the step
// kind emits), then highlight entities/attributes over it. Non-JSON falls back to raw.
function _fmtOutputNodes(text, ctx) {
	let s = String(text || "");
	try { s = JSON.stringify(JSON.parse(s), null, 2); } catch { /* not JSON → keep raw */ }
	return _highlightNodes(s, ctx);
}

// A collapsible text box for the step-text popup. `content` is a raw string (verbatim
// view) OR an array of DOM nodes (formatted highlight). Clamped to a few lines with a
// bottom fade + "show all" toggle; the toggle/fade hide when the text already fits.
function _clampedTextSection(label, content) {
	const isStr = typeof content === "string";
	const nodes = isStr ? null : (Array.isArray(content) ? content : [content]);
	const empty = isStr ? !content.trim() : !(nodes && nodes.length);
	const pre = el("pre", {});
	if (empty) pre.textContent = "(none)";
	else if (isStr) pre.textContent = content;
	else for (const n of nodes) pre.appendChild(n);
	const fade = el("div", { class: "stx-fade" });
	const body = el("div", { class: "stx-body" }, pre, fade);
	const toggle = el("button", { class: "stx-toggle", type: "button", text: "show all" });
	toggle.onclick = () => {
		const on = body.classList.toggle("expanded");
		fade.style.display = on ? "none" : "";
		toggle.textContent = on ? "clamp" : "show all";
	};
	requestAnimationFrame(() => {
		if (empty || body.scrollHeight <= body.clientHeight + 1) { toggle.style.display = "none"; fade.style.display = "none"; }
	});
	const chars = empty ? 0 : pre.textContent.length;
	return el("div", { class: "stx-section" },
		el("div", { class: "stx-sec-head" },
			el("span", { class: "stx-sec-label", text: label }),
			el("span", { class: "stx-sec-meta", text: chars ? `${chars.toLocaleString()} chars` : "empty" }),
			toggle),
		body);
}

// Make `panel` draggable by `handle` (its header). Switches to absolute top/left on
// first grab, clamps so the header stays on-screen, and remembers the final spot.
function _dragStepPanel(panel, handle) {
	handle.addEventListener("pointerdown", (e) => {
		if (e.target.closest("button")) return;  // header buttons aren't drag handles
		const r = panel.getBoundingClientRect();
		panel.style.left = `${r.left}px`; panel.style.top = `${r.top}px`;
		panel.style.right = "auto"; panel.style.bottom = "auto";
		const sx = e.clientX, sy = e.clientY, ox = r.left, oy = r.top;
		const move = (ev) => {
			const w = panel.offsetWidth;
			const nx = Math.min(window.innerWidth - 80, Math.max(80 - w, ox + (ev.clientX - sx)));
			const ny = Math.min(window.innerHeight - 36, Math.max(0, oy + (ev.clientY - sy)));
			panel.style.left = `${nx}px`; panel.style.top = `${ny}px`;
		};
		const up = () => {
			window.removeEventListener("pointermove", move);
			window.removeEventListener("pointerup", up);
			stepPanelPos = { left: panel.style.left, top: panel.style.top };
		};
		window.addEventListener("pointermove", move);
		window.addEventListener("pointerup", up);
		e.preventDefault();
	});
}

// Draggable, NON-modal window showing one step's exact text (input = system+user
// prompt, reasoning, output), each clamped initially. The page stays fully usable
// behind it (no backdrop) and it defaults to the bottom-left. Sourced from the
// per-step teacher-forcing export (`tfExport`), so it works for ANY step regardless
// of whether attention is computed.
function openStepTextPanel(rec) {
	const ev = rec.step.event_index;
	const name = rec.step.template ?? rec.step.step ?? "?";
	const node = rec.step.node ? ` · ${rec.step.node}` : "";
	document.getElementById("tf-step-text-panel")?.remove();  // single instance

	const scroll = el("div", { class: "stx-scroll" }, el("div", { class: "stx-loading", text: "loading step text…" }));
	const onKey = (e) => { if (e.key === "Escape") close(); };
	function close() { panel.remove(); document.removeEventListener("keydown", onKey); }
	const fmtBtn = el("button", {
		class: `stx-fmt${stepPanelFmt ? " on" : ""}`, type: "button", style: "margin-left:auto",
		title: "toggle formatting — highlight scene entities (attention-colored) + attributes, pretty-print the output",
		text: "✦ format",
		onclick: () => {
			stepPanelFmt = !stepPanelFmt;
			try { localStorage.setItem("tf-steptext-fmt", stepPanelFmt ? "1" : "0"); } catch { /* private mode */ }
			fmtBtn.classList.toggle("on", stepPanelFmt);
			render();
		},
	});
	const head = el("div", { class: "stx-head" },
		el("span", { class: "stx-grip", title: "drag to move", text: "⠿" }),
		el("span", { class: "stx-title", text: `step #${rec.idx + 1}` }),
		el("span", { class: "stx-sub", text: `${name}${node}` }),
		fmtBtn,
		el("button", { class: "stx-close", type: "button", style: "margin-left:6px", title: "close (Esc)", text: "×", onclick: close }));
	const panel = el("div", { id: "tf-step-text-panel", class: "stx-float" }, head, scroll);
	if (stepPanelPos) { panel.style.left = stepPanelPos.left; panel.style.top = stepPanelPos.top; panel.style.right = "auto"; panel.style.bottom = "auto"; }
	document.body.appendChild(panel);
	_dragStepPanel(panel, head);
	document.addEventListener("keydown", onKey);

	let data = null;  // { text, ctx } — fetched once, re-rendered on toggle
	function render() {
		if (!data) return;
		const { text: t, ctx } = data;
		const input = [t.system, t.user].filter((x) => x && String(x).trim()).join("\n\n────────\n\n");
		const kids = [];
		if (stepPanelFmt) kids.push(el("div", { class: "stx-hint", text: "entities underlined in their scene-tree attention color · attributes colored · output pretty-printed" }));
		kids.push(
			_clampedTextSection("input", stepPanelFmt ? _highlightNodes(input, ctx) : input),
			_clampedTextSection("reasoning", stepPanelFmt ? _highlightNodes(t.reasoning || "", ctx) : (t.reasoning || "")),
			_clampedTextSection("output", stepPanelFmt ? _fmtOutputNodes(t.output || "", ctx) : (t.output || "")));
		scroll.replaceChildren(...kids);
	}

	// Fetch the step's reconstructed text AND its attention (best-effort — attention
	// may be uncomputed; then entities fall back to flat kind colors). Build the
	// entity → attention map once so toggling format is instant.
	Promise.all([
		Promise.resolve(api.tfExport(state.run, state.slot, state.model, ev)),
		Promise.resolve(api.attentionGet(state.run, state.slot, state.model, ev, { view: "agg" })).catch(() => null),
	]).then(([exp, a]) => {
		if (!document.body.contains(panel)) return;  // closed before it landed
		const ent = new Map(); let max = 0;
		const push = (list) => { for (const e of list || []) { const sc = e.score || 0; if (!ent.has(e.id) || sc > ent.get(e.id).score) ent.set(e.id, { score: sc, kind: e.kind }); if (sc > max) max = sc; } };
		if (a && a.agg) { push(a.agg.scene && a.agg.scene.entityTotals); push(a.agg.to_place && a.agg.to_place.entityTotals); }
		const kindOf = new Map();
		for (const e of exp?.scene_map || []) kindOf.set(e.id, e.kind);
		for (const e of exp?.to_place_map || []) if (!kindOf.has(e.id)) kindOf.set(e.id, "object");
		for (const e of exp?.output_map || []) if (!kindOf.has(e.id)) kindOf.set(e.id, "object");
		const entities = new Set([...kindOf.keys(), ...ent.keys()]);
		data = { text: (exp && exp.text) || {}, ctx: { ent, max, entities, kindOf } };
		render();
	}).catch((e) => {
		if (!document.body.contains(panel)) return;
		scroll.replaceChildren(el("div", { class: "stx-loading", text: `couldn't load step text: ${e && e.message ? e.message : e}` }));
	});
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

// --- Step-story zone plan (hero narrative + attention-colored entity phrases) --
// The active zone the current step operates in: its own node when that's a zone,
// else the region that emitted it (a bbox step can run on a peer object).
function stepTargetZone(step) {
	if (!step || !step.node) return null;
	const obs = state.obs;
	const n = obs?.nodes?.get(step.node);
	if (n && n.kind !== "zone") { const r = emittingRegion(obs, step.node); if (r) return r; }
	return step.node;
}

// Candidate phrases for an entity id: its distilled noun phrase (imagePrompt) and
// its id slug turned back into words (dropping a trailing numeric suffix) — the
// forms most likely to appear verbatim in authored plan prose.
function planEntityPhrases(id) {
	const out = [];
	const n = state.obs?.nodes?.get(id);
	if (n?.imagePrompt) out.push(String(n.imagePrompt));
	const words = String(id).replace(/[_-]+/g, " ").trim().split(/\s+/).filter((w) => w && !/^\d+$/.test(w));
	if (words.length) out.push(words.join(" "));
	return out;
}
const _PLAN_STOP = new Set(["zone", "area", "region", "space", "room", "level", "floor", "wall", "ceiling", "side", "center", "edge", "object", "frame", "root", "the", "and", "with"]);
const _normPhrase = (p) => String(p).toLowerCase().replace(/\s+/g, " ").trim();
const _escapeRePlan = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
// Whole-word, whitespace- and plural-flexible regex for a normalized phrase.
function _planPhraseRe(norm) {
	const words = norm.split(" ").map(_escapeRePlan);
	const last = words.length - 1;
	return `\\b${words.map((w, i) => (i === last ? `${w}s?` : w)).join("\\s+")}\\b`;
}
// Per-step entity → attention context used to color the plan phrases.
function planCtx(agg) {
	const ent = new Map(); let max = 0;
	for (const e of (agg?.entityTotals || [])) { const sc = e.score || 0; ent.set(e.id, { score: sc, kind: e.kind }); if (sc > max) max = sc; }
	return { ent, max };
}

// Render `text` into DOM nodes, wrapping phrases that name a scene entity in a
// span whose underline color scales with the attention that entity drew this step
// (heatColor of score / peak). Longest phrases win; matches never overlap. Each
// span registers for the tree <-> plan hover cross-highlight.
function planHighlightNodes(text, ctx) {
	const s = String(text || "");
	if (!s) return [document.createTextNode("")];
	const seen = new Set(), cands = [];
	for (const [id, info] of ctx.ent) {
		for (const p of planEntityPhrases(id)) {
			const norm = _normPhrase(p);
			if (norm.length < 4) continue;
			if (!norm.includes(" ") && _PLAN_STOP.has(norm)) continue; // skip generic single words
			const key = `${id}::${norm}`;
			if (seen.has(key)) continue; seen.add(key);
			cands.push({ norm, id, info });
		}
	}
	if (!cands.length) return [document.createTextNode(s)];
	cands.sort((a, b) => b.norm.length - a.norm.length); // most specific first
	const claims = [];
	const overlaps = (a, b) => claims.some((c) => a < c.end && b > c.start);
	for (const c of cands) {
		let re; try { re = new RegExp(_planPhraseRe(c.norm), "gi"); } catch { continue; }
		let m;
		while ((m = re.exec(s))) {
			if (m.index === re.lastIndex) { re.lastIndex++; continue; }
			const a = m.index, b = a + m[0].length;
			if (!overlaps(a, b)) claims.push({ start: a, end: b, id: c.id, score: c.info.score || 0, kind: c.info.kind });
		}
	}
	if (!claims.length) return [document.createTextNode(s)];
	claims.sort((a, b) => a.start - b.start);
	const out = []; let last = 0;
	for (const c of claims) {
		if (c.start < last) continue;
		if (c.start > last) out.push(document.createTextNode(s.slice(last, c.start)));
		out.push(planEntitySpan(s.slice(c.start, c.end), c, ctx));
		last = c.end;
	}
	if (last < s.length) out.push(document.createTextNode(s.slice(last)));
	return out;
}

function planEntitySpan(text, claim, ctx) {
	const color = ctx.max > 0 ? heatColor(claim.score / ctx.max) : entityHex(claim.kind, claim.id);
	const span = el("span", {
		class: "plan-ent", "data-entity-id": claim.id,
		title: `${claim.id} · attention ${(claim.score || 0).toFixed(4)} — hover to locate in the tree / 3D`,
		style: `border-bottom:2px solid ${color};background:${_alpha(color, 0.16)}`,
		text,
		onclick: () => { state.viewer?.select?.(claim.id, { frame: true }); state.viewer?.setAttnHighlight?.([{ id: claim.id, weight: 1 }]); },
	});
	treeHoverRegister(claim.id, span, true);
	return span;
}

// Clamp long prose to a few lines with a show more / less toggle.
function clampProse(nodes) {
	const clampEl = el("div", { class: "hero-plan-clamp" }, ...nodes);
	const toggle = el("button", { class: "hero-plan-toggle", type: "button", text: "show more" });
	toggle.onclick = () => { toggle.textContent = clampEl.classList.toggle("expanded") ? "show less" : "show more"; };
	requestAnimationFrame(() => { if (clampEl.scrollHeight <= clampEl.clientHeight + 2) toggle.style.display = "none"; });
	return el("div", { class: "hero-plan" }, clampEl, toggle);
}

// The active zone's plan (its authored prompt when THIS step is the zone-plan
// step), with scene-entity phrases highlighted by attention. Null when there's no
// plan/prompt text for the zone.
function stepPlanBlock(step, agg, targetZone) {
	if (!targetZone) return null;
	const node = state.obs?.nodes?.get(targetZone);
	if (!node) return null;
	const isPlanStep = /(^|_)zone_plan/.test(step?.template ?? step?.step ?? "");
	const text = isPlanStep ? (node.prompt || node.plan || "") : (node.plan || node.prompt || "");
	if (!text || !String(text).trim()) return null;
	const nodes = planHighlightNodes(String(text), planCtx(agg));
	const label = isPlanStep ? "zone prompt" : "zone plan";
	return el("div", { class: "hero-plan-wrap" },
		el("div", { class: "hero-plan-lab" },
			el("span", { class: "map-sw", style: `background:${entityHex(node.kind || "zone", targetZone)}` }),
			el("span", { text: ` ${label} · ${targetZone}` })),
		clampProse(nodes));
}

// entity → attention + kind context for the id-based highlighter (_highlightNodes
// / _fmtOutputNodes) — mirrors how openStepTextPanel colors output/reasoning.
function stepTextCtx() {
	const a = state.attn, exp = state.export;
	const ent = new Map(); let max = 0;
	const push = (list) => { for (const e of list || []) { const sc = e.score || 0; if (!ent.has(e.id) || sc > ent.get(e.id).score) ent.set(e.id, { score: sc, kind: e.kind }); if (sc > max) max = sc; } };
	if (a && a.agg) { push(a.agg.scene && a.agg.scene.entityTotals); push(a.agg.to_place && a.agg.to_place.entityTotals); }
	const kindOf = new Map();
	for (const e of exp?.scene_map || []) kindOf.set(e.id, e.kind);
	for (const e of exp?.to_place_map || []) if (!kindOf.has(e.id)) kindOf.set(e.id, "object");
	for (const e of exp?.output_map || []) if (!kindOf.has(e.id)) kindOf.set(e.id, "object");
	return { ent, max, entities: new Set([...kindOf.keys(), ...ent.keys()]), kindOf };
}

// This step's OUTPUT (emitted JSON), pretty-printed with scene ids + attribute
// keys colored by attention (reuses the step-text popup's formatter). Clamped.
function stepOutputBlock() {
	const out = state.export?.text?.output;
	if (out == null || !String(out).trim()) return null;
	return _clampedTextSection("output", _fmtOutputNodes(String(out), stepTextCtx()));
}

// This step's REASONING nodes (entity-highlighted), or null when none captured.
function stepReasoningNodes() {
	const r = state.export?.text?.reasoning;
	if (r == null || !String(r).trim()) return null;
	return _highlightNodes(String(r), stepTextCtx());
}

// A short breadcrumb for the detached windows' subtitle.
function stepSubLabel() {
	const step = state.steps[state.stepIdx];
	return step ? `${step.template ?? step.step ?? "?"} · step ${state.stepIdx + 1}/${state.steps.length}` : "";
}

// --- detachable text windows (drag-out zone plan + output; reasoning popup) ----
// Body-mounted .stx-float windows (single instance per key) rebuilt from CURRENT
// state, so they refresh in place on step navigation (see refreshStoryWindows).
const _storyWins = new Map(); // key -> { win, bodyHost, subEl, build }
let _storyZ = 80;

function openStoryWindow(key, title, build) {
	const existing = _storyWins.get(key);
	if (existing) { // already open → refresh + raise
		existing.build = build;
		existing.bodyHost.replaceChildren(build());
		existing.subEl.textContent = stepSubLabel();
		existing.win.style.zIndex = String(++_storyZ);
		return;
	}
	const bodyHost = el("div", { class: "stx-scroll" }, build());
	const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } };
	const close = () => { win.remove(); document.removeEventListener("keydown", onKey, true); _storyWins.delete(key); };
	const subEl = el("span", { class: "stx-sub", text: stepSubLabel() });
	const head = el("div", { class: "stx-head" },
		el("span", { class: "stx-grip", title: "drag to move", text: "⠿" }),
		el("span", { class: "stx-title", text: title }), subEl, el("span", { style: "flex:1" }),
		el("button", { class: "stx-close", title: "close (Esc)", text: "×", onclick: close }));
	const win = el("div", { class: "stx-float story-win", id: `tf-story-win-${key}` }, head, bodyHost);
	const off = _storyWins.size * 26;
	win.style.right = `${24 + off}px`; win.style.top = `${96 + off}px`; win.style.left = "auto"; win.style.bottom = "auto";
	win.style.zIndex = String(++_storyZ);
	document.body.appendChild(win);
	_dragStepPanel(win, head);
	document.addEventListener("keydown", onKey, true);
	_storyWins.set(key, { win, bodyHost, subEl, build });
}
function refreshStoryWindows() {
	for (const [, e] of _storyWins) { try { e.bodyHost.replaceChildren(e.build()); e.subEl.textContent = stepSubLabel(); } catch { /* ignore */ } }
}
export function closeStoryWindows() { for (const [, e] of _storyWins) e.win.remove(); _storyWins.clear(); }

// Window bodies — recomputed from current state on every (re)render.
function planOutBody() {
	const step = state.steps[state.stepIdx], a = state.attn;
	const plan = (step && a) ? stepPlanBlock(step, aggregateAttn(a), stepTargetZone(step)) : null;
	return el("div", {},
		plan || el("div", { class: "muted", text: "no zone plan" }),
		el("div", { style: "margin-top:10px" }, stepOutputBlock() || el("div", { class: "muted", text: "no output" })));
}
function reasoningBody() {
	const nodes = stepReasoningNodes();
	if (!nodes) return el("div", { class: "stx-loading", text: "no reasoning captured for this step" });
	const pre = el("pre", {}); for (const n of nodes) pre.appendChild(n);
	return el("div", { class: "stx-section" }, el("div", { class: "stx-body expanded" }, pre));
}

// Zone-plan/output toolbar: pop the plan + output out into a draggable window, and
// open the (longer) reasoning as its own popup.
function stepStoryToolbar() {
	return el("div", { class: "hero-plan-tools" },
		el("button", { class: "hero-plan-btn", title: "pop the zone plan + output out into a draggable window", text: "⤢ pop out",
			onclick: () => openStoryWindow("planout", "zone plan · output", planOutBody) }),
		el("button", { class: "hero-plan-btn", title: "show this step's reasoning in a draggable popup", text: "🧠 reasoning",
			onclick: () => openStoryWindow("reason", "reasoning", reasoningBody) }));
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
	const step = state.steps[state.stepIdx];
	// The zone the step operates in — glowed in the tree, and its plan narrated in
	// the hero directly to the RIGHT of the trajectory graph (which caps at ~600px,
	// leaving room), with attention-colored entity phrases.
	const targetZone = stepTargetZone(step);
	const planBlock = stepPlanBlock(step, agg, targetZone);
	const outputBlock = stepOutputBlock(); // this step's emitted JSON, under the plan
	const storyCol = (planBlock || outputBlock)
		? el("div", { class: "hero-story-plan" }, stepStoryToolbar(), planBlock, outputBlock)
		: null;
	const storyRow = el("div", { class: "hero-story-row" },
		el("div", { class: "hero-story-graph" }, summaryTrajectory(a, agg)),
		storyCol);
	const hero = stepCmp.active
		? reportHero("Step comparison", `${stepCmp.peers.length} selected steps`, graphs.comparePeerAttributes(stepCmp, "attribute profile"))
		: reportHero(hasStepSelection ? "Selected step" : "Step story", `${a.tokens.length} tokens · ${(a.selected_heads || []).length} heads`, storyRow);
	const side = selectionDrawer(selectionSummary());
	const entityFocus = stepCmp.active
		? graphs.comparePeerEntities(stepCmp, "selected step entities")
		: reportCard("entity focus", "top attended scene objects", summaryEntities(agg), summaryKind(agg));
	const treeCard = (opts = {}) => reportCard("attention tree",
		`scene structure · badge ${opts.heatOnly ? "color" : "width"} = this step's attention · hover for full name · click → focus`,
		attentionTree([{ step, a, agg }], { targetZone, ...opts }));
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
	if (!rows.length) return workspaceShell("kind", choices, reportHero("Step kind", null, reportEmpty(`loading ${computed.length} step analyses…`)), selectionDrawer(selectionSummary()), [], null, { done: 0, total: computed.length, label: `loading ${computed.length} step analyses…` });
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
	return workspaceShell("kind", choices, hero, side, below, kindCmp.active ? null : useRows,
		missing > 0 ? { done: rows.length, total: computed.length, label: `loading ${rows.length}/${computed.length} step analyses` } : (kindCmp.loading ? true : null));
}

const _swatch = (color, title) => el("span", { class: "abl-sw", title, style: `background:${color}` });
const _treatSort = (a, b) => (METHOD_ORDER.indexOf(a.method) - METHOD_ORDER.indexOf(b.method)) || (a.xml === b.xml ? 0 : a.xml ? -1 : 1);

// The ablation COMPARISON BUILDER (replaces arbitrary run-picking). You choose a
// FACTOR (the axis compared) + a SCOPE (kind); the peers are the factor's levels,
// auto-aggregated across steps — shown as a transparent readout. A collapsible
// manual drill-down (kind → firing → treatment) can narrow to hand-picked variants.
function ablationSelectionDrawer(cat, factor, cmp) {
	const cur = String(state.run);
	const items = ablationRunItems.length ? ablationRunItems : [{ name: cur, ablation: null }];
	const eligible = items.filter((it) => it.name !== cur && it.name.includes("__abl-") && _ablComputed.has(it.name));
	const nUncomputed = items.filter((it) => it.name !== cur && it.name.includes("__abl-")).length - eligible.length;
	const kindsAll = [...new Set(eligible.map((it) => ablMeta(it.name)?.kind).filter(Boolean))].sort();
	if (ablScope !== "all" && !kindsAll.includes(ablScope)) ablScope = "all";

	// --- factor (compare axis) ---
	const factorFixed = cat === "xml" || cat === "ordering";
	const factorBtns = el("div", { class: "graph-toolbar", style: "gap:6px" },
		el("span", { class: "muted", style: "font-size:11px;align-self:center", text: "compare:" }),
		...["method", "xml"].map((f) => el("button", {
			text: f, class: factor === f ? "on" : "",
			...(factorFixed ? { disabled: "" } : {}),
			title: factorFixed ? `fixed to “${factor}” by the ${cat === "xml" ? "XML-existence" : "scene-ordering"} view` : `compare ablation ${f} levels`,
			onclick: factorFixed ? undefined : () => { ablFactor = f; renderReportWorkspace({ scrollSelection: false }); },
		})));

	// --- scope (kind) ---
	const scopeSel = el("select", { style: "font-size:11px;padding:2px 6px;min-width:120px", title: "aggregate across every kind, or focus one",
		onchange: (e) => { ablScope = e.target.value; renderReportWorkspace({ scrollSelection: false }); } },
		el("option", { value: "all", text: `all kinds (${kindsAll.length})`, ...(ablScope === "all" ? { selected: "" } : {}) }),
		...kindsAll.map((k) => el("option", { value: k, text: k, ...(ablScope === k ? { selected: "" } : {}) })));
	const scopeRow = el("div", { class: "graph-toolbar" }, el("span", { class: "muted", style: "font-size:11px;align-self:center", text: "scope:" }), scopeSel);

	// --- auto-aggregated groups readout (what's being compared) ---
	const groups = cmp.peers.length
		? el("div", { class: "abl-groups" }, ...cmp.peers.map((p) => el("div", { class: "abl-group", title: (p.names || []).join("\n") },
			_swatch(p.color), el("span", { class: "abl-group-name", text: p.key }), el("span", { class: "abl-group-n", text: `${p.n || p.rows.length}` }))))
		: el("div", { class: "muted", style: "font-size:11px;padding:4px 2px", text: _ablComputedLoading ? "scanning…" : "no groups in scope" });
	const groupsWrap = el("div", {}, el("div", { class: "sel-drawer-title", style: "margin:9px 0 4px", text: `${factor} groups · aggregated across steps${ablManual && (state.pins.runs || []).length ? " · narrowed to pins" : ""}` }), groups);

	// --- color key (factor-aware) ---
	const colorKey = factor === "xml"
		? el("div", { class: "abl-colorkey" }, el("span", { class: "muted", text: "xml:" }),
			el("span", { class: "abl-key" }, _swatch(XML_LEVEL_COLOR["xml \u2713"]), "on"), el("span", { class: "abl-key" }, _swatch(XML_LEVEL_COLOR["xml \u2717"]), "off"))
		: el("div", { class: "abl-colorkey" }, el("span", { class: "muted", text: "method:" }),
			...METHOD_ORDER.filter((m) => m !== "attend").map((m) => el("span", { class: "abl-key" }, _swatch(ORDER_METHOD_COLORS[m] || "#7aa2f7"), m)));

	// --- collapsible manual drill-down (kind → firing → treatment) ---
	const manualSection = ablManualDrillDown(cur, eligible);

	const head = el("div", { class: "sel-drawer-head" },
		el("span", { class: "sel-drawer-title", text: `comparison · ${eligible.length} computed` }),
		el("button", { class: "abl-quick", title: "re-scan which variants have computed attention", onclick: () => { ensureAblationComputed(true); } }, _ablComputedLoading ? "…" : "↻"));

	return el("div", { class: "rep-side-inner" },
		head, factorBtns, scopeRow, groupsWrap, colorKey,
		nUncomputed ? el("div", { class: "muted", style: "font-size:10.5px;margin-top:6px", text: `${nUncomputed} not yet computed (⚗ tab) — excluded` }) : null,
		el("div", { class: "abl-sep" }),
		manualSection);
}

// The manual drill-down: collapsible kind → firing (@cut) → treatment tree that
// pins individual variants. When "use pins" is on, the comparison narrows to them
// (still auto-grouped by the current factor). Off by default — the auto groups
// are the primary path.
function ablManualDrillDown(cur, eligible) {
	const nPinned = (state.pins.runs || []).length;
	const head = el("button", { class: "abl-kind-row", title: "hand-pick specific variants / firings to narrow the comparison",
		onclick: () => { ablManualOpen = !ablManualOpen; renderReportWorkspace({ scrollSelection: false }); } },
		el("span", { class: "abl-chev", text: ablManualOpen ? "▾" : "▸" }),
		el("span", { class: "abl-kind-name", text: "manual drill-down" }),
		el("span", { class: "abl-kind-meta", text: nPinned ? `${nPinned} pinned` : "optional" }));
	if (!ablManualOpen) return el("div", { class: "abl-kind" }, head);

	const usePins = el("label", { class: "abl-chk", style: "margin:4px 2px 6px", title: "when on, only the pinned variants below drive the comparison (still auto-grouped by the factor)" },
		el("input", { type: "checkbox", ...(ablManual ? { checked: "" } : {}), onchange: (e) => { ablManual = e.target.checked; renderReportWorkspace({ scrollSelection: false }); } }),
		` use pins to narrow${nPinned ? ` (${nPinned})` : ""}`);

	// kind -> Map(cut -> [{name, method, xml, kind, cut}])
	const kinds = new Map();
	for (const it of eligible) {
		const f = ablationFamilyOf(it);
		const t = (it.ablation && it.ablation.treatment) || {};
		const cuts = kinds.get(f.kind || "?") || kinds.set(f.kind || "?", new Map()).get(f.kind || "?");
		const cut = String(f.cut);
		(cuts.get(cut) || cuts.set(cut, []).get(cut)).push({ name: it.name, method: t.shuffle_method || "order", xml: t.xml_tags !== false, kind: f.kind, cut, rep: f.rep });
	}
	const kindList = [...kinds.keys()].sort();
	const kindEls = kindList.map((kind) => {
		const cuts = kinds.get(kind);
		const allNames = [...cuts.values()].flat().map((r) => r.name);
		const np = allNames.filter((n) => isPinned("runs", n)).length;
		const open = ablExpandKind.has(kind);
		const kh = el("button", { class: `abl-kind-row${np ? " has-sel" : ""}`, title: `${kind} · ${allNames.length}`,
			onclick: () => { open ? ablExpandKind.delete(kind) : ablExpandKind.add(kind); renderReportWorkspace({ scrollSelection: false }); } },
			el("span", { class: "abl-chev", text: open ? "▾" : "▸" }),
			el("span", { class: "abl-kind-name", text: kind }),
			el("span", { class: "abl-kind-meta", text: np ? `${np}/${allNames.length}` : String(allNames.length) }));
		if (!open) return el("div", { class: "abl-kind" }, kh);
		const cutEls = [...cuts.entries()].sort((a, b) => Number(a[0]) - Number(b[0])).map(([cut, rows]) => {
			const trows = rows.sort(_treatSort).map((r) => el("div", { class: "abl-treat", title: r.name },
				selectionCheckbox("runs", r.name, isPinned("runs", r.name) ? "unpin" : "pin this variant"),
				_swatch(ablColor(r.kind, r.cut, r.method, r.xml)),
				el("span", { class: "sel-group-name", text: `${r.method}${r.xml ? "" : " · noxml"}${r.rep > 1 ? ` · r${r.rep}` : ""}` })));
			return el("div", { class: "abl-cut" }, el("div", { class: "abl-cut-row" }, el("span", { class: "abl-cut-name", text: `@${cut}` })), ...trows);
		});
		return el("div", { class: "abl-kind" }, kh, ...cutEls);
	});
	return el("div", { class: "abl-kind" }, head, usePins, el("div", { class: "sel-tree", style: "max-height:38vh" }, ...kindEls));
}

// name -> { method, xml, kind } from the fetched ablation variants' treatments.
function ablTreatmentByName() {
	const m = new Map();
	for (const it of ablationRunItems) {
		const a = it.ablation;
		if (a && a.treatment) m.set(it.name, { method: a.treatment.shuffle_method || "order", xml: a.treatment.xml_tags !== false, kind: a.target_step_kind || "" });
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
function renderAblationStats(variantSamples) {
	// Each variant is one replicate (its own step firing) — the auto-discovered
	// factorial sample. Grouping is marginal per factor (method / xml). We run BOTH
	// a DIFFERENCE test (one-way ANOVA: is any level different?) and an EQUIVALENCE
	// test (TOST: are two levels the SAME within a margin?) per attribute.
	const samples = variantSamples.map((v) => {
		const totals = overviewAggregate(v.rows).componentTotals || [];
		return { method: v.method, xml: v.xml, comps: new Map(totals.map((c) => [c.component, c.score])) };
	}).filter((s) => s.comps.size);
	if (samples.length < 3) {
		return reportCard("statistics — method & XML effects", "needs ≥3 computed variants in scope",
			reportEmpty("compute more variants (⚗ tab), or widen the scope — each computed variant is one replicate"));
	}
	const methods = [...new Set(samples.map((s) => s.method))];
	const comps = [...new Set(samples.flatMap((s) => [...s.comps.keys()]))];
	const baseline = methods.includes("order") ? "order" : methods[0];
	const nOn = samples.filter((s) => s.xml).length, nOff = samples.length - nOn;

	const rows = comps.map((comp) => {
		const val = (s) => s.comps.get(comp);
		const has = (s) => val(s) != null;
		const all = samples.filter(has).map(val);
		const grand = all.length ? mean(all) : 0;
		const margin = ablTostDelta * Math.abs(grand); // equivalence bound, per-attribute (± of its mean)
		const methodScores = methods.map((mth) => samples.filter((s) => s.method === mth && has(s)).map(val));
		const methodMean = new Map(methods.map((mth, i) => [mth, methodScores[i].length ? mean(methodScores[i]) : null]));
		const xmlOn = samples.filter((s) => s.xml && has(s)).map(val);
		const xmlOff = samples.filter((s) => !s.xml && has(s)).map(val);
		// two-way (method × xml) cells for the ISOLATED test — each main effect then
		// tested against the within-cell error, not the other factor's variance.
		const xmlLevels = [true, false];
		const cells = methods.map((mth) => xmlLevels.map((xv) => samples.filter((s) => s.method === mth && s.xml === xv && has(s)).map(val)));
		const tw = (methods.length >= 2 && xmlOn.length && xmlOff.length) ? twoWayAnova(cells) : { ok: false };
		const baseScores = methodScores[methods.indexOf(baseline)] || [];
		const methodEquiv = new Map(methods.filter((m) => m !== baseline)
			.map((m) => [m, tost(methodScores[methods.indexOf(m)] || [], baseScores, margin)]));
		return {
			comp, grand, margin, methodMean,
			aMethod: oneWayAnova(methodScores), aXml: oneWayAnova([xmlOn, xmlOff]), tw,
			xmlOnM: xmlOn.length ? mean(xmlOn) : null, xmlOffM: xmlOff.length ? mean(xmlOff) : null,
			tXml: tost(xmlOn, xmlOff, margin), methodEquiv,
		};
	});
	const byMethodP = [...rows].filter((r) => r.aMethod.ok).sort((a, b) => a.aMethod.p - b.aMethod.p);
	const byXmlP = [...rows].filter((r) => r.aXml.ok).sort((a, b) => a.aXml.p - b.aXml.p);
	const sigMethod = byMethodP.filter((r) => r.aMethod.p < 0.05).map((r) => r.comp);
	const sigXml = byXmlP.filter((r) => r.aXml.p < 0.05).map((r) => r.comp);
	const equivXml = rows.filter((r) => r.tXml.ok && r.tXml.equiv).map((r) => r.comp);
	const pctMargin = Math.round(ablTostDelta * 100);

	// ---- controls + summary ----
	const DELTAS = [["±10%", 0.1], ["±20%", 0.2], ["±35%", 0.35]];
	const marginRow = el("div", { class: "graph-toolbar" },
		el("span", { class: "muted", style: "font-size:11px;align-self:center;margin-right:2px", text: "TOST equivalence margin (± of each attribute's mean):" }),
		...DELTAS.map(([lbl, dlt]) => el("button", { text: lbl, class: ablTostDelta === dlt ? "on" : "",
			title: `two means count as “the same” if within ±${Math.round(dlt * 100)}% of the attribute's mean`,
			onclick: () => { ablTostDelta = dlt; renderReportWorkspace({ scrollSelection: false }); } })));
	const summary = el("div", { class: "anv-summary" },
		el("div", {}, el("b", { text: "scene order differs" }), ` (ANOVA p<0.05): ${sigMethod.length ? sigMethod.join(", ") : "—"}`),
		el("div", {}, el("b", { text: "XML on/off differs" }), ` (ANOVA p<0.05): ${sigXml.length ? sigXml.join(", ") : "—"}`),
		el("div", {}, el("b", { text: "XML on ≈ off" }), ` (TOST p<0.05, ±${pctMargin}%): ${equivXml.length ? equivXml.join(", ") : "—"}`));

	// ---- difference: one-way ANOVA table (ranked by p·method) ----
	const pCell = (a) => el("td", { class: `anv-p${a.ok && a.p < 0.05 ? " sig" : ""}`, title: a.ok ? `F(${a.dfB},${a.dfW})=${_fmtNum(a.F)} · η²=${_fmtNum(a.eta2)}` : "not enough replicates" }, a.ok ? _fmtP(a.p) : "n/a");
	const anovaHead = el("tr", {}, el("th", { text: "#" }), el("th", { text: "attribute" }),
		...methods.map((mth) => el("th", { title: `mean over ${mth} variants`, text: mth })),
		el("th", { text: "F" }), el("th", { text: "p·method" }), el("th", { text: "η²" }),
		el("th", { class: "anv-div", text: `xml✓ (${nOn})` }), el("th", { text: `xml✗ (${nOff})` }), el("th", { text: "p·xml" }));
	const anovaBody = byMethodP.map((r, i) => el("tr", {},
		el("td", { class: "anv-rank", text: String(i + 1) }),
		el("td", { class: "anv-attr", text: r.comp }),
		...methods.map((mth) => el("td", { text: _fmtNum(r.methodMean.get(mth)) })),
		el("td", { text: r.aMethod.ok ? _fmtNum(r.aMethod.F) : "—" }), pCell(r.aMethod),
		el("td", { text: r.aMethod.ok ? _fmtNum(r.aMethod.eta2) : "—" }),
		el("td", { class: "anv-div", text: _fmtNum(r.xmlOnM) }), el("td", { text: _fmtNum(r.xmlOffM) }), pCell(r.aXml)));
	const anovaTable = byMethodP.length
		? el("table", { class: "abl-anova" }, el("thead", {}, anovaHead), el("tbody", {}, ...anovaBody))
		: reportEmpty("not enough replicates per method to test differences");

	// ---- ranked by p-value (top movers per factor) ----
	const rankList = (list, pOf, effOf) => el("ol", { class: "anv-rank-list" },
		...list.slice(0, 10).map((r) => el("li", {},
			el("span", { class: "anv-rank-attr", text: r.comp }),
			el("span", { class: `anv-rank-p${pOf(r) < 0.05 ? " sig" : ""}`, text: _fmtP(pOf(r)) }),
			el("span", { class: "anv-rank-eff", text: `η² ${_fmtNum(effOf(r))}` }))));
	const rankings = el("div", { class: "anv-rank-cols" },
		el("div", {}, el("div", { class: "anv-rank-head", text: "most moved by scene order" }),
			byMethodP.length ? rankList(byMethodP, (r) => r.aMethod.p, (r) => r.aMethod.eta2) : reportEmpty("—")),
		el("div", {}, el("div", { class: "anv-rank-head", text: "most moved by XML" }),
			byXmlP.length ? rankList(byXmlP, (r) => r.aXml.p, (r) => r.aXml.eta2) : reportEmpty("—")));

	// ---- equivalence: TOST (xml on vs off), ranked most-equivalent first ----
	const tostRows = [...rows].filter((r) => r.tXml.ok).sort((a, b) => a.tXml.p - b.tXml.p);
	const verdict = (r) => r.tXml.equiv ? el("span", { class: "anv-badge equiv", text: "equivalent" })
		: (r.aXml.ok && r.aXml.p < 0.05) ? el("span", { class: "anv-badge diff", text: "different" })
			: el("span", { class: "anv-badge incon", text: "inconclusive" });
	const tostHead = el("tr", {}, el("th", { text: "#" }), el("th", { text: "attribute" }), el("th", { text: "xml✓" }), el("th", { text: "xml✗" }), el("th", { text: "Δ" }), el("th", { text: "±margin" }), el("th", { text: "TOST p" }), el("th", { text: "verdict" }));
	const tostBody = tostRows.map((r, i) => el("tr", {},
		el("td", { class: "anv-rank", text: String(i + 1) }),
		el("td", { class: "anv-attr", text: r.comp }),
		el("td", { text: _fmtNum(r.xmlOnM) }), el("td", { text: _fmtNum(r.xmlOffM) }),
		el("td", { text: _fmtNum(r.tXml.diff) }), el("td", { text: `±${_fmtNum(r.margin)}` }),
		el("td", { class: `anv-p${r.tXml.equiv ? " sig" : ""}`, text: _fmtP(r.tXml.p) }),
		el("td", {}, verdict(r))));
	const tostTable = tostRows.length
		? el("table", { class: "abl-anova" }, el("thead", {}, tostHead), el("tbody", {}, ...tostBody))
		: reportEmpty("need ≥2 variants on each side (xml on / off) to test equivalence");
	const methodEquivSummary = methods.filter((m) => m !== baseline).map((m) => {
		const eqN = rows.filter((r) => r.methodEquiv.get(m)?.equiv).length;
		const okN = rows.filter((r) => r.methodEquiv.get(m)?.ok).length;
		return el("div", {}, el("b", { text: `${m} ≈ ${baseline}` }), ` on ${eqN}/${okN || 0} attributes`);
	});

	// Isolated effects: two-way ANOVA (method × XML). Shows marginal (one-way,
	// pooled over the other factor) → isolated (two-way, other factor partitioned
	// out) p per effect, so the variance-removal is visible; plus the interaction.
	const twRows = rows.filter((r) => r.tw && r.tw.ok);
	const isoCell = (marg, iso) => el("td", { class: "anv-p", title: `marginal (pooled over the other factor) p=${_fmtP(marg)} → isolated (two-way) p=${_fmtP(iso)}` },
		el("span", { style: "opacity:.5", text: _fmtP(marg) }), el("span", { style: "opacity:.5", text: " → " }),
		el("span", { style: iso < 0.05 ? "color:var(--green,#8bd17c);font-weight:700" : "", text: _fmtP(iso) }));
	let twoWayCard;
	if (!twRows.length) {
		const why = methods.length < 2 ? "needs ≥2 shuffle methods in scope"
			: (samples.every((s) => s.xml) || samples.every((s) => !s.xml)) ? "needs BOTH xml on and off in scope"
				: "needs every method×xml cell filled with >1 variant — raise reps (n) on the board, or widen the scope so each cell has replicates";
		twoWayCard = reportCard("isolated effects · two-way ANOVA (method × XML)", "isolates each factor from the other's variance", reportEmpty(why));
	} else {
		const sorted = [...twRows].sort((a, b) => Math.min(a.tw.A.p, a.tw.B.p) - Math.min(b.tw.A.p, b.tw.B.p));
		const head = el("tr", {}, el("th", { text: "#" }), el("th", { text: "attribute" }),
			el("th", { text: "method  (marg → isolated)" }), el("th", { text: "xml  (marg → isolated)" }), el("th", { text: "method×xml" }), el("th", { text: "err df" }));
		const body = sorted.map((r, i) => el("tr", {},
			el("td", { class: "anv-rank", text: String(i + 1) }),
			el("td", { class: "anv-attr", text: r.comp }),
			isoCell(r.aMethod.ok ? r.aMethod.p : NaN, r.tw.A.p),
			isoCell(r.aXml.ok ? r.aXml.p : NaN, r.tw.B.p),
			el("td", { class: `anv-p${r.tw.AB.p < 0.05 ? " sig" : ""}`, title: `interaction η²=${_fmtNum(r.tw.AB.eta2)} — does XML's effect depend on the method?`, text: _fmtP(r.tw.AB.p) }),
			el("td", { style: "opacity:.6", text: String(r.tw.dfError) })));
		twoWayCard = reportCard("isolated effects · two-way ANOVA (method × XML)",
			"each main effect tested against the WITHIN-cell error, so the OTHER factor's variance is partitioned out (not left in the denominator). marginal → isolated shows the effect: isolated p is usually smaller when the other factor carried variance. method×xml = does XML's effect depend on the method?",
			el("div", { class: "anv-summary", style: "margin-bottom:8px" }, el("div", { text: `pooled within-cell error df ${twRows[0].tw.dfError} · ${methods.length} methods × 2 xml` })), body.length ? el("table", { class: "abl-anova" }, el("thead", {}, head), el("tbody", {}, ...body)) : null);
	}

	return el("div", { class: "abl-q-cards" },
		reportCard("statistics — method & XML effects",
			`${samples.length} variants · ${methods.length} methods · ANOVA (is it different?) + TOST (is it the same?) · each variant = one replicate`,
			marginRow, summary),
		reportCard("difference · one-way ANOVA (marginal)", "each factor pooled over the OTHER (the other factor's variance inflates the error → higher p). ranked by p·method · η² = variance explained", anovaTable),
		twoWayCard,
		reportCard("ranked by p-value", "the attributes each factor moves most (top 10)", rankings),
		reportCard("equivalence · TOST",
			`xml on vs off, ranked most-equivalent first · “equivalent” = both means statistically within ±${pctMargin}% of the attribute's mean`,
			tostTable,
			methodEquivSummary.length ? el("div", { class: "anv-summary", style: "margin-top:10px" }, el("div", { class: "anv-rank-head", text: `scene order ≈ ${baseline}? (TOST equivalence, per attribute)` }), ...methodEquivSummary) : null));
}

// ---- scene-ordering analysis ----------------------------------------------

// The scope (kind) filter is "all" (every scene-context kind — next_object,
// child_bbox_batch, *_decompose, …) or one specific kind. It's shared by the
// auto-comparison and the ordering graphs. Kind buttons are built from the kinds
// actually present, so it never silently drops a whole kind like next_object.
function ablKindMatch(tpl) {
	return ablScope === "all" ? hasSceneContext(tpl) : tpl === ablScope;
}

// tf-export scene_map → { entityId: normalized position } (0 = first … 1 = last),
// cached per (run, cell, event). Positions come from the PROMPT STRUCTURE, so
// they exist for any variant regardless of attention compute; a miss kicks a
// lazy fetch that re-renders on arrival.
function ensureOrder(run, ev) {
	const key = `${run}\u0000${state.slot}\u0000${state.model}\u0000${ev}`;
	const cached = orderCache.get(key);
	if (cached) return cached === "loading" ? null : cached;
	orderCache.set(key, "loading");
	Promise.resolve(api.tfExport(run, state.slot, state.model, ev)).then((exp) => {
		// Rank OBJECTS only (skip zone headers) so an object's normalized position
		// spreads 0→1 over the objects it competes with, not diluted by interleaved
		// zone markers.
		const sm = ((exp && exp.scene_map) || []).filter((e) => e.kind === "object");
		const denom = Math.max(1, sm.length - 1);
		orderCache.set(key, new Map(sm.map((e, i) => [e.id, i / denom])));
		if (state.reportView === "ablation") renderReportWorkspace({ scrollSelection: false });
	}).catch(() => { orderCache.set(key, new Map()); });
	return null;
}

// (position, attention) points across ALL rows in the comparison, peer-agnostic:
// every row is tagged with its source `_run` + `_method` (see tagRows), so this
// works whether the peers are aggregated factor levels or manually pinned runs.
// x = the OBJECT's position in the (shuffled) context (objects-only ranking, so it
// spreads 0→1 cleanly); y = attention to the object (attr=null) or to one of its
// attributes. Colored by shuffle method.
function ablOrderPoints(cmp, attr) {
	const raw = [];
	let loading = false;
	let kept = 0;
	let seen = 0;
	for (const p of cmp.peers) {
		for (const r of p.rows) {
			if (!ablKindMatch(r.template)) continue;
			const run = r._run || String(p.key);
			const method = r._method || "order";
			const color = ORDER_METHOD_COLORS[method] || "#7aa2f7";
			const omap = ensureOrder(run, r.step.event_index);
			if (!omap) { loading = true; continue; }
			// "Focus on objects that matter": an object is plotted only if its
			// attention SHARE of this step's whole context (its score / the step's
			// total scene attention) is ≥ r. r=0 shows everything.
			const total = r.agg.entityTotals.reduce((s, e) => s + (e.score || 0), 0) || 1;
			for (const e of r.agg.entityTotals) {
				if (entityKindLabel(e.kind, e.id) !== "object") continue;
				const pos = omap.get(e.id);
				if (pos == null) continue;
				seen += 1;
				if ((e.score || 0) / total < ablFocusR) continue;
				kept += 1;
				let y;
				if (attr) { const c = (e.components || []).find((x) => x.component === attr); if (!c) continue; y = c.score; }
				else y = e.score;
				raw.push({ x: pos, y, color, method, label: `${e.id} · ${method} · ${(100 * (e.score || 0) / total).toFixed(1)}% @${pos.toFixed(2)}` });
			}
		}
	}
	return { raw, loading, kept, seen };
}
function ablBinMean(raw, nb = 10) {
	const bins = Array.from({ length: nb }, () => []);
	for (const p of raw) bins[Math.min(nb - 1, Math.max(0, Math.floor(p.x * nb)))].push(p.y);
	const out = [];
	bins.forEach((vals, i) => { if (vals.length) out.push({ x: (i + 0.5) / nb, y: vals.reduce((a, b) => a + b, 0) / vals.length }); });
	return out;
}
function ablOrderLegend(raw) {
	const methods = [...new Set(raw.map((p) => p.method))];
	return methods.map((m) => ({ label: m, color: ORDER_METHOD_COLORS[m] || "#7aa2f7" })).concat([{ label: "bin mean", color: "#e8fcff" }]);
}

// Companion to the ordering scatter: sweep the focus threshold and, at each step,
// re-correlate position vs attention per shuffle method. Gathers every object's
// (position, y, share) ONCE (focus-agnostic), then filters by share per step — so
// the curve re-uses exactly the scatter's data, just trimming the low-attention
// tail as focus rises. Needs ≥ MIN_N objects to report ρ at a threshold; the sweep
// is ascending so once one step falls short every higher one does too (→ break).
const RHO_SWEEP = [0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03];
function ablOrderRhoCurve(cmp, attr) {
	const MIN_N = 5;
	const byMethod = new Map(); // method -> [{ pos, y, share }]
	let loading = false;
	for (const p of cmp.peers) for (const r of p.rows) {
		if (!ablKindMatch(r.template)) continue;
		const run = r._run || String(p.key);
		const method = r._method || "order";
		const omap = ensureOrder(run, r.step.event_index);
		if (!omap) { loading = true; continue; }
		const total = r.agg.entityTotals.reduce((s, e) => s + (e.score || 0), 0) || 1;
		for (const e of r.agg.entityTotals) {
			if (entityKindLabel(e.kind, e.id) !== "object") continue;
			const pos = omap.get(e.id);
			if (pos == null) continue;
			let y;
			if (attr) { const c = (e.components || []).find((x) => x.component === attr); if (!c) continue; y = c.score; }
			else y = e.score;
			const arr = byMethod.get(method) || byMethod.set(method, []).get(method);
			arr.push({ pos, y, share: (e.score || 0) / total });
		}
	}
	const series = [];
	for (const [method, pts] of byMethod) {
		const line = [];
		for (const r of RHO_SWEEP) {
			const kept = pts.filter((q) => q.share >= r);
			if (kept.length < MIN_N) break;
			line.push({ x: r, y: spearman(kept.map((q) => q.pos), kept.map((q) => q.y)), n: kept.length });
		}
		if (line.length) series.push({ method, color: ORDER_METHOD_COLORS[method] || "#7aa2f7", points: line });
	}
	return { series, loading };
}
function ablOrderAttributes(cmp) {
	const freq = new Map();
	for (const p of cmp.peers) for (const r of p.rows) {
		if (!ablKindMatch(r.template)) continue;
		for (const e of r.agg.entityTotals) for (const c of (e.components || [])) freq.set(c.component, (freq.get(c.component) || 0) + 1);
	}
	return [...freq.entries()].sort((a, b) => b[1] - a[1]).map(([k]) => k);
}

// XML existence: an "all kinds" spider first (xml on vs off, pooled across every
// kind), then ONE step kind chosen from a selector — instead of laying every kind
// out. Fed by per-variant samples ({ kind, xml, rows }).
function renderAblationXml(samples) {
	if (!samples.length) return reportCard("XML existence", "no computed variants in scope", reportEmpty("compute variants (xml on / off) in the ⚗ ablation tab, or widen the scope"));
	const onV = samples.filter((v) => v.xml), offV = samples.filter((v) => !v.xml);
	const allCard = graphs.xmlSplitCard("all kinds — attribute split by XML",
		`${onV.length} xml\u2713 · ${offV.length} xml\u2717 variants · pooled across every kind`,
		onV.flatMap((v) => v.rows), offV.flatMap((v) => v.rows));

	const kinds = [...new Set(samples.map((v) => v.kind))].sort();
	if (ablXmlKind && !kinds.includes(ablXmlKind)) ablXmlKind = null;
	if (!ablXmlKind && kinds.length) ablXmlKind = kinds[0];
	const kindSel = el("select", { style: "font-size:11px;padding:2px 6px;min-width:150px", title: "which step kind to break down",
		onchange: (e) => { ablXmlKind = e.target.value; renderReportWorkspace({ scrollSelection: false }); } },
		...kinds.map((k) => el("option", { value: k, text: k, ...(k === ablXmlKind ? { selected: "" } : {}) })));
	const kindRow = el("div", { class: "graph-toolbar" },
		el("span", { class: "muted", style: "font-size:11px;align-self:center", text: "break down by step kind:" }),
		kinds.length ? kindSel : el("span", { class: "muted", style: "font-size:11px", text: "—" }));

	const sel = samples.filter((v) => v.kind === ablXmlKind);
	const selOn = sel.filter((v) => v.xml), selOff = sel.filter((v) => !v.xml);
	const kindCard = ablXmlKind
		? graphs.xmlSplitCard(`${ablXmlKind} \u2014 attribute split by XML`, `${selOn.length} xml\u2713 · ${selOff.length} xml\u2717 variants`, selOn.flatMap((v) => v.rows), selOff.flatMap((v) => v.rows))
		: reportCard("by step kind", null, reportEmpty("no kinds in scope"));

	return el("div", { class: "abl-q-cards" }, allCard, kindRow, kindCard);
}

// Two xy-scatters answering "does the ordering of the scene context change how
// the model attends?": (1) attention to the object vs its position, (2) attention
// to a chosen ATTRIBUTE of the object vs its position. A focus (min attention
// share) + an attribute dropdown keep it uncrowded.
function renderAblationOrdering(cmp) {
	// Focus on objects that MATTER: only plot objects whose attention share of the
	// step's whole context is ≥ r (r=0 = every object). Denoises the long tail the
	// wait we should do lile 0.5 percent and 0.25 percnent
	const FOCUS = [
		["all", 0],
		["≥2%", 0.02],
		["≥0.5%", 0.005],
		["≥0.25%", 0.0025],
		["≥0.1%", 0.001],
	  ];
	const focusRow = el("div", { class: "graph-toolbar" },
		el("span", { class: "muted", style: "font-size:11px;align-self:center;margin-right:2px", text: "focus (min attention share):" }),
		...FOCUS.map(([lbl, r]) => el("button", { text: lbl, class: ablFocusR === r ? "on" : "",
			title: r ? `only objects with ≥ ${Math.round(r * 100)}% of the step's context attention` : "every object (no focus)",
			onclick: () => { ablFocusR = r; renderReportWorkspace({ scrollSelection: false }); } })),
		el("button", { class: ablLogY ? "on" : "", style: "margin-left:auto",
			title: "log-scale the attention (y) axis of the scatters AND the focus (x) axis of the ρ-vs-focus curves — both are heavy-tailed, so a few values dominate a linear axis",
			onclick: () => { ablLogY = !ablLogY; renderReportWorkspace({ scrollSelection: false }); } }, "log ↕ attention"));

	const g1 = ablOrderPoints(cmp, null);
	const focusNote = ablFocusR ? ` · focus ≥${Math.round(ablFocusR * 100)}% (${g1.kept}/${g1.seen} objects)` : "";
	const card1 = graphs.ablationOrderCard("attention to the object vs its position",
		g1.loading ? "loading context positions…" : `${g1.raw.length} object·steps · y = attention to the object${ablLogY ? " · log" : ""}${focusNote}`,
		g1.raw, ablBinMean(g1.raw), { legend: ablOrderLegend(g1.raw), logY: ablLogY });
	const rho1 = ablOrderRhoCurve(cmp, null);
	const rhoCard1 = graphs.rhoCurveCard("\u03c1 vs focus \u2014 attention to the object",
		rho1.loading ? "loading context positions…" : "Spearman \u03c1(position, attention) as the focus threshold rises \u2014 one line per scene order; dives from 0 ⇒ the objects that matter are the early ones",
		rho1.series, { xLabel: "focus  (min attention share)", markX: ablFocusR, logX: ablLogY });

	const attrs = ablOrderAttributes(cmp);
	if (ablOrderAttr && !attrs.includes(ablOrderAttr)) ablOrderAttr = null;
	if (!ablOrderAttr && attrs.length) ablOrderAttr = attrs[0];
	const attrSel = el("select", { style: "font-size:11px;padding:2px 6px;min-width:130px",
		title: "which attribute of the object to measure attention to",
		onchange: (e) => { ablOrderAttr = e.target.value; renderReportWorkspace({ scrollSelection: false }); } },
		...attrs.map((a) => el("option", { value: a, text: a, ...(a === ablOrderAttr ? { selected: "" } : {}) })));
	const attrRow = el("div", { class: "graph-toolbar" },
		el("span", { class: "muted", style: "font-size:11px;align-self:center", text: "attribute:" }),
		attrs.length ? attrSel : el("span", { class: "muted", style: "font-size:11px", text: "— none available —" }));
	let card2, rhoCard2 = null;
	if (!attrs.length) {
		// Distinguish "no rows match the scope" from "rows exist but carry no
		// per-object attribute split" (only the latter needs a recompute).
		const why = g1.loading ? "loading…"
			: g1.raw.length ? "these results predate the per-object attribute split — recompute the scoped variants (⚗ tab) to populate it"
				: "no positioned objects in scope — widen the scope on the left or compute variants of a scene-context kind";
		card2 = reportCard("attention to an attribute vs object position", "no per-object attribute breakdown available", reportEmpty(why));
	} else {
		const g2 = ablOrderPoints(cmp, ablOrderAttr);
		card2 = graphs.ablationOrderCard(`attention to \u201c${ablOrderAttr}\u201d vs object position`,
			g2.loading ? "loading context positions…" : `${g2.raw.length} object·steps · y = attention to the object's \u201c${ablOrderAttr}\u201d${ablLogY ? " · log" : ""}`,
			g2.raw, ablBinMean(g2.raw), { legend: ablOrderLegend(g2.raw), logY: ablLogY });
		const rho2 = ablOrderRhoCurve(cmp, ablOrderAttr);
		rhoCard2 = graphs.rhoCurveCard(`\u03c1 vs focus \u2014 attention to \u201c${ablOrderAttr}\u201d`,
			rho2.loading ? "loading context positions…" : `Spearman \u03c1(position, \u201c${ablOrderAttr}\u201d attention) as the focus threshold rises \u2014 one line per scene order`,
			rho2.series, { xLabel: "focus  (min attention share)", markX: ablFocusR, logX: ablLogY });
	}
	return el("div", { class: "abl-q-cards" }, focusRow, card1, rhoCard1, attrRow, card2, ...(rhoCard2 ? [rhoCard2] : []));
}

// ---- spatial-relevance analysis -------------------------------------------

// Kick the (cheap, GPU-free) geometry compute: per distinct firing (@cut) among
// the scoped variants, fetch the base scene's per-object distance + ray-trace
// visibility to that step's target. Cached; the loading bar tracks it.
function ablComputeSpatial() {
	if (ablSpatialLoading) return;
	ablSpatialComputed = true;
	ablSpatialLoading = true;
	renderReportWorkspace({ scrollSelection: false });
	const base = String(state.run), slot = state.slot, model = state.model;
	const cuts = [...new Set(ablScopedVariants().map((it) => ablMeta(it.name)?.cut).filter((c) => c != null))];
	Promise.resolve(pool(cuts, 4, async (cut) => {
		const key = `${base}\u0000${slot}\u0000${model}\u0000${cut}`;
		if (ablSpatialCache.has(key)) return;
		try {
			const r = await api.spatialRanks(base, slot, model, cut);
			ablSpatialCache.set(key, new Map((r.objects || []).map((o) => [o.id, o])));
		} catch { ablSpatialCache.set(key, new Map()); }
	})).then(() => { ablSpatialLoading = false; if (state.reportView === "ablation") renderReportWorkspace({ scrollSelection: false }); });
}

// Rank objects by closeness / visibility WITHIN a cut (variant-independent geometry),
// then emit ONE point per (variant, object): x/y = the cut's ranks, attn = the
// variant's attention, and the variant's type tags (kind / method / xml) so the
// scatter can be colored by any of them. Bigger circle = more attention.
function ablSpatialPoints(cmp) {
	const base = String(state.run), slot = state.slot, model = state.model;
	const rankCache = new Map(); // cut -> Map(objId -> {distRank, rayRank}) | null (geometry missing)
	const rankFor = (cut) => {
		if (rankCache.has(cut)) return rankCache.get(cut);
		const sp = ablSpatialCache.get(`${base}\u0000${slot}\u0000${model}\u0000${cut}`);
		if (!sp) { rankCache.set(cut, null); return null; }
		const objs = [...sp.entries()].map(([id, g]) => ({ id, distance: g.distance, rayHits: g.ray_hits || 0 }));
		const rank = new Map(objs.map((o) => [o.id, {}]));
		[...objs].sort((a, b) => a.distance - b.distance).forEach((o, i) => { rank.get(o.id).distRank = i + 1; });
		[...objs].sort((a, b) => b.rayHits - a.rayHits).forEach((o, i) => { rank.get(o.id).rayRank = i + 1; });
		rankCache.set(cut, rank);
		return rank;
	};
	const points = [];
	let missing = false;
	for (const p of cmp.peers) for (const r of p.rows) {
		const m = ablMeta(r._run);
		if (!m || m.cut == null) continue;
		const rank = rankFor(m.cut);
		if (!rank) { missing = true; continue; }
		const total = r.agg.entityTotals.reduce((s, e) => s + (e.score || 0), 0) || 1;
		for (const e of r.agg.entityTotals) {
			if (entityKindLabel(e.kind, e.id) !== "object") continue;
			const rk = rank.get(e.id);
			if (!rk || rk.distRank == null) continue;
			points.push({
				x: rk.distRank, y: rk.rayRank, attn: e.score || 0, share: (e.score || 0) / total,
				kind: m.kind, method: m.method, xml: m.xml,
				label: `${e.id} @${m.cut} · ${m.method}${m.xml ? "" : "·noxml"} · ${(100 * (e.score || 0) / total).toFixed(1)}% ctx · dist#${rk.distRank} vis#${rk.rayRank}`,
			});
		}
	}
	return { points, missing };
}

// Categorical legend for the spatial scatter, keyed on the chosen "color by" axis.
function ablSpatialCats(points, by) {
	if (by === "kind") {
		const ks = [...new Set(points.map((p) => p.kind).filter(Boolean))].sort();
		return ks.map((k, i) => ({ key: k, color: COMPARE_COLORS[i % COMPARE_COLORS.length], label: k }));
	}
	if (by === "xml") {
		return [{ key: "xml \u2713", color: "#6bd96e", label: "xml on" }, { key: "xml \u2717", color: "#ff6b9d", label: "xml off" }]
			.filter((c) => points.some((p) => p.cat === c.key));
	}
	const ms = [...new Set(points.map((p) => p.method).filter(Boolean))];
	return METHOD_ORDER.filter((m) => ms.includes(m)).concat(ms.filter((m) => !METHOD_ORDER.includes(m)))
		.map((m) => ({ key: m, color: ORDER_METHOD_COLORS[m] || "#7aa2f7", label: m }));
}

const SPATIAL_COLORBY = [["method", "scene order"], ["kind", "step kind"], ["xml", "xml"]];
function renderAblationSpatial(cmp) {
	const computeBtn = el("div", { class: "graph-toolbar" },
		el("button", { class: "abl-dr-btn primary", title: "cast a sphere of rays + measure distance from each step's target to every object (occlusion-aware geometry, no GPU)",
			onclick: ablComputeSpatial }, ablSpatialLoading ? "computing…" : (ablSpatialComputed ? "↻ recompute spatial ranks" : "⚡ compute spatial ranks")),
		el("span", { class: "muted", style: "font-size:11px;align-self:center;margin-left:auto;margin-right:2px", text: "color by:" }),
		...SPATIAL_COLORBY.map(([k, lbl]) => el("button", { text: lbl, class: ablSpatialColorBy === k ? "on" : "",
			title: `color each object·firing by its ${lbl}`,
			onclick: () => { ablSpatialColorBy = k; renderReportWorkspace({ scrollSelection: false }); } })),
		el("span", { style: "width:10px" }),
		el("button", { class: ablSpatial3D ? "on" : "", title: "3D view — floor = distance × visibility rank, height (z) = attention · drag to rotate, scroll to zoom, hover for details",
			onclick: () => { ablSpatial3D = !ablSpatial3D; renderReportWorkspace({ scrollSelection: false }); } }, ablSpatial3D ? "3D ✓" : "3D"),
		...(ablSpatial3D ? [
			el("button", { class: ablSpatialLogZ ? "on" : "", title: "log-scale the attention height — raw attention is heavy-tailed, so a few objects tower over the rest",
				onclick: () => { ablSpatialLogZ = !ablSpatialLogZ; renderReportWorkspace({ scrollSelection: false }); } }, "log ↕ height"),
			el("button", { class: ablSpatialCoM ? "on" : "", title: "show each category's attention-weighted center of mass — a haloed sphere at its mean (distance, visibility, attention)",
				onclick: () => { ablSpatialCoM = !ablSpatialCoM; renderReportWorkspace({ scrollSelection: false }); } }, "◎ centroids"),
			el("button", { class: "abl-dr-btn sm", title: "reset the camera to the default 3D angle", onclick: () => resetSpatial3DView() }, "⟳ reset view"),
		] : []));
	if (!ablSpatialComputed) {
		return el("div", { class: "abl-q-cards" }, computeBtn,
			reportCard("attention vs spatial relevance", "distance + ray-trace visibility to the target, vs attention",
				reportEmpty("click ⚡ compute — for each firing it casts rays + measures distance from the target node to every object (no GPU), then plots attention against closeness / visibility")));
	}
	const { points, missing } = ablSpatialPoints(cmp);
	for (const p of points) p.cat = ablSpatialColorBy === "kind" ? p.kind : ablSpatialColorBy === "xml" ? (p.xml ? "xml \u2713" : "xml \u2717") : p.method;
	const cats = ablSpatialCats(points, ablSpatialColorBy);
	const byLabel = SPATIAL_COLORBY.find((c) => c[0] === ablSpatialColorBy)?.[1] || ablSpatialColorBy;
	const busy = missing || ablSpatialLoading;
	const encoding = ablSpatial3D ? `height (z) = attention${ablSpatialLogZ ? " (log)" : ""} · floor = distance × visibility rank · drag to rotate · scroll to zoom` : "bigger + more opaque = more attention · bottom-left corner = close + visible to the target";
	const sub = `${points.length} object·firings · colored by ${byLabel} · ${encoding}${busy ? " · computing…" : ""}`;
	if (ablSpatial3D) {
		// Interactive WebGL viewport (three.js). It's an imperative canvas, so we
		// return a fixed-height mount + a legend, then (after this tree is in the DOM)
		// move the singleton viewport into it on the next frame — camera survives the
		// rebuild since the instance is reused.
		const mount = el("div", { id: "tf-abl-3d-mount", style: "width:100%;height:480px;position:relative;background:#0d0e12;border:1px solid var(--line);border-radius:10px;overflow:hidden" });
		const legend = el("div", { class: "attr-legend compact", style: "justify-content:center;margin-top:6px" },
			...cats.filter((c) => c.label).map((c) => el("span", { class: "pin-pill" },
				el("i", { style: `width:11px;height:11px;border-radius:50%;background:${c.color};display:inline-block;flex:none` }), el("span", { text: c.label }))),
			el("span", { class: "muted", style: "font-size:11px;align-self:center", text: "↕ height = attention" }));
		requestAnimationFrame(() => { const host = document.getElementById("tf-abl-3d-mount"); if (host) mountSpatial3D(host, points, { cats, logZ: ablSpatialLogZ, com: ablSpatialCoM }); });
		return el("div", { class: "abl-q-cards" }, computeBtn, reportCard("attention vs spatial relevance", sub, el("div", {}, mount, legend)));
	}
	return el("div", { class: "abl-q-cards" }, computeBtn, graphs.spatialScatterCard("attention vs spatial relevance", sub, points, { cats }));
}

// Ablation view: the comparison is AUTO-DISCOVERED (factor levels aggregated
// across steps), not hand-selected. The FACTOR is implied by the section
// (xml-existence → xml, scene-ordering → method) or a toggle for the generic
// compare. XML-existence + ANOVA use per-variant samples (they need replicate
// granularity); the rest use the aggregated factor-level peers.
function renderAblationWorkspace() {
	const choices = [
		{ id: "xml", label: "xml existence" }, { id: "ordering", label: "scene ordering" },
		{ id: "spatial", label: "spatial relevance" }, { id: "breakdown", label: "breakdown" },
		{ id: "attributes", label: "attributes" }, { id: "stats", label: "statistics" },
		{ id: "entities", label: "entities" }, { id: "heads", label: "heads" },
	];
	ensureAblationRuns();
	ensureAblationComputed();
	const cat = currentCategory("ablation", choices);
	const factor = cat === "xml" ? "xml" : cat === "ordering" ? "method" : ablFactor;
	const cmp = ablAutoPeers(factor);
	const side = ablationSelectionDrawer(cat, factor, cmp);
	const scoped = ablScopedVariants();
	const nScoped = scoped.length;
	// Loading bar: scanning attention status → indeterminate; aggregating variant
	// rows → determinate (how many of the scoped variants have loaded).
	const loadedN = scoped.filter((it) => state.runRowsCache.get(it.name)?.rows?.length).length;
	const loading = _ablComputedLoading ? true
		: (nScoped && loadedN < nScoped) ? { done: loadedN, total: nScoped, label: `aggregating attention · ${loadedN}/${nScoped} variants` }
			: null;
	if (!nScoped) {
		return workspaceShell("ablation", choices,
			reportHero("Ablation comparison", _ablComputedLoading ? "checking which variants have attention…" : "no computed variants in scope",
				reportEmpty(_ablComputedLoading ? "scanning…" : "compute variants in the ⚗ ablation tab (or widen the scope on the left), then they aggregate + compare here automatically")),
			side, [], null, loading);
	}
	const loadingNote = loading ? " · aggregating…" : "";
	if (cat === "xml") {
		return workspaceShell("ablation", choices,
			reportHero("XML existence", `${nScoped} variants${loadingNote} · attribute split · XML on vs off · all kinds, then one`, renderAblationXml(ablVariantSamples())),
			side, [], null, loading);
	}
	if (cat === "ordering") {
		return workspaceShell("ablation", choices,
			reportHero("Scene ordering", `${cmp.peers.length} method groups${loadingNote} · does context order change what it attends to?`, renderAblationOrdering(cmp)),
			side, [], null, loading);
	}
	if (cat === "spatial") {
		return workspaceShell("ablation", choices,
			reportHero("Spatial relevance", `${nScoped} variants${loadingNote} · does it attend to what's close / visible to the target?`, renderAblationSpatial(cmp)),
			side, [], null, ablSpatialLoading ? true : loading);
	}
	if (cat === "stats") {
		return workspaceShell("ablation", choices,
			reportHero("Ablation statistics", `${nScoped} variants${loadingNote} · each variant = one replicate · ANOVA (difference) + TOST (equivalence)`, renderAblationStats(ablVariantSamples())),
			side, [], null, loading);
	}
	if (cat === "breakdown") {
		// The step/kind/scene token-type breakdown, but the x-axis is the ablation
		// item: tag every peer's rows with their factor level → the grouped bucket
		// stacks draw one column per level (order/random/… or xml ✓/✗).
		const rows = [];
		cmp.peers.forEach((p, i) => { for (const r of p.rows) { r._ablGroup = p.key; r._ablLabel = p.key; r._ablOrd = i; } rows.push(...p.rows); });
		const colorMap = new Map(cmp.peers.map((p) => [p.key, p.color]));
		const byLabel = factor === "xml" ? "XML on vs off" : "scene order";
		return workspaceShell("ablation", choices,
			reportHero("Attention breakdown", `${cmp.peers.length} ${factor} groups${loadingNote} · category × subcategory + word / token types as spiders · prompt <tags> — one polygon/column per ${byLabel}`,
				el("div", { class: "abl-q-cards" }, ...graphs.renderAblationBreakdown(rows, colorMap))),
			side, [], null, loading);
	}
	const factorLabel = factor === "xml" ? "XML on vs off" : "shuffle methods";
	const hero = reportHero("Ablation comparison", `${cmp.peers.length} ${factor} groups${loadingNote} · aggregated across steps`, graphs.comparePeerAttributes(cmp, `attribute profile by ${factorLabel}`));
	let below;
	if (cat === "entities") below = [graphs.comparePeerEntities(cmp, `entities by ${factorLabel}`)];
	else if (cat === "heads") below = [graphs.comparePeerHeads(cmp, `heads / layers by ${factorLabel}`)];
	else below = [graphs.comparePeerAttributes(cmp, `attribute profile by ${factorLabel}`)];
	return workspaceShell("ablation", choices, hero, side, below, null, loading);
}

function renderSceneWorkspace() {
	const choices = [
		{ id: "main", label: "overview" }, { id: "tree", label: "structure" }, { id: "entities", label: "entities" },
		{ id: "attributes", label: "attributes" }, { id: "heads", label: "heads" }, { id: "tokenOrder", label: "breakdown" },
		{ id: "modules", label: "board" },
	];
	const { computed, rows, missing, order } = overviewRows();
	if (!computed.length) return workspaceShell("scene", choices, reportHero("Scene organization", null, reportEmpty("compute steps to build the scene-level view")), selectionDrawer(selectionSummary()), []);
	if (!rows.length) return workspaceShell("scene", choices, reportHero("Scene organization", null, reportEmpty(`loading ${computed.length} step analyses…`)), selectionDrawer(selectionSummary()), [], null, { done: 0, total: computed.length, label: `loading ${computed.length} step analyses…` });
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
	return workspaceShell("scene", choices, hero, side, below, sceneCmp.active ? null : rows,
		missing > 0 ? { done: rows.length, total: computed.length, label: `loading ${rows.length}/${computed.length} step analyses` } : (sceneCmp.loading ? true : null));
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
