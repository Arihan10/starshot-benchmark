// The unified ablation control board. "⚗ board" opens a full-screen screen that
// is the ONE place to configure, launch, monitor, and reset ablations.
//
// It reads the base cell's firing timeline (/tf-steps) and renders, per step
// kind, the full intended matrix: rows = the chosen experiment axis's levels (for
// coordinate, the DISTINCT levels for THAT kind), columns = temporal injections
// (the last-N firings of that kind). Every cell is a live
// status square whether or not its variant exists yet — click a not-created
// cell to launch it, an idle/failed one to (re)start it, a done/running one to
// open it in /tf. Bulk controls launch a whole kind or everything missing, and
// "reset" deletes ablation runs. Naming + launch come from ablationcore.js, so
// the board and the guided wizard produce identical runs.

import { api, SERVER_URL } from "./api.js";
import { state } from "./state.js";
import { el, toast, openModal } from "./ui.js";
import {
	SHUFFLE_METHODS, METHOD_HINT, slug, ALLOWED_MODEL, preferredModel, MAX_ABLATION_BATCH, hasSceneContext,
	expandExperimentForKind, EXPERIMENT_AXES, COORD_MODES, SCHEMA_MODES, GRAVITY_MODES, variantName, legacyVariantName, treatmentTag, loadBaseTemplates, launchVariant,
} from "./ablationcore.js";

let pollTimer = null;
let rendering = false;

// Board config (the intended sweep). Seeded from the active run on first open.
const cfg = {
	baseRun: null, slot: null, model: null, label: "",
	_newLabel: false, // label field in "➕ new label…" (free text) mode vs the dropdown
	labelByCell: {},  // `${slot}\0${model}` -> last-selected label (labels are decoupled per model)
	lastN: 3, reps: 1, temperature: 0.7,
	// The SINGLE experiment axis this board sweeps (its levels are the matrix rows);
	// every other axis stays at baseline, so experiments never cross-multiply. Add a
	// future experiment by registering it in ablationcore.EXPERIMENT_AXES /
	// expandExperiment(ForKind) + a level control in renderConfig's
	// `experimentControls` switch.
	experiment: "shuffle",
	methods: new Set(SHUFFLE_METHODS), xmlOn: true, xmlOff: true,
	coordModes: new Set(COORD_MODES.filter((c) => c.id !== "baseline").map((c) => c.id)),
	schemaModes: new Set(SCHEMA_MODES.filter((s) => s.id !== "baseline").map((s) => s.id)),
	gravityModes: new Set(GRAVITY_MODES.map((g) => g.id)), // none + q1..q4 (all launched)
	distractors: [0], seed: 1,
	// Attention-steering probe: targets to inject a "focus on X" directive for.
	// Each adds an attend variant next to its no-directive baseline.
	attendTargets: [],
};
// The selected base cell's available slots/models (may differ from the active
// run), plus its firing timeline.
let baseCells = { run: null, slots: [], models: [], defaultModel: null };
let timeline = { key: null, kinds: [], firingsByKind: {}, loading: false };
// The distinct ablation LABELS that already exist at the current base cell (each
// with its variant count), most-used first. The label is baked into every variant
// run NAME, so the board can only map a launched experiment back onto its matrix
// when cfg.label MATCHES the label it was launched under — hence the label field
// is a dropdown of these (default = an existing/started one), not a free text box
// that silently mismatches and shows every cell as "not created".
let cellLabels = []; // [{ label, n }] — recomputed on every cell change

async function pool(items, limit, fn) {
	const queue = [...items];
	const workers = Array.from({ length: Math.min(limit, queue.length) || 1 }, async () => {
		while (queue.length) { const it = queue.shift(); try { await fn(it); } catch { /* skip */ } }
	});
	await Promise.all(workers);
}

const $ = (id) => document.getElementById(id);
const isOpen = () => $("ablation-board")?.classList.contains("open");
const baseRunNames = () => (state.runs || []).map((r) => r.name).filter((n) => !n.includes("__abl-"));

// Coarse cell status now comes from the server's /run-status tail scan (see
// variantStatus); the client no longer pulls whole (huge) event logs to derive
// it. The inherited-base-error nuance lives there: a completion beats an earlier
// inherited run.error, and an error counts only if it's the latest marker.
const statusCache = new Map(); // variant name -> { slot, model, state }
// Only "done" is truly terminal (safe to stop re-reading). An "error" must keep
// being re-checked: an inherited base error can show transiently while a variant
// is still finishing (it resolves to "done" once run.done lands), and a rerun of a
// genuine error should update live. Caching "error" as terminal is exactly what
// pinned cells red forever until a manual reload.
const isTerminal = (s) => s === "done";

// Read a variant's cell status via the server's cheap tail-scan endpoint — NO
// full events.jsonl pull (a variant's inherited log can be hundreds of MB) and
// NO run hydration. The cell is the variant's recorded (slot, model).
async function variantStatus(name, slot, model, activeNames = null) {
	try {
		const { status } = await api.runStatus(name, slot, model);
		let state = status || "idle";
		// A variant whose log LOOKS in-progress (activity past an inherited base
		// error, no completion) but that ISN'T in the server's live task table is
		// actually stuck/interrupted, not running — surface it as idle so it reads
		// honestly and stays startable, instead of a permanent false spinner.
		if (state === "running" && activeNames && !activeNames.has(name)) state = "idle";
		return { slot, model, state };
	} catch { return { slot, model, state: "idle" }; }
}

function startable(cell) {
	if (!cell || !cell.slot || !cell.model) return false;
	return cell.state === "idle" || cell.state === "paused" || cell.state === "error";
}

// (Re)start an EXISTING variant = resume its cell (never re-create — that would
// 409). Launching a not-created variant goes through launchVariant instead.
async function startVariant(name, cell) {
	await api.resume(name, cell.slot, cell.model);
}

function tfHref(base, slot, model, variantNames) {
	const q = new URLSearchParams({ run: base, view: "ablation", pins: [base, ...variantNames].join(",") });
	if (slot) q.set("slot", slot);
	if (model) q.set("model", model);
	return `/tf?${q.toString()}`;
}

// The last-N firing event indices of a kind = its temporal injection columns.
function injectionsFor(kind) {
	return (timeline.firingsByKind[kind] || []).slice(-Math.max(1, cfg.lastN));
}

// The matrix columns for a kind: the last-N firings (intended injections) UNIONED
// with any firing that ALREADY has a launched variant for the current run (label),
// so a launched cut is never hidden just because lastN shrank since launch —
// otherwise its variants read as "not created" and get needlessly re-launched.
function columnsFor(kind, existing) {
	const cuts = new Set(injectionsFor(kind));
	if (existing) for (const cell of existing.values()) {
		if (cell && cell.kind === kind && (cell.label || "") === (cfg.label || "") && cell.cut != null) cuts.add(Number(cell.cut));
	}
	return [...cuts].sort((a, b) => a - b);
}

// The step kinds the CURRENT experiment sweeps = every scene-context kind in the
// timeline. Coordinate no longer drops non-bbox kinds (they vary along the INPUT
// axis); each kind's DISTINCT level set — bbox by input+output, non-bbox by input
// only — is resolved per-kind via expandExperimentForKind, and a kind whose level
// set comes out empty is skipped where the matrices are built.
function activeKinds() {
	return timeline.kinds;
}

// ---- base cell + timeline -------------------------------------------------

// IMPORTANT: never hydrate the whole base run here — large runs (many scenes ×
// models + branch sims) blow memory. /slots is a cheap in-memory summary, so we
// only surface cells that are ALREADY hydrated (the ones you've been working
// on), and default to the dashboard's active cell (which you're viewing, so it
// has data). events_count is 0 for un-hydrated cells; that's fine — pick one you
// know ran from the scene/model dropdowns.
async function selectBase(base) {
	cfg.baseRun = base;
	let slots = [];
	// The active run's cells are ALREADY loaded by the dashboard — reuse them so
	// opening the board on it triggers zero extra hydration. Only a different base
	// pays a /slots call (which lazily hydrates that run server-side).
	if (base === state.run && (state.slots || []).length) {
		slots = state.slots;
		baseCells = { run: base, slots: slots.map((s) => s.id), models: state.models || [], defaultModel: state.defaultModel || null, slotObjs: slots };
	} else {
		try {
			const data = await api.slots(base);
			slots = data.slots || [];
			baseCells = { run: base, slots: slots.map((s) => s.id), models: data.models || [], defaultModel: data.default_model || null, slotObjs: slots };
		} catch { slots = []; baseCells = { run: base, slots: [], models: [], defaultModel: null, slotObjs: [] }; }
	}
	// Keep only open-source models, and default to gemma when present.
	baseCells.models = baseCells.models.filter(ALLOWED_MODEL);
	baseCells.defaultModel = preferredModel(baseCells.models) || null;
	const { models } = baseCells;
	let pick = null;
	// Keep the current cell if still valid + populated.
	if (cfg.slot && cfg.model && baseCells.slots.includes(cfg.slot) && models.includes(cfg.model) && cellEvents(cfg.slot, cfg.model) > 0) {
		pick = { slot: cfg.slot, model: cfg.model };
	}
	// Prefer the dashboard's active SCENE with a gemma/qwen model that ran there —
	// this is exactly where the wizard launches (active scene + gemma), so the
	// board opens on the same cell and actually shows those variants. (state.model
	// itself may be a closed model like gemini-flash, which we never use.)
	if (!pick && base === state.run && state.slot && baseCells.slots.includes(state.slot)) {
		const m = (models.includes(state.model) && cellEvents(state.slot, state.model) > 0)
			? state.model
			: models.find((mm) => cellEvents(state.slot, mm) > 0);
		if (m) pick = { slot: state.slot, model: m };
	}
	// Otherwise the first ALREADY-hydrated (slot, model) with events.
	for (const s of slots) {
		if (pick) break;
		for (const m of models) if (cellEvents(s.id, m) > 0) { pick = { slot: s.id, model: m }; break; }
	}
	// Last resort: first slot + default model (may have no timeline yet).
	if (!pick && slots.length && models.length) pick = { slot: slots[0].id, model: baseCells.defaultModel || models[0] };
	cfg.slot = pick?.slot || null;
	cfg.model = pick?.model || null;
	timeline = { key: null, kinds: [], firingsByKind: {}, loading: false };
	await refreshCell();
}

// Point the board at a different (slot, model) — used by the "variants at other
// cells" hint so wizard launches on another scene are one click away.
function switchCell(s, m) {
	cfg.slot = s;
	cfg.model = m;
	refreshCell();
}

// Logged-event count for a base cell (0 = never ran / empty).
function cellEvents(slotId, m) {
	return (baseCells.slotObjs || []).find((s) => s.id === slotId)?.runs?.[m]?.events_count ?? 0;
}
// Does a scene have data under any model? Which model to prefer for it?
function sceneHasData(slotId) { return (baseCells.models || []).some((m) => cellEvents(slotId, m) > 0); }
function bestModelFor(slotId) { return (baseCells.models || []).find((m) => cellEvents(slotId, m) > 0) || cfg.model || baseCells.defaultModel; }

async function ensureTimeline() {
	const key = `${cfg.baseRun}\u0000${cfg.slot}\u0000${cfg.model}`;
	if (!cfg.baseRun || !cfg.slot || !cfg.model) { timeline = { key, kinds: [], firingsByKind: {}, loading: false }; return; }
	if (timeline.key === key && !timeline.loading) return;
	statusCache.clear(); // the cell changed → variant event paths differ
	timeline = { key, kinds: [], firingsByKind: {}, loading: true };
	try {
		const resp = await api.tfSteps(cfg.baseRun, cfg.slot, cfg.model);
		const steps = resp.steps || [];
		const firingsByKind = {};
		const kinds = [];
		for (const s of steps) {
			const k = s.template ?? s.step;
			// GATE 1a (design): only kinds whose prompt renders scene context — a
			// scene-context ablation is meaningless on root plans / overall_bbox /
			// image_prompt. GATE 1b (runtime): drop firings whose scene context is
			// actually EMPTY (has_scene=false) — nothing to shuffle / attend to.
			if (!k || !hasSceneContext(k) || s.has_scene === false) continue;
			if (!firingsByKind[k]) { firingsByKind[k] = []; kinds.push(k); }
			firingsByKind[k].push(s.event_index);
		}
		timeline = { key, kinds, firingsByKind, loading: false };
	} catch (e) {
		timeline = { key, kinds: [], firingsByKind: {}, loading: false, error: e.message };
	}
}

// The distinct ablation labels that ALREADY exist at the current base cell (scoped
// to slot+model; legacy runs without a recorded cell count for any cell), most-used
// first. Cheap /runs read, refreshed on every cell change so the label dropdown
// always reflects what's actually been started here.
async function ensureCellLabels() {
	if (!cfg.baseRun || !cfg.slot || !cfg.model) { cellLabels = []; return; }
	try {
		// One-read discovery: every variant of this base (nested + legacy flat),
		// already scoped server-side — no /runs scan + name-prefix filter.
		const list = (await api.ablations(cfg.baseRun)).variants ?? [];
		const byLabel = new Map(); // label ("" = none) -> { n, treatments: [] }
		for (const r of list) {
			const abl = r && r.ablation;
			if (!abl) continue;
			// A variant belongs to the cell it launched from; legacy runs without a
			// recorded cell are counted regardless (they map onto any cell).
			if (abl.slot && abl.model && (abl.slot !== cfg.slot || abl.model !== cfg.model)) continue;
			const l = (abl && abl.label) ? String(abl.label) : "";
			const e = byLabel.get(l) || byLabel.set(l, { n: 0, treatments: [] }).get(l);
			e.n += 1;
			if (abl && abl.treatment) e.treatments.push(abl.treatment);
		}
		// Each label carries its DERIVED experiment (+ the levels it was launched
		// under), so selecting it can repaint the board as the experiment it is.
		cellLabels = [...byLabel.entries()].map(([label, e]) => ({ label, n: e.n, derived: deriveExperiment(e.treatments) }))
			.sort((a, b) => b.n - a.n || (a.label < b.label ? -1 : 1));
	} catch { cellLabels = []; }
}

// Snap cfg.label onto an EXISTING started label so the matrix actually maps the
// variants you launched — a wrong/empty label is exactly why a started run "isn't
// captured" on the board. With nothing started yet, drop into new-label mode so a
// first experiment can still be created.
function applyLabelDefault() {
	if (!cellLabels.length) { cfg._newLabel = true; return; }
	// An existing label now covers cfg.label (incl. a just-created new one) → use it.
	if (cellLabels.some((x) => x.label === cfg.label)) { cfg._newLabel = false; return; }
	if (cfg._newLabel) return; // user is deliberately composing a brand-new label
	cfg.label = cellLabels[0].label; // otherwise snap to the most-used started label
}

// Infer which experiment axis a label's variants belong to (launches are
// disentangled — one axis at a time — so the first non-baseline axis wins), plus
// the LEVELS actually launched, so switching to a label repaints the board as the
// experiment it really is (a coordinate label wouldn't map onto a shuffle matrix).
// Returns null when only baselines are present (can't tell — leave the axis as-is).
function deriveExperiment(treatments) {
	const coordModes = new Set(), schemaModes = new Set(), gravityModes = new Set(), methods = new Set(), attend = new Set(), dist = new Set();
	let anyXmlOff = false;
	for (const t of treatments || []) {
		if (!t) continue;
		if (t.coord_mode && t.coord_mode !== "baseline") coordModes.add(t.coord_mode);
		if (t.schema_mode && t.schema_mode !== "baseline") schemaModes.add(t.schema_mode);
		if (t.gravity_mode && t.gravity_mode !== "baseline") gravityModes.add(t.gravity_mode);
		if (t.shuffle_method && t.shuffle_method !== "order") methods.add(t.shuffle_method);
		if (t.attend_target) attend.add(String(t.attend_target));
		if (t.distractors) dist.add(Number(t.distractors));
		if (t.xml_tags === false) anyXmlOff = true;
	}
	if (coordModes.size) return { experiment: "coordinate", coordModes };
	if (schemaModes.size) return { experiment: "schema", schemaModes };
	if (gravityModes.size) return { experiment: "gravity", gravityModes };
	if (attend.size) return { experiment: "attend", attendTargets: [...attend] };
	if (dist.size) return { experiment: "distractors", distractors: [...dist] };
	if (methods.size) return { experiment: "shuffle", methods };
	if (anyXmlOff) return { experiment: "xml" };
	return null;
}

// Align the board's experiment axis (+ its level selections) to the experiment the
// given label was launched under, so its matrix rows show every launched variant.
// No-op for a label with no derivable experiment (legacy / baseline-only), leaving
// the current axis untouched.
function applyExperimentFromLabel(label) {
	const d = cellLabels.find((x) => x.label === label)?.derived;
	if (!d) return;
	cfg.experiment = d.experiment;
	if (d.coordModes) cfg.coordModes = new Set(d.coordModes);
	if (d.schemaModes) cfg.schemaModes = new Set(d.schemaModes);
	if (d.gravityModes) cfg.gravityModes = new Set(d.gravityModes);
	if (d.methods) cfg.methods = new Set(d.methods);
	if (d.attendTargets) cfg.attendTargets = d.attendTargets.slice();
	if (d.distractors) cfg.distractors = d.distractors.slice();
}

const cellKeyOf = () => `${cfg.slot}\u0000${cfg.model}`;

// Labels are DECOUPLED PER MODEL: each (scene, model) cell remembers the run label
// you were last on, so switching model restores THAT model's run (not a label that
// only exists under the other model). Falls through to applyLabelDefault when the
// cell has no remembered/valid label yet.
function restoreCellLabel() {
	const remembered = cfg.labelByCell[cellKeyOf()];
	if (remembered != null && cellLabels.some((x) => x.label === remembered)) { cfg.label = remembered; cfg._newLabel = false; }
}

// The single path every cell change funnels through (base / scene / model switch):
// reload the timeline + started labels, restore this cell's remembered label (or
// snap a default), ALIGN the experiment axis to that label, then repaint. So
// switching model auto-picks the right (model-scoped) label + its experiment.
async function refreshCell() {
	await ensureTimeline();
	await ensureCellLabels();
	restoreCellLabel();
	applyLabelDefault();
	if (!cfg._newLabel) applyExperimentFromLabel(cfg.label);
	renderConfig();
	renderMatrices();
}

// ---- existing variants ----------------------------------------------------

// The FAMILY tag (shared across a cell's replicates) recovered from a variant's
// STORED, launch-time treatment. This is the ROBUST match key: it never depends on
// the run-name string (base/slot/model slugging) or on the board's current cfg, so
// a launched run maps back to its matrix cell even after a reload re-derives state.
// The stored seed is the per-replicate EFFECTIVE seed (base + rep-1); the family
// tag uses the BASE seed, so undo the per-rep bump before tagging.
function tagOfStoredTreatment(t, rep) {
	if (!t) return "order";
	const baseSeed = (Number(t.seed) || 0) - (Math.max(1, Number(rep) || 1) - 1);
	return treatmentTag({
		method: t.shuffle_method || "order",
		xml: t.xml_tags !== false,
		distractors: t.distractors || 0,
		seed: baseSeed > 0 ? baseSeed : 0,
		attend: t.attend_target || "",
		coord: t.coord_mode || "baseline",
		schema: t.schema_mode || "baseline",
		gravity: t.gravity_mode || "baseline",
	});
}

async function fetchExisting(base) {
	// One-read discovery: this base's variants (nested + legacy flat), scoped
	// server-side and carrying `run_id` (the API key) + the `ablation` block.
	const list = (await api.ablations(base)).variants ?? [];
	// The server's live task table — which variant runs are ACTUALLY executing now
	// (keyed by run name). Lets us tell a truly-running variant from one whose log
	// merely looks unfinished (inherited base error). Empty on failure → nothing
	// treated as active (so nothing shows a false "running").
	let activeNames = new Set();
	try { const a = await api.activeGenerations(); activeNames = new Set((a.pipelines || []).map((p) => p.run)); } catch { /* server down — treat none as active */ }
	const ofBase = list.filter((r) => r && r.ablation);
	// Which cells this base's variants live in (for the "switch to where the
	// variants are" hint when the board is looking at a different cell).
	const otherCells = new Map(); // "slot\u0000model" -> count, excluding the current cell
	for (const r of ofBase) {
		const s = r.ablation?.slot, m = r.ablation?.model;
		if (!s || !m || (s === cfg.slot && m === cfg.model)) continue;
		const k = `${s}\u0000${m}`;
		otherCells.set(k, (otherCells.get(k) || 0) + 1);
	}
	// A variant belongs to the exact cell it launched from — only THIS cell's
	// variants (legacy ones without a recorded cell are shown regardless).
	const mine = ofBase.filter((r) => {
		const abl = r.ablation;
		if (abl && abl.slot && abl.model) return abl.slot === cfg.slot && abl.model === cfg.model;
		return true;
	});
	const cellOf = (r) => ({ slot: r.ablation?.slot || cfg.slot, model: r.ablation?.model || cfg.model });
	// Refresh status only for non-terminal / uncached variants, capped, so the
	// poll never balloons — done/error don't change, so they stay cached.
	const toFetch = mine.filter((r) => !isTerminal(statusCache.get(r.name)?.state)).slice(0, 80);
	await pool(toFetch, 6, async (r) => { const c = cellOf(r); statusCache.set(r.name, await variantStatus(r.name, c.slot, c.model, activeNames)); });
	const map = new Map();
	for (const r of mine) {
		const c = cellOf(r);
		const s = statusCache.get(r.name) || { slot: c.slot, model: c.model, state: "idle" };
		const a = r.ablation || {};
		const rep = Number(a.replicate) || 1;
		const kind = a.target_step_kind ?? null;
		// Carry the immutable launch-time coordinates alongside status so familyRepInfo
		// can map this run to its cell by META (kind, cut, label, tag), not by a
		// reconstructed name — the fix for "launched runs not coming back on reload".
		map.set(r.name, {
			slot: s.slot, model: s.model, state: s.state,
			kind, cut: a.cut ?? null, label: a.label || "", rep,
			tag: kind != null ? tagOfStoredTreatment(a.treatment, rep) : null,
		});
	}
	return { map, otherCells };
}

// ---- launching ------------------------------------------------------------

async function ensureTemplates() {
	try { return await loadBaseTemplates(cfg.baseRun); }
	catch (e) { toast(`couldn't load base prompts: ${e.message}`, "err"); return null; }
}

function variantParams(kind, t, cut, rep = 1) {
	return { baseRun: cfg.baseRun, slot: cfg.slot, model: cfg.model, label: cfg.label, lastN: cfg.lastN, rep, temperature: cfg.temperature,
		kind, cut, method: t.method, xml: t.xml, distractors: t.distractors, seed: t.seed, attend: t.attend, coord: t.coord, schema: t.schema, gravity: t.gravity, tag: t.tag };
}

// ---- unified execution queue ----------------------------------------------
// ONE persistent queue drives everything — launching new variants, resuming
// stalled ones, and running a drag-selected group all APPEND to it. Up to
// MAX_ABLATION_BATCH items execute at once; the rest wait. Adding work while the
// queue runs just extends it (no "already running" bounce), and ⏹ stop drains it.
let queue = [];               // [{ type:"launch", key, params } | { type:"resume", key, name, cell }]
let queueActive = false;      // the drain loop is running
let queueCancel = false;      // ⏹ stop
let queueDone = 0, queueTotal = 0;
const queuedKeys = new Set(); // in-flight keys, so re-adding the same cell is a no-op
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function queueProgress() {
	const sub = $("abl-board-sub");
	if (sub && queueActive) sub.textContent = `queue: ${queueDone}/${queueTotal} done · ≤${MAX_ABLATION_BATCH} running${queueCancel ? " · stopping" : ""}`;
}

async function waitForVariant(name, slot, model) {
	const deadline = Date.now() + 6 * 60 * 1000; // safety cap so a stuck run can't wedge a worker
	while (!queueCancel && Date.now() < deadline) {
		const st = (await variantStatus(name, slot, model)).state;
		statusCache.set(name, { slot, model, state: st });
		if (st === "done" || st === "error") return;
		await sleep(2500);
	}
}

async function runQueueItem(it, tmpls) {
	try {
		if (it.type === "launch") {
			const r = await launchVariant(it.params, tmpls);
			if (r.ok) { statusCache.delete(r.name); await waitForVariant(r.name, it.params.slot, it.params.model); }
		} else {
			statusCache.delete(it.name);
			await startVariant(it.name, it.cell);
			await waitForVariant(it.name, it.cell.slot, it.cell.model);
		}
	} catch { /* one item failing must not wedge the whole queue */ }
}

// APPEND work; start the drain loop if idle. Deduped by key. Returns how many were
// newly queued.
function enqueueWork(items) {
	if (!ALLOWED_MODEL(cfg.model || "")) { toast(`ablations are limited to gemma / qwen — got "${cfg.model}"`, "err"); return 0; }
	let added = 0;
	for (const it of items || []) {
		if (!it || queuedKeys.has(it.key)) continue;
		queuedKeys.add(it.key);
		queue.push(it);
		added += 1;
	}
	if (added) { queueTotal += added; drainQueue(); }
	return added;
}

async function drainQueue() {
	if (queueActive) { queueProgress(); return; } // running — workers pick up the appended items
	queueActive = true;
	queueCancel = false;
	const tmpls = await ensureTemplates();
	if (!tmpls) { queueActive = false; queue = []; queuedKeys.clear(); queueTotal = 0; queueDone = 0; return; }
	queueProgress();
	// Outer loop re-runs the worker pool if items were appended during teardown
	// (race-safe): a new enqueue that lands as the last worker exits is caught here.
	while (queue.length && !queueCancel) {
		const worker = async () => {
			while (!queueCancel) {
				const it = queue.shift();
				if (!it) break;
				await runQueueItem(it, tmpls);
				queuedKeys.delete(it.key);
				queueDone += 1;
				queueProgress();
				if (isOpen()) renderMatrices();
			}
		};
		await Promise.all(Array.from({ length: MAX_ABLATION_BATCH }, worker));
	}
	queueActive = false;
	const stopped = queueCancel;
	if (stopped) { queue = []; queuedKeys.clear(); }
	const done = queueDone, total = queueTotal;
	queueDone = 0; queueTotal = 0; queueCancel = false;
	toast(`queue ${stopped ? "stopped" : "done"}: ${done}/${total}`, stopped ? "err" : "ok");
	// Promote a freshly launched (esp. brand-new-label) experiment into the label
	// dropdown so it's immediately selectable — applyLabelDefault only ever keeps
	// the current selection here (a matching label exits new-label mode), never
	// snaps it away.
	if (isOpen()) { await ensureCellLabels(); applyLabelDefault(); renderConfig(); }
	renderMatrices();
}

function stopQueue() { if (queueActive) { queueCancel = true; queueProgress(); } else toast("nothing running"); }

// Queue a list of launch params (missing replicates, etc.).
function launchMany(list) {
	const n = enqueueWork((list || []).map((p) => ({ type: "launch", key: variantName(p), params: p })));
	if (n) toast(`queued ${n} launch${n === 1 ? "" : "es"}`);
	else if ((list || []).length) toast("those are already queued / running");
	else toast("nothing to launch");
}

// Queue a resume of every STALLED variant (errored / idle / interrupted) across
// the sweep — appends to the shared queue like everything else.
async function rerunErrored() {
	const { map: existing } = await fetchExisting(cfg.baseRun);
	const stalled = [...existing.entries()].filter(([, c]) => startable(c));
	if (!stalled.length) { toast("no stalled variants to resume"); return; }
	const n = enqueueWork(stalled.map(([name, cell]) => ({ type: "resume", key: name, name, cell })));
	if (n) toast(`queued ${n} resume${n === 1 ? "" : "s"}`);
	else toast("stalled variants already queued");
}

// Run a DRAG-SELECTED group of cells: bring each up to n (queue missing reps) and
// resume any stalled reps it already has.
async function runSelectedCells(host) {
	const cells = [...selectedCells.values()];
	selectedCells.clear();
	for (const c of host.querySelectorAll(".abl-cell.sel")) c.classList.remove("sel");
	updateSelBar(host);
	if (!cells.length) return;
	const { map: existing } = await fetchExisting(cfg.baseRun);
	const desired = Math.max(cfg.reps, 1);
	const items = [];
	for (const c of cells) {
		const byRep = familyRepInfo(existing, c.kind, c.cut, c.tag);
		for (const k of missingReps(byRep, desired)) items.push({ type: "launch", key: nameFor(c.kind, c.cut, c.tag, k), params: variantParams(c.kind, c.t, c.cut, k) });
		for (const r of byRep.values()) if (startable(r.cell)) items.push({ type: "resume", key: r.name, name: r.name, cell: r.cell });
	}
	const n = enqueueWork(items);
	toast(n ? `queued ${n} from ${cells.length} cell${cells.length === 1 ? "" : "s"}` : "selection already complete / queued");
}

// ---- rubber-band group select over the cells (mirrors the /tf drawer) -------
// Selection is keyed on cell IDENTITY (kind\0tag\0cut), not DOM nodes, so it
// survives the board's 4 s poll re-renders (matrixCell re-applies `.sel`).
let drag = null;
const selectedCells = new Map(); // key -> { kind, t, cut, tag }
const cellKey = (kind, tag, cut) => `${kind}\u0000${tag}\u0000${cut}`;

function updateSelBar(host) {
	const n = selectedCells.size;
	let bar = document.getElementById("abl-selbar");
	if (!n) { if (bar) bar.remove(); return; }
	if (!bar) { bar = el("div", { id: "abl-selbar" }); document.body.appendChild(bar); }
	bar.replaceChildren(
		el("span", { class: "abl-selbar-n", text: `${n} cell${n === 1 ? "" : "s"} selected` }),
		el("button", { class: "abl-start primary", title: `queue every selected cell up to n=${Math.max(cfg.reps, 1)} + resume its stalled reps`, onclick: () => runSelectedCells(host) }, "▶ run selected"),
		el("button", { class: "abl-start", onclick: () => { selectedCells.clear(); for (const c of host.querySelectorAll(".abl-cell.sel")) c.classList.remove("sel"); updateSelBar(host); } }, "clear"));
}

function initBoardDrag() {
	const host = $("abl-board-body");
	if (!host || host._dragInit) return;
	host._dragInit = true;
	host.addEventListener("mousedown", (e) => {
		if (e.button !== 0) return;
		if (e.target.closest("button, a, select, input, .abl-mx-colh, .abl-mx-rowh")) return; // headers keep their own clicks
		drag = { x0: e.clientX, y0: e.clientY, moved: false, rect: null, shift: e.shiftKey };
	});
	document.addEventListener("mousemove", (e) => {
		if (!drag) return;
		if (!drag.moved && Math.abs(e.clientX - drag.x0) + Math.abs(e.clientY - drag.y0) < 6) return;
		if (!drag.moved) {
			drag.moved = true;
			if (!drag.shift) { selectedCells.clear(); for (const c of host.querySelectorAll(".abl-cell.sel")) c.classList.remove("sel"); }
			drag.rect = el("div", { class: "abl-drag" });
			document.body.appendChild(drag.rect);
		}
		const l = Math.min(e.clientX, drag.x0), t = Math.min(e.clientY, drag.y0);
		const r = Math.max(e.clientX, drag.x0), b = Math.max(e.clientY, drag.y0);
		Object.assign(drag.rect.style, { left: `${l}px`, top: `${t}px`, width: `${r - l}px`, height: `${b - t}px` });
		for (const cell of host.querySelectorAll(".abl-cell")) {
			if (!cell._sel) continue;
			const q = cell.getBoundingClientRect();
			const hit = q.bottom >= t && q.top <= b && q.right >= l && q.left <= r;
			if (hit) { selectedCells.set(cell._selKey, cell._sel); cell.classList.add("sel"); }
			else if (!drag.shift) { selectedCells.delete(cell._selKey); cell.classList.remove("sel"); }
		}
		updateSelBar(host);
	});
	document.addEventListener("mouseup", () => {
		if (!drag) return;
		const wasDrag = drag.moved;
		if (drag.rect) drag.rect.remove();
		drag = null;
		if (!wasDrag) return; // a plain click → let the cell's own handler run
		const eat = (ev) => { ev.stopPropagation(); ev.preventDefault(); };
		document.addEventListener("click", eat, { capture: true, once: true });
		setTimeout(() => document.removeEventListener("click", eat, true), 0);
		updateSelBar(host);
	});
}

async function resetAll() {
	// Variants are folded under each base run now, so gather them per base via the
	// ablations index rather than scanning /runs for `__abl-` names.
	const data = await api.runs();
	const bases = (Array.isArray(data) ? data : (data.runs ?? [])).map((r) => r.name).filter(Boolean);
	const names = [];
	await pool(bases, 6, async (b) => {
		try { for (const v of ((await api.ablations(b)).variants ?? [])) names.push(v.run_id || v.name); } catch { /* skip */ }
	});
	if (!names.length) { toast("no ablation runs to delete"); return; }
	if (!window.confirm(`Delete ALL ${names.length} ablation run(s)? They are stopped and removed from disk.`)) return;
	let ok = 0, fail = 0;
	await pool(names, 3, async (n) => { try { await api.deleteRun(n); ok += 1; } catch { fail += 1; } });
	toast(`deleted ${ok}${fail ? `, ${fail} failed` : ""}`, fail ? "err" : "ok");
	renderMatrices();
}

// Remove (delete from disk) every run of the CURRENTLY SELECTED label at THIS cell
// (scene + model) — the granular counterpart to the global reset, so you can drop
// one experiment without nuking the rest. Scoped to the cell so a same-named run on
// another model is untouched (labels are decoupled per model).
async function removeLabel() {
	const label = cfg.label;
	const list = (await api.ablations(cfg.baseRun)).variants ?? [];
	const names = list.filter((r) => {
		const a = r && r.ablation;
		if (!a) return false;
		if (a.slot && a.model && (a.slot !== cfg.slot || a.model !== cfg.model)) return false;
		return ((a.label) ? String(a.label) : "") === label;
	}).map((r) => r.run_id || r.name);
	if (!names.length) { toast("no runs for this label at this cell"); return; }
	if (!window.confirm(`Delete ${names.length} run(s) for label "${label || "(no label)"}" at ${cfg.slot} / ${cfg.model}? They are stopped and removed from disk.`)) return;
	let ok = 0, fail = 0;
	await pool(names, 3, async (n) => { try { await api.deleteRun(n); ok += 1; } catch { fail += 1; } });
	toast(`deleted ${ok}${fail ? `, ${fail} failed` : ""}`, fail ? "err" : "ok");
	delete cfg.labelByCell[cellKeyOf()];
	await ensureCellLabels();
	applyLabelDefault();
	if (!cfg._newLabel) applyExperimentFromLabel(cfg.label);
	renderConfig();
	renderMatrices();
}

// ---- matrix rendering -----------------------------------------------------

const nameFor = (kind, cut, tag, rep) => variantName({ baseRun: cfg.baseRun, slot: cfg.slot, model: cfg.model, label: cfg.label, kind, cut, tag, rep });
const legacyNameFor = (kind, cut, tag, rep) => legacyVariantName({ baseRun: cfg.baseRun, label: cfg.label, kind, cut, tag, rep });

// A cell's replicate family: which rep indices already exist, mapped to their
// status. PRIMARY match is by the variant's IMMUTABLE stored meta (kind, cut,
// label, family-tag) — so a launched run maps back to its cell regardless of any
// run-name drift (that's what made runs "never come back" on reload and caused
// re-launches to mint duplicates). Name reconstruction (new + legacy) is kept only
// as a FALLBACK for old runs whose meta lacks these fields.
function familyRepInfo(existing, kind, cut, tag) {
	const base = nameFor(kind, cut, tag);
	const legacyBase = legacyNameFor(kind, cut, tag);
	const repOf = (name, b) => {
		if (name === b) return 1;
		if (name.startsWith(`${b}-r`)) { const n = Number(name.slice(b.length + 2)); if (Number.isInteger(n) && n > 1) return n; }
		return null;
	};
	const label = cfg.label || "";
	const byRep = new Map();
	for (const [name, cell] of existing) {
		// META match (robust): same kind@cut, same run label, same family tag.
		let rep = (cell && cell.kind != null && cell.kind === kind && Number(cell.cut) === Number(cut)
			&& (cell.label || "") === label && cell.tag === tag) ? cell.rep : null;
		// FALLBACK: reconstructed-name match for legacy runs without stored meta.
		if (rep == null) rep = repOf(name, base) ?? repOf(name, legacyBase);
		if (rep != null && !byRep.has(rep)) byRep.set(rep, { rep, name, cell });
	}
	return byRep;
}
// The rep indices to launch to bring a family up to `target` (smallest gaps first).
function missingReps(byRep, target) {
	const out = [];
	for (let k = 1; k <= target && out.length < (target - byRep.size); k++) if (!byRep.has(k)) out.push(k);
	return out;
}

// Open the /tf ablation view comparing a set of (created) variant runs. tfHref
// pins the base as the reference peer, so `names` are the variants to overlay.
function openTfCompare(names, note) {
	const created = names.filter(Boolean);
	if (!created.length) { toast(note || "no created runs to compare yet"); return; }
	window.open(tfHref(cfg.baseRun, cfg.slot, cfg.model, created), "_blank", "noopener");
}

// Click a cell → a detail card: what it is, its status, and its actions
// (launch / start / open its scene / compare vs baseline). Loads nothing extra —
// just the info already on hand + links — so it's cheap and deliberate (no more
// accidental single-click mass launches).
function openVariantCard(kind, t, cut, existing) {
	const base = nameFor(kind, cut, t.tag);
	const byRep = familyRepInfo(existing, kind, cut, t.tag);
	const reps = [...byRep.values()].sort((a, b) => a.rep - b.rep);
	const slot = cfg.slot, model = cfg.model;
	const desired = Math.max(cfg.reps, 1);
	const baselineNames = [...familyRepInfo(existing, kind, cut, "order").values()].map((r) => r.name);
	const row = (k, v) => el("div", { class: "abl-card-row" }, el("span", { class: "abl-card-k", text: k }), el("span", { class: "abl-card-v", text: v }));
	openModal(`${kind} · injection @${cut}`, (close) => {
		const actions = [el("button", { text: "close", onclick: close })];
		const doneN = reps.filter((r) => r.cell.state === "done").length;
		const body = [
			el("div", { class: "abl-card-name", text: base }),
			row("treatment", t.tag),
			t.attend ? row("attend →", t.attend) : null,
			row("cell", `${slot} · ${model}`),
			row("replicates", reps.length ? `${reps.length} · ${doneN} done` : "none yet"),
		].filter(Boolean);
		if (reps.length) {
			body.push(el("div", { class: "abl-card-reps" }, ...reps.map((r) =>
				el("span", { class: "abl-pill", title: r.name }, el("span", { class: `dot ${r.cell.state}` }), `r${r.rep} ${r.cell.state}`))));
		} else {
			body.push(el("div", { class: "m-hint", text: `not launched yet — forks the base here and re-infers this treated step ${desired}×; each replicate bumps the RNG seed for an independent draw.` }));
		}
		// launch the missing replicates to reach the target n
		const missing = missingReps(byRep, desired).map((k) => variantParams(kind, t, cut, k));
		if (missing.length) {
			actions.push(el("button", { class: "primary", text: `▶ launch ${missing.length} replicate${missing.length > 1 ? "s" : ""} → n=${desired}`,
				onclick: async () => { close(); await launchMany(missing); } }));
		} else if (reps.length) {
			const nextRep = Math.max(...reps.map((r) => r.rep)) + 1;
			actions.push(el("button", { class: "primary", text: "+1 replicate",
				title: "add one more independent replicate (bumps the RNG seed)",
				onclick: async () => { close(); await launchMany([variantParams(kind, t, cut, nextRep)]); } }));
		}
		// restart any idle / errored replicates in place
		const restartable = reps.filter((r) => startable(r.cell));
		if (restartable.length) actions.push(el("button", { text: `↻ restart ${restartable.length}`,
			onclick: () => { close(); const n = enqueueWork(restartable.map((r) => ({ type: "resume", key: r.name, name: r.name, cell: r.cell }))); if (n) toast(`queued ${n} resume${n === 1 ? "" : "s"}`); } }));
		if (reps.length) {
			const rep1 = reps.find((r) => r.rep === 1) || reps[0];
			const sceneHref = `/tf?${new URLSearchParams({ run: rep1.name, slot, model }).toString()}`;
			const cmpHref = tfHref(cfg.baseRun, slot, model, [...new Set([...baselineNames, ...reps.map((r) => r.name)])]);
			body.push(el("div", { class: "abl-card-actions" },
				el("a", { class: "abl-open", href: sceneHref, target: "_blank", rel: "noopener", title: "open this variant's 3D scene in /tf" }, "scene ↗"),
				el("a", { class: "abl-open", href: cmpHref, target: "_blank", rel: "noopener", title: "compare all replicates + baseline in /tf" }, "compare vs baseline ↗")));
		}
		return { body, actions };
	});
}

function matrixCell(kind, t, cut, existing) {
	const base = nameFor(kind, cut, t.tag);
	const reps = [...familyRepInfo(existing, kind, cut, t.tag).values()];
	let node;
	if (!reps.length) {
		node = el("div", { class: "abl-cell notcreated", title: `${base}\nnot created · click to launch ${cfg.reps} replicate${cfg.reps > 1 ? "s" : ""} · drag to group-select`,
			onclick: () => openVariantCard(kind, t, cut, existing) }, el("span", { class: "abl-plus", text: "+" }));
	} else {
		const total = reps.length;
		const doneN = reps.filter((r) => r.cell.state === "done").length;
		const runningN = reps.filter((r) => r.cell.state === "running").length;
		const errN = reps.filter((r) => r.cell.state === "error").length;
		const idleN = reps.filter((r) => r.cell.state === "idle" || r.cell.state === "paused").length;
		// Colour priority: running (something is live) → error → done → idle. The
		// number is the replicate count: N when all done, else done/total.
		const agg = runningN ? "running" : errN ? "error" : doneN ? "done" : "idle";
		const label = doneN === total ? String(total) : `${doneN}/${total}`;
		node = el("div", {
			class: `abl-cell ${agg}`,
			title: `${base}\n${total} replicate${total > 1 ? "s" : ""} · ${doneN} done${errN ? ` · ${errN} error` : ""}${runningN ? ` · ${runningN} running` : ""}${idleN ? ` · ${idleN} idle` : ""} · click for details · drag to group-select`,
			onclick: () => openVariantCard(kind, t, cut, existing),
		}, el("span", { class: `abl-cnt ${agg}`, text: label }));
	}
	// Identity for the rubber-band group select; re-apply if still selected across
	// a re-render.
	node._sel = { kind, t, cut, tag: t.tag };
	node._selKey = cellKey(kind, t.tag, cut);
	if (selectedCells.has(node._selKey)) node.classList.add("sel");
	return node;
}

// All existing run names in a family (every replicate) — for compare links.
const familyNames = (existing, kind, cut, tag) => [...familyRepInfo(existing, kind, cut, tag).values()].map((r) => r.name);

// Compact ablation-LEVEL label for a treatment row (the matrix "rank"): the coord
// condition (L2L / LG2G / …), the shuffle method, xml on/off, the attend target,
// etc. — the readable experiment level, NOT the raw filename slug/tag. Keyed on
// the board's active experiment axis.
const _COORD_IN = { both: "LG", local: "L", global: "G" };
const _COORD_OUT = { local: "L", global: "G" };
function coordCompact(id) {
	const c = COORD_MODES.find((x) => x.id === id);
	return c ? `${_COORD_IN[c.input] || "?"}2${_COORD_OUT[c.output] || "?"}` : (id || "baseline");
}
function levelLabel(t) {
	switch (cfg.experiment) {
		case "coordinate": return coordCompact(t.coord);
		case "schema": return t.schema || "soft-JSON";
		case "gravity": return t.gravity === "none" ? "no tag" : (t.gravity ? `close@${String(t.gravity).toUpperCase()}` : "baseline");
		case "shuffle": return t.method || "order";
		case "xml": return t.xml === false ? "xml off" : "xml on";
		case "attend": return t.attend ? `→ ${slug(t.attend, 14)}` : "baseline";
		case "distractors": return t.distractors ? `d${t.distractors}` : "baseline";
		default: return t.tag;
	}
}

function kindMatrix(kind, treatments, existing) {
	const cuts = columnsFor(kind, existing);
	const desired = Math.max(cfg.reps, 1);
	const missing = [];
	let createdRuns = 0;
	const allNames = [];
	for (const t of treatments) for (const cut of cuts) {
		const byRep = familyRepInfo(existing, kind, cut, t.tag);
		createdRuns += byRep.size;
		for (const r of byRep.values()) allNames.push(r.name);
		for (const k of missingReps(byRep, desired)) missing.push(variantParams(kind, t, cut, k));
	}
	const launchKindBtn = missing.length
		? el("button", { class: "abl-start", title: `launch ${missing.length} replicate run(s) to bring every cell of this kind to n=${desired}`,
			onclick: () => { if (window.confirm(`Queue ${missing.length} replicate(s) for ${kind} (→ n=${desired})? (${MAX_ABLATION_BATCH} run at a time, the rest wait)`)) launchMany(missing); } },
			`▶ launch ${missing.length} → n=${desired}`)
		: null;
	const openKind = el("a", {
		class: "abl-open", target: "_blank", rel: "noopener",
		href: allNames.length ? tfHref(cfg.baseRun, cfg.slot, cfg.model, allNames) : "#",
		title: allNames.length ? `compare this kind's ${allNames.length} created run(s) in /tf` : "launch runs first, then compare",
		...(allNames.length ? {} : { style: "opacity:.4;pointer-events:none" }),
	}, "↗ /tf");

	const grid = el("div", { class: "abl-matrix", style: `grid-template-columns: 168px repeat(${cuts.length}, 52px)` });
	grid.appendChild(el("div", { class: "abl-mx-corner", title: "rows = treatments · columns = temporal injection · cell number = replicates done · click a header to compare that row/column in /tf", text: "treat ╲ fire" }));
	cuts.forEach((c, i) => grid.appendChild(el("div", { class: "abl-mx-colh clickable", title: `firing event #${c} · injection ${i + 1}/${cuts.length} — click to compare all treatments (all replicates) at this injection in /tf`, text: `#${i + 1}`,
		onclick: () => openTfCompare(treatments.flatMap((t) => familyNames(existing, kind, c, t.tag)), "no runs at this injection yet") })));
	for (const t of treatments) {
		grid.appendChild(el("div", { class: "abl-mx-rowh clickable", title: `${levelLabel(t)} · ${t.tag} — click to compare this level (all replicates) across injections in /tf`, text: levelLabel(t),
			onclick: () => openTfCompare(cuts.flatMap((cut) => familyNames(existing, kind, cut, t.tag)), "no runs for this level yet") }));
		for (const cut of cuts) grid.appendChild(matrixCell(kind, t, cut, existing));
	}
	return el("div", { class: "abl-kind" },
		el("div", { class: "abl-kind-head" },
			el("span", { class: "abl-kind-name", text: kind }),
			el("span", { class: "abl-kind-dim", text: `${treatments.length} treatments × ${cuts.length} injection${cuts.length === 1 ? "" : "s"} · ${createdRuns} run${createdRuns === 1 ? "" : "s"} · n=${desired}` }),
			launchKindBtn,
			openKind),
		grid);
}

async function renderMatrices() {
	if (rendering || !isOpen() || drag) return; // never rebuild the grid mid drag-select
	rendering = true;
	const body = $("abl-board-body");
	const sub = $("abl-board-sub");
	initBoardDrag();
	try {
		if (!cfg.baseRun) { body.replaceChildren(el("div", { class: "abl-empty", text: "pick a base run to configure an ablation sweep" })); return; }
		if (timeline.loading) { body.replaceChildren(el("div", { class: "abl-empty", text: "loading base timeline…" })); return; }
		if (!cfg.slot || !cfg.model) { body.replaceChildren(el("div", { class: "abl-empty", text: "pick a scene + model" })); return; }
		if (!timeline.kinds.length) {
			body.replaceChildren(el("div", { class: "abl-empty", text: timeline.error ? `couldn't read base timeline: ${timeline.error}` : "this base cell has no scene-context step kinds — root plans, overall_bbox, and image_prompt are excluded. Pick a cell that has run." }));
			return;
		}
		const { map: existing, otherCells } = await fetchExisting(cfg.baseRun);
		// Per-kind level set: coordinate gives bbox steps all selected modes and
		// non-bbox steps their collapsed input-only reps; a kind whose set is empty
		// (e.g. only output-only coord modes picked, on a non-bbox kind) is skipped.
		const treatmentsFor = (kind) => expandExperimentForKind(cfg.experiment, cfg, kind);
		const kinds = activeKinds().filter((k) => treatmentsFor(k).length);
		if (!kinds.length) {
			body.replaceChildren(el("div", { class: "abl-empty", text: "no cells for this experiment — pick at least one non-baseline level (coordinate: a bbox step for the output-frame levels, or any scene-context step for the input-only levels)." }));
			return;
		}
		const desired = Math.max(cfg.reps, 1);
		let cells = 0, createdRuns = 0, minReps = Infinity;
		for (const kind of kinds) for (const t of treatmentsFor(kind)) for (const cut of columnsFor(kind, existing)) {
			cells += 1;
			const n = familyRepInfo(existing, kind, cut, t.tag).size;
			createdRuns += n;
			minReps = Math.min(minReps, n);
		}
		// While the queue runs, queueProgress() owns the sub-line — don't clobber it.
		if (sub && !queueActive) sub.textContent = `${cfg.baseRun} · ${cfg.slot} · ${cfg.model} — ${createdRuns} run${createdRuns === 1 ? "" : "s"} across ${cells} cells · n=${desired} · min ${cells ? (minReps === Infinity ? 0 : minReps) : 0}/cell`;
		// A variant belongs to the cell it launched from; if this base has variants
		// at OTHER cells (e.g. launched from the wizard on a different scene), point
		// there so they're never "lost".
		const banner = otherCells.size
			? el("div", { class: "abl-hint" },
				el("span", { text: `${existing.size ? "also — " : ""}ablation variants at other cells:` }),
				...[...otherCells.entries()].map(([k, n]) => {
					const [s, m] = k.split("\u0000");
					return el("button", { class: "abl-start", title: `switch the board to ${s} / ${m}`, onclick: () => switchCell(s, m) }, `${s} / ${m} (${n}) →`);
				}))
			: null;
		body.replaceChildren(...(banner ? [banner] : []), ...kinds.map((kind) => kindMatrix(kind, treatmentsFor(kind), existing)));
		updateSelBar(body); // re-sync the selection bar (cells re-applied .sel above)
	} catch (e) {
		body.replaceChildren(el("div", { class: "abl-empty", text: `board error: ${e.message}` }));
	} finally {
		rendering = false;
	}
}

// ---- config bar -----------------------------------------------------------

function renderConfig() {
	const host = $("abl-board-config");
	if (!host) return;
	const sel = (opts, value, onchange, title) =>
		el("select", { title, onchange: (e) => onchange(e.target.value) },
			opts.map((o) => el("option", { value: o, text: o, ...(o === value ? { selected: "" } : {}) })));

	// The experiment picker — ONE axis sweeps at a time (its levels are the matrix
	// rows). Modular: registering an axis in EXPERIMENT_AXES + a case in
	// `experimentControls` below is all it takes to add a future experiment.
	const expSel = el("select", { title: "experiment axis to sweep — only its levels vary; every other axis stays at baseline (no cross-product)",
		onchange: (e) => { cfg.experiment = e.target.value; renderConfig(); renderMatrices(); } },
		EXPERIMENT_AXES.map((ax) => el("option", { value: ax.id, text: ax.label, title: ax.hint, ...(cfg.experiment === ax.id ? { selected: "" } : {}) })));
	const experimentControls = () => {
		if (cfg.experiment === "shuffle") {
			return SHUFFLE_METHODS.filter((m) => m !== "order").map((m) => el("label", { class: "abl-chk", title: METHOD_HINT[m] },
				el("input", { type: "checkbox", ...(cfg.methods.has(m) ? { checked: "" } : {}),
					onchange: (e) => { e.target.checked ? cfg.methods.add(m) : cfg.methods.delete(m); renderMatrices(); } }), m));
		}
		if (cfg.experiment === "xml") {
			return [el("span", { class: "m-hint", text: "baseline (tags on) vs stripped — no levels to configure" })];
		}
		if (cfg.experiment === "coordinate") {
			return [...COORD_MODES.filter((c) => c.id !== "baseline").map((c) => el("label", { class: "abl-chk", title: c.hint },
				el("input", { type: "checkbox", ...(cfg.coordModes.has(c.id) ? { checked: "" } : {}),
					onchange: (e) => { e.target.checked ? cfg.coordModes.add(c.id) : cfg.coordModes.delete(c.id); renderMatrices(); } }), c.label)),
				el("span", { class: "m-hint", text: "baseline (L/G→L) = base cell (no fork) · bbox steps vary input + output · other steps vary input only (levels collapse, so no double-count)" })];
		}
		if (cfg.experiment === "schema") {
			return [...SCHEMA_MODES.filter((s) => s.id !== "baseline").map((s) => el("label", { class: "abl-chk", title: s.hint },
				el("input", { type: "checkbox", ...(cfg.schemaModes.has(s.id) ? { checked: "" } : {}),
					onchange: (e) => { e.target.checked ? cfg.schemaModes.add(s.id) : cfg.schemaModes.delete(s.id); renderMatrices(); } }), s.label)),
				el("span", { class: "m-hint", text: "baseline (soft-JSON) = base cell (no fork) · every scene-context step re-rendered in the chosen format" })];
		}
		if (cfg.experiment === "gravity") {
			return [...GRAVITY_MODES.map((g) => el("label", { class: "abl-chk", title: g.hint },
				el("input", { type: "checkbox", ...(cfg.gravityModes.has(g.id) ? { checked: "" } : {}),
					onchange: (e) => { e.target.checked ? cfg.gravityModes.add(g.id) : cfg.gravityModes.delete(g.id); renderMatrices(); } }), g.label)),
				el("span", { class: "m-hint", text: "all levels launched · `no tags` is the anchor the graph subtracts (the base cell keeps the real VERY_IMPORTANT tags, so it's not the baseline)" })];
		}
		if (cfg.experiment === "attend") {
			return [el("input", { class: "abl-text", type: "text", value: cfg.attendTargets.join(", "), placeholder: "e.g. the frog statue",
				title: "comma-separate targets; each adds an attend variant beside the no-directive baseline",
				oninput: (e) => { cfg.attendTargets = e.target.value.split(",").map((s) => s.trim()).filter(Boolean); renderMatrices(); } })];
		}
		return [el("input", { class: "abl-text", type: "text", value: cfg.distractors.filter((d) => d).join(","), placeholder: "2,4",
			title: "distractor counts (comma-separated) — not yet wired server-side",
			oninput: (e) => { const v = e.target.value.split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => Number.isFinite(n) && n > 0); cfg.distractors = v.length ? v : [0]; renderMatrices(); } })];
	};

	const lastN = el("input", { class: "abl-num", type: "number", min: "1", max: "10", value: String(cfg.lastN),
		title: "temporal injections per kind = its last-N firings",
		onchange: (e) => { cfg.lastN = Math.max(1, Number(e.target.value) || 3); renderMatrices(); } });
	const reps = el("input", { class: "abl-num", type: "number", min: "1", max: "20", value: String(cfg.reps),
		title: "replicates per cell (n) — each is an independent re-run of the SAME config with a bumped RNG seed, so it's a fresh draw that feeds the data + statistical tests (raises certainty). rep 1 keeps the original run name.",
		onchange: (e) => { cfg.reps = Math.max(1, Number(e.target.value) || 1); renderMatrices(); } });
	const temp = el("input", { class: "abl-num", type: "number", min: "0", max: "2", step: "0.1", value: String(cfg.temperature),
		title: "generation temperature for the re-inferred treated step. >0 makes replicates DIFFER (independent draws that actually raise certainty) even for the deterministic shuffle methods (order/distance/raytrace); 0 = greedy (replicates would be identical). Combined with the per-replicate seed for reproducibility.",
		onchange: (e) => { const v = Number(e.target.value); cfg.temperature = Number.isFinite(v) ? Math.max(0, Math.min(2, v)) : 0.7; renderMatrices(); } });
	// Label field: a DROPDOWN of the labels already STARTED at this cell (default =
	// an existing one) so a launched experiment maps onto the matrix — the label is
	// baked into every variant run name, so a free-text mismatch would show every
	// cell as "not created". "➕ new label…" reveals a text box to start a fresh one.
	const NEW_LABEL = "\u0000new";
	const labelSel = el("select", { class: "abl-text",
		title: "which started experiment (label) to show — its variants map onto the matrix. Pick the one you launched under, or “➕ new label…” to start a new experiment.",
		onchange: (e) => {
			if (e.target.value === NEW_LABEL) { cfg._newLabel = true; cfg.label = ""; renderConfig(); renderMatrices(); return; }
			cfg._newLabel = false; cfg.label = e.target.value;
			cfg.labelByCell[cellKeyOf()] = cfg.label; // remember this model's chosen run (decoupled per model)
			applyExperimentFromLabel(cfg.label); // selecting a label → its experiment + levels
			renderConfig(); // experiment axis / levels changed → rebuild the controls
			renderMatrices();
		} },
		...cellLabels.map((x) => el("option", { value: x.label, text: `${x.label || "(no label)"} · ${x.derived ? x.derived.experiment + " · " : ""}${x.n}`,
			...(!cfg._newLabel && x.label === cfg.label ? { selected: "" } : {}) })),
		el("option", { value: NEW_LABEL, text: "\u2795 new label\u2026", ...(cfg._newLabel ? { selected: "" } : {}) }));
	const labelBox = cfg._newLabel
		? el("input", { class: "abl-text", type: "text", value: cfg.label, placeholder: "new label",
			title: "name this new experiment — it's baked into the variant run names",
			oninput: (e) => { cfg.label = e.target.value.trim(); renderMatrices(); } })
		: null;
	// Remove the selected run's variants at THIS cell (granular delete, model-scoped).
	const removeBtn = (!cfg._newLabel && cellLabels.some((x) => x.label === cfg.label))
		? el("button", { class: "abl-start danger", title: `remove (delete from disk) all runs for label "${cfg.label || "(no label)"}" at ${cfg.slot} / ${cfg.model}`, onclick: removeLabel }, "🗑 remove")
		: null;

	// Scene select: populated scenes first + a marker so it's obvious which ran.
	const scenesSorted = [...baseCells.slots].sort((a, b) => (sceneHasData(b) ? 1 : 0) - (sceneHasData(a) ? 1 : 0) || (a < b ? -1 : 1));
	const nWithData = baseCells.slots.filter(sceneHasData).length;
	const sceneSel = el("select", { title: `base scene — ${nWithData}/${baseCells.slots.length} have run`,
		onchange: (e) => { cfg.slot = e.target.value; if (cellEvents(cfg.slot, cfg.model) <= 0) cfg.model = bestModelFor(cfg.slot); refreshCell(); } },
		scenesSorted.map((id) => el("option", { value: id, text: sceneHasData(id) ? id : `${id} · empty`, ...(id === cfg.slot ? { selected: "" } : {}) })));
	// Model select: mark which models have run for the current scene.
	const modelSel = el("select", { title: "base model", onchange: (e) => { cfg.model = e.target.value; refreshCell(); } },
		baseCells.models.map((m) => el("option", { value: m, text: cellEvents(cfg.slot, m) > 0 ? m : `${m} · empty`, ...(m === cfg.model ? { selected: "" } : {}) })));

	host.replaceChildren(
		el("span", { class: "abl-cfg-group" }, el("span", { class: "abl-cfg-lbl", text: "base" }),
			sel(baseRunNames(), cfg.baseRun, (v) => selectBase(v), "base run to fork from")),
		el("span", { class: "abl-cfg-group" }, el("span", { class: "abl-cfg-lbl", text: "scene" }), sceneSel),
		el("span", { class: "abl-cfg-group" }, el("span", { class: "abl-cfg-lbl", text: "model" }), modelSel),
		el("span", { class: "abl-cfg-group" }, el("span", { class: "abl-cfg-lbl", text: "label" }), labelSel, ...(labelBox ? [labelBox] : []), ...(removeBtn ? [removeBtn] : [])),
		el("span", { class: "abl-cfg-group" }, el("span", { class: "abl-cfg-lbl", text: "last-N" }), lastN),
		el("span", { class: "abl-cfg-group" }, el("span", { class: "abl-cfg-lbl", text: "reps (n)" }), reps),
		el("span", { class: "abl-cfg-group" }, el("span", { class: "abl-cfg-lbl", text: "temp" }), temp),
		el("span", { class: "abl-cfg-sep" }),
		el("span", { class: "abl-cfg-group" }, el("span", { class: "abl-cfg-lbl", text: "experiment" }), expSel),
		el("span", { class: "abl-cfg-group" }, ...experimentControls()),
		el("span", { class: "abl-cfg-sep" }),
		el("span", { class: "abl-cfg-group" },
			el("span", { class: "abl-pill" }, el("span", { class: "dot notcreated" }), "not created"),
			el("span", { class: "abl-pill" }, el("span", { class: "dot idle" }), "idle"),
			el("span", { class: "abl-pill" }, el("span", { class: "dot running" }), "running"),
			el("span", { class: "abl-pill" }, el("span", { class: "dot done" }), "done"),
			el("span", { class: "abl-pill" }, el("span", { class: "dot error" }), "error")),
		el("span", { style: "margin-left:auto" }),
		el("button", { class: "abl-start", title: `fill every cell up to the reps (n=${Math.max(cfg.reps, 1)}) target — launches only the missing replicates`,
			onclick: launchAllToN }, `▶ launch all → n=${Math.max(cfg.reps, 1)}`),
		el("button", { class: "abl-start", title: "resume EVERY stalled variant across the sweep in batch (errored + idle/interrupted, in place) — e.g. after a server restart or a fixed issue",
			onclick: rerunErrored }, "↻ resume stalled"),
		el("button", { class: "abl-start primary", title: "raise the FLOOR by one: run every cell until the minimum replicate count is m+1 (adds one independent replicate to the laggards). Click repeatedly to keep raising certainty evenly.",
			onclick: runAllPlusOne }, "▶ run all +1"),
		el("button", { class: "abl-start danger", title: "stop the execution queue — jobs already on the GPU finish + reappear on refresh; pending queued items are dropped",
			onclick: stopQueue }, "⏹ stop"));
}

// Collect the replicate launches needed to bring every intended cell up to `target`.
async function collectToTarget(targetFn) {
	const { map: existing } = await fetchExisting(cfg.baseRun);
	const families = [];
	for (const kind of activeKinds()) {
		const treatments = expandExperimentForKind(cfg.experiment, cfg, kind);
		for (const t of treatments) for (const cut of columnsFor(kind, existing)) {
			families.push({ kind, t, cut, byRep: familyRepInfo(existing, kind, cut, t.tag) });
		}
	}
	const target = targetFn(families);
	const out = [];
	for (const f of families) for (const k of missingReps(f.byRep, target)) out.push(variantParams(f.kind, f.t, f.cut, k));
	return { out, target, families };
}

// "launch all → n": every cell filled to the reps (n) target.
async function launchAllToN() {
	const desired = Math.max(cfg.reps, 1);
	const { out } = await collectToTarget(() => desired);
	if (!out.length) { toast(`every cell already has n=${desired} replicates`); return; }
	if (!window.confirm(`Queue ${out.length} replicate(s) to bring every cell to n=${desired}? ${MAX_ABLATION_BATCH} run at a time, the rest wait.`)) return;
	launchMany(out);
}

// "run all +1": raise the minimum replicate count by one across all cells — only
// runs a cell that is AT the current minimum m, bringing every cell to ≥ m+1. So
// certainty grows evenly one replicate at a time (per the m+1 rule).
async function runAllPlusOne() {
	const { out, target, families } = await collectToTarget((fs) => (fs.length ? Math.min(...fs.map((f) => f.byRep.size)) : 0) + 1);
	if (!families.length) { toast("no cells to run"); return; }
	if (!out.length) { toast("everything is already at the target — raise reps (n) or add more"); return; }
	if (!window.confirm(`Run all cells up to n=${target} (raising the floor by one)? Queues ${out.length} replicate(s) — ${MAX_ABLATION_BATCH} at a time, the rest wait.`)) return;
	launchMany(out);
}

// ---- lifecycle ------------------------------------------------------------

export async function openAblationBoard() {
	const panel = $("ablation-board");
	if (!panel) return;
	panel.classList.add("open");
	if (!cfg.baseRun) {
		const names = baseRunNames();
		const initial = state.run && names.includes(state.run) ? state.run : names[0];
		if (initial) { await selectBase(initial); } else { renderConfig(); renderMatrices(); }
	} else {
		await refreshCell();
	}
	clearInterval(pollTimer);
	pollTimer = setInterval(() => { if (isOpen()) renderMatrices(); }, 4000);
}

export function closeAblationBoard() {
	$("ablation-board")?.classList.remove("open");
	clearInterval(pollTimer);
	pollTimer = null;
	queueCancel = true; // stop feeding the execution queue when the board is closed
}

export function initAblationBoard() {
	$("btn-ablation-board")?.addEventListener("click", openAblationBoard);
	$("abl-board-close")?.addEventListener("click", closeAblationBoard);
	$("abl-board-refresh")?.addEventListener("click", () => { toast("refreshing…"); renderMatrices(); });
	$("abl-board-reset")?.addEventListener("click", resetAll);
	window.addEventListener("keydown", (e) => { if (e.key === "Escape" && isOpen()) { e.preventDefault(); closeAblationBoard(); } });
}
