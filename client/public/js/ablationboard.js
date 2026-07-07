// The unified ablation control board. "⚗ board" opens a full-screen screen that
// is the ONE place to configure, launch, monitor, and reset ablations.
//
// It reads the base cell's firing timeline (/tf-steps) and renders, per step
// kind, the full intended matrix: rows = treatments (shuffle × xml), columns =
// temporal injections (the last-N firings of that kind). Every cell is a live
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
	expandTreatments, variantName, loadBaseTemplates, launchVariant,
} from "./ablationcore.js";

let pollTimer = null;
let rendering = false;

// Board config (the intended sweep). Seeded from the active run on first open.
const cfg = {
	baseRun: null, slot: null, model: null, label: "",
	lastN: 3, methods: new Set(SHUFFLE_METHODS), xmlOn: true, xmlOff: true,
	distractors: [0], seed: 1,
	// Attention-steering probe: targets to inject a "focus on X" directive for.
	// Each adds an attend variant next to its no-directive baseline.
	attendTargets: [],
};
// The selected base cell's available slots/models (may differ from the active
// run), plus its firing timeline.
let baseCells = { run: null, slots: [], models: [], defaultModel: null };
let timeline = { key: null, kinds: [], firingsByKind: {}, loading: false };

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

// Coarse cell status from a raw events.jsonl (last lifecycle marker wins). Avoids
// hydrating the whole variant run just to read one cell's status.
function statusFromEvents(text) {
	let lifecycle = null;
	let anyEvent = false;
	for (const line of text.split("\n")) {
		if (!line.trim()) continue;
		let ev;
		try { ev = JSON.parse(line); } catch { continue; }
		anyEvent = true;
		switch (ev.kind) {
			case "run.start": case "run.resume": lifecycle = "running"; break;
			case "run.done": case "ablation.complete": lifecycle = "done"; break;
			case "run.error": lifecycle = "error"; break;
			case "run.paused": lifecycle = "paused"; break;
			default: break;
		}
	}
	// A committed lifecycle wins; otherwise a non-empty log clearly ran (treat as
	// running, never grey/idle), and only a truly empty cell is idle.
	if (lifecycle) return lifecycle;
	return anyEvent ? "running" : "idle";
}

const statusCache = new Map(); // variant name -> { slot, model, state }
const isTerminal = (s) => s === "done" || s === "error";

// Read a variant's cell status straight off the artifact server — a plain file
// read, NO run hydration (which would load all slot×model cells and OOM on big
// runs). The cell is the variant's recorded (slot, model), not the board's.
async function variantStatus(name, slot, model) {
	const rel = `${encodeURIComponent(name)}/${encodeURIComponent(slot)}/${encodeURIComponent(model)}/events.jsonl`;
	try {
		const res = await fetch(`${SERVER_URL}/artifacts/${rel}`, { cache: "no-store" });
		if (!res.ok) return { slot, model, state: "idle" };
		const text = await res.text();
		return { slot, model, state: text.trim() ? statusFromEvents(text) : "idle" };
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
	await ensureTimeline();
	renderConfig();
	renderMatrices();
}

// Point the board at a different (slot, model) — used by the "variants at other
// cells" hint so wizard launches on another scene are one click away.
function switchCell(s, m) {
	cfg.slot = s;
	cfg.model = m;
	ensureTimeline().then(() => { renderConfig(); renderMatrices(); });
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
			// Only kinds whose prompt renders scene context — a scene-context
			// ablation is meaningless on root plans / overall_bbox / image_prompt.
			if (!k || !hasSceneContext(k)) continue;
			if (!firingsByKind[k]) { firingsByKind[k] = []; kinds.push(k); }
			firingsByKind[k].push(s.event_index);
		}
		timeline = { key, kinds, firingsByKind, loading: false };
	} catch (e) {
		timeline = { key, kinds: [], firingsByKind: {}, loading: false, error: e.message };
	}
}

// ---- existing variants ----------------------------------------------------

async function fetchExisting(base) {
	const data = await api.runs();
	const list = Array.isArray(data) ? data : (data.runs ?? []);
	const prefix = `${slug(base, 36)}__abl-`;
	const ofBase = list.filter((r) => {
		if (!r || typeof r !== "object") return false;
		const abl = r.ablation;
		return (abl && abl.base_run) ? abl.base_run === base : String(r.name).startsWith(prefix);
	});
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
	await pool(toFetch, 6, async (r) => { const c = cellOf(r); statusCache.set(r.name, await variantStatus(r.name, c.slot, c.model)); });
	const map = new Map();
	for (const r of mine) { const c = cellOf(r); map.set(r.name, statusCache.get(r.name) || { slot: c.slot, model: c.model, state: "idle" }); }
	return { map, otherCells };
}

// ---- launching ------------------------------------------------------------

async function ensureTemplates() {
	try { return await loadBaseTemplates(cfg.baseRun); }
	catch (e) { toast(`couldn't load base prompts: ${e.message}`, "err"); return null; }
}

function variantParams(kind, t, cut) {
	return { baseRun: cfg.baseRun, slot: cfg.slot, model: cfg.model, label: cfg.label, lastN: cfg.lastN,
		kind, cut, method: t.method, xml: t.xml, distractors: t.distractors, seed: t.seed, attend: t.attend, tag: t.tag };
}

// Launch the WHOLE list, but keep only MAX_ABLATION_BATCH RUNNING at once: each
// queue slot launches a variant, waits for it to actually finish (its treated
// step commits + the run stops at run.done), then pulls the next. So 24 queued →
// 8 running, 16 waiting — no request flood, no manual re-launching.
let launching = false;
let launchCancel = false;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitForVariant(name, slot, model) {
	const deadline = Date.now() + 6 * 60 * 1000; // safety cap so a stuck run can't wedge the queue
	while (!launchCancel && Date.now() < deadline) {
		const st = (await variantStatus(name, slot, model)).state;
		statusCache.set(name, { slot, model, state: st });
		if (st === "done" || st === "error") return;
		await sleep(2500);
	}
}

async function launchMany(list) {
	if (!list.length) { toast("nothing to launch"); return; }
	if (!ALLOWED_MODEL(cfg.model || "")) { toast(`ablations are limited to open-source models (gemma / qwen) — got "${cfg.model}"`, "err"); return; }
	if (launching) { toast("a launch queue is already running — wait or ⏹ stop", "err"); return; }
	const tmpls = await ensureTemplates();
	if (!tmpls) return;
	launching = true;
	launchCancel = false;
	const total = list.length;
	let done = 0, ok = 0, fail = 0;
	const sub = $("abl-board-sub");
	const progress = () => { if (sub) sub.textContent = `queue: ${done}/${total} done · up to ${MAX_ABLATION_BATCH} running${launchCancel ? " · stopping" : ""}`; };
	progress();
	try {
		await pool(list, MAX_ABLATION_BATCH, async (p) => {
			if (launchCancel) return;
			const r = await launchVariant(p, tmpls);
			if (!r.ok) { fail += 1; done += 1; progress(); return; }
			ok += 1;
			// Gate the queue on REAL completion so only ~MAX_ABLATION_BATCH run at once.
			await waitForVariant(r.name, p.slot, p.model);
			done += 1;
			progress();
			if (isOpen()) renderMatrices();
		});
	} finally {
		launching = false;
	}
	toast(`launch queue ${launchCancel ? "stopped" : "done"}: ${ok} run${fail ? `, ${fail} failed` : ""}`, fail ? "err" : "ok");
	renderMatrices();
}

async function resetAll() {
	const data = await api.runs();
	const abl = (Array.isArray(data) ? data : (data.runs ?? [])).filter((r) => String(r?.name).includes("__abl-"));
	if (!abl.length) { toast("no ablation runs to delete"); return; }
	if (!window.confirm(`Delete ALL ${abl.length} ablation run(s)? They are stopped and removed from disk.`)) return;
	let ok = 0, fail = 0;
	await pool(abl.map((r) => r.name), 3, async (n) => { try { await api.deleteRun(n); ok += 1; } catch { fail += 1; } });
	toast(`deleted ${ok}${fail ? `, ${fail} failed` : ""}`, fail ? "err" : "ok");
	renderMatrices();
}

// ---- matrix rendering -----------------------------------------------------

const nameFor = (kind, cut, tag) => variantName({ baseRun: cfg.baseRun, label: cfg.label, kind, cut, tag });

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
function openVariantCard(kind, t, cut, cell, existing) {
	const name = nameFor(kind, cut, t.tag);
	const stateName = cell ? cell.state : "notcreated";
	const slot = cell?.slot || cfg.slot;
	const model = cell?.model || cfg.model;
	const baseline = nameFor(kind, cut, "order");
	const row = (k, v) => el("div", { class: "abl-card-row" }, el("span", { class: "abl-card-k", text: k }), el("span", { class: "abl-card-v", text: v }));
	openModal(`${kind} · injection @${cut}`, (close) => {
		const sceneHref = `/tf?${new URLSearchParams({ run: name, slot, model }).toString()}`;
		const cmpHref = tfHref(cfg.baseRun, slot, model, baseline !== name && existing?.has(baseline) ? [baseline, name] : [name]);
		const actions = [el("button", { text: "close", onclick: close })];
		const body = [
			el("div", { class: "abl-card-name", text: name }),
			row("treatment", t.tag),
			t.attend ? row("attend →", t.attend) : null,
			row("cell", `${slot} · ${model}`),
			row("status", stateName),
		].filter(Boolean);
		if (stateName === "notcreated") {
			body.push(el("div", { class: "m-hint", text: "not launched yet — forks the base here and re-infers this treated step." }));
			actions.push(el("button", { class: "primary", text: "▶ launch this variant",
				onclick: async () => { close(); await launchMany([variantParams(kind, t, cut)]); } }));
		} else {
			body.push(el("div", { class: "abl-card-actions" },
				el("a", { class: "abl-open", href: sceneHref, target: "_blank", rel: "noopener", title: "open this variant's 3D scene in /tf" }, "scene ↗"),
				el("a", { class: "abl-open", href: cmpHref, target: "_blank", rel: "noopener", title: "compare this variant's attention against its baseline in /tf" }, "compare vs baseline ↗")));
			if (startable(cell)) {
				actions.push(el("button", { class: "primary", text: stateName === "error" ? "↻ relaunch" : "▶ start",
					onclick: async () => { close(); try { await startVariant(name, cell); toast(`started ${name}`); } catch (e) { toast(`failed: ${e.message}`, "err"); } renderMatrices(); } }));
			}
		}
		return { body, actions };
	});
}

function matrixCell(kind, t, cut, cell, existing) {
	const name = nameFor(kind, cut, t.tag);
	const stateName = cell ? cell.state : "notcreated";
	return el("div", {
		class: `abl-cell ${stateName}`,
		title: `${name}\n${stateName} · click for details`,
		onclick: () => openVariantCard(kind, t, cut, cell, existing),
	}, stateName === "notcreated" ? el("span", { class: "abl-plus", text: "+" }) : el("span", { class: `dot ${stateName}` }));
}

function kindMatrix(kind, treatments, existing) {
	const cuts = injectionsFor(kind);
	const missing = [];
	const created = [];
	for (const t of treatments) for (const cut of cuts) {
		const name = nameFor(kind, cut, t.tag);
		if (existing.has(name)) created.push(name); else missing.push(variantParams(kind, t, cut));
	}
	const launchKindBtn = missing.length
		? el("button", { class: "abl-start", title: `launch the ${missing.length} not-yet-created variant(s) of this kind`,
			onclick: () => { if (window.confirm(`Queue ${missing.length} missing variant(s) for ${kind}? (${MAX_ABLATION_BATCH} run at a time, the rest wait)`)) launchMany(missing); } },
			`▶ launch ${missing.length} missing`)
		: null;
	const openKind = el("a", {
		class: "abl-open", target: "_blank", rel: "noopener",
		href: created.length ? tfHref(cfg.baseRun, cfg.slot, cfg.model, created) : "#",
		title: created.length ? `compare this kind's ${created.length} created run(s) in /tf` : "launch runs first, then compare",
		...(created.length ? {} : { style: "opacity:.4;pointer-events:none" }),
	}, "↗ /tf");

	const grid = el("div", { class: "abl-matrix", style: `grid-template-columns: 168px repeat(${cuts.length}, 52px)` });
	grid.appendChild(el("div", { class: "abl-mx-corner", title: "rows = treatments · columns = temporal injection · click a header to compare that row/column in /tf", text: "treat ╲ fire" }));
	cuts.forEach((c, i) => grid.appendChild(el("div", { class: "abl-mx-colh clickable", title: `firing event #${c} · injection ${i + 1}/${cuts.length} — click to compare all treatments at this injection in /tf`, text: `#${i + 1}`,
		onclick: () => openTfCompare(treatments.map((t) => nameFor(kind, c, t.tag)).filter((n) => existing.has(n)), "no runs at this injection yet") })));
	for (const t of treatments) {
		grid.appendChild(el("div", { class: "abl-mx-rowh clickable", title: `${t.tag} — click to compare this treatment across injections in /tf`, text: t.tag,
			onclick: () => openTfCompare(cuts.map((cut) => nameFor(kind, cut, t.tag)).filter((n) => existing.has(n)), "no runs for this treatment yet") }));
		for (const cut of cuts) grid.appendChild(matrixCell(kind, t, cut, existing.get(nameFor(kind, cut, t.tag)), existing));
	}
	return el("div", { class: "abl-kind" },
		el("div", { class: "abl-kind-head" },
			el("span", { class: "abl-kind-name", text: kind }),
			el("span", { class: "abl-kind-dim", text: `${treatments.length} treatments × ${cuts.length} injection${cuts.length === 1 ? "" : "s"} · ${created.length} run` }),
			launchKindBtn,
			openKind),
		grid);
}

async function renderMatrices() {
	if (rendering || !isOpen()) return;
	rendering = true;
	const body = $("abl-board-body");
	const sub = $("abl-board-sub");
	try {
		if (!cfg.baseRun) { body.replaceChildren(el("div", { class: "abl-empty", text: "pick a base run to configure an ablation sweep" })); return; }
		if (timeline.loading) { body.replaceChildren(el("div", { class: "abl-empty", text: "loading base timeline…" })); return; }
		if (!cfg.slot || !cfg.model) { body.replaceChildren(el("div", { class: "abl-empty", text: "pick a scene + model" })); return; }
		if (!timeline.kinds.length) {
			body.replaceChildren(el("div", { class: "abl-empty", text: timeline.error ? `couldn't read base timeline: ${timeline.error}` : "this base cell has no scene-context step kinds — root plans, overall_bbox, and image_prompt are excluded. Pick a cell that has run." }));
			return;
		}
		const { map: existing, otherCells } = await fetchExisting(cfg.baseRun);
		const treatments = expandTreatments(cfg);
		let total = 0, made = 0;
		for (const kind of timeline.kinds) for (const t of treatments) for (const cut of injectionsFor(kind)) {
			total += 1;
			if (existing.has(variantName({ baseRun: cfg.baseRun, label: cfg.label, kind, cut, tag: t.tag }))) made += 1;
		}
		if (sub) sub.textContent = `${cfg.baseRun} · ${cfg.slot} · ${cfg.model} — ${made}/${total} launched`;
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
		body.replaceChildren(...(banner ? [banner] : []), ...timeline.kinds.map((kind) => kindMatrix(kind, treatments, existing)));
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

	const treatmentChecks = SHUFFLE_METHODS.map((m) =>
		el("label", { class: "abl-chk", title: METHOD_HINT[m] },
			el("input", { type: "checkbox", ...(cfg.methods.has(m) ? { checked: "" } : {}),
				onchange: (e) => { e.target.checked ? cfg.methods.add(m) : cfg.methods.delete(m); renderMatrices(); } }), m));

	const xmlOn = el("label", { class: "abl-chk", title: "keep prompt XML tags" },
		el("input", { type: "checkbox", ...(cfg.xmlOn ? { checked: "" } : {}), onchange: (e) => { cfg.xmlOn = e.target.checked; renderMatrices(); } }), "xml");
	const xmlOff = el("label", { class: "abl-chk", title: "strip prompt XML tags" },
		el("input", { type: "checkbox", ...(cfg.xmlOff ? { checked: "" } : {}), onchange: (e) => { cfg.xmlOff = e.target.checked; renderMatrices(); } }), "no-xml");

	const lastN = el("input", { class: "abl-num", type: "number", min: "1", max: "10", value: String(cfg.lastN),
		title: "temporal injections per kind = its last-N firings",
		onchange: (e) => { cfg.lastN = Math.max(1, Number(e.target.value) || 3); renderMatrices(); } });
	const label = el("input", { class: "abl-text", type: "text", value: cfg.label, placeholder: "label (optional)",
		title: "distinguishes this sweep from other ablations off the same base",
		oninput: (e) => { cfg.label = e.target.value.trim(); renderMatrices(); } });
	const attend = el("input", { class: "abl-text", type: "text", value: cfg.attendTargets.join(", "), placeholder: "e.g. the frog statue",
		title: "attention-steering probe: injects a “pay attention to X” directive into each treated step. comma-separate targets; each adds an attend variant beside its no-directive baseline. compare X's attention across the two in /tf.",
		oninput: (e) => { cfg.attendTargets = e.target.value.split(",").map((s) => s.trim()).filter(Boolean); renderMatrices(); } });

	// Scene select: populated scenes first + a marker so it's obvious which ran.
	const scenesSorted = [...baseCells.slots].sort((a, b) => (sceneHasData(b) ? 1 : 0) - (sceneHasData(a) ? 1 : 0) || (a < b ? -1 : 1));
	const nWithData = baseCells.slots.filter(sceneHasData).length;
	const sceneSel = el("select", { title: `base scene — ${nWithData}/${baseCells.slots.length} have run`,
		onchange: (e) => { cfg.slot = e.target.value; if (cellEvents(cfg.slot, cfg.model) <= 0) cfg.model = bestModelFor(cfg.slot); ensureTimeline().then(() => { renderConfig(); renderMatrices(); }); } },
		scenesSorted.map((id) => el("option", { value: id, text: sceneHasData(id) ? id : `${id} · empty`, ...(id === cfg.slot ? { selected: "" } : {}) })));
	// Model select: mark which models have run for the current scene.
	const modelSel = el("select", { title: "base model", onchange: (e) => { cfg.model = e.target.value; ensureTimeline().then(renderMatrices); } },
		baseCells.models.map((m) => el("option", { value: m, text: cellEvents(cfg.slot, m) > 0 ? m : `${m} · empty`, ...(m === cfg.model ? { selected: "" } : {}) })));

	host.replaceChildren(
		el("span", { class: "abl-cfg-group" }, el("span", { class: "abl-cfg-lbl", text: "base" }),
			sel(baseRunNames(), cfg.baseRun, (v) => selectBase(v), "base run to fork from")),
		el("span", { class: "abl-cfg-group" }, el("span", { class: "abl-cfg-lbl", text: "scene" }), sceneSel),
		el("span", { class: "abl-cfg-group" }, el("span", { class: "abl-cfg-lbl", text: "model" }), modelSel),
		el("span", { class: "abl-cfg-group" }, label),
		el("span", { class: "abl-cfg-group" }, el("span", { class: "abl-cfg-lbl", text: "last-N" }), lastN),
		el("span", { class: "abl-cfg-sep" }),
		el("span", { class: "abl-cfg-group" }, el("span", { class: "abl-cfg-lbl", text: "shuffle" }), ...treatmentChecks),
		el("span", { class: "abl-cfg-group" }, xmlOn, xmlOff),
		el("span", { class: "abl-cfg-sep" }),
		el("span", { class: "abl-cfg-group" }, el("span", { class: "abl-cfg-lbl", text: "attend→" }), attend),
		el("span", { class: "abl-cfg-sep" }),
		el("span", { class: "abl-cfg-group" },
			el("span", { class: "abl-pill" }, el("span", { class: "dot notcreated" }), "not created"),
			el("span", { class: "abl-pill" }, el("span", { class: "dot idle" }), "idle"),
			el("span", { class: "abl-pill" }, el("span", { class: "dot running" }), "running"),
			el("span", { class: "abl-pill" }, el("span", { class: "dot done" }), "done"),
			el("span", { class: "abl-pill" }, el("span", { class: "dot error" }), "error")),
		el("span", { style: "margin-left:auto" }),
		el("button", { class: "abl-start", title: "launch every not-yet-created variant across all kinds",
			onclick: launchAllMissing }, "▶ launch all missing"));
}

async function launchAllMissing() {
	const { map: existing } = await fetchExisting(cfg.baseRun);
	const treatments = expandTreatments(cfg);
	const missing = [];
	for (const kind of timeline.kinds) for (const t of treatments) for (const cut of injectionsFor(kind)) {
		if (!existing.has(variantName({ baseRun: cfg.baseRun, label: cfg.label, kind, cut, tag: t.tag }))) missing.push(variantParams(kind, t, cut));
	}
	if (!missing.length) { toast("everything is already launched"); return; }
	if (!window.confirm(`Queue ${missing.length} variant(s) across ${timeline.kinds.length} kind(s)? ${MAX_ABLATION_BATCH} run at a time and the rest wait in line — no need to relaunch.`)) return;
	launchMany(missing);
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
		renderConfig();
		await ensureTimeline();
		renderMatrices();
	}
	clearInterval(pollTimer);
	pollTimer = setInterval(() => { if (isOpen()) renderMatrices(); }, 4000);
}

export function closeAblationBoard() {
	$("ablation-board")?.classList.remove("open");
	clearInterval(pollTimer);
	pollTimer = null;
	launchCancel = true; // stop feeding the launch queue when the board is closed
}

export function initAblationBoard() {
	$("btn-ablation-board")?.addEventListener("click", openAblationBoard);
	$("abl-board-close")?.addEventListener("click", closeAblationBoard);
	$("abl-board-refresh")?.addEventListener("click", () => { toast("refreshing…"); renderMatrices(); });
	$("abl-board-reset")?.addEventListener("click", resetAll);
	window.addEventListener("keydown", (e) => { if (e.key === "Escape" && isOpen()) { e.preventDefault(); closeAblationBoard(); } });
}
