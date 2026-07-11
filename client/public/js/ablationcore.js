// Shared ablation naming + launch logic, used by BOTH the guided wizard
// (ablation.js) and the unified board (ablationboard.js) so the two never drift
// — a variant's run name must be identical wherever it's created, or the board
// can't map an existing run back onto its matrix cell.
//
// An ablation forks a BASE run at the last-N firings of a step kind, applies a
// treatment (scene-shuffle method, XML tags on/off, distractors), and re-infers
// only the treated tail via the existing branch → save_run_from_branches →
// resume flow (no new server surface).

import { api } from "./api.js";

export const SHUFFLE_METHODS = ["order", "random", "distance", "raytrace"];

// Ablations are restricted to OPEN-SOURCE models (gemma / qwen). Closed models
// (gemini-flash, gpt, opus, …) are never offered or launched — the guard that
// stops a sweep from silently firing e.g. gemini-flash cells. Shared by the
// wizard + board so neither can pick a "dangerous" model.
// Max variants launched per action (wizard "launch all" + board). Each launched
// variant spawns a server-side re-inference; firing a whole sweep at once
// saturates the LLM transport (mass `llm.transport_retry`, results never landing
// cleanly), so both entry points cap here and hold the rest back.
export const MAX_ABLATION_BATCH = 8;

export const ALLOWED_MODEL = (m) => /gemma|qwen/i.test(String(m));
export const preferredModel = (models) => (models || []).find((m) => /gemma/i.test(m)) || (models || []).find(ALLOWED_MODEL) || null;
export const METHOD_HINT = {
	order: "current insertion order (baseline)",
	random: "seeded shuffle",
	distance: "nearest-first by bbox center",
	raytrace: "most-visible-first (base: falls back to order)",
};
// The one step kind never swept (the scene root plan).
export const ROOT_KINDS = new Set(["zone_plan_root"]);

// Step kinds whose prompt actually renders SCENE_CONTEXT (from the minglun
// prompt set) — the only ones where a scene-context ablation (shuffle /
// distractor) is meaningful. Root variants (…_root), overall_bbox, and
// image_prompt have no scene context, so they're never offered. Ablations are
// limited to these.
export const SCENE_CONTEXT_KINDS = new Set([
	"zone_plan", "zone_decompose", "anchor_decompose", "encapsulating_decompose",
	"negative_space_decompose", "object_bbox_batch", "child_bbox_batch", "next_object",
]);
export const hasSceneContext = (kind) => SCENE_CONTEXT_KINDS.has(kind);

// The coordinate-frame experiment: a SINGLE enumerated axis (never two crossed
// toggles), so it fans out to EXACTLY these conditions. Each level maps to how
// the two bbox solvers SEE coordinates in scene context (input representation)
// and which frame they must EMIT (output frame). The L->G case is intentionally
// excluded — local-only input gives the model no absolute anchor to produce a
// global answer. `tag` is the run-name fragment (baseline carries none) and MUST
// match the server's app/ablation/config._COORD_TAGS so a variant maps to one cell.
export const COORD_MODES = [
	{ id: "baseline", label: "L/G → L (baseline)", input: "both", output: "local", tag: "", hint: "both frames shown · emit local (current behaviour)" },
	{ id: "lg2g", label: "L/G → G", input: "both", output: "global", tag: "crd-LG2G", hint: "both frames shown · emit GLOBAL" },
	{ id: "l2l", label: "L → L", input: "local", output: "local", tag: "crd-L2L", hint: "local-only shown · emit local" },
	{ id: "g2g", label: "G → G", input: "global", output: "global", tag: "crd-G2G", hint: "global-only shown · emit GLOBAL" },
	{ id: "g2l", label: "G → L", input: "global", output: "local", tag: "crd-G2L", hint: "global-only shown · emit local" },
];
export const coordTagOf = (id) => (COORD_MODES.find((c) => c.id === id) || COORD_MODES[0]).tag;
// The INPUT representation a coord mode SHOWS in scene context (`both` | `local` |
// `global`). A non-bbox step only ever PERCEIVES this — the emit/output frame is
// invisible to it — so two modes with the same input render an IDENTICAL scene
// context there (baseline ≡ lg2g, and g2g ≡ g2l).
export const coordInputOf = (id) => (COORD_MODES.find((c) => c.id === id) || COORD_MODES[0]).input;

// The two coordinate-EMITTING solvers. The OUTPUT (emit) frame axis exists ONLY
// here, so only these kinds vary along it (all 5 modes are distinct). Every OTHER
// scene-context kind still varies along the INPUT axis (which coordinates it is
// SHOWN), but the output frame is invisible to it — so its modes collapse onto the
// distinct input reps (see coordModesForKind). A coordinate experiment therefore
// forks bbox AND non-bbox kinds; they just get different level sets.
export const BBOX_KINDS = new Set(["object_bbox_batch", "child_bbox_batch"]);

// The canonical coord mode representing each INPUT rep on a non-bbox step (where
// the output frame is irrelevant): both→baseline, local→l2l, global→g2g.
const COORD_INPUT_REP = { both: "baseline", local: "l2l", global: "g2g" };

// The coord modes actually DISTINCT for a step of `kind`, given the user-selected
// non-baseline modes. The bbox solvers keep every selected mode (input AND output
// both matter). A non-bbox step collapses the selection onto its distinct INPUT
// reps — mapping each mode to its canonical representative and dropping `baseline`
// (= the base cell, never forked) — so its input-only variant is launched (and
// later counted) EXACTLY ONCE per {local, global}, never doubled: lg2g would
// re-render the baseline's scene context, and g2l would re-render g2g's.
export function coordModesForKind(kind, selectedModes) {
	const sel = [];
	for (const id of selectedModes || []) if (id && id !== "baseline") sel.push(id);
	if (BBOX_KINDS.has(kind)) return sel;
	const out = [];
	const seen = new Set();
	for (const id of sel) {
		const rep = COORD_INPUT_REP[coordInputOf(id)] || "baseline";
		if (rep === "baseline" || seen.has(rep)) continue;
		seen.add(rep);
		out.push(rep);
	}
	return out;
}

// The scene-context SCHEMA experiment: a SINGLE enumerated axis — how the treated
// step's scene context is SERIALIZED (soft-JSON / XML / prose), holding the
// information constant. INPUT-only (the bbox OUTPUT stays JSON) and applied
// UNIFORMLY to every scene-context kind, so — unlike coord — there is no per-kind
// collapse. `baseline` (soft-JSON) IS the base cell, so it carries no run-name tag
// and is never forked. Tags MUST match the server's app/ablation/config._SCHEMA_TAGS
// so a variant maps to one cell whether launched from the wizard or the board.
export const SCHEMA_MODES = [
	{ id: "baseline", label: "soft-JSON (baseline)", tag: "", hint: "current production format · the base cell (not forked)" },
	{ id: "xml", label: "XML", tag: "sch-XML", hint: "structured tags, same info" },
	{ id: "prose", label: "prose", tag: "sch-PROSE", hint: "natural language, same info" },
];
export const schemaTagOf = (id) => (SCHEMA_MODES.find((s) => s.id === id) || SCHEMA_MODES[0]).tag;

// The XML-gravity experiment: strip the treated step's instruction-block tags,
// then slide a NEUTRAL closing </prompt> across the block's word-count QUARTERS
// (opening <prompt> fixed at the block start). Measures how a tag pulls attention
// to nearby text vs distance. These are the LAUNCHED levels — `none` (tags off) is
// the comparison anchor the graph subtracts; q1..q4 place the closing tag after
// each quarter (q4 = full wrap). Unlike coord/schema the untouched base cell is
// NOT the baseline (it carries the real, semantically-loaded VERY_IMPORTANT tags),
// so every level here is forked. Tags MUST match server config._GRAVITY_TAGS.
export const GRAVITY_MODES = [
	{ id: "none", label: "no tags (anchor)", tag: "grav-none", hint: "instruction XML tags stripped — the baseline the others subtract" },
	{ id: "q1", label: "close after Q1", tag: "grav-Q1", hint: "<prompt> opens at the block start · </prompt> after the 1st word-count quarter" },
	{ id: "q2", label: "close after Q2", tag: "grav-Q2", hint: "</prompt> after the 2nd quarter" },
	{ id: "q3", label: "close after Q3", tag: "grav-Q3", hint: "</prompt> after the 3rd quarter" },
	{ id: "q4", label: "close after Q4 (full wrap)", tag: "grav-Q4", hint: "</prompt> at the block end — full wrap" },
];
export const gravityTagOf = (id) => (GRAVITY_MODES.find((g) => g.id === id) || {}).tag || "";

// ---- unified ablation-axis registry ----------------------------------------
// ONE descriptor per experiment axis, so every READ surface (the /tf compute
// matrix + the analysis factor picker) derives its axis + levels from HERE rather
// than hardcoding each one. Adding a new dimension so it's recognized everywhere =
// (1) a treatment field + gate on the server, (2) a `*_MODES` list, and (3) an
// entry below — the /tf matrix columns, run-axis detection, and the analysis
// factor then light up automatically (no more "?" columns for a new axis).
//
// Descriptors read the STORED ablation treatment (run.json's `treatment`:
// `coord_mode` / `schema_mode` / `shuffle_method` / `xml_tags` / `attend_target` /
// `distractors`) — the shape the /tf side reads back. `field` + `baseline` drive
// detection; `levelKey(t)` → the matrix column key; `levelLabel(key)` /
// `levelRank(key)` → its display + order; `factor` is the analysis-workspace
// compare axis (null = not a comparable factor: free-form attend / unwired
// distractors); `baselineIsCell` = the baseline is the un-forked base cell (so the
// /tf comparison reuses its already-computed rows instead of a launched variant).
const _rankOf = (modes) => { const m = new Map(modes.map((x, i) => [x.id, i])); return (k) => (m.has(k) ? m.get(k) : 99); };
const _COORD_IN = { both: "LG", local: "L", global: "G" };
const _COORD_OUT = { local: "L", global: "G" };
// Compact coordinate condition (L2L / LG2G / …) from a coord_mode id.
export const coordCompact = (id) => { const c = COORD_MODES.find((x) => x.id === id); return c ? `${_COORD_IN[c.input] || "?"}2${_COORD_OUT[c.output] || "?"}` : (id || "?"); };
const _schemaLabel = (id) => (SCHEMA_MODES.find((s) => s.id === id) || {}).label || id || "?";
const _METHOD_RANK = (k) => { const i = SHUFFLE_METHODS.indexOf(k); return i < 0 ? 99 : i; };
const _GRAVITY_ORDER = ["baseline", "none", "q1", "q2", "q3", "q4"];
const _gravityLabel = (id) => (id === "baseline" ? "baseline" : (GRAVITY_MODES.find((g) => g.id === id) || {}).label || id || "?");
const _gravityRank = (k) => { const i = _GRAVITY_ORDER.indexOf(k); return i < 0 ? 99 : i; };

export const ABLATION_AXES = [
	{ id: "coordinate", label: "coordinate frame", field: "coord_mode", baseline: "baseline", factor: "coord", modes: COORD_MODES, baselineIsCell: true,
		levelKey: (t) => t.coord_mode || "baseline", levelLabel: coordCompact, levelRank: _rankOf(COORD_MODES) },
	{ id: "schema", label: "scene-context schema", field: "schema_mode", baseline: "baseline", factor: "schema", modes: SCHEMA_MODES, baselineIsCell: true,
		levelKey: (t) => t.schema_mode || "baseline", levelLabel: _schemaLabel, levelRank: _rankOf(SCHEMA_MODES) },
	{ id: "gravity", label: "XML gravity check", field: "gravity_mode", baseline: "baseline", factor: "gravity", modes: GRAVITY_MODES, baselineIsCell: false,
		levelKey: (t) => t.gravity_mode || "baseline", levelLabel: _gravityLabel, levelRank: _gravityRank },
	{ id: "shuffle", label: "scene-context order", field: "shuffle_method", baseline: "order", factor: "method", modes: null,
		levelKey: (t) => t.shuffle_method || "order", levelLabel: (k) => k, levelRank: _METHOD_RANK },
	{ id: "xml", label: "XML tags", field: "xml_tags", baseline: true, factor: "xml", modes: null,
		levelKey: (t) => (t.xml_tags === false ? "off" : "on"), levelLabel: (k) => (k === "off" ? "xml off" : "xml on"), levelRank: (k) => (k === "off" ? 1 : 0) },
	{ id: "attend", label: "attention steering", field: "attend_target", baseline: "", factor: null, modes: null,
		levelKey: (t) => (t.attend_target ? String(t.attend_target) : ""), levelLabel: (k) => (k ? `→ ${String(k).slice(0, 12)}` : "baseline"), levelRank: (k) => (k ? 1 : 0) },
	{ id: "distractors", label: "distractors", field: "distractors", baseline: 0, factor: null, modes: null,
		levelKey: (t) => String(Number(t.distractors) || 0), levelLabel: (k) => (Number(k) ? `d${k}` : "baseline"), levelRank: (k) => Number(k) || 0 },
];
const _AXIS_BY_ID = new Map(ABLATION_AXES.map((a) => [a.id, a]));
const _AXIS_BY_FACTOR = new Map(ABLATION_AXES.filter((a) => a.factor).map((a) => [a.factor, a]));
export const ablationAxis = (id) => _AXIS_BY_ID.get(id) || null;
export const ablationAxisByFactor = (factor) => _AXIS_BY_FACTOR.get(factor) || null;
// The analysis-workspace compare factors, in registry order (coord, schema, method, xml).
export const ABLATION_FACTORS = ABLATION_AXES.filter((a) => a.factor).map((a) => a.factor);

// Is a STORED treatment non-baseline on this axis?
function _axisActive(ax, t) { const v = t ? t[ax.field] : undefined; return v != null && v !== ax.baseline; }
// The axis a single treatment varies (first non-baseline in registry order), or null.
export function axisOfTreatment(t) { for (const ax of ABLATION_AXES) if (_axisActive(ax, t)) return ax; return null; }
// The axis a SET of variant treatments shares. Launches are disentangled (one axis
// each), so the highest-priority non-baseline axis seen wins — a shuffle run's own
// `order` baseline replicate (all-default) then doesn't hide the real axis.
export function axisOfTreatments(treatments) {
	let best = null, bestRank = Infinity;
	for (const t of treatments || []) {
		const ax = axisOfTreatment(t); if (!ax) continue;
		const r = ABLATION_AXES.indexOf(ax); if (r < bestRank) { best = ax; bestRank = r; }
	}
	return best;
}
// The {key,label,rank} a treatment sits at on `ax` — one matrix column coordinate.
export function levelOfTreatment(ax, t) { const key = ax.levelKey(t); return { key, label: ax.levelLabel(key), rank: ax.levelRank(key) }; }

export function slug(text, limit) {
	const cleaned = String(text).replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[-.]+|[-.]+$/g, "");
	return (cleaned.slice(0, limit) || "x");
}

// XML-off treatment: drop lines that are ONLY an open/close tag (the minglun
// prompts author tags on their own lines), keeping the content between them.
export function stripXmlLines(text) {
	return String(text)
		.split("\n")
		.filter((line) => !/^\s*<\/?[A-Za-z][\w-]*>\s*$/.test(line))
		.join("\n");
}

// The filename-safe treatment descriptor embedded in a variant's run name.
// `attend` is the attention-steering target (empty = the no-directive baseline).
export function treatmentTag({ method, xml, distractors, seed, attend, coord, schema, gravity }) {
	return [
		method || "order",
		coord ? (coordTagOf(coord) || null) : null,  // baseline coord → "" → omitted
		schema ? (schemaTagOf(schema) || null) : null,  // baseline schema → "" → omitted
		gravity ? (gravityTagOf(gravity) || null) : null,  // baseline gravity → "" → omitted
		xml ? null : "noxml",
		attend ? `att-${slug(attend, 12)}` : null,
		distractors ? `d${distractors}` : null,
		seed ? `s${seed}` : null,
	].filter(Boolean).join("_") || "order";
}

// Attention-steering directive: append an explicit "focus on X" instruction to
// the step's user prompt. The BASELINE is the same variant with attend="" (no
// directive), so pinning the two in /tf and reading X's attention answers
// "does telling the model to attend to X actually raise its attention to X?".
export function appendAttendInstruction(userTemplate, target, xml) {
	const body = `For THIS step specifically, pay heightened, focused attention to: ${target}. Prioritize and weight ${target} above the other scene-context elements when reasoning.`;
	const block = xml ? `\n\n<attention_directive>\n${body}\n</attention_directive>` : `\n\n${body}`;
	return `${userTemplate}${block}`;
}

// The prompt override for one treated step: strip XML (if xml off) and/or append
// the attend directive. Returns {} when nothing changes (falls back to the base
// prompt). Requires the base template map when xml-off or attend is in play.
export function buildStepOverride(templates, kind, { xml, attend }) {
	const st = templates ? templates[kind] : null;
	if (!st) return {};
	let system = st.system;
	let user = st.user;
	let changed = false;
	if (!xml) { system = stripXmlLines(system); user = stripXmlLines(user); changed = true; }
	if (attend) { user = appendAttendInstruction(user, attend, xml); changed = true; }
	return changed ? { [kind]: { system, user } } : {};
}

// The treatment matrix (one point = one per-fork variant), de-duped by tag. A
// seed only matters when it drives randomness (random shuffle / distractor
// pick), so it's collapsed to 0 otherwise to avoid identical-but-named runs.
export function expandTreatments({ methods, xmlOn, xmlOff, distractors, seed, attendTargets }) {
	const xmls = [];
	if (xmlOn) xmls.push(true);
	if (xmlOff) xmls.push(false);
	const dcounts = distractors && distractors.length ? distractors : [0];
	// "" (baseline, no directive) is always included so every attend variant has
	// a matching baseline to compare against.
	const attends = (attendTargets && attendTargets.length) ? ["", ...attendTargets] : [""];
	const out = [];
	const seen = new Set();
	for (const method of methods) {
		for (const xml of xmls) {
			for (const d of dcounts) {
				for (const attend of attends) {
					const s = (method === "random" || d) ? (seed || 0) : 0;
					const tag = treatmentTag({ method, xml, distractors: d, seed: s, attend });
					if (seen.has(tag)) continue;
					seen.add(tag);
					out.push({ method, xml, distractors: d, seed: s, attend, tag });
				}
			}
		}
	}
	return out;
}

// The neutral point every experiment shares: the L/G->L, insertion-order,
// xml-on, no-distractor, no-attend baseline. Each experiment layers its axis's
// NON-baseline levels on top of this, so two experiments never cross-multiply.
export const BASELINE_TREATMENT = Object.freeze({
	method: "order", xml: true, distractors: 0, seed: 0, attend: "", coord: "baseline", schema: "baseline", gravity: "baseline",
});

// The single experimentation axes the wizard offers — one per launch.
export const EXPERIMENT_AXES = [
	{ id: "coordinate", label: "coordinate frame", hint: "how steps SEE coordinates (input, every step) + how the bbox solvers EMIT them (output); bbox steps vary all 5 levels, other steps vary input only" },
	{ id: "schema", label: "scene-context schema", hint: "serialize the treated step's scene context as soft-JSON / XML / prose (same info, different structure) — every scene-context step" },
	{ id: "gravity", label: "XML gravity check", hint: "strip the instruction tags, then slide a neutral </prompt> across the block's word-count quarters — how a tag pulls attention to nearby text vs distance" },
	{ id: "shuffle", label: "scene-context order", hint: "reorder the scene items shown to the treated step" },
	{ id: "xml", label: "XML tags", hint: "keep vs strip the prompt's XML section tags" },
	{ id: "attend", label: "attention steering", hint: "append an explicit 'focus on X' directive" },
	{ id: "distractors", label: "distractors", hint: "inject irrelevant objects into the scene (not yet wired server-side)" },
];

// DECOUPLED, one-axis-at-a-time expansion: vary ONLY `axis`, pinning every other
// axis at baseline. Returns the shared baseline plus one point per selected level
// of the chosen axis (de-duped by tag) — never a cartesian product, so launching
// experiment "coordinate" and experiment "shuffle" stay independent (no A×D
// permutations). The wizard picks exactly ONE axis per launch, which is what
// guarantees the disentanglement.
export function expandExperiment(axis, cfg) {
	const pts = [];
	const seen = new Set();
	const push = (over) => {
		const t = { ...BASELINE_TREATMENT, ...over };
		const tag = treatmentTag(t);
		if (seen.has(tag)) return;  // collapses a level that equals baseline
		seen.add(tag);
		pts.push({ ...t, tag });
	};
	// The shared baseline (order · xml-on · L/G→L · soft-JSON) is the comparison
	// anchor. For the COORDINATE and SCHEMA axes the baseline IS the base cell
	// (already computed — no fork, no recompute; the /tf view reuses it, see
	// report.ablAutoPeers), so we emit only the non-baseline conditions there.
	// coordinate/schema reuse the base cell as their baseline (no fork); gravity's
	// anchor is the launched `none` level (the base cell keeps the real, loaded VII
	// tags), so it too emits ONLY explicit levels — never the shared baseline point.
	if (axis !== "coordinate" && axis !== "schema" && axis !== "gravity") push({});
	if (axis === "shuffle") {
		for (const m of cfg.methods || []) push({ method: m, seed: m === "random" ? (cfg.seed || 0) : 0 });
	} else if (axis === "xml") {
		push({ xml: false });
	} else if (axis === "coordinate") {
		for (const id of cfg.coordModes || []) if (id && id !== "baseline") push({ coord: id });
	} else if (axis === "schema") {
		for (const id of cfg.schemaModes || []) if (id && id !== "baseline") push({ schema: id });
	} else if (axis === "gravity") {
		for (const id of cfg.gravityModes || []) if (id && id !== "baseline") push({ gravity: id });
	} else if (axis === "attend") {
		for (const t of cfg.attendTargets || []) if (t) push({ attend: t });
	} else if (axis === "distractors") {
		for (const d of cfg.distractors || []) if (d) push({ distractors: d, seed: cfg.seed || 0 });
	}
	return pts;
}

// The treatments for ONE step kind under `axis`. Identical to expandExperiment for
// every axis EXCEPT `coordinate`, whose level set is KIND-dependent: a non-bbox
// step only varies along the INPUT axis, so its selected modes are collapsed onto
// their distinct input reps (see coordModesForKind) — this is what stops an
// input-only variant from being launched (and later counted) twice. Every
// per-kind render/launch path MUST use this (not expandExperiment) so bbox and
// non-bbox kinds each get their correct, disentangled level set.
export function expandExperimentForKind(axis, cfg, kind) {
	if (axis !== "coordinate") return expandExperiment(axis, cfg);
	return expandExperiment("coordinate", { ...cfg, coordModes: coordModesForKind(kind, cfg.coordModes || []) });
}

// The deterministic, filesystem-safe variant run name. MUST stay in lockstep
// across the wizard + board (both call this) so existing runs map onto cells.
//
// The name encodes the FULL cell identity — base, slot, model, kind@cut, treatment
// tag, replicate — because a variant is anchored to ONE (slot, model) cell and the
// run NAME is the collision key (`save_run_from_branches` 409s on a duplicate
// name). Without slot+model, the same kind@cut-tag on two models / scenes would map
// to the same run name and the second launch would be BLOCKED (and be
// indistinguishable). `rep` ≥ 2 appends `-r{rep}` — the family = the name without it.
export function variantName({ baseRun, slot, model, label, kind, cut, tag, rep }) {
	const cellPart = [slot, model].filter(Boolean).map((s) => slug(s, 16)).join("-");
	const labelPart = label ? `${slug(label, 20)}-` : "";
	const repPart = rep && rep > 1 ? `-r${rep}` : "";
	return `${slug(baseRun, 28)}__abl-${cellPart ? `${cellPart}-` : ""}${labelPart}${slug(kind, 18)}@${cut}-${slug(tag, 24)}${repPart}`;
}

// The PRE-disambiguation name (no slot/model). Kept ONLY so the board still maps
// variants launched before slot+model were added to the name — new launches always
// use variantName().
export function legacyVariantName({ baseRun, label, kind, cut, tag, rep }) {
	const labelPart = label ? `${slug(label, 20)}-` : "";
	const repPart = rep && rep > 1 ? `-r${rep}` : "";
	return `${slug(baseRun, 36)}__abl-${labelPart}${slug(kind, 20)}@${cut}-${slug(tag, 28)}${repPart}`;
}

// The base cell's prompt snapshot, keyed by step → {system, user}. Needed to
// build the XML-off override. Cached per base run.
const _templateCache = new Map();
export async function loadBaseTemplates(baseRun) {
	if (_templateCache.has(baseRun)) return _templateCache.get(baseRun);
	const tmpl = await api.promptTemplates(baseRun);
	const map = {};
	for (const s of tmpl.steps || []) map[s.step] = { system: s.system || "", user: s.user || "" };
	_templateCache.set(baseRun, map);
	return map;
}
export function clearTemplateCache(baseRun) {
	if (baseRun) _templateCache.delete(baseRun); else _templateCache.clear();
}

// Fork the base at (kind@cut), apply the treatment override, save as the
// auto-named variant run (prefix shared, meshes hardlinked), then resume it to
// re-infer the treated tail. `templates` is the base prompt map (from
// loadBaseTemplates) — required only for the XML-off override.
export async function launchVariant({ baseRun, slot, model, label, lastN, kind, cut, method, xml, distractors, seed, attend, coord, schema, gravity, tag, rep, temperature }, templates) {
	// The coordinate + schema + gravity axes are applied SERVER-SIDE (coord/schema at
	// pipeline bind; gravity rewrites the treated step's instruction-block tags inside
	// llm.call_llm, keyed on gravity_mode), so they need no client prompt override —
	// only xml/attend rewrite the snapshot here.
	const overrides = buildStepOverride(templates, kind, { xml, attend });
	const r = rep && rep > 1 ? rep : 1;
	const name = variantName({ baseRun, slot, model, label, kind, cut, tag: tag ?? treatmentTag({ method, xml, distractors, seed, attend, coord, schema, gravity }), rep: r });
	// Independent replicate: bump the RNG seed by (rep−1) so the random shuffle (and
	// any seeded sampling) DIFFERS per replicate. rep 1 keeps the base seed. Without
	// this a re-run would be identical and would only fake-narrow the CI. (Note:
	// deterministic methods — order/distance/raytrace — still produce the same
	// prompt regardless of seed, so their replicates only vary if generation itself
	// is stochastic.)
	const effSeed = (seed || 0) + (r - 1);
	try {
		const br = await api.createBranch(baseRun, slot, model, { event_index: cut, step: kind, overrides });
		const bid = br?.branch?.id;
		const saved = await api.saveRunFromBranches({
			name, base_run: baseRun, overrides, branches: bid ? [bid] : [],
			version_label: `abl${label ? ":" + label : ""}:${kind}@${cut}:${tag ?? ""}${r > 1 ? `:r${r}` : ""}`,
			ablation: {
				// slot/model record the exact cell this variant forked from + ran in,
				// so the board can read its status / open its scene at the RIGHT cell
				// (variant names don't encode the cell). `replicate` groups a cell's
				// re-runs into one family for the board counts + the /tf statistics.
				label: label || null, base_run: baseRun, slot, model, target_step_kind: kind, cut, last_n: lastN, replicate: r,
				treatment: { shuffle_method: method, xml_tags: xml, section_order: "default", distractors, seed: effSeed, attend_target: attend || null, coord_mode: coord || "baseline", schema_mode: schema || "baseline", gravity_mode: gravity || "baseline", temperature: (temperature != null && temperature !== "" ? Number(temperature) : null) },
			},
		});
		// The server is authoritative for a variant's run id: ablation variants fold
		// under their base (`<base>/ablations/<experiment>/<variant>`), so `saved.current`
		// is the real (nested) id — resume/poll THAT, not the flat `name` we proposed,
		// or the resume 404s (the flat name isn't a run anymore).
		const runId = (saved && saved.current) || name;
		await api.resume(runId, slot, model);
		if (bid) { try { await api.branchDiscard(bid); } catch { /* best-effort cleanup */ } }
		return { ok: true, name: runId };
	} catch (e) {
		const msg = String(e.message);
		return { ok: false, name, msg: msg.includes("409") ? "already exists" : msg };
	}
}
