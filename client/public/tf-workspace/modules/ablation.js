// Ablation view for the analysis workspace. You FIRST choose a RUN (one launched
// experiment); only then does it scan + load that run's variants and render the
// graphs GROUPED BY the axis that run varied (coordinate frame L2L/LG2G/G2G/G2L,
// scene-context schema soft-JSON/XML/prose, shuffle method, or xml on/off). Each
// level gets a distinct color (legend). Off
// each variant's treated-step attention:
//   1) attribute profile spider (one polygon per level); for the coordinate axis a
//      SECOND version merges local+global origin into one axis (compare G=L=L+G).
//   2) structure-vs-content — for EACH attribute, one bar per level, each bar split
//      into 3 segments (context / frame / content) by length-normalized attention
//      density (mass ÷ tokens); hue = level, segment shade = role. Roles come from
//      buckets.attr_role (server-classified per attribute's real serialized form).
//   3) word/token-type comparison — GROUPED BARS (one bar per level per token
//      class), attention share by token class. Shown LAST.
// Variants are scoped by the SAME header selectors (region → treated node, step →
// treated kind/cut) on top of the chosen run.

import { el } from "../../js/ui.js";
import { api } from "../../js/api.js";
import { $, state, ALL, entityKindLabel, bumpLoad, ATTR_AXIS_ORDER, COLORS } from "./state.js";
import { pool, SCENE_CONTEXT_KINDS, poolComponents, poolKindTotals, contextPoints } from "./data.js";
import { spiderChart, chartHost, svgEl, fontScale, chartLegend, hexA, pctFmt, escTip } from "./charts.js";
import { COORD_MODES, SCHEMA_MODES } from "../../js/ablationcore.js";

const METHOD_COLORS = { order: "#7aa2f7", random: "#e0a94a", distance: "#6bd96e", raytrace: "#b46aff", attend: "#ff6b9d" };
const METHOD_ORDER = ["order", "random", "distance", "raytrace", "attend"];
const XML_ON = "#6bd96e", XML_OFF = "#ff6b9d";
// Word/token-type classes (matches the backend classifier + tf-legacy): a display
// order + per-class color so the token-type spider labels its OWN axes (not the
// attribute abbreviations). Structural punctuation shares a blue ramp.
const TYPE_ORDER = ["number", "spatial", "content", "entity_name", "function", "bracket", "separator", "quote", "operator", "whitespace", "other"];
const TYPE_COLORS = {
	number: "#4af0e0", spatial: "#e0a94a", content: "#f7768e", entity_name: "#5be584", function: "#c98bdb",
	bracket: "#7aa2f7", separator: "#4d7fd6", quote: "#a9c7ff", operator: "#33507e", whitespace: "#3a3f4b", other: "#9098a6",
};
// Short axis labels so the ring text doesn't clip (the legend keeps the full name).
const TYPE_ABBR = { number: "num", spatial: "spat", content: "cont", entity_name: "ent", function: "fn", bracket: "brkt", separator: "sep", quote: "qt", operator: "op", whitespace: "ws", other: "oth" };
// Full, un-abbreviated class names (for the grouped-bar x-axis) + a one-line
// explanation with examples for each — surfaced as a hover tooltip on the label.
// Mirrors the backend classifier (server/app/attention/semantic.py:classify_tokens).
const TYPE_FULL = {
	number: "number", spatial: "spatial", content: "content", entity_name: "entity name",
	function: "function", bracket: "bracket", separator: "separator", quote: "quote",
	operator: "operator", whitespace: "whitespace", other: "other",
};
const TYPE_DESC = {
	number: "Numeric literals — the coordinate & dimension values the model writes, e.g. 0.5, -3, 12.75.",
	spatial: "Spatial-relation words that place things, e.g. above, below, on, inside, left, right, between, near.",
	content: "Open-class descriptive words (nouns / adjectives / verbs), e.g. table, wooden, place, large.",
	entity_name: "Tokens inside a scene entity's name / id, e.g. sofa, oak_table.",
	function: "Closed-class grammar words (the glue), e.g. the, a, of, to, with, is, and, for.",
	bracket: "JSON structural brackets / braces / parens: { } [ ] ( ).",
	separator: "JSON separators: the : between a key and its value, and , between items.",
	quote: "Quote marks around JSON keys & string values: \" ' `.",
	operator: "Operator symbols, e.g. = (and other residual symbol tokens).",
	whitespace: "Whitespace tokens — spaces, newlines, and indentation.",
	other: "Everything else — sentence punctuation & stray symbols, e.g. . ! ? # / *.",
};
const typeColor = (c) => TYPE_COLORS[c] || "#9098a6";
const typeLabel = (c) => TYPE_ABBR[c] || c;
const typeFull = (c) => TYPE_FULL[c] || String(c).replace(/_/g, " ");
const typeRank = (c) => { const i = TYPE_ORDER.indexOf(c); return i < 0 ? 99 : i; };
const MAX_VARIANTS = 40; // default cap on the heavy agg fetches (~4MB each)
const HARD_MAX = 100;    // ceiling even when "load all" is on (avoid exhausting memory)
let _loadAll = false;    // user opted to load beyond the default cap

// --- experiment axis + levels (the view is driven by the selected RUN) -----------
// A run's variants all share ONE experiment axis (launches are disentangled); the
// graphs group/color by that axis's LEVELS — coordinate frame (L2L / LG2G / …), the
// shuffle method, or xml on/off — never a fixed, irrelevant method/xml split.
const COORD_LEVEL_COLOR = { baseline: "#7aa2f7", lg2g: "#e0a94a", l2l: "#6bd96e", g2g: "#b46aff", g2l: "#ff6b9d" };
const _CIN = { both: "LG", local: "L", global: "G" }, _COUT = { local: "L", global: "G" };
const coordCompact = (id) => { const c = COORD_MODES.find((x) => x.id === id); return c ? `${_CIN[c.input]}2${_COUT[c.output]}` : (id || "?"); };
const _COORD_RANK = new Map(COORD_MODES.map((c, i) => [c.id, i]));
// Scene-context SCHEMA axis: baseline (soft-JSON = the base cell) / xml / prose. A
// short level label + color + rank, mirroring the coord axis. `baseline` is the
// un-forked base cell (loaded like the coord LG2L baseline).
const SCHEMA_LEVEL_COLOR = { baseline: "#7aa2f7", xml: "#e0a94a", prose: "#6bd96e" };
const SCHEMA_SHORT = { baseline: "soft-JSON", xml: "XML", prose: "prose" };
const schemaCompact = (id) => SCHEMA_SHORT[id] || id || "?";
const _SCHEMA_RANK = new Map(SCHEMA_MODES.map((s, i) => [s.id, i]));
// XML-gravity axis: `none` (tags stripped — the subtraction anchor) + q1..q4
// (neutral </prompt> closed after each word-count quarter). Every level is a
// launched variant (the base cell keeps the real VII tags, so it is NOT a level).
const GRAVITY_LEVEL_COLOR = { none: "#7aa2f7", q1: "#e0a94a", q2: "#6bd96e", q3: "#b46aff", q4: "#ff6b9d" };
const GRAVITY_SHORT = { none: "no tag", q1: "close@Q1", q2: "close@Q2", q3: "close@Q3", q4: "close@Q4" };
const _GRAV_RANK = { none: 0, q1: 1, q2: 2, q3: 3, q4: 4 };

// Which experiment axis a set of variant rows belongs to (first non-baseline wins).
function experimentAxis(rows) {
	let coord = false, schema = false, gravity = false, method = false, xml = false;
	for (const r of rows || []) {
		if (r.coord && r.coord !== "baseline") coord = true;
		else if (r.schema && r.schema !== "baseline") schema = true;
		else if (r.gravity && r.gravity !== "baseline") gravity = true;
		else if (r.method && r.method !== "order") method = true;
		else if (r.xml === false) xml = true;
	}
	return coord ? "coord" : schema ? "schema" : gravity ? "gravity" : method ? "method" : xml ? "xml" : "method";
}
// The variant's LEVEL within `axis`: { key, label, color, rank } — the group it's
// plotted under (a coord condition / a schema format / a shuffle method / xml on-off).
function levelOf(v, axis) {
	if (axis === "coord") { const id = v.coord || "baseline"; return { key: id, label: coordCompact(id), color: COORD_LEVEL_COLOR[id] || "#7aa2f7", rank: _COORD_RANK.has(id) ? _COORD_RANK.get(id) : 99 }; }
	if (axis === "schema") { const id = v.schema || "baseline"; return { key: id, label: schemaCompact(id), color: SCHEMA_LEVEL_COLOR[id] || "#7aa2f7", rank: _SCHEMA_RANK.has(id) ? _SCHEMA_RANK.get(id) : 99 }; }
	if (axis === "gravity") { const id = v.gravity || "none"; return { key: id, label: GRAVITY_SHORT[id] || id, color: GRAVITY_LEVEL_COLOR[id] || "#7aa2f7", rank: _GRAV_RANK[id] ?? 99 }; }
	if (axis === "xml") { const on = v.xml !== false; return { key: on ? "xml \u2713" : "xml \u2717", label: on ? "xml \u2713" : "xml \u2717", color: on ? XML_ON : XML_OFF, rank: on ? 0 : 1 }; }
	const m = v.method || "order"; const i = METHOD_ORDER.indexOf(m); return { key: m, label: m, color: METHOD_COLORS[m] || "#7aa2f7", rank: i < 0 ? 99 : i };
}
const axisLabel = (axis) => axis === "coord" ? "coordinate frame" : axis === "schema" ? "scene-context schema" : axis === "gravity" ? "XML gravity (tag distance)" : axis === "xml" ? "XML on / off" : "scene order (shuffle)";

const card = (title, sub, ...body) => {
	const head = el("div", { class: "card-head" }, el("span", { class: "card-title", text: title }));
	if (sub) head.appendChild(el("span", { class: "card-sub", text: sub }));
	return el("div", { class: "card" }, head, el("div", { class: "card-body" }, ...body.filter(Boolean)));
};
const empty = (m) => el("div", { class: "empty", text: m });
const vh = (f, mn, mx) => Math.max(mn, Math.min(Math.round(window.innerHeight * f), mx));
const cellKey = () => `${state.run}:${state.slot}:${state.model}`;

// --- rank-correlation helpers (no external dep) ------------------------------
function rankArr(a) {
	const idx = a.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]);
	const r = new Array(a.length);
	let i = 0;
	while (i < idx.length) {
		let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
		const avg = (i + j) / 2 + 1;
		for (let k = i; k <= j; k++) r[idx[k][1]] = avg;
		i = j + 1;
	}
	return r;
}
function pearson(x, y) {
	const n = x.length; if (n < 2) return 0;
	const mx = x.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n;
	let sxy = 0, sxx = 0, syy = 0;
	for (let i = 0; i < n; i++) { const dx = x[i] - mx, dy = y[i] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
	return sxy / (Math.sqrt(sxx * syy) || 1);
}
const spearman = (x, y) => pearson(rankArr(x), rankArr(y));
// Normal CDF (Abramowitz & Stegun 26.2.17) + Student-t two-tailed p (normal limit
// for large df) → a Spearman ρ WITH its significance for the order↔attention read.
function normalCdf(z) {
	const x = Math.abs(z), t = 1 / (1 + 0.2316419 * x), d = 0.3989423 * Math.exp(-x * x / 2);
	const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
	return z >= 0 ? 1 - p : p;
}
function tPValueTwoTail(t, df) {
	if (!isFinite(t) || df < 1) return 1;
	const z = Math.abs(t) * (df >= 30 ? 1 : Math.sqrt(1 - 1 / (4 * df)));
	return Math.max(0, Math.min(1, 2 * (1 - normalCdf(z))));
}
// Spearman ρ + two-tailed p (t-approx on ranks). { rho, p, n }.
export function spearmanTrend(xs, ys) {
	const n = Math.min(xs.length, ys.length);
	if (n < 3) return { rho: 0, p: 1, n };
	const rho = spearman(xs.slice(0, n), ys.slice(0, n));
	const den = 1 - rho * rho;
	const t = den > 1e-12 ? rho * Math.sqrt((n - 2) / den) : (rho >= 0 ? Infinity : -Infinity);
	return { rho, p: tPValueTwoTail(t, n - 2), n };
}
// "Nice" evenly-spaced axis ticks within [lo, hi] (~target of them), with 0
// always included when the range straddles it.
function niceTicks(lo, hi, target = 5) {
	const span = (hi - lo) || 1;
	const mag = Math.pow(10, Math.floor(Math.log10(span / target)));
	const norm = span / target / mag;
	const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
	const ticks = [];
	for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) ticks.push(+t.toFixed(6));
	if (lo <= 0 && hi >= 0 && !ticks.some((t) => Math.abs(t) < 1e-9)) { ticks.push(0); ticks.sort((a, b) => a - b); }
	return ticks;
}

// --- variant discovery + loading --------------------------------------------

// Fallback discovery via the ablations index (server without /ablation-compare).
// No cache warming, so probeVariant/loadAgg fetch per variant (the original N+1) —
// functional. Scoped to the base run, so `combos` only covers this base's cells.
async function loadVariantsViaRuns() {
	const list = (await api.ablations(state.run)).variants ?? [];
	const out = []; const combos = new Map();
	for (const r of list) {
		const a = r && r.ablation;
		if (!a || a.cut == null || !SCENE_CONTEXT_KINDS.has(a.target_step_kind)) continue;
		const ck = `${a.base_run} · ${a.slot} · ${a.model}`;
		combos.set(ck, (combos.get(ck) || 0) + 1);
		if (a.slot === state.slot && a.model === state.model) {
			const t = a.treatment || {};
			out.push({ name: r.run_id || r.name, kind: a.target_step_kind, cut: Number(a.cut), label: a.label || "", coord: t.coord_mode || "baseline", schema: t.schema_mode || "baseline", gravity: t.gravity_mode || "baseline", method: t.shuffle_method || "order", xml: t.xml_tags !== false });
		}
	}
	state._ablCombos = [...combos.entries()].map(([k, n]) => ({ k, n })).sort((a, b) => b.n - a.n);
	return out;
}

async function loadVariants() {
	// ONE server call discovers this cell's variants AND batches each treated step's
	// `abl` projection (+ cross-cell combos for the empty state), replacing the old
	// /runs filter + per-variant index probe + per-variant fetch. We warm `_probe`
	// (treated ev) and `state.variantRows` (the row incl. `a`) from the response so
	// the render's probeVariant/loadAgg are cache hits — no per-variant N+1.
	let resp;
	try { resp = await api.ablationCompare(state.run, state.slot, state.model); }
	catch (e) { console.warn("[tf-workspace] ablation-compare unavailable — /runs fallback (restart the server for the fast path):", e?.message || e); return loadVariantsViaRuns(); }
	const out = [];
	for (const rv of resp.variants || []) {
		const a = rv.ablation;
		if (!a || a.cut == null || !SCENE_CONTEXT_KINDS.has(a.target_step_kind)) continue;
		const t = a.treatment || {};
		const v = { name: rv.name, kind: a.target_step_kind, cut: Number(a.cut), label: a.label || "", coord: t.coord_mode || "baseline", schema: t.schema_mode || "baseline", gravity: t.gravity_mode || "baseline", method: t.shuffle_method || "order", xml: t.xml_tags !== false };
		out.push(v);
		_probe.set(rv.name, rv.ev ?? null);
		if (rv.a) state.variantRows.set(rv.name, { ...v, ev: rv.ev, a: rv.a });
	}
	// which cells DO have variants — so the empty state can point the user there
	state._ablCombos = (resp.combos || []).map((c) => ({ k: c.k, n: c.n }));
	return out;
}

// The base run's node at a given cut (the treated region), from the base steps.
function nodeOfCut(cut) {
	const s = state.steps.find((x) => Number(x.event_index) === Number(cut));
	return s ? (s.node ?? "?") : null;
}

// Scope the variants by the header selectors:
//   region + specific step (event_index) → exactly that (kind, cut)
//   ALL region + a kind                  → that kind, any cut
//   region + ALL steps                   → any kind whose treated node = region
//   ALL + ALL                            → everything
function scopedVariants() {
	const variants = state.variants || [];
	const region = state.region, step = state.step;
	const specific = region !== ALL && step !== ALL; // step is a base event_index
	let kind = null, cut = null, node = null;
	if (specific) {
		cut = Number(step);
		const s = state.steps.find((x) => String(x.event_index) === String(step));
		kind = s ? (s.template ?? s.step) : null;
	} else if (region === ALL && step !== ALL) {
		kind = step; // step is a template kind
	}
	if (region !== ALL) node = region;
	return variants.filter((v) => {
		// RUN scope: only the selected experiment (label). null = not yet chosen.
		if (state.ablRun != null && (v.label || "") !== state.ablRun) return false;
		if (kind && v.kind !== kind) return false;
		if (specific && cut != null && v.cut !== cut) return false;
		if (node && nodeOfCut(v.cut) !== node) return false;
		return true;
	});
}

// The distinct RUN labels at this cell (for the run picker), each with its axis +
// count, named runs first. Drives which experiment the graphs show.
function cellRuns() {
	const by = new Map();
	for (const v of state.variants || []) { const l = v.label || ""; (by.get(l) || by.set(l, []).get(l)).push(v); }
	return [...by.entries()].map(([label, vs]) => ({ label, axis: experimentAxis(vs), n: vs.length }))
		.sort((a, b) => (a.label ? 0 : 1) - (b.label ? 0 : 1) || (b.n - a.n) || (a.label < b.label ? -1 : 1));
}

// PHASE 1 (cheap): a disk-scan (attention-index — no hydration, no event-log
// download) tells us which of a variant's OWN steps are computed. A variant only
// computes its re-inferred treated tail, so its last computed step IS the treated
// step. Returns that event_index, or null when the variant has no attention.
// ~10ms each, so we can scan every scoped variant.
const _probe = new Map();
async function probeVariant(v) {
	if (_probe.has(v.name)) return _probe.get(v.name);
	let ev = null;
	try {
		const idx = await api.attentionIndex(v.name, state.slot, state.model);
		const computed = [...(idx.fresh || idx.computed || []), ...(idx.stale || [])].map(Number).filter((n) => Number.isFinite(n));
		if (computed.length) ev = Math.max(...computed);
	} catch { ev = null; }
	_probe.set(v.name, ev);
	return ev;
}
// PHASE 2 (the load): the variant's rollups. Prefers the ultra-light `abl` view
// (~50KB) so we can load EVERY variant; falls back to the heavier `agg` (~4MB) on
// an older server that doesn't know `abl` (which then gets a safety cap). Cached.
let _liteMode = null; // null unknown · true = server has the light `abl` view · false = fell back to agg
async function loadAgg(v, ev) {
	if (state.variantRows.has(v.name)) return state.variantRows.get(v.name);
	let row = null, a = null;
	try {
		if (_liteMode === false) {
			a = await api.attentionGet(v.name, state.slot, state.model, ev, { view: "agg" });
		} else {
			try { a = await api.attentionGet(v.name, state.slot, state.model, ev, { view: "abl" }); _liteMode = true; }
			catch { _liteMode = false; a = await api.attentionGet(v.name, state.slot, state.model, ev, { view: "agg" }); }
		}
		if (a) row = { name: v.name, kind: v.kind, cut: v.cut, label: v.label, coord: v.coord, schema: v.schema, gravity: v.gravity, method: v.method, xml: v.xml, ev, a };
	} catch { row = null; }
	state.variantRows.set(v.name, row);
	return row;
}

// The base-cell BASELINE (coord + schema axes): L/G→L (coord) and soft-JSON
// (schema) are the base cell's OWN behaviour — NEVER launched as a variant — so we
// pull the BASE run's ALREADY-COMPUTED attention at the fork point (`cut` = the
// base event_index the variant forked from) and present it as the `baseline`
// level. Returns null (skipped) when that base step has no computed attention.
// Cached under a base-key so repaints don't re-fetch it. Uses the SAME light/heavy
// view mode the variants settled on.
async function loadBaseStep(cut) {
	const key = `__base__:${cut}`;
	if (state.variantRows.has(key)) return state.variantRows.get(key);
	const s = state.steps.find((x) => Number(x.event_index) === Number(cut));
	const kind = s ? (s.template ?? s.step) : "?";
	let row = null, a = null;
	try {
		if (_liteMode === false) a = await api.attentionGet(state.run, state.slot, state.model, cut, { view: "agg" });
		else { try { a = await api.attentionGet(state.run, state.slot, state.model, cut, { view: "abl" }); _liteMode = true; } catch { _liteMode = false; a = await api.attentionGet(state.run, state.slot, state.model, cut, { view: "agg" }); } }
		if (a) row = { name: state.run, kind, cut, label: state.ablRun || "", coord: "baseline", schema: "baseline", gravity: "baseline", method: "order", xml: true, ev: cut, a, _base: true };
	} catch { row = null; }
	state.variantRows.set(key, row);
	return row;
}

// --- token-type + structure(attr_role) roll-ups ------------------------------
// Read the compact per-segment totals shipped by the light `abl` view; fall back
// to summing the heavier `agg`/`compact` `buckets` grids when only those exist.
function typeTotals(a) {
	if (Array.isArray(a && a.token_types)) return a.token_types; // abl summary
	const b = a && a.buckets;
	if (!b || !Array.isArray(b.type)) return [];
	const names = b.type_names || [], toks = b.type_tokens || [], tot = names.map(() => 0);
	for (const row of b.type) row.forEach((v, i) => { if (i < tot.length) tot[i] += (v || 0); });
	return names.map((name, i) => ({ name, mass: tot[i], tokens: toks[i] || 0 }));
}
export function attrRoleTotals(a) {
	if (Array.isArray(a && a.attr_roles)) return a.attr_roles; // abl summary
	const b = a && a.buckets;
	if (!b || !Array.isArray(b.attr_role)) return [];
	const names = b.attr_role_names || [], toks = b.attr_role_tokens || [], tot = names.map(() => 0);
	for (const row of b.attr_role) row.forEach((v, i) => { if (i < tot.length) tot[i] += (v || 0); });
	return names.map((name, i) => ({ name, mass: tot[i], tokens: toks[i] || 0 }));
}
// Per-quarter attention totals for the XML-gravity readout — the light `abl`
// summary (`a.gravity`) or the summed `agg`/`compact` bucket grid.
function gravityTotals(a) {
	if (Array.isArray(a && a.gravity)) return a.gravity; // abl summary
	const b = a && a.buckets;
	if (!b || !Array.isArray(b.gravity)) return [];
	const names = b.gravity_names || [], toks = b.gravity_tokens || [], tot = names.map(() => 0);
	for (const row of b.gravity) row.forEach((v, i) => { if (i < tot.length) tot[i] += (v || 0); });
	return names.map((name, i) => ({ name, mass: tot[i], tokens: toks[i] || 0 }));
}
// The three attribute-token roles (server-classified per attribute's real
// serialized form): content = the value, frame = the punctuation framing it
// (colon / brackets / quotes / commas / tags), context = the key name itself.
// Stack order bottom→top + the opacity SHADE each role is drawn at, so a bar's HUE
// reads as its treatment level while the three segments read as the roles.
export const STRUCT_ROLES = [
	{ key: "context", label: "context (the key name)", op: 0.32 },
	{ key: "frame", label: "frame (brackets · quotes · punctuation)", op: 0.62 },
	{ key: "content", label: "content (the value)", op: 1 },
];
let _structSumAll = false; // structure graph: sum ALL attributes into one aggregate bar per level
let _structNorm = true;    // structure graph: normalized (each bar = 100% density) vs raw attention (bar height = mass)
// XML-gravity scatter — SECONDARY x-axis lens (a distance reference). ref = which
// tag anchors the distance:
//   "close" — the MOVED </prompt> (signed: ◄ before · after ►); the default view.
//   "open"  — the FIXED <prompt> at block start (distance INTO the block).
// abs folds a signed displacement to |distance|.
let _gravRef = "close";
let _gravAbs = false;
let _gravLogY = true; // y on a SYMLOG scale (signed log) — the Δ spans decades; toggle to linear
const _gravBarOff = new Set(); // per-sentence bar card: treatments toggled OFF (empty = show all)
let _gravRawHeat = false; // heatmap: show RAW per-token attention instead of Δ vs the no-tag baseline
let _gravHideNextObject = true; // exclude next_object — gravity readout is glitchy on that kind

function gravityRows(rows) {
	return _gravHideNextObject ? rows.filter((r) => r.kind !== "next_object") : rows;
}

function gravNextObjectToggle() {
	const on = (b) => b ? "background:rgba(107,217,110,0.18);border-color:rgba(107,217,110,0.55);color:#cfe8d0" : "";
	return el("button", { class: "mini-toggle", style: on(_gravHideNextObject),
		title: _gravHideNextObject ? "include next_object (gravity readout is glitchy on this kind)" : "exclude next_object from gravity graphs",
		onclick: () => { _gravHideNextObject = !_gravHideNextObject; renderAblation(); } },
		_gravHideNextObject ? "skip next_object" : "next_object on");
}

// --- graphs -----------------------------------------------------------------

// Attribute spider grouped by the run's experiment LEVELS (coord conditions /
// shuffle methods / xml on-off).
function spiderGroups(rows, axis) {
	const groups = new Map(); // key -> { level, rows }
	for (const r of rows) { const lv = levelOf(r, axis); const g = groups.get(lv.key) || groups.set(lv.key, { level: lv, rows: [] }).get(lv.key); g.rows.push(r); }
	return [...groups.values()].sort((a, b) => a.level.rank - b.level.rank).map((g) => ({ label: g.level.label, color: g.level.color, rows: g.rows }));
}
// GRAPH 2 — attribute spider (one polygon per experiment level). `combined` merges
// local_origin + global_origin into ONE "origin" axis so L→L (local only), G→G
// (global only) and L/G (both) become comparable on a single origin axis (G=L=L+G).
function ablSpiderCard(rows, axis, combined = false) {
	const groups = spiderGroups(rows, axis);
	const profiles = groups.map((g) => {
		let comps = poolComponents(g.rows);
		if (combined) {
			let origin = 0; const rest = [];
			for (const c of comps) { if (c.component === "local_origin" || c.component === "global_origin") origin += c.score; else rest.push(c); }
			if (origin > 0) rest.push({ component: "origin", score: origin });
			comps = rest;
		}
		return { label: `${g.label} (${g.rows.length})`, color: g.color, map: new Map(comps.map((c) => [c.component, c.score])) };
	}).filter((p) => p.map.size);
	// Combined view swaps the two origin axes for one merged "origin" axis.
	const axes = combined ? [...ATTR_AXIS_ORDER.filter((c) => c !== "local_origin" && c !== "global_origin"), "origin"] : null;
	const body = profiles.length
		? chartHost((w) => spiderChart(profiles, { size: Math.min(w, vh(0.6, 300, 560)), ...(axes ? { axes } : {}) }), (w) => w)
		: empty("no attribute attention in scope");
	const title = combined ? "attribute profile · origin combined" : "attribute profile";
	const sub = combined
		? `local + global origin merged into one “origin” axis (compare L→L, G→G, L/G where G=L=L+G) · by ${axisLabel(axis)}`
		: `by ${axisLabel(axis)} · ${rows.length} variants · hover an axis for its value`;
	return card(title, sub, body);
}

// --- bar charts (treatment levels compared side-by-side) --------------------
// A HORIZONTAL category label centered under its band. Shows the full name
// (cat.full ?? cat.label) and, when it's wider than its band, compresses it to
// fit so full names stay on one line without overlapping (never tilted). When
// cat.tip is set, attaches a hover explanation (native title + help cursor).
function catAxisLabel(cx, y, cat, { fs = 1, bandW = 40, fill = "rgba(220,230,245,0.78)" } = {}) {
	const text = cat.full ?? cat.label ?? "";
	const fsz = +(10 * fs).toFixed(1);
	const attrs = { x: cx.toFixed(1), y: y.toFixed(1), fill: cat.color || fill, "font-size": fsz, "text-anchor": "middle", "dominant-baseline": "hanging" };
	const estW = String(text).length * fsz * 0.56; // rough glyph-width estimate
	const maxW = Math.max(14, bandW - 3);
	if (estW > maxW) { attrs.textLength = maxW.toFixed(1); attrs.lengthAdjust = "spacingAndGlyphs"; }
	const t = svgEl("text", attrs, text);
	if (cat.tip) { t.style.cursor = "help"; t.appendChild(svgEl("title", null, cat.tip)); }
	return t;
}
// A grouped bar chart: `cats` on the x-axis, one bar per `series` (level) within
// each category, colored by the LEVEL so the treatment reads by color (legend).
function groupedBarChart(w, cats, series, { yFmt = (v) => v.toFixed(2), height = 320 } = {}) {
	const W = Math.round(w), H = Math.round(height), fs = fontScale(W), fpx = (b) => +(b * fs).toFixed(1);
	const padL = 48, padR = 10, padT = 10, padB = Math.round(30 + 8 * fs);
	const px0 = padL, px1 = W - padR, py0 = padT, py1 = H - padB;
	let yMax = 0;
	for (const s of series) for (const v of s.values) if (v > yMax) yMax = v;
	yMax = yMax || 1;
	const Y = (v) => py1 - (Math.max(0, Math.min(yMax, v)) / yMax) * (py1 - py0);
	const svg = svgEl("svg", { class: "gsvg", viewBox: `0 0 ${W} ${H}`, width: W, height: H, style: "display:block" });
	for (const f of [0, 0.25, 0.5, 0.75, 1]) {
		const yy = py1 - f * (py1 - py0);
		svg.appendChild(svgEl("line", { x1: px0, y1: yy, x2: px1, y2: yy, stroke: "rgba(255,255,255,0.07)" }));
		svg.appendChild(svgEl("text", { x: px0 - 5, y: yy, fill: "rgba(220,230,245,0.6)", "font-size": fpx(10.5), "text-anchor": "end", "dominant-baseline": "middle" }, yFmt(yMax * f)));
	}
	const bandW = (px1 - px0) / Math.max(1, cats.length);
	const gap = Math.min(7, bandW * 0.14), barW = Math.max(2, (bandW - gap) / Math.max(1, series.length) - 1);
	cats.forEach((cat, ci) => {
		const bx0 = px0 + ci * bandW + gap / 2;
		series.forEach((s, si) => {
			const v = s.values[ci] || 0, x = bx0 + si * (barW + 1), y = Y(v);
			const rect = svgEl("rect", { class: "gpt", x: x.toFixed(1), y: y.toFixed(1), width: barW.toFixed(1), height: Math.max(0, py1 - y).toFixed(1), fill: s.color, rx: 1 });
			rect.appendChild(svgEl("title", null, `${cat.label} · ${s.label}: ${yFmt(v)}`));
			svg.appendChild(rect);
		});
		const cx = px0 + (ci + 0.5) * bandW;
		svg.appendChild(catAxisLabel(cx, py1 + fpx(5), cat, { fs, bandW, fill: "rgba(220,230,245,0.72)" }));
	});
	return svg;
}
// A grouped STACKED bar chart: `cats` on x, one bar per `series` (level) within
// each, and each bar STACKED into `roles` segments — hue = level, segment shade =
// role (via opacity). `series[i].values[c]` is an object keyed by role → value.
// With `normalize`, every bar is drawn to FULL height (each segment = its SHARE of
// the bar total), so the role PROPORTION is directly comparable across bars; a %
// label is printed on each big-enough segment.
export function stackedGroupedBarChart(w, cats, series, roles, { height = 340, normalize = false } = {}) {
	const W = Math.round(w), H = Math.round(height), fs = fontScale(W), fpx = (b) => +(b * fs).toFixed(1);
	const padL = 46, padR = 10, padT = 10, padB = Math.round(32 + 8 * fs);
	const px0 = padL, px1 = W - padR, py0 = padT, py1 = H - padB;
	let yMax = normalize ? 1 : 0;
	if (!normalize) for (const s of series) for (const val of s.values) { const t = roles.reduce((a, r) => a + (val[r.key] || 0), 0); if (t > yMax) yMax = t; }
	yMax = yMax || 1;
	const Y = (v) => py1 - (Math.max(0, Math.min(yMax, v)) / yMax) * (py1 - py0);
	const yFmt = normalize ? pctFmt : (v) => v.toFixed(3);
	const svg = svgEl("svg", { class: "gsvg", viewBox: `0 0 ${W} ${H}`, width: W, height: H, style: "display:block" });
	for (const f of [0, 0.25, 0.5, 0.75, 1]) {
		const yy = py1 - f * (py1 - py0);
		svg.appendChild(svgEl("line", { x1: px0, y1: yy, x2: px1, y2: yy, stroke: "rgba(255,255,255,0.07)" }));
		svg.appendChild(svgEl("text", { x: px0 - 5, y: yy, fill: "rgba(220,230,245,0.6)", "font-size": fpx(10), "text-anchor": "end", "dominant-baseline": "middle" }, yFmt(yMax * f)));
	}
	const bandW = (px1 - px0) / Math.max(1, cats.length);
	const gap = Math.min(12, bandW * 0.18), barW = Math.max(3, (bandW - gap) / Math.max(1, series.length) - 1);
	cats.forEach((cat, ci) => {
		const bx0 = px0 + ci * bandW + gap / 2;
		series.forEach((s, si) => {
			const val = s.values[ci] || {}, x = bx0 + si * (barW + 1);
			const total = roles.reduce((a, r) => a + (val[r.key] || 0), 0);
			let acc = 0;
			for (const r of roles) {
				const raw = val[r.key] || 0;
				const v = normalize ? (total > 0 ? raw / total : 0) : raw; // plotted height
				if (v > 1e-9) {
					const yTop = Y(acc + v), yBot = Y(acc), hgt = Math.max(0.6, yBot - yTop);
					const rect = svgEl("rect", { class: "gpt", x: x.toFixed(1), y: yTop.toFixed(1), width: barW.toFixed(1), height: hgt.toFixed(1), fill: hexA(s.color, r.op), stroke: "rgba(10,12,18,0.55)", "stroke-width": 0.5 });
					rect.appendChild(svgEl("title", null, `${cat.label} · ${r.key}\n${s.label}: ${normalize ? `${(v * 100).toFixed(0)}% · ` : ""}${raw.toFixed(4)} /token`));
					svg.appendChild(rect);
					// share label on the segment when it's big enough to read
					if (normalize && hgt > 13 && barW >= 22) svg.appendChild(svgEl("text", { x: (x + barW / 2).toFixed(1), y: ((yTop + yBot) / 2).toFixed(1), fill: "rgba(255,255,255,0.95)", stroke: "rgba(10,12,18,0.7)", "stroke-width": 2.4, "paint-order": "stroke", "font-size": fpx(9.5), "text-anchor": "middle", "dominant-baseline": "middle" }, `${Math.round(v * 100)}%`));
				}
				acc += v;
			}
		});
		const cx = px0 + (ci + 0.5) * bandW;
		svg.appendChild(catAxisLabel(cx, py1 + fpx(5), cat, { fs, bandW }));
	});
	return svg;
}

// The distinct experiment LEVELS present in `rows` (deduped, in rank order) — the
// treatment conditions the graphs compare, which the shared top legend keys.
function levelsPresent(rows, axis) {
	const m = new Map();
	for (const r of rows) { const lv = levelOf(r, axis); if (!m.has(lv.key)) m.set(lv.key, lv); }
	return [...m.values()].sort((a, b) => a.rank - b.rank);
}
// Prominent LEVEL legend (the treatment conditions LG2L / LG2G / L2L / … or methods
// / xml). With `sticky` it's the ONE shared color key pinned to the top of the
// ablation view (so it stays visible while you scroll every graph); large, bold
// swatches so the color key reads clearly.
function levelLegendEl(series, { sticky = false } = {}) {
	const items = series.map((s) => el("div", { class: "lg" }, el("span", { class: "sw", style: `background:${s.color};width:15px;height:15px;border-radius:3px` }), el("span", { text: s.label })));
	const label = sticky ? [el("span", { class: "muted", style: "font-weight:500;font-size:11px;margin-right:2px", text: "levels:" })] : [];
	return el("div", { class: "chart-legend",
		style: sticky
			? "position:sticky;top:0;z-index:6;gap:16px;align-items:center;padding:9px 12px;margin:0 0 10px;background:var(--panel,#12141a);border:1px solid var(--line,#2a2e37);border-radius:8px;font-size:13px;font-weight:600;box-shadow:0 2px 10px rgba(0,0,0,.28)"
			: "gap:18px;margin-top:10px;font-size:13px;font-weight:600" },
		...label, ...items);
}

// GRAPH 1 — token-type comparison: attention SHARE by token class (number /
// spatial / separator / bracket / word / …) as GROUPED BARS — x = token class, one
// bar per experiment level (colored by level), so treatments read by color.
function ablTypeCard(rows, axis) {
	const byLevel = new Map();
	for (const r of rows) {
		const tt = typeTotals(r.a);
		const sum = tt.reduce((s, t) => s + (t.mass || 0), 0) || 1;
		const lv = levelOf(r, axis);
		const g = byLevel.get(lv.key) || byLevel.set(lv.key, { level: lv, acc: new Map(), n: 0 }).get(lv.key);
		g.n += 1;
		for (const t of tt) g.acc.set(t.name, (g.acc.get(t.name) || 0) + (t.mass || 0) / sum); // per-variant share, pooled
	}
	const levels = [...byLevel.values()].sort((a, b) => a.level.rank - b.level.rank);
	const typeSet = new Set();
	for (const g of levels) for (const k of g.acc.keys()) typeSet.add(k);
	const types = [...typeSet].sort((a, b) => typeRank(a) - typeRank(b) || (a < b ? -1 : 1));
	if (!levels.length || !types.length) {
		return card("word / token types", `attention share by token class · one bar per ${axisLabel(axis)}`,
			empty("no token-type buckets — recompute this run's attention (restart the API server so the light view carries them)"));
	}
	const cats = types.map((t) => ({ label: typeLabel(t), full: typeFull(t), tip: TYPE_DESC[t], color: typeColor(t) }));
	const series = levels.map((g) => ({ key: g.level.key, label: `${g.level.label} (${g.n})`, color: g.level.color, values: types.map((t) => (g.acc.get(t) || 0) / (g.n || 1)) }));
	const hf = (w) => Math.round(vh(0.46, 240, 440));
	const body = el("div", {}, chartHost((w, h) => groupedBarChart(w, cats, series, { yFmt: pctFmt, height: h }), hf));
	return card("word / token types", `attention share by token class · one bar per ${axisLabel(axis)} (bar color = level, see top legend) · hover a class name for what it means`, body);
}

// GRAPH 3 — structure vs content: for EACH attribute, one bar per experiment level,
// and each bar is split into 3 SEGMENTS — context / frame / content — sized by the
// length-normalized attention density (mass ÷ tokens) on that role's tokens. Bar
// hue = level (legend); segment shade = role. The role split is server-classified
// per attribute's real serialized form (buckets.attr_role).
function ablStructureCard(rows, axis) {
	const byLevel = new Map();
	for (const r of rows) {
		const lv = levelOf(r, axis);
		const g = byLevel.get(lv.key) || byLevel.set(lv.key, { level: lv, mass: new Map(), toks: new Map() }).get(lv.key);
		for (const seg of attrRoleTotals(r.a)) {
			const bar = String(seg.name).indexOf("|"); if (bar < 0) continue;
			const mk = seg.name; // `${attr}|${role}`
			g.mass.set(mk, (g.mass.get(mk) || 0) + (seg.mass || 0));
			g.toks.set(mk, (g.toks.get(mk) || 0) + (seg.tokens || 0));
		}
	}
	const levels = [...byLevel.values()].sort((a, b) => a.level.rank - b.level.rank);
	const density = (g, attr, role) => { const mk = `${attr}|${role}`; const t = g.toks.get(mk) || 0; return t ? (g.mass.get(mk) || 0) / t : 0; };
	const attrSet = new Set();
	for (const g of levels) for (const k of g.mass.keys()) attrSet.add(k.slice(0, k.lastIndexOf("|")));
	const attrTotal = (a) => levels.reduce((s, g) => s + STRUCT_ROLES.reduce((t, r) => t + density(g, a, r.key), 0), 0);
	const attrs = [...attrSet].filter((a) => attrTotal(a) > 1e-9).sort((a, b) => attrTotal(b) - attrTotal(a));
	if (!levels.length || !attrs.length) {
		return card("structure vs content", `context / frame / content attention per attribute · by ${axisLabel(axis)}`,
			empty("no attribute role-split (context/frame/content) tokens for these variants — recompute this run's attention (analysis v9+)"));
	}
	// SUM-ALL: one aggregate bar per level (Σmass ÷ Σtokens per role over every
	// attribute). Otherwise one bar per attribute. Normalized → density (mass ÷
	// tokens); non-normalized → raw mass, so the bar HEIGHT is the attention paid.
	const rawMass = (g, attr, role) => g.mass.get(`${attr}|${role}`) || 0;
	const valOf = (g, attr, role) => _structNorm ? density(g, attr, role) : rawMass(g, attr, role);
	let cats, series;
	if (_structSumAll) {
		cats = [{ label: "all attributes" }];
		series = levels.map((g) => {
			const v = {};
			for (const r of STRUCT_ROLES) { let m = 0, t = 0; for (const a of attrs) { const mk = `${a}|${r.key}`; m += g.mass.get(mk) || 0; t += g.toks.get(mk) || 0; } v[r.key] = _structNorm ? (t ? m / t : 0) : m; }
			return { key: g.level.key, label: g.level.label, color: g.level.color, values: [v] };
		});
	} else {
		cats = attrs.map((a) => ({ label: a }));
		series = levels.map((g) => ({ key: g.level.key, label: g.level.label, color: g.level.color,
			values: attrs.map((a) => ({ context: valOf(g, a, "context"), frame: valOf(g, a, "frame"), content: valOf(g, a, "content") })) }));
	}
	const hf = (w) => Math.round(vh(0.5, 280, 500));
	// Toggles: normalize (each bar = 100% vs bar height = raw attention) + sum-all.
	const toggle = el("div", { style: "display:flex;justify-content:flex-end;gap:8px;margin:-2px 0 6px" },
		el("button", { class: `mini-toggle${_structNorm ? " on" : ""}`, title: "normalized: each bar = 100% (length-normalized density) · off: bar height = total attention (mass)",
			onclick: () => { _structNorm = !_structNorm; renderAblation(); } }, "normalize"),
		el("button", { class: "mini-toggle", title: _structSumAll ? "show each attribute separately" : "sum every attribute into one aggregate composition per level",
			onclick: () => { _structSumAll = !_structSumAll; renderAblation(); } }, _structSumAll ? "▦ per attribute" : "Σ sum all attributes"));
	// ROLE legend — the shade ramp (faint→solid) drawn in a neutral reference hue.
	// (The LEVEL color key is the shared sticky legend at the top of the view.)
	const REF = "#aeb9cc";
	const roleLegend = el("div", { class: "chart-legend", style: "gap:14px;margin-top:5px;font-size:11.5px" },
		el("span", { class: "muted", text: "segment shade (faint→solid):" }),
		...STRUCT_ROLES.map((r) => el("div", { class: "lg" }, el("span", { class: "sw", style: `background:${hexA(REF, r.op)};width:13px;height:13px` }), el("span", { text: r.label.split(" (")[0] }))));
	const body = el("div", {}, toggle, chartHost((w, h) => stackedGroupedBarChart(w, cats, series, STRUCT_ROLES, { height: h, normalize: _structNorm }), hf), roleLegend);
	return card("structure vs content — per attribute",
		`${_structSumAll ? "all attributes summed" : "one bar per attribute"} · ${_structNorm ? "each bar = 100% of its context / frame / content attention (length-normalized mass ÷ tokens) so the PROPORTION compares" : "bar HEIGHT = total attention (mass), split context / frame / content"} · bar hue = ${axisLabel(axis)}, shade = role`,
		body);
}

// GRAPH — composition: how the treated step's SCENE attention splits across entity
// KINDS (zones / objects / frames), as GROUPED BARS — x = kind, one bar per
// experiment level so the split reads by TREATMENT color. It's the Data tab's
// composition pie, but broken out per treatment (bar hue = level) instead of a
// single pooled pie. Measured over the whole generation ("scene") — the only
// region the light `abl` projection carries.
function ablCompositionCard(rows, axis) {
	const byLevel = new Map();
	for (const r of rows) {
		const lv = levelOf(r, axis);
		const g = byLevel.get(lv.key) || byLevel.set(lv.key, { level: lv, rows: [], n: 0 }).get(lv.key);
		g.rows.push(r); g.n += 1;
	}
	const levels = [...byLevel.values()].sort((a, b) => a.level.rank - b.level.rank);
	const KINDS = [
		{ key: "zone", label: "zones", color: COLORS.zone },
		{ key: "object", label: "objects", color: COLORS.object },
		{ key: "frame", label: "frames", color: COLORS.frame },
	];
	const shareRows = levels.map((g) => {
		const kt = poolKindTotals(g.rows, "scene");
		const total = (kt.zone || 0) + (kt.object || 0) + (kt.frame || 0);
		return { level: g.level, n: g.n, total, shares: KINDS.map((k) => (total ? (kt[k.key] || 0) / total : 0)) };
	});
	if (!levels.length || !shareRows.some((s) => s.total > 0)) {
		return card("composition \u2014 attention by entity kind", `zones / objects / frames \u00b7 by ${axisLabel(axis)}`,
			empty("no scene-entity attention for these variants \u2014 recompute this run's attention"));
	}
	const cats = KINDS.map((k) => ({ label: k.label, color: k.color }));
	const series = shareRows.map((s) => ({ key: s.level.key, label: `${s.level.label} (${s.n})`, color: s.level.color, values: s.shares }));
	const hf = (w) => Math.round(vh(0.46, 240, 440));
	const body = el("div", {}, chartHost((w, h) => groupedBarChart(w, cats, series, { yFmt: pctFmt, height: h }), hf));
	return card("composition \u2014 attention by entity kind",
		`share of the treated step's SCENE attention on zones / objects / frames \u00b7 one bar per ${axisLabel(axis)} (bar color = level, see top legend) \u00b7 pooled over the whole generation`,
		body);
}

// GRAPH — attention vs scene ORDER, colored by TREATMENT: one dot per scene entity
// (x = its position in the — for shuffle, reordered — scene context, y = attention
// per-step normalized), with a per-treatment binned-mean trend + Spearman ρ. This
// is the Data tab's composition scatter, but one series per experiment level so the
// order↔attention relationship reads by treatment color (does the shuffle method
// change WHERE attention lands?).
function orderAttnScatter(w, groups, { height = 340 } = {}) {
	const W = Math.round(w), H = Math.round(height), fs = fontScale(W), F = (b) => +(b * fs).toFixed(1);
	const padL = Math.round(40 + 12 * fs), padR = 14, padT = Math.round(8 + 6 * fs), padB = Math.round(30 + 14 * fs);
	const px0 = padL, px1 = W - padR, py0 = padT, py1 = H - padB;
	const X = (x) => px0 + Math.max(0, Math.min(1, x)) * (px1 - px0);
	const Y = (y) => py1 - Math.max(0, Math.min(1, y)) * (py1 - py0);
	const svg = svgEl("svg", { class: "gsvg", viewBox: `0 0 ${W} ${H}`, width: W, height: H, style: "display:block" });
	for (const gy of [0, 0.25, 0.5, 0.75, 1]) {
		const y = Y(gy);
		svg.appendChild(svgEl("line", { x1: px0, y1: y.toFixed(1), x2: px1, y2: y.toFixed(1), stroke: "rgba(255,255,255,0.06)" }));
		svg.appendChild(svgEl("text", { x: (px0 - 5).toFixed(1), y: y.toFixed(1), fill: "rgba(220,230,245,0.6)", "font-size": F(10), "text-anchor": "end", "dominant-baseline": "middle" }, gy.toFixed(2)));
	}
	for (const gx of [0, 0.25, 0.5, 0.75, 1]) {
		const x = X(gx);
		svg.appendChild(svgEl("line", { x1: x.toFixed(1), y1: py0, x2: x.toFixed(1), y2: py1, stroke: "rgba(255,255,255,0.05)" }));
		svg.appendChild(svgEl("text", { x: x.toFixed(1), y: (py1 + F(13)).toFixed(1), fill: "rgba(220,230,245,0.6)", "font-size": F(10), "text-anchor": "middle" }, gx.toFixed(2)));
	}
	// faint points per treatment (sampled so a dense cloud stays light), then the
	// bold per-treatment binned-mean trend on top.
	for (const g of groups) {
		const stride = Math.max(1, Math.ceil(g.pts.length / 700));
		for (let i = 0; i < g.pts.length; i += stride) { const p = g.pts[i]; svg.appendChild(svgEl("circle", { cx: X(p.x).toFixed(1), cy: Y(p.y).toFixed(1), r: (2.2 * fs).toFixed(1), fill: g.color, "fill-opacity": 0.16 })); }
	}
	const NB = 8;
	for (const g of groups) {
		const bins = Array.from({ length: NB }, () => ({ sy: 0, sx: 0, n: 0 }));
		for (const p of g.pts) { let bi = Math.floor(p.x * NB); bi = Math.max(0, Math.min(NB - 1, bi)); const b = bins[bi]; b.sy += p.y; b.sx += p.x; b.n++; }
		const line = bins.filter((b) => b.n).map((b) => ({ x: b.sx / b.n, y: b.sy / b.n }));
		if (line.length < 2) continue;
		let d = ""; line.forEach((p, k) => { d += (k ? "L" : "M") + X(p.x).toFixed(1) + "," + Y(p.y).toFixed(1) + " "; });
		svg.appendChild(svgEl("path", { d: d.trim(), fill: "none", stroke: "rgba(8,10,16,0.7)", "stroke-width": 4, "stroke-linejoin": "round" }));
		svg.appendChild(svgEl("path", { d: d.trim(), fill: "none", stroke: g.color, "stroke-width": 2.4, "stroke-linejoin": "round" }));
	}
	svg.appendChild(svgEl("text", { x: px1, y: (py1 + F(27)).toFixed(1), fill: "rgba(220,230,245,0.82)", "font-size": F(11), "text-anchor": "end" }, "order in scene context  (0 = first \u2192 1 = last)"));
	const my = ((py0 + py1) / 2).toFixed(1);
	svg.appendChild(svgEl("text", { x: 12, y: my, fill: "rgba(220,230,245,0.82)", "font-size": F(11), "text-anchor": "middle", transform: `rotate(-90 12 ${my})` }, "attention (per-step normalized)"));
	return el("div", { class: "gwrap" }, svg);
}
function ablOrderScatterCard(rows, axis) {
	const byLevel = new Map();
	for (const r of rows) { const lv = levelOf(r, axis); const g = byLevel.get(lv.key) || byLevel.set(lv.key, { level: lv, rows: [] }).get(lv.key); g.rows.push(r); }
	const groups = [...byLevel.values()].sort((a, b) => a.level.rank - b.level.rank).map((g) => {
		const pts = contextPoints(g.rows, "scene").map((p) => ({ x: p.x, y: p.y }));
		const rho = pts.length >= 3 ? spearman(pts.map((p) => p.x), pts.map((p) => p.y)) : null;
		return { level: g.level, color: g.level.color, pts, rho, n: pts.length };
	}).filter((g) => g.pts.length);
	if (!groups.length) {
		return card("composition \u2014 attention vs scene order", `by ${axisLabel(axis)}`,
			empty("no positioned scene entities for these variants \u2014 recompute this run's attention"));
	}
	const hf = (w) => Math.round(vh(0.5, 300, 520));
	const legend = chartLegend(groups.map((g) => ({ key: g.level.key, label: `${g.level.label}${g.rho != null ? ` \u00b7 \u03c1=${g.rho.toFixed(2)}` : ""} (${g.n})`, color: g.color })));
	const body = el("div", {}, chartHost((w) => orderAttnScatter(w, groups, { height: hf(w) }), hf), legend);
	const sub = `one dot = one scene entity \u00b7 x = its ORDER in the (shuffled) scene context \u00b7 y = attention (per-step normalized) \u00b7 bold line = per-treatment binned mean \u00b7 \u03c1 = Spearman(order, attention) per ${axisLabel(axis)} \u00b7 colored by treatment (see legend)`;
	return card("composition \u2014 attention vs scene order", sub, body);
}

// ρ vs focus: as you keep only higher-attention objects (raise the min attention
// share), does the position↔attention correlation hold? These are exported and
// reused by the data view (a single pooled line) and the ablation view (one line
// per shuffle method).
export const FOCUS_THRESHOLDS = [0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15];
// Build one pooled ρ-vs-focus series over `rows` for a scene region
// ("scene" | "reasoning" | "output"). Returns [] when too sparse to correlate.
export function rhoFocusSeries(rows, region = "scene", color = "#7aa2f7", label = "attention") {
	const pool = [];
	for (const r of rows) {
		const ents = (r.a.scene_entities || []).filter((e) => e.token_span && e.token_span.length && entityKindLabel(e.kind, e.id) === "object");
		if (ents.length < 3) continue;
		const totals = new Map((((r.a.agg || {})[region] || {}).entityTotals || []).map((e) => [e.id, e.score]));
		const sorted = [...ents].sort((a, b) => a.token_span[0] - b.token_span[0]);
		const denom = Math.max(1, sorted.length - 1);
		const sum = sorted.reduce((s, e) => s + (totals.get(e.id) || 0), 0) || 1;
		sorted.forEach((e, i) => { const a = totals.get(e.id) || 0; pool.push({ pos: i / denom, attn: a, share: a / sum }); });
	}
	const points = [];
	for (const t of FOCUS_THRESHOLDS) {
		const kept = pool.filter((p) => p.share >= t);
		if (kept.length < 3) continue;
		points.push({ x: t, y: spearman(kept.map((p) => p.pos), kept.map((p) => p.attn)), n: kept.length });
	}
	return points.length ? [{ label, color, points }] : [];
}
// Adaptive-y ρ-vs-focus line chart. `series`: [{label,color,points:[{x,y,n}]}].
// A legend is drawn only when there is more than one series.
export function rhoCurveChart(series, opts = {}) {
	if (!series || !series.length) return el("div", { class: "empty", text: opts.empty || "not enough data to correlate" });
	const W = Math.round(opts.width || 640), H = Math.round(opts.height || 320);
	const thresholds = opts.thresholds || FOCUS_THRESHOLDS, n = thresholds.length;
	const fs = fontScale(W), F = (b) => +(b * fs).toFixed(1);
	const padL = Math.round(40 + 12 * fs), padR = 14, padT = Math.round(8 + 6 * fs), padB = Math.round(30 + 14 * fs);
	const px0 = padL, px1 = W - padR, py0 = padT, py1 = H - padB;
	const X = (i) => px0 + (n <= 1 ? 0.5 : i / (n - 1)) * (px1 - px0);
	// adaptive y-range: fit the actual ρ values (always keep 0 in view for the sign),
	// so a narrow band of ρ uses the full height instead of hugging the middle.
	const allY = series.flatMap((s) => s.points.map((p) => p.y));
	let lo = Math.min(0, ...allY), hi = Math.max(0, ...allY);
	if (hi - lo < 0.08) { hi += 0.04; lo -= 0.04; }
	const pad = (hi - lo) * 0.08; lo = Math.max(-1, lo - pad); hi = Math.min(1, hi + pad);
	const Y = (v) => py1 - ((Math.max(lo, Math.min(hi, v)) - lo) / ((hi - lo) || 1)) * (py1 - py0);
	const svg = svgEl("svg", { class: "gsvg", viewBox: `0 0 ${W} ${H}`, width: W, height: H, style: "display:block" });
	for (const gy of niceTicks(lo, hi)) {
		const yy = Y(gy), zero = Math.abs(gy) < 1e-9;
		svg.appendChild(svgEl("line", { x1: px0, y1: yy.toFixed(1), x2: px1, y2: yy.toFixed(1), stroke: zero ? "rgba(255,255,255,0.28)" : "rgba(255,255,255,0.06)" }));
		svg.appendChild(svgEl("text", { x: px0 - 6, y: yy.toFixed(1), fill: "rgba(220,230,245,0.6)", "font-size": F(11), "text-anchor": "end", "dominant-baseline": "middle" }, zero ? "0" : gy.toFixed(2)));
	}
	thresholds.forEach((t, i) => svg.appendChild(svgEl("text", { x: X(i).toFixed(1), y: (py1 + F(14)).toFixed(1), fill: "rgba(220,230,245,0.6)", "font-size": F(10.5), "text-anchor": "middle" }, t === 0 ? "all" : `\u2265${(t * 100).toFixed(t < 0.01 ? 2 : 1)}%`)));
	svg.appendChild(svgEl("text", { x: px1, y: (py1 + F(27)).toFixed(1), fill: "rgba(220,230,245,0.82)", "font-size": F(12), "text-anchor": "end" }, opts.xLabel || "focus  (min attention share)"));
	const my = ((py0 + py1) / 2).toFixed(1);
	svg.appendChild(svgEl("text", { x: 12, y: my, fill: "rgba(220,230,245,0.82)", "font-size": F(12), "text-anchor": "middle", transform: `rotate(-90 12 ${my})` }, opts.yLabel || "Spearman \u03c1 (position · attention)"));
	const tIdx = new Map(thresholds.map((t, i) => [t, i]));
	for (const s of series) {
		const pts = [...s.points].filter((p) => tIdx.has(p.x)).sort((a, b) => tIdx.get(a.x) - tIdx.get(b.x));
		if (!pts.length) continue;
		let d = ""; pts.forEach((p, k) => { d += (k ? "L" : "M") + X(tIdx.get(p.x)).toFixed(1) + "," + Y(p.y).toFixed(1) + " "; });
		svg.appendChild(svgEl("path", { d: d.trim(), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round" }));
		for (const p of pts) { const dot = svgEl("circle", { cx: X(tIdx.get(p.x)).toFixed(1), cy: Y(p.y).toFixed(1), r: (3.2 * fs).toFixed(1), fill: s.color }); dot.appendChild(svgEl("title", null, `${s.label} · focus \u2265${(p.x * 100).toFixed(2)}% · \u03c1=${p.y.toFixed(2)} · n=${p.n}`)); svg.appendChild(dot); }
	}
	const wrap = el("div", { class: "gwrap" }, svg);
	if (series.length > 1) wrap.appendChild(chartLegend(series.map((s) => ({ key: s.label, label: s.label, color: s.color }))));
	return wrap;
}

// --- orchestration -----------------------------------------------------------

// Toolbar: FIRST pick which run (experiment) to visualize; the graphs then group
// by the axis that run was launched under (shown, read-only, beside the picker).
function toolbar(runs, axis) {
	const runSel = el("select", { class: "abl-run-sel", title: "which run (experiment) to visualize — the graphs group by the axis it varied",
		onchange: (e) => { state.ablRun = e.target.value; renderAblation(); } },
		...runs.map((r) => el("option", { value: r.label, text: `${r.label || "(no label)"} · ${axisLabel(r.axis)} · ${r.n}`, ...(r.label === state.ablRun ? { selected: "" } : {}) })));
	return el("div", { class: "abl-toolbar" },
		el("div", { class: "abl-tool-grp" }, el("span", { text: "run" }), runSel),
		el("div", { class: "abl-tool-grp" }, el("span", { text: "experiment" }), el("span", { class: "abl-exp-name", style: "font-weight:600", text: axisLabel(axis) })));
}

// --- GRAPH (gravity) — per-SENTENCE tag distance vs baseline-subtracted attn --
// Per sentence of each qN variant: its per-token attention MINUS the no-tag
// ("none") baseline, against the signed token distance from the moved </prompt>.
// A gravity well = |Δ| peaks near the tag (distance ~0) and decays with distance.
// Modeled on the data tab's per-instruction VII view — one dot per sentence.
//
// MULTI-STEP-KIND: the instruction block (its sentences) is TEMPLATE text, so it
// DIFFERS by step kind (anchor_decompose ≠ object_bbox_batch ≠ …) — sentence `s3`
// is a different sentence in each. So the baseline and the sentence identity are
// keyed by `${kind}\0s{i}`, never by `s{i}` alone: each kind subtracts its OWN
// no-tag baseline and carries its OWN per-step text. Distance (center − tag) is
// relative to each point's own kind's tag, so every kind projects onto the one
// shared distance axis correctly. Each point carries its kind + full sentence.
function gravityModel(rows) {
	const noneSum = new Map(), noneN = new Map();   // `${kind}\0s{i}` -> no-tag density sum / count
	const variants = [];
	// Diagnostics for an accurate empty state: per qN variant, is the stored gravity
	// readout per-SENTENCE (ready), only positions/QUARTER-era (stale → recompute),
	// or ABSENT (no <prompt> block on this kind, or attention predates the feature)?
	const stat = { nQn: 0, sent: 0, quarter: 0, absent: 0, absentKinds: new Set() };
	for (const r of rows) {
		const lv = r.gravity || "baseline";
		const kind = r.kind || "?";
		const g = (r.a && r.a.meta && r.a.meta.gravity) || {};
		const spos = new Map((g.sentences || []).map((s) => [`s${s.i}`, s])); // s{i} -> {i, tok_start, tok_end, snippet}
		const dens = new Map();                          // sentence name (s{i}) -> attention/token
		for (const x of gravityTotals(r.a)) dens.set(x.name, x.tokens ? x.mass / x.tokens : 0);
		if (lv === "none") {
			for (const [nm, d] of dens) { const k = `${kind}\u0000${nm}`; noneSum.set(k, (noneSum.get(k) || 0) + d); noneN.set(k, (noneN.get(k) || 0) + 1); }
		} else if (lv !== "baseline") {
			stat.nQn++;
			const hasSent = (g.sentences || []).some((s) => s && s.tok_start != null);
			const hasPos = g.open_tok != null && g.close_tok != null;
			if (hasSent) stat.sent++;
			else if (hasPos) stat.quarter++;              // positions but no per-sentence buckets = quarter-era
			else { stat.absent++; stat.absentKinds.add(kind); } // no gravity readout at all
			variants.push({ lv, kind, dens, spos, open: g.open_tok, close: g.close_tok });
		}
	}
	const base = new Map();                              // per-(kind,sentence) no-tag mean density
	for (const [k, s] of noneSum) base.set(k, s / (noneN.get(k) || 1));
	const points = [];
	for (const v of variants) {
		if (v.open == null || v.close == null) continue; // need BOTH tag tokens so every view can anchor
		for (const [nm, d] of v.dens) {
			const b = base.get(`${v.kind}\u0000${nm}`); if (b == null) continue; // subtract THIS kind's own baseline
			const sm = v.spos.get(nm); if (!sm || sm.tok_start == null || sm.tok_end == null) continue;
			const center = (sm.tok_start + sm.tok_end) / 2; // the view (ref/abs) is applied later, per-point, in gravX
			points.push({ center, open: v.open, close: v.close, delta: d - b, level: v.lv,
				kind: v.kind, sent: sm.i, snippet: sm.snippet || nm, color: GRAVITY_LEVEL_COLOR[v.lv] || "#7aa2f7" });
		}
	}
	return { points, hasBaseline: base.size > 0, kinds: [...new Set(points.map((p) => p.kind))].sort(), stat };
}

// The plotted x for a point under the current view: token distance from the moved
// </prompt> ("close") or the fixed <prompt> ("open"), folded to |distance| when
// abs is on. gravMag is the always-≥0 magnitude used for the |dist|–Δ correlation
// (view-independent so the ρ summary stays comparable across the two references).
function gravX(p, ref, abs) {
	const raw = ref === "open" ? (p.center - p.open) : (p.center - p.close);
	return abs ? Math.abs(raw) : raw;
}
function gravMag(p, ref) {
	return ref === "open" ? (p.center - p.open) : Math.abs(p.center - p.close);
}
function gravXLabel(ref, abs) {
	if (ref === "open") return abs ? "|token distance| from <prompt> (start)" : "token distance from <prompt> (start)  \u25ba deeper";
	return abs ? "|token distance| from </prompt>" : "token distance from </prompt>   (\u25c4 before \u00b7 after \u25ba)";
}

// Signed-x, signed-y scatter (the shared scatterChart is fixed to x>=0 / y>=0, so
// this axis draws its own). x is per the current view (gravX): distance from the
// moved </prompt> or the fixed <prompt>, |folded| when abs. A dashed guide marks
// the anchoring tag (x=0); y = Δ attention / token vs the no-tag baseline. Returns
// a positioned wrapper so a per-dot hover tooltip can RETRIEVE the sentence itself
// (step kind + full text) — one dot = one sentence of one variant.
function gravityScatter(w, points, { height = 340, ref = "close", abs = false, logY = false } = {}) {
	const W = Math.round(w), H = Math.round(height), fs = fontScale(W), F = (b) => +(b * fs).toFixed(1);
	const padL = Math.round(56 + 10 * fs), padR = 14, padT = 10, padB = Math.round(30 + 14 * fs);
	const px0 = padL, px1 = W - padR, py0 = padT, py1 = H - padB;
	const xs = points.map((p) => gravX(p, ref, abs));
	const tagLabel = ref === "open" ? "<prompt>" : "</prompt>"; // the anchoring tag sits at x=0
	let xLo = Math.min(0, ...xs), xHi = Math.max(0, ...xs);
	if (!(xHi - xLo > 1e-9)) { xHi += 1; xLo -= 1; }
	// y is per-token Δ (mass ÷ tokens, minus the no-tag baseline). logY = SYMLOG: a
	// SIGNED log so the ± Δ spans decades; L (the linear↔log knee) = median |Δ| (≥ a
	// floor) so the heavy tail spreads while values near 0 stay readable.
	const absd = points.map((p) => Math.abs(p.delta)).filter((v) => v > 0).sort((a, b) => a - b);
	const maxAbs = absd.length ? absd[absd.length - 1] : 1;
	const L = Math.max(maxAbs * 1e-3, absd.length ? absd[Math.floor(absd.length / 2)] : maxAbs) || 1e-12;
	const tOf = logY ? (y) => Math.sign(y) * Math.log10(1 + Math.abs(y) / L) : (y) => y;
	let tLo = Math.min(0, ...points.map((p) => tOf(p.delta))), tHi = Math.max(0, ...points.map((p) => tOf(p.delta)));
	const tp = (tHi - tLo) * 0.08 || 1e-6; tLo -= tp; tHi += tp;
	const X = (x) => px0 + ((x - xLo) / ((xHi - xLo) || 1)) * (px1 - px0);
	const Y = (y) => py1 - ((tOf(y) - tLo) / ((tHi - tLo) || 1)) * (py1 - py0);
	const svg = svgEl("svg", { class: "gsvg", viewBox: `0 0 ${W} ${H}`, width: W, height: H, style: "display:block" });
	for (const gx of niceTicks(xLo, xHi)) {
		const x = X(gx), zero = Math.abs(gx) < 1e-9;
		svg.appendChild(svgEl("line", { x1: x.toFixed(1), y1: py0, x2: x.toFixed(1), y2: py1, stroke: zero ? "rgba(255,255,255,0.3)" : "rgba(255,255,255,0.06)" }));
		svg.appendChild(svgEl("text", { x: x.toFixed(1), y: (py1 + F(13)).toFixed(1), fill: "rgba(220,230,245,0.6)", "font-size": F(10), "text-anchor": "middle" }, String(Math.round(gx))));
	}
	// The anchoring tag guide at x=0 (dashed + labeled), even off a nice tick.
	{ const x = X(0);
		svg.appendChild(svgEl("line", { x1: x.toFixed(1), y1: py0, x2: x.toFixed(1), y2: py1, stroke: "rgba(255,255,255,0.32)", "stroke-dasharray": "2 3" }));
		svg.appendChild(svgEl("text", { x: x.toFixed(1), y: (py0 + F(9)).toFixed(1), fill: "rgba(220,230,245,0.72)", "font-size": F(9.5), "text-anchor": "middle" }, tagLabel)); }
	// y ticks — log: 0 + ±L·10^k decades; linear: nice ticks over the Δ domain.
	let yticks;
	if (logY) { yticks = [0]; for (let v = L; v <= maxAbs * 1.5; v *= 10) yticks.push(v, -v); }
	else yticks = niceTicks(tLo, tHi);
	for (const gy of yticks) {
		const y = Y(gy), zero = Math.abs(gy) < 1e-12;
		svg.appendChild(svgEl("line", { x1: px0, y1: y.toFixed(1), x2: px1, y2: y.toFixed(1), stroke: zero ? "rgba(255,255,255,0.3)" : "rgba(255,255,255,0.06)" }));
		svg.appendChild(svgEl("text", { x: (px0 - 5).toFixed(1), y: y.toFixed(1), fill: "rgba(220,230,245,0.6)", "font-size": F(9), "text-anchor": "end", "dominant-baseline": "middle" }, zero ? "0" : gy.toExponential(0)));
	}
	// Positioned wrapper so the hover tooltip can float over the plot.
	const wrap = el("div", { class: "chart-host", style: "position:relative;width:100%" });
	const tip = el("div", { class: "graph-tip", style: "max-width:360px" });
	const showTip = (p, xv, ev) => {
		const dtxt = `${xv >= 0 ? "+" : ""}${Math.round(xv)} tok from ${ref === "open" ? "<prompt>" : "</prompt>"}`;
		tip.innerHTML = `<div class="xh">${escTip(GRAVITY_SHORT[p.level] || p.level)}${p.kind ? " \u00b7 " + escTip(p.kind) : ""} \u00b7 s${p.sent}</div>`
			+ `<div class="r"><span>\u0394 / token</span><b>${escTip((p.delta >= 0 ? "+" : "") + p.delta.toExponential(2))}</b></div>`
			+ `<div class="r"><span>distance</span><b>${escTip(dtxt)}</b></div>`
			+ `<div style="margin-top:6px;white-space:normal;line-height:1.35;opacity:.92">${escTip(p.snippet || "")}</div>`;
		const wr = wrap.getBoundingClientRect();
		const tw = tip.offsetWidth || 220, th = tip.offsetHeight || 90;
		let left = ev.clientX - wr.left + 14; if (left + tw > wr.width) left = ev.clientX - wr.left - tw - 14;
		let top = ev.clientY - wr.top + 12; if (top + th > wr.height) top = ev.clientY - wr.top - th - 12;
		tip.style.left = Math.max(0, left) + "px"; tip.style.top = Math.max(0, top) + "px"; tip.style.opacity = "1";
	};
	const hideTip = () => { tip.style.opacity = "0"; };
	// Least-squares trend — computed here but drawn LAST so it sits above the dots.
	let trendPath = null;
	if (points.length >= 2) {
		let sx = 0, sy = 0, sxx = 0, sxy = 0; const n = points.length;
		for (let i = 0; i < n; i++) { const x = xs[i], y = points[i].delta; sx += x; sy += y; sxx += x * x; sxy += x * y; }
		const den = n * sxx - sx * sx, slope = den ? (n * sxy - sx * sy) / den : 0, inter = (sy - slope * sx) / n;
		let d = ""; const NSEG = 48;
		for (let k = 0; k <= NSEG; k++) { const x = xLo + (xHi - xLo) * k / NSEG; d += (k ? "L" : "M") + X(x).toFixed(1) + "," + Y(inter + slope * x).toFixed(1) + " "; }
		trendPath = svgEl("path", { d: d.trim(), fill: "none", stroke: "#ff2b2b", "stroke-width": 2.5, "stroke-dasharray": "5 4", "pointer-events": "none" });
	}
	points.forEach((p, i) => {
		const xv = xs[i];
		const dot = svgEl("circle", { cx: X(xv).toFixed(1), cy: Y(p.delta).toFixed(1), r: (4 * fs).toFixed(1), fill: p.color, "fill-opacity": 0.85, style: "cursor:pointer" });
		dot.addEventListener("pointerenter", (e) => showTip(p, xv, e));
		dot.addEventListener("pointermove", (e) => showTip(p, xv, e));
		dot.addEventListener("pointerleave", hideTip);
		svg.appendChild(dot);
	});
	if (trendPath) svg.appendChild(trendPath); // above dots (SVG paint order = DOM order)
	svg.appendChild(svgEl("text", { x: px1, y: (py1 + F(27)).toFixed(1), fill: "rgba(220,230,245,0.82)", "font-size": F(11), "text-anchor": "end" }, gravXLabel(ref, abs)));
	const my = ((py0 + py1) / 2).toFixed(1);
	svg.appendChild(svgEl("text", { x: 13, y: my, fill: "rgba(220,230,245,0.82)", "font-size": F(11), "text-anchor": "middle", transform: `rotate(-90 13 ${my})` }, `\u0394 attention / token  (vs no-tag)${logY ? "  \u00b7 symlog" : ""}`));
	wrap.appendChild(svg); wrap.appendChild(tip);
	return wrap;
}

// GRAPH — the gravity readout: baseline-subtracted per-quarter attention vs the
// closing tag's distance, with the pooled |dist|-vs-Δ Spearman as the summary.
function ablGravityCard(rows) {
	const model = gravityModel(rows);
	if (!model.points.length) {
		// Accurate, per-scope diagnosis instead of a blanket "recompute": the empty
		// state has genuinely different causes across step kinds.
		const s = model.stat || { nQn: 0 };
		let msg;
		if (!s.nQn) msg = "no gravity (q1\u2013q4) variants in this scope \u2014 pick a run / step that has them";
		else if (s.sent > 0 && !model.hasBaseline) msg = "compute the `no tags` variant \u2014 it's the subtraction anchor every level's \u0394 is measured against";
		else if (s.quarter > 0 && s.sent === 0) msg = `${s.quarter} variant${s.quarter === 1 ? "" : "s"} hold only the OLD quarter readout \u2014 use \u21bb recompute run for the per-sentence readout (they predate it; no re-inference needed)`;
		else msg = `no per-sentence gravity here \u2014 \u21bb recompute run; if a step kind stays empty it has NO <VERY_IMPORTANT_INSTRUCTIONS> block for gravity to act on${(s.absentKinds && s.absentKinds.size) ? ` (e.g. ${[...s.absentKinds].join(", ")})` : ""}`;
		return card("XML gravity \u2014 tag pull vs distance", "no per-sentence points in scope", empty(msg));
	}
	const ref = _gravRef === "open" ? "open" : "close"; // span removed; </prompt> (default) · <prompt> as a secondary lens
	const abs = _gravAbs;
	const trend = spearmanTrend(model.points.map((p) => gravMag(p, ref)), model.points.map((p) => p.delta));
	const levels = [...new Set(model.points.map((p) => p.level))].sort((a, b) => (_GRAV_RANK[a] ?? 99) - (_GRAV_RANK[b] ?? 99));
	const legend = levelLegendEl(levels.map((l) => ({ label: GRAVITY_SHORT[l] || l, color: GRAVITY_LEVEL_COLOR[l] || "#7aa2f7" })));
	const hf = (w) => Math.round(vh(0.5, 300, 520));
	// SECONDARY x-axis lens: which tag anchors the distance — the moved </prompt>
	// (default) or the fixed <prompt> — plus fold-to-|distance|. (Span view removed.)
	const on = (b) => b ? "background:rgba(107,217,110,0.18);border-color:rgba(107,217,110,0.55);color:#cfe8d0" : "";
	const refBtn = (id, label, title) => el("button", { class: "mini-toggle", style: on(ref === id), title, onclick: () => { _gravRef = id; renderAblation(); } }, label);
	const absBtn = el("button", { class: "mini-toggle", style: on(abs),
		title: abs ? "show signed displacement (before/after the tag)" : "fold to absolute distance |d| (collapse before vs after)",
		onclick: () => { _gravAbs = !_gravAbs; renderAblation(); } }, "|abs|");
	const logBtn = el("button", { class: "mini-toggle", style: on(_gravLogY),
		title: _gravLogY ? "switch to a linear y-axis" : "symlog y-axis \u2014 signed log so the per-token \u0394 spans decades",
		onclick: () => { _gravLogY = !_gravLogY; renderAblation(); } }, "log y");
	const controls = el("div", { style: "display:flex;gap:6px;justify-content:flex-end;align-items:center;margin:-2px 0 6px;flex-wrap:wrap" },
		el("span", { class: "faint", style: "font-size:11px;margin-right:2px", text: "distance from" }),
		refBtn("close", "</prompt>", "signed token distance from the MOVED closing tag (default)"),
		refBtn("open", "<prompt>", "token distance from the FIXED opening tag \u2014 how far INTO the block a sentence sits"),
		absBtn, logBtn);
	const body = el("div", {}, controls, chartHost((w) => gravityScatter(w, model.points, { height: hf(w), ref, abs, logY: _gravLogY }), hf), legend);
	const interp = ref === "open" ? "\u03c1<0 = the <prompt> opener pulls nearby (early) sentences"
		: "\u03c1<0 = a gravity well (closer to </prompt> draws more)";
	const xdesc = ref === "open" ? `${abs ? "|token distance|" : "token distance"} from the fixed <prompt>`
		: `${abs ? "|token distance|" : "signed token distance"} from the moved </prompt>`;
	const kindNote = model.kinds.length > 1
		? ` \u00b7 ${model.kinds.length} step kinds projected (${model.kinds.join(", ")}) \u2014 each baseline-matched to its OWN sentences`
		: "";
	const sub = `each dot = one SENTENCE of one variant \u00b7 x = ${xdesc} \u00b7 y = per-token attention MINUS the no-tag baseline \u00b7 dashed = least-squares trend \u00b7 dist\u2013\u0394 \u03c1=${trend.rho.toFixed(2)} (p=${trend.p.toFixed(3)}, n=${trend.n}) \u2014 ${interp}${kindNote} \u00b7 hover a dot for its sentence`;
	return card("XML gravity \u2014 tag pull vs distance", sub, body);
}

// --- GRAPH (gravity) — distance-binned mean ± 95% CI ---------------------------
// The same points as the scatter, but the cloud is REDUCED to a running mean with a
// 95% CI band over evenly spaced distance bins — a NONLINEAR summary that can show
// a WELL (a peak at the tag, x=0) that a single straight regression cannot. Faint
// raw dots stay for context; thin dashed per-treatment mean lines test whether the
// shape is treatment-independent (they should coincide near x=0 if it is real).
function gravityBinned(w, points, { height = 340, ref = "close", abs = false, logY = false, nBins = 16 } = {}) {
	const W = Math.round(w), H = Math.round(height), fs = fontScale(W), F = (b) => +(b * fs).toFixed(1);
	const padL = Math.round(56 + 10 * fs), padR = 14, padT = 10, padB = Math.round(30 + 14 * fs);
	const px0 = padL, px1 = W - padR, py0 = padT, py1 = H - padB;
	const xs = points.map((p) => gravX(p, ref, abs));
	let xLo = Math.min(0, ...xs), xHi = Math.max(0, ...xs);
	if (!(xHi - xLo > 1e-9)) { xHi += 1; xLo -= 1; }
	// bin a subset over the shared [xLo,xHi]; per bin: mean + SEM → 95% CI.
	const binOf = (subset) => {
		const B = Array.from({ length: nBins }, () => ({ sy: 0, syy: 0, sx: 0, n: 0 }));
		for (const p of subset) {
			const xv = gravX(p, ref, abs);
			let bi = Math.floor(((xv - xLo) / ((xHi - xLo) || 1)) * nBins);
			bi = Math.max(0, Math.min(nBins - 1, bi));
			const b = B[bi]; b.sy += p.delta; b.syy += p.delta * p.delta; b.sx += xv; b.n++;
		}
		return B.map((b) => {
			if (!b.n) return null;
			const mean = b.sy / b.n;
			const varr = b.n > 1 ? Math.max(0, (b.syy - b.sy * b.sy / b.n) / (b.n - 1)) : 0;
			const sem = b.n > 1 ? Math.sqrt(varr / b.n) : 0;
			return { x: b.sx / b.n, mean, lo: mean - 1.96 * sem, hi: mean + 1.96 * sem, n: b.n };
		}).filter(Boolean);
	};
	const overall = binOf(points);
	const yvals = points.map((p) => p.delta).concat(overall.flatMap((s) => [s.lo, s.hi]));
	const absd = yvals.map(Math.abs).filter((v) => v > 0).sort((a, b) => a - b);
	const maxAbs = absd.length ? absd[absd.length - 1] : 1;
	const Lk = Math.max(maxAbs * 1e-3, absd.length ? absd[Math.floor(absd.length / 2)] : maxAbs) || 1e-12;
	const tOf = logY ? (y) => Math.sign(y) * Math.log10(1 + Math.abs(y) / Lk) : (y) => y;
	let tLo = Math.min(0, ...yvals.map(tOf)), tHi = Math.max(0, ...yvals.map(tOf));
	const tp = (tHi - tLo) * 0.08 || 1e-6; tLo -= tp; tHi += tp;
	const X = (x) => px0 + ((x - xLo) / ((xHi - xLo) || 1)) * (px1 - px0);
	const Y = (y) => py1 - ((tOf(y) - tLo) / ((tHi - tLo) || 1)) * (py1 - py0);
	const svg = svgEl("svg", { class: "gsvg", viewBox: `0 0 ${W} ${H}`, width: W, height: H, style: "display:block" });
	for (const gx of niceTicks(xLo, xHi)) {
		const x = X(gx), zero = Math.abs(gx) < 1e-9;
		svg.appendChild(svgEl("line", { x1: x.toFixed(1), y1: py0, x2: x.toFixed(1), y2: py1, stroke: zero ? "rgba(255,255,255,0.3)" : "rgba(255,255,255,0.06)" }));
		svg.appendChild(svgEl("text", { x: x.toFixed(1), y: (py1 + F(13)).toFixed(1), fill: "rgba(220,230,245,0.6)", "font-size": F(10), "text-anchor": "middle" }, String(Math.round(gx))));
	}
	{ const x = X(0);
		svg.appendChild(svgEl("line", { x1: x.toFixed(1), y1: py0, x2: x.toFixed(1), y2: py1, stroke: "rgba(255,255,255,0.32)", "stroke-dasharray": "2 3" }));
		svg.appendChild(svgEl("text", { x: x.toFixed(1), y: (py0 + F(9)).toFixed(1), fill: "rgba(220,230,245,0.72)", "font-size": F(9.5), "text-anchor": "middle" }, ref === "open" ? "<prompt>" : "</prompt>")); }
	let yticks;
	if (logY) { yticks = [0]; for (let v = Lk; v <= maxAbs * 1.5; v *= 10) yticks.push(v, -v); }
	else yticks = niceTicks(tLo, tHi);
	for (const gy of yticks) {
		const y = Y(gy), zero = Math.abs(gy) < 1e-12;
		svg.appendChild(svgEl("line", { x1: px0, y1: y.toFixed(1), x2: px1, y2: y.toFixed(1), stroke: zero ? "rgba(255,255,255,0.3)" : "rgba(255,255,255,0.06)" }));
		svg.appendChild(svgEl("text", { x: (px0 - 5).toFixed(1), y: y.toFixed(1), fill: "rgba(220,230,245,0.6)", "font-size": F(9), "text-anchor": "end", "dominant-baseline": "middle" }, zero ? "0" : gy.toExponential(0)));
	}
	// faint raw dots for context
	points.forEach((p, i) => svg.appendChild(svgEl("circle", { cx: X(xs[i]).toFixed(1), cy: Y(p.delta).toFixed(1), r: (2.2 * fs).toFixed(1), fill: p.color, "fill-opacity": 0.13 })));
	// per-treatment thin dashed mean lines
	const byLv = new Map();
	for (const p of points) { const a = byLv.get(p.level) || byLv.set(p.level, []).get(p.level); a.push(p); }
	for (const [lv, subset] of [...byLv.entries()].sort((a, b) => (_GRAV_RANK[a[0]] ?? 99) - (_GRAV_RANK[b[0]] ?? 99))) {
		const st = binOf(subset); if (st.length < 2) continue;
		const color = GRAVITY_LEVEL_COLOR[lv] || "#7aa2f7";
		let d = ""; st.forEach((s, k) => { d += (k ? "L" : "M") + X(s.x).toFixed(1) + "," + Y(s.mean).toFixed(1) + " "; });
		svg.appendChild(svgEl("path", { d: d.trim(), fill: "none", stroke: color, "stroke-width": 1.2, "stroke-opacity": 0.55, "stroke-dasharray": "3 3" }));
	}
	const wrap = el("div", { class: "chart-host", style: "position:relative;width:100%" });
	const tip = el("div", { class: "graph-tip", style: "max-width:300px" });
	// overall CI band + pooled mean line
	if (overall.length >= 2) {
		let up = "", dn = "";
		overall.forEach((s, k) => { up += (k ? "L" : "M") + X(s.x).toFixed(1) + "," + Y(s.hi).toFixed(1) + " "; });
		for (let k = overall.length - 1; k >= 0; k--) dn += "L" + X(overall[k].x).toFixed(1) + "," + Y(overall[k].lo).toFixed(1) + " ";
		svg.appendChild(svgEl("path", { d: (up + dn + "Z").trim(), fill: "rgba(255,43,43,0.14)", stroke: "none", "pointer-events": "none" }));
		let d = ""; overall.forEach((s, k) => { d += (k ? "L" : "M") + X(s.x).toFixed(1) + "," + Y(s.mean).toFixed(1) + " "; });
		svg.appendChild(svgEl("path", { d: d.trim(), fill: "none", stroke: "#ff2b2b", "stroke-width": 2.5, "stroke-linejoin": "round", "pointer-events": "none" }));
	}
	const showTip = (s, ev) => {
		tip.innerHTML = `<div class="xh">${Math.round(s.x)} tok from ${ref === "open" ? "<prompt>" : "</prompt>"}</div>`
			+ `<div class="r"><span>mean \u0394/token</span><b>${escTip((s.mean >= 0 ? "+" : "") + s.mean.toExponential(2))}</b></div>`
			+ `<div class="r"><span>95% CI</span><b>${escTip(s.lo.toExponential(1))} \u2026 ${escTip(s.hi.toExponential(1))}</b></div>`
			+ `<div class="r"><span>n sentences</span><b>${s.n}</b></div>`;
		const wr = wrap.getBoundingClientRect(); const tw = tip.offsetWidth || 200, th = tip.offsetHeight || 70;
		let left = ev.clientX - wr.left + 14; if (left + tw > wr.width) left = ev.clientX - wr.left - tw - 14;
		let top = ev.clientY - wr.top + 12; if (top + th > wr.height) top = ev.clientY - wr.top - th - 12;
		tip.style.left = Math.max(0, left) + "px"; tip.style.top = Math.max(0, top) + "px"; tip.style.opacity = "1";
	};
	const hideTip = () => { tip.style.opacity = "0"; };
	for (const s of overall) {
		const dot = svgEl("circle", { cx: X(s.x).toFixed(1), cy: Y(s.mean).toFixed(1), r: (3.6 * fs).toFixed(1), fill: "#ff2b2b", style: "cursor:pointer" });
		dot.addEventListener("pointerenter", (e) => showTip(s, e));
		dot.addEventListener("pointermove", (e) => showTip(s, e));
		dot.addEventListener("pointerleave", hideTip);
		svg.appendChild(dot);
	}
	svg.appendChild(svgEl("text", { x: px1, y: (py1 + F(27)).toFixed(1), fill: "rgba(220,230,245,0.82)", "font-size": F(11), "text-anchor": "end" }, gravXLabel(ref, abs)));
	const my = ((py0 + py1) / 2).toFixed(1);
	svg.appendChild(svgEl("text", { x: 13, y: my, fill: "rgba(220,230,245,0.82)", "font-size": F(11), "text-anchor": "middle", transform: `rotate(-90 13 ${my})` }, `mean \u0394 / token${logY ? "  \u00b7 symlog" : ""}`));
	wrap.appendChild(svg); wrap.appendChild(tip);
	return wrap;
}

function ablGravityBinnedCard(rows) {
	const model = gravityModel(rows);
	if (!model.points.length) return null; // the scatter card above already diagnoses the empty state
	const ref = _gravRef === "open" ? "open" : "close";
	const abs = _gravAbs;
	const hf = (w) => Math.round(vh(0.42, 260, 440));
	const levels = [...new Set(model.points.map((p) => p.level))].sort((a, b) => (_GRAV_RANK[a] ?? 99) - (_GRAV_RANK[b] ?? 99));
	const legend = levelLegendEl(levels.map((l) => ({ label: GRAVITY_SHORT[l] || l, color: GRAVITY_LEVEL_COLOR[l] || "#7aa2f7" })));
	const body = el("div", {}, chartHost((w) => gravityBinned(w, model.points, { height: hf(w), ref, abs, logY: _gravLogY }), hf), legend);
	const sub = `the scatter above, REDUCED to a running mean \u00b1 95% CI over distance bins (red line = pooled, band = CI) \u00b7 thin dashed = per-treatment mean \u00b7 a peak at ${ref === "open" ? "<prompt>" : "</prompt>"} (x=0) = a gravity well \u00b7 obeys the distance-from / |abs| / log controls above`;
	return card("XML gravity \u2014 distance-binned mean \u00b1 CI", sub, body);
}

// --- MODEL (gravity, PER-KIND detail) — per-sentence Δ vs no-tag --------------
// Only well-defined for ONE step kind (sentences are template text that differs by
// kind), so it renders only when the scope is a single kind. Model: per cut match
// each treatment to its no-tag baseline, then OHLC the per-cut Δ per sentence.
// Feeds the sentence×treatment heatmap and the per-sentence profile lines; `closes`
// track each treatment's </prompt> on the natural token axis.
function gravityBarModel(rows) {
	const kinds = new Set();
	const noneByCut = new Map();                       // cut -> Map(i -> density)
	const noneSum = new Map(), noneN = new Map();      // sentence i -> pooled no-tag sum / count
	const lvl = new Map();                             // level -> { sum, n, cuts: Map(cut -> Map(i->density)) }
	const tokLen = new Map(), snip = new Map();
	const closeRep = new Map();
	for (const r of rows) {
		const lv = r.gravity || "baseline";
		if (lv === "baseline") continue;
		const g = (r.a && r.a.meta && r.a.meta.gravity) || {};
		const sents = g.sentences || [];
		if (!sents.length) continue;
		kinds.add(r.kind || "?");
		const cut = r.cut;
		const dens = new Map();
		for (const x of gravityTotals(r.a)) dens.set(x.name, x.tokens ? x.mass / x.tokens : 0);
		for (const s of sents) {
			if (s.tok_start != null && s.tok_end != null && !tokLen.has(s.i)) tokLen.set(s.i, Math.max(1, s.tok_end - s.tok_start));
			if (!snip.has(s.i)) snip.set(s.i, s.snippet || `s${s.i}`);
			const d = dens.get(`s${s.i}`); if (d == null) continue;
			if (lv === "none") {
				const m = noneByCut.get(cut) || noneByCut.set(cut, new Map()).get(cut);
				m.set(s.i, d);
				noneSum.set(s.i, (noneSum.get(s.i) || 0) + d); noneN.set(s.i, (noneN.get(s.i) || 0) + 1);
			} else {
				const L = lvl.get(lv) || lvl.set(lv, { sum: new Map(), n: new Map(), cuts: new Map() }).get(lv);
				L.sum.set(s.i, (L.sum.get(s.i) || 0) + d); L.n.set(s.i, (L.n.get(s.i) || 0) + 1);
				const cm = L.cuts.get(cut) || L.cuts.set(cut, new Map()).get(cut);
				cm.set(s.i, d);
			}
		}
		if (lv !== "none" && g.close_tok != null && !closeRep.has(lv)) closeRep.set(lv, { close: g.close_tok, sents });
	}
	if (kinds.size !== 1) return { singleKind: false, kinds: [...kinds].sort() };
	const base = new Map();
	for (const [i, s] of noneSum) base.set(i, s / (noneN.get(i) || 1));
	const idxs = [...tokLen.keys()].sort((a, b) => a - b);
	const sentences = []; const xOf = new Map(); let acc = 0;
	for (const i of idxs) { const wtok = tokLen.get(i); sentences.push({ i, tokLen: wtok, x0: acc, x1: acc + wtok, snippet: snip.get(i) }); xOf.set(i, [acc, acc + wtok]); acc += wtok; }
	const totalTok = acc || 1;
	const levels = [...lvl.keys()].sort((a, b) => (_GRAV_RANK[a] ?? 99) - (_GRAV_RANK[b] ?? 99));
	const candles = [];
	for (const lv of levels) {
		const L = lvl.get(lv);
		for (const [i] of L.sum) {
			const b = base.get(i), pos = xOf.get(i); if (b == null || !pos) continue;
			const perCut = [...L.cuts.entries()].sort((a, b) => a[0] - b[0])
				.map(([cut, m]) => {
					const nb = noneByCut.get(cut)?.get(i);
					return { cut, delta: (m.get(i) ?? 0) - (nb != null ? nb : b) };
				});
			const deltas = perCut.map((p) => p.delta);
			const mean = L.sum.get(i) / (L.n.get(i) || 1) - b;
			let open, high, low, close;
			if (deltas.length) {
				open = deltas[0]; close = deltas[deltas.length - 1];
				high = Math.max(...deltas); low = Math.min(...deltas);
			} else { open = high = low = close = mean; }
			candles.push({ level: lv, i, x0: pos[0], x1: pos[1], tokLen: tokLen.get(i), snippet: snip.get(i),
				color: GRAVITY_LEVEL_COLOR[lv] || "#7aa2f7", open, high, low, close, mean, raw: L.sum.get(i) / (L.n.get(i) || 1), nCuts: deltas.length });
		}
	}
	const closes = [];
	for (const lv of levels) {
		const rep = closeRep.get(lv); if (!rep) continue;
		let jIdx = null;
		for (const s of rep.sents) if (s.tok_end != null && s.tok_end <= rep.close) jIdx = s.i;
		const pos = jIdx != null ? xOf.get(jIdx) : null;
		closes.push({ level: lv, x: pos ? pos[1] : totalTok, color: GRAVITY_LEVEL_COLOR[lv] || "#7aa2f7" });
	}
	return { singleKind: true, kind: [...kinds][0], sentences, candles, closes, totalTok, levels, hasBaseline: base.size > 0 };
}

// --- GRAPH (gravity, PER-KIND) — per-sentence PROFILE LINES (one per treatment) -
// Same natural-token x as the candles, but each treatment is a LINE across the
// sentences (y = per-token Δ vs no-tag), with a translucent band = its cut range
// (low‥high). Overlapping LINES separate far better than overlapping candles, so
// trends read at a glance. Dashed colored verticals = each treatment's </prompt>.
function gravityProfiles(w, model, { height = 320, logY = false, levels = null } = {}) {
	const visLv = (levels || model.levels).filter((l) => model.candles.some((c) => c.level === l));
	const W = Math.round(w), H = Math.round(height), fs = fontScale(W), F = (b) => +(b * fs).toFixed(1);
	const padL = Math.round(56 + 10 * fs), padR = 14, padT = Math.round(16 + 6 * fs), padB = Math.round(28 + 14 * fs);
	const px0 = padL, px1 = W - padR, py0 = padT, py1 = H - padB;
	const cand = model.candles.filter((c) => visLv.includes(c.level));
	const vals = cand.flatMap((c) => [c.high, c.low, c.mean]);
	const absd = vals.map(Math.abs).filter((v) => v > 0).sort((a, b) => a - b);
	const maxAbs = absd.length ? absd[absd.length - 1] : 1;
	const Lk = Math.max(maxAbs * 1e-3, absd.length ? absd[Math.floor(absd.length / 2)] : maxAbs) || 1e-12;
	const tOf = logY ? (y) => Math.sign(y) * Math.log10(1 + Math.abs(y) / Lk) : (y) => y;
	let tLo = Math.min(0, ...vals.map(tOf)), tHi = Math.max(0, ...vals.map(tOf));
	const tp = (tHi - tLo) * 0.08 || 1e-6; tLo -= tp; tHi += tp;
	const ctr = (c) => (c.x0 + c.x1) / 2;
	const X = (x) => px0 + (x / model.totalTok) * (px1 - px0);
	const Y = (y) => py1 - ((tOf(y) - tLo) / ((tHi - tLo) || 1)) * (py1 - py0);
	const svg = svgEl("svg", { class: "gsvg", viewBox: `0 0 ${W} ${H}`, width: W, height: H, style: "display:block" });
	let yticks;
	if (logY) { yticks = [0]; for (let v = Lk; v <= maxAbs * 1.5; v *= 10) yticks.push(v, -v); }
	else yticks = niceTicks(tLo, tHi);
	for (const gy of yticks) {
		const y = Y(gy), zero = Math.abs(gy) < 1e-12;
		svg.appendChild(svgEl("line", { x1: px0, y1: y.toFixed(1), x2: px1, y2: y.toFixed(1), stroke: zero ? "rgba(255,255,255,0.32)" : "rgba(255,255,255,0.06)" }));
		svg.appendChild(svgEl("text", { x: (px0 - 5).toFixed(1), y: y.toFixed(1), fill: "rgba(220,230,245,0.6)", "font-size": F(9), "text-anchor": "end", "dominant-baseline": "middle" }, zero ? "0" : gy.toExponential(0)));
	}
	for (const s of model.sentences) { const x = X(s.x1); svg.appendChild(svgEl("line", { x1: x.toFixed(1), y1: py0, x2: x.toFixed(1), y2: py1, stroke: "rgba(255,255,255,0.05)" })); }
	const wrap = el("div", { class: "chart-host", style: "position:relative;width:100%" });
	const tip = el("div", { class: "graph-tip", style: "max-width:360px" });
	const showTip = (c, ev) => {
		const ohlc = c.nCuts > 1 ? `<div class="r"><span>range (cuts)</span><b>${escTip(c.low.toExponential(1))} \u2026 ${escTip(c.high.toExponential(1))}</b></div>` : "";
		tip.innerHTML = `<div class="xh">${escTip(GRAVITY_SHORT[c.level] || c.level)} \u00b7 ${escTip(model.kind)} \u00b7 s${c.i}</div>`
			+ `<div class="r"><span>\u0394 / token</span><b>${escTip((c.mean >= 0 ? "+" : "") + c.mean.toExponential(2))}</b></div>` + ohlc
			+ `<div style="margin-top:6px;white-space:normal;line-height:1.35;opacity:.92">${escTip(c.snippet || "")}</div>`;
		const wr = wrap.getBoundingClientRect(); const tw = tip.offsetWidth || 220, th = tip.offsetHeight || 90;
		let left = ev.clientX - wr.left + 14; if (left + tw > wr.width) left = ev.clientX - wr.left - tw - 14;
		let top = ev.clientY - wr.top + 12; if (top + th > wr.height) top = ev.clientY - wr.top - th - 12;
		tip.style.left = Math.max(0, left) + "px"; tip.style.top = Math.max(0, top) + "px"; tip.style.opacity = "1";
	};
	const hideTip = () => { tip.style.opacity = "0"; };
	const byLv = new Map();
	for (const c of cand) { const a = byLv.get(c.level) || byLv.set(c.level, []).get(c.level); a.push(c); }
	for (const lv of visLv) {
		const arr = (byLv.get(lv) || []).slice().sort((a, b) => ctr(a) - ctr(b));
		if (!arr.length) continue;
		const color = arr[0].color;
		if (arr.length > 1) {
			let up = "", dn = "";
			arr.forEach((c, k) => { up += (k ? "L" : "M") + X(ctr(c)).toFixed(1) + "," + Y(c.high).toFixed(1) + " "; });
			for (let k = arr.length - 1; k >= 0; k--) dn += "L" + X(ctr(arr[k])).toFixed(1) + "," + Y(arr[k].low).toFixed(1) + " ";
			svg.appendChild(svgEl("path", { d: (up + dn + "Z").trim(), fill: hexA(color, 0.12), stroke: "none", "pointer-events": "none" }));
		}
		let d = ""; arr.forEach((c, k) => { d += (k ? "L" : "M") + X(ctr(c)).toFixed(1) + "," + Y(c.mean).toFixed(1) + " "; });
		svg.appendChild(svgEl("path", { d: d.trim(), fill: "none", stroke: color, "stroke-width": 2.2, "stroke-linejoin": "round" }));
		for (const c of arr) {
			const dot = svgEl("circle", { cx: X(ctr(c)).toFixed(1), cy: Y(c.mean).toFixed(1), r: (3.4 * fs).toFixed(1), fill: color, style: "cursor:pointer" });
			dot.addEventListener("pointerenter", (e) => showTip(c, e));
			dot.addEventListener("pointermove", (e) => showTip(c, e));
			dot.addEventListener("pointerleave", hideTip);
			svg.appendChild(dot);
		}
	}
	for (const cl of model.closes.filter((c) => visLv.includes(c.level))) {
		const x = X(cl.x);
		svg.appendChild(svgEl("line", { x1: x.toFixed(1), y1: py0, x2: x.toFixed(1), y2: py1, stroke: cl.color, "stroke-width": 2, "stroke-dasharray": "4 3" }));
		svg.appendChild(svgEl("text", { x: x.toFixed(1), y: (py0 - 3).toFixed(1), fill: cl.color, "font-size": F(9.5), "text-anchor": "middle" }, `\u2039/\u203a ${GRAVITY_SHORT[cl.level] || cl.level}`));
	}
	svg.appendChild(svgEl("text", { x: px1, y: (py1 + F(25)).toFixed(1), fill: "rgba(220,230,245,0.82)", "font-size": F(11), "text-anchor": "end" }, "token position in block (natural order) \u25ba"));
	const my = ((py0 + py1) / 2).toFixed(1);
	svg.appendChild(svgEl("text", { x: 13, y: my, fill: "rgba(220,230,245,0.82)", "font-size": F(11), "text-anchor": "middle", transform: `rotate(-90 13 ${my})` }, `\u0394 attention / token${logY ? "  \u00b7 symlog" : ""}`));
	wrap.appendChild(svg); wrap.appendChild(tip);
	return wrap;
}

function ablGravityProfileCard(rows) {
	const model = gravityBarModel(rows);
	if (!model.singleKind || !model.hasBaseline || !model.candles.length) return null; // the candle card covers the empty/multi-kind states
	const visLv = model.levels.filter((l) => !_gravBarOff.has(l));
	const hf = (w) => Math.round(vh(0.42, 260, 460));
	const legend = chartLegend(model.levels.map((l) => ({ key: l, label: GRAVITY_SHORT[l] || l, color: GRAVITY_LEVEL_COLOR[l] || "#7aa2f7" })), {
		onToggle: (key) => { _gravBarOff.has(key) ? _gravBarOff.delete(key) : _gravBarOff.add(key); renderAblation(); },
		off: _gravBarOff,
	});
	const chart = visLv.length
		? chartHost((w) => gravityProfiles(w, model, { height: hf(w), logY: _gravLogY, levels: visLv }), hf)
		: empty("all treatments hidden \u2014 click a legend swatch to show one");
	const body = el("div", {},
		el("div", { class: "faint", style: "font-size:11px;margin:-2px 0 6px;text-align:right" }, "click a swatch to filter treatments"),
		chart, legend);
	const sub = `${model.kind} \u00b7 one LINE per treatment across sentences (natural token order) \u00b7 band = spread across cuts (low\u2013high) \u00b7 y = per-token \u0394 vs no-tag${_gravLogY ? " (symlog)" : ""} \u00b7 dashed verticals = </prompt> close \u00b7 hover a marker for its sentence`;
	return card("XML gravity \u2014 per-sentence profiles", sub, body);
}

// Heatmap color for a signed, unit-normalized value (t∈[-1,1]): CYAN = low (below
// the no-tag baseline), YELLOW = high (above), through a dim slate mid so near-zero
// cells recede; opacity grows with |t| so the strongest pulls read first.
function gravDivColor(t) {
	t = Math.max(-1, Math.min(1, t || 0));
	const high = [250, 204, 21], low = [34, 211, 238], mid = [120, 134, 156];
	const lerp = (a, b, u) => a.map((v, i) => Math.round(v + (b[i] - v) * u));
	const c = t >= 0 ? lerp(mid, high, t) : lerp(mid, low, -t);
	return `rgba(${c[0]},${c[1]},${c[2]},${(0.16 + 0.74 * Math.abs(t)).toFixed(3)})`;
}
// SEQUENTIAL heatmap color for a non-negative unit-normalized value (t∈[0,1]) — used
// for RAW per-token attention (no baseline subtracted, so everything is ≥0): dim
// slate (low) → bright yellow (high), opacity growing with magnitude.
function gravSeqColor(t) {
	t = Math.max(0, Math.min(1, t || 0));
	const hi = [250, 204, 21], lo = [40, 52, 71];
	const c = lo.map((v, i) => Math.round(v + (hi[i] - v) * t));
	return `rgba(${c[0]},${c[1]},${c[2]},${(0.2 + 0.75 * t).toFixed(3)})`;
}

// --- GRAPH (gravity, PER-KIND) — sentence × treatment HEATMAP ------------------
// Rows = sentences (natural order, top→bottom), columns = treatments (q1→q4 by
// close position). Cell = per-token Δ vs no-tag (diverging color). A per-column
// "waterline" marks where that treatment's </prompt> sits — scanning columns
// left→right it descends, and a gravity well shows as yellow cells hugging the line.
function gravityHeatmap(w, model, { rowH = 22, levels = null, raw = false } = {}) {
	const visLv = (levels || model.levels).filter((l) => model.candles.some((c) => c.level === l));
	const sents = model.sentences;
	const valOf = (c) => (raw ? c.raw : c.mean);
	const W = Math.round(w), fs = fontScale(W), F = (b) => +(b * fs).toFixed(1);
	const padL = Math.round(120 * Math.min(1.15, Math.max(0.85, fs))), padR = 16, padT = 26, padB = 22;
	const H = Math.round(padT + padB + Math.max(1, sents.length) * rowH);
	const px0 = padL, px1 = W - padR, py0 = padT, py1 = H - padB;
	const nCol = Math.max(1, visLv.length), nRow = Math.max(1, sents.length);
	const cw = (px1 - px0) / nCol, ch = (py1 - py0) / nRow;
	const cell = new Map(); let mx = 0;
	for (const c of model.candles) { if (!visLv.includes(c.level)) continue; const v = valOf(c); cell.set(`${c.level}\u0000${c.i}`, v); mx = Math.max(mx, Math.abs(v)); }
	mx = mx || 1;
	const svg = svgEl("svg", { class: "gsvg", viewBox: `0 0 ${W} ${H}`, width: W, height: H, style: "display:block" });
	const wrap = el("div", { class: "chart-host", style: "position:relative;width:100%" });
	const tip = el("div", { class: "graph-tip", style: "max-width:360px" });
	const place = (ev) => {
		const wr = wrap.getBoundingClientRect(); const tw = tip.offsetWidth || 220, th = tip.offsetHeight || 90;
		let left = ev.clientX - wr.left + 14; if (left + tw > wr.width) left = ev.clientX - wr.left - tw - 14;
		let top = ev.clientY - wr.top + 12; if (top + th > wr.height) top = ev.clientY - wr.top - th - 12;
		tip.style.left = Math.max(0, left) + "px"; tip.style.top = Math.max(0, top) + "px"; tip.style.opacity = "1";
	};
	const showTip = (lv, s, v, ev) => {
		tip.innerHTML = `<div class="xh">${escTip(GRAVITY_SHORT[lv] || lv)} \u00b7 ${escTip(model.kind)} \u00b7 s${s.i}</div>`
			+ `<div class="r"><span>${raw ? "attn / token" : "\u0394 / token"}</span><b>${v == null ? "\u2014" : escTip((!raw && v >= 0 ? "+" : "") + v.toExponential(2))}</b></div>`
			+ `<div style="margin-top:6px;white-space:normal;line-height:1.35;opacity:.92">${escTip(s.snippet || "")}</div>`;
		place(ev);
	};
	// Row-label hover: the y-axis shows a truncated head, so reveal the FULL
	// sentence (same tip surface as the cells) on hover of the label itself.
	const showRowTip = (s, ev) => {
		tip.innerHTML = `<div class="xh">${escTip(model.kind)} \u00b7 s${s.i}</div>`
			+ `<div style="margin-top:6px;white-space:normal;line-height:1.35;opacity:.92">${escTip(s.snippet || `sentence ${s.i}`)}</div>`;
		place(ev);
	};
	const hideTip = () => { tip.style.opacity = "0"; };
	visLv.forEach((lv, ci) => svg.appendChild(svgEl("text", { x: (px0 + (ci + 0.5) * cw).toFixed(1), y: (py0 - 9).toFixed(1), fill: GRAVITY_LEVEL_COLOR[lv] || "#cfe8d0", "font-size": F(11), "text-anchor": "middle", "font-weight": "600" }, GRAVITY_SHORT[lv] || lv)));
	sents.forEach((s, ri) => {
		const y = py0 + ri * ch;
		// Truncated (ellipsis) + hoverable: the full sentence would overflow the
		// axis gutter, so show a short head and reveal the rest on hover.
		const full = s.snippet || `sentence ${s.i}`;
		const label = full.length > 20 ? full.slice(0, 19) + "\u2026" : full;
		const lbl = svgEl("text", { x: (px0 - 8).toFixed(1), y: (y + ch / 2).toFixed(1), fill: "rgba(220,230,245,0.72)", "font-size": F(9.5), "text-anchor": "end", "dominant-baseline": "middle", style: "cursor:pointer" }, label);
		lbl.addEventListener("pointerenter", (e) => showRowTip(s, e));
		lbl.addEventListener("pointermove", (e) => showRowTip(s, e));
		lbl.addEventListener("pointerleave", hideTip);
		svg.appendChild(lbl);
		visLv.forEach((lv, ci) => {
			const v = cell.get(`${lv}\u0000${s.i}`), x = px0 + ci * cw;
			const rect = svgEl("rect", { x: x.toFixed(1), y: y.toFixed(1), width: Math.max(1, cw - 1).toFixed(1), height: Math.max(1, ch - 1).toFixed(1),
				fill: v == null ? "rgba(255,255,255,0.03)" : (raw ? gravSeqColor(v / mx) : gravDivColor(v / mx)), style: "cursor:pointer" });
			rect.addEventListener("pointerenter", (e) => showTip(lv, s, v, e));
			rect.addEventListener("pointermove", (e) => showTip(lv, s, v, e));
			rect.addEventListener("pointerleave", hideTip);
			svg.appendChild(rect);
		});
	});
	for (const cl of model.closes.filter((c) => visLv.includes(c.level))) {
		const ci = visLv.indexOf(cl.level); if (ci < 0) continue;
		let rowPos = 0; for (const s of sents) { if (cl.x >= s.x1 - 1e-6) rowPos++; else break; }
		const y = py0 + rowPos * ch, x = px0 + ci * cw;
		svg.appendChild(svgEl("line", { x1: x.toFixed(1), y1: y.toFixed(1), x2: (x + cw).toFixed(1), y2: y.toFixed(1), stroke: cl.color, "stroke-width": 2.5 }));
	}
	wrap.appendChild(svg); wrap.appendChild(tip);
	return wrap;
}

function ablGravityHeatmapCard(rows) {
	const model = gravityBarModel(rows);
	if (!model.singleKind || !model.hasBaseline || !model.candles.length) return null;
	const visLv = model.levels.filter((l) => !_gravBarOff.has(l));
	const rowH = 22, hh = 26 + 22 + Math.max(1, model.sentences.length) * rowH;
	const legend = chartLegend(model.levels.map((l) => ({ key: l, label: GRAVITY_SHORT[l] || l, color: GRAVITY_LEVEL_COLOR[l] || "#7aa2f7" })), {
		onToggle: (key) => { _gravBarOff.has(key) ? _gravBarOff.delete(key) : _gravBarOff.add(key); renderAblation(); },
		off: _gravBarOff,
	});
	const raw = _gravRawHeat;
	const chart = visLv.length
		? chartHost((w) => gravityHeatmap(w, model, { rowH, levels: visLv, raw }), () => hh)
		: empty("all treatments hidden \u2014 click a legend swatch to show one");
	const rawBtn = el("button", { class: "mini-toggle",
		style: raw ? "background:rgba(107,217,110,0.18);border-color:rgba(107,217,110,0.55);color:#cfe8d0" : "",
		title: raw ? "showing RAW per-token attention (no-tag baseline NOT subtracted) \u2014 click to subtract it (\u0394)" : "subtract the no-tag baseline (\u0394, default) \u2014 click to show RAW per-token attention instead",
		onclick: () => { _gravRawHeat = !_gravRawHeat; renderAblation(); } }, raw ? "raw attention" : "\u0394 vs no-tag");
	const grad = raw
		? "linear-gradient(90deg, rgba(40,52,71,0.6), rgba(250,204,21,0.95))"
		: "linear-gradient(90deg, rgba(34,211,238,0.9), rgba(120,134,156,0.4), rgba(250,204,21,0.9))";
	const scale = el("div", { class: "faint", style: "display:flex;gap:10px;align-items:center;font-size:11px;justify-content:flex-end;margin:-2px 0 6px" },
		rawBtn, el("span", { style: "flex:1" }),
		el("span", {}, raw ? "low" : "low (below baseline)"),
		el("span", { style: `width:120px;height:10px;border-radius:3px;background:${grad}` }),
		el("span", {}, raw ? "high" : "high (above)"));
	const body = el("div", {}, scale, chart, legend);
	const sub = raw
		? `${model.kind} \u00b7 rows = sentences (natural order) \u00d7 columns = treatments \u00b7 color = RAW per-token attention (no-tag baseline NOT subtracted) \u00b7 the colored waterline in each column = that treatment's </prompt> (it descends q1\u2192q4) \u00b7 hover a cell, or a row label for the full sentence`
		: `${model.kind} \u00b7 rows = sentences (natural order) \u00d7 columns = treatments \u00b7 color = per-token \u0394 vs no-tag (cyan = low / below baseline, yellow = high / above) \u00b7 the colored waterline in each column = that treatment's </prompt> (it descends q1\u2192q4) \u00b7 hover a cell, or a row label for the full sentence`;
	return card("XML gravity \u2014 sentence \u00d7 treatment heatmap", sub, body);
}

const emptyBig = (msg, sub, link) => el("div", { class: "empty" },
	el("span", { class: "big", text: "\u2697" }), el("div", { text: msg }),
	sub ? el("div", { class: "faint", style: "margin-top:6px", text: sub }) : null,
	link ? el("div", { style: "margin-top:10px" }, el("a", { class: "pill", href: "/tf-legacy", text: "open legacy ablation \u2197" })) : null);

// The CHOOSE-RUN-FIRST gate: nothing scans or loads until you pick a run here.
// Each button is one launched experiment (label) with its derived axis + count.
function runPickerGate(runs) {
	return el("div", { class: "empty", style: "max-width:640px;margin:0 auto" },
		el("span", { class: "big", text: "\u2697" }),
		el("div", { text: "Choose a run to visualize" }),
		el("div", { class: "faint", style: "margin:6px 0 14px", text: "each run is one launched experiment — its variants + the matching graphs load only after you pick it." }),
		el("div", { style: "display:flex;flex-wrap:wrap;gap:8px;justify-content:center" },
			...runs.map((r) => el("button", {
				class: "abl-run-pick",
				style: "display:flex;flex-direction:column;align-items:flex-start;gap:2px;padding:8px 12px;border:1px solid var(--line,#333);border-radius:8px;background:transparent;color:inherit;cursor:pointer;min-width:150px",
				onclick: () => { state.ablRun = r.label; renderAblation(); },
			},
				el("span", { style: "font-weight:600", text: r.label || "(no label)" }),
				el("span", { class: "faint", style: "font-size:12px", text: `${axisLabel(r.axis)} · ${r.n} variant${r.n === 1 ? "" : "s"}` })))));
}

export async function renderAblation() {
	const host = $("ablation-inner");
	if (!host) return;
	const token = bumpLoad();
	const dl = $("dv-loading");
	if (dl) dl.classList.add("on");
	const finish = () => { if (dl && token === state.loadToken) dl.classList.remove("on"); };

	// progress UI (determinate: variants loaded / total)
	const fill = el("div", { class: "abl-progress-fill", style: "width:0%" });
	const label = el("div", { class: "abl-progress-label", text: "discovering ablation variants…" });
	host.replaceChildren(el("div", { class: "abl-loading" }, label, el("div", { class: "abl-progress" }, fill)));
	const setProg = (d, t, msg) => { fill.style.width = t ? `${Math.round((100 * d) / t)}%` : "0%"; label.textContent = msg || `loading variants ${d}/${t}…`; };

	try {
		if (state._ablCell !== cellKey()) { state.variants = await loadVariants(); state._ablCell = cellKey(); state.variantRows = new Map(); _probe.clear(); _loadAll = false; state.ablRun = null; }
	} catch (e) { if (token === state.loadToken) { host.replaceChildren(empty(`failed to load variants: ${e.message}`)); finish(); } return; }
	if (token !== state.loadToken) return;

	// RUN FIRST: nothing scans or loads until you explicitly pick which run
	// (experiment) to visualize; the graphs then group by the axis that run varied.
	const runs = cellRuns();
	if (!runs.length) {
		const combos = state._ablCombos || [];
		host.replaceChildren(el("div", { class: "empty" },
			el("span", { class: "big", text: "\u2697" }),
			el("div", { text: "no ablation variants for this run / scene / model." }),
			el("div", { class: "faint", style: "margin:10px 0 6px", text: combos.length ? "variants exist for these cells — switch run / scene / model above:" : "ablation variants are created in the ⚗ ablation drawer." }),
			...combos.slice(0, 8).map((c) => el("div", { class: "muted", style: "font-family:ui-monospace,Menlo,monospace;font-size:12.5px", text: `${c.k}  ·  ${c.n} variants` }))));
		finish();
		return;
	}
	if (state.ablRun == null || !runs.some((r) => r.label === state.ablRun)) { host.replaceChildren(runPickerGate(runs)); finish(); return; }
	const axis = (runs.find((r) => r.label === state.ablRun) || {}).axis || "method";

	const scoped = scopedVariants();
	if (!scoped.length) {
		host.replaceChildren(toolbar(runs, axis), emptyBig("no variants in this run match the region / step selection.", "switch the run above, or widen the region / step selection (ALL regions / ALL steps).", false));
		finish();
		return;
	}

	// PHASE 1 — scan every scoped variant cheaply to find which have attention.
	setProg(0, scoped.length, `scanning ${scoped.length} variant${scoped.length === 1 ? "" : "s"}…`);
	let scanned = 0; const withAttn = [];
	await pool(scoped, 8, async (v) => {
		const ev = await probeVariant(v);
		scanned++;
		if (token === state.loadToken) setProg(scanned, scoped.length, `scanning ${scanned}/${scoped.length} variants…`);
		if (token === state.loadToken && ev != null) withAttn.push({ v, ev });
	});
	if (token !== state.loadToken) return;
	if (!withAttn.length) { host.replaceChildren(toolbar(runs, axis), emptyBig(`${scoped.length} variant${scoped.length === 1 ? "" : "s"} in scope, but none have computed attention.`, "compute their attention in the ⚗ ablation drawer, then reload.", true)); finish(); return; }

	// PHASE 2 — establish light vs heavy by loading the first computed variant,
	// then load the rest. The light `abl` view loads EVERY variant (~50KB each);
	// the heavy `agg` fallback (older server, pre-restart) is capped for safety.
	setProg(0, withAttn.length, `loading ${withAttn.length} computed variant${withAttn.length === 1 ? "" : "s"}…`);
	await loadAgg(withAttn[0].v, withAttn[0].ev); // sets _liteMode + caches
	if (token !== state.loadToken) return;
	const lite = _liteMode === true;
	const capN = lite ? withAttn.length : (_loadAll ? HARD_MAX : MAX_VARIANTS);
	const capped = withAttn.length <= capN ? withAttn : [...withAttn].sort((a, b) => b.v.cut - a.v.cut).slice(0, capN);
	const rows = []; let done = 0;
	await pool(capped, lite ? 8 : 5, async ({ v, ev }) => {
		const r = await loadAgg(v, ev);
		done++;
		if (token === state.loadToken) setProg(done, capped.length, `loading ${done}/${capped.length} computed variants…`);
		if (token === state.loadToken && r) rows.push(r);
	});
	if (token !== state.loadToken) return;
	if (!rows.length) { host.replaceChildren(toolbar(runs, axis), emptyBig("could not load attention for the computed variants.", "try reloading, or recompute in the ⚗ ablation drawer.", true)); finish(); return; }
	rows.sort((a, b) => a.cut - b.cut || a.method.localeCompare(b.method));

	// Base-cell baseline (coord + schema axes): both baselines are the base cell's
	// OWN behaviour — L/G→L for coord, soft-JSON for schema — never launched as a
	// variant. Pull the BASE run's already-computed attention at each fork point
	// (cut) and add it as the `baseline` level so it appears alongside the launched
	// conditions. Skipped per-cut when that base step wasn't computed. (The
	// structure graph only shows it if the base step carries the attr_role split —
	// analysis v9+; otherwise it's simply absent there.)
	const nVar = rows.length; // launched variants, before the base-cell baseline is appended
	let baselineN = 0;
	if (axis === "coord" || axis === "schema") {
		const cuts = [...new Set(rows.filter((r) => !r._base).map((r) => r.cut))];
		await pool(cuts, lite ? 8 : 5, async (cut) => {
			const r = await loadBaseStep(cut);
			if (token === state.loadToken && r) { rows.push(r); baselineN++; }
		});
		if (token !== state.loadToken) return;
	}

	let note;
	if (lite) {
		note = el("div", { class: "abl-note muted", text: `${nVar} computed variant${nVar === 1 ? "" : "s"} · ${scoped.length} in scope · optimized (light) view` });
	} else if (withAttn.length > capped.length && !_loadAll) {
		note = el("div", { class: "abl-note muted" },
			el("span", { text: `${nVar} of ${withAttn.length} computed variants shown · heavy view (~4MB each). ` }),
			el("button", { class: "mini-toggle", title: `load up to ${HARD_MAX} variants`, text: `load ${Math.min(HARD_MAX, withAttn.length)}`, onclick: () => { _loadAll = true; renderAblation(); } }),
			el("span", { text: " · restart the API server to enable the optimized view (loads all)" }));
	} else if (withAttn.length > capped.length) {
		note = el("div", { class: "abl-note muted", text: `${nVar} of ${withAttn.length} shown (heavy view, max ${HARD_MAX}) · restart the API server for the optimized view to load all` });
	} else {
		note = el("div", { class: "abl-note muted", text: `${nVar} computed variant${nVar === 1 ? "" : "s"} · ${scoped.length} in scope` });
	}
	if (baselineN) { const bn = axis === "schema" ? "soft-JSON" : "LG2L"; note.appendChild(el("span", { text: ` · +${bn} baseline (from base cell, ${baselineN} step${baselineN === 1 ? "" : "s"})` })); }
	// The ONE shared treatment-color key, pinned to the top of the view (sticky) so
	// it applies to every graph below (the spiders have no per-series legend of
	// their own) and stays visible while scrolling.
	const topLegend = levelLegendEl(levelsPresent(rows, axis), { sticky: true });
	// XML-gravity is a FOCUSED view: only its dedicated tag-distance readout. The
	// scene-context graphs (attribute profile spider, structure-vs-content, token
	// types) are incidental on a gravity variant and live in the Data tab for the
	// main sequence instead — so the gravity view stays uncluttered.
	if (axis === "gravity") {
		// Scope-driven layout: ALL step kinds -> the aggregate distance readout
		// (binned mean ± CI) ONLY; a SINGLE step kind (zoomed via the region/step
		// picker) -> the per-sentence detail (improved sentence×treatment heatmap +
		// profile lines). The natural-order candles were retired. An empty scope
		// falls back to the scatter card, which carries the per-cause "why is this
		// empty" diagnosis.
		const gravRows = gravityRows(rows);
		const nSkip = rows.length - gravRows.length;
		const kindFilter = el("div", { style: "display:flex;gap:8px;justify-content:flex-end;align-items:center;margin:2px 0 8px;flex-wrap:wrap" },
			el("span", { class: "faint", style: "font-size:11px", text: "step kinds" }),
			gravNextObjectToggle(),
			nSkip ? el("span", { class: "faint", style: "font-size:11px", text: nSkip === rows.length ? "all variants are next_object — toggle to include" : `${nSkip} next_object variant${nSkip === 1 ? "" : "s"} excluded` }) : null);
		const detail = gravityBarModel(gravRows).singleKind
			? [ablGravityHeatmapCard(gravRows), ablGravityProfileCard(gravRows)].filter(Boolean)
			: [];
		let gravCards;
		if (detail.length) gravCards = detail;                                    // single kind: per-sentence detail
		else if (gravityModel(gravRows).points.length) gravCards = [ablGravityBinnedCard(gravRows)];  // all kinds: binned mean ± CI only
		else gravCards = [ablGravityCard(gravRows)];                              // nothing in scope: diagnostic empty state
		host.replaceChildren(toolbar(runs, axis), note, topLegend, kindFilter, ...gravCards);
		finish();
		return;
	}
	// Graphs: (1) attribute profile spider (+ a combined-origin version for the
	// coordinate axis), (2) structure-vs-content as per-attribute stacked bars, and
	// (3) the word/token-type comparison as bars — LAST, per the requested layout.
	const attrRow = el("div", { class: "dv-row" },
		ablSpiderCard(rows, axis, false),
		...(axis === "coord" ? [ablSpiderCard(rows, axis, true)] : []));
	host.replaceChildren(
		toolbar(runs, axis), note, topLegend,
		...(axis === "method" ? [ablOrderScatterCard(rows, axis), ablCompositionCard(rows, axis)] : []),
		attrRow,
		ablStructureCard(rows, axis),
		ablTypeCard(rows, axis));
	finish();
}
