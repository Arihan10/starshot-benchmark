// Data layer for the /tf data view: loads a cell's steps + attention status,
// resolves the region/step nav selection into a set of steps, fetches their
// (lightweight, token-free) `agg` analyses, and provides the aggregation helpers
// each card consumes. Every card here works off the `agg` view — the compact
// payload minus the heavy per-token array — which still carries scene entities,
// the per-entity/attribute rollups, the per-output-item rollups, and the buckets.

import { api } from "../../js/api.js";
import { state, ALL, entityKindLabel, buildCellObs } from "./state.js";

// Bounded-concurrency map (so an ALL/ALL selection doesn't fire 30 fetches at once).
export async function pool(items, size, fn) {
	const q = [...items]; const workers = [];
	for (let i = 0; i < Math.min(size, q.length); i++) {
		workers.push((async () => { while (q.length) { const it = q.shift(); await fn(it); } })());
	}
	await Promise.all(workers);
}

const stepTemplate = (s) => s.template ?? s.step ?? "?";
const stepNode = (s) => s.node ?? "?";

// --- loading -----------------------------------------------------------------

export async function loadRuns() {
	const data = await api.runs();
	const runs = Array.isArray(data) ? data : (data.runs ?? []);
	// Hide ablation variants from the top-level picker (they live under their base).
	return runs.map((r) => (typeof r === "string" ? r : r.name)).filter((n) => !n.includes("__abl-"));
}

export async function loadCell(run) {
	const data = await api.slots(run);
	state.slots = data.slots ?? [];
	state.models = data.models ?? [];
	return { slots: state.slots, models: state.models, defaultModel: data.defaultModel };
}

// Fetch this cell's steps, event history (for frame recovery), and attention
// status together. Populates state.steps / state.obs / state.attnStatus.
export async function loadSteps(run, slot, model) {
	const [stepsResp, events, computed] = await Promise.all([
		api.tfSteps(run, slot, model),
		api.eventsHistory(run, slot, model),
		api.attentionList(run, slot, model, {}).catch(() => ({ computed: [], stale: [] })),
	]);
	state.steps = stepsResp.steps ?? [];
	state.events = events ?? [];
	state.obs = buildCellObs(events ?? []);
	state.attnStatus = {};
	for (const ev of computed.computed ?? []) state.attnStatus[ev] = "ready";
	for (const ev of computed.stale ?? []) if (!state.attnStatus[ev]) state.attnStatus[ev] = "stale";
	return state.steps;
}

// The logged cache.llm call for a step (its exact system/user prompts + reasoning
// + output), used by the per-step prompt viewer.
export function stepLLM(ev) {
	return (state.events || []).find((e) => e && e.kind === "cache.llm" && e.index === ev) || null;
}

// Step kinds that carry scene context (the only ones that can be ablated).
export const SCENE_CONTEXT_KINDS = new Set([
	"zone_plan", "zone_decompose", "anchor_decompose", "encapsulating_decompose",
	"negative_space_decompose", "object_bbox_batch", "child_bbox_batch", "next_object",
]);

// Steps that carry a stored attention analysis (ready or stale) AND have scene
// context — the only ones a graph can draw.
export function computedSteps() {
	return state.steps.filter((s) => {
		const st = state.attnStatus[s.event_index];
		return (st === "ready" || st === "stale") && s.has_scene !== false;
	});
}

// Region options: distinct scene nodes among computed steps, in pipeline order.
export function regionOptions() {
	const seen = new Map();
	for (const s of computedSteps()) {
		const node = stepNode(s);
		if (!seen.has(node)) seen.set(node, { id: node, count: 0, ord: s.event_index });
		seen.get(node).count++;
	}
	return [...seen.values()].sort((a, b) => a.ord - b.ord);
}

// Step options for the current region:
//   ALL region  → distinct step KINDS (templates), aggregated across the scene.
//   a region    → that node's individual steps (keyed by event_index).
export function stepOptions() {
	const cs = computedSteps();
	if (state.region === ALL) {
		const seen = new Map();
		for (const s of cs) {
			const t = stepTemplate(s);
			if (!seen.has(t)) seen.set(t, { value: t, label: t, count: 0, ord: s.event_index });
			seen.get(t).count++;
		}
		return [...seen.values()].sort((a, b) => a.ord - b.ord);
	}
	return cs.filter((s) => stepNode(s) === state.region)
		.sort((a, b) => a.event_index - b.event_index)
		.map((s) => ({ value: String(s.event_index), label: stepTemplate(s), count: 1, ord: s.event_index }));
}

// Resolve the nav selection into the concrete steps to load.
export function selectedSteps() {
	const cs = computedSteps();
	const { region, step } = state;
	if (region === ALL) {
		if (step === ALL) return cs;
		return cs.filter((s) => stepTemplate(s) === step); // step = a kind
	}
	const inRegion = cs.filter((s) => stepNode(s) === region);
	if (step === ALL) return inRegion;
	return inRegion.filter((s) => String(s.event_index) === String(step)); // step = event_index
}

const aggKey = (ev, view) => `${state.run}:${state.slot}:${state.model}:${ev}:${view}`;
async function getAnalysis(ev, view) {
	// A cached compact (which carries tokens) satisfies an agg request too.
	const ck = aggKey(ev, "compact");
	if (state.aggCache.has(ck)) return state.aggCache.get(ck);
	const key = aggKey(ev, view);
	if (state.aggCache.has(key)) return state.aggCache.get(key);
	const a = await api.attentionGet(state.run, state.slot, state.model, ev, { view });
	state.aggCache.set(key, a);
	return a;
}

// Turn a cell-aggregate response into `[{event_index, template, node, a}]`, zipping
// each batched analysis `a` onto its step (template/node from the nav steps), and
// warming `aggCache` so a later single-step read is a hit.
function rowsFromBatch(steps, resp, view) {
	const byEv = new Map((resp?.steps || []).map((st) => [st.event_index, st.a]));
	for (const st of resp?.steps || []) if (st.a) state.aggCache.set(aggKey(st.event_index, view), st.a);
	const rows = [];
	for (const s of steps) {
		const a = byEv.get(s.event_index);
		if (a) rows.push({ event_index: s.event_index, template: stepTemplate(s), node: stepNode(s), a });
	}
	rows.sort((x, y) => x.event_index - y.event_index);
	return rows;
}

// Per-step loader (bounded pool) — the FALLBACK when the batched `cell-aggregate`
// endpoint is unavailable (server predates it) or errors, so the view still fills
// instead of silently going blank. `token == null` disables the staleness guard.
async function loadRowsPooled(steps, token, view) {
	const rows = [];
	await pool(steps, 6, async (s) => {
		let a; try { a = await getAnalysis(s.event_index, view); } catch { return; }
		if ((token != null && token !== state.loadToken) || !a) return;
		rows.push({ event_index: s.event_index, template: stepTemplate(s), node: stepNode(s), a });
	});
	rows.sort((x, y) => x.event_index - y.event_index);
	return rows;
}

// Load the analyses for the current selection. A SINGLE-step selection pulls the
// `compact` view (per-token data for the output/tag segmentation); a MULTI-step
// selection is fetched in ONE batched `cell-aggregate` request (token-free `agg`)
// instead of one `attentionGet` per step — with a per-step fallback if that
// endpoint isn't there. Token-guarded so a stale selection never wins.
export async function loadRows(token) {
	const sel = selectedSteps();
	if (sel.length === 1) {
		const s = sel[0];
		let a; try { a = await getAnalysis(s.event_index, "compact"); } catch { return []; }
		if (token !== state.loadToken || !a) return [];
		return [{ event_index: s.event_index, template: stepTemplate(s), node: stepNode(s), a }];
	}
	try {
		const resp = await api.cellAggregate(state.run, state.slot, state.model, { view: "agg", evs: sel.map((s) => s.event_index) });
		if (token !== state.loadToken) return [];
		const rows = rowsFromBatch(sel, resp, "agg");
		if (rows.length) return rows; // else fall through to per-step (defensive)
	} catch (e) { console.warn("[tf-workspace] cell-aggregate unavailable — per-step fallback (restart the server for the fast path):", e?.message || e); }
	return loadRowsPooled(sel, token, "agg");
}

// Load EVERY computed step's bucket grids in ONE batched request — used by the VII
// report, which reads only `a.buckets`. The `buckets` view keeps the whole-cell
// payload small; falls back to per-step `agg` (which also carries buckets).
export async function loadAllRows() {
	const cs = computedSteps();
	if (!cs.length) return [];
	try {
		const resp = await api.cellAggregate(state.run, state.slot, state.model, { view: "buckets", evs: cs.map((s) => s.event_index) });
		const rows = rowsFromBatch(cs, resp, "buckets");
		if (rows.length) return rows;
	} catch (e) { console.warn("[tf-workspace] cell-aggregate(buckets) unavailable — per-step fallback:", e?.message || e); }
	return loadRowsPooled(cs, null, "agg");
}

// --- aggregation helpers -----------------------------------------------------

function pm(vals) {
	const n = vals.length || 1;
	const m = vals.reduce((a, b) => a + b, 0) / n;
	const v = vals.reduce((a, b) => a + (b - m) * (b - m), 0) / n;
	return { m, s: Math.sqrt(v) };
}

// Mean per-attribute attention across the selection's steps (absent = 0), for the
// spider. Returns [{ component, score, sd }] sorted desc.
export function poolComponents(rows) {
	const n = rows.length || 1;
	const map = new Map();
	for (const r of rows) for (const c of ((r.a.agg || {}).scene || {}).componentTotals || []) {
		if (!map.has(c.component)) map.set(c.component, []);
		map.get(c.component).push(c.score);
	}
	return [...map.entries()].map(([component, vals]) => {
		const p = pm(vals.concat(Array(Math.max(0, n - vals.length)).fill(0)));
		return { component, score: p.m, sd: p.s };
	}).sort((a, b) => b.score - a.score);
}

// Total attention split across zone / object / frame across the selection, for
// the composition pie. `region` selects which generation region the attention is
// measured over: "reasoning" | "output" | "scene" (both). Summed.
export function poolKindTotals(rows, region = "scene") {
	const kt = { zone: 0, object: 0, frame: 0 };
	for (const r of rows) for (const e of ((r.a.agg || {})[region] || {}).entityTotals || []) {
		kt[entityKindLabel(e.kind, e.id)] += e.score;
	}
	return kt;
}

// Context-order points for the subsidiary scatter: one dot per scene entity, x =
// its order in the scene context (0 = first … 1 = last), y = attention to it
// (per-step normalized so steps overlay), colored by kind. Pooled over the rows.
// `region` = "reasoning" | "output" | "scene" (both).
export function contextPoints(rows, region = "scene") {
	const pts = [];
	for (const r of rows) {
		const ents = (r.a.scene_entities || []).filter((e) => e.token_span && e.token_span.length);
		if (ents.length < 2) continue;
		const totals = new Map((((r.a.agg || {})[region] || {}).entityTotals || []).map((e) => [e.id, e.score]));
		const starts = ents.map((e) => e.token_span[0]);
		const lo = Math.min(...starts), span = Math.max(...starts) - lo || 1;
		const scores = ents.map((e) => totals.get(e.id) || 0);
		const maxScore = Math.max(1e-9, ...scores);
		const sumScore = scores.reduce((a, b) => a + b, 0) || 1;
		for (const e of ents) {
			const attn = totals.get(e.id) || 0;
			pts.push({
				x: (e.token_span[0] - lo) / span,
				y: attn / maxScore,            // plotted (per-step normalized so steps overlay)
				attn,                          // raw attention
				share: attn / sumScore,        // fraction of the step's entity attention (for the focus filter)
				kind: entityKindLabel(e.kind, e.id),
				id: e.id,
			});
		}
	}
	return pts;
}

// Per-output-item rollups across the selection, for the output card. Each item is
// one emitted object/region; its `attrs` are the per-attribute attention it drew
// (the sub-bins). Kept in emission order, concatenated across steps.
export function poolOutputs(rows) {
	const items = [];
	for (const r of rows) {
		for (const o of (r.a.agg || {}).outputs || []) {
			const attrs = ((o.scene || {}).componentTotals || [])
				.map((c) => ({ component: c.component, score: c.score }))
				.filter((c) => c.score > 0)
				.sort((a, b) => b.score - a.score);
			const total = attrs.reduce((s, c) => s + c.score, 0);
			const entities = ((o.scene || {}).entityTotals || [])
				.map((e) => ({ id: e.id, kind: e.kind, score: e.score }))
				.sort((a, b) => b.score - a.score).slice(0, 8);
			items.push({ id: o.entity, label: o.entity, step: r.template, ev: r.event_index, n: o.n || 0, attrs, total, entities });
		}
	}
	return items;
}

// First OUTPUT token index in a compact analysis (tokens before it are reasoning).
export function outStartOf(a) {
	const t = a && a.tokens;
	if (a && a.out_start != null) return a.out_start;
	if (!t) return 0;
	const i = t.findIndex((x) => x && x.output_entity != null);
	return i < 0 ? t.length : i;
}

// Segment a single step's OUTPUT token stream into emitted items and, within each,
// the JSON attribute fields being written (mirrors the legacy segment-output).
// Needs a.tokens (compact view). Returns { items:[{i0,label}], fields:[{i,label}], outStart } or null.
export function outputSegments(a) {
	const toks = (a && a.tokens) || [];
	if (!toks.length) return null;
	const outStart = outStartOf(a);
	const items = [];
	for (let i = outStart; i < toks.length; i++) {
		const e = toks[i].output_entity;
		if (e == null) continue;
		if (!items.length || items[items.length - 1].label !== e) items.push({ i0: i, label: e });
	}
	// Reconstruct the emitted text + a char→token map, then find every `"field":` key.
	let text = ""; const offTok = [];
	for (let i = outStart; i < toks.length; i++) { const t = toks[i].text || ""; for (let c = 0; c < t.length; c++) offTok.push(i); text += t; }
	const fields = []; const re = /"([A-Za-z_][\w]*)"\s*:/g; let m;
	while ((m = re.exec(text))) { const tok = offTok[m.index]; if (tok != null) fields.push({ i: tok, label: m[1] }); }
	return { items, fields, outStart };
}

// Map a token index (into a.tokens) → x on the tag-breakdown progression axis,
// consistent with sectionProgression's xs: reasoning is compressed into
// [0, outFrac], output into [outFrac, 1], scaled by meanNq.
export function progXOfToken(a) {
	const b = (a && a.buckets) || {}; const toks = (a && a.tokens) || [];
	const N = toks.length || 1, os = Math.min(N, outStartOf(a));
	const meanNq = b.n_query || b.n_tokens || N || 1;
	const nb = b.n_buckets || ((b.region && b.region.length) || 1);
	const outFrac = nb ? ((b.out_bucket || 0) / nb) : 0;
	return (i) => {
		const frac = i <= os ? (os > 0 ? (i / os) * outFrac : 0) : outFrac + ((i - os) / Math.max(1, N - os)) * (1 - outFrac);
		return frac * meanNq;
	};
}

// --- organized <tag> progression (tag breakdown) ----------------------------

function validBuckets(a) {
	const b = a && a.buckets;
	return b && b.region_names && Array.isArray(b.region) ? b : null;
}
// The tag-breakdown shows each organized <tag> as ONE layer — VII sub-sentences
// (VERY_IMPORTANT_INSTRUCTIONS#NN) collapse back to the whole section here (the
// per-instruction split is the VII card's job, not this overview).
const _collapseTag = (t) => (t || "section").replace(/#\d+$/, "");
function sectionTags(rows) {
	const set = new Set();
	for (const r of rows) {
		const b = validBuckets(r.a);
		if (!b) continue;
		(b.region_meta || []).forEach((m) => { if (m && m.category === "text" && m.sub === "organized") set.add(_collapseTag(m.tag)); });
	}
	return [...set].sort();
}
function sectionRowsFor(b, tags, tagIdx) {
	const meta = b.region_meta || [], grid = b.region || [];
	return grid.map((row) => {
		const o = new Array(tags.length).fill(0);
		meta.forEach((m, c) => { if (m && m.category === "text" && m.sub === "organized") { const k = tagIdx.get(_collapseTag(m.tag)); if (k != null) o[k] += row[c]; } });
		return o;
	});
}
function resampleGrid(grid, G) {
	const B = grid.length;
	if (!B) return [];
	return Array.from({ length: G }, (_, g) => {
		const x = (G === 1 ? 0 : g / (G - 1)) * (B - 1);
		const lo = Math.floor(x), hi = Math.min(B - 1, lo + 1), t = x - lo;
		return grid[lo].map((v, k) => v * (1 - t) + grid[hi][k] * t);
	});
}
// Aggregate organized-<tag> mass over the selection on a shared progression grid.
// Returns { grid:[G][tags], tags, G, meanNq, n } or null.
export function sectionProgression(rows) {
	const G = 32, tags = sectionTags(rows);
	if (!tags.length) return null;
	const tagIdx = new Map(tags.map((t, i) => [t, i]));
	const acc = []; let n = 0, nqSum = 0;
	for (const r of rows) {
		const b = validBuckets(r.a);
		if (!b || !(b.region || []).length) continue;
		const rg = resampleGrid(sectionRowsFor(b, tags, tagIdx), G);
		if (!acc.length) rg.forEach((row) => acc.push(row.slice()));
		else rg.forEach((row, i) => row.forEach((v, k) => { acc[i][k] += v; }));
		n++; nqSum += (b.n_query || b.n_tokens || G);
	}
	if (!n) return null;
	return { grid: acc.map((row) => row.map((v) => v / n)), tags, G, meanNq: nqSum / n || 1, n };
}

// --- per-instruction VERY_IMPORTANT_INSTRUCTIONS attention (attention-saving) --
// The VII prompt section is split (server-side) into ordered leaves
// prompt.VERY_IMPORTANT_INSTRUCTIONS#NN, each carrying a `snippet` of its text. We
// COMPILE those leaves into whole SENTENCES on the client: walk them in reading
// (#NN) order and accumulate snippet + attention + tokens until the text closes a
// sentence (. ! ?), then emit. So a full instruction is ONE segment regardless of
// how finely the server chunked it — older results were split per clause; this
// rejoins them (and sums their attention, which is why a sentence's share is
// larger than any lone clause's). Each sentence's mean attention share (mean over
// the progression buckets, then mean over the steps that contain it) is rolled up
// by its TEXT, so the same instruction merges across steps/kinds while distinct
// instructions stay separate (the VII wording differs per step kind). Returns
// [{ key, label, score, err, tokens, n, perTok }] sorted by attention desc.
// `err` is the standard error of the mean share across the generation buckets
// (pooled over steps) — how (un)certain / spiky the attention estimate is, used
// for the scatter's y error bars. `perTok` (share ÷ tokens) surfaces
// long-but-ignored instructions — the trim candidates.
const _isViiMeta = (m) => m && m.category === "text" && m.sub === "organized"
	&& typeof m.tag === "string" && m.tag.startsWith("VERY_IMPORTANT_INSTRUCTIONS");
const _viiOrd = (tag) => { const m = /#(\d+)$/.exec(tag || ""); return m ? +m[1] : 0; };
const _endsSentence = (s) => /[.!?]["')\]]?\s*$/.test((s || "").trim());
export function viiInstructions(rows) {
	const acc = new Map(); // sentence text -> { key, label, tokens, samples:[per-bucket shares], nSteps, kinds:Map }
	for (const r of rows) {
		const kind = r.template || "?"; // the step kind whose VII wording carries this instruction
		const b = validBuckets(r.a);
		if (!b) continue;
		const meta = b.region_meta || [], grid = b.region || [], toks = b.region_tokens || [];
		const nb = grid.length || 1;
		// this step's VII leaves in reading order (#NN), then compile into sentences
		const leaves = [];
		meta.forEach((m, c) => { if (_isViiMeta(m)) leaves.push({ c, ord: _viiOrd(m.tag), snip: m.snippet || m.tag, tok: toks[c] || 0 }); });
		leaves.sort((a, b) => a.ord - b.ord);
		let txt = "", tk = 0, has = false, perBucket = new Array(nb).fill(0);
		const flush = () => {
			const key = txt.trim();
			if (key && has) {
				const rec = acc.get(key) || { key, label: key, tokens: 0, samples: [], nSteps: 0, kinds: new Map() };
				for (const v of perBucket) rec.samples.push(v); // pool this step's per-bucket sentence shares
				rec.tokens = Math.max(rec.tokens, tk); rec.nSteps += 1;
				rec.kinds.set(kind, (rec.kinds.get(kind) || 0) + 1); // tally which step kind(s) carry this instruction
				acc.set(key, rec);
			}
			txt = ""; tk = 0; has = false; perBucket = new Array(nb).fill(0);
		};
		for (const lf of leaves) {
			for (let bi = 0; bi < nb; bi++) perBucket[bi] += (grid[bi] && grid[bi][lf.c]) || 0; // per-bucket sentence share
			tk += lf.tok; txt += (txt ? " " : "") + lf.snip; has = true;
			if (_endsSentence(lf.snip)) flush();
		}
		flush(); // trailing text with no terminal punctuation
	}
	return [...acc.values()].map((rec) => {
		const N = rec.samples.length || 1;
		const score = rec.samples.reduce((a, v) => a + v, 0) / N; // mean attention share over generation
		const variance = rec.samples.reduce((a, v) => a + (v - score) ** 2, 0) / N;
		const err = Math.sqrt(variance / N); // standard error of the mean (uncertainty of the estimate)
		// dominant step kind = the one that carried this instruction in the most steps (colors its dot + rank row)
		let kind = "?", best = -1;
		for (const [k, c] of rec.kinds) if (c > best) { best = c; kind = k; }
		return { key: rec.key, label: rec.label, score, err, tokens: rec.tokens, n: rec.nSteps, perTok: rec.tokens ? score / rec.tokens : score, kind, kinds: [...rec.kinds.keys()] };
	}).sort((a, b) => b.score - a.score);
}

// Shared model behind the VII scatter (card + report): normalized points, a
// least-squares fit of attention vs length, and a per-instruction "z" = how many
// standard errors BELOW the trend line it sits (the score the user ranks by —
// distance below ÷ sigma). Points are ranked by z desc (most below + most
// certain first); `marked` is the top slice (the trim picks / green rings).
// Returns null when there are no split instructions. `max` = the top instruction's
// mean share (the y-normalizer), so callers can recover raw shares.
const _SIG_FLOOR = 0.01; // sigma floor (normalized) so near-zero-error points don't blow up z
export function viiScatterModel(rows) {
	const items = viiInstructions(rows).filter((it) => it.key !== "VERY_IMPORTANT_INSTRUCTIONS");
	if (!items.length) return null;
	const max = Math.max(1e-9, ...items.map((it) => it.score));
	const P = items.map((it) => ({ it, key: it.key, label: it.label, x: it.tokens, y: it.score / max, ey: it.err / max, share: it.score, tokens: it.tokens, kind: it.kind, kinds: it.kinds }));
	const n = P.length;
	let sx = 0, sy = 0, sxx = 0, sxy = 0;
	for (const p of P) { sx += p.x; sy += p.y; sxx += p.x * p.x; sxy += p.x * p.y; }
	const denom = n * sxx - sx * sx;
	const slope = denom ? (n * sxy - sx * sy) / denom : 0;
	const intercept = (sy - slope * sx) / (n || 1);
	const fit = (x) => intercept + slope * x;
	P.forEach((p) => { p.resid = p.y - fit(p.x); p.z = -p.resid / Math.max(p.ey, _SIG_FLOOR); }); // z = sigmas below the trend
	const ranked = [...P].sort((a, b) => b.z - a.z); // most below + most certain first
	const K = Math.max(1, Math.min(6, Math.round(n * 0.15)));
	const marked = new Set(ranked.filter((p) => p.z > 0).slice(0, K).map((p) => p.key));
	const xMaxTok = Math.max(...P.map((p) => p.x), 1);
	return { items, P, max, slope, intercept, fit, ranked, marked, xMaxTok, n };
}
