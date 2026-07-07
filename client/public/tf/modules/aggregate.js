// Aggregation helpers that normalize a step's PRECOMPUTED attention payload into
// the shapes the summary/overview/placement blocks consume, plus the cell-wide
// p/m rollup. Imports only leaves (state for kind labeling, uncertainty for pm).

import { entityKindLabel } from "./state.js";
import { pm } from "./uncertainty.js";

// First query-token index that emits output (has an output_entity); tokens
// before it are reasoning, at/after are the output trace. tokens.length if the
// step emits no mapped output (all reasoning).
export function outputStartTok(a) {
	const i = (a.tokens || []).findIndex((t) => t.output_entity != null);
	return i < 0 ? (a.tokens?.length ?? 0) : i;
}

// Bucket a small entityTotals list into zone/object/frame using the client's
// kind labeling (which recovers "frame" from the obs model — a distinction the
// server-side aggregate can't make).
function kindTotalsOf(entityTotals) {
	const kt = { zone: 0, object: 0, frame: 0 };
	for (const e of entityTotals || []) kt[entityKindLabel(e.kind, e.id)] += e.score;
	return kt;
}

// Normalize a step's PRECOMPUTED scene aggregate (from the compact payload) into
// the shape the summary/overview blocks consume. Replaces the old per-token walk
// — the heavy per-token entity lists no longer cross the wire.
export function aggregateAttn(a) {
	const agg = (a && a.agg) || {};
	const scene = agg.scene || { entityTotals: [], componentTotals: [] };
	const entityTotals = scene.entityTotals || [];
	return {
		entityTotals,
		componentTotals: scene.componentTotals || [],
		kindTotals: kindTotalsOf(entityTotals),
		mass: agg.mass || [],
		entropy: agg.entropy || [],
	};
}

// Does this step carry a to-place (bbox-batch) readout? (Precomputed in agg.)
export function hasToPlace(a) { return !!(a && a.agg && a.agg.to_place); }

// Cell-wide entity + attribute attention as a p/m signal: the MEAN score across
// every computed step (steps where an item is absent count as 0) with the sample
// spread (sd) across steps. `score` is the mean; `sd` is the ± the graphs draw.
export function overviewAggregate(rows) {
	const n = rows.length || 1;
	const entity = new Map(), comp = new Map();
	for (const r of rows) {
		for (const e of r.agg.entityTotals) { const cur = entity.get(e.id) || { id: e.id, kind: e.kind, vals: [] }; cur.vals.push(e.score); entity.set(e.id, cur); }
		for (const c of r.agg.componentTotals) { const cur = comp.get(c.component) || { component: c.component, vals: [] }; cur.vals.push(c.score); comp.set(c.component, cur); }
	}
	const fold = (vals) => pm(vals.concat(Array(Math.max(0, n - vals.length)).fill(0)));
	return {
		entityTotals: [...entity.values()]
			.map((e) => { const p = fold(e.vals); return { id: e.id, kind: e.kind, score: p.m, sd: p.s }; })
			.sort((a, b) => b.score - a.score),
		componentTotals: [...comp.values()]
			.map((c) => { const p = fold(c.vals); return { component: c.component, score: p.m, sd: p.s }; })
			.sort((a, b) => b.score - a.score),
	};
}
