// Shared state + constants for the (new) /tf attention inspector. Deliberately
// small: the mutable `state`, DOM helper `$`, the color/label constants, and the
// scene-tree "frame" recovery helpers. Everything else (loading, aggregation,
// rendering) lives in focused modules that import from here.

import { createObsModel, emittedStep } from "../../js/events.js";

export const $ = (id) => document.getElementById(id);

// Sentinel for the "ALL regions" / "ALL steps" nav options.
export const ALL = "__ALL__";

// Entity-kind colors, matched to the 3D viewer + legacy inspector.
export const COLORS = {
	zone: "#ff6b6b",
	object: "#6bd96e",
	frame: "#7fb3d5", // encapsulating shell (walls / floor / ground)
	to_place: "#e0a94a",
	output: "#b46aff",
	variable: "#4af0e0",
};

// Per-attribute (component) colors + short chip labels — the spider axes + the
// output card's property sub-bins share these so a given attribute reads the
// same hue everywhere.
export const COMPONENT_COLORS = {
	name: "#8ab4ff", prompt: "#6bd96e", noun_phrase: "#9ece6a", description: "#7fd6c2",
	placement: "#e0a94a", relationships: "#ff6b9d", dimensions: "#b46aff",
	global_origin: "#c78bff", local_origin: "#9d7bd9", orientation: "#4af0e0",
	yaw: "#39c5cf", proxy_shape: "#c8a06a", parent: "#ff5f5f", parent_region: "#ff9e64",
};
export const COMPONENT_ABBR = {
	name: "id", prompt: "prm", noun_phrase: "nn", description: "dsc",
	placement: "plc", relationships: "rel", dimensions: "dim",
	global_origin: "org", local_origin: "loc", orientation: "ori",
	yaw: "yaw", proxy_shape: "pxy", parent: "par", parent_region: "reg",
};
export const compHex = (c) => COMPONENT_COLORS[c] ?? "#888";

// Canonical attribute axis order for the spider (fixed so a given attribute sits
// at the same clock position every render).
export const ATTR_AXIS_ORDER = [
	"name", "noun_phrase", "prompt", "description",
	"placement", "relationships", "dimensions", "orientation", "yaw",
	"proxy_shape", "parent", "parent_region", "global_origin", "local_origin",
];

// Build an obs model over the WHOLE cell history — a stable id → node lookup used
// only to recover each entity's true kind (the export/attention maps only carry
// "zone"/"object", so encapsulating shells must be recovered as "frame").
// LEGACY: needs the full events.jsonl. Superseded by obsFromTree (below), which
// reads the server's compact projection instead of folding ~100MB client-side.
export function buildCellObs(events) {
	const m = createObsModel();
	for (const e of events) m.feed(e);
	return m.model;
}

// The obs model built from the server's compact scene-tree projection (api.tfTree)
// — a stable id → node lookup (kind, structural parent) plus each node's EMITTING
// provenance (region + pass + call index). Replaces downloading + folding the
// whole (100+ MB) events.jsonl just to recover id→kind/parent/region. The shape
// mirrors createObsModel().model so isFrameEntity / emittedStep / emittingRegion
// read it unchanged (kept in sync with /tf's obsFromTree).
export function obsFromTree(tree) {
	const order = Array.isArray(tree?.order) ? tree.order : [];
	const src = (tree && tree.nodes) || {};
	const nodes = new Map();
	const provenance = new Map();
	for (const id of order) {
		const n = src[id] || {};
		nodes.set(id, { id, parentId: n.parent_id ?? null, kind: n.kind ?? "zone" });
		if (n.emitted_by != null || n.region != null) {
			provenance.set(id, [{
				relation: "emitted_by",
				call: { node: n.region ?? null, index: n.call_index ?? null, template: n.emitted_by ?? null, step: n.emitted_by ?? null },
			}]);
		}
	}
	return { nodes, order, provenance, calls: [], log: [], specs: new Map(), errorCount: 0, maxIndex: -1 };
}

// An encapsulating shell ("frame"): a node emitted by the encapsulating_decompose
// pass. The attention maps label it a generic "object"; the 3D viewer colors it
// light blue, so we recover the distinction from the obs model.
export function isFrameEntity(id) {
	const m = state.obs;
	if (!m) return false;
	return m.nodes.get(id)?.kind === "frame" || emittedStep(m, id) === "encapsulating_decompose";
}

// Display kind for a scene entity: zone / frame / object.
export function entityKindLabel(kind, id) {
	if (kind === "zone") return "zone";
	return isFrameEntity(id) ? "frame" : "object";
}

// Swatch color for a scene entity (matches the 3D viewer).
export function entityHex(kind, id) {
	if (kind === "zone") return COLORS.zone;
	if (isFrameEntity(id)) return COLORS.frame;
	return COLORS.object;
}

export const state = {
	// --- nav selection ---
	run: null,
	slot: null,
	model: null,
	region: ALL,   // ALL | <node id>
	step: ALL,     // ALL | <event_index> (specific region) | <template> (ALL regions)
	view: "data",  // data | content | ablation

	// --- cell data ---
	runs: [],
	slots: [],     // [{ id, runs: { model: { events_count } } }]
	models: [],
	steps: [],     // tf-steps (each: event_index, template/step, node, has_scene, render_until)
	events: null,  // LAZY full cache.llm history (per-step prompt/content/VII only) — pulled by ensureEvents(), not on cell load
	eventsKey: null, // `${run}/${slot}/${model}` the loaded `events` belong to (staleness guard)
	obs: null,     // obs model (id → node kind/parent + provenance) from api.tfTree — frame recovery, no events.jsonl fold
	attnStatus: {},// event_index -> "ready" | "stale" | "none" | ...

	// --- ablation view ---
	ablRun: null,  // selected RUN (experiment label; "" = the no-label runs) — drives which experiment + graphs are shown
	ablOrderLogY: (() => { try { return localStorage.getItem("tf-abl-order-logy") === "1"; } catch { return false; } })(), // scene-ordering scatter: log-scale the attention axis
	ablFocus: (() => { try { return Number(localStorage.getItem("tf-abl-focus")) || 0; } catch { return 0; } })(), // scene-ordering: min attention share filter
	variants: [],  // discovered ablation variants for the current cell
	variantRows: new Map(), // variant name -> loaded row { name, kind, cut, label, coord, method, xml, a } | null

	// --- per-step analysis cache (agg view: compact minus tokens) ---
	aggCache: new Map(), // `${run}:${slot}:${model}:${ev}` -> agg analysis

	// --- render state ---
	rows: [],      // currently-loaded rows for the selection: [{ event_index, template, node, a }]
	loadToken: 0,  // bumped on every selection change; async loads apply only if still current
	pieFilter: null, // null | "zone" | "object" | "frame" — composition pie slice filter
	compRegion: (() => { try { return localStorage.getItem("tf-comp-region") || "reasoning"; } catch { return "reasoning"; } })(), // reasoning | output | scene(both) — which generation region the composition is measured over
	scatterLogY: (() => { try { return localStorage.getItem("tf-scatter-logy") === "1"; } catch { return false; } })(), // log-scale the context-order scatter's attention axis
	compFocus: (() => { try { return Number(localStorage.getItem("tf-comp-focus")) || 0; } catch { return 0; } })(), // context-order scatter: min attention-SHARE filter (focus)
	lastN: (() => { try { return Number(localStorage.getItem("tf-lastn")) || 0; } catch { return 0; } })(), // data-view window: keep only the last N steps of the selection (0 = all)

	// --- content view (per-step "structure" page: 3D · tree · plan/output/reasoning) ---
	viewer3d: null,            // the content view's three.js viewer (created once, reused)
	applyBaseHighlight: null,  // re-apply the 3D view's default attention shading (set by content.js)
};

export const bumpLoad = () => ++state.loadToken;

// --- content-view cross-highlight registry (tree badge <-> plan/output phrase <-> 3D) ---
// Rebuilt each content render. Maps an entity id to the DOM nodes representing it
// this render; hovering any one glows them all and lights the entity in 3D.
export const ctHover = new Map();
export function ctHoverReset() { ctHover.clear(); }
export function ctHoverRegister(id, node) {
	if (!id || !node) return;
	let e = ctHover.get(id);
	if (!e) { e = []; ctHover.set(id, e); }
	e.push(node);
	node.addEventListener("mouseenter", () => ctSetHover(id, true));
	node.addEventListener("mouseleave", () => ctSetHover(id, false));
}
function ctSetHover(id, on) {
	const nodes = ctHover.get(id);
	if (nodes) for (const n of nodes) n.classList.toggle("hot", on);
	// Add a strong GREEN box on the hovered entity WITHOUT clearing the attention
	// overlay, so every other entity's attention stays visible.
	if (on) state.viewer3d?.setHoverHighlight?.(id);
	else state.viewer3d?.clearHoverHighlight?.();
}
