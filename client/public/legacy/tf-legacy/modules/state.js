// Shared state + constants for the /tf inspector. This is the leaf every other
// module imports directly (mirrors the dashboard's js/state.js): the single
// mutable `state` object, the DOM helper `$`, color/label constants, the
// client-side LRU caches, and the step-kind helpers. Imports only leaves
// (events.js) so there's no init-order hazard.

import { createObsModel, emittedStep } from "../../js/events.js";

export const $ = (id) => document.getElementById(id);

export const SEQ_SPEED = 8; // end-to-end runs play faster than the single-step default

// Object-placement steps: while these play we hide the encapsulating FRAMES
// (walls/roof/shell) of the current node's ANCESTOR zones — keeping the root's
// shell/ground — so you can see the objects going into the zone instead of
// staring at the outer walls. (`anchor_decompose`/`next_object` invent the
// objects; `object_bbox_batch` places them.)
export const PLACEMENT_STEPS = new Set(["anchor_decompose", "next_object", "object_bbox_batch"]);

export const COLORS = {
	zone: "#ff6b6b",
	object: "#6bd96e",
	frame: "#7fb3d5", // encapsulating shell (floor/roof slab, walls, ground) — matches the 3D viewer
	to_place: "#e0a94a",
	output: "#b46aff",
	variable: "#4af0e0",
};

// Per-component colors + short chip labels for the component-level highlight.
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
export const compHex = (comp) => COMPONENT_COLORS[comp] ?? "#888";

// Full obs model over the WHOLE cell history — a stable id → node lookup used
// only to recover each entity's true kind for swatch coloring; the per-step
// scene-tree step selector (renderPipeline) reads it for the node hierarchy.
export function buildCellObs(events) {
	const m = createObsModel();
	for (const e of events) m.feed(e);
	return m.model;
}

// An encapsulating shell ("frame": floor/roof slabs, walls, ground). The
// tf-export + attention maps only carry kind "zone"/"object" (derived from the
// scene-context text), so the frame distinction is recovered from the obs model
// — a node whose kind is "frame", i.e. emitted by the encapsulating_decompose
// pass. Matches how the 3D viewer colors these boxes light blue.
export function isFrameEntity(id) {
	const m = state.obs;
	if (!m) return false;
	return m.nodes.get(id)?.kind === "frame" || emittedStep(m, id) === "encapsulating_decompose";
}

// Swatch color for a scene entity, matching the 3D viewer: zones red, frames
// (encapsulating shells) light blue, all other objects green.
export function entityHex(kind, id) {
	if (kind === "zone") return COLORS.zone;
	if (isFrameEntity(id)) return COLORS.frame;
	return COLORS.object;
}

// Display kind for a scene entity: an encapsulating shell reads as "frame"
// rather than the export's generic "object" (zones stay "zone").
export function entityKindLabel(kind, id) {
	if (kind === "zone") return "zone";
	return isFrameEntity(id) ? "frame" : "object";
}

export const state = {
	run: null,
	slot: null,
	model: null,
	slots: [],
	models: [],
	sceneAttentionCounts: new Map(), // slot id -> number of computed/stale attention analyses
	sceneRowsCache: new Map(), // slot id -> { rows, order, loading }
	runRowsCache: new Map(), // run name -> { rows, order, loading } (ablation cross-run compare)
	steps: [],
	stepIdx: 0,
	presentSeqId: 0, // bumped to cancel an in-flight end-to-end stitched present run
	events: [], // full event history for the cell
	obs: null, // full-history obs model — id→node lookup for entity kind (frames)
	export: null, // current tf-export payload
	highlight: "scene", // scene | to_place | output | variables | none
	component: "entity", // entity | all | <component name> — sub-filter for scene/to_place
	sort: "position", // position | id | kind | parent — map-list ordering
	seqExpanded: false, // clamp the reconstructed sequence until expanded
	pendingFocus: null, // {entity, mark} to scroll to after a re-render
	attn: null, // loaded COMPACT attention analysis for the current step (scalars + precomputed aggregates)
	attnHead: 0, // index into attn.selected_heads
	attnToken: 0, // index into attn.tokens
	// Lazy per-token detail (the top entities/attributes for the token-detail
	// table + 3D highlight). The compact payload omits it; we fetch just the
	// scrubbed token on demand so the browser never holds the whole (huge) result.
	tokenDetail: null, // { ev, i, heads } currently loaded, or null
	tokenDetailCache: new Map(), // `${ev}:${i}` -> heads[] (per-cell LRU)
	tokenDetailPending: new Set(), // `${ev}:${i}` fetches in flight (dedupe)
	// Cross-step LRU of the small compact payloads so back/forth is instant and a
	// step's result is fetched (and parsed) at most once per cell visit.
	compactCache: new Map(), // `${run}:${slot}:${model}:${ev}` -> compact analysis
	maxHeads: 32, // how many top (layer, head) pairs get per-token detail (compute param)
	attnModel: { open: true, hf_url: null, hf_path: null }, // is this cell's model open-weight? (from tf-steps)
	// Per-step (keyed by event_index) attention status — the single source of
	// truth for the queue + per-step indicators, mirrored from the server on
	// every poll. "ready" = a stored result exists; state.attn holds the one
	// currently displayed.
	attnStatus: {}, // event_index -> "none" | "queued" | "running" | "ready" | "error"
	attnErrors: {}, // event_index -> last failure message (from the server)
	// Steps we (re)requested whose displayed copy must be refetched once ready —
	// a force-recompute overwrites the file in place, so "ready" alone wouldn't
	// tell the UI to reload the (now stale) shown result.
	attnPendingReload: new Set(),
	// Frontend-driven compute-all window: rather than dumping every step onto
	// Modal, we keep only ATTN_WINDOW jobs outstanding and top up as they finish —
	// so Modal's queue stays small and simply drains/clears if we stop requeueing.
	// { evs:Set<remaining>, sent:Set<dispatched>, force:bool } or null when idle.
	attnPlan: null,
	attnServer: { queued: [], running: [], computed: [] }, // last server queue snapshot
	// Cross-step cache for the overview tab: event_index -> stored analysis, so
	// summarizing the whole cell fetches each step's result at most once.
	stepAnalyses: new Map(),
	expandedBlocks: new Set(), // map blocks the user un-clamped (keyed "map:<name>")
	// Monotonic token bumped on every cell/step change; async fetches capture it
	// and only apply their result if it still matches — prevents a slow response
	// from a previous step/cell painting the current view (event_index alone can
	// collide across cells).
	viewToken: 0,
	viewer: null,
	// The large analysis report is a fullscreen overlay. `reportView` is the
	// workspace mode; `drawerTab` is the in-drawer tab to restore highlight to when
	// the report closes.
	reportView: null, // "step" | "kind" | "scene" | null
	lastReportMode: "step",
	reportRestoreOpen: false, // set from saved state on load; boot reopens the overlay if true
	showErr: (() => { try { return localStorage.getItem("tf-show-err") !== "0"; } catch { return true; } })(), // ± error bars on graphs
	tokenOrderXMode: (() => { try { return localStorage.getItem("tf-token-x") || "n"; } catch { return "n"; } })(), // token-ordering x-axis basis: n | ratio | log
	segmentOutput: (() => { try { return localStorage.getItem("tf-seg-output") !== "0"; } catch { return true; } })(), // within-step breakdown graphs: overlay output item/attribute segment lines (on by default)
	outputZoom: (() => { try { return localStorage.getItem("tf-output-zoom") === "1"; } catch { return false; } })(), // within-step breakdown graphs: trim x-axis to just the output region (drop reasoning)
	segSel: null, // within-step: id of the clicked output item segment (breakdown drill-down); ephemeral, cleared on step change
	bucketsNormalize: (() => { try { return localStorage.getItem("tf-buckets-norm") === "1"; } catch { return false; } })(), // stacked-area graphs: normalize mass by scene token count
	// stacked-area graphs (kind/scene): how to lay out the X-axis —
	// "progression" (within-step token order) | "step" | "kind" | "zone". "" = the
	// level default (kind→step, scene→kind). Migrates the old by-step boolean.
	bucketsGroup: (() => { try { return localStorage.getItem("tf-buckets-group") || (localStorage.getItem("tf-buckets-bystep") === "1" ? "step" : ""); } catch { return ""; } })(),
	reportLastN: (() => {
		try {
			const v = Number(localStorage.getItem("tf-report-lastn") ?? localStorage.getItem("tf-tree-lastn"));
			return v > 0 ? v : 3;
		} catch { return 3; }
	})(), // workspace window: last N computed steps (all graphs)
	reportCategory: { step: "main", kind: "main", scene: "main" },
	// Scope-granular FOCUS (the single "primary" item you're inspecting), kept
	// separate from the compare set (state.pins). Step scope focuses via
	// state.stepIdx (it drives the 3D cursor); kind/scene scope carry their own
	// focus so browsing steps doesn't silently move the kind/scene view. A stale
	// value auto-heals in focusedKind()/focusedScene().
	focusKind: null, // kind scope: focused step-kind (null → derive from the current step)
	focusScene: null, // scene scope: focused scene slot (null → the loaded slot)
	compareTree: "kind", // "kind" | "zone" — left drawer grouping for compare selection
	openSelectGroups: new Set(),
	closedSelectGroups: new Set(),
	pins: { steps: [], entities: [], heads: [], kinds: [], scenes: [], runs: [] },
	graphModules: [], // ordered saved board modules; see report.defaultGraphModules()
	reportSelections: { stepIdx: 0, token: 0, head: 0 },
	drawerTab: "attention",
	// Cross-module 3D-highlight hooks, set by the owning modules so the shared
	// hover logic below (a state.js leaf) can repaint the default highlight WITHOUT
	// importing attnPanel/present (which would create import cycles).
	//   applyAttnHighlight   — attnPanel's token/head default highlight (set by overview.js)
	//   restoreAttnHighlight — when the 3D popup window is open it OWNS the viewer
	//                          shading; hover-out restores its aggregate via this.
	applyAttnHighlight: null,
	restoreAttnHighlight: null,
};

export const bumpView = () => ++state.viewToken;

// --- cross-highlight registry (attention tree badges <-> zone-plan phrases) ---
// Rebuilt every report render (reset in renderReportWorkspace before the hero and
// the tree are built). Maps an entity id to the DOM nodes that represent it this
// render — its tree badge(s) and its highlighted zone-plan phrase(s) — so hovering
// any one glows them all (and softly lights the entity in 3D). DOM refs are from
// the CURRENT render only, so the map is cleared each render to avoid stale nodes.
export const treeHover = new Map();
export function treeHoverReset() { treeHover.clear(); }
export function treeHoverRegister(id, node, isPhrase = false) {
	if (!id || !node) return;
	let e = treeHover.get(id);
	if (!e) { e = { badges: [], phrases: [] }; treeHover.set(id, e); }
	(isPhrase ? e.phrases : e.badges).push(node);
	node.addEventListener("mouseenter", () => setHoverEntity(id, true));
	node.addEventListener("mouseleave", () => setHoverEntity(id, false));
}
function setHoverEntity(id, on) {
	const e = treeHover.get(id);
	if (e) for (const n of [...e.badges, ...e.phrases]) n.classList.toggle("tree-hot", on);
	if (on) state.viewer?.setAttnHighlight?.([{ id, weight: 1 }]);
	else (state.restoreAttnHighlight || state.applyAttnHighlight)?.();
}

// Max attention jobs we keep outstanding on Modal at once. "Compute all" enqueues
// only this many and tops up as they finish (see startAttnPlan/attnPump), so the
// GPU queue never holds the whole cell and stops cleanly when we stop requeueing.
// Sized to keep the fanned-out consumer pool fed (up to 4 parallel consumers per
// model — H200, or 2×B200 each = 8×B200 total — at ~1 per 3 queued steps) without
// over-committing the shared per-model FIFO.
export const ATTN_WINDOW = 16;

// LRU caps for the client-side attention caches (bounded so long sessions don't
// grow unbounded — the whole point is to NOT hold everything in memory).
export const COMPACT_CACHE_MAX = 24;       // steps' compact payloads
export const TOKEN_DETAIL_CACHE_MAX = 256; // per-token detail records (per cell)

// Insert into a Map used as an LRU, evicting the oldest entries past `max`.
export function lruSet(map, key, val, max) {
	if (map.has(key)) map.delete(key);
	map.set(key, val);
	while (map.size > max) map.delete(map.keys().next().value);
	return val;
}

// Evict every cached copy of a step's analysis — used before a force recompute
// so the refreshed result is refetched rather than served from a stale cache.
export function dropStepCaches(ev) {
	state.compactCache.delete(ev);
	state.stepAnalyses.delete(ev);
	const pre = `${ev}:`;
	for (const k of [...state.tokenDetailCache.keys()]) if (k.startsWith(pre)) state.tokenDetailCache.delete(k);
	if (state.tokenDetail && state.tokenDetail.ev === ev) state.tokenDetail = null;
}

// Bbox-batch step templates that carry a to-place batch (kept in sync with the
// server's schema.BBOX_TEMPLATES). Used to target the to-place placement readout.
export const BBOX_STEP_TEMPLATES = new Set(["object_bbox_batch", "child_bbox_batch"]);

export const ATTN_STATUS_LABEL = { none: "not computed", queued: "queued", running: "computing…", ready: "computed", stale: "stale · recompute", error: "failed" };

export const TEMPLATE_PALETTE = ["#7aa2f7", "#6bd96e", "#e0a94a", "#b46aff", "#4af0e0", "#ff6b9d", "#c8a06a", "#9ece6a", "#ff9e64", "#39c5cf", "#ff5f5f", "#8ab4ff"];
export const templateColor = (t, order) => TEMPLATE_PALETTE[Math.max(0, order.indexOf(t)) % TEMPLATE_PALETTE.length];
export const groupByTemplate = (rows) => { const m = new Map(); for (const r of rows) { if (!m.has(r.template)) m.set(r.template, []); m.get(r.template).push(r); } return m; };

// An object_bbox_batch resolves the bboxes for whichever decompose / next pass
// ran IMMEDIATELY BEFORE it, so subcategorize it by that pass — the step it's
// "for" (bbox · anchor / next / encap / neg-space). Every other step kind groups
// by its plain template.
const _DECOMP_SUB = { anchor_decompose: "anchor", next_object: "next", encapsulating_decompose: "encap", negative_space_decompose: "neg-space" };
export function stepKind(step) {
	const t = step.template ?? step.step ?? "?";
	if (t === "object_bbox_batch") {
		const i = state.steps.indexOf(step);
		const prev = i > 0 ? state.steps[i - 1] : null;
		const sub = prev && _DECOMP_SUB[prev.template ?? prev.step];
		if (sub) return `object_bbox_batch · ${sub}`;
	}
	return t;
}

// The focused KIND at kind scope, decoupled from the 3D step cursor. When
// `available` (the kinds actually present) is given, a stale focus that no
// longer exists heals to the current step's kind, then to the first available
// kind — so the kind view never points at a kind that isn't there.
export function focusedKind(available = null) {
	const derived = state.steps[state.stepIdx] ? stepKind(state.steps[state.stepIdx]) : null;
	const ok = (k) => k != null && (!available || available.includes(k));
	if (ok(state.focusKind)) return state.focusKind;
	if (ok(derived)) return derived;
	return available && available.length ? available[0] : derived;
}

// The focused SCENE at scene scope — the loaded slot unless one is pinned as focus.
export function focusedScene() {
	return state.focusScene ?? state.slot;
}
