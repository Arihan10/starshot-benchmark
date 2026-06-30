// The slot overlay: enlarged 3D view of one cell (source run or its
// simulation branch) with the observability dock on the right.

import { api } from "./api.js";
import {
	state,
	emit,
	on,
	cellSummary,
	cellBranches,
	branchSummaryById,
} from "./state.js";
import { el, toast, stepUntilSelect, openModal } from "./ui.js";
import {
	openStream,
	dispatchSceneEvent,
	applySceneProjection,
	createObsModel,
	emittedStep,
} from "./events.js";
import { statusView } from "./status.js";
import { createObsDock } from "./obstree.js";
import { createTracePanel } from "./tracepanel.js";
import * as inquiry from "./inquiry.js";

const overlayEl = document.getElementById("overlay");
const titleEl = document.getElementById("overlay-title");
const crumbsEl = document.getElementById("overlay-crumbs");
const dotEl = document.getElementById("overlay-dot");
const statusEl = document.getElementById("overlay-status");
const actionBtn = document.getElementById("overlay-action");
const resetBtn = document.getElementById("overlay-reset");
const btnZoneLayers = document.getElementById("btn-zone-layers");
const zoneLayersLegendEl = document.getElementById("zone-layers-legend");

let viewer = null;
let dock = null;
let tracePanel = null;
let stream = null;
let obs = createObsModel();
let renderQueued = false;
let openSeq = 0; // monotonically increasing guard for async open races
let lastLayersSig = null; // skips rebuilding the layer legend when unchanged
// Asset view: which mesh build the 3D view shows + the generate gate's polling.
let assetMode = "library"; // "library" | "generated"
let optimizedView = true; // generated meshes: optimized KTX2/Meshopt twin vs raw
// Per-object override of the scene-wide optimized/raw toggle: node id ->
// "optimized" | "raw". Absent = follow `optimizedView`. Reset whenever the
// scene-wide toggle flips (it sets a fresh baseline for every object).
let objOptMode = new Map();
let genPollTimer = null;
let genMeshSig = null; // last generate-status mesh signature, to skip redundant work
let lastGenMesh = new Map(); // node id -> { url, v, optUrl, optV, unoptUrl, unoptV, ... } from the last poll
let loadedGen = new Map(); // node id -> the variant url currently loaded in the main viewer
let busyNodes = new Set(); // node ids currently queued/processing on a mesh backend
let assetBtn = null;
let genBtn = null;
let optBtn = null;
let genStatusEl = null;

// Cache-bust a generated mesh URL by its mtime token, so a regenerated asset
// (same id + path, new bytes) reloads instead of serving a stale cached GLB.
const withV = (url, v) => url + (url.includes("?") ? "&" : "?") + "v=" + v;

export function initOverlay(sceneViewer) {
	viewer = sceneViewer;
	// The right dock stays a pure node tree + log here — the left panel owns the
	// per-node emit lineage, so the dock doesn't also flip into trace mode.
	dock = createObsDock(document.getElementById("obsdock"), { trace: false });
	// The left emittance-trace inspector: opened by 3D selection, it shows the
	// picked node's root→node LLM-call lineage. Breadcrumb clicks re-aim the
	// selection (and camera); close clears the selection (which hides it).
	tracePanel = createTracePanel(document.getElementById("trace-panel"), {
		onNavigate: focusNode,
		onClose: () => viewer.clearSelection(),
		onInquire: openInquiryForCall,
		// Per-object generated-asset controls, live only while viewing a source
		// cell's generated build: regenerate on a chosen backend + symmetry ops +
		// prefab link/unlink.
		actions: {
			available: () =>
				!!state.view && !state.view.branch && assetMode === "generated",
			symmetryOf: (id) => {
				const m = lastGenMesh.get(id);
				return m ? { plane: m.plane, was: m.was } : null;
			},
			// The focused node's prefab group: its canonical, whether it's a reuse
			// (vs the group's source), and how many objects share the group. null
			// until the node has a generated mesh.
			prefabOf: (id) => {
				const m = lastGenMesh.get(id);
				if (!m || !m.canonical) return null;
				let groupSize = 0;
				for (const v of lastGenMesh.values())
					if (v.canonical === m.canonical) groupSize += 1;
				return {
					canonical: m.canonical,
					isReuse: m.canonical !== id,
					groupSize,
				};
			},
			// Other prefab groups this object could be linked into — one entry per
			// distinct canonical (excluding the node's own group), labeled by the
			// canonical's prompt and sized by membership.
			linkTargets: (id) => {
				const me = lastGenMesh.get(id);
				const myCanon = me ? me.canonical : null;
				const counts = new Map();
				for (const v of lastGenMesh.values()) {
					if (!v.canonical) continue;
					counts.set(v.canonical, (counts.get(v.canonical) || 0) + 1);
				}
				const out = [];
				for (const [canon, size] of counts) {
					if (canon === myCanon) continue;
					const node = obs.model.nodes.get(canon);
					out.push({
						canonical: canon,
						label: node?.prompt || canon,
						size,
					});
				}
				out.sort((a, b) => a.label.localeCompare(b.label));
				return out;
			},
			isBusy: (id) => busyNodes.has(id),
			imagePromptOf: (id) => lastGenMesh.get(id)?.imagePrompt ?? null,
			// This object's effective render variant ("optimized" | "raw"), or null
			// when it can't be toggled (only one variant on disk, e.g. mid-build).
			optimizedOf: (id) => {
				const m = lastGenMesh.get(id);
				if (!m || !m.optUrl || !m.unoptUrl) return null;
				return (
					objOptMode.get(id) ?? (optimizedView ? "optimized" : "raw")
				);
			},
			onRegenerate: regenerateNode,
			onSymmetrize: symmetrizeNode,
			onUnsymmetrize: unsymmetrizeNode,
			onLink: linkNode,
			onReorient: reorientNode,
			onGlassify: glassifyNode,
			onReset: resetNode,
			onSetOptimized: setObjectOptimized,
			// Delete is build-agnostic — it wipes the object from BOTH event logs
			// and every build's files — so unlike the generated-only actions above
			// it's offered on any SOURCE cell in either asset view, gated by
			// `deletable` rather than `available`.
			deletable: () => !!state.view && !state.view.branch,
			onDelete: deleteNode,
		},
		// Mini-preview mesh source: the generated mesh while in the generated view
		// (null — bbox only — when this node has none yet), the library asset otherwise.
		// This is the TRANSFORMED/optimized twin — used for the reference image and
		// the download link; the 3D preview itself shows the raw mesh below.
		meshUrlFor: (id, node) => {
			if (state.view && !state.view.branch && assetMode === "generated") {
				const m = lastGenMesh.get(id);
				return m && m.url ? withV(m.url, m.v) : null;
			}
			return node.meshUrl ?? null;
		},
		// The per-object 3D view shows the RAW mesh straight from the generation
		// API (Trellis/Hunyuan) rather than the transformed/optimized object: in
		// the generated view it's the raw-dir `<id>.raw.glb` (carried as `m.raw`);
		// in the library view it sits beside the placed mesh as `<id>.raw.glb`.
		rawMeshUrlFor: (id, node) => {
			if (state.view && !state.view.branch && assetMode === "generated") {
				const m = lastGenMesh.get(id);
				return m && m.raw ? withV(m.raw, m.v) : null;
			}
			return node.meshUrl
				? node.meshUrl.replace(/\.glb(\?|$)/, ".raw.glb$1")
				: null;
		},
	});

	// Tree ↔ 3D linking. A node-row click toggles selection and frames the
	// bbox (dimming the rest); a call click guarantees its node is focused
	// without toggling it off. 3D-side picks highlight + reveal the tree row,
	// and the hover tooltip reads seed/plan/image text from the obs model.
	viewer.setNodeInfo((id) => obs.model.nodes.get(id) ?? null);
	// Color objects in 3D by the decomposition step that emitted them (next_object
	// purple, anchor green, negative_space brown) — read from the same provenance
	// the tree shows as "via {step}". `recolorAll` (below) repaints once a load's
	// history has folded, since the scene projection paints bboxes first.
	viewer.setOriginOf((id) => emittedStep(obs.model, id));
	// A 3D pick (or any selection) reveals the dock row AND drives the left
	// emittance-trace panel; deselecting hides it. The panel only opens while a
	// scene is actually up (state.view set) — never on a stray selection event
	// with no cell loaded.
	viewer.onSelect((id) => {
		dock.markSelected(id, { scroll: true });
		if (id && state.view) tracePanel.show(obs.model, id);
		else tracePanel.hide();
	});
	dock.setOnNodeClick((id, { ensureSelected = false } = {}) => {
		if (!viewer.hasBbox(id)) return;
		if (ensureSelected && viewer.getSelected() === id) return;
		viewer.select(id, { frame: true });
	});
	// Per-node hiding: the tree's eye buttons and the canvas right-click share
	// the viewer's hidden set; any change re-renders the tree so eye states
	// and row dimming follow.
	dock.setHiddenApi({
		isHidden: viewer.isHidden,
		toggle: viewer.toggleHidden,
	});
	viewer.onHiddenChange(() => dock.renderTree(obs.model));

	// "why?" on any call row (dock OR the left emittance-trace panel) → the
	// decision-inquiry chat for that step. Read-only, so it's shared across
	// source AND branch views and reads the live view at click time.
	dock.setOnInquire(openInquiryForCall);

	document
		.getElementById("overlay-close")
		.addEventListener("click", closeOverlay);
	document.addEventListener("keydown", (ev) => {
		if (
			ev.key === "Escape" &&
			overlayEl.classList.contains("open") &&
			!document.getElementById("modal-root").firstChild
		) {
			// Progressive close: an open emittance-trace panel takes the first
			// Escape (deselect → panel hides); a second one leaves the scene.
			if (tracePanel.isOpen()) {
				viewer.clearSelection();
			} else {
				closeOverlay();
			}
			// When the full scene was opened from the compare view (stacked beneath),
			// don't let the SAME Escape also close compare — return to it instead.
			// The overlay's keydown is registered before compare's (see main init).
			ev.stopImmediatePropagation();
		}
	});
	for (const btn of document.querySelectorAll(
		"#viewer-toggles [data-toggle]",
	)) {
		const key = btn.dataset.toggle;
		const sync = () => btn.classList.toggle("off", !viewer.toggles[key]);
		btn.addEventListener("click", () => {
			viewer.toggles[key] = !viewer.toggles[key];
			viewer.refreshVisibility();
			sync();
		});
		sync();
	}
	// Zone-layers view: a dedicated mode (not a per-category toggle), so it gets
	// its own handler — flip the viewer's mode, then (re)build the depth legend.
	btnZoneLayers.addEventListener("click", () => {
		viewer.setZoneLayers(!viewer.getZoneLayers().enabled);
		refreshZoneLayers();
	});
	document
		.getElementById("btn-refit")
		.addEventListener("click", () => viewer.fit());
	document
		.getElementById("btn-unhide-all")
		.addEventListener("click", () => viewer.unhideAll());
	setupAssetControls();
	actionBtn.addEventListener("click", onAction);
	resetBtn.addEventListener("click", onReset);
	on("slots", () => {
		if (state.view) renderHeader();
	});
	on("open-cell", openCell);
	initObsResizer();
	refreshZoneLayers();
}

// The zone-layers legend: a swatch per nesting depth present in the scene, plus
// an "all" chip. Depth chips multi-select — toggle any combination (e.g. L0 +
// L1) to show just those layers; "all" (or clearing every chip) restores the
// full set. Active chips are highlighted. Shown only while the viewer's
// zone-layers view is on, and rebuilt only when the depth set / active layers
// actually change so streamed growth doesn't thrash the row.
function refreshZoneLayers() {
	if (!viewer) return;
	const info = viewer.getZoneLayers();
	btnZoneLayers.classList.toggle("off", !info.enabled);
	zoneLayersLegendEl.style.display = info.enabled ? "" : "none";
	const sig = `${info.enabled}|${info.active.join(",")}|${info.layers.map((l) => l.depth).join(",")}`;
	if (sig === lastLayersSig) return;
	lastLayersSig = sig;
	zoneLayersLegendEl.textContent = "";
	if (!info.enabled) return;
	zoneLayersLegendEl.appendChild(
		el("span", { class: "zl-title", text: "zone layers" }),
	);
	zoneLayersLegendEl.appendChild(
		el("button", {
			class: `zl-chip${info.active.length === 0 ? " on" : ""}`,
			text: "all",
			title: "show every zone layer",
			onclick: () => {
				viewer.clearZoneLayers();
				refreshZoneLayers();
			},
		}),
	);
	for (const layer of info.layers) {
		const hex = `#${layer.colorHex.toString(16).padStart(6, "0")}`;
		zoneLayersLegendEl.appendChild(
			el(
				"button",
				{
					class: `zl-chip${info.active.includes(layer.depth) ? " on" : ""}`,
					title: `toggle zone layer ${layer.depth} (multi-select)`,
					onclick: () => {
						viewer.toggleZoneLayer(layer.depth);
						refreshZoneLayers();
					},
				},
				el("span", { class: "zl-sw", style: `background:${hex}` }),
				`L${layer.depth}`,
			),
		);
	}
}

// ── from-scratch generated assets: view toggle + generate gate + per-object ──
// regenerate / symmetry. The generated build reuses the library scene's layout
// (same node ids + bboxes), so switching modes just re-attaches the OTHER mesh
// set onto the same boxes; the per-object actions live in the trace panel.

function setupAssetControls() {
	const refit = document.getElementById("btn-refit");
	if (!refit || assetBtn) return;
	assetBtn = el("button", {
		id: "btn-asset-mode",
		title: "toggle between the asset-library meshes and the from-scratch generated build",
		text: "generated",
		onclick: () =>
			setAssetMode(assetMode === "generated" ? "library" : "generated"),
	});
	genBtn = el("button", {
		id: "btn-generate",
		title: "build (or resume) this scene's from-scratch generated assets",
		text: "⚡ generate",
		onclick: onGenerate,
	});
	optBtn = el("button", {
		id: "btn-asset-optimized",
		title: "generated view: optimized (KTX2/Meshopt) served meshes vs the raw Trellis output",
		text: "optimized",
		onclick: toggleOptimized,
	});
	genStatusEl = el("span", { id: "gen-status", class: "gen-status" });
	refit.after(assetBtn, genBtn, optBtn, genStatusEl);
	syncAssetControls();
}

// The toggle shows only on source cells (branches always render library meshes);
// the generate button + status appear only while the generated view is active.
function syncAssetControls() {
	if (!assetBtn) return;
	const show = !!state.view && !state.view.branch;
	assetBtn.style.display = show ? "" : "none";
	const gen = show && assetMode === "generated";
	genBtn.style.display = gen ? "" : "none";
	optBtn.style.display = gen ? "" : "none";
	genStatusEl.style.display = gen ? "" : "none";
	assetBtn.classList.toggle("on", assetMode === "generated");
	assetBtn.textContent =
		assetMode === "generated" ? "generated ✓" : "generated";
	optBtn.classList.toggle("on", optimizedView);
	optBtn.textContent = optimizedView ? "optimized ✓" : "raw";
}

function setAssetMode(mode) {
	if (!state.view || state.view.branch || mode === assetMode) return;
	assetMode = mode;
	stopGenPoll();
	clearGeneratedState();
	// Drop the current build's meshes so the incoming view shows ONLY its own —
	// generated mode never layers over leftover library meshes (and vice versa).
	viewer.clearMeshes();
	syncAssetControls();
	tracePanel.rerenderInfo();
	if (assetMode === "generated") {
		pollGenerated(); // loads each built generated mesh incrementally
	} else {
		loadLibraryMeshes();
	}
}

function toggleOptimized() {
	if (!state.view || state.view.branch || assetMode !== "generated") return;
	optimizedView = !optimizedView;
	// The scene-wide flip is a fresh baseline — drop per-object overrides so every
	// object follows the new global mode.
	objOptMode = new Map();
	syncAssetControls();
	// Re-pull from the other source (optimized twin vs raw): drop what's loaded
	// and let the poll re-attach from the freshly-chosen dir.
	stopGenPoll();
	loadedGen = new Map();
	genMeshSig = null;
	viewer.clearMeshes();
	tracePanel.rerenderInfo();
	pollGenerated();
}

// Forget the generated-view bookkeeping (on cell open, mode switch, or toggle).
function clearGeneratedState() {
	lastGenMesh = new Map();
	loadedGen = new Map();
	busyNodes = new Set();
	objOptMode = new Map();
	genMeshSig = null;
	if (genStatusEl) genStatusEl.textContent = "";
}

// Library meshes load as one bundle (small, complete) onto the current scene.
function loadLibraryMeshes() {
	if (!state.view) return;
	const { slot, model, branch } = state.view;
	viewer.prefetchBundle(
		branch
			? api.branchMeshesUrl(branch)
			: api.meshesUrl(state.run, slot, model, {}),
	);
}

async function onGenerate() {
	if (!state.view || state.view.branch) return;
	const { slot, model } = state.view;
	try {
		await api.generate(state.run, slot, model);
		toast(`generating ${slot} · ${model}…`, "ok");
		pollGenerated();
	} catch (e) {
		toast(e.message, "err");
	}
}

function stopGenPoll() {
	if (genPollTimer) {
		clearTimeout(genPollTimer);
		genPollTimer = null;
	}
}

// Total objects/frames in the loaded scene — the denominator for the generate
// gate's X/Y progress (every concrete node gets one generated mesh).
function concreteNodeCount() {
	let n = 0;
	for (const node of obs.model.nodes.values()) {
		if (node.kind === "object" || node.kind === "frame") n += 1;
	}
	return n;
}

// The mesh variant to show for one object: its per-object override if set, else
// the scene-wide `optimizedView`. Falls back to whichever variant exists (then
// the status' global url) so a half-built object still renders.
function variantFor(m) {
	const mode = objOptMode.get(m.id) ?? (optimizedView ? "optimized" : "raw");
	if (mode === "optimized" && m.optUrl) return { url: m.optUrl, v: m.optV };
	if (mode === "raw" && m.unoptUrl) return { url: m.unoptUrl, v: m.unoptV };
	if (m.optUrl) return { url: m.optUrl, v: m.optV };
	if (m.unoptUrl) return { url: m.unoptUrl, v: m.unoptV };
	return { url: m.url, v: m.v };
}

// Attach each object's chosen variant to the main viewer, replacing in place when
// its loaded variant url changed (a regen, or a per-object/global raw↔optimized
// flip). `meshes` are status entries (or reconstructed `{ id, ...lastGenMesh }`).
function attachGeneratedMeshes(meshes) {
	for (const m of meshes) {
		const { url, v } = variantFor(m);
		if (!url) continue;
		const tagged = withV(url, v);
		if (loadedGen.get(m.id) === tagged) continue;
		const had = loadedGen.has(m.id);
		loadedGen.set(m.id, tagged);
		viewer.loadModel({ id: m.id, url: tagged }, api.absUrl(tagged), {
			replace: had,
		});
	}
}

// Poll the generate gate while a build/regen is in flight: refresh the symmetry
// map (the trace-panel hint), re-attach meshes when the finished-id set grows,
// and keep a small status label. Self-stops once nothing is running.
async function pollGenerated() {
	stopGenPoll();
	if (!state.view || state.view.branch || assetMode !== "generated") return;
	const { slot, model } = state.view;
	const run = state.run;
	let status = null;
	try {
		status = await api.generateStatus(run, slot, model, {
			optimized: optimizedView,
		});
	} catch {
		/* transient — the next poll (if still in generated mode) retries */
	}
	if (
		!state.view ||
		state.view.slot !== slot ||
		state.view.model !== model ||
		state.view.branch ||
		assetMode !== "generated"
	)
		return;
	if (!status) return;
	const meshes = status.meshes ?? [];
	lastGenMesh = new Map(
		meshes.map((m) => [
			m.id,
			{
				url: m.url,
				raw: m.raw,
				v: m.v,
				plane: m.sym,
				was: m.symWas,
				canonical: m.canonical ?? m.id,
				imagePrompt: m.imagePrompt ?? null,
				optUrl: m.optUrl,
				optV: m.optV,
				unoptUrl: m.unoptUrl,
				unoptV: m.unoptV,
			},
		]),
	);
	// Expand the server's busy set (directly queued/processing node ids) to
	// include every prefab group member whose canonical is busy — a regeneration
	// propagates to the whole group, so all members should be locked out.
	const rawBusy = new Set(status.busy ?? []);
	busyNodes = new Set(rawBusy);
	for (const [id, m] of lastGenMesh) {
		if (rawBusy.has(m.canonical)) busyNodes.add(id);
	}
	// Attach newly-built / regenerated / mode-changed meshes; loadedGen is keyed
	// by the loaded variant url, so a swap (per-object raw↔optimized, or a regen
	// bumping mtime) re-attaches and an already-current one is never re-downloaded.
	attachGeneratedMeshes(meshes);
	const busySig = [...busyNodes].sort().join(",");
	const sig =
		meshes.map((m) => `${m.id}:${m.v}:${m.sym}:${m.canonical}`).join("|") +
		"|busy:" +
		busySig;
	if (sig !== genMeshSig) {
		genMeshSig = sig;
		tracePanel.rerenderInfo();
	}
	const total = concreteNodeCount();
	const done = status.count ?? 0;
	const frac = total ? `${done}/${total}` : `${done}`;
	genStatusEl.textContent = status.running
		? `building… ${frac}`
		: `${frac} generated`;
	if (status.running) genPollTimer = setTimeout(pollGenerated, 1500);
}

// Per-object actions from the trace panel (generated build only). Each enqueues
// server-side work + polls so the new mesh swaps in when it lands. A plain
// regenerate propagates across the object's prefab group; an `unlink` regenerate
// first pulls the object out of its group so it diverges alone; `link` moves it
// into another object's group (re-deriving its mesh, no backend call).
async function regenerateNode(id, opts) {
	if (!state.view || state.view.branch) return;
	const { slot, model } = state.view;
	const unlink = !!opts.unlink;
	// Disable the asset's controls immediately — before the enqueue round-trip and
	// the (possibly slow) noun-phrase LLM step — so the button can't be re-fired.
	// The next /generate-status poll keeps it busy server-side (the node is marked
	// queued at enqueue); a failed enqueue re-enables it below.
	busyNodes.add(id);
	tracePanel.rerenderInfo();
	try {
		await api.regenerate(state.run, slot, model, id, {
			...opts,
			unlink,
			propagate: !unlink,
		});
		toast(
			`${unlink ? "unlinking + regenerating" : "regenerating"} ${id}${opts.regenNounPhrase ? " (+ new noun phrase)" : ""} · ${opts.backend}…`,
			"ok",
		);
		pollGenerated();
	} catch (e) {
		busyNodes.delete(id);
		tracePanel.rerenderInfo();
		toast(e.message, "err");
	}
}

async function linkNode(id, target, { group = false } = {}) {
	if (!state.view || state.view.branch) return;
	const { slot, model } = state.view;
	try {
		await api.link(state.run, slot, model, id, target, { group });
		toast(`linking ${group ? "group" : id} → ${target}…`, "ok");
		pollGenerated();
	} catch (e) {
		toast(e.message, "err");
	}
}

async function symmetrizeNode(id, opts) {
	if (!state.view || state.view.branch) return;
	const { slot, model } = state.view;
	try {
		await api.symmetrize(state.run, slot, model, id, {
			...opts,
			propagate: true,
		});
		toast(`symmetrizing ${id}…`, "ok");
		pollGenerated();
	} catch (e) {
		toast(e.message, "err");
	}
}

async function unsymmetrizeNode(id) {
	if (!state.view || state.view.branch) return;
	const { slot, model } = state.view;
	try {
		await api.unsymmetrize(state.run, slot, model, id, { propagate: true });
		toast(`un-symmetrizing ${id}…`, "ok");
		pollGenerated();
	} catch (e) {
		toast(e.message, "err");
	}
}

// Change the object's "front view" — rotate its raw mesh 90° about an axis so a
// different face points +Z. Propagates across the prefab group (server bakes it
// into the canonical's raw + re-derives every reuse).
async function reorientNode(id, opts) {
	if (!state.view || state.view.branch) return;
	const { slot, model } = state.view;
	try {
		await api.reorient(state.run, slot, model, id, {
			...opts,
			propagate: true,
		});
		toast(`re-fronting ${id}…`, "ok");
		pollGenerated();
	} catch (e) {
		toast(e.message, "err");
	}
}

// Force the window/glass transparency transform onto this object (and its whole
// prefab group) regardless of the pipeline's keyword + symmetry gates: the server
// bakes white texels to near-clear and re-optimizes each member's served mesh.
// Fast local reprocess, so it mirrors the symmetry handlers (no busy lock).
async function glassifyNode(id) {
	if (!state.view || state.view.branch) return;
	const { slot, model } = state.view;
	try {
		await api.glassify(state.run, slot, model, id);
		toast(`applying glass transparency to ${id} + prefab group…`, "ok");
		pollGenerated();
	} catch (e) {
		toast(e.message, "err");
	}
}

// Rebuild this object (and its prefab group) from the pristine raw mesh, dropping
// any in-place served edit (e.g. a forced glassify) while keeping its current
// symmetry. Fast local reprocess; mirrors the symmetry handlers (no busy lock).
async function resetNode(id) {
	if (!state.view || state.view.branch) return;
	const { slot, model } = state.view;
	try {
		await api.reset(state.run, slot, model, id);
		toast(`resetting ${id} + prefab group from raw…`, "ok");
		pollGenerated();
	} catch (e) {
		toast(e.message, "err");
	}
}

// Flip ONE object between its optimized (KTX2/Meshopt) and unoptimized ("raw")
// rescaled mesh in the main scene — a pure view swap (both variants already
// exist on disk), no rebuild and no server call. Overrides the scene-wide toggle
// for this object until the global toggle is flipped (which resets overrides).
function setObjectOptimized(id, mode) {
	if (!state.view || state.view.branch || assetMode !== "generated") return;
	const m = lastGenMesh.get(id);
	if (!m) return;
	objOptMode.set(id, mode);
	attachGeneratedMeshes([{ id, ...m }]);
	tracePanel.rerenderInfo();
}

// Permanently wipe an object from the open SOURCE cell — every reference in BOTH
// event logs (library + generated) and its mesh + image files in every build.
// Guarded by an explicit danger confirm so it can't be a stray click; the wipe
// is IRREVERSIBLE. On success the cell reloads so the scene + observability drop
// the node (orphaned children re-anchor to its region and a prefab canonical
// hands off, server-side). Works in either asset view; never on a branch.
function deleteNode(id) {
	const { slot, model, branch } = state.view ?? {};
	if (!slot || branch) return;
	const node = obs.model.nodes.get(id);
	const kind = node?.kind ?? "object";
	const desc = node?.prompt ? ` — “${node.prompt}”` : "";
	const wasGenerated = assetMode === "generated";
	openModal(`delete ${kind} ${id}?`, (close, setError) => ({
		body: [
			el("div", {
				class: "m-hint",
				text:
					`Permanently removes ${id}${desc} from ${slot} · ${model}: every reference in BOTH event logs ` +
					"(library + generated) and its mesh + image files across every build. Any object anchored to it " +
					"is re-parented to its region; if it is a prefab canonical, the shared mesh passes to one of its " +
					"reuses. This cannot be undone.",
			}),
		],
		actions: [
			el("button", { text: "cancel", onclick: close }),
			el("button", {
				class: "danger",
				text: "delete permanently",
				onclick: async () => {
					try {
						await api.deleteObject(state.run, slot, model, id);
					} catch (e) {
						setError(e.message);
						return;
					}
					close();
					toast(`deleted ${id}`, "ok");
					// Reload so the scene + obs reflect the wipe (openCell resets to
					// the library view); restore the generated view if that's where
					// the delete fired AND we're still on the same cell.
					await openCell({ slot, model, branch: false });
					if (
						wasGenerated &&
						state.view &&
						state.view.slot === slot &&
						state.view.model === model &&
						!state.view.branch
					)
						setAssetMode("generated");
					emit("poll-now");
				},
			}),
		],
	}));
}

// Drag the divider between the canvas and the observability dock to set the
// dock width; the canvas reflows via its ResizeObserver. Persisted so the
// chosen width survives reloads.
const OBSDOCK_WIDTH_KEY = "starshot.obsdockWidth";
const OBSDOCK_MIN = 320;
const CANVAS_MIN = 360;

function initObsResizer() {
	const resizer = document.getElementById("obsdock-resizer");
	const dock = document.getElementById("obsdock");
	const body = document.getElementById("overlay-body");
	let saved = NaN;
	try {
		saved = Number(localStorage.getItem(OBSDOCK_WIDTH_KEY));
	} catch {
		/* private mode */
	}
	if (saved >= OBSDOCK_MIN) dock.style.width = `${saved}px`;

	let dragging = false;
	resizer.addEventListener("pointerdown", (ev) => {
		dragging = true;
		resizer.classList.add("dragging");
		resizer.setPointerCapture(ev.pointerId);
		ev.preventDefault();
	});
	resizer.addEventListener("pointermove", (ev) => {
		if (!dragging) return;
		const rect = body.getBoundingClientRect();
		const max = Math.max(OBSDOCK_MIN, rect.width - CANVAS_MIN);
		const width = Math.max(
			OBSDOCK_MIN,
			Math.min(rect.right - ev.clientX, max),
		);
		dock.style.width = `${Math.round(width)}px`;
	});
	const end = (ev) => {
		if (!dragging) return;
		dragging = false;
		resizer.classList.remove("dragging");
		try {
			resizer.releasePointerCapture(ev.pointerId);
		} catch {
			/* already released */
		}
		try {
			localStorage.setItem(
				OBSDOCK_WIDTH_KEY,
				String(parseInt(dock.style.width, 10) || OBSDOCK_MIN),
			);
		} catch {
			/* private mode */
		}
	};
	resizer.addEventListener("pointerup", end);
	resizer.addEventListener("pointercancel", end);
}

// Re-aim the selection at a lineage node (a breadcrumb click in the trace
// panel). A node with a 3D box is selected + framed — that drives the panel via
// onSelect; a box-less ancestor (e.g. the root before its box paints) just
// updates the dock row + panel directly.
function focusNode(id) {
	if (viewer.hasBbox(id)) {
		viewer.select(id, { frame: true });
	} else {
		dock.markSelected(id, { scroll: true });
		tracePanel.show(obs.model, id);
	}
}

// Open the decision-inquiry chat grounded in a call — shared by the dock and the
// left trace panel's "why?" buttons. Reads the live view so it carries the
// current cell/run/branch context.
function openInquiryForCall(call) {
	if (!state.view) return;
	const { slot, model, branch } = state.view;
	inquiry.openInquiry(call, { run: state.run, slot, model, branch });
}

function scheduleTreeRender() {
	if (renderQueued) return;
	renderQueued = true;
	requestAnimationFrame(() => {
		renderQueued = false;
		dock.renderTree(obs.model, { streamed: true });
		// Fold any newly-streamed calls into the open emit trace too (no-ops
		// when nothing the panel shows changed, or while the user is reading it).
		tracePanel.refresh(obs.model, { streamed: true });
		if (state.view?.branch && state.lab.simStep)
			dock.renderPinned(obs.model);
		// New zones may have streamed in — refresh the legend so a new depth's
		// swatch appears (no-ops when the depth set is unchanged).
		refreshZoneLayers();
	});
}

export async function openCell({
	slot,
	model,
	branch = false,
	forceLive = false,
}) {
	const seq = ++openSeq;
	const run = state.run;
	// Keep the camera when this re-renders the SAME cell while the overlay is
	// already open — the branch selector swapping between the source run and its
	// per-LLM simulation lineages, or a revert/step reload. Swapping like that
	// shouldn't pull the user off the part of the scene they're inspecting. A
	// fresh open (overlay closed) or a different cell still frames the scene anew.
	const keepCamera =
		overlayEl.classList.contains("open") &&
		!!state.view &&
		state.view.slot === slot &&
		state.view.model === model;
	stream?.close();
	stream = null;
	// A fresh cell open always starts on the library build; the generated view
	// is opt-in per open via the asset toggle.
	assetMode = "library";
	stopGenPoll();
	clearGeneratedState();
	state.view = { slot, model, branch };
	// Close any inquiry chat that belongs to a different cell/mode (same-cell
	// re-subscribes from resume/step keep it open).
	inquiry.notifyView({ run, slot, model, branch });
	overlayEl.classList.add("open");
	viewer.setActive(true);
	viewer.clear({ keepCamera });
	obs = createObsModel();
	dock.resetDock();
	// Selection is per-cell (viewer.clear drops it without notifying), so the
	// left trace panel can't carry a stale focus into the incoming scene.
	tracePanel.reset();
	dock.setPinStep(branch ? state.lab.simStep : null);
	// Per-call revert (the ⏪ on each call row): a source cell rewinds + re-runs
	// from the cut; a branch rewinds ITS OWN log to before the step and pauses
	// there (source untouched), ready to step forward again.
	dock.setOnRevert(branch ? revertBranchToCall : revertToCall);
	// "+ sim" on a zone row drops it into the prompt lab's simulation slots. A
	// source-cell action — you fork a NEW branch from the source, not from one.
	dock.setOnAddSim(
		branch ? null : (node) => emit("add-sim-target", { slot, model, node }),
	);
	renderHeader();
	// Clear any stale layer legend from the previous cell while this one loads;
	// it refills once the scene + history fold in below.
	refreshZoneLayers();

	let projection = { nodes: [], last_index: -1 };
	try {
		projection = branch
			? await api.branchScene(branch)
			: await api.scene(run, slot, model, {});
	} catch (e) {
		toast(`scene load failed: ${e.message}`, "err");
	}
	if (seq !== openSeq) return;
	applySceneProjection(viewer, projection);
	viewer.prefetchBundle(
		branch
			? api.branchMeshesUrl(branch)
			: api.meshesUrl(run, slot, model, {}),
	);

	let history = [];
	try {
		history = branch
			? await api.branchEventsHistory(run, branch)
			: await api.eventsHistory(run, slot, model);
	} catch {
		/* never-started cell */
	}
	if (seq !== openSeq) return;
	for (const event of history) obs.feed(event);
	// The scene projection painted bboxes before this history folded, so objects
	// were colored the default green; repaint now that each node's emitting step
	// is known. (Streamed bboxes color correctly on paint — their decompose call
	// always precedes the bbox event.)
	viewer.recolorAll();
	dock.renderTree(obs.model);
	// Depths are known now that history folded — (re)build the layer legend.
	refreshZoneLayers();
	if (branch && state.lab.simStep) dock.renderPinned(obs.model);
	renderHeader(); // error message comes from the just-loaded log

	// `forceLive` subscribes without waiting for the next poll — used right
	// after a revert relaunches the cell, so the polled summary is still stale.
	// A cell parked at a step gate (status paused WITH a pending call) still has
	// a live task, so subscribe — the next "step" streams its events.
	const summary = currentSummary();
	const info = branch ? branchSummaryById(branch) : summary;
	const live = forceLive || info?.status === "running" || !!info?.pending;
	if (live) {
		const since = Math.max(obs.model.maxIndex, projection.last_index ?? -1);
		subscribe(seq, since);
	}
}

function subscribe(seq, since) {
	stream = openStream(buildUrl(since), {
		onEvent: (event) => {
			if (seq !== openSeq) return;
			if (!obs.feed(event)) return;
			dispatchSceneEvent(viewer, event);
			scheduleTreeRender();
		},
		onTerminal: () => {
			if (seq !== openSeq) return;
			renderHeader();
			emit("poll-now");
		},
	});
}

function buildUrl(since) {
	const { slot, model, branch } = state.view;
	return branch
		? api.branchEventsUrl(branch, { since })
		: api.eventsUrl(state.run, slot, model, { since });
}

// Re-attach the SSE tail to the CURRENT cell without reloading it — used on
// resume/start in place of a full `openCell`, so the scene + observability dock
// stay exactly as they are and only new events fold in. Rolls back any folded
// terminal sentinel first (resume truncates it server-side and reuses its
// index), then tails from where we left off.
function relinkStream() {
	stream?.close();
	stream = null;
	obs.rewindTerminal();
	subscribe(openSeq, obs.model.maxIndex);
	renderHeader();
}

function currentSummary() {
	if (!state.view) return null;
	return cellSummary(state.view.slot, state.view.model);
}

function renderHeader() {
	if (!state.view) return;
	const { slot, model, branch } = state.view;
	const summary = currentSummary();
	const branchInfo = branch ? branchSummaryById(branch) : null;
	// A per-LLM lineage is pinned to a model; surface it so the cell's many
	// lineages aren't all indistinguishably labeled with the cell's base model.
	const pin =
		branchInfo?.pin && branchInfo.pin !== model ? branchInfo.pin : null;
	titleEl.textContent = `${slot} · ${model}${pin ? " → " + pin : ""}`;
	crumbsEl.textContent = `${state.run}${branch ? (pin ? ` · sim on ${pin}` : " · simulation branch") : ""}`;
	const cellInfo = branch ? branchInfo : summary;
	const view = statusView(cellInfo);
	const status = view.state;
	const pending = cellInfo?.pending;
	dotEl.className = `dot ${view.dot}`;
	// One canonical status message (status.js), plus the overlay-only extras:
	// the stepped-mode tag, the event count, and the error reason.
	let statusText = view.label;
	if (!branch && summary?.stepped) statusText += " · stepped";
	statusText += ` · ${cellInfo?.events_count ?? 0} events`;
	if (status === "error") {
		// Put the failure reason where the eye lands first; the log strip below
		// has the full trail.
		const err = obs.lastError?.();
		if (err) statusText += ` — ${err.text}`;
	}
	statusEl.textContent = statusText;
	statusEl.title = statusText;
	statusEl.classList.toggle("is-error", status === "error");

	// Action button: source cells start/resume/pause; branches pause/resume.
	// "Live" = a running task OR one parked at a step gate (pending) — both are
	// pausable; only a cell with no live task resumes/starts.
	const live = status === "running" || !!pending;
	let label = null;
	if (branch) {
		if (live) label = "pause sim";
		else if (status === "paused" || status === "error")
			label = "resume sim";
	} else {
		if (live) label = "pause";
		else if (status === "idle") label = "start";
		else if (status === "paused") label = "resume";
		else if (status === "error") label = "retry";
	}
	actionBtn.style.display = label ? "" : "none";
	actionBtn.textContent = label ?? "";
	resetBtn.style.display = branch ? "none" : "";

	// One-call-at-a-time stepping controls. Shown whenever the cell is gated:
	// a live branch always is; a source cell whenever it's in step mode (so
	// the button is there even when paused with no live gate — incl. after a
	// restart). `done` cells have nothing left to step.
	const stepped = branch || !!summary?.stepped;
	const canStep = stepped && status !== "done";
	let stepBtn = document.getElementById("overlay-step");
	let autoBtn = document.getElementById("overlay-auto");
	if (canStep) {
		if (!stepBtn) {
			stepBtn = el("button", { id: "overlay-step", class: "primary" });
			actionBtn.before(stepBtn);
		}
		if (!autoBtn) {
			autoBtn = el("button", { id: "overlay-auto" });
			actionBtn.before(autoBtn);
		}
		stepBtn.textContent = pending ? `step: ${pending.step}` : "step";
		stepBtn.title = pending
			? `run the pending ${pending.step} call on ${pending.node ?? "?"}`
			: "run the next LLM call, then pause again";
		autoBtn.textContent = "run rest";
		autoBtn.title = "finish without further pauses";
		stepBtn.onclick = () => stepCurrent(false);
		autoBtn.onclick = () => stepCurrent(true);
		// Source cells (not branches) can fast-forward to a target step.
		let untilSel = document.getElementById("overlay-step-until");
		if (!branch && state.steps.length) {
			if (!untilSel) {
				untilSel = stepUntilSelect(
					state.steps,
					(until) => stepCurrent(false, until),
					{ label: "until…" },
				);
				untilSel.id = "overlay-step-until";
				autoBtn.before(untilSel);
			}
		} else {
			untilSel?.remove();
		}
	} else {
		stepBtn?.remove();
		autoBtn?.remove();
		document.getElementById("overlay-step-until")?.remove();
	}

	// Branch access: a selector to jump between the source run and EACH of the
	// cell's downstream simulations (lineages). Available even on a DONE cell —
	// whose only source action is reset — so sims that branched off earlier (and
	// may still be running / paused) are always reachable here. Picking one loads
	// it; from a branch its own step/pause/resume controls take over.
	document.getElementById("overlay-flip")?.remove(); // legacy single-branch flip
	let branchSel = document.getElementById("overlay-branch-sel");
	const cellBs = cellBranches(slot, model);
	if (branch || cellBs.length) {
		const sig = `${branch ?? ""}|${cellBs.map((b) => `${b.id}:${statusView(b).state}`).join(",")}`;
		// Don't rebuild the <select> while the user has it open (a poll would close
		// the dropdown); refresh only when the option set / statuses actually change.
		if (
			!branchSel ||
			(branchSel.dataset.sig !== sig &&
				document.activeElement !== branchSel)
		) {
			if (!branchSel) {
				branchSel = el("select", {
					id: "overlay-branch-sel",
					title: "view the source run or one of its downstream simulations",
				});
				// Read the CURRENT view at change time — this <select> element persists
				// across renders/cells (only its options are rebuilt), so capturing the
				// creating render's slot/model would go stale: switching while viewing a
				// DIFFERENT cell's branch would re-open under the wrong cell, whose
				// options exclude the viewed branch → the dropdown blanks out.
				branchSel.addEventListener("change", () => {
					const v = state.view;
					if (v)
						openCell({
							slot: v.slot,
							model: v.model,
							branch: branchSel.value || null,
						});
				});
				actionBtn.before(branchSel);
			}
			branchSel.dataset.sig = sig;
			branchSel.replaceChildren(
				el("option", { value: "", text: "view: source run" }),
				...cellBs.map((b) => {
					const lab = b.pin ?? b.model ?? model;
					const node = b.last_step?.node;
					const st = statusView(b).state;
					return el("option", {
						value: b.id,
						text: `⑂ sim · ${lab}${node ? " @ " + node : ""} · ${st}`,
					});
				}),
			);
			branchSel.value = branch ?? "";
		}
	} else if (branchSel) {
		branchSel.remove();
	}
	syncAssetControls();
}

// Revert the open source cell to just before `call` — confirms first (it
// drops every later step + its meshes), then re-runs the pipeline from the cut
// and reloads the overlay streaming the re-run live.
function revertToCall(call) {
	const { slot, model, branch } = state.view ?? {};
	if (!slot || branch) return;
	const step = call.template ?? call.step ?? "this step";
	openModal(`revert ${slot} · ${model}?`, (close, setError) => ({
		body: [
			el("div", {
				class: "m-hint",
				text:
					`Truncates this slot's log to just before its ${step} call (#${call.index}), ` +
					"drops every later step and its meshes, then re-runs the pipeline from there.",
			}),
		],
		actions: [
			el("button", { text: "cancel", onclick: close }),
			el("button", {
				class: "danger",
				text: "revert & re-run",
				onclick: async () => {
					try {
						await api.rewind(state.run, slot, model, call.index);
					} catch (e) {
						setError(e.message);
						return;
					}
					close();
					toast(
						`reverted ${slot} · ${model} to #${call.index}`,
						"ok",
					);
					// The cell is already relaunching server-side; reload at the cut and
					// stream the re-run (forceLive — the polled summary is still stale).
					openCell({ slot, model, branch: false, forceLive: true });
					emit("poll-now");
				},
			}),
		],
	}));
}

// Revert the open simulation BRANCH to just before `call` — confirms first (it
// drops every later step + its meshes on this branch), then truncates the
// branch's log and PAUSES there. The source run is untouched; stepping forward
// re-runs from the cut under the branch's edits. The branch mirror of
// `revertToCall`, but non-destructive to the source (and it pauses rather than
// auto-re-running, since a branch advances one manual step at a time).
function revertBranchToCall(call) {
	const { slot, model, branch } = state.view ?? {};
	if (!slot || !branch) return;
	const step = call.template ?? call.step ?? "this step";
	openModal(`revert this simulation?`, (close, setError) => ({
		body: [
			el("div", {
				class: "m-hint",
				text:
					`Truncates this simulation branch to just before its ${step} call (#${call.index}), ` +
					"drops every later step and its meshes, and pauses there — “step” then re-runs from the cut. " +
					"The source run is untouched.",
			}),
		],
		actions: [
			el("button", { text: "cancel", onclick: close }),
			el("button", {
				class: "danger",
				text: "revert & pause",
				onclick: async () => {
					// overrides=null keeps the branch's own edit set (the overlay isn't the
					// prompt lab, so it doesn't re-apply the lab's live drafts).
					try {
						await api.branchRewind(branch, call.index);
					} catch (e) {
						setError(e.message);
						return;
					}
					close();
					toast(`reverted simulation to #${call.index}`, "ok");
					// Reload the (now-paused, truncated) branch at the cut; emit a poll so
					// the header status catches up from the still-stale summary.
					openCell({ slot, model, branch });
					emit("poll-now");
				},
			}),
		],
	}));
}

async function stepCurrent(auto, until = null) {
	const { slot, model, branch } = state.view ?? {};
	if (!slot) return;
	try {
		if (branch) {
			await api.branchStep(branch, { auto });
		} else {
			const r = await api.cellStep(state.run, slot, model, {
				auto,
				until,
			});
			// A paused cell that was just relaunched (vs. a live gate released over
			// the existing stream) needs a fresh subscription to watch the call land.
			if (r.result === "launched") {
				setTimeout(() => {
					if (
						state.view &&
						state.view.slot === slot &&
						state.view.model === model &&
						!state.view.branch
					) {
						openCell({ slot, model, branch: false });
					}
				}, 250);
			}
		}
		emit("poll-now");
	} catch (e) {
		toast(e.message, "err");
	}
}

async function onAction() {
	const { slot, model, branch } = state.view ?? {};
	if (!slot) return;
	const summary = currentSummary();
	const info = branch ? branchSummaryById(branch) : summary;
	const status = info?.status;
	// Parked-at-gate (pending) counts as live, so the button pauses (breaks out
	// of stepping) rather than mis-firing a resume on a cell that's still gated.
	const pausing = status === "running" || !!info?.pending;
	try {
		if (branch) {
			if (pausing) await api.branchPause(branch);
			else await api.branchResume(branch);
		} else {
			if (pausing) await api.pause(state.run, slot, model);
			else await api.resume(state.run, slot, model);
		}
		emit("poll-now");
		// Bail if the view moved while the request was in flight.
		if (
			!state.view ||
			state.view.slot !== slot ||
			state.view.model !== model ||
			state.view.branch !== branch
		)
			return;
		// Re-wire the live stream IN PLACE rather than reloading the cell — no
		// viewer.clear / dock reset, so the scene + observability stay put.
		// Pausing lets the open stream wind down on its run.paused; resuming/
		// starting re-tails so new events fold into the existing view.
		if (pausing) renderHeader();
		else relinkStream();
	} catch (e) {
		toast(e.message, "err");
	}
}

async function onReset() {
	const { slot, model } = state.view ?? {};
	if (!slot) return;
	if (!confirm(`Wipe ${slot} · ${model} on "${state.run}" and start fresh?`))
		return;
	try {
		await api.reset(state.run, slot, model, true);
		emit("poll-now");
		openCell({ slot, model, branch: false });
	} catch (e) {
		toast(e.message, "err");
	}
}

export function closeOverlay() {
	openSeq += 1;
	stream?.close();
	stream = null;
	stopGenPoll();
	assetMode = "library";
	state.view = null;
	overlayEl.classList.remove("open");
	viewer.setActive(false);
	dock.setPinStep(null);
	tracePanel.reset();
	inquiry.closeInquiry();
}
