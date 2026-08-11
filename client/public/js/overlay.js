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
import { el, toast, stepUntilSelect, openModal, buildObjectsMenu, promptCapValue, fmtDurationMs } from "./ui.js";
import {
	openStream,
	dispatchSceneEvent,
	applySceneProjection,
	createObsModel,
	emittedStep,
	nodeStepsOf,
} from "./events.js";
import { statusView } from "./status.js";
import { createObsDock } from "./obstree.js";
import { createTracePanel } from "./tracepanel.js";
import { createInvestigator } from "./investigator.js";
import { initGenFailPanel } from "./genfails.js";
import { initReplay } from "./replay.js";

const overlayEl = document.getElementById("overlay");
const titleEl = document.getElementById("overlay-title");
const crumbsEl = document.getElementById("overlay-crumbs");
const dotEl = document.getElementById("overlay-dot");
const statusEl = document.getElementById("overlay-status");
const actionBtn = document.getElementById("overlay-action");
const resetBtn = document.getElementById("overlay-reset");
const btnZoneLayers = document.getElementById("btn-zone-layers");
const btnReplay = document.getElementById("btn-replay");
const zoneLayersLegendEl = document.getElementById("zone-layers-legend");

let viewer = null;
let dock = null;
let tracePanel = null;
let replay = null; // structural-call stepper over the event log (replay.js)
let stream = null;
let obs = createObsModel();
let renderQueued = false;
let openSeq = 0; // monotonically increasing guard for async open races

// The obs model every node-INSPECTION surface reads: the hover tooltip, the
// emittance trace, the node tree. While replay owns the scene this is its
// prefix-only model, so nothing describes a call that hasn't happened at the
// current cut — a zone with no plan yet shows no plan. The streaming machinery
// (`obs.feed`, `obs.model.maxIndex` for the SSE cursor) always uses the real
// model; only what the user READS is cut back.
const viewObs = () => (replay?.isActive() ? replay.model() : obs.model);
let lastLayersSig = null; // skips rebuilding the layer legend when unchanged
// Asset view: which mesh build the 3D view shows + the generate gate's polling.
let assetMode = "library"; // "library" | "generated"
let optimizedView = true; // generated meshes: optimized KTX2/Meshopt twin vs raw
let liteOn = false; // when on, show the lite presentation tier (overrides optimized/raw)
// The effective scene-wide variant for generated meshes.
const genVariant = () => (liteOn ? "lite" : optimizedView ? "optimized" : "raw");
// The from-scratch generated build is VERSIONED: a cell can hold many isolated
// builds (V1/V2/V3…). `genVersion` is the one the generated view shows + builds
// (null = let the server resolve the latest); `genVersions` is every id on disk.
let genVersion = null;
let genVersions = [];
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
let liteBtn = null;
let liteGenBtn = null;
let genStatusEl = null;
let verSel = null; // generated-version <select>
let newVerBtn = null; // "＋ new version" button
let genFails = null; // per-object report of what this build didn't return

// Cache-bust a generated mesh URL by its mtime token, so a regenerated asset
// (same id + path, new bytes) reloads instead of serving a stale cached GLB.
const withV = (url, v) => url + (url.includes("?") ? "&" : "?") + "v=" + v;

// Set when this cell was opened from the api-log, so the header can offer a
// "← logs" return (the log stays one Escape/back away, scene loaded behind it).
let cameFromLog = false;
let toLogsBtn = null;

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
		onInquire: investigateCall,
		onOpenLog: openLogForCall,
		// "obs ↗" on a trace step → expand + reveal that same call in the
		// observability dock (its full, untruncated bytes vs. the trace's node slice).
		onOpenInObs: (call) => dock.expandCall(call),
		loadCallBytes,
		// Project (or clear) a focused-object map onto its mesh; returns whether it
		// took. `mapProjectionOf` reads the viewer's live state back for the panel.
		onProjectMap: (id, desc) =>
			desc ? viewer.setMapProjection(id, desc) : viewer.clearMapProjection(),
		mapProjectionOf: (id) => viewer.getMapProjection(id),
		// Interior gate for a focused zone. Source cells only — a branch has its
		// own log and no gate endpoint of its own.
		gateFor: (zoneId) =>
			state.view && !state.view.branch
				? api.gate(state.run, state.view.slot, state.view.model, zoneId)
				: null,
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
			// Every OTHER object this asset can be linked to — one entry per
			// object (canonical OR reuse), excluding the focused node itself and
			// the rest of its own group (linking within a group is a no-op). The
			// server resolves the chosen object to its prefab group's canonical,
			// so ANY member is a valid target. Each entry carries the object's id
			// (the visible label), its prompt (a hover tooltip), and the size of
			// the group it would join.
			linkTargets: (id) => {
				const me = lastGenMesh.get(id);
				const myCanon = me ? me.canonical : null;
				const groupSize = new Map();
				for (const v of lastGenMesh.values())
					groupSize.set(
						v.canonical,
						(groupSize.get(v.canonical) || 0) + 1,
					);
				const out = [];
				for (const [nid, m] of lastGenMesh) {
					if (nid === id || m.canonical === myCanon) continue;
					const node = obs.model.nodes.get(nid);
					out.push({
						id: nid,
						prompt: node?.prompt || "",
						size: groupSize.get(m.canonical) || 1,
					});
				}
				out.sort((a, b) => a.id.localeCompare(b.id));
				return out;
			},
			isBusy: (id) => busyNodes.has(id),
			imagePromptOf: (id) => lastGenMesh.get(id)?.imagePrompt ?? null,
			// True when this object renders (stale optimized twin) but its raw /
			// unoptimized mesh is missing — a regenerate that died midway. The
			// panel badges it so the "ghost" isn't silent and the user knows to
			// regenerate to rebuild it.
			incompleteOf: (id) => lastGenMesh.get(id)?.incomplete ?? false,
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
			onUnlink: unlinkNode,
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
	viewer.setNodeInfo((id) => viewObs().nodes.get(id) ?? null);
	// Hovering a bbox in 3D lists the pipeline steps that ran on/for the node
	// (its calls + the provenance that named/placed it) under its base info —
	// read from the same folded obs model the tree uses.
	viewer.setNodeSteps((id) => nodeStepsOf(viewObs(), id));
	// Color objects in 3D by the decomposition step that emitted them (next_object
	// purple, anchor green, negative_space brown) — read from the same provenance
	// the tree shows as "via {step}". `recolorAll` (below) repaints once a load's
	// history has folded, since the scene projection paints bboxes first.
	viewer.setOriginOf((id) => emittedStep(viewObs(), id));
	// A 3D pick (or any selection) reveals the dock row AND drives the left
	// emittance-trace panel; deselecting hides it. The panel only opens while a
	// scene is actually up (state.view set) — never on a stray selection event
	// with no cell loaded.
	viewer.onSelect((id) => {
		dock.markSelected(id, { scroll: true });
		if (id && state.view) tracePanel.show(viewObs(), id);
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
	viewer.onHiddenChange(() => dock.renderTree(viewObs()));

	// "why?" on any call row (dock OR the left emittance-trace panel) → open the
	// investigator with that step attached. Reads the live view at click time, so
	// it's shared across source AND branch views.
	dock.setOnInquire(investigateCall);
	dock.setOnOpenLog(openLogForCall);
	dock.setCallBytesLoader(loadCallBytes);

	const closeBtn = document.getElementById("overlay-close");
	closeBtn.addEventListener("click", closeOverlay);
	// When this cell was reached from the api-log, offer a one-click return to
	// it. The log overlays the scene (which stays loaded), so it's a fast toggle.
	toLogsBtn = el("button", {
		class: "ov-to-logs",
		text: "← logs",
		title: "return to the api log (scene stays loaded)",
		onclick: () => emit("open-flights"),
	});
	toLogsBtn.style.display = "none";
	closeBtn.after(toLogsBtn);
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
	// The objects layer is a multiselect (anchors / next / negative space /
	// frames, filtered by emitting step) rather than a single toggle — insert it
	// at the head of the row. The remaining layers stay plain on/off buttons.
	const togglesRow = document.getElementById("viewer-toggles");
	togglesRow.prepend(buildObjectsMenu(viewer));
	// The row wraps when the canvas is narrow, so the panels floating directly
	// above it (lighting, the generated-failure report) can't use a fixed offset.
	// Publish its live height for them to sit on.
	if ("ResizeObserver" in window) {
		const host = document.getElementById("canvas-host");
		const publishHeight = () =>
			host.style.setProperty("--toggles-h", `${togglesRow.offsetHeight}px`);
		new ResizeObserver(publishHeight).observe(togglesRow);
		publishHeight();
	}
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
	// Replay: hands the scene over to a paced re-walk of the committed log.
	// Leaving it repaints the cell the normal way, since replay tore the scene
	// down to rebuild it and has no full-scene state of its own to restore.
	replay = initReplay({
		viewer,
		host: document.getElementById("canvas-host"),
		// Every time the cut moves, the panels have to be re-read against the
		// prefix model — otherwise the tree and trace keep describing the finished
		// run while the scene shows an earlier step. Recolor too: an object's colour
		// comes from the decompose that emitted it, which may not have run yet.
		onCut: () => {
			viewer.recolorAll();
			dock.renderTree(viewObs());
			tracePanel.refresh(viewObs());
			refreshZoneLayers();
		},
		onExit: () => {
			btnReplay.classList.add("off");
			const view = state.view;
			if (view) openCell({ slot: view.slot, model: view.model, branch: view.branch });
		},
	});
	btnReplay.addEventListener("click", () => {
		btnReplay.classList.toggle("off", !replay.toggle());
	});
	document
		.getElementById("btn-refit")
		.addEventListener("click", () => viewer.fit());
	document
		.getElementById("btn-unhide-all")
		.addEventListener("click", () => viewer.unhideAll());
	// First-person camera toggle: the button enters FP (pointer lock); the viewer
	// flips back to orbit on Esc, so mirror its state via onCameraModeChange.
	const btnCameraMode = document.getElementById("btn-camera-mode");
	btnCameraMode.addEventListener("click", () =>
		viewer.setCameraMode(viewer.getCameraMode() === "fp" ? "orbit" : "fp"),
	);
	viewer.onCameraModeChange((mode) => {
		btnCameraMode.classList.toggle("on", mode === "fp");
		btnCameraMode.textContent =
			mode === "fp" ? "first-person ✓" : "first-person";
	});
	setupAssetControls();
	actionBtn.addEventListener("click", onAction);
	resetBtn.addEventListener("click", onReset);
	on("slots", () => {
		if (state.view) renderHeader();
	});
	on("open-cell", openCell);
	initObsResizer();
	setupInvestigator();
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
	// The generated build is versioned: this <select> picks which build (V1/V2/V3…)
	// the view shows + the ⚡ generate button builds/resumes; "＋ new version" spins
	// up the next empty one. Both are generated-view-only (see syncAssetControls).
	verSel = el("select", {
		id: "btn-gen-version",
		title: "which generated version (V1/V2/V3…) to view + build/resume",
		onchange: () => onVersionChange(verSel.value),
	});
	newVerBtn = el("button", {
		id: "btn-gen-new-version",
		title: "start a brand-new generated version, built fresh and kept separate from the others",
		text: "＋ new version",
		onclick: onNewVersion,
	});
	// Quality tier WITHIN the selected version — orthogonal to it, so every
	// version carries the same raw / optimized / lite set.
	liteBtn = el("button", {
		id: "btn-asset-lite",
		title: "generated view: show the LITE presentation tier — visually identical to raw, far smaller (overrides optimized/raw)",
		text: "lite",
		onclick: toggleLite,
	});
	liteGenBtn = el("button", {
		id: "btn-build-lite",
		title: "build (or resume) the selected version's lite presentation assets (objects-generated-lite)",
		text: "⚡ build lite",
		onclick: onBuildLite,
	});
	genStatusEl = el("span", { id: "gen-status", class: "gen-status" });
	// Clicking a failed object frames it in 3D, so "which one is chair_4?" is one
	// click rather than a hunt — the box is there even when the mesh never landed.
	genFails = initGenFailPanel({ onSelect: focusNode });
	refit.after(
		assetBtn,
		genBtn,
		verSel,
		newVerBtn,
		optBtn,
		liteBtn,
		liteGenBtn,
		genFails.button,
		genStatusEl,
	);
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
	verSel.style.display = gen ? "" : "none";
	newVerBtn.style.display = gen ? "" : "none";
	optBtn.style.display = gen ? "" : "none";
	liteBtn.style.display = gen ? "" : "none";
	liteGenBtn.style.display = gen ? "" : "none";
	genStatusEl.style.display = gen ? "" : "none";
	genFails.setVisible(gen);
	assetBtn.classList.toggle("on", assetMode === "generated");
	assetBtn.textContent =
		assetMode === "generated" ? "generated ✓" : "generated";
	// optimized/raw sub-toggle; the lite toggle overrides it (dimmed while on).
	optBtn.classList.toggle("on", !liteOn && optimizedView);
	optBtn.textContent = optimizedView ? "optimized ✓" : "raw";
	optBtn.disabled = liteOn;
	liteBtn.classList.toggle("on", liteOn);
	liteBtn.textContent = liteOn ? "lite ✓" : "lite";
	syncVersionSelector();
}

// Rebuild the version <select> from `genVersions`/`genVersion` (newest first).
// Skipped while the user has it open (a poll must not close the dropdown) and
// when nothing changed; an empty list shows a placeholder until the first build.
function syncVersionSelector() {
	if (!verSel) return;
	const sig = `${genVersions.join(",")}|${genVersion ?? ""}`;
	if (document.activeElement === verSel) return;
	if (verSel.dataset.sig !== sig) {
		verSel.dataset.sig = sig;
		const opts = genVersions.length
			? genVersions
					.slice()
					.reverse()
					.map((v) => el("option", { value: String(v), text: `V${v}` }))
			: [el("option", { value: "", text: "no versions yet" })];
		verSel.replaceChildren(...opts);
	}
	if (genVersion != null && genVersions.map(String).includes(String(genVersion)))
		verSel.value = String(genVersion);
}

function setAssetMode(mode) {
	if (!state.view || state.view.branch || mode === assetMode) return;
	assetMode = mode;
	stopGenPoll();
	clearGeneratedState();
	// Drop the current build's meshes so the incoming view shows ONLY its own —
	// generated mode never layers over leftover library meshes (and vice versa).
	viewer.clearMeshes();
	tracePanel.clearProjection(); // clearMeshes ended it viewer-side
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
	liteOn = false; // the optimized/raw toggle takes over from lite
	repullGenerated();
}

// Separate toggle for the lite presentation tier: when on, the scene shows the
// lite variant (overriding the optimized/raw toggle); off falls back to it.
function toggleLite() {
	if (!state.view || state.view.branch || assetMode !== "generated") return;
	liteOn = !liteOn;
	repullGenerated();
}

// Re-pull the generated meshes from the freshly-chosen variant dir: the scene-
// wide flip is a fresh baseline, so drop per-object overrides + what's loaded and
// let the poll re-attach.
function repullGenerated() {
	objOptMode = new Map();
	syncAssetControls();
	stopGenPoll();
	loadedGen = new Map();
	genMeshSig = null;
	viewer.clearMeshes();
	tracePanel.clearProjection(); // clearMeshes ended it viewer-side
	tracePanel.rerenderInfo();
	pollGenerated();
}

// Build (or resume) the SELECTED version's lite assets, then poll until done
// and — if the lite view is active — re-pull to show them.
async function onBuildLite() {
	if (!state.view || state.view.branch) return;
	const { slot, model } = state.view;
	try {
		await api.buildLite(state.run, slot, model, { version: genVersion });
		toast(`building lite for ${slot} · ${model}…`, "ok");
		pollLiteBuild();
	} catch (e) {
		toast(e.message, "err");
	}
}

let liteBuildTimer = null;
async function pollLiteBuild() {
	if (liteBuildTimer) {
		clearTimeout(liteBuildTimer);
		liteBuildTimer = null;
	}
	if (!state.view || state.view.branch) return;
	const { slot, model } = state.view;
	const version = genVersion;
	let st = null;
	try {
		st = await api.buildLiteStatus(state.run, slot, model, { version });
	} catch {
		/* transient — retried below if still running */
	}
	// Cell or version changed under us — stop.
	if (
		!state.view ||
		state.view.slot !== slot ||
		state.view.model !== model ||
		genVersion !== version
	)
		return;
	if (!st) return;
	if (genStatusEl && assetMode === "generated") {
		genStatusEl.textContent = st.running
			? `building lite… ${st.done}/${st.total}`
			: st.ok === false
				? "lite build failed"
				: `${st.done} lite`;
	}
	if (st.running) {
		liteBuildTimer = setTimeout(pollLiteBuild, 1500);
	} else if (liteOn && assetMode === "generated") {
		repullGenerated(); // freshly-built lite meshes are now on disk
	}
}

// Forget the generated-view bookkeeping (on cell open, mode switch, or toggle).
// Resets the selected version too, so the next poll re-resolves the cell's
// latest build.
function clearGeneratedState() {
	lastGenMesh = new Map();
	loadedGen = new Map();
	busyNodes = new Set();
	objOptMode = new Map();
	genMeshSig = null;
	genVersion = null;
	genVersions = [];
	if (genStatusEl) genStatusEl.textContent = "";
	genFails?.clear();
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

// ⚡ generate: build/resume the SELECTED version (the server resolves the latest
// when none is selected yet). Adopt the version + list it returns so the selector
// reflects a first-ever build (which mints V1) immediately.
async function onGenerate() {
	if (!state.view || state.view.branch) return;
	const { slot, model } = state.view;
	try {
		const r = await api.generate(state.run, slot, model, { version: genVersion });
		if (r?.version != null) genVersion = String(r.version);
		if (Array.isArray(r?.versions)) genVersions = r.versions.map(String);
		syncVersionSelector();
		toast(`generating ${slot} · ${model} · V${genVersion}…`, "ok");
		pollGenerated();
	} catch (e) {
		toast(e.message, "err");
	}
}

// ＋ new version: mint the next version and build it fresh. It starts empty, so
// drop the current version's meshes and let the poll re-attach V<new>'s as they land.
async function onNewVersion() {
	if (!state.view || state.view.branch) return;
	const { slot, model } = state.view;
	try {
		const r = await api.generate(state.run, slot, model, { newVersion: true });
		if (r?.version != null) genVersion = String(r.version);
		if (Array.isArray(r?.versions)) genVersions = r.versions.map(String);
		loadedGen = new Map();
		lastGenMesh = new Map();
		busyNodes = new Set();
		objOptMode = new Map();
		genMeshSig = null;
		genFails.clear();
		viewer.clearMeshes();
		tracePanel.clearProjection();
		tracePanel.rerenderInfo();
		syncVersionSelector();
		toast(`building new version V${genVersion} of ${slot} · ${model}…`, "ok");
		pollGenerated();
	} catch (e) {
		toast(e.message, "err");
	}
}

// Switch which generated version the view shows — like the optimized toggle, it
// drops the current meshes and lets the poll re-attach the chosen version's set
// (each version is fully isolated on disk).
function onVersionChange(v) {
	if (!state.view || state.view.branch || assetMode !== "generated") return;
	if (!v || String(v) === String(genVersion)) return;
	genVersion = String(v);
	stopGenPoll();
	loadedGen = new Map();
	lastGenMesh = new Map();
	busyNodes = new Set();
	objOptMode = new Map();
	genMeshSig = null;
	genFails.clear();
	viewer.clearMeshes();
	tracePanel.clearProjection();
	tracePanel.rerenderInfo();
	syncVersionSelector();
	pollGenerated();
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
	const mode = objOptMode.get(m.id) ?? genVariant();
	if (mode === "lite" && m.liteUrl) return { url: m.liteUrl, v: m.liteV };
	if (mode === "optimized" && m.optUrl) return { url: m.optUrl, v: m.optV };
	if (mode === "raw" && m.unoptUrl) return { url: m.unoptUrl, v: m.unoptV };
	// Fallbacks so a half-built object still renders (prefer optimized, then
	// lite, then the raw twin, then the status url).
	if (m.optUrl) return { url: m.optUrl, v: m.optV };
	if (m.liteUrl) return { url: m.liteUrl, v: m.liteV };
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
			variant: genVariant(),
			version: genVersion,
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
	// Adopt the resolved version + full list so the selector reflects the server's
	// truth (a first poll resolves "latest"; a new build appears in the list).
	if (Array.isArray(status.versions)) genVersions = status.versions.map(String);
	if (status.version != null) genVersion = String(status.version);
	syncVersionSelector();
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
				liteUrl: m.liteUrl,
				liteV: m.liteV,
				// Renders (stale optimized twin) but its raw/unoptimized backing is
				// missing — a regen that died midway. Surfaced so it isn't silent.
				incomplete: !!m.incomplete,
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
	// Torn "ghost" assets: a served twin whose raw/unoptimized backing is missing
	// (a regen that didn't finish). Exclude any currently rebuilding (busy). Flag
	// them so the loss isn't silent — regenerating each rebuilds its raw.
	const ghosts = meshes.filter(
		(m) => m.incomplete && !busyNodes.has(m.id),
	).length;
	const base = status.running ? `building… ${frac}` : `${frac} generated`;
	genStatusEl.textContent = base + (ghosts ? ` · ⚠ ${ghosts} incomplete` : "");
	genStatusEl.title = ghosts
		? `${ghosts} object${ghosts === 1 ? "" : "s"} render from a stale optimized mesh but are missing their raw/unoptimized files (an unfinished regenerate). Regenerate each to rebuild it.`
		: "";
	// Which objects came back with nothing, and why. Folded server-side from the
	// version's own events.generated.jsonl — no SSE stream carries that log, so
	// this poll is the only route any of it has to the screen.
	genFails.render(status.failures, { running: !!status.running });
	if (status.running) genPollTimer = setTimeout(pollGenerated, 1500);
}

// Per-object actions from the trace panel (generated build only). Each enqueues
// server-side work + polls so the new mesh swaps in when it lands. A plain
// regenerate propagates across the object's prefab group; `unlink` splits the
// object out of its group into a standalone asset with its own raw mesh (no
// backend call); `link` moves it into another object's group (re-deriving its
// mesh, no backend call).
async function regenerateNode(id, opts) {
	if (!state.view || state.view.branch) return;
	const { slot, model } = state.view;
	// Disable the asset's controls immediately — before the enqueue round-trip and
	// the (possibly slow) noun-phrase LLM step — so the button can't be re-fired.
	// The next /generate-status poll keeps it busy server-side (the node is marked
	// queued at enqueue); a failed enqueue re-enables it below.
	busyNodes.add(id);
	tracePanel.rerenderInfo();
	try {
		await api.regenerate(state.run, slot, model, id, { ...opts, version: genVersion });
		toast(
			`regenerating ${id}${opts.regenNounPhrase ? " (+ new noun phrase)" : ""} · ${opts.backend}…`,
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
		await api.link(state.run, slot, model, id, target, { group, version: genVersion });
		toast(`linking ${group ? "group" : id} → ${target}…`, "ok");
		pollGenerated();
	} catch (e) {
		toast(e.message, "err");
	}
}

// Split an object out of its prefab group into a standalone asset with its own
// raw mesh (the server clones the shared geometry) so it no longer tracks the
// group; the user can then regenerate it on its own. Fast local re-derivation,
// no backend call — mirrors linkNode (no busy lock).
async function unlinkNode(id) {
	if (!state.view || state.view.branch) return;
	const { slot, model } = state.view;
	try {
		await api.unlink(state.run, slot, model, id, { version: genVersion });
		toast(`unlinking ${id}…`, "ok");
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
			version: genVersion,
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
		await api.unsymmetrize(state.run, slot, model, id, { propagate: true, version: genVersion });
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
			version: genVersion,
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
		await api.glassify(state.run, slot, model, id, { version: genVersion });
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
		await api.resetMesh(state.run, slot, model, id, { version: genVersion });
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
const INVESTIGATOR_WIDTH_KEY = "starshot.investigatorWidth";
const INVESTIGATOR_MIN = 340;
let investigatorOpen = false;
let inv = null; // the shared investigator chat instance (see investigator.js)

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
	let dockRight = 0;
	let bodyLeft = 0;
	resizer.addEventListener("pointerdown", (ev) => {
		dragging = true;
		resizer.classList.add("dragging");
		resizer.setPointerCapture(ev.pointerId);
		ev.preventDefault();
		// The dock's right edge is pinned during the drag (its right neighbour —
		// the investigator column when open, else the body edge — is fixed-width),
		// so capture it once and size the dock from it. This keeps the resizer
		// correct whether or not the investigator column sits to the dock's right,
		// where the body's right edge is no longer the dock's.
		dockRight = dock.getBoundingClientRect().right;
		bodyLeft = body.getBoundingClientRect().left;
	});
	resizer.addEventListener("pointermove", (ev) => {
		if (!dragging) return;
		const max = Math.max(OBSDOCK_MIN, dockRight - bodyLeft - CANVAS_MIN);
		const width = Math.max(OBSDOCK_MIN, Math.min(dockRight - ev.clientX, max));
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

// The per-slot investigator: a docked, toggleable chat column at the right edge
// of the overlay (right of the observability dock). The header button flips it;
// investigator.js renders the chat + owns its per-cell threads, and openCell
// (re)binds it to the viewed cell.
function setupInvestigator() {
	inv = createInvestigator(document.getElementById("investigator"), {
		onClose: () => setInvestigator(false),
		loadCallBytes,
	});
	document
		.getElementById("overlay-investigate")
		.addEventListener("click", () => setInvestigator(!investigatorOpen));
	initInvestigatorResizer();
}

function setInvestigator(open) {
	investigatorOpen = open;
	document.getElementById("investigator").classList.toggle("open", open);
	document.getElementById("investigator-resizer").style.display = open ? "" : "none";
	document.getElementById("overlay-investigate").classList.toggle("on", open);
	if (open) inv.onShown();
}

// Drag the divider left of the investigator column to set its width; the canvas
// (flex) absorbs the change. Clamped so the obs dock + a minimum canvas survive,
// and persisted like the obs dock's width.
function initInvestigatorResizer() {
	const resizer = document.getElementById("investigator-resizer");
	const inv = document.getElementById("investigator");
	const dock = document.getElementById("obsdock");
	const body = document.getElementById("overlay-body");
	let saved = NaN;
	try {
		saved = Number(localStorage.getItem(INVESTIGATOR_WIDTH_KEY));
	} catch {
		/* private mode */
	}
	if (saved >= INVESTIGATOR_MIN) inv.style.width = `${saved}px`;

	let dragging = false;
	let invRight = 0;
	let bodyLeft = 0;
	let reserved = 0; // obs dock + the two resizers — kept clear for the canvas min
	resizer.addEventListener("pointerdown", (ev) => {
		dragging = true;
		resizer.classList.add("dragging");
		resizer.setPointerCapture(ev.pointerId);
		ev.preventDefault();
		invRight = inv.getBoundingClientRect().right; // pinned to the body's right edge
		bodyLeft = body.getBoundingClientRect().left;
		reserved = dock.getBoundingClientRect().width + 14;
	});
	resizer.addEventListener("pointermove", (ev) => {
		if (!dragging) return;
		const max = Math.max(INVESTIGATOR_MIN, invRight - bodyLeft - reserved - CANVAS_MIN);
		const width = Math.max(INVESTIGATOR_MIN, Math.min(invRight - ev.clientX, max));
		inv.style.width = `${Math.round(width)}px`;
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
				INVESTIGATOR_WIDTH_KEY,
				String(parseInt(inv.style.width, 10) || INVESTIGATOR_MIN),
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
		// Box-less ancestor: viewer.select isn't called, so end any projection here.
		viewer.clearMapProjection();
		dock.markSelected(id, { scroll: true });
		tracePanel.show(viewObs(), id);
	}
}

// A node to focus once the incoming cell has loaded it — set when the api log
// opens a cell via `emit("open-cell", { focus })`. Retried as history folds /
// events stream, then cleared, so a log→scene jump lands on the exact place.
let pendingFocus = null;
function tryPendingFocus() {
	if (!pendingFocus) return;
	const id = pendingFocus;
	if (viewer.hasBbox(id) || obs.model.nodes.has(id)) {
		pendingFocus = null;
		focusNode(id);
	}
}

// "why?" on a call row (dock OR the left trace panel) → open the investigator
// with that step attached as deep context. The per-step chat is now just the
// shared investigator focused on one call; it reads the live view (which the
// investigator is already bound to via openCell), so it works for source AND
// branch cells alike.
function investigateCall(call) {
	if (!state.view || call?.index == null) return;
	setInvestigator(true);
	inv.attachStep(call.index);
}

// Slim history/live streams omit each call's heavy prompt/output/reasoning/
// variables bytes (so a multi-GB scene's tree loads); the dock, trace panel, and
// investigator fetch them per-call on demand through this. It hydrates the SHARED
// call object in place, so a call is fetched at most once no matter which panel
// opens it first, and is a no-op once hydrated (or when the call already carries
// bytes — compare/legacy full events). Reads the CURRENT cell's context.
async function loadCallBytes(call) {
	if (!call || call.index == null || call.system !== undefined) return call;
	const { slot, model, branch } = state.view ?? {};
	if (!branch && (!slot || !model)) return call;
	try {
		const bytes = branch
			? await api.branchEventBytes(branch, call.index)
			: await api.eventBytes(state.run, slot, model, call.index);
		if (bytes && typeof bytes === "object") Object.assign(call, bytes);
	} catch {
		/* leave the call slim — the panels show a placeholder */
	}
	return call;
}

// "log ↗" on a call row → open the api log to this exact call. The flight
// ledger keys rows by scene (`slot`): a source cell is `<run>/<slot>/<model>`,
// a branch is `<run>/_branches/<bid>`. The call is matched there by its
// generation_id (OpenRouter) or t_request (direct), the same join the log uses.
function openLogForCall(call) {
	if (!state.view || !call) return;
	const { slot, model, branch } = state.view;
	const scene = branch
		? `${state.run}/_branches/${branch}`
		: `${state.run}/${slot}/${model}`;
	emit("open-flight", {
		run: state.run,
		scene,
		generation_id: call.generation_id ?? null,
		t_request: call.t_request ?? null,
	});
}

function scheduleTreeRender() {
	if (renderQueued) return;
	renderQueued = true;
	requestAnimationFrame(() => {
		renderQueued = false;
		dock.renderTree(viewObs(), { streamed: true });
		// Fold any newly-streamed calls into the open emit trace too (no-ops
		// when nothing the panel shows changed, or while the user is reading it).
		tracePanel.refresh(viewObs(), { streamed: true });
		if (state.view?.branch && state.lab.simStep)
			dock.renderPinned(viewObs());
		// New zones may have streamed in — refresh the legend so a new depth's
		// swatch appears (no-ops when the depth set is unchanged).
		refreshZoneLayers();
		tryPendingFocus(); // node from a log→scene jump may have just streamed in
	});
}

export async function openCell({
	slot,
	model,
	branch = false,
	forceLive = false,
	focus = null,
	fromLog = false,
}) {
	const seq = ++openSeq;
	const run = state.run;
	pendingFocus = focus || null;
	// Keep the camera when this re-renders the SAME cell while the overlay is
	// already open — the branch selector swapping between the source run and its
	// per-LLM simulation lineages, or a revert/step reload. Swapping like that
	// shouldn't pull the user off the part of the scene they're inspecting. A
	// fresh open (overlay closed) or a different cell still frames the scene anew.
	const sameCell =
		!!state.view && state.view.slot === slot && state.view.model === model;
	const keepCamera = overlayEl.classList.contains("open") && sameCell;
	// Opening from the log arms the "← logs" return; opening a *different* cell
	// any other way clears it. Same-cell reloads (revert/branch) keep it as-is.
	if (fromLog) cameFromLog = true;
	else if (!sameCell) cameFromLog = false;
	stream?.close();
	stream = null;
	// Replay owns the scene while it is active and its scrub position is tied to
	// one log, so any reopen — cell swap, revert reload, or its own exit — drops
	// it. `stop` deliberately doesn't fire `onExit`; that would recurse in here.
	replay?.stop();
	btnReplay.classList.add("off");
	// A fresh cell open always starts on the library build; the generated view
	// is opt-in per open via the asset toggle.
	assetMode = "library";
	stopGenPoll();
	clearGeneratedState();
	state.view = { slot, model, branch };
	overlayEl.classList.add("open");
	viewer.setActive(true);
	viewer.clear({ keepCamera });
	obs = createObsModel();
	// (Re)bind the investigator to this cell + its fresh obs model; preload its
	// base grounding when the panel is already open (no focus steal on reloads).
	inv.setContext({ run, slot, model, branch }, obs, { fetch: investigatorOpen });
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
			? await api.branchEventsHistorySlim(run, branch)
			: await api.eventsHistorySlim(run, slot, model);
	} catch {
		/* never-started cell */
	}
	if (seq !== openSeq) return;
	// Replay walks this same array rather than re-reading the log.
	replay.load(history);
	for (const event of history) obs.feed(event);
	// The scene projection painted bboxes before this history folded, so objects
	// were colored the default green; repaint now that each node's emitting step
	// is known. (Streamed bboxes color correctly on paint — their decompose call
	// always precedes the bbox event.)
	viewer.recolorAll();
	dock.renderTree(viewObs());
	// Depths are known now that history folded — (re)build the layer legend.
	refreshZoneLayers();
	if (branch && state.lab.simStep) dock.renderPinned(viewObs());
	renderHeader(); // error message comes from the just-loaded log
	tryPendingFocus(); // a log→scene jump lands on its node once history folds

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
			// The tree keeps streaming, but the 3D scene belongs to replay while
			// it is active — otherwise a running cell drops meshes in ahead of the
			// scrub and the build order on screen stops being the real one.
			if (!replay?.isActive()) dispatchSceneEvent(viewer, event);
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
	// Measured execution time, pauses excluded. `spans` > 1 means the cell was
	// interrupted and resumed, so the total is the work across those stretches.
	const activeS = cellInfo?.timing?.active_s;
	if (activeS) {
		const spans = cellInfo.timing.spans ?? 1;
		statusText += ` · ran ${fmtDurationMs(activeS * 1000)}`;
		if (spans > 1) statusText += ` (resumed ${spans - 1}×)`;
	}
	if (status === "error") {
		// Put the failure reason where the eye lands first; the log strip below
		// has the full trail.
		const err = obs.lastError?.();
		if (err) statusText += ` — ${err.text}`;
	}
	statusEl.textContent = statusText;
	statusEl.title = statusText;
	statusEl.classList.toggle("is-error", status === "error");
	if (toLogsBtn) toLogsBtn.style.display = cameFromLog ? "" : "none";

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

	// Spend-cap override: offered on any started source cell (not only capped
	// ones) so the ceiling can be changed anytime. A capped cell shows it as the
	// primary "set cap & continue" — the only way past the cap, since a plain
	// resume/retry is refused server-side while capped (so the action button is
	// hidden then); otherwise it's a secondary "set cap" beside that action.
	let capBtn = document.getElementById("overlay-cap-override");
	if (!branch && summary?.cap && status !== "idle" && status !== "done") {
		if (!capBtn) {
			capBtn = el("button", { id: "overlay-cap-override" });
			actionBtn.before(capBtn);
		}
		const atCap = status === "capped";
		capBtn.classList.toggle("primary", atCap);
		capBtn.textContent = atCap ? "set cap & continue" : "set cap";
		capBtn.title = atCap
			? "set this cell's spend cap above its spend (0 = no cap) and resume the run"
			: "set this cell's spend cap (0 = no cap)";
		capBtn.onclick = onCapOverride;
	} else {
		capBtn?.remove();
	}

	// Graceful pause: a live source cell can "finish & pause" — stop issuing new
	// LLM calls but let the in-flight one finish + commit before pausing, vs the
	// action button's immediate hard pause. Hidden on branches and non-live cells.
	let softBtn = document.getElementById("overlay-soft-pause");
	if (!branch && live) {
		if (!softBtn) {
			softBtn = el("button", { id: "overlay-soft-pause" });
			actionBtn.before(softBtn);
		}
		softBtn.textContent = "finish & pause";
		softBtn.title = "stop issuing new LLM calls, let the in-flight call finish and commit, then pause — resume won't re-run or re-bill it";
		softBtn.onclick = onSoftPause;
	} else {
		softBtn?.remove();
	}

	// One-call-at-a-time stepping controls. Shown whenever the cell is gated:
	// a live branch always is; a source cell whenever it's in step mode (so
	// the button is there even when paused with no live gate — incl. after a
	// restart). `done` cells have nothing left to step.
	// A capped cell can't step either (stepping is refused until the cap is
	// overridden), so only its override button shows — not a dead-end "step".
	const stepped = branch || !!summary?.stepped;
	const canStep = stepped && status !== "done" && status !== "capped";
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
					() => state.steps,
					(until, before) => stepCurrent(false, until, before),
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

async function stepCurrent(auto, until = null, before = false) {
	const { slot, model, branch } = state.view ?? {};
	if (!slot) return;
	try {
		if (branch) {
			await api.branchStep(branch, { auto });
		} else {
			const r = await api.cellStep(state.run, slot, model, {
				auto,
				until,
				untilBefore: before,
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

// Graceful pause (source cells): stop new LLM calls, let the in-flight one finish
// and commit, then pause. The cell stays "running" until it drains, then flips to
// paused on the next poll — so no reload/stream re-wire here, just a heads-up.
async function onSoftPause() {
	const { slot, model, branch } = state.view ?? {};
	if (!slot || branch) return;
	try {
		await api.pauseSoft(state.run, slot, model);
		toast("finishing the in-flight call, then pausing…", "ok");
		emit("poll-now");
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

// Set a cell's spend cap to any value — available anytime, not just when capped.
// When the new ceiling clears a capped cell's spend the server resumes it, and
// only then do we reopen streaming the live run (forceLive — the polled summary
// still says capped until the next poll lands).
function onCapOverride() {
	const { slot, model, branch } = state.view ?? {};
	if (!slot || branch) return;
	const cap = currentSummary()?.cap;
	promptCapValue(`spend cap · ${slot} · ${model}`, { current: cap?.limit ?? 0 }, async (v) => {
		const r = await api.capOverride(state.run, slot, model, v);
		emit("poll-now");
		if (
			r.resumed &&
			state.view &&
			state.view.slot === slot &&
			state.view.model === model &&
			!state.view.branch
		)
			openCell({ slot, model, branch: false, forceLive: true });
	});
}

export function closeOverlay() {
	openSeq += 1;
	stream?.close();
	stream = null;
	stopGenPoll();
	replay?.stop();
	btnReplay.classList.add("off");
	assetMode = "library";
	state.view = null;
	cameFromLog = false;
	if (toLogsBtn) toLogsBtn.style.display = "none";
	overlayEl.classList.remove("open");
	viewer.setActive(false);
	dock.setPinStep(null);
	tracePanel.reset();
	setInvestigator(false);
	inv.reset();
}
