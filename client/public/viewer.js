import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";
import { KTX2Loader } from "three/addons/loaders/KTX2Loader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const SERVER_URL = document
	.querySelector('meta[name="server-url"]')
	.getAttribute("content");

const SLOT_STORAGE_KEY = "starshot.selectedSlot";
const MODEL_STORAGE_KEY = "starshot.selectedModel";
const VERSION_STORAGE_KEY = "starshot.selectedVersion";
const BBOX_VISIBLE_STORAGE_KEY = "starshot.bboxesVisible";
const FRAMES_VISIBLE_STORAGE_KEY = "starshot.framesVisible";
const MESHES_VISIBLE_STORAGE_KEY = "starshot.meshesVisible";
const SELECT_MODE_STORAGE_KEY = "starshot.selectMode";
const SOLID_FILL_STORAGE_KEY = "starshot.solidFill";
const GRID_VISIBLE_STORAGE_KEY = "starshot.gridVisible";
const TABS_STORAGE_KEY = "starshot.openTabs";
const ASSET_MODE_STORAGE_KEY = "starshot.assetMode";
const GEN_OPTIMIZED_STORAGE_KEY = "starshot.genOptimized";
const REGEN_SOURCE_STORAGE_KEY = "starshot.regenSource";

const statusEl = document.getElementById("status");
const costTrackerEl = document.getElementById("cost-tracker");
const costPillEl = document.getElementById("cost-pill");
const costPillSummaryEl = costPillEl.querySelector(".cost-pill-summary");
const costDropdownEl = document.getElementById("cost-dropdown");
const logEl = document.getElementById("log");
const logBodyEl = document.getElementById("log-body");
const logToggleEl = document.getElementById("log-toggle");
const logGenerateEl = document.getElementById("log-generate");
const slotBarEl = document.getElementById("slot-bar");
const slotBarToggleEl = document.getElementById("slot-bar-toggle");
const controlsBarEl = document.getElementById("controls-bar");
const controlsBarToggleEl = document.getElementById("controls-bar-toggle");
const resetEl = document.getElementById("slot-reset");
const resumeEl = document.getElementById("slot-resume");
const modelPickerEl = document.getElementById("model-picker");
const runPickerEl = document.getElementById("run-picker");
const runTabsBarEl = document.getElementById("run-tabs-bar");
const runTabAddEl = document.getElementById("run-tab-add");
const runNewEl = document.getElementById("run-new");
const saveAllEl = document.getElementById("save-all");
const snapshotAllEl = document.getElementById("snapshot-all");
const resumeAllEl = document.getElementById("resume-all");
const stopAllEl = document.getElementById("stop-all");
const resetAllEl = document.getElementById("reset-all");
const startCellsEl = document.getElementById("start-cells");
const startModalEl = document.getElementById("start-modal");
const startModalCloseEl = document.getElementById("start-modal-close");
const startVersionsEl = document.getElementById("start-versions");
const startModelsEl = document.getElementById("start-models");
const startSlotsEl = document.getElementById("start-slots");
const startModalCountEl = document.getElementById("start-modal-count");
const startModalGoEl = document.getElementById("start-modal-go");
const versionBarEl = document.getElementById("version-bar");
const versionLaunchAllEl = document.getElementById("version-launch-all");
const versionArchiveAllEl = document.getElementById("version-archive-all");
const bboxToggleEl = document.getElementById("bbox-toggle");
const framesToggleEl = document.getElementById("frames-toggle");
const meshesToggleEl = document.getElementById("meshes-toggle");
const selectModeToggleEl = document.getElementById("select-mode-toggle");
const solidFillToggleEl = document.getElementById("solid-fill-toggle");
const gridToggleEl = document.getElementById("grid-toggle");
const assetModeToggleEl = document.getElementById("asset-mode-toggle");
const genOptimizeToggleEl = document.getElementById("gen-optimize-toggle");
const generateGateEl = document.getElementById("generate-gate");
const genVersionPickerEl = document.getElementById("gen-version-picker");
const genVersionNewEl = document.getElementById("gen-version-new");
const genFromImagesEl = document.getElementById("gen-from-images");
// Asset source the cell renders: "library" (pre-built meshes matched from the
// asset library, served from objects/ — the default) or "generated" (a
// from-scratch Nano-Banana + Trellis build, served from one version's
// generated/<version>/objects-generated-optimized/). A pure view switch: only
// the /meshes folder changes; the scene tree, log, and event stream all stay on
// the library build. Declared up here so slotMeshesUrl() can read it from its
// first call.
let assetMode =
	localStorage.getItem(ASSET_MODE_STORAGE_KEY) === "generated"
		? "generated"
		: "library";
// Within "generated" mode only: render the OPTIMIZED twin
// (objects-generated-optimized/ — decimated + KTX2 + Meshopt, the served
// default) or the raw, un-optimized Trellis mesh (objects-generated/<id>.glb —
// already bbox-fitted but ~100x heavier). A pure view switch over the same
// scene: flipping it re-streams the mesh bundle (and re-points every per-mesh
// url) at the other folder. Persisted across cells like assetMode; never
// touches the library build.
let genOptimized = localStorage.getItem(GEN_OPTIMIZED_STORAGE_KEY) !== "raw";
// Which generated VERSION the cell is viewing in "generated" mode, and the full
// set available. A cell holds any number of from-scratch versions (same layout,
// independently generated assets), each in generated/<version>/. `genVersion` is
// null until resolved (the gate poll adopts the latest); `genVersions` drives the
// picker. Both reset on cell switch (clearScene); not persisted across cells.
let genVersion = null;
let genVersions = [];
let generating = false; // a from-scratch build is in flight for the selected version
let _genWasRunning = false; // last poll saw a build running (to detect completion)
let _genCellKey = null; // (cell, version) the gen-gate poll is tracking
// id -> version token (the optimized GLB's mtime, from GET /generate) for each
// attached generated mesh. Lets the gate poll spot a regenerated asset (same
// id, new bytes) and reload just it with a cache-busted URL. Reset on cell
// switch (clearScene) and asset-mode reload.
const genMeshVersions = new Map();
// id -> { plane, was } for each generated mesh, from GET /generate. `plane` is
// the current symmetry plane ("none" | "xy" | "xz"); `was` is the plane it used
// to be mirrored across if it's since been un-symmetrized, else null — letting
// the detail panel tell mirrored / un-symmetrized / never-symmetrized apart.
// Drives the symmetry readout + the un-symmetrize button's state. Same lifecycle
// as genMeshVersions (per version).
const genMeshSymmetry = new Map();
// id -> bool: a per-asset OVERRIDE of the scene-wide `genOptimized` setting,
// set from the detail panel's per-asset toggle. Empty on every fresh scene load
// (so each asset follows the scene setting, as before); an entry flips just that
// mesh to the other folder (optimized twin ↔ raw Trellis mesh) while the rest of
// the scene stays put. Cleared whenever the scene re-streams (clearScene +
// reloadMeshesForAssetMode) so a freshly loaded scene always starts uniform.
const genMeshOptimized = new Map();
// ids the user clicked "regenerate" on whose new mesh hasn't landed yet — drives
// the detail button's disabled/label state. Cleared when the new version lands
// or the build goes idle.
const regeneratingIds = new Set();
// ids the user clicked "un-symmetrize" on whose reprocessed mesh hasn't landed
// yet. Same lifecycle as regeneratingIds; both gate the detail action buttons so
// only one rebuild op runs per node at a time.
const unsymmetrizingIds = new Set();
// ids the user clicked "symmetrize" on (the inverse op) whose reprocessed mesh
// hasn't landed yet. Parallel to unsymmetrizingIds with the same lifecycle — kept
// separate so the in-flight button label is correct even though `symmetry.applied`
// (and thus the polled `sym`) may update mid-op before the new mesh lands.
const symmetrizingIds = new Set();
// Sticky plane + kept-half for the symmetrize control, so the gate poll's
// re-renders don't reset a mid-selection. plane: "xy" (front/back along Z) | "xz"
// (top/bottom along Y); keepPositive: keep the +half (else the −half).
let symmetrizePlane = "xy";
let symmetrizeKeepPositive = true;
// Regenerate image source (persisted). false = "from scratch" (new Nano-Banana
// image + new mesh); true = "from image" (reuse the existing image, rebuild only
// the mesh). Read by regenerateAsset; toggled from the object detail panel.
let regenReuseImage =
	localStorage.getItem(REGEN_SOURCE_STORAGE_KEY) === "image";

// Active (run, slot, model) cell selection + the lists backing the run / model /
// version pickers. Declared here — above the queue panel, which renders during
// module init — so that first paint can read these without hitting the temporal
// dead zone. Mutated by the slot/run lifecycle functions further down.
let currentSource = null;
let currentSlotId = null;
let currentModel = null;
let currentRun = null;
let availableModels = [];
let availableRuns = []; // [{name, modified_at, has_prompt_snapshot}, ...]
let availableVersions = []; // [{id, run_name, label, status}, ...] from GET /versions
let defaultModelAlias = null;
let slotSummaries = []; // latest /slots `slots` array, for tab rendering
let slotNeedsResume = false;

const exportGlbEl = document.getElementById("export-glb");
const replayGifEl = document.getElementById("replay-gif");
const replayModalEl = document.getElementById("replay-modal");
const replayCloseEl = document.getElementById("replay-close");
const replayEventCountEl = document.getElementById("replay-event-count");
const replayDurationEstEl = document.getElementById("replay-duration-est");
const replayIntervalEl = document.getElementById("replay-interval");
const replayIntervalValEl = document.getElementById("replay-interval-val");
const replayResolutionEl = document.getElementById("replay-resolution");
const replayResolutionValEl = document.getElementById("replay-resolution-val");
const replayProgressBarEl = document.getElementById("replay-progress-bar");
const replayStatusEl = document.getElementById("replay-status");
const replayPreviewEl = document.getElementById("replay-preview");
const replayRenderEl = document.getElementById("replay-render");
const replayDownloadEl = document.getElementById("replay-download");
const replayStageEl = document.getElementById("replay-stage");
const replayPreviewCanvasEl = document.getElementById("replay-preview-canvas");
const replayResultImgEl = document.getElementById("replay-result-img");
const replayPreviewCtx = replayPreviewCanvasEl.getContext("2d");

// URL of the last-encoded gif blob, kept so we can revoke when the modal
// closes or a fresh render starts.
let lastGifUrl = null;
let lastGifBlob = null;

function showReplayPlaceholder() {
	replayStageEl.classList.remove("show-result");
	replayStageEl.classList.add("empty");
	replayPreviewCtx.clearRect(
		0,
		0,
		replayPreviewCanvasEl.width,
		replayPreviewCanvasEl.height,
	);
}

function drawReplayFrame(srcCanvas) {
	// Match the preview canvas's backing buffer to the source so frames keep
	// their native aspect; CSS object-fit handles the visual letterboxing.
	if (
		replayPreviewCanvasEl.width !== srcCanvas.width ||
		replayPreviewCanvasEl.height !== srcCanvas.height
	) {
		replayPreviewCanvasEl.width = srcCanvas.width;
		replayPreviewCanvasEl.height = srcCanvas.height;
	}
	replayPreviewCtx.drawImage(srcCanvas, 0, 0);
	replayStageEl.classList.remove("empty");
	replayStageEl.classList.remove("show-result");
}

function showReplayGifResult(blob) {
	if (lastGifUrl) URL.revokeObjectURL(lastGifUrl);
	lastGifBlob = blob;
	lastGifUrl = URL.createObjectURL(blob);
	replayResultImgEl.src = lastGifUrl;
	replayStageEl.classList.add("show-result");
	replayStageEl.classList.remove("empty");
	replayDownloadEl.disabled = false;
}
const assetsEl = document.getElementById("assets");
const assetsBodyEl = document.getElementById("assets-body");
const assetsCountEl = document.getElementById("assets-count");
const assetsMissingEl = document.getElementById("assets-missing");
const assetsHeaderEl = document.getElementById("assets-header");
const assetsToggleEl = document.getElementById("assets-toggle");
const trellisQueueEl = document.getElementById("trellis-queue");
const trellisQueueHeaderEl = document.getElementById("trellis-queue-header");
const trellisQueueToggleEl = document.getElementById("trellis-queue-toggle");
const trellisQueueCountsEl = document.getElementById("trellis-queue-counts");
// The body is rendered dynamically — one section per concurrency pool — from
// the `pools` the server reports, so its inner markup lives in JS, not HTML.
const trellisQueueBodyEl = document.getElementById("trellis-queue-body");
// Fallback pool sections used only if a snapshot predates the per-pool `pools`
// shape (e.g. an old server). Live caps/labels come from `/trellis/queue`.
const FALLBACK_QUEUE_POOLS = [
	{ id: "modal", label: "Trellis · Modal", cap: 100 },
	{ id: "hunyuan-tencent", label: "Hunyuan 3.1 · Tencent", cap: 1 },
];
const treeEl = document.getElementById("tree");
const treeBodyEl = document.getElementById("tree-body");
const treeDetailEl = document.getElementById("tree-detail");
const treeHeaderEl = document.getElementById("tree-header");
const treeToggleEl = document.getElementById("tree-toggle");
const treeActiveEl = document.getElementById("tree-active");
const treeExpandEl = document.getElementById("tree-expand");
const treeSearchEl = document.getElementById("tree-search");
const treeSearchClearEl = document.getElementById("tree-search-clear");
const treeSearchCountEl = document.getElementById("tree-search-count");
const treeModalEl = document.getElementById("tree-modal");
const treeModalBodyEl = document.getElementById("tree-modal-body");
const treeModalCloseEl = document.getElementById("tree-modal-close");
const treeModalSearchEl = document.getElementById("tree-modal-search");
const treeFlowEl = document.getElementById("tree-flow");
const flowModalEl = document.getElementById("flow-modal");
const flowViewportEl = document.getElementById("flow-viewport");
const flowStageEl = document.getElementById("flow-stage");
const flowEdgesEl = document.getElementById("flow-edges");
const flowNodesEl = document.getElementById("flow-nodes");
const flowEmptyEl = document.getElementById("flow-empty");
const flowSearchEl = document.getElementById("flow-search");
const flowFitEl = document.getElementById("flow-fit");
const flowZoomInEl = document.getElementById("flow-zoom-in");
const flowZoomOutEl = document.getElementById("flow-zoom-out");
const flowCloseEl = document.getElementById("flow-close");
const flowPauseEl = document.getElementById("flow-pause");
const sandboxPanelEl = document.getElementById("sandbox-panel");
const sandboxStepPillEl = document.getElementById("sandbox-step-pill");
const sandboxStepMetaEl = document.getElementById("sandbox-step-meta");
const sandboxPosEl = document.getElementById("sandbox-pos");
const sandboxSystemEl = document.getElementById("sandbox-system");
const sandboxUserEl = document.getElementById("sandbox-user");
const sandboxSystemFieldEl = document.getElementById("sandbox-system-field");
const sandboxUserFieldEl = document.getElementById("sandbox-user-field");
const sandboxOutputWrapEl = document.getElementById("sandbox-output-wrap");
const sandboxOutputEl = document.getElementById("sandbox-output");
const sandboxRenderNoteEl = document.getElementById("sandbox-render-note");
const sandboxReasoningEl = document.getElementById("sandbox-reasoning");
const sandboxReasoningBodyEl = document.getElementById(
	"sandbox-reasoning-body",
);
const sandboxTestEl = document.getElementById("sandbox-test");
const sandboxResetEl = document.getElementById("sandbox-reset");
const sandboxPrevEl = document.getElementById("sandbox-prev");
const sandboxNextEl = document.getElementById("sandbox-next");
const sandboxSimulateEl = document.getElementById("sandbox-simulate");
const sandboxRunStepEl = document.getElementById("sandbox-runstep");
const sandboxRerunEl = document.getElementById("sandbox-rerun");
const sandboxRunRestEl = document.getElementById("sandbox-runrest");
const sandboxBackEl = document.getElementById("sandbox-back");
const sandboxBreakoutEl = document.getElementById("sandbox-breakout");
const sandboxCloseEl = document.getElementById("sandbox-close");
const sandboxStatusEl = document.getElementById("sandbox-status");
const sandboxExpandEl = document.getElementById("sandbox-expand");
const sandboxCopyEl = document.getElementById("sandbox-copy");

// Execution-flow graph layout constants + state. Declared up here (before
// `animate()` first runs) because the render loop reads `flowModalOpen` /
// `_flowRenderPending` / `_flowLastRender` every frame — leaving them in a
// `let` further down would put them in the temporal dead zone on the first
// synchronous `animate()` call. The graph's functions live lower in the file.
const FLOW = {
	INDENT: 32, // horizontal indent per depth level (outline)
	ROW_H: 56, // vertical slot per node row
	GUTTER: 14, // connector spine offset inside a node's left edge
	NODE_W: 210,
	NODE_H: 42, // nominal node height (edge anchors + bbox height)
	PAD: 80, // breathing room around the whole graph
	GRID: 24, // background dot pitch (matches the viewport CSS)
};
let flowModalOpen = false;
let flowPanX = 0;
let flowPanY = 0;
let flowZoom = 1;
let flowSearchQuery = "";
let _flowRenderPending = false;
let _flowLastRender = 0;
let _flowLastWidth = 0;
let _flowLastHeight = 0;
let _flowPanning = false;
let _flowStartX = 0;
let _flowStartY = 0;
let _flowStartPanX = 0;
let _flowStartPanY = 0;
// Last-rendered exec graph + node positions, cached so sidebar clicks and the
// locate search can center the canvas without rebuilding the layout.
let _flowGraph = null;
let _flowPositions = new Map();

// --- log panel --------------------------------------------------------------

const KIND_COLOR = {
	"run.start": "#9ad4ff",
	"run.done": "#8bd17c",
	"run.error": "#ff8080",
	"run.paused": "#e09050",
	bbox: "#e0c271",
	image: "#f6a96a",
	model: "#c586d1",
	step: "#4a8fd8",
	"divider.decompose": "#e0c271",
	"generation.decompose": "#c586d1",
	"mesh.error": "#ff8080",
};

function setStatus(text, cls = "hdr") {
	statusEl.innerHTML = "";
	const p = document.createElement("p");
	p.className = `line ${cls}`;
	p.textContent = text;
	statusEl.appendChild(p);
}

function fmtValue(v) {
	if (Array.isArray(v)) return "[" + v.map(fmtValue).join(", ") + "]";
	if (typeof v === "number")
		return Number.isInteger(v) ? String(v) : v.toFixed(2);
	if (v && typeof v === "object")
		return (
			"{" +
			Object.entries(v)
				.map(([k, x]) => `${k}=${fmtValue(x)}`)
				.join(", ") +
			"}"
		);
	if (typeof v === "string") return v;
	return String(v);
}

// The log panel is a glanceable activity feed, not a payload inspector — some
// events (e.g. cache.llm) carry hundreds of KB of prompt/output text. Each
// rendered field value is clipped to a short preview; the full value still
// lives in events.jsonl and the observability modal.
const MAX_LOG_VALUE_LEN = 40;
function truncateLogValue(text) {
	return text.length > MAX_LOG_VALUE_LEN
		? text.slice(0, MAX_LOG_VALUE_LEN) + "…"
		: text;
}

// Fields whose values are LLM thinking / chain-of-thought traces. Stripped at
// the display layer so the log panel stays focused on pipeline signal; the
// underlying events.jsonl still contains them for debugging.
const HIDDEN_LOG_FIELDS = new Set(["reasoning", "thinking"]);

// Build a single log line (<p>) for an event. Kept separate from insertion so
// the on-demand "generate log" backfill can batch every buffered event into a
// fragment instead of thrashing layout one append at a time.
function buildLogLine(event) {
	const { kind, index, ...rest } = event;
	const fields = Object.fromEntries(
		Object.entries(rest).filter(([k]) => !HIDDEN_LOG_FIELDS.has(k)),
	);
	const p = document.createElement("p");
	p.className = "line";
	if (typeof index === "number") p.dataset.eventIndex = String(index);

	if (typeof index === "number") {
		const btn = document.createElement("button");
		btn.className = "rewind";
		btn.type = "button";
		btn.textContent = "↶ rewind";
		btn.title = `Rewind to event ${index} (discards this event and everything after)`;
		btn.addEventListener("click", (ev) => {
			ev.stopPropagation();
			rewindTo(index);
		});
		p.appendChild(btn);
	}

	if (typeof index === "number") {
		const idx = document.createElement("span");
		idx.className = "idx";
		idx.textContent = `#${index}`;
		p.appendChild(idx);
	}

	const tag = document.createElement("span");
	tag.className = "step";
	tag.textContent = `[${kind}]`;
	tag.style.color = KIND_COLOR[kind] ?? "#8bd17c";
	p.appendChild(tag);

	const entries = Object.entries(fields);
	if (entries.length === 0) {
		p.appendChild(document.createTextNode(""));
	} else {
		for (const [k, v] of entries) {
			const kv = document.createElement("span");
			kv.className = "kv";
			const label = document.createElement("span");
			label.className = "k";
			label.textContent = ` ${k}=`;
			kv.appendChild(label);
			kv.appendChild(
				document.createTextNode(truncateLogValue(fmtValue(v))),
			);
			p.appendChild(kv);
		}
	}
	return p;
}

// The log panel is gated: a <p> per event is too expensive to build for every
// event on large runs, so events accumulate in `recordedEvents` (cheap) and are
// only materialized into the DOM once the user clicks "generate log". Until
// then appendEvent is a no-op; the tree, assets, and observability still update.
let logEnabled = false;

function appendEvent(event) {
	if (!logEnabled) return;
	const atBottom =
		logBodyEl.scrollHeight - logBodyEl.scrollTop - logBodyEl.clientHeight <
		1;
	logBodyEl.appendChild(buildLogLine(event));
	if (atBottom) logBodyEl.scrollTop = logBodyEl.scrollHeight;
}

function clearLog() {
	for (const child of Array.from(logBodyEl.querySelectorAll(".line"))) {
		child.remove();
	}
	logEnabled = false;
	logGenerateEl.style.display = "";
}

// Materialize every buffered event into the log on demand, then keep appending
// live. Batched through a fragment so even a multi-thousand-event run renders
// in a single layout pass.
function generateLog() {
	if (logEnabled) return;
	logEnabled = true;
	logGenerateEl.style.display = "none";
	const frag = document.createDocumentFragment();
	for (const event of recordedEvents) frag.appendChild(buildLogLine(event));
	logBodyEl.appendChild(frag);
	logBodyEl.scrollTop = logBodyEl.scrollHeight;
}

// --- collapsible bars --------------------------------------------------------

function toggleCollapse(el, toggleEl) {
	const collapsed = el.classList.toggle("collapsed");
	toggleEl.textContent = collapsed ? "▸" : "▾";
}

slotBarToggleEl.addEventListener("click", () =>
	toggleCollapse(slotBarEl, slotBarToggleEl),
);
slotBarEl
	.querySelector(".bar-label")
	.addEventListener("click", () =>
		toggleCollapse(slotBarEl, slotBarToggleEl),
	);
controlsBarEl
	.querySelector(".ctrl-header")
	.addEventListener("click", () =>
		toggleCollapse(controlsBarEl, controlsBarToggleEl),
	);
document
	.getElementById("log-header")
	.addEventListener("click", () => toggleCollapse(logEl, logToggleEl));
logGenerateEl.addEventListener("click", (e) => {
	e.stopPropagation(); // don't also toggle the header's collapse
	generateLog();
});

// id -> error message for every mesh that errored during the current run.
// Drives the per-node "error" phase in the tree and the aggregated count
// shown on run.done so silent partial failures don't slip past as success.
const meshErrors = new Map();

function clearMeshErrors() {
	meshErrors.clear();
	retryingIds.clear();
	runFinished = false;
}

// ids that the user clicked "retry" on whose follow-up image/model/mesh.error
// hasn't landed yet. Drives button disabled state + label so the user can't
// double-fire a retry mid-flight. Cleared on slot switch / reset / rewind
// (same lifecycle as meshErrors). The set is also pruned when a retry-targeted
// event arrives (mesh.error → retry available again; image/model → success).
const retryingIds = new Set();

// Flips to true on the live `run.done` for this slot. Lets post-run mesh
// updates (retry success / retry failure) refresh the top status line so a
// stale "run complete — N meshes failed" doesn't outlive the N it counted.
// Reset on slot switch / reset / rewind / resume alongside the rest of the
// per-slot state.
let runFinished = false;

function refreshPostRunStatus() {
	if (!runFinished) return;
	const inFlight = retryingIds.size;
	if (inFlight > 0) {
		const ids = [...retryingIds];
		const head = ids.slice(0, 3).join(", ");
		const suffix = ids.length > 3 ? `, +${ids.length - 3}` : "";
		setStatus(
			`retrying ${inFlight} mesh${inFlight === 1 ? "" : "es"}: ${head}${suffix}`,
		);
		return;
	}
	if (meshErrors.size > 0) showRunCompleteWithErrors();
	else setStatus("run complete");
}

async function retryMesh(id) {
	if (currentSlotId === null) return;
	if (retryingIds.has(id)) return;
	retryingIds.add(id);
	// Local optimistic state: clear the error so the asset/tree both reflect
	// an in-flight retry. The server-side mesh.retry event will arrive too,
	// but doing it locally first removes the visible-flicker between click
	// and the SSE round-trip.
	meshErrors.delete(id);
	upsertAsset(id, { status: "pending", errorMessage: null });
	if (treeNodes.has(id)) treeSetPhase(id, "generating_mesh");

	// The slot's SSE may have closed itself on the prior run's run.done.
	// Re-subscribe (without resetting highestEventIndex) so the snapshot
	// is deduped and the retry's new events flow through the live queue.
	if (currentSource === null) {
		subscribe(slotEventsUrl(currentSlotId, currentModel));
	}

	try {
		const res = await fetch(
			new URL(
				`/slots/${encodeURIComponent(currentSlotId)}/${encodeURIComponent(currentModel)}/retry-mesh/${encodeURIComponent(id)}?run=${encodeURIComponent(currentRun)}`,
				SERVER_URL,
			),
			{ method: "POST" },
		);
		if (!res.ok) {
			const detail = await res.text();
			const msg = `retry failed: HTTP ${res.status} ${detail}`;
			retryingIds.delete(id);
			meshErrors.set(id, msg);
			treeSetPhase(id, "error");
			upsertAsset(id, { status: "error", errorMessage: msg });
		}
	} catch (e) {
		retryingIds.delete(id);
		const msg = `retry failed: ${e.message}`;
		meshErrors.set(id, msg);
		treeSetPhase(id, "error");
		upsertAsset(id, { status: "error", errorMessage: msg });
	}
}

// Render-or-update a retry button inside `container`. `status` is the asset
// status; the button is shown only for `error` (retry) or while a prior retry
// is in flight (greyed out so the user can't double-click). `cls` lets the
// caller scope the CSS (`asset-retry` vs `detail-retry`).
function syncRetryButton(container, id, status, cls) {
	let btn = container.querySelector(`.${cls}`);
	const retrying = retryingIds.has(id);
	const visible = status === "error" || retrying;
	if (!visible) {
		if (btn) btn.remove();
		return;
	}
	if (!btn) {
		btn = document.createElement("button");
		btn.type = "button";
		btn.className = cls;
		btn.addEventListener("click", (ev) => {
			ev.stopPropagation();
			retryMesh(id);
		});
		container.appendChild(btn);
	}
	btn.classList.toggle("retrying", retrying);
	btn.disabled = retrying;
	btn.textContent = retrying ? "retrying…" : "retry mesh";
	btn.title = retrying
		? "Re-running banana + Trellis for this mesh"
		: "Re-run banana + Trellis for this mesh (fresh API calls)";
}

// Re-roll a GENERATED asset from scratch (Nano-Banana + Trellis + optimize),
// propagating across its prefab group: the canonical is rebuilt and every object
// sharing its mesh is re-derived (propagate=true). Fire-and-forget on the server;
// the gate poll detects each changed mesh by its bumped version token and swaps
// it in. `regeneratingIds` drives the detail button's disabled/label state.
async function regenerateAsset(id, backend = "trellis") {
	if (currentSlotId === null || currentModel === null || genVersion == null)
		return;
	if (regeneratingIds.has(id)) return;
	regeneratingIds.add(id);
	if (id === selectedBboxId) renderTreeDetail();
	const backendLabel =
		backend === "hunyuan-tencent"
			? "Hunyuan 3.1"
			: backend === "hunyuan"
				? "Hunyuan"
				: "Trellis";
	const sourceLabel = regenReuseImage
		? "reusing existing image"
		: "Nano-Banana";
	setStatus(
		`regenerating ${id} + objects sharing its mesh in v${genVersion} — ${sourceLabel} + ${backendLabel} (this can take a while)…`,
	);
	try {
		const res = await fetch(
			new URL(
				`/slots/${encodeURIComponent(currentSlotId)}/${encodeURIComponent(currentModel)}/regenerate/${encodeURIComponent(id)}?run=${encodeURIComponent(currentRun)}&version=${encodeURIComponent(genVersion)}&propagate=true&backend=${encodeURIComponent(backend)}&reuse_image=${regenReuseImage}`,
				SERVER_URL,
			),
			{ method: "POST" },
		);
		if (!res.ok) {
			const body = await res.json().catch(() => ({}));
			setStatus(
				`regenerate failed: ${body.detail ?? `HTTP ${res.status}`}`,
				"err",
			);
			regeneratingIds.delete(id);
			if (id === selectedBboxId) renderTreeDetail();
			return;
		}
		// Reflect the in-flight build immediately; the gate poll confirms + swaps in
		// the new mesh once its optimize pass lands.
		generating = true;
		_genWasRunning = true;
		updateGenerateGate();
	} catch (e) {
		setStatus(`regenerate failed: ${e.message}`, "err");
		regeneratingIds.delete(id);
		if (id === selectedBboxId) renderTreeDetail();
	}
}

// Reveal a GENERATED asset's full, un-mirrored mesh. The server reprocesses the
// existing raw mesh with the symmetry mirror turned off — no Nano-Banana, no mesh
// backend, so it's effectively instant — and propagates across the prefab group
// like regenerate. The gate poll detects the changed mesh by its bumped version
// token and swaps it in; `unsymmetrizingIds` drives the button's disabled/label.
async function unsymmetrizeAsset(id) {
	if (currentSlotId === null || currentModel === null || genVersion == null)
		return;
	if (regeneratingIds.has(id) || unsymmetrizingIds.has(id)) return;
	unsymmetrizingIds.add(id);
	if (id === selectedBboxId) renderTreeDetail();
	setStatus(
		`un-symmetrizing ${id} + objects sharing its mesh in v${genVersion} — reprocessing the existing mesh (no AI)…`,
	);
	try {
		const res = await fetch(
			new URL(
				`/slots/${encodeURIComponent(currentSlotId)}/${encodeURIComponent(currentModel)}/unsymmetrize/${encodeURIComponent(id)}?run=${encodeURIComponent(currentRun)}&version=${encodeURIComponent(genVersion)}&propagate=true`,
				SERVER_URL,
			),
			{ method: "POST" },
		);
		if (!res.ok) {
			const body = await res.json().catch(() => ({}));
			setStatus(
				`un-symmetrize failed: ${body.detail ?? `HTTP ${res.status}`}`,
				"err",
			);
			unsymmetrizingIds.delete(id);
			if (id === selectedBboxId) renderTreeDetail();
			return;
		}
		// Reflect the in-flight reprocess immediately; the gate poll confirms + swaps
		// the un-mirrored mesh in once its optimize pass lands.
		generating = true;
		_genWasRunning = true;
		updateGenerateGate();
	} catch (e) {
		setStatus(`un-symmetrize failed: ${e.message}`, "err");
		unsymmetrizingIds.delete(id);
		if (id === selectedBboxId) renderTreeDetail();
	}
}

// Mirror a GENERATED asset across `plane` ("xy"|"xz"), keeping `keepPositive`'s
// half — the inverse of unsymmetrizeAsset. The plane + direction are passed
// straight to the server, which reprocesses the existing raw mesh (no AI, no log
// lookup, no symmetry LLM call) and propagates across the prefab group. The gate
// poll detects the changed mesh by its bumped version token and swaps it in;
// `symmetrizingIds` drives the button's in-flight label.
async function symmetrizeAsset(id, plane, keepPositive) {
	if (currentSlotId === null || currentModel === null || genVersion == null)
		return;
	if (
		regeneratingIds.has(id) ||
		unsymmetrizingIds.has(id) ||
		symmetrizingIds.has(id)
	)
		return;
	symmetrizingIds.add(id);
	if (id === selectedBboxId) renderTreeDetail();
	setStatus(
		`symmetrizing ${id} + objects sharing its mesh in v${genVersion} across the ${plane.toUpperCase()} plane — reprocessing the existing mesh (no AI)…`,
	);
	try {
		const res = await fetch(
			new URL(
				`/slots/${encodeURIComponent(currentSlotId)}/${encodeURIComponent(currentModel)}/symmetrize/${encodeURIComponent(id)}?run=${encodeURIComponent(currentRun)}&version=${encodeURIComponent(genVersion)}&plane=${encodeURIComponent(plane)}&keep_positive=${keepPositive}&propagate=true`,
				SERVER_URL,
			),
			{ method: "POST" },
		);
		if (!res.ok) {
			const body = await res.json().catch(() => ({}));
			setStatus(
				`symmetrize failed: ${body.detail ?? `HTTP ${res.status}`}`,
				"err",
			);
			symmetrizingIds.delete(id);
			if (id === selectedBboxId) renderTreeDetail();
			return;
		}
		// Reflect the in-flight reprocess immediately; the gate poll confirms + swaps
		// the mirrored mesh in once its optimize pass lands.
		generating = true;
		_genWasRunning = true;
		updateGenerateGate();
	} catch (e) {
		setStatus(`symmetrize failed: ${e.message}`, "err");
		symmetrizingIds.delete(id);
		if (id === selectedBboxId) renderTreeDetail();
	}
}

function showRunCompleteWithErrors() {
	statusEl.innerHTML = "";
	const head = document.createElement("p");
	head.className = "line warn";
	const n = meshErrors.size;
	head.textContent = `run complete — ${n} mesh${n === 1 ? "" : "es"} failed`;
	statusEl.appendChild(head);
	const ids = [...meshErrors.keys()];
	const shown = ids.slice(0, 6);
	const detail = document.createElement("p");
	detail.className = "line warn";
	const suffix =
		ids.length > shown.length ? `, +${ids.length - shown.length} more` : "";
	detail.textContent = shown.join(", ") + suffix;
	detail.title = ids.map((id) => `${id}: ${meshErrors.get(id)}`).join("\n");
	statusEl.appendChild(detail);
}

// --- asset browser ----------------------------------------------------------

// id -> { imageUrl, prompt, modelUrl, status: "pending" | "loaded" | "error", errorMessage }
const assets = new Map();

function assetStatus(a) {
	return a.status ?? "pending";
}

// Baked artifact urls (the image/model events' `url`, the scene projection's
// image_url/mesh_url) freeze the run/slot/model the cell was generated under, so
// renaming, snapshotting, or copying a run leaves them pointing at a dir that no
// longer exists (404). Re-root the cell prefix onto the live cell so the file
// resolves wherever the events.jsonl now lives; a no-op when the names match.
function rerootArtifactUrl(url) {
	if (typeof url !== "string") return url;
	if (currentRun === null || currentSlotId === null || currentModel === null)
		return url;
	const m = url.match(/\/artifacts\/[^/]+\/[^/]+\/[^/]+\/(.+)$/);
	if (!m) return url;
	return `/artifacts/${encodeURIComponent(currentRun)}/${encodeURIComponent(
		currentSlotId,
	)}/${encodeURIComponent(currentModel)}/${m[1]}`;
}

// Thumbnail url for an asset card, resolved for the active asset mode so the
// list agrees with the detail preview. "generated": the selected version's
// render, built from the LIVE cell and gated on the mesh having attached so an
// un-built asset shows the placeholder instead of 404ing. "library": the baked
// reference image, re-rooted onto the live cell.
function assetImageUrl(id) {
	if (assetMode === "generated") {
		if (
			currentRun === null ||
			currentSlotId === null ||
			currentModel === null ||
			genVersion == null ||
			!modelsById.has(id)
		)
			return null;
		return generatedImageUrl(
			currentSlotId,
			currentModel,
			currentRun,
			genVersion,
			id,
			genMeshVersions.get(id),
		);
	}
	const a = assets.get(id);
	return a?.imageUrl ? rerootArtifactUrl(a.imageUrl) : null;
}

function upsertAsset(id, patch) {
	const cur = assets.get(id) ?? {
		imageUrl: null,
		prompt: null,
		modelUrl: null,
		status: "pending",
	};
	assets.set(id, { ...cur, ...patch });
	renderAsset(id);
	assetsCountEl.textContent = `(${assets.size})`;
	updateMissingMeshCount();
	if (id === selectedBboxId) renderTreeDetail();
}

function renderAsset(id) {
	const a = assets.get(id);
	if (!a) return;
	let card = assetsBodyEl.querySelector(`[data-id="${CSS.escape(id)}"]`);
	if (!card) {
		card = document.createElement("div");
		card.className = "asset-card";
		card.dataset.id = id;
		card.innerHTML = `
      <a class="asset-thumb-link" target="_blank" rel="noopener">
        <div class="asset-thumb placeholder">no image</div>
      </a>
      <div class="asset-body">
        <div class="asset-id"></div>
        <div class="asset-status pending">pending</div>
        <div class="asset-prompt"></div>
      </div>
    `;
		assetsBodyEl.appendChild(card);
		const promptEl = card.querySelector(".asset-prompt");
		promptEl.addEventListener("click", () => {
			promptEl.classList.toggle("expanded");
		});
	}
	card.querySelector(".asset-id").textContent = id;

	const status = assetStatus(a);
	card.className = `asset-card status-${status}`;
	const statusTag = card.querySelector(".asset-status");
	statusTag.className = `asset-status ${status}`;
	statusTag.textContent =
		status === "error" && a.errorMessage
			? `error: ${a.errorMessage}`
			: status;
	syncRetryButton(
		card.querySelector(".asset-body"),
		id,
		status,
		"asset-retry",
	);

	const link = card.querySelector(".asset-thumb-link");
	const thumb = card.querySelector(".asset-thumb");
	const imageUrl = assetImageUrl(id);
	if (imageUrl) {
		const absImg = new URL(imageUrl, SERVER_URL).toString();
		link.href = absImg;
		if (thumb.tagName !== "IMG") {
			const img = document.createElement("img");
			img.className = "asset-thumb";
			img.loading = "lazy";
			img.alt = id;
			img.src = absImg;
			thumb.replaceWith(img);
		} else if (thumb.src !== absImg) {
			thumb.src = absImg;
		}
	}

	const promptEl = card.querySelector(".asset-prompt");
	promptEl.textContent = a.prompt ?? "";
}

function clearAssets() {
	assets.clear();
	assetsBodyEl.innerHTML = "";
	assetsCountEl.textContent = "(0)";
}

// True once the server has produced a mesh for this node. A recorded
// `mesh.error` makes it false even if a stale modelUrl from a pre-error
// attempt lingers. In library mode we trust the projection/`model` event's
// modelUrl so opening a finished cell doesn't flash every mesh as missing
// while the bundle is still streaming; in generated mode that url points at
// the library mesh, not the viewed version's, so only an attached mesh counts.
function nodeHasMesh(id) {
	const a = assets.get(id);
	if (a?.status === "error") return false;
	if (modelsById.has(id)) return true;
	return assetMode === "library" && !!a?.modelUrl;
}

// Count concrete nodes (objects + frames — zones are abstract and never carry
// a mesh) whose mesh never landed, and surface it in the assets header so a
// partial or interrupted build's gaps are visible at a glance.
function updateMissingMeshCount() {
	let missing = 0;
	for (const node of treeNodes.values()) {
		if (node.kind !== "object" && node.kind !== "frame") continue;
		if (!nodeHasMesh(node.id)) missing += 1;
	}
	assetsMissingEl.textContent = `· ${missing} missing`;
	assetsMissingEl.classList.toggle("has-missing", missing > 0);
}

assetsHeaderEl.addEventListener("click", () => {
	const collapsed = assetsEl.classList.toggle("collapsed");
	assetsToggleEl.textContent = collapsed ? "▸" : "▾";
});

// --- trellis queue panel ----------------------------------------------------
//
// Authoritative state comes from the server via GET /trellis/queue. We do
// NOT derive queue state from streamed events: SSE replays the full event
// log on every subscribe, and any historical `trellis.submit` whose run
// was killed before logging `trellis.done` would leak as a stale
// "processing" row forever. Polling the live snapshot sidesteps that
// entirely — a server restart resets the queue to empty (correct
// behaviour), and a single poll trumps any amount of historical noise.
//
// Each backend declares a concurrency POOL (Trellis/Modal vs Hunyuan 3.1/
// Tencent); the server tags every row with its `backend` + `pool`, and the
// panel renders one section per pool, each with its OWN cap — so the two never
// look like one shared budget. Rows from cells other than the one currently
// viewed are shown greyed with a slot tag and aren't clickable (their node ids
// aren't in this cell's tree).

// Latest snapshot from /trellis/queue:
//   { pools: [{id, label, cap}, ...],
//     entries: [{slot_id, job_id, state, since, task_id?, backend, pool}, ...] }
// `since` is the server's epoch-seconds timestamp from when the job entered
// the queue (set on the server side, persists across client reconnects).
// Render uses it directly so the elapsed timer reflects true wall-clock
// age, not "time since this browser first noticed the row". `pool` buckets the
// row into its section; `backend` drives the per-row badge.
let trellisQueueSnapshot = { pools: [], entries: [] };

// Short display label for a backend scope (as tagged on a queue row), so a
// Hunyuan 3.1 row is unmistakable even within its section.
function backendLabel(backend) {
	switch (backend) {
		case "hunyuan_tencent":
		case "hunyuan-tencent":
			return "Hunyuan 3.1";
		case "hunyuan":
		case "hunyuan-omni":
			return "Hunyuan";
		case "trellis":
			return "Trellis";
		default:
			return backend || "?";
	}
}

function fmtElapsed(ms) {
	const s = Math.floor(ms / 1000);
	if (s < 60) return `${s}s`;
	const m = Math.floor(s / 60);
	if (m < 60) return `${m}m ${s % 60}s`;
	return `${Math.floor(m / 60)}h ${m % 60}m`;
}

async function pollTrellisQueue() {
	try {
		const resp = await fetch(`${SERVER_URL}/trellis/queue`, {
			cache: "no-store",
		});
		if (!resp.ok) return;
		trellisQueueSnapshot = await resp.json();
		renderTrellisQueue();
	} catch {
		// Server transiently down / network blip — keep the last good snapshot
		// on screen. Next tick will refresh.
	}
}

// One queue row. Greyed + non-clickable when it belongs to a cell other than
// the one on screen (its node id isn't in this cell's tree). Carries a backend
// badge so a Hunyuan 3.1 row reads differently from a Trellis one.
function renderQueueRow(row, kind, currentCompositeId, now) {
	const el = document.createElement("div");
	el.className = `tq-row ${kind}`;
	const inThisSlot = row.slot_id === currentCompositeId;
	if (inThisSlot) {
		el.addEventListener("click", () => selectTreeNode(row.job_id));
	} else {
		el.classList.add("other-slot");
		el.title = `from cell ${row.slot_id} — switch to inspect`;
	}
	const dot = document.createElement("span");
	dot.className = "tq-dot";
	el.appendChild(dot);
	if (row.backend) {
		const badge = document.createElement("span");
		badge.className = `tq-backend backend-${row.pool ?? "modal"}`;
		badge.textContent = backendLabel(row.backend);
		el.appendChild(badge);
	}
	const idEl = document.createElement("span");
	idEl.className = "tq-id";
	idEl.textContent = row.job_id;
	el.appendChild(idEl);
	if (!inThisSlot) {
		const slotTag = document.createElement("span");
		slotTag.className = "tq-slot-tag";
		slotTag.textContent = row.slot_id;
		el.appendChild(slotTag);
	}
	// Elapsed only shown for processing rows — we want to measure how long the
	// backend takes to return a mesh once it actually starts, not how long a row
	// sat in the local FIFO. Server's `since` resets on state transition, so this
	// is true processing-elapsed when shown.
	if (kind === "processing") {
		const elapsed = document.createElement("span");
		elapsed.className = "tq-elapsed";
		elapsed.textContent = fmtElapsed(Math.max(0, now - row.sinceMs));
		el.appendChild(elapsed);
	}
	return el;
}

// A processing/waiting sub-list inside a pool section.
function renderQueueList(rows, kind, cap, currentCompositeId, now) {
	const wrap = document.createElement("div");
	wrap.className = "tq-section";
	const label = document.createElement("div");
	label.className = "tq-label";
	label.textContent = kind;
	const capEl = document.createElement("span");
	capEl.className = "tq-cap";
	capEl.textContent =
		kind === "processing" && cap != null
			? `(${rows.length}/${cap})`
			: `(${rows.length})`;
	label.appendChild(document.createTextNode(" "));
	label.appendChild(capEl);
	wrap.appendChild(label);

	const list = document.createElement("div");
	list.className = "tq-list";
	if (rows.length === 0) {
		const empty = document.createElement("div");
		empty.className = "tq-empty";
		empty.textContent = kind === "processing" ? "(idle)" : "(none)";
		list.appendChild(empty);
	} else {
		for (const row of rows)
			list.appendChild(
				renderQueueRow(row, kind, currentCompositeId, now),
			);
	}
	wrap.appendChild(list);
	return wrap;
}

function renderTrellisQueue() {
	const now = Date.now();
	const entries = trellisQueueSnapshot.entries ?? [];
	const pools =
		trellisQueueSnapshot.pools && trellisQueueSnapshot.pools.length
			? trellisQueueSnapshot.pools
			: FALLBACK_QUEUE_POOLS;

	// Bucket every entry by its server-tagged pool; an unknown pool falls into the
	// first (Modal) section so nothing is dropped.
	const byPool = new Map(pools.map((p) => [p.id, []]));
	for (const e of entries) {
		const poolId = byPool.has(e.pool) ? e.pool : pools[0].id;
		byPool.get(poolId).push({ ...e, sinceMs: (e.since ?? 0) * 1000 });
	}

	const currentCompositeId =
		currentRun !== null && currentSlotId !== null && currentModel !== null
			? `${currentRun}/${currentSlotId}/${currentModel}`
			: null;

	trellisQueueBodyEl.textContent = "";
	let totalProcessing = 0;
	let totalWaiting = 0;

	for (const pool of pools) {
		const rows = byPool.get(pool.id) ?? [];
		const processing = rows
			.filter((r) => r.state === "processing")
			.sort((a, b) => a.sinceMs - b.sinceMs);
		const waiting = rows
			.filter((r) => r.state !== "processing")
			.sort((a, b) => a.sinceMs - b.sinceMs);
		totalProcessing += processing.length;
		totalWaiting += waiting.length;

		const section = document.createElement("div");
		section.className = `tq-pool pool-${pool.id}`;
		const head = document.createElement("div");
		head.className = "tq-pool-head";
		const title = document.createElement("span");
		title.className = "tq-pool-title";
		title.textContent = pool.label ?? pool.id;
		head.appendChild(title);
		const summary = document.createElement("span");
		summary.className = "tq-pool-summary";
		summary.textContent = `${processing.length}/${pool.cap ?? "—"} · ${waiting.length} waiting`;
		head.appendChild(summary);
		section.appendChild(head);

		section.appendChild(
			renderQueueList(
				processing,
				"processing",
				pool.cap,
				currentCompositeId,
				now,
			),
		);
		section.appendChild(
			renderQueueList(waiting, "waiting", null, currentCompositeId, now),
		);
		trellisQueueBodyEl.appendChild(section);
	}

	trellisQueueCountsEl.textContent = `${totalProcessing} processing · ${totalWaiting} waiting`;
}

// Poll every 1.5s. Cheap (an in-memory dict snapshot on the server) and
// keeps the UI within a heartbeat of reality without flooding.
setInterval(pollTrellisQueue, 1500);
// Re-render once a second between polls so elapsed timers tick smoothly.
// Only the processing rows show an elapsed timer, so this is a no-op when
// the queue is empty or only contains waiting entries.
setInterval(() => {
	const hasProcessing = (trellisQueueSnapshot.entries ?? []).some(
		(e) => e.state === "processing",
	);
	if (hasProcessing) renderTrellisQueue();
}, 1000);
// Kick an initial fetch so the panel populates without waiting a full tick.
pollTrellisQueue();

trellisQueueHeaderEl.addEventListener("click", () => {
	const collapsed = trellisQueueEl.classList.toggle("collapsed");
	trellisQueueToggleEl.textContent = collapsed ? "▸" : "▾";
});

renderTrellisQueue();

// --- tree view --------------------------------------------------------------

// Mirror of the server-side recursion. Nodes are upserted by `bbox` (when
// placed) or by `divider.decompose` (announces children before their bboxes
// are resolved, so the tree shows pending placeholders). The `step` event
// drives the per-node phase badge and the global "active" highlight.
//
// Extra fields used by the hover tooltip:
//   plan         — zone plan text from `divider.zone_plan` (zones only)
//   imagePrompt  — noun phrase actually sent to Banana+Trellis, from the
//                  `image` event (objects/frames once they've been generated)
const treeNodes = new Map(); // id -> { id, parentId, prompt, kind, phase, order, plan?, imagePrompt? }
const treeChildren = new Map(); // parentId -> [childIds] in insertion order
// LLM call traces per node. Populated from `llm.call` events as they stream;
// every entry is one structured-output call (system + user + output +
// reasoning) tagged with the pipeline step that issued it. The observability
// modal reads this to show every prompt that shaped each node, side-by-side
// with the node's ancestors and descendants. Insertion order is preserved so
// the calls render in the order the pipeline made them.
const nodeLlmCalls = new Map(); // id -> [{step, system, user, output, reasoning, model, cached, eventIndex}]
// Provenance per node — the LLM calls that *brought this node into existence*
// (rather than calls *issued from* it). Filled in as we observe `cache.llm`
// events whose outputs name this id. The two relations we care about:
//   "emitted_by" — a decomposition call whose output named this id
//                  (zone_decompose, anchor_decompose, encapsulating_decompose,
//                  negative_space_decompose, next_object). The emitting step
//                  is also what tells us "was this an anchor object or a
//                  follow-up next_object loop pick?" — the user explicitly
//                  asked for this distinction.
//   "placed_by"  — a bbox-batch call that assigned this id its concrete bbox
//                  (child_bbox_batch, object_bbox_batch).
// Same shape as `nodeLlmCalls` entries plus a `relation` tag and the parent
// `node` id from the call (so the UI can label "via parent_zone.anchor_decompose").
const nodeProvenance = new Map(); // id -> [{relation, call: {...}}]
// Monotonic id stamped on every recorded LLM call so the execution-flow graph
// can give each step node a stable key, and a step-node click can scroll the
// observability modal to that exact call block (`data-call-key`).
let _llmCallUid = 0;
let treeRootId = null;
let treeActiveId = null;
let treeOrderCounter = 0;

// Per-node mesh-visibility overrides. Toggled from two places that share
// this state: the tree's per-row visibility button and right-clicking a
// mesh on the canvas. Hides the mesh + solid fill but LEAVES the bbox
// wireframe visible — useful for peeking through an outer object without
// losing the volumetric reference. Hiding a zone transitively hides every
// descendant (ancestor walk in `effectivelyHidden`); hiding a frame or
// object only hides that node itself, since you usually want to peek
// past a single piece of geometry without losing its contents. State is
// per-run — cleared on slot switch / rewind / reset, not persisted.
const hiddenIds = new Set();

function treeUpsert(id, patch) {
	const cur = treeNodes.get(id);
	if (!cur) {
		const parentId = patch.parentId ?? null;
		treeNodes.set(id, {
			id,
			parentId,
			prompt: null,
			kind: "zone",
			phase: "pending",
			order: treeOrderCounter++,
			...patch,
		});
		if (parentId !== null) {
			const arr = treeChildren.get(parentId) ?? [];
			if (!arr.includes(id)) arr.push(id);
			treeChildren.set(parentId, arr);
		} else if (treeRootId === null) {
			treeRootId = id;
		}
		return;
	}
	// Existing node: merge patch, and keep the parent -> children index in
	// sync. Decompose events may only know "this was emitted by parent X",
	// while the later bbox event carries the actual structural parent.
	const prevParent = cur.parentId;
	Object.assign(cur, patch);
	const nextParent = cur.parentId ?? null;
	if (prevParent !== nextParent) {
		if (prevParent !== null) {
			const prevKids = treeChildren.get(prevParent) ?? [];
			treeChildren.set(
				prevParent,
				prevKids.filter((cid) => cid !== id),
			);
		} else if (treeRootId === id) {
			treeRootId = null; // was mis-rooted
		}
		if (nextParent !== null) {
			const nextKids = treeChildren.get(nextParent) ?? [];
			if (!nextKids.includes(id)) nextKids.push(id);
			treeChildren.set(nextParent, nextKids);
		} else if (treeRootId === null) {
			treeRootId = id;
		}
	}
}

function treeSetPhase(id, phase) {
	const cur = treeNodes.get(id);
	if (!cur) {
		// Step fired before any bbox / decompose — stash the phase so it renders
		// as soon as we have the node.
		treeUpsert(id, { phase });
	} else {
		cur.phase = phase;
	}
	if (phase !== "done") {
		treeActiveId = id;
	} else if (treeActiveId === id) {
		// A node finishing doesn't move the focus elsewhere on its own; leave
		// the highlight on it until the next step event moves it.
	}
	scheduleRenderTree();
	if (
		id === selectedBboxId ||
		(selectedBboxId && treeIsAncestorOf(selectedBboxId, id))
	) {
		renderTreeDetail();
	}
}

function recordLlmCall(event) {
	// Feed the per-run spend tracker (keyed by event index, so the backfill /
	// live-tail overlap doesn't double-count).
	trackLlmCost(event);
	// Bucket the call under its `node` id. The server stamps that field on
	// every `llm.call`; if it's missing (older log line, or a call site we
	// haven't tagged) bucket it under "_unattributed" so the modal can still
	// surface it under the root view rather than dropping it on the floor.
	const id = event.node || "_unattributed";
	const call = {
		uid: _llmCallUid++,
		step: event.step || "(unknown step)",
		// Output-schema class name (e.g. "BboxBatchOutput"). The prompt-tuning
		// sandbox sends this to POST /llm/test so the server can resolve the
		// right schema to re-run this exact step under an edited prompt.
		schema: event.schema ?? null,
		system: event.system ?? "",
		user: event.user ?? "",
		output: event.output ?? null,
		reasoning: event.reasoning ?? "",
		model: event.model ?? "",
		cached: !!event.cached,
		eventIndex: typeof event.index === "number" ? event.index : null,
		// The "node" field on the event is the call site's node id — for
		// decompose/bbox-batch calls, that's the *parent*. Carried through so the
		// provenance render can label "from parent_zone.anchor_decompose".
		parentNode: event.node || null,
	};
	const list = nodeLlmCalls.get(id) ?? [];
	list.push(call);
	nodeLlmCalls.set(id, list);
	// A new call adds a step node (and maybe scene children) to the exec graph;
	// flag a refresh so it appears live even when no tree mutation accompanies
	// this cache.llm event. Drained, throttled, by the animate() loop.
	if (flowModalOpen) _flowRenderPending = true;
	// Now scan the output for ids this call brought into existence and back-
	// fill provenance for each. The output shape is structured-output, so we
	// know exactly which fields name child ids:
	//   * `children: [{id, ...}]`          — divider.zone_decompose
	//   * `assignments: [{id, bbox}]`      — child_bbox_batch + object_bbox_batch
	//   * `objects: [{id, ...}]`           — anchor_decompose, encapsulating_decompose,
	//                                        negative_space_decompose
	//   * `object: {id, ...}` (non-null)   — next_object (single emitted spec)
	// Anything else (zone_plan, image_prompt, overall_bbox) doesn't emit ids.
	const out = event.output;
	if (!out || typeof out !== "object") return;
	const isBboxStep =
		call.step === "child_bbox_batch" || call.step === "object_bbox_batch";
	const relation = isBboxStep ? "placed_by" : "emitted_by";
	function tag(childId) {
		if (!childId || childId === id) return; // never attach a node to itself
		const arr = nodeProvenance.get(childId) ?? [];
		// Dedup on (step + parentNode + eventIndex) — events can replay during
		// SSE reconnects and we don't want the same trace entry twice.
		const key = `${call.step}|${call.parentNode}|${call.eventIndex}`;
		if (
			arr.some(
				(e) =>
					`${e.call.step}|${e.call.parentNode}|${e.call.eventIndex}` ===
					key,
			)
		) {
			return;
		}
		arr.push({ relation, call });
		nodeProvenance.set(childId, arr);
	}
	if (Array.isArray(out.children)) {
		for (const c of out.children) tag(c?.id);
	}
	if (Array.isArray(out.assignments)) {
		for (const a of out.assignments) tag(a?.id);
	}
	if (Array.isArray(out.objects)) {
		for (const o of out.objects) tag(o?.id);
	}
	if (out.object && typeof out.object === "object") {
		tag(out.object.id);
	}
}

function treeClear() {
	treeNodes.clear();
	treeChildren.clear();
	nodeLlmCalls.clear();
	nodeProvenance.clear();
	treeRootId = null;
	treeActiveId = null;
	treeOrderCounter = 0;
	hiddenIds.clear();
	treeBodyEl.innerHTML = "";
	treeActiveEl.textContent = "";
	destroyDetailPreview();
	treeDetailEl.innerHTML = "";
	treeEl.classList.remove("detail-open");
	updateMissingMeshCount();
	// The per-run spend tracker shares the tree's lifecycle: a view switch /
	// reset / rewind wipes the LLM calls it aggregates.
	clearCostTracker();
}

// --- per-run LLM spend tracker ----------------------------------------------
//
// Every billable LLM request lands as a `cache.llm` event. We bucket the run's
// reasoning calls (the selected model) by pipeline step and pool the
// gemini-flash-lite library-matching calls separately, then price each with
// OpenRouter's published per-token rates. The map is keyed by the event's
// `index` so the same call arriving via both the history backfill and the live
// SSE tail (their ranges can overlap) is counted once.

// USD per token (prompt / completion). Source: OpenRouter model catalog
// (openrouter.ai/api/v1/models), captured 2026-06. Keyed by the OpenRouter
// model id stamped on each cache.llm event. An id missing here contributes 0
// cost but is still counted as a request, so a newly-added model degrades to a
// request-only row until its price is filled in.
const MODEL_PRICING = {
	"google/gemini-3.5-flash": { in: 0.0000015, out: 0.000009 },
	"google/gemini-3.1-flash-lite": { in: 0.00000025, out: 0.0000015 },
	"google/gemini-3.1-pro-preview": { in: 0.000002, out: 0.000012 },
	"openai/gpt-5.5": { in: 0.000005, out: 0.00003 },
	"anthropic/claude-opus-4.6": { in: 0.000005, out: 0.000025 },
	"deepseek/deepseek-v4-pro": { in: 0.000000435, out: 0.00000087 },
	"anthropic/claude-opus-4.8": { in: 0.000005, out: 0.000025 },
};

// eventIndex -> { step, model, tokensIn, tokensOut, isMatch, exact }
const llmCostCalls = new Map();
let _costFallbackKey = 0;
let _costRenderPending = false;

// Rough token count when the server didn't log usage (older runs). ~4 chars per
// token is the usual English heuristic — fine for a spend *estimate*.
function estTokens(s) {
	return s ? Math.ceil(String(s).length / 4) : 0;
}

function trackLlmCost(event) {
	const key =
		typeof event.index === "number"
			? event.index
			: `u${_costFallbackKey++}`;
	const haveUsage =
		Number.isFinite(event.tokens_in) && Number.isFinite(event.tokens_out);
	const tokensIn = Number.isFinite(event.tokens_in)
		? event.tokens_in
		: estTokens(event.system) + estTokens(event.user);
	const out = event.output;
	const outText =
		typeof out === "string" ? out : out == null ? "" : JSON.stringify(out);
	const tokensOut = Number.isFinite(event.tokens_out)
		? event.tokens_out
		: estTokens(outText) + estTokens(event.reasoning);
	// Library matching always runs on gemini-flash-lite. Identify it by its
	// step (new logs) or output schema (every prior log) so the split survives
	// even when the run's reasoning model is also flash-lite.
	const isMatch =
		event.step === "library_match" || event.schema === "LibraryMatchOutput";
	llmCostCalls.set(key, {
		step: event.step || "(unknown step)",
		model: event.model || "",
		tokensIn,
		tokensOut,
		isMatch,
		exact: haveUsage,
	});
	scheduleCostRender();
}

function llmCallCost(c) {
	const p = MODEL_PRICING[c.model];
	if (!p) return 0;
	return c.tokensIn * p.in + c.tokensOut * p.out;
}

function fmtCost(v) {
	v = v || 0;
	return v >= 1 ? "$" + v.toFixed(2) : "$" + v.toFixed(4);
}

function clearCostTracker() {
	llmCostCalls.clear();
	scheduleCostRender();
}

function scheduleCostRender() {
	if (_costRenderPending) return;
	_costRenderPending = true;
	requestAnimationFrame(() => {
		_costRenderPending = false;
		renderCostTracker();
	});
}

function costRow(label, count, cost, className) {
	const row = document.createElement("div");
	row.className = className;
	const step = document.createElement("span");
	step.className = "cost-step";
	step.textContent = label;
	const reqs = document.createElement("span");
	reqs.className = "cost-count";
	reqs.textContent = String(count);
	const amt = document.createElement("span");
	amt.className = "cost-amt";
	amt.textContent = fmtCost(cost);
	row.append(step, reqs, amt);
	return row;
}

function costSectionHead(name, tag, count, cost) {
	const head = document.createElement("div");
	head.className = "cost-section-head";
	const nameEl = document.createElement("span");
	nameEl.className = "cost-sec-name";
	nameEl.textContent = name;
	const tagEl = document.createElement("span");
	tagEl.className = "cost-sec-model";
	tagEl.textContent = tag;
	const sumEl = document.createElement("span");
	sumEl.className = "cost-sec-sum";
	sumEl.textContent = `${fmtCost(cost)} · ${count} req`;
	head.append(nameEl, tagEl, sumEl);
	return head;
}

function renderCostTracker() {
	const byStep = new Map(); // step -> { count, cost }
	let reasoningCount = 0;
	let reasoningCost = 0;
	let matchCount = 0;
	let matchCost = 0;
	let anyEstimated = false;
	for (const c of llmCostCalls.values()) {
		const cost = llmCallCost(c);
		if (!c.exact) anyEstimated = true;
		if (c.isMatch) {
			matchCount += 1;
			matchCost += cost;
		} else {
			reasoningCount += 1;
			reasoningCost += cost;
			const e = byStep.get(c.step) ?? { count: 0, cost: 0 };
			e.count += 1;
			e.cost += cost;
			byStep.set(c.step, e);
		}
	}
	const totalCount = reasoningCount + matchCount;
	const totalCost = reasoningCost + matchCost;
	costPillSummaryEl.textContent = `${fmtCost(totalCost)} · ${totalCount} req`;

	costDropdownEl.innerHTML = "";
	if (totalCount === 0) {
		const empty = document.createElement("div");
		empty.className = "cost-empty";
		empty.textContent = "no LLM requests yet";
		costDropdownEl.appendChild(empty);
		return;
	}

	if (reasoningCount > 0) {
		const section = document.createElement("div");
		section.className = "cost-section";
		section.appendChild(
			costSectionHead(
				currentModel ?? "selected model",
				"reasoning",
				reasoningCount,
				reasoningCost,
			),
		);
		const steps = [...byStep.entries()].sort(
			(a, b) => b[1].cost - a[1].cost,
		);
		for (const [step, agg] of steps) {
			section.appendChild(
				costRow(
					step.replace(/_/g, " "),
					agg.count,
					agg.cost,
					"cost-row",
				),
			);
		}
		costDropdownEl.appendChild(section);
	}

	if (matchCount > 0) {
		const div = document.createElement("div");
		div.className = "cost-divider";
		costDropdownEl.appendChild(div);
		const section = document.createElement("div");
		section.className = "cost-section";
		section.appendChild(
			costSectionHead(
				"gemini-flash-lite",
				"matching",
				matchCount,
				matchCost,
			),
		);
		costDropdownEl.appendChild(section);
	}

	const div = document.createElement("div");
	div.className = "cost-divider";
	costDropdownEl.appendChild(div);
	costDropdownEl.appendChild(
		costRow("total", totalCount, totalCost, "cost-total"),
	);

	if (anyEstimated) {
		const note = document.createElement("div");
		note.className = "cost-est-note";
		note.textContent = "≈ some calls estimated from text (no token log)";
		costDropdownEl.appendChild(note);
	}
}

costPillEl.addEventListener("click", () => {
	costTrackerEl.classList.toggle("collapsed");
});

// Paint the empty state on boot. Deferred via rAF (scheduleCostRender), so it
// runs after module init — `currentModel` is no longer in its TDZ by then.
scheduleCostRender();

// True if `id` itself is hidden, or any ZONE ancestor is hidden. The
// zone-only ancestor rule is what makes hiding a zone hide everything
// underneath it without having to mark each child individually, while
// still letting the user hide a single frame/object to peek inside it
// without taking its children offscreen too.
function effectivelyHidden(id) {
	let cur = treeNodes.get(id);
	let isSelf = true;
	while (cur) {
		if (hiddenIds.has(cur.id) && (isSelf || cur.kind === "zone"))
			return true;
		cur = cur.parentId ? treeNodes.get(cur.parentId) : null;
		isSelf = false;
	}
	return false;
}

function toggleNodeHidden(id) {
	if (hiddenIds.has(id)) hiddenIds.delete(id);
	else hiddenIds.add(id);
	// Effective-hidden status changed for this node and (potentially) every
	// descendant, so re-evaluate visibility for the whole subtree.
	refreshSubtreeVisibility(id);
	renderTree();
}

function refreshSubtreeVisibility(rootId) {
	const stack = [rootId];
	while (stack.length) {
		const cur = stack.pop();
		applyModelVisibility(cur);
		applyBboxVisibility(cur);
		applySolidFillVisibility(cur);
		const kids = treeChildren.get(cur) ?? [];
		for (const k of kids) stack.push(k);
	}
}

function truncate(s, n = 60) {
	if (!s) return "";
	return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function treeIsAncestorOf(ancestorId, descendantId) {
	let cur = treeNodes.get(descendantId)?.parentId ?? null;
	while (cur !== null) {
		if (cur === ancestorId) return true;
		cur = treeNodes.get(cur)?.parentId ?? null;
	}
	return false;
}

function treeHasDescendantPhase(id, predicate) {
	const stack = [...(treeChildren.get(id) ?? [])];
	while (stack.length) {
		const curId = stack.pop();
		const node = treeNodes.get(curId);
		if (!node) continue;
		if (predicate(node.phase ?? "pending")) return true;
		const kids = treeChildren.get(curId) ?? [];
		for (const kid of kids) stack.push(kid);
	}
	return false;
}

function treeDisplayPhase(id) {
	const node = treeNodes.get(id);
	const ownPhase = node?.phase ?? "pending";
	if (ownPhase === "error") return "error";
	if (treeHasDescendantPhase(id, (phase) => phase === "error"))
		return "error";

	const active = treeActiveId !== null ? treeNodes.get(treeActiveId) : null;
	const activePhase = active?.phase ?? null;
	const hasActiveDescendant =
		treeActiveId !== null &&
		treeActiveId !== id &&
		activePhase !== "done" &&
		activePhase !== "error" &&
		treeIsAncestorOf(id, treeActiveId);
	if (hasActiveDescendant) return "building_children";
	return ownPhase;
}

function renderTreeNode(id, ctx) {
	const node = treeNodes.get(id);
	if (!node) return null;
	if (ctx && !ctx.visible.has(id)) return null;
	const wrap = document.createElement("div");
	const classes = ["tree-node"];
	if (id === treeActiveId) classes.push("active");
	if (id === selectedBboxId) classes.push("selected");
	if (ctx && ctx.matches.has(id)) classes.push("matched", "match-highlight");
	if (effectivelyHidden(id)) classes.push("vis-hidden");
	wrap.className = classes.join(" ");
	wrap.dataset.id = id;

	const row = document.createElement("div");
	row.className = "tree-row";
	// Click the row (not a nested child-tree row) to select this node.
	row.addEventListener("click", (ev) => {
		ev.stopPropagation();
		selectTreeNode(id);
		// When the execution-flow canvas is open, a sidebar click also zooms it
		// onto the matching scene node so the two views stay coupled.
		if (flowModalOpen) {
			renderFlow();
			flowCenterOnScene(id);
		}
	});

	// Per-node visibility toggle. Reflects only the self-hidden state — a
	// descendant of a hidden ancestor still shows ● because its own bit is
	// off; the row's `vis-hidden` class communicates effective state.
	const visBtn = document.createElement("button");
	visBtn.type = "button";
	const selfHidden = hiddenIds.has(id);
	visBtn.className = `tree-vis-toggle${selfHidden ? " hidden" : ""}`;
	visBtn.textContent = selfHidden ? "○" : "●";
	visBtn.title = selfHidden ? "Show node" : "Hide node";
	visBtn.addEventListener("click", (ev) => {
		ev.stopPropagation();
		toggleNodeHidden(id);
	});
	row.appendChild(visBtn);

	const idEl = document.createElement("span");
	idEl.className = `tree-id ${node.kind}`;
	idEl.textContent = node.id;
	row.appendChild(idEl);

	const promptEl = document.createElement("span");
	promptEl.className = "tree-prompt";
	promptEl.textContent = truncate(node.prompt, 80);
	promptEl.title = node.prompt ?? "";
	row.appendChild(promptEl);

	const phaseEl = document.createElement("span");
	const displayPhase = treeDisplayPhase(id);
	phaseEl.className = `tree-phase phase-${displayPhase}`;
	phaseEl.textContent = displayPhase;
	row.appendChild(phaseEl);

	wrap.appendChild(row);

	const childIds = treeChildren.get(id) ?? [];
	if (childIds.length > 0) {
		const kidsEl = document.createElement("div");
		kidsEl.className = "tree-children";
		for (const cid of childIds) {
			const cEl = renderTreeNode(cid, ctx);
			if (cEl) kidsEl.appendChild(cEl);
		}
		if (kidsEl.childNodes.length > 0) wrap.appendChild(kidsEl);
	}
	return wrap;
}

let treeSearchQuery = "";
// Index of the currently-focused match within `orderedMatches`. Repeated
// Enter advances; Shift+Enter goes back. -1 = nothing selected yet for
// this query (next Enter selects [0]).
let treeMatchIndex = -1;

// A node is a `match` if its id or prompt contains the query.
// `visible` = matches ∪ ancestors-of-matches ∪ descendants-of-matches.
// Returns null when no query (everything visible, no match highlighting).
function computeTreeFilter() {
	const q = treeSearchQuery.trim().toLowerCase();
	if (!q) return null;
	const matches = new Set();
	for (const [id, node] of treeNodes) {
		const idHit = id.toLowerCase().includes(q);
		const promptHit = (node.prompt ?? "").toLowerCase().includes(q);
		if (idHit || promptHit) matches.add(id);
	}
	const visible = new Set(matches);
	// Walk ancestors up.
	for (const id of matches) {
		let cur = treeNodes.get(id);
		while (cur && cur.parentId) {
			if (visible.has(cur.parentId)) break;
			visible.add(cur.parentId);
			cur = treeNodes.get(cur.parentId);
		}
	}
	// Walk descendants down so users can see the matched subtree expanded.
	const stack = [...matches];
	while (stack.length) {
		const id = stack.pop();
		const kids = treeChildren.get(id) ?? [];
		for (const cid of kids) {
			if (!visible.has(cid)) {
				visible.add(cid);
				stack.push(cid);
			}
		}
	}
	return { matches, visible };
}

// Match order for "Enter selects first match" — use insertion order so
// the first match in the rendered tree is what gets selected.
function orderedMatches(filter) {
	if (!filter) return [];
	const sorted = [...treeNodes.values()]
		.filter((n) => filter.matches.has(n.id))
		.sort((a, b) => a.order - b.order);
	return sorted.map((n) => n.id);
}

// Coalesced tree re-render. A single SSE snapshot can fire `renderTree`
// hundreds of times (one per bbox / step / model), and each call rebuilds the
// entire tree DOM — O(events × nodes), a major contributor to switch lag.
// Streaming paths call `scheduleRenderTree`, which only flips a flag; the
// animate() loop drains it with a single rebuild per frame. User-interaction
// paths (search / select / hide) keep calling `renderTree` directly so their
// feedback stays synchronous (and so code that reads the freshly-built DOM on
// the next line still works).
let _treeRenderPending = false;
function scheduleRenderTree() {
	_treeRenderPending = true;
}

function renderTree() {
	// A synchronous render satisfies any pending coalesced one.
	_treeRenderPending = false;
	treeBodyEl.innerHTML = "";
	const filter = computeTreeFilter();
	if (treeRootId !== null) {
		const el = renderTreeNode(treeRootId, filter);
		if (el) treeBodyEl.appendChild(el);
	}
	if (treeActiveId !== null) {
		const n = treeNodes.get(treeActiveId);
		if (n) treeActiveEl.textContent = `${treeDisplayPhase(n.id)} · ${n.id}`;
	} else {
		treeActiveEl.textContent = "";
	}
	if (filter) {
		const n = filter.matches.size;
		if (n === 0) {
			treeSearchCountEl.textContent = "no matches";
		} else if (treeMatchIndex >= 0) {
			treeSearchCountEl.textContent = `${(treeMatchIndex % n) + 1}/${n}`;
		} else {
			treeSearchCountEl.textContent = `${n} match${n === 1 ? "" : "es"}`;
		}
	} else {
		treeSearchCountEl.textContent = "";
	}
	updateMissingMeshCount();
}

treeHeaderEl.addEventListener("click", () => {
	const collapsed = treeEl.classList.toggle("collapsed");
	treeToggleEl.textContent = collapsed ? "▸" : "▾";
});

treeSearchEl.addEventListener("input", () => {
	treeSearchQuery = treeSearchEl.value;
	treeSearchClearEl.classList.toggle("visible", treeSearchQuery.length > 0);
	treeMatchIndex = -1;
	renderTree();
});

treeSearchEl.addEventListener("keydown", (ev) => {
	if (ev.key === "Enter") {
		ev.preventDefault();
		const ordered = orderedMatches(computeTreeFilter());
		if (ordered.length === 0) return;
		if (treeMatchIndex < 0) {
			treeMatchIndex = ev.shiftKey ? ordered.length - 1 : 0;
		} else {
			const delta = ev.shiftKey ? -1 : 1;
			treeMatchIndex =
				(treeMatchIndex + delta + ordered.length) % ordered.length;
		}
		const target = ordered[treeMatchIndex];
		// Bypass selectTreeNode's toggle behaviour — re-pressing Enter on the
		// same node should cycle, not deselect.
		if (selectedBboxId !== target) selectTreeNode(target);
		const row = treeBodyEl.querySelector(
			`.tree-node[data-id="${CSS.escape(target)}"]`,
		);
		if (row) row.scrollIntoView({ block: "nearest", behavior: "smooth" });
		renderTree();
	} else if (ev.key === "Escape") {
		treeSearchEl.value = "";
		treeSearchEl.dispatchEvent(new Event("input"));
	}
});

treeSearchClearEl.addEventListener("click", () => {
	treeSearchEl.value = "";
	treeSearchEl.dispatchEvent(new Event("input"));
	treeSearchEl.focus();
});

// --- three.js scene ---------------------------------------------------------

const host = document.getElementById("canvas-host");
// preserveDrawingBuffer keeps the WebGL framebuffer readable after present,
// which is required for the gif export path (gif.js calls getImageData on
// the canvas). Small perf cost on every frame; acceptable for a debug tool.
const renderer = new THREE.WebGLRenderer({
	antialias: true,
	preserveDrawingBuffer: true,
});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x101114);
// Physically-based output so streamed PBR GLBs read with real depth instead of
// flat: sRGB framebuffer + ACES filmic tonemapping (graceful highlight rolloff)
// and soft shadow maps. The IBL environment + key light are set up with the
// scene below.
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const sceneRoot = new THREE.Group();
scene.add(sceneRoot);

// Bboxes live in a sibling group so they don't participate in fit-to-scene,
// and so clearScene can nuke them independently.
const bboxRoot = new THREE.Group();
scene.add(bboxRoot);

// --- prompt-tuning sandbox state --------------------------------------------
//
// The sandbox lets the user rewind the canvas to a single pipeline step and
// re-run that step's LLM call under an edited prompt — non-destructively. The
// real run is never mutated: we PAUSE it (server), detach the SSE, and
// simulate the rewind purely by toggling the visibility of already-loaded
// scene objects (every object remembers the event index that created it, and
// while `rewindCutoffIndex` is set, anything created at/after the cursor is
// hidden). A tested step's output is drawn into `sandboxOverlayRoot` on top of
// that rewound state. Breaking out clears the cutoff + overlay, restoring the
// exact scene, and resumes the run if we were the ones who paused it.
const sandboxOverlayRoot = new THREE.Group();
scene.add(sandboxOverlayRoot);
let sandboxActive = false;
// Ordered list of every recorded LLM-call step (across all nodes) by event
// index — the spine the user walks with prev/next. Each entry is a
// nodeLlmCalls `call` object.
let sandboxSteps = [];
let sandboxCursor = -1;
// Whether THIS sandbox session paused the run (so break-out knows to resume).
let sandboxPausedByUs = false;
// While set, scene-object visibility is gated to "created before this event
// index"; null = show everything (normal mode).
let rewindCutoffIndex = null;
// id -> event index at which its bbox / model first appeared. Built from
// `recordedEvents` when a sandbox session starts.
const bboxCreatedIndex = new Map();
const modelCreatedIndex = new Map();
// In-flight POST /llm/test guard so double-clicks can't stack calls.
let sandboxTesting = false;
// --- branch (downstream step-through) state ---
// When `branchActive`, the cell has been forked at the tuned step (server-side,
// under <cell>/_branch) and the deviated subtree is re-simulating one step at a
// time: the pipeline PAUSES before each downstream LLM call so you can edit its
// (pre-filled, re-rendered) prompt. We keep the original scene rewound to the
// deviation point and render the branch's downstream nodes — bboxes AND real
// meshes — into sandboxOverlayRoot as its events stream. Break-out deletes it.
let branchActive = false;
let branchSource = null; // EventSource over /branch/events
let branchDeviationIndex = -1; // original event index the fork deviated at
let branchGen = 0; // bumped per session to bail stale async mesh loads
let branchDone = false;
// `branchSteps` is the ordered list of the branch's steps: the committed ones
// (each carries the prompt that was actually committed to the branch log, plus
// its output + reasoning) followed by the live frontier (the next, un-run step,
// editable to run). `branchCursor` is the step being viewed; prev/next move it
// NON-destructively (observability). Only an explicit re-run truncates +
// invalidates downstream.
let branchSteps = [];
let branchCursor = -1;
let branchStepBusy = false; // a proceed / re-run is in flight
let branchAuto = false; // "run rest" is streaming autonomously (no pauses)
let branchRebuilding = false; // a re-run reopen's snapshot is replaying
let branchReopenTarget = 0; // cursor to settle on once a rebuild finishes
// The in-progress prompt carried from EDIT mode into the branch's first step,
// so entering simulation doesn't make you retype the edit. Consumed once.
let branchFirstPrompt = null;
const branchOverlayBboxIds = new Set(); // deviated ids already drawn as overlay wireframes
const branchOverlayMeshes = new Map(); // id -> loaded GLB object3d in the overlay
let bboxesShown = localStorage.getItem(BBOX_VISIBLE_STORAGE_KEY) !== "0";
function applyBboxToggleLabel() {
	bboxToggleEl.textContent = `bboxes: ${bboxesShown ? "on" : "off"}`;
	bboxToggleEl.classList.toggle("off", !bboxesShown);
}
applyBboxToggleLabel();
bboxToggleEl.addEventListener("click", () => {
	bboxesShown = !bboxesShown;
	localStorage.setItem(BBOX_VISIBLE_STORAGE_KEY, bboxesShown ? "1" : "0");
	applyBboxToggleLabel();
	refreshAllBboxVisibility();
	// Picking rule depends on bboxesShown — re-pick at the current cursor
	// position so the hover updates immediately instead of waiting for a
	// mouse move.
	pointerDirty = true;
});

let framesShown = localStorage.getItem(FRAMES_VISIBLE_STORAGE_KEY) !== "0";
function applyFramesToggleLabel() {
	framesToggleEl.textContent = `frames: ${framesShown ? "on" : "off"}`;
	framesToggleEl.classList.toggle("off", !framesShown);
}
applyFramesToggleLabel();
framesToggleEl.addEventListener("click", () => {
	framesShown = !framesShown;
	localStorage.setItem(FRAMES_VISIBLE_STORAGE_KEY, framesShown ? "1" : "0");
	applyFramesToggleLabel();
	refreshAllFrameModelVisibility();
	refreshAllSolidFillVisibility();
});

// Master switch for all generated GLB meshes (objects + frames). Off = pure
// bbox view. Independent of the frames toggle, which scopes only to frame
// meshes; meshes:off wins over frames:on.
let meshesShown = localStorage.getItem(MESHES_VISIBLE_STORAGE_KEY) !== "0";
function applyMeshesToggleLabel() {
	meshesToggleEl.textContent = `meshes: ${meshesShown ? "on" : "off"}`;
	meshesToggleEl.classList.toggle("off", !meshesShown);
}
applyMeshesToggleLabel();
meshesToggleEl.addEventListener("click", () => {
	meshesShown = !meshesShown;
	localStorage.setItem(MESHES_VISIBLE_STORAGE_KEY, meshesShown ? "1" : "0");
	applyMeshesToggleLabel();
	refreshAllFrameModelVisibility();
	// Picking falls back to solid fills / bboxes when meshes are hidden — kick
	// the hover so it re-evaluates without waiting for the next mouse move.
	pointerDirty = true;
});

// Selection-mode switch: "all" (default — mesh/fill picking, every kind is
// selectable) vs. "zones" (only zone bboxes are pickable, so the user can
// click a containing zone to see how frames/objects sit inside it via the
// dim-on-select highlight).
let selectMode =
	localStorage.getItem(SELECT_MODE_STORAGE_KEY) === "zones" ? "zones" : "all";
function applySelectModeToggleLabel() {
	selectModeToggleEl.textContent = `select: ${selectMode}`;
	selectModeToggleEl.classList.toggle("zones", selectMode === "zones");
}
applySelectModeToggleLabel();
selectModeToggleEl.addEventListener("click", () => {
	selectMode = selectMode === "zones" ? "all" : "zones";
	localStorage.setItem(SELECT_MODE_STORAGE_KEY, selectMode);
	applySelectModeToggleLabel();
	// Hover under the cursor is now stale — different pick set — so re-evaluate
	// immediately instead of waiting for the next mouse move.
	pointerDirty = true;
});

// Solid-fill mode — drops a solid mesh into every object/frame bbox using its
// proxy shape (or the AABB itself when no proxy_shape is set). Zones stay
// wireframe. Intended for bbox-only mode so the scene reads as solids without
// running Trellis. Independent of the bbox wireframe toggle.
let solidFillShown = localStorage.getItem(SOLID_FILL_STORAGE_KEY) === "1";
function applySolidFillToggleLabel() {
	solidFillToggleEl.textContent = `fill: ${solidFillShown ? "on" : "off"}`;
	solidFillToggleEl.classList.toggle("off", !solidFillShown);
}
applySolidFillToggleLabel();
solidFillToggleEl.addEventListener("click", () => {
	solidFillShown = !solidFillShown;
	localStorage.setItem(SOLID_FILL_STORAGE_KEY, solidFillShown ? "1" : "0");
	applySolidFillToggleLabel();
	if (solidFillShown) rebuildAllSolidFills();
	else clearSolidFills();
});

const bboxes = new Map(); // id -> THREE.Box3Helper
const proxies = new Map(); // id -> THREE.Mesh (wireframe proxy silhouette)
const solidFills = new Map(); // id -> THREE.Mesh (solid proxy/AABB fill)
const modelsById = new Map(); // id -> THREE.Object3D (the loaded gltf.scene)
let hoveredBboxId = null;

const BBOX_COLOR_DEFAULT = 0xff3b3b;
const BBOX_COLOR_OBJECT = 0x6bd96e;
const BBOX_COLOR_FRAME = 0x7fb3d5;
const BBOX_COLOR_PROXY = 0xb46aff;
const BBOX_COLOR_HOVER = 0xffe14a;
const BBOX_COLOR_SELECTED = 0x4af0e0;
const BBOX_COLOR_ORIENT = 0xffa033;
const BBOX_DIM_OPACITY = 0.35;
const PROXY_BASE_OPACITY = 0.55;
const PROXY_DIM_OPACITY = 0.2;
let selectedBboxId = null;

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let pointerDirty = false;
// Whether the cursor is currently over the WebGL canvas. Gates Shift-driven
// hover refreshes so the zones-only override never paints a phantom highlight
// when the pointer is off-canvas.
let pointerInsideCanvas = false;
let lastPointerClientX = 0;
let lastPointerClientY = 0;
let controlsInteracting = false;

const tooltip = document.createElement("div");
tooltip.id = "bbox-tooltip";
tooltip.style.cssText = [
	"position: fixed",
	"padding: 5px 9px",
	"background: rgba(22, 24, 29, 0.94)",
	"color: #e6e6e6",
	"border: 1px solid #2a2d35",
	"border-radius: 4px",
	"font: 12px ui-monospace, SFMono-Regular, Menlo, monospace",
	"pointer-events: none",
	"display: none",
	// Above the chrome incl. the sandbox panel (z-index 60) so a hover label on a
	// magenta overlay box near the panel edge isn't occluded; still below modals.
	"z-index: 70",
	"max-width: 360px",
	"white-space: pre-wrap",
	"line-height: 1.35",
].join("; ");
document.body.appendChild(tooltip);

const TOOLTIP_KIND_COLOR = {
	zone: "#9ad4ff",
	object: "#8bd17c",
	frame: "#7fb3d5",
};

const camera = new THREE.PerspectiveCamera(
	50,
	window.innerWidth / window.innerHeight,
	0.05,
	5000,
);
camera.position.set(8, 6, 10);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 1, 0);
controls.update();

// Once the user drags the camera, stop auto-fitting so subsequent runs
// preserve their chosen angle. The flag deliberately survives clearScene.
let cameraUserMoved = false;
controls.addEventListener("start", () => {
	cameraUserMoved = true;
	controlsInteracting = true;
	setHoveredBbox(null);
	tooltip.style.display = "none";
});
controls.addEventListener("end", () => {
	controlsInteracting = false;
	pointerDirty = true;
});

// --- WASD fly controls (complementary to OrbitControls) --------------------
// WASD strafes on the horizontal plane relative to the camera direction;
// Q/E moves world-down/up; Shift multiplies speed. Translates camera and
// target together so OrbitControls' pivot follows the camera.
const pressedKeys = new Set();
let _lastMoveT = performance.now();
const _MOVE_KEYS = new Set(["w", "a", "s", "d", "q", "e", "r", "f"]);

function _isTypingTarget(t) {
	return (
		t instanceof HTMLElement &&
		(t.tagName === "INPUT" ||
			t.tagName === "TEXTAREA" ||
			t.isContentEditable)
	);
}

window.addEventListener("keydown", (ev) => {
	if (_isTypingTarget(ev.target)) return;
	const k = ev.key.toLowerCase();
	if (_MOVE_KEYS.has(k)) {
		pressedKeys.add(k);
		ev.preventDefault();
	} else if (k === "shift") {
		// First Shift press also flips picking to zones-only; refresh hover so the
		// highlight reflects the new pick set without needing a mouse jiggle.
		if (!pressedKeys.has("shift")) {
			pressedKeys.add("shift");
			if (pointerInsideCanvas) pointerDirty = true;
		}
	}
});

window.addEventListener("keyup", (ev) => {
	const k = ev.key.toLowerCase();
	pressedKeys.delete(k);
	// Releasing Shift drops the zones-only override; refresh hover for the
	// full pick set.
	if (k === "shift" && pointerInsideCanvas) pointerDirty = true;
});

// Alt-tab / focus-loss: drop held keys so they don't stick on.
window.addEventListener("blur", () => {
	pressedKeys.clear();
	if (pointerInsideCanvas) pointerDirty = true;
});

const _fwd = new THREE.Vector3();
const _right = new THREE.Vector3();
const _worldUp = new THREE.Vector3(0, 1, 0);
const _move = new THREE.Vector3();

function _applyKeyboardMove(dt) {
	if (pressedKeys.size === 0) return;
	const shifted = pressedKeys.has("shift");
	const speed = 2 * (shifted ? 3 : 1) * dt;

	_fwd.subVectors(controls.target, camera.position);
	_fwd.y = 0;
	if (_fwd.lengthSq() === 0) return;
	_fwd.normalize();
	_right.crossVectors(_fwd, _worldUp).normalize();

	_move.set(0, 0, 0);
	if (pressedKeys.has("w")) _move.addScaledVector(_fwd, speed);
	if (pressedKeys.has("s")) _move.addScaledVector(_fwd, -speed);
	if (pressedKeys.has("d")) _move.addScaledVector(_right, speed);
	if (pressedKeys.has("a")) _move.addScaledVector(_right, -speed);
	if (pressedKeys.has("e")) _move.addScaledVector(_worldUp, speed);
	if (pressedKeys.has("q")) _move.addScaledVector(_worldUp, -speed);

	if (_move.lengthSq() !== 0) {
		camera.position.add(_move);
		controls.target.add(_move);
		cameraUserMoved = true;
	}

	// Dolly toward / away from the orbit target. Held key = continuous zoom;
	// ~1.5x per second baseline, 4x with shift.
	if (pressedKeys.has("r") || pressedKeys.has("f")) {
		const rate = shifted ? 4 : 1.5;
		let factor = 1;
		if (pressedKeys.has("r")) factor *= Math.pow(1 / rate, dt);
		if (pressedKeys.has("f")) factor *= Math.pow(rate, dt);
		_dolly(factor);
	}
}

function _dolly(factor) {
	// factor < 1 zooms in, factor > 1 zooms out. Implemented as scaling the
	// camera->target distance so OrbitControls' pivot semantics stay intact.
	const offset = camera.position.clone().sub(controls.target);
	const dist = offset.length();
	if (dist === 0) return;
	const minDist = 0.05;
	const maxDist = 4000;
	const newDist = Math.max(minDist, Math.min(maxDist, dist * factor));
	offset.multiplyScalar(newDist / dist);
	camera.position.copy(controls.target).add(offset);
	cameraUserMoved = true;
}

// Image-based lighting. A prefiltered RoomEnvironment drives indirect light +
// reflections on every PBR mesh — this is what lifts the streamed GLBs out of
// "flat": without an environment, MeshStandardMaterial has nothing to reflect.
// Passing the renderer calibrates the room's intensity for physically-based
// lights (the renderer default since three r155), so the IBL isn't ~180x dim.
// The returned PMREM texture outlives the generator, so both temporaries are
// disposed immediately.
{
	const pmrem = new THREE.PMREMGenerator(renderer);
	const envScene = new RoomEnvironment(renderer);
	scene.environment = pmrem.fromScene(envScene, 0.04).texture;
	envScene.dispose();
	pmrem.dispose();
}
// The IBL is calibrated bright; keep it as fill + reflections only so the key
// light below can actually shape the scene. At full strength it washes out all
// directionality and shadows. Tunable live via the lighting panel.
scene.environmentIntensity = 0.35;

// Gentle sky/ground fill on top of the IBL so undersides never read fully black.
const hemiLight = new THREE.HemisphereLight(0xffffff, 0x202028, 0.2);
scene.add(hemiLight);

// Key light — the only shadow caster. Its position, target, and shadow frustum
// are fit to the scene bounds whenever geometry changes (updateSceneLighting),
// so shadows stay crisp whether the scene is a small room or a large arena.
const SHADOW_MAP_SIZE = 4096;
const dir = new THREE.DirectionalLight(0xffffff, 3.5);
dir.castShadow = true;
dir.shadow.mapSize.set(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
dir.shadow.bias = -0.0001;
// normalBias is set per-scene in updateSceneLighting, scaled to the shadow
// texel's world size — a fixed value over-/under-biases as the frustum scales,
// which is the main cause of the self-shadow acne flicker on large scenes.
scene.add(dir);
scene.add(dir.target);

// Shadow reception is re-gated on this shared uniform (see
// patchMaterialReceiveShadow) instead of three's per-object `receiveShadow`
// uniform, which doesn't reach these streamed PBR materials. The "cast shadows"
// toggle flips `.value`; the caster itself stays on so toggling needs no
// recompile.
const _forceReceiveShadow = { value: true };

const axesHelper = new THREE.AxesHelper(1);
axesHelper.material.toneMapped = false; // keep the reference axes' raw colors
scene.add(axesHelper);

// Ground plane that renders ONLY the shadows cast onto it (transparent
// everywhere else), so contact shadows read clearly and the panel's "shadow"
// slider has something to drive. Lives on `scene` (not `sceneRoot`), so it's
// excluded from fit-to-scene, picking, and the .glb export. depthWrite off +
// depthTest on means it overlays the ground but is occluded by objects in front.
const shadowCatcher = new THREE.Mesh(
	new THREE.PlaneGeometry(1, 1),
	new THREE.ShadowMaterial({ opacity: 0.4, depthWrite: false }),
);
shadowCatcher.rotation.x = -Math.PI / 2;
shadowCatcher.receiveShadow = true;
shadowCatcher.renderOrder = -1;
scene.add(shadowCatcher);

// Fixed key-light direction (upper front-right); only the distance + shadow
// frustum scale with the scene. Recomputed from the sceneRoot bounds so the
// shadow map tightly wraps the geometry at any scale. `box` may be null/empty
// (bbox-only or freshly-cleared scenes) — fall back to a sane default volume.
const _LIGHT_DIR = new THREE.Vector3(4, 8, 6).normalize();
const _lightCenter = new THREE.Vector3();
const _lightSize = new THREE.Vector3();
// Remembered so the lighting panel can refit the shadow frustum to the current
// scene when only the sun direction changes — no fresh Box3 traversal needed.
let _lastSceneBox = null;
function updateSceneLighting(box) {
	const hasGeom = !!box && !box.isEmpty();
	_lastSceneBox = hasGeom ? box : null;
	if (hasGeom) {
		box.getCenter(_lightCenter);
		box.getSize(_lightSize);
	} else {
		_lightCenter.set(0, 0, 0);
		_lightSize.set(20, 20, 20);
	}
	// True bounding-sphere radius (half the diagonal), not half the max extent —
	// so the ortho shadow frustum actually encloses the whole scene and its
	// texels/depth precision are spent tightly on the geometry (crisp contact
	// shadows) rather than on empty margin.
	const radius = Math.max(
		0.5,
		0.5 *
			Math.sqrt(
				_lightSize.x * _lightSize.x +
					_lightSize.y * _lightSize.y +
					_lightSize.z * _lightSize.z,
			),
	);
	const minY = hasGeom ? box.min.y : 0;
	const dist = radius * 3;

	dir.position.copy(_lightCenter).addScaledVector(_LIGHT_DIR, dist);
	dir.target.position.copy(_lightCenter);
	dir.target.updateMatrixWorld();

	const cam = dir.shadow.camera;
	// Hug the bounding sphere: lateral extent just covers it, near/far bracket it
	// tightly so depth resolution (and thus contact-shadow accuracy) is maximized.
	const extent = radius * 1.05;
	cam.left = -extent;
	cam.right = extent;
	cam.top = extent;
	cam.bottom = -extent;
	cam.near = Math.max(0.01, dist - radius * 1.1);
	cam.far = dist + radius * 1.1;
	cam.updateProjectionMatrix();

	// Normal-offset bias scaled to the shadow texel's world size (~2 texels) —
	// the primary defense against self-shadow acne. Generated/decimated meshes
	// (computed smooth normals that diverge from their flat faces) flicker badly
	// without it; scaling to the frustum keeps acne suppression consistent from a
	// small room to a large arena instead of a fixed value that's too weak when
	// the scene (and thus each texel) is large.
	dir.shadow.normalBias = ((2 * extent) / SHADOW_MAP_SIZE) * 2.0;

	// Sit the catcher just above the lowest geometry so its contact shadows
	// overlay the floor (or open ground) and stay visible from above.
	shadowCatcher.position.set(
		_lightCenter.x,
		minY + radius * 0.003,
		_lightCenter.z,
	);
	shadowCatcher.scale.set(radius * 6, radius * 6, 1);
}
updateSceneLighting(null);

// --- lighting panel ---------------------------------------------------------
// Live controls for the lighting configured above. Each input writes straight
// to the renderer / lights so the scene updates as you drag; values persist.
// Bumped to .v2 to discard any pre-rebalance saved state (the old defaults let
// the environment wash out all directional lighting + shadows).
const LIGHTING_STORAGE_KEY = "starshot.lighting.v2";
const LIGHTING_DEFAULTS = {
	exposure: 1.0,
	key: 3.5, // directional dominates so the scene has a clear lit/shadowed side
	fill: 0.2,
	env: 0.35, // IBL is fill + reflections only, not the main light
	shadow: 0.4,
	azimuth: 34, // ≈ atan2(4, 6) — matches the initial (4, 8, 6) key direction
	elevation: 48, // ≈ asin(8 / |(4, 8, 6)|)
	shadows: true,
};
// Each slider field maps to inputs #lp-<field> + readout #lp-<field>-val.
const LP_CONTROLS = [
	{ field: "exposure", fmt: (v) => v.toFixed(2) },
	{ field: "key", fmt: (v) => v.toFixed(1) },
	{ field: "fill", fmt: (v) => v.toFixed(2) },
	{ field: "env", fmt: (v) => v.toFixed(2) },
	{ field: "shadow", fmt: (v) => v.toFixed(2) },
	{ field: "azimuth", fmt: (v) => `${Math.round(v)}°` },
	{ field: "elevation", fmt: (v) => `${Math.round(v)}°` },
];

const lightingState = (() => {
	let saved = null;
	try {
		saved = JSON.parse(
			localStorage.getItem(LIGHTING_STORAGE_KEY) || "null",
		);
	} catch {}
	return {
		...LIGHTING_DEFAULTS,
		...(saved && typeof saved === "object" ? saved : {}),
	};
})();

function persistLighting() {
	try {
		localStorage.setItem(
			LIGHTING_STORAGE_KEY,
			JSON.stringify(lightingState),
		);
	} catch {}
}

// Sun direction from azimuth (0°=+Z front, 90°=+X right) + elevation, then
// refit the shadow frustum to the last known scene bounds.
function setLightDirFromAngles(azDeg, elDeg) {
	const az = THREE.MathUtils.degToRad(azDeg);
	const el = THREE.MathUtils.degToRad(elDeg);
	const cosEl = Math.cos(el);
	_LIGHT_DIR
		.set(Math.sin(az) * cosEl, Math.sin(el), Math.cos(az) * cosEl)
		.normalize();
	updateSceneLighting(_lastSceneBox);
}

// Scalar params apply directly to the live objects (read each render).
function applyScalarLighting() {
	renderer.toneMappingExposure = lightingState.exposure;
	dir.intensity = lightingState.key;
	hemiLight.intensity = lightingState.fill;
	scene.environmentIntensity = lightingState.env;
	shadowCatcher.material.opacity = lightingState.shadow;
	// Caster stays on (dir.castShadow set once at init); gate reception through
	// the shared uniform so toggling needs no shader recompile and can't black
	// out the scene by emptying the shadow-map array mid-run.
	_forceReceiveShadow.value = lightingState.shadows;
	shadowCatcher.visible = lightingState.shadows;
}

function applyLightingState() {
	applyScalarLighting();
	setLightDirFromAngles(lightingState.azimuth, lightingState.elevation);
}

function syncLightingControls() {
	for (const c of LP_CONTROLS) {
		document.getElementById(`lp-${c.field}`).value = lightingState[c.field];
		document.getElementById(`lp-${c.field}-val`).textContent = c.fmt(
			lightingState[c.field],
		);
	}
	document.getElementById("lp-shadows-enabled").checked =
		lightingState.shadows;
}

for (const c of LP_CONTROLS) {
	const input = document.getElementById(`lp-${c.field}`);
	const val = document.getElementById(`lp-${c.field}-val`);
	input.addEventListener("input", () => {
		lightingState[c.field] = parseFloat(input.value);
		val.textContent = c.fmt(lightingState[c.field]);
		if (c.field === "azimuth" || c.field === "elevation") {
			setLightDirFromAngles(
				lightingState.azimuth,
				lightingState.elevation,
			);
		} else {
			applyScalarLighting();
		}
		persistLighting();
	});
}

document
	.getElementById("lp-shadows-enabled")
	.addEventListener("change", (ev) => {
		lightingState.shadows = ev.target.checked;
		applyScalarLighting();
		persistLighting();
	});

const lightingPanelEl = document.getElementById("lighting-panel");
const lightingToggleEl = document.getElementById("lighting-toggle");
function setLightingPanelOpen(open) {
	lightingPanelEl.classList.toggle("open", open);
	lightingToggleEl.classList.toggle("active", open);
}
lightingToggleEl.addEventListener("click", () =>
	setLightingPanelOpen(!lightingPanelEl.classList.contains("open")),
);
document
	.getElementById("lighting-close")
	.addEventListener("click", () => setLightingPanelOpen(false));
document.getElementById("lighting-reset").addEventListener("click", () => {
	Object.assign(lightingState, LIGHTING_DEFAULTS);
	applyLightingState();
	syncLightingControls();
	persistLighting();
});

// Push persisted/default values onto both the scene and the controls now that
// the lights + DOM exist.
syncLightingControls();
applyLightingState();

// Debug hook: inspect the live lighting/shadow state from the devtools console
// via `__lighting.report()`. Handy for diagnosing whether the shadow-camera
// frustum actually encloses the scene.
window.__lighting = {
	dir,
	hemiLight,
	shadowCatcher,
	renderer,
	scene,
	sceneRoot,
	updateSceneLighting,
	report() {
		const box = new THREE.Box3().setFromObject(sceneRoot);
		const empty = box.isEmpty();
		const size = empty
			? [0, 0, 0]
			: box.getSize(new THREE.Vector3()).toArray();
		const center = empty
			? [0, 0, 0]
			: box.getCenter(new THREE.Vector3()).toArray();
		let meshes = 0;
		let casters = 0;
		let receivers = 0;
		const matTypes = new Set();
		sceneRoot.traverse((o) => {
			if (!o.isMesh) return;
			meshes++;
			if (o.castShadow) casters++;
			if (o.receiveShadow) receivers++;
			const m = Array.isArray(o.material) ? o.material[0] : o.material;
			if (m) matTypes.add(m.type);
		});
		const c = dir.shadow.camera;
		return {
			boxEmpty: empty,
			size,
			center,
			lightPos: dir.position.toArray(),
			targetPos: dir.target.position.toArray(),
			dirCastShadow: dir.castShadow,
			shadowMapEnabled: renderer.shadowMap.enabled,
			shadowCam: {
				l: c.left,
				r: c.right,
				t: c.top,
				b: c.bottom,
				near: c.near,
				far: c.far,
			},
			meshes,
			casters,
			receivers,
			matTypes: [...matTypes],
		};
	},
	// Drop a fresh box caster onto a fresh plane receiver (both vanilla
	// MeshStandardMaterial, no Meshopt/quantization, standard three shadow path)
	// at the scene center. If THIS box shadows THIS plane, three's shadows work
	// for ordinary standard receivers here and the bug is specific to the loaded
	// (quantized) GLBs; if it doesn't, the break is global to standard materials.
	shadowTest() {
		this.clearTest();
		const box = new THREE.Box3().setFromObject(sceneRoot);
		const center = box.isEmpty()
			? new THREE.Vector3()
			: box.getCenter(new THREE.Vector3());
		const size = box.isEmpty()
			? new THREE.Vector3(10, 10, 10)
			: box.getSize(new THREE.Vector3());
		const r = Math.max(2, 0.25 * Math.max(size.x, size.y, size.z));
		const y0 = box.isEmpty() ? 0 : box.min.y;

		const plane = new THREE.Mesh(
			new THREE.PlaneGeometry(r * 4, r * 4),
			new THREE.MeshStandardMaterial({ color: 0x888888, roughness: 1 }),
		);
		plane.rotation.x = -Math.PI / 2;
		plane.position.set(center.x, y0 + r * 0.5, center.z);
		plane.receiveShadow = true;
		plane.castShadow = false;

		const cube = new THREE.Mesh(
			new THREE.BoxGeometry(r, r, r),
			new THREE.MeshStandardMaterial({ color: 0xff5533, roughness: 0.8 }),
		);
		cube.position.set(center.x, y0 + r * 1.4, center.z);
		cube.castShadow = true;
		cube.receiveShadow = true;

		scene.add(plane);
		scene.add(cube);
		this._test = [plane, cube];
		// Make sure the shadow frustum encloses the test rig (it sits at center,
		// already covered, but refit defensively).
		updateSceneLighting(new THREE.Box3().setFromObject(sceneRoot));
		console.log(
			"shadowTest: red cube on gray plane at scene center (y=",
			(y0 + r * 0.5).toFixed(2),
			"). Orbit to it — does the cube cast a shadow on the gray plane?",
		);
		return "added test rig — call __lighting.clearTest() to remove";
	},
	clearTest() {
		for (const o of this._test ?? []) {
			scene.remove(o);
			o.geometry?.dispose?.();
			o.material?.dispose?.();
		}
		this._test = [];
	},
};

// Infinite ground grid: a huge plane with a procedural grid shader. Lines
// antialias via screen-space derivatives and fade with distance so the plane
// never looks like it has an edge. Fade distance is driven from camera
// distance each frame so detail scales naturally as the user zooms.
const gridGeom = new THREE.PlaneGeometry(100000, 100000);
gridGeom.rotateX(-Math.PI / 2);
const gridMat = new THREE.ShaderMaterial({
	uniforms: {
		uCameraPos: { value: new THREE.Vector3() },
		uMinorColor: { value: new THREE.Color(0x202020) },
		uMajorColor: { value: new THREE.Color(0x505050) },
		uMinorSpacing: { value: 1.0 },
		uMajorSpacing: { value: 10.0 },
		uFadeStart: { value: 20.0 },
		uFadeEnd: { value: 200.0 },
	},
	vertexShader: `
    varying vec3 vWorldPos;
    void main() {
      vec4 wp = modelMatrix * vec4(position, 1.0);
      vWorldPos = wp.xyz;
      gl_Position = projectionMatrix * viewMatrix * wp;
    }
  `,
	fragmentShader: `
    uniform vec3 uCameraPos;
    uniform vec3 uMinorColor;
    uniform vec3 uMajorColor;
    uniform float uMinorSpacing;
    uniform float uMajorSpacing;
    uniform float uFadeStart;
    uniform float uFadeEnd;
    varying vec3 vWorldPos;

    float gridLine(vec2 p, float spacing) {
      vec2 q = p / spacing;
      vec2 g = abs(fract(q - 0.5) - 0.5) / fwidth(q);
      return 1.0 - min(min(g.x, g.y), 1.0);
    }

    void main() {
      float minor = gridLine(vWorldPos.xz, uMinorSpacing);
      float major = gridLine(vWorldPos.xz, uMajorSpacing);
      float d = distance(vWorldPos.xz, uCameraPos.xz);
      float fade = 1.0 - smoothstep(uFadeStart, uFadeEnd, d);
      float alpha = max(minor * 0.5, major) * fade;
      if (alpha < 0.002) discard;
      vec3 col = mix(uMinorColor, uMajorColor, major);
      gl_FragColor = vec4(col, alpha);
    }
  `,
	transparent: true,
	depthWrite: false,
	side: THREE.DoubleSide,
});
const groundGrid = new THREE.Mesh(gridGeom, gridMat);
groundGrid.renderOrder = -1;
scene.add(groundGrid);

let gridShown = localStorage.getItem(GRID_VISIBLE_STORAGE_KEY) !== "0";
function applyGridToggle() {
	gridToggleEl.textContent = `grid: ${gridShown ? "on" : "off"}`;
	gridToggleEl.classList.toggle("off", !gridShown);
	groundGrid.visible = gridShown;
}
applyGridToggle();
gridToggleEl.addEventListener("click", () => {
	gridShown = !gridShown;
	localStorage.setItem(GRID_VISIBLE_STORAGE_KEY, gridShown ? "1" : "0");
	applyGridToggle();
});

// The controls/topbar grows and reflows (tabs, slots, versions, row-wrapping,
// collapse/expand), so its height isn't fixed. Pin the tree just below the
// topbar's live bottom edge instead of a hard-coded top, so a tall controls
// panel can never overlap the tree. The CSS top/max-height are only the
// pre-JS fallback.
const topbarEl = document.getElementById("topbar");
const TREE_TOP_GAP = 12;
let _treeLayoutPending = false;
function layoutTree() {
	const top =
		Math.round(topbarEl.getBoundingClientRect().bottom) + TREE_TOP_GAP;
	treeEl.style.top = `${top}px`;
	// Anchor the tree's bottom edge as before (20px margin + 30vh reserved for
	// the lower-left panels); only its height flexes as the topbar grows.
	treeEl.style.maxHeight = `calc(100vh - ${top + 20}px - 30vh)`;
}
function scheduleTreeLayout() {
	if (_treeLayoutPending) return;
	_treeLayoutPending = true;
	requestAnimationFrame(() => {
		_treeLayoutPending = false;
		layoutTree();
	});
}
new ResizeObserver(scheduleTreeLayout).observe(topbarEl);
scheduleTreeLayout();

window.addEventListener("resize", () => {
	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();
	renderer.setSize(window.innerWidth, window.innerHeight);
	scheduleTreeLayout();
});

// Coalescing flag for fitToScene (drained once per frame in animate, below):
// a burst of per-mesh attach completions collapses into one O(scene) Box3
// refit per frame instead of one traversal per mesh.
let _fitScenePending = false;
function scheduleFitToScene() {
	_fitScenePending = true;
}

function animate() {
	requestAnimationFrame(animate);
	const now = performance.now();
	const dt = Math.min(0.1, (now - _lastMoveT) / 1000);
	_lastMoveT = now;
	_applyKeyboardMove(dt);
	controls.update();

	// Drain coalesced UI work once per frame: collapse a burst of streamed tree
	// mutations into one DOM rebuild, and a burst of mesh completions into one
	// camera refit. Both run before render so the frame reflects them.
	if (_treeRenderPending) {
		renderTree();
		if (flowModalOpen) _flowRenderPending = true;
	}
	if (_fitScenePending) {
		_fitScenePending = false;
		fitToScene();
	}
	// Refresh the execution-flow graph from the same coalesced tree mutations,
	// throttled so a burst of streamed nodes rebuilds the canvas at most a few
	// times a second rather than every frame.
	if (flowModalOpen && _flowRenderPending && now - _flowLastRender > 150) {
		_flowRenderPending = false;
		_flowLastRender = now;
		renderFlow();
	}

	gridMat.uniforms.uCameraPos.value.copy(camera.position);
	const camDist = Math.max(1, camera.position.distanceTo(controls.target));
	gridMat.uniforms.uFadeStart.value = camDist * 0.5;
	gridMat.uniforms.uFadeEnd.value = camDist * 6.0;

	if (pointerDirty && !controlsInteracting) {
		pointerDirty = false;
		raycaster.setFromCamera(pointer, camera);
		if (sandboxActive) {
			// The magenta overlay boxes take priority; fall back to the still-visible
			// frozen context (pre-rewind originals) so those keep their tooltip too.
			const oid = pickHoveredOverlayId();
			if (oid !== null) {
				setHoveredOverlay(oid);
				setHoveredBbox(null);
				positionOverlayTooltip(
					lastPointerClientX,
					lastPointerClientY,
					oid,
				);
			} else {
				setHoveredOverlay(null);
				const hoveredId = pickHoveredBboxId();
				setHoveredBbox(hoveredId);
				if (hoveredId !== null) {
					positionTooltip(
						lastPointerClientX,
						lastPointerClientY,
						hoveredId,
					);
				} else {
					tooltip.style.display = "none";
				}
			}
		} else {
			const hoveredId = pickHoveredBboxId();
			setHoveredBbox(hoveredId);
			if (hoveredId !== null) {
				positionTooltip(
					lastPointerClientX,
					lastPointerClientY,
					hoveredId,
				);
			} else {
				tooltip.style.display = "none";
			}
		}
	}

	renderer.render(scene, camera);
}
animate();

function clearScene() {
	resetModelQueue();
	while (sceneRoot.children.length > 0) {
		const child = sceneRoot.children[0];
		sceneRoot.remove(child);
		disposeObject3D(child);
	}
	for (const helper of bboxes.values()) {
		bboxRoot.remove(helper);
		helper.geometry?.dispose?.();
		helper.material?.dispose?.();
	}
	bboxes.clear();
	for (const mesh of proxies.values()) {
		bboxRoot.remove(mesh);
		mesh.geometry?.dispose?.();
		mesh.material?.dispose?.();
	}
	proxies.clear();
	clearSolidFills();
	modelsById.clear();
	genMeshVersions.clear();
	genMeshSymmetry.clear();
	genMeshOptimized.clear();
	regeneratingIds.clear();
	unsymmetrizingIds.clear();
	symmetrizingIds.clear();
	// The next cell resolves its own generated versions; the gate poll re-adopts
	// the latest and repopulates the picker.
	genVersion = null;
	genVersions = [];
	clearSandboxOverlay();
	hoveredBboxId = null;
	selectedBboxId = null;
	updateOrientationIndicator();
	tooltip.style.display = "none";
}

// Fit controls to the union of all loaded models after each addition.
// Skipped once the user has manually adjusted the camera.
function fitToScene() {
	const box = new THREE.Box3().setFromObject(sceneRoot);
	// Refit the key light + shadow frustum to the scene whenever geometry
	// changes — even after the user has taken camera control, so only the
	// camera-fit half below bails on cameraUserMoved.
	updateSceneLighting(box);
	if (cameraUserMoved) return;
	if (box.isEmpty()) return;
	const size = box.getSize(new THREE.Vector3());
	const center = box.getCenter(new THREE.Vector3());
	const radius = 0.5 * Math.max(size.x, size.y, size.z);
	if (!isFinite(radius) || radius === 0) return;
	controls.target.copy(center);
	const dist = radius / Math.tan((camera.fov * Math.PI) / 360);
	const dirVec = new THREE.Vector3(1, 0.7, 1).normalize();
	camera.position.copy(center).addScaledVector(dirVec, dist * 1.6);
	camera.near = Math.max(0.01, radius / 100);
	camera.far = Math.max(100, radius * 100);
	camera.updateProjectionMatrix();
	controls.update();
}

// --- model loading ----------------------------------------------------------

// Optimized-library GLBs (assets-optimized + rebake_runs.py) are Meshopt-
// compressed with KTX2/Basis textures — extensionsRequired lists
// EXT_meshopt_compression, KHR_mesh_quantization, KHR_texture_basisu — so a bare
// GLTFLoader can't parse them. Wire the Basis transcoder (shipped with three,
// served at /vendor/three/... by client/server.mjs) and the Meshopt decoder into
// every loader. detectSupport(renderer) picks the GPU's transcode target.
// Both geometry (Meshopt) and texture (KTX2) decode default to a single thread,
// which serializes a scene's hundreds of meshes and is the main load-time
// bottleneck. Give each a worker pool scaled to the machine so they decode in
// parallel, off the main thread. One knob drives both.
const DECODE_WORKERS = Math.min(
	16,
	Math.max(4, navigator.hardwareConcurrency || 4),
);

const ktx2Loader = new KTX2Loader()
	.setTranscoderPath("/vendor/three/examples/jsm/libs/basis/")
	.setWorkerLimit(DECODE_WORKERS)
	.detectSupport(renderer);

// EXT_meshopt_compression decode otherwise runs synchronously on the main
// thread even though GLTFLoader awaits it (decodeGltfBufferAsync just wraps a
// blocking WASM call); a worker pool routes it off-thread so vertex/index
// decode parallelizes across the streamed bundle like KTX2 above. Called once
// on the shared singleton — every GLTFLoader that uses it inherits the pool.
MeshoptDecoder.useWorkers(DECODE_WORKERS);

function configureGltfLoader(gltfLoader) {
	return gltfLoader
		.setKTX2Loader(ktx2Loader)
		.setMeshoptDecoder(MeshoptDecoder);
}

const loader = configureGltfLoader(new GLTFLoader());

// Re-gate a material's shadow term on our `_forceReceiveShadow` uniform instead
// of three's per-object `receiveShadow`, which doesn't reach these streamed PBR
// materials (so they'd never sample the shadow map). The replaced chunk lives
// inside `#if defined( USE_SHADOWMAP )`, so this is inert in the mini-viewer
// (no shadow map → block compiled out; the injected uniform is just unused).
function patchMaterialReceiveShadow(m) {
	if (!m || m.userData.__recvPatched) return;
	m.userData.__recvPatched = true;
	const prevCompile = m.onBeforeCompile;
	m.onBeforeCompile = (shader, rndr) => {
		if (prevCompile) prevCompile(shader, rndr);
		shader.uniforms.uForceReceiveShadow = _forceReceiveShadow;
		shader.fragmentShader = shader.fragmentShader
			.replace(
				"#include <common>",
				"#include <common>\nuniform bool uForceReceiveShadow;",
			)
			.replace(
				"#include <lights_fragment_begin>",
				THREE.ShaderChunk.lights_fragment_begin.replace(
					/\(\s*directLight\.visible\s*&&\s*receiveShadow\s*\)/g,
					"( directLight.visible && uForceReceiveShadow )",
				),
			);
	};
	// Distinct, stable cache key so patched materials get their own program
	// (and three doesn't warn about onBeforeCompile without a cache key).
	const prevKey = m.customProgramCacheKey?.bind(m);
	m.customProgramCacheKey = () => "recvForce1|" + (prevKey ? prevKey() : "");
	m.needsUpdate = true;
}

// Every streamed/loaded GLB gets the same treatment: double-sided (Trellis
// shells are often single-sided), shadow-enabled so the key light grounds them
// and they self-shadow, and the receive-shadow re-gate above. The shadow flags
// are harmless no-ops in the mini-viewer, whose renderer has no shadow map.
function prepareLoadedScene(root) {
	root.traverse((child) => {
		if (!child.isMesh) return;
		child.castShadow = true;
		child.receiveShadow = true;
		// Generated (Trellis) GLBs ship without a NORMAL attribute. Lighting still
		// works (three derives flat normals per-fragment) and casting works (the
		// depth pass ignores normals), but the shadow-RECEIVE vertex shader does
		// normalize(transformedNormal) → normalize(0) = NaN, making
		// vDirectionalShadowCoord NaN so getShadow() always returns "lit" — no
		// received shadows, no acne even at zero bias. Computing normals gives a
		// valid shadow coordinate, which is what lets generated meshes receive.
		if (child.geometry && !child.geometry.getAttribute("normal")) {
			child.geometry.computeVertexNormals();
		}
		if (!child.material) return;
		const mats = Array.isArray(child.material)
			? child.material
			: [child.material];
		for (const m of mats) {
			m.side = THREE.DoubleSide;
			// Cast only back faces into the shadow map (front-face culling). A lit
			// front face is then always in front of the stored (far) depth, so it
			// can never self-compare into shadow — the robust cure for self-shadow
			// acne, independent of the vertex normals normalBias relies on (which
			// diverge from the faces on decimated/optimized meshes). normalBias
			// above still covers thin/flat pieces where front≈back.
			m.shadowSide = THREE.BackSide;
			patchMaterialReceiveShadow(m);
		}
	});
}

// Concurrent loads — each GLB fetches/parses/uploads independently. The
// browser's per-origin connection limit (~6) naturally throttles the network
// side. `sceneGen` invalidates loads still in flight when the scene is reset
// (rewind / fresh snapshot on reconnect).
let sceneGen = 0;

// When the gif replay is running, every GLB has been preloaded up front (see
// preloadReplayGlbs) and stashed here keyed by node id. `loadModel` checks
// this first so `model` events attach the cached scene synchronously and the
// gif frame containing the event also contains the mesh — instead of the
// mesh popping in several frames later when the async fetch finally lands.
let replayPreloadCache = null;

function resetModelQueue() {
	sceneGen += 1;
}

// Dispose a material *and* every texture it references — material.dispose()
// frees the shader program but leaves uploaded textures resident in GPU memory
// until each is disposed individually. Used by the streamed bundle path so
// swapping a mesh never strands the previous GLB's textures in VRAM.
function disposeMaterial(material) {
	if (!material) return;
	for (const value of Object.values(material)) {
		if (value && value.isTexture) value.dispose();
	}
	material.dispose?.();
}

function disposeObject3D(root) {
	root?.traverse?.((n) => {
		if (!n.isMesh) return;
		n.geometry?.dispose?.();
		const mats = Array.isArray(n.material)
			? n.material
			: n.material
				? [n.material]
				: [];
		for (const m of mats) disposeMaterial(m);
	});
}

function parseGlb(arrayBuffer) {
	return new Promise((resolve, reject) => {
		loader.parse(arrayBuffer, "", resolve, reject);
	});
}

// --- one-connection scene bundle (streamed + progressive) ------------------
//
// Opening a built scene used to fire one HTTP GET per object GLB, which the
// browser caps at ~6 concurrent per origin (and which contend with the SSE
// stream + polls on the same origin). Instead we pull the whole cell's GLBs
// over ONE connection — GET /slots/<slot>/<model>/meshes streams a
// length-prefixed binary bundle — and attach each mesh AS IT ARRIVES.
//
// Streaming (not buffering the whole response) keeps JS memory bounded to a
// small window of in-flight GLBs regardless of total scene size, and meshes
// appear progressively. `model` events coordinate via `streamDone`: an id the
// bundle already attached needs only bookkeeping; anything it didn't carry (a
// post-snapshot mesh, a retry, a parse miss) falls back to an individual fetch.
// Tied to `sceneGen` so a stale scene's bundle is ignored.
const _MESH_BUNDLE_MAGIC = "SMB1";
let meshBundle = {
	gen: -1,
	attached: new Set(),
	streamDone: Promise.resolve(),
	abort: null,
};

function prefetchMeshBundle(slotId, model, gen) {
	if (!slotId || !model) return;
	// Supersede any still-streaming bundle from a scene we've left.
	meshBundle.abort?.abort?.();
	const abort = new AbortController();
	let resolveDone;
	const streamDone = new Promise((r) => {
		resolveDone = r;
	});
	// `streaming` is a synchronous flag (streamDone is a promise) so the gate
	// poll can cheaply tell a bundle is still in flight and defer its not-yet-
	// attached ids to it rather than racing it with per-mesh fetches.
	const state = {
		gen,
		attached: new Set(),
		streamDone,
		abort,
		streaming: true,
	};
	meshBundle = state;
	(async () => {
		try {
			const res = await fetch(slotMeshesUrl(slotId, model), {
				cache: "no-store",
				signal: abort.signal,
			});
			if (res.ok && res.body) {
				await consumeMeshBundleStream(res.body.getReader(), state);
			}
		} catch {
			// Aborted (superseded) / network / parse error — whatever attached so
			// far stays; the rest is picked up per-object by the model-event
			// fallback once streamDone resolves.
		} finally {
			state.streaming = false;
			resolveDone();
		}
	})();
}

// Pull exact byte counts out of a ReadableStream reader, holding only the
// not-yet-consumed chunks (each dropped as it's read), so peak memory tracks
// the largest single frame — never the whole multi-GB bundle.
function _byteStreamReader(reader) {
	const chunks = [];
	let avail = 0;
	let ended = false;
	async function readExact(n) {
		while (avail < n) {
			if (ended) return null;
			const { done, value } = await reader.read();
			if (done) {
				ended = true;
				continue;
			}
			if (value && value.length) {
				chunks.push(value);
				avail += value.length;
			}
		}
		const out = new Uint8Array(n);
		let filled = 0;
		while (filled < n) {
			const c = chunks[0];
			const take = Math.min(c.length, n - filled);
			out.set(c.subarray(0, take), filled);
			filled += take;
			if (take === c.length) chunks.shift();
			else chunks[0] = c.subarray(take);
			avail -= take;
		}
		return out;
	}
	return { readExact };
}

// Diagnostic: log per-phase first-switch load timings to the console under a
// `[load]` prefix. Flip off once the bottleneck is understood.
const LOAD_TIMING = true;

// Parse + attach up to this many meshes concurrently. Reading frames off the
// stream stays sequential on the one connection; only the parse/decode fans
// out, bounding peak memory to ~MAX_INFLIGHT in-flight GLBs.
const MESH_BUNDLE_MAX_INFLIGHT = 20;

async function consumeMeshBundleStream(reader, state) {
	const r = _byteStreamReader(reader);
	const dec = new TextDecoder();
	const magic = await r.readExact(4);
	if (!magic || dec.decode(magic) !== _MESH_BUNDLE_MAGIC) return;
	const t0 = performance.now();
	let count = 0;
	let bytes = 0;
	let firstMeshMs = 0;
	const inflight = new Set();
	while (true) {
		if (state.gen !== sceneGen || state.abort.signal.aborted) break;
		const idLenB = await r.readExact(4);
		if (!idLenB) break;
		const idLen = new DataView(idLenB.buffer).getUint32(0, true);
		const idB = await r.readExact(idLen);
		if (!idB) break;
		const id = dec.decode(idB);
		const glbLenB = await r.readExact(4);
		if (!glbLenB) break;
		const glbLen = new DataView(glbLenB.buffer).getUint32(0, true);
		const glbB = await r.readExact(glbLen);
		if (!glbB) break;
		count++;
		bytes += glbLen;
		const p = attachBundleMesh(id, glbB.buffer, state.gen, state)
			.then(() => {
				if (firstMeshMs === 0) firstMeshMs = performance.now() - t0;
			})
			.finally(() => inflight.delete(p));
		inflight.add(p);
		// Hold the read at MAX_INFLIGHT in-flight parses; resume once one drains.
		if (inflight.size >= MESH_BUNDLE_MAX_INFLIGHT)
			await Promise.race(inflight);
	}
	await Promise.allSettled(inflight);
	if (LOAD_TIMING) {
		console.info(
			`[load] mesh bundle: ${count} meshes · ${(bytes / 1e6).toFixed(1)}MB · ` +
				`first paint ${firstMeshMs | 0}ms · decode+attach ${(performance.now() - t0) | 0}ms`,
		);
	}
	// One camera refit after the whole bundle lands — a per-mesh fitToScene()
	// is an O(scene) Box3 traversal, i.e. quadratic across a large scene.
	if (state.gen === sceneGen) fitToScene();
}

// Parse one streamed GLB and add it to the scene. Mirrors the attach half of
// _loadModelNow; on a parse failure it bails without attaching, leaving the id
// for the model-event fallback to fetch fresh.
async function attachBundleMesh(id, glbBuffer, gen, state) {
	if (gen !== sceneGen) return;
	let gltf;
	try {
		gltf = await parseGlb(glbBuffer);
	} catch {
		return;
	}
	if (gen !== sceneGen) {
		disposeObject3D(gltf.scene);
		return;
	}
	prepareLoadedScene(gltf.scene);
	gltf.scene.name = `mesh:${id}`;
	gltf.scene.userData.pickId = id;
	const prev = modelsById.get(id);
	if (prev) {
		sceneRoot.remove(prev);
		disposeObject3D(prev);
	}
	sceneRoot.add(gltf.scene);
	modelsById.set(id, gltf.scene);
	state.attached.add(id);
	applyModelVisibility(id);
	upsertAsset(id, { status: "loaded" });
}

async function loadModel(event) {
	if (replayPreloadCache) {
		const cached = replayPreloadCache.get(event.id);
		if (cached) {
			if (cached.gltf) {
				attachPreloadedGlb(event, cached.gltf);
			} else if (cached.error) {
				appendEvent({
					kind: "model.error",
					id: event.id,
					message: cached.error.message,
				});
				upsertAsset(event.id, {
					status: "error",
					errorMessage: cached.error.message,
				});
			}
			return;
		}
	}
	// A scene bundle streaming for this generation attaches meshes progressively.
	// Wait for it to finish, then adopt what it attached (bookkeeping only) or
	// fetch this id individually if the bundle didn't carry it (a mesh generated
	// after the snapshot, a retry, or a parse miss).
	const gen = sceneGen;
	const bundle = meshBundle;
	if (bundle.gen === gen) {
		await bundle.streamDone;
		if (gen !== sceneGen) return;
		if (modelsById.has(event.id)) {
			applyModelVisibility(event.id);
			upsertAsset(event.id, { status: "loaded", modelUrl: event.url });
			return;
		}
	}
	_loadModelNow(event, gen);
}

// Attach a pre-fetched GLB scene to the live tree. Mirrors the second half
// of _loadModelNow (after the await) so the live and replay paths agree on
// material side-fixing, prev-model disposal, and asset bookkeeping.
function attachPreloadedGlb(event, gltf) {
	prepareLoadedScene(gltf.scene);
	gltf.scene.name = `${event.artifact_kind}:${event.id}`;
	gltf.scene.userData.pickId = event.id;
	const prevModel = modelsById.get(event.id);
	if (prevModel) {
		sceneRoot.remove(prevModel);
		disposeObject3D(prevModel);
	}
	sceneRoot.add(gltf.scene);
	modelsById.set(event.id, gltf.scene);
	applyModelVisibility(event.id);
	scheduleFitToScene();
	upsertAsset(event.id, { status: "loaded", modelUrl: event.url });
}

async function _loadModelNow(event, gen) {
	if (gen !== sceneGen) return;
	const absUrl = new URL(event.url, SERVER_URL).toString();
	// Skip a re-load when this id already errored on the *same URL* during
	// this scene generation. The server occasionally emits `model` more
	// than once for one id (anchor completion loop, cached replay), and
	// re-running GLTFLoader on a known-bad GLB just spams the same parse
	// error. A new URL or a new sceneGen still goes through.
	const prior = assets.get(event.id);
	if (prior?.status === "error" && prior.modelUrl === event.url) return;
	upsertAsset(event.id, { modelUrl: event.url });
	try {
		const gltf = await loader.loadAsync(absUrl);
		if (gen !== sceneGen) return;
		prepareLoadedScene(gltf.scene);
		gltf.scene.name = `${event.artifact_kind}:${event.id}`;
		gltf.scene.userData.pickId = event.id;
		const prevModel = modelsById.get(event.id);
		if (prevModel) {
			sceneRoot.remove(prevModel);
			disposeObject3D(prevModel);
		}
		sceneRoot.add(gltf.scene);
		modelsById.set(event.id, gltf.scene);
		applyModelVisibility(event.id);
		scheduleFitToScene();
		upsertAsset(event.id, { status: "loaded" });
	} catch (e) {
		appendEvent({ kind: "model.error", id: event.id, message: e.message });
		upsertAsset(event.id, { status: "error", errorMessage: e.message });
	}
}

// --- bbox overlays ----------------------------------------------------------

// `{ id, origin: [x,y,z], dimensions: [dx,dy,dz], proxy_shape?: ... }` —
// matches the Python BoundingBox+Node serialization. Signed and
// zero-valued dimensions are allowed (walls/floors are flat). If a proxy
// shape is set, we draw its wireframe silhouette in addition to the AABB
// wireframe so the user can see what the LLM and surface-snap are
// actually reasoning about.
function loadBbox(event) {
	const { id, origin, dimensions } = event;
	if (bboxes.has(id)) {
		const prev = bboxes.get(id);
		bboxRoot.remove(prev);
		prev.geometry?.dispose?.();
		prev.material?.dispose?.();
		if (hoveredBboxId === id) hoveredBboxId = null;
	}
	if (proxies.has(id)) {
		const prev = proxies.get(id);
		bboxRoot.remove(prev);
		prev.geometry?.dispose?.();
		prev.material?.dispose?.();
		proxies.delete(id);
	}
	disposeSolidFill(id);
	const ox = origin[0],
		oy = origin[1],
		oz = origin[2];
	const fx = ox + dimensions[0],
		fy = oy + dimensions[1],
		fz = oz + dimensions[2];
	const box3 = new THREE.Box3(
		new THREE.Vector3(Math.min(ox, fx), Math.min(oy, fy), Math.min(oz, fz)),
		new THREE.Vector3(Math.max(ox, fx), Math.max(oy, fy), Math.max(oz, fz)),
	);
	const helper = new THREE.Box3Helper(box3, BBOX_COLOR_DEFAULT);
	// Always-transparent so we can dim non-selected bboxes by adjusting opacity
	// without triggering shader recompiles (toggling `transparent` would).
	helper.material.transparent = true;
	helper.material.opacity = 1;
	helper.material.toneMapped = false; // keep the exact debug palette under ACES
	helper.userData.bboxId = id;
	const nodeKind = event.node_kind ?? "zone";
	helper.userData.nodeKind = nodeKind;
	helper.userData.proxyShape = event.proxy_shape ?? null;
	helper.userData.origin = origin;
	helper.userData.dimensions = dimensions;
	helper.userData.orientation = event.orientation ?? 0;
	bboxRoot.add(helper);
	bboxes.set(id, helper);

	const proxyMesh = buildProxyWireframe(
		event.proxy_shape,
		origin,
		dimensions,
	);
	if (proxyMesh !== null) {
		bboxRoot.add(proxyMesh);
		proxies.set(id, proxyMesh);
	}

	if (solidFillShown && nodeKind !== "zone") {
		const fill = buildSolidFill(
			event.proxy_shape,
			origin,
			dimensions,
			nodeKind,
		);
		if (fill !== null) {
			fill.userData.pickId = id;
			bboxRoot.add(fill);
			solidFills.set(id, fill);
			applySolidFillVisibility(id);
		}
	}

	// If this id is already selected (user clicked before bbox arrived, or a
	// bbox is being replaced), reapply the selection color.
	applyBboxColor(id);
	applyBboxVisibility(id);
	if (id === selectedBboxId) updateOrientationIndicator();
}

function buildProxyWireframe(proxyShape, origin, dimensions) {
	if (!proxyShape) return null;
	const sx = Math.abs(dimensions[0]);
	const sy = Math.abs(dimensions[1]);
	const sz = Math.abs(dimensions[2]);
	if (sx === 0 || sy === 0 || sz === 0) return null;
	const cx = origin[0] + dimensions[0] / 2;
	const cy = origin[1] + dimensions[1] / 2;
	const cz = origin[2] + dimensions[2] / 2;
	const yMin = Math.min(origin[1], origin[1] + dimensions[1]);

	let geom;
	let anchorY;
	if (proxyShape === "SPHERE") {
		// Ellipsoid inscribed in the AABB: unit sphere (diameter 1) scaled
		// to each AABB extent.
		geom = new THREE.SphereGeometry(0.5, 24, 16);
		geom.scale(sx, sy, sz);
		anchorY = cy;
	} else if (proxyShape === "HEMISPHERE") {
		// Top hemisphere with equatorial disk on the AABB's bottom face.
		// thetaLength = PI/2 starting at the north pole gives the upper half.
		geom = new THREE.SphereGeometry(
			0.5,
			24,
			16,
			0,
			Math.PI * 2,
			0,
			Math.PI / 2,
		);
		// The unit hemisphere spans y in [0, 0.5]; scale y by (sy / 0.5) so
		// the apex reaches +sy above the equator.
		geom.scale(sx, sy * 2, sz);
		anchorY = yMin;
	} else if (proxyShape === "CAPSULE") {
		const r = Math.min(sx, sz) / 2;
		const cylHeight = Math.max(0, sy - 2 * r);
		geom = new THREE.CapsuleGeometry(r, cylHeight, 8, 24);
		anchorY = cy;
	} else {
		return null;
	}

	const mat = new THREE.MeshBasicMaterial({
		color: BBOX_COLOR_DEFAULT,
		wireframe: true,
		transparent: true,
		opacity: PROXY_BASE_OPACITY,
		toneMapped: false,
	});
	const mesh = new THREE.Mesh(geom, mat);
	mesh.position.set(cx, anchorY, cz);
	mesh.renderOrder = 1;
	return mesh;
}

// Solid-fill counterpart to buildProxyWireframe. Same geometry math, but
// returns a lit, opaque mesh. When proxyShape is null/unset we fall back to
// a solid AABB box — types.py treats a missing proxy_shape as "the AABB is
// the proxy". `nodeKind` only steers the tint so frames read as walls vs.
// objects in the scene.
const SOLID_FILL_COLOR = {
	object: 0x4f7a45,
	frame: 0x4a6a82,
};

function buildSolidFill(proxyShape, origin, dimensions, nodeKind) {
	const sx = Math.abs(dimensions[0]);
	const sy = Math.abs(dimensions[1]);
	const sz = Math.abs(dimensions[2]);
	if (sx === 0 || sy === 0 || sz === 0) return null;
	const cx = origin[0] + dimensions[0] / 2;
	const cy = origin[1] + dimensions[1] / 2;
	const cz = origin[2] + dimensions[2] / 2;
	const yMin = Math.min(origin[1], origin[1] + dimensions[1]);

	let geom;
	let anchorY = cy;
	if (proxyShape === "SPHERE") {
		geom = new THREE.SphereGeometry(0.5, 32, 20);
		geom.scale(sx, sy, sz);
	} else if (proxyShape === "HEMISPHERE") {
		geom = new THREE.SphereGeometry(
			0.5,
			32,
			20,
			0,
			Math.PI * 2,
			0,
			Math.PI / 2,
		);
		geom.scale(sx, sy * 2, sz);
		anchorY = yMin;
	} else if (proxyShape === "CAPSULE") {
		const r = Math.min(sx, sz) / 2;
		const cylHeight = Math.max(0, sy - 2 * r);
		geom = new THREE.CapsuleGeometry(r, cylHeight, 12, 32);
	} else {
		// No proxy: fill the AABB itself as a solid box.
		geom = new THREE.BoxGeometry(sx, sy, sz);
	}

	const mat = new THREE.MeshLambertMaterial({
		color: SOLID_FILL_COLOR[nodeKind] ?? SOLID_FILL_COLOR.object,
	});
	const mesh = new THREE.Mesh(geom, mat);
	mesh.position.set(cx, anchorY, cz);
	return mesh;
}

function disposeSolidFill(id) {
	const prev = solidFills.get(id);
	if (!prev) return;
	bboxRoot.remove(prev);
	prev.geometry?.dispose?.();
	prev.material?.dispose?.();
	solidFills.delete(id);
}

function clearSolidFills() {
	for (const id of Array.from(solidFills.keys())) disposeSolidFill(id);
}

function rebuildAllSolidFills() {
	clearSolidFills();
	for (const [id, helper] of bboxes) {
		const nodeKind = helper.userData.nodeKind ?? "zone";
		if (nodeKind === "zone") continue;
		const origin = helper.userData.origin;
		const dimensions = helper.userData.dimensions;
		if (!origin || !dimensions) continue;
		const fill = buildSolidFill(
			helper.userData.proxyShape ?? null,
			origin,
			dimensions,
			nodeKind,
		);
		if (fill === null) continue;
		fill.userData.pickId = id;
		bboxRoot.add(fill);
		solidFills.set(id, fill);
		applySolidFillVisibility(id);
	}
}

// Rewind gate for the prompt-tuning sandbox: the scene is "rewound" to a step
// by hiding every object created at/after the cursor's event index. Outside
// the sandbox (`rewindCutoffIndex === null`) everything passes. An id with no
// recorded creation index (log still backfilling) is treated as present so we
// never spuriously blank a node we simply lack provenance for.
function withinRewind(id, kind) {
	if (rewindCutoffIndex === null) return true;
	const map = kind === "model" ? modelCreatedIndex : bboxCreatedIndex;
	const created = map.get(id);
	if (created === undefined) return true;
	return created < rewindCutoffIndex;
}

function applySolidFillVisibility(id) {
	const mesh = solidFills.get(id);
	if (!mesh) return;
	const isFrame = treeNodes.get(id)?.kind === "frame";
	const frameOk = isFrame ? framesShown : true;
	mesh.visible =
		frameOk && !effectivelyHidden(id) && withinRewind(id, "bbox");
}

function refreshAllSolidFillVisibility() {
	for (const id of solidFills.keys()) applySolidFillVisibility(id);
}

function applyBboxColor(id) {
	const helper = id !== null ? bboxes.get(id) : null;
	if (!helper) return;
	const base = helper.userData.proxyShape
		? BBOX_COLOR_PROXY
		: helper.userData.nodeKind === "object"
			? BBOX_COLOR_OBJECT
			: helper.userData.nodeKind === "frame"
				? BBOX_COLOR_FRAME
				: BBOX_COLOR_DEFAULT;
	const color =
		id === selectedBboxId
			? BBOX_COLOR_SELECTED
			: id === hoveredBboxId
				? BBOX_COLOR_HOVER
				: base;
	helper.material.color.setHex(color);
	const proxy = proxies.get(id);
	if (proxy) proxy.material.color.setHex(color);
}

function applyBboxVisibility(id) {
	// Bbox visibility is independent of the per-node hide state — hiding a
	// node hides only its mesh + solid fill, leaving the wireframe bbox as
	// a volumetric reference (and as the right-click handle for un-hiding).
	//
	// When something is selected, every OTHER bbox is dimmed (not hidden) so
	// the selected one stands out without losing the rest of the scene as
	// spatial reference. Hover gets full opacity so the user can see what
	// they're about to pick.
	const visible =
		(id === selectedBboxId || id === hoveredBboxId || bboxesShown) &&
		withinRewind(id, "bbox");
	const dim =
		selectedBboxId !== null &&
		id !== selectedBboxId &&
		id !== hoveredBboxId;
	const helper = bboxes.get(id);
	if (helper) {
		helper.visible = visible;
		helper.material.opacity = dim ? BBOX_DIM_OPACITY : 1;
	}
	const proxy = proxies.get(id);
	if (proxy) {
		proxy.visible = visible;
		proxy.material.opacity = dim ? PROXY_DIM_OPACITY : PROXY_BASE_OPACITY;
	}
}

function refreshAllBboxVisibility() {
	for (const id of bboxes.keys()) applyBboxVisibility(id);
}

function applyModelVisibility(id) {
	const model = modelsById.get(id);
	if (!model) return;
	const isFrame = treeNodes.get(id)?.kind === "frame";
	const frameOk = isFrame ? framesShown : true;
	model.visible =
		meshesShown &&
		frameOk &&
		!effectivelyHidden(id) &&
		withinRewind(id, "model");
}

function refreshAllFrameModelVisibility() {
	for (const id of modelsById.keys()) applyModelVisibility(id);
}

// Re-apply every visibility rule across bboxes, proxies, solid fills, and
// meshes. The sandbox calls this whenever the rewind cutoff moves so the whole
// scene snaps to the rewound state in one pass.
function refreshAllVisibility() {
	refreshAllBboxVisibility();
	refreshAllSolidFillVisibility();
	refreshAllFrameModelVisibility();
}

function setHoveredBbox(id) {
	if (id === hoveredBboxId) return;
	const prev = hoveredBboxId;
	hoveredBboxId = id;
	applyBboxColor(prev);
	applyBboxColor(id);
	if (prev !== null) applyBboxVisibility(prev);
	if (id !== null) applyBboxVisibility(id);
}

// --- orientation indicator --------------------------------------------------
// When an object is selected, an arrow from its bbox center points along the
// node's authored facing so the user can read its intended orientation. The
// facing reproduces the placement bake (server glb_place.py `_quat_y`): the
// asset's front — +Z at orientation 0 — yawed by `orientation` degrees about
// +Y, the same rotation three.js applies to the mesh, so the arrow lines up
// with the rendered object. Zones/frames carry no meaningful facing, so the
// arrow is shown for objects only. One reusable helper, repositioned per
// selection and hidden when nothing (or a non-object) is selected.
const _orientAxisY = new THREE.Vector3(0, 1, 0);
let orientationArrow = null;

function updateOrientationIndicator() {
	const helper = selectedBboxId !== null ? bboxes.get(selectedBboxId) : null;
	const origin = helper?.userData.origin;
	const dimensions = helper?.userData.dimensions;
	if (
		!helper ||
		helper.userData.nodeKind !== "object" ||
		!origin ||
		!dimensions
	) {
		if (orientationArrow) orientationArrow.visible = false;
		return;
	}
	const deg = Number(helper.userData.orientation) || 0;
	const facing = new THREE.Vector3(0, 0, 1).applyQuaternion(
		new THREE.Quaternion().setFromAxisAngle(
			_orientAxisY,
			THREE.MathUtils.degToRad(deg),
		),
	);
	const center = new THREE.Vector3(
		origin[0] + dimensions[0] / 2,
		origin[1] + dimensions[1] / 2,
		origin[2] + dimensions[2] / 2,
	);
	// Length: distance from center to the front face along `facing`, plus ~50%
	// so the head clears the wireframe. Using the exit distance (not a raw max
	// extent) keeps the arrow proportional for thin/elongated objects too.
	const hx = Math.abs(dimensions[0]) / 2;
	const hz = Math.abs(dimensions[2]) / 2;
	let reach = Infinity;
	if (Math.abs(facing.x) > 1e-6)
		reach = Math.min(reach, hx / Math.abs(facing.x));
	if (Math.abs(facing.z) > 1e-6)
		reach = Math.min(reach, hz / Math.abs(facing.z));
	if (!isFinite(reach)) reach = Math.max(hx, hz);
	const length = Math.max(0.15, reach * 1.5);
	if (!orientationArrow) {
		orientationArrow = new THREE.ArrowHelper(
			facing,
			center,
			length,
			BBOX_COLOR_ORIENT,
		);
		// Draw over the mesh/bbox so the direction stays legible even when the
		// arrow's base sits inside a solid object.
		orientationArrow.renderOrder = 3;
		for (const part of [orientationArrow.line, orientationArrow.cone]) {
			part.material.depthTest = false;
			part.material.depthWrite = false;
			part.material.transparent = true;
			part.material.toneMapped = false;
		}
		bboxRoot.add(orientationArrow);
	} else {
		orientationArrow.position.copy(center);
		orientationArrow.setDirection(facing);
		orientationArrow.setColor(BBOX_COLOR_ORIENT);
	}
	orientationArrow.setLength(length, length * 0.32, length * 0.22);
	orientationArrow.visible = true;
}

// Fit the camera to a single Box3 — parameterised variant of fitToScene.
// Used by tree-click selection.
function frameBbox(box) {
	const size = box.getSize(new THREE.Vector3());
	const center = box.getCenter(new THREE.Vector3());
	const radius = Math.max(0.5 * Math.max(size.x, size.y, size.z), 0.5);
	controls.target.copy(center);
	const dist = radius / Math.tan((camera.fov * Math.PI) / 360);
	const dirVec = new THREE.Vector3(1, 0.7, 1).normalize();
	camera.position.copy(center).addScaledVector(dirVec, dist * 1.8);
	camera.near = Math.max(0.01, radius / 100);
	camera.far = Math.max(100, radius * 100);
	camera.updateProjectionMatrix();
	controls.update();
}

function selectTreeNode(id) {
	const prev = selectedBboxId;
	// Toggle off if re-clicking the same node.
	selectedBboxId = prev === id ? null : id;
	if (prev !== null) applyBboxColor(prev);
	if (selectedBboxId !== null) applyBboxColor(selectedBboxId);
	// Selection state gates every other bbox's visibility (see
	// applyBboxVisibility) — when none→something or something→none, every
	// peer's visibility flips. A→B leaves the peers untouched, but the
	// refresh is cheap, so just re-apply uniformly.
	refreshAllBboxVisibility();
	updateOrientationIndicator();
	renderTree();
	renderTreeDetail();
	if (selectedBboxId !== null) {
		const helper = bboxes.get(selectedBboxId);
		if (helper) {
			// User took explicit camera control — stop auto-fit from later snapping
			// the view back to the full scene when new meshes land.
			cameraUserMoved = true;
			frameBbox(helper.box);
		}
	}
}

// --- detail panel -----------------------------------------------------------
// When a node is selected, the tree-body is hidden and this panel takes over
// the same slot to show the full prompts/plans/image-prompts for that node.
// The panel re-renders on every event that mutates the selected node (bbox,
// zone_plan, image, step) so it stays in sync with streamed updates.

function fmtMeters(arr) {
	if (!Array.isArray(arr) || arr.length !== 3) return "—";
	return arr
		.map((n) => (Number.isInteger(n) ? String(n) : n.toFixed(2)))
		.join(", ");
}

function ancestorChain(id) {
	const chain = [];
	let cur = treeNodes.get(id)?.parentId ?? null;
	while (cur !== null) {
		const node = treeNodes.get(cur);
		if (!node) break;
		chain.unshift(node);
		cur = node.parentId ?? null;
	}
	return chain;
}

// Persistent preview state for the detail panel. We hold onto the container
// and live WebGL viewer across `renderTreeDetail` calls so frequent re-renders
// (phase ticks, sibling bbox updates) don't reset the user's camera or thrash
// the GLB load. Rebuilt only when the selected id or its underlying urls
// change; destroyed on deselect / slot switch.
let detailPreviewState = null; // { id, modelUrl, imageUrl, container, viewer }

function destroyDetailPreview() {
	if (detailPreviewState?.viewer) {
		try {
			detailPreviewState.viewer.dispose();
		} catch {}
	}
	if (detailPreviewState?.container?.parentNode) {
		detailPreviewState.container.parentNode.removeChild(
			detailPreviewState.container,
		);
	}
	detailPreviewState = null;
}

// Stand-alone GLB viewer inside the detail panel. Lives in its own scene/
// camera/renderer (separate from the main sandbox) so spinning the preview
// doesn't move the main camera. Returns a dispose() handle.
function mountMiniViewer(container, modelUrl) {
	const scene = new THREE.Scene();
	scene.background = new THREE.Color(0x0e1014);
	const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
	// Default to a +Z-facing front view (overwritten once the model's bounds are
	// known); keeps the orientation reference consistent even pre-load.
	camera.position.set(0, 1.2, 3);
	const renderer = new THREE.WebGLRenderer({ antialias: true });
	renderer.setPixelRatio(window.devicePixelRatio || 1);
	renderer.outputColorSpace = THREE.SRGBColorSpace;
	renderer.domElement.className = "detail-mini-canvas";
	container.appendChild(renderer.domElement);
	const controls = new OrbitControls(camera, renderer.domElement);
	controls.enableDamping = true;
	controls.dampingFactor = 0.1;
	controls.enablePan = true;
	scene.add(new THREE.AmbientLight(0xffffff, 0.85));
	const dir = new THREE.DirectionalLight(0xffffff, 0.7);
	dir.position.set(5, 8, 6);
	scene.add(dir);

	function resize() {
		const w = container.clientWidth || 280;
		const h = Math.max(180, Math.round(w * 0.7));
		renderer.setSize(w, h, false);
		renderer.domElement.style.width = w + "px";
		renderer.domElement.style.height = h + "px";
		camera.aspect = w / h;
		camera.updateProjectionMatrix();
	}
	resize();
	const ro = new ResizeObserver(resize);
	ro.observe(container);

	let disposed = false;
	let model = null;
	const localLoader = configureGltfLoader(new GLTFLoader());
	localLoader
		.loadAsync(new URL(modelUrl, SERVER_URL).toString())
		.then((gltf) => {
			if (disposed) return;
			model = gltf.scene;
			prepareLoadedScene(model);
			scene.add(model);
			const box = new THREE.Box3().setFromObject(model);
			if (!box.isEmpty()) {
				const center = new THREE.Vector3();
				box.getCenter(center);
				const size = new THREE.Vector3();
				box.getSize(size);
				const maxDim = Math.max(size.x, size.y, size.z) || 1;
				const dist = maxDim * 2.8;
				// Frame from world +Z (the canonical front: +Z = toward viewer) with a
				// gentle elevation, so the world +Z axis always faces the camera. That
				// fixed reference lets the user gauge the base model's orientation —
				// azimuth is pinned to +Z (no X offset), only a slight downward tilt.
				camera.position
					.copy(center)
					.add(
						new THREE.Vector3(0, 0.4, 1)
							.normalize()
							.multiplyScalar(dist),
					);
				camera.near = Math.max(maxDim / 1000, 0.001);
				camera.far = Math.max(maxDim * 100, 100);
				camera.updateProjectionMatrix();
				controls.target.copy(center);
				controls.update();
			}
		})
		.catch(() => {
			/* keep the empty viewer; user already sees an error in the asset row */
		});

	let rafId = 0;
	function tick() {
		if (disposed) return;
		controls.update();
		renderer.render(scene, camera);
		rafId = requestAnimationFrame(tick);
	}
	tick();

	return {
		dispose() {
			disposed = true;
			cancelAnimationFrame(rafId);
			try {
				ro.disconnect();
			} catch {}
			try {
				controls.dispose();
			} catch {}
			if (model) {
				disposeObject3D(model);
			}
			try {
				renderer.dispose();
			} catch {}
			if (renderer.domElement.parentNode) {
				renderer.domElement.parentNode.removeChild(renderer.domElement);
			}
		},
	};
}

// The image + mesh the detail preview renders, resolved for the active asset
// mode. "generated" previews the selected version's from-scratch build — both
// served from generated/<version>/objects-generated-optimized/<id>.{png,glb} —
// gated on the mesh having
// attached for this cell, so the preview tracks what's on screen and never
// points at a not-yet-built file. "library" reads the folded scene projection
// / live events (the assets map), as before.
function detailPreviewUrls(node) {
	if (assetMode === "generated") {
		if (
			currentSlotId === null ||
			currentModel === null ||
			genVersion == null ||
			!modelsById.has(node.id)
		) {
			return { imageUrl: null, modelUrl: null };
		}
		// Carry the version token so a regenerate (which bumps it) changes the urls
		// → ensureDetailPreview rebuilds the image + mini-viewer with fresh bytes.
		const v = genMeshVersions.get(node.id);
		return {
			imageUrl: generatedImageUrl(
				currentSlotId,
				currentModel,
				currentRun,
				genVersion,
				node.id,
				v,
			),
			modelUrl: generatedArtifactUrl(
				currentSlotId,
				currentModel,
				currentRun,
				genVersion,
				node.id,
				v,
			),
		};
	}
	const a = assets.get(node.id);
	return {
		imageUrl: a?.imageUrl ? rerootArtifactUrl(a.imageUrl) : null,
		modelUrl:
			a && a.status === "loaded" && a.modelUrl
				? rerootArtifactUrl(a.modelUrl)
				: null,
	};
}

// Builds (or returns the cached) preview container for `node`. Returns null
// when nothing has been generated yet so the caller can skip the section.
function ensureDetailPreview(node) {
	const { imageUrl, modelUrl } = detailPreviewUrls(node);
	const modelLoaded = !!modelUrl;
	if (!imageUrl && !modelLoaded) {
		destroyDetailPreview();
		return null;
	}
	if (
		detailPreviewState &&
		detailPreviewState.id === node.id &&
		detailPreviewState.imageUrl === imageUrl &&
		detailPreviewState.modelUrl === modelUrl
	) {
		return detailPreviewState.container;
	}
	destroyDetailPreview();

	const wrap = document.createElement("div");
	wrap.className = "detail-section detail-preview";
	const label = document.createElement("div");
	label.className = "label";
	label.textContent = "preview";
	wrap.appendChild(label);

	if (imageUrl) {
		const absImg = new URL(imageUrl, SERVER_URL).toString();
		const link = document.createElement("a");
		link.className = "detail-preview-image-link";
		link.href = absImg;
		link.target = "_blank";
		link.rel = "noopener";
		const img = document.createElement("img");
		img.className = "detail-preview-image";
		img.loading = "lazy";
		img.alt = node.id;
		img.src = absImg;
		link.appendChild(img);
		wrap.appendChild(link);
	}

	let viewer = null;
	if (modelLoaded) {
		const viewerWrap = document.createElement("div");
		viewerWrap.className = "detail-preview-viewer";
		wrap.appendChild(viewerWrap);
		viewer = mountMiniViewer(viewerWrap, modelUrl);
	}

	detailPreviewState = {
		id: node.id,
		imageUrl,
		modelUrl,
		container: wrap,
		viewer,
	};
	return wrap;
}

function renderTreeDetail() {
	const id = selectedBboxId;
	if (id === null) {
		destroyDetailPreview();
		treeEl.classList.remove("detail-open");
		treeDetailEl.innerHTML = "";
		return;
	}
	const node = treeNodes.get(id);
	if (!node) {
		destroyDetailPreview();
		treeEl.classList.remove("detail-open");
		treeDetailEl.innerHTML = "";
		return;
	}
	treeEl.classList.add("detail-open");
	// Detach the live preview before wiping the panel so its WebGL canvas and
	// OrbitControls listeners survive the rebuild and we don't lose the user's
	// camera position on every event tick.
	if (detailPreviewState?.container?.parentNode === treeDetailEl) {
		treeDetailEl.removeChild(detailPreviewState.container);
	}
	treeDetailEl.textContent = "";

	// Back-to-tree button — also clears the selection so the bbox highlight
	// drops and the tree resumes its normal listing.
	const back = document.createElement("button");
	back.type = "button";
	back.className = "detail-back";
	back.textContent = "← back to tree";
	back.addEventListener("click", () => selectTreeNode(id)); // toggle off
	treeDetailEl.appendChild(back);
	// Quick jump into the full-tree modal — keeps the user's current selection
	// so they land on the same card in the larger view.
	const openFull = document.createElement("button");
	openFull.type = "button";
	openFull.className = "detail-open-full";
	openFull.textContent = "open full tree ⛶";
	openFull.title = "Open every node's prompts, plans, and bboxes in a modal";
	openFull.addEventListener("click", openTreeModal);
	treeDetailEl.appendChild(openFull);

	// Header row: kind tag + id.
	const idRow = document.createElement("div");
	idRow.className = "detail-id-row";
	const kindTag = document.createElement("span");
	kindTag.className = "detail-kind";
	kindTag.textContent = `[${node.kind ?? "zone"}]`;
	idRow.appendChild(kindTag);
	const idEl = document.createElement("span");
	idEl.className = `detail-id ${node.kind ?? "zone"}`;
	idEl.textContent = node.id;
	idRow.appendChild(idEl);
	treeDetailEl.appendChild(idRow);

	// Hierarchy — vertical, indented list of ancestors with the current node
	// at the bottom. Each ancestor row is clickable to jump selection. Shows
	// the [kind] tag and a short prompt preview so the user can see *what*
	// each parent is, not just its id.
	const chain = ancestorChain(id);
	if (chain.length > 0) {
		const hier = document.createElement("div");
		hier.className = "detail-hierarchy";
		function hierRow(n, depth, isCurrent) {
			const row = document.createElement(isCurrent ? "div" : "a");
			row.className = `detail-hier-row${isCurrent ? " current" : ""}`;
			row.style.paddingLeft = `${depth * 12}px`;
			if (!isCurrent) {
				row.addEventListener("click", (ev) => {
					ev.stopPropagation();
					selectTreeNode(n.id);
				});
			}
			if (depth > 0) {
				const branch = document.createElement("span");
				branch.className = "detail-hier-branch";
				branch.textContent = "└";
				row.appendChild(branch);
			}
			const kindEl = document.createElement("span");
			kindEl.className = `detail-hier-kind ${n.kind ?? "zone"}`;
			kindEl.textContent = `[${n.kind ?? "zone"}]`;
			row.appendChild(kindEl);
			const idEl2 = document.createElement("span");
			idEl2.className = `detail-hier-id ${n.kind ?? "zone"}`;
			idEl2.textContent = n.id;
			row.appendChild(idEl2);
			if (n.prompt) {
				const promptEl2 = document.createElement("span");
				promptEl2.className = "detail-hier-prompt";
				promptEl2.textContent = truncate(n.prompt, 50);
				promptEl2.title = n.prompt;
				row.appendChild(promptEl2);
			}
			return row;
		}
		for (let i = 0; i < chain.length; i++) {
			hier.appendChild(hierRow(chain[i], i, false));
		}
		hier.appendChild(hierRow(node, chain.length, true));
		treeDetailEl.appendChild(hier);
	}

	// Meta row: phase / bbox / proxy shape.
	const meta = document.createElement("div");
	meta.className = "detail-meta-row";
	function metaEntry(label, value) {
		const span = document.createElement("span");
		const lbl = document.createElement("span");
		lbl.textContent = `${label} `;
		span.appendChild(lbl);
		const b = document.createElement("b");
		b.textContent = value;
		span.appendChild(b);
		return span;
	}
	meta.appendChild(metaEntry("phase:", treeDisplayPhase(node.id)));
	if (Array.isArray(node.origin) && Array.isArray(node.dimensions)) {
		meta.appendChild(metaEntry("origin:", `[${fmtMeters(node.origin)}]`));
		meta.appendChild(
			metaEntry("size:", `[${fmtMeters(node.dimensions)}] m`),
		);
	}
	if (node.proxyShape) {
		meta.appendChild(metaEntry("proxy:", node.proxyShape));
	}
	treeDetailEl.appendChild(meta);

	// Prompt / plan / image-prompt sections — labelled so a reader can tell
	// which pipeline step authored each piece of text.
	function section(label, text) {
		const wrap = document.createElement("div");
		wrap.className = "detail-section";
		const lab = document.createElement("div");
		lab.className = "label";
		lab.textContent = label;
		wrap.appendChild(lab);
		const body = document.createElement("div");
		body.className = "body";
		if (text) {
			body.textContent = text;
		} else {
			body.classList.add("detail-empty");
			body.textContent = "(not yet authored)";
		}
		wrap.appendChild(body);
		treeDetailEl.appendChild(wrap);
	}

	section("seed prompt", node.prompt);
	if (node.kind === "zone" || node.plan) {
		section("zone plan", node.plan);
	}
	if (node.kind !== "zone") {
		section("image prompt", node.imagePrompt);
	}

	// Retry control — only meaningful for non-zone meshes that errored (or are
	// mid-retry from this client). Zones never produce meshes, so a retry would
	// have nothing to regenerate.
	if (node.kind !== "zone") {
		const a = assets.get(id);
		const status = a ? assetStatus(a) : "pending";
		if (status === "error" || retryingIds.has(id)) {
			const wrap = document.createElement("div");
			wrap.className = "detail-section";
			syncRetryButton(wrap, id, status, "detail-retry");
			treeDetailEl.appendChild(wrap);
		}
	}

	// Regenerate control — generated mode only. Re-rolls this one generated asset
	// from scratch (Nano-Banana + Trellis + optimize); the gate poll swaps the new
	// mesh in once it lands. Available for any non-zone node, including ones whose
	// generated asset failed or hasn't been built yet.
	if (
		assetMode === "generated" &&
		genVersion != null &&
		node.kind !== "zone"
	) {
		const regenerating = regeneratingIds.has(id);
		const unsymmetrizing = unsymmetrizingIds.has(id);
		const symmetrizing = symmetrizingIds.has(id);
		// Any in-flight rebuild op on this node disables all of its action buttons, so
		// the user can't fire two at once; each button keeps its own in-flight label.
		const busy = regenerating || unsymmetrizing || symmetrizing;
		// Symmetry state for this asset (from the gate poll). undefined = not
		// reported yet (mesh not built). Otherwise { plane, was }: plane xy/xz =
		// mirrored; plane none WITH a `was` = un-symmetrized (was mirrored across
		// `was`); plane none with no `was` = never symmetrized.
		const symInfo = genMeshSymmetry.get(id);
		const sym = symInfo?.plane;
		const symWas = symInfo?.was ?? null;
		const mirrored = sym === "xy" || sym === "xz";
		const unsymmetrized = sym === "none" && symWas != null;

		// Per-asset optimize override (view-only) — shown once this asset has a
		// mesh on screen to swap. Flips just this asset between its optimized twin
		// and the raw, un-optimized Trellis mesh; the rest of the scene stays on
		// the scene-wide `optimized` setting, and the override resets whenever the
		// scene re-streams. Disabled while a swap or a rebuild of this node is in
		// flight so the two reloads can't race.
		if (modelsById.has(id)) {
			const optimized = meshOptimizedFor(id);
			const overridden = genMeshOptimized.has(id);
			const swapping = _genLoading.has(id);
			const optWrap = document.createElement("div");
			optWrap.className = "detail-section";
			const optBtn = document.createElement("button");
			optBtn.type = "button";
			optBtn.className = "detail-optimize" + (optimized ? "" : " raw");
			optBtn.disabled = swapping || busy;
			optBtn.textContent = swapping
				? "swapping mesh…"
				: `mesh: ${optimized ? "optimized" : "raw"}${overridden ? " · override" : ""}`;
			optBtn.title = optimized
				? "This asset is showing its optimized twin (decimated + KTX2 + Meshopt). Click to load just this asset's raw, un-optimized Trellis mesh (~100× heavier) — the rest of the scene stays on the scene-wide optimized setting."
				: "This asset is showing its raw, un-optimized Trellis mesh (~100× heavier). Click to load just this asset's optimized twin instead.";
			optBtn.addEventListener("click", (ev) => {
				ev.stopPropagation();
				toggleMeshOptimized(id);
			});
			optWrap.appendChild(optBtn);
			treeDetailEl.appendChild(optWrap);
		}

		// Regenerate-source toggle: "from scratch" (new image + mesh) vs "from
		// image" (reuse the existing image, rebuild only the mesh). Applies to all
		// three backend regenerate buttons below; un-symmetrize is unaffected.
		const srcWrap = document.createElement("div");
		srcWrap.className = "detail-section";
		const srcToggle = document.createElement("button");
		srcToggle.type = "button";
		srcToggle.className =
			"detail-regen-source" + (regenReuseImage ? " from-image" : "");
		srcToggle.textContent = regenReuseImage
			? "regen source: from image"
			: "regen source: from scratch";
		srcToggle.title = regenReuseImage
			? "Regenerate reuses this asset's existing image and rebuilds only the mesh (no new Nano-Banana). Click to switch to from-scratch."
			: "Regenerate makes a new Nano-Banana image AND a new mesh. Click to reuse the existing image instead (rebuild the mesh only).";
		srcToggle.addEventListener("click", (ev) => {
			ev.stopPropagation();
			regenReuseImage = !regenReuseImage;
			try {
				localStorage.setItem(
					REGEN_SOURCE_STORAGE_KEY,
					regenReuseImage ? "image" : "scratch",
				);
			} catch {}
			renderTreeDetail();
		});
		srcWrap.appendChild(srcToggle);
		treeDetailEl.appendChild(srcWrap);

		const wrap = document.createElement("div");
		wrap.className = "detail-section";
		// One button per mesh backend (Trellis | Hunyuan Omni | Hunyuan 3.1). The
		// symmetry actions live in their own section below; all share the per-id gate.
		const makeRegenButton = (label, backend, backendLabel) => {
			const btn = document.createElement("button");
			btn.type = "button";
			btn.className = "detail-retry";
			btn.classList.toggle("retrying", busy);
			btn.disabled = busy;
			btn.textContent = regenerating ? "regenerating…" : label;
			btn.title = regenerating
				? `Re-running Nano-Banana + ${backendLabel} for this generated asset`
				: `Re-run Nano-Banana + ${backendLabel} for this asset and re-derive every object sharing its prefab mesh`;
			btn.addEventListener("click", (ev) => {
				ev.stopPropagation();
				regenerateAsset(id, backend);
			});
			return btn;
		};
		wrap.appendChild(
			makeRegenButton("regenerate (trellis)", "trellis", "Trellis"),
		);
		wrap.appendChild(
			makeRegenButton("regenerate (hunyuan)", "hunyuan", "Hunyuan"),
		);
		// Hunyuan 3D 3.1 Rapid via Tencent's direct API — paced one-at-a-time
		// server-side (Tencent's per-account concurrent-task limit is 1).
		wrap.appendChild(
			makeRegenButton(
				"regenerate (hunyuan 3.1)",
				"hunyuan-tencent",
				"Hunyuan 3.1",
			),
		);
		treeDetailEl.appendChild(wrap);
		// Symmetry readout: whether this asset's served mesh is currently mirrored
		// (and across which plane) or showing its full, un-mirrored geometry. Shown
		// once the gate poll has reported this built asset's state.
		if (sym !== undefined) {
			const symWrap = document.createElement("div");
			symWrap.className = "detail-section";
			const symLab = document.createElement("div");
			symLab.className = "label";
			symLab.textContent = "symmetry";
			const symBody = document.createElement("div");
			symBody.className = "body";
			symBody.textContent = mirrored
				? `mirrored across ${sym.toUpperCase()} plane`
				: unsymmetrized
					? `un-symmetrized (was ${symWas.toUpperCase()} plane)`
					: "none (never symmetrized)";
			symWrap.appendChild(symLab);
			symWrap.appendChild(symBody);
			treeDetailEl.appendChild(symWrap);
		}
		// Symmetry action — its own section, separate from the regenerate buttons.
		// Mirrored → un-symmetrize (reveal the full un-mirrored mesh, no AI). Not
		// mirrored (un-symmetrized or never) → a symmetrize control: pick a plane +
		// kept half and mirror it (no AI, no symmetry LLM call). Both reprocess the
		// existing raw mesh and propagate across the prefab group. While `sym` is
		// unreported (mesh not built yet) we optimistically offer the symmetrize
		// control; the server skips gracefully if no raw exists.
		const symActWrap = document.createElement("div");
		symActWrap.className = "detail-section";
		if (mirrored) {
			const unBtn = document.createElement("button");
			unBtn.type = "button";
			unBtn.className = "detail-retry detail-unsymmetrize";
			unBtn.classList.toggle("retrying", busy);
			unBtn.disabled = busy;
			unBtn.textContent = unsymmetrizing
				? "un-symmetrizing…"
				: `un-symmetrize (${sym})`;
			unBtn.title = unsymmetrizing
				? "Rebuilding this asset's mesh from its original un-mirrored model"
				: `Currently mirrored across the ${sym.toUpperCase()} plane. Reveal the full, un-mirrored mesh (no AI calls). Propagates across the prefab group.`;
			unBtn.addEventListener("click", (ev) => {
				ev.stopPropagation();
				unsymmetrizeAsset(id);
			});
			symActWrap.appendChild(unBtn);
		} else {
			// Plane toggle (XY ↔ XZ) + kept-half toggle (+ ↔ −), both sticky across
			// re-renders via the module vars, then the symmetrize action button.
			const planeBtn = document.createElement("button");
			planeBtn.type = "button";
			planeBtn.className = "detail-regen-source";
			planeBtn.disabled = busy;
			planeBtn.textContent = `symmetrize plane: ${symmetrizePlane.toUpperCase()}`;
			planeBtn.title =
				symmetrizePlane === "xy"
					? "Mirror across the XY plane (front/back along Z). Click to switch to XZ (top/bottom along Y)."
					: "Mirror across the XZ plane (top/bottom along Y). Click to switch to XY (front/back along Z).";
			planeBtn.addEventListener("click", (ev) => {
				ev.stopPropagation();
				symmetrizePlane = symmetrizePlane === "xy" ? "xz" : "xy";
				renderTreeDetail();
			});
			symActWrap.appendChild(planeBtn);

			const dirBtn = document.createElement("button");
			dirBtn.type = "button";
			dirBtn.className = "detail-regen-source";
			dirBtn.disabled = busy;
			const dirLabel =
				symmetrizePlane === "xy"
					? symmetrizeKeepPositive
						? "keep front (+Z)"
						: "keep back (−Z)"
					: symmetrizeKeepPositive
						? "keep top (+Y)"
						: "keep bottom (−Y)";
			dirBtn.textContent = `direction: ${dirLabel}`;
			dirBtn.title =
				"Which half of the mesh to keep and mirror onto the other side. Click to flip.";
			dirBtn.addEventListener("click", (ev) => {
				ev.stopPropagation();
				symmetrizeKeepPositive = !symmetrizeKeepPositive;
				renderTreeDetail();
			});
			symActWrap.appendChild(dirBtn);

			const symBtn = document.createElement("button");
			symBtn.type = "button";
			symBtn.className = "detail-retry detail-symmetrize";
			symBtn.classList.toggle("retrying", busy);
			symBtn.disabled = busy;
			symBtn.textContent = symmetrizing
				? "symmetrizing…"
				: `symmetrize (${symmetrizePlane.toUpperCase()})`;
			symBtn.title = symmetrizing
				? "Mirroring this asset's mesh across the chosen plane"
				: `Mirror this asset across the ${symmetrizePlane.toUpperCase()} plane, ${
						symmetrizeKeepPositive
							? "keeping the +half"
							: "keeping the −half"
					} (no AI calls). Propagates across the prefab group.`;
			symBtn.addEventListener("click", (ev) => {
				ev.stopPropagation();
				symmetrizeAsset(id, symmetrizePlane, symmetrizeKeepPositive);
			});
			symActWrap.appendChild(symBtn);
		}
		treeDetailEl.appendChild(symActWrap);
	}

	// Generated image + interactive mini 3D viewer when this node has assets.
	// Built (or reused) by ensureDetailPreview so the WebGL context isn't torn
	// down on incidental re-renders.
	const preview = ensureDetailPreview(node);
	if (preview) treeDetailEl.appendChild(preview);

	// Children list — for zones with sub-zones, lets the user drill into a
	// child without leaving the detail view.
	const childIds = treeChildren.get(id) ?? [];
	if (childIds.length > 0) {
		const wrap = document.createElement("div");
		wrap.className = "detail-section";
		const lab = document.createElement("div");
		lab.className = "label";
		lab.textContent = `children (${childIds.length})`;
		wrap.appendChild(lab);
		const list = document.createElement("div");
		list.className = "detail-children";
		for (const cid of childIds) {
			const a = document.createElement("a");
			const cn = treeNodes.get(cid);
			a.textContent = cn
				? `${cn.id} — ${truncate(cn.prompt ?? "", 60)}`
				: cid;
			a.addEventListener("click", () => selectTreeNode(cid));
			list.appendChild(a);
		}
		wrap.appendChild(list);
		treeDetailEl.appendChild(wrap);
	}

	treeDetailEl.scrollTop = 0;
}

// --- pipeline observability modal ------------------------------------------
// Focused on ONE node at a time. The modal shows that node's full LLM call
// trace (system instruction + user input + output + reasoning for every
// pipeline step that touched it) alongside the same trace for every
// ancestor and every descendant. This is the "why did the LLM author this?"
// view — not a clone of the sidebar tree.
//
// The focus follows `selectedBboxId`. Clicking any id inside the modal
// re-selects + re-focuses. With nothing selected we focus the root so the
// modal still shows something useful when the user just wants to read the
// scene plan.

let treeModalOpen = false;
let treeModalQuery = "";
// `treeModalFocusId` lets the modal hold its own focus without clobbering
// the scene selection. Pinned only via the in-modal id-click; otherwise we
// follow `selectedBboxId` (or root as a final fallback).
let treeModalFocusId = null;
// User-controlled expand/collapse overrides per node, so re-renders during
// a streaming run don't snap their `<details>` shut. Default per-node state
// derives from role (ancestor/focus/descendant); this map only stores the
// deltas the user toggled.
const treeModalNodeOpen = new Map(); // id -> bool

function modalFocusId() {
	if (treeModalFocusId && treeNodes.has(treeModalFocusId)) {
		return treeModalFocusId;
	}
	if (selectedBboxId && treeNodes.has(selectedBboxId)) {
		return selectedBboxId;
	}
	return treeRootId;
}

function openTreeModal() {
	treeModalOpen = true;
	// Snap focus to the current selection on open, so the modal opens on
	// the node the user was looking at — not whatever they last pinned in
	// a previous modal session.
	treeModalFocusId = selectedBboxId ?? treeRootId;
	treeModalNodeOpen.clear();
	treeModalEl.classList.add("open");
	renderTreeModal();
	setTimeout(() => treeModalSearchEl?.focus(), 0);
}

function closeTreeModal() {
	treeModalOpen = false;
	treeModalEl.classList.remove("open");
}

function focusModalOn(id) {
	if (!treeNodes.has(id)) return;
	treeModalFocusId = id;
	// Auto-open the new focus so the user immediately sees its calls.
	treeModalNodeOpen.set(id, true);
	if (selectedBboxId !== id) selectTreeNode(id);
	renderTreeModal();
	// Scroll the focused card into view so the layout shift doesn't strand
	// the user at the top of the modal.
	requestAnimationFrame(() => {
		const target = treeModalBodyEl.querySelector(
			`.tm-card[data-id="${CSS.escape(id)}"]`,
		);
		if (target) target.scrollIntoView({ block: "center" });
	});
}

function descendantsDFS(id) {
	// (node, depth-from-focus) pairs, focus excluded. Used to render the
	// descendant section below the focused card.
	const out = [];
	const start = treeChildren.get(id) ?? [];
	const stack = start.map((cid) => [cid, 1]);
	while (stack.length) {
		const [cid, depth] = stack.shift();
		const node = treeNodes.get(cid);
		if (!node) continue;
		out.push([node, depth]);
		const kids = treeChildren.get(cid) ?? [];
		for (const k of kids) stack.push([k, depth + 1]);
	}
	return out;
}

// `<pre>` block that holds an LLM prompt verbatim. Wrapped, scrollable when
// large, capped in height so the modal layout doesn't get blown out by a
// 4k-line user prompt — the user can scroll inside if they want more.
function preBlock(text, { mono = true, cap = "320px" } = {}) {
	const pre = document.createElement("pre");
	pre.className = mono ? "tm-pre mono" : "tm-pre";
	pre.style.maxHeight = cap;
	pre.textContent = text ?? "";
	return pre;
}

function copyButton(text, label = "copy") {
	const b = document.createElement("button");
	b.type = "button";
	b.className = "tm-copy";
	b.textContent = label;
	b.title = "copy to clipboard";
	b.addEventListener("click", async (ev) => {
		ev.stopPropagation();
		try {
			await navigator.clipboard.writeText(text ?? "");
			b.textContent = "copied";
			setTimeout(() => {
				b.textContent = label;
			}, 900);
		} catch {
			b.textContent = "copy failed";
			setTimeout(() => {
				b.textContent = label;
			}, 1200);
		}
	});
	return b;
}

function llmCallBlock(call, { defaultOpen, query = "" }) {
	const filtering = query.length > 0;

	// Output is pretty-printed JSON because the structured-output schema is what
	// the rest of the pipeline consumes; match against the same text the user sees.
	const outText =
		call.output === null
			? "(no output)"
			: (() => {
					try {
						return JSON.stringify(call.output, null, 2);
					} catch {
						return String(call.output);
					}
				})();

	// When searching, only the boxes containing the term render. A call with no
	// matching box is dropped entirely (return null) so "calls issued from this
	// node" / provenance collapse to just the hits.
	const sysHit = !!call.system && call.system.toLowerCase().includes(query);
	const userHit = !!call.user && call.user.toLowerCase().includes(query);
	const outHit = outText.toLowerCase().includes(query);
	const reasHit =
		!!call.reasoning && call.reasoning.toLowerCase().includes(query);
	if (filtering && !sysHit && !userHit && !outHit && !reasHit) return null;

	const det = document.createElement("details");
	det.className = "tm-llm-call";
	if (call.cached) det.classList.add("cached");
	if (call.uid != null) det.dataset.callKey = String(call.uid);
	det.open = filtering ? true : defaultOpen;

	const sum = document.createElement("summary");
	sum.className = "tm-llm-summary";
	const stepEl = document.createElement("span");
	stepEl.className = "tm-llm-step";
	stepEl.textContent = call.step;
	sum.appendChild(stepEl);
	if (call.model) {
		const modelEl = document.createElement("span");
		modelEl.className = "tm-llm-model";
		modelEl.textContent = call.model;
		sum.appendChild(modelEl);
	}
	if (call.cached) {
		const tag = document.createElement("span");
		tag.className = "tm-llm-tag cached";
		tag.textContent = "cached";
		tag.title =
			"Output reused from a previous identical (model, system, user) call in this run.";
		sum.appendChild(tag);
	}
	det.appendChild(sum);

	function section(label, body, extra) {
		const wrap = document.createElement("div");
		wrap.className = "tm-llm-section";
		const head = document.createElement("div");
		head.className = "tm-llm-section-head";
		const lab = document.createElement("span");
		lab.className = "tm-llm-section-label";
		lab.textContent = label;
		head.appendChild(lab);
		if (extra) head.appendChild(extra);
		wrap.appendChild(head);
		wrap.appendChild(body);
		return wrap;
	}

	// System instruction — the prompt that defines the LLM's role for this
	// step. Usually long-ish boilerplate; collapsed by default so the user
	// sees the dynamic input + output first (opened when it's the search hit).
	if (!filtering || sysHit) {
		const sysWrap = document.createElement("details");
		sysWrap.className = "tm-llm-subblock";
		if (sysHit) sysWrap.open = true;
		const sysSum = document.createElement("summary");
		sysSum.textContent = `system instruction (${call.system.length.toLocaleString()} chars)`;
		sysWrap.appendChild(sysSum);
		sysWrap.appendChild(preBlock(call.system, { cap: "260px" }));
		sysWrap.appendChild(copyButton(call.system, "copy system"));
		det.appendChild(section("system", sysWrap));
	}

	// User input — the actual rendered context for this call. This is the
	// most-changing part call-to-call so it's open by default inside the
	// already-open detail block.
	if (!filtering || userHit) {
		const userWrap = document.createElement("div");
		userWrap.appendChild(preBlock(call.user, { cap: "360px" }));
		userWrap.appendChild(copyButton(call.user, "copy input"));
		det.appendChild(
			section(
				`input (${call.user.length.toLocaleString()} chars)`,
				userWrap,
			),
		);
	}

	// Output — pretty-print JSON. We render with json formatting because the
	// structured-output schema is what the rest of the pipeline consumes.
	if (!filtering || outHit) {
		const outWrap = document.createElement("div");
		outWrap.appendChild(preBlock(outText, { cap: "360px" }));
		outWrap.appendChild(copyButton(outText, "copy output"));
		det.appendChild(section("output", outWrap));
	}

	// Reasoning is optional (only present when the provider returns CoT).
	// Collapsed by default — it's noisy (opened when it's the search hit).
	if (call.reasoning && (!filtering || reasHit)) {
		const reasWrap = document.createElement("details");
		reasWrap.className = "tm-llm-subblock";
		if (reasHit) reasWrap.open = true;
		const reasSum = document.createElement("summary");
		reasSum.textContent = `reasoning (${call.reasoning.length.toLocaleString()} chars)`;
		reasWrap.appendChild(reasSum);
		reasWrap.appendChild(preBlock(call.reasoning, { cap: "260px" }));
		det.appendChild(section("reasoning", reasWrap));
	}

	return det;
}

function renderObsCard(node, { role, depth, query = "" }) {
	// role: "ancestor" | "focus" | "descendant"
	// depth is only used to indent descendants visually.
	// When `query` is set the card is filtered down to the text boxes that
	// contain the term; if nothing in it matches we bail and return null so the
	// caller drops the card (and its section heading) entirely.
	const filtering = query.length > 0;
	if (filtering && !nodeMatchesQuery(node, query)) return null;

	const card = document.createElement("div");
	const kind = node.kind ?? "zone";
	card.className = `tm-card tm-obs-card kind-${kind} role-${role}`;
	card.dataset.id = node.id;
	if (role === "focus") card.classList.add("selected");
	if (role === "descendant" && depth > 0) {
		card.style.marginLeft = `${Math.min(depth, 6) * 16}px`;
	}

	// Decide default open/closed. The focused node opens by default; ancestors
	// and descendants stay closed so the focus stays visually dominant.
	// User-toggled state takes precedence — but a live query forces open so the
	// matching boxes are visible without extra clicks.
	const userPref = treeModalNodeOpen.get(node.id);
	const detailsOpen = filtering
		? true
		: userPref !== undefined
			? userPref
			: role === "focus";

	const det = document.createElement("details");
	det.className = "tm-obs-details";
	det.open = detailsOpen;
	det.addEventListener("toggle", () => {
		// Don't persist open/closed while a search is forcing matches open — that
		// would clobber the user's pre-search expand state, which we restore on clear.
		if (treeModalQuery.trim()) return;
		treeModalNodeOpen.set(node.id, det.open);
	});

	const sum = document.createElement("summary");
	sum.className = "tm-obs-summary";

	const roleEl = document.createElement("span");
	roleEl.className = `tm-obs-role role-${role}`;
	roleEl.textContent =
		role === "focus"
			? "focus"
			: role === "ancestor"
				? "ancestor"
				: `child ·d${depth}`;
	sum.appendChild(roleEl);

	const kindEl = document.createElement("span");
	kindEl.className = "tm-kind";
	kindEl.textContent = `[${kind}]`;
	sum.appendChild(kindEl);

	const idEl = document.createElement("span");
	idEl.className = `tm-id ${kind}`;
	idEl.textContent = node.id;
	idEl.title = "Click to focus this node";
	idEl.addEventListener("click", (ev) => {
		ev.stopPropagation();
		ev.preventDefault();
		focusModalOn(node.id);
	});
	sum.appendChild(idEl);

	const phaseEl = document.createElement("span");
	const displayPhase = treeDisplayPhase(node.id);
	phaseEl.className = `tm-phase phase-${displayPhase}`;
	phaseEl.textContent = displayPhase;
	sum.appendChild(phaseEl);

	const calls = nodeLlmCalls.get(node.id) ?? [];
	const callsTag = document.createElement("span");
	callsTag.className = "tm-obs-callcount";
	callsTag.textContent =
		calls.length === 1 ? "1 llm call" : `${calls.length} llm calls`;
	sum.appendChild(callsTag);

	// "via {step}" pill — surfaces which decomposition step admitted this
	// node at a glance, so the user can immediately see anchor vs next_object
	// vs encapsulating vs negative_space vs zone_decompose, without expanding.
	// Pulled from the provenance trace (first emitted_by entry).
	const provenance = nodeProvenance.get(node.id) ?? [];
	const emittedBy = provenance.find((p) => p.relation === "emitted_by");
	if (emittedBy) {
		const via = document.createElement("span");
		via.className = `tm-obs-via via-${emittedBy.call.step}`;
		via.textContent = `via ${emittedBy.call.step}`;
		via.title = `Emitted by ${emittedBy.call.step} call on ${emittedBy.call.parentNode || "?"}`;
		sum.appendChild(via);
	}

	if (node.prompt) {
		const promptTeaser = document.createElement("span");
		promptTeaser.className = "tm-obs-teaser";
		promptTeaser.textContent = truncate(node.prompt, 80);
		promptTeaser.title = node.prompt;
		sum.appendChild(promptTeaser);
	}

	det.appendChild(sum);

	// Body: the per-node prompts captured by the pipeline (seed/plan/image)
	// plus every llm.call recorded against this node.
	const body = document.createElement("div");
	body.className = "tm-obs-body";

	// Count every box left after filtering so the card can drop itself when a
	// query matches nothing inside it (despite passing the cheap node-level bail).
	let visibleBoxes = 0;

	function addTextSection(label, text) {
		if (!text) return;
		if (filtering && !text.toLowerCase().includes(query)) return;
		const wrap = document.createElement("div");
		wrap.className = "tm-section";
		const lab = document.createElement("div");
		lab.className = "tm-section-label";
		lab.textContent = label;
		wrap.appendChild(lab);
		const b = document.createElement("div");
		b.className = "tm-section-body";
		b.textContent = text;
		wrap.appendChild(b);
		body.appendChild(wrap);
		visibleBoxes++;
	}

	addTextSection("seed prompt", node.prompt);
	if (kind === "zone" && node.plan) addTextSection("zone plan", node.plan);
	if (kind !== "zone" && node.imagePrompt)
		addTextSection("image prompt", node.imagePrompt);

	// Provenance — the LLM calls that *brought this node into existence*.
	// Distinct from the calls section below (which is calls *issued from* this
	// node). For an object, this is the only place to see the anchor_decompose
	// / next_object / object_bbox_batch calls that named & placed it — those
	// calls live on the parent zone's id. Surfaced as full collapsible call
	// blocks so the user can read the exact system + user + output that drove
	// the placement decision.
	if (provenance.length > 0) {
		const provWrap = document.createElement("div");
		provWrap.className = "tm-obs-provenance";
		const lab = document.createElement("div");
		lab.className = "tm-section-label tm-obs-provenance-label";
		lab.textContent = "provenance — calls that emitted & placed this node";
		provWrap.appendChild(lab);
		const callDefaultOpen = role === "focus";
		let shownProv = 0;
		for (const entry of provenance) {
			const block = llmCallBlock(entry.call, {
				defaultOpen: callDefaultOpen,
				query,
			});
			if (!block) continue; // filtered out — drop its relation line too
			const relLine = document.createElement("div");
			relLine.className = "tm-obs-provenance-rel";
			const rel = document.createElement("span");
			rel.className = `tm-obs-provenance-tag rel-${entry.relation}`;
			rel.textContent =
				entry.relation === "emitted_by" ? "emitted by" : "placed by";
			relLine.appendChild(rel);
			const onNode = document.createElement("a");
			onNode.className = "tm-obs-provenance-on";
			onNode.textContent = `${entry.call.step} on ${entry.call.parentNode || "?"}`;
			if (entry.call.parentNode && treeNodes.has(entry.call.parentNode)) {
				onNode.title = `Click to focus ${entry.call.parentNode}`;
				onNode.addEventListener("click", (ev) => {
					ev.stopPropagation();
					focusModalOn(entry.call.parentNode);
				});
			} else {
				onNode.classList.add("dead");
			}
			relLine.appendChild(onNode);
			provWrap.appendChild(relLine);
			provWrap.appendChild(block);
			shownProv++;
		}
		if (shownProv > 0) {
			body.appendChild(provWrap);
			visibleBoxes += shownProv;
		}
	}

	// LLM call traces — calls issued *from* this node (zone_plan,
	// zone_decompose, anchor_decompose, image_prompt, etc).
	const callsWrap = document.createElement("div");
	callsWrap.className = "tm-llm-calls";
	const callsHeader = document.createElement("div");
	callsHeader.className = "tm-section-label tm-llm-calls-label";
	callsHeader.textContent = "calls issued from this node";
	callsWrap.appendChild(callsHeader);
	let shownCalls = 0;
	if (calls.length === 0) {
		// No calls at all — keep the explanatory placeholder only when not
		// filtering (a search shouldn't surface an empty section).
		if (!filtering) {
			const empty = document.createElement("div");
			empty.className = "tm-llm-empty";
			empty.textContent = "(no LLM calls issued from this node)";
			callsWrap.appendChild(empty);
		}
	} else {
		// Focus card opens every call; ancestors/descendants leave them collapsed.
		const callDefaultOpen = role === "focus";
		for (const call of calls) {
			const block = llmCallBlock(call, {
				defaultOpen: callDefaultOpen,
				query,
			});
			if (!block) continue;
			callsWrap.appendChild(block);
			shownCalls++;
		}
	}
	// When filtering, only surface the calls section if it has a hit; otherwise
	// keep it (it carries the always-on section / "(no LLM calls…)" placeholder).
	if (!filtering || shownCalls > 0) {
		body.appendChild(callsWrap);
		visibleBoxes += shownCalls;
	}

	det.appendChild(body);
	card.appendChild(det);

	// A query that matched the node-level bail but no actual box (e.g. matched
	// only output's compact JSON, not the pretty form shown) leaves nothing to
	// read — drop the card so only boxes containing the term remain.
	if (filtering && visibleBoxes === 0) return null;
	return card;
}

function renderModalSectionHeading(label, count, hint) {
	const h = document.createElement("div");
	h.className = "tm-obs-heading";
	const lab = document.createElement("span");
	lab.className = "tm-obs-heading-label";
	lab.textContent = label;
	h.appendChild(lab);
	const c = document.createElement("span");
	c.className = "tm-obs-heading-count";
	c.textContent = count === 1 ? "1 node" : `${count} nodes`;
	h.appendChild(c);
	if (hint) {
		const hi = document.createElement("span");
		hi.className = "tm-obs-heading-hint";
		hi.textContent = hint;
		h.appendChild(hi);
	}
	return h;
}

// Box-level inclusion tests for the modal search. A search filters the modal
// down to the text boxes that actually contain the term, so these only consider
// box content — system / input / output / reasoning for calls, and the per-node
// seed / plan / image prompts. id / step / model labels are deliberately
// excluded, since they aren't text boxes. The authoritative per-box filtering
// lives in llmCallBlock + renderObsCard; these just let a card bail before it
// builds its DOM (and the final empty-card guard keeps the result exact).
function callMatchesQuery(call, q) {
	return (
		(!!call.system && call.system.toLowerCase().includes(q)) ||
		(!!call.user && call.user.toLowerCase().includes(q)) ||
		(!!call.reasoning && call.reasoning.toLowerCase().includes(q)) ||
		JSON.stringify(call.output ?? "")
			.toLowerCase()
			.includes(q)
	);
}

function nodeMatchesQuery(node, q) {
	if ((node.prompt ?? "").toLowerCase().includes(q)) return true;
	if ((node.plan ?? "").toLowerCase().includes(q)) return true;
	if ((node.imagePrompt ?? "").toLowerCase().includes(q)) return true;
	const calls = nodeLlmCalls.get(node.id) ?? [];
	if (calls.some((c) => callMatchesQuery(c, q))) return true;
	const prov = nodeProvenance.get(node.id) ?? [];
	return prov.some((p) => callMatchesQuery(p.call, q));
}

function renderTreeModal() {
	if (!treeModalOpen) return;
	// Re-renders happen on every streamed event while the modal is open. Plain
	// `scrollTop` preservation isn't enough — when new LLM calls land on the
	// focused card (or any card above the user's viewport), absolute pixel
	// positions shift downward and the user ends up looking at a different
	// card. Anchor the restore to whichever card is currently closest to the
	// top of the viewport, then snap that same card back to the same offset
	// after the rebuild.
	const bodyRectTop = treeModalBodyEl.getBoundingClientRect().top;
	let anchorId = null;
	let anchorOffset = 0;
	for (const card of treeModalBodyEl.querySelectorAll(".tm-card")) {
		const r = card.getBoundingClientRect();
		const offset = r.top - bodyRectTop;
		// Pick the first card whose top is at or below the viewport top — that's
		// the one the user is reading. Bail once we pass it; cards lower down
		// can't be a better anchor.
		if (offset >= -2) {
			anchorId = card.dataset.id;
			anchorOffset = offset;
			break;
		}
	}
	treeModalBodyEl.innerHTML = "";

	if (treeNodes.size === 0) {
		const empty = document.createElement("div");
		empty.className = "tm-empty";
		empty.textContent =
			"no nodes yet — start a run to populate observability";
		treeModalBodyEl.appendChild(empty);
		return;
	}

	const focusId = modalFocusId();
	if (!focusId) {
		const empty = document.createElement("div");
		empty.className = "tm-empty";
		empty.textContent =
			"no focused node — click a node in the sidebar tree";
		treeModalBodyEl.appendChild(empty);
		return;
	}
	const focusNode = treeNodes.get(focusId);
	if (!focusNode) return;

	const ancestors = ancestorChain(focusId); // root → ... → parent
	const descendants = descendantsDFS(focusId); // [[node, depth], ...]

	const q = treeModalQuery.trim().toLowerCase();

	// Header strip: shows the active focus path so the user has a breadcrumb
	// back to the root even when the ancestors section is collapsed.
	const breadcrumb = document.createElement("div");
	breadcrumb.className = "tm-obs-breadcrumb";
	const crumbLabel = document.createElement("span");
	crumbLabel.className = "tm-obs-crumb-label";
	crumbLabel.textContent = "path:";
	breadcrumb.appendChild(crumbLabel);
	const chain = [...ancestors, focusNode];
	chain.forEach((n, i) => {
		if (i > 0) {
			const sep = document.createElement("span");
			sep.className = "tm-obs-crumb-sep";
			sep.textContent = "›";
			breadcrumb.appendChild(sep);
		}
		const link = document.createElement("a");
		link.className = `tm-obs-crumb ${n.kind ?? "zone"}${n.id === focusId ? " current" : ""}`;
		link.textContent = n.id;
		link.addEventListener("click", (ev) => {
			ev.preventDefault();
			focusModalOn(n.id);
		});
		breadcrumb.appendChild(link);
	});
	treeModalBodyEl.appendChild(breadcrumb);

	// Each block builds its cards first so the section heading can report the
	// post-filter count, and an entire section (heading included) drops out when
	// a query filters it to nothing.
	let visibleCards = 0;

	// Ancestors block — collapsed by default per-card.
	if (ancestors.length > 0) {
		const cards = [];
		for (const a of ancestors) {
			const card = renderObsCard(a, {
				role: "ancestor",
				depth: 0,
				query: q,
			});
			if (card) cards.push(card);
		}
		if (cards.length > 0) {
			treeModalBodyEl.appendChild(
				renderModalSectionHeading(
					"ancestors",
					cards.length,
					q
						? "matching boxes only"
						: "every LLM call that shaped the chain from root → focus",
				),
			);
			for (const card of cards) treeModalBodyEl.appendChild(card);
			visibleCards += cards.length;
		}
	}

	// Focus block.
	const focusCard = renderObsCard(focusNode, {
		role: "focus",
		depth: 0,
		query: q,
	});
	if (focusCard) {
		treeModalBodyEl.appendChild(
			renderModalSectionHeading(
				"focused node",
				1,
				q
					? "matching boxes only"
					: "every LLM call captured for this node, expanded",
			),
		);
		treeModalBodyEl.appendChild(focusCard);
		visibleCards++;
	}

	// Descendants block.
	if (descendants.length > 0) {
		const cards = [];
		for (const [d, depth] of descendants) {
			const card = renderObsCard(d, {
				role: "descendant",
				depth,
				query: q,
			});
			if (card) cards.push(card);
		}
		if (cards.length > 0) {
			treeModalBodyEl.appendChild(
				renderModalSectionHeading(
					"descendants",
					cards.length,
					q ? "matching boxes only" : "indented by depth from focus",
				),
			);
			for (const card of cards) treeModalBodyEl.appendChild(card);
			visibleCards += cards.length;
		}
	}

	// A query that hit nothing anywhere — say so instead of a blank modal.
	if (q && visibleCards === 0) {
		const empty = document.createElement("div");
		empty.className = "tm-empty";
		empty.textContent = `no matches for “${treeModalQuery.trim()}”`;
		treeModalBodyEl.appendChild(empty);
	}

	if (anchorId) {
		const target = treeModalBodyEl.querySelector(
			`.tm-card[data-id="${CSS.escape(anchorId)}"]`,
		);
		if (target) {
			const newOffset =
				target.getBoundingClientRect().top -
				treeModalBodyEl.getBoundingClientRect().top;
			treeModalBodyEl.scrollTop += newOffset - anchorOffset;
		}
	}
}

treeExpandEl?.addEventListener("click", (ev) => {
	ev.stopPropagation(); // don't toggle the tree-header collapse
	openTreeModal();
});
treeModalCloseEl?.addEventListener("click", closeTreeModal);
treeModalEl?.addEventListener("click", (ev) => {
	if (ev.target === treeModalEl) closeTreeModal();
});
treeModalSearchEl?.addEventListener("input", () => {
	treeModalQuery = treeModalSearchEl.value;
	renderTreeModal();
});
window.addEventListener("keydown", (ev) => {
	if (ev.key === "Escape" && treeModalOpen) {
		if (
			document.activeElement === treeModalSearchEl &&
			treeModalSearchEl.value
		) {
			treeModalSearchEl.value = "";
			treeModalQuery = "";
			renderTreeModal();
			return;
		}
		closeTreeModal();
	}
});

// ===========================================================================
// Execution-flow graph — a pannable canvas of the ACTUAL run, not the semantic
// scene tree. Two interleaved node kinds:
//   * scene node `s:<id>`  — a zone/object/frame the run produced.
//   * step node  `t:<uid>` — one LLM call (zone_plan, zone_decompose,
//                            anchor_decompose, object_bbox_single, …).
// Parenting reflects ONLY which step generated what: a scene node's children
// are the steps issued from it (in execution order); a step node's children
// are the scene nodes its output emitted — so a scene node's parent is the
// step that created it. Clicking a scene node opens its observability; clicking
// a step node opens that exact call's system / input / output / reasoning. The
// graph reads the same `treeNodes`/`nodeLlmCalls` state the sidebar and the
// observability modal maintain, so it stays in sync with the live SSE stream
// and replay scrubbing for free (see the `animate()` refresh hook). State +
// layout constants are declared near the top of the file (before the render
// loop first runs); the functions and wiring live here.
// ===========================================================================

function flowSceneKindClass(kind) {
	return kind === "object" ? "object" : kind === "frame" ? "frame" : "zone";
}

// The scene ids a step's output brought into existence. We read the emitting
// fields only (children / objects / object) — NOT bbox `assignments` — so each
// scene node has exactly one generating step and the graph stays a tree.
function execEmittedIds(call) {
	const out = call?.output;
	if (!out || typeof out !== "object") return [];
	const ids = [];
	if (Array.isArray(out.children))
		for (const c of out.children) if (c?.id) ids.push(c.id);
	if (Array.isArray(out.objects))
		for (const o of out.objects) if (o?.id) ids.push(o.id);
	if (out.object && typeof out.object === "object" && out.object.id)
		ids.push(out.object.id);
	return ids;
}

// Assemble the exec graph from the recorded LLM calls + scene nodes.
function buildExecGraph() {
	const nodes = new Map();
	const childrenOf = new Map();
	const addChild = (parentKey, childKey) => {
		const arr = childrenOf.get(parentKey);
		if (arr) arr.push(childKey);
		else childrenOf.set(parentKey, [childKey]);
	};

	for (const [id, n] of treeNodes) {
		nodes.set(`s:${id}`, {
			key: `s:${id}`,
			type: "scene",
			id,
			sceneKind: n.kind ?? "zone",
			prompt: n.prompt ?? null,
			phase: n.phase ?? "pending",
			order: n.order ?? 0,
		});
	}

	const hasParent = new Set();
	// Each recorded call becomes a step node under the scene that issued it; its
	// emitted ids become that step's scene children (first emitter wins).
	for (const [ownerId, calls] of nodeLlmCalls) {
		const ownerKey = `s:${ownerId}`;
		if (!nodes.has(ownerKey)) {
			// Call issued from an id with no scene node (e.g. "_unattributed").
			nodes.set(ownerKey, {
				key: ownerKey,
				type: "scene",
				id: ownerId,
				sceneKind: "zone",
				prompt: null,
				phase: "pending",
				order: -1,
				synthetic: true,
			});
		}
		const ordered = calls
			.map((c, i) => ({ c, i }))
			.sort((a, b) => (a.c.eventIndex ?? a.i) - (b.c.eventIndex ?? b.i));
		for (const { c } of ordered) {
			const stepKey = `t:${c.uid}`;
			const emitted = [...new Set(execEmittedIds(c))];
			nodes.set(stepKey, {
				key: stepKey,
				type: "step",
				step: c.step || "(step)",
				ownerId,
				call: c,
				eventIndex: c.eventIndex ?? null,
				cached: !!c.cached,
				model: c.model || "",
				generated: emitted.length,
			});
			addChild(ownerKey, stepKey);
			hasParent.add(stepKey);
			for (const gid of emitted) {
				const childKey = `s:${gid}`;
				if (!nodes.has(childKey) || hasParent.has(childKey)) continue;
				addChild(stepKey, childKey);
				hasParent.add(childKey);
			}
		}
	}

	// Roots: the real scene root first, then anything still unparented (orphan
	// scenes whose generating call we never saw, synthetic owners, etc.).
	const roots = [];
	if (treeRootId && nodes.has(`s:${treeRootId}`))
		roots.push(`s:${treeRootId}`);
	for (const [key] of nodes) {
		if (key === `s:${treeRootId}`) continue;
		if (!hasParent.has(key)) roots.push(key);
	}
	return { nodes, childrenOf, roots };
}

// Indented-outline layout: every node takes its own row (top-to-bottom in
// execution / emission order) and is indented by depth. Width is bounded by the
// tree's depth instead of its leaf count, so deep/bushy graphs grow downward
// rather than sprawling sideways. Same-depth nodes do NOT share a row — that
// rigid banding is what made the old layout spread too thin horizontally.
function flowLayout(graph) {
	const positions = new Map();
	const visited = new Set();
	let row = 0;
	function place(key, depth) {
		visited.add(key);
		positions.set(key, {
			x: depth * FLOW.INDENT,
			y: row * FLOW.ROW_H,
			depth,
		});
		row += 1;
		for (const k of graph.childrenOf.get(key) ?? []) {
			if (graph.nodes.has(k) && !visited.has(k)) place(k, depth + 1);
		}
	}
	for (const r of graph.roots) {
		if (visited.has(r)) continue;
		place(r, 0);
		row += 1; // blank spacer row between separate root trees
	}
	for (const [key] of graph.nodes) if (!visited.has(key)) place(key, 0);
	let maxX = 0;
	let maxY = 0;
	for (const p of positions.values()) {
		if (p.x > maxX) maxX = p.x;
		if (p.y > maxY) maxY = p.y;
	}
	return {
		positions,
		width: maxX + FLOW.NODE_W + FLOW.PAD * 2,
		height: maxY + FLOW.NODE_H + FLOW.PAD * 2,
	};
}

function flowSearchHit(text) {
	const q = flowSearchQuery.trim().toLowerCase();
	if (!q) return null;
	return (text ?? "").toLowerCase().includes(q);
}

function buildSceneFlowNode(node, pos) {
	const el = document.createElement("div");
	el.className = `flow-node flow-scene ${flowSceneKindClass(node.sceneKind)}`;
	el.dataset.key = node.key;
	el.dataset.id = node.id;
	el.style.left = `${pos.x + FLOW.PAD}px`;
	el.style.top = `${pos.y + FLOW.PAD}px`;
	el.style.width = `${FLOW.NODE_W}px`;
	if (node.id === treeActiveId) el.classList.add("active");
	if (node.id === selectedBboxId) el.classList.add("selected");
	const hit = flowSearchHit(`${node.id} ${node.prompt ?? ""}`);
	if (hit === true) el.classList.add("hit");
	else if (hit === false) el.classList.add("dimmed");

	const head = document.createElement("div");
	head.className = "flow-node-head";
	const dot = document.createElement("span");
	dot.className = `flow-dot phase-${node.phase ?? "pending"}`;
	head.appendChild(dot);
	const idEl = document.createElement("span");
	idEl.className = "flow-node-id";
	idEl.textContent = node.id;
	head.appendChild(idEl);
	el.appendChild(head);

	if (node.prompt) {
		const p = document.createElement("div");
		p.className = "flow-node-prompt";
		p.textContent = truncate(node.prompt, 52);
		el.appendChild(p);
	}
	el.title = `${node.sceneKind} ${node.id} — click for observability`;
	el.addEventListener("click", (ev) => {
		ev.stopPropagation();
		openObsFor(node.id);
	});
	return el;
}

function buildStepFlowNode(node, pos) {
	const el = document.createElement("div");
	el.className = `flow-node flow-step s-${node.step}`;
	el.dataset.key = node.key;
	el.style.left = `${pos.x + FLOW.PAD}px`;
	el.style.top = `${pos.y + FLOW.PAD}px`;
	el.style.width = `${FLOW.NODE_W}px`;
	if (node.cached) el.classList.add("cached");
	const hit = flowSearchHit(`${node.step} ${node.ownerId}`);
	if (hit === true) el.classList.add("hit");
	else if (hit === false) el.classList.add("dimmed");

	const head = document.createElement("div");
	head.className = "flow-step-head";
	const nameEl = document.createElement("span");
	nameEl.className = "flow-step-name";
	nameEl.textContent = node.step;
	head.appendChild(nameEl);
	if (node.generated > 0) {
		const gen = document.createElement("span");
		gen.className = "flow-step-gen";
		gen.textContent = `→${node.generated}`;
		gen.title = `emitted ${node.generated} node${node.generated === 1 ? "" : "s"}`;
		head.appendChild(gen);
	}
	if (node.cached) {
		const c = document.createElement("span");
		c.className = "flow-step-cached";
		c.textContent = "cached";
		head.appendChild(c);
	}
	// "Rewind & tune here": exits the graph, rewinds the canvas to this step,
	// and opens the prompt-tuning sandbox on it. Only meaningful for steps with
	// a recorded event index (the position the rewind cuts at).
	if (node.eventIndex != null) {
		const tune = document.createElement("button");
		tune.type = "button";
		tune.className = "flow-step-tune";
		tune.textContent = "⤺ tune";
		tune.title =
			"Rewind the canvas to this step and edit its prompt — re-runs only this step, non-destructively";
		tune.addEventListener("click", (ev) => {
			ev.stopPropagation();
			enterSandboxAtStep(node.call);
		});
		head.appendChild(tune);
	}
	el.appendChild(head);
	const sub = document.createElement("div");
	sub.className = "flow-step-sub";
	sub.textContent = node.model || `on ${node.ownerId}`;
	el.appendChild(sub);

	el.title = `${node.step} on ${node.ownerId} — click for input / output / reasoning · ⤺ tune to rewind & edit its prompt`;
	el.addEventListener("click", (ev) => {
		ev.stopPropagation();
		openObsForStep(node.ownerId, node.call.uid);
	});
	return el;
}

function renderFlow() {
	if (!flowModalOpen) return;
	const graph = buildExecGraph();
	const { positions, width, height } = flowLayout(graph);
	_flowGraph = graph;
	_flowPositions = positions;
	_flowLastWidth = width;
	_flowLastHeight = height;

	flowEmptyEl.classList.toggle("show", positions.size === 0);
	flowStageEl.style.width = `${width}px`;
	flowStageEl.style.height = `${height}px`;

	// Edges: an outline connector per parent — a spine dropping from the parent's
	// gutter with a rounded tick into each child's left edge. Children of a given
	// node share a type, so the overlapping spine stays one color. Tinted by child.
	flowEdgesEl.setAttribute("width", String(width));
	flowEdgesEl.setAttribute("height", String(height));
	let edges = "";
	const R = 6; // connector corner radius
	for (const [key, p] of positions) {
		const sx = p.x + FLOW.PAD + FLOW.GUTTER; // spine x, just inside the parent's left edge
		const sy = p.y + FLOW.PAD + FLOW.NODE_H; // parent bottom
		for (const ck of graph.childrenOf.get(key) ?? []) {
			const cp = positions.get(ck);
			if (!cp) continue;
			const child = graph.nodes.get(ck);
			const cy = cp.y + FLOW.PAD + FLOW.NODE_H / 2; // child vertical center
			const cx = cp.x + FLOW.PAD; // child left edge
			const cls =
				child.type === "step" ? "to-step" : `to-${child.sceneKind}`;
			edges += `<path class="flow-edge ${cls}" d="M ${sx.toFixed(1)} ${sy.toFixed(1)} L ${sx.toFixed(1)} ${(cy - R).toFixed(1)} Q ${sx.toFixed(1)} ${cy.toFixed(1)} ${(sx + R).toFixed(1)} ${cy.toFixed(1)} L ${cx.toFixed(1)} ${cy.toFixed(1)}" />`;
		}
	}
	flowEdgesEl.innerHTML = edges;

	const frag = document.createDocumentFragment();
	for (const [key, p] of positions) {
		const node = graph.nodes.get(key);
		frag.appendChild(
			node.type === "step"
				? buildStepFlowNode(node, p)
				: buildSceneFlowNode(node, p),
		);
	}
	flowNodesEl.replaceChildren(frag);
	applyFlowTransform();
}

function applyFlowTransform() {
	flowStageEl.style.transform = `translate(${flowPanX}px, ${flowPanY}px) scale(${flowZoom})`;
	// Pan/zoom the background dot-grid with the content for an infinite-canvas feel.
	flowViewportEl.style.backgroundPosition = `${flowPanX}px ${flowPanY}px`;
	flowViewportEl.style.backgroundSize = `${FLOW.GRID * flowZoom}px ${FLOW.GRID * flowZoom}px`;
}

function flowClampZoom(z) {
	return Math.max(0.15, Math.min(2.5, z));
}

function flowZoomAt(cx, cy, factor) {
	const next = flowClampZoom(flowZoom * factor);
	const k = next / flowZoom;
	flowPanX = cx - (cx - flowPanX) * k;
	flowPanY = cy - (cy - flowPanY) * k;
	flowZoom = next;
	applyFlowTransform();
}

function fitFlow() {
	const w = _flowLastWidth || 1;
	const h = _flowLastHeight || 1;
	const vw = flowViewportEl.clientWidth || 1;
	const vh = flowViewportEl.clientHeight || 1;
	// The outline is narrow and grows downward: fit its width at up to 1:1 so
	// nodes stay readable, then anchor to the top and let the user scroll down —
	// shrinking a tall tree to fit vertically would make everything tiny.
	flowZoom = flowClampZoom(Math.min(vw / w, 1) * 0.94);
	flowPanX = (vw - w * flowZoom) / 2;
	flowPanY = h * flowZoom <= vh ? (vh - h * flowZoom) / 2 : 0;
	applyFlowTransform();
}

function openObsFor(id) {
	if (!treeNodes.has(id)) return;
	if (!treeModalOpen) openTreeModal();
	focusModalOn(id);
}

// Step-node click: open the observability modal on the step's owner node and
// scroll its specific call block (system / input / output / reasoning) into
// view — the exact panes the observability pipeline shows.
function openObsForStep(ownerId, callUid) {
	if (!treeModalOpen) openTreeModal();
	if (treeNodes.has(ownerId)) focusModalOn(ownerId);
	requestAnimationFrame(() => {
		const focusCard =
			treeModalBodyEl.querySelector(".tm-obs-card.role-focus") ||
			treeModalBodyEl;
		const block = focusCard.querySelector(
			`.tm-llm-call[data-call-key="${CSS.escape(String(callUid))}"]`,
		);
		if (!block) return;
		block.open = true;
		block.scrollIntoView({ block: "center" });
		block.classList.add("tm-llm-flash");
		setTimeout(() => block.classList.remove("tm-llm-flash"), 1400);
	});
}

// Center the canvas on a scene node — used when a sidebar-tree row is clicked
// while the flow canvas is open. Bumps zoom in if the user was zoomed far out.
function flowCenterOnScene(id) {
	const pos = _flowPositions.get(`s:${id}`);
	if (!pos) return;
	if (flowZoom < 0.75) flowZoom = 0.9;
	const vw = flowViewportEl.clientWidth || 1;
	const vh = flowViewportEl.clientHeight || 1;
	const nx = pos.x + FLOW.PAD + FLOW.NODE_W / 2;
	const ny = pos.y + FLOW.PAD + FLOW.NODE_H / 2;
	flowPanX = vw / 2 - nx * flowZoom;
	flowPanY = vh / 2 - ny * flowZoom;
	applyFlowTransform();
}

function openFlowModal() {
	flowModalOpen = true;
	flowModalEl.classList.add("open");
	// Float the sidebar tree above the canvas so it stays usable while the flow
	// graph is open (CSS bumps #tree's z-index under body.flow-open).
	document.body.classList.add("flow-open");
	flowSearchEl.value = flowSearchQuery;
	updateFlowPauseButton();
	renderFlow();
	// Defer fit until the viewport has its open dimensions.
	requestAnimationFrame(() => {
		renderFlow();
		fitFlow();
	});
}

function closeFlowModal() {
	flowModalOpen = false;
	flowModalEl.classList.remove("open");
	document.body.classList.remove("flow-open");
}

function toggleFlowModal() {
	if (flowModalOpen) closeFlowModal();
	else openFlowModal();
}

// Center the view on the first node (scene or step) matching the query.
function flowLocate(query) {
	const q = query.trim().toLowerCase();
	if (!q || !_flowGraph) return;
	for (const [key, p] of _flowPositions) {
		const node = _flowGraph.nodes.get(key);
		if (!node) continue;
		const hay =
			node.type === "step"
				? `${node.step} ${node.ownerId}`
				: `${node.id} ${node.prompt ?? ""}`;
		if (hay.toLowerCase().includes(q)) {
			const vw = flowViewportEl.clientWidth || 1;
			const vh = flowViewportEl.clientHeight || 1;
			const nx = p.x + FLOW.PAD + FLOW.NODE_W / 2;
			const ny = p.y + FLOW.PAD + FLOW.NODE_H / 2;
			flowPanX = vw / 2 - nx * flowZoom;
			flowPanY = vh / 2 - ny * flowZoom;
			applyFlowTransform();
			return;
		}
	}
}

treeFlowEl?.addEventListener("click", (ev) => {
	ev.stopPropagation(); // don't toggle the tree-header collapse
	toggleFlowModal();
});
flowCloseEl?.addEventListener("click", closeFlowModal);
flowFitEl?.addEventListener("click", fitFlow);
flowZoomInEl?.addEventListener("click", () =>
	flowZoomAt(
		flowViewportEl.clientWidth / 2,
		flowViewportEl.clientHeight / 2,
		1.2,
	),
);
flowZoomOutEl?.addEventListener("click", () =>
	flowZoomAt(
		flowViewportEl.clientWidth / 2,
		flowViewportEl.clientHeight / 2,
		1 / 1.2,
	),
);
flowSearchEl?.addEventListener("input", () => {
	flowSearchQuery = flowSearchEl.value;
	renderFlow();
	flowLocate(flowSearchQuery);
});
flowModalEl?.addEventListener("pointerdown", (ev) => {
	// Backdrop (area outside the panel) closes the modal.
	if (ev.target === flowModalEl) closeFlowModal();
});
flowViewportEl?.addEventListener("pointerdown", (ev) => {
	if (ev.button !== 0) return;
	if (ev.target.closest(".flow-node")) return; // let the node handle its click
	_flowPanning = true;
	_flowStartX = ev.clientX;
	_flowStartY = ev.clientY;
	_flowStartPanX = flowPanX;
	_flowStartPanY = flowPanY;
	flowViewportEl.classList.add("grabbing");
	flowViewportEl.setPointerCapture?.(ev.pointerId);
});
flowViewportEl?.addEventListener("pointermove", (ev) => {
	if (!_flowPanning) return;
	const dx = ev.clientX - _flowStartX;
	const dy = ev.clientY - _flowStartY;
	flowPanX = _flowStartPanX + dx;
	flowPanY = _flowStartPanY + dy;
	applyFlowTransform();
});
function flowEndPan(ev) {
	if (!_flowPanning) return;
	_flowPanning = false;
	flowViewportEl.classList.remove("grabbing");
	flowViewportEl.releasePointerCapture?.(ev.pointerId);
}
flowViewportEl?.addEventListener("pointerup", flowEndPan);
flowViewportEl?.addEventListener("pointercancel", flowEndPan);
flowViewportEl?.addEventListener(
	"wheel",
	(ev) => {
		ev.preventDefault();
		const rect = flowViewportEl.getBoundingClientRect();
		const factor = ev.deltaY < 0 ? 1.05 : 1 / 1.05;
		flowZoomAt(ev.clientX - rect.left, ev.clientY - rect.top, factor);
	},
	{ passive: false },
);
// Capture-phase Escape so the observability modal (a bubble-phase handler that
// sits on top, z-index 1100) closes first; only once it is gone does Escape
// close the flow graph beneath it.
window.addEventListener(
	"keydown",
	(ev) => {
		if (ev.key !== "Escape") return;
		if (!flowModalOpen || treeModalOpen) return;
		if (document.activeElement === flowSearchEl && flowSearchEl.value) {
			flowSearchEl.value = "";
			flowSearchQuery = "";
			renderFlow();
			return;
		}
		ev.stopImmediatePropagation();
		closeFlowModal();
	},
	true,
);

// ============================================================================
// Prompt-tuning sandbox
//
// Rewind the canvas to a single pipeline step and re-run that step's LLM call
// under an edited prompt, non-destructively. We never touch the recorded log:
// the run is PAUSED (server) and the SSE detached so client state freezes, the
// rewind is faked by hiding scene objects created at/after the step (see
// `withinRewind`), and a tested step's output is drawn into
// `sandboxOverlayRoot`. Break-out clears the cutoff + overlay (instant restore)
// and resumes the run if we paused it.
// ============================================================================

const SANDBOX_OVERLAY_COLOR = 0xff4fdd; // vivid magenta — distinct from every bbox color
// Every magenta box drawn into sandboxOverlayRoot is registered here with the
// metadata needed to identify it (id, kind, prompt), so it hovers (tooltip +
// highlight) and selects exactly like a normal-canvas bbox. Cleared with the
// overlay. Hover/select reuse the canvas hover/selected colors for consistency.
const sandboxOverlayBoxes = new Map(); // overlayId -> { helper, box, id, kind, prompt }
let _overlayBoxSeq = 0;
let hoveredOverlayId = null;
let selectedOverlayId = null;
// True when the viewed run was live (running) at sandbox entry, so break-out
// reconnects the SSE to catch up on what the resumed pipeline emits.
let sandboxWasLive = false;
// Synchronous re-entrancy latch: enterSandboxAtStep awaits the pause POST, so a
// fast second click would otherwise start a parallel entry and clobber
// sandboxPausedByUs. Held from the first click until setup completes.
let sandboxEntering = false;
// Expanded (near-fullscreen) panel preference — persisted so it sticks across
// sessions, like the other view toggles.
const SANDBOX_EXPANDED_KEY = "starshot:sandbox-expanded";
let sandboxExpanded = (() => {
	try {
		return localStorage.getItem(SANDBOX_EXPANDED_KEY) === "1";
	} catch {
		return false;
	}
})();

function applySandboxExpanded() {
	sandboxPanelEl.classList.toggle("expanded", sandboxExpanded);
	if (sandboxExpandEl) {
		sandboxExpandEl.textContent = sandboxExpanded ? "⤡" : "⤢";
		sandboxExpandEl.title = sandboxExpanded
			? "Collapse the panel"
			: "Expand the panel for easier reading of the system / input / output";
	}
}

function clearSandboxOverlay() {
	while (sandboxOverlayRoot.children.length > 0) {
		const child = sandboxOverlayRoot.children[0];
		sandboxOverlayRoot.remove(child);
		// disposeObject3D walks meshes (branch GLB groups); the direct dispose
		// covers Box3Helpers (LineSegments — not meshes, so the walk skips them).
		disposeObject3D(child);
		child.geometry?.dispose?.();
		child.material?.dispose?.();
	}
	sandboxOverlayBoxes.clear();
	hoveredOverlayId = null;
	selectedOverlayId = null;
	if (sandboxActive) tooltip.style.display = "none";
}

// Record the event index each node's bbox / mesh first appeared at, so the
// rewind cutoff can hide everything created from a given step onward.
function buildCreationIndexMaps() {
	bboxCreatedIndex.clear();
	modelCreatedIndex.clear();
	for (const e of recordedEvents) {
		if (typeof e.index !== "number" || typeof e.id !== "string") continue;
		if (e.kind === "bbox") {
			if (!bboxCreatedIndex.has(e.id))
				bboxCreatedIndex.set(e.id, e.index);
		} else if (e.kind === "model") {
			if (!modelCreatedIndex.has(e.id))
				modelCreatedIndex.set(e.id, e.index);
		}
	}
}

// Every recorded LLM-call step across all nodes, ordered by the event index it
// was logged at — the execution-order spine prev/next walks. Steps without an
// event index (legacy logs) can't be positioned, so they're skipped, and
// duplicates are collapsed by event index: the live tail and the background
// history backfill both feed recordLlmCall with no per-index dedup, so a cell
// opened mid-run can hold two entries for one call. They carry identical
// content, so keeping the first is exact.
function collectSandboxSteps() {
	const byIndex = new Map();
	for (const [, calls] of nodeLlmCalls) {
		for (const c of calls) {
			if (
				typeof c.eventIndex === "number" &&
				!byIndex.has(c.eventIndex)
			) {
				byIndex.set(c.eventIndex, c);
			}
		}
	}
	return [...byIndex.values()].sort((a, b) => a.eventIndex - b.eventIndex);
}

function setSandboxStatus(msg, cls = "") {
	if (!sandboxStatusEl) return;
	sandboxStatusEl.textContent = msg || "";
	sandboxStatusEl.className = cls; // "", "err", or "ok"
}

// Flag the system/user fields whose text diverges from the recorded prompt.
function markSandboxEdited() {
	// Baseline is the viewed branch step when stepping through a branch,
	// otherwise the original recorded step being tuned.
	const base = branchActive
		? branchSteps[branchCursor]
		: sandboxSteps[sandboxCursor];
	if (!base) return;
	sandboxSystemFieldEl.classList.toggle(
		"edited",
		sandboxSystemEl.value !== (base.system ?? ""),
	);
	sandboxUserFieldEl.classList.toggle(
		"edited",
		sandboxUserEl.value !== (base.user ?? ""),
	);
}

async function enterSandboxAtStep(call) {
	if (call == null || typeof call.eventIndex !== "number") return;
	if (sandboxActive) {
		// Already tuning — just jump the cursor to the clicked step. Match by
		// event index (stable) rather than uid, which differs across the
		// dedup-collapsed duplicates.
		const idx = sandboxSteps.findIndex(
			(c) => c.eventIndex === call.eventIndex,
		);
		if (idx >= 0) gotoSandboxStep(idx);
		return;
	}
	if (sandboxEntering) return; // a parallel entry is mid-flight (async pause)
	sandboxEntering = true;
	try {
		// Freeze the run for the session: detach the SSE FIRST (so the run.paused
		// sentinel never lands in our recorded log / bumps highestEventIndex), then
		// pause it server-side if it was live.
		const wasRunning = currentRunInfo()?.status === "running";
		sandboxWasLive = wasRunning;
		if (currentSource) {
			currentSource.close();
			currentSource = null;
		}
		sandboxPausedByUs = false;
		if (wasRunning && currentSlotId && currentModel) {
			try {
				const res = await fetch(
					new URL(
						`/slots/${encodeURIComponent(currentSlotId)}/${encodeURIComponent(currentModel)}/pause?run=${encodeURIComponent(currentRun)}`,
						SERVER_URL,
					),
					{ method: "POST" },
				);
				if (res.ok) sandboxPausedByUs = true;
			} catch {}
		}

		sandboxActive = true;
		buildCreationIndexMaps();
		sandboxSteps = collectSandboxSteps();
		const cursor = sandboxSteps.findIndex(
			(c) => c.eventIndex === call.eventIndex,
		);
		// Clear any selection so the rewound scene isn't dimmed by it.
		if (selectedBboxId !== null) {
			const prev = selectedBboxId;
			selectedBboxId = null;
			applyBboxColor(prev);
		}
		closeFlowModal();
		sandboxPanelEl.classList.add("open");
		document.body.classList.add("sandbox-open");
		applySandboxExpanded();
		applySandboxMode(); // fresh session is always edit-mode (Phase A)
		gotoSandboxStep(cursor >= 0 ? cursor : 0);
		refreshSlots();
	} finally {
		sandboxEntering = false;
	}
}

// Rewind to step `i`: hide everything from its event index onward, prefill the
// editors with its recorded prompts, and reset the output area.
function gotoSandboxStep(i) {
	if (!sandboxActive || i < 0 || i >= sandboxSteps.length) return;
	sandboxCursor = i;
	const call = sandboxSteps[i];
	clearSandboxOverlay();
	rewindCutoffIndex = call.eventIndex;
	refreshAllVisibility();

	sandboxSystemEl.value = call.system ?? "";
	sandboxUserEl.value = call.user ?? "";
	markSandboxEdited();

	sandboxStepPillEl.textContent = call.step ?? "(step)";
	sandboxStepMetaEl.textContent = "";
	const onEl = document.createElement("span");
	onEl.textContent = "on ";
	const ownerEl = document.createElement("b");
	ownerEl.textContent = call.parentNode ?? "—";
	sandboxStepMetaEl.append(onEl, ownerEl);
	if (call.model) {
		const modEl = document.createElement("span");
		modEl.textContent = `   ·   ${call.model}`;
		sandboxStepMetaEl.append(modEl);
	}
	sandboxPosEl.textContent = `step ${i + 1} / ${sandboxSteps.length}`;

	sandboxOutputWrapEl.classList.remove("show");
	sandboxOutputEl.textContent = "";
	sandboxReasoningBodyEl.textContent = "";
	updateSimulateButton();
	sandboxPrevEl.disabled = i <= 0;
	sandboxNextEl.disabled = i >= sandboxSteps.length - 1;
	sandboxTestEl.disabled = !call.schema;
	setSandboxStatus(
		"rewound — optionally test, or 'simulate downstream' to step through from here",
	);
}

async function testSandboxStep() {
	if (!sandboxActive || branchActive || sandboxTesting) return;
	const call = sandboxSteps[sandboxCursor];
	if (!call || !call.schema) return;
	sandboxTesting = true;
	sandboxTestEl.classList.add("busy");
	sandboxTestEl.disabled = true;
	setSandboxStatus("re-running this step…");
	try {
		const res = await fetch(
			new URL(
				`/llm/test?run=${encodeURIComponent(currentRun)}`,
				SERVER_URL,
			),
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					system: sandboxSystemEl.value,
					user: sandboxUserEl.value,
					schema_name: call.schema,
					model: call.model || "",
				}),
			},
		);
		if (!res.ok) {
			setSandboxStatus(
				`test failed: HTTP ${res.status} — ${await res.text()}`,
				"err",
			);
			return;
		}
		showSandboxResult(call, await res.json());
	} catch (e) {
		setSandboxStatus(`test failed: ${e.message}`, "err");
	} finally {
		sandboxTesting = false;
		sandboxTestEl.classList.remove("busy");
		// Only re-own the button state if we're still on the step we tested —
		// otherwise gotoSandboxStep already set it for the step the user moved to.
		if (sandboxActive && sandboxSteps[sandboxCursor]?.uid === call.uid) {
			sandboxTestEl.disabled = !call.schema;
		}
	}
}

function showSandboxResult(call, data) {
	// Guard against a result landing after the user already broke out or stepped.
	if (!sandboxActive || sandboxSteps[sandboxCursor]?.uid !== call.uid) return;
	const output = data.output ?? null;
	sandboxOutputWrapEl.classList.add("show");
	sandboxOutputEl.textContent = JSON.stringify(output, null, 2);
	const reasoning = (data.reasoning ?? "").trim();
	sandboxReasoningEl.style.display = reasoning ? "" : "none";
	sandboxReasoningBodyEl.textContent = reasoning;

	const rendered = renderSandboxOverlay(call, output);
	if (rendered > 0) {
		sandboxRenderNoteEl.textContent = `▲ ${rendered} box${rendered === 1 ? "" : "es"} rendered on the canvas (magenta)`;
		sandboxRenderNoteEl.classList.remove("nothing");
	} else {
		sandboxRenderNoteEl.textContent =
			"no spatial geometry in this step's output — shown as JSON below";
		sandboxRenderNoteEl.classList.add("nothing");
	}
	const tok =
		data.tokens_in != null
			? ` · ${data.tokens_in}+${data.tokens_out ?? 0} tok`
			: "";
	setSandboxStatus(`done${tok}`, "ok");
}

// Min corner of an axis-aligned box given a (possibly signed-dimension) origin.
function bboxMinCorner(origin, dims) {
	return [
		Math.min(origin[0], origin[0] + dims[0]),
		Math.min(origin[1], origin[1] + dims[1]),
		Math.min(origin[2], origin[2] + dims[2]),
	];
}

// Draw the renderable part of a tested step's output as magenta overlay boxes
// on top of the rewound scene. Returns how many boxes were drawn.
//   * overall_bbox  → one world-frame box (the root canvas)
//   * *_bbox_batch  → one box per assignment, converted from the owner
//                     region's local frame back to world coordinates
// Decompose / plan / next_object / image_prompt outputs carry no geometry → 0.
function renderSandboxOverlay(call, output) {
	clearSandboxOverlay();
	if (!output || typeof output !== "object") return 0;
	const boxes = [];
	if (
		output.bbox &&
		Array.isArray(output.bbox.origin) &&
		Array.isArray(output.bbox.dimensions)
	) {
		boxes.push({
			origin: output.bbox.origin,
			dimensions: output.bbox.dimensions,
			meta: {
				id: call.parentNode ?? null,
				kind: treeNodes.get(call.parentNode)?.kind ?? null,
			},
		});
	}
	if (Array.isArray(output.assignments)) {
		// Batch assignments are authored in the parent region's local frame; the
		// owner (call.parentNode) was placed before this step so its world bbox is
		// known. Sibling-anchored children (rare) are approximated against the
		// region — fine for a preview.
		const owner = treeNodes.get(call.parentNode);
		const pmin =
			owner &&
			Array.isArray(owner.origin) &&
			Array.isArray(owner.dimensions)
				? bboxMinCorner(owner.origin, owner.dimensions)
				: [0, 0, 0];
		for (const a of output.assignments) {
			const bb = a && a.bbox;
			if (
				!bb ||
				!Array.isArray(bb.origin) ||
				!Array.isArray(bb.dimensions)
			)
				continue;
			boxes.push({
				origin: [
					bb.origin[0] + pmin[0],
					bb.origin[1] + pmin[1],
					bb.origin[2] + pmin[2],
				],
				dimensions: bb.dimensions,
				meta: {
					id: a.id ?? null,
					prompt: typeof a.prompt === "string" ? a.prompt : null,
				},
			});
		}
	}
	for (const b of boxes) addSandboxOverlayBox(b.origin, b.dimensions, b.meta);
	return boxes.length;
}

// Draw one magenta overlay box. `meta` ({id, kind, prompt}) is what the
// hover tooltip / selection surface to say "what this box is"; anonymous boxes
// (no id) still register so they can be hovered, just with a generic label.
function addSandboxOverlayBox(origin, dimensions, meta = null) {
	const ox = origin[0],
		oy = origin[1],
		oz = origin[2];
	const fx = ox + dimensions[0],
		fy = oy + dimensions[1],
		fz = oz + dimensions[2];
	const box3 = new THREE.Box3(
		new THREE.Vector3(Math.min(ox, fx), Math.min(oy, fy), Math.min(oz, fz)),
		new THREE.Vector3(Math.max(ox, fx), Math.max(oy, fy), Math.max(oz, fz)),
	);
	const helper = new THREE.Box3Helper(box3, SANDBOX_OVERLAY_COLOR);
	// Draw on top of everything so the proposed box reads clearly against the
	// rewound scene regardless of depth.
	helper.material.depthTest = false;
	helper.material.transparent = true;
	helper.renderOrder = 999;
	const overlayId = meta?.id ?? `__overlay_${_overlayBoxSeq++}`;
	helper.userData.overlayId = overlayId;
	sandboxOverlayRoot.add(helper);
	sandboxOverlayBoxes.set(overlayId, {
		helper,
		box: box3,
		id: meta?.id ?? null,
		kind: meta?.kind ?? null,
		prompt: meta?.prompt ?? null,
	});
	applyOverlayColor(overlayId);
	return helper;
}

// Paint a single overlay box's color from its hover/selected state — selected
// (cyan) beats hovered (yellow) beats the default magenta.
function applyOverlayColor(oid) {
	if (oid == null) return;
	const entry = sandboxOverlayBoxes.get(oid);
	if (!entry) return;
	const color =
		oid === selectedOverlayId
			? BBOX_COLOR_SELECTED
			: oid === hoveredOverlayId
				? BBOX_COLOR_HOVER
				: SANDBOX_OVERLAY_COLOR;
	entry.helper.material.color.setHex(color);
}

function setHoveredOverlay(oid) {
	if (oid === hoveredOverlayId) return;
	const prev = hoveredOverlayId;
	hoveredOverlayId = oid;
	applyOverlayColor(prev);
	applyOverlayColor(oid);
}

function selectOverlay(oid) {
	const prev = selectedOverlayId;
	selectedOverlayId = prev === oid ? null : oid; // re-click clears
	applyOverlayColor(prev);
	applyOverlayColor(selectedOverlayId);
}

// Smallest-volume overlay box the ray crosses — the deepest/most-specific box
// under the cursor, matching the zone picker (boxes nest + overlap).
const _overlayHit = new THREE.Vector3();
const _overlaySize = new THREE.Vector3();
function pickHoveredOverlayId() {
	let bestId = null;
	let bestVol = Infinity;
	for (const [oid, entry] of sandboxOverlayBoxes) {
		if (!entry.helper.visible) continue;
		if (!raycaster.ray.intersectBox(entry.box, _overlayHit)) continue;
		entry.box.getSize(_overlaySize);
		const vol = _overlaySize.x * _overlaySize.y * _overlaySize.z;
		if (vol < bestVol) {
			bestVol = vol;
			bestId = oid;
		}
	}
	return bestId;
}

// ===========================================================================
// Downstream simulation (branch)
//
// "Simulate downstream" commits the current step's tested output as a deviation
// (POST /branch) and the server re-runs the whole pipeline from there in an
// isolated `<cell>/_branch`. We keep the original scene rewound to the
// deviation point and stream the branch's DOWNSTREAM nodes — bboxes + real
// meshes — into the overlay as they're produced, so the change visibly
// cascades. The panel flips to read-only and shows the branch's re-rendered
// prompts. "Back to edit" drops the branch and returns to tuning; break-out
// deletes it and restores the original run.
// ===========================================================================

function updateSimulateButton() {
	if (!sandboxSimulateEl) return;
	// Downstream simulation is just a mode — no test required. Enabled on any
	// step (it forks before that step and steps through from there).
	sandboxSimulateEl.disabled = !(
		sandboxActive &&
		!branchActive &&
		!!sandboxSteps[sandboxCursor]
	);
}

// Toggle the panel between EDIT (Phase A — tune the original) and BRANCH
// (Phase B — step through the simulated downstream, editing each step's
// re-rendered prompt before it runs).
function applySandboxMode() {
	const branch = branchActive;
	sandboxPanelEl.classList.toggle("branch-mode", branch);
	// Prompts stay editable in both modes (tune the original in EDIT; edit each
	// step's prompt in BRANCH to run / re-run it).
	sandboxSystemEl.readOnly = false;
	sandboxUserEl.readOnly = false;
	// EDIT-only buttons.
	for (const el of [sandboxTestEl, sandboxResetEl, sandboxSimulateEl]) {
		if (el) el.style.display = branch ? "none" : "";
	}
	// BRANCH-only buttons (runstep/rerun/runrest visibility within branch is
	// refined per viewed step by updateBranchControls).
	for (const el of [
		sandboxRunStepEl,
		sandboxRerunEl,
		sandboxRunRestEl,
		sandboxBackEl,
	]) {
		if (el) el.style.display = branch ? "" : "none";
	}
	// prev / next exist in both modes (navigate original steps in EDIT, branch
	// steps in BRANCH).
	for (const el of [sandboxPrevEl, sandboxNextEl]) {
		if (el) el.style.display = "";
	}
	if (branch) updateBranchControls();
	else updateSimulateButton();
}

async function simulateDownstream() {
	if (!sandboxActive || branchActive) return;
	const call = sandboxSteps[sandboxCursor];
	if (!call) return;
	sandboxSimulateEl.disabled = true;
	setSandboxStatus("forking a branch & stepping through downstream…");
	// Carry the in-progress edit into the first pause (this very step) so you
	// don't retype it; consumed on the first branch.step.pending.
	branchFirstPrompt = {
		system: sandboxSystemEl.value,
		user: sandboxUserEl.value,
	};
	let res;
	try {
		res = await fetch(
			new URL(
				`/slots/${encodeURIComponent(currentSlotId)}/${encodeURIComponent(currentModel)}/branch?run=${encodeURIComponent(currentRun)}`,
				SERVER_URL,
			),
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ deviation_index: call.eventIndex }),
			},
		);
	} catch (e) {
		setSandboxStatus(`branch failed: ${e.message}`, "err");
		branchFirstPrompt = null;
		updateSimulateButton();
		return;
	}
	if (!res.ok) {
		setSandboxStatus(
			`branch failed: HTTP ${res.status} — ${await res.text()}`,
			"err",
		);
		branchFirstPrompt = null;
		updateSimulateButton();
		return;
	}
	enterBranchView(call.eventIndex);
}

function enterBranchView(deviationIndex) {
	branchActive = true;
	branchDone = false;
	branchSteps = [];
	branchCursor = -1;
	branchRebuilding = false;
	branchStepBusy = false;
	branchAuto = false;
	branchDeviationIndex = deviationIndex;
	branchGen += 1;
	branchOverlayBboxIds.clear();
	branchOverlayMeshes.clear();
	// Keep the original scene rewound to the fork point; the branch's downstream
	// renders on top (bboxes + meshes) as it streams.
	clearSandboxOverlay();
	rewindCutoffIndex = deviationIndex;
	refreshAllVisibility();
	applySandboxMode();
	sandboxStepPillEl.textContent = "branch";
	sandboxStepMetaEl.textContent = "simulating downstream…";
	sandboxPosEl.textContent = "";
	sandboxOutputWrapEl.classList.remove("show");
	setSandboxStatus("forked — resolving the first downstream step…");
	openBranchStream();
}

function openBranchStream() {
	closeBranchStream();
	const gen = branchGen;
	const url = new URL(
		`/slots/${encodeURIComponent(currentSlotId)}/${encodeURIComponent(currentModel)}/branch/events?run=${encodeURIComponent(currentRun)}`,
		SERVER_URL,
	);
	const es = new EventSource(url);
	branchSource = es;
	es.onmessage = (ev) => {
		if (gen !== branchGen) return;
		let data;
		try {
			data = JSON.parse(ev.data);
		} catch {
			return;
		}
		dispatchBranchEvent(data);
	};
	es.onerror = () => {
		if (es.readyState === EventSource.CLOSED && branchSource === es)
			branchSource = null;
	};
}

function closeBranchStream() {
	if (branchSource) {
		branchSource.close();
		branchSource = null;
	}
}

function dispatchBranchEvent(e) {
	const kind = e.kind;
	const idx = typeof e.index === "number" ? e.index : -1;
	if (kind === "branch.step.pending") {
		// A new frontier pause (the next, un-run step). A pause means we're not
		// running autonomously.
		branchAuto = false;
		branchSteps.push({
			step: e.step || "(step)",
			owner: e.node || null,
			model: e.model || "",
			system: e.system ?? "",
			user: e.user ?? "",
			output: null,
			reasoning: "",
			ran: false,
			pauseIndex: idx,
			llmIndex: null,
		});
		branchStepBusy = false;
		onBranchStepsGrew();
		return;
	}
	if (kind === "cache.llm") {
		if (idx <= branchDeviationIndex) return;
		// A step ran → commit it. Normally it's the current frontier (last, un-run);
		// under "run rest" (no pauses) there's no frontier, so append it.
		const last = branchSteps[branchSteps.length - 1];
		if (last && !last.ran) {
			last.ran = true;
			if (typeof e.system === "string") last.system = e.system;
			if (typeof e.user === "string") last.user = e.user;
			last.output = e.output ?? null;
			last.reasoning = e.reasoning ?? "";
			last.llmIndex = idx;
		} else {
			branchSteps.push({
				step: e.step || "(step)",
				owner: e.node || null,
				model: e.model || "",
				system: e.system ?? "",
				user: e.user ?? "",
				output: e.output ?? null,
				reasoning: e.reasoning ?? "",
				ran: true,
				pauseIndex: null,
				llmIndex: idx,
			});
		}
		// Stay busy while running autonomously so the controls don't flicker open
		// mid-stream; "run.done" clears it.
		if (!branchAuto) branchStepBusy = false;
		onBranchStepsGrew();
		return;
	}
	if (kind === "bbox" && typeof e.id === "string") {
		if (idx > branchDeviationIndex) addBranchBbox(e);
		return;
	}
	if (kind === "model" && typeof e.id === "string") {
		if (idx > branchDeviationIndex) loadBranchMesh(e.id, e.url);
		return;
	}
	if (kind === "run.done") {
		branchDone = true;
		branchAuto = false;
		branchStepBusy = false;
		if (branchRebuilding) {
			maybeSettleRebuild();
			return;
		}
		if (branchCursor >= 0) renderBranchStep(branchCursor);
		else updateBranchControls();
		return;
	}
	if (kind === "run.error") {
		branchAuto = false;
		branchStepBusy = false;
		branchRebuilding = false; // don't leave the controls wedged if a rebuild errors
		if (branchCursor < 0 && branchSteps.length)
			branchCursor = branchSteps.length - 1;
		setSandboxStatus(`branch error: ${e.message ?? "unknown"}`, "err");
		if (branchCursor >= 0) renderBranchStep(branchCursor);
		else updateBranchControls();
	}
}

// React to branchSteps changing. During a rebuild (after a re-run reopen), hold
// until the re-run target step re-commits, then settle the cursor on it.
// Otherwise focus the first step on the initial pause (carrying the EDIT-mode
// edit) or refresh the currently-viewed step (its controls / result).
function onBranchStepsGrew() {
	if (branchRebuilding) {
		maybeSettleRebuild();
		return;
	}
	if (branchCursor === -1) {
		const carried = branchFirstPrompt;
		branchFirstPrompt = null;
		renderBranchStep(0, carried);
	} else {
		renderBranchStep(branchCursor);
	}
}

// A re-run reopen replays the (fast) committed prefix and then makes a REAL
// (slow) LLM call for the re-run target itself. So we can't settle on a timer —
// we settle once the target step has actually re-committed (or the branch ends),
// landing the cursor on the target's fresh result.
function maybeSettleRebuild() {
	const t = branchReopenTarget;
	const ready =
		branchDone ||
		(t >= 0 &&
			t < branchSteps.length &&
			!!branchSteps[t] &&
			branchSteps[t].ran);
	if (!ready) {
		setSandboxStatus("re-running this step (invalidating later steps)…");
		return;
	}
	branchRebuilding = false;
	branchStepBusy = false;
	const target = branchSteps.length
		? Math.max(0, Math.min(t, branchSteps.length - 1))
		: -1;
	if (target >= 0) renderBranchStep(target);
	else updateBranchControls();
}

// Set the step header (pill + "<verb> <owner> · model").
function setBranchHeader(verb, step, owner, model) {
	sandboxStepPillEl.textContent = step ?? "(step)";
	sandboxStepMetaEl.textContent = "";
	const v = document.createElement("span");
	v.textContent = `${verb} `;
	const o = document.createElement("b");
	o.textContent = owner ?? "—";
	sandboxStepMetaEl.append(v, o);
	if (model) {
		const m = document.createElement("span");
		m.textContent = `   ·   ${model}`;
		sandboxStepMetaEl.append(m);
	}
	sandboxPosEl.textContent = "";
}

// Render the step at `i`. The frontier (un-run, last) shows its editable
// re-rendered prompt to run. A committed step shows the prompt that was
// actually committed to the branch log + its output + reasoning, editable so
// it can be re-run. `override` seeds the textareas (the carried EDIT-mode edit
// on the first step).
function renderBranchStep(i, override) {
	if (i < 0 || i >= branchSteps.length) {
		updateBranchControls();
		return;
	}
	branchCursor = i;
	const s = branchSteps[i];
	const isFrontier = !s.ran;
	setBranchHeader(
		isFrontier ? "paused before" : "committed",
		s.step,
		s.owner,
		s.model,
	);
	sandboxPosEl.textContent = `step ${i + 1} / ${branchSteps.length}`;
	sandboxSystemEl.readOnly = false;
	sandboxUserEl.readOnly = false;
	sandboxSystemEl.value = override
		? (override.system ?? "")
		: (s.system ?? "");
	sandboxUserEl.value = override ? (override.user ?? "") : (s.user ?? "");
	markSandboxEdited();
	if (s.ran && s.output != null) {
		sandboxOutputWrapEl.classList.add("show");
		sandboxOutputEl.textContent = JSON.stringify(s.output, null, 2);
		sandboxRenderNoteEl.textContent = "output of this step";
		sandboxRenderNoteEl.classList.remove("nothing");
		const r = (s.reasoning ?? "").trim();
		sandboxReasoningEl.style.display = r ? "" : "none";
		sandboxReasoningBodyEl.textContent = r;
	} else {
		sandboxOutputWrapEl.classList.remove("show");
		sandboxOutputEl.textContent = "";
		sandboxReasoningEl.style.display = "none";
		sandboxReasoningBodyEl.textContent = "";
	}
	if (isFrontier) {
		setSandboxStatus(
			"paused — edit this step's prompt, then run it (or run the rest)",
		);
	} else {
		setSandboxStatus(
			"committed step — edit + re-run to change it (invalidates later steps)",
		);
	}
	updateBranchControls();
}

function updateBranchControls() {
	if (!branchActive) return;
	const s = branchSteps[branchCursor];
	const onFrontier = !!s && !s.ran;
	const onCommitted = !!s && s.ran;
	const idle = !branchStepBusy && !branchRebuilding;
	// prev / next: pure (non-destructive) observability navigation.
	if (sandboxPrevEl) sandboxPrevEl.disabled = !(idle && branchCursor > 0);
	if (sandboxNextEl)
		sandboxNextEl.disabled = !(
			idle && branchCursor < branchSteps.length - 1
		);
	// Run (frontier) — run the next, un-run step.
	if (sandboxRunStepEl) {
		sandboxRunStepEl.style.display = onFrontier ? "" : "none";
		sandboxRunStepEl.disabled = !(onFrontier && idle && !branchDone);
	}
	// Re-run (committed) — DESTRUCTIVE: invalidates everything after this step.
	if (sandboxRerunEl) {
		sandboxRerunEl.style.display = onCommitted ? "" : "none";
		sandboxRerunEl.disabled = !(onCommitted && idle && s.llmIndex != null);
	}
	// Run rest — only from the frontier.
	if (sandboxRunRestEl) {
		sandboxRunRestEl.style.display = onFrontier ? "" : "none";
		sandboxRunRestEl.disabled = !(onFrontier && idle && !branchDone);
	}
}

// Low-level: tell the server to advance past the current pause. Returns true on
// success. Does not manage busy — callers do.
async function sendBranchProceed(body) {
	try {
		const res = await fetch(
			new URL(
				`/slots/${encodeURIComponent(currentSlotId)}/${encodeURIComponent(currentModel)}/branch/step?run=${encodeURIComponent(currentRun)}`,
				SERVER_URL,
			),
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(body),
			},
		);
		if (!res.ok) {
			setSandboxStatus(`step failed: HTTP ${res.status}`, "err");
			return false;
		}
		return true;
	} catch (err) {
		setSandboxStatus(`step failed: ${err.message}`, "err");
		return false;
	}
}

async function sendBranchRerun(llmIndex, system, user) {
	try {
		const res = await fetch(
			new URL(
				`/slots/${encodeURIComponent(currentSlotId)}/${encodeURIComponent(currentModel)}/branch/rerun?run=${encodeURIComponent(currentRun)}`,
				SERVER_URL,
			),
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ llm_index: llmIndex, system, user }),
			},
		);
		if (!res.ok) {
			setSandboxStatus(`re-run failed: HTTP ${res.status}`, "err");
			return false;
		}
		return true;
	} catch (err) {
		setSandboxStatus(`re-run failed: ${err.message}`, "err");
		return false;
	}
}

// Observability navigation — non-destructive (just moves the viewed step).
function navBranch(delta) {
	if (!branchActive || branchStepBusy || branchRebuilding) return;
	const i = branchCursor + delta;
	if (i < 0 || i >= branchSteps.length) return;
	renderBranchStep(i);
}

// Run the frontier step with the edited prompt; the stream commits it + appends
// the next frontier.
async function runBranchStep() {
	if (!branchActive || branchStepBusy || branchRebuilding) return;
	const s = branchSteps[branchCursor];
	if (!s || s.ran) return;
	branchStepBusy = true;
	updateBranchControls();
	setSandboxStatus("running this step…");
	const ok = await sendBranchProceed({
		system: sandboxSystemEl.value,
		user: sandboxUserEl.value,
	});
	if (!ok) {
		branchStepBusy = false;
		updateBranchControls();
	}
}

// Run the rest of the branch from the frontier with no further pauses.
async function runBranchRest() {
	if (!branchActive || branchStepBusy || branchRebuilding || branchDone)
		return;
	const s = branchSteps[branchCursor];
	if (!s || s.ran) return; // only from the frontier
	branchStepBusy = true;
	branchAuto = true;
	updateBranchControls();
	setSandboxStatus("running the rest of the branch…");
	const ok = await sendBranchProceed({
		system: sandboxSystemEl.value,
		user: sandboxUserEl.value,
		auto: true,
	});
	if (!ok) {
		branchStepBusy = false;
		branchAuto = false;
		updateBranchControls();
	}
}

// Re-run a COMMITTED step with the edited prompt: invalidates everything after
// it (server truncates + replays), then re-syncs from the rebuilt branch log,
// landing back on this step's new result.
async function reRunBranchStep() {
	if (!branchActive || branchStepBusy || branchRebuilding) return;
	const s = branchSteps[branchCursor];
	if (!s || !s.ran || s.llmIndex == null) return;
	const target = branchCursor;
	branchStepBusy = true;
	branchAuto = false; // re-run drops us back into interactive stepping
	updateBranchControls();
	setSandboxStatus("re-running this step (invalidating later steps)…");
	const ok = await sendBranchRerun(
		s.llmIndex,
		sandboxSystemEl.value,
		sandboxUserEl.value,
	);
	if (!ok) {
		branchStepBusy = false;
		updateBranchControls();
		return;
	}
	// Rebuild from the truncated + re-run log; settle the cursor back on this step.
	branchReopenTarget = target;
	branchRebuilding = true;
	branchSteps = [];
	branchCursor = -1;
	branchDone = false;
	branchGen += 1; // invalidate stale mesh loads from the undone steps
	branchOverlayBboxIds.clear();
	branchOverlayMeshes.clear();
	clearSandboxOverlay();
	openBranchStream(); // fresh snapshot rebuilds the overlay + the step history
}

// Draw a deviated branch node's (world-frame) bbox into the overlay. The branch
// emits world coordinates already, so no parent-frame conversion is needed.
function addBranchBbox(e) {
	if (branchOverlayBboxIds.has(e.id)) return;
	if (!Array.isArray(e.origin) || !Array.isArray(e.dimensions)) return;
	branchOverlayBboxIds.add(e.id);
	addSandboxOverlayBox(e.origin, e.dimensions, {
		id: e.id,
		kind: e.node_kind ?? "object",
		prompt: typeof e.prompt === "string" ? e.prompt : null,
	});
}

async function loadBranchMesh(id, url) {
	if (!url) return;
	const gen = branchGen;
	let gltf;
	try {
		gltf = await loader.loadAsync(new URL(url, SERVER_URL).toString());
	} catch {
		return;
	}
	if (gen !== branchGen || !branchActive) {
		disposeObject3D(gltf.scene);
		return;
	}
	gltf.scene.traverse((child) => {
		if (child.isMesh && child.material) {
			const mats = Array.isArray(child.material)
				? child.material
				: [child.material];
			for (const m of mats) m.side = THREE.DoubleSide;
		}
	});
	const prev = branchOverlayMeshes.get(id);
	if (prev) {
		sandboxOverlayRoot.remove(prev);
		disposeObject3D(prev);
	}
	sandboxOverlayRoot.add(gltf.scene);
	branchOverlayMeshes.set(id, gltf.scene);
}

// Local-only branch teardown (no DELETE) — shared by discardBranch + the silent
// cell-switch teardown.
function clearBranchStateLocal() {
	branchActive = false;
	branchGen += 1; // invalidate any pending mesh loads
	branchDone = false;
	branchSteps = [];
	branchCursor = -1;
	branchRebuilding = false;
	branchStepBusy = false;
	branchAuto = false;
	branchFirstPrompt = null;
	closeBranchStream();
	branchOverlayBboxIds.clear();
	branchOverlayMeshes.clear();
	clearSandboxOverlay();
}

async function discardBranch() {
	if (!branchActive) return;
	const slotId = currentSlotId,
		model = currentModel,
		run = currentRun;
	clearBranchStateLocal();
	try {
		await fetch(
			new URL(
				`/slots/${encodeURIComponent(slotId)}/${encodeURIComponent(model)}/branch?run=${encodeURIComponent(run)}`,
				SERVER_URL,
			),
			{ method: "DELETE" },
		);
	} catch {}
}

// Phase B -> Phase A: drop the branch and return to tuning the step we forked.
async function backToEdit() {
	if (!branchActive) return;
	const cursor = sandboxCursor;
	await discardBranch();
	applySandboxMode();
	gotoSandboxStep(cursor);
}

// Break out: discard all tuning, restore the full scene instantly (the objects
// were only hidden), resume the run if we paused it, and reconnect the SSE if
// the run had been live.
async function exitSandbox() {
	if (!sandboxActive) return;
	// Capture the cell we're restoring — clearing `sandbox-open` below re-enables
	// the chrome, so the user could switch cells during the resume await; the
	// resume must target the cell we paused, and the reconnect must bail if we've
	// since navigated away.
	const slotId = currentSlotId;
	const model = currentModel;
	const run = currentRun;
	// Drop any active branch (cancels its server task + deletes the temp dir).
	if (branchActive) await discardBranch();
	sandboxActive = false;
	rewindCutoffIndex = null;
	clearSandboxOverlay();
	refreshAllVisibility();
	sandboxPanelEl.classList.remove("open");
	document.body.classList.remove("sandbox-open");
	sandboxSteps = [];
	sandboxCursor = -1;
	const resume = sandboxPausedByUs;
	const reconnect = sandboxWasLive;
	sandboxPausedByUs = false;
	sandboxWasLive = false;
	if (resume && slotId && model) {
		try {
			await fetch(
				new URL(
					`/slots/${encodeURIComponent(slotId)}/${encodeURIComponent(model)}/resume?run=${encodeURIComponent(run)}`,
					SERVER_URL,
				),
				{ method: "POST" },
			);
		} catch {}
	}
	// Only reconnect if we're still viewing the same cell (a switch during the
	// await already tore down + reloaded its own stream).
	if (
		reconnect &&
		currentSlotId === slotId &&
		currentModel === model &&
		!currentSource
	) {
		setStatus(
			"tuning discarded — original run restored, streaming events…",
		);
		subscribe(`${slotEventsUrl(slotId, model)}&since=${highestEventIndex}`);
	} else if (currentSlotId === slotId && currentModel === model) {
		setStatus("tuning discarded — original run restored");
	}
	refreshSlots();
}

// Hard teardown with no awaited server calls — used when the cell itself is
// going away (slot/model/run switch, reset). The follow-up clearScene wipes the
// overlay. Fires a best-effort branch DELETE for the cell we're leaving.
function teardownSandboxSilently() {
	if (!sandboxActive && !branchActive) return;
	if (branchActive) {
		const slotId = currentSlotId,
			model = currentModel,
			run = currentRun;
		clearBranchStateLocal();
		if (slotId && model && run) {
			try {
				fetch(
					new URL(
						`/slots/${encodeURIComponent(slotId)}/${encodeURIComponent(model)}/branch?run=${encodeURIComponent(run)}`,
						SERVER_URL,
					),
					{ method: "DELETE", keepalive: true },
				).catch(() => {});
			} catch {}
		}
	}
	sandboxActive = false;
	rewindCutoffIndex = null;
	clearSandboxOverlay();
	sandboxPanelEl.classList.remove("open");
	document.body.classList.remove("sandbox-open");
	sandboxSteps = [];
	sandboxCursor = -1;
	sandboxPausedByUs = false;
	sandboxWasLive = false;
}

function updateFlowPauseButton() {
	if (!flowPauseEl) return;
	const running = currentRunInfo()?.status === "running";
	flowPauseEl.disabled = !running;
	flowPauseEl.classList.toggle("is-running", running);
}

async function pauseCurrentCell() {
	if (!currentSlotId || !currentModel) return;
	if (currentRunInfo()?.status !== "running") return;
	flowPauseEl.disabled = true;
	try {
		const res = await fetch(
			new URL(
				`/slots/${encodeURIComponent(currentSlotId)}/${encodeURIComponent(currentModel)}/pause?run=${encodeURIComponent(currentRun)}`,
				SERVER_URL,
			),
			{ method: "POST" },
		);
		if (res.ok) {
			// slot_pause flips the cell to "paused" synchronously, so awaiting the
			// status refresh before re-reading lets the button settle correctly
			// (the run.paused SSE event also lands and closes the live stream).
			setStatus(
				"run paused — open a step and 'tune' to rewind & edit its prompt",
			);
			await refreshSlots();
		} else {
			setStatus(`pause failed: HTTP ${res.status}`, "err");
		}
	} catch (e) {
		setStatus(`pause failed: ${e.message}`, "err");
	}
	updateFlowPauseButton();
}

sandboxTestEl?.addEventListener("click", testSandboxStep);
sandboxResetEl?.addEventListener("click", () => {
	const call = sandboxSteps[sandboxCursor];
	if (!call) return;
	sandboxSystemEl.value = call.system ?? "";
	sandboxUserEl.value = call.user ?? "";
	markSandboxEdited();
	setSandboxStatus("prompts reset to the recorded values");
});
sandboxPrevEl?.addEventListener("click", () =>
	branchActive ? navBranch(-1) : gotoSandboxStep(sandboxCursor - 1),
);
sandboxNextEl?.addEventListener("click", () =>
	branchActive ? navBranch(1) : gotoSandboxStep(sandboxCursor + 1),
);
sandboxSimulateEl?.addEventListener("click", simulateDownstream);
sandboxRunStepEl?.addEventListener("click", runBranchStep);
sandboxRerunEl?.addEventListener("click", reRunBranchStep);
sandboxRunRestEl?.addEventListener("click", runBranchRest);
sandboxBackEl?.addEventListener("click", backToEdit);
sandboxBreakoutEl?.addEventListener("click", () => exitSandbox());
sandboxCloseEl?.addEventListener("click", () => exitSandbox());
sandboxSystemEl?.addEventListener("input", markSandboxEdited);
sandboxUserEl?.addEventListener("input", markSandboxEdited);
sandboxExpandEl?.addEventListener("click", () => {
	sandboxExpanded = !sandboxExpanded;
	try {
		localStorage.setItem(SANDBOX_EXPANDED_KEY, sandboxExpanded ? "1" : "0");
	} catch {}
	applySandboxExpanded();
});
sandboxCopyEl?.addEventListener("click", async () => {
	const text = sandboxOutputEl.textContent || "";
	if (!text) return;
	try {
		await navigator.clipboard.writeText(text);
		sandboxCopyEl.textContent = "✓ copied";
		setTimeout(() => {
			sandboxCopyEl.textContent = "⧉ copy";
		}, 1200);
	} catch {
		setSandboxStatus(
			"copy failed — select the output text manually",
			"err",
		);
	}
});
flowPauseEl?.addEventListener("click", pauseCurrentCell);

function positionTooltip(clientX, clientY, id) {
	const node = treeNodes.get(id);
	const kind = node?.kind ?? "zone";
	const kindColor = TOOLTIP_KIND_COLOR[kind] ?? "#e6e6e6";

	// Build with DOM nodes rather than innerHTML so the prompt (LLM-authored
	// text) can't smuggle markup into the tooltip.
	tooltip.textContent = "";
	const head = document.createElement("div");
	const kindEl = document.createElement("span");
	kindEl.textContent = `[${kind}]`;
	kindEl.style.color = kindColor;
	head.appendChild(kindEl);
	head.appendChild(document.createTextNode(` ${id}`));
	tooltip.appendChild(head);

	// Sections: each carries a small label so the user can tell which prompt
	// came from which pipeline step. Order is "earliest in the pipeline first"
	// so reading top-to-bottom matches the order of LLM rewrites.
	//   seed  — zone_decompose / object_decomp output (what the LLM was given
	//           as the brief for this node)
	//   plan  — zone_plan output (zones only)
	//   image — image-prompt noun phrase actually sent to Banana+Trellis
	//           (objects/frames only, once generated)
	const sections = [];
	if (node?.prompt) sections.push(["seed", node.prompt]);
	if (kind === "zone" && node?.plan) sections.push(["plan", node.plan]);
	if (
		kind !== "zone" &&
		node?.imagePrompt &&
		node.imagePrompt !== node.prompt
	) {
		sections.push(["image", node.imagePrompt]);
	}
	for (const [label, text] of sections) {
		const row = document.createElement("div");
		row.style.marginTop = "4px";
		row.style.color = "#bdbdbd";
		const lbl = document.createElement("span");
		lbl.textContent = `${label}: `;
		lbl.style.color = "#7a8190";
		row.appendChild(lbl);
		row.appendChild(document.createTextNode(text));
		tooltip.appendChild(row);
	}

	placeTooltip(clientX, clientY);
}

// Flip left/up when the tooltip would overflow the viewport so the cursor can
// keep approaching the hovered box from any direction. Assumes tooltip content
// is already set.
function placeTooltip(clientX, clientY) {
	tooltip.style.display = "block";
	tooltip.style.left = "0px";
	tooltip.style.top = "0px";
	const w = tooltip.offsetWidth;
	const h = tooltip.offsetHeight;
	const pad = 12;
	let x = clientX + pad;
	let y = clientY + pad;
	if (x + w > window.innerWidth) x = clientX - pad - w;
	if (y + h > window.innerHeight) y = clientY - pad - h;
	tooltip.style.left = `${Math.max(0, x)}px`;
	tooltip.style.top = `${Math.max(0, y)}px`;
}

// Tooltip for a magenta overlay box — same look as the canvas bbox tooltip,
// built from the box's own metadata (the branch/tested node isn't in the tree).
function positionOverlayTooltip(clientX, clientY, oid) {
	const entry = sandboxOverlayBoxes.get(oid);
	if (!entry) {
		tooltip.style.display = "none";
		return;
	}
	const kind = entry.kind;
	tooltip.textContent = "";
	const head = document.createElement("div");
	if (kind) {
		const kindEl = document.createElement("span");
		kindEl.textContent = `[${kind}]`;
		kindEl.style.color = TOOLTIP_KIND_COLOR[kind] ?? "#e6e6e6";
		head.appendChild(kindEl);
		head.appendChild(document.createTextNode(" "));
	}
	head.appendChild(document.createTextNode(entry.id ?? "(box)"));
	tooltip.appendChild(head);
	if (entry.prompt) {
		const row = document.createElement("div");
		row.style.marginTop = "4px";
		row.style.color = "#bdbdbd";
		const lbl = document.createElement("span");
		lbl.textContent = "seed: ";
		lbl.style.color = "#7a8190";
		row.appendChild(lbl);
		row.appendChild(document.createTextNode(entry.prompt));
		tooltip.appendChild(row);
	}
	placeTooltip(clientX, clientY);
}

// Zones-only picking is active when the toggle is set to "zones" OR the user
// is holding Shift — a momentary override to grab the containing zone without
// flipping (and persisting) the saved mode.
function zonesOnlyActive() {
	return selectMode === "zones" || pressedKeys.has("shift");
}

// Mesh-based picking: raycast against actual geometry the user sees —
// loaded GLB meshes first, then solid-fill proxies when the model hasn't
// arrived. Zones never have meshes, so they're unreachable here and must
// be selected from the tree (or via zone-mode picking, below).
const _pickRoots = [];
function pickHoveredBboxId() {
	if (zonesOnlyActive()) return pickHoveredZoneBboxId();
	_pickRoots.length = 0;
	for (const model of modelsById.values()) {
		if (model.visible) _pickRoots.push(model);
	}
	for (const fill of solidFills.values()) {
		if (fill.visible) _pickRoots.push(fill);
	}
	if (_pickRoots.length === 0) return null;
	const hits = raycaster.intersectObjects(_pickRoots, true);
	for (const hit of hits) {
		let node = hit.object;
		while (node) {
			const pid = node.userData?.pickId;
			if (pid != null && !effectivelyHidden(pid)) return pid;
			node = node.parent;
		}
	}
	return null;
}

// Zone-only picker. Zones nest (children fit inside their parent and
// siblings don't overlap), so the smallest-volume zone whose AABB the ray
// crosses is the deepest one containing the click — which is what the user
// wants when they say "select this zone." Hidden zones are skipped.
const _zoneHit = new THREE.Vector3();
const _zoneSize = new THREE.Vector3();
function pickHoveredZoneBboxId() {
	let bestId = null;
	let bestVol = Infinity;
	for (const [id, helper] of bboxes) {
		if (helper.userData.nodeKind !== "zone") continue;
		if (effectivelyHidden(id)) continue;
		if (!withinRewind(id, "bbox")) continue; // don't grab a rewound-away zone
		if (!raycaster.ray.intersectBox(helper.box, _zoneHit)) continue;
		helper.box.getSize(_zoneSize);
		const vol = _zoneSize.x * _zoneSize.y * _zoneSize.z;
		if (vol < bestVol) {
			bestVol = vol;
			bestId = id;
		}
	}
	return bestId;
}

// Right-click picker. Tries the mesh picker first, then falls back to a
// bbox raycast so a hidden object (whose bbox is still visible) can be
// re-clicked to bring its mesh back. The fallback intentionally does NOT
// filter by `effectivelyHidden` — hidden ids are exactly what we want to
// be able to click on the bbox to un-hide.
const _rightClickBoxHit = new THREE.Vector3();
function pickRightClickId() {
	const meshId = pickHoveredBboxId();
	if (meshId !== null) return meshId;
	let bestId = null;
	let bestDist = Infinity;
	for (const [id, helper] of bboxes) {
		if (!helper.visible) continue;
		// In zone-only mode the bbox fallback must also stay zone-restricted —
		// otherwise right-clicking past a zone hit would hide a non-zone the
		// user can't even select via left-click.
		if (zonesOnlyActive() && helper.userData.nodeKind !== "zone") continue;
		if (!raycaster.ray.intersectBox(helper.box, _rightClickBoxHit))
			continue;
		const dist = _rightClickBoxHit.distanceToSquared(camera.position);
		if (dist < bestDist) {
			bestDist = dist;
			bestId = id;
		}
	}
	return bestId;
}

renderer.domElement.addEventListener("pointermove", (ev) => {
	const rect = renderer.domElement.getBoundingClientRect();
	pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
	pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
	lastPointerClientX = ev.clientX;
	lastPointerClientY = ev.clientY;
	pointerInsideCanvas = true;
	pointerDirty = true;
});

renderer.domElement.addEventListener("pointerleave", () => {
	pointerInsideCanvas = false;
	setHoveredBbox(null);
	setHoveredOverlay(null);
	tooltip.style.display = "none";
});

// Click-to-select on the canvas. OrbitControls also listens for pointerdown
// to start orbiting, so we have to distinguish a click from the end of an
// orbit drag. Approach: snapshot the down position + time, and only treat
// pointerup as a selection click when the cursor barely moved and the gesture
// was short. Anything more is a camera drag — pass through to OrbitControls.
const CLICK_MAX_MOVE_PX = 4;
const CLICK_MAX_DURATION_MS = 400;
let _downX = 0;
let _downY = 0;
let _downT = 0;
let _downButton = -1;

renderer.domElement.addEventListener("pointerdown", (ev) => {
	_downX = ev.clientX;
	_downY = ev.clientY;
	_downT = performance.now();
	_downButton = ev.button;
});

renderer.domElement.addEventListener("pointerup", (ev) => {
	if (_downButton !== 0 || ev.button !== 0) return; // left-click only
	const dx = ev.clientX - _downX;
	const dy = ev.clientY - _downY;
	const dt = performance.now() - _downT;
	if (Math.hypot(dx, dy) > CLICK_MAX_MOVE_PX || dt > CLICK_MAX_DURATION_MS)
		return;

	// Reuse the hover picker so selecting matches whatever the hover
	// tooltip is showing.
	const rect = renderer.domElement.getBoundingClientRect();
	pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
	pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
	raycaster.setFromCamera(pointer, camera);
	if (sandboxActive) {
		// Select the magenta overlay box (highlight it); the tooltip already names it.
		selectOverlay(pickHoveredOverlayId());
		return;
	}
	const id = pickHoveredBboxId();
	if (id !== null) {
		selectTreeNode(id);
		// Scroll the corresponding tree row into view so the link between the
		// 3D click and the (now-open) detail panel is obvious if the user closes
		// the detail view.
		const row = treeBodyEl.querySelector(
			`.tree-node[data-id="${CSS.escape(id)}"]`,
		);
		if (row) row.scrollIntoView({ block: "nearest" });
	}
});

// Right-click toggles per-node hide for the picked id — same Set as the
// tree's per-row visibility button, so the two stay in sync. The mesh
// disappears, the bbox stays as a volumetric reference and as the click
// target for un-hiding. Suppresses the browser's default context menu.
renderer.domElement.addEventListener("contextmenu", (ev) => {
	ev.preventDefault();
	const rect = renderer.domElement.getBoundingClientRect();
	pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
	pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
	raycaster.setFromCamera(pointer, camera);
	const id = pickRightClickId();
	if (id !== null) toggleNodeHidden(id);
});

// --- event dispatch ---------------------------------------------------------

// Event-index high-water mark. The server re-replays the entire snapshot
// from index 0 on every SSE (re)connect. EventSource auto-reconnects on its
// own (server idle, network blip), so without this guard we'd wipe and
// reload every model on every reconnect — which is exactly the "models keep
// reloading" behaviour the user reports when inspecting a finished run. We
// dedupe by index instead and let already-processed events fall through.
// Reset to -1 only on explicit user-driven state wipes (slot switch, reset,
// rewind), where a fresh replay genuinely needs to be re-applied.
let highestEventIndex = -1;

// Buffered event log for the active slot — captured as events stream in so
// the replay-to-gif feature can re-dispatch the build from scratch. Reset by
// the same hooks that reset `highestEventIndex` (slot switch / rewind /
// reset). Stored in dispatch order, deduped by index.
const recordedEvents = [];

function dispatch(event) {
	// While the prompt-tuning sandbox is open the live stream is detached and
	// client state is frozen — ignore any stray event so recordedEvents / tree /
	// scene stay exactly as they were. Break-out reconnects with ?since= to
	// catch up, so nothing is lost.
	if (sandboxActive) return;
	if (typeof event.index === "number") {
		if (event.index <= highestEventIndex) return;
		highestEventIndex = event.index;
	}
	recordedEvents.push(event);
	updateReplayButton();
	appendEvent(event);
	switch (event.kind) {
		case "run.start":
			setStatus(`run :: ${event.model}`);
			break;
		case "run.done":
			runFinished = true;
			if (meshErrors.size > 0) showRunCompleteWithErrors();
			else setStatus("run complete");
			if (currentSource) {
				currentSource.close();
				currentSource = null;
			}
			refreshSlots();
			break;
		case "run.error":
			runFinished = true;
			setStatus(`error: ${event.message}`, "err");
			if (currentSource) {
				currentSource.close();
				currentSource = null;
			}
			refreshSlots();
			break;
		case "run.paused":
			runFinished = true;
			setStatus("paused");
			if (currentSource) {
				currentSource.close();
				currentSource = null;
			}
			refreshSlots();
			break;
		case "mesh.error":
			// Surface the failure: track for the run.done summary, paint the
			// tree node + asset card as errored so users see it without grepping
			// the log panel.
			retryingIds.delete(event.id);
			meshErrors.set(event.id, event.message ?? "unknown error");
			treeSetPhase(event.id, "error");
			upsertAsset(event.id, {
				status: "error",
				errorMessage: event.message,
			});
			refreshPostRunStatus();
			break;
		case "mesh.retry":
			// Server-side retry kickoff. The user may have triggered it from
			// *this* client (retryingIds already set) or another tab; either
			// way, clear the prior error state so the asset/tree both flip back
			// to in-flight and the retry button greys out.
			retryingIds.add(event.id);
			meshErrors.delete(event.id);
			upsertAsset(event.id, { status: "pending", errorMessage: null });
			if (treeNodes.has(event.id))
				treeSetPhase(event.id, "generating_mesh");
			refreshPostRunStatus();
			break;
		case "bbox":
			loadBbox(event);
			treeUpsert(event.id, {
				parentId: event.parent_id ?? null,
				prompt: event.prompt ?? null,
				kind: event.node_kind ?? "zone",
				origin: event.origin,
				dimensions: event.dimensions,
				proxyShape: event.proxy_shape ?? null,
			});
			scheduleRenderTree();
			if (event.id === selectedBboxId) renderTreeDetail();
			break;
		case "divider.decompose":
		case "divider.zone_decompose":
			// Pre-declare children so the tree shows them (in pending state) before
			// their bboxes resolve. Use the child's structural parent when present;
			// sibling-parented zones should not be shown under the emitting node.
			for (const c of event.children ?? []) {
				treeUpsert(c.id, {
					parentId: c.parent ?? event.node,
					prompt: c.prompt,
					kind: "zone",
				});
			}
			scheduleRenderTree();
			break;
		case "divider.zone_plan":
			// Stash the authored zone plan on the node so the tooltip can surface
			// it. The plan often arrives before the bbox is resolved, so upsert
			// (which tolerates a half-formed node) rather than gating on existence.
			if (event.node && typeof event.plan === "string") {
				treeUpsert(event.node, { plan: event.plan });
				if (event.node === selectedBboxId) renderTreeDetail();
			}
			break;
		case "step":
			treeSetPhase(event.node, event.phase);
			break;
		case "mesh.submit":
			// Object mesh generation kicked off — show it on the tree.
			treeSetPhase(event.id, "generating_mesh");
			break;
		case "image":
			upsertAsset(event.id, {
				imageUrl: event.url,
				prompt: event.prompt,
			});
			// Image-prompt noun phrase (post-rewrite) — distinct from the seed
			// prompt stored on the bbox event. Tooltip + detail panel show both.
			if (typeof event.prompt === "string") {
				treeUpsert(event.id, { imagePrompt: event.prompt });
				if (event.id === selectedBboxId) renderTreeDetail();
			}
			break;
		case "model":
			loadModel(event);
			treeSetPhase(event.id, "done");
			retryingIds.delete(event.id);
			// A `model` event for this id is proof the mesh exists now, so any
			// prior `mesh.error` for the same id is stale. Without this clear,
			// snapshot replay (where a past error then a later success both
			// appear in the recorded log) leaves `meshErrors` permanently
			// overcounted — the tree shows `done` but the run-complete summary
			// still reads "N meshes failed". Equally applies to in-pipeline
			// recoveries that ship a `model` without a preceding `mesh.retry`.
			meshErrors.delete(event.id);
			if (event.id === selectedBboxId) renderTreeDetail();
			refreshPostRunStatus();
			break;
		case "cache.llm":
			recordLlmCall(event);
			break;
		// Everything else is already shown as a log line above.
	}
}

// --- slot picker + model picker + run lifecycle ------------------------------

// Every slot has N parallel runs — one per model alias from `availableModels`.
// The viewer shows one (slot, model) cell at a time; switching either
// dimension closes the active SSE, clears the scene, and reconnects to that
// cell's stream. The Trellis queue is global, so other cells stay in-flight
// on the same Modal pool while we're looking at one of them.
//
// The state these functions read/mutate (currentRun / currentSlotId /
// currentModel / availableModels / …) is declared near the top of the module so
// init-time code — notably the queue panel's first render — can read it without
// hitting the temporal dead zone.

// --- run tabs (multiple open run-views) -------------------------------------
//
// Each tab pins a (run, slot, model) position; the active tab is what the
// single canvas shows. Switching tabs reuses switchRun/switchView, so only one
// cell streams at a time (one WebGL context, one SSE) while the others stay one
// click away — the natural way to A/B-compare renditions across runs. The run
// picker, version bar, slot bar, and model picker all edit the *active* tab
// (via syncActiveTab in switchView); openTabs persists across reloads.
const openTabs = []; // [{ id, run, slot, model }]
let activeTabId = null;

function currentRunInfo() {
	const slot = slotSummaries.find((s) => s.id === currentSlotId);
	return slot?.runs?.[currentModel] ?? null;
}

function renderSlotTabs() {
	for (const child of Array.from(slotBarEl.querySelectorAll(".slot-tab"))) {
		child.remove();
	}
	for (const s of slotSummaries) {
		const tab = document.createElement("button");
		tab.type = "button";
		tab.className = "slot-tab" + (s.id === currentSlotId ? " active" : "");
		tab.dataset.slotId = s.id;
		tab.title = s.prompt ?? "";

		const status = s.runs?.[currentModel]?.status ?? "idle";
		const dot = document.createElement("span");
		dot.className = `slot-dot status-${status}`;
		tab.appendChild(dot);

		const label = document.createElement("span");
		label.textContent = s.id;
		tab.appendChild(label);

		tab.addEventListener("click", () => switchSlot(s.id));
		slotBarEl.insertBefore(tab, slotBarToggleEl);
	}
}

function populateModelPicker() {
	modelPickerEl.innerHTML = "";
	for (const alias of availableModels) {
		const opt = document.createElement("option");
		opt.value = alias;
		opt.textContent = alias;
		modelPickerEl.appendChild(opt);
	}
	if (currentModel) modelPickerEl.value = currentModel;
}

function updateResumeButton() {
	const status = currentRunInfo()?.status;
	resumeEl.className = "";
	resumeEl.style.display = "none";
	if (status === "idle") {
		resumeEl.style.display = "";
		resumeEl.className = "paused";
		resumeEl.textContent = "start";
		resumeEl.title = "Start this run";
	} else if (status === "paused") {
		resumeEl.style.display = "";
		resumeEl.className = "paused";
		resumeEl.textContent = "resume";
		resumeEl.title = "Resume the interrupted run";
	} else if (status === "error") {
		resumeEl.style.display = "";
		resumeEl.className = "error";
		resumeEl.textContent = "retry";
		resumeEl.title = "Retry the failed run";
	} else if (status === "running") {
		resumeEl.style.display = "";
		resumeEl.className = "running";
		resumeEl.textContent = "pause";
		resumeEl.title = "Pause this run";
	} else if (
		slotNeedsResume &&
		currentSlotId !== null &&
		currentModel !== null
	) {
		slotNeedsResume = false;
		subscribe(slotEventsUrl(currentSlotId, currentModel));
	}
}

async function refreshSlots() {
	try {
		// Scope the status poll to OUR viewed run. Every cell request is
		// run-scoped now, so we must ask for this version's statuses explicitly
		// rather than whatever run happens to be the server's last-activated
		// global — otherwise another tab/version flips our dots out from under us.
		const url = new URL("/slots", SERVER_URL);
		if (currentRun) url.searchParams.set("run", currentRun);
		const res = await fetch(url);
		if (!res.ok) return;
		const payload = await res.json();
		availableModels = payload.models ?? [];
		defaultModelAlias = payload.default_model ?? availableModels[0] ?? null;
		slotSummaries = payload.slots ?? [];
		if (payload.run && !currentRun) {
			// First load only: adopt the server's default run. We never *steal* a
			// run another client activated — requests are run-scoped, so a global
			// flip elsewhere no longer means our view should follow it.
			currentRun = payload.run;
			if (runPickerEl.value !== currentRun)
				runPickerEl.value = currentRun;
		}
		// Re-populate the picker if the model list changed (or this is the
		// first refresh). Cheap to redo every tick — the <option>s are flat.
		if (modelPickerEl.options.length !== availableModels.length) {
			populateModelPicker();
		}
		renderSlotTabs();
		updateResumeButton();
		// Keep the flow-modal pause button in sync with the live run status
		// (e.g. when a run finishes on its own while the graph is open).
		updateFlowPauseButton();
	} catch {
		// Transient; next tick will retry.
	}
}

function populateRunPicker() {
	runPickerEl.innerHTML = "";
	for (const r of availableRuns) {
		const opt = document.createElement("option");
		opt.value = r.name;
		opt.textContent = r.has_prompt_snapshot
			? r.name
			: `${r.name} (no snapshot)`;
		runPickerEl.appendChild(opt);
	}
	if (currentRun) runPickerEl.value = currentRun;
}

async function refreshRuns() {
	try {
		const res = await fetch(new URL("/runs", SERVER_URL));
		if (!res.ok) return;
		const payload = await res.json();
		availableRuns = payload.runs ?? [];
		if (!currentRun) currentRun = payload.current ?? null;
		populateRunPicker();
	} catch {
		// Transient; next tick will retry.
	}
}

function resetClientStateForRunSwitch() {
	// Switching to a different run means every cached scene/log/asset/tree
	// entry is stale — they belong to the prior run's (slot, model) cells.
	// Tear down before we refetch slots and re-subscribe.
	if (currentSource) {
		currentSource.close();
		currentSource = null;
	}
	clearScene();
	clearLog();
	clearAssets();
	treeClear();
	clearMeshErrors();
	highestEventIndex = -1;
	recordedEvents.length = 0;
	updateReplayButton();
}

async function switchRun(name) {
	if (!name || name === currentRun) return;
	teardownSandboxSilently();
	runPickerEl.disabled = true;
	try {
		const res = await fetch(
			new URL(`/runs/${encodeURIComponent(name)}/activate`, SERVER_URL),
			{ method: "POST" },
		);
		if (!res.ok) {
			setStatus(`run switch failed: HTTP ${res.status}`, "err");
			// Snap the picker back to whatever the server still thinks is active.
			if (currentRun) runPickerEl.value = currentRun;
			return;
		}
		currentRun = name;
		setStatus(`run :: ${name}`);
		resetClientStateForRunSwitch();
		await refreshSlots();
		if (currentSlotId && currentModel) {
			// Re-render the same (slot, model) selection, now backed by the new
			// run's events.jsonl + idle/paused/done state.
			switchView(currentSlotId, currentModel);
		}
	} finally {
		runPickerEl.disabled = false;
	}
}

// --- pipeline versions (V1/V2/V3/V4) ----------------------------------------
//
// Each version is a reserved run (v1-legacy-xml / v2-frame-first /
// v3-decomp-first / v4-decomp-first-all). The version bar is a specialized
// run-switcher: clicking a button activates that run via switchRun()
// (single-canvas swap), and "launch all" starts every version's cell on the
// current (slot, model) so they generate concurrently and isolated. The dot on
// each button mirrors that version's cell status for the viewed (slot, model).

function renderVersionBar() {
	versionBarEl.innerHTML = "";
	for (const v of availableVersions) {
		const tab = document.createElement("button");
		tab.type = "button";
		tab.className =
			"version-tab" + (v.run_name === currentRun ? " active" : "");
		tab.title = `${v.label} — run "${v.run_name}"`;
		const dot = document.createElement("span");
		dot.className = `slot-dot status-${v.status ?? "idle"}`;
		tab.appendChild(dot);
		const label = document.createElement("span");
		label.textContent = v.label;
		tab.appendChild(label);
		tab.addEventListener("click", () => selectVersion(v.run_name));
		versionBarEl.appendChild(tab);
	}
}

async function refreshVersions() {
	try {
		const url = new URL("/versions", SERVER_URL);
		if (currentSlotId) url.searchParams.set("slot", currentSlotId);
		if (currentModel) url.searchParams.set("model", currentModel);
		const res = await fetch(url);
		if (!res.ok) return;
		const payload = await res.json();
		availableVersions = payload.versions ?? [];
		renderVersionBar();
	} catch {
		// Transient; next tick will retry.
	}
}

async function selectVersion(runName) {
	// Persist the choice and swap which version the single canvas is viewing.
	// switchRun handles activate + teardown + re-subscribe; the other versions'
	// background tasks are untouched and keep generating.
	try {
		localStorage.setItem(VERSION_STORAGE_KEY, runName);
	} catch {}
	await switchRun(runName);
	renderVersionBar();
}

async function launchAllVersions() {
	// Start all three versions on the current (slot, model) cell so they run
	// concurrently and isolated, then re-subscribe the viewed cell's stream.
	if (currentSlotId === null || currentModel === null) {
		setStatus("pick a slot + model before launching versions", "warn");
		return;
	}
	versionLaunchAllEl.disabled = true;
	try {
		const res = await fetch(
			new URL(
				`/versions/${encodeURIComponent(currentSlotId)}/${encodeURIComponent(currentModel)}/launch`,
				SERVER_URL,
			),
			{ method: "POST" },
		);
		if (!res.ok) {
			setStatus(`launch all failed: HTTP ${res.status}`, "err");
			return;
		}
		const payload = await res.json();
		const started = (payload.versions ?? []).filter(
			(v) => v.started,
		).length;
		setStatus(
			`launched ${started}/3 versions on ${currentSlotId} · ${currentModel} — streaming current…`,
		);
		// Mirror resumeSlot: tear the viewed cell down and re-subscribe so its
		// freshly-started stream shows immediately. The other versions stream in
		// the background; switch to them with the version buttons.
		slotNeedsResume = false;
		resetClientStateForRunSwitch();
		subscribe(slotEventsUrl(currentSlotId, currentModel));
		refreshSlots();
		refreshVersions();
	} catch (e) {
		setStatus(`launch all failed: ${e.message}`, "err");
	} finally {
		versionLaunchAllEl.disabled = false;
	}
}

versionLaunchAllEl.addEventListener("click", launchAllVersions);

async function archiveAllVersions() {
	// Copy every version run (V1/V2/V3) that has data into timestamped,
	// loadable archive runs, so the live version cells can be reset and re-run
	// for a fresh V1 vs V2 without losing the current rendition. Stays on the
	// current version; archives appear in the run picker and stream meshes from
	// their own dir, so each copy is self-contained.
	versionArchiveAllEl.disabled = true;
	const prevLabel = versionArchiveAllEl.textContent;
	versionArchiveAllEl.textContent = "archiving…";
	try {
		const res = await fetch(new URL("/versions/snapshot", SERVER_URL), {
			method: "POST",
		});
		if (!res.ok) {
			const detail = await res.text();
			setStatus(`archive failed: ${detail}`, "err");
			return;
		}
		const payload = await res.json();
		const names = (payload.snapshots ?? []).map((s) => s.snapshot);
		setStatus(
			names.length
				? `archived ${names.length} version run${names.length === 1 ? "" : "s"} → ${names.join(", ")} — load any of them from the run picker later`
				: "nothing to archive yet",
		);
		await refreshRuns();
	} catch (e) {
		setStatus(`archive failed: ${e.message}`, "err");
	} finally {
		versionArchiveAllEl.disabled = false;
		versionArchiveAllEl.textContent = prevLabel;
	}
}

versionArchiveAllEl.addEventListener("click", archiveAllVersions);

async function createRun() {
	const name = window.prompt(
		"New run name (e.g. iteration14, ab_test_v3):",
		"",
	);
	if (!name) return;
	runNewEl.disabled = true;
	try {
		const res = await fetch(new URL("/runs", SERVER_URL), {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ name: name.trim() }),
		});
		if (!res.ok) {
			const detail = await res.text();
			setStatus(`create run failed: ${detail}`, "err");
			return;
		}
		const payload = await res.json();
		currentRun = payload.current;
		setStatus(`run created :: ${currentRun}`);
		await refreshRuns();
		resetClientStateForRunSwitch();
		await refreshSlots();
		if (currentSlotId && currentModel) {
			switchView(currentSlotId, currentModel);
		}
	} finally {
		runNewEl.disabled = false;
	}
}

function switchView(slotId, modelAlias) {
	// Common path for both slot and model switches: tear down the current
	// SSE + scene, persist the new selection, and load the cell (scene +
	// log backfill) unless it's idle. Error/paused cells still load so the
	// log panel shows what happened; slotNeedsResume drives retry/resume UI.
	// Abandon any open tuning session first — the cell it rewound is going away.
	teardownSandboxSilently();
	if (currentSource) {
		currentSource.close();
		currentSource = null;
	}
	clearScene();
	clearLog();
	clearAssets();
	treeClear();
	clearMeshErrors();
	highestEventIndex = -1;
	recordedEvents.length = 0;
	updateReplayButton();
	currentSlotId = slotId;
	currentModel = modelAlias;
	try {
		localStorage.setItem(SLOT_STORAGE_KEY, slotId);
	} catch {}
	try {
		localStorage.setItem(MODEL_STORAGE_KEY, modelAlias);
	} catch {}
	if (modelPickerEl.value !== modelAlias) modelPickerEl.value = modelAlias;

	const status = currentRunInfo()?.status;
	slotNeedsResume =
		status === "idle" || status === "paused" || status === "error";
	renderSlotTabs();
	updateResumeButton();
	const cellLabel = `${slotId} · ${modelAlias}`;
	if (status === "idle") {
		setStatus(`slot :: ${cellLabel} — idle`);
	} else {
		if (slotNeedsResume) {
			setStatus(`slot :: ${cellLabel} — ${status}`);
		} else {
			setStatus(`slot :: ${cellLabel}`);
		}
		// Error/paused cells still have an events.jsonl — load scene + backfill
		// logs so switching to a failed run shows what went wrong.
		loadCellScene(slotId, modelAlias);
	}
	syncActiveTab();
}

function switchSlot(id) {
	if (id === currentSlotId) return;
	switchView(id, currentModel);
}

function switchModel(alias) {
	if (alias === currentModel) return;
	if (!availableModels.includes(alias)) return;
	switchView(currentSlotId, alias);
}

// --- run tab bar -------------------------------------------------------------

function nextTabId() {
	return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

function runExists(run) {
	return availableRuns.some((r) => r.name === run);
}

function persistTabs() {
	try {
		localStorage.setItem(
			TABS_STORAGE_KEY,
			JSON.stringify({ tabs: openTabs, active: activeTabId }),
		);
	} catch {}
}

// Pull the loaded run/slot/model back onto the active tab so the pickers,
// version bar, slot bar, and model picker all "edit" whichever tab is open.
// Called at the tail of switchView (the single choke point for run/slot/model
// changes), so every selection path keeps the tab + its label in sync.
function syncActiveTab() {
	const tab = openTabs.find((t) => t.id === activeTabId);
	if (!tab) return;
	tab.run = currentRun;
	tab.slot = currentSlotId;
	tab.model = currentModel;
	persistTabs();
	renderTabBar();
	// Keep the run picker + version highlight reflecting the active tab's run
	// immediately (otherwise they'd lag until the next refresh poll).
	if (currentRun && runPickerEl.value !== currentRun) {
		runPickerEl.value = currentRun;
	}
	renderVersionBar();
}

function renderTabBar() {
	if (!runTabsBarEl) return;
	for (const el of Array.from(runTabsBarEl.querySelectorAll(".run-tab"))) {
		el.remove();
	}
	for (const tab of openTabs) {
		const btn = document.createElement("button");
		btn.type = "button";
		btn.className = "run-tab" + (tab.id === activeTabId ? " active" : "");
		btn.title = `${tab.run} · ${tab.slot ?? "—"} · ${tab.model ?? "—"}`;

		const label = document.createElement("span");
		label.className = "rt-label";
		label.textContent = tab.run ?? "(run)";
		btn.appendChild(label);

		const subText = [tab.slot, tab.model].filter(Boolean).join(" · ");
		if (subText) {
			const sub = document.createElement("span");
			sub.className = "rt-sub";
			sub.textContent = subText;
			btn.appendChild(sub);
		}

		btn.addEventListener("click", () => activateTab(tab.id));

		// The last tab can't be closed — there's always one open view.
		if (openTabs.length > 1) {
			const close = document.createElement("span");
			close.className = "run-tab-close";
			close.textContent = "×";
			close.title = "Close this tab";
			close.addEventListener("click", (ev) => {
				ev.stopPropagation();
				closeTab(tab.id);
			});
			btn.appendChild(close);
		}

		runTabsBarEl.insertBefore(btn, runTabAddEl);
	}
}

// Make `id` the active tab and load its (run, slot, model) into the canvas.
// Reuses the existing switch machinery: a different run goes through switchRun
// (activate + teardown + reload); a same-run/different-cell tab swaps via
// switchView; an identical view (e.g. a freshly duplicated tab) is a no-op.
async function activateTab(id) {
	const tab = openTabs.find((t) => t.id === id);
	if (!tab) return;
	activeTabId = id;
	const wantSlot = tab.slot ?? currentSlotId ?? slotSummaries[0]?.id ?? null;
	const wantModel =
		tab.model ??
		currentModel ??
		defaultModelAlias ??
		availableModels[0] ??
		null;
	renderTabBar();

	if (
		tab.run === currentRun &&
		wantSlot === currentSlotId &&
		wantModel === currentModel
	) {
		syncActiveTab();
		return;
	}
	if (tab.run !== currentRun) {
		// Pre-seed the cell so switchRun's trailing switchView targets it.
		currentSlotId = wantSlot;
		currentModel = wantModel;
		await switchRun(tab.run);
	} else {
		switchView(wantSlot, wantModel);
	}
}

// Open a new tab cloning the active view; the user then retargets its run via
// the run picker / version bar to compare against the original.
function addTab() {
	const base = openTabs.find((t) => t.id === activeTabId);
	const tab = {
		id: nextTabId(),
		run: base?.run ?? currentRun,
		slot: base?.slot ?? currentSlotId,
		model: base?.model ?? currentModel,
	};
	openTabs.push(tab);
	activateTab(tab.id);
}

function closeTab(id) {
	if (openTabs.length <= 1) return;
	const idx = openTabs.findIndex((t) => t.id === id);
	if (idx === -1) return;
	const wasActive = id === activeTabId;
	openTabs.splice(idx, 1);
	if (wasActive) {
		const next = openTabs[Math.min(idx, openTabs.length - 1)];
		activateTab(next.id);
	} else {
		persistTabs();
		renderTabBar();
	}
}

// Restore persisted tabs on boot, dropping any whose run no longer exists.
// Returns null when there's nothing valid to restore (first run, or all runs
// deleted) so the caller seeds a fresh single tab.
function loadSavedTabs() {
	let raw = null;
	try {
		raw = localStorage.getItem(TABS_STORAGE_KEY);
	} catch {}
	if (!raw) return null;
	let parsed;
	try {
		parsed = JSON.parse(raw);
	} catch {
		return null;
	}
	// Drop tabs whose run is gone; repair a slot/model the server no longer
	// reports to null so the loader falls back to a valid default.
	const tabs = (parsed?.tabs ?? [])
		.filter((t) => t && typeof t.run === "string" && runExists(t.run))
		.map((t) => ({
			id: typeof t.id === "string" ? t.id : nextTabId(),
			run: t.run,
			slot: slotSummaries.some((s) => s.id === t.slot) ? t.slot : null,
			model: availableModels.includes(t.model) ? t.model : null,
		}));
	if (tabs.length === 0) return null;
	return { tabs, active: parsed.active };
}

runTabAddEl.addEventListener("click", addTab);

// Every cell request names its run/version explicitly so the three
// concurrently-running versions can't bleed into one another through a shared
// server-side selector. `run` defaults to the viewed run; the start picker
// passes other versions' run names to drive their cells in the background.
function slotEventsUrl(slotId, model, run = currentRun) {
	return new URL(
		`/slots/${encodeURIComponent(slotId)}/${encodeURIComponent(model)}/events?run=${encodeURIComponent(run)}`,
		SERVER_URL,
	).toString();
}

function slotMeshesUrl(slotId, model, run = currentRun) {
	// `mode` is the asset toggle — the only request it touches. "library" streams
	// objects/; "generated" streams generated/<version>/objects-generated-optimized
	// (the selected version, or the server's latest when `version` is omitted).
	// Everything else (scene, events, history) stays pinned to the library build.
	const u = new URL(
		`/slots/${encodeURIComponent(slotId)}/${encodeURIComponent(model)}/meshes?run=${encodeURIComponent(run)}&mode=${encodeURIComponent(assetMode)}`,
		SERVER_URL,
	);
	if (assetMode === "generated" && genVersion != null) {
		u.searchParams.set("version", genVersion);
	}
	// `optimized=0` re-points the bundle at the raw objects-generated/ folder; the
	// server defaults to the optimized twin, so only send it when raw is selected.
	if (assetMode === "generated" && !genOptimized) {
		u.searchParams.set("optimized", "0");
	}
	return u.toString();
}

function slotSceneUrl(slotId, model, run = currentRun) {
	return new URL(
		`/slots/${encodeURIComponent(slotId)}/${encodeURIComponent(model)}/scene?run=${encodeURIComponent(run)}`,
		SERVER_URL,
	).toString();
}

// The full event log on disk, served by the /artifacts static mount. Used to
// backfill the side panels (log / observability / gif) in the background after
// the scene is already painted from /scene — no SSE replay. Artifacts are
// genuinely path-scoped by run, so this already isolates per version.
function historyUrl(slotId, model, run = currentRun) {
	return new URL(
		`/artifacts/${encodeURIComponent(run)}/${encodeURIComponent(slotId)}/${encodeURIComponent(model)}/events.jsonl`,
		SERVER_URL,
	).toString();
}

async function resetSlot(id, model, skipConfirm = false) {
	if (!skipConfirm) {
		const ok = window.confirm(
			`Wipe runs/${id}/${model}/ and restart the pipeline for this cell?`,
		);
		if (!ok) return;
	}
	teardownSandboxSilently();
	resetEl.disabled = true;
	try {
		const res = await fetch(
			new URL(
				`/slots/${encodeURIComponent(id)}/${encodeURIComponent(model)}/reset?run=${encodeURIComponent(currentRun)}`,
				SERVER_URL,
			),
			{ method: "POST" },
		);
		if (!res.ok) {
			setStatus(`reset failed: HTTP ${res.status}`, "err");
			return;
		}
		if (currentSource) {
			currentSource.close();
			currentSource = null;
		}
		clearScene();
		clearLog();
		clearAssets();
		treeClear();
		clearMeshErrors();
		highestEventIndex = -1;
		recordedEvents.length = 0;
		updateReplayButton();
		slotNeedsResume = false;
		setStatus(`slot ${id} · ${model} reset — streaming events…`);
		loadCellScene(id, model, { forceLive: true });
		refreshSlots();
	} catch (e) {
		setStatus(`reset failed: ${e.message}`, "err");
	} finally {
		resetEl.disabled = false;
	}
}

function subscribe(url) {
	// Fresh scene loads (highestEventIndex === -1) replay the whole snapshot, so
	// prefetch the cell's GLBs in one request; loadModel consumes the bundle as
	// `model` events arrive. Mid-run re-subscribes (mesh retry, post-replay
	// reconnect) keep their already-loaded meshes — those events dedupe by index
	// — so they skip the prefetch.
	if (
		highestEventIndex === -1 &&
		currentSlotId !== null &&
		currentModel !== null
	) {
		prefetchMeshBundle(currentSlotId, currentModel, sceneGen);
	}
	const es = new EventSource(url);
	currentSource = es;
	es.onmessage = (ev) => {
		let data;
		try {
			data = JSON.parse(ev.data);
		} catch {
			return;
		}
		dispatch(data);
	};
	es.onerror = () => {
		// EventSource auto-reconnects on transient failures; only surface a hard close.
		if (es.readyState === EventSource.CLOSED && currentSource === es) {
			setStatus("stream closed", "err");
			currentSource = null;
		}
	};
}

// --- CQRS scene load (derive-on-read) --------------------------------------
//
// Opening an existing cell no longer replays the whole event log. Instead:
//   1. GET /scene → a folded projection of the current scene; paint it directly
//      (bboxes + tree + asset cards) via the same builders dispatch() uses.
//   2. Start the one-connection mesh bundle.
//   3. If the run is still active, tail the live SSE at ?since=last_index
//      (catch-up + new events only — no history).
//   4. Once the meshes have streamed, backfill the FULL event log into the side
//      panels (log / observability / gif) in the background, WITHOUT re-running
//      scene handlers (the projection already built the scene).
//
// subscribe() (full replay) stays for fresh/resumed runs (reset/rewind/retry)
// where history is empty or being freshly built.

// Build scene + tree + asset panel directly from the /scene projection. Reuses
// the live builders so there's a single source of truth for how a node renders.
function applySceneProjection(nodes) {
	for (const n of nodes) {
		const patch = {
			parentId: n.parent_id ?? null,
			prompt: n.prompt ?? null,
			kind: n.node_kind ?? "zone",
		};
		if (n.plan != null) patch.plan = n.plan;
		if (n.image_prompt != null) patch.imagePrompt = n.image_prompt;
		// Mirror the live `dispatch` "bbox" handler so renderTreeDetail's
		// origin/size/proxy rows render the same whether the cell was opened
		// from /scene or watched live over SSE.
		if (n.origin != null) patch.origin = n.origin;
		if (n.dimensions != null) patch.dimensions = n.dimensions;
		if (n.proxy_shape != null) patch.proxyShape = n.proxy_shape;
		treeUpsert(n.id, patch);
		if (n.origin && n.dimensions) {
			loadBbox({
				id: n.id,
				origin: n.origin,
				dimensions: n.dimensions,
				proxy_shape: n.proxy_shape,
				node_kind: n.node_kind,
				orientation: n.orientation,
			});
		}
		if (n.image_url) {
			upsertAsset(n.id, {
				imageUrl: n.image_url,
				prompt: n.image_prompt ?? n.prompt ?? null,
			});
		}
		if (n.mesh_url) upsertAsset(n.id, { modelUrl: n.mesh_url });
		if (n.error) {
			meshErrors.set(n.id, n.error);
			upsertAsset(n.id, { status: "error", errorMessage: n.error });
		}
		if (n.phase) {
			// Set phase directly (not treeSetPhase) to skip its per-call render/focus
			// churn across hundreds of nodes; the single scheduleRenderTree() below covers all.
			const cur = treeNodes.get(n.id);
			if (cur) cur.phase = n.phase;
		}
	}
	scheduleRenderTree();
}

// Panel-only consumer for the background history load: feeds the gif buffer,
// log panel, and observability modal — never the scene (the projection owns it).
function backfillPanels(event) {
	recordedEvents.push(event);
	appendEvent(event);
	if (event.kind === "cache.llm") recordLlmCall(event);
}

// Stream the full event log into the side panels AFTER the meshes have loaded,
// so the heavy parse never contends with GLB texture finalization. Chunked so
// it yields to the main thread instead of freezing it.
async function backfillHistoryInBackground(slotId, model, gen) {
	try {
		await meshBundle.streamDone;
	} catch {}
	if (gen !== sceneGen) return;
	const t0 = performance.now();
	let text;
	try {
		const res = await fetch(historyUrl(slotId, model), {
			cache: "no-store",
		});
		if (!res.ok) return;
		text = await res.text();
	} catch {
		return;
	}
	if (gen !== sceneGen) return;
	const tFetched = performance.now();
	const lines = text.split("\n");
	let i = 0;
	let count = 0;
	const step = () => {
		if (gen !== sceneGen) return;
		const end = Math.min(i + 200, lines.length);
		for (; i < end; i++) {
			const line = lines[i].trim();
			if (!line) continue;
			let event;
			try {
				event = JSON.parse(line);
			} catch {
				continue;
			}
			backfillPanels(event);
			count++;
		}
		if (i < lines.length) setTimeout(step, 0);
		else {
			updateReplayButton();
			if (LOAD_TIMING) {
				console.info(
					`[load] backfill: ${count} events · ${(text.length / 1e6).toFixed(1)}MB · ` +
						`fetch ${(tFetched - t0) | 0}ms · parse+render ${(performance.now() - tFetched) | 0}ms`,
				);
			}
		}
	};
	step();
}

async function loadCellScene(slotId, model, { forceLive = false } = {}) {
	const gen = sceneGen;
	const t0 = performance.now();
	let payload;
	try {
		const res = await fetch(slotSceneUrl(slotId, model), {
			cache: "no-store",
		});
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		payload = await res.json();
	} catch (e) {
		setStatus(`scene load failed: ${e.message}`, "err");
		return;
	}
	// Bail if the user switched cells while /scene was in flight.
	if (gen !== sceneGen || currentSlotId !== slotId || currentModel !== model)
		return;

	const tFetch = performance.now();
	applySceneProjection(payload.nodes ?? []);
	if (LOAD_TIMING) {
		console.info(
			`[load] /scene: ${payload.nodes?.length ?? 0} nodes · ` +
				`fetch ${(tFetch - t0) | 0}ms · projection ${(performance.now() - tFetch) | 0}ms`,
		);
	}
	highestEventIndex =
		typeof payload.last_index === "number" ? payload.last_index : -1;
	prefetchMeshBundle(slotId, model, sceneGen);
	// In generated mode, resolve this cell's versions + selected version now so the
	// picker populates immediately instead of after the next poll tick.
	if (assetMode === "generated") refreshGenerateGate();

	if (forceLive || currentRunInfo()?.status === "running") {
		// Active run (or just reset/resumed): tail only the events past the
		// projection cut. forceLive bypasses the status check, which is racy right
		// after a POST flips the run to running.
		subscribe(`${slotEventsUrl(slotId, model)}&since=${highestEventIndex}`);
	} else {
		// Finished cell: no live stream. Mark the run done so post-run mesh retries
		// refresh the status line correctly. Error/paused keep the status line
		// switchView already set — backfill will populate the log panel.
		runFinished = true;
		const status = currentRunInfo()?.status;
		if (status !== "error" && status !== "paused") {
			if (meshErrors.size > 0) showRunCompleteWithErrors();
			else setStatus("run complete");
		}
	}

	backfillHistoryInBackground(slotId, model, gen);
}

async function rewindTo(index) {
	// The prompt-tuning sandbox owns the scene + freezes the log; a real
	// (destructive) rewind here would truncate events.jsonl out from under it.
	if (sandboxActive) return;
	if (currentSlotId === null || currentModel === null) return;
	if (currentSource) {
		currentSource.close();
		currentSource = null;
	}
	clearScene();
	clearLog();
	clearAssets();
	treeClear();
	clearMeshErrors();
	highestEventIndex = -1;
	recordedEvents.length = 0;
	updateReplayButton();
	setStatus(
		`POST /slots/${currentSlotId}/${currentModel}/rewind to ${index} …`,
	);

	let res;
	try {
		res = await fetch(
			new URL(
				`/slots/${encodeURIComponent(currentSlotId)}/${encodeURIComponent(currentModel)}/rewind?run=${encodeURIComponent(currentRun)}`,
				SERVER_URL,
			),
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ to_event_index: index }),
			},
		);
	} catch (e) {
		setStatus(`rewind failed: ${e.message}`, "err");
		return;
	}
	if (!res.ok) {
		setStatus(`HTTP ${res.status}: ${await res.text()}`, "err");
		return;
	}
	slotNeedsResume = false;
	setStatus(`rewound to ${index} — streaming events…`);
	subscribe(slotEventsUrl(currentSlotId, currentModel));
	refreshSlots();
}

resetEl.addEventListener("click", () => {
	if (currentSlotId !== null && currentModel !== null) {
		resetSlot(currentSlotId, currentModel);
	}
});

async function resetAll() {
	// Wipe every started cell on the current run back to IDLE — fanned out
	// across all slots × all models — WITHOUT restarting anything. We only touch
	// cells that have data (events_count > 0); never-run cells are already idle.
	// Reset no longer implies "start": the user then opens "start cells…" and
	// launches exactly the (version, slot, model) cells they want, so one click
	// can't spawn dozens of pipelines.
	const run = currentRun;
	const cells = [];
	for (const s of slotSummaries) {
		for (const model of availableModels) {
			if ((s.runs?.[model]?.events_count ?? 0) > 0) {
				cells.push({ id: s.id, model });
			}
		}
	}
	if (cells.length === 0) {
		setStatus(`no started cells to reset on run "${run}"`);
		return;
	}
	const ok = window.confirm(
		`Wipe ${cells.length} started cell(s) across all models on run "${run}" back to idle?\n\nThis permanently deletes their generated meshes + event logs. Nothing is restarted — use "start cells…" to launch the ones you want.`,
	);
	if (!ok) return;
	resetAllEl.disabled = true;
	try {
		const viewedReset = cells.some(
			(c) => c.id === currentSlotId && c.model === currentModel,
		);
		const results = await Promise.all(
			cells.map((c) =>
				fetch(
					new URL(
						`/slots/${encodeURIComponent(c.id)}/${encodeURIComponent(c.model)}/reset?run=${encodeURIComponent(run)}&start=false`,
						SERVER_URL,
					),
					{ method: "POST" },
				)
					.then((r) => ({ cell: `${c.id}·${c.model}`, ok: r.ok }))
					.catch(() => ({ cell: `${c.id}·${c.model}`, ok: false })),
			),
		);
		if (viewedReset) {
			// The on-screen cell is now an empty idle cell — tear its scene down so
			// it doesn't keep showing stale meshes from the wiped run.
			if (currentSource) {
				currentSource.close();
				currentSource = null;
			}
			clearScene();
			clearLog();
			clearAssets();
			treeClear();
			clearMeshErrors();
			highestEventIndex = -1;
			recordedEvents.length = 0;
			updateReplayButton();
			slotNeedsResume = false;
		}
		const failures = results.filter((r) => r && r.ok === false);
		if (failures.length > 0) {
			const names = failures.map((f) => f.cell).join(", ");
			setStatus(
				`reset all on "${run}": ${cells.length - failures.length} ok, ${failures.length} failed (${names})`,
				"err",
			);
		} else {
			setStatus(
				`reset ${cells.length} cell${cells.length === 1 ? "" : "s"} on "${run}" to idle — pick cells to start`,
			);
		}
		await refreshSlots();
		refreshVersions();
	} catch (e) {
		setStatus(`reset all failed: ${e.message}`, "err");
	} finally {
		resetAllEl.disabled = false;
	}
}

resetAllEl.addEventListener("click", resetAll);

// --- start picker: launch exactly the chosen version × model × slot cells ---
//
// "reset all" wipes cells to idle without starting them; this is how you then
// start precisely the cells you want. The three checkbox groups are
// independent and the launch set is their cross product. Cells that are
// already running/done are skipped server-side (resume returns 400/409), so
// over-selecting is harmless.

function startCheckedValues(containerEl) {
	return Array.from(
		containerEl.querySelectorAll("input[type=checkbox]:checked"),
	).map((cb) => cb.value);
}

function updateStartCount() {
	const v = startCheckedValues(startVersionsEl).length;
	const m = startCheckedValues(startModelsEl).length;
	const s = startCheckedValues(startSlotsEl).length;
	const n = v * m * s;
	startModalCountEl.textContent = `${n} cell${n === 1 ? "" : "s"} — ${v} version${v === 1 ? "" : "s"} × ${m} model${m === 1 ? "" : "s"} × ${s} slot${s === 1 ? "" : "s"}`;
	startModalGoEl.disabled = n === 0;
	startModalGoEl.textContent =
		n === 0 ? "start selected" : `start ${n} cell${n === 1 ? "" : "s"}`;
}

function buildStartOptions(containerEl, items) {
	containerEl.innerHTML = "";
	for (const it of items) {
		const label = document.createElement("label");
		label.className = "start-opt";
		label.title = it.title ?? "";
		const cb = document.createElement("input");
		cb.type = "checkbox";
		cb.value = it.value;
		cb.checked = !!it.checked;
		cb.addEventListener("change", updateStartCount);
		const text = document.createElement("span");
		text.textContent = it.label;
		label.append(cb, text);
		containerEl.appendChild(label);
	}
}

function openStartModal() {
	if (
		availableVersions.length === 0 ||
		availableModels.length === 0 ||
		slotSummaries.length === 0
	) {
		setStatus(
			"nothing to start yet — versions/models/slots not loaded",
			"warn",
		);
		return;
	}
	// Default to the version + model you're currently viewing, with no slots
	// pre-checked so the count starts at 0 and you opt slots in deliberately.
	buildStartOptions(
		startVersionsEl,
		availableVersions.map((ver) => ({
			value: ver.run_name,
			label: ver.label ?? ver.run_name,
			checked: ver.run_name === currentRun,
		})),
	);
	buildStartOptions(
		startModelsEl,
		availableModels.map((m) => ({
			value: m,
			label: m,
			checked: m === currentModel,
		})),
	);
	buildStartOptions(
		startSlotsEl,
		slotSummaries.map((s) => ({
			value: s.id,
			label: s.id,
			title: s.prompt ?? "",
			checked: false,
		})),
	);
	updateStartCount();
	startModalEl.classList.add("open");
}

function closeStartModal() {
	startModalEl.classList.remove("open");
}

async function startSelectedCells() {
	const runs = startCheckedValues(startVersionsEl);
	const models = startCheckedValues(startModelsEl);
	const slots = startCheckedValues(startSlotsEl);
	const cells = [];
	for (const run of runs) {
		for (const slot of slots) {
			for (const model of models) cells.push({ run, slot, model });
		}
	}
	if (cells.length === 0) return;
	startModalGoEl.disabled = true;
	try {
		const results = await Promise.all(
			cells.map((c) =>
				fetch(
					new URL(
						`/slots/${encodeURIComponent(c.slot)}/${encodeURIComponent(c.model)}/resume?run=${encodeURIComponent(c.run)}`,
						SERVER_URL,
					),
					{ method: "POST" },
				)
					.then((r) => ({ ...c, ok: r.ok }))
					.catch(() => ({ ...c, ok: false })),
			),
		);
		const started = results.filter((r) => r.ok).length;
		const skipped = results.length - started;
		setStatus(
			`started ${started}/${results.length} cell${results.length === 1 ? "" : "s"}` +
				(skipped ? ` — ${skipped} skipped (already running/done)` : ""),
		);
		const viewedStarted = cells.some(
			(c) =>
				c.run === currentRun &&
				c.slot === currentSlotId &&
				c.model === currentModel,
		);
		closeStartModal();
		if (viewedStarted) {
			// The viewed cell just started — rewire its scene + SSE so it streams
			// live instead of waiting for the next manual reselect.
			if (currentSource) {
				currentSource.close();
				currentSource = null;
			}
			clearScene();
			clearLog();
			clearAssets();
			treeClear();
			clearMeshErrors();
			highestEventIndex = -1;
			recordedEvents.length = 0;
			updateReplayButton();
			slotNeedsResume = false;
			loadCellScene(currentSlotId, currentModel, { forceLive: true });
		}
		await refreshSlots();
		refreshVersions();
	} finally {
		startModalGoEl.disabled = false;
		updateStartCount();
	}
}

startCellsEl.addEventListener("click", openStartModal);
startModalGoEl.addEventListener("click", startSelectedCells);
startModalCloseEl.addEventListener("click", closeStartModal);
startModalEl.addEventListener("click", (ev) => {
	// Click on the dimmed backdrop (outside the panel) closes.
	if (ev.target === startModalEl) closeStartModal();
	// Select-all / -none chips in each column header.
	const bulk = ev.target.closest("[data-start-bulk]");
	if (bulk) {
		const container = document.getElementById(bulk.dataset.startTarget);
		if (container) {
			const check = bulk.dataset.startBulk === "all";
			for (const cb of container.querySelectorAll(
				"input[type=checkbox]",
			)) {
				cb.checked = check;
			}
			updateStartCount();
		}
	}
});
document.addEventListener("keydown", (ev) => {
	if (ev.key === "Escape" && startModalEl.classList.contains("open")) {
		closeStartModal();
	}
});

modelPickerEl.addEventListener("change", () => {
	switchModel(modelPickerEl.value);
});

runPickerEl.addEventListener("change", () => {
	switchRun(runPickerEl.value);
});

runNewEl.addEventListener("click", createRun);

async function saveAllAndNew() {
	const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
	const name = `save_${ts}`;
	saveAllEl.disabled = true;
	try {
		const res = await fetch(new URL("/runs", SERVER_URL), {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ name }),
		});
		if (!res.ok) {
			const detail = await res.text();
			setStatus(`save all failed: ${detail}`, "err");
			return;
		}
		const prevRun = currentRun;
		const payload = await res.json();
		currentRun = payload.current;
		setStatus(
			`saved run "${prevRun}" — switched to fresh run "${currentRun}"`,
		);
		await refreshRuns();
		resetClientStateForRunSwitch();
		await refreshSlots();
		if (currentSlotId && currentModel) {
			switchView(currentSlotId, currentModel);
		}
	} finally {
		saveAllEl.disabled = false;
	}
}

saveAllEl.addEventListener("click", saveAllAndNew);

async function snapshotAll() {
	snapshotAllEl.disabled = true;
	try {
		const res = await fetch(new URL("/runs/snapshot", SERVER_URL), {
			method: "POST",
		});
		if (!res.ok) {
			const detail = await res.text();
			setStatus(`snapshot failed: ${detail}`, "err");
			return;
		}
		const payload = await res.json();
		setStatus(`snapshot saved as "${payload.snapshot}"`);
		await refreshRuns();
	} finally {
		snapshotAllEl.disabled = false;
	}
}

snapshotAllEl.addEventListener("click", snapshotAll);

async function resumeAll() {
	// Fans out POST /slots/<slot>/<model>/resume for every PAUSED or ERRORED
	// cell on the active model: paused cells continue, errored cells retry.
	// Idle (never-started), running, and done cells are skipped — idle cells
	// are launched from "start cells…", not here. If the viewed cell gets
	// resumed, route it through resumeSlot() so the scene + SSE rewire —
	// non-viewed cells just need the kick, their events will flow next time
	// the user switches to them.
	if (currentModel === null) return;
	const model = currentModel;
	const resumable = slotSummaries.filter((s) =>
		["paused", "error"].includes(s.runs?.[model]?.status),
	);
	if (resumable.length === 0) {
		setStatus(`no paused or errored cells on ${model}`);
		return;
	}
	resumeAllEl.disabled = true;
	try {
		const tasks = resumable.map((s) => {
			if (s.id === currentSlotId) {
				// Viewed cell — wire SSE + clear scene via the existing helper.
				return resumeSlot(s.id, model);
			}
			return fetch(
				new URL(
					`/slots/${encodeURIComponent(s.id)}/${encodeURIComponent(model)}/resume?run=${encodeURIComponent(currentRun)}`,
					SERVER_URL,
				),
				{ method: "POST" },
			).then((r) => ({ slot: s.id, ok: r.ok, status: r.status }));
		});
		const results = await Promise.all(tasks);
		const failures = results.filter((r) => r && r.ok === false);
		if (failures.length > 0) {
			const names = failures.map((f) => f.slot).join(", ");
			setStatus(
				`resume all on ${model}: ${resumable.length - failures.length} ok, ${failures.length} failed (${names})`,
				"err",
			);
		} else {
			setStatus(
				`resumed ${resumable.length} cell${resumable.length === 1 ? "" : "s"} on ${model}`,
			);
		}
		refreshSlots();
	} catch (e) {
		setStatus(`resume all failed: ${e.message}`, "err");
	} finally {
		resumeAllEl.disabled = false;
	}
}

resumeAllEl.addEventListener("click", resumeAll);

async function stopAll() {
	// Preemptive global kill switch: POST /generations/stop halts EVERY in-flight
	// generation process-wide (all runs, all versions, all cells) — pipeline
	// builds, from-scratch generates, and mesh retries — and leaves them
	// resumable/retryable (running cells become paused). Nothing on disk is
	// touched, so this is a safe interrupt, not a reset. The viewed cell's open
	// SSE stream delivers its own run.paused; refreshSlots/refreshVersions repaint
	// the rest of the dashboard's status dots.
	const ok = window.confirm(
		"Stop ALL in-flight generations across every run and version?\n\nRunning pipeline cells become paused and from-scratch builds halt. Nothing is deleted — resume a cell or re-press generate/retry to pick up where it left off.",
	);
	if (!ok) return;
	stopAllEl.disabled = true;
	try {
		const res = await fetch(new URL("/generations/stop", SERVER_URL), {
			method: "POST",
		});
		if (!res.ok) {
			setStatus(`stop all failed: HTTP ${res.status}`, "err");
			return;
		}
		const payload = await res.json();
		const np = (payload.stopped_pipelines ?? []).length;
		const ng = (payload.stopped_generates ?? []).length;
		const nr = payload.stopped_retries ?? 0;
		if (np + ng + nr === 0) {
			setStatus("nothing in flight to stop");
		} else {
			setStatus(
				`stopped ${np} pipeline${np === 1 ? "" : "s"}, ${ng} generate${ng === 1 ? "" : "s"}, ${nr} retr${nr === 1 ? "y" : "ies"} — all resumable`,
			);
		}
		refreshSlots();
		refreshVersions();
	} catch (e) {
		setStatus(`stop all failed: ${e.message}`, "err");
	} finally {
		stopAllEl.disabled = false;
	}
}

stopAllEl.addEventListener("click", stopAll);

async function resumeSlot(id, model) {
	resumeEl.disabled = true;
	try {
		const res = await fetch(
			new URL(
				`/slots/${encodeURIComponent(id)}/${encodeURIComponent(model)}/resume?run=${encodeURIComponent(currentRun)}`,
				SERVER_URL,
			),
			{ method: "POST" },
		);
		if (!res.ok) {
			setStatus(`resume failed: HTTP ${res.status}`, "err");
			return;
		}
		slotNeedsResume = false;
		if (currentSource) {
			currentSource.close();
			currentSource = null;
		}
		clearScene();
		clearLog();
		clearAssets();
		treeClear();
		clearMeshErrors();
		highestEventIndex = -1;
		recordedEvents.length = 0;
		updateReplayButton();
		setStatus(`resumed — streaming events…`);
		subscribe(slotEventsUrl(id, model));
		refreshSlots();
	} catch (e) {
		setStatus(`resume failed: ${e.message}`, "err");
	} finally {
		resumeEl.disabled = false;
	}
}

async function pauseSlot(id, model) {
	resumeEl.disabled = true;
	try {
		const res = await fetch(
			new URL(
				`/slots/${encodeURIComponent(id)}/${encodeURIComponent(model)}/pause?run=${encodeURIComponent(currentRun)}`,
				SERVER_URL,
			),
			{ method: "POST" },
		);
		if (!res.ok) {
			setStatus(`pause failed: HTTP ${res.status}`, "err");
			return;
		}
		setStatus("paused");
		refreshSlots();
	} catch (e) {
		setStatus(`pause failed: ${e.message}`, "err");
	} finally {
		resumeEl.disabled = false;
	}
}

resumeEl.addEventListener("click", () => {
	if (currentSlotId === null || currentModel === null) return;
	const status = currentRunInfo()?.status;
	if (status === "running") {
		pauseSlot(currentSlotId, currentModel);
	} else {
		resumeSlot(currentSlotId, currentModel);
	}
});

document.getElementById("zoom-in").addEventListener("click", () => _dolly(0.8));
document
	.getElementById("zoom-out")
	.addEventListener("click", () => _dolly(1.25));

// --- GLB export --------------------------------------------------------------

exportGlbEl.addEventListener("click", async () => {
	if (modelsById.size === 0) return;
	exportGlbEl.disabled = true;
	exportGlbEl.textContent = "exporting…";
	try {
		// Temporarily force all models visible so hidden frames are included.
		const wasHidden = [];
		sceneRoot.traverse((obj) => {
			if (!obj.visible) {
				wasHidden.push(obj);
				obj.visible = true;
			}
		});

		const exporter = new GLTFExporter();
		const glb = await exporter.parseAsync(sceneRoot, { binary: true });

		for (const obj of wasHidden) obj.visible = false;

		const blob = new Blob([glb], { type: "model/gltf-binary" });
		const url = URL.createObjectURL(blob);
		const a = document.createElement("a");
		a.href = url;
		const stem =
			currentSlotId && currentModel
				? `${currentSlotId}_${currentModel}`
				: (currentSlotId ?? "scene");
		a.download = `${stem}.glb`;
		a.click();
		URL.revokeObjectURL(url);
	} catch (e) {
		appendEvent({
			kind: "run.error",
			message: `GLB export failed: ${e.message}`,
		});
	} finally {
		exportGlbEl.disabled = false;
		exportGlbEl.textContent = "export .glb";
	}
});

// --- 360° panorama capture ----------------------------------------------------
//
// Captures an equirectangular 360° snapshot from the current camera position:
// the live scene is rendered six times (cube faces, 90° fov) into a square
// scissored viewport on the main canvas — full parity with what's on screen
// (ACES tonemapping, sRGB, shadows, IBL, antialiasing; preserveDrawingBuffer is
// already on for the gif exporter, so the canvas is readable after render) —
// then CPU-stitched into a standard equirect panorama. The stitch uses the same
// direction→uv convention as three's EquirectangularReflectionMapping
// (u = atan2(z, x)/2π + 0.5, v = asin(y)/π + 0.5), so the JPEGs drop into any
// three.js scene as a background. Captures accumulate into a "tour" (world
// position + view direction per pano); "save tour" downloads tour.json + one
// JPEG per pano for the studio's /pano matterport-style walkthrough page.

const panoCaptureEl = document.getElementById("pano-capture");
const panoSaveEl = document.getElementById("pano-save");
const panoClearEl = document.getElementById("pano-clear");
const panoAutoEl = document.getElementById("pano-auto");

const panoTour = []; // { id, position: [x,y,z], forward: [x,y,z], blob }
let panoBusy = false;

const PANO_FACE_TARGET = 1280; // device px per cube face, capped by canvas size
const PANO_WIDTH_CAP = 4096; // equirect output width cap (height = width / 2)

// forward/up per face; right = cross(forward, up) — matches what lookAt builds,
// so the analytic projection in the stitch agrees with the render exactly.
// Order: +X, -X, +Y, -Y, +Z, -Z (indexed as axis*2 + (negative ? 1 : 0)).
const PANO_FACES = [
	{ f: [1, 0, 0], up: [0, 1, 0] },
	{ f: [-1, 0, 0], up: [0, 1, 0] },
	{ f: [0, 1, 0], up: [0, 0, 1] },
	{ f: [0, -1, 0], up: [0, 0, -1] },
	{ f: [0, 0, 1], up: [0, 1, 0] },
	{ f: [0, 0, -1], up: [0, 1, 0] },
];
const PANO_FACE_BASIS = PANO_FACES.map(({ f, up }) => ({
	f,
	up,
	right: [
		f[1] * up[2] - f[2] * up[1],
		f[2] * up[0] - f[0] * up[2],
		f[0] * up[1] - f[1] * up[0],
	],
}));

function updatePanoButtons() {
	panoCaptureEl.disabled = panoBusy;
	panoSaveEl.disabled = panoTour.length === 0 || panoBusy;
	panoClearEl.disabled = panoTour.length === 0 || panoBusy;
	if (panoAutoEl) panoAutoEl.disabled = panoBusy;
	panoSaveEl.textContent = `save tour (${panoTour.length})`;
}

// Render the six cube faces through the main renderer and read each back as
// ImageData. Synchronous; the animate loop repaints the viewport next frame.
function renderPanoFaces() {
	const canvas = renderer.domElement;
	const dpr = renderer.getPixelRatio();
	const prevSize = renderer.getSize(new THREE.Vector2());
	const faceSize = Math.min(PANO_FACE_TARGET, canvas.width, canvas.height);
	const faceCss = faceSize / dpr; // setViewport/Scissor multiply by dpr

	const faceCam = new THREE.PerspectiveCamera(90, 1, camera.near, camera.far);
	faceCam.position.copy(camera.position);

	const crop = document.createElement("canvas");
	crop.width = faceSize;
	crop.height = faceSize;
	const cropCtx = crop.getContext("2d", { willReadFrequently: true });

	// Debug wireframes don't belong in a "realistic" pano.
	const bboxWasVisible = bboxRoot.visible;
	bboxRoot.visible = false;

	const faces = [];
	try {
		renderer.setScissorTest(true);
		for (const { f, up } of PANO_FACES) {
			faceCam.up.set(up[0], up[1], up[2]);
			faceCam.lookAt(
				camera.position.x + f[0],
				camera.position.y + f[1],
				camera.position.z + f[2],
			);
			renderer.setViewport(0, 0, faceCss, faceCss);
			renderer.setScissor(0, 0, faceCss, faceCss);
			renderer.render(scene, faceCam);
			// Viewport (0,0) is the canvas' bottom-left; drawImage's source rect
			// is top-left-origin device pixels.
			cropCtx.drawImage(
				canvas,
				0,
				canvas.height - faceSize,
				faceSize,
				faceSize,
				0,
				0,
				faceSize,
				faceSize,
			);
			faces.push(cropCtx.getImageData(0, 0, faceSize, faceSize));
		}
	} finally {
		renderer.setScissorTest(false);
		renderer.setViewport(0, 0, prevSize.x, prevSize.y);
		renderer.setScissor(0, 0, prevSize.x, prevSize.y);
		bboxRoot.visible = bboxWasVisible;
	}
	return { faces, faceSize };
}

// Stitch six face ImageDatas into one equirect ImageData (bilinear sampling).
// Chunked by rows so the tab stays responsive on 4096×2048 outputs.
async function stitchPanoEquirect(faces, faceSize, onProgress) {
	const W = Math.min(PANO_WIDTH_CAP, faceSize * 4);
	const H = W / 2;
	const out = new ImageData(W, H);
	const o = out.data;
	const S = faceSize;
	const maxIdx = S - 1;

	for (let row = 0; row < H; row++) {
		const v = 1 - (row + 0.5) / H;
		const phi = (v - 0.5) * Math.PI;
		const dy = Math.sin(phi);
		const cosPhi = Math.cos(phi);
		let oi = row * W * 4;
		for (let col = 0; col < W; col++, oi += 4) {
			const az = ((col + 0.5) / W - 0.5) * 2 * Math.PI;
			const dx = cosPhi * Math.cos(az);
			const dz = cosPhi * Math.sin(az);

			const ax = Math.abs(dx);
			const ay = Math.abs(dy);
			const az2 = Math.abs(dz);
			let faceIdx;
			if (ax >= ay && ax >= az2) faceIdx = dx > 0 ? 0 : 1;
			else if (ay >= az2) faceIdx = dy > 0 ? 2 : 3;
			else faceIdx = dz > 0 ? 4 : 5;

			const { f, up, right } = PANO_FACE_BASIS[faceIdx];
			const t = dx * f[0] + dy * f[1] + dz * f[2];
			const u2 = (dx * right[0] + dy * right[1] + dz * right[2]) / t;
			const v2 = (dx * up[0] + dy * up[1] + dz * up[2]) / t;

			// Face pixel coords (image y down) + bilinear weights.
			const px = (u2 * 0.5 + 0.5) * S - 0.5;
			const py = (0.5 - v2 * 0.5) * S - 0.5;
			let x0 = Math.floor(px);
			let y0 = Math.floor(py);
			const fx = px - x0;
			const fy = py - y0;
			x0 = x0 < 0 ? 0 : x0 > maxIdx ? maxIdx : x0;
			y0 = y0 < 0 ? 0 : y0 > maxIdx ? maxIdx : y0;
			const x1 = x0 < maxIdx ? x0 + 1 : maxIdx;
			const y1 = y0 < maxIdx ? y0 + 1 : maxIdx;

			const d = faces[faceIdx].data;
			const i00 = (y0 * S + x0) * 4;
			const i10 = (y0 * S + x1) * 4;
			const i01 = (y1 * S + x0) * 4;
			const i11 = (y1 * S + x1) * 4;
			const w00 = (1 - fx) * (1 - fy);
			const w10 = fx * (1 - fy);
			const w01 = (1 - fx) * fy;
			const w11 = fx * fy;

			o[oi] = d[i00] * w00 + d[i10] * w10 + d[i01] * w01 + d[i11] * w11;
			o[oi + 1] =
				d[i00 + 1] * w00 +
				d[i10 + 1] * w10 +
				d[i01 + 1] * w01 +
				d[i11 + 1] * w11;
			o[oi + 2] =
				d[i00 + 2] * w00 +
				d[i10 + 2] * w10 +
				d[i01 + 2] * w01 +
				d[i11 + 2] * w11;
			o[oi + 3] = 255;
		}
		if (row % 128 === 127) {
			onProgress?.(row / H);
			await sleep(0);
		}
	}

	const outCanvas = document.createElement("canvas");
	outCanvas.width = W;
	outCanvas.height = H;
	outCanvas.getContext("2d").putImageData(out, 0, 0);
	return new Promise((resolve, reject) =>
		outCanvas.toBlob(
			(blob) =>
				blob ? resolve(blob) : reject(new Error("JPEG encode failed")),
			"image/jpeg",
			0.92,
		),
	);
}

function downloadPanoBlob(blob, filename) {
	const url = URL.createObjectURL(blob);
	const a = document.createElement("a");
	a.href = url;
	a.download = filename;
	a.click();
	URL.revokeObjectURL(url);
}

// Bake one placed mesh's geometry into world-space, float32, position-only
// geometry. Reading every vertex through `fromBufferAttribute` DENORMALIZES
// quantized attributes, and writing into a FRESH Float32 array sidesteps the
// classic trap that broke the proxy: the placed library GLBs are Meshopt /
// KHR_mesh_quantization, so three keeps their POSITION as an INTEGER buffer with
// the dequantization folded into the node matrix. Calling
// `geometry.clone().applyMatrix4(matrixWorld)` writes world-space floats back
// into that integer buffer — truncating every vertex onto the integer grid (and,
// for normalized attributes, collapsing the whole scene toward the origin),
// which is exactly the snapped/sharded ~10%-size blob the proxy was showing.
function bakeWorldGeometry(mesh) {
	const src = mesh.geometry.getAttribute("position");
	if (!src) return null;
	const count = src.count;
	const positions = new Float32Array(count * 3);
	const v = new THREE.Vector3();
	const m = mesh.matrixWorld;
	for (let i = 0; i < count; i++) {
		v.fromBufferAttribute(src, i).applyMatrix4(m);
		positions[i * 3] = v.x;
		positions[i * 3 + 1] = v.y;
		positions[i * 3 + 2] = v.z;
	}
	const g = new THREE.BufferGeometry();
	g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
	// Keep the topology (a copy, detached from the source buffer). Normals are
	// dropped: the server strips them and /pano recomputes them on the proxy.
	const idx = mesh.geometry.getIndex();
	if (idx) g.setIndex(new THREE.BufferAttribute(idx.array.slice(), 1));
	return g;
}

// The owning object's id for a mesh: walk up to the node the loader tagged with
// `pickId` (set on every placed `gltf.scene`). Carries object identity into the
// baked proxy so the walkthrough can name / address individual objects.
function pickIdOf(obj) {
	for (let cur = obj; cur; cur = cur.parent) {
		if (cur.userData?.pickId) return cur.userData.pickId;
	}
	return null;
}

// Bake the live scene into one material-free, world-space GLB (geometry only:
// each placed mesh baked into world space — the same frame the pano positions
// were captured in). This is the merged stand-in the server's /proxy decimator
// reduces to a few-thousand-triangle projection proxy. Returns the binary GLB
// ArrayBuffer, or null when the scene has no meshes.
//
// Each source object's baked meshes hang under their own node, named with the
// object id, so the proxy keeps per-object identity through the decimation pass
// (gltf-transform preserves node names) and the viewer can address objects.
async function buildMergedSceneGlbBuffer() {
	if (modelsById.size === 0) return null;
	const root = new THREE.Group();
	const mat = new THREE.MeshStandardMaterial();
	const geoms = [];
	const objNodes = new Map(); // object id (or null) -> its node under `root`
	sceneRoot.updateWorldMatrix(true, true);
	sceneRoot.traverse((o) => {
		if (!o.isMesh || !o.geometry) return;
		const g = bakeWorldGeometry(o);
		if (!g) return;
		const id = pickIdOf(o);
		const key = id ?? "";
		let node = objNodes.get(key);
		if (!node) {
			node = new THREE.Group();
			if (id) node.name = id;
			objNodes.set(key, node);
			root.add(node);
		}
		node.add(new THREE.Mesh(g, mat));
		geoms.push(g);
	});
	if (geoms.length === 0) return null;
	try {
		const exporter = new GLTFExporter();
		return await exporter.parseAsync(root, {
			binary: true,
			onlyVisible: false,
		});
	} finally {
		for (const g of geoms) g.dispose();
		mat.dispose();
	}
}

// Build the merged scene and hand it to /proxy, returning the decimated proxy
// blob for the downloadable tour bundle (manual "save tour" flow).
async function buildProxyGlbBlob() {
	const glb = await buildMergedSceneGlbBuffer();
	if (!glb) return null;
	const res = await fetch(new URL("/proxy", SERVER_URL).toString(), {
		method: "POST",
		headers: { "Content-Type": "model/gltf-binary" },
		body: glb,
	});
	if (!res.ok) throw new Error(`server /proxy → ${res.status}`);
	return new Blob([await res.arrayBuffer()], { type: "model/gltf-binary" });
}

// Render + stitch one 360° equirectangular pano from the CURRENT camera position
// (the capture is orientation-independent — renderPanoFaces builds its own
// axis-aligned face cameras). Returns the JPEG blob.
async function capturePanoBlob(onProgress) {
	const { faces, faceSize } = renderPanoFaces();
	return stitchPanoEquirect(faces, faceSize, onProgress);
}

// --- bird's-eye minimap slices (one per Y level) ------------------------------
//
// The companion to the 360 captures: group the anchors into Y "levels" (storeys
// — anchors within MINIMAP_LEVEL_EPS metres of each other) and render one
// top-down orthographic slice of the scene per level. The slice is a horizontal
// SLAB at camera level: cut above the head (drops the roof, so we see in) AND a
// bit below the lowest camera in the group (drops the floor-and-below, so an
// upper storey's slice can't show the floor beneath it). The prod client shows
// the matching slice as a minimap and dots the level's anchors onto it, mapping
// each anchor's world XZ through the stored `bounds`.

const MINIMAP_LEVEL_EPS = 1.5; // metres; anchors within this Y gap share a level
const MINIMAP_RES = 1024; // longest output side in device px (capped by the canvas)
const MINIMAP_PAD_FRAC = 0.04; // breathing room around the scene footprint
const MINIMAP_SLICE_BELOW = 2; // metres below the level's lowest anchor for the floor cut

// Cluster anchor Ys into levels by gap; returns [{ y, minY, indices }] low→high,
// where y is the level's median camera height (its top slice-cut + client match
// key) and minY is its lowest camera (the bottom slice-cut rides just under it).
function groupAnchorLevels(positions) {
	const order = positions
		.map((_, i) => i)
		.sort((a, b) => positions[a][1] - positions[b][1]);
	const groups = [];
	let cur = null;
	for (const i of order) {
		const y = positions[i][1];
		if (!cur || y - cur.lastY > MINIMAP_LEVEL_EPS) {
			cur = { indices: [], ys: [], lastY: y };
			groups.push(cur);
		}
		cur.indices.push(i);
		cur.ys.push(y);
		cur.lastY = y;
	}
	return groups.map((g) => {
		const ys = g.ys.slice().sort((a, b) => a - b);
		return { y: ys[(ys.length - 1) >> 1], minY: ys[0], indices: g.indices };
	});
}

// Render one top-down slice of the live scene into a PNG blob. Reuses the main
// renderer/canvas (so tonemapping + sRGB match the panos), scissored to a
// footprint-aspect viewport then read back — the pano-face pattern. The ortho
// camera looks straight down with -Z "up" in the image, so the stored `bounds`
// map world (x,z) → image (left,top) as ((x-minX)/W, (z-minZ)/D).
async function captureMinimapBlob(bounds, cutTop, cutBottom, yTop, yBot) {
	const canvas = renderer.domElement;
	const dpr = renderer.getPixelRatio();
	const prevSize = renderer.getSize(new THREE.Vector2());

	const W = bounds.maxX - bounds.minX;
	const D = bounds.maxZ - bounds.minZ;
	const cx = (bounds.minX + bounds.maxX) / 2;
	const cz = (bounds.minZ + bounds.maxZ) / 2;

	// Output pixels preserve the footprint aspect, capped by MINIMAP_RES and the
	// drawing buffer (we read back from the canvas, so we can't exceed it).
	const cap = Math.min(MINIMAP_RES, canvas.width, canvas.height);
	let pw;
	let ph;
	if (W >= D) {
		pw = cap;
		ph = Math.max(1, Math.round((cap * D) / W));
	} else {
		ph = cap;
		pw = Math.max(1, Math.round((cap * W) / D));
	}

	const cam = new THREE.OrthographicCamera(
		-W / 2,
		W / 2,
		D / 2,
		-D / 2,
		0.1,
		yTop - yBot + 4,
	);
	cam.position.set(cx, yTop + 2, cz);
	cam.up.set(0, 0, -1);
	cam.lookAt(cx, yBot, cz);
	cam.updateProjectionMatrix();

	// World clip planes bounding a horizontal SLAB: keep cutBottom <= y <= cutTop.
	// The top cut opens the roof (drops everything above the head); the bottom cut
	// drops the floor-and-below — including lower storeys, so an upper level's slice
	// can't show the floor beneath it. Global clipping planes intersect (a fragment
	// outside EITHER is dropped).
	const planeTop = new THREE.Plane(new THREE.Vector3(0, -1, 0), cutTop);
	const planeBottom = new THREE.Plane(new THREE.Vector3(0, 1, 0), -cutBottom);
	const prevClip = renderer.clippingPlanes;
	const prevBg = scene.background;
	const prevClear = renderer.getClearColor(new THREE.Color());
	const prevAlpha = renderer.getClearAlpha();
	const prevShadow = renderer.shadowMap.enabled;
	const bboxWasVisible = bboxRoot.visible;

	const crop = document.createElement("canvas");
	crop.width = pw;
	crop.height = ph;

	try {
		bboxRoot.visible = false; // debug wireframes don't belong on the map
		// Flat, evenly-lit floor plan reads clearer than a top-down cast-shadow
		// render; the clipping-plane swap below forces a program refresh, so this
		// toggle takes effect for the slice pass.
		renderer.shadowMap.enabled = false;
		renderer.clippingPlanes = [planeTop, planeBottom];
		scene.background = null;
		renderer.setClearColor(0x0c0d10, 1);
		renderer.setScissorTest(true);
		renderer.setViewport(0, 0, pw / dpr, ph / dpr);
		renderer.setScissor(0, 0, pw / dpr, ph / dpr);
		renderer.render(scene, cam);
		// Viewport (0,0) is the canvas' bottom-left; drawImage's source rect is
		// top-left-origin device pixels.
		crop.getContext("2d").drawImage(
			canvas,
			0,
			canvas.height - ph,
			pw,
			ph,
			0,
			0,
			pw,
			ph,
		);
	} finally {
		renderer.setScissorTest(false);
		renderer.setViewport(0, 0, prevSize.x, prevSize.y);
		renderer.setScissor(0, 0, prevSize.x, prevSize.y);
		renderer.clippingPlanes = prevClip;
		renderer.shadowMap.enabled = prevShadow;
		scene.background = prevBg;
		renderer.setClearColor(prevClear, prevAlpha);
		bboxRoot.visible = bboxWasVisible;
	}
	return new Promise((resolve, reject) =>
		crop.toBlob(
			(blob) =>
				blob
					? resolve(blob)
					: reject(new Error("minimap encode failed")),
			"image/png",
		),
	);
}

async function uploadMinimap(cell, minimapId, blob) {
	const res = await fetch(
		new URL(
			`${cell.base}/tour/minimap/${encodeURIComponent(minimapId)}?${cell.run}`,
			SERVER_URL,
		).toString(),
		{
			method: "PUT",
			headers: { "Content-Type": "image/png" },
			body: blob,
		},
	);
	if (!res.ok) throw new Error(`upload ${minimapId} → ${res.status}`);
}

// Group the captured anchors by level, render + persist one bird's-eye slice
// per level, and return the manifest `minimaps` array (empty on any failure —
// the tour stays valid without them).
async function buildMinimaps(cell, panoMeta, onLevel) {
	const positions = panoMeta.map((p) => p.position);
	if (positions.length === 0) return [];
	const box = new THREE.Box3().setFromObject(sceneRoot);
	if (box.isEmpty()) return [];
	const pad =
		MINIMAP_PAD_FRAC *
		Math.max(box.max.x - box.min.x, box.max.z - box.min.z, 1);
	const bounds = {
		minX: box.min.x - pad,
		maxX: box.max.x + pad,
		minZ: box.min.z - pad,
		maxZ: box.max.z + pad,
	};
	const levels = groupAnchorLevels(positions);
	const minimaps = [];
	for (let li = 0; li < levels.length; li++) {
		onLevel?.(li, levels.length);
		const file = `minimap-${li}.png`;
		// Top cut above the head (median camera height); bottom cut a bit below the
		// level's lowest camera — a slab at camera level, isolated from other storeys.
		const blob = await captureMinimapBlob(
			bounds,
			levels[li].y,
			levels[li].minY - MINIMAP_SLICE_BELOW,
			box.max.y,
			box.min.y,
		);
		await uploadMinimap(cell, `minimap-${li}`, blob);
		minimaps.push({ level: li, y: levels[li].y, file, bounds });
	}
	return minimaps;
}

panoCaptureEl.addEventListener("click", async () => {
	if (panoBusy) return;
	panoBusy = true;
	panoCaptureEl.disabled = true;
	updatePanoButtons();
	try {
		panoCaptureEl.textContent = "rendering…";
		const blob = await capturePanoBlob((frac) => {
			panoCaptureEl.textContent = `stitching ${Math.round(frac * 100)}%`;
		});
		const fwd = camera.getWorldDirection(new THREE.Vector3());
		panoTour.push({
			id: `pano-${String(panoTour.length).padStart(3, "0")}`,
			position: camera.position.toArray(),
			forward: fwd.toArray(),
			blob,
		});
	} catch (e) {
		appendEvent({
			kind: "run.error",
			message: `360 capture failed: ${e.message}`,
		});
	} finally {
		panoCaptureEl.textContent = "📷 capture 360";
		panoCaptureEl.disabled = false;
		panoBusy = false;
		updatePanoButtons();
	}
});

panoSaveEl.addEventListener("click", async () => {
	if (panoTour.length === 0 || panoBusy) return;
	panoBusy = true;
	updatePanoButtons();
	// Decimate the live scene into a low-poly projection proxy first; the
	// manifest only advertises proxy.glb if it actually built, so /pano falls
	// back to its sphere mode when the proxy is missing.
	let proxyBlob = null;
	try {
		panoSaveEl.textContent = "building proxy…";
		proxyBlob = await buildProxyGlbBlob();
	} catch (e) {
		appendEvent({
			kind: "run.error",
			message: `proxy build failed (saving panos without it): ${e.message}`,
		});
	}
	const manifest = {
		version: 1,
		proxy: proxyBlob ? "proxy.glb" : null,
		panos: panoTour.map((p) => ({
			id: p.id,
			file: `${p.id}.jpg`,
			position: p.position,
			forward: p.forward,
		})),
	};
	try {
		panoSaveEl.textContent = "saving…";
		downloadPanoBlob(
			new Blob([JSON.stringify(manifest, null, 2)], {
				type: "application/json",
			}),
			"tour.json",
		);
		// Space the downloads out so the browser doesn't coalesce/drop them.
		if (proxyBlob) {
			await sleep(250);
			downloadPanoBlob(proxyBlob, "proxy.glb");
		}
		for (const p of panoTour) {
			await sleep(250);
			downloadPanoBlob(p.blob, `${p.id}.jpg`);
		}
	} finally {
		panoBusy = false;
		updatePanoButtons();
	}
});

panoClearEl.addEventListener("click", () => {
	if (panoBusy) return;
	panoTour.length = 0;
	updatePanoButtons();
});

// --- auto-tour: LLM-planned anchors → auto-capture → server-persisted tour ----
//
// The "other side of the coin". The server reads THIS cell's scene hierarchy,
// has a lightweight model propose capture anchor points, and returns them. We
// then drive the existing capture machinery over those anchors (set the camera,
// render a 360, upload it), decimate + upload the proxy, and write the manifest —
// so the whole tour persists under /artifacts/<cell>/tour/ for /pano to load by
// URL. The manual capture/save flow is untouched.

function cellQuery() {
	if (!currentRun || !currentSlotId || !currentModel) return null;
	return {
		base: `/slots/${encodeURIComponent(currentSlotId)}/${encodeURIComponent(currentModel)}`,
		run: `run=${encodeURIComponent(currentRun)}`,
	};
}

async function uploadPano(cell, panoId, blob) {
	const res = await fetch(
		new URL(
			`${cell.base}/tour/pano/${encodeURIComponent(panoId)}?${cell.run}`,
			SERVER_URL,
		).toString(),
		{
			method: "PUT",
			headers: { "Content-Type": "image/jpeg" },
			body: blob,
		},
	);
	if (!res.ok) throw new Error(`upload ${panoId} → ${res.status}`);
}

panoAutoEl?.addEventListener("click", async () => {
	if (panoBusy) return;
	const cell = cellQuery();
	if (!cell) {
		appendEvent({
			kind: "run.error",
			message: "auto-tour: no active run/slot/model",
		});
		return;
	}
	if (modelsById.size === 0) {
		appendEvent({
			kind: "run.error",
			message: "auto-tour: scene has no meshes loaded yet",
		});
		return;
	}
	panoBusy = true;
	updatePanoButtons();
	// Lock the camera so the animate loop's OrbitControls.update() doesn't fight
	// the positions we set per anchor; restore the view afterwards.
	const camSnapshot = {
		pos: camera.position.clone(),
		target: controls.target.clone(),
		userMoved: cameraUserMoved,
	};
	cameraUserMoved = true;
	try {
		panoAutoEl.textContent = "planning anchors…";
		const planRes = await fetch(
			new URL(`${cell.base}/anchors?${cell.run}`, SERVER_URL).toString(),
			{ method: "POST" },
		);
		if (!planRes.ok) throw new Error(`/anchors → ${planRes.status}`);
		const plan = await planRes.json();
		const anchors = Array.isArray(plan.anchors) ? plan.anchors : [];
		if (anchors.length === 0)
			throw new Error("planner returned no anchors");

		await fetch(
			new URL(
				`${cell.base}/tour/reset?${cell.run}`,
				SERVER_URL,
			).toString(),
			{
				method: "POST",
			},
		);

		const panoMeta = [];
		for (let i = 0; i < anchors.length; i++) {
			const a = anchors[i];
			const pos = Array.isArray(a.position) ? a.position : [0, 0, 0];
			const id =
				typeof a.id === "string" && a.id
					? a.id
					: `anchor-${String(i).padStart(3, "0")}`;
			camera.position.set(pos[0], pos[1], pos[2]);
			// Each capture is a full 360°; forward only seeds /pano's initial view.
			const forward = [0, 0, -1];
			panoAutoEl.textContent = `capturing ${i + 1}/${anchors.length}…`;
			const blob = await capturePanoBlob();
			await uploadPano(cell, id, blob);
			panoMeta.push({
				id,
				file: `${id}.jpg`,
				position: pos,
				forward,
				reason: typeof a.reason === "string" ? a.reason : undefined,
				name: typeof a.name === "string" ? a.name : undefined,
			});
		}

		// Decimate + persist the proxy from the merged scene.
		let hasProxy = false;
		panoAutoEl.textContent = "building proxy…";
		const merged = await buildMergedSceneGlbBuffer();
		if (merged) {
			const proxyRes = await fetch(
				new URL(
					`${cell.base}/tour/proxy?${cell.run}`,
					SERVER_URL,
				).toString(),
				{
					method: "POST",
					headers: { "Content-Type": "model/gltf-binary" },
					body: merged,
				},
			);
			hasProxy = proxyRes.ok;
			if (!proxyRes.ok) {
				appendEvent({
					kind: "run.error",
					message: `auto-tour proxy → ${proxyRes.status} (tour saved without it)`,
				});
			}
		}

		// Bird's-eye minimap slices, grouped by Y level. Best-effort: a failure
		// here leaves the tour fully usable, just without the minimap overlay.
		let minimaps = [];
		try {
			minimaps = await buildMinimaps(cell, panoMeta, (li, n) => {
				panoAutoEl.textContent = `rendering minimap ${li + 1}/${n}…`;
			});
		} catch (e) {
			minimaps = [];
			appendEvent({
				kind: "run.error",
				message: `auto-tour minimaps failed (tour saved without them): ${e.message}`,
			});
		}

		panoAutoEl.textContent = "writing manifest…";
		const manifest = {
			version: 1,
			proxy: hasProxy ? "proxy.glb" : null,
			planner_model: typeof plan.model === "string" ? plan.model : null,
			namer_model:
				typeof plan.namer_model === "string" ? plan.namer_model : null,
			planner_reasoning:
				typeof plan.reasoning === "string" ? plan.reasoning : null,
			panos: panoMeta,
			minimaps,
		};
		const manRes = await fetch(
			new URL(
				`${cell.base}/tour/manifest?${cell.run}`,
				SERVER_URL,
			).toString(),
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(manifest),
			},
		);
		if (!manRes.ok) throw new Error(`/tour/manifest → ${manRes.status}`);
		const { tour_url } = await manRes.json();
		const absUrl = new URL(tour_url, SERVER_URL).toString();
		setStatus(
			`auto-tour ready · ${panoMeta.length} panos · open /pano?tour=${absUrl}`,
			"hdr",
		);
		appendEvent({
			kind: "run.done",
			message: `auto-tour persisted: ${absUrl}`,
		});
	} catch (e) {
		appendEvent({
			kind: "run.error",
			message: `auto-tour failed: ${e.message}`,
		});
	} finally {
		camera.position.copy(camSnapshot.pos);
		controls.target.copy(camSnapshot.target);
		cameraUserMoved = camSnapshot.userMoved;
		controls.update();
		panoAutoEl.textContent = "⚡ auto-tour";
		panoBusy = false;
		updatePanoButtons();
	}
});

// Boot: fetch active run first (so /slots reflects the right cell set),
// then load slot + model lists, pick remembered (or defaults), subscribe.
(async () => {
	await refreshRuns();
	await refreshSlots();
	if (slotSummaries.length === 0) {
		setStatus("no slots reported by server", "err");
		return;
	}
	if (availableModels.length === 0) {
		setStatus("no models reported by server", "err");
		return;
	}
	let savedSlot = null;
	let savedModel = null;
	try {
		savedSlot = localStorage.getItem(SLOT_STORAGE_KEY);
	} catch {}
	try {
		savedModel = localStorage.getItem(MODEL_STORAGE_KEY);
	} catch {}
	const slotPick =
		slotSummaries.find((s) => s.id === savedSlot)?.id ??
		slotSummaries[0].id;
	const modelPick = availableModels.includes(savedModel)
		? savedModel
		: (defaultModelAlias ?? availableModels[0]);

	// Restore the open tabs (each pins a run + slot + model). First boot after
	// this feature ships has none → seed a single tab, honoring the previously-
	// remembered version run so existing users land where they left off.
	const saved = loadSavedTabs();
	if (saved) {
		openTabs.push(...saved.tabs);
		activeTabId = saved.tabs.some((t) => t.id === saved.active)
			? saved.active
			: saved.tabs[0].id;
	} else {
		let savedVersion = null;
		try {
			savedVersion = localStorage.getItem(VERSION_STORAGE_KEY);
		} catch {}
		const initRun =
			savedVersion && runExists(savedVersion) ? savedVersion : currentRun;
		const tab = {
			id: nextTabId(),
			run: initRun,
			slot: slotPick,
			model: modelPick,
		};
		openTabs.push(tab);
		activeTabId = tab.id;
	}
	renderTabBar();

	// Drive the initial canvas from the active tab.
	const active = openTabs.find((t) => t.id === activeTabId);
	const bootSlot = active.slot ?? slotPick;
	const bootModel = active.model ?? modelPick;
	if (active.run && active.run !== currentRun) {
		// Pre-seed the cell so switchRun's trailing switchView targets it.
		currentSlotId = bootSlot;
		currentModel = bootModel;
		await switchRun(active.run);
	} else {
		switchView(bootSlot, bootModel);
	}
	await refreshVersions();
})();

// Keep tab status dots + run list fresh — runs change less often, so a
// slower cadence is fine.
setInterval(refreshSlots, 2000);
setInterval(refreshRuns, 10000);
setInterval(refreshVersions, 2000);

// --- asset mode: library ▸ generated ----------------------------------------
//
// A pure client-side view switch over the SAME scene. The tree, bboxes, log,
// and event stream always come from the library build (events.jsonl); flipping
// the toggle only re-points the one-connection mesh bundle at the other folder
// (objects/ ▸ generated/<version>/objects-generated-optimized/). Each generated
// version is an independent from-scratch take on the same layout; a version is
// produced on demand by the gate button ("⚡ generate" resumes the selected
// version, "+ new version" forks a fresh one), which runs Nano-Banana + Trellis
// reusing the library layout — same bboxes, freshly generated meshes.

function slotGenerateUrl(
	slotId,
	model,
	run = currentRun,
	version = genVersion,
	optimized = true,
) {
	// Shared by the GET poll and the POST resume — both target one version
	// (omitted ⇒ the server's latest, used before a version is resolved).
	const u = new URL(
		`/slots/${encodeURIComponent(slotId)}/${encodeURIComponent(model)}/generate?run=${encodeURIComponent(run)}`,
		SERVER_URL,
	);
	if (version != null) u.searchParams.set("version", version);
	// The GET poll passes the toggle so it enumerates the same folder the bundle
	// streams (raw ↔ optimized), keeping per-mesh version tokens consistent. POST
	// callers leave it at the default — the resume/new build is folder-agnostic.
	if (!optimized) u.searchParams.set("optimized", "0");
	return u.toString();
}

function updateGenerateGate() {
	const cellReady = currentSlotId !== null && currentModel !== null;
	const show = assetMode === "generated";
	if (generateGateEl) {
		generateGateEl.style.display = show ? "" : "none";
		generateGateEl.disabled = generating || !cellReady;
		// No version yet → the first press creates version 1; otherwise it resumes
		// the selected version.
		generateGateEl.textContent = generating
			? "generating…"
			: genVersion == null
				? "⚡ generate"
				: `⚡ generate v${genVersion}`;
	}
	if (genVersionNewEl) {
		// A new version is its own isolated build, so it stays available even while
		// the currently-viewed version is mid-build (the server runs versions
		// concurrently); only the ⚡ resume button is gated on `generating`.
		genVersionNewEl.style.display = show ? "" : "none";
		genVersionNewEl.disabled = !cellReady;
	}
	if (genFromImagesEl) {
		// Forks a fresh version that reuses the CURRENTLY-VIEWED version's images, so
		// it only makes sense once a version is selected. Like "+ new version" it's a
		// separate isolated build, so it's not gated on `generating`.
		genFromImagesEl.style.display =
			show && genVersion != null ? "" : "none";
		genFromImagesEl.disabled = !cellReady;
		genFromImagesEl.textContent = `⚡ rebuild from v${genVersion ?? ""} imgs`;
	}
	if (genVersionPickerEl) {
		// Only meaningful once at least one version exists; hidden in library mode.
		genVersionPickerEl.style.display =
			show && genVersions.length > 0 ? "" : "none";
		genVersionPickerEl.disabled = !cellReady;
	}
	if (genOptimizeToggleEl) {
		// Same lifecycle as the version picker: only meaningful once a version's
		// assets exist, hidden in library mode.
		genOptimizeToggleEl.style.display =
			show && genVersions.length > 0 ? "" : "none";
		genOptimizeToggleEl.disabled = !cellReady;
	}
}

// Repopulate the version dropdown from `genVersions`, keeping `genVersion`
// selected. Cheap enough to call on every gate poll; rebuilds only on change.
function renderGenVersionPicker() {
	if (!genVersionPickerEl) return;
	const want =
		genVersions.map((v) => `v${v}`).join("|") + `@${genVersion ?? ""}`;
	if (genVersionPickerEl.dataset.sig === want) return;
	genVersionPickerEl.dataset.sig = want;
	genVersionPickerEl.innerHTML = "";
	for (const v of genVersions) {
		const opt = document.createElement("option");
		opt.value = v;
		opt.textContent = `v${v}`;
		if (v === genVersion) opt.selected = true;
		genVersionPickerEl.appendChild(opt);
	}
}

function applyAssetModeUI() {
	if (assetModeToggleEl) {
		assetModeToggleEl.textContent = `assets: ${assetMode}`;
		assetModeToggleEl.classList.toggle(
			"generated",
			assetMode === "generated",
		);
	}
	if (genOptimizeToggleEl) {
		genOptimizeToggleEl.textContent = `optimized: ${genOptimized ? "on" : "off"}`;
		genOptimizeToggleEl.classList.toggle("off", !genOptimized);
	}
	renderGenVersionPicker();
	updateGenerateGate();
}

// Swap the rendered meshes for the active asset mode WITHOUT disturbing the
// tree/bboxes: drop the attached GLBs, bump sceneGen so any in-flight bundle is
// ignored, then re-stream from the now-current mode's folder.
function reloadMeshesForAssetMode() {
	if (currentSlotId === null || currentModel === null) return;
	for (const obj of modelsById.values()) {
		sceneRoot.remove(obj);
		disposeObject3D(obj);
	}
	modelsById.clear();
	// Bundle reload re-attaches everything; drop version baselines so the next
	// gate poll re-establishes them against the freshly streamed meshes, and drop
	// every per-asset optimize override so the reloaded scene starts uniform on
	// the (possibly just-changed) scene-wide setting.
	genMeshVersions.clear();
	genMeshSymmetry.clear();
	genMeshOptimized.clear();
	resetModelQueue();
	updateMissingMeshCount();
	prefetchMeshBundle(currentSlotId, currentModel, sceneGen);
}

// Per-asset effective optimize state: the override the user pinned from the
// detail panel, else the scene-wide default. The single place every per-mesh url
// reads, so one toggled asset streams from the other folder while the rest of
// the scene stays on the scene setting.
function meshOptimizedFor(id) {
	return genMeshOptimized.has(id) ? genMeshOptimized.get(id) : genOptimized;
}

// The generated subfolder a url points at: the served optimized twin (default)
// or the raw, bbox-fitted Trellis mesh. `id` is optional — omitted (bundle /
// global paths) it reads the scene-wide setting; given, it honors that asset's
// per-asset override. Drives every per-mesh url (incremental load + detail
// preview) so they track whichever folder the asset should show.
function generatedSubdir(id) {
	const optimized = id != null ? meshOptimizedFor(id) : genOptimized;
	return optimized ? "objects-generated-optimized" : "objects-generated";
}

function generatedArtifactUrl(slotId, model, run, version, id, v) {
	// Served from the selected version + this asset's effective optimize folder
	// (its per-asset override, else the scene-wide toggle): the optimized twin
	// (decimated + KTX2 + Meshopt) by default, or the raw objects-generated/ mesh
	// when toggled off — the same folders the /meshes?mode=generated bundle uses.
	// `v` (the GLB's mtime) busts the loader + browser cache so a regenerated
	// asset re-fetches instead of serving stale bytes keyed on the bare URL.
	const bust = v != null ? `?v=${encodeURIComponent(v)}` : "";
	return new URL(
		`/artifacts/${encodeURIComponent(run)}/${encodeURIComponent(slotId)}/${encodeURIComponent(model)}/generated/${encodeURIComponent(version)}/${generatedSubdir(id)}/${encodeURIComponent(id)}.glb${bust}`,
		SERVER_URL,
	).toString();
}

// The generated build's preview image: the Nano-Banana render copied beside each
// mesh (<id>.png in both the raw and optimized folders), so the detail preview's
// image matches the mesh shown in "generated" mode. Reads the same folder as the
// toggle. `v` busts the cache on regenerate, same as the GLB.
function generatedImageUrl(slotId, model, run, version, id, v) {
	const bust = v != null ? `?v=${encodeURIComponent(v)}` : "";
	return new URL(
		`/artifacts/${encodeURIComponent(run)}/${encodeURIComponent(slotId)}/${encodeURIComponent(model)}/generated/${encodeURIComponent(version)}/${generatedSubdir(id)}/${encodeURIComponent(id)}.png${bust}`,
		SERVER_URL,
	).toString();
}

// Attach one generated GLB by id. Mirrors attachBundleMesh's attach half. On a
// load failure (the file can be caught mid-write while a build is running) it
// bails WITHOUT recording an error, so the next poll simply retries it once the
// export has finished — no permanent skip, no completion-time full reload.
const _genLoading = new Set();
async function loadGeneratedMesh(id, url, gen) {
	if (gen !== sceneGen) return;
	let gltf;
	try {
		gltf = await loader.loadAsync(url);
	} catch {
		return;
	}
	if (gen !== sceneGen) {
		disposeObject3D(gltf.scene);
		return;
	}
	prepareLoadedScene(gltf.scene);
	gltf.scene.name = `mesh:${id}`;
	gltf.scene.userData.pickId = id;
	const prev = modelsById.get(id);
	if (prev) {
		sceneRoot.remove(prev);
		disposeObject3D(prev);
	}
	sceneRoot.add(gltf.scene);
	modelsById.set(id, gltf.scene);
	applyModelVisibility(id);
	upsertAsset(id, { status: "loaded" });
	scheduleFitToScene();
}

// Reconcile attached generated meshes against the gate's reported set. `meshes`
// is [{id, v}] where v is the optimized GLB's mtime. Three cases per id:
//   * not attached            → load it (a build producing meshes as it goes).
//   * attached, version unseen → record the baseline (bundle-loaded), no reload.
//   * attached, version changed → a regenerate landed; reload in place with a
//     cache-busted URL (loadGeneratedMesh disposes the old mesh and swaps the
//     new one in), then clear the regenerating flag + refresh the open detail.
function syncGeneratedMeshes(meshes, slotId, model, run, version) {
	for (const { id, v } of meshes) {
		if (_genLoading.has(id)) continue;
		const attached = modelsById.has(id);
		const known = genMeshVersions.get(id);
		if (attached && known === undefined) {
			genMeshVersions.set(id, v); // bundle-loaded — baseline only, don't reload
			continue;
		}
		if (attached && known === v) continue; // unchanged
		// While the one-connection bundle for this scene is still streaming, let it
		// deliver not-yet-attached ids instead of racing it with a per-mesh fetch —
		// raw (un-optimized) meshes are ~100x heavier, so a duplicate pull is very
		// costly. Once the bundle finishes (streaming=false) the poll tops up
		// anything it didn't carry (parse miss, or a mesh built after the snapshot).
		if (!attached && meshBundle.gen === sceneGen && meshBundle.streaming)
			continue;
		const isReload = attached;
		genMeshVersions.set(id, v);
		_genLoading.add(id);
		const gen = sceneGen;
		loadGeneratedMesh(
			id,
			generatedArtifactUrl(slotId, model, run, version, id, v),
			gen,
		).finally(() => {
			_genLoading.delete(id);
			const wasRegenerating = regeneratingIds.delete(id);
			const wasUnsymmetrizing = unsymmetrizingIds.delete(id);
			const wasSymmetrizing = symmetrizingIds.delete(id);
			if (
				(isReload ||
					wasRegenerating ||
					wasUnsymmetrizing ||
					wasSymmetrizing) &&
				id === selectedBboxId
			) {
				renderTreeDetail();
			}
		});
	}
}

// Per-asset optimize override: flip ONE generated asset between its optimized
// twin and its raw Trellis mesh, leaving every other asset on the scene-wide
// `genOptimized` setting. Records an override (or drops it when the choice lands
// back on the scene default), then reloads just this mesh — generatedSubdir(id)
// already points its url at the now-effective folder. The override is view-only
// and resets whenever the scene re-streams (cell switch, asset-mode /
// scene-optimize / version change). Generated mode only.
//
// We deliberately don't touch genMeshVersions here: the gate poll keeps that
// baselined to the scene-default folder's mtime, so its steady-state
// `known === v` check still skips this id (no revert), while a regenerate that
// bumps the mtime reloads it from the override folder via generatedSubdir(id).
function toggleMeshOptimized(id) {
	if (assetMode !== "generated" || genVersion == null) return;
	if (currentSlotId === null || currentModel === null) return;
	if (_genLoading.has(id)) return;
	const next = !meshOptimizedFor(id);
	if (next === genOptimized) genMeshOptimized.delete(id);
	else genMeshOptimized.set(id, next);
	const v = genMeshVersions.get(id);
	const gen = sceneGen;
	_genLoading.add(id);
	loadGeneratedMesh(
		id,
		generatedArtifactUrl(
			currentSlotId,
			currentModel,
			currentRun,
			genVersion,
			id,
			v,
		),
		gen,
	).finally(() => {
		_genLoading.delete(id);
		if (id === selectedBboxId) renderTreeDetail();
	});
	// Repaint the detail button (label + disabled) immediately, before the load.
	if (id === selectedBboxId) renderTreeDetail();
}

// The gate: kick off a from-scratch build for the open cell. The button is
// disabled while one is in flight (and re-enabled once the poll sees it done),
// so a build can't be triggered on top of itself.
async function startGenerate() {
	if (generating || currentSlotId === null || currentModel === null) return;
	const slotId = currentSlotId,
		model = currentModel,
		run = currentRun;
	let body;
	try {
		const res = await fetch(
			slotGenerateUrl(slotId, model, run, genVersion),
			{ method: "POST" },
		);
		body = await res.json().catch(() => ({}));
		if (!res.ok) {
			setStatus(
				`generate failed: ${body.detail ?? `HTTP ${res.status}`}`,
				"err",
			);
			return;
		}
	} catch (e) {
		setStatus(`generate failed: ${e.message}`, "err");
		return;
	}
	// Adopt the version the server resolved (creates v1 when the cell had none),
	// switching the view to it if it differs from what we were showing.
	if (body.version != null && body.version !== genVersion) {
		selectGenVersion(body.version);
	}
	generating = true;
	_genWasRunning = true;
	updateGenerateGate();
	setStatus(
		`generating v${genVersion ?? body.version} assets — Nano-Banana + Trellis (this can take a while)…`,
	);
}

// Switch the viewed generated version: stream that version's finished meshes now;
// the gate poll tops up the rest and records per-mesh version tokens.
function selectGenVersion(v) {
	if (v == null) return;
	const next = String(v);
	if (next === genVersion) return;
	genVersion = next;
	if (!genVersions.includes(next)) {
		genVersions = [...genVersions, next].sort(
			(a, b) => Number(a) - Number(b),
		);
	}
	applyAssetModeUI();
	reloadMeshesForAssetMode();
	if (selectedBboxId !== null) renderTreeDetail();
}

// Create a fresh from-scratch version of this scene (its own generated/<v>/
// folder) and start building into it. The new (empty) version is selected
// immediately; assets pop in as the build produces them.
async function newGenVersion() {
	if (currentSlotId === null || currentModel === null) return;
	const slotId = currentSlotId,
		model = currentModel,
		run = currentRun;
	let body;
	try {
		const u = new URL(slotGenerateUrl(slotId, model, run, null));
		u.searchParams.set("new", "true");
		const res = await fetch(u.toString(), { method: "POST" });
		body = await res.json().catch(() => ({}));
		if (!res.ok) {
			setStatus(
				`new version failed: ${body.detail ?? `HTTP ${res.status}`}`,
				"err",
			);
			return;
		}
	} catch (e) {
		setStatus(`new version failed: ${e.message}`, "err");
		return;
	}
	if (body.version != null) selectGenVersion(body.version);
	generating = true;
	_genWasRunning = true;
	updateGenerateGate();
	setStatus(
		`generating new version v${body.version} — Nano-Banana + Trellis (this can take a while)…`,
	);
}

// Fork a new version that REUSES the currently-viewed version's Nano-Banana
// images and rebuilds every mesh fresh with Trellis — no new image generation.
// Use it to re-roll a bad mesh pass while keeping a good image pass: select the
// version whose images you like, click this, get a new version built from them.
async function regenerateSceneFromImages() {
	if (currentSlotId === null || currentModel === null || genVersion == null)
		return;
	const slotId = currentSlotId,
		model = currentModel,
		run = currentRun;
	const sourceVersion = genVersion;
	let body;
	try {
		const res = await fetch(
			new URL(
				`/slots/${encodeURIComponent(slotId)}/${encodeURIComponent(model)}/generate-from-images/${encodeURIComponent(sourceVersion)}?run=${encodeURIComponent(run)}`,
				SERVER_URL,
			),
			{ method: "POST" },
		);
		body = await res.json().catch(() => ({}));
		if (!res.ok) {
			setStatus(
				`rebuild from images failed: ${body.detail ?? `HTTP ${res.status}`}`,
				"err",
			);
			return;
		}
	} catch (e) {
		setStatus(`rebuild from images failed: ${e.message}`, "err");
		return;
	}
	if (body.version != null) selectGenVersion(body.version);
	generating = true;
	_genWasRunning = true;
	updateGenerateGate();
	setStatus(
		`generating v${body.version} from v${sourceVersion}'s ${body.seeded_images ?? ""} image(s) — Trellis only, no new Nano-Banana (this can take a while)…`,
	);
}

// Poll the open cell's build state while the generated view is active: keep the
// gate's disabled state in sync, and pull the freshly built meshes in the
// moment a build we were watching finishes.
async function refreshGenerateGate() {
	const cellKey =
		currentSlotId !== null && currentModel !== null
			? `${currentRun}/${currentSlotId}/${currentModel}/${genVersion ?? ""}`
			: null;
	if (cellKey !== _genCellKey) {
		_genCellKey = cellKey;
		_genWasRunning = false;
	}
	if (
		assetMode !== "generated" ||
		currentSlotId === null ||
		currentModel === null
	) {
		_genWasRunning = false;
		return;
	}
	const slotId = currentSlotId;
	const model = currentModel;
	const run = currentRun;
	const reqVersion = genVersion; // version this poll asked about (may be null)
	const reqOptimized = genOptimized; // folder this poll asked about (raw ↔ optimized)
	let status;
	try {
		const res = await fetch(
			slotGenerateUrl(slotId, model, run, reqVersion, reqOptimized),
			{ cache: "no-store" },
		);
		if (!res.ok) return;
		status = await res.json();
	} catch {
		return;
	}
	// Bail if the user switched cell/mode or picked a different version while the
	// poll was in flight — a later poll handles the new selection.
	if (
		assetMode !== "generated" ||
		currentSlotId !== slotId ||
		currentModel !== model ||
		currentRun !== run ||
		genVersion !== reqVersion ||
		genOptimized !== reqOptimized
	) {
		return;
	}
	// Refresh the available versions + picker.
	const nextVersions = status.versions ?? [];
	const versionsChanged = nextVersions.join("|") !== genVersions.join("|");
	genVersions = nextVersions;
	// Adopt the server's resolved version when we hadn't picked one yet. The cell
	// load already streamed the latest (version omitted), which equals this, so no
	// reload — syncGeneratedMeshes below just records the per-mesh version tokens.
	if (genVersion == null && status.version != null) {
		genVersion = status.version;
	}
	if (versionsChanged || genVersionPickerEl?.value !== genVersion)
		renderGenVersionPicker();
	generating = !!status.running;
	updateGenerateGate();
	// Reconcile attached meshes with the build's set for the version we're viewing:
	// new ones pop in as the build produces them, and a regenerated asset (same id,
	// bumped version) reloads in place. A node caught mid-write just isn't added
	// this tick and is retried next (loadGeneratedMesh doesn't cache the failure).
	if (status.version === genVersion && genVersion != null) {
		const meshes =
			status.meshes ?? (status.ids ?? []).map((id) => ({ id, v: null }));
		for (const m of meshes)
			genMeshSymmetry.set(m.id, {
				plane: m.sym ?? "none",
				was: m.symWas ?? null,
			});
		syncGeneratedMeshes(meshes, slotId, model, run, genVersion);
	}
	if (_genWasRunning && !status.running) {
		setStatus(
			`generated ${status.count} asset(s) — showing v${genVersion}`,
		);
		// Build idle: anything still flagged regenerating/un-symmetrizing either
		// landed (cleared in syncGeneratedMeshes) or failed without changing — drop
		// the stragglers so the detail buttons don't stick on their in-flight label.
		if (
			regeneratingIds.size > 0 ||
			unsymmetrizingIds.size > 0 ||
			symmetrizingIds.size > 0
		) {
			regeneratingIds.clear();
			unsymmetrizingIds.clear();
			symmetrizingIds.clear();
			if (selectedBboxId !== null) renderTreeDetail();
		}
	}
	_genWasRunning = !!status.running;
}

assetModeToggleEl?.addEventListener("click", () => {
	assetMode = assetMode === "library" ? "generated" : "library";
	try {
		localStorage.setItem(ASSET_MODE_STORAGE_KEY, assetMode);
	} catch {}
	applyAssetModeUI();
	// Bulk-load the target build's meshes over one connection; the gate poll then
	// tops up any that finish afterward and syncs the gate's enabled state.
	reloadMeshesForAssetMode();
	// Re-resolve the open detail preview for the new mode (its urls read
	// assetMode); the generated mesh repaints once the bundle re-streams it.
	if (selectedBboxId !== null) renderTreeDetail();
	// Populate the version picker + resolve the selected version now rather than
	// waiting up to a full poll tick.
	if (assetMode === "generated") refreshGenerateGate();
});
genOptimizeToggleEl?.addEventListener("click", () => {
	genOptimized = !genOptimized;
	try {
		localStorage.setItem(
			GEN_OPTIMIZED_STORAGE_KEY,
			genOptimized ? "optimized" : "raw",
		);
	} catch {}
	applyAssetModeUI();
	// Re-stream the cell's meshes from the now-current folder (raw ↔ optimized);
	// the gate poll then reconciles per-mesh version tokens against that folder.
	reloadMeshesForAssetMode();
	// The open detail preview reads the toggle for its image + mini-viewer urls.
	if (selectedBboxId !== null) renderTreeDetail();
	// Re-resolve the gate against the new folder now rather than waiting a tick.
	refreshGenerateGate();
});
genVersionPickerEl?.addEventListener("change", () => {
	selectGenVersion(genVersionPickerEl.value);
});
genVersionNewEl?.addEventListener("click", newGenVersion);
genFromImagesEl?.addEventListener("click", regenerateSceneFromImages);
generateGateEl?.addEventListener("click", startGenerate);
applyAssetModeUI();
setInterval(refreshGenerateGate, 2000);

// --- replay → gif ----------------------------------------------------------
//
// User clicks "replay → gif" → a modal opens. The user can preview the
// build (re-dispatches the entire `recordedEvents` log against the live
// scene) and then export an animated GIF.
//
// Timing model: one frame per recorded event, each held for the chosen
// `interval` (ms). That single knob is both the gif frame delay and the
// preview step delay, so encoded duration is frames × interval + a short end
// hold — no fps/speed coupling. We walk the log, dispatch each event, draw
// once, and (in record mode) push the canvas pixels into a gif.js encoder.
// Mesh GLB loads remain async — they'll pop into the gif whenever the loader
// resolves, which for cached artifacts is usually within a frame or two.

function updateReplayButton() {
	if (!replayGifEl) return;
	replayGifEl.disabled = recordedEvents.length === 0;
}
updateReplayButton();

let replayActive = false; // true while preview or record is running
let replayCancelRequested = false;
let replayInProgress = false; // gates concurrent invocations of run()

// Default max output width for the encoded gif. The renderer's backing buffer
// can be 2-3× the CSS size on retina, which makes gifs enormous; we downscale
// every captured frame through an offscreen canvas to this cap before encoding.
// The replay modal's resolution dropdown overrides this per-render (see
// readReplayResolution); this is the fallback when the control is unparseable
// and the value of its default-selected option.
const GIF_MAX_WIDTH = 540;

function setReplayStatus(text, cls = "") {
	replayStatusEl.textContent = text;
	replayStatusEl.className = cls;
}

function setReplayProgress(frac) {
	replayProgressBarEl.style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`;
}

// ms each recorded event ("change") is held on screen. This is the single
// timing knob: it's both the gif frame delay and the preview step delay, so
// the encoded duration is simply frames × interval (+ end hold) with no
// fps/speed coupling.
function readReplayInterval() {
	return parseInt(replayIntervalEl.value, 10) || 50;
}

// Selected gif width cap from the resolution dropdown. The "native" option has
// value 0 → no cap (Infinity), letting the downscale factor clamp to 1 so the
// gif encodes at the renderer's full backing-buffer size.
function readReplayResolution() {
	const v = parseInt(replayResolutionEl.value, 10);
	if (Number.isNaN(v)) return GIF_MAX_WIDTH;
	return v <= 0 ? Infinity : v;
}

// Show the resulting encoded dimensions for the current pick next to the
// dropdown, using the same downscale math runReplay applies so the label
// matches what actually gets encoded.
function updateResolutionEstimate() {
	const maxW = readReplayResolution();
	const srcW = renderer.domElement.width;
	const srcH = renderer.domElement.height;
	const scale = Math.min(1, maxW / srcW);
	const w = Math.max(2, Math.round(srcW * scale));
	const h = Math.max(2, Math.round(srcH * scale));
	replayResolutionValEl.textContent = `${w}×${h}`;
}

// One frame per recorded event — each "change" gets its own gif frame, held
// for the chosen interval. Logs longer than REPLAY_MAX_FRAMES group events
// evenly so file size / encode time stay bounded; the duration estimate reads
// the real frame count so it always matches the encode.
const REPLAY_MAX_FRAMES = 600;

// Final still frame appended after the build completes so the finished scene
// registers before the gif loops. Counted in the duration estimate.
const REPLAY_HOLD_MS = 800;

function planFrames(events) {
	const perFrame = Math.max(1, Math.ceil(events.length / REPLAY_MAX_FRAMES));
	const batches = [];
	for (let i = 0; i < events.length; i += perFrame) {
		batches.push(events.slice(i, i + perFrame));
	}
	return batches;
}

function updateDurationEstimate() {
	const interval = readReplayInterval();
	const batches = planFrames(recordedEvents);
	const seconds = (batches.length * interval + REPLAY_HOLD_MS) / 1000;
	replayEventCountEl.textContent = String(recordedEvents.length);
	replayDurationEstEl.textContent =
		batches.length === 0
			? "—"
			: `${seconds.toFixed(1)}s · ${batches.length} frames`;
}

replayIntervalEl.addEventListener("input", () => {
	replayIntervalValEl.textContent = `${replayIntervalEl.value} ms`;
	updateDurationEstimate();
});
replayResolutionEl.addEventListener("change", updateResolutionEstimate);

function openReplayModal() {
	if (recordedEvents.length === 0) return;
	replayModalEl.classList.add("open");
	replayIntervalValEl.textContent = `${replayIntervalEl.value} ms`;
	updateDurationEstimate();
	updateResolutionEstimate();
	setReplayStatus("idle");
	setReplayProgress(0);
	showReplayPlaceholder();
}

function closeReplayModal() {
	// If a replay is mid-run, request cancellation; the run loop checks the
	// flag between frames and bails out, then we close. If nothing is
	// running, close immediately.
	if (replayActive) {
		replayCancelRequested = true;
		return;
	}
	replayModalEl.classList.remove("open");
	// Free the gif blob URL when the modal closes — keeping it alive across
	// sessions would leak memory if the user repeatedly opens/saves.
	if (lastGifUrl) {
		URL.revokeObjectURL(lastGifUrl);
		lastGifUrl = null;
		lastGifBlob = null;
		replayResultImgEl.removeAttribute("src");
		replayDownloadEl.disabled = true;
	}
}

replayGifEl.addEventListener("click", openReplayModal);
replayCloseEl.addEventListener("click", closeReplayModal);
replayModalEl.addEventListener("click", (ev) => {
	if (ev.target === replayModalEl) closeReplayModal();
});

// Walk the events for every `model` emission and fetch + parse each GLB in
// parallel before dispatch starts. The returned Map<id, { gltf } | { error }>
// is consumed by loadModel during dispatch so meshes attach synchronously
// and land in the same gif frame as their emitting event. Re-emissions of
// the same id collapse to the latest url (mirrors the live last-write-wins
// behavior).
async function preloadReplayGlbs(events) {
	const byId = new Map();
	for (const e of events) {
		if (e.kind === "model" && e.url) byId.set(e.id, e);
	}
	const total = byId.size;
	const cache = new Map();
	setReplayProgress(0);
	if (total === 0) {
		setReplayStatus("no meshes to preload");
		return cache;
	}
	setReplayStatus(`preloading meshes… 0/${total}`, "recording");
	let done = 0;
	await Promise.all(
		Array.from(byId.values()).map(async (e) => {
			const absUrl = new URL(e.url, SERVER_URL).toString();
			try {
				const gltf = await loader.loadAsync(absUrl);
				cache.set(e.id, { gltf });
			} catch (err) {
				cache.set(e.id, { error: err });
			} finally {
				done += 1;
				setReplayStatus(
					`preloading meshes… ${done}/${total}`,
					"recording",
				);
				setReplayProgress(done / total);
			}
		}),
	);
	return cache;
}

// Snapshot the live scene + ephemeral UI state, wipe everything, replay the
// recorded log, optionally encode frames into a gif, then leave the scene
// holding the final-state of the replay (which matches the live state).
// Returns the encoded blob in record mode, or null in preview mode.
async function runReplay({ record }) {
	if (replayInProgress) return null;
	// Replay tears the scene down and re-dispatches the log; refuse while the
	// prompt-tuning sandbox owns the canvas + frozen state.
	if (sandboxActive) return null;
	replayInProgress = true;
	replayActive = true;
	replayCancelRequested = false;
	setReplayProgress(0);

	// Disconnect from the live SSE stream so server-pushed events don't race
	// the replay. The user can reconnect via the resume / slot-switch paths
	// afterwards; for a finished run nothing's incoming anyway.
	const reconnectAfter = currentSource !== null;
	const slotForReconnect = currentSlotId;
	const modelForReconnect = currentModel;
	if (currentSource) {
		currentSource.close();
		currentSource = null;
	}

	// Snapshot the recorded log + the current camera state so we restore them
	// after the replay finishes (or the user cancels). The replay reuses the
	// same dispatch path which resets `highestEventIndex` and clears state, so
	// we have to take care to leave the user where they started.
	const events = recordedEvents.slice();
	const camSnapshot = {
		pos: camera.position.clone(),
		target: controls.target.clone(),
		userMoved: cameraUserMoved,
	};

	// Wipe everything that the replay will rebuild.
	clearScene();
	clearLog();
	clearAssets();
	treeClear();
	clearMeshErrors();
	highestEventIndex = -1;
	recordedEvents.length = 0;
	updateReplayButton();

	const batches = planFrames(events);
	const frameIntervalMs = readReplayInterval();

	let gif = null;
	let gifScaler = null;
	let gifScalerCtx = null;
	if (record) {
		if (typeof window.GIF === "undefined") {
			setReplayStatus("gif.js library failed to load", "err");
			replayActive = false;
			replayInProgress = false;
			return null;
		}
		// The renderer's backing buffer is `cssSize * devicePixelRatio`, so on
		// a retina display this is 2-3× the visible canvas. Feeding that
		// straight into gif.js produces enormous (10-100MB) gifs. Downscale
		// every frame through an offscreen canvas to the width cap chosen in
		// the resolution dropdown before adding it.
		const maxWidth = readReplayResolution();
		const srcW = renderer.domElement.width;
		const srcH = renderer.domElement.height;
		const scale = Math.min(1, maxWidth / srcW);
		const gifW = Math.max(2, Math.round(srcW * scale));
		const gifH = Math.max(2, Math.round(srcH * scale));
		gifScaler = document.createElement("canvas");
		gifScaler.width = gifW;
		gifScaler.height = gifH;
		gifScalerCtx = gifScaler.getContext("2d");
		gif = new window.GIF({
			workers: 2,
			// gif.js's `quality` is inverted: HIGHER = coarser palette quantization
			// = smaller files. 10 is the default; 30 trims size further with mild
			// banding in mesh gradients (fine for benchmark gifs of mostly flat
			// wireframes + a dark background).
			quality: 30,
			workerScript: "/vendor/gifjs/gif.worker.js",
			width: gifW,
			height: gifH,
			background: "#101114",
		});
	}

	// Lock the camera for the duration of the replay — otherwise fitToScene
	// would refit on every mesh load and the gif would jitter. The user's
	// current view persists through clearScene() (camera is in its own scene
	// graph), so the gif captures whatever angle they're looking at right now;
	// camSnapshot still restores the same view afterwards.
	cameraUserMoved = true;

	// Start the preview stage with a blank frame so the user sees the modal
	// switch out of the placeholder state immediately.
	replayStageEl.classList.remove("show-result");
	replayStageEl.classList.remove("empty");

	// Preload every GLB referenced by the recorded events so the dispatch
	// loop can attach meshes synchronously — otherwise mesh fetches lag the
	// events that triggered them and the gif cuts off mid-load. If the user
	// cancels during preload, fall through: the dispatch loop will see the
	// cancel flag, skip itself, and let the normal cleanup tail run (which
	// restores the camera + reconnects the SSE stream).
	replayPreloadCache = await preloadReplayGlbs(events);
	setReplayProgress(0);
	if (!replayCancelRequested) {
		setReplayStatus(
			record ? "recording…" : "previewing…",
			record ? "recording" : "",
		);
	}

	for (let i = 0; i < batches.length; i++) {
		if (replayCancelRequested) break;
		const batch = batches[i];
		for (const ev of batch) dispatchForReplay(ev);
		// One render after each batch (renderer.render is also driven by the
		// animate() loop, but doing it explicitly here guarantees a fresh frame
		// before capture).
		controls.update();
		renderer.render(scene, camera);
		if (gif) {
			gifScalerCtx.drawImage(
				renderer.domElement,
				0,
				0,
				gifScaler.width,
				gifScaler.height,
			);
			gif.addFrame(gifScaler, { copy: true, delay: frameIntervalMs });
		}
		// Mirror the freshly-rendered frame into the in-modal preview canvas
		// so the user can watch the build play out without needing to look
		// past the modal at the main viewport.
		drawReplayFrame(renderer.domElement);
		setReplayProgress(i / batches.length);
		// Yield to the browser so it can paint the preview frame and the user
		// can see progress in real time.
		await sleep(frameIntervalMs);
	}

	// Final hold frame so viewers see the completed scene linger for a beat
	// before the gif loops. Kept short (REPLAY_HOLD_MS) and folded into the
	// duration estimate so the gif never runs noticeably past the build.
	if (gif && !replayCancelRequested) {
		controls.update();
		renderer.render(scene, camera);
		gifScalerCtx.drawImage(
			renderer.domElement,
			0,
			0,
			gifScaler.width,
			gifScaler.height,
		);
		gif.addFrame(gifScaler, { copy: true, delay: REPLAY_HOLD_MS });
		drawReplayFrame(renderer.domElement);
	}
	setReplayProgress(1);

	let blob = null;
	if (gif && !replayCancelRequested) {
		setReplayStatus("encoding gif…", "recording");
		blob = await new Promise((resolve) => {
			gif.on("finished", resolve);
			gif.render();
		});
		setReplayStatus(
			`gif ready · ${(blob.size / 1024).toFixed(1)} KB`,
			"done",
		);
		// Swap the stage to an <img> of the encoded blob — animated gifs loop
		// by default in the browser, so the user sees the final result play.
		showReplayGifResult(blob);
	} else if (replayCancelRequested) {
		setReplayStatus("cancelled");
	} else {
		setReplayStatus("preview done", "done");
	}

	// Restore camera state.
	camera.position.copy(camSnapshot.pos);
	controls.target.copy(camSnapshot.target);
	cameraUserMoved = camSnapshot.userMoved;
	controls.update();

	replayActive = false;
	replayInProgress = false;
	replayPreloadCache = null;
	// recordedEvents has been refilled by dispatchForReplay during the loop;
	// re-enable the button to reflect that.
	updateReplayButton();

	// Reconnect to the live stream so newly-arriving events flow back into
	// the buffer. The server replays the snapshot from index 0 on reconnect,
	// which re-populates `recordedEvents` for the next gif export.
	if (
		reconnectAfter &&
		slotForReconnect !== null &&
		modelForReconnect !== null
	) {
		subscribe(slotEventsUrl(slotForReconnect, modelForReconnect));
	}

	// If the user clicked the close button while a replay was running, honor
	// that now that we've cleaned up.
	if (replayCancelRequested) {
		replayModalEl.classList.remove("open");
	}
	return blob;
}

// Sibling of dispatch() used during replay. The buffer was wiped before
// replay started; we refill it as we re-dispatch so the user can immediately
// run another preview/export without waiting for an SSE reconnect.
function dispatchForReplay(event) {
	if (typeof event.index === "number") {
		if (event.index <= highestEventIndex) return;
		highestEventIndex = event.index;
	}
	recordedEvents.push(event);
	appendEvent(event);
	switch (event.kind) {
		case "run.start":
			setStatus(`run :: ${event.model}`);
			break;
		case "run.done":
			if (meshErrors.size > 0) showRunCompleteWithErrors();
			else setStatus("run complete");
			break;
		case "run.error":
			setStatus(`error: ${event.message}`, "err");
			break;
		case "run.paused":
			setStatus("paused");
			break;
		case "mesh.error":
			meshErrors.set(event.id, event.message ?? "unknown error");
			treeSetPhase(event.id, "error");
			upsertAsset(event.id, {
				status: "error",
				errorMessage: event.message,
			});
			break;
		case "mesh.retry":
			meshErrors.delete(event.id);
			upsertAsset(event.id, { status: "pending", errorMessage: null });
			if (treeNodes.has(event.id))
				treeSetPhase(event.id, "generating_mesh");
			break;
		case "bbox":
			loadBbox(event);
			treeUpsert(event.id, {
				parentId: event.parent_id ?? null,
				prompt: event.prompt ?? null,
				kind: event.node_kind ?? "zone",
				origin: event.origin,
				dimensions: event.dimensions,
				proxyShape: event.proxy_shape ?? null,
			});
			scheduleRenderTree();
			break;
		case "divider.decompose":
		case "divider.zone_decompose":
			for (const c of event.children ?? []) {
				treeUpsert(c.id, {
					parentId: c.parent ?? event.node,
					prompt: c.prompt,
					kind: "zone",
				});
			}
			scheduleRenderTree();
			break;
		case "divider.zone_plan":
			if (event.node && typeof event.plan === "string") {
				treeUpsert(event.node, { plan: event.plan });
			}
			break;
		case "step":
			treeSetPhase(event.node, event.phase);
			break;
		case "mesh.submit":
			treeSetPhase(event.id, "generating_mesh");
			break;
		case "image":
			upsertAsset(event.id, {
				imageUrl: event.url,
				prompt: event.prompt,
			});
			if (typeof event.prompt === "string") {
				treeUpsert(event.id, { imagePrompt: event.prompt });
			}
			break;
		case "model":
			loadModel(event);
			treeSetPhase(event.id, "done");
			meshErrors.delete(event.id);
			break;
		case "cache.llm":
			recordLlmCall(event);
			break;
	}
}

function sleep(ms) {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

function lockReplayButtons(locked) {
	replayPreviewEl.disabled = locked;
	replayRenderEl.disabled = locked;
	// Download stays disabled until a successful render completes — flip it
	// off here so an in-flight render can't be re-downloaded mid-encode, but
	// don't blindly re-enable on unlock; showReplayGifResult will.
	if (locked) replayDownloadEl.disabled = true;
}

replayPreviewEl.addEventListener("click", async () => {
	if (replayInProgress) return;
	lockReplayButtons(true);
	try {
		await runReplay({ record: false });
	} finally {
		lockReplayButtons(false);
		// Re-enable download only if we still have a previously-rendered gif.
		replayDownloadEl.disabled = lastGifBlob === null;
	}
});

replayRenderEl.addEventListener("click", async () => {
	if (replayInProgress) return;
	lockReplayButtons(true);
	try {
		await runReplay({ record: true });
	} finally {
		lockReplayButtons(false);
		replayDownloadEl.disabled = lastGifBlob === null;
	}
});

replayDownloadEl.addEventListener("click", () => {
	if (!lastGifBlob) return;
	const a = document.createElement("a");
	a.href = lastGifUrl;
	const gifStem =
		currentSlotId && currentModel
			? `${currentSlotId}_${currentModel}`
			: (currentSlotId ?? "replay");
	a.download = `${gifStem}.gif`;
	a.click();
});
