// The run board: a pannable/zoomable plane of every (slot × model) cell in
// the active run, each card showing live status + the step it is on.

import { el, openModal, toast, stepUntilSelect } from "./ui.js";
import { state, emit, on, cellKey, cellSummary } from "./state.js";
import { api } from "./api.js";
import { createViewer } from "./scene3d.js";
import { applySceneProjection, createObsModel } from "./events.js";
import { renderObsTree, renderObsTrace } from "./obsmini.js";
import { statusView } from "./status.js";
import { closeOverlay } from "./overlay.js";
import { createRunCombo } from "./runcombo.js";

const boardEl = document.getElementById("board");
const planeEl = document.getElementById("board-plane");
const gridEl = document.getElementById("board-grid");
const scenesEl = document.getElementById("board-scenes");
const viewBtn = document.getElementById("btn-board-view");

const view = { x: 24, y: 24, scale: 1 };

function applyTransform() {
  planeEl.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
}

// --- pan / zoom -----------------------------------------------------------------

let panning = null;
boardEl.addEventListener("pointerdown", (ev) => {
  if (ev.button !== 0) return;
  if (ev.target.closest(".cell-card")) return; // card clicks open the overlay (or toggle selection)
  if (state.selectMode && ev.target.closest(".grid-head")) return; // header clicks select a row/column
  panning = { x: ev.clientX, y: ev.clientY };
  boardEl.classList.add("panning");
  boardEl.setPointerCapture(ev.pointerId);
});
boardEl.addEventListener("pointermove", (ev) => {
  if (!panning) return;
  view.x += ev.clientX - panning.x;
  view.y += ev.clientY - panning.y;
  panning = { x: ev.clientX, y: ev.clientY };
  applyTransform();
});
const endPan = (ev) => {
  if (!panning) return;
  panning = null;
  boardEl.classList.remove("panning");
  if (ev.pointerId !== undefined) {
    try { boardEl.releasePointerCapture(ev.pointerId); } catch { /* already released */ }
  }
};
boardEl.addEventListener("pointerup", endPan);
boardEl.addEventListener("pointercancel", endPan);

boardEl.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  const rect = boardEl.getBoundingClientRect();
  const px = ev.clientX - rect.left;
  const py = ev.clientY - rect.top;
  const prev = view.scale;
  const next = Math.min(2.2, Math.max(0.25, prev * Math.exp(-ev.deltaY * 0.0012)));
  // Zoom toward the cursor: keep the plane point under it fixed.
  view.x = px - ((px - view.x) / prev) * next;
  view.y = py - ((py - view.y) / prev) * next;
  view.scale = next;
  applyTransform();
}, { passive: false });

// --- cards ----------------------------------------------------------------------

function cellCard(slot, model) {
  const summary = slot.runs?.[model] ?? { status: "idle", events_count: 0 };
  const branches = summary.branches ?? [];
  const view = statusView(summary);
  const selecting = state.selectMode;
  const selected = state.selection.has(cellKey(slot.id, model));
  const card = el(
    "div",
    {
      class: `cell-card${summary.events_count === 0 && !branches.length ? " empty" : ""}` +
        `${selecting ? " selectable" : ""}${selected ? " selected" : ""}`,
      title: slot.prompt ?? "",
      // Select mode → toggle selection; compare-runs mode → open this cell in
      // both runs side by side; otherwise open the SOURCE scene (a cell's sim
      // branches are reached from the prompt lab).
      onclick: () => {
        if (state.selectMode) toggleSel(slot.id, model);
        else if (state.compareMode) emit("open-runcompare", { slot: slot.id, model });
        else emit("open-cell", { slot: slot.id, model, branch: null });
      },
    },
    el("div", { class: "head" },
      selecting ? el("span", { class: `sel-box${selected ? " on" : ""}`, text: selected ? "✓" : "" }) : null,
      el("span", { class: `dot ${view.dot}` }),
      el("span", { class: "slot-name", text: slot.id }),
      summary.stepped ? el("span", { class: "step-mode-tag", text: "step mode", title: "one LLM call per step — advance from the cell view or “step all”" }) : null,
    ),
    el("div", { class: "step-line", text: view.label }),
    el("div", { class: "meta-line", text: `${model} · ${summary.events_count} events · ${view.state}` }),
  );
  if (branches.length) {
    const first = branches[0];
    const bview = statusView(first);
    // Clickable: jump straight into a downstream simulation (the overlay's
    // branch selector then reaches the others) — reachable even on a done cell.
    // In select mode it folds into the card's selection toggle instead.
    card.appendChild(
      el("div", { class: "branch-chip", title: "open this cell's downstream simulation",
        onclick: (ev) => {
          ev.stopPropagation();
          if (state.selectMode) { toggleSel(slot.id, model); return; }
          if (state.compareMode) { emit("open-runcompare", { slot: slot.id, model }); return; }
          emit("open-cell", { slot: slot.id, model, branch: first.id });
        } },
        el("span", { class: `dot ${bview.dot}` }),
        el("span", { text: branches.length === 1
          ? `sim · ${bview.label}`
          : `${branches.length} sims · ${bview.state}…` }),
      ),
    );
  }
  return card;
}

export function renderBoard() {
  const models = state.models;
  gridEl.style.gridTemplateColumns = `130px repeat(${models.length}, 230px)`;
  gridEl.textContent = "";
  gridEl.appendChild(el("div", { class: "grid-head" }));
  for (const m of models) {
    gridEl.appendChild(el("div", {
      class: "grid-head col", text: m,
      title: state.selectMode ? "select / clear this model's whole column" : "",
      onclick: state.selectMode ? () => toggleColumn(m) : null,
    }));
  }
  for (const slot of state.slots) {
    gridEl.appendChild(el("div", {
      class: "grid-head", text: slot.id,
      title: state.selectMode ? "select / clear this slot's whole row" : (slot.prompt ?? ""),
      onclick: state.selectMode ? () => toggleRow(slot.id) : null,
    }));
    for (const m of models) gridEl.appendChild(cellCard(slot, m));
  }
  applyTransform();
}

// --- multi-select: pick an arbitrary set of (slot × model) cells and run
//     resume / reset / step / pause on all of them at once (vs. one-or-all). ---

const selectBtn = document.getElementById("btn-select");

function toggleSel(slotId, model) {
  const k = cellKey(slotId, model);
  if (state.selection.has(k)) state.selection.delete(k);
  else state.selection.add(k);
  emit("selection");
}

function selectedCells() {
  const out = [];
  for (const k of state.selection) {
    const i = k.lastIndexOf("|"); // slot ids may contain spaces but never "|"
    out.push({ slot: k.slice(0, i), model: k.slice(i + 1) });
  }
  return out;
}

// Toggle a set of keys as a group: if every one is already selected, clear
// them; otherwise add them all. Drives the row/column header shortcuts.
function toggleKeys(keys) {
  const allOn = keys.length > 0 && keys.every((k) => state.selection.has(k));
  for (const k of keys) { if (allOn) state.selection.delete(k); else state.selection.add(k); }
  emit("selection");
}
function toggleRow(slotId) { toggleKeys(state.models.map((m) => cellKey(slotId, m))); }
function toggleColumn(model) { toggleKeys(state.slots.map((s) => cellKey(s.id, model))); }
function selectAllCells() {
  for (const s of state.slots) for (const m of state.models) state.selection.add(cellKey(s.id, m));
  emit("selection");
}

function setSelectMode(on) {
  state.selectMode = on;
  if (on) {
    if (state.compareMode) setCompareMode(false); // the two board modes are exclusive
    if (boardMode === "scenes") setBoardMode("cards"); // cards view shows the per-cell checkboxes
  }
  if (!on) state.selection.clear();
  emit("selection");
}

if (selectBtn) selectBtn.addEventListener("click", () => setSelectMode(!state.selectMode));

// --- run-compare mode: clicking a cell opens it across two runs side by side. ---

const compareBtn = document.getElementById("btn-compare-runs");
const compareRunBSel = document.getElementById("compare-run-b");

// Run B is picked with the same searchable combobox as the topbar's active-run
// picker (runcombo.js), scoped to every run BUT the active A. Picking sets
// state.compareRunB — it does NOT switch the active run.
const compareCombo = compareRunBSel
  ? createRunCombo(compareRunBSel, {
      getRuns: () => state.runs.filter((r) => r.name !== state.run),
      getSelected: () => state.compareRunB,
      onPick: (name) => { state.compareRunB = name; renderCompareUI(); },
      placeholder: "pick run B",
    })
  : null;

function setCompareMode(on) {
  state.compareMode = on;
  if (on) {
    if (state.selectMode) setSelectMode(false); // the two board modes are exclusive
    if (boardMode === "scenes") setBoardMode("cards"); // compare opens from the cards board
  }
  renderCompareUI();
}

// Reflect the mode on the toggle + show the run-B picker (every run but the
// active A). Cheap and idempotent, so it's safe to call on every poll.
function renderCompareUI() {
  if (!compareBtn) return;
  compareBtn.classList.toggle("on", state.compareMode);
  compareRunBSel.classList.toggle("on", state.compareMode);
  // Keep the chosen B valid as run A / the run set changes.
  const others = state.runs.map((r) => r.name).filter((n) => n !== state.run);
  if (!state.compareRunB || !others.includes(state.compareRunB)) {
    state.compareRunB = others[0] ?? null;
  }
  compareCombo?.render();
}

if (compareBtn) compareBtn.addEventListener("click", () => setCompareMode(!state.compareMode));

// Run a per-cell endpoint across the whole selection in parallel; individual
// failures (resuming a done cell, stepping a non-stepped one, …) are counted as
// skipped rather than aborting the batch — mirroring "start cells" / "reset all".
async function runBulk(verb, perCell, cells = selectedCells()) {
  if (!cells.length) return;
  const results = await Promise.all(cells.map(async (c) => {
    try { await perCell(c); return true; } catch { return false; }
  }));
  const ok = results.filter(Boolean).length;
  toast(
    `${verb}: ${ok}/${results.length} cell${results.length === 1 ? "" : "s"}` +
      (ok < results.length ? " — rest skipped/failed" : ""),
    ok ? "ok" : "err",
  );
  emit("poll-now");
}

function bulkResume() {
  const stepped = steppedCheck.checked ? true : null; // null ⇒ keep each cell's current mode
  runBulk("resumed/started", (c) => api.resume(state.run, c.slot, c.model, stepped));
}
function bulkPause() { runBulk("paused", (c) => api.pause(state.run, c.slot, c.model)); }
// Override the spend cap on just the capped cells in the selection (each raises
// one band + resumes). Filtered client-side so healthy cells aren't touched.
function bulkOverride() {
  const capped = selectedCells().filter((c) => cellSummary(c.slot, c.model)?.status === "capped");
  if (!capped.length) { toast("no capped cells in the selection", "err"); return; }
  runBulk("caps overridden", (c) => api.capOverride(state.run, c.slot, c.model), capped);
}
function bulkStep(until, before = false) {
  runBulk(until ? `stepping → ${before ? "before " : ""}${until}` : "stepped",
    (c) => api.cellStep(state.run, c.slot, c.model, { until: until || null, untilBefore: before }));
}
function bulkReset(start) {
  const cells = selectedCells();
  if (!cells.length) return;
  const n = cells.length;
  openModal(`reset ${n} cell${n === 1 ? "" : "s"}${start ? " & start" : ""}?`, (close) => ({
    body: [
      el("div", { class: "m-hint", text:
        `Permanently deletes the events, generated meshes, and simulation branches of ${n} selected cell${n === 1 ? "" : "s"}` +
        (start ? ", then starts each one fresh." : " and leaves them idle.") }),
    ],
    actions: [
      el("button", { text: "cancel", onclick: close }),
      el("button", { class: "danger", text: `reset${start ? " & start" : ""} ${n}`, onclick: () => {
        close();
        closeOverlay(); // a viewed cell may be among those wiped
        runBulk(start ? "reset & started" : "reset to idle", (c) => api.reset(state.run, c.slot, c.model, start));
      } }),
    ],
  }));
}

// The contextual action bar, built once (so the "stepped" toggle + "until"
// picker keep their state across board re-renders) and shown only in select mode.
const steppedCheck = el("input", { type: "checkbox" });
const bulkCount = el("span", { class: "bulk-count" });
const untilSel = stepUntilSelect(
  () => state.steps,
  (until, before) => bulkStep(until, before),
  { label: "step until…", title: "run every selected cell up to the next call of a step — pause before or after it" },
);
const bulkActionEls = [
  el("button", { class: "primary", title: "start idle / resume paused selected cells", onclick: bulkResume }, "resume / start"),
  el("label", { class: "bulk-stepped", title: "resume/start in step mode (pause before each LLM call)" }, steppedCheck, "stepped"),
  el("button", { title: "pause selected running cells", onclick: bulkPause }, "pause"),
  el("button", { title: "advance each selected cell by one LLM call", onclick: () => bulkStep(null) }, "step"),
  untilSel,
  el("button", { title: "override the spend cap on selected capped cells & resume them", onclick: bulkOverride }, "override caps"),
  el("button", { class: "danger", title: "wipe selected cells back to idle", onclick: () => bulkReset(false) }, "reset → idle"),
  el("button", { class: "danger", title: "wipe selected cells and start them fresh", onclick: () => bulkReset(true) }, "reset & start"),
];
const bulkBar = el("div", { id: "bulk-bar" },
  el("span", { class: "bulk-check", text: "☑" }),
  bulkCount,
  el("button", { class: "bulk-mini", title: "select every cell on this run", onclick: selectAllCells }, "all"),
  el("button", { class: "bulk-mini", title: "clear the selection", onclick: () => { state.selection.clear(); emit("selection"); } }, "none"),
  el("span", { class: "bulk-sep" }),
  ...bulkActionEls,
  el("span", { style: "margin-left:auto" }),
  el("button", { class: "bulk-done", title: "leave select mode", onclick: () => setSelectMode(false) }, "done"),
);
document.body.appendChild(bulkBar);
// The controls that act on the selection — disabled while nothing is selected.
const bulkNeedsSelection = bulkActionEls.filter((n) => n.tagName === "BUTTON" || n.tagName === "SELECT");

function renderSelectionUI() {
  document.body.classList.toggle("select-mode", state.selectMode);
  if (selectBtn) {
    selectBtn.classList.toggle("on", state.selectMode);
    selectBtn.textContent = state.selectMode ? "exit select" : "select";
  }
  bulkBar.classList.toggle("open", state.selectMode);
  const n = state.selection.size;
  bulkCount.textContent = `${n} selected`;
  for (const node of bulkNeedsSelection) node.disabled = n === 0;
  steppedCheck.disabled = n === 0;
}

// --- scenes view: a grid of live 3D canvases, one per active run -----------------
//
// One viewer per active cell, created lazily as its tile scrolls into view
// (and paused when it leaves) so a run with many started cells doesn't open
// dozens of WebGL contexts at once. Each tile's scene reloads as the cell
// makes progress, throttled so we don't re-pull every mesh on every poll.

let boardMode = (() => {
  try { return localStorage.getItem("starshot.boardMode") === "scenes" ? "scenes" : "cards"; }
  catch { return "cards"; }
})();
const sceneTiles = new Map(); // cellKey -> tile ref
let scenesRun = null;         // run the current tiles belong to (keys repeat across runs)
// Per-run curation of the grid: `hidden` cells are removed from it; `extra`
// cells are added beyond the default (every active cell). Persisted so the
// curated set survives reloads.
let sceneHidden = new Set();
let sceneExtra = new Set();
const SCENE_RELOAD_MS = 6000;
const SCENE_GAP = 14;         // must match #board-scenes gap
const SCENE_PAD = 16;         // must match #board-scenes padding
const SCENE_HEADER = 31;      // approx .scene-tile-head height
const SCENE_TARGET_ASPECT = 1.5; // mild landscape preference for the canvas

// Choose the column count that makes every tile as large as possible while
// fitting all `n` in the W×H area — so the whole set is always on-screen,
// scaled to fill, no scroll.
function bestColumns(n, W, H) {
  let best = 1;
  let bestScore = -Infinity;
  for (let c = 1; c <= n; c++) {
    const r = Math.ceil(n / c);
    const cellW = (W - (c - 1) * SCENE_GAP) / c;
    const cellH = (H - (r - 1) * SCENE_GAP) / r;
    const canvasH = cellH - SCENE_HEADER;
    if (cellW <= 20 || canvasH <= 20) continue;
    const aspect = cellW / canvasH;
    const fit = Math.min(aspect, SCENE_TARGET_ASPECT) / Math.max(aspect, SCENE_TARGET_ASPECT);
    const score = cellW * canvasH * (0.45 + 0.55 * fit); // area, mildly favoring a landscape canvas
    if (score > bestScore) { bestScore = score; best = c; }
  }
  return best;
}

function layoutScenes() {
  const n = sceneTiles.size;
  if (!n) { scenesEl.style.gridTemplateColumns = "1fr"; scenesEl.style.gridTemplateRows = ""; return; }
  const W = scenesEl.clientWidth - 2 * SCENE_PAD;
  const H = scenesEl.clientHeight - 2 * SCENE_PAD;
  if (W <= 0 || H <= 0) return;
  const c = bestColumns(n, W, H);
  scenesEl.style.gridTemplateColumns = `repeat(${c}, 1fr)`;
  scenesEl.style.gridTemplateRows = `repeat(${Math.ceil(n / c)}, 1fr)`;
}

let layoutRaf = 0;
window.addEventListener("resize", () => {
  if (boardMode !== "scenes") return;
  cancelAnimationFrame(layoutRaf);
  layoutRaf = requestAnimationFrame(layoutScenes);
});

// --- right-side observability drawer (opened from a tile's mini pipeline) --------

const drawerEl = document.getElementById("obs-drawer");
const drawerBodyEl = document.getElementById("obs-drawer-body");
const drawerTitleEl = document.getElementById("obs-drawer-title");
const drawerSubEl = document.getElementById("obs-drawer-sub");
const drawer = { key: null, slot: null, model: null, focusId: null }; // cell + focused object
const drawerExpanded = new Set();   // call indices expanded to their bytes

function drawerToggle(call) {
  if (drawerExpanded.has(call.index)) drawerExpanded.delete(call.index);
  else drawerExpanded.add(call.index);
  renderDrawer();
}

function renderDrawer() {
  const ref = drawer.key ? sceneTiles.get(drawer.key) : null;
  const model = ref?.obsModel ?? null;
  // Trace mode: a node was clicked — show its lineage + the calls that emitted it.
  if (drawer.focusId && model && model.nodes.has(drawer.focusId)) {
    renderObsTrace(drawerBodyEl, model, drawer.focusId, {
      expanded: drawerExpanded,
      onToggle: drawerToggle,
      onRevert: revertDrawerCall,
      onNavigate: (id) => { drawer.focusId = id; renderDrawer(); },
      onBack: () => { drawer.focusId = null; renderDrawer(); },
    });
    return;
  }
  drawer.focusId = null;
  renderObsTree(drawerBodyEl, model, {
    detailed: true,
    expanded: drawerExpanded,
    onToggle: drawerToggle,
    onRevert: revertDrawerCall,
    onNodeClick: (id) => { drawer.focusId = id; renderDrawer(); },
  });
}

function openObsDrawer(slot, model) {
  drawer.key = cellKey(slot, model);
  drawer.slot = slot;
  drawer.model = model;
  drawer.focusId = null;
  drawerExpanded.clear();
  drawerTitleEl.textContent = slot;
  drawerSubEl.textContent = model;
  drawerEl.classList.add("open");
  renderDrawer();
}

// Revert the drawer's cell to just before `call` — confirm, truncate the log,
// then re-run the pipeline from the cut. The tile's scene + obs reload at the
// cut and the polling loop refreshes them as the re-run progresses.
function revertDrawerCall(call) {
  const slot = drawer.slot;
  const model = drawer.model;
  if (!slot) return;
  const step = call.template ?? call.step ?? "this step";
  openModal(`revert ${slot} · ${model}?`, (close, setError) => ({
    body: [
      el("div", { class: "m-hint", text:
        `Truncates this slot's log to just before its ${step} call (#${call.index}), ` +
        "drops every later step and its meshes, then re-runs the pipeline from there." }),
    ],
    actions: [
      el("button", { text: "cancel", onclick: close }),
      el("button", { class: "danger", text: "revert & re-run", onclick: async () => {
        try { await api.rewind(state.run, slot, model, call.index); }
        catch (e) { setError(e.message); return; }
        close();
        toast(`reverted ${slot} · ${model} to #${call.index}`, "ok");
        const ref = sceneTiles.get(cellKey(slot, model));
        if (ref) { ref.loadedCount = -1; loadTile(ref, slot, model); } // refresh scene + obs + drawer
        emit("poll-now");
      } }),
    ],
  }));
}

function closeObsDrawer() {
  drawerEl.classList.remove("open");
  drawer.key = null;
  drawer.focusId = null;
}

document.getElementById("obs-drawer-close").addEventListener("click", closeObsDrawer);
document.addEventListener("keydown", (ev) => {
  if (ev.key !== "Escape" || !drawerEl.classList.contains("open")) return;
  if (document.getElementById("overlay").classList.contains("open")) return;
  if (document.getElementById("modal-root").firstChild) return;
  closeObsDrawer();
});
// Opening a full cell overlay covers the drawer — close it so it doesn't linger.
on("open-cell", closeObsDrawer);

// The cells the grid shows: every active (started) cell by default, minus the
// ones the user removed, plus any they explicitly added.
function displayedCells() {
  const out = [];
  for (const slot of state.slots) {
    for (const m of state.models) {
      const key = cellKey(slot.id, m);
      if (sceneHidden.has(key)) continue;
      const s = slot.runs?.[m];
      const active = !!(s && s.status && s.status !== "idle");
      if (active || sceneExtra.has(key)) out.push({ slot, model: m, summary: s ?? { status: "idle", events_count: 0 } });
    }
  }
  return out;
}

function curationKey(run) { return `starshot.sceneCuration.${run}`; }

function loadCuration(run) {
  sceneHidden = new Set();
  sceneExtra = new Set();
  try {
    const v = JSON.parse(localStorage.getItem(curationKey(run)) || "null");
    if (v) { sceneHidden = new Set(v.hidden || []); sceneExtra = new Set(v.extra || []); }
  } catch { /* corrupt entry — default to no curation */ }
}

function saveCuration() {
  if (!scenesRun) return;
  try {
    localStorage.setItem(curationKey(scenesRun), JSON.stringify({ hidden: [...sceneHidden], extra: [...sceneExtra] }));
  } catch { /* private mode */ }
}

function removeSceneCell(slot, model) {
  const key = cellKey(slot, model);
  sceneHidden.add(key);
  sceneExtra.delete(key);
  saveCuration();
  reconcileScenes();
}

function addSceneCell(slot, model) {
  const key = cellKey(slot, model);
  sceneHidden.delete(key);
  sceneExtra.add(key);
  saveCuration();
  reconcileScenes();
}

// Pick cells NOT currently in the grid (removed ones + never-started ones) to
// add. Clicking a row adds it and drops it from the list; the grid updates live.
function addSceneCellPicker() {
  const shown = new Set(displayedCells().map((c) => cellKey(c.slot.id, c.model)));
  const rows = [];
  for (const slot of state.slots) {
    for (const m of state.models) {
      const key = cellKey(slot.id, m);
      if (shown.has(key)) continue;
      const status = slot.runs?.[m]?.status ?? "idle";
      rows.push({ slot, model: m, status });
    }
  }
  openModal("add slots to the scene grid", (close) => {
    const list = el("div", { class: "scene-add-list" });
    const fill = () => {
      list.textContent = "";
      const stillShown = new Set(displayedCells().map((c) => cellKey(c.slot.id, c.model)));
      const remaining = rows.filter((r) => !stillShown.has(cellKey(r.slot.id, r.model)));
      if (!remaining.length) { list.appendChild(el("div", { class: "muted", style: "padding:8px", text: "every slot is already in the grid." })); return; }
      for (const r of remaining) {
        list.appendChild(el("button", { class: "scene-add-row", onclick: () => { addSceneCell(r.slot.id, r.model); fill(); } },
          el("span", { class: `dot ${r.status}` }),
          el("span", { class: "scene-add-name", text: r.slot.id }),
          el("span", { class: "scene-add-model", text: r.model }),
          el("span", { class: "muted", text: r.status }),
        ));
      }
    };
    fill();
    return {
      body: [ el("div", { class: "m-hint", text: "Add a slot's 3D scene to the grid. Removed and never-started slots show here." }), list ],
      actions: [ el("button", { class: "primary", text: "done", onclick: close }) ],
    };
  });
}

function loadTile(ref, slot, model) {
  loadTileScene(ref, slot, model);
  loadTileObs(ref, slot, model);
}

function loadTileScene(ref, slot, model) {
  const viewer = ref.viewer;
  if (!viewer) return;
  api.scene(state.run, slot, model, {}).then((proj) => {
    if (ref.viewer !== viewer) return; // disposed / reloaded meanwhile
    viewer.clear();
    applySceneProjection(viewer, proj);
    viewer.prefetchBundle(api.meshesUrl(state.run, slot, model, {}));
  }).catch(() => { /* non-fatal — leave the tile empty */ });
}

// Fold the cell's full event log into an obs model for the tile's mini
// pipeline (and the drawer, when it's showing this cell).
function loadTileObs(ref, slot, model) {
  const key = cellKey(slot, model);
  api.eventsHistory(state.run, slot, model, {}).then((events) => {
    if (!sceneTiles.has(key) || sceneTiles.get(key) !== ref) return; // torn down
    const m = createObsModel();
    for (const e of events) m.feed(e);
    ref.obsModel = m.model;
    renderObsTree(ref.obsBody, ref.obsModel, { detailed: false });
    if (drawer.key === key) renderDrawer();
  }).catch(() => { /* never-started cell / fetch race — leave the mini empty */ });
}

function makeSceneTile(slot, model) {
  const host = el("div", { class: "scene-tile-host" });
  const dot = el("span", { class: "dot idle" });
  const status = el("span", { class: "scene-tile-status" });
  const head = el("div", { class: "scene-tile-head", title: "open this run",
    onclick: () => emit("open-cell", { slot: slot.id, model, branch: false }) },
    dot,
    el("span", { class: "scene-tile-name", text: slot.id }),
    el("span", { class: "scene-tile-model", text: model }),
    status,
    el("button", { class: "scene-tile-x", text: "✕", title: "remove from the scene grid",
      onclick: (ev) => { ev.stopPropagation(); removeSceneCell(slot.id, model); } }),
  );
  // Mini observability pipeline overlaid on the canvas; click → right drawer.
  const obsBody = el("div", { class: "obsm-mini-body" });
  const obsPanel = el("div", { class: "obsm-mini", title: "open full observability",
    onclick: (ev) => { ev.stopPropagation(); openObsDrawer(slot.id, model); } },
    el("div", { class: "obsm-mini-head" }, el("span", { text: "pipeline ⤢" })),
    obsBody,
  );
  host.appendChild(obsPanel);
  const tile = el("div", { class: "scene-tile", title: slot.prompt ?? "" }, head, host);
  const ref = {
    tile, host, viewer: null, io: null, loadedCount: -1, lastLoad: 0,
    obsBody, obsModel: null,
    setStatus: (s) => { const v = statusView(s); dot.className = `dot ${v.dot}`; status.textContent = v.label; },
  };
  ref.io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        if (!ref.viewer) {
          ref.viewer = createViewer(host, { keyboard: false });
          ref.loadedCount = state.slots.find((x) => x.id === slot.id)?.runs?.[model]?.events_count ?? 0;
          ref.lastLoad = performance.now();
          loadTile(ref, slot.id, model);
        }
        ref.viewer.setActive(true);
      } else {
        ref.viewer?.setActive(false);
      }
    }
  }, { threshold: 0.05 });
  ref.io.observe(host);
  return ref;
}

function teardownTile(ref) {
  if (ref.io) { ref.io.disconnect(); ref.io = null; }
  if (ref.viewer) { ref.viewer.dispose(); ref.viewer = null; }
}

function clearScenes() {
  for (const ref of sceneTiles.values()) teardownTile(ref);
  sceneTiles.clear();
  scenesEl.textContent = "";
  closeObsDrawer();
}

function reconcileScenes() {
  // Cell keys (slot|model) repeat across runs, so a run switch must drop the
  // old run's tiles AND load that run's curated set.
  if (scenesRun !== state.run) { clearScenes(); scenesRun = state.run; loadCuration(state.run); }
  const cells = displayedCells();
  const wanted = new Set(cells.map((c) => cellKey(c.slot.id, c.model)));
  for (const [key, ref] of [...sceneTiles]) {
    if (!wanted.has(key)) {
      teardownTile(ref); ref.tile.remove(); sceneTiles.delete(key);
      if (drawer.key === key) closeObsDrawer();
    }
  }
  const empty = scenesEl.querySelector(".scene-empty");
  if (!cells.length) {
    if (!empty) scenesEl.appendChild(el("div", { class: "scene-empty", text: "no slots in the grid — click “+ add slot” to add some." }));
    layoutScenes();
    return;
  }
  empty?.remove();
  const now = performance.now();
  for (const { slot, model, summary } of cells) {
    const key = cellKey(slot.id, model);
    let ref = sceneTiles.get(key);
    if (!ref) {
      ref = makeSceneTile(slot, model);
      sceneTiles.set(key, ref);
    }
    scenesEl.appendChild(ref.tile); // (re)append keeps grid in slot/model order
    ref.setStatus(summary);
    // Refresh a live tile's scene as the cell progresses, throttled.
    const count = summary.events_count ?? 0;
    if (ref.viewer && count !== ref.loadedCount && now - ref.lastLoad > SCENE_RELOAD_MS) {
      ref.loadedCount = count;
      ref.lastLoad = now;
      loadTile(ref, slot.id, model);
    }
  }
  // Size the grid so every tile fits the area at once, scaled to fill.
  layoutScenes();
}

const sceneAddFab = document.getElementById("scene-add-fab");
sceneAddFab.addEventListener("click", addSceneCellPicker);

function applyBoardMode() {
  const scenes = boardMode === "scenes";
  boardEl.style.display = scenes ? "none" : "";
  scenesEl.classList.toggle("open", scenes);
  sceneAddFab.classList.toggle("open", scenes);
  viewBtn.textContent = scenes ? "cards ▦" : "scenes ▦";
  if (scenes) reconcileScenes();
  else { clearScenes(); renderBoard(); }
}

function setBoardMode(mode) {
  boardMode = mode;
  try { localStorage.setItem("starshot.boardMode", mode); } catch { /* private mode */ }
  applyBoardMode();
}

viewBtn.addEventListener("click", () => setBoardMode(boardMode === "scenes" ? "cards" : "scenes"));

on("slots", () => { if (boardMode === "scenes") reconcileScenes(); else renderBoard(); renderSelectionUI(); renderCompareUI(); });
on("selection", () => { if (boardMode !== "scenes") renderBoard(); renderSelectionUI(); });
applyTransform();
applyBoardMode();
renderSelectionUI();
renderCompareUI();
