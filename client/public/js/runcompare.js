// Run-compare: the SAME (slot × model) cell viewed across TWO runs side by side.
// A = the active board run, B = state.compareRunB. Read-only — each pane owns an
// independent 3D viewer (its own orbit/zoom/toggles/hide) AND a full
// observability dock (node tree, calls→exact bytes, emit-trace, "why?", log
// strip), so the comparison keeps every inspection affordance the single-cell
// overlay has, just twice. The slot/model grid is run-independent, so the cell
// exists in both runs (B may simply be empty if that run never ran it).
//
// Cells are read straight from each run's logged artifacts (the API read path
// is run-parameterized), so opening this never touches the active run or the
// pipeline — it's pure visualization of two finished (or in-progress) runs.

import { api } from "./api.js";
import { state, on } from "./state.js";
import { el, toast, buildObjectsMenu } from "./ui.js";
import { createViewer } from "./scene3d.js";
import { createObsDock } from "./obstree.js";
import { applySceneProjection, createObsModel, emittedStep } from "./events.js";
import { statusView } from "./status.js";
import * as inquiry from "./inquiry.js";

let root = null;
let titleEl = null;
let subEl = null;
let paneA = null;
let paneB = null;

let built = false;
let openSeq = 0;
let openTarget = null; // {slot, model} currently shown
let ready = false;     // suppress camera sync until both scenes have settled
let linked = true;
let syncing = false;   // re-entrancy guard for the A<->B camera copy

// Per-side visibility layers, overlaid on each canvas (mirrors the compare view).
// Objects (anchors / next / negative space / frames) are the multiselect
// dropdown prepended below; these are the remaining plain on/off layers.
const RC_TOGGLES = [
  ["zones", "zones"],
  ["meshes", "meshes"],
  ["grid", "grid"],
  ["bboxes", "bboxes"],
  ["proxies", "proxies"],
];

function buildToggleBar(host, viewer) {
  const bar = el("div", { class: "cmp-toggles" });
  bar.appendChild(buildObjectsMenu(viewer));
  for (const [key, label] of RC_TOGGLES) {
    const btn = el("button", {
      class: viewer.toggles[key] ? "" : "off",
      text: label,
      title: `toggle ${label} on this side`,
      onclick: () => {
        viewer.toggles[key] = !viewer.toggles[key];
        viewer.refreshVisibility();
        btn.classList.toggle("off", !viewer.toggles[key]);
      },
    });
    bar.appendChild(btn);
  }
  host.appendChild(bar);
}

// Build one pane: a viewer + a full obs dock, wired together exactly like the
// overlay (node↔3D linking, hide, color-by-origin, "why?"), but read-only — no
// per-call revert, no "+ sim", no pinned panel. The viewer is keyboard:false so
// the global WASD fly-keys don't drive BOTH panes at once; mouse orbit / zoom
// (per-canvas) keeps each independently controllable.
function buildPane(prefix) {
  const host = document.getElementById(`${prefix}-host`);
  const obsHost = document.getElementById(`${prefix}-obs`);
  const viewer = createViewer(host, { keyboard: false });
  const dock = createObsDock(obsHost);
  const pane = {
    run: null,
    hasScene: false,
    viewer,
    dock,
    obs: createObsModel(),
    host,
    obsHost,
    runBadgeEl: document.getElementById(`${prefix}-run`),
    dotEl: document.getElementById(`${prefix}-dot`),
    statusEl: document.getElementById(`${prefix}-status`),
    obsToggleBtn: document.getElementById(`${prefix}-obs-toggle`),
  };
  // The viewer reads node info + origin color from THIS pane's obs model; the
  // model object is swapped on each open, so the closures read pane.obs live.
  viewer.setNodeInfo((id) => pane.obs.model.nodes.get(id) ?? null);
  viewer.setOriginOf((id) => emittedStep(pane.obs.model, id));
  viewer.onSelect((id) => dock.markSelected(id, { scroll: true }));
  dock.setOnNodeClick((id, { ensureSelected = false } = {}) => {
    if (!viewer.hasBbox(id)) return;
    if (ensureSelected && viewer.getSelected() === id) return;
    viewer.select(id, { frame: true });
  });
  dock.setHiddenApi({ isHidden: viewer.isHidden, toggle: viewer.toggleHidden });
  viewer.onHiddenChange(() => dock.renderTree(pane.obs.model));
  // "why?" opens the decision-inquiry chat grounded in THIS pane's run/cell
  // (read-only — it just continues the step's own conversation).
  dock.setOnInquire((call) => {
    if (!openTarget) return;
    inquiry.openInquiry(call, { run: pane.run, slot: openTarget.slot, model: openTarget.model, branch: null });
  });
  dock.setOnRevert(null);
  dock.setOnAddSim(null);
  dock.setPinStep(null);
  buildToggleBar(host, viewer);
  pane.obsToggleBtn.addEventListener("click", () => {
    const open = pane.obsHost.classList.toggle("open");
    pane.obsToggleBtn.textContent = open ? "obs ▴" : "obs ▾";
  });
  return pane;
}

function ensure() {
  if (built) return;
  built = true;
  paneA = buildPane("rc-a");
  paneB = buildPane("rc-b");
  // Keep both cameras locked to the same vantage for an honest spatial A/B,
  // unless the user unticks "link cameras".
  const link = (from, to) =>
    from.viewer.onCameraChange(() => {
      if (!ready || !linked || syncing) return;
      syncing = true;
      to.viewer.setView(from.viewer.getView());
      syncing = false;
    });
  link(paneA, paneB);
  link(paneB, paneA);
}

function renderPaneHeader(pane, run, history) {
  pane.runBadgeEl.textContent = run;
  pane.runBadgeEl.title = run;
  if (!history.length && !pane.hasScene) {
    pane.dotEl.className = "dot idle";
    pane.statusEl.textContent = "no data in this run";
    pane.statusEl.title = `${run} has no events for this cell`;
    return;
  }
  const view = statusView(deriveStatus(history));
  pane.dotEl.className = `dot ${view.dot}`;
  const label = `${view.label} · ${history.length} events`;
  pane.statusEl.textContent = label;
  pane.statusEl.title = label;
}

// A read-only run has no live summary to poll, so derive its terminal state
// from the last lifecycle marker in the committed log (the server folds the
// same markers into the status it serves for the active run).
function deriveStatus(history) {
  let last = null;
  for (const e of history) {
    if (typeof e.kind === "string" && e.kind.startsWith("run.")) last = e.kind;
  }
  switch (last) {
    case "run.done": return { status: "done" };
    case "run.error": return { status: "error" };
    case "run.paused": return { status: "paused" };
    case "run.start": return { status: "running" }; // started, no terminal logged
    default: return { status: history.length ? "done" : "idle" };
  }
}

// Load one run's cell into a pane: paint its scene, stream its mesh bundle, fold
// its event log into the obs model + dock, and render the status header.
async function loadPane(pane, run, slot, model, seq) {
  const { viewer, dock } = pane;
  pane.run = run;
  pane.hasScene = false;
  viewer.setActive(true);
  viewer.clear();
  pane.obs = createObsModel();
  dock.resetDock();
  pane.runBadgeEl.textContent = run;
  pane.runBadgeEl.title = run;
  pane.dotEl.className = "dot idle";
  pane.statusEl.textContent = "loading…";

  let proj = { nodes: [], last_index: -1 };
  try {
    proj = await api.scene(run, slot, model, {});
  } catch (e) {
    if (seq === openSeq) toast(`${run}: scene load failed: ${e.message}`, "err");
  }
  if (seq !== openSeq) return;
  const nodes = proj.nodes ?? [];
  pane.hasScene = nodes.length > 0;
  applySceneProjection(viewer, proj);
  if (nodes.some((n) => n.mesh_url)) viewer.prefetchBundle(api.meshesUrl(run, slot, model, {}));

  let history = [];
  try {
    history = await api.eventsHistory(run, slot, model);
  } catch { /* never-started cell in this run */ }
  if (seq !== openSeq) return;
  for (const e of history) pane.obs.feed(e);
  viewer.recolorAll();
  dock.renderTree(pane.obs.model);
  renderPaneHeader(pane, run, history);
}

async function openRunCompare({ slot, model }) {
  const runA = state.run;
  const runB = state.compareRunB;
  if (!runA) return;
  if (!runB) { toast("pick a second run (B) to compare against", "err"); return; }
  ensure();
  const seq = ++openSeq;
  openTarget = { slot, model };
  ready = false;
  inquiry.closeInquiry();
  titleEl.textContent = `${slot} · ${model}`;
  subEl.textContent = `${runA}  ↔  ${runB}`;
  root.classList.add("open");
  paneA.viewer.setActive(true);
  paneB.viewer.setActive(true);
  paneA.statusEl.textContent = paneB.statusEl.textContent = "loading…";
  // Run B (and A, defensively) must be loaded server-side before /scene +
  // /meshes will serve them; idempotent + non-activating, so run A's board is
  // never disturbed. Event history reads from disk regardless, so a failure
  // here just degrades the 3D paint, not the observability.
  try {
    await Promise.all([api.hydrateRun(runA), api.hydrateRun(runB)]);
  } catch { /* surfaced per-pane by the scene load below */ }
  if (seq !== openSeq) return;
  await Promise.all([
    loadPane(paneA, runA, slot, model, seq),
    loadPane(paneB, runB, slot, model, seq),
  ]);
  if (seq !== openSeq) return;
  // Frame each scene, then lock B to A's vantage (or A to B's if A is empty),
  // and only then arm the live camera sync so the framing doesn't fight it.
  setTimeout(() => {
    if (seq !== openSeq) return;
    paneA.viewer.fit();
    paneB.viewer.fit();
    if (linked) {
      if (paneA.hasScene) paneB.viewer.setView(paneA.viewer.getView());
      else if (paneB.hasScene) paneA.viewer.setView(paneB.viewer.getView());
    }
    ready = true;
  }, 500);
}

function closeRunCompare() {
  // Drop the view; the viewers persist (deactivated) so reopening doesn't churn
  // WebGL contexts.
  openSeq += 1;
  openTarget = null;
  ready = false;
  root.classList.remove("open");
  paneA?.viewer.setActive(false);
  paneB?.viewer.setActive(false);
  paneA?.viewer.clear();
  paneB?.viewer.clear();
  inquiry.closeInquiry();
}

export function initRunCompare() {
  root = document.getElementById("runcompare");
  titleEl = document.getElementById("runcompare-title");
  subEl = document.getElementById("runcompare-sub");
  document.getElementById("runcompare-close").addEventListener("click", closeRunCompare);
  document.getElementById("runcompare-sync").addEventListener("click", () => {
    if (paneA && paneB) paneB.viewer.setView(paneA.viewer.getView());
  });
  const linkCb = document.getElementById("runcompare-link");
  linked = linkCb.checked;
  linkCb.addEventListener("change", () => { linked = linkCb.checked; });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && root.classList.contains("open") && !document.getElementById("modal-root").firstChild) {
      closeRunCompare();
    }
  });
  on("open-runcompare", openRunCompare);
}
