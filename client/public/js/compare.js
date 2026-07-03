// Side-by-side 3D compare: the cell's live run (PREVIOUS) against a forest of
// simulation LINEAGES (CURRENT), with the two cameras optionally locked and a
// text panel showing the input diff + both outputs. Launched from a review
// card's / sims-list "compare 3D" button.
//
// Parallel LLMs are PERSISTENT branches, one lineage per LLM. Two independent
// selectors drive the simulation side:
//   * VIEW   — which lineage's scene/text shows on the CURRENT pane.
//   * NEXT   — which LLM(s) the next "step ▶" runs.
// "step ▶" advances ONLY the selected LLMs: an existing lineage continues on
// its own model (gemini→gemini), and a newly-selected LLM forks a fresh lineage
// from the ORIGINAL run at the current depth (so a step run on an LLM that
// didn't run the prior step uses the original's prior output as context). No
// canonical pick — every lineage lives and grows; the lineages are real sims
// (also visible / manageable in the prompt lab).

import { api } from "./api.js";
import { state, on, emit, cellSummary, cellBranches, branchSummaryById } from "./state.js";
import { el, toast, diffPre, fmtJson } from "./ui.js";
import { createViewer } from "./scene3d.js";
import { applySceneProjection, createObsModel, emittedStep } from "./events.js";
import { renderObsTree } from "./obsmini.js";
import { overridesPayload } from "./promptlab.js";
import { statusView } from "./status.js";

const root = document.getElementById("compare");
const titleEl = document.getElementById("compare-title");
const subEl = document.getElementById("compare-sub");
const textEl = document.getElementById("compare-text");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let prevViewer = null;
let curViewer = null;
let linked = true;
let ready = false; // suppress camera sync until both scenes have settled
let syncing = false; // re-entrancy guard for the A<->B copy
let openSeq = 0;
// The cell currently shown, so the slots poll can re-paint both sides as they
// step — keeping the original (previous) side in lockstep with the simulation.
let openTarget = null;
let prevSig = null;
let curSig = null;
let lastPrevPaint = 0;
let lastCurPaint = 0;
const CMP_RELOAD_MS = 3500;

// --- PREVIOUS (original) side: scrub the source cell's committed log ---------
// Projects the source scene at any past step via `?until_index`; nothing
// re-runs. `origAuto` tracks the source's latest committed scene; scrubbing
// back pins it to a step until "live".
let origSteps = []; // ordered cache.llm events of the original's committed log
let origPtr = 0;    // scrub position into origSteps (only when !origAuto)
let origAuto = false;
let forkIndex = null; // the fork event index in the source log — PREVIOUS opens here
let forkPtr = -1;     // origSteps position of the fork step
let origInit = false; // has the scrubber been anchored at the fork yet?
let branchSteps = []; // ordered cache.llm events of the VIEWED lineage
let prevLabel = null;
let curLabel = null;
let curStepBtn = null; // simulation "step ▶" / "run rest" — locked while in flight
let curRunBtn = null;

// --- CURRENT side: the per-LLM lineage forest --------------------------------
// `lineages`: alias -> { llm, branch, depth }. `viewLLM` is the lineage shown
// in 3D + text; `selectedModels` is the next-step set; `round` is how many
// "step ▶" presses past the fork we've walked (a new LLM forks from the source
// at the round-th committed step, so its prior steps are the original's).
let lineages = new Map();
let viewLLM = null;
let selectedModels = new Set();
let round = 0;
let modelBarEl = null;   // the "run next step on" multi-select chips (NEXT)
let lineageBarEl = null; // the per-lineage "view" chips (VIEW)
let viewStep = null;
let viewNode = null;
let simBusy = false; // a step/revert is orchestrating — controls + poll back off
let curHold = 0;     // brief window after an eager repaint where the poll won't touch CURRENT

// Each side keeps its OWN observability tree (folded from its event log).
let prevObs = createObsModel();
let curObs = createObsModel();
const prevExpanded = new Set();
const curExpanded = new Set();
let prevTreeEl = null;
let curTreeEl = null;

function curLineage() {
  return viewLLM ? (lineages.get(viewLLM) ?? null) : null;
}

// The obs model the CURRENT viewer reads object-origin colors from — the viewed
// lineage's own provenance, so object coloring follows whichever lineage is on.
function curOriginModel() {
  return curObs.model;
}

function renderSideTree(treeEl, obs, expanded, onRevert = null) {
  if (!treeEl || !treeEl.classList.contains("open")) return; // only when visible
  renderObsTree(treeEl, obs.model, {
    detailed: true,
    expanded,
    onRevert, // per-call ⏪ — wired only for the simulation (current) side
    onToggle: (call) => {
      if (expanded.has(call.index)) expanded.delete(call.index);
      else expanded.add(call.index);
      renderSideTree(treeEl, obs, expanded, onRevert);
    },
  });
}

// "<step> · <zone>" for a logged cache.llm event — the observability the
// per-side labels surface.
function stepDesc(s) {
  if (!s) return "?";
  const name = s.template ?? s.step ?? "?";
  return s.node ? `${name} · ${s.node}` : name;
}

function cellSig(slot, model) {
  const s = cellSummary(slot, model);
  const lin = curLineage();
  const b = lin ? branchSummaryById(lin.branch) : null;
  return {
    prev: String(s?.events_count ?? ""),
    srcRunning: s?.status === "running",
    cur: b ? `${b.events_count ?? ""}|${b.status ?? ""}|${b.pending?.step ?? ""}|${b.pending?.node ?? ""}` : "none",
    running: b?.status === "running" && !b?.pending,
  };
}

// Re-paint whichever side advanced. Source progress repaints PREVIOUS; the
// VIEWED lineage's progress repaints CURRENT. Each is throttled only while its
// cell auto-runs, so a live "run rest" doesn't re-pull a whole mesh bundle
// every poll. The lineage bar (every lineage's status) refreshes each poll.
function refreshOpenCompare() {
  if (!root.classList.contains("open") || !openTarget) return;
  const { slot, model } = openTarget;
  const sig = cellSig(slot, model);
  const now = performance.now();
  const seq = openSeq;
  if (sig.prev !== prevSig) {
    prevSig = sig.prev;
    if (!(sig.srcRunning && now - lastPrevPaint < CMP_RELOAD_MS)) {
      lastPrevPaint = now;
      loadOrigSteps(seq);
      if (origAuto) paint(prevViewer, slot, model, {}, seq);
    }
  }
  // CURRENT is ours while we're orchestrating (simBusy) or in the cooldown after
  // an eager repaint (curHold); otherwise the poll follows the viewed lineage as
  // it progresses (throttled while it's mid-call so a "run rest" doesn't thrash).
  if (!simBusy && now >= curHold && sig.cur !== curSig) {
    if (!(sig.running && now - lastCurPaint < CMP_RELOAD_MS)) {
      curSig = sig.cur;
      lastCurPaint = now;
      refreshCurLive(seq);
    }
  }
  renderLineageBar();
  updateStepLabels();
}

// Repaint the CURRENT side from the VIEWED lineage's live state.
async function refreshCurLive(seq) {
  const lin = curLineage();
  if (!lin) return;
  await loadBranchSteps(seq); // refresh the viewed lineage's step list + obs tree
  if (seq !== openSeq || !openTarget) return;
  await paintCur(lin.branch, seq);
  const last = branchSteps[branchSteps.length - 1] ?? null;
  if (last) { viewStep = last.template ?? last.step; viewNode = last.node ?? null; }
  loadTextForView(seq);
}

// --- per-side step controls (built once, into the column headers) ------------

// The committed pipeline steps the user scrubs/reverts by — gated cache.llm
// calls only. Excludes `library_match` + `image_prompt` (mechanical per-object
// service calls that auto-play through when stepping, so they aren't steps).
const isStepEvent = (e) =>
  e.kind === "cache.llm" && typeof e.index === "number" &&
  e.step !== "library_match" && e.step !== "image_prompt";

async function loadOrigSteps(seq) {
  if (!openTarget) return;
  let evs = [];
  try { evs = await api.eventsHistory(state.run, openTarget.slot, openTarget.model); }
  catch { evs = []; }
  if (seq !== openSeq) return;
  origSteps = evs.filter(isStepEvent);
  // Anchor the scrubber at the fork the FIRST time the list loads.
  if (!origInit && forkIndex != null) {
    let fp = origSteps.findIndex((s) => s.index === forkIndex);
    if (fp < 0) fp = origSteps.filter((s) => s.index < forkIndex).length;
    forkPtr = fp;
    if (fp >= origSteps.length) origAuto = true; // fork past the last step = full
    else { origAuto = false; origPtr = fp; }
    origInit = true;
  }
  const m = createObsModel();
  for (const e of evs) m.feed(e);
  prevObs = m;
  prevViewer?.recolorAll();
  renderSideTree(prevTreeEl, prevObs, prevExpanded);
  updateStepLabels();
}

// Project PREVIOUS at the current scrub position: cut right BEFORE step
// `origPtr` (the scene about to run it). Default = the fork. `origAuto` = full.
function repaintPrev(seq) {
  if (!openTarget) return;
  const { slot, model } = openTarget;
  const opts = origAuto ? {} : { untilIndex: origSteps[origPtr]?.index ?? forkIndex };
  paint(prevViewer, slot, model, opts, seq);
}

function scrubOrig(delta) {
  if (!origSteps.length || !openTarget) return;
  const p = (origAuto ? origSteps.length : origPtr) + delta;
  if (p >= origSteps.length) {
    origAuto = true;
    origPtr = origSteps.length;
  } else {
    origAuto = false;
    origPtr = Math.max(0, p);
  }
  repaintPrev(openSeq);
  updateStepLabels();
}

function goLive() {
  origAuto = true;
  origPtr = origSteps.length;
  repaintPrev(openSeq);
  updateStepLabels();
}

// Pull a fresh /slots snapshot and refresh the shared cell state so
// branchSummaryById() reflects each lineage's CURRENT server state right after
// an eager action (the background poll is too coarse / can resolve out of order).
async function awaitSlots() {
  const payload = await api.slots(state.run);
  state.slots = payload.slots;
}

// The committed source call of (step, node) — the original to diff the viewed
// lineage's step against. Found in the committed-log step list PREVIOUS loads.
function sourceCallFor(step, node) {
  if (!step) return null;
  return origSteps.find(
    (s) => (s.template ?? s.step) === step && (node == null || s.node === node),
  ) ?? null;
}

// "step ▶": advance every SELECTED LLM by one step, in parallel. An existing
// lineage continues on its own model; a newly-selected LLM forks a fresh
// lineage from the ORIGINAL run at the current depth (its prior steps are the
// original's committed outputs). `auto` runs the selected existing lineages to
// completion (a freshly-forked LLM runs one step; run-rest again to finish it).
async function stepSim(auto) {
  if (!openTarget || simBusy) return;
  if (selectedModels.size === 0) { toast("pick at least one LLM to run the step on", "err"); return; }
  const seq = openSeq;
  const { slot, model } = openTarget;
  simBusy = true;
  updateSimControls();
  const overrides = overridesPayload();
  // A newly-joining LLM forks from the source at the round-th committed step
  // past the fork (so it carries the original's prior steps as context).
  const forkAt = forkPtr >= 0 ? (origSteps[forkPtr + round] ?? null) : null;
  const want = state.models.filter((m) => selectedModels.has(m));
  if (curLabel) curLabel.textContent = `running ${want.length} LLM${want.length === 1 ? "" : "s"}…`;
  try {
    await Promise.all(want.map(async (llm) => {
      const lin = lineages.get(llm);
      if (lin) {
        try { await api.branchStep(lin.branch, { auto }); if (!auto) lin.depth += 1; }
        catch (e) { toast(`${llm}: ${e.message ?? e}`, "err"); }
        return;
      }
      if (!forkAt) {
        toast(`${llm}: the original run has no step at this depth to fork from`, "err");
        return;
      }
      try {
        const resp = await api.createBranch(state.run, slot, model, {
          event_index: forkAt.index,
          step: forkAt.template ?? forkAt.step,
          overrides,
          model: llm,
        });
        const bid = resp.branch?.id;
        if (bid) lineages.set(llm, { llm, branch: bid, depth: round + 1 });
      } catch (e) { toast(`${llm}: ${e.message ?? e}`, "err"); }
    }));
    if (seq !== openSeq) return;
    round += 1;
    // Keep the view on a lineage that just ran (the current one if it was
    // selected, else the first selected with a lineage).
    if (!viewLLM || !lineages.has(viewLLM) || !selectedModels.has(viewLLM)) {
      viewLLM = want.find((m) => lineages.has(m)) ?? viewLLM;
    }
    renderLineageBar();
    await settleAndRefresh(seq); // poll each lineage to park, repaint the viewed one
  } catch (e) {
    if (seq === openSeq) toast(`step failed: ${e.message ?? e}`, "err");
  } finally {
    if (seq === openSeq) {
      simBusy = false;
      curHold = performance.now() + 1500;
      updateStepLabels();
      updateSimControls();
    }
    emit("poll-now");
  }
}

const runRest = () => stepSim(true);

// Poll /slots until the VIEWED lineage parks (or finishes / errors), refreshing
// every lineage's chip as they land, then paint the viewed lineage + its text.
async function settleAndRefresh(seq) {
  const deadline = performance.now() + 180000;
  while (performance.now() < deadline) {
    if (seq !== openSeq) return;
    await awaitSlots();
    if (seq !== openSeq) return;
    renderLineageBar();
    updateStepLabels();
    const lin = curLineage();
    const b = lin ? branchSummaryById(lin.branch) : null;
    const parked = b && (b.pending || ["paused", "done", "error"].includes(b.status));
    if (!lin || parked) break;
    await sleep(800);
  }
  if (seq !== openSeq) return;
  await loadBranchSteps(seq);
  const last = branchSteps[branchSteps.length - 1] ?? null;
  if (last) { viewStep = last.template ?? last.step; viewNode = last.node ?? null; }
  const lin = curLineage();
  if (lin) await paintCur(lin.branch, seq, true);
  await loadTextForView(seq);
}

// Switch the CURRENT view to a lineage — fetch + clear-paint its scene (so
// meshes swap cleanly even when two lineages reused a node id), refresh its obs
// tree + step list, and point the text at its latest committed step.
async function viewLineage(llm) {
  const lin = lineages.get(llm);
  if (!lin) return;
  const seq = openSeq;
  viewLLM = llm;
  curSig = null; // force the poll to re-track this lineage
  renderLineageBar();
  await loadBranchSteps(seq);
  if (seq !== openSeq) return;
  // Keep the camera put — swapping which lineage is viewed shouldn't move the
  // user's vantage on the (comparable) scene.
  await paintCur(lin.branch, seq, true, true);
  if (seq !== openSeq) return;
  const last = branchSteps[branchSteps.length - 1] ?? null;
  if (last) { viewStep = last.template ?? last.step; viewNode = last.node ?? null; }
  await loadTextForView(seq);
  if (seq !== openSeq) return;
  updateStepLabels();
  curHold = performance.now() + 1200;
}

// Paint a lineage's scene into the CURRENT viewer. `clear` does a full reload
// (used on view-switch / revert) so a same-id node from another lineage can't
// linger; otherwise it streams the bundle only when the cut references meshes
// the viewer lacks (cheap live progress repaint).
async function paintCur(branch, seq, clear = false, keepCamera = false) {
  if (!curViewer || !branch) return;
  try {
    const proj = await api.branchScene(branch);
    if (seq !== openSeq) return;
    if (clear) curViewer.clear({ keepCamera });
    applySceneProjection(curViewer, proj);
    const needBundle = clear || (proj.nodes ?? []).some((n) => n.mesh_url && !curViewer.hasModel(n.id));
    if (needBundle) curViewer.prefetchBundle(api.branchMeshesUrl(branch));
  } catch (e) {
    if (seq === openSeq) toast(`compare scene load failed: ${e.message}`, "err");
  }
}

async function loadBranchSteps(seq) {
  const lin = curLineage();
  if (!openTarget || !lin) { branchSteps = []; return; }
  let evs = [];
  try { evs = await api.branchEventsHistory(state.run, lin.branch); }
  catch { evs = []; }
  if (seq !== openSeq) return;
  branchSteps = evs.filter(isStepEvent);
  const m = createObsModel();
  for (const e of evs) m.feed(e);
  curObs = m;
  curViewer?.recolorAll();
  renderSideTree(curTreeEl, curObs, curExpanded, onSimRevert);
  updateStepLabels();
}

// Revert the VIEWED lineage to BEFORE event `index`: truncate there, drop the
// meshes generated at/after it, refresh its edit set to the lab's CURRENT
// drafts, and PAUSE — a following "step ▶" re-runs that step under the current
// edits on whichever LLM(s) the picker selects.
async function rewindBranchTo(call) {
  const lin = curLineage();
  if (!openTarget || simBusy || !lin) return;
  const seq = openSeq;
  simBusy = true;
  updateSimControls();
  try {
    await api.branchRewind(lin.branch, call.index, overridesPayload());
    if (seq !== openSeq) return;
    viewStep = call.template ?? call.step ?? null;
    viewNode = call.node ?? null;
    await awaitSlots();
    if (seq !== openSeq) return;
    await loadBranchSteps(seq);
    if (seq !== openSeq) return;
    await paintCur(lin.branch, seq, true, true); // keep the camera across the revert
    if (seq !== openSeq) return;
    await loadTextForView(seq);
    if (seq !== openSeq) return;
    curSig = null;
    curHold = performance.now() + 1500;
    updateStepLabels();
    toast(`reverted ${viewLLM} to before ${stepDesc(call)} — pick LLM(s) + “step ▶” to re-run`);
  } catch (e) {
    if (seq === openSeq) toast(`revert failed: ${e.message ?? e}`, "err");
  } finally {
    if (seq === openSeq) { simBusy = false; updateSimControls(); }
    emit("poll-now");
  }
}

// Header "⟲ revert": walk back the viewed lineage's LAST committed step.
async function revertSim() {
  if (!openTarget || simBusy) return;
  await loadBranchSteps(openSeq);
  const last = branchSteps[branchSteps.length - 1];
  if (!last) { toast("this lineage has no committed step to revert"); return; }
  rewindBranchTo(last);
}

function onSimRevert(call) {
  rewindBranchTo(call);
}

// "step ▶" / "run rest" are live whenever we're not mid-orchestration and at
// least one LLM is selected (existing lineages continue, new ones fork).
function updateSimControls() {
  if (!curStepBtn || !curRunBtn) return;
  const blocked = simBusy || selectedModels.size === 0;
  curStepBtn.disabled = blocked;
  curRunBtn.disabled = blocked;
  curStepBtn.textContent = selectedModels.size > 1 ? `step ▶ (${selectedModels.size})` : "step ▶";
}

function updateStepLabels() {
  if (!openTarget) return;
  const n = origSteps.length;
  if (prevLabel) {
    if (!n) {
      prevLabel.textContent = "fork baseline";
    } else if (origAuto) {
      prevLabel.textContent = `live · full scene (${n} steps)`;
    } else {
      const tag = origPtr === forkPtr ? "fork ⑂ " : "before ";
      prevLabel.textContent = `${tag}${stepDesc(origSteps[origPtr])} (${origPtr + 1}/${n})`;
    }
  }
  if (curLabel && !simBusy) {
    const lin = curLineage();
    const b = lin ? branchSummaryById(lin.branch) : null;
    if (!lin) {
      curLabel.textContent = lineages.size ? "pick a lineage to view" : "no lineages — pick LLM(s) + “step ▶”";
    } else {
      curLabel.textContent = `${viewLLM} · ${statusView(b).label}`;
    }
  }
  updateSimControls();
}

// A header button that shows/hides one side's docked observability tree.
function treeToggle(getEl, getObs, expanded, onRevert = null) {
  const b = el("button", { class: "cmp-step-btn", text: "tree ▾", title: "show/hide this side's observability tree" });
  b.addEventListener("click", () => {
    const open = getEl().classList.toggle("open");
    b.textContent = open ? "tree ▴" : "tree ▾";
    if (open) renderSideTree(getEl(), getObs(), expanded, onRevert);
  });
  return b;
}

// Pop a scene open in the full-scene inspector (overlay). `branch=false` opens
// the original (source) run; `branch=true` opens the VIEWED lineage by its id.
function openFull(branch) {
  if (!openTarget) return;
  const lin = curLineage();
  emit("open-cell", { slot: openTarget.slot, model: openTarget.model, branch: branch ? (lin?.branch ?? null) : null });
}

function buildStepControls() {
  const prevHead = document.getElementById("cmp-prev-head");
  const curHead = document.getElementById("cmp-cur-head");
  prevTreeEl = document.getElementById("cmp-prev-obs");
  curTreeEl = document.getElementById("cmp-cur-obs");
  prevLabel = el("span", { class: "cmp-step-label" });
  prevHead.appendChild(el("span", { class: "cmp-step-ctl" }, [
    el("button", { class: "cmp-step-btn", text: "◀", title: "previous committed step", onclick: () => scrubOrig(-1) }),
    el("button", { class: "cmp-step-btn", text: "▶", title: "next committed step", onclick: () => scrubOrig(1) }),
    prevLabel,
    el("button", { class: "cmp-step-btn", text: "live ▸|", title: "follow the original's latest committed step", onclick: goLive }),
    treeToggle(() => prevTreeEl, () => prevObs, prevExpanded),
    el("button", { class: "cmp-step-btn", text: "open scene ↗", title: "open the original run's FULL scene in the inspector", onclick: () => openFull(false) }),
  ]));
  curLabel = el("span", { class: "cmp-step-label" });
  curStepBtn = el("button", { class: "cmp-step-btn", text: "step ▶", title: "run the next step on every selected LLM — each continues its own lineage; a newly-picked LLM forks from the original", onclick: () => stepSim(false) });
  curRunBtn = el("button", { class: "cmp-step-btn", text: "run rest", title: "run the selected lineages to completion", onclick: () => runRest() });
  curHead.appendChild(el("span", { class: "cmp-step-ctl" }, [
    curLabel,
    el("button", { class: "cmp-step-btn", text: "⟲ revert", title: "revert the viewed lineage's last committed step under the current edits, then re-run it", onclick: revertSim }),
    curStepBtn,
    curRunBtn,
    treeToggle(() => curTreeEl, () => curObs, curExpanded, onSimRevert),
    el("button", { class: "cmp-step-btn", text: "open scene ↗", title: "open the viewed lineage's FULL scene in the inspector", onclick: () => openFull(true) }),
  ]));

  // Above the simulation canvas: which LLMs the next step runs on (NEXT), and
  // the live lineages to view (VIEW).
  const bar = el("div", { class: "cmp-sim-bar" });
  modelBarEl = el("div", { class: "cmp-pick-row" });
  lineageBarEl = el("div", { class: "cmp-pick-row cmp-cands", style: "display:none" });
  bar.appendChild(modelBarEl);
  bar.appendChild(lineageBarEl);
  curHead.after(bar);

  renderModelBar();
  renderLineageBar();
  updateSimControls();
}

// NEXT selector: the LLMs the next "step ▶" runs (multi-select; ≥1 stays on).
function renderModelBar() {
  if (!modelBarEl) return;
  modelBarEl.textContent = "";
  modelBarEl.appendChild(el("span", { class: "cmp-pick-lab", text: "next step on" }));
  for (const m of state.models) {
    const on = selectedModels.has(m);
    const has = lineages.has(m);
    modelBarEl.appendChild(el("button", {
      class: `cmp-model-chip${on ? " on" : ""}`,
      text: has ? `${m} ⮑` : m,
      title: on
        ? `the next step runs on ${m}${has ? " (continues its lineage)" : " (forks a new lineage from the original)"} — click to drop it`
        : `also run the next step on ${m}${has ? " (continues its lineage)" : " (forks a new lineage from the original)"}`,
      onclick: () => {
        if (selectedModels.has(m)) { if (selectedModels.size > 1) selectedModels.delete(m); }
        else selectedModels.add(m);
        renderModelBar();
        updateSimControls();
      },
    }));
  }
}

// VIEW selector: one chip per live lineage. Clicking shows that lineage's
// scene/text on the CURRENT pane. The dot reflects each lineage's status.
function renderLineageBar() {
  if (!lineageBarEl) return;
  lineageBarEl.textContent = "";
  if (lineages.size === 0) { lineageBarEl.style.display = "none"; return; }
  lineageBarEl.style.display = "";
  lineageBarEl.appendChild(el("span", { class: "cmp-pick-lab", text: "view" }));
  for (const lin of lineages.values()) {
    const b = branchSummaryById(lin.branch);
    const status = statusView(b).dot;
    const sel = lin.llm === viewLLM;
    lineageBarEl.appendChild(el("button", {
      class: `cmp-cand${sel ? " on" : ""}${b?.status === "error" ? " err" : ""}`,
      title: `view ${lin.llm}'s lineage · ${status}`,
      onclick: () => viewLineage(lin.llm),
    },
      el("span", { class: `dot ${status}` }),
      el("span", { text: lin.llm }),
    ));
  }
}

// Per-side visibility layers, overlaid on each canvas.
const CMP_TOGGLES = [
  ["objects", "objects"],
  ["frames", "frames"],
  ["zones", "zones"],
  ["meshes", "meshes"],
  ["grid", "grid"],
  ["bboxes", "bboxes"],
  ["proxies", "proxies"],
];

function buildToggleBar(host, viewer) {
  const bar = el("div", { class: "cmp-toggles" });
  for (const [key, label] of CMP_TOGGLES) {
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

function ensureViewers() {
  if (prevViewer) return;
  prevViewer = createViewer(document.getElementById("cmp-prev-host"));
  curViewer = createViewer(document.getElementById("cmp-cur-host"));
  const link = (a, b) =>
    a.onCameraChange(() => {
      if (!ready || !linked || syncing) return;
      syncing = true;
      b.setView(a.getView());
      syncing = false;
    });
  link(prevViewer, curViewer);
  link(curViewer, prevViewer);
  prevViewer.setOriginOf((id) => emittedStep(prevObs.model, id));
  curViewer.setOriginOf((id) => emittedStep(curOriginModel(), id));
  buildToggleBar(document.getElementById("cmp-prev-host"), prevViewer);
  buildToggleBar(document.getElementById("cmp-cur-host"), curViewer);
  buildStepControls();
}

// PREVIOUS-side paint: the source cell's scene at a cut (opts.untilIndex) or full.
async function paint(viewer, slot, model, opts, seq) {
  try {
    const proj = await api.scene(state.run, slot, model, opts);
    if (seq !== openSeq) return;
    applySceneProjection(viewer, proj);
    const needBundle = (proj.nodes ?? []).some((n) => n.mesh_url && !viewer.hasModel(n.id));
    if (needBundle) viewer.prefetchBundle(api.meshesUrl(state.run, slot, model, opts));
  } catch (e) {
    if (seq === openSeq) toast(`compare scene load failed: ${e.message}`, "err");
  }
}

// Plain-language status of a lineage + whether it's waiting on the user.
// The label comes from the shared renderer; `wait` (paused ⇒ needs a "step ▶")
// drives the per-lineage prompt.
function branchStatusText(b) {
  if (!b) return { dot: "idle", text: "no lineage", wait: false };
  const v = statusView(b);
  return { dot: v.dot, text: v.label, wait: v.state === "paused" };
}

// Load + render the input/output diff for the CURRENT view step (viewStep /
// viewNode) of the VIEWED lineage against the original's same call.
async function loadTextForView(seq) {
  if (!openTarget) return;
  const { slot, model } = openTarget;
  const step = viewStep;
  const node = viewNode;
  if (!step) { textEl.replaceChildren(); return; }
  const lin = curLineage();
  let cur = null;
  if (lin) {
    try { cur = await api.branchStepEvent(lin.branch, step, node); }
    catch { cur = null; }
    if (seq !== openSeq) return;
  }
  const src = sourceCallFor(step, node);
  let prevIndex = src ? src.index : null;
  if (prevIndex == null && step === openTarget.step && (node == null || node === openTarget.node)) {
    prevIndex = openTarget.index;
  }
  let prev = null;
  if (prevIndex != null) {
    try { prev = await api.stepEvent(state.run, slot, model, prevIndex, step); }
    catch { prev = null; }
    if (seq !== openSeq) return;
  }
  renderText(prev, cur, step, node);
}

// Whether the input/system diffs are expanded. Off by default (the panel shows
// just the two outputs); persisted across re-renders + model switches so a poll
// or a VIEW change doesn't re-collapse a diff the user opened.
let textDiffsOpen = false;

function renderText(prev, cur, step, node) {
  const frag = document.createDocumentFragment();
  const status = branchStatusText(curLineage() ? branchSummaryById(curLineage().branch) : null);
  frag.appendChild(el("div", { class: "ct-status" }, [
    el("span", { class: `dot ${status.dot}` }),
    el("span", { text: `${viewLLM ?? "simulation"}: ${status.text}` }),
  ]));

  // Default view: the two outputs side by side. The lineage picked on the right
  // (VIEW) drives the "simulation" column, so switching models swaps it.
  const simText = cur
    ? fmtJson(cur.output)
    : `not run on ${viewLLM ?? "this lineage"} yet — pick LLM(s) + “step ▶”.`;
  frag.appendChild(el("div", { class: "ct-grid" }, [
    el("div", { class: "ct-sec" }, [
      el("div", { class: "ct-h", text: "original output" }),
      el("pre", { class: "fit-full", text: prev ? fmtJson(prev.output) : "(no original — branch diverged here)" }),
    ]),
    el("div", { class: "ct-sec" }, [
      el("div", { class: "ct-h", text: `simulation output${viewLLM ? " · " + viewLLM : ""}` }),
      el("pre", { class: "fit-full", text: simText }),
    ]),
  ]));

  // The input / system diffs are tucked behind an expander — collapsed by
  // default, only built when there's something to show.
  const diffs = el("div", { class: "ct-diffs", style: textDiffsOpen ? "" : "display:none" });
  if (cur) {
    if (((prev && prev.system) || "") !== (cur.system || "")) {
      diffs.appendChild(el("div", { class: "ct-sec" }, [
        el("div", { class: "ct-h", text: "system · diff (original → simulation)" }),
        diffPre((prev && prev.system) || "", cur.system || ""),
      ]));
    }
    diffs.appendChild(el("div", { class: "ct-sec" }, [
      el("div", { class: "ct-h", text: "input · diff (original → simulation)" }),
      diffPre((prev && prev.user) || "", cur.user || ""),
    ]));
  } else if (prev) {
    diffs.appendChild(el("div", { class: "ct-sec" }, [
      el("div", { class: "ct-h", text: "original input · this step" }),
      el("pre", { class: "fit-full", text: prev.user || "" }),
    ]));
  }
  if (diffs.childNodes.length) {
    const label = (open) => (open ? "hide input / system diff ▴" : "show input / system diff ▾");
    const toggle = el("button", {
      class: "cmp-step-btn ct-expand",
      text: label(textDiffsOpen),
      onclick: () => {
        textDiffsOpen = !textDiffsOpen;
        diffs.style.display = textDiffsOpen ? "" : "none";
        toggle.textContent = label(textDiffsOpen);
      },
    });
    frag.appendChild(toggle);
    frag.appendChild(diffs);
  }
  textEl.replaceChildren(frag);
}

async function openCompare({ slot, model, step, node, index, branch }) {
  if (!branch) { toast("no simulation branch for this call — simulate it first", "err"); return; }
  ensureViewers();
  const seq = ++openSeq;
  openTarget = { slot, model, step, node, index };
  prevSig = null;
  curSig = null;
  lastPrevPaint = performance.now();
  lastCurPaint = performance.now();
  ready = false;
  origAuto = false;
  origPtr = 0;
  origSteps = [];
  forkIndex = index;
  forkPtr = -1;
  origInit = false;
  branchSteps = [];
  simBusy = false;
  curHold = 0;
  round = 0;
  viewStep = step;
  viewNode = node;
  // Seed the lineage forest with the entry sim, then rediscover the rest of
  // this exploration — every other branch of the same cell forked at the same
  // event (the prompt lab groups them), keyed by its pinned LLM. So opening
  // compare on a grouped sim shows ALL its LLMs, and "step ▶" defaults to
  // advancing every existing lineage.
  lineages = new Map();
  const entry = branchSummaryById(branch);
  const entryLLM = entry?.pin ?? model;
  lineages.set(entryLLM, { llm: entryLLM, branch, depth: 0 });
  viewLLM = entryLLM;
  for (const b of cellBranches(slot, model)) {
    if (b.id === branch || b.fork_index !== index) continue;
    const llm = b.pin ?? model;
    if (!lineages.has(llm)) lineages.set(llm, { llm, branch: b.id, depth: 0 });
  }
  selectedModels = new Set(lineages.keys());
  renderModelBar();
  renderLineageBar();
  prevObs = createObsModel();
  curObs = createObsModel();
  prevExpanded.clear();
  curExpanded.clear();
  if (prevTreeEl) prevTreeEl.textContent = "";
  if (curTreeEl) curTreeEl.textContent = "";
  updateStepLabels();
  titleEl.textContent = `${slot} · ${model}`;
  subEl.textContent = step ? `from: ${step}` : "";
  root.classList.add("open");
  textEl.classList.add("open");
  prevViewer.setActive(true);
  curViewer.setActive(true);
  prevViewer.clear();
  curViewer.clear();
  await Promise.all([
    paint(prevViewer, slot, model, { untilIndex: index }, seq),
    paintCur(branch, seq, true),
  ]);
  if (seq !== openSeq) return;
  await Promise.all([
    loadOrigSteps(seq),   // original-side scrubber + observability tree
    loadBranchSteps(seq), // viewed lineage's step list + observability tree
  ]);
  if (seq !== openSeq) return;
  // Anchor `round` at the entry lineage's committed steps past the fork, so a
  // sim already stepped in the prompt lab forks new LLMs at the right depth.
  round = branchSteps.filter((s) => typeof s.index === "number" && s.index >= forkIndex).length;
  const head = branchSteps[branchSteps.length - 1] ?? null;
  if (head) { viewStep = head.template ?? head.step; viewNode = head.node ?? null; }

  const sig0 = cellSig(slot, model);
  prevSig = sig0.prev;
  curSig = sig0.cur;
  setTimeout(() => {
    if (seq !== openSeq) return;
    curViewer.fit();
    prevViewer.setView(curViewer.getView());
    ready = true;
  }, 700);
  loadTextForView(seq);
}

function closeCompare() {
  // Lineages PERSIST as real sims (manage them from the prompt lab) — closing
  // compare just drops the view, it doesn't discard any branch.
  openSeq += 1;
  openTarget = null;
  lineages = new Map();
  viewLLM = null;
  ready = false;
  simBusy = false;
  root.classList.remove("open");
  prevViewer?.setActive(false);
  curViewer?.setActive(false);
  prevViewer?.clear();
  curViewer?.clear();
}

export function initCompare() {
  document.getElementById("compare-close").addEventListener("click", closeCompare);
  document.getElementById("compare-sync").addEventListener("click", () => {
    if (prevViewer && curViewer) prevViewer.setView(curViewer.getView());
  });
  const linkCb = document.getElementById("compare-link");
  linked = linkCb.checked;
  linkCb.addEventListener("change", () => { linked = linkCb.checked; });
  document.getElementById("compare-text-toggle").addEventListener("click", () => {
    textEl.classList.toggle("open");
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && root.classList.contains("open") && !document.getElementById("modal-root").firstChild) {
      closeCompare();
    }
  });
  on("open-compare", openCompare);
  // Re-paint live as cells/lineages step (keeps the two sides in lockstep).
  on("slots", refreshOpenCompare);
}
