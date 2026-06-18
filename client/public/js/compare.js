// Side-by-side 3D compare: the cell's live run (PREVIOUS) against its
// simulation branch (CURRENT — the prompt-lab edit run downstream), with the
// two cameras optionally locked together and a text panel showing the input
// diff + both outputs. Launched from a review card's "compare 3D" button.

import { api } from "./api.js";
import { state, on, emit, cellSummary } from "./state.js";
import { el, toast, diffPre, fmtJson } from "./ui.js";
import { createViewer } from "./scene3d.js";
import { applySceneProjection, createObsModel } from "./events.js";
import { renderObsTree } from "./obsmini.js";
import { overridesPayload } from "./promptlab.js";

const root = document.getElementById("compare");
const titleEl = document.getElementById("compare-title");
const subEl = document.getElementById("compare-sub");
const textEl = document.getElementById("compare-text");

let prevViewer = null;
let curViewer = null;
let linked = true;
let ready = false; // suppress camera sync until both scenes have settled
let syncing = false; // re-entrancy guard for the A<->B copy
let openSeq = 0;
// The cell currently shown, so the slots poll can re-paint both sides as they
// step — keeping the original (previous) side in lockstep with the simulation
// (current) side instead of frozen at the moment the compare opened.
let openTarget = null;
let prevSig = null;
let curSig = null;
let lastPrevPaint = 0;
let lastCurPaint = 0;
const CMP_RELOAD_MS = 3500;

// --- independent per-side stepping -------------------------------------------
// The two sides advance separately. PREVIOUS (original) scrubs the source cell's
// COMMITTED event log — its source of truth — projecting the scene at any past
// step via `?until_index`; nothing re-runs. CURRENT (simulation) advances the
// live branch with `branchStep`. `origAuto` means PREVIOUS tracks the source's
// latest committed scene; scrubbing back pins it to a step until "live".
let origSteps = []; // ordered cache.llm events of the original's committed log
let origPtr = 0;    // scrub position into origSteps (only when !origAuto)
let origAuto = false;
let forkIndex = null; // the branch's fork event index — PREVIOUS opens here
let forkPtr = -1;     // origSteps position of the fork step
let origInit = false; // has the scrubber been anchored at the fork yet?
let branchSteps = []; // ordered cache.llm events of the simulation branch
let prevLabel = null;
let curLabel = null;
let curStepBtn = null; // simulation "step ▶" / "run rest" — locked while in flight
let curRunBtn = null;
let curModelSel = null; // which LLM the NEXT simulation step runs on (per-step A/B)
// In-flight lock: block stepping the simulation again until the previous call
// lands. Set optimistically on click; cleared by the poll once the branch is
// seen running and then idle again (or it advances / errors).
let simBusy = false;
let simRan = false;
let simBusyCount = -1;

// Each side keeps its OWN observability tree (folded from its event log) so the
// pipeline + per-node LLM calls are inspectable per side. Docked below each
// canvas, toggled from the column header; call rows expand to their bytes.
let prevObs = createObsModel();
let curObs = createObsModel();
const prevExpanded = new Set();
const curExpanded = new Set();
let prevTreeEl = null;
let curTreeEl = null;

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
  const b = s?.branch ?? null;
  return {
    prev: String(s?.events_count ?? ""),
    srcRunning: s?.status === "running",
    cur: b ? `${b.events_count ?? ""}|${b.status ?? ""}|${b.last_step?.node ?? ""}|${b.last_step?.phase ?? ""}` : "none",
    running: b?.status === "running",
  };
}

// Re-paint whichever side advanced, keeping the two in lockstep: source-cell
// progress repaints the original, branch progress repaints the simulation. Each
// side is throttled only while ITS cell is auto-running, so a live "run rest"
// doesn't re-pull a whole mesh bundle every poll; discrete stepping paints at
// once.
function refreshOpenCompare() {
  if (!root.classList.contains("open") || !openTarget) return;
  const { slot, model } = openTarget;
  const sig = cellSig(slot, model);
  const now = performance.now();
  const seq = openSeq;
  if (sig.prev !== prevSig) {
    prevSig = sig.prev;
    // Throttle disk/scene pulls while the source auto-runs; discrete steps pull
    // at once. Refresh the scrubber length either way; only follow live (repaint
    // PREVIOUS) when it isn't pinned to a scrubbed-back step.
    if (!(sig.srcRunning && now - lastPrevPaint < CMP_RELOAD_MS)) {
      lastPrevPaint = now;
      loadOrigSteps(seq);
      if (origAuto) paint(prevViewer, slot, model, {}, seq);
    }
  }
  if (sig.cur !== curSig && !(sig.running && now - lastCurPaint < CMP_RELOAD_MS)) {
    curSig = sig.cur;
    lastCurPaint = now;
    loadBranchSteps(seq); // refresh the simulation's step list + obs tree
    paint(curViewer, slot, model, { branch: true }, seq);
    // Keep the diff/output + status live as the branch re-runs the step — it was
    // previously loaded only once at open, so it froze on the pre-re-run state.
    loadText(slot, model, openTarget.step, openTarget.node, openTarget.index, seq);
  }
  updateStepLabels();
}

// --- per-side step controls (built once, into the column headers) ------------

// The committed pipeline steps the user scrubs/reverts by — gated cache.llm
// calls only. Excludes `library_match`: a mechanical per-object service call
// (the live gate skips it too) that runs automatically after object_bbox_batch,
// so it's never an individual step to stop on.
const isStepEvent = (e) =>
  e.kind === "cache.llm" && typeof e.index === "number" && e.step !== "library_match";

async function loadOrigSteps(seq) {
  if (!openTarget) return;
  let evs = [];
  try { evs = await api.eventsHistory(state.run, openTarget.slot, openTarget.model); }
  catch { evs = []; }
  if (seq !== openSeq) return;
  origSteps = evs.filter(isStepEvent);
  // Anchor the scrubber at the fork the FIRST time the list loads, so PREVIOUS
  // sits on the original's state when the branch branched off. Don't re-anchor
  // on later refreshes (that would undo the user's scrubbing).
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
    origAuto = true; // cut past the last committed step = the full scene
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

async function stepSim(auto) {
  if (!openTarget || simBusy) return; // already stepping — wait for it to land
  const { slot, model } = openTarget;
  simBusy = true;
  simRan = false;
  simBusyCount = cellSummary(slot, model)?.branch?.events_count ?? -1;
  updateSimControls(); // disable step/run-rest immediately
  try {
    // Run this step on the chosen LLM (null = the branch's current model).
    await api.branchStep(state.run, slot, model, auto, null, curModelSel?.value || null);
    // The call is now in flight; the poll (updateSimControls) re-enables once
    // the branch finishes it and returns to a gated/paused state.
  } catch (e) {
    simBusy = false;
    updateSimControls();
    const msg = String(e?.message ?? "");
    toast(msg.includes("409") ? "simulation: nothing to advance" : `step failed: ${msg}`, "err");
  }
}

async function loadBranchSteps(seq) {
  if (!openTarget) return;
  let evs = [];
  try { evs = await api.eventsHistory(state.run, openTarget.slot, openTarget.model, { branch: true }); }
  catch { evs = []; }
  if (seq !== openSeq) return;
  branchSteps = evs.filter(isStepEvent);
  const m = createObsModel();
  for (const e of evs) m.feed(e);
  curObs = m;
  renderSideTree(curTreeEl, curObs, curExpanded, onSimRevert);
  updateStepLabels();
}

// Revert the branch to BEFORE event `index` (a branch cache.llm call): truncate
// there, drop the meshes generated at/after it, refresh the edit set to the
// lab's CURRENT drafts, and PAUSE — the per-call analog of the source run's
// obs-tree revert, but non-destructive (branch only). A following "step ▶"
// re-runs that step under the current snapshot + edits on whichever LLM the
// selector picks, so the same step can be A/B'd across models.
async function rewindBranchTo(index, label) {
  if (!openTarget) return;
  const { slot, model } = openTarget;
  try {
    await api.branchRewind(state.run, slot, model, index, overridesPayload());
    // Revert leaves the branch PAUSED at the cut. Drop the step lock and pull a
    // fresh summary now so "step ▶" lights up immediately (pick a model, step to
    // re-run) instead of staying disabled on the stale pre-revert "running" state.
    simBusy = false;
    simRan = false;
    emit("poll-now");
    updateSimControls();
    toast(`reverted simulation to before ${label} — pick an LLM + “step ▶” to re-run it`);
  } catch (e) {
    toast(`revert failed: ${e.message}`, "err");
  }
}

// Header "⟲ revert": walk back the LAST committed step. Click repeatedly to
// walk further; or use the ⏪ on any call in the tree to revert straight to it.
async function revertSim() {
  if (!openTarget) return;
  await loadBranchSteps(openSeq);
  const last = branchSteps[branchSteps.length - 1];
  if (!last) {
    toast("simulation: no committed step to revert");
    return;
  }
  rewindBranchTo(last.index, stepDesc(last));
}

// Per-call ⏪ in the simulation obs tree: revert to before this exact call so
// it (and everything downstream) re-runs from here, on whatever LLM you choose.
function onSimRevert(call) {
  rewindBranchTo(call.index, stepDesc(call));
}

// Disable "step ▶" / "run rest" while a call is in flight so a second step
// can't be queued before the previous one finishes on the simulated canvas.
// `simBusy` is the optimistic lock from click until the branch is observed
// running and then idle (covers the click→running poll gap); it also clears if
// the branch advances (fast step) or hits done/error.
function updateSimControls() {
  if (!curStepBtn || !curRunBtn) return;
  const b = openTarget ? (cellSummary(openTarget.slot, openTarget.model)?.branch ?? null) : null;
  // A branch paused AT A GATE reports status "running" (its task is alive,
  // blocked on the gate) WITH `pending` set — that's awaiting a manual step,
  // which IS steppable, not a call in flight. Only "running" with no pending
  // call is genuinely mid-flight, so lock the buttons only then.
  const running = !!b && b.status === "running" && !b.pending;
  if (running) simRan = true;
  if (simBusy && b && ((simRan && !running) || b.events_count !== simBusyCount
      || b.status === "done" || b.status === "error")) {
    simBusy = false;
  }
  // Nothing to advance once done; gated/awaiting/paused/error are steppable.
  const blocked = !b || running || simBusy || b.status === "done";
  curStepBtn.disabled = blocked;
  curRunBtn.disabled = blocked;
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
      // Cut sits right BEFORE origSteps[origPtr]; at the fork that's exactly
      // where the branch diverged from the original.
      const tag = origPtr === forkPtr ? "fork ⑂ " : "before ";
      prevLabel.textContent = `${tag}${stepDesc(origSteps[origPtr])} (${origPtr + 1}/${n})`;
    }
  }
  if (curLabel) {
    const b = cellSummary(openTarget.slot, openTarget.model)?.branch;
    if (!b) {
      curLabel.textContent = "no branch";
    } else if (b.pending) {
      // Gated, awaiting the next call — show exactly what it'll run + where.
      curLabel.textContent = `awaiting · ${stepDesc(b.pending)}`;
    } else if (b.status === "running") {
      // A call is in flight — show IT, not the stale last committed step.
      curLabel.textContent = `running · ${b.current ? stepDesc(b.current) : "…"}`;
    } else {
      const last = branchSteps[branchSteps.length - 1];
      curLabel.textContent = `${b.status ?? ""}${last ? " · " + stepDesc(last) : ""}`;
    }
  }
  updateSimControls();
}

// A header button that shows/hides one side's docked observability tree, and
// renders it on first open (later refreshes ride the per-side step loads).
function treeToggle(getEl, getObs, expanded, onRevert = null) {
  const b = el("button", { class: "cmp-step-btn", text: "tree ▾", title: "show/hide this side's observability tree" });
  b.addEventListener("click", () => {
    const open = getEl().classList.toggle("open");
    b.textContent = open ? "tree ▴" : "tree ▾";
    if (open) renderSideTree(getEl(), getObs(), expanded, onRevert);
  });
  return b;
}

// Pop this slot open in the full-scene inspector (the overlay: obs tree, hover
// tooltips, per-object provenance) — a careful look at what each object is,
// beyond the side-by-side. The overlay stacks above compare, so closing it
// returns here. `branch` picks which side's scene: false = the original run,
// true = the simulation branch.
function openFull(branch) {
  if (!openTarget) return;
  emit("open-cell", { slot: openTarget.slot, model: openTarget.model, branch });
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
    el("button", { class: "cmp-step-btn", text: "open scene ↗", title: "open the original run's FULL scene in the inspector — obs tree, hover, per-object detail", onclick: () => openFull(false) }),
  ]));
  curLabel = el("span", { class: "cmp-step-label" });
  // Which LLM the NEXT step runs on — independent of the model that built the
  // pre-branch scene. Pick a model, "step ▶", then "⟲ revert" + pick another +
  // "step ▶" to A/B the SAME scene context across models.
  curModelSel = el("select", { class: "cmp-step-model", title: "run the next simulation step on this LLM — revert + re-step to compare models on the same scene context" },
    (state.models || []).map((m) => el("option", { value: m, text: m })));
  curStepBtn = el("button", { class: "cmp-step-btn", text: "step ▶", title: "advance the simulation one LLM call (on the selected LLM)", onclick: () => stepSim(false) });
  curRunBtn = el("button", { class: "cmp-step-btn", text: "run rest", title: "run the simulation to completion (on the selected LLM)", onclick: () => stepSim(true) });
  curHead.appendChild(el("span", { class: "cmp-step-ctl" }, [
    curLabel,
    el("button", { class: "cmp-step-btn", text: "⟲ revert", title: "revert the simulation's last committed step under the current edits (then \u201Cstep \u25B6\u201D to re-run it)", onclick: revertSim }),
    curModelSel,
    curStepBtn,
    curRunBtn,
    treeToggle(() => curTreeEl, () => curObs, curExpanded, onSimRevert),
    el("button", { class: "cmp-step-btn", text: "open scene ↗", title: "open the simulation branch's FULL scene in the inspector — obs tree, hover, per-object detail", onclick: () => openFull(true) }),
  ]));
  updateSimControls();
}

// Per-side visibility layers, overlaid on each canvas. objects/frames/zones are
// per-category toggles (off removes that kind's meshes AND bboxes); meshes mutes
// all meshes but keeps bboxes; bboxes mutes all boxes but keeps meshes; grid is
// the floor. Each side is independent, so you can strip the original to bboxes
// while keeping full meshes on the simulation.
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
  buildToggleBar(document.getElementById("cmp-prev-host"), prevViewer);
  buildToggleBar(document.getElementById("cmp-cur-host"), curViewer);
  buildStepControls();
}

async function paint(viewer, slot, model, opts, seq) {
  try {
    const proj = await api.scene(state.run, slot, model, opts);
    if (seq !== openSeq) return;
    applySceneProjection(viewer, proj); // prunes to EXACTLY this cut, then loads bboxes
    // Only stream the (large) mesh bundle when the cut references meshes the
    // viewer lacks — a backward scrub/revert just prunes, so it skips re-fetch.
    const needBundle = (proj.nodes ?? []).some((n) => n.mesh_url && !viewer.hasModel(n.id));
    if (needBundle) viewer.prefetchBundle(api.meshesUrl(state.run, slot, model, opts));
  } catch (e) {
    if (seq === openSeq) toast(`compare scene load failed: ${e.message}`, "err");
  }
}

// Plain-language status of the simulation branch + whether it's waiting on the
// user. The whole point of compare is iterative testing, so make it obvious if
// a side is mid-call, paused awaiting a manual step, done, or errored.
function branchStatusText(b) {
  if (!b) return { dot: "idle", text: "no simulation branch", wait: false };
  if (b.pending) return { dot: "paused", text: `paused — press “step ▶” to run ${stepDesc(b.pending)}`, wait: true };
  if (b.status === "running") return { dot: "running", text: `running ${b.current ? stepDesc(b.current) : "a call"}…`, wait: false };
  if (b.status === "error") return { dot: "error", text: "error — check the branch log", wait: false };
  if (b.status === "done") return { dot: "done", text: "done — simulation complete", wait: false };
  if (b.status === "paused") return { dot: "paused", text: "paused — press “step ▶” to advance", wait: true };
  return { dot: b.status ?? "idle", text: b.status ?? "starting…", wait: false };
}

// Original (the forked call) vs simulation (the branch's RE-RUN of it). Diffs
// the exact bytes sent — system AND user — so an edit to either shows. Refreshed
// as the branch advances, so it's never frozen on the pre-re-run snapshot.
async function loadText(slot, model, step, node, index, seq) {
  const status = branchStatusText(cellSummary(slot, model)?.branch ?? null);
  const [prevR, curR] = await Promise.allSettled([
    index == null
      ? Promise.reject(new Error("no index"))
      : api.stepEvent(state.run, slot, model, index, step),
    api.branchStepEvent(state.run, slot, model, step, node),
  ]);
  if (seq !== openSeq) return;
  const prev = prevR.status === "fulfilled" ? prevR.value : null;
  const cur = curR.status === "fulfilled" ? curR.value : null;

  const frag = document.createDocumentFragment();
  frag.appendChild(el("div", { class: "ct-status" }, [
    el("span", { class: `dot ${status.dot}` }),
    el("span", { text: `simulation: ${status.text}` }),
  ]));

  if (!cur) {
    // No branch call for THIS node/step yet — the branch hasn't re-run it, so
    // there's genuinely nothing to diff (don't show a stale/other-node call).
    frag.appendChild(el("div", { class: "m-hint", text:
      `Hasn't re-run ${step}${node ? " @ " + node : ""} yet — nothing to compare. ` +
      (status.wait ? "Press “step ▶” to run it under your current (saved) prompt." : "It'll appear once the branch reaches this call.") }));
    if (prev) {
      frag.appendChild(el("div", { class: "ct-sec" }, [
        el("div", { class: "ct-h", text: "original input · this step" }),
        el("pre", { class: "fit-full", text: prev.user || "" }),
      ]));
    }
    textEl.replaceChildren(frag);
    return;
  }

  if ((prev && prev.system) !== cur.system) {
    frag.appendChild(el("div", { class: "ct-sec" }, [
      el("div", { class: "ct-h", text: "system · diff (original → simulation)" }),
      diffPre((prev && prev.system) || "", cur.system || ""),
    ]));
  }
  frag.appendChild(el("div", { class: "ct-sec" }, [
    el("div", { class: "ct-h", text: "input · diff (original → simulation)" }),
    diffPre((prev && prev.user) || "", cur.user || ""),
  ]));
  frag.appendChild(el("div", { class: "ct-grid" }, [
    el("div", { class: "ct-sec" }, [
      el("div", { class: "ct-h", text: "original output" }),
      el("pre", { class: "fit-full", text: prev ? fmtJson(prev.output) : "(unavailable)" }),
    ]),
    el("div", { class: "ct-sec" }, [
      el("div", { class: "ct-h", text: "simulation output" }),
      el("pre", { class: "fit-full", text: fmtJson(cur.output) }),
    ]),
  ]));
  textEl.replaceChildren(frag);
}

async function openCompare({ slot, model, step, node, index }) {
  ensureViewers();
  const seq = ++openSeq;
  openTarget = { slot, model, step, node, index };
  prevSig = null;
  curSig = null;
  lastPrevPaint = performance.now();
  lastCurPaint = performance.now();
  ready = false;
  // PREVIOUS opens on the original's state WHEN THE BRANCH FORKED (until_index =
  // the fork event) — the shared baseline the simulation diverged from, NOT the
  // full latest scene. The scrubber re-anchors to the fork once origSteps loads;
  // ◀/▶ then step the original back/forward, "live ▸|" jumps to the full scene.
  origAuto = false;
  origPtr = 0;
  origSteps = [];
  forkIndex = index;
  forkPtr = -1;
  origInit = false;
  branchSteps = [];
  simBusy = false;
  simRan = false;
  prevObs = createObsModel();
  curObs = createObsModel();
  prevExpanded.clear();
  curExpanded.clear();
  if (prevTreeEl) prevTreeEl.textContent = "";
  if (curTreeEl) curTreeEl.textContent = "";
  updateStepLabels();
  // Default the per-step LLM to the cell's own model (what built the scene);
  // the user can switch it to run the next step on a different model.
  if (curModelSel) curModelSel.value = model;
  titleEl.textContent = `${slot} · ${model}`;
  subEl.textContent = step ? `step: ${step}` : "";
  root.classList.add("open");
  textEl.classList.add("open"); // surface the input diff alongside the 3D
  prevViewer.setActive(true);
  curViewer.setActive(true);
  prevViewer.clear();
  curViewer.clear();
  await Promise.all([
    // PREVIOUS = the original's prefix BEFORE the forked call — the shared
    // baseline the simulation diverged from. The poll leaves it pinned here; it
    // only follows the source live once the user scrubs to "live ▸|".
    paint(prevViewer, slot, model, { untilIndex: index }, seq),
    paint(curViewer, slot, model, { branch: true }, seq),
  ]);
  if (seq !== openSeq) return;
  loadOrigSteps(seq);   // original-side scrubber + observability tree
  loadBranchSteps(seq); // simulation-side step list + observability tree

  // Baseline the progress signatures so the first poll doesn't re-paint
  // needlessly; subsequent steps move them and trigger a synced re-paint.
  const sig0 = cellSig(slot, model);
  prevSig = sig0.prev;
  curSig = sig0.cur;
  // Settle the framing once, then align previous to current and start syncing.
  setTimeout(() => {
    if (seq !== openSeq) return;
    curViewer.fit();
    prevViewer.setView(curViewer.getView());
    ready = true;
  }, 700);
  loadText(slot, model, step, node, index, seq);
}

function closeCompare() {
  openSeq += 1;
  openTarget = null;
  ready = false;
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
  // Re-paint live as cells/branches step (keeps the two sides in lockstep).
  on("slots", refreshOpenCompare);
}
