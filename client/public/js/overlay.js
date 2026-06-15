// The slot overlay: enlarged 3D view of one cell (source run or its
// simulation branch) with the observability dock on the right.

import { api } from "./api.js";
import { state, emit, on, cellSummary } from "./state.js";
import { el, toast, stepUntilSelect, openModal } from "./ui.js";
import { openStream, dispatchSceneEvent, applySceneProjection, createObsModel } from "./events.js";
import * as obstree from "./obstree.js";

const overlayEl = document.getElementById("overlay");
const titleEl = document.getElementById("overlay-title");
const crumbsEl = document.getElementById("overlay-crumbs");
const dotEl = document.getElementById("overlay-dot");
const statusEl = document.getElementById("overlay-status");
const actionBtn = document.getElementById("overlay-action");
const resetBtn = document.getElementById("overlay-reset");

let viewer = null;
let stream = null;
let obs = createObsModel();
let renderQueued = false;
let openSeq = 0; // monotonically increasing guard for async open races

export function initOverlay(sceneViewer) {
  viewer = sceneViewer;

  // Tree ↔ 3D linking. A node-row click toggles selection and frames the
  // bbox (dimming the rest); a call click guarantees its node is focused
  // without toggling it off. 3D-side picks highlight + reveal the tree row,
  // and the hover tooltip reads seed/plan/image text from the obs model.
  viewer.setNodeInfo((id) => obs.model.nodes.get(id) ?? null);
  viewer.onSelect((id) => obstree.markSelected(id, { scroll: true }));
  obstree.setOnNodeClick((id, { ensureSelected = false } = {}) => {
    if (!viewer.hasBbox(id)) return;
    if (ensureSelected && viewer.getSelected() === id) return;
    viewer.select(id, { frame: true });
  });
  // Per-node hiding: the tree's eye buttons and the canvas right-click share
  // the viewer's hidden set; any change re-renders the tree so eye states
  // and row dimming follow.
  obstree.setHiddenApi({ isHidden: viewer.isHidden, toggle: viewer.toggleHidden });
  viewer.onHiddenChange(() => obstree.renderTree(obs.model));

  document.getElementById("overlay-close").addEventListener("click", closeOverlay);
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && overlayEl.classList.contains("open")
        && !document.getElementById("modal-root").firstChild) {
      closeOverlay();
    }
  });
  for (const btn of document.querySelectorAll("#viewer-toggles [data-toggle]")) {
    const key = btn.dataset.toggle;
    const sync = () => btn.classList.toggle("off", !viewer.toggles[key]);
    btn.addEventListener("click", () => {
      viewer.toggles[key] = !viewer.toggles[key];
      viewer.refreshVisibility();
      sync();
    });
    sync();
  }
  document.getElementById("btn-refit").addEventListener("click", () => viewer.fit());
  actionBtn.addEventListener("click", onAction);
  resetBtn.addEventListener("click", onReset);
  on("slots", () => { if (state.view) renderHeader(); });
  on("open-cell", openCell);
  initObsResizer();
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
  try { saved = Number(localStorage.getItem(OBSDOCK_WIDTH_KEY)); } catch { /* private mode */ }
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
    const width = Math.max(OBSDOCK_MIN, Math.min(rect.right - ev.clientX, max));
    dock.style.width = `${Math.round(width)}px`;
  });
  const end = (ev) => {
    if (!dragging) return;
    dragging = false;
    resizer.classList.remove("dragging");
    try { resizer.releasePointerCapture(ev.pointerId); } catch { /* already released */ }
    try { localStorage.setItem(OBSDOCK_WIDTH_KEY, String(parseInt(dock.style.width, 10) || OBSDOCK_MIN)); } catch { /* private mode */ }
  };
  resizer.addEventListener("pointerup", end);
  resizer.addEventListener("pointercancel", end);
}

function scheduleTreeRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    obstree.renderTree(obs.model);
    if (state.view?.branch && state.lab.simStep) obstree.renderPinned(obs.model);
  });
}

export async function openCell({ slot, model, branch = false }) {
  const seq = ++openSeq;
  const run = state.run;
  stream?.close();
  stream = null;
  state.view = { slot, model, branch };
  overlayEl.classList.add("open");
  viewer.setActive(true);
  viewer.clear();
  obs = createObsModel();
  obstree.resetDock();
  obstree.setPinStep(branch ? state.lab.simStep : null);
  // Revert is a source-cell action (branches are discarded, not rewound).
  obstree.setOnRevert(branch ? null : revertToCall);
  renderHeader();

  let projection = { nodes: [], last_index: -1 };
  try {
    projection = await api.scene(run, slot, model, { branch });
  } catch (e) {
    toast(`scene load failed: ${e.message}`, "err");
  }
  if (seq !== openSeq) return;
  applySceneProjection(viewer, projection);
  viewer.prefetchBundle(api.meshesUrl(run, slot, model, { branch }));

  let history = [];
  try {
    history = await api.eventsHistory(run, slot, model, { branch });
  } catch { /* never-started cell */ }
  if (seq !== openSeq) return;
  for (const event of history) obs.feed(event);
  obstree.renderTree(obs.model);
  if (branch && state.lab.simStep) obstree.renderPinned(obs.model);
  renderHeader(); // error message comes from the just-loaded log

  const summary = currentSummary();
  const live = (branch ? summary?.branch?.status : summary?.status) === "running";
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
  return api.eventsUrl(state.run, slot, model, { branch, since });
}

function currentSummary() {
  if (!state.view) return null;
  return cellSummary(state.view.slot, state.view.model);
}

function renderHeader() {
  if (!state.view) return;
  const { slot, model, branch } = state.view;
  const summary = currentSummary();
  const branchInfo = summary?.branch ?? null;
  titleEl.textContent = `${slot} · ${model}`;
  crumbsEl.textContent = `${state.run}${branch ? " · simulation branch" : ""}`;
  const status = branch ? (branchInfo?.status ?? "?") : (summary?.status ?? "?");
  dotEl.className = `dot ${status}`;
  const pending = branch ? branchInfo?.pending : summary?.pending;
  let statusText = branch
    ? `branch ${status} · ${branchInfo?.events_count ?? 0} events`
    : `${status}${summary?.stepped ? " · stepped" : ""} · ${summary?.events_count ?? 0} events`;
  if (pending) statusText += ` · awaiting step: ${pending.step} @ ${pending.node ?? "?"}`;
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
  let label = null;
  if (branch) {
    if (status === "running") label = "pause sim";
    else if (status === "paused" || status === "error") label = "resume sim";
  } else {
    if (status === "running") label = "pause";
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
        untilSel = stepUntilSelect(state.steps, (until) => stepCurrent(false, until), { label: "until…" });
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

  // Source/branch flip when both exist.
  let flip = document.getElementById("overlay-flip");
  if (branchInfo || branch) {
    if (!flip) {
      flip = el("button", { id: "overlay-flip" });
      actionBtn.before(flip);
    }
    flip.textContent = branch ? "view source" : "view sim branch";
    flip.onclick = () => openCell({ slot, model, branch: !branch });
  } else if (flip) {
    flip.remove();
  }
}

// Revert the open source cell to just before `call` — confirms first (it
// drops every later step + its meshes), then reloads the overlay at the cut.
function revertToCall(call) {
  const { slot, model, branch } = state.view ?? {};
  if (!slot || branch) return;
  const step = call.template ?? call.step ?? "this step";
  openModal(`revert ${slot} · ${model}?`, (close, setError) => ({
    body: [
      el("div", { class: "m-hint", text:
        `Truncates this slot's log to just before its ${step} call (#${call.index}) and ` +
        "drops every later step and its meshes. The slot lands paused there — resume, step, " +
        "or re-run (with edited prompts) to continue." }),
    ],
    actions: [
      el("button", { text: "cancel", onclick: close }),
      el("button", { class: "danger", text: "revert", onclick: async () => {
        try { await api.rewind(state.run, slot, model, call.index); }
        catch (e) { setError(e.message); return; }
        close();
        toast(`reverted ${slot} · ${model} to #${call.index}`, "ok");
        openCell({ slot, model, branch: false }); // reload scene + tree at the cut
        emit("poll-now");
      } }),
    ],
  }));
}

async function stepCurrent(auto, until = null) {
  const { slot, model, branch } = state.view ?? {};
  if (!slot) return;
  try {
    if (branch) {
      await api.branchStep(state.run, slot, model, auto);
    } else {
      const r = await api.cellStep(state.run, slot, model, { auto, until });
      // A paused cell that was just relaunched (vs. a live gate released over
      // the existing stream) needs a fresh subscription to watch the call land.
      if (r.result === "launched") {
        setTimeout(() => {
          if (state.view && state.view.slot === slot && state.view.model === model && !state.view.branch) {
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
  const status = branch ? summary?.branch?.status : summary?.status;
  try {
    if (branch) {
      if (status === "running") await api.branchPause(state.run, slot, model);
      else await api.branchResume(state.run, slot, model);
    } else {
      if (status === "running") await api.pause(state.run, slot, model);
      else await api.resume(state.run, slot, model);
    }
    emit("poll-now");
    // Re-open to (re)wire SSE against the new lifecycle state.
    setTimeout(() => { if (state.view?.slot === slot) openCell({ slot, model, branch }); }, 350);
  } catch (e) {
    toast(e.message, "err");
  }
}

async function onReset() {
  const { slot, model } = state.view ?? {};
  if (!slot) return;
  if (!confirm(`Wipe ${slot} · ${model} on "${state.run}" and start fresh?`)) return;
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
  state.view = null;
  overlayEl.classList.remove("open");
  viewer.setActive(false);
  obstree.setPinStep(null);
}
