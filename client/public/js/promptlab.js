// The prompt lab: pick a template step, edit it (variables stay tokens —
// scene context never expands here), test the edit against logged events
// across cells, simulate downstream in parallel branches, and persist the
// result as a new version and/or a new run.

import { api } from "./api.js";
import { state, emit, on, cellKey, targetKey, cellSummary } from "./state.js";
import { el, fitToggle, toast, openModal, field, fmtJson, diffPre } from "./ui.js";
import { createViewer } from "./scene3d.js";
import { applySceneProjection } from "./events.js";

const labEl = document.getElementById("lab");
const runEl = document.getElementById("lab-run");
const stepsEl = document.getElementById("lab-steps");
const varsEl = document.getElementById("lab-vars");
const sysEl = document.getElementById("lab-sys");
const usrEl = document.getElementById("lab-usr");
const eventsEl = document.getElementById("lab-events");
const evCountEl = document.getElementById("lab-ev-count");
const selCountEl = document.getElementById("lab-sel-count");
const testBtn = document.getElementById("lab-test");
const simBtn = document.getElementById("lab-simulate");
const reviewEl = document.getElementById("review");
const reviewGridEl = document.getElementById("review-grid");
const reviewStepEl = document.getElementById("review-step");
const reviewCountEl = document.getElementById("review-count");
const reviewSectionsEl = document.getElementById("review-sections");
const reviewColsEl = document.getElementById("review-cols");
const reviewTestAllEl = document.getElementById("review-test-all");
const reviewTestSummaryEl = document.getElementById("review-test-summary");
const labStepAllEl = document.getElementById("lab-step-all");
const reviewStepAllEl = document.getElementById("review-step-all");
const eventStepBtns = []; // current event-panel step buttons, refreshed each poll

// Which text sections the review cards show, and how many cards per row —
// the comparison view's "fill the space with what I care about" controls.
const REVIEW_SHOW_KEY = "starshot.reviewShow";
const REVIEW_COLS_KEY = "starshot.reviewCols";
const REVIEW_3D_KEY = "starshot.review3d";
const REVIEW_SECTIONS = ["input", "output", "system", "reasoning"];
const reviewShow = loadReviewShow();
let reviewCols = loadReviewCols();
// "Convert to 3D": each card shows its slot's CURRENT scene (one live viewer
// per card) instead of text, so a new prompt's effect can be eyeballed across
// every slot side by side. Distinct from the per-card "compare 3D" (live run
// vs simulated-edit branch of one slot).
let review3d = (() => { try { return localStorage.getItem(REVIEW_3D_KEY) === "1"; } catch { return false; } })();
// 3D before/after layers, toggled independently so each reads on its own:
// "original" = the recorded boxes, "proposed" = the tested-edit overlay.
const REVIEW_ORIGINAL_KEY = "starshot.reviewShowOriginal";
const REVIEW_PROPOSED_KEY = "starshot.reviewShowProposed";
let reviewShowOriginal = (() => { try { return localStorage.getItem(REVIEW_ORIGINAL_KEY) !== "0"; } catch { return true; } })();
let reviewShowProposed = (() => { try { return localStorage.getItem(REVIEW_PROPOSED_KEY) !== "0"; } catch { return true; } })();

// Progress of the in-flight test batch(es), so the summary reads honestly
// ("testing 3/9…") instead of the live-concurrency count, which looked like
// only TEST_CONCURRENCY tests had been launched. Survives nested batches
// (a per-card "test" fired mid-"test all") via the ref count.
let testBatch = null; // { total, done } | null when idle
let testBatchRefs = 0;

function loadReviewShow() {
  try {
    const v = JSON.parse(localStorage.getItem(REVIEW_SHOW_KEY));
    if (Array.isArray(v) && v.length) return new Set(v.filter((s) => REVIEW_SECTIONS.includes(s)));
  } catch { /* default below */ }
  return new Set(["input", "output"]);
}
function loadReviewCols() {
  const n = Number(localStorage.getItem(REVIEW_COLS_KEY));
  return n >= 1 && n <= 4 ? n : 2;
}

let lastSavedVersion = null;

export function initLab() {
  document.getElementById("btn-lab").addEventListener("click", openLab);
  document.getElementById("lab-close").addEventListener("click", () => {
    closeReview();
    labEl.classList.remove("open");
  });
  // Template editors expand to fit their full text (and keep tracking it
  // while editing) via the shared control.
  document.getElementById("lab-sys-label").appendChild(fitToggle(sysEl));
  document.getElementById("lab-usr-label").appendChild(fitToggle(usrEl));
  document.getElementById("lab-ev-refresh").addEventListener("click", () => loadEvents());
  document.getElementById("lab-ev-review").addEventListener("click", openReview);
  document.getElementById("review-close").addEventListener("click", closeReview);
  reviewColsEl.value = String(reviewCols);
  reviewColsEl.addEventListener("change", () => {
    reviewCols = Number(reviewColsEl.value) || 2;
    try { localStorage.setItem(REVIEW_COLS_KEY, String(reviewCols)); } catch { /* private mode */ }
    applyReviewCols();
  });
  reviewTestAllEl.addEventListener("click", () => testEvents(state.lab.events));
  labStepAllEl.addEventListener("click", stepAllCells);
  reviewStepAllEl.addEventListener("click", stepAllCells);
  labStepAllEl.after(labUntilEl);
  reviewStepAllEl.after(reviewUntilEl);
  labUntilEl.after(labExitEl);
  reviewUntilEl.after(reviewExitEl);
  document.addEventListener("keydown", (ev) => {
    // Escape closes the review canvas only when the scene overlay (which sits
    // above it and owns its own Escape) and modals aren't open.
    if (ev.key === "Escape" && reviewEl.classList.contains("open")
        && !document.getElementById("overlay").classList.contains("open")
        && !document.getElementById("modal-root").firstChild) {
      closeReview();
    }
  });
  document.getElementById("lab-apply").addEventListener("click", applyToRunModal);
  document.getElementById("lab-save-version").addEventListener("click", saveVersionModal);
  document.getElementById("lab-save-run").addEventListener("click", saveRunModal);
  testBtn.addEventListener("click", runTests);
  simBtn.addEventListener("click", simulateDownstream);
  sysEl.addEventListener("input", onEdit);
  usrEl.addEventListener("input", onEdit);
  on("slots", () => {
    if (!labEl.classList.contains("open")) return;
    renderSims();
    updateStepAllButtons();
    refreshEventStepBtns(); // keep per-cell step buttons live between event reloads
    pollEvents(); // auto-pick up newly-completed events (incl. in the review canvas)
  });
}

export async function openLab() {
  const lab = state.lab;
  runEl.textContent = `run: ${state.run}`;
  try {
    const payload = await api.promptTemplates(state.run);
    lab.templates = new Map(payload.steps.map((s) => [s.step, s]));
  } catch (e) {
    toast(`prompt lab unavailable: ${e.message}`, "err");
    return;
  }
  lab.open = true;
  labEl.classList.add("open");
  renderSteps();
  if (!lab.step) selectStep(lab.templates.keys().next().value);
  else selectStep(lab.step, { keepSelection: true });
  updateStepAllButtons();
}

function draftFor(step) {
  const lab = state.lab;
  if (!lab.drafts.has(step)) {
    const t = lab.templates.get(step);
    lab.drafts.set(step, { system: t.system, user: t.user });
  }
  return lab.drafts.get(step);
}

function isEdited(step) {
  const t = state.lab.templates.get(step);
  const d = state.lab.drafts.get(step);
  return !!t && !!d && (d.system !== t.system || d.user !== t.user);
}

function overridesPayload() {
  const out = {};
  for (const step of state.lab.templates.keys()) {
    if (isEdited(step)) {
      const d = state.lab.drafts.get(step);
      out[step] = { system: d.system, user: d.user };
    }
  }
  return out;
}

function renderSteps() {
  stepsEl.textContent = "";
  for (const step of state.lab.templates.keys()) {
    stepsEl.appendChild(
      el("div", {
        class: `lab-step-row${step === state.lab.step ? " active" : ""}`,
        onclick: () => selectStep(step),
      },
      el("span", { text: step }),
      isEdited(step) ? el("span", { class: "edited-flag", text: "●", title: "edited" }) : null,
      ),
    );
  }
}

function selectStep(step, { keepSelection = false } = {}) {
  const lab = state.lab;
  lab.step = step;
  if (!keepSelection) {
    lab.selected.clear();
    lab.tests.clear();
  }
  const t = lab.templates.get(step);
  const d = draftFor(step);
  sysEl.value = d.system;
  usrEl.value = d.user;
  // Programmatic value swaps don't fire `input` — nudge the fit controls so
  // an expanded editor re-sizes to the newly-loaded template.
  sysEl.dispatchEvent(new Event("input"));
  usrEl.dispatchEvent(new Event("input"));
  varsEl.textContent = "";
  // The full vocabulary is injectable everywhere; chips outside this step's
  // native set are dimmed — they resolve to empty/placeholder at call time.
  const native = new Set(t.native ?? t.variables);
  for (const v of t.variables) {
    const isNative = native.has(v);
    varsEl.appendChild(
      el("span", {
        class: `var-chip${isNative ? "" : " dim"}`,
        text: `\`{${v}}\``,
        title: isNative
          ? "click to insert at the user-template cursor"
          : "click to insert — not natively populated for this step, renders empty/placeholder here",
        onclick: () => insertAtCursor(usrEl, `\`{${v}}\``),
      }),
    );
  }
  renderSteps();
  // When stepping drives the lab while the review canvas is open, keep its
  // header in sync — loadEvents() repaints the grid for the new step.
  if (reviewEl.classList.contains("open")) reviewStepEl.textContent = step;
  loadEvents();
}

function insertAtCursor(textarea, text) {
  const start = textarea.selectionStart ?? textarea.value.length;
  const end = textarea.selectionEnd ?? start;
  textarea.value = textarea.value.slice(0, start) + text + textarea.value.slice(end);
  textarea.selectionStart = textarea.selectionEnd = start + text.length;
  textarea.focus();
  onEdit();
}

function onEdit() {
  const d = draftFor(state.lab.step);
  d.system = sysEl.value;
  d.user = usrEl.value;
  renderSteps();
}

// --- event candidates -----------------------------------------------------------

let eventsSig = null;
let eventPollInFlight = false;

function eventsSignature(events) {
  return events
    .map((e) => `${e.slot}|${e.model}|${e.index}|${e.branch_live ? 1 : 0}|${e.output_preview ?? ""}`)
    .join("\n");
}

// `silent` = the auto-refresh path: never shows a loading state, and only
// touches the DOM when the event set actually changed (so it won't reset
// scroll or selection on every poll).
async function loadEvents({ silent = false } = {}) {
  const lab = state.lab;
  if (!lab.step) return;
  if (!silent) { eventsEl.textContent = ""; evCountEl.textContent = "…"; }
  let payload;
  try {
    payload = await api.stepEvents(state.run, lab.step);
  } catch (e) {
    if (!silent) {
      evCountEl.textContent = "";
      eventsEl.appendChild(el("div", { class: "muted", style: "padding:8px", text: e.message }));
    }
    return;
  }
  // The lab may have moved on while the request was in flight (step switch,
  // run switch, closed) — discard a stale response.
  if (!labEl.classList.contains("open") || lab.step !== payload.step) return;
  lab.events = payload.events;
  // Drop selections that no longer exist (run reset, step switch, etc.).
  const valid = new Set(lab.events.map((e) => targetKey(e.slot, e.model, e.index)));
  for (const k of [...lab.selected]) if (!valid.has(k)) lab.selected.delete(k);
  const sig = eventsSignature(lab.events);
  if (silent && sig === eventsSig) return; // unchanged — leave the DOM (and scroll) alone
  eventsSig = sig;
  renderEvents();
  if (reviewEl.classList.contains("open")) reconcileReview();
}

async function pollEvents() {
  // Tied to the run's slots poll; guard against overlap on a slow request.
  if (eventPollInFlight || !labEl.classList.contains("open") || !state.lab.step) return;
  eventPollInFlight = true;
  try { await loadEvents({ silent: true }); } finally { eventPollInFlight = false; }
}

function renderEvents() {
  const lab = state.lab;
  evCountEl.textContent = `${lab.events.length} eligible`;
  eventsEl.textContent = "";
  eventStepBtns.length = 0; // rebuilt below alongside the cell groups
  renderSims();
  if (lab.events.length === 0) {
    eventsEl.appendChild(el("div", {
      class: "muted", style: "padding:8px",
      text: "no logged calls of this step in this run yet — run cells first (only calls made after the versioning cutover carry re-renderable variables)",
    }));
    updateActionBar();
    return;
  }
  const groups = new Map();
  for (const ev of lab.events) {
    const k = cellKey(ev.slot, ev.model);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(ev);
  }
  for (const [k, evs] of groups) {
    const [slot, model] = k.split("|");
    const live = cellHasLiveBranch(slot, model) || evs.some((e) => e.branch_live);
    const stepBtn = makeStepBtn(slot, model);
    if (!live) stepBtn.style.marginLeft = "auto"; // else the sim-live tag holds the auto-margin
    eventStepBtns.push({ btn: stepBtn, slot, model });
    const group = el("div", { class: "ev-cell-group" },
      el("div", { class: "g-head" },
        el("span", { text: `${slot} · ${model}` }),
        el("span", { class: "muted", text: `${evs.length} call${evs.length === 1 ? "" : "s"}` }),
        live ? el("span", {
          class: "sim-live-tag",
          text: "sim live",
          title: "this slot already has a simulation branch — simulating it again asks before replacing it",
        }) : null,
        stepBtn,
      ),
    );
    for (const ev of evs) group.appendChild(eventRow(ev));
    eventsEl.appendChild(group);
    refreshStepBtn(stepBtn, slot, model);
  }
  updateActionBar();
}

function eventRow(ev) {
  const lab = state.lab;
  const key = targetKey(ev.slot, ev.model, ev.index);
  const row = el("div", { class: `ev-row${lab.selected.has(key) ? " selected" : ""}` });
  // The pick area toggles test/simulate selection; the scene button jumps to
  // the actual 3D run for this slot (selection no longer the only action).
  const pick = el("div", { class: "pick", onclick: () => {
    if (lab.selected.has(key)) lab.selected.delete(key);
    else lab.selected.add(key);
    row.classList.toggle("selected", lab.selected.has(key));
    updateActionBar();
  } },
    el("input", { type: "checkbox", ...(lab.selected.has(key) ? { checked: "" } : {}), onclick: (e) => e.preventDefault() }),
    el("span", { class: "node-tag", text: ev.node ?? "?" }),
    el("span", { class: "muted", text: ev.model_id ?? "" }),
    el("span", { class: "idx", text: `#${ev.index}` }),
  );
  row.appendChild(
    el("div", { class: "r1" },
      pick,
      el("button", {
        class: "ev-scene-btn",
        text: "scene →",
        title: "open this slot's 3D scene",
        onclick: () => emit("open-cell", { slot: ev.slot, model: ev.model, branch: false }),
      }),
    ),
  );
  row.appendChild(el("div", { class: "preview", text: ev.output_preview ?? "", title: ev.output_preview ?? "" }));
  const test = lab.tests.get(key);
  if (test) row.appendChild(testBlock(ev, test));
  return row;
}

// --- stepping source cells (drive the live run from the lab) --------------------
//
// "step" / "step all" advance the actual run cells in step mode (api.cellStep /
// api.stepAll), then jump the lab — and the review canvas, if open — to the
// prompt of the step that just ran, so you can walk the pipeline one LLM call
// at a time and inspect each prompt as it fires.

function canStepCell(c) {
  return !!c && c.status !== "done" && (!!c.pending || ["idle", "paused", "error"].includes(c.status));
}

function steppedCellCount() {
  let n = 0;
  for (const s of state.slots) for (const m of state.models) {
    const c = s.runs?.[m];
    if (c?.stepped && c.status !== "done") n += 1;
  }
  return n;
}

// "step all until <step>" — fast-forward every stepped cell to the next run
// of a target step, pause them all there, and jump the lab/review to it. The
// option list is filled lazily once `state.steps` loads.
function makeUntilSelect() {
  const sel = el("select", { class: "step-until", style: "display:none",
    title: "fast-forward every stepped cell to the next run of a step" },
    el("option", { value: "", text: "all until…" }));
  sel.addEventListener("change", () => { const v = sel.value; sel.value = ""; if (v) stepAllUntil(v); });
  return sel;
}
const labUntilEl = makeUntilSelect();
const reviewUntilEl = makeUntilSelect();

function makeExitBtn() {
  const b = el("button", { class: "ev-scene-btn", style: "display:none", text: "exit stepping",
    title: "run every stepped cell to completion and leave step mode" });
  b.addEventListener("click", exitStepping);
  return b;
}
const labExitEl = makeExitBtn();
const reviewExitEl = makeExitBtn();

async function stepAllUntil(until) {
  let r;
  try { r = await api.stepAll(state.run, { until }); }
  catch (e) { toast(e.message, "err"); return; }
  toast(`fast-forwarding ${r.advanced.length} cell${r.advanced.length === 1 ? "" : "s"} to ${until}`,
    r.advanced.length ? "ok" : "err");
  navigateToStep(until);
  emit("poll-now");
}

// Let every stepped cell run to completion and leave step mode.
async function exitStepping() {
  let r;
  try { r = await api.stepAll(state.run, { auto: true }); }
  catch (e) { toast(e.message, "err"); return; }
  toast(`finishing ${r.advanced.length} stepped cell${r.advanced.length === 1 ? "" : "s"} — leaving step mode`,
    r.advanced.length ? "ok" : "err");
  emit("poll-now");
}

function updateStepAllButtons() {
  const n = steppedCellCount();
  for (const [btn, sel, exit] of [[labStepAllEl, labUntilEl, labExitEl], [reviewStepAllEl, reviewUntilEl, reviewExitEl]]) {
    btn.style.display = n ? "" : "none";
    btn.textContent = `step all (${n})`;
    if (state.steps.length && sel.options.length <= 1) {
      for (const s of state.steps) sel.appendChild(el("option", { value: s, text: `▸ ${s}` }));
    }
    sel.style.display = (n && state.steps.length) ? "" : "none";
    exit.style.display = n ? "" : "none";
  }
}

function makeStepBtn(slot, model) {
  return el("button", { class: "ev-scene-btn", style: "display:none", text: "step",
    onclick: () => stepCell(slot, model) });
}

// Reflects a cell's live steppability/pending step onto its button; hides it
// when the cell can't be stepped (running mid-call, or done).
function refreshStepBtn(btn, slot, model) {
  if (!btn) return;
  const c = cellSummary(slot, model);
  const ok = canStepCell(c);
  btn.style.display = ok ? "" : "none";
  if (!ok) return;
  btn.textContent = c.pending?.step ? `step: ${c.pending.step}` : "step";
  btn.title = c.pending?.step
    ? `run the pending ${c.pending.step} call on ${c.pending.node ?? "?"}, then show that step's prompt`
    : "advance this cell one LLM call, then show that step's prompt";
}

function refreshEventStepBtns() {
  for (const { btn, slot, model } of eventStepBtns) refreshStepBtn(btn, slot, model);
}

// Jump the lab (and review canvas, if open) to a step's prompt.
function navigateToStep(step) {
  if (state.lab.templates.has(step)) selectStep(step, { keepSelection: true });
}

async function stepCell(slot, model) {
  // The step about to run is the step that will have "just run" once released.
  const ran = cellSummary(slot, model)?.pending?.step ?? null;
  try {
    await api.cellStep(state.run, slot, model);
  } catch (e) { toast(e.message, "err"); return; }
  if (ran) navigateToStep(ran);
  emit("poll-now");
}

async function stepAllCells() {
  // Following the pipeline only makes sense when every gated cell sat on the
  // same step; otherwise advance them all and leave the lab where it is.
  const pendings = new Set();
  for (const s of state.slots) for (const m of state.models) {
    const c = s.runs?.[m];
    if (c?.stepped && c.pending?.step) pendings.add(c.pending.step);
  }
  let r;
  try {
    r = await api.stepAll(state.run);
  } catch (e) { toast(e.message, "err"); return; }
  toast(`advanced ${r.advanced.length} stepped cell${r.advanced.length === 1 ? "" : "s"} one step`,
    r.advanced.length ? "ok" : "err");
  if (pendings.size === 1) navigateToStep([...pendings][0]);
  emit("poll-now");
}

// --- review canvas: every event's input + output, side by side ------------------
//
// Reconciles against `lab.events`: newly-completed events get cards (full
// bytes fetched per-card), re-run events refetch, vanished events drop —
// without tearing down the grid, so per-card show/hide/expand state and the
// scroll position survive the silent auto-refresh.

let reviewSeq = 0;
const reviewCards = new Map(); // key -> { card, body, ev, full, preview }

function openReview() {
  const lab = state.lab;
  if (!lab.step) return;
  if (!lab.events.length) { toast("no events to review for this step yet", "err"); return; }
  reviewSeq += 1;
  reviewCards.clear();
  reviewGridEl.textContent = "";
  reviewStepEl.textContent = lab.step;
  renderReviewControls();
  applyReviewCols();
  reviewEl.classList.add("open");
  updateStepAllButtons();
  reconcileReview();
}

function applyReviewCols() {
  reviewGridEl.style.gridTemplateColumns = `repeat(${reviewCols}, minmax(0, 1fr))`;
}

function renderReviewControls() {
  reviewSectionsEl.textContent = "";
  for (const s of REVIEW_SECTIONS) {
    reviewSectionsEl.appendChild(el("span", {
      class: `rc-pill${reviewShow.has(s) ? " on" : ""}`,
      text: s,
      title: `show/hide ${s} across every card`,
      onclick: () => {
        if (reviewShow.has(s)) reviewShow.delete(s);
        else reviewShow.add(s);
        try { localStorage.setItem(REVIEW_SHOW_KEY, JSON.stringify([...reviewShow])); } catch { /* private mode */ }
        renderReviewControls();
        for (const ref of reviewCards.values()) renderCardBody(ref);
      },
    }));
  }
  reviewSectionsEl.appendChild(el("span", {
    class: `rc-pill rc-3d${review3d ? " on" : ""}`,
    text: "3D ▦",
    title: "convert every card to its slot's current 3D scene — compare scenes across slots side by side",
    onclick: () => setReview3d(!review3d),
  }));
  // In 3D mode, let the before/after layers be isolated: hide the recorded
  // boxes to read the proposal alone, or hide the proposal to see the scene.
  if (review3d) {
    reviewSectionsEl.appendChild(vizPill("original", reviewShowOriginal, () => {
      reviewShowOriginal = !reviewShowOriginal; afterVizToggle();
    }));
    reviewSectionsEl.appendChild(vizPill("proposed", reviewShowProposed, () => {
      reviewShowProposed = !reviewShowProposed; afterVizToggle();
    }));
  }
}

function vizPill(label, on, onClick) {
  return el("span", {
    class: `rc-pill${on ? " on" : ""}`,
    text: label,
    title: `show/hide the ${label} boxes in the 3D before/after`,
    onclick: onClick,
  });
}

function afterVizToggle() {
  try {
    localStorage.setItem(REVIEW_ORIGINAL_KEY, reviewShowOriginal ? "1" : "0");
    localStorage.setItem(REVIEW_PROPOSED_KEY, reviewShowProposed ? "1" : "0");
  } catch { /* private mode */ }
  renderReviewControls();
  for (const ref of reviewCards.values()) applyCardViz(ref);
}

function setReview3d(on) {
  review3d = on;
  try { localStorage.setItem(REVIEW_3D_KEY, on ? "1" : "0"); } catch { /* private mode */ }
  renderReviewControls();
  for (const ref of reviewCards.values()) renderCardBody(ref);
}

function reconcileReview() {
  const lab = state.lab;
  const seq = reviewSeq;
  const wanted = new Set(lab.events.map((e) => targetKey(e.slot, e.model, e.index)));
  for (const [key, ref] of [...reviewCards]) {
    if (!wanted.has(key)) { teardownCard3d(ref); ref.card.remove(); reviewCards.delete(key); }
  }
  for (const ev of lab.events) {
    const key = targetKey(ev.slot, ev.model, ev.index);
    let ref = reviewCards.get(key);
    if (!ref) {
      ref = reviewCard(ev);
      reviewCards.set(key, ref);
      // In 3D mode mount the canvas immediately — it needs the slot's scene,
      // not the step-event text, so don't wait on the fetch.
      if (review3d) renderCardBody(ref);
      fetchReviewCard(ev, ref, seq);
    } else if (ref.preview !== ev.output_preview) {
      // Same call slot re-ran (e.g. a re-run-step) — its logged output changed.
      ref.preview = ev.output_preview;
      ref.ev = ev;
      fetchReviewCard(ev, ref, seq);
      if (review3d) reloadCard3d(ref); // the slot's scene changed too
    }
    // appendChild moves an existing node, preserving its state — keeps the
    // grid in server (slot/model/index) order as new cards arrive.
    reviewGridEl.appendChild(ref.card);
    // The "compare 3D" button is live whenever this slot has a simulation
    // branch, the "step" button whenever it can advance — refresh both each
    // poll so they track the cell the moment it changes.
    updateCmpBtn(ref);
    refreshStepBtn(ref.stepBtn, ev.slot, ev.model);
  }
  reviewCountEl.textContent = `${lab.events.length} event${lab.events.length === 1 ? "" : "s"}`;
  updateReviewTestSummary();
}

async function fetchReviewCard(ev, ref, seq) {
  try {
    const full = await api.stepEvent(state.run, ev.slot, ev.model, ev.index, state.lab.step);
    if (seq !== reviewSeq) return;
    ref.full = full;
    ref.card.classList.remove("err");
    renderCardBody(ref);
  } catch (e) {
    if (seq !== reviewSeq) return;
    ref.full = null;
    ref.card.classList.add("err");
    ref.body.textContent = "";
    ref.body.appendChild(el("div", { class: "rv-loading", text: `failed to load: ${e.message}` }));
  }
}

function closeReview() {
  reviewSeq += 1; // abandon any in-flight fills
  for (const ref of reviewCards.values()) teardownCard3d(ref); // release WebGL contexts
  reviewCards.clear();
  reviewEl.classList.remove("open");
}

function reviewCard(ev) {
  const body = el("div", { class: "rv-body" }, el("div", { class: "rv-loading", text: "loading…" }));
  const cmpBtn = el("button", {
    class: "ev-scene-btn",
    style: "display:none",
    text: "compare 3D ⇄",
    title: "live run vs simulated-edit branch, side by side in 3D",
    onclick: () => emit("open-compare", {
      slot: ev.slot, model: ev.model, step: state.lab.step, node: ev.node, index: ev.index,
    }),
  });
  const stepBtn = makeStepBtn(ev.slot, ev.model);
  const card = el("div", { class: "rv-card" },
    el("div", { class: "rv-head" },
      el("span", { class: "rv-slot", text: ev.slot }),
      el("span", { class: "rv-meta", text: ev.model }),
      el("span", { class: "rv-meta", text: `${ev.node ?? "?"} · #${ev.index}` }),
      el("button", {
        class: "ev-scene-btn",
        style: "margin-left:auto",
        text: "test",
        title: "run the current prompt edit on this event",
        onclick: () => testEvents([ev]),
      }),
      stepBtn,
      cmpBtn,
      el("button", {
        class: "ev-scene-btn",
        text: "scene →",
        title: "open this slot's 3D scene",
        onclick: () => emit("open-cell", { slot: ev.slot, model: ev.model, branch: false }),
      }),
    ),
    body,
  );
  return {
    card, body, cmpBtn, stepBtn, ev, full: null, preview: ev.output_preview,
    // 3D-grid mode: lazily-created per-card viewer + its viewport observer.
    host3d: null, viewer3d: null, io3d: null,
  };
}

// Shows the per-card "compare 3D" button only when this slot has a live
// simulation branch — the source of the "current" 3D scene.
function updateCmpBtn(ref) {
  ref.cmpBtn.style.display = cellHasLiveBranch(ref.ev.slot, ref.ev.model) ? "" : "none";
}

// Rebuilds a card's body from the current section toggles + any test result —
// called on load, on a section toggle, and whenever this card's test changes.
// Two states: PREVIOUS is the step's logged input/output; CURRENT is the live
// test of the prompt-lab draft, shown once this card has been tested.
function renderCardBody(ref) {
  if (review3d) { renderCard3d(ref); return; }
  teardownCard3d(ref); // back to text — release the card's WebGL context
  const body = ref.body;
  body.textContent = "";
  if (!ref.full) {
    body.appendChild(el("div", { class: "rv-loading", text: "loading…" }));
    return;
  }
  const full = ref.full;
  updateCmpBtn(ref);
  refreshStepBtn(ref.stepBtn, ref.ev.slot, ref.ev.model);
  const test = state.lab.tests.get(targetKey(ref.ev.slot, ref.ev.model, ref.ev.index));
  const tested = test && test.status === "done";
  if (reviewShow.has("input")) {
    // Once tested, diff the logged input (previous) against the draft's
    // rendered input (current); otherwise just show the logged input.
    if (tested) body.appendChild(diffInputSection("input · diff (previous → current)", full.user ?? "", test.result.user ?? ""));
    else body.appendChild(reviewSection("input · user", full.user ?? ""));
  }
  if (reviewShow.has("output")) {
    if (tested) body.appendChild(abOutputSection(full.output, test.result));
    else body.appendChild(reviewSection("output", fmtJson(full.output)));
    if (test && test.status === "queued") {
      body.appendChild(el("div", { class: "t-status", text: "queued for test…" }));
    } else if (test && test.status === "running") {
      body.appendChild(el("div", { class: "t-status", text: "testing edit… (live LLM call)" }));
    } else if (test && test.status === "error") {
      body.appendChild(el("div", { class: "t-status err", text: `test failed: ${test.error}` }));
    }
  }
  if (reviewShow.has("system")) body.appendChild(reviewSection("system", full.system ?? ""));
  if (reviewShow.has("reasoning")) body.appendChild(reviewSection("reasoning", full.reasoning || "(none)"));
}

// --- 3D grid mode -----------------------------------------------------------------
//
// One live viewer per card showing that slot's CURRENT scene. Viewers are
// created lazily when a card scrolls into view (and paused when it leaves) so
// a big review doesn't open dozens of WebGL contexts at once; keyboard nav is
// off so a keypress can't drive every canvas. `dispose()` reclaims the context
// when the card leaves 3D mode / is removed / the review closes.
//
// 3D "after" for "test edit on all": for the bbox steps, the tested output is
// overlaid as proposed (magenta) boxes on the card's live scene (the
// "before") — modeled on the old tune sandbox's renderSandboxOverlay.

// The only steps whose output is directly spatial; everything else has no 3D
// "after" without a full downstream re-run (simulation).
const BBOX_OVERLAY_STEPS = new Set(["overall_bbox", "child_bbox_batch", "object_bbox_batch"]);

function minCorner(origin, dims) {
  return [
    Math.min(origin[0], origin[0] + dims[0]),
    Math.min(origin[1], origin[1] + dims[1]),
    Math.min(origin[2], origin[2] + dims[2]),
  ];
}

// A tested step's proposed boxes in WORLD frame: overall_bbox is already world;
// *_bbox_batch assignments are authored in the owner region's local frame, so
// shift each by that region's min corner (read from the current scene).
function proposedBoxes(step, output, ownerId, sceneNodes) {
  if (!output || typeof output !== "object") return [];
  if (step === "overall_bbox") {
    const bb = output.bbox;
    return bb && Array.isArray(bb.origin) && Array.isArray(bb.dimensions)
      ? [{ origin: bb.origin, dimensions: bb.dimensions }] : [];
  }
  if (!Array.isArray(output.assignments)) return [];
  const owner = sceneNodes?.get(ownerId);
  const pmin = owner && Array.isArray(owner.origin) && Array.isArray(owner.dimensions)
    ? minCorner(owner.origin, owner.dimensions) : [0, 0, 0];
  const out = [];
  for (const a of output.assignments) {
    const bb = a && a.bbox;
    if (!bb || !Array.isArray(bb.origin) || !Array.isArray(bb.dimensions)) continue;
    out.push({
      origin: [bb.origin[0] + pmin[0], bb.origin[1] + pmin[1], bb.origin[2] + pmin[2]],
      dimensions: bb.dimensions,
    });
  }
  return out;
}

// Draw (or clear) the card's proposed-placement overlay from its current draft
// test. Only the bbox steps have a spatial "after"; everything else clears it.
function applyCardOverlay(ref) {
  const v = ref.viewer3d;
  if (!v) return;
  const step = state.lab.step;
  const test = state.lab.tests.get(targetKey(ref.ev.slot, ref.ev.model, ref.ev.index));
  const boxes = (review3d && ref.sceneNodes && BBOX_OVERLAY_STEPS.has(step) && test?.status === "done")
    ? proposedBoxes(step, test.result.output, ref.ev.node, ref.sceneNodes)
    : [];
  ref.overlayCount = boxes.length;
  v.setOverlayBoxes(boxes);
  applyCardViz(ref);
}

// Apply the global original/proposed visibility toggles to one card's viewer
// (the recorded boxes and the proposed overlay are independent layers).
function applyCardViz(ref) {
  const v = ref.viewer3d;
  if (!v) return;
  v.setBboxesVisible(reviewShowOriginal);
  v.setOverlayVisible(reviewShowProposed);
  cardOverlayLegend(ref, reviewShowProposed ? (ref.overlayCount ?? 0) : 0);
}

// A small magenta caption so the overlay reads as the tested proposal, not
// part of the recorded scene.
function cardOverlayLegend(ref, n) {
  if (!ref.host3d) return;
  let lg = ref.host3d.querySelector(".rv-3d-legend");
  if (n > 0) {
    if (!lg) {
      lg = el("div", { class: "rv-3d-legend",
        style: "position:absolute;top:4px;left:6px;font-size:10px;color:#ff3df5;pointer-events:none;text-shadow:0 0 3px #000;z-index:1" });
      ref.host3d.appendChild(lg);
    }
    lg.textContent = `▢ proposed (test) · ${n} box${n === 1 ? "" : "es"}`;
  } else if (lg) {
    lg.remove();
  }
}

function renderCard3d(ref) {
  updateCmpBtn(ref);
  refreshStepBtn(ref.stepBtn, ref.ev.slot, ref.ev.model);
  if (!ref.host3d) ref.host3d = el("div", { class: "rv-3d", style: "position:relative" });
  if (ref.body.firstChild !== ref.host3d) ref.body.replaceChildren(ref.host3d);
  applyCardOverlay(ref); // refresh the proposed-placement overlay (e.g. after a test lands)
  if (ref.io3d || ref.viewer3d) return; // already wired
  ref.io3d = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        if (!ref.viewer3d) {
          ref.viewer3d = createViewer(ref.host3d, { keyboard: false });
          loadCard3d(ref);
        }
        ref.viewer3d.setActive(true);
      } else {
        ref.viewer3d?.setActive(false);
      }
    }
  }, { threshold: 0.05 });
  ref.io3d.observe(ref.host3d);
}

async function loadCard3d(ref) {
  const viewer = ref.viewer3d;
  if (!viewer) return;
  const { slot, model } = ref.ev;
  try {
    const proj = await api.scene(state.run, slot, model, {});
    if (ref.viewer3d !== viewer) return; // disposed / reloaded while in flight
    viewer.clear();
    applySceneProjection(viewer, proj);
    viewer.prefetchBundle(api.meshesUrl(state.run, slot, model, {}));
    // World boxes of every placed node — lets the overlay convert a tested
    // batch step's parent-local boxes back to world.
    ref.sceneNodes = new Map((proj.nodes ?? []).map((n) => [n.id, n]));
    applyCardOverlay(ref);
  } catch { /* leave the grid cell empty — non-fatal */ }
}

function reloadCard3d(ref) {
  if (ref.viewer3d) loadCard3d(ref);
}

function teardownCard3d(ref) {
  if (ref.io3d) { ref.io3d.disconnect(); ref.io3d = null; }
  if (ref.viewer3d) { ref.viewer3d.dispose(); ref.viewer3d = null; }
}

// Exact diff of a step's rendered input (old → new), word-refined.
function diffInputSection(label, oldUser, newUser) {
  const pre = diffPre(oldUser, newUser);
  return el("div", { class: "rv-sec" },
    el("div", { class: "lab" }, el("span", { text: label }), fitToggle(pre)),
    pre,
  );
}

// Previous (logged) vs current (live test of the draft) output, side by side,
// with the at-a-glance impact signal.
function abOutputSection(previousOutput, result) {
  const changed = isChanged(result);
  const prevPre = el("pre", { text: fmtJson(previousOutput) });
  const curPre = el("pre", { text: fmtJson(result.output) });
  return el("div", { class: "rv-sec" },
    el("div", { class: "lab" },
      el("span", { text: "output — previous vs current" }),
      el("span", { class: changed ? "rv-changed" : "rv-unchanged", text: changed ? "● changed" : "○ unchanged" }),
    ),
    el("div", { class: "rv-ab" },
      el("div", {}, el("div", { class: "ab-h", text: "previous" }), prevPre, fitToggle(prevPre)),
      el("div", { class: "ab-edited" }, el("div", { class: "ab-h", text: `current · ${result.tokens_out ?? "?"} tok` }), curPre, fitToggle(curPre)),
    ),
  );
}

function isChanged(result) {
  try { return JSON.stringify(result.output) !== JSON.stringify(result.original_output); }
  catch { return true; }
}

function updateReviewTestSummary() {
  // While a batch runs, show completed/total progress — every targeted card
  // is being tested, just bounded to TEST_CONCURRENCY live at once.
  if (testBatch) {
    reviewTestSummaryEl.textContent = `testing ${testBatch.done}/${testBatch.total}…`;
    return;
  }
  let tested = 0;
  let changed = 0;
  for (const ev of state.lab.events) {
    const t = state.lab.tests.get(targetKey(ev.slot, ev.model, ev.index));
    if (t && t.status === "done") { tested += 1; if (isChanged(t.result)) changed += 1; }
  }
  reviewTestSummaryEl.textContent = tested ? `${changed}/${tested} changed by edit` : "";
}

function reviewSection(label, text) {
  const pre = el("pre", { text });
  return el("div", { class: "rv-sec" },
    el("div", { class: "lab" }, el("span", { text: label }), fitToggle(pre)),
    pre,
  );
}

function testBlock(ev, test) {
  const wrap = el("div", { class: "ev-test" });
  if (test.status === "queued") {
    wrap.appendChild(el("div", { class: "t-status", text: "queued for test…" }));
  } else if (test.status === "running") {
    wrap.appendChild(el("div", { class: "t-status", text: "testing… (live LLM call)" }));
  } else if (test.status === "error") {
    wrap.appendChild(el("div", { class: "t-status err", text: `test failed: ${test.error}` }));
  } else {
    const r = test.result;
    wrap.appendChild(el("div", { class: "t-status ok", text: `tested · ${r.tokens_out ?? "?"} tok out` }));
    const origPre = el("pre", { text: fmtJson(r.original_output) });
    const editedPre = el("pre", { text: fmtJson(r.output) });
    wrap.appendChild(
      el("div", { class: "cmp" },
        el("div", {},
          el("div", { class: "lab" }, el("span", { text: "previous output" }), fitToggle(origPre)),
          origPre),
        el("div", {},
          el("div", { class: "lab" }, el("span", { text: "current output" }), fitToggle(editedPre)),
          editedPre),
      ),
    );
    const details = el("pre", { text: `SYSTEM SENT\n${r.system}\n\nUSER SENT\n${r.user}\n\nREASONING\n${r.reasoning || "(none)"}`, style: "display:none" });
    wrap.appendChild(
      el("div", { style: "display:flex;gap:6px;margin-top:4px;align-items:center" },
        el("button", {
          style: "font-size:10px;padding:1px 7px",
          text: "sent bytes + reasoning",
          onclick: () => { details.style.display = details.style.display === "none" ? "" : "none"; },
        }),
        fitToggle(details),
      ),
    );
    wrap.appendChild(details);
  }
  return wrap;
}

function updateActionBar() {
  const lab = state.lab;
  const n = lab.selected.size;
  selCountEl.textContent = `${n} selected${lab.sims.size ? ` · ${lab.sims.size} simulating` : ""}`;
  testBtn.disabled = n === 0;
  simBtn.disabled = n === 0;
}

// --- testing ---------------------------------------------------------------------

// Push a test-state change to whichever surface is showing this event: the
// review canvas updates just the one card (no list churn); otherwise the
// events panel row re-renders.
function reflectTest(key) {
  if (reviewEl.classList.contains("open")) {
    const ref = reviewCards.get(key);
    if (ref) renderCardBody(ref);
    updateReviewTestSummary();
  } else {
    renderEvents();
  }
}

// Run the current draft against one event, storing the A/B result in the
// shared `lab.tests` (so the events panel and review canvas agree).
async function testEvent(ev) {
  const lab = state.lab;
  const d = draftFor(lab.step);
  const key = targetKey(ev.slot, ev.model, ev.index);
  lab.tests.set(key, { status: "running" });
  reflectTest(key);
  try {
    const result = await api.promptTest({
      run: state.run,
      slot: ev.slot,
      model: ev.model,
      event_index: ev.index,
      step: lab.step,
      system_template: d.system,
      user_template: d.user,
    });
    lab.tests.set(key, { status: "done", result });
  } catch (e) {
    lab.tests.set(key, { status: "error", error: e.message });
  }
  reflectTest(key);
}

// Bounded-concurrency batch tester — drives "test on selected", per-card
// "test", and "test edit on all" so a whole run's worth of slots can be
// A/B'd in one click without flooding the provider.
const TEST_CONCURRENCY = 5;
async function testEvents(events) {
  if (!events.length) return;
  const lab = state.lab;
  const queue = [...events];
  // Mark every target "queued" up front so all of them show activity now, not
  // just the first TEST_CONCURRENCY a worker grabs — and so a 9-card "test
  // all" doesn't look like it launched only 5.
  for (const ev of events) {
    const key = targetKey(ev.slot, ev.model, ev.index);
    const cur = lab.tests.get(key);
    if (!cur || cur.status === "done" || cur.status === "error") {
      lab.tests.set(key, { status: "queued" });
    }
  }
  testBatch = testBatch
    ? { total: testBatch.total + events.length, done: testBatch.done }
    : { total: events.length, done: 0 };
  testBatchRefs += 1;
  reviewTestAllEl.disabled = true;
  testBtn.disabled = true;
  if (reviewEl.classList.contains("open")) {
    for (const ev of events) {
      const ref = reviewCards.get(targetKey(ev.slot, ev.model, ev.index));
      if (ref) renderCardBody(ref);
    }
    updateReviewTestSummary();
  } else {
    renderEvents();
  }
  try {
    const workers = Array.from(
      { length: Math.min(TEST_CONCURRENCY, queue.length) },
      async () => {
        while (queue.length) {
          await testEvent(queue.shift());
          if (testBatch) testBatch.done += 1;
          if (reviewEl.classList.contains("open")) updateReviewTestSummary();
        }
      },
    );
    await Promise.all(workers);
  } finally {
    testBatchRefs -= 1;
    if (testBatchRefs === 0) testBatch = null;
    reviewTestAllEl.disabled = testBatchRefs > 0;
    updateActionBar();
    if (reviewEl.classList.contains("open")) updateReviewTestSummary();
    else renderEvents();
  }
}

async function runTests() {
  const lab = state.lab;
  const targets = lab.events.filter((ev) => lab.selected.has(targetKey(ev.slot, ev.model, ev.index)));
  await testEvents(targets);
}

// --- downstream simulation ---------------------------------------------------------

// A cell already simulating: the freshest signal is the slots poll, with the
// lab's own session record and the event payload's load-time flag as
// fallbacks (a branch may have been forked in an earlier session).
function cellHasLiveBranch(slot, model) {
  return Boolean(cellSummary(slot, model)?.branch)
    || state.lab.sims.has(cellKey(slot, model));
}

async function simulateDownstream() {
  const lab = state.lab;
  // Branches carry the lab's FULL edit set (every drafted step), so the
  // simulation matches exactly what "save to new run" would persist.
  const overrides = overridesPayload();
  // One branch per cell, forked at the EARLIEST selected event of that cell —
  // the edited templates then apply to every later firing of those steps too.
  const perCell = new Map();
  for (const ev of lab.events) {
    const key = targetKey(ev.slot, ev.model, ev.index);
    if (!lab.selected.has(key)) continue;
    const ck = cellKey(ev.slot, ev.model);
    const cur = perCell.get(ck);
    if (!cur || ev.index < cur.index) perCell.set(ck, ev);
  }
  if (perCell.size === 0) return;

  // Re-simulating a cell DISCARDS its live branch (events + branch meshes,
  // source cell untouched) and re-forks at the new branch point — never do
  // that silently.
  const conflicts = [...perCell.values()]
    .filter((ev) => cellHasLiveBranch(ev.slot, ev.model) || ev.branch_live);
  if (conflicts.length > 0) {
    openModal("replace live simulations?", (close) => ({
      body: [
        el("div", { class: "m-hint", text:
          `${conflicts.length} of the selected slot${conflicts.length === 1 ? " has" : "s have"} a live simulation branch. ` +
          "Simulating again discards that branch — its downstream events and generated meshes — " +
          "and re-forks at the new branch point with the current edits. Source cells are untouched." }),
        el("div", { class: "check-grid" },
          conflicts.map((ev) => el("label", {},
            el("span", { class: `dot ${cellSummary(ev.slot, ev.model)?.branch?.status ?? "running"}` }),
            `${ev.slot} · ${ev.model}`,
          )),
        ),
      ],
      actions: [
        el("button", { text: "cancel", onclick: close }),
        el("button", { class: "danger", text: "replace & simulate", onclick: () => {
          close();
          launchBranches(perCell, overrides);
        } }),
      ],
    }));
    return;
  }
  await launchBranches(perCell, overrides);
}

async function launchBranches(perCell, overrides) {
  const lab = state.lab;
  simBtn.disabled = true;
  let started = 0;
  for (const [ck, ev] of perCell) {
    const tkey = targetKey(ev.slot, ev.model, ev.index);
    const test = lab.tests.get(tkey);
    const seed = test?.status === "done"
      ? {
          system: test.result.system,
          user: test.result.user,
          output: test.result.output,
          reasoning: test.result.reasoning ?? "",
          tokens_in: test.result.tokens_in,
          tokens_out: test.result.tokens_out,
        }
      : null;
    try {
      await api.createBranch(state.run, ev.slot, ev.model, {
        event_index: ev.index,
        step: lab.step,
        overrides,
        seed,
      });
      lab.sims.set(ck, { slot: ev.slot, model: ev.model, eventIndex: ev.index });
      started += 1;
    } catch (e) {
      toast(`${ev.slot}·${ev.model}: ${e.message}`, "err");
    }
  }
  lab.simStep = lab.step;
  lab.simEditedSteps = Object.keys(overrides);
  simBtn.disabled = false;
  if (started > 0) {
    toast(`simulating downstream on ${started} cell${started === 1 ? "" : "s"}`, "ok");
    emit("poll-now");
    renderSims();
  }
}

function simStatus(slot, model) {
  const s = state.slots.find((x) => x.id === slot);
  return s?.runs?.[model]?.branch ?? null;
}

function renderSims() {
  const lab = state.lab;
  let host = document.getElementById("lab-sims");
  if (lab.sims.size === 0) { host?.remove(); updateActionBar(); return; }
  if (!host) {
    host = el("div", { id: "lab-sims", style: "padding:0 0 8px 0" });
    eventsEl.before(host);
  }
  host.textContent = "";
  const edited = lab.simEditedSteps?.length ? lab.simEditedSteps.join(", ") : lab.simStep;
  host.appendChild(el("div", { class: "pane-head", style: "border-top:1px solid var(--line)" },
    el("span", { text: `simulations · edits: ${edited}`, title: `branched at ${lab.simStep} events` }),
  ));
  const body = el("div", { style: "padding:4px 8px" });
  for (const sim of lab.sims.values()) {
    const b = simStatus(sim.slot, sim.model);
    const status = b?.status ?? "starting";
    const pending = b?.pending ?? null;
    const ls = b?.last_step;
    const row = el("div", { class: "sim-row" },
      el("span", { class: `dot ${pending ? "paused" : status}` }),
      el("span", { class: "cell-name", text: `${sim.slot} · ${sim.model}`,
        title: "open the branch view",
        onclick: () => emit("open-cell", { slot: sim.slot, model: sim.model, branch: true }) }),
      el("span", {
        class: "step-line",
        text: pending
          ? `awaiting step: ${pending.step} @ ${pending.node ?? "?"}`
          : ls ? `${ls.node} · ${ls.phase}` : status,
      }),
    );
    if (pending) {
      // One LLM call at a time: each press runs exactly the pending call.
      row.appendChild(el("button", { class: "primary", text: "step", onclick: () => simAction(sim, "step", row) }));
      row.appendChild(el("button", { text: "run rest", title: "finish this branch without further pauses", onclick: () => simAction(sim, "auto", row) }));
    }
    if (status === "running" && !b?.auto && !pending) {
      // Between gates: a call is in flight; the next gate appears when it lands.
      row.appendChild(el("span", { class: "muted", text: "step running…" }));
    }
    if (status === "running") {
      row.appendChild(el("button", { text: "pause", onclick: () => simAction(sim, "pause", row) }));
    } else if (status === "paused" || status === "error") {
      row.appendChild(el("button", { text: "resume", onclick: () => simAction(sim, "resume", row) }));
    }
    row.appendChild(el("button", { class: "danger", text: "break out", onclick: () => simAction(sim, "discard", row) }));
    body.appendChild(row);
  }
  host.appendChild(body);
  updateActionBar();
}

async function simAction(sim, action, row) {
  for (const b of row.querySelectorAll("button")) b.disabled = true;
  try {
    if (action === "step") await api.branchStep(state.run, sim.slot, sim.model);
    else if (action === "auto") await api.branchStep(state.run, sim.slot, sim.model, true);
    else if (action === "pause") await api.branchPause(state.run, sim.slot, sim.model);
    else if (action === "resume") await api.branchResume(state.run, sim.slot, sim.model);
    else {
      await api.branchDiscard(state.run, sim.slot, sim.model);
      state.lab.sims.delete(cellKey(sim.slot, sim.model));
      if (state.lab.sims.size === 0) state.lab.simStep = null;
    }
    emit("poll-now");
    renderSims();
  } catch (e) {
    toast(e.message, "err");
    renderSims();
  }
}

// --- persistence -----------------------------------------------------------------

// The in-place iteration loop: write the lab's edits into THIS run's
// snapshot (optionally syncing the run's source version folder), then rewind
// every slot to the edited step's first call and relaunch — so each
// edit→rerun→compare round runs across all slots in two clicks.
function applyToRunModal() {
  const lab = state.lab;
  const overrides = overridesPayload();
  const editedSteps = Object.keys(overrides);
  const hasEdits = editedSteps.length > 0;
  // Always openable: with edits it applies + (optionally) re-runs; with no
  // edits it's just the re-run-from-a-step control, so you can rewind a slot
  // to any earlier step without first having to make a throwaway change.
  const allSteps = [...lab.templates.keys()];
  const versionLabel = state.runs.find((r) => r.name === state.run)?.prompt_version;
  const syncCheck = el("input", { type: "checkbox", checked: "" });
  const rerunSel = el("select", {},
    el("option", { value: "", text: "don't re-run anything yet" }),
    hasEdits ? el("option", { value: "*", text: "re-run from earliest edited step (recommended)" }) : null,
    allSteps.map((s) => el("option", { value: s, text: `re-run from first ${s} call` })),
  );
  // With edits, default to the earliest affected step; with none, default to
  // the step currently open in the lab (the one you're likely iterating on).
  rerunSel.value = hasEdits ? "*" : (lab.step && allSteps.includes(lab.step) ? lab.step : "");

  openModal(hasEdits ? `apply edits to ${state.run}` : `re-run a step in ${state.run}`, (close, setError) => ({
    body: [
      hasEdits
        ? el("div", { class: "m-hint", text: `writes into this run's snapshot: ${editedSteps.join(", ")}` })
        : el("div", { class: "m-hint", text: "no pending edits — this just rewinds and re-runs the chosen step against the run's current prompts." }),
      hasEdits
        ? el("label", { style: "display:flex;gap:8px;align-items:center;color:var(--text-dim)" },
            syncCheck, `also update the source version (${versionLabel ?? "unknown"})`)
        : null,
      field(hasEdits ? "then" : "re-run from", rerunSel),
      el("div", { class: "m-hint", text:
        "re-running rewinds each slot to just before its first call of that step and relaunches: " +
        "everything earlier replays from the log, the step re-runs, and the downstream cascade " +
        "regenerates. Slots that never ran the step are skipped." }),
    ],
    actions: [
      el("button", { text: "cancel", onclick: close }),
      el("button", { class: "primary", text: hasEdits ? "apply" : "re-run", onclick: async () => {
        if (!hasEdits && !rerunSel.value) { setError("pick a step to re-run"); return; }
        try {
          if (hasEdits) {
            const applied = await api.updateRunPrompts(state.run, overrides, syncCheck.checked);
            let msg = `snapshot updated (${applied.applied.join(", ")})`;
            if (syncCheck.checked) {
              msg += applied.version_synced
                ? ` · version "${applied.version_synced}" synced`
                : " · source version not found, not synced";
            }
            toast(msg, "ok");
          }
          if (rerunSel.value) {
            const steps = rerunSel.value === "*" ? editedSteps : [rerunSel.value];
            const r = await api.rerunStep(state.run, steps);
            toast(`re-running from ${steps.join("/")} on ${r.rerun.length} slot${r.rerun.length === 1 ? "" : "s"}` +
              (r.skipped.length ? ` (${r.skipped.length} skipped — never ran it)` : ""), "ok");
          }
          close();
          // Applied edits become the snapshot's canonical templates — reload
          // so the edited flags clear and the event list refreshes.
          await openLab();
          emit("poll-now");
        } catch (e) { setError(e.message); }
      } }),
    ],
  }));
}

function saveVersionModal() {
  const overrides = overridesPayload();
  if (Object.keys(overrides).length === 0) {
    toast("no template edits to save", "err");
    return;
  }
  const input = el("input", { type: "text", placeholder: "e.g. baseline-tighter-bboxes" });
  openModal("save to new version", (close, setError) => ({
    body: [
      field("version name", input),
      el("div", { class: "m-hint", text: `versions/<name>/ = this run's snapshot + your edits to: ${Object.keys(overrides).join(", ")}` }),
    ],
    actions: [
      el("button", { text: "cancel", onclick: close }),
      el("button", { class: "primary", text: "save version", onclick: async () => {
        try {
          const { name } = await api.saveVersion({
            name: input.value.trim(),
            base_run: state.run,
            overrides,
          });
          lastSavedVersion = name;
          toast(`saved version "${name}"`, "ok");
          close();
        } catch (e) { setError(e.message); }
      } }),
    ],
  }));
}

function saveRunModal() {
  const lab = state.lab;
  const overrides = overridesPayload();
  if (Object.keys(overrides).length === 0) {
    toast("no template edits to save", "err");
    return;
  }
  if (lab.sims.size === 0) {
    toast("nothing simulated yet — simulate downstream first", "err");
    return;
  }
  const input = el("input", { type: "text", placeholder: "e.g. tighter-bboxes-run" });
  const cells = [...lab.sims.values()];
  openModal("save branches to new run", (close, setError) => ({
    body: [
      field("run name", input),
      el("div", { class: "m-hint",
        text: `copies ${cells.length} simulated cell${cells.length === 1 ? "" : "s"} (paused or not) into a fresh run whose prompt snapshot includes your edit — every branch resumes there seamlessly` }),
      el("div", { class: "check-grid" },
        cells.map((c) => el("label", {}, el("span", { class: "dot " + (simStatus(c.slot, c.model)?.status ?? "idle") }), `${c.slot} · ${c.model}`)),
      ),
    ],
    actions: [
      el("button", { text: "cancel", onclick: close }),
      el("button", { class: "primary", text: "save run", onclick: async () => {
        try {
          const payload = await api.saveRunFromBranches({
            name: input.value.trim(),
            base_run: state.run,
            overrides,
            cells: cells.map((c) => ({ slot: c.slot, model: c.model })),
            version_label: lastSavedVersion,
          });
          toast(`run "${payload.current}" created (${payload.copied.length} cells)`, "ok");
          close();
          labEl.classList.remove("open");
          lab.sims.clear();
          lab.simStep = null;
          emit("switch-run", payload.current);
        } catch (e) { setError(e.message); }
      } }),
    ],
  }));
}
