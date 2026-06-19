// The prompt lab: pick a template step, edit it (variables stay tokens —
// scene context never expands here), test the edit against logged events
// across cells, simulate downstream in parallel branches, and persist the
// result as a new version and/or a new run.

import { api } from "./api.js";
import { state, emit, on, cellKey, targetKey, cellSummary, cellBranches, branchSummaryById } from "./state.js";
import { el, fitToggle, toast, openModal, field, fmtJson, diffPre } from "./ui.js";
import { createViewer } from "./scene3d.js";
import { applySceneProjection } from "./events.js";
import { statusView } from "./status.js";

const labEl = document.getElementById("lab");
const runEl = document.getElementById("lab-run");
const stepsEl = document.getElementById("lab-steps");
const varsEl = document.getElementById("lab-vars");
const sysEl = document.getElementById("lab-sys");
const usrEl = document.getElementById("lab-usr");
const selectBtn = document.getElementById("lab-select");
const selSummaryEl = document.getElementById("lab-sel-summary");
const selCountEl = document.getElementById("lab-sel-count");
const testBtn = document.getElementById("lab-test");
const simBtn = document.getElementById("lab-simulate");
// The debug grid lives in the lab's right pane (#lab-right) — the review
// canvas, relocated. It's live whenever the lab is open, scoped to the
// selected slots/zones, and toggles between 3D scenes and text output diffs.
// The `review*` identifiers below name the card-rendering machinery it reuses
// verbatim; there is no longer a separate review screen.
const reviewGridEl = document.getElementById("lab-grid");
const reviewStepEl = document.getElementById("lab-grid-step");
const reviewCountEl = document.getElementById("lab-grid-count");
const reviewSectionsEl = document.getElementById("lab-sections");
const reviewColsEl = document.getElementById("lab-cols");
const reviewTestSummaryEl = document.getElementById("lab-test-summary");
const labStepAllEl = document.getElementById("lab-step-all");

// Which text sections the review cards show, and how many cards per row —
// the comparison view's "fill the space with what I care about" controls.
const REVIEW_SHOW_KEY = "starshot.reviewShow";
const REVIEW_COLS_KEY = "starshot.reviewCols";
const REVIEW_3D_KEY = "starshot.review3d";
const REVIEW_SECTIONS = ["input", "output", "system", "reasoning"];
const reviewShow = loadReviewShow();
let reviewCols = loadReviewCols();
// "Convert to 3D": each card shows its cell's scene (one live viewer per card)
// instead of text — the source, or its simulation branch when one is live, so
// a prompt's effect can be eyeballed across every slot side by side. Before/
// after comparison lives on the dedicated compare screen (per-card "compare
// 3D ⇄"), not as an overlay on this canvas.
let review3d = (() => { try { return localStorage.getItem(REVIEW_3D_KEY) !== "0"; } catch { return true; } })();

// Progress of the in-flight test batch(es), so the summary reads honestly
// ("testing 3/9…") instead of the live-concurrency count, which looked like
// only TEST_CONCURRENCY tests had been launched. Survives nested batches
// (a per-card "test" fired mid-"test all") via the ref count.
let testBatch = null; // { total, done } | null when idle
let testBatchRefs = 0;
// Bumped on every step (re)selection. An in-flight batch captures it and bails
// the moment it changes, so tests from the old step can't re-populate lab.tests
// or skew the new step's progress count.
let testSeq = 0;

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

// The debug-grid selection is per-run; `selInitialized` lets us seed it once
// (to every started cell) on first open of a run, then leave the user's
// choices alone — including a deliberate "select nothing".
let selRun = null;
let selInitialized = false;

export function initLab() {
  document.getElementById("btn-lab").addEventListener("click", openLab);
  document.getElementById("lab-close").addEventListener("click", () => {
    teardownGrid();
    labEl.classList.remove("open");
  });
  // Template editors expand to fit their full text (and keep tracking it
  // while editing) via the shared control.
  document.getElementById("lab-sys-label").appendChild(fitToggle(sysEl));
  document.getElementById("lab-usr-label").appendChild(fitToggle(usrEl));
  document.getElementById("lab-ev-refresh").addEventListener("click", () => loadEvents());
  selectBtn.addEventListener("click", openSelectionDialog);
  reviewColsEl.value = String(reviewCols);
  reviewColsEl.addEventListener("change", () => {
    reviewCols = Number(reviewColsEl.value) || 2;
    try { localStorage.setItem(REVIEW_COLS_KEY, String(reviewCols)); } catch { /* private mode */ }
    applyReviewCols();
  });
  labStepAllEl.addEventListener("click", stepAllCells);
  labStepAllEl.after(labUntilEl);
  labUntilEl.after(labExitEl);
  document.getElementById("lab-apply").addEventListener("click", applyToRunModal);
  document.getElementById("lab-save-version").addEventListener("click", saveVersionModal);
  document.getElementById("lab-save-run").addEventListener("click", saveRunModal);
  // The two action-bar buttons are contextual: test/simulate when the selection
  // is idle, step-sims/break-out once it's simulating (see updateActionBar).
  // The "step sims until…" select sits between them, shown only in branch mode.
  testBtn.addEventListener("click", () => {
    if (selectedBranches().length > 0) stepSelectedBranches();
    else testEvents(selectedEvents());
  });
  testBtn.after(simUntilEl);
  simBtn.addEventListener("click", () => {
    if (selectedBranches().length > 0) breakOutSelected();
    else simulateDownstream();
  });
  sysEl.addEventListener("input", onEdit);
  usrEl.addEventListener("input", onEdit);
  initLabResizer();
  on("slots", () => {
    if (!labEl.classList.contains("open")) return;
    pruneStaleSims(); // forget branches the server dropped (source rewound/reset elsewhere)
    renderSims();
    updateStepAllButtons();
    refreshCardStepBtns(); // keep card step buttons live between event reloads
    refreshCardBranchBtns(); // keep per-card simulate/compare buttons tracking each cell's branch
    refreshCardScenes(); // swap a card to its branch scene + follow it as it runs
    pollEvents(); // auto-pick up newly-completed events in the grid
  });
  // The overlay's "+ sim" zone button feeds targets here, so you can pick a zone
  // off the 3D scene instead of hunting for it in the step's call list.
  on("add-sim-target", addSimTarget);
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
  // Variable hover samples are run-independent (a fixed sample scene rendered by
  // the real injection functions) — fetch once, lazily, non-fatally.
  if (varSamples === null) {
    try { varSamples = await api.variableSamples(); }
    catch { varSamples = {}; }
  }
  // The slot/zone selection belongs to a run; on a run switch re-seed it from
  // that run's events on the next load (the session reset already cleared it).
  if (selRun !== state.run) { selRun = state.run; selInitialized = false; }
  lab.open = true;
  labEl.classList.add("open");
  renderReviewControls();
  applyReviewCols();
  renderSteps();
  updateSelectionSummary();
  if (!lab.step) selectStep(lab.templates.keys().next().value);
  else selectStep(lab.step, { keepSelection: true });
  updateStepAllButtons();
  renderSims();
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

export function overridesPayload() {
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

// --- variable hover preview ---------------------------------------------------
//
// Hovering a variable chip shows what that token actually expands to: a sample
// rendered SERVER-SIDE by the same scene_context injection functions the live
// pipeline uses (fetched once into `varSamples`), so a missing-context or
// rendering bug surfaces here without running a scene. The popover bridges hover
// to itself so big samples (SCENE_CONTEXT, TO_PLACE) can be scrolled.

let varSamples = null;
let varPreviewEl = null;
let varPreviewHideTimer = null;

function ensureVarPreview() {
  if (varPreviewEl) return varPreviewEl;
  varPreviewEl = el("div", { class: "var-preview" });
  varPreviewEl.addEventListener("mouseenter", () => clearTimeout(varPreviewHideTimer));
  varPreviewEl.addEventListener("mouseleave", scheduleHideVarPreview);
  document.body.appendChild(varPreviewEl);
  return varPreviewEl;
}

function scheduleHideVarPreview() {
  clearTimeout(varPreviewHideTimer);
  varPreviewHideTimer = setTimeout(() => varPreviewEl?.classList.remove("open"), 160);
}

function hideVarPreview() {
  clearTimeout(varPreviewHideTimer);
  varPreviewEl?.classList.remove("open");
}

function showVarPreview(chip, name, isNative) {
  const pop = ensureVarPreview();
  clearTimeout(varPreviewHideTimer);
  const sample = varSamples?.[name];
  pop.textContent = "";
  pop.appendChild(el("div", { class: "var-preview-head" },
    el("span", { class: "var-preview-name", text: `\`{${name}}\`` }),
    el("span", { class: "var-preview-note", text: isNative ? "sample render" : "sample render · not populated for this step" }),
  ));
  pop.appendChild(el("pre", {
    class: "var-preview-body",
    text: sample && sample.trim()
      ? sample
      : (varSamples ? "(renders empty for the sample scene)" : "(sample unavailable)"),
  }));
  pop.appendChild(el("div", { class: "var-preview-foot", text: "click the chip to insert it at the user-template cursor" }));
  pop.classList.add("open");
  // Position under the chip, clamped to the viewport; flip above if it'd overflow.
  const r = chip.getBoundingClientRect();
  const left = Math.max(8, Math.min(r.left, window.innerWidth - pop.offsetWidth - 12));
  let top = r.bottom + 6;
  if (top + pop.offsetHeight > window.innerHeight - 8) top = Math.max(8, r.top - 6 - pop.offsetHeight);
  pop.style.left = `${Math.round(left)}px`;
  pop.style.top = `${Math.round(top)}px`;
}

function selectStep(step, { keepSelection = false } = {}) {
  const lab = state.lab;
  lab.step = step;
  // A real step change abandons the previous step's in-flight card fetches and
  // its per-event test results (the events differ). The slot/zone selection is
  // step-independent, so it persists across the switch.
  reviewSeq += 1;
  // Invalidate any running test batch — its events belong to the old step — and
  // drop its progress so the new step doesn't show a phantom "testing N/M…".
  testSeq += 1;
  testBatch = null;
  if (!keepSelection) lab.tests.clear();
  const t = lab.templates.get(step);
  const d = draftFor(step);
  sysEl.value = d.system;
  usrEl.value = d.user;
  // Programmatic value swaps don't fire `input` — nudge the fit controls so
  // an expanded editor re-sizes to the newly-loaded template.
  sysEl.dispatchEvent(new Event("input"));
  usrEl.dispatchEvent(new Event("input"));
  varsEl.textContent = "";
  hideVarPreview();
  // The full vocabulary is injectable everywhere; chips outside this step's
  // native set are dimmed — they resolve to empty/placeholder at call time.
  // Hovering a chip previews its real rendered sample (see showVarPreview).
  const native = new Set(t.native ?? t.variables);
  for (const v of t.variables) {
    const isNative = native.has(v);
    const chip = el("span", {
      class: `var-chip${isNative ? "" : " dim"}`,
      text: `\`{${v}}\``,
      onclick: () => insertAtCursor(usrEl, `\`{${v}}\``),
    });
    chip.addEventListener("mouseenter", () => showVarPreview(chip, v, isNative));
    chip.addEventListener("mouseleave", scheduleHideVarPreview);
    varsEl.appendChild(chip);
  }
  renderSteps();
  reviewStepEl.textContent = step; // grid header tracks the open step
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

// --- selection: which slots (cells) and zones the debug grid shows ---------------
//
// Selection is scoped PER cell: `lab.selection` maps a cellKey to the set of
// node ids (zones) chosen for it — an EMPTY set means "every zone of this
// cell". This is per-slot, so zone A of one slot and zone B of another can be
// picked independently (no forced cross-product). Cell + node identities are
// step-independent, so a choice survives a step switch — the grid just
// re-materializes whichever of the new step's calls fall in each cell's chosen
// zones. `selectedEvents()` is the live materialization for the open step;
// every surface (grid, test, simulate, counts) reads from it.

let eventsSig = null;
let eventPollInFlight = false;

function selectedEvents() {
  const sel = state.lab.selection;
  return state.lab.events.filter((e) => {
    const zones = sel.get(cellKey(e.slot, e.model));
    if (!zones) return false;                     // cell not selected
    return zones.size === 0 || zones.has(e.node); // empty set ⇒ all zones
  });
}

// The distinct cells in the current selection. Once any selected cell has a
// live simulation branch, the action bar swaps from test/simulate to branch
// controls (step the sims / break out).
function selectedCellList() {
  const seen = new Set();
  const out = [];
  for (const ev of selectedEvents()) {
    const ck = cellKey(ev.slot, ev.model);
    if (seen.has(ck)) continue;
    seen.add(ck);
    out.push({ slot: ev.slot, model: ev.model });
  }
  return out;
}
// Every TOP-LEVEL simulation branch of the selected cells (a cell can carry
// several — different zones / parallel sims), as server-truth summaries.
function selectedBranches() {
  const out = [];
  for (const c of selectedCellList()) out.push(...cellBranches(c.slot, c.model));
  return out;
}

// On first open of a run, seed the selection to every started cell (so the
// grid isn't empty); afterwards respect the user's choice (including "none").
// Cells are step-independent, so this is sourced from the run's slots rather
// than the current step's events.
function ensureDefaultSelection() {
  const lab = state.lab;
  if (selInitialized) return;
  const started = [];
  for (const s of state.slots) {
    for (const m of state.models) {
      const c = s.runs?.[m];
      if (c && ((c.events_count ?? 0) > 0 || c.branches?.length)) started.push(cellKey(s.id, m));
    }
  }
  if (!started.length) return; // nothing started yet — seed on a later render
  selInitialized = true;
  if (lab.selection.size === 0) for (const k of started) lab.selection.set(k, new Set()); // all started cells, all zones
}

function updateSelectionSummary() {
  const sel = state.lab.selection;
  const nc = sel.size;
  if (!nc) { selSummaryEl.textContent = "no slots selected"; return; }
  let scoped = 0; // slots narrowed to specific zones
  for (const zones of sel.values()) if (zones.size) scoped += 1;
  selSummaryEl.textContent = `${nc} slot${nc === 1 ? "" : "s"} · ${scoped ? `${scoped} zone-filtered` : "all zones"}`;
}

// Add a zone picked off the 3D scene (overlay "+ sim") to the simulation slots.
// Narrows that cell to the chosen zone(s): an unselected or all-zones cell
// becomes "just this zone", an already-zone-filtered cell accumulates it — you
// build up a precise target list straight from the scene.
function addSimTarget({ slot, model, node }) {
  if (!slot || !model || !node) return;
  const lab = state.lab;
  // Claim this run's selection so opening the lab later doesn't reseed over it.
  selRun = state.run;
  selInitialized = true;
  const ck = cellKey(slot, model);
  let zones = lab.selection.get(ck);
  if (!zones || zones.size === 0) { zones = new Set(); lab.selection.set(ck, zones); }
  zones.add(node);
  updateSelectionSummary();
  if (labEl.classList.contains("open")) renderGrid();
  toast(`added ${node} (${slot} · ${model}) to simulation slots`, "ok");
}

// The top-bar picker: a per-slot tree. Each started cell is a row you select;
// expanding it narrows that slot to specific zones of the current step
// (default: all zones). Because narrowing is per slot, zone A of one slot and
// zone B of another are independently pickable. Edits stage in a working copy
// (cellKey -> Set<node id>) and commit on "apply".
function openSelectionDialog() {
  const lab = state.lab;
  const cells = [];
  for (const s of state.slots) {
    for (const m of state.models) {
      const c = s.runs?.[m];
      if (c && ((c.events_count ?? 0) > 0 || c.branches?.length)) {
        cells.push({ slot: s.id, model: m, status: c.status ?? "idle", events: c.events_count ?? 0 });
      }
    }
  }
  // This step's distinct zones (node ids) per cell — the calls a slot can be
  // narrowed to right now.
  const stepZonesByCell = new Map();
  for (const e of lab.events) {
    if (!e.node) continue;
    const ck = cellKey(e.slot, e.model);
    if (!stepZonesByCell.has(ck)) stepZonesByCell.set(ck, new Set());
    stepZonesByCell.get(ck).add(e.node);
  }
  // Staging copy of the selection, plus which slot blocks are expanded. Open
  // any slot that already carries a specific zone filter so it's visible.
  const pick = new Map();
  for (const [ck, zones] of lab.selection) pick.set(ck, new Set(zones));
  const expanded = new Set();
  for (const [ck, zones] of pick) if (zones.size) expanded.add(ck);

  const slotFilter = el("input", { type: "text", placeholder: "filter slots by name…", style: "flex:1;min-width:80px" });
  const modelSel = el("select", { style: "max-width:130px" },
    el("option", { value: "", text: "all models" }),
    [...new Set(cells.map((c) => c.model))].sort().map((m) => el("option", { value: m, text: m })));
  // Filter the per-slot zone lists by name — typing auto-expands the slots that
  // have a matching zone, so a named zone is reachable without expanding each.
  const zoneFilter = el("input", { type: "text", placeholder: "filter zones by name…", style: "width:100%" });
  const list = el("div", { class: "sel-tree" });

  const visibleCells = () => cells.filter((c) =>
    (!modelSel.value || c.model === modelSel.value) &&
    (!slotFilter.value.trim() || c.slot.toLowerCase().includes(slotFilter.value.trim().toLowerCase())));

  function render() {
    const top = list.scrollTop;
    list.textContent = "";
    const vis = visibleCells();
    if (!vis.length) { list.appendChild(el("div", { class: "muted", text: "no started slots match." })); return; }
    for (const c of vis) list.appendChild(renderBlock(c));
    list.scrollTop = top;
  }

  function renderBlock(c) {
    const ck = cellKey(c.slot, c.model);
    const sel = pick.get(ck);                 // undefined ⇒ not selected
    const selected = sel !== undefined;
    const specific = selected && sel.size > 0; // narrowed to a zone subset
    // Show every zone this step calls PLUS any already-picked zone (so a filter
    // set at another step is still visible and removable here).
    const stepZones = stepZonesByCell.get(ck) ?? new Set();
    const allZones = [...new Set([...stepZones, ...(sel ?? [])])].sort();
    const zq = zoneFilter.value.trim().toLowerCase();
    const zonesShown = zq ? allZones.filter((z) => z.toLowerCase().includes(zq)) : allZones;
    // A zone-name query force-opens the slots that have a match.
    const open = expanded.has(ck) || (zq !== "" && zonesShown.length > 0);

    const slotCb = el("input", { type: "checkbox", ...(selected ? { checked: "" } : {}) });
    slotCb.addEventListener("change", () => {
      if (slotCb.checked) pick.set(ck, new Set());
      else pick.delete(ck);
      render();
    });
    const summary = specific ? `${sel.size} zone${sel.size === 1 ? "" : "s"}` : (selected ? "all zones" : "off");
    const caret = el("button", {
      class: "fit-btn",
      text: `${open ? "▾" : "▸"} ${summary}`,
      title: "expand to narrow this slot to specific zones of this step",
      onclick: () => { if (open) expanded.delete(ck); else expanded.add(ck); render(); },
    });
    const header = el("div", { class: "sel-slot" },
      slotCb,
      el("span", { class: `dot ${c.status}` }),
      el("span", { text: `${c.slot} · ${c.model}` }),
      el("span", { class: "muted", style: "margin-left:auto", text: `${c.events} ev` }),
      caret);
    const block = el("div", { class: `sel-block${selected ? " on" : ""}` }, header);

    if (open) {
      const sub = el("div", { class: "sel-zones" });
      if (!allZones.length) {
        sub.appendChild(el("div", { class: "muted", text: "no per-zone calls of this step for this slot." }));
      } else if (!zonesShown.length) {
        sub.appendChild(el("div", { class: "muted", text: `no zones match “${zoneFilter.value.trim()}”.` }));
      } else {
        if (!zq) { // "all zones" reset is moot while filtering by name
          const allCb = el("input", { type: "checkbox", ...(selected && !specific ? { checked: "" } : {}) });
          allCb.addEventListener("change", () => { pick.set(ck, new Set()); render(); }); // back to all zones
          sub.appendChild(el("label", { class: "sel-all" }, allCb, el("span", { text: "all zones" })));
        }
        for (const z of zonesShown) {
          const on = specific && sel.has(z);
          const zb = el("input", { type: "checkbox", ...(on ? { checked: "" } : {}) });
          zb.addEventListener("change", () => {
            let set = pick.get(ck);
            if (!set) { set = new Set(); pick.set(ck, set); } // checking a zone selects the slot
            if (zb.checked) set.add(z); else set.delete(z);
            render();
          });
          sub.appendChild(el("label", {}, zb, el("span", { text: z }),
            stepZones.has(z) ? null
              : el("span", { class: "muted", style: "margin-left:auto;font-size:10px", text: "no call this step" })));
        }
      }
      block.appendChild(sub);
    }
    return block;
  }

  slotFilter.addEventListener("input", render);
  modelSel.addEventListener("change", render);
  zoneFilter.addEventListener("input", render);
  const selectAll = () => { for (const c of visibleCells()) { const ck = cellKey(c.slot, c.model); if (!pick.has(ck)) pick.set(ck, new Set()); } render(); };
  const selectNone = () => { for (const c of visibleCells()) pick.delete(cellKey(c.slot, c.model)); render(); };

  openModal("select slots / zones to iterate on", (close) => {
    render();
    return {
      body: [
        el("div", { class: "m-hint", text: "Pick slots to iterate on; expand a slot to narrow it to specific zones of this step (default: all zones). Or add zones straight off the 3D scene with “+ sim”. The choice persists as you switch steps and shows as live canvases on the right." }),
        el("div", { class: "m-field sel-field" },
          el("div", { style: "display:flex;gap:6px;align-items:center" }, slotFilter, modelSel,
            el("button", { class: "fit-btn", text: "all", onclick: selectAll }),
            el("button", { class: "fit-btn", text: "none", onclick: selectNone })),
          zoneFilter,
          list),
      ],
      actions: [
        el("button", { text: "cancel", onclick: close }),
        el("button", { class: "primary", text: "apply", onclick: () => {
          lab.selection = pick;
          selInitialized = true;
          close();
          updateSelectionSummary();
          renderGrid();
        } }),
      ],
    };
  });
}

function eventsSignature(events) {
  return events
    .map((e) => `${e.slot}|${e.model}|${e.index}|${e.branch_live ? 1 : 0}|${e.output_preview ?? ""}`)
    .join("\n");
}

// `silent` = the auto-refresh path: never shows a loading state, and only
// touches the DOM when the event set actually changed (so it won't reset
// scroll or test state on every poll).
async function loadEvents({ silent = false } = {}) {
  const lab = state.lab;
  if (!lab.step) return;
  if (!silent) reviewCountEl.textContent = "…";
  let payload;
  try {
    payload = await api.stepEvents(state.run, lab.step);
  } catch (e) {
    if (!silent) { teardownGrid(); reviewCountEl.textContent = ""; gridMessage(`failed to load events: ${e.message}`); }
    return;
  }
  // The lab may have moved on while the request was in flight (step switch,
  // run switch, closed) — discard a stale response.
  if (!labEl.classList.contains("open") || lab.step !== payload.step) return;
  lab.events = payload.events;
  // Drop tests whose event no longer exists (run reset, step switch, etc.).
  const valid = new Set(lab.events.map((e) => targetKey(e.slot, e.model, e.index)));
  for (const k of [...lab.tests.keys()]) if (!valid.has(k)) lab.tests.delete(k);
  const sig = eventsSignature(lab.events);
  if (silent && sig === eventsSig) return; // unchanged — leave the grid alone
  eventsSig = sig;
  renderGrid();
}

async function pollEvents() {
  // Tied to the run's slots poll; guard against overlap on a slow request.
  if (eventPollInFlight || !labEl.classList.contains("open") || !state.lab.step) return;
  eventPollInFlight = true;
  try { await loadEvents({ silent: true }); } finally { eventPollInFlight = false; }
}

// --- stepping source cells (drive the live run from the lab) --------------------
//
// "step" / "step all" advance the actual run cells in step mode (api.cellStep /
// api.stepAll), then jump the lab (and its debug grid) to the prompt of the
// step that just ran, so you can walk the pipeline one LLM call at a time and
// inspect each prompt as it fires.

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
// of a target step, pause them all there, and jump the lab to it. The option
// list is filled lazily once `state.steps` loads.
function makeUntilSelect() {
  const sel = el("select", { class: "step-until", style: "display:none",
    title: "fast-forward every stepped cell to the next run of a step" },
    el("option", { value: "", text: "all until…" }));
  sel.addEventListener("change", () => { const v = sel.value; sel.value = ""; if (v) stepAllUntil(v); });
  return sel;
}
const labUntilEl = makeUntilSelect();

function makeExitBtn() {
  const b = el("button", { class: "ev-scene-btn", style: "display:none", text: "exit stepping",
    title: "run every stepped cell to completion and leave step mode" });
  b.addEventListener("click", exitStepping);
  return b;
}
const labExitEl = makeExitBtn();

// "step sims until <step>" — fast-forward every selected simulation branch to
// the next call of a target step, pausing each there (the branch mirror of
// "step all until"). Shown in the action bar only while the selection is
// simulating; option list filled lazily in updateActionBar.
function makeSimUntilSelect() {
  const sel = el("select", { class: "step-until", style: "display:none",
    title: "fast-forward every selected simulation branch to the next call of a step" },
    el("option", { value: "", text: "step until…" }));
  sel.addEventListener("change", () => { const v = sel.value; sel.value = ""; if (v) stepSelectedBranches(v); });
  return sel;
}
const simUntilEl = makeSimUntilSelect();

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
  labStepAllEl.style.display = n ? "" : "none";
  labStepAllEl.textContent = `step all (${n})`;
  if (state.steps.length && labUntilEl.options.length <= 1) {
    for (const s of state.steps) labUntilEl.appendChild(el("option", { value: s, text: `▸ ${s}` }));
  }
  labUntilEl.style.display = (n && state.steps.length) ? "" : "none";
  labExitEl.style.display = n ? "" : "none";
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

function refreshCardStepBtns() {
  for (const ref of reviewCards.values()) refreshStepBtn(ref.stepBtn, ref.ev.slot, ref.ev.model);
}

// Jump the lab (and its debug grid) to a step's prompt.
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
  // Advance the active simulation branches in lockstep with the source cells, so
  // one "step all" moves BOTH sides of the 3D compare forward together instead
  // of leaving the original behind. Best-effort per branch — one that's done or
  // mid-call simply isn't advanced.
  const sims = [...state.lab.sims.values()];
  if (sims.length) {
    await Promise.allSettled(sims.map((s) => api.branchStep(s.id)));
  }
  if (pendings.size === 1) navigateToStep([...pendings][0]);
  emit("poll-now");
}

// --- debug grid: the selected slots' scenes / output diffs, side by side --------
//
// Reconciles against selectedEvents(): newly-completed events get cards (full
// bytes fetched per-card), re-run events refetch, vanished/deselected events
// drop — without tearing down the grid, so per-card show/hide/expand state and
// the scroll position survive the silent auto-refresh.

let reviewSeq = 0;
const reviewCards = new Map(); // key -> { card, body, ev, full, preview }

// Tear the grid down completely (lab close / load error): abandon in-flight
// fills and release every card's WebGL context.
function teardownGrid() {
  reviewSeq += 1;
  for (const ref of reviewCards.values()) teardownCard3d(ref);
  reviewCards.clear();
  reviewGridEl.textContent = "";
}

// A single full-width message in the empty grid (no selection / no events).
function gridMessage(text) {
  let m = reviewGridEl.querySelector(".grid-empty");
  if (!m) { m = el("div", { class: "grid-empty" }); reviewGridEl.appendChild(m); }
  m.textContent = text;
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
    title: "convert every card to its cell's 3D scene (source, or its simulation branch) — compare across slots side by side",
    onclick: () => setReview3d(!review3d),
  }));
}

function setReview3d(on) {
  review3d = on;
  try { localStorage.setItem(REVIEW_3D_KEY, on ? "1" : "0"); } catch { /* private mode */ }
  renderReviewControls();
  // The card set + granularity differ by mode (3D = one canvas per cell, text =
  // one card per zone-event), so rebuild rather than re-skin the existing cards.
  teardownGrid();
  renderGrid();
}

function renderGrid() {
  ensureDefaultSelection();
  const lab = state.lab;
  const seq = reviewSeq;
  const all = selectedEvents();
  // 3D shows ONE canvas per cell — the scene is per slot×model, not per zone —
  // so dedup to the first event of each cell. Otherwise a multi-zone cell spins
  // up N identical viewers (redundant, and enough of them exhaust the browser's
  // WebGL contexts). Text mode keeps one card per zone-event (distinct diffs).
  const zonesPerCell = new Map();
  for (const ev of all) {
    const ck = cellKey(ev.slot, ev.model);
    zonesPerCell.set(ck, (zonesPerCell.get(ck) ?? 0) + 1);
  }
  let events = all;
  if (review3d) {
    const byCell = new Map();
    for (const ev of all) {
      const ck = cellKey(ev.slot, ev.model);
      if (!byCell.has(ck)) byCell.set(ck, ev);
    }
    events = [...byCell.values()];
  }
  const wanted = new Set(events.map((e) => targetKey(e.slot, e.model, e.index)));
  for (const [key, ref] of [...reviewCards]) {
    if (!wanted.has(key)) { teardownCard3d(ref); ref.card.remove(); reviewCards.delete(key); }
  }
  if (events.length === 0) {
    gridMessage(lab.selection.size === 0
      ? "no slots selected — use “select slots / zones…” in the top bar to choose which to iterate on."
      : "the selected slots have no calls of this step in their chosen zones.");
  } else {
    reviewGridEl.querySelector(".grid-empty")?.remove();
    for (const ev of events) {
      const key = targetKey(ev.slot, ev.model, ev.index);
      const zoneCount = review3d ? (zonesPerCell.get(cellKey(ev.slot, ev.model)) ?? 1) : 1;
      let ref = reviewCards.get(key);
      if (!ref) {
        ref = reviewCard(ev, zoneCount);
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
      refreshBranchBtns(ref);
      refreshStepBtn(ref.stepBtn, ev.slot, ev.model);
    }
  }
  reviewCountEl.textContent = `${events.length} card${events.length === 1 ? "" : "s"}`;
  updateSelectionSummary();
  updateActionBar();
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

function reviewCard(ev, zoneCount = 1) {
  const body = el("div", { class: "rv-body" }, el("div", { class: "rv-loading", text: "loading…" }));
  const cmpBtn = el("button", {
    class: "ev-scene-btn",
    style: "display:none",
    text: "compare 3D ⇄",
    title: "live run vs simulated-edit branch, side by side in 3D",
    onclick: () => {
      const b = cardBranch(ev.slot, ev.model, ev.index);
      emit("open-compare", {
        slot: ev.slot, model: ev.model, step: state.lab.step, node: ev.node, index: ev.index,
        branch: b?.id ?? null,
      });
    },
  });
  // "simulate" — fork a downstream branch from THIS call alone, so a single
  // slot can be simulated on its own (and added to a set that's already
  // running), not only via the action-bar batch. Hidden once the cell is
  // branched (refreshBranchBtns), where "compare 3D" + the sims list take over.
  const simCellBtn = el("button", {
    class: "ev-scene-btn",
    style: "display:none",
    text: "simulate",
    title: "fork a downstream simulation branch from this call — starts on its own, even while other slots are simulating",
    onclick: (e) => simulateCell(ev, e.currentTarget),
  });
  const stepBtn = makeStepBtn(ev.slot, ev.model);
  const card = el("div", { class: "rv-card" },
    el("div", { class: "rv-head" },
      el("span", { class: "rv-slot", text: ev.slot }),
      el("span", { class: "rv-meta", text: ev.model }),
      // 3D card stands for the whole cell (one scene); show its zone count.
      // Text card is per zone-event; show that zone + log index.
      el("span", { class: "rv-meta", text: zoneCount > 1 ? `${zoneCount} zones` : `${ev.node ?? "?"} · #${ev.index}` }),
      el("button", {
        class: "ev-scene-btn",
        style: "margin-left:auto",
        text: "test",
        title: "run the current prompt edit on this event",
        onclick: () => testEvents([ev]),
      }),
      simCellBtn,
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
    card, body, cmpBtn, simCellBtn, stepBtn, ev, full: null, preview: ev.output_preview,
    // 3D-grid mode: lazily-created per-card viewer + its viewport observer.
    host3d: null, viewer3d: null, io3d: null,
  };
}

// A card's two branch-dependent buttons are complements: "compare 3D" needs a
// live branch (it's the "current" side), while "simulate" only makes sense when
// there ISN'T one yet — one branch per slot, so you break out to re-simulate.
// Toggling both here keeps the per-card "start a sim on its own" affordance in
// lockstep with the cell's live-branch state, in every place a card re-renders.
// The TOP-LEVEL simulation branch forked from THIS card's call (its origin cell
// + fork event match) — a card maps to its own-zone sim. The most recent wins
// if the same event was simulated more than once. `null` when this call hasn't
// been simulated.
function cardBranch(slot, model, eventIndex) {
  const list = cellBranches(slot, model).filter((b) => b.fork_index === eventIndex);
  return list.length ? list[list.length - 1] : null;
}

function refreshBranchBtns(ref) {
  const b = cardBranch(ref.ev.slot, ref.ev.model, ref.ev.index);
  ref.branchId = b?.id ?? null;
  // "compare 3D" needs this card's branch; "simulate" is always available now
  // (fork another sim of this call any time).
  ref.cmpBtn.style.display = b ? "" : "none";
  ref.simCellBtn.style.display = "";
}

function refreshCardBranchBtns() {
  for (const ref of reviewCards.values()) refreshBranchBtns(ref);
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
  refreshBranchBtns(ref);
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
// One live viewer per card showing that cell's scene — its simulation BRANCH
// when one is live (so a downstream simulation's result shows here, evolving as
// the branch runs), otherwise the source cell. Viewers are created lazily when
// a card scrolls into view (and paused when it leaves) so a big grid doesn't
// open dozens of WebGL contexts at once; keyboard nav is off so a keypress
// can't drive every canvas. `dispose()` reclaims the context when the card
// leaves 3D mode / is removed / the grid is torn down.
//
// "test edits on selected" is a cheap single-call preview surfaced as the text
// output diff; "simulate downstream" forks a branch and runs it, and the card
// switches to that branch's scene (the real, full result). Before/after 3D
// comparison is the dedicated compare screen's job — not an overlay here.

// A corner caption so a card reads clearly as the simulation branch (the
// downstream-simulated result) rather than the source scene.
function cardSceneBadge(ref, branched) {
  if (!ref.host3d) return;
  let b = ref.host3d.querySelector(".rv-3d-branch");
  if (branched) {
    if (!b) {
      b = el("div", { class: "rv-3d-branch",
        style: "position:absolute;top:4px;right:6px;font-size:10px;color:var(--purple);pointer-events:none;text-shadow:0 0 3px #000;z-index:1" });
      ref.host3d.appendChild(b);
    }
    b.textContent = "▸ sim branch";
  } else if (b) {
    b.remove();
  }
}

function renderCard3d(ref) {
  refreshBranchBtns(ref);
  refreshStepBtn(ref.stepBtn, ref.ev.slot, ref.ev.model);
  if (!ref.host3d) ref.host3d = el("div", { class: "rv-3d", style: "position:relative" });
  if (ref.body.firstChild !== ref.host3d) ref.body.replaceChildren(ref.host3d);
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

// What a card's scene should currently show: its own simulation branch (when
// this call has been simulated) vs the source cell, and how far that side has
// progressed — so a poll reloads the canvas as the branch runs (or is
// discarded), not only when the source event's output changes.
function cardSceneState(ref) {
  const { slot, model, index } = ref.ev;
  const b = cardBranch(slot, model, index);
  if (b) return { side: "branch", id: b.id, count: b.events_count ?? 0, status: b.status ?? "starting" };
  const c = cellSummary(slot, model);
  return { side: "source", id: null, count: c?.events_count ?? 0, status: c?.status ?? "idle" };
}

async function loadCard3d(ref) {
  const viewer = ref.viewer3d;
  if (!viewer) return;
  const { slot, model } = ref.ev;
  // Pull this card's branch scene + meshes when it has one, so "simulate"
  // shows the branched result here. Stamp the side/progress we're loading so
  // refreshCardScenes can tell when a reload is due.
  const st = cardSceneState(ref);
  const branched = st.side === "branch";
  ref.sceneState = st;
  ref.lastSceneLoad = performance.now();
  try {
    const proj = branched
      ? await api.branchScene(st.id)
      : await api.scene(state.run, slot, model, {});
    if (ref.viewer3d !== viewer) return; // disposed / reloaded while in flight
    viewer.clear();
    applySceneProjection(viewer, proj);
    viewer.prefetchBundle(branched ? api.branchMeshesUrl(st.id) : api.meshesUrl(state.run, slot, model, {}));
    cardSceneBadge(ref, branched);
  } catch { /* leave the grid cell empty — non-fatal */ }
}

function reloadCard3d(ref) {
  if (ref.viewer3d) loadCard3d(ref);
}

// Per-poll: reload a mounted card's scene when its branch appears/ends (side
// flip — reload now, it's the moment the user simulated) or its current side
// progresses (throttled, so a running branch doesn't re-pull every poll).
const CARD_SCENE_RELOAD_MS = 6000;
function refreshCardScenes() {
  if (!review3d) return;
  const now = performance.now();
  for (const ref of reviewCards.values()) {
    if (!ref.viewer3d) continue; // not mounted (off-screen) — loads on mount
    const st = cardSceneState(ref);
    const prev = ref.sceneState;
    const sideFlipped = !prev || prev.side !== st.side || prev.id !== st.id;
    const statusChanged = prev && prev.status !== st.status;
    const progressed = prev && prev.count !== st.count;
    if (!sideFlipped && !statusChanged && !progressed) continue;
    // Side flip (just simulated / broke out) and status change (branch settled
    // to done/paused) reload now; mere mid-run progress is throttled so a
    // running branch doesn't re-pull its whole mesh bundle every poll.
    if (!sideFlipped && !statusChanged && now - (ref.lastSceneLoad ?? 0) < CARD_SCENE_RELOAD_MS) continue;
    loadCard3d(ref);
  }
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
  for (const ev of selectedEvents()) {
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

function updateActionBar() {
  const lab = state.lab;
  const n = selectedEvents().length;
  const branched = selectedBranches();
  selCountEl.textContent = `${n} event${n === 1 ? "" : "s"} selected${lab.sims.size ? ` · ${lab.sims.size} simulating` : ""}`;
  if (branched.length > 0) {
    // The selection is already simulating: re-testing or re-simulating makes no
    // sense, so the bar drives the branches instead — step them all one call,
    // or break them all out (which returns the bar to test/simulate).
    const steppable = branched.filter((b) => b.status !== "done");
    testBtn.textContent = `step sims (${steppable.length})`;
    testBtn.title = "advance every selected simulation branch one LLM call";
    testBtn.disabled = steppable.length === 0;
    if (state.steps.length && simUntilEl.options.length <= 1) {
      for (const s of state.steps) simUntilEl.appendChild(el("option", { value: s, text: `▸ ${s}` }));
    }
    simUntilEl.style.display = state.steps.length ? "" : "none";
    simBtn.textContent = `break out (${branched.length})`;
    simBtn.title = "discard every selected simulation branch (its downstream events + meshes)";
    simBtn.classList.add("danger");
    simBtn.disabled = false;
  } else {
    testBtn.textContent = "test edits on selected";
    testBtn.title = "Run the current prompt edit on every selected slot/zone and highlight the difference";
    testBtn.disabled = n === 0;
    simUntilEl.style.display = "none";
    simBtn.textContent = "simulate downstream";
    simBtn.title = "";
    simBtn.classList.remove("danger");
    simBtn.disabled = n === 0;
  }
}

// --- testing ---------------------------------------------------------------------

// Push a test-state change to this event's card in the grid + refresh the
// "changed by edit" summary.
function reflectTest(key) {
  const ref = reviewCards.get(key);
  if (ref) renderCardBody(ref);
  updateReviewTestSummary();
}

// Run the current draft against one event, storing the A/B result in the
// shared `lab.tests` (so the card body + summary agree).
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

// Bounded-concurrency batch tester — drives "test edits on selected" and the
// per-card "test" so a whole run's worth of slots can be A/B'd in one click
// without flooding the provider.
const TEST_CONCURRENCY = 5;
async function testEvents(events) {
  if (!events.length) return;
  const lab = state.lab;
  const queue = [...events];
  const mySeq = testSeq; // this batch belongs to the current step; bail if it changes
  // Mark every target "queued" up front so all of them show activity now, not
  // just the first TEST_CONCURRENCY a worker grabs — and so a 9-card batch
  // doesn't look like it launched only 5.
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
  testBtn.disabled = true;
  for (const ev of events) {
    const ref = reviewCards.get(targetKey(ev.slot, ev.model, ev.index));
    if (ref) renderCardBody(ref);
  }
  updateReviewTestSummary();
  try {
    const workers = Array.from(
      { length: Math.min(TEST_CONCURRENCY, queue.length) },
      async () => {
        while (queue.length) {
          if (mySeq !== testSeq) break; // step switched mid-batch — stop, don't taint the new step
          await testEvent(queue.shift());
          if (testBatch && mySeq === testSeq) testBatch.done += 1;
          updateReviewTestSummary();
        }
      },
    );
    await Promise.all(workers);
  } finally {
    testBatchRefs -= 1;
    if (testBatchRefs === 0) testBatch = null;
    updateActionBar();
    updateReviewTestSummary();
  }
}

// --- downstream simulation ---------------------------------------------------------

// A cell already simulating: the freshest signal is the slots poll, with the
// lab's own session record and the event payload's load-time flag as
// fallbacks (a branch may have been forked in an earlier session).
// The server's flat `_branches/` folder is the source of truth; lab.sims is a
// session view keyed by branch id that self-heals against each /slots poll:
// drop entries the poll no longer reports (broken out, or the source cell was
// rewound/reset), and ADOPT any top-level branch the server reports that we
// aren't tracking yet (forked via apply-to-run, or rehydrated after a restart).
// Just-forked entries get a grace window before pruning so the race with the
// poll doesn't flicker them away.
const SIM_PRUNE_GRACE_MS = 4000;
function pruneStaleSims() {
  const lab = state.lab;
  const now = performance.now();
  const live = new Map(); // branchId -> summary
  for (const s of state.slots) {
    for (const alias of Object.keys(s.runs ?? {})) {
      for (const b of (s.runs[alias].branches ?? [])) live.set(b.id, b);
    }
  }
  for (const [bid, sim] of [...lab.sims]) {
    if (live.has(bid)) continue;
    if (now - (sim.createdAt ?? 0) < SIM_PRUNE_GRACE_MS) continue; // just forked — let the poll catch up
    lab.sims.delete(bid);
  }
  for (const [bid, b] of live) {
    if (!lab.sims.has(bid)) {
      lab.sims.set(bid, { id: bid, slot: b.slot, model: b.model, eventIndex: b.fork_index, step: lab.simStep, node: null, createdAt: 0 });
    }
  }
  if (lab.sims.size === 0) lab.simStep = null;
}

async function simulateDownstream() {
  // Branches carry the lab's FULL edit set (every drafted step), so the
  // simulation matches exactly what "save to new run" would persist. Each
  // selected event forks its OWN branch — multiple zones of one slot now run as
  // several independent sims, and re-simulating an event just adds another.
  const overrides = overridesPayload();
  const events = selectedEvents();
  if (!events.length) return;
  await launchBranches(events, overrides);
}

async function launchBranches(events, overrides) {
  const lab = state.lab;
  simBtn.disabled = true;
  let started = 0;
  for (const ev of events) {
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
      const resp = await api.createBranch(state.run, ev.slot, ev.model, {
        event_index: ev.index,
        step: lab.step,
        overrides,
        seed,
      });
      const bid = resp.branch?.id;
      if (bid) {
        lab.sims.set(bid, {
          id: bid, slot: ev.slot, model: ev.model,
          eventIndex: ev.index, step: lab.step, node: ev.node, createdAt: performance.now(),
        });
      }
      started += 1;
    } catch (e) {
      toast(`${ev.slot}·${ev.model}: ${e.message}`, "err");
    }
  }
  lab.simStep = lab.step;
  lab.simEditedSteps = Object.keys(overrides);
  simBtn.disabled = false;
  if (started > 0) {
    toast(`simulating downstream on ${started} branch${started === 1 ? "" : "es"}`, "ok");
    emit("poll-now");
    renderSims();
    refreshCardBranchBtns(); // reveal "compare 3D" on the just-branched cards at once
    refreshCardScenes(); // flip the just-branched cards to their branch scene now
  }
}

// Simulate a SINGLE call on its own — the per-card mirror of the action-bar
// "simulate downstream". Forks a NEW branch at exactly this card's event,
// carrying the lab's full edit set + any vetted test seed, and runs it
// independently of whatever else is simulating (many branches per cell now).
async function simulateCell(ev, btn) {
  if (btn) btn.disabled = true;
  try {
    await launchBranches([ev], overridesPayload());
  } finally {
    if (btn) btn.disabled = false;
  }
}

function renderSims() {
  const lab = state.lab;
  let host = document.getElementById("lab-sims");
  if (lab.sims.size === 0) { host?.remove(); updateActionBar(); return; }
  if (!host) {
    host = el("div", { id: "lab-sims", style: "padding:0 0 8px 0" });
    reviewGridEl.before(host);
  }
  host.textContent = "";
  const edited = lab.simEditedSteps?.length ? lab.simEditedSteps.join(", ") : lab.simStep;
  host.appendChild(el("div", { class: "pane-head", style: "border-top:1px solid var(--line)" },
    el("span", { text: `simulations · edits: ${edited}`, title: `branched at ${lab.simStep} events` }),
  ));
  const body = el("div", { style: "padding:4px 8px" });
  // Collapse the per-LLM lineages of one downstream exploration (same slot +
  // model + fork event) into a single group, so the list stays readable across
  // many LLMs. A lone sim renders as a plain row.
  for (const g of simGroups()) {
    if (g.sims.length === 1) { body.appendChild(simRow(g.sims[0], false)); continue; }
    const grp = el("div", { class: "sim-group", style: "border-left:2px solid var(--line); margin:6px 0 6px 2px; padding-left:6px" });
    const forkLabel = g.node ? ` ⑂ ${g.node}` : "";
    grp.appendChild(el("div", { class: "sim-group-head", style: "display:flex; gap:8px; align-items:center; padding:2px 0" },
      el("span", { class: "cell-name", text: `${g.slot} · ${g.model}${forkLabel}`,
        title: "open the original cell's scene",
        onclick: () => emit("open-cell", { slot: g.slot, model: g.model, branch: null }) }),
      el("span", { class: "muted", text: `${g.sims.length} LLMs` }),
      el("button", { text: "compare", title: "compare all these LLM lineages side by side in 3D",
        onclick: () => emit("open-compare", { slot: g.slot, model: g.model, step: g.step, node: g.node, index: g.eventIndex, branch: g.sims[0].id }) }),
      el("button", { class: "danger", text: "break out all", title: "discard every LLM lineage in this group",
        onclick: () => breakOutGroup(g) }),
    ));
    for (const sim of g.sims) grp.appendChild(simRow(sim, true));
    body.appendChild(grp);
  }
  host.appendChild(body);
  updateActionBar();
}

// Group live sims by (slot, model, fork event) — the per-LLM lineages of one
// downstream exploration share a key, so they collapse into one entry.
function simGroups() {
  const groups = new Map();
  for (const sim of state.lab.sims.values()) {
    const key = `${sim.slot}|${sim.model}|${sim.eventIndex}`;
    let g = groups.get(key);
    if (!g) {
      g = { key, slot: sim.slot, model: sim.model, eventIndex: sim.eventIndex, node: null, step: null, sims: [] };
      groups.set(key, g);
    }
    // First sim that knows the zone/step labels the whole group (compare-forked
    // lineages adopted from the poll carry no node).
    if (g.node == null) g.node = sim.node ?? null;
    if (g.step == null) g.step = sim.step ?? null;
    g.sims.push(sim);
  }
  return [...groups.values()];
}

// One sim's row + its controls. `grouped` rows sit under a group header and
// label by the lineage's LLM (the header carries slot·model + zone); standalone
// rows show the full cell.
function simRow(sim, grouped) {
  const b = branchSummaryById(sim.id);
  const view = statusView(b);
  const status = b?.status ?? "starting";
  const pending = b?.pending ?? null;
  const pin = b?.pin && b.pin !== sim.model ? b.pin : null;
  const forkLabel = sim.node ? ` ⑂ ${sim.node}` : "";
  const name = grouped
    ? (pin ?? sim.model)
    : `${sim.slot} · ${sim.model}${pin ? " → " + pin : ""}${forkLabel}`;
  const row = el("div", { class: "sim-row" },
    el("span", { class: `dot ${view.dot}` }),
    el("span", { class: "cell-name", text: name, title: "open this lineage's branch view",
      onclick: () => emit("open-cell", { slot: sim.slot, model: sim.model, branch: sim.id }) }),
    el("span", { class: "step-line", text: view.label }),
  );
  if (!grouped) {
    row.appendChild(el("button", { text: "compare", title: "live run vs this simulation, side by side in 3D",
      onclick: () => emit("open-compare", { slot: sim.slot, model: sim.model, step: sim.step, node: sim.node, index: sim.eventIndex, branch: sim.id }) }));
  }
  if (pending) {
    // One LLM call at a time: each press runs exactly the pending call.
    row.appendChild(el("button", { class: "primary", text: "step", onclick: () => simAction(sim, "step", row) }));
    row.appendChild(el("button", { text: "run rest", title: "finish this lineage without further pauses", onclick: () => simAction(sim, "auto", row) }));
  }
  if (status === "running" && !b?.auto && !pending) {
    // Between gates: a call is in flight; the next gate appears when it lands.
    row.appendChild(el("span", { class: "muted", text: "step running…" }));
  }
  // Parked at a gate (pending) is still a live task → pause breaks out of it;
  // only a branch with no live task (hard-paused / errored) resumes.
  if (status === "running" || pending) {
    row.appendChild(el("button", { text: "pause", onclick: () => simAction(sim, "pause", row) }));
  } else if (status === "paused" || status === "error") {
    row.appendChild(el("button", { text: "resume", onclick: () => simAction(sim, "resume", row) }));
  }
  row.appendChild(el("button", { text: "replace source",
    title: "promote this lineage to be the slot's source run (overwrites the original)",
    onclick: () => replaceSourceWithSim(sim) }));
  row.appendChild(el("button", { class: "danger", text: "break out", onclick: () => simAction(sim, "discard", row) }));
  return row;
}

// Break out of every lineage in a group at once.
async function breakOutGroup(g) {
  for (const sim of [...g.sims]) {
    try { await api.branchDiscard(sim.id); state.lab.sims.delete(sim.id); }
    catch (e) { toast(`${sim.slot}·${sim.model}: ${e.message}`, "err"); }
  }
  if (state.lab.sims.size === 0) state.lab.simStep = null;
  toast("broke out of the group's lineages", "ok");
  emit("poll-now");
  renderSims();
  refreshCardScenes();
}

async function simAction(sim, action, row) {
  for (const b of row.querySelectorAll("button")) b.disabled = true;
  try {
    if (action === "step") await api.branchStep(sim.id);
    else if (action === "auto") await api.branchStep(sim.id, { auto: true });
    else if (action === "pause") await api.branchPause(sim.id);
    else if (action === "resume") await api.branchResume(sim.id);
    else {
      await api.branchDiscard(sim.id);
      state.lab.sims.delete(sim.id);
      if (state.lab.sims.size === 0) state.lab.simStep = null;
    }
    emit("poll-now");
    renderSims();
    refreshCardScenes(); // follow the branch (or revert to source on break-out)
  } catch (e) {
    toast(e.message, "err");
    renderSims();
  }
}

// Promote a single simulation branch to BE its source cell (overwrites the
// original run's events + meshes, then consumes the branch). The deliberate,
// per-slot "replace source with the simulation" — confirmed because it's
// destructive to the source.
function replaceSourceWithSim(sim) {
  openModal(`replace ${sim.slot} · ${sim.model} with its simulation?`, (close, setError) => ({
    body: [
      el("div", { class: "m-hint", text:
        "The simulation branch becomes this slot's source run — its events and meshes replace the original's, " +
        "and the branch is consumed. The previous source state is discarded." }),
    ],
    actions: [
      el("button", { text: "cancel", onclick: close }),
      el("button", { class: "danger", text: "replace source", onclick: async () => {
        try { await api.branchCommit(sim.id); }
        catch (e) { setError(e.message); return; }
        close();
        state.lab.sims.delete(sim.id);
        if (state.lab.sims.size === 0) state.lab.simStep = null;
        toast(`replaced ${sim.slot} · ${sim.model} with its simulation`, "ok");
        emit("poll-now");
        renderSims();
        refreshCardScenes();
      } }),
    ],
  }));
}

// Action-bar batch controls when the selection is simulating: step every
// selected branch one call (or fast-forward to `until`), or break them all
// out. Like "step all", each branch QUEUES rather than erroring when it isn't
// sitting at a gate. Per-branch errors toast but don't abort the rest.
async function stepSelectedBranches(until = null) {
  const branches = selectedBranches().filter((b) => b.status !== "done");
  if (!branches.length) return;
  testBtn.disabled = true;
  await Promise.all(branches.map(async (b) => {
    try { await api.branchStep(b.id, { until }); }
    catch (e) { toast(`${b.slot}·${b.model}: ${e.message}`, "err"); }
  }));
  emit("poll-now");
  renderSims();
  refreshCardScenes();
}

async function breakOutSelected() {
  const branches = selectedBranches();
  if (!branches.length) return;
  simBtn.disabled = true;
  await Promise.all(branches.map(async (b) => {
    try {
      await api.branchDiscard(b.id);
      state.lab.sims.delete(b.id);
    } catch (e) { toast(`${b.slot}·${b.model}: ${e.message}`, "err"); }
  }));
  if (state.lab.sims.size === 0) state.lab.simStep = null;
  toast(`broke out of ${branches.length} simulation${branches.length === 1 ? "" : "s"}`, "ok");
  emit("poll-now");
  renderSims();
  refreshCardScenes();
  updateActionBar();
}

// --- persistence -----------------------------------------------------------------

// The in-place iteration loop: write the lab's edits into THIS run's snapshot
// (optionally syncing the run's source version folder), then SIMULATE from a
// step — forking a NON-destructive branch in every slot that ran it, under the
// freshly-edited snapshot. The source run is untouched, so the original output
// stays for comparison; promote a branch to source later from the simulations
// list ("replace source") when you're happy.
function applyToRunModal() {
  const lab = state.lab;
  const overrides = overridesPayload();
  const editedSteps = Object.keys(overrides);
  const hasEdits = editedSteps.length > 0;
  // Always openable: with edits it applies + (optionally) simulates; with no
  // edits it's just the simulate-from-a-step control.
  const allSteps = [...lab.templates.keys()];
  const versionLabel = state.runs.find((r) => r.name === state.run)?.prompt_version;
  // ON by default when applying edits (keep the version in step); OFF when there
  // are none, where ticking it is an explicit "overwrite the source version with
  // this run's full prompts" — a hard replacement of ALL steps.
  const syncCheck = el("input", { type: "checkbox", ...(hasEdits ? { checked: "" } : {}) });
  const simSel = el("select", {},
    el("option", { value: "", text: "don't simulate yet" }),
    hasEdits ? el("option", { value: "*", text: "simulate from earliest edited step (recommended)" }) : null,
    allSteps.map((s) => el("option", { value: s, text: `simulate from first ${s} call` })),
  );
  // With edits, default to the earliest affected step; with none, default to
  // the step currently open in the lab (the one you're likely iterating on).
  simSel.value = hasEdits ? "*" : (lab.step && allSteps.includes(lab.step) ? lab.step : "");

  openModal(hasEdits ? `apply edits to ${state.run}` : `update ${state.run}`, (close, setError) => ({
    body: [
      hasEdits
        ? el("div", { class: "m-hint", text: `writes into this run's snapshot: ${editedSteps.join(", ")}` })
        : el("div", { class: "m-hint", text: "no pending edits — sync the source version to this run's prompts and/or simulate a step." }),
      versionLabel
        ? el("label", { style: "display:flex;gap:8px;align-items:center;color:var(--text-dim)" },
            syncCheck, `save this run's full prompts to source version "${versionLabel}" — overwrites ALL its steps (so edits applied earlier without syncing land too)`)
        : null,
      field(hasEdits ? "then" : "simulate from", simSel),
      el("div", { class: "m-hint", text:
        "simulating forks a branch in each slot at its first call of that step and runs it downstream under " +
        "the run's (edited) snapshot — the source run is untouched, so you can compare and then promote a " +
        "branch to source from the simulations list. Slots that never ran the step are skipped." }),
    ],
    actions: [
      el("button", { text: "cancel", onclick: close }),
      el("button", { class: "primary", text: hasEdits ? "apply" : "update", onclick: async () => {
        const wantWrite = hasEdits || syncCheck.checked;
        if (!wantWrite && !simSel.value) {
          setError("nothing to do — edit a prompt, pick a step to simulate, or check “save to source version”");
          return;
        }
        try {
          if (wantWrite) {
            const applied = await api.updateRunPrompts(state.run, overrides, syncCheck.checked);
            const parts = [];
            if (editedSteps.length) parts.push(`snapshot updated (${applied.applied.join(", ")})`);
            if (syncCheck.checked) {
              parts.push(applied.version_synced
                ? `version "${applied.version_synced}" overwritten with this run's prompts`
                : "source version not found, not synced");
            }
            if (parts.length) toast(parts.join(" · "), "ok");
          }
          if (simSel.value) {
            const steps = simSel.value === "*" ? editedSteps : [simSel.value];
            const r = await api.simulateStep(state.run, steps);
            // The server forked the branches; the next /slots poll adopts them
            // into lab.sims (pruneStaleSims reconciles), so no manual seeding.
            lab.simStep = steps[0];
            lab.simEditedSteps = editedSteps.length ? editedSteps : steps;
            toast(`simulating from ${steps.join("/")} on ${r.simulated.length} slot${r.simulated.length === 1 ? "" : "s"}` +
              (r.skipped.length ? ` (${r.skipped.length} skipped — never ran it)` : ""), r.simulated.length ? "ok" : "err");
          }
          close();
          // The applied edits are now the snapshot's canonical templates: drop
          // every draft + test (stale against the new snapshot) and reload, so
          // no pre-save prompt can linger in the lab.
          lab.drafts = new Map();
          lab.tests = new Map();
          await openLab();
          emit("poll-now");
          renderSims();
          refreshCardScenes();
        } catch (e) { setError(e.message); }
      } }),
    ],
  }));
}

function saveVersionModal() {
  const overrides = overridesPayload();
  const edited = Object.keys(overrides);
  // No gate on pending drafts: the run's snapshot already holds every edit
  // applied to it, so a new version is its FULL snapshot (+ any unsaved drafts).
  const input = el("input", { type: "text", placeholder: "e.g. baseline-tighter-bboxes" });
  openModal("save to new version", (close, setError) => ({
    body: [
      field("version name", input),
      el("div", { class: "m-hint", text: edited.length
        ? `versions/<name>/ = this run's full prompt snapshot + your unsaved edits to: ${edited.join(", ")}`
        : "versions/<name>/ = this run's full prompt snapshot — all steps, including everything already applied to the run" }),
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
  const sims = [...lab.sims.values()];
  openModal("save branches to new run", (close, setError) => ({
    body: [
      field("run name", input),
      el("div", { class: "m-hint",
        text: `copies ${sims.length} simulation${sims.length === 1 ? "" : "s"} (paused or not) into a fresh run whose prompt snapshot includes your edit — every branch resumes there seamlessly. When two sims share a slot the later wins.` }),
      el("div", { class: "check-grid" },
        sims.map((s) => el("label", {}, el("span", { class: "dot " + (branchSummaryById(s.id)?.status ?? "idle") }), `${s.slot} · ${s.model}${s.node ? " ⑂ " + s.node : ""}`)),
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
            branches: sims.map((s) => s.id),
            version_label: lastSavedVersion,
          });
          toast(`run "${payload.current}" created (${payload.copied.length} cells)`, "ok");
          close();
          teardownGrid();
          labEl.classList.remove("open");
          lab.sims.clear();
          lab.simStep = null;
          emit("switch-run", payload.current);
        } catch (e) { setError(e.message); }
      } }),
    ],
  }));
}

// Drag the divider between the template editor and the debug grid to set the
// grid's width; the editor reflows to fill the rest. Persisted so the chosen
// split survives reloads. Mirrors the overlay's observability-dock resizer.
const LAB_DEBUG_WIDTH_KEY = "starshot.labDebugWidth";
const LAB_DEBUG_MIN = 360;
const LAB_EDITOR_MIN = 420; // step list + editor keep at least this much

function initLabResizer() {
  const resizer = document.getElementById("lab-resizer");
  const pane = document.getElementById("lab-right");
  const body = document.getElementById("lab-body");
  let saved = NaN;
  try { saved = Number(localStorage.getItem(LAB_DEBUG_WIDTH_KEY)); } catch { /* private mode */ }
  if (saved >= LAB_DEBUG_MIN) pane.style.width = `${saved}px`;

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
    const max = Math.max(LAB_DEBUG_MIN, rect.width - LAB_EDITOR_MIN);
    const width = Math.max(LAB_DEBUG_MIN, Math.min(rect.right - ev.clientX, max));
    pane.style.width = `${Math.round(width)}px`;
  });
  const end = (ev) => {
    if (!dragging) return;
    dragging = false;
    resizer.classList.remove("dragging");
    try { resizer.releasePointerCapture(ev.pointerId); } catch { /* already released */ }
    try { localStorage.setItem(LAB_DEBUG_WIDTH_KEY, String(parseInt(pane.style.width, 10) || LAB_DEBUG_MIN)); } catch { /* private mode */ }
  };
  resizer.addEventListener("pointerup", end);
  resizer.addEventListener("pointercancel", end);
}
