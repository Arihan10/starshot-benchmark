// Boot + topbar: run selection, run creation, cell launching, status polling.
// Owns the canonical state.slots refresh loop every panel renders from.

import { api } from "./api.js";
import { state, emit, on } from "./state.js";
import { el, toast, openModal, field, stepUntilSelect } from "./ui.js";
import { createViewer } from "./scene3d.js";
import { renderBoard } from "./board.js";
import { initOverlay, closeOverlay } from "./overlay.js";
import { initLab } from "./promptlab.js";
import { initCompare } from "./compare.js";
import { initRunCompare } from "./runcompare.js";
import { initCost } from "./cost.js";
import { initQueuePanel } from "./queue.js";
import { initLighting } from "./lighting.js";
import { createRunCombo } from "./runcombo.js";

const LAST_RUN_KEY = "starshot.lastRun";
const runPickerEl = document.getElementById("run-picker");
const statusTextEl = document.getElementById("status-text");

// --- runs ------------------------------------------------------------------------

// The active-run selector is a searchable combobox (see runcombo.js) — the
// replacement for the native <select> whose order couldn't be sorted. Built once
// at boot; `renderRunPicker` just refreshes its label + open list.
let runCombo = null;

function runLabel(r) {
  return r.prompt_version ? `${r.name} · ${r.prompt_version}` : `${r.name} (legacy)`;
}

function renderRunPicker() {
  runCombo?.render();
}

async function refreshRuns() {
  const payload = await api.runs();
  state.runs = payload.runs;
  if (!state.run) state.run = payload.current || payload.runs[0]?.name || null;
  renderRunPicker();
  return payload;
}

// Clear the prompt-lab session (drafts, selected step, loaded templates, sims).
// Lab state is per-run, so EVERY run change must wipe it — otherwise a draft
// left over from the previous run is compared against the new run's snapshot
// and shows as a phantom "unsaved change" (most visibly on the default-selected
// root zone-plan step). Used by both switchRun and createRun.
function resetLabSession() {
  document.getElementById("lab").classList.remove("open");
  state.lab.open = false;
  state.lab.step = null;
  state.lab.templates = new Map();
  state.lab.drafts = new Map();
  state.lab.events = [];
  state.lab.selection = new Map();
  state.lab.simModels = new Map();
  state.lab.tests = new Map();
  state.lab.sims = new Map();
  state.lab.simStep = null;
  state.lab.simEditedSteps = [];
  state.lab.atomicLocks = new Set();
}

async function switchRun(name) {
  if (!name || name === state.run) return;
  try {
    await api.activateRun(name);
  } catch (e) {
    toast(`run switch failed: ${e.message}`, "err");
    renderRunPicker();
    return;
  }
  state.run = name;
  try { localStorage.setItem(LAST_RUN_KEY, name); } catch { /* private mode */ }
  closeOverlay();
  // Board selection is per-run (cell keys repeat across runs) — drop it so a
  // bulk action never lands on the previous run's cells.
  state.selection.clear();
  state.selectMode = false;
  emit("selection");
  // Lab state belongs to the previous run; sims (if any) keep running
  // server-side on that run and stay reachable by switching back.
  resetLabSession();
  renderRunPicker();
  await refreshSlots();
  toast(`run :: ${name}`);
}

runCombo = createRunCombo(runPickerEl, {
  getRuns: () => state.runs,
  getSelected: () => state.run,
  onPick: switchRun,
  buttonLabel: runLabel,
  placeholder: "select a run",
});
on("switch-run", async (name) => {
  await refreshRuns();
  await switchRun(name);
});

// --- slots polling -----------------------------------------------------------------

let pollTimer = null;

async function refreshSlots() {
  if (!state.run) {
    state.slots = [];
    emit("slots");
    updateStatusText();
    return;
  }
  try {
    const payload = await api.slots(state.run);
    state.models = payload.models;
    state.defaultModel = payload.default_model;
    state.slots = payload.slots;
    // The pipeline step list (for "step until X") is run-independent, so fetch
    // it once per session from the run's prompt snapshot.
    if (!state.steps.length) {
      try {
        const t = await api.promptTemplates(state.run);
        // image_prompt auto-plays when stepping (it's not a gated step), so it's
        // not a valid "step until" target — drop it from the steppable list.
        state.steps = t.steps.map((s) => s.step).filter((s) => s !== "image_prompt");
      } catch { /* old runs without a snapshot — step-until stays unavailable */ }
    }
    emit("slots");
  } catch (e) {
    statusTextEl.textContent = `slots poll failed: ${e.message}`;
    return;
  }
  updateStatusText();
}

function updateStatusText() {
  const stepAllBtn = document.getElementById("btn-step-all");
  if (!state.run) {
    statusTextEl.textContent = "no runs yet — create one with “+ new run”";
    stepAllBtn.style.display = "none";
    stepAllUntilEl.style.display = "none";
    exitSteppingEl.style.display = "none";
    return;
  }
  let running = 0;
  let done = 0;
  let error = 0;
  let sims = 0;
  let waiting = 0;
  let stepped = 0;
  for (const slot of state.slots) {
    for (const m of state.models) {
      const c = slot.runs?.[m];
      if (!c) continue;
      if (c.status === "running") running += 1;
      else if (c.status === "done") done += 1;
      else if (c.status === "error") error += 1;
      sims += c.branches?.length ?? 0;
      if (c.pending) waiting += 1;
      if (c.stepped && c.status !== "done") stepped += 1;
    }
  }
  statusTextEl.textContent =
    `${state.run} — ${running} running · ${done} done · ${error} error` +
    (waiting ? ` · ${waiting} awaiting step` : "") +
    (sims ? ` · ${sims} sim${sims === 1 ? "" : "s"}` : "");
  // "step all" targets every stepped cell, advancing each one call regardless
  // of whether it's paused at a live gate, mid-call, or paused with no task.
  stepAllBtn.style.display = stepped > 0 ? "" : "none";
  stepAllBtn.textContent = `step all (${stepped})`;
  stepAllUntilEl.style.display = (stepped > 0 && state.steps.length) ? "" : "none";
  exitSteppingEl.style.display = stepped > 0 ? "" : "none";
}

document.getElementById("btn-step-all").addEventListener("click", async () => {
  try {
    const r = await api.stepAll(state.run);
    toast(`advanced ${r.advanced.length} stepped slot${r.advanced.length === 1 ? "" : "s"} one step`, "ok");
    refreshSlots();
  } catch (e) {
    toast(e.message, "err");
  }
});

// "step all until <step>": fast-forward every stepped slot to the next run of
// the chosen step and pause them all there — gets the whole experiment onto
// the same step for side-by-side comparison.
const stepAllUntilEl = stepUntilSelect(
  () => state.steps,
  async (until, before) => {
    try {
      const r = await api.stepAll(state.run, { until, untilBefore: before });
      toast(`fast-forwarding ${r.advanced.length} slot${r.advanced.length === 1 ? "" : "s"} to ${before ? "before " : ""}${until}`, "ok");
      refreshSlots();
    } catch (e) {
      toast(e.message, "err");
    }
  },
  { label: "all until…", title: "run every stepped slot up to the next call of a step — pause before or after it" },
);
stepAllUntilEl.style.display = "none";
document.getElementById("btn-step-all").after(stepAllUntilEl);

// "exit stepping": let every stepped slot run to completion and leave step mode.
const exitSteppingEl = el("button", {
  id: "btn-exit-stepping", style: "display:none",
  title: "run every stepped slot to completion and leave step mode",
}, "exit stepping");
stepAllUntilEl.after(exitSteppingEl);
exitSteppingEl.addEventListener("click", async () => {
  try {
    const r = await api.stepAll(state.run, { auto: true });
    toast(`finishing ${r.advanced.length} stepped slot${r.advanced.length === 1 ? "" : "s"} — leaving step mode`, "ok");
    refreshSlots();
  } catch (e) {
    toast(e.message, "err");
  }
});

function startPolling() {
  clearInterval(pollTimer);
  pollTimer = setInterval(refreshSlots, 2500);
  setInterval(() => refreshRuns().catch(() => {}), 12000);
}

on("poll-now", refreshSlots);

// --- new run / start cells ----------------------------------------------------------

async function newRunModal() {
  let versions = [];
  try {
    versions = (await api.versions()).versions.map((v) => v.name);
  } catch (e) {
    toast(`failed to load prompt versions: ${e.message}`, "err");
    return;
  }
  const nameInput = el("input", { type: "text", placeholder: "e.g. iteration15" });
  const versionSel = el("select", {}, versions.map((v) => el("option", { value: v, text: v })));
  const activeVersion = state.runs.find((r) => r.name === state.run)?.prompt_version;
  if (activeVersion && versions.includes(activeVersion)) versionSel.value = activeVersion;

  // Fork-a-fresh-version flow: the run starts on its own copy of a base
  // version (default baseline), ready for in-place iteration from the lab.
  const forkCheck = el("input", { type: "checkbox" });
  const forkName = el("input", { type: "text", placeholder: "e.g. root-plan-rewrite" });
  const forkRow = el("div", { class: "m-field", style: "display:none" },
    el("span", { text: "new version name (forked from the selection above)" }),
    forkName,
  );
  forkCheck.addEventListener("change", () => {
    forkRow.style.display = forkCheck.checked ? "" : "none";
  });

  openModal("new run", (close, setError) => ({
    body: [
      field("run name", nameInput),
      field("prompt version (or fork base)", versionSel),
      el("label", { style: "display:flex;gap:8px;align-items:center;color:var(--text-dim)" },
        forkCheck, "create a NEW version from it for this run (iterate it in the prompt lab)"),
      forkRow,
      el("div", { class: "m-hint", text: "the chosen version's templates are copied into the run as its snapshot; with a fork, lab edits applied to the run can be kept in sync with the new version folder" }),
      versions.length === 0 ? el("div", { class: "m-error", text: "no prompt versions found — add a folder under versions/ first" }) : null,
    ],
    actions: [
      el("button", { text: "cancel", onclick: close }),
      el("button", { class: "primary", text: "create run", onclick: async () => {
        try {
          let version = versionSel.value;
          if (forkCheck.checked) {
            const newName = forkName.value.trim();
            if (!newName) {
              setError("name the new version (or untick the fork option)");
              return;
            }
            await api.forkVersion(newName, versionSel.value);
            version = newName;
          }
          const payload = await api.createRun(nameInput.value.trim(), version);
          close();
          toast(`run created :: ${payload.current} (prompts: ${version})`, "ok");
          await refreshRuns();
          state.run = payload.current;
          // A brand-new run starts with a clean prompt lab — reset the session
          // so a draft left over from the previously-active run isn't compared
          // against the new run's snapshot and flagged as a phantom edit.
          resetLabSession();
          state.selection.clear();
          state.selectMode = false;
          emit("selection");
          try { localStorage.setItem(LAST_RUN_KEY, payload.current); } catch { /* ignore */ }
          renderRunPicker();
          await refreshSlots();
        } catch (e) { setError(e.message); }
      } }),
    ],
  }));
}

function startCellsModal() {
  if (!state.run) return;
  const modelChecks = state.models.map((m) =>
    el("label", {}, el("input", { type: "checkbox", value: m, ...(m === state.defaultModel ? { checked: "" } : {}) }), m));
  const slotChecks = state.slots.map((s) =>
    el("label", {}, el("input", { type: "checkbox", value: s.id }), s.id));
  const steppedCheck = el("input", { type: "checkbox", checked: "" });
  openModal(`start cells on ${state.run}`, (close, setError) => ({
    body: [
      el("div", { class: "m-field" }, el("span", { text: "models" }), el("div", { class: "check-grid" }, modelChecks)),
      el("div", { class: "m-field" }, el("span", { text: "slots" }), el("div", { class: "check-grid" }, slotChecks)),
      el("label", { style: "display:flex;gap:8px;align-items:center;color:var(--text-dim)" },
        steppedCheck,
        "run one step at a time (each slot pauses before every LLM call — the prompt-iteration mode)"),
      el("div", { class: "m-hint", text: "launches the cross product; running/done cells are skipped automatically. Untick stepping for a full benchmark run." }),
    ],
    actions: [
      el("button", { text: "cancel", onclick: close }),
      el("button", { class: "primary", text: "start selected", onclick: async () => {
        const models = modelChecks.filter((l) => l.firstChild.checked).map((l) => l.firstChild.value);
        const slots = slotChecks.filter((l) => l.firstChild.checked).map((l) => l.firstChild.value);
        if (models.length === 0 || slots.length === 0) {
          setError("pick at least one model and one slot");
          return;
        }
        close();
        const stepped = steppedCheck.checked;
        const results = await Promise.all(slots.flatMap((s) => models.map(async (m) => {
          try { await api.resume(state.run, s, m, stepped); return true; }
          catch { return false; }
        })));
        const ok = results.filter(Boolean).length;
        toast(
          `started ${ok}/${results.length} cells${stepped ? " (stepped)" : ""}` +
            (ok < results.length ? " — rest skipped: running/done/capped/legacy" : ""),
          ok ? "ok" : "err",
        );
        refreshSlots();
      } }),
    ],
  }));
}

// Launch a new run (B) that reuses this run's ROOT zone plans: pick the cells
// to carry over + a prompt version, and each cell is re-run from just after its
// root zone plan under the new version (the plan is held fixed, everything
// below it regenerates). Ticking "hold the overall bounding box" also carries
// each cell's root bbox, so the scene canvas is identical in both runs. Server
// seeds only cells with the copied step(s) committed.
async function abTestModal() {
  if (!state.run) return;
  const sourceRun = state.run;
  const candidates = [];
  for (const slot of state.slots) {
    for (const m of state.models) {
      const c = slot.runs?.[m];
      if (c && (c.events_count ?? 0) > 0) {
        candidates.push({ slot: slot.id, model: m, status: c.status ?? "idle", events: c.events_count ?? 0 });
      }
    }
  }
  if (!candidates.length) {
    toast(`no started cells in "${sourceRun}" to A/B from`, "err");
    return;
  }
  let versions = [];
  try {
    versions = (await api.versions()).versions.map((v) => v.name);
  } catch (e) {
    toast(`failed to load prompt versions: ${e.message}`, "err");
    return;
  }
  const nameInput = el("input", { type: "text", placeholder: "e.g. ab-newplan" });
  const versionSel = el("select", {}, versions.map((v) => el("option", { value: v, text: v })));
  const activeVersion = state.runs.find((r) => r.name === sourceRun)?.prompt_version;
  if (activeVersion && versions.includes(activeVersion)) versionSel.value = activeVersion;

  // Same fork-a-fresh-version affordance as "new run", so the A/B can be run on
  // a brand-new version copied from the base (then iterated in the prompt lab).
  const forkCheck = el("input", { type: "checkbox" });
  const forkName = el("input", { type: "text", placeholder: "e.g. root-plan-rewrite" });
  const forkRow = el("div", { class: "m-field", style: "display:none" },
    el("span", { text: "new version name (forked from the selection above)" }),
    forkName,
  );
  forkCheck.addEventListener("change", () => { forkRow.style.display = forkCheck.checked ? "" : "none"; });

  // Also carry each cell's root overall bounding box (not just its zone plan),
  // so both runs share the same scene canvas and only the fill varies.
  const bboxCheck = el("input", { type: "checkbox" });

  const cellChecks = candidates.map((c) =>
    el("label", {},
      el("input", { type: "checkbox", value: `${c.slot}|${c.model}`, checked: "" }),
      el("span", { class: `dot ${c.status}` }),
      el("span", { text: `${c.slot} · ${c.model}` }),
      el("span", { class: "muted", style: "margin-left:auto", text: `${c.events} ev` }),
    ));

  openModal("launch A/B test", (close, setError) => ({
    body: [
      field("run name (B)", nameInput),
      field("prompt version (or fork base)", versionSel),
      el("label", { style: "display:flex;gap:8px;align-items:center;color:var(--text-dim)" },
        forkCheck, "create a NEW version from it for this run (iterate it in the prompt lab)"),
      forkRow,
      el("div", { class: "m-field" },
        el("span", { text: `cells to A/B from “${sourceRun}” (keeps each cell's root zone plan)` }),
        el("div", { class: "check-grid" }, cellChecks)),
      el("label", { style: "display:flex;gap:8px;align-items:center;color:var(--text-dim)" },
        bboxCheck, "also hold the overall bounding box fixed (copy each cell's root bbox call too)"),
      el("div", { class: "m-hint", text: "Run B copies each selected cell's log up to its root zone plan, then re-runs from there on the chosen version — the plan is held fixed, everything below it (the overall bounding box included) regenerates. Tick the box above to also copy the root overall bounding box so the scene canvas is identical in both runs. Cells missing the copied step are skipped." }),
    ],
    actions: [
      el("button", { text: "cancel", onclick: close }),
      el("button", { class: "primary", text: "create & launch", onclick: async () => {
        const name = nameInput.value.trim();
        if (!name) { setError("name the new run"); return; }
        const cells = cellChecks
          .filter((l) => l.firstChild.checked)
          .map((l) => {
            const v = l.firstChild.value;
            const i = v.lastIndexOf("|"); // slot ids may contain spaces but never "|"
            return { slot: v.slice(0, i), model: v.slice(i + 1) };
          });
        if (!cells.length) { setError("pick at least one cell"); return; }
        try {
          let version = versionSel.value;
          if (forkCheck.checked) {
            const newName = forkName.value.trim();
            if (!newName) { setError("name the new version (or untick the fork option)"); return; }
            await api.forkVersion(newName, versionSel.value);
            version = newName;
          }
          const payload = await api.abTest(name, version, sourceRun, cells, bboxCheck.checked);
          close();
          const nSeeded = payload.seeded?.length ?? 0;
          const nSkipped = payload.skipped?.length ?? 0;
          const skipWhat = bboxCheck.checked ? "without a plan or overall bbox" : "without a plan";
          toast(
            `A/B run :: ${payload.current} — launched ${nSeeded} cell${nSeeded === 1 ? "" : "s"}` +
              (nSkipped ? ` (skipped ${nSkipped} ${skipWhat})` : ""),
            "ok",
          );
          await refreshRuns();
          state.run = payload.current;
          // Brand-new run → clean prompt lab + no stale board selection.
          resetLabSession();
          state.selection.clear();
          state.selectMode = false;
          emit("selection");
          try { localStorage.setItem(LAST_RUN_KEY, payload.current); } catch { /* ignore */ }
          renderRunPicker();
          await refreshSlots();
        } catch (e) { setError(e.message); }
      } }),
    ],
  }));
}

// Copy a whole slot folder (every model cell + its meshes) from THIS run into
// another run, overwriting that run's slot. Warns first about any populated
// cells in the destination slot that the copy would delete/replace.
async function copySlotModal() {
  if (!state.run) return;
  const sourceRun = state.run;
  const populated = state.slots.filter((s) => state.models.some((m) => (s.runs?.[m]?.events_count ?? 0) > 0));
  if (!populated.length) { toast(`no slots with data in "${sourceRun}" to copy`, "err"); return; }
  const others = state.runs.map((r) => r.name).filter((n) => n !== sourceRun);
  if (!others.length) { toast("no other run to copy into — create one first", "err"); return; }

  const slotSel = el("select", {}, populated.map((s) => el("option", { value: s.id, text: s.id })));
  const destSel = el("select", {}, others.map((n) => el("option", { value: n, text: n })));
  const warnEl = el("div", { class: "m-hint" });

  // Preflight the destination: list any of its cells under the chosen slot that
  // already have data (they'd be deleted/replaced by the overwrite).
  async function refreshWarn() {
    const destRun = destSel.value;
    const slotId = slotSel.value;
    warnEl.className = "m-hint";
    warnEl.textContent = "checking destination…";
    let cells = null;
    try {
      const payload = await api.slots(destRun);
      const s = payload.slots.find((x) => x.id === slotId);
      cells = s ? Object.entries(s.runs || {}).filter(([, c]) => (c?.events_count ?? 0) > 0).map(([m]) => m) : [];
    } catch { cells = null; }
    if (destSel.value !== destRun || slotSel.value !== slotId) return; // a later change superseded this
    if (cells === null) { warnEl.textContent = "couldn't read the destination run's contents."; return; }
    if (cells.length) {
      warnEl.className = "m-error";
      warnEl.textContent = `⚠ Deletes & replaces ${cells.length} existing cell${cells.length === 1 ? "" : "s"} in "${destRun}/${slotId}": ${cells.join(", ")}.`;
    } else {
      warnEl.textContent = `"${destRun}/${slotId}" is empty — nothing will be overwritten.`;
    }
  }
  slotSel.addEventListener("change", refreshWarn);
  destSel.addEventListener("change", refreshWarn);

  openModal(`copy a slot from "${sourceRun}"`, (close, setError) => {
    refreshWarn();
    return {
      body: [
        field("slot", slotSel),
        field("copy into run", destSel),
        el("div", { class: "m-hint", text:
          `Copies the entire "${sourceRun}/<slot>" folder (every model cell + its meshes) into the chosen run, overwriting that run's slot. The source run is untouched.` }),
        warnEl,
      ],
      actions: [
        el("button", { text: "cancel", onclick: close }),
        el("button", { class: "danger", text: "copy slot", onclick: async () => {
          const slotId = slotSel.value;
          const destRun = destSel.value;
          try {
            const r = await api.copySlot(destRun, sourceRun, slotId);
            close();
            const nCopied = r.copied?.length ?? 0;
            const nReplaced = r.replaced?.length ?? 0;
            toast(`copied "${slotId}" → "${destRun}" — ${nCopied} cell${nCopied === 1 ? "" : "s"}${nReplaced ? `, replaced ${nReplaced}` : ""}`, "ok");
            await refreshRuns();
            if (destRun === state.run) await refreshSlots();
          } catch (e) { setError(e.message); }
        } }),
      ],
    };
  });
}

// Copy a single cell (one slot's log + meshes for one model) into another model
// and/or run — the cross-model / cross-run transplant. The source is a cell on
// THIS run; the destination is any run's cell for the SAME slot (content is
// scene-specific). The destination cell is overwritten; the source is untouched.
async function copyCellModal() {
  if (!state.run) return;
  const sourceRun = state.run;
  const dataSlots = state.slots.filter((s) => state.models.some((m) => (s.runs?.[m]?.events_count ?? 0) > 0));
  if (!dataSlots.length) { toast(`no started cells in "${sourceRun}" to copy`, "err"); return; }

  const slotSel = el("select", {}, dataSlots.map((s) => el("option", { value: s.id, text: s.id })));
  const fromSel = el("select");
  const toRunSel = el("select", {}, state.runs.map((r) => el("option", { value: r.name, text: r.name })));
  toRunSel.value = sourceRun;
  const toSel = el("select", {}, state.models.map((m) => el("option", { value: m, text: m })));
  const warnEl = el("div", { class: "m-hint" });

  // The chosen source slot's models that actually have data — the copy sources.
  function refreshFromModels() {
    const s = dataSlots.find((x) => x.id === slotSel.value);
    fromSel.textContent = "";
    for (const m of state.models) {
      const n = s?.runs?.[m]?.events_count ?? 0;
      if (n > 0) fromSel.appendChild(el("option", { value: m, text: `${m} · ${n} ev` }));
    }
  }

  // Flag when the destination cell already holds data (it'll be replaced) or is
  // the source itself. Reads the active run live, any other run on demand.
  async function refreshWarn() {
    const destRun = toRunSel.value, slotId = slotSel.value, srcModel = fromSel.value, destModel = toSel.value;
    if (destRun === sourceRun && destModel === srcModel) {
      warnEl.className = "m-error";
      warnEl.textContent = "source and destination are the same cell — pick a different model or run.";
      return;
    }
    warnEl.className = "m-hint";
    warnEl.textContent = "checking destination…";
    let count = null;
    try {
      const slots = destRun === sourceRun ? state.slots : (await api.slots(destRun)).slots;
      count = slots.find((x) => x.id === slotId)?.runs?.[destModel]?.events_count ?? 0;
    } catch { count = null; }
    if (toRunSel.value !== destRun || slotSel.value !== slotId || toSel.value !== destModel || fromSel.value !== srcModel) return;
    if (count === null) { warnEl.textContent = "couldn't read the destination run."; return; }
    warnEl.className = count > 0 ? "m-error" : "m-hint";
    warnEl.textContent = count > 0
      ? `⚠ "${destRun} / ${slotId} / ${destModel}" has ${count} event${count === 1 ? "" : "s"} — deleted & replaced.`
      : `"${destRun} / ${slotId} / ${destModel}" is empty — nothing overwritten.`;
  }

  slotSel.addEventListener("change", () => { refreshFromModels(); refreshWarn(); });
  fromSel.addEventListener("change", refreshWarn);
  toRunSel.addEventListener("change", refreshWarn);
  toSel.addEventListener("change", refreshWarn);
  refreshFromModels();
  // Pre-pick a destination model different from the source, so the common
  // within-run cross-model copy is valid without extra clicks.
  const alt = state.models.find((m) => m !== fromSel.value);
  if (alt) toSel.value = alt;

  openModal(`copy a cell from "${sourceRun}"`, (close, setError) => {
    refreshWarn();
    return {
      body: [
        field("slot (shared by both cells)", slotSel),
        field("from model", fromSel),
        field("to run", toRunSel),
        field("to model", toSel),
        el("div", { class: "m-hint", text:
          "Copies one model's cell — its whole log + meshes — into another model and/or run for the same slot, overwriting the destination. Cost/usage stay attributed to the source model; the source cell is untouched." }),
        warnEl,
      ],
      actions: [
        el("button", { text: "cancel", onclick: close }),
        el("button", { class: "danger", text: "copy cell", onclick: async () => {
          const destRun = toRunSel.value, slotId = slotSel.value, srcModel = fromSel.value, destModel = toSel.value;
          if (destRun === sourceRun && destModel === srcModel) { setError("pick a different destination cell"); return; }
          try {
            const r = await api.copyCell(destRun, sourceRun, slotId, srcModel, destModel);
            close();
            toast(`copied "${slotId}" · ${srcModel} → ${destRun} / ${destModel} — ${r.events} event${r.events === 1 ? "" : "s"}${r.replaced ? " (replaced)" : ""}`, "ok");
            await refreshRuns();
            if (destRun === state.run) await refreshSlots();
          } catch (e) { setError(e.message); }
        } }),
      ],
    };
  });
}

function resetAllModal() {
  if (!state.run) return;
  // Every started cell on this run (anything with logged events, or a live
  // simulation branch). reset wipes each back to idle without restarting.
  const cells = [];
  for (const slot of state.slots) {
    for (const m of state.models) {
      const c = slot.runs?.[m];
      if (c && ((c.events_count ?? 0) > 0 || c.branches?.length)) cells.push({ slot: slot.id, model: m });
    }
  }
  if (cells.length === 0) {
    toast(`no started cells to reset on "${state.run}"`);
    return;
  }
  openModal(`reset all on ${state.run}`, (close) => ({
    body: [
      el("div", { class: "m-hint", text:
        `Wipe ${cells.length} started cell${cells.length === 1 ? "" : "s"} back to idle — permanently deletes their events, generated meshes, and any simulation branches. Nothing is restarted; use “start cells…” afterward.` }),
    ],
    actions: [
      el("button", { text: "cancel", onclick: close }),
      el("button", { class: "danger", text: `reset ${cells.length} cell${cells.length === 1 ? "" : "s"}`, onclick: async () => {
        close();
        closeOverlay(); // a viewed cell may be among those wiped
        const results = await Promise.all(cells.map(async (c) => {
          try { await api.reset(state.run, c.slot, c.model, false); return true; }
          catch { return false; }
        }));
        const ok = results.filter(Boolean).length;
        toast(
          `reset ${ok}/${results.length} cell${results.length === 1 ? "" : "s"} to idle` +
            (ok < results.length ? " — some failed" : ""),
          ok === results.length ? "ok" : "err",
        );
        refreshSlots();
      } }),
    ],
  }));
}

document.getElementById("btn-new-run").addEventListener("click", newRunModal);
document.getElementById("btn-start-cells").addEventListener("click", startCellsModal);
document.getElementById("btn-reset-all").addEventListener("click", resetAllModal);
document.getElementById("btn-ab-test").addEventListener("click", abTestModal);
document.getElementById("btn-copy-slot").addEventListener("click", copySlotModal);
document.getElementById("btn-copy-cell").addEventListener("click", copyCellModal);

// --- boot ------------------------------------------------------------------------

(async () => {
  const viewer = createViewer(document.getElementById("canvas-host"), { lighting: true });
  initOverlay(viewer);
  initLighting(viewer);
  initLab();
  initCompare();
  initRunCompare();
  initCost();
  initQueuePanel();

  try {
    const payload = await refreshRuns();
    // Land on the run we were on last time this browser quit, falling back
    // to the server's current (newest) run.
    let saved = null;
    try { saved = localStorage.getItem(LAST_RUN_KEY); } catch { /* ignore */ }
    if (saved && saved !== state.run && state.runs.some((r) => r.name === saved)) {
      await api.activateRun(saved).catch(() => {});
      state.run = saved;
      renderRunPicker();
    } else if (payload.current) {
      try { localStorage.setItem(LAST_RUN_KEY, payload.current); } catch { /* ignore */ }
    }
  } catch (e) {
    statusTextEl.textContent = `server unreachable: ${e.message}`;
  }
  await refreshSlots();
  renderBoard();
  startPolling();
})();
