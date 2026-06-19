// Boot + topbar: run selection, run creation, cell launching, status polling.
// Owns the canonical state.slots refresh loop every panel renders from.

import { api } from "./api.js";
import { state, emit, on } from "./state.js";
import { el, toast, openModal, field } from "./ui.js";
import { createViewer } from "./scene3d.js";
import { renderBoard } from "./board.js";
import { initOverlay, closeOverlay } from "./overlay.js";
import { initLab } from "./promptlab.js";
import { initCompare } from "./compare.js";
import { initCost } from "./cost.js";

const LAST_RUN_KEY = "starshot.lastRun";
const runPickerEl = document.getElementById("run-picker");
const statusTextEl = document.getElementById("status-text");

// --- runs ------------------------------------------------------------------------

function renderRunPicker() {
  runPickerEl.textContent = "";
  for (const r of state.runs) {
    runPickerEl.appendChild(el("option", {
      value: r.name,
      text: r.prompt_version ? `${r.name} · ${r.prompt_version}` : `${r.name} (legacy)`,
    }));
  }
  if (state.run) runPickerEl.value = state.run;
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
  state.lab.tests = new Map();
  state.lab.sims = new Map();
  state.lab.simStep = null;
  state.lab.simEditedSteps = [];
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

runPickerEl.addEventListener("change", () => switchRun(runPickerEl.value));
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
        state.steps = t.steps.map((s) => s.step);
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
  if (state.steps.length && stepAllUntilEl.options.length <= 1) {
    for (const s of state.steps) stepAllUntilEl.appendChild(el("option", { value: s, text: `▸ ${s}` }));
  }
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
const stepAllUntilEl = el("select", {
  id: "step-all-until", class: "step-until", style: "display:none",
  title: "fast-forward every stepped slot to the next run of a step",
}, el("option", { value: "", text: "all until…" }));
document.getElementById("btn-step-all").after(stepAllUntilEl);
stepAllUntilEl.addEventListener("change", async () => {
  const until = stepAllUntilEl.value;
  stepAllUntilEl.value = "";
  if (!until) return;
  try {
    const r = await api.stepAll(state.run, { until });
    toast(`fast-forwarding ${r.advanced.length} slot${r.advanced.length === 1 ? "" : "s"} to ${until}`, "ok");
    refreshSlots();
  } catch (e) {
    toast(e.message, "err");
  }
});

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
            (ok < results.length ? " — rest skipped: running/done/legacy" : ""),
          ok ? "ok" : "err",
        );
        refreshSlots();
      } }),
    ],
  }));
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

// --- boot ------------------------------------------------------------------------

(async () => {
  const viewer = createViewer(document.getElementById("canvas-host"));
  initOverlay(viewer);
  initLab();
  initCompare();
  initCost();

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
