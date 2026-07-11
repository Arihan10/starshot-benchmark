// Single mutable app state + a topic bus. Every module reads `state`,
// mutates it ONLY inside its own flow, and emits a topic so dependent
// panels re-render. Keeps cross-panel wiring explicit and greppable.

export const state = {
  run: null,            // active run name
  runs: [],             // [{name, modified_at, prompt_version}]
  versions: [],         // [name]
  models: [],           // model aliases (column order)
  defaultModel: null,
  slots: [],            // /slots payload: [{id, prompt, runs: {alias: cellSummary}}]
  steps: [],            // pipeline step/template ids (for "step until X"); fetched once

  // Board multi-select: an arbitrary set of (slot × model) cells to drive
  // resume/reset/step/pause on together — vs. acting on a single cell or all.
  // Per-run (cell keys repeat across runs), so a run switch clears it.
  selectMode: false,
  selection: new Set(), // cellKey ("slot|model") of every selected cell

  // Run-compare mode: clicking a board cell opens that (slot × model) from the
  // active run (A) and `compareRunB` (B) side by side, read-only. The slot/model
  // grid is run-independent, so the cell exists in both runs (B may be empty).
  compareMode: false,
  compareRunB: null,    // the run to compare the active run against

  // What the single 3D viewer is showing. null = board.
  view: null,           // {slot, model, branch: branchId|null}

  // Prompt lab session (survives navigating to overlays and back).
  lab: {
    open: false,
    step: null,            // selected template step
    templates: new Map(),  // step -> {system, user, variables}
    drafts: new Map(),     // step -> {system, user} (edited copies)
    events: [],            // step-events payload for `step`
    // The debug-grid selection, scoped PER cell so you can pick different zones
    // from different slots (no forced cross-product). A cell present in the map
    // is selected; its Set lists the node ids (zones) to show — an EMPTY set
    // means "every zone of this cell". Cell + node identities are
    // step-independent, so the choice survives a step switch: the grid just
    // re-materializes whichever of the new step's calls fall in each cell's
    // chosen zones.
    selection: new Map(),  // cellKey -> Set<node id>  (empty set ⇒ all zones)
    tests: new Map(),      // "slot|model|index" -> {status, result?, error?}
    // Which LLM(s) a selected cell's downstream simulations run on, keyed by
    // cellKey "slot|model" so every selected zone of a slot forks on the SAME
    // LLMs (NOT per event — the grid dedups to one card per cell in 3D). Absent ⇒
    // just the cell's base model (one branch, today's behavior); a multi-alias
    // set forks one pinned lineage per LLM so a single "simulate" A/Bs the step
    // across models. Step-independent like `selection`; reset on a run switch.
    simModels: new Map(),  // cellKey "slot|model" -> Set<model alias>
    // Live simulation branches keyed by their server branch id (NOT the cell),
    // so one cell can carry several at once (different zones / parallel sims).
    sims: new Map(),       // branchId -> {id, slot, model, eventIndex, createdAt}
    simStep: null,         // step the simulation branched at (pins obs panel)
    simEditedSteps: [],    // every step whose edit the live branches carry
    // A second prompt version to A/B against: when set, "simulate downstream"
    // forks an extra lineage per target under this version, side by side with
    // one under the run's current prompts. null ⇒ single-version simulate.
    compareVersion: null,
    // Zone node ids the user has locked to atomic for their next simulation:
    // each forked branch forces `is_atomic=true` on its zone, so it's tested as
    // a leaf (no further decomposition). Step-independent like `selection`.
    atomicLocks: new Set(),
  },
};

export const cellKey = (slot, model) => `${slot}|${model}`;
export const targetKey = (slot, model, index) => `${slot}|${model}|${index}`;

const listeners = new Map(); // topic -> Set<fn>

export function on(topic, fn) {
  if (!listeners.has(topic)) listeners.set(topic, new Set());
  listeners.get(topic).add(fn);
  return () => listeners.get(topic).delete(fn);
}

export function emit(topic, payload) {
  for (const fn of listeners.get(topic) ?? []) {
    try { fn(payload); } catch (e) { console.error(`[bus:${topic}]`, e); }
  }
}

export function cellSummary(slot, model) {
  const s = state.slots.find((x) => x.id === slot);
  return s?.runs?.[model] ?? null;
}

// The cell's TOP-LEVEL simulation branches (the /slots poll embeds them per
// cell; fan-out children are not listed here). Always an array.
export function cellBranches(slot, model) {
  return cellSummary(slot, model)?.branches ?? [];
}

// Find a branch summary anywhere in the current /slots snapshot by its id.
export function branchSummaryById(bid) {
  if (!bid) return null;
  for (const s of state.slots) {
    for (const alias of Object.keys(s.runs ?? {})) {
      const b = (s.runs[alias].branches ?? []).find((x) => x.id === bid);
      if (b) return b;
    }
  }
  return null;
}
