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

  // What the single 3D viewer is showing. null = board.
  view: null,           // {slot, model, branch: bool}

  // Prompt lab session (survives navigating to overlays and back).
  lab: {
    open: false,
    step: null,            // selected template step
    templates: new Map(),  // step -> {system, user, variables}
    drafts: new Map(),     // step -> {system, user} (edited copies)
    events: [],            // step-events payload for `step`
    selected: new Set(),   // "slot|model|index" keys
    tests: new Map(),      // "slot|model|index" -> {status, result?, error?}
    sims: new Map(),       // "slot|model" -> {slot, model, eventIndex}
    simStep: null,         // step the simulation branched at (pins obs panel)
    simEditedSteps: [],    // every step whose edit the live branches carry
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
