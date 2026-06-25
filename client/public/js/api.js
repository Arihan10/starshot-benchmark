// Typed wrappers for every server endpoint the client uses. All errors
// surface as Error(detail) so callers can toast them uniformly.

export const SERVER_URL = document
  .querySelector('meta[name="server-url"]')
  .getAttribute("content");

function u(path, params = {}) {
  const url = new URL(path, SERVER_URL);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null) url.searchParams.set(k, v);
  }
  return url;
}

async function request(path, { method = "GET", body, params } = {}) {
  const res = await fetch(u(path, params), {
    method,
    cache: "no-store",
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const data = await res.json();
      if (data && data.detail) detail = String(data.detail);
    } catch { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return res.json();
}

const cellPath = (slot, model, tail = "") =>
  `/slots/${encodeURIComponent(slot)}/${encodeURIComponent(model)}${tail}`;

// Read a JSONL event log straight from the artifact server (full history,
// cache.llm payloads included). Tolerates a torn final line written mid-flush.
async function readEventsJsonl(rel) {
  const res = await fetch(u(`/artifacts/${rel}`), { cache: "no-store" });
  if (!res.ok) return [];
  const text = await res.text();
  const out = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    try { out.push(JSON.parse(line)); } catch { /* torn tail line mid-write */ }
  }
  return out;
}

export const api = {
  // --- runs + versions ---
  runs: () => request("/runs"),
  versions: () => request("/versions"),
  createRun: (name, promptVersion) =>
    request("/runs", { method: "POST", body: { name, prompt_version: promptVersion } }),
  activateRun: (name) => request(`/runs/${encodeURIComponent(name)}/activate`, { method: "POST" }),
  // Load a run's cells into memory WITHOUT activating it, so its /scene +
  // /meshes are readable next to the active run (the run-compare view). No-op
  // if already loaded; launches nothing.
  hydrateRun: (name) => request(`/runs/${encodeURIComponent(name)}/hydrate`, { method: "POST" }),
  slots: (run) => request("/slots", { params: { run } }),

  // --- cell lifecycle ---
  // `stepped` opts the cell in/out of one-call-at-a-time execution
  // (null/undefined keeps its current mode).
  resume: (run, slot, model, stepped = null) =>
    request(cellPath(slot, model, "/resume"), { method: "POST", params: { run, stepped } }),
  // `auto` runs the cell to completion; `until` fast-forwards to the next run
  // of that step and pauses there. A plain step (neither) advances one call —
  // queued if the cell is mid-call, so a "step all" never skips a slow model.
  cellStep: (run, slot, model, { auto = false, until = null } = {}) =>
    request(cellPath(slot, model, "/step"), { method: "POST", params: { run }, body: { auto, until } }),
  stepAll: (run, { auto = false, until = null } = {}) =>
    request(`/runs/${encodeURIComponent(run)}/step-all`, { method: "POST", params: { auto, until } }),
  pause: (run, slot, model) => request(cellPath(slot, model, "/pause"), { method: "POST", params: { run } }),
  reset: (run, slot, model, start = false) =>
    request(cellPath(slot, model, "/reset"), { method: "POST", params: { run, start } }),
  // Revert a slot to before a given event: truncates its log there, drops the
  // dropped nodes' meshes, and leaves it paused at that point.
  rewind: (run, slot, model, toEventIndex) =>
    request(cellPath(slot, model, "/rewind"), { method: "POST", params: { run }, body: { to_event_index: toEventIndex } }),
  retryMesh: (run, slot, model, nodeId) =>
    request(cellPath(slot, model, `/retry-mesh/${encodeURIComponent(nodeId)}`), { method: "POST", params: { run } }),

  // --- source cell data ---
  //   untilIndex → project/bundle only the prefix BEFORE that event (the
  //     compare view's "previous" pane: the original's state at the fork step).
  scene: (run, slot, model, { untilIndex } = {}) =>
    request(cellPath(slot, model, "/scene"), { params: { run, until_index: untilIndex } }),
  eventsUrl: (run, slot, model, { since } = {}) =>
    u(cellPath(slot, model, "/events"), { run, since }).toString(),
  meshesUrl: (run, slot, model, { untilIndex } = {}) =>
    u(cellPath(slot, model, "/meshes"), { run, until_index: untilIndex }).toString(),
  // Full source-cell history backfill (cache.llm payloads included) from disk.
  async eventsHistory(run, slot, model) {
    return readEventsJsonl(`${run}/${slot}/${model}/events.jsonl`);
  },
  artifactUrl: (path) => u(`/artifacts/${path}`).toString(),
  absUrl: (path) => new URL(path, SERVER_URL).toString(),

  // --- simulation branches (keyed by branch id) ---
  // A branch is a first-class fork living in the run's flat `_branches/<id>/`
  // temp folder. Many per cell coexist; parallel-LLM previews are child
  // branches. All control is keyed by the opaque branch id.
  branchScene: (bid) => request(`/branches/${encodeURIComponent(bid)}/scene`),
  branchStatus: (bid) => request(`/branches/${encodeURIComponent(bid)}`),
  branchEventsUrl: (bid, { since } = {}) =>
    u(`/branches/${encodeURIComponent(bid)}/events`, { since }).toString(),
  //   sinceIndex → only the meshes placed at/after that event (a fan-out child's
  //     NEW meshes over the shared prefix already on screen).
  branchMeshesUrl: (bid, { sinceIndex } = {}) =>
    u(`/branches/${encodeURIComponent(bid)}/meshes`, { since_index: sinceIndex }).toString(),
  async branchEventsHistory(run, bid) {
    return readEventsJsonl(`${run}/_branches/${bid}/events.jsonl`);
  },

  // --- prompt lab ---
  promptTemplates: (run) => request(`/runs/${encodeURIComponent(run)}/prompt-templates`),
  // Every `{VARIABLE}` rendered against a fixed sample scene by the real
  // server-side injection functions — the lab's hover preview. Run-independent.
  variableSamples: () => request("/variable-samples"),
  stepEvents: (run, step) => request(`/runs/${encodeURIComponent(run)}/step-events`, { params: { step } }),
  stepEvent: (run, slot, model, index, step) =>
    request(`/runs/${encodeURIComponent(run)}/step-event`, { params: { run, slot, model, index, step } }),
  branchStepEvent: (bid, step, node) =>
    request(`/branches/${encodeURIComponent(bid)}/step-event`, { params: { step, node } }),
  promptTest: (body) => request("/prompt-test", { method: "POST", body }),
  // Fork a NEW simulation branch from a cell. `body` is {event_index, step,
  // overrides, seed?, model?}; returns {branch:{id, …}}. Many branches per cell
  // coexist (fork different zones — or LLMs — for parallel sims). `body.model`
  // (an alias) PINS the fork to that LLM and runs one step (compare's per-LLM
  // lineage); omit it to pause at the forked step on the cell's base model.
  createBranch: (run, slot, model, body) =>
    request(cellPath(slot, model, "/branch"), { method: "POST", params: { run }, body }),
  // Every TOP-LEVEL sim branch of a run (children/fan-out previews excluded).
  listBranches: (run) => request(`/runs/${encodeURIComponent(run)}/branches`, { params: { run } }),
  // `auto` runs the branch to completion; `until` (a template id) fast-forwards
  // it to the next call of that step. A plain step queues if the branch is
  // mid-call (so a batch "step sims" never errors on a not-yet-gated branch).
  // `modelOverride` (a model alias) re-aims the next gated call at a chosen LLM.
  branchStep: (bid, { auto = false, until = null, model = null } = {}) =>
    request(`/branches/${encodeURIComponent(bid)}/step`, { method: "POST", body: { auto, until, model } }),
  // Revert a branch to before `toEventIndex` and pause there; a following
  // branchStep re-runs from the cut under the current snapshot + `overrides`
  // (the lab's live edit set). `overrides=null` keeps the branch's edits.
  branchRewind: (bid, toEventIndex, overrides = null) =>
    request(`/branches/${encodeURIComponent(bid)}/rewind`, { method: "POST", body: { to_event_index: toEventIndex, overrides } }),
  // Promote a branch to BE its origin source cell (replace + discard).
  branchCommit: (bid) => request(`/branches/${encodeURIComponent(bid)}/commit`, { method: "POST" }),
  branchPause: (bid) => request(`/branches/${encodeURIComponent(bid)}/pause`, { method: "POST" }),
  branchResume: (bid) => request(`/branches/${encodeURIComponent(bid)}/resume`, { method: "POST" }),
  branchDiscard: (bid) => request(`/branches/${encodeURIComponent(bid)}`, { method: "DELETE" }),
  saveVersion: (body) => request("/versions/save", { method: "POST", body }),
  saveRunFromBranches: (body) => request("/runs/from-branches", { method: "POST", body }),
  forkVersion: (name, base) => request("/versions/fork", { method: "POST", body: { name, base } }),
  updateRunPrompts: (run, overrides, updateVersion) =>
    request(`/runs/${encodeURIComponent(run)}/prompt-templates`, {
      method: "PUT",
      body: { overrides, update_version: updateVersion },
    }),
  // Reset the run's snapshot back to its base source version (the inverse of
  // updateRunPrompts' version-sync) — discards the lab's in-place prompt edits.
  restoreRunPrompts: (run) =>
    request(`/runs/${encodeURIComponent(run)}/prompt-templates/restore`, { method: "POST" }),
  // Fork a simulation branch in each cell at its earliest call of any of
  // `steps` (non-destructive — source untouched). Replaces the old destructive
  // rerun-step.
  simulateStep: (run, steps, cells = null) =>
    request(`/runs/${encodeURIComponent(run)}/simulate-step`, { method: "POST", body: { steps, cells } }),

  // --- decision inquiry ---
  // Continue a step's own LLM conversation. `body` carries the step's `model`,
  // its system prompt, and the running `messages` thread (seeded with the
  // call's input + output); the reply comes from that same model. Returns
  // {answer, reasoning, model}.
  inquire: (body) => request("/inquire", { method: "POST", body }),
};
