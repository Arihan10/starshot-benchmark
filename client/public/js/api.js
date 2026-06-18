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

export const api = {
  // --- runs + versions ---
  runs: () => request("/runs"),
  versions: () => request("/versions"),
  createRun: (name, promptVersion) =>
    request("/runs", { method: "POST", body: { name, prompt_version: promptVersion } }),
  activateRun: (name) => request(`/runs/${encodeURIComponent(name)}/activate`, { method: "POST" }),
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

  // --- cell data ---
  //   branch:true → the cell's simulation fork.
  //   untilIndex → project/bundle only the source prefix BEFORE that event
  //     (the compare view's "previous" pane: the original's state at the fork
  //     step). Source-only; ignored for branch reads.
  scene: (run, slot, model, { branch = false, untilIndex } = {}) =>
    request(cellPath(slot, model, branch ? "/branch/scene" : "/scene"), { params: { run, until_index: untilIndex } }),
  eventsUrl: (run, slot, model, { branch = false, since } = {}) =>
    u(cellPath(slot, model, branch ? "/branch/events" : "/events"), { run, since }).toString(),
  meshesUrl: (run, slot, model, { branch = false, untilIndex } = {}) =>
    u(cellPath(slot, model, branch ? "/branch/meshes" : "/meshes"), { run, until_index: untilIndex }).toString(),
  // Full history backfill (cache.llm payloads included) straight from disk.
  async eventsHistory(run, slot, model, { branch = false } = {}) {
    const sub = branch ? "_branch/" : "";
    const rel = `${run}/${slot}/${model}/${sub}events.jsonl`;
    const res = await fetch(u(`/artifacts/${rel}`), { cache: "no-store" });
    if (!res.ok) return [];
    const text = await res.text();
    const out = [];
    for (const line of text.split("\n")) {
      if (!line.trim()) continue;
      try { out.push(JSON.parse(line)); } catch { /* torn tail line mid-write */ }
    }
    return out;
  },
  artifactUrl: (path) => u(`/artifacts/${path}`).toString(),
  absUrl: (path) => new URL(path, SERVER_URL).toString(),

  // --- prompt lab ---
  promptTemplates: (run) => request(`/runs/${encodeURIComponent(run)}/prompt-templates`),
  // Every `{VARIABLE}` rendered against a fixed sample scene by the real
  // server-side injection functions — the lab's hover preview. Run-independent.
  variableSamples: () => request("/variable-samples"),
  stepEvents: (run, step) => request(`/runs/${encodeURIComponent(run)}/step-events`, { params: { step } }),
  stepEvent: (run, slot, model, index, step) =>
    request(`/runs/${encodeURIComponent(run)}/step-event`, { params: { run, slot, model, index, step } }),
  branchStepEvent: (run, slot, model, step, node) =>
    request(`/runs/${encodeURIComponent(run)}/branch-step-event`, { params: { run, slot, model, step, node } }),
  promptTest: (body) => request("/prompt-test", { method: "POST", body }),
  createBranch: (run, slot, model, body) =>
    request(cellPath(slot, model, "/branch"), { method: "POST", params: { run }, body }),
  // `auto` runs the branch to completion; `until` (a template id) fast-forwards
  // it to the next call of that step. A plain step queues if the branch is
  // mid-call (so a batch "step sims" never errors on a not-yet-gated branch).
  // `modelOverride` (a model alias) re-aims the next gated call at a chosen LLM
  // (compare's per-step A/B); null keeps the branch's current model.
  branchStep: (run, slot, model, auto = false, until = null, modelOverride = null) =>
    request(cellPath(slot, model, "/branch/step"), { method: "POST", params: { run }, body: { auto, until, model: modelOverride } }),
  // Revert a simulation branch to before `toEventIndex` and pause there; a
  // following branchStep re-runs from the cut under the current snapshot +
  // `overrides` (the lab's live edit set). `overrides=null` keeps the branch's
  // existing edits.
  branchRewind: (run, slot, model, toEventIndex, overrides = null) =>
    request(cellPath(slot, model, "/branch/rewind"), { method: "POST", params: { run }, body: { to_event_index: toEventIndex, overrides } }),
  // Promote a cell's simulation branch to BE the source cell (replace + discard).
  commitBranch: (run, slot, model) =>
    request(cellPath(slot, model, "/branch/commit"), { method: "POST", params: { run } }),
  branchPause: (run, slot, model) =>
    request(cellPath(slot, model, "/branch/pause"), { method: "POST", params: { run } }),
  branchResume: (run, slot, model) =>
    request(cellPath(slot, model, "/branch/resume"), { method: "POST", params: { run } }),
  branchDiscard: (run, slot, model) =>
    request(cellPath(slot, model, "/branch"), { method: "DELETE", params: { run } }),
  saveVersion: (body) => request("/versions/save", { method: "POST", body }),
  saveRunFromBranches: (body) => request("/runs/from-branches", { method: "POST", body }),
  forkVersion: (name, base) => request("/versions/fork", { method: "POST", body: { name, base } }),
  updateRunPrompts: (run, overrides, updateVersion) =>
    request(`/runs/${encodeURIComponent(run)}/prompt-templates`, {
      method: "PUT",
      body: { overrides, update_version: updateVersion },
    }),
  // Fork a simulation branch in each cell at its earliest call of any of
  // `steps` (non-destructive — source untouched). Replaces the old destructive
  // rerun-step.
  simulateStep: (run, steps, cells = null) =>
    request(`/runs/${encodeURIComponent(run)}/simulate-step`, { method: "POST", body: { steps, cells } }),

  // --- decision inquiry ---
  // Ask the reviewer (Claude Opus 4.8, xhigh) why a step's subject model
  // decided what it did. `body` carries the step grounding (step/model/system/
  // user/output/reasoning) plus the running `messages` thread; returns
  // {answer, reasoning, model}.
  inquire: (body) => request("/inquire", { method: "POST", body }),
};
