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
  scene: (run, slot, model, { branch = false } = {}) =>
    request(cellPath(slot, model, branch ? "/branch/scene" : "/scene"), { params: { run } }),
  eventsUrl: (run, slot, model, { branch = false, since } = {}) =>
    u(cellPath(slot, model, branch ? "/branch/events" : "/events"), { run, since }).toString(),
  meshesUrl: (run, slot, model, { branch = false } = {}) =>
    u(cellPath(slot, model, branch ? "/branch/meshes" : "/meshes"), { run }).toString(),
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
  stepEvents: (run, step) => request(`/runs/${encodeURIComponent(run)}/step-events`, { params: { step } }),
  stepEvent: (run, slot, model, index, step) =>
    request(`/runs/${encodeURIComponent(run)}/step-event`, { params: { run, slot, model, index, step } }),
  branchStepEvent: (run, slot, model, step, node) =>
    request(`/runs/${encodeURIComponent(run)}/branch-step-event`, { params: { run, slot, model, step, node } }),
  promptTest: (body) => request("/prompt-test", { method: "POST", body }),
  createBranch: (run, slot, model, body) =>
    request(cellPath(slot, model, "/branch"), { method: "POST", params: { run }, body }),
  branchStep: (run, slot, model, auto = false) =>
    request(cellPath(slot, model, "/branch/step"), { method: "POST", params: { run }, body: { auto } }),
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
  rerunStep: (run, steps, cells = null) =>
    request(`/runs/${encodeURIComponent(run)}/rerun-step`, { method: "POST", body: { steps, cells } }),
};
