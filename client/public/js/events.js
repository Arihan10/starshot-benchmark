// Event plumbing: the single SSE stream, the scene dispatch (events → 3D),
// and the observability model (events → node tree + LLM calls).

import { api } from "./api.js";

// --- SSE ----------------------------------------------------------------------

// One live stream at a time, owned by whichever panel is open. close() is
// idempotent; a terminal event closes the socket (status polls take over).
export function openStream(url, { onEvent, onTerminal } = {}) {
  const es = new EventSource(url);
  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    es.close();
  };
  es.onmessage = (ev) => {
    let event;
    try { event = JSON.parse(ev.data); } catch { return; }
    onEvent?.(event);
    if (["run.done", "run.error", "run.paused"].includes(event.kind)) {
      close();
      onTerminal?.(event);
    }
  };
  es.onerror = () => {
    // Browser auto-reconnects with a full snapshot replay; index dedup in the
    // consumers makes that harmless. A dead server just keeps erroring until
    // the panel is closed.
  };
  return { close };
}

// --- scene dispatch -------------------------------------------------------------

// The subset of event kinds that change the 3D scene. Mirrors the server's
// /scene projection cases.
export function dispatchSceneEvent(viewer, event) {
  switch (event.kind) {
    case "bbox":
      viewer.loadBbox(event);
      return;
    case "divider.zone_decompose":
    case "divider.decompose":
      for (const c of event.children ?? []) {
        if (c && typeof c === "object" && c.id) viewer.setKind(c.id, "zone");
      }
      return;
    case "model":
      if (event.url && !viewer.hasModel(event.id)) {
        viewer.loadModel(event, api.absUrl(event.url));
      }
      return;
    default:
  }
}

// Paint the server's CQRS /scene projection (fast first paint; meshes follow
// via the bundle stream + live model events).
export function applySceneProjection(viewer, projection) {
  for (const n of projection.nodes ?? []) {
    if (n.node_kind) viewer.setKind(n.id, n.node_kind);
    if (Array.isArray(n.origin) && Array.isArray(n.dimensions)) {
      viewer.loadBbox({
        id: n.id,
        origin: n.origin,
        dimensions: n.dimensions,
        node_kind: n.node_kind ?? "zone",
        proxy_shape: n.proxy_shape,
      });
    }
  }
}

// --- observability model ----------------------------------------------------------
//
// Folds the FULL event log (history backfill + live tail) into the node tree
// the obs dock renders: nodes (id/parent/kind/prompt/phase) each carrying its
// LLM calls in execution order. Calls keep the raw event object — the logged
// system/user/output/reasoning ARE the post-injection ground truth bytes.

// --- log strip classification -------------------------------------------------
//
// The non-verbose event log: lifecycle + every error/warning, with the
// firehose kinds (cache.llm payloads, bbox/step/image/model traffic,
// provider-job bookkeeping) left out. Errors are what you open an errored
// cell to read; retries/accepted-invalid rows explain degraded output.

const LOG_BUILDERS = {
  "run.start": (e) => ["info", `run started · ${e.model ?? ""}`],
  "run.done": () => ["info", "run done"],
  "run.paused": () => ["info", "run paused"],
  "run.error": (e) => ["error", e.message ?? "run error"],
  "mesh.error": (e) => ["error", `${e.id}: ${e.message ?? "mesh error"}`],
  "mesh.retry": (e) => ["warn", `${e.id}: mesh retried`],
  "mesh.submit": null,
  "llm.json_decode_error": (e) =>
    [e.final ? "error" : "warn", `LLM JSON decode${e.final ? " (final attempt)" : ""}: ${e.reason ?? ""}`],
  "llm.retry": (e) => ["warn", `LLM parse retry: ${e.reason ?? ""}`],
  "llm.transport_retry": (e) =>
    ["warn", `provider flap (attempt ${e.attempt ?? "?"}, backoff ${e.backoff_s ?? "?"}s): ${e.reason ?? ""}`],
  "llm.validation_retry": (e) =>
    ["warn", `${e.step ?? "step"}: output ids mismatched (attempt ${e.attempt ?? "?"}): ${e.reason ?? ""}`],
  "generation.decompose.retry": (e) =>
    ["warn", `${e.zone}: decompose rejected (attempt ${e.attempt ?? "?"}): ${e.reason ?? ""}`],
  "generation.next.retry": (e) =>
    ["warn", `${e.zone}: next-object rejected (attempt ${e.attempt ?? "?"}): ${e.reason ?? ""}`],
  "generation.decompose.accept_invalid": (e) =>
    ["warn", `${e.zone}: accepted decompose with dangling refs: ${e.reason ?? ""}`],
  "generation.next.accept_invalid": (e) =>
    ["warn", `${e.zone}: accepted next-object with dangling refs: ${e.reason ?? ""}`],
  "generation.next.stuck": (e) =>
    ["warn", `${e.zone}: completion loop stuck re-proposing ${e.id ?? "?"} — stopped`],
  "generation.dedup_drop": (e) =>
    ["warn", `${e.zone}: dropped duplicate ${e.id} (${e.reason ?? ""})`],
  "generation.decompose.no_bounding": (e) =>
    ["info", `${e.zone}: no bounding geometry needed`],
  "divider.validate.referenced_ids.accept_invalid": (e) =>
    ["warn", `${e.node}: accepted children with dangling refs: ${e.reason ?? ""}`],
  "library.asset_missing": (e) => ["warn", `${e.id}: library asset missing (${e.library_id})`],
  "library.bounds_missing": (e) => ["warn", `${e.id}: library bounds missing (${e.library_id})`],
};

function classifyLogEvent(event) {
  const builder = LOG_BUILDERS[event.kind];
  if (builder === null) return null;
  if (builder) {
    const [severity, text] = builder(event);
    return { index: event.index, kind: event.kind, severity, text };
  }
  // Unmapped kinds: surface anything that smells like a failure, skip the rest.
  if (/\.(error|failed)$/.test(event.kind ?? "")) {
    return {
      index: event.index,
      kind: event.kind,
      severity: "error",
      text: String(event.message ?? event.reason ?? ""),
    };
  }
  return null;
}

export function createObsModel() {
  const model = {
    nodes: new Map(),  // id -> node record
    order: [],         // node ids in first-seen order
    calls: [],         // every cache.llm event, log order
    log: [],           // filtered lifecycle/error/warning entries, log order
    errorCount: 0,
    maxIndex: -1,
  };

  function node(id) {
    let n = model.nodes.get(id);
    if (!n) {
      n = {
        id, parentId: null, prompt: null, plan: null, imagePrompt: null,
        kind: "zone", phase: null, calls: [], meshUrl: null, error: null,
      };
      model.nodes.set(id, n);
      model.order.push(id);
    }
    return n;
  }

  function feed(event) {
    const idx = typeof event.index === "number" ? event.index : null;
    if (idx !== null) {
      if (idx <= model.maxIndex) return false; // SSE reconnect replay — dedupe
      model.maxIndex = idx;
    }
    const entry = classifyLogEvent(event);
    if (entry) {
      model.log.push(entry);
      if (entry.severity === "error") model.errorCount += 1;
    }
    switch (event.kind) {
      case "cache.llm": {
        model.calls.push(event);
        node(event.node ?? "(unattributed)").calls.push(event);
        return true;
      }
      case "bbox": {
        const n = node(event.id);
        n.parentId = event.parent_id ?? n.parentId;
        n.prompt = event.prompt ?? n.prompt;
        n.kind = event.node_kind ?? n.kind;
        return true;
      }
      case "divider.zone_decompose":
      case "divider.decompose": {
        for (const c of event.children ?? []) {
          if (!c || typeof c !== "object" || !c.id) continue;
          const n = node(c.id);
          n.parentId = n.parentId ?? (c.parent || event.node);
          n.prompt = n.prompt ?? c.prompt;
          n.kind = "zone";
        }
        return true;
      }
      case "divider.zone_plan": {
        if (typeof event.node === "string" && typeof event.plan === "string") {
          node(event.node).plan = event.plan;
        }
        return true;
      }
      case "image": {
        if (typeof event.id === "string" && typeof event.prompt === "string") {
          node(event.id).imagePrompt = event.prompt;
        }
        return true;
      }
      case "step": {
        if (typeof event.node === "string") node(event.node).phase = event.phase ?? null;
        return true;
      }
      case "model": {
        const n = node(event.id);
        n.meshUrl = event.url ?? n.meshUrl;
        n.error = null;
        return true;
      }
      case "mesh.error": {
        node(event.id).error = event.message ?? "mesh error";
        return true;
      }
      default:
        return entry !== null || idx !== null;
    }
  }

  function reset() {
    model.nodes.clear();
    model.order.length = 0;
    model.calls.length = 0;
    model.log.length = 0;
    model.errorCount = 0;
    model.maxIndex = -1;
  }

  function lastError() {
    for (let i = model.log.length - 1; i >= 0; i--) {
      if (model.log[i].severity === "error") return model.log[i];
    }
    return null;
  }

  // The latest call of `templateStep` — feeds the pinned panel during
  // downstream simulation.
  function lastCallOf(templateStep) {
    for (let i = model.calls.length - 1; i >= 0; i--) {
      const c = model.calls[i];
      if (c.template === templateStep || (!c.template && c.step === templateStep)) return c;
    }
    return null;
  }

  return { model, feed, reset, lastCallOf, lastError };
}
