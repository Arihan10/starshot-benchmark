// The observability dock: node tree with every LLM call — each call expands
// IN PLACE to its system / user / output bytes (ground truth from the event
// log) — plus the pinned "last executed event of the edited step" panel used
// during downstream simulation and the error/lifecycle log strip.

import { el, fitToggle, fmtJson, shortBytes, foldedPre } from "./ui.js";
import { callEmits } from "./obsmini.js";

const bodyEl = document.getElementById("obs-body");
const pinnedEl = document.getElementById("obs-pinned");
const countEl = document.getElementById("obs-count");
const logEl = document.getElementById("obs-log");
const logBodyEl = document.getElementById("obs-log-body");
const logCountsEl = document.getElementById("obs-log-counts");

document.getElementById("obs-log-head").addEventListener("click", () => {
  logUserToggled = true;
  logEl.classList.toggle("open");
});

// The log strip auto-opens the first time an error lands (the whole point of
// opening an errored cell), unless the user has explicitly collapsed it.
let logUserToggled = false;
let logAutoOpened = false;

let pinStep = null;
let selectedNodeId = null;
// When set, the dock shows the emit TRACE of this node (its root→node lineage
// with the calls that created it) instead of the full tree. Cleared by the
// "← full pipeline" button and on dock reset.
let focusId = null;
// Overlay-provided: clicking a node row (or a call) focuses its bbox in 3D.
let onNodeClick = () => {};
// Overlay-provided: "revert this slot to before this call" (truncate the log,
// then re-run from there). null hides the per-call revert affordance (e.g.
// when viewing a branch).
let onRevert = null;
// Overlay-provided: "ask why the model did this" — opens the decision-inquiry
// chat grounded in this call. Read-only, so it's offered on branch calls too.
let onInquire = null;
// Overlay-provided: "add this zone to the prompt lab's downstream-simulation
// slots". null hides the affordance (e.g. on a branch view — you simulate from
// the source, not from an existing branch).
let onAddSim = null;
// Overlay-provided hidden-node API — the SAME set the canvas right-click
// toggles, so the eye buttons and right-click stay in sync.
let hiddenApi = { isHidden: () => false, toggle: () => {} };

export function setPinStep(step) {
  pinStep = step;
  pinnedEl.classList.toggle("visible", !!step);
}

export function setOnNodeClick(fn) {
  onNodeClick = fn;
}

export function setOnRevert(fn) {
  onRevert = fn;
}

export function setOnInquire(fn) {
  onInquire = fn;
}

export function setOnAddSim(fn) {
  onAddSim = fn;
}

export function setHiddenApi(api) {
  hiddenApi = api;
}

// Selection sync FROM the 3D viewer: highlight the row and reveal it.
export function markSelected(id, { scroll = true } = {}) {
  selectedNodeId = id;
  for (const row of bodyEl.querySelectorAll(".node-row.selected")) {
    row.classList.remove("selected");
  }
  if (id === null) return;
  const row = bodyEl.querySelector(`.node-row[data-id="${CSS.escape(id)}"]`);
  if (row) {
    row.classList.add("selected");
    if (scroll) row.scrollIntoView({ block: "nearest" });
  }
}

// Calls expanded in place, keyed by event index so the accordion state
// survives the full re-renders that streamed events trigger.
const expandedCalls = new Set();
let currentModel = null; // last-rendered model, for expandCall re-renders
// Streamed events re-call renderTree constantly. Repaint the body only when
// what it actually shows changed (these track the last painted signature +
// view), so reading a node's trace mid-run isn't wiped — and scrolled back to
// the top — every time an unrelated step finishes.
let lastBodySig = null;
let lastViewKey = null;

function toggleCall(call) {
  if (expandedCalls.has(call.index)) expandedCalls.delete(call.index);
  else expandedCalls.add(call.index);
  if (currentModel) renderTree(currentModel);
}

// Expand a specific call (pinned panel's "open") and bring it into view.
export function expandCall(call) {
  focusId = null; // show the full tree so the call is findable
  expandedCalls.add(call.index);
  if (currentModel) renderTree(currentModel);
  const row = bodyEl.querySelector(`[data-call="${call.index}"]`);
  if (row) row.scrollIntoView({ block: "start" });
}

function callRow(call, emitFocus) {
  const seeded = call.seeded === true;
  const expanded = expandedCalls.has(call.index);
  const emits = callEmits(call, emitFocus);
  const wrap = el("div", { class: "obs-call-wrap", dataset: { call: call.index ?? "" } });
  wrap.appendChild(
    el(
      "div",
      {
        class: `obs-call${call.template === pinStep ? " hl" : ""}${seeded ? " seeded" : ""}${expanded ? " open" : ""}${emits ? " emits" : ""}`,
        onclick: () => {
          // Expanding a call also focuses its node in 3D (zoom + dim rest), so
          // the bytes on screen and the box on canvas stay tied together.
          if (typeof call.node === "string") onNodeClick(call.node, { ensureSelected: true });
          toggleCall(call);
        },
        title: seeded ? "seeded from a prompt-lab test result" : "",
      },
      el("span", { class: "call-caret", text: expanded ? "▾" : "▸" }),
      el("span", { class: "step-badge", text: call.template ?? call.step ?? "?" }),
      emits ? el("span", { class: "emit-badge", text: "emitted here" }) : null,
      el("span", { class: "muted", text: call.schema ?? "" }),
      el("span", {
        class: "call-meta",
        text: `#${call.index ?? "?"} · ${call.tokens_out ?? "?"} tok${seeded ? " · seeded" : ""}`,
      }),
      onRevert ? el("button", {
        class: "call-revert",
        text: "⏪ revert",
        title: "revert this slot to just before this step — truncates the log here, drops every later step + its meshes, then re-runs from here",
        onclick: (ev) => { ev.stopPropagation(); onRevert(call); },
      }) : null,
      onInquire ? el("button", {
        class: "call-ask",
        text: "why?",
        title: "continue this step's conversation with the model that made it — ask it anything",
        onclick: (ev) => { ev.stopPropagation(); onInquire(call); },
      }) : null,
    ),
  );
  if (expanded) wrap.appendChild(callDetail(call));
  return wrap;
}

// The in-place detail: exact bytes sent/received. System, user, and output
// are open; reasoning and per-variable values stay one click away.
function callDetail(call) {
  const detail = el("div", { class: "call-detail" });
  detail.appendChild(el("div", {
    class: "muted",
    style: "margin-bottom:6px",
    text: [
      call.model ?? "",
      `tokens ${call.tokens_in ?? "?"} in / ${call.tokens_out ?? "?"} out`,
      call.seeded ? "SEEDED from a vetted prompt-lab test (no live call)" : null,
    ].filter(Boolean).join(" · "),
  }));
  detail.appendChild(section("system — exact bytes sent", call.system ?? "", { variables: call.variables }));
  detail.appendChild(section("user — exact bytes sent", call.user ?? "", { variables: call.variables }));
  detail.appendChild(section("output — exact bytes received", fmtJson(call.output)));
  if (call.reasoning) detail.appendChild(section("reasoning", call.reasoning, { open: false }));
  if (call.variables && typeof call.variables === "object") {
    const wrap = el("div", { class: "detail-section" },
      el("div", { class: "lab", text: `resolved variables (${Object.keys(call.variables).length}) — what each token was injected as` }),
    );
    for (const [name, value] of Object.entries(call.variables)) {
      const row = el("div", { class: "var-row" });
      const pre = el("pre", { text: String(value) });
      row.appendChild(
        el("div", { class: "var-head", onclick: () => row.classList.toggle("open") },
          el("span", { text: `\`{${name}}\`` }),
          el("span", { class: "size", text: shortBytes(String(value)) }),
          fitToggle(pre),
        ),
      );
      row.appendChild(pre);
      wrap.appendChild(row);
    }
    detail.appendChild(wrap);
  }
  detail.addEventListener("click", (ev) => ev.stopPropagation());
  return detail;
}

// Clicking a node focuses its bbox in 3D AND shows its emit trace in the dock.
function selectNode(id) {
  onNodeClick(id, { toggle: true });
  focusId = id;
  if (currentModel) renderTree(currentModel);
}

function nodeRowEl(model, id) {
  const n = model.nodes.get(id);
  const hidden = hiddenApi.isHidden(id);
  // "via {step}" — the decomposition step that named this node, shown inline so
  // its origin (anchor vs next_object vs encapsulating vs negative_space vs
  // zone_decompose) reads at a glance without expanding. Click the row for the
  // full "generated by {step} on {region}" provenance + exact bytes.
  const emittedBy = (model.provenance?.get(id) ?? []).find((p) => p.relation === "emitted_by");
  const viaStep = emittedBy ? (emittedBy.call.template ?? emittedBy.call.step ?? "?") : null;
  return el("div", {
    class: `node-row${id === selectedNodeId ? " selected" : ""}${hidden ? " hidden-node" : ""}`,
    dataset: { id },
    title: n.prompt ?? "",
    onclick: () => selectNode(id),
  },
    el("span", { class: `dot ${n.error ? "error" : n.phase === "done" ? "done" : n.calls.length ? "running" : "idle"}` }),
    el("span", { class: "node-id", text: id }),
    el("span", { class: "node-kind", text: `${n.kind}${n.phase ? ` · ${n.phase}` : ""}` }),
    viaStep ? el("span", {
      class: "obs-via",
      text: `via ${viaStep}`,
      title: `generated by ${viaStep} on ${emittedBy.call.node ?? "?"}`,
    }) : null,
    el("button", {
      class: "node-eye",
      text: hidden ? "🚫" : "👁",
      title: hidden
        ? "show this node's mesh again (or right-click its bbox in 3D)"
        : "hide this node's mesh — hiding a zone hides its whole subtree (or right-click it in 3D)",
      onclick: (ev) => {
        ev.stopPropagation();
        hiddenApi.toggle(id);
      },
    }),
    // Zones only: drop this zone into the prompt lab's downstream-simulation
    // slots so you can iterate on / fork its prompts from the scene directly.
    onAddSim && n.kind === "zone" ? el("button", {
      class: "node-sim",
      text: "+ sim",
      title: "add this zone to the prompt lab's downstream-simulation slots",
      onclick: (ev) => {
        ev.stopPropagation();
        onAddSim(id);
      },
    }) : null,
  );
}

function nodeBlock(model, id, depth) {
  const n = model.nodes.get(id);
  const children = model.order.filter((cid) => model.nodes.get(cid)?.parentId === id);
  const block = el("div", { class: "obs-node", style: depth ? `margin-left:${Math.min(depth, 6) * 10}px` : "" });
  block.appendChild(nodeRowEl(model, id));
  if (n.calls.length) {
    block.appendChild(el("div", { class: "obs-calls" }, n.calls.map((c) => callRow(c))));
  }
  for (const cid of children) block.appendChild(nodeBlock(model, cid, depth + 1));
  return block;
}

// A compact fingerprint of everything the body currently DRAWS, so a streamed
// re-render can no-op when nothing visible changed. Trace mode keys off the
// focused node's lineage (other nodes finishing don't touch it); full-tree mode
// keys off every node. Selection is excluded — `markSelected` patches the row
// class directly, so it never needs a full repaint.
function bodySignature(model) {
  const sig = (id) => {
    const n = model.nodes.get(id);
    if (!n) return `${id}:x`;
    const e = (model.provenance?.get(id) ?? []).find((p) => p.relation === "emitted_by");
    const via = e ? (e.call.template ?? e.call.step ?? "") : "";
    return `${id}:${n.parentId ?? ""}:${n.kind}:${n.phase ?? ""}:${n.calls.length}:${hiddenApi.isHidden(id) ? 1 : 0}:${n.error ? 1 : 0}:${via}`;
  };
  const exp = [...expandedCalls].sort((a, b) => a - b).join(",");
  if (focusId !== null && model.nodes.has(focusId)) {
    const lineage = [];
    const seen = new Set();
    let cur = focusId;
    while (cur && model.nodes.has(cur) && !seen.has(cur)) {
      seen.add(cur);
      lineage.unshift(cur);
      cur = model.nodes.get(cur).parentId;
    }
    const prov = (model.provenance?.get(focusId) ?? []).map((p) => `${p.relation}#${p.call.index}`).join(",");
    return `trace|${focusId}|exp:${exp}|prov:${prov}|${lineage.map(sig).join(";")}`;
  }
  return `tree|exp:${exp}|${model.order.map(sig).join(";")}`;
}

export function renderTree(model, { streamed = false } = {}) {
  currentModel = model;
  countEl.textContent = `${model.calls.length} llm calls · ${model.nodes.size} nodes`;
  const viewKey = focusId !== null && model.nodes.has(focusId) ? `trace:${focusId}` : "tree";
  const sig = bodySignature(model);
  // A live (streamed) update must NOT tear the body down while the user is
  // engaged with it — scrolled in, or reading an expanded call. Rebuilding from
  // the model re-creates every call's detail with default-open sections (the
  // inner "hide"/fold state is ephemeral DOM, tracked nowhere), so a collapsed
  // prompt springs back open and the re-expansion shoves what you were reading
  // out from under the scroll. So we hold the body steady for streamed events
  // while engaged; user-driven renders (open, navigate, expand/collapse, hide)
  // always repaint, and the log strip + 3D scene keep updating live regardless.
  const engaged = bodyEl.scrollTop > 0 || expandedCalls.size > 0;
  if (sig !== lastBodySig && !(streamed && engaged)) {
    // Preserve scroll across a same-view repaint (a live data update); switching
    // view (into/out of a trace) starts at the top.
    const prevScroll = viewKey === lastViewKey ? bodyEl.scrollTop : 0;
    lastBodySig = sig;
    lastViewKey = viewKey;
    bodyEl.textContent = "";
    if (focusId !== null && model.nodes.has(focusId)) {
      renderTrace(model, focusId);
    } else {
      focusId = null;
      const roots = model.order.filter((id) => {
        const p = model.nodes.get(id)?.parentId;
        return !p || !model.nodes.has(p);
      });
      for (const id of roots) bodyEl.appendChild(nodeBlock(model, id, 0));
    }
    bodyEl.scrollTop = prevScroll;
  }
  if (pinStep) renderPinned(model);
  renderLog(model);
}

// "How this node was generated": the decompose call that NAMED it (emitted_by)
// and the bbox-batch call that PLACED it (placed_by), each shown as
// "generated/placed by {step} on {region}" with the region click-to-focus and
// the full call expandable to its exact system / user / output / reasoning
// bytes. These calls ran on the node's region, not on the node itself, so this
// is the only place they surface for an object — especially a negative-space
// object whose region isn't its structural parent. Null when nothing names it
// (e.g. the root).
function provenanceBlock(model, fid) {
  const prov = model.provenance?.get(fid) ?? [];
  if (!prov.length) return null;
  const wrap = el("div", { class: "obs-prov" },
    el("div", { class: "obs-prov-lab", text: "how this node was generated" }),
  );
  for (const entry of prov) {
    const step = entry.call.template ?? entry.call.step ?? "?";
    const region = entry.call.node ?? "?";
    const linked = region !== "?" && model.nodes.has(region);
    wrap.appendChild(el("div", { class: `obs-prov-rel ${entry.relation}` },
      el("span", {
        class: "obs-prov-tag",
        text: entry.relation === "emitted_by" ? "generated by" : "placed by",
      }),
      el("span", {
        class: `obs-prov-on${linked ? " link" : ""}`,
        text: `${step} on ${region}`,
        onclick: linked ? () => selectNode(region) : null,
      }),
    ));
    wrap.appendChild(callRow(entry.call, fid));
  }
  return wrap;
}

// The emit trace of one node: root → … → node, each lineage node with the LLM
// calls attributed to it (the call that actually emitted the node is badged).
function renderTrace(model, fid) {
  const lineage = [];
  const seen = new Set();
  let cur = fid;
  while (cur && model.nodes.has(cur) && !seen.has(cur)) {
    seen.add(cur);
    lineage.unshift(cur);
    cur = model.nodes.get(cur).parentId;
  }
  const crumbs = el("div", { class: "obs-crumbs" });
  lineage.forEach((id, i) => {
    if (i) crumbs.appendChild(el("span", { class: "obs-crumb-sep", text: "›" }));
    crumbs.appendChild(el("span", {
      class: `obs-crumb${id === fid ? " cur" : ""}`,
      text: id,
      onclick: id === fid ? null : () => selectNode(id),
    }));
  });
  bodyEl.appendChild(el("div", { class: "obs-trace-head" },
    el("button", { class: "obs-back", text: "← full pipeline", onclick: () => { focusId = null; renderTree(currentModel); } }),
    crumbs,
  ));
  const prov = provenanceBlock(model, fid);
  if (prov) bodyEl.appendChild(prov);
  for (const id of lineage) {
    const n = model.nodes.get(id);
    const block = el("div", { class: `obs-node${id === fid ? " trace-focus" : ""}` });
    block.appendChild(nodeRowEl(model, id));
    block.appendChild(n.calls.length
      ? el("div", { class: "obs-calls" }, n.calls.map((c) => callRow(c, fid)))
      : el("div", { class: "muted", style: "margin:2px 0 6px 16px", text: "— no calls attributed to this node" }));
    bodyEl.appendChild(block);
  }
}

export function renderLog(model) {
  const warns = model.log.length - model.errorCount -
    model.log.filter((e) => e.severity === "info").length;
  logCountsEl.textContent =
    `${model.log.length} entries` +
    (model.errorCount ? ` · ${model.errorCount} error${model.errorCount === 1 ? "" : "s"}` : "") +
    (warns > 0 ? ` · ${warns} warn` : "");
  logCountsEl.classList.toggle("has-errors", model.errorCount > 0);
  if (model.errorCount > 0 && !logAutoOpened && !logUserToggled) {
    logAutoOpened = true;
    logEl.classList.add("open");
  }
  logBodyEl.textContent = "";
  for (const entry of model.log) {
    logBodyEl.appendChild(
      el("div", { class: `log-row ${entry.severity}` },
        el("span", { class: "idx", text: `#${entry.index ?? "?"}` }),
        el("span", { class: "kind", text: entry.kind }),
        el("span", { class: "msg", text: entry.text, title: entry.text }),
      ),
    );
  }
  // Newest entries matter most while streaming — keep the tail in view.
  if (logEl.classList.contains("open")) logBodyEl.scrollTop = logBodyEl.scrollHeight;
}

// --- pinned last-event-of-step panel ---------------------------------------------

let lastPinnedCall = null;

export function renderPinned(model, lastCallOf) {
  if (!pinStep) return;
  const call = (lastCallOf ?? defaultLastCallOf)(model, pinStep);
  if (call === lastPinnedCall && pinnedEl.childElementCount > 0) return;
  lastPinnedCall = call;
  pinnedEl.textContent = "";
  pinnedEl.appendChild(
    el("div", { class: "pin-label" },
      el("span", { text: `last ${pinStep} call in this slot` }),
      call ? el("button", {
        style: "margin-left:auto;font-size:10px;padding:1px 7px",
        text: "open in tree",
        onclick: () => expandCall(call),
      }) : null,
    ),
  );
  if (!call) {
    pinnedEl.appendChild(el("div", { class: "muted", text: "not executed yet in this branch" }));
    return;
  }
  pinnedEl.appendChild(pinSection("input (user)", call.user ?? "", call.variables));
  pinnedEl.appendChild(pinSection("output", fmtJson(call.output)));
  if (call.reasoning) pinnedEl.appendChild(pinSection("reasoning", call.reasoning));
}

function defaultLastCallOf(model, step) {
  for (let i = model.calls.length - 1; i >= 0; i--) {
    const c = model.calls[i];
    if (c.template === step || (!c.template && c.step === step)) return c;
  }
  return null;
}

function pinSection(label, text, variables = null) {
  const pre = variables ? foldedPre(text, variables) : el("pre", { text });
  return el("div", { class: "pin-section" },
    el("div", { class: "lab" }, el("span", { text: `${label} · ${shortBytes(text)}` }), fitToggle(pre)),
    pre,
  );
}

// --- shared section builder (ground-truth bytes) -----------------------------------

function section(label, text, { open = true, variables = null } = {}) {
  // With `variables`, fold the big scene-context blocks behind expanders so
  // the prompt's own wording shows by default. Copy still copies full bytes.
  const pre = variables ? foldedPre(text, variables) : el("pre", { text });
  if (!open) pre.style.display = "none";
  const toggle = el("button", {
    style: "font-size:10px;padding:1px 7px",
    text: open ? "hide" : "show",
    onclick: () => {
      const shown = pre.style.display !== "none";
      pre.style.display = shown ? "none" : "";
      toggle.textContent = shown ? "show" : "hide";
    },
  });
  const copy = el("button", {
    style: "font-size:10px;padding:1px 7px",
    text: "copy",
    onclick: () => navigator.clipboard?.writeText(text),
  });
  return el("div", { class: "detail-section" },
    el("div", { class: "lab" },
      el("span", { text: `${label} · ${shortBytes(text)}` }), toggle, copy, fitToggle(pre)),
    pre,
  );
}

export function resetDock() {
  bodyEl.textContent = "";
  pinnedEl.textContent = "";
  lastPinnedCall = null;
  selectedNodeId = null;
  currentModel = null;
  focusId = null;
  lastBodySig = null;
  lastViewKey = null;
  expandedCalls.clear();
  countEl.textContent = "";
  logBodyEl.textContent = "";
  logCountsEl.textContent = "";
  logCountsEl.classList.remove("has-errors");
  logEl.classList.remove("open");
  logUserToggled = false;
  logAutoOpened = false;
}
