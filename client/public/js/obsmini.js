// Self-contained observability tree renderer for the scenes grid. Unlike the
// singleton overlay dock (obstree.js), this renders a folded obs model
// (events.js createObsModel) into ANY container — so every scene tile can show
// a mini pipeline and the right-side drawer can show the full one.
//
// `detailed:false` (tiles) → just the node hierarchy + status (a glanceable
// pipeline). `detailed:true` (drawer) → each node's LLM calls, expandable in
// place to their exact system / user / output bytes.

import { el, fmtJson, foldedPre } from "./ui.js";

function statusDot(n) {
  if (n.error) return "error";
  if (n.phase === "done") return "done";
  return n.calls.length ? "running" : "idle";
}

export function renderObsTree(container, model, { detailed = false, expanded = null, onToggle = null, onRevert = null } = {}) {
  container.textContent = "";
  if (!model || !model.order.length) {
    container.appendChild(el("div", { class: "obsm-empty", text: "no pipeline yet" }));
    return;
  }
  const roots = model.order.filter((id) => {
    const p = model.nodes.get(id)?.parentId;
    return !p || !model.nodes.has(p);
  });
  for (const id of roots) container.appendChild(nodeBlock(model, id, 0, { detailed, expanded, onToggle, onRevert }));
}

function nodeBlock(model, id, depth, opts) {
  const n = model.nodes.get(id);
  const children = model.order.filter((cid) => model.nodes.get(cid)?.parentId === id);
  const block = el("div", { class: "obsm-node", style: depth ? `margin-left:${Math.min(depth, 6) * 9}px` : "" });
  const tail = n.phase
    ? el("span", { class: "obsm-phase", text: n.phase })
    : (n.calls.length ? el("span", { class: "obsm-phase", text: `${n.calls.length}×` }) : null);
  block.appendChild(el("div", {
    class: `obsm-row${opts.onNodeClick ? " clickable" : ""}`,
    title: opts.onNodeClick ? `trace what emitted ${id}` : (n.prompt ?? ""),
    onclick: opts.onNodeClick ? () => opts.onNodeClick(id) : null,
  },
    el("span", { class: `dot ${statusDot(n)}` }),
    el("span", { class: "obsm-id", text: id }),
    tail,
  ));
  if (opts.detailed && n.calls.length) {
    block.appendChild(el("div", { class: "obsm-calls" }, n.calls.map((c) => callRow(c, opts))));
  }
  for (const cid of children) block.appendChild(nodeBlock(model, cid, depth + 1, opts));
  return block;
}

// The full lineage trace of one node: root → … → node, each lineage node with
// the LLM calls attributed to it (expandable to exact bytes). The call(s)
// whose output references the node — the step that actually emitted it — are
// badged "emitted here".
export function renderObsTrace(container, model, focusId, { expanded = null, onToggle = null, onRevert = null, onNavigate = null, onBack = null } = {}) {
  container.textContent = "";
  if (!model || !model.nodes.has(focusId)) {
    container.appendChild(el("div", { class: "obsm-empty", text: "node not found" }));
    return;
  }
  const lineage = [];
  const seen = new Set();
  let cur = focusId;
  while (cur && model.nodes.has(cur) && !seen.has(cur)) {
    seen.add(cur);
    lineage.unshift(cur);
    cur = model.nodes.get(cur).parentId;
  }
  const crumbs = el("div", { class: "obsm-crumbs" });
  lineage.forEach((id, i) => {
    if (i) crumbs.appendChild(el("span", { class: "obsm-crumb-sep", text: "›" }));
    crumbs.appendChild(el("span", {
      class: `obsm-crumb${id === focusId ? " cur" : ""}`,
      text: id,
      onclick: id === focusId ? null : () => onNavigate?.(id),
    }));
  });
  container.appendChild(el("div", { class: "obsm-trace-head" },
    el("button", { class: "obsm-back", text: "← full pipeline", onclick: () => onBack?.() }),
    crumbs,
  ));
  for (const id of lineage) {
    const n = model.nodes.get(id);
    const calls = n.calls.length
      ? el("div", { class: "obsm-calls" }, n.calls.map((c) => callRow(c, { expanded, onToggle, onRevert, emitFocus: focusId })))
      : el("div", { class: "obsm-empty", text: "— no calls attributed to this node" });
    container.appendChild(el("div", { class: `obsm-trace-node${id === focusId ? " focus" : ""}` },
      el("div", { class: "obsm-row" },
        el("span", { class: `dot ${statusDot(n)}` }),
        el("span", { class: "obsm-id", text: id }),
        n.phase ? el("span", { class: "obsm-phase", text: n.phase }) : null,
      ),
      calls,
    ));
  }
}

// Does this call's output reference `focusId` (i.e. it's the step that emitted
// that node)? Quoted match so `"tree"` doesn't false-match `"tree_trunk"`.
export function callEmits(call, focusId) {
  if (!focusId) return false;
  try {
    const s = typeof call.output === "string" ? call.output : JSON.stringify(call.output ?? "");
    return s.includes(`"${focusId}"`) || s.includes(`'${focusId}'`);
  } catch { return false; }
}

function callRow(call, opts) {
  const isOpen = opts.expanded?.has(call.index);
  const emits = callEmits(call, opts.emitFocus);
  const wrap = el("div", { class: "obsm-call-wrap" });
  wrap.appendChild(el("div", { class: `obsm-call${isOpen ? " open" : ""}${emits ? " emits" : ""}`, onclick: () => opts.onToggle?.(call) },
    el("span", { class: "caret", text: isOpen ? "▾" : "▸" }),
    el("span", { class: "step-badge", text: call.template ?? call.step ?? "?" }),
    emits ? el("span", { class: "emit-badge", text: "emitted here" }) : null,
    el("span", { class: "muted", text: `#${call.index ?? "?"} · ${call.tokens_out ?? "?"} tok` }),
    opts.onRevert ? el("button", {
      class: "call-revert",
      text: "⏪",
      title: "revert this slot to just before this step — truncates the log here and drops every later step + its meshes",
      onclick: (ev) => { ev.stopPropagation(); opts.onRevert(call); },
    }) : null,
  ));
  if (isOpen) wrap.appendChild(callDetail(call));
  return wrap;
}

function callDetail(call) {
  const d = el("div", { class: "obsm-detail" });
  d.appendChild(section("system — exact bytes sent", call.system ?? "", call.variables));
  d.appendChild(section("user — exact bytes sent", call.user ?? "", call.variables));
  d.appendChild(section("output — exact bytes received", fmtJson(call.output)));
  if (call.reasoning) d.appendChild(section("reasoning", call.reasoning));
  d.addEventListener("click", (ev) => ev.stopPropagation()); // don't collapse on inner clicks
  return d;
}

function section(label, text, variables) {
  return el("div", { class: "obsm-sec" },
    el("div", { class: "obsm-sec-lab", text: label }),
    variables ? foldedPre(text, variables) : el("pre", { text }),
  );
}
