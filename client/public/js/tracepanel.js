// The emittance-trace inspector: a left panel in the single-cell overlay that
// appears when an object/zone is picked in 3D and shows how it was generated —
// the chain of REGIONS that emitted it (root → … → node, climbing each node's
// emitting region, NOT its structural parent) with the LLM calls that brought
// each into existence.
//
// Layout per node: a collapsible row (the focused node open, its ancestors
// collapsed) whose body is the calls that generated that node — its emitted_by /
// placed_by calls plus any step that ran on it — each call expandable to its
// bytes. In a call, the OUTPUT is truncated to just the section about that node
// (expand for the full output); input / system / reasoning start collapsed.
//
// It reads the folded obs model + provenance (events.js); it owns only its
// open + expanded state and a live-refresh guard so a streamed event never tears
// down something the user is reading. The overlay drives it off 3D selection.

import { el, foldedPre, fmtJson, shortBytes } from "./ui.js";
import { emittanceLineage, extractRelevantOutput } from "./events.js";

function statusDot(n) {
  if (!n) return "idle";
  if (n.error) return "error";
  if (n.phase === "done") return "done";
  return n.calls.length ? "running" : "idle";
}

export function createTracePanel(hostEl, { onNavigate = () => {}, onClose = () => {} } = {}) {
  const nodeLabelEl = el("span", { class: "tp-node" });
  const bodyEl = el("div", { id: "trace-panel-body" });
  hostEl.replaceChildren(
    el("div", { id: "trace-panel-head" },
      el("div", { class: "tp-head-top" },
        el("span", { class: "title", text: "emittance trace" }),
        el("button", {
          class: "tp-close",
          text: "close ✕",
          title: "close the trace (Esc / click empty space)",
          onclick: () => onClose(),
        }),
      ),
      nodeLabelEl,
    ),
    bodyEl,
  );

  let model = null;
  let focusId = null;
  const expandedNodes = new Set(); // lineage node ids whose call list is open
  const expandedCalls = new Set(); // call indices whose detail is open
  let lastSig = null;

  // The calls that GENERATED a node: its provenance — emitted_by (the decompose
  // that named it) + placed_by (the bbox step that placed it), both run on the
  // emitting region. This is the same set for the focused node and every
  // ancestor, so each reads the same way. The root has no provenance, so it
  // falls back to the calls that ran on it (overall_bbox / root plan+decompose)
  // — never empty. Ordered by execution; each entry keeps its relation to badge.
  function nodeCalls(id) {
    const prov = (model.provenance?.get(id) ?? []).filter((p) => p.call?.index != null);
    if (prov.length) {
      return prov
        .slice()
        .sort((a, b) => (a.call.index ?? 0) - (b.call.index ?? 0))
        .map((p) => ({ call: p.call, relation: p.relation }));
    }
    return (model.nodes.get(id)?.calls ?? [])
      .slice()
      .sort((a, b) => (a.index ?? 0) - (b.index ?? 0))
      .map((call) => ({ call, relation: null }));
  }

  // A text section that starts collapsed (just its header); the toggle reveals
  // the full bytes. Scene-context blocks fold behind their own expanders.
  function collapsedSection(label, text, { variables = null } = {}) {
    const body = variables ? foldedPre(text, variables) : el("pre", { text });
    body.style.display = "none";
    const toggle = el("button", {
      class: "tp-toggle",
      text: "expand",
      onclick: (ev) => {
        ev.stopPropagation();
        const hidden = body.style.display === "none";
        body.style.display = hidden ? "" : "none";
        toggle.textContent = hidden ? "collapse" : "expand";
      },
    });
    return el("div", { class: "obsm-sec" },
      el("div", { class: "tp-sec-head" },
        el("span", { class: "tp-sec-lab", text: `${label} · ${shortBytes(text)}` }),
        toggle,
      ),
      body,
    );
  }

  // The output section: shows only the slice about this node by default, with a
  // toggle to the full output. When nothing could be sliced out (the node is the
  // call's region, or a node-specific step) the full output already shows.
  function outputSection(call, nodeId) {
    const { value, truncated } = extractRelevantOutput(call.output, nodeId);
    const relText = fmtJson(value);
    const fullText = fmtJson(call.output);
    const body = el("pre", { text: relText });
    let full = false;
    const toggle = truncated
      ? el("button", {
          class: "tp-toggle",
          text: "show full",
          onclick: (ev) => {
            ev.stopPropagation();
            full = !full;
            body.textContent = full ? fullText : relText;
            toggle.textContent = full ? "show this node" : "show full";
          },
        })
      : null;
    return el("div", { class: "obsm-sec" },
      el("div", { class: "tp-sec-head" },
        el("span", {
          class: "tp-sec-lab",
          text: `output${truncated ? ` · ${nodeId}` : ""} · ${shortBytes(fullText)}`,
        }),
        toggle,
      ),
      body,
    );
  }

  function callDetail(call, nodeId) {
    const detail = el("div", { class: "obsm-detail tp-call-detail" },
      el("div", {
        class: "muted",
        style: "margin-bottom:6px",
        text: [call.model ?? "", `${call.tokens_in ?? "?"} in / ${call.tokens_out ?? "?"} out tok`]
          .filter(Boolean).join(" · "),
      }),
      outputSection(call, nodeId),
      collapsedSection("input (user)", call.user ?? "", { variables: call.variables }),
      collapsedSection("system", call.system ?? "", { variables: call.variables }),
      call.reasoning ? collapsedSection("reasoning", call.reasoning) : null,
    );
    detail.addEventListener("click", (ev) => ev.stopPropagation());
    return detail;
  }

  // One call row, expandable IN PLACE (no full repaint, so other open calls /
  // toggled sections stay put). `relation` badges how it relates to the node.
  function callRow({ call, relation }, nodeId) {
    const wrap = el("div", { class: "tp-call-wrap" });
    const caret = el("span", { class: "caret" });
    let detail = null;
    const sync = () => {
      const open = expandedCalls.has(call.index);
      caret.textContent = open ? "▾" : "▸";
      row.classList.toggle("open", open);
    };
    const row = el("div", {
      class: `obsm-call tp-call${relation === "emitted_by" ? " emits" : ""}`,
      onclick: () => {
        if (expandedCalls.has(call.index)) {
          expandedCalls.delete(call.index);
          detail?.remove();
          detail = null;
        } else {
          expandedCalls.add(call.index);
          detail = callDetail(call, nodeId);
          wrap.appendChild(detail);
        }
        sync();
      },
    },
      caret,
      el("span", { class: "step-badge", text: call.template ?? call.step ?? "?" }),
      relation === "emitted_by" ? el("span", { class: "emit-badge", text: "emitted here" }) : null,
      relation === "placed_by" ? el("span", { class: "tp-place-badge", text: "placed here" }) : null,
      el("span", { class: "muted", text: `#${call.index ?? "?"} · ${call.tokens_out ?? "?"} tok` }),
    );
    wrap.appendChild(row);
    if (expandedCalls.has(call.index)) {
      detail = callDetail(call, nodeId);
      wrap.appendChild(detail);
    }
    sync();
    return wrap;
  }

  function nodeBlock(id, focused) {
    const n = model.nodes.get(id);
    const open = expandedNodes.has(id);
    const calls = nodeCalls(id);
    const emittedBy = (model.provenance?.get(id) ?? []).find((p) => p.relation === "emitted_by");
    const via = emittedBy ? (emittedBy.call.template ?? emittedBy.call.step ?? "?") : null;
    const caret = el("span", { class: "caret", text: open ? "▾" : "▸" });
    const row = el("div", {
      class: "obsm-row clickable tp-node-row",
      title: n?.prompt ?? "",
      onclick: () => {
        if (expandedNodes.has(id)) expandedNodes.delete(id);
        else expandedNodes.add(id);
        paint();
      },
    },
      caret,
      el("span", { class: `dot ${statusDot(n)}` }),
      el("span", { class: "obsm-id", text: id }),
      n ? el("span", { class: "tp-kind", text: n.kind }) : null,
      via ? el("span", { class: "obsm-via", text: `via ${via}`, title: `generated by ${via} on ${emittedBy.call.node ?? "?"}` }) : null,
      el("span", { class: "tp-count", text: `${calls.length} call${calls.length === 1 ? "" : "s"}` }),
    );
    const block = el("div", { class: `obsm-trace-node tp-node${focused ? " focus" : ""}` }, row);
    if (open) {
      block.appendChild(calls.length
        ? el("div", { class: "obsm-calls" }, calls.map((c) => callRow(c, id)))
        : el("div", { class: "muted", style: "margin:2px 0 6px 18px", text: "no LLM calls recorded for this node" }));
    }
    return block;
  }

  function paint() {
    const n = model?.nodes.get(focusId);
    nodeLabelEl.textContent = n ? `${focusId} · ${n.kind}` : (focusId ?? "");
    nodeLabelEl.title = n?.prompt ?? "";
    // Hold the scroll across a rebuild (a node-collapse/expand or a streamed
    // update) so the view doesn't jump to the top; show() zeroes it first for a
    // fresh node.
    const prevScroll = bodyEl.scrollTop;
    bodyEl.textContent = "";
    const chain = emittanceLineage(model, focusId);
    const crumbs = el("div", { class: "obsm-crumbs" });
    chain.forEach((id, i) => {
      if (i) crumbs.appendChild(el("span", { class: "obsm-crumb-sep", text: "›" }));
      crumbs.appendChild(el("span", {
        class: `obsm-crumb${id === focusId ? " cur" : ""}`,
        text: id,
        onclick: id === focusId ? null : () => onNavigate(id),
      }));
    });
    bodyEl.appendChild(el("div", { class: "obsm-trace-head" }, crumbs));
    for (const id of chain) bodyEl.appendChild(nodeBlock(id, id === focusId));
    bodyEl.scrollTop = prevScroll;
    lastSig = signature(chain);
  }

  // A fingerprint of everything the trace currently DRAWS, so a streamed
  // re-render no-ops when nothing visible changed.
  function signature(chain) {
    if (!model || focusId === null || !model.nodes.has(focusId)) return null;
    const nodeSig = (id) => {
      const n = model.nodes.get(id);
      const prov = (model.provenance?.get(id) ?? []).map((p) => `${p.relation}#${p.call.index}`).join(",");
      return `${id}:${n?.calls.length ?? 0}:${prov}:${expandedNodes.has(id) ? 1 : 0}`;
    };
    const exp = [...expandedCalls].sort((a, b) => a - b).join(",");
    return `${focusId}|exp:${exp}|${chain.map(nodeSig).join(";")}`;
  }

  // Open (or re-target) the panel on a node. A different node starts fresh: only
  // it expanded, its ancestors collapsed, no calls open, scrolled to the top.
  function show(m, id) {
    model = m;
    if (!m || !m.nodes.has(id)) { hide(); return; }
    if (id !== focusId) {
      focusId = id;
      expandedNodes.clear();
      expandedNodes.add(id);
      expandedCalls.clear();
      bodyEl.scrollTop = 0;
    }
    hostEl.classList.add("open");
    lastSig = null;
    paint();
  }

  // Fold newly-streamed calls into the open trace, but never while the user is
  // engaged (a call expanded, or scrolled in) — that would collapse what they're
  // reading. User-driven renders (show / navigate / node toggle) always paint.
  function refresh(m, { streamed = false } = {}) {
    model = m;
    if (focusId === null || !hostEl.classList.contains("open")) return;
    if (!m || !m.nodes.has(focusId)) { hide(); return; }
    const engaged = expandedCalls.size > 0 || bodyEl.scrollTop > 0;
    if (streamed && engaged) return;
    if (signature(emittanceLineage(m, focusId)) === lastSig) return;
    paint();
  }

  function hide() {
    hostEl.classList.remove("open");
  }

  // Full teardown on cell open/close — selection is per-cell, so the panel must
  // not carry a stale focus or expansions into the next scene.
  function reset() {
    focusId = null;
    expandedNodes.clear();
    expandedCalls.clear();
    model = null;
    lastSig = null;
    bodyEl.textContent = "";
    hide();
  }

  return {
    show,
    refresh,
    hide,
    reset,
    isOpen: () => hostEl.classList.contains("open"),
    focusId: () => focusId,
  };
}
