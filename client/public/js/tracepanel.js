// The emittance-trace inspector: a left panel in the single-cell overlay that
// appears when an object/zone is picked in 3D. Two stacked parts:
//
//   1. an INFO block for the focused node — a live mini 3D preview (its mesh +
//      bbox, with the real baked orientation) plus its debug fields: prompt, a
//      zone's plan, orientation (semantic text + resolved degrees), dimensions,
//      world origin, proxy shape, structural parent, placement prose, and relationships.
//   2. the EMITTANCE trace below it — the chain of REGIONS that generated the
//      node (root → … → node, climbing each node's emitting region, NOT its
//      structural parent), each with the LLM calls that brought it into being,
//      each call's output truncated to just that node's slice.
//
// It reads the folded obs model + provenance (events.js). The info block is
// rebuilt only when the focus changes (or its data streams in), so the mini
// viewer's WebGL context survives the body's frequent rebuilds.

import { el, foldedPre, fmtJson, shortBytes } from "./ui.js";
import { emittanceLineage, extractRelevantOutput } from "./events.js";
import { createViewer } from "./scene3d.js";
import { api } from "./api.js";

const AXES_KEY = "starshot.traceAxes";   // mini-canvas axes toggle preference
const TRACE_W_KEY = "starshot.traceWidth"; // resizable sidebar width

// A node's OWN characterization steps — ones that ran ON the node itself rather
// than emitting/placing it from its parent: a zone's plan (zone_plan, root or
// nested) and the root's overall bbox. Shown in the trace alongside the
// provenance (emitted_by / placed_by). A zone's decompose / bbox-batch steps are
// deliberately NOT here — those generate its CHILDREN, so they live in the
// children's traces, not the zone's.
const OWN_STEPS = new Set(["zone_plan", "overall_bbox"]);

function statusDot(n) {
  if (!n) return "idle";
  if (n.error) return "error";
  if (n.phase === "done") return "done";
  return n.calls.length ? "running" : "idle";
}

const fmtVec = (a) => `(${a.map((x) => (+x).toFixed(2)).join(", ")})`;
const fmtDims = (a) => a.map((x) => (+x).toFixed(2)).join(" × ");

export function createTracePanel(hostEl, { onNavigate = () => {}, onClose = () => {}, onInquire = null, actions = null, meshUrlFor = (_id, node) => node.meshUrl ?? null } = {}) {
  const nodeLabelEl = el("span", { class: "tp-node" });

  // Mini preview + its overlaid canonical-axes toggle. The axes are world-
  // aligned (one global front view), so the gizmo is drawn AT the focused node
  // to show how its baked orientation sits against +X/+Y/+Z. Toggle persists.
  const previewHost = el("div", { class: "tp-preview" });
  let axesOn = false;
  try { axesOn = localStorage.getItem(AXES_KEY) === "1"; } catch { /* private mode */ }
  let lastGeom = null; // {center, size} of the focused node, for the axes gizmo
  const axesBtn = el("button", {
    class: `tp-axes-btn${axesOn ? " on" : ""}`,
    text: "axes",
    title: "show the canonical X/Y/Z axes at this node (X red · Y green · Z blue)",
    onclick: () => {
      axesOn = !axesOn;
      try { localStorage.setItem(AXES_KEY, axesOn ? "1" : "0"); } catch { /* private mode */ }
      applyAxes();
    },
  });
  const axesLegend = el("div", { class: "tp-axes-legend", style: axesOn ? "" : "display:none" },
    el("span", { class: "ax ax-x", text: "X" }),
    el("span", { class: "ax ax-y", text: "Y" }),
    el("span", { class: "ax ax-z", text: "Z" }),
  );
  previewHost.append(axesBtn, axesLegend);

  // Reference image (the Nano-Banana / library photo the mesh was generated
  // from), shown beneath the 3D preview. Hidden when absent or it fails to load.
  const previewImg = el("img", { class: "tp-preview-img", alt: "reference image" });
  const imgWrap = el("div", { class: "tp-img-wrap" },
    el("div", { class: "tp-field-lab", text: "reference image" }),
    previewImg,
  );
  imgWrap.style.display = "none";
  previewImg.onerror = () => { imgWrap.style.display = "none"; };

  const fieldsEl = el("div", { class: "tp-fields" });
  const infoEl = el("div", { class: "tp-info" }, previewHost, imgWrap, fieldsEl);
  const bodyEl = el("div", { class: "tp-body" });
  const scrollEl = el("div", { id: "trace-panel-scroll" }, infoEl, bodyEl);
  const resizer = el("div", { id: "trace-panel-resizer", title: "drag to resize the panel" });
  hostEl.replaceChildren(
    el("div", { id: "trace-panel-inner" },
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
      scrollEl,
    ),
    resizer,
  );
  initResizer(resizer);

  // Sync the axes gizmo to the toggle + the focused node's geometry. Safe before
  // the mini viewer exists (created lazily) — renderInfo re-applies once it does.
  function applyAxes() {
    axesBtn.classList.toggle("on", axesOn);
    axesLegend.style.display = axesOn ? "" : "none";
    miniViewer?.setAxes(axesOn && !!lastGeom, lastGeom ?? {});
  }

  // Drag the right edge to resize the sidebar. Width lives in the `--trace-w`
  // CSS var so the panel AND the canvas-overlay shift (the :has() rule) track
  // together; clamped to the scene canvas and persisted across reloads.
  function initResizer(handle) {
    try {
      const saved = Number(localStorage.getItem(TRACE_W_KEY));
      if (saved >= 300) {
        // Cap a width saved on a wider screen so it can't swallow a narrow one.
        const w = Math.min(saved, Math.max(300, window.innerWidth - 200));
        document.documentElement.style.setProperty("--trace-w", `${w}px`);
      }
    } catch { /* private mode */ }
    let dragging = false;
    handle.addEventListener("pointerdown", (ev) => {
      dragging = true;
      handle.setPointerCapture(ev.pointerId);
      document.body.classList.add("trace-resizing");
      ev.preventDefault();
    });
    handle.addEventListener("pointermove", (ev) => {
      if (!dragging) return;
      const host = hostEl.parentElement; // #canvas-host
      const left = hostEl.getBoundingClientRect().left;
      const max = Math.max(360, (host?.clientWidth ?? 1200) - 120);
      const w = Math.max(300, Math.min(ev.clientX - left, max));
      document.documentElement.style.setProperty("--trace-w", `${Math.round(w)}px`);
    });
    const end = (ev) => {
      if (!dragging) return;
      dragging = false;
      document.body.classList.remove("trace-resizing");
      try { handle.releasePointerCapture(ev.pointerId); } catch { /* already released */ }
      try {
        const w = parseInt(getComputedStyle(document.documentElement).getPropertyValue("--trace-w"), 10);
        if (w) localStorage.setItem(TRACE_W_KEY, String(w));
      } catch { /* private mode */ }
    };
    handle.addEventListener("pointerup", end);
    handle.addEventListener("pointercancel", end);
  }

  let model = null;
  let focusId = null;
  const expandedNodes = new Set(); // lineage node ids whose call list is open
  const expandedCalls = new Set(); // call indices whose detail is open
  let lastSig = null;
  let lastInfoSig = null;
  // One reusable mini viewer for the whole panel lifetime (1 WebGL context);
  // created lazily, never disposed — just cleared + reloaded per focus, and
  // paused (setActive false) whenever the panel is hidden.
  let miniViewer = null;
  function ensureMini() {
    if (!miniViewer) miniViewer = createViewer(previewHost, { keyboard: false });
    return miniViewer;
  }

  // ── focused-node info (rebuilt only on focus change / live data fill-in) ──

  // The object/subregion spec the emitting decompose call named this node with —
  // the source of its semantic orientation text, placement prose, structural
  // parent, and relationships (none of which live on the obs node itself).
  function focusedSpec(id) {
    const emitted = (model.provenance?.get(id) ?? []).find((p) => p.relation === "emitted_by");
    if (!emitted) return null;
    const { value, truncated } = extractRelevantOutput(emitted.call.output, id);
    return truncated && value && typeof value === "object" ? value : null;
  }

  function fieldGroup(label, valueNode) {
    return el("div", { class: "tp-field-group" },
      el("div", { class: "tp-field-lab", text: label }),
      typeof valueNode === "string" ? el("div", { class: "tp-field-val", text: valueNode }) : valueNode,
    );
  }

  function prop(label, value) {
    return el("div", { class: "tp-prop" },
      el("span", { class: "tp-prop-lab", text: label }),
      typeof value === "string" ? el("span", { class: "tp-prop-val", text: value }) : value,
    );
  }

  // Per-object generated-asset controls — shown in the info block only while the
  // overlay is viewing the from-scratch generated build (actions.available()).
  // regenerate rebuilds the mesh fresh on the chosen backend (Trellis / Hunyuan
  // on Modal / Hunyuan 3.1 on Tencent); symmetrize / unsymmetrize mirror or
  // reveal the raw mesh with no backend call. All propagate across the prefab group.
  function buildActions(id) {
    const backendSel = el("select", { class: "tp-act-sel", title: "mesh backend for regenerate" },
      ...["trellis", "hunyuan", "hunyuan-tencent"].map((b) => el("option", { value: b, text: b })));
    const reuse = el("input", { type: "checkbox" });
    const sym = actions.symmetryOf?.(id);
    const mirrored = !!sym && (sym.plane === "xy" || sym.plane === "xz");

    // Symmetry is one toggle: a mirrored asset shows ONLY un-symmetrize, an
    // un-mirrored one shows ONLY symmetrize (+ the plane to mirror across).
    let symRow;
    if (mirrored) {
      symRow = el("div", { class: "tp-act-row" },
        el("button", {
          class: "tp-act-btn", text: "unsymmetrize",
          title: `reveal the full, un-mirrored mesh (currently mirrored across ${sym.plane})`,
          onclick: () => actions.onUnsymmetrize(id),
        }),
      );
    } else {
      const planeSel = el("select", { class: "tp-act-sel", title: "mirror plane" },
        el("option", { value: "xy", text: "xy · front/back" }),
        el("option", { value: "xz", text: "xz · top/bottom" }),
      );
      // Which half to keep, mirrored onto the other. Labels follow the plane
      // (xy: front/back along Z, xz: top/bottom along Y); value = keep_positive.
      const keepSel = el("select", { class: "tp-act-sel", title: "which half to keep, then mirror onto the other" });
      const KEEP = {
        xy: [["true", "keep front"], ["false", "keep back"]],
        xz: [["true", "keep top"], ["false", "keep bottom"]],
      };
      const syncKeep = () => keepSel.replaceChildren(
        ...KEEP[planeSel.value].map(([v, t]) => el("option", { value: v, text: t })),
      );
      syncKeep();
      planeSel.addEventListener("change", syncKeep);
      symRow = el("div", { class: "tp-act-row" },
        el("button", {
          class: "tp-act-btn", text: "symmetrize",
          title: "mirror this asset across the chosen plane, keeping the chosen half",
          onclick: () => actions.onSymmetrize(id, {
            plane: planeSel.value,
            keepPositive: keepSel.value === "true",
          }),
        }),
        planeSel,
        keepSel,
      );
    }
    const symText = mirrored
      ? `mirrored · ${sym.plane}`
      : sym && sym.was ? `un-symmetrized (was ${sym.was})` : "";
    return el("div", { class: "tp-actions" },
      el("div", { class: "tp-field-lab", text: "generated asset" }),
      el("div", { class: "tp-act-row" },
        el("button", {
          class: "tp-act-btn", text: "regenerate",
          title: "rebuild this asset's mesh fresh on the chosen backend (propagates to its prefab group)",
          onclick: () => actions.onRegenerate(id, { backend: backendSel.value, reuseImage: reuse.checked }),
        }),
        backendSel,
        el("label", { class: "tp-act-check", title: "reuse the existing reference image (skip Nano-Banana)" }, reuse, "reuse image"),
      ),
      symRow,
      symText ? el("div", { class: "tp-act-sym", text: symText }) : null,
    );
  }

  function infoSig(id) {
    const n = model.nodes.get(id);
    if (!n) return null;
    const emitted = (model.provenance?.get(id) ?? []).find((p) => p.relation === "emitted_by");
    return [
      id, n.prompt, n.plan, n.kind, n.phase, n.meshUrl, n.proxyShape, n.orientation,
      Array.isArray(n.origin) ? n.origin.join(",") : "",
      Array.isArray(n.dimensions) ? n.dimensions.join(",") : "",
      emitted?.call?.index ?? "",
    ].join("|");
  }

  function renderInfo(id) {
    fieldsEl.textContent = "";
    const n = model.nodes.get(id);
    if (!n) { previewHost.style.display = "none"; miniViewer?.setActive(false); lastInfoSig = null; return; }
    const spec = focusedSpec(id);
    const isObject = n.kind === "object" || n.kind === "frame";
    const hasGeom = Array.isArray(n.origin) && Array.isArray(n.dimensions);

    // Mini 3D preview: the node's bbox (always, when known) + its mesh (when one
    // exists), with the orientation baked into the served GLB. Zones show just
    // their bbox volume; objects show the oriented mesh inside its box.
    // The axes gizmo sits at the node's bbox center, sized to its largest span.
    lastGeom = hasGeom
      ? {
          center: [
            n.origin[0] + n.dimensions[0] / 2,
            n.origin[1] + n.dimensions[1] / 2,
            n.origin[2] + n.dimensions[2] / 2,
          ],
          size: Math.max(Math.abs(n.dimensions[0]), Math.abs(n.dimensions[1]), Math.abs(n.dimensions[2])) * 0.75 || 1,
        }
      : null;
    const meshUrl = meshUrlFor(id, n);
    // The reference image sits next to the mesh on disk as `<id>.png` (library
    // and generated alike), so it follows whichever mesh the mode resolved to.
    const imageUrl = meshUrl ? meshUrl.replace(/\.glb(\?|$)/, ".png$1") : null;
    if (imageUrl) {
      previewImg.src = api.absUrl(imageUrl);
      imgWrap.style.display = "";
    } else {
      imgWrap.style.display = "none";
    }
    if (meshUrl || hasGeom) {
      previewHost.style.display = "";
      const mv = ensureMini();
      mv.clear();
      if (hasGeom) {
        mv.loadBbox({ id, origin: n.origin, dimensions: n.dimensions, node_kind: n.kind, proxy_shape: n.proxyShape ?? null });
      }
      if (meshUrl) mv.loadModel({ id, url: meshUrl }, api.absUrl(meshUrl));
      mv.setActive(true);
      applyAxes();
    } else {
      previewHost.style.display = "none";
      miniViewer?.setActive(false);
    }

    if (n.prompt) fieldsEl.appendChild(fieldGroup("prompt", n.prompt));
    // A zone's plan (from zone_plan) characterizes the zone itself, so surface it
    // right under the prompt — a selected zone reads as prompt → plan, not just
    // its prompt. Only zones carry a plan.
    if (n.kind === "zone" && n.plan) fieldsEl.appendChild(fieldGroup("plan", n.plan));

    const props = el("div", { class: "tp-props" });
    const addProp = (label, value) => { if (value !== null && value !== undefined && value !== "") props.appendChild(prop(label, value)); };
    addProp("kind", n.kind + (n.phase ? ` · ${n.phase}` : ""));
    if (isObject) {
      const text = typeof spec?.orientation === "string" ? spec.orientation : null;
      const deg = typeof n.orientation === "number" ? `${n.orientation}°` : null;
      const parts = [text ? `“${text}”` : null, deg].filter(Boolean);
      if (parts.length) addProp("orientation", parts.join(" · "));
    }
    if (Array.isArray(n.dimensions)) addProp("dimensions", fmtDims(n.dimensions));
    if (Array.isArray(n.origin)) addProp("origin (world)", fmtVec(n.origin));
    if (n.proxyShape) addProp("proxy", n.proxyShape);
    if (spec?.parent) addProp("structural parent", `${spec.parent}${spec.parent_relationship_kind ? ` · ${spec.parent_relationship_kind}` : ""}`);
    if (props.childElementCount) fieldsEl.appendChild(props);

    if (spec?.placement) fieldsEl.appendChild(fieldGroup("placement", spec.placement));

    const rels = Array.isArray(spec?.relationships) ? spec.relationships : [];
    if (rels.length) {
      const list = el("div", { class: "tp-rels" }, rels.map((r) => {
        const linked = typeof r.target === "string" && model.nodes.has(r.target);
        return el("div", { class: "tp-rel" },
          el("span", { class: "tp-rel-kind", text: r.kind ?? "?" }),
          el("span", { class: "tp-rel-arrow", text: "→" }),
          el("span", {
            class: `tp-rel-target${linked ? " link" : ""}`,
            text: r.target ?? "?",
            onclick: linked ? () => onNavigate(r.target) : null,
          }),
        );
      }));
      fieldsEl.appendChild(fieldGroup(`relationships (${rels.length})`, list));
    }

    if (meshUrl) {
      fieldsEl.appendChild(prop("mesh", el("a", { class: "tp-link", href: api.absUrl(meshUrl), target: "_blank", text: "open .glb ↗" })));
    }

    if (actions && actions.available() && isObject) {
      fieldsEl.appendChild(buildActions(id));
    }

    lastInfoSig = infoSig(id);
  }

  // ── emittance trace body (rebuilt on node/call toggle + streamed updates) ──

  // The calls that GENERATED + characterized a node: its provenance — emitted_by
  // (the decompose that named it) + placed_by (the bbox step that placed it) —
  // PLUS its own steps (a zone's plan; the root's overall bbox), which run on the
  // node itself so they're attributed, not provenance. Same shape for the focused
  // node and every ancestor zone. Falls back to anything attributed if a node has
  // none of the above, so a row is never bare. Ordered by execution.
  function nodeCalls(id) {
    const byIndex = new Map();
    for (const p of model.provenance?.get(id) ?? []) {
      if (p.call?.index != null) byIndex.set(p.call.index, { call: p.call, relation: p.relation });
    }
    for (const c of model.nodes.get(id)?.calls ?? []) {
      if (c.index != null && OWN_STEPS.has(c.step) && !byIndex.has(c.index)) {
        byIndex.set(c.index, { call: c, relation: null });
      }
    }
    let entries = [...byIndex.values()];
    if (!entries.length) {
      entries = (model.nodes.get(id)?.calls ?? []).map((c) => ({ call: c, relation: null }));
    }
    return entries.sort((a, b) => (a.call.index ?? 0) - (b.call.index ?? 0));
  }

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
        el("span", { class: "tp-sec-lab", text: `output${truncated ? ` · ${nodeId}` : ""} · ${shortBytes(fullText)}` }),
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
      onInquire ? el("button", {
        class: "call-ask",
        text: "why?",
        style: "margin-left:auto",
        title: "continue this step's conversation with the model that made it — ask it anything",
        onclick: (ev) => { ev.stopPropagation(); onInquire(call); },
      }) : null,
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
    const prevScroll = scrollEl.scrollTop;
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
    bodyEl.appendChild(el("div", { class: "obsm-trace-head" },
      el("div", { class: "tp-trace-title", text: "generated by" }),
      crumbs,
    ));
    for (const id of chain) bodyEl.appendChild(nodeBlock(id, id === focusId));
    scrollEl.scrollTop = prevScroll;
    lastSig = signature(chain);
  }

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

  // ── lifecycle ──

  function show(m, id) {
    model = m;
    if (!m || !m.nodes.has(id)) { hide(); return; }
    const newFocus = id !== focusId;
    if (newFocus) {
      focusId = id;
      expandedNodes.clear();
      expandedNodes.add(id);
      expandedCalls.clear();
    }
    hostEl.classList.add("open");
    if (newFocus) renderInfo(id);
    lastSig = null;
    paint();
    if (newFocus) scrollEl.scrollTop = 0;
  }

  function refresh(m, { streamed = false } = {}) {
    model = m;
    if (focusId === null || !hostEl.classList.contains("open")) return;
    if (!m || !m.nodes.has(focusId)) { hide(); return; }
    const engaged = expandedCalls.size > 0 || scrollEl.scrollTop > 0;
    if (streamed && engaged) return;
    if (infoSig(focusId) !== lastInfoSig) renderInfo(focusId);
    if (signature(emittanceLineage(m, focusId)) !== lastSig) paint();
  }

  function hide() {
    hostEl.classList.remove("open");
    miniViewer?.setActive(false);
  }

  function reset() {
    focusId = null;
    expandedNodes.clear();
    expandedCalls.clear();
    model = null;
    lastSig = null;
    lastInfoSig = null;
    fieldsEl.textContent = "";
    bodyEl.textContent = "";
    previewHost.style.display = "none";
    lastGeom = null;
    miniViewer?.setActive(false);
    miniViewer?.setAxes(false);
    miniViewer?.clear();
    hide();
  }

  // Re-render the focused node's info block (e.g. when the asset mode toggles,
  // so the generated-asset actions appear/disappear without a focus change).
  function rerenderInfo() {
    if (focusId !== null && model && model.nodes.has(focusId)) renderInfo(focusId);
  }

  return {
    show,
    refresh,
    hide,
    reset,
    rerenderInfo,
    isOpen: () => hostEl.classList.contains("open"),
    focusId: () => focusId,
  };
}
