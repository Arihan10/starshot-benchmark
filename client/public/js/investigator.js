// The scene investigator: a docked, toggleable chat that reasons about a WHOLE
// cell's scene. `createInvestigator(hostEl)` builds one independent instance into
// `hostEl` (own threads, DOM, obs-model binding), so the single-cell overlay and
// each run-compare pane get their own from one implementation — this is the
// shared chat component the per-step "why?" now routes into (via `attachStep`).
//
// It's given the faithful base grounding — the pipeline explainer, every prompt
// template, the full final scene context, and a timeline of every executed
// step's OUTPUT + REASONING + step-specific variable values — but NOT each step's
// rendered system/input; the reviewer reconstructs what a step saw from
// (template + variables + scene). You can @-mention specific steps to fold their
// exact bytes in as deep context, Cursor-style; clicking "why?" on a call row is
// exactly that — open the panel + attach that one step.
//
// Threads live per instance keyed run|slot|model|branch, so toggling the panel or
// re-opening a cell restores its conversation + attachments. The composed system
// prompt is forwarded through the `/inquire` reviewer (Claude Opus 4.8, xhigh).

import { el, openModal, fmtJson, shortBytes } from "./ui.js";
import { api } from "./api.js";
import { on } from "./state.js";
import { emittingRegion } from "./events.js";
import { renderMarkdown } from "./markdown.js";

// ── static grounding: the framing, pipeline explainer, and variable glossary ──

const INVESTIGATOR_FRAMING = `You are a senior spatial-reasoning analyst investigating a complete or mid-construction 3D scene that an automated pipeline built from a single text prompt. The pipeline uses language models at every reasoning step: it recursively decomposes the prompt into a tree of regions (zones) and objects, each with an axis-aligned bounding box, an orientation, and spatial relationships, then generates a mesh for every leaf object and composes them into one scene. The pipeline should be built to handle any kind of scene - modern houses, platformer levels, shooter maps, etc. - your job is to answer any questions that the user may have about the scene, by concretely analyzing the evidence and data provided.

You are given, below, everything needed to reason about WHY the scene turned out the way it did:
  - PIPELINE — how the pipeline works, its steps, and the world frame.
  - VARIABLES — what each \`{VARIABLE}\` token in the templates means.
  - PROMPT TEMPLATES — the exact instructions (with variable tokens) each step operates under.
  - CURRENT SCENE — the full, final scene: the root, its global objects, and the subregion tree with every region's plan, bbox, and inline objects.
  - EXECUTION TIMELINE — every LLM step that actually ran, in order: which template on which node, its structured OUTPUT, its private REASONING, and its step-specific variable VALUES.
  - FOCUS — when the developer pins specific steps (or a scene object/zone) as most relevant to the question, those steps are pulled OUT of the timeline and shown in full here: their exact system/input bytes, with heavy scene-wide variables left as tokens. A step is never in both the timeline and FOCUS.

You are NOT given each step's fully-rendered system/user message, except the FOCUS ones. Reconstruct what a non-focus step saw by combining its TEMPLATE with the scene state at that point (replay the timeline) and its logged variable VALUES. When you need a step's exact bytes, the developer attaches it — it then moves from the timeline into FOCUS.

A developer benchmarking the pipeline's spatial reasoning will ask you holistic questions about the scene — why an object sits where it does, whether a placement blocks traversal or overlaps badly, whether a decomposition was sensible, what the intent behind the layout was, where the reasoning went wrong.

Rules:
  - Ground every claim in the provided material. Cite specific ids, coordinates, dimensions, steps (by template + node), and template clauses. A model's own REASONING is the strongest evidence of intent — quote it when it explains a decision.
  - Separate what a step actually reasoned from what you are inferring from its output + the scene. When the evidence is silent, say so plainly: "the reasoning doesn't address this" and "this looks like an unexplained / arbitrary choice" are valid, valuable answers. Never invent motives.
  - Reason spatially and quantitatively: the world frame is right-handed, Y-up, metres — +X is right, +Y is up, +Z is toward the viewer (front), -Z is away (back). A bounding box is an origin (min corner) + dimensions. Geometry is additive, so well-formed scenes keep boxes flush rather than overlapping.
  - Answer in GitHub-flavoured Markdown: lead with the direct answer, then the evidence. Use headings, bullet lists, tables, and \`inline code\` for ids/coords. Keep it tight and skimmable.
  - Do not use dense jargon and goofy structured documents, output cleanly using flowing paragraphs with no massive assumed priors and dense jargon.
  - Base all advice regarding the pipeline's prompts on the principle of generalizability - avoid simply saying "give negative examples" or "explicit rule" unless as a last resort. The prompts should convey the character and role to the LLM, not explicitly say what to do unless absolutely necessary.
  `;

const PIPELINE_STEPS = [
  ["zone_plan_root", "once per run — plan the overall scene from the user prompt and decide whether it is atomic (a leaf) or should be subdivided."],
  ["overall_bbox", "once per run — size the root canvas (the scene's overall bounding box) from the scene plan."],
  ["zone_plan", "per nested region — plan one subregion (its narrative + role) and decide if it is atomic."],
  ["zone_decompose_root / zone_decompose", "split a non-atomic region into its top-level subregions (ids + seed prompts + placement + relationships)."],
  ["child_bbox_batch", "the SUBREGION SOLVER — resolve every child subregion's bbox in one call (pure spatial reasoning from the placement prose)."],
  ["encapsulating_decompose", "decide + enumerate a region's shell / perimeter / boundary objects (walls, ground, fences)."],
  ["anchor_decompose", "for an atomic region — enumerate the defining anchor objects of the leaf."],
  ["negative_space_decompose", "fill the interstitial gaps of a region (and the root) with ambient / connective objects."],
  ["object_bbox_batch", "the OBJECT SOLVER — resolve a batch of objects' bboxes + yaw in one call from their placement prose (locked in permanently)."],
  ["next_object", "loop on an atomic region — propose more objects or declare the region complete."],
  ["image_prompt", "per object — distill the object into a concise noun phrase for image generation (runs on a reduced scene context; omitted from the base context below — attach one to inspect it)."],
];

const VARIABLE_GLOSSARY = [
  ["ROOT_PROMPT / ROOT_PLAN", "the user's scene prompt and the scene plan authored by zone_plan_root."],
  ["ROOT_HEADER", "canonical scene header: root prompt, plan, and overall bbox in prose."],
  ["ROOT_OBJECTS", "objects anchored directly to the root (shared shells / ground / ambient fill)."],
  ["SCENE_CONTEXT", "the embedded subregion tree — every subregion with plan, bbox, relationships, and inline objects; the step's target region is tagged inline with `<-- TARGET:`."],
  ["ZONE_ID / ZONE_PROMPT / ZONE_PLAN / ZONE_PLACEMENT", "the region a step is acting on: its id, seed prompt, plan, and semantic placement within its parent."],
  ["ZONE_DIMENSIONS / ZONE_ORIGIN", "the target region's bbox size and world origin corner."],
  ["ZONE_OBJECTS", "objects already placed directly inside the target region."],
  ["PARENT_ZONE_ID / PARENT_ZONE_PLAN / PARENT_ZONE_ORIGIN", "the target region's enclosing parent region (one level up)."],
  ["TO_PLACE", "the batch of child/object specs whose bboxes a solver step must emit (id, parent, relationship kind, parent dims/origin, proxy shape, orientation, prompt, placement, relationships)."],
  ["RETRY_BLOCK", "empty on the first attempt; after a rejected attempt, the prior emissions + rejection reasons the step must fix."],
  ["ADJACENT_ZONES", "the regions adjacent to the target (nearest in each direction), rendered like SCENE_CONTEXT but trimmed to neighbours."],
  ["OBJECT_PROMPT / OBJECT_DIMENSIONS / PROXY_SHAPE", "for image_prompt: the object's prompt, size, and collision-proxy shape (BOX / SPHERE / CAPSULE / HEMISPHERE)."],
  ["SIBLING_OBJECTS / ROOT_OBJECTS_BRIEF / OTHER_SUBREGIONS_BRIEF / PRIOR_SUBJECTS", "the reduced, aesthetic-coherence context image_prompt runs on."],
];

function pipelineDoc(mode) {
  const phase = mode === "bfs" ? "breadth-first (BFS)" : "depth-first (DFS)";
  return `# PIPELINE

A text-to-3D pipeline turns one prompt (e.g. "a cozy two-room cottage") into a fully parametric 3D scene in two phases.

Phase 1 — DIVIDER (this run traverses the tree ${phase}): recursively decomposes the prompt into a tree. The root is planned and sized, then split into subregions; each subregion is planned, and either subdivided further (non-atomic) or filled with objects (atomic). Every region and object gets an axis-aligned bounding box (origin = min corner, plus dimensions), a semantic placement, an orientation, and optional relationships to peers. Coordinate assignment is done by dedicated "solver" steps (\`child_bbox_batch\`, \`object_bbox_batch\`) that translate the semantic placement prose into concrete boxes + yaw; once emitted, a box is locked.

Phase 2 — GENERATION: for each leaf object, \`image_prompt\` distills a noun phrase, an image is generated, and a mesh (Trellis / Hunyuan) is produced, oriented, and fitted into the object's bounding box. Geometry is purely additive, so the divider is expected to keep boxes flush (adjacent) rather than overlapping unless an intersection is deliberate.

World frame: right-handed, Y-up, metres. +X right, +Y up, +Z toward the viewer (front), -Z away (back). Orientation is a global yaw about +Y: 0° faces +Z (toward viewer), 90° faces +X (right); the solver picks one of -180/-135/-90/-45/0/45/90/135/180.

Pipeline steps (each is a separate LLM call; templates for all of them are below):
${PIPELINE_STEPS.map(([n, d]) => `- \`${n}\` — ${d}`).join("\n")}

# VARIABLES

Templates reference live scene state via \`{VARIABLE}\` tokens. Each step natively populates the subset relevant to it; others render empty. What each means:
${VARIABLE_GLOSSARY.map(([n, d]) => `- \`${n}\` — ${d}`).join("\n")}`;
}

const SUGGESTIONS = [
  "Give me a high-level read on this scene's spatial logic and whether it's coherent.",
  "Are any objects placed so they block traversal, float, or overlap badly? Cite ids and coordinates.",
  "Which decomposition or placement decisions look questionable, and what does the reasoning say?",
  "What was the intent behind the overall layout — does it match the prompt?",
];

// ── module-level (shared across instances): pure helpers ──

// The run's CURRENT prompt snapshot, fetched fresh on each base build — NOT
// cached client-side. The prompt lab's "apply to run" / "revert to base" edit the
// snapshot in place, so a per-run cache would serve stale templates after an
// edit. The payload is small and the server caches it, so re-fetching per base
// build (open / ↻ refresh) is cheap and always reflects the active snapshot.
async function loadTemplates(run) {
  const payload = await api.promptTemplates(run);
  return payload.steps;
}

function convId(c) {
  return c ? `${c.run}|${c.slot}|${c.model}|${c.branch || "src"}` : null;
}

// The zone_plan / zone_plan_root call bucketed on a zone — the "plan of that
// zone" step, or null if it hasn't run.
function zonePlanCallIndex(model, zoneId) {
  const node = model?.nodes.get(zoneId);
  if (!node) return null;
  const c = node.calls.find((call) => {
    const step = call.template ?? call.step;
    return step === "zone_plan" || step === "zone_plan_root";
  });
  return c ? c.index : null;
}

function templatesBlock(templates) {
  // image_prompt (the per-object noun-phrase distiller) is left out of the base
  // context — its template + image wrappers bloat it and are rarely relevant to
  // a holistic spatial question. Attach a specific image_prompt step to see it.
  return (templates ?? []).filter((t) => t.step !== "image_prompt").map((t) => [
    `### ${t.step}${t.native?.length ? `  (native variables: ${t.native.join(", ")})` : ""}`,
    "SYSTEM template:",
    t.system ?? "(none)",
    "",
    "USER template:",
    t.user ?? "(none)",
  ].join("\n")).join("\n\n");
}

function sceneBlock(bundle) {
  return [
    "## Root",
    bundle.root_header || "(the root scene has not been resolved yet)",
    "",
    "## Objects anchored directly to the root (shells / ground / ambient fill)",
    bundle.root_objects || "(none)",
    "",
    "## Subregion tree (every region with its plan, bbox, and inline objects)",
    bundle.scene_context || "(no subregions have been placed yet)",
  ].join("\n");
}

function timelineBlock(steps) {
  if (!steps?.length) return "(no LLM steps have executed in this cell yet)";
  return steps.map((s) => {
    const vars = s.variables && Object.keys(s.variables).length
      ? Object.entries(s.variables).map(([k, v]) =>
          String(v).includes("\n") ? `\`{${k}}\`:\n${v}` : `\`{${k}}\` = ${v}`).join("\n")
      : "(no step-specific variables logged)";
    return [
      `### #${s.index} · ${s.template ?? s.step} · on ${s.node ?? "?"}`,
      "variables:",
      vars,
      "",
      `output: ${fmtJson(s.output)}`,
      "",
      `reasoning: ${s.reasoning ? s.reasoning : "(none exposed)"}`,
    ].join("\n");
  }).join("\n\n");
}

// Scene-wide variables whose rendered values would blow up context if repeated
// in every attached step. In an attached step's exact bytes they're collapsed
// back to their `{TOKEN}`; their values are defined ONCE (the scene in CURRENT
// SCENE, the fixed image wrappers in the glossary), and the reviewer
// reconstructs a mid-run step's scene from the timeline. Step-specific values
// (ZONE_*, TO_PLACE, RETRY_BLOCK, OBJECT_*) stay inline — they're the per-step
// delta and aren't duplicated across steps. Mirrors the server's timeline strip.
const HEAVY_VARS = [
  "SCENE_CONTEXT", "SCENE_CONTEXT_COMPACT", "ROOT_OBJECTS", "ROOT_HEADER",
  "ROOT_OBJECTS_BRIEF", "OTHER_SUBREGIONS_BRIEF", "ADJACENT_ZONES",
  "SIBLING_OBJECTS", "ZONE_OBJECTS",
  "IMAGE_TEMPLATE_FRONT", "IMAGE_TEMPLATE_SIDE", "IMAGE_TEMPLATE_TOP",
];

// Replace each heavy variable's rendered value in `text` with its `{TOKEN}`, so
// the exact prompt is shown without inlining (and duplicating) the big blocks.
// Longest values first, so a value nested inside a bigger one can't half-match.
function collapseHeavyVars(text, variables) {
  if (!text) return "";
  if (!variables || typeof variables !== "object") return text;
  const present = HEAVY_VARS
    .map((name) => ({ name, val: variables[name] }))
    .filter((v) => typeof v.val === "string" && v.val.length >= 40 && text.includes(v.val))
    .sort((a, b) => b.val.length - a.val.length);
  let out = text;
  for (const { name, val } of present) out = out.split(val).join(`\`{${name}}\``);
  return out;
}

// ── the shared chat component ────────────────────────────────────────────────

export function createInvestigator(hostEl, { onClose = () => {} } = {}) {
  let refs = null; // { title, sub, status, body, attach, input, sendBtn, mention }
  let ctx = null; // { run, slot, model, branch }
  let currentId = null;
  let obsModel = null; // the live obs-model wrapper (obsModel.model.calls / .nodes)
  let blurTimer = null;
  let mention = null; // { start, items:[{index,label,call}], active } | null
  // convId -> { turns:[{role,content,reasoning?,error?}], attached:Set<callIndex>,
  //            base:{ bundle, templates }|null, baseFetching, baseError, busy }.
  // `busy` is PER THREAD (per cell/branch), so an in-flight request on one never
  // blocks sending on another — each conversation runs independently.
  const threads = new Map();

  function thread() {
    return currentId ? threads.get(currentId) : null;
  }

  // ── base context (scene + timeline + templates) ─────────────────────────────

  async function ensureBase() {
    const id = currentId;
    const c = ctx;
    const t = id ? threads.get(id) : null;
    if (!t || !c || t.base || t.baseFetching) return;
    t.baseFetching = true;
    t.baseError = null;
    if (id === currentId) updateStatus();
    try {
      const [bundle, templates] = await Promise.all([
        c.branch ? api.branchInvestigator(c.branch) : api.investigator(c.run, c.slot, c.model),
        loadTemplates(c.run),
      ]);
      t.base = { bundle, templates };
    } catch (e) {
      t.baseError = e.message;
    } finally {
      t.baseFetching = false;
      if (id === currentId) { updateStatus(); renderTranscript(); }
    }
  }

  function refreshBase() {
    const t = thread();
    if (!t) return;
    t.base = null;
    t.baseError = null;
    ensureBase();
  }

  // ── system-prompt composition ─────────────────────────────────────────────

  function callByIndex(idx) {
    return obsModel?.model.calls.find((c) => c.index === idx) ?? null;
  }

  function attachmentBlock(idx) {
    const c = callByIndex(idx);
    if (!c) return `### Focus step #${idx}\n(this step is no longer in the loaded event log)`;
    // Show the EXACT bytes, but with the heavy scene-wide variables collapsed
    // back to their `{TOKEN}` — their values live once in CURRENT SCENE, so
    // attaching many steps never re-dumps the scene context. Step-specific values
    // (zone fields, TO_PLACE, RETRY_BLOCK, …) stay inline as rendered.
    return [
      `### Focus step #${c.index} — ${c.template ?? c.step} on ${c.node ?? "?"} (subject model: ${c.model ?? "?"})`,
      "",
      "SYSTEM (exact bytes — heavy scene-wide variables shown as their `{TOKEN}`):",
      collapseHeavyVars(c.system, c.variables) || "(empty)",
      "",
      "INPUT / user message (exact bytes — heavy variables shown as their `{TOKEN}`):",
      collapseHeavyVars(c.user, c.variables) || "(empty)",
      "",
      "OUTPUT:",
      fmtJson(c.output),
      "",
      "REASONING:",
      c.reasoning || "(the subject model exposed no separate reasoning)",
    ].join("\n");
  }

  function composeSystem(t) {
    const mode = document.documentElement.dataset.pipeline || "dfs";
    const attached = t.attached;
    // Focus (attached) steps are PULLED OUT of the timeline and shown in full
    // below — never in both places, so attaching a step never duplicates it.
    const timelineSteps = attached.size
      ? t.base.bundle.steps.filter((s) => !attached.has(s.index))
      : t.base.bundle.steps;
    const parts = [
      INVESTIGATOR_FRAMING,
      pipelineDoc(mode),
      `# PROMPT TEMPLATES (every pipeline step except image_prompt — raw, with \`{VARIABLE}\` tokens; attach an image_prompt step to see its template)\n\n${templatesBlock(t.base.templates)}`,
      `# CURRENT SCENE\n\n${sceneBlock(t.base.bundle)}`,
      `# EXECUTION TIMELINE (every step that ran, in order, EXCEPT the per-object image_prompt calls and the FOCUS steps below — its output, reasoning, and step-specific variable values)\n\n${timelineBlock(timelineSteps)}`,
    ];
    if (attached.size) {
      // Execution order, so a mentioned object's steps read as its history:
      // zone_plan → emitted → placed.
      const ordered = [...attached].sort((a, b) => a - b);
      parts.push(
        "# FOCUS — the step requests directly relevant to the question (pulled OUT of the timeline above and shown in full: exact system + input bytes, with the heavy scene-wide variables (`{SCENE_CONTEXT}`, `{ROOT_OBJECTS}`, `{ROOT_HEADER}`, …) collapsed to their tokens — defined once in CURRENT SCENE; reconstruct a mid-run step's scene from the timeline)\n\n" +
        ordered.map(attachmentBlock).join("\n\n---\n\n"),
      );
    }
    return parts.join("\n\n");
  }

  // ── rendering ──────────────────────────────────────────────────────────────

  function renderShell() {
    if (!refs) return;
    if (!ctx) { refs.title.textContent = "investigator"; refs.sub.textContent = ""; return; }
    refs.title.textContent = "investigator";
    const cell = `${ctx.slot} · ${ctx.model}${ctx.branch ? " · sim" : ""}`;
    refs.sub.textContent = `${cell}  ·  reviewer: claude-opus-4.8 · xhigh`;
  }

  function updateStatus() {
    if (!refs) return;
    const t = thread();
    if (!t) { refs.status.textContent = ""; return; }
    const nodes = obsModel?.model.nodes.size ?? 0;
    const attached = t.attached.size;
    let base;
    if (t.baseFetching) base = "preparing scene context…";
    else if (t.baseError) base = `context failed: ${t.baseError} — click ↻`;
    else if (t.base) base = `${t.base.bundle.steps.length} steps · ${nodes} nodes`;
    else base = "context not loaded";
    refs.status.textContent = base + (attached ? ` · ${attached} in focus` : "");
  }

  // The send controls reflect the CURRENTLY-VIEWED thread's own in-flight state,
  // so a request on one cell/branch never disables another's input/button.
  function syncSendControls() {
    if (!refs) return;
    const b = !!thread()?.busy;
    refs.sendBtn.disabled = b;
    refs.sendBtn.textContent = b ? "thinking…" : "ask";
    refs.input.disabled = b;
  }

  function renderAttach() {
    if (!refs) return;
    const t = thread();
    refs.attach.replaceChildren();
    if (!t || !t.attached.size) { refs.attach.style.display = "none"; return; }
    refs.attach.style.display = "";
    refs.attach.appendChild(el("span", { class: "ivg-attach-lab", text: "focus:" }));
    for (const idx of t.attached) {
      const c = callByIndex(idx);
      const label = c ? `${c.template ?? c.step} · ${c.node ?? "?"} · #${idx}` : `#${idx} (gone)`;
      refs.attach.appendChild(el("span", { class: "ivg-pill", title: c?.node ? `${c.template ?? c.step} on ${c.node}` : label },
        el("span", { class: "ivg-pill-lab", text: label }),
        el("button", {
          class: "ivg-pill-x", text: "✕", title: "remove this step from the context",
          onclick: () => { t.attached.delete(idx); renderAttach(); updateStatus(); },
        }),
      ));
    }
  }

  function renderTranscript() {
    if (!refs) return;
    const t = thread();
    refs.body.replaceChildren();
    if (!t) {
      refs.body.appendChild(el("div", { class: "ivg-hint" }, el("div", { text: "Open a cell to investigate its scene." })));
      return;
    }
    const showPending = !!t.busy;
    if (!t.turns.length && !showPending) {
      refs.body.appendChild(el("div", { class: "ivg-hint" },
        el("div", { text: "Ask holistic questions about this scene — the reviewer has the full scene, every prompt template, and each step's output + reasoning. Type " }),
        el("div", { class: "ivg-hint-emph", text: "@ an object to attach its history (its zone's plan → emitted → placed), a zone to attach its plan + emitted + placed, or @ a step directly (the “why?” buttons attach a step for you)." }),
        el("div", { class: "ivg-suggest" }, SUGGESTIONS.map((s) =>
          el("button", { class: "ivg-chip", text: s, onclick: () => sendTurn(s) }))),
      ));
    }
    for (const turn of t.turns) refs.body.appendChild(bubble(turn));
    if (showPending) refs.body.appendChild(pendingBubble(t));
    refs.body.scrollTop = refs.body.scrollHeight;
  }

  function bubble(turn) {
    if (turn.role === "user") {
      return el("div", { class: "ivg-msg user" }, el("div", { class: "ivg-bubble", text: turn.content }));
    }
    const wrap = el("div", { class: `ivg-msg assistant${turn.error ? " error" : ""}` });
    if (turn.error) {
      wrap.appendChild(el("div", { class: "ivg-bubble", text: turn.content }));
      return wrap;
    }
    const bub = el("div", { class: "ivg-bubble md-bubble" });
    bub.appendChild(renderMarkdown(turn.content));
    wrap.appendChild(bub);
    if (turn.reasoning) {
      const pre = el("pre", { class: "ivg-reasoning", text: turn.reasoning, style: "display:none" });
      const tog = el("button", {
        class: "ivg-think-toggle", text: "▸ thinking",
        onclick: () => {
          const shown = pre.style.display !== "none";
          pre.style.display = shown ? "none" : "";
          tog.textContent = shown ? "▸ thinking" : "▾ thinking";
        },
      });
      wrap.appendChild(tog);
      wrap.appendChild(pre);
    }
    return wrap;
  }

  function pendingBubble(t) {
    const msg = t.baseFetching || !t.base ? "reading the scene…" : "analyzing the scene…";
    return el("div", { class: "ivg-msg assistant pending" },
      el("div", { class: "ivg-bubble" },
        el("span", { class: "ivg-dots" }, el("i"), el("i"), el("i")),
        el("span", { text: msg }),
      ),
    );
  }

  // ── @-mention: attach a scene object (its emitted/placed steps) or a step ────

  // Mentionable items — objects/zones first, then raw steps. Mentioning a node
  // attaches its history in one go: its provenance calls (the decompose that
  // EMITTED it + the bbox that PLACED it) PLUS the relevant zone_plan — a zone
  // attaches its OWN plan; an object attaches its emitting zone's plan, i.e. the
  // start of that object's history (plan → emitted → placed). A STEP attaches
  // just that call.
  function mentionCandidates(query) {
    const model = obsModel?.model;
    const q = query.trim().toLowerCase();
    const objects = [];
    const steps = [];
    if (model) {
      for (const id of model.order) {
        const node = model.nodes.get(id);
        const nodeKind = node?.kind ?? "node";
        const idxSet = new Set(
          (model.provenance?.get(id) ?? []).map((p) => p.call?.index).filter((i) => i != null),
        );
        const planZone = nodeKind === "zone" ? id : emittingRegion(model, id);
        const planIdx = planZone ? zonePlanCallIndex(model, planZone) : null;
        if (planIdx != null) idxSet.add(planIdx);
        const indices = [...idxSet];
        if (!indices.length) continue; // e.g. the root before anything named / planned it
        if (q && !`${id} ${node?.prompt ?? ""}`.toLowerCase().includes(q)) continue;
        objects.push({ kind: "object", id, nodeKind, indices });
      }
      for (const c of model.calls) {
        const label = `${c.template ?? c.step ?? "?"} · ${c.node ?? "?"} · #${c.index}`;
        if (q && !label.toLowerCase().includes(q)) continue;
        steps.push({ kind: "step", index: c.index, call: c });
      }
    }
    return [...objects.slice(0, 40), ...steps.slice(0, 40)];
  }

  function openMention(query) {
    mention = { start: mention?.start ?? refs.input.selectionStart, items: mentionCandidates(query), active: 0 };
    renderMention();
  }

  function refreshMention() {
    const input = refs.input;
    const pos = input.selectionStart;
    const before = input.value.slice(0, pos);
    const m = before.match(/(?:^|\s)@([\w./#-]*)$/);
    if (!m) { closeMention(); return; }
    mention = { start: pos - m[1].length - 1, items: mentionCandidates(m[1]), active: 0 };
    renderMention();
  }

  // mousedown (not click) so it fires before the textarea blur closes the popup.
  function mentionRow(it, i) {
    const t = thread();
    const active = i === mention.active;
    const handlers = {
      onmousedown: (ev) => { ev.preventDefault(); selectMention(it); },
      onmouseenter: () => { mention.active = i; syncMentionActive(); },
    };
    if (it.kind === "object") {
      const allAttached = t && it.indices.every((idx) => t.attached.has(idx));
      return el("div", { class: `ivg-mention-row${active ? " active" : ""}${allAttached ? " on" : ""}`, ...handlers },
        el("span", { class: "ivg-mention-kind", text: it.nodeKind === "zone" ? "zone" : "obj" }),
        el("span", { class: "ivg-mention-node", text: it.id }),
        el("span", { class: "ivg-mention-idx", text: `${it.indices.length} step${it.indices.length === 1 ? "" : "s"}` }),
        allAttached ? el("span", { class: "ivg-mention-tick", text: "✓" }) : null,
      );
    }
    const c = it.call;
    const attached = t?.attached.has(it.index);
    return el("div", { class: `ivg-mention-row${active ? " active" : ""}${attached ? " on" : ""}`, ...handlers },
      el("span", { class: "ivg-mention-step", text: c.template ?? c.step ?? "?" }),
      el("span", { class: "ivg-mention-node", text: c.node ?? "?" }),
      el("span", { class: "ivg-mention-idx", text: `#${it.index}` }),
      attached ? el("span", { class: "ivg-mention-tick", text: "✓" }) : null,
    );
  }

  function renderMention() {
    const pop = refs.mention;
    pop.replaceChildren();
    if (!mention) { pop.style.display = "none"; return; }
    if (!mention.items.length) {
      pop.appendChild(el("div", { class: "ivg-mention-empty", text: (obsModel?.model.calls.length ? "no matching objects or steps" : "nothing to mention yet") }));
      pop.style.display = "block";
      return;
    }
    let lastKind = null;
    mention.items.forEach((it, i) => {
      if (it.kind !== lastKind) {
        lastKind = it.kind;
        pop.appendChild(el("div", { class: "ivg-mention-head", text: it.kind === "object" ? "objects & zones" : "steps" }));
      }
      pop.appendChild(mentionRow(it, i));
    });
    pop.style.display = "block";
  }

  function syncMentionActive() {
    const rows = refs.mention.querySelectorAll(".ivg-mention-row");
    rows.forEach((r, i) => r.classList.toggle("active", i === mention.active));
  }

  function selectMention(it) {
    const t = thread();
    if (t) {
      // An object folds in its emitted + placed steps; a step is just itself.
      if (it.kind === "object") for (const idx of it.indices) t.attached.add(idx);
      else t.attached.add(it.index);
      renderAttach();
      updateStatus();
    }
    // Keep the mention IN the message text — replace any partial "@query" the
    // user typed with the canonical token (or insert it at the caret when opened
    // via the button). It's sent verbatim, so the question reads "…@sofa…"
    // alongside the context the mention pulled in.
    const input = refs.input;
    const token = it.kind === "object"
      ? `@${it.id}`
      : `@${it.call.template ?? it.call.step ?? "step"}#${it.index}`;
    const caret = input.selectionStart;
    const hasQuery = mention && typeof mention.start === "number"
      && input.value[mention.start] === "@" && caret >= mention.start;
    const from = hasQuery ? mention.start : caret;
    const before = input.value.slice(0, from);
    const after = input.value.slice(caret);
    // Space it off from surrounding words: a leading space when inserting the
    // token (via the button) right after non-space text, and a trailing one
    // unless the next char already is one.
    const lead = !hasQuery && before && !/\s$/.test(before) ? " " : "";
    const insert = lead + token + (after.startsWith(" ") ? "" : " ");
    input.value = before + insert + after;
    input.selectionStart = input.selectionEnd = before.length + insert.length;
    closeMention();
    refs.input.focus();
  }

  function moveMention(delta) {
    if (!mention || !mention.items.length) return;
    mention.active = (mention.active + delta + mention.items.length) % mention.items.length;
    syncMentionActive();
    refs.mention.querySelectorAll(".ivg-mention-row")[mention.active]?.scrollIntoView({ block: "nearest" });
  }

  function closeMention() {
    if (blurTimer) { clearTimeout(blurTimer); blurTimer = null; }
    mention = null;
    if (refs) { refs.mention.replaceChildren(); refs.mention.style.display = "none"; }
  }

  function onInputKeydown(ev) {
    if (mention) {
      if (ev.key === "ArrowDown") { ev.preventDefault(); moveMention(1); return; }
      if (ev.key === "ArrowUp") { ev.preventDefault(); moveMention(-1); return; }
      if (ev.key === "Enter" || ev.key === "Tab") {
        if (mention.items.length) { ev.preventDefault(); selectMention(mention.items[mention.active]); return; }
      }
      if (ev.key === "Escape") { ev.preventDefault(); ev.stopPropagation(); closeMention(); return; }
    }
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); sendTurn(); }
  }

  // ── send ─────────────────────────────────────────────────────────────────

  async function sendTurn(preset) {
    const id = currentId;
    const t = id ? threads.get(id) : null;
    // Gate only on THIS thread's in-flight state — a request on another cell /
    // branch must not block this one. (The thread object is stable, so it stays
    // valid while the user views / sends on other threads meanwhile.)
    if (!t || t.busy) return;
    const text = (preset ?? refs.input.value).trim();
    if (!text) return;
    t.turns.push({ role: "user", content: text });
    refs.input.value = "";
    closeMention();
    t.busy = true;
    syncSendControls();
    renderTranscript();
    await ensureBase();
    let result = null;
    let error = null;
    if (!t.base) {
      error = new Error(t.baseError || "couldn't load the scene context for this cell");
    } else {
      try {
        result = await api.inquire({
          system: composeSystem(t),
          messages: t.turns.map((m) => ({ role: m.role, content: m.content })),
        });
      } catch (e) {
        error = e;
      }
    }
    if (error) t.turns.push({ role: "assistant", content: error.message, error: true });
    else t.turns.push({ role: "assistant", content: result.answer || "(empty response)", reasoning: result.reasoning || "" });
    t.busy = false;
    // Only touch the UI if this thread is still on screen — otherwise the answer
    // is stored on its (background) thread and shown when it's next viewed.
    if (currentId === id) { syncSendControls(); renderTranscript(); refs.input.focus(); }
  }

  // ── context transparency modal ─────────────────────────────────────────────

  function showContextModal() {
    const t = thread();
    openModal("investigator context", (close) => ({
      body: [
        el("div", { class: "m-hint", text:
          "The exact system prompt sent to the reviewer with each question — pipeline explainer, every prompt template, the full scene, the step timeline, and any attached steps. Auto-assembled; refreshes as the scene grows or attachments change." }),
        (() => {
          if (!t) return el("div", { class: "m-hint", text: "No cell open." });
          if (!t.base) return el("div", { class: "m-hint", text: t.baseFetching ? "Scene context is still loading…" : "Scene context not loaded — open the panel on a started cell, or click ↻." });
          const text = composeSystem(t);
          const pre = el("pre", { style: "max-height:60vh;overflow:auto;background:#0c0d10;border:1px solid var(--line-2);border-radius:5px;padding:8px;white-space:pre-wrap;word-break:break-word", text });
          return el("div", {},
            el("div", { class: "muted", style: "margin-bottom:6px", text: `${shortBytes(text)} · ${t.base.bundle.steps.length} steps · ${t.attached.size} attached` }),
            pre,
          );
        })(),
      ],
      actions: [el("button", { class: "primary", text: "close", onclick: close })],
    }));
  }

  // ── DOM build ──────────────────────────────────────────────────────────────

  const title = el("span", { class: "ivg-title", text: "investigator" });
  const sub = el("span", { class: "ivg-sub muted" });
  const ctxBtn = el("button", {
    class: "ivg-hbtn", text: "context",
    title: "view the exact grounding + question context being sent to the reviewer",
    onclick: showContextModal,
  });
  const refreshBtn = el("button", {
    class: "ivg-hbtn", text: "↻",
    title: "re-fetch the scene context + step timeline (the cell may have advanced)",
    onclick: refreshBase,
  });
  const clearBtn = el("button", {
    class: "ivg-hbtn", text: "clear",
    title: "clear this cell's conversation (keeps its attached steps)",
    onclick: () => { const t = thread(); if (t) { t.turns = []; renderTranscript(); refs.input.focus(); } },
  });
  const closeBtn = el("button", { class: "ivg-hbtn", text: "close ✕", onclick: () => onClose() });
  const head = el("div", { class: "ivg-head" },
    el("div", { class: "ivg-head-text" }, title, sub),
    el("span", { style: "margin-left:auto" }),
    ctxBtn, refreshBtn, clearBtn, closeBtn,
  );

  const status = el("div", { class: "ivg-status muted" });
  const body = el("div", { class: "ivg-body" });
  const attach = el("div", { class: "ivg-attach", style: "display:none" });

  const input = el("textarea", {
    class: "ivg-input", rows: "3", spellcheck: "false",
    placeholder: "Ask about the whole scene…  (@ to attach an object or step · Enter to send · Shift+Enter for a newline)",
  });
  input.addEventListener("input", refreshMention);
  input.addEventListener("keydown", onInputKeydown);
  // Close the mention popup on blur, but let a refocus (e.g. clicking the
  // "@ add" button, which briefly blurs the input then refocuses it) cancel
  // that close so the popup it just opened doesn't flash shut.
  input.addEventListener("blur", () => { blurTimer = setTimeout(closeMention, 150); });
  input.addEventListener("focus", () => { if (blurTimer) { clearTimeout(blurTimer); blurTimer = null; } });
  const addStepBtn = el("button", {
    class: "ivg-add", text: "@ add",
    title: "mention an object (attaches its history: zone plan → emitted → placed), a zone (its plan + emitted + placed), or a specific pipeline step",
    onclick: () => { openMention(""); input.focus(); },
  });
  const sendBtn = el("button", { class: "primary ivg-send", text: "ask", onclick: () => sendTurn() });
  const mentionPop = el("div", { class: "ivg-mention", style: "display:none" });
  const foot = el("div", { class: "ivg-foot" },
    mentionPop,
    el("div", { class: "ivg-inputrow" }, input, el("div", { class: "ivg-footbtns" }, addStepBtn, sendBtn)),
  );

  hostEl.replaceChildren(head, status, body, attach, foot);
  refs = { title, sub, status, body, attach, input, sendBtn, mention: mentionPop };
  renderTranscript();
  updateStatus();

  // A prompt-lab edit to a run's snapshot (apply-to-run / revert-to-base)
  // invalidates every built base for that run's cells, so the next use re-fetches
  // the CURRENT templates + scene. The bound thread refreshes live; the rest
  // rebuild lazily on their next open / ↻ / send.
  on("prompts-applied", (run) => {
    let currentTouched = false;
    for (const [id, t] of threads) {
      if (!id.startsWith(`${run}|`)) continue;
      t.base = null;
      t.baseError = null;
      if (id === currentId) currentTouched = true;
    }
    if (currentTouched) { updateStatus(); ensureBase(); }
  });

  // ── public instance API ──────────────────────────────────────────────────

  return {
    // Bind the panel to a cell + its live obs model. `fetch` = start loading the
    // base grounding now (passed when the panel is already open, so switching
    // cells preloads without stealing input focus).
    setContext(next, model, { fetch = false } = {}) {
      obsModel = model ?? null;
      ctx = next ?? null;
      currentId = convId(ctx);
      if (currentId && !threads.has(currentId)) {
        threads.set(currentId, { turns: [], attached: new Set(), base: null, baseFetching: false, baseError: null, busy: false });
      }
      closeMention();
      renderShell();
      renderAttach();
      renderTranscript();
      syncSendControls(); // reflect the newly-viewed thread's own in-flight state
      updateStatus();
      if (fetch) ensureBase();
    },
    // Attach one step (a call index) as deep context and surface it — the entry
    // point the per-step "why?" buttons use.
    attachStep(index) {
      const t = thread();
      if (!t || index == null) return;
      t.attached.add(index);
      renderAttach();
      updateStatus();
      ensureBase();
      refs.input.focus();
    },
    // The column just became visible — load the base grounding + focus.
    onShown() {
      if (!currentId) return;
      ensureBase();
      updateStatus();
      refs.input.focus();
    },
    // Drop the (now-stale) obs-model reference but KEEP threads so re-opening the
    // cell restores its conversation.
    reset() {
      ctx = null;
      currentId = null;
      obsModel = null;
      closeMention();
      renderShell();
      renderAttach();
      renderTranscript();
      syncSendControls();
      updateStatus();
    },
  };
}
