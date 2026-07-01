// The decision-inquiry panel: a persistent, per-call chat that asks Claude
// Opus 4.8 (xhigh reasoning) WHY the subject model decided what it did in one
// pipeline step. Opened from the "why?" button on a call row in the obs dock.
//
// The reviewer's FULL system prompt — the analyst framing plus the step's exact
// system / input / output / reasoning — is assembled HERE and shown in an
// editable, prefilled box, so what you see is byte-for-byte what gets sent (and
// you can tweak it before asking). The server is a thin pass-through. Threads
// live in-memory keyed by run|slot|model|branch|callIndex, so reopening a
// step's panel restores both its prompt and its conversation, and the dock's
// streaming re-renders never disturb it (the panel is mounted on <body>).

import { el, shortBytes, fmtJson } from "./ui.js";
import { api } from "./api.js";

// The analyst framing prepended to every inquiry. The step's grounding (built
// per call below) is appended to form the exact system prompt sent — and both
// are surfaced verbatim in the panel's editable prompt box.
const INQUIRY_SYSTEM = `You are a senior analyst performing a forensic post-mortem of a SINGLE decision made by another language model (the "subject model"). The subject model is one step of an automated pipeline that turns a text prompt into a 3D scene by recursively decomposing it into zones and objects, each with an axis-aligned bounding box and spatial relationships. The canonical world frame is right-handed, Y-up, meters: +X is right, +Y is up, +Z is toward the viewer (front), -Z is away (back).

You are given the EXACT material that defined and resulted from the subject model's call for this one step:
  - SYSTEM PROMPT: the instructions the subject model operated under.
  - INPUT: the exact user message it received (it already contains any injected scene context).
  - OUTPUT: the structured result it produced.
  - REASONING: the subject model's own private chain-of-thought, if it exposed one.

A developer benchmarking the pipeline will ask you why the subject model did particular things — why it placed an object at certain coordinates, why it decomposed a zone the way it did, why some downstream consequence followed, and so on.

Rules:
  - Ground every claim in the provided SYSTEM / INPUT / OUTPUT / REASONING. Quote the relevant lines or fields.
  - The REASONING is the strongest evidence of intent: when it explains a decision, cite it directly. When it is absent or silent on the asked-about point, say so plainly and separate what the subject model actually reasoned from what you are inferring purely from its prompt and output.
  - Never invent motives the evidence does not support. "The reasoning does not address this" and "this looks like an unexplained or arbitrary choice" are valid, valuable answers.
  - Be concrete: reference specific ids, coordinates, dimensions, and prompt clauses. Keep answers tight and skimmable; lead with the direct answer, then the evidence.`;

// The step's grounding block, mirroring exactly what the server used to append:
// verbatim system / input / output / reasoning straight off the obs-model call.
function groundingFor(call) {
  return [
    `=== SUBJECT MODEL: ${call.model || "unknown"} ===`,
    `=== PIPELINE STEP: ${call.template ?? call.step ?? "unknown"} ===`,
    "",
    "--- SYSTEM PROMPT the subject model operated under ---",
    call.system || "(empty)",
    "",
    "--- INPUT (user message) the subject model received ---",
    call.user || "(empty)",
    "",
    "--- OUTPUT the subject model produced ---",
    fmtJson(call.output),
    "",
    "--- The subject model's PRIVATE REASONING ---",
    call.reasoning || "(the subject model exposed no separate reasoning trace)",
  ].join("\n");
}

function defaultPrompt(call) {
  return `${INQUIRY_SYSTEM}\n\n${groundingFor(call)}`;
}

const conversations = new Map(); // convId -> { system, turns: [{role, content, reasoning?, error?}] }

let panel = null;
let refs = null; // { title, sub, prompt, body, input, sendBtn }
let current = null; // { id, call, ctx }
let busy = false;
let busyId = null; // the conversation awaiting a reply (drives the pending bubble)

const SUGGESTIONS = [
  "Why did it choose these dimensions and position?",
  "Why did it decompose things the way it did?",
  "What in the input or its reasoning drove this output?",
];

function convId(ctx, call) {
  // The branch id (not a mere has-branch flag) is part of the key: a cell's
  // sibling sims share run|slot|model AND the same call.index for a co-forked
  // step, so only the branch id tells their threads apart. `|| "src"` folds the
  // no-branch sentinels (false/null) to one stable source token.
  return `${ctx.run}|${ctx.slot}|${ctx.model}|${ctx.branch || "src"}|${call.index}`;
}

export function openInquiry(call, ctx) {
  ensurePanel();
  current = { id: convId(ctx, call), call, ctx };
  if (!conversations.has(current.id)) {
    conversations.set(current.id, { system: defaultPrompt(call), turns: [] });
  }
  renderShell();
  renderTranscript();
  panel.classList.add("open");
  refs.input.focus();
}

export function closeInquiry() {
  panel?.classList.remove("open");
}

// Close the panel when the viewed cell changes — including a switch between two
// sibling sims of the SAME cell (same run/slot/model), so a chat about one
// branch's step never lingers over another's. Same-view reopens (resume / step
// re-subscribe openCell with the identical view) keep it. Falsy branch sentinels
// (false/null) both mean "source", so normalize before the identity compare.
export function notifyView(view) {
  if (!panel?.classList.contains("open") || !current) return;
  const c = current.ctx;
  if (
    !view ||
    c.run !== view.run || c.slot !== view.slot ||
    c.model !== view.model || (c.branch || null) !== (view.branch || null)
  ) {
    closeInquiry();
  }
}

function ensurePanel() {
  if (panel) return;
  const title = el("span", { class: "inq-title" });
  const sub = el("span", { class: "inq-sub muted" });
  const clearBtn = el("button", {
    class: "inq-clear", text: "clear chat", title: "clear this step's conversation (keeps the prompt)",
    onclick: () => {
      if (!current) return;
      conversations.get(current.id).turns = [];
      renderTranscript();
      refs.input.focus();
    },
  });
  const closeBtn = el("button", { text: "close ✕", style: "margin-left:8px", onclick: closeInquiry });
  const head = el("div", { class: "inq-head" },
    el("div", { class: "inq-head-text" }, title, sub),
    el("span", { style: "margin-left:auto" }), clearBtn, closeBtn,
  );

  const prompt = el("div", { class: "inq-prompt" });
  const body = el("div", { class: "inq-body" });

  const input = el("textarea", {
    class: "inq-input", rows: "2", spellcheck: "false",
    placeholder: "Ask why the model did this…  (Enter to send · Shift+Enter for a newline)",
  });
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); sendTurn(); }
  });
  const sendBtn = el("button", { class: "primary inq-send", text: "ask", onclick: () => sendTurn() });
  const foot = el("div", { class: "inq-foot" }, input, sendBtn);

  panel = el("div", { id: "inquiry-panel" }, head, prompt, body, foot);
  document.body.appendChild(panel);
  refs = { title, sub, prompt, body, input, sendBtn };

  // Esc closes the chat first (capture phase + stopPropagation keeps the
  // overlay/drawer Esc handlers from also firing), unless a modal owns Esc.
  document.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape" || !panel.classList.contains("open")) return;
    if (document.getElementById("modal-root").firstChild) return;
    closeInquiry();
    ev.stopPropagation();
  }, true);
}

function renderShell() {
  const { call } = current;
  const step = call.template ?? call.step ?? "step";
  refs.title.textContent = `why? · ${step}`;
  refs.sub.textContent = `subject: ${call.model ?? "?"}  ·  reviewer: claude-opus-4.8 · xhigh`;
  renderPrompt();
  syncBusyControls();
}

// The exact prompt sent to the reviewer — prefilled with the analyst framing +
// this step's grounding, editable, and the source of truth for what gets sent.
function renderPrompt() {
  const { call } = current;
  const conv = conversations.get(current.id);
  refs.prompt.textContent = "";
  refs.prompt.classList.add("open"); // visible by default so the prompt is right there

  const size = el("span", { class: "muted size", text: shortBytes(conv.system) });
  const ta = el("textarea", { class: "inq-prompt-text", spellcheck: "false" });
  ta.value = conv.system;
  ta.addEventListener("input", () => {
    conv.system = ta.value;
    size.textContent = shortBytes(conv.system);
  });

  const reset = el("button", {
    class: "inq-prompt-reset", text: "reset",
    title: "restore the prefilled prompt for this step (discards edits)",
    onclick: (ev) => {
      ev.stopPropagation();
      conv.system = defaultPrompt(call);
      ta.value = conv.system;
      size.textContent = shortBytes(conv.system);
    },
  });
  const headRow = el("div", {
    class: "inq-prompt-head",
    title: "the exact system prompt sent to the reviewer — edit it freely",
    onclick: (ev) => { if (!ev.target.closest(".inq-prompt-reset")) refs.prompt.classList.toggle("open"); },
  },
    el("span", { class: "caret", text: "▸" }),
    el("span", { text: "exact prompt sent to the reviewer" }),
    size, reset,
  );

  refs.prompt.appendChild(headRow);
  refs.prompt.appendChild(el("div", { class: "inq-prompt-body" }, ta));
}

function renderTranscript() {
  if (!current) return;
  const turns = conversations.get(current.id).turns;
  const showingPending = busy && busyId === current.id;
  refs.body.textContent = "";
  if (!turns.length && !showingPending) {
    refs.body.appendChild(el("div", { class: "inq-hint" },
      el("div", { text:
        "Ask why the subject model made a choice in this step. Expand the prompt above to see (and edit) the exact bytes the reviewer receives." }),
      el("div", { class: "inq-suggest" }, SUGGESTIONS.map((s) =>
        el("button", { class: "inq-chip", text: s, onclick: () => sendTurn(s) }))),
    ));
  }
  for (const turn of turns) refs.body.appendChild(bubble(turn));
  if (showingPending) refs.body.appendChild(pendingBubble());
  refs.body.scrollTop = refs.body.scrollHeight;
}

function bubble(turn) {
  if (turn.role === "user") {
    return el("div", { class: "inq-msg user" }, el("div", { class: "inq-bubble", text: turn.content }));
  }
  const wrap = el("div", { class: `inq-msg assistant${turn.error ? " error" : ""}` });
  wrap.appendChild(el("div", { class: "inq-bubble", text: turn.content }));
  if (turn.reasoning) {
    const pre = el("pre", { class: "inq-reasoning", text: turn.reasoning, style: "display:none" });
    const tog = el("button", {
      class: "inq-think-toggle", text: "▸ thinking",
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

function pendingBubble() {
  return el("div", { class: "inq-msg assistant pending" },
    el("div", { class: "inq-bubble" },
      el("span", { class: "inq-dots" }, el("i"), el("i"), el("i")),
      el("span", { text: "analyzing the step…" }),
    ),
  );
}

function syncBusyControls() {
  if (!refs) return;
  refs.sendBtn.disabled = busy;
  refs.sendBtn.textContent = busy ? "thinking…" : "ask";
  refs.input.disabled = busy;
}

function setBusy(on, id) {
  busy = on;
  busyId = on ? id : null;
  syncBusyControls();
}

async function sendTurn(preset) {
  if (busy || !current) return;
  const text = (preset ?? refs.input.value).trim();
  if (!text) return;
  const { id } = current;
  const conv = conversations.get(id);
  conv.turns.push({ role: "user", content: text });
  refs.input.value = "";
  setBusy(true, id);
  renderTranscript();
  let result = null;
  let error = null;
  try {
    result = await api.inquire({
      // The exact, possibly-edited prompt shown in the panel — verbatim.
      system: conv.system,
      messages: conv.turns.map((t) => ({ role: t.role, content: t.content })),
    });
  } catch (e) {
    error = e;
  }
  // The answer belongs to `conv` even if the user switched conversations while
  // it was in flight — re-render only if we're still on that thread.
  if (error) conv.turns.push({ role: "assistant", content: error.message, error: true });
  else conv.turns.push({ role: "assistant", content: result.answer || "(empty response)", reasoning: result.reasoning || "" });
  setBusy(false, null);
  if (current && current.id === id) {
    renderTranscript();
    refs.input.focus();
  }
}
