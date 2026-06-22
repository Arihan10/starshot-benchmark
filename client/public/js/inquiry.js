// The decision-inquiry panel: a persistent, per-call chat that CONTINUES the
// LLM call a pipeline step made. Opened from the "why?" button on a call row in
// the obs dock.
//
// The thread is seeded with that call's real conversation — its exact system
// prompt, the user message it received, and the output it produced — and every
// new turn is sent back to the SAME model, so it answers as itself, picking up
// where it left off. No analyst persona, no appended grounding: you're just
// talking to the model that made the decision, and can type anything. Threads
// live in-memory keyed by run|slot|model|branch|callIndex, so reopening a
// step's panel restores its conversation, and the dock's streaming re-renders
// never disturb it (the panel is mounted on <body>).

import { el, shortBytes, fmtJson } from "./ui.js";
import { api } from "./api.js";

// Seed the chat with the step's real conversation: the user message it received
// and the output it produced (its reasoning kept for display, not replayed as
// content). The system prompt lives separately on the conversation (editable in
// the panel) and is sent as the system message. Together these ARE the call —
// continuing the thread sends them straight back to the model, no framing added.
function seedTurns(call) {
  return [
    { role: "user", content: call.user || "(empty)", seed: true },
    { role: "assistant", content: fmtJson(call.output), reasoning: call.reasoning || "", seed: true },
  ];
}

function seedConversation(call) {
  return { model: call.model || "", system: call.system || "", turns: seedTurns(call) };
}

const conversations = new Map(); // convId -> { model, system, turns: [{role, content, reasoning?, seed?, error?}] }

let panel = null;
let refs = null; // { title, sub, prompt, body, input, sendBtn }
let current = null; // { id, call, ctx }
let busy = false;
let busyId = null; // the conversation awaiting a reply (drives the pending bubble)

const SUGGESTIONS = [
  "Why did you choose these dimensions and this position?",
  "Why did you decompose it the way you did?",
  "What in the input drove this output?",
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
    conversations.set(current.id, seedConversation(call));
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
    class: "inq-clear", text: "clear chat", title: "drop your follow-ups — back to the step's original call",
    onclick: () => {
      if (!current) return;
      conversations.get(current.id).turns = seedTurns(current.call);
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
  // Always-visible "drop the step's own reasoning into the box" button. The
  // step always emits a reasoning trace, so it sits right by the input ready to
  // paste it in to quote or edit — it never sends on its own.
  const reasonBtn = el("button", {
    class: "inq-reason", type: "button", text: "↩ reasoning",
    title: "paste this step's reasoning trace into the message box to quote or edit — doesn't send",
    onclick: () => injectReasoning(current?.call?.reasoning),
  });
  const sendBtn = el("button", { class: "primary inq-send", text: "ask", onclick: () => sendTurn() });
  const foot = el("div", { class: "inq-foot" }, input, reasonBtn, sendBtn);

  panel = el("div", { id: "inquiry-panel" }, head, prompt, body, foot);
  document.body.appendChild(panel);
  refs = { title, sub, prompt, body, input, reasonBtn, sendBtn };

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
  refs.sub.textContent = `continuing this call · ${call.model ?? "?"}`;
  renderPrompt();
  syncBusyControls();
}

// The system prompt the step's model ran under — sent verbatim as the system
// message of the continued conversation, editable, and collapsed by default so
// the conversation itself stays the focus.
function renderPrompt() {
  const { call } = current;
  const conv = conversations.get(current.id);
  refs.prompt.textContent = "";

  const size = el("span", { class: "muted size", text: shortBytes(conv.system) });
  const ta = el("textarea", { class: "inq-prompt-text", spellcheck: "false" });
  ta.value = conv.system;
  ta.addEventListener("input", () => {
    conv.system = ta.value;
    size.textContent = shortBytes(conv.system);
  });

  const reset = el("button", {
    class: "inq-prompt-reset", text: "reset",
    title: "restore this step's original system prompt (discards edits)",
    onclick: (ev) => {
      ev.stopPropagation();
      conv.system = call.system || "";
      ta.value = conv.system;
      size.textContent = shortBytes(conv.system);
    },
  });
  const headRow = el("div", {
    class: "inq-prompt-head",
    title: "the system prompt this step's model ran under — sent as the system message (plus a short directive to reply in prose, not the step's JSON); edit it freely",
    onclick: (ev) => { if (!ev.target.closest(".inq-prompt-reset")) refs.prompt.classList.toggle("open"); },
  },
    el("span", { class: "caret", text: "▸" }),
    el("span", { text: "system prompt the model ran under" }),
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
  for (const turn of turns) refs.body.appendChild(bubble(turn));
  // Until the first follow-up, invite one below the seeded call — the seed
  // turns above are the step's own input/output, so the model picks up there.
  const hasFollowup = turns.some((t) => !t.seed);
  if (!hasFollowup && !showingPending) {
    refs.body.appendChild(el("div", { class: "inq-hint" },
      el("div", { text:
        "Continue the conversation with the model that ran this step — it picks up right where it left off and replies in plain prose (not the step's JSON). Ask it anything." }),
      el("div", { class: "inq-suggest" }, SUGGESTIONS.map((s) =>
        el("button", { class: "inq-chip", text: s, onclick: () => sendTurn(s) }))),
    ));
  }
  if (showingPending) refs.body.appendChild(pendingBubble());
  refs.body.scrollTop = refs.body.scrollHeight;
}

function bubble(turn) {
  const seed = turn.seed ? " seed" : "";
  if (turn.role === "user") {
    return el("div", { class: `inq-msg user${seed}` }, el("div", { class: "inq-bubble", text: turn.content }));
  }
  const wrap = el("div", { class: `inq-msg assistant${seed}${turn.error ? " error" : ""}` });
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
    const actions = el("div", { class: "inq-think-actions" }, tog);
    // The seed turn IS this step — its reasoning has the always-visible footer
    // "↩ reasoning" button, so per-bubble inject is only for follow-up replies.
    if (!turn.seed) {
      actions.appendChild(el("button", {
        class: "inq-think-inject", text: "→ message",
        title: "drop this reply's reasoning into the message box to quote, edit, or ask about it",
        onclick: () => injectReasoning(turn.reasoning),
      }));
    }
    wrap.appendChild(actions);
    wrap.appendChild(pre);
  }
  return wrap;
}

function pendingBubble() {
  return el("div", { class: "inq-msg assistant pending" },
    el("div", { class: "inq-bubble" },
      el("span", { class: "inq-dots" }, el("i"), el("i"), el("i")),
      el("span", { text: "thinking…" }),
    ),
  );
}

function syncBusyControls() {
  if (!refs) return;
  refs.sendBtn.disabled = busy;
  refs.sendBtn.textContent = busy ? "thinking…" : "ask";
  refs.input.disabled = busy;
  refs.reasonBtn.disabled = busy || !current?.call?.reasoning;
}

function setBusy(on, id) {
  busy = on;
  busyId = on ? id : null;
  syncBusyControls();
}

// Drop a reasoning trace into the message box so it can be quoted, edited, or
// asked about before sending. Splices at the caret when the box is focused,
// else appends; pads with newlines so the block keeps its own lines.
function injectReasoning(text) {
  if (!refs || !text) return;
  const input = refs.input;
  const focused = document.activeElement === input;
  const start = focused ? input.selectionStart : input.value.length;
  const end = focused ? input.selectionEnd : input.value.length;
  const before = input.value.slice(0, start);
  const after = input.value.slice(end);
  const lead = before && !before.endsWith("\n") ? "\n" : "";
  const tail = after && !after.startsWith("\n") ? "\n" : "";
  input.value = before + lead + text + tail + after;
  const caret = before.length + lead.length + text.length;
  input.focus();
  input.setSelectionRange(caret, caret);
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
      model: conv.model,
      // The step's own system prompt (editable in the panel) + its full
      // conversation so far — sent verbatim, continued by the step's own model.
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
