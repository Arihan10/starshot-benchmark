// Tiny DOM + modal + toast helpers shared by every panel.

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "title") node.title = v;
    else if (k.startsWith("on") && typeof v === "function") {
      node.addEventListener(k.slice(2), v);
    } else if (k === "dataset") {
      for (const [dk, dv] of Object.entries(v)) node.dataset[dk] = dv;
    } else if (typeof v === "boolean") {
      if (v) node.setAttribute(k, "");
    } else {
      node.setAttribute(k, v);
    }
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

const toastRoot = document.getElementById("toast-root");

export function toast(message, cls = "") {
  const t = el("div", { class: `toast ${cls}`, text: message, title: message });
  toastRoot.appendChild(t);
  setTimeout(() => t.remove(), cls === "err" ? 6000 : 3200);
}

const modalRoot = document.getElementById("modal-root");

// One modal at a time. `build(close, setError)` returns {body, actions}.
export function openModal(title, build) {
  closeModal();
  const back = el("div", { class: "modal-back" });
  const errorEl = el("div", { class: "m-error" });
  const close = () => back.remove();
  const setError = (msg) => { errorEl.textContent = msg ?? ""; };
  const { body, actions } = build(close, setError);
  const panel = el(
    "div", { class: "modal-panel" },
    el("header", {}, title, el("span", { class: "x", text: "×", onclick: close })),
    el("div", { class: "m-body" }, ...body, errorEl),
    el("footer", {}, ...actions),
  );
  back.addEventListener("click", (ev) => { if (ev.target === back) close(); });
  back.appendChild(panel);
  modalRoot.appendChild(back);
  const first = panel.querySelector("input, select, textarea");
  if (first) first.focus();
  return close;
}

export function closeModal() {
  modalRoot.textContent = "";
}

document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && modalRoot.firstChild) {
    closeModal();
    ev.stopPropagation();
  }
});

export function field(labelText, control) {
  return el("label", { class: "m-field" }, el("span", { text: labelText }), control);
}

// Ask for a spend-cap ceiling (USD; 0 = no cap) in a modal, then hand the
// entered value to `onSubmit`. Errors thrown by `onSubmit` surface inline; a
// clean resolve closes the modal. Shared by the cost tracker, cell overlay, and
// the board's bulk action.
export function promptCapValue(title, { current = 0, submitLabel = "set cap" } = {}, onSubmit) {
  const input = el("input", {
    type: "number", min: "0", step: "1",
    value: current > 0 ? String(current) : "",
  });
  openModal(title, (close, setError) => ({
    body: [
      field("spend cap · USD", input),
      el("div", { class: "m-hint", text:
        "the cell auto-pauses once settled spend reaches this. Enter 0 for no cap." }),
    ],
    actions: [
      el("button", { text: "cancel", onclick: close }),
      el("button", { class: "primary", text: submitLabel, onclick: async () => {
        const v = Number(input.value);
        if (input.value.trim() === "" || !Number.isFinite(v) || v < 0) {
          setError("enter a cap of 0 or more (0 = no cap)");
          return;
        }
        try { await onSubmit(v); close(); }
        catch (e) { setError(e.message); }
      } }),
    ],
  }));
}

// --- step-until picker -------------------------------------------------------
// A custom popover (NOT a native <select>) that picks a target step AND where to
// stop relative to it: pause BEFORE the step's next call (it doesn't run) or
// AFTER it (run through it, pause at the following call). The before/after
// choice is a single shared mode, persisted so it's consistent across every
// step-until control in the app. `onPick(step, before)` fires on a step click.

const STEP_UNTIL_DIR_KEY = "starshot.stepUntilBefore";
let stepUntilBefore = false;
try {
  stepUntilBefore = localStorage.getItem(STEP_UNTIL_DIR_KEY) === "1";
} catch {
  /* private mode */
}

let openStepPop = null; // { pop, trigger, cleanup } — only one open at a time

function closeStepUntilPop() {
  if (!openStepPop) return;
  openStepPop.cleanup();
  openStepPop.pop.remove();
  openStepPop = null;
}

// Anchor the popover under the trigger (flipping above / shifting left near the
// viewport edges). Fixed-positioned in viewport coords, so it re-anchors on
// scroll/resize rather than being clipped by an overflow container.
function positionStepPop(pop, trigger) {
  const r = trigger.getBoundingClientRect();
  pop.style.visibility = "hidden";
  pop.style.top = "0px";
  pop.style.left = "0px";
  const pw = pop.offsetWidth;
  const ph = pop.offsetHeight;
  let top = r.bottom + 4;
  if (top + ph > window.innerHeight - 8) top = Math.max(8, r.top - 4 - ph);
  let left = r.left;
  if (left + pw > window.innerWidth - 8) left = Math.max(8, window.innerWidth - 8 - pw);
  pop.style.top = `${Math.round(top)}px`;
  pop.style.left = `${Math.round(left)}px`;
  pop.style.visibility = "";
}

// `steps` may be an array or a getter — evaluated when the popover OPENS, so a
// control built before the step list loads still shows the current steps.
export function stepUntilSelect(steps, onPick, { label = "until…", title } = {}) {
  const trigger = el("button", {
    class: "step-until",
    text: label,
    title: title ?? "run up to the next call of a step — pause before it or after it",
  });
  trigger.addEventListener("click", (ev) => {
    ev.stopPropagation();
    if (openStepPop && openStepPop.trigger === trigger) { closeStepUntilPop(); return; }
    closeStepUntilPop();
    openStepUntilPop(trigger, steps, onPick);
  });
  return trigger;
}

function openStepUntilPop(trigger, steps, onPick) {
  const stepList = typeof steps === "function" ? (steps() || []) : (steps || []);
  const pop = el("div", { class: "step-until-pop" });

  // before/after toggle (shared, persisted mode) at the top.
  const beforeBtn = el("button", { class: "su-seg-btn", text: "before", title: "pause in front of the step — it does NOT run" });
  const afterBtn = el("button", { class: "su-seg-btn", text: "after", title: "run through the step, then pause at the next call" });
  const syncSeg = () => {
    beforeBtn.classList.toggle("on", stepUntilBefore);
    afterBtn.classList.toggle("on", !stepUntilBefore);
  };
  const setDir = (b) => {
    stepUntilBefore = b;
    try { localStorage.setItem(STEP_UNTIL_DIR_KEY, b ? "1" : "0"); } catch { /* private mode */ }
    syncSeg();
  };
  beforeBtn.addEventListener("click", (e) => { e.stopPropagation(); setDir(true); });
  afterBtn.addEventListener("click", (e) => { e.stopPropagation(); setDir(false); });
  syncSeg();
  pop.appendChild(el("div", { class: "su-seg-row" },
    el("span", { class: "su-seg-lab", text: "pause" }),
    el("div", { class: "su-seg" }, beforeBtn, afterBtn),
  ));

  if (!stepList.length) {
    pop.appendChild(el("div", { class: "su-empty", text: "no steps loaded yet" }));
  } else {
    const list = el("div", { class: "su-list" });
    for (const s of stepList) {
      list.appendChild(el("button", {
        class: "su-step", text: s,
        onclick: (e) => {
          e.stopPropagation();
          const before = stepUntilBefore;
          closeStepUntilPop();
          onPick(s, before);
        },
      }));
    }
    pop.appendChild(list);
  }

  document.body.appendChild(pop);
  positionStepPop(pop, trigger);

  const onDocClick = (e) => { if (!pop.contains(e.target) && e.target !== trigger) closeStepUntilPop(); };
  // Capture Escape so it closes only the popover (not the overlay/modal behind it).
  const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); closeStepUntilPop(); } };
  const onReflow = () => positionStepPop(pop, trigger);
  setTimeout(() => document.addEventListener("click", onDocClick), 0); // skip the opening click
  document.addEventListener("keydown", onKey, true);
  window.addEventListener("resize", onReflow);
  window.addEventListener("scroll", onReflow, true);
  openStepPop = {
    pop,
    trigger,
    cleanup: () => {
      document.removeEventListener("click", onDocClick);
      document.removeEventListener("keydown", onKey, true);
      window.removeEventListener("resize", onReflow);
      window.removeEventListener("scroll", onReflow, true);
    },
  };
}

// Objects multiselect dropdown for a 3D viewer — the replacement for the old
// single "objects" toggle. It filters objects by the decomposition step that
// emitted each (anchors / next / negative space) plus frames, all bound to
// `viewer.toggles.{anchors,next,negativeSpace,frames}`. Ticking "none" masks
// every category off and disables the rest until it's unticked — the individual
// picks are remembered (masked, not cleared), so unticking restores them.
// Shared by every viewer with a toggle row (main overlay + both compare panes).
const OBJECT_MENU_CATEGORIES = [
  ["anchors", "anchors"],
  ["next", "next"],
  ["negativeSpace", "negative space"],
  ["frames", "frames"],
];

// One document-level closer for every objects menu (registered lazily, once):
// a click that isn't kept inside an open menu collapses them all.
let objMenuCloserAdded = false;
function ensureObjMenuCloser() {
  if (objMenuCloserAdded) return;
  objMenuCloserAdded = true;
  document.addEventListener("click", () => {
    for (const m of document.querySelectorAll(".vt-menu.open")) m.classList.remove("open");
  });
}

export function buildObjectsMenu(viewer) {
  ensureObjMenuCloser();
  const noneCb = el("input", { type: "checkbox" });
  const catCbs = OBJECT_MENU_CATEGORIES.map(([key]) =>
    el("input", { type: "checkbox", checked: !!viewer.toggles[key] }));
  const btn = el("button", {
    class: "vt-menu-btn",
    title: "choose which object categories (by emitting step) and frames to show",
    text: "objects ▾",
  });
  const pop = el("div", { class: "vt-menu-pop" },
    el("label", { class: "vt-menu-item vt-menu-none" }, noneCb, "none"),
    el("div", { class: "vt-menu-sep" }),
    ...OBJECT_MENU_CATEGORIES.map(([, label], i) =>
      el("label", { class: "vt-menu-item" }, catCbs[i], label)),
  );
  const menu = el("div", { class: "vt-menu" }, btn, pop);

  const apply = () => {
    const none = noneCb.checked;
    OBJECT_MENU_CATEGORIES.forEach(([key], i) => {
      catCbs[i].disabled = none;
      // NONE masks every category off WITHOUT clearing the checkboxes, so
      // unticking it restores exactly what was showing before.
      viewer.toggles[key] = none ? false : catCbs[i].checked;
    });
    btn.classList.toggle("off", none || !catCbs.some((cb) => cb.checked));
    viewer.refreshVisibility();
  };
  noneCb.addEventListener("change", apply);
  for (const cb of catCbs) cb.addEventListener("change", apply);
  // Toggle this popup (closing any other open one first); keep clicks inside the
  // menu from reaching the document closer so ticking a box doesn't collapse it.
  btn.addEventListener("click", () => {
    const wasOpen = menu.classList.contains("open");
    for (const m of document.querySelectorAll(".vt-menu.open")) m.classList.remove("open");
    if (!wasOpen) menu.classList.add("open");
  });
  menu.addEventListener("click", (ev) => ev.stopPropagation());
  apply(); // push the initial checkbox state onto the viewer
  return menu;
}

export function fmtJson(value) {
  try { return JSON.stringify(value, null, 2); } catch { return String(value); }
}

// The big dynamic scene-state blocks injected into prompts. In the obs prompt
// view we fold these behind expandable placeholders by default so the prompt's
// own wording is readable without the scene dump.
const CONTEXT_VARS = [
  "SCENE_CONTEXT", "ROOT_OBJECTS", "TO_PLACE", "RETRY_BLOCK",
  "SIBLING_OBJECTS", "ROOT_OBJECTS_BRIEF", "OTHER_SUBREGIONS_BRIEF",
];
const CONTEXT_MIN = 80; // leave small/empty placeholders inline

function foldWidget(name, val) {
  const w = el("span", { class: "ctx-fold" });
  const collapsed = `⟨ {${name}} · ${shortBytes(val)} — expand ⟩`;
  const btn = el("span", { class: "ctx-fold-btn", text: collapsed });
  let valNode = null;
  btn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    if (valNode) { valNode.remove(); valNode = null; btn.textContent = collapsed; }
    else { btn.textContent = `⟨ {${name}} · collapse ⟩\n`; valNode = document.createTextNode(val); w.appendChild(valNode); }
  });
  w.appendChild(btn);
  return w;
}

// A <pre> showing `text` with each scene-context variable value (located via
// the call's logged `variables`) collapsed behind a per-occurrence toggle.
export function foldedPre(text, variables, cls = "") {
  const pre = el("pre", { class: cls });
  const folds = [];
  if (variables && typeof variables === "object") {
    for (const name of CONTEXT_VARS) {
      const val = variables[name];
      if (typeof val !== "string" || val.length < CONTEXT_MIN) continue;
      const idx = text.indexOf(val);
      if (idx >= 0) folds.push({ start: idx, end: idx + val.length, name, val });
    }
  }
  folds.sort((a, b) => a.start - b.start);
  const clean = [];
  let lastEnd = -1;
  for (const f of folds) if (f.start >= lastEnd) { clean.push(f); lastEnd = f.end; }
  if (!clean.length) { pre.textContent = text; return pre; }
  let pos = 0;
  for (const f of clean) {
    if (f.start > pos) pre.appendChild(document.createTextNode(text.slice(pos, f.start)));
    pre.appendChild(foldWidget(f.name, f.val));
    pos = f.end;
  }
  if (pos < text.length) pre.appendChild(document.createTextNode(text.slice(pos)));
  return pre;
}

// "expand ↔ clamp" control for any text box: display blocks (pre) drop their
// max-height clamp; textareas auto-grow to fit their content and keep
// tracking it while edited. Defaults to EXPANDED (`fitted`) so boxes fill the
// space and show their full content without a click; the button then clamps
// back to a fixed scrollable height.
// Nearest scrollable ancestor — auto-grow briefly collapses the textarea to
// measure its content, which clamps this container's scrollTop; we snapshot and
// restore it so typing never yanks the view to the top.
function scrollableAncestor(node) {
  for (let p = node.parentElement; p; p = p.parentElement) {
    const oy = getComputedStyle(p).overflowY;
    if ((oy === "auto" || oy === "scroll") && p.scrollHeight > p.clientHeight) return p;
  }
  return null;
}

export function fitToggle(target, { fitted = true } = {}) {
  const isTextarea = target.tagName === "TEXTAREA";
  let active = false;
  const sync = () => {
    if (!active) return;
    const sc = scrollableAncestor(target);
    const top = sc ? sc.scrollTop : 0;
    target.style.height = "auto";
    target.style.height = `${target.scrollHeight + 2}px`;
    if (sc) sc.scrollTop = top;
  };
  const set = (on) => {
    active = on;
    btn.textContent = on ? "clamp" : "expand";
    btn.title = on
      ? "clamp back to a fixed scrollable height"
      : "expand to full height to fit the text";
    if (isTextarea) {
      if (on) {
        target.addEventListener("input", sync);
        sync();
      } else {
        target.removeEventListener("input", sync);
        target.style.height = "";
      }
    } else {
      target.classList.toggle("fit-full", on);
    }
  };
  const btn = el("button", {
    class: "fit-btn",
    onclick: (ev) => { ev.stopPropagation(); set(!active); },
  });
  set(fitted);
  return btn;
}

export function shortBytes(s) {
  const n = (s ?? "").length;
  if (n < 1024) return `${n} ch`;
  return `${(n / 1024).toFixed(1)}k ch`;
}

// --- diff -----------------------------------------------------------------------
//
// Line-level LCS with word-level refinement of replaced runs — the exact
// changed words inside a long prompt, while staying linear in the common
// (mostly-unchanged) case. Returns ordered { op: equal|insert|delete, text }.

function lcsOpcodes(a, b) {
  // Myers-ish via DP LCS table; inputs are arrays of tokens (lines or words),
  // capped by callers so the O(n·m) table stays small.
  const n = a.length;
  const m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const ops = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { ops.push(["equal", a[i]]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { ops.push(["delete", a[i]]); i++; }
    else { ops.push(["insert", b[j]]); j++; }
  }
  while (i < n) ops.push(["delete", a[i++]]);
  while (j < m) ops.push(["insert", b[j++]]);
  return ops;
}

const _WORD_CAP = 4000;

function wordSegments(oldStr, newStr) {
  const a = (oldStr || "").match(/\s+|\S+/g) || [];
  const b = (newStr || "").match(/\s+|\S+/g) || [];
  if (a.length > _WORD_CAP || b.length > _WORD_CAP) {
    const out = [];
    if (oldStr) out.push({ op: "delete", text: oldStr });
    if (newStr) out.push({ op: "insert", text: newStr });
    return out;
  }
  return lcsOpcodes(a, b).map(([op, text]) => ({ op, text }));
}

const _LINE_CAP = 4_000_000; // n·m ceiling for the line LCS table

export function diffSegments(oldStr, newStr) {
  const a = (oldStr || "").split(/(?<=\n)/);
  const b = (newStr || "").split(/(?<=\n)/);
  if (a.length * b.length > _LINE_CAP) {
    const out = [];
    if (oldStr) out.push({ op: "delete", text: oldStr });
    if (newStr) out.push({ op: "insert", text: newStr });
    return out;
  }
  const segs = [];
  let pendDel = "";
  let pendIns = "";
  const flush = () => {
    if (pendDel || pendIns) {
      // A replaced run — refine to word level so only changed words light up.
      for (const s of wordSegments(pendDel, pendIns)) segs.push(s);
      pendDel = "";
      pendIns = "";
    }
  };
  for (const [op, text] of lcsOpcodes(a, b)) {
    if (op === "equal") { flush(); segs.push({ op: "equal", text }); }
    else if (op === "delete") pendDel += text;
    else pendIns += text;
  }
  flush();
  // Coalesce adjacent same-op segments for cleaner rendering.
  const out = [];
  for (const s of segs) {
    if (!s.text) continue;
    const last = out[out.length - 1];
    if (last && last.op === s.op) last.text += s.text;
    else out.push({ ...s });
  }
  return out;
}

// Render diff segments into a <pre>-like element with ins/del highlighting.
export function diffPre(oldStr, newStr) {
  const pre = el("pre", { class: "diff" });
  for (const seg of diffSegments(oldStr, newStr)) {
    if (seg.op === "equal") pre.appendChild(document.createTextNode(seg.text));
    else pre.appendChild(el("span", { class: seg.op === "insert" ? "diff-ins" : "diff-del", text: seg.text }));
  }
  return pre;
}

// --- LLM-call timing (shared by the sidebar call rows' hover) ----------------
// Formatted to match the api log's own cards so a call reads identically whether
// hovered in the scene sidebar or opened in the flight ledger. All values come
// from the slim `cache.llm` event the obs tree already holds — no log fetch.

const _MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

// Provider latency, formatted like the log ("340 ms" / "1.2 s" / "1m 5s").
export function fmtDurationMs(ms) {
  if (ms == null) return "--";
  if (ms < 1000) return `${ms} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const m = Math.floor(ms / 60_000);
  return `${m}m ${Math.round((ms % 60_000) / 1000)}s`;
}

// Exact wall-clock of a call boundary ("Jul 18, 05:38:12.481 PM"); `ts` is epoch seconds.
export function fmtClockTime(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  let h = d.getHours();
  const ap = h >= 12 ? "PM" : "AM";
  h = h % 12 || 12;
  const p = (n, w = 2) => String(n).padStart(w, "0");
  return `${_MON[d.getMonth()]} ${d.getDate()}, ${p(h)}:${p(d.getMinutes())}:${p(d.getSeconds())}.${p(d.getMilliseconds(), 3)} ${ap}`;
}

// Completion tokens per wall-clock second of the (final) provider call — the
// log's throughput. null when either side is missing (a seeded / legacy call).
export function callThroughput(call) {
  if (!call || call.tokens_out == null || !call.flight_ms) return null;
  return call.tokens_out / (call.flight_ms / 1000);
}

// A call's execution stats as a multi-line hover string for the sidebar call
// rows: generation time (provider latency, + retry count), token throughput,
// token counts, and the exact request/response wall-clock times. "" when the
// call carries no timing (a seeded prompt-lab result / legacy log), so callers
// can `|| fallback` another title.
export function callTimingTitle(call) {
  if (!call) return "";
  const lines = [];
  if (call.flight_ms != null) {
    const tries = (call.attempts ?? 1) > 1 ? ` · ${call.attempts} attempts` : "";
    lines.push(`generation time: ${fmtDurationMs(call.flight_ms)}${tries}`);
  }
  const tp = callThroughput(call);
  if (tp != null) lines.push(`throughput: ${tp.toFixed(1)} tok/s`);
  if (call.tokens_in != null || call.tokens_out != null)
    lines.push(`tokens: ${call.tokens_in ?? "?"} in → ${call.tokens_out ?? "?"} out`);
  if (call.t_request != null) lines.push(`requested: ${fmtClockTime(call.t_request)}`);
  if (call.t_response != null) lines.push(`responded: ${fmtClockTime(call.t_response)}`);
  return lines.join("\n");
}
