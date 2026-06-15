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

// Compact "fast-forward to a step" picker: choosing a step fires onPick(step)
// then resets to the placeholder. Used to step (one cell or all) until the
// next run of a target step.
export function stepUntilSelect(steps, onPick, { label = "until…", title } = {}) {
  const sel = el("select", {
    class: "step-until",
    title: title ?? "fast-forward to the next run of a step, then pause there",
  },
    el("option", { value: "", text: label }),
    (steps || []).map((s) => el("option", { value: s, text: `▸ ${s}` })),
  );
  sel.addEventListener("change", () => {
    const v = sel.value;
    sel.value = "";
    if (v) onPick(v);
  });
  return sel;
}

export function fmtJson(value) {
  try { return JSON.stringify(value, null, 2); } catch { return String(value); }
}

// The big dynamic scene-state blocks injected into prompts. In the obs prompt
// view we fold these behind expandable placeholders by default so the prompt's
// own wording is readable without the scene dump.
const CONTEXT_VARS = ["SCENE_CONTEXT", "ROOT_OBJECTS", "TO_PLACE", "RETRY_BLOCK"];
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
export function fitToggle(target, { fitted = true } = {}) {
  const isTextarea = target.tagName === "TEXTAREA";
  let active = false;
  const sync = () => {
    if (!active) return;
    target.style.height = "auto";
    target.style.height = `${target.scrollHeight + 2}px`;
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
