// A tiny, self-contained markdown → DOM renderer for the investigator chat.
//
// It builds real DOM nodes (never innerHTML), so LLM-authored answers can't
// smuggle markup — the same safety posture the 3D tooltip uses. It covers the
// subset a reasoning model actually emits: headings, paragraphs with soft line
// breaks, fenced + inline code, ordered/unordered (and nested) lists,
// blockquotes, horizontal rules, GFM pipe tables, links, bold, italic, and
// strikethrough.
//
// One deliberate domain tweak: `_`/`__` are NOT treated as emphasis. Scene ids
// in this pipeline are snake_case everywhere (`living_room`, `coffee_table`), so
// underscore emphasis would constantly mangle them — only `*`/`**` italicise.

import { el } from "./ui.js";

export function renderMarkdown(src) {
  const root = el("div", { class: "md" });
  const lines = String(src ?? "").replace(/\r\n?/g, "\n").split("\n");
  for (const node of parseBlocks(lines)) root.appendChild(node);
  return root;
}

// ── block level ──────────────────────────────────────────────────────────────

const HEADING_RE = /^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
const HR_RE = /^ {0,3}([-*_])\s*(?:\1\s*){2,}$/;
const BQ_RE = /^ {0,3}>\s?(.*)$/;
const UL_RE = /^(\s*)([-*+])(\s+)(.*)$/;
const OL_RE = /^(\s*)(\d{1,9})([.)])(\s+)(.*)$/;
const TABLE_SEP_RE = /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$/;

const isBlank = (l) => /^\s*$/.test(l);
const leadingSpaces = (l) => (l.match(/^\s*/)?.[0].length ?? 0);

function fenceOpen(line) {
  const m = line.match(/^(\s*)(`{3,}|~{3,})\s*([\w+-]*)\s*$/);
  return m ? { indent: m[1].length, ch: m[2][0], len: m[2].length, lang: m[3] } : null;
}

function matchItem(line) {
  let m = line.match(UL_RE);
  if (m) return { ordered: false, indent: m[1].length, contentCol: m[1].length + 1 + m[3].length, text: m[4] };
  m = line.match(OL_RE);
  if (m) return { ordered: true, indent: m[1].length, contentCol: m[1].length + m[2].length + 1 + m[4].length, text: m[5] };
  return null;
}

function parseBlocks(lines) {
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (isBlank(line)) { i++; continue; }

    const fence = fenceOpen(line);
    if (fence) { const r = parseFence(lines, i, fence); out.push(r.node); i = r.next; continue; }

    const h = line.match(HEADING_RE);
    if (h) { out.push(heading(h[1].length, h[2])); i++; continue; }

    if (HR_RE.test(line)) { out.push(el("hr", { class: "md-hr" })); i++; continue; }

    if (BQ_RE.test(line)) { const r = parseQuote(lines, i); out.push(r.node); i = r.next; continue; }

    if (line.includes("|") && i + 1 < lines.length && TABLE_SEP_RE.test(lines[i + 1]) && lines[i + 1].includes("|")) {
      const r = parseTable(lines, i);
      if (r) { out.push(r.node); i = r.next; continue; }
    }

    if (matchItem(line)) { const r = parseList(lines, i); out.push(r.node); i = r.next; continue; }

    const r = parseParagraph(lines, i);
    out.push(r.node); i = r.next;
  }
  return out;
}

function heading(level, text) {
  const lv = Math.min(level, 6);
  return el(`h${lv}`, { class: `md-h md-h${lv}` }, ...inlineNodes(text));
}

function parseFence(lines, i, fence) {
  const body = [];
  let j = i + 1;
  for (; j < lines.length; j++) {
    const m = lines[j].match(/^(\s*)(`{3,}|~{3,})\s*$/);
    if (m && m[2][0] === fence.ch && m[2].length >= fence.len) { j++; break; }
    body.push(fence.indent ? lines[j].slice(fence.indent) : lines[j]);
  }
  const code = el("code", { class: "md-code", text: body.join("\n") });
  if (fence.lang) code.dataset.lang = fence.lang;
  return { node: el("pre", { class: "md-pre" }, code), next: j };
}

function parseQuote(lines, i) {
  const inner = [];
  let j = i;
  for (; j < lines.length; j++) {
    const m = lines[j].match(BQ_RE);
    if (!m) break;
    inner.push(m[1]);
  }
  const quote = el("blockquote", { class: "md-quote" });
  for (const node of parseBlocks(inner)) quote.appendChild(node);
  return { node: quote, next: j };
}

// Paragraph: consecutive lines until a blank line or the start of another block.
// Single newlines become soft breaks (<br>) so a model's line structure survives
// in the chat instead of collapsing to one run-on line.
function parseParagraph(lines, i) {
  const buf = [];
  let j = i;
  for (; j < lines.length; j++) {
    const line = lines[j];
    if (isBlank(line)) break;
    if (fenceOpen(line) || HEADING_RE.test(line) || HR_RE.test(line) || BQ_RE.test(line) || matchItem(line)) break;
    if (line.includes("|") && j + 1 < lines.length && TABLE_SEP_RE.test(lines[j + 1])) break;
    buf.push(line);
  }
  const p = el("p", { class: "md-p" });
  buf.forEach((line, k) => {
    if (k) p.appendChild(el("br"));
    for (const n of inlineNodes(line)) p.appendChild(n);
  });
  return { node: p, next: j };
}

// Lists: each item's content (its first-line text + any more-indented
// continuation / nested lines) is dedented and re-parsed as blocks, so nested
// lists, multi-paragraph items, and item-level code blocks all just work.
function parseList(lines, i) {
  const first = matchItem(lines[i]);
  const base = first.indent;
  const ordered = first.ordered;
  const list = el(ordered ? "ol" : "ul", { class: ordered ? "md-ol" : "md-ul" });
  let j = i;
  while (j < lines.length) {
    const line = lines[j];
    if (isBlank(line)) {
      const k = nextNonBlank(lines, j + 1);
      if (k === -1) { j++; break; }
      const nm = matchItem(lines[k]);
      const continues = (nm && nm.indent >= base) || leadingSpaces(lines[k]) > base;
      if (!continues) break;
      j++;
      continue;
    }
    const m = matchItem(line);
    if (!(m && m.indent === base && m.ordered === ordered)) break;
    const itemLines = [line.slice(m.contentCol)];
    j++;
    while (j < lines.length) {
      const l2 = lines[j];
      if (isBlank(l2)) { itemLines.push(""); j++; continue; }
      const m2 = matchItem(l2);
      if (m2 && m2.indent === base) break; // next sibling at this level
      if (leadingSpaces(l2) > base) { itemLines.push(l2.slice(Math.min(m.contentCol, leadingSpaces(l2)))); j++; continue; }
      break; // dedented, non-item → list ends
    }
    while (itemLines.length && itemLines[itemLines.length - 1] === "") itemLines.pop();
    const li = el("li", { class: "md-li" });
    for (const node of parseBlocks(itemLines)) li.appendChild(node);
    // Tight item (a lone paragraph) → unwrap so it doesn't render with block
    // spacing; keep the <p> when an item holds more (e.g. a nested list).
    if (li.childNodes.length === 1 && li.firstChild.classList?.contains("md-p")) {
      const p = li.firstChild;
      li.replaceChildren(...p.childNodes);
    }
    list.appendChild(li);
  }
  return { node: list, next: j };
}

function nextNonBlank(lines, i) {
  for (let j = i; j < lines.length; j++) if (!isBlank(lines[j])) return j;
  return -1;
}

function parseTable(lines, i) {
  const aligns = splitRow(lines[i + 1]).map((c) => {
    const l = c.startsWith(":");
    const r = c.endsWith(":");
    return r && l ? "center" : r ? "right" : l ? "left" : "";
  });
  const header = splitRow(lines[i]);
  if (!header.length) return null;
  const table = el("table", { class: "md-table" });
  const thead = el("thead");
  const htr = el("tr");
  header.forEach((cell, k) => {
    const th = el("th", aligns[k] ? { style: `text-align:${aligns[k]}` } : {});
    for (const n of inlineNodes(cell)) th.appendChild(n);
    htr.appendChild(th);
  });
  thead.appendChild(htr);
  table.appendChild(thead);
  const tbody = el("tbody");
  let j = i + 2;
  for (; j < lines.length; j++) {
    const line = lines[j];
    if (isBlank(line) || !line.includes("|")) break;
    const cells = splitRow(line);
    const tr = el("tr");
    for (let k = 0; k < header.length; k++) {
      const td = el("td", aligns[k] ? { style: `text-align:${aligns[k]}` } : {});
      for (const n of inlineNodes(cells[k] ?? "")) td.appendChild(n);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  return { node: table, next: j };
}

// Split a table row into cells on unescaped pipes, dropping the optional
// leading/trailing border pipes.
function splitRow(line) {
  let s = line.trim();
  s = s.replace(/^\|/, "").replace(/\|$/, "");
  const cells = [];
  let cur = "";
  for (let k = 0; k < s.length; k++) {
    const ch = s[k];
    if (ch === "\\" && s[k + 1] === "|") { cur += "|"; k++; continue; }
    if (ch === "|") { cells.push(cur.trim()); cur = ""; continue; }
    cur += ch;
  }
  cells.push(cur.trim());
  return cells;
}

// ── inline level ─────────────────────────────────────────────────────────────

// Emphasis uses `*`/`**` only (never `_`), with a flanking rule (no space just
// inside the markers) so ids and stray asterisks aren't misread.
const BOLD_RE = /\*\*(\S(?:[\s\S]*?\S)?)\*\*/;
const ITAL_RE = /\*(\S(?:[\s\S]*?\S)?)\*/;
const STRIKE_RE = /~~(\S(?:[\s\S]*?\S)?)~~/;
const LINK_RE = /\[([^\]]*)\]\(\s*(<[^>]*>|[^)\s]+)(?:\s+"[^"]*")?\s*\)/;

// Split off inline code spans first (their content is verbatim), then format
// the rest for links + emphasis.
function inlineNodes(text) {
  const out = [];
  const re = /(`+)([\s\S]*?)\1/g;
  let last = 0;
  let m;
  while ((m = re.exec(text))) {
    if (m.index > last) formatInto(out, text.slice(last, m.index));
    let code = m[2];
    if (code.length > 1 && code.startsWith(" ") && code.endsWith(" ") && code.trim()) code = code.slice(1, -1);
    out.push(el("code", { class: "md-code-inline", text: code }));
    last = re.lastIndex;
  }
  if (last < text.length) formatInto(out, text.slice(last));
  return out;
}

function formatInto(out, text) {
  for (const n of formatInline(text)) out.push(n);
}

function formatInline(text) {
  const out = [];
  let rest = text;
  while (rest) {
    const cands = [];
    let mm;
    if ((mm = LINK_RE.exec(rest))) cands.push({ i: mm.index, len: mm[0].length, make: () => link(mm[1], mm[2]) });
    if ((mm = BOLD_RE.exec(rest))) { const g = mm[1]; cands.push({ i: mm.index, len: mm[0].length, make: () => wrap("strong", "md-strong", g) }); }
    if ((mm = ITAL_RE.exec(rest))) { const g = mm[1]; cands.push({ i: mm.index, len: mm[0].length, make: () => wrap("em", "md-em", g) }); }
    if ((mm = STRIKE_RE.exec(rest))) { const g = mm[1]; cands.push({ i: mm.index, len: mm[0].length, make: () => wrap("s", "md-strike", g) }); }
    if (!cands.length) { out.push(document.createTextNode(unescapeMd(rest))); break; }
    // Earliest wins; on a tie prefer the longer match (so ** beats *).
    cands.sort((a, b) => a.i - b.i || b.len - a.len);
    const c = cands[0];
    if (c.i > 0) out.push(document.createTextNode(unescapeMd(rest.slice(0, c.i))));
    out.push(c.make());
    rest = rest.slice(c.i + c.len);
  }
  return out;
}

function wrap(tag, cls, inner) {
  return el(tag, { class: cls }, ...formatInline(inner));
}

function link(text, rawUrl) {
  const url = rawUrl.replace(/^<|>$/g, "");
  const safe = sanitizeUrl(url);
  const a = el("a", { class: "md-link", ...(safe ? { href: safe, target: "_blank", rel: "noopener noreferrer" } : {}) }, ...formatInline(text || url));
  return a;
}

// Only http(s) / mailto / anchors / relative paths are linkable; anything else
// (javascript:, data:, …) renders as a plain, unclickable span.
function sanitizeUrl(url) {
  const u = url.trim();
  if (/^(https?:|mailto:)/i.test(u)) return u;
  if (/^[#/.]/.test(u)) return u;
  if (/^[a-z][a-z0-9+.-]*:/i.test(u)) return null;
  return u;
}

function unescapeMd(s) {
  return s.replace(/\\([\\`*{}[\]()#+\-.!~>|])/g, "$1");
}
