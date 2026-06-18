// Per-run LLM spend tracker. Every billable call is a `cache.llm` event; the
// server sums each cell's tokens per model into the slots payload (`usage`), and
// we price them here with OpenRouter's published per-token rates — surfacing
// each slot's cost AND the run's accumulative total in a collapsible top-bar
// pill. Live: it re-renders on every slots poll.

import { state, on } from "./state.js";
import { el } from "./ui.js";

// USD per token (prompt / completion), keyed by the OpenRouter model id stamped
// on each cache.llm event. Source: OpenRouter model catalog. A model missing
// here costs 0 but still counts requests, so a newly-added model degrades to a
// request-only row until its rate is filled in.
const MODEL_PRICING = {
  "google/gemini-3.5-flash":       { in: 0.0000015,  out: 0.000009 },
  "google/gemini-3.1-flash-lite":  { in: 0.00000025, out: 0.0000015 },
  "google/gemini-3.1-pro-preview": { in: 0.000002,   out: 0.000012 },
  "openai/gpt-5.5":                { in: 0.000005,    out: 0.00003 },
  "anthropic/claude-opus-4.6":     { in: 0.000005,    out: 0.000025 },
  "anthropic/claude-opus-4.8":     { in: 0.000005,    out: 0.000025 },
  "anthropic/claude-fable-latest": { in: 0.000005,    out: 0.000025 }, // claude-family estimate
  "deepseek/deepseek-v4-pro":      { in: 0.000000435, out: 0.00000087 },
};

let trackerEl = null;
let pillSummaryEl = null;
let dropdownEl = null;

const fmtCost = (v) => {
  v = v || 0;
  return v >= 1 ? `$${v.toFixed(2)}` : `$${v.toFixed(4)}`;
};

// Cost + request count for one cell's per-model usage map ({model: {in,out,req}}).
function cellCost(usage) {
  let cost = 0;
  let req = 0;
  for (const [model, u] of Object.entries(usage || {})) {
    const p = MODEL_PRICING[model];
    if (p) cost += (u.in || 0) * p.in + (u.out || 0) * p.out;
    req += u.req || 0;
  }
  return { cost, req };
}

function costRow(label, count, cost, cls) {
  return el("div", { class: cls },
    el("span", { class: "cost-step", text: label }),
    el("span", { class: "cost-count", text: String(count) }),
    el("span", { class: "cost-amt", text: fmtCost(cost) }),
  );
}

function costSectionHead(name, tag, count, cost) {
  return el("div", { class: "cost-section-head" },
    el("span", { class: "cost-sec-name", text: name }),
    el("span", { class: "cost-sec-model", text: tag }),
    el("span", { class: "cost-sec-sum", text: `${fmtCost(cost)} · ${count} req` }),
  );
}

function renderCost() {
  if (!trackerEl) return;
  // Group cells by slot: each slot is a section (its own total), each model a
  // row beneath it — so you read both the per-slot cost and the run total.
  const bySlot = new Map(); // slotId -> { rows: [{model, cost, req}], cost, req }
  let totalCost = 0;
  let totalReq = 0;
  for (const s of state.slots) {
    for (const [model, c] of Object.entries(s.runs || {})) {
      const { cost, req } = cellCost(c?.usage);
      if (req === 0) continue;
      let g = bySlot.get(s.id);
      if (!g) { g = { rows: [], cost: 0, req: 0 }; bySlot.set(s.id, g); }
      g.rows.push({ model, cost, req });
      g.cost += cost;
      g.req += req;
      totalCost += cost;
      totalReq += req;
    }
  }
  pillSummaryEl.textContent = `${fmtCost(totalCost)} · ${totalReq} req`;
  dropdownEl.textContent = "";
  if (bySlot.size === 0) {
    dropdownEl.appendChild(el("div", { class: "cost-empty", text: "no LLM requests yet on this run" }));
    return;
  }
  for (const [slotId, g] of [...bySlot].sort((a, b) => b[1].cost - a[1].cost)) {
    const section = el("div", { class: "cost-section" },
      costSectionHead(slotId, `${g.rows.length} model${g.rows.length === 1 ? "" : "s"}`, g.req, g.cost));
    for (const r of g.rows.sort((a, b) => b.cost - a.cost)) {
      section.appendChild(costRow(r.model, r.req, r.cost, "cost-row"));
    }
    dropdownEl.appendChild(section);
  }
  dropdownEl.appendChild(el("div", { class: "cost-divider" }));
  dropdownEl.appendChild(costRow("run total", totalReq, totalCost, "cost-total"));
}

export function initCost() {
  trackerEl = document.getElementById("cost-tracker");
  if (!trackerEl) return;
  pillSummaryEl = trackerEl.querySelector(".cost-pill-summary");
  dropdownEl = document.getElementById("cost-dropdown");
  document.getElementById("cost-pill").addEventListener("click", () => {
    trackerEl.classList.toggle("collapsed");
  });
  on("slots", renderCost); // live with the run's status poll
  renderCost();
}
