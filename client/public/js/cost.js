// Per-run LLM spend tracker. Every billable call logs a `cache.llm` event with
// its OpenRouter `generation_id`; a server-side backfill sweep prices each
// against OpenRouter's settled `total_cost` (caching discounts and all) and the
// per-model sums land in the slots payload (`usage.cost`) — so this pill matches
// OpenRouter's activity log. Cost settles a beat after each call, so the pill
// also shows how many are still `resolving`; when that hits 0 the run's spend has
// fully caught up (and it's safe to shut the server down without losing any).
// Collapsible top-bar pill, re-rendered on every slots poll.

import { state, on } from "./state.js";
import { el } from "./ui.js";

let trackerEl = null;
let pillSummaryEl = null;
let dropdownEl = null;

const fmtCost = (v) => {
  v = v || 0;
  return v >= 1 ? `$${v.toFixed(2)}` : `$${v.toFixed(4)}`;
};

// Authoritative cost + request + unresolved-lookup counts for one cell's
// per-model usage map ({model: {in, out, req, cost, pending}}). `cost` is
// OpenRouter's settled USD already summed server-side; `pending` is how many of
// this cell's calls the backfill hasn't priced yet. We just total across models.
function cellCost(usage) {
  let cost = 0;
  let req = 0;
  let pending = 0;
  for (const u of Object.values(usage || {})) {
    cost += u.cost || 0;
    req += u.req || 0;
    pending += u.pending || 0;
  }
  return { cost, req, pending };
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
  let totalPending = 0;
  for (const s of state.slots) {
    for (const [model, c] of Object.entries(s.runs || {})) {
      const { cost, req, pending } = cellCost(c?.usage);
      if (req === 0) continue;
      let g = bySlot.get(s.id);
      if (!g) { g = { rows: [], cost: 0, req: 0 }; bySlot.set(s.id, g); }
      g.rows.push({ model, cost, req });
      g.cost += cost;
      g.req += req;
      totalCost += cost;
      totalReq += req;
      totalPending += pending;
    }
  }
  // `resolving` = calls OpenRouter hasn't settled the cost of yet; 0 ⇒ the run's
  // spend has fully caught up (the shutdown-safe signal).
  pillSummaryEl.textContent =
    `${fmtCost(totalCost)} · ${totalReq} req` + (totalPending ? ` · ${totalPending} resolving` : "");
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
  if (totalPending) {
    dropdownEl.appendChild(el("div", { class: "cost-empty", text:
      `${totalPending} call${totalPending === 1 ? "" : "s"} still resolving — total rises as OpenRouter settles them` }));
  }
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
