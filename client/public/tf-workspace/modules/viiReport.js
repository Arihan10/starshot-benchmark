// Standalone HTML report for the VERY_IMPORTANT_INSTRUCTIONS attention view.
// For EACH step kind in the cell it renders: the length-vs-attention scatter (with
// the linear fit, ±1 SE whiskers and green trim rings), the worst-performing
// instructions verbatim (ranked by how many sigma they sit BELOW the trend), and
// the entire VERY_IMPORTANT_INSTRUCTIONS block of that kind's prompt verbatim — so
// the report is a self-contained, offline artifact for tightening the prompts.

import { el, toast } from "../../js/ui.js";
import { state } from "./state.js";
import { viiScatterModel, loadAllRows, stepLLM } from "./data.js";
import { scatterChart, escTip } from "./charts.js";

const viiHeat = (t) => `hsl(${Math.round(210 - Math.max(0, Math.min(1, t)) * 170)}, 70%, 58%)`;

// Pull the <VERY_IMPORTANT_INSTRUCTIONS> block out of a prompt (verbatim, trimmed).
function extractVii(text) {
	if (!text) return "";
	const m = /<VERY_IMPORTANT_INSTRUCTIONS>([\s\S]*?)<\/VERY_IMPORTANT_INSTRUCTIONS>/i.exec(text);
	return m ? m[1].replace(/^\n+|\s+$/g, "") : "";
}
// The full VII text for a kind — from the first of its steps whose logged call we have.
function kindVii(kindRows) {
	for (const r of kindRows) {
		const e = stepLLM(r.event_index);
		if (!e) continue;
		const v = extractVii(e.user) || extractVii(e.system);
		if (v) return v;
	}
	return "";
}

// Render the kind's scatter to a serialized <svg> string. No `tip` → dots keep a
// native <title>, so hover still works in the static report.
function reportScatter(model) {
	const { P, fit, marked, xMaxTok } = model;
	const pts = P.map((p) => ({
		...p, r: 4.5, color: viiHeat(p.y),
		label: `${(p.share * 100).toFixed(3)}% attn · ${p.tokens}t · ${p.z >= 0 ? "+" : ""}${p.z.toFixed(1)}σ${marked.has(p.key) ? " · TRIM" : ""} — ${p.label}`,
	}));
	const g = scatterChart(pts, {
		width: 640, height: 640, yMax: 1, xMax: xMaxTok,
		xLabel: "instruction length (tokens)", xFmt: (v) => String(Math.round(v)), yFmt: (v) => v.toFixed(2),
		line: { points: [{ x: 0, y: fit(0) }, { x: xMaxTok, y: fit(xMaxTok) }], color: "#ff2b2b", width: 2, dash: "5 4" },
		yErr: (p) => p.ey,
		ring: (p) => (marked.has(p.key) ? "#39d98a" : null),
	});
	const svg = g.querySelector && g.querySelector("svg");
	if (!svg) return "";
	svg.setAttribute("xmlns", "http://www.w3.org/2000/svg"); // standalone doc needs the namespace
	return svg.outerHTML;
}

function renderKindSection(kind, kindRows, model, viiText) {
	const nSteps = kindRows.length;
	let meta = `${nSteps} step${nSteps === 1 ? "" : "s"}`;
	let plot = "", side = "";
	if (model) {
		meta += ` · ${model.items.length} instructions · trend ${model.slope >= 0 ? "+" : "−"}${Math.abs(model.slope * 100).toFixed(2)}%/tok · ${model.marked.size} trim pick${model.marked.size === 1 ? "" : "s"}`;
		plot = `<div class="plot">${reportScatter(model)}</div>`;
		const worst = model.ranked.filter((p) => p.z > 0).slice(0, 12);
		side = `<div class="side"><h3>worst-performing instructions <span class="hint">below trend, by σ (distance below ÷ error)</span></h3>`
			+ (worst.length
				? `<ol class="worst">` + worst.map((p) =>
					`<li${model.marked.has(p.key) ? ' class="pick"' : ""}>`
					+ `<div class="wm"><span class="z">${p.z.toFixed(1)}σ below</span><span class="sh">${(p.share * 100).toFixed(3)}% attn</span><span class="tk">${p.tokens} tok</span></div>`
					+ `<div class="wt">${escTip(p.label)}</div></li>`).join("") + `</ol>`
				: `<p class="none">no instruction sits below the trend.</p>`)
			+ `</div>`;
	}
	const vii = `<h3>full VERY_IMPORTANT_INSTRUCTIONS <span class="hint">verbatim</span></h3>`
		+ (viiText ? `<pre class="vii">${escTip(viiText)}</pre>` : `<p class="none">prompt text unavailable (events not loaded for this step).</p>`);
	return `<section><h2>${escTip(kind)}</h2><div class="kmeta">${escTip(meta)}</div>`
		+ `<div class="cols">${plot}${side}</div>${vii}</section>`;
}

const REPORT_CSS = `
:root { --bg:#0d0f14; --panel:#151922; --line:#242b39; --text:#e6edf7; --dim:#9aa7bd; --faint:#6b788f; --accent:#7aa2f7; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 28px 32px 60px; }
header { border-bottom: 1px solid var(--line); padding-bottom: 18px; margin-bottom: 22px; }
h1 { font-size: 21px; margin: 0 0 6px; letter-spacing: 0.2px; }
.scope { font-family: ui-monospace, Menlo, monospace; color: var(--accent); font-size: 13px; }
.when { color: var(--faint); font-size: 12px; margin-top: 3px; }
.intro { color: var(--dim); font-size: 13px; max-width: 1000px; margin: 14px 0 0; }
section { border: 1px solid var(--line); background: var(--panel); border-radius: 12px; padding: 18px 20px; margin: 20px 0; }
h2 { font-size: 17px; margin: 0; font-family: ui-monospace, Menlo, monospace; color: #cfe3ff; }
.kmeta { color: var(--dim); font-size: 12.5px; margin: 4px 0 14px; }
h3 { font-size: 13px; margin: 4px 0 8px; color: var(--text); text-transform: none; }
h3 .hint { color: var(--faint); font-weight: 400; font-size: 11.5px; margin-left: 6px; }
.cols { display: flex; gap: 22px; align-items: flex-start; flex-wrap: wrap; }
.plot { flex: 0 1 auto; min-width: 0; }
.plot svg { display: block; overflow: visible; max-width: 100%; height: auto; }
.side { flex: 1 1 320px; min-width: 280px; }
ol.worst { list-style: none; margin: 0; padding: 0; counter-reset: w; }
ol.worst li { padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px; margin-bottom: 7px; background: #10141c; }
ol.worst li.pick { border-color: #2c6e52; box-shadow: inset 0 0 0 1px rgba(57,217,138,0.25); }
.wm { display: flex; gap: 12px; font-size: 11.5px; font-variant-numeric: tabular-nums; margin-bottom: 4px; }
.wm .z { color: #39d98a; font-weight: 600; }
.wm .sh { color: var(--dim); }
.wm .tk { color: var(--faint); }
.wt { font-size: 12.5px; color: var(--text); line-height: 1.45; }
.none { color: var(--faint); font-size: 12.5px; }
pre.vii { white-space: pre-wrap; word-break: break-word; background: #10141c; border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; font: 12.5px/1.55 ui-monospace, Menlo, monospace; color: var(--dim); overflow-x: auto; margin: 0; }
`;

function htmlDoc(scope, body) {
	return `<!doctype html><html lang="en"><head><meta charset="utf-8">`
		+ `<meta name="viewport" content="width=device-width, initial-scale=1">`
		+ `<title>VII attention report · ${escTip(scope)}</title><style>${REPORT_CSS}</style></head><body>`
		+ `<header><h1>VERY_IMPORTANT_INSTRUCTIONS · attention report</h1>`
		+ `<div class="scope">${escTip(scope)}</div>`
		+ `<div class="when">generated ${escTip(new Date().toLocaleString())}</div>`
		+ `<p class="intro">For each step kind: instructions are plotted by length (x) vs the attention they draw (y, normalized so the most-attended = 1). The dashed line is the linear fit; whiskers are ±1 standard error. Green rings and the "worst-performing" list are the instructions farthest BELOW the trend with the most certainty — scored by σ = (distance below the line) ÷ (its standard error) — i.e. long, reliably-ignored instructions that are the best candidates to tighten or cut. Each kind's full VERY_IMPORTANT_INSTRUCTIONS block is included verbatim.</p></header>`
		+ body + `</body></html>`;
}

// Group all rows by step kind (template), in pipeline order (min event_index).
function groupByKind(rows) {
	const by = new Map();
	for (const r of rows) { const k = r.template || "?"; if (!by.has(k)) by.set(k, []); by.get(k).push(r); }
	const kinds = [...by.keys()].sort((a, b) =>
		Math.min(...by.get(a).map((r) => r.event_index)) - Math.min(...by.get(b).map((r) => r.event_index)));
	return { by, kinds };
}

export function buildViiReport(allRows) {
	const { by, kinds } = groupByKind(allRows);
	const sections = [];
	for (const k of kinds) {
		const kr = by.get(k);
		const model = viiScatterModel(kr);
		const viiText = kindVii(kr);
		if (!model && !viiText) continue; // nothing to show for this kind
		sections.push(renderKindSection(k, kr, model, viiText));
	}
	const scope = [state.run, state.slot, state.model].filter(Boolean).join(" · ");
	if (!sections.length) return htmlDoc(scope, `<p class="none">No VERY_IMPORTANT_INSTRUCTIONS attention found — recompute the VII split (⚗ VII sample) in the /tf inspector first.</p>`);
	return htmlDoc(scope, sections.join("\n"));
}

function reportName() {
	const base = [state.run, state.slot, state.model].filter(Boolean).join("_").replace(/[^a-z0-9_.-]+/gi, "-");
	return `vii-report_${base || "cell"}.html`;
}

// Button handler: load every computed step, build the report, open it (falling
// back to a download if the tab is blocked).
export async function runViiReport(btn) {
	const label = btn.textContent;
	btn.disabled = true; btn.textContent = "generating…";
	try {
		const allRows = await loadAllRows();
		if (!allRows.length) { toast("no computed steps to report", "err"); return; }
		const html = buildViiReport(allRows);
		const url = URL.createObjectURL(new Blob([html], { type: "text/html" }));
		const win = window.open(url, "_blank");
		if (!win) { const a = el("a", { href: url, download: reportName() }); document.body.appendChild(a); a.click(); a.remove(); }
		setTimeout(() => URL.revokeObjectURL(url), 60000);
	} catch (e) {
		toast(`report failed: ${e.message}`, "err");
	} finally {
		btn.disabled = false; btn.textContent = label;
	}
}
