// Global ops bar (bottom-left, every page): a HARD STOP button + a live task
// monitor. Self-injects its own DOM + styles and only depends on api.js, so it
// works identically on the dashboard and /tf. Poll is cheap (server-side
// /generations/active just reads in-memory task tables).

import { api } from "./api.js";
import { getAttnProvider } from "./attnbus.js";

let pollTimer = null;
let expanded = false;

const STYLE = `
#ops-bar { position: fixed; left: 10px; bottom: 10px; z-index: 200; display: flex; align-items: flex-end; gap: 6px; font: 11px ui-monospace, Menlo, monospace; pointer-events: none; opacity: 0.62; transition: opacity 0.15s; }
#ops-bar > * { pointer-events: auto; }
#ops-bar:hover, #ops-bar.busy { opacity: 1; }
#ops-mon { min-width: 0; max-width: 320px; background: rgba(20,22,28,0.92); border: 1px solid #2a2d35; border-radius: 7px; overflow: hidden; }
#ops-mon-head { display: flex; align-items: center; gap: 6px; padding: 4px 8px; cursor: pointer; user-select: none; }
#ops-mon-dot { width: 7px; height: 7px; border-radius: 50%; background: #5a5e68; flex-shrink: 0; }
#ops-bar.busy #ops-mon-dot { background: #f0b450; }
#ops-mon-caret { color: #6a7080; font-size: 9px; }
#ops-mon-title { color: #8a8f99; display: none; }
#ops-bar.open #ops-mon-title { display: inline; }
#ops-mon-count { color: #8a8f99; }
#ops-mon-body { max-height: 240px; overflow: auto; border-top: 1px solid #2a2d35; padding: 4px 0; display: none; }
#ops-bar.open #ops-mon-body { display: block; min-width: 210px; }
.ops-row { display: flex; gap: 8px; padding: 3px 9px; white-space: nowrap; }
.ops-row .r-main { color: #b8bcc4; overflow: hidden; text-overflow: ellipsis; }
.ops-row .r-step { color: #7a808c; margin-left: auto; }
.ops-empty { color: #7a808c; padding: 5px 9px; }
.ops-seg { color: #6a7080; font-size: 9px; text-transform: uppercase; letter-spacing: 0.05em; padding: 5px 9px 2px; }
.ops-row + .ops-seg, .ops-empty + .ops-seg { border-top: 1px solid #23262e; margin-top: 3px; }
#ops-stop { background: rgba(58,22,32,0.92); color: #ff8080; border: 1px solid #5a2230; border-radius: 7px; padding: 4px 9px; font: 600 11px ui-monospace, Menlo, monospace; cursor: pointer; }
#ops-stop:hover { background: #4a1a28; border-color: #ff8080; }
#ops-stop:disabled { opacity: 0.6; cursor: default; }
#ops-sync { background: rgba(22,32,52,0.92); color: #7aa2f7; border: 1px solid #2a3a5a; border-radius: 7px; padding: 4px 9px; font: 600 11px ui-monospace, Menlo, monospace; cursor: pointer; }
#ops-sync:hover { background: #1a2740; border-color: #7aa2f7; }
#ops-sync:disabled { opacity: 0.6; cursor: default; }
`;

function rowEl(main, step) {
	const r = document.createElement("div");
	r.className = "ops-row";
	const m = document.createElement("span");
	m.className = "r-main";
	m.textContent = main;
	m.title = main;
	const s = document.createElement("span");
	s.className = "r-step";
	s.textContent = step;
	r.append(m, s);
	return r;
}
function rowEmpty(t) {
	const d = document.createElement("div");
	d.className = "ops-empty";
	d.textContent = t;
	return d;
}
function segHeader(t) {
	const d = document.createElement("div");
	d.className = "ops-seg";
	d.textContent = t;
	return d;
}
// One attention-queue line (main sequence / ablation): running + queued while
// busy, else the computed / total progress.
function attnRow(label, q) {
	q = q || { running: 0, queued: 0, computed: 0, total: 0 };
	const busy = (q.running || 0) + (q.queued || 0);
	const right = busy
		? `${q.running || 0} running${q.queued ? ` · ${q.queued} queued` : ""}`
		: `${q.computed || 0}/${q.total || 0} ✓`;
	return rowEl(label, right);
}

// `a` = /generations/active (OpenRouter/pipeline tasks); `attn` = the /tf attention
// provider snapshot (null off /tf). The list is SEGMENTED so the Modal attention
// compute reads separately from the OpenRouter pipeline work.
function render(a, attn) {
	const bar = document.getElementById("ops-bar");
	if (!bar) return;
	const count = document.getElementById("ops-mon-count");
	const body = document.getElementById("ops-mon-body");
	const syncB = document.getElementById("ops-sync");
	// The attention ⟳ sync only shows where a provider was published (i.e. /tf).
	if (syncB) syncB.style.display = attn ? "" : "none";
	const segmented = !!attn; // headers only make sense when there are ≥2 segments

	const rows = [];
	let orActive = 0;
	if (!a) {
		if (segmented) rows.push(segHeader("openrouter · pipeline"));
		rows.push(rowEmpty("server unreachable"));
	} else {
		const pipes = a.pipelines || [];
		orActive = pipes.length + (a.generates?.length || 0) + (a.regens?.length || 0) + (a.retries || 0);
		if (segmented) rows.push(segHeader("openrouter · pipeline"));
		for (const p of pipes.slice(0, 20)) rows.push(rowEl(`${p.run} · ${p.slot}/${p.model}`, p.step || ""));
		if (pipes.length > 20) rows.push(rowEmpty(`+${pipes.length - 20} more pipelines`));
		if (a.generates?.length) rows.push(rowEl("scene builds", String(a.generates.length)));
		if (a.regens?.length) rows.push(rowEl("regens", String(a.regens.length)));
		if (a.retries) rows.push(rowEl("mesh retries", String(a.retries)));
		if (a.mesh_queue) rows.push(rowEl("mesh queue", String(a.mesh_queue)));
		if (!orActive && !a.mesh_queue) rows.push(rowEmpty("nothing running"));
	}

	let attnActive = 0;
	if (attn) {
		attnActive = (attn.running || 0) + (attn.queued || 0);
		rows.push(segHeader(`modal · attention${attn.cell ? ` · ${attn.cell}` : ""}`));
		rows.push(attnRow("main sequence", attn.main));
		if ((attn.abl?.total || 0) > 0) rows.push(attnRow("ablation", attn.abl));
	}
	body.replaceChildren(...rows);

	const active = orActive + attnActive;
	bar.classList.toggle("busy", active > 0);
	count.textContent = a === null && !attn ? "—" : active ? `${active} active` : "idle";
}

async function poll() {
	if (!document.getElementById("ops-bar")) return;
	let a = null;
	try { a = await api.activeGenerations(); } catch { /* server down / endpoint missing */ }
	let attn = null;
	try { attn = getAttnProvider()?.snapshot?.() ?? null; } catch { /* provider not ready */ }
	render(a, attn);
}

async function stopAll() {
	const btn = document.getElementById("ops-stop");
	if (!btn) return;
	btn.disabled = true;
	btn.textContent = "stopping…";
	try {
		const r = await api.stopAllGenerations();
		const n = (r.stopped_pipelines?.length || 0) + (r.stopped_generates?.length || 0) + (r.stopped_regens?.length || 0) + (r.stopped_retries || 0);
		btn.textContent = `stopped ${n}`;
	} catch {
		btn.textContent = "stop failed";
	}
	setTimeout(() => { btn.textContent = "⏹ stop"; btn.disabled = false; }, 1600);
	poll();
}

// Re-sync the Modal attention-compute queue for the current /tf cell (main sequence
// + ablation) via the published provider — pulls finished results + surfaces jobs
// running elsewhere, then resumes polling. Replaces the per-panel /tf sync buttons.
async function syncAttn() {
	const p = getAttnProvider();
	const btn = document.getElementById("ops-sync");
	if (!p || !btn) return;
	btn.disabled = true;
	const prev = btn.textContent;
	btn.textContent = "⟳ syncing…";
	try { await p.sync(); } catch { /* best-effort — Modal unreachable */ }
	btn.disabled = false;
	btn.textContent = prev;
	poll();
}

function build() {
	if (document.getElementById("ops-bar")) return;
	const style = document.createElement("style");
	style.textContent = STYLE;
	document.head.appendChild(style);
	const bar = document.createElement("div");
	bar.id = "ops-bar";
	bar.innerHTML = `
		<div id="ops-mon">
			<div id="ops-mon-head">
				<span id="ops-mon-dot"></span>
				<span id="ops-mon-caret">▸</span>
				<span id="ops-mon-title">tasks</span>
				<span id="ops-mon-count">·</span>
			</div>
			<div id="ops-mon-body"></div>
		</div>
		<button id="ops-sync" title="Re-sync the Modal attention-compute queue for the current /tf cell (main sequence + ablation) — pulls finished results + surfaces jobs running elsewhere, then resumes polling. Dispatches no new compute." style="display:none">⟳ sync attn</button>
		<button id="ops-stop" title="Hard-stop every in-flight generation across all runs, cells, and meshes">⏹ stop</button>`;
	document.body.appendChild(bar);
	document.getElementById("ops-mon-head").addEventListener("click", () => {
		expanded = !expanded;
		bar.classList.toggle("open", expanded);
		document.getElementById("ops-mon-caret").textContent = expanded ? "▾" : "▸";
	});
	document.getElementById("ops-stop").addEventListener("click", stopAll);
	document.getElementById("ops-sync").addEventListener("click", syncAttn);
	poll();
	clearInterval(pollTimer);
	pollTimer = setInterval(poll, 3000);
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", build);
else build();
