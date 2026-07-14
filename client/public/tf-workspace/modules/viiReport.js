// Standalone, INTERACTIVE HTML report for the VERY_IMPORTANT_INSTRUCTIONS
// attention view. For EACH step kind in the cell it renders (client-side, from an
// embedded JSON payload):
//   • a length-vs-attention scatter whose dots are HOVERABLE (read the full
//     instruction) and CLICKABLE (pin it across every view), with the linear fit,
//     ±1 SE whiskers and green trim rings;
//   • a right panel toggled between the Δ (attention − fit) BAR CHART (x-axis
//     sortable several ways), the ★ most-attended ranking, and the σ-trim list;
//   • the entire VERY_IMPORTANT_INSTRUCTIONS block of that kind's prompt verbatim.
// The whole thing is self-contained (data + a serialized runtime), so it stays a
// portable, offline artifact for tightening the prompts.

import { el, toast } from "../../js/ui.js";
import { state } from "./state.js";
import { viiScatterModel, loadAllRows, stepLLM, ensureEvents } from "./data.js";
import { escTip } from "./charts.js";

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

// The per-kind data the embedded runtime charts from (kept small + JSON-safe).
function sectionPayload(kind, nSteps, model) {
	return {
		kind, nSteps, slope: model.slope, intercept: model.intercept, xMaxTok: model.xMaxTok,
		marked: [...model.marked],
		P: model.P.map((p) => ({ key: p.key, label: p.label, x: p.x, y: p.y, ey: p.ey, share: p.share, resid: p.resid, z: p.z, tokens: p.tokens })),
	};
}

// The static shell for one kind: heading + meta + the (empty) interactive host the
// runtime mounts into (data-idx → payload index) + the verbatim VII block.
function sectionShell(idx, kind, meta, viiText) {
	const viz = idx >= 0 ? `<div class="viz" data-idx="${idx}"></div>` : `<p class="none">no split instructions with attention for this kind.</p>`;
	const vii = `<h3>full VERY_IMPORTANT_INSTRUCTIONS <span class="hint">verbatim</span></h3>`
		+ (viiText ? `<pre class="vii">${escTip(viiText)}</pre>` : `<p class="none">prompt text unavailable (events not loaded for this step).</p>`);
	return `<section><h2>${escTip(kind)}</h2><div class="kmeta">${escTip(meta)}</div>${viz}${vii}</section>`;
}

// ---------------------------------------------------------------------------
// The embedded client runtime. Self-contained (references ONLY browser globals +
// window.__VII__), serialized into the report via .toString() and IIFE-invoked —
// so it needs no module imports and runs from a blob: URL. It mirrors the live
// VII card: hoverable/clickable scatter + a toggled right panel (Δ bar chart /
// attended / trim), all colored by attention (blue→red).
// ---------------------------------------------------------------------------
function viiReportRuntime() {
	const NS = "http://www.w3.org/2000/svg";
	const DATA = window.__VII__ || [];
	const S = (tag, a, t) => { const n = document.createElementNS(NS, tag); if (a) for (const k in a) { if (a[k] == null) continue; n.setAttribute(k, a[k]); } if (t != null) n.appendChild(document.createTextNode(String(t))); return n; };
	const E = (tag, a, ...kids) => { const n = document.createElement(tag); if (a) for (const k in a) { if (k === "class") n.className = a[k]; else if (k === "text") n.textContent = a[k]; else if (a[k] != null) n.setAttribute(k, a[k]); } for (const c of kids) { if (c != null) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c); } return n; };
	const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
	const heat = (t) => `hsl(${Math.round(210 - Math.max(0, Math.min(1, t)) * 170)}, 70%, 58%)`;
	const niceMax = (v) => { if (!(v > 0)) return 1; const p = Math.pow(10, Math.floor(Math.log10(v))); const n = v / p; const s = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10; return s * p; };

	const tip = E("div", { class: "rtip" });
	document.body.appendChild(tip);
	const showTip = (html, cx, cy) => {
		tip.innerHTML = html; tip.style.opacity = "1";
		const tw = tip.offsetWidth || 220, th = tip.offsetHeight || 72;
		let l = cx + 16; if (l + tw > window.innerWidth) l = cx - tw - 16;
		let t = cy + 16; if (t + th > window.innerHeight) t = cy - th - 16;
		tip.style.left = Math.max(4, l) + "px"; tip.style.top = Math.max(4, t) + "px";
	};
	const hideTip = () => { tip.style.opacity = "0"; };
	const tipHtml = (p, m) => {
		const up = p.resid >= 0;
		return `<div class="xh">${(p.share * 100).toFixed(3)}% attn · ${p.tokens} tok · ${p.z >= 0 ? "+" : ""}${p.z.toFixed(1)}σ</div>`
			+ `<div class="tt">Δ ${up ? "+" : "−"}${Math.abs(p.resid * 100).toFixed(1)}% vs length-fit${m.marked.indexOf(p.key) >= 0 ? " · ★ trim candidate" : ""}</div>`
			+ `<div class="ts">${esc(p.label)}</div>`;
	};

	const redraws = [];
	let rz = null;
	window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => { for (const f of redraws) f(); }, 140); });

	DATA.forEach((m, i) => { const host = document.querySelector('.viz[data-idx="' + i + '"]'); if (host) mount(host, m); });

	function mount(host, m) {
		let mode = "delta", sortKey = "delta", desc = true, sel = null;
		const scatterHost = E("div", { class: "vz-scatter" });
		const right = E("div", { class: "vz-right" });
		host.appendChild(E("div", { class: "vz-grid" }, E("div", { class: "vz-left" }, scatterHost, legend()), right));
		const onPick = (key) => { sel = (sel === key ? null : key); drawScatter(scatterHost, m, { sel, onPick }); renderRight(); };
		const drawS = () => drawScatter(scatterHost, m, { sel, onPick });
		function renderRight() {
			right.replaceChildren();
			const seg = E("div", { class: "seg" });
			[["delta", "Δ chart"], ["attended", "★ attended"], ["trim", "σ trim"]].forEach(([id, lab]) => {
				const b = E("button", { class: "segb" + (mode === id ? " on" : ""), text: lab });
				b.onclick = () => { mode = id; renderRight(); };
				seg.appendChild(b);
			});
			right.appendChild(seg);
			if (mode === "delta") {
				const tools = E("div", { class: "vz-tools" });
				const ss = E("select", { class: "vz-sort", title: "sort the x-axis" });
				[["delta", "sort: Δ attn−fit"], ["attention", "sort: attention"], ["length", "sort: length"]].forEach(([v, t]) => { const o = E("option", { value: v, text: t }); if (v === sortKey) o.selected = true; ss.appendChild(o); });
				ss.onchange = (e) => { sortKey = e.target.value; renderRight(); };
				const db = E("button", { class: "vz-dir", title: "sort direction", text: desc ? "↓ high" : "↑ low" });
				db.onclick = () => { desc = !desc; renderRight(); };
				tools.append(ss, db);
				right.appendChild(tools);
				const bh = E("div", { class: "vz-bar" });
				right.appendChild(bh);
				drawBar(bh, m, { sortKey, desc, sel, onPick });
			} else {
				const arr = mode === "attended" ? [...m.P].sort((a, b) => b.share - a.share) : [...m.P].sort((a, b) => b.z - a.z);
				const note = mode === "attended" ? "instructions the model attends to MOST" : "farthest BELOW the trend (σ = distance ÷ error) — the trim candidates";
				right.appendChild(E("div", { class: "vz-note", text: note }));
				const list = E("div", { class: "vz-list" });
				arr.forEach((p) => list.appendChild(row(p, m, mode, sel, onPick)));
				right.appendChild(list);
			}
		}
		redraws.push(() => { drawS(); if (mode === "delta") renderRight(); });
		drawS(); renderRight();
	}

	function legend() {
		return E("div", { class: "leg" },
			E("span", { text: "attention" }),
			E("span", { class: "ramp" }),
			E("span", { text: "low → high" }),
			E("span", { class: "leg-ring", text: "◯ trim pick" }),
			E("span", { class: "leg-hint", text: "hover a point to read it · click to pin" }));
	}

	function row(p, m, mode, sel, onPick) {
		const up = p.resid >= 0;
		const r = E("button", { class: "vz-row" + (p.key === sel ? " on" : "") + (m.marked.indexOf(p.key) >= 0 ? " pick" : ""), style: `box-shadow:inset 3px 0 0 ${heat(p.y)}` });
		const meta = E("div", { class: "vz-m" });
		if (mode === "trim") meta.appendChild(E("span", { class: "z", text: `${p.z.toFixed(1)}σ below` }));
		meta.appendChild(E("span", { text: `${(p.share * 100).toFixed(3)}% attn` }));
		meta.appendChild(E("span", { text: `${p.tokens} tok` }));
		meta.appendChild(E("span", { class: up ? "up" : "dn", text: `Δ ${up ? "+" : "−"}${Math.abs(p.resid * 100).toFixed(1)}%` }));
		r.append(meta, E("div", { class: "vz-t", text: p.label }));
		r.onclick = () => onPick(p.key);
		r.onpointerenter = (ev) => showTip(tipHtml(p, m), ev.clientX, ev.clientY);
		r.onpointermove = (ev) => showTip(tipHtml(p, m), ev.clientX, ev.clientY);
		r.onpointerleave = hideTip;
		return r;
	}

	function drawScatter(host, m, ui) {
		const W = Math.max(240, Math.floor(host.clientWidth || 600)), H = Math.min(W, 540);
		const fs = Math.max(1, Math.min(1.5, W / 560));
		const padL = Math.round(44 + 8 * fs), padR = 14, padT = 12, padB = Math.round(24 + 12 * fs);
		const px0 = padL, px1 = W - padR, py0 = padT, py1 = H - padB;
		const xMax = m.xMaxTok || 1;
		const X = (x) => px0 + (x / xMax) * (px1 - px0);
		const Y = (y) => py1 - Math.max(0, Math.min(1, y)) * (py1 - py0);
		const s = S("svg", { class: "gsvg", viewBox: `0 0 ${W} ${H}`, width: W, height: H });
		for (const f of [0, .25, .5, .75, 1]) {
			const yy = py1 - f * (py1 - py0);
			s.appendChild(S("line", { x1: px0, y1: yy, x2: px1, y2: yy, stroke: "rgba(255,255,255,0.07)" }));
			s.appendChild(S("text", { x: px0 - 6, y: yy, fill: "rgba(220,230,245,0.62)", "font-size": 11 * fs, "text-anchor": "end", "dominant-baseline": "middle" }, f.toFixed(2)));
		}
		for (const f of [0, .5, 1]) { const xv = xMax * f; s.appendChild(S("text", { x: X(xv), y: py1 + 14 * fs, fill: "rgba(220,230,245,0.62)", "font-size": 11 * fs, "text-anchor": "middle" }, Math.round(xv))); }
		s.appendChild(S("text", { x: px1, y: py1 + 26 * fs, fill: "rgba(220,230,245,0.82)", "font-size": 12 * fs, "text-anchor": "end" }, "instruction length (tokens)"));
		const midY = py0 + (py1 - py0) / 2, lx = px0 - Math.round(36 * fs);
		s.appendChild(S("text", { x: lx, y: midY, fill: "rgba(220,230,245,0.7)", "font-size": 11.5 * fs, "text-anchor": "middle", transform: `rotate(-90 ${lx} ${midY})` }, "attention (norm.)"));
		s.appendChild(S("rect", { x: px0, y: py0, width: Math.max(0, px1 - px0), height: Math.max(0, py1 - py0), fill: "none", style: "pointer-events:all;cursor:crosshair" }));
		const f0 = m.intercept, f1 = m.intercept + m.slope * xMax;
		s.appendChild(S("line", { x1: X(0), y1: Y(f0), x2: X(xMax), y2: Y(f1), stroke: "rgba(8,10,16,0.78)", "stroke-width": 4, "stroke-linecap": "round" }));
		s.appendChild(S("line", { x1: X(0), y1: Y(f0), x2: X(xMax), y2: Y(f1), stroke: "#ff2b2b", "stroke-width": 2, "stroke-dasharray": "5 4", "stroke-linecap": "round" }));
		for (const p of m.P) { if (p.ey > 0) { const cx = X(p.x); s.appendChild(S("line", { x1: cx, y1: Y(Math.min(1, p.y + p.ey)), x2: cx, y2: Y(Math.max(0, p.y - p.ey)), stroke: heat(p.y), "stroke-width": 1.1, "stroke-opacity": .38 })); } }
		for (const p of m.P) { if (m.marked.indexOf(p.key) >= 0) s.appendChild(S("circle", { cx: X(p.x), cy: Y(p.y), r: 8, fill: "none", stroke: "#39d98a", "stroke-width": 2, "stroke-opacity": .9 })); }
		const dots = [];
		for (const p of m.P) {
			const cx = X(p.x), cy = Y(p.y), on = p.key === ui.sel;
			const d = S("circle", { cx, cy, r: on ? 6.5 : 4.5, fill: heat(p.y), "fill-opacity": on ? 1 : 0.85, stroke: on ? "#fff" : "none", "stroke-width": on ? 1.8 : 0, style: "cursor:pointer" });
			dots.push({ p, d, cx, cy, on }); s.appendChild(d);
		}
		host.replaceChildren(s);
		let cur = null;
		const reset = () => { if (cur) { cur.d.setAttribute("r", cur.on ? 6.5 : 4.5); cur.d.setAttribute("stroke", cur.on ? "#fff" : "none"); cur.d.setAttribute("stroke-width", cur.on ? 1.8 : 0); cur = null; } };
		const pick = (ev) => {
			const rect = s.getBoundingClientRect(), sc = W / (rect.width || W);
			const vx = (ev.clientX - rect.left) * sc, vy = (ev.clientY - rect.top) * sc;
			let best = null, bd = Infinity;
			for (const o of dots) { const dd = (o.cx - vx) * (o.cx - vx) + (o.cy - vy) * (o.cy - vy); if (dd < bd) { bd = dd; best = o; } }
			return best && bd <= (26 * sc) * (26 * sc) ? best : null;
		};
		s.addEventListener("pointermove", (ev) => {
			const b = pick(ev);
			if (!b) { reset(); hideTip(); return; }
			if (b !== cur) { reset(); b.d.setAttribute("r", 7.5); b.d.setAttribute("stroke", "#fff"); b.d.setAttribute("stroke-width", 2); cur = b; }
			showTip(tipHtml(b.p, m), ev.clientX, ev.clientY);
		});
		s.addEventListener("pointerleave", () => { reset(); hideTip(); });
		s.addEventListener("click", (ev) => { const b = pick(ev); if (b) ui.onPick(b.p.key); });
	}

	function drawBar(host, m, ui) {
		const W = Math.max(240, Math.floor(host.clientWidth || 360)), H = 300;
		const fs = Math.max(1, Math.min(1.4, W / 420));
		const padL = Math.round(42 + 8 * fs), padR = 12, padT = 14, padB = 16;
		const px0 = padL, px1 = W - padR, py0 = padT, py1 = H - padB;
		const d = ui.desc ? -1 : 1;
		const P = [...m.P].sort((a, b) => ui.sortKey === "attention" ? (a.share - b.share) * d : ui.sortKey === "length" ? (a.tokens - b.tokens) * d : (a.resid - b.resid) * d);
		const vals = P.map((p) => p.resid);
		const maxV = Math.max(...vals, 0), minV = Math.min(...vals, 0);
		const yTop = maxV > 0 ? niceMax(maxV) : 0, yBot = minV < 0 ? -niceMax(-minV) : 0;
		const span = (yTop - yBot) || 1;
		const Y = (v) => py1 - ((v - yBot) / span) * (py1 - py0);
		const s = S("svg", { class: "gsvg", viewBox: `0 0 ${W} ${H}`, width: W, height: H });
		const ticks = [...new Set([yBot, yBot / 2, 0, yTop / 2, yTop])].filter((t) => t >= yBot - 1e-9 && t <= yTop + 1e-9);
		for (const t of ticks) { const yy = Y(t); s.appendChild(S("line", { x1: px0, y1: yy, x2: px1, y2: yy, stroke: Math.abs(t) < 1e-12 ? "rgba(255,255,255,0.28)" : "rgba(255,255,255,0.07)" })); s.appendChild(S("text", { x: px0 - 6, y: yy, fill: "rgba(220,230,245,0.62)", "font-size": 11 * fs, "text-anchor": "end", "dominant-baseline": "middle" }, `${(t * 100).toFixed(0)}%`)); }
		s.appendChild(S("text", { x: px0, y: py0 - 3, fill: "rgba(220,230,245,0.8)", "font-size": 11 * fs, "text-anchor": "start" }, "attention − fit"));
		const n = P.length, bw = (px1 - px0) / (n || 1), gap = Math.min(3, bw * 0.2), zeroY = Y(0);
		const bars = [];
		P.forEach((p, i) => {
			const x = px0 + i * bw + gap / 2, w = Math.max(1, bw - gap), yv = Y(p.resid);
			const top = Math.min(zeroY, yv), h = Math.max(1, Math.abs(yv - zeroY)), on = p.key === ui.sel;
			const rect = S("rect", { x: x.toFixed(2), y: top.toFixed(2), width: w.toFixed(2), height: h.toFixed(2), fill: heat(p.y), "fill-opacity": on ? 1 : 0.85, rx: Math.min(2, w / 3).toFixed(1), stroke: on ? "#fff" : "none", "stroke-width": on ? 1.4 : 0, style: "cursor:pointer" });
			bars.push({ p, rect, cx: x + w / 2, on }); s.appendChild(rect);
		});
		host.replaceChildren(s);
		let cur = null;
		const reset = () => { if (cur) { cur.rect.setAttribute("stroke", cur.on ? "#fff" : "none"); cur.rect.setAttribute("stroke-width", cur.on ? 1.4 : 0); cur = null; } };
		const pick = (ev) => { const rect = s.getBoundingClientRect(), sc = W / (rect.width || W), vx = (ev.clientX - rect.left) * sc; if (vx < px0 - 2 || vx > px1 + 2) return null; let best = null, bd = Infinity; for (const o of bars) { const dd = Math.abs(o.cx - vx); if (dd < bd) { bd = dd; best = o; } } return best; };
		s.addEventListener("pointermove", (ev) => { const b = pick(ev); if (!b) { reset(); hideTip(); return; } if (b !== cur) { reset(); b.rect.setAttribute("stroke", "#fff"); b.rect.setAttribute("stroke-width", 1.6); cur = b; } showTip(tipHtml(b.p, m), ev.clientX, ev.clientY); });
		s.addEventListener("pointerleave", () => { reset(); hideTip(); });
		s.addEventListener("click", (ev) => { const b = pick(ev); if (b) ui.onPick(b.p.key); });
	}
}

const REPORT_CSS = `
:root { --bg:#0d0f14; --panel:#151922; --line:#242b39; --text:#e6edf7; --dim:#9aa7bd; --faint:#6b788f; --accent:#7aa2f7; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 28px 32px 60px; }
header { border-bottom: 1px solid var(--line); padding-bottom: 18px; margin-bottom: 22px; }
h1 { font-size: 21px; margin: 0 0 6px; letter-spacing: 0.2px; }
.scope { font-family: ui-monospace, Menlo, monospace; color: var(--accent); font-size: 13px; }
.when { color: var(--faint); font-size: 12px; margin-top: 3px; }
.intro { color: var(--dim); font-size: 13px; max-width: 1040px; margin: 14px 0 0; }
section { border: 1px solid var(--line); background: var(--panel); border-radius: 12px; padding: 18px 20px; margin: 20px 0; }
h2 { font-size: 17px; margin: 0; font-family: ui-monospace, Menlo, monospace; color: #cfe3ff; }
.kmeta { color: var(--dim); font-size: 12.5px; margin: 4px 0 14px; }
h3 { font-size: 13px; margin: 16px 0 8px; color: var(--text); text-transform: none; }
h3 .hint { color: var(--faint); font-weight: 400; font-size: 11.5px; margin-left: 6px; }
.none { color: var(--faint); font-size: 12.5px; }
pre.vii { white-space: pre-wrap; word-break: break-word; background: #10141c; border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; font: 12.5px/1.55 ui-monospace, Menlo, monospace; color: var(--dim); overflow-x: auto; margin: 0; }

/* interactive viz */
.viz { margin: 6px 0 4px; }
.vz-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(300px, 40%); gap: 22px; align-items: start; }
@media (max-width: 900px) { .vz-grid { grid-template-columns: 1fr; } }
.vz-left { min-width: 0; }
.vz-scatter, .vz-bar { width: 100%; min-width: 0; }
.gsvg { display: block; overflow: visible; max-width: 100%; }
.vz-right { min-width: 0; }
.leg { display: flex; gap: 10px; align-items: center; color: var(--faint); font-size: 11.5px; margin-top: 8px; flex-wrap: wrap; }
.leg .ramp { width: 120px; height: 10px; border-radius: 3px; background: linear-gradient(90deg, hsl(210,70%,58%), hsl(125,70%,58%), hsl(40,70%,58%)); }
.leg .leg-ring { color: #39d98a; }
.leg .leg-hint { margin-left: auto; font-style: italic; }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; margin-bottom: 10px; }
.segb { background: transparent; border: 0; color: var(--dim); padding: 5px 12px; font: inherit; font-size: 12px; cursor: pointer; border-right: 1px solid var(--line); }
.segb:last-child { border-right: 0; }
.segb.on { background: var(--accent); color: #0b1020; font-weight: 600; }
.vz-tools { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }
.vz-sort, .vz-dir { font-size: 12px; padding: 3px 8px; border-radius: 6px; background: #10141c; border: 1px solid var(--line); color: var(--text); cursor: pointer; }
.vz-note { color: var(--faint); font-size: 11.5px; margin: 2px 0 8px; }
.vz-list { list-style: none; margin: 0; padding: 0; max-height: 560px; overflow: auto; display: flex; flex-direction: column; gap: 6px; }
.vz-row { display: block; text-align: left; width: 100%; background: #10141c; border: 1px solid var(--line); border-radius: 8px; padding: 7px 9px; color: var(--text); font: inherit; cursor: pointer; }
.vz-row:hover { background: #17243b; }
.vz-row.on { border-color: var(--accent); background: #17243b; }
.vz-row.pick { box-shadow: inset 0 0 0 1px rgba(57,217,138,0.25) !important; }
.vz-m { display: flex; gap: 10px; font-size: 11.5px; font-variant-numeric: tabular-nums; margin-bottom: 3px; color: var(--dim); flex-wrap: wrap; }
.vz-m .z { color: #39d98a; font-weight: 600; }
.vz-m .up { color: #39d98a; }
.vz-m .dn { color: #ff8f8f; }
.vz-t { font-size: 12.5px; color: var(--text); line-height: 1.45; }
.rtip { position: fixed; z-index: 50; pointer-events: none; background: rgba(12,15,22,0.97); border: 1px solid var(--line); border-radius: 8px; padding: 7px 9px; font-size: 12px; max-width: 360px; opacity: 0; transition: opacity .08s; box-shadow: 0 10px 28px rgba(0,0,0,0.55); }
.rtip .xh { font-weight: 600; color: #cfe3ff; font-variant-numeric: tabular-nums; margin-bottom: 3px; }
.rtip .tt { color: var(--dim); margin-bottom: 4px; }
.rtip .ts { color: var(--text); line-height: 1.42; }
`;

function htmlDoc(scope, body, scriptJs) {
	return `<!doctype html><html lang="en"><head><meta charset="utf-8">`
		+ `<meta name="viewport" content="width=device-width, initial-scale=1">`
		+ `<title>VII attention report · ${escTip(scope)}</title><style>${REPORT_CSS}</style></head><body>`
		+ `<header><h1>VERY_IMPORTANT_INSTRUCTIONS · attention report</h1>`
		+ `<div class="scope">${escTip(scope)}</div>`
		+ `<div class="when">generated ${escTip(new Date().toLocaleString())}</div>`
		+ `<p class="intro">For each step kind, instructions are plotted by length (x) vs the attention they draw (y, normalized so the most-attended = 1). <b>Hover any point to read its full instruction; click to pin it across every view.</b> The dashed red line is the linear length-fit; whiskers are ±1 standard error; green rings mark the σ-trim candidates (farthest BELOW the trend, scored by σ = distance ÷ error — long, reliably-ignored instructions worth cutting). Toggle the right panel between the <b>Δ chart</b> (attention − fit per instruction, x-axis sortable by Δ, attention or length), the <b>★ attended</b> ranking, and the <b>σ trim</b> list. Instructions are grouped by a variable-masked template, so the same instruction merges across steps even when its ids / coordinates differ. Each kind's full VERY_IMPORTANT_INSTRUCTIONS block is included verbatim.</p></header>`
		+ body
		+ (scriptJs ? `<script>${scriptJs}</script>` : "")
		+ `</body></html>`;
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
	const sections = [], payload = [];
	for (const k of kinds) {
		const kr = by.get(k);
		const model = viiScatterModel(kr);
		const viiText = kindVii(kr);
		if (!model && !viiText) continue; // nothing to show for this kind
		let meta = `${kr.length} step${kr.length === 1 ? "" : "s"}`, idx = -1;
		if (model) {
			meta += ` · ${model.items.length} instructions · trend ${model.slope >= 0 ? "+" : "−"}${Math.abs(model.slope * 100).toFixed(2)}%/tok · ${model.marked.size} trim pick${model.marked.size === 1 ? "" : "s"}`;
			idx = payload.length;
			payload.push(sectionPayload(k, kr.length, model));
		}
		sections.push(sectionShell(idx, k, meta, viiText));
	}
	const scope = [state.run, state.slot, state.model].filter(Boolean).join(" · ");
	if (!sections.length) return htmlDoc(scope, `<p class="none">No VERY_IMPORTANT_INSTRUCTIONS attention found — recompute the VII split (⚗ VII sample) in the /tf inspector first.</p>`, "");
	// `</` escaped so a stray "</script>" inside any instruction can't close the tag.
	const scriptJs = payload.length
		? `window.__VII__=${JSON.stringify(payload).replace(/</g, "\\u003c")};(${viiReportRuntime.toString()})();`
		: "";
	return htmlDoc(scope, sections.join("\n"), scriptJs);
}

function reportName() {
	const base = [state.run, state.slot, state.model].filter(Boolean).join("_").replace(/[^a-z0-9_.-]+/gi, "-");
	return `vii-report_${base || "cell"}.html`;
}

// Button handler: load every computed step, build the report, and DOWNLOAD it as
// a self-contained .html file (the report is fully interactive offline).
export async function runViiReport(btn) {
	const label = btn.textContent;
	btn.disabled = true; btn.textContent = "generating…";
	try {
		const allRows = await loadAllRows();
		if (!allRows.length) { toast("no computed steps to report", "err"); return; }
		await ensureEvents(); // the report quotes each kind's VII prompt verbatim (stepLLM)
		const html = buildViiReport(allRows);
		const name = reportName();
		const url = URL.createObjectURL(new Blob([html], { type: "text/html" }));
		const a = el("a", { href: url, download: name });
		document.body.appendChild(a); a.click(); a.remove();
		setTimeout(() => URL.revokeObjectURL(url), 60000);
		toast(`saved ${name}`, "ok");
	} catch (e) {
		toast(`report failed: ${e.message}`, "err");
	} finally {
		btn.disabled = false; btn.textContent = label;
	}
}
