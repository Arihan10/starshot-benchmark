// Self-contained chart toolkit for the /tf data view. Ported/trimmed from the
// legacy inspector's proven renderers so the visuals match, but with a small,
// explicit dependency surface (el + a few color constants) instead of the legacy
// dependency-injection + global-state machinery.
//
// Exposes: spiderChart (canvas radial), pieChart (donut), stackAreaChart (SVG),
// scatterChart (SVG), plus the shared frame/hover/legend helpers.

import { el } from "../../js/ui.js";
import { COMPONENT_ABBR, compHex, ATTR_AXIS_ORDER } from "./state.js";

const SVGNS = "http://www.w3.org/2000/svg";
export function svgEl(tag, attrs, text) {
	const n = document.createElementNS(SVGNS, tag);
	if (attrs) for (const [k, v] of Object.entries(attrs)) { if (v == null) continue; n.setAttribute(k, v); }
	if (text != null) n.appendChild(document.createTextNode(String(text)));
	return n;
}
export function hexA(hex, a) {
	const h = String(hex).replace("#", "");
	const b = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
	const num = parseInt(b, 16);
	return `rgba(${(num >> 16) & 255},${(num >> 8) & 255},${num & 255},${a})`;
}
export function escTip(s) {
	return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
export function niceMax(v) {
	if (!(v > 0)) return 1;
	const p = Math.pow(10, Math.floor(Math.log10(v)));
	const n = v / p;
	const step = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
	return step * p;
}
export const pctFmt = (v) => `${(v * 100).toFixed(v > 0 && v < 0.1 ? 1 : 0)}%`;

// Font scale: chart text grows with the chart's width, capped, so labels stay
// legible on big displays without ballooning on small ones. Returns ~1 at a
// ~760px chart, up to 1.7 on wide ones.
export function fontScale(W) { return Math.max(1, Math.min(1.7, (W || 760) / 760)); }
const fpx = (base, fs) => +(base * fs).toFixed(1);

// --- responsive mounting -----------------------------------------------------
// Charts render at the ACTUAL pixel size of their host (not a fixed viewBox that
// CSS then scales), so text/dots stay crisp and legible and every graph fills the
// space it's given. `draw(w, h)` returns a node; `heightFn(w)` picks the height.
// A single ResizeObserver repaints on container-width changes; a window-resize
// listener covers viewport-height changes (for vh-based heights). Hosts removed
// from the DOM are auto-unobserved.
const _hosts = new Set();
const _ro = new ResizeObserver((entries) => {
	for (const e of entries) {
		const host = e.target;
		if (!host.isConnected) { _ro.unobserve(host); _hosts.delete(host); continue; }
		_paint(host);
	}
});
function _paint(host) {
	const w = Math.max(220, Math.floor(host.clientWidth || 0));
	if (!w) return;
	if (host.__w === w) return; // width unchanged → skip (height changes don't feed back)
	host.__w = w;
	host.replaceChildren(host.__draw(w, Math.round(host.__heightFn(w))));
}
let _rz = null;
window.addEventListener("resize", () => {
	clearTimeout(_rz);
	_rz = setTimeout(() => { for (const h of _hosts) { if (!h.isConnected) { _hosts.delete(h); continue; } h.__w = 0; _paint(h); } }, 120);
});
export function mountChart(host, draw, heightFn) {
	host.__draw = draw;
	host.__heightFn = heightFn || ((w) => Math.round(w * 0.34));
	host.__w = 0;
	_hosts.add(host);
	_ro.observe(host);
	_paint(host); // paints now if already sized; else the observer fires once laid out
	return host;
}
// Force a re-draw at the current width (e.g. after an interaction changes what the
// draw closure should render, like a hover-focus or a zoom).
export function repaint(host) { host.__w = 0; _paint(host); }
let _uid = 0;
// A block host that fills its parent's width; mount a chart into it.
export function chartHost(draw, heightFn) {
	const host = el("div", { class: "chart-host", style: "width:100%" });
	mountChart(host, draw, heightFn);
	return host;
}

// Clickable/plain legend row of swatches. `onToggle(key)` + `off` (Set of off
// keys) make it a filter control; omit them for a static legend.
export function chartLegend(items, { onToggle = null, off = null } = {}) {
	return el("div", { class: "chart-legend" }, ...items.filter(Boolean).map((it) => {
		const isOff = off && off.has(it.key);
		const row = el("div", { class: `lg${onToggle ? " clickable" : ""}${isOff ? " off" : ""}` },
			el("span", { class: "sw", style: `background:${it.color}` }),
			el("span", { text: it.label }));
		if (onToggle) row.onclick = () => onToggle(it.key);
		return row;
	}));
}

// Shared axis frame: y grid + ticks (0..yMax), three x ticks, axis label,
// optional output shade. Returns the <svg>, coordinate mappers, and rect.
function svgFrame(W, H, { padL, padR, padT, padB, xMin, xMax, yMax, yFmt, xFmt, xLabel, shade, xTicks = true, logY = false, fs = 1 }) {
	const px0 = padL, px1 = W - padR, py0 = padT, py1 = H - padB;
	const xr = (xMax - xMin) || 1; yMax = yMax || 1;
	const X = (x) => px0 + ((x - xMin) / xr) * (px1 - px0);
	// log-Y: a real log10 axis over [yFloor, yMax] so a heavy-tailed / near-zero
	// distribution spreads across the height instead of hugging the baseline.
	const yFloor = logY ? Math.max(yMax * 1e-3, 1e-6) : 0;
	const lY0 = logY ? Math.log10(yFloor) : 0, lY1 = logY ? Math.log10(yMax || 1) : 0, lSpan = (lY1 - lY0) || 1;
	const Y = logY
		? (y) => py1 - ((Math.log10(Math.min(yMax, Math.max(yFloor, y))) - lY0) / lSpan) * (py1 - py0)
		: (y) => py1 - (Math.max(0, Math.min(yMax, y)) / yMax) * (py1 - py0);
	const svg = svgEl("svg", { class: "gsvg", viewBox: `0 0 ${W} ${H}`, width: W, height: H });
	if (shade && shade.to > shade.from) svg.appendChild(svgEl("rect", { x: X(shade.from), y: py0, width: Math.max(0, X(shade.to) - X(shade.from)), height: py1 - py0, fill: "rgba(122,79,208,0.13)" }));
	for (const f of [0, 0.25, 0.5, 0.75, 1]) {
		const yy = py1 - f * (py1 - py0);
		const val = logY ? Math.pow(10, lY0 + f * lSpan) : yMax * f;
		svg.appendChild(svgEl("line", { x1: px0, y1: yy, x2: px1, y2: yy, stroke: "rgba(255,255,255,0.07)" }));
		svg.appendChild(svgEl("text", { x: px0 - 6, y: yy, fill: "rgba(220,230,245,0.62)", "font-size": fpx(12, fs), "text-anchor": "end", "dominant-baseline": "middle" }, yFmt ? yFmt(val) : val.toFixed(2)));
	}
	if (xTicks) for (const f of [0, 0.5, 1]) {
		const xv = xMin + xr * f;
		svg.appendChild(svgEl("text", { x: X(xv), y: py1 + fpx(14, fs), fill: "rgba(220,230,245,0.62)", "font-size": fpx(12, fs), "text-anchor": "middle" }, xFmt ? xFmt(xv) : String(Math.round(xv))));
	}
	if (xLabel) svg.appendChild(svgEl("text", { x: px1, y: py1 + fpx(27, fs), fill: "rgba(220,230,245,0.82)", "font-size": fpx(12.5, fs), "text-anchor": "end" }, xLabel));
	return { svg, X, Y, px0, px1, py0, py1, yMax, W, H, fs };
}
function svgCorner(fr, text) {
	if (text) fr.svg.appendChild(svgEl("text", { x: fr.px1 - 2, y: fr.py0 + 1, fill: "rgba(230,238,250,0.9)", "font-size": fpx(12.5, fr.fs || 1), "font-weight": 600, "text-anchor": "end", "dominant-baseline": "hanging" }, text));
}

// Crosshair + value tooltip. entries: [{label,color,values[]}] aligned to xs.
function chartHover(wrap, fr, xs, entries, { xFmt, vFmt } = {}) {
	if (!xs || !xs.length || !entries.length) return;
	const svg = fr.svg;
	const guide = svgEl("line", { y1: fr.py0, y2: fr.py1, stroke: "rgba(255,255,255,0.42)", "stroke-width": 1, "stroke-dasharray": "3 3", visibility: "hidden", "pointer-events": "none" });
	svg.appendChild(guide);
	const tip = el("div", { class: "graph-tip" });
	wrap.appendChild(tip);
	const xf = xFmt || ((v) => String(Math.round(v)));
	const vf = vFmt || ((v) => v.toFixed(3));
	const hide = () => { guide.setAttribute("visibility", "hidden"); tip.style.opacity = "0"; };
	svg.addEventListener("pointermove", (ev) => {
		const r = svg.getBoundingClientRect();
		const vx = (ev.clientX - r.left) * (fr.W / r.width);
		const vy = (ev.clientY - r.top) * (fr.H / r.height);
		if (vx < fr.px0 - 2 || vx > fr.px1 + 2 || vy < fr.py0 - 2 || vy > fr.py1 + 2) return hide();
		let idx = 0, best = Infinity;
		for (let i = 0; i < xs.length; i++) { const dd = Math.abs(fr.X(xs[i]) - vx); if (dd < best) { best = dd; idx = i; } }
		const gx = fr.X(xs[idx]);
		guide.setAttribute("x1", gx); guide.setAttribute("x2", gx); guide.setAttribute("visibility", "visible");
		const rowsHtml = entries.map((e) => ({ e, v: +(e.values[idx] || 0) })).filter((o) => o.v > 1e-9).sort((a, b) => b.v - a.v).slice(0, 10)
			.map((o) => `<div class="r"><span><span class="sw" style="background:${o.e.color}"></span>${escTip(o.e.label)}</span><b>${escTip(vf(o.v))}</b></div>`).join("");
		tip.innerHTML = `<div class="xh">${escTip(xf(xs[idx]))}</div>` + (rowsHtml || `<div style="opacity:.5">no attention here</div>`);
		const wr = wrap.getBoundingClientRect();
		const tw = tip.offsetWidth || 150, th = tip.offsetHeight || 60;
		let left = ev.clientX - wr.left + 14; if (left + tw > wr.width) left = ev.clientX - wr.left - tw - 14;
		let top = ev.clientY - wr.top + 12; if (top + th > wr.height) top = ev.clientY - wr.top - th - 12;
		tip.style.left = Math.max(0, left) + "px"; tip.style.top = Math.max(0, top) + "px"; tip.style.opacity = "1";
	});
	svg.addEventListener("pointerleave", hide);
}

// Vertical segment markers (output item = major, attribute field = minor), drawn
// with the plot's (possibly zoomed) X mapping. vlines: [{x, label, major}].
function drawVlines(fr, vlines) {
	if (!vlines || !vlines.length) return;
	const fs = fr.fs || 1;
	let lastMaj = -1e9, lastMin = -1e9;
	for (const v of vlines) {
		const px = fr.X(v.x);
		if (px < fr.px0 - 0.5 || px > fr.px1 + 0.5) continue;
		fr.svg.appendChild(svgEl("line", { x1: px.toFixed(1), y1: fr.py0, x2: px.toFixed(1), y2: fr.py1, stroke: v.major ? "rgba(255,255,255,0.34)" : "rgba(160,185,225,0.18)", "stroke-width": v.major ? 1 : 0.6, "stroke-dasharray": v.major ? null : "2 3" }));
		if (!v.label) continue;
		if (v.major) { if (px - lastMaj < 40 * fs) continue; lastMaj = px;
			fr.svg.appendChild(svgEl("text", { x: (px + 4).toFixed(1), y: fr.py0 + 2, fill: "rgba(240,246,255,0.98)", stroke: "rgba(8,10,18,0.94)", "stroke-width": 3.5, "paint-order": "stroke", "font-size": fpx(12.5, fs), "font-weight": 700, "text-anchor": "start", "dominant-baseline": "hanging" }, v.label));
		} else { if (px - lastMin < 16 * fs) continue; lastMin = px;
			fr.svg.appendChild(svgEl("text", { x: (px + 1).toFixed(1), y: fr.py1 - 4, fill: "rgba(224,234,250,0.96)", stroke: "rgba(8,10,18,0.92)", "stroke-width": 3, "paint-order": "stroke", "font-size": fpx(11, fs), "text-anchor": "start", transform: `rotate(-90 ${(px + 1).toFixed(1)} ${(fr.py1 - 4).toFixed(1)})` }, v.label));
		}
	}
}

// --- stacked area ------------------------------------------------------------
// layers: [{label,color,values[]}] aligned to xs (ascending). opts: { width,
// height, xLabel, xFmt, yFmt, share (cap y at 1), yMax, legend, shade, onToggle,
// off, xMin, xMax (domain override / zoom), vlines, bands, selBand, onBand }.
// `bands` renders a hoverable "object lane" above the plot at FULL-domain
// positions (so hovering to switch objects works even while the plot is zoomed).
export function stackAreaChart(layers, xs, opts = {}) {
	layers = layers.filter((L) => (L.values || []).some((v) => v > 1e-9));
	if (!layers.length || !(xs || []).length) return el("div", { class: "empty", text: opts.empty || "no data to plot" });
	// A single-column selection can't form a band → widen it to a flat full-width column.
	if (xs.length === 1) { const x0 = xs[0]; xs = [x0 - 0.4, x0 + 0.4]; layers = layers.map((L) => ({ ...L, values: [L.values[0], L.values[0]] })); }
	const n = xs.length, W = Math.round(opts.width || 960), H = Math.round(opts.height || 220);
	const fs = fontScale(W);
	const hasLane = !!(opts.bands && opts.bands.length);
	const laneY = 5, laneH = Math.round(20 * fs);
	const padL = Math.round(34 + 12 * fs), padR = 14, padT = hasLane ? laneY + laneH + Math.round(8 * fs) : Math.round(6 + 6 * fs), padB = Math.round(22 + 16 * fs);
	const fullMin = xs[0], fullMax = xs[n - 1];
	const xMin = opts.xMin != null ? opts.xMin : fullMin, xMax = opts.xMax != null ? opts.xMax : fullMax;
	const cum = xs.map(() => 0);
	const tops = layers.map((L) => L.values.map((v, i) => (cum[i] += v)));
	const yMax = opts.yMax != null ? opts.yMax : opts.share ? Math.max(1, ...cum) : niceMax(Math.max(...cum, 1e-9));
	const fr = svgFrame(W, H, { padL, padR, padT, padB, xMin, xMax, yMax, yFmt: opts.yFmt, xFmt: opts.xFmt, xLabel: opts.xLabel, shade: opts.shade, fs });
	// clip the stacked areas to the plot rect so a zoomed domain doesn't spill out
	const clipId = `clip${++_uid}`;
	const defs = svgEl("defs", null);
	const cp = svgEl("clipPath", { id: clipId });
	cp.appendChild(svgEl("rect", { x: fr.px0, y: fr.py0, width: fr.px1 - fr.px0, height: fr.py1 - fr.py0 }));
	defs.appendChild(cp); fr.svg.appendChild(defs);
	const areaG = svgEl("g", { "clip-path": `url(#${clipId})` });
	let lower = xs.map(() => 0);
	layers.forEach((L, li) => {
		const upper = tops[li];
		let dstr = "";
		for (let i = 0; i < n; i++) dstr += (i ? "L" : "M") + fr.X(xs[i]).toFixed(2) + "," + fr.Y(upper[i]).toFixed(2) + " ";
		for (let i = n - 1; i >= 0; i--) dstr += "L" + fr.X(xs[i]).toFixed(2) + "," + fr.Y(lower[i]).toFixed(2) + " ";
		const area = svgEl("path", { d: dstr + "Z", fill: L.color, "fill-opacity": 0.85, stroke: L.color, "stroke-width": 0.6, "stroke-opacity": 0.9 });
		area.appendChild(svgEl("title", null, L.label));
		areaG.appendChild(area);
		lower = upper.slice();
	});
	fr.svg.appendChild(areaG);
	drawVlines(fr, opts.vlines);
	// object band lane (full-domain positions, above the plot)
	if (hasLane) {
		const fx = (x) => padL + ((x - fullMin) / ((fullMax - fullMin) || 1)) * (fr.px1 - padL);
		const laneFont = fpx(11, fs), chW = 6.4 * fs;
		fr.svg.appendChild(svgEl("rect", { x: padL, y: laneY, width: fr.px1 - padL, height: laneH, fill: "rgba(255,255,255,0.03)", rx: 3 }));
		for (const bd of opts.bands) {
			const a = fx(bd.x0), b = Math.max(fx(bd.x1), a + 2), on = bd.id === opts.selBand;
			const rect = svgEl("rect", { x: a.toFixed(1), y: laneY, width: (b - a).toFixed(1), height: laneH, fill: on ? "rgba(122,162,247,0.5)" : "rgba(122,162,247,0.16)", stroke: on ? "#7aa2f7" : "rgba(122,162,247,0.3)", "stroke-width": on ? 1.2 : 0.6, rx: 2, style: "cursor:zoom-in" });
			rect.appendChild(svgEl("title", null, `${bd.label} — hover to zoom to its output span`));
			if (opts.onBand) { rect.addEventListener("pointerenter", () => opts.onBand(bd.id)); }
			fr.svg.appendChild(rect);
			if (b - a > 3 * chW) fr.svg.appendChild(svgEl("text", { x: (a + 4).toFixed(1), y: laneY + laneH / 2, fill: on ? "#eaf1ff" : "rgba(224,232,246,0.82)", "font-size": laneFont, "dominant-baseline": "middle", style: "pointer-events:none" }, bd.label.length > (b - a) / chW ? bd.label.slice(0, Math.max(1, Math.floor((b - a) / chW))) + "…" : bd.label));
		}
		if (opts.onBand) fr.svg.addEventListener("pointerleave", () => opts.onBand(null));
	}
	svgCorner(fr, opts.corner);
	const wrap = el("div", { class: "gwrap" }, fr.svg);
	chartHover(wrap, fr, xs.filter((x) => x >= xMin - 1e-9 && x <= xMax + 1e-9), layers.map((L) => ({ label: L.label, color: L.color, values: L.values.filter((_, i) => xs[i] >= xMin - 1e-9 && xs[i] <= xMax + 1e-9) })), { xFmt: opts.xFmt, vFmt: opts.yFmt || pctFmt });
	if (opts.legend) wrap.appendChild(chartLegend(opts.legend, { onToggle: opts.onToggle, off: opts.off }));
	return wrap;
}

// --- scatter -----------------------------------------------------------------
// points: [{x,y,color,r,label}]. opts: { height, xLabel, xFmt, yFmt, xMax, yMax,
// corrLabel, legend, refLine }.
export function scatterChart(points, opts = {}) {
	if (!points.length) return el("div", { class: "empty", text: opts.empty || "no data to plot" });
	const W = Math.round(opts.width || 960), H = Math.round(opts.height || 300), fs = fontScale(W);
	const padL = Math.round(34 + 12 * fs), padR = 14, padT = Math.round(6 + 6 * fs), padB = Math.round(22 + 16 * fs);
	const xMax = opts.xMax != null ? opts.xMax : niceMax(Math.max(...points.map((p) => p.x), 1e-9));
	const yMax = opts.yMax != null ? opts.yMax : niceMax(Math.max(...points.map((p) => p.y), 1e-9));
	const fr = svgFrame(W, H, { padL, padR, padT, padB, xMin: 0, xMax, yMax, yFmt: opts.yFmt, xFmt: opts.xFmt, xLabel: opts.xLabel, logY: opts.logY, fs });
	// transparent catch layer so pointer events fire across the whole plot, not just on the (tiny) dots
	fr.svg.appendChild(svgEl("rect", { x: fr.px0, y: fr.py0, width: Math.max(0, fr.px1 - fr.px0), height: Math.max(0, fr.py1 - fr.py0), fill: "none", style: "pointer-events:all" }));
	if (opts.refLine) { const m = Math.min(xMax, yMax); fr.svg.appendChild(svgEl("line", { x1: fr.X(0), y1: fr.Y(0), x2: fr.X(m), y2: fr.Y(m), stroke: "rgba(255,255,255,0.25)", "stroke-width": 1, "stroke-dasharray": "4 3" })); }
	// y error bars (whiskers with end caps), drawn under the fit line + dots
	if (opts.yErr) {
		for (const p of points) {
			const e = opts.yErr(p) || 0;
			if (!(e > 0)) continue;
			const cx = +fr.X(p.x).toFixed(2), yHi = fr.Y(Math.min(yMax, p.y + e)).toFixed(2), yLo = fr.Y(Math.max(0, p.y - e)).toFixed(2);
			const op = (opts.dim && opts.dim(p)) ? 0.1 : 0.4, col = p.color || "#7aa2f7", cap = 3;
			fr.svg.appendChild(svgEl("line", { x1: cx, y1: yHi, x2: cx, y2: yLo, stroke: col, "stroke-width": 1.1, "stroke-opacity": op }));
			fr.svg.appendChild(svgEl("line", { x1: cx - cap, y1: yHi, x2: cx + cap, y2: yHi, stroke: col, "stroke-width": 1.1, "stroke-opacity": op }));
			fr.svg.appendChild(svgEl("line", { x1: cx - cap, y1: yLo, x2: cx + cap, y2: yLo, stroke: col, "stroke-width": 1.1, "stroke-opacity": op }));
		}
	}
	const xf = opts.xFmt || ((v) => v.toFixed(2)), yf = opts.yFmt || ((v) => v.toFixed(2));
	// Draw dimmed points first and highlighted (hot) ones last, so matches sit on top.
	const ordered = opts.hot ? [...points].sort((a, b) => (opts.hot(a) ? 1 : 0) - (opts.hot(b) ? 1 : 0)) : points;
	// persistent ring marks (e.g. trim candidates), under the dots
	if (opts.ring) {
		for (const p of points) {
			const c = opts.ring(p); if (!c) continue;
			fr.svg.appendChild(svgEl("circle", { cx: +fr.X(p.x).toFixed(2), cy: +fr.Y(p.y).toFixed(2), r: (p.r || 3.2) + 3.8, fill: "none", stroke: c, "stroke-width": 2, "stroke-opacity": (opts.dim && opts.dim(p)) ? 0.15 : 1 }));
		}
	}
	const dots = [];
	for (const p of ordered) {
		const cx = +fr.X(p.x).toFixed(2), cy = +fr.Y(p.y).toFixed(2), r0 = p.r || 3.2;
		const dim = opts.dim ? opts.dim(p) : false, hot = opts.hot ? opts.hot(p) : false;
		const base = { r: hot ? r0 + 2.2 : r0, stroke: hot ? "#fff" : "none", sw: hot ? 1.6 : 0 };
		const dot = svgEl("circle", {
			cx, cy, r: base.r, fill: p.color || "#7aa2f7",
			"fill-opacity": dim ? 0.12 : 0.82, stroke: base.stroke, "stroke-width": base.sw,
		});
		if (!opts.tip) dot.appendChild(svgEl("title", null, `${p.label ? p.label + "  " : ""}(${xf(p.x)}, ${yf(p.y)})`));
		dots.push({ p, dot, cx, cy, r0, base });
		fr.svg.appendChild(dot);
	}
	// optional polyline overlay (bin-mean trend / linear fit) — drawn LAST so it sits
	// ON TOP of the dots (high z), BRIGHT RED by default, with a dark casing so it
	// reads clearly through a dense point cloud.
	if (opts.line && (opts.line.points || []).length > 1) {
		const lp = [...opts.line.points].sort((a, b) => a.x - b.x);
		let d = ""; lp.forEach((p, i) => { d += (i ? "L" : "M") + fr.X(p.x).toFixed(1) + "," + fr.Y(p.y).toFixed(1) + " "; });
		const dash = opts.line.dash || null, w = opts.line.width || 2.5, col = opts.line.color || "#ff2b2b";
		fr.svg.appendChild(svgEl("path", { d: d.trim(), fill: "none", stroke: "rgba(8,10,16,0.78)", "stroke-width": w + 2, "stroke-dasharray": dash, "stroke-linejoin": "round", "stroke-linecap": "round" }));
		fr.svg.appendChild(svgEl("path", { d: d.trim(), fill: "none", stroke: col, "stroke-width": w, "stroke-dasharray": dash, "stroke-linejoin": "round", "stroke-linecap": "round" }));
	}
	svgCorner(fr, opts.corrLabel);
	const wrap = el("div", { class: "gwrap" }, fr.svg);
	if (opts.tip) scatterHover(wrap, fr, dots, opts.tip);
	if (opts.legend) wrap.appendChild(chartLegend(opts.legend));
	return wrap;
}

// Nearest-dot hover for scatterChart: on pointer move find the closest dot within
// a hit radius, emphasize it (grow + white ring), and show tipFn(point) HTML in a
// floating .graph-tip. Fixes the tiny-target problem of native <title> hovers.
function scatterHover(wrap, fr, dots, tipFn) {
	if (!dots.length) return;
	const svg = fr.svg;
	const tip = el("div", { class: "graph-tip scatter-tip" });
	wrap.appendChild(tip);
	let cur = null;
	const reset = (d) => { d.dot.setAttribute("r", d.base.r); d.dot.setAttribute("stroke", d.base.stroke); d.dot.setAttribute("stroke-width", d.base.sw); };
	const hide = () => { tip.style.opacity = "0"; if (cur) { reset(cur); cur = null; } };
	svg.addEventListener("pointermove", (ev) => {
		const r = svg.getBoundingClientRect();
		const vx = (ev.clientX - r.left) * (fr.W / (r.width || fr.W));
		const vy = (ev.clientY - r.top) * (fr.H / (r.height || fr.H));
		let best = null, bd = Infinity;
		for (const d of dots) { const dd = (d.cx - vx) ** 2 + (d.cy - vy) ** 2; if (dd < bd) { bd = dd; best = d; } }
		const HIT = 30;
		if (!best || bd > HIT * HIT) return hide();
		if (best !== cur) { // only restyle when the nearest dot changes
			if (cur) reset(cur);
			best.dot.setAttribute("r", best.r0 + 2.6); best.dot.setAttribute("stroke", "#fff"); best.dot.setAttribute("stroke-width", 1.7);
			best.dot.parentNode.appendChild(best.dot); // raise to top
			cur = best;
			tip.innerHTML = tipFn(best.p);
		}
		const wr = wrap.getBoundingClientRect();
		const tw = tip.offsetWidth || 200, th = tip.offsetHeight || 60;
		let left = ev.clientX - wr.left + 14; if (left + tw > wr.width) left = ev.clientX - wr.left - tw - 14;
		let top = ev.clientY - wr.top + 12; if (top + th > wr.height) top = ev.clientY - wr.top - th - 12;
		tip.style.left = Math.max(0, left) + "px"; tip.style.top = Math.max(0, top) + "px"; tip.style.opacity = "1";
	});
	svg.addEventListener("pointerleave", hide);
}

// --- joint scatter (spatial relevance) --------------------------------------
// Square-ish scatter colored by category, with per-category smoothed marginal
// histograms (top = x, right = y). points: [{x,y,cat,attn,r,o,label}]. opts:
// { width, height, cats:[{key,color,label}], coms:[{cat,x,y,color,label}] (drawn
// as attention-weighted center-of-mass markers), xMax, yMax, xLabel, yLabel,
// corrLabel, sizeLabel, xFmt, yFmt }.
export function jointScatter(points, opts = {}) {
	if (!points.length) return el("div", { class: "empty", text: opts.empty || "no data to plot" });
	const cats = (opts.cats && opts.cats.length) ? opts.cats : [{ key: "", color: "#7aa2f7", label: "" }];
	const colorOf = new Map(cats.map((c) => [c.key, c.color]));
	// SQUARE: one side = min(available width, height cap); the main plot is a true
	// square (equal visual scale on both axes, right for two rankings).
	const S = Math.max(260, Math.min(Math.round(opts.width || 640), Math.round(opts.height || opts.width || 640)));
	const W = S, H = S, fs = fontScale(S);
	const m = Math.round(52 * fs), gap = 6, padL = Math.round(30 + 14 * fs), padR = 12, padT = Math.round(8 * fs), padB = Math.round(24 + 14 * fs);
	const mainL = padL, mainT = padT + m + gap;
	const side = Math.min(W - padL - padR - m - gap, H - padT - padB - m - gap);
	const mainR = mainL + side, mainB = mainT + side;
	const xMax = opts.xMax || Math.max(...points.map((p) => p.x), 1);
	const yMax = opts.yMax || Math.max(...points.map((p) => p.y), 1);
	const X = (x) => mainL + (x / (xMax || 1)) * (mainR - mainL);
	const Y = (y) => mainB - (y / (yMax || 1)) * (mainB - mainT);
	const xf = opts.xFmt || ((v) => String(Math.round(v))), yf = opts.yFmt || ((v) => String(Math.round(v)));
	const svg = svgEl("svg", { class: "gsvg", viewBox: `0 0 ${W} ${H}`, width: W, height: H, style: "display:block;margin:0 auto" });
	for (const f of [0, 0.25, 0.5, 0.75, 1]) {
		const gy = mainB - f * (mainB - mainT), gx = mainL + f * (mainR - mainL);
		svg.appendChild(svgEl("line", { x1: mainL, y1: gy.toFixed(1), x2: mainR, y2: gy.toFixed(1), stroke: "rgba(255,255,255,0.06)" }));
		svg.appendChild(svgEl("line", { x1: gx.toFixed(1), y1: mainT, x2: gx.toFixed(1), y2: mainB, stroke: "rgba(255,255,255,0.06)" }));
		svg.appendChild(svgEl("text", { x: gx.toFixed(1), y: mainB + fpx(14, fs), fill: "rgba(220,230,245,0.6)", "font-size": fpx(11, fs), "text-anchor": "middle" }, xf(f * xMax)));
		svg.appendChild(svgEl("text", { x: mainL - 6, y: gy.toFixed(1), fill: "rgba(220,230,245,0.6)", "font-size": fpx(11, fs), "text-anchor": "end", "dominant-baseline": "middle" }, yf(f * yMax)));
	}
	if (opts.xLabel) svg.appendChild(svgEl("text", { x: ((mainL + mainR) / 2).toFixed(1), y: H - 4, fill: "rgba(220,230,245,0.85)", "font-size": fpx(12, fs), "text-anchor": "middle" }, opts.xLabel));
	if (opts.yLabel) { const my = ((mainT + mainB) / 2).toFixed(1); svg.appendChild(svgEl("text", { x: 12, y: my, fill: "rgba(220,230,245,0.85)", "font-size": fpx(12, fs), "text-anchor": "middle", transform: `rotate(-90 12 ${my})` }, opts.yLabel)); }
	// per-category smoothed marginal histograms
	const nb = opts.nbins || Math.min(Math.max(6, Math.round(Math.max(xMax, yMax))), 24);
	const binOf = (vals, mx) => { const b = new Array(nb).fill(0); for (const v of vals) b[Math.min(nb - 1, Math.max(0, Math.floor((v / (mx || 1)) * nb)))] += 1; return b; };
	const smooth = (b) => b.map((_, i) => (b[i - 1] || 0) * 0.25 + b[i] * 0.5 + (b[i + 1] || 0) * 0.25);
	const topB = cats.map((c) => smooth(binOf(points.filter((p) => p.cat === c.key).map((p) => p.x), xMax)));
	const topMax = Math.max(1e-9, ...topB.flat());
	cats.forEach((c, ci) => {
		let dd = `M ${mainL.toFixed(1)} ${(padT + m).toFixed(1)}`;
		topB[ci].forEach((v, i) => { dd += ` L ${(mainL + ((i + 0.5) / nb) * (mainR - mainL)).toFixed(1)} ${((padT + m) - (v / topMax) * (m - 3)).toFixed(1)}`; });
		dd += ` L ${mainR.toFixed(1)} ${(padT + m).toFixed(1)} Z`;
		svg.appendChild(svgEl("path", { d: dd, fill: hexA(c.color, 0.3), stroke: c.color, "stroke-width": 1 }));
	});
	const rgtB = cats.map((c) => smooth(binOf(points.filter((p) => p.cat === c.key).map((p) => p.y), yMax)));
	const rgtMax = Math.max(1e-9, ...rgtB.flat()), rx0 = mainR + gap;
	cats.forEach((c, ci) => {
		let dd = `M ${rx0.toFixed(1)} ${mainB.toFixed(1)}`;
		rgtB[ci].forEach((v, i) => { dd += ` L ${(rx0 + (v / rgtMax) * (m - 3)).toFixed(1)} ${(mainB - ((i + 0.5) / nb) * (mainB - mainT)).toFixed(1)}`; });
		dd += ` L ${rx0.toFixed(1)} ${mainT.toFixed(1)} Z`;
		svg.appendChild(svgEl("path", { d: dd, fill: hexA(c.color, 0.3), stroke: c.color, "stroke-width": 1 }));
	});
	for (const p of points) {
		const dot = svgEl("circle", { cx: X(p.x).toFixed(2), cy: Y(p.y).toFixed(2), r: p.r || 3.4, fill: colorOf.get(p.cat) || "#7aa2f7", "fill-opacity": p.o ?? 0.62 });
		dot.appendChild(svgEl("title", null, p.label || `(${xf(p.x)}, ${yf(p.y)})`));
		svg.appendChild(dot);
	}
	// attention-weighted center-of-mass markers (mass = attention): a haloed ◎ at
	// each category's CoM so you can read where its attention actually concentrates.
	for (const c of opts.coms || []) {
		if (c.x == null || c.y == null) continue;
		const cx = X(c.x), cy = Y(c.y), R = 8 * fs;
		svg.appendChild(svgEl("circle", { cx: cx.toFixed(1), cy: cy.toFixed(1), r: (R + 2).toFixed(1), fill: "none", stroke: "rgba(8,10,18,0.85)", "stroke-width": 4 }));
		svg.appendChild(svgEl("circle", { cx: cx.toFixed(1), cy: cy.toFixed(1), r: R.toFixed(1), fill: "none", stroke: c.color, "stroke-width": 2.4 }));
		svg.appendChild(svgEl("circle", { cx: cx.toFixed(1), cy: cy.toFixed(1), r: (R * 0.34).toFixed(1), fill: c.color, stroke: "rgba(8,10,18,0.85)", "stroke-width": 1 }));
		svg.appendChild(svgEl("text", { x: (cx + R + 3).toFixed(1), y: cy.toFixed(1), fill: c.color, stroke: "rgba(8,10,18,0.9)", "stroke-width": 3, "paint-order": "stroke", "font-size": fpx(11, fs), "font-weight": 700, "dominant-baseline": "middle" }, `◎ ${c.label ?? c.cat}`));
	}
	if (opts.corrLabel) svg.appendChild(svgEl("text", { x: (mainR - 2).toFixed(1), y: mainT + 2, fill: "rgba(230,238,250,0.9)", "font-size": fpx(12, fs), "font-weight": 600, "text-anchor": "end", "dominant-baseline": "hanging" }, opts.corrLabel));
	const wrap = el("div", { class: "gwrap" }, svg);
	const items = cats.filter((c) => c.label).map((c) => ({ key: c.key, label: c.label, color: c.color }));
	if (items.length) wrap.appendChild(chartLegend(items));
	if (opts.sizeLabel) wrap.appendChild(el("div", { class: "muted", style: "text-align:center;font-size:11px;margin-top:2px", text: opts.sizeLabel }));
	return wrap;
}

// --- pie / donut -------------------------------------------------------------
// slices: [{key,label,value,color}]. opts: { active (key|null), onSlice(key) }.
// Clicking a slice (or legend row) toggles the active filter; the active slice
// is emphasized and the rest dimmed.
export function pieChart(slices, opts = {}) {
	const total = slices.reduce((s, x) => s + Math.max(0, x.value), 0);
	if (!(total > 0)) return el("div", { class: "empty", text: "no attention to compose" });
	const active = opts.active ?? null;
	const R = 92, r0 = 46, cx = 100, cy = 100;
	const S = Math.max(150, Math.round(opts.size || 240));
	const svg = svgEl("svg", { class: "pie-svg", viewBox: "0 0 200 200", width: S, height: S });
	let ang = -Math.PI / 2;
	for (const s of slices) {
		const frac = Math.max(0, s.value) / total;
		if (frac <= 0) continue;
		const a0 = ang, a1 = ang + frac * Math.PI * 2; ang = a1;
		const large = a1 - a0 > Math.PI ? 1 : 0;
		const p = (a, r) => [cx + Math.cos(a) * r, cy + Math.sin(a) * r];
		const [x0, y0] = p(a0, R), [x1, y1] = p(a1, R), [x2, y2] = p(a1, r0), [x3, y3] = p(a0, r0);
		const dim = active && active !== s.key;
		const path = svgEl("path", {
			class: `pie-slice${dim ? " dim" : ""}`,
			d: `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${R} ${R} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)} L ${x2.toFixed(2)} ${y2.toFixed(2)} A ${r0} ${r0} 0 ${large} 0 ${x3.toFixed(2)} ${y3.toFixed(2)} Z`,
			fill: s.color, stroke: "#0c0d10", "stroke-width": active === s.key ? 2.5 : 1.5,
		});
		path.appendChild(svgEl("title", null, `${s.label} · ${(frac * 100).toFixed(1)}% · ${s.value.toFixed(3)}`));
		if (opts.onSlice) path.onclick = () => opts.onSlice(active === s.key ? null : s.key);
		svg.appendChild(path);
	}
	// center total
	svg.appendChild(svgEl("text", { x: cx, y: cy - 4, fill: "rgba(220,230,245,0.85)", "font-size": 15, "font-weight": 700, "text-anchor": "middle" }, active ? slices.find((s) => s.key === active) ? `${((slices.find((s) => s.key === active).value / total) * 100).toFixed(0)}%` : "" : "100%"));
	svg.appendChild(svgEl("text", { x: cx, y: cy + 13, fill: "rgba(160,170,185,0.7)", "font-size": 9, "text-anchor": "middle", "letter-spacing": "0.08em" }, active ? String(active).toUpperCase() : "ATTENTION"));
	const legend = el("div", { class: "pie-legend" }, ...slices.filter((s) => s.value > 0).map((s) => {
		const frac = s.value / total, dim = active && active !== s.key;
		const row = el("div", { class: `row${active === s.key ? " active" : ""}${dim ? " dim" : ""}` },
			el("span", { class: "sw", style: `background:${s.color}` }),
			el("span", { class: "nm", text: s.label }),
			el("span", { class: "val", text: `${(frac * 100).toFixed(1)}%` }));
		if (opts.onSlice) row.onclick = () => opts.onSlice(active === s.key ? null : s.key);
		return row;
	}));
	return el("div", { class: "pie-wrap" }, svg, legend);
}

// --- spider / radial (attributes) -------------------------------------------
// Each profile normalized to its attribute SHARES (value / its own sum), then a
// shared scale maps the largest share to the ring. Hovering an axis shows the
// attribute's real attention value.
function spiderNormalize(profiles, components) {
	const props = profiles.map((p) => {
		const raw = components.map((c) => Math.max(0, p.map.get(c) || 0));
		const sum = raw.reduce((a, b) => a + b, 0) || 1;
		return raw.map((v) => v / sum);
	});
	const scale = Math.max(1e-9, ...props.flat());
	return props.map((vals) => vals.map((v) => v / scale));
}

// Canvas has no per-glyph DOM, so hit-test the recorded axis anchors on mousemove
// and set the canvas title to "<attribute> = <value>".
function attachAxisTooltip(canvas, spots, logicalSize) {
	const onMove = (ev) => {
		const rect = canvas.getBoundingClientRect();
		// spots are in LOGICAL (S) coordinates; map the pointer into that space.
		const sx = logicalSize / (rect.width || 1), sy = logicalSize / (rect.height || 1);
		const mx = (ev.clientX - rect.left) * sx, my = (ev.clientY - rect.top) * sy;
		let hit = null;
		for (const s of spots) {
			const left = s.align === "right" ? s.lx - 64 : s.align === "center" ? s.lx - 30 : s.lx - 6;
			const right = s.align === "left" ? s.lx + 64 : s.align === "center" ? s.lx + 30 : s.lx + 6;
			if (mx >= left && mx <= right && Math.abs(my - s.ly) <= 12) { hit = s; break; }
		}
		const t = hit ? `${hit.component} = ${hit.value.toFixed(4)}` : "";
		if (canvas.title !== t) { canvas.title = t; canvas.style.cursor = hit ? "help" : "default"; }
	};
	canvas.addEventListener("mousemove", onMove);
	canvas.addEventListener("mouseleave", () => { canvas.title = ""; canvas.style.cursor = "default"; });
}

// profiles: [{ label, color, map: Map(component -> value) }]. Returns a wrap with
// the canvas ring + an axis legend decoding the abbreviations. `size` (px) makes
// it fill the space it's given; the canvas is rendered at devicePixelRatio for
// crisp text at any scale.
export function spiderChart(profiles, { size = 340, axes = null, labelOf = null, colorOf = null } = {}) {
	const present = new Set(profiles.flatMap((p) => [...p.map.keys()]));
	// Axes: an explicit ordered list (e.g. token-type classes) restricted to those
	// actually present, else the fixed attribute axis order. `labelOf`/`colorOf` let
	// non-attribute spiders (token types) label/color their own axes instead of
	// borrowing the attribute abbreviations/hues.
	const components = axes
		? [...axes.filter((c) => present.has(c)), ...[...present].filter((c) => !axes.includes(c))]
		: [...ATTR_AXIS_ORDER, ...[...present].filter((c) => !ATTR_AXIS_ORDER.includes(c))];
	const lab = labelOf || ((c) => COMPONENT_ABBR[c] ?? c);
	const col = colorOf || ((c) => compHex(c));
	const norm = spiderNormalize(profiles, components);
	const S = Math.max(220, Math.round(size)), sc = S / 340, dpr = Math.min(2, window.devicePixelRatio || 1);
	const canvas = el("canvas", { class: "attr-radial" });
	canvas.width = Math.round(S * dpr); canvas.height = Math.round(S * dpr);
	canvas.style.width = S + "px"; canvas.style.height = S + "px";
	const ctx = canvas.getContext("2d");
	ctx.scale(dpr, dpr);
	const cx = S / 2, cy = S / 2, r0 = 16 * sc, r1 = 112 * sc, labR = 28 * sc, labR2 = 22 * sc, font = Math.max(11, Math.round(13 * sc));
	ctx.clearRect(0, 0, S, S);
	// concentric rings
	ctx.strokeStyle = "rgba(255,255,255,0.08)"; ctx.lineWidth = 1;
	for (const rr of [0.25, 0.5, 0.75, 1]) {
		ctx.beginPath();
		components.forEach((_, i) => {
			const ang = -Math.PI / 2 + (i / components.length) * Math.PI * 2;
			const x = cx + Math.cos(ang) * (r0 + (r1 - r0) * rr), y = cy + Math.sin(ang) * (r0 + (r1 - r0) * rr);
			i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
		});
		ctx.closePath(); ctx.stroke();
	}
	// spokes + labels
	const labelSpots = [];
	components.forEach((component, i) => {
		const ang = -Math.PI / 2 + (i / components.length) * Math.PI * 2;
		const gx = cx + Math.cos(ang) * r1, gy = cy + Math.sin(ang) * r1;
		ctx.strokeStyle = "rgba(255,255,255,0.12)";
		ctx.beginPath(); ctx.moveTo(cx + Math.cos(ang) * r0, cy + Math.sin(ang) * r0); ctx.lineTo(gx, gy); ctx.stroke();
		const lx = cx + Math.cos(ang) * (r1 + labR), ly = cy + Math.sin(ang) * (r1 + labR2);
		ctx.fillStyle = col(component);
		ctx.font = `${font}px ui-monospace, Menlo, monospace`;
		const align = lx < cx - 8 ? "right" : lx > cx + 8 ? "left" : "center";
		ctx.textAlign = align; ctx.textBaseline = "middle";
		ctx.fillText(lab(component), lx, ly, 64 * sc);
		labelSpots.push({ component, lx, ly, align, value: profiles[0] ? (profiles[0].map.get(component) || 0) : 0 });
	});
	attachAxisTooltip(canvas, labelSpots, S);
	// polygons
	profiles.forEach((p, pi) => {
		const color = p.color || "#7aa2f7";
		ctx.beginPath();
		components.forEach((_, i) => {
			const ang = -Math.PI / 2 + (i / components.length) * Math.PI * 2;
			const val = Math.max(0, Math.min(1, norm[pi][i]));
			const x = cx + Math.cos(ang) * (r0 + (r1 - r0) * val), y = cy + Math.sin(ang) * (r0 + (r1 - r0) * val);
			i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
		});
		ctx.closePath();
		ctx.fillStyle = `${color}33`; ctx.strokeStyle = color; ctx.lineWidth = 2;
		ctx.fill(); ctx.stroke();
		// vertex dots
		components.forEach((_, i) => {
			const ang = -Math.PI / 2 + (i / components.length) * Math.PI * 2;
			const val = Math.max(0, Math.min(1, norm[pi][i]));
			if (val <= 0.001) return;
			const x = cx + Math.cos(ang) * (r0 + (r1 - r0) * val), y = cy + Math.sin(ang) * (r0 + (r1 - r0) * val);
			ctx.beginPath(); ctx.arc(x, y, 2.6, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
		});
	});
	const axisLegend = el("div", { class: "attr-axis-legend" }, ...components.map((c) => el("span", { class: "axis-pill", title: c },
		el("i", { style: `background:${col(c)}` }),
		el("b", { text: lab(c) }),
		el("span", { text: c }))));
	return el("div", {}, el("div", { class: "attr-radial-wrap" }, canvas), axisLegend);
}
