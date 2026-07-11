// Content view — the per-step "structure" page, ported to the workspace's
// conventions: a 3D scene shaded by attention, the scene/attention badge tree,
// and the active zone's plan + the step's output + reasoning (entity phrases
// colored by attention). Tree badges, plan phrases, and output/reasoning entities
// all cross-highlight each other and the 3D scene (bidirectional, via ctHover).
//
// It is a SINGLE-step view: the header region + step selectors must resolve to one
// step (like the "prompts" button). Reuses the shared data layer + the same
// three.js viewer engine as the legacy inspector (scene3d.createViewer).

import { el } from "../../js/ui.js";
import { api } from "../../js/api.js";
import { createViewer } from "../../js/scene3d.js";
import { applySceneProjection, emittingRegion } from "../../js/events.js";
import {
	$, state, ALL, COLORS, COMPONENT_COLORS, entityHex, entityKindLabel,
	bumpLoad, ctHoverReset, ctHoverRegister,
} from "./state.js";
import { selectedSteps, loadRows, stepLLM } from "./data.js";

// --- small helpers -----------------------------------------------------------

const fmtNum = (v) => (v >= 1 ? v.toFixed(1) : v >= 0.01 ? v.toFixed(3) : v.toFixed(4));
// attention heat ramp (dark blue → bright yellow), matches the legacy tree.
const heat = (t) => { t = Math.max(0, Math.min(1, t)); return `hsl(${(212 - 162 * t).toFixed(0)}, 70%, ${(32 + 28 * t).toFixed(0)}%)`; };
function alpha(hex, a) {
	const h = String(hex).replace("#", "");
	const n = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
	const r = parseInt(n.slice(0, 2), 16) || 0, g = parseInt(n.slice(2, 4), 16) || 0, b = parseInt(n.slice(4, 6), 16) || 0;
	return `rgba(${r},${g},${b},${a})`;
}
const card = (title, sub, ...body) => {
	const head = el("div", { class: "card-head" }, el("span", { class: "card-title", text: title }));
	if (sub) head.appendChild(el("span", { class: "card-sub", text: sub }));
	return el("div", { class: "card" }, head, el("div", { class: "card-body" }, ...body.filter(Boolean)));
};
const emptyCard = (msg, big = "▦") => el("div", { class: "empty" }, el("span", { class: "big", text: big }), el("div", { text: msg }));

// The zone a step operates in: its own node when that's a zone, else the region
// that emitted it (a bbox step can run on a peer object).
function targetZoneOf(step) {
	if (!step || !step.node) return null;
	const n = state.obs?.nodes?.get(step.node);
	if (n && n.kind !== "zone") { const r = emittingRegion(state.obs, step.node); if (r) return r; }
	return step.node;
}

// id → { score, kind } and the peak, from a single step's scene aggregate.
function entityScores(a) {
	const ent = new Map(); let max = 0;
	for (const e of (((a.agg || {}).scene || {}).entityTotals || [])) { const sc = e.score || 0; ent.set(e.id, { score: sc, kind: e.kind }); if (sc > max) max = sc; }
	return { ent, max };
}

// Mean per-entity attention across a multi-step selection (absent counts as 0) —
// the aggregate the tree + 3D shading use when more than one step is in scope, so
// the structure view works for ANY selection (just less rich than a single step).
function aggregateRows(rows) {
	if (rows.length === 1) return entityScores(rows[0].a);
	const n = rows.length || 1;
	const acc = new Map();
	for (const r of rows) for (const e of (((r.a.agg || {}).scene || {}).entityTotals || [])) {
		const cur = acc.get(e.id) || { kind: e.kind, sum: 0 }; cur.sum += e.score || 0; acc.set(e.id, cur);
	}
	const ent = new Map(); let max = 0;
	for (const [id, v] of acc) { const mean = v.sum / n; ent.set(id, { score: mean, kind: v.kind }); if (mean > max) max = mean; }
	return { ent, max };
}

// --- entity mention highlighting (output / reasoning: id + attribute keys) ----

const _escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
function entityCtx(a) {
	const { ent, max } = entityScores(a);
	const kindOf = new Map();
	for (const e of a.scene_entities || []) kindOf.set(e.id, e.kind);
	for (const [id, info] of ent) if (!kindOf.has(id)) kindOf.set(id, info.kind);
	return { ent, max, kindOf, entities: new Set([...kindOf.keys(), ...ent.keys()]) };
}
function entitySpan(id, ctx) {
	const info = ctx.ent.get(id);
	const kind = (info && info.kind) || ctx.kindOf.get(id) || "object";
	const color = info && ctx.max > 0 ? heat(info.score / ctx.max) : entityHex(kind, id);
	const span = el("span", {
		class: "ct-ent", text: id,
		title: `${id} · ${entityKindLabel(kind, id)}${info ? ` · attention ${info.score.toFixed(4)}` : ""} — hover to locate in the tree / 3D`,
		style: `border-bottom:2px solid ${color};background:${alpha(color, 0.14)}`,
		onclick: () => focusEntity(id),
	});
	ctHoverRegister(id, span);
	return span;
}
// Wrap scene-entity ids + JSON attribute keys in `text` (used for output/reasoning).
function highlightNodes(text, ctx) {
	const s = String(text || "");
	if (!s) return [document.createTextNode("")];
	const ids = [...ctx.entities].filter(Boolean).sort((a, b) => b.length - a.length);
	const attrs = Object.keys(COMPONENT_COLORS);
	const idAlt = ids.map(_escapeRe).join("|");
	const atAlt = attrs.map(_escapeRe).join("|");
	const src = [idAlt ? `\\b(?:${idAlt})\\b` : null, atAlt ? `"(?:${atAlt})"(?=\\s*:)` : null].filter(Boolean).join("|");
	if (!src) return [document.createTextNode(s)];
	const re = new RegExp(src, "g");
	const out = []; let last = 0, m;
	while ((m = re.exec(s))) {
		if (m.index === re.lastIndex) { re.lastIndex++; continue; }
		if (m.index > last) out.push(document.createTextNode(s.slice(last, m.index)));
		const hit = m[0];
		if (hit.startsWith('"')) out.push(el("span", { class: "ct-attr", style: `color:${COMPONENT_COLORS[hit.slice(1, -1)] || "#888"}`, text: hit }));
		else out.push(entitySpan(hit, ctx));
		last = re.lastIndex;
	}
	if (last < s.length) out.push(document.createTextNode(s.slice(last)));
	return out;
}
function fmtOutputNodes(text, ctx) {
	let s = String(text || "");
	try { s = JSON.stringify(JSON.parse(s), null, 2); } catch { /* not JSON → raw */ }
	return highlightNodes(s, ctx);
}

// --- zone-plan prose matcher (noun-phrase / id-slug, colored by attention) ----

const PLAN_STOP = new Set(["zone", "area", "region", "space", "room", "level", "floor", "wall", "ceiling", "side", "center", "edge", "object", "frame", "root", "the", "and", "with"]);
const normPhrase = (p) => String(p).toLowerCase().replace(/\s+/g, " ").trim();
function entityPhrases(id) {
	const out = [];
	const n = state.obs?.nodes?.get(id);
	if (n?.imagePrompt) out.push(String(n.imagePrompt));
	const words = String(id).replace(/[_-]+/g, " ").trim().split(/\s+/).filter((w) => w && !/^\d+$/.test(w));
	if (words.length) out.push(words.join(" "));
	return out;
}
function planPhraseRe(norm) {
	const words = norm.split(" ").map(_escapeRe);
	const last = words.length - 1;
	return `\\b${words.map((w, i) => (i === last ? `${w}s?` : w)).join("\\s+")}\\b`;
}
function planHighlightNodes(text, a) {
	const s = String(text || "");
	if (!s) return [document.createTextNode("")];
	const { ent, max } = entityScores(a);
	const seen = new Set(), cands = [];
	for (const [id, info] of ent) {
		for (const p of entityPhrases(id)) {
			const norm = normPhrase(p);
			if (norm.length < 4) continue;
			if (!norm.includes(" ") && PLAN_STOP.has(norm)) continue;
			const key = `${id}::${norm}`;
			if (seen.has(key)) continue; seen.add(key);
			cands.push({ norm, id, info });
		}
	}
	if (!cands.length) return [document.createTextNode(s)];
	cands.sort((a, b) => b.norm.length - a.norm.length);
	const claims = [];
	const overlaps = (a, b) => claims.some((c) => a < c.end && b > c.start);
	for (const c of cands) {
		let re; try { re = new RegExp(planPhraseRe(c.norm), "gi"); } catch { continue; }
		let m;
		while ((m = re.exec(s))) {
			if (m.index === re.lastIndex) { re.lastIndex++; continue; }
			const x = m.index, y = x + m[0].length;
			if (!overlaps(x, y)) claims.push({ start: x, end: y, id: c.id, score: c.info.score || 0, kind: c.info.kind });
		}
	}
	if (!claims.length) return [document.createTextNode(s)];
	claims.sort((a, b) => a.start - b.start);
	const out = []; let last = 0;
	for (const c of claims) {
		if (c.start < last) continue;
		if (c.start > last) out.push(document.createTextNode(s.slice(last, c.start)));
		const color = max > 0 ? heat(c.score / max) : entityHex(c.kind, c.id);
		const span = el("span", {
			class: "ct-ent", text: s.slice(c.start, c.end),
			title: `${c.id} · attention ${(c.score || 0).toFixed(4)} — hover to locate in the tree / 3D`,
			style: `border-bottom:2px solid ${color};background:${alpha(color, 0.16)}`,
			onclick: () => focusEntity(c.id),
		});
		ctHoverRegister(c.id, span);
		out.push(span);
		last = c.end;
	}
	if (last < s.length) out.push(document.createTextNode(s.slice(last)));
	return out;
}

// Clamp long prose with a show more / less toggle.
function clampProse(nodes) {
	const clampEl = el("div", { class: "ct-plan-clamp" }, ...nodes);
	const toggle = el("button", { class: "ct-plan-toggle", type: "button", text: "show more" });
	toggle.onclick = () => { toggle.textContent = clampEl.classList.toggle("expanded") ? "show less" : "show more"; };
	requestAnimationFrame(() => { if (clampEl.scrollHeight <= clampEl.clientHeight + 2) toggle.style.display = "none"; });
	return el("div", {}, clampEl, toggle);
}

// --- attention badge tree (width + heat) ------------------------------------

function buildTree(agg, targetZone, heatOnly) {
	const m = state.obs;
	if (!m) return emptyCard("no scene structure");
	const { ent, max: maxOne0 } = agg;
	const score = new Map([...ent].map(([id, v]) => [id, v.score]));
	const own = (id) => Math.max(0, score.get(id) || 0);
	const kindOf = new Map([...ent].map(([id, v]) => [id, v.kind]));
	const isZoneKind = (id) => ((kindOf.get(id) ?? m.nodes.get(id)?.kind) === "zone");
	const regionOf = (id) => { const r = emittingRegion(m, id); return r && r !== id && m.nodes.has(r) ? r : null; };
	const kidsOf = new Map();
	for (const id of m.order) { const p = regionOf(id); if (!p) continue; if (!kidsOf.has(p)) kidsOf.set(p, []); kidsOf.get(p).push(id); }
	const childrenOf = (id) => kidsOf.get(id) || [];
	const roots = m.order.filter((id) => !regionOf(id));
	const rootSet = new Set(roots);
	const subtotal = new Map(), calcd = new Set();
	const calc = (id) => { if (calcd.has(id)) return 0; calcd.add(id); let s = own(id); for (const c of childrenOf(id)) s += calc(c); subtotal.set(id, s); return s; };
	roots.forEach(calc);
	const attended = (id) => (subtotal.get(id) || 0) > 0;
	const isContainer = (id) => isZoneKind(id) || childrenOf(id).some(attended);
	const INDENT = 18;

	const rows = [], seen = new Set();
	const walk = (id, depth) => {
		if (seen.has(id)) return; seen.add(id);
		rows.push({ depth, zone: true, ids: [id] });
		const ks = childrenOf(id).filter(attended);
		const leaves = ks.filter((c) => !isContainer(c)).sort((x, y) => own(y) - own(x));
		const zones = ks.filter((c) => isContainer(c)).sort((x, y) => (subtotal.get(y) || 0) - (subtotal.get(x) || 0));
		if (leaves.length) rows.push({ depth: depth + 1, zone: false, ids: leaves });
		for (const c of zones) walk(c, depth + 1);
	};
	roots.filter(attended).sort((x, y) => (subtotal.get(y) || 0) - (subtotal.get(x) || 0)).forEach((r) => walk(r, 0));
	if (!rows.length) return el("div", { class: "muted", style: "font-size:12px", text: "no attended scene entities for this step" });

	const FILL = 86;
	const maxRow = Math.max(...rows.map((r) => r.ids.reduce((s, id) => s + own(id), 0)), 1e-9);
	const maxOne = Math.max(maxOne0, 1e-9);

	const badge = (id, zone, heatMode) => {
		const n = m.nodes.get(id);
		const isRoot = rootSet.has(id);
		const kindLab = isRoot ? "root" : entityKindLabel(n?.kind, id);
		const reg = regionOf(id);
		const isTarget = id === targetZone;
		const t = own(id) / maxOne;
		let style, cls;
		if (heatMode) {
			const dark = !isRoot && t > 0.52;
			style = `background:${isRoot ? "#464d5b" : heat(t)};border-color:${isRoot ? "#6b7280" : heat(Math.min(1, t + 0.12))};color:${dark ? "#141414" : "#e8eefc"}`;
			cls = `ct-badge heat${zone ? " zone" : ""}${isRoot ? " root" : ""}${isTarget ? " target" : ""}`;
		} else {
			const hex = isRoot ? "#9aa7bd" : entityHex(n?.kind, id);
			const wpct = isRoot ? FILL : (FILL * own(id)) / maxRow;
			style = `width:${wpct.toFixed(3)}%;background:${hex}2e;border-color:${hex}`;
			cls = `ct-badge${zone ? " zone" : ""}${isRoot ? " root" : ""}${isTarget ? " target" : ""}`;
		}
		const b = el("div", {
			class: cls, style,
			title: `${id} · ${kindLab}${reg ? ` · in ${reg}` : ""}${isTarget ? " · current step's zone" : ""} · attention ${fmtNum(own(id))}`,
			onclick: () => focusEntity(id),
		}, el("span", { class: "ct-badge-lab", text: id }));
		ctHoverRegister(id, b);
		return b;
	};

	const treeView = (heatMode) => el("div", { class: `ct-tree-view${heatMode ? " ct-heat" : ""}` },
		...rows.map((r) => el("div", { class: "ct-row", style: `margin-left:${r.depth * INDENT}px` }, ...r.ids.map((id) => badge(id, r.zone, heatMode)))));

	const legend = el("div", { class: "ct-tree-legend" },
		...[["root", "#9aa7bd"], ["zone", COLORS.zone], ["frame", COLORS.frame], ["object", COLORS.object]]
			.map(([k, hex]) => el("span", {}, el("i", { style: `background:${hex}` }), el("span", { text: ` ${k}` }))));

	const sub = (capText, heatMode) => el("div", {}, el("div", { class: "ct-tree-cap", text: capText }), treeView(heatMode));
	return el("div", { class: "ct-tree" },
		...(heatOnly ? [] : [sub("badge width = attention · hover for full name · click → focus", false)]),
		sub("badge color = attention · uniform width", true),
		legend);
}

// --- 3D viewer (created once, reused; shown in a draggable/resizable popout) ---

const MINW_KEY = "tf-ct-minw", KINDS_KEY = "tf-ct-kinds";
let ctMinW = (() => { try { const v = Number(localStorage.getItem(MINW_KEY)); return v >= 0 && v <= 0.95 ? v : 0.12; } catch { return 0.12; } })();

function ensureViewer() {
	if (state.viewer3d && state._ctHost) return true;
	try {
		state._ctHost = el("div", { class: "ct-canvas-host" });
		state.viewer3d = createViewer(state._ctHost, { keyboard: true, lighting: true });
		return true;
	} catch { state.viewer3d = null; state._ctHost = null; return false; }
}

// Focus an entity in 3D (select + frame) and glow it.
function focusEntity(id) {
	state.viewer3d?.select?.(id, { frame: true });
	state.viewer3d?.setAttnHighlight?.([{ id, weight: 1 }]);
}

// The 3D popout: a draggable + resizable floating window that reparents the shared
// viewer host into itself, shows the scene shaded by the CURRENT selection's
// attention, and refreshes in place on selection change. Not shown until opened.
let _win3d = null;
export function close3DWindow() {
	if (!_win3d) return;
	document.removeEventListener("keydown", _win3d.onKey, true);
	state.viewer3d?.clearAttnHighlight?.();
	state.viewer3d?.setActive?.(false);
	if (state._ctHost && state._ctHost.parentNode) state._ctHost.parentNode.removeChild(state._ctHost); // keep the host alive
	_win3d.win.remove();
	_win3d = null;
}
function open3DWindow() {
	if (_win3d) { _win3d.win.style.zIndex = String(++_winZ); refresh3DWindow(); return; }
	if (!ensureViewer()) return;
	const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); close3DWindow(); } };
	const sub = el("span", { class: "ct-win-sub" });
	const head = el("div", { class: "ct-win-head" },
		el("span", { style: "color:var(--text-faint)", text: "⠿" }),
		el("span", { class: "ct-win-title", text: "3D scene" }), sub, el("span", { style: "flex:1" }),
		el("button", { class: "ct-win-close", title: "close (Esc)", text: "×", onclick: close3DWindow }));
	const controls = el("div", { class: "ct-3d-controls" });
	const body = el("div", { class: "ct-win-3d-body" }, state._ctHost, controls);
	const grip = el("div", { class: "ct-win-resize", title: "drag to resize" });
	const win = el("div", { class: "ct-win ct-win-3d" }, head, body, grip);
	win.style.zIndex = String(++_winZ);
	document.body.appendChild(win);
	dragWin(win, head);
	resizeWin(win, grip);
	document.addEventListener("keydown", onKey, true);
	state.viewer3d.setActive(true);
	_win3d = { win, sub, controls, onKey, active: null };
	refresh3DWindow();
}
// Recompute shading (over the current selection's aggregate) + reload the scene,
// in place — so the popout tracks region/step changes without being recreated.
function refresh3DWindow() {
	if (!_win3d) return;
	const token = state.loadToken;
	const rows = state.rows || [];
	const sel = selectedSteps();
	if (!sel.length || !rows.length) { _win3d.sub.textContent = "no step in scope"; return; }
	const agg = aggregateRows(rows);
	_win3d.sub.textContent = sel.length === 1 ? `${sel[0].template ?? sel[0].step} · ${sel[0].node ?? ""}` : `${sel.length} steps`;

	const kindsPresent = ["zone", "frame", "object"].filter((k) => [...agg.ent].some(([id, v]) => entityKindLabel(v.kind, id) === k && v.score > 0));
	let active = _win3d.active;
	if (!active) {
		active = new Set(kindsPresent);
		try { const saved = (localStorage.getItem(KINDS_KEY) || "").split(",").filter(Boolean).filter((k) => kindsPresent.includes(k)); if (saved.length) active = new Set(saved); } catch { /* ignore */ }
		_win3d.active = active;
	}
	const shadeItems = () => {
		const items = [...agg.ent].filter(([id, v]) => v.score > 0 && active.has(entityKindLabel(v.kind, id)) && (state.viewer3d?.hasBbox?.(id) ?? true));
		const max = Math.max(1e-9, ...items.map(([, v]) => v.score));
		return items.map(([id, v]) => ({ id, weight: v.score / max }));
	};
	const applyBase = () => state.viewer3d?.setAttnHighlight?.(shadeItems(), { gamma: 2.0, minWeight: ctMinW, contrast: true });
	state.applyBaseHighlight = applyBase;

	const minwLab = el("span", { class: "ct-minw-lab", text: `${Math.round(ctMinW * 100)}%` });
	const minw = el("div", { class: "ct-minw" }, el("span", { text: "min attn" }),
		el("input", { type: "range", min: "0", max: "0.9", step: "0.01", value: String(ctMinW), title: "hide entities below this fraction of the peak attention",
			oninput: (e) => { ctMinW = Number(e.target.value); try { localStorage.setItem(MINW_KEY, String(ctMinW)); } catch { /* ignore */ } minwLab.textContent = `${Math.round(ctMinW * 100)}%`; applyBase(); } }),
		minwLab);
	const kindBtns = kindsPresent.map((k) => {
		const b = el("button", { class: `ct-kind${active.has(k) ? " on" : ""}`, title: `toggle ${k}s` }, el("i", { style: `background:${COLORS[k] || "#888"}` }), el("span", { text: k }));
		b.onclick = () => { if (active.has(k)) active.delete(k); else active.add(k); b.classList.toggle("on", active.has(k)); try { localStorage.setItem(KINDS_KEY, [...active].join(",")); } catch { /* ignore */ } applyBase(); };
		return b;
	});
	_win3d.controls.replaceChildren(
		el("div", { class: "ct-3d-legend" }, el("span", { text: "attention" }), el("span", { class: "muted", text: "low" }), el("span", { class: "ct-3d-bar" }), el("span", { class: "muted", text: "high" })),
		minw,
		kindsPresent.length > 1 ? el("div", { class: "ct-minw" }, el("span", { text: "types" }), ...kindBtns) : null);

	// load the most-built scene in scope, then shade it
	const renderUntil = Math.max(...sel.map((s) => (s.render_until != null ? s.render_until : -1)));
	const opts = renderUntil >= 0 ? { untilIndex: renderUntil } : {};
	state.viewer3d.setActive(true);
	state.viewer3d.clear();
	api.scene(state.run, state.slot, state.model, opts).then((proj) => {
		if (token !== state.loadToken || !_win3d) return;
		applySceneProjection(state.viewer3d, proj);
		state.viewer3d.prefetchBundle(api.meshesUrl(state.run, state.slot, state.model, opts));
		applyBase();
	}).catch(() => { /* non-fatal */ });
}
function resizeWin(win, grip) {
	grip.addEventListener("pointerdown", (e) => {
		const r = win.getBoundingClientRect();
		const sx = e.clientX, sy = e.clientY, ow = r.width, oh = r.height;
		const move = (ev) => { win.style.width = `${Math.max(300, ow + (ev.clientX - sx))}px`; win.style.height = `${Math.max(240, oh + (ev.clientY - sy))}px`; };
		const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
		window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
		e.preventDefault(); e.stopPropagation();
	});
}

// --- story: zone plan + output + reasoning ----------------------------------

function planBlock(step, a, targetZone) {
	const node = state.obs?.nodes?.get(targetZone);
	if (!node) return null;
	const isPlanStep = /(^|_)zone_plan/.test(step.template ?? step.step ?? "");
	const text = isPlanStep ? (node.prompt || node.plan || "") : (node.plan || node.prompt || "");
	if (!text || !String(text).trim()) return null;
	return el("div", { style: "margin-bottom:14px" },
		el("div", { class: "ct-plan-lab" }, el("span", { class: "sw", style: `background:${entityHex(node.kind || "zone", targetZone)}` }), el("span", { text: ` ${isPlanStep ? "zone prompt" : "zone plan"} · ${targetZone}` })),
		clampProse(planHighlightNodes(String(text), a)));
}
function outputBlock(a, llm) {
	const out = llm?.output;
	if (out == null || !String(out).trim()) return null;
	const pre = el("pre", {}); for (const n of fmtOutputNodes(String(out), entityCtx(a))) pre.appendChild(n);
	return el("div", { class: "ct-out" }, el("div", { class: "ct-plan-lab", text: "output" }), el("div", { class: "ct-code" }, pre));
}
function reasoningNodes(a, llm) {
	const r = llm?.reasoning;
	if (r == null || !String(r).trim()) return null;
	return highlightNodes(String(r), entityCtx(a));
}

// --- detached (drag-out) windows --------------------------------------------

const _wins = new Map(); // key -> { win, body, sub, build }
let _winZ = 80;
function subLabel(step) { return `${step.template ?? step.step ?? "?"} · ${step.node ?? ""}`; }

function openWindow(key, title, step, build) {
	const existing = _wins.get(key);
	if (existing) { existing.build = build; existing.body.replaceChildren(build()); existing.sub.textContent = subLabel(step); existing.win.style.zIndex = String(++_winZ); return; }
	const body = el("div", { class: "ct-win-body" }, build());
	const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } };
	const close = () => { win.remove(); document.removeEventListener("keydown", onKey, true); _wins.delete(key); };
	const sub = el("span", { class: "ct-win-sub", text: subLabel(step) });
	const head = el("div", { class: "ct-win-head" }, el("span", { style: "color:var(--text-faint)", text: "⠿" }), el("span", { class: "ct-win-title", text: title }), sub, el("span", { style: "flex:1" }), el("button", { class: "ct-win-close", title: "close (Esc)", text: "×", onclick: close }));
	const win = el("div", { class: "ct-win" }, head, body);
	const off = _wins.size * 26; win.style.top = `${96 + off}px`; win.style.right = `${24 + off}px`; win.style.zIndex = String(++_winZ);
	document.body.appendChild(win);
	dragWin(win, head);
	document.addEventListener("keydown", onKey, true);
	_wins.set(key, { win, body, sub, build });
}
function refreshWindows(step) { for (const [, e] of _wins) { try { e.body.replaceChildren(e.build()); e.sub.textContent = subLabel(step); } catch { /* ignore */ } } }
function closeTextWindows() { for (const [, e] of _wins) e.win.remove(); _wins.clear(); }
export function closeContentWindows() { closeTextWindows(); close3DWindow(); }
function dragWin(win, handle) {
	handle.addEventListener("pointerdown", (e) => {
		if (e.target.closest("button")) return;
		const r = win.getBoundingClientRect();
		win.style.left = `${r.left}px`; win.style.top = `${r.top}px`; win.style.right = "auto";
		const sx = e.clientX, sy = e.clientY, ox = r.left, oy = r.top;
		const move = (ev) => {
			const w = win.offsetWidth;
			win.style.left = `${Math.min(window.innerWidth - 80, Math.max(80 - w, ox + (ev.clientX - sx)))}px`;
			win.style.top = `${Math.min(window.innerHeight - 36, Math.max(0, oy + (ev.clientY - sy)))}px`;
		};
		const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
		window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
		e.preventDefault();
	});
}

// --- orchestration -----------------------------------------------------------

export async function renderContent() {
	if (state.view !== "content") return;
	const host = $("content-inner");
	if (!host) return;
	const token = bumpLoad();
	ctHoverReset();

	const sel = selectedSteps();
	if (!sel.length) {
		state.viewer3d?.clearAttnHighlight?.();
		host.replaceChildren(card("content", "no computed step", emptyCard("no computed attention for this selection — pick a region + step with computed attention.")));
		return;
	}

	$("dv-loading").classList.add("on");
	if (!host.querySelector(".card")) host.replaceChildren(emptyCard("loading…", "⧗"));
	let rows;
	try { rows = await loadRows(token); } catch (e) { if (token === state.loadToken) host.replaceChildren(emptyCard(`failed: ${e.message}`)); $("dv-loading").classList.remove("on"); return; }
	if (token !== state.loadToken) return;
	$("dv-loading").classList.remove("on");
	if (!rows.length) { host.replaceChildren(emptyCard("no attention data for this selection")); return; }
	state.rows = rows;

	// The structure view works for ANY selection: a single step is the richest
	// (adds its zone plan + output + reasoning); a multi-step scope shows the
	// aggregate attention tree (mean over the steps).
	const single = rows.length === 1;
	const step = single ? (sel.find((s) => String(s.event_index) === String(rows[0].event_index)) || sel[0]) : null;
	const agg = aggregateRows(rows);
	const targetZone = single ? targetZoneOf(step) : (state.region !== ALL ? state.region : null);

	// toolbar: the 3D popout always; the per-step plan/output + reasoning popouts
	// only when the scope is a single step.
	const tools = el("div", { class: "ct-toolbar" },
		el("button", { class: "ct-btn", title: "open the scene shaded by this selection's attention in a draggable 3D window", text: "⤢ 3D scene", onclick: open3DWindow }));
	if (single) {
		tools.appendChild(el("button", { class: "ct-btn", title: "pop the zone plan + output into a draggable window", text: "⤢ plan · output", onclick: () => openWindow("planout", "zone plan · output", step, () => planOutBody(step)) }));
		tools.appendChild(el("button", { class: "ct-btn", title: "show this step's reasoning in a draggable popup", text: "🧠 reasoning", onclick: () => openWindow("reason", "reasoning", step, () => reasoningBody(step)) }));
	}

	const treeCard = card("attention tree",
		single ? "scene structure · this step's attention" : `scene structure · mean over ${rows.length} steps${state.region !== ALL ? ` · ${state.region}` : ""}`,
		buildTree(agg, targetZone, false));

	const cards = [treeCard];
	if (single) {
		const a = rows[0].a, llm = stepLLM(step.event_index);
		const plan = planBlock(step, a, targetZone);
		cards.push(card("step story", `${step.template ?? step.step} · ${step.node ?? ""}`, plan || el("div", { class: "muted", text: "no zone plan for this step" }), outputBlock(a, llm)));
	} else {
		cards.push(card("step story", `${rows.length} steps in scope`,
			el("div", { class: "muted", text: "the zone plan, output and reasoning are per-step — narrow to a single step (pick a region, then a specific step) to see them." })));
	}

	host.replaceChildren(tools, ...cards);
	if (single) refreshWindows(step); else closeTextWindows(); // per-step popouts don't apply to a multi-step scope
	if (_win3d) refresh3DWindow();
}

// Window body builders — recomputed from the CURRENT step on every refresh.
function currentStep() { const sel = selectedSteps(); return sel.length === 1 ? sel[0] : null; }
function currentRowA() {
	const step = currentStep(); if (!step) return null;
	const key = `${state.run}:${state.slot}:${state.model}:${step.event_index}:compact`;
	return state.aggCache.get(key) || (state.rows || []).find((r) => String(r.event_index) === String(step.event_index))?.a || null;
}
function planOutBody(step) {
	const a = currentRowA(); const s = currentStep() || step;
	if (!a) return el("div", { class: "muted", text: "loading…" });
	const plan = planBlock(s, a, targetZoneOf(s));
	return el("div", {}, plan || el("div", { class: "muted", text: "no zone plan" }), el("div", { style: "margin-top:10px" }, outputBlock(a, stepLLM(s.event_index)) || el("div", { class: "muted", text: "no output" })));
}
function reasoningBody(step) {
	const a = currentRowA(); const s = currentStep() || step;
	const nodes = a ? reasoningNodes(a, stepLLM(s.event_index)) : null;
	if (!nodes) return el("div", { class: "muted", text: "no reasoning captured for this step" });
	const pre = el("pre", { style: "margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.55 ui-monospace,Menlo,monospace" });
	for (const n of nodes) pre.appendChild(n);
	return pre;
}
