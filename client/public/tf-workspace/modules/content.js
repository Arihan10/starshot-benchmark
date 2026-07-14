// Content view — the per-step "structure" page as a RESIZABLE TILED workspace
// (replacing the old floating/drawable windows):
//
//   ┌───────────────┬───────────────────────────┐
//   │ Attention Tree │ 3D scene (attention-shaded)│
//   │                ├─────────────┬─────────────┤
//   │ Original       │ Zone plan   │ Attention ·  │
//   │ content        │ (upstream)  │ sorted list  │
//   └───────────────┴─────────────┴─────────────┘
//
// Tree badges, plan phrases, output/reasoning entities, and the sorted list all
// cross-highlight each other and the 3D scene (bidirectional, via ctHover). It is
// a SINGLE-step view for the per-step tiles (plan / output); the tree + 3D + list
// work for any selection (aggregate). Reuses the shared three.js viewer engine.

import { el } from "../../js/ui.js";
import { api } from "../../js/api.js";
import { createViewer } from "../../js/scene3d.js";
import { applySceneProjection, emittingRegion } from "../../js/events.js";
import {
	$, state, ALL, COLORS, COMPONENT_COLORS, entityHex, entityKindLabel,
	bumpLoad, ctHoverReset, ctHoverRegister,
} from "./state.js";
import { selectedSteps, loadRows, stepLLM, ensureEvents } from "./data.js";

// --- small helpers -----------------------------------------------------------

const fmtNum = (v) => (v >= 1 ? v.toFixed(1) : v >= 0.01 ? v.toFixed(3) : v.toFixed(4));
// legacy attention heat ramp (dark blue → bright yellow) — kept for the text spans.
const heat = (t) => { t = Math.max(0, Math.min(1, t)); return `hsl(${(212 - 162 * t).toFixed(0)}, 70%, ${(32 + 28 * t).toFixed(0)}%)`; };
// SHARED attention scale: low = yellow, high = red — used by the heat badges + the
// sorted list + the 3D wireframe overlay (via scene3d's hotRamp) so every
// attention read across the content view is consistent.
const attnColor = (t) => { t = Math.max(0, Math.min(1, t)); return `hsl(${(58 - 58 * t).toFixed(0)}, ${(86 + 8 * t).toFixed(0)}%, ${(55 - 6 * t).toFixed(0)}%)`; };
function alpha(hex, a) {
	const h = String(hex).replace("#", "");
	const n = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
	const r = parseInt(n.slice(0, 2), 16) || 0, g = parseInt(n.slice(2, 4), 16) || 0, b = parseInt(n.slice(4, 6), 16) || 0;
	return `rgba(${r},${g},${b},${a})`;
}
const emptyCard = (msg, big = "▦") => el("div", { class: "empty" }, el("span", { class: "big", text: big }), el("div", { text: msg }));

// The zone a step operates in: its own node when that's a zone, else the region
// that emitted it (a bbox step can run on a peer object).
function targetZoneOf(step) {
	if (!step || !step.node) return null;
	const n = state.obs?.nodes?.get(step.node);
	if (n && n.kind !== "zone") { const r = emittingRegion(state.obs, step.node); if (r) return r; }
	return step.node;
}

// The nearest ANCESTOR zone of `zoneId` (the upstream parent it belongs to), or
// null at the root. Used so the zone-plan tile shows the governing parent plan.
function parentZoneOf(zoneId) {
	const m = state.obs;
	if (!m || !zoneId) return null;
	let cur = m.nodes.get(zoneId);
	cur = cur?.parentId ? m.nodes.get(cur.parentId) : null;
	let hops = 0;
	while (cur && hops < 64) {
		if (cur.kind === "zone") return cur.id;
		cur = cur.parentId ? m.nodes.get(cur.parentId) : null;
		hops += 1;
	}
	return null;
}

// id → { score, kind } and the peak, from a single step's scene aggregate.
function entityScores(a) {
	const ent = new Map(); let max = 0;
	for (const e of (((a.agg || {}).scene || {}).entityTotals || [])) { const sc = e.score || 0; ent.set(e.id, { score: sc, kind: e.kind }); if (sc > max) max = sc; }
	return { ent, max };
}

// Mean per-entity attention across a multi-step selection (absent counts as 0) —
// the aggregate the tree + 3D shading use when more than one step is in scope.
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
		// A coloured RIGHT edge encodes the entity KIND (zone / object / frame),
		// so kinds stay distinguishable even in heat mode where the badge body is
		// attention-coloured. (In width mode this matches the body colour.)
		const rightHex = isRoot ? "#9aa7bd" : entityHex(n?.kind, id);
		let style, cls;
		if (heatMode) {
			const hot = attnColor(t); // yellow (low) → red (high) — the shared scale
			style = `background:${isRoot ? "#464d5b" : hot};border-color:${isRoot ? "#6b7280" : hot};color:${isRoot ? "#e8eefc" : (t > 0.5 ? "#fff" : "#2a1e08")};border-right:3px solid ${rightHex}`;
			cls = `ct-badge heat${zone ? " zone" : ""}${isRoot ? " root" : ""}${isTarget ? " target" : ""}`;
		} else {
			const hex = isRoot ? "#9aa7bd" : entityHex(n?.kind, id);
			const wpct = isRoot ? FILL : (FILL * own(id)) / maxRow;
			style = `width:${wpct.toFixed(3)}%;background:${hex}2e;border-color:${hex};border-right:3px solid ${rightHex}`;
			cls = `ct-badge${zone ? " zone" : ""}${isRoot ? " root" : ""}${isTarget ? " target" : ""}`;
		}
		const b = el("div", {
			class: cls, style,
			title: `${id} · ${kindLab}${reg ? ` · in ${reg}` : ""}${isTarget ? " · current step's zone" : ""} · attention ${fmtNum(own(id))}`,
			onclick: () => focusEntity(id),
		}, el("span", { class: "ct-badge-lab", text: id }));
		if (!heatMode) { b.dataset.id = id; b.dataset.rel = isRoot ? "1" : t.toFixed(4); } // for the leader-arm label layout
		ctHoverRegister(id, b);
		return b;
	};

	// Width mode gives each ZONE its own full-width row; OBJECTS + FRAMES share a
	// row (clipped names drop to a label lane below, joined by a leader arm). Heat
	// badges are uniform width, so their names always fit.
	const treeView = (heatMode) => {
		const view = el("div", { class: `ct-tree-view${heatMode ? " ct-heat" : ""}` });
		for (const r of rows) {
			const indent = `margin-left:${r.depth * INDENT}px`;
			const rowEl = el("div", { class: `ct-row${r.zone ? " zone-row" : " leaf-row"}`, style: (heatMode || r.zone) ? indent : "" },
				...r.ids.map((id) => badge(id, r.zone, heatMode)));
			if (heatMode) { view.appendChild(rowEl); continue; }
			if (r.zone) {
				for (const id of r.ids) rowEl.appendChild(el("span", { class: "ct-zone-name-ext", text: id, title: id, onclick: () => focusEntity(id) }));
				view.appendChild(rowEl);
			} else {
				const lane = el("div", { class: "ct-labels" });
				const arms = document.createElementNS(SVGNS, "svg"); arms.setAttribute("class", "ct-arms");
				const group = el("div", { class: "ct-rowgroup", style: indent }, rowEl, lane);
				group.appendChild(arms);
				view.appendChild(group);
			}
		}
		return view;
	};

	const legend = el("div", { class: "ct-tree-legend" },
		...[["root", "#9aa7bd"], ["zone", COLORS.zone], ["frame", COLORS.frame], ["object", COLORS.object]]
			.map(([k, hex]) => el("span", {}, el("i", { style: `background:${hex}` }), el("span", { text: ` ${k}` }))));

	const sub = (capText, heatMode) => el("div", {}, el("div", { class: "ct-tree-cap", text: capText }), treeView(heatMode));
	const treeRoot = el("div", { class: "ct-tree" },
		...(heatOnly ? [] : [sub("badge width = attention · right edge = kind · hover for full name · click → focus", false)]),
		sub("badge colour = attention (yellow→red) · right edge = kind", true),
		legend);
	if (!heatOnly) fitTreeLabels(treeRoot);
	return treeRoot;
}

const SVGNS = "http://www.w3.org/2000/svg";
// Leader-arm label layout tunables (mirrors the tf overview tree).
const _LBL = { H: 15, DROP: 8, GAP: 3, LGAP: 8, PAD: 12, MAX: 14, REL_MIN: 0.13 };
let _measCtx = null;

// Place below-row labels + leader arms for ONE leaf rowgroup (objects/frames share
// a row). A badge shows its name inline only when the text fits its measured width;
// otherwise — if the entity is relevant enough — the name moves to the label lane
// below and a 3-segment arm points from the badge down to it. Idempotent.
function layoutLeafGroup(group) {
	const rowEl = group.querySelector(".ct-row");
	const lane = group.querySelector(".ct-labels");
	const arms = group.querySelector(".ct-arms");
	if (!rowEl || !lane || !arms) return;
	lane.replaceChildren();
	while (arms.firstChild) arms.removeChild(arms.firstChild);
	const badges = [...rowEl.children];
	if (!badges.length) return;
	for (const b of badges) { // clear stale glow bindings (badges persist across relayouts)
		if (b._glowOn) { b.removeEventListener("mouseenter", b._glowOn); b.removeEventListener("mouseleave", b._glowOff); b._glowOn = b._glowOff = null; }
		b.classList.remove("ct-hot");
	}
	_measCtx = _measCtx || document.createElement("canvas").getContext("2d");
	const cs = getComputedStyle(badges[0]);
	_measCtx.font = `${cs.fontSize} ${cs.fontFamily}`;
	let shorts = [];
	for (const b of badges) {
		const fits = _measCtx.measureText(b.dataset.id || "").width <= b.clientWidth - _LBL.PAD - 2;
		b.classList.toggle("lab-out", !fits);
		if (!fits && Number(b.dataset.rel) >= _LBL.REL_MIN) shorts.push(b);
	}
	if (shorts.length > _LBL.MAX) shorts = shorts.sort((a, z) => Number(z.dataset.rel) - Number(a.dataset.rel)).slice(0, _LBL.MAX);
	if (!shorts.length) { lane.style.height = "0px"; arms.style.display = "none"; return; }
	arms.style.display = "";
	const groupW = group.clientWidth;
	const items = shorts
		.map((b) => { const d = lane.appendChild(el("div", { class: "ct-label", text: b.dataset.id, title: b.dataset.id, onclick: () => focusEntity(b.dataset.id) })); d.dataset.id = b.dataset.id; return { b, d }; })
		.map((it) => ({ ...it, bc: it.b.offsetLeft + it.b.offsetWidth / 2, lw: it.d.offsetWidth }))
		.sort((a, z) => a.bc - z.bc);
	for (const it of items) { // hovering a badge OR its label glows both + the arm
		const on = () => { it.b.classList.add("ct-hot"); it.d.classList.add("ct-hot"); it.p?.classList.add("hot"); };
		const off = () => { it.b.classList.remove("ct-hot"); it.d.classList.remove("ct-hot"); it.p?.classList.remove("hot"); };
		it.b._glowOn = on; it.b._glowOff = off;
		it.b.addEventListener("mouseenter", on); it.b.addEventListener("mouseleave", off);
		it.d.addEventListener("mouseenter", on); it.d.addEventListener("mouseleave", off);
	}
	const levelRight = [];
	for (const it of items) {
		const px = Math.max(0, Math.min(it.bc - it.lw / 2, Math.max(0, groupW - it.lw)));
		let lvl = levelRight.findIndex((r) => px >= r + _LBL.LGAP);
		if (lvl === -1) { lvl = levelRight.length; levelRight.push(0); }
		levelRight[lvl] = px + it.lw;
		it.x = px; it.lvl = lvl;
		it.d.style.left = `${px}px`;
		it.d.style.top = `${_LBL.DROP + lvl * (_LBL.H + _LBL.GAP)}px`;
	}
	const used = levelRight.length || 1;
	lane.style.height = `${_LBL.DROP + used * (_LBL.H + _LBL.GAP) + 2}px`;
	const gW = group.clientWidth, gH = group.clientHeight;
	arms.setAttribute("width", gW); arms.setAttribute("height", gH);
	arms.setAttribute("viewBox", `0 0 ${gW} ${gH}`);
	const by = rowEl.offsetHeight;
	for (const it of items) {
		const lcx = it.x + it.lw / 2;
		const ly = lane.offsetTop + _LBL.DROP + it.lvl * (_LBL.H + _LBL.GAP);
		const mid = by + Math.max(3, (ly - by) / 2);
		const p = document.createElementNS(SVGNS, "path");
		p.setAttribute("d", `M${it.bc.toFixed(1)} ${by}L${it.bc.toFixed(1)} ${mid.toFixed(1)}L${lcx.toFixed(1)} ${mid.toFixed(1)}L${lcx.toFixed(1)} ${ly.toFixed(1)}`);
		p.setAttribute("fill", "none"); p.setAttribute("stroke", "rgba(255,255,255,0.26)"); p.setAttribute("stroke-width", "1");
		arms.appendChild(p); it.p = p; // link so a badge/label hover can glow it too
	}
}

// Post-layout label pass for the width tree; re-runs when its WIDTH changes.
function fitTreeLabels(root) {
	let lastW = -1;
	const run = () => {
		for (const row of root.querySelectorAll(".ct-tree-view:not(.ct-heat) .ct-row.zone-row")) {
			const lab = row.querySelector(".ct-badge.zone .ct-badge-lab");
			if (lab) row.classList.toggle("show-ext", lab.scrollWidth - lab.clientWidth > 1);
		}
		for (const g of root.querySelectorAll(".ct-tree-view:not(.ct-heat) .ct-rowgroup")) layoutLeafGroup(g);
	};
	requestAnimationFrame(() => { lastW = Math.round(root.clientWidth); run(); });
	try { new ResizeObserver(() => { const w = Math.round(root.clientWidth); if (w !== lastW) { lastW = w; run(); } }).observe(root); } catch { /* ignore */ }
}

// --- 3D viewer (created once, reused; mounted inline into the 3D tile) --------

const KINDS_KEY = "tf-ct-kinds", TOPFRAC_KEY = "tf-ct-topfrac";
// Attention-threshold cutoff as a FRACTION of the ranked entities shown (1 = all).
// The slider steps per-object over this ranking (feature: deterministic control).
let ctTopFrac = (() => { try { const v = Number(localStorage.getItem(TOPFRAC_KEY)); return v > 0 && v <= 1 ? v : 1; } catch { return 1; } })();
let ctKinds = null; // Set of shown entity kinds (persisted); (re)initialised per render
// "Post-zone" view: extend the render forward to everything placed up to (but
// before) the NEXT zone plan, so a step's downstream effects are visible; the
// entities produced in that window are highlighted BLUE. Single-step only.
const POSTZONE_KEY = "tf-ct-postzone";
let ctPostZone = (() => { try { return localStorage.getItem(POSTZONE_KEY) === "1"; } catch { return false; } })();
// Blue "focus" highlight: the current zone (and, in post-zone mode, the step's
// results) are drawn in flashing blue and every OTHER wireframe is dimmed. On by
// default; toggleable since the flash/dim can distract during close inspection.
const HL_KEY = "tf-ct-highlight";
let ctHighlight = (() => { try { return localStorage.getItem(HL_KEY) !== "0"; } catch { return true; } })();
const _isZonePlan = (t) => /(^|_)zone_plan/.test(t || "");
// The event index of the NEXT zone-plan step after `step` — the boundary the
// post-zone view stops just before. null = this is the last zone, so render to the
// end of the scene.
function nextZonePlanIndex(step) {
	let best = null;
	for (const s of state.steps || []) {
		const ev = s.event_index;
		if (ev == null || ev <= step.event_index) continue;
		if (!_isZonePlan(s.template ?? s.step)) continue;
		if (best == null || ev < best) best = ev;
	}
	return best;
}

function ensureViewer() {
	if (state.viewer3d && state._ctHost) return true;
	try {
		state._ctHost = el("div", { class: "ct-canvas-host" });
		state.viewer3d = createViewer(state._ctHost, { keyboard: true, lighting: true });
		return true;
	} catch { state.viewer3d = null; state._ctHost = null; return false; }
}

// Focus an entity in 3D: frame the camera on it + mark it strong green, without
// collapsing the attention overlay (every other entity's attention stays put).
function focusEntity(id) {
	state.viewer3d?.select?.(id, { frame: true });
	state.viewer3d?.setHoverHighlight?.(id);
}

// The kinds present (with attention) in the current aggregate.
function kindsPresentOf(agg) {
	return ["zone", "frame", "object"].filter((k) => [...agg.ent].some(([id, v]) => entityKindLabel(v.kind, id) === k && v.score > 0));
}

// Load the most-built scene in scope + shade it by attention, and (re)build the
// inline controls (per-object threshold slider · type toggles · hide-meshes ·
// post-zone). `controls` is the controls container INSIDE the 3D tile body;
// `step`/`single` describe the current selection (post-zone needs a single step).
function refresh3DTile(controls, agg, step, single, targetZone) {
	if (!state.viewer3d) return;
	const token = state.loadToken;
	const sel = selectedSteps();
	const rows = state.rows || [];

	const kindsPresent = kindsPresentOf(agg);
	if (!ctKinds) {
		ctKinds = new Set(kindsPresent);
		try { const saved = (localStorage.getItem(KINDS_KEY) || "").split(",").filter(Boolean).filter((k) => kindsPresent.includes(k)); if (saved.length) ctKinds = new Set(saved); } catch { /* ignore */ }
	}

	// Ranked, kind-filtered, present-in-scene entities (highest attention first).
	// The current zone is excluded while the blue focus highlight is on — it's
	// drawn in blue, so keeping it out of the attention overlay (and its colour
	// scale + slider count) stops the two highlights stacking + conflicting.
	const ranked = () => [...agg.ent]
		.filter(([id, v]) => v.score > 0 && ctKinds.has(entityKindLabel(v.kind, id)) && (state.viewer3d?.hasBbox?.(id) ?? true) && !(ctHighlight && id === targetZone))
		.map(([id, v]) => ({ id, score: v.score }))
		.sort((a, b) => b.score - a.score);

	// Shade: LOG attention scale — map each score's log between the min/max
	// attended present, so differences at the (crowded) low end stay legible and
	// the max attended maps to full intensity. Show the top-K by the per-object
	// threshold, colour yellow→red via the shared hot ramp.
	const applyShade = () => {
		const items = ranked();
		const total = items.length;
		const scores = items.map((i) => i.score).filter((s) => s > 0);
		const lmax = scores.length ? Math.log(Math.max(...scores)) : 0;
		const lmin = scores.length ? Math.log(Math.min(...scores)) : 0;
		const span = lmax - lmin;
		const flat = !(span > 1e-6); // one item / all-equal → everything full
		const wOf = (s) => (flat || s <= 0) ? 1 : Math.max(0, Math.min(1, (Math.log(s) - lmin) / span));
		const k = Math.max(1, Math.min(total || 1, Math.round(ctTopFrac * (total || 1)) || (total || 1)));
		const shown = items.slice(0, k).map((i) => ({ id: i.id, weight: wOf(i.score) }));
		// gamma 1: the log mapping already shapes the scale; the hot ramp then
		// makes higher (log-)attention read redder + more opaque.
		state.viewer3d.setAttnHighlight(shown, { gamma: 1, hotRamp: true });
		return { total, k };
	};
	state.applyBaseHighlight = applyShade;

	// Entity-type toggles drive the viewer's category visibility too, so turning
	// a kind off hides its WIREFRAMES (and meshes), not just its attention overlay.
	// A kind with no toggle button (not present with attention) stays visible, so a
	// stale off-state can't hide wireframes the user has no control to bring back.
	const applyKindVis = () => {
		const t = state.viewer3d?.toggles;
		if (!t) return;
		const on = (k) => !kindsPresent.includes(k) || ctKinds.has(k);
		t.zones = on("zone");
		t.frames = on("frame");
		const obj = on("object");
		t.anchors = obj; t.next = obj; t.negativeSpace = obj;
		state.viewer3d.refreshVisibility?.();
	};

	// Blue focus highlight, kept in closure vars so the toggle can re-apply without
	// re-fetching the scene (`loadScene` fills them). The current/target zone gets the
	// PRONOUNCED style; the post-zone step results keep the ORIGINAL (subtler) blue.
	let hlZone = [];      // current/target zone → pronounced
	let hlResults = [];   // post-zone step results → original look
	const applyHL = () => {
		if (!ctHighlight || (!hlZone.length && !hlResults.length)) { state.viewer3d.clearResultHighlight?.(); return; }
		state.viewer3d.setResultHighlight?.(hlZone, { flash: true, dimOthers: false }); // pronounced blue flash; other outlines stay fully opaque
		if (hlResults.length) state.viewer3d.addResultHighlight?.(hlResults, { flash: true }); // original post-zone look (appends)
	};

	const buildControls = () => {
		const items = ranked();
		const total = items.length;
		const k = Math.max(1, Math.min(total || 1, Math.round(ctTopFrac * (total || 1)) || (total || 1)));
		// per-object threshold slider: one step = one entity in/out of the view.
		const slLab = el("span", { class: "ct-minw-lab", text: total ? `top ${k}/${total}` : "—" });
		const slider = el("input", {
			type: "range", min: "1", max: String(Math.max(1, total)), step: "1", value: String(k),
			title: "attention threshold — each step adds/removes one entity (ranked by attention)",
			...(total ? {} : { disabled: "" }),
			oninput: (e) => {
				const v = Number(e.target.value);
				ctTopFrac = total ? v / total : 1;
				try { localStorage.setItem(TOPFRAC_KEY, String(ctTopFrac)); } catch { /* ignore */ }
				slLab.textContent = `top ${v}/${total}`;
				applyShade();
			},
		});
		const sl = el("div", { class: "ct-minw" }, el("span", { text: "show" }), slider, slLab);
		// entity-type toggles
		const kindBtns = kindsPresent.map((kk) => {
			const b = el("button", { class: `ct-kind${ctKinds.has(kk) ? " on" : ""}`, title: `toggle ${kk}s` }, el("i", { style: `background:${COLORS[kk] || "#888"}` }), el("span", { text: kk }));
			b.onclick = () => {
				if (ctKinds.has(kk)) ctKinds.delete(kk); else ctKinds.add(kk);
				if (!ctKinds.size) ctKinds.add(kk); // never empty
				b.classList.toggle("on", ctKinds.has(kk));
				try { localStorage.setItem(KINDS_KEY, [...ctKinds].join(",")); } catch { /* ignore */ }
				applyKindVis(); // hide/show this kind's wireframes + meshes too
				buildControls(); applyShade(); // total changed → rebuild the slider bounds
			};
			return b;
		});
		// hide / unhide ALL meshes (matches the main client's control)
		const meshesOn = state.viewer3d?.getVisibility?.().meshes !== false;
		const hideBtn = el("button", { class: `ct-kind${meshesOn ? " on" : ""}`, title: "hide / show every mesh (the wireframe boxes stay)" },
			el("span", { text: meshesOn ? "◧ meshes" : "▢ meshes" }));
		hideBtn.onclick = () => {
			const v = state.viewer3d.getVisibility().meshes === false; // → turning ON
			state.viewer3d.setMeshesVisible(v);
			hideBtn.classList.toggle("on", v);
			hideBtn.replaceChildren(el("span", { text: v ? "◧ meshes" : "▢ meshes" }));
		};
		// POST-ZONE toggle (single step only): extend the render to before the next
		// zone plan, with this step's results highlighted blue.
		const postBtn = single ? el("button", { class: `ct-kind${ctPostZone ? " on" : ""}`, title: "post-zone: show everything placed up to (before) the NEXT zone plan — this step's downstream effects; its results are highlighted blue" },
			el("span", { text: `${ctPostZone ? "◉" : "○"} post-zone` })) : null;
		if (postBtn) postBtn.onclick = () => {
			ctPostZone = !ctPostZone;
			try { localStorage.setItem(POSTZONE_KEY, ctPostZone ? "1" : "0"); } catch { /* ignore */ }
			postBtn.classList.toggle("on", ctPostZone);
			postBtn.replaceChildren(el("span", { text: `${ctPostZone ? "◉" : "○"} post-zone` }));
			loadScene();
		};
		// BLUE highlight toggle: current zone (+ post-zone results) flash in blue,
		// other wireframes dim. On by default.
		const hlBtn = el("button", { class: `ct-kind${ctHighlight ? " on" : ""}`, title: "highlight the current zone (and step results) in flashing blue; dim the other wireframes" },
			el("span", { text: `${ctHighlight ? "◉" : "○"} highlight` }));
		hlBtn.onclick = () => {
			ctHighlight = !ctHighlight;
			try { localStorage.setItem(HL_KEY, ctHighlight ? "1" : "0"); } catch { /* ignore */ }
			hlBtn.classList.toggle("on", ctHighlight);
			hlBtn.replaceChildren(el("span", { text: `${ctHighlight ? "◉" : "○"} highlight` }));
			applyHL();
			buildControls(); applyShade(); // current zone enters/leaves the attention overlay
		};
		const blueSw = el("span", { style: "display:inline-block;width:12px;height:10px;border-radius:2px;background:#3d8bff;border:1px solid rgba(255,255,255,0.25)" });
		const blueLabel = (ctPostZone && single) ? "current zone · step results" : "current zone";
		controls.replaceChildren(
			el("div", { class: "ct-3d-legend" }, el("span", { text: "attention (log)" }), el("span", { class: "muted", text: "low" }), el("span", { class: "ct-3d-bar" }), el("span", { class: "muted", text: "high" })),
			ctHighlight ? el("div", { class: "ct-3d-legend" }, blueSw, el("span", { class: "muted", text: blueLabel })) : null,
			sl,
			kindsPresent.length > 1 ? el("div", { class: "ct-minw" }, el("span", { text: "types" }), ...kindBtns) : null,
			el("div", { class: "ct-minw" }, hideBtn, hlBtn, ...(postBtn ? [postBtn] : [])),
		);
	};

	// Load the scene (normal: up to render_until; post-zone: up to before the next
	// zone plan) + shade attention, and — in post-zone mode — BLUE-highlight the
	// entities produced as this step's results (present at the boundary but not
	// before the step ran).
	const loadScene = () => {
		if (!sel.length || !rows.length) return;
		const renderUntil = Math.max(...sel.map((s) => (s.render_until != null ? s.render_until : -1)));
		let untilIndex = renderUntil >= 0 ? renderUntil : null;
		let beforeIdx = null;
		const postMode = ctPostZone && single && step;
		if (postMode) {
			untilIndex = nextZonePlanIndex(step); // null → render to the end of the scene
			beforeIdx = step.event_index;          // the scene BEFORE this step's products
		}
		const opts = untilIndex != null ? { untilIndex } : {};
		state.viewer3d.setActive(true);
		state.viewer3d.clear(); // also clears the result highlight
		const scenes = [api.scene(state.run, state.slot, state.model, opts)];
		if (postMode) scenes.push(api.scene(state.run, state.slot, state.model, { untilIndex: beforeIdx }));
		Promise.all(scenes).then(([proj, before]) => {
			if (token !== state.loadToken || state.view !== "content") return;
			applySceneProjection(state.viewer3d, proj);
			state.viewer3d.prefetchBundle(api.meshesUrl(state.run, state.slot, state.model, opts));
			applyKindVis(); // reflect the type toggles on the freshly-built wireframes/meshes
			buildControls(); // bboxes now exist → accurate per-object slider bounds / toggles
			applyShade();
			// Blue focus highlight: the current zone always (pronounced), plus this
			// step's results in post-zone mode (renderable at the boundary but not
			// before it ran) — kept as a SEPARATE group so they retain the original look.
			hlZone = (targetZone && (state.viewer3d.hasBbox?.(targetZone) ?? false)) ? [targetZone] : [];
			const results = new Set();
			if (postMode && before) {
				const boxed = (nodes) => (nodes || []).filter((n) => Array.isArray(n.origin) && Array.isArray(n.dimensions)).map((n) => n.id);
				const had = new Set(boxed(before.nodes));
				for (const id of boxed(proj.nodes)) if (!had.has(id) && id !== targetZone) results.add(id);
			}
			hlResults = [...results];
			applyHL();
		}).catch(() => { /* non-fatal */ });
	};

	buildControls();
	loadScene();
}

// --- story: zone plan + output + reasoning ----------------------------------

function planProseFrom(output) {
	let o = output;
	if (typeof o === "string") { const s = o.trim(); if (!s) return null; try { o = JSON.parse(s); } catch { return s; } }
	if (o && typeof o === "object") return typeof o.plan === "string" ? o.plan : typeof o.description === "string" ? o.description : null;
	return null;
}
// The active zone's plan text, recovered from the event log: the zone's OWN
// zone_plan output, else the prompt the parent zone_decompose assigned it.
function zonePlanText(zoneId) {
	if (!zoneId) return null;
	const evs = state.events || [];
	for (const e of evs) {
		if (e?.kind !== "cache.llm" || e.node !== zoneId) continue;
		if (/(^|_)zone_plan/.test(e.step ?? e.template ?? "")) { const t = planProseFrom(e.output); if (t && t.trim()) return t; }
	}
	for (const e of evs) {
		if (e?.kind !== "cache.llm" || !/zone_decompose/.test(e.step ?? e.template ?? "")) continue;
		let o = e.output; if (typeof o === "string") { try { o = JSON.parse(o); } catch { o = null; } }
		const hit = (o?.subregions || []).find((s) => s && s.id === zoneId);
		if (hit && typeof hit.prompt === "string" && hit.prompt.trim()) return hit.prompt;
	}
	return null;
}
// A labelled plan block for one zone id.
function planFor(zoneId, a, capText) {
	const text = zonePlanText(zoneId);
	if (!text || !String(text).trim()) return null;
	const node = state.obs?.nodes?.get(zoneId);
	return el("div", { style: "margin-bottom:14px" },
		el("div", { class: "ct-plan-lab" }, el("span", { class: "sw", style: `background:${entityHex(node?.kind || "zone", zoneId || "zone")}` }), el("span", { text: ` ${capText}${zoneId ? " · " + zoneId : ""}` })),
		clampProse(planHighlightNodes(String(text), a)));
}
// The zone-plan tile body. Feature: when viewing a SUBZONE, show the UPSTREAM
// PARENT zone plan it belongs to (the governing context), then the subzone's own
// plan below. At the root there's no parent, so just the local plan.
function zonePlanBody(a, targetZone) {
	if (!targetZone) return el("div", { class: "muted", text: "no zone in scope for this step" });
	const parent = parentZoneOf(targetZone);
	const blocks = [];
	if (parent) {
		blocks.push(planFor(parent, a, "upstream zone plan"));
		blocks.push(planFor(targetZone, a, "this subzone"));
	} else {
		blocks.push(planFor(targetZone, a, "zone plan"));
	}
	const kept = blocks.filter(Boolean);
	if (!kept.length) return el("div", { class: "muted", text: `no zone plan recovered for ${parent || targetZone}` });
	return el("div", {}, ...kept);
}

// The step's structured output, pretty-printed, entity ids highlighted.
function outputBlock(a, llm) {
	const out = llm?.output;
	if (out == null) return null;
	let text;
	if (typeof out === "string") text = out;
	else { try { text = JSON.stringify(out, null, 2); } catch { text = String(out); } }
	if (!text || !text.trim()) return null;
	const pre = el("pre", {}); for (const n of fmtOutputNodes(text, entityCtx(a))) pre.appendChild(n);
	return el("div", { class: "ct-out" }, el("div", { class: "ct-plan-lab", text: "output" }), el("div", { class: "ct-code" }, pre));
}
function reasoningBlock(a, llm) {
	const r = llm?.reasoning;
	if (r == null || !String(r).trim()) return null;
	const pre = el("pre", { style: "margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.55 ui-monospace,Menlo,monospace" });
	for (const n of highlightNodes(String(r), entityCtx(a))) pre.appendChild(n);
	return el("div", { class: "ct-out" }, el("div", { class: "ct-plan-lab", text: "reasoning" }), el("div", { class: "ct-code" }, pre));
}
// "Original content" tile: the step's own generated output + reasoning.
function originalContentBody(single, step, rows) {
	if (!single) return el("div", { class: "muted", text: "output + reasoning are per-step — narrow to a single step (pick a region, then a step) to see them." });
	const a = rows[0].a, llm = stepLLM(step.event_index);
	const parts = [outputBlock(a, llm), reasoningBlock(a, llm)].filter(Boolean);
	if (!parts.length) return el("div", { class: "muted", text: "no output or reasoning captured for this step" });
	return el("div", {}, ...parts);
}

// --- attention-sorted list tile ---------------------------------------------

let ctSortTab = "all"; // all | zone | object | frame
function sortedListBody(agg) {
	const wrap = el("div", {});
	const render = () => {
		const items = [...agg.ent]
			.map(([id, v]) => ({ id, score: v.score, kind: entityKindLabel(v.kind, id) }))
			.filter((i) => i.score > 0 && (ctSortTab === "all" || i.kind === ctSortTab))
			.sort((a, b) => b.score - a.score);
		const tabs = el("div", { class: "ct-sort-tabs" },
			...["all", "zone", "object", "frame"].map((t) => el("button", { class: `ct-sort-tab${t === ctSortTab ? " on" : ""}`, text: t, onclick: () => { ctSortTab = t; render(); } })));
		const list = el("div", { class: "ct-sort-list" }, ...items.map((it, i) => {
			const row = el("div", { class: "ct-sort-row", title: `${it.id} · ${it.kind} · attention ${it.score.toFixed(4)} — hover to locate · click → focus`, onclick: () => focusEntity(it.id) },
				el("span", { class: "ct-sort-rank", text: String(i + 1) }),
				el("span", { class: "ct-sort-sw", style: `background:${entityHex(it.kind, it.id)}` }),
				el("span", { class: "ct-sort-nm", text: it.id }),
				el("span", { class: "ct-sort-val", text: it.score.toFixed(3) }));
			ctHoverRegister(it.id, row);
			return row;
		}));
		wrap.replaceChildren(tabs, items.length ? list : el("div", { class: "muted", style: "font-size:12px", text: "no attended entities" }));
	};
	render();
	return wrap;
}

// --- tiles + resizable splitters --------------------------------------------

function tile(title, sub, body, { cls = "", tools = null } = {}) {
	const head = el("div", { class: "ct-tile-head" }, el("span", { class: "ct-tile-title", text: title }));
	if (sub) head.appendChild(el("span", { class: "ct-tile-sub", text: sub }));
	if (tools) head.appendChild(tools);
	const kids = Array.isArray(body) ? body : [body];
	return el("div", { class: `ct-tile ${cls}` }, head, el("div", { class: "ct-tile-body" }, ...kids));
}

// A draggable splitter that resizes a grid track by writing a px CSS var on
// `target` (the grid element that declares the var). Persisted to localStorage.
function mkSplit(axis, target, varName, min = 130) {
	const split = el("div", { class: `ct-split ${axis === "x" ? "col" : "row"}` });
	const key = `tf-ct-split-${varName}`;
	try { const saved = localStorage.getItem(key); if (saved) target.style.setProperty(varName, saved); } catch { /* ignore */ }
	split.addEventListener("pointerdown", (e) => {
		e.preventDefault();
		split.classList.add("dragging");
		try { split.setPointerCapture(e.pointerId); } catch { /* ignore */ }
		const move = (ev) => {
			const rect = target.getBoundingClientRect();
			const v = axis === "x"
				? Math.max(min, Math.min(rect.width - min, ev.clientX - rect.left))
				: Math.max(min, Math.min(rect.height - min, ev.clientY - rect.top));
			const px = `${Math.round(v)}px`;
			target.style.setProperty(varName, px);
			try { localStorage.setItem(key, px); } catch { /* ignore */ }
		};
		const up = () => { split.classList.remove("dragging"); window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
		window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
	});
	return split;
}

// --- orchestration -----------------------------------------------------------

// Called when leaving the content view / switching cell: park the shared 3D
// viewer (stop drawing, clear the shading) and detach its canvas host so the
// next mount reparents it cleanly. (Named closeContentWindows for app.js.)
export function closeContentWindows() {
	state.viewer3d?.clearAttnHighlight?.();
	state.viewer3d?.setActive?.(false);
	if (state._ctHost && state._ctHost.parentNode) state._ctHost.parentNode.removeChild(state._ctHost);
}

export async function renderContent() {
	if (state.view !== "content") return;
	const host = $("content-inner");
	if (!host) return;
	const token = bumpLoad();
	ctHoverReset();

	const sel = selectedSteps();
	if (!sel.length) {
		state.viewer3d?.clearAttnHighlight?.();
		host.replaceChildren(emptyCard("no computed attention for this selection — pick a region + step with computed attention."));
		return;
	}

	$("dv-loading").classList.add("on");
	if (!host.querySelector(".ct-grid")) host.replaceChildren(emptyCard("loading…", "⧗"));
	let rows;
	try {
		[rows] = await Promise.all([
			loadRows(token),
			ensureEvents().catch((e) => { console.warn("[tf-workspace] events load failed — output/reasoning text unavailable:", e); return null; }),
		]);
	} catch (e) { if (token === state.loadToken) host.replaceChildren(emptyCard(`failed: ${e.message}`)); $("dv-loading").classList.remove("on"); return; }
	if (token !== state.loadToken) return;
	$("dv-loading").classList.remove("on");
	if (!rows.length) { host.replaceChildren(emptyCard("no attention data for this selection")); return; }
	state.rows = rows;

	const single = rows.length === 1;
	const step = single ? (sel.find((s) => String(s.event_index) === String(rows[0].event_index)) || sel[0]) : null;
	const agg = aggregateRows(rows);
	const targetZone = single ? targetZoneOf(step) : (state.region !== ALL ? state.region : null);
	const a = single ? rows[0].a : null;
	ensureViewer();

	// --- tiles ---
	const treeSub = single ? "this step's attention" : `mean over ${rows.length} steps${state.region !== ALL ? ` · ${state.region}` : ""}`;
	const treeTile = tile("attention tree", treeSub, buildTree(agg, targetZone, false), { cls: "ct-tree-tile" });
	const origSub = single ? `${step.template ?? step.step} · ${step.node ?? ""}` : `${rows.length} steps`;
	const origTile = tile("original content", origSub, originalContentBody(single, step, rows));

	// 3D tile: the shared viewer host + an inline controls strip, both DIRECT
	// children of the (flex-column) tile body so the canvas fills and the controls
	// pin below.
	const controls = el("div", { class: "ct-3d-controls" });
	const threeDTile = tile("3D scene", single ? `${step.template ?? step.step} · ${step.node ?? ""}` : `${rows.length} steps`, [state._ctHost, controls], { cls: "ct-3d" });

	const zoneTile = tile("zone plan", targetZone ? (parentZoneOf(targetZone) ? `upstream of ${targetZone}` : targetZone) : "", single || targetZone ? zonePlanBody(a || (rows[0] && rows[0].a), targetZone) : el("div", { class: "muted", text: "pick a single step / region to see its zone plan" }));
	const sortTile = tile("attention · sorted", "highest-attended first", sortedListBody(agg));

	// --- assemble the resizable grid ---
	const grid = el("div", { class: "ct-grid" });
	const leftCol = el("div", { class: "ct-col left" });
	leftCol.append(treeTile, mkSplit("y", leftCol, "--ct-lT"), origTile);
	const rightCol = el("div", { class: "ct-col right" });
	const rbottom = el("div", { class: "ct-rbottom" });
	rbottom.append(zoneTile, mkSplit("x", rbottom, "--ct-bL"), sortTile);
	rightCol.append(threeDTile, mkSplit("y", rightCol, "--ct-rT"), rbottom);
	grid.append(leftCol, mkSplit("x", grid, "--ct-cL"), rightCol);

	host.replaceChildren(grid);
	refresh3DTile(controls, agg, step, single, targetZone);
}
