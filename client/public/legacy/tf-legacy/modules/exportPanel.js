// Export / reconstruction panel: the reconstructed teacher-forcing sequence with
// its colored entity/component spans, the scene / to-place / output / variable
// maps, reconstruction-validity readout, and the cross-highlight `jumpTo` that
// selects an entity in 3D + scrolls the sequence.

import { el } from "../../js/ui.js";
import { $, state, COLORS, COMPONENT_ABBR, compHex, entityHex } from "./state.js";
import { cssEsc } from "./util.js";
import { block, kv } from "./widgets.js";

export function renderExport() {
	const e = state.export;
	const wrap = $("tf-export-wrap");
	if (!e) { wrap.replaceChildren(); return; }
	$("tf-export-badge").replaceChildren(el("span", { class: "pill", text: "reconstructed" }));

	const p = e.params;
	const b = e.boundaries;
	const paramsKv = el("div", { class: "kv" },
		kv("model", e.meta.model_id),
		kv("schema", p.response_format ?? "—"),
		kv("reasoning_effort", p.reasoning_effort ?? "—"),
		kv("top_logprobs", p.top_logprobs ?? "(not captured)"),
		kv("tokens in/out", `${p.tokens_in ?? "?"} / ${p.tokens_out ?? "?"}`),
		kv("output source", e.meta.output_source),
		kv("completion @", `${b.completion_start} / ${b.total_len} chars`),
		kv("input frame", `full [0, ${e.frames.input.end}) → raw user`),
		kv("output frame", `full [${e.frames.output.start}, ${e.frames.output.end}) → raw output (logprobs frame)`),
	);

	const legend = el("div", { class: "seg-legend exp-lab" }, el("span", { text: "highlight:" }),
		...[["scene", "zones/objects"], ["to_place", "to-place"], ["output", "output"], ["variables", "variables"], ["none", "none"]]
			.map(([key, label]) => el("button", {
				text: label, class: state.highlight === key ? "on" : "",
				onclick: () => { state.highlight = key; renderExport(); },
			})),
	);

	// Component sub-filter: "whole" entity spans, "all" components colored
	// separately, or a single component (placement / relationships / dimensions
	// / parent / …) isolated across every entity.
	const compLegend = el("div", { class: "seg-legend exp-lab" }, el("span", { text: "component:" }),
		...["entity", "all", ...presentComponents()].map((c) => el("button", {
			text: c === "entity" ? "whole" : (c === "all" ? "all" : (COMPONENT_ABBR[c] ?? c)),
			title: c, class: state.component === c ? "on" : "",
			onclick: () => { state.component = c; renderExport(); },
		})),
	);
	const seq = el("div", { id: "tf-seq", class: state.seqExpanded ? "" : "clamped" });
	const seqBlock = el("div", { class: "exp-block" },
		el("div", { class: "exp-lab" },
			el("span", { text: "reconstructed sequence (input → reasoning → output)" }),
			el("span", { class: "muted", text: `${e.text.full.length} chars` }),
			el("button", {
				style: "margin-left:auto;font-size:11px;padding:2px 8px",
				text: state.seqExpanded ? "clamp" : "expand",
				onclick: () => { state.seqExpanded = !state.seqExpanded; renderExport(); },
			}),
		),
		legend,
		compLegend,
		seq,
	);

	// Lead with the reconstructed sequence (the star), then the entity maps, and
	// tuck generation params at the bottom. The step/node/event-index that used
	// to head this panel now lives in the timeline bar. Filter nulls —
	// replaceChildren coerces null to the literal text "null".
	wrap.replaceChildren(...[
		reconstructionValidity(),
		seqBlock,
		entityMapBlock(`scene context · ${e.scene_map.length} entities`, e.scene_map, "scene"),
		e.to_place_map.length ? entityMapBlock(`to-place batch · ${e.to_place_map.length}`, e.to_place_map, "to_place") : null,
		e.output_map.length ? clampedBlock(`output assignments · ${e.output_map.length}`, "map:output", mapList(e.output_map.map((m) => ({ ...m, color: "output" })), "id")) : null,
		clampedBlock(`variables · ${e.variables_map.length}`, "map:variables", mapList(e.variables_map.map((m) => ({ ...m, id: m.name, color: "variable" })), "id")),
		block("generation params", paramsKv),
	].filter(Boolean));

	buildSeq(seq, e.text.full, currentSpans(), b.completion_start);
	if (state.pendingFocus) { const f = state.pendingFocus; state.pendingFocus = null; jumpTo(f.entity, f.mark); }
}

// Reconstruction validity: how closely THIS step's teacher-forced forward
// reproduces the model's ORIGINAL per-token logprobs (captured from the API).
// A small mean |Δ logprob| means we fed the model the same input it originally
// saw — i.e. the native reconstruction + span mapping is faithful. Only shown
// when the step's analysis is loaded (the comparison lives in the result).
function reconstructionValidity() {
	const a = state.attn, e = state.export;
	if (!a || !e || a.meta?.event_index !== e.meta?.event_index) return null;
	const lc = a.logprob_check || {};
	let cls, txt;
	if (lc.aligned > 0 && lc.mean_abs_delta != null) {
		const ok = lc.mean_abs_delta < 0.5; // nats — small deltas expected from kernel/precision
		cls = ok ? "ok" : "warn";
		txt = `${ok ? "✓ validated" : "⚠ divergent"} — ${lc.aligned} output tokens · mean |Δ logprob| ${lc.mean_abs_delta.toFixed(3)} · max ${lc.max_abs_delta.toFixed(3)}`;
	} else if (lc.remote_available) {
		cls = "warn"; txt = "remote logprobs present but no output tokens aligned (span mismatch)";
	} else if (a.meta?.mock) {
		cls = "none"; txt = "mock result — recompute on the GPU to validate against original logprobs";
	} else {
		cls = "none"; txt = "no captured logprobs for this step — re-run the cell with logprob capture to validate";
	}
	return el("div", { class: "exp-block" },
		el("div", { class: "exp-lab", text: "reconstruction validity — teacher-forced logprobs vs original" }),
		el("div", { class: `recon-validity ${cls}`, title: lc.note || "", text: txt }),
	);
}

// A scene/to-place map block: a header with a sort control, then either entity
// rows (each with its parent link + component chips) or, when a component
// filter is active, the matching component occurrences.
// Blocks whose lists can get long are clamped to a scrollable box by default;
// an expand toggle un-clamps them inline.
function toggleBlock(key) {
	if (state.expandedBlocks.has(key)) state.expandedBlocks.delete(key);
	else state.expandedBlocks.add(key);
	renderExport();
}

function expandToggle(key) {
	return el("button", {
		style: "font-size:11px;padding:2px 8px", text: state.expandedBlocks.has(key) ? "clamp" : "expand",
		onclick: () => toggleBlock(key),
	});
}

function clampedBlock(label, key, body) {
	const expanded = state.expandedBlocks.has(key);
	const toggle = expandToggle(key);
	toggle.style.marginLeft = "auto";
	return el("div", { class: "exp-block" },
		el("div", { class: "exp-lab" }, el("span", { text: label }), toggle),
		el("div", { class: expanded ? "" : "clamp-list" }, body),
	);
}

function entityMapBlock(label, entries, layer) {
	const key = `map:${layer}`;
	const expanded = state.expandedBlocks.has(key);
	const sortSel = el("select", { style: "margin-left:auto;font-size:11px;padding:1px 4px",
		onchange: (ev) => { state.sort = ev.target.value; renderExport(); } },
		...["position", "id", "kind", "parent"].map((s) => el("option", { value: s, text: `sort: ${s}`, ...(state.sort === s ? { selected: "" } : {}) })),
	);
	return el("div", { class: "exp-block" },
		el("div", { class: "exp-lab" }, el("span", { text: label }), sortSel, expandToggle(key)),
		el("div", { class: expanded ? "" : "clamp-list" }, entityMapBody(entries, layer)),
	);
}

function entityMapBody(entries, layer) {
	if (!entries.length) {
		return el("div", { class: "muted", text: layer === "scene"
			? "no scene entities in this step's context (image_prompt uses a reduced view; the earliest steps have an empty scene)."
			: "nothing to place in this step." });
	}
	const comp = state.component;
	if (comp !== "entity") {
		const rows = [];
		for (const m of entries) for (const c of m.components ?? []) {
			if (comp === "all" || c.component === comp) rows.push({ entity: m.id, comp: c.component, kind: m.kind, start: c.start, end: c.end, user_rel: c.user_rel });
		}
		if (!rows.length) return el("div", { class: "muted", text: `no "${comp}" component in these ${entries.length} entries` });
		sortRows(rows);
		return el("div", { class: "map-list" }, rows.map((r) => el("div", {
			class: "map-row", title: `${r.entity} · ${r.comp} · full [${r.start}, ${r.end})${r.user_rel ? ` · user [${r.user_rel[0]}, ${r.user_rel[1]})` : ""}`,
			onclick: () => focusEntityComponent(layer, r.entity, r.comp),
		},
			el("span", { class: "map-sw", style: `background:${compHex(r.comp)}` }),
			el("span", { class: "mid", text: r.entity }),
			el("span", { class: "comp-tag", text: r.comp }),
			el("span", { class: "span", text: `${r.start}–${r.end}` }),
		)));
	}
	return el("div", { class: "map-list" }, sortEntities(entries.slice(), layer).map((m) => entityRow(m, layer)));
}

function entityRow(m, layer) {
	const kindColor = layer === "scene" ? entityHex(m.kind, m.id) : COLORS.to_place;
	const chips = (m.components ?? []).filter((c) => c.component !== "name").map((c) => el("span", {
		class: "chip", style: `background:${compHex(c.component)}`, title: `${c.component} — highlight + jump`,
		text: COMPONENT_ABBR[c.component] ?? c.component,
		onclick: (ev) => { ev.stopPropagation(); focusEntityComponent(layer, m.id, c.component); },
	}));
	const parentLink = m.parent ? el("span", {
		class: "parent-link", title: `parent: ${m.parent} — select it`, text: `↖ ${m.parent}`,
		onclick: (ev) => { ev.stopPropagation(); jumpTo(m.parent); },
	}) : null;
	return el("div", {
		class: "map-row", title: `${m.id}${m.source ? ` · from ${m.source}` : ""} · full [${m.start}, ${m.end})${m.user_rel ? ` · user [${m.user_rel[0]}, ${m.user_rel[1]})` : ""}${m.parent ? ` · parent=${m.parent}` : ""}`,
		onclick: () => jumpTo(m.id),
	},
		el("span", { class: "map-sw", style: `background:${kindColor}` }),
		el("span", { class: "mid", text: m.id }),
		parentLink,
		el("span", { class: "chips" }, chips),
		el("span", { class: "span", text: `${m.start}–${m.end}` }),
	);
}

function sortEntities(arr, layer) {
	const by = state.sort;
	if (by === "id") return arr.sort((x, y) => x.id.localeCompare(y.id));
	if (by === "kind") return arr.sort((x, y) => String(x.kind || "").localeCompare(String(y.kind || "")) || x.start - y.start);
	if (by === "parent") return arr.sort((x, y) => String(x.parent || "~").localeCompare(String(y.parent || "~")) || x.start - y.start);
	return arr.sort((x, y) => x.start - y.start);
}

function sortRows(rows) {
	const by = state.sort;
	if (by === "id") rows.sort((x, y) => x.entity.localeCompare(y.entity) || x.start - y.start);
	else if (by === "kind") rows.sort((x, y) => String(x.kind || "").localeCompare(String(y.kind || "")) || x.start - y.start);
	else rows.sort((x, y) => x.start - y.start);
}

function currentSpans() {
	const e = state.export;
	if (!e) return [];
	if (state.highlight === "output") return e.output_map.map((m) => ({ id: m.id, entity: m.id, start: m.start, end: m.end, hex: COLORS.output }));
	if (state.highlight === "variables") return e.variables_map.map((m) => ({ id: m.name, entity: m.name, start: m.start, end: m.end, hex: COLORS.variable }));
	if (state.highlight === "scene" || state.highlight === "to_place") {
		const entries = state.highlight === "scene" ? e.scene_map : e.to_place_map;
		if (state.component === "entity") {
			return entries.map((m) => ({
				id: m.id, entity: m.id, start: m.start, end: m.end,
				hex: state.highlight === "scene" ? entityHex(m.kind, m.id) : COLORS.to_place,
			}));
		}
		// component-level: one span per matching component of every entry.
		const spans = [];
		for (const m of entries) {
			for (const c of m.components ?? []) {
				if (state.component === "all" || c.component === state.component) {
					spans.push({ id: `${m.id}.${c.component}`, entity: m.id, comp: c.component, start: c.start, end: c.end, hex: compHex(c.component) });
				}
			}
		}
		return spans;
	}
	return [];
}

// Components present across BOTH the scene and to-place entries (the filter is
// a global "component focus", so its choices are stable regardless of which
// layer is highlighted). Returned in a stable, meaningful order.
function presentComponents() {
	const e = state.export;
	if (!e) return [];
	const seen = new Set();
	for (const m of [...(e.scene_map ?? []), ...(e.to_place_map ?? [])]) {
		for (const c of m.components ?? []) seen.add(c.component);
	}
	const order = ["name", "prompt", "noun_phrase", "description", "placement", "relationships", "dimensions", "orientation", "yaw", "proxy_shape", "parent", "parent_region", "global_origin", "local_origin"];
	return order.filter((c) => seen.has(c));
}

// Render `text` into `container`, wrapping each (non-overlapping, sorted) span
// in a colored <mark>, and marking the completion boundary. Because a single
// highlight layer's spans all sit on one side of the boundary, the boundary
// never falls inside a span — so we can insert it during plain-text emission.
function buildSeq(container, text, spans, boundary) {
	container.textContent = "";
	const clean = [];
	let lastEnd = -1;
	for (const s of spans.slice().sort((a, b) => a.start - b.start)) {
		if (s.start >= lastEnd && s.end > s.start) { clean.push(s); lastEnd = s.end; }
	}
	const emitPlain = (from, to) => {
		if (to <= from) return;
		if (boundary != null && from < boundary && boundary < to) {
			container.appendChild(document.createTextNode(text.slice(from, boundary)));
			container.appendChild(boundaryEl());
			container.appendChild(document.createTextNode(text.slice(boundary, to)));
		} else {
			container.appendChild(document.createTextNode(text.slice(from, to)));
		}
	};
	let pos = 0;
	for (const s of clean) {
		emitPlain(pos, s.start);
		const label = s.comp ? `${s.entity} · ${s.comp}` : s.id;
		const mk = el("mark", { text: text.slice(s.start, s.end), title: `${label}  [${s.start}, ${s.end})`, dataset: { id: s.id } });
		mk.style.background = s.hex ?? "#888";
		mk.onclick = () => jumpTo(s.entity ?? s.id, s.id);
		container.appendChild(mk);
		pos = s.end;
	}
	emitPlain(pos, text.length);
}

function boundaryEl() {
	return el("span", {
		class: "boundary",
		style: "color:var(--accent);font-weight:700;background:#0d1830;padding:0 3px;border-radius:3px",
		text: " ⟼ model completion (teacher-forced) ⟼ ",
	});
}

// Select `entityId` in the 3D viewer (if it has a bbox) and scroll the
// sequence to the given mark — falling back to the entity's whole-span mark or
// its first component mark when the exact one isn't currently highlighted.
export function jumpTo(entityId, exactMarkId = null) {
	if (state.viewer?.hasBbox?.(entityId)) state.viewer.select(entityId, { frame: true });
	const root = $("tf-seq");
	if (!root) return;
	let mk = exactMarkId && root.querySelector(`mark[data-id="${cssEsc(exactMarkId)}"]`);
	if (!mk) mk = root.querySelector(`mark[data-id="${cssEsc(entityId)}"]`) || root.querySelector(`mark[data-id^="${cssEsc(entityId + ".")}"]`);
	if (mk) mk.scrollIntoView({ block: "center" });
}

// Focus one entity's component: switch the highlight to its layer + that
// component (so it's colored in the sequence), then scroll to the occurrence
// after the re-render.
function focusEntityComponent(layer, entityId, comp) {
	state.highlight = layer;
	state.component = comp;
	state.pendingFocus = { entity: entityId, mark: `${entityId}.${comp}` };
	renderExport();
}

function mapList(rows, idKey) {
	if (!rows.length) return el("div", { class: "muted", text: "(none)" });
	return el("div", { class: "map-list" }, rows.map((m) => el("div", {
		class: "map-row",
		title: `${m[idKey]}${m.source ? ` · from ${m.source}` : ""} · full [${m.start}, ${m.end})${m.output_rel ? ` · output [${m.output_rel[0]}, ${m.output_rel[1]})` : m.user_rel ? ` · user [${m.user_rel[0]}, ${m.user_rel[1]})` : ""} — click to select in 3D + scroll the sequence`,
		onclick: () => jumpTo(m[idKey]),
	},
		el("span", { class: "map-sw", style: `background:${COLORS[m.color] ?? "#888"}` }),
		el("span", { class: "mid", text: m.source ? `${m[idKey]}  ·  ${m.source}` : m[idKey] }),
		el("span", { class: "span", text: `${m.start}–${m.end}` }),
	)));
}
