// Placement tab (bbox-batch steps): one card per placed object showing what its
// bbox-generating OUTPUT tokens attended to — the to-place batch (incl. its own
// spec) vs. the scene, the specific objects, and the attributes it leaned on.

import { el } from "../../js/ui.js";
import { $, state, COLORS, entityHex } from "./state.js";
import { hasToPlace } from "./aggregate.js";
import { summaryBar } from "./widgets.js";
import { jumpTo } from "./exportPanel.js";
import { applyAttnHighlight } from "./attnPanel.js";

export function renderPlacement() {
	const host = $("tf-placement-panel");
	if (!host || $("tf-tab-placement")?.hidden) return;
	const note = (txt) => host.replaceChildren(el("div", { class: "empty-note" }, el("span", { text: txt })));
	const step = state.steps[state.stepIdx];
	const displayed = !!(state.attn && step && state.attn.meta && state.attn.meta.event_index === step.event_index);
	if (!displayed) return note("compute this step's attention to see placement");
	const a = state.attn;
	if (!hasToPlace(a)) return note("this step doesn't place a batch — the placement view is for bbox-batch steps");
	// Per-object rollups are PRECOMPUTED (agg.outputs); only steps that actually
	// place objects carry a to-place readout per output object.
	const outs = ((a.agg && a.agg.outputs) || []).filter((o) => o.to_place);
	if (!outs.length) return note("no per-object output tokens mapped for this step");
	// Token indices per placed object (compact tokens carry output_entity) — for
	// the click-to-scrub head.
	const idxs = new Map();
	a.tokens.forEach((t, ti) => {
		if (t.output_entity == null) return;
		if (!idxs.has(t.output_entity)) idxs.set(t.output_entity, []);
		idxs.get(t.output_entity).push(ti);
	});
	const tpMeta = new Map((a.to_place_entities || []).map((e) => [e.id, e]));
	const bars = (list, colorFor, onclickFor) => {
		const max = list[0]?.score ?? 1;
		return list.length
			? el("div", { class: "sbars" }, ...list.map((e) => summaryBar({
				color: colorFor(e), label: e.label ?? e.id ?? e.component, value: e.score, max,
				title: `${e.id ?? e.component} · Σ ${e.score.toFixed(4)}`, onclick: onclickFor && onclickFor(e),
			})))
			: el("div", { class: "muted", style: "font-size:11px", text: "—" });
	};
	const cards = outs.map((o) => {
		const id = o.entity;
		const gi = idxs.get(id) || [];
		const tpMass = o.to_place?.mass ?? 0, scMass = o.scene?.mass ?? 0;
		const smax = Math.max(tpMass, scMass, 1e-9);
		const meta = tpMeta.get(id);
		const head = el("div", { class: "pl-obj-head", title: "focus in 3D + center the token scrubber on this object", onclick: () => {
			if (gi.length) state.attnToken = gi[0]; jumpTo(id); applyAttnHighlight([{ id, weight: 1 }]);
		} },
			el("span", { class: "pl-sw", style: `background:${entityHex(meta?.kind || "object", id)}` }),
			el("span", { class: "pl-obj-id", text: id }),
			el("span", { class: "pl-obj-meta", text: `${o.n} tok${meta?.parent ? ` · in ${meta.parent}` : ""}` }),
		);
		const split = el("div", { class: "sbars" },
			summaryBar({ color: COLORS.to_place, label: "to-place batch", value: tpMass, max: smax, title: "attention on the objects being placed (incl. its own spec)" }),
			summaryBar({ color: "#4af0e0", label: "scene", value: scMass, max: smax, title: "attention on the surrounding scene" }),
		);
		const placed = (o.to_place?.entityTotals || []).slice(0, 6).map((e) => ({ ...e, label: e.id === id ? `${e.id} ← self` : e.id }));
		const scene = (o.scene?.entityTotals || []).slice(0, 6);
		const attrMap = new Map();
		for (const c of [...(o.scene?.componentTotals || []), ...(o.to_place?.componentTotals || [])]) attrMap.set(c.component, (attrMap.get(c.component) || 0) + c.score);
		const attrs = [...attrMap.entries()].map(([component, score]) => ({ component, score })).sort((x, y) => y.score - x.score).slice(0, 6);
		return el("div", { class: "pl-card" },
			head,
			el("div", { class: "pl-sub", text: "to-place vs scene" }), split,
			el("div", { class: "pl-sub", text: "placed objects it attended to" }),
			bars(placed, () => COLORS.to_place, (e) => () => { jumpTo(e.id); applyAttnHighlight([{ id: e.id, weight: 1 }]); }),
			el("div", { class: "pl-sub", text: "scene objects it attended to" }),
			bars(scene, (e) => entityHex(e.kind, e.id), (e) => () => { jumpTo(e.id); applyAttnHighlight([{ id: e.id, weight: 1 }]); }),
			...(attrs.length ? [el("div", { class: "pl-sub", text: "attributes leaned on" }), bars(attrs, () => "#8ab4ff", null)] : []),
		);
	});
	const scroller = $("tf-tab-placement");
	const savedTop = scroller ? scroller.scrollTop : 0;
	host.replaceChildren(
		el("div", { class: "muted", style: "font-size:11px;margin-bottom:10px",
			text: `${outs.length} placed object${outs.length > 1 ? "s" : ""} · what each attended to while its bbox was generated` }),
		...cards,
	);
	if (scroller) scroller.scrollTop = savedTop;
}
