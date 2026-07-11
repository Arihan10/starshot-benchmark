// Shared DOM widgets used by the summary/overview/placement analytics and the
// export panel: labeled blocks, the error-bar-aware horizontal bar, stat cards,
// and small key/value + full-width helpers.

import { el } from "../../js/ui.js";
import { state } from "./state.js";
import { barError, significant, fmtPM, fmtNum } from "./uncertainty.js";

export function summaryBlock(label, hint, ...body) {
	return el("div", { class: "exp-block" },
		el("div", { class: "exp-lab", text: label }),
		...body,
		hint ? el("div", { class: "hint", text: hint }) : null,
	);
}

// Reusable horizontal bar. Pass `sd` (the p/m signal's spread) to overlay an
// error bar: a whisker centered on the fill, an "×" when the error is too big to
// read, and a "± s" value. Near-zero, non-significant values are flagged "ns".
export function summaryBar({ color, label, value, max, title, onclick, sd = null }) {
	const denom = max || 1e-9;
	const pct = Math.max(2, (100 * value) / denom);
	const hasErr = state.showErr && sd != null && sd > 0;
	const track = el("span", { class: "sbar-track" }, el("span", { class: "sbar-fill", style: `width:${pct}%;background:${color}` }));
	if (hasErr) {
		const g = barError({ value, err: sd, max: denom });
		if (g.tooBig) {
			track.appendChild(el("span", { class: "sbar-err-x", style: `left:${Math.min(94, g.fillPct)}%`, text: "×" }));
		} else {
			track.appendChild(el("span", { class: "sbar-err", style: `left:${g.loPct}%;width:${Math.max(0, g.hiPct - g.loPct)}%` }));
		}
	}
	const ns = hasErr && !significant(value, sd);
	return el("div", {
		class: `sbar-row${onclick ? " clickable" : ""}${ns ? " ns" : ""}`,
		title: title || (hasErr ? fmtPM(value, sd) : undefined),
		...(onclick ? { onclick } : {}),
	},
		el("span", { class: "sbar-sw", style: `background:${color}` }),
		el("span", { class: "sbar-lab", text: label }),
		track,
		el("span", { class: "sbar-val", text: hasErr ? fmtPM(value, sd) : fmtNum(value) }),
	);
}

// Tag a block to span every masonry column (full-width banner in the report).
export function wide(node) { if (node) node.classList.add("wide"); return node; }

export function statCard(k, v, sub) {
	return el("div", { class: "stat", title: `${k}: ${v}${sub ? ` (${sub})` : ""}` },
		el("div", { class: "s-k", text: k }),
		el("div", { class: "s-v" }, el("span", { text: String(v) }), sub ? el("small", { text: ` ${sub}` }) : null),
	);
}

export function block(label, body) {
	return el("div", { class: "exp-block" }, el("div", { class: "exp-lab", text: label }), body);
}

export function kv(k, v) {
	return el("div", { class: "kv-row", style: "display:contents" },
		el("span", { class: "k", text: k }), el("span", { class: "v", text: String(v) }));
}
