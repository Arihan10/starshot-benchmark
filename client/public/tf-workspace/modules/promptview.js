// Per-step prompt viewer. When a single step is targeted (via the header
// region/step selectors), the "prompts" button opens this popup showing that
// step's exact logged call: the system + user prompts and the model's reasoning
// + output (completion). Sourced from the step's cache.llm event.

import { el } from "../../js/ui.js";
import { $, state } from "./state.js";
import { stepLLM } from "./data.js";

function section(label, text) {
	const t = String(text ?? "");
	const pre = el("pre", { class: "pv-pre" });
	pre.textContent = t || "(none)";
	return el("div", { class: "pv-sec" },
		el("div", { class: "pv-sec-head" },
			el("span", { class: "pv-sec-label", text: label }),
			el("span", { class: "pv-sec-meta", text: t ? `${t.length.toLocaleString()} chars` : "empty" })),
		el("div", { class: "pv-sec-body" }, pre));
}

export function openPromptView(step) {
	const root = $("modal-root");
	if (!root || !step) return;
	const ev = step.event_index;
	const e = stepLLM(ev);
	const tmpl = step.template ?? step.step ?? "?";
	const node = step.node ? ` · ${step.node}` : "";

	const onKey = (k) => { if (k.key === "Escape") close(); };
	function close() { root.replaceChildren(); document.removeEventListener("keydown", onKey); }

	let body;
	if (!e) {
		body = el("div", { class: "pv-empty", text: "no logged prompt for this step (a non-LLM step, or events not loaded)." });
	} else {
		body = el("div", { class: "pv-scroll" },
			section("system prompt", e.system),
			section("user prompt", e.user),
			section("reasoning", e.reasoning),
			section("output (completion)", e.output));
	}
	const meta = e ? [e.model, e.schema, e.tokens_in != null ? `in ${e.tokens_in}` : null, e.tokens_out != null ? `out ${e.tokens_out}` : null].filter(Boolean).join(" · ") : "";
	const panel = el("div", { class: "pv-panel" },
		el("div", { class: "pv-head" },
			el("span", { class: "pv-title", text: `step ${ev} · ${tmpl}${node}` }),
			meta ? el("span", { class: "pv-meta", text: meta }) : null,
			el("span", { style: "flex:1" }),
			el("button", { class: "pv-close", title: "close (Esc)", text: "✕", onclick: close }),
		),
		body);
	const overlay = el("div", { class: "pv-overlay", onclick: (k) => { if (k.target === overlay) close(); } }, panel);
	root.replaceChildren(overlay);
	document.addEventListener("keydown", onKey);
}
