// Per-object account of what the from-scratch generated build did NOT return.
//
// The generated build logs to its own events.generated.jsonl, and no SSE stream
// carries that file — so unlike the library pipeline, whose mesh.error events
// reach the log panel live, nothing about a failed generated asset ever reached
// the screen. The gate simply rendered a smaller number than the scene's object
// count and stopped polling. The generate-status response now folds that log into
// a `failures[]` array; this panel is where it surfaces.
//
// Rows nest under the prefab CANONICAL whose mesh they share, because one backend
// failure silently takes its whole group with it — a flat list of twelve rows
// hides that eleven of them are collateral damage from the twelfth.

import { el, toast } from "./ui.js";

// Rendered rows per section. A badly broken build can produce hundreds; this
// keeps the DOM bounded while "copy" still yields the complete list.
const MAX_ROWS = 80;

// kind -> how it reads, and how loud it should be. The `warn` kinds are
// consequences rather than root causes: `reuse` is another object's failure
// landing on this one, `stuck` is the build dying mid-job.
const KINDS = {
	mesh: { label: "build failed", tone: "err" },
	abandoned: { label: "no result", tone: "err" },
	optimize: { label: "optimizer", tone: "err" },
	stuck: { label: "interrupted", tone: "warn" },
	reuse: { label: "shared mesh", tone: "warn" },
	unbuilt: { label: "not started", tone: "dim" },
};

export function initGenFailPanel({ onSelect } = {}) {
	let rows = [];
	let sig = null;
	let inGeneratedView = false;

	const btn = el("button", {
		id: "btn-gen-failures",
		class: "gf-btn",
		onclick: () => setOpen(!panel.classList.contains("open")),
	});
	btn.style.display = "none";

	const bodyEl = el("div", { class: "gf-body" });
	const noteEl = el("div", { class: "gf-note" });
	const panel = el(
		"div",
		{ id: "genfail-panel" },
		el(
			"div",
			{ class: "gf-head" },
			el("span", { class: "gf-title", text: "not returned" }),
			el("span", {
				class: "gf-act gf-copy",
				text: "copy",
				title: "copy every row (including any not shown) to the clipboard",
				onclick: copyAll,
			}),
			el("span", { class: "gf-act gf-x", text: "×", title: "close", onclick: () => setOpen(false) }),
		),
		noteEl,
		bodyEl,
	);
	// Anchored to the canvas, not to its button: the toggle row is a wide flex row
	// whose right-hand buttons can sit past the viewport edge, and a
	// button-anchored popover goes off-screen with them. This sits where
	// #lighting-panel does and slides clear of the trace panel by the same rule.
	document.getElementById("canvas-host")?.appendChild(panel);

	function setOpen(next) {
		panel.classList.toggle("open", next);
		btn.classList.toggle("on", next);
	}

	// Order: each failing canonical, immediately followed by the objects that lost
	// their mesh because of it. A reuse whose canonical did NOT itself fail (it
	// broke for its own reason) has no parent here and stays top level.
	function flatten(list) {
		const roots = new Set(list.filter((r) => r.canonical === r.id).map((r) => r.id));
		const isChild = (r) => r.canonical !== r.id && roots.has(r.canonical);
		const kids = new Map();
		for (const r of list.filter(isChild)) {
			kids.set(r.canonical, [...(kids.get(r.canonical) ?? []), r]);
		}
		const out = [];
		for (const r of list) {
			if (isChild(r)) continue;
			out.push({ row: r, child: false });
			for (const k of kids.get(r.id) ?? []) out.push({ row: k, child: true });
		}
		return out;
	}

	function rowEl({ row, child }) {
		const meta = KINDS[row.kind] ?? { label: row.kind, tone: "warn" };
		return el(
			"div",
			{
				class: `gf-row gf-${meta.tone}${child ? " gf-child" : ""}`,
				title: `${row.id} — ${row.message}`,
				onclick: () => onSelect?.(row.id),
			},
			el("span", { class: "gf-dot" }),
			el("span", { class: "gf-id", text: row.id }),
			el("span", { class: "gf-kind", text: meta.label }),
			row.stale
				? el("span", {
						class: "gf-stale",
						text: "old asset",
						// The one failure mode invisible in the viewport: a regenerate is
						// non-destructive, so a failed one leaves the previous mesh
						// rendering exactly as if it had worked.
						title: "Still showing its PREVIOUS mesh — the rebuild never landed",
					})
				: null,
			el("span", { class: "gf-msg", text: row.message }),
		);
	}

	function section(title, list, { grouped }) {
		const ordered = grouped ? flatten(list) : list.map((row) => ({ row, child: false }));
		const shown = ordered.slice(0, MAX_ROWS);
		const wrap = el(
			"div",
			{ class: "gf-section" },
			el(
				"div",
				{ class: "gf-sec-head" },
				el("span", { text: title }),
				el("span", { class: "gf-sec-n", text: String(list.length) }),
			),
			...shown.map(rowEl),
		);
		if (ordered.length > shown.length) {
			wrap.appendChild(
				el("div", {
					class: "gf-more",
					text: `… ${ordered.length - shown.length} more — “copy” has the full list`,
				}),
			);
		}
		return wrap;
	}

	async function copyAll() {
		const text = rows
			.map((r) =>
				[
					r.id,
					r.kind,
					r.canonical !== r.id ? `via ${r.canonical}` : "",
					r.stale ? "old asset still shown" : "",
					r.message,
				]
					.filter(Boolean)
					.join("\t"),
			)
			.join("\n");
		try {
			await navigator.clipboard.writeText(text);
			toast(`copied ${rows.length} row${rows.length === 1 ? "" : "s"}`, "ok");
		} catch {
			toast("clipboard unavailable", "err");
		}
	}

	function paint(running) {
		// "not started" objects are split out: on an interrupted build they can
		// number in the hundreds and would otherwise bury the handful that broke.
		const failed = rows.filter((r) => r.kind !== "unbuilt");
		const unbuilt = rows.filter((r) => r.kind === "unbuilt");
		btn.textContent = `⚠ ${rows.length} not returned`;
		btn.title =
			`${rows.length} object${rows.length === 1 ? "" : "s"} in this build have no new asset` +
			(unbuilt.length ? ` (${failed.length} failed, ${unbuilt.length} never started)` : "");
		noteEl.textContent = running ? "build still running — this list is not final" : "";
		noteEl.style.display = running ? "" : "none";
		bodyEl.textContent = "";
		if (failed.length) bodyEl.appendChild(section("failed", failed, { grouped: true }));
		if (unbuilt.length) bodyEl.appendChild(section("never started", unbuilt, { grouped: false }));
	}

	function sync() {
		const show = inGeneratedView && rows.length > 0;
		btn.style.display = show ? "" : "none";
		if (!show) setOpen(false);
	}

	return {
		// The toolbar badge; the caller drops it into #viewer-toggles. The panel it
		// opens lives on #canvas-host and is managed here.
		button: btn,
		render(failures, { running = false } = {}) {
			rows = Array.isArray(failures) ? failures : [];
			const next = `${running}|${rows.map((r) => `${r.id}:${r.kind}:${r.stale ? 1 : 0}`).join(",")}`;
			if (next !== sig) {
				sig = next;
				paint(running);
			}
			sync();
		},
		// Called on cell / version / mode changes: the previous build's failures
		// must never linger over a different one.
		clear() {
			rows = [];
			sig = null;
			bodyEl.textContent = "";
			sync();
		},
		setVisible(show) {
			inGeneratedView = show;
			sync();
		},
	};
}
