// The /tf ablation drawer: for a scene tagged with ablations, lists this cell's
// ablation variants and lets you COMPUTE their treated-step attention right
// here — closing the loop (launch → re-infer treated step → compute attention →
// compare in the analysis workspace). Attention is what the ablation view
// overlays, so a variant is only useful once its treated step is computed.
//
// Two things make this feel live:
//   * Status checks are HYDRATION-FREE. The treated step's event_index is read
//     straight from the variant's events.jsonl via /artifacts (the variant halts
//     right after re-inferring it, so it's the last cache.llm event), and
//     "computed?" is a plain existence check of attention/{idx}.json — no run
//     hydration, no Modal poll. So the initial "checking…" resolves fast.
//   * The compute queue mirrors the board's launch queue: bounded concurrency,
//     colored per-variant state (queued / running / done / error), a progress
//     bar + count, and a ⏹ stop. Same handling quality on both surfaces.
//
// Select a GROUP by dragging a rubber-band box across the rows (shift-drag adds
// to the selection), then "⚡ compute selected".

import { api, SERVER_URL } from "../../js/api.js";
import { el, toast } from "../../js/ui.js";
import { MAX_ABLATION_BATCH } from "../../js/ablationcore.js";
import { state, $ } from "./state.js";
import { pool } from "./util.js";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const SKIP_STEPS = new Set(["library_match", "image_prompt"]); // matches /tf-steps

let variants = [];            // [{ name, kind, cut, tag }]
const attnState = new Map();  // name -> "computed"|"queued"|"running"|"error"|"none" (absent = checking)
const treatedIdx = new Map(); // name -> its OWN treated-step event_index (stable, cached)
const selected = new Set();   // names picked via the rubber-band drag
let renderToken = 0;          // guards a stale scene's async load from repainting

// Compute queue state (mirrors ablationboard.js's launch queue).
let computing = false;
let computeCancel = false;
let queueDone = 0;
let queueTotal = 0;

function tagOf(abl) {
	const t = abl.treatment || {};
	return [t.shuffle_method || "order", t.xml_tags === false ? "noxml" : null, t.attend_target ? "att" : null, t.distractors ? `d${t.distractors}` : null]
		.filter(Boolean).join("_");
}

// This cell's ablation variants (same base run + slot + model as the /tf cell).
async function fetchVariants() {
	try {
		const data = await api.runs();
		const list = Array.isArray(data) ? data : (data.runs ?? []);
		variants = list
			.filter((r) => {
				const a = r && r.ablation;
				return a && a.base_run === state.run && a.slot === state.slot && a.model === state.model && a.cut != null;
			})
			.map((r) => ({ name: r.name, kind: r.ablation.target_step_kind, cut: Number(r.ablation.cut), tag: tagOf(r.ablation) }));
	} catch { variants = []; }
}

// ---- hydration-free status (the responsiveness fix) -----------------------

const cellRel = (name) => `${encodeURIComponent(name)}/${encodeURIComponent(state.slot)}/${encodeURIComponent(state.model)}`;

// Treated step's event_index, read from the variant's OWN events.jsonl via
// /artifacts (a plain file read — NO run hydration, NO /tf-steps). The variant
// halts right after re-inferring its treated step, so it's the last cache.llm
// event; prefer the last one whose template/step matches the target kind.
// `index` == the event's line position, and attention lands at
// attention/{index}.json. Cached (stable per variant).
async function resolveTreated(v) {
	if (treatedIdx.has(v.name)) return treatedIdx.get(v.name);
	let idx = null;
	try {
		const res = await fetch(`${SERVER_URL}/artifacts/${cellRel(v.name)}/events.jsonl`, { cache: "no-store" });
		if (res.ok) {
			const text = await res.text();
			let lastOfKind = null;
			let lastAny = null;
			for (const line of text.split("\n")) {
				if (!line.trim()) continue;
				let e;
				try { e = JSON.parse(line); } catch { continue; }
				if (e.kind !== "cache.llm" || typeof e.index !== "number" || SKIP_STEPS.has(e.step)) continue;
				lastAny = e.index;
				if ((e.template ?? e.step) === v.kind) lastOfKind = e.index;
			}
			idx = lastOfKind != null ? lastOfKind : lastAny;
		}
	} catch { idx = null; }
	treatedIdx.set(v.name, idx);
	return idx;
}

// Is the treated step's attention already on disk? A plain existence check of
// attention/{idx}.json via /artifacts — NO hydration, NO Modal poll. (Freshly
// computed blobs land on disk once pulled, which waitComputed triggers.)
async function isComputedOnDisk(name, idx) {
	if (idx == null) return false;
	try {
		const res = await fetch(`${SERVER_URL}/artifacts/${cellRel(name)}/attention/${idx}.json`, { cache: "no-store" });
		return res.ok;
	} catch { return false; }
}

// Bulk status check: two cheap file reads per variant → fast enough to feel
// instant, so we run it at high concurrency. Never clobbers an in-flight
// compute (queued/running) or an already-known computed.
async function refreshAttn() {
	const slot0 = state.slot;
	const model0 = state.model;
	await pool(variants, 12, async (v) => {
		if (state.slot !== slot0 || state.model !== model0) return;
		const cur = attnState.get(v.name);
		if (cur === "queued" || cur === "running" || cur === "computed") return;
		const idx = await resolveTreated(v);
		attnState.set(v.name, (await isComputedOnDisk(v.name, idx)) ? "computed" : "none");
	});
}

// ---- compute queue (same shape as the board's launch queue) ---------------

// Poll until the treated step's attention lands on disk (or errors / times
// out). attentionList triggers the server→disk pull, so this both waits AND
// makes the result appear for the cheap existence check afterwards.
async function waitComputed(name, idx) {
	const deadline = Date.now() + 6 * 60 * 1000;
	while (!computeCancel && Date.now() < deadline) {
		try {
			const s = await api.attentionList(name, state.slot, state.model, { maxHeads: state.maxHeads });
			const done = new Set([...(s.computed || []), ...(s.stale || [])].map(Number));
			if (done.has(idx)) return true;
			const errs = s.errors || {};
			if (errs[idx] != null || errs[String(idx)] != null) return false;
		} catch { /* transient — keep polling */ }
		await sleep(3000);
	}
	return false;
}

// Enqueue one variant's treated-step attention, then wait for it to finish.
async function computeOne(v) {
	const idx = await resolveTreated(v);
	if (idx == null) { attnState.set(v.name, "none"); paint(); return; }
	attnState.set(v.name, "running");
	paint();
	try {
		await api.attentionEnqueue(v.name, state.slot, state.model, { eventIndices: [idx], maxHeads: state.maxHeads });
	} catch { attnState.set(v.name, "error"); paint(); return; }
	attnState.set(v.name, (await waitComputed(v.name, idx)) ? "computed" : "error");
	paint();
}

// The SAME bounded queue the board uses: MAX_ABLATION_BATCH compute at once, the
// rest wait in line; each slot enqueues then waits for the real result before
// pulling the next. ⏹ stop flips computeCancel.
async function computeMany(list) {
	if (computing) { toast("a compute queue is already running — wait or ⏹ stop"); return; }
	const todo = (list || []).filter((v) => attnState.get(v.name) !== "computed");
	if (!todo.length) { toast("nothing to compute — already done"); return; }
	const slot0 = state.slot;
	const model0 = state.model;
	computing = true;
	computeCancel = false;
	queueTotal = todo.length;
	queueDone = 0;
	selected.clear(); // the queue's colored state is now the view; drop the picker highlight
	for (const v of todo) attnState.set(v.name, "queued");
	paint();
	try {
		await pool(todo, MAX_ABLATION_BATCH, async (v) => {
			if (computeCancel || state.slot !== slot0 || state.model !== model0) return;
			await computeOne(v);
			queueDone += 1;
			paint();
		});
	} finally {
		computing = false;
	}
	toast(`compute queue ${computeCancel ? "stopped" : "done"}: ${queueDone}/${queueTotal}`, computeCancel ? "err" : "ok");
	paint();
}

// ---- rendering ------------------------------------------------------------

// Absent status = still checking (shown as a faint pulsing dot), so the list
// appears instantly and fills in without ever flashing "no variants".
function paint() {
	const host = $("tf-ablation-panel");
	if (!host || drag) return; // never rebuild rows out from under an active drag
	if (!variants.length) { host.replaceChildren(el("div", { class: "empty-note", text: "no ablation variants for this scene / model" })); return; }

	const computedN = variants.filter((v) => attnState.get(v.name) === "computed").length;
	const checking = variants.some((v) => !attnState.has(v.name));
	const status = computing
		? `queue ${queueDone}/${queueTotal} · ≤${MAX_ABLATION_BATCH} at a time${computeCancel ? " · stopping" : ""}`
		: checking ? "checking…" : `${computedN}/${variants.length} attention ✓`;

	const selBtn = selected.size
		? el("button", { class: "abl-dr-btn primary", title: "compute the treated-step attention for the dragged selection",
			onclick: () => computeMany(variants.filter((v) => selected.has(v.name))) }, `⚡ compute selected (${selected.size})`)
		: null;
	const clearBtn = selected.size
		? el("button", { class: "abl-dr-btn sm", title: "clear selection", onclick: () => { selected.clear(); paint(); } }, "clear")
		: null;
	const mainBtn = computing
		? el("button", { class: "abl-dr-btn danger", title: "stop the compute queue", onclick: () => { computeCancel = true; paint(); } }, "⏹ stop")
		: el("button", { class: "abl-dr-btn", title: `compute every variant's treated-step attention — ${MAX_ABLATION_BATCH} at a time, the rest queue`,
			onclick: () => computeMany(variants) }, "⚡ compute all");

	const head = el("div", { class: "abl-dr-head" },
		el("span", { class: "abl-dr-count", text: `${variants.length} variant${variants.length === 1 ? "" : "s"} · ${status}` }),
		el("span", { class: "abl-dr-grow" }),
		selBtn, clearBtn, mainBtn);

	const bar = computing
		? el("div", { class: "abl-dr-bar" }, el("div", { class: "abl-dr-bar-fill", style: `width:${queueTotal ? Math.round((100 * queueDone) / queueTotal) : 0}%` }))
		: null;

	const byKind = new Map();
	for (const v of variants) { if (!byKind.has(v.kind)) byKind.set(v.kind, []); byKind.get(v.kind).push(v); }
	const sections = [];
	for (const [kind, vs] of byKind) {
		sections.push(el("div", { class: "abl-dr-kindhead", text: kind }));
		for (const v of vs) {
			const st = attnState.get(v.name);
			const dot = st === "computed" ? "done" : st === "running" ? "running" : st === "queued" ? "queued"
				: st === "error" ? "error" : st === "none" ? "none" : "checking";
			const action = st === "computed"
				? el("span", { class: "abl-dr-ok", title: "attention computed", text: "✓" })
				: st === "running"
					? el("span", { class: "abl-dr-pending", title: "computing on GPU", text: "●" })
					: st === "queued"
						? el("span", { class: "abl-dr-pending", title: "queued", text: "…" })
						: st === "error"
							? el("button", { class: "abl-dr-btn sm", title: "compute failed — retry", onclick: (e) => { e.stopPropagation(); computeMany([v]); } }, "↻")
							: st === "none"
								? el("button", { class: "abl-dr-btn sm", title: "compute this variant's treated-step attention", onclick: (e) => { e.stopPropagation(); computeMany([v]); } }, "⚡")
								: el("span", { class: "abl-dr-pending", title: "checking attention status", text: "·" });
			const row = el("div", { class: `abl-dr-row${selected.has(v.name) ? " sel" : ""}`, title: v.name },
				el("span", { class: `abl-dr-dot ${dot}` }),
				el("span", { class: "abl-dr-cut", text: `@${v.cut}` }),
				el("span", { class: "abl-dr-tag", text: v.tag }),
				action);
			row._v = v;
			sections.push(row);
		}
	}
	host.replaceChildren(head, ...(bar ? [bar] : []),
		el("div", { class: "abl-dr-hint", text: "drag across rows to select a group, then ⚡ compute selected (shift-drag adds). then open workspace → ablation to compare." }),
		...sections);
}

// ---- rubber-band group select over the rows -------------------------------

let drag = null;

function initDrawerDrag() {
	const host = $("tf-ablation-panel");
	if (!host || host._dragInit) return;
	host._dragInit = true;

	host.addEventListener("mousedown", (e) => {
		if (e.button !== 0) return;
		if (e.target.closest("button, a, select, input")) return; // let the ⚡ buttons work
		drag = { x0: e.clientX, y0: e.clientY, moved: false, rect: null, shift: e.shiftKey };
	});

	document.addEventListener("mousemove", (e) => {
		if (!drag) return;
		if (!drag.moved && Math.abs(e.clientX - drag.x0) + Math.abs(e.clientY - drag.y0) < 6) return;
		if (!drag.moved) {
			drag.moved = true;
			if (!drag.shift) { selected.clear(); for (const r of host.querySelectorAll(".abl-dr-row.sel")) r.classList.remove("sel"); }
			drag.rect = el("div", { class: "abl-dr-drag" });
			document.body.appendChild(drag.rect);
		}
		const l = Math.min(e.clientX, drag.x0), t = Math.min(e.clientY, drag.y0);
		const r = Math.max(e.clientX, drag.x0), b = Math.max(e.clientY, drag.y0);
		Object.assign(drag.rect.style, { left: `${l}px`, top: `${t}px`, width: `${r - l}px`, height: `${b - t}px` });
		for (const row of host.querySelectorAll(".abl-dr-row")) {
			const v = row._v;
			if (!v) continue;
			const q = row.getBoundingClientRect();
			const hit = q.bottom >= t && q.top <= b && q.right >= l && q.left <= r;
			if (hit) { selected.add(v.name); row.classList.add("sel"); }
			else if (!drag.shift) { selected.delete(v.name); row.classList.remove("sel"); }
		}
		const count = host.querySelector(".abl-dr-count");
		if (count) count.textContent = selected.size
			? `${selected.size} selected — release, then ⚡ compute selected`
			: `${variants.length} variant${variants.length === 1 ? "" : "s"} — drag to select`;
	});

	document.addEventListener("mouseup", () => {
		if (!drag) return;
		const wasDrag = drag.moved;
		if (drag.rect) drag.rect.remove();
		drag = null;
		if (!wasDrag) return; // a plain click → let the row's ⚡ handler run
		// Swallow the click that trails the drag so a button under the release point doesn't fire.
		const eat = (ev) => { ev.stopPropagation(); ev.preventDefault(); };
		document.addEventListener("click", eat, { capture: true, once: true });
		setTimeout(() => document.removeEventListener("click", eat, true), 0);
		paint(); // reflect the final selection (surfaces "⚡ compute selected (N)")
	});
}

let renderedCell = "";

export async function renderAblationDrawer() {
	const host = $("tf-ablation-panel");
	if (!host) return;
	initDrawerDrag();
	// Only reset state on an ACTUAL cell change — toggling tabs on the same cell
	// must not wipe an in-flight compute queue's colored progress.
	const cell = `${state.run}\u0000${state.slot}\u0000${state.model}`;
	const cellChanged = cell !== renderedCell;
	renderedCell = cell;
	const token = ++renderToken;
	if (cellChanged || !variants.length) {
		host.replaceChildren(el("div", { class: "empty-note", text: "loading ablation variants…" }));
	}
	await fetchVariants();
	if (token !== renderToken) return; // scene changed mid-load — abandon
	if (cellChanged) {
		if (computing) computeCancel = true; // wind down the previous cell's queue
		attnState.clear();
		treatedIdx.clear();
		selected.clear();
	}
	paint();             // show the list immediately (statuses "·" = checking)
	if (!variants.length) return;
	await refreshAttn(); // now cheap (/artifacts reads) — fills in ✓ / compute fast
	if (token !== renderToken) return;
	paint();
}
