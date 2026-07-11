// The /tf ablation drawer: for a scene tagged with ablations, lists this cell's
// ablation variants and lets you COMPUTE their treated-step attention right
// here — closing the loop (launch → re-infer treated step → compute attention →
// compare in the analysis workspace). Attention is what the ablation view
// overlays, so a variant is only useful once its treated step is computed.
//
// Two things make this feel live:
//   * The INITIAL status is hydration-free. Each variant's cell is asked for its
//     version-aware attention index (fresh / stale event indices) via a cheap
//     disk scan — no run hydration, no Modal poll — so the list resolves fast and
//     old (component-less) results read as stale, not done.
//   * The compute queue mirrors the board's launch queue: bounded concurrency,
//     colored per-variant state (queued / running / done / stale / error), a
//     progress bar + count, and a ⏹ stop. While a variant computes we poll the
//     SERVER-OWNED queue (attentionList — the same source the main /tf framework
//     mirrors) and use the same status priority (running > queued > computed >
//     error > stale > none): a backed-up queue keeps resetting the stall clock, so
//     it's never mistaken for a failure. Clicking ✕ re-reads that queue and
//     reports the truth — "still computing" for a queued item, the worker
//     exception for a genuine failure — instead of a raw status dump.
//
// Select a GROUP by dragging a rubber-band box across the rows (shift-drag adds
// to the selection), then "⚡ compute selected".

import { api, SERVER_URL } from "../../js/api.js";
import { el, toast, openModal } from "../../js/ui.js";
import { hasSceneContext } from "../../js/ablationcore.js";
import { state, $, ATTN_WINDOW } from "./state.js";
import { pool } from "./util.js";

const SKIP_STEPS = new Set(["library_match", "image_prompt"]); // matches /tf-steps

let variants = [];            // [{ name, kind, cut, tag }]
const byName = new Map();     // name -> variant record (kind/cut/tag) for the queue
const attnState = new Map();  // name -> "computed"|"queued"|"running"|"error"|"stale"|"none" (absent = checking)
const attnError = new Map();  // name -> last failure message (worker exception + code)
const treatedIdx = new Map(); // name -> its OWN treated-step event_index (stable, cached)
const selected = new Set();   // names picked via the rubber-band drag
let renderToken = 0;          // guards a stale scene's async load from repainting

// Unified compute queue — the SAME model as attnQueue.js (the main /tf framework),
// adapted for per-variant cells: a PLAN of variants we want computed, the set
// already dispatched to Modal, and a single self-terminating poll loop. There are
// NO per-variant blocking loops — enqueue is fire-and-forget and the loop mirrors
// the server-owned queue, tops the window up, and settles on its own.
const plan = new Set();       // variant names we want computed (≈ state.attnPlan.evs)
const sent = new Set();       // variants already enqueued this session (≈ plan.sent)
const attempts = new Map();   // name -> enqueue attempts (bounds re-dispatch of lost jobs)
const noneStreak = new Map(); // name -> consecutive polls a SENT variant read "none"
let planForce = false;        // recompute even when a (stale) file already exists
let pollTimer = null;         // the ONE poll loop (≈ attnQueue._attnPollTimer)
let queueTotal = 0, queueDone = 0; // batch progress for the header

function tagOf(abl) {
	const t = abl.treatment || {};
	const rep = Number(abl.replicate) || 1;
	return [t.shuffle_method || "order", t.xml_tags === false ? "noxml" : null, t.attend_target ? "att" : null, t.distractors ? `d${t.distractors}` : null, rep > 1 ? `r${rep}` : null]
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
				// GATE: only this cell's variants, AND only scene-context step kinds
				// (overall_bbox, *_root and other no-scene kinds are never ablatable).
				return a && a.base_run === state.run && a.slot === state.slot && a.model === state.model
					&& a.cut != null && hasSceneContext(a.target_step_kind);
			})
			.map((r) => ({ name: r.name, kind: r.ablation.target_step_kind, cut: Number(r.ablation.cut), tag: tagOf(r.ablation) }));
		byName.clear();
		for (const v of variants) byName.set(v.name, v);
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

// Status refresh in two phases so reload is BOTH fast AND consistent with the
// main framework (which mirrors the server-owned queue via attentionList):
//   Phase 1 — cheap DISK scan (attentionIndex: fresh/stale) for every variant, no
//     hydration / no Modal poll, so computed ✓ and stale land instantly.
//   Phase 2 — for variants still "none" (nothing on disk), ask the LIVE server
//     queue (attentionList — same source + status priority as attnQueue) so an
//     in-flight compute reads as queued/running (and a persisted error surfaces)
//     even after a full page reload, instead of looking un-started. Bounded to the
//     pending set so an all-done cell pays ZERO Modal polls.
// Variants the poll loop already owns must not be touched here (it's the writer
// for the plan); nor an in-flight compute mark this session.
async function refreshAttn() {
	const slot0 = state.slot;
	const model0 = state.model;
	const live = () => state.slot === slot0 && state.model === model0;
	await pool(variants, 12, async (v) => {
		if (!live() || plan.has(v.name)) return;
		const cur = attnState.get(v.name);
		if (cur === "queued" || cur === "running") return;
		try {
			const s = await api.attentionIndex(v.name, state.slot, state.model, { maxHeads: state.maxHeads });
			attnState.set(v.name, (s.fresh || []).length ? "computed" : (s.stale || []).length ? "stale" : "none");
		} catch { attnState.set(v.name, "none"); }
	});
	if (!live()) return;
	paint(); // disk truth shown immediately; the live-queue pass fills in below
	const pending = variants.filter((v) => attnState.get(v.name) === "none" && !plan.has(v.name));
	if (!pending.length) return;
	await pool(pending, 8, async (v) => {
		if (!live() || plan.has(v.name) || attnState.get(v.name) !== "none") return;
		try {
			const idx = await resolveTreated(v);
			if (idx == null || !live()) return;
			const s = await api.attentionList(v.name, state.slot, state.model, { maxHeads: state.maxHeads });
			if (statusFromSnapshot(s, idx) !== "none") applyVariantStatus(v.name, s, idx); // one writer; keep the ⚡ if truly not started
		} catch { /* leave as none — the ⚡ still lets them start it */ }
	});
}

// ---- unified compute queue (the attnQueue.js model, per-variant cells) -----

const MAX_ATTEMPTS = 3;       // re-dispatch a lost/lagged enqueue at most this many times
const NONE_GRACE = 3;         // …only after this many consecutive "none" polls (registration lag)
const _has = (arr, idx) => (arr || []).some((x) => Number(x) === idx);
const _pending = (n) => { const st = attnState.get(n); return st === "queued" || st === "running"; };
function queueActive() { return plan.size > 0; }

// Normalize a cell's attentionList snapshot for ONE event index into the SAME
// status vocabulary the main framework uses (running > queued > computed > error >
// stale > none): a live (re)compute wins over a file on disk, an error wins over a
// lingering stale file. ("computed" is this view's word for the main "ready".)
function statusFromSnapshot(s, idx) {
	const errs = s.errors || {};
	if (_has(s.running, idx)) return "running";
	if (_has(s.queued, idx)) return "queued";
	if (_has(s.computed, idx)) return "computed";
	if ((errs[idx] ?? errs[String(idx)]) != null) return "error";
	if (_has(s.stale, idx)) return "stale";
	return "none";
}

// THE ONE place a variant's status is written from a server poll (≈ attnQueue
// .applyServerStatus). Refresh, the poll loop and "why" all funnel through here so
// the map can never disagree with itself.
function applyVariantStatus(name, s, idx) {
	const st = statusFromSnapshot(s, idx);
	if (st === "error") { const e = s.errors || {}; attnError.set(name, String(e[idx] ?? e[String(idx)] ?? "attention compute failed (worker error)")); }
	else if (st === "computed" || st === "stale") attnError.delete(name);
	attnState.set(name, st);
	return st;
}

// Align the optimistic mark with what the server accepted (≈ attnQueue
// .reconcileEnqueue). We send exactly one index, so each list is [idx] or [].
function reconcileEnqueue(name, resp, force) {
	if (!force && (resp?.skipped_fresh || []).length) attnState.set(name, "computed"); // already fresh — nothing to do
	else if ((resp?.queued || []).length || (resp?.already_active || []).length) attnState.set(name, "queued");
	else if (force) attnState.set(name, "queued"); // server picks it up next poll
	else if (attnState.get(name) === "queued") attnState.set(name, "none"); // nothing happened
}

// Fire-and-forget enqueue of ONE variant's treated step, reconciling the response.
// Never blocks on the GPU compute — the poll loop observes completion.
async function enqueueVariant(n) {
	const v = byName.get(n);
	if (!v) { plan.delete(n); return; }
	// GATE: never send a no-scene-context kind (defensive — the list is filtered).
	if (!hasSceneContext(v.kind)) { attnState.set(n, "none"); plan.delete(n); return; }
	sent.add(n);
	attempts.set(n, (attempts.get(n) || 0) + 1);
	noneStreak.delete(n);
	const force = planForce || attnState.get(n) === "stale"; // recompute-to-refresh forces past the stale file
	attnError.delete(n);
	const idx = await resolveTreated(v);
	if (idx == null) { attnState.set(n, "error"); attnError.set(n, "couldn't resolve the treated step's event index from the variant's events.jsonl"); plan.delete(n); queueDone += 1; return; }
	attnState.set(n, "queued");
	try {
		const resp = await api.attentionEnqueue(v.name, state.slot, state.model, { eventIndices: [idx], force, maxHeads: state.maxHeads });
		reconcileEnqueue(n, resp, force);
		if (attnState.get(n) === "computed") { plan.delete(n); queueDone += 1; } // server said already fresh
	} catch (e) {
		sent.delete(n); // let the next pump retry this one
		attnState.set(n, "error"); attnError.set(n, `enqueue failed: ${e.message || e}`); plan.delete(n); queueDone += 1;
	}
}

// Dispatch the next variants so at most ATTN_WINDOW are OUTSTANDING on Modal
// (≈ attnQueue.attnPump). Outstanding = SENT variants still queued/running — an
// optimistic, not-yet-sent "queued" doesn't occupy the GPU, so it isn't counted.
async function drawerPump() {
	if (!plan.size) return;
	const slot0 = state.slot, model0 = state.model;
	const outstanding = [...plan].filter((n) => sent.has(n) && _pending(n)).length;
	const room = ATTN_WINDOW - outstanding;
	if (room <= 0) return;
	const toSend = [];
	for (const n of plan) {
		if (sent.has(n)) continue;
		const st = attnState.get(n);
		if (st === "computed" || st === "running") continue;
		toSend.push(n);
		if (toSend.length >= room) break;
	}
	if (!toSend.length) return;
	await pool(toSend, Math.min(toSend.length, 6), async (n) => {
		if (state.slot !== slot0 || state.model !== model0 || !plan.has(n)) return;
		await enqueueVariant(n);
	});
}

// The ONE self-terminating poll loop (≈ attnQueue.startAttnPoll → syncAttnStatus →
// attnPlanReconcile). Each tick: mirror every SENT-and-unsettled variant through
// the single writer, drop the finished/failed from the plan, re-dispatch lost jobs,
// top the window up, and reschedule while anything is still pending.
function startPoll() {
	if (pollTimer) return;
	const tick = async () => {
		pollTimer = null;
		const slot0 = state.slot, model0 = state.model;
		const watch = [...plan].filter((n) => sent.has(n));
		await pool(watch, 6, async (n) => {
			if (state.slot !== slot0 || state.model !== model0 || !plan.has(n)) return;
			const v = byName.get(n); if (!v) return;
			try {
				const idx = await resolveTreated(v);
				if (idx == null) return;
				const s = await api.attentionList(v.name, state.slot, state.model, { maxHeads: state.maxHeads });
				applyVariantStatus(n, s, idx);
			} catch { /* transient — keep the current mark, retry next tick */ }
		});
		if (state.slot !== slot0 || state.model !== model0) { plan.clear(); sent.clear(); return; } // switched cell → abandon
		for (const n of [...plan]) {
			if (!sent.has(n)) continue; // not dispatched yet — leave the optimistic "queued"
			const st = attnState.get(n);
			if (st === "computed" || st === "stale" || st === "error") { plan.delete(n); queueDone += 1; continue; }
			if (st === "running" || st === "queued") { noneStreak.delete(n); continue; } // server has it → keep waiting
			// "none": the server doesn't (yet) know this job. Tolerate registration lag,
			// then re-dispatch a lost enqueue; give up after MAX_ATTEMPTS.
			const streak = (noneStreak.get(n) || 0) + 1; noneStreak.set(n, streak);
			if (streak < NONE_GRACE) { attnState.set(n, "queued"); continue; }
			if ((attempts.get(n) || 0) < MAX_ATTEMPTS) { sent.delete(n); noneStreak.delete(n); attnState.set(n, "queued"); } // re-pump
			else { attnState.set(n, "error"); attnError.set(n, `the server never registered this compute after ${MAX_ATTEMPTS} attempts — the worker may be down or rejecting it (failed stamp check).`); plan.delete(n); queueDone += 1; }
		}
		await drawerPump(); // top the window back up as jobs finish
		paint();
		if (queueActive()) pollTimer = setTimeout(tick, 1200);
		else { queueTotal = 0; queueDone = 0; attempts.clear(); noneStreak.clear(); paint(); } // whole plan resolved → settle
	};
	pollTimer = setTimeout(tick, 400);
}

// Stop requeueing + polling; jobs already on Modal drain on their own and reappear
// on the next refresh (≈ leaving the page in the main framework).
function stopQueue() {
	plan.clear(); sent.clear(); attempts.clear(); noneStreak.clear();
	if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
	queueTotal = 0; queueDone = 0;
	paint();
}

// Explain a variant's status in a modal. Re-reads the LIVE server queue (the same
// attentionList the main framework uses) and RECONCILES the row before deciding —
// so a variant that is actually still queued/running (not failed) is reported as
// "still computing", not dumped as a raw failure. Only a genuine worker error /
// stall shows the red diagnosis.
async function showError(v) {
	let msg = attnError.get(v.name);
	let live = attnState.get(v.name);
	let idx = null;
	try {
		idx = await resolveTreated(v);
		const s = await api.attentionList(v.name, state.slot, state.model, { maxHeads: state.maxHeads });
		live = statusFromSnapshot(s, idx);
		const errs = s.errors || {};
		msg = errs[idx] ?? errs[String(idx)] ?? msg;
		if (!plan.has(v.name) || live === "computed" || live === "error") { attnState.set(v.name, live); paint(); } // self-heal a stale-shown row
	} catch { /* keep the cached status/message */ }
	const busy = live === "queued" || live === "running";
	const title = busy ? `still computing · ${v.kind}@${v.cut}` : live === "computed" ? `computed · ${v.kind}@${v.cut}` : `attention failed · ${v.kind}@${v.cut}`;
	const body = busy
		? `This variant hasn't failed — its treated step (event ${idx ?? "?"}) is ${live} in the server attention queue. The row turns ✓ when the GPU result lands; nothing to do but wait (or ⏹ stop the queue).`
		: live === "computed"
			? "This variant is computed now — open workspace → ablation to compare it."
			: (msg || "no error reported — it likely left the queue without a result landing (stamp check failed / superseded prompt), or landed STALE. Try ↻ recompute; if it persists, redeploy the Modal worker to the current ANALYSIS_VERSION.");
	const tone = busy ? "color:#e8d48a;background:#1c1a12;border:1px solid rgba(232,193,74,0.3)" : live === "computed" ? "color:#8fe0a0;background:#12190f;border:1px solid rgba(107,217,110,0.3)" : "color:#ff9e9e;background:#1a1216;border:1px solid rgba(255,128,128,0.3)";
	openModal(title, (close) => ({
		body: [
			el("div", { class: "abl-dr-tag", style: "margin-bottom:8px;word-break:break-all", text: v.name }),
			el("pre", { style: `white-space:pre-wrap;word-break:break-word;font-size:12px;max-width:560px;border-radius:8px;padding:10px;margin:0;${tone}` }, body),
		],
		actions: [
			...(busy ? [] : [el("button", { class: "abl-dr-btn primary", onclick: () => { close(); computeMany([v]); } }, live === "computed" ? "↻ recompute" : "↻ retry")]),
			el("button", { class: "abl-dr-btn", onclick: close }, "close"),
		],
	}));
}

// Public entry (⚡ buttons): add variants to the PLAN, kick the window, and start
// the single poll loop — which handles dispatch, top-up and settle. Additive: a
// second ⚡ while a queue is running just extends the same plan (like the main
// framework's compute-all), so there's no "already running" bounce.
function computeMany(list) {
	const todo = (list || []).filter((v) => hasSceneContext(v.kind) && attnState.get(v.name) !== "computed" && !plan.has(v.name));
	if (!todo.length) { toast(queueActive() ? "those are already queued" : "nothing to compute — already done"); return; }
	if (!queueActive()) { queueTotal = 0; queueDone = 0; } // fresh batch → reset the counter
	queueTotal += todo.length;
	for (const v of todo) {
		byName.set(v.name, v);
		plan.add(v.name);
		attempts.delete(v.name); noneStreak.delete(v.name);
		if (attnState.get(v.name) !== "stale") attnState.set(v.name, "queued"); // keep stale so enqueue forces past it
	}
	selected.clear(); // the queue's colored state is now the view; drop the picker highlight
	paint();
	drawerPump().catch(() => {}).then(startPoll); // dispatch first window, then poll
}

// ---- rendering ------------------------------------------------------------

// Absent status = still checking (shown as a faint pulsing dot), so the list
// appears instantly and fills in without ever flashing "no variants".
function paint() {
	const host = $("tf-ablation-panel");
	if (!host || drag) return; // never rebuild rows out from under an active drag
	if (!variants.length) { host.replaceChildren(el("div", { class: "empty-note", text: "no ablation variants for this scene / model" })); return; }

	const computedN = variants.filter((v) => attnState.get(v.name) === "computed").length;
	const staleN = variants.filter((v) => attnState.get(v.name) === "stale").length;
	const failedN = variants.filter((v) => attnState.get(v.name) === "error").length;
	const checking = variants.some((v) => !attnState.has(v.name));
	const active = queueActive();
	const runningN = variants.filter((v) => attnState.get(v.name) === "running").length;
	const status = active
		? `queue ${queueDone}/${queueTotal} · ${runningN} running · ≤${ATTN_WINDOW} outstanding`
		: checking ? "checking…" : `${computedN}/${variants.length} attention ✓${staleN ? ` · ${staleN} stale` : ""}${failedN ? ` · ${failedN} failed` : ""}`;

	const selBtn = selected.size
		? el("button", { class: "abl-dr-btn primary", title: "compute the treated-step attention for the dragged selection",
			onclick: () => computeMany(variants.filter((v) => selected.has(v.name))) }, `⚡ compute selected (${selected.size})`)
		: null;
	const clearBtn = selected.size
		? el("button", { class: "abl-dr-btn sm", title: "clear selection", onclick: () => { selected.clear(); paint(); } }, "clear")
		: null;
	const mainBtn = active
		? el("button", { class: "abl-dr-btn danger", title: "stop queueing (jobs already on the GPU finish + reappear on refresh)", onclick: stopQueue }, "⏹ stop")
		: el("button", { class: "abl-dr-btn", title: `compute every variant's treated-step attention — ${ATTN_WINDOW} kept outstanding on the GPU, the rest queue`,
			onclick: () => computeMany(variants) }, "⚡ compute all");

	const head = el("div", { class: "abl-dr-head" },
		el("span", { class: "abl-dr-count", text: `${variants.length} variant${variants.length === 1 ? "" : "s"} · ${status}` }),
		el("span", { class: "abl-dr-grow" }),
		selBtn, clearBtn, mainBtn);

	const bar = active
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
				: st === "error" ? "error" : st === "stale" ? "stale" : st === "none" ? "none" : "checking";
			const action = st === "computed"
				? el("span", { class: "abl-dr-ok", title: "attention computed", text: "✓" })
				: st === "running"
					? el("span", { class: "abl-dr-pending", title: "computing on GPU", text: "●" })
					: st === "queued"
						? el("span", { class: "abl-dr-pending", title: "queued", text: "…" })
						: st === "error"
							? el("button", { class: "abl-dr-btn sm danger", title: attnError.get(v.name) || "click to diagnose — re-reads the live server queue", onclick: (e) => { e.stopPropagation(); showError(v); } }, "✕ why")
							: st === "stale"
								? el("button", { class: "abl-dr-btn sm", title: "attention predates the per-object attribute data — recompute to refresh", onclick: (e) => { e.stopPropagation(); computeMany([v]); } }, "↻")
								: st === "none"
									? el("button", { class: "abl-dr-btn sm", title: "compute this variant's treated-step attention", onclick: (e) => { e.stopPropagation(); computeMany([v]); } }, "⚡")
									: el("span", { class: "abl-dr-pending", title: "checking attention status", text: "·" });
			const row = el("div", { class: `abl-dr-row${selected.has(v.name) ? " sel" : ""}${st === "error" ? " err" : ""}`, title: st === "error" ? (attnError.get(v.name) || v.name) : v.name },
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
		stopQueue();       // wind down the previous cell's queue (clears plan/sent/poll)
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
