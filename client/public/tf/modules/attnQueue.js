// Attention compute orchestration: the model-open gate + badge, the windowed
// "compute all" / bbox plan (keeps only ATTN_WINDOW jobs on Modal, tops up as
// they finish), the server-status mirror (the ONE place per-step status is
// written from a server response), the self-terminating poll loop, and the
// single-step compute.

import { el } from "../../js/ui.js";
import { api } from "../../js/api.js";
import { $, state, ATTN_WINDOW, ATTN_STATUS_LABEL, BBOX_STEP_TEMPLATES, dropStepCaches } from "./state.js";
import { renderTimeline, refreshTimeline, loadAttention } from "./render.js";
import { renderAttention } from "./attnPanel.js";
import { updateReportCtx, syncTabHighlight } from "./tabs.js";
import * as report from "./report.js";

// Attention analysis needs OPEN HF weights; the server (tf-steps.attention) marks
// closed/API models (gpt/claude/gemini/...) as gated. These drive the compute
// buttons + the model badge.
function attnModelOpen() { return state.attnModel?.open !== false; }
function closedModelMsg() {
	return `${state.attnModel?.model_id || state.model} is a closed model — attention needs open HF weights`;
}

// Small status in the compute bar: an HF link for open models, or a gated note.
export function renderAttnModelBadge() {
	const host = $("tf-attn-model");
	if (!host) return;
	const m = state.attnModel || {};
	const open = m.open !== false, disabled = !open;
	for (const id of ["tf-attn-all", "tf-attn-bbox"]) { const b = $(id); if (b) b.disabled = disabled; }
	if (open && m.hf_url) {
		host.replaceChildren(el("a", { class: "pill", href: m.hf_url, target: "_blank", rel: "noreferrer",
			title: `open weights on Hugging Face${m.gpu ? ` · runs on ${m.gpu}` : ""}`, text: `🤗 ${m.hf_path || "open"}` }));
	} else if (open) {
		host.replaceChildren();
	} else {
		host.replaceChildren(el("span", { class: "attn-closed", title: closedModelMsg(),
			text: "closed model · attention needs open weights" }));
	}
}

// GLOBAL "compute attention · all steps": rather than dumping the whole cell onto
// Modal, register a windowed PLAN (see startAttnPlan) that keeps only ATTN_WINDOW
// jobs outstanding and tops up as they finish. The GPU queue stays small and, if
// the user navigates away or stops, we simply stop requeueing and Modal drains.
export async function computeAllAttention() {
	if (!state.steps.length) return;
	if (!attnModelOpen()) { $("tf-attn-progress").textContent = closedModelMsg(); return; }
	const force = $("tf-attn-force").checked; // else additive: server reuses unchanged steps
	const prog = $("tf-attn-progress");
	// Skip steps with no scene context (root plans / overall bbox) — there'd be
	// no scene-attending heads, so don't queue them at all. (has_scene may be
	// undefined on an older server response → don't filter in that case.)
	const evs = state.steps.filter((s) => s.has_scene !== false).map((s) => s.event_index);
	if (!evs.length) { prog.textContent = "no steps with scene context"; return; }
	if (force) {
		// Force: every step re-enters the plan — including ones already "ready".
		for (const ev of evs) {
			state.attnStatus[ev] = "queued";
			state.attnPendingReload.add(ev);
			dropStepCaches(ev);
			if (isCurrent(ev) && state.attn?.meta?.event_index === ev) state.attn = null;
		}
	} else {
		const needsWork = (ev) => {
			const st = state.attnStatus[ev];
			return st === "none" || st === "stale" || st === "error";
		};
		for (const ev of evs) if (needsWork(ev)) state.attnStatus[ev] = "queued";
	}
	renderTimeline(); renderAttention();
	prog.textContent = force ? `queuing ${evs.length} steps…` : "queuing…";
	await startAttnPlan(evs, force);
}

// FORCE-recompute only the bbox-batch steps — refreshes the to-place readout
// without touching (or depending on the freshness of) every other step. Uses
// force=true so it ALWAYS recomputes, regardless of the server's is_fresh state.
export async function recomputeBbox() {
	if (!state.steps.length) return;
	if (!attnModelOpen()) { $("tf-attn-progress").textContent = closedModelMsg(); return; }
	const prog = $("tf-attn-progress");
	const evs = state.steps
		.filter((s) => BBOX_STEP_TEMPLATES.has(s.template) && s.has_scene !== false)
		.map((s) => s.event_index);
	if (!evs.length) { prog.textContent = "no bbox-batch steps in this cell"; return; }
	for (const ev of evs) { state.attnStatus[ev] = "queued"; state.attnPendingReload.add(ev); dropStepCaches(ev); }
	renderTimeline(); renderAttention();
	prog.textContent = `queuing ${evs.length} bbox step${evs.length > 1 ? "s" : ""}…`;
	await startAttnPlan(evs, true);
}

// Per-step status tallied into queue counts (drives the counts row, the
// progress bar, and the top-bar progress text).
function attnCounts() {
	let ready = 0, running = 0, queued = 0, stale = 0, errors = 0;
	for (const s of state.steps) {
		const st = state.attnStatus[s.event_index];
		if (st === "ready") ready++;
		else if (st === "running") running++;
		else if (st === "queued") queued++;
		else if (st === "stale") stale++;
		else if (st === "error") errors++;
	}
	return { ready, running, queued, stale, errors, total: state.steps.length };
}

// Compact top-bar progress line (next to the ⚡ button).
export function updateBatchProgress() {
	const prog = $("tf-attn-progress");
	if (!prog || !state.steps.length) return;
	const c = attnCounts();
	const tail = `${c.stale ? ` · ${c.stale} stale` : ""}${c.errors ? ` · ${c.errors} failed` : ""}`;
	prog.textContent = (c.running || c.queued)
		? `${c.ready}/${c.total} · ${c.running} running · ${c.queued} queued${tail}`
		: `✓ ${c.ready}/${c.total}${tail}`;
}

// Normalize event indices from the server (JSON may stringify dict keys).
function evSet(arr) { return new Set((arr ?? []).map((x) => Number(x))); }
function evErr(errors, ev) { return errors[String(ev)] ?? errors[ev] ?? null; }

// After enqueue: align optimistic queue marks with what the server accepted.
function reconcileEnqueue(requested, resp, { force = false } = {}) {
	const queued = evSet(resp?.queued);
	const skipped = evSet(resp?.skipped_fresh);
	const active = evSet(resp?.already_active);
	for (const ev of requested) {
		if (!force && skipped.has(ev)) state.attnStatus[ev] = "ready";
		else if (queued.has(ev) || active.has(ev)) state.attnStatus[ev] = active.has(ev) ? (state.attnStatus[ev] || "queued") : "queued";
		else if (force) state.attnStatus[ev] = "queued"; // server will pick up on next poll
		else if (state.attnStatus[ev] === "queued") state.attnStatus[ev] = "none";
	}
}

// --- windowed compute-all plan ----------------------------------------------
// We don't hand Modal the whole cell at once; we keep at most ATTN_WINDOW jobs
// outstanding and dispatch the next batch only as earlier ones finish. The poll
// loop drives the top-ups (attnPump), so if the user leaves / switches cell we
// just stop requeueing and Modal's small queue drains on its own.

function attnPlanActive() { return !!(state.attnPlan && state.attnPlan.evs.size); }

// Register a compute-all/bbox plan and dispatch the first window immediately.
async function startAttnPlan(evs, force) {
	state.attnPlan = { evs: new Set(evs.map(Number)), sent: new Set(), force: !!force };
	$("tf-attn-all").disabled = true;
	if ($("tf-attn-bbox")) $("tf-attn-bbox").disabled = true;
	// NB: compute-all does NOT reset the queue — the GPU queue is shared per model,
	// so resetting here would nuke OTHER scenes' pending compute. Stale pending is
	// cleared on server restart instead (see the server-startup reset).
	await syncAttnStatus(); // current server truth → drops already-done, marks the rest queued
	if (state.attnPlan) await attnPump(); // send the first ≤ATTN_WINDOW batch (unless we switched cell)
	renderTimeline(); renderAttention(); updateBatchProgress();
	startAttnPoll(); // keeps pumping + polling until the plan (and queue) is empty
}

// Dispatch the next batch so at most ATTN_WINDOW jobs are outstanding on Modal.
// Outstanding is measured from the SERVER snapshot (not our optimistic marks),
// so we never run ahead of what the GPU is actually holding.
async function attnPump() {
	const plan = state.attnPlan;
	if (!plan || !plan.evs.size) return;
	const srv = state.attnServer || { queued: [], running: [] };
	const outstanding = (srv.queued?.length || 0) + (srv.running?.length || 0);
	const room = ATTN_WINDOW - outstanding;
	if (room <= 0) return; // Modal is full — wait for jobs to finish before topping up
	const onServer = new Set([...(srv.queued || []), ...(srv.running || [])].map(Number));
	const toSend = [];
	for (const ev of plan.evs) {
		if (plan.sent.has(ev) || onServer.has(ev)) continue; // already dispatched / on Modal
		toSend.push(ev);
		if (toSend.length >= room) break;
	}
	if (!toSend.length) return;
	for (const ev of toSend) plan.sent.add(ev);
	try {
		const resp = await api.attentionEnqueue(state.run, state.slot, state.model,
			{ eventIndices: toSend, force: plan.force, maxHeads: state.maxHeads });
		reconcileEnqueue(toSend, resp, { force: plan.force });
	} catch (e) {
		for (const ev of toSend) plan.sent.delete(ev); // let the next tick retry this batch
		$("tf-attn-progress").textContent = `enqueue failed: ${e.message}`;
	}
}

// Reconcile the plan against fresh server status: drop finished/failed steps and
// keep still-pending ones showing "queued". Clears the plan once every step is
// accounted for (which lets the poll loop settle + re-enable the buttons).
function attnPlanReconcile() {
	const plan = state.attnPlan;
	if (!plan) return;
	for (const ev of [...plan.evs]) {
		const st = state.attnStatus[ev];
		// Non-force: any stored result is "done". Force: only once OUR recompute
		// (post-dispatch) has landed — otherwise the pre-existing file looks done.
		const finished = plan.force ? (st === "ready" && plan.sent.has(ev)) : (st === "ready");
		const failed = st === "error" && plan.sent.has(ev); // surfaced already; don't retry-loop
		if (finished || failed) { plan.evs.delete(ev); continue; }
		if (st !== "running") state.attnStatus[ev] = "queued"; // planned/in-flight → show pending
	}
	if (!plan.evs.size) state.attnPlan = null; // whole plan resolved
}

// Mirror the server-owned queue (computed / running / queued / errors) into the
// per-step status map — the ONE place status is written from a server response.
export function applyServerStatus(r) {
	const ready = evSet(r.computed);
	const stale = evSet(r.stale);
	const running = evSet(r.running);
	const queued = evSet(r.queued);
	const errors = r.errors ?? {};
	state.attnErrors = Object.fromEntries(Object.entries(errors).map(([k, v]) => [Number(k), v]));
	// Stash the raw queue snapshot so attnPump can size the next window against
	// what Modal is ACTUALLY holding (not our optimistic per-step marks).
	state.attnServer = { queued: [...queued], running: [...running], computed: [...ready] };
	for (const step of state.steps) {
		const ev = Number(step.event_index);
		const prev = state.attnStatus[ev];
		// If the shown step silently re-enters the queue (server chose to recompute
		// it — content changed, or the top-N head count was raised past what's
		// stored), its displayed copy is now stale: reload it once it finishes.
		if ((running.has(ev) || queued.has(ev)) && isCurrent(ev)) state.attnPendingReload.add(ev);
		// running/queued win over ready/stale: a (force) recompute can be in flight
		// while a file still exists — show it as busy, not done. "stale" = on disk
		// but an old analysis version (needs recompute), distinct from fresh "ready".
		if (running.has(ev)) { state.attnStatus[ev] = "running"; state.stepAnalyses.delete(ev); } // recompute → drop cached overview data
		else if (queued.has(ev)) state.attnStatus[ev] = "queued";
		else if (ready.has(ev)) {
			// A step can ENTER ready without our ever seeing it running/queued to arm
			// the reload above (line ~525): a fast ADOPT lands between two polls, or a
			// pre-written MOCK on disk (which reads as "stale") is swapped for the real
			// GPU result (stale→ready). If we're already caching/showing a copy, it's
			// now out of date — arm a one-shot reload (busts loadAttention's cache
			// serve) and drop the overview analysis so cell-wide stats repull too.
			// Without this the stale/mock map sticks until a full page reload.
			const showing = state.attn?.meta?.event_index === ev; // may be held even if LRU-evicted from compactCache
			if (prev !== "ready" && (showing || state.compactCache.has(ev) || state.stepAnalyses.has(ev))) {
				state.attnPendingReload.add(ev);
				state.stepAnalyses.delete(ev);
			}
			state.attnStatus[ev] = "ready";
		}
		// A failed (re)compute wins over a lingering stale file: surface the
		// failure instead of silently reverting to the outdated map.
		else if (evErr(errors, ev) != null) state.attnStatus[ev] = "error";
		else if (stale.has(ev)) state.attnStatus[ev] = "stale";
		else if (["ready", "stale", "error"].includes(prev)) state.attnStatus[ev] = "none"; // file gone, no error
		else if (prev === "queued" || prev === "running") state.attnStatus[ev] = "none"; // server lost track — don't spin forever
	}
	attnPlanReconcile(); // drop finished steps from the window; keep the rest "queued"
	report.setSceneAttentionCount(state.slot, r);
	syncTabHighlight();
}

// Anything still working in this cell?
export function hasPending() {
	return state.steps.some((s) => {
		const st = state.attnStatus[s.event_index];
		return st === "queued" || st === "running";
	});
}

function isCurrent(ev) {
	const step = state.steps[state.stepIdx];
	return !!step && step.event_index === ev;
}

// Refresh ONLY the current step's status pill + compute button (used while the
// step's result is already displayed, so polling doesn't rebuild its body).
function refreshHead(ev) {
	const st = state.attnStatus[ev] || "none";
	const pill = document.getElementById("tf-attn-stpill");
	if (pill) { pill.className = `attn-stpill st-${st}`; pill.textContent = ATTN_STATUS_LABEL[st]; }
	const btn = document.getElementById("tf-attn-recompute");
	if (btn) {
		const displayed = !!(state.attn && state.attn.meta && state.attn.meta.event_index === ev);
		btn.textContent = (displayed || st === "ready") ? "↻ recompute this step" : "▶ compute this step";
		btn.disabled = !(state.export && st !== "running" && st !== "queued");
	}
}

// Signature of the COMPUTED (ready) step set — the only thing a status poll can
// change that the report workspace actually renders. Gating the rebuild on this
// (instead of rebuilding every poll) keeps the analysis view + scroll position
// stable while a batch computes; a rebuild happens only when a new result lands.
let _lastReportReady = null;
function reportReadySig() {
	const s = state.attnStatus || {};
	const ready = [];
	for (const ev in s) if (s[ev] === "ready") ready.push(ev);
	return ready.sort((a, b) => Number(a) - Number(b)).join(",");
}
// Throttle the poll-driven rebuild: during an active batch a step finishes every
// poll, so an un-throttled rebuild rebuilds the whole (heavy) workspace every tick
// and the view feels frozen. Coalesce to at most one rebuild per REPORT_RENDER_MS.
const REPORT_RENDER_MS = 1500;
let _reportRenderAt = 0, _reportRenderTimer = null;
function _renderReportNow() {
	_reportRenderAt = Date.now(); _reportRenderTimer = null;
	if (state.reportView) report.renderReportWorkspace({ scrollSelection: false });
}
function scheduleReportRender() {
	const since = Date.now() - _reportRenderAt;
	if (since >= REPORT_RENDER_MS) { _renderReportNow(); return; }
	if (!_reportRenderTimer) _reportRenderTimer = setTimeout(_renderReportNow, REPORT_RENDER_MS - since);
}
export function resetReportReadySig() {
	_lastReportReady = null;
	if (_reportRenderTimer) { clearTimeout(_reportRenderTimer); _reportRenderTimer = null; }
}

// One poll: pull the server-owned queue, mirror it into per-step status, and
// repaint the (cheap) queue panel. The heavy attention body is only touched
// when the current step needs it — loaded when it just turned ready (or its
// displayed copy went stale via a recompute); otherwise just its status pill is
// refreshed, so a result you're viewing isn't rebuilt from under you.
export async function syncAttnStatus() {
	const run = state.run, slot = state.slot, model = state.model;
	let r;
	try { r = await api.attentionList(run, slot, model, { maxHeads: state.maxHeads }); } catch { return false; }
	if (state.run !== run || state.slot !== slot || state.model !== model) return false; // switched cell
	applyServerStatus(r);
	refreshTimeline();
	updateBatchProgress();
	updateReportCtx();
	// Only rebuild the report when the computed-step set actually changed — a poll
	// that changes nothing (or only queued/running progress) must NOT rebuild it,
	// or it snaps scroll + flickers the charts under the user.
	if (state.reportView) {
		const sig = reportReadySig();
		if (sig !== _lastReportReady) { _lastReportReady = sig; scheduleReportRender(); }
	}
	const cur = state.steps[state.stepIdx];
	const ev = cur && cur.event_index;
	if (ev == null) return true;
	const displayed = !!(state.attn && state.attn.meta && state.attn.meta.event_index === ev);
	if (state.attnStatus[ev] === "ready" && (!displayed || state.attnPendingReload.has(ev))) {
		loadAttention(ev, state.viewToken); // just became ready (or a recompute finished) → load it
	} else if (displayed) {
		refreshHead(ev); // result already shown: only update the status pill/button
	} else {
		renderAttention(); // still waiting: keep the queued/running/error body fresh
	}
	return true;
}

let _attnPollTimer = null;
// Poll the server for status while anything is queued/running, then stop and
// settle the UI. Self-terminating so we don't hammer the API when idle.
export function startAttnPoll() {
	if (_attnPollTimer) return;
	const tick = async () => {
		_attnPollTimer = null;
		const ok = await syncAttnStatus();
		if (ok) await attnPump(); // frontend-driven requeue: top the window back up as jobs finish
		// Keep polling while the GPU is busy OR the plan still has steps to dispatch;
		// there can be a lull between a window finishing and the next one landing.
		if (hasPending() || attnPlanActive() || !ok) {
			_attnPollTimer = setTimeout(tick, ok ? 900 : 1500);
		} else {
			state.attnPlan = null; // plan complete — stop requeueing (Modal's queue is empty)
			$("tf-attn-all").disabled = false; // queue drained — re-enable both compute buttons
			if ($("tf-attn-bbox")) $("tf-attn-bbox").disabled = false;
			updateBatchProgress();
		}
	};
	_attnPollTimer = setTimeout(tick, 400);
}

// Compute (or force-recompute) THIS step: enqueue on the server and let the poll
// pick up the result — no blocking POST held open for the whole GPU compute.
export function computeStep(ev, force) {
	if (!attnModelOpen()) { $("tf-attn-progress").textContent = closedModelMsg(); return; }
	state.attnStatus[ev] = "queued";
	if (force) {
		state.attnPendingReload.add(ev);
		dropStepCaches(ev);
		if (isCurrent(ev) && state.attn?.meta?.event_index === ev) state.attn = null;
	}
	renderTimeline(); renderAttention();
	api.attentionEnqueue(state.run, state.slot, state.model, { eventIndices: [ev], force, maxHeads: state.maxHeads })
		.then(async (resp) => { reconcileEnqueue([ev], resp, { force }); await syncAttnStatus(); startAttnPoll(); })
		.catch((e) => {
			state.attnStatus[ev] = "error";
			state.attnErrors = { ...(state.attnErrors || {}), [ev]: e.message };
			renderTimeline(); renderAttention();
		});
}
