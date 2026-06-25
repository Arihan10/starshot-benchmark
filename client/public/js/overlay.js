// The slot overlay: enlarged 3D view of one cell (source run or its
// simulation branch) with the observability dock on the right.

import { api } from "./api.js";
import {
	state,
	emit,
	on,
	cellSummary,
	cellBranches,
	branchSummaryById,
} from "./state.js";
import { el, toast, stepUntilSelect, openModal } from "./ui.js";
import {
	openStream,
	dispatchSceneEvent,
	applySceneProjection,
	createObsModel,
	emittedStep,
} from "./events.js";
import { statusView } from "./status.js";
import * as obstree from "./obstree.js";
import * as inquiry from "./inquiry.js";

const overlayEl = document.getElementById("overlay");
const titleEl = document.getElementById("overlay-title");
const crumbsEl = document.getElementById("overlay-crumbs");
const dotEl = document.getElementById("overlay-dot");
const statusEl = document.getElementById("overlay-status");
const actionBtn = document.getElementById("overlay-action");
const resetBtn = document.getElementById("overlay-reset");
const btnZoneLayers = document.getElementById("btn-zone-layers");
const zoneLayersLegendEl = document.getElementById("zone-layers-legend");

let viewer = null;
let stream = null;
let obs = createObsModel();
let renderQueued = false;
let openSeq = 0; // monotonically increasing guard for async open races
let lastLayersSig = null; // skips rebuilding the layer legend when unchanged

export function initOverlay(sceneViewer) {
	viewer = sceneViewer;

	// Tree ↔ 3D linking. A node-row click toggles selection and frames the
	// bbox (dimming the rest); a call click guarantees its node is focused
	// without toggling it off. 3D-side picks highlight + reveal the tree row,
	// and the hover tooltip reads seed/plan/image text from the obs model.
	viewer.setNodeInfo((id) => obs.model.nodes.get(id) ?? null);
	// Color objects in 3D by the decomposition step that emitted them (next_object
	// purple, anchor green, negative_space brown) — read from the same provenance
	// the tree shows as "via {step}". `recolorAll` (below) repaints once a load's
	// history has folded, since the scene projection paints bboxes first.
	viewer.setOriginOf((id) => emittedStep(obs.model, id));
	viewer.onSelect((id) => obstree.markSelected(id, { scroll: true }));
	obstree.setOnNodeClick((id, { ensureSelected = false } = {}) => {
		if (!viewer.hasBbox(id)) return;
		if (ensureSelected && viewer.getSelected() === id) return;
		viewer.select(id, { frame: true });
	});
	// Per-node hiding: the tree's eye buttons and the canvas right-click share
	// the viewer's hidden set; any change re-renders the tree so eye states
	// and row dimming follow.
	obstree.setHiddenApi({
		isHidden: viewer.isHidden,
		toggle: viewer.toggleHidden,
	});
	viewer.onHiddenChange(() => obstree.renderTree(obs.model));

	// "why?" on any call row → the decision-inquiry chat for that step. Read-only,
	// so it's wired once (source AND branch views) and reads the live view at
	// click time for the cell/run/branch context.
	obstree.setOnInquire((call) => {
		if (!state.view) return;
		const { slot, model, branch } = state.view;
		inquiry.openInquiry(call, { run: state.run, slot, model, branch });
	});

	document
		.getElementById("overlay-close")
		.addEventListener("click", closeOverlay);
	document.addEventListener("keydown", (ev) => {
		if (
			ev.key === "Escape" &&
			overlayEl.classList.contains("open") &&
			!document.getElementById("modal-root").firstChild
		) {
			closeOverlay();
			// When the full scene was opened from the compare view (stacked beneath),
			// don't let the SAME Escape also close compare — return to it instead.
			// The overlay's keydown is registered before compare's (see main init).
			ev.stopImmediatePropagation();
		}
	});
	for (const btn of document.querySelectorAll(
		"#viewer-toggles [data-toggle]",
	)) {
		const key = btn.dataset.toggle;
		const sync = () => btn.classList.toggle("off", !viewer.toggles[key]);
		btn.addEventListener("click", () => {
			viewer.toggles[key] = !viewer.toggles[key];
			viewer.refreshVisibility();
			sync();
		});
		sync();
	}
	// Zone-layers view: a dedicated mode (not a per-category toggle), so it gets
	// its own handler — flip the viewer's mode, then (re)build the depth legend.
	btnZoneLayers.addEventListener("click", () => {
		viewer.setZoneLayers(!viewer.getZoneLayers().enabled);
		refreshZoneLayers();
	});
	document
		.getElementById("btn-refit")
		.addEventListener("click", () => viewer.fit());
	actionBtn.addEventListener("click", onAction);
	resetBtn.addEventListener("click", onReset);
	on("slots", () => {
		if (state.view) renderHeader();
	});
	on("open-cell", openCell);
	initObsResizer();
	refreshZoneLayers();
}

// The zone-layers legend: a swatch per nesting depth present in the scene, plus
// an "all" chip. Depth chips multi-select — toggle any combination (e.g. L0 +
// L1) to show just those layers; "all" (or clearing every chip) restores the
// full set. Active chips are highlighted. Shown only while the viewer's
// zone-layers view is on, and rebuilt only when the depth set / active layers
// actually change so streamed growth doesn't thrash the row.
function refreshZoneLayers() {
	if (!viewer) return;
	const info = viewer.getZoneLayers();
	btnZoneLayers.classList.toggle("off", !info.enabled);
	zoneLayersLegendEl.style.display = info.enabled ? "" : "none";
	const sig = `${info.enabled}|${info.active.join(",")}|${info.layers.map((l) => l.depth).join(",")}`;
	if (sig === lastLayersSig) return;
	lastLayersSig = sig;
	zoneLayersLegendEl.textContent = "";
	if (!info.enabled) return;
	zoneLayersLegendEl.appendChild(
		el("span", { class: "zl-title", text: "zone layers" }),
	);
	zoneLayersLegendEl.appendChild(
		el("button", {
			class: `zl-chip${info.active.length === 0 ? " on" : ""}`,
			text: "all",
			title: "show every zone layer",
			onclick: () => {
				viewer.clearZoneLayers();
				refreshZoneLayers();
			},
		}),
	);
	for (const layer of info.layers) {
		const hex = `#${layer.colorHex.toString(16).padStart(6, "0")}`;
		zoneLayersLegendEl.appendChild(
			el(
				"button",
				{
					class: `zl-chip${info.active.includes(layer.depth) ? " on" : ""}`,
					title: `toggle zone layer ${layer.depth} (multi-select)`,
					onclick: () => {
						viewer.toggleZoneLayer(layer.depth);
						refreshZoneLayers();
					},
				},
				el("span", { class: "zl-sw", style: `background:${hex}` }),
				`L${layer.depth}`,
			),
		);
	}
}

// Drag the divider between the canvas and the observability dock to set the
// dock width; the canvas reflows via its ResizeObserver. Persisted so the
// chosen width survives reloads.
const OBSDOCK_WIDTH_KEY = "starshot.obsdockWidth";
const OBSDOCK_MIN = 320;
const CANVAS_MIN = 360;

function initObsResizer() {
	const resizer = document.getElementById("obsdock-resizer");
	const dock = document.getElementById("obsdock");
	const body = document.getElementById("overlay-body");
	let saved = NaN;
	try {
		saved = Number(localStorage.getItem(OBSDOCK_WIDTH_KEY));
	} catch {
		/* private mode */
	}
	if (saved >= OBSDOCK_MIN) dock.style.width = `${saved}px`;

	let dragging = false;
	resizer.addEventListener("pointerdown", (ev) => {
		dragging = true;
		resizer.classList.add("dragging");
		resizer.setPointerCapture(ev.pointerId);
		ev.preventDefault();
	});
	resizer.addEventListener("pointermove", (ev) => {
		if (!dragging) return;
		const rect = body.getBoundingClientRect();
		const max = Math.max(OBSDOCK_MIN, rect.width - CANVAS_MIN);
		const width = Math.max(
			OBSDOCK_MIN,
			Math.min(rect.right - ev.clientX, max),
		);
		dock.style.width = `${Math.round(width)}px`;
	});
	const end = (ev) => {
		if (!dragging) return;
		dragging = false;
		resizer.classList.remove("dragging");
		try {
			resizer.releasePointerCapture(ev.pointerId);
		} catch {
			/* already released */
		}
		try {
			localStorage.setItem(
				OBSDOCK_WIDTH_KEY,
				String(parseInt(dock.style.width, 10) || OBSDOCK_MIN),
			);
		} catch {
			/* private mode */
		}
	};
	resizer.addEventListener("pointerup", end);
	resizer.addEventListener("pointercancel", end);
}

function scheduleTreeRender() {
	if (renderQueued) return;
	renderQueued = true;
	requestAnimationFrame(() => {
		renderQueued = false;
		obstree.renderTree(obs.model, { streamed: true });
		if (state.view?.branch && state.lab.simStep)
			obstree.renderPinned(obs.model);
		// New zones may have streamed in — refresh the legend so a new depth's
		// swatch appears (no-ops when the depth set is unchanged).
		refreshZoneLayers();
	});
}

export async function openCell({
	slot,
	model,
	branch = false,
	forceLive = false,
}) {
	const seq = ++openSeq;
	const run = state.run;
	// Keep the camera when this re-renders the SAME cell while the overlay is
	// already open — the branch selector swapping between the source run and its
	// per-LLM simulation lineages, or a revert/step reload. Swapping like that
	// shouldn't pull the user off the part of the scene they're inspecting. A
	// fresh open (overlay closed) or a different cell still frames the scene anew.
	const keepCamera =
		overlayEl.classList.contains("open") &&
		!!state.view &&
		state.view.slot === slot &&
		state.view.model === model;
	stream?.close();
	stream = null;
	state.view = { slot, model, branch };
	// Close any inquiry chat that belongs to a different cell/mode (same-cell
	// re-subscribes from resume/step keep it open).
	inquiry.notifyView({ run, slot, model, branch });
	overlayEl.classList.add("open");
	viewer.setActive(true);
	viewer.clear({ keepCamera });
	obs = createObsModel();
	obstree.resetDock();
	obstree.setPinStep(branch ? state.lab.simStep : null);
	// Per-call revert (the ⏪ on each call row): a source cell rewinds + re-runs
	// from the cut; a branch rewinds ITS OWN log to before the step and pauses
	// there (source untouched), ready to step forward again.
	obstree.setOnRevert(branch ? revertBranchToCall : revertToCall);
	// "+ sim" on a zone row drops it into the prompt lab's simulation slots. A
	// source-cell action — you fork a NEW branch from the source, not from one.
	obstree.setOnAddSim(
		branch ? null : (node) => emit("add-sim-target", { slot, model, node }),
	);
	renderHeader();
	// Clear any stale layer legend from the previous cell while this one loads;
	// it refills once the scene + history fold in below.
	refreshZoneLayers();

	let projection = { nodes: [], last_index: -1 };
	try {
		projection = branch
			? await api.branchScene(branch)
			: await api.scene(run, slot, model, {});
	} catch (e) {
		toast(`scene load failed: ${e.message}`, "err");
	}
	if (seq !== openSeq) return;
	applySceneProjection(viewer, projection);
	viewer.prefetchBundle(
		branch
			? api.branchMeshesUrl(branch)
			: api.meshesUrl(run, slot, model, {}),
	);

	let history = [];
	try {
		history = branch
			? await api.branchEventsHistory(run, branch)
			: await api.eventsHistory(run, slot, model);
	} catch {
		/* never-started cell */
	}
	if (seq !== openSeq) return;
	for (const event of history) obs.feed(event);
	// The scene projection painted bboxes before this history folded, so objects
	// were colored the default green; repaint now that each node's emitting step
	// is known. (Streamed bboxes color correctly on paint — their decompose call
	// always precedes the bbox event.)
	viewer.recolorAll();
	obstree.renderTree(obs.model);
	// Depths are known now that history folded — (re)build the layer legend.
	refreshZoneLayers();
	if (branch && state.lab.simStep) obstree.renderPinned(obs.model);
	renderHeader(); // error message comes from the just-loaded log

	// `forceLive` subscribes without waiting for the next poll — used right
	// after a revert relaunches the cell, so the polled summary is still stale.
	// A cell parked at a step gate (status paused WITH a pending call) still has
	// a live task, so subscribe — the next "step" streams its events.
	const summary = currentSummary();
	const info = branch ? branchSummaryById(branch) : summary;
	const live = forceLive || info?.status === "running" || !!info?.pending;
	if (live) {
		const since = Math.max(obs.model.maxIndex, projection.last_index ?? -1);
		subscribe(seq, since);
	}
}

function subscribe(seq, since) {
	stream = openStream(buildUrl(since), {
		onEvent: (event) => {
			if (seq !== openSeq) return;
			if (!obs.feed(event)) return;
			dispatchSceneEvent(viewer, event);
			scheduleTreeRender();
		},
		onTerminal: () => {
			if (seq !== openSeq) return;
			renderHeader();
			emit("poll-now");
		},
	});
}

function buildUrl(since) {
	const { slot, model, branch } = state.view;
	return branch
		? api.branchEventsUrl(branch, { since })
		: api.eventsUrl(state.run, slot, model, { since });
}

// Re-attach the SSE tail to the CURRENT cell without reloading it — used on
// resume/start in place of a full `openCell`, so the scene + observability dock
// stay exactly as they are and only new events fold in. Rolls back any folded
// terminal sentinel first (resume truncates it server-side and reuses its
// index), then tails from where we left off.
function relinkStream() {
	stream?.close();
	stream = null;
	obs.rewindTerminal();
	subscribe(openSeq, obs.model.maxIndex);
	renderHeader();
}

function currentSummary() {
	if (!state.view) return null;
	return cellSummary(state.view.slot, state.view.model);
}

function renderHeader() {
	if (!state.view) return;
	const { slot, model, branch } = state.view;
	const summary = currentSummary();
	const branchInfo = branch ? branchSummaryById(branch) : null;
	// A per-LLM lineage is pinned to a model; surface it so the cell's many
	// lineages aren't all indistinguishably labeled with the cell's base model.
	const pin =
		branchInfo?.pin && branchInfo.pin !== model ? branchInfo.pin : null;
	titleEl.textContent = `${slot} · ${model}${pin ? " → " + pin : ""}`;
	crumbsEl.textContent = `${state.run}${branch ? (pin ? ` · sim on ${pin}` : " · simulation branch") : ""}`;
	const cellInfo = branch ? branchInfo : summary;
	const view = statusView(cellInfo);
	const status = view.state;
	const pending = cellInfo?.pending;
	dotEl.className = `dot ${view.dot}`;
	// One canonical status message (status.js), plus the overlay-only extras:
	// the stepped-mode tag, the event count, and the error reason.
	let statusText = view.label;
	if (!branch && summary?.stepped) statusText += " · stepped";
	statusText += ` · ${cellInfo?.events_count ?? 0} events`;
	if (status === "error") {
		// Put the failure reason where the eye lands first; the log strip below
		// has the full trail.
		const err = obs.lastError?.();
		if (err) statusText += ` — ${err.text}`;
	}
	statusEl.textContent = statusText;
	statusEl.title = statusText;
	statusEl.classList.toggle("is-error", status === "error");

	// Action button: source cells start/resume/pause; branches pause/resume.
	// "Live" = a running task OR one parked at a step gate (pending) — both are
	// pausable; only a cell with no live task resumes/starts.
	const live = status === "running" || !!pending;
	let label = null;
	if (branch) {
		if (live) label = "pause sim";
		else if (status === "paused" || status === "error")
			label = "resume sim";
	} else {
		if (live) label = "pause";
		else if (status === "idle") label = "start";
		else if (status === "paused") label = "resume";
		else if (status === "error") label = "retry";
	}
	actionBtn.style.display = label ? "" : "none";
	actionBtn.textContent = label ?? "";
	resetBtn.style.display = branch ? "none" : "";

	// One-call-at-a-time stepping controls. Shown whenever the cell is gated:
	// a live branch always is; a source cell whenever it's in step mode (so
	// the button is there even when paused with no live gate — incl. after a
	// restart). `done` cells have nothing left to step.
	const stepped = branch || !!summary?.stepped;
	const canStep = stepped && status !== "done";
	let stepBtn = document.getElementById("overlay-step");
	let autoBtn = document.getElementById("overlay-auto");
	if (canStep) {
		if (!stepBtn) {
			stepBtn = el("button", { id: "overlay-step", class: "primary" });
			actionBtn.before(stepBtn);
		}
		if (!autoBtn) {
			autoBtn = el("button", { id: "overlay-auto" });
			actionBtn.before(autoBtn);
		}
		stepBtn.textContent = pending ? `step: ${pending.step}` : "step";
		stepBtn.title = pending
			? `run the pending ${pending.step} call on ${pending.node ?? "?"}`
			: "run the next LLM call, then pause again";
		autoBtn.textContent = "run rest";
		autoBtn.title = "finish without further pauses";
		stepBtn.onclick = () => stepCurrent(false);
		autoBtn.onclick = () => stepCurrent(true);
		// Source cells (not branches) can fast-forward to a target step.
		let untilSel = document.getElementById("overlay-step-until");
		if (!branch && state.steps.length) {
			if (!untilSel) {
				untilSel = stepUntilSelect(
					state.steps,
					(until) => stepCurrent(false, until),
					{ label: "until…" },
				);
				untilSel.id = "overlay-step-until";
				autoBtn.before(untilSel);
			}
		} else {
			untilSel?.remove();
		}
	} else {
		stepBtn?.remove();
		autoBtn?.remove();
		document.getElementById("overlay-step-until")?.remove();
	}

	// Branch access: a selector to jump between the source run and EACH of the
	// cell's downstream simulations (lineages). Available even on a DONE cell —
	// whose only source action is reset — so sims that branched off earlier (and
	// may still be running / paused) are always reachable here. Picking one loads
	// it; from a branch its own step/pause/resume controls take over.
	document.getElementById("overlay-flip")?.remove(); // legacy single-branch flip
	let branchSel = document.getElementById("overlay-branch-sel");
	const cellBs = cellBranches(slot, model);
	if (branch || cellBs.length) {
		const sig = `${branch ?? ""}|${cellBs.map((b) => `${b.id}:${statusView(b).state}`).join(",")}`;
		// Don't rebuild the <select> while the user has it open (a poll would close
		// the dropdown); refresh only when the option set / statuses actually change.
		if (
			!branchSel ||
			(branchSel.dataset.sig !== sig &&
				document.activeElement !== branchSel)
		) {
			if (!branchSel) {
				branchSel = el("select", {
					id: "overlay-branch-sel",
					title: "view the source run or one of its downstream simulations",
				});
				// Read the CURRENT view at change time — this <select> element persists
				// across renders/cells (only its options are rebuilt), so capturing the
				// creating render's slot/model would go stale: switching while viewing a
				// DIFFERENT cell's branch would re-open under the wrong cell, whose
				// options exclude the viewed branch → the dropdown blanks out.
				branchSel.addEventListener("change", () => {
					const v = state.view;
					if (v)
						openCell({
							slot: v.slot,
							model: v.model,
							branch: branchSel.value || null,
						});
				});
				actionBtn.before(branchSel);
			}
			branchSel.dataset.sig = sig;
			branchSel.replaceChildren(
				el("option", { value: "", text: "view: source run" }),
				...cellBs.map((b) => {
					const lab = b.pin ?? b.model ?? model;
					const node = b.last_step?.node;
					const st = statusView(b).state;
					return el("option", {
						value: b.id,
						text: `⑂ sim · ${lab}${node ? " @ " + node : ""} · ${st}`,
					});
				}),
			);
			branchSel.value = branch ?? "";
		}
	} else if (branchSel) {
		branchSel.remove();
	}
}

// Revert the open source cell to just before `call` — confirms first (it
// drops every later step + its meshes), then re-runs the pipeline from the cut
// and reloads the overlay streaming the re-run live.
function revertToCall(call) {
	const { slot, model, branch } = state.view ?? {};
	if (!slot || branch) return;
	const step = call.template ?? call.step ?? "this step";
	openModal(`revert ${slot} · ${model}?`, (close, setError) => ({
		body: [
			el("div", {
				class: "m-hint",
				text:
					`Truncates this slot's log to just before its ${step} call (#${call.index}), ` +
					"drops every later step and its meshes, then re-runs the pipeline from there.",
			}),
		],
		actions: [
			el("button", { text: "cancel", onclick: close }),
			el("button", {
				class: "danger",
				text: "revert & re-run",
				onclick: async () => {
					try {
						await api.rewind(state.run, slot, model, call.index);
					} catch (e) {
						setError(e.message);
						return;
					}
					close();
					toast(
						`reverted ${slot} · ${model} to #${call.index}`,
						"ok",
					);
					// The cell is already relaunching server-side; reload at the cut and
					// stream the re-run (forceLive — the polled summary is still stale).
					openCell({ slot, model, branch: false, forceLive: true });
					emit("poll-now");
				},
			}),
		],
	}));
}

// Revert the open simulation BRANCH to just before `call` — confirms first (it
// drops every later step + its meshes on this branch), then truncates the
// branch's log and PAUSES there. The source run is untouched; stepping forward
// re-runs from the cut under the branch's edits. The branch mirror of
// `revertToCall`, but non-destructive to the source (and it pauses rather than
// auto-re-running, since a branch advances one manual step at a time).
function revertBranchToCall(call) {
	const { slot, model, branch } = state.view ?? {};
	if (!slot || !branch) return;
	const step = call.template ?? call.step ?? "this step";
	openModal(`revert this simulation?`, (close, setError) => ({
		body: [
			el("div", {
				class: "m-hint",
				text:
					`Truncates this simulation branch to just before its ${step} call (#${call.index}), ` +
					"drops every later step and its meshes, and pauses there — “step” then re-runs from the cut. " +
					"The source run is untouched.",
			}),
		],
		actions: [
			el("button", { text: "cancel", onclick: close }),
			el("button", {
				class: "danger",
				text: "revert & pause",
				onclick: async () => {
					// overrides=null keeps the branch's own edit set (the overlay isn't the
					// prompt lab, so it doesn't re-apply the lab's live drafts).
					try {
						await api.branchRewind(branch, call.index);
					} catch (e) {
						setError(e.message);
						return;
					}
					close();
					toast(`reverted simulation to #${call.index}`, "ok");
					// Reload the (now-paused, truncated) branch at the cut; emit a poll so
					// the header status catches up from the still-stale summary.
					openCell({ slot, model, branch });
					emit("poll-now");
				},
			}),
		],
	}));
}

async function stepCurrent(auto, until = null) {
	const { slot, model, branch } = state.view ?? {};
	if (!slot) return;
	try {
		if (branch) {
			await api.branchStep(branch, { auto });
		} else {
			const r = await api.cellStep(state.run, slot, model, {
				auto,
				until,
			});
			// A paused cell that was just relaunched (vs. a live gate released over
			// the existing stream) needs a fresh subscription to watch the call land.
			if (r.result === "launched") {
				setTimeout(() => {
					if (
						state.view &&
						state.view.slot === slot &&
						state.view.model === model &&
						!state.view.branch
					) {
						openCell({ slot, model, branch: false });
					}
				}, 250);
			}
		}
		emit("poll-now");
	} catch (e) {
		toast(e.message, "err");
	}
}

async function onAction() {
	const { slot, model, branch } = state.view ?? {};
	if (!slot) return;
	const summary = currentSummary();
	const info = branch ? branchSummaryById(branch) : summary;
	const status = info?.status;
	// Parked-at-gate (pending) counts as live, so the button pauses (breaks out
	// of stepping) rather than mis-firing a resume on a cell that's still gated.
	const pausing = status === "running" || !!info?.pending;
	try {
		if (branch) {
			if (pausing) await api.branchPause(branch);
			else await api.branchResume(branch);
		} else {
			if (pausing) await api.pause(state.run, slot, model);
			else await api.resume(state.run, slot, model);
		}
		emit("poll-now");
		// Bail if the view moved while the request was in flight.
		if (
			!state.view ||
			state.view.slot !== slot ||
			state.view.model !== model ||
			state.view.branch !== branch
		)
			return;
		// Re-wire the live stream IN PLACE rather than reloading the cell — no
		// viewer.clear / dock reset, so the scene + observability stay put.
		// Pausing lets the open stream wind down on its run.paused; resuming/
		// starting re-tails so new events fold into the existing view.
		if (pausing) renderHeader();
		else relinkStream();
	} catch (e) {
		toast(e.message, "err");
	}
}

async function onReset() {
	const { slot, model } = state.view ?? {};
	if (!slot) return;
	if (!confirm(`Wipe ${slot} · ${model} on "${state.run}" and start fresh?`))
		return;
	try {
		await api.reset(state.run, slot, model, true);
		emit("poll-now");
		openCell({ slot, model, branch: false });
	} catch (e) {
		toast(e.message, "err");
	}
}

export function closeOverlay() {
	openSeq += 1;
	stream?.close();
	stream = null;
	state.view = null;
	overlayEl.classList.remove("open");
	viewer.setActive(false);
	obstree.setPinStep(null);
	inquiry.closeInquiry();
}
