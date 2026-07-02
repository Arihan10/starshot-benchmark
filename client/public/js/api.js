// Typed wrappers for every server endpoint the client uses. All errors
// surface as Error(detail) so callers can toast them uniformly.

export const SERVER_URL = document
	.querySelector('meta[name="server-url"]')
	.getAttribute("content");

function u(path, params = {}) {
	const url = new URL(path, SERVER_URL);
	for (const [k, v] of Object.entries(params)) {
		if (v !== undefined && v !== null) url.searchParams.set(k, v);
	}
	return url;
}

async function request(path, { method = "GET", body, params } = {}) {
	const res = await fetch(u(path, params), {
		method,
		cache: "no-store",
		headers:
			body !== undefined
				? { "Content-Type": "application/json" }
				: undefined,
		body: body !== undefined ? JSON.stringify(body) : undefined,
	});
	if (!res.ok) {
		let detail = `HTTP ${res.status}`;
		try {
			const data = await res.json();
			if (data && data.detail) detail = String(data.detail);
		} catch {
			/* non-JSON error body */
		}
		throw new Error(detail);
	}
	return res.json();
}

const cellPath = (slot, model, tail = "") =>
	`/slots/${encodeURIComponent(slot)}/${encodeURIComponent(model)}${tail}`;

// Read a JSONL event log straight from the artifact server (full history,
// cache.llm payloads included). Tolerates a torn final line written mid-flush.
async function readEventsJsonl(rel) {
	const res = await fetch(u(`/artifacts/${rel}`), { cache: "no-store" });
	if (!res.ok) return [];
	const text = await res.text();
	const out = [];
	for (const line of text.split("\n")) {
		if (!line.trim()) continue;
		try {
			out.push(JSON.parse(line));
		} catch {
			/* torn tail line mid-write */
		}
	}
	return out;
}

export const api = {
	// --- runs + versions ---
	runs: () => request("/runs"),
	versions: () => request("/versions"),
	createRun: (name, promptVersion) =>
		request("/runs", {
			method: "POST",
			body: { name, prompt_version: promptVersion },
		}),
	// Launch a NEW run (B) seeded with `sourceRun`'s ROOT zone plans: each
	// listed {slot, model} cell is copied through its root zone plan and started
	// on `promptVersion`, so the pipeline replays that plan and re-derives the
	// rest under the new prompts. Returns {current, seeded:[...], skipped:[...]}.
	abTest: (name, promptVersion, sourceRun, cells) =>
		request("/runs/ab-test", {
			method: "POST",
			body: {
				name,
				prompt_version: promptVersion,
				source_run: sourceRun,
				cells,
			},
		}),
	activateRun: (name) =>
		request(`/runs/${encodeURIComponent(name)}/activate`, {
			method: "POST",
		}),
	// Load a run's cells into memory WITHOUT activating it, so its /scene +
	// /meshes are readable next to the active run (the run-compare view). No-op
	// if already loaded; launches nothing.
	hydrateRun: (name) =>
		request(`/runs/${encodeURIComponent(name)}/hydrate`, {
			method: "POST",
		}),
	slots: (run) => request("/slots", { params: { run } }),
	// Live snapshot of the process-global mesh queue — every in-flight + waiting
	// generation across the Modal Trellis/Hunyuan pool and the Tencent Hunyuan 3.1
	// pool. { pools: [{id,label,cap}], entries: [{slot_id,job_id,state,backend,pool,…}] }.
	trellisQueue: () => request("/trellis/queue"),

	// --- cell lifecycle ---
	// `stepped` opts the cell in/out of one-call-at-a-time execution
	// (null/undefined keeps its current mode).
	resume: (run, slot, model, stepped = null) =>
		request(cellPath(slot, model, "/resume"), {
			method: "POST",
			params: { run, stepped },
		}),
	// `auto` runs the cell to completion; `until` runs THROUGH the next call of
	// that step (it executes), pausing before the following one. A plain step
	// (neither) advances one call — queued if the cell is mid-call, so a "step
	// all" never skips a slow model.
	cellStep: (run, slot, model, { auto = false, until = null } = {}) =>
		request(cellPath(slot, model, "/step"), {
			method: "POST",
			params: { run },
			body: { auto, until },
		}),
	stepAll: (run, { auto = false, until = null } = {}) =>
		request(`/runs/${encodeURIComponent(run)}/step-all`, {
			method: "POST",
			params: { auto, until },
		}),
	pause: (run, slot, model) =>
		request(cellPath(slot, model, "/pause"), {
			method: "POST",
			params: { run },
		}),
	reset: (run, slot, model, start = false) =>
		request(cellPath(slot, model, "/reset"), {
			method: "POST",
			params: { run, start },
		}),
	// Revert a slot to before a given event: truncates its log there, drops the
	// dropped nodes' meshes, and leaves it paused at that point.
	rewind: (run, slot, model, toEventIndex) =>
		request(cellPath(slot, model, "/rewind"), {
			method: "POST",
			params: { run },
			body: { to_event_index: toEventIndex },
		}),
	retryMesh: (run, slot, model, nodeId) =>
		request(
			cellPath(slot, model, `/retry-mesh/${encodeURIComponent(nodeId)}`),
			{ method: "POST", params: { run } },
		),
	// Permanently wipe ONE object from the cell: every reference in both event
	// logs (library + generated), its mesh/image files in every build dir, all
	// reindexed. Orphaned children re-anchor to the object's region; a wiped
	// prefab canonical hands its role (and raw mesh) to a reuse. Irreversible;
	// tears down branches + in-flight tasks first and does not auto-resume.
	deleteObject: (run, slot, model, nodeId) =>
		request(
			cellPath(slot, model, `/delete-object/${encodeURIComponent(nodeId)}`),
			{ method: "POST", params: { run } },
		),

	// --- from-scratch generated assets (Nano-Banana + a mesh backend) ---
	// The single generated build of a cell, built/regenerated alongside the
	// library build and viewed by flipping `meshesUrl(..., { mode: "generated" })`.
	// generate(): build/resume the whole scene's generated assets (409 if a build
	// is already in flight). generateStatus(): whether a build/regen is running +
	// the finished ids (each with its symmetry plane, served `url`, and the raw
	// generation-API `raw` mesh url for the per-object view; plus each mesh's
	// prefab `canonical`). regenerate/symmetrize/unsymmetrize act on ONE object's
	// generated mesh; with propagate they apply to its whole prefab group.
	// `unlink` splits the object out of its group into a standalone asset (its own
	// raw, no rebuild); `link` moves it into another object's group. backend ∈
	// {trellis, hunyuan, hunyuan-tencent}.
	generate: (run, slot, model) =>
		request(cellPath(slot, model, "/generate"), {
			method: "POST",
			params: { run },
		}),
	generateStatus: (run, slot, model, { optimized = true } = {}) =>
		request(cellPath(slot, model, "/generate"), {
			params: { run, optimized: optimized ? 1 : 0 },
		}),
	regenerate: (
		run,
		slot,
		model,
		nodeId,
		{
			backend = "trellis",
			propagate = true,
			reuseImage = false,
			regenNounPhrase = false,
		} = {},
	) =>
		request(
			cellPath(slot, model, `/regenerate/${encodeURIComponent(nodeId)}`),
			{
				method: "POST",
				params: {
					run,
					backend,
					propagate,
					reuse_image: reuseImage,
					regen_noun_phrase: regenNounPhrase,
				},
			},
		),
	// Link a generated object INTO another object's prefab group (target = any
	// member of that group) so it shares the group's mesh — the inverse of
	// `unlink`. Re-derives the object's mesh from the group canonical; no backend
	// call.
	link: (run, slot, model, nodeId, target, { group = false } = {}) =>
		request(cellPath(slot, model, `/link/${encodeURIComponent(nodeId)}`), {
			method: "POST",
			params: { run, target, group },
		}),
	// Split a generated object OUT of its prefab group into a standalone asset with
	// its own copy of the shared raw mesh (the inverse of `link`), so it stops
	// sharing and can then be regenerated alone. No backend call — a fast local
	// re-derivation on the regen worker.
	unlink: (run, slot, model, nodeId) =>
		request(cellPath(slot, model, `/unlink/${encodeURIComponent(nodeId)}`), {
			method: "POST",
			params: { run },
		}),
	symmetrize: (
		run,
		slot,
		model,
		nodeId,
		{ plane = "xy", keepPositive = true, propagate = true } = {},
	) =>
		request(
			cellPath(slot, model, `/symmetrize/${encodeURIComponent(nodeId)}`),
			{
				method: "POST",
				params: { run, plane, keep_positive: keepPositive, propagate },
			},
		),
	unsymmetrize: (run, slot, model, nodeId, { propagate = true } = {}) =>
		request(
			cellPath(
				slot,
				model,
				`/unsymmetrize/${encodeURIComponent(nodeId)}`,
			),
			{
				method: "POST",
				params: { run, propagate },
			},
		),
	// Change an object's "front view" (which face points +Z in its raw mesh) by
	// rotating the raw 90° about `axis` ('x' pitch, 'y' yaw, 'z' roll). Bakes into
	// the prefab canonical's raw + propagates to every reuse.
	reorient: (
		run,
		slot,
		model,
		nodeId,
		{ axis = "y", degrees = 90, propagate = true } = {},
	) =>
		request(
			cellPath(slot, model, `/reorient/${encodeURIComponent(nodeId)}`),
			{
				method: "POST",
				params: { run, axis, degrees, propagate },
			},
		),
	// Force the window/glass transparency transform (white texels → near-clear)
	// onto an object's served mesh, bypassing the pipeline's keyword + symmetry
	// gates. Applies to the whole prefab group; dropped by a later regenerate/reset.
	glassify: (run, slot, model, nodeId) =>
		request(
			cellPath(slot, model, `/glassify/${encodeURIComponent(nodeId)}`),
			{ method: "POST", params: { run } },
		),
	// Rebuild an object's served mesh from its pristine raw, dropping any in-place
	// served edit (e.g. a forced glassify) while keeping its current symmetry.
	// Applies to the whole prefab group. (Named `resetMesh` so it doesn't collide
	// with the cell-level `reset` above — they're distinct operations.)
	resetMesh: (run, slot, model, nodeId) =>
		request(
			cellPath(slot, model, `/reset-mesh/${encodeURIComponent(nodeId)}`),
			{ method: "POST", params: { run } },
		),

	// --- source cell data ---
	//   untilIndex → project/bundle only the prefix BEFORE that event (the
	//     compare view's "previous" pane: the original's state at the fork step).
	scene: (run, slot, model, { untilIndex } = {}) =>
		request(cellPath(slot, model, "/scene"), {
			params: { run, until_index: untilIndex },
		}),
	eventsUrl: (run, slot, model, { since } = {}) =>
		u(cellPath(slot, model, "/events"), { run, since }).toString(),
	//   mode="generated" streams the cell's from-scratch generated build instead
	//     of the library objects/; optimized=false streams the raw bbox-fitted
	//     Trellis mesh instead of the served KTX2/Meshopt twin.
	meshesUrl: (run, slot, model, { untilIndex, mode, optimized } = {}) =>
		u(cellPath(slot, model, "/meshes"), {
			run,
			until_index: untilIndex,
			mode,
			optimized: optimized === false ? 0 : undefined,
		}).toString(),
	// Full source-cell history backfill (cache.llm payloads included) from disk.
	async eventsHistory(run, slot, model) {
		return readEventsJsonl(`${run}/${slot}/${model}/events.jsonl`);
	},
	artifactUrl: (path) => u(`/artifacts/${path}`).toString(),
	absUrl: (path) => new URL(path, SERVER_URL).toString(),

	// --- simulation branches (keyed by branch id) ---
	// A branch is a first-class fork living in the run's flat `_branches/<id>/`
	// temp folder. Many per cell coexist; parallel-LLM previews are child
	// branches. All control is keyed by the opaque branch id.
	branchScene: (bid) => request(`/branches/${encodeURIComponent(bid)}/scene`),
	branchStatus: (bid) => request(`/branches/${encodeURIComponent(bid)}`),
	branchEventsUrl: (bid, { since } = {}) =>
		u(`/branches/${encodeURIComponent(bid)}/events`, { since }).toString(),
	//   sinceIndex → only the meshes placed at/after that event (a fan-out child's
	//     NEW meshes over the shared prefix already on screen).
	branchMeshesUrl: (bid, { sinceIndex } = {}) =>
		u(`/branches/${encodeURIComponent(bid)}/meshes`, {
			since_index: sinceIndex,
		}).toString(),
	async branchEventsHistory(run, bid) {
		return readEventsJsonl(`${run}/_branches/${bid}/events.jsonl`);
	},

	// --- prompt lab ---
	promptTemplates: (run) =>
		request(`/runs/${encodeURIComponent(run)}/prompt-templates`),
	// Every `{VARIABLE}` rendered against a fixed sample scene by the real
	// server-side injection functions — the lab's hover preview. Run-independent.
	variableSamples: () => request("/variable-samples"),
	stepEvents: (run, step) =>
		request(`/runs/${encodeURIComponent(run)}/step-events`, {
			params: { step },
		}),
	stepEvent: (run, slot, model, index, step) =>
		request(`/runs/${encodeURIComponent(run)}/step-event`, {
			params: { run, slot, model, index, step },
		}),
	branchStepEvent: (bid, step, node) =>
		request(`/branches/${encodeURIComponent(bid)}/step-event`, {
			params: { step, node },
		}),
	promptTest: (body) => request("/prompt-test", { method: "POST", body }),
	// Fork a NEW simulation branch from a cell. `body` is {event_index, step,
	// overrides, seed?, model?}; returns {branch:{id, …}}. Many branches per cell
	// coexist (fork different zones — or LLMs — for parallel sims). `body.model`
	// (an alias) PINS the fork to that LLM and runs one step (compare's per-LLM
	// lineage); omit it to pause at the forked step on the cell's base model.
	createBranch: (run, slot, model, body) =>
		request(cellPath(slot, model, "/branch"), {
			method: "POST",
			params: { run },
			body,
		}),
	// Every TOP-LEVEL sim branch of a run (children/fan-out previews excluded).
	listBranches: (run) =>
		request(`/runs/${encodeURIComponent(run)}/branches`, {
			params: { run },
		}),
	// `auto` runs the branch to completion; `until` (a template id) runs it
	// THROUGH the next call of that step (it executes), pausing before the
	// following one. A plain step queues if the branch is mid-call (so a batch
	// "step sims" never errors on a not-yet-gated branch).
	// `modelOverride` (a model alias) re-aims the next gated call at a chosen LLM.
	branchStep: (bid, { auto = false, until = null, model = null } = {}) =>
		request(`/branches/${encodeURIComponent(bid)}/step`, {
			method: "POST",
			body: { auto, until, model },
		}),
	// Revert a branch to before `toEventIndex` and pause there; a following
	// branchStep re-runs from the cut under the current snapshot + `overrides`
	// (the lab's live edit set). `overrides=null` keeps the branch's edits.
	branchRewind: (bid, toEventIndex, overrides = null) =>
		request(`/branches/${encodeURIComponent(bid)}/rewind`, {
			method: "POST",
			body: { to_event_index: toEventIndex, overrides },
		}),
	// Promote a branch to BE its origin source cell (replace + discard).
	branchCommit: (bid) =>
		request(`/branches/${encodeURIComponent(bid)}/commit`, {
			method: "POST",
		}),
	branchPause: (bid) =>
		request(`/branches/${encodeURIComponent(bid)}/pause`, {
			method: "POST",
		}),
	branchResume: (bid) =>
		request(`/branches/${encodeURIComponent(bid)}/resume`, {
			method: "POST",
		}),
	branchDiscard: (bid) =>
		request(`/branches/${encodeURIComponent(bid)}`, { method: "DELETE" }),
	saveVersion: (body) => request("/versions/save", { method: "POST", body }),
	saveRunFromBranches: (body) =>
		request("/runs/from-branches", { method: "POST", body }),
	forkVersion: (name, base) =>
		request("/versions/fork", { method: "POST", body: { name, base } }),
	updateRunPrompts: (run, overrides, updateVersion) =>
		request(`/runs/${encodeURIComponent(run)}/prompt-templates`, {
			method: "PUT",
			body: { overrides, update_version: updateVersion },
		}),
	// Reset the run's snapshot back to its base source version (the inverse of
	// updateRunPrompts' version-sync) — discards the lab's in-place prompt edits.
	//   step → revert ONLY that step's prompt (the per-prompt revert); omit to
	//     restore every step from the base version.
	restoreRunPrompts: (run, step = null) =>
		request(`/runs/${encodeURIComponent(run)}/prompt-templates/restore`, {
			method: "POST",
			params: { step },
		}),
	// Fork a simulation branch in each cell at its earliest call of any of
	// `steps` (non-destructive — source untouched). Replaces the old destructive
	// rerun-step.
	simulateStep: (run, steps, cells = null) =>
		request(`/runs/${encodeURIComponent(run)}/simulate-step`, {
			method: "POST",
			body: { steps, cells },
		}),

	// --- decision inquiry ---
	// Ask the reviewer (Claude Opus 4.8, xhigh) why a step's subject model
	// decided what it did. `body` carries the client-assembled reviewer `system`
	// (analyst framing + the step's system/input/output/reasoning grounding) plus
	// the running `messages` thread; returns {answer, reasoning, model}.
	inquire: (body) => request("/inquire", { method: "POST", body }),
};
