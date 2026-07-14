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

async function request(path, { method = "GET", body, params, cache = "no-store" } = {}) {
	const res = await fetch(u(path, params), {
		method,
		// Most calls bypass the HTTP cache (no-store). Immutable-ish GETs pass
		// cache: "no-cache" so the browser STORES the response + its ETag and
		// revalidates with If-None-Match — the server 304s unchanged content, so
		// the (0.8-5 MB) body isn't re-transferred while staleness is impossible.
		cache,
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

// A run id can now be a nested PATH (ablation variants fold under their base as
// `<base>/ablations/<experiment>/<variant>`). When a run goes in the URL PATH
// (not a query param), encode each segment but keep the `/` separators so the
// server's `:path` routes match. Query-param uses (`?run=`) need no special care.
const encPath = (s) => String(s).split("/").map(encodeURIComponent).join("/");

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
	activateRun: (name) =>
		request(`/runs/${encodeURIComponent(name)}/activate`, {
			method: "POST",
		}),
	// Delete a run and its on-disk artifacts (stops any live cells first).
	// Refuses the active run. Used by the ablation board's reset.
	deleteRun: (name) =>
		request(`/runs/${encPath(name)}`, { method: "DELETE" }),
	// Load a run's cells into memory WITHOUT activating it, so its /scene +
	// /meshes are readable next to the active run (the run-compare view). No-op
	// if already loaded; launches nothing.
	hydrateRun: (name) =>
		request(`/runs/${encodeURIComponent(name)}/hydrate`, {
			method: "POST",
		}),
	slots: (run) => request("/slots", { params: { run } }),
	// Every ablation variant of a base run in ONE read (nested manifest + legacy
	// flat, merged server-side). Replaces filtering /runs by ablation.base_run +
	// the __abl- name prefix. Each variant carries { run_id, experiment, slot,
	// model, target_step_kind, cut, replicate, treatment, legacy_flat? }.
	ablations: (baseRun) => request(`/runs/${encodeURIComponent(baseRun)}/ablations`),
	// Coarse cell lifecycle status (done/error/paused/running/idle) via a cheap
	// server-side tail scan — NO full events.jsonl pull, NO hydration. Backs the
	// ablation board's per-variant status poll.
	runStatus: (run, slot, model) =>
		request(`/run-status/${encPath(run)}/${encodeURIComponent(slot)}/${encodeURIComponent(model)}`),
	// Live snapshot of the process-global mesh queue — every in-flight + waiting
	// generation across the Modal Trellis/Hunyuan pool and the Tencent Hunyuan 3.1
	// pool. { pools: [{id,label,cap}], entries: [{slot_id,job_id,state,backend,pool,…}] }.
	trellisQueue: () => request("/trellis/queue"),
	// Hard-stop every in-flight generation process-wide (all runs/cells/meshes).
	stopAllGenerations: () => request("/generations/stop", { method: "POST" }),
	// Process-wide snapshot of what's running now (for the task monitor).
	activeGenerations: () => request("/generations/active"),

	// --- cell lifecycle ---
	// `stepped` opts the cell in/out of one-call-at-a-time execution;
	// `logprobs` opts it in/out of top-20 token-logprob capture. Either
	// null/undefined keeps the cell's current mode for that flag.
	resume: (run, slot, model, stepped = null, logprobs = null) =>
		request(cellPath(slot, model, "/resume"), {
			method: "POST",
			params: { run, stepped, logprobs },
		}),
	// `auto` runs the cell to completion; `until` fast-forwards to the next run
	// of that step and pauses there. A plain step (neither) advances one call —
	// queued if the cell is mid-call, so a "step all" never skips a slow model.
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
	// Applies to the whole prefab group.
	reset: (run, slot, model, nodeId) =>
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
	// The captured top-token logprobs for one LLM call — present only when the
	// cell was started with capture on AND the provider served them. Keyed by
	// the call's content hash (the `cache.llm` event's `key`); resolves to the
	// parsed sidecar { text, tokens:[{start,end,token,logprob,top:[…]}], … } or
	// null when absent. The token stream concatenates back to `text`, which is
	// the exact response the model emitted — so tokens map 1:1 onto the output.
	async logprobs(run, slot, model, key) {
		if (!key) return null;
		const res = await fetch(
			u(`/artifacts/${run}/${slot}/${model}/logprobs/${key}.json`),
			{ cache: "no-store" },
		);
		if (!res.ok) return null;
		try {
			return await res.json();
		} catch {
			return null;
		}
	},

	// --- teacher-forcing export (Gemma-faithful reconstruction) ---
	// `tfSteps` is the cell's linear step timeline (every LLM call in order, each
	// with `render_until` = the log index to render the 3D scene up to).
	// `tfExport` reconstructs one step's full input+reasoning+output sequence
	// with id→char-span maps (scene zones/objects, to-place, output, variables).
	tfSteps: (run, slot, model) =>
		request(cellPath(slot, model, "/tf-steps"), { params: { run } }),
	// The compact scene-tree + emitting provenance for a cell (server projection):
	// `{ order:[id…], nodes:{ id: { kind, parent_id, region?, emitted_by?, call_index? } } }`.
	// Replaces downloading + folding the full events.jsonl client-side just to
	// recover the id→kind/parent/emitting-region lookup the /tf drawer + tree need.
	tfTree: (run, slot, model) =>
		request(cellPath(slot, model, "/tf-tree"), { params: { run } }),
	tfExport: (run, slot, model, eventIndex) =>
		request(cellPath(slot, model, `/tf-export/${eventIndex}`), {
			params: { run },
			cache: "no-cache", // ETag-revalidated: 304s the reconstruction on repeat step nav
		}),

	// --- attention analysis (teacher-forced, real Modal GPU compute) ---
	// The browser NEVER blocks on a compute. `attentionEnqueue` schedules one or
	// many steps on the server and returns immediately; `attentionList` is the
	// status poll — the server-owned queue: `{ computed, running, queued, errors }`
	// per step; `attentionGet` fetches one stored result (404 until computed).
	// max_query_tokens: 0 (default) scores EVERY completion token — the full output
	// trajectory. A positive value bounds only the long reasoning region (every
	// OUTPUT token is always kept), trading reasoning detail for smaller files.
	attentionEnqueue: (run, slot, model, { eventIndices, force = false, maxQueryTokens = 0, maxHeads = 32, topK = 12 } = {}) =>
		request(cellPath(slot, model, "/attention"), {
			method: "POST",
			params: { run },
			body: { event_indices: eventIndices, force, max_heads: maxHeads, top_k: topK, max_query_tokens: maxQueryTokens },
		}),
	// Reset a cell's PENDING compute (clears the model's stale queue partition +
	// this cell's queued/running) so a fresh compute-all never inherits a prior
	// run's phantom jobs. Committed results are preserved. Reuses the enqueue
	// route with an empty item list + `reset` flag (fires even with nothing to send).
	attentionReset: (run, slot, model) =>
		request(cellPath(slot, model, "/attention"), {
			method: "POST",
			params: { run },
			body: { event_indices: [], reset: true },
		}),
	attentionList: (run, slot, model, { maxHeads = 0 } = {}) =>
		request(cellPath(slot, model, "/attention"), { params: { run, max_heads: maxHeads || undefined } }),
	// CHEAP, version-aware "what attention is already computed for this cell?" — a
	// pure disk scan (no run hydration, no Modal poll) returning { fresh, stale }
	// event indices. The ablation drawer lists this FIRST, then pulls only what's
	// missing/stale, instead of 404-probing each guessed index.
	attentionIndex: (run, slot, model, { maxHeads = 0 } = {}) =>
		request(`/attention-index/${encPath(run)}/${encodeURIComponent(slot)}/${encodeURIComponent(model)}`, { params: { max_heads: maxHeads || undefined } }),
	// Per-object spatial relevance to a step's target node: distance + ray-trace
	// visibility, from scene geometry (no GPU). Powers the ablation "does it attend
	// to what's close / visible?" graph. `ev` = the step's cache.llm event index.
	spatialRanks: (run, slot, model, ev) =>
		request(cellPath(slot, model, "/spatial-ranks"), { params: { run, ev } }),
	// Projected views of one step's stored analysis so the browser never pulls the
	// (potentially hundreds-of-MB) full result. `compact` (default) = scalars +
	// head grid + entity maps + precomputed aggregates; `token` = one token's full
	// per-head detail (lazy scrub); `present` = per-token head-summed detail;
	// `full` = the whole thing (debug/back-compat).
	attentionGet: (run, slot, model, eventIndex, { view = "compact", i } = {}) =>
		request(cellPath(slot, model, `/attention/${eventIndex}`), { params: { run, view, i }, cache: "no-cache" }),
	// Batch a cell's stored per-step attention projections into ONE request (kills
	// the per-step `attentionGet` N+1) plus a server-computed cell-wide rollup.
	// `evs` = the explicit event indices to pull (array → comma list); omit to
	// batch every computed step. `view`: agg (token-free rollups the data cards
	// pool) | buckets (just the VII grids) | abl (ultra-light) | compact. Sent
	// no-cache so an unchanged cell can 304 (ETag) instead of re-shipping the batch.
	cellAggregate: (run, slot, model, { view = "agg", evs } = {}) =>
		request(cellPath(slot, model, "/attention/cell-aggregate"), {
			params: { run, view, evs: Array.isArray(evs) ? evs.join(",") : evs },
			cache: "no-cache",
		}),
	// Discover a base cell's ablation variants server-side + batch each variant's
	// treated-step `abl` projection in ONE request (collapses the client's /runs
	// filter + per-variant index probe + per-variant fetch — the ablation N+1).
	// `run` = the BASE run; `kind` optionally filters by target step kind. Returns
	// { base_run, slot, model, variants:[{name, ablation, ev, a}], combos }.
	ablationCompare: (run, slot, model, { kind } = {}) =>
		request(cellPath(slot, model, "/ablation-compare"), {
			params: { run, kind },
			cache: "no-cache",
		}),

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
	// `auto` runs the branch to completion; `until` (a template id) fast-forwards
	// it to the next call of that step. A plain step queues if the branch is
	// mid-call (so a batch "step sims" never errors on a not-yet-gated branch).
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
	restoreRunPrompts: (run) =>
		request(`/runs/${encodeURIComponent(run)}/prompt-templates/restore`, {
			method: "POST",
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
	// Continue a step's own LLM conversation. `body` carries the step's `model`,
	// its system prompt, and the running `messages` thread (seeded with the
	// call's input + output); the reply comes from that same model. Returns
	// {answer, reasoning, model}.
	inquire: (body) => request("/inquire", { method: "POST", body }),
};
