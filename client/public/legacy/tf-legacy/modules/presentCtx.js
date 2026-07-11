// Present-mode glue between /tf and the self-contained present.js overlay. Builds
// the per-step present context (analysis + text + per-object assets + zone
// badges), launches single-step present, and stitches the longest contiguous
// computed run into one continuous end-to-end playback.

import { $, state, PLACEMENT_STEPS, SEQ_SPEED, isFrameEntity, entityHex, entityKindLabel, compHex, COLORS } from "./state.js";
import { api } from "../../js/api.js";
import { applySceneProjection } from "../../js/events.js";
import { openPresent, closePresent, advancePresent } from "./present.js";
import { renderAttention } from "./attnPanel.js";
import { renderScene } from "./render.js";

// Build the present-mode context for ONE step: its present-view analysis, the
// clean reasoning/output text, and per-object assets + zone-badge data. Shared by
// the single-step present and the end-to-end run. `fullProj` (the whole cell's
// scene, fetched once) is reused for asset/zone lookup.
async function buildPresentCtx(step, token, fullProj = null) {
	const run = state.run, slot = state.slot, model = state.model;
	let exp;
	try { exp = await api.tfExport(run, slot, model, step.event_index); } catch { return null; }
	if (token !== state.viewToken) return null;
	// The objects this step emits + their char spans (output_rel) — used to time
	// each object's card as its JSON is written.
	const outputMap = exp?.output_map ?? [];
	// Per-object asset (resolved bbox + rendered preview) from the FULL scene so a
	// mesh assigned in a later pass is still available; plus the decomposed zone's
	// bbox for the in-scene badge.
	const assets = {};
	let zone = null;
	let hideFrames = [];   // ancestor-zone frames to hide during placement steps (item: declutter)
	let hiddenBuildup = []; // nodes that don't exist YET at this step — hidden so the run BUILDS UP
	try {
		const proj = fullProj || await api.scene(run, slot, model, {});
		const nodes = proj.nodes || [];
		const ids = new Set(outputMap.map((m) => m.id));
		for (const n of nodes) {
			if (n.id === step.node && Array.isArray(n.origin) && Array.isArray(n.dimensions)) {
				zone = { id: n.id, origin: n.origin, dimensions: n.dimensions };
			}
			if (!ids.has(n.id)) continue;
			const meshUrl = typeof n.mesh_url === "string" ? n.mesh_url : null;
			assets[n.id] = {
				id: n.id,
				origin: Array.isArray(n.origin) ? n.origin : null,
				dimensions: Array.isArray(n.dimensions) ? n.dimensions : null,
				node_kind: n.node_kind,
				proxy_shape: n.proxy_shape,
				glb: meshUrl ? [api.absUrl(meshUrl.replace(/\.glb(\?|$)/, ".raw.glb$1")), api.absUrl(meshUrl)] : [],
				pngUrl: meshUrl ? api.absUrl(meshUrl.replace(/\.glb(\?|$)/, ".png$1")) : null,
			};
		}
		// Item #6: on placement steps, hide the current node's ANCESTOR-zone frames
		// (walls/shell), but keep the root's frames (ground). Walk parent_id up.
		if (PLACEMENT_STEPS.has(step.template ?? step.step ?? "")) {
			const parentOf = new Map(nodes.map((n) => [n.id, n.parent_id]));
			const ancestors = new Set();
			for (let cur = parentOf.get(step.node); cur; cur = parentOf.get(cur)) ancestors.add(cur);
			hideFrames = nodes
				.filter((n) => n.node_kind === "frame" && n.parent_id !== "root" && ancestors.has(n.parent_id))
				.map((n) => n.id);
		}
		// Build-up (stitched run only, where the FULL scene is loaded): everything
		// that doesn't exist yet at this step's render_until stays hidden until its
		// own step reveals it — so the video assembles instead of showing it all.
		if (fullProj) {
			const opts = step.render_until != null ? { untilIndex: step.render_until } : {};
			try {
				const baseProj = await api.scene(run, slot, model, opts);
				if (token !== state.viewToken) return null;
				// A zone_decompose creates its child-zone NODES (id/parent/kind) a step
				// BEFORE child_bbox_batch commits their actual boxes — so a node only
				// counts as PLACED once it has a committed bbox. Otherwise a just-
				// decomposed zone would pop into the scene (with its FINAL box) early,
				// before its child_bbox step. Bbox-less placeholders stay hidden.
				const placed = new Set(
					(baseProj.nodes || [])
						.filter((n) => Array.isArray(n.origin) && Array.isArray(n.dimensions))
						.map((n) => n.id),
				);
				hiddenBuildup = nodes.map((n) => n.id).filter((id) => !placed.has(id));
			} catch { /* if the per-step scene fails, fall back to the full scene */ }
		}
	} catch { /* non-fatal — present still works with placeholder cards + no manifest */ }
	// How many of THIS STEP'S KIND the zone ALREADY holds before this step (by
	// provenance — which region a node was EMITTED ONTO), so an incremental pass
	// builds on the existing total instead of restarting at 0.
	const wantFrame = (step.template ?? step.step) === "encapsulating_decompose";
	let zoneBase = 0;
	if (zone && state.obs) {
		for (const [id, prov] of state.obs.provenance) {
			let firstOnZone = Infinity;
			for (const p of prov) {
				if (p.relation === "emitted_by" && p.call?.node === zone.id && typeof p.call?.index === "number") {
					firstOnZone = Math.min(firstOnZone, p.call.index);
				}
			}
			if (firstOnZone >= step.event_index) continue;
			const frame = isFrameEntity(id);
			const kindZone = state.obs.nodes.get(id)?.kind === "zone" && !frame;
			if (wantFrame ? !frame : (frame || kindZone)) continue;
			zoneBase++;
		}
	}
	const reasoning = exp?.text?.reasoning ?? "", output = exp?.text?.output ?? "";
	const reasonChars = String(reasoning).replace(/\s*\n+\s*/g, " ").trim().length;
	const outputChars = String(output).length;
	// The per-token attention (head-summed entities/attributes/to-place) the compact
	// payload omits — loaded on demand, but ONLY when the step actually has a computed
	// map. Leading build steps (root plan/decompose/bbox/encap frames) usually have no
	// attention: rather than 404-ing and dropping them, they become BUILD-ONLY clips —
	// the scene keeps assembling (frames included) with the reasoning/output stream, but
	// no attention graph. We synthesize a frame count from the text so they still get a
	// proportional span on the timeline (no flash / no mismatch).
	const computed = ["ready", "stale"].includes(state.attnStatus[step.event_index]);
	let presentAnalysis = null;
	if (computed) {
		try { presentAnalysis = await api.attentionGet(run, slot, model, step.event_index, { view: "present" }); }
		catch { presentAnalysis = null; }
		if (token !== state.viewToken) return null;
	}
	const buildOnly = !(presentAnalysis?.tokens || []).length;
	// Timeline metrics (frames = tokens, or synthesized from text for build-only) so the
	// orchestrator can lay every step out on ONE summed, frame-weighted timeline — the
	// reason/output char split is how output-only mode locates the reasoning→output edge.
	const frameCount = buildOnly
		? Math.max(24, Math.min(600, Math.round((reasonChars + outputChars) / 3)))
		: presentAnalysis.tokens.length;
	return {
		viewer: state.viewer, host: $("tf-canvas-host"), analysis: presentAnalysis, buildOnly, step,
		stepTemplate: step.template ?? step.step ?? "",
		modelLabel: model || state.attn?.meta?.model_id || "the model",
		entityHex, entityKindLabel, compHex, COLORS,
		reasoning, output,
		outputMap, assets, zone, zoneBase,
		frameCount, reasonChars, outputChars,
		hideFrames, hiddenBuildup,
	};
}

// Launch present mode for the CURRENT step. Requires its attention map computed +
// on screen (the menu item is disabled otherwise). Reuses the loaded 3D scene.
async function presentCurrentStep() {
	const step = state.steps[state.stepIdx];
	const displayed = !!(state.attn && step && state.attn.meta && state.attn.meta.event_index === step.event_index);
	if (!displayed) return;
	state.presentSeqId++; // cancel any stitched run in flight
	const token = state.viewToken;
	const btn = $("tf-present-btn"), btnText = btn?.textContent;
	if (btn) { btn.disabled = true; btn.textContent = "loading…"; }
	const ctx = await buildPresentCtx(step, token);
	if (btn) { btn.disabled = false; btn.textContent = btnText; }
	if (!ctx || token !== state.viewToken) {
		if (!ctx) $("tf-attn-progress").textContent = "present: couldn't load attention detail";
		return;
	}
	ctx.onClose = () => renderAttention(); // restore the /tf 3D cross-highlight
	openPresent(ctx);
}

// The longest CONTIGUOUS run of computed (ready/stale) steps — what end-to-end
// stitches together.
export function longestComputedRun() {
	const ok = (s) => ["ready", "stale"].includes(state.attnStatus[s.event_index]);
	let best = [], cur = [];
	state.steps.forEach((s, i) => {
		if (ok(s)) { cur.push(i); if (cur.length > best.length) best = cur; }
		else cur = [];
	});
	return best;
}

// What end-to-end actually plays: every step from the very FIRST through the end of
// the longest computed run. The leading/interior steps that have no attention (root
// plan, decompose, bbox, encapsulating frames) ride along as BUILD-ONLY clips so the
// scene assembles from nothing — you watch the frames get built — instead of jumping
// into the middle with walls that never appear.
function endToEndIndices() {
	const run = longestComputedRun();
	if (!run.length) return [];
	const end = run[run.length - 1];
	return Array.from({ length: end + 1 }, (_, i) => i);
}

// End-to-end: stitch the present clips of the longest contiguous computed run
// into one continuous playback (each clip auto-advances to the next). The whole
// run is FULLY BUFFERED (scene + meshes + every clip's data) before playback so
// it's smooth start-to-finish, and any step that fails to load is skipped rather
// than stalling the run. `outputOnly` skips every step's reasoning ramp.
// The top-of-/tf buffering bar for an end-to-end run. `frac` (0..1) → determinate
// fill (steps buffered / total); pass null for an indeterminate sweep; false hides.
function setLoading(on, frac = null) {
	const bar = $("tf-loading");
	if (!bar) return;
	bar.classList.toggle("on", !!on);
	bar.classList.toggle("indet", !!on && frac == null);
	if (frac != null) bar.style.setProperty("--p", String(Math.max(0, Math.min(1, frac))));
}

async function presentSequence({ outputOnly }) {
	const indices = endToEndIndices();
	if (!indices.length) { $("tf-attn-progress").textContent = "present: no computed steps to stitch"; return; }
	const token = state.viewToken;
	const seq = ++state.presentSeqId; // supersedes any prior run
	setLoading(true); // /tf buffering bar (indeterminate) until the first clip plays
	let fullProj = null;
	try { fullProj = await api.scene(state.run, state.slot, state.model, {}); } catch { /* assets optional */ }
	if (seq !== state.presentSeqId) { setLoading(false); return; }
	const total = indices.length;
	const cache = new Map();    // k -> Promise<ctx|null>
	const resolved = new Map(); // k -> ctx|null (once built)

	// ONE unified, frame-weighted timeline for the whole run. Every step occupies a
	// contiguous span (its playable frame count); `offsets` are the running sums, so
	// the global playhead = offsets[k] + (local frames into step k). This is the
	// single source of truth for the progress bar + seeking, which is why switching
	// steps can no longer "reset" anything — position is always global. Spans start
	// as estimates and firm up as each clip buffers (shared object → present.js sees
	// the updates live).
	const startOf = (ctx) => (outputOnly && ctx.reasonChars + ctx.outputChars)
		? Math.min(ctx.frameCount - 1, Math.max(0, Math.round((ctx.reasonChars / (ctx.reasonChars + ctx.outputChars)) * ctx.frameCount)))
		: 0;
	const tl = { total: 1, offsets: [], spans: [], starts: [] };
	const recomputeTL = () => {
		const known = [...resolved.values()].filter((c) => c).map((c) => Math.max(1, c.frameCount - startOf(c)));
		const est = known.length ? Math.round(known.reduce((a, b) => a + b, 0) / known.length) : 60;
		let off = 0;
		for (let k = 0; k < total; k++) {
			const c = resolved.get(k);
			const span = c === null ? 0 : (c ? Math.max(1, c.frameCount - startOf(c)) : est); // null = skipped step
			tl.starts[k] = c ? startOf(c) : 0;
			tl.spans[k] = span;
			tl.offsets[k] = off;
			off += span;
		}
		tl.total = Math.max(1, off);
	};
	recomputeTL();
	const build = (k) => {
		if (k < 0 || k >= total) return Promise.resolve(null);
		if (!cache.has(k)) {
			// A single step's failure must NEVER reject — it becomes null (a skipped
			// span on the timeline) so one bad step can't tear down the whole run.
			const p = buildPresentCtx(state.steps[indices[k]], token, fullProj)
				.then((ctx) => ctx || null).catch(() => null)
				.then((ctx) => { resolved.set(k, ctx); recomputeTL(); return ctx; });
			cache.set(k, p);
		}
		return cache.get(k);
	};

	// FULLY BUFFER before playing so playback is smooth end-to-end (rather than
	// laggy/streaming): load the scene + ALL meshes, then every clip's data. The
	// determinate loading bar tracks it; each step is failure-isolated.
	if (fullProj && state.viewer) {
		try {
			state.viewer.clear();
			applySceneProjection(state.viewer, fullProj);
			await state.viewer.prefetchBundle(api.meshesUrl(state.run, state.slot, state.model, {}));
		} catch { /* keep whatever scene is loaded */ }
	}
	if (seq !== state.presentSeqId) { setLoading(false); return; }
	let done = 0;
	setLoading(true, 0);
	await Promise.all(indices.map((_, k) => build(k).then(() => {
		done++;
		if (seq === state.presentSeqId) setLoading(done < total, done / total);
	})));
	if (seq !== state.presentSeqId) { setLoading(false); return; }
	setLoading(false);
	recomputeTL();
	if (![...resolved.values()].some((c) => c && (c.buildOnly || (c.analysis?.tokens || []).length))) {
		$("tf-attn-progress").textContent = "present: no playable steps in the computed run";
		return;
	}
	let gen = 0; // guards against natural-advance vs. scrub-seek races (last call wins)
	const playAt = async (k, seekFrac = 0) => {
		const g = ++gen;
		if (seq !== state.presentSeqId) return;              // superseded / cancelled
		if (k >= total) { closePresent(); return; }          // run complete → onClose restores /tf
		const ctx = await build(k);
		if (seq !== state.presentSeqId || g !== gen) return; // cancelled or superseded mid-fetch
		if (!ctx || (!ctx.buildOnly && !(ctx.analysis?.tokens || []).length)) { playAt(k + 1); return; } // skip only broken steps
		ctx.outputOnly = outputOnly;
		ctx.speed = SEQ_SPEED;
		ctx.startFrame = startOf(ctx); // authoritative start (present.js reuses it → timeline stays consistent)
		ctx.seekFrac = seekFrac;
		ctx.sequence = {
			index: k, total, tl,
			// Cross-clip seek: the target step differs from the one playing, so the
			// orchestrator swaps clips (present.js seeks locally when it's the same step).
			onCross: (targetK, localFrac) => playAt(targetK, localFrac),
		};
		ctx.onEnded = () => playAt(k + 1);
		ctx.onClose = () => { // user bailed (Esc/✕) or run finished → cancel + restore
			if (seq === state.presentSeqId) state.presentSeqId++;
			setLoading(false); // stop buffering the bar for a run that's no longer playing
			const cur = state.steps[state.stepIdx];
			if (cur) renderScene(cur); // restore the launch step's 3D scene
			renderAttention();
		};
		// A bad clip must not stop the run — if opening it throws, skip to the next.
		try { advancePresent(ctx); } // opens on the first clip, then swaps in place → seamless
		catch (err) { console.error("[present] clip failed:", err); playAt(k + 1); }
	};
	playAt(0);
}

// The present dropdown: single-step vs end-to-end (with / without reasoning).
export function wirePresentMenu() {
	const wrap = $("tf-present-wrap"), btn = $("tf-present-btn"), menu = $("tf-present-menu");
	if (!wrap || !btn || !menu) return;
	const close = () => { menu.hidden = true; };
	btn.onclick = (e) => {
		e.stopPropagation();
		if (!menu.hidden) { close(); return; }
		const step = state.steps[state.stepIdx];
		const displayed = !!(state.attn && step && state.attn.meta && state.attn.meta.event_index === step.event_index);
		const run = endToEndIndices();
		menu.querySelector('[data-mode="step"]').disabled = !displayed;
		menu.querySelector('[data-mode="e2e"]').disabled = !run.length;
		menu.querySelector('[data-mode="e2e-out"]').disabled = !run.length;
		const sec = menu.querySelector(".tf-pm-sec span");
		if (sec) sec.textContent = run.length ? `${run.length} steps` : "none computed";
		menu.hidden = false;
	};
	menu.onclick = (e) => {
		const item = e.target.closest?.(".tf-pm-item");
		if (!item || item.disabled) return;
		close();
		const mode = item.dataset.mode;
		if (mode === "step") presentCurrentStep();
		else if (mode === "e2e") presentSequence({ outputOnly: false });
		else if (mode === "e2e-out") presentSequence({ outputOnly: true });
	};
	document.addEventListener("click", (e) => { if (!wrap.contains(e.target)) close(); });
}
