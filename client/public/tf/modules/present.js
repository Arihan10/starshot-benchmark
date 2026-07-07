// "Watch it build" present mode. Replays one LLM step's generation as a stream
// of thought: the reasoning + output type out along the bottom, and — the star —
// an object INVENTORY on the right fills as the model emits objects.
//
//   • decompose steps (anchor / next / encap / neg-space): the model is
//     INVENTING objects, so each emitted child pops into the tray as its
//     would-be asset (a rendered preview, or a placeholder when none exists).
//   • object_bbox_batch steps: the model is PLACING those objects, so each
//     emitted assignment then FLIES from the tray into the 3D scene and its
//     bounding box slots into place at the resolved location.
//
// A dim neural glow on attended objects + a scene-mass EEG stay as the ambient
// "brain" heartbeat. Fullscreen, looping, capture-ready.
//
// Decoupled from tf.js: everything arrives in `ctx` (viewer + step + analysis +
// the emitted objects with their resolved bboxes + asset preview urls), so
// there is no import cycle.

import { el } from "../../js/ui.js";

const $ = (id) => document.getElementById(id);
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const ease = (t) => t * t * (3 - 2 * t);
const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
const SPEEDS = [0.5, 1, 2, 4, 8, 16, 32];
const DEFAULT_SPEED = 4; // reasoning trace default — output phase still lingers (OUTPUT_SLOWDOWN)
const CORE_COLOR = "#9db8ff"; // the graph's central "focus" hub
const TAU = Math.PI * 2;
// Below this peak per-head scene mass the model is barely looking at the scene —
// skip lighting objects (the normalized top entities are just noise). Tunable.
const MIN_SCENE_MASS = 0.02;
// The output/placement phase is the payoff (objects materialize + get placed),
// so play it slower than the reasoning ramp so it reads.
const OUTPUT_SLOWDOWN = 0.45;
const NEW_OBJECT_COLOR = "#ffd166"; // objects THIS step places — a warm gold
const GROW_MS = 460;  // bbox slot-in growth once a thumbnail lands
const FLY_MS = 620;   // tray → 3D destination flight time
const THUMB_PX = 132;          // card render backing size (square)
const THUMB_MAX_PER_FRAME = 6; // cap GL readbacks/frame; ≤ this many cards ⇒ every card refreshes every frame (smooth)
const ROT_SPEED = 0.5;         // turntable spin (rad/s) — calm + natural
// Attention "focus dial": a two-ring web around the model's focus (the hub).
// The INNER ring holds the attributes it consults (placement / dimensions /
// relationships / …); the OUTER ring holds the objects it attends. Each attended
// (object, attribute) is an edge object→attribute, and every attribute links
// inward to the hub — so attention reads as objects → attributes → focus.
// Attribute directions fill the largest angular gap as they appear (never clumps).
// What each pipeline step EMITS — the authoritative source for the tray noun
// (asset node_kind is unreliable per step): decompose/bbox passes make zones, the
// encapsulating pass makes frames, the anchor/next/batch passes make objects.
const STEP_KIND = {
	zone_plan_root: "zone", zone_plan: "zone",
	overall_bbox: "zone", zone_decompose_root: "zone", zone_decompose: "zone", child_bbox_batch: "zone",
	encapsulating_decompose: "frame",
	anchor_decompose: "object", next_object: "object", object_bbox_batch: "object",
};

const MAX_EDGES = 8;           // cap (object, attribute) links (evict the weakest beyond this)
const MAX_OBJS = 4;            // cap objects on the outer ring (evict the weakest beyond this)
const MAX_EDGES_PER_OBJ = 2;   // cap spokes PER object so one can't monopolise the dial (diversity)
// Admission gate: a NEW object/zone must be attended across several tokens (and
// accumulate enough weight) before it earns a ring slot — and the bar RISES as
// the outer ring fills, so newcomers compete against what's already up there.
const ENTER_HITS_MIN = 3;      // attended tokens needed to enter an EMPTY ring
const ENTER_HITS_RANGE = 6;    // …plus up to this many more as the ring fills (→ 9 when full)
const ENTER_SCORE_MIN = 0.55;  // accumulated attention weight needed to enter an empty ring
const ENTER_SCORE_COMP = 3;    // …scaled up this much as the ring fills (competition)
const CAND_DECAY_TAU = 0.9;    // s — candidate appearances fade so scattered glances never accrete
const CUM_DECAY_TAU = 1.2;     // s — dwell score fades FAST so incumbents demote & lose their slot quickly (turnover)
const GRAPH_TOP_OBJS = 5;      // per-token: only the top N scene objects can enter the graph
const GRAPH_CONCENTRATION = 2; // must exceed (1/n_scene) × this to count as focused attention
const R_IN = 0.42, R_OUT = 0.84; // inner ring = attributes, outer ring = objects (fraction of dial radius)
// Elastic ring layout: nodes spring toward a home angle, repel neighbours, and
// wander more when rarely attended (low cumulative frequency).
const RELAX_DAMP = 0.74;
const RELAX_RAD_DAMP = 0.78;
const RELAX_REP = 5.2; // neighbour repulsion on each ring (radians/s² scale)

let S = null; // active session, or null when closed

// --- open / close ------------------------------------------------------------

export function openPresent(ctx) {
	if (S) closePresent(true); // silent: reopening (e.g. next clip in a stitched run) isn't an exit
	if (!ctx.analysis || !(ctx.analysis.tokens || []).length) { ctx.onEnded?.(); return; } // nothing to play
	const host = ctx.host;
	const overlay = $("tf-present");
	// Session scaffolding (set up ONCE): the fullscreen stage, the live 3D viewer,
	// controls + listeners, and the graph — all PERSIST across an end-to-end run so
	// swapping to the next step's clip never tears the session down (no reparent /
	// overlay flash). Per-step state is (re)loaded by applyClip.
	S = {
		host, overlay, origParent: host.parentNode, origNext: host.nextSibling,
		// speed is set ONCE for the session (from the run) and then owned by the user —
		// clip swaps must not reset it (see applyClip).
		playing: true, loop: false, speed: ctx.speed || DEFAULT_SPEED, tps: 8, // tokens/sec at 1×
		last: 0, raf: 0, hud: true, dpr: 1,
		reduced: window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches || false,
		cards: [], flies: [], rotCursor: -1,
		// attention graph: nodes SPAWN as the model consults an (object, attribute);
		// kept alive across clips so a stitched run CROSSFADES attention (old nodes
		// decay/prune as the next step's fire in) instead of hard-cutting.
		graph: { nodes: new Map(), compAngles: new Map(), attrs: new Map(), objs: new Map(), cand: new Map(), pulses: [], core: 0, coreTarget: 0, center: { x: 0, y: 0 }, R: 0 },
	};
	// Move the live 3D viewer into the fullscreen stage (reuses the loaded scene
	// + meshes; the ResizeObserver on the host re-fits it to fullscreen).
	$("tfp-stage").appendChild(host);
	overlay.hidden = false;
	overlay.setAttribute("aria-hidden", "false");
	overlay.classList.remove("hud-off");
	ctx.viewer?.clearAttnHighlight?.(); // drop the static /tf cross-highlight; neural glow takes over
	ctx.viewer?.setSceneDim?.(true);
	ctx.viewer?.setAutoOrbit?.(!S.reduced, 0.5);
	ctx.viewer?.setActive?.(true);
	wireControls();
	window.addEventListener("keydown", onKey, true);
	window.addEventListener("resize", sizeCanvases);
	if (!applyClip(ctx)) { closePresent(true); ctx.onEnded?.(); return; }
	sizeCanvases();
	S.last = performance.now();
	S.raf = requestAnimationFrame(tick);
}

// Advance a LIVE session to the next step's clip in place — the running rAF loop
// keeps going, so an end-to-end run plays as one continuous video. Falls back to
// a fresh openPresent when nothing is open yet.
export function advancePresent(ctx) {
	if (!S) { openPresent(ctx); return; }
	if (!applyClip(ctx)) ctx.onEnded?.(); // empty step → let the run skip to the next
}

// Load ONE step's clip into the (already set-up) session: playhead, per-token
// data, thought stream, object tray, zone badge, and per-clip viewer overlays.
// Leaves the session scaffolding + attention graph untouched. Returns false if
// the step has nothing to play.
function applyClip(ctx) {
	// Build-only clip: a leading step (root plan / decompose / bbox / encapsulating
	// frames) with NO computed attention. We fabricate empty frames so the 3D scene
	// still assembles (the frames get built) and the reasoning/output still streams,
	// but the attention graph + EEG stay dark (see the attn-off class below).
	const buildOnly = !!ctx.buildOnly || !(ctx.analysis?.tokens || []).length;
	const a = buildOnly
		? { tokens: Array.from({ length: Math.max(1, ctx.frameCount || 1) }, () => ({ heads: [] })), selected_heads: [], meta: {} }
		: ctx.analysis;
	const data = buildData(a, ctx);
	const template = ctx.stepTemplate || "";
	// output-only (end-to-end without reasoning): begin the playhead at the
	// reasoning→output boundary (mapped from the char split the reveal uses) so the
	// reasoning ramp is skipped for ANY step kind, including prose-only zone plans.
	// In a stitched run the orchestrator computes this (ctx.startFrame) so its summed
	// timeline and our playhead agree exactly; single-step present derives it here.
	let startFrame = ctx.startFrame;
	if (typeof startFrame !== "number") {
		const reasonChars = String(ctx.reasoning ?? "").replace(/\s*\n+\s*/g, " ").trim().length;
		const totalChars = reasonChars + String(ctx.output ?? "").length;
		startFrame = (ctx.outputOnly && totalChars)
			? Math.min(data.frames.length - 1, Math.max(0, Math.round((reasonChars / totalChars) * data.frames.length)))
			: 0;
	}
	startFrame = Math.max(0, Math.min(data.frames.length - 1, startFrame));
	S.ctx = ctx; S.a = a; S.data = data; S.template = template; S.buildOnly = buildOnly;
	// Dark graph/EEG while a build-only step plays (nothing to attend to yet).
	S.overlay.classList.toggle("attn-off", buildOnly);
	// An object_bbox_batch MANIFESTS objects into 3D; every other step just fills the tray.
	S.placing = template === "object_bbox_batch";
	S.outputOnly = !!ctx.outputOnly; S.startFrame = startFrame;
	S.onEnded = ctx.onEnded || null; S.sequence = ctx.sequence || null;
	S.tl = ctx.sequence?.tl || null; // shared summed-timeline (whole-run bar + seeking)
	S.zoneBadge = (template !== "object_bbox_batch" && ctx.zone && Array.isArray(ctx.zone.origin) && Array.isArray(ctx.zone.dimensions)) ? ctx.zone : null;
	S.zoneBase = ctx.zoneBase || 0;
	// zone_plan (incl. zone_plan_root): its output is JSON — de-JSON it everywhere
	// (badge when there's a zone, and always the thought stream). The root has no
	// single zone to badge, so the stream de-JSON is what surfaces its plan.
	S.zonePlan = /^zone_plan/.test(template);
	// Seek into the clip (used when scrubbing the whole-run progress bar lands here).
	const seekFrac = Math.max(0, Math.min(0.999, ctx.seekFrac || 0));
	S.pos = startFrame + seekFrac * (data.frames.length - startFrame);
	S.lastFired = -1; S.revealed = -1; S.ended = false; S.playing = true;
	S.cards = []; S.flies = []; S.rotCursor = -1;
	// Clear the previous clip's in-scene overlays (placement boxes, glow, focus,
	// hidden set, in-flight thumbnails) so nothing bleeds across the cut.
	ctx.viewer?.clearResolving?.();
	ctx.viewer?.clearNeuralActivation?.();
	ctx.viewer?.setFocus?.(null);
	ctx.viewer?.clearPresentHidden?.();
	$("tfp-fly")?.replaceChildren();

	$("tfp-model").textContent = ctx.modelLabel || "the model";
	const seq = $("tfp-seq");
	if (seq) seq.textContent = S.sequence ? ` · end-to-end ${S.sequence.index + 1}/${S.sequence.total}${S.outputOnly ? " · output" : ""}` : "";
	$("tfp-sub").replaceChildren(
		el("span", { text: (ctx.step?.template ?? ctx.step?.step ?? "?") }),
		el("span", { class: "dim", text: " on " }),
		el("span", { text: (ctx.step?.node ?? "?") }),
	);
	$("tfp-graph-sub").textContent = buildOnly ? "building scene" : `${data.nLayers} layers · ${data.nHeads} heads`;

	// In-scene status badge over the decomposed zone (its name here; count via
	// updateInvSub, position via updateZoneBadge once it's projected).
	const zbadge = $("tfp-zone-badge");
	if (S.zoneBadge) {
		$("tfp-zone-name").textContent = shortId(ctx.zone.id);
		zbadge.classList.toggle("plan", S.zonePlan);
		$("tfp-zone-count-row").hidden = S.zonePlan; // plan streams prose, not a count
		$("tfp-zone-plan").textContent = "";
		const noun = invNoun();
		// Tint the badge to the emitted kind: frames blue, zones red, objects green
		// (default). Cleared on plan clips so a prior clip's tint doesn't bleed.
		zbadge.classList.toggle("frames", !S.zonePlan && noun === "frames");
		zbadge.classList.toggle("zones", !S.zonePlan && noun === "zones");
		if (!S.zonePlan) {
			$("tfp-zone-count").textContent = String(S.zoneBase);
			zbadge.querySelector(".tfp-zone-cap").textContent = noun; // "frames"/"zones"/"objects"
		}
	}
	zbadge.hidden = true;

	buildStream();
	buildInventory();
	// What to hide in the 3D scene for this clip:
	//  · build-up: nodes that don't exist yet at this step (stitched run only),
	//  · ancestor frames: decluttered walls on placement steps,
	//  · placeables: this step's objects, until their thumbnail flies in.
	const hidden = new Set(ctx.hideFrames || []);
	if (S.sequence) for (const id of ctx.hiddenBuildup || []) hidden.add(id);
	if (S.placing) for (const c of S.cards) if (c.canPlace) hidden.add(c.id);
	ctx.viewer?.setPresentHidden?.([...hidden]);
	setPlay(true);
	setSpeed(S.speed || DEFAULT_SPEED); // keep the user's chosen speed across clip swaps (don't reset to ctx.speed)
	return true;
}

export function closePresent(silent = false) {
	if (!S) return;
	cancelAnimationFrame(S.raf);
	window.removeEventListener("keydown", onKey, true);
	window.removeEventListener("resize", sizeCanvases);
	for (const f of S.flies) f.node.remove();
	$("tfp-fly")?.replaceChildren();
	$("tfp-zone-badge").hidden = true;
	S.ctx.viewer?.clearResolving?.();
	S.ctx.viewer?.clearThumbAssets?.(); // free the offscreen GLBs + render target
	S.ctx.viewer?.clearPresentHidden?.(); // restore objects hidden during placement
	S.ctx.viewer?.setFocus?.(null);
	S.ctx.viewer?.clearNeuralActivation?.();
	S.ctx.viewer?.setSceneDim?.(false);
	S.ctx.viewer?.setAutoOrbit?.(false);
	if (S.origParent) S.origParent.insertBefore(S.host, S.origNext); // restore the viewer
	S.ctx.viewer?.setActive?.(true);
	S.overlay.hidden = true;
	S.overlay.setAttribute("aria-hidden", "true");
	S.overlay.classList.remove("hud-off");
	const onClose = S.ctx.onClose;
	S = null;
	// silent close = we're immediately reopening (next clip in a stitched run), so
	// don't run the exit handler that restores /tf's own 3D cross-highlight.
	if (!silent) onClose?.();
}

// --- static "attention in 3D" view -------------------------------------------
// Reuses the present view's fullscreen 3D module — the SAME overlay + stage +
// live-viewer reparenting + turntable orbit — but WITHOUT token playback. It
// simply shades every scene bounding box by its aggregated attention weight via
// the viewer's setAttnHighlight (cool → hot). Separate session state (`A`) from
// the playback session (`S`) so the two never clobber each other.
let A = null;

export function openAttn3D(ctx) {
	if (S) closePresent(true);      // playback and the static view can't share the stage
	if (A) closeAttn3D(true);       // reopening (e.g. new scope) — silent, no exit handler
	const host = ctx.host;
	const overlay = $("tf-present");
	A = { host, overlay, origParent: host.parentNode, origNext: host.nextSibling, ctx };
	A.savedShow = ctx.viewer?.getVisibility?.() || null; // restore mesh/bbox visibility on close
	$("tfp-stage").appendChild(host); // move the live viewer into the fullscreen stage
	overlay.hidden = false;
	overlay.setAttribute("aria-hidden", "false");
	overlay.classList.remove("hud-off", "attn-off");
	overlay.classList.add("attn3d");  // CSS hides the playback HUD (graph/tray/transport/EEG)
	$("tfp-model").textContent = ctx.modelLabel || "the model";
	const seq = $("tfp-seq"); if (seq) seq.textContent = "";
	$("tfp-sub").replaceChildren(
		el("span", { text: ctx.title || "attention in 3D" }),
		ctx.sub ? el("span", { class: "dim", text: ` · ${ctx.sub}` }) : el("span", {}),
	);
	// bbox-only toggle: hide the meshes so only the wireframe boxes + attention
	// shading remain (persisted). Starts from the saved preference.
	let bboxOnly = false;
	try { bboxOnly = localStorage.getItem("tf-attn3d-bbox") === "1"; } catch { /* ignore */ }
	const bboxBtn = el("button", { class: `tfp-attn-toggle${bboxOnly ? " on" : ""}`, title: "show only bounding boxes (hide meshes)", text: "bbox only" });
	const applyBboxOnly = () => {
		bboxBtn.classList.toggle("on", bboxOnly);
		ctx.viewer?.setMeshesVisible?.(!bboxOnly);
		if (bboxOnly) ctx.viewer?.setBboxesVisible?.(true);
		else if (A?.savedShow) ctx.viewer?.setBboxesVisible?.(A.savedShow.bboxes);
	};
	bboxBtn.onclick = () => {
		bboxOnly = !bboxOnly;
		try { localStorage.setItem("tf-attn3d-bbox", bboxOnly ? "1" : "0"); } catch { /* ignore */ }
		applyBboxOnly();
	};
	// Attended entities to shade: [{ id, score, kind }] (kind ∈ zone|frame|object).
	// Fall back to legacy pre-normalized items for older callers.
	const entities = ctx.entities || (ctx.items || []).map((it) => ({ id: it.id, score: it.weight ?? 1, kind: it.kind || "object" }));
	const kindColors = ctx.kindColors || {};
	// Type filter: one toggle per kind present. Persisted across opens.
	const KIND_ORDER = ["zone", "frame", "object"];
	const presentKinds = KIND_ORDER.filter((k) => entities.some((e) => (e.kind || "object") === k));
	for (const e of entities) { const k = e.kind || "object"; if (!presentKinds.includes(k)) presentKinds.push(k); }
	const activeKinds = new Set(presentKinds);
	try {
		const saved = (localStorage.getItem("tf-attn3d-kinds") || "").split(",").filter(Boolean);
		const keep = saved.filter((k) => presentKinds.includes(k));
		if (keep.length) { activeKinds.clear(); keep.forEach((k) => activeKinds.add(k)); }
	} catch { /* ignore */ }
	// min-attention slider: entities below this fraction of the PEAK attention are
	// not highlighted at all. Re-shades live as you drag; persisted across opens.
	const baseOpts = ctx.highlightOpts || { gamma: 2.0, minWeight: 0.15, contrast: true };
	let minW = baseOpts.minWeight ?? 0.15;
	try { const v = Number(localStorage.getItem("tf-attn3d-minw")); if (v >= 0 && v <= 0.95) minW = v; } catch { /* ignore */ }
	// Recompute the shaded set from the active kinds + bbox presence, normalizing
	// weight over THAT visible subset so each shown type uses the full hot ramp
	// (objects aren't crushed by a huge zone — or by a zone that isn't even in the
	// currently-rendered scene). Missing-bbox entities can't be shaded, so they're
	// excluded from the scale too.
	const computeItems = () => {
		const vis = entities.filter((e) => activeKinds.has(e.kind || "object") && (ctx.viewer?.hasBbox?.(e.id) ?? true));
		const max = Math.max(...vis.map((e) => e.score), 1e-9);
		return vis.map((e) => ({ id: e.id, weight: e.score / max }));
	};
	const applyHighlight = () => ctx.viewer?.setAttnHighlight?.(computeItems(), { ...baseOpts, minWeight: minW });
	const minwLab = el("span", { class: "tfp-attn-minw-lab", text: `${Math.round(minW * 100)}%` });
	const minwSlider = el("input", {
		type: "range", min: "0", max: "0.9", step: "0.01", value: String(minW), class: "tfp-attn-minw-slider",
		title: "hide entities below this fraction of the peak attention",
		oninput: (ev) => {
			minW = Number(ev.target.value);
			try { localStorage.setItem("tf-attn3d-minw", String(minW)); } catch { /* ignore */ }
			minwLab.textContent = `${Math.round(minW * 100)}%`;
			applyHighlight();
		},
	});
	const minwGroup = el("div", { class: "tfp-attn-minw" },
		el("span", { class: "tfp-attn-minw-cap", text: "min attention" }), minwSlider, minwLab);
	// Type filter buttons (one per present kind); toggling re-normalizes over the
	// visible subset so isolating a type makes it span the full color ramp.
	const kindBtns = presentKinds.map((k) => {
		const b = el("button", { class: `tfp-attn-kind${activeKinds.has(k) ? " on" : ""}`, title: `toggle ${k}s` },
			el("i", { class: "tfp-attn-kind-dot", style: kindColors[k] ? `background:${kindColors[k]}` : "" }),
			el("span", { text: k }));
		b.onclick = () => {
			if (activeKinds.has(k)) activeKinds.delete(k); else activeKinds.add(k);
			b.classList.toggle("on", activeKinds.has(k));
			try { localStorage.setItem("tf-attn3d-kinds", [...activeKinds].join(",")); } catch { /* ignore */ }
			applyHighlight();
		};
		return b;
	});
	const kindGroup = presentKinds.length > 1
		? el("div", { class: "tfp-attn-kinds" }, el("span", { class: "tfp-attn-kinds-cap", text: "types" }), ...kindBtns)
		: null;
	// Floating legend + type filter + slider + toggle + close (transport bar hidden in attn3d).
	const panel = el("div", { id: "tfp-attn" },
		el("div", { class: "tfp-attn-legend" },
			el("span", { text: "attention" }),
			el("span", { class: "tfp-attn-lo", text: "low" }),
			el("span", { class: "tfp-attn-bar" }),
			el("span", { class: "tfp-attn-hi", text: "high" })),
		kindGroup,
		minwGroup,
		bboxBtn,
		el("button", { class: "tfp-attn-close", title: "close (Esc)", text: "✕ close", onclick: () => closeAttn3D() }),
	);
	overlay.appendChild(panel);
	A.panel = panel;
	const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches || false;
	ctx.viewer?.setActive?.(true);
	ctx.viewer?.setSceneDim?.(true);
	ctx.viewer?.setAutoOrbit?.(!reduced, 0.4);
	ctx.viewer?.clearNeuralActivation?.();
	// Stronger contrast + drop faint entities for the static view (see setAttnHighlight).
	applyHighlight();
	applyBboxOnly();
	window.addEventListener("keydown", onAttnKey, true);
}

function onAttnKey(e) {
	if (e.key === "Escape") { e.preventDefault(); closeAttn3D(); }
}

export function closeAttn3D(silent = false) {
	if (!A) return;
	window.removeEventListener("keydown", onAttnKey, true);
	const { ctx } = A;
	ctx.viewer?.clearAttnHighlight?.();
	ctx.viewer?.setSceneDim?.(false);
	ctx.viewer?.setAutoOrbit?.(false);
	if (A.savedShow) { // undo any bbox-only toggle
		ctx.viewer?.setMeshesVisible?.(A.savedShow.meshes);
		ctx.viewer?.setBboxesVisible?.(A.savedShow.bboxes);
	}
	if (A.origParent) A.origParent.insertBefore(A.host, A.origNext); // restore the viewer
	ctx.viewer?.setActive?.(true);
	A.panel?.remove();
	A.overlay.hidden = true;
	A.overlay.setAttribute("aria-hidden", "true");
	A.overlay.classList.remove("attn3d");
	const onClose = ctx.onClose;
	A = null;
	if (!silent) onClose?.();
}

// --- precompute per-token frames (scene mass + top attended entities) --------

function buildData(a, ctx) {
	const heads = a.selected_heads || [];
	const tokens = a.tokens || [];
	const nSceneEntities = Math.max(1, a.meta?.n_scene_entities ?? a.scene_entities?.length ?? 1);
	const uniformShare = 1 / nSceneEntities;
	const minEntShare = uniformShare * GRAPH_CONCENTRATION;
	let massMax = 1e-9;
	const frames = tokens.map((t) => {
		const hs = t.heads || [];
		let sm = 0, peak = 0, tpPeak = 0;
		const em = new Map();  // entity -> {id,kind,score}   (drives the 3D neural glow)
		const am = new Map();  // "entity|component" -> {..}  (drives the graph — the important part)
		const tpm = new Map(); // to-place entity -> score     (inventory attention halos)
		for (const h of hs) {
			sm += h.scale || 0;
			peak = Math.max(peak, h.scale || 0);
			for (const e of h.top_entities || []) {
				const ce = em.get(e.id) || { id: e.id, kind: e.kind, score: 0 };
				ce.score += e.score; em.set(e.id, ce);
				// Per-ATTRIBUTE attention: what OF the object is being consulted —
				// its dimensions, placement, relationships, … the meaningful axes.
				for (const [comp, sc] of Object.entries(e.components || {})) {
					if (!(sc > 0)) continue;
					const key = `${e.id}|${comp}`;
					const ca = am.get(key) || { key, entity: e.id, kind: e.kind, component: comp, score: 0 };
					ca.score += sc; am.set(key, ca);
				}
			}
			const tp = h.to_place;
			if (tp) {
				tpPeak = Math.max(tpPeak, tp.scale || 0);
				for (const e of tp.top_entities || []) tpm.set(e.id, (tpm.get(e.id) || 0) + e.score);
			}
		}
		const mass = hs.length ? sm / hs.length : 0;
		massMax = Math.max(massMax, mass);
		const ents = [...em.values()].sort((x, y) => y.score - x.score);
		const entSum = ents.reduce((s, e) => s + e.score, 0) || 1;
		for (const e of ents) {
			e.share = e.score / entSum;
			e.color = ctx.entityHex(e.kind, e.id);
		}
		const emax = ents[0]?.score || 1;
		for (const e of ents) e.weight = e.score / emax;
		// Early pipeline steps have few scene objects — uniform attention makes
		// every object look "fully attended" if we only normalize to the token max.
		// Require a share above the uniform baseline (scaled by n_scene) and cap
		// to the top GRAPH_TOP_OBJS so the dial differentiates focus per step.
		const qualEnts = [];
		for (const e of ents) {
			if (qualEnts.length >= GRAPH_TOP_OBJS) break;
			if (e.share >= minEntShare || (qualEnts.length === 0 && e.score > 0)) qualEnts.push(e);
		}
		const qualIds = new Set(qualEnts.map((e) => e.id));
		const attrs = [...am.values()]
			.filter((x) => qualIds.has(x.entity))
			.sort((x, y) => y.score - x.score)
			.slice(0, GRAPH_TOP_OBJS * 2);
		const amax = attrs[0]?.score || 1;
		for (const x of attrs) x.weight = x.score / amax;
		const tpEnts = [...tpm.entries()].map(([id, score]) => ({ id, score })).sort((x, y) => y.score - x.score);
		const tmax = tpEnts[0]?.score || 1;
		for (const e of tpEnts) e.weight = e.score / tmax;
		return { mass, peak, ents, qualEnts, attrs, tpEnts, tpPeak, output_entity: t.output_entity };
	});
	const outStart = frames.findIndex((f) => f.output_entity != null);
	const hasToPlace = (a.meta?.to_place_present ?? false) || frames.some((f) => f.tpPeak > 0);
	// Per to-place object, the attention it RECEIVES accumulated across the
	// generation (indexed by frame, so the playhead reveals it building up to the
	// step total). This is the "how the model attends to EACH to-place object"
	// aggregate — a property of the target object, not of whatever's being placed.
	const tpIds = new Set();
	for (const f of frames) for (const e of f.tpEnts) tpIds.add(e.id);
	const tpCum = new Map([...tpIds].map((id) => [id, new Float64Array(frames.length)]));
	const run = new Map();
	frames.forEach((f, k) => {
		for (const e of f.tpEnts) run.set(e.id, (run.get(e.id) || 0) + e.score);
		for (const id of tpIds) tpCum.get(id)[k] = run.get(id) || 0;
	});
	const last = frames.length - 1;
	let tpMaxTotal = 1e-9;
	for (const id of tpIds) tpMaxTotal = Math.max(tpMaxTotal, tpCum.get(id)[last] || 0);
	return {
		frames, massMax, nSceneEntities, minEntShare, outStart: outStart < 0 ? frames.length : outStart,
		nLayers: new Set(heads.map((h) => h.layer)).size, nHeads: heads.length, hasToPlace, tpCum, tpMaxTotal,
	};
}

// --- playback loop -----------------------------------------------------------

function tick(now) {
	if (!S) return;
	try {
		const dt = Math.min(0.05, (now - S.last) / 1000);
		S.last = now;
		const n = S.data.frames.length;
		if (S.playing) {
			// Linger on the payoff: once the model crosses from reasoning into writing
			// the output objects, slow the playhead so the tray + placement read.
			const slow = (S.reveal || 0) >= S.reasonLen ? OUTPUT_SLOWDOWN : 1;
			S.pos += S.speed * S.tps * slow * dt;
			if (S.pos >= n) {
				if (S.loop) { S.pos = S.startFrame; S.lastFired = -1; resetPlacement(); }
				else if (S.onEnded) {
					// Stitched run: fire onEnded ONCE and hold on the last frame (loop
					// stays alive) until the next clip is swapped in via applyClip — so
					// the transition is seamless rather than a stop/teardown/restart.
					if (!S.ended) { S.ended = true; S.pos = n - 0.001; S.playing = false; S.onEnded(); }
				}
				else { S.pos = n - 0.001; setPlay(false); }
			}
		}
		const i = Math.max(0, Math.min(n - 1, Math.floor(S.pos)));
		S.reveal = S.total ? Math.round((S.pos / n) * S.total) : 0;
		if (i !== S.lastFired) { fireToken(i, now); S.lastFired = i; }
		updateGraph(dt);
		drawGraph(now);
		updateCardAttn(i, dt);
		drawEeg(i);
		updateStream();
		syncInventory(now);
		rotateThumbs(now);
		updateFlies(now);
		updateZoneBadge();
		updateScrub(i);
	} catch (err) {
		console.error("[present] tick failed:", err);
	}
	S.raf = requestAnimationFrame(tick);
}

// Fire everything for token i: the attention graph SPAWNS/re-lights a node per
// (object, attribute) it consults this token (dimensions of X, placement of Y),
// the hub core tracks scene mass, and the ambient 3D neural glow lights the
// strongest attended objects (dimmer during placement so the gold boxes stay the
// star). Skips the scene when the model isn't looking.
function fireToken(i, now) {
	const f = S.data.frames[i];
	if (!f) return;
	S.graph.coreTarget = Math.max(S.graph.coreTarget || 0, clamp01(f.mass / (S.data.massMax || 1)));
	if (f.peak < MIN_SCENE_MASS) { S.ctx.viewer?.setNeuralActivation?.([]); return; }
	// Attribute spawn threshold rises when the scene is small — otherwise every
	// object clears the bar and the dial looks identical on early steps.
	const graphMinW = Math.max(0.18, 1.8 / S.data.nSceneEntities);
	for (const x of f.attrs) if (x.weight >= graphMinW) considerNode(x, i, now);
	if (S.graph.pulses.length > 60) S.graph.pulses.splice(0, S.graph.pulses.length - 60);
	const gain = (S.reveal >= S.reasonLen) ? 0.4 : 0.8;
	const glowEnts = (f.qualEnts?.length ? f.qualEnts : f.ents).slice(0, 3);
	S.ctx.viewer?.setNeuralActivation?.(glowEnts.map((e) => ({ id: e.id, weight: e.weight * gain, color: e.color })));
	// Per-token to-place attention: flash the cards THIS token attends to (how much
	// each token — reasoning or output — looks at the object it needs to place).
	// The persistent halo (updateCardAttn) is the running aggregate; this is the live pulse.
	if (S.data.hasToPlace && (f.tpPeak || 0) >= MIN_SCENE_MASS) {
		for (const e of f.tpEnts) {
			const card = S.cardById?.get(e.id);
			if (card) card.pulse = Math.max(card.pulse || 0, e.weight);
		}
	}
}

// Give an attribute its spoke direction the first time it appears, dropping it
// into the LARGEST current angular gap so the dial stays balanced; the angle is
// then stable for that attribute (beads share it) until the spoke empties.
function assignCompAngle(comp) {
	const g = S.graph;
	if (g.compAngles.has(comp)) return g.compAngles.get(comp);
	const used = [...g.compAngles.values()].sort((a, b) => a - b);
	let ang;
	if (!used.length) ang = -Math.PI / 2;                 // first spoke points up
	else if (used.length === 1) ang = used[0] + Math.PI;  // second directly opposite
	else {
		let widest = -1, mid = used[0];
		for (let i = 0; i < used.length; i++) {
			const a = used[i], b = (i + 1 < used.length ? used[i + 1] : used[0] + TAU);
			if (b - a > widest) { widest = b - a; mid = a + (b - a) / 2; }
		}
		ang = mid;
	}
	ang = ((ang % TAU) + TAU) % TAU;
	g.compAngles.set(comp, ang);
	return ang;
}

// Drop attributes/objects that no longer back any edge. Mark them `dying` so
// spawn EMA can fade them out instead of an instant pop.
function pruneRings() {
	const g = S.graph;
	const liveComp = new Set(), liveEnt = new Set();
	for (const e of g.nodes.values()) { liveComp.add(e.component); liveEnt.add(e.entity); }
	for (const k of [...g.attrs.keys()]) {
		if (liveComp.has(k)) continue;
		const m = g.attrs.get(k);
		if (m) m.dying = true;
		else { g.attrs.delete(k); g.compAngles.delete(k); }
	}
	for (const k of [...g.objs.keys()]) {
		if (liveEnt.has(k)) continue;
		const o = g.objs.get(k);
		if (o) o.dying = true;
		else g.objs.delete(k);
	}
}

// Cumulative attention score for one attribute across all its edges.
function attrFreq(comp) {
	let s = 0;
	for (const e of S.graph.nodes.values()) if (e.component === comp) s += e.cum;
	return s;
}

// Total cumulative attention an object has received across all its edges.
function objTotal(entity) {
	let s = 0;
	for (const e of S.graph.nodes.values()) if (e.entity === entity) s += e.cum;
	return s;
}

// Gate a per-token (object, attribute) hit into the graph. An object ALREADY on
// the dial (or an existing edge) re-lights immediately; a NEWCOMER instead
// accrues "appearances" (distinct attended tokens) + weight and is only admitted
// once it clears a bar that grows with how full the outer ring is — so objects
// compete for the limited slots rather than every fleeting glance popping in.
function considerNode(x, i, now) {
	const g = S.graph;
	if (g.objs.has(x.entity) || g.nodes.has(x.key)) { spawnGraphNode(x, now); return; }
	const cand = g.cand.get(x.entity) || { score: 0, hits: 0, lastTok: -1 };
	if (i !== cand.lastTok) { cand.hits += 1; cand.lastTok = i; } // count DISTINCT tokens
	cand.score += x.weight;
	g.cand.set(x.entity, cand);
	const occ = MAX_OBJS ? g.objs.size / MAX_OBJS : 0;            // 0 (empty) … 1 (ring full)
	const needHits = Math.ceil(ENTER_HITS_MIN + ENTER_HITS_RANGE * occ);
	const needScore = ENTER_SCORE_MIN * (1 + ENTER_SCORE_COMP * occ);
	if (cand.hits >= needHits && cand.score >= needScore) {
		// Compete on the accumulated strength (not a single token) so eviction is fair.
		if (spawnGraphNode(x, now, cand.score)) g.cand.delete(x.entity);
	}
}

// Register one attended (object, attribute) as an edge object→attribute, adding
// its attribute to the inner ring and its object to the outer ring on first
// sight. A full link budget evicts the weakest edge; a full outer ring evicts the
// weakest object. `admitScore` (a newcomer's accumulated appearances weight, from
// considerNode) is what it competes with for a slot; cumulative score ranks "how
// much the model dwelt" on it. Returns true if the edge is now on the graph.
function spawnGraphNode(x, now, admitScore) {
	const g = S.graph;
	const strength = admitScore != null ? admitScore : x.score;
	let edge = g.nodes.get(x.key);
	if (!edge) {
		if (g.objs.has(x.entity)) {                                // spokes-per-object cap
			let cnt = 0; for (const e of g.nodes.values()) if (e.entity === x.entity) cnt++;
			if (cnt >= MAX_EDGES_PER_OBJ) return false;            // don't let one object hog links
		}
		if (!g.objs.has(x.entity) && g.objs.size >= MAX_OBJS) {   // outer ring full
			let wk = null, wc = Infinity;
			for (const id of g.objs.keys()) {
				const o = g.objs.get(id);
				if (o?.dying) continue;
				const tot = objTotal(id); if (tot < wc) { wc = tot; wk = id; }
			}
			if (wk == null || wc >= strength) return false;        // newcomer too weak for a slot
			for (const [k, e] of [...g.nodes]) if (e.entity === wk) g.nodes.delete(k);
			const wo = g.objs.get(wk);
			if (wo) wo.dying = true;
			else g.objs.delete(wk);
			pruneRings();
		}
		if (g.nodes.size >= MAX_EDGES) {                           // link budget full
			let mk = null, mc = Infinity;
			for (const [k, nd] of g.nodes) if (nd.cum < mc) { mc = nd.cum; mk = k; }
			if (mk == null || mc >= strength) return false;
			g.nodes.delete(mk); pruneRings();
		}
		ensureAttr(x.component, now);
		ensureObj(x.entity, x.component, x.kind, now);
		// Seed cum with the accumulated candidate strength so a just-admitted object
		// doesn't immediately look like the weakest and get evicted next token.
		edge = { key: x.key, entity: x.entity, component: x.component, act: 0, actTarget: 0, cum: Math.max(0, strength - x.score) };
		g.nodes.set(x.key, edge);
	}
	// Responsive: a hit pops the edge straight to its weight (no smoothing lag);
	// updateGraph then decays it fast and PRUNES it once the model looks away, so
	// nodes track the live attention instead of accumulating and persisting.
	edge.actTarget = Math.max(edge.actTarget || 0, x.weight);
	edge.act = Math.max(edge.act || 0, x.weight);
	edge.cum += x.score;
	g.pulses.push({ key: x.key, t: 0, w: x.weight });
	// a fresh hit nudges its endpoints — the dial breathes with attention traffic
	const attr = g.attrs.get(x.component), obj = g.objs.get(x.entity);
	const kick = 0.06 * x.weight;
	if (attr) attr.vel = (attr.vel || 0) + (Math.random() - 0.5) * kick;
	if (obj) obj.vel = (obj.vel || 0) + (Math.random() - 0.5) * kick * 1.2;
	return true;
}

// Inner-ring attribute: claim an even direction the first time it's consulted.
function ensureAttr(comp, now) {
	const g = S.graph;
	if (g.attrs.has(comp)) {
		const m = g.attrs.get(comp);
		if (m?.dying) { m.dying = false; m.spawn = Math.max(m.spawn || 0, 0.15); }
		return;
	}
	const home = assignCompAngle(comp);
	g.attrs.set(comp, {
		spawnT: now, spawn: 0, dying: false, phase: Math.random() * TAU,
		angle: home, home, vel: 0, radOff: 0, radVel: 0,
	});
}

// Outer-ring object: place it near the attribute it first arrived through, fanned
// around that direction so objects sharing an attribute don't stack on top.
function ensureObj(entity, comp, kind, now) {
	const g = S.graph;
	if (g.objs.has(entity)) {
		const o = g.objs.get(entity);
		if (o?.dying) { o.dying = false; o.spawn = Math.max(o.spawn || 0, 0.15); }
		return;
	}
	const base = g.compAngles.get(comp) ?? assignCompAngle(comp);
	let cnt = 0; for (const o of g.objs.values()) if (o.primary === comp && !o.dying) cnt++;
	const step = Math.min(0.4, 0.42 * neighborGap(comp));
	const off = cnt ? (cnt % 2 ? 1 : -1) * step * Math.ceil(cnt / 2) : 0;
	const home = base + off;
	g.objs.set(entity, {
		angle: home, home, primary: comp, kind,
		color: S.ctx.entityHex?.(kind, entity) || "#6bd96e",
		spawnT: now, spawn: 0, dying: false, phase: Math.random() * TAU, vel: 0, radOff: 0, radVel: 0,
	});
}

// Apply each inventory card's attention halo: how much the model has attended to
// THIS to-place object, accumulated up to the current token `i` (a per-target
// aggregate). Normalized to the step's peak total, so halos build up as the
// generation proceeds and settle into a "who got looked at most" ranking. Only
// on to-place-bearing steps; otherwise the halos stay off.
function updateCardAttn(i, dt) {
	if (!S.data.hasToPlace || !S.cards.length) return;
	const cum = S.data.tpCum, norm = S.data.tpMaxTotal || 1e-9;
	const decay = Math.exp(-dt / 0.32); // the per-token pulse fades fast between tokens
	for (const c of S.cards) {
		const recv = cum.get(c.id)?.[i] || 0;
		const w = clamp01(recv / norm);
		c.attn = w;
		c.pulse = (c.pulse || 0) * decay;
		c.node.style.setProperty("--attn", w > 0.01 ? w.toFixed(3) : "0");
		c.node.style.setProperty("--pulse", c.pulse > 0.01 ? c.pulse.toFixed(3) : "0");
	}
}

// --- attention graph (canvas): spawned (object, attribute) nodes -------------

// The dial only needs the hub center + radius; bead positions derive from their
// spoke angle + radial slot (resize-safe), so this just tracks the geometry. A
// small symmetric margin leaves room for the rim labels, which hug inward.
function layoutGraph() {
	const c = $("tfp-graph-canvas");
	const topPad = 14 * S.dpr; // floating header overlays the canvas top
	const cx = c.width / 2, cy = (c.height + topPad) / 2;
	S.graph.center = { x: cx, y: cy };
	S.graph.R = Math.max(20 * S.dpr, Math.min(cx, cy - topPad * 0.4) - 15 * S.dpr);
}

function updateGraph(dt) {
	const g = S.graph;
	// No smoothing / no playback-speed time-stretch: activation decays FAST and
	// an edge that goes quiet is deleted, so beads appear on attention and vanish
	// promptly after — a live, responsive dial rather than a persistent web.
	const targetDecay = Math.exp(-dt / 0.1);
	const actFollow = 1 - Math.exp(-dt / 0.07);
	const cumDecay = Math.exp(-dt / CUM_DECAY_TAU);
	const dead = [];
	for (const [k, n] of g.nodes) {
		n.actTarget = (n.actTarget || 0) * targetDecay;
		n.act = (n.act || 0) + ((n.actTarget || 0) - (n.act || 0)) * actFollow;
		n.cum = (n.cum || 0) * cumDecay; // dwell fades → eviction favours the currently-active
		// Prune early: once an edge goes even slightly quiet it's dropped, so nodes
		// demote and disappear promptly instead of lingering at a faint glow.
		if (n.act < 0.05 && n.actTarget < 0.05) dead.push(k);
	}
	for (const k of dead) g.nodes.delete(k);
	if (dead.length) pruneRings(); // orphaned attrs/objs → fade out (below)
	// Candidate objects (not yet admitted) decay so a scattered one-off glance
	// never slowly accretes into a slot — only sustained attention earns entry.
	const candDecay = Math.exp(-dt / CAND_DECAY_TAU);
	for (const [k, cd] of g.cand) {
		cd.score *= candDecay; cd.hits *= candDecay;
		if (cd.score < 0.02) g.cand.delete(k);
	}
	g.coreTarget = (g.coreTarget || 0) * Math.exp(-dt / 0.42);
	const coreFollow = 1 - Math.exp(-dt / 0.2);
	g.core = (g.core || 0) + ((g.coreTarget || 0) - (g.core || 0)) * coreFollow;
	const adv = dt / 0.42; // pulse travel — fixed & quick, never stalls at high speed
	for (const p of g.pulses) p.t += adv;
	g.pulses = g.pulses.filter((p) => p.t < 1);
	// spawn EMA: ease ring nodes in when they appear, out fast when they go quiet
	const spStep = 1 - Math.exp(-dt / 0.13);
	for (const [k, m] of [...g.attrs]) {
		if (m.dying) {
			m.spawn = Math.max(0, (m.spawn || 0) - (m.spawn || 0) * spStep);
			if (m.spawn < 0.02) { g.attrs.delete(k); g.compAngles.delete(k); }
		} else {
			m.spawn = (m.spawn || 0) + (1 - (m.spawn || 0)) * spStep;
		}
	}
	for (const [k, o] of [...g.objs]) {
		if (o.dying) {
			o.spawn = Math.max(0, (o.spawn || 0) - (o.spawn || 0) * spStep);
			if (o.spawn < 0.02) g.objs.delete(k);
		} else {
			o.spawn = (o.spawn || 0) + (1 - (o.spawn || 0)) * spStep;
		}
	}
	relaxGraph(dt);
}

// Shortest signed angular distance (–π…π).
function angleDiff(to, from) {
	let d = to - from;
	while (d > Math.PI) d -= TAU;
	while (d < -Math.PI) d += TAU;
	return d;
}

// Frequency-weighted elastic layout on each ring: high-attention nodes stay near
// their home slot; low-attention ones get pushed aside; linked objects drift
// toward their attributes. A little radial wobble keeps the rings from feeling
// mechanically perfect.
function relaxGraph(dt) {
	const g = S.graph;
	if (!g.attrs.size && !g.objs.size) return;
	const steps = S.reduced ? 1 : 2, h = dt / steps;
	for (let s = 0; s < steps; s++) {
		const attrList = [...g.attrs.entries()];
		let maxAF = 1e-9;
		for (const [c] of attrList) maxAF = Math.max(maxAF, attrFreq(c));

		for (const [comp, nd] of attrList) {
			if (nd.angle == null) { nd.angle = g.compAngles.get(comp) ?? nd.home ?? 0; nd.home ??= nd.angle; nd.vel ??= 0; }
			const freq = attrFreq(comp) / maxAF;
			const stiff = 1.4 + 4.2 * freq;
			let force = stiff * angleDiff(nd.home, nd.angle);

			for (const [comp2, nd2] of attrList) {
				if (comp2 === comp) continue;
				const da = angleDiff(nd2.angle, nd.angle);
				const f2 = attrFreq(comp2) / maxAF;
				const sep = 0.22 + 0.16 * (2 - freq - f2);
				if (Math.abs(da) < sep) {
					force -= Math.sign(da || 1) * (sep - Math.abs(da)) * RELAX_REP;
					nd.radOff = (nd.radOff || 0) + (sep - Math.abs(da)) * 0.22;
				}
			}

			nd.vel = (nd.vel || 0) * RELAX_DAMP + force * h;
			nd.angle += nd.vel * h;
			const maxDrift = 0.34 + 0.36 * (1 - freq);
			const drift = angleDiff(nd.angle, nd.home);
			if (Math.abs(drift) > maxDrift) { nd.angle = nd.home + Math.sign(drift) * maxDrift; nd.vel *= 0.42; }
			nd.radVel = (nd.radVel || 0) * RELAX_RAD_DAMP - (nd.radOff || 0) * 2.2;
			nd.radOff = Math.max(-0.32, Math.min(0.32, (nd.radOff || 0) + nd.radVel * h));
			g.compAngles.set(comp, nd.angle);
		}

		const objList = [...g.objs.entries()];
		let maxOF = 1e-9;
		for (const [id] of objList) maxOF = Math.max(maxOF, objTotal(id));

		for (const [ent, nd] of objList) {
			nd.home ??= nd.angle; nd.vel ??= 0;
			const freq = objTotal(ent) / maxOF;
			let force = (1.1 + 3.2 * freq) * angleDiff(nd.home, nd.angle);

			for (const [ent2, nd2] of objList) {
				if (ent2 === ent) continue;
				const da = angleDiff(nd2.angle, nd.angle);
				const f2 = objTotal(ent2) / maxOF;
				const sep = 0.18 + 0.14 * (2 - freq - f2);
				if (Math.abs(da) < sep) {
					force -= Math.sign(da || 1) * (sep - Math.abs(da)) * (RELAX_REP * 0.9);
					nd.radOff = (nd.radOff || 0) + (sep - Math.abs(da)) * 0.28;
				}
			}

			// pull toward the attributes this object is linked to (by edge weight)
			let pull = 0, wSum = 0;
			for (const e of g.nodes.values()) {
				if (e.entity !== ent) continue;
				const a = g.attrs.get(e.component);
				if (!a) continue;
				const w = e.cum + e.act * 2.5;
				pull += angleDiff(a.angle, nd.angle) * w;
				wSum += w;
			}
			if (wSum > 0) force += (pull / wSum) * 1.6;

			nd.vel = nd.vel * RELAX_DAMP + force * h;
			nd.angle += nd.vel * h;
			const maxDrift = 0.44 + 0.42 * (1 - freq);
			const drift = angleDiff(nd.angle, nd.home);
			if (Math.abs(drift) > maxDrift) { nd.angle = nd.home + Math.sign(drift) * maxDrift; nd.vel *= 0.42; }
			nd.radVel = (nd.radVel || 0) * RELAX_RAD_DAMP - (nd.radOff || 0) * 2.0;
			nd.radOff = Math.max(-0.38, Math.min(0.38, (nd.radOff || 0) + nd.radVel * h));
		}
	}
}

// Angular room around a spoke before it meets its nearest neighbour — lets a
// lone attribute's beads fan out to fill the dial instead of stacking in a line.
function neighborGap(comp) {
	const g = S.graph;
	if (g.compAngles.size <= 1) return TAU;
	const a0 = g.compAngles.get(comp);
	let best = TAU;
	for (const [c, a] of g.compAngles) {
		if (c === comp) continue;
		let d = Math.abs((a - a0) % TAU); d = Math.min(d, TAU - d);
		if (d < best) best = d;
	}
	return best;
}

// Spawn-in grow with a slight overshoot (easeOutBack), shared by both rings.
const easeOutBack = (s) => s < 1 ? 1 + 2.70158 * (s - 1) ** 3 + 1.70158 * (s - 1) ** 2 : 1;

// Quadratic-bezier control point — bows the segment slightly for soft-arc spokes.
function _beamCtrl(x0, y0, x1, y1, bow = 0.08) {
	return { x: (x0 + x1) / 2 - (y1 - y0) * bow, y: (y0 + y1) / 2 + (x1 - x0) * bow };
}
// Point at parameter t ∈ [0,1] along a quadratic from P0 → P1 through ctrl.
function _bezierPt(x0, y0, x1, y1, ctrl, t) {
	const u = 1 - t;
	return {
		x: u * u * x0 + 2 * u * t * ctrl.x + t * t * x1,
		y: u * u * y0 + 2 * u * t * ctrl.y + t * t * y1,
	};
}

// Collision-avoided label placer: skips a label that would leave the canvas or
// overlap one already drawn — the guarantee that keeps the dial legible. Draw
// the important labels (attribute names) first so they win ties over bead ids.
function makeLabeler(g, c, dpr) {
	const placed = [];
	return (text, ax, ay, align, font, fill) => {
		g.font = font; g.textAlign = align; g.textBaseline = "middle";
		const w = g.measureText(text).width, h = 12 * dpr;
		const x0 = align === "left" ? ax : align === "right" ? ax - w : ax - w / 2;
		const rect = { x0: x0 - 2 * dpr, y0: ay - h / 2, x1: x0 + w + 2 * dpr, y1: ay + h / 2 };
		if (rect.x0 < 2 * dpr || rect.x1 > c.width - 2 * dpr || rect.y0 < 2 * dpr || rect.y1 > c.height - 2 * dpr) return false;
		for (const r of placed) if (!(rect.x1 < r.x0 || rect.x0 > r.x1 || rect.y1 < r.y0 || rect.y0 > r.y1)) return false;
		placed.push(rect);
		g.shadowBlur = 3 * dpr; g.shadowColor = "rgba(0,0,0,0.9)";
		g.fillStyle = fill; g.fillText(text, ax, ay);
		g.shadowBlur = 0;
		return true;
	};
}
const prettyComp = (c) => String(c).replace(/_/g, " ");

function drawGraph(now) {
	const t = now / 1000;
	const c = $("tfp-graph-canvas");
	const g = c.getContext("2d");
	g.clearRect(0, 0, c.width, c.height);
	const gr = S.graph, { x: cx, y: cy } = gr.center, R = gr.R, dpr = S.dpr, breathOn = !S.reduced;
	const compC = (k) => S.ctx.compHex?.(k) || "#8ab4ff";

	// live activation of each attribute (inner) and object (outer), from their edges
	const attrAct = new Map(), objAct = new Map();
	for (const e of gr.nodes.values()) {
		attrAct.set(e.component, Math.max(attrAct.get(e.component) || 0, e.act));
		objAct.set(e.entity, Math.max(objAct.get(e.entity) || 0, e.act));
	}
	const aPos = (comp) => {
		const m = gr.attrs.get(comp), ang = m?.angle ?? gr.compAngles.get(comp) ?? -Math.PI / 2;
		const sp = easeOutBack(clamp01(m?.spawn ?? 0));
		const r = R * R_IN * (1 + (m?.radOff || 0) * 0.12) * sp;
		return { x: cx + r * Math.cos(ang), y: cy + r * Math.sin(ang), ang, sp: clamp01(m?.spawn ?? 0) };
	};
	const oPos = (ent) => {
		const o = gr.objs.get(ent), ang = o ? o.angle : -Math.PI / 2;
		const sp = easeOutBack(clamp01(o?.spawn ?? 0));
		const r = R * R_OUT * (1 + (o?.radOff || 0) * 0.1) * sp;
		return { x: cx + r * Math.cos(ang), y: cy + r * Math.sin(ang), ang, sp: clamp01(o?.spawn ?? 0) };
	};

	// --- backdrop: the two guide rings (inner = attributes, outer = objects) ---
	g.save();
	g.lineWidth = dpr;
	g.beginPath(); g.arc(cx, cy, R * R_IN, 0, TAU); g.strokeStyle = "rgba(150,168,214,0.11)"; g.stroke();
	g.beginPath(); g.arc(cx, cy, R * R_OUT, 0, TAU); g.strokeStyle = "rgba(150,168,214,0.06)"; g.stroke();
	g.restore();

	// --- additive glow layer: links, pulses, ring nodes, hub ---
	g.save();
	g.globalCompositeOperation = "lighter";
	// attribute → hub (curved inner spokes)
	for (const [comp] of gr.attrs) {
		const a = aPos(comp), act = attrAct.get(comp) || 0;
		const ctrl = _beamCtrl(cx, cy, a.x, a.y);
		g.beginPath(); g.moveTo(cx, cy); g.quadraticCurveTo(ctrl.x, ctrl.y, a.x, a.y);
		g.lineWidth = (0.6 + 1.8 * act) * dpr; g.strokeStyle = hexA(compC(comp), (0.08 + 0.4 * act) * a.sp); g.stroke();
	}
	// object → attribute (curved bipartite edges — the attention links)
	for (const e of gr.nodes.values()) {
		const a = aPos(e.component), o = oPos(e.entity);
		const ctrl = _beamCtrl(o.x, o.y, a.x, a.y, 0.06);
		g.beginPath(); g.moveTo(o.x, o.y); g.quadraticCurveTo(ctrl.x, ctrl.y, a.x, a.y);
		g.lineWidth = (0.5 + 1.6 * e.act) * dpr; g.strokeStyle = hexA(compC(e.component), (0.05 + 0.4 * e.act) * Math.min(a.sp, o.sp)); g.stroke();
	}
	// pulses stream object → attribute → hub along the same curved paths
	for (const pl of gr.pulses) {
		const e = gr.nodes.get(pl.key);
		if (!e) continue;
		const a = aPos(e.component), o = oPos(e.entity);
		let px, py;
		if (pl.t < 0.5) {
			const q = ease(pl.t / 0.5), ctrl = _beamCtrl(o.x, o.y, a.x, a.y, 0.06);
			({ x: px, y: py } = _bezierPt(o.x, o.y, a.x, a.y, ctrl, q));
		} else {
			const q = ease((pl.t - 0.5) / 0.5), ctrl = _beamCtrl(cx, cy, a.x, a.y);
			// traverse the hub→attribute spoke in reverse (attribute at t=1, hub at t=0)
			({ x: px, y: py } = _bezierPt(cx, cy, a.x, a.y, ctrl, 1 - q));
		}
		glowDot(g, px, py, (1.4 + 2.4 * pl.w) * dpr, compC(e.component), (0.3 + 0.7 * pl.t) * (0.5 + 0.5 * pl.w));
	}
	// outer ring: object beads (kind-colored)
	for (const [ent, o] of gr.objs) {
		const p = oPos(ent), act = objAct.get(ent) || 0, breath = breathOn ? 0.9 + 0.1 * Math.sin(t * Math.PI + o.phase) : 1;
		glowDot(g, p.x, p.y, (2.6 + 5 * act) * dpr * breath * p.sp, o.color, (0.34 + 0.66 * act) * p.sp);
	}
	// inner ring: attribute beads (attribute-colored)
	for (const [comp, m] of gr.attrs) {
		const p = aPos(comp), act = attrAct.get(comp) || 0, breath = breathOn ? 0.9 + 0.1 * Math.sin(t * Math.PI + m.phase) : 1;
		glowDot(g, p.x, p.y, (2.4 + 4.5 * act) * dpr * breath * p.sp, compC(comp), (0.42 + 0.58 * act) * p.sp);
	}
	// hub halo, pulsing with scene mass — kept soft so it never blows out to white
	const cb = breathOn ? 0.94 + 0.06 * Math.sin(t * Math.PI * 1.3) : 1;
	glowDot(g, cx, cy, (7 + 9 * gr.core) * dpr * cb, CORE_COLOR, 0.16 + 0.24 * gr.core, 2.4);
	g.restore();
	// crisp hub core on top
	g.beginPath(); g.arc(cx, cy, (2.6 + 2.2 * gr.core) * dpr * cb, 0, TAU);
	g.fillStyle = hexA(CORE_COLOR, 0.9); g.fill();

	// --- labels (collision-avoided): attributes (inner, grow outward) first,
	//     then objects (outer, grow inward) so both stay inside the dial ---
	const label = makeLabeler(g, c, dpr), nodeR = 7 * dpr;
	for (const [comp] of [...gr.attrs].sort((a, b) => (attrAct.get(b[0]) || 0) - (attrAct.get(a[0]) || 0))) {
		const p = aPos(comp);
		if (p.sp < 0.55) continue;
		const act = attrAct.get(comp) || 0, right = p.x >= cx;
		label(shortId(prettyComp(comp), 13), p.x + (right ? nodeR : -nodeR), p.y, right ? "left" : "right",
			`700 ${9 * dpr}px ui-sans-serif, system-ui, sans-serif`, hexA(compC(comp), 0.6 + 0.4 * act));
	}
	for (const [ent] of [...gr.objs].sort((a, b) => (objAct.get(b[0]) || 0) - (objAct.get(a[0]) || 0))) {
		const p = oPos(ent);
		if (p.sp < 0.6) continue;
		const act = objAct.get(ent) || 0, right = p.x >= cx;
		label(shortId(ent, 15), p.x + (right ? -nodeR : nodeR), p.y, right ? "right" : "left",
			`${9.5 * dpr}px ui-monospace, Menlo, monospace`, `rgba(222,226,234,${0.5 + 0.45 * act})`);
	}
}

// --- object inventory (the tray) ---------------------------------------------

// Build one tray card per emitted object. Its reveal span (reasoning length +
// its output_rel) lets us pop the card in exactly when the model writes that
// object out — in lockstep with the thought stream — and (on a placement step)
// fly it into the scene once its assignment is fully written.
function buildInventory() {
	const grid = $("tfp-inv-grid");
	grid.replaceChildren();
	S.cards = [];
	S.cardById = new Map();
	S.invCount = 0; // how many objects have populated the tray (decompose counter)
	const kindTally = {}; // which kind this step emits (zone/object/frame) → tray noun
	for (const m of S.ctx.outputMap || []) {
		const rel = m.output_rel;
		if (!Array.isArray(rel)) continue;
		const asset = (S.ctx.assets || {})[m.id] || {};
		const color = S.ctx.entityHex?.(asset.node_kind, m.id) || "#6bd96e";
		const kindLabel = S.ctx.entityKindLabel?.(asset.node_kind, m.id) || "object";
		kindTally[kindLabel] = (kindTally[kindLabel] || 0) + 1;
		// Start on a placeholder glyph; the real mesh render (or a still fallback)
		// swaps in once it loads.
		const thumb = el("div", { class: "tfp-card-thumb ph" }, placeholderGlyph(color));
		// The thumb + its meter live in a square "shot". The attention highlight is a
		// glow on the OBJECT'S OWN SILHOUETTE (a drop-shadow that follows the mesh's
		// alpha — see CSS), plus a quantitative meter, so it reads as "this object".
		const shot = el("div", { class: "tfp-card-shot" },
			thumb,
			el("div", { class: "tfp-card-meter" }, el("div", { class: "tfp-card-meter-fill" })), // quantitative bar
		);
		const node = el("div", { class: "tfp-card", title: `${m.id} · ${kindLabel}` },
			shot,
			el("div", { class: "tfp-card-id", text: shortId(m.id) }),
			el("span", { class: "tfp-card-dot", style: `background:${color};color:${color}` }),
		);
		node.dataset.state = "pending";
		node.style.setProperty("--attn", "0");
		node.style.setProperty("--pulse", "0");
		const card = {
			id: m.id, node, thumb, color, pngUrl: asset.pngUrl || null,
			canvasEl: null, assetReady: false, phase: Math.random() * Math.PI * 2, attn: 0, pulse: 0,
			start: S.reasonLen + rel[0], end: S.reasonLen + rel[1],
			origin: asset.origin, dimensions: asset.dimensions,
			canPlace: S.placing && Array.isArray(asset.origin) && Array.isArray(asset.dimensions),
			state: "pending", placedT: 0, flew: false, shown: false,
		};
		S.cards.push(card);
		S.cardById.set(m.id, card);
		loadThumb(card, asset.glb || []);
	}
	// The dominant emitted kind decides the tray noun (zones vs objects vs frames)
	// so a zone-decompose step reads "zones", not "objects".
	S.invKind = Object.entries(kindTally).sort((a, b) => b[1] - a[1])[0]?.[0] || null;
	// Placement (object_bbox_batch) shows the whole known batch upfront as ghost
	// slots that then fly out. Decompose INVENTS objects, so its tray starts empty
	// and each card is appended (populated) as the model emits it — see syncInventory.
	if (S.placing) for (const c of S.cards) { grid.appendChild(c.node); c.shown = true; }
	$("tfp-inv-title").textContent = S.placing ? `placing ${invNoun()}` : `inventing ${invNoun()}`;
	// A step that emits nothing (rare) shouldn't show an empty tray.
	$("tfp-inventory").style.display = S.cards.length ? "flex" : "none";
	updateInvSub();
}

function placeholderGlyph(color) {
	return el("div", { class: "tfp-ph-glyph", style: `--c:${color}` });
}

// Tray noun from the kind this step emits: zones (decompose/root bbox), frames
// (encapsulating shells), else objects. Driven by the step TEMPLATE (authoritative),
// falling back to the emitted cards' kind then "object".
function invNoun() {
	const kind = STEP_KIND[S.template] || S.invKind || "object";
	return kind === "zone" ? "zones" : kind === "frame" ? "frames" : "objects";
}

function updateInvSub() {
	const sub = $("tfp-inv-sub");
	if (S.placing) {
		const done = S.cards.filter((c) => c.state !== "pending").length;
		sub.classList.remove("counting");
		sub.textContent = `${done} / ${S.cards.length}`;
		return;
	}
	// Decompose: a live tally of objects invented into the zone being decomposed.
	// When we can float the badge over the zone in 3D, the number lives THERE
	// (a status indicator on the zone); otherwise it falls back into the header.
	const zone = shortId(S.ctx.step?.node ?? "");
	const total = S.zoneBase + S.invCount; // builds on what the zone already held (next_object etc.)
	if (S.zoneBadge) {
		sub.classList.remove("counting", "bump", "frames", "zones");
		sub.textContent = zone ? `in ${zone}` : "inventing";
		const cnt = $("tfp-zone-count");
		cnt.textContent = String(total);
		cnt.classList.remove("bump"); void cnt.offsetWidth; cnt.classList.add("bump");
	} else {
		const noun = invNoun();
		sub.classList.add("counting");
		sub.classList.toggle("frames", noun === "frames"); // blue for the encap frame count
		sub.classList.toggle("zones", noun === "zones"); // red for the zone count
		sub.textContent = zone ? `${total} in ${zone}` : `${total} ${noun}`;
		sub.classList.remove("bump"); void sub.offsetWidth; sub.classList.add("bump");
	}
}

// Float the zone status badge over the decomposed zone's top-center in 3D,
// tracking the (auto-orbiting) camera. Hidden when there's no zone or it's
// behind the camera.
function updateZoneBadge() {
	const zb = $("tfp-zone-badge");
	if (!S.zoneBadge) { if (!zb.hidden) zb.hidden = true; return; }
	const [ox, oy, oz] = S.zoneBadge.origin, [dx, dy, dz] = S.zoneBadge.dimensions;
	const scr = S.ctx.viewer?.project?.([ox + dx / 2, oy + dy, oz + dz / 2]);
	if (!scr || scr.behind) { zb.hidden = true; return; }
	zb.hidden = false;
	zb.style.left = `${scr.x}px`;
	zb.style.top = `${scr.y}px`;
	if (S.zonePlan) updateZonePlan();
}

// Parse a zone_plan's JSON output into prose + atomicity, plus the raw char span
// of the prose VALUE so streaming stays synced to when the model emits it. Handles
// BOTH shapes: nested zones use {"description": ...}, the ROOT uses {"plan": ...}
// — both may carry "is_atomic". Returns null when it isn't that shape (→ stream
// the raw text).
function parsePlanJson(raw) {
	let parsed;
	try { parsed = JSON.parse(raw); } catch { return null; }
	if (!parsed || typeof parsed !== "object") return null;
	const key = typeof parsed.description === "string" ? "description"
		: typeof parsed.plan === "string" ? "plan" : null;
	if (!key) return null;
	let dStart = -1, dEnd = -1;
	const keyIdx = raw.indexOf(`"${key}"`);
	if (keyIdx >= 0) {
		const colon = raw.indexOf(":", keyIdx + key.length + 2);
		const openQ = colon >= 0 ? raw.indexOf('"', colon + 1) : -1;
		if (openQ >= 0) {
			dStart = openQ + 1;
			for (let j = dStart; j < raw.length; j++) { // closing UNESCAPED quote
				if (raw[j] === "\\") { j++; continue; }
				if (raw[j] === '"') { dEnd = j; break; }
			}
		}
	}
	return { desc: parsed[key], hasAtomic: "is_atomic" in parsed, atomic: parsed.is_atomic === true, dStart, dEnd };
}

// Stream the zone's plan prose over the zone as the output phase reveals it —
// this REPLACES the "N objects" count badge for zone_plan steps (a plan emits
// text, not objects). The raw JSON is scripted away: only the `description`
// streams (synced to its span in the output), with a live caret, and an
// atomicity pill drops in once the description finishes.
function updateZonePlan() {
	const reveal = S.reveal || 0;
	const oo = S.planFromOutput
		? Math.max(0, Math.min(S.output.length, reveal - S.reasonLen))
		: Math.max(0, Math.min(S.reason.length, reveal));
	const node = $("tfp-zone-plan"), plan = S.plan;
	if (plan) {
		const span = plan.dEnd > plan.dStart ? plan.dEnd - plan.dStart : 0;
		const frac = span ? clamp01((oo - plan.dStart) / span) : clamp01(oo / Math.max(1, S.output.length));
		const shownLen = Math.round(frac * plan.desc.length);
		if (shownLen === S.zonePlanShown) return; // dedupe on shown chars, not oo, so
		S.zonePlanShown = shownLen;               // the atomic pill doesn't re-pop each frame
		const done = shownLen >= plan.desc.length;
		const kids = [el("span", { text: plan.desc.slice(0, shownLen) })];
		if (!done) kids.push(el("span", { class: "tfp-cursor" }));
		else if (plan.hasAtomic) kids.push(el("span", { class: `tfp-atomic ${plan.atomic ? "on" : "off"}`, text: plan.atomic ? "◆ atomic" : "subdivides" }));
		node.replaceChildren(...kids);
		return;
	}
	if (oo === S.zonePlanShown) return;
	S.zonePlanShown = oo;
	node.replaceChildren( // non-JSON plan: stream the raw text verbatim
		el("span", { text: S.planText.slice(0, oo) }),
		el("span", { class: "tfp-cursor" }),
	);
}

// Load the object's ACTUAL mesh into the viewer's offscreen thumbnailer, then
// render it live in the card. Zones / missing assets fall back to a still
// preview, then to the placeholder glyph — so a card always shows *something*.
function loadThumb(card, glbUrls) {
	const viewer = S.ctx.viewer;
	if (glbUrls.length && viewer?.loadThumbAsset) {
		viewer.loadThumbAsset(card.id, glbUrls)
			.then((ok) => { if (S && S.cards.includes(card)) (ok ? attachThumbCanvas(card) : fallbackThumb(card)); })
			.catch(() => { if (S && S.cards.includes(card)) fallbackThumb(card); });
	} else {
		fallbackThumb(card);
	}
}

function attachThumbCanvas(card) {
	const canvas = el("canvas", { class: "tfp-card-canvas", width: THUMB_PX, height: THUMB_PX });
	card.canvasEl = canvas;
	card.assetReady = true;
	card.thumb.classList.remove("ph");
	card.thumb.replaceChildren(canvas);
	S.ctx.viewer?.renderThumb?.(card.id, canvas, card.phase); // first frame immediately
}

function fallbackThumb(card) {
	if (!card.pngUrl) return; // keep the placeholder glyph
	const img = el("img", { src: card.pngUrl, alt: card.id });
	img.addEventListener("error", () => { img.remove(); card.thumb.classList.add("ph"); card.thumb.replaceChildren(placeholderGlyph(card.color)); });
	card.thumb.classList.remove("ph");
	card.thumb.replaceChildren(img);
}

// Spin the loaded meshes as a gentle turntable. Every ready card is refreshed
// each frame (up to a per-frame cap so the sync GL readbacks stay bounded); the
// yaw is derived from wall-clock, so a card that's skipped on a busy frame still
// resumes at the correct angle — no drift, no stutter.
function rotateThumbs(now) {
	if (S.reduced) return; // reduced-motion: the one-shot render on load is enough
	const ready = S.cards.filter((c) => c.assetReady && c.canvasEl && c.node.isConnected);
	if (!ready.length) return;
	const yaw = (now / 1000) * ROT_SPEED;
	const count = Math.min(ready.length, THUMB_MAX_PER_FRAME);
	for (let k = 0; k < count; k++) {
		S.rotCursor = (S.rotCursor + 1) % ready.length;
		const c = ready[S.rotCursor];
		S.ctx.viewer?.renderThumb?.(c.id, c.canvasEl, yaw + c.phase);
	}
}

// Advance every card to the phase implied by the current reveal, and (on a
// placement step) feed the resolved-box set + camera focus to the viewer.
//   pending  reveal hasn't reached the object's json yet (ghost slot)
//   emitted  its json is being / has been written (materialized in the tray)
//   placed   (placement steps only) its assignment is complete → flew into 3D
function syncInventory(now) {
	if (!S.cards.length) return;
	const reveal = S.reveal || 0;
	const items = [];
	let cx = 0, cy = 0, cz = 0, np = 0;
	let changed = false;
	for (const c of S.cards) {
		// Materialize a touch into the object's span so the card lands with the
		// first tokens of its definition, not only once it's fully written.
		const emitThresh = c.start + Math.max(1, (c.end - c.start) * 0.12);
		const emitted = reveal >= emitThresh;
		const done = reveal >= c.end;
		const want = !emitted ? "pending" : (done && c.canPlace ? "placed" : "emitted");
		if (want !== c.state) {
			if (want === "placed") {
				c.placedT = now;
				if (S.playing) spawnFly(c, now);   // forward playback → fly it in
				else { c.placedT = now - FLY_MS - GROW_MS; c.flew = false; } // scrub → snap grown
			} else {
				c.placedT = 0; c.flew = false;
			}
			// Decompose: the object is being INVENTED, so populate the tray with it
			// now (append) instead of un-greying a pre-shown ghost; drop it back off
			// if the playhead is scrubbed before its span.
			if (!S.placing) {
				if (want !== "pending" && !c.shown) { $("tfp-inv-grid").appendChild(c.node); c.shown = true; S.invCount++; }
				else if (want === "pending" && c.shown) { c.node.remove(); c.shown = false; S.invCount = Math.max(0, S.invCount - 1); }
			}
			c.state = want;
			c.node.dataset.state = want;
			changed = true;
		}
		if (c.state === "placed") {
			// The box grows only AFTER the thumbnail has "landed" (flew ? after
			// the flight : immediately), so it slots in where the asset arrives.
			const airborne = c.flew && (now - c.placedT) < FLY_MS;
			if (!airborne) {
				const p = clamp01((now - c.placedT - (c.flew ? FLY_MS : 0)) / GROW_MS);
				items.push({ id: c.id, origin: c.origin, dimensions: c.dimensions, progress: p, color: NEW_OBJECT_COLOR });
			}
			const [ox, oy, oz] = c.origin, [dx, dy, dz] = c.dimensions;
			cx += ox + dx / 2; cy += oy + dy / 2; cz += oz + dz / 2; np++;
		}
	}
	if (changed) updateInvSub();
	if (S.placing) {
		S.ctx.viewer?.setResolving?.(items);
		S.ctx.viewer?.setFocus?.(np ? [cx / np, cy / np, cz / np] : null);
	}
}

// Reset all placement state (on loop / when the clip restarts).
function resetPlacement() {
	for (const c of S.cards) {
		c.state = "pending"; c.node.dataset.state = "pending"; c.placedT = 0; c.flew = false;
		// Decompose empties its tray so it re-populates from scratch on replay.
		if (!S.placing && c.shown) { c.node.remove(); c.shown = false; }
	}
	if (!S.placing) S.invCount = 0;
	for (const f of S.flies) f.node.remove();
	S.flies = [];
	if (S.placing) { S.ctx.viewer?.setResolving?.([]); S.ctx.viewer?.setFocus?.(null); }
	// The graph is cumulative ("what it consulted"), so a full replay resets it.
	S.graph.nodes.clear(); S.graph.compAngles.clear(); S.graph.attrs.clear(); S.graph.objs.clear(); S.graph.cand.clear(); S.graph.pulses = []; S.graph.core = 0; S.graph.coreTarget = 0;
	updateInvSub();
}

// --- fly-into-scene manifest --------------------------------------------------

// A thumbnail clone that flies from the tray card to where its object sits in
// the 3D scene; the bbox then slots in where it lands (see syncInventory).
function spawnFly(card, now) {
	const from = card.thumb.getBoundingClientRect();
	if (!from.width) return; // tray not laid out → skip the flourish, box grows in place
	card.flew = true;
	const clone = el("div", { class: "tfp-fly-item" });
	const img = card.thumb.querySelector("img");
	if (card.assetReady && card.canvasEl) clone.appendChild(el("img", { src: card.canvasEl.toDataURL() })); // snapshot the mesh render
	else if (img) clone.appendChild(el("img", { src: img.src }));
	else { clone.classList.add("ph"); clone.appendChild(placeholderGlyph(card.color)); }
	clone.style.left = `${from.left}px`;
	clone.style.top = `${from.top}px`;
	clone.style.width = `${from.width}px`;
	clone.style.height = `${from.height}px`;
	$("tfp-fly").appendChild(clone);
	S.flies.push({ node: clone, card, t0: now, cxFrom: from.left + from.width / 2, cyFrom: from.top + from.height / 2 });
}

function updateFlies(now) {
	if (!S.flies.length) return;
	const keep = [];
	for (const f of S.flies) {
		const p = clamp01((now - f.t0) / FLY_MS);
		const e = easeOutCubic(p);
		const scr = S.ctx.viewer?.project?.(centerOf(f.card));
		const tx = scr && !scr.behind ? scr.x : f.cxFrom;
		const ty = scr && !scr.behind ? scr.y : f.cyFrom;
		const dx = (tx - f.cxFrom) * e, dy = (ty - f.cyFrom) * e;
		f.node.style.transform = `translate(${dx}px, ${dy}px) scale(${1 - 0.72 * e})`;
		f.node.style.opacity = String(1 - 0.85 * e);
		if (p < 1) keep.push(f); else f.node.remove();
	}
	S.flies = keep;
}

function centerOf(c) {
	return [c.origin[0] + c.dimensions[0] / 2, c.origin[1] + c.dimensions[1] / 2, c.origin[2] + c.dimensions[2] / 2];
}

// --- scene-mass EEG ----------------------------------------------------------

function drawEeg(cur) {
	const c = $("tfp-eeg");
	const g = c.getContext("2d");
	const w = c.width, h = c.height, n = S.data.frames.length, pad = 3 * S.dpr;
	g.clearRect(0, 0, w, h);
	const mm = S.data.massMax || 1;
	const X = (i) => (i / Math.max(1, n - 1)) * w;
	const Y = (v) => h - pad - clamp01(v / mm) * (h - 2 * pad);
	if (S.data.outStart > 0 && S.data.outStart < n) {
		g.fillStyle = "rgba(122,79,208,0.16)";
		g.fillRect(X(S.data.outStart), 0, w - X(S.data.outStart), h);
	}
	g.beginPath(); g.moveTo(0, h);
	S.data.frames.forEach((f, i) => g.lineTo(X(i), Y(f.mass)));
	g.lineTo(X(n - 1), h); g.closePath();
	g.fillStyle = "rgba(74,240,224,0.13)"; g.fill();
	g.beginPath();
	S.data.frames.forEach((f, i) => { const x = X(i), y = Y(f.mass); i ? g.lineTo(x, y) : g.moveTo(x, y); });
	g.strokeStyle = "#4af0e0"; g.lineWidth = 1.5 * S.dpr;
	g.shadowBlur = 8 * S.dpr; g.shadowColor = "#4af0e0"; g.stroke(); g.shadowBlur = 0;
	g.strokeStyle = "rgba(255,255,255,0.85)"; g.lineWidth = S.dpr;
	g.beginPath(); g.moveTo(X(cur), 0); g.lineTo(X(cur), h); g.stroke();
}

// --- thought stream ----------------------------------------------------------

// The stream renders the SERVER reconstruction (verbatim reasoning + output),
// revealed proportionally to the playhead — so it's always clean text (no
// per-token decode artifacts), tinted reasoning (blue) vs output (violet), with
// a caret + faint "ghost" of the upcoming text.
function buildStream() {
	// Reasoning: collapse newline runs for flowing prose. Output: kept VERBATIM
	// so the output_map's `output_rel` char spans line up for object placement.
	S.reason = String(S.ctx.reasoning ?? "").replace(/\s*\n+\s*/g, " ").trim();
	S.output = String(S.ctx.output ?? "");
	S.reasonLen = S.reason.length;
	S.total = S.reasonLen + S.output.length;
	// zone_plan streams its PLAN prose over the zone (see updateZonePlan): prefer
	// the output channel when it carries the plan, else fall back to reasoning.
	S.planFromOutput = !!S.output.trim();
	S.planText = S.planFromOutput ? S.output : S.reason;
	// zone_plan output is JSON ({description, is_atomic}); parse it so we can
	// stream only the prose description + flag atomicity, not the raw JSON.
	S.plan = S.zonePlan ? parsePlanJson(S.output) : null;
	S.zonePlanShown = -1;
	S.reveal = 0;
	S.revealed = -1;
	S.cursorEl = el("span", { class: "tfp-cursor" });
	updateStream();
}

function updateStream() {
	const reveal = S.reveal || 0;
	if (reveal === S.revealed) return;
	S.revealed = reveal;
	const rr = Math.max(0, Math.min(S.reasonLen, reveal));
	const oo = Math.max(0, Math.min(S.output.length, reveal - S.reasonLen));
	const kids = [];
	if (!S.outputOnly) { // output-only run: hide the reasoning channel entirely
		kids.push(el("span", { class: "tok reason", text: S.reason.slice(0, rr) }));
		if (oo <= 0) kids.push(S.cursorEl);
		kids.push(el("span", { class: "tok reason pending", text: S.reason.slice(rr) }));
	}
	if (S.zonePlan && S.plan) {
		// zone_plan output is JSON — show ONLY the prose plan in the stream (raw JSON
		// scripted away), synced to when the model actually writes the value.
		const plan = S.plan;
		const span = plan.dEnd > plan.dStart ? plan.dEnd - plan.dStart : 0;
		const frac = span ? clamp01((oo - plan.dStart) / span) : clamp01(oo / Math.max(1, S.output.length));
		const shownLen = Math.round(frac * plan.desc.length);
		kids.push(el("span", { class: "tok out", text: plan.desc.slice(0, shownLen) }));
		if (shownLen < plan.desc.length) kids.push(S.cursorEl);
		kids.push(el("span", { class: "tok out pending", text: plan.desc.slice(shownLen) }));
	} else {
		kids.push(el("span", { class: "tok out", text: S.output.slice(0, oo) }));
		if (oo > 0 || S.outputOnly) kids.push(S.cursorEl);
		kids.push(el("span", { class: "tok out pending", text: S.output.slice(oo) }));
	}
	const stream = $("tfp-stream");
	stream.replaceChildren(...kids);
	stream.scrollTop = Math.max(0, S.cursorEl.offsetTop - stream.clientHeight / 2);
}

// --- controls / transport ----------------------------------------------------

function wireControls() {
	$("tfp-play").onclick = () => setPlay(!S.playing);
	const speedSel = $("tfp-speed");
	speedSel.replaceChildren();
	for (const s of SPEEDS) {
		speedSel.appendChild(el("option", { value: String(s), text: `${s}×` }));
	}
	speedSel.onchange = () => setSpeed(Number(speedSel.value) || DEFAULT_SPEED);
	$("tfp-loop").onclick = () => { S.loop = !S.loop; $("tfp-loop").classList.toggle("on", S.loop); };
	$("tfp-hud").onclick = toggleHud;
	$("tfp-close").onclick = () => closePresent(); // (not `= closePresent` — the click event must not become `silent`)
	const scrub = $("tfp-scrub");
	// Suppress the auto-tracking write ONLY while the user is actively dragging
	// (a plain click keeps the range focused afterward, which would otherwise
	// freeze the bar — so don't gate on focus).
	scrub.onpointerdown = () => { S.scrubbing = true; };
	const endScrub = () => { S.scrubbing = false; };
	scrub.onpointerup = endScrub;
	scrub.onpointercancel = endScrub;
	scrub.onchange = endScrub;
	scrub.oninput = () => {
		const frac = Number(scrub.value) / 1000;
		seekTo(frac);
	};
}

// Seek to a whole-timeline fraction (0..1). In a stitched run the bar spans the
// ENTIRE summed timeline: map the global fraction → (step, local offset). Seeking
// within the current clip is instant (just move the playhead); landing on a
// different step hands off to the orchestrator to swap clips.
function seekTo(frac) {
	if (S.tl) {
		const tl = S.tl;
		const gTarget = Math.max(0, Math.min(tl.total - 1e-4, frac * tl.total));
		let k = 0;
		while (k < tl.offsets.length - 1 && gTarget >= tl.offsets[k] + tl.spans[k]) k++;
		const span = tl.spans[k] || 1;
		const localFrac = Math.max(0, Math.min(0.999, (gTarget - tl.offsets[k]) / span));
		if (k === S.sequence.index) {
			S.pos = S.startFrame + localFrac * (S.data.frames.length - S.startFrame);
			S.lastFired = -1;
		} else {
			S.sequence.onCross?.(k, localFrac); // cross-clip → rebuild at the target step
		}
		return;
	}
	S.pos = frac * S.data.frames.length; S.lastFired = -1;
}

function setPlay(on) { S.playing = on; $("tfp-play").textContent = on ? "❚❚" : "▶"; }
function setSpeed(v) {
	S.speed = v;
	const sel = $("tfp-speed");
	if (sel) sel.value = String(v);
}
function toggleHud() { S.hud = !S.hud; S.overlay.classList.toggle("hud-off", !S.hud); requestAnimationFrame(sizeCanvases); }

function updateScrub(i) {
	const scrub = $("tfp-scrub"), n = S.data.frames.length;
	if (S.tl) {
		// One unified, frame-weighted timeline across every clip: the global playhead
		// is this step's offset + how far we are into it, so the bar advances smoothly
		// and NEVER resets when stepping between clips.
		const tl = S.tl, k = S.sequence.index;
		const local = Math.max(0, S.pos - S.startFrame);
		const overall = ((tl.offsets[k] || 0) + local) / Math.max(1, tl.total);
		if (!S.scrubbing) scrub.value = String(Math.round(overall * 1000));
		$("tfp-pos").textContent = `${Math.round(overall * 100)}% · step ${k + 1}/${S.sequence.total}`;
		return;
	}
	if (!S.scrubbing) scrub.value = String(Math.round((S.pos / n) * 1000));
	$("tfp-pos").textContent = `${i + 1} / ${n}`;
}

function onKey(e) {
	if (!S) return;
	const k = e.key;
	if (k === "Escape") { e.preventDefault(); closePresent(); }
	else if (k === " ") { e.preventDefault(); setPlay(!S.playing); }
	else if (k === "h" || k === "H") { e.preventDefault(); toggleHud(); }
	else if (k === "ArrowLeft") { e.preventDefault(); setPlay(false); S.pos = Math.max(0, Math.floor(S.pos) - 1); S.lastFired = -1; }
	else if (k === "ArrowRight") { e.preventDefault(); setPlay(false); S.pos = Math.min(S.data.frames.length - 0.001, Math.floor(S.pos) + 1); S.lastFired = -1; }
}

// --- sizing / helpers --------------------------------------------------------

function sizeCanvases() {
	if (!S) return;
	S.dpr = Math.min(2, window.devicePixelRatio || 1);
	for (const id of ["tfp-eeg", "tfp-graph-canvas"]) {
		const cv = $(id), r = cv.getBoundingClientRect();
		cv.width = Math.max(1, Math.round(r.width * S.dpr));
		cv.height = Math.max(1, Math.round(r.height * S.dpr));
	}
	layoutGraph();
}

function glowDot(g, x, y, r, color, alpha, blur = 3) {
	const rad = Math.max(0.5, r);
	g.beginPath();
	g.arc(x, y, rad, 0, Math.PI * 2);
	g.shadowBlur = rad * blur;
	g.shadowColor = color;
	g.fillStyle = hexA(color, clamp01(alpha));
	g.fill();
	g.shadowBlur = 0;
}

function hexA(hex, a) {
	let h = String(hex).replace("#", "");
	if (h.length === 3) h = h.split("").map((c) => c + c).join("");
	const n = parseInt(h, 16) || 0;
	return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${clamp01(a)})`;
}

function shortId(id, max = 16) {
	const s = String(id);
	return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}
