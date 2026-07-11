// Attention-ablation setup wizard. "⚗ ablation…" opens this panel.
//
// An ablation is anchored to an existing BASE run. It forks the base at the
// last-N firings of the swept step kinds (one fork per firing), applies a
// treatment, and re-infers only the treated tail — the committed prefix is
// shared (replayed from the base, meshes hardlinked), so common segments are
// stored/computed once and each fork only diverges where needed. Variants are
// auto-named runs in the `__abl-` namespace (hidden from the dashboard run
// picker) and reuse the existing branch → save_run_from_branches → resume →
// attention endpoints (no new harness). Fire them INDIVIDUALLY (▶ per variant)
// or all at once.
//
// DISENTANGLED BY DESIGN: a launch varies exactly ONE experimentation axis
// (coordinate frame / scene-order / xml / attend / distractors), pinning every
// other axis at baseline. So two experiments (e.g. coordinate vs scene-order)
// never cross-multiply into A×D permutations — each is its own launch, its own
// wizard page. Within the coordinate axis the per-kind level set is resolved by
// ablationcore.expandExperimentForKind (bbox steps vary input+output; every other
// step varies input only, its modes collapsed so nothing is double-counted).

import { api } from "./api.js";
import { state } from "./state.js";
import { el, field, openModal, toast } from "./ui.js";
import {
	SHUFFLE_METHODS, METHOD_HINT, hasSceneContext, ALLOWED_MODEL, preferredModel,
	MAX_ABLATION_BATCH, expandExperimentForKind, EXPERIMENT_AXES, COORD_MODES, SCHEMA_MODES, GRAVITY_MODES,
	variantName, legacyVariantName, loadBaseTemplates, launchVariant, clearTemplateCache,
} from "./ablationcore.js";

export async function openAblationWizard() {
	const runNames = (state.runs || []).map((r) => r.name).filter((n) => !n.includes("__abl-"));
	if (!runNames.length) {
		toast("no runs yet — create a base run first", "err");
		return;
	}
	// Ablations run only on open-source models (gemma / qwen) — closed models like
	// gemini-flash are never offered, so a sweep can't fire them.
	const models = (state.models || []).filter(ALLOWED_MODEL);
	const slots = (state.slots || []).map((s) => s.id);
	if (!models.length) {
		toast("no open-source (gemma / qwen) models available for ablation", "err");
		return;
	}

	const cfg = {
		baseRun: state.run && runNames.includes(state.run) ? state.run : runNames[0],
		label: "",
		slot: (state.slot && slots.includes(state.slot)) ? state.slot : (slots[0] || ""),
		model: preferredModel(models) || models[0] || "",
		lastN: 3,
		kinds: new Set(), // included step kinds (populated after fetch = all fired kinds except root)
		// The SINGLE experimentation axis this launch varies. Everything else is
		// pinned at baseline (see ablationcore.BASELINE_TREATMENT).
		experiment: "coordinate",
		// Per-axis level selections (only the active axis's is used):
		coordModes: new Set(["lg2g", "l2l", "g2g", "g2l"]), // baseline auto-included
		schemaModes: new Set(["xml", "prose"]), // baseline (soft-JSON) = base cell, not launched
		gravityModes: new Set(["none", "q1", "q2", "q3", "q4"]), // all launched (base cell keeps the real VII tag)
		methods: new Set(["random", "distance", "raytrace"]), // order = baseline, auto-included
		attendTargets: [],
		distractors: [],
		seed: 1,
	};
	let baseSteps = []; // the base cell's full step list (event_index + template)
	let kindCounts = {}; // step kind -> firing count
	let allKinds = []; // fired kinds except root, in first-seen order

	const STEP_TITLES = ["base run + cell", "sweep — kinds × last-N", "experiment axis", "axis levels", "review & launch"];
	const NSTEPS = STEP_TITLES.length;
	const bodyHost = el("div", {});
	const stepLabel = el("div", { class: "m-hint" });
	const backBtn = el("button", { text: "‹ back" });
	const nextBtn = el("button", { class: "primary", text: "next ›" });
	const launchBtn = el("button", { class: "primary", text: `⚗ launch batch (≤${MAX_ABLATION_BATCH})` });
	let step = 0;

	// Level arrays for the active axis, fed to expandExperimentForKind.
	function expArgs() {
		return {
			methods: [...cfg.methods],
			coordModes: [...cfg.coordModes],
			schemaModes: [...cfg.schemaModes],
			gravityModes: [...cfg.gravityModes],
			attendTargets: cfg.attendTargets.slice(),
			distractors: cfg.distractors.slice(),
			seed: cfg.seed,
		};
	}

	// Firing event-index cuts per kind (the last-N firings — fewer if incomplete).
	// GATE: skip firings whose scene context is EMPTY (has_scene=false) — a
	// scene-context ablation on them is a no-op.
	function cutsForKind(k) {
		const firings = baseSteps.filter((s) => (s.template ?? s.step) === k && s.has_scene !== false).map((s) => s.event_index);
		return firings.slice(-Math.max(1, cfg.lastN));
	}

	// Kinds actually forked for the active experiment = the checked scene-context
	// kinds. Coordinate no longer drops non-bbox kinds: they vary along the INPUT
	// axis (which coordinates the step is SHOWN). Each kind's DISTINCT level set —
	// bbox by input+output, non-bbox by input only — is resolved per-kind in
	// buildVariants via expandExperimentForKind, so nothing is double-counted.
	function activeKinds() {
		return allKinds.filter((k) => cfg.kinds.has(k));
	}

	// The full variant set: forked kinds × their last-N firing cuts × the
	// treatments DISTINCT for that kind. For the coordinate axis the per-kind level
	// set differs (bbox: all selected modes; non-bbox: collapsed to distinct input
	// reps), so treatments are resolved INSIDE the kind loop, never once for all.
	function buildVariants() {
		const args = expArgs();
		const out = [];
		const seen = new Set();
		for (const k of activeKinds()) {
			const ts = expandExperimentForKind(cfg.experiment, args, k);
			for (const cut of cutsForKind(k)) {
				for (const t of ts) {
					const name = variantName({ baseRun: cfg.baseRun, slot: cfg.slot, model: cfg.model, label: cfg.label, kind: k, cut, tag: t.tag });
					if (seen.has(name)) continue;
					seen.add(name);
					out.push({ name, kind: k, cut, ...t });
				}
			}
		}
		return out;
	}

	openModal("ablation setup", (close, setError) => {
		const runSel = () =>
			el("select", { onchange: (e) => { cfg.baseRun = e.target.value; clearTemplateCache(e.target.value); } },
				runNames.map((n) => el("option", { value: n, text: n, ...(n === cfg.baseRun ? { selected: "" } : {}) })));
		const slotSel = () =>
			el("select", { onchange: (e) => { cfg.slot = e.target.value; } },
				slots.map((s) => el("option", { value: s, text: s, ...(s === cfg.slot ? { selected: "" } : {}) })));
		const modelSel = () =>
			el("select", { onchange: (e) => { cfg.model = e.target.value; } },
				models.map((m) => el("option", { value: m, text: m, ...(m === cfg.model ? { selected: "" } : {}) })));

		function renderBase() {
			const ver = state.runs?.find((r) => r.name === cfg.baseRun)?.prompt_version;
			const labelInput = el("input", { type: "text", value: cfg.label, placeholder: "e.g. coord-sweep",
				onchange: (e) => { cfg.label = e.target.value.trim(); } });
			return [
				el("div", { class: "m-hint", text: "The ablation inherits this base cell's prompts + committed scene; every variant shares the prefix and only its treated tail re-runs." }),
				field("ablation label (optional)", labelInput),
				field("base run", runSel()),
				ver ? el("div", { class: "m-hint", text: `prompt version: ${ver}` }) : null,
				field("slot (scene)", slotSel()),
				field("model", modelSel()),
				el("div", { class: "m-hint", text: "Variant runs are auto-named and hidden from the run picker — they signify the base they branch off from, the kind@firing they treat, and your label." }),
			];
		}

		function renderSweep() {
			if (!allKinds.length) {
				return [el("div", { class: "m-hint", text: "No step kinds found for this cell. Go back and pick a base cell that has run." })];
			}
			const lastNInput = el("input", { type: "number", min: "1", max: "10", value: String(cfg.lastN),
				onchange: (e) => { cfg.lastN = Math.max(1, Number(e.target.value) || 3); render(); } });
			const kindGrid = el("div", { class: "check-grid" },
				allKinds.map((k) => el("label", { title: `fires ${kindCounts[k]} time(s)` },
					el("input", { type: "checkbox", value: k, ...(cfg.kinds.has(k) ? { checked: "" } : {}),
						onchange: (e) => { e.target.checked ? cfg.kinds.add(k) : cfg.kinds.delete(k); render(); } }),
					`${k} (${kindCounts[k]})`)));
			const forks = allKinds.filter((k) => cfg.kinds.has(k)).reduce((n, k) => n + cutsForKind(k).length, 0);
			return [
				el("div", { class: "m-hint", text: `The sweep forks the base at the last ${cfg.lastN} firing(s) of each checked kind (only scene-context kinds are offered — root plans, overall_bbox, and image_prompt are excluded). A kind with fewer firings treats what it has.` }),
				field("last N firings per kind", lastNInput),
				el("div", { class: "m-field" }, el("span", { text: "step kinds to sweep" }), kindGrid),
				el("div", { class: "m-hint", text: `${forks} fork point${forks === 1 ? "" : "s"} across the checked kinds (× this experiment's levels = the variant count). For a coordinate experiment the bbox solvers vary by input AND output frame, while every other kind varies by input only (its levels collapse — see the axis page).` }),
			];
		}

		// STEP 3: pick the SINGLE axis this launch varies (radio → unambiguous).
		function renderExperiment() {
			const list = el("div", { class: "check-grid", style: "grid-template-columns:1fr" },
				EXPERIMENT_AXES.map((ax) =>
					el("label", { title: ax.hint, style: "display:flex;gap:8px;align-items:flex-start;padding:3px 0" },
						el("input", { type: "radio", name: "abl-experiment", value: ax.id, ...(cfg.experiment === ax.id ? { checked: "" } : {}),
							onchange: (e) => { if (e.target.checked) { cfg.experiment = ax.id; render(); } } }),
						el("span", {}, el("span", { style: "font-weight:600", text: ax.label }), el("span", { class: "muted", style: "margin-left:6px;font-size:12px", text: ax.hint })))));
			return [
				el("div", { class: "m-hint", text: "Pick ONE axis to vary. Every other axis stays at baseline (insertion-order, xml-on, L/G→L coordinates, no distractors, no directive), so this experiment can't cross-multiply with another. Run separate launches for separate axes." }),
				el("div", { class: "m-field" }, el("span", { text: "experimentation axis" }), list),
			];
		}

		// STEP 4: the dedicated page for the chosen axis's levels.
		function renderAxisConfig() {
			const anchor = el("div", { class: "m-hint", text: "The baseline (order · xml-on · L/G→L) is always included as the comparison anchor — you're choosing which NON-baseline levels to add." });
			if (cfg.experiment === "coordinate") {
				const grid = el("div", { class: "check-grid", style: "grid-template-columns:1fr" },
					COORD_MODES.filter((c) => c.id !== "baseline").map((c) =>
						el("label", { title: c.hint, style: "display:flex;gap:8px;align-items:flex-start;padding:3px 0" },
							el("input", { type: "checkbox", value: c.id, ...(cfg.coordModes.has(c.id) ? { checked: "" } : {}),
								onchange: (e) => { e.target.checked ? cfg.coordModes.add(c.id) : cfg.coordModes.delete(c.id); render(); } }),
							el("span", {}, el("span", { style: "font-weight:600", text: c.label }), el("span", { class: "muted", style: "margin-left:6px;font-size:12px", text: c.hint })))));
				return [
					el("div", { class: "m-hint", text: "Coordinate frame — one enumerated axis (L=local, G=global; the L→G case is intentionally excluded). Each level controls how coordinates are SHOWN in scene context (the INPUT, on EVERY treated step) and, for the two bbox solvers, which frame they must EMIT (the OUTPUT)." }),
					el("div", { class: "m-hint", text: "The L/G→L baseline is the base cell itself — already computed, NOT re-forked or recomputed — so pick only the non-baseline conditions to launch; /tf compares them against the base." }),
					el("div", { class: "m-field" }, el("span", { text: "coordinate conditions" }), grid),
					el("div", { class: "m-hint", text: "The bbox solvers (object_bbox_batch / child_bbox_batch) vary along ALL selected levels. Non-bbox kinds emit no coordinates, so the OUTPUT frame is invisible to them — their levels collapse onto the distinct INPUT reps (L/G→L ≡ L/G→G, and G→G ≡ G→L there), launched once per local / global input. So an input-only step is never double-counted." }),
				];
			}
			if (cfg.experiment === "schema") {
				const grid = el("div", { class: "check-grid", style: "grid-template-columns:1fr" },
					SCHEMA_MODES.filter((s) => s.id !== "baseline").map((s) =>
						el("label", { title: s.hint, style: "display:flex;gap:8px;align-items:flex-start;padding:3px 0" },
							el("input", { type: "checkbox", value: s.id, ...(cfg.schemaModes.has(s.id) ? { checked: "" } : {}),
								onchange: (e) => { e.target.checked ? cfg.schemaModes.add(s.id) : cfg.schemaModes.delete(s.id); render(); } }),
							el("span", {}, el("span", { style: "font-weight:600", text: s.label }), el("span", { class: "muted", style: "margin-left:6px;font-size:12px", text: s.hint })))));
				return [
					el("div", { class: "m-hint", text: "Scene-context schema — re-render the treated step's scene context in another SERIALIZATION (same information, different structure), on EVERY scene-context step. The bbox OUTPUT stays JSON; only the input format changes." }),
					el("div", { class: "m-hint", text: "The soft-JSON baseline is the base cell itself — already computed, NOT re-forked or recomputed — so pick only the non-baseline formats to launch; /tf compares them against the base." }),
					el("div", { class: "m-field" }, el("span", { text: "scene-context formats" }), grid),
				];
			}
			if (cfg.experiment === "gravity") {
				const grid = el("div", { class: "check-grid", style: "grid-template-columns:1fr" },
					GRAVITY_MODES.map((g) =>
						el("label", { title: g.hint, style: "display:flex;gap:8px;align-items:flex-start;padding:3px 0" },
							el("input", { type: "checkbox", value: g.id, ...(cfg.gravityModes.has(g.id) ? { checked: "" } : {}),
								onchange: (e) => { e.target.checked ? cfg.gravityModes.add(g.id) : cfg.gravityModes.delete(g.id); render(); } }),
							el("span", {}, el("span", { style: "font-weight:600", text: g.label }), el("span", { class: "muted", style: "margin-left:6px;font-size:12px", text: g.hint })))));
				return [
					el("div", { class: "m-hint", text: "XML gravity check — strip the treated step's instruction-block tags, then slide a neutral closing </prompt> across its word-count QUARTERS (opening <prompt> fixed at the block start). Measures how a tag pulls attention to the surrounding text as a function of distance." }),
					el("div", { class: "m-hint", text: "EVERY level here is launched — the untouched base cell keeps the real, semantically-loaded VERY_IMPORTANT tags, so it is NOT the anchor. `no tags` is the comparison baseline the graph subtracts to isolate the tag's pull." }),
					el("div", { class: "m-field" }, el("span", { text: "tag positions to launch" }), grid),
				];
			}
			if (cfg.experiment === "shuffle") {
				const grid = el("div", { class: "check-grid" },
					SHUFFLE_METHODS.filter((m) => m !== "order").map((m) =>
						el("label", { title: METHOD_HINT[m] },
							el("input", { type: "checkbox", value: m, ...(cfg.methods.has(m) ? { checked: "" } : {}),
								onchange: (e) => { e.target.checked ? cfg.methods.add(m) : cfg.methods.delete(m); render(); } }),
							m)));
				const seedInput = el("input", { type: "number", value: String(cfg.seed), onchange: (e) => { cfg.seed = Number(e.target.value) || 0; } });
				return [
					el("div", { class: "m-hint", text: "Scene-context order — reorder the scene items shown to the treated step. `order` is the baseline; add the reorderings to compare against it." }),
					anchor,
					el("div", { class: "m-field" }, el("span", { text: "reorderings" }), grid),
					field("RNG seed (random shuffle)", seedInput),
				];
			}
			if (cfg.experiment === "xml") {
				return [
					el("div", { class: "m-hint", text: "XML tags — a fixed two-level axis: baseline (tags kept) vs stripped. Stripping drops lines that are only an open/close tag from the treated step's prompt (applied as a launch-time snapshot override)." }),
					anchor,
					el("div", { class: "m-hint", text: "No levels to configure — this experiment launches exactly {baseline, xml-off}." }),
				];
			}
			if (cfg.experiment === "attend") {
				const ta = el("textarea", { rows: "4", placeholder: "one attention target per line, e.g.\nthe floor plan\nthe adjacent kitchen",
					style: "width:100%;font:12px ui-monospace,Menlo,monospace",
					onchange: (e) => { cfg.attendTargets = String(e.target.value).split("\n").map((s) => s.trim()).filter(Boolean); } });
				ta.value = cfg.attendTargets.join("\n");
				return [
					el("div", { class: "m-hint", text: "Attention steering — append an explicit 'focus on X' directive to the treated step's user prompt (a launch-time snapshot override). Baseline = no directive." }),
					anchor,
					el("div", { class: "m-field" }, el("span", { text: "attention targets (one per line)" }), ta),
				];
			}
			// distractors
			const distInput = el("input", { type: "text", value: cfg.distractors.join(","), placeholder: "2,4",
				onchange: (e) => { cfg.distractors = String(e.target.value).split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => Number.isFinite(n) && n > 0); } });
			const seedInput = el("input", { type: "number", value: String(cfg.seed), onchange: (e) => { cfg.seed = Number(e.target.value) || 0; } });
			return [
				el("div", { class: "m-hint", text: "Distractors — inject N irrelevant objects into the treated step's scene context. NOTE: not yet wired server-side (placeholder axis); levels are recorded in the run name only." }),
				anchor,
				field("distractor counts (comma-separated)", distInput),
				field("RNG seed (distractor pick)", seedInput),
			];
		}

		// --- launch (shared by ▶ per-variant and "launch all") ------------------
		let reviewRows = new Map(); // variant name -> { setStatus, button }

		async function ensureBaseTemplates() {
			try { return await loadBaseTemplates(cfg.baseRun); }
			catch (e) { setError(`couldn't load the base prompts: ${e.message}`); return null; }
		}

		async function launchOne(v) {
			// Skip if this exact config already exists under EITHER the new
			// (slot+model) name or the legacy name — the wizard used to rely on the
			// server 409 to skip existing, but the new name won't 409 against a
			// legacy run, so check explicitly (avoids a duplicate + a double-counted
			// replicate in the stats).
			const known = new Set((state.runs || []).map((r) => r.name));
			const newName = variantName({ baseRun: cfg.baseRun, slot: cfg.slot, model: cfg.model, label: cfg.label, kind: v.kind, cut: v.cut, tag: v.tag });
			const legName = legacyVariantName({ baseRun: cfg.baseRun, label: cfg.label, kind: v.kind, cut: v.cut, tag: v.tag });
			if (known.has(newName) || known.has(legName)) return { ok: false, msg: "already exists", name: newName };
			const tmpls = await ensureBaseTemplates();
			if (!tmpls) return { ok: false, msg: "no base prompts" };
			return launchVariant({
				baseRun: cfg.baseRun, slot: cfg.slot, model: cfg.model, label: cfg.label, lastN: cfg.lastN,
				kind: v.kind, cut: v.cut, method: v.method, xml: v.xml, distractors: v.distractors,
				seed: v.seed, attend: v.attend, coord: v.coord, schema: v.schema, gravity: v.gravity, tag: v.tag,
			}, tmpls);
		}

		function renderReview() {
			const variants = buildVariants();
			reviewRows = new Map();
			const axLabel = (EXPERIMENT_AXES.find((a) => a.id === cfg.experiment) || {}).label || cfg.experiment;
			const list = el("div", { style: "max-height:240px;overflow:auto;border:1px solid var(--line,#333);border-radius:6px;padding:6px 8px" });
			if (!variants.length) {
				list.appendChild(el("div", { class: "m-hint", text: "no variants — check at least one kind and one non-baseline level for this axis" }));
			} else {
				let curKind = null;
				for (const v of variants) {
					if (v.kind !== curKind) {
						curKind = v.kind;
						list.appendChild(el("div", { style: "margin:6px 0 2px;font-size:11px;color:var(--text-dim,#8a8f99);text-transform:uppercase;letter-spacing:0.04em", text: v.kind }));
					}
					const status = el("span", { class: "muted", style: "margin-left:auto;font-size:11px;white-space:nowrap" });
					const btn = el("button", { text: "▶", title: "launch just this variant",
						onclick: async () => {
							btn.disabled = true;
							status.textContent = "creating…";
							const res = await launchOne(v);
							status.textContent = res.ok ? "✓ launched" : `✗ ${res.msg}`;
							if (!res.ok) btn.disabled = false;
							toast(res.ok ? `launched ${v.name}` : `failed: ${res.msg}`, res.ok ? "ok" : "err");
						} });
					const row = el("div", { style: "display:flex;align-items:center;gap:8px;padding:1px 0;font:12px ui-monospace,Menlo,monospace",
						title: `kind=${v.kind} · cut=@${v.cut} · coord=${v.coord} · schema=${v.schema || "baseline"} · gravity=${v.gravity || "baseline"} · method=${v.method} · xml=${v.xml} · attend=${v.attend || "-"}` },
						btn, el("span", { style: "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1", text: `@${v.cut} · ${v.tag}` }), status);
					reviewRows.set(v.name, { setStatus: (t) => { status.textContent = t; }, button: btn, status });
					list.appendChild(row);
				}
			}
			return [
				el("div", { class: "m-hint", text: `${cfg.baseRun} · ${cfg.slot} · ${cfg.model}${cfg.label ? ` · ${cfg.label}` : ""} — experiment: ${axLabel}` }),
				el("div", { class: "m-field" }, el("span", { text: `${variants.length} variant run${variants.length === 1 ? "" : "s"} — ▶ launches one; “launch batch” fires up to ${MAX_ABLATION_BATCH} at a time (throttled to avoid a request flood). Launch again for more.` }), list),
				el("div", { class: "m-hint", text: "Each variant = fork the base at its kind@cut ▸ save as this run (prefix shared) ▸ resume to re-infer the treated step (LLM; meshes skipped). Compute + compare attention in the /tf ablation view." }),
			];
		}

		const RENDERERS = [renderBase, renderSweep, renderExperiment, renderAxisConfig, renderReview];

		function render() {
			setError("");
			stepLabel.textContent = `step ${step + 1} / ${NSTEPS} — ${STEP_TITLES[step]}`;
			bodyHost.replaceChildren(...RENDERERS[step]().filter(Boolean));
			backBtn.style.display = step > 0 ? "" : "none";
			nextBtn.style.display = step < NSTEPS - 1 ? "" : "none";
			launchBtn.style.display = step === NSTEPS - 1 ? "" : "none";
		}

		backBtn.onclick = () => { if (step > 0) { step -= 1; render(); } };

		nextBtn.onclick = async () => {
			if (step === 0) {
				if (!cfg.baseRun || !cfg.slot || !cfg.model) { setError("pick a base run, slot, and model"); return; }
				nextBtn.disabled = true;
				try {
					const resp = await api.tfSteps(cfg.baseRun, cfg.slot, cfg.model);
					baseSteps = resp.steps || [];
					kindCounts = {};
					// GATE: only scene-context kinds with a NON-EMPTY firing are offered
					// (root plans / overall_bbox / image_prompt and empty-scene firings
					// are excluded — a scene-context ablation there is a no-op).
					for (const s of baseSteps) { const k = s.template ?? s.step; if (k && s.has_scene !== false) kindCounts[k] = (kindCounts[k] || 0) + 1; }
					allKinds = [...new Set(baseSteps.filter((s) => s.has_scene !== false).map((s) => s.template ?? s.step).filter(Boolean))].filter(hasSceneContext);
					cfg.kinds = new Set(allKinds); // default: sweep every scene-context kind (coordinate varies bbox by input+output, other kinds by input only)
				} catch (e) {
					setError(`couldn't load steps for this cell: ${e.message}`);
					nextBtn.disabled = false;
					return;
				}
				nextBtn.disabled = false;
				if (!allKinds.length) { setError("this cell has no scene-context step kinds (root plans / overall_bbox / image_prompt don't count) — pick a base cell that has run"); return; }
			} else if (step === 1) {
				if (!cfg.kinds.size) { setError("check at least one step kind to sweep"); return; }
			} else if (step === 3) {
				if (!buildVariants().length) { setError("select at least one non-baseline level (and a compatible kind) to vary for this experiment"); return; }
			}
			step += 1;
			render();
		};

		// "launch batch": run up to MAX_ABLATION_BATCH sequentially, holding the
		// rest back so a sweep can't flood the LLM transport (which caused mass
		// transport retries + results never landing). Launch again for more.
		launchBtn.onclick = async () => {
			const variants = buildVariants();
			if (!variants.length) { setError("nothing to launch — check the axis levels"); return; }
			// Prefer variants that aren't already launched (skip ✓-marked rows).
			const pending = variants.filter((v) => reviewRows.get(v.name)?.status?.textContent !== "✓ launched");
			const batch = pending.slice(0, MAX_ABLATION_BATCH);
			const held = pending.length - batch.length;
			launchBtn.disabled = true;
			let ok = 0;
			let fail = 0;
			for (const v of batch) {
				const row = reviewRows.get(v.name);
				if (row?.button) row.button.disabled = true;
				row?.setStatus("creating…");
				const res = await launchOne(v);
				row?.setStatus(res.ok ? "✓ launched" : `✗ ${res.msg}`);
				res.ok ? (ok += 1) : (fail += 1);
			}
			launchBtn.disabled = false;
			toast(`ablation: ${ok} launched${fail ? `, ${fail} failed` : ""}${held ? ` · ${held} held back (throttled — launch again when these finish)` : ""}`, fail ? "err" : "ok");
		};

		render();
		return {
			body: [stepLabel, bodyHost],
			actions: [el("button", { text: "cancel", onclick: close }), backBtn, nextBtn, launchBtn],
		};
	});
}
