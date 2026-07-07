// Attention-ablation setup wizard. "⚗ ablation…" opens this panel.
//
// An ablation is anchored to an existing BASE run. Launching SWEEPS every step
// kind except zone_plan_root: for each kind it forks the base at the last-N
// firings of that kind (one fork per firing), applies a treatment, and re-infers
// only the treated tail — the committed prefix is shared (replayed from the
// base, meshes hardlinked), so common segments are stored/computed once and each
// fork only diverges where needed. Variants are auto-named runs in the `__abl-`
// namespace (hidden from the dashboard run picker) and reuse the existing
// branch → save_run_from_branches → resume → attention endpoints (no new
// harness). Fire them INDIVIDUALLY (▶ per variant) or all at once.

import { api } from "./api.js";
import { state } from "./state.js";
import { el, field, openModal, toast } from "./ui.js";
import { SHUFFLE_METHODS, METHOD_HINT, hasSceneContext, ALLOWED_MODEL, preferredModel, MAX_ABLATION_BATCH, expandTreatments, variantName, loadBaseTemplates, launchVariant, clearTemplateCache } from "./ablationcore.js";

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
		methods: new Set(SHUFFLE_METHODS),
		xmlOn: true,
		xmlOff: true,
		distractors: [0],
		seed: 1,
	};
	let baseSteps = []; // the base cell's full step list (event_index + template)
	let kindCounts = {}; // step kind -> firing count
	let allKinds = []; // fired kinds except root, in first-seen order

	const STEP_TITLES = ["base run + cell", "sweep — kinds × last-N", "treatment axes", "review & launch"];
	const bodyHost = el("div", {});
	const stepLabel = el("div", { class: "m-hint" });
	const backBtn = el("button", { text: "‹ back" });
	const nextBtn = el("button", { class: "primary", text: "next ›" });
	const launchBtn = el("button", { class: "primary", text: `⚗ launch batch (≤${MAX_ABLATION_BATCH})` });
	let step = 0;

	// The treatment matrix (one point = a per-fork variant), de-duped.
	function treatments() {
		return expandTreatments(cfg);
	}

	// Firing event-index cuts per kind (the last-N firings — fewer if incomplete).
	function cutsForKind(k) {
		const firings = baseSteps.filter((s) => (s.template ?? s.step) === k).map((s) => s.event_index);
		return firings.slice(-Math.max(1, cfg.lastN));
	}

	// The full variant set: selected kinds × their last-N firing cuts × treatments.
	// Each variant carries its own cut (fork point) and kind.
	function buildVariants() {
		const ts = treatments();
		const out = [];
		const seen = new Set();
		for (const k of allKinds) {
			if (!cfg.kinds.has(k)) continue;
			for (const cut of cutsForKind(k)) {
				for (const t of ts) {
					const name = variantName({ baseRun: cfg.baseRun, label: cfg.label, kind: k, cut, tag: t.tag });
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
			const labelInput = el("input", { type: "text", value: cfg.label, placeholder: "e.g. shuffle-sweep",
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
				el("div", { class: "m-hint", text: `${forks} fork point${forks === 1 ? "" : "s"} across the checked kinds (× the treatment matrix = the variant count).` }),
			];
		}

		function renderTreatment() {
			const methodGrid = el("div", { class: "check-grid" },
				SHUFFLE_METHODS.map((m) =>
					el("label", { title: METHOD_HINT[m] },
						el("input", { type: "checkbox", value: m, ...(cfg.methods.has(m) ? { checked: "" } : {}),
							onchange: (e) => { e.target.checked ? cfg.methods.add(m) : cfg.methods.delete(m); } }),
						m)));
			const xmlOnCheck = el("input", { type: "checkbox", ...(cfg.xmlOn ? { checked: "" } : {}), onchange: (e) => { cfg.xmlOn = e.target.checked; } });
			const xmlOffCheck = el("input", { type: "checkbox", ...(cfg.xmlOff ? { checked: "" } : {}), onchange: (e) => { cfg.xmlOff = e.target.checked; } });
			const distInput = el("input", { type: "text", value: cfg.distractors.join(","), placeholder: "0,2",
				onchange: (e) => {
					cfg.distractors = String(e.target.value).split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => Number.isFinite(n) && n >= 0);
					if (!cfg.distractors.length) cfg.distractors = [0];
				} });
			const seedInput = el("input", { type: "number", value: String(cfg.seed), onchange: (e) => { cfg.seed = Number(e.target.value) || 0; } });
			return [
				el("div", { class: "m-field" }, el("span", { text: "scene-context shuffle methods" }), methodGrid),
				el("div", { class: "m-field" }, el("span", { text: "XML tags" }),
					el("div", { class: "check-grid" }, el("label", {}, xmlOnCheck, "with tags"), el("label", {}, xmlOffCheck, "stripped (xml off)"))),
				field("distractor counts (comma-separated)", distInput),
				field("RNG seed (random shuffle / distractor pick)", seedInput),
				el("div", { class: "m-hint", text: "Each fork gets one variant per point in this matrix (identical variants de-duped). XML on/off + scene-shuffle (order/random/distance) apply now; distractor injection is the remaining scene-hook piece." }),
			];
		}

		// --- launch (shared by ▶ per-variant and "launch all") ------------------
		let reviewRows = new Map(); // variant name -> { setStatus, button }

		async function ensureBaseTemplates() {
			try { return await loadBaseTemplates(cfg.baseRun); }
			catch (e) { setError(`couldn't load the base prompts: ${e.message}`); return null; }
		}

		async function launchOne(v) {
			const tmpls = await ensureBaseTemplates();
			if (!tmpls) return { ok: false, msg: "no base prompts" };
			return launchVariant({
				baseRun: cfg.baseRun, slot: cfg.slot, model: cfg.model, label: cfg.label, lastN: cfg.lastN,
				kind: v.kind, cut: v.cut, method: v.method, xml: v.xml, distractors: v.distractors, seed: v.seed, attend: v.attend, tag: v.tag,
			}, tmpls);
		}

		function renderReview() {
			const variants = buildVariants();
			reviewRows = new Map();
			const list = el("div", { style: "max-height:260px;overflow:auto;border:1px solid var(--line,#333);border-radius:6px;padding:6px 8px" });
			if (!variants.length) {
				list.appendChild(el("div", { class: "m-hint", text: "no variants — check at least one kind, one method, and one XML option" }));
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
						title: `kind=${v.kind} · cut=@${v.cut} · method=${v.method} · xml=${v.xml} · distractors=${v.distractors} · seed=${v.seed}` },
						btn, el("span", { style: "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1", text: `@${v.cut} · ${v.tag}` }), status);
					reviewRows.set(v.name, { setStatus: (t) => { status.textContent = t; }, button: btn, status });
					list.appendChild(row);
				}
			}
			return [
				el("div", { class: "m-hint", text: `${cfg.baseRun} · ${cfg.slot} · ${cfg.model}${cfg.label ? ` · ${cfg.label}` : ""}` }),
				el("div", { class: "m-field" }, el("span", { text: `${variants.length} variant run${variants.length === 1 ? "" : "s"} — ▶ launches one; “launch batch” fires up to ${MAX_ABLATION_BATCH} at a time (throttled to avoid a request flood). Launch again for more.` }), list),
				el("div", { class: "m-hint", text: "Each variant = fork the base at its kind@cut with prompt overrides ▸ save as this run (prefix shared) ▸ resume to re-infer (LLM; meshes skipped). Compute + compare attention in the /tf ablation view." }),
			];
		}

		const RENDERERS = [renderBase, renderSweep, renderTreatment, renderReview];

		function render() {
			setError("");
			stepLabel.textContent = `step ${step + 1} / 4 — ${STEP_TITLES[step]}`;
			bodyHost.replaceChildren(...RENDERERS[step]().filter(Boolean));
			backBtn.style.display = step > 0 ? "" : "none";
			nextBtn.style.display = step < 3 ? "" : "none";
			launchBtn.style.display = step === 3 ? "" : "none";
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
					for (const s of baseSteps) { const k = s.template ?? s.step; if (k) kindCounts[k] = (kindCounts[k] || 0) + 1; }
					allKinds = [...new Set(baseSteps.map((s) => s.template ?? s.step).filter(Boolean))].filter(hasSceneContext);
					cfg.kinds = new Set(allKinds); // default: sweep every kind
				} catch (e) {
					setError(`couldn't load steps for this cell: ${e.message}`);
					nextBtn.disabled = false;
					return;
				}
				nextBtn.disabled = false;
				if (!allKinds.length) { setError("this cell has no scene-context step kinds (root plans / overall_bbox / image_prompt don't count) — pick a base cell that has run"); return; }
			} else if (step === 1) {
				if (!cfg.kinds.size) { setError("check at least one step kind to sweep"); return; }
			} else if (step === 2) {
				if (!cfg.methods.size) { setError("check at least one shuffle method"); return; }
				if (!cfg.xmlOn && !cfg.xmlOff) { setError("check at least one XML option"); return; }
			}
			step += 1;
			render();
		};

		// "launch batch": run up to MAX_ABLATION_BATCH sequentially, holding the
		// rest back so a sweep can't flood the LLM transport (which caused mass
		// transport retries + results never landing). Launch again for more.
		launchBtn.onclick = async () => {
			const variants = buildVariants();
			if (!variants.length) { setError("nothing to launch — check the axes"); return; }
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
