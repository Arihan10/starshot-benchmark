// Shared ablation naming + launch logic, used by BOTH the guided wizard
// (ablation.js) and the unified board (ablationboard.js) so the two never drift
// — a variant's run name must be identical wherever it's created, or the board
// can't map an existing run back onto its matrix cell.
//
// An ablation forks a BASE run at the last-N firings of a step kind, applies a
// treatment (scene-shuffle method, XML tags on/off, distractors), and re-infers
// only the treated tail via the existing branch → save_run_from_branches →
// resume flow (no new server surface).

import { api } from "./api.js";

export const SHUFFLE_METHODS = ["order", "random", "distance", "raytrace"];

// Ablations are restricted to OPEN-SOURCE models (gemma / qwen). Closed models
// (gemini-flash, gpt, opus, …) are never offered or launched — the guard that
// stops a sweep from silently firing e.g. gemini-flash cells. Shared by the
// wizard + board so neither can pick a "dangerous" model.
// Max variants launched per action (wizard "launch all" + board). Each launched
// variant spawns a server-side re-inference; firing a whole sweep at once
// saturates the LLM transport (mass `llm.transport_retry`, results never landing
// cleanly), so both entry points cap here and hold the rest back.
export const MAX_ABLATION_BATCH = 8;

export const ALLOWED_MODEL = (m) => /gemma|qwen/i.test(String(m));
export const preferredModel = (models) => (models || []).find((m) => /gemma/i.test(m)) || (models || []).find(ALLOWED_MODEL) || null;
export const METHOD_HINT = {
	order: "current insertion order (baseline)",
	random: "seeded shuffle",
	distance: "nearest-first by bbox center",
	raytrace: "most-visible-first (base: falls back to order)",
};
// The one step kind never swept (the scene root plan).
export const ROOT_KINDS = new Set(["zone_plan_root"]);

// Step kinds whose prompt actually renders SCENE_CONTEXT (from the minglun
// prompt set) — the only ones where a scene-context ablation (shuffle /
// distractor) is meaningful. Root variants (…_root), overall_bbox, and
// image_prompt have no scene context, so they're never offered. Ablations are
// limited to these.
export const SCENE_CONTEXT_KINDS = new Set([
	"zone_plan", "zone_decompose", "anchor_decompose", "encapsulating_decompose",
	"negative_space_decompose", "object_bbox_batch", "child_bbox_batch", "next_object",
]);
export const hasSceneContext = (kind) => SCENE_CONTEXT_KINDS.has(kind);

export function slug(text, limit) {
	const cleaned = String(text).replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[-.]+|[-.]+$/g, "");
	return (cleaned.slice(0, limit) || "x");
}

// XML-off treatment: drop lines that are ONLY an open/close tag (the minglun
// prompts author tags on their own lines), keeping the content between them.
export function stripXmlLines(text) {
	return String(text)
		.split("\n")
		.filter((line) => !/^\s*<\/?[A-Za-z][\w-]*>\s*$/.test(line))
		.join("\n");
}

// The filename-safe treatment descriptor embedded in a variant's run name.
// `attend` is the attention-steering target (empty = the no-directive baseline).
export function treatmentTag({ method, xml, distractors, seed, attend }) {
	return [
		method || "order",
		xml ? null : "noxml",
		attend ? `att-${slug(attend, 12)}` : null,
		distractors ? `d${distractors}` : null,
		seed ? `s${seed}` : null,
	].filter(Boolean).join("_") || "order";
}

// Attention-steering directive: append an explicit "focus on X" instruction to
// the step's user prompt. The BASELINE is the same variant with attend="" (no
// directive), so pinning the two in /tf and reading X's attention answers
// "does telling the model to attend to X actually raise its attention to X?".
export function appendAttendInstruction(userTemplate, target, xml) {
	const body = `For THIS step specifically, pay heightened, focused attention to: ${target}. Prioritize and weight ${target} above the other scene-context elements when reasoning.`;
	const block = xml ? `\n\n<attention_directive>\n${body}\n</attention_directive>` : `\n\n${body}`;
	return `${userTemplate}${block}`;
}

// The prompt override for one treated step: strip XML (if xml off) and/or append
// the attend directive. Returns {} when nothing changes (falls back to the base
// prompt). Requires the base template map when xml-off or attend is in play.
export function buildStepOverride(templates, kind, { xml, attend }) {
	const st = templates ? templates[kind] : null;
	if (!st) return {};
	let system = st.system;
	let user = st.user;
	let changed = false;
	if (!xml) { system = stripXmlLines(system); user = stripXmlLines(user); changed = true; }
	if (attend) { user = appendAttendInstruction(user, attend, xml); changed = true; }
	return changed ? { [kind]: { system, user } } : {};
}

// The treatment matrix (one point = one per-fork variant), de-duped by tag. A
// seed only matters when it drives randomness (random shuffle / distractor
// pick), so it's collapsed to 0 otherwise to avoid identical-but-named runs.
export function expandTreatments({ methods, xmlOn, xmlOff, distractors, seed, attendTargets }) {
	const xmls = [];
	if (xmlOn) xmls.push(true);
	if (xmlOff) xmls.push(false);
	const dcounts = distractors && distractors.length ? distractors : [0];
	// "" (baseline, no directive) is always included so every attend variant has
	// a matching baseline to compare against.
	const attends = (attendTargets && attendTargets.length) ? ["", ...attendTargets] : [""];
	const out = [];
	const seen = new Set();
	for (const method of methods) {
		for (const xml of xmls) {
			for (const d of dcounts) {
				for (const attend of attends) {
					const s = (method === "random" || d) ? (seed || 0) : 0;
					const tag = treatmentTag({ method, xml, distractors: d, seed: s, attend });
					if (seen.has(tag)) continue;
					seen.add(tag);
					out.push({ method, xml, distractors: d, seed: s, attend, tag });
				}
			}
		}
	}
	return out;
}

// The deterministic, filesystem-safe variant run name. MUST stay in lockstep
// across the wizard + board (both call this) so existing runs map onto cells.
export function variantName({ baseRun, label, kind, cut, tag }) {
	const labelPart = label ? `${slug(label, 20)}-` : "";
	return `${slug(baseRun, 36)}__abl-${labelPart}${slug(kind, 20)}@${cut}-${slug(tag, 28)}`;
}

// The base cell's prompt snapshot, keyed by step → {system, user}. Needed to
// build the XML-off override. Cached per base run.
const _templateCache = new Map();
export async function loadBaseTemplates(baseRun) {
	if (_templateCache.has(baseRun)) return _templateCache.get(baseRun);
	const tmpl = await api.promptTemplates(baseRun);
	const map = {};
	for (const s of tmpl.steps || []) map[s.step] = { system: s.system || "", user: s.user || "" };
	_templateCache.set(baseRun, map);
	return map;
}
export function clearTemplateCache(baseRun) {
	if (baseRun) _templateCache.delete(baseRun); else _templateCache.clear();
}

// Fork the base at (kind@cut), apply the treatment override, save as the
// auto-named variant run (prefix shared, meshes hardlinked), then resume it to
// re-infer the treated tail. `templates` is the base prompt map (from
// loadBaseTemplates) — required only for the XML-off override.
export async function launchVariant({ baseRun, slot, model, label, lastN, kind, cut, method, xml, distractors, seed, attend, tag }, templates) {
	const overrides = buildStepOverride(templates, kind, { xml, attend });
	const name = variantName({ baseRun, label, kind, cut, tag: tag ?? treatmentTag({ method, xml, distractors, seed, attend }) });
	try {
		const br = await api.createBranch(baseRun, slot, model, { event_index: cut, step: kind, overrides });
		const bid = br?.branch?.id;
		await api.saveRunFromBranches({
			name, base_run: baseRun, overrides, branches: bid ? [bid] : [],
			version_label: `abl${label ? ":" + label : ""}:${kind}@${cut}:${tag ?? ""}`,
			ablation: {
				// slot/model record the exact cell this variant forked from + ran in,
				// so the board can read its status / open its scene at the RIGHT cell
				// (variant names don't encode the cell).
				label: label || null, base_run: baseRun, slot, model, target_step_kind: kind, cut, last_n: lastN,
				treatment: { shuffle_method: method, xml_tags: xml, section_order: "default", distractors, seed, attend_target: attend || null },
			},
		});
		await api.resume(name, slot, model);
		if (bid) { try { await api.branchDiscard(bid); } catch { /* best-effort cleanup */ } }
		return { ok: true, name };
	} catch (e) {
		const msg = String(e.message);
		return { ok: false, name, msg: msg.includes("409") ? "already exists" : msg };
	}
}
