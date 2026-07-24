// Reflective-surface discriminator — the ONE place that decides which objects
// keep view-dependent reflections (mirrors, glass, water, polished metal/stone,
// screens) and which are forced fully MATTE. Shared by the Stage-5 capture
// (splatcapture.js) and the debug mesh preview (splatviewer.js) so the reference
// frames and the preview agree.
//
// WHY NAME-BASED: Trellis tags nearly every material metalness=1 / roughness=1
// at the SCALAR level (the real values, if any, live in the metallic-roughness
// MAP), so a scalar threshold can't tell a mirror from a wall — every surface
// reads as "fully metallic, fully rough". View-dependent reflections are the
// single most expensive thing Stage-6 SH has to reconstruct, so we keep them
// ONLY where they are a defining visual feature (this list) and make everything
// else view-INDEPENDENT. Matching is TOKEN-based on the object's id (the LLM's
// descriptive name), same technique as emissive.js.

// Single-token names that mark a high-signal reflective surface. Edit freely —
// this is the curated vocabulary. Keep it FOCUSED: everything not matched here
// is matted, so over-including defeats the purpose.
export const REFLECTIVE_INCLUDE = new Set([
	// mirrors + glass
	"mirror", "mirrors", "mirrored", "glass", "window", "windows", "windowpane",
	"windshield", "pane", "panes", "glazing", "skylight",
	// water
	"water", "puddle", "puddles", "pool", "pond", "lake", "river", "stream",
	"fountain", "waterfall", "aquarium",
	// metals
	"chrome", "chromed", "steel", "stainless", "metal", "metallic", "brass",
	"bronze", "copper", "gold", "gilded", "silver", "silverware", "aluminum",
	"aluminium", "pewter", "platinum", "titanium",
	// metal fixtures / tableware (near-mirror finishes)
	"faucet", "faucets", "tap", "sink", "kettle", "teapot", "toaster",
	"cutlery", "tableware", "chandelier",
	// glossy stone / ceramic
	"marble", "granite", "tile", "tiles", "tiled", "porcelain", "ceramic",
	"glazed", "terrazzo", "quartz", "onyx",
	// finishes + gems
	"polished", "lacquered", "varnished", "glossy", "gloss", "shiny", "crystal",
	"gem", "gemstone", "jewel", "jewelry", "jewellery", "diamond",
	// screens (also emissive; a surface can be both)
	"screen", "screens", "monitor", "monitors", "tv", "television",
]);

// Two-word phrases (adjacent tokens) that also mark a reflective surface.
export const REFLECTIVE_INCLUDE_BIGRAMS = new Set([
	"stainless steel", "polished concrete", "polished stone", "polished floor",
	"glass panel", "window pane", "marble floor", "granite countertop",
	"granite counter", "ceramic tile", "still water", "water surface",
	"glass table", "glass door", "glass wall", "metal railing", "chrome finish",
	"mirror wall", "tile floor", "brushed metal",
]);

// Tokens that VETO a match even when an include token is present — finishes that
// make an otherwise-reflective material NON-reflective (frosted glass, rusted
// metal). Kept to finish descriptors so a "wooden_mirror_frame" still counts.
export const REFLECTIVE_EXCLUDE = new Set([
	"frosted", "matte", "sandblasted", "rusty", "rusted", "tarnished",
	"corroded", "unpolished", "opaque",
]);

function idTokens(id) {
	return String(id || "").toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
}

// True when the object id names a high-signal reflective surface (lists above).
export function isReflectiveName(id) {
	const toks = idTokens(id);
	for (const t of toks) if (REFLECTIVE_EXCLUDE.has(t)) return false;
	for (const t of toks) if (REFLECTIVE_INCLUDE.has(t)) return true;
	for (let i = 0; i + 1 < toks.length; i++) {
		if (REFLECTIVE_INCLUDE_BIGRAMS.has(toks[i] + " " + toks[i + 1])) return true;
	}
	return false;
}

// Force ONE material fully matte / view-independent, in place. Zeroing the
// scalars is not enough — Trellis carries the real metal/rough signal in the
// metallic-roughness MAP (scalars are multiplied by it), so the maps are nulled
// too, and the physical specular layers zeroed. At metalness 0 / roughness 1 the
// specular lobe collapses to a soft blur, so the sun + emissive point lights no
// longer paint a moving highlight — only diffuse (view-independent) shading and
// the neutral env's flat ambient remain. (The env is KEPT for legibility.)
export function forceMatte(m) {
	if (!m || !(m.isMeshStandardMaterial || m.isMeshPhysicalMaterial)) return;
	m.metalness = 0;
	m.roughness = 1;
	m.metalnessMap = null;
	m.roughnessMap = null;
	if (m.isMeshPhysicalMaterial) {
		m.clearcoat = 0;
		m.sheen = 0;
		m.iridescence = 0;
		m.specularIntensity = 0;
	}
	m.needsUpdate = true;
}

// Full-matte every material under `objectRoot` UNLESS its id names a reflective
// surface (then its authored PBR + maps are kept so reflections.js can bake a
// scene probe into it). Returns true when the object was matted, false when
// kept reflective — call once per top-level object.
export function matteNonReflective(objectRoot, id) {
	if (isReflectiveName(id)) return false;
	objectRoot.traverse((o) => {
		if (!o.isMesh || !o.material) return;
		const mats = Array.isArray(o.material) ? o.material : [o.material];
		for (const m of mats) forceMatte(m);
	});
	return true;
}
