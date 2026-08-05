// Transmissive-surface discriminator — the ONE place that decides which objects
// are GENUINELY see-through (window glass, panes, glazing) and therefore need the
// weighted-blended OIT path. Shared by the Stage-5 capture (via oit.js
// prepareOITScene) and the debug mesh preview, so the reference frames and the
// preview agree.
//
// WHY NAME-BASED: Trellis tags nearly every material `alphaMode=BLEND`, which
// three.js turns into `material.transparent = true`. Measured on real cells that
// flag carries almost no information — 0 of 40 objects on a swamp, 38 of 807 on a
// house, but 377 of 383 on a hotel room, where virtually every wall, floor and
// sofa claims to be transparent. Trusting it routes the whole scene onto the OIT
// layer, which costs THREE full geometry passes per view instead of one.
//
// The only thing in this pipeline that makes a surface actually transmissive is
// `server/app/utils/glass.py`, which drives white texels to alpha 0.065 — and it
// only ever fires on objects whose prompt / noun phrase contains "window" or
// "glass". So a name gate over that same vocabulary is exactly as selective as
// the transform that created the transparency. It cannot be less selective by
// construction.
//
// It bakes that alpha into base-colour TEXELS, not into a material scalar, and
// those texels live in a GPU-compressed KTX2 the CPU cannot cheaply read back —
// which is why this is a name test rather than an alpha test. Materials that
// declare transparency in a way we CAN read (a sub-cutoff constant alpha, or a
// dedicated alphaMap) are honoured directly by oit.js and do not need to be
// listed here.
//
// ERR TOWARD INCLUSION. A false positive costs only speed (the object keeps the
// OIT path it has today). A false negative costs correctness — real glass would
// render opaque and would start writing depth. The asymmetry is deliberate, and
// `&oitall=1` on the capture URL restores the old permissive behaviour for an A/B.

// Single-token names that mark a see-through surface. Mirrors and polished metal
// belong in reflective.js, not here: they are opaque, they just reflect.
//
// The list was tuned against the two real BLEND-heavy cells rather than guessed —
// each entry below was checked for what it actually pulls onto the OIT layer, and
// three plausible-sounding candidates were REJECTED for pulling in opaque hardware:
// "shower" (matches shower_floor_pan / handheld_shower_wand / flexible_shower_hose
// / shower_downlight_*), "curtain" (matches curtain_track_housing — the see-through
// one is "sheer"), and "display" (a display SCREEN is opaque + emissive; the glass
// case is covered by the bigrams). "atrium" and "sunroom" are spaces whose floors
// and ceilings are opaque, so they are out too.
export const TRANSMISSIVE_INCLUDE = new Set([
	// the glass.py vocabulary itself — the only automatic producer of real alpha
	"glass", "glasses", "window", "windows",
	// close synonyms a descriptive id plausibly uses for the same physical thing
	"windowpane", "windowpanes", "pane", "panes", "paned", "glazing", "glazed",
	"windshield", "windscreen", "skylight", "skylights", "clerestory",
	"transom", "fanlight", "porthole", "sidelight",
	// hotel/bathroom partitions are glass far more often than not (measured: 8
	// objects, 2.1% of a hotel cell's BLEND faces — cheap insurance either way)
	"partition", "partitions",
	// translucent fabric and optics
	"sheer", "lens", "lenses",
	// enclosures whose defining feature is that you see through them
	"greenhouse", "conservatory", "aquarium", "terrarium", "vitrine", "showcase",
	// explicitly transparent materials
	"plexiglass", "perspex", "acrylic", "lucite", "transparent", "translucent",
]);

// Two-word phrases (adjacent tokens) that also mark a see-through surface, for
// names that split into separate tokens (e.g. "glass_wall" -> ["glass","wall"]
// is already caught by "glass", but "display_case" needs the pair).
export const TRANSMISSIVE_INCLUDE_BIGRAMS = new Set([
	"display case", "glass case", "glass wall", "glass door", "glass panel",
	"glass partition", "glass railing", "glass balustrade", "glass roof",
	"glass table", "glass top", "window pane", "window wall", "curtain wall",
	"see through", "shower screen", "shower enclosure", "shower door",
	"sliding door", "french door", "patio door", "bay window", "picture window",
	"sheer curtain", "sheer panel",
]);

// Tokens that VETO a match even when an include token is present — finishes that
// make an otherwise-transmissive surface opaque. Kept to finish descriptors so a
// "wooden_window_frame" still counts (its frame is opaque anyway, and the mesh's
// own pane material is what carries the alpha).
export const TRANSMISSIVE_EXCLUDE = new Set([
	"opaque", "frosted", "sandblasted", "boarded", "bricked", "blocked",
	"shuttered", "curtained", "blackout", "mirrored", "solid",
]);

function idTokens(id) {
	return String(id || "").toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
}

// True when the object id names a genuinely see-through surface (lists above).
export function isTransmissiveName(id) {
	const toks = idTokens(id);
	for (const t of toks) if (TRANSMISSIVE_EXCLUDE.has(t)) return false;
	for (const t of toks) if (TRANSMISSIVE_INCLUDE.has(t)) return true;
	for (let i = 0; i + 1 < toks.length; i++) {
		if (TRANSMISSIVE_INCLUDE_BIGRAMS.has(toks[i] + " " + toks[i + 1])) return true;
	}
	return false;
}
