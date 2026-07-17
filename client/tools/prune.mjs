// prune.mjs — conservative pruning + scene cleanup as splat-transform ACTIONS,
// a modular add-on for the SOG side-script (client/tools/ply-to-sog.mjs). At
// conservative, visually-lossless thresholds:
//   • opacity kill  — drop splats below ~min-opacity (default 2%). Vanilla
//     training leaves a low-opacity haze that contributes ~nothing to pixels.
//   • floater removal — splat-transform --filter-floaters (SuperSplat-style):
//     remove Gaussians not contributing to any solid voxel. This also discards
//     far-flung out-of-region junk, i.e. the safe, native-space version of
//     "crop to the region users can see".
//   • (advanced) box crop — an OPT-IN explicit --filter-box. NOTE: splat-transform
//     applies a coordinate-convention flip on load, so its box space is NOT world
//     space (a world-space box silently crops to a sliver). Off by default; only
//     used when an explicit box is passed, and the box is in splat-transform's
//     own coordinates — verify the result.
// Deliberately stops short of aggressive importance pruning / --decimate, where
// soft shadows and thin structures start to erode.
//
// splat-transform exposes `opacity` NORMALIZED to [0,1] (verified via --stats),
// so --min-opacity is a plain fraction — no logit conversion needed.

export const PRUNE_DEFAULTS = {
    minOpacity: 0.02, // [0,1]; below this a splat is visually inert
    floaters: true,
    floatersParams: "0.05,0.1,0.004", // splat-transform defaults (conservative)
};

// Parse an explicit crop box "x,y,z,X,Y,Z" → { box:[6 numbers], source } or null.
// The box is in splat-transform's coordinate space (see the coordinate caveat
// above), so this is an advanced, opt-in knob — there is no auto-derivation.
export function parseCropBox(crop) {
    if (!crop) return null;
    const v = crop.split(",").map(Number);
    if (v.length !== 6 || v.some((n) => !Number.isFinite(n))) {
        throw new Error(`--crop must be 6 numbers "x,y,z,X,Y,Z", got "${crop}"`);
    }
    return { box: v, source: "explicit" };
}

// splat-transform action args for the requested filters, applied in order:
// opacity → crop → floaters. `--filter-floaters` is GPU-only, so callers drop it
// on a CPU fallback by passing floaters:false.
export function buildPruneActions({
    minOpacity = PRUNE_DEFAULTS.minOpacity,
    cropBox = null,
    floaters = PRUNE_DEFAULTS.floaters,
    floatersParams = PRUNE_DEFAULTS.floatersParams,
} = {}) {
    const actions = [];
    if (minOpacity != null && minOpacity > 0) {
        actions.push("--filter-value", `opacity,gte,${minOpacity}`);
    }
    if (cropBox) actions.push("--filter-box", cropBox.join(","));
    if (floaters) actions.push("--filter-floaters", floatersParams);
    return actions;
}
