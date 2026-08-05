// The lever catalogue — every knob the playground exposes, in one table.
//
// Each entry names a value the PlayCanvas gsplat renderer reads, where that
// value lives (`target`), how to present it, and what it actually does (`hint`,
// shown as the control's tooltip). viewer.js applies them; main.js builds the
// panel from the same table, so a knob is added in exactly one place and the UI
// and the renderer can't drift apart.
//
// `target` routes the write:
//   scene  → app.scene.gsplat        (global: LOD policy, streaming, rasterizer)
//   entity → entity.gsplat           (per-splat: which LOD this instance picks)
//   net    → octree.assetLoader      (how chunk fetches are issued)
//   app    → the app / device itself
//
// Defaults mirror the engine's own so the panel opens on stock behaviour, and
// the ones that differ are called out in their hint.

const DPR = typeof window === "undefined" ? 1 : window.devicePixelRatio || 1;

const m = (v) => `${v} m`;
const deg = (v) => `${v}\u00B0`;
const px = (v) => `${v} px`;
const mult = (v) => `${v.toFixed(1)}\u00D7`;
const num = (v) => String(v);
const frac = (v) => v.toFixed(2);

// Panel order and which sections open on first paint. `compile` and `adaptive`
// carry hand-built bodies (see compile.js and the adaptive controller); the rest
// are filled from LEVERS below.
export const GROUPS = [
    {
        id: "compile",
        title: "compile a bundle",
        blurb: "PLY to streamed SOG",
        open: false,
    },
    {
        id: "delivery",
        title: "network delivery",
        blurb: "server-side shaping",
        open: true,
    },
    {
        id: "fetch",
        title: "chunk fetching",
        blurb: "how requests are issued",
        open: false,
    },
    {
        id: "lod",
        title: "LOD selection",
        blurb: "which level a node picks",
        open: true,
    },
    {
        id: "stream",
        title: "streaming & residency",
        blurb: "when levels change",
        open: true,
    },
    {
        id: "raster",
        title: "rasterization",
        blurb: "per-splat GPU cost",
        open: false,
    },
    {
        id: "engine",
        title: "engine & debug",
        blurb: "",
        open: false,
    },
    {
        id: "adaptive",
        title: "adaptive quality",
        blurb: "best quality at a target fps",
        open: true,
    },
];

export const LEVERS = [
    // --- how the engine issues requests --------------------------------------
    {
        key: "maxConcurrentLoads",
        target: "net",
        group: "fetch",
        kind: "range",
        label: "concurrent chunk loads",
        min: 1, max: 16, step: 1, int: true, def: 2, fmt: num,
        hint:
            "Chunk fetches the engine keeps in flight. The engine ships 2, which " +
            "is conservative: a streamed chunk is 6 files (meta.json + 5 WebP " +
            "textures), so 2 in flight is ~12 requests against the browser's " +
            "~6-socket-per-origin limit on HTTP/1.1. Raise it to hide latency; " +
            "on a capped link it only splits the same pipe more ways.",
    },
    {
        key: "maxRetries",
        target: "net",
        group: "fetch",
        kind: "range",
        label: "retries per chunk",
        min: 0, max: 5, step: 1, int: true, def: 2, fmt: num,
        hint: "Re-attempts before a chunk is marked failed and its node keeps the coarser level.",
    },

    // --- which level each node picks (per-splat) ------------------------------
    {
        key: "lodBaseDistance",
        target: "entity",
        group: "lod",
        kind: "range",
        label: "base distance",
        min: 0.5, max: 120, step: 0.5, def: 5, fmt: m,
        hint:
            "Nodes closer than this render at full detail (LOD 0). Distance is " +
            "measured to the node's box, not its centre, and is FOV-corrected — " +
            "zooming in pulls detail in without touching this.",
    },
    {
        key: "lodMultiplier",
        target: "entity",
        group: "lod",
        kind: "range",
        label: "multiplier",
        min: 1.2, max: 8, step: 0.1, def: 3, fmt: mult,
        hint:
            "Each coarser level takes over this much farther out, so the ladder is " +
            "base, base\u00D7m, base\u00D7m\u00B2, … Lower drops detail sooner (cheaper, softer).",
    },
    {
        key: "lodRangeMin",
        target: "entity",
        group: "lod",
        kind: "range",
        label: "finest level allowed",
        min: 0, max: 7, step: 1, int: true, def: 0, fmt: num,
        hint: "Floor on detail. Raise it to forbid LOD 0 entirely — the cheapest way to cap cost.",
    },
    {
        key: "lodRangeMax",
        target: "entity",
        group: "lod",
        kind: "range",
        label: "coarsest level allowed",
        min: 0, max: 7, step: 1, int: true, def: 7, fmt: num,
        hint:
            "Ceiling on coarseness; clamped to the asset's level count on load. " +
            "Pin both ends to the same value to inspect one level in isolation.",
    },

    // --- when levels change, and what stays resident --------------------------
    {
        key: "splatBudget",
        target: "scene",
        group: "stream",
        kind: "range",
        label: "splat budget",
        min: 0, max: 8_000_000, step: 50_000, int: true, def: 0,
        fmt: (v) => (v <= 0 ? "uncapped" : `${(v / 1e6).toFixed(2)}M`),
        hint:
            "Hard cap on rendered Gaussians. Over budget, the engine coarsens the " +
            "FARTHEST nodes first (distance-bucketed) and scales the whole distance " +
            "ladder to converge — so the cap holds without a fixed quality cliff.",
    },
    {
        key: "lodUnderfillLimit",
        target: "scene",
        group: "stream",
        kind: "range",
        label: "underfill limit",
        min: 0, max: 4, step: 1, int: true, def: 0, fmt: num,
        hint:
            "The anti-pop-in lever. While the level a node WANTS is still " +
            "downloading, let it draw an already-resident level up to this many " +
            "steps coarser instead of nothing. 0 (the engine default) means a node " +
            "with nothing loaded stays a hole until its fetch lands.",
    },
    {
        key: "cooldownTicks",
        target: "scene",
        group: "stream",
        kind: "range",
        label: "chunk cooldown",
        min: 0, max: 600, step: 10, int: true, def: 100,
        fmt: (v) => (v <= 0 ? "evict at once" : `${v} frames`),
        hint:
            "Frames a chunk nobody references stays in memory before it is unloaded. " +
            "High keeps turning around free; 0 evicts immediately and re-downloads " +
            "the moment you look back.",
    },
    {
        key: "lodUpdateDistance",
        target: "scene",
        group: "stream",
        kind: "range",
        label: "update distance",
        min: 0, max: 20, step: 0.25, def: 1, fmt: m,
        hint:
            "Camera travel needed before levels are re-evaluated. Larger means fewer " +
            "re-evaluations (and fewer fetch storms) but detail lags behind you.",
    },
    {
        key: "lodUpdateAngle",
        target: "scene",
        group: "stream",
        kind: "range",
        label: "update angle",
        min: 0, max: 90, step: 1, int: true, def: 0,
        fmt: (v) => (v <= 0 ? "off" : deg(v)),
        hint:
            "Rotation that also triggers a re-evaluation. Off by default, so " +
            "spinning in place never re-selects levels — worth turning on when the " +
            "behind-penalty is doing work.",
    },
    {
        key: "lodBehindPenalty",
        target: "scene",
        group: "stream",
        kind: "range",
        label: "behind penalty",
        min: 1, max: 8, step: 0.5, def: 1, fmt: mult,
        hint:
            "Inflates the distance of nodes behind the camera, so what you can't see " +
            "loads coarse and the budget goes to what you can. Scales with how far " +
            "behind: straight back gets the full penalty.",
    },

    // --- per-splat GPU cost ---------------------------------------------------
    {
        key: "minPixelSize",
        target: "scene",
        group: "raster",
        kind: "range",
        label: "min pixel size",
        min: 0, max: 8, step: 0.25, def: 2, fmt: px,
        hint:
            "Floor on a splat's on-screen size. Keeps distant sub-pixel Gaussians " +
            "from flickering as they slip between samples, at the cost of blur.",
    },
    {
        key: "minContribution",
        target: "scene",
        group: "raster",
        kind: "range",
        label: "min contribution",
        min: 0, max: 16, step: 0.5, def: 3, fmt: num,
        hint: "Discards splats contributing less than this, trimming overdraw in dense regions.",
    },
    {
        key: "foveationStrength",
        target: "scene",
        group: "raster",
        kind: "range",
        label: "foveation",
        min: 0, max: 1, step: 0.05, def: 0,
        fmt: (v) => (v <= 0 ? "off" : frac(v)),
        hint:
            "Spends less on the periphery and keeps the centre sharp. Off by " +
            "default; the strongest single frame-time lever on a fill-bound scene.",
    },
    {
        key: "foveationCenter",
        target: "scene",
        group: "raster",
        kind: "range",
        label: "foveation centre",
        min: 0, max: 1, step: 0.05, def: 0.3, fmt: frac,
        hint: "Radius of the full-quality centre before foveation starts falling off.",
    },
    {
        key: "alphaClip",
        target: "scene",
        group: "raster",
        kind: "range",
        label: "alpha clip",
        min: 0, max: 1, step: 0.01, def: 0.3, fmt: frac,
        hint: "Cuts each Gaussian's footprint where its alpha falls below this — smaller quads, less overdraw.",
    },
    {
        key: "colorUpdateAngle",
        target: "scene",
        group: "raster",
        kind: "range",
        label: "SH update angle",
        min: 0, max: 45, step: 1, int: true, def: 10, fmt: deg,
        hint:
            "How far the camera may turn before view-dependent colour is " +
            "re-evaluated. Inert on our splats: stage 6 trains degree 0, so the SOG " +
            "carries one colour per Gaussian and there is nothing to re-shade.",
    },
    {
        key: "antiAlias",
        target: "scene",
        group: "raster",
        kind: "toggle",
        label: "anti-alias",
        def: false,
        hint: "Widens each Gaussian by the pixel footprint before evaluation — steadier thin geometry.",
    },
    {
        key: "radialSorting",
        target: "scene",
        group: "raster",
        kind: "toggle",
        label: "radial sorting",
        def: false,
        hint:
            "Sorts by distance to the camera instead of by view depth. Correct for " +
            "wide FOVs and re-sorts on translation rather than on rotation.",
    },

    // --- engine plumbing ------------------------------------------------------
    {
        key: "renderer",
        target: "scene",
        group: "engine",
        kind: "select",
        label: "renderer",
        def: "auto",
        options: [
            { value: "auto", label: "auto (GPU sort on WebGPU)" },
            { value: "cpu", label: "raster + CPU sort" },
            { value: "gpu", label: "raster + GPU sort (WebGPU)" },
        ],
        hint:
            "This is also the frustum-culling switch. GPU sort culls each octree " +
            "node against the frustum in a compute pass before sorting, so " +
            "off-screen nodes cost nothing; CPU sort has no culling at all and " +
            "throttles LOD updates while its worker queue is deep. GPU sort needs " +
            "WebGPU and silently falls back to CPU sort on WebGL, so check the " +
            "frustum-cull readout rather than this menu to see what you got.",
    },
    {
        key: "dataFormat",
        target: "scene",
        group: "engine",
        kind: "select",
        label: "work buffer format",
        def: "compact",
        options: [
            { value: "compact", label: "compact (default)" },
            { value: "large", label: "large (HDR colour)" },
        ],
        hint: "Precision of the shared work buffer every resident splat is baked into. Compact halves its colour cost.",
    },
    {
        key: "debug",
        target: "scene",
        group: "engine",
        kind: "select",
        label: "debug view",
        def: "none",
        options: [
            { value: "none", label: "none" },
            { value: "lod", label: "colorize by LOD level" },
            { value: "nodes", label: "octree node bounds" },
            { value: "aabbs", label: "splat bounds" },
            { value: "shUpdate", label: "flash on colour re-bake" },
        ],
        hint:
            "Colorize-by-LOD is the fastest way to see the distance ladder and the " +
            "budget's coarsening at work. Node bounds draws the octree boxes, which " +
            "are exactly what frustum culling tests.",
    },
    {
        key: "pixelRatio",
        target: "app",
        group: "engine",
        kind: "range",
        label: "render scale",
        min: 0.4, max: Math.max(2, DPR), step: 0.05, def: DPR, fmt: mult,
        hint: `Device pixel ratio the canvas renders at (this display is ${DPR.toFixed(2)}\u00D7). The bluntest fill-rate lever there is.`,
    },
    {
        key: "onDemandRender",
        target: "viewer",
        group: "engine",
        kind: "toggle",
        label: "render on demand",
        def: false,
        hint:
            "Draw only when something changed — the camera moved, or the engine " +
            "asked for a frame because chunks landed. Streaming keeps running " +
            "either way (it is driven per frame, not per render), so a parked " +
            "camera costs no GPU while the scene finishes refining.",
    },
];

export const LEVERS_BY_KEY = new Map(LEVERS.map((l) => [l.key, l]));

export const defaults = () =>
    Object.fromEntries(LEVERS.map((l) => [l.key, l.def]));

// Whole-panel stances, because the levers only make sense together: the engine
// defaults are tuned for a huge scene on a weak device, which on a strong one buys
// nothing and costs a lot of visible detail.
//
// The pivot between them is WHAT DOES THE DISTANT WORK. Leaning on LOD is cheap but
// the coarse levels are only as good as the ladder they were built from; leaning on
// the splat budget keeps full-detail geometry and lets the engine decide per node
// what to give up, which looks better but needs the frame time to be there.
//
// `null` means "this lever's own default" (used for values that depend on the
// display or the loaded asset). `lodRangeMax` is clamped to the asset's real level
// count when applied.
export const PRESETS = [
    {
        id: "quality",
        label: "quality first",
        blurb: "full detail out to 60m, coarsest levels forbidden",
        hint:
            "Pushes LOD 0 out to 60m and forbids everything coarser than level 1, so " +
            "the badly-decimated far levels never appear. The splat budget becomes the " +
            "only safety net, and the behind-penalty spends it on what you're facing. " +
            "Anti-alias on with a 1px floor handles distant shimmer, which is the job " +
            "coarse LODs would otherwise be doing.",
        values: {
            lodBaseDistance: 60, lodMultiplier: 4, lodRangeMin: 0, lodRangeMax: 1,
            splatBudget: 3_500_000, lodUnderfillLimit: 1, lodBehindPenalty: 3,
            cooldownTicks: 250, lodUpdateDistance: 0.5, lodUpdateAngle: 20,
            minPixelSize: 1, minContribution: 2, foveationStrength: 0,
            antiAlias: true, pixelRatio: null,
        },
    },
    {
        id: "balanced",
        label: "balanced",
        blurb: "LOD past 20m, quality levers near",
        hint:
            "LOD handles the far field from 20m out, while the near field keeps a 1.5px " +
            "floor and anti-alias. A 2M budget catches the worst frames without " +
            "flattening the scene the rest of the time.",
        values: {
            lodBaseDistance: 20, lodMultiplier: 3, lodRangeMin: 0, lodRangeMax: null,
            splatBudget: 2_000_000, lodUnderfillLimit: 1, lodBehindPenalty: 2,
            cooldownTicks: 150, lodUpdateDistance: 1, lodUpdateAngle: 15,
            minPixelSize: 1.5, minContribution: 3, foveationStrength: 0,
            antiAlias: true, pixelRatio: null,
        },
    },
    {
        id: "performance",
        label: "performance",
        blurb: "lean on LOD, foveation and a tight budget",
        hint:
            "For a weak GPU or a scene past a few million Gaussians: LOD from 6m, a " +
            "hard 800k budget, foveation on, and a 4x behind-penalty so nothing off to " +
            "the sides is paid for at full rate. Underfill 2 keeps holes out while it " +
            "streams.",
        values: {
            lodBaseDistance: 6, lodMultiplier: 2.5, lodRangeMin: 0, lodRangeMax: null,
            splatBudget: 800_000, lodUnderfillLimit: 2, lodBehindPenalty: 4,
            cooldownTicks: 60, lodUpdateDistance: 1.5, lodUpdateAngle: 10,
            minPixelSize: 2, minContribution: 4, foveationStrength: 0.35,
            antiAlias: false, pixelRatio: null,
        },
    },
    {
        id: "engine",
        label: "engine defaults",
        blurb: "every lever back to PlayCanvas stock",
        hint: "What the engine ships, for comparing against anything above.",
        values: null,
    },
];
