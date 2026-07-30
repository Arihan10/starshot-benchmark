#!/usr/bin/env node
// splat-to-web-sog.mjs — re-encode trained splats into WEB-SIZED .sog bundles.
//
// Takes ONE FILE OR A WHOLE FOLDER and produces a `.web.sog` beside each input
// (or under --out), so the same command serves a single hand-picked splat and a
// sweep over every cell a run produced. Unlike ply-to-sog.mjs — which is a
// faithful 1:1 encode of one model, and the thing the server's on-demand
// compress button drives — this is the DELIVERY encode: it is allowed to throw
// detail away to hit a size a browser can actually stream.
//
// WHAT ACTUALLY DRIVES THE SIZE. Measured against two encodes of the same scene
// (699K splats → 12.6 MB, 6.98M → 102 MB), a SOG costs roughly:
//
//     means_l 3.0 + means_u 0.8 + quats 3.2 + scales 2.9 + sh0 3.2  ≈ 13.2 B/splat
//     + shN_labels 1.7 B/splat + a FIXED ~2.3 MB shN centroid palette
//
// so SPLAT COUNT dominates and the spherical harmonics are a rounding error next
// to it. Stripping SH entirely (--sh 0) saves ~1.7 B/splat plus the palette, and
// costs every view-dependent highlight the trainer spent its degree-3 bands on
// (splat/stage6.py). Decimating is therefore the primary lever here and --sh is
// the last resort, which is the opposite of the instinct.
//
// THE TWO-PASS SHAPE. splat-transform's `--decimate` must be the FINAL action and
// must write a .ply, so any run that decimates is necessarily:
//
//     pass 1   in → [filters] → --decimate N → temp.ply
//     pass 2   temp.ply → out.web.sog
//
// A run that only filters collapses to a single pass. Both are handled below; the
// temps are removed unless --keep-temp.
//
// 2DGS INPUTS are flattened to 3DGS first (a thin scale_2 inserted — see
// ply-utils.flattenFileIfNeeded), because SOG is a three-scale format. Already-3DGS
// inputs are passed through untouched, and non-PLY inputs (.sog/.spz/.ksplat) are
// already 3DGS by construction.
//
// IT ALSO REPORTS THE WORLD-SPACE BOUNDS of what it wrote, decoded out of the
// output's own meta.json. A trained splat is only useful to a viewer if it lands in
// the same frame as the rest of the scene, and a mis-framed encode is invisible
// until something is rendered against it — so the number that would have caught it
// is printed at the point the file is made.
//
// --translate / --rotate / --scale BAKE A FRAME CORRECTION into the output. Trainers
// that renormalize the scene on ingest (Postshot does) hand back a splat whose origin
// is nowhere near the world the rest of the pipeline works in, and the fix belongs in
// the asset rather than in every viewer that loads it. Derive the numbers by nudging
// the splat at runtime against known world geometry first — bounding-box arithmetic
// alone cannot tell a translation from an axis flip, since both move an AABB's corners
// the same way — then bake what you confirmed.
//
// Usage:
//   node tools/splat-to-web-sog.mjs <file | folder> [--out <file | folder>]
//        [--translate x,y,z] [--rotate x,y,z] [--scale f]
//        [--max-splats N | --target-mb M | --no-decimate] [--sh 0|1|2|3] [--no-prune]
//        [--min-opacity V] [--no-floaters] [--morton] [--gpu <n|cpu>]
//        [--iterations N] [--suffix .web.sog] [--force] [--dry-run] [--keep-temp]

import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import * as zlib from "node:zlib";
import { fileURLToPath } from "node:url";

import { flattenFileIfNeeded, readPlyHeader } from "./ply-utils.mjs";
import { PRUNE_DEFAULTS, buildPruneActions } from "./prune.mjs";

const CLI = fileURLToPath(
    new URL("../node_modules/@playcanvas/splat-transform/bin/cli.mjs", import.meta.url),
);

// Every splat container splat-transform can READ. `.compressed.ply` ends in
// `.ply`, so it is covered by the plain extension test.
const SPLAT_EXTS = [".ply", ".sog", ".spz", ".ksplat", ".splat"];

// Trainer-built LOD rungs (splat/stage6 `_export_lod`) sit beside the base model
// as <stem>.lod1.ply, .lod2.ply, … Encoding each of them as its own delivery
// splat is never what a folder sweep means — the ladder belongs to the streamed
// LOD bundle (ply-to-lod-sog.mjs), not here.
const LOD_SIBLING = /\.lod\d+\.ply$/i;

// Which container to read when several in one folder are the SAME model — a cell
// routinely holds both `trained.ply` and `trained.sog`, and both resolve to the
// same `trained.web.sog`. Least lossy first: re-encoding the PLY beats re-encoding
// an already-quantized SOG, which would compound the loss for no reason.
const SOURCE_PRIORITY = [".ply", ".sog", ".spz", ".ksplat", ".splat"];

// Bytes a finished SOG costs, from the measurements in the header comment. Used
// ONLY to turn --target-mb into a splat count; the actual size is measured and
// reported afterwards, and `size_estimate_error_pct` says how far off this was so
// the coefficients can be re-fitted against real output rather than trusted.
const SIZE_MODEL = {
    withSh: { perSplat: 14.9, fixed: 2.3 * 1024 * 1024 },
    noSh: { perSplat: 13.2, fixed: 0 },
};

// Below this a delivery splat has lost the plot — a target that implies fewer is
// clamped here and flagged, rather than silently writing a scene made of confetti.
const MIN_USEFUL_SPLATS = 50_000;

const DEFAULTS = {
    // The delivery budget. 1.5M splats lands around 25 MB with harmonics kept,
    // which streams acceptably and still holds up at walking distance. Explicit
    // and deterministic, so a re-run of the same input gives the same file.
    maxSplats: 1_500_000,
    targetMb: null, // when set, derives maxSplats from SIZE_MODEL instead
    sh: null, // null = keep every band the input carries
    // Frame correction, baked into the output. All null = the input's frame is
    // kept exactly as-is, which is the right default for a splat already in world
    // space; a renormalizing trainer's output needs these.
    translate: null,
    rotate: null,
    scale: null,
    prune: true,
    minOpacity: PRUNE_DEFAULTS.minOpacity,
    floaters: PRUNE_DEFAULTS.floaters,
    floatersParams: PRUNE_DEFAULTS.floatersParams,
    morton: false,
    gpu: null,
    iterations: null,
    suffix: ".web.sog",
    out: null,
    force: false,
    dryRun: false,
    keepTemp: false,
    flatScale: 1e-4,
};

function parseArgs(argv) {
    const pos = [];
    const opt = { ...DEFAULTS };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === "--out") opt.out = argv[++i];
        else if (a === "--translate" || a === "-t") opt.translate = argv[++i];
        else if (a === "--rotate" || a === "-r") opt.rotate = argv[++i];
        else if (a === "--scale") opt.scale = argv[++i];
        else if (a === "--max-splats") opt.maxSplats = parseInt(argv[++i], 10);
        else if (a === "--target-mb") opt.targetMb = parseFloat(argv[++i]);
        else if (a === "--sh") opt.sh = parseInt(argv[++i], 10);
        else if (a === "--no-prune") opt.prune = false;
        else if (a === "--min-opacity") opt.minOpacity = parseFloat(argv[++i]);
        else if (a === "--no-floaters") opt.floaters = false;
        else if (a === "--floaters") opt.floatersParams = argv[++i];
        else if (a === "--morton") opt.morton = true;
        else if (a === "--gpu") opt.gpu = argv[++i];
        else if (a === "--iterations") opt.iterations = parseInt(argv[++i], 10);
        else if (a === "--suffix") opt.suffix = argv[++i];
        else if (a === "--force") opt.force = true;
        else if (a === "--dry-run") opt.dryRun = true;
        else if (a === "--keep-temp") opt.keepTemp = true;
        else if (a === "--flat-scale") opt.flatScale = parseFloat(argv[++i]);
        else if (a === "--no-decimate") opt.maxSplats = Infinity;
        else pos.push(a);
    }
    return { pos, opt };
}

const USAGE =
    "Usage: node tools/splat-to-web-sog.mjs <file | folder> [--out <file | folder>]\n" +
    "       [--translate x,y,z] [--rotate x,y,z] [--scale f]\n" +
    "       [--max-splats N | --target-mb M] [--no-decimate] [--sh 0|1|2|3]\n" +
    "       [--no-prune] [--min-opacity V] [--no-floaters] [--floaters size,op,min]\n" +
    "       [--morton] [--gpu <n|cpu>] [--iterations N] [--suffix .web.sog]\n" +
    "       [--force] [--dry-run] [--keep-temp] [--flat-scale F]";

// --- input discovery ---------------------------------------------------------

function isSplatFile(name) {
    const lower = name.toLowerCase();
    return SPLAT_EXTS.some((e) => lower.endsWith(e));
}

// Everything under `dir` worth encoding, depth-first and sorted so a sweep is
// reproducible. Outputs of a previous run are skipped by suffix, which is what
// makes re-running over the same folder idempotent rather than exponential.
function walk(dir, suffix, found = []) {
    for (const e of fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) =>
        a.name.localeCompare(b.name),
    )) {
        const p = path.join(dir, e.name);
        if (e.isDirectory()) walk(p, suffix, found);
        else if (
            e.isFile() &&
            isSplatFile(e.name) &&
            !e.name.toLowerCase().endsWith(suffix.toLowerCase()) &&
            !LOD_SIBLING.test(e.name)
        ) {
            found.push(p);
        }
    }
    return found;
}

// Where one input's output lands.
//   • no --out          → beside the input, <stem><suffix>
//   • --out <file.sog>  → exactly that (single input only)
//   • --out <folder>    → mirrored UNDER that folder, keeping the input's path
//     relative to the root that was swept. Cells all name their model the same
//     thing (runs/<run>/<slot>/<model>/splat/trained.ply), so flattening to
//     basenames would have every cell overwrite the last one.
function outputFor(input, root, opt) {
    const stem = path.basename(input).replace(/\.[^.]+$/, "");
    const name = stem + opt.suffix;
    if (!opt.out) return path.join(path.dirname(input), name);
    if (opt.out.toLowerCase().endsWith(".sog")) return opt.out;
    const rel = root === input ? name : path.join(path.relative(root, path.dirname(input)), name);
    return path.join(opt.out, rel);
}

function sourceRank(file) {
    const lower = file.toLowerCase();
    const i = SOURCE_PRIORITY.findIndex((e) => lower.endsWith(e));
    return i < 0 ? SOURCE_PRIORITY.length : i;
}

// Collapse inputs that resolve to the SAME output down to the best source. Without
// this a folder holding trained.ply + trained.sog encodes twice to one path: the
// second run either overwrites the first or — worse — sees the file it just wrote,
// finds it newer than its input, and reports "up to date" for work never done.
// The losers are returned rather than dropped, so the report still accounts for
// every file that was looked at.
function resolveCollisions(inputs, root, opt) {
    const byOut = new Map();
    for (const input of inputs) {
        const out = path.resolve(outputFor(input, root, opt));
        const cur = byOut.get(out);
        if (!cur || sourceRank(input) < sourceRank(cur)) byOut.set(out, input);
    }
    const chosen = new Set(byOut.values());
    const dropped = [];
    for (const f of inputs) {
        if (chosen.has(f)) continue;
        const winner = byOut.get(path.resolve(outputFor(f, root, opt)));
        dropped.push({
            in: f,
            out: outputFor(f, root, opt),
            skipped: `same output as ${path.basename(winner)} — encoding that instead`,
        });
    }
    return { keep: inputs.filter((f) => chosen.has(f)), dropped };
}

// --- SOG inspection ----------------------------------------------------------

// A SOG is a ZIP. Its entries are written with streaming sizes (general-purpose
// flag bit 3), so the local headers carry zeroes and the CENTRAL DIRECTORY is the
// only place the real compressed size lives — hence the walk from the EOCD rather
// than a scan from the front of the file.
function readZipEntry(file, wanted) {
    const buf = fs.readFileSync(file);
    const EOCD_SIG = 0x06054b50;
    let eocd = -1;
    for (let i = buf.length - 22; i >= 0 && i >= buf.length - 22 - 0xffff; i--) {
        if (buf.readUInt32LE(i) === EOCD_SIG) {
            eocd = i;
            break;
        }
    }
    if (eocd < 0) throw new Error("not a zip (no end-of-central-directory record)");
    let at = buf.readUInt32LE(eocd + 16); // central directory offset
    const entries = buf.readUInt16LE(eocd + 10);
    for (let n = 0; n < entries; n++) {
        if (buf.readUInt32LE(at) !== 0x02014b50) throw new Error("bad central directory entry");
        const method = buf.readUInt16LE(at + 10);
        const compSize = buf.readUInt32LE(at + 20);
        const nameLen = buf.readUInt16LE(at + 28);
        const extraLen = buf.readUInt16LE(at + 30);
        const commentLen = buf.readUInt16LE(at + 32);
        const localAt = buf.readUInt32LE(at + 42);
        const name = buf.toString("latin1", at + 46, at + 46 + nameLen);
        if (name === wanted) {
            // The local header repeats the name/extra with its OWN lengths (the
            // extra field routinely differs from the central copy), so the data
            // offset has to be recomputed from it rather than reused.
            const lNameLen = buf.readUInt16LE(localAt + 26);
            const lExtraLen = buf.readUInt16LE(localAt + 28);
            const start = localAt + 30 + lNameLen + lExtraLen;
            const raw = buf.subarray(start, start + compSize);
            return method === 0 ? raw : zlib.inflateRawSync(raw);
        }
        at += 46 + nameLen + extraLen + commentLen;
    }
    throw new Error(`no '${wanted}' entry`);
}

// SOG stores positions log-companded, so the meta's means mins/maxs are in that
// space, not world space. This is splat-transform's own inverse.
const invLogTransform = (v) => {
    const e = Math.exp(Math.abs(v)) - 1;
    return v < 0 ? -e : e;
};

// The written splat's true count and WORLD-SPACE bounds. Returned as null on any
// failure: a report that can't be produced must not fail an encode that already
// succeeded.
function inspectSog(file) {
    try {
        const meta = JSON.parse(readZipEntry(file, "meta.json").toString("utf8"));
        const mins = meta?.means?.mins;
        const maxs = meta?.means?.maxs;
        const bounds =
            Array.isArray(mins) && Array.isArray(maxs)
                ? {
                      min: mins.map((v) => +invLogTransform(v).toFixed(3)),
                      max: maxs.map((v) => +invLogTransform(v).toFixed(3)),
                  }
                : null;
        return { count: meta?.count ?? null, sh_bands: meta?.shN?.bands ?? 0, bounds };
    } catch {
        return null;
    }
}

// --- splat-transform driver --------------------------------------------------

function runCli(args) {
    return new Promise((resolve) => {
        const child = spawn(process.execPath, [CLI, ...args], { stdio: "inherit" });
        child.on("close", (code) => resolve(code ?? 1));
    });
}

// Run one stage, falling back to CPU exactly as ply-to-sog.mjs does. `build` is
// re-invoked for the retry because the floater filter is GPU-only and has to be
// dropped from the args, not merely disabled at runtime.
async function runStage(build, opt, label) {
    let code = await runCli(build({ gpu: opt.gpu, floaters: opt.floaters }));
    if (code === 0) return;
    if (opt.gpu) throw new Error(`${label} failed (exit ${code})`);
    console.error(
        `[web-sog] ${label} failed on GPU — retrying with -g cpu` +
            (opt.prune && opt.floaters ? " (floater filter skipped)" : ""),
    );
    code = await runCli(build({ gpu: "cpu", floaters: false }));
    if (code !== 0) throw new Error(`${label} failed on CPU too (exit ${code})`);
}

// splat-transform works in a FLIPPED coordinate space: it negates X and Y on load
// and again on save, so the DATA round-trips unchanged — but every spatial ACTION
// in between is expressed in that internal frame, not the one the file stores.
//
// Measured, so it can be re-checked rather than believed: a splat sitting at
// x[-10.97, 19.05] y[16.42, 44.23] z[-22.33, 13.01], asked for `--translate -1,-2,-3`,
// came back at x[-9.97, 20.05] y[18.42, 46.23] z[-25.33, 10.01] — X and Y moved the
// wrong way by exactly the amount requested, Z moved the right way. Identical for the
// `--opt value` and `--opt=value` forms, so this is the convention and not an argument
// parsing quirk. prune.mjs flags the same trap for --filter-box.
//
// The flags on THIS tool therefore take the correction in WORLD space — the frame the
// offset was measured in and the frame the viewer renders — and this converts. Euler
// angles follow the same rule (conjugating a rotation by diag(-1,-1,1) negates the X
// and Y angles and leaves Z), and a uniform scale is unaffected by a reflection.
function toTransformSpace(v, flag) {
    const p = v.split(",").map((s) => Number(s.trim()));
    if (p.length !== 3 || p.some((n) => !Number.isFinite(n))) {
        throw new Error(`${flag} expects 3 comma-separated numbers, got "${v}"`);
    }
    return `${-p[0]},${-p[1]},${p[2]}`;
}

// The frame correction, applied FIRST so everything downstream — and anything the
// caller later filters with a box — sees the corrected coordinates. Order within
// the transform is scale, rotate, translate: the usual reading of "put it at this
// size, this orientation, then this place".
function transformActions(opt) {
    const actions = [];
    if (opt.scale != null) actions.push("--scale", String(opt.scale));
    if (opt.rotate != null) actions.push("--rotate", toTransformSpace(opt.rotate, "--rotate"));
    if (opt.translate != null) {
        actions.push("--translate", toTransformSpace(opt.translate, "--translate"));
    }
    return actions;
}

// The filters that run BEFORE any decimation: drop NaN/Inf gaussians (a trained
// scene reliably carries a few, and one is enough to poison a quantization
// range), then the conservative opacity + floater pass, then the SH band cut.
function filterActions(opt, floaters) {
    const actions = ["--filter-nan"];
    if (opt.prune) {
        actions.push(
            ...buildPruneActions({
                minOpacity: opt.minOpacity,
                floaters,
                floatersParams: opt.floatersParams,
            }),
        );
    }
    if (opt.sh != null) actions.push("--filter-harmonics", String(opt.sh));
    return actions;
}

// SH degree from a PLY's f_rest_* property count: band n carries 3·((n+1)²−1)
// coefficients, i.e. 9 / 24 / 45 for degrees 1 / 2 / 3.
function shBandsFromRestCount(n) {
    if (n >= 45) return 3;
    if (n >= 24) return 2;
    if (n >= 9) return 1;
    return 0;
}

// Splat count implied by a --target-mb budget, per SIZE_MODEL.
//
// `inputBands` is what the SOURCE actually carries, which is not the same question
// as what --sh asks for: a surfel cloud is already degree 0, so budgeting it for a
// harmonics palette it will never have overshot by 60% on the first real run. The
// model keys off what will EXIST in the output — the lower of the two.
function countForTargetMb(mb, opt, inputBands = 3) {
    const bands = opt.sh != null ? Math.min(opt.sh, inputBands) : inputBands;
    const model = bands > 0 ? SIZE_MODEL.withSh : SIZE_MODEL.noSh;
    const budget = mb * 1024 * 1024 - model.fixed;
    return Math.max(1, Math.floor(budget / model.perSplat));
}

// --- per-file encode ---------------------------------------------------------

async function encodeOne(input, root, opt) {
    const output = outputFor(input, root, opt);
    const inBytes = fs.statSync(input).size;

    // A folder sweep is re-run constantly (a new cell trains, the rest are done),
    // so an output already newer than its input is left alone unless forced.
    if (!opt.force && fs.existsSync(output)) {
        if (fs.statSync(output).mtimeMs >= fs.statSync(input).mtimeMs) {
            return { in: input, out: output, skipped: "up to date" };
        }
    }

    // Source count, when the container will tell us cheaply — a PLY from its
    // header, a SOG from its meta.json. Knowing it is what lets an input already
    // under budget skip the decimation pass entirely instead of round-tripping
    // through a temp PLY to "reduce" it to more splats than it has.
    const lower = input.toLowerCase();
    let sourceCount = null;
    let sourceBands = 3; // assume full harmonics when the container won't say
    if (lower.endsWith(".ply")) {
        try {
            const h = readPlyHeader(input);
            sourceCount = h.count;
            sourceBands = shBandsFromRestCount(
                h.props.filter((p) => p.startsWith("f_rest")).length,
            );
        } catch {
            sourceCount = null;
        }
    } else if (lower.endsWith(".sog")) {
        const info = inspectSog(input);
        sourceCount = info?.count ?? null;
        if (info) sourceBands = info.sh_bands ?? 0;
    }

    let target = Number.isFinite(opt.maxSplats) ? opt.maxSplats : null;
    let clamped = false;
    if (opt.targetMb != null) {
        target = countForTargetMb(opt.targetMb, opt, sourceBands);
        if (target < MIN_USEFUL_SPLATS) {
            target = MIN_USEFUL_SPLATS;
            clamped = true;
        }
    }
    // Decimating UP is not a thing — a source already under the budget is encoded
    // whole, in one pass.
    const decimating = target != null && (sourceCount == null || sourceCount > target);

    if (opt.dryRun) {
        return {
            in: input,
            out: output,
            dry_run: true,
            count_in: sourceCount,
            decimate_to: decimating ? target : null,
        };
    }

    fs.mkdirSync(path.dirname(output), { recursive: true });
    const tmpDir = os.tmpdir();
    const temps = [];
    const t0 = Date.now();

    try {
        // 2DGS → 3DGS, if this is a PLY that needs it.
        let source = input;
        let flattened = false;
        if (input.toLowerCase().endsWith(".ply")) {
            const tmp = path.join(tmpDir, `websog-flat-${process.pid}-${Date.now()}.ply`);
            const r = flattenFileIfNeeded(input, tmp, opt.flatScale);
            if (r.flattened) {
                temps.push(tmp);
                source = tmp;
                flattened = true;
            } else {
                fs.rmSync(tmp, { force: true });
            }
        }

        // Pass 1 (only when decimating): every filter plus the decimation, into a
        // temp PLY — the CLI requires decimate to be last and to write a .ply.
        if (decimating) {
            const decimated = path.join(tmpDir, `websog-dec-${process.pid}-${Date.now()}.ply`);
            const src = source;
            await runStage(
                ({ gpu, floaters }) => {
                    const args = [src, ...transformActions(opt), ...filterActions(opt, floaters)];
                    args.push("--decimate", String(target), decimated, "-w");
                    args.push("--scratch-dir", tmpDir);
                    if (gpu) args.push("-g", gpu);
                    return args;
                },
                opt,
                `decimate ${path.basename(input)} → ${target}`,
            );
            temps.push(decimated);
            source = decimated;
        }

        // Pass 2: encode to SOG. The filters already ran in pass 1 when decimating,
        // so they are only applied here on the single-pass path.
        const src = source;
        const encodeFilters = decimating ? [] : null;
        await runStage(
            ({ gpu, floaters }) => {
                // Pass 1 already transformed + filtered when decimating; doing either
                // again here would apply the frame correction twice.
                const args = [
                    src,
                    ...(decimating ? [] : transformActions(opt)),
                    ...(encodeFilters ?? filterActions(opt, floaters)),
                ];
                if (opt.morton) args.push("--morton-order");
                args.push(output, "-w");
                if (gpu) args.push("-g", gpu);
                if (opt.iterations) args.push("-i", String(opt.iterations));
                return args;
            },
            opt,
            `encode ${path.basename(output)}`,
        );

        const outBytes = fs.statSync(output).size;
        const info = inspectSog(output);
        const result = {
            in: input,
            out: output,
            count_in: sourceCount,
            count_out: info?.count ?? null,
            sh_bands_out: info?.sh_bands ?? null,
            // Decoded from the output itself — this is the number that says whether
            // the splat will land where the rest of the scene is.
            world_bounds: info?.bounds ?? null,
            flattened_to_3dgs: flattened,
            decimated_to: decimating ? target : null,
            pruned: opt.prune,
            transform:
                opt.translate || opt.rotate || opt.scale
                    ? { translate: opt.translate, rotate: opt.rotate, scale: opt.scale }
                    : null,
            in_bytes: inBytes,
            out_bytes: outBytes,
            out_mb: +(outBytes / 1024 / 1024).toFixed(2),
            ratio: +(inBytes / outBytes).toFixed(2),
            elapsed_s: +((Date.now() - t0) / 1000).toFixed(1),
        };
        if (info?.count) result.bytes_per_splat = +(outBytes / info.count).toFixed(2);
        if (opt.targetMb != null) {
            result.target_mb = opt.targetMb;
            result.size_estimate_error_pct = +(
                ((outBytes - opt.targetMb * 1024 * 1024) / (opt.targetMb * 1024 * 1024)) *
                100
            ).toFixed(1);
            if (clamped) result.target_clamped_to = MIN_USEFUL_SPLATS;
        }
        return result;
    } finally {
        if (opt.keepTemp) {
            if (temps.length) console.error(`[web-sog] kept temps: ${temps.join(", ")}`);
        } else {
            for (const t of temps) fs.rmSync(t, { force: true });
        }
    }
}

// --- entry -------------------------------------------------------------------

async function main() {
    const { pos, opt } = parseArgs(process.argv.slice(2));
    if (pos.length < 1) {
        console.error(USAGE);
        process.exit(1);
    }
    if (!fs.existsSync(CLI)) {
        console.error(`splat-transform not installed at ${CLI} — run \`npm install\` in client/`);
        process.exit(1);
    }
    if (opt.sh != null && !(opt.sh >= 0 && opt.sh <= 3)) {
        console.error(`--sh must be 0, 1, 2 or 3 (got ${opt.sh})`);
        process.exit(1);
    }
    if (opt.targetMb != null && !(opt.targetMb > 0)) {
        console.error(`--target-mb must be positive (got ${opt.targetMb})`);
        process.exit(1);
    }
    // Fail on a malformed transform before any encoding starts, not per-file
    // partway through a sweep.
    try {
        transformActions(opt);
    } catch (e) {
        console.error(e.message);
        process.exit(1);
    }
    if (opt.scale != null && !(Number(opt.scale) > 0)) {
        console.error(`--scale must be a positive number (got ${opt.scale})`);
        process.exit(1);
    }

    const root = path.resolve(pos[0]);
    if (!fs.existsSync(root)) {
        console.error(`no such file or folder: ${root}`);
        process.exit(1);
    }
    const isDir = fs.statSync(root).isDirectory();
    const found = isDir ? walk(root, opt.suffix) : [root];
    const { keep: inputs, dropped } = resolveCollisions(found, root, opt);
    if (found.length === 0) {
        console.error(`no splat files (${SPLAT_EXTS.join(", ")}) under ${root}`);
        process.exit(1);
    }
    if (!isDir && !isSplatFile(root)) {
        console.error(`not a splat file (${SPLAT_EXTS.join(", ")}): ${root}`);
        process.exit(1);
    }
    if (opt.out && opt.out.toLowerCase().endsWith(".sog") && inputs.length > 1) {
        console.error(
            `--out names a single .sog file but ${inputs.length} inputs matched — ` +
                "pass a folder instead",
        );
        process.exit(1);
    }

    console.error(
        `[web-sog] ${inputs.length} input(s)` +
            (opt.targetMb != null
                ? ` · target ${opt.targetMb} MB/file`
                : Number.isFinite(opt.maxSplats)
                  ? ` · max ${opt.maxSplats.toLocaleString()} splats`
                  : " · no decimation") +
            (opt.sh != null ? ` · SH ≤ ${opt.sh}` : "") +
            (opt.prune ? "" : " · prune off") +
            (opt.translate ? ` · translate ${opt.translate}` : "") +
            (opt.rotate ? ` · rotate ${opt.rotate}` : "") +
            (opt.scale ? ` · scale ${opt.scale}` : ""),
    );

    // Serial on purpose: each encode saturates the GPU and several GB of RAM, so
    // running two at once is slower than running them in turn, and a folder sweep
    // over a run of cells would otherwise thrash.
    const results = [...dropped];
    const failures = [];
    for (let i = 0; i < inputs.length; i++) {
        const rel = path.relative(process.cwd(), inputs[i]);
        console.error(`[web-sog] (${i + 1}/${inputs.length}) ${rel}`);
        try {
            results.push(await encodeOne(inputs[i], root, opt));
        } catch (e) {
            const message = e && e.message ? e.message : String(e);
            console.error(`[web-sog] FAILED ${rel}: ${message}`);
            failures.push({ in: inputs[i], error: message });
        }
    }

    const written = results.filter((r) => !r.skipped && !r.dry_run);
    console.log(
        JSON.stringify(
            {
                root,
                found: found.length,
                inputs: inputs.length,
                written: written.length,
                skipped: results.filter((r) => r.skipped).length,
                failed: failures.length,
                total_out_mb: +(
                    written.reduce((s, r) => s + (r.out_bytes ?? 0), 0) /
                    1024 /
                    1024
                ).toFixed(2),
                results,
                failures,
            },
            null,
            1,
        ),
    );
    if (failures.length) process.exit(1);
}

main();
