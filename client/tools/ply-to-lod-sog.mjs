#!/usr/bin/env node
// ply-to-lod-sog.mjs — build a PlayCanvas "Streamed SOG" (a `lod-meta.json`
// manifest + chunked per-LOD `.sog` units) from a trained/healed splat PLY, so
// the playground can STREAM level-of-detail over the network: PlayCanvas' gsplat
// LOD system loads coarse chunks first, then refines per-octree-node by camera
// distance / splat budget (see client/public/js/playground/*).
//
// Streamed SOG needs MULTIPLE LOD levels — a single-LOD input trips a
// splat-transform edge case (calcBound on a lone level). We source levels two
// ways, in priority order:
//   1) DISCOVER a training-built ladder beside the input — <stem>.lod1.ply,
//      <stem>.lod2.ply, … (splat/stage6 `_export_lod`). Trainer-built levels beat
//      decimation, so we prefer them when present.
//   2) DECIMATE the base ourselves into a geometric ladder (merge-based, via
//      splat-transform) when no sibling ladder exists.
//
// Our PLYs are 2DGS (scale_0/scale_1 only); SOG is a 3DGS format, so every level
// is first flattened to a temp 3DGS PLY (ply-utils.flattenFileIfNeeded), exactly
// like ply-to-sog.mjs. Temp files are removed unless --keep-temp.
//
// Usage:
//   node tools/ply-to-lod-sog.mjs <in.ply> <outdir | outdir/lod-meta.json>
//        [--gpu <n|cpu>] [--levels N] [--ratio R] [--sh-iterations N]
//        [--chunk-count K] [--chunk-extent M] [--no-discover]
//        [--flat-scale F] [--keep-temp]

import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

import { flattenFileIfNeeded, parsePlyHeader } from "./ply-utils.mjs";

const CLI = fileURLToPath(
    new URL("../node_modules/@playcanvas/splat-transform/bin/cli.mjs", import.meta.url),
);

function parseArgs(argv) {
    const pos = [];
    const opt = {
        gpu: null,
        levels: 4, // total LOD levels (LOD0 + generated coarser ones) for the decimate path
        ratio: 0.35, // each generated level keeps this fraction of the previous level's count
        shIterations: null,
        chunkCount: 256, // -C, K gaussians per chunk (< default 512 → better spatial coverage)
        chunkExtent: 16, // -X, chunk size in world units
        harmonics: null, // strip SH bands > n (0 = flat colour). null keeps all bands
        nodeHeap: null, // --max-old-space-size for the splat-transform child (huge scenes)
        discover: true, // prefer a sibling <stem>.lodK.ply ladder when present
        flatScale: 1e-4,
        keepTemp: false,
    };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === "--gpu") opt.gpu = argv[++i];
        else if (a === "--levels") opt.levels = parseInt(argv[++i], 10);
        else if (a === "--ratio") opt.ratio = parseFloat(argv[++i]);
        else if (a === "--sh-iterations") opt.shIterations = parseInt(argv[++i], 10);
        else if (a === "--chunk-count") opt.chunkCount = parseInt(argv[++i], 10);
        else if (a === "--chunk-extent") opt.chunkExtent = parseFloat(argv[++i]);
        else if (a === "--filter-harmonics") opt.harmonics = parseInt(argv[++i], 10);
        else if (a === "--node-heap") opt.nodeHeap = parseInt(argv[++i], 10);
        else if (a === "--no-discover") opt.discover = false;
        else if (a === "--flat-scale") opt.flatScale = parseFloat(argv[++i]);
        else if (a === "--keep-temp") opt.keepTemp = true;
        else pos.push(a);
    }
    return { pos, opt };
}

// Node flags prefixed to every splat-transform child (e.g. a bigger heap for
// many-million-gaussian scenes). Set once from opt in main().
let NODE_ARGS = [];

function runCli(args) {
    return new Promise((resolve) => {
        const child = spawn(process.execPath, [...NODE_ARGS, CLI, ...args], { stdio: "inherit" });
        child.on("close", (code) => resolve(code ?? 1));
    });
}

async function runOrThrow(args, what) {
    let code = await runCli(args);
    if (code !== 0) throw new Error(`splat-transform ${what} failed (exit ${code})`);
}

// Resolve the manifest path + its directory from an outdir OR an explicit
// .../lod-meta.json (the runtime REQUIRES the basename be exactly lod-meta.json).
function resolveOut(outArg) {
    const isManifest = path.basename(outArg) === "lod-meta.json";
    const dir = isManifest ? path.dirname(outArg) : outArg;
    return { dir, manifest: path.join(dir, "lod-meta.json") };
}

// Total bytes of the emitted bundle: the manifest + every chunk file (each LOD is
// an unbundled SOG folder — meta.json + .webp textures — under `dir`).
function dirBytes(dir) {
    let sum = 0;
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
        const p = path.join(dir, e.name);
        sum += e.isDirectory() ? dirBytes(p) : fs.statSync(p).size;
    }
    return sum;
}

// A flattened-to-3DGS copy of `src` in `tmpDir` (or `src` itself when already
// 3DGS). Returns the path to feed splat-transform.
function flatten(src, tmpDir, tag, flatScale, temps) {
    const dst = path.join(tmpDir, `lodsog-${tag}-${process.pid}.ply`);
    const r = flattenFileIfNeeded(src, dst, flatScale);
    if (r.flattened) {
        temps.push(dst);
        return { file: dst, header: r.header };
    }
    return { file: src, header: r.header };
}

// Levels 1..N-1 by discovering a trainer-built ladder beside the base, else by
// decimating the (flattened) base into a geometric count ladder.
async function buildLevels(inPath, lod0, tmpDir, opt, temps) {
    const stem = inPath.slice(0, -path.extname(inPath).length);
    if (opt.discover) {
        const siblings = [];
        for (let k = 1; ; k++) {
            const p = `${stem}.lod${k}.ply`;
            if (!fs.existsSync(p)) break;
            siblings.push(flatten(p, tmpDir, `d${k}`, opt.flatScale, temps).file);
        }
        if (siblings.length > 0) return { levels: siblings, source: "discovered" };
    }

    const base = Math.max(1, Math.floor(opt.levels) - 1);
    const levels = [];
    let count = lod0.header.count;
    for (let k = 1; k <= base; k++) {
        count = Math.round(count * opt.ratio);
        if (count < 1000) break; // a coarser level than this adds nothing useful
        const out = path.join(tmpDir, `lodsog-gen${k}-${process.pid}.ply`);
        // Decimate is CPU/merge-based (no GPU) and must be the FINAL action with a
        // .ply output — one invocation per level, always from LOD0 (the full model).
        // Strip SH first (when requested) so the temps — and the merge — stay light.
        const dec = [lod0.file];
        if (opt.harmonics != null) dec.push("--filter-harmonics", String(opt.harmonics));
        dec.push("--decimate", String(count), out, "-w");
        await runOrThrow(dec, `decimate lod${k} (${count})`);
        temps.push(out);
        levels.push(out);
    }
    return { levels, source: "decimated" };
}

async function main() {
    const { pos, opt } = parseArgs(process.argv.slice(2));
    if (pos.length < 2) {
        console.error(
            "Usage: node tools/ply-to-lod-sog.mjs <in.ply> <outdir | outdir/lod-meta.json> " +
                "[--gpu <n|cpu>] [--levels N] [--ratio R] [--sh-iterations N] " +
                "[--chunk-count K] [--chunk-extent M] [--filter-harmonics 0|1|2|3] " +
                "[--node-heap MB] [--no-discover] [--flat-scale F] [--keep-temp]",
        );
        process.exit(1);
    }
    if (!fs.existsSync(CLI)) {
        console.error(`splat-transform not installed at ${CLI} — run \`npm install\` in client/`);
        process.exit(1);
    }
    if (opt.nodeHeap) NODE_ARGS = [`--max-old-space-size=${opt.nodeHeap}`];
    const [inPath, outArg] = pos;
    const { dir: outDir, manifest } = resolveOut(outArg);
    fs.mkdirSync(outDir, { recursive: true });

    const tmpDir = os.tmpdir();
    const temps = [];
    const t0 = Date.now();
    try {
        const lod0 = flatten(inPath, tmpDir, "d0", opt.flatScale, temps);
        const { levels, source } = await buildLevels(inPath, lod0, tmpDir, opt, temps);
        if (levels.length === 0) {
            throw new Error(
                "only one LOD level available — streamed SOG needs ≥2. Provide a " +
                    "<stem>.lodK.ply ladder or raise --levels on a denser model.",
            );
        }

        // input [ACTIONS] input [ACTIONS] … lod-meta.json [GLOBALS]. Each `-l k`
        // tags the PRECEDING input as LOD level k (LOD0 = full detail). Chunk
        // sizing uses the long options — the short `-C`/`-X` from older releases
        // now mean --filter-cluster / (unassigned) in splat-transform v3.
        const args = [
            "-w",
            "--lod-chunk-count", String(opt.chunkCount),
            "--lod-chunk-extent", String(opt.chunkExtent),
        ];
        args.push(lod0.file, "-l", "0");
        levels.forEach((f, i) => args.push(f, "-l", String(i + 1)));
        args.push(manifest, "--filter-nan");
        // Strip SH on the full-detail LOD0 too, so every level matches the temps.
        if (opt.harmonics != null) args.push("--filter-harmonics", String(opt.harmonics));
        if (opt.gpu) args.push("-g", opt.gpu);
        if (opt.shIterations) args.push("-i", String(opt.shIterations));
        await runOrThrow(args, "bundle lod-meta.json");
    } catch (e) {
        console.error(`[ply-to-lod-sog] ${e.message}`);
        process.exit(1);
    } finally {
        if (!opt.keepTemp) for (const t of temps) fs.rmSync(t, { force: true });
        else console.error(`[ply-to-lod-sog] kept ${temps.length} temp file(s) in ${tmpDir}`);
    }

    const meta = JSON.parse(fs.readFileSync(manifest, "utf8"));
    const inBytes = fs.statSync(inPath).size;
    const outBytes = dirBytes(outDir);
    console.log(
        JSON.stringify(
            {
                in: inPath,
                out: manifest,
                representation_in: parsePlyHeader(fs.readFileSync(inPath)).props.includes("scale_2")
                    ? "3dgs"
                    : "2dgs",
                harmonics: opt.harmonics,
                lod_levels: meta.lodLevels ?? null,
                counts: meta.counts ?? null,
                files: Array.isArray(meta.filenames) ? meta.filenames.length : null,
                in_bytes: inBytes,
                out_bytes: outBytes,
                ratio: +(inBytes / outBytes).toFixed(2),
                elapsed_s: +((Date.now() - t0) / 1000).toFixed(1),
            },
            null,
            1,
        ),
    );
}

main();
