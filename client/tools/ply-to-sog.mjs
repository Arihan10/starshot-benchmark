#!/usr/bin/env node
// ply-to-sog.mjs — SOG-encode a trained splat PLY with PlayCanvas' own encoder
// (@playcanvas/splat-transform: means/quats/scales/colours quantized into WebP
// textures, bundled as a single .sog — typically 15-20x smaller than the PLY).
//
// Our trained.ply is 2DGS (scale_0/scale_1 only); SOG is a 3DGS format, so a
// 2DGS input is first flattened to 3DGS by inserting a thin scale_2 into a temp
// PLY (see ply-utils.flattenFileIfNeeded). The temp file is deleted after encoding.
//
// OPTIONAL --prune (see prune.mjs): a modular, conservative cleanup layered on
// top of the encode — kill sub-~2% opacity splats and remove floaters (which
// also discards out-of-region junk) — as built-in splat-transform actions applied
// in the SAME pass before compression. Off unless --prune is set. An explicit
// --crop box is available but advanced (splat-transform coordinate space; see
// prune.mjs) and off by default.
//
// SOG compression runs on the GPU via WebGPU by default; if that fails we retry
// once with -g cpu (the GPU-only floater filter is skipped on that fallback)
// unless a device was pinned explicitly.
//
// Usage:
//   node tools/ply-to-sog.mjs <in.ply> <out.sog | out/meta.json>
//        [--gpu <n|cpu>] [--iterations N] [--flat-scale F] [--no-flatten] [--keep-temp]
//        [--prune] [--min-opacity V] [--no-floaters] [--floaters size,op,min]
//        [--crop x,y,z,X,Y,Z]   (advanced; splat-transform coords)

import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

import { flattenFileIfNeeded, readPlyHeader } from "./ply-utils.mjs";
import { PRUNE_DEFAULTS, buildPruneActions, parseCropBox } from "./prune.mjs";

const CLI = fileURLToPath(
    new URL("../node_modules/@playcanvas/splat-transform/bin/cli.mjs", import.meta.url),
);

function parseArgs(argv) {
    const pos = [];
    const opt = {
        gpu: null,
        iterations: null,
        flatScale: 1e-4,
        flatten: true,
        keepTemp: false,
        // pruning (all off unless --prune)
        prune: false,
        minOpacity: PRUNE_DEFAULTS.minOpacity,
        crop: null, // advanced, explicit box in splat-transform coords
        floaters: PRUNE_DEFAULTS.floaters,
        floatersParams: PRUNE_DEFAULTS.floatersParams,
    };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === "--no-flatten") opt.flatten = false;
        else if (a === "--keep-temp") opt.keepTemp = true;
        else if (a === "--gpu") opt.gpu = argv[++i];
        else if (a === "--iterations") opt.iterations = parseInt(argv[++i], 10);
        else if (a === "--flat-scale") opt.flatScale = parseFloat(argv[++i]);
        else if (a === "--prune") opt.prune = true;
        else if (a === "--min-opacity") opt.minOpacity = parseFloat(argv[++i]);
        else if (a === "--crop") opt.crop = argv[++i];
        else if (a === "--no-floaters") opt.floaters = false;
        else if (a === "--floaters") opt.floatersParams = argv[++i];
        else pos.push(a);
    }
    return { pos, opt };
}

function runCli(args) {
    return new Promise((resolve) => {
        const child = spawn(process.execPath, [CLI, ...args], { stdio: "inherit" });
        child.on("close", (code) => resolve(code ?? 1));
    });
}

// input [ACTIONS] output [GLOBALS] — actions attach to the preceding input.
function buildArgs(input, actions, output, opt, gpuOverride) {
    const args = [input, ...actions, output, "-w"];
    const gpu = gpuOverride ?? opt.gpu;
    if (gpu) args.push("-g", gpu);
    if (opt.iterations) args.push("-i", String(opt.iterations));
    return args;
}

// Output size: the bundled .sog file, or meta.json + its sibling .webp textures.
function outputBytes(outPath) {
    if (path.basename(outPath) !== "meta.json") return fs.statSync(outPath).size;
    const dir = path.dirname(outPath);
    return fs
        .readdirSync(dir)
        .filter((f) => f === "meta.json" || f.endsWith(".webp"))
        .reduce((sum, f) => sum + fs.statSync(path.join(dir, f)).size, 0);
}

async function main() {
    const { pos, opt } = parseArgs(process.argv.slice(2));
    if (pos.length < 2) {
        console.error(
            "Usage: node tools/ply-to-sog.mjs <in.ply> <out.sog | out/meta.json> " +
                "[--gpu <n|cpu>] [--iterations N] [--flat-scale F] [--no-flatten] [--keep-temp] " +
                "[--prune] [--min-opacity V] [--no-floaters] [--floaters size,op,min] " +
                "[--crop x,y,z,X,Y,Z]",
        );
        process.exit(1);
    }
    if (!fs.existsSync(CLI)) {
        console.error(`splat-transform not installed at ${CLI} — run \`npm install\` in client/`);
        process.exit(1);
    }
    const [inPath, outPath] = pos;

    // Flatten 2DGS → 3DGS to a temp PLY when needed (SOG is a 3-scale format).
    let cliInput = inPath;
    let tempPath = null;
    let header;
    if (opt.flatten) {
        tempPath = path.join(os.tmpdir(), `sog-flat-${process.pid}-${Date.now()}.ply`);
        const r = flattenFileIfNeeded(inPath, tempPath, opt.flatScale);
        header = r.header;
        if (r.flattened) cliInput = tempPath;
        else {
            fs.rmSync(tempPath, { force: true }); // was already 3DGS — no temp needed
            tempPath = null;
        }
    } else {
        header = readPlyHeader(inPath);
    }

    // Conservative pruning (opt-in): opacity + floaters, with an optional explicit
    // box crop (advanced; splat-transform coordinate space).
    let cropInfo = null;
    let actions = [];
    if (opt.prune) {
        cropInfo = parseCropBox(opt.crop);
        actions = buildPruneActions({
            minOpacity: opt.minOpacity,
            cropBox: cropInfo && cropInfo.box,
            floaters: opt.floaters,
            floatersParams: opt.floatersParams,
        });
    }

    const t0 = Date.now();
    try {
        let code = await runCli(buildArgs(cliInput, actions, outPath, opt));
        if (code !== 0 && !opt.gpu) {
            // CPU fallback: SOG on CPU works, but the GPU-only floater filter can't.
            const cpuActions = opt.prune
                ? buildPruneActions({
                      minOpacity: opt.minOpacity,
                      cropBox: cropInfo && cropInfo.box,
                      floaters: false,
                  })
                : [];
            console.error(
                "[ply-to-sog] GPU encode failed — retrying with -g cpu" +
                    (opt.prune && opt.floaters ? " (floater filter skipped on CPU)" : ""),
            );
            code = await runCli(buildArgs(cliInput, cpuActions, outPath, opt, "cpu"));
        }
        if (code !== 0) process.exit(code);
    } finally {
        if (tempPath && !opt.keepTemp) fs.rmSync(tempPath, { force: true });
        else if (tempPath) console.error(`[ply-to-sog] kept temp: ${tempPath}`);
    }

    const inBytes = fs.statSync(inPath).size;
    const outBytes = outputBytes(outPath);
    console.log(
        JSON.stringify(
            {
                in: inPath,
                out: outPath,
                count_in: header.count, // post-prune count is printed by splat-transform above
                representation_in: cliInput === inPath && !tempPath ? "3dgs" : "2dgs",
                flattened_to_3dgs: !!tempPath,
                pruned: opt.prune,
                prune: opt.prune
                    ? {
                          min_opacity: opt.minOpacity,
                          crop: cropInfo ? { box: cropInfo.box, source: cropInfo.source } : null,
                          floaters: opt.floaters ? opt.floatersParams : false,
                      }
                    : null,
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
