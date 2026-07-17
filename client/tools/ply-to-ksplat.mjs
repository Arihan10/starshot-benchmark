#!/usr/bin/env node
// ply-to-ksplat.mjs — convert a trained splat PLY into mkkellogg's compressed
// .ksplat using the GaussianSplats3D library's OWN quantizer (no hand-rolled
// packing): PlyParser -> SplatBufferGenerator(compressionLevel) -> SplatBuffer.
//
// Compression levels (the built-in quantization):
//   0 = float32 (no compression)
//   1 = 16-bit position / scale / rotation / SH   (near-lossless, ~3x)
//   2 = level 1 + 8-bit SH coefficients            (SH0: 8-bit DC colour too)
//
// Our trained.ply is 2DGS (scale_0, scale_1 only); .ksplat / SplatBuffer is a
// 3DGS (three-scale) format, so a 2DGS input is flattened to 3DGS first by
// inserting a very thin scale_2 (= ln(flat_scale)). Pass --no-flatten to skip.
//
// Usage:
//   node tools/ply-to-ksplat.mjs <in.ply> <out.ksplat>
//        [--level 0|1|2] [--alpha N] [--sh N] [--flat-scale F] [--no-flatten]
//
// Runs from anywhere under client/ so `@mkkellogg/gaussian-splats-3d` and its
// `three` peer resolve from client/node_modules.

import * as fs from "node:fs";
import * as path from "node:path";
import * as GS from "@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js";

function parseArgs(argv) {
    const pos = [];
    const opt = { level: 1, alpha: 1, sh: 0, flatScale: 1e-6, flatten: true };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === "--no-flatten") opt.flatten = false;
        else if (a === "--level") opt.level = parseInt(argv[++i], 10);
        else if (a === "--alpha") opt.alpha = parseInt(argv[++i], 10);
        else if (a === "--sh") opt.sh = parseInt(argv[++i], 10);
        else if (a === "--flat-scale") opt.flatScale = parseFloat(argv[++i]);
        else pos.push(a);
    }
    return { pos, opt };
}

// A clean, 4-byte-aligned ArrayBuffer of exactly this Buffer's bytes.
function toArrayBuffer(buf) {
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

function parsePlyHeader(buf) {
    const scan = buf.toString("latin1", 0, Math.min(buf.length, 1 << 16));
    const end = scan.indexOf("end_header\n");
    if (end < 0) throw new Error("not a PLY (no end_header)");
    const headerBytes = end + "end_header\n".length;
    let count = null;
    const props = [];
    let inVertex = false;
    for (const line of scan.slice(0, end).split("\n")) {
        const p = line.trim().split(/\s+/);
        if (p[0] === "element" && p[1] === "vertex") {
            count = parseInt(p[2], 10);
            inVertex = true;
        } else if (p[0] === "element") {
            inVertex = false;
        } else if (inVertex && p[0] === "property") {
            if (p[1] !== "float" && p[1] !== "float32") {
                throw new Error(`non-float property '${p[p.length - 1]}' unsupported`);
            }
            props.push(p[p.length - 1]);
        }
    }
    if (count === null) throw new Error("PLY has no vertex element");
    return { count, props, headerBytes };
}

// Insert a thin scale_2 (= ln(flatScale)) after scale_1 so a 2DGS PLY becomes a
// valid 3DGS PLY that the ksplat parser accepts (the overview §12 flatten).
function flattenTo3dgs(buf, header, flatScale) {
    const { count, props, headerBytes } = header;
    const P = props.length;
    const insertAt = props.indexOf("scale_1") + 1;
    const src = new Float32Array(toArrayBuffer(buf.subarray(headerBytes)), 0, count * P);
    const Pn = P + 1;
    const out = new Float32Array(count * Pn);
    const logFlat = Math.log(flatScale);
    for (let i = 0; i < count; i++) {
        const s = i * P;
        const d = i * Pn;
        out.set(src.subarray(s, s + insertAt), d);
        out[d + insertAt] = logFlat;
        out.set(src.subarray(s + insertAt, s + P), d + insertAt + 1);
    }
    const newProps = [...props.slice(0, insertAt), "scale_2", ...props.slice(insertAt)];
    const headerStr =
        "ply\nformat binary_little_endian 1.0\n" +
        `element vertex ${count}\n` +
        newProps.map((n) => `property float ${n}\n`).join("") +
        "end_header\n";
    return Buffer.concat([
        Buffer.from(headerStr, "ascii"),
        Buffer.from(out.buffer, out.byteOffset, out.byteLength),
    ]);
}

function main() {
    const { pos, opt } = parseArgs(process.argv.slice(2));
    if (pos.length < 2) {
        console.error(
            "Usage: node tools/ply-to-ksplat.mjs <in.ply> <out.ksplat> " +
                "[--level 0|1|2] [--alpha N] [--sh N] [--flat-scale F] [--no-flatten]",
        );
        process.exit(1);
    }
    const [inPath, outPath] = pos;
    const fileData = fs.readFileSync(inPath);
    const header = parsePlyHeader(fileData);
    const is2dgs = header.props.includes("scale_1") && !header.props.includes("scale_2");

    let plyBuf = fileData;
    let flattened = false;
    if (is2dgs && opt.flatten) {
        plyBuf = flattenTo3dgs(fileData, header, opt.flatScale);
        flattened = true;
    }

    const splatArray = GS.PlyParser.parseToUncompressedSplatArray(toArrayBuffer(plyBuf), opt.sh);
    const generator = GS.SplatBufferGenerator.getStandardGenerator(opt.alpha, opt.level);
    const splatBuffer = generator.generateFromUncompressedSplatArray(splatArray);
    const outData = Buffer.from(splatBuffer.bufferData);

    fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });
    fs.writeFileSync(outPath, outData);

    const inBytes = fileData.length;
    const outBytes = outData.length;
    console.log(
        JSON.stringify(
            {
                in: inPath,
                out: outPath,
                count: header.count,
                representation_in: is2dgs ? "2dgs" : "3dgs",
                flattened_to_3dgs: flattened,
                compression_level: opt.level,
                sh_degree: opt.sh,
                in_bytes: inBytes,
                out_bytes: outBytes,
                ratio: +(inBytes / outBytes).toFixed(2),
                bytes_per_splat_out: +(outBytes / header.count).toFixed(2),
            },
            null,
            1,
        ),
    );
}

main();
