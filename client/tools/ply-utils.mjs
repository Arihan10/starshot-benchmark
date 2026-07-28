// ply-utils.mjs — shared binary-PLY helpers for the splat side-scripts.
//
// A 2DGS-trained splat is scale_0/scale_1 only; the SOG and ksplat encoders are
// both 3DGS (three-scale) formats and don't recognize a 2-scale PLY as gaussians.
// `flattenFileIfNeeded` inserts a thin scale_2 (= ln(flatScale)) so the PLY reads
// as a valid, near-flat 3DGS splat (the overview §12 flatten).
//
// Both helpers work without the PLY resident: fs.readFileSync throws over 2 GiB,
// and a trained 3DGS scene clears that easily (13.5M gaussians × 62 float props =
// 3.3 GB). splat-transform reads the PLY lazily too, so nothing here needs more
// than the header plus one batch of gaussians.

import * as fs from "node:fs";

// parsePlyHeader never scans past this, so reading this much off the front sees
// exactly what a whole-file read would.
const HEADER_SCAN_BYTES = 1 << 16;

// Gaussians per flatten batch (~16 MB in, ~16 MB out at 62 properties).
const FLATTEN_BATCH = 1 << 16;

export function parsePlyHeader(buf) {
    const scan = buf.toString("latin1", 0, Math.min(buf.length, HEADER_SCAN_BYTES));
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

// The header of the PLY at `inPath`, read off the front of the file.
export function readPlyHeader(inPath) {
    const fd = fs.openSync(inPath, "r");
    try {
        const buf = Buffer.alloc(HEADER_SCAN_BYTES);
        const read = fs.readSync(fd, buf, 0, HEADER_SCAN_BYTES, 0);
        return parsePlyHeader(buf.subarray(0, read));
    } finally {
        fs.closeSync(fd);
    }
}

export function isTwoDgs(header) {
    return header.props.includes("scale_1") && !header.props.includes("scale_2");
}

// If `inPath` is 2DGS, stream a flattened 3DGS copy to `outPath` — scale_2
// (= ln(flatScale)) inserted after scale_1 — and return { flattened: true,
// header }. If it's already 3DGS, return { flattened: false } without writing
// anything (the caller uses the original file).
export function flattenFileIfNeeded(inPath, outPath, flatScale = 1e-4) {
    const header = readPlyHeader(inPath);
    if (!isTwoDgs(header)) return { flattened: false, header };

    const { count, props, headerBytes } = header;
    const P = props.length;
    const Pn = P + 1;
    const insertAt = props.indexOf("scale_1") + 1;
    const newProps = [...props.slice(0, insertAt), "scale_2", ...props.slice(insertAt)];
    const logFlat = Math.log(flatScale);
    const src = new Float32Array(FLATTEN_BATCH * P);
    const dst = new Float32Array(FLATTEN_BATCH * Pn);
    const srcBytes = Buffer.from(src.buffer);
    const dstBytes = Buffer.from(dst.buffer);

    const inFd = fs.openSync(inPath, "r");
    const outFd = fs.openSync(outPath, "w");
    try {
        fs.writeSync(
            outFd,
            Buffer.from(
                "ply\nformat binary_little_endian 1.0\n" +
                    `element vertex ${count}\n` +
                    newProps.map((p) => `property float ${p}\n`).join("") +
                    "end_header\n",
                "ascii",
            ),
        );
        for (let base = 0; base < count; base += FLATTEN_BATCH) {
            const batch = Math.min(FLATTEN_BATCH, count - base);
            const wanted = batch * P * 4;
            const read = fs.readSync(inFd, srcBytes, 0, wanted, headerBytes + base * P * 4);
            if (read !== wanted) {
                throw new Error(`PLY truncated at gaussian ${base}: read ${read}/${wanted} bytes`);
            }
            for (let i = 0; i < batch; i++) {
                const s = i * P;
                const d = i * Pn;
                dst.set(src.subarray(s, s + insertAt), d);
                dst[d + insertAt] = logFlat;
                dst.set(src.subarray(s + insertAt, s + P), d + insertAt + 1);
            }
            fs.writeSync(outFd, dstBytes, 0, batch * Pn * 4);
        }
    } finally {
        fs.closeSync(inFd);
        fs.closeSync(outFd);
    }
    return { flattened: true, header };
}
