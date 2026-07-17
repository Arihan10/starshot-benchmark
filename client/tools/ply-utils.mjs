// ply-utils.mjs — shared binary-PLY helpers for the splat side-scripts.
//
// The trained splat is 2DGS (scale_0/scale_1 only); the SOG and ksplat encoders
// are both 3DGS (three-scale) formats and don't recognize a 2-scale PLY as
// gaussians. `flattenTo3dgs` inserts a thin scale_2 (= ln(flatScale)) so the PLY
// reads as a valid, near-flat 3DGS splat (the overview §12 flatten).

import * as fs from "node:fs";

// A clean, 4-byte-aligned ArrayBuffer of exactly this Buffer's bytes.
export function toArrayBuffer(buf) {
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

export function parsePlyHeader(buf) {
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

export function isTwoDgs(header) {
    return header.props.includes("scale_1") && !header.props.includes("scale_2");
}

// Insert a thin scale_2 (= ln(flatScale)) after scale_1, turning a 2DGS PLY into
// a valid 3DGS PLY the SOG/ksplat encoders accept. Returns a new Buffer.
export function flattenTo3dgs(buf, header, flatScale = 1e-4) {
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

// Read `inPath`; if it's 2DGS, write a flattened 3DGS copy to `outPath` and
// return { flattened: true, header }. If already 3DGS, return { flattened: false }
// without writing (the caller uses the original file).
export function flattenFileIfNeeded(inPath, outPath, flatScale = 1e-4) {
    const buf = fs.readFileSync(inPath);
    const header = parsePlyHeader(buf);
    if (!isTwoDgs(header)) return { flattened: false, header };
    fs.writeFileSync(outPath, flattenTo3dgs(buf, header, flatScale));
    return { flattened: true, header };
}
