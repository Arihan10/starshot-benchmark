import { NodeIO } from "@gltf-transform/core";
import { ALL_EXTENSIONS } from "@gltf-transform/extensions";
import { MeshoptDecoder } from "meshoptimizer";
import { readdirSync } from "node:fs";
import { join } from "node:path";

const dir = process.argv[2];
await MeshoptDecoder.ready;
const io = new NodeIO()
    .registerExtensions(ALL_EXTENSIONS)
    .registerDependencies({ "meshopt.decoder": MeshoptDecoder });

const files = readdirSync(dir).filter((f) => f.endsWith(".glb") && !f.includes(".raw."));
let ok = 0, fail = 0;
const fails = [];
let noPrim = 0, noPos = 0;
for (const f of files) {
    try {
        const doc = await io.read(join(dir, f));
        const meshes = doc.getRoot().listMeshes();
        let prims = 0, positions = 0, verts = 0;
        for (const m of meshes) {
            for (const p of m.listPrimitives()) {
                prims++;
                const pos = p.getAttribute("POSITION");
                if (pos) { positions++; verts += pos.getCount(); }
            }
        }
        if (prims === 0) noPrim++;
        if (positions === 0) noPos++;
        ok++;
        if (ok <= 3 || prims === 0 || positions === 0)
            console.log(`OK   ${f}  meshes=${meshes.length} prims=${prims} verts=${verts}`);
    } catch (e) {
        fail++;
        fails.push(`${f}: ${e.message || e}`);
    }
}
console.log(`\n=== ${ok} ok, ${fail} fail, ${noPrim} zero-primitive, ${noPos} zero-position (of ${files.length}) ===`);
for (const x of fails.slice(0, 15)) console.log("FAIL", x);
