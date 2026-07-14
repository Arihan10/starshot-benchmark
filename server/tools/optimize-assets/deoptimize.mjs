#!/usr/bin/env node
/**
 * De-optimize splat-source GLBs back to vanilla glTF the Python (trimesh) Stage-1
 * assembler can read.
 *
 * The asset-library build's served meshes (objects-optimized/) are stored with
 * EXT_meshopt_compression + KHR_mesh_quantization + KHR_texture_basisu (KTX2) —
 * none of which trimesh can decode. This rewrites each to a vanilla glTF:
 *   - EXT_meshopt_compression   → decoded on read (MeshoptDecoder registered)
 *   - KHR_mesh_quantization     → removed via dequantize() (integer → float)
 *   - KHR_texture_basisu (KTX2) → base-color transcoded to PNG (Basis → RGBA →
 *                                 PNG) so trimesh reads real albedo and Stage 2
 *                                 samples per-texel colour; the other PBR maps
 *                                 (normal/metallic-roughness/emissive/occlusion),
 *                                 unused by the unlit surfel pass, are dropped
 *
 * Placement is baked into vertices (see bakeWorldTransforms): meshopt stores the
 * room placement as an ancestor node transform that trimesh doesn't compose, so
 * we fold local→world into the vertex data, leaving world-space geometry at
 * identity nodes — matching the generated build. Faces are preserved exactly (no
 * decimation/welding/reindexing) so the assembler's world-AABB checks stay valid.
 * Materials are kept (baseColorFactor, alphaMode) so opacity/PBR flags survive.
 *
 * Usage:
 *   node deoptimize.mjs --in-dir <optimized_dir> --out-dir <vanilla_dir>
 *   node deoptimize.mjs --file <in.glb> --out-file <out.glb>
 */

import { NodeIO, Logger } from "@gltf-transform/core";
import { ALL_EXTENSIONS } from "@gltf-transform/extensions";
import { dequantize, prune, transformMesh } from "@gltf-transform/functions";
import { MeshoptDecoder, MeshoptEncoder } from "meshoptimizer";
import sharp from "sharp";
import path from "node:path";
import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { mkdir, readdir } from "node:fs/promises";

function parseArgs(argv) {
  const o = { inDir: null, outDir: null, file: null, outFile: null };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => argv[++i];
    switch (arg) {
      case "--in-dir": o.inDir = path.resolve(next()); break;
      case "--out-dir": o.outDir = path.resolve(next()); break;
      case "--file": o.file = path.resolve(next()); break;
      case "--out-file": o.outFile = path.resolve(next()); break;
      default: throw new Error(`unknown argument: ${arg}`);
    }
  }
  return o;
}

// Transforms log at INFO by default, which floods a whole-cell run; keep errors.
const QUIET_LOGGER = new Logger(Logger.Verbosity.ERROR);

// Extension DECLARATIONS to strip after decoding. Even once the geometry is
// decoded to standard accessors and the KTX2 textures are pruned, gltf-transform
// leaves these in extensionsUsed/Required — and trimesh refuses to load a file
// whose extensionsRequired names anything it can't decode. Disposing them makes
// the output truly vanilla glTF.
const DROP_EXTENSIONS = new Set([
  "EXT_meshopt_compression",
  "KHR_mesh_quantization",
  "KHR_texture_basisu",
]);

async function makeIO() {
  await Promise.all([MeshoptDecoder.ready, MeshoptEncoder.ready]);
  return new NodeIO()
    .registerExtensions(ALL_EXTENSIONS)
    .registerDependencies({
      "meshopt.decoder": MeshoptDecoder,
      "meshopt.encoder": MeshoptEncoder,
    });
}

// --- KTX2/Basis base-color → PNG -------------------------------------------
// The library's textures are KTX2 (Basis ETC1S/UASTC), which trimesh can't
// decode. We transcode the base-color map to RGBA with the vendored Basis
// transcoder (three.js's) and re-embed it as PNG (via sharp), so Stage 2 samples
// real per-texel albedo. The transcoder WASM is instantiated once and reused.
const require = createRequire(import.meta.url);
const BASIS_RGBA32 = 13; // basis transcoder_texture_format cTFRGBA32

let _basisPromise = null;
function getBasis() {
  if (!_basisPromise) {
    const BASIS = require("./vendor/basis/basis_transcoder.cjs");
    const wasmBinary = readFileSync(
      new URL("./vendor/basis/basis_transcoder.wasm", import.meta.url),
    );
    _basisPromise = BASIS({ wasmBinary }).then((Module) => {
      Module.initializeBasis();
      return Module;
    });
  }
  return _basisPromise;
}

// Decode a KTX2 image (level 0) to a PNG buffer, or null if it can't be read.
async function ktx2ToPng(ktx2Bytes) {
  const Module = await getBasis();
  const file = new Module.KTX2File(new Uint8Array(ktx2Bytes));
  try {
    if (!file.isValid() || !file.startTranscoding()) return null;
    const width = file.getWidth();
    const height = file.getHeight();
    const rgba = new Uint8Array(
      file.getImageTranscodedSizeInBytes(0, 0, 0, BASIS_RGBA32),
    );
    if (!file.transcodeImage(rgba, 0, 0, 0, BASIS_RGBA32, 0, -1, -1)) return null;
    return await sharp(Buffer.from(rgba), {
      raw: { width, height, channels: 4 },
    })
      .png()
      .toBuffer();
  } finally {
    file.close();
    file.delete();
  }
}

// Transcode each material's base-color KTX2 → PNG in place (real albedo Stage 2
// samples per-texel) and drop the PBR maps the unlit pass doesn't use. Textures
// are shared, so each is transcoded once. A texture that won't transcode is
// dropped so the mesh still loads (falling back to baseColorFactor / grey).
// baseColorFactor + alphaMode are kept, so opacity/PBR flags survive.
async function transcodeMaterials(document) {
  const seen = new Map(); // Texture -> "ok" | "fail"
  for (const material of document.getRoot().listMaterials()) {
    material.setNormalTexture(null);
    material.setMetallicRoughnessTexture(null);
    material.setEmissiveTexture(null);
    material.setOcclusionTexture(null);
    const tex = material.getBaseColorTexture();
    if (!tex) continue;
    if (!seen.has(tex)) {
      let status = "ok";
      if (tex.getMimeType() === "image/ktx2") {
        const png = await ktx2ToPng(tex.getImage());
        if (png) tex.setImage(png).setMimeType("image/png");
        else status = "fail";
      }
      seen.set(tex, status);
    }
    if (seen.get(tex) === "fail") material.setBaseColorTexture(null);
  }
}

// gltfpack/meshopt store an asset's ROOM PLACEMENT as an ancestor node transform
// (translation + non-uniform scale) sitting above the mesh node's own
// quantization scale. gltf-transform composes the whole ancestor chain, but
// trimesh applies only the mesh node's own transform — so a Python reader sees
// every asset normalized at the origin, losing placement (identical AABBs, all
// stacked). Bake the full local→world matrix into vertex data (transformMesh
// inverse-transposes normals for the non-uniform scale) and neutralize node
// transforms, yielding world-space geometry at identity nodes — exactly what the
// generated build already produces, so Stage 1/2 read placement directly.
// Assumes each mesh is instanced by a single node, which holds for these
// per-object GLBs (one mesh, one placement).
function bakeWorldTransforms(document) {
  const meshNodes = document
    .getRoot()
    .listNodes()
    .filter((n) => n.getMesh());
  // Snapshot every world matrix first — transformMesh only edits vertex data, not
  // node transforms, so ancestors stay valid across bakes; reset happens after.
  const worlds = meshNodes.map((n) => n.getWorldMatrix());
  meshNodes.forEach((node, i) => transformMesh(node.getMesh(), worlds[i]));
  for (const node of document.getRoot().listNodes()) {
    node.setTranslation([0, 0, 0]).setRotation([0, 0, 0, 1]).setScale([1, 1, 1]);
  }
}

async function deoptimizeFile(io, inPath, outPath) {
  const document = await io.read(inPath); // meshopt decoded here (decoder registered)
  document.setLogger(QUIET_LOGGER);
  await document.transform(dequantize()); // KHR_mesh_quantization → float accessors
  bakeWorldTransforms(document); // ancestor placement → vertices (trimesh-readable)
  await transcodeMaterials(document); // base-color KTX2 → PNG; drop other PBR maps
  await document.transform(prune()); // drop the now-orphaned dropped textures
  // Geometry is standard float and base-color is PNG, but the extension
  // declarations linger — dispose them so extensionsRequired is empty and trimesh
  // will load the file.
  for (const ext of document.getRoot().listExtensionsUsed()) {
    if (DROP_EXTENSIONS.has(ext.extensionName)) ext.dispose();
  }
  await io.write(outPath, document);
}

async function main() {
  const opts = parseArgs(process.argv);
  const io = await makeIO();

  if (opts.file) {
    if (!opts.outFile) throw new Error("--file requires --out-file");
    await mkdir(path.dirname(opts.outFile), { recursive: true });
    await deoptimizeFile(io, opts.file, opts.outFile);
    return;
  }

  if (!opts.inDir || !opts.outDir) {
    throw new Error("need --in-dir and --out-dir (or --file and --out-file)");
  }
  if (!existsSync(opts.inDir)) {
    throw new Error(`input dir not found: ${opts.inDir}`);
  }
  await mkdir(opts.outDir, { recursive: true });

  // Placed meshes only — the `<id>.glb` the viewer renders, not the
  // `<id>.raw.glb` pre-placement intermediates (mirrors the assembler).
  const files = (await readdir(opts.inDir)).filter(
    (f) => /\.glb$/i.test(f) && !/\.raw\.glb$/i.test(f),
  );

  let ok = 0;
  let errors = 0;
  for (const f of files) {
    try {
      await deoptimizeFile(io, path.join(opts.inDir, f), path.join(opts.outDir, f));
      ok++;
    } catch (err) {
      errors++;
      console.error(`[deopt] ${f}: ${err?.message ?? err}`);
    }
  }
  console.log(`[deopt] ${ok} ok, ${errors} error(s) → ${opts.outDir}`);
  // Fail the process only if nothing converted — a few bad meshes shouldn't sink
  // the whole cell (the assembler records those ids as holes).
  if (errors > 0 && ok === 0) process.exitCode = 1;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
