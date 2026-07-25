#!/usr/bin/env node
/**
 * Offline asset-library optimizer (the "Layer 1" precompute pass).
 *
 * Reads the heavy Trellis GLBs from <input> and writes drastically smaller
 * GLBs to <output> — a NEW folder. The live pipeline keeps pointing at the
 * original `assets/`, so nothing in production breaks while this is validated;
 * adopting the optimized set later is a one-line ASSETS_DIR change.
 *
 * Per asset the geometry is welded, quadric-decimated (Meshopt), pruned, its
 * textures are downscaled and re-encoded to KTX2/Basis (or WebP), and the
 * geometry is finally Meshopt-compressed + quantized:
 *
 *     weld -> simplify -> dedup/prune -> resize + KTX2 (or WebP) -> meshopt
 *
 * A ~488K-triangle / ~22 MB asset typically lands around ~15K triangles /
 * a few hundred KB, which is what fixes both the streaming size and the
 * per-placement VRAM that currently makes large scenes fail to load.
 *
 * It also writes optimize_manifest.json carrying each asset's post-optimization
 * world-space AABB. The later per-placement "bake" derives its rescale matrix
 * from that AABB, and the optimized GLB uses Meshopt/quantized geometry that
 * trimesh (Python) can't read — so bounds are captured here, where the mesh
 * is already decoded.
 *
 * The <id>.png reference images are copied across unchanged so the output is a
 * complete drop-in for `assets/` (both the generation step and the viewer's
 * detail preview look for <id>.png beside <id>.glb).
 *
 * Textures default to KTX2/Basis ETC1S — GPU-compressed, so they stay small in
 * VRAM (~4-8x less than the RGBA8 a WebP/PNG decodes to), not just on disk.
 * The viewer must wire KTX2Loader to display them (the online three-gltf-viewer
 * reads them today). `--texture-format webp` falls back to plain WebP, which the
 * current viewer already renders but which decodes to full RGBA8 in VRAM.
 *
 * Three presets pick the tuning knobs (all run the identical transform chain):
 *   optimized  web-streaming tier — aggressive decimation + small ETC1S maps.
 *              Tiny, but visibly degraded. -> assets-optimized/  (default)
 *   lite       presentation tier (ported from the Unity replay) — near-lossless,
 *              error-bounded geometry + high-res UASTC. Visually identical to the
 *              raw asset at ~5x smaller / ~10x less texture VRAM. -> assets-lite/
 *   splat      splat-pipeline render/sample tier — optimized-grade geometry (the
 *              error bound does the visually-safe decimation) with lite-grade
 *              1024px base-color maps, ETC1S so they stay GPU-compressed in VRAM,
 *              and every non-base-color map STRIPPED (the splat pipeline is
 *              unlit: albedo + alpha is all it reads). -> assets-splat/
 * Explicit flags (--target-tris, --texture-size, --uastc/--no-uastc, ...)
 * override the chosen preset.
 *
 * Usage:
 *   npm install
 *   node optimize.mjs                          # optimized preset (default), all assets
 *   node optimize.mjs --preset lite            # lite preset -> assets-lite/
 *   node optimize.mjs --limit 3                 # smoke test on first 3 assets
 *   node optimize.mjs --preset lite --texture-size 2048   # override a preset knob
 *   node optimize.mjs --texture-format webp     # WebP instead of KTX2
 *   node optimize.mjs --force                   # re-optimize even if output exists
 *   node optimize.mjs --input <dir> --output <dir>
 *   node optimize.mjs --preset lite --file <in.glb> --out-file <out.glb>  # single file
 */

import { NodeIO, Logger } from "@gltf-transform/core";
import { ALL_EXTENSIONS } from "@gltf-transform/extensions";
import {
  dedup,
  dequantize,
  weld,
  simplify,
  prune,
  textureCompress,
  meshopt,
  getBounds,
} from "@gltf-transform/functions";
import { ktx2 } from "ktx2-encoder/gltf-transform";
import { MeshoptEncoder, MeshoptDecoder, MeshoptSimplifier } from "meshoptimizer";
import sharp from "sharp";
import os from "node:os";
import path from "node:path";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { mkdir, readdir, stat, copyFile, writeFile } from "node:fs/promises";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_INPUT = path.resolve(HERE, "../../app/assets_library/assets");
const OUTPUT_DIRS = {
  optimized: path.resolve(HERE, "../../app/assets_library/assets-optimized"),
  lite: path.resolve(HERE, "../../app/assets_library/assets-lite"),
  splat: path.resolve(HERE, "../../app/assets_library/assets-splat"),
};

// Named tuning presets. All run the IDENTICAL transform chain (weld ->
// simplify -> prune -> resize + KTX2 -> meshopt); only these knobs differ.
//   optimized  web-streaming tier: aggressive decimation + small ETC1S maps.
//              Tiny on disk, but visibly degraded (the original default).
//   lite       presentation tier ported from the Unity replay: error-bounded
//              (near-lossless) geometry + high-res UASTC, so it stays visually
//              identical to the raw asset while landing ~5x smaller on disk and
//              ~10x lighter in texture VRAM. `targetTris` is a high ceiling —
//              `error` does the (visually safe) decimation, since Trellis
//              massively over-tessellates flat regions.
//   splat      splat-pipeline render/sample tier (see splat/overview.md §5):
//              geometry decimated like `optimized` (in practice the error bound,
//              not the ceiling, is what stops the simplifier on Trellis meshes)
//              but with full-res 1024px base color kept, encoded ETC1S so a whole
//              scene's textures stay GPU-compressed (~0.5-1 B/texel in VRAM vs 16
//              for decoded float RGBA). stripMaps drops normal/MR/emissive/
//              occlusion up front — the splat pipeline is unlit and reads only
//              base color + alpha, so they'd waste encode time, disk, and VRAM.
const PRESETS = {
  optimized: { targetTris: 15000, error: 0.01, textureSize: 256, uastc: false },
  lite: { targetTris: 150000, error: 0.005, textureSize: 1024, uastc: true },
  splat: { targetTris: 60000, error: 0.01, textureSize: 1024, uastc: false, stripMaps: true },
};

function parseArgs(argv) {
  const opts = {
    input: DEFAULT_INPUT,
    output: null, // resolved from the preset below unless --output is given
    file: null,
    outFile: null,
    preset: "optimized",
    textureFormat: "ktx2",
    concurrency: Math.min(4, Math.max(1, os.cpus().length)),
    force: false,
    pngs: true,
    limit: Infinity,
    proxy: false,
  };
  // Tuning knobs default to the chosen preset's value; an explicit flag wins.
  const override = {};
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => argv[++i];
    switch (arg) {
      case "--input": opts.input = path.resolve(next()); break;
      case "--output": opts.output = path.resolve(next()); break;
      case "--file": opts.file = path.resolve(next()); break;
      case "--out-file": opts.outFile = path.resolve(next()); break;
      case "--preset": opts.preset = next(); break;
      case "--target-tris": override.targetTris = Number(next()); break;
      case "--error": override.error = Number(next()); break;
      case "--texture-size": override.textureSize = Number(next()); break;
      case "--texture-format": opts.textureFormat = next(); break;
      case "--uastc": override.uastc = true; break;
      case "--no-uastc": override.uastc = false; break;
      case "--strip-maps": override.stripMaps = true; break;
      case "--no-strip-maps": override.stripMaps = false; break;
      case "--concurrency": opts.concurrency = Math.max(1, Number(next())); break;
      case "--limit": opts.limit = Number(next()); break;
      case "--force": opts.force = true; break;
      case "--no-pngs": opts.pngs = false; break;
      // Geometry-only decimation for a projection proxy: skip all texture work,
      // strip materials, and emit a plain (un-meshopt) GLB any GLTFLoader reads.
      case "--proxy": opts.proxy = true; break;
      default:
        throw new Error(`unknown argument: ${arg}`);
    }
  }
  const preset = PRESETS[opts.preset];
  if (!preset) {
    throw new Error(
      `--preset must be one of ${Object.keys(PRESETS).join(", ")}, got: ${opts.preset}`,
    );
  }
  opts.targetTris = override.targetTris ?? preset.targetTris;
  opts.error = override.error ?? preset.error;
  opts.textureSize = override.textureSize ?? preset.textureSize;
  opts.uastc = override.uastc ?? preset.uastc;
  opts.stripMaps = override.stripMaps ?? preset.stripMaps ?? false;
  if (!opts.output) opts.output = OUTPUT_DIRS[opts.preset] ?? OUTPUT_DIRS.optimized;
  if (opts.textureFormat !== "ktx2" && opts.textureFormat !== "webp") {
    throw new Error(`--texture-format must be 'ktx2' or 'webp', got: ${opts.textureFormat}`);
  }
  return opts;
}

// Transforms (weld/prune/...) log at INFO by default, which floods a
// full-library run; keep only real errors.
const QUIET_LOGGER = new Logger(Logger.Verbosity.ERROR);

// KTX2/Basis encoding needs raw RGBA in Node (gltf-transform hands the encoder
// the compressed image bytes), so decode each texture with sharp. ensureAlpha
// guarantees the 4-channel R,G,B,A layout the Basis encoder expects.
const sharpImageDecoder = async (buffer) => {
  const { data, info } = await sharp(buffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { data, width: info.width, height: info.height };
};

const MB = (bytes) => (bytes / 1024 / 1024).toFixed(2) + "MB";
const kT = (tris) => (tris / 1000).toFixed(0) + "k";

function triangleCount(document) {
  let tris = 0;
  for (const mesh of document.getRoot().listMeshes()) {
    for (const prim of mesh.listPrimitives()) {
      const indices = prim.getIndices();
      const position = prim.getAttribute("POSITION");
      const count = indices ? indices.getCount() : position ? position.getCount() : 0;
      tris += Math.floor(count / 3);
    }
  }
  return tris;
}

function sceneBounds(document) {
  const scene = document.getRoot().getDefaultScene() ?? document.getRoot().listScenes()[0];
  if (!scene) return null;
  const { min, max } = getBounds(scene);
  return { min, max };
}

// The actual optimization pass for one GLB (read -> weld/decimate/prune ->
// resize + KTX2 -> meshopt -> write). Shared by the batch library pass and the
// per-asset generate-gate pass so both go through the identical pipeline.
async function optimizeFile(io, inPath, outPath, opts) {
  const document = await io.read(inPath);
  document.setLogger(QUIET_LOGGER);
  const srcTris = triangleCount(document);
  const ratio = Math.min(1, opts.targetTris / Math.max(1, srcTris));

  // Projection proxy (--proxy): strip materials + every non-POSITION attribute
  // UP FRONT. The proxy is re-textured by reprojecting the panos and gets fresh
  // normals in the viewer, so it needs geometry only — and dropping UV/normal
  // seams is what lets the simplifier actually reach a low triangle budget
  // (attribute discontinuities otherwise lock most edges, plateauing ~half the
  // source).
  if (opts.proxy) {
    for (const mesh of document.getRoot().listMeshes()) {
      for (const prim of mesh.listPrimitives()) {
        prim.setMaterial(null);
        for (const semantic of prim.listSemantics()) {
          if (semantic !== "POSITION") prim.setAttribute(semantic, null);
        }
      }
    }
  }

  // Unlit tiers (preset splat) keep only base color + alpha: detach the other
  // PBR maps up front so the prune() below drops their textures before the
  // (expensive) resize + KTX2 encode ever sees them.
  if (opts.stripMaps) {
    for (const material of document.getRoot().listMaterials()) {
      material.setNormalTexture(null);
      material.setMetallicRoughnessTexture(null);
      material.setEmissiveTexture(null);
      material.setOcclusionTexture(null);
    }
  }

  const simplifyOpts = { simplifier: MeshoptSimplifier, ratio, error: opts.error };
  // Proxy: keep open boundaries pinned so floor/wall perimeters and the open
  // edges of non-watertight Trellis shells can't collapse inward into gaps.
  if (opts.proxy) simplifyOpts.lockBorder = true;

  await document.transform(dedup(), weld(), simplify(simplifyOpts), prune());

  // Measure on un-quantized geometry; meshopt quantization would shift bounds
  // by sub-millimetre amounts, well under the rescale step's 1e-3 tolerance.
  const outTris = triangleCount(document);
  const bounds = sceneBounds(document);

  // Proxy: emit a PLAIN GLB regardless of how the source was compressed — undo
  // quantization and detach any geometry/texture-compression extensions so the
  // /pano walkthrough loads it with a bare GLTFLoader (no Meshopt/KTX2/Draco).
  if (opts.proxy) {
    await document.transform(dequantize());
    for (const ext of document.getRoot().listExtensionsUsed()) {
      if (/meshopt|quantization|basisu|draco/i.test(ext.extensionName)) {
        ext.dispose();
      }
    }
    await io.write(outPath, document);
    return { srcTris, outTris, bounds };
  }

  // Downscale every texture, then (default) re-encode to GPU-compressed KTX2.
  // For the KTX2 path the resize target is lossless PNG so ETC1S isn't stacked
  // on WebP's lossy pass; the WebP path keeps WebP as the final on-disk format.
  await document.transform(
    textureCompress({
      encoder: sharp,
      targetFormat: opts.textureFormat === "ktx2" ? "png" : "webp",
      resize: [opts.textureSize, opts.textureSize],
    }),
  );
  if (opts.textureFormat === "ktx2") {
    await document.transform(
      ktx2({ isUASTC: opts.uastc, imageDecoder: sharpImageDecoder, generateMipmap: true }),
    );
  }

  await document.transform(meshopt({ encoder: MeshoptEncoder, level: "high" }));
  await io.write(outPath, document);

  return { srcTris, outTris, bounds };
}

async function optimizeOne(io, fileName, opts) {
  const inPath = path.join(opts.input, fileName);
  const outPath = path.join(opts.output, fileName);
  const id = fileName.replace(/\.glb$/i, "");
  const inBytes = (await stat(inPath)).size;

  if (!opts.force && existsSync(outPath)) {
    // Re-read the (small) optimized file so the manifest stays authoritative
    // on resume — bounds especially, which the bake step depends on.
    const doc = await io.read(outPath);
    return {
      id,
      status: "skipped",
      inBytes,
      outBytes: (await stat(outPath)).size,
      outTris: triangleCount(doc),
      bounds: sceneBounds(doc),
    };
  }

  const { srcTris, outTris, bounds } = await optimizeFile(io, inPath, outPath, opts);

  return {
    id,
    status: "done",
    inBytes,
    outBytes: (await stat(outPath)).size,
    srcTris,
    outTris,
    bounds,
  };
}

async function copyReferencePng(fileName, opts) {
  const png = fileName.replace(/\.glb$/i, ".png");
  const src = path.join(opts.input, png);
  const dst = path.join(opts.output, png);
  if (existsSync(src) && (opts.force || !existsSync(dst))) {
    await copyFile(src, dst);
  }
}

async function runPool(items, concurrency, worker) {
  const results = new Array(items.length);
  let cursor = 0;
  const runners = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (cursor < items.length) {
      const index = cursor++;
      results[index] = await worker(items[index], index);
    }
  });
  await Promise.all(runners);
  return results;
}

async function makeIO() {
  await Promise.all([MeshoptDecoder.ready, MeshoptEncoder.ready, MeshoptSimplifier.ready]);
  return new NodeIO()
    .registerExtensions(ALL_EXTENSIONS)
    .registerDependencies({
      "meshopt.encoder": MeshoptEncoder,
      "meshopt.decoder": MeshoptDecoder,
    });
}

async function main() {
  const opts = parseArgs(process.argv);

  // Single-file mode: optimize one GLB to --out-file. The generate gate calls
  // this per freshly generated asset so it gets the same decimate + KTX2 +
  // Meshopt pass as the library. No manifest is written — generated meshes are
  // already placed in world space, so the bake step (which needs the manifest's
  // per-orientation bounds) doesn't apply.
  if (opts.file) {
    if (!existsSync(opts.file)) throw new Error(`input file not found: ${opts.file}`);
    if (!opts.outFile) throw new Error("--file requires --out-file");
    await mkdir(path.dirname(opts.outFile), { recursive: true });
    const io = await makeIO();
    const r = await optimizeFile(io, opts.file, opts.outFile, opts);
    console.log(`[opt] ${path.basename(opts.file)}: ${kT(r.srcTris)} -> ${kT(r.outTris)} tris`);
    return;
  }

  if (!existsSync(opts.input)) {
    throw new Error(`input dir not found: ${opts.input}`);
  }
  await mkdir(opts.output, { recursive: true });
  const io = await makeIO();

  let files = (await readdir(opts.input)).filter((f) => /\.glb$/i.test(f)).sort();
  if (Number.isFinite(opts.limit)) files = files.slice(0, opts.limit);

  console.log(`[opt] ${files.length} assets`);
  console.log(`[opt] in:  ${opts.input}`);
  console.log(`[opt] out: ${opts.output}`);
  const texDesc =
    opts.textureFormat === "ktx2" ? `ktx2/${opts.uastc ? "uastc" : "etc1s"}` : "webp";
  console.log(
    `[opt] preset ${opts.preset}: target ~${kT(opts.targetTris)} tris (err ${opts.error}), ` +
      `${opts.textureSize}px ${texDesc}, ` +
      `concurrency ${opts.concurrency}${opts.force ? ", force" : ""}`,
  );

  let done = 0;
  const results = await runPool(files, opts.concurrency, async (fileName) => {
    let entry;
    try {
      entry = await optimizeOne(io, fileName, opts);
      if (opts.pngs) await copyReferencePng(fileName, opts);
    } catch (err) {
      entry = { id: fileName.replace(/\.glb$/i, ""), status: "error", message: String(err?.message ?? err) };
    }
    done++;
    const tag = `(${done}/${files.length})`;
    if (entry.status === "done") {
      console.log(
        `[opt] ${tag} done  ${entry.id}  ${MB(entry.inBytes)} -> ${MB(entry.outBytes)}  ` +
          `${kT(entry.srcTris)} -> ${kT(entry.outTris)} tris`,
      );
    } else if (entry.status === "skipped") {
      console.log(`[opt] ${tag} skip  ${entry.id} (exists)`);
    } else {
      console.error(`[opt] ${tag} ERROR ${entry.id}: ${entry.message}`);
    }
    return entry;
  });

  const ok = results.filter((r) => r.status === "done");
  const errors = results.filter((r) => r.status === "error");
  const totalIn = results.reduce((s, r) => s + (r.inBytes ?? 0), 0);
  const totalOut = results.reduce((s, r) => s + (r.outBytes ?? 0), 0);

  await writeFile(
    path.join(opts.output, "optimize_manifest.json"),
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        options: {
          preset: opts.preset,
          targetTris: opts.targetTris,
          error: opts.error,
          textureSize: opts.textureSize,
          textureFormat: opts.textureFormat,
          uastc: opts.uastc,
        },
        assets: results,
      },
      null,
      2,
    ),
  );

  console.log(
    `\n[opt] finished: ${ok.length} optimized, ` +
      `${results.length - ok.length - errors.length} skipped, ${errors.length} errors`,
  );
  if (totalIn > 0) {
    console.log(
      `[opt] size: ${MB(totalIn)} -> ${MB(totalOut)} ` +
        `(${(totalIn / Math.max(1, totalOut)).toFixed(1)}x smaller)`,
    );
  }
  console.log(`[opt] manifest -> ${path.join(opts.output, "optimize_manifest.json")}`);
  if (errors.length) process.exitCode = 1;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
