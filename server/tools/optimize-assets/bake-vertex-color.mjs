#!/usr/bin/env node
/**
 * Vertex-color scene bake — a tiny, fully-usable preview for scenes whose meshes
 * can't be decimated while textured.
 *
 * Some meshes (Trellis auto-UV output) shatter into thousands of tiny UV islands
 * — their split vertices lock the UV-preserving simplifier, so they can't shrink
 * while keeping a texture, and the sloppy simplifier smears the texture if you
 * force it. This sidesteps the conflict: per object, sample the texture into
 * per-vertex COLOR_0, drop the texture + UVs, THEN sloppy-decimate (safe now —
 * there's no UV mapping left to break) and merge. The output carries no textures
 * at all (just vertex colors), so it stays small; the only loss is per-vertex
 * (Gouraud) shading instead of full-resolution texture.
 *
 * Reads the RAW objects (PNG textures sharp can decode); the optimized twins are
 * KTX2/Basis, which would need a transcoder we don't have.
 *
 * Usage:
 *   node bake-vertex-color.mjs --inputs-dir <raw dir> --out-file <scene.glb>
 * Prints a JSON stats line last.
 */

import { Document, Logger, NodeIO } from "@gltf-transform/core";
import { ALL_EXTENSIONS } from "@gltf-transform/extensions";
import {
  dedup,
  dequantize,
  mergeDocuments,
  meshopt,
  prune,
  unpartition,
} from "@gltf-transform/functions";
import { MeshoptDecoder, MeshoptEncoder, MeshoptSimplifier } from "meshoptimizer";
import sharp from "sharp";
import path from "node:path";
import { mkdir, readdir, stat } from "node:fs/promises";

// Fixed bake configuration — the /lite export is one quality, no knobs.
const TARGET_TRIS = 3_000_000; // whole-scene triangle ceiling (vertex colors are cheap)
const SAMPLE_SIZE = 512; // texture downscaled to this before per-vertex sampling

function parseArgs(argv) {
  const opts = { inputsDir: null, outFile: null };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => argv[++i];
    switch (arg) {
      case "--inputs-dir": opts.inputsDir = path.resolve(next()); break;
      case "--out-file": opts.outFile = path.resolve(next()); break;
      default: throw new Error(`unknown argument: ${arg}`);
    }
  }
  if (!opts.outFile) throw new Error("--out-file is required");
  if (!opts.inputsDir) throw new Error("--inputs-dir is required");
  return opts;
}

const QUIET = new Logger(Logger.Verbosity.ERROR);
const ELEMENT_SIZE = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };
const MB = (b) => (b / 1048576).toFixed(2) + "MB";

function triangleCount(doc) {
  let tris = 0;
  for (const mesh of doc.getRoot().listMeshes())
    for (const prim of mesh.listPrimitives()) {
      const idx = prim.getIndices();
      const pos = prim.getAttribute("POSITION");
      tris += Math.floor((idx ? idx.getCount() : pos ? pos.getCount() : 0) / 3);
    }
  return tris;
}

async function makeIO() {
  await Promise.all([MeshoptDecoder.ready, MeshoptEncoder.ready, MeshoptSimplifier.ready]);
  return new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
    "meshopt.encoder": MeshoptEncoder,
    "meshopt.decoder": MeshoptDecoder,
  });
}

// Decode a glTF texture to a small tightly-packed RGBA buffer for fast sampling.
async function decodeImage(bytes, size) {
  const { data, info } = await sharp(Buffer.from(bytes))
    .resize(size, size, { fit: "fill" })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { data, w: info.width, h: info.height };
}

const srgbToLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

// Bilinear sample, repeat-wrapped, glTF UV convention (origin top-left, v down).
// Returns linear-light 0..1 RGB (glTF COLOR_0 is linear; the source PNG is sRGB).
function sampleLinear(img, u, v, out) {
  u -= Math.floor(u);
  v -= Math.floor(v);
  const x = u * (img.w - 1);
  const y = v * (img.h - 1);
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, img.w - 1);
  const y1 = Math.min(y0 + 1, img.h - 1);
  const fx = x - x0;
  const fy = y - y0;
  const i00 = (y0 * img.w + x0) * 4;
  const i10 = (y0 * img.w + x1) * 4;
  const i01 = (y1 * img.w + x0) * 4;
  const i11 = (y1 * img.w + x1) * 4;
  for (let c = 0; c < 3; c++) {
    const top = img.data[i00 + c] + (img.data[i10 + c] - img.data[i00 + c]) * fx;
    const bot = img.data[i01 + c] + (img.data[i11 + c] - img.data[i01 + c]) * fx;
    out[c] = srgbToLinear((top + (bot - top) * fy) / 255);
  }
}

// Sample each primitive's base-color texture into a COLOR_0 attribute, then
// strip the texture + UVs and matte the material so the vertex colors show.
async function bakeColors(doc, sampleSize) {
  const cache = new Map(); // texture -> decoded image
  for (const mesh of doc.getRoot().listMeshes()) {
    for (const prim of mesh.listPrimitives()) {
      const position = prim.getAttribute("POSITION");
      const uv = prim.getAttribute("TEXCOORD_0");
      if (!position) continue;
      const count = position.getCount();
      const material = prim.getMaterial();
      const factor = material ? material.getBaseColorFactor() : [1, 1, 1, 1];
      const tex = material ? material.getBaseColorTexture() : null;

      let img = null;
      if (tex && uv) {
        if (!cache.has(tex)) {
          try {
            cache.set(tex, await decodeImage(tex.getImage(), sampleSize));
          } catch {
            cache.set(tex, null);
          }
        }
        img = cache.get(tex);
      }

      const colors = new Uint8Array(count * 4);
      const uvArr = uv ? uv.getArray() : null;
      const rgb = [factor[0], factor[1], factor[2]];
      for (let v = 0; v < count; v++) {
        if (img && uvArr) sampleLinear(img, uvArr[v * 2], uvArr[v * 2 + 1], rgb);
        colors[v * 4] = Math.round(Math.min(1, rgb[0] * factor[0]) * 255);
        colors[v * 4 + 1] = Math.round(Math.min(1, rgb[1] * factor[1]) * 255);
        colors[v * 4 + 2] = Math.round(Math.min(1, rgb[2] * factor[2]) * 255);
        colors[v * 4 + 3] = 255;
      }
      const buffer = doc.getRoot().listBuffers()[0] ?? doc.createBuffer();
      prim.setAttribute(
        "COLOR_0",
        doc.createAccessor().setType("VEC4").setNormalized(true).setBuffer(buffer).setArray(colors),
      );
      // Drop UV streams; the bake replaced the texture mapping.
      for (const semantic of prim.listSemantics()) {
        if (semantic.startsWith("TEXCOORD")) prim.setAttribute(semantic, null);
      }
      if (material) {
        material
          .setBaseColorTexture(null)
          .setMetallicRoughnessTexture(null)
          .setNormalTexture(null)
          .setOcclusionTexture(null)
          .setEmissiveTexture(null)
          .setBaseColorFactor([1, 1, 1, 1])
          .setMetallicFactor(0)
          .setRoughnessFactor(1);
      }
    }
  }
}

// Sloppy-decimate each primitive (geometry only) to at most `cap` triangles,
// compacting COLOR_0 + POSITION around the survivors. Safe for vertex colors:
// there are no UVs left to scramble.
function decimateObject(doc, cap) {
  const buffer = doc.getRoot().listBuffers()[0] ?? doc.createBuffer();
  for (const mesh of doc.getRoot().listMeshes()) {
    for (const prim of mesh.listPrimitives()) {
      if (prim.getMode() !== 4) continue; // TRIANGLES
      const position = prim.getAttribute("POSITION");
      const indicesAcc = prim.getIndices();
      if (!position || !indicesAcc) continue;
      const pos = position.getArray();
      if (!(pos instanceof Float32Array)) continue;
      let idx = indicesAcc.getArray();
      if (!(idx instanceof Uint32Array)) idx = new Uint32Array(idx);

      const targetCount = Math.max(3, Math.min(idx.length, cap * 3));
      if (targetCount >= idx.length) continue;
      const [dstIdx] = MeshoptSimplifier.simplifySloppy(idx, pos, 3, null, targetCount, 1.0);
      if (dstIdx.length < 3 || dstIdx.length >= idx.length) continue;

      const vertCount = pos.length / 3;
      const oldToNew = new Int32Array(vertCount).fill(-1);
      let unique = 0;
      for (let i = 0; i < dstIdx.length; i++) {
        const v = dstIdx[i];
        if (oldToNew[v] === -1) oldToNew[v] = unique++;
      }
      for (const semantic of prim.listSemantics()) {
        const acc = prim.getAttribute(semantic);
        const comps = ELEMENT_SIZE[acc.getType()] ?? 1;
        const src = acc.getArray();
        const dst = new src.constructor(unique * comps);
        for (let v = 0; v < vertCount; v++) {
          const nv = oldToNew[v];
          if (nv === -1) continue;
          for (let c = 0; c < comps; c++) dst[nv * comps + c] = src[v * comps + c];
        }
        prim.setAttribute(
          semantic,
          doc.createAccessor().setType(acc.getType()).setNormalized(acc.getNormalized()).setBuffer(buffer).setArray(dst),
        );
        if (acc.listParents().length === 1) acc.dispose();
      }
      const reindexed = new Uint32Array(dstIdx.length);
      for (let i = 0; i < dstIdx.length; i++) reindexed[i] = oldToNew[dstIdx[i]];
      const idxArray = unique <= 65534 ? new Uint16Array(reindexed) : reindexed;
      prim.setIndices(doc.createAccessor().setType("SCALAR").setBuffer(buffer).setArray(idxArray));
      if (indicesAcc.listParents().length === 1) indicesAcc.dispose();
    }
  }
}

async function collectInputs(inputsDir) {
  const names = (await readdir(inputsDir))
    .filter((f) => /\.glb$/i.test(f) && !/\.raw\.glb$/i.test(f))
    .sort();
  return names.map((n) => path.join(inputsDir, n));
}

// Name each object's top-level node with its source id (the GLB's filename
// stem) so the merged scene — and any client that loads it — can name / address
// the individual objects. Multi-root sources collapse under one named wrapper.
function labelObject(doc, id) {
  for (const scene of doc.getRoot().listScenes()) {
    const roots = scene.listChildren();
    if (roots.length === 1) {
      roots[0].setName(id);
    } else if (roots.length > 1) {
      const wrap = doc.createNode(id);
      for (const child of roots) {
        scene.removeChild(child);
        wrap.addChild(child);
      }
      scene.addChild(wrap);
    }
  }
}

async function main() {
  const opts = parseArgs(process.argv);
  const files = await collectInputs(opts.inputsDir);
  if (files.length === 0) throw new Error("no input GLBs found");

  const io = await makeIO();
  const cap = Math.max(200, Math.floor(TARGET_TRIS / files.length)); // per-object triangle cap
  let srcTris = 0;

  const target = new Document();
  target.setLogger(QUIET);
  let skipped = 0;
  for (const file of files) {
    try {
      const src = await io.read(file);
      src.setLogger(QUIET);
      await src.transform(dequantize()); // float positions/UVs for sampling + simplify
      srcTris += triangleCount(src);
      await bakeColors(src, SAMPLE_SIZE);
      decimateObject(src, cap);
      labelObject(src, path.basename(file).replace(/\.glb$/i, ""));
      mergeDocuments(target, src);
    } catch (err) {
      // A single malformed object (e.g. a NaN-bounded Trellis mesh, whose JSON
      // chunk won't even parse) shouldn't sink the whole scene — skip it.
      skipped++;
      console.error(`[vcolor] skip ${path.basename(file)}: ${err.message}`);
    }
  }

  const scenes = target.getRoot().listScenes();
  if (scenes.length === 0) throw new Error("no readable objects (all inputs failed)");
  const mainScene = scenes[0];
  for (let i = 1; i < scenes.length; i++) {
    for (const child of scenes[i].listChildren()) mainScene.addChild(child);
    scenes[i].dispose();
  }
  mainScene.setName("scene");
  target.getRoot().setDefaultScene(mainScene);

  // dedup shares repeated placements of the same asset; prune drops orphans;
  // unpartition collapses buffers for the GLB; meshopt compresses.
  await target.transform(dedup(), prune(), unpartition(), meshopt({ encoder: MeshoptEncoder, level: "high" }));

  await mkdir(path.dirname(opts.outFile), { recursive: true });
  await io.write(opts.outFile, target);

  const outTris = triangleCount(target);
  const outBytes = (await stat(opts.outFile)).size;
  const baked = files.length - skipped;
  console.error(
    `[vcolor] ${baked}/${files.length} objects -> ${MB(outBytes)}, ${(srcTris / 1000).toFixed(0)}k -> ${(outTris / 1000).toFixed(0)}k tris`,
  );
  console.log(JSON.stringify({ objects: baked, skipped, srcTris, outTris, outBytes }));
}

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exit(1);
});
