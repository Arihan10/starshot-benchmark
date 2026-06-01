#!/usr/bin/env node
/**
 * Backfills optimize_manifest.json with each asset's world-space AABB under
 * every allowed yaw orientation.
 *
 * Why this exists: the server "bake" fits each placement's mesh into its target
 * bbox by scaling the *rotated* mesh's extents to the bbox extents (per axis,
 * matching the original rescale_mesh_to_bbox contract). The optimized GLBs use
 * Meshopt/quantized geometry that the Python pipeline can't decode, so the
 * rotated bounds are precomputed here — in Node, where gltf-transform can read
 * them — and consumed at bake time.
 *
 * The bounds are measured exactly the way the client will place the mesh: a
 * parent node carrying the yaw rotation, measured with getBounds (which applies
 * the GLB's own quantization node transforms). That keeps this consistent with
 * the bake's 2-node placement (outer = scale + translate, inner = yaw).
 *
 * Usage:
 *   node augment-bounds.mjs              # whole manifest
 *   node augment-bounds.mjs --limit 3     # first 3 assets (validation)
 */

import { NodeIO } from "@gltf-transform/core";
import { ALL_EXTENSIONS } from "@gltf-transform/extensions";
import { getBounds } from "@gltf-transform/functions";
import { MeshoptDecoder, MeshoptEncoder } from "meshoptimizer";
import path from "node:path";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { readFile, writeFile } from "node:fs/promises";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(HERE, "../../app/assets_library/assets-optimized");

// The Orientation enum (degrees). -180 and 180 are the same rotation but both
// are kept so a lookup by the emitted value can never miss.
const ORIENTATIONS = [-180, -135, -90, -45, 0, 45, 90, 135, 180];

// glTF quaternion [x,y,z,w] for a right-handed yaw about +Y by `deg`. Matches
// trimesh.rotation_matrix(radians(deg), [0,1,0]) and three.js setFromAxisAngle
// so bounds, bake, and client all agree on the rotation.
function quatY(deg) {
  const half = (deg * Math.PI) / 180 / 2;
  return [0, Math.sin(half), 0, Math.cos(half)];
}

const round = (v) => Math.round(v * 1e5) / 1e5;

function boundsByOrientation(doc) {
  const root = doc.getRoot();
  const scene = root.getDefaultScene() ?? root.listScenes()[0];
  if (!scene) return null;
  const baseRoots = scene.listChildren();
  if (baseRoots.length === 0) return null;
  const out = {};
  for (const deg of ORIENTATIONS) {
    const wrap = doc.createNode(`yaw_${deg}`).setRotation(quatY(deg));
    for (const n of baseRoots) {
      scene.removeChild(n);
      wrap.addChild(n);
    }
    scene.addChild(wrap);
    const { min, max } = getBounds(scene);
    out[String(deg)] = { min: min.map(round), max: max.map(round) };
    // Restore the scene graph for the next orientation.
    for (const n of baseRoots) {
      wrap.removeChild(n);
      scene.addChild(n);
    }
    scene.removeChild(wrap);
    wrap.dispose();
  }
  return out;
}

async function main() {
  const limitArg = process.argv.indexOf("--limit");
  const limit = limitArg !== -1 ? Number(process.argv[limitArg + 1]) : Infinity;

  await Promise.all([MeshoptDecoder.ready, MeshoptEncoder.ready]);
  const io = new NodeIO()
    .registerExtensions(ALL_EXTENSIONS)
    .registerDependencies({
      "meshopt.encoder": MeshoptEncoder,
      "meshopt.decoder": MeshoptDecoder,
    });

  const manifestPath = path.join(OUT, "optimize_manifest.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  const assets = manifest.assets ?? [];

  let done = 0;
  let skipped = 0;
  let errors = 0;
  for (const entry of assets) {
    if (done >= limit) break;
    const glb = path.join(OUT, `${entry.id}.glb`);
    if (entry.status === "error" || !existsSync(glb)) {
      skipped++;
      continue;
    }
    try {
      const doc = await io.read(glb);
      const bounds = boundsByOrientation(doc);
      if (bounds) entry.bounds_by_orientation = bounds;
      done++;
      if (done % 100 === 0) console.log(`[aug] ${done} processed`);
    } catch (e) {
      entry.bounds_error = String(e?.message ?? e);
      errors++;
      console.error(`[aug] ERROR ${entry.id}: ${entry.bounds_error}`);
    }
  }

  manifest.bounds_orientation_added_at = new Date().toISOString();
  await writeFile(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`[aug] done=${done} skipped=${skipped} errors=${errors}`);
  console.log(`[aug] manifest -> ${manifestPath}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
