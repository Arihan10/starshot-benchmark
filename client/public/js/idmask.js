// =============================================================================
// idmask.js — per-pixel object ID masks for the matterport walkthrough.
// =============================================================================
//
// WHY. The walkthrough used to answer "which object is under the cursor?" by
// raycasting the projection proxy — but that proxy is the WHOLE scene decimated
// to 40 000 triangles (server/app/services/proxy.py), which on a 400-object cell
// is ~100 triangles per object. Its silhouettes don't match what you see, and
// decimation opens holes a ray passes straight through, so both the hover pick
// and the highlight it drives are wrong in ways that get worse the more complex
// the object is.
//
// WHAT INSTEAD. The capture worker is holding the FULL-resolution meshes, and
// the viewer is pinned to the capture point (a walkthrough only ever stands ON
// an anchor; it can't free-roam). So visibility from an anchor is a fixed
// function of view DIRECTION, known at bake time. We re-render the same six cube
// faces with a flat per-object colour, stitch them into an equirectangular ID
// plane in the same frame as the pano, and store it beside the JPEG. Hovering is
// then one array read and highlighting is one texture lookup — neither touches
// the proxy, and neither degrades as geometry gets complex.
//
// OCCLUSION IS BAKED, NOT ORDERED. The ID plane is rasterized with the same
// camera and its own depth buffer, so each pixel already names the object you
// can actually see there. There is no per-object mask ordering (a single order
// per anchor can't express mutual occlusion — a chair leg in front of a table in
// front of the chair's seat) and no runtime depth test against the proxy (which
// is the error we're removing). Two passes, matching what the colour pano's
// depth buffer does exactly:
//   1. layer 0            — the opaque scene.
//   2. OIT_LAYER, cutout  — transparent meshes, keeping only fragments with
//                           α ≥ OIT_OPAQUE. That is precisely the rule oit.js's
//                           depth pre-pass uses, so window frames and mullions
//                           claim their pixels while real glass stays see-through
//                           in the mask exactly as it does in the image.
//
// -----------------------------------------------------------------------------
// THE ID CONVENTION
// -----------------------------------------------------------------------------
// A GLOBAL object index is a 1-based integer assigned by the capture worker in
// mesh-bundle STREAM order (deterministic — the server streams `sorted(*.glb)`),
// so index i names `manifest.objects[i - 1]`. Index 0 is reserved: it means
// background (sky / void / nothing rendered). The object's own string id — the
// GLB file stem — is the same identifier used as `userData.objectId` during
// capture, as the proxy node name (encodeGlb), and as the dollhouse node name
// (bake-vertex-color.mjs `labelObject`), so one namespace spans every
// representation of the scene.
//
// In the render pass the index is written as bytes: R = index & 255,
// G = index >> 8, B = 0, A = 255 — read back as a little-endian uint16.
//
// -----------------------------------------------------------------------------
// ANTIALIASING: WHY THERE IS A COVERAGE PLANE
// -----------------------------------------------------------------------------
// An id buffer cannot be antialiased. The rasterizer hands each pixel exactly one
// triangle and averaging two object indices is meaningless, so a mask boundary is
// hard-quantized to the raster grid — however fine that grid is. Magnify it, and a
// walkthrough magnifies heavily (a 4096-wide equirect spans 360°, so one texel is
// ~2 screen px at 75° FOV and several times that zoomed in), and the boundary reads
// as a staircase: locally smooth, globally stepped.
//
// Raising the equirect width does NOT fix that. It resamples the same hard face
// raster, and past `face_size * 4` it is pure upsampling — bigger stair treads, the
// same number of them, for more bytes. The subpixel information has to be MEASURED
// at raster time.
//
// So the ID faces are rendered SUPERSAMPLED and each output texel stores two
// things: the id that won the most samples, and the FRACTION of samples that winner
// took. That fraction is coverage — the same quantity MSAA resolves to — and it
// puts the boundary back at its true sub-texel position when a viewer reconstructs
// it. `supersample` 2 yields the five distinct edge levels of 4x MSAA; 3 and 4 give
// ten and seventeen.
//
// -----------------------------------------------------------------------------
// THE SID1 CONTAINER (one file per anchor, `{anchor}.sid`)
// -----------------------------------------------------------------------------
// Little-endian throughout (every platform we run on). Header is 24 bytes:
//
//   offset  type              field
//   0       char[4]           magic "SID1"
//   4       u16               width
//   6       u16               height
//   8       u8                planes            (1 = ids only, 2 = ids + coverage)
//   9       u8                id_filter         (0 = raw, 1 = sub-left)
//   10      u8                index_bytes       (1 or 2)
//   11      u8                coverage_filter   (0 = raw, 1 = sub-left)
//   12      u16               palette_count
//   14      u16               supersample       (samples per texel = this squared)
//   16      u32               id_bytes
//   20      u32               coverage_bytes    (0 when planes == 1)
//   24      u16[palette_count]  LOCAL index (1-based) -> GLOBAL object index
//   ...     u8[id_bytes]        deflate(filter(id plane))
//   ...     u8[coverage_bytes]  deflate(filter(coverage plane))
//
// Both planes are top-down and row-major. The id plane holds one LOCAL index per
// pixel, 0 = background; localizing keeps it 8-bit for essentially every anchor (a
// room sees far fewer than 255 of a scene's few hundred objects), and one that sees
// more flips index_bytes to 2. The coverage plane is one byte per pixel, 255 = the
// winner owns the texel outright — which is true of all but the boundary bands, so
// it costs very little once compressed. `filter` 1 is PNG's "Sub" predictor (per-row
// horizontal delta in the plane's own dtype, wrapping), the same trick
// splat/stage5.py's SZF frames use, which turns flat runs into zeros for deflate to
// crush. The palette rides INSIDE the container so an anchor costs exactly one fetch.
//
// RECONSTRUCTION. A texel's share of object A is `coverage` where the texel's id IS
// A, and `1 - coverage` where it is a texel A bleeds into — at a two-object edge the
// winner's leftover is A's. That second case only holds NEAR A, so a reader must
// gate it on A appearing somewhere in the sampled neighbourhood; ungated, an
// unrelated B/C edge across the room would read as 40% A. See tourview.js.
//
// -----------------------------------------------------------------------------
// THE EQUIRECT CONVENTION (must match the colour stitch, or masks slide off)
// -----------------------------------------------------------------------------
//   direction -> uv:  u = atan2(z, x) / 2pi + 0.5      (col = u * width)
//                     v = asin(y) / pi + 0.5           (row = (1 - v) * height)
// Row 0 is straight UP, matching tourcapture.js's `stitchPanoBlob` and
// prod_client's `makePanoMaterial`. NOTE the row flip: the plane is stored
// top-down while a GL texture's t = 0 is its first row in memory, so a SHADER
// sampling `maskTexture` (which is `flipY = false`) must use `t = 1 - v`.

import * as THREE from "three";
import { OIT_LAYER, OIT_OPAQUE } from "./oit.js";

export const ID_MASK_MAGIC = "SID1";
export const ID_MASK_HEADER_BYTES = 24;
export const ID_BACKGROUND = 0;
export const ID_MAX = 65535;
export const FILTER_RAW = 0;
export const FILTER_SUBLEFT = 1;
// Samples per output texel = this squared. 2 is the 4x-MSAA-equivalent default.
export const DEFAULT_SUPERSAMPLE = 2;

const MAGIC_U32 = 0x31444953; // "SID1" read as a little-endian uint32
const TAU = Math.PI * 2;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --- capture: per-object ID materials ----------------------------------------

const ID_VERTEX = /* glsl */ `
    #if defined(ID_MAP) || defined(ID_ALPHAMAP)
    varying vec2 vIdUv;
    #endif
    void main() {
        #if defined(ID_MAP) || defined(ID_ALPHAMAP)
        vIdUv = uv;
        #endif
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

// Writes the index bytes RAW: a plain ShaderMaterial gets none of three.js'
// tone-mapping / colour-space fragment chunks, and the target carries
// NoColorSpace, so the byte we ask for is the byte we read back.
const ID_FRAGMENT = /* glsl */ `
    uniform vec2 uId;
    #ifdef ID_CUTOUT
    uniform float uCutoff;
    uniform float uOpacity;
    #endif
    #ifdef ID_MAP
    uniform sampler2D uMap;
    #endif
    #ifdef ID_ALPHAMAP
    uniform sampler2D uAlphaMap;
    #endif
    #if defined(ID_MAP) || defined(ID_ALPHAMAP)
    varying vec2 vIdUv;
    #endif
    void main() {
        #ifdef ID_CUTOUT
        float a = uOpacity;
        #ifdef ID_MAP
        a *= texture2D(uMap, vIdUv).a;
        #endif
        #ifdef ID_ALPHAMAP
        a *= texture2D(uAlphaMap, vIdUv).g;
        #endif
        if (a < uCutoff) discard;
        #endif
        gl_FragColor = vec4(uId, 0.0, 1.0);
    }
`;

// One ID material per source material. `cutoff` mirrors whatever discards
// fragments in the LIT pass: OIT_OPAQUE for a mesh on the OIT layer (oit.js's
// depth pre-pass rule), else the material's own alphaTest. Alpha is composed the
// same way three.js composes it — opacity × map.a × alphaMap.g — so a mesh whose
// opacity alone is sub-cutoff discards everywhere and claims no pixels at all.
function makeIdMaterial(index, source, cutoff, hasUv) {
    const useMap = cutoff > 0 && hasUv && !!source.map;
    const useAlphaMap = cutoff > 0 && hasUv && !!source.alphaMap;
    const defines = {};
    if (cutoff > 0) defines.ID_CUTOUT = "";
    if (useMap) defines.ID_MAP = "";
    if (useAlphaMap) defines.ID_ALPHAMAP = "";
    return new THREE.ShaderMaterial({
        defines,
        uniforms: {
            uId: {
                value: new THREE.Vector2(index & 255, (index >> 8) & 255).divideScalar(255),
            },
            uMap: { value: useMap ? source.map : null },
            uAlphaMap: { value: useAlphaMap ? source.alphaMap : null },
            uCutoff: { value: cutoff },
            uOpacity: { value: source.opacity ?? 1 },
        },
        vertexShader: ID_VERTEX,
        fragmentShader: ID_FRAGMENT,
        side: source.side,
    });
}

// Give every mesh under `root` an ID material carrying this object's global
// index. MUST run after the OIT gate (prepareOITScene), since the cutoff depends
// on which layer a mesh landed on — capturecore.js's prepareCaptureObject
// enforces that order.
export function attachIdMaterials(root, index) {
    if (!Number.isInteger(index) || index < 1 || index > ID_MAX) {
        throw new Error(`idmask: object index ${index} outside 1..${ID_MAX}`);
    }
    root.traverse((mesh) => {
        if (!mesh.isMesh || !mesh.material) return;
        const cutoff = mesh.layers.isEnabled(OIT_LAYER) ? OIT_OPAQUE : 0;
        const hasUv = !!mesh.geometry?.getAttribute("uv");
        const build = (m) => makeIdMaterial(index, m, cutoff || m.alphaTest || 0, hasUv);
        mesh.userData.idMaterial = Array.isArray(mesh.material)
            ? mesh.material.map(build)
            : build(mesh.material);
    });
}

// Anything in the scene that ISN'T a registered object — a lighting rig's shadow
// catcher, a helper — draws through this instead: an unregistered mesh painting
// its lit colour into the plane would read back as a garbage object index, and a
// mask whose whole value is being exact can't have a "mostly" in it.
const NULL_ID_MATERIAL = new THREE.ShaderMaterial({
    vertexShader: "void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
    fragmentShader: "void main() { gl_FragColor = vec4(0.0); }",
    colorWrite: false,
    depthWrite: false,
    depthTest: false,
});

// Flip the whole scene between its lit materials and its ID materials, saving the
// lit one on first use so this owns the swap outright. Cheap enough to toggle per
// anchor (one traversal, a few thousand assignments).
export function setIdMode(root, on) {
    root.traverse((mesh) => {
        if (!mesh.isMesh || !mesh.material) return;
        if (on) {
            if (mesh.userData.shadeMaterial === undefined) {
                mesh.userData.shadeMaterial = mesh.material;
            }
            mesh.material = mesh.userData.idMaterial ?? NULL_ID_MATERIAL;
        } else if (mesh.userData.shadeMaterial !== undefined) {
            mesh.material = mesh.userData.shadeMaterial;
        }
    });
}

// --- capture: the ID render pass ---------------------------------------------

// A flat RGBA8 target plus the two-pass render described at the top. The caller
// owns the camera (capturecore aims the SAME one the colour pass uses, so the
// two rasterizations can't drift) and the size, which is deliberately NOT the
// colour face size: the ID faces are supersampled so the stitch can measure
// coverage, and their resolution is the real ceiling on mask sharpness.
export function createIdRenderer(renderer, size) {
    const target = new THREE.WebGLRenderTarget(size, size, {
        format: THREE.RGBAFormat,
        type: THREE.UnsignedByteType,
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        generateMipmaps: false,
        depthBuffer: true,
        stencilBuffer: false,
    });
    target.texture.colorSpace = THREE.NoColorSpace;

    function render(scene, camera) {
        const prevTarget = renderer.getRenderTarget();
        const prevAutoClear = renderer.autoClear;
        renderer.setRenderTarget(target);
        renderer.setClearColor(0x000000, 1);
        renderer.autoClear = true; // clears colour (= background id 0) and depth
        camera.layers.set(0);
        renderer.render(scene, camera);
        renderer.autoClear = false; // keep the opaque depth for the glass pass
        camera.layers.set(OIT_LAYER);
        renderer.render(scene, camera);
        camera.layers.set(0);
        renderer.autoClear = prevAutoClear;
        renderer.setRenderTarget(prevTarget);
        return target;
    }

    return { target, size, render, dispose: () => target.dispose() };
}

// Read one rendered face back as top-down global indices (GL rows arrive
// bottom-up), decoding R + G<<8 in the same sweep.
export function readIdFace(renderer, target, size) {
    const raw = new Uint8Array(size * size * 4);
    renderer.readRenderTargetPixels(target, 0, 0, size, size, raw);
    const out = new Uint16Array(size * size);
    for (let row = 0; row < size; row++) {
        let src = (size - 1 - row) * size * 4;
        let dst = row * size;
        for (let x = 0; x < size; x++, src += 4, dst++) {
            out[dst] = raw[src] | (raw[src + 1] << 8);
        }
    }
    return out;
}

// --- capture: cube faces -> equirect ID plane --------------------------------

// The sub-texel sample positions one output texel spans, in face-texel units. An
// output texel covers `supersample` face texels per axis, so its footprint is
// [-SS/2, +SS/2] and the strata sit on the SS face texels inside it.
//
// Exported because the COLOUR stitch has to average exactly the samples this
// votes on. The pano and the mask share a pose and a projection, but that alone
// doesn't make their edges land in the same place — an edge is where the RASTER
// says it is, so the two only agree if they reduce the same face texels over the
// same footprint. This is that footprint.
export function subTexelOffsets(supersample) {
    const SS = Math.max(1, supersample | 0);
    const offset = new Float64Array(SS);
    for (let k = 0; k < SS; k++) offset[k] = k + 0.5 - SS * 0.5;
    return offset;
}

// Stitch six SUPERSAMPLED ID faces into one equirect plane of GLOBAL indices plus
// a matching coverage plane. `faceBasis` is the caller's cube-face table
// (forward/up/right per face) — the SAME one the colour stitch uses, so the two
// images register pixel for pixel — and `supersample` is how many face texels span
// one output texel per axis, which the caller guarantees by rendering the faces at
// `width / 4 * supersample`.
//
// Per output texel we take the supersample² face texels its footprint covers,
// give it the id holding the most of them, and record that id's share as coverage.
// Ties go to the texel nearest the exact direction, then to a foreground id over
// background — which is what keeps a chair leg thinner than a texel alive rather
// than leaving it to a coin flip.
//
// The direction → face-texel map costs two trig calls per output texel, and at
// 4096×2048 that alone is 17M calls; azimuth depends only on the column, so it is
// hoisted into a table and the inner loop is pure arithmetic.
//
// The sample box is taken inside ONE face (the one the texel's centre direction
// lands on), so along the twelve cube seams it clamps rather than reaching into the
// neighbouring face, and coverage there degrades toward binary. Ids stay correct —
// only the antialiasing of a boundary lying within a texel of a seam is affected,
// which is a handful of texels per anchor.
export async function stitchIdPlane(
    faces,
    faceBasis,
    faceSize,
    width,
    height,
    supersample = DEFAULT_SUPERSAMPLE,
    onProgress,
) {
    const ids = new Uint16Array(width * height);
    const coverage = new Uint8Array(width * height);
    const S = faceSize;
    const maxIdx = S - 1;
    const SS = Math.max(1, supersample | 0);
    const samples = SS * SS;

    const cosAz = new Float64Array(width);
    const sinAz = new Float64Array(width);
    for (let col = 0; col < width; col++) {
        const az = ((col + 0.5) / width - 0.5) * TAU;
        cosAz[col] = Math.cos(az);
        sinAz[col] = Math.sin(az);
    }
    const offset = subTexelOffsets(SS);
    const tallyId = new Uint16Array(samples);
    const tallyN = new Uint8Array(samples);

    for (let row = 0; row < height; row++) {
        const v = 1 - (row + 0.5) / height;
        const phi = (v - 0.5) * Math.PI;
        const dy = Math.sin(phi);
        const cosPhi = Math.cos(phi);
        const ay = Math.abs(dy);
        let oi = row * width;
        for (let col = 0; col < width; col++, oi++) {
            const dx = cosPhi * cosAz[col];
            const dz = cosPhi * sinAz[col];

            const ax = Math.abs(dx);
            const az2 = Math.abs(dz);
            let faceIdx;
            if (ax >= ay && ax >= az2) faceIdx = dx > 0 ? 0 : 1;
            else if (ay >= az2) faceIdx = dy > 0 ? 2 : 3;
            else faceIdx = dz > 0 ? 4 : 5;

            const { f, up, right } = faceBasis[faceIdx];
            const t = dx * f[0] + dy * f[1] + dz * f[2];
            const u2 = (dx * right[0] + dy * right[1] + dz * right[2]) / t;
            const v2 = (dx * up[0] + dy * up[1] + dz * up[2]) / t;
            const px = (u2 * 0.5 + 0.5) * S - 0.5;
            const py = (0.5 - v2 * 0.5) * S - 0.5;

            const d = faces[faceIdx];
            const cx = px < 0 ? 0 : px > maxIdx ? maxIdx : Math.round(px);
            const cy = py < 0 ? 0 : py > maxIdx ? maxIdx : Math.round(py);
            const nearest = d[cy * S + cx];

            let n = 0;
            for (let sy = 0; sy < SS; sy++) {
                const fy = py + offset[sy];
                const y = fy < 0 ? 0 : fy > maxIdx ? maxIdx : Math.round(fy);
                const rowBase = y * S;
                for (let sx = 0; sx < SS; sx++) {
                    const fx = px + offset[sx];
                    const x = fx < 0 ? 0 : fx > maxIdx ? maxIdx : Math.round(fx);
                    const id = d[rowBase + x];
                    let k = 0;
                    while (k < n && tallyId[k] !== id) k++;
                    if (k === n) {
                        tallyId[n] = id;
                        tallyN[n] = 1;
                        n++;
                    } else {
                        tallyN[k]++;
                    }
                }
            }

            let win = 0;
            let bestScore = -1;
            for (let k = 0; k < n; k++) {
                const score =
                    tallyN[k] * 4 +
                    (tallyId[k] === nearest ? 2 : 0) +
                    (tallyId[k] !== ID_BACKGROUND ? 1 : 0);
                if (score > bestScore) {
                    bestScore = score;
                    win = k;
                }
            }
            ids[oi] = tallyId[win];
            coverage[oi] = Math.round((tallyN[win] / samples) * 255);
        }
        if ((row & 127) === 127) {
            onProgress?.(row / height);
            await sleep(0);
        }
    }
    return { ids, coverage };
}

// --- the SID1 codec ----------------------------------------------------------

async function deflate(bytes) {
    const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("deflate"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function inflate(bytes) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
}

// Global indices -> a compact local alphabet, so the plane stays 8-bit unless
// this anchor genuinely sees more than 255 objects.
function paletteize(plane) {
    const seen = new Uint8Array(ID_MAX + 1);
    const ids = [];
    for (let i = 0; i < plane.length; i++) {
        const g = plane[i];
        if (g !== ID_BACKGROUND && seen[g] === 0) {
            seen[g] = 1;
            ids.push(g);
        }
    }
    ids.sort((a, b) => a - b);
    const remap = new Uint16Array(ID_MAX + 1);
    for (let i = 0; i < ids.length; i++) remap[ids[i]] = i + 1;
    const indexBytes = ids.length <= 255 ? 1 : 2;
    const local = indexBytes === 1
        ? new Uint8Array(plane.length)
        : new Uint16Array(plane.length);
    for (let i = 0; i < plane.length; i++) local[i] = remap[plane[i]];
    return { palette: Uint16Array.from(ids), local, indexBytes };
}

function planeBytes(local) {
    return new Uint8Array(local.buffer, local.byteOffset, local.byteLength);
}

function subLeft(local, width, height) {
    const out = local.constructor === Uint8Array
        ? new Uint8Array(local.length)
        : new Uint16Array(local.length);
    const wrap = local.constructor === Uint8Array ? 0xff : 0xffff;
    for (let row = 0; row < height; row++) {
        const o = row * width;
        out[o] = local[o];
        for (let x = 1; x < width; x++) out[o + x] = (local[o + x] - local[o + x - 1]) & wrap;
    }
    return planeBytes(out);
}

function unSubLeft(local, width, height) {
    for (let row = 0; row < height; row++) {
        const o = row * width;
        for (let x = 1; x < width; x++) local[o + x] += local[o + x - 1];
    }
}

// Compress one plane, trying both predictors when `filter` is "auto" and keeping
// the smaller — the header records which won, so nothing downstream has to guess.
async function packPlane(plane, width, height, filter) {
    const candidates = filter === "auto" ? [FILTER_SUBLEFT, FILTER_RAW] : [filter];
    let best = null;
    for (const f of candidates) {
        const payload = await deflate(
            f === FILTER_SUBLEFT ? subLeft(plane, width, height) : planeBytes(plane),
        );
        if (!best || payload.length < best.payload.length) best = { filter: f, payload };
    }
    return best;
}

// Encode an equirect id plane (GLOBAL indices) and its coverage plane into one
// SID1 buffer. `coverage` may be omitted, which writes a single-plane container
// that reads back as fully-covered — sharp-edged, but valid.
export async function encodeIdMask(
    ids,
    coverage,
    width,
    height,
    { filter = "auto", supersample = DEFAULT_SUPERSAMPLE } = {},
) {
    if (ids.length !== width * height) {
        throw new Error(`idmask: id plane is ${ids.length}, expected ${width * height}`);
    }
    if (coverage && coverage.length !== ids.length) {
        throw new Error("idmask: coverage plane size does not match the id plane");
    }
    const { palette, local, indexBytes } = paletteize(ids);
    const idPack = await packPlane(local, width, height, filter);
    const covPack = coverage ? await packPlane(coverage, width, height, filter) : null;

    const bytes = new Uint8Array(
        ID_MASK_HEADER_BYTES +
            palette.length * 2 +
            idPack.payload.length +
            (covPack?.payload.length ?? 0),
    );
    const view = new DataView(bytes.buffer);
    view.setUint32(0, MAGIC_U32, true);
    view.setUint16(4, width, true);
    view.setUint16(6, height, true);
    bytes[8] = covPack ? 2 : 1;
    bytes[9] = idPack.filter;
    bytes[10] = indexBytes;
    bytes[11] = covPack ? covPack.filter : 0;
    view.setUint16(12, palette.length, true);
    view.setUint16(14, supersample, true);
    view.setUint32(16, idPack.payload.length, true);
    view.setUint32(20, covPack?.payload.length ?? 0, true);
    let at = ID_MASK_HEADER_BYTES;
    bytes.set(new Uint8Array(palette.buffer, palette.byteOffset, palette.byteLength), at);
    at += palette.length * 2;
    bytes.set(idPack.payload, at);
    at += idPack.payload.length;
    if (covPack) bytes.set(covPack.payload, at);
    return {
        bytes,
        palette,
        indexBytes,
        filter: idPack.filter,
        idBytes: idPack.payload.length,
        coverageBytes: covPack?.payload.length ?? 0,
    };
}

// Decode a SID1 buffer. `plane` holds LOCAL indices (what a shader compares
// against); `palette` maps those to global object indices, and `localOf` is the
// inverse a viewer needs to light up a known object.
export async function decodeIdMask(buffer) {
    if (buffer.byteLength < ID_MASK_HEADER_BYTES) throw new Error("idmask: truncated header");
    const view = new DataView(buffer);
    if (view.getUint32(0, true) !== MAGIC_U32) throw new Error("idmask: not a SID1 container");
    const width = view.getUint16(4, true);
    const height = view.getUint16(6, true);
    const planes = view.getUint8(8);
    const filter = view.getUint8(9);
    const indexBytes = view.getUint8(10);
    const coverageFilter = view.getUint8(11);
    const paletteCount = view.getUint16(12, true);
    const supersample = view.getUint16(14, true);
    const idBytes = view.getUint32(16, true);
    const coverageBytes = view.getUint32(20, true);
    const paletteEnd = ID_MASK_HEADER_BYTES + paletteCount * 2;
    if (buffer.byteLength < paletteEnd + idBytes + coverageBytes) {
        throw new Error("idmask: truncated payload");
    }

    const count = width * height;
    const palette = new Uint16Array(buffer.slice(ID_MASK_HEADER_BYTES, paletteEnd));
    const rawIds = await inflate(new Uint8Array(buffer, paletteEnd, idBytes));
    const plane = indexBytes === 1
        ? new Uint8Array(rawIds.buffer, rawIds.byteOffset, count)
        : new Uint16Array(rawIds.buffer, rawIds.byteOffset, count);
    if (filter === FILTER_SUBLEFT) unSubLeft(plane, width, height);

    let coverage;
    if (planes >= 2 && coverageBytes > 0) {
        coverage = await inflate(new Uint8Array(buffer, paletteEnd + idBytes, coverageBytes));
        coverage = new Uint8Array(coverage.buffer, coverage.byteOffset, count);
        if (coverageFilter === FILTER_SUBLEFT) unSubLeft(coverage, width, height);
    } else {
        coverage = new Uint8Array(count).fill(255); // ids-only container: hard edges
    }

    const localOf = new Map();
    for (let i = 0; i < palette.length; i++) localOf.set(palette[i], i + 1);
    return {
        width,
        height,
        indexBytes,
        filter,
        supersample,
        palette,
        plane,
        coverage,
        localOf,
        bytes: buffer.byteLength,
    };
}

// --- viewing -----------------------------------------------------------------

// The GLOBAL object index a direction lands on (0 = background). `radius` > 0
// takes a majority over a (2r+1)² block, which stops a one-pixel sliver under the
// cursor flickering the hover between two objects.
export function sampleIdMask(mask, dx, dy, dz, radius = 0) {
    const len = Math.hypot(dx, dy, dz) || 1;
    const u = Math.atan2(dz / len, dx / len) / TAU + 0.5;
    const v = Math.asin(Math.min(1, Math.max(-1, dy / len))) / Math.PI + 0.5;
    const col = Math.floor(u * mask.width);
    const row = Math.floor((1 - v) * mask.height);
    if (radius <= 0) return globalAt(mask, col, row);

    let win = globalAt(mask, col, row);
    let best = 0;
    const tally = new Map();
    for (let y = row - radius; y <= row + radius; y++) {
        for (let x = col - radius; x <= col + radius; x++) {
            const g = globalAt(mask, x, y);
            const n = (tally.get(g) ?? 0) + 1;
            tally.set(g, n);
            if (n > best) {
                best = n;
                win = g;
            }
        }
    }
    return win;
}

function globalAt(mask, col, row) {
    const x = ((col % mask.width) + mask.width) % mask.width; // the equirect wraps in u
    const y = row < 0 ? 0 : row >= mask.height ? mask.height - 1 : row;
    const local = mask.plane[y * mask.width + x];
    return local === 0 ? ID_BACKGROUND : mask.palette[local - 1];
}

// The decoded mask as one GPU texture, id and coverage INTERLEAVED so a shader
// gets both from a single tap (it needs four taps per pixel already, and doubling
// that to read a second texture is the wrong place to spend). An 8-bit mask goes
// to RG8 as (local id, coverage); a 16-bit one to RGBA8 as (id lo, id hi,
// coverage, 255). Nearest + no mips because ids don't interpolate, repeat in u for
// the seam, and flipY OFF — which is why a shader samples at t = 1 - v (see the
// convention note at the top).
export function createMaskTexture(mask) {
    const n = mask.width * mask.height;
    const wide = mask.indexBytes === 2;
    const data = new Uint8Array(n * (wide ? 4 : 2));
    if (wide) {
        for (let i = 0, o = 0; i < n; i++, o += 4) {
            const local = mask.plane[i];
            data[o] = local & 255;
            data[o + 1] = local >> 8;
            data[o + 2] = mask.coverage[i];
            data[o + 3] = 255;
        }
    } else {
        for (let i = 0, o = 0; i < n; i++, o += 2) {
            data[o] = mask.plane[i];
            data[o + 1] = mask.coverage[i];
        }
    }
    const tex = new THREE.DataTexture(
        data,
        mask.width,
        mask.height,
        wide ? THREE.RGBAFormat : THREE.RGFormat,
        THREE.UnsignedByteType,
    );
    tex.magFilter = THREE.NearestFilter;
    tex.minFilter = THREE.NearestFilter;
    tex.generateMipmaps = false;
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.unpackAlignment = 1;
    tex.colorSpace = THREE.NoColorSpace;
    tex.needsUpdate = true;
    return tex;
}
