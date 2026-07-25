// =============================================================================
// capturecore.js — the ONE headless-capture render pipeline.
// =============================================================================
//
// Both headless captures render through this module, so their output is the same
// pixels. The ONLY thing that differs between them is the camera pattern:
//
//   • Stage 5 (splatcapture.js) renders the planned directed views (one shot per
//     camera in the coverage plan) and reads back colour + packed depth.
//   • The matterport tour (tourcapture.js) renders SIX 90° cube faces per anchor
//     and stitches them into an equirectangular panorama.
//
// Everything else is shared and must stay shared: the renderer configuration
// (linear HDR, no global tone mapping, frozen shadow map), the per-object
// material prep (reflective.js's matte discriminator + oit.js's transparent-mesh
// patch), the light / shadow / reflection bakes, and the weighted-blended OIT +
// ACES-filmic present. Previously the tour capture used the debug viewer's
// single-pass lit rig instead, which silently dropped real glass (its depth-proxy
// alpha-test cuts at 0.8, while glass bakes to ~0.065 alpha), dropped the scene
// reflection probes, and left every Trellis surface at metalness=1 — so panos
// didn't match the references. This module exists so that can't drift again.

import * as THREE from "three";
import { createLightRig } from "./splatlight.js";
import { bakeReflectionProbes } from "./reflections.js";
import { applyEmissiveLighting } from "./emissive.js";
import { matteNonReflective } from "./reflective.js";
import {
    OIT_LAYER,
    TONEMAP_GLSL,
    setOITBlend,
    prepareOITScene,
    collectOITMaterials,
    decorateOITLights,
    bakeShadowMap,
} from "./oit.js";

export const DEPTH_CODE_MAX = 65535; // matches splat/stage5.py _DEPTH_CODE_MAX

// Per-page OIT pass selector (0 accumulate · 1 revealage · 2 depth pre-pass · 3
// opaque passthrough). The transparent materials patched by prepareOITScene, the
// reflection bake, and the render passes must all read the SAME object, so it
// lives here as the module's singleton rather than being threaded through.
export const oitPass = { value: 0 };

// The capture renderer: the scene is shaded in LINEAR light into a half-float
// target and the present pass owns the whole display transform (ACES + sRGB), so
// global tone mapping stays OFF and the encode is deterministic across three.js
// versions. Shadow maps render ONCE (static scene + sun) — autoUpdate off, with a
// single bake via oit.js bakeShadowMap.
export function createCaptureRenderer({ onContextLost, previewSize = 256 } = {}) {
    const renderer = new THREE.WebGLRenderer({
        antialias: false,
        alpha: false,
        preserveDrawingBuffer: false,
        powerPreference: "high-performance",
    });
    if (!renderer.capabilities.isWebGL2) throw new Error("WebGL2 unavailable");
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.shadowMap.autoUpdate = false;
    renderer.setSize(previewSize, previewSize); // preview canvas only; targets carry the real res
    document.body.appendChild(renderer.domElement);
    if (onContextLost) {
        renderer.domElement.addEventListener("webglcontextlost", (e) => {
            e.preventDefault();
            onContextLost(new Error("WebGL context lost"));
        });
    }
    return renderer;
}

// Per-loaded-object prep, in the order the pipeline requires: stamp the object's
// id (emissive.js / reflective.js match on it), decide reflectivity BEFORE the OIT
// patch (matteNonReflective keeps the curated reflective names and mattes the
// rest), then route transparent meshes onto the OIT layer.
export function prepareCaptureObject(root, id) {
    root.userData.objectId = id;
    matteNonReflective(root, id);
    prepareOITScene(root, oitPass);
}

// The light / shadow / reflection bakes, in the order they depend on each other:
// emissive fixtures first (they add shadow-casting point lights), then the sun +
// hemi fill + IBL rig fitted to the scene, then ONE shadow bake seeing every
// caster on both layers, then the per-object scene reflection probes.
export function setupCaptureLighting(renderer, scene, { lighting, background }) {
    const sceneBox = new THREE.Box3().setFromObject(scene);
    const emissive = applyEmissiveLighting(scene, scene);
    // decorateOITLights puts the sun + hemi on all layers so the glass moved onto
    // OIT_LAYER by prepareOITScene is still lit (not just IBL-lit).
    const rig = createLightRig(renderer, scene, {
        defaults: lighting,
        decorateLights: decorateOITLights,
    });
    rig.refit(sceneBox);
    bakeShadowMap(renderer, scene);
    const refl = bakeReflectionProbes(renderer, scene, { background, oitPass });
    return { emissive, rig, refl, sceneBox };
}

// Capture pipeline, per view: shade the lit scene in LINEAR HDR, then tone-map +
// sRGB-encode in our OWN ShaderMaterial (deterministic across three.js versions)
// → rtColor (RGBA8), and pack window-space depth → rtDepth (RG8 log-uint16 codes).
// Opaque scenes take one scene pass + present; scenes with transparent meshes run
// weighted-blended OIT (renderOIT) so glass composites order-independently, and the
// composite emits the same ACES+sRGB colour plus the accumulated coverage in alpha.
// `exposure` and `uLinear` (curve-off, for the self-test) parameterize the present.
//
// `width`/`height` size the targets: Stage 5 passes its square reference
// resolution, the tour passes its square cube-face size, and the tour's overhead
// minimap slices pass their own rectangle.
export function createCapture(
    renderer,
    width,
    height,
    near,
    far,
    fovDeg,
    background,
    exposure = 1.0,
) {
    const depthTexture = new THREE.DepthTexture(width, height);
    depthTexture.type = THREE.FloatType;
    const targetOpts = {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        generateMipmaps: false,
        stencilBuffer: false,
    };
    // Lit scene → linear HDR + depth. Half-float holds values > 1 so the tone map
    // (present pass) has real highlight detail to compress.
    const rtScene = new THREE.WebGLRenderTarget(width, height, {
        ...targetOpts,
        depthBuffer: true,
        depthTexture,
        type: THREE.HalfFloatType,
    });
    // Final 8-bit readback target for colour: the present pass writes already-
    // encoded sRGB bytes, so nothing else touches them.
    const rtColor = new THREE.WebGLRenderTarget(width, height, {
        ...targetOpts,
        depthBuffer: false,
    });
    // Depth pack target is RG8 (two bytes/pixel): the pack shader writes the
    // log-u16 code's low byte to R and high byte to G, so the readback bytes ARE
    // the little-endian uint16 code — no CPU swizzle, and half the readback.
    const rtDepth = new THREE.WebGLRenderTarget(width, height, {
        ...targetOpts,
        depthBuffer: false,
        format: THREE.RGFormat,
        type: THREE.UnsignedByteType,
    });

    const fullscreenVS = /* glsl */ `
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = vec4(position.xy, 0.0, 1.0);
        }
    `;

    // Present: linear HDR → ACES-filmic (three.js' exact fit, so the splat looks
    // like the mesh viewer) → sRGB, 8-bit. uLinear = true bypasses the curve
    // (clamp only) for the self-test, whose checks are written against raw values.
    const presentMaterial = new THREE.ShaderMaterial({
        uniforms: {
            tScene: { value: rtScene.texture },
            uExposure: { value: exposure },
            uLinear: { value: false },
        },
        vertexShader: fullscreenVS,
        fragmentShader: /* glsl */ `
            precision highp float;
            uniform sampler2D tScene;
            uniform float uExposure;
            uniform bool uLinear;
            varying vec2 vUv;
            ${TONEMAP_GLSL}
            void main() {
                vec4 s = texture2D(tScene, vUv);
                if (uLinear) {
                    gl_FragColor = vec4(clamp(s.rgb, 0.0, 1.0), s.a);
                    return;
                }
                vec3 c = linearToSrgb(acesFilmic(s.rgb * uExposure));
                gl_FragColor = vec4(c, s.a);
            }
        `,
        depthTest: false,
        depthWrite: false,
    });

    const packMaterial = new THREE.ShaderMaterial({
        uniforms: {
            tDepth: { value: depthTexture },
            uNear: { value: near },
            uFar: { value: far },
        },
        vertexShader: fullscreenVS,
        fragmentShader: /* glsl */ `
            precision highp float;
            uniform sampler2D tDepth;
            uniform float uNear;
            uniform float uFar;
            varying vec2 vUv;
            void main() {
                float d = texture2D(tDepth, vUv).r;             // window-space [0,1]
                if (d >= 1.0) {                                  // nothing wrote depth
                    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);     // code 0 = background
                    return;
                }
                float zndc = d * 2.0 - 1.0;
                float z = 2.0 * uNear * uFar / (uFar + uNear - zndc * (uFar - uNear));
                // splat/stage5.py encode_depth_u16: log-spaced code 1..65535.
                float t = clamp(log(z / uNear) / log(uFar / uNear), 0.0, 1.0);
                float code = floor(t * ${(DEPTH_CODE_MAX - 1).toFixed(1)} + 0.5) + 1.0;
                float hi = floor(code / 256.0);
                float lo = code - hi * 256.0;
                // LOW byte -> R, HIGH byte -> G: on an RG8 target the readback bytes
                // are [lo, hi] = the code as a little-endian uint16, so no swizzle.
                gl_FragColor = vec4(lo / 255.0, hi / 255.0, 0.0, 1.0);
            }
        `,
        depthTest: false,
        depthWrite: false,
    });

    // A fullscreen triangle (no index, no camera transform — clip-space verts)
    // per pass, both driven by one throwaway ortho-less camera.
    const fsGeometries = [];
    function fullscreenScene(material) {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute(
            "position",
            new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3),
        );
        geo.setAttribute(
            "uv",
            new THREE.BufferAttribute(new Float32Array([0, 0, 2, 0, 0, 2]), 2),
        );
        fsGeometries.push(geo);
        const scene = new THREE.Scene();
        const tri = new THREE.Mesh(geo, material);
        tri.frustumCulled = false;
        scene.add(tri);
        return scene;
    }
    // OIT composite (glass scenes): resolve accumulate/revealage over the opaque
    // image in LINEAR, then ACES + sRGB exactly like the present pass. Alpha
    // carries the accumulated coverage 1 − Π(1−αᵢ).
    const compositeMaterial = new THREE.ShaderMaterial({
        uniforms: {
            uOpaque: { value: rtScene.texture },
            uAccum: { value: null },
            uReveal: { value: null },
            uExposure: { value: exposure },
        },
        vertexShader: fullscreenVS,
        fragmentShader: /* glsl */ `
            precision highp float;
            uniform sampler2D uOpaque;
            uniform sampler2D uAccum;
            uniform sampler2D uReveal;
            uniform float uExposure;
            varying vec2 vUv;
            ${TONEMAP_GLSL}
            void main() {
                vec4 accum = texture2D(uAccum, vUv);
                float reveal = texture2D(uReveal, vUv).r;
                vec4 op = texture2D(uOpaque, vUv);
                vec3 oit = accum.rgb / max(accum.a, 1e-5);
                vec3 lin = mix(oit, op.rgb, reveal);
                float coverage = 1.0 - reveal * (1.0 - op.a);
                gl_FragColor = vec4(linearToSrgb(acesFilmic(lin * uExposure)), coverage);
            }
        `,
        depthTest: false,
        depthWrite: false,
    });

    const presentScene = fullscreenScene(presentMaterial);
    const packScene = fullscreenScene(packMaterial);
    const compositeScene = fullscreenScene(compositeMaterial);
    const fsCamera = new THREE.Camera();

    const camera = new THREE.PerspectiveCamera(fovDeg, width / height, near, far);
    const bg = new THREE.Color(background[0], background[1], background[2]);

    // Accumulate + revealage targets, created on first glass use so opaque-only
    // scenes never allocate them. Both SHARE rtScene's depth texture, so glass is
    // depth-tested against the opaque surfaces and never occludes the room behind.
    let accumTarget = null;
    let revealTarget = null;
    function ensureOITTargets() {
        if (accumTarget) return;
        const opts = { ...targetOpts, type: THREE.HalfFloatType, depthBuffer: true, depthTexture };
        accumTarget = new THREE.WebGLRenderTarget(width, height, opts);
        revealTarget = new THREE.WebGLRenderTarget(width, height, opts);
        compositeMaterial.uniforms.uAccum.value = accumTarget.texture;
        compositeMaterial.uniforms.uReveal.value = revealTarget.texture;
    }

    // The scene's transparent materials, gathered once (the scene is static per
    // capture). Empty → opaque scene → the single-pass fast path.
    let oitMats = null;

    // Weighted-blended OIT for a glass scene, mirroring scene3d.js renderFrame:
    // opaque (layer 0) → rtScene, then three glass-only sub-passes (layer 1) — depth
    // pre-pass, accumulate, revealage — composited to rtColor. The shadow map stays
    // static (renderer.shadowMap.autoUpdate is off), so it is built once.
    function renderOIT(scene, cam) {
        ensureOITTargets();
        // opaque → rtScene: linear colour + the depth glass tests against.
        cam.layers.set(0);
        renderer.autoClear = true;
        renderer.setClearColor(bg, 0);
        renderer.setRenderTarget(rtScene);
        renderer.render(scene, cam);
        // glass sub-passes (layer 1); keep the opaque colour/depth (no auto-clear).
        renderer.autoClear = false;
        cam.layers.set(OIT_LAYER);
        oitPass.value = 2; // depth pre-pass: α≥OIT_OPAQUE writes depth, no colour
        for (const m of oitMats) {
            m.depthWrite = true;
            m.colorWrite = false;
        }
        renderer.setRenderTarget(accumTarget);
        renderer.render(scene, cam);
        oitPass.value = 0; // accumulate (additive), depth-tested, no depth write
        for (const m of oitMats) {
            m.depthWrite = false;
            m.colorWrite = true;
            setOITBlend(m, true);
        }
        renderer.setRenderTarget(accumTarget);
        renderer.setClearColor(0x000000, 0);
        renderer.clear(true, false, false);
        renderer.render(scene, cam);
        oitPass.value = 1; // revealage: dst *= (1 − α)
        for (const m of oitMats) setOITBlend(m, false);
        renderer.setRenderTarget(revealTarget);
        renderer.setClearColor(0xffffff, 1);
        renderer.clear(true, false, false);
        renderer.render(scene, cam);
        // composite over opaque → rtColor (ACES + sRGB + coverage alpha).
        cam.layers.set(0);
        renderer.autoClear = true;
        renderer.setRenderTarget(rtColor);
        renderer.render(compositeScene, fsCamera);
    }

    // Render `scene` through ANY camera into rtColor (+ rtDepth). This is the one
    // render path: the perspective view helper below and the tour's overhead
    // orthographic slices both come through here, so nothing can diverge.
    function renderCamera(scene, cam) {
        if (oitMats === null) oitMats = collectOITMaterials(scene);
        if (oitMats.length === 0) {
            // Opaque scene: one lit pass → present. Unchanged fast path.
            renderer.setClearColor(bg, 0); // alpha 0 = empty coverage
            renderer.setRenderTarget(rtScene);
            renderer.render(scene, cam); // lit, linear HDR + depth
            renderer.setRenderTarget(rtColor);
            renderer.render(presentScene, fsCamera); // ACES + sRGB → 8-bit
        } else {
            renderOIT(scene, cam);
        }
        renderer.setRenderTarget(rtDepth);
        renderer.render(packScene, fsCamera); // depth → log-u16 codes
    }

    const lookTarget = new THREE.Vector3();
    function renderView(scene, pos, face) {
        camera.position.set(pos[0], pos[1], pos[2]);
        camera.up.set(face.up[0], face.up[1], face.up[2]);
        lookTarget
            .set(face.forward[0], face.forward[1], face.forward[2])
            .add(camera.position);
        camera.lookAt(lookTarget);
        renderCamera(scene, camera);
    }

    function setLinear(on) {
        presentMaterial.uniforms.uLinear.value = !!on;
    }

    // For short-lived instances (the tour's per-level minimap slices). Stage 5
    // holds one capture for the whole run and never calls this.
    function dispose() {
        for (const t of [rtScene, rtColor, rtDepth, accumTarget, revealTarget]) t?.dispose();
        depthTexture.dispose();
        for (const m of [presentMaterial, packMaterial, compositeMaterial]) m.dispose();
        for (const g of fsGeometries) g.dispose();
    }

    return { rtColor, rtDepth, renderView, renderCamera, camera, setLinear, dispose };
}

// Read an 8-bit RGBA target back TOP-DOWN (GL rows arrive bottom-up). `opaque`
// forces alpha to 255: rtColor's alpha carries OIT *coverage*, which is 0 over
// background — fine for the Stage-5 mask, but a JPEG/PNG needs the pixels solid.
export function readTargetTopDown(renderer, target, w, h, opaque = false) {
    const raw = new Uint8Array(w * h * 4);
    renderer.readRenderTargetPixels(target, 0, 0, w, h, raw);
    const data = new Uint8ClampedArray(w * h * 4);
    const rowBytes = w * 4;
    for (let row = 0; row < h; row++) {
        const src = (h - 1 - row) * rowBytes;
        data.set(raw.subarray(src, src + rowBytes), row * rowBytes);
    }
    if (opaque) {
        for (let i = 3; i < data.length; i += 4) data[i] = 255;
    }
    return { data, width: w, height: h };
}
