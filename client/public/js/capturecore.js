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
// The tour takes one further pass per face — the flat per-pixel object-ID mask
// that drives its hover + highlight (idmask.js). It goes through this module's
// camera for exactly the reason everything else is shared: a mask that rasterized
// differently from its pano would sit off the image it labels.
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
// The per-pixel object-ID pass (the tour's hover/highlight masks). It rides on
// THIS module's camera so the mask rasterizes exactly like the colour image.
import { attachIdMaterials, createIdRenderer } from "./idmask.js";

export const DEPTH_CODE_MAX = 65535; // matches splat/stage5.py _DEPTH_CODE_MAX

// Per-page OIT pass selector (0 accumulate · 1 revealage · 2 depth pre-pass · 3
// opaque passthrough). The transparent materials patched by prepareOITScene, the
// reflection bake, and the render passes must all read the SAME object, so it
// lives here as the module's singleton rather than being threaded through.
export const oitPass = { value: 0 };

// PER-PASS cost, which is the only way to tell a scene that is expensive to
// RASTERIZE from one that is merely expensive to WALK. Each `renderer.render()`
// traverses the whole scene graph to build its render list, so a pass that draws ten
// objects out of eight hundred still pays for all eight hundred — and that cost is
// invisible in a single wall-clock number for the frame.
//
//   cpu    ms on the JS thread: traversal + material setup + draw submission.
//   gpu    ms the device actually spent (SAMPLED — see below).
//   calls  draw calls issued.
//
// High cpu + high calls + low gpu = submission-bound, and the fix is fewer, bigger
// draws (merge/instance) or fewer passes. High gpu = genuinely fill-bound, and the
// fix is fewer pixels or cheaper shading. Those two want opposite changes, which is
// why guessing between them is expensive.
//
// GPU timing is SAMPLED one view in `gpuEvery`, because TIME_ELAPSED_EXT permits
// only one query in flight at a time: timing every pass of every view would
// serialize the very passes it is measuring. `gpuStatus` says whether the extension
// was there at all — Chromium needs --enable-webgl-draft-extensions for it.
export const renderProfile = {
    enabled: false,
    gpuEvery: 64,
    cpu: Object.create(null),
    gpu: Object.create(null),
    calls: Object.create(null),
    views: 0,
    gpuViews: 0,
    gpuStatus: "off",
};

// Wraps each pass with CPU/draw-call accounting and, on sampled views, a GPU timer
// query. A no-op unless `renderProfile.enabled`, so the tour capture and the debug
// viewers pay nothing.
function makePassProfiler(renderer) {
    const gl = renderer.getContext();
    let ext = null;
    let probed = false;
    const free = [];
    const live = []; // { pass, query } — results arrive some frames later
    let open = null;
    let sampling = false;
    let t0 = 0;

    function probe() {
        probed = true;
        // WebGL1 contexts expose a different, incompatible extension; we are always
        // WebGL2 here, so only the _webgl2 spelling is worth asking for.
        ext = gl.getExtension("EXT_disjoint_timer_query_webgl2");
        renderProfile.gpuStatus = ext ? "timer-query" : "unavailable";
    }

    // Reap whatever finished. GPU_DISJOINT means the device rescheduled mid-measure
    // and every timing in flight is meaningless, so they all go on the floor.
    function drain() {
        if (!ext) return;
        if (gl.getParameter(ext.GPU_DISJOINT_EXT)) {
            for (const e of live) free.push(e.query);
            live.length = 0;
            return;
        }
        for (let i = live.length - 1; i >= 0; i--) {
            const e = live[i];
            if (!gl.getQueryParameter(e.query, gl.QUERY_RESULT_AVAILABLE)) continue;
            const ns = gl.getQueryParameter(e.query, gl.QUERY_RESULT);
            renderProfile.gpu[e.pass] = (renderProfile.gpu[e.pass] || 0) + ns / 1e6;
            free.push(e.query);
            live.splice(i, 1);
        }
    }

    function beginView() {
        if (!renderProfile.enabled) return;
        if (!probed) probe();
        drain();
        renderProfile.views += 1;
        sampling = !!ext && renderProfile.views % renderProfile.gpuEvery === 0;
        if (sampling) renderProfile.gpuViews += 1;
    }

    function begin() {
        if (!renderProfile.enabled) return;
        t0 = performance.now();
        if (!sampling || open) return;
        open = free.pop() || gl.createQuery();
        gl.beginQuery(ext.TIME_ELAPSED_EXT, open);
    }

    // Called immediately after the pass's `renderer.render()`, which is when
    // `info.render.calls` still holds that pass's count (three.js resets it at the
    // START of each render).
    function end(pass) {
        if (!renderProfile.enabled) return;
        renderProfile.cpu[pass] = (renderProfile.cpu[pass] || 0) + (performance.now() - t0);
        renderProfile.calls[pass] =
            (renderProfile.calls[pass] || 0) + renderer.info.render.calls;
        if (!open) return;
        gl.endQuery(ext.TIME_ELAPSED_EXT);
        live.push({ pass, query: open });
        open = null;
    }

    return { beginView, begin, end };
}

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
// id (emissive.js / reflective.js / transmissive.js all match on it), decide
// reflectivity BEFORE the OIT patch (matteNonReflective keeps the curated
// reflective names and mattes the rest), then route genuinely see-through meshes
// onto the OIT layer. Returns prepareOITScene's `{ oitMeshes, forcedOpaque }` so
// the caller can report how the transmissivity gate classified the scene.
//
// `idIndex` (the tour's 1-based global object index — see idmask.js) additionally
// builds this object's ID materials. It must happen AFTER the OIT gate, since the
// layer a mesh lands on decides the alpha cutoff its mask uses; ordering it here
// is what keeps that invariant off the call sites. Stage 5 passes no index and
// allocates nothing.
export function prepareCaptureObject(root, id, idIndex = 0) {
    root.userData.objectId = id;
    matteNonReflective(root, id);
    const gate = prepareOITScene(root, oitPass);
    if (idIndex > 0) attachIdMaterials(root, idIndex);
    return gate;
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
    const prof = makePassProfiler(renderer);
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
        prof.begin();
        renderer.render(scene, cam);
        prof.end("opaque");
        // glass sub-passes (layer 1); keep the opaque colour/depth (no auto-clear).
        renderer.autoClear = false;
        cam.layers.set(OIT_LAYER);
        oitPass.value = 2; // depth pre-pass: α≥OIT_OPAQUE writes depth, no colour
        for (const m of oitMats) {
            m.depthWrite = true;
            m.colorWrite = false;
        }
        renderer.setRenderTarget(accumTarget);
        prof.begin();
        renderer.render(scene, cam);
        prof.end("glass");
        oitPass.value = 0; // accumulate (additive), depth-tested, no depth write
        for (const m of oitMats) {
            m.depthWrite = false;
            m.colorWrite = true;
            setOITBlend(m, true);
        }
        renderer.setRenderTarget(accumTarget);
        renderer.setClearColor(0x000000, 0);
        renderer.clear(true, false, false);
        prof.begin();
        renderer.render(scene, cam);
        prof.end("glass");
        oitPass.value = 1; // revealage: dst *= (1 − α)
        for (const m of oitMats) setOITBlend(m, false);
        renderer.setRenderTarget(revealTarget);
        renderer.setClearColor(0xffffff, 1);
        renderer.clear(true, false, false);
        prof.begin();
        renderer.render(scene, cam);
        prof.end("glass");
        // composite over opaque → rtColor (ACES + sRGB + coverage alpha).
        cam.layers.set(0);
        renderer.autoClear = true;
        renderer.setRenderTarget(rtColor);
        prof.begin();
        renderer.render(compositeScene, fsCamera);
        prof.end("present");
    }

    // Render `scene` through ANY camera into rtColor (+ rtDepth). This is the one
    // render path: the perspective view helper below and the tour's overhead
    // orthographic slices both come through here, so nothing can diverge.
    function renderCamera(scene, cam) {
        if (oitMats === null) oitMats = collectOITMaterials(scene);
        prof.beginView();
        if (oitMats.length === 0) {
            // Opaque scene: one lit pass → present. Unchanged fast path.
            renderer.setClearColor(bg, 0); // alpha 0 = empty coverage
            renderer.setRenderTarget(rtScene);
            prof.begin();
            renderer.render(scene, cam); // lit, linear HDR + depth
            prof.end("opaque");
            renderer.setRenderTarget(rtColor);
            prof.begin();
            renderer.render(presentScene, fsCamera); // ACES + sRGB → 8-bit
            prof.end("present");
        } else {
            renderOIT(scene, cam);
        }
        renderer.setRenderTarget(rtDepth);
        prof.begin();
        renderer.render(packScene, fsCamera); // depth → log-u16 codes
        prof.end("depth");
    }

    const lookTarget = new THREE.Vector3();
    // `viewFov` (degrees) renders this one view at its own angle. Stage 4 derives
    // FOV per camera from how far away its subject is, so a plan carries a range
    // rather than one value. Only near/far reach the depth-pack shader
    // (`z = 2·n·f / (f + n − zndc·(f − n))` is FOV-independent), so nothing but
    // the projection matrix has to change — and it is only rebuilt when the angle
    // actually differs, since consecutive views often share one.
    function aimCamera(pos, face, viewFov) {
        if (viewFov && Math.abs(viewFov - camera.fov) > 1e-6) {
            camera.fov = viewFov;
            camera.updateProjectionMatrix();
        }
        camera.position.set(pos[0], pos[1], pos[2]);
        camera.up.set(face.up[0], face.up[1], face.up[2]);
        lookTarget
            .set(face.forward[0], face.forward[1], face.forward[2])
            .add(camera.position);
        camera.lookAt(lookTarget);
    }

    function renderView(scene, pos, face, viewFov) {
        aimCamera(pos, face, viewFov);
        renderCamera(scene, camera);
    }

    // The object-ID twin of renderView: the same pose and the same projection
    // matrix, rendered flat into idmask.js's own RGBA8 target (returned, for the
    // caller to read back). Built on first use, so a capture that never asks for
    // masks — Stage 5 — allocates nothing. Sharing `camera` is the point: the mask
    // and the pano cannot rasterize differently if they can't disagree on a pose.
    //
    // `idSize` is deliberately NOT the colour face size. The ID pass is rendered
    // supersampled so the stitch can measure sub-texel coverage (ids themselves
    // can't be antialiased — see idmask.js), and since a 90° frustum is the same
    // frustum at any resolution, only the target changes.
    let idPass = null;
    function renderIdView(scene, pos, face, idSize) {
        if (idPass && idPass.size !== idSize) {
            idPass.dispose();
            idPass = null;
        }
        if (!idPass) idPass = createIdRenderer(renderer, idSize);
        aimCamera(pos, face);
        return idPass.render(scene, camera);
    }

    function setLinear(on) {
        presentMaterial.uniforms.uLinear.value = !!on;
    }

    // For short-lived instances (the tour's per-level minimap slices). Stage 5
    // holds one capture for the whole run and never calls this.
    function dispose() {
        for (const t of [rtScene, rtColor, rtDepth, accumTarget, revealTarget]) t?.dispose();
        depthTexture.dispose();
        idPass?.dispose();
        for (const m of [presentMaterial, packMaterial, compositeMaterial]) m.dispose();
        for (const g of fsGeometries) g.dispose();
    }

    return {
        rtColor,
        rtDepth,
        renderView,
        renderIdView,
        renderCamera,
        camera,
        setLinear,
        dispose,
    };
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
