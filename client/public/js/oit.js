// Weighted-blended order-independent transparency (OIT), shared by the mesh
// renderers so they agree fragment-for-fragment: the headless Stage-5 capture
// (splatcapture.js) and the debug splat viewer's mesh mode (splatviewer.js). A
// single alpha-blended pass is draw-order dependent and cannot show objects
// behind concave / double-sided glass (they go black); WBOIT composites them
// correctly regardless of order. Ported from the board viewer (scene3d.js),
// which keeps its own inline copy.

import * as THREE from "three";

export const OIT_LAYER = 1;
// α at/above which a transparent fragment counts as SOLID (writes depth + occludes):
// window frames/mullions stay opaque, sub-cutoff glass blends and stays see-through.
export const OIT_OPAQUE = 0.8;

// TEMPORARY (view-independent training): bake the lighting by forcing every surface
// matte — metalness 0, roughness 1 — so the fixed rig contributes only diffuse
// shading + soft IBL irradiance + shadows, with NO camera-dependent specular
// highlights or environment mirror reflections (the moving "glare" a degree-0 /
// low-SH splat can't fit). This makes every reference frame view-consistent.
// REVERT when spherical-harmonic / view-dependent training returns: set to false
// (restores the assets' authored metallic-roughness) AND bump COLOR_PIPELINE in
// splat/stage5.py so the matte frames re-render.
export const BAKE_MATTE = true;

// ACES-filmic + sRGB (three.js' exact fit), shared by the opaque present pass and
// the OIT composite so both paths store identical display colour.
export const TONEMAP_GLSL = /* glsl */ `
    vec3 acesFilmic(vec3 x) {
        const mat3 inMat = mat3(
            vec3(0.59719, 0.07600, 0.02840),
            vec3(0.35458, 0.90834, 0.13383),
            vec3(0.04823, 0.01566, 0.83777)
        );
        const mat3 outMat = mat3(
            vec3( 1.60475, -0.10208, -0.00327),
            vec3(-0.53108,  1.10813, -0.07276),
            vec3(-0.07367, -0.00605,  1.07602)
        );
        x = inMat * x;
        vec3 a = x * (x + 0.0245786) - 0.000090537;
        vec3 b = x * (0.983729 * x + 0.4329510) + 0.238081;
        x = a / b;
        x = outMat * x;
        return clamp(x, 0.0, 1.0);
    }
    vec3 linearToSrgb(vec3 c) {
        c = clamp(c, 0.0, 1.0);
        vec3 lo = c * 12.92;
        vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
        return mix(lo, hi, step(vec3(0.0031308), c));
    }
`;

// Patch one transparent material for OIT. `oitPass` is the caller's shared {value}
// selector (0 accumulate · 1 revealage · 2 depth pre-pass). Verbatim from
// scene3d.js so every renderer agrees fragment-for-fragment.
export function patchMaterialOIT(m, oitPass) {
    if (m.userData.__oitPatched) return;
    m.userData.__oitPatched = true;
    m.transparent = true;
    m.depthWrite = false;
    m.depthTest = true;
    m.blending = THREE.CustomBlending;
    m.blendEquation = THREE.AddEquation;
    m.blendEquationAlpha = THREE.AddEquation;
    const prev = m.onBeforeCompile;
    m.onBeforeCompile = (shader, rndr) => {
        if (prev) prev(shader, rndr);
        shader.uniforms.uOITPass = oitPass;
        shader.fragmentShader = shader.fragmentShader
            .replace(
                "#include <common>",
                "#include <common>\nuniform float uOITPass;",
            )
            .replace(
                "#include <dithering_fragment>",
                `#include <dithering_fragment>
                float _a = gl_FragColor.a;
                if (uOITPass > 1.5) { if (_a < ${OIT_OPAQUE}) discard; }
                else {
                    float _ac = _a >= ${OIT_OPAQUE} ? 1.0 : _a;
                    if (uOITPass < 0.5) {
                        float _w = clamp(pow(min(1.0, _ac * 10.0) + 0.01, 3.0) * 1e8 * pow(1.0 - gl_FragCoord.z * 0.9, 3.0), 1e-2, 3e3);
                        gl_FragColor = vec4(gl_FragColor.rgb * _ac * _w, _ac * _w);
                    } else {
                        gl_FragColor = vec4(_ac);
                    }
                }`,
            );
    };
    const prevKey = m.customProgramCacheKey?.bind(m);
    m.customProgramCacheKey = () => "oit1|" + (prevKey ? prevKey() : "");
    m.needsUpdate = true;
}

// accum: additive (ONE, ONE); revealage: dst *= (1 − srcAlpha).
export function setOITBlend(m, accum) {
    m.blendSrc = accum ? THREE.OneFactor : THREE.ZeroFactor;
    m.blendDst = accum ? THREE.OneFactor : THREE.OneMinusSrcAlphaFactor;
    m.blendSrcAlpha = m.blendSrc;
    m.blendDstAlpha = m.blendDst;
}

// In-place lit-scene prep: generate missing normals, cast/receive shadows, force
// DoubleSide with back-face shadowing (Trellis winding is unreliable), and route
// transparent meshes onto the OIT layer with the weighted-blend patch. PBR maps
// are kept whole so metals/glossies still reflect; opaque materials keep their
// default depth write.
export function prepareOITScene(root, oitPass) {
    root.traverse((child) => {
        if (!child.isMesh || !child.material) return;
        if (child.geometry && !child.geometry.getAttribute("normal")) {
            child.geometry.computeVertexNormals();
        }
        child.castShadow = true;
        child.receiveShadow = true;
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        let oit = false;
        for (const m of mats) {
            m.side = THREE.DoubleSide;
            m.shadowSide = THREE.BackSide;
            // Bake lighting view-independently: matte kills the specular lobe and
            // turns the IBL into blurred (view-independent) irradiance — no glare.
            if (BAKE_MATTE && (m.isMeshStandardMaterial || m.isMeshPhysicalMaterial)) {
                m.metalness = 0;
                m.roughness = 1;
            }
            if (m.transparent) oit = true;
        }
        if (oit) {
            for (const m of mats) patchMaterialOIT(m, oitPass);
            child.layers.set(OIT_LAYER);
            child.userData.__oit = true;
        }
    });
}

// The transparent (OIT) materials under `root`, gathered for the per-pass state
// toggling in the render loop. Empty → the scene is opaque and the caller can take
// its single-pass fast path.
export function collectOITMaterials(root) {
    const mats = [];
    root.traverse((o) => {
        if (!o.userData.__oit || !o.material) return;
        if (Array.isArray(o.material)) mats.push(...o.material);
        else mats.push(o.material);
    });
    return mats;
}

// Glass is moved onto OIT_LAYER, so the sun + hemisphere fill must illuminate that
// layer too (a three light only affects layers it shares with the object, and
// OIT_LAYER is not the default). Pass as createLightRig's `decorateLights` hook.
export function decorateOITLights({ hemi, key }) {
    hemi.layers.enableAll();
    key.layers.enableAll();
}

// Screen compositor for the interactive mesh preview (splatviewer.js): render the
// opaque scene to a linear-HDR target, then present it (opaque scene) or resolve
// the glass through the WBOIT sub-passes, tone-mapping to the canvas with
// TONEMAP_GLSL so the preview matches the Stage-5 capture. `oitPass` is the same
// selector the scene's materials were patched with; exposure is read from
// renderer.toneMappingExposure (set by the shared light rig). Mirrors scene3d.js's
// renderFrame, minus the depth/coverage readback the headless capture needs.
export function createScreenOIT(renderer, oitPass) {
    const _size = new THREE.Vector2();
    const _clear = new THREE.Color();
    let opaqueTarget = null;
    let accumTarget = null;
    let revealTarget = null;

    const fullscreenVS = /* glsl */ `
        varying vec2 vUv;
        void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
    `;
    const present = new THREE.ShaderMaterial({
        uniforms: { tScene: { value: null }, uExposure: { value: 1 } },
        vertexShader: fullscreenVS,
        fragmentShader: /* glsl */ `
            precision highp float;
            uniform sampler2D tScene;
            uniform float uExposure;
            varying vec2 vUv;
            ${TONEMAP_GLSL}
            void main() {
                vec4 s = texture2D(tScene, vUv);
                gl_FragColor = vec4(linearToSrgb(acesFilmic(s.rgb * uExposure)), 1.0);
            }
        `,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
    });
    const compose = new THREE.ShaderMaterial({
        uniforms: {
            uOpaque: { value: null },
            uAccum: { value: null },
            uReveal: { value: null },
            uExposure: { value: 1 },
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
                vec3 oit = accum.rgb / max(accum.a, 1e-5);
                vec3 lin = mix(oit, texture2D(uOpaque, vUv).rgb, reveal);
                gl_FragColor = vec4(linearToSrgb(acesFilmic(lin * uExposure)), 1.0);
            }
        `,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
    });

    const quadScene = new THREE.Scene();
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), present);
    quad.frustumCulled = false;
    quadScene.add(quad);
    const quadCamera = new THREE.Camera();

    function disposeTargets() {
        if (!opaqueTarget) return;
        const depth = opaqueTarget.depthTexture; // shared across the three targets
        opaqueTarget.dispose();
        accumTarget.dispose();
        revealTarget.dispose();
        depth?.dispose();
        opaqueTarget = accumTarget = revealTarget = null;
    }

    function ensureTargets() {
        renderer.getDrawingBufferSize(_size);
        const w = Math.max(1, _size.x | 0);
        const h = Math.max(1, _size.y | 0);
        if (opaqueTarget && opaqueTarget.width === w && opaqueTarget.height === h) return;
        disposeTargets();
        const depthTexture = new THREE.DepthTexture(w, h);
        const opts = {
            type: THREE.HalfFloatType,
            minFilter: THREE.NearestFilter,
            magFilter: THREE.NearestFilter,
            depthTexture,
        };
        opaqueTarget = new THREE.WebGLRenderTarget(w, h, opts);
        accumTarget = new THREE.WebGLRenderTarget(w, h, opts);
        revealTarget = new THREE.WebGLRenderTarget(w, h, opts);
    }

    // Draw `scene` from `camera` to the canvas with OIT-correct glass. `oitMats` is
    // the scene's transparent materials (collectOITMaterials); empty runs the
    // single opaque pass.
    function render(scene, camera, oitMats) {
        const exposure = renderer.toneMappingExposure;
        renderer.getClearColor(_clear);
        const prevAlpha = renderer.getClearAlpha();
        const prevAutoClear = renderer.autoClear;
        ensureTargets();

        // opaque (layer 0) → opaqueTarget: linear colour + the depth glass tests against.
        camera.layers.set(0);
        renderer.autoClear = true;
        renderer.setRenderTarget(opaqueTarget);
        renderer.setClearColor(_clear, 1);
        renderer.render(scene, camera);

        if (!oitMats || oitMats.length === 0) {
            present.uniforms.tScene.value = opaqueTarget.texture;
            present.uniforms.uExposure.value = exposure;
            quad.material = present;
            renderer.autoClear = prevAutoClear;
            renderer.setRenderTarget(null);
            renderer.setClearColor(_clear, prevAlpha);
            renderer.render(quadScene, quadCamera);
            return;
        }

        // glass sub-passes (layer 1), reusing the opaque depth + shadow map.
        const prevShadowAuto = renderer.shadowMap.autoUpdate;
        renderer.autoClear = false;
        renderer.shadowMap.autoUpdate = false;
        camera.layers.set(OIT_LAYER);
        oitPass.value = 2; // depth pre-pass: α≥OIT_OPAQUE writes depth, no colour
        for (const m of oitMats) {
            m.depthWrite = true;
            m.colorWrite = false;
        }
        renderer.setRenderTarget(accumTarget);
        renderer.render(scene, camera);
        oitPass.value = 0; // accumulate (additive), depth-tested, no depth write
        for (const m of oitMats) {
            m.depthWrite = false;
            m.colorWrite = true;
            setOITBlend(m, true);
        }
        renderer.setRenderTarget(accumTarget);
        renderer.setClearColor(0x000000, 0);
        renderer.clear(true, false, false);
        renderer.render(scene, camera);
        oitPass.value = 1; // revealage: dst *= (1 − α)
        for (const m of oitMats) setOITBlend(m, false);
        renderer.setRenderTarget(revealTarget);
        renderer.setClearColor(0xffffff, 1);
        renderer.clear(true, false, false);
        renderer.render(scene, camera);

        // composite over opaque → canvas (ACES + sRGB).
        camera.layers.set(0);
        renderer.shadowMap.autoUpdate = prevShadowAuto;
        renderer.autoClear = prevAutoClear;
        renderer.setRenderTarget(null);
        renderer.setClearColor(_clear, prevAlpha);
        compose.uniforms.uOpaque.value = opaqueTarget.texture;
        compose.uniforms.uAccum.value = accumTarget.texture;
        compose.uniforms.uReveal.value = revealTarget.texture;
        compose.uniforms.uExposure.value = exposure;
        quad.material = compose;
        renderer.render(quadScene, quadCamera);
    }

    function dispose() {
        disposeTargets();
        present.dispose();
        compose.dispose();
        quad.geometry.dispose();
    }

    return { render, dispose };
}
