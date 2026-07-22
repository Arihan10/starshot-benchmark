// Shared lit-rendering rig for the splat asset tier — the ONE source of truth
// for the "what gets trained" look, used by the headless reference capture
// (splatcapture.js), the splat debug viewer's "original" mesh mode
// (splatviewer.js), and the main mesh viewer (scene3d.js). Mirrors the fixed
// bake rig in splat/stage5.py (LIGHTING): image-based ambient (RoomEnvironment)
// + a hemisphere fill + one shadow-casting sun, shaded in linear light and
// displayed through an ACES-filmic + sRGB transform.

import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const SHADOW_MAP_SIZE = 4096;

// Client-side source of truth (mirrors splat/stage5.py LIGHTING). `shadow` is the
// ground shadow-catcher opacity — only used by rigs built with `catcher: true`
// (the main viewer); the capture + splat viewer bake shadows onto the meshes
// themselves and ignore it. Angles: 0° azimuth = +Z (front), 90° = +X (right).
export const LIGHTING_DEFAULTS = {
    exposure: 1.0,
    key: 3.5,
    fill: 0.2,
    env: 0.35,
    shadow: 0.4,
    azimuth: 34,
    elevation: 48,
    shadows: true,
};

// Accept either the client schema (azimuth/elevation) or the server/transforms
// schema (azimuth_deg/elevation_deg) and return the client schema, filled from
// the defaults. Extra keys (e.g. tone_mapping) are ignored.
export function normalizeLighting(raw) {
    const o = { ...LIGHTING_DEFAULTS };
    if (raw && typeof raw === "object") {
        for (const k of ["exposure", "key", "fill", "env", "shadow", "shadows"]) {
            if (raw[k] != null) o[k] = raw[k];
        }
        o.azimuth = raw.azimuth ?? raw.azimuth_deg ?? o.azimuth;
        o.elevation = raw.elevation ?? raw.elevation_deg ?? o.elevation;
    }
    return o;
}

// Matte-PBR conversion for ONE splat-tier glTF material. The tier strips the
// metallic-roughness map and glTF's default metalnessFactor is 1 (fully metal →
// near-black under an env), so pin a matte dielectric (metalness 0, roughness 1)
// for clean diffuse shading. Base-color map is sRGB (decoded for shading);
// DoubleSide since Trellis winding is unreliable; BLEND surfaces keep depthWrite
// off so the depth behind glass survives.
export function toLitMaterial(orig) {
    const m = new THREE.MeshStandardMaterial();
    m.map = orig.map || null;
    if (m.map) m.map.colorSpace = THREE.SRGBColorSpace;
    if (orig.color) m.color.copy(orig.color);
    m.metalness = 0.0;
    m.roughness = 1.0;
    m.opacity = orig.opacity ?? 1;
    m.transparent = orig.transparent === true;
    m.alphaTest = orig.alphaTest || 0;
    m.side = THREE.DoubleSide;
    m.depthWrite = !m.transparent;
    m.vertexColors = orig.vertexColors === true;
    return m;
}

// In-place: give a loaded glTF scene the normals shading needs (generated meshes
// often ship without them), shadow flags, and matte-lit materials (disposing the
// originals). The lit twin of splatviewer's old `makeUnlit`.
export function prepareLitScene(root) {
    root.traverse((o) => {
        if (!o.isMesh || !o.material) return;
        if (o.geometry && !o.geometry.getAttribute("normal")) {
            o.geometry.computeVertexNormals();
        }
        o.castShadow = true;
        o.receiveShadow = true;
        const orig = o.material;
        o.material = Array.isArray(orig) ? orig.map(toLitMaterial) : toLitMaterial(orig);
        for (const mm of Array.isArray(orig) ? orig : [orig]) mm.dispose();
    });
}

// Interactive display transform (linear-shaded scene → ACES-filmic → sRGB, with
// shadow maps). The headless capture does this in its own present-pass shader for
// deterministic bytes, so it does NOT call this. Returns the previous renderer
// state so a caller can scope the transform to one view mode and restore it.
export function applyMeshToneMapping(renderer) {
    const prev = {
        toneMapping: renderer.toneMapping,
        toneMappingExposure: renderer.toneMappingExposure,
        outputColorSpace: renderer.outputColorSpace,
        shadowEnabled: renderer.shadowMap.enabled,
        shadowType: renderer.shadowMap.type,
    };
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    return prev;
}

export function restoreToneMapping(renderer, prev) {
    if (!prev) return;
    renderer.toneMapping = prev.toneMapping;
    renderer.toneMappingExposure = prev.toneMappingExposure;
    renderer.outputColorSpace = prev.outputColorSpace;
    renderer.shadowMap.enabled = prev.shadowEnabled;
    renderer.shadowMap.type = prev.shadowType;
}

// Build the fixed IBL + hemisphere + shadow-casting sun rig on `scene`, fit to a
// scene bounding box. Options:
//   catcher        — add a transparent ground ShadowMaterial plane (main viewer);
//                    the capture + splat viewer bake shadows onto meshes only.
//   keepCasterOn   — never toggle the sun's castShadow from the `shadows` flag
//                    (the main viewer gates RECEPTION through its own uniform via
//                    `onShadows` instead, avoiding shader recompiles); default
//                    off, so `shadows: false` disables the caster entirely.
//   decorateLights — hook to tweak the created lights, e.g. layers.enableAll().
//   onShadows      — called with the shadows flag on every apply (uniform gating).
//   defaults       — initial lighting config (client or server schema).
// Returns { key, hemi, catcher, setLighting, refit, getLighting, setEnabled,
//           dispose }.
export function createLightRig(renderer, scene, opts = {}) {
    const {
        catcher: withCatcher = false,
        keepCasterOn = false,
        decorateLights = null,
        onShadows = null,
        defaults = LIGHTING_DEFAULTS,
    } = opts;
    const state = normalizeLighting(defaults);

    const pmrem = new THREE.PMREMGenerator(renderer);
    const envScene = new RoomEnvironment(renderer);
    const envTex = pmrem.fromScene(envScene, 0.04).texture;
    envScene.dispose();
    pmrem.dispose();
    scene.environment = envTex;

    const hemi = new THREE.HemisphereLight(0xffffff, 0x202028, state.fill);
    const key = new THREE.DirectionalLight(0xffffff, state.key);
    key.castShadow = true;
    key.shadow.mapSize.set(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
    key.shadow.bias = -0.0001;
    if (decorateLights) decorateLights({ hemi, key });
    scene.add(hemi, key, key.target);

    let catcher = null;
    if (withCatcher) {
        catcher = new THREE.Mesh(
            new THREE.PlaneGeometry(1, 1),
            new THREE.ShadowMaterial({ opacity: state.shadow, depthWrite: false }),
        );
        catcher.rotation.x = -Math.PI / 2;
        catcher.receiveShadow = true;
        catcher.renderOrder = -1;
        scene.add(catcher);
    }

    const lightDir = new THREE.Vector3(4, 8, 6).normalize();
    const center = new THREE.Vector3();
    const dim = new THREE.Vector3();
    let lastBox = null;
    let enabled = true;

    // Fit the ortho shadow frustum to the scene's bounding sphere (depth precision
    // spent tightly on it); place the sun `dist` back along `lightDir` from center.
    function fitShadow(box) {
        const hasGeom = !!box && !box.isEmpty();
        lastBox = hasGeom ? box : null;
        if (hasGeom) {
            box.getCenter(center);
            box.getSize(dim);
        } else {
            center.set(0, 0, 0);
            dim.set(20, 20, 20);
        }
        const radius = Math.max(0.5, 0.5 * Math.hypot(dim.x, dim.y, dim.z));
        const minY = hasGeom ? box.min.y : 0;
        const dist = radius * 3;
        key.position.copy(center).addScaledVector(lightDir, dist);
        key.target.position.copy(center);
        key.target.updateMatrixWorld();
        const cam = key.shadow.camera;
        const extent = radius * 1.05;
        cam.left = -extent;
        cam.right = extent;
        cam.top = extent;
        cam.bottom = -extent;
        cam.near = Math.max(0.01, dist - radius * 1.1);
        cam.far = dist + radius * 1.1;
        cam.updateProjectionMatrix();
        // Normal-offset bias scaled to the shadow texel's world size — the main
        // defense against self-shadow acne, consistent across scene scales.
        key.shadow.normalBias = ((2 * extent) / SHADOW_MAP_SIZE) * 2.0;
        if (catcher) {
            catcher.position.set(center.x, minY + radius * 0.003, center.z);
            catcher.scale.set(radius * 6, radius * 6, 1);
        }
    }

    function applyAngles() {
        const az = THREE.MathUtils.degToRad(state.azimuth);
        const el = THREE.MathUtils.degToRad(state.elevation);
        const cosEl = Math.cos(el);
        lightDir.set(Math.sin(az) * cosEl, Math.sin(el), Math.cos(az) * cosEl).normalize();
        fitShadow(lastBox);
    }

    function applyScalars() {
        renderer.toneMappingExposure = state.exposure;
        key.intensity = state.key;
        hemi.intensity = state.fill;
        scene.environmentIntensity = state.env;
        if (!keepCasterOn) key.castShadow = state.shadows;
        if (catcher) {
            catcher.material.opacity = state.shadow;
            catcher.visible = enabled && state.shadows;
        }
        if (onShadows) onShadows(state.shadows);
    }

    applyScalars();
    applyAngles();
    fitShadow(null);

    return {
        key,
        hemi,
        catcher,
        setLighting(partial) {
            Object.assign(state, partial);
            applyScalars();
            applyAngles();
        },
        refit(box) {
            fitShadow(box);
        },
        getLighting: () => ({ ...state }),
        // Toggle the whole rig (IBL + lights + catcher) so a viewer can scope it
        // to one view mode; leaves `state` intact for the next enable.
        setEnabled(on) {
            enabled = !!on;
            scene.environment = enabled ? envTex : null;
            hemi.visible = enabled;
            key.visible = enabled;
            if (catcher) catcher.visible = enabled && state.shadows;
        },
        dispose() {
            scene.remove(hemi, key, key.target);
            if (catcher) {
                scene.remove(catcher);
                catcher.geometry.dispose();
                catcher.material.dispose();
            }
            if (scene.environment === envTex) scene.environment = null;
            envTex.dispose();
        },
    };
}
