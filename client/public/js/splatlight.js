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
		for (const k of [
			"exposure",
			"key",
			"fill",
			"env",
			"shadow",
			"shadows",
		]) {
			if (raw[k] != null) o[k] = raw[k];
		}
		o.azimuth = raw.azimuth ?? raw.azimuth_deg ?? o.azimuth;
		o.elevation = raw.elevation ?? raw.elevation_deg ?? o.elevation;
	}
	return o;
}

// Splat-tier material prep for ONE glTF material, applied IN PLACE. The capture
// bakes VIEW-INDEPENDENT lighting only: we force the surface to a matte
// dielectric so the shading is purely diffuse (sun + hemisphere fill + diffuse
// IBL irradiance) plus shadows — identical from every camera. Metallic-roughness
// reflections and specular highlights are the view-dependent effects we
// deliberately drop: a flat (SH-free, degree-0) splat cannot carry them, so if
// they were baked per view they would only smear into a dull average. Base
// colour, alpha/transparency, normal, AO and emissive are kept untouched.
// Plus two structural tweaks: DoubleSide (Trellis winding is unreliable) and
// depthWrite off for BLEND so the depth behind glass survives.
export function toLitMaterial(orig) {
	orig.side = THREE.DoubleSide;
	orig.depthWrite = orig.transparent !== true;
	// Matte dielectric: kill the specular lobe's view dependence. Roughness 1 +
	// metalness 0 turns the env map into flat (irradiance-like) ambient instead
	// of a moving reflection, and drops the sun's sharp highlight.
	orig.metalness = 0;
	orig.roughness = 1;
	orig.metalnessMap = null;
	orig.roughnessMap = null;
	// Secondary specular lobes some glTF (MeshPhysicalMaterial) surfaces add are
	// all pure view-dependent gloss — neutralize them when present.
	if ("clearcoat" in orig) orig.clearcoat = 0;
	if ("sheen" in orig) orig.sheen = 0;
	if ("specularIntensity" in orig) orig.specularIntensity = 0;
	if ("iridescence" in orig) orig.iridescence = 0;
	orig.needsUpdate = true;
	return orig;
}

// Alpha at/above which a transparent fragment counts as SOLID — it writes depth
// and occludes (window frames, mirrors, mostly-opaque panels) — while genuine
// glass below it keeps blending and stays see-through. Mirrors scene3d.js's
// weighted-blended OIT `OIT_OPAQUE`, so the mesh preview + the headless capture
// match the board viewer without its multi-pass OIT (which needs render-loop
// control the mkkellogg preview loop and the capture don't have here).
const OIT_OPAQUE = 0.8;

// A colour-less, alpha-tested DEPTH material matched to a transparent source. It
// renders in the OPAQUE queue and writes depth ONLY where the source's alpha
// (base-colour texel × factor) reaches OIT_OPAQUE, so solid regions occlude
// per-fragment while sub-cutoff glass writes no depth and stays see-through. It
// shares the source `map`, so the same per-texel alpha decides (glass.py bakes
// panes to ~0.065 inside an otherwise-opaque frame) — the single-pass equivalent
// of scene3d's OIT depth pre-pass. `opacity` carries constant-alpha glass with no map.
function depthProxyMaterial(orig) {
	const m = new THREE.MeshBasicMaterial();
	m.map = orig.map || null;
	m.opacity = orig.opacity ?? 1;
	m.alphaTest = OIT_OPAQUE;
	m.side = THREE.DoubleSide;
	m.colorWrite = false; // depth only — never contributes colour
	m.depthWrite = true;
	m.depthTest = true;
	m.transparent = false; // opaque queue → runs before the blend pass
	return m;
}

// Aligns a multi-material mesh's proxy array with its geometry groups: an opaque
// sub-material already writes depth in its own pass, so its proxy slot is a no-op.
function nullProxyMaterial() {
	return new THREE.MeshBasicMaterial({
		colorWrite: false,
		depthWrite: false,
		depthTest: false,
	});
}

// In-place: give a loaded glTF scene the normals shading needs (generated meshes
// often ship without them), shadow flags, and the splat-tier material prep (see
// toLitMaterial). Materials are forced matte, so the capture bakes VIEW-
// INDEPENDENT diffuse lighting; normal/AO/emissive maps still apply. Transparent
// meshes also get a child DEPTH PROXY
// (see depthProxyMaterial) so their solid parts occlude the scene behind — the fix
// for mirror/opaque panels rendering see-through and the wall/floor bleeding
// through them, without a full multi-pass OIT. The lit twin of splatviewer's old
// `makeUnlit`.
export function prepareLitScene(root) {
	const attach = []; // [mesh, proxy]; added after the walk, not mid-traverse
	root.traverse((o) => {
		if (!o.isMesh || !o.material) return;
		if (o.geometry && !o.geometry.getAttribute("normal")) {
			o.geometry.computeVertexNormals();
		}
		o.castShadow = true;
		o.receiveShadow = true;
		const mats = Array.isArray(o.material) ? o.material : [o.material];
		for (const m of mats) toLitMaterial(m);
		if (mats.some((m) => m.transparent === true)) {
			const proxyMat = Array.isArray(o.material)
				? o.material.map((m) =>
						m.transparent ? depthProxyMaterial(m) : nullProxyMaterial(),
					)
				: depthProxyMaterial(o.material);
			const proxy = new THREE.Mesh(o.geometry, proxyMat);
			proxy.castShadow = false;
			proxy.receiveShadow = false;
			proxy.frustumCulled = o.frustumCulled;
			proxy.userData.__depthProxy = true;
			attach.push([o, proxy]);
		}
	});
	for (const [mesh, proxy] of attach) mesh.add(proxy);
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
			new THREE.ShadowMaterial({
				opacity: state.shadow,
				depthWrite: false,
			}),
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
		lightDir
			.set(Math.sin(az) * cosEl, Math.sin(el), Math.cos(az) * cosEl)
			.normalize();
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
