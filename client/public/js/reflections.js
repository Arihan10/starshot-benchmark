// Per-reflective-object scene reflection probes for the Stage-5 capture.
//
// The capture renders LIT PBR (splatcapture.js) so metals / water / glossies show
// view-dependent reflections that Stage-6 degree-3 SH reconstructs. Without a
// scene-aware environment those reflections would only show the generic
// RoomEnvironment IBL (a studio that isn't in the scene). This module bakes a REAL
// reflection of the scene into each reflective object:
//
//   * find the curated reflective objects (reflective.js name discriminator),
//   * bake a cube map of the surrounding OPAQUE scene from the object's reflective
//     centroid — the object hidden so it doesn't occlude/reflect itself — under
//     the SAME fixed rig (sun + shadows + RoomEnvironment IBL) the frames use,
//   * PMREM-prefilter it and assign it as that object's material `envMap`.
//
// Baked ONCE before the render loop (the scene and the lighting are both static),
// so it adds a small one-time cost to Stage 5 and NOTHING to Stage 6 — the
// reflections are baked into the reference frames, exactly the view-dependent
// signal the splat trains on. Cube contents are OPAQUE-only: glass lives on the
// OIT layer (excluded by the default camera layer mask), which also sidesteps the
// weighted-blended OIT passes a plain cube render can't reproduce. Empty
// directions clear to the capture background — what a camera at that point would
// actually see — so reflections stay consistent with the frames' void.

import * as THREE from "three";
import { OIT_LAYER, OIT_PASS_OPAQUE } from "./oit.js";
import { isReflectiveName } from "./reflective.js";

// Defaults mirrored by splat/stage5.py LIGHTING["reflections"] (the server-side
// twin recorded in capture.json for stale-frame invalidation).
export const REFLECTION_DEFAULTS = {
	enabled: true,
	maxRough: 0.5, // legacy — reflectivity is now name-gated (reflective.js); kept for stage5/capture.json parity
	cubeSize: 256, // per-probe cube-face resolution (PMREM prefilters/downsamples)
	maxProbes: 24, // cap probes (each = 6 opaque scene renders + one PMREM)
	intensity: 1.0, // envMapIntensity applied to the reflective materials
	near: 0.05,
	far: 2000,
};

// The MeshStandard materials on one (curated reflective) object subtree + their
// shared world-space bbox. NO roughness gate: Trellis writes scalar
// metalness=roughness=1 on everything, so reflectivity is decided per-object by
// NAME (reflective.js) upstream; each kept material's own metallic-roughness map
// then modulates how sharply it mirrors the baked scene probe.
function scanObject(root) {
	const mats = new Set();
	const box = new THREE.Box3();
	root.traverse((o) => {
		if (!o.isMesh || !o.material) return;
		const list = Array.isArray(o.material) ? o.material : [o.material];
		let any = false;
		for (const m of list) {
			if (m && m.isMeshStandardMaterial) {
				mats.add(m);
				any = true;
			}
		}
		if (any) box.expandByObject(o);
	});
	if (mats.size === 0 || box.isEmpty()) return null;
	return { mats, center: box.getCenter(new THREE.Vector3()) };
}

// Bake per-reflective-object scene-reflection env maps. `background` is the
// capture's [r,g,b] clear (void reflects it, matching the frames). Returns
// `{ probes, dispose }`. Safe no-op when disabled or when nothing is reflective.
export function bakeReflectionProbes(renderer, scene, opts = {}) {
	const o = { ...REFLECTION_DEFAULTS, ...opts };
	if (!o.enabled) return { probes: 0, dispose() {} };

	scene.updateMatrixWorld(true); // centroids + culling need current world matrices

	// One probe per top-level object that holds a reflective mesh, at its reflective
	// centroid. `root` scopes WHICH children are the per-object groups: the capture
	// adds one gltf.scene per object straight to `scene` (root = scene), while the
	// debug viewer wraps them all in one meshGroup (root = meshGroup) — passing it
	// makes both probe per real object, not once for the whole wrapper. The cube
	// always renders the full `scene` (lights + everything), so reflections see the
	// entire world regardless.
	const objectsRoot = o.root || scene;
	const targets = [];
	for (const child of objectsRoot.children) {
		if (child.isLight || child.isCamera) continue;
		if (!isReflectiveName(child.userData?.objectId)) continue; // curated reflective only
		const found = scanObject(child);
		if (found) targets.push({ object: child, ...found });
		if (targets.length >= o.maxProbes) break;
	}
	if (targets.length === 0) return { probes: 0, dispose() {} };

	const pmrem = new THREE.PMREMGenerator(renderer);
	pmrem.compileCubemapShader();

	// Save renderer state the bake perturbs (CubeCamera.update restores the render
	// target + tone mapping itself, but uses the current clear colour to fill void).
	const prevClear = new THREE.Color();
	renderer.getClearColor(prevClear);
	const prevAlpha = renderer.getClearAlpha();
	const prevAutoClear = renderer.autoClear;
	const prevTarget = renderer.getRenderTarget();
	const bg = Array.isArray(o.background)
		? new THREE.Color(o.background[0], o.background[1], o.background[2])
		: new THREE.Color(0, 0, 0);
	renderer.autoClear = true;
	renderer.setClearColor(bg, 1);

	// The meshes prepareOITScene put on the OIT layer carry additive OIT blending,
	// which a plain cube render cannot reproduce. Render them into the cubes as
	// OPAQUE geometry instead: flip each to normal blending + depth write, switch the
	// shared `oitPass` to the opaque passthrough, and let the cube cameras see the
	// OIT layer too. Restored after the bake.
	//
	// This used to be the difference between a working probe and a black one, because
	// Trellis's blanket alphaMode=BLEND put ENTIRE scenes on the OIT layer and a
	// layer-0-only probe therefore captured nothing. The transmissivity gate
	// (transmissive.js) now keeps ordinary surfaces on layer 0, so the stakes are
	// lower — but real glass still lives here and a probe that ignored it would
	// reflect windows as holes.
	const oitMats = [];
	scene.traverse((obj) => {
		const list = obj.material
			? Array.isArray(obj.material)
				? obj.material
				: [obj.material]
			: [];
		for (const m of list) {
			if (m && m.userData && m.userData.__oitPatched && !oitMats.includes(m)) oitMats.push(m);
		}
	});
	const oitSaved = oitMats.map((m) => ({
		m,
		blending: m.blending,
		depthWrite: m.depthWrite,
		depthTest: m.depthTest,
	}));
	for (const m of oitMats) {
		m.blending = THREE.NormalBlending;
		m.depthWrite = true;
		m.depthTest = true;
	}
	const prevOitPass = o.oitPass ? o.oitPass.value : null;
	if (o.oitPass) o.oitPass.value = OIT_PASS_OPAQUE;

	// The caller bakes the (static) shadow map with ALL casters first (oit.js
	// bakeShadowMap, autoUpdate then off); freeze autoUpdate here too so the
	// per-probe self-hides below can never re-bake it — every cube face reuses that
	// one complete shadow map (so reflections carry the scene's shadows).
	const prevShadowAuto = renderer.shadowMap.autoUpdate;
	renderer.shadowMap.autoUpdate = false;

	const owned = []; // { envRT, mats } — keep the PMREM targets (they ARE the envMaps)
	for (const t of targets) {
		const cubeRT = new THREE.WebGLCubeRenderTarget(o.cubeSize, {
			type: THREE.HalfFloatType, // scene is linear HDR; keep highlights for the reflection
		});
		const cam = new THREE.CubeCamera(o.near, o.far, cubeRT);
		// See BOTH the opaque layer and the OIT layer (the latter now rendered as
		// opaque via the passthrough above) — otherwise all-BLEND scenes bake black.
		for (const c of cam.children) {
			c.layers.set(0);
			c.layers.enable(OIT_LAYER);
		}
		scene.add(cam);
		cam.position.copy(t.center);
		cam.updateMatrixWorld(true);
		const wasVisible = t.object.visible;
		t.object.visible = false; // don't reflect/occlude self from its own centroid
		cam.update(renderer, scene);
		t.object.visible = wasVisible;
		scene.remove(cam);

		const envRT = pmrem.fromCubemap(cubeRT.texture);
		for (const m of t.mats) {
			m.envMap = envRT.texture;
			m.envMapIntensity = o.intensity;
			m.needsUpdate = true;
		}
		cubeRT.dispose(); // the PMREM env target is what the materials keep
		owned.push({ envRT, mats: t.mats });
	}

	pmrem.dispose();
	if (o.oitPass) o.oitPass.value = prevOitPass;
	for (const s of oitSaved) {
		s.m.blending = s.blending;
		s.m.depthWrite = s.depthWrite;
		s.m.depthTest = s.depthTest;
	}
	renderer.setRenderTarget(prevTarget);
	renderer.setClearColor(prevClear, prevAlpha);
	renderer.autoClear = prevAutoClear;
	renderer.shadowMap.autoUpdate = prevShadowAuto;

	return {
		probes: targets.length,
		dispose() {
			for (const e of owned) {
				for (const m of e.mats) {
					m.envMap = null;
					m.needsUpdate = true;
				}
				e.envRT.dispose();
			}
			owned.length = 0;
		},
	};
}
