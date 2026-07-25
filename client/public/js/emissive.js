// Emissive meshes — the ONE place that decides which objects EMIT light (lamps,
// sconces, screens, fire, …) and makes them both GLOW and ILLUMINATE the scene.
// Shared by the Stage-5 capture (splatcapture.js) and the debug mesh preview
// (splatviewer.js) so the reference frames and the preview agree.
//
// Designation is by the object's id — the LLM's descriptive name (e.g.
// "bedside_lamp", "atrium_strip_light_array_north") — matched against the curated
// token lists below. Keep the lists here, separate from the render code, so the
// vocabulary is easy to audit and extend. Matching is TOKEN-based (the id is split
// on non-alphanumerics), which avoids substring false positives like "ledge"→"led"
// or "bathroom_light_switch" (the EXCLUDE token `switch` vetoes it).
//
// For each emissive object we (1) set an HDR `emissive` glow (captured into the
// refs + reflected by the cube probes) and stop it from casting a shadow (a light
// shouldn't shadow its own light), and (2) add a shadow-casting PointLight at its
// centre so it actually brightens nearby surfaces — baked once (autoUpdate off)
// alongside the sun by oit.js `bakeShadowMap`, so occlusion is respected (no light
// leaking through walls).

import * as THREE from "three";

// Single-token names that mark a light-emitting object. Edit freely — this is the
// curated vocabulary. (Compound words with no separator, e.g. "spotlight", must be
// listed whole; "spot_light" is caught by the "light" token.)
export const EMISSIVE_INCLUDE = new Set([
	"light", "lights", "lighting", "lamp", "lamps", "lamplight", "lamppost",
	"lantern", "lanterns", "sconce", "sconces", "chandelier", "pendant",
	"candle", "candles", "candlestick", "candelabra", "bulb", "bulbs", "neon",
	"luminaire", "torch", "torchiere", "spotlight", "downlight", "uplight",
	"backlight", "floodlight", "headlight", "streetlight", "streetlamp",
	"nightlight", "skylight", "lightbox", "led", "leds", "fluorescent",
	"halogen", "incandescent", "screen", "screens", "monitor", "monitors",
	"display", "tv", "television", "projector", "fireplace", "hearth",
	"brazier", "firepit", "campfire", "bonfire", "furnace", "flame", "flames",
	"ember", "embers", "coals", "lava", "magma", "glow", "glowing",
]);

// Two-word phrases (adjacent tokens) that also mark a light — for names that split
// into separate tokens (e.g. "fire_pit" → ["fire","pit"]).
export const EMISSIVE_INCLUDE_BIGRAMS = new Set([
	"fire pit", "fire bowl", "light strip", "strip light", "light bar",
	"light panel", "light array", "neon sign", "led strip",
]);

// Tokens that VETO a match even when an include token is present (false positives).
export const EMISSIVE_EXCLUDE = new Set([
	"switch", "switches", "socket", "sockets", "cord", "cable", "wire",
	"conduit", "ledge", "ledger", "extinguisher", "hydrant", "alarm",
	"unlit", "shadow", "reflection",
]);

// Illumination + glow parameters. Tunable defaults; the callers may override.
export const EMISSIVE_DEFAULTS = {
	glowIntensity: 3.0, // emissiveIntensity (HDR) — how bright the mesh itself glows
	lightIntensity: 8.0, // PointLight intensity (candela, physical decay) — the KEY dial
	decay: 2.0, // physical inverse-square falloff
	distance: 0, // 0 = no cutoff (pure inverse-square)
	castShadow: true, // emissive lights may cast shadows — but only the first
	// `maxShadowLights` do (see below): each shadow-casting light binds a shadow-map
	// SAMPLER, and WebGL only guarantees 16 fragment texture units (ANGLE/D3D11 caps
	// exactly there). With the sun's shadow + the env map + a PBR material's own maps
	// already using ~8, too many shadow lights overflow the limit and the material
	// shaders fail to link → those meshes render BLACK. So cap the shadow-casters;
	// the remaining lights still illuminate WITHOUT shadows (no sampler cost).
	shadowMapSize: 512, // per emissive light (cube); modest to bound VRAM
	shadowBias: -0.001,
	maxLights: 10, // total emissive point lights (illumination coverage)
	maxShadowLights: 4, // of those, how many cast shadows (texture-unit budget)
	warmWhite: 0xffe8c8, // fallback light colour when the fixture colour is ~white
};

function idTokens(id) {
	return String(id || "").toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
}

// True when the object id names a light-emitting object (see the lists above).
export function isEmissiveName(id) {
	const toks = idTokens(id);
	for (const t of toks) if (EMISSIVE_EXCLUDE.has(t)) return false;
	for (const t of toks) if (EMISSIVE_INCLUDE.has(t)) return true;
	for (let i = 0; i + 1 < toks.length; i++) {
		if (EMISSIVE_INCLUDE_BIGRAMS.has(toks[i] + " " + toks[i + 1])) return true;
	}
	return false;
}

// Make one object's meshes GLOW: emissive = base colour (× base-colour map) at an
// HDR intensity, and stop them casting shadows (a light must not shadow its own
// light / trap it inside a shell). Returns a representative colour for the cast
// light (the fixture's tint, or warm white when it's ~white/unset).
function makeGlow(object, o) {
	let r = 0;
	let g = 0;
	let b = 0;
	let n = 0;
	object.traverse((m) => {
		if (!m.isMesh || !m.material) return;
		m.castShadow = false; // emits, doesn't shadow
		const mats = Array.isArray(m.material) ? m.material : [m.material];
		for (const mat of mats) {
			if (!mat) continue;
			// Only materials whose shader HAS an `emissive` uniform may carry one.
			// Every lit material (Standard/Physical/Phong/Lambert/Toon) initialises
			// `emissive` in its constructor; MeshBasicMaterial has none — and
			// three.js's refreshUniformsCommon dereferences `uniforms.emissive.value`
			// for ANY material with a truthy `emissive`, so stamping one onto an unlit
			// material throws on the first render. This skips the alpha-tested DEPTH
			// PROXIES prepareLitScene attaches to transparent meshes (a glass lamp
			// shade / lit panel gets one): they're colorWrite:false, so glowing them
			// would be a no-op anyway, and their default-white `color` would wash out
			// the cast light's tint below.
			if (mat.emissive === undefined) continue;
			mat.emissive = new THREE.Color(0xffffff);
			if (mat.map) mat.emissiveMap = mat.map; // glow follows the fixture's texture
			mat.emissiveIntensity = o.glowIntensity;
			mat.needsUpdate = true;
			if (mat.color) {
				r += mat.color.r;
				g += mat.color.g;
				b += mat.color.b;
				n += 1;
			}
		}
	});
	const col = new THREE.Color(o.warmWhite);
	if (n > 0) {
		const avg = new THREE.Color(r / n, g / n, b / n);
		// Use the fixture's own tint when it's saturated; else a warm white (a
		// metalness=1 fixture usually has a white baseColorFactor with the colour in
		// the texture, so "white" → fall back to warm white).
		const max = Math.max(avg.r, avg.g, avg.b);
		const min = Math.min(avg.r, avg.g, avg.b);
		if (max > 0.05 && max - min > 0.06) col.copy(avg);
	}
	return col;
}

// Designate + light the emissive objects among `objectsRoot`'s children (the per-
// object groups). GLOWS every match (free), then adds a PointLight to `scene` for
// the LARGEST fixtures up to `maxLights`, of which only the first `maxShadowLights`
// cast shadows (WebGL 16-texture-unit budget — see EMISSIVE_DEFAULTS). Returns
// `{ count, dispose }`; the shadow-casting lights are baked later by oit.js
// `bakeShadowMap`, so call this BEFORE that + before the reflection bake. Safe
// no-op when nothing matches.
export function applyEmissiveLighting(objectsRoot, scene, opts = {}) {
	const o = { ...EMISSIVE_DEFAULTS, ...opts };
	objectsRoot.updateMatrixWorld(true);
	// Glow every emissive object (free), collecting world centre + size so the
	// biggest fixtures get the (capped) point lights.
	const box = new THREE.Box3();
	const cands = [];
	for (const child of objectsRoot.children) {
		if (child.isLight || child.isCamera) continue;
		const id = child.userData?.objectId || child.name || "";
		if (!isEmissiveName(id)) continue;
		const color = makeGlow(child, o);
		box.setFromObject(child);
		if (box.isEmpty()) continue;
		cands.push({
			color,
			center: box.getCenter(new THREE.Vector3()),
			diag: box.getSize(new THREE.Vector3()).length(),
		});
	}
	const lights = [];
	if (o.lightIntensity > 0) {
		cands.sort((a, b) => b.diag - a.diag); // biggest (most impactful) first
		for (let i = 0; i < cands.length && lights.length < o.maxLights; i++) {
			const c = cands[i];
			const light = new THREE.PointLight(c.color, o.lightIntensity, o.distance, o.decay);
			light.position.copy(c.center);
			light.layers.enableAll(); // reach the OIT-layer geometry (all BLEND meshes)
			// Only the first maxShadowLights cast — each shadow map is a texture unit,
			// and too many overflow the GPU's 16-unit limit → black meshes. The rest
			// still illuminate, just without occlusion.
			if (o.castShadow && lights.length < o.maxShadowLights) {
				light.castShadow = true;
				light.shadow.mapSize.set(o.shadowMapSize, o.shadowMapSize);
				light.shadow.bias = o.shadowBias;
				light.shadow.camera.near = Math.max(0.02, c.diag * 0.05);
				light.shadow.camera.far = Math.max(c.diag * 6, 10);
			}
			scene.add(light);
			lights.push(light);
		}
	}
	return {
		count: lights.length,
		dispose() {
			for (const l of lights) {
				scene.remove(l);
				l.shadow?.map?.dispose?.();
				l.dispose?.();
			}
			lights.length = 0;
		},
	};
}
