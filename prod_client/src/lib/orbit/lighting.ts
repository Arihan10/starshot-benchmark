// The lit-rendering rig for the dollhouse / proxy views.
//
// Related to client/public/js/splatlight.js (the capture rig, mirrored from
// splat/stage5.py) but deliberately NOT the same numbers, because it lights very
// different geometry. The capture shades RAW meshes: real topology, real normals,
// real textures, rendered as hero frames. This lights the vertex-colour DOLLHOUSE:
// `simplifySloppy`-decimated to a fraction of its triangles, no normals, no
// textures, viewed from outside as a map. A 3.5-intensity sun with hard shadow
// maps — correct for the capture — turns that into a mess of blown facets and
// black slivers cast by every stray bolt.
//
// So this rig is AMBIENT-DOMINANT: a neutral uniform environment (not three's
// RoomEnvironment, whose emissive studio panels reflect as a fake hotspot) plus a
// hemisphere fill carry the exposure, and a soft directional key only supplies
// enough gradient to read form. Shading stays in linear light and is displayed
// through ACES-filmic + sRGB.
//
// No shadow maps. On decimated, non-manifold geometry a hard sun produces
// self-shadow acne and hard-edged noise from hundreds of small objects, which
// actively hurts legibility on an overview — and skipping the pass keeps the
// renderer light. Form comes from the key/fill gradient instead.

import {
	ACESFilmicToneMapping,
	type Box3,
	Color,
	DirectionalLight,
	HemisphereLight,
	MathUtils,
	PMREMGenerator,
	Scene,
	SRGBColorSpace,
	type Texture,
	Vector3,
	type WebGLRenderer,
} from "three";

// Tuned for the decimated vertex-colour dollhouse. Angles: 0° azimuth = +Z,
// 90° = +X. Raise `key` for more dramatic form, raise `env`/`fill` for a flatter,
// brighter read.
export const LIGHTING = {
	exposure: 1.0,
	key: 1.2, // soft gradient, not a hero sun (the capture rig uses 3.5)
	fill: 0.4,
	env: 0.7, // ambient does the heavy lifting here
	azimuth: 34,
	elevation: 48,
};

export class LightRig {
	private readonly key: DirectionalLight;
	private readonly hemi: HemisphereLight;
	private readonly envTex: Texture;
	private readonly lightDir = new Vector3();
	private readonly center = new Vector3();

	constructor(
		renderer: WebGLRenderer,
		private readonly scene: Scene,
	) {
		const pmrem = new PMREMGenerator(renderer);
		const envScene = new Scene();
		envScene.background = new Color(0xffffff);
		this.envTex = pmrem.fromScene(envScene).texture;
		pmrem.dispose();
		scene.environment = this.envTex;
		scene.environmentIntensity = LIGHTING.env;

		this.hemi = new HemisphereLight(0xffffff, 0x404652, LIGHTING.fill);
		this.key = new DirectionalLight(0xffffff, LIGHTING.key);
		scene.add(this.hemi, this.key, this.key.target);

		// The display transform. Materials shade in linear light; ACES + sRGB happen
		// on the way out. (The pano / projection shaders are deliberately NOT colour
		// managed — their panos already carry this transform baked in — which is why
		// the composer works in an sRGB buffer and blits verbatim.)
		renderer.toneMapping = ACESFilmicToneMapping;
		renderer.toneMappingExposure = LIGHTING.exposure;
		renderer.outputColorSpace = SRGBColorSpace;

		const az = MathUtils.degToRad(LIGHTING.azimuth);
		const el = MathUtils.degToRad(LIGHTING.elevation);
		const cosEl = Math.cos(el);
		this.lightDir
			.set(Math.sin(az) * cosEl, Math.sin(el), Math.cos(az) * cosEl)
			.normalize();
	}

	// Aim the key at the scene from a distance that scales with it, so the same
	// angles read the same way at any scene size.
	fit(box: Box3 | null) {
		if (box && !box.isEmpty()) box.getCenter(this.center);
		else this.center.set(0, 0, 0);
		const radius = box && !box.isEmpty() ? box.getSize(new Vector3()).length() * 0.5 : 10;
		this.key.position
			.copy(this.center)
			.addScaledVector(this.lightDir, Math.max(1, radius) * 3);
		this.key.target.position.copy(this.center);
		this.key.target.updateMatrixWorld();
	}

	dispose() {
		this.scene.remove(this.hemi, this.key, this.key.target);
		if (this.scene.environment === this.envTex) this.scene.environment = null;
		this.envTex.dispose();
		this.hemi.dispose();
		this.key.dispose();
	}
}
