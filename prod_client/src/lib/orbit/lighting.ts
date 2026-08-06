
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

export const LIGHTING = {
	exposure: 1.0,
	key: 1.2,
	fill: 0.4,
	env: 0.7,
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
