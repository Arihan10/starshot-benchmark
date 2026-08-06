import {
	Box3,
	type Group,
	type Material,
	Mesh,
	PlaneGeometry,
	type Scene,
	type ShaderMaterial,
	Sphere,
	Vector3,
} from "three";
import {
	DUMMY_TEX,
	makeProjectionMaterial,
	PROJ_K,
	SPHERE_RADIUS,
} from "./materials";
import type { PanoEntry } from "./panoTextures";

export class Projection {
	readonly material: ShaderMaterial = makeProjectionMaterial();
	private base: Mesh | null = null;
	private backdropRadius = SPHERE_RADIUS;

	get proxyBase(): Mesh | null {
		return this.base;
	}

	setup(root: Group, sphereA: Mesh) {
		root.traverse((o) => {
			const m = o as Mesh;
			if (!m.isMesh || !m.geometry) return;
			m.geometry.computeVertexNormals();
			m.material = this.material;
			m.frustumCulled = false;
		});
		const box = new Box3().setFromObject(root);
		const sph = box.getBoundingSphere(new Sphere());
		this.backdropRadius = Math.max(80, sph.radius * 4);
		sphereA.material = this.material;
		sphereA.scale.setScalar(this.backdropRadius / SPHERE_RADIUS);
		sphereA.renderOrder = -1;
	}

	project(
		panos: PanoEntry[],
		caps: ReadonlyArray<readonly [number, number]>,
		request: (i: number) => void,
		sphereA: Mesh,
		camPos: Vector3,
	) {
		const u = this.material.uniforms;
		const ready: Array<[number, number]> = [];
		for (const [idx, weight] of caps) {
			const p = panos[idx];
			if (!p) continue;
			request(idx);
			if (p.texture) ready.push([idx, weight]);
		}
		let wsum = 0;
		for (const [, weight] of ready) wsum += weight;
		if (wsum <= 0) wsum = 1;
		for (let k = 0; k < PROJ_K; k++) {
			if (k < ready.length) {
				const [idx, weight] = ready[k];
				u.uTex.value[k] = panos[idx].texture;
				(u.uCenter.value[k] as Vector3).fromArray(panos[idx].position);
				u.uWeight.value[k] = weight / wsum;
			} else {
				u.uTex.value[k] = DUMMY_TEX;
				u.uWeight.value[k] = 0;
			}
		}
		u.uCount.value = ready.length;
		sphereA.position.copy(camPos);
	}

	buildBase(proxyGroup: Group, scene: Scene) {
		const box = new Box3().setFromObject(proxyGroup, true);
		if (box.isEmpty()) return;
		const size = box.getSize(new Vector3());
		const center = box.getCenter(new Vector3());
		const base = new Mesh(new PlaneGeometry(size.x, size.z), this.material);
		base.rotation.x = -Math.PI / 2;
		base.position.set(
			center.x,
			box.min.y - Math.max(0.01, size.y * 0.002),
			center.z,
		);
		base.frustumCulled = false;
		base.visible = false;
		scene.add(base);
		this.base = base;
	}

	syncBase(proxyVisible: boolean) {
		if (this.base) this.base.visible = proxyVisible;
	}

	setBaseMaterial(mat: Material) {
		if (this.base) this.base.material = mat;
	}

	clearBase(scene: Scene) {
		if (!this.base) return;
		scene.remove(this.base);
		this.base.geometry.dispose();
		this.base = null;
	}
}
