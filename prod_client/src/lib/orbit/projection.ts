import {
	Box3,
	type Group,
	type Material,
	Mesh,
	type PerspectiveCamera,
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

// View-dependent texture mapping. Owns the shared projection shader (one
// material across every proxy mesh + the floor base) and, each frame, blends the
// K captures nearest the camera onto the proxy geometry — gluing the panos to
// real surfaces so they parallax correctly as you move. The backdrop sphere is
// driven here too (its nearest pano is the projected sky).
export class Projection {
	readonly material: ShaderMaterial = makeProjectionMaterial();
	private base: Mesh | null = null;
	private backdropRadius = SPHERE_RADIUS;
	private readonly camDist2: number[] = [];

	get proxyBase(): Mesh | null {
		return this.base;
	}

	// Re-skin proxy meshes with the projection shader, recompute normals (the
	// decimation dropped usable ones), and size the backdrop to the scene extent.
	setup(root: Group, sphereA: Mesh, sphereAMat: ShaderMaterial) {
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
		sphereA.scale.setScalar(this.backdropRadius / SPHERE_RADIUS);
		sphereA.renderOrder = -1;
		sphereAMat.uniforms.opacity.value = 1;
		sphereAMat.depthTest = true; // let the opaque proxy occlude the backdrop
	}

	update(
		camera: PerspectiveCamera,
		panos: PanoEntry[],
		request: (i: number) => void,
		sphereA: Mesh,
		sphereAMat: ShaderMaterial,
	) {
		if (panos.length === 0) return;
		const u = this.material.uniforms;
		const cam = camera.position;
		this.camDist2.length = panos.length;
		for (let i = 0; i < panos.length; i++) {
			const p = panos[i].position;
			const dx = cam.x - p[0];
			const dy = cam.y - p[1];
			const dz = cam.z - p[2];
			this.camDist2[i] = dx * dx + dy * dy + dz * dz;
		}
		const order = panos
			.map((_, i) => i)
			.sort((a, b) => this.camDist2[a] - this.camDist2[b]);
		const K = Math.min(PROJ_K, panos.length);
		// Load on movement: kick off the K nearest captures, but project only the
		// ones already loaded (blurred placeholder counts) so we never block.
		const ready: number[] = [];
		for (let k = 0; k < K; k++) {
			request(order[k]);
			if (panos[order[k]].texture) ready.push(order[k]);
		}
		let wsum = 0;
		const w: number[] = [];
		for (let k = 0; k < ready.length; k++) {
			const ww = 1 / (this.camDist2[ready[k]] + 0.25);
			w.push(ww);
			wsum += ww;
		}
		for (let k = 0; k < PROJ_K; k++) {
			if (k < ready.length) {
				const idx = ready[k];
				u.uTex.value[k] = panos[idx].texture;
				(u.uCenter.value[k] as Vector3).fromArray(panos[idx].position);
				u.uWeight.value[k] = w[k] / wsum;
			} else {
				u.uTex.value[k] = DUMMY_TEX;
				u.uWeight.value[k] = 0;
			}
		}
		u.uCount.value = ready.length;
		sphereAMat.uniforms.map.value = ready.length
			? panos[ready[0]].texture
			: DUMMY_TEX;
		sphereA.position.copy(cam);
	}

	// A floor slab spanning the proxy's footprint, sat just under its lowest point,
	// backing "proxy leaks" (gaps in the proxy floor). It shares the proxy's
	// material, so the panos project onto it just like the floor: at a capture
	// point the projection equals the backdrop image, so the slab is invisible,
	// and it picks up the live projected floor as you move — rather than showing
	// through holes as a flat fill. Kept out of proxyGroup so it isn't
	// registered/picked as an addressable object; visibility is synced to the proxy.
	buildBase(proxyGroup: Group, scene: Scene) {
		const box = new Box3().setFromObject(proxyGroup, true);
		if (box.isEmpty()) return;
		const size = box.getSize(new Vector3());
		const center = box.getCenter(new Vector3());
		const base = new Mesh(new PlaneGeometry(size.x, size.z), this.material);
		base.rotation.x = -Math.PI / 2; // lie flat, normal up
		// A hair below the lowest vertex so it never z-fights a coincident floor.
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

	// The base belongs to the proxy, so it shows exactly when the proxy does.
	syncBase(proxyVisible: boolean) {
		if (this.base) this.base.visible = proxyVisible;
	}

	// Matte (proxy view) vs the projection shader, mirroring the proxy's skin.
	setBaseMaterial(mat: Material) {
		if (this.base) this.base.material = mat;
	}

	clearBase(scene: Scene) {
		if (!this.base) return;
		scene.remove(this.base);
		this.base.geometry.dispose(); // shares the proj/poly singletons — don't dispose them
		this.base = null;
	}
}
