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

// View-dependent texture mapping. Owns the shared projection shader (one material
// across every proxy mesh, the floor base, AND the backdrop sphere) and projects
// an explicit, focused set of captures onto it each frame — gluing the panos to
// real surfaces so they parallax as you move. The walkthrough only ever sits AT a
// capture (project that one) or glides BETWEEN two (project the from/to pair,
// time-weighted), so the set is never more than the couple you're actually
// travelling through — see `project`.
export class Projection {
	readonly material: ShaderMaterial = makeProjectionMaterial();
	private base: Mesh | null = null;
	private backdropRadius = SPHERE_RADIUS;

	get proxyBase(): Mesh | null {
		return this.base;
	}

	// Re-skin proxy meshes with the projection shader, recompute normals (the
	// decimation dropped usable ones), and size the backdrop to the scene extent.
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
		// The backdrop wears the SAME VDTM material as the proxy, so the far field is
		// the identical continuous K-capture blend as the near field. Driving it from a
		// single "nearest capture" equirect instead made it hard-cut every time the
		// nearest changed — and mid-move the nearest is often a THIRD anchor you pass
		// close to, so its whole vantage flashed across the sky for a few frames (and
		// flickered outright wherever two captures were near-equidistant). Blending
		// removes the switch entirely and matches the proxy across its silhouette.
		// The shader is world-space (it measures direction from each capture's centre),
		// so the sphere is free to ride with the camera — no placement invariant.
		sphereA.material = this.material;
		sphereA.scale.setScalar(this.backdropRadius / SPHERE_RADIUS);
		sphereA.renderOrder = -1;
	}

	// Project an explicit set of captures (index → weight) onto the proxy + the
	// backdrop. `caps` is the FOCUSED set the engine hands us: one capture at rest,
	// or the from/to pair while gliding a hop (time-weighted). We deliberately do
	// NOT blend the K nearest by camera distance — that re-showed every anchor you
	// merely pass near as a faint, offset room on the far backdrop (their parallax
	// can't cancel at sphere range → "duplicated many times") and lurched whenever
	// the nearest set churned (the mid-hop flash). A capture only contributes once
	// its texture is resident; a blurred placeholder counts, so we never block.
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
		// The backdrop rides the camera so it always encloses it; with a single
		// capture (at rest) that makes it an exact skybox — zero parallax error.
		sphereA.position.copy(camPos);
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
