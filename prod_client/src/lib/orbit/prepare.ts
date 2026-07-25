// Make a loaded GLB shadeable, and get its transparency queues right.
//
// The prod-client port of splatlight.js `prepareLitScene`. Two concrete bugs in
// our own assets make this mandatory rather than cosmetic:
//
//  1. NO NORMALS. Generated meshes (and the vertex-colour dollhouse bake) ship
//     with no NORMAL attribute at all, and MeshStandardMaterial with no normals
//     shades to black — the "dark and unlit" symptom. The fix is NOT to synthesize
//     smooth ones: the dollhouse is `simplifySloppy`-decimated, so averaging face
//     normals across vertices the simplifier welded together from unrelated facets
//     produces normals that match no real surface — which lights up as violently
//     faceted, blown-out chaos. Instead those materials switch to FLAT shading,
//     taking the true geometric normal per triangle from screen-space derivatives.
//     Authored normals, when a mesh has them, are left alone.
//
//  2. OPAQUE GEOMETRY MARKED `BLEND`. The dollhouse bake writes fully opaque
//     vertex colours (alpha is hard-set to 255) but leaves glTF alphaMode at
//     BLEND. GLTFLoader faithfully turns that into `transparent = true,
//     depthWrite = false`, so EVERY mesh lands in the back-to-front transparent
//     queue with no depth buffer — objects sort per-object by centroid and walls
//     render through each other. Re-classifying those materials puts them back in
//     the opaque pass, where per-fragment depth does the occluding.
//
// Genuinely transparent surfaces keep blending, and additionally get a DEPTH
// PROXY: a colour-less alpha-tested twin that renders in the opaque pass and
// writes depth only where the surface is effectively solid. That's the cheap
// equivalent of scene3d.js's OIT depth pre-pass — it fixes the case that actually
// looks broken (a mirror or window frame you can see the room through) without the
// three extra render targets and composite pass of full weighted-blended OIT.

import {
	DoubleSide,
	Mesh,
	MeshBasicMaterial,
	type Material,
	type Object3D,
	type Texture,
} from "three";

// Alpha at/above which a transparent fragment counts as SOLID. Mirrors
// scene3d.js's weighted-blended OIT `OIT_OPAQUE` and splatlight.js, so every
// renderer in the project draws the same cutoff.
const OIT_OPAQUE = 0.8;

// The PBR knobs we normalize. They live on MeshStandardMaterial /
// MeshPhysicalMaterial rather than the base Material type.
type StdMaterial = Material & {
	flatShading?: boolean;
	metalness?: number;
	roughness?: number;
	metalnessMap?: Texture | null;
	roughnessMap?: Texture | null;
	map?: Texture | null;
	alphaMap?: Texture | null;
	clearcoat?: number;
	sheen?: number;
	specularIntensity?: number;
	iridescence?: number;
};

// Force a matte dielectric, matching splatlight.js `toLitMaterial`. The panos were
// baked with view-INDEPENDENT lighting only (a flat splat can't carry a moving
// specular lobe), so the dollhouse has to drop the same view-dependent gloss or it
// won't match what you see when you step inside. Base colour, alpha, normal, AO
// and emissive are left alone.
function toLitMaterial(m: StdMaterial) {
	m.side = DoubleSide; // generated winding is unreliable
	if ("metalness" in m) m.metalness = 0;
	if ("roughness" in m) m.roughness = 1;
	if ("metalnessMap" in m) m.metalnessMap = null;
	if ("roughnessMap" in m) m.roughnessMap = null;
	if ("clearcoat" in m) m.clearcoat = 0;
	if ("sheen" in m) m.sheen = 0;
	if ("specularIntensity" in m) m.specularIntensity = 0;
	if ("iridescence" in m) m.iridescence = 0;
	m.needsUpdate = true;
}

// Can this material actually be seen through? A BLEND flag alone doesn't mean it
// can. A base-colour map may carry per-texel alpha we can't cheaply inspect, so a
// textured BLEND surface is taken at its word.
function isEffectivelyOpaque(m: StdMaterial): boolean {
	if (!m.transparent) return true;
	if (m.opacity < 1) return false;
	if (m.alphaMap) return false;
	if (m.map) return false;
	return true;
}

// Colour-less, alpha-tested depth twin of a transparent material: renders in the
// OPAQUE queue and writes depth ONLY where the source's alpha reaches OIT_OPAQUE,
// so solid regions occlude per-fragment while sub-cutoff glass writes no depth and
// stays see-through. Shares the source `map` so the same per-texel alpha decides.
function depthProxyMaterial(orig: StdMaterial): MeshBasicMaterial {
	const m = new MeshBasicMaterial();
	m.map = orig.map ?? null;
	m.opacity = orig.opacity;
	m.alphaTest = OIT_OPAQUE;
	m.side = DoubleSide;
	m.colorWrite = false; // depth only — never contributes colour
	m.depthWrite = true;
	m.depthTest = true;
	m.transparent = false; // opaque queue → runs before the blend pass
	return m;
}

// Keeps a multi-material proxy array aligned with the geometry's groups: an opaque
// sub-material already writes depth in its own pass, so its slot is a no-op.
function nullProxyMaterial(): MeshBasicMaterial {
	return new MeshBasicMaterial({
		colorWrite: false,
		depthWrite: false,
		depthTest: false,
	});
}

export function prepareLitScene(root: Object3D): void {
	const attach: Array<[Mesh, Mesh]> = []; // added after the walk, not mid-traverse
	root.traverse((o) => {
		const mesh = o as Mesh;
		if (!mesh.isMesh || !mesh.material) return;
		// No authored normals → shade flat (true per-triangle geometric normals)
		// rather than inventing smooth ones this geometry can't support.
		const flat = !!mesh.geometry && !mesh.geometry.getAttribute("normal");

		const mats = (
			Array.isArray(mesh.material) ? mesh.material : [mesh.material]
		) as StdMaterial[];
		for (const m of mats) {
			toLitMaterial(m);
			if (flat && "flatShading" in m) m.flatShading = true;
			if (isEffectivelyOpaque(m)) {
				m.transparent = false;
				m.depthWrite = true;
			} else {
				m.depthWrite = false; // let the depth behind glass survive
			}
		}

		if (!mats.some((m) => m.transparent)) return;
		const proxyMat = Array.isArray(mesh.material)
			? mats.map((m) => (m.transparent ? depthProxyMaterial(m) : nullProxyMaterial()))
			: depthProxyMaterial(mats[0]);
		const proxy = new Mesh(mesh.geometry, proxyMat);
		proxy.castShadow = false;
		proxy.receiveShadow = false;
		proxy.frustumCulled = mesh.frustumCulled;
		proxy.userData.depthProxy = true;
		proxy.raycast = () => {}; // never hovered, picked, or addressed
		attach.push([mesh, proxy]);
	});
	for (const [mesh, proxy] of attach) mesh.add(proxy);
}
