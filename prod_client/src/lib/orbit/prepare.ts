
import {
	DoubleSide,
	Mesh,
	MeshBasicMaterial,
	type Material,
	type Object3D,
	type Texture,
} from "three";

const OIT_OPAQUE = 0.8;

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

function toLitMaterial(m: StdMaterial) {
	m.side = DoubleSide;
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

function isEffectivelyOpaque(m: StdMaterial): boolean {
	if (!m.transparent) return true;
	if (m.opacity < 1) return false;
	if (m.alphaMap) return false;
	if (m.map) return false;
	return true;
}

function depthProxyMaterial(orig: StdMaterial): MeshBasicMaterial {
	const m = new MeshBasicMaterial();
	m.map = orig.map ?? null;
	m.opacity = orig.opacity;
	m.alphaTest = OIT_OPAQUE;
	m.side = DoubleSide;
	m.colorWrite = false;
	m.depthWrite = true;
	m.depthTest = true;
	m.transparent = false;
	return m;
}

function nullProxyMaterial(): MeshBasicMaterial {
	return new MeshBasicMaterial({
		colorWrite: false,
		depthWrite: false,
		depthTest: false,
	});
}

export function prepareLitScene(root: Object3D): void {
	const attach: Array<[Mesh, Mesh]> = [];
	root.traverse((o) => {
		const mesh = o as Mesh;
		if (!mesh.isMesh || !mesh.material) return;
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
				m.depthWrite = false;
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
		proxy.raycast = () => {};
		attach.push([mesh, proxy]);
	});
	for (const [mesh, proxy] of attach) mesh.add(proxy);
}
