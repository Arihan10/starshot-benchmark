import { LinearFilter, RepeatWrapping, type Group, type Texture, TextureLoader } from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";

const gltfLoader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);

export async function loadGLB(url: string): Promise<Group> {
	const gltf = await gltfLoader.loadAsync(url);
	return gltf.scene;
}

const textureLoader = new TextureLoader();

function prepPanoTexture(tex: Texture): Texture {
	tex.generateMipmaps = false;
	tex.minFilter = LinearFilter;
	tex.magFilter = LinearFilter;
	tex.wrapS = RepeatWrapping;
	return tex;
}

export async function loadPanoTexture(url: string): Promise<Texture> {
	return prepPanoTexture(await textureLoader.loadAsync(url));
}
