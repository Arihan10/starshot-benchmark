import {
	BackSide,
	DataTexture,
	DoubleSide,
	MeshStandardMaterial,
	RGBAFormat,
	ShaderMaterial,
	Vector3,
} from "three";

export const SPHERE_RADIUS = 60;
export const PROJ_K = 4;

export const DUMMY_TEX = new DataTexture(
	new Uint8Array([0, 0, 0, 255]),
	1,
	1,
	RGBAFormat,
);
DUMMY_TEX.needsUpdate = true;

export function makePanoMaterial(): ShaderMaterial {
	return new ShaderMaterial({
		uniforms: { map: { value: null }, opacity: { value: 1.0 } },
		vertexShader: `
			varying vec3 vDir;
			void main() {
				vDir = position;
				gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
			}
		`,
		fragmentShader: `
			uniform sampler2D map;
			uniform float opacity;
			varying vec3 vDir;
			void main() {
				vec3 d = normalize(vDir);
				vec2 uv = vec2(
					atan(d.z, d.x) / 6.28318530718 + 0.5,
					asin(clamp(d.y, -1.0, 1.0)) / 3.14159265359 + 0.5
				);
				gl_FragColor = vec4(texture2D(map, uv).rgb, opacity);
			}
		`,
		side: BackSide,
		transparent: true,
		depthWrite: false,
		depthTest: false,
	});
}

export function makeProjectionMaterial(): ShaderMaterial {
	return new ShaderMaterial({
		uniforms: {
			uTex: { value: Array.from({ length: PROJ_K }, () => DUMMY_TEX) },
			uCenter: {
				value: Array.from({ length: PROJ_K }, () => new Vector3()),
			},
			uWeight: { value: new Float32Array(PROJ_K) },
			uCount: { value: 0 },
		},
		side: DoubleSide,
		vertexShader: `
			varying vec3 vWorldPos;
			varying vec3 vWorldNormal;
			void main() {
				vec4 wp = modelMatrix * vec4(position, 1.0);
				vWorldPos = wp.xyz;
				vWorldNormal = mat3(modelMatrix) * normal;
				gl_Position = projectionMatrix * viewMatrix * wp;
			}
		`,
		fragmentShader: `
			uniform sampler2D uTex[${PROJ_K}];
			uniform vec3 uCenter[${PROJ_K}];
			uniform float uWeight[${PROJ_K}];
			uniform int uCount;
			varying vec3 vWorldPos;
			varying vec3 vWorldNormal;
			const float TAU = 6.28318530718;
			const float PI = 3.14159265359;

			vec2 dirToEquirect(vec3 d) {
				return vec2(atan(d.z, d.x) / TAU + 0.5, asin(clamp(d.y, -1.0, 1.0)) / PI + 0.5);
			}
			vec3 accumOne(int i, sampler2D tex, vec3 n, inout float wsum) {
				if (i >= uCount) return vec3(0.0);
				vec3 dir = normalize(vWorldPos - uCenter[i]);
				float face = clamp(dot(n, -dir), 0.0, 1.0);
				float w = uWeight[i] * (0.2 + 0.8 * face);
				wsum += w;
				return w * texture2D(tex, dirToEquirect(dir)).rgb;
			}
			void main() {
				vec3 n = normalize(vWorldNormal);
				float wsum = 0.0;
				vec3 col = vec3(0.0);
				col += accumOne(0, uTex[0], n, wsum);
				col += accumOne(1, uTex[1], n, wsum);
				col += accumOne(2, uTex[2], n, wsum);
				col += accumOne(3, uTex[3], n, wsum);
				gl_FragColor = vec4(col / max(wsum, 1e-4), 1.0);
			}
		`,
	});
}

export function makePolyMaterial(): MeshStandardMaterial {
	return new MeshStandardMaterial({
		color: 0x9aa7b4,
		roughness: 0.9,
		metalness: 0.0,
		flatShading: true,
		side: DoubleSide,
	});
}
