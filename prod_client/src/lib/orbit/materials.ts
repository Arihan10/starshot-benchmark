import {
	BackSide,
	Color,
	DataTexture,
	DoubleSide,
	MeshStandardMaterial,
	RGBAFormat,
	ShaderMaterial,
	Vector2,
	Vector3,
} from "three";

// Backdrop sphere radius, and how many panos blend per fragment (the K nearest
// the camera) in projection mode.
export const SPHERE_RADIUS = 60;
export const PROJ_K = 4;

// 1×1 black stand-in so the sampler array is always fully bound; unused slots
// carry weight 0 and never contribute.
export const DUMMY_TEX = new DataTexture(
	new Uint8Array([0, 0, 0, 255]),
	1,
	1,
	RGBAFormat,
);
DUMMY_TEX.needsUpdate = true;

// Equirect backdrop / sphere-mode pano. Same direction→uv convention as the
// capture stitch: u = atan2(z,x)/2π + 0.5, v = asin(y)/π + 0.5. An opacity
// uniform drives crossfades without dragging in any lighting/tonemap chunks.
export function makePanoMaterial(): ShaderMaterial {
	return new ShaderMaterial({
		uniforms: { map: { value: null }, opacity: { value: 1.0 } },
		vertexShader: /* glsl */ `
			varying vec3 vDir;
			void main() {
				vDir = position;
				gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
			}
		`,
		fragmentShader: /* glsl */ `
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

// View-dependent texture mapping: for each of the K nearest captures, turn the
// fragment's world position into a direction from that capture point, sample its
// equirect there, and blend by proximity + surface-facing. Gluing the texture to
// real geometry is what gives correct parallax as the camera moves.
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
		vertexShader: /* glsl */ `
			varying vec3 vWorldPos;
			varying vec3 vWorldNormal;
			void main() {
				vec4 wp = modelMatrix * vec4(position, 1.0);
				vWorldPos = wp.xyz;
				vWorldNormal = mat3(modelMatrix) * normal;
				gl_Position = projectionMatrix * viewMatrix * wp;
			}
		`,
		fragmentShader: /* glsl */ `
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

// The object silhouette, drawn from the ID mask (see idMasks.ts). For each screen
// pixel: turn the view ray into an equirect texel and ask how much of it belongs
// to the hovered object.
//
// "How much" rather than "whether" is what keeps the edge off the texel grid. A
// binary answer quantizes the boundary however fine the mask is, and a walkthrough
// magnifies enough to show it, so the mask carries the sub-texel coverage its
// supersampled raster measured. Each of the four neighbouring texels contributes
// its share of the object — its own coverage where the texel is ours, the winner's
// leftover where it is a texel we bleed into — and those are bilerped into a
// continuous field.
//
// THICKNESS comes from a ring rather than from that field, because coverage only
// carries signal within about a texel of the boundary: a stroke derived from its
// gradient saturates around a pixel wide and then drifts with zoom. So the ring
// asks the mask directly — step `uWidth` screen pixels out in twelve directions
// and count how many land outside. For a straight edge that count is
// (1/pi)·acos(t/R), smooth in the distance to the boundary, which gives a stroke of
// any requested width, feathered inward, at a fixed weight on screen. The steps are
// taken in DIRECTION space (rotate the ray, then re-project); doing it in uv would
// blow up where u wraps and where the equirect stretches at the poles.
//
// The stroke is drawn from the boundary INWARD. Centred on it, half its width would
// spill onto whatever lies beyond — painting over the neighbour, or over the
// occluder, that the mask says owns those pixels.
export function makeMaskHighlightMaterial(): ShaderMaterial {
	return new ShaderMaterial({
		uniforms: {
			uMask: { value: null },
			uTexel: { value: new Vector2(1, 1) },
			uLocal: { value: 0 },
			uWide: { value: false },
			uEdge: { value: new Color(0xbfe8ff) },
			uWidth: { value: 3.0 }, // stroke width in SCREEN pixels
		},
		vertexShader: /* glsl */ `
			varying vec3 vDir;
			void main() {
				vDir = position;
				gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
			}
		`,
		fragmentShader: /* glsl */ `
			uniform sampler2D uMask;
			uniform vec2 uTexel;
			uniform float uLocal;
			uniform bool uWide;
			uniform vec3 uEdge;
			uniform float uWidth;
			varying vec3 vDir;

			// (local id, coverage) in one tap — the mask is interleaved so four
			// neighbours cost four samples, not eight.
			vec2 tap(vec2 uv) {
				vec4 t = texture2D(uMask, uv);
				return uWide ? vec2(t.r * 255.0 + t.g * 65280.0, t.b) : vec2(t.r * 255.0, t.g);
			}
			float ours(vec2 s) {
				return abs(s.x - uLocal) < 0.5 ? 1.0 : 0.0;
			}
			float share(vec2 s, float near) {
				return mix(near * (1.0 - s.y), s.y, ours(s));
			}
			// The plane is stored top-down and the texture is flipY = false, so
			// t = 1 - v puts row 0 straight up.
			vec2 dirToUv(vec3 d) {
				return vec2(
					atan(d.z, d.x) / 6.28318530718 + 0.5,
					1.0 - (asin(clamp(d.y, -1.0, 1.0)) / 3.14159265359 + 0.5)
				);
			}
			float coverageAt(vec2 uv) {
				vec2 t = uv / uTexel - 0.5;
				vec2 f = fract(t);
				vec2 b = (floor(t) + 0.5) * uTexel;
				vec2 s00 = tap(b);
				vec2 s10 = tap(b + vec2(uTexel.x, 0.0));
				vec2 s01 = tap(b + vec2(0.0, uTexel.y));
				vec2 s11 = tap(b + uTexel);
				// Claiming a neighbour's leftover is only meaningful next to us.
				float near = max(max(ours(s00), ours(s10)), max(ours(s01), ours(s11)));
				return mix(
					mix(share(s00, near), share(s10, near), f.x),
					mix(share(s01, near), share(s11, near), f.x),
					f.y
				);
			}
			void main() {
				if (uLocal < 0.5) discard;
				vec3 d = normalize(vDir);
				// Computed with no branching above it: derivatives are undefined in
				// non-uniform control flow, so discarding outside-fragments first
				// would corrupt fwidth for the very fragments the stroke lands on.
				float c = coverageAt(dirToUv(d));

				// Radians per screen pixel, measured on the direction itself so it
				// stays finite at the seam and at the poles.
				float angPerPx = length(fwidth(d));
				vec3 axis = abs(d.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
				vec3 tx = normalize(cross(axis, d));
				vec3 ty = cross(d, tx);
				float r = uWidth * angPerPx;

				float outside = 0.0;
				for (int i = 0; i < 12; i++) {
					float a = float(i) * 0.5235987756; // 30°
					vec3 s = normalize(d + (tx * cos(a) + ty * sin(a)) * r);
					outside += 1.0 - ours(tap(dirToUv(s)));
				}
				// Half the ring is outside when we sit on the boundary and none of
				// it once we are a full width inside, so doubling gives a stroke
				// solid at the edge and faded to nothing uWidth pixels in.
				float band = clamp((outside / 12.0) * 2.0, 0.0, 1.0);
				// The sub-pixel-accurate outer edge; band is the body.
				float inside = clamp((c - 0.5) / max(fwidth(c), 1e-5) + 0.5, 0.0, 1.0);

				float a = inside * band;
				if (a <= 0.004) discard;
				gl_FragColor = vec4(uEdge, a);
			}
		`,
		side: BackSide,
		transparent: true,
		depthWrite: false,
		depthTest: false,
	});
}

// Flat-shaded matte for the bare dollhouse proxy (when no separate lite export
// exists, the proxy itself stands in for the overview).
export function makePolyMaterial(): MeshStandardMaterial {
	return new MeshStandardMaterial({
		color: 0x9aa7b4,
		roughness: 0.9,
		metalness: 0.0,
		flatShading: true,
		side: DoubleSide,
	});
}
