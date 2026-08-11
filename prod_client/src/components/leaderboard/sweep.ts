import * as THREE from "three";

const AT = [0, 0.36, 0.5, 0.72, 1];
const NAMES = [
	"--sweep-1",
	"--sweep-2",
	"--sweep-3",
	"--sweep-4",
	"--sweep-0",
];

const ANGLE = (104 * Math.PI) / 180;
const ACROSS = 0.7071;
const RISE = 0.8165;
export const SWEEP_AXIS = new THREE.Vector3(1, 0, -1)
    .multiplyScalar(Math.sin(ANGLE) / (2 * ACROSS))
    .addScaledVector(new THREE.Vector3(0, 1, 0), Math.cos(ANGLE) / RISE)
    .normalize();

export const SWEEP_SPAN =
    Math.abs(SWEEP_AXIS.x) + Math.abs(SWEEP_AXIS.y) + Math.abs(SWEEP_AXIS.z);

export function sweepStops(): THREE.Color[] {
    const probe = document.createElement("span");
    probe.style.display = "none";
    document.body.appendChild(probe);
    try {
        return NAMES.map((name) => {
            probe.style.color = `var(${name})`;
            return new THREE.Color().setStyle(getComputedStyle(probe).color);
        });
    } finally {
        probe.remove();
    }
}

export const PLINTH_ATTRIBUTE = "aPlinth";

export function paintSweep(material: THREE.Material): {
    stops: { value: THREE.Color[] };
} {
    const stops = { value: AT.map(() => new THREE.Color(1, 1, 1)) };
    const at = { value: AT.slice() };

    material.onBeforeCompile = (shader) => {
        shader.uniforms.uSweepStops = stops;
        shader.uniforms.uSweepAt = at;

        shader.vertexShader = shader.vertexShader
            .replace(
                "#include <common>",
                `#include <common>
				attribute float ${PLINTH_ATTRIBUTE};
				varying float vPlinth;
				varying float vSweep;`,
            )
            .replace(
                "#include <begin_vertex>",
                `#include <begin_vertex>
				vPlinth = ${PLINTH_ATTRIBUTE};
				vSweep = clamp(
					dot(position, vec3(${SWEEP_AXIS.x}, ${SWEEP_AXIS.y}, ${SWEEP_AXIS.z}))
						/ ${SWEEP_SPAN} + 0.5,
					0.0, 1.0
				);`,
            );

        shader.fragmentShader = shader.fragmentShader
            .replace(
                "#include <common>",
                `#include <common>
				uniform vec3 uSweepStops[${AT.length}];
				uniform float uSweepAt[${AT.length}];
				varying float vPlinth;
				varying float vSweep;

				vec3 sweepAt(float t) {
					vec3 c = uSweepStops[0];
					for (int i = 1; i < ${AT.length}; i++) {
						c = mix(c, uSweepStops[i], clamp(
							(t - uSweepAt[i - 1]) / max(uSweepAt[i] - uSweepAt[i - 1], 1e-5),
							0.0, 1.0
						));
					}
					return c;
				}`,
            )
            .replace(
                "#include <color_fragment>",
                `#include <color_fragment>
				diffuseColor.rgb = mix(diffuseColor.rgb, sweepAt(vSweep), vPlinth);`,
            );
    };

    return { stops };
}
